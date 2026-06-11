"""
src/triple_compare.py

Session 20: join model, sportsbook, and Polymarket probabilities per match.

Pipeline:
  1. Run Dixon-Coles on every WC group-stage fixture.
  2. Subtract Session 11/12 calibration bias from each model prob and
     renormalise -> model_corr.
  3. Left-join sportsbook_odds.csv on (home_team, away_team).
  4. Left-join polymarket_odds.csv on (home_team, away_team).
     Today this contributes nothing (header-only) but the per-match h2h
     markets will appear closer to kickoff (see PROJECT.md §6); the same
     script picks them up with no code change.
  5. Compute divergence metrics against model_corr, plus a categorical
     divergence_type per match (disagree_on_favorite /
     model_over_concentrated / model_under_concentrated). Session 22's
     Claude commentary uses divergence_type to pick the right prompt shape.
  6. Flag matches whose max single-outcome |gap| >= DIV_FLAG_THRESHOLD.
     Session 33: host advantage is now modelled, so no matches are excluded.
  7. Write data/processed/triple_compare.csv and print the top-10 gaps.

Run from project root:
    python src/triple_compare.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from clock import today as _clock_today, divergence_snapshot_dir as _div_snap_dir
from dixon_coles import predict_match

# --- configuration ----------------------------------------------------------

RATINGS_PATH   = Path("data/processed/elo_ratings_2026.csv")
FIXTURES_PATH  = Path("data/processed/fixtures_2026.csv")
BACKTEST_PATH  = Path("data/processed/backtest_2024.csv")
BOOK_PATH      = Path("data/processed/sportsbook_odds.csv")
POLY_PATH      = Path("data/processed/polymarket_odds.csv")
OUT_PATH       = Path("data/processed/triple_compare.csv")

# Session 33: use each fixture's actual neutral flag so per-match bars
# match the simulation (which now applies a 60-Elo home advantage for
# USA/Mexico/Canada's 9 home group matches).
USE_FIXTURE_NEUTRAL = True

# Flag a match if max single-outcome |model_corr - book| >= this.
# Raised from 0.08 after the Session 20 first run flagged 28/50 matches.
# At 0.15 we surface ~10-15 genuinely interesting gaps; the bias correction
# handles the small structural offsets (+/-3-4pp draw/away) on its own.
DIV_FLAG_THRESHOLD = 0.15


# --- bias correction --------------------------------------------------------

def compute_bias_offsets(backtest_path: Path) -> tuple[dict[str, float], int]:
    """
    Mean calibration bias from the Session 11/12 backtest.

        bias_outcome = mean(p_outcome_predicted) - frequency(actual_is_outcome)

    Positive bias  => model over-predicts that outcome
    Negative bias  => model under-predicts that outcome
    The three biases sum to 0 by construction.

    Reading from backtest_2024.csv rather than hardcoding Session 12's
    numbers means the offsets follow automatically if we ever refit.
    """
    df = pd.read_csv(backtest_path)
    bias = {
        "home": float(df["p_home"].mean() - (df["actual"] == 0).mean()),
        "draw": float(df["p_draw"].mean() - (df["actual"] == 1).mean()),
        "away": float(df["p_away"].mean() - (df["actual"] == 2).mean()),
    }
    return bias, len(df)


def bias_correct(
    p_home: float, p_draw: float, p_away: float, bias: dict[str, float],
) -> tuple[float, float, float]:
    """Subtract bias, clip negatives to 0, renormalise."""
    a = max(p_home - bias["home"], 0.0)
    b = max(p_draw - bias["draw"], 0.0)
    c = max(p_away - bias["away"], 0.0)
    s = a + b + c
    if s <= 0:
        # Defensive; shouldn't happen with sane probs and ~few-pp offsets.
        return p_home, p_draw, p_away
    return a / s, b / s, c / s


# --- divergence classification ----------------------------------------------

def classify_divergence(row) -> str:
    """
    Categorise the model-vs-book gap. Three mutually-exclusive types,
    each mapping to a distinct Claude-commentary shape in Session 22:

      disagree_on_favorite      Model and book pick different argmax
                                outcomes. The strongest signal: the two
                                sources name different winners.

      model_over_concentrated   Same favorite, but model is more confident
                                on that outcome than the book. Typical
                                cause: Elo gap that the market discounts
                                for form/injury/squad reasons.

      model_under_concentrated  Same favorite, but the book is more
                                confident than the model. Typical cause:
                                two teams with similar Elos where the
                                market has prior information (reputation,
                                recent form) that the model doesn't see.

    Returns '' when sportsbook data is missing.
    """
    if pd.isna(row["p_home_book"]):
        return ""
    model_probs = (row["p_home_model_corr"],
                   row["p_draw_model_corr"],
                   row["p_away_model_corr"])
    book_probs = (row["p_home_book"],
                  row["p_draw_book"],
                  row["p_away_book"])
    model_argmax = max(range(3), key=lambda i: model_probs[i])
    book_argmax = max(range(3), key=lambda i: book_probs[i])
    if model_argmax != book_argmax:
        return "disagree_on_favorite"
    if model_probs[book_argmax] > book_probs[book_argmax]:
        return "model_over_concentrated"
    return "model_under_concentrated"


# --- formatting helpers -----------------------------------------------------

def _pct_triple(h: float, d: float, a: float) -> str:
    if any(pd.isna(x) for x in (h, d, a)):
        return "    --      "
    return f"{h*100:4.1f}/{d*100:4.1f}/{a*100:4.1f}"


# --- main pipeline ----------------------------------------------------------

def main() -> None:
    # 1. Load inputs ---------------------------------------------------------
    print("Loading inputs...")
    ratings_df = pd.read_csv(RATINGS_PATH)
    ratings = dict(zip(ratings_df["team"], ratings_df["elo"]))

    fixtures = pd.read_csv(FIXTURES_PATH, parse_dates=["date"])
    print(f"  fixtures:    {len(fixtures)} WC matches")

    def _dedupe_market_rows(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        if df.empty:
            return df
        pair_key = df.apply(
            lambda r: tuple(sorted((r["home_team"], r["away_team"]))),
            axis=1,
        )
        df = df.assign(_pair=pair_key)
        duped = df[df["_pair"].duplicated(keep=False)]
        if len(duped):
            print(f"\nWARNING: dropped {len(duped)//2} duplicate {source_name} rows on unordered team pair:")
            for _, r in duped.iterrows():
                print(f"  {source_name}: {r['home_team']} vs {r['away_team']} -> "
                      f"p_home={r.get('p_home')} p_draw={r.get('p_draw')} p_away={r.get('p_away')}")
            df = df.drop_duplicates(subset=["_pair"], keep="first")
            df = df.drop(columns=["_pair"]).reset_index(drop=True)
        else:
            df = df.drop(columns=["_pair"])
        return df

    book = pd.read_csv(BOOK_PATH)
    book = _dedupe_market_rows(book, "sportsbook")
    print(f"  sportsbook:  {len(book)} match rows")

    poly = pd.read_csv(POLY_PATH)
    poly = _dedupe_market_rows(poly, "Polymarket")
    poly_state = "populated" if len(poly) else "header-only, will auto-populate"
    print(f"  polymarket:  {len(poly)} match rows ({poly_state})")

    bias, n_back = compute_bias_offsets(BACKTEST_PATH)
    print(f"\nBias offsets from {n_back} backtest matches "
          f"(positive => model over-predicts):")
    print(f"  home: {bias['home']:+.4f}")
    print(f"  draw: {bias['draw']:+.4f}")
    print(f"  away: {bias['away']:+.4f}")

    # Early exit when the group stage is complete (fixtures_2026.csv is empty).
    # All 72 group matches have been played and moved to matches_clean.csv.
    # Write header-only outputs so downstream consumers have a stable schema.
    if len(fixtures) == 0:
        empty_cols = [
            "date", "home_team", "away_team", "neutral_used",
            "p_home_model", "p_draw_model", "p_away_model",
            "p_home_model_corr", "p_draw_model_corr", "p_away_model_corr",
            "p_home_book", "p_draw_book", "p_away_book", "n_books",
            "p_home_poly", "p_draw_poly", "p_away_poly", "poly_volume",
            "div_model_book_home", "div_model_book_draw", "div_model_book_away",
            "div_model_book_max", "div_model_book_l1",
            "div_model_poly_l1", "div_book_poly_l1",
            "divergence_type", "note", "flag_divergent",
        ]
        empty_df = pd.DataFrame(columns=empty_cols)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(OUT_PATH, index=False)
        div_snap_dir = Path(_div_snap_dir())
        div_snap_dir.mkdir(parents=True, exist_ok=True)
        div_snap_path = div_snap_dir / f"{_clock_today().isoformat()}.csv"
        empty_df.to_csv(div_snap_path, index=False)
        print("\nGroup stage complete — no upcoming fixtures to compare.")
        print(f"  Wrote header-only: {OUT_PATH}")
        print(f"  Wrote header-only: {div_snap_path}")
        return

    # 2. Run model on every fixture -----------------------------------------
    print(f"\nRunning model on {len(fixtures)} fixtures "
          f"(USE_FIXTURE_NEUTRAL={USE_FIXTURE_NEUTRAL})...")
    rows = []
    for fx in fixtures.itertuples(index=False):
        neutral = bool(fx.neutral) if USE_FIXTURE_NEUTRAL else True
        pred = predict_match(fx.home_team, fx.away_team, ratings, neutral=neutral)
        p_h, p_d, p_a = pred["p_home_win"], pred["p_draw"], pred["p_away_win"]
        ph_c, pd_c, pa_c = bias_correct(p_h, p_d, p_a, bias)
        rows.append({
            "date":              fx.date,
            "home_team":         fx.home_team,
            "away_team":         fx.away_team,
            "neutral_used":      neutral,
            "p_home_model":      round(p_h, 4),
            "p_draw_model":      round(p_d, 4),
            "p_away_model":      round(p_a, 4),
            "p_home_model_corr": round(ph_c, 4),
            "p_draw_model_corr": round(pd_c, 4),
            "p_away_model_corr": round(pa_c, 4),
        })
    df = pd.DataFrame(rows)

    # 3. Left-join sportsbook ----------------------------------------------
    book_renamed = book[["home_team", "away_team",
                         "p_home", "p_draw", "p_away", "n_books"]].rename(
        columns={"p_home": "p_home_book",
                 "p_draw": "p_draw_book",
                 "p_away": "p_away_book"})
    df = df.merge(book_renamed, on=["home_team", "away_team"], how="left")

    matched_book = int(df["p_home_book"].notna().sum())
    print(f"\nSportsbook coverage: {matched_book}/{len(df)} fixtures matched.")
    missing = df.loc[df["p_home_book"].isna(), ["home_team", "away_team"]]
    if len(missing):
        print("  Missing sportsbook for:")
        for _, r in missing.iterrows():
            print(f"    {r['home_team']} vs {r['away_team']}")

    # 4. Left-join Polymarket (today: empty; tomorrow: not) ----------------
    if len(poly) > 0:
        cols_needed = ["home_team", "away_team", "p_home", "p_draw", "p_away"]
        # tolerate older/newer schemas; volume is optional
        keep = [c for c in cols_needed + ["volume"] if c in poly.columns]
        poly_renamed = poly[keep].rename(columns={
            "p_home": "p_home_poly",
            "p_draw": "p_draw_poly",
            "p_away": "p_away_poly",
            "volume": "poly_volume",
        })
        df = df.merge(poly_renamed, on=["home_team", "away_team"], how="left")
        if "poly_volume" not in df.columns:
            df["poly_volume"] = np.nan
    else:
        df["p_home_poly"] = np.nan
        df["p_draw_poly"] = np.nan
        df["p_away_poly"] = np.nan
        df["poly_volume"] = np.nan

    matched_poly = int(df["p_home_poly"].notna().sum())
    print(f"Polymarket coverage: {matched_poly}/{len(df)} fixtures matched.")

    # 5. Divergence metrics -------------------------------------------------
    df["div_model_book_home"] = (df["p_home_model_corr"] - df["p_home_book"]).round(4)
    df["div_model_book_draw"] = (df["p_draw_model_corr"] - df["p_draw_book"]).round(4)
    df["div_model_book_away"] = (df["p_away_model_corr"] - df["p_away_book"]).round(4)
    abs_comp = df[["div_model_book_home",
                   "div_model_book_draw",
                   "div_model_book_away"]].abs()
    df["div_model_book_max"] = abs_comp.max(axis=1).round(4)
    df["div_model_book_l1"]  = abs_comp.sum(axis=1).round(4)

    df["div_model_poly_l1"] = (
        (df["p_home_model_corr"] - df["p_home_poly"]).abs()
        + (df["p_draw_model_corr"] - df["p_draw_poly"]).abs()
        + (df["p_away_model_corr"] - df["p_away_poly"]).abs()
    ).round(4)
    df["div_book_poly_l1"] = (
        (df["p_home_book"] - df["p_home_poly"]).abs()
        + (df["p_draw_book"] - df["p_draw_poly"]).abs()
        + (df["p_away_book"] - df["p_away_poly"]).abs()
    ).round(4)

    df["divergence_type"] = df.apply(classify_divergence, axis=1)

    # 6. Note + flag --------------------------------------------------------
    # Session 33: host advantage is now modelled (neutral_used=False for host
    # home matches), so no matches are excluded from flagging.
    df["note"] = ""

    df["flag_divergent"] = (
        (df["div_model_book_max"] >= DIV_FLAG_THRESHOLD)
        & df["p_home_book"].notna()
    )

    # 7. Save ---------------------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if len(df) != len(fixtures):
        sys.exit(
            f"FATAL: expected {len(fixtures)} rows in triple_compare, got {len(df)}. "
            "Aborting to prevent duplicate fixture publishing."
        )
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows -> {OUT_PATH}")

    # 7b. Dated divergence snapshot (mirrors monte_carlo.py pattern) ------
    div_snap_dir = Path(_div_snap_dir())
    div_snap_dir.mkdir(parents=True, exist_ok=True)
    div_snap_path = div_snap_dir / f"{_clock_today().isoformat()}.csv"
    df.to_csv(div_snap_path, index=False)
    print(f"Snapshot  {len(df)} rows -> {div_snap_path}")

    # 8. Top-10 ranked table -----------------------------------------------
    print(f"\nTop 10 model-vs-book divergences "
          f"(flag threshold = {DIV_FLAG_THRESHOLD:.0%}):\n")

    type_label = {
        "disagree_on_favorite":     "disagree",
        "model_over_concentrated":  "over   ",
        "model_under_concentrated": "under  ",
        "":                         "       ",
    }

    header = (f"  {'date':<10}  {'matchup':<38}  "
              f"{'model H/D/A':<16}  {'book H/D/A':<16}  "
              f"{'max':>5}  type      flg  note")
    print(header)
    print("  " + "-" * (len(header) - 2))

    ranked = (df.dropna(subset=["p_home_book"])
                .sort_values("div_model_book_max", ascending=False)
                .head(10))

    for _, r in ranked.iterrows():
        matchup = f"{r['home_team']} vs {r['away_team']}"
        model_s = _pct_triple(r["p_home_model_corr"],
                              r["p_draw_model_corr"],
                              r["p_away_model_corr"])
        book_s  = _pct_triple(r["p_home_book"],
                              r["p_draw_book"],
                              r["p_away_book"])
        flag = "*" if r["flag_divergent"] else " "
        date_s = (r["date"].date().isoformat()
                  if hasattr(r["date"], "date") else str(r["date"]))
        print(f"  {date_s:<10}  {matchup:<38}  "
              f"{model_s:<16}  {book_s:<16}  "
              f"{r['div_model_book_max']*100:>4.1f}%  "
              f"{type_label[r['divergence_type']]}  "
              f"{flag:>3}  {r['note']}")

    n_flag = int(df["flag_divergent"].sum())
    n_host = int((~df["neutral_used"]).sum())
    print(f"\n{n_flag} matches flagged divergent. "
          f"{n_host} host home matches (neutral_used=False, host advantage applied).")

    # Type breakdown across all matches with sportsbook data
    have_book = df["p_home_book"].notna()
    type_counts = df.loc[have_book, "divergence_type"].value_counts()
    print(f"\nDivergence type breakdown (across {int(have_book.sum())} matches "
          f"with sportsbook data):")
    for k in ("disagree_on_favorite", "model_over_concentrated",
              "model_under_concentrated"):
        print(f"  {k:<28} {int(type_counts.get(k, 0))}")


if __name__ == "__main__":
    main()