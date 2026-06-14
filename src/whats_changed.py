"""
src/whats_changed.py

Session 26: compute daily diff data for the "What changed today" panel.

Pure pandas I/O — no model imports, no display-name mapping, no URLs.
The site generator (generate_site.py) handles all presentation.

Functions
---------
_two_newest(dir_path) -> (prev | None, curr | None)
compute_title_movers(prev_df, curr_df, top_n=5) -> list[dict]
compute_advance_movers(prev_df, curr_df, top_n=5) -> list[dict]
compute_fresh_divergences(prev_div_df, curr_div_df, top_n=3) -> list[dict]
compute_top_divergences(curr_div_df, top_n=5) -> list[dict]

Run from project root to execute self-tests:
    python src/whats_changed.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# MC simulation noise is ≈ ±0.5pp (PROJECT.md §6); suppress sub-noise moves
# so the panel only shows real shifts, not random jitter between runs with
# unchanged inputs.
MIN_MOVE_PP = 0.5

_COL_CHAMPION = "p_champion"         # sums to 1.0 across the 48-team snapshot
_COL_ADVANCE  = "p_advance"          # sums to 32.0 (32 group-stage advancers)
_COL_MAG      = "div_model_book_max" # per-match divergence magnitude column


def _two_newest(
    dir_path: Path | str,
) -> tuple[Optional[Path], Optional[Path]]:
    """Return (prev, curr) — the two newest dated .csv files in dir_path,
    sorted lexically (ISO-date filenames sort chronologically). Returns
    (None, curr) when only one file exists, (None, None) when zero."""
    p = Path(dir_path)
    if not p.exists():
        return None, None
    csvs = sorted(p.glob("*.csv"))
    if not csvs:
        return None, None
    if len(csvs) == 1:
        return None, csvs[-1]
    return csvs[-2], csvs[-1]


def compute_title_movers(
    prev_df: Optional[pd.DataFrame],
    curr_df: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """Top-N title-odds movers: Δ = curr.p_champion − prev.p_champion,
    joined on team name. Drops moves below MIN_MOVE_PP; ranks by |Δ|.
    Returns [] when prev_df is None (baseline / first run)."""
    if prev_df is None:
        return []
    merged = curr_df[["team", _COL_CHAMPION]].merge(
        prev_df[["team", _COL_CHAMPION]],
        on="team", suffixes=("_curr", "_prev"),
    )
    merged["delta"] = merged[f"{_COL_CHAMPION}_curr"] - merged[f"{_COL_CHAMPION}_prev"]
    merged = merged[merged["delta"].abs() * 100 >= MIN_MOVE_PP]
    merged = (merged
              .assign(_abs=merged["delta"].abs())
              .sort_values("_abs", ascending=False)
              .drop(columns="_abs"))
    return [
        {
            "team":      row["team"],
            "prev":      row[f"{_COL_CHAMPION}_prev"],
            "curr":      row[f"{_COL_CHAMPION}_curr"],
            "delta":     row["delta"],
            "direction": "up" if row["delta"] > 0 else "down",
        }
        for _, row in merged.head(top_n).iterrows()
    ]


def compute_advance_movers(
    prev_df: Optional[pd.DataFrame],
    curr_df: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """Top-N group-advance movers: Δ = curr.p_advance − prev.p_advance.
    Same floor and ranking as compute_title_movers. Returns [] when
    prev_df is None."""
    if prev_df is None:
        return []
    merged = curr_df[["team", _COL_ADVANCE]].merge(
        prev_df[["team", _COL_ADVANCE]],
        on="team", suffixes=("_curr", "_prev"),
    )
    merged["delta"] = merged[f"{_COL_ADVANCE}_curr"] - merged[f"{_COL_ADVANCE}_prev"]
    merged = merged[merged["delta"].abs() * 100 >= MIN_MOVE_PP]
    merged = (merged
              .assign(_abs=merged["delta"].abs())
              .sort_values("_abs", ascending=False)
              .drop(columns="_abs"))
    return [
        {
            "team":      row["team"],
            "prev":      row[f"{_COL_ADVANCE}_prev"],
            "curr":      row[f"{_COL_ADVANCE}_curr"],
            "delta":     row["delta"],
            "direction": "up" if row["delta"] > 0 else "down",
        }
        for _, row in merged.head(top_n).iterrows()
    ]


def compute_fresh_divergences(
    prev_div_df: Optional[pd.DataFrame],
    curr_div_df: pd.DataFrame,
    top_n: int = 3,
) -> list[dict]:
    """Top-N freshly-flagged divergences: flag_divergent==True in curr and
    the (home_team, away_team) pair was NOT flagged in prev. Ranked by
    div_model_book_max descending. Returns [] when prev_div_df is None."""
    if prev_div_df is None:
        return []
    flagged = curr_div_df[curr_div_df["flag_divergent"] == True].copy()
    if flagged.empty:
        return []
    prev_flagged_pairs = set(
        zip(
            prev_div_df.loc[prev_div_df["flag_divergent"] == True, "home_team"],
            prev_div_df.loc[prev_div_df["flag_divergent"] == True, "away_team"],
        )
    )
    fresh = flagged[
        ~flagged.apply(
            lambda r: (r["home_team"], r["away_team"]) in prev_flagged_pairs,
            axis=1,
        )
    ].sort_values(_COL_MAG, ascending=False)
    return [
        {
            "home_team":       row["home_team"],
            "away_team":       row["away_team"],
            "date":            str(row["date"]),
            "divergence_type": row["divergence_type"],
            "magnitude":       float(row[_COL_MAG]),
        }
        for _, row in fresh.head(top_n).iterrows()
    ]


def compute_top_divergences(
    curr_div_df: Optional[pd.DataFrame],
    top_n: int = 5,
) -> list[dict]:
    """Top-N currently-flagged divergences among the remaining fixtures,
    ranked by div_model_book_max descending.

    Unlike compute_fresh_divergences (newly-flagged vs yesterday), this is the
    steady, always-populated signal: every flagged fixture is a candidate, not
    just the new ones. Reuses flag_divergent — the established detector that
    already offsets the ~4pp draw/away bias (§6) — rather than a fresh
    magnitude cutoff, so the panel surfaces exactly the gaps the detector
    already trusts, just sorted. Each row also carries the model's favoured
    outcome ("home"/"draw"/"away") for the generator to resolve to a name.

    Returns [] when the frame is None/empty, lacks the flag column, or nothing
    is flagged (late tournament: header-only triple_compare once the group
    stage's flagged fixtures have all been played)."""
    if curr_div_df is None or curr_div_df.empty:
        return []
    if "flag_divergent" not in curr_div_df.columns:
        return []
    flagged = curr_div_df[curr_div_df["flag_divergent"] == True].copy()
    if flagged.empty:
        return []
    flagged = flagged.sort_values(_COL_MAG, ascending=False)
    out = []
    for _, row in flagged.head(top_n).iterrows():
        probs = (row["p_home_model_corr"],
                 row["p_draw_model_corr"],
                 row["p_away_model_corr"])
        fav = ("home", "draw", "away")[max(range(3), key=lambda i: probs[i])]
        out.append({
            "home_team":       row["home_team"],
            "away_team":       row["away_team"],
            "date":            str(row["date"]),
            "divergence_type": row["divergence_type"],
            "magnitude":       float(row[_COL_MAG]),
            "fav_outcome":     fav,
        })
    return out


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # --- _two_newest --------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        assert _two_newest(p) == (None, None), "empty dir should return (None, None)"

        (p / "2026-05-13.csv").touch()
        prev, curr = _two_newest(p)
        assert prev is None, "one file: prev should be None"
        assert curr is not None, "one file: curr should be set"

        (p / "2026-05-14.csv").touch()
        prev, curr = _two_newest(p)
        assert prev.name == "2026-05-13.csv", f"expected 05-13, got {prev.name}"
        assert curr.name == "2026-05-14.csv", f"expected 05-14, got {curr.name}"

    # non-existent directory
    assert _two_newest("/nonexistent/path/xyz") == (None, None), \
        "non-existent dir should return (None, None)"

    # --- synthetic snapshot DataFrames -------------------------------------
    teams = ["Spain", "France", "Argentina", "Brazil", "England"]
    prev_snap = pd.DataFrame({
        "team":      teams,
        _COL_CHAMPION: [0.30, 0.15, 0.20, 0.10, 0.08],
        _COL_ADVANCE:  [0.95, 0.90, 0.92, 0.85, 0.88],
    })
    curr_snap = pd.DataFrame({
        "team":      teams,
        _COL_CHAMPION: [0.28, 0.17, 0.21, 0.10, 0.085],
        _COL_ADVANCE:  [0.94, 0.91, 0.93, 0.85, 0.882],
    })

    # prev=None returns []
    assert compute_title_movers(None, curr_snap) == [], "None prev → []"
    assert compute_advance_movers(None, curr_snap) == [], "None prev → []"

    # correct ordering: Spain (−2pp) and France (+2pp) are top movers
    title_movers = compute_title_movers(prev_snap, curr_snap)
    assert len(title_movers) >= 2, f"expected ≥2 movers, got {len(title_movers)}"
    assert title_movers[0]["team"] in ("Spain", "France"), \
        f"unexpected top mover: {title_movers[0]['team']}"

    # direction flags
    spain = next(m for m in title_movers if m["team"] == "Spain")
    assert spain["direction"] == "down", "Spain −2pp should be 'down'"
    france = next(m for m in title_movers if m["team"] == "France")
    assert france["direction"] == "up", "France +2pp should be 'up'"

    # Brazil: Δ = 0 → excluded
    assert not any(m["team"] == "Brazil" for m in title_movers), \
        "Brazil (no change) should not appear"

    # MIN_MOVE_PP floor: 0.1pp moves should be suppressed
    tiny_curr = prev_snap.copy()
    tiny_curr[_COL_CHAMPION] = prev_snap[_COL_CHAMPION] + 0.001  # 0.1pp
    assert compute_title_movers(prev_snap, tiny_curr) == [], \
        "sub-floor moves should return []"

    # England: Δ = +0.5pp exactly → included (≥ floor)
    edge_curr = prev_snap.copy()
    edge_curr.loc[edge_curr["team"] == "England", _COL_CHAMPION] += 0.005
    edge_movers = compute_title_movers(prev_snap, edge_curr)
    assert any(m["team"] == "England" for m in edge_movers), \
        "exactly-at-floor move (0.5pp) should be included"

    # advance movers: Brazil Δ = 0 → excluded
    adv_movers = compute_advance_movers(prev_snap, curr_snap)
    assert not any(m["team"] == "Brazil" for m in adv_movers), \
        "Brazil unchanged should not appear in advance movers"

    # --- compute_fresh_divergences -----------------------------------------
    curr_div = pd.DataFrame({
        "home_team":      ["Spain", "France", "Brazil"],
        "away_team":      ["Morocco", "Belgium", "Argentina"],
        "date":           ["2026-06-11", "2026-06-12", "2026-06-13"],
        "divergence_type": ["model_over_concentrated",
                            "disagree_on_favorite",
                            "model_under_concentrated"],
        _COL_MAG:         [0.20, 0.18, 0.16],
        "flag_divergent": [True, True, True],
    })

    # prev=None → []
    assert compute_fresh_divergences(None, curr_div) == [], "None prev → []"

    # all three are new (empty prev)
    prev_div_empty = pd.DataFrame({
        "home_team": [], "away_team": [], _COL_MAG: [], "flag_divergent": []
    })
    fresh = compute_fresh_divergences(prev_div_empty, curr_div, top_n=3)
    assert len(fresh) == 3, f"expected 3 fresh, got {len(fresh)}"
    assert fresh[0]["home_team"] == "Spain", \
        f"top magnitude should be Spain, got {fresh[0]['home_team']}"

    # Spain already in prev → only 2 fresh, France first
    prev_div_one = pd.DataFrame({
        "home_team":      ["Spain"],
        "away_team":      ["Morocco"],
        _COL_MAG:         [0.18],
        "flag_divergent": [True],
    })
    fresh2 = compute_fresh_divergences(prev_div_one, curr_div, top_n=3)
    assert len(fresh2) == 2, f"expected 2 fresh, got {len(fresh2)}"
    assert fresh2[0]["home_team"] == "France", \
        f"France should be first after Spain excluded, got {fresh2[0]['home_team']}"

    # no flagged rows in curr → []
    unflagged = curr_div.copy()
    unflagged["flag_divergent"] = False
    assert compute_fresh_divergences(prev_div_empty, unflagged) == [], \
        "no flagged matches should return []"

    # --- compute_top_divergences -------------------------------------------
    top_div = pd.DataFrame({
        "home_team":         ["Spain", "France", "Brazil", "England"],
        "away_team":         ["Morocco", "Belgium", "Argentina", "Croatia"],
        "date":              ["2026-06-11", "2026-06-12", "2026-06-13", "2026-06-14"],
        "divergence_type":   ["model_over_concentrated", "disagree_on_favorite",
                              "model_under_concentrated", "model_over_concentrated"],
        "p_home_model_corr": [0.70, 0.30, 0.45, 0.55],
        "p_draw_model_corr": [0.20, 0.30, 0.25, 0.25],
        "p_away_model_corr": [0.10, 0.40, 0.30, 0.20],
        _COL_MAG:            [0.22, 0.19, 0.25, 0.16],
        "flag_divergent":    [True, True, True, False],
    })

    # None / empty frame → []
    assert compute_top_divergences(None) == [], "None → []"
    assert compute_top_divergences(pd.DataFrame()) == [], "empty df → []"

    # sorted by magnitude desc; unflagged England (highest-strength row dropped)
    td = compute_top_divergences(top_div, top_n=5)
    assert len(td) == 3, f"expected 3 flagged, got {len(td)}"
    assert [d["home_team"] for d in td] == ["Brazil", "Spain", "France"], \
        f"wrong sort order: {[d['home_team'] for d in td]}"
    assert td[0]["magnitude"] == 0.25, "top magnitude should be Brazil's 0.25"

    # model favourite outcome resolved correctly
    assert td[1]["fav_outcome"] == "home", "Spain 0.70 home → fav home"
    assert td[2]["fav_outcome"] == "away", "France 0.40 away → fav away"

    # N-capping
    assert len(compute_top_divergences(top_div, top_n=2)) == 2, "top_n should cap"

    # nothing flagged → []
    none_flagged = top_div.copy()
    none_flagged["flag_divergent"] = False
    assert compute_top_divergences(none_flagged) == [], "no flagged → []"

    # missing flag column → [] (header-only / malformed frame)
    assert compute_top_divergences(top_div.drop(columns=["flag_divergent"])) == [], \
        "missing flag_divergent column should return []"

    print("All self-tests passed.")
