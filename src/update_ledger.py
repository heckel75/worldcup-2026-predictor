"""
src/update_ledger.py — prediction ledger management (Session 29).

Maintains data/processed/wc_predictions.csv: a frozen record of the model's
published probabilities for each WC match, plus the actual result once played.

Schema (matches calibration.py's CANON, which selects by column name):
    match_key, date, home_team, away_team,
    p_home, p_draw, p_away,                      # frozen corrected model probs
    p_home_book, p_draw_book, p_away_book,       # frozen sportsbook (NaN if unposted)
    p_home_poly, p_draw_poly, p_away_poly,       # frozen Polymarket (NaN if unposted)
    poly_volume, neutral_used,
    forecast_ts, outcome,         # outcome in {H, D, A} or "" when unplayed
    actual_home_score, actual_away_score,        # filled by attach_results
    result_note                   # free-text full-result note (e.g. "won 4-3 on
                                  # penalties"); attached, not frozen. Grading is
                                  # on the 90-min scoreline; this is display only.

Public API:
    freeze_new_forecasts(ledger_df, upcoming_fixtures_df, triple_df, today,
                         lookahead_days) -> pd.DataFrame
    attach_results(ledger_df, played_df) -> pd.DataFrame

No datetime.now() inside the core functions — today is always passed in so
self-tests can control the date window without patching.
"""
from __future__ import annotations

import datetime as dt
import json
import re

import pandas as pd

LEDGER_SCHEMA = [
    "match_key", "date", "home_team", "away_team",
    "p_home", "p_draw", "p_away",
    "p_home_book", "p_draw_book", "p_away_book",
    "p_home_poly", "p_draw_poly", "p_away_poly",
    "poly_volume", "neutral_used",
    "lambda_home", "lambda_away", "scoreline_grid", "top_scorelines",
    "forecast_ts", "outcome",
    "actual_home_score", "actual_away_score", "result_note",
]

# Frozen-at-freeze-time market columns, as named in triple_compare.csv (same
# names in the ledger). Missing market data freezes as NaN.
MARKET_COLS = [
    "p_home_book", "p_draw_book", "p_away_book",
    "p_home_poly", "p_draw_poly", "p_away_poly",
    "poly_volume", "neutral_used",
]

# Frozen-at-freeze-time exact-score columns (Session: exact-score view), as
# named in triple_compare.csv. lambda_* freeze as rounded floats; scoreline_grid
# and top_scorelines are opaque JSON strings frozen verbatim (no re-parse /
# re-serialize, which would reorder keys or re-cast and break the round-trip).
# The 8 pre-existing scoreline-less ledger rows stay empty by design — their
# source grids are gone and a recompute would use post-match ratings.
SCORELINE_COLS = [
    "lambda_home", "lambda_away", "scoreline_grid", "top_scorelines",
]


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Header-extend an old-schema ledger: add any missing columns as NaN,
    in LEDGER_SCHEMA order. Existing values are never touched."""
    df = df.copy()
    for col in LEDGER_SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA
    return df[LEDGER_SCHEMA + [c for c in df.columns if c not in LEDGER_SCHEMA]]


# ---------------------------------------------------------------------------
# Key helpers (identical pattern to generate_previews.py / generate_site.py)
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def make_match_key(date_iso: str, home: str, away: str) -> str:
    return f"{date_iso}_{slugify(home)}_vs_{slugify(away)}"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _freeze_p(val):
    """Round a probability/lambda to 4dp, or pd.NA when missing."""
    return round(float(val), 4) if pd.notna(val) else pd.NA


def _refreeze_row(ledger_df: pd.DataFrame, idx, row, now_ts: str) -> None:
    """Overwrite an UNLOCKED ledger row's forecast columns with this build's
    triple_compare values (FREEZE-C re-freeze: "frozen = last published").

    COLUMN-WISE NaN-SAFE: each column is written ONLY when the incoming value is
    non-NaN; a NaN incoming value leaves the existing frozen value in place. This
    is the guard that makes a re-freeze safe when a market has dropped to NaN —
    a kicked-off match the fetchers no longer price (a "limbo" row) keeps its
    good frozen book/poly instead of being nulled out.

    Touches ONLY forecast columns (model probs, markets, scoreline, forecast_ts).
    Never outcome / actual_home_score / actual_away_score / result_note — those
    are attach_results-owned and define the lock.
    """
    for ledger_col, src_col in (("p_home", "p_home_model_corr"),
                                ("p_draw", "p_draw_model_corr"),
                                ("p_away", "p_away_model_corr")):
        v = row.get(src_col)
        if pd.notna(v):
            ledger_df.at[idx, ledger_col] = round(float(v), 4)
    for col in MARKET_COLS:
        v = row.get(col)
        if pd.notna(v):
            ledger_df.at[idx, col] = round(float(v), 4) if col.startswith("p_") else v
    for col in SCORELINE_COLS:
        v = row.get(col)
        if v is not None and pd.notna(v):
            ledger_df.at[idx, col] = round(float(v), 4) if col.startswith("lambda_") else v
    # forecast_ts records the LAST re-freeze (the binding pre-kickoff snapshot).
    ledger_df.at[idx, "forecast_ts"] = now_ts


def freeze_new_forecasts(
    ledger_df: pd.DataFrame,
    upcoming_fixtures_df: pd.DataFrame,
    triple_df: pd.DataFrame,
    today: dt.date,
    lookahead_days: int,
) -> pd.DataFrame:
    """
    Ensure every fixture in the freeze window [today, today + lookahead_days]
    carries the LATEST pre-match forecast in the ledger, then return it.

    FREEZE-C — re-freeze-until-played, then lock (replaces freeze-once):
      - A windowed fixture with NO ledger row gets one CREATED (robustness: the
        row exists well before kickoff, so a missed build can't lose the match).
      - A windowed fixture with an UNLOCKED row gets its forecast columns
        OVERWRITTEN with this build's triple_compare values, so the binding
        freeze is the LAST build before kickoff ("frozen = last published").
      - A row is permanently immutable (LOCKED) when EITHER:
          (a) attach_results has written its outcome — the match was played; OR
          (b) it has left the forecast set — no triple_compare row, so the
              left-join leaves model probs (p_*_model_corr) NaN. (triple_compare
              always computes model probs for a live fixture and drops played
              matches, so NaN model ⟺ gone from the forecast set.)
        Locked rows are never created and never overwritten.

    Because freeze runs BEFORE attach in a build (update.py), the entry build's
    protection for a just-played match is lock-(b): clean_data / derive_ko_fixtures
    have already removed it from triple_compare, so its merge model probs are NaN
    and it is skipped. Lock-(a) then holds it on every subsequent build. The
    column-wise NaN-safe overwrite (_refreeze_row) is the third layer: even a
    row still in the window whose MARKET went NaN keeps its good frozen markets.

    Probabilities come from triple_df's bias-corrected model columns plus the
    market and scoreline columns from the same row. Never touches outcome /
    actual scores / result_note (attach_results owns those).
    """
    ledger_df = _ensure_schema(ledger_df)

    # Map existing rows by key (first occurrence) for in-place re-freeze.
    key_to_idx: dict[str, int] = {}
    for i, k in zip(ledger_df.index, ledger_df["match_key"]):
        if pd.notna(k) and k not in key_to_idx:
            key_to_idx[k] = i

    # Resolve the date window
    cutoff_lo = pd.Timestamp(today)
    cutoff_hi = pd.Timestamp(today + dt.timedelta(days=lookahead_days))

    # Ensure fixture dates are Timestamps
    fixtures = upcoming_fixtures_df.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"])
    window = fixtures[
        (fixtures["date"] >= cutoff_lo) & (fixtures["date"] <= cutoff_hi)
    ]

    if window.empty:
        return ledger_df

    # Join corrected probs + market columns from triple_df on (home_team, away_team)
    triple_cols = [
        "home_team", "away_team",
        "p_home_model_corr", "p_draw_model_corr", "p_away_model_corr",
    ] + [c for c in MARKET_COLS + SCORELINE_COLS if c in triple_df.columns]
    triple = triple_df[triple_cols].copy()
    merged = window.merge(triple, on=["home_team", "away_team"], how="left")

    now_ts = dt.datetime.now().isoformat(timespec="seconds")
    new_rows = []
    created: set[str] = set()
    refrozen = 0
    for _, row in merged.iterrows():
        date_iso = row["date"].date().isoformat()
        key = make_match_key(date_iso, row["home_team"], row["away_team"])

        # Lock-(b): a fixture with no live model forecast (left-join NaN) has
        # left the forecast set — never create, never overwrite.
        if any(pd.isna(row.get(c)) for c in
               ("p_home_model_corr", "p_draw_model_corr", "p_away_model_corr")):
            continue

        if key in key_to_idx:
            idx = key_to_idx[key]
            # Lock-(a): played. Once attach_results writes an outcome the row is
            # permanently immutable — its frozen probs are the last pre-kickoff
            # values and must never change.
            outcome_val = ledger_df.at[idx, "outcome"]
            if pd.notna(outcome_val) and str(outcome_val).strip() != "":
                continue
            _refreeze_row(ledger_df, idx, row, now_ts)
            refrozen += 1
            continue

        if key in created:
            continue  # defensive: same fixture appeared twice in this build

        rec = {
            "match_key":   key,
            "date":        date_iso,
            "home_team":   row["home_team"],
            "away_team":   row["away_team"],
            "p_home":      round(float(row["p_home_model_corr"]), 4),
            "p_draw":      round(float(row["p_draw_model_corr"]), 4),
            "p_away":      round(float(row["p_away_model_corr"]), 4),
            "forecast_ts": now_ts,
            "outcome":     "",
            "actual_home_score": pd.NA,
            "actual_away_score": pd.NA,
        }
        for col in MARKET_COLS:
            val = row.get(col)
            if col.startswith("p_"):
                rec[col] = _freeze_p(val)
            else:
                rec[col] = val if pd.notna(val) else pd.NA
        # Exact-score columns: lambdas as rounded floats; the two JSON strings
        # frozen verbatim (opaque — never re-parsed/re-serialized here).
        for col in SCORELINE_COLS:
            val = row.get(col)
            if col.startswith("lambda_"):
                rec[col] = _freeze_p(val)
            else:
                rec[col] = val if (val is not None and pd.notna(val)) else pd.NA
        new_rows.append(rec)
        created.add(key)

    if refrozen:
        print(f"  Re-froze {refrozen} unlocked forecast row(s) to the latest pre-match values.")

    if not new_rows:
        # Ensure any legacy duplicates are collapsed even if no new rows were added.
        if not ledger_df.empty:
            deduped = ledger_df.drop_duplicates(subset=["match_key"], keep="first")
            if len(deduped) != len(ledger_df):
                print(f"  WARNING: collapsed {len(ledger_df) - len(deduped)} duplicate ledger rows")
            return deduped.reset_index(drop=True)
        return ledger_df

    new_df = pd.DataFrame(new_rows, columns=LEDGER_SCHEMA)
    ledger_df = pd.concat([ledger_df, new_df], ignore_index=True)
    if not ledger_df.empty:
        deduped = ledger_df.drop_duplicates(subset=["match_key"], keep="first")
        if len(deduped) != len(ledger_df):
            print(f"  WARNING: collapsed {len(ledger_df) - len(deduped)} duplicate ledger rows")
        ledger_df = deduped.reset_index(drop=True)
    return ledger_df


def attach_results(
    ledger_df: pd.DataFrame,
    played_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each unscored ledger row (outcome == ""), check whether the match now
    appears in played_df. If yes, set outcome to "H", "D", or "A" graded to 90
    minutes: a knockout tie decided in extra time or on penalties (the played
    row's `decided_in` in {"et","pens"}) grades as the DRAW it was at 90,
    regardless of the ET-inclusive scoreline entered for display; otherwise the
    outcome is derived from the scoreline. `advanced` (who progressed) is read
    elsewhere for the sim/bracket and never as the W/D/L outcome here. Also fills
    actual_home_score / actual_away_score for any matched row whose score cells
    are still NaN.

    Fills result_note (a free-text full-result annotation, e.g. "won 4-3 on
    penalties") from the played row for any row whose note is still blank and
    that carries a non-blank note in played_df — keyed off the note column /
    played-row presence (the score-loop pattern), NOT the outcome mask, so it
    self-heals on rebuild and an already-scored row backfills its note. Display
    only: never affects the graded outcome.

    All three fills are idempotent — populated values are never overwritten; the
    self-heal also fixes rows scored before these columns existed.

    Never changes p_home / p_draw / p_away / frozen market columns / forecast_ts.
    A match that was played but never frozen is not back-filled.
    """
    if ledger_df.empty:
        return ledger_df
    ledger_df = _ensure_schema(ledger_df)

    # Build a lookup: match_key -> outcome string from played results
    played = played_df.copy()
    played["date"] = pd.to_datetime(played["date"])

    def _outcome(row) -> str:
        # Grade to 90 minutes. A knockout tie level after 90 that was settled in
        # extra time or on penalties grades as the DRAW it was at 90, regardless
        # of the ET-inclusive scoreline entered for display (decided_in in
        # {"et","pens"}). `advanced` (progression) is read by the sim/bracket,
        # never here. Blank / "regulation" / any other value → derive from the
        # scoreline. .get() degrades safely for played_df frames that predate the
        # column (self-tests, old matches_clean).
        decided_raw = row.get("decided_in")
        decided = ("" if (decided_raw is None or pd.isna(decided_raw))
                   else str(decided_raw).strip().lower())
        if decided in ("et", "pens"):
            return "D"
        if row["home_score"] > row["away_score"]:
            return "H"
        if row["home_score"] < row["away_score"]:
            return "A"
        return "D"

    played["_key"] = played.apply(
        lambda r: make_match_key(
            r["date"].date().isoformat(), r["home_team"], r["away_team"]
        ),
        axis=1,
    )
    played["_outcome"] = played.apply(_outcome, axis=1)
    result_map: dict[str, str] = dict(zip(played["_key"], played["_outcome"]))
    score_map: dict[str, tuple[int, int]] = {
        k: (int(h), int(a))
        for k, h, a in zip(played["_key"], played["home_score"], played["away_score"])
    }
    # Free-text full-result notes (only non-blank ones — a blank note means
    # "no annotation" and must never overwrite an existing note). played_df may
    # predate the column (matches_clean built before the feed carried it), so
    # .get() degrades to no notes rather than KeyError.
    note_series = played.get("result_note")
    note_map: dict[str, str] = (
        {
            k: str(n).strip()
            for k, n in zip(played["_key"], note_series)
            if pd.notna(n) and str(n).strip() != ""
        }
        if note_series is not None else {}
    )

    ledger = ledger_df.copy()
    unscored_mask = ledger["outcome"].isna() | (ledger["outcome"] == "")
    for idx in ledger[unscored_mask].index:
        key = ledger.at[idx, "match_key"]
        if key in result_map:
            ledger.at[idx, "outcome"] = result_map[key]

    # Fill actual scores wherever they're still missing (never overwrite).
    scoreless_mask = ledger["actual_home_score"].isna()
    for idx in ledger[scoreless_mask].index:
        key = ledger.at[idx, "match_key"]
        if key in score_map:
            hg, ag = score_map[key]
            ledger.at[idx, "actual_home_score"] = hg
            ledger.at[idx, "actual_away_score"] = ag

    # Fill the full-result note wherever it's still blank and a note exists in
    # the feed (score-loop pattern: keyed off the note column being empty +
    # played-row presence, not the outcome mask — self-heals on rebuild).
    note_missing_mask = ledger["result_note"].isna() | (ledger["result_note"] == "")
    for idx in ledger[note_missing_mask].index:
        key = ledger.at[idx, "match_key"]
        if key in note_map:
            ledger.at[idx, "result_note"] = note_map[key]

    return ledger


def find_unfrozen_played(
    ledger_df: pd.DataFrame,
    played_df: pd.DataFrame,
) -> list[str]:
    """Return match_keys of played 2026 WC matches that have NO ledger row
    (played-but-never-frozen) — the belt-and-suspenders detector for FREEZE-C.

    With re-freeze-until-played, every match should get a ledger row created
    well before kickoff, so this should always be empty. A non-empty result
    means a match was played without ever freezing: its match page, verdict,
    "Who's been closer" credit, calibration row and Divergence Log entry will
    all be missing (the §6 played-but-never-frozen vanish), and the pre-match
    forecast is unrecoverable. update.py logs a loud WARNING on a non-empty
    return.

    Date-bounded to 2026 (§6: matches_clean spans all WC editions).
    """
    if played_df is None or played_df.empty:
        return []
    played = played_df.copy()
    played["date"] = pd.to_datetime(played["date"])
    mask = (
        (played.get("tournament") == "FIFA World Cup")
        & (played["date"].dt.year == 2026)
        & played["home_score"].notna()
        & played["away_score"].notna()
    )
    played = played[mask]
    if played.empty:
        return []
    ledger_keys = (
        set(ledger_df["match_key"].dropna())
        if "match_key" in ledger_df.columns else set()
    )
    missing = []
    for _, r in played.iterrows():
        key = make_match_key(r["date"].date().isoformat(), r["home_team"], r["away_team"])
        if key not in ledger_keys:
            missing.append(key)
    return missing


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TODAY = dt.date(2026, 6, 11)

    # Build synthetic fixtures: three matches on day 0, +1, +2
    _dates = [TODAY, TODAY + dt.timedelta(1), TODAY + dt.timedelta(2)]
    _teams = [
        ("Alpha", "Beta"),
        ("Gamma", "Delta"),
        ("Epsilon", "Zeta"),
    ]
    fixtures_df = pd.DataFrame({
        "date":      [pd.Timestamp(d) for d in _dates],
        "home_team": [t[0] for t in _teams],
        "away_team": [t[1] for t in _teams],
        "neutral":   [True, True, True],
    })

    # Synthetic triple_df with corrected probs + market columns.
    # Match B has no sportsbook market (NaN must freeze as NaN).
    triple_df = pd.DataFrame({
        "home_team":           [t[0] for t in _teams],
        "away_team":           [t[1] for t in _teams],
        "p_home_model_corr":   [0.50, 0.30, 0.45],
        "p_draw_model_corr":   [0.25, 0.35, 0.28],
        "p_away_model_corr":   [0.25, 0.35, 0.27],
        "p_home_book":         [0.48, float("nan"), 0.44],
        "p_draw_book":         [0.27, float("nan"), 0.29],
        "p_away_book":         [0.25, float("nan"), 0.27],
        "p_home_poly":         [0.47, 0.32, 0.43],
        "p_draw_poly":         [0.28, 0.33, 0.30],
        "p_away_poly":         [0.25, 0.35, 0.27],
        "poly_volume":         [100000, 5000, 20000],
        "neutral_used":        [True, False, True],
        "lambda_home":         [1.80, 1.20, 1.55],
        "lambda_away":         [1.10, 1.20, 1.05],
        # Opaque JSON strings, exactly as triple_compare emits them.
        "scoreline_grid":      [json.dumps([[0.1, 0.05], [0.05, 0.02]]),
                                json.dumps([[0.08, 0.06], [0.06, 0.03]]),
                                json.dumps([[0.09, 0.05], [0.05, 0.02]])],
        "top_scorelines":      [json.dumps([{"score": "1-0", "prob": 0.12}]),
                                json.dumps([{"score": "1-1", "prob": 0.11}]),
                                json.dumps([{"score": "2-1", "prob": 0.10}])],
    })

    # Empty ledger with correct schema
    empty_ledger = pd.DataFrame(columns=LEDGER_SCHEMA)

    # --- Test 1: freeze appends only imminent fixtures (lookahead=1) ---
    ledger = freeze_new_forecasts(
        empty_ledger, fixtures_df, triple_df, TODAY, lookahead_days=1
    )
    assert len(ledger) == 2, f"Expected 2 rows, got {len(ledger)}"
    keys_frozen = set(ledger["match_key"])
    key_a = make_match_key(TODAY.isoformat(), "Alpha", "Beta")
    key_b = make_match_key((TODAY + dt.timedelta(1)).isoformat(), "Gamma", "Delta")
    key_c = make_match_key((TODAY + dt.timedelta(2)).isoformat(), "Epsilon", "Zeta")
    assert key_a in keys_frozen, "Match A (today) should be frozen"
    assert key_b in keys_frozen, "Match B (today+1) should be frozen"
    assert key_c not in keys_frozen, "Match C (today+2) should NOT be frozen"
    assert (ledger["outcome"] == "").all(), "All outcomes should be empty after freeze"

    # --- Test 1b: market columns frozen alongside model probs ---
    row_a = ledger.loc[ledger["match_key"] == key_a].iloc[0]
    assert float(row_a["p_home_book"]) == 0.48, "Book prob should freeze for match A"
    assert float(row_a["p_home_poly"]) == 0.47, "Poly prob should freeze for match A"
    assert float(row_a["poly_volume"]) == 100000, "poly_volume should freeze"
    assert bool(row_a["neutral_used"]) is True, "neutral_used should freeze"
    row_b = ledger.loc[ledger["match_key"] == key_b].iloc[0]
    assert pd.isna(row_b["p_home_book"]), "Absent book market must freeze as NaN"
    assert float(row_b["p_home_poly"]) == 0.32, "Poly prob should freeze for match B"
    assert pd.isna(row_a["actual_home_score"]), "Scores empty after freeze"

    # --- Test 1c: exact-score columns freeze; JSON strings round-trip verbatim ---
    assert float(row_a["lambda_home"]) == 1.80, "lambda_home should freeze"
    assert float(row_a["lambda_away"]) == 1.10, "lambda_away should freeze"
    assert json.loads(row_a["scoreline_grid"]) == [[0.1, 0.05], [0.05, 0.02]], \
        "scoreline_grid must round-trip through json.loads identically"
    assert json.loads(row_a["top_scorelines"]) == [{"score": "1-0", "prob": 0.12}], \
        "top_scorelines must round-trip through json.loads identically"

    # --- Test 2: re-running freeze with IDENTICAL inputs adds no rows and leaves
    #     values unchanged (FREEZE-C: re-freeze is idempotent on identical input) ---
    p_home_a_before = float(ledger.loc[ledger["match_key"] == key_a, "p_home"].iloc[0])
    book_a_before   = float(ledger.loc[ledger["match_key"] == key_a, "p_home_book"].iloc[0])
    ledger2 = freeze_new_forecasts(
        ledger, fixtures_df, triple_df, TODAY, lookahead_days=1
    )
    assert len(ledger2) == 2, "Re-freeze should not add duplicate rows"
    assert float(ledger2.loc[ledger2["match_key"] == key_a, "p_home"].iloc[0]) == p_home_a_before, \
        "Re-freeze with identical triple must leave model probs unchanged"
    assert float(ledger2.loc[ledger2["match_key"] == key_a, "p_home_book"].iloc[0]) == book_a_before, \
        "Re-freeze with identical triple must leave market probs unchanged"

    # --- Test 3: attach_results fills outcome without touching p_* ---
    p_home_before = float(ledger.loc[ledger["match_key"] == key_a, "p_home"].iloc[0])
    played_df = pd.DataFrame({
        "date":       [pd.Timestamp(TODAY)],
        "home_team":  ["Alpha"],
        "away_team":  ["Beta"],
        "home_score": [2],
        "away_score": [1],
        "tournament": ["FIFA World Cup"],
    })
    book_before = float(ledger.loc[ledger["match_key"] == key_a, "p_home_book"].iloc[0])
    ledger3 = attach_results(ledger, played_df)
    outcome_a = ledger3.loc[ledger3["match_key"] == key_a, "outcome"].iloc[0]
    assert outcome_a == "H", f"Alpha 2–1 Beta should be 'H', got '{outcome_a}'"
    p_home_after = float(ledger3.loc[ledger3["match_key"] == key_a, "p_home"].iloc[0])
    assert p_home_before == p_home_after, "p_home must not change after attach_results"
    book_after = float(ledger3.loc[ledger3["match_key"] == key_a, "p_home_book"].iloc[0])
    assert book_before == book_after, "frozen book prob must not change after attach_results"
    outcome_b = ledger3.loc[ledger3["match_key"] == key_b, "outcome"].iloc[0]
    assert outcome_b == "", "Match B not yet played — outcome should remain empty"

    # --- Test 3b: actual scores attached; populated scores never overwritten ---
    row_a3 = ledger3.loc[ledger3["match_key"] == key_a].iloc[0]
    assert int(row_a3["actual_home_score"]) == 2 and int(row_a3["actual_away_score"]) == 1, \
        "attach_results should fill actual scores"
    rescored = played_df.copy()
    rescored["home_score"] = [5]   # bogus re-report must not overwrite
    ledger3b = attach_results(ledger3, rescored)
    assert int(ledger3b.loc[ledger3b["match_key"] == key_a, "actual_home_score"].iloc[0]) == 2, \
        "Populated actual scores must never be overwritten"

    # --- Test 3c: old-schema ledger self-heals (scored row, no score columns) ---
    legacy = ledger3[["match_key", "date", "home_team", "away_team",
                      "p_home", "p_draw", "p_away", "forecast_ts", "outcome"]].copy()
    healed = attach_results(legacy, played_df)
    assert int(healed.loc[healed["match_key"] == key_a, "actual_home_score"].iloc[0]) == 2, \
        "Old-schema scored row should get actual scores filled"
    assert "p_home_book" in healed.columns, "_ensure_schema should header-extend old ledgers"

    # --- Test 4: played-but-never-frozen match has no ledger row ---
    never_frozen_played = pd.DataFrame({
        "date":       [pd.Timestamp(TODAY + dt.timedelta(2))],
        "home_team":  ["Epsilon"],
        "away_team":  ["Zeta"],
        "home_score": [0],
        "away_score": [0],
        "tournament": ["FIFA World Cup"],
    })
    ledger4 = attach_results(ledger, never_frozen_played)
    assert key_c not in set(ledger4["match_key"]), \
        "Never-frozen match must not appear in ledger after attach_results"

    # --- Test 5: KO penalty shootout grades on the 90-min scoreline (a draw),
    #     and the full-result note attaches as display-only text ---
    ko_fixtures = pd.DataFrame({
        "date":      [pd.Timestamp(TODAY)],
        "home_team": ["Theta"],
        "away_team": ["Iota"],
        "neutral":   [True],
    })
    ko_triple = pd.DataFrame({
        "home_team":         ["Theta"],
        "away_team":         ["Iota"],
        "p_home_model_corr": [0.40],
        "p_draw_model_corr": [0.30],
        "p_away_model_corr": [0.30],
    })
    ko_ledger = freeze_new_forecasts(
        pd.DataFrame(columns=LEDGER_SCHEMA), ko_fixtures, ko_triple,
        TODAY, lookahead_days=0,
    )
    key_ko = make_match_key(TODAY.isoformat(), "Theta", "Iota")
    assert key_ko in set(ko_ledger["match_key"]), "KO fixture should freeze"
    assert ko_ledger.loc[ko_ledger["match_key"] == key_ko, "result_note"].isna().all(), \
        "result_note is unset at freeze time (attached, not frozen)"

    # Level after 90 min, advanced set, full-result note present — the pens case.
    ko_played = pd.DataFrame({
        "date":        [pd.Timestamp(TODAY)],
        "home_team":   ["Theta"],
        "away_team":   ["Iota"],
        "home_score":  [1],
        "away_score":  [1],
        "advanced":    ["Theta"],            # progression only — must NOT grade
        "decided_in":  ["pens"],             # explicit pens enum → draw
        "result_note": ["won 4-3 on penalties"],
        "tournament":  ["FIFA World Cup"],
    })
    ko_scored = attach_results(ko_ledger, ko_played)
    row_ko = ko_scored.loc[ko_scored["match_key"] == key_ko].iloc[0]
    assert row_ko["outcome"] == "D", \
        f"pens match must grade as a draw on the 90-min score, got '{row_ko['outcome']}'"
    assert row_ko["result_note"] == "won 4-3 on penalties", \
        f"result_note must attach, got '{row_ko['result_note']}'"
    assert int(row_ko["actual_home_score"]) == 1 and int(row_ko["actual_away_score"]) == 1, \
        "actual scores should be the 90-min scoreline"

    # Idempotent + never-overwrite: re-running with a different note keeps the first.
    ko_played_b = ko_played.copy()
    ko_played_b["result_note"] = ["won on penalties (re-report)"]
    ko_scored_b = attach_results(ko_scored, ko_played_b)
    assert ko_scored_b.loc[ko_scored_b["match_key"] == key_ko, "result_note"].iloc[0] \
        == "won 4-3 on penalties", "Populated result_note must never be overwritten"

    # A played row with a blank note leaves result_note unset (no annotation).
    blank_note_played = pd.DataFrame({
        "date":        [pd.Timestamp(TODAY)],
        "home_team":   ["Alpha"],
        "away_team":   ["Beta"],
        "home_score":  [2],
        "away_score":  [1],
        "result_note": [""],
        "tournament":  ["FIFA World Cup"],
    })
    blank_scored = attach_results(ledger, blank_note_played)
    assert pd.isna(blank_scored.loc[blank_scored["match_key"] == key_a, "result_note"].iloc[0]), \
        "A blank feed note must leave result_note unset (no annotation)"

    # --- Test 6: ET-DECIDED KO (won in extra time, NOT level at 90 on the
    #     entered ET-inclusive scoreline) grades to 90 as a DRAW via
    #     decided_in="et", even though the scoreline is a home win. Display keeps
    #     the ET score; advanced records the winner; result_note attaches. This
    #     is the seam decided_in closes — a 3-2 aet grades D, not H. ---
    et_fixtures = pd.DataFrame({
        "date":      [pd.Timestamp(TODAY)],
        "home_team": ["Xi"], "away_team": ["Omicron"], "neutral": [True],
    })
    et_triple = pd.DataFrame({
        "home_team": ["Xi"], "away_team": ["Omicron"],
        "p_home_model_corr": [0.40], "p_draw_model_corr": [0.30], "p_away_model_corr": [0.30],
    })
    et_ledger = freeze_new_forecasts(
        pd.DataFrame(columns=LEDGER_SCHEMA), et_fixtures, et_triple,
        TODAY, lookahead_days=0,
    )
    key_et = make_match_key(TODAY.isoformat(), "Xi", "Omicron")
    et_played = pd.DataFrame({
        "date":        [pd.Timestamp(TODAY)],
        "home_team":   ["Xi"], "away_team": ["Omicron"],
        "home_score":  [3], "away_score": [2],   # ET-inclusive display scoreline
        "advanced":    ["Xi"],
        "decided_in":  ["et"],
        "result_note": ["After extra time"],
        "tournament":  ["FIFA World Cup"],
    })
    et_scored = attach_results(et_ledger, et_played)
    row_et = et_scored.loc[et_scored["match_key"] == key_et].iloc[0]
    assert row_et["outcome"] == "D", \
        f"ET-won KO must grade to 90 as a draw via decided_in, got '{row_et['outcome']}'"
    assert int(row_et["actual_home_score"]) == 3 and int(row_et["actual_away_score"]) == 2, \
        "displayed scoreline stays the ET-inclusive 3-2"
    assert row_et["result_note"] == "After extra time", \
        "result_note must attach untouched on an ET-decided KO"

    # --- Test 6b: REGULATION (blank decided_in) still grades from the scoreline.
    #     Re-attach onto the still-unscored et_ledger with decided_in blank. ---
    reg_played = et_played.copy()
    reg_played["decided_in"] = [""]          # blank = regulation
    reg_scored = attach_results(et_ledger, reg_played)
    assert reg_scored.loc[reg_scored["match_key"] == key_et, "outcome"].iloc[0] == "H", \
        "regulation (blank decided_in) must grade from the scoreline (3-2 -> H)"

    # =====================================================================
    # FREEZE-C tests: re-freeze-while-unplayed, lock on play, NaN-safe guard.
    # =====================================================================

    # --- FC-1: re-freeze UPDATES an unlocked row when the triple changes ---
    #     (the WC_ASOF_DATE "two consecutive pre-match builds" case, unit-level:
    #      both calls pass `today` explicitly so no real build is needed.)
    fc_fixtures = pd.DataFrame({
        "date":      [pd.Timestamp(TODAY)],
        "home_team": ["Kappa"], "away_team": ["Lambda"], "neutral": [True],
    })
    fc_triple_v1 = pd.DataFrame({
        "home_team": ["Kappa"], "away_team": ["Lambda"],
        "p_home_model_corr": [0.50], "p_draw_model_corr": [0.25], "p_away_model_corr": [0.25],
        "p_home_book": [0.48], "p_draw_book": [0.27], "p_away_book": [0.25],
        "p_home_poly": [0.47], "p_draw_poly": [0.28], "p_away_poly": [0.25],
        "poly_volume": [100000], "neutral_used": [True],
    })
    fc = freeze_new_forecasts(pd.DataFrame(columns=LEDGER_SCHEMA), fc_fixtures,
                              fc_triple_v1, TODAY, lookahead_days=1)
    key_fc = make_match_key(TODAY.isoformat(), "Kappa", "Lambda")
    assert float(fc.loc[fc["match_key"] == key_fc, "p_home"].iloc[0]) == 0.50
    # Second pre-match build: model + markets moved.
    fc_triple_v2 = fc_triple_v1.copy()
    fc_triple_v2["p_home_model_corr"] = [0.60]
    fc_triple_v2["p_home_book"] = [0.55]
    fc2 = freeze_new_forecasts(fc, fc_fixtures, fc_triple_v2, TODAY, lookahead_days=1)
    assert len(fc2) == 1, "re-freeze must not duplicate the row"
    assert float(fc2.loc[fc2["match_key"] == key_fc, "p_home"].iloc[0]) == 0.60, \
        "FC-1: unlocked row's model prob must UPDATE to the latest build, not stick"
    assert float(fc2.loc[fc2["match_key"] == key_fc, "p_home_book"].iloc[0]) == 0.55, \
        "FC-1: unlocked row's market must UPDATE to the latest build"

    # --- FC-2: LOCK ON PLAY — once attach writes an outcome, a later build does
    #     NOT overwrite the frozen probs (they hold the last pre-kickoff value) ---
    fc_played = pd.DataFrame({
        "date": [pd.Timestamp(TODAY)], "home_team": ["Kappa"], "away_team": ["Lambda"],
        "home_score": [2], "away_score": [0], "tournament": ["FIFA World Cup"],
    })
    fc_locked = attach_results(fc2, fc_played)
    assert fc_locked.loc[fc_locked["match_key"] == key_fc, "outcome"].iloc[0] == "H"
    fc_triple_v3 = fc_triple_v1.copy()
    fc_triple_v3["p_home_model_corr"] = [0.99]   # a build trying to clobber
    fc_triple_v3["p_home_book"] = [0.99]
    fc_after = freeze_new_forecasts(fc_locked, fc_fixtures, fc_triple_v3, TODAY, lookahead_days=1)
    assert float(fc_after.loc[fc_after["match_key"] == key_fc, "p_home"].iloc[0]) == 0.60, \
        "FC-2: a LOCKED (played) row's model prob must NOT change"
    assert float(fc_after.loc[fc_after["match_key"] == key_fc, "p_home_book"].iloc[0]) == 0.55, \
        "FC-2: a LOCKED (played) row's market must NOT change"

    # --- FC-3: CLOBBER REFUSAL — frozen, then GONE from the forecast set (no
    #     triple row -> left-join NaN model = lock-(b)). A build must not touch it,
    #     even if it weren't yet outcome-locked. Use an UNLOCKED row to prove the
    #     lock-(b) path alone (not relying on the outcome lock). ---
    gone_triple = pd.DataFrame(columns=[
        "home_team", "away_team",
        "p_home_model_corr", "p_draw_model_corr", "p_away_model_corr",
    ])  # the fixture is in the window but absent from triple -> NaN model on merge
    fc_gone = freeze_new_forecasts(fc2, fc_fixtures, gone_triple, TODAY, lookahead_days=1)
    assert float(fc_gone.loc[fc_gone["match_key"] == key_fc, "p_home"].iloc[0]) == 0.60, \
        "FC-3: a row gone from the forecast set (NaN model) must NOT be overwritten"
    assert float(fc_gone.loc[fc_gone["match_key"] == key_fc, "p_home_book"].iloc[0]) == 0.55, \
        "FC-3: gone-from-set row's market must be preserved (no clobber)"

    # --- FC-4: LIMBO / NaN-MARKET — an unlocked row still in window+triple with
    #     VALID model but NaN incoming book/poly: model re-freezes, markets are
    #     PRESERVED, not nulled. This is the guard protecting today's 6 KO rows. ---
    limbo_triple = pd.DataFrame({
        "home_team": ["Kappa"], "away_team": ["Lambda"],
        "p_home_model_corr": [0.70], "p_draw_model_corr": [0.20], "p_away_model_corr": [0.10],
        "p_home_book": [float("nan")], "p_draw_book": [float("nan")], "p_away_book": [float("nan")],
        "p_home_poly": [float("nan")], "p_draw_poly": [float("nan")], "p_away_poly": [float("nan")],
        "poly_volume": [float("nan")], "neutral_used": [True],
    })
    fc_limbo = freeze_new_forecasts(fc2, fc_fixtures, limbo_triple, TODAY, lookahead_days=1)
    assert float(fc_limbo.loc[fc_limbo["match_key"] == key_fc, "p_home"].iloc[0]) == 0.70, \
        "FC-4: model must re-freeze normally when only the market is NaN"
    assert float(fc_limbo.loc[fc_limbo["match_key"] == key_fc, "p_home_book"].iloc[0]) == 0.55, \
        "FC-4: a NaN incoming book must NOT null the good frozen book (limbo guard)"
    assert float(fc_limbo.loc[fc_limbo["match_key"] == key_fc, "p_home_poly"].iloc[0]) == 0.47, \
        "FC-4: a NaN incoming poly must NOT null the good frozen poly (limbo guard)"

    # --- FC-5: detector finds a played-but-never-frozen match ---
    detector_played = pd.DataFrame({
        "date": [pd.Timestamp(TODAY), pd.Timestamp(TODAY)],
        "home_team": ["Kappa", "Mu"], "away_team": ["Lambda", "Nu"],
        "home_score": [2, 1], "away_score": [0, 1],
        "tournament": ["FIFA World Cup", "FIFA World Cup"],
    })
    # fc2 has only Kappa-Lambda frozen; Mu-Nu is played-but-never-frozen.
    missing = find_unfrozen_played(fc2, detector_played)
    key_munu = make_match_key(TODAY.isoformat(), "Mu", "Nu")
    assert missing == [key_munu], f"detector should flag Mu-Nu only, got {missing}"
    # When every played match has a frozen row, the detector is silent.
    assert find_unfrozen_played(fc_locked, fc_played) == [], \
        "detector must be empty when the played match has a ledger row"

    print("update_ledger.py self-tests passed")
