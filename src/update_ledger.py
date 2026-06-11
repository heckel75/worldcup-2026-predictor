"""
src/update_ledger.py — prediction ledger management (Session 29).

Maintains data/processed/wc_predictions.csv: a frozen record of the model's
published probabilities for each WC match, plus the actual result once played.

Schema (matches calibration.py's CANON, which selects by column name):
    match_key, date, home_team, away_team,
    p_home, p_draw, p_away,
    forecast_ts, outcome          # outcome in {H, D, A} or "" when unplayed

Public API:
    freeze_new_forecasts(ledger_df, upcoming_fixtures_df, triple_df, today,
                         lookahead_days) -> pd.DataFrame
    attach_results(ledger_df, played_df) -> pd.DataFrame

No datetime.now() inside the core functions — today is always passed in so
self-tests can control the date window without patching.
"""
from __future__ import annotations

import datetime as dt
import re

import pandas as pd

LEDGER_SCHEMA = [
    "match_key", "date", "home_team", "away_team",
    "p_home", "p_draw", "p_away",
    "forecast_ts", "outcome",
]


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

def freeze_new_forecasts(
    ledger_df: pd.DataFrame,
    upcoming_fixtures_df: pd.DataFrame,
    triple_df: pd.DataFrame,
    today: dt.date,
    lookahead_days: int,
) -> pd.DataFrame:
    """
    Append one ledger row per fixture whose date is within [today, today +
    lookahead_days] and whose match_key is not already in the ledger.

    Probabilities come from triple_df's bias-corrected model columns
    (p_home_model_corr, p_draw_model_corr, p_away_model_corr).

    Never modifies existing rows.
    """
    # Existing keys — guard against empty or missing column
    if "match_key" in ledger_df.columns and len(ledger_df):
        existing_keys: set[str] = set(ledger_df["match_key"].dropna())
    else:
        existing_keys = set()

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

    # Join corrected probs from triple_df on (home_team, away_team)
    triple = triple_df[
        ["home_team", "away_team",
         "p_home_model_corr", "p_draw_model_corr", "p_away_model_corr"]
    ].copy()
    merged = window.merge(triple, on=["home_team", "away_team"], how="left")

    now_ts = dt.datetime.now().isoformat(timespec="seconds")
    new_rows = []
    for _, row in merged.iterrows():
        date_iso = row["date"].date().isoformat()
        key = make_match_key(date_iso, row["home_team"], row["away_team"])
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_rows.append({
            "match_key":   key,
            "date":        date_iso,
            "home_team":   row["home_team"],
            "away_team":   row["away_team"],
            "p_home":      round(float(row["p_home_model_corr"]), 4),
            "p_draw":      round(float(row["p_draw_model_corr"]), 4),
            "p_away":      round(float(row["p_away_model_corr"]), 4),
            "forecast_ts": now_ts,
            "outcome":     "",
        })

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
    appears in played_df. If yes, set outcome to "H", "D", or "A".

    Never changes p_home / p_draw / p_away / forecast_ts.
    A match that was played but never frozen is not back-filled.
    """
    if ledger_df.empty:
        return ledger_df

    # Build a lookup: match_key -> outcome string from played results
    played = played_df.copy()
    played["date"] = pd.to_datetime(played["date"])

    def _outcome(row) -> str:
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

    ledger = ledger_df.copy()
    unscored_mask = ledger["outcome"].isna() | (ledger["outcome"] == "")
    for idx in ledger[unscored_mask].index:
        key = ledger.at[idx, "match_key"]
        if key in result_map:
            ledger.at[idx, "outcome"] = result_map[key]

    return ledger


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

    # Synthetic triple_df with corrected probs
    triple_df = pd.DataFrame({
        "home_team":           [t[0] for t in _teams],
        "away_team":           [t[1] for t in _teams],
        "p_home_model_corr":   [0.50, 0.30, 0.45],
        "p_draw_model_corr":   [0.25, 0.35, 0.28],
        "p_away_model_corr":   [0.25, 0.35, 0.27],
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

    # --- Test 2: re-running freeze is a no-op ---
    ledger2 = freeze_new_forecasts(
        ledger, fixtures_df, triple_df, TODAY, lookahead_days=1
    )
    assert len(ledger2) == 2, "Re-freeze should not add duplicate rows"

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
    ledger3 = attach_results(ledger, played_df)
    outcome_a = ledger3.loc[ledger3["match_key"] == key_a, "outcome"].iloc[0]
    assert outcome_a == "H", f"Alpha 2–1 Beta should be 'H', got '{outcome_a}'"
    p_home_after = float(ledger3.loc[ledger3["match_key"] == key_a, "p_home"].iloc[0])
    assert p_home_before == p_home_after, "p_home must not change after attach_results"
    outcome_b = ledger3.loc[ledger3["match_key"] == key_b, "outcome"].iloc[0]
    assert outcome_b == "", "Match B not yet played — outcome should remain empty"

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

    print("update_ledger.py self-tests passed")
