"""
src/backfill_ledger_markets.py — one-time market-column backfill (Session 36).

Ledger rows frozen before Session 36 carry only the model probs; this fills
their frozen market columns (book/Polymarket probs, volume, neutral flag)
from the divergence snapshot dated the day before the match — the freeze day,
when triple_compare last saw the fixture — falling back to the nearest
earlier snapshot if that file is absent.

Idempotent: a row with any market column already populated is never touched;
model probs, outcome, actual scores, and forecast_ts are never touched.
Prints every backfilled row.

Run from project root: python src/backfill_ledger_markets.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from update_ledger import LEDGER_SCHEMA, MARKET_COLS, _ensure_schema

LEDGER_PATH = Path("data/processed/wc_predictions.csv")
SNAPS_DIR = Path("data/processed/divergence_snapshots")


def _snapshot_for(match_date: dt.date) -> Path | None:
    """Snapshot dated match_date - 1 (the freeze day), else nearest earlier."""
    freeze_day = match_date - dt.timedelta(days=1)
    candidates = sorted(SNAPS_DIR.glob("*.csv"))
    eligible = [p for p in candidates if p.stem <= freeze_day.isoformat()]
    return eligible[-1] if eligible else None


def main() -> None:
    ledger = _ensure_schema(pd.read_csv(LEDGER_PATH))

    snap_cache: dict[Path, pd.DataFrame] = {}
    backfilled = 0
    for idx, row in ledger.iterrows():
        if any(pd.notna(row[c]) for c in MARKET_COLS):
            continue  # already populated — never touch

        match_date = dt.date.fromisoformat(str(row["date"]))
        snap_path = _snapshot_for(match_date)
        if snap_path is None:
            print(f"  WARNING: no snapshot on or before {match_date} for {row['match_key']} — skipped")
            continue

        if snap_path not in snap_cache:
            snap_cache[snap_path] = pd.read_csv(snap_path)
        snap = snap_cache[snap_path]
        hit = snap[(snap["home_team"] == row["home_team"])
                   & (snap["away_team"] == row["away_team"])]
        if hit.empty:
            print(f"  WARNING: {row['match_key']} not in {snap_path.name} — skipped")
            continue

        src = hit.iloc[0]
        for col in MARKET_COLS:
            val = src[col]
            if pd.isna(val):
                continue  # absent market at freeze time stays NaN
            ledger.at[idx, col] = round(float(val), 4) if col.startswith("p_") else val
        backfilled += 1
        print(f"  {row['match_key']}  <- {snap_path.name}  "
              f"book {src['p_home_book']}/{src['p_draw_book']}/{src['p_away_book']}  "
              f"poly {src['p_home_poly']}/{src['p_draw_poly']}/{src['p_away_poly']}  "
              f"vol {src['poly_volume']}  neutral {src['neutral_used']}")

    if backfilled:
        ledger.to_csv(LEDGER_PATH, index=False)
        print(f"Backfilled {backfilled} row(s) -> {LEDGER_PATH}")
    else:
        print("Nothing to backfill — all rows already carry market columns.")


if __name__ == "__main__":
    main()
