"""
src/clean_data.py
Clean and filter the international football results dataset.
Splits played matches (training data) from future fixtures (predictions).
"""

import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clock import clean_output_path as _clean_output_path

# --- config ---
RAW_PATH    = Path("data/raw/results.csv")
# WC_MANUAL_RESULTS lets the dry-run harness point at a temp feed
# without touching the real file.
MANUAL_PATH = Path(os.environ.get("WC_MANUAL_RESULTS",
                                  "data/raw/wc_results_manual.csv"))
PROCESSED_DIR = Path("data/processed")
# TRAINING_PATH is the default; actual write path is resolved at runtime
# via _clock.clean_output_path() so dry-run replays redirect to a temp file.
TRAINING_PATH = PROCESSED_DIR / "matches_clean.csv"
FIXTURES_PATH = PROCESSED_DIR / "fixtures_2026.csv"
START_DATE = "2018-01-01"

# Common name variants we want to collapse to one canonical form.
# Add more here if the sanity-check print at the end reveals others.
TEAM_NAME_MAP = {
    "United States": "USA",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Republic of Ireland": "Ireland",
}


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    training_path = Path(_clean_output_path())
    training_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Kaggle dataset
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df):,} rows from {RAW_PATH}")

    # 1b. Concat hand-maintained WC results (manual rows first so they win dedup).
    # Always concat even when the feed has zero data rows: the manual CSV schema
    # carries the 'advanced' column that historical results.csv lacks, so the
    # concat is how that column enters df (filled with NaN for historical rows).
    if MANUAL_PATH.exists():
        manual_df = pd.read_csv(MANUAL_PATH)
        df = pd.concat([manual_df, df], ignore_index=True)
        if len(manual_df) > 0:
            print(f"Prepended {len(manual_df):,} manual rows from {MANUAL_PATH}")

    # 2. Parse dates
    df["date"] = pd.to_datetime(df["date"])

    # 3. Filter by date
    df = df[df["date"] >= START_DATE].copy()
    print(f"After {START_DATE} filter: {len(df):,} rows")

    # 4. Standardize team names (both home and away columns)
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)

    # 4b. Dedup: keep the first occurrence (manual row) when both sources have the match
    before = len(df)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} duplicate row(s) (manual source takes precedence)")

    # 5. Split played vs unplayed.
    # Future fixtures have NaN in score columns.
    played = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    fixtures = df[df["home_score"].isna() | df["away_score"].isna()].copy()

    # Scores come in as float because of the NaNs in the original CSV. Convert.
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)

    # 6. Schema guard: every column in the manual feed must survive to output.
    # Catches future manual-feed columns (referee, venue, …) that get silently
    # dropped by a future clean_data change before they reach matches_clean.csv.
    if MANUAL_PATH.exists():
        missing = [c for c in manual_df.columns if c not in played.columns]
        if missing:
            raise ValueError(
                f"Column(s) from {MANUAL_PATH} were dropped during processing: "
                f"{missing}. Update clean_data.py to carry them through explicitly."
            )

    # 7. Save
    played.to_csv(training_path, index=False)
    fixtures.to_csv(FIXTURES_PATH, index=False)

    # 8. Sanity report — eyeball this output
    print(f"\nTraining set:  {len(played):,} matches -> {training_path}")
    print(f"Fixtures set:  {len(fixtures):,} matches -> {FIXTURES_PATH}")
    print(f"\nDate range (training): {played['date'].min().date()} to {played['date'].max().date()}")
    if len(fixtures) > 0:
        print(f"Date range (fixtures): {fixtures['date'].min().date()} to {fixtures['date'].max().date()}")

    print(f"\nTop tournament types in training set:")
    print(played["tournament"].value_counts().head(8))

    # The big one: list every unique team appearing in the 2026 WC fixtures.
    # If anything here looks weird ("Korea Republic", "USMNT", etc.) add it
    # to TEAM_NAME_MAP and re-run.
    wc_fixtures = fixtures[fixtures["tournament"] == "FIFA World Cup"]
    wc_teams = sorted(set(wc_fixtures["home_team"]) | set(wc_fixtures["away_team"]))
    print(f"\nTeams in 2026 WC fixtures ({len(wc_teams)}):")
    for t in wc_teams:
        print(f"  {t}")


if __name__ == "__main__":
    main()