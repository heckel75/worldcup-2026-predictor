"""
src/clean_data.py
Clean and filter the international football results dataset.
Splits played matches (training data) from future fixtures (predictions).
"""

import pandas as pd
from pathlib import Path

# --- config ---
RAW_PATH = Path("data/raw/results.csv")
PROCESSED_DIR = Path("data/processed")
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

    # 1. Load
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df):,} rows from {RAW_PATH}")

    # 2. Parse dates
    df["date"] = pd.to_datetime(df["date"])

    # 3. Filter by date
    df = df[df["date"] >= START_DATE].copy()
    print(f"After {START_DATE} filter: {len(df):,} rows")

    # 4. Standardize team names (both home and away columns)
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)

    # 5. Split played vs unplayed.
    # Future fixtures have NaN in score columns.
    played = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    fixtures = df[df["home_score"].isna() | df["away_score"].isna()].copy()

    # Scores come in as float because of the NaNs in the original CSV. Convert.
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)

    # 6. Save
    played.to_csv(TRAINING_PATH, index=False)
    fixtures.to_csv(FIXTURES_PATH, index=False)

    # 7. Sanity report — eyeball this output
    print(f"\nTraining set:  {len(played):,} matches → {TRAINING_PATH}")
    print(f"Fixtures set:  {len(fixtures):,} matches → {FIXTURES_PATH}")
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