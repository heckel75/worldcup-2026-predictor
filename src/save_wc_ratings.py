"""
Run the historical Elo pass and save WC 2026 team ratings.

Replays all played matches in matches_clean.csv to build current ratings,
then dumps the 48 World Cup qualifiers to data/processed/elo_ratings_2026.csv.

Run from project root:
    python src/save_wc_ratings.py
"""

import pandas as pd
from elo import EloSystem


def main():
    # 1. Load played matches and rebuild Elo
    matches = pd.read_csv("data/processed/matches_clean.csv")
    print(f"Loaded {len(matches)} played matches")

    elo = EloSystem()
    for row in matches.itertuples(index=False):
        elo.update_match(
            home_team=row.home_team,
            away_team=row.away_team,
            home_score=row.home_score,
            away_score=row.away_score,
            tournament=row.tournament,
            neutral=bool(row.neutral),
        )
    print(f"Total teams rated: {len(elo.ratings)}")

    # 2. Pull the 48 WC team names from the fixtures file
    fixtures = pd.read_csv("data/processed/fixtures_2026.csv")
    wc_teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    print(f"WC teams in fixtures: {len(wc_teams)}")

    # 3. Build rating table, sorted high to low
    rows = [(t, round(elo.get_rating(t), 1)) for t in wc_teams]
    df = pd.DataFrame(rows, columns=["team", "elo"]).sort_values(
        "elo", ascending=False
    ).reset_index(drop=True)

    # 4. Save and print
    out_path = "data/processed/elo_ratings_2026.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()