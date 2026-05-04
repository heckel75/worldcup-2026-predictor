"""Run the full historical pass and sanity-check the ratings."""

import pandas as pd
from elo import EloSystem


def main():
    df = pd.read_csv("data/processed/matches_clean.csv")
    df = df.sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(df)} matches")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    elo = EloSystem()

    for _, row in df.iterrows():
        elo.update_match(
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_score=int(row["home_score"]),
            away_score=int(row["away_score"]),
            tournament=row["tournament"],
            neutral=row['neutral'], 
        )

    print(f"\nTotal teams rated: {len(elo.ratings)}")
    print("\nTop 20 teams by Elo:")
    for i, (team, rating) in enumerate(elo.top_n(20), 1):
        print(f"  {i:2}. {team:<25} {rating:.0f}")


if __name__ == "__main__":
    main()