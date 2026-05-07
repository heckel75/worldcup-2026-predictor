"""
Replay history and log pre-match Elo + actual result for every match.

Output: data/processed/match_elo_log.csv
Used by Session 9 to fit the (Elo difference -> expected goals) mapping.

Run from project root:
    python src/log_match_elo.py
"""

import pandas as pd
from elo import EloSystem, HOME_ADVANTAGE
from save_wc_ratings import load_seeds


def main():
    seeds = load_seeds()
    matches = pd.read_csv("data/processed/matches_clean.csv")
    print(f"Replaying {len(matches)} matches with {len(seeds)} seeded teams")

    elo = EloSystem(seed_ratings=seeds)
    rows = []

    for m in matches.itertuples(index=False):
        # Snapshot ratings BEFORE the match
        elo_h_pre = elo.get_rating(m.home_team)
        elo_a_pre = elo.get_rating(m.away_team)
        neutral = bool(m.neutral)

        # Effective Elo difference, folding in home advantage
        eff_diff = elo_h_pre - elo_a_pre + (0 if neutral else HOME_ADVANTAGE)

        rows.append({
            "date": m.date,
            "home_team": m.home_team,
            "away_team": m.away_team,
            "neutral": neutral,
            "elo_h_pre": elo_h_pre,
            "elo_a_pre": elo_a_pre,
            "eff_diff": eff_diff,
            "home_score": m.home_score,
            "away_score": m.away_score,
            "goal_diff": m.home_score - m.away_score,   # signed, home perspective
            "total_goals": m.home_score + m.away_score,
        })

        # Now update Elo with the actual result (same as before)
        elo.update_match(
            home_team=m.home_team,
            away_team=m.away_team,
            home_score=m.home_score,
            away_score=m.away_score,
            tournament=m.tournament,
            neutral=neutral,
        )

    df = pd.DataFrame(rows)
    out_path = "data/processed/match_elo_log.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} rows")
    print("\nSummary:")
    print(df[["eff_diff", "goal_diff", "total_goals"]].describe().round(2))


if __name__ == "__main__":
    main()