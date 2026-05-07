"""
Session 9, Step 2: look at how actual goal difference and total goals
relate to pre-match Elo difference.

Produces:
  - printed bin table
  - plot saved to data/processed/goals_vs_elo.png

Run from project root:
    python src/explore_goals_vs_elo.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("data/processed/match_elo_log.csv")
    print(f"Loaded {len(df)} matches")

    # Bin by effective Elo difference. Symmetric bins so we can eyeball
    # whether the relationship looks linear and roughly symmetric.
    bins = [-1500, -600, -400, -250, -150, -75, -25, 25, 75, 150, 250, 400, 600, 1500]
    df["bin"] = pd.cut(df["eff_diff"], bins=bins)

    summary = df.groupby("bin", observed=True).agg(
        n=("goal_diff", "size"),
        mean_diff=("goal_diff", "mean"),
        mean_total=("total_goals", "mean"),
        mid=("eff_diff", "mean"),
    ).round(3)
    print("\nBin summary:")
    print(summary.to_string())

    # Two-panel plot: goal diff vs Elo diff, and total goals vs Elo diff.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.scatter(summary["mid"], summary["mean_diff"], s=summary["n"] / 5)
    ax1.axhline(0, color="grey", lw=0.5)
    ax1.axvline(0, color="grey", lw=0.5)
    ax1.set_xlabel("Effective Elo difference (home − away, +60 if non-neutral)")
    ax1.set_ylabel("Mean actual goal difference")
    ax1.set_title("Goal difference vs Elo difference")

    ax2.scatter(summary["mid"], summary["mean_total"], s=summary["n"] / 5)
    ax2.set_xlabel("Effective Elo difference")
    ax2.set_ylabel("Mean total goals")
    ax2.set_title("Total goals vs Elo difference")

    plt.tight_layout()
    out = "data/processed/goals_vs_elo.png"
    plt.savefig(out, dpi=120)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()