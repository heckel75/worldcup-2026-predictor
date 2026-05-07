"""Session 9: spot-check elo_to_lambdas on canonical matchups."""

import pandas as pd
from strengths import elo_to_lambdas


def main():
    df = pd.read_csv("data/processed/elo_ratings_2026.csv")
    elo = dict(zip(df["team"], df["elo"]))

    print(f"{'Matchup':<40} {'λ_h':>5} {'λ_a':>5}  notes")
    print("-" * 75)

    cases = [
        ("Spain",   "Argentina",   True),
        ("Brazil",  "Germany",     True),
        ("France",  "England",     True),
        ("USA",     "Mexico",      True),
        ("France",  "Haiti",       True),
        ("England", "New Zealand", True),
    ]

    for h, a, neutral in cases:
        if h not in elo or a not in elo:
            print(f"  (skipping {h} vs {a} — team not in WC ratings file)")
            continue
        lh, la = elo_to_lambdas(elo[h], elo[a], neutral)
        print(f"{h+' vs '+a:<40} {lh:>5.2f} {la:>5.2f}  {notes_for(lh, la)}")

    # Pure-synthetic checks
    print("\nSynthetic (independent of WC ratings file):")
    for label, eh, ea, neutral in [
        ("equal teams, neutral",      2000, 2000, True),
        ("equal teams, home for H",   2000, 2000, False),
        ("100 Elo edge, neutral",     2050, 1950, True),
        ("500 Elo edge, neutral",     2000, 1500, True),
        ("1000 Elo edge, neutral",    2000, 1000, True),
    ]:
        lh, la = elo_to_lambdas(eh, ea, neutral)
        print(f"  {label:<32} -> λ_h={lh:.2f}  λ_a={la:.2f}  total={lh+la:.2f}  diff={lh-la:.2f}")


def notes_for(lh, la):
    total = lh + la
    diff = lh - la
    if abs(diff) < 0.3:
        shape = "even"
    elif abs(diff) < 1.0:
        shape = "slight edge"
    elif abs(diff) < 2.5:
        shape = "clear favorite"
    else:
        shape = "blowout"
    return f"total≈{total:.1f}, {shape}"


if __name__ == "__main__":
    main()