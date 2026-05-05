"""Verify the elo_seeds_2018.csv covers the right teams.

Two checks:
  1. Every seed team should exist in data/processed/matches_clean.csv.
     If not, it's likely a name mismatch (the seed is wasted).
  2. Every WC qualifier in data/processed/elo_ratings_2026.csv should have
     a seed. Those without will start at 1500 — fine for minnows, bad for
     anyone Brazil-tier.

Run from project root:  python src/verify_seeds.py
"""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "data" / "raw" / "elo_seeds_2018.csv"
MATCHES = ROOT / "data" / "processed" / "matches_clean.csv"
WC_RATINGS = ROOT / "data" / "processed" / "elo_ratings_2026.csv"


def load_seed_teams() -> set[str]:
    teams = set()
    with SEEDS.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            teams.add(row["team"])
    return teams


def load_match_teams() -> set[str]:
    """Every team that appears as home or away in matches_clean.csv."""
    teams = set()
    with MATCHES.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            teams.add(row["home_team"])
            teams.add(row["away_team"])
    return teams


def load_wc_teams() -> set[str]:
    teams = set()
    with WC_RATINGS.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            teams.add(row["team"])
    return teams


def main():
    seeds = load_seed_teams()
    matches = load_match_teams()
    wc = load_wc_teams()

    print(f"Seed file:       {len(seeds)} teams")
    print(f"matches_clean:   {len(matches)} unique teams")
    print(f"WC qualifiers:   {len(wc)} teams")
    print()

    # Check 1: seeds that don't appear in match data
    orphan_seeds = sorted(seeds - matches)
    print(f"=== Check 1: seed teams NOT found in matches_clean.csv ===")
    if not orphan_seeds:
        print("  None. All seeds will be applied.")
    else:
        print(f"  {len(orphan_seeds)} orphan seed(s) — these names don't match our match data:")
        for t in orphan_seeds:
            print(f"    - {t!r}")
        print("  Likely cause: name mismatch. These seeds are wasted unless we fix the name.")
    print()

    # Check 2: WC qualifiers without a seed
    unseeded_wc = sorted(wc - seeds)
    print(f"=== Check 2: WC qualifiers WITHOUT a seed ===")
    if not unseeded_wc:
        print("  None. Every WC team gets a seeded starting rating.")
    else:
        print(f"  {len(unseeded_wc)} WC qualifier(s) will default to 1500:")
        for t in unseeded_wc:
            print(f"    - {t!r}")
        print("  This is fine for minnows. Worry only if a strong team is here.")
    print()

    # Bonus: closest-name suggestions for orphans (cheap edit-distance heuristic)
    if orphan_seeds:
        print("=== Suggested fuzzy matches for orphan seeds ===")
        for orphan in orphan_seeds:
            candidates = _closest(orphan, matches, n=3)
            print(f"  {orphan!r} -> {candidates}")


def _closest(query: str, pool: set[str], n: int = 3) -> list[str]:
    """Cheap shared-substring score, no external deps."""
    q = query.lower()
    scored = []
    for cand in pool:
        c = cand.lower()
        # Score = length of longest common prefix + bonus if q is substring of c or vice versa
        score = 0
        for a, b in zip(q, c):
            if a == b:
                score += 1
            else:
                break
        if q in c or c in q:
            score += 5
        scored.append((score, cand))
    scored.sort(reverse=True)
    return [c for s, c in scored[:n] if s > 0]


if __name__ == "__main__":
    main()
