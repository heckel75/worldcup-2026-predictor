"""
Session 15: Monte Carlo aggregation.

Wraps simulate_tournament() in an N-iteration loop, tallies per-team
furthest-round outcomes, and saves the result as a dated snapshot for
day-over-day diffing.

Stage probabilities per team:
  p_advance   — top 2 in group OR one of 8 best thirds (reach R32)
  p_r16       — won R32 (i.e. reached R16)
  p_qf        — reached QF
  p_sf        — reached SF
  p_final     — reached the final
  p_champion  — won the cup

These are MONOTONIC non-increasing per team by construction:
  p_advance >= p_r16 >= p_qf >= p_sf >= p_final >= p_champion.

Quirk inherited from simulate.py: the label "SF" never appears in
team_furthest_round, because both semifinal losers play the third-place
playoff and get reclassified as "third_place" / "fourth_place". So
"reached the SF" = {fourth_place, third_place, runner_up, winner}.
We handle this in STAGE_BY_LABEL below.

Run from project root:
    python src/monte_carlo.py
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bracket import TEAM_TO_GROUP
from simulate import simulate_tournament


# Integer stage levels: 0 = didn't advance, 6 = won the cup.
# Used to map team_furthest_round labels onto cumulative reach.
STAGE_BY_LABEL: dict[str, int] = {
    "group_stage":   0,
    "R32":           1,
    "R16":           2,
    "QF":            3,
    "SF":            4,   # never actually emitted; included for safety
    "fourth_place":  4,
    "third_place":   4,
    "runner_up":     5,
    "winner":        6,
}

# (column name, minimum stage level that counts for this probability)
STAGE_COLS: list[tuple[str, int]] = [
    ("p_advance",  1),
    ("p_r16",      2),
    ("p_qf",       3),
    ("p_sf",       4),
    ("p_final",    5),
    ("p_champion", 6),
]


def run_monte_carlo(
    ratings: dict[str, float],
    fixtures: list[dict],
    n_sims: int = 10_000,
    seed: Optional[int] = None,
    progress_every: int = 1000,
) -> pd.DataFrame:
    """Run n_sims simulated tournaments and return per-team probabilities.

    Parameters
    ----------
    ratings : dict[team -> Elo]
    fixtures : list[{"home_team", "away_team"}]
    n_sims : int
        Number of tournament simulations. 10,000 gives ~0.4pp standard
        error at p=0.20 and ~0.2pp at p=0.05.
    seed : int, optional
        Master RNG seed for reproducibility. The whole 10k sequence is
        deterministic given the seed.
    progress_every : int
        Print a progress line every K sims. 0 to silence.

    Returns
    -------
    DataFrame, one row per team, columns:
      team, p_advance, p_r16, p_qf, p_sf, p_final, p_champion
    sorted by p_champion descending.
    """
    rng = np.random.default_rng(seed)
    teams = sorted(TEAM_TO_GROUP.keys())

    # exact[team] = length-7 vector counting sims where team's
    # furthest stage was *exactly* level k.
    exact = {t: np.zeros(7, dtype=np.int64) for t in teams}

    t0 = time.time()
    for i in range(1, n_sims + 1):
        result = simulate_tournament(ratings, fixtures, rng)
        for team, label in result["team_furthest_round"].items():
            exact[team][STAGE_BY_LABEL[label]] += 1
        if progress_every and i % progress_every == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (n_sims - i) / rate
            print(
                f"  ... {i:>6} / {n_sims}   "
                f"{rate:5.1f} sims/s   "
                f"ETA {eta:5.0f}s"
            )

    # Cumulative-from-the-right: reach[k] = # sims with stage >= k.
    rows = []
    for team in teams:
        e = exact[team]
        reach = e[::-1].cumsum()[::-1]
        row = {"team": team}
        for col, level in STAGE_COLS:
            row[col] = reach[level] / n_sims
        rows.append(row)

    df = (
        pd.DataFrame(rows)
        .sort_values("p_champion", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    return df


def save_snapshot(
    df: pd.DataFrame,
    snapshots_dir: str | Path = "data/processed/snapshots",
    date: Optional[dt.date] = None,
) -> Path:
    """Save a dated snapshot CSV. Overwrites if same date already exists
    (intentional — re-running for the same day reproduces the same numbers
    because the seed is date-derived)."""
    date = date or dt.date.today()
    snap_dir = Path(snapshots_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    out_path = snap_dir / f"{date.isoformat()}.csv"

    df_out = df.copy()
    df_out.insert(0, "snapshot_date", date.isoformat())
    df_out.to_csv(out_path, index=False)
    return out_path


# ----------------------------------------------------------------------
# Sanity checks on a Monte Carlo dataframe
# ----------------------------------------------------------------------

# Each round has a known number of teams in it across every sim.
# These sums are EXACT each sim, so the column sum is also exact.
EXPECTED_SUMS: dict[str, int] = {
    "p_advance":  32,
    "p_r16":      16,
    "p_qf":        8,
    "p_sf":        4,
    "p_final":     2,
    "p_champion":  1,
}


def _sanity_check(df: pd.DataFrame) -> None:
    """Validate per-team monotonicity and per-round expected sums."""
    prob_cols = [c for c, _ in STAGE_COLS]

    for _, row in df.iterrows():
        probs = [row[c] for c in prob_cols]
        for a, b in zip(probs, probs[1:]):
            assert a + 1e-12 >= b, (
                f"non-monotonic for {row['team']}: "
                f"{dict(zip(prob_cols, probs))}"
            )

    for col, exp in EXPECTED_SUMS.items():
        got = df[col].sum()
        assert abs(got - exp) < 1e-9, (
            f"{col} sums to {got}, expected exactly {exp}"
        )

    print("✅ Monte Carlo sanity checks pass.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _load_ratings() -> dict[str, float]:
    df = pd.read_csv("data/processed/elo_ratings_2026.csv")
    return dict(zip(df["team"], df["elo"]))


def _load_fixtures() -> list[dict]:
    df = pd.read_csv("data/processed/fixtures_2026.csv")
    return df[["home_team", "away_team"]].to_dict("records")


def main() -> None:
    today = dt.date.today()
    seed = int(today.strftime("%Y%m%d"))
    n_sims = 10_000

    print("World Cup 2026 — Monte Carlo forecast")
    print(f"Date       : {today.isoformat()}")
    print(f"Seed       : {seed} (YYYYMMDD)")
    print(f"Simulations: {n_sims:,}")
    print()

    ratings = _load_ratings()
    fixtures = _load_fixtures()
    print(f"Loaded {len(ratings)} team ratings and {len(fixtures)} fixtures.")
    print()

    df = run_monte_carlo(ratings, fixtures, n_sims=n_sims, seed=seed)

    _sanity_check(df)

    out = save_snapshot(df, date=today)
    print(f"\nSnapshot saved: {out}")

    print("\n" + "=" * 70)
    print("TOP 10 TITLE CONTENDERS")
    print("=" * 70)
    for _, r in df.head(10).iterrows():
        print(
            f"  {r['team']:<22}  champ={r['p_champion']:6.1%}  "
            f"final={r['p_final']:6.1%}  qf={r['p_qf']:6.1%}  "
            f"adv={r['p_advance']:6.1%}"
        )


if __name__ == "__main__":
    main()