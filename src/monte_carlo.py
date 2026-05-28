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
from clock import today as _clock_today
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
    known_results: Optional[dict] = None,
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
    known_results : dict, optional
        Pinned results to pass through to simulate_tournament on every
        iteration. Group results keyed by (home, away) → (hg, ag);
        knockout results keyed by frozenset({a, b}) → winner string.
        None means every match is sampled (pre-tournament behaviour).

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
        result = simulate_tournament(
            ratings, fixtures, rng, known_results=known_results
        )
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
    date = date or _clock_today()
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

    print("[OK] Monte Carlo sanity checks pass.")


# ----------------------------------------------------------------------
# Helper: build known_results from a played-matches DataFrame
# ----------------------------------------------------------------------

def build_known_results(
    played_wc_df: pd.DataFrame,
    group_pairs: set[tuple[str, str]],
) -> dict:
    """Build a known_results dict from played WC matches for simulate_tournament.

    Parameters
    ----------
    played_wc_df : DataFrame
        Rows from matches_clean.csv filtered to FIFA World Cup rows with
        scores present. Expected columns: home_team, away_team, home_score,
        away_score, and optionally advanced (for KO penalty decisions).
    group_pairs : set of (home, away) tuples
        The 72 group-stage fixture pairs from fixtures_2026.csv. Used to
        partition group results from knockout results.

    Returns
    -------
    dict
        Group results: (home, away) → (hg, ag)
        KO results:    frozenset({a, b}) → winner string
    """
    known: dict = {}
    for _, row in played_wc_df.iterrows():
        h, a = str(row["home_team"]), str(row["away_team"])
        hg, ag = int(row["home_score"]), int(row["away_score"])
        if (h, a) in group_pairs:
            known[(h, a)] = (hg, ag)
        elif (a, h) in group_pairs:
            raise ValueError(
                f"Group match {h!r} vs {a!r} in the feed is in the wrong "
                f"orientation — the fixture is registered as {a!r} (home) vs "
                f"{h!r} (away). Swap home_team/away_team in "
                f"wc_results_manual.csv."
            )
        else:
            # Knockout result
            if hg > ag:
                winner = h
            elif ag > hg:
                winner = a
            else:
                adv = row.get("advanced", None)
                if pd.isna(adv) or not str(adv).strip():
                    raise ValueError(
                        f"KO match {h} vs {a} ended {hg}–{ag} (draw after 90') "
                        f"but 'advanced' column is missing or empty in the feed. "
                        f"Add the team that advanced (won on penalties) to the "
                        f"'advanced' column in wc_results_manual.csv."
                    )
                winner = str(adv).strip()
            known[frozenset({h, a})] = winner
    return known


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _load_ratings() -> dict[str, float]:
    df = pd.read_csv("data/processed/elo_ratings_2026.csv")
    return dict(zip(df["team"], df["elo"]))


def _load_fixtures() -> list[dict]:
    """Return ALL 72 WC 2026 group-stage fixtures (played + unplayed).

    simulate_tournament needs the full fixture list even for groups that are
    already complete — if a group's matches aren't in the fixture list, the
    simulation produces 0-pt records for every team and ranking becomes
    arbitrary (invariant B failure once results are pinned via known_results).
    """
    # Unplayed fixtures still in fixtures_2026.csv
    unplayed_df = pd.read_csv("data/processed/fixtures_2026.csv")
    unplayed = unplayed_df[["home_team", "away_team"]].to_dict("records")

    # Played WC 2026 group-stage fixtures from matches_clean.csv
    played_df = pd.read_csv("data/processed/matches_clean.csv", parse_dates=["date"])
    wc_2026 = played_df[
        (played_df["tournament"] == "FIFA World Cup")
        & (played_df["date"] >= "2026-06-01")
        & played_df["home_score"].notna()
    ]
    played_group = [
        {"home_team": str(r["home_team"]), "away_team": str(r["away_team"])}
        for _, r in wc_2026.iterrows()
        if TEAM_TO_GROUP.get(r["home_team"]) == TEAM_TO_GROUP.get(r["away_team"])
        and TEAM_TO_GROUP.get(r["home_team"]) is not None
    ]

    # played first so known_results always wins; no duplicates since
    # clean_data.py moves played rows out of fixtures_2026.csv.
    return played_group + unplayed


def _load_played_wc(fixtures: list[dict]) -> dict:
    """Load played WC 2026 matches from matches_clean.csv and build known_results.

    group_pairs is built from two sources so it doesn't shrink as matches
    are played (clean_data.py moves played rows out of fixtures_2026.csv):
      1. currently-unplayed fixtures (from fixtures_2026.csv)
      2. any played WC 2026 match where both teams are in the same GROUPS
         letter (those were group-stage matches, just no longer in fixtures)
    """
    played_df = pd.read_csv("data/processed/matches_clean.csv", parse_dates=["date"])
    # Restrict to 2026 WC matches only: historical WC KO results (2018, 2022)
    # have draw scorelines that went to penalties but no 'advanced' column.
    wc_played = played_df[
        (played_df["tournament"] == "FIFA World Cup")
        & played_df["home_score"].notna()
        & played_df["away_score"].notna()
        & (played_df["date"] >= "2026-06-01")
    ].copy()

    if wc_played.empty:
        return {}

    # Start with currently-unplayed fixture pairs.
    group_pairs: set[tuple[str, str]] = {
        (fx["home_team"], fx["away_team"]) for fx in fixtures
    }
    # Augment with played group matches (same group = group stage, not KO).
    for _, row in wc_played.iterrows():
        h, a = str(row["home_team"]), str(row["away_team"])
        if TEAM_TO_GROUP.get(h) == TEAM_TO_GROUP.get(a) and TEAM_TO_GROUP.get(h):
            group_pairs.add((h, a))

    return build_known_results(wc_played, group_pairs)


def main() -> None:
    today = _clock_today()
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

    known_results = _load_played_wc(fixtures)
    n_group = sum(1 for k in known_results if isinstance(k, tuple))
    n_ko    = sum(1 for k in known_results if isinstance(k, frozenset))
    if known_results:
        print(f"Known results: {n_group} group match(es), {n_ko} knockout match(es) pinned.")
    else:
        print("No played WC results found — full simulation (pre-tournament mode).")
    print()

    df = run_monte_carlo(
        ratings, fixtures, n_sims=n_sims, seed=seed, known_results=known_results or None
    )

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
