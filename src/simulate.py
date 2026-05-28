"""
Simulate one full World Cup 2026 tournament.

Session 14: plays all 104 matches given current ratings + fixtures, and
returns a dict describing the entire bracket. Session 15 wraps this in a
10,000-iteration Monte Carlo loop.

Session 16: replaced the backtracking third-place slot placeholder with
FIFA's published Annex C lookup (the 495-row table mapping "which 8 of
the 12 thirds advance" → "which R32 slot each one fills"). The Annex C
data lives in data/raw/r32_annex_c.csv (built by src/build_annex_c.py);
bracket.resolve_third_place_slots does the lookup.

Session 30: added known_results (optional dict for pinning played match
outcomes) and return_results (optional flag to expose all 104 per-match
results). These two features make the simulator result-aware so the
rolling daily update stays correct once the tournament starts.

Two kinds of match sampling:
  - Group stage  -> sample a SCORELINE from the Dixon-Coles grid
                    (we need goals for the FIFA tiebreakers).
  - Knockout     -> sample a WINNER from W/D/L; the draw mass is split
                    50/50 between sides (penalty shootouts ≈ coin flips
                    at international level).

v1 simplification: every match uses neutral=True. The three hosts
(USA, Mexico, Canada) play their group games at home and probably
deserve some advantage; deferred until we have a divergence-detector
signal that it matters.

Run from project root:
    python src/simulate.py
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from bracket import (
    GROUPS,
    TEAM_TO_GROUP,
    R32_BRACKET,
    R16_BRACKET,
    QF_BRACKET,
    SF_BRACKET,
    THIRD_PLACE_MATCH,
    FINAL_MATCH,
    rank_group,
    rank_third_place,
    resolve_third_place_slots,
    _aggregate_stats,
)
from dixon_coles import predict_match


# ----------------------------------------------------------------------
# Sampling primitives
# ----------------------------------------------------------------------

def _sample_scoreline(
    grid: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample (home_goals, away_goals) from the Dixon-Coles scoreline grid.

    The grid is (max_goals+1) x (max_goals+1) and sums to 1.
    We flatten, draw one cell, decode back to (i, j).
    """
    flat = grid.ravel()
    idx = rng.choice(flat.size, p=flat)
    n_cols = grid.shape[1]
    return int(idx // n_cols), int(idx % n_cols)


def _sample_ko_winner(
    home: str,
    away: str,
    p_home: float,
    p_draw: float,
    p_away: float,
    rng: np.random.Generator,
) -> str:
    """Resolve a knockout match. The draw mass is split 50/50.

    International penalty shootouts are statistically close to fair
    coin flips (Apesteguia & Palacios-Huerta 2010 notwithstanding —
    the effect is small enough that for one match it doesn't matter).
    Modeling extra-time with shrunken lambdas adds variance without
    moving the mean much, so we keep this simple for v1.
    """
    p_home_overall = p_home + 0.5 * p_draw
    return home if rng.random() < p_home_overall else away


# ----------------------------------------------------------------------
# Third-place R32 slot assignment (FIFA Annex C)
# ----------------------------------------------------------------------

def _assign_thirds(top_8_thirds: list[str]) -> dict[int, str]:
    """Map R32 match_id → third-placed team for the 8 advancing thirds,
    per FIFA's Annex C (495-row published lookup).

    The lookup in bracket.resolve_third_place_slots returns
    {slot_id: group_letter} — e.g. {"1A": "E", ...} meaning "at slot 1A,
    group E's third-placed team plays". We translate to {match_id: team}.
    """
    team_group = {t: TEAM_TO_GROUP[t] for t in top_8_thirds}
    group_to_team = {g: t for t, g in team_group.items()}
    qualifying_groups = set(team_group.values())

    slot_to_group = resolve_third_place_slots(qualifying_groups)

    # Each R32 match that involves a 3rd-place team has exactly one "1X"
    # slot (the group winner) and one "3..." slot. The "1X" tells us which
    # Annex C key applies to that match.
    assignment: dict[int, str] = {}
    for mid, slot_a, slot_b in R32_BRACKET:
        winner_slot = next(
            (s for s in (slot_a, slot_b) if s.startswith("1")), None
        )
        third_slot = next(
            (s for s in (slot_a, slot_b) if s.startswith("3")), None
        )
        if third_slot is None:
            continue  # match between two runners-up — no 3rd-place team
        assignment[mid] = group_to_team[slot_to_group[winner_slot]]
    return assignment


# ----------------------------------------------------------------------
# known_results helper
# ----------------------------------------------------------------------

def build_known_results(
    match_results: list[dict],
    group_pairs: Optional[set] = None,
) -> dict:
    """Build a known_results dict from simulate_tournament's match_results output.

    Parameters
    ----------
    match_results : list of dicts with keys: home, away, hg, ag, round, winner
        As returned by simulate_tournament(..., return_results=True)["match_results"].
    group_pairs : set of (home, away) tuples, optional
        If supplied, only group rounds are keyed by tuple; all others by frozenset.
        If None, the "round" field is used: "group_stage" → tuple, else frozenset.

    Returns
    -------
    dict
        Group results: (home, away) → (hg, ag)
        KO results:    frozenset({a, b}) → winner string
    """
    known: dict = {}
    for r in match_results:
        h, a = r["home"], r["away"]
        is_group = (
            (group_pairs is not None and (h, a) in group_pairs)
            or (group_pairs is None and r["round"] == "group_stage")
        )
        if is_group:
            known[(h, a)] = (r["hg"], r["ag"])
        else:
            known[frozenset({h, a})] = r["winner"]
    return known


# ----------------------------------------------------------------------
# Main simulator
# ----------------------------------------------------------------------

ROUND_LABELS = [
    "group_stage", "R32", "R16", "QF", "SF",
    "fourth_place", "third_place", "runner_up", "winner",
]


def simulate_tournament(
    ratings: dict[str, float],
    fixtures: list[dict],
    rng: Optional[np.random.Generator] = None,
    known_results: Optional[dict] = None,
    return_results: bool = False,
) -> dict:
    """Simulate the 2026 World Cup once.

    Parameters
    ----------
    ratings : dict[team -> Elo]
        Pre-tournament Elo ratings, typically loaded from
        data/processed/elo_ratings_2026.csv.
    fixtures : list[dict]
        The 72 group-stage fixtures. Each dict needs "home_team" and
        "away_team" keys.
    rng : np.random.Generator, optional
        Pass a seeded RNG (np.random.default_rng(seed)) for reproducible
        runs. Defaults to a fresh unseeded generator.
    known_results : dict, optional
        Pinned results from matches already played.
          Group results: (home, away) → (hg, ag)   — substituted before
            _aggregate_stats so tiebreakers use the real scorelines.
          KO results: frozenset({a, b}) → winner string — forces the
            winner at that node instead of sampling.
        None means every match is sampled (pre-tournament behaviour).
        Defensive check: if a KO frozenset is found but the winner is not
        one of the two teams at that node, a ValueError is raised (catches
        typos in the feed early, rather than silently sampling the wrong
        match).
    return_results : bool, optional
        When True, include "match_results" in the returned dict: a list of
        dicts with keys (home, away, hg, ag, round, winner) for all 104
        matches. hg/ag are None for knockout matches (no scoreline sampled).

    Returns
    -------
    dict with:
        winner, runner_up, third_place, fourth_place : str
        group_results       : {letter: [{team, rank, pts, gd, gf, ga}, ...]}
        knockout_matches    : [{match_id, round, home_team, away_team,
                                winner, loser}]
        team_furthest_round : {team_name: ROUND_LABEL}
        match_results       : list[dict]  (only when return_results=True)

    `team_furthest_round` is what Session 15 aggregates over many runs
    to compute P(reach R16), P(win cup), etc.
    """
    rng = rng if rng is not None else np.random.default_rng()
    kr = known_results or {}

    all_match_results: list[dict] = []  # populated only when return_results=True

    # ---------- 1. Group stage: sample or pin 72 scorelines ----------
    sim_group_matches: list[dict] = []
    for fx in fixtures:
        h, a = fx["home_team"], fx["away_team"]
        if (h, a) in kr:
            hg, ag = kr[(h, a)]
        else:
            pred = predict_match(h, a, ratings, neutral=True)
            hg, ag = _sample_scoreline(pred["scoreline_grid"], rng)
        sim_group_matches.append({
            "home_team": h, "away_team": a,
            "home_score": hg, "away_score": ag,
        })
        if return_results:
            w = h if hg > ag else (a if ag > hg else None)
            all_match_results.append({
                "home": h, "away": a, "hg": hg, "ag": ag,
                "round": "group_stage", "winner": w,
            })

    # ---------- 2. Group standings ----------
    matches_by_group: dict[str, list[dict]] = {}
    group_results: dict[str, list[dict]] = {}
    winners: dict[str, str] = {}
    runners_up: dict[str, str] = {}
    thirds: list[str] = []

    for letter, teams in GROUPS.items():
        team_set = set(teams)
        gms = [
            m for m in sim_group_matches
            if m["home_team"] in team_set and m["away_team"] in team_set
        ]
        matches_by_group[letter] = gms
        order = rank_group(teams, gms)
        stats = _aggregate_stats(gms, team_set)
        group_results[letter] = [
            {"team": t, "rank": i + 1, **stats[t]}
            for i, t in enumerate(order)
        ]
        winners[letter] = order[0]
        runners_up[letter] = order[1]
        thirds.append(order[2])

    # ---------- 3. Pick 8 advancing thirds + assign to slots ----------
    ranked_thirds = rank_third_place(thirds, matches_by_group)
    top_8 = ranked_thirds[:8]
    third_assign = _assign_thirds(top_8)  # match_id -> team name

    # ---------- 4. Knockout rounds ----------
    knockout_log: list[dict] = []
    winners_by_match: dict[int, str] = {}

    def resolve_slot(slot: str, match_id: int) -> str:
        """Resolve a R32 slot string ('1A', '2C', '3CEFHI') to a team."""
        if slot.startswith("1"):
            return winners[slot[1]]
        if slot.startswith("2"):
            return runners_up[slot[1]]
        if slot.startswith("3"):
            return third_assign[match_id]
        raise ValueError(f"unknown slot syntax: {slot!r}")

    def play_knockout(mid: int, rd: str, home: str, away: str) -> None:
        pair = frozenset({home, away})
        if pair in kr:
            # Pinned result: validate before using (catches feed typos)
            forced = kr[pair]
            if forced not in (home, away):
                raise ValueError(
                    f"known_results has winner {forced!r} for "
                    f"{home!r} vs {away!r} (match {mid}, {rd}), "
                    f"but that team is not in this match. "
                    f"Check wc_results_manual.csv for a typo."
                )
            winner = forced
        else:
            pred = predict_match(home, away, ratings, neutral=True)
            winner = _sample_ko_winner(
                home, away,
                pred["p_home_win"], pred["p_draw"], pred["p_away_win"],
                rng,
            )
        loser = away if winner == home else home
        winners_by_match[mid] = winner
        knockout_log.append({
            "match_id": mid, "round": rd,
            "home_team": home, "away_team": away,
            "winner": winner, "loser": loser,
        })
        if return_results:
            all_match_results.append({
                "home": home, "away": away, "hg": None, "ag": None,
                "round": rd, "winner": winner,
            })

    # R32 — slots reference group standings
    for mid, slot_a, slot_b in R32_BRACKET:
        play_knockout(
            mid, "R32",
            resolve_slot(slot_a, mid),
            resolve_slot(slot_b, mid),
        )

    # R16, QF, SF — sources reference earlier match IDs
    for round_name, bracket in [
        ("R16", R16_BRACKET),
        ("QF",  QF_BRACKET),
        ("SF",  SF_BRACKET),
    ]:
        for mid, src_a, src_b in bracket:
            play_knockout(
                mid, round_name,
                winners_by_match[src_a],
                winners_by_match[src_b],
            )

    # Third-place playoff: losers of the two semifinals
    sf_ids = [mid for mid, _, _ in SF_BRACKET]
    sf_losers = [
        next(m for m in knockout_log if m["match_id"] == sid)["loser"]
        for sid in sf_ids
    ]
    play_knockout(
        THIRD_PLACE_MATCH, "third_place_playoff",
        sf_losers[0], sf_losers[1],
    )

    # Final: winners of the two semifinals
    play_knockout(
        FINAL_MATCH, "final",
        winners_by_match[sf_ids[0]],
        winners_by_match[sf_ids[1]],
    )

    # ---------- 5. team_furthest_round ----------
    # Strategy: walk stages in order, marking BOTH participants at each
    # stage's label. By the end, non-podium teams hold their final stage.
    # Then override the four podium positions explicitly.
    furthest: dict[str, str] = {t: "group_stage" for t in TEAM_TO_GROUP}
    for rd_label in ("R32", "R16", "QF", "SF"):
        for m in knockout_log:
            if m["round"] == rd_label:
                furthest[m["home_team"]] = rd_label
                furthest[m["away_team"]] = rd_label

    third_match = next(m for m in knockout_log if m["round"] == "third_place_playoff")
    final_match = next(m for m in knockout_log if m["round"] == "final")

    furthest[third_match["winner"]] = "third_place"
    furthest[third_match["loser"]] = "fourth_place"
    furthest[final_match["winner"]] = "winner"
    furthest[final_match["loser"]] = "runner_up"

    result = {
        "winner":              final_match["winner"],
        "runner_up":           final_match["loser"],
        "third_place":         third_match["winner"],
        "fourth_place":        third_match["loser"],
        "group_results":       group_results,
        "knockout_matches":    knockout_log,
        "team_furthest_round": furthest,
    }
    if return_results:
        result["match_results"] = all_match_results
    return result


# ----------------------------------------------------------------------
# Demo / sanity check
# ----------------------------------------------------------------------

def _load_ratings() -> dict[str, float]:
    df = pd.read_csv("data/processed/elo_ratings_2026.csv")
    return dict(zip(df["team"], df["elo"]))


def _load_fixtures() -> list[dict]:
    df = pd.read_csv("data/processed/fixtures_2026.csv")
    return df[["home_team", "away_team"]].to_dict("records")


def main() -> None:
    ratings = _load_ratings()
    fixtures = _load_fixtures()

    seed = 42
    print(f"Simulating one tournament with seed={seed}\n")
    rng = np.random.default_rng(seed)
    result = simulate_tournament(ratings, fixtures, rng, return_results=True)

    # Group standings
    print("=" * 70)
    print("GROUP STAGE")
    print("=" * 70)
    for letter in sorted(GROUPS):
        print(f"\nGroup {letter}:")
        for e in result["group_results"][letter]:
            print(
                f"  {e['rank']}. {e['team']:<24}  "
                f"pts={e['pts']}  gd={e['gd']:+d}  "
                f"gf={e['gf']}  ga={e['ga']}"
            )

    # Knockout
    print("\n" + "=" * 70)
    print("KNOCKOUT STAGE")
    print("=" * 70)
    round_labels = [
        ("R32", "Round of 32"),
        ("R16", "Round of 16"),
        ("QF",  "Quarterfinals"),
        ("SF",  "Semifinals"),
        ("third_place_playoff", "Third-place playoff"),
        ("final", "Final"),
    ]
    for rd, title in round_labels:
        ms = [m for m in result["knockout_matches"] if m["round"] == rd]
        print(f"\n{title}:")
        for m in ms:
            print(
                f"  {m['home_team']:<24} vs {m['away_team']:<24}  "
                f"-> winner: {m['winner']}"
            )

    # Podium
    print("\n" + "=" * 70)
    print("PODIUM")
    print("=" * 70)
    print(f"  1st  Winner     : {result['winner']}")
    print(f"  2nd  Runner-up  : {result['runner_up']}")
    print(f"  3rd  Third place: {result['third_place']}")
    print(f"  4th  Fourth     : {result['fourth_place']}")

    # Sanity: every team accounted for exactly once
    print("\nFurthest-round distribution (should sum to 48):")
    counts = Counter(result["team_furthest_round"].values())
    for label in ROUND_LABELS:
        print(f"  {label:<14}: {counts.get(label, 0)}")
    total = sum(counts.values())
    print(f"  {'TOTAL':<14}: {total}")
    assert total == 48, f"expected 48 teams, got {total}"
    # SF is expected to be 0: both SF losers play the 3rd-place playoff
    # and end up as third_place or fourth_place.
    assert counts.get("SF", 0) == 0, (
        "SF count should be 0 — SF losers should all be reclassified "
        "as third_place or fourth_place."
    )
    print("\n[OK] Sanity checks pass.")

    # --- Result-aware self-tests (Session 30) ---
    print("\nRunning result-aware self-tests...")

    group_pairs = {(fx["home_team"], fx["away_team"]) for fx in fixtures}

    # 1. Fully pinned tournament reproduces truth exactly.
    # Build known_results from the seed-42 run (all 104 matches).
    known_all = build_known_results(result["match_results"])
    rng_b = np.random.default_rng(999)  # different seed — irrelevant when all pinned
    result_pinned = simulate_tournament(ratings, fixtures, rng_b, known_results=known_all)
    assert result_pinned["winner"] == result["winner"], (
        f"fully pinned: champion mismatch — expected {result['winner']!r}, "
        f"got {result_pinned['winner']!r}"
    )
    for team in TEAM_TO_GROUP:
        expected = result["team_furthest_round"][team]
        got = result_pinned["team_furthest_round"][team]
        assert got == expected, (
            f"fully pinned: {team!r} furthest round — expected {expected!r}, got {got!r}"
        )
    print("[OK] Fully pinned tournament reproduces truth exactly.")

    # 2. Partially pinned (group stage only): 4th-place teams never advance.
    # Build group-only known_results from the seed-42 run.
    group_results_only = build_known_results(
        [r for r in result["match_results"] if r["round"] == "group_stage"]
    )
    # Identify teams that finished 4th in their group in the truth run.
    eliminated_4th = {
        next(s["team"] for s in standings if s["rank"] == 4)
        for standings in result["group_results"].values()
    }
    # Run 10 simulations — with group results pinned, group standings are
    # deterministic, so 4th-place teams can never appear in the knockout stage.
    for i in range(10):
        r = simulate_tournament(
            ratings, fixtures, np.random.default_rng(i * 7),
            known_results=group_results_only,
        )
        for t in eliminated_4th:
            got = r["team_furthest_round"][t]
            assert got == "group_stage", (
                f"partially pinned (run {i}): {t!r} finished 4th in truth but "
                f"reached {got!r} with group results pinned"
            )
    print("[OK] Partially pinned: 4th-place group teams never advance.")

    # 3. Defensive check: known KO winner not in the match raises ValueError.
    bad_kr = {frozenset({"Spain", "France"}): "Germany"}  # Germany not playing
    try:
        simulate_tournament(ratings, fixtures, np.random.default_rng(0),
                            known_results=bad_kr)
        # If Spain and France don't meet, no error is expected; if they do, it raises.
        print("[OK] Defensive check: Spain vs France did not meet in this run (no error).")
    except ValueError as e:
        assert "Germany" in str(e), f"unexpected ValueError text: {e}"
        print("[OK] Defensive check: bad KO winner raised ValueError as expected.")

    print("\n[OK] All result-aware self-tests pass.")


if __name__ == "__main__":
    main()
