"""Pure R32 bracket resolver: played group results → the 16 R32 matchups.

Session 38a. This is the display-layer port of simulate.py's bracket glue
(resolve_slot + _assign_thirds + the group-standings build loop, simulate.py
~257-339), so the populated-bracket view (Session 38b) can wire known data into
a known shape without importing the simulation layer.

Firewall (38a)
--------------
This module imports `bracket` read-only and NOTHING from the simulation/pipeline
layer (no simulate, monte_carlo, triple_compare, update_ledger, update). It is a
pure consumer in the verdict.py / divergence_log.py tradition: feed it played
group scorelines, get back a fully-resolved 32-team R32.

The gotcha (38-RECON)
---------------------
`bracket.resolve_third_place_slots(...)` returns {slot_id: group_letter} keyed on
the **1X winner-slot id**, NOT the "3..." string — e.g. '1A' -> 'E' means group
E's third-placed team plays in the match whose group-winner slot is 1A. A "3..."
slot is therefore resolved by finding the winner-slot it's paired against in
R32_BRACKET, looking *that* up in the dict, and taking that group's third. This
is ported verbatim from simulate._assign_thirds rather than reconstructed.
"""
from __future__ import annotations

import csv
from pathlib import Path

from bracket import (
    GROUPS,
    TEAM_TO_GROUP,
    R32_BRACKET,
    rank_group,
    rank_third_place,
    resolve_third_place_slots,
)


# ----------------------------------------------------------------------
# Ported glue: thirds assignment + slot resolution
# ----------------------------------------------------------------------
# Both functions are faithful ports of simulate.py (_assign_thirds at ~103,
# resolve_slot at ~290). Kept structurally identical so a future drift in the
# sim glue is easy to diff against this copy.

def _assign_thirds(top_8_thirds: list[str]) -> dict[int, str]:
    """Map R32 match_id -> third-placed team for the 8 advancing thirds,
    per FIFA's Annex C (495-row published lookup).

    resolve_third_place_slots returns {slot_id: group_letter} keyed on the "1X"
    winner each third faces (e.g. {"1A": "E", ...} = "group E's third plays at
    slot 1A"). We translate that to {match_id: team}.
    """
    team_group = {t: TEAM_TO_GROUP[t] for t in top_8_thirds}
    group_to_team = {g: t for t, g in team_group.items()}
    qualifying_groups = set(team_group.values())

    slot_to_group = resolve_third_place_slots(qualifying_groups)

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


def _resolve_slot(
    slot: str,
    match_id: int,
    winners: dict[str, str],
    runners_up: dict[str, str],
    third_assign: dict[int, str],
) -> str:
    """Resolve a R32 slot string ('1A', '2C', '3CEFHI') to a team."""
    if slot.startswith("1"):
        return winners[slot[1]]
    if slot.startswith("2"):
        return runners_up[slot[1]]
    if slot.startswith("3"):
        return third_assign[match_id]
    raise ValueError(f"unknown slot syntax: {slot!r}")


# ----------------------------------------------------------------------
# Core resolver (pure)
# ----------------------------------------------------------------------

def resolve_r32(
    group_matches: list[dict],
) -> tuple[dict[int, tuple[str, str]], dict[str, str]]:
    """Resolve all 16 R32 matchups from completed group-stage results.

    Parameters
    ----------
    group_matches : list of dicts
        Played group fixtures, each {"home_team", "away_team", "home_score",
        "away_score"} — the same shape simulate.py's standings loop consumes.
        Must cover all 12 groups' 6 round-robin matches (group stage complete).

    Returns
    -------
    (bracket, slot_to_team)
        bracket      : {match_id (73-88): (team_a, team_b)}
        slot_to_team : {slot_string: team} for all 32 R32 slots — the
                       underlying map (Session 38b may want it).
    """
    # ---- 1. Group standings (port of simulate.py ~257-279) ----
    matches_by_group: dict[str, list[dict]] = {}
    winners: dict[str, str] = {}
    runners_up: dict[str, str] = {}
    thirds: list[str] = []

    for letter, teams in GROUPS.items():
        team_set = set(teams)
        gms = [
            m for m in group_matches
            if m["home_team"] in team_set and m["away_team"] in team_set
        ]
        matches_by_group[letter] = gms
        order = rank_group(teams, gms)
        winners[letter] = order[0]
        runners_up[letter] = order[1]
        thirds.append(order[2])

    # ---- 2. Pick 8 advancing thirds + assign to slots (port of ~282-284) ----
    ranked_thirds = rank_third_place(thirds, matches_by_group)
    top_8 = ranked_thirds[:8]
    third_assign = _assign_thirds(top_8)

    # ---- 3. Resolve every R32 slot (port of ~333-339) ----
    bracket: dict[int, tuple[str, str]] = {}
    slot_to_team: dict[str, str] = {}
    for mid, slot_a, slot_b in R32_BRACKET:
        team_a = _resolve_slot(slot_a, mid, winners, runners_up, third_assign)
        team_b = _resolve_slot(slot_b, mid, winners, runners_up, third_assign)
        bracket[mid] = (team_a, team_b)
        slot_to_team[slot_a] = team_a
        slot_to_team[slot_b] = team_b

    return bracket, slot_to_team


# ----------------------------------------------------------------------
# Production loader (the only file-format adapter)
# ----------------------------------------------------------------------

_MATCHES_CLEAN = Path("data/processed/matches_clean.csv")


def _load_group_results(path: Path = _MATCHES_CLEAN) -> list[dict]:
    """Read the 2026 WC group-stage scorelines from matches_clean.csv and adapt
    them to resolve_r32's input shape.

    A row is a WC group fixture iff tournament == "FIFA World Cup", both teams
    are in GROUPS, they share a group letter (cross-group rows are knockout),
    and both scores are present. Returns [{home_team, away_team, home_score,
    away_score}] with integer scores. Run from the project root.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run from the project root (not from src/)."
        )

    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("tournament") != "FIFA World Cup":
                continue
            h, a = row["home_team"], row["away_team"]
            if h not in TEAM_TO_GROUP or a not in TEAM_TO_GROUP:
                continue
            if TEAM_TO_GROUP[h] != TEAM_TO_GROUP[a]:
                continue  # cross-group → knockout, not a group fixture
            hs, as_ = row.get("home_score", ""), row.get("away_score", "")
            if hs in (None, "") or as_ in (None, ""):
                continue  # unplayed
            out.append({
                "home_team": h, "away_team": a,
                "home_score": int(float(hs)), "away_score": int(float(as_)),
            })
    return out


# ----------------------------------------------------------------------
# Self-tests (run: python src/bracket_resolve.py)
# ----------------------------------------------------------------------

def _synthetic_complete_groups() -> list[dict]:
    """Deterministic, tie-free complete group stage: 12 groups x 6 matches.

    Within every group a strict A>B>C>D hierarchy (9/6/3/0 pts) makes standings
    unambiguous. The third-placed team (C) scores (1 + group_index) against D,
    so each group's third has a distinct (gd, gf) — rank_third_place is fully
    ordered with no ties, and the best-8 are the 8 highest-indexed groups
    (E..L), spanning 8 distinct groups.
    """
    matches: list[dict] = []
    for gi, letter in enumerate("ABCDEFGHIJKL"):
        A, B, C, D = GROUPS[letter]
        c_goals = 1 + gi  # third's goal tally vs D — distinct per group
        results = [
            (A, B, 3, 0), (A, C, 3, 0), (A, D, 3, 0),
            (B, C, 2, 0), (B, D, 2, 0),
            (C, D, c_goals, 0),
        ]
        for h, a, hs, as_ in results:
            matches.append({
                "home_team": h, "away_team": a,
                "home_score": hs, "away_score": as_,
            })
    return matches


def _self_test() -> None:
    all_48 = {t for ts in GROUPS.values() for t in ts}
    group_matches = _synthetic_complete_groups()
    bracket, slot_to_team = resolve_r32(group_matches)

    # 1. All 16 matchups resolved, ids 73-88, no unfilled slot.
    assert set(bracket) == {mid for mid, _, _ in R32_BRACKET}, \
        f"R32 ids mismatch: {sorted(bracket)}"
    for mid, (ta, tb) in bracket.items():
        assert ta and tb, f"match {mid} has an unfilled slot: {(ta, tb)}"

    # 2. 32 distinct teams across the bracket.
    teams = [t for pair in bracket.values() for t in pair]
    assert len(teams) == 32, f"expected 32 slots, got {len(teams)}"
    assert len(set(teams)) == 32, "duplicate team in the R32"

    # 3. No same-group R32 rematch.
    for mid, (ta, tb) in bracket.items():
        assert TEAM_TO_GROUP[ta] != TEAM_TO_GROUP[tb], \
            f"match {mid} is a group rematch: {ta} vs {tb}"

    # 4. The 8 advancing thirds come from 8 distinct groups.
    third_teams = [t for slot, t in slot_to_team.items() if slot.startswith("3")]
    assert len(third_teams) == 8, f"expected 8 third slots, got {len(third_teams)}"
    third_groups = {TEAM_TO_GROUP[t] for t in third_teams}
    assert len(third_groups) == 8, \
        f"thirds span {len(third_groups)} groups, expected 8: {sorted(third_groups)}"
    # With the synthetic gradient the best-8 thirds are groups E..L.
    assert third_groups == set("EFGHIJKL"), \
        f"expected thirds from E..L, got {sorted(third_groups)}"

    # 5. Every resolved team is one of the 48.
    assert set(teams) <= all_48, f"unknown team(s): {set(teams) - all_48}"

    print("bracket_resolve.py resolve_r32 self-test passed "
          "(16 matchups, 32 distinct teams, no group rematch, "
          "8 thirds from 8 distinct groups E-L, all in 48)")


if __name__ == "__main__":
    _self_test()
