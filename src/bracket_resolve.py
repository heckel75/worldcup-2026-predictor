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
    R16_BRACKET,
    QF_BRACKET,
    SF_BRACKET,
    THIRD_PLACE_MATCH,
    FINAL_MATCH,
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
# Full bracket resolver (Session 38b) — group results → populated knockout
# ----------------------------------------------------------------------
# resolve_bracket walks the whole match-ID tree (R32 73-88 → R16 89-96 → QF
# 97-100 → SF 101-102 → final 104, plus the third-place play-off 103) feeding
# resolve_r32 the completed group standings and looking each later round up by
# the advancing-team field. Pure: it consumes played rows, returns a render
# shape, and imports nothing from the sim/pipeline.

# Match 103 (third place) is the one node contested by the two SF LOSERS, not
# winners; every other upstream node takes the winners of its two feeders.
_SF_IDS = [mid for mid, _, _ in SF_BRACKET]            # [101, 102]
_R32_IDS = {mid for mid, _, _ in R32_BRACKET}          # 73..88

# match_id -> (source_a_id, source_b_id) for every node above the R32. The
# final (104) is the winners of the two semifinals; the third-place play-off
# (103) is special-cased (SF losers) in resolve_bracket, not listed here.
_UPSTREAM: dict[int, tuple[int, int]] = {
    mid: (src_a, src_b)
    for mid, src_a, src_b in (*R16_BRACKET, *QF_BRACKET, *SF_BRACKET)
}
_UPSTREAM[FINAL_MATCH] = (_SF_IDS[0], _SF_IDS[1])


def resolve_bracket(played_matches: list[dict] | None = None) -> dict:
    """Resolve the full 2026 knockout bracket from played results.

    Parameters
    ----------
    played_matches : list of dicts, optional
        Played WC fixtures (the same injectable pattern as resolve_r32). Each
        row is either a GROUP fixture (same 2026 group, "home_score"/"away_score"
        present, "advanced" blank) or a KNOCKOUT fixture ("advanced" naming the
        team that went through). When None, loads from matches_clean.csv via
        _load_wc_results (2026-scoped — see _split_results).

    Returns
    -------
    dict with:
        complete    : bool   — True once all 16 R32 cells hold real teams.
        rounds      : list[{label, matches}] in R32→Final order. Each match is
                      {match_id, team_a, team_b, winner, loser, played} with
                      internal team names (None where the slot is undetermined).
        third_place : the same match dict for node 103, or None when incomplete.

    Until the group stage is complete (< 72 group fixtures present) the resolver
    returns {"complete": False, "rounds": [], "third_place": None} WITHOUT
    calling resolve_r32 — that resolver assumes complete groups, so feeding it a
    partial group stage is undefined. This is what makes the bracket page hold
    its placeholder pre-completion.
    """
    if played_matches is None:
        group_rows, ko_rows = _load_wc_results()
    else:
        group_rows, ko_rows = _split_results(played_matches)

    incomplete = {"complete": False, "rounds": [], "third_place": None}
    if len(group_rows) < 72:
        return incomplete  # group stage not finished — hold the placeholder

    r32_bracket, _slot_to_team = resolve_r32(group_rows)

    # Defensive: resolve_r32 should fill all 16 from complete groups, but if any
    # slot came back undetermined, report incomplete rather than draw a half-tree.
    if not all(
        r32_bracket[mid][0] and r32_bracket[mid][1] for mid in _R32_IDS
    ):
        return incomplete

    # Unordered-pair → advancing team, from the played KO rows only.
    ko_winner: dict[frozenset[str], str] = {
        frozenset({r["home_team"], r["away_team"]}): r["advanced"]
        for r in ko_rows
    }

    memo: dict[int, dict] = {}

    def node(mid: int) -> dict:
        if mid in memo:
            return memo[mid]

        if mid in _R32_IDS:
            team_a, team_b = r32_bracket[mid]
        elif mid == THIRD_PLACE_MATCH:
            team_a = node(_SF_IDS[0])["loser"]
            team_b = node(_SF_IDS[1])["loser"]
        else:
            src_a, src_b = _UPSTREAM[mid]
            team_a = node(src_a)["winner"]
            team_b = node(src_b)["winner"]

        res = {
            "match_id": mid,
            "team_a": team_a,
            "team_b": team_b,
            "winner": None,
            "loser": None,
            "played": False,
        }
        # A node is played only when both teams are determined AND a KO row
        # names one of them as the advancing side.
        if team_a is not None and team_b is not None:
            adv = ko_winner.get(frozenset({team_a, team_b}))
            if adv in (team_a, team_b):
                res["winner"] = adv
                res["loser"] = team_b if adv == team_a else team_a
                res["played"] = True
        memo[mid] = res
        return res

    rounds = [
        {"label": label, "matches": [node(mid) for mid in ids]}
        for label, ids in (
            ("Round of 32",   [mid for mid, _, _ in R32_BRACKET]),
            ("Round of 16",   [mid for mid, _, _ in R16_BRACKET]),
            ("Quarter-finals", [mid for mid, _, _ in QF_BRACKET]),
            ("Semi-finals",   _SF_IDS),
            ("Final",         [FINAL_MATCH]),
        )
    ]

    return {
        "complete": True,
        "rounds": rounds,
        "third_place": node(THIRD_PLACE_MATCH),
    }


# ----------------------------------------------------------------------
# Production loader (the only file-format adapter)
# ----------------------------------------------------------------------

_MATCHES_CLEAN = Path("data/processed/matches_clean.csv")


def _split_results(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Classify played rows into (group_rows, ko_rows).

    A KNOCKOUT row is any row whose "advanced" field names a team (the manual
    feed sets `advanced` for KO matches only). A GROUP row is the 38a predicate:
    both teams in the same 2026 group, both scores present, `advanced` blank.
    Classifying KO by the `advanced` field (not by cross-group membership) is
    deliberate — two teams from the same 2026 group CAN meet again in the
    knockout stage, and a cross-group test would misfile that match.

    group_rows -> [{home_team, away_team, home_score, away_score}] (ints)
    ko_rows    -> [{home_team, away_team, advanced}]
    """
    group: list[dict] = []
    ko: list[dict] = []
    for r in rows:
        h, a = r["home_team"], r["away_team"]
        if h not in TEAM_TO_GROUP or a not in TEAM_TO_GROUP:
            continue
        adv = r.get("advanced")
        # blank covers "", None, and a float nan (str(nan) == "nan")
        adv_set = adv is not None and str(adv).strip() not in ("", "nan", "None")
        if adv_set:
            ko.append({"home_team": h, "away_team": a, "advanced": str(adv).strip()})
            continue
        if TEAM_TO_GROUP[h] != TEAM_TO_GROUP[a]:
            continue  # cross-group with no advancing team — unresolved, skip
        hs, as_ = r.get("home_score", ""), r.get("away_score", "")
        if hs in (None, "") or as_ in (None, ""):
            continue  # unplayed group fixture
        group.append({
            "home_team": h, "away_team": a,
            "home_score": int(float(hs)), "away_score": int(float(as_)),
        })
    return group, ko


def _load_wc_results(path: Path = _MATCHES_CLEAN) -> tuple[list[dict], list[dict]]:
    """Read the 2026 WC played rows from matches_clean.csv, split into
    (group_rows, ko_rows) by _split_results.

    Scoped to tournament == "FIFA World Cup" AND a 2026 date. The edition bound
    is load-bearing: matches_clean also holds the 2018 and 2022 World Cups under
    the same tournament string, and the KO lookup matches on the unordered team
    pair — without the date bound a 2022 final rematch (or any old-edition pair)
    would collide with a 2026 one, and four old group matches between teams that
    now share a 2026 group would pollute the standings. Run from the project root.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run from the project root (not from src/)."
        )

    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("tournament") != "FIFA World Cup":
                continue
            if not str(row.get("date", "")).startswith("2026"):
                continue  # exclude the 2018/2022 editions — see docstring
            rows.append(row)
    return _split_results(rows)


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


def _synthetic_ko(group_matches: list[dict]) -> tuple[list[dict], str]:
    """Layer a deterministic, fully-played knockout stage on top of a complete
    group stage. Winner at every node = the alphabetically-first team (pure
    function of the matchup), so the whole tree resolves to one champion.

    Returns (ko_rows, champion) where ko_rows are {home_team, away_team,
    advanced} for all 16 R32 + 8 R16 + 4 QF + 2 SF + 1 third-place + 1 final.
    """
    r32, _ = resolve_r32(group_matches)
    win: dict[int, str] = {}
    lose: dict[int, str] = {}
    ko_rows: list[dict] = []

    def play(mid: int, a: str, b: str) -> None:
        w, l = (a, b) if a < b else (b, a)   # deterministic pick
        win[mid], lose[mid] = w, l
        ko_rows.append({"home_team": a, "away_team": b, "advanced": w})

    for mid in (m for m, _, _ in R32_BRACKET):
        play(mid, *r32[mid])
    for mid, src_a, src_b in (*R16_BRACKET, *QF_BRACKET, *SF_BRACKET):
        play(mid, win[src_a], win[src_b])
    play(THIRD_PLACE_MATCH, lose[_SF_IDS[0]], lose[_SF_IDS[1]])
    play(FINAL_MATCH, win[_SF_IDS[0]], win[_SF_IDS[1]])
    return ko_rows, win[FINAL_MATCH]


def _self_test_bracket() -> None:
    group_matches = _synthetic_complete_groups()

    # ---- 1. Fully-played tree resolves to the synthetic champion ----
    ko_rows, champion = _synthetic_ko(group_matches)
    full = resolve_bracket(group_matches + ko_rows)

    assert full["complete"], "full bracket should be complete"
    sizes = [len(r["matches"]) for r in full["rounds"]]
    assert sizes == [16, 8, 4, 2, 1], f"round sizes wrong: {sizes}"
    assert full["third_place"] is not None, "third-place node missing"

    # Every cell populated and played, and a single winner at the final.
    for rnd in full["rounds"]:
        for m in rnd["matches"]:
            assert m["team_a"] and m["team_b"], f"unfilled cell: {m}"
            assert m["played"] and m["winner"], f"node not resolved: {m}"
    final_node = full["rounds"][-1]["matches"][0]
    assert final_node["winner"] == champion, (
        f"final winner {final_node['winner']!r} != champion {champion!r}"
    )

    # The third-place node is contested by exactly the two SF losers.
    sf_nodes = full["rounds"][3]["matches"]
    sf_losers = {n["loser"] for n in sf_nodes}
    tp = full["third_place"]
    assert {tp["team_a"], tp["team_b"]} == sf_losers, (
        f"third-place feeders {{ {tp['team_a']}, {tp['team_b']} }} "
        f"!= SF losers {sf_losers}"
    )

    # ---- 2. Groups complete, only the R32 played (day-one state) ----
    r32_only = ko_rows[:16]
    partial = resolve_bracket(group_matches + r32_only)
    assert partial["complete"], "groups complete -> bracket complete"
    for m in partial["rounds"][0]["matches"]:           # R32
        assert m["played"] and m["winner"], f"R32 should be played: {m}"
    for rnd in partial["rounds"][1:]:                    # R16 and beyond
        for m in rnd["matches"]:
            assert not m["played"] and m["winner"] is None, \
                f"node above R32 should be unplayed: {m}"
    # R16 cells carry the determined R32 winners; deeper feeders are None.
    assert all(m["team_a"] and m["team_b"] for m in partial["rounds"][1]["matches"]), \
        "R16 cells should hold the resolved R32 winners"
    assert partial["third_place"]["team_a"] is None, \
        "third-place feeders unresolved until the SFs are played"

    # ---- 3. Group stage incomplete -> placeholder ----
    short = resolve_bracket(group_matches[:-1] + ko_rows)   # drop one group game
    assert short == {"complete": False, "rounds": [], "third_place": None}, \
        f"incomplete groups should hold the placeholder, got {short!r}"

    print("bracket_resolve.py resolve_bracket self-test passed "
          "(full tree -> champion, third-place = SF losers; "
          "partial R32-only renders; incomplete groups -> placeholder)")


if __name__ == "__main__":
    _self_test()
    _self_test_bracket()
