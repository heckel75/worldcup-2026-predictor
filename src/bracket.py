"""
2026 FIFA World Cup bracket: groups, knockout structure, tiebreakers.

Sources
-------
- Groups & bracket: FIFA Final Draw, Dec 5 2025.
  Wikipedia: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage
- Tiebreakers: FIFA 2026 regulations (via ESPN summary).
- Annex C (R32 third-place slot lookup): data/raw/r32_annex_c.csv,
  built by src/build_annex_c.py from FIFA's published 495-row table.

Conventions
-----------
- Team names follow the project's dataset conventions (Turkey, Czech Republic,
  Ivory Coast, ...). The dashboard layer in Week 5 will translate to official
  display names (Türkiye, Czechia, Côte d'Ivoire). Don't change them here.
- Group letters match FIFA's official assignments. The R32 slot syntax
  ("1A", "2C", "3CEFHI") is the same one FIFA uses in their published bracket
  so it's easy to cross-check.
- Group-stage fixtures live in `data/processed/fixtures_2026.csv`, not here.
  This module owns the *structure*; the CSV owns the *schedule*.
"""
from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Optional


# ----------------------------------------------------------------------
# Groups
# ----------------------------------------------------------------------

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",      "South Africa",            "South Korea",  "Czech Republic"],
    "B": ["Canada",      "Bosnia and Herzegovina",  "Qatar",        "Switzerland"],
    "C": ["Brazil",      "Morocco",                 "Haiti",        "Scotland"],
    "D": ["USA",         "Paraguay",                "Australia",    "Turkey"],
    "E": ["Germany",     "Curaçao",                 "Ivory Coast",  "Ecuador"],
    "F": ["Netherlands", "Japan",                   "Sweden",       "Tunisia"],
    "G": ["Belgium",     "Egypt",                   "Iran",         "New Zealand"],
    "H": ["Spain",       "Cape Verde",              "Saudi Arabia", "Uruguay"],
    "I": ["France",      "Senegal",                 "Iraq",         "Norway"],
    "J": ["Argentina",   "Algeria",                 "Austria",      "Jordan"],
    "K": ["Portugal",    "DR Congo",                "Uzbekistan",   "Colombia"],
    "L": ["England",     "Croatia",                 "Ghana",        "Panama"],
}

TEAM_TO_GROUP: dict[str, str] = {
    team: letter for letter, teams in GROUPS.items() for team in teams
}


# ----------------------------------------------------------------------
# Bracket structure
# ----------------------------------------------------------------------
# R32 slot syntax:
#   "1X"     = winner of group X
#   "2X"     = runner-up of group X
#   "3XYZWV" = the third-placed team from one of those 5 groups
#              (which one — per Annex C of the FIFA regs — depends on
#               which 8 third-placed teams advance)

R32_BRACKET: list[tuple[int, str, str]] = [
    (73, "2A", "2B"     ),
    (74, "1E", "3ABCDF" ),
    (75, "1F", "2C"     ),
    (76, "1C", "2F"     ),
    (77, "1I", "3CDFGH" ),
    (78, "2E", "2I"     ),
    (79, "1A", "3CEFHI" ),
    (80, "1L", "3EHIJK" ),
    (81, "1D", "3BEFIJ" ),
    (82, "1G", "3AEHIJ" ),
    (83, "2K", "2L"     ),
    (84, "1H", "2J"     ),
    (85, "1B", "3EFGIJ" ),
    (86, "1J", "2H"     ),
    (87, "1K", "3DEIJL" ),
    (88, "2D", "2G"     ),
]

# Later rounds: each entry is (match_id, source_a, source_b)
# where source_a and source_b are the IDs of the matches whose winners advance.

R16_BRACKET: list[tuple[int, int, int]] = [
    (89, 74, 77),
    (90, 73, 75),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
]

QF_BRACKET: list[tuple[int, int, int]] = [
    (97,  89, 90),
    (98,  93, 94),
    (99,  91, 92),
    (100, 95, 96),
]

SF_BRACKET: list[tuple[int, int, int]] = [
    (101, 97, 98),
    (102, 99, 100),
]

THIRD_PLACE_MATCH = 103   # losers of 101 and 102
FINAL_MATCH       = 104   # winners of 101 and 102


# ----------------------------------------------------------------------
# Match-stats helpers
# ----------------------------------------------------------------------

def _played(m: Mapping) -> bool:
    """A match dict counts as played iff both scores are concrete numbers."""
    hs, as_ = m.get("home_score"), m.get("away_score")
    return hs not in (None, "") and as_ not in (None, "")


def _aggregate_stats(
    matches: Iterable[Mapping],
    teams: set[str],
) -> dict[str, dict[str, int]]:
    """
    Build per-team {pts, gf, ga, gd} from any iterable of match dicts.
    Only matches where BOTH teams are in `teams` and the match is played
    contribute to the totals.
    """
    stats = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
    for m in matches:
        h, a = m["home_team"], m["away_team"]
        if h not in teams or a not in teams or not _played(m):
            continue
        hs, as_ = int(m["home_score"]), int(m["away_score"])
        stats[h]["gf"] += hs;  stats[h]["ga"] += as_
        stats[a]["gf"] += as_; stats[a]["ga"] += hs
        if hs > as_:
            stats[h]["pts"] += 3
        elif hs < as_:
            stats[a]["pts"] += 3
        else:
            stats[h]["pts"] += 1
            stats[a]["pts"] += 1
    for s in stats.values():
        s["gd"] = s["gf"] - s["ga"]
    return stats


# ----------------------------------------------------------------------
# Group-stage tiebreakers
# ----------------------------------------------------------------------
# FIFA 2026 tiebreaker order for the group stage:
#   1. Most points in the group
#   2. If 3+ teams remain tied on points, head-to-head criteria:
#        2a. H2H points among still-tied teams
#        2b. H2H goal difference
#        2c. H2H goals scored
#      (Re-applied recursively to any sub-tied subset that h2h splits off.)
#   3. If still tied — or if only 2 teams were tied on points to begin with —
#      drop to overall criteria:
#        3a. GD in all group matches
#        3b. Goals scored in all group matches
#        3c. Fair play (omitted — we don't model bookings)
#        3d. FIFA world ranking

def rank_group(
    group_teams: list[str],
    group_matches: Sequence[Mapping],
    fifa_ranks: Optional[Mapping[str, int]] = None,
) -> list[str]:
    """Return the 4 group teams in finishing order, best first."""
    stats = _aggregate_stats(group_matches, set(group_teams))
    by_pts = sorted(group_teams, key=lambda t: -stats[t]["pts"])

    out: list[str] = []
    i = 0
    while i < len(by_pts):
        # Walk forward to find the next block of teams equal on points
        j = i + 1
        while j < len(by_pts) and stats[by_pts[j]]["pts"] == stats[by_pts[i]]["pts"]:
            j += 1
        block = by_pts[i:j]

        if len(block) == 1:
            out.append(block[0])
        elif len(block) == 2:
            # FIFA 2026: two-way ties skip head-to-head, go to overall
            out.extend(_break_by_overall(block, group_matches, fifa_ranks))
        else:
            out.extend(_break_by_h2h(block, group_matches, fifa_ranks))
        i = j
    return out


def _break_by_h2h(
    tied: Sequence[str],
    all_matches: Sequence[Mapping],
    fifa_ranks: Optional[Mapping[str, int]],
) -> list[str]:
    """Resolve a 3+ team tie via head-to-head, recursing on remaining sub-ties."""
    tied_set = set(tied)
    h2h = _aggregate_stats(
        (m for m in all_matches
         if m["home_team"] in tied_set and m["away_team"] in tied_set),
        tied_set,
    )

    def key(t: str) -> tuple:
        s = h2h[t]
        return (-s["pts"], -s["gd"], -s["gf"])

    sorted_tied = sorted(tied, key=key)
    out: list[str] = []
    i = 0
    while i < len(sorted_tied):
        j = i + 1
        while j < len(sorted_tied) and key(sorted_tied[j]) == key(sorted_tied[i]):
            j += 1
        sub = sorted_tied[i:j]
        if len(sub) == 1:
            out.append(sub[0])
        elif len(sub) == len(tied):
            # H2H made zero progress — fall to overall criteria
            out.extend(_break_by_overall(sub, all_matches, fifa_ranks))
        else:
            # Smaller still-tied subset — re-apply h2h on just that subset
            out.extend(_break_by_h2h(sub, all_matches, fifa_ranks))
        i = j
    return out


def _break_by_overall(
    teams: Sequence[str],
    all_matches: Sequence[Mapping],
    fifa_ranks: Optional[Mapping[str, int]],
) -> list[str]:
    """Final fallback: overall GD, goals, then FIFA world ranking."""
    stats = _aggregate_stats(all_matches, set(teams))

    def key(t: str) -> tuple:
        s = stats[t]
        rank = fifa_ranks.get(t, 999) if fifa_ranks else 999
        # Fair play (criterion 3c) omitted: no booking data in our pipeline.
        return (-s["gd"], -s["gf"], rank)

    return sorted(teams, key=key)


# ----------------------------------------------------------------------
# Third-place team ranking
# ----------------------------------------------------------------------
# Different rules from the group-stage tiebreakers: NO head-to-head, because
# the 12 third-placed teams haven't all played each other.
#   1. Points    2. GD    3. Goals scored    4. Fair play    5. FIFA ranking

def rank_third_place(
    third_place_teams: list[str],
    group_matches_by_letter: Mapping[str, Sequence[Mapping]],
    fifa_ranks: Optional[Mapping[str, int]] = None,
) -> list[str]:
    """Return the 12 third-placed teams ranked best-first; top 8 advance to R32."""
    stats: dict[str, dict[str, int]] = {}
    for t in third_place_teams:
        g = TEAM_TO_GROUP[t]
        # Pass the FULL group team set: _aggregate_stats only counts a match
        # when BOTH teams are in the set, so a single-team set {t} silently
        # zeroes every stat (the team never plays itself) and the best-8
        # selection collapses to group-letter order. See Session DRC-FIX.
        stats[t] = _aggregate_stats(group_matches_by_letter[g], set(GROUPS[g]))[t]

    def key(t: str) -> tuple:
        s = stats[t]
        rank = fifa_ranks.get(t, 999) if fifa_ranks else 999
        return (-s["pts"], -s["gd"], -s["gf"], rank)

    return sorted(third_place_teams, key=key)


# ----------------------------------------------------------------------
# FIFA Annex C: third-place R32 slot resolution
# ----------------------------------------------------------------------
# When 8 of the 12 third-placed teams advance to the R32, FIFA's published
# 495-row Annex C table specifies which group's third-placed team fills each
# of the 8 R32 slots that face a group winner. The table can't be derived
# from the slot-family rules alone — diagnostics on the encoded CSV show
# every Q-set has multiple constraint-valid assignments (median ~16), so
# FIFA's choice is a design decision we have to honor verbatim.
#
# Source: data/raw/r32_annex_c.csv  (built by src/build_annex_c.py)

_ANNEX_C_PATH = Path("data/raw/r32_annex_c.csv")

# The 8 R32 slots that hold a third-placed team, in CSV column order.
# Keyed by the "1X" winner each slot faces.
_THIRD_PLACE_SLOT_IDS: tuple[str, ...] = (
    "1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L",
)

# Cached at first call; re-reading 495 rows on every simulated tournament
# would dominate the Monte Carlo cost.
_ANNEX_C_CACHE: Optional[dict[frozenset[str], dict[str, str]]] = None


def _load_annex_c() -> dict[frozenset[str], dict[str, str]]:
    """Load and cache the 495-row Annex C lookup table.

    Returns
    -------
    dict
        frozenset of 8 group letters (Q-set) -> {slot_id: group_letter}
    """
    global _ANNEX_C_CACHE
    if _ANNEX_C_CACHE is not None:
        return _ANNEX_C_CACHE

    if not _ANNEX_C_PATH.exists():
        raise FileNotFoundError(
            f"{_ANNEX_C_PATH} not found. Run `python src/build_annex_c.py` "
            "from the project root, and run scripts from the project root "
            "(not from inside src/)."
        )

    table: dict[frozenset[str], dict[str, str]] = {}
    with _ANNEX_C_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = frozenset(row[f"Q{i}"] for i in range(1, 9))
            assignment = {
                sid: row[f"slot_{sid}"] for sid in _THIRD_PLACE_SLOT_IDS
            }
            table[key] = assignment

    if len(table) != 495:
        raise RuntimeError(
            f"Annex C: expected 495 rows in {_ANNEX_C_PATH}, got {len(table)}. "
            "Rebuild with `python src/build_annex_c.py`."
        )
    _ANNEX_C_CACHE = table
    return table


def resolve_third_place_slots(
    qualifying_groups: Iterable[str],
) -> dict[str, str]:
    """
    Given the 8 group letters whose third-placed teams qualified for the R32,
    return {slot_id: group_letter} per FIFA Annex C.

    Example
    -------
    >>> resolve_third_place_slots(["E","F","G","H","I","J","K","L"])
    {'1A': 'E', '1B': 'J', '1D': 'I', '1E': 'F',
     '1G': 'H', '1I': 'G', '1K': 'L', '1L': 'K'}

    Reading the example: at slot 1A (Match 79, "1A vs 3CEFHI") the third
    placed team from group E plays. The dict's slot ids are the "1X" winners
    that the third-placed teams face, which is how the published table is
    indexed.

    Raises
    ------
    ValueError
        If `qualifying_groups` doesn't contain exactly 8 distinct letters
        from A-L.
    KeyError
        If the set isn't in Annex C — would only fire if the CSV is incomplete.
    """
    key = frozenset(qualifying_groups)
    if len(key) != 8 or not key.issubset(set("ABCDEFGHIJKL")):
        raise ValueError(
            f"resolve_third_place_slots needs exactly 8 distinct group letters "
            f"from A-L; got {sorted(qualifying_groups)}"
        )
    table = _load_annex_c()
    if key not in table:
        raise KeyError(
            f"Annex C row missing for groups {sorted(key)} — rebuild the CSV."
        )
    return dict(table[key])


# ----------------------------------------------------------------------
# Sanity check (run with `python src/bracket.py`)
# ----------------------------------------------------------------------

def _sanity_check() -> bool:
    fixtures = Path("data/processed/fixtures_2026.csv")
    if not fixtures.exists():
        print(f"⚠️  {fixtures} not found — run from project root.")
        return False

    teams_csv: set[str] = set()
    pairs: list[tuple[str, str]] = []
    with fixtures.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            teams_csv.add(row["home_team"])
            teams_csv.add(row["away_team"])
            pairs.append((row["home_team"], row["away_team"]))

    teams_groups = {t for ts in GROUPS.values() for t in ts}

    extra_csv = teams_csv - teams_groups
    extra_grp = teams_groups - teams_csv
    if extra_csv or extra_grp:
        print("❌ Team-name mismatch:")
        if extra_csv: print(f"   only in CSV:    {sorted(extra_csv)}")
        if extra_grp: print(f"   only in GROUPS: {sorted(extra_grp)}")
        return False
    print(f"✅ 48 team names match between GROUPS and {fixtures.name}")

    # Each group must have exactly 6 round-robin fixtures (4 choose 2)
    for letter, teams in GROUPS.items():
        ts = set(teams)
        n = sum(1 for h, a in pairs if h in ts and a in ts)
        if n != 6:
            print(f"❌ Group {letter}: expected 6 fixtures, found {n}")
            return False
    print("✅ Each group has exactly 6 fixtures (72 total)")

    # Bracket shape
    assert len(R32_BRACKET) == 16
    assert len(R16_BRACKET) == 8
    assert len(QF_BRACKET)  == 4
    assert len(SF_BRACKET)  == 2
    print("✅ Bracket: 16 R32 + 8 R16 + 4 QF + 2 SF + 1 third-place + 1 final")

    # R32 must reference every group winner, every runner-up, and 8 distinct
    # third-place slot families.
    slots = {s for _, a, b in R32_BRACKET for s in (a, b)}
    missing_w = {f"1{x}" for x in "ABCDEFGHIJKL"} - slots
    missing_r = {f"2{x}" for x in "ABCDEFGHIJKL"} - slots
    thirds = {s for s in slots if s.startswith("3")}
    assert not missing_w, f"missing group winners: {missing_w}"
    assert not missing_r, f"missing group runners-up: {missing_r}"
    assert len(thirds) == 8, f"expected 8 third-place slots, got {len(thirds)}"
    print("✅ R32 covers all 12 winners + 12 runners-up + 8 third-place slots")

    # Later-round IDs must reference real earlier-round matches
    r32_ids = {m for m, _, _ in R32_BRACKET}
    r16_ids = {m for m, _, _ in R16_BRACKET}
    qf_ids  = {m for m, _, _ in QF_BRACKET}
    for m, a, b in R16_BRACKET:
        assert a in r32_ids and b in r32_ids, f"R16 {m}: bad source {a},{b}"
    for m, a, b in QF_BRACKET:
        assert a in r16_ids and b in r16_ids, f"QF {m}: bad source {a},{b}"
    for m, a, b in SF_BRACKET:
        assert a in qf_ids and b in qf_ids, f"SF {m}: bad source {a},{b}"
    print("✅ R16/QF/SF source matches all reference earlier rounds correctly")

    _self_test_tiebreakers()
    _self_test_third_place_selection()
    _self_test_annex_c()
    return True


def _self_test_tiebreakers() -> None:
    """
    Hand-built scenario for rank_group:
      A beats B 1-0, B beats C 2-0, C beats A 1-0, all draw 0-0 with D.
      → A, B, C all on 4 pts (1W 1D 1L); D on 3 pts.
      H2H among A/B/C: each has 3 pts; GD is B=+1, A=0, C=-1.
      Expected finishing order: B, A, C, D.
    """
    teams = ["A", "B", "C", "D"]
    m = lambda h, a, hs, as_: {
        "home_team": h, "away_team": a, "home_score": hs, "away_score": as_,
    }
    matches = [
        m("A", "B", 1, 0), m("B", "C", 2, 0), m("C", "A", 1, 0),
        m("A", "D", 0, 0), m("B", "D", 0, 0), m("C", "D", 0, 0),
    ]
    order = rank_group(teams, matches)
    assert order == ["B", "A", "C", "D"], f"tiebreaker self-test failed: {order}"
    print("✅ Tiebreaker self-test (3-way h2h cycle): order is B > A > C > D")


def _self_test_third_place_selection() -> None:
    """
    rank_third_place must order the 12 third-placed teams by FIFA criteria
    (points, then GD, then GF), NOT by group-letter order.

    Regression for Session DRC-FIX: the old code passed a single-team set
    {t} to _aggregate_stats, which zeroes every stat (a team never plays
    itself), so the stable sort fell back to input order — group-letter
    order A..L — and the best-8 selection was ALWAYS groups A-H regardless
    of record.

    Records are built deliberately reverse to letter order: each strong
    third (groups E-L) has >=3 pts, each weak third (groups A-D) has <=2 pts,
    so the top-8 by record is exactly {E,F,G,H,I,J,K,L}. The anchor is the
    live case that exposed the bug: a Group-K third on 4 pts / +1 GD must
    rank above a Group-A third on 1 pt / -4 GD.

    The input `thirds` list is passed in group-letter order A..L (as
    simulate.py builds it), so under the OLD code top-8 would be {A..H} and
    this test fails; under the fix top-8 is {E..L} by record and it passes.
    """
    # letter -> the chosen third team's (t_goals, opp_goals) results vs
    # distinct group-mates, and (in the comment) the implied pts/gd/gf.
    spec: dict[str, list[tuple[int, int]]] = {
        "A": [(0, 0), (0, 4)],          # 1 pt,  -4 gd, 0 gf   <- weak anchor
        "B": [(1, 1), (0, 1)],          # 1 pt,  -1 gd, 1 gf
        "C": [(0, 0), (0, 0)],          # 2 pts,  0 gd, 0 gf
        "D": [(1, 1), (0, 0)],          # 2 pts,  0 gd, 1 gf
        "E": [(2, 1), (0, 2)],          # 3 pts, -1 gd, 2 gf
        "F": [(0, 0), (0, 0), (0, 0)],  # 3 pts,  0 gd, 0 gf
        "G": [(1, 1), (1, 1), (1, 1)],  # 3 pts,  0 gd, 3 gf
        "H": [(1, 0), (0, 1)],          # 3 pts,  0 gd, 1 gf
        "I": [(3, 1)],                  # 3 pts, +2 gd, 3 gf
        "J": [(1, 0)],                  # 3 pts, +1 gd, 1 gf
        "K": [(2, 1), (1, 1)],          # 4 pts, +1 gd, 3 gf   <- strong anchor
        "L": [(2, 0)],                  # 3 pts, +2 gd, 2 gf
    }

    thirds: list[str] = []
    group_matches_by_letter: dict[str, list[dict]] = {}
    for letter in "ABCDEFGHIJKL":            # group-letter order A..L
        teams = GROUPS[letter]
        third = teams[0]                     # any real group member will do
        opponents = teams[1:]
        matches = [
            {"home_team": third, "away_team": opponents[i],
             "home_score": tg, "away_score": og}
            for i, (tg, og) in enumerate(spec[letter])
        ]
        thirds.append(third)
        group_matches_by_letter[letter] = matches

    ranked = rank_third_place(thirds, group_matches_by_letter)
    top8 = set(ranked[:8])
    expected_top8 = {GROUPS[L][0] for L in "EFGHIJKL"}
    assert top8 == expected_top8, (
        "third-place selection picked the wrong 8 — got "
        f"{sorted(top8)}, expected the 8 best by record {sorted(expected_top8)}. "
        "If this is {A..H}, rank_third_place is zeroing stats again "
        "(single-team set bug)."
    )

    # Anchor: the Group-K third (4 pts/+1) must outrank the Group-A third
    # (1 pt/-4), not merely lose to alphabetical order.
    k_third, a_third = GROUPS["K"][0], GROUPS["A"][0]
    assert ranked.index(k_third) < ranked.index(a_third), (
        f"{k_third} (4 pts) should rank above {a_third} (1 pt); "
        f"got order {ranked}"
    )
    print("✅ Third-place selection self-test: top-8 chosen by record "
          "(E-L), not group-letter order (A-H)")


def _self_test_annex_c() -> None:
    """
    Spot-check resolve_third_place_slots against row 1 of Annex C
    (Q = {E,F,G,H,I,J,K,L}, the case where groups A-D's 3rd-place teams are
    eliminated), plus structural checks across all 495 rows.
    """
    # Row 1 from the FIFA-published table:
    expected_row_1 = {
        "1A": "E", "1B": "J", "1D": "I", "1E": "F",
        "1G": "H", "1I": "G", "1K": "L", "1L": "K",
    }
    got = resolve_third_place_slots("EFGHIJKL")
    assert got == expected_row_1, f"Annex C row 1 mismatch: {got}"

    # Slot families from R32_BRACKET (e.g. "1A vs 3CEFHI" → 1A: CEFHI)
    slot_families: dict[str, set[str]] = {}
    for _, a, b in R32_BRACKET:
        winner = next((s for s in (a, b) if s.startswith("1")), None)
        third = next((s for s in (a, b) if s.startswith("3")), None)
        if winner is not None and third is not None:
            slot_families[winner] = set(third[1:])

    # Cross-check every row of Annex C: slot-family membership + no rematch.
    table = _load_annex_c()
    for q_set, assignment in table.items():
        for slot_id, group in assignment.items():
            assert group in slot_families[slot_id], (
                f"Q={sorted(q_set)}: slot {slot_id} got 3{group}, "
                f"not in family {sorted(slot_families[slot_id])}"
            )
            assert group != slot_id[1], (
                f"Q={sorted(q_set)}: slot {slot_id} would be a group rematch"
            )
    print(f"✅ Annex C: 495 rows loaded, row-1 spot-check + all rows respect "
          "slot families and no-rematch")


if __name__ == "__main__":
    _sanity_check()