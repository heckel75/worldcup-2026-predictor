"""
2026 FIFA World Cup bracket: groups, knockout structure, tiebreakers.

Sources
-------
- Groups & bracket: FIFA Final Draw, Dec 5 2025.
  Wikipedia: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage
- Tiebreakers: FIFA 2026 regulations (via ESPN summary).

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

from collections.abc import Iterable, Mapping, Sequence
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
#               which 8 third-placed teams advance; Session 16 resolves it)

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
        stats[t] = _aggregate_stats(group_matches_by_letter[g], {t})[t]

    def key(t: str) -> tuple:
        s = stats[t]
        rank = fifa_ranks.get(t, 999) if fifa_ranks else 999
        return (-s["pts"], -s["gd"], -s["gf"], rank)

    return sorted(third_place_teams, key=key)


# ----------------------------------------------------------------------
# Sanity check (run with `python src/bracket.py`)
# ----------------------------------------------------------------------

def _sanity_check() -> bool:
    import csv
    from pathlib import Path

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


if __name__ == "__main__":
    _sanity_check()