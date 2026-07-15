"""Derive knockout-stage fixtures from the resolved bracket (Session A-pipeline).

Pure pipeline helper. The model/sim are date-agnostic and KO fixtures aren't
entered anywhere in the data layer, so the published per-match KO dates live here
as a constant — the firewall-safe analog of generate_site's DISPLAY_NAMES /
KO_ROUND_DATES. Both triple_compare and the per-match fetchers consume
derive_ko_fixtures() to turn the bracket's resolved pairs into forecastable /
fetchable fixture rows ({date, home_team, away_team, neutral}).

Keyed by the unordered team pair (frozenset), NOT match_id or slot order — the
bracket gives us the pair, and the §6 rule is "never trust a slot/order." Dates
are added per round as the tournament advances (5a ships R32 exact; R16+ extend
into KO_MATCH_DATES from the daily checklist).

Run self-tests:  python src/ko_fixtures.py
"""
from __future__ import annotations

import datetime as dt

# Published 2026 Round-of-32 schedule. The (a, b) order in each tuple is
# irrelevant — the key is the unordered pair; derive_ko_fixtures takes the
# home/away orientation from the bracket, never from this list.
_R32_DATED_PAIRS: list[tuple[str, str, dt.date]] = [
    ("South Africa", "Canada",                 dt.date(2026, 6, 28)),
    ("Brazil",       "Japan",                   dt.date(2026, 6, 29)),
    ("Germany",      "Paraguay",                dt.date(2026, 6, 29)),
    ("Netherlands",  "Morocco",                 dt.date(2026, 6, 29)),
    ("Ivory Coast",  "Norway",                  dt.date(2026, 6, 30)),
    ("France",       "Sweden",                  dt.date(2026, 6, 30)),
    ("Mexico",       "Ecuador",                 dt.date(2026, 6, 30)),
    ("England",      "DR Congo",                dt.date(2026, 7, 1)),
    ("Belgium",      "Senegal",                 dt.date(2026, 7, 1)),
    ("USA",          "Bosnia and Herzegovina",  dt.date(2026, 7, 1)),
    ("Spain",        "Austria",                 dt.date(2026, 7, 2)),
    ("Portugal",     "Croatia",                 dt.date(2026, 7, 2)),
    ("Switzerland",  "Algeria",                 dt.date(2026, 7, 2)),
    ("Australia",    "Egypt",                   dt.date(2026, 7, 3)),
    ("Argentina",    "Cape Verde",              dt.date(2026, 7, 3)),
    ("Colombia",     "Ghana",                   dt.date(2026, 7, 3)),
]

# Published 2026 Round-of-16 schedule (dated per match slot 89–96). Added once
# the R32 draw resolved the pairings — the "extend per-round" daily-checklist
# step (§5a / §11). Order within each tuple is irrelevant (keyed by frozenset).
_R16_DATED_PAIRS: list[tuple[str, str, dt.date]] = [
    ("Paraguay",     "France",                  dt.date(2026, 7, 4)),   # match 89
    ("Canada",       "Morocco",                 dt.date(2026, 7, 4)),   # match 90
    ("Brazil",       "Norway",                  dt.date(2026, 7, 5)),   # match 91
    ("Mexico",       "England",                 dt.date(2026, 7, 5)),   # match 92
    ("Portugal",     "Spain",                   dt.date(2026, 7, 6)),   # match 93
    ("USA",          "Belgium",                 dt.date(2026, 7, 6)),   # match 94
    ("Argentina",    "Egypt",                   dt.date(2026, 7, 7)),   # match 95
    ("Switzerland",  "Colombia",                dt.date(2026, 7, 7)),   # match 96
]

# Published 2026 Quarter-final schedule (dated per match slot 97–100). Added once
# the R16 results resolved the pairings — the "extend per-round" daily-checklist
# step (§5a / §11). Order within each tuple is irrelevant (keyed by frozenset).
# Argentina–Switzerland shares Jul 11 with Norway–England; the venue-local kickoff
# rolls past midnight in some timezones but it's the same tournament match day.
_QF_DATED_PAIRS: list[tuple[str, str, dt.date]] = [
    ("France",       "Morocco",                 dt.date(2026, 7, 9)),    # match 97
    ("Spain",        "Belgium",                 dt.date(2026, 7, 10)),   # match 99
    ("Norway",       "England",                 dt.date(2026, 7, 11)),   # match 98
    ("Argentina",    "Switzerland",             dt.date(2026, 7, 11)),   # match 100
]

# Published 2026 Semi-final schedule (dated per match slot 101–102). Added once
# the QF results resolved the pairings — the "extend per-round" daily-checklist
# step (§5a / §11). Order within each tuple is irrelevant (keyed by frozenset).
_SF_DATED_PAIRS: list[tuple[str, str, dt.date]] = [
    ("France",       "Spain",                   dt.date(2026, 7, 14)),   # match 101
    ("England",      "Argentina",               dt.date(2026, 7, 15)),   # match 102
]

# Published 2026 Third-place play-off + Final schedule (match slots 103–104).
# Added once the semi-finals resolved the pairings — the "extend per-round"
# daily-checklist step (§5a / §11). The third-place match (node 103) is contested
# by the two SF losers; the final (104) by the two SF winners. Order within each
# tuple is irrelevant (keyed by frozenset).
_FINAL_DATED_PAIRS: list[tuple[str, str, dt.date]] = [
    ("France",       "England",                 dt.date(2026, 7, 18)),   # match 103 (3rd place)
    ("Spain",        "Argentina",               dt.date(2026, 7, 19)),   # match 104 (final)
]

KO_MATCH_DATES: dict[frozenset, dt.date] = {
    frozenset({a, b}): d
    for a, b, d in _R32_DATED_PAIRS + _R16_DATED_PAIRS + _QF_DATED_PAIRS
                   + _SF_DATED_PAIRS + _FINAL_DATED_PAIRS
}


def _ko_fixture_row(m: dict) -> dict | None:
    """A forecastable fixture row for a single resolved-but-unplayed, dated KO
    match dict (a bracket_resolve node), or None when it isn't ready — a slot
    still TBD, the match already played, or its unordered pair has no published
    date in KO_MATCH_DATES yet (a later round — 5a).

    Row shape: {"date": "YYYY-MM-DD", "home_team", "away_team", "neutral": True}
    — internal team names from the bracket (home = team_a, away = team_b), neutral
    True per §6 (KO venues aren't a modelled home advantage; orientation is kept
    only for stable joins).
    """
    a, b = m["team_a"], m["team_b"]
    if not a or not b or m["played"]:
        return None
    d = KO_MATCH_DATES.get(frozenset({a, b}))
    if d is None:
        return None
    return {"date": d.isoformat(), "home_team": a, "away_team": b, "neutral": True}


def derive_ko_fixtures(bracket: dict) -> list[dict]:
    """Forecastable KO fixture rows for the shallowest populated-but-incomplete
    round of `bracket` (a bracket_resolve.resolve_bracket() output dict), plus the
    third-place play-off once its feeders are resolved.

    A match is emitted only when both teams are resolved, it hasn't been played,
    AND its unordered pair has a published date in KO_MATCH_DATES; pairs without a
    date yet (later rounds — 5a) are omitted so no dateless row ever reaches
    match_key. See _ko_fixture_row for the row shape.

    The third-place play-off (node 103) lives in bracket["third_place"], NOT in
    bracket["rounds"], so it is emitted independently: it becomes forecastable the
    moment the two semi-finals are played (its feeders are the SF losers), which
    coincides with the final being the active round. It is kept separate from the
    round loop so it still surfaces if the final is entered before it.

    Returns [] when the bracket isn't complete (group stage unfinished) or nothing
    is currently dated, unplayed, and populated.
    """
    if not bracket.get("complete"):
        return []
    out: list[dict] = []
    for rnd in bracket["rounds"]:
        matches = rnd["matches"]
        populated = any(m["team_a"] and m["team_b"] for m in matches)
        complete = all(m["played"] for m in matches)
        if populated and not complete:
            out = [r for m in matches if (r := _ko_fixture_row(m)) is not None]
            break
    tp = bracket.get("third_place")
    if tp is not None:
        r = _ko_fixture_row(tp)
        if r is not None:
            out.append(r)
    return out


# ----------------------------------------------------------------------
# Self-tests (run: python src/ko_fixtures.py)
# ----------------------------------------------------------------------

def _self_test() -> None:
    assert len(KO_MATCH_DATES) == 32, \
        f"expected 32 KO dates (16 R32 + 8 R16 + 4 QF + 2 SF + 2 final/3rd), " \
        f"got {len(KO_MATCH_DATES)}"

    def m(a, b, played=False, winner=None):
        return {"match_id": 0, "team_a": a, "team_b": b,
                "winner": winner, "loser": None, "played": played}

    # Incomplete bracket -> [].
    assert derive_ko_fixtures({"complete": False}) == []

    # R32: a dated unplayed pair, a played pair (skip), a dated unplayed pair;
    # plus a TBD R16 below. Shallowest incomplete round is R32.
    bracket = {
        "complete": True,
        "rounds": [
            {"label": "Round of 32", "matches": [
                m("South Africa", "Canada"),                        # dated, unplayed
                m("Brazil", "Japan", played=True, winner="Brazil"),  # played -> skip
                m("Germany", "Paraguay"),                            # dated, unplayed
            ]},
            {"label": "Round of 16", "matches": [m(None, None)]},
        ],
        "third_place": None,
    }
    fx = derive_ko_fixtures(bracket)
    assert len(fx) == 2, fx
    for r in fx:
        assert set(r) == {"date", "home_team", "away_team", "neutral"}, r
        assert r["neutral"] is True, r
    sa = next(r for r in fx if r["home_team"] == "South Africa")
    assert sa["away_team"] == "Canada" and sa["date"] == "2026-06-28", sa  # orientation = team_a/team_b
    ge = next(r for r in fx if r["home_team"] == "Germany")
    assert ge["away_team"] == "Paraguay" and ge["date"] == "2026-06-29", ge

    # Shallowest incomplete round is R16 (R32 done) but R16 has no published
    # date yet -> [] (no dateless rows).
    bracket2 = {
        "complete": True,
        "rounds": [
            {"label": "Round of 32", "matches": [
                m("South Africa", "Canada", played=True, winner="Canada")]},
            {"label": "Round of 16", "matches": [m("Canada", "Brazil")]},  # undated
        ],
        "third_place": None,
    }
    assert derive_ko_fixtures(bracket2) == [], derive_ko_fixtures(bracket2)

    # Final active + third-place play-off both emitted. SFs are complete (so the
    # Final is the shallowest incomplete round in `rounds`), and the third-place
    # node — which is NOT in `rounds` — is appended independently.
    bracket3 = {
        "complete": True,
        "rounds": [
            {"label": "Semi-finals", "matches": [
                m("France", "Spain", played=True, winner="Spain"),
                m("England", "Argentina", played=True, winner="Argentina")]},
            {"label": "Final", "matches": [m("Spain", "Argentina")]},
        ],
        "third_place": m("France", "England"),
    }
    fx3 = derive_ko_fixtures(bracket3)
    assert len(fx3) == 2, fx3
    fin = next(r for r in fx3
               if {r["home_team"], r["away_team"]} == {"Spain", "Argentina"})
    assert fin["date"] == "2026-07-19", fin
    tpr = next(r for r in fx3
               if {r["home_team"], r["away_team"]} == {"France", "England"})
    assert tpr["date"] == "2026-07-18", tpr

    print("ko_fixtures.py self-test passed "
          "(32 KO dates = 16 R32 + 8 R16 + 4 QF + 2 SF + 2 final/3rd; shape; "
          "skip-played; omit-undated; bracket orientation; final+3rd-place)")


if __name__ == "__main__":
    _self_test()
