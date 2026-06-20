"""Shared "who was closer" verdict logic — the single source of truth.

Extracted verbatim from generate_site._verdict (Session DIVLOG-1) so both the
index scoreboard (generate_site.py) and the Divergence Log attribution
scoreboard (divergence_log.py) tally against ONE implementation. Never
reimplement this — the verdict logic has been corrected twice (Session 37
follow-up, 2026-06-20) and the no-credit contract for "wash"/"all_missed"/
"markets" is load-bearing for the public scoreboard.

Pure module: depends only on pandas (for NaN handling). _OUTCOME_SLOT and
_pct0 are private copies, deliberately duplicated from generate_site so this
module imports nothing from the site layer.
"""
from __future__ import annotations

import pandas as pd

# Canonical verdict buckets. Both consumers tally against this one set, so a
# new branch can't silently desync the two scoreboards. "model"/"books"/
# "Polymarket" are CREDITED (a distinct source won); "markets"/"wash"/
# "all_missed" are NO-CREDIT buckets — they record the outcome without
# advancing any single source's win count.
VERDICT_BUCKETS = ("model", "books", "Polymarket", "markets", "wash", "all_missed")

_OUTCOME_SLOT = {"H": "home", "D": "draw", "A": "away"}


def _pct0(p: float) -> str:
    """Probability -> whole-percent string (match bars use integers)."""
    return f"{round(p * 100)}%"


def _verdict(rec: dict) -> dict | None:
    """Which source's frozen pre-match forecast was closest to what happened.

    A source "participates" if it froze all three probs; it "picked" the
    outcome if its OWN argmax over (home, draw, away) equals the realized
    result. Four ordered branches (after the <2-participant guard):
      1. Fewer than two participating sources -> no contest, None.
      2. No participating source picked the outcome -> "all_missed" (everyone
         forecast the wrong result — an upset nobody saw), credits nobody.
      3. Every participating source picked the outcome AND the top-two
         on-outcome probs are within 2pp -> "wash" (sources agreed), credits
         nobody. Tightened from Session 37: now requires ALL to have picked it,
         so a 2-of-3 where the model missed no longer launders into a wash.
      4. The two highest on-outcome probs are within 2pp of each other and BOTH
         are markets (books + Polymarket), >2pp clear of the model -> "markets"
         (the markets jointly beat the model). Credits NEITHER market
         individually — counted in its own scoreboard bucket, no books/Poly
         column bump (Session 37 "distinct wins only"). This is the USA-Australia
         case the old logic mislabelled "sources agreed".
      5. Otherwise the single source with the highest on-outcome prob, separated
         by >2pp from the next -> "{source} closest", credits that source.

    A bunched residual that none of 2-4 resolve (e.g. only one market present,
    or the model tied with a single market) is too close to separate -> "wash",
    credits nobody. Only a distinct source winner ("model"/"books"/"Polymarket")
    credits the index scoreboard; "wash", "all_missed" and "markets" credit
    nobody."""
    slot = _OUTCOME_SLOT[rec["outcome"]]
    slots = ("home", "draw", "away")
    # (name, prob_on_outcome, picked_outcome)
    entries = []
    for name, tmpl in (("model", "p_{}"),
                       ("books", "p_{}_book"),
                       ("Polymarket", "p_{}_poly")):
        vals = {s: rec.get(tmpl.format(s)) for s in slots}
        if any(pd.isna(v) for v in vals.values()):
            continue  # a source freezes all three probs or none
        probs = {s: float(v) for s, v in vals.items()}
        argmax = max(slots, key=lambda s: probs[s])
        entries.append((name, probs[slot], argmax == slot))
    if len(entries) < 2:                                          # branch 1
        return None

    parts = [f"{'Model gave this result' if name == 'model' else name} {_pct0(p)}"
             for name, p, _ in entries]
    body = ", ".join(parts)

    if not any(picked for _, _, picked in entries):              # branch 2
        return {"winner": "all_missed", "text": f"{body} — all sources missed."}

    ranked = sorted(entries, key=lambda e: e[1], reverse=True)
    bunched = (ranked[0][1] - ranked[1][1]) <= 0.02

    if all(picked for _, _, picked in entries) and bunched:      # branch 3
        return {"winner": "wash", "text": f"{body} — sources agreed."}

    markets = {"books", "Polymarket"}
    if (bunched and len(entries) >= 3                            # branch 4
            and ranked[0][0] in markets and ranked[1][0] in markets
            and (ranked[1][1] - ranked[2][1]) > 0.02):
        return {"winner": "markets", "text": f"{body} — markets closest."}

    if not bunched:                                              # branch 5
        return {"winner": ranked[0][0], "text": f"{body} — {ranked[0][0]} closest."}

    # Bunched residual nothing above resolved: credit nobody.
    return {"winner": "wash", "text": f"{body} — sources agreed."}


# ----------------------------------------------------------------------
# Self-test (run: python src/verdict.py --test). verdict.py owns these cases;
# generate_site.py and divergence_log.py both import _verdict, so pinning the
# logic here guards both consumers.
# ----------------------------------------------------------------------

def _test_verdict() -> None:
    nan = float("nan")

    # 1. All sources favoured a team that didn't win -> all_missed.
    r = {"outcome": "H",
         "p_home": 0.20, "p_draw": 0.30, "p_away": 0.50,
         "p_home_book": 0.25, "p_draw_book": 0.30, "p_away_book": 0.45,
         "p_home_poly": 0.25, "p_draw_poly": 0.30, "p_away_poly": 0.45}
    assert _verdict(r)["winner"] == "all_missed", _verdict(r)

    # 2. All three correctly favoured home; top two within 2pp -> wash.
    r = {"outcome": "H",
         "p_home": 0.92, "p_draw": 0.05, "p_away": 0.03,
         "p_home_book": 0.92, "p_draw_book": 0.05, "p_away_book": 0.03,
         "p_home_poly": 0.93, "p_draw_poly": 0.04, "p_away_poly": 0.03}
    assert _verdict(r)["winner"] == "wash", _verdict(r)

    # 3. Model highest-correct by >2pp, its own argmax is home -> model.
    r = {"outcome": "H",
         "p_home": 0.55, "p_draw": 0.25, "p_away": 0.20,
         "p_home_book": 0.40, "p_draw_book": 0.35, "p_away_book": 0.25,
         "p_home_poly": 0.41, "p_draw_poly": 0.34, "p_away_poly": 0.25}
    assert _verdict(r)["winner"] == "model", _verdict(r)

    # 4. Fewer than two participating sources -> no contest.
    r = {"outcome": "H",
         "p_home": 0.50, "p_draw": 0.30, "p_away": 0.20,
         "p_home_book": nan, "p_draw_book": nan, "p_away_book": nan,
         "p_home_poly": nan, "p_draw_poly": nan, "p_away_poly": nan}
    assert _verdict(r) is None, _verdict(r)

    # 5. USA-Australia shape: both markets right and bunched, model missed
    #    (its argmax is away) -> markets, no individual credit.
    r = {"outcome": "H",
         "p_home": 0.35, "p_draw": 0.20, "p_away": 0.45,
         "p_home_book": 0.55, "p_draw_book": 0.25, "p_away_book": 0.20,
         "p_home_poly": 0.54, "p_draw_poly": 0.26, "p_away_poly": 0.20}
    assert _verdict(r)["winner"] == "markets", _verdict(r)

    # 6. Markets right but one clearly higher than the other (>2pp), model
    #    missed -> that market closest, credited.
    r = {"outcome": "H",
         "p_home": 0.35, "p_draw": 0.20, "p_away": 0.45,
         "p_home_book": 0.55, "p_draw_book": 0.25, "p_away_book": 0.20,
         "p_home_poly": 0.50, "p_draw_poly": 0.30, "p_away_poly": 0.20}
    assert _verdict(r)["winner"] == "books", _verdict(r)

    print("verdict.py _verdict self-tests passed")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _test_verdict()
