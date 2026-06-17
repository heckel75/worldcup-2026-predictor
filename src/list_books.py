"""list_books.py — read-only diagnostic.

Scans data/raw/odds_api/**/*.json (the raw Odds API dumps) and reports every
distinct bookmaker / venue that appears: its title + key, how many events it
quotes across all dumps, and which market keys it offers. Each venue is tagged
as a traditional bookmaker, a betting EXCHANGE, or a PREDICTION-MARKET, and the
summary flags whether any lay-side market (*_lay) is present.

Read-only: this script opens files for reading only and never writes, deletes,
or modifies anything. Run from the project root:  python src/list_books.py
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

ODDS_DIR = os.path.join("data", "raw", "odds_api")

# Venue keys (The Odds API bookmaker keys) classified as betting exchanges.
EXCHANGE_KEYS = {
    "betfair",
    "betfair_ex_uk",
    "betfair_ex_eu",
    "betfair_ex_au",
    "betfair_ex_us",
    "matchbook",
    "smarkets",
    "betdaq",
}

# Venue keys classified as prediction markets.
PREDICTION_MARKET_KEYS = {
    "kalshi",
    "polymarket",
    "predictit",
    "manifold",
}

# Substrings used as a fallback when an exact key isn't in the sets above
# (The Odds API sometimes suffixes region/variant, e.g. betfair_ex_uk).
EXCHANGE_HINTS = ("betfair", "matchbook", "smarkets", "betdaq")
PREDICTION_MARKET_HINTS = ("kalshi", "polymarket", "predictit", "manifold")


def classify(key: str) -> str:
    """Return 'EXCHANGE', 'PREDICTION-MARKET', or 'bookmaker' for a venue key."""
    k = key.lower()
    if k in EXCHANGE_KEYS or any(h in k for h in EXCHANGE_HINTS):
        return "EXCHANGE"
    if k in PREDICTION_MARKET_KEYS or any(h in k for h in PREDICTION_MARKET_HINTS):
        return "PREDICTION-MARKET"
    return "bookmaker"


def main() -> None:
    pattern = os.path.join(ODDS_DIR, "**", "*.json")
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f"No JSON files found under {ODDS_DIR!r}. Nothing to scan.")
        return

    # key -> {"title": str, "events": int, "markets": set[str]}
    venues: dict[str, dict] = defaultdict(
        lambda: {"title": "", "events": 0, "markets": set()}
    )
    files_scanned = 0
    files_skipped = 0

    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            files_skipped += 1
            continue
        files_scanned += 1

        # Each populated dump is a list of event dicts; empty dumps are "[]".
        if not isinstance(data, list):
            continue
        for event in data:
            if not isinstance(event, dict):
                continue
            for bk in event.get("bookmakers", []) or []:
                key = bk.get("key")
                if not key:
                    continue
                rec = venues[key]
                rec["title"] = bk.get("title") or rec["title"] or key
                rec["events"] += 1
                for mkt in bk.get("markets", []) or []:
                    mkey = mkt.get("key")
                    if mkey:
                        rec["markets"].add(mkey)

    # ------------------------------------------------------------------ report
    print(f"Scanned {files_scanned} JSON file(s) under {ODDS_DIR}"
          f"{f' ({files_skipped} unreadable, skipped)' if files_skipped else ''}.")
    print()

    if not venues:
        print("No bookmakers/venues found in any dump (all files empty?).")
        return

    # Group venue keys by class for ordered printing.
    by_class: dict[str, list[str]] = defaultdict(list)
    for key in venues:
        by_class[classify(key)].append(key)

    order = ["bookmaker", "EXCHANGE", "PREDICTION-MARKET"]
    for cls in order:
        keys = sorted(by_class.get(cls, []), key=lambda k: venues[k]["events"], reverse=True)
        if not keys:
            continue
        print(f"=== {cls} ({len(keys)}) ===")
        for key in keys:
            rec = venues[key]
            markets = ", ".join(sorted(rec["markets"])) or "(none)"
            print(f"  {rec['title']:<28} [{key:<18}] "
                  f"events={rec['events']:<6} markets: {markets}")
        print()

    # ----------------------------------------------------------------- summary
    total = len(venues)
    n_book = len(by_class.get("bookmaker", []))
    n_exch = len(by_class.get("EXCHANGE", []))
    n_pred = len(by_class.get("PREDICTION-MARKET", []))

    exch_titles = sorted(venues[k]["title"] for k in by_class.get("EXCHANGE", []))
    pred_titles = sorted(venues[k]["title"] for k in by_class.get("PREDICTION-MARKET", []))

    all_markets = sorted({m for rec in venues.values() for m in rec["markets"]})
    lay_markets = [m for m in all_markets if m.endswith("_lay")]

    print("=== SUMMARY ===")
    print(f"  Total distinct venues : {total}")
    print(f"    bookmakers          : {n_book}")
    print(f"    exchanges           : {n_exch}")
    print(f"    prediction-markets  : {n_pred}")
    print(f"  Exchanges present       : {', '.join(exch_titles) if exch_titles else 'none'}")
    print(f"  Prediction-markets      : {', '.join(pred_titles) if pred_titles else 'none'}")
    print(f"  Distinct market keys    : {', '.join(all_markets) if all_markets else 'none'}")
    if lay_markets:
        print(f"  Lay (*_lay) markets     : YES - {', '.join(lay_markets)}")
    else:
        print("  Lay (*_lay) markets     : none found")


if __name__ == "__main__":
    main()
