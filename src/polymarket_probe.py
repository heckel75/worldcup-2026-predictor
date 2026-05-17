"""
src/polymarket_probe.py

Session 19, step 1: discovery probe. Find out what WC 2026 markets exist on
Polymarket so we can build fetch_polymarket.py against real data, not guesses.

No API key — Gamma is public.

Run from project root:
    python src/polymarket_probe.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

GAMMA = "https://gamma-api.polymarket.com"
RAW_DUMP_DIR = Path("data/raw/polymarket")

# Cast a wide net — we eyeball the printout and tighten in step 2.
KEYWORDS = ("world cup", "fifa", "wc 2026", "wc2026", "wc-2026", "soccer")


def get(path: str, params: dict | None = None) -> Any:
    """Thin GET wrapper. Returns parsed JSON. Raises on non-2xx."""
    r = requests.get(f"{GAMMA}{path}", params=params or {}, timeout=30)
    print(f"  GET {r.url}  HTTP {r.status_code}")
    r.raise_for_status()
    return r.json()


def looks_wc_related(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in KEYWORDS)


def find_wc_tags() -> list[dict]:
    """Page through /tags and keep ones with WC-flavoured labels or slugs."""
    print("\n--- step 1: list tags, keep WC-related ---")
    matches: list[dict] = []
    offset = 0
    page = 500
    while True:
        batch = get("/tags", {"limit": page, "offset": offset})
        if not batch:
            break
        for tag in batch:
            label = tag.get("label") or ""
            slug = tag.get("slug") or ""
            if looks_wc_related(label) or looks_wc_related(slug):
                matches.append(tag)
        if len(batch) < page:
            break
        offset += page

    for t in matches:
        tid = str(t.get("id"))
        slug = str(t.get("slug") or "")
        label = t.get("label") or ""
        print(f"  id={tid:<6}  slug={slug:<35}  label={label}")
    return matches


def events_for_tag(tag_id: Any) -> list[dict]:
    """Open events tagged with tag_id."""
    out: list[dict] = []
    offset = 0
    page = 100
    while True:
        batch = get("/events", {
            "tag_id": tag_id,
            "closed": "false",
            "limit": page,
            "offset": offset,
        })
        if not batch:
            break
        out.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return out


def _parse_jsonlike(value: Any) -> Any:
    """outcomes/outcomePrices come back as JSON-encoded strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def summarise_event(event: dict) -> None:
    markets = event.get("markets") or []
    total_volume = sum(float(m.get("volume") or 0) for m in markets)
    print(f"\n  Event: {event.get('title')!r}")
    print(f"    slug={event.get('slug')}  markets={len(markets)}  "
          f"total volume={total_volume:,.0f}")
    for m in markets[:3]:
        q = m.get("question")
        outcomes = _parse_jsonlike(m.get("outcomes"))
        prices = _parse_jsonlike(m.get("outcomePrices"))
        vol = float(m.get("volume") or 0)
        print(f"      - {q!r}")
        print(f"        outcomes={outcomes}  prices={prices}  vol={vol:,.0f}")


def main() -> None:
    RAW_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    tags = find_wc_tags()
    if not tags:
        print("\nNo WC-related tags found. Widen KEYWORDS and re-run.")
        return

    print(f"\n--- step 2: fetch events for {len(tags)} tag(s) ---")
    all_events_by_tag: dict[str, list[dict]] = {}
    for tag in tags:
        tid = tag.get("id")
        slug = str(tag.get("slug") or "")
        print(f"\n[tag {tid} / {slug}]")
        events = events_for_tag(tid)
        print(f"  -> {len(events)} open event(s)")
        all_events_by_tag[slug] = events
        for ev in events:
            summarise_event(ev)

    raw_path = RAW_DUMP_DIR / f"probe_{ts}.json"
    raw_path.write_text(json.dumps({
        "fetched_at": ts,
        "tags": tags,
        "events_by_tag_slug": all_events_by_tag,
    }, indent=2))
    print(f"\n--- saved raw probe to {raw_path} ---")
    print("Eyeball the output above. Next step picks:")
    print("  (a) the event slug holding the title-winner sub-markets, and")
    print("  (b) where (if anywhere) per-match markets live.")


if __name__ == "__main__":
    main()