"""
src/fetch_polymarket.py

Session 19: pull WC 2026 probabilities from the Polymarket Gamma API.

Three outputs, mirroring Session 18's sportsbook fetcher:
  data/processed/polymarket_outrights.csv  — title winner, one row per team
  data/processed/polymarket_groups.csv     — group winner, one row per team
  data/processed/polymarket_odds.csv       — per-match (header-only for now;
                                              Polymarket has not posted per-match
                                              markets as of 2026-05-17)

Spread (over-round) stripping: each multi-outcome event is N binary Yes/No
child markets. Sum of Yes prices across the event is slightly > 1 because of
bid/ask spread. We sum, divide by the sum to get fair probabilities — same
idea as the proportional vig-strip in fetch_odds.py.

Gamma is public. No API key. Generous rate limits.

Run from project root:
    python src/fetch_polymarket.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# --- config --------------------------------------------------------------

GAMMA = "https://gamma-api.polymarket.com"

# Slugs discovered in the probe. /tags doesn't surface them, so we look up by
# exact slug. If a slug 404s the script reports it and continues.
TITLE_EVENT_SLUG = "2026-fifa-world-cup-winner-595"
GROUP_LETTERS = list("abcdefghijkl")

PROCESSED_DIR = Path("data/processed")
RAW_DUMP_DIR = Path("data/raw/polymarket")
OUTRIGHTS_PATH = PROCESSED_DIR / "polymarket_outrights.csv"
GROUPS_PATH = PROCESSED_DIR / "polymarket_groups.csv"
MATCHES_PATH = PROCESSED_DIR / "polymarket_odds.csv"
FIXTURES_PATH = PROCESSED_DIR / "fixtures_2026.csv"

# Polymarket-name → our internal name. Mirrors fetch_odds.py's map.
# Extend based on the diagnostic at the end if needed.
POLYMARKET_TEAM_MAP: dict[str, str] = {
    "United States": "USA",
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Democratic Republic of Congo": "DR Congo",
    "Congo DR": "DR Congo",
}

# Pull the team name out of the market question text.
TITLE_QUESTION = re.compile(
    r"^Will\s+(.+?)\s+win\s+(?:the\s+)?(?:2026\s+)?FIFA\s+World\s+Cup(?:\s+2026)?\??$",
    re.I,
)
GROUP_QUESTION = re.compile(
    r"^Will\s+(?:the\s+)?(.+?)\s+win\s+Group\s+[A-L]\b.*\??$",
    re.I,
)


# --- API plumbing --------------------------------------------------------

def fetch_event(slug: str) -> dict | None:
    """Hit /events?slug=<slug>. Returns the event dict (markets nested) or None."""
    r = requests.get(f"{GAMMA}/events", params={"slug": slug}, timeout=30)
    print(f"  GET {r.url}  HTTP {r.status_code}")
    if not r.ok:
        return None
    data = r.json()
    return data[0] if data else None


# --- parsing -------------------------------------------------------------

def _parse_jsonlike(value: Any) -> Any:
    """outcomes/outcomePrices ship as JSON-encoded strings inside the JSON."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def yes_price_and_volume(market: dict) -> tuple[float | None, float]:
    """Polymarket child markets are binary. Return (yes_price, volume)."""
    outcomes = _parse_jsonlike(market.get("outcomes")) or []
    prices = _parse_jsonlike(market.get("outcomePrices")) or []
    volume = float(market.get("volume") or 0)
    if len(outcomes) != len(prices):
        return None, volume
    for label, price in zip(outcomes, prices):
        if str(label).strip().lower() == "yes":
            try:
                return float(price), volume
            except (TypeError, ValueError):
                return None, volume
    return None, volume


def normalise_team(name: str) -> str:
    return POLYMARKET_TEAM_MAP.get(name, name).strip()


def extract_team(question: str, pattern: re.Pattern) -> str | None:
    m = pattern.match((question or "").strip())
    return m.group(1).strip() if m else None


def parse_title_event(event: dict | None) -> pd.DataFrame:
    if not event:
        return pd.DataFrame(columns=["team", "p_winner_raw", "volume"])
    rows, skipped = [], []
    for market in event.get("markets") or []:
        question = market.get("question") or ""
        team = extract_team(question, TITLE_QUESTION)
        if not team:
            skipped.append(question)
            continue
        yes, volume = yes_price_and_volume(market)
        if yes is None:
            continue
        rows.append({
            "team": normalise_team(team),
            "p_winner_raw": yes,
            "volume": volume,
        })
    if skipped:
        print(f"    skipped {len(skipped)} unparsed title market(s); first few:")
        for q in skipped[:3]:
            print(f"      - {q!r}")
    return pd.DataFrame(rows)


def parse_group_event(letter: str, event: dict | None) -> pd.DataFrame:
    if not event:
        return pd.DataFrame(columns=["group", "team", "p_winner_raw", "volume"])
    rows, skipped = [], []
    for market in event.get("markets") or []:
        question = market.get("question") or ""
        team = extract_team(question, GROUP_QUESTION)
        if not team:
            skipped.append(question)
            continue
        yes, volume = yes_price_and_volume(market)
        if yes is None:
            continue
        rows.append({
            "group": letter.upper(),
            "team": normalise_team(team),
            "p_winner_raw": yes,
            "volume": volume,
        })
    if skipped:
        print(f"    group {letter.upper()}: skipped {len(skipped)} unparsed; first:")
        for q in skipped[:2]:
            print(f"      - {q!r}")
    return pd.DataFrame(rows)


def renormalise(df: pd.DataFrame) -> pd.DataFrame:
    """Divide p_winner_raw by its sum, round, drop the raw column."""
    out = df.copy()
    total = out["p_winner_raw"].sum()
    out["p_winner"] = (out["p_winner_raw"] / total).round(4) if total > 0 else 0.0
    return out.drop(columns=["p_winner_raw"])


# --- main ----------------------------------------------------------------

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    qualified: set[str] = set()
    if FIXTURES_PATH.exists():
        fx = pd.read_csv(FIXTURES_PATH)
        qualified = set(fx["home_team"]) | set(fx["away_team"])

    # --- title winner ---
    print(f"\n=== Title winner: {TITLE_EVENT_SLUG} ===")
    title_event = fetch_event(TITLE_EVENT_SLUG)
    if title_event:
        (RAW_DUMP_DIR / f"title_{ts}.json").write_text(json.dumps(title_event, indent=2))
    df_title_raw = parse_title_event(title_event)
    print(f"  parsed {len(df_title_raw)} title markets")

    # --- group winners ---
    print(f"\n=== Group winners ===")
    group_frames = []
    for letter in GROUP_LETTERS:
        slug = f"fifa-world-cup-group-{letter}-winner"
        ev = fetch_event(slug)
        if ev:
            (RAW_DUMP_DIR / f"group_{letter}_{ts}.json").write_text(json.dumps(ev, indent=2))
        group_frames.append(parse_group_event(letter, ev))
    df_groups_raw = (pd.concat(group_frames, ignore_index=True)
                     if group_frames else pd.DataFrame())

    # --- name diagnostic (run BEFORE filtering, so mismatches surface) ---
    if qualified:
        seen: set[str] = set()
        if not df_title_raw.empty:
            seen.update(df_title_raw["team"].tolist())
        if not df_groups_raw.empty:
            seen.update(df_groups_raw["team"].tolist())
        unmatched = sorted(seen - qualified)
        if unmatched:
            print(f"\n!! {len(unmatched)} Polymarket team name(s) not in fixtures_2026.csv:")
            for t in unmatched:
                print(f"   {t!r}  -> non-qualified, or add to POLYMARKET_TEAM_MAP")
        missing = sorted(qualified - seen)
        if missing:
            print(f"\n!! {len(missing)} qualified team(s) NOT found on Polymarket:")
            for t in missing:
                print(f"   {t!r}  -> name-mapping issue, or no market exists")
        if not unmatched and not missing:
            print("\nAll 48 qualified team names map to Polymarket markets. ✓")

    # --- filter to qualified, renormalise, save title ---
    if not df_title_raw.empty:
        df_title = (df_title_raw[df_title_raw["team"].isin(qualified)].copy()
                    if qualified else df_title_raw.copy())
        n_dropped = len(df_title_raw) - len(df_title)
        if n_dropped:
            print(f"\nTitle: dropped {n_dropped} non-qualified team(s) before renormalising.")
        if not df_title.empty:
            df_title = renormalise(df_title)
            df_title["volume"] = df_title["volume"].round(0).astype(int)
            df_title = (df_title[["team", "p_winner", "volume"]]
                          .sort_values("p_winner", ascending=False)
                          .reset_index(drop=True))
            df_title.to_csv(OUTRIGHTS_PATH, index=False)
            print(f"\nSaved {len(df_title)} teams -> {OUTRIGHTS_PATH}")
            print(df_title.head(10).to_string(index=False))

    # --- groups: renormalise per group (each event is its own prob space) ---
    if not df_groups_raw.empty:
        per_group = []
        for letter, frame in df_groups_raw.groupby("group", sort=True):
            f = (frame[frame["team"].isin(qualified)].copy()
                 if qualified else frame.copy())
            if f.empty:
                continue
            f = renormalise(f)
            per_group.append(f)
        df_groups = pd.concat(per_group, ignore_index=True) if per_group else pd.DataFrame()
        if not df_groups.empty:
            df_groups["volume"] = df_groups["volume"].round(0).astype(int)
            df_groups = (df_groups[["group", "team", "p_winner", "volume"]]
                           .sort_values(["group", "p_winner"], ascending=[True, False])
                           .reset_index(drop=True))
            df_groups.to_csv(GROUPS_PATH, index=False)
            print(f"\nSaved {len(df_groups)} rows over "
                  f"{df_groups['group'].nunique()} groups -> {GROUPS_PATH}")
            print(df_groups.to_string(index=False))

    # --- per-match (header-only) ---
    print(f"\n=== Per-match ===")
    print("  No per-match WC markets on Polymarket as of 2026-05-17.")
    print("  Writing header-only file so Session 20 has a stable join schema.")
    empty = pd.DataFrame(columns=[
        "commence_time", "home_team", "away_team",
        "p_home", "p_draw", "p_away", "volume", "low_liquidity",
    ])
    empty.to_csv(MATCHES_PATH, index=False)
    print(f"  saved -> {MATCHES_PATH}")


if __name__ == "__main__":
    main()