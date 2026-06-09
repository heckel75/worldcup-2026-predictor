"""
src/fetch_odds.py

Session 18: pull WC 2026 sportsbook odds from The Odds API, strip the vig,
save fair probabilities to data/processed/.

Vig stripping: a book's decimal odds (e.g. 2.50 / 3.40 / 3.00) imply
probabilities that sum to >1 because the book bakes in a margin. We convert
each book's quote to implied probs, normalise each book to sum to 1
(proportional method), then average across books and renormalise once more.

Output:
  data/processed/sportsbook_odds.csv        one row per match (h2h)
  data/processed/sportsbook_outrights.csv   one row per team (title)
  data/raw/odds_api/<key>_<market>_<ts>.json  raw API responses (gitignored)

Run from project root:
  python src/fetch_odds.py

Requires ODDS_API_KEY in .env (get one free at https://the-odds-api.com).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# --- config --------------------------------------------------------------

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
REGIONS = "eu,us,uk"          # combined => more books => tighter averages
ODDS_FORMAT = "decimal"
DATE_FORMAT = "iso"

PROCESSED_DIR = Path("data/processed")
RAW_DUMP_DIR = Path("data/raw/odds_api")
MATCHES_PATH = PROCESSED_DIR / "sportsbook_odds.csv"
OUTRIGHTS_PATH = PROCESSED_DIR / "sportsbook_outrights.csv"
FIXTURES_PATH = PROCESSED_DIR / "fixtures_2026.csv"

# Sportsbook-name → our internal name. Internal names follow the Kaggle
# dataset conventions (see TEAM_NAME_MAP in clean_data.py). Start with the
# usual suspects; the script prints unmatched names at the end so we can
# extend this map and re-run.
SPORTSBOOK_TEAM_MAP: dict[str, str] = {
    "United States": "USA",
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
}


# --- API plumbing --------------------------------------------------------

def get_api_key() -> str:
    load_dotenv()
    key = os.getenv("ODDS_API_KEY")
    if not key:
        sys.exit("ODDS_API_KEY not found. Add it to .env in the project root.")
    return key


def list_world_cup_sports(api_key: str) -> list[dict]:
    """Sport-listing endpoint is FREE (doesn't count against quota).
    Filter to anything WC-related so we don't hardcode keys we can't see yet."""
    r = requests.get(
        f"{ODDS_API_BASE}/sports",
        params={"all": "true", "apiKey": api_key},
        timeout=20,
    )
    r.raise_for_status()
    return [
        s for s in r.json()
        if "world_cup" in s["key"].lower() and "soccer" in s["key"].lower()
    ]


def fetch_odds(api_key: str, sport_key: str, markets: str) -> list[dict]:
    """One paid call. Returns event list (possibly empty)."""
    r = requests.get(
        f"{ODDS_API_BASE}/sports/{sport_key}/odds",
        params={
            "apiKey": api_key,
            "regions": REGIONS,
            "markets": markets,
            "oddsFormat": ODDS_FORMAT,
            "dateFormat": DATE_FORMAT,
        },
        timeout=30,
    )
    used = r.headers.get("x-requests-used", "?")
    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"  [{sport_key} / {markets}] HTTP {r.status_code}  "
          f"quota used={used} remaining={remaining}")
    if r.status_code == 422:
        # market not offered on this sport key — non-fatal
        return []
    r.raise_for_status()
    return r.json()


# --- vig stripping -------------------------------------------------------

def strip_vig_proportional(odds: dict[str, float]) -> dict[str, float]:
    """{outcome: decimal_odds} -> {outcome: fair_probability}, sum=1."""
    implied = {k: 1.0 / v for k, v in odds.items() if v and v > 1.0}
    s = sum(implied.values())
    if s <= 0:
        return {}
    return {k: p / s for k, p in implied.items()}


def average_books(per_book_probs: list[dict[str, float]]) -> dict[str, float]:
    """Average vig-stripped probs across books, then renormalise."""
    if not per_book_probs:
        return {}
    keys = set().union(*(p.keys() for p in per_book_probs))
    out = {k: sum(p.get(k, 0.0) for p in per_book_probs) / len(per_book_probs)
           for k in keys}
    s = sum(out.values())
    return {k: v / s for k, v in out.items()} if s > 0 else out


# --- per-event parsing ---------------------------------------------------

def normalise(team: str) -> str:
    return SPORTSBOOK_TEAM_MAP.get(team, team)


def parse_h2h_event(event: dict) -> dict | None:
    """One match -> one row with averaged W/D/L probabilities."""
    home = event.get("home_team")
    away = event.get("away_team")
    if not home or not away:
        return None

    per_book = []
    for book in event.get("bookmakers", []):
        for mkt in book.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            odds = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
            stripped = strip_vig_proportional(odds)
            if stripped:
                per_book.append(stripped)

    if not per_book:
        return None

    avg = average_books(per_book)
    return {
        "commence_time": event.get("commence_time"),
        "home_team": normalise(home),
        "away_team": normalise(away),
        "p_home": round(avg.get(home, 0.0), 4),
        "p_draw": round(avg.get("Draw", 0.0), 4),
        "p_away": round(avg.get(away, 0.0), 4),
        "n_books": len(per_book),
    }


def parse_outright_events(events: list[dict]) -> pd.DataFrame:
    """Futures markets: one event with N team outcomes. Aggregate across books."""
    rows = []
    for event in events:
        per_book = []
        for book in event.get("bookmakers", []):
            for mkt in book.get("markets", []):
                if mkt.get("key") != "outrights":
                    continue
                odds = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
                stripped = strip_vig_proportional(odds)
                if stripped:
                    per_book.append(stripped)
        if not per_book:
            continue
        avg = average_books(per_book)
        for team, p in avg.items():
            rows.append({
                "team": normalise(team),
                "p_winner": p,
                "n_books": len(per_book),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Defensive: collapse duplicates across multiple WC-related sport keys.
    df = (df.groupby("team", as_index=False)
            .agg(p_winner=("p_winner", "mean"), n_books=("n_books", "max")))
    # Renormalise after any cross-event averaging.
    df["p_winner"] = df["p_winner"] / df["p_winner"].sum()
    df["p_winner"] = df["p_winner"].round(4)
    return df.sort_values("p_winner", ascending=False).reset_index(drop=True)


# --- main ----------------------------------------------------------------

def main():
    api_key = get_api_key()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Load fixture list up-front: used for Fix 1 (orient) and Fix 2 (drop).
    fixture_lookup: dict[frozenset, tuple[str, str]] = {}
    qualified: set[str] = set()
    if FIXTURES_PATH.exists():
        fx_df = pd.read_csv(FIXTURES_PATH)
        qualified = set(fx_df["home_team"]) | set(fx_df["away_team"])
        for _, r in fx_df.iterrows():
            pair: frozenset = frozenset((r["home_team"], r["away_team"]))
            fixture_lookup[pair] = (r["home_team"], r["away_team"])

    print("Discovering WC-related soccer sport keys (free call)...")
    wc_sports = list_world_cup_sports(api_key)
    for s in wc_sports:
        print(f"  {s['key']:40s} active={s['active']}  {s['title']}")
    if not wc_sports:
        sys.exit("No WC-related sport keys returned. Aborting.")

    # --- matches (h2h) ---
    print("\n=== Match odds (h2h) ===")
    all_match_rows: list[dict] = []
    for s in wc_sports:
        if "winner" in s["key"]:        # futures live elsewhere
            continue
        events = fetch_odds(api_key, s["key"], markets="h2h")
        (RAW_DUMP_DIR / f"{s['key']}_h2h_{ts}.json").write_text(
            json.dumps(events, indent=2)
        )
        print(f"  {len(events)} events")
        for ev in events:
            row = parse_h2h_event(ev)
            if row:
                all_match_rows.append(row)

    # Fix 1 (orient) + Fix 2 (drop): match each row to fixtures_2026.csv on
    # the unordered team pair, orient to the fixture's home/away convention,
    # and drop any row whose pair doesn't appear in the fixtures at all.
    # Sportsbook h2h is 2-way (no draw), so only p_home/p_away are swapped.
    if fixture_lookup:
        oriented: list[dict] = []
        for row in all_match_rows:
            pair = frozenset((row["home_team"], row["away_team"]))
            fx_orientation = fixture_lookup.get(pair)
            if fx_orientation is None:
                print(f"  DROPPED: {row['home_team']} vs {row['away_team']} — not in fixtures")
                continue
            fx_home, fx_away = fx_orientation
            if (row["home_team"], row["away_team"]) != (fx_home, fx_away):
                print(f"  FLIPPED: {row['home_team']}/{row['away_team']} → {fx_home}/{fx_away}")
                row = dict(row)
                row["home_team"], row["away_team"] = fx_home, fx_away
                row["p_home"], row["p_away"] = row["p_away"], row["p_home"]
            oriented.append(row)
        all_match_rows = oriented

    if all_match_rows:
        df_m = (pd.DataFrame(all_match_rows)
                  .sort_values("commence_time")
                  .reset_index(drop=True))
        df_m.to_csv(MATCHES_PATH, index=False)
        print(f"\nSaved {len(df_m)} matches -> {MATCHES_PATH}")
        print(df_m.head(10).to_string(index=False))
    else:
        print("\nNo match-level odds returned. Books may not have posted full")
        print("h2h markets yet; outright (title) odds usually appear first.")

    # --- outrights ---
    print("\n=== Outright odds (title winner) ===")
    all_outright_events: list[dict] = []
    for s in wc_sports:
        events = fetch_odds(api_key, s["key"], markets="outrights")
        (RAW_DUMP_DIR / f"{s['key']}_outrights_{ts}.json").write_text(
            json.dumps(events, indent=2)
        )
        print(f"  {len(events)} events")
        all_outright_events.extend(events)

    df_o = parse_outright_events(all_outright_events)

    # --- name diagnostic: compare API team names to our fixture list ---
    # Run BEFORE filtering so we still surface real name mismatches.
    # (qualified already loaded at the top of main via fixture_lookup build)
    if qualified:
        seen: set[str] = set()
        for r in all_match_rows:
            seen.update([r["home_team"], r["away_team"]])
        if not df_o.empty:
            seen.update(df_o["team"].tolist())
        unmatched = sorted(seen - qualified)
        if unmatched:
            print(f"\n!! {len(unmatched)} Odds-API team name(s) not in fixtures_2026.csv:")
            for t in unmatched:
                print(f"   {t!r}  -> non-qualified team, or add to SPORTSBOOK_TEAM_MAP")
        else:
            print("\nAll Odds-API team names map to internal names. ✓")

    # Filter outrights to teams actually in the WC. Books quote futures on
    # non-qualified teams (Italy, Denmark, ...) too; their mass slightly
    # deflates qualified teams' probabilities, so drop and renormalise.
    if not df_o.empty and qualified:
        n_before = len(df_o)
        df_o = df_o[df_o["team"].isin(qualified)].copy()
        df_o["p_winner"] = df_o["p_winner"] / df_o["p_winner"].sum()
        df_o["p_winner"] = df_o["p_winner"].round(4)
        df_o = df_o.sort_values("p_winner", ascending=False).reset_index(drop=True)
        dropped = n_before - len(df_o)
        if dropped:
            print(f"\nDropped {dropped} non-qualified team(s) from outrights; "
                  f"renormalised over the {len(df_o)} qualified teams.")

    if not df_o.empty:
        df_o.to_csv(OUTRIGHTS_PATH, index=False)
        print(f"\nSaved {len(df_o)} teams -> {OUTRIGHTS_PATH}")
        print(df_o.head(15).to_string(index=False))
    else:
        print("\nNo outright odds returned.")


if __name__ == "__main__":
    main()