"""
src/fetch_polymarket.py

Session 19 (rewritten Session 35a): pull WC 2026 probabilities from the
Polymarket Gamma API.

Three outputs, mirroring Session 18's sportsbook fetcher:
  data/processed/polymarket_outrights.csv  — title winner, one row per team
  data/processed/polymarket_groups.csv     — group winner, one row per team
  data/processed/polymarket_odds.csv       — per-match (group-stage h2h)

Spread (over-round) stripping: each multi-outcome event is N binary Yes/No
child markets. Sum of Yes prices across the event is slightly > 1 because of
bid/ask spread. We sum, divide by the sum to get fair probabilities — same
idea as the proportional vig-strip in fetch_odds.py.

Session 35a: Polymarket restructured its WC markets since Session 19 — the
old slugs (2026-fifa-world-cup-winner-595, fifa-world-cup-group-{letter}-winner)
now 404. Current slugs are world-cup-winner / world-cup-group-{letter}-winner.
Per-match h2h markets now exist too, under the soccer-fifwc series (id 11433,
slug pattern fifwc-{home}-{away}-{YYYY-MM-DD}). Don't confuse that with the
fif-{...} qualifier/friendly series (id 10238) — same slug-prefix shape, wrong
event type; a slug guess like fif-mex-rsa-... 200s but parses to nothing.
Each h2h event is THREE independent binary Yes/No markets ("Will {home}
win...?", "...end in a draw?", "Will {away} win...?"), not one 3-way market —
so the existing binary-market reader (yes_price_and_volume) works unchanged;
we bucket the three questions by side and renormalise like groups/title.

Slugs drift. fetch_event_resilient() falls back to /public-search when a seed
slug 404s/empties and prints a loud "update this slug" warning instead of
quietly returning nothing — the failure mode that caused this rewrite.

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

# Slugs re-derived from the live Gamma API in the Session 35a recon (the old
# ones from Session 19 now 404). If these drift again, fetch_event_resilient
# falls back to search and tells you the new slug to paste in here.
TITLE_EVENT_SLUG = "world-cup-winner"
GROUP_SLUG_TEMPLATE = "world-cup-group-{letter}-winner"
GROUP_LETTERS = list("abcdefghijkl")

# Per-match h2h events live under the soccer-fifwc series — NOT the
# similarly-prefixed fif-{...} qualifier/friendly series (id 10238, a decoy
# for slug-guessing: same fif- shape, different competition entirely).
MATCH_SERIES_ID = 11433
MATCH_SERIES_SLUG = "soccer-fifwc"
MATCH_SLUG_PREFIX = "fifwc-"
# Each fixture also spawns sibling derivative-market events; keep only the
# base h2h event (home win / draw / away win).
MATCH_SLUG_EXCLUDE_SUFFIXES = (
    "-halftime-result",
    "-first-half-result",
    "-second-half-result",
    "-first-to-score",
    "-exact-score",
    "-more-markets",
)
MATCH_SLUG_BASE_PARTS = 6

# Below this combined (home+draw+away) USD volume, flag a match as thin.
# The tournament opener (Mexico vs South Africa) totalled ~$396k combined;
# smaller-team group matches will likely run far lower. $10k is a starting
# floor chosen to catch genuinely thin markets without flagging everything —
# revisit once the full-tournament volume distribution is visible.
LOW_LIQUIDITY_VOLUME_FLOOR = 10_000

PROCESSED_DIR = Path("data/processed")
RAW_DUMP_DIR = Path("data/raw/polymarket")
OUTRIGHTS_PATH = PROCESSED_DIR / "polymarket_outrights.csv"
GROUPS_PATH = PROCESSED_DIR / "polymarket_groups.csv"
MATCHES_PATH = PROCESSED_DIR / "polymarket_odds.csv"
FIXTURES_PATH = PROCESSED_DIR / "fixtures_2026.csv"

# SOFT floor for the outright file — see fetch_odds.py for the full rationale.
# < 2 means degenerate; the floor stays tiny so a legitimately shrinking field
# (the book drops eliminated teams) is never blocked. The wc_teams filter, not
# this guard, is the real fix for the fixtures-shrink cliff.
MIN_OUTRIGHT_ROWS = 2

# Polymarket-name → our internal name. Mirrors fetch_odds.py's map.
# Extend based on the diagnostic at the end if needed. Don't add names
# pre-emptively — Polymarket's "name" field matches our internal convention
# for most teams verbatim (e.g. "Curaçao" needs no entry); let the diagnostic
# tell us what's actually missing.
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
    "IR Iran": "Iran",
}

# Pull the team name out of the market question text (title/group events only —
# per-match events carry structured team data in event["teams"], no regex needed).
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


def search_events(query: str) -> list[dict]:
    """Fallback discovery via /public-search when a seed slug 404s or empties."""
    r = requests.get(f"{GAMMA}/public-search", params={"q": query}, timeout=30)
    print(f"  GET {r.url}  HTTP {r.status_code}")
    if not r.ok:
        return []
    return r.json().get("events") or []


def fetch_event_resilient(seed_slug: str, search_query: str, slug_hint: str) -> dict | None:
    """
    Try the seed slug first (cheap, exact — works as long as Polymarket hasn't
    reshuffled). If it 404s/empties, search by query, prefer a result whose
    slug contains slug_hint, and print a loud "update this slug" warning so
    the next drift surfaces immediately instead of silently degrading to the
    all-empty failure mode that triggered this rewrite.
    """
    event = fetch_event(seed_slug)
    if event:
        return event
    print(f"  !! seed slug {seed_slug!r} returned nothing — searching {search_query!r}")
    results = search_events(search_query)
    hinted = [e for e in results if slug_hint in (e.get("slug") or "")]
    candidates = hinted or results
    if not candidates:
        print(f"  !! search for {search_query!r} returned nothing either — giving up on this event")
        return None
    found = candidates[0]
    found_slug = found.get("slug")
    print(f"  !! found via search instead: slug={found_slug!r} title={found.get('title')!r}")
    print(f"  !! UPDATE THE SEED SLUG IN THIS FILE: {seed_slug!r} -> {found_slug!r}")
    return fetch_event(found_slug)


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


# --- per-match parsing (Session 35a) --------------------------------------

def _name_tokens(name: str) -> list[str]:
    """
    Split a team name into matchable word tokens (alnum runs, len > 2).

    Needed because event["teams"][].name and the market question text don't
    always agree on separators/conjunctions for multi-word names — e.g. the
    structured name is "Bosnia-Herzegovina" but the question reads "Will
    Bosnia and Herzegovina win...?". Token-set matching handles that; a
    plain substring check on the full lowercased name doesn't.
    """
    return [w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) > 2]


def classify_match_side(question: str, home_name: str, away_name: str) -> str | None:
    """
    Bucket a per-match child market by its question text. Markets aren't
    guaranteed to come back in a fixed order, so match on content rather than
    position: the draw market mentions "draw"; the two win markets each name
    one side's team (matched via _name_tokens — see its docstring).
    """
    q = (question or "").lower()
    if "draw" in q:
        return "draw"
    home_tokens = _name_tokens(home_name)
    if home_tokens and all(t in q for t in home_tokens):
        return "home"
    away_tokens = _name_tokens(away_name)
    if away_tokens and all(t in q for t in away_tokens):
        return "away"
    return None


def parse_match_event(event: dict) -> dict | None:
    """
    One per-match h2h event -> one row of fair W/D/L probabilities, or None
    if the shape isn't the expected three-binary-markets layout.

    Team names come from the structured event["teams"] (name + ordering),
    not from regex-on-question — cleaner, and the alternative (abbreviation)
    is unreliable: Curaçao shows abbreviation "kor" in at least one event.
    """
    teams = event.get("teams") or []
    home = next((t for t in teams if t.get("ordering") == "home"), None)
    away = next((t for t in teams if t.get("ordering") == "away"), None)
    if not home or not away:
        return None
    home_name = home.get("name") or ""
    away_name = away.get("name") or ""

    sides: dict[str, tuple[float, float]] = {}
    for market in event.get("markets") or []:
        side = classify_match_side(market.get("question") or "", home_name, away_name)
        if side is None or side in sides:
            continue
        yes, volume = yes_price_and_volume(market)
        if yes is not None:
            sides[side] = (yes, volume)

    if set(sides) != {"home", "draw", "away"}:
        return None

    total = sum(p for p, _ in sides.values())
    if total <= 0:
        return None

    return {
        "slug": event.get("slug"),
        "commence_time": event.get("eventDate"),
        "home_team": normalise_team(home_name),
        "away_team": normalise_team(away_name),
        "p_home": round(sides["home"][0] / total, 4),
        "p_draw": round(sides["draw"][0] / total, 4),
        "p_away": round(sides["away"][0] / total, 4),
        "volume": round(sum(v for _, v in sides.values())),
    }


def _match_pair_key(row: dict) -> tuple[str, str]:
    return tuple(sorted((row["home_team"], row["away_team"])))


def _match_row_priority(row: dict) -> tuple[tuple[str, str], int, int, str]:
    slug = row.get("slug") or ""
    is_base = len(slug.split("-")) == MATCH_SLUG_BASE_PARTS
    pair = _match_pair_key(row)
    return (pair, -(row.get("volume") or 0), 0 if is_base else 1, slug)


def _dedupe_match_rows(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    keep: list[dict] = []
    dropped: list[dict] = []
    for row in sorted(rows, key=_match_row_priority):
        pair = _match_pair_key(row)
        if pair in seen:
            dropped.append(row)
            continue
        seen.add(pair)
        keep.append(row)

    for row in dropped:
        print("  DROPPED duplicate Polymarket per-match row:"
              f" {row['home_team']} vs {row['away_team']}"
              f" slug={row.get('slug')!r} volume={row.get('volume')}"
              f" p_home={row.get('p_home')} p_draw={row.get('p_draw')}"
              f" p_away={row.get('p_away')}")

    for row in keep:
        if (row.get("volume") or 0) == 0:
            print("  LOW-LIQUIDITY WARNING: selected 0-volume Polymarket row for"
                  f" {row['home_team']} vs {row['away_team']}"
                  f" slug={row.get('slug')!r} p_home={row.get('p_home')}"
                  f" p_draw={row.get('p_draw')} p_away={row.get('p_away')}")
    return keep


def fetch_match_events() -> list[dict]:
    """
    Page through the soccer-fifwc series and keep only base h2h events:
    slug fifwc-{home}-{away}-{date} with exactly 3 child markets. Excludes
    derivative sibling events (halftime/exact-score/more-markets) and
    anything outside this series — in particular the decoy fif-{...}
    qualifier/friendly series (id 10238), which uses the same slug-prefix
    shape but is a different competition.
    """
    events: list[dict] = []
    offset = 0
    while True:
        r = requests.get(
            f"{GAMMA}/events",
            params={"series_id": MATCH_SERIES_ID, "limit": 100, "offset": offset},
            timeout=30,
        )
        print(f"  GET {r.url}  HTTP {r.status_code}")
        if not r.ok:
            break
        batch = r.json()
        if not batch:
            break
        events.extend(batch)
        offset += len(batch)
        if len(batch) < 100:
            break

    base = []
    for ev in events:
        slug = ev.get("slug") or ""
        if not slug.startswith(MATCH_SLUG_PREFIX):
            continue
        if len(slug.split("-")) != MATCH_SLUG_BASE_PARTS:
            continue
        if len(ev.get("markets") or []) != 3:
            continue
        base.append(ev)
    return base


# --- main ----------------------------------------------------------------

def save_outright_guarded(df_new, path: Path, label: str) -> bool:
    """Write df_new to path unless it's degenerately short (< MIN_OUTRIGHT_ROWS)
    AND a fuller file already exists on disk (then keep the existing one). SOFT —
    never raises or aborts. Returns True if it wrote, False if it kept the stale
    file. Mirrors fetch_odds.py's helper (§6 per-fetcher defensive)."""
    n_new = len(df_new)
    if n_new < MIN_OUTRIGHT_ROWS and path.exists():
        try:
            n_old = len(pd.read_csv(path))
        except Exception:
            n_old = 0
        if n_old > n_new:
            print(f"\n!! {label}: new outright file has only {n_new} row(s) "
                  f"(< {MIN_OUTRIGHT_ROWS}); keeping the existing {n_old}-row "
                  f"file, NOT overwriting.")
            return False
    df_new.to_csv(path, index=False)
    if n_new < MIN_OUTRIGHT_ROWS:
        print(f"\n!! {label}: outright file is short ({n_new} row(s) "
              f"< {MIN_OUTRIGHT_ROWS}) with no fuller existing file — wrote anyway.")
    print(f"\nSaved {n_new} teams -> {path}")
    return True


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    qualified: set[str] = set()
    fixture_pairs: set[tuple[str, str]] = set()
    if FIXTURES_PATH.exists():
        fx = pd.read_csv(FIXTURES_PATH)
        qualified = set(fx["home_team"]) | set(fx["away_team"])
        fixture_pairs = set(zip(fx["home_team"], fx["away_team"]))

    # The OUTRIGHT (title-winner) filter keys off the FIXED 48-team WC set, NOT
    # fixtures_2026.csv, which shrinks as clean_data moves played fixtures out
    # (empty once the group stage ends) — keying off it wrongly drops every
    # already-advanced team from the title market. Mirrors the fetch_odds.py
    # fix (Session OUTRIGHT-FIX) and save_wc_ratings.py. Per-match h2h and the
    # group-winner markets keep keying off `qualified`/`fixture_pairs` — those
    # are tied to the group stage and are meant to wind down (PROJECT.md §6).
    from bracket import GROUPS
    wc_teams = {t for teams in GROUPS.values() for t in teams}

    # --- title winner ---
    print("\n=== Title winner ===")
    title_event = fetch_event_resilient(TITLE_EVENT_SLUG, "World Cup Winner", "winner")
    if title_event:
        (RAW_DUMP_DIR / f"title_{ts}.json").write_text(json.dumps(title_event, indent=2))
    df_title_raw = parse_title_event(title_event)
    print(f"  parsed {len(df_title_raw)} title markets")

    # --- group winners ---
    print("\n=== Group winners ===")
    group_frames = []
    for letter in GROUP_LETTERS:
        slug = GROUP_SLUG_TEMPLATE.format(letter=letter)
        ev = fetch_event_resilient(
            slug, f"World Cup Group {letter.upper()} Winner", f"group-{letter}"
        )
        if ev:
            (RAW_DUMP_DIR / f"group_{letter}_{ts}.json").write_text(json.dumps(ev, indent=2))
        group_frames.append(parse_group_event(letter, ev))
    df_groups_raw = (pd.concat(group_frames, ignore_index=True)
                     if group_frames else pd.DataFrame())

    # --- per-match h2h ---
    print("\n=== Per-match (h2h) ===")
    match_events = fetch_match_events()
    print(f"  {len(match_events)} base h2h events under series "
          f"{MATCH_SERIES_SLUG!r} (id {MATCH_SERIES_ID})")
    if match_events:
        (RAW_DUMP_DIR / f"matches_{ts}.json").write_text(json.dumps(match_events, indent=2))
    match_rows, unparsed = [], []
    for ev in match_events:
        row = parse_match_event(ev)
        if row:
            match_rows.append(row)
        else:
            unparsed.append(ev.get("slug"))
    if unparsed:
        print(f"  !! {len(unparsed)} event(s) had an unexpected market shape, skipped:")
        for s in unparsed[:5]:
            print(f"     {s!r}")
    df_matches_raw = pd.DataFrame(match_rows)
    print(f"  parsed {len(df_matches_raw)} per-match events")

    # --- name diagnostic (run BEFORE filtering, so mismatches surface) ---
    if qualified:
        seen: set[str] = set()
        if not df_title_raw.empty:
            seen.update(df_title_raw["team"].tolist())
        if not df_groups_raw.empty:
            seen.update(df_groups_raw["team"].tolist())
        if not df_matches_raw.empty:
            seen.update(df_matches_raw["home_team"].tolist())
            seen.update(df_matches_raw["away_team"].tolist())
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

    # --- filter to the fixed 48-team WC set, renormalise, save title ---
    if not df_title_raw.empty:
        df_title = df_title_raw[df_title_raw["team"].isin(wc_teams)].copy()
        n_dropped = len(df_title_raw) - len(df_title)
        if n_dropped:
            print(f"\nTitle: dropped {n_dropped} non-WC team(s) before renormalising.")
        if not df_title.empty:
            df_title = renormalise(df_title)
            df_title["volume"] = df_title["volume"].round(0).astype(int)
            df_title = (df_title[["team", "p_winner", "volume"]]
                          .sort_values("p_winner", ascending=False)
                          .reset_index(drop=True))
            if save_outright_guarded(df_title, OUTRIGHTS_PATH, "Polymarket"):
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

    # --- per-match: orient to fixture, flag liquidity, save ---
    # Match on the UNORDERED team pair, not Polymarket's home/away listing —
    # for the 9 host group-stage matches (USA/Mexico/Canada, neutral=False,
    # 60-Elo home bump applied — see PROJECT.md §6) Polymarket lists the
    # visiting team first while our fixture correctly lists the host as home.
    # The probability follows the TEAM, not Polymarket's slot: when we swap
    # home_team/away_team to match the fixture's orientation, p_home/p_away
    # swap with them (p_draw is unaffected either way).
    print("\n=== Per-match: orienting to fixtures & saving ===")
    if not df_matches_raw.empty:
        fixture_lookup = {frozenset((h, a)): (h, a) for h, a in fixture_pairs}
        oriented, flipped, unmatched = [], [], []
        for row in df_matches_raw.to_dict("records"):
            fx_orientation = fixture_lookup.get(frozenset((row["home_team"], row["away_team"])))
            if fx_orientation is None:
                unmatched.append(row)
                oriented.append(row)
                continue
            fx_home, fx_away = fx_orientation
            if (row["home_team"], row["away_team"]) == (fx_home, fx_away):
                oriented.append(row)
            else:
                flipped_row = dict(row)
                flipped_row["home_team"], flipped_row["away_team"] = fx_home, fx_away
                flipped_row["p_home"], flipped_row["p_away"] = row["p_away"], row["p_home"]
                oriented.append(flipped_row)
                flipped.append((row["home_team"], row["away_team"], fx_home, fx_away))

        oriented = _dedupe_match_rows(oriented)
        df_matches = pd.DataFrame(oriented)
        if "slug" in df_matches.columns:
            df_matches = df_matches.drop(columns=["slug"])
        n_matched = len(df_matches) - len(unmatched)
        print(f"  {n_matched}/{len(df_matches)} events matched a fixture in fixtures_2026.csv")
        if flipped:
            print(f"  Oriented {len(flipped)} event(s) to the fixture's home/away convention "
                  f"(host-match team-order disagreement; p_home/p_away swapped to follow "
                  f"the team, p_draw unchanged):")
            for pm_h, pm_a, fx_h, fx_a in flipped:
                print(f"     Polymarket: {pm_h} vs {pm_a}  ->  fixture: {fx_h} vs {fx_a}")
        if unmatched:
            print(f"  !! {len(unmatched)} event(s) did NOT match any fixture "
                  f"(name-mapping issue, non-group-stage market, or already-played fixture):")
            for r in unmatched:
                print(f"     {r['home_team']} vs {r['away_team']}  ({r['commence_time']})")

        df_matches["low_liquidity"] = df_matches["volume"] < LOW_LIQUIDITY_VOLUME_FLOOR
        df_matches["volume"] = df_matches["volume"].astype(int)
        df_matches = (df_matches[["commence_time", "home_team", "away_team",
                                  "p_home", "p_draw", "p_away", "volume", "low_liquidity"]]
                        .sort_values("commence_time")
                        .reset_index(drop=True))
        df_matches.to_csv(MATCHES_PATH, index=False)
        n_thin = int(df_matches["low_liquidity"].sum())
        print(f"\nSaved {len(df_matches)} matches -> {MATCHES_PATH} "
              f"({n_thin} flagged low-liquidity, combined volume < ${LOW_LIQUIDITY_VOLUME_FLOOR:,})")
        print(df_matches.head(10).to_string(index=False))
    else:
        print("  No per-match events parsed — writing header-only file so "
              "downstream consumers keep a stable join schema.")
        empty = pd.DataFrame(columns=[
            "commence_time", "home_team", "away_team",
            "p_home", "p_draw", "p_away", "volume", "low_liquidity",
        ])
        empty.to_csv(MATCHES_PATH, index=False)
        print(f"  saved -> {MATCHES_PATH}")


if __name__ == "__main__":
    main()
