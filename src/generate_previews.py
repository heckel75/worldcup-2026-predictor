"""
src/generate_previews.py

Session 21: generate 3-paragraph Claude previews for every WC 2026 fixture.

Pipeline:
  1. Load data/processed/triple_compare.csv (model + sportsbook + Polymarket).
  2. For each match, build a structured numerical input block.
  3. Hash the rounded inputs (1pp resolution) plus prompt version + model.
  4. If data/processed/previews/<key>.json already exists with the same hash,
     skip -- the inputs haven't materially moved. Otherwise call Claude and
     write a fresh JSON.
  5. Print a summary: cached / generated / failed.

Caching philosophy: model probabilities are deterministic given fixed Elo +
rho; only sportsbook lines and (eventually) Polymarket prices drift day to
day. Rounding to 1pp keeps trivial book ticks from triggering regeneration.
Prompt-version and model-name in the hash mean any prompt or model change
invalidates the whole cache automatically.

Run from project root:
    python src/generate_previews.py            # process all fixtures
    python src/generate_previews.py --limit 3  # smoke test on first 3
    python src/generate_previews.py --force    # ignore cache, regenerate all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from anthropic import Anthropic, APIError
from dotenv import load_dotenv

# --- configuration ----------------------------------------------------------

TRIPLE_PATH   = Path("data/processed/triple_compare.csv")
CACHE_DIR     = Path("data/processed/previews")

# Haiku 4.5 is fast and cheap (~$0.15 for all 104 matches). If quality
# disappoints, flip to "claude-sonnet-4-6" -- the hash includes model name
# so the swap automatically invalidates the cache.
MODEL         = "claude-haiku-4-5-20251001"

# Bump when the system or user prompt changes in any way that should
# invalidate cached previews.
PROMPT_VERSION = 3

# Generous enough for ~180 words; Claude almost always finishes well below.
MAX_TOKENS = 400


# --- prompt -----------------------------------------------------------------

SYSTEM_PROMPT = """You write concise, analytical match previews for a World Cup 2026 forecast dashboard.

OUTPUT: exactly three short paragraphs of plain text, separated by single blank lines. No headers, no markdown, no preamble, no sign-off. Aim for 120-180 words total.

CONTENT RULES (strict):
- Use ONLY the numerical inputs supplied. Do not invent or reference players, coaches, injuries, suspensions, recent form, head-to-head history, tactics, formations, or any squad/news context. You have none of that information.
- NO EXTERNAL KNOWLEDGE about the teams or tournament. This includes which countries are hosts, defending champions, recent tournament results, qualifying campaigns, regional rivalries, FIFA rankings, historical pedigree, or geographic context. The prompt tells you the venue is neutral -- do not contradict or qualify that. If a fact is not in the input block below, you do not know it and must not reference it, even implicitly.
- "The model" refers to the statistical forecast (Dixon-Coles on Elo). "The market" refers to sportsbook consensus. Use those terms when distinguishing sources.
- Round percentages to whole numbers in prose ("28%" not "27.6%").
- No cliches ("clash of titans", "must-win", "on paper", "all eyes on", "battle"). No predictions phrased as certainties -- probabilities are probabilities.
- Present tense, third person, measured analytical register.
- ARITHMETIC: any numerical gap or comparison you state in prose must match the inputs. If you write "X percentage points apart" or "nearly even" or "closely matched", the actual difference must support that claim. Verify before writing. When in doubt, describe the spread qualitatively without naming a specific pp figure.

STRUCTURE:
- Paragraph 1: Name the model's favorite and its headline win probability. Contrast briefly with the underdog's chance.
- Paragraph 2: Describe the expected shape from the W/D/L spread -- is the gap wide or narrow, is a draw plausible, how competitive is the match likely to be?
- Paragraph 3: Compare model and market. If broadly aligned (gaps under ~5pp), say so plainly. If they diverge meaningfully, name the gap and note that the dashboard does not claim to know which source is right.

If sportsbook data is absent, paragraph 3 should note that no market consensus is available yet rather than fabricating one."""


def build_user_prompt(row: pd.Series) -> str:
    """Render the numerical inputs as a clean key:value block."""
    lines = [
        f"Match: {row['home_team']} vs {row['away_team']}",
        f"Date: {pd.to_datetime(row['date']).date().isoformat()}",
        "Venue: neutral",
        "",
        "Model probabilities (bias-corrected):",
        f"  {row['home_team']} win: {row['p_home_model_corr']*100:.1f}%",
        f"  Draw: {row['p_draw_model_corr']*100:.1f}%",
        f"  {row['away_team']} win: {row['p_away_model_corr']*100:.1f}%",
        "",
    ]
    if pd.notna(row["p_home_book"]):
        n_books = int(row["n_books"])
        lines.extend([
            f"Sportsbook consensus ({n_books} books, vig stripped):",
            f"  {row['home_team']} win: {row['p_home_book']*100:.1f}%",
            f"  Draw: {row['p_draw_book']*100:.1f}%",
            f"  {row['away_team']} win: {row['p_away_book']*100:.1f}%",
            "",
        ])
    else:
        lines.append("Sportsbook: no consensus market available yet.")
        lines.append("")

    if pd.notna(row.get("p_home_poly")):
        lines.extend([
            "Polymarket prices:",
            f"  {row['home_team']} win: {row['p_home_poly']*100:.1f}%",
            f"  Draw: {row['p_draw_poly']*100:.1f}%",
            f"  {row['away_team']} win: {row['p_away_poly']*100:.1f}%",
        ])
    else:
        lines.append("Polymarket: per-match market not yet posted.")

    return "\n".join(lines)


# --- caching ----------------------------------------------------------------

def slugify(s: str) -> str:
    """Filesystem-safe team slug. 'Bosnia and Herzegovina' -> 'bosnia_and_herzegovina'."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def match_key(row: pd.Series) -> str:
    """Stable per-match filename stem. Date prefix orders files chronologically."""
    date = pd.to_datetime(row["date"]).date().isoformat()
    return f"{date}_{slugify(row['home_team'])}_vs_{slugify(row['away_team'])}"


def make_input_hash(row: pd.Series) -> str:
    """
    Hash the rounded inputs that materially affect the preview. Excludes
    n_books deliberately -- book count drift shouldn't trigger regeneration.
    """
    def r(x): return round(float(x), 2) if pd.notna(x) else None

    payload = {
        "home":  row["home_team"],
        "away":  row["away_team"],
        "model_probs": [r(row["p_home_model_corr"]),
                        r(row["p_draw_model_corr"]),
                        r(row["p_away_model_corr"])],
        "book_probs": (
            [r(row["p_home_book"]), r(row["p_draw_book"]), r(row["p_away_book"])]
            if pd.notna(row["p_home_book"]) else None
        ),
        "poly_probs": (
            [r(row["p_home_poly"]), r(row["p_draw_poly"]), r(row["p_away_poly"])]
            if pd.notna(row.get("p_home_poly")) else None
        ),
        "model_name":     MODEL,
        "prompt_version": PROMPT_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()


def load_cached(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# --- API call ---------------------------------------------------------------

def generate_preview(client: Anthropic, user_prompt: str) -> str:
    """Single Claude call. Raises on API failure; caller handles."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # Concatenate text blocks (usually a single one for this prompt shape).
    parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
    return "\n".join(parts).strip()


# --- main pipeline ----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N fixtures (smoke test).")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate every preview, ignoring cache.")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not found in environment.")
        print("Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = Anthropic()
    df = pd.read_csv(TRIPLE_PATH)
    if args.limit:
        df = df.head(args.limit)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model:          {MODEL}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Cache dir:      {CACHE_DIR}")
    print(f"Fixtures:       {len(df)}\n")

    n_cached = n_generated = n_failed = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    for _, row in df.iterrows():
        key = match_key(row)
        path = CACHE_DIR / f"{key}.json"
        current_hash = make_input_hash(row)
        cached = load_cached(path)

        if (not args.force
                and cached is not None
                and cached.get("input_hash") == current_hash):
            n_cached += 1
            print(f"  [cache] {key}")
            continue

        user_prompt = build_user_prompt(row)
        try:
            text = generate_preview(client, user_prompt)
        except APIError as e:
            n_failed += 1
            failures.append((key, str(e)))
            print(f"  [FAIL ] {key}  -- {e}")
            continue
        except Exception as e:  # network, timeout, anything else
            n_failed += 1
            failures.append((key, repr(e)))
            print(f"  [FAIL ] {key}  -- {e!r}")
            continue

        record = {
            "match_key":      key,
            "home_team":      row["home_team"],
            "away_team":      row["away_team"],
            "date":           pd.to_datetime(row["date"]).date().isoformat(),
            "input_hash":     current_hash,
            "generated_at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model":          MODEL,
            "prompt_version": PROMPT_VERSION,
            "preview_text":   text,
        }
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        n_generated += 1
        print(f"  [new  ] {key}")

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s.  cached={n_cached}  generated={n_generated}  failed={n_failed}")
    if failures:
        print("\nFailures:")
        for k, msg in failures:
            print(f"  {k}: {msg}")


if __name__ == "__main__":
    main()