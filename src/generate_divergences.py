"""
src/generate_divergences.py

Session 22: layered divergence commentary on top of preview pipeline.

For each match in triple_compare.csv we produce one of three outputs:

  1. "commentary" - a one-paragraph Claude analysis of the model-vs-book
     gap, branched by divergence_type. Only for rows with
     flag_divergent=True (Session 20: max single-outcome gap >= 15pp,
     excluding host-country matches).

  2. "note" - a fixed one-line acknowledgement for rows where the
     "host advantage not modeled in v1" note was attached in
     triple_compare.py. No LLM call.

  3. "skip" - no output. Match agrees within 15pp, no special note,
     or no sportsbook data.

Outputs land in data/processed/divergences/<match_key>.json with a
"kind" discriminator field so the dashboard renders them appropriately.

Caching for commentaries mirrors generate_previews.py: hash the rounded
model+book probs plus divergence_type, prompt version, and model name.
Notes are deterministic and skipped if already on disk with the same
note_reason.

Run from project root:
    python src/generate_divergences.py            # process all
    python src/generate_divergences.py --limit 10 # smoke test
    python src/generate_divergences.py --force    # ignore cache
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
from typing import Literal

import pandas as pd
from anthropic import Anthropic, APIError
from dotenv import load_dotenv

# --- configuration ----------------------------------------------------------

TRIPLE_PATH = Path("data/processed/triple_compare.csv")
CACHE_DIR   = Path("data/processed/divergences")

MODEL = "claude-haiku-4-5-20251001"

# Bump when SYSTEM_PROMPT or FRAMING_NOTES change in any way that should
# invalidate cached commentaries. The hash includes this, so any bump
# invalidates the whole cache automatically.
PROMPT_VERSION = 2

MAX_TOKENS = 250  # one paragraph, ~60-100 words; safety margin

Bucket = Literal["commentary", "note", "skip"]


# --- prompt -----------------------------------------------------------------

SYSTEM_PROMPT = """You write short, analytical divergence commentary for a World Cup 2026 forecast dashboard.

OUTPUT: exactly one paragraph of plain text, 60-100 words. No headers, no markdown, no preamble, no sign-off.

CONTENT RULES (strict):
- Use ONLY the numerical inputs supplied. Do not invent or reference players, coaches, injuries, suspensions, recent form, head-to-head history, tactics, formations, strategy, recent results, or qualifying campaigns. You have none of that information.
- NO EXTERNAL KNOWLEDGE about the teams or tournament. This includes hosting status, defending champions, regional rivalries, FIFA rankings, geographic context, or historical pedigree. The venue is neutral -- do not contradict that.
- "The model" refers to a statistical forecast built on Elo ratings and a Dixon-Coles goal model. It has NO access to injuries, current form, lineup changes, or news since its last update.
- "The market" refers to sportsbook consensus, which prices in real-money flows and may reflect information the model cannot see.

PERCENTAGES:
- ALWAYS render percentages as whole integers. Write "28%" not "27.6%" and not "27.5%". Decimals in percentages are forbidden. Round half-up when needed: 56.6% becomes 57%, 39.5% becomes 40%.

ARITHMETIC (strict):
- Verify EVERY numerical or comparative claim against the inputs before writing it. This includes qualitative comparisons like "nearly equal", "almost identical", "much higher", "twice as much", "in opposite directions" -- these are claims too and must be supported by the actual numbers.
- Do NOT combine gaps across outcomes into a single "total gap", "overall gap", or "opposite-direction gap" number. When you cite a percentage-point gap, it must be the gap between two specific probabilities of one specific outcome (e.g. the gap on the home win, or the gap on the draw).
- When in doubt, describe the spread qualitatively without naming a specific pp figure.

STRUCTURE (in this order, in a single paragraph):
1. Name the most striking gap concretely (which outcome, how many percentage points, which side is higher).
2. Briefly gesture at what kinds of information might explain the gap -- phrased as possibilities, not claims. The ONLY categories you may reference are: recent form, injuries, or lineup news. Do not mention tactics, formations, strategy, mentality, motivation, or anything else. Do NOT name specific players, coaches, or events.
3. Close with the honest framing: the dashboard surfaces the gap as a signal, not a verdict on which source is right.

PROHIBITED:
- Claiming either source is "right" or "wrong".
- Inventing specific causes (e.g., "Brazil's striker is injured").
- Mentioning tactics, formations, strategy, or game-plan.
- Combining gaps across outcomes into a single "total" or "overall" number.
- Decimal percentages.
- Sportswriting cliches ("the bookies sense an upset", "value pick", "the market is bullish on", "in form")."""


FRAMING_NOTES = {
    "model_under_concentrated": (
        "Divergence shape: model and market agree on the favorite, but the "
        "market is MORE confident than the model. Frame the commentary around "
        "what kind of information might be sharpening the market's conviction."
    ),
    "model_over_concentrated": (
        "Divergence shape: model and market agree on the favorite, but the "
        "market is LESS confident than the model -- i.e., the market hedges "
        "more. Frame the commentary around what kind of information might be "
        "tempering the market's confidence."
    ),
    "disagree_on_favorite": (
        "Divergence shape: model and market name DIFFERENT favorites. Report "
        "the two favorites as separate facts (e.g. \"the model favors X at A%; "
        "the market favors Y at B%\") and, if you cite a percentage-point gap, "
        "cite the gap on ONE specific outcome only. Do NOT combine into a "
        "\"total gap\" or \"opposite-direction gap\" -- there are two "
        "independent shifts on different outcomes here, not one number."
    ),
}


def build_user_prompt(row: pd.Series) -> str:
    """Render the data block plus the divergence_type framing note."""
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
        f"Sportsbook consensus ({int(row['n_books'])} books, vig stripped):",
        f"  {row['home_team']} win: {row['p_home_book']*100:.1f}%",
        f"  Draw: {row['p_draw_book']*100:.1f}%",
        f"  {row['away_team']} win: {row['p_away_book']*100:.1f}%",
        "",
    ]
    if pd.notna(row.get("p_home_poly")):
        lines.extend([
            "Polymarket prices:",
            f"  {row['home_team']} win: {row['p_home_poly']*100:.1f}%",
            f"  Draw: {row['p_draw_poly']*100:.1f}%",
            f"  {row['away_team']} win: {row['p_away_poly']*100:.1f}%",
            "",
        ])
    lines.append(FRAMING_NOTES[row["divergence_type"]])
    lines.append("")
    lines.append("Write the divergence commentary now.")
    return "\n".join(lines)


# --- bucket logic -----------------------------------------------------------

def decide_bucket(row: pd.Series) -> Bucket:
    """
    Categorise a fixture into commentary / note / skip.
    Mutually exclusive by construction: triple_compare.py sets
    flag_divergent=False whenever a host note is attached.
    """
    note = row.get("note")
    if pd.notna(note) and str(note).strip():
        return "note"
    if bool(row.get("flag_divergent", False)):
        return "commentary"
    return "skip"


# --- slug + key (must match generate_previews.py) --------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def match_key(row: pd.Series) -> str:
    date = pd.to_datetime(row["date"]).date().isoformat()
    return f"{date}_{slugify(row['home_team'])}_vs_{slugify(row['away_team'])}"


# --- input hash for commentaries -------------------------------------------

def make_input_hash(row: pd.Series) -> str:
    """
    Hash the rounded model + book probs, plus divergence_type, prompt
    version, and model name. divergence_type is in the hash because it
    branches the prompt -- a borderline match that flips category should
    regenerate even if the probs barely moved.
    """
    def r(x):
        return round(float(x), 2) if pd.notna(x) else None

    payload = {
        "home": row["home_team"],
        "away": row["away_team"],
        "model_probs": [r(row["p_home_model_corr"]),
                        r(row["p_draw_model_corr"]),
                        r(row["p_away_model_corr"])],
        "book_probs":  [r(row["p_home_book"]),
                        r(row["p_draw_book"]),
                        r(row["p_away_book"])],
        "divergence_type": row["divergence_type"],
        "model_name":      MODEL,
        "prompt_version":  PROMPT_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()


# --- record builders --------------------------------------------------------

def build_note_record(row: pd.Series) -> dict:
    """Fixed acknowledgement for host-excluded matches. No LLM."""
    return {
        "match_key":    match_key(row),
        "home_team":    row["home_team"],
        "away_team":    row["away_team"],
        "date":         pd.to_datetime(row["date"]).date().isoformat(),
        "kind":         "note",
        "note_reason":  str(row["note"]).strip(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_commentary_record(
    row: pd.Series, text: str, input_hash: str,
) -> dict:
    return {
        "match_key":       match_key(row),
        "home_team":       row["home_team"],
        "away_team":       row["away_team"],
        "date":            pd.to_datetime(row["date"]).date().isoformat(),
        "kind":            "commentary",
        "divergence_type": row["divergence_type"],
        "max_gap_pp":      round(float(row["div_model_book_max"]) * 100, 1),
        "input_hash":      input_hash,
        "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model":           MODEL,
        "prompt_version":  PROMPT_VERSION,
        "commentary_text": text,
    }


# --- cache helpers ----------------------------------------------------------

def load_cached(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# --- LLM call --------------------------------------------------------------

def generate_commentary(client: Anthropic, user_prompt: str) -> str:
    """Single Claude call. Raises on API failure; caller handles."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
    return "\n".join(parts).strip()


# --- main pipeline ----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N fixtures (smoke test).")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate every commentary, ignoring cache.")
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

    bucket_counts = {"commentary": 0, "note": 0, "skip": 0}
    n_cached = n_generated = n_failed = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    for _, row in df.iterrows():
        bucket = decide_bucket(row)
        bucket_counts[bucket] += 1

        if bucket == "skip":
            continue

        key = match_key(row)
        path = CACHE_DIR / f"{key}.json"

        if bucket == "note":
            existing = load_cached(path)
            current_reason = str(row["note"]).strip()
            if (not args.force
                    and existing is not None
                    and existing.get("kind") == "note"
                    and existing.get("note_reason") == current_reason):
                n_cached += 1
                print(f"  [cache] note         {key}")
                continue
            record = build_note_record(row)
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                            encoding="utf-8")
            n_generated += 1
            print(f"  [new  ] note         {key}")
            continue

        # commentary bucket
        current_hash = make_input_hash(row)
        cached = load_cached(path)
        if (not args.force
                and cached is not None
                and cached.get("kind") == "commentary"
                and cached.get("input_hash") == current_hash):
            n_cached += 1
            print(f"  [cache] commentary   {key}")
            continue

        user_prompt = build_user_prompt(row)
        try:
            text = generate_commentary(client, user_prompt)
        except APIError as e:
            n_failed += 1
            failures.append((key, str(e)))
            print(f"  [FAIL ] commentary   {key}  -- {e}")
            continue
        except Exception as e:
            n_failed += 1
            failures.append((key, repr(e)))
            print(f"  [FAIL ] commentary   {key}  -- {e!r}")
            continue

        record = build_commentary_record(row, text, current_hash)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        n_generated += 1
        print(f"  [new  ] commentary   {key}  "
              f"({row['divergence_type']}, "
              f"{row['div_model_book_max']*100:.1f}pp)")

    dt = time.time() - t0
    print(f"\nBucket counts: {bucket_counts}")
    print(f"Cache:         cached={n_cached}  generated={n_generated}  failed={n_failed}")
    print(f"Elapsed:       {dt:.1f}s")

    if failures:
        print("\nFailures:")
        for k, msg in failures:
            print(f"  {k}: {msg}")


if __name__ == "__main__":
    main()