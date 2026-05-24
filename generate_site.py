"""
Session 23–25: Static site generator.

Reads the processed forecast artifacts and renders Jinja2 templates into
docs/, the folder GitHub Pages serves. This script is a pure CONSUMER of
data: it never imports the model or recomputes anything. The model layer
produces dated CSV snapshots; this turns the latest one into HTML.

Presentation that belongs to the SITE (official display names, group
letters, heatmap tiers) lives here, not in the model — see PROJECT.md §4.

Pages produced:
    docs/index.html        tournament survival grid (Session 24)
    docs/matches/*.html     one per WC fixture (Session 25)

Run from project root:
    python generate_site.py
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import whats_changed

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path("templates")
STATIC_DIR = Path("static")
SNAPSHOTS_DIR = Path("data/processed/snapshots")
DIVERGENCE_SNAPS_DIR = Path("data/processed/divergence_snapshots")
TRIPLE_PATH = Path("data/processed/triple_compare.csv")
PREVIEWS_DIR = Path("data/processed/previews")
DIVERGENCES_DIR = Path("data/processed/divergences")
OUTPUT_DIR = Path("docs")
MATCHES_OUT = OUTPUT_DIR / "matches"


# ----------------------------------------------------------------------
# Presentation constants (site layer owns these; model uses its own names)
# ----------------------------------------------------------------------

# Internal dataset name -> official tournament display name. Only the teams
# whose official name differs from our dataset convention appear here;
# everything else passes through unchanged. The MODEL keeps internal names.
DISPLAY_NAMES: dict[str, str] = {
    "Turkey":         "Türkiye",
    "Czech Republic": "Czechia",
    "Ivory Coast":    "Côte d'Ivoire",
    "Cape Verde":     "Cabo Verde",
}

# Group letter per team. Mirrors bracket.GROUPS, duplicated here on purpose
# so the generator stays decoupled from the model layer (src/ isn't on the
# import path when this runs from the repo root). The draw is fixed, so this
# can't drift in practice.
_GROUPS: dict[str, list[str]] = {
    "A": ["Mexico",      "South Africa",           "South Korea",  "Czech Republic"],
    "B": ["Canada",      "Bosnia and Herzegovina", "Qatar",        "Switzerland"],
    "C": ["Brazil",      "Morocco",                "Haiti",        "Scotland"],
    "D": ["USA",         "Paraguay",               "Australia",    "Turkey"],
    "E": ["Germany",     "Curaçao",                "Ivory Coast",  "Ecuador"],
    "F": ["Netherlands", "Japan",                  "Sweden",       "Tunisia"],
    "G": ["Belgium",     "Egypt",                  "Iran",         "New Zealand"],
    "H": ["Spain",       "Cape Verde",             "Saudi Arabia", "Uruguay"],
    "I": ["France",      "Senegal",                "Iraq",         "Norway"],
    "J": ["Argentina",   "Algeria",                "Austria",      "Jordan"],
    "K": ["Portugal",    "DR Congo",               "Uzbekistan",   "Colombia"],
    "L": ["England",     "Croatia",                "Ghana",        "Panama"],
}
TEAM_GROUP: dict[str, str] = {t: g for g, ts in _GROUPS.items() for t in ts}

# Survival columns rendered with the green ramp, in tournament order:
# (snapshot column, short header label). p_champion is handled separately —
# it gets the gold "hero" ramp.
SURVIVAL_COLS: list[tuple[str, str]] = [
    ("p_advance", "R32"),
    ("p_r16",     "R16"),
    ("p_qf",      "QF"),
    ("p_sf",      "SF"),
    ("p_final",   "Final"),
]

# Heatmap tier boundaries. A probability is in tier i if it's below the i-th
# cutoff; past the last cutoff it's the top tier. These MUST match the bucket
# comments in style.css :root.
_SURV_CUTS  = (0.01, 0.10, 0.25, 0.50, 0.75)   # -> t-surv-0 .. t-surv-5
_CHAMP_CUTS = (0.01, 0.03, 0.08, 0.15, 0.25)   # -> t-champ-0 .. t-champ-5

# Human label per divergence_type, for the per-match callout (Session 25).
DIV_LABELS: dict[str, str] = {
    "model_under_concentrated": "Market more confident than the model",
    "model_over_concentrated":  "Market hedges more than the model",
    "disagree_on_favorite":     "Model and market disagree on the favourite",
}


def _tier(p: float, cuts: tuple[float, ...]) -> int:
    for i, hi in enumerate(cuts):
        if p < hi:
            return i
    return len(cuts)


def _fmt(p: float) -> str:
    """Probability -> percent string; em-dash for anything that rounds to 0."""
    pct = p * 100
    return "—" if pct < 0.05 else f"{pct:.1f}%"


def _pct0(p: float) -> str:
    """Probability -> whole-percent string (match bars use integers)."""
    return f"{round(p * 100)}%"


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

def disp(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def slugify(s: str) -> str:
    """Filesystem-safe team slug — MUST match generate_previews.slugify."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def match_key(date_iso: str, home: str, away: str) -> str:
    """Per-match stem — MUST match generate_previews.match_key."""
    return f"{date_iso}_{slugify(home)}_vs_{slugify(away)}"


def _date_human(date_iso: str) -> str:
    """'2026-06-13' -> '13 June 2026' (no %-d: that breaks on Windows)."""
    d = dt.date.fromisoformat(date_iso)
    return f"{d.day} {d.strftime('%B')} {d.year}"


# ----------------------------------------------------------------------
# Index page (Session 24 survival grid)
# ----------------------------------------------------------------------

def _latest_snapshot() -> Path:
    """Return the most recent snapshot CSV.

    Snapshot filenames are ISO dates (YYYY-MM-DD.csv), so a plain lexical
    sort is also chronological; the last element is the newest.
    """
    snaps = sorted(SNAPSHOTS_DIR.glob("*.csv"))
    if not snaps:
        raise FileNotFoundError(
            f"No snapshots in {SNAPSHOTS_DIR}/. Run `python src/monte_carlo.py` first."
        )
    return snaps[-1]


def _load_teams(snapshot: Path) -> list[dict]:
    """Load a snapshot into ranked, render-ready per-team dicts.

    Each dict carries the display name, group letter, the five survival
    cells (text + green-tier class), and the champion cell (text + gold-tier
    class). The template just iterates — no formatting or bucketing in Jinja.
    """
    df = pd.read_csv(snapshot)
    df = df.sort_values("p_champion", ascending=False, kind="mergesort")

    teams: list[dict] = []
    for rec in df.to_dict("records"):
        name = rec["team"]
        survival = [
            {"text": _fmt(rec[col]), "tier": f"t-surv-{_tier(rec[col], _SURV_CUTS)}"}
            for col, _label in SURVIVAL_COLS
        ]
        champ_p = rec["p_champion"]
        teams.append({
            "name":  disp(name),
            "group": TEAM_GROUP.get(name, "?"),
            "survival": survival,
            "champ": {
                "text": _fmt(champ_p),
                "tier": f"t-champ-{_tier(champ_p, _CHAMP_CUTS)}",
            },
        })
    return teams


# ----------------------------------------------------------------------
# Per-match pages (Session 25)
# ----------------------------------------------------------------------

def _segments(home_p, draw_p, away_p) -> list[dict]:
    """Three-way bar segments. flex-basis uses the exact float so widths sum
    to 100%; the printed label is a whole percent. Hide the inline label on
    segments too thin to fit it (heavy mismatches), so it never clips badly."""
    out = []
    for cls, p in (("home", home_p), ("draw", draw_p), ("away", away_p)):
        w = round(float(p) * 100, 2)
        out.append({"cls": cls, "w": w, "pct": _pct0(p), "show": w >= 8})
    return out


def _source(label, home_p, draw_p, away_p, home_name, away_name, sub=None) -> dict:
    aria = (f"{label}: {home_name} {_pct0(home_p)}, draw {_pct0(draw_p)}, "
            f"{away_name} {_pct0(away_p)}")
    return {"label": label, "available": True, "sub": sub, "aria": aria,
            "segments": _segments(home_p, draw_p, away_p)}


def _absent(label, text) -> dict:
    return {"label": label, "available": False, "absent_text": text}


def _headline_gap(row, home_name, away_name) -> str:
    """Largest single-outcome model-vs-book gap, stated deterministically as
    one fact (which outcome, which source is higher, the pp gap). Computed
    here — never taken from Claude — so it can't drift from the numbers."""
    outcomes = [
        (home_name, row["p_home_model_corr"], row["p_home_book"]),
        ("a draw",  row["p_draw_model_corr"], row["p_draw_book"]),
        (away_name, row["p_away_model_corr"], row["p_away_book"]),
    ]
    label, m, b = max(outcomes, key=lambda o: abs(o[1] - o[2]))
    gap = round(abs(m - b) * 100)
    higher = "market" if b > m else "model"
    hi, lo = (b, m) if b > m else (m, b)
    return (f"The {higher} puts {label} {gap} pp higher than the "
            f"{'model' if higher == 'market' else 'market'} — "
            f"{round(hi*100)}% vs {round(lo*100)}%.")


def _load_divergence(key: str, row, home_name, away_name) -> dict | None:
    """Read divergences/<key>.json if present. Commentary rows get the
    computed headline + Claude paragraph; host 'note' rows get a muted
    caveat; everything else has no file and returns None."""
    path = DIVERGENCES_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if rec.get("kind") == "note":
        return {"tone": "note", "label": "Caveat",
                "headline": None, "text": rec.get("note_reason", "")}

    has_book = pd.notna(row["p_home_book"])
    return {
        "tone": "commentary",
        "label": DIV_LABELS.get(rec.get("divergence_type"), "Divergence"),
        "headline": _headline_gap(row, home_name, away_name) if has_book else None,
        "text": rec.get("commentary_text", ""),
    }


def _load_preview(key: str) -> list[str]:
    """Read previews/<key>.json -> list of paragraphs (split on blank lines)."""
    path = PREVIEWS_DIR / f"{key}.json"
    if not path.exists():
        return []
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [p.strip() for p in rec.get("preview_text", "").split("\n\n") if p.strip()]


def _build_match(row) -> dict:
    """Turn one triple_compare row into a render-ready match context."""
    home, away = row["home_team"], row["away_team"]
    home_d, away_d = disp(home), disp(away)
    key = match_key(row["date"], home, away)

    sources = [_source("Model",
                       row["p_home_model_corr"], row["p_draw_model_corr"],
                       row["p_away_model_corr"], home_d, away_d,
                       sub="bias-corrected")]

    if pd.notna(row["p_home_book"]):
        sources.append(_source("Sportsbook",
                               row["p_home_book"], row["p_draw_book"],
                               row["p_away_book"], home_d, away_d,
                               sub=f"{int(row['n_books'])} books, vig stripped"))
    else:
        sources.append(_absent("Sportsbook", "No consensus market posted yet."))

    if pd.notna(row.get("p_home_poly")):
        sources.append(_source("Polymarket",
                               row["p_home_poly"], row["p_draw_poly"],
                               row["p_away_poly"], home_d, away_d))
    else:
        sources.append(_absent("Polymarket", "Per-match market not yet posted."))

    return {
        "key":          key,
        "home_display": home_d,
        "away_display": away_d,
        "group":        TEAM_GROUP.get(home, "?"),
        "date_human":   _date_human(row["date"]),
        "sources":      sources,
        "divergence":   _load_divergence(key, row, home_d, away_d),
        "preview_paras": _load_preview(key),
    }


def _load_matches() -> list[dict]:
    """All fixtures from triple_compare.csv, in fixture (date) order."""
    if not TRIPLE_PATH.exists():
        raise FileNotFoundError(
            f"{TRIPLE_PATH} not found. Run `python src/triple_compare.py` first."
        )
    df = pd.read_csv(TRIPLE_PATH)
    return [_build_match(row) for _, row in df.iterrows()]


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def _build_env() -> Environment:
    """Jinja2 environment rooted at templates/, with HTML autoescaping on."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


def _render_page(env: Environment, template_name: str, out_path: Path, **context) -> None:
    """Render one template to one output file."""
    html = env.get_template(template_name).render(**context)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def _copy_static() -> None:
    """Copy each file in static/ into docs/ (flat). No-op if static/ is absent."""
    if not STATIC_DIR.exists():
        return
    for item in STATIC_DIR.iterdir():
        if item.is_file():
            shutil.copy2(item, OUTPUT_DIR / item.name)


def build_site() -> None:
    snapshot = _latest_snapshot()
    snapshot_date = snapshot.stem  # filename is the date
    teams = _load_teams(snapshot)
    matches = _load_matches()

    # --- what-changed diff data ------------------------------------------
    prev_snap_path, _ = whats_changed._two_newest(SNAPSHOTS_DIR)
    prev_snap_df = pd.read_csv(prev_snap_path) if prev_snap_path else None
    curr_snap_df = pd.read_csv(snapshot)

    prev_div_path, _ = whats_changed._two_newest(DIVERGENCE_SNAPS_DIR)
    prev_div_df = pd.read_csv(prev_div_path) if prev_div_path else None
    curr_div_df = pd.read_csv(TRIPLE_PATH) if TRIPLE_PATH.exists() else None

    title_movers   = whats_changed.compute_title_movers(prev_snap_df, curr_snap_df)
    advance_movers = whats_changed.compute_advance_movers(prev_snap_df, curr_snap_df)
    fresh_divs = (
        whats_changed.compute_fresh_divergences(prev_div_df, curr_div_df)
        if curr_div_df is not None else []
    )

    for m in title_movers + advance_movers:
        m["team_display"] = disp(m["team"])
        delta_pp = round(m["delta"] * 100, 1)
        m["delta_fmt"] = f"+{delta_pp:.1f}pp" if delta_pp > 0 else f"{delta_pp:.1f}pp"

    for d in fresh_divs:
        d["home_display"] = disp(d["home_team"])
        d["away_display"] = disp(d["away_team"])
        d["match_url"] = (
            f"matches/{match_key(d['date'], d['home_team'], d['away_team'])}.html"
        )
        d["div_type_label"] = DIV_LABELS.get(d["divergence_type"], d["divergence_type"])
        d["magnitude_fmt"] = f"{round(d['magnitude'] * 100)}pp"

    env = _build_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- index (top-level: root="") ---
    _render_page(
        env, "index.html", OUTPUT_DIR / "index.html",
        teams=teams,
        survival_labels=[label for _, label in SURVIVAL_COLS],
        title_movers=title_movers,
        advance_movers=advance_movers,
        fresh_divergences=fresh_divs,
        root="", generated_at=generated_at, snapshot_date=snapshot_date,
    )

    # --- match pages (one folder deep: root="../") ---
    for m in matches:
        _render_page(
            env, "match.html", MATCHES_OUT / f"{m['key']}.html",
            m=m,
            root="../", generated_at=generated_at, snapshot_date=snapshot_date,
        )

    _copy_static()
    (OUTPUT_DIR / ".nojekyll").touch()

    print(f"Built site -> {OUTPUT_DIR}/")
    print(f"   snapshot : {snapshot.name} ({len(teams)} teams)")
    print(f"   pages    : index.html + {len(matches)} match pages")


if __name__ == "__main__":
    build_site()