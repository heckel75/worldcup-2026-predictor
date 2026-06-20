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
import calibration
import clock
import divergence_log
import make_og_cards
import whats_changed
from verdict import VERDICT_BUCKETS, _verdict

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path("templates")
STATIC_DIR = Path("static")
SNAPSHOTS_DIR = Path("data/processed/snapshots")
DIVERGENCE_SNAPS_DIR = Path("data/processed/divergence_snapshots")
TRIPLE_PATH = Path("data/processed/triple_compare.csv")
SB_OUTRIGHTS_PATH = Path("data/processed/sportsbook_outrights.csv")
PM_OUTRIGHTS_PATH = Path("data/processed/polymarket_outrights.csv")
TITLE_ODDS_TOP_N = 16
PREVIEWS_DIR = Path("data/processed/previews")
DIVERGENCES_DIR = Path("data/processed/divergences")
OUTPUT_DIR = Path("docs")
MATCHES_OUT = OUTPUT_DIR / "matches"
OG_OUT = OUTPUT_DIR / "og"          # per-match Open Graph cards (Session OG)
WC_PREDS_PATH = Path("data/processed/wc_predictions.csv")
# Scored ledger rows needed before the calibration page switches from the
# backtest seed to live WC data — below this a reliability diagram is noise.
MIN_LIVE_N = 24
CUSTOM_DOMAIN = "worldcup.divergencelog.com"
SITE_URL = f"https://{CUSTOM_DOMAIN}"


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

    # Round-depth sort cascade: teams tied on the *printed* title odds (1 dp,
    # matching _fmt) break by how deep they're projected to go — P(final),
    # then SF, QF, R16, advance — all on raw floats. Final tiebreak is the
    # display name (ascending) so the all-zero-tail is deterministic. The
    # champion key is rounded first so the visible top of the table, where
    # title odds differ, is unchanged; only near-ties fall through the cascade.
    def _sort_key(rec: dict) -> tuple:
        return (
            -round(rec["p_champion"] * 100, 1),  # printed precision (1 dp %)
            -rec["p_final"],
            -rec["p_sf"],
            -rec["p_qf"],
            -rec["p_r16"],
            -rec["p_advance"],
            disp(rec["team"]),                   # ascending alphabetical
        )

    records = sorted(df.to_dict("records"), key=_sort_key)

    teams: list[dict] = []
    for rec in records:
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
# Title-odds page (Session TITLE-ODDS) — standalone three-source view of
# every contender's championship probability: model vs sportsbook vs
# Polymarket. Pure consumer: reads the newest MC snapshot (p_champion) and
# the two outright-winner CSVs (p_winner). All three already use internal
# team names, so the join is on the raw name; DISPLAY_NAMES is applied only
# at render. No model import.
# ----------------------------------------------------------------------

def _load_outright(path: Path) -> "pd.Series | None":
    """team -> p_winner Series from an outright CSV, or None if absent.
    Mirrors the Session 25 absent-market handling: a missing file just
    means every source value renders as an em-dash, never a crash."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "team" not in df.columns or "p_winner" not in df.columns:
        return None
    return df.set_index("team")["p_winner"]


def _odds_cell(p, max_p: float) -> dict:
    """One source's title prob -> {text, width%}. Bars share a single scale
    (the largest value across all three sources) so the model's structural
    top-heaviness reads honestly: market bars really are shorter because the
    markets assign the favourites less. Absent value -> em-dash, zero width."""
    if p is None or pd.isna(p):
        return {"text": "—", "w": 0.0, "absent": True}
    return {"text": _fmt(float(p)),
            "w": round(float(p) / max_p * 100, 1) if max_p else 0.0,
            "absent": False}


def _gap_fmt(model_p: float, book_p) -> str:
    """Signed model − sportsbook gap in pp (the launch-hook column). '—' when
    no sportsbook price exists for the team."""
    if book_p is None or pd.isna(book_p):
        return "—"
    pp = (float(model_p) - float(book_p)) * 100
    return f"+{pp:.1f}pp" if pp >= 0 else f"{pp:.1f}pp"


def _load_title_odds(snapshot: Path) -> list[dict]:
    """Top-N contenders by model p_champion, each with the three source
    probabilities (model / sportsbook / Polymarket) and a model−book gap."""
    model = pd.read_csv(snapshot)[["team", "p_champion"]]
    sb = _load_outright(SB_OUTRIGHTS_PATH)
    pm = _load_outright(PM_OUTRIGHTS_PATH)

    top = model.sort_values("p_champion", ascending=False).head(TITLE_ODDS_TOP_N)

    # Shared bar scale: the largest plotted value across all three sources.
    vals = list(top["p_champion"])
    for rec in top.to_dict("records"):
        for series in (sb, pm):
            if series is not None and rec["team"] in series.index:
                vals.append(float(series[rec["team"]]))
    max_p = max(vals) if vals else 1.0

    rows = []
    for rec in top.to_dict("records"):
        team = rec["team"]
        m = float(rec["p_champion"])
        sb_p = float(sb[team]) if sb is not None and team in sb.index else None
        pm_p = float(pm[team]) if pm is not None and team in pm.index else None
        rows.append({
            "team":  disp(team),
            "model": _odds_cell(m, max_p),
            "book":  _odds_cell(sb_p, max_p),
            "poly":  _odds_cell(pm_p, max_p),
            "gap":   _gap_fmt(m, sb_p),
        })
    return rows


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


def _load_divergence(key: str, row, home_name, away_name, flagged: bool = True) -> dict | None:
    """Read divergences/<key>.json if present. Commentary rows get the
    computed headline + Claude paragraph; host 'note' rows get a muted
    caveat; everything else has no file and returns None.

    `flagged` gates the commentary callout on the fixture's LIVE
    divergence status: generate_divergences.py only refreshes flagged
    matches, so a fixture that has since dropped below the flag threshold
    keeps a stale commentary file on disk. Upcoming pages pass the live
    triple_compare flag_divergent; played pages render the frozen
    pre-match commentary unconditionally (default True)."""
    path = DIVERGENCES_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    if rec.get("kind") == "note":
        return None  # stale pre-Session-33 cache; host advantage is now modelled

    if not flagged:
        return None  # de-flagged fixture; suppress the stale commentary

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


# ----------------------------------------------------------------------
# Exact-score view (scoreline grid + top-3 + expected goals)
# Pure consumer: reads persisted columns only — no model import, no Poisson.
# The matrix is [home][away] end-to-end (axis 0 = home goals = vertical).
# Never transpose; fix orientation in the axis mapping, not the stored data.
# ----------------------------------------------------------------------

_SC_CELL  = 34       # px per heatmap cell
_SC_N     = 6        # 6x6 grid, index 5 == "5+"
_SC_PAD_L = 52       # room for y-axis title + tick labels
_SC_PAD_T = 24
_SC_PAD_R = 14
_SC_PAD_B = 42       # room for x-axis ticks + title
_SC_TICKS = ["0", "1", "2", "3", "4", "5+"]


def _scoreline_svg(grid: list[list[float]], home_d: str, away_d: str) -> dict:
    """Pre-compute SVG geometry for the 6x6 scoreline heatmap.

    grid is [home][away]: row i = home goals (vertical), col j = away goals
    (horizontal). Cell shading scales with probability; the i==j diagonal is
    flagged so the template can mark the draw line.
    """
    plot = _SC_N * _SC_CELL
    flat = [float(p) for rowp in grid for p in rowp]
    max_p = max(flat) or 1.0

    cells = []
    for i in range(_SC_N):           # home goals -> vertical (axis 0)
        for j in range(_SC_N):       # away goals -> horizontal (axis 1)
            p = float(grid[i][j])
            rel = p / max_p
            x = _SC_PAD_L + j * _SC_CELL
            y = _SC_PAD_T + i * _SC_CELL
            cells.append({
                "x": x, "y": y, "w": _SC_CELL,
                "tx": x + _SC_CELL / 2, "ty": y + _SC_CELL / 2 + 4,
                "opacity": round(rel, 3),
                "diag": i == j,
                "pct": f"{round(p * 100)}" if p >= 0.03 else "",
                "txtcolor": "#ffffff" if rel >= 0.5 else "#1b1a17",
                "label": (f"{home_d} {_SC_TICKS[i]} – {away_d} {_SC_TICKS[j]}: "
                          f"{round(p * 100)}%"),
            })

    xticks = [{"x": _SC_PAD_L + j * _SC_CELL + _SC_CELL / 2,
               "y": _SC_PAD_T + plot + 15, "t": _SC_TICKS[j]}
              for j in range(_SC_N)]
    yticks = [{"x": _SC_PAD_L - 9,
               "y": _SC_PAD_T + i * _SC_CELL + _SC_CELL / 2 + 4, "t": _SC_TICKS[i]}
              for i in range(_SC_N)]

    cy = _SC_PAD_T + plot / 2
    return {
        "w": _SC_PAD_L + plot + _SC_PAD_R,
        "h": _SC_PAD_T + plot + _SC_PAD_B,
        "plot": plot,
        "pad_l": _SC_PAD_L, "pad_t": _SC_PAD_T,
        "cells": cells, "xticks": xticks, "yticks": yticks,
        "x_axis_label": f"{away_d} goals →",
        "x_axis_x": _SC_PAD_L + plot / 2,
        "x_axis_y": _SC_PAD_T + plot + 36,
        "y_axis_label": f"{home_d} goals ↓",
        "y_axis_x": 14, "y_axis_y": cy, "y_axis_rot": f"rotate(-90 14 {cy})",
        "aria": (f"Scoreline probability heatmap: {home_d} goals (rows) "
                 f"versus {away_d} goals (columns)."),
    }


def _build_scoreline(src, home_d: str, away_d: str) -> dict | None:
    """Render-ready exact-score block from the persisted columns, or None.

    `src` is a triple_compare row (upcoming) or a frozen ledger rec (played);
    both expose lambda_home / lambda_away / scoreline_grid / top_scorelines.
    Any absent/empty value (the 16 pre-existing scoreline-less ledger rows, or
    a future NaN) skips the whole block — a page has the full block or none.
    """
    grid_raw = src.get("scoreline_grid")
    top_raw = src.get("top_scorelines")
    # Valid persisted values are always JSON strings; NaN is a float -> skip.
    if not isinstance(grid_raw, str) or not grid_raw.strip():
        return None
    if not isinstance(top_raw, str) or not top_raw.strip():
        return None
    lam_h, lam_a = src.get("lambda_home"), src.get("lambda_away")
    if lam_h is None or lam_a is None or pd.isna(lam_h) or pd.isna(lam_a):
        return None
    try:
        grid = json.loads(grid_raw)
        tops = json.loads(top_raw)
    except (json.JSONDecodeError, TypeError):
        return None

    lam_h, lam_a = float(lam_h), float(lam_a)
    diff = lam_h - lam_a
    if abs(diff) < 0.005:
        diff_text = "dead level"
    else:
        side = home_d if diff > 0 else away_d
        diff_text = f"{side} +{abs(diff):.2f}"

    top3 = [{"score": str(t["score"]).replace("-", "–"),
             "pct": _fmt(float(t["prob"]))}
            for t in tops]

    return {
        "lam_h": f"{lam_h:.2f}",
        "lam_a": f"{lam_a:.2f}",
        "diff_text": diff_text,
        "top3": top3,
        "svg": _scoreline_svg(grid, home_d, away_d),
    }


def _build_match(row) -> dict:
    """Turn one triple_compare row into a render-ready match context."""
    home, away = row["home_team"], row["away_team"]
    home_d, away_d = disp(home), disp(away)
    key = match_key(row["date"], home, away)

    neutral_used = bool(row.get("neutral_used", True))
    venue_label = "neutral venue" if neutral_used else f"home venue — {home_d}"

    # Model favourite + divergence flag, for the index "today / next up" block.
    _probs = (row["p_home_model_corr"], row["p_draw_model_corr"], row["p_away_model_corr"])
    _fav = ("home", "draw", "away")[max(range(3), key=lambda i: _probs[i])]
    model_fav = home_d if _fav == "home" else away_d if _fav == "away" else "Draw"
    flag_divergent = bool(row["flag_divergent"]) if pd.notna(row.get("flag_divergent")) else False

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
        "played":       False,
        "score":        None,
        "home_display": home_d,
        "away_display": away_d,
        "group":        TEAM_GROUP.get(home, "?"),
        "date_iso":     row["date"],
        "date_human":   _date_human(row["date"]),
        "venue_label":  venue_label,
        "sources":      sources,
        "divergence":   _load_divergence(key, row, home_d, away_d, flagged=flag_divergent),
        "scoreline":    _build_scoreline(row, home_d, away_d),
        "model_fav":    model_fav,
        "flag_divergent": flag_divergent,
        "preview_paras": _load_preview(key),
    }


# ----------------------------------------------------------------------
# Played matches (Session 36) — rendered from the frozen ledger, never
# from triple_compare.csv (played fixtures drop out of the live pipeline)
# ----------------------------------------------------------------------

_OUTCOME_SLOT = {"H": "home", "D": "draw", "A": "away"}

# _verdict + VERDICT_BUCKETS now live in src/verdict.py (Session DIVLOG-1) so the
# index scoreboard and the Divergence Log attribution scoreboard share one
# implementation. Imported at the top of this module.


def _build_played_match(rec: dict) -> dict:
    """Turn one scored ledger row into a render-ready match context. The
    probability bars are the FROZEN pre-match forecast; the preview and
    divergence caches persist on disk keyed by match_key."""
    home, away = rec["home_team"], rec["away_team"]
    home_d, away_d = disp(home), disp(away)
    key = rec["match_key"]

    # neutral_used arrives as bool, "True"/"False" string, or NaN (legacy)
    neutral_used = str(rec.get("neutral_used")).lower() != "false"
    venue_label = "neutral venue" if neutral_used else f"home venue — {home_d}"

    hg, ag = rec.get("actual_home_score"), rec.get("actual_away_score")
    score = f"{int(hg)}–{int(ag)}" if pd.notna(hg) and pd.notna(ag) else None

    sources = [_source("Model", rec["p_home"], rec["p_draw"], rec["p_away"],
                       home_d, away_d, sub="bias-corrected")]
    if pd.notna(rec.get("p_home_book")):
        sources.append(_source("Sportsbook",
                               rec["p_home_book"], rec["p_draw_book"],
                               rec["p_away_book"], home_d, away_d,
                               sub="vig stripped"))
    else:
        sources.append(_absent("Sportsbook", "No consensus market was posted pre-match."))
    if pd.notna(rec.get("p_home_poly")):
        sources.append(_source("Polymarket",
                               rec["p_home_poly"], rec["p_draw_poly"],
                               rec["p_away_poly"], home_d, away_d))
    else:
        sources.append(_absent("Polymarket", "No per-match market was posted pre-match."))

    # _load_divergence/_headline_gap expect triple_compare column names; the
    # frozen ledger values are exactly the corrected-model + book probs.
    aliased = {
        "p_home_model_corr": rec["p_home"],
        "p_draw_model_corr": rec["p_draw"],
        "p_away_model_corr": rec["p_away"],
        "p_home_book": rec.get("p_home_book"),
        "p_draw_book": rec.get("p_draw_book"),
        "p_away_book": rec.get("p_away_book"),
    }

    return {
        "key":          key,
        "played":       True,
        "score":        score,
        "verdict":      _verdict(rec),
        "home_display": home_d,
        "away_display": away_d,
        "group":        TEAM_GROUP.get(home, "?"),
        "date_iso":     rec["date"],
        "date_human":   _date_human(rec["date"]),
        "venue_label":  venue_label,
        "sources":      sources,
        "divergence":   _load_divergence(key, aliased, home_d, away_d),
        "scoreline":    _build_scoreline(rec, home_d, away_d),
        "preview_paras": _load_preview(key),
    }


def _load_played() -> dict[str, dict]:
    """Scored ledger rows keyed by match_key. A fixture is played iff its
    ledger row has a result attached — never inferred from dates or from
    triple_compare (which drops played fixtures)."""
    if not WC_PREDS_PATH.exists():
        return {}
    ledger = pd.read_csv(WC_PREDS_PATH)
    if "outcome" not in ledger.columns:
        return {}
    return {
        rec["match_key"]: rec
        for rec in ledger.to_dict("records")
        if rec.get("outcome") in _OUTCOME_SLOT
    }


def _by_date(matches: list[dict]) -> list[dict]:
    """Bucket match contexts by date_iso, chronological — for the schedule page."""
    by_date: dict[str, list[dict]] = {}
    for m in matches:
        by_date.setdefault(m["date_iso"], []).append(m)
    return [
        {"date_iso": d, "date_human": _date_human(d), "fixtures": by_date[d]}
        for d in sorted(by_date)
    ]


def _load_matches() -> list[dict]:
    """All 72 fixtures: unplayed from triple_compare.csv (live forecast),
    played from the frozen ledger (clean_data moves scored fixtures out of
    the live pipeline, so triple_compare no longer carries them). If a key
    somehow appears in both, the played/ledger version wins."""
    if not TRIPLE_PATH.exists():
        raise FileNotFoundError(
            f"{TRIPLE_PATH} not found. Run `python src/triple_compare.py` first."
        )
    played = _load_played()
    df = pd.read_csv(TRIPLE_PATH)
    matches = [
        _build_match(row)
        for _, row in df.iterrows()
        if match_key(row["date"], row["home_team"], row["away_team"]) not in played
    ]
    matches += [_build_played_match(rec) for rec in played.values()]
    matches.sort(key=lambda m: (m["date_iso"], m["key"]))
    return matches


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def _build_env() -> Environment:
    """Jinja2 environment rooted at templates/, with HTML autoescaping on."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    # Daily-update cadence stamp, exposed to every template. Routed through
    # clock.today() so it respects WC_ASOF_DATE during dry runs / as-of tests.
    # No %-d / %-m — those break on Windows (Session 25 bug).
    _d = clock.today()
    env.globals["build_date"] = f"{_d.day} {_d.strftime('%B %Y')}"
    return env


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


# ----------------------------------------------------------------------
# Calibration page (Session 27)
# ----------------------------------------------------------------------

_CAL_SVG_SIZE = 280
_CAL_SVG_PAD  = 32   # pixels of padding on every side


def _cal_svg(summary: dict) -> dict:
    """Convert reliability bins to SVG pixel coords for the diagram."""
    plot = _CAL_SVG_SIZE - 2 * _CAL_SVG_PAD
    dots = []
    for b in summary["bins"]:
        if pd.isna(b["mean_pred"]) or pd.isna(b["obs_freq"]):
            continue
        x = round(_CAL_SVG_PAD + b["mean_pred"] * plot, 1)
        y = round(_CAL_SVG_SIZE - _CAL_SVG_PAD - b["obs_freq"] * plot, 1)
        r = max(5, min(12, b["n"] // 8))
        dots.append({
            "x": x, "y": y, "r": r,
            "label": (f"{round(b['mean_pred'] * 100)}% predicted, "
                      f"{round(b['obs_freq'] * 100)}% actual "
                      f"(n={b['n']})"),
        })
    return {
        "size": _CAL_SVG_SIZE,
        "pad":  _CAL_SVG_PAD,
        "plot": plot,
        "dots": dots,
        "diag": {
            "x1": _CAL_SVG_PAD, "y1": _CAL_SVG_SIZE - _CAL_SVG_PAD,
            "x2": _CAL_SVG_SIZE - _CAL_SVG_PAD, "y2": _CAL_SVG_PAD,
        },
    }


def _fmt_cal(summary: dict) -> dict:
    """Pre-format a calibration summary dict for the template."""
    rows = []
    for r in summary["per_outcome"]:
        gap_pp = r["gap"] * 100
        rows.append({
            "outcome":   r["outcome"],
            "pred":      f"{r['pred'] * 100:.1f}%",
            "obs":       f"{r['obs'] * 100:.1f}%",
            "gap":       f"+{gap_pp:.1f}pp" if gap_pp >= 0 else f"{gap_pp:.1f}pp",
            "highlight": r["outcome"] == "Draw",
        })
    return {
        "label":    summary["label"],
        "n":        summary["n"],
        "brier":    f"{summary['brier']:.3f}",
        "accuracy": f"{summary['accuracy'] * 100:.1f}%",
        "rows":     rows,
    }


def _methodology_stats(primary_sum: dict, full_sum: dict) -> dict:
    """Stats for the methodology page: live numbers from calibration + raw backtest.

    draw_gap_pp comes from full_sum (the ~4pp bias that drives the match-page correction),
    not from primary_sum (majors only — a noisier ~9pp on a smaller sample).
    """
    draw_row = next(r for r in full_sum["per_outcome"] if r["outcome"] == "Draw")
    draw_gap_pp = round(abs(draw_row["gap"]) * 100)

    # Euro-only accuracy: filter raw backtest, reuse calibration.accuracy()
    bt_raw = pd.read_csv("data/processed/backtest_2024.csv")
    euro_raw = bt_raw[bt_raw["tournament"] == "UEFA Euro"].copy()
    euro_df = euro_raw[["home_team", "away_team", "p_home", "p_draw", "p_away"]].copy()
    euro_df["outcome"] = euro_raw["actual"].map({0: "H", 1: "D", 2: "A"})
    euro_acc = calibration.accuracy(euro_df)

    return {
        "n":             primary_sum["n"],
        "accuracy":      f"{primary_sum['accuracy'] * 100:.1f}%",
        "brier":         f"{primary_sum['brier']:.3f}",
        "log_loss":      f"{primary_sum['log_loss']:.3f}",
        "euro_accuracy": f"{euro_acc * 100:.1f}%",
        "draw_gap_pp":   draw_gap_pp,
    }


# ----------------------------------------------------------------------
# Divergence Log (Session DIVLOG-1) — archive table + attribution scoreboard.
# generate_site is the pure consumer/renderer; src/divergence_log.py owns the
# computation (it returns internal names, this applies DISPLAY_NAMES + links).
# ----------------------------------------------------------------------

_GAP_BUCKET_LABEL = {key: label for key, label, _lo, _hi in divergence_log.GAP_BUCKETS}

# Verdict winner -> archive display label (markets/wash/all_missed are the
# no-credit buckets; mirrors the index scoreboard's vocabulary).
_VERDICT_LABEL = {
    "model":      "Model closest",
    "books":      "Sportsbook closest",
    "Polymarket": "Polymarket closest",
    "markets":    "Markets closest",
    "wash":       "Sources agreed",
    "all_missed": "All missed",
}


def _divlog_prob_fmt(p: float) -> str:
    return "—" if pd.isna(p) else f"{round(p * 100)}%"


def _build_divlog() -> tuple[dict, pd.DataFrame]:
    """Enrich divergence_log.build() output for the template + flat CSV.

    Returns (template_context, csv_dataframe). CSV carries display names so the
    public export is readable; the template gets the same rows plus link/data
    attributes for client-side filtering and sorting."""
    out = divergence_log.build(WC_PREDS_PATH)
    rows = out["rows"]

    csv_records = []
    for r in rows:
        r["home_display"] = disp(r["home_team"])
        r["away_display"] = disp(r["away_team"])
        r["match_label"] = f"{r['home_display']} v {r['away_display']}"
        r["match_url"] = f"matches/{r['match_key']}.html"
        r["date_human"] = _date_human(r["date"])
        r["p_model_fmt"] = _divlog_prob_fmt(r["p_model"])
        r["p_book_fmt"] = _divlog_prob_fmt(r["p_book"])
        r["p_poly_fmt"] = _divlog_prob_fmt(r["p_poly"])
        r["gap_pp"] = None if pd.isna(r["gap"]) else round(r["gap"] * 100)
        r["gap_fmt"] = "—" if r["gap_pp"] is None else f"{r['gap_pp']}pp"
        r["gap_sort"] = -1 if r["gap_pp"] is None else r["gap_pp"]
        r["bucket_label"] = _GAP_BUCKET_LABEL.get(r["gap_bucket"], "No market")
        r["bucket_attr"] = r["gap_bucket"] or "none"
        r["div_type_label"] = DIV_LABELS.get(r["divergence_type"], "—")
        r["div_type_attr"] = r["divergence_type"] or "none"
        r["verdict_label"] = _VERDICT_LABEL.get(r["verdict_winner"], "—")

        csv_records.append({
            "date": r["date"],
            "match": r["match_label"],
            "stage": r["stage"],
            "group": r["group"],
            "outcome": r["outcome"],
            "result": r["result_label"],
            "score": r["score"] if r["score"] else "",
            "p_model_on_outcome": "" if pd.isna(r["p_model"]) else round(r["p_model"], 4),
            "p_sportsbook_on_outcome": "" if pd.isna(r["p_book"]) else round(r["p_book"], 4),
            "p_polymarket_on_outcome": "" if pd.isna(r["p_poly"]) else round(r["p_poly"], 4),
            "gap_pp": "" if r["gap_pp"] is None else r["gap_pp"],
            "gap_bucket": r["bucket_label"] if r["gap_bucket"] else "",
            "divergence_type": r["divergence_type"],
            "flagged": r["flagged"],
            "verdict": r["verdict_label"],
        })

    # Distinct stage values present, for the filter dropdown.
    stages = sorted({r["stage"] for r in rows})

    scoreboard = out["scoreboard"]
    if scoreboard:
        for s in scoreboard["sources"]:
            s["win_rate_fmt"] = ("—" if s["win_rate"] is None
                                 else f"{s['win_rate'] * 100:.0f}%")
            s["brier_fmt"] = "—" if s["brier"] is None else f"{s['brier']:.3f}"
            s["log_loss_fmt"] = "—" if s["log_loss"] is None else f"{s['log_loss']:.3f}"

    context = {
        "rows": rows,
        "stages": stages,
        "gap_buckets": [(key, label) for key, label, _lo, _hi in divergence_log.GAP_BUCKETS],
        "div_types": list(DIV_LABELS.items()),
        "scoreboard": scoreboard,
        "n_played": out["n_played"],
    }
    csv_df = pd.DataFrame(csv_records)
    return context, csv_df


def build_site() -> None:
    snapshot = _latest_snapshot()
    snapshot_date = snapshot.stem  # filename is the date
    teams = _load_teams(snapshot)
    title_odds = _load_title_odds(snapshot)
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

    # --- current top divergences panel (Session 37) ----------------------
    top_divergences = (
        whats_changed.compute_top_divergences(curr_div_df)
        if curr_div_df is not None else []
    )
    for d in top_divergences:
        d["home_display"] = disp(d["home_team"])
        d["away_display"] = disp(d["away_team"])
        d["match_url"] = (
            f"matches/{match_key(d['date'], d['home_team'], d['away_team'])}.html"
        )
        d["div_type_label"] = DIV_LABELS.get(d["divergence_type"], d["divergence_type"])
        d["magnitude_fmt"] = f"{round(d['magnitude'] * 100)}pp"
        fav = d["fav_outcome"]
        d["favorite"] = ("Draw" if fav == "draw"
                         else d["home_display"] if fav == "home"
                         else d["away_display"])

    # --- played-match scoreboard + schedule buckets (Session 36) ----------
    played_matches = [m for m in matches if m["played"]]
    unplayed_matches = [m for m in matches if not m["played"]]

    # "markets", "wash" and "all_missed" are no-credit buckets — only a distinct
    # source winner advances the per-source tally (the += below is keyed by
    # winner, so every possible winner must exist as a key or a verdict would
    # KeyError). "markets" counts joint-market wins without crediting books or
    # Polymarket individually.
    verdict_counts = {b: 0 for b in VERDICT_BUCKETS}
    for m in played_matches:
        if m["verdict"]:
            verdict_counts[m["verdict"]["winner"]] += 1
    n_verdicts = sum(verdict_counts.values())
    scoreboard = {"n": n_verdicts, **verdict_counts} if n_verdicts else None

    today_iso = clock.today().isoformat()
    today_fixtures = [m for m in unplayed_matches if m["date_iso"] <= today_iso]
    upcoming_days = _by_date([m for m in unplayed_matches if m["date_iso"] > today_iso])
    results_days = _by_date(played_matches)

    # --- index "results + today" block (Session 37) ----------------------
    # Anchored to the ledger PLAY-FRONTIER, not wall-clock: the once-daily
    # morning run can sit on either side of midnight without mis-bucketing.
    #
    #   frontier      = earliest date_iso that still has an unplayed fixture
    #                   (= the current/next slate). This one rule also yields
    #                   the rest-day "next up {date}" case for free — when no
    #                   match is on today, the earliest unplayed date is the
    #                   next slate; no clock.today() fallback branch needed.
    #   today block   = that whole frontier slate (the MID-SLATE rule: a match
    #                   already scored on the frontier date stays here and
    #                   shows its score inline, rather than jumping to results).
    #   results block = the last fully-completed slate STRICTLY BEFORE the
    #                   frontier (every date < frontier is fully played).
    #
    # Played data is ledger-sourced (§6); played/unplayed come off match["played"],
    # which is set only by a ledger row with a result attached. clock.today() is
    # used ONLY for the cosmetic "Today" vs "next up {date}" label — never for
    # which matches land in which half.
    unplayed_dates = [m["date_iso"] for m in unplayed_matches]
    frontier = min(unplayed_dates) if unplayed_dates else None

    if frontier is not None:
        today_block = {
            "is_today": frontier == today_iso,   # label only; bucketing is frontier-based
            "date_human": _date_human(frontier),
            "fixtures": sorted((m for m in matches if m["date_iso"] == frontier),
                               key=lambda m: m["key"]),
        }
    else:
        today_block = None  # tournament complete — nothing unplayed left

    done_dates = [m["date_iso"] for m in played_matches
                  if frontier is None or m["date_iso"] < frontier]
    if done_dates:
        results_date = max(done_dates)
        results_block = {
            "date_human": _date_human(results_date),
            "fixtures": sorted((m for m in played_matches if m["date_iso"] == results_date),
                               key=lambda m: m["key"]),
        }
    else:
        results_block = None

    env = _build_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    og_image = f"{SITE_URL}/launch.png"

    # --- index (top-level: root="") ---
    _render_page(
        env, "index.html", OUTPUT_DIR / "index.html",
        teams=teams,
        survival_labels=[label for _, label in SURVIVAL_COLS],
        title_movers=title_movers,
        advance_movers=advance_movers,
        fresh_divergences=fresh_divs,
        top_divergences=top_divergences,
        scoreboard=scoreboard,
        results_block=results_block,
        today_block=today_block,
        root="", generated_at=generated_at, snapshot_date=snapshot_date,
        page_title="World Cup 2026 Forecast — model vs market predictions",
        meta_description=(
            "A World Cup 2026 forecast dashboard combining a statistical model, "
            "sportsbook odds, and Polymarket prices. See where the three sources "
            "agree — and where they diverge."
        ),
        canonical_url=f"{SITE_URL}/",
        og_image=og_image,
    )

    # --- match pages (one folder deep: root="../") ---
    # Per-match OG card (Session OG): point og:image at the match's own card
    # when it renders, else fall back to launch.png. SOFT — render_og_card
    # never raises, so a card failure can't abort this FATAL build stage.
    for m in matches:
        card_ok = make_og_cards.render_og_card(m, OG_OUT / f"{m['key']}.png")
        m_og_image = f"{SITE_URL}/og/{m['key']}.png" if card_ok else og_image
        _render_page(
            env, "match.html", MATCHES_OUT / f"{m['key']}.html",
            m=m,
            root="../", generated_at=generated_at, snapshot_date=snapshot_date,
            page_title=(
                f"{m['home_display']} vs {m['away_display']}"
                f" — prediction, odds & probabilities | World Cup 2026"
            ),
            meta_description=(
                f"{m['home_display']} vs {m['away_display']} on {m['date_human']}: "
                f"statistical model, sportsbook and Polymarket win probabilities "
                f"for the 2026 World Cup, updated after every match."
            ),
            canonical_url=f"{SITE_URL}/matches/{m['key']}.html",
            og_image=m_og_image,
        )

    # --- calibration page (top-level: root="") ---
    wc_df = calibration.load_wc_predictions(str(WC_PREDS_PATH))
    live_n = len(wc_df) if wc_df is not None else 0
    if live_n >= MIN_LIVE_N:
        primary_sum = calibration.summarize(wc_df, "Live WC predictions")
        use_wc = True
    else:
        primary_sum = calibration.summarize(
            calibration.load_backtest(majors_only=True),
            "Backtest — Euro 2024 + Copa América",
        )
        use_wc = False
    full_sum = calibration.summarize(
        calibration.load_backtest(), "Full backtest (2024, all competitions)"
    )
    _render_page(
        env, "calibration.html", OUTPUT_DIR / "calibration.html",
        primary=_fmt_cal(primary_sum),
        full=_fmt_cal(full_sum),
        svg=_cal_svg(primary_sum),
        use_wc=use_wc,
        live_n=live_n,
        min_live_n=MIN_LIVE_N,
        root="", generated_at=generated_at, snapshot_date=snapshot_date,
        page_title="Calibration — how accurate is the World Cup 2026 model?",
        meta_description=(
            "Reliability diagram and Brier scores showing how well the "
            "World Cup 2026 prediction model is calibrated against actual results."
        ),
        canonical_url=f"{SITE_URL}/calibration.html",
        og_image=og_image,
    )

    # --- methodology page (top-level: root="") ---
    _render_page(
        env, "methodology.html", OUTPUT_DIR / "methodology.html",
        stats=_methodology_stats(primary_sum, full_sum),
        root="", generated_at=generated_at, snapshot_date=snapshot_date,
        page_title="Methodology — how the World Cup 2026 Predictor works",
        meta_description=(
            "How we combine Elo ratings, Dixon-Coles goals model, Monte Carlo "
            "simulation, sportsbook odds, and Polymarket to forecast every "
            "2026 World Cup match."
        ),
        canonical_url=f"{SITE_URL}/methodology.html",
        og_image=og_image,
    )

    # --- title-odds page (top-level: root="") ---
    _render_page(
        env, "title-odds.html", OUTPUT_DIR / "title-odds.html",
        title_odds=title_odds,
        root="", generated_at=generated_at, snapshot_date=snapshot_date,
        page_title="Title odds — World Cup 2026 winner, model vs market",
        meta_description=(
            "Who will win the 2026 World Cup? Championship probability for every "
            "contender from a statistical model, sportsbook odds, and Polymarket "
            "prices, side by side."
        ),
        canonical_url=f"{SITE_URL}/title-odds.html",
        og_image=og_image,
    )

    # --- schedule page (top-level: root="") ---
    _render_page(
        env, "schedule.html", OUTPUT_DIR / "schedule.html",
        today_fixtures=today_fixtures,
        today_human=_date_human(today_iso),
        upcoming_days=upcoming_days,
        results_days=results_days,
        root="", generated_at=generated_at, snapshot_date=snapshot_date,
        page_title="Schedule — World Cup 2026 match dates and predictions",
        meta_description=(
            "Full 2026 World Cup schedule grouped by match day, with links to "
            "statistical predictions, sportsbook odds, and Polymarket prices "
            "for every fixture."
        ),
        canonical_url=f"{SITE_URL}/schedule.html",
        og_image=og_image,
    )

    # --- divergence log (top-level: root="") + flat CSV export ---
    divlog_ctx, divlog_csv = _build_divlog()
    _render_page(
        env, "divergence-log.html", OUTPUT_DIR / "divergence-log.html",
        **divlog_ctx,
        root="", generated_at=generated_at, snapshot_date=snapshot_date,
        page_title="Divergence Log — model vs market, every result scored",
        meta_description=(
            "A match-by-match archive of where the statistical model, sportsbook "
            "odds, and Polymarket agreed and diverged on the 2026 World Cup — and "
            "which source called each result, with per-source Brier and log loss."
        ),
        canonical_url=f"{SITE_URL}/divergence-log.html",
        og_image=og_image,
    )
    divlog_csv.to_csv(OUTPUT_DIR / "divergence-log.csv", index=False,
                      encoding="utf-8")

    _copy_static()
    (OUTPUT_DIR / ".nojekyll").touch()
    (OUTPUT_DIR / "CNAME").write_text(CUSTOM_DOMAIN + "\n")

    # --- sitemap.xml and robots.txt ---
    static_page_urls = [
        f"{SITE_URL}/",
        f"{SITE_URL}/title-odds.html",
        f"{SITE_URL}/schedule.html",
        f"{SITE_URL}/divergence-log.html",
        f"{SITE_URL}/calibration.html",
        f"{SITE_URL}/methodology.html",
    ] + [f"{SITE_URL}/matches/{m['key']}.html" for m in matches]
    sitemap_entries = "".join(
        f"  <url><loc>{url}</loc><lastmod>{snapshot_date}</lastmod></url>\n"
        for url in static_page_urls
    )
    (OUTPUT_DIR / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + sitemap_entries
        + "</urlset>\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "robots.txt").write_text(
        "User-agent: *\nAllow: /\n"
        f"Sitemap: {SITE_URL}/sitemap.xml\n",
        encoding="utf-8",
    )

    print(f"Built site -> {OUTPUT_DIR}/")
    print(f"   snapshot : {snapshot.name} ({len(teams)} teams)")
    print(f"   pages    : index.html + title-odds.html + schedule.html + divergence-log.html + calibration.html + methodology.html + {len(matches)} match pages")


# ----------------------------------------------------------------------
# Self-test (run: python generate_site.py --test) — keeps the default
# invocation a pure build. _verdict() is the one piece of non-trivial logic
# in this consumer worth pinning.
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

    print("generate_site.py _verdict self-tests passed")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test_verdict()
    else:
        build_site()