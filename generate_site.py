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
import bracket_resolve
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

# Knockout-round date windows (display only — the model/sim are date-agnostic and
# KO fixtures aren't in the pipeline yet, so these live here, not in a data file).
# Keyed by the exact round labels resolve_bracket() emits, plus the third-place
# play-off node. Used to date the next-round card (index) and the KO schedule.
KO_ROUND_DATES: dict[str, tuple[dt.date, dt.date]] = {
    "Round of 32":          (dt.date(2026, 6, 28), dt.date(2026, 7, 3)),
    "Round of 16":          (dt.date(2026, 7, 4),  dt.date(2026, 7, 7)),
    "Quarter-finals":       (dt.date(2026, 7, 9),  dt.date(2026, 7, 11)),
    "Semi-finals":          (dt.date(2026, 7, 14), dt.date(2026, 7, 15)),
    "Third-place play-off": (dt.date(2026, 7, 18), dt.date(2026, 7, 18)),
    "Final":                (dt.date(2026, 7, 19), dt.date(2026, 7, 19)),
}

# Group fixtures (Jun 11–27) and the knockouts (Jun 28+) occupy disjoint date
# ranges, so "date >= this" cleanly tells a KO match from a group match — robust
# against the §6 case where two same-group teams meet again in the KO (the
# unordered pair alone can't distinguish that group match from the KO rematch;
# the date can).
_KO_START_ISO = KO_ROUND_DATES["Round of 32"][0].isoformat()

# Compact round labels for the schedule's date chip (full labels are too long).
_KO_STAGE_SHORT = {
    "Round of 32": "R32", "Round of 16": "R16", "Quarter-finals": "QF",
    "Semi-finals": "SF", "Third-place play-off": "3rd", "Final": "Final",
}


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


def _ko_date_range(label: str) -> str:
    """KO round label -> '28 Jun – 3 Jul' (single day -> '18 Jul'). No %-d
    (Windows-safe); en dash with surrounding spaces, matching the date style."""
    start, end = KO_ROUND_DATES[label]
    fmt = lambda d: f"{d.day} {d.strftime('%b')}"
    return fmt(start) if start == end else f"{fmt(start)} – {fmt(end)}"


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


# ----------------------------------------------------------------------
# Survival-grid alive/eliminated split (Session INDEX-KO). Once the knockout
# stage arrives the grid is dominated by teams already out, so we split it: teams
# still able to win (champion prob > 0) stay full-size at the top; eliminated
# teams collapse into a <details>, grouped by the round they went out in. The
# exit round is read from the newest snapshot's realized reached-round probs
# (pinned results make them exact 1.0/0.0, but we compare with a tolerance,
# never float equality).
# ----------------------------------------------------------------------
_ALIVE_TOL = 1e-9        # champion prob above this => still alive
_REALIZED  = 0.999       # a realized (pinned) reached-round 1.0, noise-tolerant

# Reached-round columns, shallow -> deep. The deepest one that reads as realized
# (>= _REALIZED) is the round a team last reached, i.e. where it went out.
_REACH_COLS = ["p_advance", "p_r16", "p_qf", "p_sf", "p_final"]

# Exit buckets, DEEPEST first (render order inside the eliminated <details>).
# Keys are the reached-round column ("group" = never advanced); labels are shown.
_EXIT_ORDER: list[tuple[str, str]] = [
    ("p_final",   "Out in the final"),
    ("p_sf",      "Out in the semi-finals"),
    ("p_qf",      "Out in the quarter-finals"),
    ("p_r16",     "Out in the Round of 16"),
    ("p_advance", "Out in the Round of 32"),
    ("group",     "Out in the group stage"),
]


def _grid_bucket(rec: dict) -> str:
    """Return 'alive' (champion prob > 0) or the exit-round bucket key: the
    deepest realized reached-round column, or 'group' if the team never advanced.
    Pure — a snapshot record in, a bucket key out."""
    if rec["p_champion"] > _ALIVE_TOL:
        return "alive"
    deepest = "group"
    for col in _REACH_COLS:
        if rec[col] >= _REALIZED:
            deepest = col
    return deepest


def _load_teams(snapshot: Path) -> dict:
    """Load a snapshot into the render-ready survival grid, split into teams
    still alive (full-size, ranked) and eliminated teams grouped by exit round.

    Returns {"alive": [team dict], "eliminated": [{label, teams}],
    "eliminated_count": int}. Each team dict carries the display name, group
    letter, the five survival cells (text + green-tier class), and the champion
    cell (text + gold-tier class). The template just iterates.
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

    def _render(rec: dict) -> dict:
        name = rec["team"]
        survival = [
            {"text": _fmt(rec[col]), "tier": f"t-surv-{_tier(rec[col], _SURV_CUTS)}"}
            for col, _label in SURVIVAL_COLS
        ]
        champ_p = rec["p_champion"]
        return {
            "name":  disp(name),
            "group": TEAM_GROUP.get(name, "?"),
            "survival": survival,
            "champ": {
                "text": _fmt(champ_p),
                "tier": f"t-champ-{_tier(champ_p, _CHAMP_CUTS)}",
            },
        }

    alive: list[dict] = []
    by_bucket: dict[str, list[dict]] = {}
    for rec in records:                       # already in cascade order
        bucket = _grid_bucket(rec)
        if bucket == "alive":
            alive.append(_render(rec))
        else:
            by_bucket.setdefault(bucket, []).append(_render(rec))

    eliminated = [
        {"label": label, "teams": by_bucket[key]}
        for key, label in _EXIT_ORDER if by_bucket.get(key)
    ]
    return {
        "alive": alive,
        "eliminated": eliminated,
        "eliminated_count": sum(len(g["teams"]) for g in eliminated),
    }


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


def _build_match(row, ko_stage: dict | None = None) -> dict:
    """Turn one triple_compare row into a render-ready match context."""
    home, away = row["home_team"], row["away_team"]
    home_d, away_d = disp(home), disp(away)
    key = match_key(row["date"], home, away)
    stage, stage_short = _match_stage(row["date"], home, away, ko_stage or {})

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
        "stage":        stage,
        "stage_short":  stage_short,
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


def _build_played_match(rec: dict, ko_stage: dict | None = None) -> dict:
    """Turn one scored ledger row into a render-ready match context. The
    probability bars are the FROZEN pre-match forecast; the preview and
    divergence caches persist on disk keyed by match_key."""
    home, away = rec["home_team"], rec["away_team"]
    home_d, away_d = disp(home), disp(away)
    key = rec["match_key"]
    stage, stage_short = _match_stage(rec["date"], home, away, ko_stage or {})

    # neutral_used arrives as bool, "True"/"False" string, or NaN (legacy)
    neutral_used = str(rec.get("neutral_used")).lower() != "false"
    venue_label = "neutral venue" if neutral_used else f"home venue — {home_d}"

    hg, ag = rec.get("actual_home_score"), rec.get("actual_away_score")
    score = f"{int(hg)}–{int(ag)}" if pd.notna(hg) and pd.notna(ag) else None

    # Full-result annotation (e.g. "won 4–3 on penalties", "2–1 a.e.t."): the
    # score above grades on 90 minutes; this is display-only. Blank/NaN -> None.
    _note = rec.get("result_note")
    result_note = (str(_note).strip()
                   if pd.notna(_note) and str(_note).strip() != "" else None)

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
        "result_note":  result_note,
        "verdict":      _verdict(rec),
        "home_display": home_d,
        "away_display": away_d,
        "group":        TEAM_GROUP.get(home, "?"),
        "stage":        stage,
        "stage_short":  stage_short,
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
    ko_stage = _ko_stage_map()
    df = pd.read_csv(TRIPLE_PATH)
    matches = [
        _build_match(row, ko_stage)
        for _, row in df.iterrows()
        if match_key(row["date"], row["home_team"], row["away_team"]) not in played
    ]
    matches += [_build_played_match(rec, ko_stage) for rec in played.values()]
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


# Named-round display order for the stage filter (tournament progression, not
# alphabetical). "Knockout" is the defensive coarse fallback; anything unknown
# sorts last.
_STAGE_ORDER = ["Group", "Round of 32", "Round of 16", "Quarter-final",
                "Semi-final", "Third place", "Final", "Knockout"]

# Rolling-chart per-source colours — existing tokens only (no new color tokens).
_DIVLOG_SRC_COLOR = {"": "var(--accent)",       # Model  — pitch green
                     "_book": "var(--accent-gold)",  # Sportsbook — gold
                     "_poly": "var(--away)"}     # Polymarket — clay
_DIVLOG_THIN_N = 10   # per-bucket "early, n=X" caveat threshold


def _fmt_src_brier(sc: dict) -> dict:
    """Format a _source_scores dict (brier + n) for a bucket cell."""
    return {"brier": "—" if sc["brier"] is None else f"{sc['brier']:.3f}",
            "n": sc["n"]}


def _hof_note(shown: int, total: int) -> str:
    """Honest count phrase for a hall-of-fame list (same thin-n register as the
    buckets/scoreboard). "" when empty, "n=N" when the whole list is shown,
    "showing top S of N" when truncated."""
    if total == 0:
        return ""
    if total <= shown:
        return f"n={total}"
    return f"showing top {shown} of {total}"


# Rolling-calibration chart geometry (hand-built inline SVG, _cal_svg idiom).
_ROLL_W, _ROLL_H = 660, 300
_ROLL_PAD_L, _ROLL_PAD_R, _ROLL_PAD_T, _ROLL_PAD_B = 48, 90, 16, 34
# Omit the first few cumulative points: a running Brier over 1-4 results is
# noise-dominated and stretches the axis, compressing the meaningful late
# convergence. Lines are DRAWN from match 5 onward, but the plotted values stay
# the true cumulative (the earlier matches are still counted, just not drawn).
_ROLL_DRAWN_START = 4   # 0-based match index; draw from the 5th match onward


def _divlog_rolling_svg(rolling: dict) -> dict | None:
    """Convert the cumulative-Brier series to SVG pixel coords (three lines on a
    shared match-progression x-axis; lower y = better calibration). Drawn from
    match 5 onward (_ROLL_DRAWN_START); the y-axis is fit to the drawn band, not
    floor-clamped. Returns None when there's too little to plot."""
    if not rolling:
        return None
    drawn = {s["suffix"]: [p for p in s["points"] if p["i"] >= _ROLL_DRAWN_START]
             for s in rolling["series"]}
    pts_all = [p for pts in drawn.values() for p in pts]
    if len(pts_all) < 2:
        return None
    idxs = [p["i"] for p in pts_all]
    i_lo, i_hi = min(idxs), max(idxs)
    if i_hi == i_lo:
        return None
    ys = [p["brier"] for p in pts_all]
    lo_raw, hi_raw = min(ys), max(ys)
    span = max(hi_raw - lo_raw, 0.02)
    ylo, yhi = lo_raw - span * 0.12, hi_raw + span * 0.12   # fit the drawn band
    plot_w = _ROLL_W - _ROLL_PAD_L - _ROLL_PAD_R
    plot_h = _ROLL_H - _ROLL_PAD_T - _ROLL_PAD_B
    base_y = _ROLL_PAD_T + plot_h

    def sx(i: int) -> float:
        return round(_ROLL_PAD_L + (i - i_lo) / (i_hi - i_lo) * plot_w, 1)

    def sy(b: float) -> float:
        return round(_ROLL_PAD_T + (yhi - b) / (yhi - ylo) * plot_h, 1)

    series = []
    for s in rolling["series"]:
        pts = drawn[s["suffix"]]
        if not pts:
            continue
        last = pts[-1]
        series.append({
            "name": s["name"],
            "color": _DIVLOG_SRC_COLOR[s["suffix"]],
            "points": " ".join(f"{sx(p['i'])},{sy(p['brier'])}" for p in pts),
            "end_x": sx(last["i"]),
            "end_y": sy(last["brier"]),
            "end_label": f"{last['brier']:.3f}",
        })

    yticks = [{"y": sy(ylo + (yhi - ylo) * t / 2),
               "label": f"{ylo + (yhi - ylo) * t / 2:.2f}"} for t in range(3)]

    model_pts = drawn[""]   # Model prices every match -> spans the full drawn range
    return {
        "w": _ROLL_W, "h": _ROLL_H,
        "pad_l": _ROLL_PAD_L, "pad_r": _ROLL_PAD_R,
        "pad_t": _ROLL_PAD_T, "base_y": base_y,
        "plot_h": plot_h,
        "series": series,
        "yticks": yticks,
        "x0_label": _date_human(model_pts[0]["date"]) if model_pts else "",
        "x1_label": _date_human(model_pts[-1]["date"]) if model_pts else "",
    }


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

    # Distinct stage values present, ordered by tournament progression.
    stages = sorted({r["stage"] for r in rows},
                    key=lambda s: (_STAGE_ORDER.index(s)
                                   if s in _STAGE_ORDER else len(_STAGE_ORDER)))

    scoreboard = out["scoreboard"]
    if scoreboard:
        for s in scoreboard["sources"]:
            s["win_rate_fmt"] = ("—" if s["win_rate"] is None
                                 else f"{s['win_rate'] * 100:.0f}%")
            s["brier_fmt"] = "—" if s["brier"] is None else f"{s['brier']:.3f}"
            s["log_loss_fmt"] = "—" if s["log_loss"] is None else f"{s['log_loss']:.3f}"

    # Gap-size buckets (Part B): the site's thesis made measurable.
    buckets = []
    for b in out["buckets"]:
        buckets.append({
            "label": b["label"],
            "n": b["n"],
            "model_wins": b["model_wins"],
            "model": _fmt_src_brier(b["model"]),
            "book": _fmt_src_brier(b["book"]),
            "poly": _fmt_src_brier(b["poly"]),
            "caveat": f"early, n={b['n']}" if 0 < b["n"] < _DIVLOG_THIN_N else "",
        })

    # Hall of fame (DIVLOG-3): a filter+sort over the SAME row dicts, so the
    # entries carry the display fields the enrichment loop above already added
    # (match_label / match_url / date_human / p_*_fmt / stage / score).
    hof = out["hall_of_fame"]

    context = {
        "rows": rows,
        "stages": stages,
        "gap_buckets": [(key, label) for key, label, _lo, _hi in divergence_log.GAP_BUCKETS],
        "div_types": list(DIV_LABELS.items()),
        "scoreboard": scoreboard,
        "n_played": out["n_played"],
        "buckets": buckets,
        "rolling_svg": _divlog_rolling_svg(out["rolling"]),
        "hall_of_fame": hof,
        "hof_wins_note": _hof_note(len(hof["wins"]), hof["n_wins"]),
        "hof_misses_note": _hof_note(len(hof["misses"]), hof["n_misses"]),
    }
    csv_df = pd.DataFrame(csv_records)
    return context, csv_df


# ----------------------------------------------------------------------
# Knockout bracket page (Session 38b) — survival grid → populated bracket.
# Pure consumer of src/bracket_resolve (which reads played results only and
# imports nothing from the sim/pipeline). The page always builds; the gate
# below decides whether it shows the bracket or a placeholder.
# ----------------------------------------------------------------------

def _group_stage_complete() -> bool:
    """True once no unplayed GROUP fixture remains, read from the live
    triple_compare frontier (played fixtures drop out of triple_compare, §6).
    A fixture is group-stage iff its two teams share a 2026 group; any such
    pair still present means the group stage is unfinished. Independent of
    bracket_resolve's own completeness flag so the gate's AND is genuinely
    defensive (this reads triple_compare; bracket_resolve reads matches_clean)."""
    if not TRIPLE_PATH.exists():
        return False
    df = pd.read_csv(TRIPLE_PATH)
    for _, r in df.iterrows():
        gh, ga = TEAM_GROUP.get(r["home_team"]), TEAM_GROUP.get(r["away_team"])
        if gh is not None and gh == ga:
            return False  # an unplayed group fixture remains
    return True


def _team_slot(name, winner, played: bool) -> dict:
    """One team line in a bracket cell. None -> a muted TBD; once played, the
    advancing side is flagged 'won' and the eliminated side 'out'."""
    if not name:
        return {"name": "TBD", "tbd": True, "won": False, "out": False}
    return {"name": name, "tbd": False,
            "won": played and name == winner,
            "out": played and name != winner}


# Knockout topology: match_id -> (feeder_a_id, feeder_b_id) for every node above
# the Round of 32. Duplicated from bracket.py's R16/QF/SF/FINAL tables rather than
# imported (Session-24 precedent: the generator stays decoupled from the model
# layer, so this small immutable tree is restated here). R32 ids 73-88 are the 16
# leaves; everything funnels up to the final (104). The third-place play-off (103)
# renders in its own detached block, not in the grid, so it is deliberately absent.
_KO_FEEDERS: dict[int, tuple[int, int]] = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    101: (97, 98), 102: (99, 100),
    104: (101, 102),
}
_KO_FINAL_ID = 104


def _ko_round_col(mid: int) -> int:
    """Grid column (round depth, 1=R32 … 5=Final) for a knockout match id."""
    if 73 <= mid <= 88:
        return 1
    if 89 <= mid <= 96:
        return 2
    if 97 <= mid <= 100:
        return 3
    if mid in (101, 102):
        return 4
    if mid == _KO_FINAL_ID:
        return 5
    raise ValueError(f"not a bracket-grid match id: {mid}")


def _ko_layout() -> dict[int, dict]:
    """Pure, topology-only single-grid placement for every knockout match
    (R32→final), so each later-round cell vertically spans — and therefore centres
    on — its two feeders (the classic funnel, exact at every level). Independent of
    which teams/results are loaded, so a still-TBD R16+ cell lands at the correct
    centred row-span.

    Leaf order comes from a depth-first walk of the final's subtree, which
    guarantees every match's descendants occupy a contiguous leaf run (planar, no
    crossing lines).

    Returns {match_id: {"col", "row_start", "row_end"}} where row_start/row_end are
    1-based CSS grid-row lines into a grid whose row 1 is the round-header row and
    rows 2..17 are the 16 leaf rows (so a leaf at 0-based position p spans grid line
    p+2 to p+3; a cell over leaves lo..hi spans lo+2 to hi+3).
    """
    leaf_index: dict[int, int] = {}

    def collect(mid: int) -> None:
        if mid not in _KO_FEEDERS:           # R32 leaf
            leaf_index[mid] = len(leaf_index)
            return
        a, b = _KO_FEEDERS[mid]
        collect(a)                           # depth-first: a's subtree, then b's
        collect(b)

    collect(_KO_FINAL_ID)

    span: dict[int, tuple[int, int]] = {}

    def resolve(mid: int) -> tuple[int, int]:
        if mid not in _KO_FEEDERS:
            p = leaf_index[mid]
            span[mid] = (p, p)
            return span[mid]
        a, b = _KO_FEEDERS[mid]
        lo_a, hi_a = resolve(a)
        lo_b, hi_b = resolve(b)
        span[mid] = (min(lo_a, lo_b), max(hi_a, hi_b))
        return span[mid]

    resolve(_KO_FINAL_ID)

    return {
        mid: {"col": _ko_round_col(mid), "row_start": lo + 2, "row_end": hi + 3}
        for mid, (lo, hi) in span.items()
    }


def _load_bracket(pair_to_key: dict) -> dict:
    """Resolve the knockout bracket and make it render-ready: internal names
    mapped through DISPLAY_NAMES, each cell linked to its match page when one
    exists (the fixture has been played and has a ledger-keyed page).

    pair_to_key maps frozenset({home_display, away_display}) -> match_key."""
    data = bracket_resolve.resolve_bracket()
    if not data["complete"]:
        return {"complete": False, "rounds": [], "third_place": None}

    layout = _ko_layout()

    def _cell(node: dict) -> dict:
        ta, tb = disp(node["team_a"]) if node["team_a"] else None, \
                 disp(node["team_b"]) if node["team_b"] else None
        winner = disp(node["winner"]) if node["winner"] else None
        key = None
        if ta and tb:
            key = pair_to_key.get(frozenset({ta, tb}))
        place = layout.get(node["match_id"], {})  # third-place (103) absent — block, not grid
        return {
            "match_id": node["match_id"],
            "played": node["played"],
            "key": key,
            "col": place.get("col"),
            "row_start": place.get("row_start"),
            "row_end": place.get("row_end"),
            "teams": [_team_slot(ta, winner, node["played"]),
                      _team_slot(tb, winner, node["played"])],
        }

    rounds = [
        {"label": r["label"],
         "col": layout[r["matches"][0]["match_id"]]["col"],
         "matches": [_cell(m) for m in r["matches"]]}
        for r in data["rounds"]
    ]
    return {
        "complete": True,
        "rounds": rounds,
        "third_place": _cell(data["third_place"]) if data["third_place"] else None,
    }


def _ko_stage_map() -> dict:
    """{frozenset({team_a, team_b}): round_label} for every populated KO match in
    the resolved bracket (internal names) — so KO match pages / schedule rows are
    labelled by round ("Round of 32") instead of a group letter. Empty until the
    bracket completes. Consumed via a date gate (_KO_START_ISO), so a group match
    between two teams that later meet in the KO is never mislabelled."""
    data = bracket_resolve.resolve_bracket()
    if not data["complete"]:
        return {}
    out: dict = {}
    for rnd in data["rounds"]:
        for m in rnd["matches"]:
            if m["team_a"] and m["team_b"]:
                out[frozenset({m["team_a"], m["team_b"]})] = rnd["label"]
    tp = data.get("third_place")
    if tp and tp["team_a"] and tp["team_b"]:
        out[frozenset({tp["team_a"], tp["team_b"]})] = "Third-place play-off"
    return out


def _match_stage(date_iso, home: str, away: str, ko_stage: dict) -> tuple:
    """(stage, stage_short) for a match: the KO round label when the match is in
    the knockout date window AND its pair is a resolved KO tie, else (None, None)
    so the caller falls back to the group label."""
    if str(date_iso) < _KO_START_ISO:
        return None, None
    stage = ko_stage.get(frozenset({home, away}))
    return stage, (_KO_STAGE_SHORT.get(stage) if stage else None)


def _next_ko_round(bracket: dict) -> dict | None:
    """The shallowest knockout round still to be decided, as a render-ready card.
    Pure consumer of _load_bracket() output — NO clock, NO model import, so it is
    deterministic on a given bracket (the index attaches the clock-based 'Up next'
    vs 'In progress' lead-in separately). No per-match links: KO match pages don't
    exist yet (A-pipeline), so every card points at bracket.html.

    Returns one of:
      {"kind": "round",    "label", "dates", "link"}  — round populated, undecided
      {"kind": "champion", "team",  "link"}           — the final has been played
      {"kind": "pending",  "label", "dates", "link"}  — group stage over but the
                                                         bracket isn't resolved yet
    """
    link = "bracket.html"
    if not bracket.get("complete"):
        return {"kind": "pending", "label": "Knockouts",
                "dates": _ko_date_range("Round of 32"), "link": link}
    final = bracket["rounds"][-1]["matches"][0]
    if final["played"]:
        team = next((t["name"] for t in final["teams"] if t["won"]), None)
        return {"kind": "champion", "team": team, "link": link}
    for r in bracket["rounds"]:
        populated = any(not t["tbd"] for m in r["matches"] for t in m["teams"])
        complete = all(m["played"] for m in r["matches"])
        if populated and not complete:
            return {"kind": "round", "label": r["label"],
                    "dates": _ko_date_range(r["label"]), "link": link}
    # Unreachable while complete and final unplayed (some round is always
    # populated-and-incomplete), but stay defensive rather than return None.
    return {"kind": "pending", "label": "Knockouts",
            "dates": _ko_date_range("Round of 32"), "link": link}


def _ko_schedule_rows(bracket: dict) -> list[dict]:
    """Full remaining KO schedule by round for the schedule page. Pure consumer.
    Each row {label, dates, ties}: `ties` are 'A vs B' display strings for a
    populated round (R32 once the groups finish), empty for a still-TBD round
    (R16+ until their feeders are played — the template renders 'TBD'). The
    third-place play-off is inserted chronologically just before the final."""
    if not bracket.get("complete"):
        return []

    def _row(label: str, matches: list[dict]) -> dict:
        ties = [
            f'{m["teams"][0]["name"]} vs {m["teams"][1]["name"]}'
            for m in matches
            if not m["teams"][0]["tbd"] and not m["teams"][1]["tbd"]
        ]
        return {"label": label, "dates": _ko_date_range(label), "ties": ties}

    rows: list[dict] = []
    for r in bracket["rounds"]:
        if r["label"] == "Final":
            tp = bracket.get("third_place")
            rows.append(_row("Third-place play-off", [tp] if tp else []))
        rows.append(_row(r["label"], r["matches"]))
    return rows


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
    advance_block  = whats_changed.compute_advance_movers(prev_snap_df, curr_snap_df)
    advance_movers = advance_block["movers"]
    advance_label  = advance_block["label"]   # "Advance from group" or "Reach the final"
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

    # --- knockout bracket (Session 38b) ---------------------------------
    # Linkable pair -> match-page key, keyed by the DISPLAY-name pair to match
    # the bracket cells (which carry display names). Built from every fixture
    # context so a played KO match links to its page once one exists.
    pair_to_key = {
        frozenset({m["home_display"], m["away_display"]}): m["key"]
        for m in matches
    }
    bracket = _load_bracket(pair_to_key)
    # Show the populated bracket only when BOTH signals agree the group stage is
    # done: the ledger/triple_compare frontier (no unplayed group fixture) AND
    # bracket_resolve's own completeness (all 16 R32 cells resolved). The two
    # read different data sources, so the AND survives a momentary disagreement.
    show_bracket = _group_stage_complete() and bracket["complete"]

    # A-display: once the group stage finishes, the KO fixtures aren't in the live
    # pipeline yet (that's A-pipeline), so today_block/upcoming_days go empty and
    # the page would read "tournament over." Surface the resolved bracket instead:
    # a compact next-round card on the index, the full remaining KO tree on the
    # schedule. Both are pure consumers of the already-loaded _load_bracket() data.
    # next_round carries no per-match links (KO match pages don't exist yet) — it
    # points at bracket.html. The clock-based "Up next" vs "In progress" lead-in is
    # attached here (kept out of _next_ko_round so that helper stays clock-free).
    next_round = _next_ko_round(bracket) if today_block is None else None
    if next_round and next_round["kind"] == "round":
        start, _ = KO_ROUND_DATES[next_round["label"]]
        next_round["lead_in"] = "In progress" if clock.today() >= start else "Up next"
    ko_schedule = _ko_schedule_rows(bracket) if show_bracket else []

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
        advance_label=advance_label,
        fresh_divergences=fresh_divs,
        top_divergences=top_divergences,
        scoreboard=scoreboard,
        results_block=results_block,
        today_block=today_block,
        next_round=next_round,
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
        ko_schedule=ko_schedule,
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

    # --- knockout bracket page (top-level: root="") ---
    _render_page(
        env, "bracket.html", OUTPUT_DIR / "bracket.html",
        bracket=bracket, show_bracket=show_bracket,
        root="", generated_at=generated_at, snapshot_date=snapshot_date,
        page_title="Bracket — World Cup 2026 knockout tree",
        meta_description=(
            "The 2026 World Cup knockout bracket — Round of 32 through the "
            "final — filled in from results as the tournament unfolds."
        ),
        canonical_url=f"{SITE_URL}/bracket.html",
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
        f"{SITE_URL}/bracket.html",
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
    print(f"   snapshot : {snapshot.name} "
          f"({len(teams['alive'])} alive + {teams['eliminated_count']} eliminated)")
    print(f"   pages    : index.html + bracket.html + title-odds.html + schedule.html + divergence-log.html + calibration.html + methodology.html + {len(matches)} match pages")
    print(f"   bracket  : {'populated' if show_bracket else 'placeholder (group stage in progress)'}")


# ----------------------------------------------------------------------
# Self-test (run: python generate_site.py --test) — keeps the default
# invocation a pure build. _verdict() is the one piece of non-trivial logic
# in this consumer worth pinning.
# ----------------------------------------------------------------------

def _test_grid_bucket() -> None:
    """Survival-grid exit-round classification (Session INDEX-KO)."""
    def rec(champ, adv, r16, qf, sf=0.0, final=0.0):
        return {"p_champion": champ, "p_advance": adv, "p_r16": r16,
                "p_qf": qf, "p_sf": sf, "p_final": final}

    assert _grid_bucket(rec(0.30, 1, 1, 1)) == "alive"          # champion > 0
    assert _grid_bucket(rec(0.0, 1, 1, 0)) == "p_r16"           # reached R16, not QF
    assert _grid_bucket(rec(0.0, 1, 0, 0)) == "p_advance"       # reached R32, not R16
    assert _grid_bucket(rec(0.0, 0, 0, 0)) == "group"           # never advanced
    # robustness as the tournament deepens (no such rows today, but must classify):
    assert _grid_bucket(rec(0.0, 1, 1, 1, 0, 0)) == "p_qf"      # reached QF, out
    assert _grid_bucket(rec(0.0, 1, 1, 1, 1, 0)) == "p_sf"      # reached SF, out
    assert _grid_bucket(rec(0.0, 1, 1, 1, 1, 1)) == "p_final"   # runner-up

    # live-shape counts: 8 alive / 8 R16 / 16 R32 / 16 group.
    from collections import Counter
    rows = ([rec(0.3, 1, 1, 1)] * 8 + [rec(0, 1, 1, 0)] * 8
            + [rec(0, 1, 0, 0)] * 16 + [rec(0, 0, 0, 0)] * 16)
    c = Counter(_grid_bucket(r) for r in rows)
    assert c["alive"] == 8 and c["p_r16"] == 8, c
    assert c["p_advance"] == 16 and c["group"] == 16, c
    print("generate_site.py _grid_bucket self-tests passed")


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


def _test_next_ko_round() -> None:
    """Pin _next_ko_round / _ko_schedule_rows on synthetic brackets — clock-free,
    so deterministic regardless of WC_ASOF_DATE."""
    def slot(name, won=False):
        return {"name": name, "tbd": name is None, "won": won, "out": False}

    def match(a, b, played=False, winner=None):
        return {"match_id": 0, "played": played,
                "teams": [slot(a, played and a == winner),
                          slot(b, played and b == winner)]}

    # Bracket not complete -> pending.
    assert _next_ko_round({"complete": False})["kind"] == "pending"

    # R32 populated, nothing played -> next round is R32.
    r32 = [match("A", "B") for _ in range(16)]
    bk = {"complete": True, "rounds": [
        {"label": "Round of 32", "matches": r32},
        {"label": "Round of 16", "matches": [match(None, None) for _ in range(8)]},
        {"label": "Quarter-finals", "matches": [match(None, None) for _ in range(4)]},
        {"label": "Semi-finals", "matches": [match(None, None) for _ in range(2)]},
        {"label": "Final", "matches": [match(None, None)]},
    ], "third_place": match(None, None)}
    nr = _next_ko_round(bk)
    assert nr["kind"] == "round" and nr["label"] == "Round of 32", nr
    assert nr["dates"] == "28 Jun – 3 Jul", nr
    assert nr["link"] == "bracket.html", nr

    # R32 fully played, R16 populated -> next round skips to R16.
    bk["rounds"][0]["matches"] = [match("A", "B", played=True, winner="A") for _ in range(16)]
    bk["rounds"][1]["matches"] = [match("A", "C") for _ in range(8)]
    assert _next_ko_round(bk)["label"] == "Round of 16", _next_ko_round(bk)

    # Final played -> champion.
    bk2 = {"complete": True, "rounds": [
        {"label": "Round of 32", "matches": [match("A", "B", played=True, winner="A")]},
        {"label": "Final", "matches": [match("A", "Z", played=True, winner="A")]},
    ], "third_place": None}
    champ = _next_ko_round(bk2)
    assert champ["kind"] == "champion" and champ["team"] == "A", champ

    # _ko_schedule_rows: R32 populated as pairs, later rounds TBD (empty ties),
    # third-place inserted before the final.
    rows = _ko_schedule_rows(bk)
    labels = [r["label"] for r in rows]
    assert labels[-2:] == ["Third-place play-off", "Final"], labels
    r32_row = next(r for r in rows if r["label"] == "Round of 32")
    assert r32_row["ties"] and r32_row["ties"][0] == "A vs B", r32_row
    final_row = next(r for r in rows if r["label"] == "Final")
    assert final_row["ties"] == [] and final_row["dates"] == "19 Jul", final_row

    print("generate_site.py _next_ko_round / _ko_schedule_rows self-tests passed")


def _test_ko_layout() -> None:
    """Pin _ko_layout geometry: 16 single-row leaves, every internal node spans —
    and is fed by adjacent feeders covering — exactly its two feeders, final spans
    all 16, columns track round depth."""
    layout = _ko_layout()

    leaves = [m for m in range(73, 89)]
    assert len(layout) == 31, len(layout)  # 16 + 8 + 4 + 2 + 1
    for mid in leaves:
        p = layout[mid]
        assert p["col"] == 1, p
        assert p["row_end"] - p["row_start"] == 1, (mid, p)  # one leaf row
    assert len({(layout[m]["row_start"]) for m in leaves}) == 16  # distinct rows

    # Each internal node = union of its two feeders, and the feeders are adjacent
    # (feeder-a's bottom grid line == feeder-b's top grid line).
    for mid, (a, b) in _KO_FEEDERS.items():
        pa, pb, pm = layout[a], layout[b], layout[mid]
        assert pa["row_end"] == pb["row_start"], (mid, pa, pb)
        assert pm["row_start"] == min(pa["row_start"], pb["row_start"]), (mid, pm)
        assert pm["row_end"] == max(pa["row_end"], pb["row_end"]), (mid, pm)

    final = layout[_KO_FINAL_ID]
    assert final["col"] == 5 and final["row_start"] == 2 and final["row_end"] == 18, final

    cols = {1: range(73, 89), 2: range(89, 97), 3: range(97, 101),
            4: (101, 102), 5: (104,)}
    for col, ids in cols.items():
        for mid in ids:
            assert layout[mid]["col"] == col, (mid, col, layout[mid])

    print("generate_site.py _ko_layout self-tests passed")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test_grid_bucket()
        _test_verdict()
        _test_next_ko_round()
        _test_ko_layout()
        divergence_log._test()   # round-naming + bucket + rolling-series tests
        whats_changed._test()    # movers (stage switch) + top-divergences
    else:
        build_site()