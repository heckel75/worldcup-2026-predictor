"""
Session 23–24: Static site generator.

Reads the processed forecast artifacts and renders Jinja2 templates into
docs/, the folder GitHub Pages serves. This script is a pure CONSUMER of
data: it never imports the model or recomputes anything. The model layer
produces dated CSV snapshots; this turns the latest one into HTML.

Presentation that belongs to the SITE (official display names, group
letters, heatmap tiers) lives here, not in the model — see PROJECT.md §4.

Run from project root:
    python generate_site.py
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path("templates")
STATIC_DIR = Path("static")
SNAPSHOTS_DIR = Path("data/processed/snapshots")
OUTPUT_DIR = Path("docs")


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


def _tier(p: float, cuts: tuple[float, ...]) -> int:
    for i, hi in enumerate(cuts):
        if p < hi:
            return i
    return len(cuts)


def _fmt(p: float) -> str:
    """Probability -> percent string; em-dash for anything that rounds to 0."""
    pct = p * 100
    return "—" if pct < 0.05 else f"{pct:.1f}%"


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
            "name":  DISPLAY_NAMES.get(name, name),
            "group": TEAM_GROUP.get(name, "?"),
            "survival": survival,
            "champ": {
                "text": _fmt(champ_p),
                "tier": f"t-champ-{_tier(champ_p, _CHAMP_CUTS)}",
            },
        })
    return teams


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

    env = _build_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Context shared by every page. `root` is the relative path back to the
    # site root: "" for top-level pages, "../" for pages in a subfolder
    # (matches/ in Session 25). base.html prefixes every link/asset with it.
    common = {
        "root": "",
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "snapshot_date": snapshot_date,
    }

    _render_page(
        env, "index.html", OUTPUT_DIR / "index.html",
        teams=teams,
        survival_labels=[label for _, label in SURVIVAL_COLS],
        **common,
    )

    _copy_static()
    (OUTPUT_DIR / ".nojekyll").touch()

    print(f"✅ Built site → {OUTPUT_DIR}/")
    print(f"   snapshot : {snapshot.name} ({len(teams)} teams)")
    print("   pages    : index.html")


if __name__ == "__main__":
    build_site()