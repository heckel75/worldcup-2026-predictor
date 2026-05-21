"""
Session 23: Static site generator.

Reads the processed forecast artifacts and renders Jinja2 templates into
docs/, the folder GitHub Pages serves. This script is a pure CONSUMER of
data: it never imports the model or recomputes anything. The model layer
produces dated CSV snapshots; this turns the latest one into HTML.

Run from project root:
    python generate_site.py

Output:
    docs/index.html   rendered placeholder page
    docs/style.css    copied from static/
    docs/.nojekyll    tells GitHub Pages to skip Jekyll processing
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
    """Load a snapshot into a list of per-team dicts, ranked by title odds."""
    df = pd.read_csv(snapshot)
    df = df.sort_values("p_champion", ascending=False, kind="mergesort")
    return df.to_dict("records")


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

    _render_page(env, "index.html", OUTPUT_DIR / "index.html", teams=teams, **common)

    _copy_static()
    (OUTPUT_DIR / ".nojekyll").touch()

    print(f"✅ Built site → {OUTPUT_DIR}/")
    print(f"   snapshot : {snapshot.name} ({len(teams)} teams)")
    print("   pages    : index.html")


if __name__ == "__main__":
    build_site()