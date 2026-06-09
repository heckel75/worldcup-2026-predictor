"""
src/make_launch_graphic.py

Render docs/launch.png — a shareable 1200×675 (16:9) social graphic that
shows model vs sportsbook vs Polymarket title odds for the top-8 teams.
Colours and typography follow the on-brand warm-paper palette in style.css.

Re-runnable: always reads the newest snapshot from data/processed/snapshots/.

Run from project root:
    python src/make_launch_graphic.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCS_DIR      = Path("docs")
SNAPSHOTS_DIR = Path("data/processed/snapshots")
SB_PATH       = Path("data/processed/sportsbook_outrights.csv")
PM_PATH       = Path("data/processed/polymarket_outrights.csv")
OUTPUT_PATH   = DOCS_DIR / "launch.png"

# ---------------------------------------------------------------------------
# On-brand palette — values from static/style.css :root
# ---------------------------------------------------------------------------

BG      = "#f7f5ef"   # --bg: warm paper
TEXT    = "#1b1a17"   # --text: near-black ink
MUTED   = "#6f6a5f"   # --muted: secondary text
BORDER  = "#e4e0d6"   # --border: hairlines
GOLD    = "#c98a12"   # model bar    — --accent-gold
GREEN_D = "#1f7a4d"   # sportsbook   — --accent (deep pitch green)
GREEN_L = "#79c08c"   # polymarket   — --t-surv-4 (lighter green)

# Mirror of DISPLAY_NAMES in generate_site.py
DISPLAY_NAMES: dict[str, str] = {
    "Turkey":         "Türkiye",
    "Czech Republic": "Czechia",
    "Ivory Coast":    "Côte d'Ivoire",
    "Cape Verde":     "Cabo Verde",
}

N_TEAMS = 8  # top-N teams by model p_champion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_snapshot() -> Path:
    snaps = sorted(SNAPSHOTS_DIR.glob("*.csv"))
    if not snaps:
        raise FileNotFoundError(f"No snapshots found in {SNAPSHOTS_DIR}.")
    return snaps[-1]


def _disp(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- load and join data -----------------------------------------------
    snap_path = _latest_snapshot()
    print(f"Snapshot: {snap_path.name}")

    snap = (pd.read_csv(snap_path)
              .sort_values("p_champion", ascending=False)
              .head(N_TEAMS)
              .reset_index(drop=True))

    sb = pd.read_csv(SB_PATH)[["team", "p_winner"]].rename(columns={"p_winner": "p_book"})
    pm = pd.read_csv(PM_PATH)[["team", "p_winner"]].rename(columns={"p_winner": "p_poly"})

    df = (snap[["team", "p_champion"]]
            .merge(sb, on="team", how="left")
            .merge(pm, on="team", how="left"))
    df["label"] = df["team"].map(_disp)

    # Reverse so rank 1 (highest model odds) appears at the top of the chart
    df = df.iloc[::-1].reset_index(drop=True)
    n = len(df)

    # ---- bar geometry -----------------------------------------------------
    bar_h   = 0.22    # height of each individual bar
    gap_in  = 0.04    # gap between bars within a team's cluster
    gap_out = 0.28    # gap between team clusters
    grp_h   = 3 * bar_h + 2 * gap_in   # total height of one team's 3-bar cluster

    # Bottom edge of each team group (lowest bar = Polymarket)
    ys_base = [i * (grp_h + gap_out) for i in range(n)]
    # The three bar offsets within each group (bottom to top: poly, book, model)
    bar_offsets = [0, bar_h + gap_in, 2 * (bar_h + gap_in)]
    bar_cols    = ["p_poly",     "p_book",    "p_champion"]
    bar_colors  = [GREEN_L,      GREEN_D,     GOLD]
    bar_names   = ["Polymarket", "Sportsbook", "Model (Dixon-Coles + Elo)"]

    # ---- build figure -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for i, row in df.iterrows():
        for offset, col, color in zip(bar_offsets, bar_cols, bar_colors):
            p = row.get(col, 0.0)
            if pd.isna(p):
                p = 0.0
            y = ys_base[i] + offset
            ax.barh(y, p * 100, height=bar_h, color=color, linewidth=0, zorder=2)
            if p >= 0.005:
                ax.text(p * 100 + 0.3, y, f"{round(p * 100)}%",
                        va="center", ha="left", fontsize=8.5,
                        color=TEXT, fontfamily="monospace")

    # ---- axes styling -----------------------------------------------------
    group_centers = [ys_base[i] + bar_h + gap_in for i in range(n)]
    ax.set_yticks(group_centers)
    ax.set_yticklabels(df["label"], fontsize=11, color=TEXT)
    ax.tick_params(axis="y", left=False, pad=6, length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)

    ax.set_xlabel("Title probability (%)", fontsize=9.5, color=MUTED, labelpad=8)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")

    all_vals = [v for col in bar_cols for v in df[col].dropna()]
    x_max = max(all_vals) * 100 * 1.18 if all_vals else 35
    ax.set_xlim(0, x_max)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.xaxis.grid(True, color=BORDER, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # ---- legend -----------------------------------------------------------
    patches = [
        mpatches.Patch(color=GOLD,    label="Model (Dixon-Coles + Elo)"),
        mpatches.Patch(color=GREEN_D, label="Sportsbook (avg, vig stripped)"),
        mpatches.Patch(color=GREEN_L, label="Polymarket"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9,
              framealpha=0, labelcolor=TEXT, handlelength=1.2, borderpad=0)

    # ---- title / subtitle (dynamic Spain gap) -----------------------------
    spain = df[df["team"] == "Spain"]
    if not spain.empty:
        s = spain.iloc[0]
        model_pct = round(s["p_champion"] * 100)
        book_pct  = round(s["p_book"] * 100) if pd.notna(s.get("p_book")) else "?"
        gap_pp    = round((s["p_champion"] - s["p_book"]) * 100) if pd.notna(s.get("p_book")) else "?"
        subtitle  = (f"Model gives Spain {model_pct}% · sportsbook gives {book_pct}%."
                     f"  A {gap_pp} pp divergence on the same favourite.")
    else:
        subtitle = "Where the statistical model and prediction markets agree — and where they diverge."

    plt.subplots_adjust(top=0.85, bottom=0.10, left=0.17, right=0.97)

    fig.text(0.02, 0.97, "Model vs Market: 2026 World Cup title odds",
             ha="left", va="top", fontsize=15, fontweight="bold", color=TEXT)
    fig.text(0.02, 0.91, subtitle,
             ha="left", va="top", fontsize=10, color=MUTED)
    fig.text(0.99, 0.01, "worldcup.divergencelog.com",
             ha="right", va="bottom", fontsize=8.5, color=MUTED)

    # ---- save -------------------------------------------------------------
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=100, bbox_inches=None, facecolor=BG)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
