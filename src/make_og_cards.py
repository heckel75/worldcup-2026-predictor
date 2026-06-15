"""
src/make_og_cards.py

Per-match Open Graph cards (Session OG). Renders one 1200×675 (16:9) PNG per
match page so a shared match link previews THAT match's forecast instead of the
shared launch.png. Display-layer only — never touches the model, sim,
triple_compare, ledger, or update pipeline.

SOFT by contract: render_og_card() NEVER raises. On any failure it logs a
warning and returns False; the caller then leaves that page's og:image pointing
at launch.png. generate_site.py is a FATAL stage in update.py, so a card error
that escaped would block the whole publish — it must not escape.

Palette/aesthetic is the warm-paper launch.png look: the theme constants are
imported from make_launch_graphic (single source of truth — not re-declared).
The three W/D/L bar colours reuse that same launch palette (green / light-green
/ gold), so the card and launch.png read as one family.

Two layouts, chosen by match_ctx["played"]:
  • upcoming → three corrected W/D/L bars + expected-goals line + most-likely score
  • played   → big final score + the three FROZEN pre-match bars (read from the
               ledger fields, never recomputed)

Cache: a SHA1 over (CARD_VERSION + played-state + final score + the three model
probs rounded to 1pp) is written to a .sha1 sidecar next to each PNG. A daily
build re-renders only cards whose key changed; flipping upcoming→played changes
the played-state + score, so the card re-renders automatically. Bump
CARD_VERSION to invalidate every card on a layout change.

Not run directly — imported by generate_site.py. A __main__ smoke test renders
one synthetic card of each kind so the file can be eyeballed standalone:
    python src/make_og_cards.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt

# Theme constants — single source of truth (do NOT re-declare the palette here).
from make_launch_graphic import BG, TEXT, MUTED, GOLD, GREEN_D, GREEN_L

# ---------------------------------------------------------------------------
# Card constants
# ---------------------------------------------------------------------------

CARD_VERSION = "og-v1"          # bump to invalidate every cached card

# Hash sidecars live OUTSIDE the served docs/ tree — pure build machinery,
# gitignored like logs/ (the rendered PNGs are the only served artifact).
# One <match_key>.sha1 per card, mirroring data/processed/previews/<key>.json.
CACHE_DIR = Path("data/processed/og_cache")

WIDTH_IN, HEIGHT_IN, DPI = 12.0, 6.75, 100   # → 1200×675, matches launch.png

# W/D/L bar colours, reused from the launch palette (home / draw / away).
COL_HOME = GREEN_D
COL_DRAW = GREEN_L
COL_AWAY = GOLD


# ---------------------------------------------------------------------------
# Reading the render-ready match context (same dict the page render consumes)
# ---------------------------------------------------------------------------

def _model_probs(m: dict) -> tuple[float, float, float]:
    """(home, draw, away) corrected/frozen model probs, read from the Model
    source's bar segments — the exact values the page shows."""
    src = m["sources"][0]            # Model is always sources[0]
    seg = {s["cls"]: float(s["w"]) / 100.0 for s in src["segments"]}
    return seg["home"], seg["draw"], seg["away"]


def _played_outcome(score: str | None) -> str | None:
    """home / draw / away from a "H–A" final score (en-dash), or None."""
    if not score:
        return None
    try:
        hg, ag = (int(x) for x in score.replace("–", "-").split("-"))
    except (ValueError, AttributeError):
        return None
    return "home" if hg > ag else "away" if ag > hg else "draw"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_key(m: dict) -> str:
    h, d, a = _model_probs(m)
    parts = [
        CARD_VERSION,
        "played" if m.get("played") else "upcoming",
        m.get("score") or "",
        f"{round(h * 100)}-{round(d * 100)}-{round(a * 100)}",
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_card(m: dict) -> plt.Figure:
    home, away = m["home_display"], m["away_display"]
    h, d, a = _model_probs(m)
    played = bool(m.get("played"))
    won = _played_outcome(m.get("score")) if played else None

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    fig.patch.set_facecolor(BG)

    # --- header: teams + meta ---------------------------------------------
    fig.text(0.05, 0.90, f"{home}  v  {away}",
             fontsize=30, fontweight="bold", color=TEXT, ha="left", va="center")

    meta = [m.get("date_human", "")]
    if m.get("group") and m["group"] != "?":
        meta.append(f"Group {m['group']}")
    if m.get("venue_label"):
        meta.append(m["venue_label"])
    fig.text(0.05, 0.815, "  ·  ".join(p for p in meta if p),
             fontsize=13, color=MUTED, ha="left", va="center")

    # --- big final score (played only) ------------------------------------
    if played and m.get("score"):
        fig.text(0.95, 0.885, m["score"],
                 fontsize=48, fontweight="bold", color=TEXT, ha="right", va="center")
        fig.text(0.95, 0.80, "FULL TIME",
                 fontsize=12, color=MUTED, ha="right", va="center")
        fig.text(0.05, 0.685, "Pre-match forecast (model, bias-corrected)",
                 fontsize=12.5, color=MUTED, ha="left", va="center")
    else:
        fig.text(0.05, 0.685, "Model forecast (Dixon-Coles + Elo, bias-corrected)",
                 fontsize=12.5, color=MUTED, ha="left", va="center")

    # --- three W/D/L bars --------------------------------------------------
    ax = fig.add_axes([0.22, 0.16, 0.72, 0.48])
    ax.set_facecolor(BG)
    rows = [
        (f"{home} win", h, COL_HOME, "home"),
        ("Draw",        d, COL_DRAW, "draw"),
        (f"{away} win", a, COL_AWAY, "away"),
    ]
    for y, (label, p, color, slot) in zip((2, 1, 0), rows):
        ax.barh(y, p * 100, height=0.62, color=color, zorder=2)
        emph = (slot == won)
        ax.text(-2, y, label, ha="right", va="center",
                fontsize=15, color=TEXT, fontweight="bold" if emph else "normal")
        suffix = "  ✓" if emph else ""
        ax.text(p * 100 + 1.2, y, f"{round(p * 100)}%{suffix}",
                ha="left", va="center", fontsize=15, color=TEXT,
                fontfamily="monospace", fontweight="bold" if emph else "normal")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 2.6)
    ax.axis("off")

    # --- upcoming: xG line + most-likely score ----------------------------
    if not played:
        sc = m.get("scoreline")
        if sc:
            top = sc["top3"][0] if sc.get("top3") else None
            bits = [f"Expected goals  {home} {sc['lam_h']} – {sc['lam_a']} {away}"]
            if top:
                bits.append(f"Most likely score  {top['score']} ({top['pct']})")
            fig.text(0.05, 0.095, "      ·      ".join(bits),
                     fontsize=13, color=MUTED, ha="left", va="center")

    # --- branding ----------------------------------------------------------
    fig.text(0.95, 0.05, "worldcup.divergencelog.com",
             fontsize=12, color=MUTED, ha="right", va="center")

    return fig


# ---------------------------------------------------------------------------
# Public entry point — SOFT (never raises)
# ---------------------------------------------------------------------------

def render_og_card(m: dict, out_path: Path) -> bool:
    """Render (or cache-skip) the OG card for match context `m` to `out_path`.

    Returns True if the card exists on disk afterwards (freshly rendered or an
    unchanged cache hit) — the caller may then point og:image at it. Returns
    False on ANY failure; the caller falls back to launch.png. Never raises.
    """
    try:
        out_path = Path(out_path)
        key = _cache_key(m)
        sidecar = CACHE_DIR / f"{out_path.stem}.sha1"
        if (out_path.exists() and sidecar.exists()
                and sidecar.read_text(encoding="utf-8").strip() == key):
            return True

        out_path.parent.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fig = _draw_card(m)
        fig.savefig(out_path, dpi=DPI, facecolor=BG)
        plt.close(fig)
        sidecar.write_text(key, encoding="utf-8")
        return True
    except Exception as exc:  # SOFT guardrail — log and fall back, never raise
        print(f"[og] WARNING: card render failed for "
              f"{m.get('key', '?')}: {exc!r} — falling back to launch.png")
        try:
            plt.close("all")
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

def _fake_ctx(played: bool) -> dict:
    seg = lambda h, d, a: [{"cls": "home", "w": h}, {"cls": "draw", "w": d},
                           {"cls": "away", "w": a}]
    base = {
        "key": "smoke",
        "played": played,
        "home_display": "Spain",
        "away_display": "Côte d'Ivoire",
        "group": "H",
        "date_human": "Thursday 18 June",
        "venue_label": "neutral venue",
        "sources": [{"label": "Model", "segments": seg(64.0, 22.0, 14.0)}],
        "scoreline": {"lam_h": "2.10", "lam_a": "0.80",
                      "top3": [{"score": "2–0", "pct": "14%"}]},
    }
    if played:
        base["score"] = "2–1"
    return base


if __name__ == "__main__":
    out = Path("docs/og")
    for kind, played in (("upcoming", False), ("played", True)):
        ok = render_og_card(_fake_ctx(played), out / f"_smoke_{kind}.png")
        print(f"{kind:9s} -> {'OK' if ok else 'FAILED'}  ({out / f'_smoke_{kind}.png'})")
