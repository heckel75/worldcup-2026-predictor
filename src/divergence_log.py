"""Divergence Log archive + attribution scoreboard (Session DIVLOG-1, Phase 1).

Pure data-consumer (Session 23 convention): reads ONLY the frozen ledger
(wc_predictions.csv) and bracket's read-only group structure. Imports nothing
from the model / sim / triple_compare / ledger-writer / update pipeline. Every
metric is recomputed at build time from frozen probs, so a de-flagged stale
commentary file can never resurrect into the archive — we never walk
data/processed/divergences/ at all here.

What it produces (consumed by generate_site.py):
  build(preds_path) -> {"rows": [...], "scoreboard": {...}, "n_played": int}

Each row recomputes, on the frozen corrected-model + market triples:
  * gap  = div_model_book_max  (triple_compare's definition, §5)
  * divergence_type            (triple_compare.classify_divergence, replicated)
  * flag_divergent             (the full verified predicate, §5)
  * stage                      (derived via bracket: same-group -> Group stage)
  * the three sources' frozen prob ON THE ACTUAL OUTCOME
  * the _verdict (imported from verdict.py — single source of truth)

The attribution scoreboard honours the verdict no-credit contract: "markets",
"wash" and "all_missed" credit no single source's win count (exactly as the
index scoreboard does). Per-source Brier + log loss are computed over every
match where that source froze all three probs, via calibration's primitives.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import bracket
import calibration
from verdict import VERDICT_BUCKETS, _verdict

# Mirror triple_compare exactly — do NOT re-tune here. (§5 / triple_compare.py)
DIV_FLAG_THRESHOLD = 0.15

_OUTCOME_SLOT = {"H": "home", "D": "draw", "A": "away"}
_RESULT_LABEL = {"H": "Home win", "D": "Draw", "A": "Away win"}

# Gap buckets (pp): the proposal's <5 / 5–15 / >15. The top edge is keyed to the
# flag threshold (>=15pp) so "15pp+" and flagged coincide. Half-open [lo, hi).
GAP_BUCKETS = [
    ("lt5",   "Under 5pp", 0.0,  0.05),
    ("5to15", "5–15pp",    0.05, DIV_FLAG_THRESHOLD),
    ("ge15",  "15pp+",     DIV_FLAG_THRESHOLD, 1.01),
]


# ----------------------------------------------------------------------
# Stage — derived via bracket's read-only group structure.
# ----------------------------------------------------------------------
# bracket maps each team to a group letter but carries NO date or pair->round
# map for the knockout stage (KO pairings are dynamic and resolved by results).
# So "via bracket" cleanly distinguishes the group stage (an intra-group pair)
# from everything else; specific KO rounds (R32/R16/...) are deferred to
# DIVLOG-2/3 once KO fixtures (with dates) exist in the feed. Today every played
# row is a group match, so this is exact; it degrades to a coarse "Knockout"
# bucket rather than guessing a round from a hardcoded (possibly wrong) date.
def derive_stage(home: str, away: str) -> tuple[str, str]:
    """Return (stage_label, group_letter). group_letter is '' for KO."""
    gh = bracket.TEAM_TO_GROUP.get(home)
    ga = bracket.TEAM_TO_GROUP.get(away)
    if gh and ga and gh == ga:
        return "Group stage", gh
    return "Knockout", ""


# ----------------------------------------------------------------------
# Divergence metrics — replicated from triple_compare on FROZEN probs.
# In the ledger the corrected model probs are p_home/p_draw/p_away; the book
# probs are p_*_book. classify_divergence/gap/flag must match triple_compare's
# definitions byte-for-byte so the archive never invents a metric.
# ----------------------------------------------------------------------
def recompute_gap(row: dict) -> float:
    """div_model_book_max: max single-outcome |model_corr - book|. NaN if no book."""
    if pd.isna(row.get("p_home_book")):
        return float("nan")
    diffs = [abs(row[f"p_{s}"] - row[f"p_{s}_book"]) for s in ("home", "draw", "away")]
    return round(max(diffs), 4)


def recompute_divergence_type(row: dict) -> str:
    """triple_compare.classify_divergence, on frozen corrected-model + book."""
    if pd.isna(row.get("p_home_book")):
        return ""
    model = (row["p_home"], row["p_draw"], row["p_away"])
    book = (row["p_home_book"], row["p_draw_book"], row["p_away_book"])
    m_arg = max(range(3), key=lambda i: model[i])
    b_arg = max(range(3), key=lambda i: book[i])
    if m_arg != b_arg:
        return "disagree_on_favorite"
    if model[b_arg] > book[b_arg]:
        return "model_over_concentrated"
    return "model_under_concentrated"


def flagged_now(row: dict) -> bool:
    """The full verified flag predicate (§5): max gap >= threshold AND book present.

    The canonical de-flag gate. Any consumer that would otherwise resurrect a
    stale commentary file must gate on THIS, not on a file's mere presence.
    """
    gap = recompute_gap(row)
    return bool(pd.notna(gap) and gap >= DIV_FLAG_THRESHOLD)


def gap_bucket(gap: float) -> str | None:
    if pd.isna(gap):
        return None
    for key, _label, lo, hi in GAP_BUCKETS:
        if lo <= gap < hi:
            return key
    return None


def _prob_on_outcome(row: dict, suffix: str) -> float:
    """Frozen prob the {suffix} source put on the ACTUAL outcome ('' = model)."""
    slot = _OUTCOME_SLOT[row["outcome"]]
    return row.get(f"p_{slot}{suffix}")


# ----------------------------------------------------------------------
# Per-source Brier + log loss — path (a): rename a source's frozen triple into
# calibration's expected p_home/p_draw/p_away frame and reuse its primitives,
# so the markets get scored by the SAME code that scores the model. A source is
# scored only on matches where it froze all three probs (NaN rows dropped).
# ----------------------------------------------------------------------
def _source_scores(df: pd.DataFrame, suffix: str) -> dict:
    cols = [f"p_home{suffix}", f"p_draw{suffix}", f"p_away{suffix}"]
    sub = df[cols + ["outcome"]].copy()
    sub.columns = ["p_home", "p_draw", "p_away", "outcome"]
    sub = sub.dropna(subset=["p_home", "p_draw", "p_away"])
    n = int(len(sub))
    if n == 0:
        return {"brier": None, "log_loss": None, "n": 0}
    return {
        "brier": calibration.brier_multiclass(sub),
        "log_loss": calibration.log_loss(sub),
        "n": n,
    }


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
def build(preds_path: str | Path) -> dict:
    """Read the frozen ledger; return archive rows + attribution scoreboard.

    Returns internal team names (generate_site applies DISPLAY_NAMES + links —
    keeps this module out of the site layer). Empty/absent ledger -> empty
    structures, never raises."""
    path = Path(preds_path)
    if not path.exists():
        return {"rows": [], "scoreboard": None, "n_played": 0}

    df = pd.read_csv(path)
    if "outcome" not in df.columns:
        return {"rows": [], "scoreboard": None, "n_played": 0}
    df = df[df["outcome"].isin(["H", "D", "A"])].copy()
    if df.empty:
        return {"rows": [], "scoreboard": None, "n_played": 0}

    rows: list[dict] = []
    counts = {b: 0 for b in VERDICT_BUCKETS}
    n_verdicts = 0

    for rec in df.to_dict("records"):
        home, away = rec["home_team"], rec["away_team"]
        stage, group = derive_stage(home, away)
        gap = recompute_gap(rec)
        v = _verdict(rec)
        if v is not None:
            counts[v["winner"]] += 1
            n_verdicts += 1

        hg, ag = rec.get("actual_home_score"), rec.get("actual_away_score")
        score = (f"{int(hg)}–{int(ag)}"
                 if pd.notna(hg) and pd.notna(ag) else None)

        rows.append({
            "match_key":       rec["match_key"],
            "date":            rec["date"],
            "home_team":       home,
            "away_team":       away,
            "stage":           stage,
            "group":           group,
            "outcome":         rec["outcome"],
            "result_label":    _RESULT_LABEL[rec["outcome"]],
            "score":           score,
            "p_model":         _prob_on_outcome(rec, ""),
            "p_book":          _prob_on_outcome(rec, "_book"),
            "p_poly":          _prob_on_outcome(rec, "_poly"),
            "gap":             gap,
            "gap_bucket":      gap_bucket(gap),
            "divergence_type": recompute_divergence_type(rec),
            "flagged":         flagged_now(rec),
            "verdict_winner":  v["winner"] if v else None,
            "verdict_text":    v["text"] if v else None,
            "model_won":       bool(v and v["winner"] == "model"),
        })

    # newest first for the archive
    rows.sort(key=lambda r: r["date"], reverse=True)

    scoreboard = {
        "n_verdicts": n_verdicts,
        "counts": counts,
        "sources": [
            {"name": "Model", "wins": counts["model"],
             "win_rate": (counts["model"] / n_verdicts) if n_verdicts else None,
             **_source_scores(df, "")},
            {"name": "Sportsbook", "wins": counts["books"],
             "win_rate": (counts["books"] / n_verdicts) if n_verdicts else None,
             **_source_scores(df, "_book")},
            {"name": "Polymarket", "wins": counts["Polymarket"],
             "win_rate": (counts["Polymarket"] / n_verdicts) if n_verdicts else None,
             **_source_scores(df, "_poly")},
        ],
    } if n_verdicts else None

    return {"rows": rows, "scoreboard": scoreboard, "n_played": int(len(df))}


# ----------------------------------------------------------------------
# Self-test (run: python src/divergence_log.py --test)
# ----------------------------------------------------------------------
def _test() -> None:
    import tempfile
    import verdict

    # Stage derivation: intra-group vs cross-group.
    assert derive_stage("Mexico", "South Africa") == ("Group stage", "A"), \
        derive_stage("Mexico", "South Africa")
    assert derive_stage("Mexico", "Brazil")[0] == "Knockout", \
        derive_stage("Mexico", "Brazil")

    # gap / divergence_type / flag replicate triple_compare on frozen probs.
    row = {"p_home": 0.60, "p_draw": 0.25, "p_away": 0.15,
           "p_home_book": 0.40, "p_draw_book": 0.30, "p_away_book": 0.30}
    assert recompute_gap(row) == 0.20, recompute_gap(row)            # max |0.20|
    assert recompute_divergence_type(row) == "model_over_concentrated"
    assert flagged_now(row) is True

    # No book -> NaN gap, '' type, unflagged.
    nob = {"p_home": 0.6, "p_draw": 0.25, "p_away": 0.15,
           "p_home_book": float("nan")}
    assert pd.isna(recompute_gap(nob))
    assert recompute_divergence_type(nob) == ""
    assert flagged_now(nob) is False

    # disagree_on_favorite: model favours home, book favours away.
    dis = {"p_home": 0.45, "p_draw": 0.25, "p_away": 0.30,
           "p_home_book": 0.30, "p_draw_book": 0.25, "p_away_book": 0.45}
    assert recompute_divergence_type(dis) == "disagree_on_favorite"

    # Gap buckets, including the flag-coincident top edge.
    assert gap_bucket(0.03) == "lt5"
    assert gap_bucket(0.05) == "5to15"
    assert gap_bucket(0.149) == "5to15"
    assert gap_bucket(0.15) == "ge15"
    assert gap_bucket(float("nan")) is None

    # End-to-end build on a tiny synthetic ledger: one model-distinct win, one
    # all-missed upset, one no-market row (verdict None -> not counted).
    cols = ["match_key", "date", "home_team", "away_team",
            "p_home", "p_draw", "p_away",
            "p_home_book", "p_draw_book", "p_away_book",
            "p_home_poly", "p_draw_poly", "p_away_poly",
            "outcome", "actual_home_score", "actual_away_score"]
    nan = float("nan")
    data = [
        # model clearly closest on a home win
        ["d1", "2026-06-11", "Mexico", "South Africa",
         0.55, 0.25, 0.20, 0.40, 0.35, 0.25, 0.41, 0.34, 0.25, "H", 2, 0],
        # everyone favoured away/draw, home won -> all_missed
        ["d2", "2026-06-12", "South Korea", "Czech Republic",
         0.20, 0.30, 0.50, 0.25, 0.30, 0.45, 0.25, 0.30, 0.45, "H", 1, 0],
        # no market frozen -> verdict None, gap NaN
        ["d3", "2026-06-13", "Brazil", "Morocco",
         0.70, 0.20, 0.10, nan, nan, nan, nan, nan, nan, "H", 3, 0],
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     newline="", encoding="utf-8") as f:
        pd.DataFrame(data, columns=cols).to_csv(f, index=False)
        tmp = f.name
    out = build(tmp)
    Path(tmp).unlink()

    assert out["n_played"] == 3, out["n_played"]
    assert out["scoreboard"]["n_verdicts"] == 2, out["scoreboard"]["n_verdicts"]
    assert out["scoreboard"]["counts"]["model"] == 1
    assert out["scoreboard"]["counts"]["all_missed"] == 1
    # rows newest-first
    assert [r["match_key"] for r in out["rows"]] == ["d3", "d2", "d1"]
    d3 = next(r for r in out["rows"] if r["match_key"] == "d3")
    assert pd.isna(d3["gap"]) and d3["gap_bucket"] is None
    assert d3["verdict_winner"] is None
    d1 = next(r for r in out["rows"] if r["match_key"] == "d1")
    assert d1["model_won"] is True and d1["verdict_winner"] == "model"
    # model scored on all 3; markets on the 2 with frozen probs
    model_src = out["scoreboard"]["sources"][0]
    poly_src = out["scoreboard"]["sources"][2]
    assert model_src["n"] == 3 and poly_src["n"] == 2

    # Also exercise the shared verdict logic from here.
    verdict._test_verdict()
    print("divergence_log.py self-tests passed")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _test()
