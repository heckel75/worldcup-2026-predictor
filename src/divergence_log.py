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
  * stage                      (date-gated named round — Group / Round of 32 /
                                … / Final — see derive_stage, DIVLOG-2)
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

import bracket_resolve
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
# Stage — date-gated named rounds (DIVLOG-2).
# ----------------------------------------------------------------------
# The group stage and the knockout stage occupy DISJOINT date windows (group
# fixtures Jun 11-27, KO from Jun 28 — §6), so the match DATE, not the team-pair,
# is the robust group-vs-KO discriminator: a same-group KO rematch (two teams who
# shared a 2026 group meeting again in the bracket) would fool a pair-based test.
# A group row's letter still comes from the team->group map; a KO row's round is
# resolved by matching its unordered pair into the populated bracket and mapping
# the match_id to a round name via the published id ranges. If a KO pair can't be
# found in the bracket (shouldn't happen for a played row) we fall back to a
# coarse "Knockout" rather than crash. Everything bracket-related is read through
# bracket_resolve (allowed, read-only) — this module imports no model/sim code.
_LAST_GROUP_DATE = "2026-06-27"   # last group-stage match day (ISO, inclusive)


def _ko_round_name(match_id: int) -> str:
    """Published FIFA 2026 match-id ranges -> round name (38-RECON)."""
    if 73 <= match_id <= 88:
        return "Round of 32"
    if 89 <= match_id <= 96:
        return "Round of 16"
    if 97 <= match_id <= 100:
        return "Quarter-final"
    if 101 <= match_id <= 102:
        return "Semi-final"
    if match_id == 103:
        return "Third place"
    if match_id == 104:
        return "Final"
    return "Knockout"


def build_ko_round_map() -> dict[frozenset, str]:
    """Unordered team-pair -> KO round name, from the populated bracket.

    Reads played results through bracket_resolve.resolve_bracket(); every
    determined node (played or not) contributes its pairing, so a played ledger
    KO row always resolves. Empty until the group stage completes (resolve_bracket
    holds its placeholder pre-completion), which is correct — pre-KO there are no
    KO rows to classify anyway."""
    bk = bracket_resolve.resolve_bracket()
    nodes = [m for rnd in bk.get("rounds", []) for m in rnd["matches"]]
    tp = bk.get("third_place")
    if tp:
        nodes.append(tp)
    out: dict[frozenset, str] = {}
    for m in nodes:
        a, b = m["team_a"], m["team_b"]
        if a and b:
            out[frozenset((a, b))] = _ko_round_name(m["match_id"])
    return out


def derive_stage(home: str, away: str, date: str,
                 ko_round_map: dict[frozenset, str]) -> tuple[str, str]:
    """Return (stage_label, group_letter). group_letter is '' for KO rounds.

    Date-gated (see the block comment): a group-window date -> "Group" + the home
    team's group letter; a later date -> the named KO round from ko_round_map, or
    coarse "Knockout" if the pair can't be resolved."""
    if str(date) <= _LAST_GROUP_DATE:
        return "Group", (bracket_resolve.TEAM_TO_GROUP.get(home) or "")
    return ko_round_map.get(frozenset((home, away)), "Knockout"), ""


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
# Performance by divergence size (DIVLOG-2 Part B) — the site's thesis made
# measurable: does a larger model-vs-market gap track better or worse
# calibration? Partition played matches by the SAME gap metric the archive/flag
# use (recompute_gap) into GAP_BUCKETS. A row with no book has a NaN gap and lands
# in no bucket (a model-vs-book gap is undefined without a book) — consistent with
# the archive, so the buckets pool exactly the priced matches the scoreboard scores.
# ----------------------------------------------------------------------
def gap_size_buckets(df: pd.DataFrame) -> list[dict]:
    """One summary row per gap bucket (coarsest-gap questions first).

    Per bucket: n (played matches whose gap falls in the bucket), model
    distinct-verdict-win count (honouring the no-credit contract — only a "model"
    verdict counts, exactly as the scoreboard tallies), and per-source Brier over
    the bucket's matches (books/poly drop their NaN rows via _source_scores, so
    each source reports its own n)."""
    records = df.to_dict("records")
    gaps = [recompute_gap(r) for r in records]
    buckets: list[dict] = []
    for key, label, lo, hi in GAP_BUCKETS:
        idx = [i for i, g in enumerate(gaps) if pd.notna(g) and lo <= g < hi]
        sub = df.iloc[idx]
        model_wins = 0
        for i in idx:
            v = _verdict(records[i])
            if v is not None and v["winner"] == "model":
                model_wins += 1
        buckets.append({
            "key": key,
            "label": label,
            "n": len(idx),
            "model_wins": model_wins,
            "model": _source_scores(sub, ""),
            "book": _source_scores(sub, "_book"),
            "poly": _source_scores(sub, "_poly"),
        })
    return buckets


# ----------------------------------------------------------------------
# Rolling calibration by source (DIVLOG-2 Part C) — cumulative per-source Brier
# over played matches in tournament order (date, then match_key for a stable
# within-day order; the ledger carries no match_id). Cumulative (not windowed) is
# cleaner at this n and needs no window choice. A market's running Brier only
# advances on matches it priced (NaN-market rows skipped); the model advances on
# every match. All three series share the match-progression x-index so the lines
# are comparable in time. Reuses calibration.brier_multiclass — the SAME primitive
# the scoreboard and buckets use, applied to the growing prefix each step.
# ----------------------------------------------------------------------
def rolling_calibration(df: pd.DataFrame) -> dict:
    """Return {"n": total_played, "series": [{name, suffix, points:[...]}]}.

    Each point is {"i", "date", "label", "brier", "n"} where i is the shared
    match index in tournament order and brier is that source's cumulative Brier
    up to and including match i (over the matches it has priced so far)."""
    ordered = df.sort_values(["date", "match_key"]).reset_index(drop=True)
    n = int(len(ordered))
    series: list[dict] = []
    for name, suffix in (("Model", ""), ("Sportsbook", "_book"),
                         ("Polymarket", "_poly")):
        cols = [f"p_home{suffix}", f"p_draw{suffix}", f"p_away{suffix}"]
        scored: list[dict] = []   # canonical-schema rows accumulated in order
        points: list[dict] = []
        for i, rec in ordered.iterrows():
            vals = [rec[c] for c in cols]
            if any(pd.isna(v) for v in vals):
                continue          # source didn't price this match
            scored.append({
                "p_home": float(vals[0]), "p_draw": float(vals[1]),
                "p_away": float(vals[2]), "outcome": rec["outcome"],
            })
            brier = calibration.brier_multiclass(pd.DataFrame(scored))
            points.append({
                "i": int(i),
                "date": rec["date"],
                "label": f"{rec['home_team']} v {rec['away_team']}",
                "brier": round(float(brier), 4),
                "n": len(scored),
            })
        series.append({"name": name, "suffix": suffix, "points": points})
    return {"n": n, "series": series}


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
def _empty() -> dict:
    return {"rows": [], "scoreboard": None, "n_played": 0,
            "buckets": [], "rolling": None}


def build(preds_path: str | Path, ko_round_map: dict | None = None) -> dict:
    """Read the frozen ledger; return archive rows + attribution scoreboard +
    gap-size buckets + rolling calibration series.

    Returns internal team names (generate_site applies DISPLAY_NAMES + links —
    keeps this module out of the site layer). Empty/absent ledger -> empty
    structures, never raises. ko_round_map is the unordered-pair -> round-name
    map (build_ko_round_map() by default); injectable so the self-test stays
    isolated from live matches_clean.csv."""
    path = Path(preds_path)
    if not path.exists():
        return _empty()

    df = pd.read_csv(path)
    if "outcome" not in df.columns:
        return _empty()
    df = df[df["outcome"].isin(["H", "D", "A"])].copy()
    if df.empty:
        return _empty()

    if ko_round_map is None:
        ko_round_map = build_ko_round_map()

    rows: list[dict] = []
    counts = {b: 0 for b in VERDICT_BUCKETS}
    n_verdicts = 0

    for rec in df.to_dict("records"):
        home, away = rec["home_team"], rec["away_team"]
        stage, group = derive_stage(home, away, rec["date"], ko_round_map)
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

    return {
        "rows": rows,
        "scoreboard": scoreboard,
        "n_played": int(len(df)),
        "buckets": gap_size_buckets(df),
        "rolling": rolling_calibration(df),
    }


# ----------------------------------------------------------------------
# Self-test (run: python src/divergence_log.py --test)
# ----------------------------------------------------------------------
def _test() -> None:
    import tempfile
    import verdict

    # Stage: date-gated named rounds (DIVLOG-2). A group-window date -> "Group" +
    # the home team's group letter; a KO-window date -> the round from the
    # pair-map; an unknown KO pair -> coarse "Knockout" (never crash).
    gmap = {frozenset(("France", "Morocco")): "Round of 16"}
    assert derive_stage("Mexico", "South Africa", "2026-06-11", gmap) == ("Group", "A"), \
        derive_stage("Mexico", "South Africa", "2026-06-11", gmap)
    assert derive_stage("France", "Morocco", "2026-07-04", gmap) == ("Round of 16", ""), \
        derive_stage("France", "Morocco", "2026-07-04", gmap)
    assert derive_stage("France", "Morocco", "2026-07-04", {}) == ("Knockout", "")
    # match-id ranges -> round names (38-RECON id map).
    assert _ko_round_name(73) == "Round of 32" and _ko_round_name(88) == "Round of 32"
    assert _ko_round_name(89) == "Round of 16" and _ko_round_name(96) == "Round of 16"
    assert _ko_round_name(97) == "Quarter-final" and _ko_round_name(100) == "Quarter-final"
    assert _ko_round_name(101) == "Semi-final" and _ko_round_name(102) == "Semi-final"
    assert _ko_round_name(103) == "Third place"
    assert _ko_round_name(104) == "Final"

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

    # End-to-end build on a tiny synthetic ledger: one model-distinct win (ge15
    # gap), one all-missed upset (5to15 gap), one no-market row (verdict None,
    # NaN gap, no bucket), one KO-date row (lt5 gap) exercising the round-map.
    cols = ["match_key", "date", "home_team", "away_team",
            "p_home", "p_draw", "p_away",
            "p_home_book", "p_draw_book", "p_away_book",
            "p_home_poly", "p_draw_poly", "p_away_poly",
            "outcome", "actual_home_score", "actual_away_score"]
    nan = float("nan")
    data = [
        # model clearly closest on a home win — gap 0.15 -> ge15 bucket
        ["d1", "2026-06-11", "Mexico", "South Africa",
         0.55, 0.25, 0.20, 0.40, 0.35, 0.25, 0.41, 0.34, 0.25, "H", 2, 0],
        # everyone favoured away/draw, home won -> all_missed — gap 0.05 -> 5to15
        ["d2", "2026-06-12", "South Korea", "Czech Republic",
         0.20, 0.30, 0.50, 0.25, 0.30, 0.45, 0.25, 0.30, 0.45, "H", 1, 0],
        # no market frozen -> verdict None, gap NaN, no bucket
        ["d3", "2026-06-13", "Brazil", "Morocco",
         0.70, 0.20, 0.10, nan, nan, nan, nan, nan, nan, "H", 3, 0],
        # KO-date row -> round from the injected map — gap 0.02 -> lt5 bucket
        ["d4", "2026-07-04", "Argentina", "Brazil",
         0.50, 0.25, 0.25, 0.48, 0.27, 0.25, 0.49, 0.26, 0.25, "H", 1, 0],
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     newline="", encoding="utf-8") as f:
        pd.DataFrame(data, columns=cols).to_csv(f, index=False)
        tmp = f.name
    # Inject the round-map so the test never touches live matches_clean.csv.
    ko = {frozenset(("Argentina", "Brazil")): "Final"}
    out = build(tmp, ko_round_map=ko)
    Path(tmp).unlink()

    assert out["n_played"] == 4, out["n_played"]
    assert out["scoreboard"]["n_verdicts"] == 3, out["scoreboard"]["n_verdicts"]
    assert out["scoreboard"]["counts"]["model"] == 1
    assert out["scoreboard"]["counts"]["all_missed"] == 1
    # rows newest-first
    assert [r["match_key"] for r in out["rows"]] == ["d4", "d3", "d2", "d1"]
    d4 = next(r for r in out["rows"] if r["match_key"] == "d4")
    assert d4["stage"] == "Final" and d4["group"] == "", d4["stage"]
    d1 = next(r for r in out["rows"] if r["match_key"] == "d1")
    assert d1["stage"] == "Group" and d1["group"] == "A", d1["stage"]
    d3 = next(r for r in out["rows"] if r["match_key"] == "d3")
    assert pd.isna(d3["gap"]) and d3["gap_bucket"] is None
    assert d3["verdict_winner"] is None
    assert d1["model_won"] is True and d1["verdict_winner"] == "model"
    # model scored on all 4; markets on the 3 with frozen probs
    model_src = out["scoreboard"]["sources"][0]
    poly_src = out["scoreboard"]["sources"][2]
    assert model_src["n"] == 4 and poly_src["n"] == 3

    # ---- Gap-size buckets (Part B) ----
    buckets = out["buckets"]
    assert [b["key"] for b in buckets] == ["lt5", "5to15", "ge15"], buckets
    by_key = {b["key"]: b for b in buckets}
    assert by_key["lt5"]["n"] == 1 and by_key["5to15"]["n"] == 1 and by_key["ge15"]["n"] == 1
    # the 3 book-carrying matches are partitioned across the buckets (d3 dropped)
    assert sum(b["n"] for b in buckets) == 3
    assert by_key["ge15"]["model_wins"] == 1        # d1's distinct model win
    # Reconciliation: book Brier pooled over buckets == scoreboard book Brier
    # (both over exactly the 3 book-carrying rows). Model/poly differ from the
    # scoreboard only because d3 has no book, so it's in the scoreboard's model
    # scoring set but in no bucket — the legitimate book-less case.
    book_sb = out["scoreboard"]["sources"][1]["brier"]
    num = sum(b["book"]["brier"] * b["book"]["n"] for b in buckets if b["book"]["n"])
    den = sum(b["book"]["n"] for b in buckets)
    assert abs(num / den - book_sb) < 1e-9, (num / den, book_sb)

    # ---- Rolling calibration (Part C) ----
    roll = out["rolling"]
    assert roll["n"] == 4, roll["n"]
    rs = {s["name"]: s for s in roll["series"]}
    assert len(rs["Model"]["points"]) == 4          # priced every match
    assert len(rs["Sportsbook"]["points"]) == 3     # d3 unpriced
    assert len(rs["Polymarket"]["points"]) == 3
    # points ordered by the shared match index; model covers 0..3
    assert [p["i"] for p in rs["Model"]["points"]] == [0, 1, 2, 3]
    assert [p["i"] for p in rs["Sportsbook"]["points"]] == [0, 1, 3]  # skips d3 at i=2
    # the final cumulative point == the scoreboard's overall Brier for that source
    assert abs(rs["Model"]["points"][-1]["brier"] - model_src["brier"]) < 1e-3
    assert abs(rs["Sportsbook"]["points"][-1]["brier"] - book_sb) < 1e-3

    # Also exercise the shared verdict logic from here.
    verdict._test_verdict()
    print("divergence_log.py self-tests passed")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _test()
