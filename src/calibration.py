"""
src/calibration.py — calibration tracker stats (Session 27).

Takes predictions + actual outcomes and reports how well the forecast's
probabilities matched reality. No model imports; reads CSVs only.
Imported by generate_site.py the same way whats_changed.py is.

Canonical schema (every consumer is normalised to this):
    home_team, away_team, p_home, p_draw, p_away, outcome   # outcome in {H,D,A}

backtest_2024.csv uses an 'actual' column (0=H, 1=D, 2=A); this module
converts it to the H/D/A string convention.
"""
from __future__ import annotations

import pandas as pd

CANON = ["home_team", "away_team", "p_home", "p_draw", "p_away", "outcome"]

# Exact tournament strings from backtest_2024.csv (verified: 51 + 32 = 83 rows).
MAJORS = {"UEFA Euro", "Copa América"}

# backtest_2024.csv 'actual' encoding: 0=home win, 1=draw, 2=away win.
_ACTUAL_MAP = {0: "H", 1: "D", 2: "A"}


def load_backtest(path="data/processed/backtest_2024.csv", majors_only=False):
    df = pd.read_csv(path)
    out = df[["home_team", "away_team", "p_home", "p_draw", "p_away"]].copy()
    out["outcome"] = df["actual"].map(_ACTUAL_MAP)
    if majors_only:
        out = out[df["tournament"].isin(MAJORS)].reset_index(drop=True)
    return out[CANON]


def load_wc_predictions(path="data/processed/wc_predictions.csv"):
    """Load live WC predictions. Returns None if file is header-only."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None
    filled = df.dropna(subset=["outcome"])
    if len(filled) == 0:
        return None
    return filled[CANON].reset_index(drop=True)


def _pooled_pairs(df):
    rows = []
    for _, m in df.iterrows():
        rows += [
            (m.p_home, int(m.outcome == "H")),
            (m.p_draw, int(m.outcome == "D")),
            (m.p_away, int(m.outcome == "A")),
        ]
    return pd.DataFrame(rows, columns=["p", "hit"])


def reliability_pooled(df, n_bins=5):
    pairs = _pooled_pairs(df)
    edges = [i / n_bins for i in range(n_bins + 1)]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi == 1.0
        sel = (pairs.p >= lo) & ((pairs.p <= hi) if last else (pairs.p < hi))
        b = pairs[sel]
        out.append({
            "bin_lo":    lo,
            "bin_hi":    hi,
            "n":         int(len(b)),
            "mean_pred": float(b.p.mean()) if len(b) else None,
            "obs_freq":  float(b.hit.mean()) if len(b) else None,
        })
    return pd.DataFrame(out)


def reliability_per_outcome(df):
    t = pd.DataFrame([
        {"outcome": "Home win", "pred": df.p_home.mean(), "obs": (df.outcome == "H").mean()},
        {"outcome": "Draw",     "pred": df.p_draw.mean(), "obs": (df.outcome == "D").mean()},
        {"outcome": "Away win", "pred": df.p_away.mean(), "obs": (df.outcome == "A").mean()},
    ])
    t["gap"] = t["obs"] - t["pred"]
    return t


def brier_multiclass(df):
    s = 0.0
    for _, m in df.iterrows():
        s += ((m.p_home - (m.outcome == "H")) ** 2
              + (m.p_draw - (m.outcome == "D")) ** 2
              + (m.p_away - (m.outcome == "A")) ** 2)
    return s / len(df)


def accuracy(df):
    pred = df[["p_home", "p_draw", "p_away"]].values.argmax(axis=1)
    actual = df.outcome.map({"H": 0, "D": 1, "A": 2}).values
    return float((pred == actual).mean())


def summarize(df, label):
    return {
        "label":       label,
        "n":           int(len(df)),
        "brier":       brier_multiclass(df),
        "accuracy":    accuracy(df),
        "bins":        reliability_pooled(df).to_dict("records"),
        "per_outcome": reliability_per_outcome(df).to_dict("records"),
    }


if __name__ == "__main__":
    # 1. deterministic Brier check
    tiny = pd.DataFrame([{
        "home_team": "A", "away_team": "B",
        "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2, "outcome": "H",
    }])
    expected = (0.5 - 1) ** 2 + 0.3 ** 2 + 0.2 ** 2
    assert abs(brier_multiclass(tiny) - expected) < 1e-9, "brier formula wrong"

    # 2. regression anchor: reproduce Session 11's majors numbers
    maj = load_backtest(majors_only=True)
    b, a = brier_multiclass(maj), accuracy(maj)
    print(f"majors n={len(maj)}  brier={b:.3f}  acc={a:.3f}")
    assert len(maj) == 83, f"expected 83 majors, got {len(maj)}"
    assert abs(b - 0.583) < 0.02, f"brier {b:.3f} not within 0.02 of 0.583"
    assert abs(a - 0.530) < 0.02, f"acc {a:.3f} not within 0.02 of 0.530"

    # 3. pooled bins account for every prediction
    full = load_backtest()
    assert reliability_pooled(full).n.sum() == 3 * len(full)

    print("calibration.py self-tests passed")
