"""
src/check_calibration.py

Session 12: a quick reliability check on the 290 backtest predictions
saved by src/backtest.py. We don't plot — Session 27 builds the real
reliability diagram for the dashboard. This is a 5-bin text table to
answer one question: do our probabilities mean what they say?

Run from project root:
    python src/check_calibration.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("data/processed/backtest_2024.csv")
N_BINS = 5
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)  # [0, .2, .4, .6, .8, 1]
GAP_FLAG = 0.10
MIN_BIN_COUNT = 15


def reliability_table(probs: np.ndarray, hits: np.ndarray) -> pd.DataFrame:
    """
    probs: predicted probability for some outcome, shape (n,)
    hits:  1 if that outcome happened, else 0, shape (n,)
    """
    rows = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        # final bin is closed on both ends so prob=1.0 is captured
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0,
                         "mean_pred": np.nan, "empirical": np.nan,
                         "gap": np.nan})
            continue
        rows.append({
            "bin":       f"{lo:.1f}-{hi:.1f}",
            "n":         n,
            "mean_pred": float(probs[mask].mean()),
            "empirical": float(hits[mask].mean()),
            "gap":       float(hits[mask].mean() - probs[mask].mean()),
        })
    return pd.DataFrame(rows)


def binary_brier(probs: np.ndarray, hits: np.ndarray) -> float:
    return float(np.mean((probs - hits) ** 2))


def print_table(label: str, df: pd.DataFrame, brier: float) -> None:
    print(f"\n{label}   (binary Brier = {brier:.3f})")
    print(f"  {'bin':<10} {'n':>4}  {'mean_pred':>9}  {'empirical':>9}  {'gap':>7}  flag")
    print(f"  {'-'*10} {'-'*4}  {'-'*9}  {'-'*9}  {'-'*7}  ----")
    for _, r in df.iterrows():
        if r["n"] == 0:
            print(f"  {r['bin']:<10} {0:>4}  {'-':>9}  {'-':>9}  {'-':>7}")
            continue
        flag = ""
        if abs(r["gap"]) > GAP_FLAG and r["n"] >= MIN_BIN_COUNT:
            flag = "  <-- check"
        print(f"  {r['bin']:<10} {int(r['n']):>4}  "
              f"{r['mean_pred']:>9.3f}  "
              f"{r['empirical']:>9.3f}  "
              f"{r['gap']:>+7.3f}{flag}")


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} backtest predictions from {CSV_PATH}")

    actual = df["actual"].values
    outcomes = [
        ("Home win", df["p_home"].values, (actual == 0).astype(int)),
        ("Draw",     df["p_draw"].values, (actual == 1).astype(int)),
        ("Away win", df["p_away"].values, (actual == 2).astype(int)),
    ]

    for label, p, h in outcomes:
        print_table(label, reliability_table(p, h), binary_brier(p, h))

    print("\nReading the table:")
    print("  gap = empirical - mean_pred")
    print("  positive gap = model was underconfident in this bin")
    print("  negative gap = model was overconfident in this bin")
    print(f"  flagged when |gap| > {GAP_FLAG:.2f} and n >= {MIN_BIN_COUNT}")


if __name__ == "__main__":
    main()