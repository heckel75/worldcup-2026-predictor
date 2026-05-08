"""
src/backtest.py

Session 11: fit Dixon-Coles rho on pre-2024-06 matches, then backtest the model
on Euro 2024 + Copa America 2024 (matches the model has never been tuned on).

Run from project root:
    python src/backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

# --- local imports ------------------------------------------------------------
# Adjust if elo_to_lambdas lives somewhere other than src/dixon_coles.py.
sys.path.insert(0, str(Path(__file__).parent))
from elo import EloSystem
from dixon_coles import elo_to_lambdas

# --- configuration ------------------------------------------------------------

MATCHES_PATH = Path("data/processed/matches_clean.csv")
SEEDS_PATH   = Path("data/raw/elo_seeds_2018.csv")
OUT_PATH     = Path("data/processed/backtest_2024.csv")

# Test window. Euro 2024: 14 Jun – 14 Jul. Copa America 2024: 20 Jun – 14 Jul.
TEST_START = pd.Timestamp("2024-06-01")
TEST_END   = pd.Timestamp("2024-07-31")

MAX_GOALS = 10  # truncate scoreline grid; tail mass is negligible


# --- data prep ----------------------------------------------------------------

def load_seeds(path: Path) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df["team"], df["elo"]))


def walk_forward_elo(matches: pd.DataFrame, seeds: dict) -> pd.DataFrame:
    """
    Replay every match in chronological order. For each match, capture
    pre-match Elo for both teams; then update Elo with the actual result.
    Adds two columns: rating_h_pre, rating_a_pre.
    """
    elo = EloSystem(seed_ratings=seeds)
    h_pre = np.empty(len(matches))
    a_pre = np.empty(len(matches))

    for i, row in enumerate(matches.itertuples(index=False)):
        h_pre[i] = elo.get_rating(row.home_team)
        a_pre[i] = elo.get_rating(row.away_team)
        # NOTE: adjust kwargs if your update_match signature is different.
        elo.update_match(
            home_team=row.home_team,
            away_team=row.away_team,
            home_score=row.home_score,
            away_score=row.away_score,
            tournament=row.tournament,
            neutral=bool(row.neutral),
        )

    out = matches.copy()
    out["rating_h_pre"] = h_pre
    out["rating_a_pre"] = a_pre
    return out


def add_lambdas(df: pd.DataFrame) -> pd.DataFrame:
    """Map (rating_h_pre, rating_a_pre, neutral) -> (lam_h, lam_a)."""
    lam_h = np.empty(len(df))
    lam_a = np.empty(len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        lh, la = elo_to_lambdas(
            row.rating_h_pre, row.rating_a_pre, neutral=bool(row.neutral)
        )
        lam_h[i] = lh
        lam_a[i] = la
    out = df.copy()
    out["lam_h"] = lam_h
    out["lam_a"] = lam_a
    return out


# --- Dixon-Coles helpers (kept local; don't depend on predict_from_lambdas) ---

def dc_grid(lam_h: float, lam_a: float, rho: float,
            max_goals: int = MAX_GOALS) -> np.ndarray:
    """Renormalized scoreline probability grid."""
    pmf_h = poisson.pmf(np.arange(max_goals + 1), lam_h)
    pmf_a = poisson.pmf(np.arange(max_goals + 1), lam_a)
    grid = np.outer(pmf_h, pmf_a)
    grid[0, 0] *= 1.0 - lam_h * lam_a * rho
    grid[0, 1] *= 1.0 + lam_h * rho
    grid[1, 0] *= 1.0 + lam_a * rho
    grid[1, 1] *= 1.0 - rho
    grid = np.clip(grid, 1e-12, None)
    grid /= grid.sum()
    return grid


def wdl_from_grid(grid: np.ndarray) -> tuple[float, float, float]:
    home = np.tril(grid, -1).sum()
    draw = np.trace(grid)
    away = np.triu(grid,  1).sum()
    return float(home), float(draw), float(away)


# --- likelihood (vectorized; standard DC convention) --------------------------

def neg_log_likelihood(rho: float,
                       lam_h: np.ndarray, lam_a: np.ndarray,
                       gh: np.ndarray, ga: np.ndarray) -> float:
    p = poisson.pmf(gh, lam_h) * poisson.pmf(ga, lam_a)
    tau = np.ones_like(p)
    m00 = (gh == 0) & (ga == 0)
    m01 = (gh == 0) & (ga == 1)
    m10 = (gh == 1) & (ga == 0)
    m11 = (gh == 1) & (ga == 1)
    tau = np.where(m00, 1.0 - lam_h * lam_a * rho, tau)
    tau = np.where(m01, 1.0 + lam_h * rho,         tau)
    tau = np.where(m10, 1.0 + lam_a * rho,         tau)
    tau = np.where(m11, 1.0 - rho,                 tau)
    p = np.clip(p * tau, 1e-12, None)
    return float(-np.log(p).sum())


# --- backtest -----------------------------------------------------------------

def actual_outcome(gh: int, ga: int) -> int:
    """0 = home win, 1 = draw, 2 = away win."""
    if gh > ga: return 0
    if gh == ga: return 1
    return 2


def backtest(test_df: pd.DataFrame, rho: float) -> dict:
    n = len(test_df)
    probs = np.empty((n, 3))
    actuals = np.empty(n, dtype=int)

    for i, row in enumerate(test_df.itertuples(index=False)):
        grid = dc_grid(row.lam_h, row.lam_a, rho)
        probs[i] = wdl_from_grid(grid)
        actuals[i] = actual_outcome(int(row.home_score), int(row.away_score))

    preds = probs.argmax(axis=1)
    accuracy = float((preds == actuals).mean())

    actual_probs = np.clip(probs[np.arange(n), actuals], 1e-12, None)
    log_loss = float(-np.log(actual_probs).mean())

    onehot = np.zeros_like(probs)
    onehot[np.arange(n), actuals] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    return {"n": n, "accuracy": accuracy, "log_loss": log_loss,
            "brier": brier, "probs": probs, "actuals": actuals}


# --- main ---------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
    matches = matches.sort_values("date").reset_index(drop=True)
    seeds = load_seeds(SEEDS_PATH)

    print(f"Walking forward through {len(matches):,} matches...")
    matches = walk_forward_elo(matches, seeds)
    matches = add_lambdas(matches)

    train = matches[matches["date"] < TEST_START].copy()
    test  = matches[(matches["date"] >= TEST_START) &
                    (matches["date"] <= TEST_END)].copy()
    print(f"  train: {len(train):,}   test: {len(test):,}")

    if len(test) == 0:
        print("No test matches in window — check that matches_clean.csv "
              "covers June-July 2024.")
        return

    # 1. Fit rho on training portion only
    print("\nFitting rho via MLE on training data...")
    fit = minimize_scalar(
        neg_log_likelihood,
        args=(train["lam_h"].values, train["lam_a"].values,
              train["home_score"].values.astype(int),
              train["away_score"].values.astype(int)),
        bounds=(-0.25, 0.10),
        method="bounded",
    )
    rho_hat = float(fit.x)
    print(f"  rho_hat = {rho_hat:+.4f}    (placeholder was -0.1000)")

    # 2. Backtest on test window
    print("\nTest window contents:")
    if "tournament" in test.columns:
        for t, sub in test.groupby("tournament"):
            print(f"  {t:35s} n={len(sub)}")

    res = backtest(test, rho_hat)
    print(f"\n  matches  : {res['n']}")
    print(f"  accuracy : {res['accuracy']:.3f}    (random 0.333; bookies ~0.55)")
    print(f"  log loss : {res['log_loss']:.3f}    (random 1.099; lower is better)")
    print(f"  Brier    : {res['brier']:.3f}    (random 0.667; lower is better)")

    # 3. Sanity baselines
    home_acc = float((res["actuals"] == 0).mean())
    elo_pick = np.where(test["rating_h_pre"].values > test["rating_a_pre"].values, 0, 2)
    elo_acc = float((elo_pick == res["actuals"]).mean())
    print("\nBaselines (sanity):")
    print(f"  always home : {home_acc:.3f}")
    print(f"  higher Elo  : {elo_acc:.3f}")

    # 4. Save per-match predictions for the calibration page later
    out = test[["date", "home_team", "away_team",
                "home_score", "away_score", "tournament"]].copy()
    out["p_home"] = res["probs"][:, 0]
    out["p_draw"] = res["probs"][:, 1]
    out["p_away"] = res["probs"][:, 2]
    out["actual"] = res["actuals"]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved per-match predictions: {OUT_PATH}")
    print(f"\nIf you trust rho_hat={rho_hat:+.4f}, update the default in "
          f"dixon_coles.py from -0.10 to this value.")


if __name__ == "__main__":
    main()