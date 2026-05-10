"""
Dixon-Coles match prediction model.

Session 10: take per-team expected goals (lambdas) from strengths.py
and produce a full scoreline distribution + W/D/L probabilities.

Pipeline:
    elo_home, elo_away
        -> elo_to_lambdas       (Session 9, in strengths.py)
        -> predict_from_lambdas (here)
        -> P(home/draw/away), most likely scoreline, full grid

Method:
    1. Build an (max_goals+1) x (max_goals+1) grid where
         grid[i, j] = P(home=i) * P(away=j)
       under independent Poisson(lam_home), Poisson(lam_away).
    2. Apply the Dixon-Coles tau correction to the four low-score cells
       (0,0), (0,1), (1,0), (1,1) using a single parameter rho.
       rho = -0.1 inflates 0-0 and 1-1 and deflates 1-0 and 0-1, matching
       the empirical pattern in international football.
    3. Renormalize (the corrections + max_goals truncation each shift mass).
    4. Aggregate: P(home win) = mass below diagonal,
                  P(draw)     = diagonal,
                  P(away win) = mass above diagonal.

rho = -0.1 is a placeholder. Session 11 will fit it from data.

Run from project root:
    python src/dixon_coles.py
"""

from math import factorial

import numpy as np
import pandas as pd

from strengths import elo_to_lambdas

""""
The literature typically finds ρ ≈ −0.10 to −0.15, and our fitted −0.027 is on the small side — likely because our training set mixes friendlies with competitive matches. 
Filed as a possible post-v1 refinement (fit ρ separately by match type, or only on competitive matches).
"""
DEFAULT_RHO = -0.027
DEFAULT_MAX_GOALS = 10


def _poisson_pmf_vector(lam: float, max_k: int) -> np.ndarray:
    """P(X=k) for k=0..max_k under Pois(lam), as a 1D numpy array."""
    ks = np.arange(max_k + 1)
    facts = np.array([factorial(int(k)) for k in ks], dtype=float)
    return np.exp(-lam) * (lam ** ks) / facts


def _dixon_coles_grid(
    lam_h: float,
    lam_a: float,
    rho: float,
    max_goals: int,
) -> np.ndarray:
    """
    Full scoreline distribution. grid[i, j] = P(home=i, away=j).
    Already renormalized to sum to 1.
    """
    p_h = _poisson_pmf_vector(lam_h, max_goals)
    p_a = _poisson_pmf_vector(lam_a, max_goals)
    grid = np.outer(p_h, p_a)  # grid[i, j] = p_h[i] * p_a[j]

    # Dixon-Coles tau correction on the four low-score cells.
    grid[0, 0] *= 1 - lam_h * lam_a * rho
    grid[0, 1] *= 1 + lam_h * rho
    grid[1, 0] *= 1 + lam_a * rho
    grid[1, 1] *= 1 - rho

    total = grid.sum()
    if total <= 0:
        # Defensive — shouldn't happen with floored lambdas and sane rho.
        raise ValueError(
            f"Scoreline grid sums to {total} "
            f"(lam_h={lam_h}, lam_a={lam_a}, rho={rho})"
        )
    return grid / total


def predict_from_lambdas(
    lambda_home: float,
    lambda_away: float,
    rho: float = DEFAULT_RHO,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> dict:
    """
    Predict a match given expected goals for each side.

    Returns a dict with:
        p_home_win, p_draw, p_away_win
        lambda_home, lambda_away
        scoreline_grid       (max_goals+1) x (max_goals+1) numpy array
        most_likely_score    (i, j) tuple
        most_likely_score_p  probability of that scoreline
    """
    grid = _dixon_coles_grid(lambda_home, lambda_away, rho, max_goals)

    p_home_win = float(np.tril(grid, k=-1).sum())   # i > j
    p_draw     = float(np.diag(grid).sum())          # i == j
    p_away_win = float(np.triu(grid, k=1).sum())     # i < j

    flat_idx = int(np.argmax(grid))
    i, j = np.unravel_index(flat_idx, grid.shape)

    return {
        "p_home_win":          p_home_win,
        "p_draw":              p_draw,
        "p_away_win":          p_away_win,
        "lambda_home":         float(lambda_home),
        "lambda_away":         float(lambda_away),
        "scoreline_grid":      grid,
        "most_likely_score":   (int(i), int(j)),
        "most_likely_score_p": float(grid[i, j]),
    }


def predict_match(
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    neutral: bool = True,
    rho: float = DEFAULT_RHO,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> dict:
    """
    Predict a match between two named teams given a ratings dict.

    `ratings` maps team_name -> Elo, e.g. loaded from
    data/processed/elo_ratings_2026.csv.
    """
    if home_team not in ratings:
        raise KeyError(f"{home_team!r} not in ratings dict")
    if away_team not in ratings:
        raise KeyError(f"{away_team!r} not in ratings dict")

    lam_h, lam_a = elo_to_lambdas(
        ratings[home_team], ratings[away_team], neutral=neutral
    )
    out = predict_from_lambdas(lam_h, lam_a, rho=rho, max_goals=max_goals)
    out["home_team"] = home_team
    out["away_team"] = away_team
    return out


# ----------------------------------------------------------------------------
# Sanity check (mirrors src/sanity_strengths.py style)
# ----------------------------------------------------------------------------

def _pct(p: float) -> str:
    return f"{p * 100:5.1f}%"


def main():
    df = pd.read_csv("data/processed/elo_ratings_2026.csv")
    ratings = dict(zip(df["team"], df["elo"]))

    cases = [
        ("Spain",   "Argentina",   True),
        ("Brazil",  "Germany",     True),
        ("France",  "England",     True),
        ("USA",     "Mexico",      True),
        ("France",  "Haiti",       True),
        ("England", "New Zealand", True),
    ]

    header = (
        f"{'Matchup':<32} {'P(H)':>6} {'Draw':>6} {'P(A)':>6}  "
        f"{'λ_h':>4} {'λ_a':>4}  {'top':>5} {'p_top':>6}"
    )
    print(header)
    print("-" * len(header))

    for h, a, neutral in cases:
        if h not in ratings or a not in ratings:
            print(f"  (skipping {h} vs {a} — team not in WC ratings file)")
            continue
        r = predict_match(h, a, ratings, neutral=neutral)
        i, j = r["most_likely_score"]
        print(
            f"{h + ' vs ' + a:<32} "
            f"{_pct(r['p_home_win'])} "
            f"{_pct(r['p_draw'])} "
            f"{_pct(r['p_away_win'])}  "
            f"{r['lambda_home']:>4.2f} {r['lambda_away']:>4.2f}  "
            f"{f'{i}-{j}':>5} {_pct(r['most_likely_score_p'])}"
        )

    # ---- Invariants ------------------------------------------------------
    print("\nSanity invariants:")

    # Equal teams, neutral -> P(H) == P(A) by symmetry
    r = predict_from_lambdas(1.5, 1.5)
    print(
        f"  equal lambdas (1.5 each):    "
        f"P(H)={_pct(r['p_home_win'])}  "
        f"P(D)={_pct(r['p_draw'])}  "
        f"P(A)={_pct(r['p_away_win'])}   "
        f"sum={r['p_home_win'] + r['p_draw'] + r['p_away_win']:.4f}"
    )

    # Heavy mismatch -> draw rare, home win dominant
    r = predict_from_lambdas(3.5, 0.4)
    print(
        f"  mismatch (3.5 vs 0.4):       "
        f"P(H)={_pct(r['p_home_win'])}  "
        f"P(D)={_pct(r['p_draw'])}  "
        f"P(A)={_pct(r['p_away_win'])}   "
        f"sum={r['p_home_win'] + r['p_draw'] + r['p_away_win']:.4f}"
    )

    # rho effect: DC should bump draw rate vs plain Poisson
    r0  = predict_from_lambdas(1.4, 1.2, rho=0.0)
    rdc = predict_from_lambdas(1.4, 1.2, rho=-0.1)
    print(
        f"  rho effect (1.4 vs 1.2):     "
        f"plain Poisson draw={_pct(r0['p_draw'])}, "
        f"DC draw={_pct(rdc['p_draw'])}  "
        f"(DC should be slightly higher)"
    )


if __name__ == "__main__":
    main()