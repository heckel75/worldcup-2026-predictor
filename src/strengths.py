"""
Convert Elo ratings to per-team expected goals (lambdas).

Constants empirically fit from data/processed/match_elo_log.csv in Session 9.
See also data/processed/goals_vs_elo.png.

Method:
    eff_diff  = elo_h - elo_a + (60 if non-neutral else 0)
    lam_diff  = DIFF_PER_ELO * eff_diff
    lam_total = BASE_TOTAL_GOALS + eff_diff^2 / TOTAL_QUADRATIC_DENOM
    lam_h     = (lam_total + lam_diff) / 2
    lam_a     = (lam_total - lam_diff) / 2

Both lambdas are floored at LAMBDA_FLOOR > 0 to guarantee well-defined
Poisson distributions in extreme mismatches.
"""

from elo import HOME_ADVANTAGE

# Empirical fit constants (Session 9)
DIFF_PER_ELO = 0.0063           # goals per Elo point
BASE_TOTAL_GOALS = 2.5          # at evenly-matched
TOTAL_QUADRATIC_DENOM = 200_000

LAMBDA_FLOOR = 0.05  # protects against zero/negative lambdas in 1000+ Elo gaps


def elo_to_lambdas(
    elo_home: float,
    elo_away: float,
    neutral: bool = True,
) -> tuple[float, float]:
    """
    Predicted expected goals (lambda) for home and away teams.
    Home advantage is applied via +HOME_ADVANTAGE Elo for the home side
    when the match is non-neutral — same convention as elo.py.
    """
    eff_diff = elo_home - elo_away + (0 if neutral else HOME_ADVANTAGE)
    lam_diff = DIFF_PER_ELO * eff_diff
    lam_total = BASE_TOTAL_GOALS + (eff_diff ** 2) / TOTAL_QUADRATIC_DENOM
    lam_h = max(LAMBDA_FLOOR, (lam_total + lam_diff) / 2)
    lam_a = max(LAMBDA_FLOOR, (lam_total - lam_diff) / 2)
    return lam_h, lam_a