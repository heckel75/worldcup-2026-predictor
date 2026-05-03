"""
Basic Elo rating system for international football.

This is the v1 — keep it simple. Session 6 will add:
  - Margin-of-victory multiplier
  - Home advantage
"""

# K-factors by match category
K_FRIENDLY = 20
K_QUALIFIER = 30
K_MAJOR = 50
K_WORLD_CUP = 60

K_BY_CATEGORY = {
    "friendly": K_FRIENDLY,
    "qualifier": K_QUALIFIER,
    "major": K_MAJOR,
    "world_cup": K_WORLD_CUP,
}


def classify_match(tournament: str) -> str:
    """Map a raw tournament name (from the Kaggle CSV) to a K-bucket."""
    t = tournament.lower()
    if "friendly" in t:
        return "friendly"
    if "world cup" in t and "qualification" not in t:
        return "world_cup"
    if "qualification" in t or "qualifier" in t:
        return "qualifier"
    # Everything else (Euro, Copa, AFCON, Asian Cup, Gold Cup, Nations League finals, etc.)
    return "major"


class EloSystem:
    def __init__(self, default_rating: float = 1500.0):
        self.default_rating = default_rating
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        """Return a team's rating; new teams start at default_rating."""
        return self.ratings.get(team, self.default_rating)

    def expected_score(self, team_a: str, team_b: str) -> float:
        """Expected score for team A (1=win, 0.5=draw, 0=loss)."""
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str,
    ) -> None:
        """Update both teams' ratings based on the result."""
        # Actual result from home team's perspective
        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5
        actual_away = 1.0 - actual_home

        # Expected scores
        expected_home = self.expected_score(home_team, away_team)
        expected_away = 1.0 - expected_home

        # K factor for this match type
        k = K_BY_CATEGORY[classify_match(tournament)]

        # Apply updates
        self.ratings[home_team] = (
            self.get_rating(home_team) + k * (actual_home - expected_home)
        )
        self.ratings[away_team] = (
            self.get_rating(away_team) + k * (actual_away - expected_away)
        )

    def top_n(self, n: int = 20) -> list[tuple[str, float]]:
        """Return the top n teams by rating."""
        return sorted(self.ratings.items(), key=lambda x: -x[1])[:n]