"""
Elo rating system for international football.

Session 5: basic Elo.
Session 6: + margin-of-victory multiplier, + home advantage.
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

# Home advantage in international football, in Elo points.
# Smaller than club football (~100) because national teams play "home" less
# often and crowds at neutral venues are usually mixed.
HOME_ADVANTAGE = 60


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


def mov_multiplier(goal_diff: int) -> float:
    """
    Margin-of-victory multiplier (World Football Elo formula).

      diff 0 or 1 -> 1.0
      diff 2      -> 1.5
      diff 3+     -> (11 + diff) / 8     # 1.75, 1.875, 2.0, 2.125, ...

    Diminishing returns: a 7-0 only counts ~2.25x a 1-0, not 7x.
    """
    diff = abs(goal_diff)
    if diff <= 1:
        return 1.0
    elif diff == 2:
        return 1.5
    else:
        return (11 + diff) / 8


class EloSystem:
    def __init__(self, default_rating: float = 1500.0):
        self.default_rating = default_rating
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        """Return a team's rating; new teams start at default_rating."""
        return self.ratings.get(team, self.default_rating)

    def expected_score(
        self,
        team_a: str,
        team_b: str,
        home_advantage: bool = False,
    ) -> float:
        """
        Expected score for team A (1=win, 0.5=draw, 0=loss).

        If home_advantage=True, team A is treated as the home side and gets
        +HOME_ADVANTAGE Elo points for this calculation only. The boost is
        NOT stored on team A's rating — it's a one-shot adjustment used
        only for predicting this match.
        """
        ra = self.get_rating(team_a) + (HOME_ADVANTAGE if home_advantage else 0)
        rb = self.get_rating(team_b)
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str,
        neutral: bool = True,
    ) -> None:
        """
        Update both teams' ratings based on the result.

        `neutral=False` applies the home-advantage bonus to the home team's
        expected score (but not to the stored rating update).
        """
        # Actual result from home team's perspective
        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5
        actual_away = 1.0 - actual_home

        # Expected scores — home advantage applied only when match isn't neutral
        expected_home = self.expected_score(
            home_team, away_team, home_advantage=not neutral
        )
        expected_away = 1.0 - expected_home

        # K factor for this match type, scaled by margin of victory
        k_base = K_BY_CATEGORY[classify_match(tournament)]
        k_eff = k_base * mov_multiplier(home_score - away_score)

        # Apply updates using stored ratings (NOT the home-boosted version)
        self.ratings[home_team] = (
            self.get_rating(home_team) + k_eff * (actual_home - expected_home)
        )
        self.ratings[away_team] = (
            self.get_rating(away_team) + k_eff * (actual_away - expected_away)
        )

    def top_n(self, n: int = 20) -> list[tuple[str, float]]:
        """Return the top n teams by rating."""
        return sorted(self.ratings.items(), key=lambda x: -x[1])[:n]