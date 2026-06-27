"""
predictor.py
Loads the trained model and provides prediction + analytics functions
for the Streamlit frontend.
"""

import pandas as pd
import numpy as np
from src.feature_engineering import (
    load_and_prepare_data,
    compute_elo_ratings,
    _build_team_match_history,
    _compute_historical_stats,
    _compute_form_stats,
    _compute_h2h_stats,
    get_feature_columns,
    categorize_tournament,
)
from src.model import load_model
from src.world_cup_2026 import (
    WORLD_CUP_2026_TEAMS,
    WORLD_CUP_TOURNAMENT,
    load_all_fixtures,
    get_fixture_dates,
    format_fixture_date,
)


class MatchPredictor:
    """Provides match predictions and analytics from trained model + dataset."""

    def __init__(self, data_path: str):
        self.df = load_and_prepare_data(data_path)
        self.elo_ratings = compute_elo_ratings(self.df)
        self.history = _build_team_match_history(self.df)
        self.model, self.label_encoder, self.metadata = load_model()
        self.feature_columns = self.metadata["feature_columns"]
        self.data_path = data_path
        self.teams = sorted(
            t for t in WORLD_CUP_2026_TEAMS
            if t in self.df["home_team"].values or t in self.df["away_team"].values
        )
        self.fixtures = load_all_fixtures(data_path)

    def refresh_fixtures(self, simulated_winners: dict | None = None):
        """Reload and resolve fixtures with the given simulated winners."""
        self.fixtures = load_all_fixtures(self.data_path, simulated_winners)

    def reload_data(self):
        """Reload the dataset and recalculate ELO/history."""
        self.df = load_and_prepare_data(self.data_path)
        self.elo_ratings = compute_elo_ratings(self.df)
        self.history = _build_team_match_history(self.df)
        self.teams = sorted(
            t for t in WORLD_CUP_2026_TEAMS
            if t in self.df["home_team"].values or t in self.df["away_team"].values
        )
        # Force reload default fixtures without simulations initially
        self.refresh_fixtures()


    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(self, home_team: str, away_team: str, tournament: str = WORLD_CUP_TOURNAMENT) -> dict:
        """
        Predict match outcome probabilities.

        Returns dict with:
            probabilities: {class: probability}
            predicted_outcome: str
            confidence: float
        """
        features = self._build_features(home_team, away_team, tournament)
        X = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)  # Safety: fill any NaN with 0

        probas = self.model.predict_proba(X)[0]
        classes = self.label_encoder.classes_

        prob_dict = dict(zip(classes, probas))
        predicted_class = classes[np.argmax(probas)]
        confidence = float(np.max(probas))

        return {
            "probabilities": {k: round(float(v), 4) for k, v in prob_dict.items()},
            "predicted_outcome": predicted_class,
            "confidence": round(confidence, 4),
            "home_team": home_team,
            "away_team": away_team,
        }

    def _build_features(self, home_team: str, away_team: str, tournament: str) -> dict:
        """Build feature vector for a hypothetical match."""
        max_idx = len(self.df)

        home_hist = self.history.get(home_team, [])
        away_hist = self.history.get(away_team, [])

        home_stats = _compute_historical_stats(home_hist)
        away_stats = _compute_historical_stats(away_hist)
        home_form = _compute_form_stats(home_hist)
        away_form = _compute_form_stats(away_hist)
        h2h = _compute_h2h_stats(self.history, home_team, away_team, max_idx + 1)

        home_elo = self.elo_ratings.get(home_team, 1500)
        away_elo = self.elo_ratings.get(away_team, 1500)

        tournament_group = categorize_tournament(tournament)
        is_world_cup = tournament_group == "World Cup"

        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "home_win_rate": home_stats["win_rate"],
            "home_draw_rate": home_stats["draw_rate"],
            "home_loss_rate": home_stats["loss_rate"],
            "home_avg_goals_scored": home_stats["avg_goals_scored"],
            "home_avg_goals_conceded": home_stats["avg_goals_conceded"],
            "home_goal_diff": home_stats["goal_difference"],
            "away_win_rate": away_stats["win_rate"],
            "away_draw_rate": away_stats["draw_rate"],
            "away_loss_rate": away_stats["loss_rate"],
            "away_avg_goals_scored": away_stats["avg_goals_scored"],
            "away_avg_goals_conceded": away_stats["avg_goals_conceded"],
            "away_goal_diff": away_stats["goal_difference"],
            "home_form_win_rate": home_form["form_win_rate"],
            "home_form_goals_scored": home_form["form_goals_scored"],
            "home_form_goals_conceded": home_form["form_goals_conceded"],
            "home_form_score": home_form["form_score"],
            "away_form_win_rate": away_form["form_win_rate"],
            "away_form_goals_scored": away_form["form_goals_scored"],
            "away_form_goals_conceded": away_form["form_goals_conceded"],
            "away_form_score": away_form["form_score"],
            "h2h_matches": h2h["h2h_matches"],
            "h2h_home_win_pct": h2h["h2h_win_pct"],
            "h2h_draw_pct": h2h["h2h_draw_pct"],
            "is_neutral": int(is_world_cup),
            "is_world_cup": int(is_world_cup),
            "is_friendly": 0,
            "is_wc_qual": 0,
        }

    # ------------------------------------------------------------------
    # Recent Form
    # ------------------------------------------------------------------

    def get_recent_form(self, team: str, n: int = 5) -> list:
        """
        Get the last n match results for a team.
        Returns list of dicts with: date, opponent, goals_for, goals_against, result, is_home
        """
        import math
        if team not in self.history:
            return []

        # Filter out matches with NaN scores (future/scheduled)
        valid = [m for m in self.history[team]
                 if not (math.isnan(m[1]) if isinstance(m[1], float) else False)]
        recent = valid[-n:]
        results = []
        for m in reversed(recent):
            results.append({
                "date": m[0].strftime("%Y-%m-%d") if hasattr(m[0], "strftime") else str(m[0]),
                "opponent": m[4],
                "goals_for": m[1],
                "goals_against": m[2],
                "result": m[3],
                "is_home": m[5],
            })
        return results

    def get_form_string(self, team: str, n: int = 5) -> list:
        """Get a simple list of W/D/L for last n matches (most recent first)."""
        form = self.get_recent_form(team, n)
        return [m["result"] for m in form]

    def get_form_summary(self, team: str, n: int = 5) -> dict:
        """Get last-n form counts and normalized form score."""
        form = self.get_form_string(team, n)
        wins = form.count("W")
        draws = form.count("D")
        losses = form.count("L")
        form_pts = wins * 3 + draws
        max_pts = len(form) * 3 if form else 1
        return {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "form_score": form_pts / max_pts if max_pts > 0 else 0.0,
            "matches": len(form),
        }

    # ------------------------------------------------------------------
    # Head-to-Head
    # ------------------------------------------------------------------

    def get_head_to_head(self, team_a: str, team_b: str, n: int = 5) -> dict:
        """
        Get head-to-head analysis between two teams.
        Returns summary stats + last n meetings.
        """
        if team_a not in self.history:
            return {"total": 0, "team_a_wins": 0, "team_b_wins": 0, "draws": 0, "meetings": []}

        all_h2h = [m for m in self.history[team_a] if m[4] == team_b]

        total = len(all_h2h)
        a_wins = sum(1 for m in all_h2h if m[3] == "W")
        draws = sum(1 for m in all_h2h if m[3] == "D")
        b_wins = total - a_wins - draws

        # Last n meetings
        recent = all_h2h[-n:] if len(all_h2h) >= n else all_h2h
        meetings = []
        for m in reversed(recent):
            team_one_goals = m[1] if m[5] else m[2]
            team_two_goals = m[2] if m[5] else m[1]

            meetings.append({
                "date": m[0].strftime("%Y-%m-%d") if hasattr(m[0], "strftime") else str(m[0]),
                "team_one": team_a,
                "team_two": team_b,
                "team_one_goals": team_one_goals,
                "team_two_goals": team_two_goals,
                "home_team": team_a if m[5] else team_b,
                "away_team": team_b if m[5] else team_a,
                "home_goals": m[1] if m[5] else m[2],
                "away_goals": m[2] if m[5] else m[1],
                "result_for_team_a": m[3],
            })

        return {
            "total": total,
            "team_a_wins": a_wins,
            "team_b_wins": b_wins,
            "draws": draws,
            "meetings": meetings,
        }

    # ------------------------------------------------------------------
    # Team Statistics
    # ------------------------------------------------------------------

    def _filter_matches_by_tournament(self, matches: list, tournament_filter: str) -> list:
        """Filter team match history by tournament category."""
        if tournament_filter == "All":
            return matches

        target = self._tournament_filter_to_group(tournament_filter)
        filtered = []
        for m in matches:
            idx = m[6]
            if idx < len(self.df) and self.df.iloc[idx]["tournament_group"] == target:
                filtered.append(m)
        return filtered

    @staticmethod
    def _tournament_filter_to_group(tournament_filter: str) -> str:
        """Map UI tournament label to internal tournament group."""
        group_map = {
            "FIFA World Cup": "World Cup",
            "World Cup qualification": "WC Qualification",
            "Friendly": "Friendly",
        }
        return group_map.get(tournament_filter, tournament_filter)

    def get_team_stats(self, team: str, tournament_filter: str = "All") -> dict:
        """Get aggregate statistics for a team, optionally filtered by tournament."""
        if team not in self.history:
            return _compute_historical_stats([])

        matches = self._filter_matches_by_tournament(self.history[team], tournament_filter)
        stats = _compute_historical_stats(matches)
        stats["elo_rating"] = self.elo_ratings.get(team, 1500)
        return stats

    # ------------------------------------------------------------------
    # Historical Performance Trends
    # ------------------------------------------------------------------

    def get_performance_trend(self, team: str, tournament_filter: str = "All") -> pd.DataFrame:
        """
        Get yearly performance trends for a team.
        Returns DataFrame with columns: year, win_rate, avg_goals_scored,
        avg_goals_conceded, matches_played
        """
        if team not in self.history:
            return pd.DataFrame()

        matches = self._filter_matches_by_tournament(self.history[team], tournament_filter)
        yearly = {}

        for m in matches:
            year = m[0].year if hasattr(m[0], "year") else int(str(m[0])[:4])
            if year not in yearly:
                yearly[year] = {"wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "n": 0}
            yearly[year]["n"] += 1
            yearly[year]["gf"] += m[1]
            yearly[year]["ga"] += m[2]
            if m[3] == "W":
                yearly[year]["wins"] += 1
            elif m[3] == "D":
                yearly[year]["draws"] += 1
            else:
                yearly[year]["losses"] += 1

        rows = []
        for year in sorted(yearly.keys()):
            d = yearly[year]
            n = d["n"]
            rows.append({
                "year": year,
                "win_rate": d["wins"] / n if n > 0 else 0,
                "avg_goals_scored": d["gf"] / n if n > 0 else 0,
                "avg_goals_conceded": d["ga"] / n if n > 0 else 0,
                "matches_played": n,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Feature Importance
    # ------------------------------------------------------------------

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Get top N feature importances as a DataFrame."""
        importance = self.metadata.get("feature_importance", {})
        df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in importance.items()
        ]).sort_values("importance", ascending=False).head(top_n)

        # Clean up feature names for display
        name_map = {
            "home_elo": "Home Elo Rating",
            "away_elo": "Away Elo Rating",
            "elo_diff": "Elo Difference",
            "home_win_rate": "Home Win Rate",
            "home_draw_rate": "Home Draw Rate",
            "home_loss_rate": "Home Loss Rate",
            "home_avg_goals_scored": "Home Avg Goals Scored",
            "home_avg_goals_conceded": "Home Avg Goals Conceded",
            "home_goal_diff": "Home Goal Difference",
            "away_win_rate": "Away Win Rate",
            "away_draw_rate": "Away Draw Rate",
            "away_loss_rate": "Away Loss Rate",
            "away_avg_goals_scored": "Away Avg Goals Scored",
            "away_avg_goals_conceded": "Away Avg Goals Conceded",
            "away_goal_diff": "Away Goal Difference",
            "home_form_win_rate": "Home Recent Form (Win Rate)",
            "home_form_goals_scored": "Home Recent Goals Scored",
            "home_form_goals_conceded": "Home Recent Goals Conceded",
            "home_form_score": "Home Form Score",
            "away_form_win_rate": "Away Recent Form (Win Rate)",
            "away_form_goals_scored": "Away Recent Goals Scored",
            "away_form_goals_conceded": "Away Recent Goals Conceded",
            "away_form_score": "Away Form Score",
            "h2h_matches": "Head-to-Head Matches",
            "h2h_home_win_pct": "H2H Home Win %",
            "h2h_draw_pct": "H2H Draw %",
            "is_neutral": "Neutral Venue",
            "is_world_cup": "World Cup Match",
            "is_friendly": "Friendly Match",
            "is_wc_qual": "WC Qualifier",
        }
        df["display_name"] = df["feature"].map(name_map).fillna(df["feature"])
        return df

    # ------------------------------------------------------------------
    # Model Info
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict:
        """Get metadata about the trained model."""
        best_name = self.metadata["best_model_name"]
        best_metrics = self.metadata["all_model_metrics"][best_name]
        return {
            "model_name": best_name,
            "accuracy": best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall": best_metrics["recall"],
            "f1_score": best_metrics["f1_score"],
            "training_date": self.metadata.get("training_date", "N/A"),
            "all_model_metrics": self.metadata["all_model_metrics"],
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_teams(self) -> list:
        """Get sorted list of World Cup 2026 teams available in the dataset."""
        return self.teams

    def get_fixtures(self) -> list:
        """Get all World Cup 2026 fixtures."""
        return self.fixtures

    def get_fixture_dates(self) -> list:
        """Get sorted fixture dates."""
        return get_fixture_dates(self.fixtures)

    def get_fixture_by_id(self, fixture_id: str) -> dict | None:
        for fixture in self.fixtures:
            if fixture["id"] == fixture_id:
                return fixture
        return None

    def format_fixture_date(self, date_str: str) -> str:
        return format_fixture_date(date_str)

    def get_confusion_matrix(self, model_name: str = None) -> dict:
        """Return confusion matrix and class labels for a trained model."""
        name = model_name or self.metadata["best_model_name"]
        metrics = self.metadata["all_model_metrics"].get(name)
        if not metrics:
            return {"matrix": [], "classes": self.metadata.get("classes", [])}
        return {
            "matrix": metrics["confusion_matrix"],
            "classes": self.metadata.get("classes", []),
            "model_name": name,
        }
