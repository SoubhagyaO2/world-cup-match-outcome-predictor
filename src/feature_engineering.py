"""
feature_engineering.py
Computes all ML features from raw match data using only historical information
(no data leakage). Features include team strength, scoring, recent form,
head-to-head, Elo ratings, and match context.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Tournament grouping
# ---------------------------------------------------------------------------
TOURNAMENT_GROUPS = {
    "FIFA World Cup": "World Cup",
    "FIFA World Cup qualification": "WC Qualification",
    "Friendly": "Friendly",
    "Confederations Cup": "Other",
    "UEFA Euro": "Other",
    "UEFA Euro qualification": "Other",
    "Copa América": "Other",
    "African Cup of Nations": "Other",
    "AFC Asian Cup": "Other",
    "AFC Asian Cup qualification": "Other",
    "Gold Cup": "Other",
}


def categorize_tournament(tournament: str) -> str:
    """Map a tournament name to a high-level category."""
    return TOURNAMENT_GROUPS.get(tournament, "Other")


# ---------------------------------------------------------------------------
# Elo Rating System
# ---------------------------------------------------------------------------
DEFAULT_ELO = 1500
K_FACTOR = 32


def _expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def compute_elo_ratings(df: pd.DataFrame) -> dict:
    """
    Compute Elo ratings for all teams by iterating through matches
    chronologically. Returns a dict mapping team -> current Elo.
    Also adds 'home_elo' and 'away_elo' columns (pre-match) to df.
    """
    elo = {}
    home_elos = []
    away_elos = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        h_elo = elo.get(home, DEFAULT_ELO)
        a_elo = elo.get(away, DEFAULT_ELO)

        home_elos.append(h_elo)
        away_elos.append(a_elo)

        # Actual scores
        if row["home_score"] > row["away_score"]:
            h_actual, a_actual = 1.0, 0.0
        elif row["home_score"] < row["away_score"]:
            h_actual, a_actual = 0.0, 1.0
        else:
            h_actual, a_actual = 0.5, 0.5

        h_expected = _expected_score(h_elo, a_elo)
        a_expected = _expected_score(a_elo, h_elo)

        elo[home] = h_elo + K_FACTOR * (h_actual - h_expected)
        elo[away] = a_elo + K_FACTOR * (a_actual - a_expected)

    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    return elo


# ---------------------------------------------------------------------------
# Historical stats computation (vectorized where possible)
# ---------------------------------------------------------------------------

def _build_team_match_history(df: pd.DataFrame) -> dict:
    """
    Build a chronological list of match records per team.
    Each record: (date, goals_for, goals_against, result, opponent, is_home)
    """
    history = {}

    for idx, row in df.iterrows():
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]
        hg = row["home_score"]
        ag = row["away_score"]

        if hg > ag:
            h_res, a_res = "W", "L"
        elif hg < ag:
            h_res, a_res = "L", "W"
        else:
            h_res, a_res = "D", "D"

        history.setdefault(home, []).append(
            (date, hg, ag, h_res, away, True, idx)
        )
        history.setdefault(away, []).append(
            (date, ag, hg, a_res, home, False, idx)
        )

    return history


def _compute_historical_stats(matches: list) -> dict:
    """Compute aggregate stats from a list of match records."""
    import math
    # Filter out matches with NaN scores (future/scheduled matches)
    matches = [m for m in matches if not (math.isnan(m[1]) if isinstance(m[1], float) else False)]
    if not matches:
        return {
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "avg_goals_scored": 0.0,
            "avg_goals_conceded": 0.0,
            "goal_difference": 0.0,
            "matches_played": 0,
        }

    n = len(matches)
    wins = sum(1 for m in matches if m[3] == "W")
    draws = sum(1 for m in matches if m[3] == "D")
    losses = sum(1 for m in matches if m[3] == "L")
    gf = sum(m[1] for m in matches)
    ga = sum(m[2] for m in matches)

    return {
        "win_rate": wins / n,
        "draw_rate": draws / n,
        "loss_rate": losses / n,
        "avg_goals_scored": gf / n,
        "avg_goals_conceded": ga / n,
        "goal_difference": (gf - ga) / n,
        "matches_played": n,
    }


def _compute_form_stats(matches: list, n: int = 5) -> dict:
    """Compute recent form stats from the last n matches."""
    import math
    # Filter out matches with NaN scores
    matches = [m for m in matches if not (math.isnan(m[1]) if isinstance(m[1], float) else False)]
    recent = matches[-n:] if len(matches) >= n else matches
    if not recent:
        return {
            "form_win_rate": 0.0,
            "form_draw_rate": 0.0,
            "form_loss_rate": 0.0,
            "form_goals_scored": 0.0,
            "form_goals_conceded": 0.0,
            "form_score": 0.0,
        }

    nr = len(recent)
    wins = sum(1 for m in recent if m[3] == "W")
    draws = sum(1 for m in recent if m[3] == "D")
    gf = sum(m[1] for m in recent)
    ga = sum(m[2] for m in recent)

    # Form score: W=3, D=1, L=0, normalized to 0-1
    form_pts = sum(3 if m[3] == "W" else (1 if m[3] == "D" else 0) for m in recent)
    max_pts = nr * 3

    return {
        "form_win_rate": wins / nr,
        "form_draw_rate": draws / nr,
        "form_loss_rate": 1 - (wins + draws) / nr,
        "form_goals_scored": gf / nr,
        "form_goals_conceded": ga / nr,
        "form_score": form_pts / max_pts if max_pts > 0 else 0.0,
    }


def _compute_h2h_stats(history: dict, team_a: str, team_b: str, match_idx: int) -> dict:
    """Compute head-to-head stats between two teams before a given match index."""
    if team_a not in history:
        return {
            "h2h_matches": 0,
            "h2h_win_pct": 0.0,
            "h2h_draw_pct": 0.0,
            "h2h_loss_pct": 0.0,
        }

    h2h = [m for m in history[team_a] if m[4] == team_b and m[6] < match_idx]
    if not h2h:
        return {
            "h2h_matches": 0,
            "h2h_win_pct": 0.0,
            "h2h_draw_pct": 0.0,
            "h2h_loss_pct": 0.0,
        }

    n = len(h2h)
    wins = sum(1 for m in h2h if m[3] == "W")
    draws = sum(1 for m in h2h if m[3] == "D")

    return {
        "h2h_matches": n,
        "h2h_win_pct": wins / n,
        "h2h_draw_pct": draws / n,
        "h2h_loss_pct": 1 - (wins + draws) / n,
    }


# ---------------------------------------------------------------------------
# Main feature engineering pipeline
# ---------------------------------------------------------------------------

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load CSV and prepare base columns."""
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Drop future/scheduled matches with missing scores
    df = df.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df["neutral"] = df["neutral"].astype(bool)
    df["tournament_group"] = df["tournament"].apply(categorize_tournament)

    # Target variable
    df["result"] = np.where(
        df["home_score"] > df["away_score"], "Home Win",
        np.where(df["home_score"] < df["away_score"], "Away Win", "Draw")
    )

    return df


def engineer_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Engineer all features for each match. Uses only historical data
    available before each match (no leakage).
    """
    if verbose:
        print("Computing Elo ratings...")
    elo_ratings = compute_elo_ratings(df)

    if verbose:
        print("Building team match histories...")
    history = _build_team_match_history(df)

    # Pre-compute cumulative histories for each team
    # We'll track how many matches each team has played up to each row
    team_match_counts = {}

    feature_rows = []
    total = len(df)

    for idx, row in df.iterrows():
        if verbose and idx % 5000 == 0:
            print(f"  Processing match {idx}/{total}...")

        home = row["home_team"]
        away = row["away_team"]

        # Get matches before this one
        home_history_before = [m for m in history[home] if m[6] < idx]
        away_history_before = [m for m in history[away] if m[6] < idx]

        # Historical stats
        home_stats = _compute_historical_stats(home_history_before)
        away_stats = _compute_historical_stats(away_history_before)

        # Recent form
        home_form = _compute_form_stats(home_history_before)
        away_form = _compute_form_stats(away_history_before)

        # Head-to-head
        h2h = _compute_h2h_stats(history, home, away, idx)

        features = {
            # Elo (home_elo and away_elo already in df from compute_elo_ratings)
            "elo_diff": row["home_elo"] - row["away_elo"],

            # Home team historical stats
            "home_win_rate": home_stats["win_rate"],
            "home_draw_rate": home_stats["draw_rate"],
            "home_loss_rate": home_stats["loss_rate"],
            "home_avg_goals_scored": home_stats["avg_goals_scored"],
            "home_avg_goals_conceded": home_stats["avg_goals_conceded"],
            "home_goal_diff": home_stats["goal_difference"],
            "home_matches_played": home_stats["matches_played"],

            # Away team historical stats
            "away_win_rate": away_stats["win_rate"],
            "away_draw_rate": away_stats["draw_rate"],
            "away_loss_rate": away_stats["loss_rate"],
            "away_avg_goals_scored": away_stats["avg_goals_scored"],
            "away_avg_goals_conceded": away_stats["avg_goals_conceded"],
            "away_goal_diff": away_stats["goal_difference"],
            "away_matches_played": away_stats["matches_played"],

            # Home team form
            "home_form_win_rate": home_form["form_win_rate"],
            "home_form_goals_scored": home_form["form_goals_scored"],
            "home_form_goals_conceded": home_form["form_goals_conceded"],
            "home_form_score": home_form["form_score"],

            # Away team form
            "away_form_win_rate": away_form["form_win_rate"],
            "away_form_goals_scored": away_form["form_goals_scored"],
            "away_form_goals_conceded": away_form["form_goals_conceded"],
            "away_form_score": away_form["form_score"],

            # Head-to-head
            "h2h_matches": h2h["h2h_matches"],
            "h2h_home_win_pct": h2h["h2h_win_pct"],
            "h2h_draw_pct": h2h["h2h_draw_pct"],

            # Context
            "is_neutral": int(row["neutral"]),
            "is_world_cup": int(row["tournament_group"] == "World Cup"),
            "is_friendly": int(row["tournament_group"] == "Friendly"),
            "is_wc_qual": int(row["tournament_group"] == "WC Qualification"),
        }

        feature_rows.append(features)

    features_df = pd.DataFrame(feature_rows, index=df.index)

    # Combine with original data
    result_df = pd.concat([df, features_df], axis=1)

    if verbose:
        print(f"Feature engineering complete. Shape: {result_df.shape}")

    return result_df


def get_feature_columns() -> list:
    """Return the list of feature column names used for ML."""
    return [
        "home_elo", "away_elo", "elo_diff",
        "home_win_rate", "home_draw_rate", "home_loss_rate",
        "home_avg_goals_scored", "home_avg_goals_conceded", "home_goal_diff",
        "away_win_rate", "away_draw_rate", "away_loss_rate",
        "away_avg_goals_scored", "away_avg_goals_conceded", "away_goal_diff",
        "home_form_win_rate", "home_form_goals_scored", "home_form_goals_conceded",
        "home_form_score",
        "away_form_win_rate", "away_form_goals_scored", "away_form_goals_conceded",
        "away_form_score",
        "h2h_matches", "h2h_home_win_pct", "h2h_draw_pct",
        "is_neutral", "is_world_cup", "is_friendly", "is_wc_qual",
    ]
