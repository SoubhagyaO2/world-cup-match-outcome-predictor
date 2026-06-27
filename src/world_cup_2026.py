"""
world_cup_2026.py
Teams and fixtures for the 2026 FIFA World Cup.
Group-stage fixtures are loaded from the dataset; knockout fixtures use
confirmed bracket matchups (as of June 2026).
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

WORLD_CUP_TOURNAMENT = "FIFA World Cup"
WORLD_CUP_LABEL = "FIFA World Cup 2026"

# 48 teams — names aligned with the international results dataset
WORLD_CUP_2026_TEAMS = sorted([
    "Algeria", "Argentina", "Australia", "Austria", "Belgium", "Bosnia and Herzegovina",
    "Brazil", "Canada", "Cape Verde", "Colombia", "Croatia", "Curaçao", "Czech Republic",
    "DR Congo", "Ecuador", "Egypt", "England", "France", "Germany", "Ghana", "Haiti",
    "Iran", "Iraq", "Ivory Coast", "Japan", "Jordan", "Mexico", "Morocco", "Netherlands",
    "New Zealand", "Norway", "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia",
    "Scotland", "Senegal", "South Africa", "South Korea", "Spain", "Sweden", "Switzerland",
    "Tunisia", "Turkey", "United States", "Uruguay", "Uzbekistan",
])

# Kickoff times (local) for group-stage matches — source: published WC 2026 schedule
GROUP_STAGE_TIMES = {
    ("2026-06-11", "Mexico", "South Africa"): "13:00",
    ("2026-06-11", "South Korea", "Czech Republic"): "20:00",
    ("2026-06-12", "Canada", "Bosnia and Herzegovina"): "15:00",
    ("2026-06-12", "United States", "Paraguay"): "18:00",
    ("2026-06-13", "Haiti", "Scotland"): "21:00",
    ("2026-06-13", "Australia", "Turkey"): "21:00",
    ("2026-06-13", "Brazil", "Morocco"): "18:00",
    ("2026-06-13", "Qatar", "Switzerland"): "12:00",
    ("2026-06-14", "Ivory Coast", "Ecuador"): "19:00",
    ("2026-06-14", "Germany", "Curaçao"): "12:00",
    ("2026-06-14", "Netherlands", "Japan"): "15:00",
    ("2026-06-14", "Sweden", "Tunisia"): "20:00",
    ("2026-06-15", "Saudi Arabia", "Uruguay"): "18:00",
    ("2026-06-15", "Spain", "Cape Verde"): "12:00",
    ("2026-06-15", "Iran", "New Zealand"): "18:00",
    ("2026-06-15", "Belgium", "Egypt"): "12:00",
    ("2026-06-16", "France", "Senegal"): "15:00",
    ("2026-06-16", "Iraq", "Norway"): "18:00",
    ("2026-06-16", "Argentina", "Algeria"): "20:00",
    ("2026-06-16", "Austria", "Jordan"): "21:00",
    ("2026-06-17", "Ghana", "Panama"): "19:00",
    ("2026-06-17", "England", "Croatia"): "15:00",
    ("2026-06-17", "Portugal", "DR Congo"): "12:00",
    ("2026-06-17", "Uzbekistan", "Colombia"): "20:00",
    ("2026-06-18", "Czech Republic", "South Africa"): "12:00",
    ("2026-06-18", "Switzerland", "Bosnia and Herzegovina"): "12:00",
    ("2026-06-18", "Canada", "Qatar"): "15:00",
    ("2026-06-18", "Mexico", "South Korea"): "19:00",
    ("2026-06-19", "Brazil", "Haiti"): "21:00",
    ("2026-06-19", "Scotland", "Morocco"): "18:00",
    ("2026-06-19", "Turkey", "Paraguay"): "20:00",
    ("2026-06-19", "United States", "Australia"): "12:00",
    ("2026-06-20", "Germany", "Ivory Coast"): "16:00",
    ("2026-06-20", "Ecuador", "Curaçao"): "19:00",
    ("2026-06-20", "Netherlands", "Sweden"): "12:00",
    ("2026-06-20", "Tunisia", "Japan"): "22:00",
    ("2026-06-21", "Uruguay", "Cape Verde"): "18:00",
    ("2026-06-21", "Spain", "Saudi Arabia"): "12:00",
    ("2026-06-21", "Belgium", "Iran"): "12:00",
    ("2026-06-21", "New Zealand", "Egypt"): "18:00",
    ("2026-06-22", "Norway", "Senegal"): "20:00",
    ("2026-06-22", "France", "Iraq"): "17:00",
    ("2026-06-22", "Argentina", "Austria"): "12:00",
    ("2026-06-22", "Jordan", "Algeria"): "20:00",
    ("2026-06-23", "England", "Ghana"): "16:00",
    ("2026-06-23", "Panama", "Croatia"): "19:00",
    ("2026-06-23", "Portugal", "Uzbekistan"): "12:00",
    ("2026-06-23", "Colombia", "DR Congo"): "20:00",
    ("2026-06-24", "Scotland", "Brazil"): "18:00",
    ("2026-06-24", "Morocco", "Haiti"): "18:00",
    ("2026-06-24", "Switzerland", "Canada"): "12:00",
    ("2026-06-24", "Bosnia and Herzegovina", "Qatar"): "12:00",
    ("2026-06-24", "Czech Republic", "Mexico"): "19:00",
    ("2026-06-24", "South Africa", "South Korea"): "19:00",
    ("2026-06-25", "Curaçao", "Ivory Coast"): "16:00",
    ("2026-06-25", "Ecuador", "Germany"): "16:00",
    ("2026-06-25", "Japan", "Sweden"): "18:00",
    ("2026-06-25", "Tunisia", "Netherlands"): "18:00",
    ("2026-06-25", "Turkey", "United States"): "19:00",
    ("2026-06-25", "Paraguay", "Australia"): "19:00",
    ("2026-06-26", "Norway", "France"): "15:00",
    ("2026-06-26", "Senegal", "Iraq"): "15:00",
    ("2026-06-26", "Egypt", "Iran"): "20:00",
    ("2026-06-26", "New Zealand", "Belgium"): "20:00",
    ("2026-06-26", "Cape Verde", "Saudi Arabia"): "19:00",
    ("2026-06-26", "Uruguay", "Spain"): "18:00",
    ("2026-06-27", "Panama", "England"): "17:00",
    ("2026-06-27", "Croatia", "Ghana"): "17:00",
    ("2026-06-27", "Algeria", "Austria"): "21:00",
    ("2026-06-27", "Jordan", "Argentina"): "21:00",
    ("2026-06-27", "Colombia", "Portugal"): "19:30",
    ("2026-06-27", "DR Congo", "Uzbekistan"): "19:30",
}

# Confirmed round-of-32 matchups (post group stage) with official match IDs
KNOCKOUT_FIXTURES = [
    {"id": "M73", "date": "2026-06-28", "time": "12:00", "team1": "South Africa", "team2": "Canada", "stage": "Round of 32"},
    {"id": "M74", "date": "2026-06-29", "time": "16:30", "team1": "Germany", "team2": "Paraguay", "stage": "Round of 32"},
    {"id": "M75", "date": "2026-06-29", "time": "19:00", "team1": "Netherlands", "team2": "Morocco", "stage": "Round of 32"},
    {"id": "M76", "date": "2026-06-29", "time": "12:00", "team1": "Brazil", "team2": "Japan", "stage": "Round of 32"},
    {"id": "M77", "date": "2026-06-30", "time": "17:00", "team1": "France", "team2": "Sweden", "stage": "Round of 32"},
    {"id": "M78", "date": "2026-06-30", "time": "12:00", "team1": "Ivory Coast", "team2": "Norway", "stage": "Round of 32"},
    {"id": "M79", "date": "2026-06-30", "time": "19:00", "team1": "Mexico", "team2": "Scotland", "stage": "Round of 32"},
    {"id": "M80", "date": "2026-07-01", "time": "17:00", "team1": "Egypt", "team2": "Czech Republic", "stage": "Round of 32"},
    {"id": "M81", "date": "2026-07-01", "time": "12:00", "team1": "England", "team2": "Cape Verde", "stage": "Round of 32"},
    {"id": "M82", "date": "2026-07-01", "time": "17:00", "team1": "United States", "team2": "Bosnia and Herzegovina", "stage": "Round of 32"},
    {"id": "M83", "date": "2026-07-02", "time": "12:00", "team1": "Spain", "team2": "Austria", "stage": "Round of 32"},
    {"id": "M84", "date": "2026-07-02", "time": "19:00", "team1": "Portugal", "team2": "Ghana", "stage": "Round of 32"},
    {"id": "M85", "date": "2026-07-03", "time": "14:00", "team1": "Australia", "team2": "Iran", "stage": "Round of 32"},
    {"id": "M86", "date": "2026-07-03", "time": "18:00", "team1": "Argentina", "team2": "Uruguay", "stage": "Round of 32"},
    {"id": "M87", "date": "2026-07-02", "time": "13:00", "team1": "Switzerland", "team2": "Belgium", "stage": "Round of 32"},
    {"id": "M88", "date": "2026-07-03", "time": "20:30", "team1": "Colombia", "team2": "Croatia", "stage": "Round of 32"},
]

# Bracket dependency mapping
BRACKET_CONFIG = {
    # Round of 16
    "M89": {"t1_src": "M73", "t2_src": "M75", "date": "2026-07-04", "time": "12:00", "stage": "Round of 16"},
    "M90": {"t1_src": "M74", "t2_src": "M77", "date": "2026-07-04", "time": "17:00", "stage": "Round of 16"},
    "M91": {"t1_src": "M76", "t2_src": "M78", "date": "2026-07-05", "time": "12:00", "stage": "Round of 16"},
    "M92": {"t1_src": "M79", "t2_src": "M80", "date": "2026-07-05", "time": "17:00", "stage": "Round of 16"},
    "M93": {"t1_src": "M83", "t2_src": "M84", "date": "2026-07-06", "time": "12:00", "stage": "Round of 16"},
    "M94": {"t1_src": "M81", "t2_src": "M82", "date": "2026-07-06", "time": "17:00", "stage": "Round of 16"},
    "M95": {"t1_src": "M86", "t2_src": "M88", "date": "2026-07-07", "time": "12:00", "stage": "Round of 16"},
    "M96": {"t1_src": "M85", "t2_src": "M87", "date": "2026-07-07", "time": "17:00", "stage": "Round of 16"},

    # Quarterfinals
    "M97": {"t1_src": "M89", "t2_src": "M90", "date": "2026-07-09", "time": "15:00", "stage": "Quarterfinals"},
    "M98": {"t1_src": "M91", "t2_src": "M92", "date": "2026-07-10", "time": "15:00", "stage": "Quarterfinals"},
    "M99": {"t1_src": "M93", "t2_src": "M94", "date": "2026-07-11", "time": "12:00", "stage": "Quarterfinals"},
    "M100": {"t1_src": "M95", "t2_src": "M96", "date": "2026-07-11", "time": "18:00", "stage": "Quarterfinals"},

    # Semifinals
    "M101": {"t1_src": "M97", "t2_src": "M98", "date": "2026-07-14", "time": "19:00", "stage": "Semifinals"},
    "M102": {"t1_src": "M99", "t2_src": "M100", "date": "2026-07-15", "time": "19:00", "stage": "Semifinals"},

    # Final
    "M103": {"t1_src": "M101", "t2_src": "M102", "date": "2026-07-19", "time": "15:00", "stage": "Final"},
}


def _fixture_id(date: str, team1: str, team2: str) -> str:
    return f"{date}|{team1}|{team2}"


def _parse_score(value) -> int | None:
    if pd.isna(value) or str(value).strip().upper() == "NA":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_shootout_winners(data_path: str) -> dict:
    """Load shootout winners from shootouts.csv if it exists."""
    shootouts_path = os.path.join(os.path.dirname(data_path), "shootouts.csv")
    winners = {}
    if not os.path.exists(shootouts_path):
        return winners
    try:
        df = pd.read_csv(shootouts_path)
        for _, row in df.iterrows():
            dt_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
            teams = frozenset([row["home_team"], row["away_team"]])
            winners[(dt_str, teams)] = row["winner"]
    except Exception as e:
        print(f"Error loading shootouts: {e}")
    return winners


def find_real_match_outcome(df: pd.DataFrame, team1: str, team2: str, date_str: str, shootout_winners: dict) -> tuple[bool, str | None, str | None]:
    """Search results dataframe for a completed match and return (played, score_str, winner_name)."""
    target_dt = pd.to_datetime(date_str)
    mask = (
        ((df["home_team"] == team1) & (df["away_team"] == team2)) |
        ((df["home_team"] == team2) & (df["away_team"] == team1))
    )
    possible_matches = df[mask].copy()
    if possible_matches.empty:
        return False, None, None

    possible_matches["date_diff"] = (pd.to_datetime(possible_matches["date"]) - target_dt).abs()
    match_row = possible_matches.sort_values("date_diff").iloc[0]

    # Only match if within 3 days tolerance
    if match_row["date_diff"].days > 3:
        return False, None, None

    home_score = _parse_score(match_row["home_score"])
    away_score = _parse_score(match_row["away_score"])

    if home_score is None or away_score is None:
        return False, None, None

    home_team = match_row["home_team"]
    away_team = match_row["away_team"]
    score_str = f"{home_score}-{away_score}" if home_team == team1 else f"{away_score}-{home_score}"

    if home_score > away_score:
        winner = home_team
    elif away_score > home_score:
        winner = away_team
    else:
        # Resolve via shootouts
        match_date_str = pd.to_datetime(match_row["date"]).strftime("%Y-%m-%d")
        teams_key = frozenset([home_team, away_team])
        winner = shootout_winners.get((match_date_str, teams_key))
        if not winner:
            winner = home_team  # fallback to avoid deadlock
    return True, score_str, winner


def load_group_stage_fixtures(data_path: str) -> list[dict]:
    """Load 2026 World Cup group-stage fixtures from the results dataset."""
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    wc = df[
        (df["tournament"] == WORLD_CUP_TOURNAMENT)
        & (df["date"] >= "2026-06-11")
        & (df["date"] <= "2026-06-27")
    ].copy()

    fixtures = []
    for _, row in wc.iterrows():
        team1 = row["home_team"]
        team2 = row["away_team"]
        if team1 not in WORLD_CUP_2026_TEAMS or team2 not in WORLD_CUP_2026_TEAMS:
            continue

        home_score = _parse_score(row["home_score"])
        away_score = _parse_score(row["away_score"])
        played = home_score is not None and away_score is not None
        kickoff = GROUP_STAGE_TIMES.get((row["date"], team1, team2), "15:00")

        fixtures.append({
            "id": _fixture_id(row["date"], team1, team2),
            "date": row["date"],
            "time": kickoff,
            "team1": team1,
            "team2": team2,
            "stage": "Group Stage",
            "played": played,
            "score": f"{home_score}-{away_score}" if played else None,
            "winner": team1 if played and home_score > away_score else (team2 if played and away_score > home_score else None)
        })

    return fixtures


def load_knockout_fixtures_dynamic(data_path: str, simulated_winners: dict | None = None) -> list[dict]:
    """Dynamically resolve knockout matchups based on real and simulated outcomes."""
    if simulated_winners is None:
        simulated_winners = {}

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    shootout_winners = load_shootout_winners(data_path)

    resolved = {}

    # 1. Resolve Round of 32
    for item in KNOCKOUT_FIXTURES:
        m_id = item["id"]
        team1 = item["team1"]
        team2 = item["team2"]

        played, score, winner = find_real_match_outcome(df, team1, team2, item["date"], shootout_winners)

        if not played and m_id in simulated_winners:
            played = True
            score = "Simulated"
            winner = simulated_winners[m_id]

        resolved[m_id] = {
            "id": m_id,
            "date": item["date"],
            "time": item["time"],
            "team1": team1,
            "team2": team2,
            "stage": item["stage"],
            "played": played,
            "score": score,
            "winner": winner
        }

    # 2. Resolve subsequent rounds
    for m_id, cfg in BRACKET_CONFIG.items():
        t1_src = cfg["t1_src"]
        t2_src = cfg["t2_src"]

        t1_winner = resolved.get(t1_src, {}).get("winner")
        t2_winner = resolved.get(t2_src, {}).get("winner")

        team1 = t1_winner if t1_winner else f"Winner Match {t1_src[1:]}"
        team2 = t2_winner if t2_winner else f"Winner Match {t2_src[1:]}"

        is_decided = bool(t1_winner and t2_winner)
        played = False
        score = None
        winner = None

        if is_decided:
            played, score, winner = find_real_match_outcome(df, team1, team2, cfg["date"], shootout_winners)

            if not played and m_id in simulated_winners:
                played = True
                score = "Simulated"
                winner = simulated_winners[m_id]

        resolved[m_id] = {
            "id": m_id,
            "date": cfg["date"],
            "time": cfg["time"],
            "team1": team1,
            "team2": team2,
            "stage": cfg["stage"],
            "played": played,
            "score": score,
            "winner": winner,
            "is_decided": is_decided
        }

    return list(resolved.values())


def load_all_fixtures(data_path: str | None = None, simulated_winners: dict | None = None) -> list[dict]:
    """Load all available World Cup 2026 fixtures (both group and dynamic knockout)."""
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "results.csv",
        )

    group = load_group_stage_fixtures(data_path)
    knockout = load_knockout_fixtures_dynamic(data_path, simulated_winners)
    fixtures = group + knockout
    fixtures.sort(key=lambda f: (f["date"], f["time"], f["team1"]))
    return fixtures


def get_fixture_dates(fixtures: list[dict]) -> list[str]:
    return sorted({f["date"] for f in fixtures})


def format_fixture_date(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%A, %b %d")
