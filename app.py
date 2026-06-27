"""
app.py
FIFA Match Outcome Predictor — Streamlit Frontend
FotMob-inspired dark theme with interactive Plotly visualizations.
"""

import os
import sys
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predictor import MatchPredictor
from src.world_cup_2026 import WORLD_CUP_TOURNAMENT, WORLD_CUP_LABEL
from download_data import download_dataset, download_shootouts

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — FotMob-inspired dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: linear-gradient(180deg, #0a0d12 0%, #0e1117 50%, #0a0d12 100%);
    }

    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero header */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00e676 0%, #00c853 50%, #69f0ae 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #8a8f98;
        font-size: 1.05rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e0e0e0;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #00e67633;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Cards */
    .stat-card {
        background: linear-gradient(145deg, #1a1d23 0%, #16191f 100%);
        border: 1px solid #2a2d35;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .stat-card:hover {
        border-color: #00e67644;
        box-shadow: 0 8px 30px rgba(0, 230, 118, 0.08);
        transform: translateY(-2px);
    }

    /* Prediction result card */
    .prediction-card {
        background: linear-gradient(145deg, #1a1d23 0%, #16191f 100%);
        border: 1px solid #00e67633;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 40px rgba(0, 230, 118, 0.1);
    }
    .prediction-winner {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00e676;
        margin: 0.5rem 0;
    }
    .prediction-confidence {
        font-size: 1rem;
        color: #8a8f98;
        font-weight: 500;
    }

    /* Probability bars */
    .prob-container {
        display: flex;
        align-items: center;
        margin: 0.6rem 0;
        gap: 0.8rem;
    }
    .prob-label {
        min-width: 100px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #e0e0e0;
    }
    .prob-bar-bg {
        flex-grow: 1;
        height: 32px;
        background: #2a2d35;
        border-radius: 16px;
        overflow: hidden;
        position: relative;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 12px;
        font-weight: 700;
        font-size: 0.85rem;
        color: #fff;
        min-width: 45px;
        transition: width 0.8s ease;
    }

    /* Form pills */
    .form-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.85rem;
        margin: 0 3px;
        color: #fff;
        transition: transform 0.2s;
    }
    .form-pill:hover {
        transform: scale(1.15);
    }
    .form-W { background: linear-gradient(135deg, #00c853, #00e676); }
    .form-D { background: linear-gradient(135deg, #ff9800, #ffb74d); }
    .form-L { background: linear-gradient(135deg, #f44336, #e57373); }

    /* Team name in form section */
    .team-form-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e0e0e0;
        margin-bottom: 0.5rem;
    }

    /* H2H summary metric */
    .h2h-metric {
        text-align: center;
        padding: 1rem;
    }
    .h2h-metric .value {
        font-size: 2rem;
        font-weight: 800;
        color: #00e676;
    }
    .h2h-metric .label {
        font-size: 0.85rem;
        color: #8a8f98;
        font-weight: 500;
        margin-top: 0.2rem;
    }

    /* Stat comparison table */
    .comparison-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
    }
    .comparison-table th {
        background: #1e222a;
        color: #8a8f98;
        padding: 0.8rem 1rem;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .comparison-table td {
        padding: 0.8rem 1rem;
        border-bottom: 1px solid #2a2d35;
        color: #e0e0e0;
        font-size: 0.95rem;
    }
    .comparison-table tr:last-child td {
        border-bottom: none;
    }
    .comparison-table .better {
        color: #00e676;
        font-weight: 700;
    }

    /* Footer */
    .model-footer {
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid #2a2d35;
        color: #555;
        font-size: 0.85rem;
    }

    /* Streamlit element overrides */
    .stSelectbox > div > div {
        background-color: #1a1d23 !important;
        border-color: #2a2d35 !important;
        border-radius: 12px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00c853, #00e676) !important;
        color: #0e1117 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.7rem 2rem !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(0, 230, 118, 0.35) !important;
        transform: translateY(-2px) !important;
    }

    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 16px;
        overflow: hidden;
    }

    /* Divider */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #2a2d35, transparent);
        margin: 2rem 0;
    }

    /* Fixtures */
    .fixtures-date-bar {
        text-align: center;
        font-size: 1.05rem;
        font-weight: 600;
        color: #e0e0e0;
        padding: 0.75rem 0;
    }
    .fixtures-day-label {
        color: #8a8f98;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
    }
    .fixture-meta {
        color: #8a8f98;
        font-size: 0.8rem;
        margin-top: 0.35rem;
    }
    div[data-testid="stButton"] > button.fixture-btn {
        background: linear-gradient(145deg, #1a1d23 0%, #16191f 100%) !important;
        color: #e0e0e0 !important;
        border: 1px solid #2a2d35 !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        text-align: left !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.5rem !important;
        box-shadow: none !important;
    }
    div[data-testid="stButton"] > button.fixture-btn:hover {
        border-color: #00e67655 !important;
        box-shadow: 0 4px 20px rgba(0, 230, 118, 0.08) !important;
        transform: none !important;
    }
    div[data-testid="stButton"] > button.fixture-btn-selected {
        border-color: #00e676 !important;
        background: linear-gradient(145deg, #1a2e22 0%, #162019 100%) !important;
    }
    div[data-testid="stButton"] > button.nav-btn {
        background: #1a1d23 !important;
        color: #e0e0e0 !important;
        border: 1px solid #2a2d35 !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        padding: 0.55rem 0.75rem !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load predictor (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results.csv")
    return MatchPredictor(data_path)


predictor = load_predictor()


# Initialize session state for simulations
if "simulated_winners" not in st.session_state:
    st.session_state["simulated_winners"] = {}

# Refresh fixtures with latest session simulated state
predictor.refresh_fixtures(st.session_state["simulated_winners"])


def sync_real_world_data():
    with st.spinner("Syncing latest match results and penalty shootouts..."):
        download_dataset(force=True)
        download_shootouts(force=True)
        # Clear the streamlit cache and reload the predictor
        st.cache_resource.clear()
        predictor.reload_data()
    st.success("Successfully synced latest data and refreshed predictor!")
    st.rerun()


# ---------------------------------------------------------------------------
# Plotly theme defaults
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e0e0e0"),
    margin=dict(l=40, r=40, t=40, b=40),
)

ACCENT_GREEN = "#00e676"
ACCENT_AMBER = "#ffb74d"
ACCENT_RED = "#e57373"
WORLD_CUP = WORLD_CUP_TOURNAMENT
DEFAULT_FIXTURE_DATE = "2026-06-28"


def _default_fixture_date_index(dates: list[str]) -> int:
    if DEFAULT_FIXTURE_DATE in dates:
        return dates.index(DEFAULT_FIXTURE_DATE)
    upcoming = [i for i, d in enumerate(dates) if d >= DEFAULT_FIXTURE_DATE]
    if upcoming:
        return upcoming[0]
    return len(dates) - 1 if dates else 0


# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">{WORLD_CUP_LABEL} Predictor</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tournament Settings & Sync Expander
# ---------------------------------------------------------------------------
with st.expander("Tournament Settings & Sync", expanded=False):
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("Sync Real-World Results", use_container_width=True, help="Downloads the latest results from GitHub and updates the ELO / bracket."):
            sync_real_world_data()
    with col_c2:
        if st.button("Reset Simulation", use_container_width=True, help="Clears all simulated progression and resets the bracket."):
            st.session_state["simulated_winners"] = {}
            predictor.refresh_fixtures(st.session_state["simulated_winners"])
            if "selected_fixture_id" in st.session_state:
                del st.session_state["selected_fixture_id"]
            if "prediction" in st.session_state:
                del st.session_state["prediction"]
            st.success("Simulation reset successfully!")
            st.rerun()
    st.caption(f"Active simulated matches: {len(st.session_state['simulated_winners'])}")


# ---------------------------------------------------------------------------
# Scheduled Fixtures
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header">Scheduled Matches</div>', unsafe_allow_html=True)
st.caption("Select a fixture to generate a match outcome prediction.")

# Stage selection filter
stages = ["All Stages", "Group Stage", "Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
selected_stage = st.selectbox("Filter by Tournament Stage", stages, index=0)

all_fixtures = predictor.get_fixtures()
if selected_stage != "All Stages":
    filtered_fixtures = [f for f in all_fixtures if f["stage"] == selected_stage]
else:
    filtered_fixtures = all_fixtures

fixture_dates = sorted({f["date"] for f in filtered_fixtures})

if not fixture_dates:
    st.warning("No fixtures found for the selected stage.")
else:
    # Ensure current date index is within bounds of the filtered dates
    if "current_stage" not in st.session_state or st.session_state.current_stage != selected_stage:
        st.session_state.current_stage = selected_stage
        st.session_state.fixture_date_idx = _default_fixture_date_index(fixture_dates)
    elif st.session_state.fixture_date_idx >= len(fixture_dates):
        st.session_state.fixture_date_idx = len(fixture_dates) - 1

    selected_date = fixture_dates[st.session_state.fixture_date_idx]

    nav_prev, nav_date, nav_next = st.columns([1, 4, 1])
    with nav_prev:
        if st.button("←", key="fixture_prev", use_container_width=True):
            st.session_state.fixture_date_idx = max(0, st.session_state.fixture_date_idx - 1)
            st.rerun()
    with nav_date:
        st.markdown(
            f'<div class="fixtures-date-bar">{predictor.format_fixture_date(selected_date)}</div>',
            unsafe_allow_html=True,
        )
    with nav_next:
        if st.button("→", key="fixture_next", use_container_width=True):
            st.session_state.fixture_date_idx = min(
                len(fixture_dates) - 1,
                st.session_state.fixture_date_idx + 1,
            )
            st.rerun()

    day_fixtures = [f for f in filtered_fixtures if f["date"] == selected_date]

    stage_label = day_fixtures[0]["stage"] if day_fixtures else "Fixtures"
    st.markdown(f'<div class="fixtures-day-label">{stage_label}</div>', unsafe_allow_html=True)

    for fixture in day_fixtures:
        # Check if teams are undecided (starts with "Winner Match")
        is_tbd = fixture["team1"].startswith("Winner Match") or fixture["team2"].startswith("Winner Match")
        button_disabled = is_tbd and not fixture["played"]

        score_suffix = ""
        if fixture.get("score"):
            if fixture["score"] == "Simulated":
                sim_winner = st.session_state["simulated_winners"].get(fixture["id"])
                if sim_winner:
                    score_suffix = f"  ·  Simulated Winner: {sim_winner}"
            else:
                score_suffix = f"  ·  {fixture['score']}"

        tbd_suffix = " 🔒 (TBD)" if button_disabled else ""
        button_label = f"{fixture['time']}   {fixture['team1']}  vs  {fixture['team2']}{score_suffix}{tbd_suffix}"
        is_selected = st.session_state.get("selected_fixture_id") == fixture["id"]
        if st.button(
            button_label,
            key=f"fixture_{fixture['id']}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
            disabled=button_disabled
        ):
            st.session_state["prediction"] = predictor.predict(fixture["team1"], fixture["team2"])
            st.session_state["pred_team_one"] = fixture["team1"]
            st.session_state["pred_team_two"] = fixture["team2"]
            st.session_state["selected_fixture_id"] = fixture["id"]
            st.session_state["selected_fixture"] = fixture
            st.rerun()


# ---------------------------------------------------------------------------
# Render results (if prediction exists)
# ---------------------------------------------------------------------------
if "prediction" in st.session_state:
    pred = st.session_state["prediction"]
    team1 = st.session_state["pred_team_one"]
    team2 = st.session_state["pred_team_two"]
    selected_fixture = st.session_state.get("selected_fixture")

    probs = pred["probabilities"]
    team1_prob = probs.get("Home Win", 0)
    draw_prob = probs.get("Draw", 0)
    team2_prob = probs.get("Away Win", 0)

    outcome_map = {
        "Home Win": f"{team1} Win",
        "Away Win": f"{team2} Win",
        "Draw": "Draw",
    }
    predicted_label = outcome_map.get(pred["predicted_outcome"], pred["predicted_outcome"])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if selected_fixture:
        st.markdown(
            f'<div class="fixtures-day-label">'
            f'{selected_fixture["date"]} · {selected_fixture["time"]} · '
            f'{team1} vs {team2}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Prediction Results
    # -----------------------------------------------------------------------
    st.markdown(f"""<div class="prediction-card">
<div class="prediction-winner">{predicted_label}</div>
<div class="prediction-confidence">Confidence: {pred['confidence']*100:.1f}%</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""<div class="stat-card">
<div class="prob-container">
<span class="prob-label">{team1}</span>
<div class="prob-bar-bg">
<div class="prob-bar-fill" style="width: {team1_prob*100:.1f}%; background: linear-gradient(90deg, #00c853, #00e676);">{team1_prob*100:.1f}%</div>
</div></div>
<div class="prob-container">
<span class="prob-label">Draw</span>
<div class="prob-bar-bg">
<div class="prob-bar-fill" style="width: {draw_prob*100:.1f}%; background: linear-gradient(90deg, #f57c00, #ffb74d);">{draw_prob*100:.1f}%</div>
</div></div>
<div class="prob-container">
<span class="prob-label">{team2}</span>
<div class="prob-bar-bg">
<div class="prob-bar-fill" style="width: {team2_prob*100:.1f}%; background: linear-gradient(90deg, #d32f2f, #e57373);">{team2_prob*100:.1f}%</div>
</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Tournament Simulation Advance Option
    is_knockout = selected_fixture.get("stage") != "Group Stage"
    is_real_completed = selected_fixture.get("played") and selected_fixture.get("score") != "Simulated"
    if is_knockout and not is_real_completed:
        st.markdown(f'<div class="section-header">Bracket Simulation</div>', unsafe_allow_html=True)
        current_winner = st.session_state["simulated_winners"].get(selected_fixture["id"])
        if current_winner:
            st.info(f"Currently advanced to the next round: **{current_winner}**")
        st.write("Advance one of these teams to progress the tournament bracket:")
        
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            is_pred_win = pred["predicted_outcome"] == "Home Win"
            btn_type = "primary" if is_pred_win else "secondary"
            if st.button(f"Advance {team1} ➔", key=f"adv_{team1}", type=btn_type, use_container_width=True):
                st.session_state["simulated_winners"][selected_fixture["id"]] = team1
                predictor.refresh_fixtures(st.session_state["simulated_winners"])
                st.success(f"Advanced {team1} successfully!")
                st.rerun()
        with col_adv2:
            is_pred_win = pred["predicted_outcome"] == "Away Win"
            btn_type = "primary" if is_pred_win else "secondary"
            if st.button(f"Advance {team2} ➔", key=f"adv_{team2}", type=btn_type, use_container_width=True):
                st.session_state["simulated_winners"][selected_fixture["id"]] = team2
                predictor.refresh_fixtures(st.session_state["simulated_winners"])
                st.success(f"Advanced {team2} successfully!")
                st.rerun()
                
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Recent Form
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-header">Recent Form</div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)

    for col, team in [(col_f1, team1), (col_f2, team2)]:
        with col:
            form = predictor.get_form_string(team, 5)
            form_pills = "".join(f'<span class="form-pill form-{r}">{r}</span>' for r in form)

            summary = predictor.get_form_summary(team, 5)
            recent = predictor.get_recent_form(team, 5)
            form_detail = "".join(
                f'<div style="display:flex; justify-content:space-between; padding:0.4rem 0; border-bottom: 1px solid #2a2d35; font-size:0.88rem; color:#b0b3b8;">'
                f'<span>{m["date"]}</span><span>vs {m["opponent"]}</span>'
                f'<span style="font-weight:600;">{m["goals_for"]}\u2013{m["goals_against"]}</span>'
                f'<span class="form-pill form-{m["result"]}" style="width:28px;height:28px;font-size:0.75rem;">{m["result"]}</span></div>'
                for m in recent
            )

            st.markdown(f"""<div class="stat-card">
<div class="team-form-name">{team}</div>
<div style="margin-bottom:1rem;">{form_pills}</div>
<div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:0.5rem; margin-bottom:1rem; text-align:center;">
<div><div style="font-size:1.4rem; font-weight:700; color:#00e676;">{summary['wins']}</div><div style="font-size:0.75rem; color:#8a8f98;">Wins</div></div>
<div><div style="font-size:1.4rem; font-weight:700; color:#ffb74d;">{summary['draws']}</div><div style="font-size:0.75rem; color:#8a8f98;">Draws</div></div>
<div><div style="font-size:1.4rem; font-weight:700; color:#e57373;">{summary['losses']}</div><div style="font-size:0.75rem; color:#8a8f98;">Losses</div></div>
<div><div style="font-size:1.4rem; font-weight:700; color:#69f0ae;">{summary['form_score']*100:.0f}%</div><div style="font-size:0.75rem; color:#8a8f98;">Form Score</div></div>
</div>
{form_detail}
</div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 3. Head-to-Head Analysis
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-header">Head-to-Head</div>', unsafe_allow_html=True)

    h2h = predictor.get_head_to_head(team1, team2, 5)

    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    metrics = [
        (col_h1, "Total Meetings", h2h["total"]),
        (col_h2, f"{team1} Wins", h2h["team_a_wins"]),
        (col_h3, "Draws", h2h["draws"]),
        (col_h4, f"{team2} Wins", h2h["team_b_wins"]),
    ]
    for col, label, value in metrics:
        with col:
            st.markdown(f"""<div class="stat-card">
<div class="h2h-metric">
<div class="value">{value}</div>
<div class="label">{label}</div>
</div></div>""", unsafe_allow_html=True)

    # Last meetings table
    if h2h["meetings"]:
        h2h_rows = "".join(
            f'<tr><td>{m["date"]}</td><td style="font-weight:600;">{m["team_one"]}</td>'
            f'<td style="font-weight:700; color: #00e676;">{m["team_one_goals"]} – {m["team_two_goals"]}</td>'
            f'<td style="font-weight:600;">{m["team_two"]}</td></tr>'
            for m in h2h["meetings"]
        )
        st.markdown(f"""<div class="stat-card">
<table class="comparison-table">
<thead><tr><th>Date</th><th>Team 1</th><th>Score</th><th>Team 2</th></tr></thead>
<tbody>{h2h_rows}</tbody>
</table></div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 4. Team Statistics Comparison
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-header">Team Statistics</div>', unsafe_allow_html=True)

    team1_stats = predictor.get_team_stats(team1, WORLD_CUP)
    team2_stats = predictor.get_team_stats(team2, WORLD_CUP)

    rows_data = [
        ("Win Rate", team1_stats["win_rate"], team2_stats["win_rate"], True, True),
        ("Avg Goals Scored", team1_stats["avg_goals_scored"], team2_stats["avg_goals_scored"], True, False),
        ("Avg Goals Conceded", team1_stats["avg_goals_conceded"], team2_stats["avg_goals_conceded"], False, False),
        ("Goal Difference", team1_stats["goal_difference"], team2_stats["goal_difference"], True, False),
        ("Elo Rating", team1_stats.get("elo_rating", 1500), team2_stats.get("elo_rating", 1500), True, False),
        ("World Cup Matches", team1_stats["matches_played"], team2_stats["matches_played"], True, False),
    ]

    table_rows = ""
    for label, t1_val, t2_val, higher_better, is_pct in rows_data:
        if higher_better:
            t1_cls = "better" if t1_val > t2_val else ""
            t2_cls = "better" if t2_val > t1_val else ""
        else:
            t1_cls = "better" if t1_val < t2_val else ""
            t2_cls = "better" if t2_val < t1_val else ""

        if is_pct:
            t1_display = f"{t1_val*100:.1f}%"
            t2_display = f"{t2_val*100:.1f}%"
        elif isinstance(t1_val, float):
            t1_display = f"{t1_val:.2f}"
            t2_display = f"{t2_val:.2f}"
        else:
            t1_display = str(int(t1_val))
            t2_display = str(int(t2_val))

        t1_class_attr = f' class="{t1_cls}"' if t1_cls else ""
        t2_class_attr = f' class="{t2_cls}"' if t2_cls else ""
        table_rows += (
            f'<tr><td{t1_class_attr}>{t1_display}</td>'
            f'<td style="text-align:center; color:#8a8f98; font-weight:600;">{label}</td>'
            f'<td style="text-align:right;"{t2_class_attr}>{t2_display}</td></tr>'
        )

    st.markdown(
        f'<div class="stat-card"><table class="comparison-table">'
        f'<thead><tr><th>{team1}</th><th style="text-align:center;">Metric</th>'
        f'<th style="text-align:right;">{team2}</th></tr></thead>'
        f"<tbody>{table_rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 5. Win Probability Visualization
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-header">Win Probability</div>', unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        fig_bar = go.Figure()
        categories = [team2, "Draw", team1]
        values = [team2_prob * 100, draw_prob * 100, team1_prob * 100]
        colors = [ACCENT_RED, ACCENT_AMBER, ACCENT_GREEN]

        fig_bar.add_trace(go.Bar(
            y=categories,
            x=values,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
                cornerradius=8,
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
            textfont=dict(size=14, color="#fff", family="Inter"),
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            height=280,
            showlegend=False,
            xaxis=dict(
                range=[0, 100],
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(size=14, color="#e0e0e0"),
            ),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_p2:
        # Gauge chart for predicted winner
        winner_prob = pred["confidence"] * 100
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=winner_prob,
            number=dict(suffix="%", font=dict(size=42, color=ACCENT_GREEN)),
            title=dict(text="Prediction Confidence", font=dict(size=16, color="#8a8f98")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#2a2d35"),
                bar=dict(color=ACCENT_GREEN, thickness=0.3),
                bgcolor="#1a1d23",
                borderwidth=0,
                steps=[
                    dict(range=[0, 33], color="rgba(42, 45, 53, 0.4)"),
                    dict(range=[33, 66], color="rgba(42, 45, 53, 0.53)"),
                    dict(range=[66, 100], color="rgba(42, 45, 53, 0.67)"),
                ],
                threshold=dict(
                    line=dict(color=ACCENT_GREEN, width=4),
                    thickness=0.75,
                    value=winner_prob,
                ),
            ),
        ))
        fig_gauge.update_layout(
            **PLOTLY_LAYOUT,
            height=280,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 6. Historical Performance Charts
    # -----------------------------------------------------------------------
    st.markdown('<div class="section-header">Historical Performance</div>', unsafe_allow_html=True)

    team1_trend = predictor.get_performance_trend(team1, WORLD_CUP)
    team2_trend = predictor.get_performance_trend(team2, WORLD_CUP)

    team1_trend = team1_trend[team1_trend["year"] >= 1950]
    team2_trend = team2_trend[team2_trend["year"] >= 1950]

    col_t1, col_t2 = st.columns(2)

    # Win Rate Trend
    with col_t1:
        fig_wr = go.Figure()
        if not team1_trend.empty:
            fig_wr.add_trace(go.Scatter(
                x=team1_trend["year"], y=team1_trend["win_rate"] * 100,
                mode="lines", name=team1,
                line=dict(color=ACCENT_GREEN, width=2.5),
                fill="tozeroy", fillcolor="rgba(0, 230, 118, 0.08)",
            ))
        if not team2_trend.empty:
            fig_wr.add_trace(go.Scatter(
                x=team2_trend["year"], y=team2_trend["win_rate"] * 100,
                mode="lines", name=team2,
                line=dict(color=ACCENT_RED, width=2.5),
                fill="tozeroy", fillcolor="rgba(229, 115, 115, 0.08)",
            ))
        fig_wr.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Win Rate (%)", font=dict(size=14)),
            height=350,
            legend=dict(orientation="h", y=-0.15),
            yaxis=dict(title="", gridcolor="#2a2d35", range=[0, 100]),
            xaxis=dict(title="", gridcolor="#2a2d35"),
        )
        st.plotly_chart(fig_wr, use_container_width=True)

    # Goals Scored Trend
    with col_t2:
        fig_gs = go.Figure()
        if not team1_trend.empty:
            fig_gs.add_trace(go.Scatter(
                x=team1_trend["year"], y=team1_trend["avg_goals_scored"],
                mode="lines", name=team1,
                line=dict(color=ACCENT_GREEN, width=2.5),
            ))
        if not team2_trend.empty:
            fig_gs.add_trace(go.Scatter(
                x=team2_trend["year"], y=team2_trend["avg_goals_scored"],
                mode="lines", name=team2,
                line=dict(color=ACCENT_RED, width=2.5),
            ))
        fig_gs.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Avg Goals Scored", font=dict(size=14)),
            height=350,
            legend=dict(orientation="h", y=-0.15),
            yaxis=dict(title="", gridcolor="#2a2d35"),
            xaxis=dict(title="", gridcolor="#2a2d35"),
        )
        st.plotly_chart(fig_gs, use_container_width=True)

    # Goals Conceded Trend (full width)
    fig_gc = go.Figure()
    if not team1_trend.empty:
        fig_gc.add_trace(go.Scatter(
            x=team1_trend["year"], y=team1_trend["avg_goals_conceded"],
            mode="lines", name=team1,
            line=dict(color=ACCENT_GREEN, width=2.5),
        ))
    if not team2_trend.empty:
        fig_gc.add_trace(go.Scatter(
            x=team2_trend["year"], y=team2_trend["avg_goals_conceded"],
            mode="lines", name=team2,
            line=dict(color=ACCENT_RED, width=2.5),
        ))
    fig_gc.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Avg Goals Conceded", font=dict(size=14)),
        height=300,
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(title="", gridcolor="#2a2d35"),
        xaxis=dict(title="", gridcolor="#2a2d35"),
    )
    st.plotly_chart(fig_gc, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Model Info Footer
    # -----------------------------------------------------------------------
    model_info = predictor.get_model_info()



    st.markdown(f"""<div class="model-footer">
<p>Model: <strong>{model_info['model_name']}</strong> | Accuracy: <strong>{model_info['accuracy']*100:.1f}%</strong> | Trained: <strong>{model_info['training_date'][:10]}</strong></p>
<p style="color:#444; font-size:0.8rem;">{WORLD_CUP_LABEL} • International Football Results</p>
</div>""", unsafe_allow_html=True)
