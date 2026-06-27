# ⚽ FIFA Match Outcome Predictor

ML-powered international football match outcome prediction with a FotMob-inspired dark-theme UI.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)

## Features

- **Match Outcome Prediction** — Home Win / Draw / Away Win probabilities
- **Recent Form Analysis** — Last 5 match results with W/D/L indicators
- **Head-to-Head Records** — Historical meetings between selected teams
- **Team Statistics Comparison** — Win rate, goals scored/conceded, Elo ratings
- **Win Probability Visualization** — Interactive bar charts and confidence gauge
- **Historical Performance Trends** — Multi-decade win rate and goal trends
- **Feature Importance** — Model explainability via feature contribution chart
- **Model Comparison** — Logistic Regression vs Random Forest vs Gradient Boosting

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| ML | Python, Pandas, NumPy, scikit-learn |
| Frontend | Streamlit |
| Visualization | Plotly |
| Model Storage | Joblib |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

This will:
- Download the dataset (~49K international matches, 1872–2026)
- Engineer 30+ features (Elo, form, H2H, team strength)
- Train and compare 3 ML models
- Save the best model to `models/`

### 3. Launch the app

```bash
streamlit run app.py
```

## Project Structure

```
fifa-match-outcome-predictor/
├── .streamlit/config.toml       # Dark theme config
├── data/results.csv             # Dataset (auto-downloaded)
├── models/
│   ├── best_model.joblib        # Trained model
│   ├── label_encoder.joblib     # Label encoder
│   └── model_metadata.json      # Metrics & feature importance
├── src/
│   ├── feature_engineering.py   # Feature computation pipeline
│   ├── model.py                 # Model training & evaluation
│   └── predictor.py             # Prediction & analytics API
├── app.py                       # Streamlit frontend
├── download_data.py             # Dataset downloader
├── train_model.py               # Training entry point
└── requirements.txt
```

## Feature Engineering

| Group | Features |
|-------|----------|
| Elo Rating | Home/Away Elo, Elo difference |
| Team Strength | Win/Draw/Loss rate, matches played |
| Scoring | Avg goals scored/conceded, goal difference |
| Recent Form | Last-5 win rate, goals, form score |
| Head-to-Head | Meetings, win/draw percentage |
| Match Context | Tournament type, neutral venue |

## Dataset

[International Football Results (1872–2026)](https://github.com/martj42/international_results) — 49,000+ men's international matches.

## License

MIT
