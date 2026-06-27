"""
train_model.py
Entry-point script: loads data, engineers features, trains models,
selects the best, and saves everything to disk.
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_data import download_dataset, download_shootouts
from src.feature_engineering import load_and_prepare_data, engineer_features
from src.model import train_and_evaluate, save_model


def main():
    print("=" * 60)
    print("  FIFA Match Outcome Predictor — Model Training Pipeline")
    print("=" * 60)

    # Step 1: Ensure data exists
    print("\n[1/4] Checking dataset...")
    data_path = download_dataset()
    download_shootouts()

    # Step 2: Load and prepare
    print("\n[2/4] Loading and preparing data...")
    df = load_and_prepare_data(data_path)
    print(f"  Loaded {len(df)} matches from {df['date'].min().year} to {df['date'].max().year}")
    print(f"  Teams: {df['home_team'].nunique()}")
    print(f"  Tournaments: {df['tournament'].nunique()}")

    # Step 3: Feature engineering
    print("\n[3/4] Engineering features...")
    df_features = engineer_features(df, verbose=True)

    # Step 4: Train and evaluate models
    print("\n[4/4] Training and evaluating models...")
    results = train_and_evaluate(df_features, verbose=True)

    # Save
    print("\nSaving best model...")
    save_model(results, verbose=True)

    print("\n" + "=" * 60)
    print("  Training complete! You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
