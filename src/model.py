"""
model.py
Trains and evaluates multiple ML models for match outcome prediction.
Selects the best model by macro F1 score and saves it via joblib.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

from src.feature_engineering import get_feature_columns

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _get_models() -> dict:
    """Return dict of model name -> model instance."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_split=10, random_state=42
        ),
    }


def train_and_evaluate(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Train multiple models, evaluate them, and return results.

    Returns dict with keys: best_model_name, models, metrics, feature_importance,
    label_encoder, feature_columns
    """
    feature_cols = get_feature_columns()

    # Filter to matches where teams have at least some history
    mask = (df["home_matches_played"] >= 5) & (df["away_matches_played"] >= 5)
    ml_df = df[mask].copy()

    if verbose:
        print(f"Training on {len(ml_df)} matches (filtered from {len(df)} total)")

    X = ml_df[feature_cols].fillna(0).values
    y = ml_df["result"].values

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    if verbose:
        print(f"Classes: {le.classes_}")
        print(f"Class distribution: {np.bincount(y_encoded)}")

    # Train/test split (chronological — use last 20% for testing)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]

    if verbose:
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Train models
    models = _get_models()
    results = {}
    best_f1 = -1
    best_name = None

    for name, model in models.items():
        if verbose:
            print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
        }

        if verbose:
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    if verbose:
        print(f"\n{'='*50}")
        print(f"Best model: {best_name} (F1: {best_f1:.4f})")

    # Feature importance for the best model
    best_model = results[best_name]["model"]
    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importance = np.mean(np.abs(best_model.coef_), axis=0)
    elif hasattr(best_model, "named_steps"):
        clf = best_model.named_steps.get("clf")
        if clf is not None and hasattr(clf, "coef_"):
            importance = np.mean(np.abs(clf.coef_), axis=0)
        else:
            importance = np.zeros(len(feature_cols))
    else:
        importance = np.zeros(len(feature_cols))

    feature_importance = dict(zip(feature_cols, importance.tolist()))

    return {
        "best_model_name": best_name,
        "models": results,
        "feature_importance": feature_importance,
        "label_encoder": le,
        "feature_columns": feature_cols,
        "best_model": best_model,
    }


def save_model(train_results: dict, verbose: bool = True):
    """Save the best model, label encoder, and metadata to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")

    joblib.dump(train_results["best_model"], model_path)
    joblib.dump(train_results["label_encoder"], encoder_path)

    # Build metrics summary for all models
    metrics_summary = {}
    for name, data in train_results["models"].items():
        metrics_summary[name] = {
            "accuracy": data["accuracy"],
            "precision": data["precision"],
            "recall": data["recall"],
            "f1_score": data["f1_score"],
            "confusion_matrix": data["confusion_matrix"],
        }

    metadata = {
        "best_model_name": train_results["best_model_name"],
        "feature_columns": train_results["feature_columns"],
        "feature_importance": train_results["feature_importance"],
        "classes": train_results["label_encoder"].classes_.tolist(),
        "all_model_metrics": metrics_summary,
        "training_date": pd.Timestamp.now().isoformat(),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"Model saved to {model_path}")
        print(f"Encoder saved to {encoder_path}")
        print(f"Metadata saved to {metadata_path}")


def load_model() -> tuple:
    """Load the saved model, label encoder, and metadata."""
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")

    model = joblib.load(model_path)
    le = joblib.load(encoder_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return model, le, metadata
