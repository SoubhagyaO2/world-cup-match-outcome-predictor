"""
download_data.py
Downloads the international football results and shootouts datasets from GitHub.
"""

import os
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
OUTPUT_PATH = os.path.join(DATA_DIR, "results.csv")

SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
SHOOTOUTS_PATH = os.path.join(DATA_DIR, "shootouts.csv")


def download_dataset(force: bool = False):
    """Download the international football results CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if not force and os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}")
        return OUTPUT_PATH

    print(f"Downloading dataset from {DATA_URL}...")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Dataset saved to {OUTPUT_PATH}")
    return OUTPUT_PATH


def download_shootouts(force: bool = False):
    """Download the penalty shootouts CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if not force and os.path.exists(SHOOTOUTS_PATH):
        print(f"Shootouts dataset already exists at {SHOOTOUTS_PATH}")
        return SHOOTOUTS_PATH

    print(f"Downloading shootouts dataset from {SHOOTOUTS_URL}...")
    response = requests.get(SHOOTOUTS_URL, timeout=60)
    response.raise_for_status()

    with open(SHOOTOUTS_PATH, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Shootouts dataset saved to {SHOOTOUTS_PATH}")
    return SHOOTOUTS_PATH


if __name__ == "__main__":
    download_dataset()
    download_shootouts()

