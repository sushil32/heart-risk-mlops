"""Loading and cleaning of the UCI Heart Disease (Cleveland) dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_clean.csv"
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]  # fmt: skip
TARGET = "target"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
BINARY = ["sex", "fbs", "exang"]
FEATURES = NUMERIC + CATEGORICAL + BINARY

# Physiologically plausible ranges; values outside are treated as missing.
VALID_RANGES = {
    "age": (18, 100),
    "trestbps": (70, 250),
    "chol": (100, 650),
    "thalach": (60, 230),
    "oldpeak": (0, 7),
    "ca": (0, 3),
}
ALLOWED_CODES = {
    "sex": {0, 1},
    "fbs": {0, 1},
    "exang": {0, 1},
    "cp": {1, 2, 3, 4},
    "restecg": {0, 1, 2},
    "slope": {1, 2, 3},
    "thal": {3, 6, 7},
}


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw UCI file (no header, '?' marks missing values)."""
    return pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: numeric types, invalid values -> NaN, binary target, no duplicates.

    Missing values are intentionally kept as NaN here and imputed inside the model pipeline,
    so imputation statistics are learned only from training folds (no leakage).
    """
    out = df.copy()
    for col in RAW_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col, (lo, hi) in VALID_RANGES.items():
        out.loc[~out[col].between(lo, hi), col] = np.nan
    for col, codes in ALLOWED_CODES.items():
        out.loc[~out[col].isin(codes), col] = np.nan

    out = out.dropna(subset=["num"])
    out[TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns="num").drop_duplicates().reset_index(drop=True)
    return out[FEATURES + [TARGET]]


def load_clean(path: str | Path = CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET]
