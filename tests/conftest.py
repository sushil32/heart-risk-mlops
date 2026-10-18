"""Shared fixtures: a small synthetic dataset so tests are fast and need no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heartrisk.data import FEATURES, TARGET
from heartrisk.features import build_pipeline


def make_synthetic(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(30, 78, n).astype(float),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(1, 5, n),
            "trestbps": rng.normal(132, 17, n).clip(90, 200).round(),
            "chol": rng.normal(246, 50, n).clip(120, 560).round(),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.normal(150, 22, n).clip(70, 202).round(),
            "exang": rng.integers(0, 2, n),
            "oldpeak": rng.exponential(1.0, n).clip(0, 6).round(1),
            "slope": rng.integers(1, 4, n),
            "ca": rng.integers(0, 4, n).astype(float),
            "thal": rng.choice([3, 6, 7], n).astype(float),
        }
    )
    risk = 0.8 * (df["cp"] == 4) + 0.6 * df["exang"] + 0.5 * df["ca"] + 0.4 * df["oldpeak"]
    df[TARGET] = (risk + rng.normal(0, 0.5, n) > risk.median()).astype(int)
    # sprinkle a few missing values like the real data
    df.loc[[3, 17], "ca"] = np.nan
    df.loc[[5], "thal"] = np.nan
    return df[FEATURES + [TARGET]]


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture(scope="session")
def fitted_pipeline(synthetic_df):
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET]
    return build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)


@pytest.fixture()
def sample_record() -> dict:
    return {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
        "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
    }  # fmt: skip
