"""Feature engineering and the reusable preprocessing + model pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heartrisk.data import BINARY, CATEGORICAL, NUMERIC

AGE_BINS = [0, 45, 55, 65, 200]
AGE_LABELS = ["lt45", "45to54", "55to64", "65plus"]


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Adds domain-derived features.

    * hr_reserve_ratio: max heart rate achieved / age-predicted max (220 - age).
      Low values indicate chronotropic incompetence, a known cardiac risk marker.
    * age_band: coarse age group, lets linear models capture non-linear age risk.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X["hr_reserve_ratio"] = X["thalach"] / (220.0 - X["age"])
        X["age_band"] = (
            pd.cut(X["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
            .astype(object)
            .fillna("unknown")
        )
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.array(base + ["hr_reserve_ratio", "age_band"], dtype=object)


ENGINEERED_NUMERIC = ["hr_reserve_ratio"]
ENGINEERED_CATEGORICAL = ["age_band"]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    binary = SimpleImputer(strategy="most_frequent")
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC + ENGINEERED_NUMERIC),
            ("cat", categorical, CATEGORICAL + ENGINEERED_CATEGORICAL),
            ("bin", binary, BINARY),
        ],
        verbose_feature_names_out=True,
    )


def build_pipeline(estimator) -> Pipeline:
    """Full pipeline: raw patient columns in -> prediction out. This is what gets saved."""
    return Pipeline(
        [
            ("features", ClinicalFeatures()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())
