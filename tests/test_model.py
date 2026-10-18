import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from heartrisk.data import FEATURES, TARGET
from heartrisk.features import ClinicalFeatures, build_pipeline, transformed_feature_names
from heartrisk.train import candidate_models, evaluate, feature_importance


def test_clinical_features_values():
    X = pd.DataFrame({"age": [40.0, 60.0, 70.0], "thalach": [180.0, 120.0, np.nan]})
    out = ClinicalFeatures().fit_transform(X)
    assert out["hr_reserve_ratio"].iloc[0] == pytest.approx(1.0)
    assert out["hr_reserve_ratio"].iloc[1] == pytest.approx(0.75)
    assert np.isnan(out["hr_reserve_ratio"].iloc[2])
    assert out["age_band"].tolist() == ["lt45", "55to64", "65plus"]


def test_preprocessing_handles_missing_and_scales(fitted_pipeline, synthetic_df):
    Xt = fitted_pipeline[:-1].transform(synthetic_df[FEATURES])
    assert not np.isnan(Xt).any()
    names = transformed_feature_names(fitted_pipeline)
    assert Xt.shape[1] == len(names)
    num_idx = [i for i, n in enumerate(names) if n.startswith("num__")]
    assert np.allclose(Xt[:, num_idx].mean(axis=0), 0, atol=1e-6)


def test_unseen_category_does_not_crash(fitted_pipeline, sample_record):
    rec = dict(sample_record, restecg=5, thal=np.nan)
    proba = fitted_pipeline.predict_proba(pd.DataFrame([rec], columns=FEATURES))
    assert proba.shape == (1, 2)


def test_pipeline_learns_signal(synthetic_df):
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET]
    pipe = build_pipeline(RandomForestClassifier(n_estimators=50, random_state=0))
    auc = cross_val_score(pipe, X, y, cv=3, scoring="roc_auc").mean()
    assert auc > 0.7


def test_evaluate_returns_all_metrics(fitted_pipeline, synthetic_df):
    m = evaluate(fitted_pipeline, synthetic_df[FEATURES], synthetic_df[TARGET])
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= v <= 1 for v in m.values())


def test_feature_importance_aligned(fitted_pipeline):
    imp = feature_importance(fitted_pipeline)
    assert len(imp) == len(transformed_feature_names(fitted_pipeline))
    assert (imp >= 0).all()


def test_candidate_grids_target_model_step():
    models = candidate_models()
    assert {"logistic_regression", "random_forest", "gradient_boosting"} <= set(models)
    for _, grid in models.values():
        assert all(k.startswith("model__") for k in grid)
    for _, grid in candidate_models(fast=True).values():
        assert all(len(v) == 1 for v in grid.values())


def test_model_roundtrip_serialisation(tmp_path, fitted_pipeline, synthetic_df):
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_pipeline, path)
    loaded = joblib.load(path)
    X = synthetic_df[FEATURES]
    np.testing.assert_allclose(loaded.predict_proba(X), fitted_pipeline.predict_proba(X))
