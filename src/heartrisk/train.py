"""Train, tune and compare classifiers with MLflow tracking; export the champion model.

Usage:  python -m heartrisk.train [--data PATH] [--experiment NAME] [--fast]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402

from heartrisk import __version__  # noqa: E402
from heartrisk.data import CLEAN_PATH, FEATURES, ROOT, load_clean, split_xy  # noqa: E402
from heartrisk.features import build_pipeline, transformed_feature_names  # noqa: E402

SEED = 42
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def candidate_models(fast: bool = False) -> dict[str, tuple[object, dict]]:
    """Estimators and their hyper-parameter grids (keys prefixed for the pipeline step)."""
    grids = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 4, 8],
                "min_samples_leaf": [1, 4],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        ),
    }
    if fast:  # tiny grids for CI smoke runs
        grids = {
            name: (est, {k: v[:1] for k, v in grid.items()}) for name, (est, grid) in grids.items()
        }
    return {
        name: (est, {f"model__{k}": v for k, v in grid.items()})
        for name, (est, grid) in grids.items()
    }


def evaluate(model, X, y) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def feature_importance(pipeline) -> pd.Series:
    names = transformed_feature_names(pipeline)
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = np.abs(model.coef_[0])
    return pd.Series(values, index=names).sort_values(ascending=False)


def save_plots(name: str, model, X_test, y_test, out_dir: Path) -> list[Path]:
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC curve (test) - {name}")
    paths.append(out_dir / f"roc_{name}.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax=ax, cmap="Purples", display_labels=["No disease", "Disease"]
    )
    ax.set_title(f"Confusion matrix - {name}")
    paths.append(out_dir / f"confusion_{name}.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    imp = feature_importance(model).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(imp.index, imp.values, color="#6c5ce7")
    ax.set_title(f"Top features - {name}")
    paths.append(out_dir / f"importance_{name}.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)
    return paths


def log_grid_children(search: GridSearchCV, parent_name: str) -> None:
    """One nested MLflow run per hyper-parameter combination (every experiment is tracked)."""
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"{parent_name}-cand{i:02d}", nested=True):
            mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
            mlflow.log_metrics(
                {f"cv_{m}": float(res[f"mean_test_{m}"][i]) for m in SCORING}
                | {f"cv_{m}_std": float(res[f"std_test_{m}"][i]) for m in SCORING}
            )
            mlflow.set_tag("rank_by_roc_auc", int(res["rank_test_roc_auc"][i]))


def comparison_plot(summary: pd.DataFrame, path: Path) -> None:
    ax = summary[["cv_roc_auc", "test_roc_auc", "test_accuracy", "test_recall"]].plot.bar(
        figsize=(8, 4.5), rot=0, colormap="viridis"
    )
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    ax.figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(ax.figure)


def train(data_path: Path, experiment: str, fast: bool = False) -> dict:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri()))
    mlflow.set_experiment(experiment)
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean(data_path)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, (estimator, grid) in candidate_models(fast).items():
        with mlflow.start_run(run_name=name) as run:
            search = GridSearchCV(
                build_pipeline(estimator), grid, cv=cv, scoring=SCORING, refit="roc_auc", n_jobs=-1
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            i = search.best_index_
            cv_metrics = {
                f"cv_{m}": float(search.cv_results_[f"mean_test_{m}"][i]) for m in SCORING
            }
            cv_std = {f"cv_{m}_std": float(search.cv_results_[f"std_test_{m}"][i]) for m in SCORING}
            test_metrics = {
                f"test_{k}": float(v) for k, v in evaluate(best, X_test, y_test).items()
            }

            mlflow.set_tags({"model_family": name, "dataset": data_path.name, "fast_mode": fast})
            mlflow.log_params(
                {
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "cv_folds": cv.n_splits,
                    "grid_size": len(search.cv_results_["params"]),
                }
                | {k.replace("model__", "best_"): v for k, v in search.best_params_.items()}
            )
            mlflow.log_metrics(cv_metrics | cv_std | test_metrics)

            with tempfile.TemporaryDirectory() as tmp:
                for p in save_plots(name, best, X_test, y_test, FIGURES_DIR):
                    mlflow.log_artifact(str(p), artifact_path="plots")
                cv_csv = Path(tmp) / f"cv_results_{name}.csv"
                pd.DataFrame(search.cv_results_).to_csv(cv_csv, index=False)
                mlflow.log_artifact(str(cv_csv), artifact_path="cv")

            sample = X_train.head(5)
            mlflow.sklearn.log_model(
                best,
                artifact_path="model",
                code_paths=[str(ROOT / "src" / "heartrisk")],
                signature=infer_signature(sample, best.predict_proba(sample)),
                input_example=sample,
            )
            log_grid_children(search, name)

            results[name] = cv_metrics | test_metrics | {"run_id": run.info.run_id}
            fitted[name] = (best, search.best_params_)
            print(f"[{name}] cv_roc_auc={cv_metrics['cv_roc_auc']:.4f} "
                  f"test_roc_auc={test_metrics['test_roc_auc']:.4f}")  # fmt: skip

    summary = pd.DataFrame(results).T
    summary.to_csv(ROOT / "reports" / "model_comparison.csv")
    comparison_plot(
        summary.drop(columns="run_id").astype(float), FIGURES_DIR / "model_comparison.png"
    )

    # Champion = best mean cross-validated ROC-AUC (test set is only for reporting).
    champion = summary["cv_roc_auc"].astype(float).idxmax()
    model, params = fitted[champion]
    # Refit champion on all data so the served model uses every labelled example.
    final = model.fit(X, y)
    joblib.dump(final, MODELS_DIR / "model.joblib")

    metadata = {
        "model_name": champion,
        "model_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": FEATURES,
        "best_params": {k.replace("model__", ""): v for k, v in params.items()},
        "metrics": {k: v for k, v in results[champion].items() if k != "run_id"},
        "mlflow_run_id": results[champion]["run_id"],
        "n_samples": len(df),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    with mlflow.start_run(run_name=f"champion-{champion}"):
        mlflow.set_tags({"stage": "champion", "source_run": results[champion]["run_id"]})
        mlflow.log_metrics({k: float(v) for k, v in metadata["metrics"].items()})
        mlflow.log_artifact(str(MODELS_DIR / "metadata.json"))
        mlflow.log_artifact(str(FIGURES_DIR / "model_comparison.png"), artifact_path="plots")
        mlflow.sklearn.log_model(
            final,
            artifact_path="model",
            code_paths=[str(ROOT / "src" / "heartrisk")],
            signature=infer_signature(X.head(5), final.predict_proba(X.head(5))),
            input_example=X.head(5),
        )

    print(f"Champion: {champion}\n{summary.drop(columns='run_id').to_string()}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CLEAN_PATH)
    parser.add_argument("--experiment", default="heart-disease-risk")
    parser.add_argument("--fast", action="store_true", help="single-point grids (CI smoke run)")
    args = parser.parse_args()
    train(args.data, args.experiment, args.fast)


if __name__ == "__main__":
    main()
