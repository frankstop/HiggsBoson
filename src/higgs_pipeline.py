#!/usr/bin/env python3
"""End-to-end Higgs Boson binary classification pipeline."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("work/matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


RANDOM_STATE = 42
TARGET_COLUMN = "Label"
DROP_COLUMNS = ["EventId", "Weight"]
MISSING_SENTINEL = -999


@dataclass
class ModelResult:
    name: str
    estimator: BaseEstimator
    precision: float
    recall: float
    f1: float
    roc_auc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Higgs classification models.")
    parser.add_argument("--data", type=Path, default=Path("data/training.csv"), help="Path to Kaggle training.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated outputs.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional row sample for faster experiments.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--demo", action="store_true", help="Run with a synthetic Higgs-like dataset.")
    parser.add_argument("--skip-grid", action="store_true", help="Skip hyperparameter tuning.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for estimators and searches.")
    return parser.parse_args()


def load_data(path: Path, demo: bool, sample_size: int | None) -> pd.DataFrame:
    if demo:
        demo_rows = sample_size or 3000
        features, target = make_classification(
            n_samples=demo_rows,
            n_features=30,
            n_informative=12,
            n_redundant=6,
            weights=[0.66, 0.34],
            class_sep=1.2,
            random_state=RANDOM_STATE,
        )
        columns = [f"DER_demo_feature_{i:02d}" for i in range(features.shape[1])]
        df = pd.DataFrame(features, columns=columns)
        df.insert(0, "EventId", np.arange(1, len(df) + 1))
        df["Weight"] = 1.0
        df[TARGET_COLUMN] = np.where(target == 1, "s", "b")
        rng = np.random.default_rng(RANDOM_STATE)
        missing_mask = rng.random(df[columns].shape) < 0.015
        df.loc[:, columns] = df[columns].mask(missing_mask, MISSING_SENTINEL)
        return df

    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {path}. Download Kaggle c/higgs-boson training.csv and place it at this path."
        )

    df = pd.read_csv(path)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def validate_and_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, Any]]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r}. Found columns: {list(df.columns)}")

    y = df[TARGET_COLUMN].map({"b": 0, "s": 1})
    if y.isna().any():
        values = sorted(df[TARGET_COLUMN].dropna().unique().tolist())
        raise ValueError(f"Target must contain Kaggle labels 'b' and 's'. Found: {values}")

    x = df.drop(columns=[TARGET_COLUMN])
    x = x.drop(columns=[col for col in DROP_COLUMNS if col in x.columns])

    categorical = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if categorical:
        raise ValueError(
            "The Higgs dataset is expected to have no categorical predictors. "
            f"Unexpected categorical columns: {categorical}"
        )

    x = x.replace(MISSING_SENTINEL, np.nan)
    feature_columns = x.columns.tolist()
    missing_counts = x.isna().sum().sort_values(ascending=False)
    target_counts = y.value_counts().sort_index()
    target_ratio = target_counts / target_counts.sum()
    imbalance_ratio = float(target_counts.min() / target_counts.max())

    eda = {
        "rows": int(len(df)),
        "features": int(len(feature_columns)),
        "missing_sentinel": MISSING_SENTINEL,
        "features_with_missing": int((missing_counts > 0).sum()),
        "total_missing_after_sentinel_conversion": int(missing_counts.sum()),
        "target_counts": {"background_0": int(target_counts.get(0, 0)), "signal_1": int(target_counts.get(1, 0))},
        "target_ratio": {"background_0": float(target_ratio.get(0, 0)), "signal_1": float(target_ratio.get(1, 0))},
        "imbalance_ratio": imbalance_ratio,
        "sampling_used": imbalance_ratio < 0.80,
    }
    return x, y.astype(int), feature_columns, eda


def make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = ImbPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_models(preprocessor: ColumnTransformer, use_sampling: bool, n_jobs: int) -> dict[str, BaseEstimator]:
    sampler_step = [("sampler", RandomOverSampler(random_state=RANDOM_STATE))] if use_sampling else []

    models: dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
        )
    else:
        raise ImportError("xgboost is required for this assignment. Install requirements.txt.")

    return {
        name: ImbPipeline(steps=[("preprocess", preprocessor), *sampler_step, ("model", model)])
        for name, model in models.items()
    }


def evaluate_model(name: str, estimator: BaseEstimator, x_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
    y_pred = estimator.predict(x_test)
    y_score = estimator.predict_proba(x_test)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return ModelResult(
        name=name,
        estimator=estimator,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc_score(y_test, y_score)),
    )


def run_baselines(
    models: dict[str, BaseEstimator],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> list[ModelResult]:
    results: list[ModelResult] = []
    fig, ax = plt.subplots(figsize=(7, 6))
    pr_fig, pr_ax = plt.subplots(figsize=(7, 6))

    for name, estimator in models.items():
        estimator.fit(x_train, y_train)
        result = evaluate_model(name, estimator, x_test, y_test)
        results.append(result)
        RocCurveDisplay.from_estimator(estimator, x_test, y_test, ax=ax, name=name)
        PrecisionRecallDisplay.from_estimator(estimator, x_test, y_test, ax=pr_ax, name=name)

    ax.set_title("ROC Curves on Hold-Out Test Set")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=160)
    plt.close(fig)

    pr_ax.set_title("Precision-Recall Curves on Hold-Out Test Set")
    pr_ax.grid(alpha=0.25)
    pr_fig.tight_layout()
    pr_fig.savefig(output_dir / "precision_recall_curves.png", dpi=160)
    plt.close(pr_fig)

    metrics = pd.DataFrame(
        [
            {
                "model": result.name,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1,
                "roc_auc": result.roc_auc,
            }
            for result in results
        ]
    ).sort_values("roc_auc", ascending=False)
    metrics.to_csv(output_dir / "metrics_baseline.csv", index=False)
    return sorted(results, key=lambda r: r.roc_auc, reverse=True)


def run_cross_validation(
    models: dict[str, BaseEstimator],
    x: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    output_dir: Path,
    n_jobs: int,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    for name, estimator in models.items():
        scores = cross_validate(estimator, x, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)

    cv_results = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    cv_results.to_csv(output_dir / "cross_validation.csv", index=False)
    return cv_results


def tune_best_model(
    best_name: str,
    best_estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
    output_dir: Path,
    n_jobs: int,
) -> GridSearchCV:
    if best_name == "Logistic Regression":
        param_grid = {
            "model__C": [0.1, 1.0, 3.0, 10.0],
            "model__class_weight": ["balanced", None],
        }
    elif best_name == "Random Forest":
        param_grid = {
            "model__n_estimators": [200, 350],
            "model__max_depth": [8, 14, None],
            "model__min_samples_leaf": [1, 3],
        }
    elif best_name == "XGBoost":
        param_grid = {
            "model__n_estimators": [200, 350],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.04, 0.08],
            "model__subsample": [0.85, 1.0],
        }
    else:
        raise ValueError(f"No tuning grid defined for {best_name}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=best_estimator,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(x_train, y_train)
    payload = {
        "selected_model": best_name,
        "scoring": "roc_auc",
        "best_score_cv_roc_auc": float(search.best_score_),
        "best_params": search.best_params_,
    }
    (output_dir / "best_model_grid_search.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return search


def compute_feature_importance(
    estimator: BaseEstimator,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str],
    output_dir: Path,
    n_jobs: int,
) -> pd.DataFrame:
    sample_size = min(4000, len(x_test))
    x_sample = x_test.sample(n=sample_size, random_state=RANDOM_STATE)
    y_sample = y_test.loc[x_sample.index]
    importance = permutation_importance(
        estimator,
        x_sample,
        y_sample,
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=n_jobs,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    feature_importance.to_csv(output_dir / "top_features.csv", index=False)

    top = feature_importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#2f5f8f")
    ax.set_title("Top Permutation Importances")
    ax.set_xlabel("Mean ROC-AUC Decrease")
    fig.tight_layout()
    fig.savefig(output_dir / "top_features.png", dpi=160)
    plt.close(fig)
    return feature_importance


def plot_target_distribution(y: pd.Series, output_dir: Path) -> None:
    labels = y.map({0: "Background", 1: "Signal"})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=labels, ax=ax, hue=labels, legend=False, palette=["#6b7280", "#2563eb"])
    ax.set_title("Target Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=160)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    eda: dict[str, Any],
    baseline_results: list[ModelResult],
    tuned_result: ModelResult | None,
    grid_search: GridSearchCV | None,
    feature_importance: pd.DataFrame,
    cv_results: pd.DataFrame,
) -> None:
    best_baseline = baseline_results[0]
    top3 = feature_importance.head(3)
    lines = [
        "# Higgs Boson Classification Summary",
        "",
        "## Dataset and preprocessing",
        "",
        f"The dataset contains {eda['rows']:,} rows and {eda['features']} model features.",
        f"The target is binary: background is encoded as 0 and Higgs signal is encoded as 1.",
        (
            f"Class counts were {eda['target_counts']['background_0']:,} background events and "
            f"{eda['target_counts']['signal_1']:,} signal events."
        ),
        (
            f"The minority-to-majority ratio was {eda['imbalance_ratio']:.3f}. "
            + (
                "Random oversampling was applied only inside the training folds."
                if eda["sampling_used"]
                else "No resampling was applied because the class balance was acceptable."
            )
        ),
        "",
        "The Kaggle Higgs data uses -999 as a sentinel for undefined kinematic quantities. "
        "Those values were converted to missing values and imputed with the median within each training split. "
        "Median imputation was selected because these variables are continuous and can contain skewed tails. "
        "Standard scaling was applied after imputation for every model so logistic regression received well-conditioned inputs.",
        "",
        "There were no categorical predictors, so no feature encoding was required.",
        "",
        "## Model comparison",
        "",
        "Three supervised models were trained: logistic regression, random forest, and XGBoost gradient boosting.",
        "",
        pd.DataFrame(
            [
                {
                    "model": result.name,
                    "precision": round(result.precision, 4),
                    "recall": round(result.recall, 4),
                    "f1": round(result.f1, 4),
                    "roc_auc": round(result.roc_auc, 4),
                }
                for result in baseline_results
            ]
        ).to_markdown(index=False),
        "",
        f"The strongest baseline model by ROC-AUC was {best_baseline.name} with ROC-AUC {best_baseline.roc_auc:.4f}.",
        "",
        "Cross-validation was used to check model stability:",
        "",
        cv_results.round(4).to_markdown(index=False),
        "",
        "## Hyperparameter tuning",
        "",
    ]

    if grid_search is None or tuned_result is None:
        lines.extend(
            [
                "Grid search was skipped for this run.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"The selected model for tuning was {grid_search.best_estimator_.steps[-1][1].__class__.__name__}.",
                f"Best cross-validated ROC-AUC during grid search: {grid_search.best_score_:.4f}.",
                f"Hold-out ROC-AUC after tuning: {tuned_result.roc_auc:.4f}.",
                f"Best parameters: `{json.dumps(grid_search.best_params_, sort_keys=True)}`.",
                "",
                "Tuning was assessed with ROC-AUC because the task is a signal-vs-background ranking problem and accuracy alone can hide poor signal recall.",
                "",
            ]
        )

    lines.extend(
        [
            "## Top features",
            "",
            "Permutation importance on the hold-out set identified the three strongest drivers:",
            "",
            top3.assign(
                importance_mean=top3["importance_mean"].round(5),
                importance_std=top3["importance_std"].round(5),
            ).to_markdown(index=False),
            "",
            "Higher permutation importance means shuffling that feature caused a larger drop in ROC-AUC, so the model depended on that variable more heavily for separating signal from background.",
            "",
            "## Generated files",
            "",
            "- `metrics_baseline.csv`",
            "- `cross_validation.csv`",
            "- `best_model_grid_search.json`",
            "- `top_features.csv`",
            "- `target_distribution.png`",
            "- `roc_curves.png`",
            "- `precision_recall_curves.png`",
            "- `top_features.png`",
        ]
    )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    df = load_data(args.data, args.demo, args.sample_size)
    x, y, feature_columns, eda = validate_and_split(df)
    plot_target_distribution(y, args.output_dir)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = make_preprocessor(feature_columns)
    models = make_models(preprocessor, use_sampling=eda["sampling_used"], n_jobs=args.n_jobs)

    baseline_results = run_baselines(models, x_train, x_test, y_train, y_test, args.output_dir)
    cv_results = run_cross_validation(models, x, y, args.cv_folds, args.output_dir, args.n_jobs)

    grid_search = None
    tuned_result = None
    best_estimator = baseline_results[0].estimator
    if not args.skip_grid:
        grid_search = tune_best_model(
            baseline_results[0].name,
            best_estimator,
            x_train,
            y_train,
            args.cv_folds,
            args.output_dir,
            args.n_jobs,
        )
        tuned_result = evaluate_model("Tuned " + baseline_results[0].name, grid_search.best_estimator_, x_test, y_test)
        pd.DataFrame(
            [
                {
                    "model": tuned_result.name,
                    "precision": tuned_result.precision,
                    "recall": tuned_result.recall,
                    "f1_score": tuned_result.f1,
                    "roc_auc": tuned_result.roc_auc,
                }
            ]
        ).to_csv(args.output_dir / "metrics_tuned.csv", index=False)
        best_estimator = grid_search.best_estimator_

    feature_importance = compute_feature_importance(
        best_estimator,
        x_test,
        y_test,
        feature_columns,
        args.output_dir,
        args.n_jobs,
    )
    write_summary(args.output_dir, eda, baseline_results, tuned_result, grid_search, feature_importance, cv_results)

    print(f"Completed Higgs pipeline. Outputs written to {args.output_dir}")
    print(classification_report(y_test, best_estimator.predict(x_test), target_names=["background", "signal"]))


if __name__ == "__main__":
    main()
