from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


FEATURE_COLUMNS = (
    "acceleration_ms2",
    "deceleration_ms2",
    "speed_limit_kmh",
    "coasting_point_m",
)
TARGET_COLUMNS = ("energy_kwh", "travel_time_s")
STATUS_COLUMNS = ("input_valid", "simulation_completed", "feasible")
REQUIRED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS + STATUS_COLUMNS

DEFAULT_DATASET_PATH = Path("results/train_experiments.csv")
DEFAULT_RESULTS_DIRECTORY = Path("results/ml")
DEFAULT_MODELS_DIRECTORY = Path("models")
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5

MODEL_DISPLAY_NAMES = {
    "linear_regression": "Linear Regression",
    "polynomial_ridge": "Polynomial Ridge Regression",
    "random_forest": "Random Forest",
}


def _parse_boolean(value: str, column: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(
        f"Row {row_number}: {column} must be either True or False."
    )


def load_dataset(
    dataset_path: Path | str,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Load completed, physically feasible rows from the frozen CSV dataset."""

    path = Path(dataset_path)
    features: list[list[float]] = []
    targets = {target: [] for target in TARGET_COLUMNS}
    source_rows: list[int] = []

    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = set(reader.fieldnames or ())
        missing = [column for column in REQUIRED_COLUMNS if column not in headers]
        if missing:
            raise ValueError(
                "Dataset is missing required columns: " + ", ".join(missing)
            )

        seen_combinations: set[tuple[float, ...]] = set()
        for row_number, row in enumerate(reader, start=2):
            statuses = {
                column: _parse_boolean(row[column], column, row_number)
                for column in STATUS_COLUMNS
            }
            if not all(statuses.values()):
                continue

            try:
                feature_values = tuple(float(row[column]) for column in FEATURE_COLUMNS)
                target_values = {
                    target: float(row[target]) for target in TARGET_COLUMNS
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Row {row_number}: feature and target values must be numeric."
                ) from error

            numeric_values = feature_values + tuple(target_values.values())
            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError(f"Row {row_number}: numeric values must be finite.")
            if feature_values in seen_combinations:
                raise ValueError(
                    f"Row {row_number}: duplicate parameter combination {feature_values}."
                )

            seen_combinations.add(feature_values)
            features.append(list(feature_values))
            for target, value in target_values.items():
                targets[target].append(value)
            source_rows.append(row_number)

    if len(features) < 10:
        raise ValueError("At least 10 usable dataset rows are required for training.")

    return (
        np.asarray(features, dtype=float),
        {
            target: np.asarray(values, dtype=float)
            for target, values in targets.items()
        },
        np.asarray(source_rows, dtype=int),
    )


def build_models(
    *, random_state: int = DEFAULT_RANDOM_STATE, random_forest_estimators: int = 300
) -> dict[str, Any]:
    """Build exactly the three approved regression model families."""

    if random_forest_estimators <= 0:
        raise ValueError("random_forest_estimators must be positive.")
    return {
        "linear_regression": LinearRegression(),
        "polynomial_ridge": Pipeline(
            steps=(
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            )
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=random_forest_estimators,
            random_state=random_state,
            n_jobs=1,
        ),
    }


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def _cross_validate(
    model: Any,
    features: np.ndarray,
    target: np.ndarray,
    *,
    cv_folds: int,
    random_state: int,
) -> dict[str, float]:
    if cv_folds < 2 or cv_folds > len(features):
        raise ValueError("cv_folds must be between 2 and the training row count.")

    fold_metrics = {metric: [] for metric in ("mae", "rmse", "r2")}
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for train_indices, validation_indices in splitter.split(features):
        fitted_model = clone(model).fit(
            features[train_indices], target[train_indices]
        )
        predicted = fitted_model.predict(features[validation_indices])
        metrics = _metrics(target[validation_indices], predicted)
        for metric, value in metrics.items():
            fold_metrics[metric].append(value)

    summary: dict[str, float] = {}
    for metric, values in fold_metrics.items():
        summary[f"cv_{metric}_mean"] = float(np.mean(values))
        summary[f"cv_{metric}_std"] = float(np.std(values, ddof=0))
    return summary


def _write_csv(path: Path, fieldnames: Sequence[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def train_and_evaluate(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    results_directory: Path | str = DEFAULT_RESULTS_DIRECTORY,
    models_directory: Path | str = DEFAULT_MODELS_DIRECTORY,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    random_forest_estimators: int = 300,
) -> dict[str, Any]:
    """Evaluate approved models, then refit and save one winner per target."""

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1.")

    dataset_path = Path(dataset_path)
    results_directory = Path(results_directory)
    models_directory = Path(models_directory)
    features, targets, source_rows = load_dataset(dataset_path)

    all_indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    models = build_models(
        random_state=random_state,
        random_forest_estimators=random_forest_estimators,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_models: dict[str, str] = {}

    for target_name in TARGET_COLUMNS:
        target = targets[target_name]
        target_metric_rows: list[dict[str, Any]] = []

        for model_name, model in models.items():
            cross_validation = _cross_validate(
                model,
                features[train_indices],
                target[train_indices],
                cv_folds=cv_folds,
                random_state=random_state,
            )
            fitted = clone(model).fit(features[train_indices], target[train_indices])
            predicted = fitted.predict(features[test_indices])
            test_metrics = _metrics(target[test_indices], predicted)

            metric_row = {
                "target": target_name,
                "model": model_name,
                **cross_validation,
                **{f"test_{name}": value for name, value in test_metrics.items()},
                "selected": False,
            }
            target_metric_rows.append(metric_row)

            for dataset_index, prediction in zip(test_indices, predicted):
                prediction_rows.append(
                    {
                        "target": target_name,
                        "model": model_name,
                        "source_csv_row": int(source_rows[dataset_index]),
                        **{
                            column: float(features[dataset_index, column_index])
                            for column_index, column in enumerate(FEATURE_COLUMNS)
                        },
                        "actual": float(target[dataset_index]),
                        "predicted": float(prediction),
                        "error": float(prediction - target[dataset_index]),
                    }
                )

        winner = min(
            target_metric_rows,
            key=lambda row: (row["cv_rmse_mean"], row["cv_mae_mean"]),
        )
        winner["selected"] = True
        selected_models[target_name] = winner["model"]
        metric_rows.extend(target_metric_rows)

        final_model = clone(models[winner["model"]]).fit(features, target)
        model_path = models_directory / f"{target_name.removesuffix('_kwh').removesuffix('_s')}_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, model_path)

        metadata = {
            "target": target_name,
            "selected_model": winner["model"],
            "selected_model_display_name": MODEL_DISPLAY_NAMES[winner["model"]],
            "feature_columns": list(FEATURE_COLUMNS),
            "training_rows": len(features),
            "feature_bounds": {
                column: {
                    "minimum": float(features[:, index].min()),
                    "maximum": float(features[:, index].max()),
                }
                for index, column in enumerate(FEATURE_COLUMNS)
            },
            "selection_rule": "lowest cross-validation RMSE, then MAE",
            "random_state": random_state,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        }
        metadata_path = model_path.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    metric_fields = (
        "target",
        "model",
        "cv_mae_mean",
        "cv_mae_std",
        "cv_rmse_mean",
        "cv_rmse_std",
        "cv_r2_mean",
        "cv_r2_std",
        "test_mae",
        "test_rmse",
        "test_r2",
        "selected",
    )
    prediction_fields = (
        "target",
        "model",
        "source_csv_row",
        *FEATURE_COLUMNS,
        "actual",
        "predicted",
        "error",
    )
    _write_csv(results_directory / "metrics.csv", metric_fields, metric_rows)
    _write_csv(
        results_directory / "test_predictions.csv",
        prediction_fields,
        prediction_rows,
    )

    split = {
        "random_state": random_state,
        "test_size": test_size,
        "training_rows": [int(source_rows[index]) for index in train_indices],
        "test_rows": [int(source_rows[index]) for index in test_indices],
    }
    results_directory.mkdir(parents=True, exist_ok=True)
    (results_directory / "split_indices.json").write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "dataset_rows": len(features),
        "training_rows": len(train_indices),
        "test_rows": len(test_indices),
        "selected_models": selected_models,
        "metrics_path": str((results_directory / "metrics.csv").resolve()),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate surrogate models on the frozen dataset."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results-directory", type=Path, default=DEFAULT_RESULTS_DIRECTORY)
    parser.add_argument("--models-directory", type=Path, default=DEFAULT_MODELS_DIRECTORY)
    return parser


def main() -> None:
    arguments = _build_argument_parser().parse_args()
    summary = train_and_evaluate(
        arguments.dataset,
        arguments.results_directory,
        arguments.models_directory,
    )
    print(f"Dataset rows: {summary['dataset_rows']}")
    print(
        f"Split: {summary['training_rows']} training, "
        f"{summary['test_rows']} test"
    )
    for target, model in summary["selected_models"].items():
        print(f"Selected for {target}: {MODEL_DISPLAY_NAMES[model]}")
    print(f"Metrics: {summary['metrics_path']}")


if __name__ == "__main__":
    main()
