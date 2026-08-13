from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

from ml_training import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODELS_DIRECTORY,
    DEFAULT_RANDOM_STATE,
    MODEL_DISPLAY_NAMES,
    TARGET_COLUMNS,
    build_models,
    fit_and_save_models,
    load_dataset,
)
from study_config import DEFAULT_REPEATED_CV_DIRECTORY


DEFAULT_OUTPUT_DIRECTORY = DEFAULT_REPEATED_CV_DIRECTORY
DEFAULT_CV_SPLITS = 5
DEFAULT_CV_REPEATS = 3
SAFETY_QUANTILE = 0.95


def regression_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_repeated_cv(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    output_directory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
    models_directory: Path | str = DEFAULT_MODELS_DIRECTORY,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_splits: int = DEFAULT_CV_SPLITS,
    cv_repeats: int = DEFAULT_CV_REPEATS,
    random_forest_estimators: int = 300,
) -> dict[str, Any]:
    """Select models using deterministic repeated K-fold cross-validation."""

    features, targets, _ = load_dataset(dataset_path)
    if cv_splits < 2 or cv_splits > len(features):
        raise ValueError("cv_splits must be between 2 and the dataset row count.")
    if cv_repeats <= 0:
        raise ValueError("cv_repeats must be positive.")
    output_directory = Path(output_directory)
    models = build_models(
        random_state=random_state,
        random_forest_estimators=random_forest_estimators,
    )

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected_models: dict[str, str] = {}
    validation_metadata: dict[str, dict[str, Any]] = {}
    splitter = RepeatedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=random_state,
    )
    splits = list(splitter.split(features))

    for target_name in TARGET_COLUMNS:
        target = targets[target_name]
        model_predictions: dict[str, list[np.ndarray]] = {
            model_name: [] for model_name in models
        }
        actual_folds: list[np.ndarray] = []
        for split_index, (train_indices, test_indices) in enumerate(splits):
            repeat_index = split_index // cv_splits + 1
            fold_index = split_index % cv_splits + 1
            actual = target[test_indices]
            actual_folds.append(actual)
            for model_name, model in models.items():
                fitted = clone(model).fit(
                    features[train_indices], target[train_indices]
                )
                predicted = fitted.predict(features[test_indices])
                model_predictions[model_name].append(predicted)
                fold_rows.append(
                    {
                        "target": target_name,
                        "model": model_name,
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "training_rows": len(train_indices),
                        "test_rows": len(test_indices),
                        **regression_metrics(actual, predicted),
                    }
                )

        combined_actual = np.concatenate(actual_folds)
        target_overall_rows: list[dict[str, Any]] = []
        for model_name in models:
            combined_prediction = np.concatenate(model_predictions[model_name])
            underprediction_error = combined_actual - combined_prediction
            overall_row = {
                "target": target_name,
                "model": model_name,
                "fold_count": len(splits),
                "prediction_count": len(combined_actual),
                **regression_metrics(combined_actual, combined_prediction),
                "underprediction_error_p95": max(
                    0.0,
                    float(np.quantile(underprediction_error, SAFETY_QUANTILE)),
                ),
                "selected": False,
            }
            target_overall_rows.append(overall_row)

        winner = min(
            target_overall_rows,
            key=lambda row: (row["rmse"], row["mae"]),
        )
        winner["selected"] = True
        selected_models[target_name] = winner["model"]
        validation_metadata[target_name] = {
            "method": "repeated_k_fold",
            "cv_splits": cv_splits,
            "cv_repeats": cv_repeats,
            "pooled_prediction_count": winner["prediction_count"],
            "residual_definition": "actual_minus_predicted",
            "underprediction_error_quantile": SAFETY_QUANTILE,
            "underprediction_error_quantile_value": winner[
                "underprediction_error_p95"
            ],
        }
        summary_rows.extend(target_overall_rows)

    fold_fields = (
        "target",
        "model",
        "repeat",
        "fold",
        "training_rows",
        "test_rows",
        "mae",
        "rmse",
        "r2",
    )
    summary_fields = (
        "target",
        "model",
        "fold_count",
        "prediction_count",
        "mae",
        "rmse",
        "r2",
        "underprediction_error_p95",
        "selected",
    )
    fold_path = output_directory / "fold_metrics.csv"
    summary_path = output_directory / "summary_metrics.csv"
    _write_csv(fold_path, fold_fields, fold_rows)
    _write_csv(summary_path, summary_fields, summary_rows)
    saved_model_paths = fit_and_save_models(
        features,
        targets,
        selected_models,
        models_directory,
        selection_rule="lowest pooled repeated-CV RMSE, then MAE",
        validation_metadata=validation_metadata,
        random_state=random_state,
        random_forest_estimators=random_forest_estimators,
    )

    return {
        "dataset_rows": len(features),
        "folds_per_target": len(splits),
        "selected_models": selected_models,
        "validation_metadata": validation_metadata,
        "saved_model_paths": {
            target: str(path) for target, path in saved_model_paths.items()
        },
        "fold_metrics_path": str(fold_path.resolve()),
        "summary_metrics_path": str(summary_path.resolve()),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate models using deterministic repeated K-fold CV."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument(
        "--models-directory", type=Path, default=DEFAULT_MODELS_DIRECTORY
    )
    return parser


def main() -> None:
    arguments = _build_argument_parser().parse_args()
    summary = evaluate_repeated_cv(
        arguments.dataset,
        arguments.output_directory,
        arguments.models_directory,
    )
    print(f"Dataset rows: {summary['dataset_rows']}")
    print(f"Repeated-CV folds per target: {summary['folds_per_target']}")
    for target, model in summary["selected_models"].items():
        print(f"Best repeated-CV model for {target}: {MODEL_DISPLAY_NAMES[model]}")
    print(f"Summary: {summary['summary_metrics_path']}")


if __name__ == "__main__":
    main()
