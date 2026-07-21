from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_training import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RANDOM_STATE,
    FEATURE_COLUMNS,
    MODEL_DISPLAY_NAMES,
    TARGET_COLUMNS,
    build_models,
    load_dataset,
)


DEFAULT_OUTPUT_DIRECTORY = Path("results/ml/structured_holdout")


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


def evaluate_structured_holdouts(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    output_directory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    random_forest_estimators: int = 300,
) -> dict[str, Any]:
    """Evaluate interpolation and boundary generalization by parameter level."""

    features, targets, _ = load_dataset(dataset_path)
    output_directory = Path(output_directory)
    models = build_models(
        random_state=random_state,
        random_forest_estimators=random_forest_estimators,
    )

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected_models: dict[str, str] = {}

    for target_name in TARGET_COLUMNS:
        target = targets[target_name]
        overall_predictions: dict[str, list[np.ndarray]] = {
            model_name: [] for model_name in models
        }
        overall_actual: list[np.ndarray] = []

        for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
            levels = np.unique(features[:, feature_index])
            if len(levels) < 2:
                raise ValueError(
                    f"Structured holdout requires at least two levels for {feature_name}."
                )

            feature_actual: list[np.ndarray] = []
            feature_predictions: dict[str, list[np.ndarray]] = {
                model_name: [] for model_name in models
            }

            for held_out_value in levels:
                test_mask = features[:, feature_index] == held_out_value
                train_mask = ~test_mask
                actual = target[test_mask]
                feature_actual.append(actual)

                if held_out_value in (levels[0], levels[-1]):
                    holdout_type = "boundary_extrapolation"
                else:
                    holdout_type = "interpolation"

                for model_name, model in models.items():
                    fitted = clone(model).fit(features[train_mask], target[train_mask])
                    predicted = fitted.predict(features[test_mask])
                    feature_predictions[model_name].append(predicted)
                    fold_rows.append(
                        {
                            "target": target_name,
                            "model": model_name,
                            "held_out_feature": feature_name,
                            "held_out_value": float(held_out_value),
                            "holdout_type": holdout_type,
                            "training_rows": int(train_mask.sum()),
                            "test_rows": int(test_mask.sum()),
                            **regression_metrics(actual, predicted),
                        }
                    )

            combined_actual = np.concatenate(feature_actual)
            overall_actual.append(combined_actual)
            for model_name in models:
                combined_prediction = np.concatenate(feature_predictions[model_name])
                overall_predictions[model_name].append(combined_prediction)
                summary_rows.append(
                    {
                        "target": target_name,
                        "model": model_name,
                        "held_out_feature": feature_name,
                        "fold_count": len(levels),
                        "prediction_count": len(combined_actual),
                        **regression_metrics(combined_actual, combined_prediction),
                        "selected": False,
                    }
                )

        combined_overall_actual = np.concatenate(overall_actual)
        target_overall_rows: list[dict[str, Any]] = []
        for model_name in models:
            combined_overall_prediction = np.concatenate(
                overall_predictions[model_name]
            )
            overall_row = {
                "target": target_name,
                "model": model_name,
                "held_out_feature": "all_features",
                "fold_count": sum(
                    len(np.unique(features[:, feature_index]))
                    for feature_index in range(len(FEATURE_COLUMNS))
                ),
                "prediction_count": len(combined_overall_actual),
                **regression_metrics(
                    combined_overall_actual, combined_overall_prediction
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
        summary_rows.extend(target_overall_rows)

    fold_fields = (
        "target",
        "model",
        "held_out_feature",
        "held_out_value",
        "holdout_type",
        "training_rows",
        "test_rows",
        "mae",
        "rmse",
        "r2",
    )
    summary_fields = (
        "target",
        "model",
        "held_out_feature",
        "fold_count",
        "prediction_count",
        "mae",
        "rmse",
        "r2",
        "selected",
    )
    fold_path = output_directory / "fold_metrics.csv"
    summary_path = output_directory / "summary_metrics.csv"
    _write_csv(fold_path, fold_fields, fold_rows)
    _write_csv(summary_path, summary_fields, summary_rows)

    return {
        "dataset_rows": len(features),
        "folds_per_target": sum(
            len(np.unique(features[:, feature_index]))
            for feature_index in range(len(FEATURE_COLUMNS))
        ),
        "selected_models": selected_models,
        "fold_metrics_path": str(fold_path.resolve()),
        "summary_metrics_path": str(summary_path.resolve()),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate models by holding out complete parameter levels."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    return parser


def main() -> None:
    arguments = _build_argument_parser().parse_args()
    summary = evaluate_structured_holdouts(
        arguments.dataset,
        arguments.output_directory,
    )
    print(f"Dataset rows: {summary['dataset_rows']}")
    print(f"Structured folds per target: {summary['folds_per_target']}")
    for target, model in summary["selected_models"].items():
        print(f"Best structured-holdout model for {target}: {MODEL_DISPLAY_NAMES[model]}")
    print(f"Summary: {summary['summary_metrics_path']}")


if __name__ == "__main__":
    main()
