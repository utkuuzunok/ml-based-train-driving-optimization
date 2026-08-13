"""Dataset validation, surrogate definitions, and fitted-model persistence."""

from __future__ import annotations

import csv
import json
import math
import platform
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from study_config import (
    DEFAULT_EXPERIMENT_OUTPUT_PATH,
    DEFAULT_MODELS_DIRECTORY as STUDY_DEFAULT_MODELS_DIRECTORY,
    FEATURE_COLUMNS,
)

TARGET_COLUMNS = ("energy_kwh", "travel_time_s")
STATUS_COLUMNS = ("input_valid", "simulation_completed", "feasible")
REQUIRED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS + STATUS_COLUMNS

DEFAULT_DATASET_PATH = DEFAULT_EXPERIMENT_OUTPUT_PATH
DEFAULT_MODELS_DIRECTORY = STUDY_DEFAULT_MODELS_DIRECTORY
DEFAULT_RANDOM_STATE = 42

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


def fit_and_save_models(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    selected_models: dict[str, str],
    models_directory: Path | str = DEFAULT_MODELS_DIRECTORY,
    *,
    selection_rule: str,
    validation_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    random_forest_estimators: int = 300,
) -> dict[str, Path]:
    """Fit selected models on all rows and save each model with its metadata."""

    if set(selected_models) != set(TARGET_COLUMNS):
        raise ValueError("A selected model is required for every target.")
    if features.ndim != 2 or features.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("Features must contain exactly the four configured columns.")

    models = build_models(
        random_state=random_state,
        random_forest_estimators=random_forest_estimators,
    )
    models_directory = Path(models_directory)
    saved_paths: dict[str, Path] = {}

    for target_name in TARGET_COLUMNS:
        model_name = selected_models[target_name]
        if model_name not in models:
            raise ValueError(f"Unknown selected model: {model_name}.")
        if len(targets[target_name]) != len(features):
            raise ValueError(f"Target length does not match features: {target_name}.")

        final_model = clone(models[model_name]).fit(features, targets[target_name])
        filename_target = target_name.removesuffix("_kwh").removesuffix("_s")
        model_path = models_directory / f"{filename_target}_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, model_path)

        metadata = {
            "target": target_name,
            "selected_model": model_name,
            "selected_model_display_name": MODEL_DISPLAY_NAMES[model_name],
            "feature_columns": list(FEATURE_COLUMNS),
            "training_rows": len(features),
            "feature_bounds": {
                column: {
                    "minimum": float(features[:, index].min()),
                    "maximum": float(features[:, index].max()),
                }
                for index, column in enumerate(FEATURE_COLUMNS)
            },
            "selection_rule": selection_rule,
            "random_state": random_state,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
        }
        if validation_metadata is not None:
            target_validation = validation_metadata.get(target_name)
            if target_validation is None:
                raise ValueError(
                    f"Validation metadata is missing for {target_name}."
                )
            metadata["validation"] = dict(target_validation)
        metadata_path = model_path.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        saved_paths[target_name] = model_path.resolve()

    return saved_paths
