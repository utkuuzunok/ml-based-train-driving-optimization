import ast
import csv
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

import ml_models
from ml_models import (
    FEATURE_COLUMNS,
    MODEL_DISPLAY_NAMES,
    STATUS_COLUMNS,
    TARGET_COLUMNS,
    build_models,
    fit_and_save_models,
    load_dataset,
)


def write_dataset(path: Path, row_count: int = 60) -> None:
    fieldnames = (*FEATURE_COLUMNS, *TARGET_COLUMNS, *STATUS_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(row_count):
            acceleration = 0.6 + 0.1 * (index % 5)
            deceleration = 0.7 + 0.1 * ((index // 5) % 4)
            speed_limit = 60.0 + 10.0 * ((index // 20) % 3)
            coasting_point = 1_000.0 + 100.0 * (index // 60)
            energy = (
                3.0 * acceleration
                + 1.5 * deceleration
                + 0.08 * speed_limit
                + 0.0005 * coasting_point
                + 0.2 * acceleration * deceleration
            )
            travel_time = (
                190.0
                - 20.0 * acceleration
                - 10.0 * deceleration
                - 0.5 * speed_limit
                + 0.002 * coasting_point
            )
            writer.writerow(
                {
                    "acceleration_ms2": acceleration,
                    "deceleration_ms2": deceleration,
                    "speed_limit_kmh": speed_limit,
                    "coasting_point_m": coasting_point,
                    "energy_kwh": energy,
                    "travel_time_s": travel_time,
                    "input_valid": True,
                    "simulation_completed": True,
                    "feasible": True,
                }
            )


class MlModelTests(unittest.TestCase):
    def test_module_has_no_simulator_dependency(self) -> None:
        syntax_tree = ast.parse(inspect.getsource(ml_models))
        imported_modules = set()
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        self.assertNotIn("train_simulator", imported_modules)
        self.assertNotIn("experiments", imported_modules)

    def test_builds_exactly_the_three_approved_models(self) -> None:
        models = build_models(random_forest_estimators=5)

        self.assertEqual(
            set(models),
            {"linear_regression", "polynomial_ridge", "random_forest"},
        )
        self.assertEqual(len(MODEL_DISPLAY_NAMES), 3)

    def test_loader_uses_only_four_features_and_two_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "dataset.csv"
            write_dataset(dataset_path)

            features, targets, source_rows = load_dataset(dataset_path)

        self.assertEqual(features.shape, (60, 4))
        self.assertEqual(set(targets), set(TARGET_COLUMNS))
        self.assertEqual(len(source_rows), 60)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_loader_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "dataset.csv"
            dataset_path.write_text("energy_kwh\n10.0\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_dataset(dataset_path)

    def test_loader_rejects_duplicate_parameter_combinations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_path = Path(temporary_directory) / "dataset.csv"
            write_dataset(dataset_path, row_count=10)
            lines = dataset_path.read_text(encoding="utf-8").splitlines()
            dataset_path.write_text(
                "\n".join((*lines, lines[1])) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "duplicate parameter"):
                load_dataset(dataset_path)

    def test_fitting_is_reproducible_and_models_are_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "dataset.csv"
            write_dataset(dataset_path)
            features, targets, _ = load_dataset(dataset_path)
            selected_models = {
                target: "random_forest" for target in TARGET_COLUMNS
            }
            validation_metadata = {
                target: {
                    "method": "test",
                    "underprediction_error_quantile": 0.95,
                    "underprediction_error_quantile_value": 1.0,
                }
                for target in TARGET_COLUMNS
            }

            first = fit_and_save_models(
                features,
                targets,
                selected_models,
                root / "first_models",
                selection_rule="test selection",
                validation_metadata=validation_metadata,
                random_forest_estimators=10,
            )
            second = fit_and_save_models(
                features,
                targets,
                selected_models,
                root / "second_models",
                selection_rule="test selection",
                validation_metadata=validation_metadata,
                random_forest_estimators=10,
            )

            self.assertEqual(set(first), set(TARGET_COLUMNS))
            self.assertEqual(set(second), set(TARGET_COLUMNS))
            candidate = np.asarray([[0.8, 0.9, 80.0, 1_000.0]])
            for target in TARGET_COLUMNS:
                first_model = joblib.load(first[target])
                second_model = joblib.load(second[target])
                first_prediction = first_model.predict(candidate)
                second_prediction = second_model.predict(candidate)
                self.assertEqual(first_prediction.shape, (1,))
                self.assertTrue(math.isfinite(float(first_prediction[0])))
                np.testing.assert_allclose(first_prediction, second_prediction)

                filename_target = target.removesuffix("_kwh").removesuffix("_s")
                first_metadata = json.loads(
                    (root / "first_models" / f"{filename_target}_model.metadata.json")
                    .read_text(encoding="utf-8")
                )
                second_metadata = json.loads(
                    (root / "second_models" / f"{filename_target}_model.metadata.json")
                    .read_text(encoding="utf-8")
                )
                self.assertEqual(first_metadata, second_metadata)
                self.assertEqual(first_metadata["validation"]["method"], "test")


if __name__ == "__main__":
    unittest.main()
