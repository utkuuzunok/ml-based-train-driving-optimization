import ast
import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import ml_evaluation
from ml_evaluation import evaluate_repeated_cv
from ml_training import FEATURE_COLUMNS, STATUS_COLUMNS, TARGET_COLUMNS


def write_sampled_dataset(path: Path) -> None:
    fieldnames = (*FEATURE_COLUMNS, *TARGET_COLUMNS, *STATUS_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        generator = np.random.default_rng(12)
        for _ in range(60):
            acceleration = generator.uniform(0.4, 1.2)
            deceleration = generator.uniform(0.5, 1.1)
            speed_limit = generator.uniform(50.0, 110.0)
            coasting_point = generator.uniform(100.0, 1_800.0)
            writer.writerow(
                {
                    "acceleration_ms2": acceleration,
                    "deceleration_ms2": deceleration,
                    "speed_limit_kmh": speed_limit,
                    "coasting_point_m": coasting_point,
                    "energy_kwh": (
                        4.0 * acceleration
                        + deceleration
                        + 0.1 * speed_limit
                        + 0.001 * coasting_point
                    ),
                    "travel_time_s": (
                        180.0
                        - 15.0 * acceleration
                        - 8.0 * deceleration
                        - 0.4 * speed_limit
                        + 0.002 * coasting_point
                    ),
                    "input_valid": True,
                    "simulation_completed": True,
                    "feasible": True,
                }
            )


class RepeatedCrossValidationTests(unittest.TestCase):
    def test_module_does_not_depend_on_simulator(self) -> None:
        syntax_tree = ast.parse(inspect.getsource(ml_evaluation))
        imported_modules = set()
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        self.assertNotIn("train_simulator", imported_modules)
        self.assertNotIn("experiments", imported_modules)

    def test_evaluates_every_model_across_repeated_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "dataset.csv"
            write_sampled_dataset(dataset_path)

            result = evaluate_repeated_cv(
                dataset_path,
                root / "evaluation",
                root / "models",
                cv_splits=5,
                cv_repeats=2,
                random_forest_estimators=5,
            )

            self.assertEqual(result["dataset_rows"], 60)
            self.assertEqual(result["folds_per_target"], 10)
            self.assertEqual(set(result["selected_models"]), set(TARGET_COLUMNS))
            self.assertTrue((root / "models" / "energy_model.joblib").is_file())
            self.assertTrue(
                (root / "models" / "travel_time_model.joblib").is_file()
            )

            with (root / "evaluation" / "fold_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as csv_file:
                fold_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(fold_rows), 2 * 3 * 10)
            self.assertEqual({row["repeat"] for row in fold_rows}, {"1", "2"})
            self.assertEqual({row["fold"] for row in fold_rows}, {"1", "2", "3", "4", "5"})

            with (root / "evaluation" / "summary_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as csv_file:
                summary_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(summary_rows), 2 * 3)
            selected = [row for row in summary_rows if row["selected"] == "True"]
            self.assertEqual(len(selected), 2)
            for row in selected:
                self.assertGreaterEqual(float(row["underprediction_error_p95"]), 0.0)

            travel_metadata = json.loads(
                (root / "models" / "travel_time_model.metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            validation = travel_metadata["validation"]
            self.assertEqual(validation["method"], "repeated_k_fold")
            self.assertEqual(validation["cv_splits"], 5)
            self.assertEqual(validation["cv_repeats"], 2)
            self.assertEqual(validation["underprediction_error_quantile"], 0.95)
            self.assertGreaterEqual(
                validation["underprediction_error_quantile_value"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
