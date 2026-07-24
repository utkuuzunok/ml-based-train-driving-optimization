import ast
import csv
import inspect
import tempfile
import unittest
from pathlib import Path

import ml_evaluation
from ml_evaluation import evaluate_structured_holdouts
from ml_training import FEATURE_COLUMNS, STATUS_COLUMNS, TARGET_COLUMNS


def write_structured_dataset(path: Path) -> None:
    fieldnames = (*FEATURE_COLUMNS, *TARGET_COLUMNS, *STATUS_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for acceleration in (0.6, 0.8, 1.0):
            for deceleration in (0.7, 0.9, 1.1):
                for speed_limit in (60.0, 80.0, 100.0):
                    for coasting_point in (1_000.0, 1_300.0, 1_600.0):
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


class StructuredEvaluationTests(unittest.TestCase):
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

    def test_evaluates_every_level_for_each_feature_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "dataset.csv"
            write_structured_dataset(dataset_path)

            result = evaluate_structured_holdouts(
                dataset_path,
                root / "evaluation",
                root / "models",
                random_forest_estimators=5,
            )

            self.assertEqual(result["dataset_rows"], 81)
            self.assertEqual(result["folds_per_target"], 12)
            self.assertEqual(set(result["selected_models"]), set(TARGET_COLUMNS))
            self.assertTrue((root / "models" / "energy_model.joblib").is_file())
            self.assertTrue(
                (root / "models" / "travel_time_model.joblib").is_file()
            )

            with (root / "evaluation" / "fold_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as csv_file:
                fold_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(fold_rows), 2 * 3 * 12)
            self.assertEqual(
                {row["holdout_type"] for row in fold_rows},
                {"boundary_extrapolation", "interpolation"},
            )

            with (root / "evaluation" / "summary_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as csv_file:
                summary_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(summary_rows), 2 * 3 * 5)
            selected = [row for row in summary_rows if row["selected"] == "True"]
            self.assertEqual(len(selected), 2)
            self.assertTrue(
                all(row["held_out_feature"] == "all_features" for row in selected)
            )


if __name__ == "__main__":
    unittest.main()
