import ast
import csv
import inspect
import tempfile
import unittest
from pathlib import Path

import model_analysis
from ml_training import (
    FEATURE_COLUMNS,
    STATUS_COLUMNS,
    TARGET_COLUMNS,
    fit_and_save_models,
    load_dataset,
)
from model_analysis import analyze_models
from optimization import optimize_surrogates


def write_dataset(path: Path) -> None:
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
                                    5.0
                                    + 2.0 * acceleration
                                    + 0.5 * deceleration
                                    + 0.08 * speed_limit
                                    + 0.001 * coasting_point
                                ),
                                "travel_time_s": (
                                    160.0
                                    - 20.0 * acceleration
                                    - 10.0 * deceleration
                                    - 0.3 * speed_limit
                                    + 0.002 * coasting_point
                                ),
                                "input_valid": True,
                                "simulation_completed": True,
                                "feasible": True,
                            }
                        )


class ModelAnalysisTests(unittest.TestCase):
    def test_module_does_not_depend_on_simulator(self) -> None:
        syntax_tree = ast.parse(inspect.getsource(model_analysis))
        imported_modules = set()
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        self.assertNotIn("train_simulator", imported_modules)
        self.assertNotIn("experiments", imported_modules)

    def test_analysis_exports_importance_sensitivity_and_robustness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "dataset.csv"
            models_directory = root / "models"
            optimization_directory = root / "optimization"
            write_dataset(dataset_path)
            features, targets, _ = load_dataset(dataset_path)
            fit_and_save_models(
                features,
                targets,
                {target: "polynomial_ridge" for target in TARGET_COLUMNS},
                models_directory,
                selection_rule="test structured holdout",
            )
            optimize_surrogates(
                dataset_path,
                models_directory,
                optimization_directory,
                sample_count=500,
                local_starts=3,
                top_results=10,
            )

            result = analyze_models(
                dataset_path,
                models_directory,
                optimization_directory / "optimal_parameters.json",
                root / "analysis",
                permutation_repeats=3,
                sensitivity_points=5,
                joint_sample_count=100,
                conservative_sample_count=500,
            )

            summary = result["summary"]
            self.assertEqual(
                summary["verification_status"], "surrogate_prediction_only"
            )
            self.assertLessEqual(
                summary["conservative_candidate"]["predicted_travel_time_s"],
                119.0,
            )
            self.assertGreaterEqual(
                summary["joint_neighborhood"]["predicted_feasible_percent"], 0.0
            )
            self.assertLessEqual(
                summary["joint_neighborhood"]["predicted_feasible_percent"], 100.0
            )

            with (root / "analysis" / "permutation_importance.csv").open(
                encoding="utf-8", newline=""
            ) as csv_file:
                importance_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(importance_rows), 8)

            with (root / "analysis" / "sensitivity_curves.csv").open(
                encoding="utf-8", newline=""
            ) as csv_file:
                sensitivity_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(sensitivity_rows), 20)


if __name__ == "__main__":
    unittest.main()
