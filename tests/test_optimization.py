import ast
import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import optimization
from ml_training import (
    FEATURE_COLUMNS,
    STATUS_COLUMNS,
    TARGET_COLUMNS,
    fit_and_save_models,
    load_dataset,
)
from optimization import (
    generate_latin_hypercube,
    optimize_surrogates,
    rank_feasible_candidates,
)


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


class OptimizationTests(unittest.TestCase):
    def test_module_does_not_depend_on_simulator(self) -> None:
        syntax_tree = ast.parse(inspect.getsource(optimization))
        imported_modules = set()
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        self.assertNotIn("train_simulator", imported_modules)
        self.assertNotIn("experiments", imported_modules)

    def test_latin_hypercube_is_reproducible_and_within_bounds(self) -> None:
        bounds = np.asarray(
            [[0.6, 1.0], [0.7, 1.1], [60.0, 100.0], [1_000.0, 1_600.0]]
        )
        first = generate_latin_hypercube(100, bounds, random_state=42)
        second = generate_latin_hypercube(100, bounds, random_state=42)

        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first >= bounds[:, 0]))
        self.assertTrue(np.all(first <= bounds[:, 1]))

    def test_exact_time_limit_is_feasible_and_time_breaks_energy_tie(self) -> None:
        candidates = np.asarray(
            [
                [0.6, 0.7, 80.0, 1_000.0],
                [0.7, 0.8, 80.0, 1_100.0],
                [0.8, 0.9, 80.0, 1_200.0],
            ]
        )
        ranked = rank_feasible_candidates(
            candidates,
            np.asarray([1.0, 1.0, 0.5]),
            np.asarray([150.0, 149.0, 150.000001]),
        )

        self.assertEqual(ranked, [1, 0])

    def test_optimizer_is_reproducible_and_exports_surrogate_only_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "dataset.csv"
            models_directory = root / "models"
            write_dataset(dataset_path)
            features, targets, _ = load_dataset(dataset_path)
            fit_and_save_models(
                features,
                targets,
                {target: "polynomial_ridge" for target in TARGET_COLUMNS},
                models_directory,
                selection_rule="test repeated cross-validation",
            )

            first = optimize_surrogates(
                dataset_path,
                models_directory,
                root / "first",
                sample_count=500,
                top_results=10,
            )
            second = optimize_surrogates(
                dataset_path,
                models_directory,
                root / "second",
                sample_count=500,
                top_results=10,
            )

            self.assertEqual(first["best_candidate"], second["best_candidate"])
            best = first["best_candidate"]
            self.assertLessEqual(best["predicted_travel_time_s"], 150.0)
            self.assertEqual(best["verification_status"], "surrogate_prediction_only")
            for column_index, column in enumerate(FEATURE_COLUMNS):
                self.assertGreaterEqual(best[column], features[:, column_index].min())
                self.assertLessEqual(best[column], features[:, column_index].max())

            result_document = json.loads(
                (root / "first" / "optimal_parameters.json").read_text()
            )
            self.assertEqual(
                result_document["models"],
                {
                    "energy_kwh": "polynomial_ridge",
                    "travel_time_s": "polynomial_ridge",
                },
            )
            self.assertEqual(
                result_document["verification_status"],
                "surrogate_prediction_only",
            )


if __name__ == "__main__":
    unittest.main()
