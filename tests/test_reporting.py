import ast
import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import reporting
from ml_training import FEATURE_COLUMNS
from reporting import generate_report_figures


class ReportingTests(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _write_report_inputs(self, results_directory: Path) -> None:
        models = ("linear_regression", "polynomial_ridge", "random_forest")
        targets = ("energy_kwh", "travel_time_s")
        self._write_csv(
            results_directory / "repeated_cv" / "summary_metrics.csv",
            [
                {
                    "target": target,
                    "model": model,
                    "rmse": 0.2 + model_index,
                }
                for target in targets
                for model_index, model in enumerate(models)
            ],
        )
        self._write_csv(
            results_directory / "robustness" / "permutation_importance.csv",
            [
                {
                    "target": target,
                    "feature": feature,
                    "rmse_increase_mean": 0.1 + feature_index,
                    "rmse_increase_std": 0.01,
                }
                for target in targets
                for feature_index, feature in enumerate(FEATURE_COLUMNS)
            ],
        )
        self._write_csv(
            results_directory / "robustness" / "sensitivity_curves.csv",
            [
                {
                    "varied_feature": feature,
                    "varied_value": value,
                    "predicted_energy_kwh": 8.0 + value / 1000.0,
                    "predicted_travel_time_s": 145.0 + value / 1000.0,
                }
                for feature in FEATURE_COLUMNS
                for value in (1.0, 2.0, 3.0)
            ],
        )

        optimization_directory = results_directory / "optimization"
        optimization_directory.mkdir(parents=True, exist_ok=True)
        (optimization_directory / "optimal_parameters.json").write_text(
            json.dumps(
                {
                    "best_observed_sample": {
                        "energy_kwh": 8.5,
                        "travel_time_s": 149.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        robustness_directory = results_directory / "robustness"
        (robustness_directory / "robustness_summary.json").write_text(
            json.dumps(
                {
                    "official_candidate": {
                        "predicted_energy_kwh": 8.2,
                        "predicted_travel_time_s": 150.0,
                    },
                    "conservative_candidate": {
                        "predicted_energy_kwh": 8.3,
                        "predicted_travel_time_s": 148.0,
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_module_does_not_depend_on_simulator(self) -> None:
        syntax_tree = ast.parse(inspect.getsource(reporting))
        imported_modules = set()
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        self.assertNotIn("train_simulator", imported_modules)
        self.assertNotIn("experiments", imported_modules)

    def test_generates_four_nonempty_png_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_directory = Path(temporary_directory) / "results"
            figures_directory = Path(temporary_directory) / "figures"
            self._write_report_inputs(results_directory)
            paths = generate_report_figures(
                results_directory, figures_directory
            )

            self.assertEqual(len(paths), 4)
            for path in paths.values():
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 10_000)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
