from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from direct_optimization import optimize_direct_simulator


def quadratic_simulator(**parameters: float) -> dict[str, float | bool]:
    acceleration = parameters["acceleration"]
    deceleration = parameters["deceleration"]
    speed_limit = parameters["speed_limit_kmh"]
    coasting_point = parameters["coasting_point"]
    return {
        "energy_kwh": (
            (acceleration - 0.8) ** 2
            + (deceleration - 0.8) ** 2
            + ((speed_limit - 80.0) / 50.0) ** 2
            + ((coasting_point - 900.0) / 1_000.0) ** 2
        ),
        "travel_time_s": 140.0,
        "feasible": True,
        "final_position_error_m": 0.0,
    }


class DirectOptimizationTests(unittest.TestCase):
    def test_multi_seed_run_is_reproducible_and_exported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = optimize_direct_simulator(
                Path(temporary_directory) / "first",
                simulator=quadratic_simulator,
                seeds=(11, 29),
                max_iterations=20,
                population_size=5,
            )
            second = optimize_direct_simulator(
                Path(temporary_directory) / "second",
                simulator=quadratic_simulator,
                seeds=(11, 29),
                max_iterations=20,
                population_size=5,
            )

            self.assertEqual(first["best_candidate"], second["best_candidate"])
            self.assertEqual(first["successful_runs"], 2)
            self.assertGreater(first["total_simulator_calls"], 0)
            saved = json.loads(Path(first["summary_path"]).read_text())
            self.assertEqual(saved["method"], first["method"])
            self.assertEqual(saved["seeds"], [11, 29])

    def test_rejects_duplicate_seeds(self) -> None:
        with self.assertRaises(ValueError):
            optimize_direct_simulator(
                simulator=quadratic_simulator,
                seeds=(11, 11),
                max_iterations=1,
                population_size=2,
            )


if __name__ == "__main__":
    unittest.main()
