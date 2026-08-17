from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coasting_analysis import compare_coasting_strategies, optimize_no_coasting


def comparison_simulator(**parameters: float) -> dict[str, object]:
    acceleration = parameters["acceleration"]
    deceleration = parameters["deceleration"]
    speed_limit = parameters["speed_limit_kmh"]
    coasting_point = parameters["coasting_point"]
    no_coasting_penalty = 2.0 if coasting_point >= 2_000.0 else 0.0
    energy = (
        8.0
        + (acceleration - 0.8) ** 2
        + (deceleration - 0.8) ** 2
        + ((speed_limit - 80.0) / 50.0) ** 2
        + no_coasting_penalty
    )
    return {
        "energy_kwh": energy,
        "travel_time_s": 140.0,
        "feasible": True,
        "final_position_error_m": 0.0,
        "history": {
            "position_m": [0.0, 2_000.0],
            "speed_ms": [0.0, 0.0],
            "energy_kwh": [0.0, energy],
        },
    }


class CoastingAnalysisTests(unittest.TestCase):
    def test_comparison_is_reproducible_verified_and_exported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            enabled_path = root / "enabled.json"
            enabled_path.write_text(
                json.dumps(
                    {
                        "best_candidate": {
                            "acceleration_ms2": 0.8,
                            "deceleration_ms2": 0.8,
                            "speed_limit_kmh": 80.0,
                            "coasting_point_m": 900.0,
                            "energy_kwh": 8.0,
                            "travel_time_s": 140.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = compare_coasting_strategies(
                enabled_path,
                root / "results",
                root / "comparison.png",
                simulator=comparison_simulator,
                seeds=(11, 29),
                max_iterations=20,
                population_size=5,
            )

            self.assertGreater(result["energy_saving_kwh"], 0.0)
            self.assertEqual(result["verification_status"], "simulator_verified")
            self.assertEqual(result["no_coasting_successful_runs"], 2)
            self.assertGreaterEqual(result["no_coasting_energy_spread_kwh"], 0.0)
            self.assertTrue(Path(result["summary_path"]).is_file())
            self.assertTrue(Path(result["comparison_path"]).is_file())
            self.assertGreater(Path(result["figure_path"]).stat().st_size, 0)

    def test_rejects_duplicate_seeds(self) -> None:
        with self.assertRaises(ValueError):
            optimize_no_coasting(
                simulator=comparison_simulator,
                seeds=(11, 11),
                max_iterations=1,
                population_size=2,
            )


if __name__ == "__main__":
    unittest.main()
