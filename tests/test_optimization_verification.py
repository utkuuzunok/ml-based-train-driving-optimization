from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from optimization_verification import verify_ranked_candidates


def write_candidates(path: Path) -> None:
    fieldnames = (
        "rank",
        "acceleration_ms2",
        "deceleration_ms2",
        "speed_limit_kmh",
        "coasting_point_m",
        "predicted_energy_kwh",
        "predicted_travel_time_s",
    )
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "rank": 1,
                "acceleration_ms2": 0.8,
                "deceleration_ms2": 0.8,
                "speed_limit_kmh": 80.0,
                "coasting_point_m": 900.0,
                "predicted_energy_kwh": 5.0,
                "predicted_travel_time_s": 149.0,
            }
        )
        writer.writerow(
            {
                "rank": 2,
                "acceleration_ms2": 1.0,
                "deceleration_ms2": 1.0,
                "speed_limit_kmh": 70.0,
                "coasting_point_m": 800.0,
                "predicted_energy_kwh": 6.0,
                "predicted_travel_time_s": 145.0,
            }
        )


def fake_simulator(**parameters: float) -> dict[str, float | bool]:
    first = parameters["acceleration"] == 0.8
    return {
        "energy_kwh": 4.8 if first else 6.2,
        "travel_time_s": 151.0 if first else 146.0,
        "feasible": True,
        "final_position_error_m": 0.0,
    }


class OptimizationVerificationTests(unittest.TestCase):
    def test_selects_best_simulator_verified_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidates_path = root / "candidates.csv"
            write_candidates(candidates_path)
            summary = verify_ranked_candidates(
                candidates_path,
                root / "verification",
                simulator=fake_simulator,
            )

            self.assertEqual(summary["candidates_checked"], 2)
            self.assertEqual(summary["verified_successful_candidates"], 1)
            self.assertEqual(
                summary["best_verified_candidate"]["surrogate_rank"], 2
            )
            saved = json.loads(Path(summary["summary_path"]).read_text())
            self.assertEqual(saved["verification_status"], "simulator_verified")


if __name__ == "__main__":
    unittest.main()
