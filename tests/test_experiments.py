import csv
import tempfile
import unittest
from pathlib import Path

from experiments import (
    CSV_COLUMNS,
    EXPECTED_COMBINATION_COUNT,
    export_results_csv,
    generate_parameter_combinations,
    run_experiments,
    run_single_experiment,
    summarize_results,
)


def successful_result(
    *,
    travel_time_s: float = 110.0,
    feasible: bool = True,
) -> dict[str, object]:
    return {
        "energy_kwh": 15.5,
        "travel_time_s": travel_time_s,
        "max_speed_kmh": 80.0,
        "final_position_error_m": 0.0,
        "final_speed_ms": 0.0,
        "feasible": feasible,
        "stopped_early": not feasible,
        "overshot_destination": False,
        "timed_out": False,
        "history": {"large": [1, 2, 3]},
    }


class ExperimentTests(unittest.TestCase):
    def test_default_grid_contains_875_combinations_in_stable_order(self) -> None:
        combinations = list(generate_parameter_combinations())

        self.assertEqual(len(combinations), EXPECTED_COMBINATION_COUNT)
        self.assertEqual(EXPECTED_COMBINATION_COUNT, 875)
        self.assertEqual(combinations[0], (0.6, 0.7, 60.0, 1_000.0))
        self.assertEqual(combinations[-1], (1.0, 1.1, 100.0, 1_600.0))

    def test_successful_run_extracts_only_scalar_metrics(self) -> None:
        row = run_single_experiment(
            0.8,
            0.9,
            80.0,
            1_300.0,
            simulator=lambda **parameters: successful_result(),
        )

        self.assertEqual(tuple(row), CSV_COLUMNS)
        self.assertTrue(row["input_valid"])
        self.assertTrue(row["simulation_completed"])
        self.assertTrue(row["feasible"])
        self.assertTrue(row["meets_time_constraint"])
        self.assertTrue(row["successful"])
        self.assertNotIn("history", row)

    def test_time_limit_is_an_independent_success_constraint(self) -> None:
        row = run_single_experiment(
            0.8,
            0.9,
            80.0,
            1_300.0,
            simulator=lambda **parameters: successful_result(
                travel_time_s=120.01,
            ),
        )

        self.assertTrue(row["feasible"])
        self.assertFalse(row["meets_time_constraint"])
        self.assertFalse(row["successful"])

    def test_time_limit_must_be_finite_and_positive(self) -> None:
        for invalid_limit in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(invalid_limit=invalid_limit):
                with self.assertRaises(ValueError):
                    run_single_experiment(
                        0.8,
                        0.9,
                        80.0,
                        1_300.0,
                        simulator=lambda **parameters: successful_result(),
                        travel_time_limit_s=invalid_limit,
                    )

    def test_infeasible_run_is_preserved_as_a_completed_row(self) -> None:
        row = run_single_experiment(
            0.8,
            0.9,
            80.0,
            1_300.0,
            simulator=lambda **parameters: successful_result(feasible=False),
        )

        self.assertTrue(row["simulation_completed"])
        self.assertFalse(row["feasible"])
        self.assertFalse(row["successful"])
        self.assertTrue(row["stopped_early"])

    def test_validation_failure_is_recorded_without_metrics(self) -> None:
        def invalid_simulator(**parameters: float) -> dict[str, object]:
            raise ValueError("invalid combination")

        row = run_single_experiment(
            -0.1,
            0.9,
            80.0,
            1_300.0,
            simulator=invalid_simulator,
        )

        self.assertFalse(row["input_valid"])
        self.assertFalse(row["simulation_completed"])
        self.assertIsNone(row["energy_kwh"])
        self.assertEqual(row["error_type"], "ValueError")
        self.assertEqual(row["error_message"], "invalid combination")

    def test_unexpected_failure_does_not_abort_the_grid(self) -> None:
        calls = 0

        def sometimes_failing_simulator(
            **parameters: float,
        ) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary failure")
            return successful_result()

        rows = run_experiments(
            accelerations=(0.7, 0.8),
            decelerations=(0.9,),
            speed_limits=(80.0,),
            coasting_points=(1_300.0,),
            simulator=sometimes_failing_simulator,
        )

        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]["simulation_completed"])
        self.assertEqual(rows[0]["error_type"], "RuntimeError")
        self.assertTrue(rows[1]["successful"])

    def test_small_grid_calls_simulator_once_per_combination(self) -> None:
        observed_parameters = []

        def recording_simulator(**parameters: float) -> dict[str, object]:
            observed_parameters.append(parameters)
            return successful_result()

        rows = run_experiments(
            accelerations=(0.7, 0.8),
            decelerations=(0.9,),
            speed_limits=(70.0, 80.0),
            coasting_points=(1_300.0,),
            simulator=recording_simulator,
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual(len(observed_parameters), 4)
        self.assertEqual(rows[0]["acceleration_ms2"], 0.7)
        self.assertEqual(rows[-1]["speed_limit_kmh"], 80.0)

    def test_csv_export_has_stable_columns_and_one_row_per_result(self) -> None:
        rows = [
            run_single_experiment(
                0.8,
                0.9,
                80.0,
                1_300.0,
                simulator=lambda **parameters: successful_result(),
            )
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "nested" / "results.csv"
            resolved_path = export_results_csv(rows, output_path)

            with resolved_path.open(encoding="utf-8", newline="") as csv_file:
                saved_rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(saved_rows), 1)
        self.assertEqual(tuple(saved_rows[0]), CSV_COLUMNS)
        self.assertEqual(saved_rows[0]["energy_kwh"], "15.5")
        self.assertEqual(saved_rows[0]["successful"], "True")

    def test_summary_counts_each_outcome(self) -> None:
        successful = run_single_experiment(
            0.8,
            0.9,
            80.0,
            1_300.0,
            simulator=lambda **parameters: successful_result(),
        )
        too_slow = run_single_experiment(
            0.8,
            0.9,
            60.0,
            1_300.0,
            simulator=lambda **parameters: successful_result(
                travel_time_s=125.0,
            ),
        )

        def invalid_simulator(**parameters: float) -> dict[str, object]:
            raise ValueError("invalid")

        invalid = run_single_experiment(
            -1.0,
            0.9,
            80.0,
            1_300.0,
            simulator=invalid_simulator,
        )
        summary = summarize_results(
            [successful, too_slow, invalid],
            elapsed_time_s=0.5,
        )

        self.assertEqual(summary["total_combinations"], 3)
        self.assertEqual(summary["completed_simulations"], 2)
        self.assertEqual(summary["feasible_simulations"], 2)
        self.assertEqual(summary["successful_simulations"], 1)
        self.assertEqual(summary["validation_failures"], 1)
        self.assertAlmostEqual(summary["success_rate_percent"], 100.0 / 3.0)
        self.assertEqual(summary["minimum_energy_kwh"], 15.5)

    def test_real_baseline_is_successful_without_plotting(self) -> None:
        row = run_single_experiment(0.8, 0.9, 80.0, 1_300.0)

        self.assertTrue(row["successful"])
        self.assertLessEqual(row["travel_time_s"], 120.0)
        self.assertEqual(row["final_speed_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
