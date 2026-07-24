import math
import unittest

from train_simulator import simulate


class TrainSimulatorTests(unittest.TestCase):
    def test_baseline_stops_at_station(self) -> None:
        result = simulate(
            acceleration=0.8,
            deceleration=0.9,
            speed_limit_kmh=80.0,
            coasting_point=1_300.0,
        )

        self.assertTrue(result["feasible"])
        self.assertAlmostEqual(result["final_speed_ms"], 0.0)
        self.assertLessEqual(abs(result["final_position_error_m"]), 1.0)
        self.assertFalse(result["stopped_early"])
        self.assertFalse(result["overshot_destination"])

    def test_early_coasting_reports_positive_stopping_error(self) -> None:
        result = simulate(
            acceleration=0.8,
            deceleration=0.9,
            speed_limit_kmh=80.0,
            coasting_point=0.0,
        )

        self.assertFalse(result["feasible"])
        self.assertTrue(result["stopped_early"])
        self.assertGreater(result["final_position_error_m"], 0.0)
        self.assertFalse(result["overshot_destination"])

    def test_signed_error_matches_final_stopping_position(self) -> None:
        route_length = 2_000.0
        result = simulate(0.8, 0.9, 80.0, 1_300.0, route_length=route_length)

        self.assertAlmostEqual(
            result["final_position_error_m"],
            route_length - result["final_position_m"],
        )
        self.assertEqual(
            result["overshot_destination"],
            result["final_position_error_m"] < -1.0,
        )

    def test_station_crossing_continues_to_stop_and_reports_overshoot(self) -> None:
        result = simulate(
            acceleration=1.0,
            deceleration=1.0,
            speed_limit_kmh=100.0,
            coasting_point=0.01,
            route_length=0.01,
            time_step=1.0,
            position_tolerance=0.001,
        )

        self.assertAlmostEqual(result["final_speed_ms"], 0.0)
        self.assertAlmostEqual(result["final_position_m"], 0.02)
        self.assertAlmostEqual(result["final_position_error_m"], -0.01)
        self.assertTrue(result["overshot_destination"])
        self.assertFalse(result["feasible"])

    def test_speed_limit_event_uses_a_partial_step_and_correct_energy(self) -> None:
        result = simulate(
            acceleration=1.0,
            deceleration=1.0,
            speed_limit_kmh=0.72,
            coasting_point=100.0,
            route_length=100.0,
            time_step=0.3,
        )
        history = result["history"]

        self.assertAlmostEqual(history["time_s"][1], 0.2)
        self.assertAlmostEqual(history["speed_ms"][1], 0.2)

        force_n = 200_000.0 * 1.0 + 3_000.0
        expected_energy_kwh = (force_n * 0.1 / 0.90) * 0.2 / 3.6e6
        self.assertAlmostEqual(history["energy_kwh"][1], expected_energy_kwh)

    def test_coasting_begins_at_the_requested_position(self) -> None:
        result = simulate(
            acceleration=1.0,
            deceleration=1.0,
            speed_limit_kmh=100.0,
            coasting_point=0.25,
            route_length=100.0,
            time_step=1.0,
        )
        history = result["history"]

        self.assertAlmostEqual(history["position_m"][1], 0.25)
        self.assertAlmostEqual(history["time_s"][1], math.sqrt(0.5))
        self.assertEqual(history["mode"][1], "accelerating")
        self.assertEqual(history["mode"][2], "coasting")

    def test_timeout_uses_exact_maximum_time(self) -> None:
        result = simulate(
            acceleration=0.8,
            deceleration=0.9,
            speed_limit_kmh=80.0,
            coasting_point=1_300.0,
            max_time=0.25,
        )

        self.assertTrue(result["timed_out"])
        self.assertAlmostEqual(result["travel_time_s"], 0.25)

    def test_history_columns_remain_aligned(self) -> None:
        history = simulate(0.8, 0.9, 80.0, 1_300.0)["history"]
        lengths = {len(values) for values in history.values()}
        self.assertEqual(lengths, {len(history["time_s"])})

    def test_baseline_history_preserves_physical_invariants(self) -> None:
        history = simulate(0.8, 0.9, 80.0, 1_300.0)["history"]

        self.assertTrue(
            all(
                later > earlier
                for earlier, later in zip(
                    history["time_s"],
                    history["time_s"][1:],
                )
            )
        )
        self.assertTrue(
            all(
                later >= earlier
                for earlier, later in zip(
                    history["position_m"],
                    history["position_m"][1:],
                )
            )
        )
        self.assertTrue(all(speed >= 0.0 for speed in history["speed_ms"]))
        self.assertLessEqual(max(history["speed_ms"]) * 3.6, 80.0)
        self.assertTrue(
            all(
                later >= earlier
                for earlier, later in zip(
                    history["energy_kwh"],
                    history["energy_kwh"][1:],
                )
            )
        )

    def test_rejects_non_finite_inputs(self) -> None:
        with self.assertRaises(ValueError):
            simulate(math.nan, 0.9, 80.0, 1_300.0)
        with self.assertRaises(ValueError):
            simulate(0.8, 0.9, math.inf, 1_300.0)

    def test_rejects_invalid_fixed_parameters(self) -> None:
        invalid_options = (
            {"route_length": 0.0},
            {"mass": 0.0},
            {"max_time": 0.0},
            {"resistance_a": -1.0},
            {"resistance_b": -1.0},
            {"resistance_c": -1.0},
        )

        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                simulate(0.8, 0.9, 80.0, 1_300.0, **options)


if __name__ == "__main__":
    unittest.main()
