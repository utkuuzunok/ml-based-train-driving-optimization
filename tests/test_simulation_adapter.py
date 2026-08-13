import unittest

from simulation_adapter import feature_mapping, run_feature_candidate


class SimulationAdapterTests(unittest.TestCase):
    def test_maps_feature_order_to_simulator_arguments(self) -> None:
        observed = {}

        def simulator(**parameters: float) -> dict[str, object]:
            observed.update(parameters)
            return {"ok": True}

        features = feature_mapping((0.8, 0.9, 80.0, 900.0))
        result = run_feature_candidate(features, simulator=simulator)

        self.assertEqual(
            observed,
            {
                "acceleration": 0.8,
                "deceleration": 0.9,
                "speed_limit_kmh": 80.0,
                "coasting_point": 900.0,
            },
        )
        self.assertEqual(result, {"ok": True})

    def test_rejects_incomplete_feature_vectors_and_mappings(self) -> None:
        with self.assertRaises(ValueError):
            feature_mapping((0.8, 0.9))
        with self.assertRaises(ValueError):
            run_feature_candidate({"acceleration_ms2": 0.8})


if __name__ == "__main__":
    unittest.main()
