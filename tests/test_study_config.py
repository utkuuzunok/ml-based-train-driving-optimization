import unittest

from experiments import DEFAULT_OUTPUT_PATH
from ml_models import DEFAULT_DATASET_PATH
from study_config import (
    BOUNDARY_CORNER_COUNT,
    EXPECTED_EXPERIMENT_COUNT,
    FEATURE_COLUMNS,
    LHS_RANDOM_STATE,
    LHS_SAMPLE_COUNT,
    PARAMETER_BOUNDS,
    REFERENCE_PARAMETERS,
    ROUTE_LENGTH_M,
    TRAVEL_TIME_LIMIT_S,
)


class StudyConfigTests(unittest.TestCase):
    def test_expanded_study_contract(self) -> None:
        self.assertEqual(TRAVEL_TIME_LIMIT_S, 150.0)
        self.assertEqual(PARAMETER_BOUNDS["acceleration_ms2"], (0.4, 1.2))
        self.assertEqual(PARAMETER_BOUNDS["deceleration_ms2"], (0.5, 1.1))
        self.assertEqual(PARAMETER_BOUNDS["speed_limit_kmh"], (50.0, 110.0))
        self.assertEqual(PARAMETER_BOUNDS["coasting_point_m"], (100.0, 1_800.0))

    def test_sampling_contract_and_reference_are_valid(self) -> None:
        self.assertEqual(LHS_SAMPLE_COUNT, 4_483)
        self.assertEqual(LHS_RANDOM_STATE, 42)
        self.assertEqual(BOUNDARY_CORNER_COUNT, 16)
        self.assertEqual(EXPECTED_EXPERIMENT_COUNT, 4_500)
        self.assertEqual(len(REFERENCE_PARAMETERS), len(FEATURE_COLUMNS))
        for feature, value in zip(FEATURE_COLUMNS, REFERENCE_PARAMETERS):
            lower, upper = PARAMETER_BOUNDS[feature]
            self.assertLessEqual(lower, value)
            self.assertLessEqual(value, upper)

        self.assertGreaterEqual(PARAMETER_BOUNDS["coasting_point_m"][0], 0.0)
        self.assertLessEqual(
            PARAMETER_BOUNDS["coasting_point_m"][1], ROUTE_LENGTH_M
        )

    def test_experiments_and_ml_share_the_dataset_path(self) -> None:
        self.assertEqual(DEFAULT_OUTPUT_PATH, DEFAULT_DATASET_PATH)
        self.assertEqual(DEFAULT_DATASET_PATH.name, "train_experiments.csv")


if __name__ == "__main__":
    unittest.main()
