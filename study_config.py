"""Shared configuration for the train-driving optimization study."""

from pathlib import Path


ROUTE_LENGTH_M = 2_000.0
TRAVEL_TIME_LIMIT_S = 150.0

FEATURE_COLUMNS = (
    "acceleration_ms2",
    "deceleration_ms2",
    "speed_limit_kmh",
    "coasting_point_m",
)

PARAMETER_BOUNDS = {
    "acceleration_ms2": (0.4, 1.2),
    "deceleration_ms2": (0.5, 1.1),
    "speed_limit_kmh": (50.0, 110.0),
    "coasting_point_m": (100.0, 1_800.0),
}

LHS_SAMPLE_COUNT = 4_483
LHS_RANDOM_STATE = 42
REFERENCE_PARAMETERS = (0.8, 0.9, 80.0, 1_300.0)
BOUNDARY_CORNER_COUNT = 2 ** len(FEATURE_COLUMNS)
EXPECTED_EXPERIMENT_COUNT = (
    LHS_SAMPLE_COUNT + BOUNDARY_CORNER_COUNT + len((REFERENCE_PARAMETERS,))
)

DEFAULT_EXPERIMENT_OUTPUT_PATH = Path("results/train_experiments.csv")
DEFAULT_ML_RESULTS_DIRECTORY = Path("results/ml")
DEFAULT_MODELS_DIRECTORY = Path("models")
DEFAULT_REPEATED_CV_DIRECTORY = Path("results/ml/repeated_cv")
DEFAULT_OPTIMIZATION_OUTPUT_DIRECTORY = Path("results/ml/optimization")
DEFAULT_DIRECT_OPTIMIZATION_OUTPUT_DIRECTORY = Path(
    "results/ml/direct_optimization"
)
DEFAULT_COASTING_COMPARISON_OUTPUT_DIRECTORY = Path(
    "results/ml/coasting_comparison"
)
DEFAULT_ROBUSTNESS_OUTPUT_DIRECTORY = Path("results/ml/robustness")
