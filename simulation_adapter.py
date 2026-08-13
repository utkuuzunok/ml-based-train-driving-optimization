"""Translate optimization feature values into simulator calls."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from study_config import FEATURE_COLUMNS
from train_simulator import simulate


Simulator = Callable[..., dict[str, Any]]

_SIMULATOR_ARGUMENTS = (
    "acceleration",
    "deceleration",
    "speed_limit_kmh",
    "coasting_point",
)


def feature_mapping(values: Sequence[float]) -> dict[str, float]:
    """Map an ordered numeric vector to the shared feature schema."""

    if len(values) != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected exactly {len(FEATURE_COLUMNS)} feature values.")
    return {
        column: float(value)
        for column, value in zip(FEATURE_COLUMNS, values)
    }


def run_feature_candidate(
    features: Mapping[str, float],
    *,
    simulator: Simulator = simulate,
) -> dict[str, Any]:
    """Run the simulator using values expressed in the ML feature schema."""

    missing = [column for column in FEATURE_COLUMNS if column not in features]
    if missing:
        raise ValueError("Missing feature values: " + ", ".join(missing))
    arguments = {
        argument: float(features[column])
        for column, argument in zip(FEATURE_COLUMNS, _SIMULATOR_ARGUMENTS)
    }
    return simulator(**arguments)
