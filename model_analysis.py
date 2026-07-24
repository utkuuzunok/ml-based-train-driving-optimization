from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from ml_training import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODELS_DIRECTORY,
    DEFAULT_RANDOM_STATE,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_models,
    load_dataset,
)
from optimization import (
    DEFAULT_TRAVEL_TIME_LIMIT_S,
    load_surrogate_models,
    optimize_surrogates,
)


DEFAULT_OFFICIAL_RESULT_PATH = Path(
    "results/ml/optimization/optimal_parameters.json"
)
DEFAULT_OUTPUT_DIRECTORY = Path("results/ml/robustness")
DEFAULT_CONSERVATIVE_TIME_LIMIT_S = 119.0


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_official_candidate(path: Path | str) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    candidate = document.get("best_predicted_candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Official result does not contain a best predicted candidate.")
    required = (*FEATURE_COLUMNS, "predicted_energy_kwh", "predicted_travel_time_s")
    if any(key not in candidate for key in required):
        raise ValueError("Official candidate is missing required values.")
    if not all(math.isfinite(float(candidate[key])) for key in required):
        raise ValueError("Official candidate values must be finite.")
    return candidate


def _held_out_permutation_importance(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
    *,
    random_state: int,
    repeats: int,
) -> list[dict[str, Any]]:
    if repeats <= 0:
        raise ValueError("permutation_repeats must be positive.")

    indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=random_state, shuffle=True
    )
    model_templates = build_models(random_state=random_state)
    rows: list[dict[str, Any]] = []

    for target_name in TARGET_COLUMNS:
        model_name = metadata[target_name]["selected_model"]
        if model_name not in model_templates:
            raise ValueError(f"Unknown selected model: {model_name}.")
        fitted = clone(model_templates[model_name]).fit(
            features[train_indices], targets[target_name][train_indices]
        )
        baseline_prediction = fitted.predict(features[test_indices])
        baseline_rmse = float(
            math.sqrt(
                mean_squared_error(
                    targets[target_name][test_indices], baseline_prediction
                )
            )
        )
        importance = permutation_importance(
            fitted,
            features[test_indices],
            targets[target_name][test_indices],
            scoring="neg_root_mean_squared_error",
            n_repeats=repeats,
            random_state=random_state,
        )
        feature_order = np.argsort(-importance.importances_mean)
        ranks = {
            int(feature_index): rank
            for rank, feature_index in enumerate(feature_order, start=1)
        }
        for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
            rows.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "feature": feature_name,
                    "rank": ranks[feature_index],
                    "baseline_test_rmse": baseline_rmse,
                    "rmse_increase_mean": float(
                        importance.importances_mean[feature_index]
                    ),
                    "rmse_increase_std": float(
                        importance.importances_std[feature_index]
                    ),
                    "repeats": repeats,
                }
            )
    return rows


def _predict_candidates(
    candidates: np.ndarray, models: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    energy = np.asarray(models["energy_kwh"].predict(candidates), dtype=float)
    travel_time = np.asarray(
        models["travel_time_s"].predict(candidates), dtype=float
    )
    return energy, travel_time


def analyze_models(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    models_directory: Path | str = DEFAULT_MODELS_DIRECTORY,
    official_result_path: Path | str = DEFAULT_OFFICIAL_RESULT_PATH,
    output_directory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    permutation_repeats: int = 30,
    sensitivity_points: int = 101,
    perturbation_fractions: Sequence[float] = (0.025, 0.05),
    joint_sample_count: int = 20_000,
    conservative_time_limit_s: float = DEFAULT_CONSERVATIVE_TIME_LIMIT_S,
    conservative_sample_count: int = 100_000,
) -> dict[str, Any]:
    """Interpret selected models and quantify local robustness of the optimum."""

    if sensitivity_points < 2 or joint_sample_count <= 0:
        raise ValueError(
            "sensitivity_points must be at least 2 and joint_sample_count positive."
        )
    if any(
        not math.isfinite(fraction) or not 0.0 < fraction <= 0.5
        for fraction in perturbation_fractions
    ):
        raise ValueError("Perturbation fractions must be in the interval (0, 0.5].")
    if not 0.0 < conservative_time_limit_s < DEFAULT_TRAVEL_TIME_LIMIT_S:
        raise ValueError("The conservative time limit must be between 0 and 120 s.")

    features, targets, _ = load_dataset(dataset_path)
    models, metadata, bounds = load_surrogate_models(models_directory)
    official = _load_official_candidate(official_result_path)
    official_values = np.asarray(
        [float(official[column]) for column in FEATURE_COLUMNS], dtype=float
    )
    if np.any(official_values < bounds[:, 0]) or np.any(
        official_values > bounds[:, 1]
    ):
        raise ValueError("Official candidate lies outside model bounds.")

    output_directory = Path(output_directory)
    importance_rows = _held_out_permutation_importance(
        features,
        targets,
        metadata,
        random_state=random_state,
        repeats=permutation_repeats,
    )
    importance_path = output_directory / "permutation_importance.csv"
    _write_csv(
        importance_path,
        (
            "target",
            "model",
            "feature",
            "rank",
            "baseline_test_rmse",
            "rmse_increase_mean",
            "rmse_increase_std",
            "repeats",
        ),
        importance_rows,
    )

    sensitivity_rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
        values = np.linspace(
            bounds[feature_index, 0],
            bounds[feature_index, 1],
            sensitivity_points,
        )
        candidates = np.repeat(
            official_values.reshape(1, -1), sensitivity_points, axis=0
        )
        candidates[:, feature_index] = values
        energy, travel_time = _predict_candidates(candidates, models)
        for value, predicted_energy, predicted_time in zip(
            values, energy, travel_time
        ):
            sensitivity_rows.append(
                {
                    "varied_feature": feature_name,
                    "varied_value": float(value),
                    **{
                        column: float(candidates[len(sensitivity_rows) % sensitivity_points, index])
                        for index, column in enumerate(FEATURE_COLUMNS)
                    },
                    "predicted_energy_kwh": float(predicted_energy),
                    "predicted_travel_time_s": float(predicted_time),
                    "predicted_feasible_120s": bool(
                        predicted_time <= DEFAULT_TRAVEL_TIME_LIMIT_S
                    ),
                }
            )
    sensitivity_path = output_directory / "sensitivity_curves.csv"
    _write_csv(
        sensitivity_path,
        (
            "varied_feature",
            "varied_value",
            *FEATURE_COLUMNS,
            "predicted_energy_kwh",
            "predicted_travel_time_s",
            "predicted_feasible_120s",
        ),
        sensitivity_rows,
    )

    official_energy, official_time = _predict_candidates(
        official_values.reshape(1, -1), models
    )
    perturbation_rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
        feature_range = bounds[feature_index, 1] - bounds[feature_index, 0]
        for fraction in perturbation_fractions:
            for direction in (-1.0, 1.0):
                candidate = official_values.copy()
                candidate[feature_index] += direction * fraction * feature_range
                if not bounds[feature_index, 0] <= candidate[feature_index] <= bounds[
                    feature_index, 1
                ]:
                    continue
                energy, travel_time = _predict_candidates(
                    candidate.reshape(1, -1), models
                )
                perturbation_rows.append(
                    {
                        "varied_feature": feature_name,
                        "direction": "increase" if direction > 0 else "decrease",
                        "fraction_of_feature_range": fraction,
                        **{
                            column: float(candidate[index])
                            for index, column in enumerate(FEATURE_COLUMNS)
                        },
                        "predicted_energy_kwh": float(energy[0]),
                        "predicted_travel_time_s": float(travel_time[0]),
                        "energy_change_kwh": float(energy[0] - official_energy[0]),
                        "travel_time_change_s": float(travel_time[0] - official_time[0]),
                        "predicted_feasible_120s": bool(
                            travel_time[0] <= DEFAULT_TRAVEL_TIME_LIMIT_S
                        ),
                    }
                )
    perturbation_path = output_directory / "local_perturbations.csv"
    _write_csv(
        perturbation_path,
        (
            "varied_feature",
            "direction",
            "fraction_of_feature_range",
            *FEATURE_COLUMNS,
            "predicted_energy_kwh",
            "predicted_travel_time_s",
            "energy_change_kwh",
            "travel_time_change_s",
            "predicted_feasible_120s",
        ),
        perturbation_rows,
    )

    generator = np.random.default_rng(random_state)
    neighborhood_radius = 0.05 * (bounds[:, 1] - bounds[:, 0])
    neighborhood_lower = np.maximum(bounds[:, 0], official_values - neighborhood_radius)
    neighborhood_upper = np.minimum(bounds[:, 1], official_values + neighborhood_radius)
    joint_candidates = generator.uniform(
        neighborhood_lower,
        neighborhood_upper,
        size=(joint_sample_count, len(FEATURE_COLUMNS)),
    )
    joint_energy, joint_time = _predict_candidates(joint_candidates, models)
    joint_feasible = joint_time <= DEFAULT_TRAVEL_TIME_LIMIT_S

    conservative = optimize_surrogates(
        dataset_path,
        models_directory,
        output_directory / "conservative_119s",
        travel_time_limit_s=conservative_time_limit_s,
        sample_count=conservative_sample_count,
        random_state=random_state,
    )["best_candidate"]

    most_important = {
        target: min(
            (row for row in importance_rows if row["target"] == target),
            key=lambda row: row["rank"],
        )["feature"]
        for target in TARGET_COLUMNS
    }
    summary = {
        "verification_status": "surrogate_prediction_only",
        "official_time_limit_s": DEFAULT_TRAVEL_TIME_LIMIT_S,
        "conservative_time_limit_s": conservative_time_limit_s,
        "official_candidate": {
            **{column: float(official_values[index]) for index, column in enumerate(FEATURE_COLUMNS)},
            "predicted_energy_kwh": float(official_energy[0]),
            "predicted_travel_time_s": float(official_time[0]),
        },
        "conservative_candidate": conservative,
        "most_important_feature": most_important,
        "local_single_parameter_scenarios": len(perturbation_rows),
        "local_single_parameter_feasible_scenarios": sum(
            bool(row["predicted_feasible_120s"]) for row in perturbation_rows
        ),
        "joint_neighborhood": {
            "sample_count": joint_sample_count,
            "maximum_fraction_of_each_feature_range": 0.05,
            "parameter_bounds": {
                column: {
                    "minimum": float(neighborhood_lower[index]),
                    "maximum": float(neighborhood_upper[index]),
                }
                for index, column in enumerate(FEATURE_COLUMNS)
            },
            "predicted_feasible_count": int(joint_feasible.sum()),
            "predicted_feasible_percent": float(100.0 * joint_feasible.mean()),
            "predicted_energy_kwh_minimum": float(joint_energy.min()),
            "predicted_energy_kwh_median": float(np.median(joint_energy)),
            "predicted_energy_kwh_maximum": float(joint_energy.max()),
            "predicted_travel_time_s_minimum": float(joint_time.min()),
            "predicted_travel_time_s_median": float(np.median(joint_time)),
            "predicted_travel_time_s_maximum": float(joint_time.max()),
        },
    }
    summary_path = output_directory / "robustness_summary.json"
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "summary": summary,
        "importance_path": str(importance_path.resolve()),
        "sensitivity_path": str(sensitivity_path.resolve()),
        "perturbation_path": str(perturbation_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interpret surrogate models and analyze optimum robustness."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--models-directory", type=Path, default=DEFAULT_MODELS_DIRECTORY
    )
    parser.add_argument(
        "--official-result", type=Path, default=DEFAULT_OFFICIAL_RESULT_PATH
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    return parser


def main() -> None:
    arguments = _build_argument_parser().parse_args()
    result = analyze_models(
        arguments.dataset,
        arguments.models_directory,
        arguments.official_result,
        arguments.output_directory,
    )
    summary = result["summary"]
    conservative = summary["conservative_candidate"]
    print(
        "Most important features: "
        + ", ".join(
            f"{target}={feature}"
            for target, feature in summary["most_important_feature"].items()
        )
    )
    print(
        "Official-neighborhood predicted feasibility: "
        f"{summary['joint_neighborhood']['predicted_feasible_percent']:.2f}%"
    )
    print(
        "Conservative candidate: "
        f"{conservative['predicted_energy_kwh']:.6f} kWh, "
        f"{conservative['predicted_travel_time_s']:.6f} s"
    )
    print("Verification: surrogate prediction only")
    print(f"Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
