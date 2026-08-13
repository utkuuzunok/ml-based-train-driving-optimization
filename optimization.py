from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from ml_models import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODELS_DIRECTORY,
    DEFAULT_RANDOM_STATE,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    load_dataset,
)
from study_config import DEFAULT_OPTIMIZATION_OUTPUT_DIRECTORY, TRAVEL_TIME_LIMIT_S


DEFAULT_OUTPUT_DIRECTORY = DEFAULT_OPTIMIZATION_OUTPUT_DIRECTORY
DEFAULT_TRAVEL_TIME_LIMIT_S = TRAVEL_TIME_LIMIT_S
DEFAULT_SAMPLE_COUNT = 100_000
DEFAULT_TOP_RESULTS = 50


def _model_paths(models_directory: Path) -> dict[str, tuple[Path, Path]]:
    return {
        "energy_kwh": (
            models_directory / "energy_model.joblib",
            models_directory / "energy_model.metadata.json",
        ),
        "travel_time_s": (
            models_directory / "travel_time_model.joblib",
            models_directory / "travel_time_model.metadata.json",
        ),
    }


def load_surrogate_models(
    models_directory: Path | str = DEFAULT_MODELS_DIRECTORY,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], np.ndarray]:
    """Load both surrogate models and validate their shared feature contract."""

    models_directory = Path(models_directory)
    models: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    expected_bounds: np.ndarray | None = None

    for target, (model_path, metadata_path) in _model_paths(models_directory).items():
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing model or metadata for {target}.")
        target_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if target_metadata.get("target") != target:
            raise ValueError(f"Model metadata target mismatch for {target}.")
        if target_metadata.get("feature_columns") != list(FEATURE_COLUMNS):
            raise ValueError(f"Feature order mismatch for {target}.")

        try:
            target_bounds = np.asarray(
                [
                    [
                        target_metadata["feature_bounds"][column]["minimum"],
                        target_metadata["feature_bounds"][column]["maximum"],
                    ]
                    for column in FEATURE_COLUMNS
                ],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid feature bounds for {target}.") from error
        if target_bounds.shape != (len(FEATURE_COLUMNS), 2):
            raise ValueError(f"Invalid feature bounds for {target}.")
        if not np.all(np.isfinite(target_bounds)) or np.any(
            target_bounds[:, 0] >= target_bounds[:, 1]
        ):
            raise ValueError(f"Feature bounds must be finite and increasing for {target}.")
        if expected_bounds is not None and not np.array_equal(
            target_bounds, expected_bounds
        ):
            raise ValueError("Energy and travel-time model bounds do not match.")

        expected_bounds = target_bounds
        metadata[target] = target_metadata
        models[target] = joblib.load(model_path)

    if expected_bounds is None:
        raise ValueError("No model bounds were loaded.")
    return models, metadata, expected_bounds


def generate_latin_hypercube(
    sample_count: int,
    bounds: np.ndarray,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """Generate deterministic space-filling samples within continuous bounds."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (len(FEATURE_COLUMNS), 2):
        raise ValueError("bounds must contain one minimum/maximum pair per feature.")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("bounds must be finite and strictly increasing.")

    generator = np.random.default_rng(random_state)
    unit_samples = np.empty((sample_count, len(FEATURE_COLUMNS)), dtype=float)
    for feature_index in range(len(FEATURE_COLUMNS)):
        strata = generator.permutation(sample_count)
        unit_samples[:, feature_index] = (
            strata + generator.random(sample_count)
        ) / sample_count
    return bounds[:, 0] + unit_samples * (bounds[:, 1] - bounds[:, 0])


def rank_feasible_candidates(
    candidates: np.ndarray,
    predicted_energy: np.ndarray,
    predicted_travel_time: np.ndarray,
    *,
    travel_time_limit_s: float = DEFAULT_TRAVEL_TIME_LIMIT_S,
) -> list[int]:
    """Return feasible indices ordered by energy, time, then parameters."""

    candidates = np.asarray(candidates, dtype=float)
    predicted_energy = np.asarray(predicted_energy, dtype=float)
    predicted_travel_time = np.asarray(predicted_travel_time, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError("candidates must have exactly four feature columns.")
    if len(predicted_energy) != len(candidates) or len(predicted_travel_time) != len(
        candidates
    ):
        raise ValueError("Prediction lengths must match the candidate count.")
    if not math.isfinite(travel_time_limit_s) or travel_time_limit_s <= 0:
        raise ValueError("travel_time_limit_s must be finite and positive.")

    valid = (
        np.all(np.isfinite(candidates), axis=1)
        & np.isfinite(predicted_energy)
        & np.isfinite(predicted_travel_time)
        & (predicted_travel_time <= travel_time_limit_s)
    )
    feasible_indices = np.flatnonzero(valid)
    return sorted(
        (int(index) for index in feasible_indices),
        key=lambda index: (
            float(predicted_energy[index]),
            float(predicted_travel_time[index]),
            *(float(value) for value in candidates[index]),
        ),
    )


def _deduplicate_candidates(candidates: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(candidates, dtype=float), decimals=12)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return candidates[np.sort(indices)]


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


def optimize_surrogates(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    models_directory: Path | str = DEFAULT_MODELS_DIRECTORY,
    output_directory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
    *,
    travel_time_limit_s: float = DEFAULT_TRAVEL_TIME_LIMIT_S,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    top_results: int = DEFAULT_TOP_RESULTS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Rank space-filling candidates by predicted energy and travel time."""

    if top_results <= 0:
        raise ValueError("top_results must be positive.")
    if not math.isfinite(travel_time_limit_s) or travel_time_limit_s <= 0:
        raise ValueError("travel_time_limit_s must be finite and positive.")

    observed_features, observed_targets, _ = load_dataset(dataset_path)
    models, metadata, bounds = load_surrogate_models(models_directory)
    if np.any(observed_features < bounds[:, 0]) or np.any(
        observed_features > bounds[:, 1]
    ):
        raise ValueError("Dataset features fall outside the saved model bounds.")

    sampled_candidates = generate_latin_hypercube(
        sample_count, bounds, random_state=random_state
    )
    candidates = _deduplicate_candidates(
        np.vstack((observed_features, sampled_candidates))
    )
    energy_model = models["energy_kwh"]
    time_model = models["travel_time_s"]
    predicted_energy = np.asarray(energy_model.predict(candidates), dtype=float)
    predicted_time = np.asarray(time_model.predict(candidates), dtype=float)
    ranked_indices = rank_feasible_candidates(
        candidates,
        predicted_energy,
        predicted_time,
        travel_time_limit_s=travel_time_limit_s,
    )
    if not ranked_indices:
        raise RuntimeError("No predicted-feasible candidate was found.")

    energy_model_name = metadata["energy_kwh"]["selected_model"]
    time_model_name = metadata["travel_time_s"]["selected_model"]
    result_rows: list[dict[str, Any]] = []
    for rank, index in enumerate(ranked_indices[:top_results], start=1):
        result_rows.append(
            {
                "rank": rank,
                **{
                    column: float(candidates[index, feature_index])
                    for feature_index, column in enumerate(FEATURE_COLUMNS)
                },
                "predicted_energy_kwh": float(predicted_energy[index]),
                "predicted_travel_time_s": float(predicted_time[index]),
                "travel_time_margin_s": float(
                    travel_time_limit_s - predicted_time[index]
                ),
                "predicted_feasible": True,
                "energy_model": energy_model_name,
                "travel_time_model": time_model_name,
                "verification_status": "surrogate_prediction_only",
            }
        )

    observed_ranked = rank_feasible_candidates(
        observed_features,
        observed_targets["energy_kwh"],
        observed_targets["travel_time_s"],
        travel_time_limit_s=travel_time_limit_s,
    )
    if not observed_ranked:
        raise RuntimeError("The dataset contains no feasible observed baseline.")
    baseline_index = observed_ranked[0]
    observed_baseline = {
        **{
            column: float(observed_features[baseline_index, feature_index])
            for feature_index, column in enumerate(FEATURE_COLUMNS)
        },
        "energy_kwh": float(observed_targets["energy_kwh"][baseline_index]),
        "travel_time_s": float(observed_targets["travel_time_s"][baseline_index]),
    }

    best = result_rows[0]
    output_directory = Path(output_directory)
    candidate_path = output_directory / "optimization_candidates.csv"
    result_path = output_directory / "optimal_parameters.json"
    candidate_fields = (
        "rank",
        *FEATURE_COLUMNS,
        "predicted_energy_kwh",
        "predicted_travel_time_s",
        "travel_time_margin_s",
        "predicted_feasible",
        "energy_model",
        "travel_time_model",
        "verification_status",
    )
    _write_csv(candidate_path, candidate_fields, result_rows)
    result_document = {
        "objective": "minimize predicted energy consumption",
        "constraint": f"predicted travel_time_s <= {travel_time_limit_s}",
        "tie_breaker": "lower predicted travel time",
        "verification_status": "surrogate_prediction_only",
        "random_state": random_state,
        "latin_hypercube_samples": sample_count,
        "unique_candidates_evaluated": len(candidates),
        "feature_bounds": {
            column: {
                "minimum": float(bounds[index, 0]),
                "maximum": float(bounds[index, 1]),
            }
            for index, column in enumerate(FEATURE_COLUMNS)
        },
        "models": {
            target: metadata[target]["selected_model"] for target in TARGET_COLUMNS
        },
        "best_predicted_candidate": best,
        "best_observed_sample": observed_baseline,
        "predicted_energy_improvement_kwh": float(
            observed_baseline["energy_kwh"] - best["predicted_energy_kwh"]
        ),
        "predicted_energy_improvement_percent": float(
            100.0
            * (observed_baseline["energy_kwh"] - best["predicted_energy_kwh"])
            / observed_baseline["energy_kwh"]
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "best_candidate": best,
        "observed_baseline": observed_baseline,
        "candidate_path": str(candidate_path.resolve()),
        "result_path": str(result_path.resolve()),
        "unique_candidates_evaluated": len(candidates),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize energy using trained surrogate models only."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--models-directory", type=Path, default=DEFAULT_MODELS_DIRECTORY
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    return parser


def main() -> None:
    arguments = _build_argument_parser().parse_args()
    summary = optimize_surrogates(
        arguments.dataset,
        arguments.models_directory,
        arguments.output_directory,
        sample_count=arguments.sample_count,
    )
    best = summary["best_candidate"]
    print(f"Unique candidates evaluated: {summary['unique_candidates_evaluated']}")
    print(f"Predicted energy: {best['predicted_energy_kwh']:.6f} kWh")
    print(f"Predicted travel time: {best['predicted_travel_time_s']:.6f} s")
    for column in FEATURE_COLUMNS:
        print(f"{column}: {best[column]:.6f}")
    print("Verification: surrogate prediction only")
    print(f"Results: {summary['result_path']}")


if __name__ == "__main__":
    main()
