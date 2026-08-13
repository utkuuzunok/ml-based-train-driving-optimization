"""Directly optimize the train simulator with Differential Evolution."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution

from simulation_adapter import Simulator, feature_mapping, run_feature_candidate
from study_config import (
    DEFAULT_DIRECT_OPTIMIZATION_OUTPUT_DIRECTORY,
    FEATURE_COLUMNS,
    PARAMETER_BOUNDS,
    TRAVEL_TIME_LIMIT_S,
)
from train_simulator import simulate


DEFAULT_SEEDS = (11, 29, 47, 71, 101)
DEFAULT_MAX_ITERATIONS = 200
DEFAULT_POPULATION_SIZE = 15
ENERGY_CONVERGENCE_RELATIVE_TOLERANCE = 0.001

def _bounds() -> list[tuple[float, float]]:
    return [PARAMETER_BOUNDS[column] for column in FEATURE_COLUMNS]


def run_differential_evolution(
    *,
    simulator: Simulator = simulate,
    seed: int,
    travel_time_limit_s: float = TRAVEL_TIME_LIMIT_S,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    population_size: int = DEFAULT_POPULATION_SIZE,
) -> dict[str, Any]:
    """Run one reproducible, constrained Differential Evolution search."""

    if not math.isfinite(travel_time_limit_s) or travel_time_limit_s <= 0:
        raise ValueError("travel_time_limit_s must be finite and positive.")
    if max_iterations <= 0 or population_size <= 0:
        raise ValueError("max_iterations and population_size must be positive.")

    cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def evaluate(values: np.ndarray) -> dict[str, Any]:
        key = tuple(float(value) for value in values)
        if key not in cache:
            try:
                result = run_feature_candidate(
                    feature_mapping(values),
                    simulator=simulator,
                )
                cache[key] = {
                    "energy_kwh": float(result["energy_kwh"]),
                    "travel_time_s": float(result["travel_time_s"]),
                    "feasible": bool(result["feasible"]),
                    "final_position_error_m": float(
                        result["final_position_error_m"]
                    ),
                }
            except (ValueError, ArithmeticError, OverflowError):
                cache[key] = {
                    "energy_kwh": math.inf,
                    "travel_time_s": math.inf,
                    "feasible": False,
                    "final_position_error_m": math.inf,
                }
        return cache[key]

    def objective(values: np.ndarray) -> float:
        result = evaluate(values)
        if not result["feasible"] or not math.isfinite(result["energy_kwh"]):
            return 1.0e12
        return float(result["energy_kwh"])

    def travel_time(values: np.ndarray) -> float:
        result = evaluate(values)
        if not result["feasible"] or not math.isfinite(result["travel_time_s"]):
            return 1.0e12
        return float(result["travel_time_s"])

    scipy_result = differential_evolution(
        objective,
        _bounds(),
        constraints=NonlinearConstraint(
            travel_time,
            -math.inf,
            travel_time_limit_s,
        ),
        seed=seed,
        maxiter=max_iterations,
        popsize=population_size,
        polish=False,
        workers=1,
        updating="immediate",
        tol=1.0e-7,
        atol=1.0e-9,
    )
    final = evaluate(np.asarray(scipy_result.x, dtype=float))
    successful = bool(
        final["feasible"]
        and final["travel_time_s"] <= travel_time_limit_s
        and math.isfinite(final["energy_kwh"])
    )
    return {
        "seed": seed,
        **{
            column: float(value)
            for column, value in zip(FEATURE_COLUMNS, scipy_result.x)
        },
        "energy_kwh": final["energy_kwh"],
        "travel_time_s": final["travel_time_s"],
        "final_position_error_m": final["final_position_error_m"],
        "successful": successful,
        "optimizer_success": bool(scipy_result.success),
        "optimizer_message": str(scipy_result.message),
        "iterations": int(scipy_result.nit),
        "objective_evaluations": int(scipy_result.nfev),
        "simulator_calls": len(cache),
    }


def optimize_direct_simulator(
    output_directory: Path | str = DEFAULT_DIRECT_OPTIMIZATION_OUTPUT_DIRECTORY,
    *,
    simulator: Simulator = simulate,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    travel_time_limit_s: float = TRAVEL_TIME_LIMIT_S,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    population_size: int = DEFAULT_POPULATION_SIZE,
) -> dict[str, Any]:
    """Run multiple DE searches and export their convergence evidence."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty sequence of unique values.")

    runs = [
        run_differential_evolution(
            simulator=simulator,
            seed=int(seed),
            travel_time_limit_s=travel_time_limit_s,
            max_iterations=max_iterations,
            population_size=population_size,
        )
        for seed in seeds
    ]
    successful_runs = [run for run in runs if run["successful"]]
    if not successful_runs:
        raise RuntimeError("No direct optimization run found a feasible solution.")

    ranked = sorted(
        successful_runs,
        key=lambda run: (
            run["energy_kwh"],
            run["travel_time_s"],
            *(run[column] for column in FEATURE_COLUMNS),
        ),
    )
    best = ranked[0]
    energy_spread = max(run["energy_kwh"] for run in successful_runs) - min(
        run["energy_kwh"] for run in successful_runs
    )
    relative_spread = energy_spread / best["energy_kwh"]
    converged_across_seeds = bool(
        len(successful_runs) == len(runs)
        and relative_spread <= ENERGY_CONVERGENCE_RELATIVE_TOLERANCE
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    runs_path = output_directory / "de_runs.csv"
    with runs_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=tuple(runs[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(runs)

    summary = {
        "method": "multi-seed constrained Differential Evolution",
        "objective": "minimize simulator energy consumption",
        "travel_time_limit_s": travel_time_limit_s,
        "parameter_bounds": PARAMETER_BOUNDS,
        "seeds": list(seeds),
        "successful_runs": len(successful_runs),
        "total_runs": len(runs),
        "energy_spread_kwh": energy_spread,
        "relative_energy_spread": relative_spread,
        "relative_energy_tolerance": ENERGY_CONVERGENCE_RELATIVE_TOLERANCE,
        "converged_across_seeds": converged_across_seeds,
        "total_simulator_calls": sum(run["simulator_calls"] for run in runs),
        "best_candidate": best,
    }
    summary_path = output_directory / "best_direct_solution.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **summary,
        "runs_path": str(runs_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize the train simulator with Differential Evolution."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_DIRECT_OPTIMIZATION_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS
    )
    arguments = parser.parse_args()
    summary = optimize_direct_simulator(
        arguments.output_directory,
        max_iterations=arguments.max_iterations,
    )
    best = summary["best_candidate"]
    print(f"Successful runs: {summary['successful_runs']}/{summary['total_runs']}")
    print(f"Converged across seeds: {summary['converged_across_seeds']}")
    print(f"Energy: {best['energy_kwh']:.6f} kWh")
    print(f"Travel time: {best['travel_time_s']:.6f} s")
    for column in FEATURE_COLUMNS:
        print(f"{column}: {best[column]:.6f}")
    print(f"Simulator calls: {summary['total_simulator_calls']}")
    print(f"Results: {summary['summary_path']}")


if __name__ == "__main__":
    main()
