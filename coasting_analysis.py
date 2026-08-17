"""Compare optimized coasting with an optimized no-coasting strategy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rsm-lab-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "rsm-lab-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution

from direct_optimization import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_SEEDS,
)
from simulation_adapter import Simulator, run_feature_candidate
from study_config import (
    DEFAULT_COASTING_COMPARISON_OUTPUT_DIRECTORY,
    DEFAULT_DIRECT_OPTIMIZATION_OUTPUT_DIRECTORY,
    DEFAULT_ML_RESULTS_DIRECTORY,
    FEATURE_COLUMNS,
    PARAMETER_BOUNDS,
    ROUTE_LENGTH_M,
    TRAVEL_TIME_LIMIT_S,
)
from train_simulator import simulate


DEFAULT_OUTPUT_DIRECTORY = DEFAULT_COASTING_COMPARISON_OUTPUT_DIRECTORY
DEFAULT_ENABLED_SOLUTION_PATH = (
    DEFAULT_DIRECT_OPTIMIZATION_OUTPUT_DIRECTORY / "best_direct_solution.json"
)
DEFAULT_FIGURE_PATH = DEFAULT_ML_RESULTS_DIRECTORY / "figures" / "coasting_comparison.png"
NO_COASTING_POINT_M = ROUTE_LENGTH_M
VARIABLE_COLUMNS = FEATURE_COLUMNS[:3]


def _no_coasting_bounds() -> list[tuple[float, float]]:
    return [PARAMETER_BOUNDS[column] for column in VARIABLE_COLUMNS]


def run_no_coasting_de(
    *,
    simulator: Simulator = simulate,
    seed: int,
    travel_time_limit_s: float = TRAVEL_TIME_LIMIT_S,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    population_size: int = DEFAULT_POPULATION_SIZE,
) -> dict[str, Any]:
    """Optimize acceleration, deceleration, and speed with coasting disabled."""

    if not math.isfinite(travel_time_limit_s) or travel_time_limit_s <= 0:
        raise ValueError("travel_time_limit_s must be finite and positive.")
    if max_iterations <= 0 or population_size <= 0:
        raise ValueError("max_iterations and population_size must be positive.")

    cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def evaluate(values: np.ndarray) -> dict[str, Any]:
        key = tuple(float(value) for value in values)
        if key not in cache:
            features = {
                **{
                    column: float(value)
                    for column, value in zip(VARIABLE_COLUMNS, values)
                },
                "coasting_point_m": NO_COASTING_POINT_M,
            }
            try:
                result = run_feature_candidate(features, simulator=simulator)
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
        _no_coasting_bounds(),
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
            for column, value in zip(VARIABLE_COLUMNS, scipy_result.x)
        },
        "coasting_point_m": NO_COASTING_POINT_M,
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


def optimize_no_coasting(
    *,
    simulator: Simulator = simulate,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    travel_time_limit_s: float = TRAVEL_TIME_LIMIT_S,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    population_size: int = DEFAULT_POPULATION_SIZE,
) -> dict[str, Any]:
    """Run reproducible no-coasting searches and select the best feasible run."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty sequence of unique values.")
    runs = [
        run_no_coasting_de(
            simulator=simulator,
            seed=int(seed),
            travel_time_limit_s=travel_time_limit_s,
            max_iterations=max_iterations,
            population_size=population_size,
        )
        for seed in seeds
    ]
    successful = [run for run in runs if run["successful"]]
    if not successful:
        raise RuntimeError("No no-coasting run found a feasible solution.")
    successful.sort(
        key=lambda run: (
            run["energy_kwh"],
            run["travel_time_s"],
            *(run[column] for column in VARIABLE_COLUMNS),
        )
    )
    return {"best_candidate": successful[0], "runs": runs}


def _load_enabled_candidate(path: Path | str) -> dict[str, float]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    candidate = document.get("best_candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Enabled-coasting result has no best candidate.")
    required = (*FEATURE_COLUMNS, "energy_kwh", "travel_time_s")
    if any(key not in candidate for key in required):
        raise ValueError("Enabled-coasting candidate is incomplete.")
    return {key: float(candidate[key]) for key in required}


def _verified_candidate(
    candidate: dict[str, float], *, simulator: Simulator
) -> tuple[dict[str, float | bool], dict[str, Any]]:
    features = {column: float(candidate[column]) for column in FEATURE_COLUMNS}
    result = run_feature_candidate(features, simulator=simulator)
    verified = {
        **features,
        "energy_kwh": float(result["energy_kwh"]),
        "travel_time_s": float(result["travel_time_s"]),
        "final_position_error_m": float(result["final_position_error_m"]),
        "feasible": bool(result["feasible"]),
    }
    return verified, result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=tuple(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(
    enabled: dict[str, float | bool],
    enabled_result: dict[str, Any],
    disabled: dict[str, float | bool],
    disabled_result: dict[str, Any],
    path: Path,
) -> None:
    strategies = (
        ("Optimized coasting", enabled, enabled_result, "#0072B2"),
        ("No intentional coasting", disabled, disabled_result, "#D55E00"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    for label, _, result, color in strategies:
        history = result["history"]
        positions = np.asarray(history["position_m"], dtype=float)
        speeds = 3.6 * np.asarray(history["speed_ms"], dtype=float)
        energies = np.asarray(history["energy_kwh"], dtype=float)
        axes[0, 0].plot(positions, speeds, label=label, color=color, linewidth=2)
        axes[0, 1].plot(positions, energies, label=label, color=color, linewidth=2)

    axes[0, 0].set_title("Speed profiles")
    axes[0, 0].set_xlabel("Position [m]")
    axes[0, 0].set_ylabel("Speed [km/h]")
    axes[0, 1].set_title("Cumulative traction energy")
    axes[0, 1].set_xlabel("Position [m]")
    axes[0, 1].set_ylabel("Energy [kWh]")

    labels = [strategy[0] for strategy in strategies]
    colors = [strategy[3] for strategy in strategies]
    energy_values = [float(strategy[1]["energy_kwh"]) for strategy in strategies]
    time_values = [float(strategy[1]["travel_time_s"]) for strategy in strategies]
    energy_bars = axes[1, 0].bar(labels, energy_values, color=colors)
    time_bars = axes[1, 1].bar(labels, time_values, color=colors)
    axes[1, 0].bar_label(energy_bars, fmt="%.3f kWh", padding=4)
    axes[1, 1].bar_label(time_bars, fmt="%.3f s", padding=4)
    axes[1, 0].set_title("Simulator-verified energy")
    axes[1, 0].set_ylabel("Energy [kWh]")
    axes[1, 1].set_title("Simulator-verified travel time")
    axes[1, 1].set_ylabel("Travel time [s]")
    axes[1, 1].axhline(
        TRAVEL_TIME_LIMIT_S,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"{TRAVEL_TIME_LIMIT_S:g} s limit",
    )
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    axes[1, 0].tick_params(axis="x", rotation=8)
    axes[1, 1].tick_params(axis="x", rotation=8)
    figure.suptitle("Effect of optimized coasting under the 150 s constraint")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def compare_coasting_strategies(
    enabled_solution_path: Path | str = DEFAULT_ENABLED_SOLUTION_PATH,
    output_directory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
    figure_path: Path | str = DEFAULT_FIGURE_PATH,
    *,
    simulator: Simulator = simulate,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    travel_time_limit_s: float = TRAVEL_TIME_LIMIT_S,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    population_size: int = DEFAULT_POPULATION_SIZE,
) -> dict[str, Any]:
    """Compare optimized enabled and disabled coasting with simulator outputs."""

    enabled_saved = _load_enabled_candidate(enabled_solution_path)
    enabled, enabled_result = _verified_candidate(enabled_saved, simulator=simulator)
    if not enabled["feasible"] or enabled["travel_time_s"] > travel_time_limit_s:
        raise RuntimeError("Enabled-coasting reference is not simulator-feasible.")

    disabled_search = optimize_no_coasting(
        simulator=simulator,
        seeds=seeds,
        travel_time_limit_s=travel_time_limit_s,
        max_iterations=max_iterations,
        population_size=population_size,
    )
    disabled, disabled_result = _verified_candidate(
        disabled_search["best_candidate"], simulator=simulator
    )
    if not disabled["feasible"] or disabled["travel_time_s"] > travel_time_limit_s:
        raise RuntimeError("No-coasting finalist failed simulator verification.")

    energy_saving_kwh = float(disabled["energy_kwh"]) - float(
        enabled["energy_kwh"]
    )
    energy_saving_percent = 100.0 * energy_saving_kwh / float(
        disabled["energy_kwh"]
    )
    successful_runs = [
        run for run in disabled_search["runs"] if run["successful"]
    ]
    no_coasting_energy_spread_kwh = max(
        run["energy_kwh"] for run in successful_runs
    ) - min(run["energy_kwh"] for run in successful_runs)
    comparison_rows = [
        {
            "strategy": "optimized_coasting",
            **enabled,
            "travel_time_limit_s": travel_time_limit_s,
            "verification_status": "simulator_verified",
        },
        {
            "strategy": "no_intentional_coasting",
            **disabled,
            "travel_time_limit_s": travel_time_limit_s,
            "verification_status": "simulator_verified",
        },
    ]

    output_directory = Path(output_directory)
    figure_path = Path(figure_path)
    runs_path = output_directory / "no_coasting_de_runs.csv"
    comparison_path = output_directory / "coasting_comparison.csv"
    summary_path = output_directory / "coasting_comparison.json"
    _write_csv(runs_path, disabled_search["runs"])
    _write_csv(comparison_path, comparison_rows)
    summary = {
        "method": "controlled direct-simulation coasting comparison",
        "travel_time_limit_s": travel_time_limit_s,
        "parameter_bounds": PARAMETER_BOUNDS,
        "physical_bounds_treated_as_hard_constraints": True,
        "no_coasting_definition": (
            "coasting point fixed at the route end; traction/cruising continues "
            "until the braking controller starts"
        ),
        "seeds": [int(seed) for seed in seeds],
        "no_coasting_successful_runs": len(successful_runs),
        "no_coasting_total_runs": len(disabled_search["runs"]),
        "no_coasting_energy_spread_kwh": no_coasting_energy_spread_kwh,
        "no_coasting_relative_energy_spread": (
            no_coasting_energy_spread_kwh / float(disabled["energy_kwh"])
        ),
        "optimized_coasting": enabled,
        "no_intentional_coasting": disabled,
        "energy_saving_kwh": energy_saving_kwh,
        "energy_saving_percent": energy_saving_percent,
        "verification_status": "simulator_verified",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_comparison(
        enabled, enabled_result, disabled, disabled_result, figure_path
    )
    return {
        **summary,
        "runs_path": str(runs_path.resolve()),
        "comparison_path": str(comparison_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "figure_path": str(figure_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare optimized coasting with a no-coasting strategy."
    )
    parser.add_argument(
        "--enabled-solution", type=Path, default=DEFAULT_ENABLED_SOLUTION_PATH
    )
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    arguments = parser.parse_args()
    result = compare_coasting_strategies(
        arguments.enabled_solution,
        arguments.output_directory,
        arguments.figure,
        max_iterations=arguments.max_iterations,
    )
    print(f"Optimized coasting energy: {result['optimized_coasting']['energy_kwh']:.6f} kWh")
    print(
        "No intentional coasting energy: "
        f"{result['no_intentional_coasting']['energy_kwh']:.6f} kWh"
    )
    print(f"Energy saving: {result['energy_saving_percent']:.3f}%")
    print(f"Results: {result['summary_path']}")
    print(f"Figure: {result['figure_path']}")


if __name__ == "__main__":
    main()
