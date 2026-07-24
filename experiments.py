from __future__ import annotations

import argparse
import csv
import itertools
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from train_simulator import simulate


ACCELERATIONS_MS2 = (0.6, 0.7, 0.8, 0.9, 1.0)
DECELERATIONS_MS2 = (0.7, 0.8, 0.9, 1.0, 1.1)
SPEED_LIMITS_KMH = (60.0, 70.0, 80.0, 90.0, 100.0)
COASTING_POINTS_M = (
    1_000.0,
    1_100.0,
    1_200.0,
    1_300.0,
    1_400.0,
    1_500.0,
    1_600.0,
)
TRAVEL_TIME_LIMIT_S = 120.0
DEFAULT_OUTPUT_PATH = Path("results/train_experiments.csv")

CSV_COLUMNS = (
    "acceleration_ms2",
    "deceleration_ms2",
    "speed_limit_kmh",
    "coasting_point_m",
    "energy_kwh",
    "travel_time_s",
    "max_speed_kmh",
    "final_position_error_m",
    "final_speed_ms",
    "input_valid",
    "simulation_completed",
    "feasible",
    "meets_time_constraint",
    "successful",
    "stopped_early",
    "overshot_destination",
    "timed_out",
    "error_type",
    "error_message",
)

EXPECTED_COMBINATION_COUNT = (
    len(ACCELERATIONS_MS2)
    * len(DECELERATIONS_MS2)
    * len(SPEED_LIMITS_KMH)
    * len(COASTING_POINTS_M)
)

ExperimentRow = dict[str, Any]
Simulator = Callable[..., dict[str, Any]]
ParameterCombination = tuple[float, float, float, float]


def generate_parameter_combinations(
    accelerations: Sequence[float] = ACCELERATIONS_MS2,
    decelerations: Sequence[float] = DECELERATIONS_MS2,
    speed_limits: Sequence[float] = SPEED_LIMITS_KMH,
    coasting_points: Sequence[float] = COASTING_POINTS_M,
) -> Iterable[ParameterCombination]:
    """Generate the Cartesian product in a deterministic column order."""

    return itertools.product(
        accelerations,
        decelerations,
        speed_limits,
        coasting_points,
    )


def _base_row(
    acceleration: float,
    deceleration: float,
    speed_limit_kmh: float,
    coasting_point: float,
) -> ExperimentRow:
    """Create a complete row whose result fields represent a failed run."""

    return {
        "acceleration_ms2": acceleration,
        "deceleration_ms2": deceleration,
        "speed_limit_kmh": speed_limit_kmh,
        "coasting_point_m": coasting_point,
        "energy_kwh": None,
        "travel_time_s": None,
        "max_speed_kmh": None,
        "final_position_error_m": None,
        "final_speed_ms": None,
        "input_valid": True,
        "simulation_completed": False,
        "feasible": False,
        "meets_time_constraint": False,
        "successful": False,
        "stopped_early": False,
        "overshot_destination": False,
        "timed_out": False,
        "error_type": "",
        "error_message": "",
    }


def run_single_experiment(
    acceleration: float,
    deceleration: float,
    speed_limit_kmh: float,
    coasting_point: float,
    *,
    simulator: Simulator = simulate,
    travel_time_limit_s: float = TRAVEL_TIME_LIMIT_S,
) -> ExperimentRow:
    """Run one combination and flatten its scalar metrics into a CSV row."""

    if not math.isfinite(travel_time_limit_s) or travel_time_limit_s <= 0:
        raise ValueError("travel_time_limit_s must be finite and positive.")

    row = _base_row(
        acceleration,
        deceleration,
        speed_limit_kmh,
        coasting_point,
    )

    try:
        result = simulator(
            acceleration=acceleration,
            deceleration=deceleration,
            speed_limit_kmh=speed_limit_kmh,
            coasting_point=coasting_point,
        )
        energy_kwh = float(result["energy_kwh"])
        travel_time_s = float(result["travel_time_s"])
        max_speed_kmh = float(result["max_speed_kmh"])
        final_position_error_m = float(result["final_position_error_m"])
        final_speed_ms = float(result["final_speed_ms"])
        feasible = bool(result["feasible"])
        stopped_early = bool(result["stopped_early"])
        overshot_destination = bool(result["overshot_destination"])
        timed_out = bool(result["timed_out"])
    except ValueError as error:
        row["input_valid"] = False
        row["error_type"] = type(error).__name__
        row["error_message"] = str(error)
        return row
    except Exception as error:
        row["error_type"] = type(error).__name__
        row["error_message"] = str(error)
        return row

    meets_time_constraint = travel_time_s <= travel_time_limit_s
    row.update(
        {
            "energy_kwh": energy_kwh,
            "travel_time_s": travel_time_s,
            "max_speed_kmh": max_speed_kmh,
            "final_position_error_m": final_position_error_m,
            "final_speed_ms": final_speed_ms,
            "simulation_completed": True,
            "feasible": feasible,
            "meets_time_constraint": meets_time_constraint,
            "successful": feasible and meets_time_constraint,
            "stopped_early": stopped_early,
            "overshot_destination": overshot_destination,
            "timed_out": timed_out,
        }
    )
    return row


def run_experiments(
    *,
    accelerations: Sequence[float] = ACCELERATIONS_MS2,
    decelerations: Sequence[float] = DECELERATIONS_MS2,
    speed_limits: Sequence[float] = SPEED_LIMITS_KMH,
    coasting_points: Sequence[float] = COASTING_POINTS_M,
    simulator: Simulator = simulate,
    travel_time_limit_s: float = TRAVEL_TIME_LIMIT_S,
) -> list[ExperimentRow]:
    """Run a parameter grid while preserving failed combinations as rows."""

    rows = []
    for acceleration, deceleration, speed_limit, coasting_point in (
        generate_parameter_combinations(
            accelerations,
            decelerations,
            speed_limits,
            coasting_points,
        )
    ):
        rows.append(
            run_single_experiment(
                acceleration,
                deceleration,
                speed_limit,
                coasting_point,
                simulator=simulator,
                travel_time_limit_s=travel_time_limit_s,
            )
        )
    return rows


def export_results_csv(
    rows: Iterable[ExperimentRow],
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Write experiment rows to CSV and return the resolved output path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path.resolve()


def summarize_results(
    rows: Sequence[ExperimentRow],
    *,
    elapsed_time_s: float,
    travel_time_limit_s: float = TRAVEL_TIME_LIMIT_S,
) -> dict[str, Any]:
    """Calculate compact batch diagnostics without retaining histories."""

    completed = [row for row in rows if row["simulation_completed"]]
    feasible = [row for row in completed if row["feasible"]]
    successful = [row for row in completed if row["successful"]]
    validation_failures = [row for row in rows if not row["input_valid"]]
    unexpected_failures = [
        row
        for row in rows
        if row["input_valid"] and not row["simulation_completed"]
    ]

    successful_energies = [row["energy_kwh"] for row in successful]
    successful_times = [row["travel_time_s"] for row in successful]

    return {
        "total_combinations": len(rows),
        "completed_simulations": len(completed),
        "feasible_simulations": len(feasible),
        "successful_simulations": len(successful),
        "validation_failures": len(validation_failures),
        "unexpected_failures": len(unexpected_failures),
        "success_rate_percent": (
            100.0 * len(successful) / len(rows) if rows else 0.0
        ),
        "minimum_energy_kwh": (
            min(successful_energies) if successful_energies else None
        ),
        "maximum_energy_kwh": (
            max(successful_energies) if successful_energies else None
        ),
        "minimum_travel_time_s": (
            min(successful_times) if successful_times else None
        ),
        "maximum_travel_time_s": (
            max(successful_times) if successful_times else None
        ),
        "travel_time_limit_s": travel_time_limit_s,
        "elapsed_time_s": elapsed_time_s,
    }


def print_summary(summary: dict[str, Any], output_path: Path) -> None:
    """Print the information needed to audit a completed experiment run."""

    print(f"Dataset path          : {output_path}")
    print(f"Total combinations    : {summary['total_combinations']}")
    print(f"Completed simulations : {summary['completed_simulations']}")
    print(f"Feasible simulations  : {summary['feasible_simulations']}")
    print(
        f"Successful (<= {summary['travel_time_limit_s']:g} s) : "
        f"{summary['successful_simulations']}"
    )
    print(f"Validation failures   : {summary['validation_failures']}")
    print(f"Unexpected failures   : {summary['unexpected_failures']}")
    print(f"Success rate          : {summary['success_rate_percent']:.1f}%")
    print(f"Elapsed time          : {summary['elapsed_time_s']:.2f} s")

    if summary["minimum_energy_kwh"] is not None:
        print(
            "Successful energy    : "
            f"{summary['minimum_energy_kwh']:.3f}–"
            f"{summary['maximum_energy_kwh']:.3f} kWh"
        )
        print(
            "Successful time      : "
            f"{summary['minimum_travel_time_s']:.2f}–"
            f"{summary['maximum_travel_time_s']:.2f} s"
        )


def main() -> None:
    """Run the approved full grid when this module is invoked explicitly."""

    parser = argparse.ArgumentParser(
        description="Generate the train-simulation experiment dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV output path (default: results/train_experiments.csv).",
    )
    arguments = parser.parse_args()

    started_at = time.perf_counter()
    rows = run_experiments()
    elapsed_time_s = time.perf_counter() - started_at
    output_path = export_results_csv(rows, arguments.output)
    summary = summarize_results(rows, elapsed_time_s=elapsed_time_s)
    print_summary(summary, output_path)


if __name__ == "__main__":
    main()
