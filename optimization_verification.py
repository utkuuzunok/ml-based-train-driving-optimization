"""Verify surrogate-ranked optimization candidates with the simulator."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from optimization import DEFAULT_OUTPUT_DIRECTORY
from simulation_adapter import Simulator, run_feature_candidate
from study_config import FEATURE_COLUMNS, TRAVEL_TIME_LIMIT_S
from train_simulator import simulate


DEFAULT_CANDIDATES_PATH = DEFAULT_OUTPUT_DIRECTORY / "optimization_candidates.csv"
DEFAULT_VERIFICATION_DIRECTORY = DEFAULT_OUTPUT_DIRECTORY / "verification"
def verify_ranked_candidates(
    candidates_path: Path | str = DEFAULT_CANDIDATES_PATH,
    output_directory: Path | str = DEFAULT_VERIFICATION_DIRECTORY,
    *,
    simulator: Simulator = simulate,
    travel_time_limit_s: float = TRAVEL_TIME_LIMIT_S,
) -> dict[str, Any]:
    """Simulate every exported candidate and select the best valid result."""

    if not math.isfinite(travel_time_limit_s) or travel_time_limit_s <= 0:
        raise ValueError("travel_time_limit_s must be finite and positive.")

    candidates_path = Path(candidates_path)
    with candidates_path.open(encoding="utf-8", newline="") as csv_file:
        candidates = list(csv.DictReader(csv_file))
    if not candidates:
        raise ValueError("The candidate file is empty.")

    verified_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        values = {column: float(candidate[column]) for column in FEATURE_COLUMNS}
        result = run_feature_candidate(values, simulator=simulator)
        actual_energy = float(result["energy_kwh"])
        actual_time = float(result["travel_time_s"])
        simulator_feasible = bool(result["feasible"])
        verified_successful = bool(
            simulator_feasible and actual_time <= travel_time_limit_s
        )
        verified_rows.append(
            {
                "surrogate_rank": int(candidate["rank"]),
                **values,
                "predicted_energy_kwh": float(candidate["predicted_energy_kwh"]),
                "actual_energy_kwh": actual_energy,
                "energy_error_kwh": actual_energy
                - float(candidate["predicted_energy_kwh"]),
                "predicted_travel_time_s": float(
                    candidate["predicted_travel_time_s"]
                ),
                "actual_travel_time_s": actual_time,
                "travel_time_error_s": actual_time
                - float(candidate["predicted_travel_time_s"]),
                "final_position_error_m": float(
                    result["final_position_error_m"]
                ),
                "simulator_feasible": simulator_feasible,
                "verified_successful": verified_successful,
            }
        )

    successful = [row for row in verified_rows if row["verified_successful"]]
    if not successful:
        raise RuntimeError("No surrogate candidate passed simulator verification.")
    best = min(
        successful,
        key=lambda row: (
            row["actual_energy_kwh"],
            row["actual_travel_time_s"],
            *(row[column] for column in FEATURE_COLUMNS),
        ),
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows_path = output_directory / "verified_candidates.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=tuple(verified_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(verified_rows)

    summary = {
        "verification_status": "simulator_verified",
        "travel_time_limit_s": travel_time_limit_s,
        "candidates_checked": len(verified_rows),
        "verified_successful_candidates": len(successful),
        "best_verified_candidate": best,
    }
    summary_path = output_directory / "best_verified_ml_solution.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **summary,
        "rows_path": str(rows_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify surrogate optimization candidates in the simulator."
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_VERIFICATION_DIRECTORY
    )
    arguments = parser.parse_args()
    summary = verify_ranked_candidates(
        arguments.candidates,
        arguments.output_directory,
    )
    best = summary["best_verified_candidate"]
    print(
        "Verified candidates: "
        f"{summary['verified_successful_candidates']}/"
        f"{summary['candidates_checked']}"
    )
    print(f"Energy: {best['actual_energy_kwh']:.6f} kWh")
    print(f"Travel time: {best['actual_travel_time_s']:.6f} s")
    print(f"Results: {summary['summary_path']}")


if __name__ == "__main__":
    main()
