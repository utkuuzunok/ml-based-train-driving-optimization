from typing import Any

from plotting import plot_results
from train_simulator import simulate


def print_summary(result: dict[str, Any]) -> None:
    """Print the simulation result in a readable format."""

    print(f"Energy               : {result['energy_kwh']:.3f} kWh")
    print(f"Travel time          : {result['travel_time_s']:.1f} s")
    print(f"Maximum speed        : {result['max_speed_kmh']:.1f} km/h")
    print(f"Final position error : {result['final_position_error_m']:.2f} m")
    print(f"Final speed          : {result['final_speed_ms']:.2f} m/s")
    print(f"Feasible             : {result['feasible']}")


def main() -> None:
    """Run the example scenario and display its results."""

    result = simulate(
        acceleration=0.8,
        deceleration=0.9,
        speed_limit_kmh=80.0,
        coasting_point=1_300.0,
    )
    print_summary(result)


if __name__ == "__main__":
    main()