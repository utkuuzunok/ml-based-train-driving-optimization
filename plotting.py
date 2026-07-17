from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


def plot_results(result: dict[str, Any]) -> None:
    """Display speed, power, energy, and driving mode plots."""

    history = result["history"]
    time_s = history["time_s"]
    position_m = history["position_m"]
    speed_kmh = [speed * 3.6 for speed in history["speed_ms"]]
    power_kw = [power / 1_000.0 for power in history["power_w"]]

    mode_values = {
        "standstill": 0,
        "accelerating": 1,
        "cruising": 2,
        "coasting": 3,
        "braking": 4,
    }
    mode_numbers = [mode_values[mode] for mode in history["mode"]]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(position_m, speed_kmh, color="tab:blue")
    axes[0, 0].set_title("Position vs. Speed")
    axes[0, 0].set_xlabel("Position [m]")
    axes[0, 0].set_ylabel("Speed [km/h]")

    axes[0, 1].plot(time_s, power_kw, color="tab:red")
    axes[0, 1].set_title("Time vs. Traction Power")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Electrical power [kW]")

    axes[1, 0].plot(time_s, history["energy_kwh"], color="tab:green")
    axes[1, 0].set_title("Time vs. Cumulative Energy")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Energy [kWh]")

    axes[1, 1].step(position_m, mode_numbers, where="post", color="tab:purple")
    axes[1, 1].set_title("Position vs. Driving Mode")
    axes[1, 1].set_xlabel("Position [m]")
    axes[1, 1].set_ylabel("Driving mode")
    axes[1, 1].set_yticks(
        list(mode_values.values()),
        list(mode_values.keys()),
    )

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)

    figure.suptitle(
        f"Energy: {result['energy_kwh']:.2f} kWh | "
        f"Travel time: {result['travel_time_s']:.1f} s"
    )
    figure.tight_layout()
    plt.show()