from __future__ import annotations

from typing import Any


def simulate(
    acceleration: float,
    deceleration: float,
    speed_limit_kmh: float,
    coasting_point: float,
    *,
    route_length: float = 2_000.0,
    mass: float = 200_000.0,
    time_step: float = 0.1,
    motor_efficiency: float = 0.90,
    resistance_a: float = 3_000.0,
    resistance_b: float = 30.0,
    resistance_c: float = 8.0,
    max_time: float = 1_200.0,
    position_tolerance: float = 1.0,
) -> dict[str, Any]:
    """Simulate a single train journey.

    Parameters
    ----------
    acceleration:
        Target net acceleration during the acceleration phase [m/s^2].
    deceleration:
        Maximum net deceleration magnitude during braking [m/s^2].
        This value must be positive.
    speed_limit_kmh:
        Operating speed limit [km/h].
    coasting_point:
        Position at which traction is removed and coasting begins [m].

    Returns
    -------
    dict
        Simulation results containing energy, travel time, stopping error,
        feasibility, and time histories used for plotting.
    """

    if acceleration <= 0:
        raise ValueError("acceleration must be positive.")
    if deceleration <= 0:
        raise ValueError("deceleration must be positive.")
    if speed_limit_kmh <= 0:
        raise ValueError("speed_limit_kmh must be positive.")
    if not 0 <= coasting_point <= route_length:
        raise ValueError("coasting_point must be within the route length.")
    if not 0 < motor_efficiency <= 1:
        raise ValueError("motor_efficiency must be between 0 and 1.")
    if time_step <= 0:
        raise ValueError("time_step must be positive.")
    if position_tolerance < 0:
        raise ValueError("position_tolerance cannot be negative.")

    speed_limit = speed_limit_kmh / 3.6

    time = 0.0
    position = 0.0
    speed = 0.0
    energy_joule = 0.0
    mode = "standstill"

    times = [time]
    positions = [position]
    speeds = [speed]
    powers = [0.0]
    energies_kwh = [0.0]
    modes = [mode]

    stopped_early = False
    braking_started = False

    while time < max_time and position < route_length:
        resistance = (
            resistance_a
            + resistance_b * speed
            + resistance_c * speed**2
        )

        remaining_distance = route_length - position
        stopping_distance = speed**2 / (2.0 * deceleration)
        braking_margin = speed * time_step
        traction_force = 0.0

        should_start_braking = (
            speed > 0
            and remaining_distance <= stopping_distance + braking_margin
        )

        # Once braking starts, remain in braking mode until the train stops.
        if braking_started or should_start_braking:
            braking_started = True
            mode = "braking"
            # Calculate the deceleration required to reach zero speed within
            # the remaining distance. The input value is treated as a maximum.
            required_deceleration = speed**2 / (2.0 * remaining_distance)
            applied_deceleration = min(deceleration, required_deceleration)
            net_acceleration = -applied_deceleration

        elif position >= coasting_point:
            mode = "coasting"
            net_acceleration = -resistance / mass

        elif speed < speed_limit:
            mode = "accelerating"
            net_acceleration = acceleration
            # Force required to overcome resistance and achieve the target
            # net acceleration.
            traction_force = mass * acceleration + resistance

        else:
            mode = "cruising"
            net_acceleration = 0.0
            # Only enough traction to overcome resistance is required to
            # maintain a constant speed.
            traction_force = resistance

        next_speed = speed + net_acceleration * time_step

        # Prevent the numerical time step from exceeding the speed limit.
        if mode == "accelerating" and next_speed > speed_limit:
            next_speed = speed_limit

        next_speed = max(0.0, next_speed)
        average_speed = 0.5 * (speed + next_speed)
        next_position = position + average_speed * time_step

        # If the final step overshoots the destination, end it at the target.
        if next_position > route_length and average_speed > 0:
            effective_dt = (route_length - position) / average_speed
            next_position = route_length
        else:
            effective_dt = time_step

        electrical_power = traction_force * average_speed / motor_efficiency
        energy_joule += electrical_power * effective_dt

        time += effective_dt
        position = next_position
        speed = next_speed

        times.append(time)
        positions.append(position)
        speeds.append(speed)
        powers.append(electrical_power)
        energies_kwh.append(energy_joule / 3.6e6)
        modes.append(mode)

        # If the train stops within the position tolerance, the journey is
        # complete. Otherwise, it is considered to have stopped too early.
        if speed <= 1e-9 and position < route_length and time > time_step:
            if route_length - position > position_tolerance:
                stopped_early = True
            break

    timed_out = time >= max_time and position < route_length
    final_position_error = route_length - position
    reached_destination = abs(final_position_error) <= position_tolerance
    stopped_at_destination = reached_destination and speed <= 0.5
    feasible = (
        reached_destination
        and stopped_at_destination
        and not stopped_early
        and not timed_out
    )

    return {
        "energy_kwh": energy_joule / 3.6e6,
        "travel_time_s": time,
        "final_position_m": position,
        "final_position_error_m": final_position_error,
        "final_speed_ms": speed,
        "max_speed_ms": max(speeds),
        "max_speed_kmh": max(speeds) * 3.6,
        "feasible": feasible,
        "stopped_early": stopped_early,
        "timed_out": timed_out,
        "history": {
            "time_s": times,
            "position_m": positions,
            "speed_ms": speeds,
            "power_w": powers,
            "energy_kwh": energies_kwh,
            "mode": modes,
        },
    }
