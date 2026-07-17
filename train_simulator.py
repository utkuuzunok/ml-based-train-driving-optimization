from __future__ import annotations

import math
from typing import Any


_NUMERICAL_EPSILON = 1e-12


def _time_to_position(
    distance: float,
    speed: float,
    acceleration: float,
) -> float:
    """Return the positive time needed to cover a distance."""

    if abs(acceleration) <= _NUMERICAL_EPSILON:
        return distance / speed

    discriminant = speed**2 + 2.0 * acceleration * distance
    return (-speed + math.sqrt(max(0.0, discriminant))) / acceleration


def _validate_inputs(
    acceleration: float,
    deceleration: float,
    speed_limit_kmh: float,
    coasting_point: float,
    route_length: float,
    mass: float,
    time_step: float,
    motor_efficiency: float,
    resistance_a: float,
    resistance_b: float,
    resistance_c: float,
    max_time: float,
    position_tolerance: float,
) -> None:
    """Validate all public simulator inputs before integration starts."""

    values = {
        "acceleration": acceleration,
        "deceleration": deceleration,
        "speed_limit_kmh": speed_limit_kmh,
        "coasting_point": coasting_point,
        "route_length": route_length,
        "mass": mass,
        "time_step": time_step,
        "motor_efficiency": motor_efficiency,
        "resistance_a": resistance_a,
        "resistance_b": resistance_b,
        "resistance_c": resistance_c,
        "max_time": max_time,
        "position_tolerance": position_tolerance,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if acceleration <= 0:
        raise ValueError("acceleration must be positive.")
    if deceleration <= 0:
        raise ValueError("deceleration must be positive.")
    if speed_limit_kmh <= 0:
        raise ValueError("speed_limit_kmh must be positive.")
    if route_length <= 0:
        raise ValueError("route_length must be positive.")
    if not 0 <= coasting_point <= route_length:
        raise ValueError("coasting_point must be within the route length.")
    if mass <= 0:
        raise ValueError("mass must be positive.")
    if time_step <= 0:
        raise ValueError("time_step must be positive.")
    if not 0 < motor_efficiency <= 1:
        raise ValueError("motor_efficiency must be between 0 and 1.")
    if resistance_a < 0 or resistance_b < 0 or resistance_c < 0:
        raise ValueError("resistance coefficients cannot be negative.")
    if max_time <= 0:
        raise ValueError("max_time must be positive.")
    if position_tolerance < 0:
        raise ValueError("position_tolerance cannot be negative.")


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
    """Simulate one train journey through the train's final stop.

    ``final_position_error_m`` is the signed station alignment error. Positive
    values mean that the train stopped before the target; negative values mean
    that it overshot the target.

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
        Simulation results containing energy, travel time, signed stopping
        error, feasibility, and time histories used for plotting.
    """

    _validate_inputs(
        acceleration,
        deceleration,
        speed_limit_kmh,
        coasting_point,
        route_length,
        mass,
        time_step,
        motor_efficiency,
        resistance_a,
        resistance_b,
        resistance_c,
        max_time,
        position_tolerance,
    )

    speed_limit = speed_limit_kmh / 3.6

    time = 0.0
    position = 0.0
    speed = 0.0
    energy_joule = 0.0

    times = [time]
    positions = [position]
    speeds = [speed]
    powers = [0.0]
    energies_kwh = [0.0]
    modes = ["standstill"]

    braking_started = False
    stopped = False

    while time < max_time - _NUMERICAL_EPSILON and not stopped:
        step_duration = min(time_step, max_time - time)
        resistance = (
            resistance_a
            + resistance_b * speed
            + resistance_c * speed**2
        )
        remaining_distance = route_length - position
        stopping_distance = speed**2 / (2.0 * deceleration)
        braking_margin = speed * step_duration
        should_start_braking = (
            speed > _NUMERICAL_EPSILON
            and remaining_distance <= stopping_distance + braking_margin
        )

        traction_force = 0.0
        if braking_started or should_start_braking:
            braking_started = True
            mode = "braking"
            if remaining_distance > _NUMERICAL_EPSILON:
                required_deceleration = speed**2 / (2.0 * remaining_distance)
                applied_deceleration = min(
                    deceleration,
                    required_deceleration,
                )
            else:
                applied_deceleration = deceleration
            net_acceleration = -applied_deceleration
        elif position >= coasting_point - _NUMERICAL_EPSILON:
            mode = "coasting"
            net_acceleration = -resistance / mass
        elif speed < speed_limit - _NUMERICAL_EPSILON:
            mode = "accelerating"
            net_acceleration = acceleration
            traction_force = mass * net_acceleration + resistance
        else:
            mode = "cruising"
            speed = speed_limit
            net_acceleration = 0.0
            traction_force = resistance

        # A train at rest cannot move while coasting or braking. This makes an
        # input such as coasting_point=0 a completed but infeasible journey.
        if speed <= _NUMERICAL_EPSILON and net_acceleration <= 0:
            speed = 0.0
            stopped = True
            break

        effective_dt = step_duration

        # End an acceleration sub-step at the exact speed-limit event. The next
        # sub-step will correctly use cruising force instead of full traction.
        if mode == "accelerating":
            time_to_speed_limit = (speed_limit - speed) / net_acceleration
            if _NUMERICAL_EPSILON < time_to_speed_limit < effective_dt:
                effective_dt = time_to_speed_limit

        # Likewise, end the traction/cruise sub-step exactly where coasting is
        # requested rather than up to one full time step after that position.
        if mode in {"accelerating", "cruising"}:
            distance_to_coasting = coasting_point - position
            if distance_to_coasting > _NUMERICAL_EPSILON:
                candidate_distance = (
                    speed * effective_dt
                    + 0.5 * net_acceleration * effective_dt**2
                )
                if candidate_distance > distance_to_coasting:
                    time_to_coasting = _time_to_position(
                        distance_to_coasting,
                        speed,
                        net_acceleration,
                    )
                    if _NUMERICAL_EPSILON < time_to_coasting < effective_dt:
                        effective_dt = time_to_coasting

        # Stop at the exact zero-speed event instead of integrating negative
        # speed for the unused remainder of the nominal time step.
        if net_acceleration < 0:
            time_to_stop = -speed / net_acceleration
            if _NUMERICAL_EPSILON < time_to_stop < effective_dt:
                effective_dt = time_to_stop

        next_speed = speed + net_acceleration * effective_dt
        if mode == "accelerating" and next_speed >= speed_limit:
            next_speed = speed_limit
        if next_speed <= _NUMERICAL_EPSILON:
            next_speed = 0.0

        next_position = (
            position
            + speed * effective_dt
            + 0.5 * net_acceleration * effective_dt**2
        )
        average_speed = 0.5 * (speed + next_speed)
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

        if speed == 0.0:
            stopped = True

    timed_out = not stopped and time >= max_time - _NUMERICAL_EPSILON
    final_position_error = route_length - position
    stopped_early = stopped and final_position_error > position_tolerance
    overshot_destination = final_position_error < -position_tolerance
    stopped_at_destination = (
        stopped and abs(final_position_error) <= position_tolerance
    )
    feasible = stopped_at_destination and not timed_out

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
        "overshot_destination": overshot_destination,
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
