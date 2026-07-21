from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rsm-lab-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "rsm-lab-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_training import FEATURE_COLUMNS, MODEL_DISPLAY_NAMES, TARGET_COLUMNS


DEFAULT_RESULTS_DIRECTORY = Path("results/ml")
DEFAULT_FIGURES_DIRECTORY = Path("results/ml/figures")

MODEL_COLORS = {
    "linear_regression": "#0072B2",
    "polynomial_ridge": "#009E73",
    "random_forest": "#D55E00",
}
TARGET_LABELS = {
    "energy_kwh": "Energy [kWh]",
    "travel_time_s": "Travel time [s]",
}
FEATURE_LABELS = {
    "acceleration_ms2": "Acceleration",
    "deceleration_ms2": "Deceleration",
    "speed_limit_kmh": "Speed limit",
    "coasting_point_m": "Coasting point",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_model_evaluation(rows: list[dict[str, str]], path: Path) -> None:
    overall = [row for row in rows if row["held_out_feature"] == "all_features"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, target in zip(axes, TARGET_COLUMNS):
        target_rows = [row for row in overall if row["target"] == target]
        target_rows.sort(key=lambda row: float(row["rmse"]), reverse=True)
        labels = [MODEL_DISPLAY_NAMES[row["model"]] for row in target_rows]
        values = [float(row["rmse"]) for row in target_rows]
        colors = [MODEL_COLORS[row["model"]] for row in target_rows]
        bars = axis.barh(labels, values, color=colors)
        axis.bar_label(bars, fmt="%.3f", padding=4)
        axis.set_title(TARGET_LABELS[target])
        axis.set_xlabel("Structured-holdout RMSE")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Generalization to unseen parameter levels")
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_feature_importance(rows: list[dict[str, str]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, target in zip(axes, TARGET_COLUMNS):
        target_rows = [row for row in rows if row["target"] == target]
        target_rows.sort(key=lambda row: float(row["rmse_increase_mean"]))
        labels = [FEATURE_LABELS[row["feature"]] for row in target_rows]
        values = [float(row["rmse_increase_mean"]) for row in target_rows]
        errors = [float(row["rmse_increase_std"]) for row in target_rows]
        bars = axis.barh(labels, values, xerr=errors, color="#0072B2", alpha=0.85)
        axis.bar_label(bars, fmt="%.3f", padding=4)
        axis.set_title(TARGET_LABELS[target])
        axis.set_xlabel("RMSE increase after permutation")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Permutation importance on the held-out test set")
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_optimization_comparison(
    official_document: dict[str, Any],
    robustness_document: dict[str, Any],
    path: Path,
) -> None:
    baseline = official_document["best_observed_grid_baseline"]
    official = robustness_document["official_candidate"]
    conservative = robustness_document["conservative_candidate"]
    points = (
        (
            "Best observed grid",
            float(baseline["travel_time_s"]),
            float(baseline["energy_kwh"]),
            "#0072B2",
            "o",
        ),
        (
            "Official ML optimum",
            float(official["predicted_travel_time_s"]),
            float(official["predicted_energy_kwh"]),
            "#D55E00",
            "X",
        ),
        (
            "Conservative ML solution",
            float(conservative["predicted_travel_time_s"]),
            float(conservative["predicted_energy_kwh"]),
            "#009E73",
            "s",
        ),
    )

    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    annotation_positions = {
        "Best observed grid": (-10, 8, "right"),
        "Official ML optimum": (8, 8, "left"),
        "Conservative ML solution": (8, 8, "left"),
    }
    for label, travel_time, energy, color, marker in points:
        axis.scatter(
            travel_time,
            energy,
            s=115,
            color=color,
            marker=marker,
            label=label,
            zorder=3,
        )
        x_offset, y_offset, alignment = annotation_positions[label]
        axis.annotate(
            f"{energy:.2f} kWh, {travel_time:.2f} s",
            (travel_time, energy),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha=alignment,
        )
    axis.axvline(120.0, color="#555555", linestyle="--", linewidth=1.4)
    axis.text(
        120.0,
        axis.get_ylim()[1],
        "120 s limit ",
        va="top",
        ha="right",
    )
    axis.set_xlabel("Travel time [s]")
    axis.set_ylabel("Energy consumption [kWh]")
    axis.set_title("Observed baseline and ML-predicted solutions")
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_sensitivity(rows: list[dict[str, str]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for axis, feature in zip(axes.flat, FEATURE_COLUMNS):
        feature_rows = [row for row in rows if row["varied_feature"] == feature]
        x_values = [float(row["varied_value"]) for row in feature_rows]
        energy = [float(row["predicted_energy_kwh"]) for row in feature_rows]
        travel_time = [float(row["predicted_travel_time_s"]) for row in feature_rows]
        energy_line = axis.plot(
            x_values, energy, color="#0072B2", label="Energy [kWh]"
        )[0]
        axis.set_title(FEATURE_LABELS[feature])
        axis.set_xlabel(feature)
        axis.set_ylabel("Energy [kWh]", color="#0072B2")
        axis.tick_params(axis="y", labelcolor="#0072B2")
        axis.grid(alpha=0.2)
        axis.spines["top"].set_visible(False)

        time_axis = axis.twinx()
        time_line = time_axis.plot(
            x_values,
            travel_time,
            color="#D55E00",
            label="Travel time [s]",
        )[0]
        time_axis.axhline(120.0, color="#555555", linestyle="--", linewidth=1.0)
        time_axis.set_ylabel("Travel time [s]", color="#D55E00")
        time_axis.tick_params(axis="y", labelcolor="#D55E00")
        time_axis.spines["top"].set_visible(False)
        axis.legend(
            [energy_line, time_line],
            ["Energy", "Travel time"],
            loc="best",
            frameon=False,
        )
    figure.suptitle("One-parameter sensitivity around the official ML optimum")
    figure.tight_layout()
    _save_figure(figure, path)


def generate_report_figures(
    results_directory: Path | str = DEFAULT_RESULTS_DIRECTORY,
    figures_directory: Path | str = DEFAULT_FIGURES_DIRECTORY,
) -> dict[str, Path]:
    """Generate deterministic presentation figures from saved ML results."""

    results_directory = Path(results_directory)
    figures_directory = Path(figures_directory)
    structured_rows = _read_csv(
        results_directory / "structured_holdout" / "summary_metrics.csv"
    )
    importance_rows = _read_csv(
        results_directory / "robustness" / "permutation_importance.csv"
    )
    sensitivity_rows = _read_csv(
        results_directory / "robustness" / "sensitivity_curves.csv"
    )
    official_document = json.loads(
        (results_directory / "optimization" / "optimal_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    robustness_document = json.loads(
        (results_directory / "robustness" / "robustness_summary.json").read_text(
            encoding="utf-8"
        )
    )

    paths = {
        "model_evaluation": figures_directory / "model_evaluation.png",
        "feature_importance": figures_directory / "feature_importance.png",
        "optimization_comparison": figures_directory
        / "optimization_comparison.png",
        "sensitivity_analysis": figures_directory / "sensitivity_analysis.png",
    }
    _plot_model_evaluation(structured_rows, paths["model_evaluation"])
    _plot_feature_importance(importance_rows, paths["feature_importance"])
    _plot_optimization_comparison(
        official_document, robustness_document, paths["optimization_comparison"]
    )
    _plot_sensitivity(sensitivity_rows, paths["sensitivity_analysis"])
    return {name: path.resolve() for name, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate presentation-ready figures from saved ML results."
    )
    parser.add_argument(
        "--results-directory", type=Path, default=DEFAULT_RESULTS_DIRECTORY
    )
    parser.add_argument(
        "--figures-directory", type=Path, default=DEFAULT_FIGURES_DIRECTORY
    )
    arguments = parser.parse_args()
    paths = generate_report_figures(
        arguments.results_directory, arguments.figures_directory
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
