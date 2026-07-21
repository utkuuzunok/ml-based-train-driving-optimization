# Energy-Efficient Train Driving Optimization

This project studies how four driving parameters affect train energy consumption
and travel time:

- acceleration,
- deceleration,
- speed limit, and
- coasting point.

The current workflow uses a physics-based simulator to create a fixed dataset,
compares regression models, and searches the trained surrogate models for a
low-energy operating strategy subject to a travel-time limit of 120 seconds.
Optimization results are model predictions until they are checked with the
simulator or a physical system.

## Project structure

| File | Responsibility |
|---|---|
| `train_simulator.py` | Train motion, stopping logic, energy, and physical feasibility |
| `plotting.py` | Visualization of one simulation history |
| `main.py` | Run and print one example scenario |
| `experiments.py` | Generate parameter combinations and export simulation results |
| `ml_training.py` | Train and evaluate Linear Regression, Polynomial Ridge, and Random Forest models |
| `ml_evaluation.py` | Evaluate unseen parameter levels with structured holdouts |
| `optimization.py` | Perform constrained continuous optimization using saved surrogate models |
| `model_analysis.py` | Calculate permutation importance, sensitivity, and robustness |
| `reporting.py` | Generate presentation-ready figures from saved results |

Simulation, visualization, single-scenario execution, batch experiments,
machine-learning evaluation, and optimization remain separate responsibilities.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproducible workflow

The repository already contains the frozen 875-row dataset and lightweight
result files. The complete pipeline can be reproduced in this order:

```bash
python ml_training.py
python ml_evaluation.py
python optimization.py
python model_analysis.py
python reporting.py
python -m unittest discover -s tests -v
```

`ml_evaluation.py` refits the structured-holdout winners on all 875 rows and
saves the local model binaries needed by the later commands. Model binaries are
excluded from Git because they can be regenerated.

To regenerate the simulation dataset deliberately:

```bash
python experiments.py --output results/train_experiments.csv
```

Dataset regeneration is not required for ordinary ML analysis.

## Dataset

The full Cartesian grid contains 875 unique combinations:

| Parameter | Values |
|---|---|
| Acceleration | 0.6 to 1.0 m/s² in 0.1 m/s² increments |
| Deceleration | 0.7 to 1.1 m/s² in 0.1 m/s² increments |
| Speed limit | 60 to 100 km/h in 10 km/h increments |
| Coasting point | 1000 to 1600 m in 100 m increments |

All 875 simulations completed and met the simulator's physical stopping
criteria. Of these, 485 also met the official travel-time condition
`travel_time_s <= 120`.

The four parameter increments describe the dataset resolution, not permanent
physical restrictions. Continuous ML optimization is restricted to the minimum
and maximum values represented in the dataset.

## Model training and evaluation

Inputs are limited to the four driving parameters. Two separate targets are
predicted:

- energy consumption in kWh,
- travel time in seconds.

The compared models are Linear Regression, degree-two Polynomial Ridge
Regression, and Random Forest. Reported metrics are MAE, RMSE, and R².

The normal random split produced very high Random Forest accuracy, but that
split places neighboring grid combinations in both training and testing.
Structured evaluation therefore removes one complete parameter level at a time.
It contains 22 folds per target and tests both interpolation and boundary
generalization.

### Structured-holdout results

| Target | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Energy | Linear Regression | 0.4715 kWh | 0.5518 kWh | 0.9911 |
| Energy | **Polynomial Ridge** | **0.0998 kWh** | **0.1286 kWh** | **0.9995** |
| Energy | Random Forest | 1.0625 kWh | 2.0114 kWh | 0.8824 |
| Travel time | Linear Regression | 3.1014 s | 3.6992 s | 0.9202 |
| Travel time | **Polynomial Ridge** | **1.3130 s** | **1.7840 s** | **0.9814** |
| Travel time | Random Forest | 3.3748 s | 5.4855 s | 0.8245 |

Polynomial Ridge is used for continuous optimization because it generalized
best to unseen parameter levels.

## Optimization results

The objective is to minimize predicted energy while enforcing
`predicted travel_time_s <= 120`. Lower predicted travel time is the tie-breaker.
The optimizer combines 100,000 deterministic Latin Hypercube samples, the
original dataset combinations, and constrained local refinement.

| Result | Acceleration | Deceleration | Speed limit | Coasting | Energy | Time |
|---|---:|---:|---:|---:|---:|---:|
| Best observed grid | 0.6000 | 1.1000 | 80.0000 | 1000 m | 17.3125 kWh | 119.8111 s |
| Official ML optimum | 1.0000 | 1.1000 | 73.6383 | 1000 m | 14.9764 kWh | 120.0000 s |
| Conservative ML solution | 1.0000 | 1.1000 | 74.5352 | 1000 m | 15.3209 kWh | 119.0000 s |

The official ML optimum predicts a 2.3361 kWh or 13.49% reduction relative to
the best observed grid combination. This is a predicted improvement rather than
a simulator-verified energy saving.

## Interpretation and robustness

Held-out permutation importance identifies speed limit as the most influential
input for both energy and travel time. Coasting point ranks second for energy,
while acceleration ranks second for travel time.

The official optimum lies exactly on the predicted 120-second boundary and is
sensitive to small parameter changes:

- 4 of 10 one-parameter perturbations remained predicted-feasible;
- 47.405% of 20,000 nearby in-bounds samples remained predicted-feasible;
- the conservative solution uses 0.3446 kWh more predicted energy but provides
  a one-second predicted margin.

## Figures

Generated figures are stored under `results/ml/figures/`:

- `model_evaluation.png`
- `feature_importance.png`
- `optimization_comparison.png`
- `sensitivity_analysis.png`

## Engineering limitations

- The dataset comes from one simulator configuration, so the models inherit its
  assumptions and any modeling errors.
- Structured-holdout error is more representative than the random-split error,
  but it does not prove performance outside the current parameter bounds.
- The two outputs are modeled independently, so their prediction errors are not
  coupled probabilistically.
- The official optimum is on an active constraint and should not be treated as
  operationally robust.
- No uncertainty interval, real-world measurement noise, route variation,
  passenger-load variation, adhesion variation, or controller quantization is
  currently represented.
- Final engineering acceptance requires simulator or physical verification of
  selected candidates when that validation becomes part of the project scope.
