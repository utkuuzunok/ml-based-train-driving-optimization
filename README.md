# ML-Based Energy-Efficient Train Driving Optimization

This project optimizes four continuous train-driving parameters:

- acceleration,
- service deceleration,
- speed limit, and
- coasting point.

The objective is to minimize simulated traction energy while completing a
2,000 m journey, stopping at the station, and satisfying
`travel_time_s <= 150.0`. A machine-learning surrogate proposes candidates;
the train simulator verifies them. A separate Differential Evolution search
provides an independent simulator-based reference.

## Project structure

| File | Responsibility |
|---|---|
| `study_config.py` | Shared parameter ranges, limits, and output paths |
| `train_simulator.py` | Train motion, stopping logic, energy, and feasibility |
| `simulation_adapter.py` | Translation between ML features and simulator inputs |
| `experiments.py` | Reproducible Latin Hypercube sampling, batch simulation, and CSV export |
| `ml_models.py` | Dataset loading, model definitions, and fitted-model persistence |
| `ml_evaluation.py` | Repeated K-fold evaluation, model selection, and final fitting |
| `optimization.py` | Rank continuous surrogate candidates without calling the simulator |
| `optimization_verification.py` | Verify ML candidates in the simulator |
| `direct_optimization.py` | Multi-seed Differential Evolution on the simulator |
| `coasting_analysis.py` | Controlled optimized-coasting versus no-coasting comparison |
| `model_analysis.py` | Optional model importance, sensitivity, and robustness analysis |
| `reporting.py` | Optional figure generation from saved results |

Simulation, dataset generation, ML prediction, candidate verification, direct
optimization, and reporting remain separate. This makes prediction errors
explicit and keeps future extensions from silently changing the simulator.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproducible workflow

Run the pipeline in this order:

```bash
python experiments.py
python ml_evaluation.py
python optimization.py
python optimization_verification.py
python direct_optimization.py
python coasting_analysis.py
python model_analysis.py
python reporting.py
python -m unittest discover -s tests -v
```

## Experiment space

The dataset contains 4,500 deterministic parameter combinations:

| Parameter | Sampling and optimization bounds |
|---|---|
| Acceleration | 0.4–1.2 m/s² |
| Deceleration | 0.5–1.1 m/s² |
| Speed limit | 50–110 km/h |
| Coasting point | 100–1,800 m |

The design consists of 4,483 Latin Hypercube samples, all 16 boundary corners,
and one reference operating point. A fixed random seed makes the 4,500
simulations reproducible.

Dataset: `results/train_experiments.csv`

- rows: 4,500 unique combinations;
- completed and physically feasible: 4,500;
- satisfy `travel_time_s <= 150.0`: 3,678;
- missing or non-finite result values: 0.

## Machine-learning models

The four driving parameters are used to predict two targets independently:

- energy consumption in kWh;
- travel time in seconds.

The compared regressors are Linear Regression, degree-two Polynomial Ridge,
and Random Forest. Metrics are MAE, RMSE, and R². Final model selection uses
deterministic 5-fold cross-validation repeated three times and chooses the
lowest pooled RMSE, with MAE as the tie-breaker. The complete comparison is:

| Target | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Energy | Linear Regression | 2.09048 kWh | 2.98815 kWh | 0.82866 |
| Energy | Polynomial Ridge | 1.38922 kWh | 1.89341 kWh | 0.93121 |
| Energy | **Random Forest** | **0.18584 kWh** | **0.37915 kWh** | **0.99724** |
| Travel time | Linear Regression | 8.78669 s | 14.59270 s | 0.61354 |
| Travel time | Polynomial Ridge | 6.25811 s | 10.56048 s | 0.79761 |
| Travel time | **Random Forest** | **1.19784 s** | **2.65842 s** | **0.98717** |

Random Forest was therefore selected from the three candidates; it was not
fixed in advance. The earlier grid-based study selected Polynomial Ridge under
a leave-one-parameter-level-out test. That result is not directly comparable:
the current LHS design contains almost entirely unique continuous values, so
holding out one exact level would usually leave only one row. Repeated K-fold
instead measures interpolation among space-filling samples within the defined
bounds.

Random K-fold splits can still be optimistic about completely unseen regions.
Accordingly, the reported accuracy is interpreted as in-domain interpolation,
not evidence of extrapolation or transfer to another train or route. Boundary
corners, simulator verification, the one-sided safety margin, and the direct DE
reference provide additional checks on the final optimization result.

The surrogate optimizer evaluates the observed dataset together with 100,000
deterministic Latin Hypercube candidates. These outputs remain predictions
until `optimization_verification.py` runs them through `simulate()`.

For robustness analysis, the travel-time safety margin is calculated from the
selected model's pooled repeated-CV residuals. The residual is defined as
`actual - predicted`, so positive values represent optimistic time predictions.
The one-sided 95th percentile is 2.87621 s, giving a conservative surrogate
limit of 147.12379 s instead of an arbitrary one-second margin. This empirical
margin is a model-risk heuristic, not a formal real-world coverage guarantee.

## Verified optimization results

| Result | Energy | Travel time | Status |
|---|---:|---:|---|
| Simulator-verified ML candidate | 8.32660 kWh | 149.93853 s | Feasible |
| Simulator-verified conservative ML candidate | 8.73069 kWh | 146.23760 s | Feasible |
| Multi-seed direct DE reference | 8.24112 kWh | 149.99910 s | Feasible |

The verified ML candidate is approximately 1.04% above the best direct DE
reference energy. This demonstrates that surrogate optimization can produce a
near-reference candidate; it does not establish ML superiority or prove a
global optimum.

All 50 candidates generated under the data-driven conservative limit satisfied
the official 150-second condition when checked with the simulator.

Five independent DE runs all found feasible candidates. Their relative energy
spread was approximately 0.004%, providing a reproducibility check within the
defined bounds. No SLSQP refinement is used.

The controlled coasting analysis keeps the official 150-second constraint and
the configured metro operating bounds fixed. It compares the best direct-DE
solution with an independently optimized strategy whose coasting point is fixed
at the 2,000 m route end, so there is no intentional coasting before braking.
Both finalists are rerun in the simulator, and the script exports their speed,
energy, and travel-time comparison figure.

| Directly optimized strategy | Energy | Travel time |
|---|---:|---:|
| Optimized coasting | 8.24112 kWh | 149.99910 s |
| No intentional coasting | 9.53793 kWh | 150.00000 s |

Under this controlled simulator comparison, optimized coasting reduces
traction energy by 1.29680 kWh, or approximately 13.60%, relative to the
independently optimized no-coasting strategy. This is a simulator-specific
comparison under the configured assumptions, not a field energy-saving claim.

## Physical assumptions and limitations

- The model represents one level, 2,000 m route and one fixed train setup.
- Energy is integrated from positive traction power using a fixed motor
  efficiency.
- Regenerative braking and auxiliary consumption are not modeled.
- Passenger load, gradients, curves, adhesion, voltage variation, and
  controller quantization are not varied.
- Acceleration and service-deceleration upper bounds are treated as technical
  metro operating constraints, not domains to expand automatically when a
  constrained optimum touches a bound.
- ML validation measures agreement with this simulator, not with a physical
  railway system.
- Every reported ML operating candidate must be checked in the simulator.
- If the fixed train or route parameters change, the pipeline must generate a
  new LHS dataset and retrain the models; a model trained for one scenario is
  not assumed to generalize to another.

## License

This project is available under the [MIT License](LICENSE).
