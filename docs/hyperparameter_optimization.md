# Hyperparameter Optimization Guide

This guide covers Bayesian hyperparameter optimization for the Minecraft speedrun simulator using **Ax/BoTorch** - a local, free alternative to cloud-based sweep services.

## Table of Contents

1. [Why Ax/BoTorch?](#why-axbotorch)
2. [How Bayesian Optimization Works](#how-bayesian-optimization-works)
3. [Installation](#installation)
4. [Configuration Reference](#configuration-reference)
5. [Running Sweeps](#running-sweeps)
6. [Understanding Results](#understanding-results)
7. [Advanced: Multi-Objective Optimization](#advanced-multi-objective-optimization)
8. [Comparison with Other Tools](#comparison-with-other-tools)

---

## Why Ax/BoTorch?

| Feature | Ax/BoTorch | W&B Sweeps | Optuna |
|---------|------------|------------|--------|
| **Cost** | Free | $50+/month | Free |
| **Runs locally** | ✅ | ❌ (cloud) | ✅ |
| **Bayesian optimization** | ✅ GP + EI | ✅ GP | ✅ TPE |
| **Multi-objective** | ✅ | ❌ | ✅ |
| **Parallel trials** | ✅ | ✅ | ✅ |
| **PyTorch integration** | Native | Plugin | Plugin |
| **Meta/FAIR backing** | ✅ | ❌ | ❌ |

**Ax** (Adaptive Experimentation) is Meta's platform for Bayesian optimization, built on **BoTorch** which uses PyTorch for GPU-accelerated Gaussian Process inference.

---

## How Bayesian Optimization Works

### The Problem

Grid search and random search are inefficient:
- **Grid search**: Exponential in dimensions (10 params × 5 values = 10M trials)
- **Random search**: Better but still wastes trials on poor regions

### The Solution: Surrogate Model + Acquisition Function

Bayesian optimization uses two components:

#### 1. Gaussian Process (GP) Surrogate

A GP models the unknown objective function `f(params) → metric`:

```
              f(params) ~ GP(μ(params), k(params, params'))
```

- **μ(params)**: Mean function (prior belief about metric)
- **k(params, params')**: Covariance function (how similar are two points?)

After observing `N` trials, the GP provides:
- **Posterior mean**: Best estimate of `f(params)` at any point
- **Posterior variance**: Uncertainty about that estimate

#### 2. Expected Improvement (EI) Acquisition

EI balances **exploitation** (try params where mean is high) vs **exploration** (try params where variance is high):

```
              EI(params) = E[max(f(params) - f_best, 0)]
```

In plain English: "How much do we expect to improve over our current best?"

### The Algorithm

```
1. Initialize: Run N random trials (Sobol quasi-random for coverage)
2. Fit GP: Train Gaussian Process on observed (params, metric) pairs
3. Optimize EI: Find params that maximize Expected Improvement
4. Evaluate: Run trial with those params, observe metric
5. Update: Add new observation to GP
6. Repeat: Go to step 2 until budget exhausted
```

### Visual Intuition

```
Metric
  ^
  |     *                    ← observed trials
  |    /|\        ?          ← GP mean (solid) ± uncertainty (shaded)
  |   / | \      /|\
  |  /  |  \    / | \
  | *   |   \  /  |  *
  |     |    \/   |
  +-----+----+----+-------→ Params
        ↑
        Next trial here (high EI: uncertain region near good results)
```

---

## Installation

```bash
# Install Ax (includes BoTorch)
pip install ax-platform

# Or with GPU support for faster GP inference
pip install ax-platform botorch gpytorch

# Verify installation
python -c "from ax.service.ax_client import AxClient; print('Ax OK')"
```

**Dependencies:**
- Python 3.9+
- PyTorch 2.0+
- NumPy, SciPy

---

## Configuration Reference

The sweep config is in `configs/ax_sweep_config.yaml`:

```yaml
experiment:
  name: minecraft_speedrun_ppo      # Experiment identifier
  objective: eval/success_rate      # Metric to optimize
  minimize: false                   # false = maximize

parameters:
  # Continuous parameters with log scale
  - name: learning_rate
    type: range                     # Continuous range
    bounds: [1e-5, 1e-3]           # [min, max]
    log_scale: true                 # Sample in log space
    value_type: float

  # Discrete choice parameters
  - name: n_steps
    type: choice                    # Discrete choices
    values: [64, 128, 256, 512]
    value_type: int

  # Continuous without log scale
  - name: gamma
    type: range
    bounds: [0.95, 0.999]
    value_type: float

generation_strategy:
  steps:
    # Phase 1: Quasi-random exploration
    - model: SOBOL
      num_trials: 10               # 10 random trials first

    # Phase 2: Bayesian optimization
    - model: BOTORCH_MODULAR
      num_trials: -1               # Run until budget exhausted

scheduler:
  total_trials: 50                 # Total trial budget
  max_pending_trials: 1            # Parallel trials (set to GPU count)

defaults:
  vf_coef: 0.5                     # Fixed (not swept)
  max_grad_norm: 0.5
```

### Parameter Types

| Type | Use Case | Example |
|------|----------|---------|
| `range` | Continuous values | learning_rate: [1e-5, 1e-3] |
| `choice` | Discrete options | batch_size: [64, 128, 256] |
| `fixed` | Constant value | vf_coef: 0.5 |

### Log Scale

Use `log_scale: true` when:
- Parameter spans orders of magnitude (learning rate, entropy coefficient)
- Smaller values need finer resolution

```yaml
# Without log_scale: samples uniformly in [1e-5, 1e-3]
#   → 99.9% of samples are in [1e-4, 1e-3]
#   → Only 0.1% explore [1e-5, 1e-4]

# With log_scale: samples uniformly in log space
#   → 50% in [1e-5, 3e-4], 50% in [3e-4, 1e-3]
#   → Equal exploration across magnitudes
```

---

## Running Sweeps

### Basic Usage

```bash
cd contrib/minecraft_sim

# Run 50-trial sweep
python examples/run_ax_sweep.py --trials 50

# Use custom config
python examples/run_ax_sweep.py --config configs/my_sweep.yaml --trials 100
```

### Parallel Trials (Multi-GPU)

```bash
# 4 parallel trials for 4 GPUs
python examples/run_ax_sweep.py --trials 100 --parallel 4
```

Each trial runs on a separate GPU:

```python
# In train_evaluate(), set device based on trial
import os
gpu_id = trial_index % num_gpus
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
```

### Resuming Sweeps

Ax automatically saves state. To resume:

```python
from ax.service.ax_client import AxClient

# Load existing experiment
ax_client = AxClient.load_from_json_file("ax_sweep_results.json")

# Continue where you left off
for i in range(additional_trials):
    params, trial_idx = ax_client.get_next_trial()
    metrics = train_evaluate(params)
    ax_client.complete_trial(trial_idx, metrics)
```

---

## Understanding Results

### Output File: `ax_sweep_results.json`

```json
{
  "experiment": {
    "name": "minecraft_speedrun_ppo",
    "trials": [
      {
        "index": 0,
        "parameters": {"learning_rate": 0.0003, "n_steps": 128, ...},
        "metrics": {"eval/success_rate": 0.23}
      },
      ...
    ]
  },
  "best_parameters": {"learning_rate": 0.00015, ...},
  "best_value": 0.67
}
```

### Analyzing Results

```python
from ax.service.ax_client import AxClient
import matplotlib.pyplot as plt

# Load results
ax_client = AxClient.load_from_json_file("ax_sweep_results.json")

# Get best parameters
best_params, best_metrics = ax_client.get_best_parameters()
print(f"Best: {best_params}")
print(f"Success rate: {best_metrics['eval/success_rate']:.2%}")

# Plot optimization trace
trials = ax_client.experiment.trials
metrics = [t.objective_mean for t in trials.values()]
plt.plot(metrics)
plt.xlabel("Trial")
plt.ylabel("Success Rate")
plt.title("Optimization Progress")
plt.savefig("sweep_progress.png")

# Get parameter importance (sensitivity analysis)
from ax.analysis.sensitivity_analysis import compute_sensitivity
sensitivities = compute_sensitivity(ax_client.experiment)
for param, importance in sensitivities.items():
    print(f"{param}: {importance:.3f}")
```

### Typical Output

```
Starting Ax sweep: 50 trials, 1 parallel

=== Trial 1/50 ===
Parameters: {'learning_rate': 0.000342, 'n_steps': 256, 'batch_size': 128, ...}
Result: {'eval/success_rate': 0.12}

=== Trial 2/50 ===
Parameters: {'learning_rate': 0.000089, 'n_steps': 64, 'batch_size': 256, ...}
Result: {'eval/success_rate': 0.08}

... (trials 3-49) ...

=== Trial 50/50 ===
Parameters: {'learning_rate': 0.000156, 'n_steps': 128, 'batch_size': 256, ...}
Result: {'eval/success_rate': 0.71}

============================================================
BEST PARAMETERS:
  learning_rate: 0.000156
  n_steps: 128
  batch_size: 256
  n_epochs: 4
  gamma: 0.9934
  gae_lambda: 0.947
  ent_coef: 0.0089
  clip_range: 0.2

Best eval/success_rate: 0.71
```

---

## Advanced: Multi-Objective Optimization

Optimize multiple metrics simultaneously (e.g., success rate AND training speed):

```python
from ax.service.ax_client import AxClient, ObjectiveProperties

ax_client = AxClient()
ax_client.create_experiment(
    name="minecraft_multi_obj",
    parameters=config["parameters"],
    objectives={
        "eval/success_rate": ObjectiveProperties(minimize=False),
        "train/time_seconds": ObjectiveProperties(minimize=True),
    },
)

# Returns Pareto-optimal solutions
pareto_frontier = ax_client.get_pareto_optimal_parameters()
```

This finds the **Pareto frontier** - solutions where you can't improve one metric without hurting another.

---

## Comparison with Other Tools

### vs. W&B Sweeps

**W&B Sweeps:**
```yaml
# wandb sweep config
method: bayes  # Just a declaration!
metric:
  name: eval/success_rate
  goal: maximize
```

The `method: bayes` in W&B is a **flag** - the actual GP/BO runs on W&B's servers. You need:
- Internet connection
- W&B account
- Paid tier for heavy usage

**Ax/BoTorch:**
- Everything runs locally
- Full control over GP kernel, acquisition function
- No usage limits

### vs. Optuna

**Optuna:**
- Uses TPE (Tree-Parzen Estimator) by default
- Lighter weight, faster startup
- Less accurate than GP for expensive evaluations

**Ax/BoTorch:**
- Full GP with BoTorch/GPyTorch
- Better sample efficiency (fewer trials needed)
- GPU-accelerated GP inference
- Better for expensive training runs

### Recommendation

| Scenario | Tool |
|----------|------|
| Quick experiments, many trials | Optuna |
| Expensive training, few trials | Ax/BoTorch |
| Need cloud logging | W&B (logging) + Ax (sweep) |
| Multi-objective | Ax/BoTorch |

---

## Troubleshooting

### "No module named 'ax'"

```bash
pip install ax-platform
```

### GP fitting fails (singular matrix)

The GP covariance matrix is ill-conditioned. Solutions:
1. Add more initial Sobol trials
2. Reduce parameter bounds
3. Add jitter: `ax_client.create_experiment(..., generation_strategy_kwargs={"jitter": 1e-4})`

### Trials keep failing

Check `train_evaluate()` for:
- OOM errors (reduce batch_size bounds)
- NaN rewards (add gradient clipping)
- Environment crashes (add try/except)

### Slow GP inference

For >100 trials, GP inference slows down. Options:
1. Use sparse GP: `model: BOTORCH_MODULAR` with inducing points
2. Prune old trials: keep only top 50%
3. Use GPU for GP: `pip install botorch[cuda]`

---

## References

- [Ax Documentation](https://ax.dev/)
- [BoTorch Documentation](https://botorch.org/)
- [Bayesian Optimization Tutorial](https://arxiv.org/abs/1807.02811)
- [GPyTorch](https://gpytorch.ai/) - Underlying GP library
