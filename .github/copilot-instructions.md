# Copilot Instructions for FreeTheEnd (minecraft_sim)

GPU-accelerated Minecraft 1.8.9 simulator for RL speedrun training. Physics + AI simulated via Vulkan compute shaders—no rendering, 600k+ steps/sec throughput.

## Architecture

```
python/minecraft_sim/   # Python package: Gymnasium envs, curriculum, reward shaping
  __init__.py           # Loads mc189_core.so, exports all public APIs
  vec_env.py            # VecDragonFightEnv, SB3 wrappers
  curriculum.py         # 6-stage StageID enum, Stage configs
  reward_shaping.py     # create_stage{1-6}_reward_shaper() functions
  progression.py        # SpeedrunProgress dataclass, ProgressTracker
cpp/                    # C++ extension (mc189_core.so)
  src/                  # C++ source files
  shaders/              # Vulkan compute shaders (.comp), compiled SPIR-V (.spv)
  CMakeLists.txt        # Build config; supports -DCPU_ONLY=ON for CI
configs/                # YAML configs (ax_sweep_config.yaml for optimization)
tests/                  # pytest suite; markers: slow, gpu, oracle
examples/               # Training scripts: train_ppo.py, run_ax_sweep.py, free_the_end.py
docs/                   # training_guide.md, hyperparameter_optimization.md, tutorial_full_speedrun.md
```

## Build & Run Commands

```bash
# Python environment (always use uv for consistency with Python 3.12)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install numpy gymnasium stable-baselines3

# Build C++ extension
cd cpp/build && cmake .. && cmake --build . --target mc189_core
cp mc189_core.cpython-*-darwin.so ../../python/minecraft_sim/

# CPU-only build (CI, no Vulkan)
cmake -DCPU_ONLY=ON .. && cmake --build . --target mc189_core

# macOS Vulkan setup
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json

# Run tests
uv run pytest tests/ -v                    # all tests
uv run pytest tests/ -m "not slow" -v      # skip slow
make test-fast                              # skip slow + gpu
```

## Training Workflow

### Quick Start (Smoke Test)
```bash
# Verify setup with a quick training run
cd contrib/minecraft_sim
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
PYTHONPATH=python uv run python examples/train_ppo.py
```

### Full Curriculum Training (6 Stages)
```bash
# Train from spawn to dragon kill (~1-5 hours depending on hardware)
PYTHONPATH=python uv run python examples/free_the_end.py
```

### Using Stable-Baselines3
```python
from minecraft_sim import SB3VecDragonFightEnv
from stable_baselines3 import PPO

env = SB3VecDragonFightEnv(num_envs=64)
model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=256, 
            gamma=0.999, verbose=1, tensorboard_log="./logs/")
model.learn(total_timesteps=100_000_000)
model.save("speedrun_agent")
```

### Curriculum Stages (StageID enum)
1. BASIC_SURVIVAL → 2. RESOURCE_GATHERING → 3. NETHER_NAVIGATION → 4. ENDERMAN_HUNTING → 5. STRONGHOLD_FINDING → 6. DRAGON_FIGHT

Stages advance at 50-70% success rate. Key hyperparameters per stage:
- Stages 1-2: `gamma=0.99`, `n_steps=128` (short horizon)
- Stages 3-6: `gamma=0.999`, `n_steps=256` (long horizon, sparse rewards)

## Hyperparameter Optimization

### Run Ax/BoTorch Sweep (Local Bayesian Optimization)
```bash
# Install Ax
uv pip install ax-platform botorch

# Run 50-trial sweep (uses configs/ax_sweep_config.yaml)
PYTHONPATH=python uv run python examples/run_ax_sweep.py --trials 50

# Multi-GPU parallel trials
PYTHONPATH=python uv run python examples/run_ax_sweep.py --trials 100 --parallel 4
```

### Key Parameters to Tune
| Parameter | Range | Impact |
|-----------|-------|--------|
| `learning_rate` | 1e-5 to 1e-3 (log) | Stability vs speed |
| `n_steps` | 64, 128, 256, 512 | Variance vs throughput |
| `gamma` | 0.95 to 0.999 | Credit assignment horizon |
| `ent_coef` | 1e-4 to 0.1 (log) | Exploration vs exploitation |

### Analyze Sweep Results
```python
from ax.service.ax_client import AxClient
ax_client = AxClient.load_from_json_file("ax_sweep_results.json")
best_params, metrics = ax_client.get_best_parameters()
print(f"Best: {best_params}, success_rate: {metrics}")
```

## Key Patterns

### Always prefix Python with `uv run`
```bash
uv run python train.py    # ✓ ensures Python 3.12
python train.py           # ✗ may use system Python
```

### Environment creation
```python
from minecraft_sim import VecDragonFightEnv, SB3VecDragonFightEnv
env = VecDragonFightEnv(num_envs=64)           # custom loops
env = SB3VecDragonFightEnv(num_envs=64)        # Stable Baselines 3
```

### C++ module loading
The `mc189_core.so` extension must be in `python/minecraft_sim/` or on `PYTHONPATH`. Check with:
```python
from minecraft_sim import check_cpp_module
assert check_cpp_module(), "Build C++ extension first"
```

## Observation & Action Spaces

- **Observation**: float32 vector, default 48 dims (position, velocity, health, dragon state, etc.)
- **Actions**: Discrete(17) — NOOP, movement, jump, attack, look, sprint, swap_weapon

## Testing Conventions

- Tests live in `tests/test_*.py`; use pytest markers to scope runs
- Shader validation: `make validate-shaders` (syntax only), `make shaders` (compile to SPIR-V)
- Oracle tests (`-m oracle`) compare against ground-truth Minecraft recordings

## Code Style

- Python: PEP 8, ruff linter (see `pyproject.toml`), `snake_case` functions, `CapWords` classes
- C++: follow existing style in `cpp/src/`
- Shaders: `.comp` for compute shaders, descriptive names (e.g., `dragon_ai.comp`)

## Key Documentation

- **Training details**: `docs/training_guide.md` — reward shaping, curriculum mechanics, expected timelines
- **Optimization guide**: `docs/hyperparameter_optimization.md` — Ax/BoTorch setup, Bayesian optimization theory
- **Full tutorial**: `docs/tutorial_full_speedrun.md` — stage-by-stage breakdown with code

## Project Status (copied from `STATUS.md`)

```markdown
# Project Status

_Last updated: 2026-01-27_

> See [README.md](README.md) for installation and usage.

## Current State

| Metric | Value |
|--------|-------|
| Tests | **1221 passed**, 50 skipped, 9 known failures |
| Shaders | **79/79** compile |
| Throughput | 264K (64 envs), 615K (256 envs) steps/sec |
| Python | 3.12 required |

## Primary Workflows (what to run)

### Training

- Smoke test PPO loop (fast sanity check):

  - `PYTHONPATH=python uv run python examples/train_ppo.py`
- Full 6-stage curriculum training:

  - `PYTHONPATH=python uv run python examples/free_the_end.py`

### Hyperparameter optimization (Ax/BoTorch)

- Run a local sweep (writes `ax_sweep_results.json`):

  - `uv pip install ax-platform botorch`
  - `PYTHONPATH=python uv run python examples/run_ax_sweep.py --trials 50`

## Features

- **6-Stage Curriculum**: Survival → Resources → Nether → Pearls → Stronghold → Dragon
- **SB3 Integration**: Ready for PPO/A2C/SAC training
- **Vectorized Environments**: 64-8192 parallel instances
- **GPU Backend**: Vulkan 1.2 compute shaders
- **CPU Backend**: Basic support for CI/development (no GPU required)

## Hardware Support

| Platform | Status |
|----------|--------|
| Apple M1/M2/M3/M4 | ✅ Full support |
| NVIDIA GTX 1060+ | ✅ Full support |
| AMD RX 580+ | ✅ Full support |
| Intel UHD 630+ | ⚠️ Experimental |

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Shader Compilation | ✅ Complete |
| 2 | Stage Integration | ✅ Complete |
| 3 | World State Management | ✅ Complete |
| 4 | Per-Stage Validation | ✅ Complete |
| 5 | CPU Backend | 🚧 Partial (bindings exposed, determinism WIP) |

## Quick Test

```bash
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json  # macOS
PYTHONPATH=python uv run pytest tests/ -q --ignore=tests/integration --ignore=tests/test_backend_integration.py
```
```

## Common Gotchas

- **Missing mc189_core**: Rebuild C++ extension and copy `.so` to `python/minecraft_sim/`
- **macOS Vulkan errors**: Set `VK_ICD_FILENAMES` to MoltenVK ICD path
- **Memory**: Each env uses ~0.5 MB VRAM; 8192 envs ≈ 4 GB
- **Relative paths**: Use `Path(__file__).parent` not absolute paths (standalone project under `contrib/`)
- **Training stalls**: Increase `ent_coef` for more exploration or check reward shaping signals
