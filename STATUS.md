# Project Status

_Last updated: 2026-01-27_

> See [README.md](README.md) for installation and usage.

## Current State

| Metric | Value |
|--------|-------|
| Tests | **1244 passed**, 53 skipped, 21 failed, 1 error (pytest tests/ -v, 0:05:19) |
| Shaders | **78/79** compile (dragon_ai_full.comp syntax error at line 310) |
| Throughput | 264K (64 envs), 615K (256 envs) steps/sec |
| Python | 3.12 required |

## Latest Test Failures (2026-01-27)

- Integration imports: `tests/integration/test_imports.py` uses a missing `name` fixture.
- Reward signals: `tests/integration/test_reward_signals.py` missing `Path` import.
- Combat rewards: `tests/integration/test_rewards.py::test_combat_rewards` negative total reward.
- Stage 4 done: raw simulator done does not terminate in `tests/integration/test_stage4_done.py`.
- Reward shaping: `_apply_reward_shaping` missing `dones` arg in multiple reward shaping tests.
- Shader syntax: `cpp/shaders/dragon_ai_full.comp` fails glslc parse at line 310.
- Stage 1 survival: player ground/movement/block-break checks fail in `tests/test_stage1_survival.py`.

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
