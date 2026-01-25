# Project Status

> See [README.md](README.md) for installation and usage.

## Current State

| Metric | Value |
|--------|-------|
| Tests | **1221 passed**, 50 skipped, 9 known failures |
| Shaders | **79/79** compile |
| Throughput | 264K (64 envs), 615K (256 envs) steps/sec |
| Python | 3.12 required |
| CI | GitHub Actions (fallback tests, lint, typecheck) |

## Features

- **6-Stage Curriculum**: Survival → Resources → Nether → Pearls → Stronghold → Dragon
- **SB3 Integration**: Ready for PPO/A2C/SAC training
- **Vectorized Environments**: 64-8192 parallel instances
- **GPU Backend**: Vulkan 1.2 compute shaders
- **CPU Fallback**: Deterministic fallback for CI/development (no GPU required)
- **Constants Module**: Single source of truth for dimensions (`constants.py`)

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
| 5 | CPU Backend | ✅ Complete (deterministic seed propagation) |
| 6 | CI/Testing Infrastructure | ✅ Complete |

## Code Quality

| Check | Description |
|-------|-------------|
| `test_fallback.py` | Tests WITHOUT C++ extension (catches dimension mismatches) |
| `test_determinism.py` | Verifies seed propagation (catches hardware entropy bugs) |
| `test_constants.py` | Ensures dimension constants are consistent |
| `check_hardcoded_dims.py` | Pre-commit hook to flag hardcoded dimensions |

## Key Constants

All dimension constants live in `python/minecraft_sim/constants.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `OBSERVATION_SIZE` | 48 | Main game observation vector |
| `PROGRESS_OBSERVATION_SIZE` | 32 | Curriculum progress tracking (separate) |
| `EXTENDED_OBSERVATION_SIZE` | 256 | Unified FreeTheEndEnv observation |
| `ACTION_SIZE` | 17 | Discrete action space |

## Quick Test

```bash
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json  # macOS
PYTHONPATH=python uv run pytest tests/ -q --ignore=tests/integration --ignore=tests/test_backend_integration.py
```

## Pre-commit Setup

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files  # Verify setup
```
