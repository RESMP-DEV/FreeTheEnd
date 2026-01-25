#!/usr/bin/env python3
"""Benchmark CPU vs GPU backend throughput.

Compares steps/sec across different environment counts for both backends.
CPU backend requires Phase 5 implementation (use_cpu flag on SimulatorConfig).
GPU backend uses the existing Vulkan compute pipeline.

Usage:
    PYTHONPATH=python uv run python examples/benchmark_backends.py
"""

import sys
import time

import numpy as np


def benchmark(use_cpu: bool, num_envs: int, num_steps: int = 10_000) -> float:
    """Run num_steps across num_envs and return aggregate steps/sec.

    Args:
        use_cpu: If True, attempt CPU backend (SimulatorConfig.use_cpu).
        num_envs: Number of parallel environments.
        num_steps: Total simulation steps per environment.

    Returns:
        Throughput in total steps/sec (num_envs * num_steps / elapsed).
        Returns 0.0 if the requested backend is unavailable.
    """
    try:
        import mc189_core
    except ImportError:
        return 0.0

    cfg = mc189_core.SimulatorConfig()
    cfg.num_envs = num_envs

    if use_cpu:
        if not hasattr(cfg, "use_cpu"):
            return 0.0  # CPU backend not yet implemented
        cfg.use_cpu = True

    try:
        sim = mc189_core.MC189Simulator(cfg)
    except RuntimeError:
        return 0.0  # Backend initialization failed

    sim.reset()
    actions = np.zeros(num_envs, dtype=np.int32)

    start = time.perf_counter()
    for _ in range(num_steps):
        sim.step(actions)
    elapsed = time.perf_counter() - start

    return num_envs * num_steps / elapsed


def main() -> None:
    env_counts = [1, 16, 64, 256]
    num_steps = 10_000

    print("Backend Benchmark")
    print("=" * 60)
    print(f"Steps per config: {num_steps:,}")
    print()
    print(f"{'Envs':>6} | {'CPU (steps/sec)':>16} | {'GPU (steps/sec)':>16} | {'Speedup':>8}")
    print("-" * 60)

    for envs in env_counts:
        cpu_rate = benchmark(use_cpu=True, num_envs=envs, num_steps=num_steps)
        gpu_rate = benchmark(use_cpu=False, num_envs=envs, num_steps=num_steps)

        cpu_str = f"{cpu_rate:>12,.0f}" if cpu_rate > 0 else "      N/A   "
        gpu_str = f"{gpu_rate:>12,.0f}" if gpu_rate > 0 else "      N/A   "

        if cpu_rate > 0 and gpu_rate > 0:
            speedup = f"{gpu_rate / cpu_rate:.1f}x"
        else:
            speedup = "---"

        print(f"{envs:>6} | {cpu_str:>16} | {gpu_str:>16} | {speedup:>8}")

    print("=" * 60)

    # Print notes about unavailable backends
    try:
        import mc189_core
        cfg = mc189_core.SimulatorConfig()
        if not hasattr(cfg, "use_cpu"):
            print("\nNote: CPU backend not available (Phase 5 not yet implemented).")
    except ImportError:
        print("\nNote: mc189_core not importable. Build the C++ extension first.")
        print("  cd cpp && mkdir -p build && cd build")
        print("  cmake .. && make -j$(nproc)")


if __name__ == "__main__":
    sys.exit(main() or 0)
