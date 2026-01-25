#!/usr/bin/env python3
"""Benchmark throughput for each curriculum stage."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import numpy as np

from minecraft_sim.curriculum import StageID
from minecraft_sim.speedrun_vec_env import SpeedrunVecEnv


def benchmark_stage(stage_id: StageID, num_envs: int = 64, num_steps: int = 1000) -> float:
    """Benchmark a single stage, return steps/sec."""
    env = SpeedrunVecEnv(num_envs=num_envs, initial_stage=stage_id)
    env.reset()
    actions = np.zeros(num_envs, dtype=np.int32)

    start = time.perf_counter()
    for _ in range(num_steps):
        env.step(actions)
    elapsed = time.perf_counter() - start

    env.close()
    return num_envs * num_steps / elapsed


if __name__ == "__main__":
    print("Per-Stage Throughput Benchmark")
    print("=" * 50)
    print(f"{'Stage':<25} {'Steps/sec':>12}")
    print("-" * 50)

    for stage_id in StageID:
        try:
            rate = benchmark_stage(stage_id)
            print(f"{stage_id.name:<25} {rate:>12,.0f}")
        except Exception as e:
            print(f"{stage_id.name:<25} {'ERROR':>12} ({e})")

    print("=" * 50)
