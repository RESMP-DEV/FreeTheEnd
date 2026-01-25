#!/usr/bin/env python3
"""Quick training run to verify GPU simulator throughput."""

import sys
import time
from pathlib import Path

import numpy as np

# Add python package to path (relative to examples/)
_EXAMPLES_DIR = Path(__file__).parent
sys.path.insert(0, str(_EXAMPLES_DIR.parent / "python"))
sys.path.insert(0, str(_EXAMPLES_DIR.parent / "cpp" / "build"))

from minecraft_sim.backend import VulkanBackend

print("=== MC189 Quick Training Run ===")
print()

# Config
num_envs = 4096
num_epochs = 20
steps_per_epoch = 64

backend = VulkanBackend(num_envs=num_envs)
print(f"Device: {backend.device_name}")
print(f"Config: {num_envs} envs × {steps_per_epoch} steps × {num_epochs} epochs")
print(f"Total: {num_envs * steps_per_epoch * num_epochs:,} steps")
print()

total_steps = 0
start = time.perf_counter()

obs = backend.reset()

for epoch in range(num_epochs):
    epoch_reward = 0

    for step in range(steps_per_epoch):
        # Random policy
        actions = np.random.randint(0, 10, size=num_envs, dtype=np.int32)
        obs, rewards, dones, _ = backend.step(actions)
        epoch_reward += rewards.sum()
        total_steps += num_envs

    elapsed = time.perf_counter() - start
    sps = total_steps / elapsed
    avg_r = epoch_reward / (num_envs * steps_per_epoch)

    print(
        f"Epoch {epoch + 1:2d}/{num_epochs} | "
        f"Steps: {total_steps:>10,} | "
        f"SPS: {sps:>12,.0f} | "
        f"Reward: {avg_r:>8.4f}"
    )

total_time = time.perf_counter() - start
final_sps = total_steps / total_time

print()
print("=" * 60)
print(f"Total steps:  {total_steps:,}")
print(f"Time:         {total_time:.2f}s")
print(f"Throughput:   {final_sps:,.0f} steps/sec ({final_sps / 1e6:.2f}M)")
print()
print("Comparison:")
print("  MineRL:       ~60 steps/sec")
print(f"  This:         {final_sps:,.0f} steps/sec")
print(f"  Speedup:      {final_sps / 60:,.0f}×")
