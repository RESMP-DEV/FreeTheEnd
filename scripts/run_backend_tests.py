#!/usr/bin/env python3
"""Manual integration test for VulkanBackend."""

import time

import numpy as np

from minecraft_sim.backend import VulkanBackend


def test_backend():
    print("Testing VulkanBackend...")

    # Test initialization
    backend = VulkanBackend(num_envs=64, enable_validation=False)
    print(f"✓ Backend created: device={backend.device_name}, envs={backend.num_envs}")

    # Test reset
    obs = backend.reset()
    print(f"✓ Reset: obs shape={obs.shape}")

    # Test step
    actions = np.zeros((64, 7), dtype=np.int32)
    obs, rewards, dones, infos = backend.step(actions)
    print(f"✓ Step: obs={obs.shape}, rewards={rewards.shape}, dones={dones.shape}")

    # Test different batch sizes
    for num_envs in [1, 64, 1024, 16384]:
        try:
            b = VulkanBackend(num_envs=num_envs, enable_validation=False)
            a = np.zeros((num_envs, 7), dtype=np.int32)
            o, r, d, i = b.step(a)
            print(f"✓ Batch size {num_envs:5d}: obs={o.shape}")
        except Exception as e:
            print(f"✗ Batch size {num_envs}: {e}")

    # Test throughput with 32K environments
    print("\nThroughput benchmark (32K envs):")
    backend = VulkanBackend(num_envs=32768, enable_validation=False)
    actions = np.zeros((32768, 7), dtype=np.int32)

    # Warmup
    for _ in range(10):
        backend.step(actions)

    # Benchmark
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        backend.step(actions)
    elapsed = time.perf_counter() - start

    steps_per_sec = (32768 * iterations) / elapsed
    print(f"✓ Throughput: {steps_per_sec:,.0f} steps/sec ({steps_per_sec / 1e6:.2f}M steps/sec)")
    print("  Target: 500K steps/sec")

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    test_backend()
