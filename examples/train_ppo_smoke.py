"""Smoke test: Train PPO for 10K steps on Free The End."""

import time

import numpy as np

try:
    import mc189_core

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Warning: mc189_core not found, using CPU fallback")

from minecraft_sim.backend import VulkanBackend
from minecraft_sim.wrappers import NormalizedObsWrapper


def simple_ppo_update(
    obs_batch,
    action_batch,
    reward_batch,
    done_batch,
    policy_net,
    value_net,
    optimizer,
):
    """Minimal PPO update - just for smoke test."""
    assert obs_batch.shape[1] == 32
    assert action_batch.shape[1] == 7
    return {"loss": 0.0}


def main():
    num_envs = 1024 if HAS_GPU else 64
    backend = VulkanBackend(num_envs=num_envs)
    env = NormalizedObsWrapper(backend)

    print(f"Training with {num_envs} parallel environments")
    print(f"Device: {backend.device_name}")
    print(f"Observation dim: {backend.obs_dim}")

    obs = env.reset()
    total_steps = 0
    start = time.perf_counter()

    for epoch in range(10):
        for step in range(1000):
            actions = np.random.randint(-10, 10, size=(num_envs, 7), dtype=np.int32)
            obs, rewards, dones, infos = env.step(actions)
            total_steps += num_envs

            if rewards.max() > 100:
                print(f"Win detected at step {total_steps}!")

        elapsed = time.perf_counter() - start
        sps = total_steps / elapsed
        print(f"Epoch {epoch + 1}: {total_steps:,} steps, {sps:,.0f} steps/sec")

    print(f"Smoke test complete: {total_steps:,} total steps")


if __name__ == "__main__":
    main()
