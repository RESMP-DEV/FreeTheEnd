#!/usr/bin/env python3
"""Train PPO on the MC189 GPU simulator."""

import time
from pathlib import Path

import numpy as np

# Check for GPU backend
try:
    import mc189_core

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("Warning: mc189_core not found")

import sys

# Add python package to path (relative to examples/)
_EXAMPLES_DIR = Path(__file__).parent
sys.path.insert(0, str(_EXAMPLES_DIR.parent / "python"))
sys.path.insert(0, str(_EXAMPLES_DIR.parent / "cpp" / "build"))

from minecraft_sim.backend import VulkanBackend


class SimplePolicyNet:
    """Simple linear policy for smoke test."""

    def __init__(self, obs_dim=48, act_dim=10):
        self.weights = np.random.randn(obs_dim, act_dim).astype(np.float32) * 0.01
        self.bias = np.zeros(act_dim, dtype=np.float32)

    def forward(self, obs):
        """Returns action logits."""
        return np.clip(obs @ self.weights + self.bias, -20, 20)

    def act(self, obs):
        """Sample actions from policy."""
        logits = self.forward(obs)
        # Stable softmax
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        # Sample
        actions = np.array([np.random.choice(len(p), p=p) for p in probs], dtype=np.int32)
        return actions, probs

    def update(self, obs, actions, advantages, lr=0.0003):
        """Simple policy gradient update."""
        logits = self.forward(obs)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)

        # Gradient of log prob
        grad = np.zeros_like(self.weights)
        for i, (o, a, adv) in enumerate(zip(obs, actions, advantages)):
            one_hot = np.zeros(10)
            one_hot[a] = 1
            grad += np.outer(o, (one_hot - probs[i])) * np.clip(adv, -10, 10)

        self.weights += lr * np.clip(grad / len(obs), -1, 1)


class SimpleValueNet:
    """Simple linear value function."""

    def __init__(self, obs_dim=48):
        self.weights = np.random.randn(obs_dim).astype(np.float32) * 0.01
        self.bias = 0.0

    def forward(self, obs):
        return np.clip(obs @ self.weights + self.bias, -100, 100)

    def update(self, obs, returns, lr=0.001):
        """Simple value function update."""
        values = self.forward(obs)
        errors = np.clip(returns - values, -10, 10)

        grad = obs.T @ errors / len(obs)
        self.weights += lr * np.clip(grad, -1, 1)
        self.bias += lr * np.clip(errors.mean(), -1, 1)

        return (errors**2).mean()


def compute_gae_batched(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute GAE for all environments in parallel (vectorized)."""
    # rewards, values, dones: (steps, envs)
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = np.zeros(N, dtype=np.float32)
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def main():
    print("=" * 60)
    print("MC189 GPU Simulator - PPO Training")
    print("=" * 60)
    print()

    # Config
    num_envs = 4096
    rollout_steps = 128
    num_epochs = 50

    total_steps_target = num_envs * rollout_steps * num_epochs
    print(f"Target: {total_steps_target:,} total steps")
    print(f"Config: {num_envs} envs × {rollout_steps} steps × {num_epochs} epochs")
    print()

    # Create environment
    print("Initializing simulator...")
    backend = VulkanBackend(num_envs=num_envs)
    print(f"Device: {backend.device_name}")
    print(f"Obs dim: {backend.obs_dim}")
    print()

    # Create networks
    obs_dim = backend.obs_dim
    policy = SimplePolicyNet(obs_dim=obs_dim, act_dim=10)
    value = SimpleValueNet(obs_dim=obs_dim)

    # Training loop
    obs = backend.reset()
    total_steps = 0
    total_rewards = 0
    episode_count = 0
    start_time = time.perf_counter()

    print("Starting training...")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Collect rollout
        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        done_buffer = []
        value_buffer = []

        epoch_reward = 0
        epoch_dones = 0

        for step in range(rollout_steps):
            # Get actions from policy
            actions, _ = policy.act(obs)

            # Get value estimates
            values = value.forward(obs)

            # Step environment
            next_obs, rewards, dones, _ = backend.step(actions)

            # Store
            obs_buffer.append(obs)
            action_buffer.append(actions)
            reward_buffer.append(rewards)
            done_buffer.append(dones.astype(np.float32))
            value_buffer.append(values)

            epoch_reward += rewards.sum()
            epoch_dones += dones.sum()

            # Reset done environments
            if dones.any():
                # Backend auto-resets, just update counter
                episode_count += dones.sum()

            obs = next_obs
            total_steps += num_envs

        # Stack buffers
        obs_batch = np.stack(obs_buffer)  # (steps, envs, obs_dim)
        action_batch = np.stack(action_buffer)  # (steps, envs)
        reward_batch = np.stack(reward_buffer)  # (steps, envs)
        done_batch = np.stack(done_buffer)  # (steps, envs)
        value_batch = np.stack(value_buffer)  # (steps, envs)

        # Compute advantages (vectorized over all envs)
        advantages, returns = compute_gae_batched(reward_batch, value_batch, done_batch)

        # Flatten for updates
        obs_flat = obs_batch.reshape(-1, obs_dim)
        action_flat = action_batch.reshape(-1)
        adv_flat = advantages.reshape(-1)
        ret_flat = returns.reshape(-1)

        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # Update networks (mini-batch)
        batch_size = 2048
        indices = np.random.permutation(len(obs_flat))

        value_loss = 0
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            policy.update(obs_flat[batch_idx], action_flat[batch_idx], adv_flat[batch_idx])
            value_loss += value.update(obs_flat[batch_idx], ret_flat[batch_idx])

        # Stats
        elapsed = time.perf_counter() - start_time
        sps = total_steps / elapsed
        avg_reward = epoch_reward / (num_envs * rollout_steps)

        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Steps: {total_steps:>10,} | "
            f"SPS: {sps:>10,.0f} | "
            f"Reward: {avg_reward:>7.4f} | "
            f"Episodes: {int(epoch_dones):>5}"
        )

    # Final stats
    total_time = time.perf_counter() - start_time
    final_sps = total_steps / total_time

    print("-" * 60)
    print("Training complete!")
    print(f"  Total steps:    {total_steps:,}")
    print(f"  Total time:     {total_time:.1f}s")
    print(f"  Throughput:     {final_sps:,.0f} steps/sec ({final_sps / 1e6:.2f}M)")
    print(f"  Episodes:       {episode_count:,}")
    print()

    # Compare to baselines
    print("Comparison to other MC simulators:")
    print("  MineRL:         ~60 steps/sec")
    print(f"  Our simulator:  {final_sps:,.0f} steps/sec")
    print(f"  Speedup:        {final_sps / 60:,.0f}×")


if __name__ == "__main__":
    main()
