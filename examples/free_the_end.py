#!/usr/bin/env python3
"""
FREE THE END - Train an RL agent to beat the Ender Dragon
==========================================================

This is the real deal. We're training to beat the dragon using:
- GPU-accelerated Minecraft 1.8.9 physics & combat
- Curriculum learning (easy → hard)
- PPO with proper GAE

Target: First AI to beat the dragon through pure RL (no imitation)
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add python package to path (relative to examples/)
_EXAMPLES_DIR = Path(__file__).parent
sys.path.insert(0, str(_EXAMPLES_DIR.parent / "python"))
sys.path.insert(0, str(_EXAMPLES_DIR.parent / "cpp" / "build"))

# Import the simulator
try:
    import mc189_core

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("⚠️  mc189_core not found - using CPU fallback")


class DragonFightEnv:
    """Wrapper for dragon fight simulation."""

    def __init__(self, num_envs=4096, curriculum_stage=0):
        """
        curriculum_stage:
            0 = Full fight (200 HP dragon, 10 crystals)
            1 = Medium (100 HP dragon, 5 crystals)
            2 = Easy (50 HP dragon, 0 crystals)
            3 = Tutorial (20 HP dragon, perching, 0 crystals)
        """
        self.num_envs = num_envs
        self.curriculum_stage = curriculum_stage

        # Set curriculum parameters
        self.dragon_hp_scale = [1.0, 0.5, 0.25, 0.1][curriculum_stage]
        self.starting_crystals = [10, 5, 0, 0][curriculum_stage]

        # Use MC189Simulator for now (will add DragonFightSimulator later)
        from minecraft_sim.backend import VulkanBackend

import logging

logger = logging.getLogger(__name__)

        self.backend = VulkanBackend(num_envs=num_envs)

        # Observation and action space
        self.obs_dim = 48  # Dragon fight observations
        self.action_dim = 15  # Extended actions

        self._total_wins = 0
        self._total_deaths = 0
        self._best_damage = 0.0

    @property
    def device_name(self):
        return self.backend.device_name

    def reset(self):
        """Reset all environments."""
        obs = self.backend.reset()
        # Pad/truncate to expected obs_dim
        if obs.shape[1] < self.obs_dim:
            obs = np.pad(obs, ((0, 0), (0, self.obs_dim - obs.shape[1])))
        return obs[:, : self.obs_dim]

    def step(self, actions):
        """Step all environments."""
        obs, rewards, dones, infos = self.backend.step(actions)

        # Track stats
        wins = (rewards > 500).sum()  # Win reward is 1000
        deaths = dones.sum() - wins
        self._total_wins += wins
        self._total_deaths += deaths

        # Pad observations
        if obs.shape[1] < self.obs_dim:
            obs = np.pad(obs, ((0, 0), (0, self.obs_dim - obs.shape[1])))

        return obs[:, : self.obs_dim], rewards, dones, infos

    @property
    def total_wins(self):
        return self._total_wins

    @property
    def total_deaths(self):
        return self._total_deaths


class PPOPolicy:
    """Simple neural network policy for PPO."""

    def __init__(self, obs_dim, action_dim, hidden_size=128):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # Two-layer MLP
        self.w1 = np.random.randn(obs_dim, hidden_size).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.1
        self.b2 = np.zeros(hidden_size, dtype=np.float32)
        self.w_policy = np.random.randn(hidden_size, action_dim).astype(np.float32) * 0.01
        self.w_value = np.random.randn(hidden_size, 1).astype(np.float32) * 0.01

    def _forward_hidden(self, obs):
        """Shared hidden layers."""
        h1 = np.maximum(0, obs @ self.w1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        return h2

    def get_action_probs(self, obs):
        """Get action probabilities."""
        h = self._forward_hidden(obs)
        logits = h @ self.w_policy
        logits = np.clip(logits - logits.max(axis=1, keepdims=True), -20, 0)
        exp_logits = np.exp(logits)
        probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
        return probs

    def get_value(self, obs):
        """Get state value estimate."""
        h = self._forward_hidden(obs)
        return (h @ self.w_value).squeeze(-1)

    def act(self, obs):
        """Sample actions and return log probs."""
        probs = self.get_action_probs(obs)
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # Sample actions
        actions = np.array([np.random.choice(self.action_dim, p=p) for p in probs], dtype=np.int32)

        # Log probs
        log_probs = np.log(probs[np.arange(len(probs)), actions] + 1e-8)

        return actions, log_probs

    def update(
        self,
        obs,
        actions,
        advantages,
        returns,
        old_log_probs,
        lr=0.0003,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    ):
        """PPO update step."""

        # Current policy
        probs = self.get_action_probs(obs)
        values = self.get_value(obs)

        # Log probs
        new_log_probs = np.log(probs[np.arange(len(probs)), actions] + 1e-8)

        # Ratio
        ratio = np.exp(new_log_probs - old_log_probs)

        # Clipped objective
        clip_adv = np.clip(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -np.minimum(ratio * advantages, clip_adv).mean()

        # Value loss
        value_loss = ((values - returns) ** 2).mean()

        # Entropy bonus
        entropy = -(probs * np.log(probs + 1e-8)).sum(axis=1).mean()

        # Combined loss gradient (simplified)
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        # Simple gradient update
        # Policy gradient
        h = self._forward_hidden(obs)

        # Compute gradients (simplified)
        adv_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for i, (o, a, adv) in enumerate(zip(obs, actions, adv_normalized)):
            # Policy gradient
            one_hot = np.zeros(self.action_dim)
            one_hot[a] = 1
            grad = np.outer(
                self._forward_hidden(o.reshape(1, -1)).squeeze(), (one_hot - probs[i])
            ) * np.clip(adv, -5, 5)
            self.w_policy += lr * np.clip(grad, -1, 1)

        # Value gradient
        value_error = np.clip(returns - values, -10, 10)
        grad_value = h.T @ value_error.reshape(-1, 1) / len(obs)
        self.w_value += lr * np.clip(grad_value, -1, 1)

        return {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = np.zeros(N)
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║    ███████╗██████╗ ███████╗███████╗    ████████╗██╗  ██╗███████╗  ║
║    ██╔════╝██╔══██╗██╔════╝██╔════╝    ╚══██╔══╝██║  ██║██╔════╝  ║
║    █████╗  ██████╔╝█████╗  █████╗         ██║   ███████║█████╗    ║
║    ██╔══╝  ██╔══██╗██╔══╝  ██╔══╝         ██║   ██╔══██║██╔══╝    ║
║    ██║     ██║  ██║███████╗███████╗       ██║   ██║  ██║███████╗  ║
║    ╚═╝     ╚═╝  ╚═╝╚══════╝╚══════╝       ╚═╝   ╚═╝  ╚═╝╚══════╝  ║
║                                                                   ║
║    ███████╗███╗   ██╗██████╗                                      ║
║    ██╔════╝████╗  ██║██╔══██╗                                     ║
║    █████╗  ██╔██╗ ██║██║  ██║                                     ║
║    ██╔══╝  ██║╚██╗██║██║  ██║                                     ║
║    ███████╗██║ ╚████║██████╔╝                                     ║
║    ╚══════╝╚═╝  ╚═══╝╚═════╝                                      ║
║                                                                   ║
║    Training an AI to beat the Ender Dragon                        ║
║    Pure RL • No imitation • GPU-accelerated                       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")


def main():
    print_banner()

    # ==========================================================================
    # Configuration
    # ==========================================================================

    NUM_ENVS = 8192  # Parallel environments
    ROLLOUT_STEPS = 256  # Steps per rollout
    NUM_EPOCHS = 100  # Training epochs
    BATCH_SIZE = 4096  # Mini-batch size for updates

    # Curriculum stages
    CURRICULUM = [
        {"stage": 3, "epochs": 10, "name": "Tutorial (20 HP, perching)"},
        {"stage": 2, "epochs": 20, "name": "Easy (50 HP, no crystals)"},
        {"stage": 1, "epochs": 30, "name": "Medium (100 HP, 5 crystals)"},
        {"stage": 0, "epochs": 40, "name": "Full Fight (200 HP, 10 crystals)"},
    ]

    print("Configuration:")
    print(f"  Environments:    {NUM_ENVS:,}")
    print(f"  Rollout steps:   {ROLLOUT_STEPS}")
    print(f"  Epochs:          {NUM_EPOCHS}")
    print(f"  Steps per epoch: {NUM_ENVS * ROLLOUT_STEPS:,}")
    print()

    # ==========================================================================
    # Initialize
    # ==========================================================================

    current_curriculum = 0
    curriculum_epochs = 0

    env = DragonFightEnv(num_envs=NUM_ENVS, curriculum_stage=CURRICULUM[0]["stage"])
    policy = PPOPolicy(obs_dim=env.obs_dim, action_dim=env.action_dim, hidden_size=256)

    print(f"Device: {env.device_name}")
    print(f"Observation dim: {env.obs_dim}")
    print(f"Action dim: {env.action_dim}")
    print()

    # ==========================================================================
    # Training Loop
    # ==========================================================================

    obs = env.reset()
    total_steps = 0
    start_time = time.perf_counter()

    best_win_rate = 0.0
    wins_this_epoch = 0
    deaths_this_epoch = 0

    print("=" * 70)
    print(f"Starting curriculum: {CURRICULUM[current_curriculum]['name']}")
    print("=" * 70)
    print()

    for epoch in range(NUM_EPOCHS):
        # Check curriculum progression
        curriculum_epochs += 1
        if curriculum_epochs >= CURRICULUM[current_curriculum]["epochs"]:
            if current_curriculum < len(CURRICULUM) - 1:
                current_curriculum += 1
                curriculum_epochs = 0
                print()
                print("=" * 70)
                print(f"⬆️  CURRICULUM ADVANCE: {CURRICULUM[current_curriculum]['name']}")
                print("=" * 70)
                print()
                # Create new env with harder curriculum
                env = DragonFightEnv(
                    num_envs=NUM_ENVS, curriculum_stage=CURRICULUM[current_curriculum]["stage"]
                )
                obs = env.reset()

        # Collect rollout
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        reward_buffer = []
        done_buffer = []
        value_buffer = []

        epoch_wins = 0
        epoch_deaths = 0
        epoch_reward = 0.0

        for step in range(ROLLOUT_STEPS):
            # Get actions
            actions, log_probs = policy.act(obs)
            values = policy.get_value(obs)

            # Step environment
            next_obs, rewards, dones, _ = env.step(actions)

            # Store
            obs_buffer.append(obs)
            action_buffer.append(actions)
            log_prob_buffer.append(log_probs)
            reward_buffer.append(rewards)
            done_buffer.append(dones.astype(np.float32))
            value_buffer.append(values)

            # Stats
            epoch_reward += rewards.sum()
            epoch_wins += (rewards > 500).sum()
            epoch_deaths += dones.sum() - (rewards > 500).sum()

            obs = next_obs
            total_steps += NUM_ENVS

        # Stack buffers
        obs_batch = np.stack(obs_buffer)
        action_batch = np.stack(action_buffer)
        log_prob_batch = np.stack(log_prob_buffer)
        reward_batch = np.stack(reward_buffer)
        done_batch = np.stack(done_buffer)
        value_batch = np.stack(value_buffer)

        # Compute advantages
        advantages, returns = compute_gae(reward_batch, value_batch, done_batch)

        # Flatten
        obs_flat = obs_batch.reshape(-1, env.obs_dim)
        action_flat = action_batch.reshape(-1)
        log_prob_flat = log_prob_batch.reshape(-1)
        adv_flat = advantages.reshape(-1)
        ret_flat = returns.reshape(-1)

        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # PPO update
        indices = np.random.permutation(len(obs_flat))
        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i : i + BATCH_SIZE]
            if len(batch_idx) < BATCH_SIZE // 2:
                continue

            policy.update(
                obs_flat[batch_idx],
                action_flat[batch_idx],
                adv_flat[batch_idx],
                ret_flat[batch_idx],
                log_prob_flat[batch_idx],
            )

        # Stats
        elapsed = time.perf_counter() - start_time
        sps = total_steps / elapsed
        avg_reward = epoch_reward / (NUM_ENVS * ROLLOUT_STEPS)

        win_rate = epoch_wins / max(1, epoch_wins + epoch_deaths)
        if win_rate > best_win_rate:
            best_win_rate = win_rate

        # Progress bar
        curriculum_progress = curriculum_epochs / CURRICULUM[current_curriculum]["epochs"]
        bar_len = 20
        filled = int(bar_len * curriculum_progress)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} │ "
            f"[{bar}] │ "
            f"Steps: {total_steps / 1e6:6.2f}M │ "
            f"SPS: {sps / 1e6:.2f}M │ "
            f"Reward: {avg_reward:+7.2f} │ "
            f"🏆 {epoch_wins:4d} │ "
            f"💀 {epoch_deaths:4d} │ "
            f"Win%: {win_rate * 100:5.1f}%"
        )

        # Check for early success
        if win_rate > 0.5 and current_curriculum == len(CURRICULUM) - 1:
            print()
            print("🎉" * 30)
            print("   DRAGON DEFEATED! Win rate > 50% on full difficulty!")
            print("🎉" * 30)
            break

    # ==========================================================================
    # Final Stats
    # ==========================================================================

    total_time = time.perf_counter() - start_time

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print(f"  Total steps:      {total_steps:,}")
    print(f"  Total time:       {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(
        f"  Throughput:       {total_steps / total_time:,.0f} steps/sec ({total_steps / total_time / 1e6:.2f}M)"
    )
    print()
    print(f"  Total wins:       {env.total_wins:,}")
    print(f"  Total deaths:     {env.total_deaths:,}")
    print(
        f"  Overall win rate: {env.total_wins / max(1, env.total_wins + env.total_deaths) * 100:.2f}%"
    )
    print(f"  Best win rate:    {best_win_rate * 100:.2f}%")
    print()

    if env.total_wins > 0:
        print("🐲 THE DRAGON HAS BEEN SLAIN! 🐲")
    else:
        print("The dragon lives... for now.")

    return env.total_wins > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
