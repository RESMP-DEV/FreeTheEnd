#!/usr/bin/env python3
"""Evaluate Stage 1: Basic Survival Model.

Runs deterministic evaluations to measure:
- Survival time (ticks alive)
- Dragon damage dealt
- Player health remaining
- Action distribution

Run with:
    cd /path/to/minecraft_sim
    source .venv/bin/activate
    export PYTHONPATH=$(pwd)/python
    export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json  # macOS only
    python training/evaluate_stage1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import os

os.environ.setdefault("VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from minecraft_sim.curriculum import StageID
from minecraft_sim.speedrun_vec_env import make_speedrun_vec_env


def evaluate_model(model_path: str, vecnorm_path: str, num_episodes: int = 100):
    """Evaluate a trained model.

    Args:
        model_path: Path to saved PPO model (.zip)
        vecnorm_path: Path to VecNormalize stats (.pkl)
        num_episodes: Number of episodes to evaluate

    Returns:
        Dictionary of evaluation metrics
    """
    print("=" * 60)
    print("Stage 1 Model Evaluation")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = PPO.load(model_path)

    # Create evaluation environment (single env for cleaner metrics)
    print("Creating evaluation environment...")
    base_env = make_speedrun_vec_env(
        num_envs=1,
        initial_stage=StageID.BASIC_SURVIVAL,
        auto_curriculum=False,
        max_ticks_per_episode=2000,
    )

    # Wrap with VecNormalize and load stats
    env = VecNormalize.load(vecnorm_path, base_env)
    env.training = False  # Don't update stats during eval
    env.norm_reward = False  # Use raw rewards for interpretable metrics

    print(f"Running {num_episodes} evaluation episodes (deterministic)...")
    print("-" * 60)

    # Tracking metrics
    results = {
        "episode_lengths": [],
        "episode_rewards": [],
        "dragon_damage_dealt": [],
        "player_health_remaining": [],
        "player_deaths": 0,
        "dragon_kills": 0,
        "action_counts": np.zeros(env.action_space.n, dtype=np.int64),
    }

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0
        initial_dragon_health = None

        while not done:
            # Deterministic prediction for evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)

            ep_reward += reward[0]
            steps += 1
            results["action_counts"][action[0]] += 1

            # Track dragon health if available in info
            info = infos[0] if infos else {}
            if "obs" in info:
                # Get raw observation for metrics
                raw_obs = info["obs"]
                if initial_dragon_health is None:
                    initial_dragon_health = raw_obs[16] if len(raw_obs) > 16 else 1.0

            done = done[0] if hasattr(done, "__len__") else done

        # Record episode results
        results["episode_lengths"].append(steps)
        results["episode_rewards"].append(ep_reward)

        # Get final info for death/win status
        info = infos[0] if infos else {}

        # Calculate dragon damage (if we tracked it)
        if initial_dragon_health is not None:
            final_dragon_health = 1.0  # Default if not available
            damage = (initial_dragon_health - final_dragon_health) * 200  # Scale to HP
            results["dragon_damage_dealt"].append(max(0, damage))

        # Progress update every 10 episodes
        if (ep + 1) % 10 == 0:
            avg_len = np.mean(results["episode_lengths"][-10:])
            avg_rew = np.mean(results["episode_rewards"][-10:])
            print(
                f"  Episodes {ep - 8:3d}-{ep + 1:3d}: avg_length={avg_len:.0f}, avg_reward={avg_rew:.2f}"
            )

    env.close()

    # Compute summary statistics
    summary = {
        "num_episodes": num_episodes,
        "mean_episode_length": np.mean(results["episode_lengths"]),
        "std_episode_length": np.std(results["episode_lengths"]),
        "min_episode_length": np.min(results["episode_lengths"]),
        "max_episode_length": np.max(results["episode_lengths"]),
        "mean_reward": np.mean(results["episode_rewards"]),
        "std_reward": np.std(results["episode_rewards"]),
        "survival_rate": np.mean(
            [l == 2000 for l in results["episode_lengths"]]
        ),  # Survived full episode
    }

    # Action distribution
    total_actions = results["action_counts"].sum()
    action_pct = results["action_counts"] / total_actions * 100

    return summary, action_pct, results


def print_results(summary: dict, action_pct: np.ndarray):
    """Print formatted evaluation results."""

    ACTION_NAMES = [
        "NOOP",
        "FORWARD",
        "BACK",
        "LEFT",
        "RIGHT",
        "FORWARD_LEFT",
        "FORWARD_RIGHT",
        "JUMP",
        "JUMP_FORWARD",
        "ATTACK",
        "ATTACK_FORWARD",
        "SPRINT",
        "LOOK_LEFT",
        "LOOK_RIGHT",
        "SWAP_WEAPON",
        "LOOK_UP",
        "LOOK_DOWN",
    ]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n📊 Episode Statistics:")
    print(f"  Total episodes:     {summary['num_episodes']}")
    print(
        f"  Mean length:        {summary['mean_episode_length']:.1f} ± {summary['std_episode_length']:.1f} ticks"
    )
    print(
        f"  Length range:       [{summary['min_episode_length']}, {summary['max_episode_length']}]"
    )
    print(f"  Mean reward:        {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"  Survival rate:      {summary['survival_rate'] * 100:.1f}% (reached max ticks)")

    print("\n🎮 Action Distribution:")
    # Sort by frequency
    sorted_indices = np.argsort(action_pct)[::-1]
    for idx in sorted_indices:
        if action_pct[idx] >= 0.1:  # Only show actions used ≥0.1%
            bar_len = int(action_pct[idx] / 2)
            bar = "█" * bar_len
            print(f"  {ACTION_NAMES[idx]:15s} {action_pct[idx]:5.1f}% {bar}")

    print("\n" + "=" * 60)

    # Assessment
    print("\n📝 Assessment:")
    if summary["survival_rate"] > 0.9:
        print("  ✅ High survival rate - agent learned basic survival")
    elif summary["survival_rate"] > 0.5:
        print("  ⚠️  Moderate survival - needs more training or reward tuning")
    else:
        print("  ❌ Low survival - check environment/rewards")

    # Check action diversity
    nonzero_actions = np.sum(action_pct > 1.0)
    if nonzero_actions >= 8:
        print("  ✅ Good action diversity - using many different actions")
    elif nonzero_actions >= 4:
        print("  ⚠️  Limited action variety - may be stuck in local optima")
    else:
        print("  ❌ Poor action diversity - agent may have collapsed policy")

    # Check for NOOP abuse
    if action_pct[0] > 50:  # NOOP > 50%
        print("  ⚠️  High NOOP rate - agent may be passive/undertrained")

    print()


def main():
    output_dir = Path(__file__).parent.parent
    model_path = output_dir / "models" / "stage1_basic_survival.zip"
    vecnorm_path = output_dir / "models" / "stage1_vecnormalize.pkl"

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run train_stage1.py first!")
        sys.exit(1)

    if not vecnorm_path.exists():
        print(f"❌ VecNormalize stats not found: {vecnorm_path}")
        print("   Run train_stage1.py first!")
        sys.exit(1)

    summary, action_pct, raw_results = evaluate_model(
        str(model_path),
        str(vecnorm_path),
        num_episodes=50,  # Fewer episodes for quick eval
    )

    print_results(summary, action_pct)


if __name__ == "__main__":
    main()
