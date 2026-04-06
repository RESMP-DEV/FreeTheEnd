#!/usr/bin/env python3
"""
Evaluate trained speedrun agent using Stable Baselines 3.

This script provides SB3-compatible evaluation for trained PPO models,
supporting both per-stage evaluation and full speedrun assessment.

Usage:
    python evaluate.py --model checkpoints/best_model.zip
    python evaluate.py --model final_model.zip --episodes 100 --stage 6
    python evaluate.py --model final_model.zip --full-speedrun --episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# Add python module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent / "cpp" / "build"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from minecraft_sim import SB3VecFreeTheEndEnv


def evaluate_stage(
    model: PPO,
    stage: int,
    num_episodes: int = 100,
    deterministic: bool = True,
    num_envs: int = 8,
    max_ticks: int = 36000,
) -> dict[str, Any]:
    """Evaluate model on a specific curriculum stage.

    Args:
        model: Trained PPO model.
        stage: Curriculum stage to evaluate (1-6).
        num_episodes: Number of evaluation episodes.
        deterministic: Whether to use deterministic policy.
        num_envs: Number of parallel environments.
        max_ticks: Maximum ticks per episode (at 20 tps).

    Returns:
        Dictionary with evaluation metrics.
    """
    env = SB3VecFreeTheEndEnv(
        num_envs=num_envs,
        start_stage=stage,
        auto_advance=False,
        max_ticks_per_episode=max_ticks,
    )

    # Track metrics
    successes: list[bool] = []
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    episodes_completed = 0
    obs = env.reset()

    pbar = tqdm(total=num_episodes, desc=f"Stage {stage}")

    while episodes_completed < num_episodes:
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(actions)

        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                # Check curriculum success (natural termination, not truncation)
                curriculum_info = info.get("curriculum", {})
                success = curriculum_info.get("success", False)

                successes.append(success)
                episode_rewards.append(ep["r"])
                episode_lengths.append(ep["l"])

                episodes_completed += 1
                pbar.update(1)

                if episodes_completed >= num_episodes:
                    break

    pbar.close()
    env.close()

    return {
        "stage": stage,
        "episodes": num_episodes,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "std_length": float(np.std(episode_lengths)) if episode_lengths else 0.0,
        "min_length": int(np.min(episode_lengths)) if episode_lengths else 0,
        "max_length": int(np.max(episode_lengths)) if episode_lengths else 0,
    }


def evaluate_full_speedrun(
    model: PPO,
    num_episodes: int = 20,
    deterministic: bool = True,
    num_envs: int = 4,
    max_ticks: int = 72000,
) -> dict[str, Any]:
    """Evaluate full speedrun from spawn to dragon kill.

    Args:
        model: Trained PPO model.
        num_episodes: Number of evaluation episodes.
        deterministic: Whether to use deterministic policy.
        num_envs: Number of parallel environments.
        max_ticks: Maximum ticks per episode (72000 = 60 minutes at 20 tps).

    Returns:
        Dictionary with full speedrun metrics.
    """
    # Start from stage 1 for full speedrun
    env = SB3VecFreeTheEndEnv(
        num_envs=num_envs,
        start_stage=1,
        auto_advance=True,  # Auto-advance through curriculum for full run
        max_ticks_per_episode=max_ticks,
    )

    victories: list[bool] = []
    completion_times: list[int] = []
    stage_reached: list[int] = []
    episode_rewards: list[float] = []

    episodes_completed = 0
    obs = env.reset()

    pbar = tqdm(total=num_episodes, desc="Full Speedrun")

    while episodes_completed < num_episodes:
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(actions)

        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                curriculum_info = info.get("curriculum", {})

                # Victory = success on stage 6 (END_FIGHT)
                current_stage = curriculum_info.get("stage", 1)
                success = curriculum_info.get("success", False)
                victory = (current_stage == 6) and success

                victories.append(victory)
                episode_rewards.append(ep["r"])

                if victory:
                    completion_times.append(ep["l"])
                stage_reached.append(current_stage)

                episodes_completed += 1
                pbar.update(1)

                if episodes_completed >= num_episodes:
                    break

    pbar.close()
    env.close()

    results: dict[str, Any] = {
        "mode": "full_speedrun",
        "episodes": num_episodes,
        "victory_rate": float(np.mean(victories)) if victories else 0.0,
        "mean_stage_reached": float(np.mean(stage_reached)) if stage_reached else 1.0,
        "max_stage_reached": int(np.max(stage_reached)) if stage_reached else 1,
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
    }

    if completion_times:
        results["mean_completion_time"] = float(np.mean(completion_times))
        results["best_completion_time"] = int(np.min(completion_times))
        results["worst_completion_time"] = int(np.max(completion_times))
        # Convert to real time (20 ticks per second)
        results["best_time_seconds"] = results["best_completion_time"] / 20.0
        results["mean_time_seconds"] = results["mean_completion_time"] / 20.0

    return results


def evaluate_with_vecnorm(
    model: PPO,
    vecnorm_path: Path | str | None,
    stage: int | None = None,
    num_episodes: int = 100,
    deterministic: bool = True,
    num_envs: int = 8,
    max_ticks: int = 36000,
) -> dict[str, Any]:
    """Evaluate with VecNormalize wrapper loaded from file.

    Args:
        model: Trained PPO model.
        vecnorm_path: Path to saved VecNormalize stats.
        stage: Stage to evaluate (1-6), or None for all stages.
        num_episodes: Number of episodes per stage.
        deterministic: Whether to use deterministic policy.
        num_envs: Number of parallel environments.
        max_ticks: Maximum ticks per episode.

    Returns:
        Dictionary with evaluation results.
    """
    # Create base environment
    base_env = SB3VecFreeTheEndEnv(
        num_envs=num_envs,
        start_stage=stage if stage else 1,
        auto_advance=False,
        max_ticks_per_episode=max_ticks,
    )

    # Apply VecNormalize if provided
    if vecnorm_path is not None:
        env = VecNormalize.load(str(vecnorm_path), base_env)
        env.training = False  # Evaluation mode
        env.norm_reward = False  # Don't normalize rewards during eval
    else:
        env = base_env

    # Track metrics
    successes: list[bool] = []
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    episodes_completed = 0
    obs = env.reset()

    stage_name = f"Stage {stage}" if stage else "All Stages"
    pbar = tqdm(total=num_episodes, desc=stage_name)

    while episodes_completed < num_episodes:
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(actions)

        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                curriculum_info = info.get("curriculum", {})
                success = curriculum_info.get("success", False)

                successes.append(success)
                episode_rewards.append(ep["r"])
                episode_lengths.append(ep["l"])

                episodes_completed += 1
                pbar.update(1)

                if episodes_completed >= num_episodes:
                    break

    pbar.close()
    env.close()

    return {
        "stage": stage,
        "episodes": num_episodes,
        "vecnorm_applied": vecnorm_path is not None,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "std_length": float(np.std(episode_lengths)) if episode_lengths else 0.0,
        "min_length": int(np.min(episode_lengths)) if episode_lengths else 0,
        "max_length": int(np.max(episode_lengths)) if episode_lengths else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Minecraft speedrun agents with SB3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate on specific stage
    python evaluate.py --model checkpoints/best_model.zip --stage 3

    # Evaluate on all stages
    python evaluate.py --model checkpoints/best_model.zip --episodes 50

    # Full speedrun evaluation
    python evaluate.py --model final_model.zip --full-speedrun --episodes 20

    # With VecNormalize stats
    python evaluate.py --model model.zip --vecnorm vecnorm.pkl --stage 6
        """,
    )
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    parser.add_argument(
        "--vecnorm", type=str, default=None, help="Path to VecNormalize stats (.pkl)"
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Specific stage to evaluate (1-6), or all if not set",
    )
    parser.add_argument(
        "--full-speedrun", action="store_true", help="Evaluate full speedrun (spawn to dragon)"
    )
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument(
        "--max-ticks", type=int, default=36000, help="Max ticks per episode (36000 = 30 min)"
    )
    parser.add_argument(
        "--output", type=str, default="eval_results.json", help="Output JSON file path"
    )
    parser.add_argument(
        "--stochastic", action="store_true", help="Use stochastic policy (default: deterministic)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    # Load model
    print(f"Loading model from {args.model}")
    model = PPO.load(args.model)

    deterministic = not args.stochastic
    results: dict[str, Any] = {"model": args.model, "evaluations": []}

    if args.full_speedrun:
        # Full speedrun evaluation
        print(f"\nEvaluating full speedrun ({args.episodes} episodes)...")
        result = evaluate_full_speedrun(
            model,
            num_episodes=args.episodes,
            deterministic=deterministic,
            num_envs=args.num_envs,
            max_ticks=args.max_ticks * 2,  # Allow longer for full speedrun
        )
        results["evaluations"].append(result)

        print("\nFull Speedrun Results:")
        print(f"  Victory rate: {result['victory_rate'] * 100:.1f}%")
        print(f"  Mean stage reached: {result['mean_stage_reached']:.2f}")
        print(f"  Max stage reached: {result['max_stage_reached']}")
        if "mean_completion_time" in result:
            print(f"  Mean time: {result['mean_time_seconds']:.1f}s")
            print(
                f"  Best time: {result['best_time_seconds']:.1f}s ({result['best_completion_time']} ticks)"
            )
    else:
        # Per-stage evaluation
        stages = [args.stage] if args.stage else list(range(1, 7))

        for stage in stages:
            print(f"\nEvaluating stage {stage} ({args.episodes} episodes)...")

            if args.vecnorm:
                result = evaluate_with_vecnorm(
                    model,
                    vecnorm_path=args.vecnorm,
                    stage=stage,
                    num_episodes=args.episodes,
                    deterministic=deterministic,
                    num_envs=args.num_envs,
                    max_ticks=args.max_ticks,
                )
            else:
                result = evaluate_stage(
                    model,
                    stage=stage,
                    num_episodes=args.episodes,
                    deterministic=deterministic,
                    num_envs=args.num_envs,
                    max_ticks=args.max_ticks,
                )

            results["evaluations"].append(result)

            print(f"\nStage {stage} Results:")
            print(f"  Success rate: {result['success_rate'] * 100:.1f}%")
            print(f"  Mean reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
            print(
                f"  Mean length: {result['mean_length']:.0f} ticks ({result['mean_length'] / 20:.1f}s)"
            )
            if result["success_rate"] > 0:
                print(
                    f"  Best length: {result['min_length']} ticks ({result['min_length'] / 20:.1f}s)"
                )

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
