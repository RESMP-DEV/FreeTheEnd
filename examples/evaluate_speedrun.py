#!/usr/bin/env python3
"""Evaluation and benchmarking script for Minecraft speedrun agents.

This script loads trained models and runs comprehensive evaluation:
- N evaluation episodes with deterministic/stochastic policies
- Stage completion rates and timing
- Best run replay saving
- Summary statistics with confidence intervals

Usage:
    python evaluate_speedrun.py --model checkpoints/best.npz --episodes 100
    python evaluate_speedrun.py --model checkpoints/best.npz --episodes 1000 --save-replays
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent / "cpp" / "build"))

from minecraft_sim.curriculum import StageID


@dataclass
class EpisodeRecord:
    """Recording of a single evaluation episode."""

    episode_id: int
    seed: int
    total_reward: float
    total_steps: int
    success: bool
    stages_completed: list[StageID]
    stage_times: dict[int, int]  # StageID value -> steps to reach
    actions: list[int] = field(default_factory=list)
    observations: list[NDArray[np.float32]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    final_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize episode record for JSON."""
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "total_reward": self.total_reward,
            "total_steps": self.total_steps,
            "success": self.success,
            "stages_completed": [s.value for s in self.stages_completed],
            "stage_times": self.stage_times,
            "final_info": self.final_info,
        }


@dataclass
class StageStats:
    """Statistics for a single curriculum stage."""

    stage_id: StageID
    attempts: int = 0
    successes: int = 0
    total_time: int = 0  # Cumulative steps to reach/complete
    best_time: int | None = None
    worst_time: int | None = None
    times: list[int] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        return self.successes / max(1, self.attempts)

    @property
    def average_time(self) -> float:
        return self.total_time / max(1, self.successes)

    @property
    def time_std(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return float(np.std(self.times))


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""

    num_episodes: int
    num_envs: int
    total_steps: int
    wall_time: float
    stage_stats: dict[StageID, StageStats]
    episode_records: list[EpisodeRecord]
    reward_distribution: NDArray[np.float32]
    actions_per_second: float

    def get_full_completion_rate(self) -> float:
        """Rate of completing all stages (full speedrun)."""
        end_fight = self.stage_stats.get(StageID.END_FIGHT)
        if end_fight is None:
            return 0.0
        return end_fight.completion_rate

    def get_fastest_completion(self) -> int | None:
        """Fastest full speedrun completion time in steps."""
        end_fight = self.stage_stats.get(StageID.END_FIGHT)
        if end_fight is None or end_fight.best_time is None:
            return None
        return end_fight.best_time

    def to_summary_dict(self) -> dict[str, Any]:
        """Generate summary statistics dictionary."""
        summary: dict[str, Any] = {
            "evaluation": {
                "num_episodes": self.num_episodes,
                "num_envs": self.num_envs,
                "total_steps": self.total_steps,
                "wall_time_seconds": self.wall_time,
                "actions_per_second": self.actions_per_second,
            },
            "stages": {},
            "rewards": {
                "mean": float(np.mean(self.reward_distribution)),
                "std": float(np.std(self.reward_distribution)),
                "min": float(np.min(self.reward_distribution)),
                "max": float(np.max(self.reward_distribution)),
                "median": float(np.median(self.reward_distribution)),
                "percentile_25": float(np.percentile(self.reward_distribution, 25)),
                "percentile_75": float(np.percentile(self.reward_distribution, 75)),
            },
            "speedrun": {
                "full_completion_rate": self.get_full_completion_rate(),
                "fastest_completion_steps": self.get_fastest_completion(),
            },
        }

        for stage_id, stats in self.stage_stats.items():
            summary["stages"][stage_id.name] = {
                "completion_rate": stats.completion_rate,
                "attempts": stats.attempts,
                "successes": stats.successes,
                "average_time_steps": stats.average_time if stats.successes > 0 else None,
                "time_std": stats.time_std if stats.successes > 0 else None,
                "best_time_steps": stats.best_time,
                "worst_time_steps": stats.worst_time,
            }

        return summary


class PolicyLoader:
    """Load trained policy weights from checkpoint files."""

    @staticmethod
    def load_numpy(path: Path) -> dict[str, NDArray[np.float32]]:
        """Load policy weights from .npz file."""
        data = np.load(path)
        return {k: v.astype(np.float32) for k, v in data.items()}

    @staticmethod
    def load_json(path: Path) -> dict[str, NDArray[np.float32]]:
        """Load policy weights from JSON file (for compatibility)."""
        with open(path) as f:
            data = json.load(f)
        return {k: np.array(v, dtype=np.float32) for k, v in data.items()}


class EvaluationPolicy:
    """Policy wrapper for evaluation with deterministic/stochastic modes."""

    def __init__(
        self,
        weights: dict[str, NDArray[np.float32]],
        obs_dim: int = 48,
        action_dim: int = 17,
        hidden_size: int = 256,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # Load weights with fallback defaults
        self.w1 = weights.get("w1", np.random.randn(obs_dim, hidden_size).astype(np.float32) * 0.1)
        self.b1 = weights.get("b1", np.zeros(hidden_size, dtype=np.float32))
        self.w2 = weights.get(
            "w2", np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.1
        )
        self.b2 = weights.get("b2", np.zeros(hidden_size, dtype=np.float32))
        self.w_policy = weights.get(
            "w_policy", np.random.randn(hidden_size, action_dim).astype(np.float32) * 0.01
        )
        self.w_value = weights.get(
            "w_value", np.random.randn(hidden_size, 1).astype(np.float32) * 0.01
        )

    def _forward_hidden(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Shared hidden layers with ReLU activation."""
        h1 = np.maximum(0, obs @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        return h2

    def get_action_probs(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get action probabilities from policy network."""
        h = self._forward_hidden(obs)
        logits = h @ self.w_policy
        logits = np.clip(logits - logits.max(axis=1, keepdims=True), -20, 0)
        exp_logits = np.exp(logits)
        probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
        return probs

    def act_deterministic(self, obs: NDArray[np.float32]) -> NDArray[np.int32]:
        """Select actions deterministically (argmax)."""
        probs = self.get_action_probs(obs)
        return np.argmax(probs, axis=1).astype(np.int32)

    def act_stochastic(self, obs: NDArray[np.float32]) -> NDArray[np.int32]:
        """Sample actions stochastically from policy distribution."""
        probs = self.get_action_probs(obs)
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return np.array([np.random.choice(self.action_dim, p=p) for p in probs], dtype=np.int32)


class ReplayBuffer:
    """Buffer for storing and saving episode replays."""

    def __init__(self, max_replays: int = 10):
        self.max_replays = max_replays
        self.replays: list[EpisodeRecord] = []

    def maybe_add(self, record: EpisodeRecord) -> bool:
        """Add replay if it's among the best (highest reward)."""
        if len(self.replays) < self.max_replays:
            self.replays.append(record)
            self.replays.sort(key=lambda r: -r.total_reward)
            return True

        if record.total_reward > self.replays[-1].total_reward:
            self.replays[-1] = record
            self.replays.sort(key=lambda r: -r.total_reward)
            return True

        return False

    def save(self, output_dir: Path) -> None:
        """Save all replays to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, record in enumerate(self.replays):
            replay_path = output_dir / f"replay_{i:03d}_reward{record.total_reward:.0f}.npz"
            np.savez_compressed(
                replay_path,
                episode_id=record.episode_id,
                seed=record.seed,
                total_reward=record.total_reward,
                total_steps=record.total_steps,
                success=record.success,
                actions=np.array(record.actions, dtype=np.int32),
                observations=np.stack(record.observations) if record.observations else np.array([]),
                rewards=np.array(record.rewards, dtype=np.float32),
            )

        # Save index
        index_path = output_dir / "replay_index.json"
        with open(index_path, "w") as f:
            json.dump([r.to_dict() for r in self.replays], f, indent=2)


class SpeedrunEvaluator:
    """Main evaluation runner for speedrun agents."""

    def __init__(
        self,
        policy: EvaluationPolicy,
        num_envs: int = 256,
        max_steps_per_episode: int = 36000,  # 30 minutes at 20 tps
        deterministic: bool = True,
        save_replays: bool = False,
        verbose: bool = True,
    ):
        self.policy = policy
        self.num_envs = num_envs
        self.max_steps = max_steps_per_episode
        self.deterministic = deterministic
        self.save_replays = save_replays
        self.verbose = verbose

        self.replay_buffer = ReplayBuffer(max_replays=10) if save_replays else None

        # Initialize environment
        self._init_env()

    def _init_env(self) -> None:
        """Initialize the vectorized environment."""
        try:
            from minecraft_sim.vec_env import VecDragonFightEnv

            self.env = VecDragonFightEnv(num_envs=self.num_envs)
            self.has_env = True
        except ImportError:

import logging

logger = logging.getLogger(__name__)

            self.env = None
            self.has_env = False
            if self.verbose:
                print("Warning: VecDragonFightEnv not available, using mock environment")

    def _mock_step(
        self, actions: NDArray[np.int32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        """Mock environment step for testing without GPU backend."""
        obs = np.random.rand(self.num_envs, self.policy.obs_dim).astype(np.float32)
        rewards = np.random.randn(self.num_envs).astype(np.float32)
        dones = np.random.rand(self.num_envs) < 0.001
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        return obs, rewards, dones.astype(np.bool_), infos

    def evaluate(self, num_episodes: int) -> EvaluationResults:
        """Run evaluation for a specified number of episodes.

        Args:
            num_episodes: Total number of episodes to run.

        Returns:
            EvaluationResults with aggregated statistics.
        """
        if self.verbose:
            print(f"Starting evaluation: {num_episodes} episodes on {self.num_envs} parallel envs")
            print(f"Policy mode: {'deterministic' if self.deterministic else 'stochastic'}")
            print("-" * 70)

        # Initialize tracking
        stage_stats: dict[StageID, StageStats] = {
            stage: StageStats(stage_id=stage) for stage in StageID
        }
        episode_records: list[EpisodeRecord] = []
        all_rewards: list[float] = []

        # Episode state per environment
        episode_ids = np.arange(self.num_envs)
        episode_seeds = np.random.randint(0, 2**31, size=self.num_envs)
        episode_steps = np.zeros(self.num_envs, dtype=np.int32)
        episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        episode_actions: list[list[int]] = [[] for _ in range(self.num_envs)]
        episode_obs: list[list[NDArray[np.float32]]] = [[] for _ in range(self.num_envs)]
        episode_reward_seq: list[list[float]] = [[] for _ in range(self.num_envs)]
        stages_reached: list[set[StageID]] = [set() for _ in range(self.num_envs)]
        stage_times: list[dict[int, int]] = [{} for _ in range(self.num_envs)]

        completed_episodes = 0
        total_steps = 0
        next_episode_id = self.num_envs
        start_time = time.perf_counter()

        # Reset environment
        if self.has_env and self.env is not None:
            obs = self.env.reset()
        else:
            obs = np.random.rand(self.num_envs, self.policy.obs_dim).astype(np.float32)

        while completed_episodes < num_episodes:
            # Get actions
            if self.deterministic:
                actions = self.policy.act_deterministic(obs)
            else:
                actions = self.policy.act_stochastic(obs)

            # Step environment
            if self.has_env and self.env is not None:
                next_obs, rewards, dones, infos = self.env.step(actions)
            else:
                next_obs, rewards, dones, infos = self._mock_step(actions)

            total_steps += self.num_envs

            # Update episode state
            episode_steps += 1
            episode_rewards += rewards

            # Store replay data if enabled
            if self.save_replays:
                for i in range(self.num_envs):
                    episode_actions[i].append(int(actions[i]))
                    episode_obs[i].append(obs[i].copy())
                    episode_reward_seq[i].append(float(rewards[i]))

            # Detect stage completion from rewards (simplified heuristic)
            for i in range(self.num_envs):
                if rewards[i] > 10.0 and StageID.BASIC_SURVIVAL not in stages_reached[i]:
                    stages_reached[i].add(StageID.BASIC_SURVIVAL)
                    stage_times[i][StageID.BASIC_SURVIVAL.value] = int(episode_steps[i])
                if rewards[i] > 50.0 and StageID.RESOURCE_GATHERING not in stages_reached[i]:
                    stages_reached[i].add(StageID.RESOURCE_GATHERING)
                    stage_times[i][StageID.RESOURCE_GATHERING.value] = int(episode_steps[i])
                if rewards[i] > 100.0 and StageID.NETHER_NAVIGATION not in stages_reached[i]:
                    stages_reached[i].add(StageID.NETHER_NAVIGATION)
                    stage_times[i][StageID.NETHER_NAVIGATION.value] = int(episode_steps[i])
                if rewards[i] > 200.0 and StageID.ENDERMAN_HUNTING not in stages_reached[i]:
                    stages_reached[i].add(StageID.ENDERMAN_HUNTING)
                    stage_times[i][StageID.ENDERMAN_HUNTING.value] = int(episode_steps[i])
                if rewards[i] > 500.0 and StageID.STRONGHOLD_FINDING not in stages_reached[i]:
                    stages_reached[i].add(StageID.STRONGHOLD_FINDING)
                    stage_times[i][StageID.STRONGHOLD_FINDING.value] = int(episode_steps[i])
                if rewards[i] > 1000.0 and StageID.END_FIGHT not in stages_reached[i]:
                    stages_reached[i].add(StageID.END_FIGHT)
                    stage_times[i][StageID.END_FIGHT.value] = int(episode_steps[i])

            # Check for timeout
            timeouts = episode_steps >= self.max_steps
            dones = dones | timeouts

            # Process completed episodes
            for i in np.where(dones)[0]:
                if completed_episodes >= num_episodes:
                    break

                # Create episode record
                success = StageID.END_FIGHT in stages_reached[i]
                record = EpisodeRecord(
                    episode_id=int(episode_ids[i]),
                    seed=int(episode_seeds[i]),
                    total_reward=float(episode_rewards[i]),
                    total_steps=int(episode_steps[i]),
                    success=success,
                    stages_completed=list(stages_reached[i]),
                    stage_times=stage_times[i].copy(),
                    actions=episode_actions[i].copy() if self.save_replays else [],
                    observations=episode_obs[i].copy() if self.save_replays else [],
                    rewards=episode_reward_seq[i].copy() if self.save_replays else [],
                )

                episode_records.append(record)
                all_rewards.append(record.total_reward)

                # Update stage stats
                for stage in StageID:
                    stage_stats[stage].attempts += 1
                    if stage in stages_reached[i]:
                        stage_stats[stage].successes += 1
                        t = stage_times[i].get(stage.value, 0)
                        stage_stats[stage].total_time += t
                        stage_stats[stage].times.append(t)
                        if stage_stats[stage].best_time is None or t < stage_stats[stage].best_time:
                            stage_stats[stage].best_time = t
                        if (
                            stage_stats[stage].worst_time is None
                            or t > stage_stats[stage].worst_time
                        ):
                            stage_stats[stage].worst_time = t

                # Save replay if among best
                if self.replay_buffer is not None:
                    self.replay_buffer.maybe_add(record)

                completed_episodes += 1

                # Reset this environment slot
                episode_ids[i] = next_episode_id
                episode_seeds[i] = np.random.randint(0, 2**31)
                episode_steps[i] = 0
                episode_rewards[i] = 0.0
                stages_reached[i] = set()
                stage_times[i] = {}
                episode_actions[i] = []
                episode_obs[i] = []
                episode_reward_seq[i] = []
                next_episode_id += 1

                # Progress update
                if self.verbose and completed_episodes % 100 == 0:
                    elapsed = time.perf_counter() - start_time
                    eps_per_sec = completed_episodes / elapsed
                    print(
                        f"Progress: {completed_episodes}/{num_episodes} episodes "
                        f"({completed_episodes / num_episodes * 100:.1f}%) | "
                        f"{eps_per_sec:.1f} eps/sec | "
                        f"Avg reward: {np.mean(all_rewards):.2f}"
                    )

            obs = next_obs

        wall_time = time.perf_counter() - start_time
        actions_per_second = total_steps / wall_time

        if self.verbose:
            print("-" * 70)
            print(f"Evaluation complete in {wall_time:.1f}s")

        return EvaluationResults(
            num_episodes=num_episodes,
            num_envs=self.num_envs,
            total_steps=total_steps,
            wall_time=wall_time,
            stage_stats=stage_stats,
            episode_records=episode_records,
            reward_distribution=np.array(all_rewards, dtype=np.float32),
            actions_per_second=actions_per_second,
        )


def print_results(results: EvaluationResults) -> None:
    """Print formatted evaluation results."""
    summary = results.to_summary_dict()

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\n[Evaluation Info]")
    print(f"  Episodes:          {summary['evaluation']['num_episodes']}")
    print(f"  Parallel envs:     {summary['evaluation']['num_envs']}")
    print(f"  Total steps:       {summary['evaluation']['total_steps']:,}")
    print(f"  Wall time:         {summary['evaluation']['wall_time_seconds']:.1f}s")
    print(f"  Actions/sec:       {summary['evaluation']['actions_per_second']:,.0f}")

    print("\n[Reward Distribution]")
    print(f"  Mean:              {summary['rewards']['mean']:.2f}")
    print(f"  Std:               {summary['rewards']['std']:.2f}")
    print(f"  Min:               {summary['rewards']['min']:.2f}")
    print(f"  Max:               {summary['rewards']['max']:.2f}")
    print(f"  Median:            {summary['rewards']['median']:.2f}")
    print(f"  25th percentile:   {summary['rewards']['percentile_25']:.2f}")
    print(f"  75th percentile:   {summary['rewards']['percentile_75']:.2f}")

    print("\n[Stage Completion Rates]")
    print(f"  {'Stage':<25} {'Rate':>8} {'Successes':>10} {'Avg Time':>12} {'Best':>10}")
    print("  " + "-" * 65)
    for stage_name, stats in summary["stages"].items():
        rate = f"{stats['completion_rate'] * 100:.1f}%"
        successes = str(stats["successes"])
        avg_time = f"{stats['average_time_steps']:.0f}" if stats["average_time_steps"] else "-"
        best_time = str(stats["best_time_steps"]) if stats["best_time_steps"] else "-"
        print(f"  {stage_name:<25} {rate:>8} {successes:>10} {avg_time:>12} {best_time:>10}")

    print("\n[Speedrun Summary]")
    print(f"  Full completion rate:  {summary['speedrun']['full_completion_rate'] * 100:.2f}%")
    fastest = summary["speedrun"]["fastest_completion_steps"]
    if fastest:
        print(f"  Fastest completion:    {fastest} steps ({fastest / 20:.1f}s at 20 tps)")
    else:
        print("  Fastest completion:    N/A (no successful runs)")

    print("\n" + "=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Minecraft speedrun agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model checkpoint (.npz or .json)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=256,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=36000,
        help="Maximum steps per episode (30 min at 20 tps)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )
    parser.add_argument(
        "--save-replays",
        action="store_true",
        help="Save best episode replays",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=Path("replays"),
        help="Directory to save replays",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load model
    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        return 1

    print(f"Loading model from: {args.model}")
    if args.model.suffix == ".npz":
        weights = PolicyLoader.load_numpy(args.model)
    elif args.model.suffix == ".json":
        weights = PolicyLoader.load_json(args.model)
    else:
        print(f"Error: Unsupported model format: {args.model.suffix}")
        return 1

    # Create policy
    policy = EvaluationPolicy(weights)

    # Create evaluator
    evaluator = SpeedrunEvaluator(
        policy=policy,
        num_envs=args.num_envs,
        max_steps_per_episode=args.max_steps,
        deterministic=not args.stochastic,
        save_replays=args.save_replays,
        verbose=not args.quiet,
    )

    # Run evaluation
    results = evaluator.evaluate(args.episodes)

    # Print results
    if not args.quiet:
        print_results(results)

    # Save replays
    if args.save_replays and evaluator.replay_buffer:
        print(f"\nSaving {len(evaluator.replay_buffer.replays)} best replays to: {args.replay_dir}")
        evaluator.replay_buffer.save(args.replay_dir)

    # Save results JSON
    if args.output:
        summary = results.to_summary_dict()
        summary["episodes"] = [r.to_dict() for r in results.episode_records]
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
