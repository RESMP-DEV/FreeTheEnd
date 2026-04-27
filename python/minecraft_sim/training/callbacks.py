"""Custom callbacks for Minecraft speedrun training."""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv


class CurriculumCallback(BaseCallback):
    """Curriculum learning callback for staged progression.

    Monitors success rate and advances curriculum stage when threshold is met.
    """

    def __init__(
        self,
        env: VecEnv,
        stage_thresholds: list[float] | None = None,
        min_episodes: int = 1000,
        success_window: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env = env
        self.stage_thresholds = stage_thresholds or [0.7] * 5
        self.min_episodes = min_episodes
        self.success_window = success_window

        self.current_stage = 1
        self.episodes_in_stage = 0
        self.success_history: deque[bool] = deque(maxlen=success_window)
        self.stage_start_timestep = 0

    def _on_step(self) -> bool:
        # Check for episode completions in info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episodes_in_stage += 1
                success = info.get("success", False) or info.get("stage_complete", False)
                self.success_history.append(success)

                # Check for stage advancement
                if self._should_advance_stage():
                    self._advance_stage()

        return True

    def _should_advance_stage(self) -> bool:
        """Check if we should advance to next stage."""
        if self.episodes_in_stage < self.min_episodes:
            return False

        if len(self.success_history) < self.success_window:
            return False

        success_rate = sum(self.success_history) / len(self.success_history)
        threshold = self.stage_thresholds[
            min(self.current_stage - 1, len(self.stage_thresholds) - 1)
        ]

        return success_rate >= threshold

    def _advance_stage(self) -> None:
        """Advance to next curriculum stage."""
        self.current_stage += 1
        self.episodes_in_stage = 0
        self.success_history.clear()
        self.stage_start_timestep = self.num_timesteps

        if self.verbose > 0:
            self.logger.record("curriculum/stage", self.current_stage)
            print(
                f"[Curriculum] Advanced to stage {self.current_stage} at timestep {self.num_timesteps:,}"
            )

        # Update environment if it supports curriculum
        if hasattr(self.env, "set_stage"):
            self.env.set_stage(self.current_stage)
        elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "set_stage"):
            self.env.unwrapped.set_stage(self.current_stage)

    def _on_training_end(self) -> None:
        if self.verbose > 0:
            print(f"[Curriculum] Final stage: {self.current_stage}")


class MetricsCallback(BaseCallback):
    """Detailed metrics logging callback.

    Tracks and logs:
    - Episode statistics (length, reward, success)
    - Stage-specific metrics
    - Training throughput (SPS)
    - Time estimates
    """

    def __init__(
        self,
        log_path: str | Path,
        save_freq: int = 10_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.save_freq = save_freq

        self.metrics: dict[str, list[Any]] = {
            "timesteps": [],
            "episode_rewards": [],
            "episode_lengths": [],
            "success_rate": [],
            "sps": [],
            "wall_time": [],
        }
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.successes: list[bool] = []

        self.start_time = time.time()
        self.last_log_timestep = 0
        self.last_log_time = self.start_time

    def _on_step(self) -> bool:
        # Collect episode info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.successes.append(info.get("success", False))

        # Periodic logging
        if self.num_timesteps - self.last_log_timestep >= self.save_freq:
            self._log_metrics()

        return True

    def _log_metrics(self) -> None:
        """Log accumulated metrics."""
        current_time = time.time()
        elapsed = current_time - self.last_log_time
        timesteps_done = self.num_timesteps - self.last_log_timestep

        # Calculate SPS
        sps = timesteps_done / max(elapsed, 1e-6)

        # Episode statistics
        if self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_length = np.mean(self.episode_lengths[-100:])
            success_rate = np.mean(self.successes[-100:]) if self.successes else 0.0
        else:
            mean_reward = 0.0
            mean_length = 0.0
            success_rate = 0.0

        # Store
        self.metrics["timesteps"].append(self.num_timesteps)
        self.metrics["episode_rewards"].append(float(mean_reward))
        self.metrics["episode_lengths"].append(float(mean_length))
        self.metrics["success_rate"].append(float(success_rate))
        self.metrics["sps"].append(float(sps))
        self.metrics["wall_time"].append(current_time - self.start_time)

        # Log to tensorboard
        if self.logger:
            self.logger.record("metrics/mean_reward", mean_reward)
            self.logger.record("metrics/mean_episode_length", mean_length)
            self.logger.record("metrics/success_rate", success_rate)
            self.logger.record("metrics/sps", sps)
            self.logger.record("metrics/episodes", len(self.episode_rewards))

        # Print progress
        if self.verbose > 0:
            total_time = current_time - self.start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            print(
                f"[Metrics] {self.num_timesteps:>10,} steps | "
                f"Reward: {mean_reward:>8.2f} | "
                f"Success: {success_rate:>6.1%} | "
                f"SPS: {sps:>8,.0f} | "
                f"Time: {hours}h{minutes:02d}m"
            )

        # Save to file
        metrics_file = self.log_path / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)

        self.last_log_timestep = self.num_timesteps
        self.last_log_time = current_time

    def _on_training_end(self) -> None:
        self._log_metrics()
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)

        if self.verbose > 0:
            print(f"[Metrics] Training complete in {hours}h{minutes:02d}m")
            print(f"[Metrics] Total timesteps: {self.num_timesteps:,}")
            print(f"[Metrics] Total episodes: {len(self.episode_rewards):,}")
            if self.episode_rewards:
                print(f"[Metrics] Final mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")


class WandBCallback(BaseCallback):
    """Weights & Biases logging callback.

    Logs metrics to wandb for experiment tracking.
    """

    def __init__(
        self,
        project: str = "minecraft-speedrun",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        log_freq: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.log_freq = log_freq

        self._wandb = None
        self.episode_rewards: list[float] = []
        self.successes: list[bool] = []

    def _on_training_start(self) -> None:
        try:
            import wandb

            self._wandb = wandb
            wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config,
                reinit=True,
            )
        except ImportError:

import logging

logger = logging.getLogger(__name__)

            if self.verbose > 0:
                print("[WandB] wandb not installed, skipping logging")
            self._wandb = None

    def _on_step(self) -> bool:
        if self._wandb is None:
            return True

        # Collect episode info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.successes.append(info.get("success", False))

        # Log periodically
        if self.num_timesteps % self.log_freq == 0 and self.episode_rewards:
            self._wandb.log(
                {
                    "timesteps": self.num_timesteps,
                    "mean_reward": np.mean(self.episode_rewards[-100:]),
                    "success_rate": np.mean(self.successes[-100:]) if self.successes else 0,
                    "episodes": len(self.episode_rewards),
                },
                step=self.num_timesteps,
            )

        return True

    def _on_training_end(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
