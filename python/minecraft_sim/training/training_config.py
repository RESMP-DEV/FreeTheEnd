"""Training configuration for Minecraft speedrun PPO agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

import logging

logger = logging.getLogger(__name__)


@dataclass
class EnvConfig:
    """Environment configuration."""

    num_envs: int = 64
    max_episode_steps: int = 36_000  # 30 min at 20 TPS
    start_stage: int = 1
    seed: int | None = None


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class CurriculumConfig:
    """Curriculum learning settings."""

    enabled: bool = True
    stage_thresholds: list[float] = field(default_factory=lambda: [0.7, 0.7, 0.7, 0.7, 0.7])
    min_episodes_per_stage: int = 1000
    success_window: int = 100


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Sub-configs
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    # Training
    total_timesteps: int = 100_000_000
    checkpoint_freq: int = 500_000
    eval_freq: int = 100_000
    log_interval: int = 10

    # Normalization
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    # Paths
    log_dir: str = "logs"
    run_name: str = "speedrun"

    # Hardware
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load config from YAML file."""
        logger.debug("TrainingConfig.from_yaml: path=%s", path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Create config from dictionary."""
        logger.debug("TrainingConfig.from_dict: data=%s", data)
        config = cls()

        # Parse sub-configs
        if "env" in data:
            for key, val in data["env"].items():
                if hasattr(config.env, key):
                    setattr(config.env, key, val)

        if "ppo" in data:
            for key, val in data["ppo"].items():
                if hasattr(config.ppo, key):
                    setattr(config.ppo, key, val)

        if "curriculum" in data:
            for key, val in data["curriculum"].items():
                if hasattr(config.curriculum, key):
                    setattr(config.curriculum, key, val)

        # Parse top-level
        for key, val in data.items():
            if key not in ("env", "ppo", "curriculum") and hasattr(config, key):
                setattr(config, key, val)

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        logger.debug("TrainingConfig.to_dict called")
        return {
            "env": {
                "num_envs": self.env.num_envs,
                "max_episode_steps": self.env.max_episode_steps,
                "start_stage": self.env.start_stage,
                "seed": self.env.seed,
            },
            "ppo": {
                "learning_rate": self.ppo.learning_rate,
                "n_steps": self.ppo.n_steps,
                "batch_size": self.ppo.batch_size,
                "n_epochs": self.ppo.n_epochs,
                "gamma": self.ppo.gamma,
                "gae_lambda": self.ppo.gae_lambda,
                "clip_range": self.ppo.clip_range,
                "ent_coef": self.ppo.ent_coef,
                "vf_coef": self.ppo.vf_coef,
                "max_grad_norm": self.ppo.max_grad_norm,
            },
            "curriculum": {
                "enabled": self.curriculum.enabled,
                "stage_thresholds": self.curriculum.stage_thresholds,
                "min_episodes_per_stage": self.curriculum.min_episodes_per_stage,
                "success_window": self.curriculum.success_window,
            },
            "total_timesteps": self.total_timesteps,
            "checkpoint_freq": self.checkpoint_freq,
            "eval_freq": self.eval_freq,
            "log_interval": self.log_interval,
            "normalize_obs": self.normalize_obs,
            "normalize_reward": self.normalize_reward,
            "clip_obs": self.clip_obs,
            "clip_reward": self.clip_reward,
            "log_dir": self.log_dir,
            "run_name": self.run_name,
            "device": self.device,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        logger.debug("TrainingConfig.to_yaml: path=%s", path)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
