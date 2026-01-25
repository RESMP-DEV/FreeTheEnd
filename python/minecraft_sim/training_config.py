"""Configurable training setup with YAML config support.

This module provides a hierarchical configuration system for RL training,
supporting YAML serialization/deserialization with nested dataclass fields.

Example:
    >>> config = TrainingConfig.from_yaml("configs/train.yaml")
    >>> config.ppo.learning_rate = 1e-4
    >>> config.to_yaml("configs/train_modified.yaml")
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


def _merge_nested(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, creating nested structure."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_nested(result[key], value)
        else:
            result[key] = value
    return result


def _is_dataclass_type(tp: type) -> bool:
    """Check if a type is a dataclass."""
    return dataclasses.is_dataclass(tp) and isinstance(tp, type)


def _dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Recursively construct a dataclass from a dictionary."""
    # Get resolved type hints (handles string annotations from __future__)
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}

    for name, value in data.items():
        if name not in hints:
            continue
        field_type = hints[name]

        # Handle Union types (including Optional which is Union[X, None])
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            # Filter out NoneType and get the first real type
            non_none_types = [a for a in args if a is not type(None)]
            if non_none_types:
                field_type = non_none_types[0]

        # Check if field is a dataclass and value is a dict
        if isinstance(value, dict) and _is_dataclass_type(field_type):
            kwargs[name] = _dataclass_from_dict(field_type, value)
        else:
            kwargs[name] = value

    return cls(**kwargs)


@dataclass
class EnvConfig:
    """Environment configuration.

    Attributes:
        num_envs: Number of parallel environments to run.
        start_stage: Initial curriculum stage (1-6).
        curriculum: Whether to enable automatic curriculum progression.
        max_episode_steps: Maximum steps per episode before truncation.
        seed: Random seed for reproducibility (None for random).
    """

    num_envs: int = 64
    start_stage: int = 1
    curriculum: bool = True
    max_episode_steps: int = 36000
    seed: int | None = None


@dataclass
class CurriculumSettings:
    """Curriculum learning settings.

    Attributes:
        enabled: Whether curriculum learning is active.
        min_episodes_per_stage: Minimum episodes before advancement check.
        advancement_threshold: Success rate (0-1) required to advance.
        allow_regression: Whether to regress on poor performance.
        regression_threshold: Success rate below which to regress (if enabled).
        max_episodes_per_stage: Maximum episodes before forced advancement.
    """

    enabled: bool = True
    min_episodes_per_stage: int = 20
    advancement_threshold: float = 0.7
    allow_regression: bool = False
    regression_threshold: float = 0.2
    max_episodes_per_stage: int = 1000


@dataclass
class PPOConfig:
    """PPO algorithm configuration.

    Attributes:
        learning_rate: Adam optimizer learning rate.
        n_steps: Number of steps to run per environment per update.
        batch_size: Minibatch size for gradient updates.
        n_epochs: Number of epochs per PPO update.
        gamma: Discount factor for rewards.
        gae_lambda: Factor for Generalized Advantage Estimation.
        clip_range: PPO clipping parameter for policy updates.
        ent_coef: Entropy coefficient for exploration.
        vf_coef: Value function coefficient in loss.
        max_grad_norm: Maximum norm for gradient clipping.
    """

    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class NetworkConfig:
    """Neural network architecture configuration.

    Attributes:
        policy_layers: Hidden layer sizes for policy network.
        value_layers: Hidden layer sizes for value network.
        activation: Activation function name (relu, tanh, elu).
        ortho_init: Whether to use orthogonal initialization.
        shared_encoder: Whether policy and value share encoder.
        use_lstm: Whether to use LSTM for temporal dependencies.
        lstm_hidden_size: LSTM hidden state size if enabled.
    """

    policy_layers: list[int] = field(default_factory=lambda: [256, 256])
    value_layers: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    ortho_init: bool = True
    shared_encoder: bool = False
    use_lstm: bool = False
    lstm_hidden_size: int = 256


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration.

    Attributes:
        log_interval: Episodes between log outputs.
        verbose: Verbosity level (0=none, 1=info, 2=debug).
        save_replay_buffer: Whether to save experience buffer.
        wandb_project: W&B project name (None to disable).
        wandb_entity: W&B entity/team name.
        tensorboard: Whether to enable TensorBoard logging.
    """

    log_interval: int = 1
    verbose: int = 1
    save_replay_buffer: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    tensorboard: bool = True


@dataclass
class TrainingConfig:
    """Complete training configuration.

    Hierarchical configuration supporting all training parameters with
    YAML serialization and validation.

    Attributes:
        run_name: Identifier for this training run.
        env: Environment configuration.
        curriculum: Curriculum learning settings.
        ppo: PPO algorithm hyperparameters.
        network: Neural network architecture.
        logging: Logging and monitoring settings.
        total_timesteps: Total environment steps to train.
        checkpoint_freq: Steps between model checkpoints.
        eval_freq: Steps between evaluation runs.
        n_eval_episodes: Episodes per evaluation.
        log_dir: Directory for logs.
        checkpoint_dir: Directory for model checkpoints.
        tensorboard_log: Directory for TensorBoard logs.
        normalize_obs: Whether to normalize observations.
        normalize_reward: Whether to normalize rewards.
        clip_obs: Observation clipping range.
        clip_reward: Reward clipping range.
        device: Compute device (auto, cpu, cuda, mps).
    """

    # Run identification
    run_name: str = "free_the_end"

    # Nested configurations
    env: EnvConfig = field(default_factory=EnvConfig)
    curriculum: CurriculumSettings = field(default_factory=CurriculumSettings)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Training settings
    total_timesteps: int = 100_000_000
    checkpoint_freq: int = 100_000
    eval_freq: int = 50_000
    n_eval_episodes: int = 10

    # Paths
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    tensorboard_log: str = "tensorboard"

    # Normalization
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    # Device
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            TrainingConfig instance with loaded values.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return _dataclass_from_dict(cls, data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            TrainingConfig instance.
        """
        return _dataclass_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            Nested dictionary representation.
        """
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def merge(self, overrides: dict[str, Any]) -> TrainingConfig:
        """Create a new config with overrides applied.

        Args:
            overrides: Dictionary of values to override.

        Returns:
            New TrainingConfig with merged values.
        """
        base = self.to_dict()
        merged = _merge_nested(base, overrides)
        return self.from_dict(merged)

    def validate(self) -> list[str]:
        """Validate configuration values.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Environment validation
        if self.env.num_envs < 1:
            errors.append("env.num_envs must be >= 1")
        if not 1 <= self.env.start_stage <= 6:
            errors.append("env.start_stage must be between 1 and 6")
        if self.env.max_episode_steps < 1:
            errors.append("env.max_episode_steps must be >= 1")

        # Curriculum validation
        if self.curriculum.advancement_threshold <= 0 or self.curriculum.advancement_threshold > 1:
            errors.append("curriculum.advancement_threshold must be in (0, 1]")
        if self.curriculum.min_episodes_per_stage < 1:
            errors.append("curriculum.min_episodes_per_stage must be >= 1")

        # PPO validation
        if self.ppo.learning_rate <= 0:
            errors.append("ppo.learning_rate must be > 0")
        if self.ppo.n_steps < 1:
            errors.append("ppo.n_steps must be >= 1")
        if self.ppo.batch_size < 1:
            errors.append("ppo.batch_size must be >= 1")
        if self.ppo.clip_range <= 0 or self.ppo.clip_range >= 1:
            errors.append("ppo.clip_range must be in (0, 1)")
        if self.ppo.gamma <= 0 or self.ppo.gamma > 1:
            errors.append("ppo.gamma must be in (0, 1]")

        # Network validation
        if not self.network.policy_layers:
            errors.append("network.policy_layers cannot be empty")
        if not self.network.value_layers:
            errors.append("network.value_layers cannot be empty")
        if self.network.activation not in ("relu", "tanh", "elu", "selu", "gelu"):
            errors.append(f"network.activation '{self.network.activation}' not supported")

        # Training validation
        if self.total_timesteps < 1:
            errors.append("total_timesteps must be >= 1")
        if self.checkpoint_freq < 1:
            errors.append("checkpoint_freq must be >= 1")

        # Normalization validation
        if self.clip_obs <= 0:
            errors.append("clip_obs must be > 0")
        if self.clip_reward <= 0:
            errors.append("clip_reward must be > 0")

        return errors


# Default configurations for common scenarios
DEFAULT_CONFIG = TrainingConfig()

FAST_ITERATION_CONFIG = TrainingConfig(
    run_name="fast_iteration",
    env=EnvConfig(num_envs=8, max_episode_steps=4000),
    ppo=PPOConfig(n_steps=64, batch_size=64),
    total_timesteps=100_000,
    checkpoint_freq=10_000,
    eval_freq=5_000,
)

FULL_TRAINING_CONFIG = TrainingConfig(
    run_name="full_training",
    env=EnvConfig(num_envs=128),
    ppo=PPOConfig(
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
    ),
    network=NetworkConfig(
        policy_layers=[512, 256, 256],
        value_layers=[512, 256, 256],
        use_lstm=True,
        lstm_hidden_size=512,
    ),
    total_timesteps=500_000_000,
    checkpoint_freq=500_000,
)
