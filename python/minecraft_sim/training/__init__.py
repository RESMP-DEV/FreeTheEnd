"""Training utilities for minecraft_sim RL agents.

This module provides:
- TrainingCallbacks: Callback class for PPO training hooks (requires stable_baselines3)
- TrainingConfig: Configuration dataclass for training hyperparameters
"""

from .training_config import TrainingConfig

__all__ = ["TrainingConfig"]

# Optional: Import callbacks only if stable_baselines3 is available
try:
    from .callbacks import CheckpointCallback, CurriculumCallback, LoggingCallback

    __all__.extend(["CurriculumCallback", "LoggingCallback", "CheckpointCallback"])
except ImportError:
    pass  # stable_baselines3 not installed
