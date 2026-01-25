"""Stable Baselines 3 compatible VecEnv wrapper for FreeTheEnd environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)

from .vec_env import StageID, VecFreeTheEndEnv

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SB3VecFreeTheEndEnv(VecEnv):
    """
    Stable Baselines 3 compatible vectorized environment.

    Implements full VecEnv interface for seamless integration with SB3
    algorithms (PPO, A2C, SAC, etc.)

    Example:
        from minecraft_sim import SB3VecFreeTheEndEnv
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize

        env = SB3VecFreeTheEndEnv(num_envs=64, start_stage=1)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10_000_000)

    Args:
        num_envs: Number of parallel environments.
        start_stage: Initial curriculum stage (1-indexed).
        curriculum: Whether to enable automatic curriculum progression.
        seeds: Optional list of seeds for each environment.
        max_episode_steps: Maximum steps per episode before truncation.
    """

    def __init__(
        self,
        num_envs: int = 64,
        start_stage: int = 1,
        curriculum: bool = True,
        seeds: list[int] | None = None,
        max_episode_steps: int = 36000,
    ) -> None:
        self._vec_env = VecFreeTheEndEnv(
            num_envs=num_envs,
            start_stage=StageID(start_stage),
            curriculum=curriculum,
            seeds=seeds,
            max_episode_steps=max_episode_steps,
        )
        self._actions: NDArray[np.int64] | None = None

        super().__init__(
            num_envs=num_envs,
            observation_space=self._vec_env.observation_space,
            action_space=self._vec_env.action_space,
        )

    def reset(self) -> VecEnvObs:
        """Reset all environments and return initial observations."""
        return self._vec_env.reset()

    def step_async(self, actions: NDArray[np.int64]) -> None:
        """Store actions for async step execution."""
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        """Execute stored actions and return results."""
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        obs, rewards, dones, infos = self._vec_env.step(self._actions)
        return obs, rewards, dones, infos

    def close(self) -> None:
        """Clean up environment resources."""
        self._vec_env.close()

    def get_attr(self, attr_name: str, indices: Sequence[int] | None = None) -> list[Any]:
        """Get attribute from the underlying environment."""
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self._vec_env, attr_name) for _ in indices]

    def set_attr(self, attr_name: str, value: Any, indices: Sequence[int] | None = None) -> None:
        """Set attribute on the underlying environment."""
        setattr(self._vec_env, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Sequence[int] | None = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        """Call a method on the underlying environment."""
        if indices is None:
            indices = range(self.num_envs)
        method = getattr(self._vec_env, method_name)
        return [method(*method_args, **method_kwargs) for _ in indices]

    def env_is_wrapped(
        self, wrapper_class: type, indices: Sequence[int] | None = None
    ) -> list[bool]:
        """Check if environments are wrapped with a specific wrapper."""
        n = len(indices) if indices else self.num_envs
        return [False] * n

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Set seeds for all environments."""
        if seed is not None:
            self._vec_env.seeds = np.arange(seed, seed + self.num_envs)
        return list(self._vec_env.seeds)

    def get_images(self) -> Sequence[NDArray[np.uint8]]:
        """Get rendered images from environments."""
        raise NotImplementedError("Rendering not yet supported")

    # Curriculum-specific methods

    def get_stage_distribution(self) -> dict[int, int]:
        """Get distribution of environments across curriculum stages."""
        return self._vec_env.get_stage_distribution()

    def set_stages(self, env_ids: NDArray[np.int64], stages: NDArray[np.int64]) -> None:
        """Set curriculum stages for specific environments."""
        self._vec_env.set_stages(env_ids, stages)
