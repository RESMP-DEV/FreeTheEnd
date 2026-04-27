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

from .vec_env import VecFreeTheEndEnv

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

import logging

logger = logging.getLogger(__name__)

        env = SB3VecFreeTheEndEnv(num_envs=64, start_stage=1)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10_000_000)

    Args:
        num_envs: Number of parallel environments.
        start_stage: Initial curriculum stage (1-indexed).
        curriculum: Whether to enable automatic curriculum progression. (Currently ignored)
        seeds: Optional list of seeds for each environment. (Currently ignored)
        max_episode_steps: Maximum steps per episode before truncation. (Currently ignored)
    """

    def __init__(
        self,
        num_envs: int = 64,
        start_stage: int = 1,
        curriculum: bool = True,
        seeds: list[int] | None = None,
        max_episode_steps: int = 36000,
    ) -> None:
        # VecFreeTheEndEnv only supports num_envs and start_stage currently
        logger.info("SB3VecFreeTheEndEnv.__init__: num_envs=%s, start_stage=%s, curriculum=%s, seeds=%s", num_envs, start_stage, curriculum, seeds)
        self._vec_env = VecFreeTheEndEnv(
            num_envs=num_envs,
            start_stage=start_stage,
        )
        self._actions: NDArray[np.int64] | None = None

        super().__init__(
            num_envs=num_envs,
            observation_space=self._vec_env.observation_space,
            action_space=self._vec_env.action_space,
        )

    def reset(self) -> VecEnvObs:
        """Reset all environments and return initial observations."""
        logger.debug("SB3VecFreeTheEndEnv.reset called")
        return self._vec_env.reset()

    def step_async(self, actions: NDArray[np.int64]) -> None:
        """Store actions for async step execution."""
        logger.debug("SB3VecFreeTheEndEnv.step_async: actions=%s", actions)
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        """Execute stored actions and return results."""
        logger.debug("SB3VecFreeTheEndEnv.step_wait called")
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        obs, rewards, dones, infos = self._vec_env.step(self._actions)
        return obs, rewards, dones, infos

    def close(self) -> None:
        """Clean up environment resources."""
        logger.info("SB3VecFreeTheEndEnv.close called")
        self._vec_env.close()

    def get_attr(self, attr_name: str, indices: Sequence[int] | None = None) -> list[Any]:
        """Get attribute from the underlying environment."""
        logger.debug("SB3VecFreeTheEndEnv.get_attr: attr_name=%s, indices=%s", attr_name, indices)
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self._vec_env, attr_name) for _ in indices]

    def set_attr(self, attr_name: str, value: Any, indices: Sequence[int] | None = None) -> None:
        """Set attribute on the underlying environment."""
        logger.debug("SB3VecFreeTheEndEnv.set_attr: attr_name=%s, value=%s, indices=%s", attr_name, value, indices)
        setattr(self._vec_env, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Sequence[int] | None = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        """Call a method on the underlying environment."""
        logger.debug("SB3VecFreeTheEndEnv.env_method: method_name=%s", method_name)
        if indices is None:
            indices = range(self.num_envs)
        method = getattr(self._vec_env, method_name)
        return [method(*method_args, **method_kwargs) for _ in indices]

    def env_is_wrapped(
        self, wrapper_class: type, indices: Sequence[int] | None = None
    ) -> list[bool]:
        """Check if environments are wrapped with a specific wrapper."""
        logger.debug("SB3VecFreeTheEndEnv.env_is_wrapped: wrapper_class=%s, indices=%s", wrapper_class, indices)
        n = len(indices) if indices else self.num_envs
        return [False] * n

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Set seeds for all environments."""
        logger.debug("SB3VecFreeTheEndEnv.seed: seed=%s", seed)
        if seed is not None:
            self._vec_env.seeds = np.arange(seed, seed + self.num_envs)
        return list(self._vec_env.seeds)

    def get_images(self) -> Sequence[NDArray[np.uint8]]:
        """Get rendered images from environments."""
        logger.debug("SB3VecFreeTheEndEnv.get_images called")
        raise NotImplementedError("Rendering not yet supported")

    # Curriculum-specific methods

    def get_stage_distribution(self) -> dict[int, int]:
        """Get distribution of environments across curriculum stages."""
        logger.debug("SB3VecFreeTheEndEnv.get_stage_distribution called")
        return self._vec_env.get_stage_distribution()

    def set_stages(self, env_ids: NDArray[np.int64], stages: NDArray[np.int64]) -> None:
        """Set curriculum stages for specific environments."""
        logger.debug("SB3VecFreeTheEndEnv.set_stages: env_ids=%s, stages=%s", env_ids, stages)
        self._vec_env.set_stages(env_ids, stages)
