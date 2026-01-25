"""Vectorized environment for parallel Minecraft Dragon Fight training.

This module provides a vectorized wrapper around the MC189Simulator
for efficient parallel environment stepping during RL training.

Compatible with:
- Stable Baselines 3 (SB3)
- CleanRL
- RLlib
- Custom training loops
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._compat import HAS_GYMNASIUM, spaces

# Import mc189_core directly (it's a .so file in this directory)
# Avoid importing from package to prevent circular imports
mc189_core = None
try:
    # First try direct import (when .so is in sys.path)
    import mc189_core as _mc189_core

    mc189_core = _mc189_core
except ImportError:
    pass

if mc189_core is None:
    try:
        # Try importing the .so from the same directory
        import importlib.util

        so_path = Path(__file__).parent / "mc189_core.cpython-312-darwin.so"
        if so_path.exists():
            spec = importlib.util.spec_from_file_location("mc189_core", so_path)
            if spec and spec.loader:
                mc189_core = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mc189_core)
    except Exception:
        pass
    except Exception:
        pass


class VecDragonFightEnv:
    """Vectorized Dragon Fight environment for parallel training.

    Wraps the MC189Simulator to provide a gym-like vectorized interface
    for training RL agents on the Minecraft Ender Dragon fight.

    Args:
        num_envs: Number of parallel environments. Defaults to 64.
        shader_dir: Path to shader directory. If None, uses default path.
        observation_size: Size of observation vector per environment. Defaults to 48.

    Attributes:
        num_envs: Number of parallel environments.
        observation_size: Size of observation vector.
        sim: The underlying MC189Simulator instance.

    Example:
        >>> env = VecDragonFightEnv(num_envs=64)
        >>> obs = env.reset()
        >>> assert obs.shape == (64, 48)
        >>> actions = np.random.randint(0, 10, size=64)
        >>> obs, rewards, dones, infos = env.step(actions)
    """

    def __init__(
        self,
        num_envs: int = 64,
        shader_dir: str | Path | None = None,
        observation_size: int = 48,
    ) -> None:
        if mc189_core is None:
            raise ImportError(
                "mc189_core module not found. "
                "Build the C++ extension: cd cpp/build && cmake .. && make"
            )

        self.num_envs = num_envs
        self.observation_size = observation_size

        config = mc189_core.SimulatorConfig()
        config.num_envs = num_envs

        if shader_dir is not None:
            config.shader_dir = str(shader_dir)
        else:
            default_shader_dir = Path(__file__).parent.parent.parent / "shaders"
            if default_shader_dir.exists():
                config.shader_dir = str(default_shader_dir)

        self.sim = mc189_core.MC189Simulator(config)
        self._actions_buffer = np.zeros(num_envs, dtype=np.int32)

    def reset(self) -> NDArray[np.float32]:
        """Reset all environments and return initial observations.

        Returns:
            Observations array of shape (num_envs, observation_size).
        """
        self.sim.reset()
        self._actions_buffer.fill(0)
        self.sim.step(self._actions_buffer)
        obs = self.sim.get_observations()
        return obs.reshape(self.num_envs, self.observation_size).astype(np.float32)

    def step(
        self, actions: NDArray[np.int32] | list[int]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        """Step all environments with the given actions.

        Args:
            actions: Action array of shape (num_envs,). Each action should
                be an integer representing the discrete action to take.

        Returns:
            Tuple of (observations, rewards, dones, infos):
                - observations: Shape (num_envs, observation_size)
                - rewards: Shape (num_envs,)
                - dones: Shape (num_envs,), True if episode ended
                - infos: List of dicts, one per environment
        """
        actions_arr = np.asarray(actions, dtype=np.int32)
        if actions_arr.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {actions_arr.shape[0]}")

        self.sim.step(actions_arr)

        obs = (
            self.sim.get_observations()
            .reshape(self.num_envs, self.observation_size)
            .astype(np.float32)
        )
        rewards = np.asarray(self.sim.get_rewards(), dtype=np.float32)
        dones = np.asarray(self.sim.get_dones(), dtype=np.bool_)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        return obs, rewards, dones, infos

    def close(self) -> None:
        """Clean up simulator resources."""
        if hasattr(self, "sim") and self.sim is not None:
            del self.sim

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.close()

    @property
    def action_space_size(self) -> int:
        """Return the number of discrete actions available."""
        if hasattr(self.sim, "get_action_space_size"):
            return self.sim.get_action_space_size()
        return 17  # Dragon fight: 17 discrete actions

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Return the shape of a single observation."""
        return (self.observation_size,)


class SB3VecDragonFightEnv:
    """Stable Baselines 3 compatible vectorized Dragon Fight environment.

    This class implements the VecEnv interface expected by SB3 and CleanRL.

    Args:
        num_envs: Number of parallel environments.
        shader_dir: Path to shader directory.

    Example:
        >>> from stable_baselines3 import PPO
        >>> env = SB3VecDragonFightEnv(num_envs=64)
        >>> model = PPO("MlpPolicy", env, verbose=1)
        >>> model.learn(total_timesteps=1_000_000)
    """

    def __init__(
        self,
        num_envs: int = 64,
        shader_dir: str | Path | None = None,
    ) -> None:
        if mc189_core is None:
            raise ImportError("mc189_core not found")
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium required for SB3VecDragonFightEnv")

        self.num_envs = num_envs
        self._obs_size = 48
        self._num_actions = 17

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._num_actions)

        # Single observation/action spaces (required by SB3)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

        # Initialize simulator
        config = mc189_core.SimulatorConfig()
        config.num_envs = num_envs
        if shader_dir is not None:
            config.shader_dir = str(shader_dir)
        else:
            default_shader_dir = Path(__file__).parent.parent.parent / "shaders"
            if default_shader_dir.exists():
                config.shader_dir = str(default_shader_dir)

        self.sim = mc189_core.MC189Simulator(config)
        self._actions_buffer = np.zeros(num_envs, dtype=np.int32)

        # Episode stats tracking (for SB3 logging)
        self._episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(num_envs, dtype=np.int32)

    def reset(self) -> NDArray[np.float32]:
        """Reset all environments."""
        self.sim.reset()
        self._actions_buffer.fill(0)
        self.sim.step(self._actions_buffer)
        self._episode_rewards.fill(0)
        self._episode_lengths.fill(0)
        return self._get_obs()

    def step(
        self, actions: NDArray[np.int32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        """Step all environments."""
        actions_arr = np.asarray(actions, dtype=np.int32).ravel()
        self.sim.step(actions_arr)

        obs = self._get_obs()
        rewards = np.asarray(self.sim.get_rewards(), dtype=np.float32)
        dones = np.asarray(self.sim.get_dones(), dtype=np.bool_)

        # Track episode stats
        self._episode_rewards += rewards
        self._episode_lengths += 1

        # Build info dicts with terminal observation and episode stats
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["episode"] = {
                    "r": float(self._episode_rewards[i]),
                    "l": int(self._episode_lengths[i]),
                }
                # Terminal observation (required by SB3)
                infos[i]["terminal_observation"] = obs[i].copy()
                # Reset episode tracking for this env
                self._episode_rewards[i] = 0
                self._episode_lengths[i] = 0

        return obs, rewards, dones, infos

    def step_async(self, actions: NDArray[np.int32]) -> None:
        """Async step (required by SB3 VecEnv interface)."""
        self._pending_actions = np.asarray(actions, dtype=np.int32)

    def step_wait(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        """Wait for async step result (required by SB3 VecEnv interface)."""
        return self.step(self._pending_actions)

    def _get_obs(self) -> NDArray[np.float32]:
        """Get observations from simulator."""
        obs = self.sim.get_observations().reshape(self.num_envs, self._obs_size)
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "sim") and self.sim is not None:
            del self.sim

    def env_is_wrapped(self, wrapper_class: type, indices: list[int] | None = None) -> list[bool]:
        """Check if envs are wrapped (required by SB3)."""
        return [False] * self.num_envs

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: list[int] | None = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        """Call method on sub-environments (required by SB3)."""
        return [None] * self.num_envs

    def get_attr(self, attr_name: str, indices: list[int] | None = None) -> list[Any]:
        """Get attribute from sub-environments (required by SB3)."""
        return [getattr(self, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices: list[int] | None = None) -> None:
        """Set attribute on sub-environments (required by SB3)."""
        pass

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Set seed for all environments (required by SB3)."""
        return [seed] * self.num_envs

    @property
    def unwrapped(self) -> SB3VecDragonFightEnv:
        """Return unwrapped environment."""
        return self


class FreeTheEndEnv:
    """Single environment for Minecraft speedrun curriculum training.

    This environment supports staged curriculum learning for training
    agents to beat Minecraft (reach The End and defeat the dragon).

    Args:
        stage: Stage ID (1-6) or None for full game. Each stage focuses on
            different skills from basic survival to the final dragon fight.
        shader_dir: Path to shader directory. If None, uses default path.
        observation_size: Size of observation vector. Defaults to 48.

    Example:
        >>> env = FreeTheEndEnv(stage=1)  # Basic survival stage
        >>> obs = env.reset()
        >>> action = env.sample_action()
        >>> obs, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        stage: int | None = None,
        shader_dir: str | Path | None = None,
        observation_size: int = 48,
    ) -> None:
        if mc189_core is None:
            raise ImportError(
                "mc189_core module not found. "
                "Build the C++ extension: cd cpp/build && cmake .. && make"
            )

        self.stage = stage
        self.observation_size = observation_size
        self._num_actions = 17

        config = mc189_core.SimulatorConfig()
        config.num_envs = 1

        if shader_dir is not None:
            config.shader_dir = str(shader_dir)
        else:
            default_shader_dir = Path(__file__).parent.parent.parent / "shaders"
            if default_shader_dir.exists():
                config.shader_dir = str(default_shader_dir)

        self.sim = mc189_core.MC189Simulator(config)
        self._step_count = 0
        self._max_steps = 36000  # 30 minutes at 20 tps

    def reset(self) -> NDArray[np.float32]:
        """Reset the environment and return initial observation.

        Returns:
            Observation array of shape (observation_size,).
        """
        self.sim.reset()
        self.sim.step(np.array([0], dtype=np.int32))
        self._step_count = 0
        obs = self.sim.get_observations()
        return np.clip(obs.flatten(), 0.0, 1.0).astype(np.float32)

    def step(
        self, action: int | np.ndarray
    ) -> tuple[NDArray[np.float32], float, bool, dict[str, Any]]:
        """Step the environment with the given action.

        Args:
            action: Discrete action index (0 to 16).

        Returns:
            Tuple of (observation, reward, done, info).
        """
        action_int = int(action) if isinstance(action, (int, np.integer)) else int(action.item())
        self._step_count += 1

        self.sim.step(np.array([action_int], dtype=np.int32))

        obs = np.clip(self.sim.get_observations().flatten(), 0.0, 1.0).astype(np.float32)
        reward = float(self.sim.get_rewards()[0])
        done = bool(self.sim.get_dones()[0]) or self._step_count >= self._max_steps

        info: dict[str, Any] = {
            "step_count": self._step_count,
            "stage": self.stage,
        }

        return obs, reward, done, info

    def sample_action(self) -> int:
        """Sample a random action from the action space."""
        return int(np.random.randint(0, self._num_actions))

    def close(self) -> None:
        """Clean up simulator resources."""
        if hasattr(self, "sim") and self.sim is not None:
            del self.sim

    @property
    def action_space_size(self) -> int:
        """Return the number of discrete actions available."""
        return self._num_actions

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Return the shape of a single observation."""
        return (self.observation_size,)


class VecFreeTheEndEnv:
    """Vectorized environment for Minecraft speedrun curriculum training.

    Wraps the MC189Simulator to provide a gym-like vectorized interface
    for training RL agents through the Minecraft speedrun curriculum.

    Args:
        num_envs: Number of parallel environments. Defaults to 64.
        start_stage: Initial stage ID (1-6). Defaults to 1.
        shader_dir: Path to shader directory. If None, uses default path.
        observation_size: Size of observation vector per environment. Defaults to 48.

    Example:
        >>> env = VecFreeTheEndEnv(num_envs=64, start_stage=1)
        >>> obs = env.reset()
        >>> assert obs.shape == (64, 48)
        >>> actions = np.random.randint(0, 17, size=64)
        >>> obs, rewards, dones, infos = env.step(actions)
    """

    def __init__(
        self,
        num_envs: int = 64,
        start_stage: int = 1,
        shader_dir: str | Path | None = None,
        observation_size: int = 48,
    ) -> None:
        if mc189_core is None:
            raise ImportError(
                "mc189_core module not found. "
                "Build the C++ extension: cd cpp/build && cmake .. && make"
            )

        self.num_envs = num_envs
        self.start_stage = start_stage
        self.observation_size = observation_size

        config = mc189_core.SimulatorConfig()
        config.num_envs = num_envs

        if shader_dir is not None:
            config.shader_dir = str(shader_dir)
        else:
            default_shader_dir = Path(__file__).parent.parent.parent / "shaders"
            if default_shader_dir.exists():
                config.shader_dir = str(default_shader_dir)

        self.sim = mc189_core.MC189Simulator(config)
        self._actions_buffer = np.zeros(num_envs, dtype=np.int32)

    def reset(self) -> NDArray[np.float32]:
        """Reset all environments and return initial observations.

        Returns:
            Observations array of shape (num_envs, observation_size).
        """
        self.sim.reset()
        self._actions_buffer.fill(0)
        self.sim.step(self._actions_buffer)
        obs = self.sim.get_observations()
        return obs.reshape(self.num_envs, self.observation_size).astype(np.float32)

    def step(
        self, actions: NDArray[np.int32] | list[int]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        """Step all environments with the given actions.

        Args:
            actions: Action array of shape (num_envs,). Each action should
                be an integer representing the discrete action to take.

        Returns:
            Tuple of (observations, rewards, dones, infos):
                - observations: Shape (num_envs, observation_size)
                - rewards: Shape (num_envs,)
                - dones: Shape (num_envs,), True if episode ended
                - infos: List of dicts, one per environment
        """
        actions_arr = np.asarray(actions, dtype=np.int32)
        if actions_arr.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {actions_arr.shape[0]}")

        self.sim.step(actions_arr)

        obs = (
            self.sim.get_observations()
            .reshape(self.num_envs, self.observation_size)
            .astype(np.float32)
        )
        rewards = np.asarray(self.sim.get_rewards(), dtype=np.float32)
        dones = np.asarray(self.sim.get_dones(), dtype=np.bool_)
        infos: list[dict[str, Any]] = [{"stage": self.start_stage} for _ in range(self.num_envs)]

        return obs, rewards, dones, infos

    def close(self) -> None:
        """Clean up simulator resources."""
        if hasattr(self, "sim") and self.sim is not None:
            del self.sim

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.close()

    @property
    def action_space_size(self) -> int:
        """Return the number of discrete actions available."""
        if hasattr(self.sim, "get_action_space_size"):
            return self.sim.get_action_space_size()
        return 17

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Return the shape of a single observation."""
        return (self.observation_size,)


class SB3VecFreeTheEndEnv:
    """SB3-compatible vectorized environment for "Free The End" speedrun curriculum.

    Implements the full VecEnv interface expected by Stable Baselines 3 with
    curriculum learning stages from basic survival to defeating the Ender Dragon.

    This environment:
    - Supports VecNormalize wrapper for observation/reward normalization
    - Tracks curriculum stage progression and stats
    - Provides terminal_observation for proper value estimation at episode end
    - Implements async step interface (step_async/step_wait)

    Args:
        num_envs: Number of parallel environments. Defaults to 64.
        start_stage: Initial curriculum stage (1-6). Defaults to 1 (BASIC_SURVIVAL).
        shader_dir: Path to shader directory. If None, uses default.
        auto_advance: Whether to automatically advance curriculum stages. Defaults to False.
        max_ticks_per_episode: Maximum ticks before truncation. Defaults to 36000 (30 min).

    Example:
        >>> from minecraft_sim import SB3VecFreeTheEndEnv
        >>> from stable_baselines3 import PPO
        >>> from stable_baselines3.common.vec_env import VecNormalize
        >>>
        >>> env = SB3VecFreeTheEndEnv(num_envs=64, start_stage=1)
        >>> env = VecNormalize(env, norm_obs=True, norm_reward=True)
        >>> model = PPO("MlpPolicy", env)
        >>> model.learn(total_timesteps=10_000_000)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int = 64,
        start_stage: int = 1,
        shader_dir: str | Path | None = None,
        auto_advance: bool = False,
        max_ticks_per_episode: int = 36000,
    ) -> None:
        if mc189_core is None:
            raise ImportError(
                "mc189_core module not found. "
                "Build the C++ extension: cd cpp/build && cmake .. && make"
            )
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium is required for SB3VecFreeTheEndEnv")

        self.num_envs = num_envs
        self._obs_size = 48
        self._num_actions = 17
        self._max_ticks = max_ticks_per_episode
        self._auto_advance = auto_advance

        # Gymnasium spaces (required by SB3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._num_actions)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

        # Initialize simulator
        config = mc189_core.SimulatorConfig()
        config.num_envs = num_envs
        if shader_dir is not None:
            config.shader_dir = str(shader_dir)
        else:
            default_shader_dir = Path(__file__).parent.parent.parent / "shaders"
            if default_shader_dir.exists():
                config.shader_dir = str(default_shader_dir)

        self.sim = mc189_core.MC189Simulator(config)
        self._pending_actions: NDArray[np.int32] | None = None

        # Per-environment episode tracking
        self._episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self._episode_ticks = np.zeros(num_envs, dtype=np.int32)

        # Curriculum tracking
        self._current_stage = max(1, min(6, start_stage))
        self._stage_episodes = np.zeros(6, dtype=np.int32)
        self._stage_successes = np.zeros(6, dtype=np.int32)
        self._stage_total_reward = np.zeros(6, dtype=np.float64)
        self._total_episodes = 0

        # Render mode (required for some SB3 wrappers)
        self.render_mode: str | None = None

    def reset(self) -> NDArray[np.float32]:
        """Reset all environments and return initial observations.

        Returns:
            Observations array of shape (num_envs, obs_size).
        """
        self.sim.reset()
        # Execute no-op step to get initial observation
        self.sim.step(np.zeros(self.num_envs, dtype=np.int32))

        self._episode_rewards.fill(0)
        self._episode_lengths.fill(0)
        self._episode_ticks.fill(0)

        return self._get_obs()

    def step_async(self, actions: NDArray[np.int32] | list[int]) -> None:
        """Initiate an asynchronous step (required by SB3 VecEnv).

        Args:
            actions: Actions to execute, shape (num_envs,).
        """
        self._pending_actions = np.asarray(actions, dtype=np.int32).ravel()

    def step_wait(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        """Wait for async step and return results (required by SB3 VecEnv).

        Returns:
            Tuple of (observations, rewards, dones, infos).
        """
        if self._pending_actions is None:
            raise RuntimeError("step_wait called before step_async")
        return self._step_impl(self._pending_actions)

    def step(
        self, actions: NDArray[np.int32] | list[int]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        """Synchronous step through all environments.

        Args:
            actions: Actions to execute, shape (num_envs,).

        Returns:
            Tuple of (observations, rewards, dones, infos).
        """
        return self._step_impl(np.asarray(actions, dtype=np.int32).ravel())

    def _step_impl(
        self, actions: NDArray[np.int32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        """Internal step implementation.

        Args:
            actions: Actions array of shape (num_envs,).

        Returns:
            Tuple of (observations, rewards, dones, infos).
        """
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {actions.shape[0]}")

        self.sim.step(actions)

        obs = self._get_obs()
        rewards = np.asarray(self.sim.get_rewards(), dtype=np.float32)
        dones = np.asarray(self.sim.get_dones(), dtype=np.bool_)

        # Update episode tracking
        self._episode_rewards += rewards
        self._episode_lengths += 1
        self._episode_ticks += 1

        # Check for truncation (max ticks reached)
        truncated = self._episode_ticks >= self._max_ticks
        dones = dones | truncated

        # Build info dicts with episode stats and terminal observations
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            if dones[i]:
                # Determine if this was a success (natural termination, not truncation)
                success = bool(self.sim.get_dones()[i]) and not truncated[i]

                # Episode info (required by SB3 for logging)
                infos[i]["episode"] = {
                    "r": float(self._episode_rewards[i]),
                    "l": int(self._episode_lengths[i]),
                    "t": int(self._episode_ticks[i]),
                }

                # Curriculum stats
                infos[i]["curriculum"] = {
                    "stage": self._current_stage,
                    "success": success,
                    "stage_episodes": int(self._stage_episodes[self._current_stage - 1]),
                    "stage_success_rate": self._get_stage_success_rate(self._current_stage),
                }

                # terminal_observation (required by SB3 for proper value estimation)
                # This is the FINAL observation before reset, not the post-reset obs
                infos[i]["terminal_observation"] = obs[i].copy()

                # TimeLimit.truncated flag (used by some SB3 wrappers)
                infos[i]["TimeLimit.truncated"] = bool(truncated[i])

                # Update curriculum stats
                stage_idx = self._current_stage - 1
                self._stage_episodes[stage_idx] += 1
                if success:
                    self._stage_successes[stage_idx] += 1
                self._stage_total_reward[stage_idx] += self._episode_rewards[i]
                self._total_episodes += 1

                # Reset episode tracking for this env
                self._episode_rewards[i] = 0
                self._episode_lengths[i] = 0
                self._episode_ticks[i] = 0

        # Auto-advance curriculum if enabled and threshold met
        if self._auto_advance:
            self._maybe_advance_stage()

        return obs, rewards, dones, infos

    def _get_obs(self) -> NDArray[np.float32]:
        """Get normalized observations from simulator."""
        obs = self.sim.get_observations().reshape(self.num_envs, self._obs_size)
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def _get_stage_success_rate(self, stage: int) -> float:
        """Get success rate for a curriculum stage."""
        idx = stage - 1
        if self._stage_episodes[idx] == 0:
            return 0.0
        return float(self._stage_successes[idx]) / float(self._stage_episodes[idx])

    def _maybe_advance_stage(self) -> None:
        """Check if curriculum should advance to next stage."""
        if self._current_stage >= 6:
            return

        # Require at least 100 episodes and 70% success rate
        idx = self._current_stage - 1
        if (
            self._stage_episodes[idx] >= 100
            and self._get_stage_success_rate(self._current_stage) >= 0.7
        ):
            self._current_stage += 1

    def close(self) -> None:
        """Release simulator resources."""
        if hasattr(self, "sim") and self.sim is not None:
            del self.sim

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.close()

    # ==========================================================================
    # SB3 VecEnv interface methods
    # ==========================================================================

    def env_is_wrapped(self, wrapper_class: type, indices: list[int] | None = None) -> list[bool]:
        """Check if environments are wrapped with a specific wrapper.

        Args:
            wrapper_class: Wrapper class to check for.
            indices: Environment indices to check. If None, checks all.

        Returns:
            List of booleans indicating if each env is wrapped.
        """
        n = len(indices) if indices is not None else self.num_envs
        return [False] * n

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: list[int] | None = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        """Call a method on sub-environments.

        Args:
            method_name: Name of method to call.
            indices: Indices of envs to call on. If None, calls on all.
            *method_args: Positional arguments for the method.
            **method_kwargs: Keyword arguments for the method.

        Returns:
            List of return values from each environment.
        """
        target_envs = indices if indices is not None else list(range(self.num_envs))
        results = []
        for _ in target_envs:
            method = getattr(self, method_name, None)
            if method is not None and callable(method):
                results.append(method(*method_args, **method_kwargs))
            else:
                results.append(None)
        return results

    def get_attr(self, attr_name: str, indices: list[int] | None = None) -> list[Any]:
        """Get an attribute from sub-environments.

        Args:
            attr_name: Name of attribute to get.
            indices: Indices of envs to get from. If None, gets from all.

        Returns:
            List of attribute values.
        """
        target_envs = indices if indices is not None else list(range(self.num_envs))
        attr_val = getattr(self, attr_name, None)
        return [attr_val] * len(target_envs)

    def set_attr(self, attr_name: str, value: Any, indices: list[int] | None = None) -> None:
        """Set an attribute on sub-environments.

        Args:
            attr_name: Name of attribute to set.
            value: Value to set.
            indices: Indices of envs to set on. If None, sets on all.
        """
        # Vectorized env doesn't have independent sub-envs, so this is a no-op
        pass

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Set random seed for all environments.

        Args:
            seed: Seed value. If None, uses random seed.

        Returns:
            List of seeds (one per environment).
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed] * self.num_envs

    def getattr_depth_check(self, name: str, already_found: bool) -> str | None:
        """Get attribute with depth check for VecEnv wrappers.

        Args:
            name: Attribute name.
            already_found: Whether attribute was already found in wrapper chain.

        Returns:
            Attribute value if found at this level, None otherwise.
        """
        if hasattr(self, name):
            return name
        return None

    @property
    def unwrapped(self) -> SB3VecFreeTheEndEnv:
        """Return the unwrapped environment (self for base env)."""
        return self

    # ==========================================================================
    # Curriculum management methods
    # ==========================================================================

    def get_curriculum_stats(self) -> dict[str, Any]:
        """Get comprehensive curriculum training statistics.

        Returns:
            Dictionary with stage-by-stage statistics.
        """
        stats = {
            "current_stage": self._current_stage,
            "total_episodes": self._total_episodes,
            "auto_advance": self._auto_advance,
            "stages": {},
        }

        stage_names = [
            "basic_survival",
            "resource_gathering",
            "nether_navigation",
            "enderman_hunting",
            "stronghold_finding",
            "end_fight",
        ]

        for i, name in enumerate(stage_names):
            stage_num = i + 1
            episodes = int(self._stage_episodes[i])
            successes = int(self._stage_successes[i])
            total_reward = float(self._stage_total_reward[i])

            stats["stages"][name] = {
                "stage_number": stage_num,
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / episodes if episodes > 0 else 0.0,
                "total_reward": total_reward,
                "avg_reward": total_reward / episodes if episodes > 0 else 0.0,
                "mastered": episodes >= 100
                and (successes / episodes if episodes > 0 else 0.0) >= 0.7,
            }

        return stats

    def set_stage(self, stage: int) -> None:
        """Manually set the current curriculum stage.

        Args:
            stage: Stage number (1-6).

        Raises:
            ValueError: If stage is out of range.
        """
        if not 1 <= stage <= 6:
            raise ValueError(f"Stage must be 1-6, got {stage}")
        self._current_stage = stage

    def get_stage(self) -> int:
        """Get the current curriculum stage.

        Returns:
            Current stage number (1-6).
        """
        return self._current_stage

    def reset_curriculum_stats(self) -> None:
        """Reset all curriculum statistics to zero."""
        self._stage_episodes.fill(0)
        self._stage_successes.fill(0)
        self._stage_total_reward.fill(0)
        self._total_episodes = 0
