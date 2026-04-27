"""Gymnasium-compatible environments for the Minecraft simulator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("gymnasium is required for minecraft_sim.env") from exc

try:
    from . import _minecraft_sim as _core
except ImportError:
    _core = None


def _require_core() -> None:
    if _core is None:
        raise ImportError("minecraft_sim C++ extension is not available")


def _normalize_seeds(seed: int | Sequence[int], num_envs: int) -> list[int]:
    if isinstance(seed, (list, tuple, np.ndarray)):
        seeds = [int(value) for value in seed]
        if len(seeds) != num_envs:
            raise ValueError("Seed list length must match num_envs")
        return seeds

    seed_sequence = np.random.SeedSequence(int(seed))
    return [int(value) for value in seed_sequence.generate_state(num_envs, dtype=np.uint64)]


class MinecraftEnv(gym.Env):
    """Single Minecraft simulator environment compatible with Gymnasium."""

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None) -> None:
        _require_core()
        self._rng = np.random.default_rng(seed)
        self._sim = _core.MinecraftSimulator(0 if seed is None else int(seed))

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(_core.OBSERVATION_SIZE),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(int(_core.ACTION_SIZE))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        seed_value = int(
            seed
            if seed is not None
            else self._rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
        )
        obs, info = self._sim.reset(seed_value)
        return np.asarray(obs, dtype=np.float32), info

    def step(
        self, action: int | np.ndarray | Sequence[float]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if isinstance(action, (np.ndarray, list, tuple)):
            action_array = np.asarray(action, dtype=np.float32)
            if action_array.ndim == 1 and action_array.size >= 8:
                return self._sim.step(action_array)

        return self._sim.step_discrete(int(action))

    def close(self) -> None:
        return None


class VecMinecraftEnv(gym.vector.VectorEnv):
    """Vectorized Minecraft environment backed by VecMinecraftSimulator."""

    def __init__(self, num_envs: int, base_seed: int | None = None) -> None:
        _require_core()
        self._rng = np.random.default_rng(base_seed)
        self._sim = _core.VecMinecraftSimulator(
            num_envs, 0 if base_seed is None else int(base_seed)
        )
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(_core.OBSERVATION_SIZE),),
            dtype=np.float32,
        )
        action_space = gym.spaces.Discrete(int(_core.ACTION_SIZE))
        super().__init__(num_envs, observation_space, action_space)

    def reset(
        self,
        *,
        seed: int | Sequence[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        del options
        if seed is None:
            self._sim.reset()
        else:
            seeds = _normalize_seeds(seed, self.num_envs)
            self._sim.reset()
            for env_idx, env_seed in enumerate(seeds):
                self._sim.reset_env(env_idx, env_seed)

        observations = np.asarray(self._sim.get_observations(), dtype=np.float32)
        infos = [{} for _ in range(self.num_envs)]
        return observations, infos

    def step(
        self, actions: np.ndarray | Sequence[int] | Sequence[Sequence[float]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions_array = np.asarray(actions)
        if actions_array.ndim == 1:
            if actions_array.shape[0] != self.num_envs:
                raise ValueError("Actions must have shape (num_envs,) or (num_envs, action_dim)")
            actions_array = np.stack(
                [_core.discrete_to_input(int(action), 1.0) for action in actions_array.tolist()],
                axis=0,
            )
        elif actions_array.ndim != 2 or actions_array.shape[0] != self.num_envs:
            raise ValueError("Actions must have shape (num_envs,) or (num_envs, action_dim)")

        obs, rewards, terminated, truncated, info = self._sim.step(
            np.asarray(actions_array, dtype=np.float32)
        )
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            info,
        )

    def sample_actions(self) -> np.ndarray:
        return np.asarray([self.action_space.sample() for _ in range(self.num_envs)])

    def close(self) -> None:
        return None


def make(seed: int | None = None) -> MinecraftEnv:
    """Factory helper for MinecraftEnv."""
    return MinecraftEnv(seed=seed)


def make_vec(num_envs: int, base_seed: int | None = None) -> VecMinecraftEnv:
    """Factory helper for VecMinecraftEnv."""
    return VecMinecraftEnv(num_envs=num_envs, base_seed=base_seed)


class DragonFightEnv(gym.Env):
    """Gymnasium-compatible Ender Dragon fight environment using mc189_core directly.

    This environment wraps MC189Simulator to provide a standard Gymnasium interface
    for the Minecraft Ender Dragon boss fight with:
    - 48-dimensional normalized observation space [0, 1]
    - 17 discrete actions (movement, look, attack, etc.)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render_mode: str | None = None,
        shader_dir: str | None = None,
    ) -> None:
        """Initialize DragonFightEnv.

        Args:
            render_mode: Rendering mode (currently only "human" supported for metadata).
            shader_dir: Path to shader directory. If None, uses default location.
        """
        super().__init__()

        self.render_mode = render_mode

        # Define Gymnasium spaces
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(48,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(17)

        # Resolve shader directory
        if shader_dir is None:
            from pathlib import Path

            shader_dir = str(Path(__file__).resolve().parents[2] / "cpp" / "shaders")

        # Initialize MC189Simulator
        try:
            import mc189_core

            config = mc189_core.SimulatorConfig()
            config.num_envs = 1
            config.shader_dir = shader_dir
            self._sim = mc189_core.MC189Simulator(config)
            self._has_sim = True
        except ImportError:

import logging

logger = logging.getLogger(__name__)

            self._sim = None
            self._has_sim = False

        # Episode tracking
        self._step_count = 0
        self._max_steps = 6000  # 5 minutes at 20 tps

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Optional seed for reproducibility.
            options: Optional reset options (unused).

        Returns:
            Tuple of (observation, info dict).
        """
        super().reset(seed=seed)
        del options  # unused

        self._step_count = 0

        if not self._has_sim:
            # Fallback: return zeros if simulator not available
            return np.zeros(48, dtype=np.float32), {}

        self._sim.reset()
        # Execute a no-op step to get initial observation
        self._sim.step(np.array([0], dtype=np.int32))
        obs = np.array(self._sim.get_observations(), dtype=np.float32).flatten()

        # Clip to [0, 1] range
        obs = np.clip(obs, 0.0, 1.0)

        return obs, {}

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Discrete action index (0 to 16).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        action_int = int(action) if isinstance(action, (int, np.integer)) else int(action.item())
        self._step_count += 1

        if not self._has_sim:
            # Fallback: return zeros if simulator not available
            obs = np.zeros(48, dtype=np.float32)
            return obs, 0.0, False, self._step_count >= self._max_steps, {}

        self._sim.step(np.array([action_int], dtype=np.int32))

        obs = np.array(self._sim.get_observations(), dtype=np.float32).flatten()
        obs = np.clip(obs, 0.0, 1.0)

        reward = float(self._sim.get_rewards()[0])
        terminated = bool(self._sim.get_dones()[0])
        truncated = self._step_count >= self._max_steps and not terminated

        info: dict[str, Any] = {
            "step_count": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Clean up resources."""
        self._sim = None
