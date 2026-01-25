# python/minecraft_sim/backend.py
"""High-level Python interface to mc189 Vulkan compute backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .constants import OBSERVATION_SIZE

try:
    import mc189_core as _mc189

    _HAS_CORE = True
except ImportError:  # pragma: no cover - fallback for environments without bindings
    _mc189 = None
    _HAS_CORE = False


_DEFAULT_SHADER_DIR = Path(__file__).resolve().parents[2] / "cpp" / "shaders"


def _resolve_shader_dir(shader_dir: str | None) -> Path:
    if shader_dir is None:
        return _DEFAULT_SHADER_DIR
    return Path(shader_dir).expanduser().resolve()


class VulkanBackend:
    """High-level interface to the mc189 Vulkan compute backend."""

    def __init__(
        self,
        num_envs: int = 1024,
        enable_validation: bool = False,
        shader_dir: str | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        self._num_envs = int(num_envs)
        self._shader_dir = _resolve_shader_dir(shader_dir)
        self._simulator: Any | None = None
        self._device_name = "Unavailable"
        self._obs_dim = OBSERVATION_SIZE  # Use constant from constants.py

        if not _HAS_CORE:
            self._observations = np.zeros((self._num_envs, self._obs_dim), dtype=np.float32)
            return

        # Create MC189Simulator using the new API
        if hasattr(_mc189, "MC189Simulator") and hasattr(_mc189, "SimulatorConfig"):
            config = _mc189.SimulatorConfig()
            config.num_envs = self._num_envs
            config.enable_validation = enable_validation
            config.shader_dir = str(self._shader_dir)
            self._simulator = _mc189.MC189Simulator(config)
            self._obs_dim = _mc189.MC189Simulator.obs_dim

        # Get device name
        if hasattr(_mc189, "get_device_info"):
            info = _mc189.get_device_info()
            if isinstance(info, dict):
                self._device_name = info.get("device_name", "Unknown")

    def step(
        self,
        actions: NDArray[np.int32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], dict[str, Any]]:
        """Execute one simulation step for all environments.

        Args:
            actions: int32 array of shape (num_envs,) with discrete action indices

        Returns:
            observations: float32 array (num_envs, obs_dim)
            rewards: float32 array (num_envs,)
            dones: bool array (num_envs,)
            infos: dict with additional info
        """
        action_array = np.asarray(actions, dtype=np.int32).ravel()
        if action_array.shape[0] != self._num_envs:
            raise ValueError(
                f"Actions must have shape ({self._num_envs},), got {action_array.shape}"
            )

        if self._simulator is not None:
            self._simulator.step(action_array)
            observations = self._simulator.get_observations()
            rewards = self._simulator.get_rewards()
            dones = self._simulator.get_dones()
            return observations, rewards, dones, {}

        # Fallback without bindings
        observations = np.zeros((self._num_envs, self._obs_dim), dtype=np.float32)
        rewards = np.zeros((self._num_envs,), dtype=np.float32)
        dones = np.zeros((self._num_envs,), dtype=bool)
        return observations, rewards, dones, {}

    def reset(
        self,
        env_ids: NDArray[np.int32] | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Reset specified environments (or all if None).

        Args:
            env_ids: Optional array of environment indices to reset. If None,
                resets all environments.
            seed: Optional seed for deterministic world generation. If None,
                generates a seed from numpy's random state, ensuring that
                `np.random.seed()` or SB3's `set_random_seed()` controls
                the simulation's randomness.

        Returns:
            Observations array of shape (num_envs, obs_dim).

        Note:
            Without an explicit seed, the C++ backend defaults to hardware
            entropy (std::random_device), which bypasses Python's random state.
            This method generates a deterministic seed from numpy when none is
            provided, ensuring reproducibility when training scripts set seeds.
        """
        # Generate deterministic seed from numpy's random state if not provided.
        # This ensures np.random.seed() / SB3's set_random_seed() cascade to C++.
        if seed is None:
            seed = int(np.random.randint(0, 2**63))

        if self._simulator is not None:
            if env_ids is None:
                # Reset all environments with the same base seed
                if hasattr(self._simulator, "reset") and callable(self._simulator.reset):
                    # Try passing seed if the C++ API supports it
                    try:
                        self._simulator.reset(seed=seed)
                    except TypeError:
                        # Fallback: C++ reset() doesn't accept seed kwarg yet
                        self._simulator.reset()
            else:
                env_ids_array = np.asarray(env_ids, dtype=np.int32).ravel()
                for i, env_id in enumerate(env_ids_array):
                    # Derive deterministic sub-seeds for specific envs
                    env_seed = seed + i
                    try:
                        self._simulator.reset(int(env_id), seed=env_seed)
                    except TypeError:
                        # Fallback: C++ reset(env_id) doesn't accept seed yet
                        self._simulator.reset(int(env_id))
            return self._simulator.get_observations()

        return np.zeros((self._num_envs, self._obs_dim), dtype=np.float32)

    @property
    def device_name(self) -> str:
        """Human-readable Vulkan device name."""
        return self._device_name

    @property
    def num_envs(self) -> int:
        """Number of vectorized environments managed by the backend."""
        return self._num_envs

    @property
    def obs_dim(self) -> int:
        """Observation dimensionality reported by the backend."""
        return self._obs_dim
