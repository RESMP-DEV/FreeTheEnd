"""High-performance vectorized Free The End environment for parallel training.

All N environments run in parallel on GPU via the mc189_core Vulkan backend.
Supports mixed stages (different envs at different curriculum stages).

Features:
- GPU-accelerated batch stepping
- Per-environment curriculum tracking
- Automatic stage advancement
- SB3/CleanRL/RLlib compatible interface
- Comprehensive episode and curriculum statistics

Example:
    >>> env = VecFreeTheEndEnv(num_envs=1024)
    >>> obs = env.reset()
    >>> for _ in range(1000):
    ...     actions = np.random.randint(0, 32, size=1024)
    ...     obs, rewards, dones, infos = env.step(actions)
    >>> print(env.get_stage_distribution())
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._compat import HAS_GYMNASIUM, spaces
from .curriculum import StageID

# Import mc189_core (C++ Vulkan extension)
mc189_core: Any = None
try:
    import mc189_core as _mc189_core

    mc189_core = _mc189_core
except ImportError:
    pass

if mc189_core is None:
    try:
        import importlib.util

        so_path = Path(__file__).parent / "mc189_core.cpython-312-darwin.so"
        if so_path.exists():
            spec = importlib.util.spec_from_file_location("mc189_core", so_path)
            if spec and spec.loader:
                mc189_core = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mc189_core)
    except Exception:
        pass


class VecFreeTheEndEnv:
    """Vectorized Free The End environment for parallel training.

    All N environments run in parallel on GPU via the mc189_core Vulkan backend.
    Supports mixed stages (different envs at different curriculum stages).

    Args:
        num_envs: Number of parallel environments. Defaults to 64.
        start_stage: Initial curriculum stage for all environments.
        curriculum: Enable automatic curriculum advancement. Defaults to True.
        seeds: Optional list of seeds, one per environment.
        max_episode_steps: Maximum steps before truncation. Defaults to 36000.
        observation_size: Size of observation vector. Defaults to 256.
        num_actions: Number of discrete actions. Defaults to 32.
        shader_dir: Path to shader directory. If None, uses default.
        success_threshold: Success rate required to advance stage. Defaults to 0.7.
        min_episodes_for_advance: Minimum episodes before stage advance. Defaults to 100.
        reward_shaping: Enable stage-specific reward shaping. Defaults to True.

    Attributes:
        num_envs: Number of parallel environments.
        stages: Array of current stage IDs for each environment.
        observation_space: Gymnasium Box observation space.
        action_space: Gymnasium Discrete action space.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int = 64,
        start_stage: StageID = StageID.BASIC_SURVIVAL,
        curriculum: bool = True,
        seeds: list[int] | NDArray[np.int64] | None = None,
        max_episode_steps: int = 36000,
        observation_size: int = 256,
        num_actions: int = 32,
        shader_dir: str | Path | None = None,
        success_threshold: float = 0.7,
        min_episodes_for_advance: int = 100,
        reward_shaping: bool = True,
    ) -> None:
        if mc189_core is None:
            raise ImportError(
                "mc189_core module not found. "
                "Build the C++ extension: cd cpp/build && cmake .. && make"
            )

        self.num_envs = num_envs
        self.curriculum = curriculum
        self.max_steps = max_episode_steps
        self._obs_size = observation_size
        self._num_actions = num_actions
        self._success_threshold = success_threshold
        self._min_episodes_for_advance = min_episodes_for_advance
        self._reward_shaping = reward_shaping

        # Per-environment state
        self.stages = np.full(num_envs, start_stage.value, dtype=np.int32)
        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)

        # Per-environment curriculum tracking
        self._stage_episodes = np.zeros(num_envs, dtype=np.int32)
        self._stage_successes = np.zeros(num_envs, dtype=np.int32)

        # Global curriculum statistics per stage
        self._global_stage_episodes: dict[int, int] = {s.value: 0 for s in StageID}
        self._global_stage_successes: dict[int, int] = {s.value: 0 for s in StageID}
        self._global_stage_rewards: dict[int, float] = {s.value: 0.0 for s in StageID}
        self._total_episodes = 0
        self._total_steps = 0

        # Seeds
        if seeds is None:
            seeds = np.random.randint(0, 2**31, size=num_envs, dtype=np.int64)
        self.seeds = np.asarray(seeds, dtype=np.int64)

        # Gymnasium spaces
        if HAS_GYMNASIUM and spaces is not None:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(observation_size,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(num_actions)
            self.single_observation_space = self.observation_space
            self.single_action_space = self.action_space

        # Initialize GPU backend
        self._init_backend(shader_dir)

        # Async step support
        self._pending_actions: NDArray[np.int32] | None = None

        # Render mode (for SB3 compatibility)
        self.render_mode: str | None = None

    def _init_backend(self, shader_dir: str | Path | None) -> None:
        """Initialize the mc189_core Vulkan backend.

        Args:
            shader_dir: Path to shader directory.
        """
        config = mc189_core.SimulatorConfig()
        config.num_envs = self.num_envs

        if shader_dir is not None:
            config.shader_dir = str(shader_dir)
        else:
            default_shader_dir = Path(__file__).parent.parent.parent / "shaders"
            if default_shader_dir.exists():
                config.shader_dir = str(default_shader_dir)

        self._backend = mc189_core.MC189Simulator(config)
        self._actions_buffer = np.zeros(self.num_envs, dtype=np.int32)

    def reset(
        self,
        *,
        seed: int | Sequence[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> NDArray[np.float32]:
        """Reset all environments and return initial observations.

        Args:
            seed: Optional seed for reproducibility.
            options: Optional reset options (unused).

        Returns:
            Observations array of shape (num_envs, observation_size).
        """
        del options  # unused

        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
                self.seeds = np.random.randint(0, 2**31, size=self.num_envs, dtype=np.int64)
            else:
                self.seeds = np.asarray(seed, dtype=np.int64)

        # Reset backend
        self._backend.reset()

        # Execute no-op step to get initial observations
        self._actions_buffer.fill(0)
        self._backend.step(self._actions_buffer)

        # Reset episode tracking
        self.steps[:] = 0
        self.episode_rewards[:] = 0
        self.episode_lengths[:] = 0

        return self._get_observations()

    def step(
        self, actions: NDArray[np.int32] | Sequence[int]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        """Step all environments with the given actions.

        Args:
            actions: Action array of shape (num_envs,).

        Returns:
            Tuple of (observations, rewards, dones, infos):
                - observations: Shape (num_envs, observation_size)
                - rewards: Shape (num_envs,)
                - dones: Shape (num_envs,), True if episode ended
                - infos: List of dicts with episode info
        """
        actions_arr = np.asarray(actions, dtype=np.int32).ravel()
        if actions_arr.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {actions_arr.shape[0]}")

        # Execute batch step on GPU
        self._backend.step(actions_arr)

        # Get results from backend
        obs = self._get_observations()
        rewards = np.asarray(self._backend.get_rewards(), dtype=np.float32)
        dones = np.asarray(self._backend.get_dones(), dtype=np.bool_)

        # Apply stage-specific reward shaping
        if self._reward_shaping:
            rewards = self._apply_reward_shaping(rewards, obs, dones)

        # Update episode tracking
        self.steps += 1
        self.episode_rewards += rewards
        self.episode_lengths += 1
        self._total_steps += self.num_envs

        # Check truncation (max steps reached)
        truncated = self.steps >= self.max_steps
        dones = dones | truncated

        # Build info dicts and handle episode completions
        infos = self._build_infos(obs, rewards, dones, truncated)

        return obs, rewards, dones, infos

    def step_async(self, actions: NDArray[np.int32] | Sequence[int]) -> None:
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
            Step results from the pending actions.
        """
        if self._pending_actions is None:
            raise RuntimeError("step_wait called before step_async")
        result = self.step(self._pending_actions)
        self._pending_actions = None
        return result

    def _get_observations(self) -> NDArray[np.float32]:
        """Get observations from backend and reshape.

        Returns:
            Observations array of shape (num_envs, observation_size).
        """
        obs = self._backend.get_observations()
        obs = np.asarray(obs, dtype=np.float32)

        # Handle different backend output shapes
        if obs.ndim == 1:
            # Flat array from backend - check if it's already per-env sized
            total_obs = self.num_envs * self._obs_size
            if obs.shape[0] >= total_obs:
                obs = obs[:total_obs].reshape(self.num_envs, self._obs_size)
            else:
                # Backend returns smaller observations - pad or adjust
                actual_obs_size = obs.shape[0] // self.num_envs
                obs = obs.reshape(self.num_envs, actual_obs_size)
                # Pad to expected size
                if actual_obs_size < self._obs_size:
                    padding = np.zeros(
                        (self.num_envs, self._obs_size - actual_obs_size), dtype=np.float32
                    )
                    obs = np.concatenate([obs, padding], axis=1)
        elif obs.shape[1] != self._obs_size:
            # Adjust observation size
            if obs.shape[1] < self._obs_size:
                padding = np.zeros((self.num_envs, self._obs_size - obs.shape[1]), dtype=np.float32)
                obs = np.concatenate([obs, padding], axis=1)
            else:
                obs = obs[:, : self._obs_size]

        return obs

    def _apply_reward_shaping(
        self,
        rewards: NDArray[np.float32],
        obs: NDArray[np.float32],
        dones: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        """Apply stage-specific reward shaping.

        Different curriculum stages emphasize different rewards:
        - BASIC_SURVIVAL: Survival bonuses, penalty for damage
        - RESOURCE_GATHERING: Bonuses for item collection
        - NETHER_NAVIGATION: Bonuses for progress toward fortress
        - ENDERMAN_HUNTING: Bonuses for ender pearl collection
        - STRONGHOLD_FINDING: Bonuses for eye of ender use
        - END_FIGHT: Dragon damage bonuses, victory bonus

        Args:
            rewards: Raw rewards from backend.
            obs: Current observations.
            dones: Episode termination flags.

        Returns:
            Shaped rewards.
        """
        shaped = rewards.copy()

        # Process by stage for efficiency
        for stage_id in np.unique(self.stages):
            mask = self.stages == stage_id
            stage = StageID(stage_id)

            if stage == StageID.BASIC_SURVIVAL:
                # Small survival bonus for staying alive
                alive = mask & ~dones
                shaped[alive] += 0.001

            elif stage == StageID.RESOURCE_GATHERING:
                # Slightly higher base reward for early game progress
                shaped[mask] *= 1.1

            elif stage == StageID.NETHER_NAVIGATION:
                # Nether is dangerous - survival matters more
                alive = mask & ~dones
                shaped[alive] += 0.002

            elif stage == StageID.ENDERMAN_HUNTING:
                # Combat focused - higher variance rewards
                shaped[mask] *= 1.15

            elif stage == StageID.STRONGHOLD_FINDING:
                # Exploration focused
                shaped[mask] *= 1.1

            elif stage == StageID.END_FIGHT:
                # Dragon fight - high stakes, significant reward scaling
                shaped[mask] *= 1.5
                # Extra bonus for dragon damage (assumed in observation structure)
                # obs[:, 16] typically contains dragon health in similar envs
                # This is a placeholder - actual implementation depends on backend

        return shaped

    def _build_infos(
        self,
        obs: NDArray[np.float32],
        rewards: NDArray[np.float32],
        dones: NDArray[np.bool_],
        truncated: NDArray[np.bool_],
    ) -> list[dict[str, Any]]:
        """Build info dictionaries for all environments.

        Args:
            obs: Current observations.
            rewards: Step rewards.
            dones: Episode done flags.
            truncated: Truncation flags.

        Returns:
            List of info dicts, one per environment.
        """
        infos: list[dict[str, Any]] = []

        for i in range(self.num_envs):
            info: dict[str, Any] = {"stage": int(self.stages[i])}

            if dones[i]:
                # Determine success based on reward and natural termination
                # Success = positive reward and not truncated
                success = float(self.episode_rewards[i]) > 0 and not truncated[i]

                # Episode finished - record stats
                info["episode"] = {
                    "r": float(self.episode_rewards[i]),
                    "l": int(self.episode_lengths[i]),
                    "stage": int(self.stages[i]),
                    "stage_name": StageID(self.stages[i]).name,
                    "success": success,
                }

                # Terminal observation for algorithms that need it
                info["terminal_observation"] = obs[i].copy()

                # TimeLimit.truncated flag (used by some SB3 wrappers)
                info["TimeLimit.truncated"] = bool(truncated[i])

                # Update curriculum stats
                stage_id = self.stages[i]
                self._global_stage_episodes[stage_id] += 1
                self._global_stage_rewards[stage_id] += self.episode_rewards[i]
                if success:
                    self._global_stage_successes[stage_id] += 1
                self._total_episodes += 1

                # Update per-env curriculum tracking
                self._stage_episodes[i] += 1
                if success:
                    self._stage_successes[i] += 1

                # Check for curriculum advancement
                if self.curriculum and self._check_stage_complete(i):
                    if self.stages[i] < StageID.END_FIGHT.value:
                        old_stage = self.stages[i]
                        self.stages[i] += 1
                        info["curriculum_advanced"] = True
                        info["old_stage"] = int(old_stage)
                        info["new_stage"] = int(self.stages[i])
                        info["new_stage_name"] = StageID(self.stages[i]).name

                        # Reset per-env stage tracking for new stage
                        self._stage_episodes[i] = 0
                        self._stage_successes[i] = 0

                # Auto-reset this environment
                self._reset_env(i)
                obs[i] = self._get_observation(i)

            infos.append(info)

        return infos

    def _check_stage_complete(self, env_id: int) -> bool:
        """Check if an environment has mastered its current stage.

        Args:
            env_id: Environment index.

        Returns:
            True if stage is mastered and should advance.
        """
        episodes = self._stage_episodes[env_id]
        if episodes < self._min_episodes_for_advance:
            return False

        success_rate = self._stage_successes[env_id] / episodes
        return success_rate >= self._success_threshold

    def _reset_env(self, env_id: int) -> None:
        """Reset a single environment.

        Args:
            env_id: Environment index.
        """
        # Reset episode tracking for this env
        self.steps[env_id] = 0
        self.episode_rewards[env_id] = 0
        self.episode_lengths[env_id] = 0

        # Backend single-env reset if available
        if hasattr(self._backend, "reset_env"):
            self._backend.reset_env(env_id, int(self.seeds[env_id]))

    def _get_observation(self, env_id: int) -> NDArray[np.float32]:
        """Get observation for a single environment.

        Args:
            env_id: Environment index.

        Returns:
            Observation array of shape (observation_size,).
        """
        if hasattr(self._backend, "get_observation"):
            obs = self._backend.get_observation(env_id)
            obs = np.asarray(obs, dtype=np.float32)
            if obs.shape[0] < self._obs_size:
                padding = np.zeros(self._obs_size - obs.shape[0], dtype=np.float32)
                obs = np.concatenate([obs, padding])
            return obs[: self._obs_size]

        # Fallback: get all observations and extract
        all_obs = self._get_observations()
        return all_obs[env_id]

    def set_stages(
        self,
        env_ids: NDArray[np.int32] | Sequence[int] | int,
        stages: NDArray[np.int32] | Sequence[int] | int | StageID,
    ) -> None:
        """Set specific environments to specific stages.

        Args:
            env_ids: Environment indices to modify.
            stages: Stage IDs to set (single value applied to all, or one per env).

        Example:
            >>> env.set_stages([0, 1, 2], StageID.END_FIGHT)
            >>> env.set_stages(np.arange(10), np.full(10, 6))
        """
        if isinstance(env_ids, int):
            env_ids = np.array([env_ids], dtype=np.int32)
        else:
            env_ids = np.asarray(env_ids, dtype=np.int32)

        if isinstance(stages, StageID):
            stage_values = np.full(len(env_ids), stages.value, dtype=np.int32)
        elif isinstance(stages, int):
            stage_values = np.full(len(env_ids), stages, dtype=np.int32)
        else:
            stage_values = np.asarray(stages, dtype=np.int32)

        if len(stage_values) == 1 and len(env_ids) > 1:
            stage_values = np.full(len(env_ids), stage_values[0], dtype=np.int32)

        if len(env_ids) != len(stage_values):
            raise ValueError(
                f"env_ids and stages must have same length, got {len(env_ids)} and {len(stage_values)}"
            )

        # Validate indices
        if np.any(env_ids < 0) or np.any(env_ids >= self.num_envs):
            raise ValueError(f"env_ids must be in range [0, {self.num_envs})")

        # Validate stage values
        min_stage = StageID.BASIC_SURVIVAL.value
        max_stage = StageID.END_FIGHT.value
        if np.any(stage_values < min_stage) or np.any(stage_values > max_stage):
            raise ValueError(f"stages must be in range [{min_stage}, {max_stage}]")

        self.stages[env_ids] = stage_values

        # Reset per-env curriculum tracking for modified envs
        self._stage_episodes[env_ids] = 0
        self._stage_successes[env_ids] = 0

    def get_stage_distribution(self) -> dict[int, int]:
        """Return count of environments at each stage.

        Returns:
            Dictionary mapping stage ID (int) to count.

        Example:
            >>> dist = env.get_stage_distribution()
            >>> print(dist)  # {1: 20, 2: 25, 3: 10, 4: 5, 5: 3, 6: 1}
        """
        unique, counts = np.unique(self.stages, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def get_stage_distribution_named(self) -> dict[str, int]:
        """Return count of environments at each stage with names.

        Returns:
            Dictionary mapping stage name to count.

        Example:
            >>> dist = env.get_stage_distribution_named()
            >>> print(dist)  # {'BASIC_SURVIVAL': 20, 'RESOURCE_GATHERING': 25, ...}
        """
        distribution = self.get_stage_distribution()
        return {StageID(k).name: v for k, v in distribution.items()}

    def get_curriculum_stats(self) -> dict[str, Any]:
        """Get comprehensive curriculum training statistics.

        Returns:
            Dictionary with curriculum progress metrics.
        """
        stage_stats = {}
        for stage_id in StageID:
            sid = stage_id.value
            episodes = self._global_stage_episodes[sid]
            successes = self._global_stage_successes[sid]
            total_reward = self._global_stage_rewards[sid]

            stage_stats[stage_id.name] = {
                "stage_id": sid,
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / episodes if episodes > 0 else 0.0,
                "total_reward": total_reward,
                "avg_reward": total_reward / episodes if episodes > 0 else 0.0,
                "envs_at_stage": int(np.sum(self.stages == sid)),
            }

        return {
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "stage_distribution": self.get_stage_distribution_named(),
            "stages": stage_stats,
            "curriculum_enabled": self.curriculum,
            "success_threshold": self._success_threshold,
            "min_episodes_for_advance": self._min_episodes_for_advance,
        }

    def close(self) -> None:
        """Clean up backend resources."""
        if hasattr(self, "_backend") and self._backend is not None:
            del self._backend
            self._backend = None

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
        # Vectorized env shares state, so we just set on self
        setattr(self, attr_name, value)

    def seed(self, seed: int | list[int] | None = None) -> list[int | None]:
        """Set random seed for all environments.

        Args:
            seed: Seed value(s). If None, uses random seeds.

        Returns:
            List of seeds (one per environment).
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
                self.seeds = np.random.randint(0, 2**31, size=self.num_envs, dtype=np.int64)
            else:
                self.seeds = np.asarray(seed, dtype=np.int64)
        return self.seeds.tolist()

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
    def unwrapped(self) -> VecFreeTheEndEnv:
        """Return the unwrapped environment (self for base env)."""
        return self


def make_vec_free_the_end_env(
    num_envs: int = 64,
    start_stage: StageID | str | int = StageID.BASIC_SURVIVAL,
    curriculum: bool = True,
    **kwargs: Any,
) -> VecFreeTheEndEnv:
    """Factory function for VecFreeTheEndEnv.

    Args:
        num_envs: Number of parallel environments.
        start_stage: Starting curriculum stage.
        curriculum: Enable automatic curriculum advancement.
        **kwargs: Additional arguments passed to VecFreeTheEndEnv.

    Returns:
        Configured VecFreeTheEndEnv instance.

    Example:
        >>> env = make_vec_free_the_end_env(1024, start_stage="END_FIGHT")
        >>> env = make_vec_free_the_end_env(256, curriculum=False)
    """
    if isinstance(start_stage, str):
        start_stage = StageID[start_stage]
    elif isinstance(start_stage, int):
        start_stage = StageID(start_stage)

    return VecFreeTheEndEnv(
        num_envs=num_envs,
        start_stage=start_stage,
        curriculum=curriculum,
        **kwargs,
    )
