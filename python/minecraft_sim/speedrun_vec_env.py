"""Vectorized environment for parallel Free The End speedrun training.

This module provides a vectorized environment wrapper that supports:
- N parallel environments running at different curriculum stages
- Automatic curriculum advancement per environment
- Efficient GPU batching across stages
- SB3/CleanRL/RLlib compatible interface
- Comprehensive episode and curriculum statistics

Example:
    >>> env = SpeedrunVecEnv(num_envs=64)
    >>> obs = env.reset()
    >>> for _ in range(10000):
    ...     actions = np.random.randint(0, 17, size=64)
    ...     obs, rewards, dones, infos = env.step(actions)
    >>> print(env.get_curriculum_stats())
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._compat import HAS_GYMNASIUM, spaces
from .curriculum import CurriculumManager, StageID, StageProgress, get_shader_set_for_stage
from .observations import decode_flat_observation
from .progress_watchdog import ProgressWatchdog, StallAlertConfig
from .progression import ProgressTracker

# Import mc189_core (C++ extension)
mc189_core = None
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


@dataclass
class EnvEpisodeStats:
    """Per-environment episode statistics."""

    episode_reward: float = 0.0
    episode_length: int = 0
    stage_id: int = 1  # StageID.BASIC_SURVIVAL
    stage_successes: int = 0
    stage_episodes: int = 0


@dataclass
class CurriculumStats:
    """Aggregate curriculum statistics."""

    total_episodes: int = 0
    total_steps: int = 0
    stage_completions: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    stage_episode_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    stage_success_rates: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    envs_per_stage: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    mean_episode_reward: float = 0.0
    mean_episode_length: float = 0.0


class SpeedrunVecEnv:
    """Vectorized environment for parallel Free The End speedrun training.

    Supports N parallel environments, each potentially at different curriculum stages.
    Environments automatically advance through the curriculum when they achieve
    the success threshold for their current stage.

    The environment efficiently batches GPU operations by grouping environments
    at the same stage together for simulation, then scatters results back to
    the correct environment indices.

    Args:
        num_envs: Number of parallel environments. Defaults to 64.
        shader_dir: Path to shader directory. If None, uses default path.
        observation_size: Size of observation vector. Defaults to 48.
        initial_stage: Starting stage for all environments. Defaults to BASIC_SURVIVAL.
        curriculum_manager: Optional pre-configured CurriculumManager.
        auto_curriculum: If True, automatically advance curriculum. Defaults to True.
        success_threshold: Success rate required to advance stage. Defaults to 0.7.
        min_episodes_for_advance: Minimum episodes before stage advance. Defaults to 100.

    Attributes:
        num_envs: Number of parallel environments.
        observation_space: Gymnasium Box observation space.
        action_space: Gymnasium Discrete action space.
        single_observation_space: Single environment observation space.
        single_action_space: Single environment action space.

    Example:
        >>> env = SpeedrunVecEnv(num_envs=64)
        >>> obs = env.reset()
        >>> print(f"Stage distribution: {env.get_stage_distribution()}")
        >>> actions = np.random.randint(0, 17, size=64)
        >>> obs, rewards, dones, infos = env.step(actions)
    """

    def __init__(
        self,
        num_envs: int = 64,
        shader_dir: str | Path | None = None,
        observation_size: int = 48,
        initial_stage: StageID = StageID.BASIC_SURVIVAL,
        curriculum_manager: CurriculumManager | None = None,
        auto_curriculum: bool = True,
        success_threshold: float = 0.7,
        min_episodes_for_advance: int = 100,
        progress_watchdog: ProgressWatchdog | StallAlertConfig | None = None,
    ) -> None:
        if mc189_core is None:
            raise ImportError(
                "mc189_core module not found. "
                "Build the C++ extension: cd cpp/build && cmake .. && make"
            )
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium required for SpeedrunVecEnv")

        self.num_envs = num_envs
        self._obs_size = observation_size
        self._num_actions = 17  # Dragon fight action space

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._num_actions)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

        # Initialize simulator with stage-specific shader set
        config = mc189_core.SimulatorConfig()
        config.num_envs = num_envs
        if shader_dir is not None:
            config.shader_dir = str(shader_dir)
        else:
            default_shader_dir = Path(__file__).parent.parent.parent / "shaders"
            if default_shader_dir.exists():
                config.shader_dir = str(default_shader_dir)

        # Load only the shaders needed for the initial stage (if backend supports it)
        if hasattr(config, "shader_set"):
            config.shader_set = get_shader_set_for_stage(initial_stage)

        self._shader_dir = config.shader_dir
        self._active_shader_stage: int = initial_stage.value
        self.sim = mc189_core.MC189Simulator(config)
        self._actions_buffer = np.zeros(num_envs, dtype=np.int32)

        # Curriculum management
        self.curriculum_manager = curriculum_manager or CurriculumManager()
        self.auto_curriculum = auto_curriculum
        self.success_threshold = success_threshold
        self.min_episodes_for_advance = min_episodes_for_advance

        # Per-environment state
        self._env_stages = np.full(num_envs, initial_stage.value, dtype=np.int32)
        self._env_stats = [EnvEpisodeStats(stage_id=initial_stage.value) for _ in range(num_envs)]

        # Episode tracking
        self._episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(num_envs, dtype=np.int32)

        # Per-stage progress tracking
        self._stage_progress: dict[int, StageProgress] = {
            stage.value: StageProgress(stage_id=stage) for stage in StageID
        }

        # Curriculum statistics
        self._curriculum_stats = CurriculumStats()
        self._total_steps = 0

        # Per-environment progress trackers
        self._progress_trackers = [ProgressTracker() for _ in range(num_envs)]

        # Progress watchdog for obsidian stall detection
        if isinstance(progress_watchdog, StallAlertConfig):
            self._progress_watchdog: ProgressWatchdog | None = ProgressWatchdog(progress_watchdog)
        elif isinstance(progress_watchdog, ProgressWatchdog):
            self._progress_watchdog = progress_watchdog
        else:
            self._progress_watchdog = None

        # Per-environment persistent inventory state across stage transitions.
        # Each entry maps item names (from SpeedrunEnv._PERSISTENT_INVENTORY_KEYS)
        # to normalized float values.  Populated on curriculum advancement so that
        # key items carry over to the next stage.
        self._env_inventory_states: list[dict[str, float]] = [{} for _ in range(num_envs)]

        # Async step support
        self._pending_actions: NDArray[np.int32] | None = None

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

        self.sim.reset()
        self._actions_buffer.fill(0)
        self.sim.step(self._actions_buffer)

        # Reset episode tracking
        self._episode_rewards.fill(0)
        self._episode_lengths.fill(0)

        # Reset per-env stats but preserve stage assignments
        for i, stats in enumerate(self._env_stats):
            stats.episode_reward = 0.0
            stats.episode_length = 0
            # Don't reset stage_id, stage_successes, stage_episodes

        return self._get_obs()

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
                - infos: List of dicts with episode stats
        """
        actions_arr = np.asarray(actions, dtype=np.int32).ravel()
        if actions_arr.shape[0] != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {actions_arr.shape[0]}")

        # Step simulator
        self.sim.step(actions_arr)
        self._total_steps += self.num_envs

        # Get results
        obs = self._get_obs()
        rewards = np.asarray(self.sim.get_rewards(), dtype=np.float32)
        dones = np.asarray(self.sim.get_dones(), dtype=np.bool_)

        # Apply stage-specific reward shaping
        rewards = self._apply_reward_shaping(rewards, obs, dones)

        # Track episode stats
        self._episode_rewards += rewards
        self._episode_lengths += 1

        # Build info dicts and handle episode completions
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                # Determine success based on reward
                success = self._episode_rewards[i] > 0

                # Record episode in curriculum stats
                stage_id = self._env_stages[i]
                self._record_episode(i, success, self._episode_rewards[i], self._episode_lengths[i])

                infos[i]["episode"] = {
                    "r": float(self._episode_rewards[i]),
                    "l": int(self._episode_lengths[i]),
                    "stage_id": int(stage_id),
                    "stage_name": StageID(stage_id).name,
                    "success": success,
                }
                infos[i]["terminal_observation"] = obs[i].copy()

                # Check for curriculum advancement
                if self.auto_curriculum:
                    advanced = self._try_advance_curriculum(i)
                    if advanced:
                        infos[i]["curriculum_advanced"] = True
                        infos[i]["new_stage_id"] = int(self._env_stages[i])
                        infos[i]["new_stage_name"] = StageID(self._env_stages[i]).name

                # Reset episode tracking for this env
                self._episode_rewards[i] = 0
                self._episode_lengths[i] = 0

            # Add stage info to all infos
            infos[i]["stage_id"] = int(self._env_stages[i])

        # Update progress trackers from observations
        for i in range(self.num_envs):
            obs_dict = self._decode_obs_to_dict(obs[i], int(self._env_stages[i]))
            self._progress_trackers[i].update_from_observation(obs_dict)
            infos[i]["progress_snapshot"] = self._progress_trackers[i].to_snapshot()

            # Feed completed episodes to progress watchdog
            if dones[i] and self._progress_watchdog is not None:
                snapshot = infos[i]["progress_snapshot"]
                alert = self._progress_watchdog.observe(
                    env_id=i,
                    progress_snapshot=snapshot,
                    stage_id=int(self._env_stages[i]),
                )
                if alert is not None:
                    infos[i]["obsidian_stall_alert"] = {
                        "episodes_since_growth": alert.episodes_since_growth,
                        "current_obsidian": alert.current_obsidian,
                        "wall_time_sec": alert.wall_time_sec,
                    }

            # Reset tracker on episode end, then restore any persisted inventory
            if dones[i]:
                self._progress_trackers[i].reset()
                if self._env_inventory_states[i]:
                    self._restore_env_inventory(i)

        return obs, rewards, dones, infos

    def step_async(self, actions: NDArray[np.int32]) -> None:
        """Async step (required by SB3 VecEnv interface).

        Args:
            actions: Action array to execute.
        """
        self._pending_actions = np.asarray(actions, dtype=np.int32)

    def step_wait(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        """Wait for async step result (required by SB3 VecEnv interface).

        Returns:
            Step results from the pending actions.
        """
        if self._pending_actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        result = self.step(self._pending_actions)
        self._pending_actions = None
        return result

    def _get_obs(self) -> NDArray[np.float32]:
        """Get observations from simulator.

        Returns:
            Observations array clipped to [0, 1].
        """
        obs = self.sim.get_observations().reshape(self.num_envs, self._obs_size)
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def _decode_obs_to_dict(self, obs: NDArray[np.float32], stage_id: int) -> dict[str, Any]:
        """Decode an observation vector into the dict format expected by ProgressTracker.

        Maps the compact simulator observation indices to semantic fields:
            [0:3] position, [3:6] velocity, [6:8] yaw/pitch,
            [8] health, [16] dragon_health, [17:20] dragon_pos,
            [24] dragon_phase, [25] dragon_distance, [28] can_hit,
            [32] crystals_remaining.

        Args:
            obs: Single environment observation of shape (obs_size,).

        Returns:
            Dictionary suitable for ProgressTracker.update_from_observation.
        """
        n = len(obs)
        if n == 256:
            return decode_flat_observation(stage_id, obs)

        health = float(obs[8]) * 20.0 if n > 8 else 20.0

        # Dimension heuristic: if dragon_health (idx 16) is active and > 0, assume End
        dragon_health_norm = float(obs[16]) if n > 16 else 0.0
        dragon_health = dragon_health_norm * 200.0

        # Infer dimension from dragon activity
        dimension = 0
        if n > 16 and dragon_health_norm > 0.0:
            dimension = 2  # End dimension

        crystals_remaining = int(obs[32] * 10.0) if n > 32 else 10

        dragon_phase = int(obs[24] * 6.0) if n > 24 else 0

        result: dict[str, Any] = {
            "player": {
                "health": health,
                "dimension": dimension,
            },
            "inventory": {},
            "dragon": {
                "is_active": dragon_health_norm > 0.0,
                "dragon_health": dragon_health,
                "crystals_remaining": crystals_remaining,
                "phase": dragon_phase,
            },
        }
        return result

    def get_progress_snapshots(self) -> list[dict[str, Any]]:
        """Return the latest progress snapshot for each environment.

        Useful for external monitoring of per-environment speedrun progression.

        Returns:
            List of snapshot dicts, one per environment, as produced by
            ProgressTracker.to_snapshot().
        """
        return [tracker.to_snapshot() for tracker in self._progress_trackers]

    def _apply_reward_shaping(
        self,
        rewards: NDArray[np.float32],
        obs: NDArray[np.float32],
        dones: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        """Apply stage-specific reward shaping.

        Different curriculum stages emphasize different rewards:
        - BASIC_SURVIVAL: Survival bonuses
        - END_FIGHT: Dragon damage bonuses

        Args:
            rewards: Raw rewards from simulator.
            obs: Current observations.
            dones: Episode termination flags.

        Returns:
            Shaped rewards.
        """
        shaped_rewards = rewards.copy()

        # Group envs by stage for efficient processing
        for stage_id in np.unique(self._env_stages):
            mask = self._env_stages == stage_id
            stage = StageID(stage_id)

            if stage == StageID.END_FIGHT:
                # Extra bonus for dragon damage in final stage
                # dragon_health_idx = 16 (from observation layout, for future use)
                # Reward is already computed by simulator, just apply multiplier
                shaped_rewards[mask] *= 1.2

            elif stage == StageID.BASIC_SURVIVAL:
                # Survival bonus for staying alive
                alive_mask = mask & ~dones
                shaped_rewards[alive_mask] += 0.01

        return shaped_rewards

    def _record_episode(self, env_id: int, success: bool, reward: float, length: int) -> None:
        """Record episode results for curriculum tracking.

        Args:
            env_id: Environment index.
            success: Whether episode was successful.
            reward: Total episode reward.
            length: Episode length in steps.
        """
        stage_id = self._env_stages[env_id]
        stats = self._env_stats[env_id]

        # Update per-env stats
        stats.stage_episodes += 1
        if success:
            stats.stage_successes += 1

        # Update stage progress
        progress = self._stage_progress[stage_id]
        progress.episodes_completed += 1
        if success:
            progress.successes += 1
        progress.total_reward += reward
        progress.best_reward = max(progress.best_reward, reward)
        alpha = 0.1
        progress.average_ticks = alpha * length + (1 - alpha) * progress.average_ticks

        # Update global stats
        self._curriculum_stats.total_episodes += 1
        self._curriculum_stats.stage_episode_counts[stage_id] += 1
        if success:
            self._curriculum_stats.stage_completions[stage_id] += 1

    def _serialize_env_inventory(self, env_id: int) -> dict[str, float]:
        """Serialize inventory-related progress for an environment.

        Captures key item counts from the ProgressTracker so they persist
        into the next curriculum stage.

        Args:
            env_id: Environment index.

        Returns:
            Dictionary mapping item names to normalized values.
        """
        p = self._progress_trackers[env_id].progress
        state: dict[str, float] = {}
        if p.iron_ingots > 0:
            state["iron_ingot"] = min(p.iron_ingots / 64.0, 1.0)
        if p.diamonds > 0:
            state["diamond"] = min(p.diamonds / 64.0, 1.0)
        if p.blaze_rods > 0:
            state["blaze_rod"] = min(p.blaze_rods / 64.0, 1.0)
        if p.ender_pearls > 0:
            state["ender_pearl"] = min(p.ender_pearls / 16.0, 1.0)
        if p.eyes_crafted > 0:
            state["eye_of_ender"] = min(p.eyes_crafted / 12.0, 1.0)
        if p.obsidian_collected > 0:
            state["obsidian"] = min(p.obsidian_collected / 64.0, 1.0)
        if p.has_iron_pickaxe:
            state["has_iron_pickaxe"] = 1.0
        if p.has_bucket:
            state["has_bucket"] = 1.0
        if p.has_iron_sword:
            state["has_sword"] = 1.0
            state["sword_material"] = 0.75  # iron
        return state

    def _restore_env_inventory(self, env_id: int) -> None:
        """Restore persisted inventory into a ProgressTracker after stage advance.

        Seeds the new stage's ProgressTracker with items carried from the
        prior stage so the agent starts with accumulated resources.

        Args:
            env_id: Environment index.
        """
        state = self._env_inventory_states[env_id]
        if not state:
            return
        p = self._progress_trackers[env_id].progress
        if "iron_ingot" in state:
            p.iron_ingots = max(p.iron_ingots, int(state["iron_ingot"] * 64))
        if "diamond" in state:
            p.diamonds = max(p.diamonds, int(state["diamond"] * 64))
        if "blaze_rod" in state:
            p.blaze_rods = max(p.blaze_rods, int(state["blaze_rod"] * 64))
        if "ender_pearl" in state:
            p.ender_pearls = max(p.ender_pearls, int(state["ender_pearl"] * 16))
        if "eye_of_ender" in state:
            p.eyes_crafted = max(p.eyes_crafted, int(state["eye_of_ender"] * 12))
        if "obsidian" in state:
            p.obsidian_collected = max(p.obsidian_collected, int(state["obsidian"] * 64))
        if "has_iron_pickaxe" in state:
            p.has_iron_pickaxe = True
        if "has_bucket" in state:
            p.has_bucket = True
        if "has_sword" in state:
            p.has_iron_sword = True

    def _try_advance_curriculum(self, env_id: int) -> bool:
        """Try to advance an environment to the next curriculum stage.

        Serializes inventory before advancing so items persist.

        Args:
            env_id: Environment index.

        Returns:
            True if environment advanced to a new stage.
        """
        stats = self._env_stats[env_id]
        current_stage = self._env_stages[env_id]

        # Check minimum episodes requirement
        if stats.stage_episodes < self.min_episodes_for_advance:
            return False

        # Calculate success rate
        success_rate = (
            stats.stage_successes / stats.stage_episodes if stats.stage_episodes > 0 else 0
        )

        # Check if threshold met
        if success_rate < self.success_threshold:
            return False

        # Try to advance to next stage
        try:
            next_stage = StageID(current_stage + 1)
        except ValueError:
            # Already at final stage
            return False

        # Persist inventory before transitioning
        self._env_inventory_states[env_id] = self._serialize_env_inventory(env_id)

        # Advance the environment
        self._env_stages[env_id] = next_stage.value
        stats.stage_id = next_stage.value
        stats.stage_successes = 0
        stats.stage_episodes = 0

        # Rebuild simulator if dominant stage changed (all envs at same stage)
        self._maybe_rebuild_for_dominant_stage()

        # Restore inventory into the fresh tracker
        self._restore_env_inventory(env_id)

        return True

    def set_stage(self, env_ids: int | Sequence[int], stage_id: StageID | int) -> None:
        """Set specific environments to a curriculum stage.

        Serializes inventory for affected environments before switching so
        that items persist into the new stage.

        Args:
            env_ids: Environment index or array of indices.
            stage_id: Target stage ID.

        Example:
            >>> env.set_stage([0, 1, 2], StageID.END_FIGHT)
            >>> env.set_stage(0, 6)  # Also accepts int
        """
        if isinstance(stage_id, StageID):
            stage_value = stage_id.value
        else:
            stage_value = int(stage_id)

        if isinstance(env_ids, int):
            env_ids = [env_ids]

        env_ids_arr = np.asarray(env_ids, dtype=np.int32)

        # Validate indices
        if np.any(env_ids_arr < 0) or np.any(env_ids_arr >= self.num_envs):
            raise ValueError(f"env_ids must be in range [0, {self.num_envs})")

        # Serialize inventory before switching stages
        for i in env_ids_arr:
            self._env_inventory_states[i] = self._serialize_env_inventory(i)

        # Set stages
        self._env_stages[env_ids_arr] = stage_value

        # Reset per-env stats for these environments
        for i in env_ids_arr:
            self._env_stats[i].stage_id = stage_value
            self._env_stats[i].stage_successes = 0
            self._env_stats[i].stage_episodes = 0

        # Rebuild simulator if dominant stage changed
        self._maybe_rebuild_for_dominant_stage()

    def _maybe_rebuild_for_dominant_stage(self) -> None:
        """Rebuild the GPU simulator if all environments share the same stage.

        When all environments converge to a single stage, the simulator can be
        rebuilt with stage-specific shader configuration for optimal performance.
        Currently a no-op; GPU backend switching will be implemented later.
        """
        unique_stages = np.unique(self._env_stages)
        if len(unique_stages) == 1:
            # All environments at same stage - future: rebuild with optimized shaders
            pass

    def get_stage_distribution(self) -> dict[str, int]:
        """Get distribution of environments across stages.

        Returns:
            Dictionary mapping stage name to count of environments.

        Example:
            >>> print(env.get_stage_distribution())
            {'BASIC_SURVIVAL': 32, 'RESOURCE_GATHERING': 20, 'END_FIGHT': 12}
        """
        distribution: dict[str, int] = {}
        for stage_id in np.unique(self._env_stages):
            count = int(np.sum(self._env_stages == stage_id))
            stage_name = StageID(stage_id).name
            distribution[stage_name] = count
        return distribution

    def advance_curriculum(self, env_id: int) -> bool:
        """Manually advance an environment to the next stage.

        Args:
            env_id: Environment index.

        Returns:
            True if advanced, False if already at final stage.
        """
        current_stage = self._env_stages[env_id]
        try:
            next_stage = StageID(current_stage + 1)
        except ValueError:
            return False

        self.set_stage(env_id, next_stage)
        return True

    def get_curriculum_stats(self) -> dict[str, Any]:
        """Get comprehensive curriculum statistics.

        Returns:
            Dictionary with curriculum progress metrics.

        Example:
            >>> stats = env.get_curriculum_stats()
            >>> print(f"Total episodes: {stats['total_episodes']}")
            >>> print(f"Stage success rates: {stats['stage_success_rates']}")
        """
        # Update stage distribution
        for stage_id in StageID:
            count = int(np.sum(self._env_stages == stage_id.value))
            self._curriculum_stats.envs_per_stage[stage_id.value] = count

        # Calculate success rates
        for stage_id, progress in self._stage_progress.items():
            if progress.episodes_completed > 0:
                rate = progress.successes / progress.episodes_completed
                self._curriculum_stats.stage_success_rates[stage_id] = rate

        # Calculate mean episode metrics
        total_eps = self._curriculum_stats.total_episodes
        if total_eps > 0:
            total_reward = sum(p.total_reward for p in self._stage_progress.values())
            total_length = sum(
                p.average_ticks * p.episodes_completed for p in self._stage_progress.values()
            )
            self._curriculum_stats.mean_episode_reward = total_reward / total_eps
            self._curriculum_stats.mean_episode_length = total_length / total_eps

        self._curriculum_stats.total_steps = self._total_steps

        return {
            "total_episodes": self._curriculum_stats.total_episodes,
            "total_steps": self._curriculum_stats.total_steps,
            "stage_distribution": {
                StageID(k).name: v
                for k, v in self._curriculum_stats.envs_per_stage.items()
                if v > 0
            },
            "stage_episode_counts": {
                StageID(k).name: v
                for k, v in self._curriculum_stats.stage_episode_counts.items()
                if v > 0
            },
            "stage_completions": {
                StageID(k).name: v
                for k, v in self._curriculum_stats.stage_completions.items()
                if v > 0
            },
            "stage_success_rates": {
                StageID(k).name: f"{v:.2%}"
                for k, v in self._curriculum_stats.stage_success_rates.items()
                if v > 0
            },
            "mean_episode_reward": self._curriculum_stats.mean_episode_reward,
            "mean_episode_length": self._curriculum_stats.mean_episode_length,
            "stage_progress": {
                StageID(k).name: {
                    "episodes": p.episodes_completed,
                    "successes": p.successes,
                    "success_rate": p.success_rate,
                    "average_reward": p.average_reward,
                    "best_reward": p.best_reward if p.best_reward != float("-inf") else None,
                    "mastered": p.mastered,
                }
                for k, p in self._stage_progress.items()
                if p.episodes_completed > 0
            },
        }

    def get_env_stages(self) -> NDArray[np.int32]:
        """Get current stage for each environment.

        Returns:
            Array of shape (num_envs,) with stage IDs.
        """
        return self._env_stages.copy()

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "sim") and self.sim is not None:
            del self.sim

    # SB3 VecEnv interface methods

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
    def unwrapped(self) -> SpeedrunVecEnv:
        """Return unwrapped environment."""
        return self

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.close()


def make_speedrun_vec_env(
    num_envs: int = 64,
    initial_stage: StageID | str = StageID.BASIC_SURVIVAL,
    auto_curriculum: bool = True,
    **kwargs: Any,
) -> SpeedrunVecEnv:
    """Factory function for SpeedrunVecEnv.

    Args:
        num_envs: Number of parallel environments.
        initial_stage: Starting curriculum stage (StageID or name string).
        auto_curriculum: Enable automatic curriculum advancement.
        **kwargs: Additional arguments passed to SpeedrunVecEnv.

    Returns:
        Configured SpeedrunVecEnv instance.

    Example:
        >>> env = make_speedrun_vec_env(64, initial_stage="END_FIGHT")
        >>> env = make_speedrun_vec_env(128, auto_curriculum=False)
    """
    if isinstance(initial_stage, str):
        initial_stage = StageID[initial_stage]

    return SpeedrunVecEnv(
        num_envs=num_envs,
        initial_stage=initial_stage,
        auto_curriculum=auto_curriculum,
        **kwargs,
    )
