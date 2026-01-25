"""Per-environment curriculum management for vectorized RL training.

This module provides curriculum advancement logic that operates at the
per-environment level, allowing each parallel environment in a vectorized
setup to progress through stages independently based on its own performance.

Unlike the global CurriculumManager in curriculum.py, this VecCurriculumManager
tracks success rates and handles advancement for each env_id separately,
enabling more efficient curriculum learning in highly parallelized training.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class StageOverride:
    """Per-stage overrides for curriculum advancement parameters.

    Attributes:
        min_episodes_to_advance: Override minimum episodes before advancement.
        advancement_threshold: Override success rate required to advance.
        min_metric_value: Minimum value of a tracked metric (e.g. blaze rod
            count) that must be maintained above this threshold for advancement.
            The metric is reported via update(..., metric_value=...).
        min_dimension_episodes: Minimum number of episodes where the environment
            was flagged as being in a specific dimension (e.g. in-nether) before
            advancement is allowed. Reported via update(..., in_target_dimension=True).
        sustained_windows: Number of consecutive evaluation windows where the
            success rate must remain at or above threshold before advancement is
            granted. Each window is sustained_window_size episodes. If the rate
            drops below threshold during the sustained period, the counter resets.
            Set to None (default) for single-check advancement.
        sustained_window_size: Number of episodes per evaluation window for
            sustained success checking. Defaults to min_episodes_to_advance if
            not set. Only used when sustained_windows is set.
    """

    min_episodes_to_advance: int | None = None
    advancement_threshold: float | None = None
    min_metric_value: float | None = None
    min_dimension_episodes: int | None = None
    sustained_windows: int | None = None
    sustained_window_size: int | None = None


@dataclass
class StageStats:
    """Per-stage statistics aggregated across all environments.

    Attributes:
        total_episodes: Total episodes completed at this stage.
        total_successes: Total successful episodes.
        total_reward: Cumulative reward earned at this stage.
        best_reward: Best single-episode reward seen.
        avg_episode_length: Running average episode length in ticks.
    """

    total_episodes: int = 0
    total_successes: int = 0
    total_reward: float = 0.0
    best_reward: float = float("-inf")
    avg_episode_length: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate for this stage."""
        if self.total_episodes == 0:
            return 0.0
        return self.total_successes / self.total_episodes

    @property
    def avg_reward(self) -> float:
        """Calculate average reward per episode."""
        if self.total_episodes == 0:
            return 0.0
        return self.total_reward / self.total_episodes


@dataclass
class AdvancementEvent:
    """Records a curriculum advancement event.

    Attributes:
        env_id: Environment that advanced.
        old_stage: Previous stage ID.
        new_stage: New stage ID.
        timestamp: Episode count when advancement occurred.
        success_rate: Success rate that triggered advancement.
    """

    env_id: int
    old_stage: int
    new_stage: int
    timestamp: int
    success_rate: float


class VecCurriculumManager:
    """Per-environment curriculum manager for vectorized training.

    Tracks curriculum progress independently for each environment in a
    vectorized setup, enabling automatic stage advancement based on
    per-environment success rates.

    Args:
        num_envs: Number of parallel environments.
        min_stage: Minimum stage ID (default 1).
        max_stage: Maximum stage ID (default 6).
        advancement_threshold: Success rate required to advance (default 0.7).
        regression_threshold: Success rate below which to regress (default 0.2).
        window_size: Number of episodes to track for success rate (default 100).
        min_episodes_to_advance: Minimum episodes before advancement allowed.
        min_episodes_to_regress: Minimum episodes before regression allowed.
        enable_regression: Whether to allow stage regression on poor performance.
        stage_overrides: Per-stage overrides for min_episodes_to_advance and
            advancement_threshold. Use StageOverride dataclass for each entry.

    Attributes:
        num_envs: Number of environments being tracked.
        env_stages: Current stage for each environment.
        stage_success_rate: Deque of recent success/failure per stage.
        stage_stats: Aggregated statistics per stage.
        advancement_history: List of advancement events.

    Example:
        >>> manager = VecCurriculumManager(num_envs=64)
        >>> # After episode ends for env 5 with success
        >>> manager.update(env_id=5, success=True, stage=1)
        >>> # Check if any envs advanced
        >>> for event in manager.get_recent_advancements():
        ...     print(f"Env {event.env_id}: {event.old_stage} -> {event.new_stage}")
    """

    def __init__(
        self,
        num_envs: int,
        min_stage: int = 1,
        max_stage: int = 6,
        advancement_threshold: float = 0.7,
        regression_threshold: float = 0.2,
        window_size: int = 100,
        min_episodes_to_advance: int = 20,
        min_episodes_to_regress: int = 50,
        enable_regression: bool = False,
        stage_overrides: dict[int, StageOverride] | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.min_stage = min_stage
        self.max_stage = max_stage
        self.advancement_threshold = advancement_threshold
        self.regression_threshold = regression_threshold
        self.window_size = window_size
        self.min_episodes_to_advance = min_episodes_to_advance
        self.min_episodes_to_regress = min_episodes_to_regress
        self.enable_regression = enable_regression
        self.stage_overrides = stage_overrides or {}

        # Per-environment stage tracking
        self.env_stages = np.ones(num_envs, dtype=np.int32) * min_stage

        # Per-environment episode counts
        self._env_episode_counts = np.zeros(num_envs, dtype=np.int32)
        self._env_episodes_at_stage = np.zeros(num_envs, dtype=np.int32)

        # Per-stage success rate tracking (shared across all envs)
        self.stage_success_rate: dict[int, deque[float]] = {
            s: deque(maxlen=window_size) for s in range(min_stage, max_stage + 1)
        }

        # Per-stage aggregated statistics
        self.stage_stats: dict[int, StageStats] = {
            s: StageStats() for s in range(min_stage, max_stage + 1)
        }

        # Advancement event history
        self.advancement_history: list[AdvancementEvent] = []
        self._total_episodes = 0

        # Per-environment recent success tracking for advancement decisions
        self._env_recent_success: dict[int, deque[float]] = {
            i: deque(maxlen=window_size) for i in range(num_envs)
        }

        # Per-environment metric tracking (e.g. blaze rod production rate)
        self._env_metric_values: NDArray[np.float32] = np.zeros(num_envs, dtype=np.float32)

        # Per-environment dimension episode counts (e.g. episodes spent in nether)
        self._env_dimension_episodes: NDArray[np.int32] = np.zeros(num_envs, dtype=np.int32)

        # Per-environment sustained window pass counters: how many consecutive
        # evaluation windows the env has maintained success rate above threshold.
        # Only relevant for stages with sustained_windows override set.
        self._env_sustained_passes: NDArray[np.int32] = np.zeros(num_envs, dtype=np.int32)
        # Episode counter within the current sustained evaluation window
        self._env_sustained_window_episodes: NDArray[np.int32] = np.zeros(num_envs, dtype=np.int32)
        # Per-env success accumulator for the current sustained window
        self._env_sustained_window_successes: NDArray[np.int32] = np.zeros(num_envs, dtype=np.int32)

    def update(
        self,
        env_id: int,
        success: bool,
        stage: int,
        reward: float = 0.0,
        episode_length: int = 0,
        metric_value: float | None = None,
        in_target_dimension: bool = False,
    ) -> bool:
        """Update curriculum state after an episode ends.

        Args:
            env_id: Environment ID that completed the episode.
            success: Whether the episode was successful.
            stage: Stage the episode was run at.
            reward: Total episode reward (optional, for stats).
            episode_length: Episode length in ticks (optional, for stats).
            metric_value: Optional metric value (e.g. blaze rod count) to track
                for stages that require maintaining production above a threshold.
            in_target_dimension: If True, this episode counts toward the
                dimension episode requirement (e.g. episode was in the Nether).

        Returns:
            True if the environment advanced to a new stage.
        """
        self._total_episodes += 1
        self._env_episode_counts[env_id] += 1
        self._env_episodes_at_stage[env_id] += 1

        # Update per-stage success rate
        success_val = 1.0 if success else 0.0
        self.stage_success_rate[stage].append(success_val)

        # Update per-env success tracking
        self._env_recent_success[env_id].append(success_val)

        # Update metric and dimension tracking
        if metric_value is not None:
            self._env_metric_values[env_id] = metric_value
        if in_target_dimension:
            self._env_dimension_episodes[env_id] += 1

        # Update aggregated stats
        stats = self.stage_stats[stage]
        stats.total_episodes += 1
        if success:
            stats.total_successes += 1
        stats.total_reward += reward
        stats.best_reward = max(stats.best_reward, reward)
        if episode_length > 0:
            alpha = 0.1
            stats.avg_episode_length = (
                alpha * episode_length + (1 - alpha) * stats.avg_episode_length
            )

        # Check for advancement
        advanced = self._check_advancement(env_id, stage)
        if advanced:
            return True

        # Check for regression
        if self.enable_regression:
            self._check_regression(env_id, stage)

        return False

    def update_batch(
        self,
        env_ids: NDArray[np.int32],
        successes: NDArray[np.bool_],
        stages: NDArray[np.int32] | None = None,
        rewards: NDArray[np.float32] | None = None,
        episode_lengths: NDArray[np.int32] | None = None,
        metric_values: NDArray[np.float32] | None = None,
        in_target_dimensions: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.bool_]:
        """Batch update for multiple environments.

        Args:
            env_ids: Array of environment IDs.
            successes: Array of success flags.
            stages: Array of stages (if None, uses env_stages).
            rewards: Array of rewards (optional).
            episode_lengths: Array of episode lengths (optional).
            metric_values: Array of metric values per env (optional).
            in_target_dimensions: Array of dimension flags per env (optional).

        Returns:
            Boolean array indicating which envs advanced.
        """
        n = len(env_ids)
        if stages is None:
            stages = self.env_stages[env_ids]
        if rewards is None:
            rewards = np.zeros(n, dtype=np.float32)
        if episode_lengths is None:
            episode_lengths = np.zeros(n, dtype=np.int32)

        advanced = np.zeros(n, dtype=np.bool_)
        for i, (env_id, success, stage, reward, length) in enumerate(
            zip(env_ids, successes, stages, rewards, episode_lengths, strict=True)
        ):
            mv = float(metric_values[i]) if metric_values is not None else None
            itd = bool(in_target_dimensions[i]) if in_target_dimensions is not None else False
            advanced[i] = self.update(
                int(env_id), bool(success), int(stage), float(reward), int(length),
                metric_value=mv, in_target_dimension=itd,
            )

        return advanced

    def _get_stage_min_episodes(self, stage: int) -> int:
        """Get minimum episodes to advance for a given stage.

        Args:
            stage: Stage ID.

        Returns:
            Minimum episodes required, using per-stage override if set.
        """
        override = self.stage_overrides.get(stage)
        if override is not None and override.min_episodes_to_advance is not None:
            return override.min_episodes_to_advance
        return self.min_episodes_to_advance

    def _get_stage_threshold(self, stage: int) -> float:
        """Get advancement threshold for a given stage.

        Args:
            stage: Stage ID.

        Returns:
            Advancement threshold, using per-stage override if set.
        """
        override = self.stage_overrides.get(stage)
        if override is not None and override.advancement_threshold is not None:
            return override.advancement_threshold
        return self.advancement_threshold

    def _check_advancement(self, env_id: int, current_stage: int) -> bool:
        """Check if environment should advance to next stage.

        Checks success rate threshold, and optionally enforces per-stage
        metric value requirements, dimension episode minimums, and sustained
        success windows defined in stage_overrides. When sustained_windows is
        configured, the environment must maintain its success rate above
        threshold across multiple consecutive evaluation windows before
        advancement is granted.

        Args:
            env_id: Environment ID.
            current_stage: Current stage of the environment.

        Returns:
            True if advancement occurred.
        """
        if current_stage >= self.max_stage:
            return False

        min_episodes = self._get_stage_min_episodes(current_stage)

        if self._env_episodes_at_stage[env_id] < min_episodes:
            return False

        # Use per-env recent success rate for advancement decision
        env_history = self._env_recent_success[env_id]
        if len(env_history) < min_episodes:
            return False

        # Calculate recent success rate for this env
        recent_rate = np.mean(list(env_history)[-min_episodes:])

        threshold = self._get_stage_threshold(current_stage)

        # Enforce per-stage metric and dimension episode constraints
        override = self.stage_overrides.get(current_stage)
        if override is not None:
            if (
                override.min_metric_value is not None
                and self._env_metric_values[env_id] < override.min_metric_value
            ):
                return False
            if (
                override.min_dimension_episodes is not None
                and self._env_dimension_episodes[env_id] < override.min_dimension_episodes
            ):
                return False

        if recent_rate < threshold:
            # Rate dropped below threshold; reset sustained progress
            if override is not None and override.sustained_windows is not None:
                self._env_sustained_passes[env_id] = 0
                self._env_sustained_window_episodes[env_id] = 0
                self._env_sustained_window_successes[env_id] = 0
            return False

        # Rate is above threshold. Check sustained window requirement.
        if override is not None and override.sustained_windows is not None:
            window_size = (
                override.sustained_window_size
                if override.sustained_window_size is not None
                else min_episodes
            )
            # Accumulate into the current sustained window
            self._env_sustained_window_episodes[env_id] += 1
            if env_history[-1] >= 0.5:  # last episode was a success
                self._env_sustained_window_successes[env_id] += 1

            # Check if current window is complete
            if self._env_sustained_window_episodes[env_id] >= window_size:
                window_rate = (
                    self._env_sustained_window_successes[env_id]
                    / self._env_sustained_window_episodes[env_id]
                )
                if window_rate >= threshold:
                    self._env_sustained_passes[env_id] += 1
                else:
                    # Window failed; reset all sustained progress
                    self._env_sustained_passes[env_id] = 0
                # Reset window counters for next window
                self._env_sustained_window_episodes[env_id] = 0
                self._env_sustained_window_successes[env_id] = 0

            # Only advance if enough consecutive windows have passed
            if self._env_sustained_passes[env_id] < override.sustained_windows:
                return False

        # Advancement granted
        old_stage = current_stage
        new_stage = current_stage + 1
        self.env_stages[env_id] = new_stage
        self._env_episodes_at_stage[env_id] = 0
        self._env_recent_success[env_id].clear()
        self._env_metric_values[env_id] = 0.0
        self._env_dimension_episodes[env_id] = 0
        self._env_sustained_passes[env_id] = 0
        self._env_sustained_window_episodes[env_id] = 0
        self._env_sustained_window_successes[env_id] = 0

        self.advancement_history.append(
            AdvancementEvent(
                env_id=env_id,
                old_stage=old_stage,
                new_stage=new_stage,
                timestamp=self._total_episodes,
                success_rate=recent_rate,
            )
        )
        return True

    def _check_regression(self, env_id: int, current_stage: int) -> bool:
        """Check if environment should regress to previous stage.

        Args:
            env_id: Environment ID.
            current_stage: Current stage of the environment.

        Returns:
            True if regression occurred.
        """
        if current_stage <= self.min_stage:
            return False

        if self._env_episodes_at_stage[env_id] < self.min_episodes_to_regress:
            return False

        env_history = self._env_recent_success[env_id]
        if len(env_history) < self.min_episodes_to_regress:
            return False

        recent_rate = np.mean(list(env_history)[-self.min_episodes_to_regress :])

        if recent_rate < self.regression_threshold:
            old_stage = current_stage
            new_stage = current_stage - 1
            self.env_stages[env_id] = new_stage
            self._env_episodes_at_stage[env_id] = 0
            self._env_recent_success[env_id].clear()
            self._env_metric_values[env_id] = 0.0
            self._env_dimension_episodes[env_id] = 0

            self.advancement_history.append(
                AdvancementEvent(
                    env_id=env_id,
                    old_stage=old_stage,
                    new_stage=new_stage,
                    timestamp=self._total_episodes,
                    success_rate=recent_rate,
                )
            )
            return True

        return False

    def get_stage(self, env_id: int) -> int:
        """Get current stage for an environment.

        Args:
            env_id: Environment ID.

        Returns:
            Current stage ID.
        """
        return int(self.env_stages[env_id])

    def get_stages(self) -> NDArray[np.int32]:
        """Get stages for all environments.

        Returns:
            Array of stage IDs, shape (num_envs,).
        """
        return self.env_stages.copy()

    def set_stage(self, env_id: int, stage: int) -> None:
        """Manually set stage for an environment.

        Args:
            env_id: Environment ID.
            stage: Stage to set.
        """
        self.env_stages[env_id] = np.clip(stage, self.min_stage, self.max_stage)
        self._env_episodes_at_stage[env_id] = 0
        self._env_recent_success[env_id].clear()
        self._env_metric_values[env_id] = 0.0
        self._env_dimension_episodes[env_id] = 0

    def set_all_stages(self, stage: int) -> None:
        """Set all environments to the same stage.

        Args:
            stage: Stage to set.
        """
        self.env_stages.fill(np.clip(stage, self.min_stage, self.max_stage))
        self._env_episodes_at_stage.fill(0)
        self._env_metric_values.fill(0.0)
        self._env_dimension_episodes.fill(0)
        for env_id in range(self.num_envs):
            self._env_recent_success[env_id].clear()

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive curriculum statistics.

        Returns:
            Dictionary with stage distribution, success rates, and stats.
        """
        stage_distribution = Counter(self.env_stages.tolist())

        success_rates = {}
        for stage, history in self.stage_success_rate.items():
            if len(history) > 0:
                success_rates[stage] = float(np.mean(history))
            else:
                success_rates[stage] = 0.0

        return {
            "stage_distribution": dict(stage_distribution),
            "success_rates": success_rates,
            "stage_stats": {
                stage: {
                    "episodes": stats.total_episodes,
                    "successes": stats.total_successes,
                    "success_rate": stats.success_rate,
                    "avg_reward": stats.avg_reward,
                    "best_reward": stats.best_reward
                    if stats.best_reward != float("-inf")
                    else None,
                    "avg_episode_length": stats.avg_episode_length,
                }
                for stage, stats in self.stage_stats.items()
            },
            "total_episodes": self._total_episodes,
            "total_advancements": len(self.advancement_history),
            "env_episode_counts": self._env_episode_counts.tolist(),
        }

    def get_recent_advancements(self, n: int = 10) -> list[AdvancementEvent]:
        """Get recent advancement events.

        Args:
            n: Number of recent events to return.

        Returns:
            List of recent AdvancementEvent objects.
        """
        return self.advancement_history[-n:]

    def get_stage_summary(self) -> str:
        """Get a human-readable summary of curriculum state.

        Returns:
            Formatted string with stage distribution and rates.
        """
        dist = Counter(self.env_stages.tolist())
        lines = [f"Curriculum Summary ({self.num_envs} envs, {self._total_episodes} episodes):"]

        for stage in range(self.min_stage, self.max_stage + 1):
            count = dist.get(stage, 0)
            pct = 100 * count / self.num_envs
            rate = self.stage_success_rate[stage]
            rate_str = f"{np.mean(rate):.1%}" if len(rate) > 0 else "N/A"
            lines.append(f"  Stage {stage}: {count:3d} envs ({pct:5.1f}%) | Success: {rate_str}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all curriculum state."""
        self.env_stages.fill(self.min_stage)
        self._env_episode_counts.fill(0)
        self._env_episodes_at_stage.fill(0)
        self._env_metric_values.fill(0.0)
        self._env_dimension_episodes.fill(0)
        self._total_episodes = 0

        for stage in self.stage_success_rate:
            self.stage_success_rate[stage].clear()

        for stage in self.stage_stats:
            self.stage_stats[stage] = StageStats()

        for env_id in range(self.num_envs):
            self._env_recent_success[env_id].clear()

        self.advancement_history.clear()

    def save_state(self) -> dict[str, Any]:
        """Save curriculum state for checkpointing.

        Returns:
            Dictionary containing all state needed to restore.
        """
        return {
            "env_stages": self.env_stages.tolist(),
            "env_episode_counts": self._env_episode_counts.tolist(),
            "env_episodes_at_stage": self._env_episodes_at_stage.tolist(),
            "total_episodes": self._total_episodes,
            "stage_success_rate": {s: list(d) for s, d in self.stage_success_rate.items()},
            "stage_stats": {
                s: {
                    "total_episodes": stats.total_episodes,
                    "total_successes": stats.total_successes,
                    "total_reward": stats.total_reward,
                    "best_reward": stats.best_reward
                    if stats.best_reward != float("-inf")
                    else None,
                    "avg_episode_length": stats.avg_episode_length,
                }
                for s, stats in self.stage_stats.items()
            },
            "advancement_history": [
                {
                    "env_id": e.env_id,
                    "old_stage": e.old_stage,
                    "new_stage": e.new_stage,
                    "timestamp": e.timestamp,
                    "success_rate": e.success_rate,
                }
                for e in self.advancement_history
            ],
            "env_recent_success": {i: list(d) for i, d in self._env_recent_success.items()},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore curriculum state from checkpoint.

        Args:
            state: Dictionary from save_state().
        """
        self.env_stages = np.array(state["env_stages"], dtype=np.int32)
        self._env_episode_counts = np.array(state["env_episode_counts"], dtype=np.int32)
        self._env_episodes_at_stage = np.array(state["env_episodes_at_stage"], dtype=np.int32)
        self._total_episodes = state["total_episodes"]

        for s, values in state["stage_success_rate"].items():
            stage = int(s)
            self.stage_success_rate[stage] = deque(values, maxlen=self.window_size)

        for s, stats_dict in state["stage_stats"].items():
            stage = int(s)
            self.stage_stats[stage] = StageStats(
                total_episodes=stats_dict["total_episodes"],
                total_successes=stats_dict["total_successes"],
                total_reward=stats_dict["total_reward"],
                best_reward=stats_dict["best_reward"]
                if stats_dict["best_reward"] is not None
                else float("-inf"),
                avg_episode_length=stats_dict["avg_episode_length"],
            )

        self.advancement_history = [AdvancementEvent(**e) for e in state["advancement_history"]]

        for i, values in state["env_recent_success"].items():
            env_id = int(i)
            self._env_recent_success[env_id] = deque(values, maxlen=self.window_size)


def create_vec_curriculum_with_stage1_overrides(
    num_envs: int,
    stage1_metadata: dict[str, Any] | None = None,
    stage1_min_episodes: int = 10,
    **kwargs: Any,
) -> VecCurriculumManager:
    """Create a VecCurriculumManager with reduced Stage 1 advancement requirements.

    Stage 1 (BASIC_SURVIVAL) uses a lower minimum episode count and optionally
    derives its advancement threshold from stage metadata. All other stages keep
    the default parameters.

    Args:
        num_envs: Number of parallel environments.
        stage1_metadata: Stage 1 metadata dict. If it contains a
            'curriculum_threshold' key, that value is used as the Stage 1
            advancement threshold override.
        stage1_min_episodes: Minimum episodes for Stage 1 advancement
            (default 10, reduced from the global default of 20).
        **kwargs: Additional keyword arguments passed to VecCurriculumManager.

    Returns:
        Configured VecCurriculumManager with Stage 1 overrides applied.
    """
    threshold_override: float | None = None
    if stage1_metadata is not None:
        threshold_override = stage1_metadata.get("curriculum_threshold")

    stage_overrides = {
        1: StageOverride(
            min_episodes_to_advance=stage1_min_episodes,
            advancement_threshold=threshold_override,
        ),
    }

    return VecCurriculumManager(
        num_envs=num_envs,
        stage_overrides=stage_overrides,
        **kwargs,
    )


def create_vec_curriculum_with_stage_overrides(
    num_envs: int,
    stage1_metadata: dict[str, Any] | None = None,
    stage2_metadata: dict[str, Any] | None = None,
    stage1_min_episodes: int = 10,
    stage2_min_episodes: int = 30,
    **kwargs: Any,
) -> VecCurriculumManager:
    """Create a VecCurriculumManager with Stage 1 and Stage 2 overrides.

    Stage 1 uses a reduced minimum episode count (fast advancement for basic
    survival). Stage 2 uses a higher success threshold (0.75 by default) and
    requires a minimum number of obsidian-related episodes before advancement,
    both drawn from Stage 2 metadata when available.

    For Stage 2, the metadata fields used are:
        - 'advancement_threshold': Success rate required (default 0.75).
        - 'min_obsidian_episodes': Minimum episodes where obsidian was mined
          before advancement is allowed (default 5). Reported to the manager
          via update(..., in_target_dimension=True) when the episode involved
          obsidian mining.

    Args:
        num_envs: Number of parallel environments.
        stage1_metadata: Stage 1 metadata dict (optional). Uses
            'curriculum_threshold' if present.
        stage2_metadata: Stage 2 metadata dict (optional). Uses
            'advancement_threshold' and 'min_obsidian_episodes' if present.
        stage1_min_episodes: Minimum episodes for Stage 1 advancement.
        stage2_min_episodes: Minimum episodes for Stage 2 advancement.
        **kwargs: Additional keyword arguments passed to VecCurriculumManager.

    Returns:
        Configured VecCurriculumManager with Stage 1 and Stage 2 overrides.
    """
    # Stage 1 override
    s1_threshold: float | None = None
    if stage1_metadata is not None:
        s1_threshold = stage1_metadata.get("curriculum_threshold")

    # Stage 2 override: higher threshold + obsidian episode gate
    s2_threshold: float = 0.75
    s2_min_obsidian: int = 5
    if stage2_metadata is not None:
        s2_threshold = stage2_metadata.get("advancement_threshold", 0.75)
        s2_min_obsidian = stage2_metadata.get("min_obsidian_episodes", 5)

    stage_overrides: dict[int, StageOverride] = {
        1: StageOverride(
            min_episodes_to_advance=stage1_min_episodes,
            advancement_threshold=s1_threshold,
        ),
        2: StageOverride(
            min_episodes_to_advance=stage2_min_episodes,
            advancement_threshold=s2_threshold,
            min_dimension_episodes=s2_min_obsidian,
        ),
    }

    # Merge with any user-provided overrides (user takes precedence)
    if "stage_overrides" in kwargs:
        user_overrides: dict[int, StageOverride] = kwargs.pop("stage_overrides")
        stage_overrides.update(user_overrides)

    return VecCurriculumManager(
        num_envs=num_envs,
        stage_overrides=stage_overrides,
        **kwargs,
    )


# Convenience alias matching the provided interface
CurriculumManager = VecCurriculumManager
