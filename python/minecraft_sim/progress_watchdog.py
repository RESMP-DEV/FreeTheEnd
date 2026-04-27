"""Training progress watchdog for detecting stalled Stage 2 obsidian growth.

Monitors progress snapshots emitted by stage environments and raises alerts
when obsidian_count fails to increase over a configurable episode window.
This catches scenarios where the agent loops on sub-optimal behavior (e.g.
mining cobblestone forever) instead of progressing toward the bucket+obsidian
success condition.

Usage:
    from minecraft_sim.progress_watchdog import ProgressWatchdog, StallAlertConfig

    config = StallAlertConfig(stall_window=50, min_obsidian_delta=1)
    watchdog = ProgressWatchdog(config)

    # In training loop:
    obs, reward, terminated, truncated, info = env.step(action)
    watchdog.observe(env_id=0, progress_snapshot=info["progress_snapshot"])

    # Or attach a callback:
    config = StallAlertConfig(
        stall_window=50,
        on_stall=lambda alert: print(f"STALL: env {alert.env_id}")
    )
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StallAlert:
    """Alert emitted when obsidian growth stalls.

    Attributes:
        env_id: Environment that stalled.
        stage_id: Stage where the stall was detected.
        episodes_since_growth: Number of episodes without obsidian increase.
        current_obsidian: Current obsidian count at time of alert.
        last_growth_obsidian: Obsidian count at the last growth event.
        wall_time_sec: Wall-clock seconds since last growth.
        snapshot: The progress snapshot that triggered the alert.
    """

    env_id: int
    stage_id: int
    episodes_since_growth: int
    current_obsidian: int
    last_growth_obsidian: int
    wall_time_sec: float
    snapshot: dict[str, Any]


@dataclass
class StallAlertConfig:
    """Configuration for obsidian growth stall detection.

    Attributes:
        stall_window: Number of consecutive episodes without obsidian growth
            before emitting an alert. Must be >= 1.
        min_obsidian_delta: Minimum obsidian increase required to reset the
            stall counter. Defaults to 1 (any growth resets).
        cooldown_episodes: After emitting an alert, suppress further alerts
            for this many episodes. Prevents alert spam.
        target_stage: Stage ID to monitor (default 2 = Resource Gathering).
        on_stall: Callback invoked with StallAlert when a stall is detected.
            If None, alerts are only logged.
        alert_level: Logging level for stall alerts. Defaults to WARNING.
    """

    stall_window: int = 50
    min_obsidian_delta: int = 1
    cooldown_episodes: int = 20
    target_stage: int = 2
    on_stall: Callable[[StallAlert], None] | None = None
    alert_level: int = logging.WARNING


@dataclass
class _EnvTrackingState:
    """Internal per-environment tracking state."""

    last_obsidian: int = 0
    last_growth_time: float = field(default_factory=time.time)
    episodes_since_growth: int = 0
    total_episodes: int = 0
    cooldown_remaining: int = 0
    alerts_emitted: int = 0
    obsidian_history: deque[int] = field(default_factory=lambda: deque(maxlen=200))


class ProgressWatchdog:
    """Monitors Stage 2 progress snapshots for obsidian growth stalls.

    Tracks per-environment obsidian counts from progress snapshots and emits
    alerts when growth stalls beyond the configured window.

    Args:
        config: Stall detection configuration.

    Example:
        >>> config = StallAlertConfig(stall_window=30)
        >>> watchdog = ProgressWatchdog(config)
        >>> # After each episode:
        >>> watchdog.observe(env_id=0, progress_snapshot={"obsidian_count": 5})
        >>> # Check stats:
        >>> stats = watchdog.get_stats()
    """

    def __init__(self, config: StallAlertConfig | None = None) -> None:
        logger.info("ProgressWatchdog.__init__: config=%s", config)
        self.config = config or StallAlertConfig()
        self._envs: dict[int, _EnvTrackingState] = {}
        self._total_alerts: int = 0
        self._total_observations: int = 0

    def _get_env_state(self, env_id: int) -> _EnvTrackingState:
        """Get or create tracking state for an environment."""
        logger.debug("ProgressWatchdog._get_env_state: env_id=%s", env_id)
        if env_id not in self._envs:
            self._envs[env_id] = _EnvTrackingState()
        return self._envs[env_id]

    def observe(
        self,
        env_id: int,
        progress_snapshot: dict[str, Any],
        stage_id: int | None = None,
    ) -> StallAlert | None:
        """Process a progress snapshot and check for obsidian stalls.

        Should be called once per episode completion (when terminated or
        truncated), passing the progress_snapshot from info dict.

        Args:
            env_id: Environment ID.
            progress_snapshot: The progress_snapshot dict from step info,
                expected to contain "obsidian_count".
            stage_id: Stage ID override. If None, defaults to config.target_stage.
                Pass this if tracking multiple stages.

        Returns:
            A StallAlert if a stall was detected, None otherwise.
        """
        logger.debug("ProgressWatchdog.observe: env_id=%s, progress_snapshot=%s, stage_id=%s", env_id, progress_snapshot, stage_id)
        self._total_observations += 1
        effective_stage = stage_id if stage_id is not None else self.config.target_stage

        # Only monitor the target stage
        if effective_stage != self.config.target_stage:
            return None

        state = self._get_env_state(env_id)
        state.total_episodes += 1

        obsidian = progress_snapshot.get("obsidian_count", 0)
        if isinstance(obsidian, float):
            obsidian = int(obsidian)
        state.obsidian_history.append(obsidian)

        # Check for growth
        delta = obsidian - state.last_obsidian
        if delta >= self.config.min_obsidian_delta:
            # Growth detected, reset stall counter
            state.last_obsidian = obsidian
            state.last_growth_time = time.time()
            state.episodes_since_growth = 0
            state.cooldown_remaining = 0
            return None

        # No sufficient growth
        state.episodes_since_growth += 1

        # Handle cooldown
        if state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1
            return None

        # Check if stall window exceeded
        if state.episodes_since_growth >= self.config.stall_window:
            alert = StallAlert(
                env_id=env_id,
                stage_id=effective_stage,
                episodes_since_growth=state.episodes_since_growth,
                current_obsidian=obsidian,
                last_growth_obsidian=state.last_obsidian,
                wall_time_sec=time.time() - state.last_growth_time,
                snapshot=progress_snapshot,
            )

            state.alerts_emitted += 1
            state.cooldown_remaining = self.config.cooldown_episodes
            self._total_alerts += 1

            logger.log(
                self.config.alert_level,
                "obsidian_stall_detected",
                extra={
                    "env_id": env_id,
                    "stage_id": effective_stage,
                    "episodes_since_growth": state.episodes_since_growth,
                    "current_obsidian": obsidian,
                    "last_growth_obsidian": state.last_obsidian,
                    "wall_time_sec": time.time() - state.last_growth_time,
                },
            )

            if self.config.on_stall:
                self.config.on_stall(alert)

            return alert

        return None

    def observe_batch(
        self,
        env_ids: list[int],
        progress_snapshots: list[dict[str, Any]],
        stage_ids: list[int] | None = None,
    ) -> list[StallAlert]:
        """Process a batch of progress snapshots.

        Convenience method for vectorized environments.

        Args:
            env_ids: List of environment IDs.
            progress_snapshots: Corresponding progress snapshots.
            stage_ids: Optional per-env stage IDs.

        Returns:
            List of StallAlerts (may be empty if no stalls detected).
        """
        logger.debug("ProgressWatchdog.observe_batch: env_ids=%s, progress_snapshots=%s, stage_ids=%s", env_ids, progress_snapshots, stage_ids)
        alerts: list[StallAlert] = []
        for i, (eid, snap) in enumerate(zip(env_ids, progress_snapshots)):
            sid = stage_ids[i] if stage_ids else None
            alert = self.observe(eid, snap, stage_id=sid)
            if alert is not None:
                alerts.append(alert)
        return alerts

    def reset_env(self, env_id: int) -> None:
        """Reset tracking state for an environment.

        Call this when an environment is restarted or reassigned to a
        different stage.

        Args:
            env_id: Environment ID to reset.
        """
        logger.debug("ProgressWatchdog.reset_env: env_id=%s", env_id)
        if env_id in self._envs:
            del self._envs[env_id]

    def get_env_status(self, env_id: int) -> dict[str, Any]:
        """Get current tracking status for an environment.

        Args:
            env_id: Environment ID.

        Returns:
            Dictionary with tracking state, or empty dict if untracked.
        """
        logger.debug("ProgressWatchdog.get_env_status: env_id=%s", env_id)
        if env_id not in self._envs:
            return {}
        state = self._envs[env_id]
        return {
            "env_id": env_id,
            "last_obsidian": state.last_obsidian,
            "episodes_since_growth": state.episodes_since_growth,
            "total_episodes": state.total_episodes,
            "alerts_emitted": state.alerts_emitted,
            "cooldown_remaining": state.cooldown_remaining,
            "wall_time_since_growth": time.time() - state.last_growth_time,
            "obsidian_history": list(state.obsidian_history),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate watchdog statistics.

        Returns:
            Dictionary with overall stall detection stats.
        """
        logger.debug("ProgressWatchdog.get_stats called")
        stalled_envs = [
            eid
            for eid, state in self._envs.items()
            if state.episodes_since_growth >= self.config.stall_window
        ]
        return {
            "total_observations": self._total_observations,
            "total_alerts": self._total_alerts,
            "tracked_envs": len(self._envs),
            "stalled_envs": len(stalled_envs),
            "stalled_env_ids": stalled_envs,
            "config": {
                "stall_window": self.config.stall_window,
                "min_obsidian_delta": self.config.min_obsidian_delta,
                "cooldown_episodes": self.config.cooldown_episodes,
                "target_stage": self.config.target_stage,
            },
        }

    def get_stalled_envs(self) -> list[int]:
        """Get list of environment IDs currently in stall state.

        Returns:
            List of env_ids where obsidian growth has stalled beyond
            the configured window.
        """
        logger.debug("ProgressWatchdog.get_stalled_envs called")
        return [
            eid
            for eid, state in self._envs.items()
            if state.episodes_since_growth >= self.config.stall_window
        ]
