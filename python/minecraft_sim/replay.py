"""Action replay system for Minecraft speedrun analysis.

Records and replays agent trajectories for:
- Post-hoc analysis of successful/failed runs
- Behavior cloning data collection
- Debugging and visualization
- Deterministic reproduction of specific runs
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import gymnasium as gym


@dataclass
class SpeedrunRecorder:
    """Records agent trajectories during speedrun attempts.

    Captures actions, observations, and rewards for later replay or analysis.
    Uses compressed npz format for efficient storage.

    Example:
        >>> recorder = SpeedrunRecorder()
        >>> recorder.seed = 12345
        >>> env = DragonFightEnv()
        >>> obs, _ = env.reset(seed=recorder.seed)
        >>> for _ in range(1000):
        ...     action = policy(obs)
        ...     obs, reward, done, _, _ = env.step(action)
        ...     recorder.record_step(action, obs, reward)
        ...     if done:
        ...         break
        >>> recorder.save("run_001.npz")
    """

    actions: list[int] = field(default_factory=list)
    observations: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_step(self, action: int | np.ndarray, obs: np.ndarray, reward: float) -> None:
        """Record a single environment step.

        Args:
            action: Action taken (discrete index or array).
            obs: Observation received after the action.
            reward: Reward received for the action.
        """
        action_int = int(action) if isinstance(action, (int, np.integer)) else int(action.item())
        self.actions.append(action_int)
        self.observations.append(obs.copy())
        self.rewards.append(float(reward))

    def record_initial_obs(self, obs: np.ndarray) -> None:
        """Record the initial observation from reset.

        Args:
            obs: Initial observation from env.reset().
        """
        if len(self.observations) == 0 or len(self.observations) == len(self.actions):
            # Insert at beginning or append if this is the first obs
            self.observations.insert(0, obs.copy())

    @property
    def total_reward(self) -> float:
        """Total accumulated reward for this run."""
        return sum(self.rewards)

    @property
    def ticks(self) -> int:
        """Number of ticks (actions) in this recording."""
        return len(self.actions)

    @property
    def duration_seconds(self) -> float:
        """Duration of the run in seconds (at 20 TPS)."""
        return self.ticks / 20.0

    def clear(self) -> None:
        """Clear all recorded data for reuse."""
        self.actions.clear()
        self.observations.clear()
        self.rewards.clear()
        self.metadata.clear()

    def save(self, path: str | Path) -> None:
        """Save recording to compressed npz file.

        Args:
            path: Output file path (should end in .npz).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Stack observations into single array for efficient storage
        obs_array = np.stack(self.observations) if self.observations else np.array([])

        np.savez_compressed(
            path,
            actions=np.array(self.actions, dtype=np.int32),
            observations=obs_array,
            rewards=np.array(self.rewards, dtype=np.float32),
            seed=np.array(self.seed, dtype=np.int64),
            total_reward=np.array(self.total_reward, dtype=np.float32),
            ticks=np.array(len(self.actions), dtype=np.int32),
            # Metadata as structured array
            metadata_keys=list(self.metadata.keys()) if self.metadata else [],
            metadata_values=[str(v) for v in self.metadata.values()] if self.metadata else [],
        )

    @classmethod
    def load(cls, path: str | Path) -> SpeedrunRecorder:
        """Load recording from npz file.

        Args:
            path: Path to the .npz file.

        Returns:
            SpeedrunRecorder with loaded data.
        """
        data = np.load(path, allow_pickle=False)

        recorder = cls()
        recorder.actions = data["actions"].tolist()
        recorder.rewards = data["rewards"].tolist()
        recorder.seed = int(data["seed"])

        # Handle observations array
        obs_array = data["observations"]
        if obs_array.size > 0:
            recorder.observations = [obs_array[i] for i in range(len(obs_array))]

        # Reconstruct metadata
        if "metadata_keys" in data and len(data["metadata_keys"]) > 0:
            keys = data["metadata_keys"].tolist()
            values = data["metadata_values"].tolist()
            recorder.metadata = dict(zip(keys, values, strict=True))

        return recorder

    def get_episode_summary(self) -> dict[str, Any]:
        """Get summary statistics for this recording.

        Returns:
            Dictionary with episode statistics.
        """
        rewards_array = np.array(self.rewards)
        return {
            "seed": self.seed,
            "ticks": self.ticks,
            "duration_seconds": self.duration_seconds,
            "total_reward": self.total_reward,
            "mean_reward": float(rewards_array.mean()) if len(rewards_array) > 0 else 0.0,
            "max_reward": float(rewards_array.max()) if len(rewards_array) > 0 else 0.0,
            "min_reward": float(rewards_array.min()) if len(rewards_array) > 0 else 0.0,
            "positive_reward_steps": int((rewards_array > 0).sum()),
            "negative_reward_steps": int((rewards_array < 0).sum()),
        }

    def __len__(self) -> int:
        """Return number of recorded steps."""
        return len(self.actions)

    def __repr__(self) -> str:
        return f"SpeedrunRecorder(seed={self.seed}, ticks={self.ticks}, reward={self.total_reward:.2f})"


class SpeedrunReplayer:
    """Replays recorded speedrun trajectories.

    Re-executes recorded actions in the environment, yielding observations
    and rewards at each step. Useful for:
    - Validating determinism (comparing replayed vs recorded observations)
    - Generating visualizations
    - Extracting features from specific runs

    Example:
        >>> recording = SpeedrunRecorder.load("run_001.npz")
        >>> replayer = SpeedrunReplayer()
        >>> for obs, reward in replayer.replay(recording):
        ...     render(obs)
    """

    def __init__(self, env_class: type | None = None, **env_kwargs: Any) -> None:
        """Initialize replayer.

        Args:
            env_class: Environment class to use for replay. If None, uses DragonFightEnv.
            **env_kwargs: Additional arguments passed to environment constructor.
        """
        self._env_class = env_class
        self._env_kwargs = env_kwargs

    def replay(
        self,
        recording: SpeedrunRecorder,
        *,
        yield_info: bool = False,
    ) -> Iterator[tuple[np.ndarray, float] | tuple[np.ndarray, float, bool, dict[str, Any]]]:
        """Replay a recorded trajectory.

        Creates a new environment instance, resets with the recorded seed,
        and replays all recorded actions.

        Args:
            recording: Recording to replay.
            yield_info: If True, also yield done flag and info dict.

        Yields:
            Tuples of (observation, reward) or (observation, reward, done, info).
        """
        env = self._create_env()
        try:
            obs, info = env.reset(seed=recording.seed)

            for action in recording.actions:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if yield_info:
                    yield obs, reward, done, info
                else:
                    yield obs, reward

                if done:
                    break
        finally:
            env.close()

    def replay_with_comparison(
        self,
        recording: SpeedrunRecorder,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, float, float, bool]]:
        """Replay and compare against recorded observations.

        Useful for verifying environment determinism.

        Args:
            recording: Recording to replay.
            rtol: Relative tolerance for observation comparison.
            atol: Absolute tolerance for observation comparison.

        Yields:
            Tuples of (replayed_obs, recorded_obs, replayed_reward, recorded_reward, match).
        """
        env = self._create_env()
        try:
            obs, _ = env.reset(seed=recording.seed)

            for i, action in enumerate(recording.actions):
                obs, reward, terminated, truncated, _ = env.step(action)

                # Get recorded observation (offset by 1 if initial obs was recorded)
                rec_obs_idx = i + 1 if len(recording.observations) > len(recording.actions) else i
                recorded_obs = (
                    recording.observations[rec_obs_idx]
                    if rec_obs_idx < len(recording.observations)
                    else obs
                )
                recorded_reward = recording.rewards[i]

                match = np.allclose(obs, recorded_obs, rtol=rtol, atol=atol) and np.isclose(
                    reward, recorded_reward, rtol=rtol, atol=atol
                )

                yield obs, recorded_obs, reward, recorded_reward, match

                if terminated or truncated:
                    break
        finally:
            env.close()

    def get_final_state(self, recording: SpeedrunRecorder) -> tuple[np.ndarray, float, bool]:
        """Replay to get the final state of a recording.

        Args:
            recording: Recording to replay.

        Returns:
            Tuple of (final_observation, total_reward, episode_done).
        """
        final_obs = None
        total_reward = 0.0
        done = False

        for result in self.replay(recording, yield_info=True):
            obs, reward, done, _ = result  # type: ignore
            final_obs = obs
            total_reward += reward
            if done:
                break

        if final_obs is None:
            env = self._create_env()
            final_obs, _ = env.reset(seed=recording.seed)
            env.close()

        return final_obs, total_reward, done

    def _create_env(self) -> gym.Env:
        """Create environment instance for replay."""
        if self._env_class is not None:
            return self._env_class(**self._env_kwargs)

        # Default to DragonFightEnv
        try:
            from .env import DragonFightEnv

            return DragonFightEnv(**self._env_kwargs)
        except ImportError:

import logging

logger = logging.getLogger(__name__)

            raise ImportError(
                "DragonFightEnv not available. Please provide an env_class to SpeedrunReplayer."
            ) from None


class TrajectoryAnalyzer:
    """Analyze recorded trajectories for insights.

    Provides utilities for:
    - Reward distribution analysis
    - Action frequency analysis
    - Observation statistics
    - Segment extraction
    """

    def __init__(self, recording: SpeedrunRecorder) -> None:
        """Initialize analyzer with a recording.

        Args:
            recording: Recording to analyze.
        """
        self.recording = recording
        self._rewards = np.array(recording.rewards)
        self._actions = np.array(recording.actions)

    def get_action_distribution(self) -> dict[int, int]:
        """Get frequency of each action.

        Returns:
            Dictionary mapping action index to count.
        """
        unique, counts = np.unique(self._actions, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist(), strict=True))

    def get_reward_statistics(self) -> dict[str, float]:
        """Get reward distribution statistics.

        Returns:
            Dictionary with reward statistics.
        """
        if len(self._rewards) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}

        return {
            "mean": float(self._rewards.mean()),
            "std": float(self._rewards.std()),
            "min": float(self._rewards.min()),
            "max": float(self._rewards.max()),
            "total": float(self._rewards.sum()),
            "positive_ratio": float((self._rewards > 0).mean()),
            "negative_ratio": float((self._rewards < 0).mean()),
        }

    def get_cumulative_reward(self) -> np.ndarray:
        """Get cumulative reward over time.

        Returns:
            Array of cumulative rewards at each timestep.
        """
        return np.cumsum(self._rewards)

    def find_high_reward_segments(
        self,
        window_size: int = 100,
        threshold: float | None = None,
    ) -> list[tuple[int, int, float]]:
        """Find segments with high reward density.

        Args:
            window_size: Size of sliding window in ticks.
            threshold: Minimum reward sum for segment. Defaults to mean + 1 std.

        Returns:
            List of (start_idx, end_idx, segment_reward) tuples.
        """
        if len(self._rewards) < window_size:
            return [(0, len(self._rewards), float(self._rewards.sum()))]

        # Compute rolling sum
        cumsum = np.concatenate([[0], np.cumsum(self._rewards)])
        rolling_sum = cumsum[window_size:] - cumsum[:-window_size]

        if threshold is None:
            threshold = float(rolling_sum.mean() + rolling_sum.std())

        segments = []
        in_segment = False
        start_idx = 0

        for i, val in enumerate(rolling_sum):
            if val >= threshold and not in_segment:
                in_segment = True
                start_idx = i
            elif val < threshold and in_segment:
                in_segment = False
                end_idx = i + window_size
                segment_reward = float(self._rewards[start_idx:end_idx].sum())
                segments.append((start_idx, end_idx, segment_reward))

        # Handle segment that extends to end
        if in_segment:
            end_idx = len(self._rewards)
            segment_reward = float(self._rewards[start_idx:end_idx].sum())
            segments.append((start_idx, end_idx, segment_reward))

        return segments

    def extract_segment(self, start: int, end: int) -> SpeedrunRecorder:
        """Extract a segment of the recording.

        Args:
            start: Start index (inclusive).
            end: End index (exclusive).

        Returns:
            New SpeedrunRecorder containing only the segment.
        """
        segment = SpeedrunRecorder()
        segment.seed = self.recording.seed
        segment.actions = self.recording.actions[start:end]
        segment.rewards = self.recording.rewards[start:end]

        # Handle observation offset
        obs_start = (
            start + 1 if len(self.recording.observations) > len(self.recording.actions) else start
        )
        obs_end = end + 1 if len(self.recording.observations) > len(self.recording.actions) else end
        segment.observations = self.recording.observations[obs_start:obs_end]

        segment.metadata = {
            "parent_seed": self.recording.seed,
            "segment_start": start,
            "segment_end": end,
        }

        return segment


def merge_recordings(recordings: list[SpeedrunRecorder]) -> SpeedrunRecorder:
    """Merge multiple recordings into one (for multi-episode analysis).

    Note: Seeds and metadata are taken from the first recording.

    Args:
        recordings: List of recordings to merge.

    Returns:
        Merged SpeedrunRecorder.
    """
    if not recordings:
        return SpeedrunRecorder()

    merged = SpeedrunRecorder()
    merged.seed = recordings[0].seed
    merged.metadata = {"merged_from": len(recordings)}

    for rec in recordings:
        merged.actions.extend(rec.actions)
        merged.observations.extend(rec.observations)
        merged.rewards.extend(rec.rewards)

    return merged


def load_recording(path: str | Path) -> SpeedrunRecorder:
    """Convenience function to load a recording.

    Args:
        path: Path to the .npz file.

    Returns:
        Loaded SpeedrunRecorder.
    """
    return SpeedrunRecorder.load(path)


def save_recording(recording: SpeedrunRecorder, path: str | Path) -> None:
    """Convenience function to save a recording.

    Args:
        recording: Recording to save.
        path: Output file path.
    """
    recording.save(path)
