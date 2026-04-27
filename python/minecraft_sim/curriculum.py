"""Curriculum learning stages for progressive Minecraft AI training.

This module defines a curriculum-based training system where an agent
progresses through increasingly difficult stages, from basic survival
to defeating the Ender Dragon.

Each stage can be trained independently and combined for end-to-end runs.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class StageID(IntEnum):
    """Stage identifiers ordered by progression difficulty."""

    BASIC_SURVIVAL = 1
    RESOURCE_GATHERING = 2
    NETHER_NAVIGATION = 3
    ENDERMAN_HUNTING = 4
    STRONGHOLD_FINDING = 5
    END_FIGHT = 6


# Default shader sets per stage. Each stage loads only the shaders it needs,
# reducing GPU memory and pipeline compilation time.
STAGE_SHADER_SETS: dict[StageID, list[str]] = {
    StageID.BASIC_SURVIVAL: [
        "overworld_gen",
        "block_breaking",
        "resource_detection",
    ],
    StageID.RESOURCE_GATHERING: [
        "furnace_tick",
        "item_physics",
        "block_updates",
    ],
    StageID.NETHER_NAVIGATION: [
        "nether_gen",
        "fortress_gen",
        "mob_ai_ghast",
        "mob_ai_blaze",
    ],
    StageID.ENDERMAN_HUNTING: [
        "mob_ai_enderman_full",
        "enderman_spawning",
    ],
    StageID.STRONGHOLD_FINDING: [
        "stronghold_gen",
        "decoration",
    ],
    StageID.END_FIGHT: [
        "dragon_fight_mvk",
        "dragon_ai_full",
        "end_terrain",
    ],
}


def get_shader_set_for_stage(stage_id: StageID | int) -> list[str]:
    """Return the list of shader names required for a given stage.

    Args:
        stage_id: Stage identifier (StageID enum or int value).

    Returns:
        List of shader base names (without .spv extension).
    """
    logger.debug("get_shader_set_for_stage: stage_id=%s", stage_id)
    if not isinstance(stage_id, StageID):
        stage_id = StageID(stage_id)
    return list(STAGE_SHADER_SETS.get(stage_id, []))


@dataclass
class SpawnConfig:
    """Configuration for agent spawn conditions."""

    biome: str = "plains"
    time_of_day: int = 0  # 0-24000 ticks, 0 = dawn
    weather: str = "clear"
    random_position: bool = True
    position: tuple[float, float, float] | None = None
    inventory: dict[str, int] = field(default_factory=dict)
    health: float = 20.0
    hunger: float = 20.0


@dataclass
class RewardConfig:
    """Reward shaping configuration for a stage."""

    sparse_reward: float = 1.0  # Given on stage completion
    dense_rewards: dict[str, float] = field(default_factory=dict)
    penalty_per_death: float = -0.5
    penalty_per_tick: float = -0.0001  # Encourages faster completion
    exploration_bonus: float = 0.01


@dataclass
class TerminationConfig:
    """Conditions that end an episode."""

    max_ticks: int = 36000  # 30 minutes at 20 tps
    max_deaths: int = 5
    success_conditions: list[str] = field(default_factory=list)
    failure_conditions: list[str] = field(default_factory=list)


@dataclass
class Stage:
    """A curriculum learning stage with full configuration.

    Attributes:
        id: Unique stage identifier from StageID enum.
        name: Human-readable stage name.
        description: Detailed description of stage objectives.
        objectives: List of specific objectives to complete.
        spawn: Spawn configuration for episode start.
        rewards: Reward shaping configuration.
        termination: Episode termination conditions.
        prerequisites: List of StageIDs that must be mastered first.
        difficulty: Relative difficulty score (1-10).
        expected_episodes: Estimated episodes to master stage.
        action_space: Subset of actions available in this stage.
        observation_space: Subset of observations relevant to stage.
        curriculum_threshold: Success rate required to advance (0-1).
    """

    id: StageID
    name: str
    description: str
    objectives: list[str]
    spawn: SpawnConfig
    rewards: RewardConfig
    termination: TerminationConfig
    prerequisites: list[StageID] = field(default_factory=list)
    difficulty: int = 1
    expected_episodes: int = 1000
    action_space: list[str] = field(default_factory=list)
    observation_space: list[str] = field(default_factory=list)
    curriculum_threshold: float = 0.7
    shader_set: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize stage to dictionary."""
        logger.debug("Stage.to_dict called")
        return {
            "id": self.id.value,
            "name": self.name,
            "description": self.description,
            "objectives": self.objectives,
            "spawn": {
                "biome": self.spawn.biome,
                "time_of_day": self.spawn.time_of_day,
                "weather": self.spawn.weather,
                "random_position": self.spawn.random_position,
                "position": self.spawn.position,
                "inventory": self.spawn.inventory,
                "health": self.spawn.health,
                "hunger": self.spawn.hunger,
            },
            "rewards": {
                "sparse_reward": self.rewards.sparse_reward,
                "dense_rewards": self.rewards.dense_rewards,
                "penalty_per_death": self.rewards.penalty_per_death,
                "penalty_per_tick": self.rewards.penalty_per_tick,
                "exploration_bonus": self.rewards.exploration_bonus,
            },
            "termination": {
                "max_ticks": self.termination.max_ticks,
                "max_deaths": self.termination.max_deaths,
                "success_conditions": self.termination.success_conditions,
                "failure_conditions": self.termination.failure_conditions,
            },
            "prerequisites": [p.value for p in self.prerequisites],
            "difficulty": self.difficulty,
            "expected_episodes": self.expected_episodes,
            "action_space": self.action_space,
            "observation_space": self.observation_space,
            "curriculum_threshold": self.curriculum_threshold,
            "shader_set": self.shader_set,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stage:
        """Deserialize stage from dictionary."""
        logger.debug("Stage.from_dict: data=%s", data)
        return cls(
            id=StageID(data["id"]),
            name=data["name"],
            description=data["description"],
            objectives=data["objectives"],
            spawn=SpawnConfig(
                biome=data["spawn"]["biome"],
                time_of_day=data["spawn"]["time_of_day"],
                weather=data["spawn"]["weather"],
                random_position=data["spawn"]["random_position"],
                position=tuple(data["spawn"]["position"]) if data["spawn"]["position"] else None,
                inventory=data["spawn"]["inventory"],
                health=data["spawn"]["health"],
                hunger=data["spawn"]["hunger"],
            ),
            rewards=RewardConfig(
                sparse_reward=data["rewards"]["sparse_reward"],
                dense_rewards=data["rewards"]["dense_rewards"],
                penalty_per_death=data["rewards"]["penalty_per_death"],
                penalty_per_tick=data["rewards"]["penalty_per_tick"],
                exploration_bonus=data["rewards"]["exploration_bonus"],
            ),
            termination=TerminationConfig(
                max_ticks=data["termination"]["max_ticks"],
                max_deaths=data["termination"]["max_deaths"],
                success_conditions=data["termination"]["success_conditions"],
                failure_conditions=data["termination"]["failure_conditions"],
            ),
            prerequisites=[StageID(p) for p in data["prerequisites"]],
            difficulty=data["difficulty"],
            expected_episodes=data["expected_episodes"],
            action_space=data["action_space"],
            observation_space=data["observation_space"],
            curriculum_threshold=data["curriculum_threshold"],
            shader_set=data.get("shader_set", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StageProgress:
    """Tracks training progress for a single stage."""

    stage_id: StageID
    episodes_completed: int = 0
    successes: int = 0
    total_reward: float = 0.0
    best_reward: float = float("-inf")
    average_ticks: float = 0.0
    mastered: bool = False

    @property
    def success_rate(self) -> float:
        """Calculate success rate over all episodes."""
        logger.debug("StageProgress.success_rate called")
        if self.episodes_completed == 0:
            return 0.0
        return self.successes / self.episodes_completed

    @property
    def average_reward(self) -> float:
        """Calculate average reward per episode."""
        logger.debug("StageProgress.average_reward called")
        if self.episodes_completed == 0:
            return 0.0
        return self.total_reward / self.episodes_completed


_UNSET = object()


class CurriculumManager:
    """Manages curriculum progression and stage transitions.

    The manager tracks agent progress through stages and determines
    when to advance to harder stages or revisit earlier ones.

    Attributes:
        stages: Dictionary of all available stages.
        progress: Progress tracking for each stage.
        current_stage: Currently active stage for training.
        stage_history: Ordered list of stages trained on.
    """

    def __init__(self, config_dir: Path | str | None = _UNSET):
        """Initialize curriculum manager.

        Args:
            config_dir: Directory containing stage config YAML files.
                       If not provided, uses default stage_configs directory.
                       If explicitly None, no stages are loaded.
        """
        logger.info("CurriculumManager.__init__: config_dir=%s", config_dir)
        self.stages: dict[StageID, Stage] = {}
        self.progress: dict[StageID, StageProgress] = {}
        self.current_stage: StageID | None = None
        self.stage_history: list[StageID] = []
        self._callbacks: list[Callable[[StageID, StageID], None]] = []

        if config_dir is _UNSET:
            config_dir = Path(__file__).parent / "stage_configs"

        if config_dir is not None:
            self.config_dir = Path(config_dir)
            self._load_stages()
        else:
            self.config_dir = Path(__file__).parent / "stage_configs"

        self._initialize_progress()

    def _load_stages(self) -> None:
        """Load stage configurations from YAML files."""
        logger.info("CurriculumManager._load_stages called")
        if not self.config_dir.exists():
            return

        loaded = False
        for config_file in sorted(self.config_dir.glob("stage_*.yaml")):
            with open(config_file) as f:
                data = yaml.safe_load(f)
                stage = Stage.from_dict(data)
                # Auto-populate shader_set from STAGE_SHADER_SETS if not specified
                if not stage.shader_set:
                    stage.shader_set = get_shader_set_for_stage(stage.id)
                self.stages[stage.id] = stage
                loaded = True

        if loaded:
            return

        try:
            from . import stage_configs
        except ImportError:
            return

        for name in getattr(stage_configs, "__all__", []):
            stage = getattr(stage_configs, name)
            self.stages[stage.id] = stage

    def _initialize_progress(self) -> None:
        """Initialize progress tracking for all stages."""
        logger.info("CurriculumManager._initialize_progress called")
        for stage_id in self.stages:
            self.progress[stage_id] = StageProgress(stage_id=stage_id)

    def register_stage(self, stage: Stage) -> None:
        """Register a new stage programmatically.

        Args:
            stage: Stage configuration to register.
        """
        logger.info("CurriculumManager.register_stage: stage=%s", stage)
        self.stages[stage.id] = stage
        if stage.id not in self.progress:
            self.progress[stage.id] = StageProgress(stage_id=stage.id)

    def get_stage(self, stage_id: StageID) -> Stage:
        """Get stage configuration by ID.

        Args:
            stage_id: Stage identifier.

        Returns:
            Stage configuration.

        Raises:
            KeyError: If stage not found.
        """
        logger.debug("CurriculumManager.get_stage: stage_id=%s", stage_id)
        return self.stages[stage_id]

    def start_training(self, stage_id: StageID | None = None) -> Stage:
        """Start training on a specific stage or the first available.

        Args:
            stage_id: Stage to start on. If None, starts at BASIC_SURVIVAL.

        Returns:
            Stage configuration for training.
        """
        logger.info("CurriculumManager.start_training: stage_id=%s", stage_id)
        if stage_id is None:
            stage_id = StageID.BASIC_SURVIVAL

        if stage_id not in self.stages:
            raise KeyError(f"Stage {stage_id} not registered")

        self.current_stage = stage_id
        self.stage_history.append(stage_id)
        return self.stages[stage_id]

    def record_episode(
        self,
        success: bool,
        reward: float,
        ticks: int,
        stage_id: StageID | None = None,
    ) -> bool:
        """Record results from a training episode.

        Args:
            success: Whether the episode succeeded.
            reward: Total reward received.
            ticks: Number of ticks the episode lasted.
            stage_id: Stage ID, defaults to current stage.

        Returns:
            True if stage was mastered after this episode.
        """
        logger.debug("CurriculumManager.record_episode: success=%s, reward=%s, ticks=%s, stage_id=%s", success, reward, ticks, stage_id)
        if stage_id is None:
            stage_id = self.current_stage
        if stage_id is None:
            raise ValueError("No current stage set")

        prog = self.progress[stage_id]
        prog.episodes_completed += 1
        if success:
            prog.successes += 1
        prog.total_reward += reward
        prog.best_reward = max(prog.best_reward, reward)

        # Rolling average for ticks
        alpha = 0.1
        prog.average_ticks = alpha * ticks + (1 - alpha) * prog.average_ticks

        # Check mastery
        stage = self.stages[stage_id]
        min_episodes = max(1, stage.expected_episodes // 10)
        if prog.episodes_completed >= min_episodes:
            if prog.success_rate >= stage.curriculum_threshold:
                prog.mastered = True

        return prog.mastered

    def should_advance(self) -> bool:
        """Check if agent should advance to next stage.

        Returns:
            True if current stage is mastered and next stage available.
        """
        logger.debug("CurriculumManager.should_advance called")
        if self.current_stage is None:
            return False

        prog = self.progress[self.current_stage]
        if not prog.mastered:
            return False

        next_stage = self._get_next_stage()
        return next_stage is not None

    def _get_next_stage(self) -> StageID | None:
        """Get the next stage in curriculum order.

        Returns:
            Next stage ID or None if at final stage.
        """
        logger.debug("CurriculumManager._get_next_stage called")
        if self.current_stage is None:
            return StageID.BASIC_SURVIVAL

        current_value = self.current_stage.value
        try:
            return StageID(current_value + 1)
        except ValueError:
            return None

    def advance_stage(self) -> Stage | None:
        """Advance to the next curriculum stage.

        Returns:
            New stage configuration or None if cannot advance.
        """
        logger.debug("CurriculumManager.advance_stage called")
        next_stage = self._get_next_stage()
        if next_stage is None:
            return None

        if next_stage not in self.stages:
            return None

        # Check prerequisites
        stage = self.stages[next_stage]
        for prereq in stage.prerequisites:
            if prereq not in self.progress or not self.progress[prereq].mastered:
                return None

        old_stage = self.current_stage
        self.current_stage = next_stage
        self.stage_history.append(next_stage)

        # Notify callbacks
        for callback in self._callbacks:
            callback(old_stage, next_stage)

        return stage

    def regress_stage(self, stage_id: StageID | None = None) -> Stage | None:
        """Return to a previous stage for reinforcement.

        Args:
            stage_id: Specific stage to return to.
                     If None, goes to previous stage.

        Returns:
            Stage configuration or None if cannot regress.
        """
        logger.debug("CurriculumManager.regress_stage: stage_id=%s", stage_id)
        if stage_id is not None:
            if stage_id in self.stages:
                old_stage = self.current_stage
                self.current_stage = stage_id
                self.stage_history.append(stage_id)
                for callback in self._callbacks:
                    callback(old_stage, stage_id)
                return self.stages[stage_id]
            return None

        if self.current_stage is None:
            return None

        current_value = self.current_stage.value
        if current_value <= 1:
            return None

        try:
            prev_stage = StageID(current_value - 1)
            old_stage = self.current_stage
            self.current_stage = prev_stage
            self.stage_history.append(prev_stage)
            for callback in self._callbacks:
                callback(old_stage, prev_stage)
            return self.stages.get(prev_stage)
        except ValueError:
            return None

    def on_stage_change(self, callback: Callable[[StageID, StageID], None]) -> None:
        """Register callback for stage transitions.

        Args:
            callback: Function called with (old_stage, new_stage).
        """
        logger.debug("CurriculumManager.on_stage_change: callback=%s", callback)
        self._callbacks.append(callback)

    def get_training_summary(self) -> dict[str, Any]:
        """Get summary of all training progress.

        Returns:
            Dictionary with progress stats for all stages.
        """
        logger.debug("CurriculumManager.get_training_summary called")
        return {
            "current_stage": self.current_stage.name if self.current_stage else None,
            "stages_mastered": sum(1 for p in self.progress.values() if p.mastered),
            "total_stages": len(self.stages),
            "total_episodes": sum(p.episodes_completed for p in self.progress.values()),
            "stage_progress": {
                stage_id.name: {
                    "episodes": prog.episodes_completed,
                    "success_rate": prog.success_rate,
                    "average_reward": prog.average_reward,
                    "best_reward": prog.best_reward,
                    "mastered": prog.mastered,
                }
                for stage_id, prog in self.progress.items()
            },
        }

    def save_progress(self, path: Path | str) -> None:
        """Save training progress to file.

        Args:
            path: Path to save JSON progress file.
        """
        logger.debug("CurriculumManager.save_progress: path=%s", path)
        path = Path(path)
        data = {
            "current_stage": self.current_stage.value if self.current_stage else None,
            "stage_history": [s.value for s in self.stage_history],
            "progress": {
                stage_id.value: {
                    "episodes_completed": prog.episodes_completed,
                    "successes": prog.successes,
                    "total_reward": prog.total_reward,
                    "best_reward": prog.best_reward if prog.best_reward != float("-inf") else None,
                    "average_ticks": prog.average_ticks,
                    "mastered": prog.mastered,
                }
                for stage_id, prog in self.progress.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_progress(self, path: Path | str) -> None:
        """Load training progress from file.

        Args:
            path: Path to JSON progress file.
        """
        logger.info("CurriculumManager.load_progress: path=%s", path)
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        self.current_stage = StageID(data["current_stage"]) if data["current_stage"] else None
        self.stage_history = [StageID(s) for s in data["stage_history"]]

        for stage_id_val, prog_data in data["progress"].items():
            stage_id = StageID(int(stage_id_val))
            self.progress[stage_id] = StageProgress(
                stage_id=stage_id,
                episodes_completed=prog_data["episodes_completed"],
                successes=prog_data["successes"],
                total_reward=prog_data["total_reward"],
                best_reward=prog_data["best_reward"]
                if prog_data["best_reward"] is not None
                else float("-inf"),
                average_ticks=prog_data["average_ticks"],
                mastered=prog_data["mastered"],
            )


# Pre-built curriculum for speedrun training
def create_speedrun_curriculum() -> CurriculumManager:
    """Create a curriculum optimized for Minecraft speedrun training.

    Returns:
        CurriculumManager with all 6 stages configured.
    """
    logger.info("create_speedrun_curriculum called")
    from .stage_configs import (
        STAGE_1_BASIC_SURVIVAL,
        STAGE_2_RESOURCE_GATHERING,
        STAGE_3_NETHER_NAVIGATION,
        STAGE_4_ENDERMAN_HUNTING,
        STAGE_5_STRONGHOLD_FINDING,
        STAGE_6_END_FIGHT,
    )

    manager = CurriculumManager()
    if manager.stages:
        return manager

    manager.register_stage(STAGE_1_BASIC_SURVIVAL)
    manager.register_stage(STAGE_2_RESOURCE_GATHERING)
    manager.register_stage(STAGE_3_NETHER_NAVIGATION)
    manager.register_stage(STAGE_4_ENDERMAN_HUNTING)
    manager.register_stage(STAGE_5_STRONGHOLD_FINDING)
    manager.register_stage(STAGE_6_END_FIGHT)
    return manager


# =============================================================================
# Vectorized Curriculum Manager for Automatic Stage Progression
# =============================================================================


@dataclass
class CurriculumConfig:
    """Configuration for automatic curriculum learning.

    Attributes:
        min_episodes_per_stage: Minimum episodes before considering advancement.
        advancement_threshold: Success rate required to advance (0.0-1.0).
        allow_regression: Whether to allow regression to earlier stages.
        regression_threshold: Success rate below which to regress (0.0-1.0).
        max_episodes_per_stage: Maximum episodes at one stage before forced advancement.
        stage_thresholds: Stage-specific threshold overrides (stage_id -> threshold).
        window_size: Rolling window size for success rate calculation.
        min_stage: Minimum stage ID (default 1).
        max_stage: Maximum stage ID (default 6).
    """

    min_episodes_per_stage: int = 20
    advancement_threshold: float = 0.7
    allow_regression: bool = False
    regression_threshold: float = 0.3
    max_episodes_per_stage: int = 1000
    stage_thresholds: dict[int, float] = field(default_factory=dict)
    window_size: int = 100
    min_stage: int = 1
    max_stage: int = 6


@dataclass
class ProgressionEvent:
    """Records a curriculum progression event (advancement or regression).

    Attributes:
        env_id: Environment that changed stages.
        old_stage: Previous stage ID.
        new_stage: New stage ID.
        episode_number: Global episode count when event occurred.
        success_rate: Success rate at time of progression.
        reason: Why progression occurred ('advancement', 'regression', 'forced').
    """

    env_id: int
    old_stage: int
    new_stage: int
    episode_number: int
    success_rate: float
    reason: str = "advancement"


class AutoCurriculumManager:
    """Manages automatic curriculum progression for vectorized environments.

    Tracks success rates per stage and automatically advances or regresses
    environments based on their performance. Each environment progresses
    independently, enabling efficient curriculum learning in parallel training.

    Args:
        num_envs: Number of parallel environments to manage.
        config: Curriculum configuration (uses defaults if None).

    Attributes:
        num_envs: Number of environments being tracked.
        config: Curriculum configuration.
        env_stages: Current stage for each environment (numpy array).
        env_episode_counts: Episodes completed per environment at current stage.

    Example:
        >>> config = CurriculumConfig(
        ...     advancement_threshold=0.8,
        ...     allow_regression=True,
        ...     regression_threshold=0.2,
        ... )
        >>> manager = AutoCurriculumManager(num_envs=64, config=config)
        >>>
        >>> # After each episode ends
        >>> for env_id, done, success, reward in episode_results:
        ...     if done:
        ...         advanced = manager.update(
        ...             env_id=env_id,
        ...             success=success,
        ...             episode_length=steps,
        ...             episode_reward=reward,
        ...         )
        ...         if advanced:
        ...             print(f"Env {env_id} advanced to stage {manager.get_stage(env_id)}")
        >>>
        >>> # Get statistics
        >>> stats = manager.get_stats()
        >>> print(f"Stage distribution: {stats['stage_distribution']}")
    """

    def __init__(
        self,
        num_envs: int,
        config: CurriculumConfig | None = None,
    ) -> None:
        logger.info("AutoCurriculumManager.__init__: num_envs=%s, config=%s", num_envs, config)
        self.num_envs = num_envs
        self.config = config or CurriculumConfig()

        # Per-environment state
        self.env_stages = np.ones(num_envs, dtype=np.int32) * self.config.min_stage
        self.env_episode_counts = np.zeros(num_envs, dtype=np.int32)

        # Per-environment success history for advancement decisions
        self._env_success_history: dict[int, deque[float]] = {
            i: deque(maxlen=self.config.window_size) for i in range(num_envs)
        }

        # Global stage statistics (aggregated across all envs)
        self.stage_successes: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.window_size)
        )
        self.stage_episode_counts: dict[int, int] = defaultdict(int)
        self.stage_reward_totals: dict[int, float] = defaultdict(float)
        self.stage_best_rewards: dict[int, float] = defaultdict(lambda: float("-inf"))

        # Progression history for logging and analysis
        self.progression_history: list[ProgressionEvent] = []
        self._total_episodes = 0

    def update(
        self,
        env_id: int,
        success: bool,
        episode_length: int = 0,
        episode_reward: float = 0.0,
    ) -> bool:
        """Update curriculum state after episode completion.

        Args:
            env_id: Environment index (0 to num_envs-1).
            success: Whether the episode achieved the stage goal.
            episode_length: Number of steps in episode (for statistics).
            episode_reward: Total reward (for statistics).

        Returns:
            True if environment was advanced or regressed to a different stage.
        """
        logger.debug("AutoCurriculumManager.update: env_id=%s, success=%s, episode_length=%s, episode_reward=%s", env_id, success, episode_length, episode_reward)
        stage = int(self.env_stages[env_id])
        success_val = 1.0 if success else 0.0

        # Update global tracking
        self._total_episodes += 1
        self.stage_successes[stage].append(success_val)
        self.stage_episode_counts[stage] += 1
        self.stage_reward_totals[stage] += episode_reward
        self.stage_best_rewards[stage] = max(self.stage_best_rewards[stage], episode_reward)

        # Update per-environment tracking
        self.env_episode_counts[env_id] += 1
        self._env_success_history[env_id].append(success_val)

        # Check for advancement
        if self._should_advance(env_id, stage):
            return self._advance_stage(env_id, stage)

        # Check for regression (if enabled)
        if self.config.allow_regression and self._should_regress(env_id, stage):
            return self._regress_stage(env_id, stage)

        return False

    def update_batch(
        self,
        env_ids: NDArray[np.int32],
        successes: NDArray[np.bool_],
        episode_lengths: NDArray[np.int32] | None = None,
        episode_rewards: NDArray[np.float32] | None = None,
    ) -> NDArray[np.bool_]:
        """Batch update for multiple environments.

        More efficient than calling update() in a loop when many environments
        complete episodes simultaneously.

        Args:
            env_ids: Array of environment IDs that completed episodes.
            successes: Array of success flags.
            episode_lengths: Array of episode lengths (optional).
            episode_rewards: Array of episode rewards (optional).

        Returns:
            Boolean array indicating which environments changed stages.
        """
        logger.debug("AutoCurriculumManager.update_batch: env_ids=%s, successes=%s, episode_lengths=%s, episode_rewards=%s", env_ids, successes, episode_lengths, episode_rewards)
        n = len(env_ids)
        if episode_lengths is None:
            episode_lengths = np.zeros(n, dtype=np.int32)
        if episode_rewards is None:
            episode_rewards = np.zeros(n, dtype=np.float32)

        changed = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            changed[i] = self.update(
                env_id=int(env_ids[i]),
                success=bool(successes[i]),
                episode_length=int(episode_lengths[i]),
                episode_reward=float(episode_rewards[i]),
            )
        return changed

    def get_stage(self, env_id: int) -> int:
        """Get current stage for environment.

        Args:
            env_id: Environment index.

        Returns:
            Current stage ID (1 to max_stage).
        """
        logger.debug("AutoCurriculumManager.get_stage: env_id=%s", env_id)
        return int(self.env_stages[env_id])

    def get_stages(self) -> NDArray[np.int32]:
        """Get stages for all environments.

        Returns:
            Array of stage IDs, shape (num_envs,).
        """
        logger.debug("AutoCurriculumManager.get_stages called")
        return self.env_stages.copy()

    def get_shader_set(self, env_id: int) -> list[str]:
        """Get the shader set for an environment's current stage.

        Args:
            env_id: Environment index.

        Returns:
            List of shader names required for the environment's current stage.
        """
        logger.debug("AutoCurriculumManager.get_shader_set: env_id=%s", env_id)
        stage = int(self.env_stages[env_id])
        return get_shader_set_for_stage(stage)

    def set_stage(self, env_id: int, stage: int) -> None:
        """Manually set stage for environment.

        Args:
            env_id: Environment index.
            stage: Stage to set (clamped to valid range).
        """
        logger.debug("AutoCurriculumManager.set_stage: env_id=%s, stage=%s", env_id, stage)
        stage = max(self.config.min_stage, min(stage, self.config.max_stage))
        self.env_stages[env_id] = stage
        self.env_episode_counts[env_id] = 0
        self._env_success_history[env_id].clear()

    def set_all_stages(self, stage: int) -> None:
        """Set all environments to the same stage.

        Args:
            stage: Stage to set (clamped to valid range).
        """
        logger.debug("AutoCurriculumManager.set_all_stages: stage=%s", stage)
        stage = max(self.config.min_stage, min(stage, self.config.max_stage))
        self.env_stages.fill(stage)
        self.env_episode_counts.fill(0)
        for env_id in range(self.num_envs):
            self._env_success_history[env_id].clear()

    def get_success_rate(self, stage: int) -> float:
        """Get current success rate for a stage.

        Args:
            stage: Stage ID.

        Returns:
            Success rate (0.0 to 1.0), or 0.0 if no episodes recorded.
        """
        logger.debug("AutoCurriculumManager.get_success_rate: stage=%s", stage)
        history = self.stage_successes.get(stage)
        if history is None or len(history) == 0:
            return 0.0
        return float(np.mean(history))

    def get_env_success_rate(self, env_id: int) -> float:
        """Get recent success rate for a specific environment.

        Args:
            env_id: Environment index.

        Returns:
            Success rate over recent episodes.
        """
        logger.debug("AutoCurriculumManager.get_env_success_rate: env_id=%s", env_id)
        history = self._env_success_history[env_id]
        if len(history) == 0:
            return 0.0
        return float(np.mean(history))

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive curriculum statistics.

        Returns:
            Dictionary containing:
                - stage_distribution: Count of envs at each stage
                - success_rates: Success rate per stage
                - episode_counts: Total episodes per stage
                - reward_stats: Reward statistics per stage
                - total_episodes: Total episodes across all stages
                - total_progressions: Number of stage changes
                - advancements: Number of advancements
                - regressions: Number of regressions
        """
        # Calculate stage distribution
        logger.debug("AutoCurriculumManager.get_stats called")
        unique, counts = np.unique(self.env_stages, return_counts=True)
        stage_distribution = dict(zip(unique.tolist(), counts.tolist()))

        # Calculate success rates and reward stats
        success_rates = {}
        reward_stats = {}
        for stage in range(self.config.min_stage, self.config.max_stage + 1):
            success_rates[stage] = self.get_success_rate(stage)
            ep_count = self.stage_episode_counts[stage]
            reward_stats[stage] = {
                "total": self.stage_reward_totals[stage],
                "average": self.stage_reward_totals[stage] / ep_count if ep_count > 0 else 0.0,
                "best": self.stage_best_rewards[stage]
                if self.stage_best_rewards[stage] != float("-inf")
                else None,
            }

        # Count advancements vs regressions
        advancements = sum(
            1 for e in self.progression_history if e.reason == "advancement" or e.reason == "forced"
        )
        regressions = sum(1 for e in self.progression_history if e.reason == "regression")

        return {
            "stage_distribution": stage_distribution,
            "success_rates": success_rates,
            "episode_counts": dict(self.stage_episode_counts),
            "reward_stats": reward_stats,
            "total_episodes": self._total_episodes,
            "total_progressions": len(self.progression_history),
            "advancements": advancements,
            "regressions": regressions,
            "env_episode_counts": self.env_episode_counts.tolist(),
        }

    def get_summary(self) -> str:
        """Get human-readable summary of curriculum state.

        Returns:
            Formatted string with stage distribution and success rates.
        """
        logger.debug("AutoCurriculumManager.get_summary called")
        stats = self.get_stats()
        lines = [
            f"AutoCurriculum Summary ({self.num_envs} envs, {self._total_episodes} episodes):",
            f"  Progressions: {stats['total_progressions']} "
            f"(+{stats['advancements']} adv, -{stats['regressions']} reg)",
        ]

        for stage in range(self.config.min_stage, self.config.max_stage + 1):
            count = stats["stage_distribution"].get(stage, 0)
            pct = 100 * count / self.num_envs
            rate = stats["success_rates"][stage]
            threshold = self.config.stage_thresholds.get(stage, self.config.advancement_threshold)
            rate_str = f"{rate:.1%}" if rate > 0 else "N/A"
            lines.append(
                f"  Stage {stage}: {count:3d} envs ({pct:5.1f}%) | "
                f"Success: {rate_str} (need {threshold:.0%})"
            )

        return "\n".join(lines)

    def get_recent_progressions(self, n: int = 10) -> list[ProgressionEvent]:
        """Get recent progression events.

        Args:
            n: Number of recent events to return.

        Returns:
            List of recent ProgressionEvent objects.
        """
        logger.debug("AutoCurriculumManager.get_recent_progressions: n=%s", n)
        return self.progression_history[-n:]

    def _should_advance(self, env_id: int, stage: int) -> bool:
        """Check if environment should advance to next stage.

        Args:
            env_id: Environment index.
            stage: Current stage.

        Returns:
            True if advancement criteria met.
        """
        logger.debug("AutoCurriculumManager._should_advance: env_id=%s, stage=%s", env_id, stage)
        if stage >= self.config.max_stage:
            return False

        episodes = self.env_episode_counts[env_id]

        # Force advancement after max episodes
        if episodes >= self.config.max_episodes_per_stage:
            return True

        # Check minimum episodes
        if episodes < self.config.min_episodes_per_stage:
            return False

        # Check success rate against threshold
        threshold = self.config.stage_thresholds.get(stage, self.config.advancement_threshold)
        success_rate = self.get_env_success_rate(env_id)

        return success_rate >= threshold

    def _should_regress(self, env_id: int, stage: int) -> bool:
        """Check if environment should regress to previous stage.

        Args:
            env_id: Environment index.
            stage: Current stage.

        Returns:
            True if regression criteria met.
        """
        logger.debug("AutoCurriculumManager._should_regress: env_id=%s, stage=%s", env_id, stage)
        if stage <= self.config.min_stage:
            return False

        # Require more episodes before regression
        min_for_regress = self.config.min_episodes_per_stage * 2
        if self.env_episode_counts[env_id] < min_for_regress:
            return False

        success_rate = self.get_env_success_rate(env_id)
        return success_rate < self.config.regression_threshold

    def _advance_stage(self, env_id: int, old_stage: int) -> bool:
        """Advance environment to next stage.

        Args:
            env_id: Environment index.
            old_stage: Current stage before advancement.

        Returns:
            True (always advances).
        """
        logger.debug("AutoCurriculumManager._advance_stage: env_id=%s, old_stage=%s", env_id, old_stage)
        new_stage = old_stage + 1
        success_rate = self.get_env_success_rate(env_id)

        # Determine reason
        reason = (
            "forced"
            if self.env_episode_counts[env_id] >= self.config.max_episodes_per_stage
            else "advancement"
        )

        self.env_stages[env_id] = new_stage
        self.env_episode_counts[env_id] = 0
        self._env_success_history[env_id].clear()

        event = ProgressionEvent(
            env_id=env_id,
            old_stage=old_stage,
            new_stage=new_stage,
            episode_number=self._total_episodes,
            success_rate=success_rate,
            reason=reason,
        )
        self.progression_history.append(event)

        logger.info(
            f"Env {env_id} advanced: stage {old_stage} -> {new_stage} "
            f"(rate={success_rate:.1%}, reason={reason})"
        )

        return True

    def _regress_stage(self, env_id: int, old_stage: int) -> bool:
        """Regress environment to previous stage.

        Args:
            env_id: Environment index.
            old_stage: Current stage before regression.

        Returns:
            True (always regresses).
        """
        logger.debug("AutoCurriculumManager._regress_stage: env_id=%s, old_stage=%s", env_id, old_stage)
        new_stage = old_stage - 1
        success_rate = self.get_env_success_rate(env_id)

        self.env_stages[env_id] = new_stage
        self.env_episode_counts[env_id] = 0
        self._env_success_history[env_id].clear()

        event = ProgressionEvent(
            env_id=env_id,
            old_stage=old_stage,
            new_stage=new_stage,
            episode_number=self._total_episodes,
            success_rate=success_rate,
            reason="regression",
        )
        self.progression_history.append(event)

        logger.warning(
            f"Env {env_id} regressed: stage {old_stage} -> {new_stage} (rate={success_rate:.1%})"
        )

        return True

    def reset(self) -> None:
        """Reset all curriculum state to initial values."""
        logger.debug("AutoCurriculumManager.reset called")
        self.env_stages.fill(self.config.min_stage)
        self.env_episode_counts.fill(0)
        self._total_episodes = 0

        for env_id in range(self.num_envs):
            self._env_success_history[env_id].clear()

        self.stage_successes.clear()
        self.stage_episode_counts.clear()
        self.stage_reward_totals.clear()
        self.stage_best_rewards.clear()
        self.progression_history.clear()

    def save_state(self) -> dict[str, Any]:
        """Save curriculum state for checkpointing.

        Returns:
            Dictionary containing all state needed to restore.
        """
        logger.debug("AutoCurriculumManager.save_state called")
        return {
            "config": {
                "min_episodes_per_stage": self.config.min_episodes_per_stage,
                "advancement_threshold": self.config.advancement_threshold,
                "allow_regression": self.config.allow_regression,
                "regression_threshold": self.config.regression_threshold,
                "max_episodes_per_stage": self.config.max_episodes_per_stage,
                "stage_thresholds": self.config.stage_thresholds,
                "window_size": self.config.window_size,
                "min_stage": self.config.min_stage,
                "max_stage": self.config.max_stage,
            },
            "env_stages": self.env_stages.tolist(),
            "env_episode_counts": self.env_episode_counts.tolist(),
            "env_success_history": {i: list(d) for i, d in self._env_success_history.items()},
            "stage_successes": {s: list(d) for s, d in self.stage_successes.items()},
            "stage_episode_counts": dict(self.stage_episode_counts),
            "stage_reward_totals": dict(self.stage_reward_totals),
            "stage_best_rewards": {
                s: r if r != float("-inf") else None for s, r in self.stage_best_rewards.items()
            },
            "progression_history": [
                {
                    "env_id": e.env_id,
                    "old_stage": e.old_stage,
                    "new_stage": e.new_stage,
                    "episode_number": e.episode_number,
                    "success_rate": e.success_rate,
                    "reason": e.reason,
                }
                for e in self.progression_history
            ],
            "total_episodes": self._total_episodes,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore curriculum state from checkpoint.

        Args:
            state: Dictionary from save_state().
        """
        # Restore config (optional - allows loading with different config)
        logger.info("AutoCurriculumManager.load_state: state=%s", state)
        if "config" in state:
            cfg = state["config"]
            self.config = CurriculumConfig(
                min_episodes_per_stage=cfg["min_episodes_per_stage"],
                advancement_threshold=cfg["advancement_threshold"],
                allow_regression=cfg["allow_regression"],
                regression_threshold=cfg["regression_threshold"],
                max_episodes_per_stage=cfg["max_episodes_per_stage"],
                stage_thresholds=cfg["stage_thresholds"],
                window_size=cfg["window_size"],
                min_stage=cfg["min_stage"],
                max_stage=cfg["max_stage"],
            )

        self.env_stages = np.array(state["env_stages"], dtype=np.int32)
        self.env_episode_counts = np.array(state["env_episode_counts"], dtype=np.int32)
        self._total_episodes = state["total_episodes"]

        # Restore per-env success history
        for i_str, values in state["env_success_history"].items():
            env_id = int(i_str)
            self._env_success_history[env_id] = deque(values, maxlen=self.config.window_size)

        # Restore stage statistics
        for s_str, values in state["stage_successes"].items():
            stage = int(s_str)
            self.stage_successes[stage] = deque(values, maxlen=self.config.window_size)

        self.stage_episode_counts = defaultdict(
            int, {int(k): v for k, v in state["stage_episode_counts"].items()}
        )
        self.stage_reward_totals = defaultdict(
            float, {int(k): v for k, v in state["stage_reward_totals"].items()}
        )
        self.stage_best_rewards = defaultdict(
            lambda: float("-inf"),
            {
                int(k): (v if v is not None else float("-inf"))
                for k, v in state["stage_best_rewards"].items()
            },
        )

        # Restore progression history
        self.progression_history = [ProgressionEvent(**e) for e in state["progression_history"]]
