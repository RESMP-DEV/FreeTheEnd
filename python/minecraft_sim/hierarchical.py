"""Hierarchical RL training for Minecraft with option-critic architecture.

This module implements a two-level hierarchical policy:
1. High-level policy: Selects sub-goals (options) over extended timeframes
2. Low-level policy: Executes primitive actions to achieve sub-goals

Sub-goal examples:
- Resource gathering: "get wood", "mine iron", "collect blaze rods"
- Navigation: "find fortress", "locate stronghold", "reach end portal"
- Combat: "kill blaze", "defeat dragon", "eliminate endermen"

The system supports:
- Automatic sub-goal detection from expert trajectories
- Option-critic style termination learning
- Intrinsic motivation for sub-goal completion
- Temporal abstraction with variable-length options
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SUB-GOAL DEFINITIONS
# =============================================================================


class SubGoalCategory(IntEnum):
    """Categories of sub-goals for organization."""

    RESOURCE = auto()  # Obtaining items/resources
    NAVIGATION = auto()  # Reaching locations
    COMBAT = auto()  # Fighting entities
    CRAFTING = auto()  # Creating items
    EXPLORATION = auto()  # Discovering world features
    SURVIVAL = auto()  # Maintaining health/hunger


class SubGoalID(IntEnum):
    """Enumeration of all predefined sub-goals.

    These form the action space for the high-level policy.
    Order roughly follows speedrun progression.
    """

    # Resource gathering (1-20)
    GET_WOOD = 1
    GET_STONE = 2
    GET_IRON = 3
    GET_DIAMONDS = 4
    GET_OBSIDIAN = 5
    GET_BLAZE_RODS = 6
    GET_ENDER_PEARLS = 7
    CRAFT_EYES_OF_ENDER = 8
    GET_FOOD = 9
    GET_BEDS = 10

    # Navigation (21-40)
    FIND_VILLAGE = 21
    FIND_NETHER_PORTAL = 22
    ENTER_NETHER = 23
    FIND_FORTRESS = 24
    FIND_BLAZE_SPAWNER = 25
    RETURN_TO_OVERWORLD = 26
    LOCATE_STRONGHOLD = 27
    FIND_END_PORTAL = 28
    ENTER_END = 29
    REACH_DRAGON_ISLAND = 30

    # Combat (41-60)
    KILL_ZOMBIES = 41
    KILL_SKELETONS = 42
    KILL_CREEPERS = 43
    KILL_BLAZES = 44
    KILL_ENDERMEN = 45
    DEFEAT_DRAGON = 46
    DESTROY_CRYSTALS = 47

    # Survival (61-70)
    HEAL_UP = 61
    EAT_FOOD = 62
    FIND_SHELTER = 63
    SLEEP = 64

    # Special (71-80)
    WAIT = 71  # Explicit no-op / temporal padding
    EXPLORE = 72  # General exploration
    CUSTOM = 73  # User-defined sub-goal


@dataclass(frozen=True, slots=True)
class SubGoal:
    """A sub-goal (option) in the hierarchical policy.

    Attributes:
        id: Unique identifier from SubGoalID enum.
        name: Human-readable name.
        category: Category for organization.
        description: Detailed description of the sub-goal.
        completion_predicate: Function to check if sub-goal is achieved.
        initiation_set: Function to check if sub-goal can be started.
        max_steps: Maximum steps before forced termination.
        intrinsic_reward: Reward given on successful completion.
        terminal: If True, episode ends when this sub-goal completes.
    """

    id: SubGoalID
    name: str
    category: SubGoalCategory
    description: str = ""
    max_steps: int = 1200  # 1 minute at 20 tps
    intrinsic_reward: float = 1.0
    terminal: bool = False

    def check_completion(self, obs: NDArray[np.float32], info: dict[str, Any]) -> bool:
        """Check if sub-goal is completed based on observation.

        Default implementation checks for specific conditions in info dict.
        Override for custom logic.
        """
        logger.debug("SubGoal.check_completion: obs=%s, info=%s", obs, info)
        if f"subgoal_{self.name}_complete" in info:
            return bool(info[f"subgoal_{self.name}_complete"])
        return False

    def check_initiation(self, obs: NDArray[np.float32], info: dict[str, Any]) -> bool:
        """Check if sub-goal can be initiated from current state.

        Default implementation returns True (always initiable).
        Override to add preconditions.
        """
        logger.info("SubGoal.check_initiation: obs=%s, info=%s", obs, info)
        return True


# =============================================================================
# SUB-GOAL REGISTRY
# =============================================================================


class SubGoalRegistry:
    """Registry of available sub-goals with completion predicates.

    Provides factory methods and condition checking for all sub-goals.
    """

    _instance: SubGoalRegistry | None = None
    _subgoals: dict[SubGoalID, SubGoal]

    def __new__(cls) -> SubGoalRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subgoals = {}
            cls._instance._register_defaults()
        return cls._instance

    def _register_defaults(self) -> None:
        """Register all default sub-goals."""
        # Resource gathering
        logger.info("SubGoalRegistry._register_defaults called")
        self.register(
            SubGoal(
                id=SubGoalID.GET_WOOD,
                name="get_wood",
                category=SubGoalCategory.RESOURCE,
                description="Collect wood logs from trees",
                max_steps=600,
                intrinsic_reward=0.5,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.GET_STONE,
                name="get_stone",
                category=SubGoalCategory.RESOURCE,
                description="Mine cobblestone with wooden pickaxe",
                max_steps=600,
                intrinsic_reward=0.5,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.GET_IRON,
                name="get_iron",
                category=SubGoalCategory.RESOURCE,
                description="Mine and smelt iron ore",
                max_steps=1200,
                intrinsic_reward=1.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.GET_DIAMONDS,
                name="get_diamonds",
                category=SubGoalCategory.RESOURCE,
                description="Find and mine diamond ore",
                max_steps=3600,
                intrinsic_reward=2.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.GET_OBSIDIAN,
                name="get_obsidian",
                category=SubGoalCategory.RESOURCE,
                description="Mine obsidian blocks",
                max_steps=1800,
                intrinsic_reward=1.5,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.GET_BLAZE_RODS,
                name="get_blaze_rods",
                category=SubGoalCategory.RESOURCE,
                description="Collect blaze rods from blazes",
                max_steps=2400,
                intrinsic_reward=2.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.GET_ENDER_PEARLS,
                name="get_ender_pearls",
                category=SubGoalCategory.RESOURCE,
                description="Obtain ender pearls from endermen or trading",
                max_steps=3600,
                intrinsic_reward=2.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.CRAFT_EYES_OF_ENDER,
                name="craft_eyes_of_ender",
                category=SubGoalCategory.CRAFTING,
                description="Craft eyes of ender from pearls and blaze powder",
                max_steps=300,
                intrinsic_reward=1.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.GET_FOOD,
                name="get_food",
                category=SubGoalCategory.RESOURCE,
                description="Obtain food items for survival",
                max_steps=600,
                intrinsic_reward=0.3,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.GET_BEDS,
                name="get_beds",
                category=SubGoalCategory.RESOURCE,
                description="Craft or find beds for respawn/explosion",
                max_steps=900,
                intrinsic_reward=0.5,
            )
        )

        # Navigation
        self.register(
            SubGoal(
                id=SubGoalID.FIND_VILLAGE,
                name="find_village",
                category=SubGoalCategory.NAVIGATION,
                description="Locate a village structure",
                max_steps=3600,
                intrinsic_reward=1.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.FIND_NETHER_PORTAL,
                name="find_nether_portal",
                category=SubGoalCategory.NAVIGATION,
                description="Find or build a nether portal",
                max_steps=2400,
                intrinsic_reward=1.5,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.ENTER_NETHER,
                name="enter_nether",
                category=SubGoalCategory.NAVIGATION,
                description="Enter the nether dimension",
                max_steps=600,
                intrinsic_reward=1.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.FIND_FORTRESS,
                name="find_fortress",
                category=SubGoalCategory.NAVIGATION,
                description="Locate a nether fortress",
                max_steps=6000,
                intrinsic_reward=2.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.FIND_BLAZE_SPAWNER,
                name="find_blaze_spawner",
                category=SubGoalCategory.NAVIGATION,
                description="Find a blaze spawner in the fortress",
                max_steps=1800,
                intrinsic_reward=1.5,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.RETURN_TO_OVERWORLD,
                name="return_to_overworld",
                category=SubGoalCategory.NAVIGATION,
                description="Return to the overworld through portal",
                max_steps=1200,
                intrinsic_reward=0.5,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.LOCATE_STRONGHOLD,
                name="locate_stronghold",
                category=SubGoalCategory.NAVIGATION,
                description="Use eyes of ender to find stronghold",
                max_steps=6000,
                intrinsic_reward=2.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.FIND_END_PORTAL,
                name="find_end_portal",
                category=SubGoalCategory.NAVIGATION,
                description="Locate the end portal room in stronghold",
                max_steps=1800,
                intrinsic_reward=1.5,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.ENTER_END,
                name="enter_end",
                category=SubGoalCategory.NAVIGATION,
                description="Activate and enter the end portal",
                max_steps=600,
                intrinsic_reward=2.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.REACH_DRAGON_ISLAND,
                name="reach_dragon_island",
                category=SubGoalCategory.NAVIGATION,
                description="Navigate to the main end island",
                max_steps=600,
                intrinsic_reward=0.5,
            )
        )

        # Combat
        self.register(
            SubGoal(
                id=SubGoalID.KILL_ZOMBIES,
                name="kill_zombies",
                category=SubGoalCategory.COMBAT,
                description="Defeat zombie enemies",
                max_steps=600,
                intrinsic_reward=0.3,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.KILL_SKELETONS,
                name="kill_skeletons",
                category=SubGoalCategory.COMBAT,
                description="Defeat skeleton enemies",
                max_steps=600,
                intrinsic_reward=0.3,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.KILL_CREEPERS,
                name="kill_creepers",
                category=SubGoalCategory.COMBAT,
                description="Defeat creeper enemies safely",
                max_steps=900,
                intrinsic_reward=0.4,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.KILL_BLAZES,
                name="kill_blazes",
                category=SubGoalCategory.COMBAT,
                description="Defeat blaze enemies for rods",
                max_steps=1200,
                intrinsic_reward=1.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.KILL_ENDERMEN,
                name="kill_endermen",
                category=SubGoalCategory.COMBAT,
                description="Defeat endermen for pearls",
                max_steps=1200,
                intrinsic_reward=1.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.DEFEAT_DRAGON,
                name="defeat_dragon",
                category=SubGoalCategory.COMBAT,
                description="Defeat the ender dragon",
                max_steps=6000,
                intrinsic_reward=10.0,
                terminal=True,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.DESTROY_CRYSTALS,
                name="destroy_crystals",
                category=SubGoalCategory.COMBAT,
                description="Destroy end crystals on towers",
                max_steps=3600,
                intrinsic_reward=3.0,
            )
        )

        # Survival
        self.register(
            SubGoal(
                id=SubGoalID.HEAL_UP,
                name="heal_up",
                category=SubGoalCategory.SURVIVAL,
                description="Regenerate health to full",
                max_steps=600,
                intrinsic_reward=0.2,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.EAT_FOOD,
                name="eat_food",
                category=SubGoalCategory.SURVIVAL,
                description="Consume food to restore hunger",
                max_steps=200,
                intrinsic_reward=0.1,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.FIND_SHELTER,
                name="find_shelter",
                category=SubGoalCategory.SURVIVAL,
                description="Find or build shelter from mobs",
                max_steps=900,
                intrinsic_reward=0.3,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.SLEEP,
                name="sleep",
                category=SubGoalCategory.SURVIVAL,
                description="Use bed to skip night",
                max_steps=400,
                intrinsic_reward=0.2,
            )
        )

        # Special
        self.register(
            SubGoal(
                id=SubGoalID.WAIT,
                name="wait",
                category=SubGoalCategory.SURVIVAL,
                description="Wait / no-op for temporal padding",
                max_steps=100,
                intrinsic_reward=0.0,
            )
        )
        self.register(
            SubGoal(
                id=SubGoalID.EXPLORE,
                name="explore",
                category=SubGoalCategory.EXPLORATION,
                description="Explore the environment",
                max_steps=1200,
                intrinsic_reward=0.1,
            )
        )

    def register(self, subgoal: SubGoal) -> None:
        """Register a sub-goal."""
        logger.info("SubGoalRegistry.register: subgoal=%s", subgoal)
        self._subgoals[subgoal.id] = subgoal

    def get(self, subgoal_id: SubGoalID) -> SubGoal:
        """Get sub-goal by ID."""
        logger.debug("SubGoalRegistry.get: subgoal_id=%s", subgoal_id)
        return self._subgoals[subgoal_id]

    def get_by_name(self, name: str) -> SubGoal | None:
        """Get sub-goal by name."""
        logger.debug("SubGoalRegistry.get_by_name: name=%s", name)
        for sg in self._subgoals.values():
            if sg.name == name:
                return sg
        return None

    def get_by_category(self, category: SubGoalCategory) -> list[SubGoal]:
        """Get all sub-goals in a category."""
        logger.debug("SubGoalRegistry.get_by_category: category=%s", category)
        return [sg for sg in self._subgoals.values() if sg.category == category]

    def all_subgoals(self) -> list[SubGoal]:
        """Get all registered sub-goals."""
        logger.debug("SubGoalRegistry.all_subgoals called")
        return list(self._subgoals.values())

    @property
    def num_subgoals(self) -> int:
        """Number of registered sub-goals."""
        logger.debug("SubGoalRegistry.num_subgoals called")
        return len(self._subgoals)


# =============================================================================
# HIERARCHICAL POLICY INTERFACES
# =============================================================================

ObsType = TypeVar("ObsType", bound=np.ndarray)
ActType = TypeVar("ActType", bound=np.ndarray | int)


@dataclass
class OptionState:
    """State of the currently executing option (sub-goal).

    Attributes:
        subgoal: The active sub-goal.
        steps_taken: Steps executed in current option.
        cumulative_reward: Total reward accumulated during option.
        start_obs: Observation when option was initiated.
        terminated: Whether option has terminated naturally.
    """

    subgoal: SubGoal
    steps_taken: int = 0
    cumulative_reward: float = 0.0
    start_obs: NDArray[np.float32] | None = None
    terminated: bool = False


class HighLevelPolicy(ABC, Generic[ObsType]):
    """Abstract base class for high-level (manager) policy.

    The high-level policy observes the environment at a coarser timescale
    and selects which sub-goal (option) to pursue.
    """

    @abstractmethod
    def select_subgoal(
        self,
        obs: ObsType,
        info: dict[str, Any],
        available_subgoals: list[SubGoalID] | None = None,
    ) -> SubGoalID:
        """Select a sub-goal based on current observation.

        Args:
            obs: Current environment observation.
            info: Additional info from environment.
            available_subgoals: Optional mask of available sub-goals.

        Returns:
            Selected sub-goal ID.
        """
        logger.debug("HighLevelPolicy.select_subgoal: obs=%s, info=%s, available_subgoals=%s", obs, info, available_subgoals)
        ...

    @abstractmethod
    def update(
        self,
        obs: ObsType,
        selected_subgoal: SubGoalID,
        option_reward: float,
        next_obs: ObsType,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> dict[str, float]:
        """Update policy after option completes.

        Args:
            obs: Observation when option started.
            selected_subgoal: The sub-goal that was selected.
            option_reward: Total reward during option execution.
            next_obs: Observation after option completed.
            terminated: Whether episode terminated.
            truncated: Whether episode was truncated.
            info: Additional info from environment.

        Returns:
            Dictionary of training metrics.
        """
        logger.debug("HighLevelPolicy.update: obs=%s, selected_subgoal=%s, option_reward=%s, next_obs=%s", obs, selected_subgoal, option_reward, next_obs)
        ...

    def get_subgoal_values(self, obs: ObsType, info: dict[str, Any]) -> NDArray[np.float32]:
        """Get Q-values for all sub-goals (optional).

        Args:
            obs: Current observation.
            info: Additional info.

        Returns:
            Array of Q-values, one per sub-goal.
        """
        logger.debug("HighLevelPolicy.get_subgoal_values: obs=%s, info=%s", obs, info)
        registry = SubGoalRegistry()
        return np.zeros(registry.num_subgoals, dtype=np.float32)


class LowLevelPolicy(ABC, Generic[ObsType, ActType]):
    """Abstract base class for low-level (worker) policy.

    The low-level policy executes primitive actions to achieve
    the sub-goal selected by the high-level policy.
    """

    @abstractmethod
    def select_action(
        self,
        obs: ObsType,
        subgoal: SubGoal,
        info: dict[str, Any],
    ) -> ActType:
        """Select primitive action given current sub-goal.

        Args:
            obs: Current environment observation.
            subgoal: Active sub-goal to work towards.
            info: Additional info from environment.

        Returns:
            Primitive action to execute.
        """
        logger.debug("LowLevelPolicy.select_action: obs=%s, subgoal=%s, info=%s", obs, subgoal, info)
        ...

    @abstractmethod
    def update(
        self,
        obs: ObsType,
        action: ActType,
        reward: float,
        next_obs: ObsType,
        done: bool,
        subgoal: SubGoal,
        info: dict[str, Any],
    ) -> dict[str, float]:
        """Update policy after primitive action.

        Args:
            obs: Observation before action.
            action: Action that was taken.
            reward: Reward received.
            next_obs: Observation after action.
            done: Whether episode ended.
            subgoal: Active sub-goal during action.
            info: Additional info from environment.

        Returns:
            Dictionary of training metrics.
        """
        logger.debug("LowLevelPolicy.update: obs=%s, action=%s, reward=%s, next_obs=%s", obs, action, reward, next_obs)
        ...

    def check_termination(
        self,
        obs: ObsType,
        subgoal: SubGoal,
        steps_taken: int,
        info: dict[str, Any],
    ) -> tuple[bool, float]:
        """Check if current option should terminate.

        Used for option-critic style learned termination.

        Args:
            obs: Current observation.
            subgoal: Active sub-goal.
            steps_taken: Steps taken in current option.
            info: Additional info.

        Returns:
            Tuple of (should_terminate, termination_probability).
        """
        # Default: terminate on completion or max steps
        logger.debug("LowLevelPolicy.check_termination: obs=%s, subgoal=%s, steps_taken=%s, info=%s", obs, subgoal, steps_taken, info)
        if subgoal.check_completion(obs, info):
            return True, 1.0
        if steps_taken >= subgoal.max_steps:
            return True, 1.0
        return False, 0.0


class TerminationCritic(ABC, Generic[ObsType]):
    """Option-critic termination function.

    Learns when to terminate an option based on value of continuing
    vs starting a new option.
    """

    @abstractmethod
    def termination_probability(
        self,
        obs: ObsType,
        subgoal: SubGoal,
        option_state: OptionState,
        info: dict[str, Any],
    ) -> float:
        """Compute probability of terminating current option.

        Args:
            obs: Current observation.
            subgoal: Active sub-goal.
            option_state: Current option execution state.
            info: Additional info.

        Returns:
            Probability in [0, 1] of terminating.
        """
        logger.debug("TerminationCritic.termination_probability: obs=%s, subgoal=%s, option_state=%s, info=%s", obs, subgoal, option_state, info)
        ...

    @abstractmethod
    def update(
        self,
        obs: ObsType,
        subgoal: SubGoal,
        terminated: bool,
        advantage: float,
        info: dict[str, Any],
    ) -> dict[str, float]:
        """Update termination function.

        Args:
            obs: Observation at termination decision.
            subgoal: Active sub-goal.
            terminated: Whether option was terminated.
            advantage: Advantage of terminating vs continuing.
            info: Additional info.

        Returns:
            Training metrics.
        """
        logger.debug("TerminationCritic.update: obs=%s, subgoal=%s, terminated=%s, advantage=%s", obs, subgoal, terminated, advantage)
        ...


# =============================================================================
# HIERARCHICAL CONTROLLER
# =============================================================================


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical RL training.

    Attributes:
        subgoal_reward_scale: Scale for intrinsic sub-goal rewards.
        extrinsic_reward_scale: Scale for environment rewards.
        termination_reg: Regularization to discourage too-frequent termination.
        temporal_discount: Discount for high-level value function.
        option_discount: Discount for low-level value function.
        max_option_steps: Maximum steps per option before forced termination.
        min_option_steps: Minimum steps before option can terminate.
        use_learned_termination: Whether to learn termination function.
        subgoal_embedding_dim: Dimension of sub-goal embeddings.
    """

    subgoal_reward_scale: float = 1.0
    extrinsic_reward_scale: float = 1.0
    termination_reg: float = 0.01
    temporal_discount: float = 0.99
    option_discount: float = 0.99
    max_option_steps: int = 1200
    min_option_steps: int = 10
    use_learned_termination: bool = True
    subgoal_embedding_dim: int = 32


class HierarchicalController(Generic[ObsType, ActType]):
    """Controller for hierarchical RL with sub-goals.

    Coordinates high-level and low-level policies, handling
    option selection, execution, and termination.

    Example:
        >>> controller = HierarchicalController(
        ...     high_policy=MyHighPolicy(),
        ...     low_policy=MyLowPolicy(),
        ... )
        >>> obs, info = env.reset()
        >>> while not done:
        ...     action = controller.step(obs, info)
        ...     obs, reward, done, truncated, info = env.step(action)
        ...     controller.receive_reward(reward, done, truncated, info)
    """

    def __init__(
        self,
        high_policy: HighLevelPolicy[ObsType],
        low_policy: LowLevelPolicy[ObsType, ActType],
        config: HierarchicalConfig | None = None,
        termination_critic: TerminationCritic[ObsType] | None = None,
    ):
        """Initialize hierarchical controller.

        Args:
            high_policy: Policy for sub-goal selection.
            low_policy: Policy for primitive action selection.
            config: Hierarchical RL configuration.
            termination_critic: Optional learned termination function.
        """
        logger.info("HierarchicalController.__init__: high_policy=%s, low_policy=%s, config=%s, termination_critic=%s", high_policy, low_policy, config, termination_critic)
        self.high_policy = high_policy
        self.low_policy = low_policy
        self.config = config or HierarchicalConfig()
        self.termination_critic = termination_critic
        self.registry = SubGoalRegistry()

        self._option_state: OptionState | None = None
        self._last_obs: ObsType | None = None
        self._episode_stats: dict[str, float] = defaultdict(float)

    @property
    def current_subgoal(self) -> SubGoal | None:
        """Currently active sub-goal."""
        logger.debug("HierarchicalController.current_subgoal called")
        return self._option_state.subgoal if self._option_state else None

    def reset(self) -> None:
        """Reset controller state for new episode."""
        logger.debug("HierarchicalController.reset called")
        self._option_state = None
        self._last_obs = None
        self._episode_stats = defaultdict(float)

    def step(self, obs: ObsType, info: dict[str, Any]) -> ActType:
        """Execute one step: select sub-goal if needed, then select action.

        Args:
            obs: Current environment observation.
            info: Additional info from environment.

        Returns:
            Primitive action to execute.
        """
        logger.debug("HierarchicalController.step: obs=%s, info=%s", obs, info)
        self._last_obs = obs

        # Check if we need to select a new sub-goal
        if self._option_state is None or self._should_terminate(obs, info):
            self._select_new_subgoal(obs, info)

        # Execute low-level policy
        assert self._option_state is not None
        action = self.low_policy.select_action(obs, self._option_state.subgoal, info)
        self._option_state.steps_taken += 1

        return action

    def receive_reward(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> dict[str, float]:
        """Process reward and update policies.

        Args:
            reward: Environment reward.
            terminated: Whether episode terminated.
            truncated: Whether episode was truncated.
            info: Additional info.

        Returns:
            Dictionary of training metrics.
        """
        logger.debug("HierarchicalController.receive_reward: reward=%s, terminated=%s, truncated=%s, info=%s", reward, terminated, truncated, info)
        if self._option_state is None or self._last_obs is None:
            return {}

        metrics: dict[str, float] = {}
        scaled_reward = reward * self.config.extrinsic_reward_scale
        self._option_state.cumulative_reward += scaled_reward

        # Update low-level policy
        low_metrics = self.low_policy.update(
            self._last_obs,
            self.low_policy.select_action(self._last_obs, self._option_state.subgoal, info),
            scaled_reward,
            self._last_obs,  # Will be updated next step
            terminated or truncated,
            self._option_state.subgoal,
            info,
        )
        metrics.update({f"low/{k}": v for k, v in low_metrics.items()})

        # Check for sub-goal completion
        if self._option_state.subgoal.check_completion(self._last_obs, info):
            intrinsic = (
                self._option_state.subgoal.intrinsic_reward * self.config.subgoal_reward_scale
            )
            self._option_state.cumulative_reward += intrinsic
            self._option_state.terminated = True
            self._episode_stats["subgoals_completed"] += 1
            self._episode_stats["intrinsic_reward"] += intrinsic

        # Episode end handling
        if terminated or truncated:
            self._finalize_option(self._last_obs, terminated, truncated, info)

        self._episode_stats["total_reward"] += reward
        self._episode_stats["steps"] += 1

        return metrics

    def _should_terminate(self, obs: ObsType, info: dict[str, Any]) -> bool:
        """Check if current option should terminate."""
        logger.debug("HierarchicalController._should_terminate: obs=%s, info=%s", obs, info)
        if self._option_state is None:
            return True

        # Mandatory termination conditions
        if self._option_state.terminated:
            return True
        if self._option_state.steps_taken >= self.config.max_option_steps:
            return True
        if self._option_state.subgoal.check_completion(obs, info):
            return True

        # Learned termination
        if (
            self.config.use_learned_termination
            and self.termination_critic is not None
            and self._option_state.steps_taken >= self.config.min_option_steps
        ):
            prob = self.termination_critic.termination_probability(
                obs, self._option_state.subgoal, self._option_state, info
            )
            if np.random.random() < prob:
                return True

        # Policy-based termination
        terminate, _ = self.low_policy.check_termination(
            obs,
            self._option_state.subgoal,
            self._option_state.steps_taken,
            info,
        )
        return terminate

    def _select_new_subgoal(self, obs: ObsType, info: dict[str, Any]) -> None:
        """Select a new sub-goal using high-level policy."""
        # Finalize previous option if exists
        logger.debug("HierarchicalController._select_new_subgoal: obs=%s, info=%s", obs, info)
        if self._option_state is not None:
            self._finalize_option(obs, False, False, info)

        # Select new sub-goal
        subgoal_id = self.high_policy.select_subgoal(obs, info)
        subgoal = self.registry.get(subgoal_id)

        self._option_state = OptionState(
            subgoal=subgoal,
            start_obs=obs.copy() if isinstance(obs, np.ndarray) else obs,
        )
        self._episode_stats["subgoals_selected"] += 1

    def _finalize_option(
        self,
        obs: ObsType,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Finalize option and update high-level policy."""
        logger.debug("HierarchicalController._finalize_option: obs=%s, terminated=%s, truncated=%s, info=%s", obs, terminated, truncated, info)
        if self._option_state is None or self._option_state.start_obs is None:
            return

        # Update high-level policy
        self.high_policy.update(
            self._option_state.start_obs,
            self._option_state.subgoal.id,
            self._option_state.cumulative_reward,
            obs,
            terminated,
            truncated,
            info,
        )

    def get_episode_stats(self) -> dict[str, float]:
        """Get statistics for current episode."""
        logger.debug("HierarchicalController.get_episode_stats called")
        return dict(self._episode_stats)


# =============================================================================
# SUB-GOAL DETECTION FROM TRAJECTORIES
# =============================================================================


@dataclass
class TrajectorySegment:
    """A segment of trajectory potentially corresponding to a sub-goal.

    Attributes:
        observations: Sequence of observations.
        actions: Sequence of actions taken.
        rewards: Sequence of rewards received.
        start_idx: Start index in original trajectory.
        end_idx: End index in original trajectory.
        detected_subgoal: Automatically detected sub-goal, if any.
        confidence: Confidence score for detection.
    """

    observations: NDArray[np.float32]
    actions: NDArray[np.int32]
    rewards: NDArray[np.float32]
    start_idx: int
    end_idx: int
    detected_subgoal: SubGoalID | None = None
    confidence: float = 0.0


class SubGoalDetector:
    """Detect sub-goals from expert trajectories.

    Uses heuristic and learned methods to segment trajectories
    into sub-goal completion events.
    """

    def __init__(
        self,
        registry: SubGoalRegistry | None = None,
        min_segment_length: int = 20,
        max_segment_length: int = 2400,
    ):
        """Initialize detector.

        Args:
            registry: Sub-goal registry. Uses singleton if None.
            min_segment_length: Minimum steps for a segment.
            max_segment_length: Maximum steps for a segment.
        """
        logger.info("SubGoalDetector.__init__: registry=%s, min_segment_length=%s, max_segment_length=%s", registry, min_segment_length, max_segment_length)
        self.registry = registry or SubGoalRegistry()
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length

        # Item ID to sub-goal mapping for acquisition events
        self._item_subgoal_map: dict[int, SubGoalID] = {
            17: SubGoalID.GET_WOOD,  # log
            4: SubGoalID.GET_STONE,  # cobblestone
            265: SubGoalID.GET_IRON,  # iron_ingot
            264: SubGoalID.GET_DIAMONDS,  # diamond
            49: SubGoalID.GET_OBSIDIAN,  # obsidian
            369: SubGoalID.GET_BLAZE_RODS,  # blaze_rod
            368: SubGoalID.GET_ENDER_PEARLS,  # ender_pearl
            381: SubGoalID.CRAFT_EYES_OF_ENDER,  # eye_of_ender
        }

        # Dimension transitions
        self._dimension_subgoal_map: dict[tuple[int, int], SubGoalID] = {
            (0, 1): SubGoalID.ENTER_NETHER,  # Overworld -> Nether
            (1, 0): SubGoalID.RETURN_TO_OVERWORLD,  # Nether -> Overworld
            (0, 2): SubGoalID.ENTER_END,  # Overworld -> End
        }

    def detect_from_trajectory(
        self,
        observations: NDArray[np.float32],
        actions: NDArray[np.int32],
        rewards: NDArray[np.float32],
        infos: list[dict[str, Any]],
    ) -> list[TrajectorySegment]:
        """Detect sub-goal segments from a trajectory.

        Args:
            observations: Array of observations, shape (T, obs_dim).
            actions: Array of actions, shape (T,).
            rewards: Array of rewards, shape (T,).
            infos: List of info dicts from environment.

        Returns:
            List of detected trajectory segments with sub-goal labels.
        """
        logger.debug("SubGoalDetector.detect_from_trajectory: observations=%s, actions=%s, rewards=%s, infos=%s", observations, actions, rewards, infos)
        segments: list[TrajectorySegment] = []
        total_steps = len(observations)

        # Detect item acquisition events
        item_events = self._detect_item_acquisitions(infos)

        # Detect dimension changes
        dimension_events = self._detect_dimension_changes(observations)

        # Detect combat events
        combat_events = self._detect_combat_events(infos)

        # Merge and sort all events
        all_events = item_events + dimension_events + combat_events
        all_events.sort(key=lambda x: x[0])  # Sort by timestep

        # Create segments between events
        prev_idx = 0
        for event_idx, subgoal_id, confidence in all_events:
            if event_idx - prev_idx >= self.min_segment_length:
                segment = TrajectorySegment(
                    observations=observations[prev_idx:event_idx],
                    actions=actions[prev_idx:event_idx],
                    rewards=rewards[prev_idx:event_idx],
                    start_idx=prev_idx,
                    end_idx=event_idx,
                    detected_subgoal=subgoal_id,
                    confidence=confidence,
                )
                segments.append(segment)
            prev_idx = event_idx

        # Handle remaining trajectory
        if total_steps - prev_idx >= self.min_segment_length:
            segment = TrajectorySegment(
                observations=observations[prev_idx:total_steps],
                actions=actions[prev_idx:total_steps],
                rewards=rewards[prev_idx:total_steps],
                start_idx=prev_idx,
                end_idx=total_steps,
                detected_subgoal=SubGoalID.EXPLORE,  # Default
                confidence=0.3,
            )
            segments.append(segment)

        return segments

    def _detect_item_acquisitions(
        self, infos: list[dict[str, Any]]
    ) -> list[tuple[int, SubGoalID, float]]:
        """Detect item acquisition events from info dicts."""
        logger.debug("SubGoalDetector._detect_item_acquisitions: infos=%s", infos)
        events: list[tuple[int, SubGoalID, float]] = []

        prev_inventory: dict[int, int] = {}
        for t, info in enumerate(infos):
            inventory = info.get("inventory", {})
            if isinstance(inventory, dict):
                for item_id, count in inventory.items():
                    item_id = int(item_id)
                    prev_count = prev_inventory.get(item_id, 0)
                    if count > prev_count and item_id in self._item_subgoal_map:
                        events.append((t, self._item_subgoal_map[item_id], 0.9))
                prev_inventory = inventory.copy()

        return events

    def _detect_dimension_changes(
        self, observations: NDArray[np.float32]
    ) -> list[tuple[int, SubGoalID, float]]:
        """Detect dimension change events from observations."""
        logger.debug("SubGoalDetector._detect_dimension_changes: observations=%s", observations)
        events: list[tuple[int, SubGoalID, float]] = []

        # Assuming dimension is at index 14 (from observations.py)
        dim_idx = 14
        if observations.shape[1] > dim_idx:
            dims = observations[:, dim_idx]
            for t in range(1, len(dims)):
                prev_dim = int(dims[t - 1] * 2)  # Unnormalize
                curr_dim = int(dims[t] * 2)
                key = (prev_dim, curr_dim)
                if key in self._dimension_subgoal_map:
                    events.append((t, self._dimension_subgoal_map[key], 0.95))

        return events

    def _detect_combat_events(
        self, infos: list[dict[str, Any]]
    ) -> list[tuple[int, SubGoalID, float]]:
        """Detect combat events (kills) from info dicts."""
        logger.debug("SubGoalDetector._detect_combat_events: infos=%s", infos)
        events: list[tuple[int, SubGoalID, float]] = []

        mob_subgoal_map: dict[str, SubGoalID] = {
            "zombie": SubGoalID.KILL_ZOMBIES,
            "skeleton": SubGoalID.KILL_SKELETONS,
            "creeper": SubGoalID.KILL_CREEPERS,
            "blaze": SubGoalID.KILL_BLAZES,
            "enderman": SubGoalID.KILL_ENDERMEN,
            "ender_dragon": SubGoalID.DEFEAT_DRAGON,
        }

        for t, info in enumerate(infos):
            killed = info.get("mob_killed", info.get("entity_killed"))
            if killed and killed in mob_subgoal_map:
                events.append((t, mob_subgoal_map[killed], 0.85))

            # Dragon crystal destruction
            if info.get("crystal_destroyed"):
                events.append((t, SubGoalID.DESTROY_CRYSTALS, 0.9))

        return events

    def save_detections(
        self,
        segments: list[TrajectorySegment],
        path: Path | str,
    ) -> None:
        """Save detected segments to JSON file."""
        logger.debug("SubGoalDetector.save_detections: segments=%s, path=%s", segments, path)
        path = Path(path)
        data = [
            {
                "start_idx": seg.start_idx,
                "end_idx": seg.end_idx,
                "detected_subgoal": seg.detected_subgoal.name if seg.detected_subgoal else None,
                "confidence": seg.confidence,
                "length": seg.end_idx - seg.start_idx,
            }
            for seg in segments
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# =============================================================================
# REWARD SHAPING FOR HIERARCHICAL RL
# =============================================================================


class HierarchicalRewardShaper:
    """Reward shaping for option-critic style training.

    Provides intrinsic motivation, subgoal bonuses, and
    temporal abstraction rewards.
    """

    def __init__(
        self,
        registry: SubGoalRegistry | None = None,
        intrinsic_scale: float = 1.0,
        curiosity_scale: float = 0.1,
        progress_scale: float = 0.5,
    ):
        """Initialize reward shaper.

        Args:
            registry: Sub-goal registry.
            intrinsic_scale: Scale for intrinsic rewards.
            curiosity_scale: Scale for curiosity/exploration bonus.
            progress_scale: Scale for progress-toward-goal rewards.
        """
        logger.info("HierarchicalRewardShaper.__init__: registry=%s, intrinsic_scale=%s, curiosity_scale=%s, progress_scale=%s", registry, intrinsic_scale, curiosity_scale, progress_scale)
        self.registry = registry or SubGoalRegistry()
        self.intrinsic_scale = intrinsic_scale
        self.curiosity_scale = curiosity_scale
        self.progress_scale = progress_scale

        self._state_counts: dict[int, int] = defaultdict(int)

    def shape_reward(
        self,
        obs: NDArray[np.float32],
        action: int,
        reward: float,
        next_obs: NDArray[np.float32],
        subgoal: SubGoal,
        info: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """Compute shaped reward with intrinsic motivation.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Extrinsic environment reward.
            next_obs: Next observation.
            subgoal: Active sub-goal.
            info: Additional info.

        Returns:
            Tuple of (shaped_reward, reward_components).
        """
        logger.debug("HierarchicalRewardShaper.shape_reward: obs=%s, action=%s, reward=%s, next_obs=%s", obs, action, reward, next_obs)
        components: dict[str, float] = {"extrinsic": reward}
        total = reward

        # Sub-goal completion bonus
        if subgoal.check_completion(next_obs, info):
            intrinsic = subgoal.intrinsic_reward * self.intrinsic_scale
            total += intrinsic
            components["subgoal_bonus"] = intrinsic

        # Curiosity / count-based exploration
        state_hash = self._hash_state(next_obs)
        visit_count = self._state_counts[state_hash]
        self._state_counts[state_hash] += 1
        curiosity = self.curiosity_scale / np.sqrt(visit_count + 1)
        total += curiosity
        components["curiosity"] = curiosity

        # Progress toward sub-goal (task-specific)
        progress = self._compute_progress(obs, next_obs, subgoal, info)
        if progress != 0:
            progress_reward = progress * self.progress_scale
            total += progress_reward
            components["progress"] = progress_reward

        return total, components

    def _hash_state(self, obs: NDArray[np.float32]) -> int:
        """Hash observation for count-based exploration."""
        # Discretize to reduce state space
        logger.debug("HierarchicalRewardShaper._hash_state: obs=%s", obs)
        discretized = (obs * 10).astype(np.int32)
        return hash(discretized.tobytes())

    def _compute_progress(
        self,
        obs: NDArray[np.float32],
        next_obs: NDArray[np.float32],
        subgoal: SubGoal,
        info: dict[str, Any],
    ) -> float:
        """Compute progress toward sub-goal.

        Returns value in [-1, 1] indicating backward/forward progress.
        """
        # Task-specific progress computation
        logger.debug("HierarchicalRewardShaper._compute_progress: obs=%s, next_obs=%s, subgoal=%s, info=%s", obs, next_obs, subgoal, info)
        if subgoal.category == SubGoalCategory.NAVIGATION:
            # Check if getting closer to goal location
            goal_pos = info.get("goal_position")
            if goal_pos is not None:
                pos_idx = slice(0, 3)  # x, y, z
                dist_before = np.linalg.norm(obs[pos_idx] - goal_pos)
                dist_after = np.linalg.norm(next_obs[pos_idx] - goal_pos)
                return float((dist_before - dist_after) / max(dist_before, 1.0))

        if subgoal.category == SubGoalCategory.COMBAT:
            # Check damage dealt to target
            damage = info.get("damage_dealt", 0)
            if damage > 0:
                return min(damage / 10.0, 1.0)

        if subgoal.category == SubGoalCategory.RESOURCE:
            # Check inventory increase
            item_gained = info.get("items_gained", 0)
            if item_gained > 0:
                return min(item_gained / 5.0, 1.0)

        return 0.0

    def reset(self) -> None:
        """Reset episode-specific state."""
        logger.debug("HierarchicalRewardShaper.reset called")
        self._state_counts.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_hierarchical_env_wrapper(
    env: Any,
    high_policy: HighLevelPolicy,
    low_policy: LowLevelPolicy,
    config: HierarchicalConfig | None = None,
) -> Callable:
    """Create a wrapped environment with hierarchical control.

    Args:
        env: Base Gymnasium environment.
        high_policy: High-level policy for sub-goal selection.
        low_policy: Low-level policy for action selection.
        config: Hierarchical RL configuration.

    Returns:
        Environment step function with hierarchical control.
    """
    logger.info("create_hierarchical_env_wrapper: env=%s, high_policy=%s, low_policy=%s, config=%s", env, high_policy, low_policy, config)
    controller = HierarchicalController(high_policy, low_policy, config)
    reward_shaper = HierarchicalRewardShaper()

    def step(obs: NDArray[np.float32], info: dict[str, Any]) -> tuple:
        logger.debug("step: obs=%s, info=%s", obs, info)
        action = controller.step(obs, info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        if controller.current_subgoal:
            shaped_reward, _ = reward_shaper.shape_reward(
                obs, action, reward, next_obs, controller.current_subgoal, next_info
            )
        else:
            shaped_reward = reward

        controller.receive_reward(shaped_reward, terminated, truncated, next_info)
        return next_obs, shaped_reward, terminated, truncated, next_info

    return step


def get_subgoal_embedding(
    subgoal_id: SubGoalID,
    embedding_dim: int = 32,
) -> NDArray[np.float32]:
    """Get learnable embedding vector for a sub-goal.

    Args:
        subgoal_id: Sub-goal identifier.
        embedding_dim: Dimension of embedding.

    Returns:
        Embedding vector of shape (embedding_dim,).
    """
    # One-hot with some structure based on category
    logger.debug("get_subgoal_embedding: subgoal_id=%s, embedding_dim=%s", subgoal_id, embedding_dim)
    registry = SubGoalRegistry()
    subgoal = registry.get(subgoal_id)

    embedding = np.zeros(embedding_dim, dtype=np.float32)

    # Category encoding (first 8 dims)
    embedding[subgoal.category.value] = 1.0

    # ID encoding (next dims)
    id_dim = min(subgoal_id.value, embedding_dim - 8)
    if 8 + id_dim < embedding_dim:
        embedding[8 + id_dim] = 1.0

    return embedding
