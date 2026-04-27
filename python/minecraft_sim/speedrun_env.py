"""Gymnasium environment for full Free The End speedrun with curriculum learning.

This module provides a unified environment that supports all 6 curriculum stages
for training an agent to complete a Minecraft 1.8.9 speedrun from spawn to
defeating the Ender Dragon.

The environment uses an extended observation space (256 floats) and action space
(32 discrete actions) to cover all mechanics needed across stages, from basic
survival to the final dragon fight.

Example:
    >>> from minecraft_sim.speedrun_env import SpeedrunEnv
    >>> env = SpeedrunEnv(stage_id=1)  # Start with basic survival
    >>> obs, info = env.reset()
    >>> for _ in range(1000):
    ...     action = env.action_space.sample()
    ...     obs, reward, terminated, truncated, info = env.step(action)
    ...     if info.get("stage_advanced"):
    ...         print(f"Advanced to stage {info['current_stage']}")
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("gymnasium is required for speedrun_env") from exc

from .curriculum import (
    CurriculumManager,
    Stage,
    StageID,
)

# =============================================================================
# Dimension Enum
# =============================================================================


class Dimension(IntEnum):
    """Minecraft dimension identifiers.

    Maps to the observation vector encoding at index 127:
    0.0 = Overworld, 0.5 = Nether, 1.0 = End.
    """

    OVERWORLD = 0
    NETHER = 1
    END = 2

    @classmethod
    def from_obs_value(cls, value: float) -> Dimension:
        """Decode dimension from the normalized observation value.

        Args:
            value: Observation value at index 127 (0.0, 0.5, or 1.0).

        Returns:
            Corresponding Dimension enum member.
        """
        if value >= 0.75:
            return cls.END
        if value >= 0.25:
            return cls.NETHER
        return cls.OVERWORLD

    def to_obs_value(self) -> float:
        """Encode dimension to the normalized observation value.

        Returns:
            Float suitable for observation index 127.
        """
        return {
            Dimension.OVERWORLD: 0.0,
            Dimension.NETHER: 0.5,
            Dimension.END: 1.0,
        }[self]


# =============================================================================
# Dimension Transition State
# =============================================================================


@dataclass
class DimensionState:
    """Tracks state preservation during dimension transitions.

    Captures player state that must persist across dimension changes
    (position per dimension, inventory, cooldowns) and records the
    transition history for reward shaping.
    """

    current: Dimension = Dimension.OVERWORLD
    previous: Dimension = Dimension.OVERWORLD
    transition_tick: int = 0
    portal_cooldown_ticks: int = 0

    # Position snapshots per dimension (preserved for return trips)
    overworld_position: tuple[float, float, float] = (0.0, 64.0, 0.0)
    nether_position: tuple[float, float, float] = (0.0, 64.0, 0.0)
    end_position: tuple[float, float, float] = (0.0, 64.0, 0.0)

    # Transition history: (tick, from_dim, to_dim)
    transitions: list[tuple[int, Dimension, Dimension]] = field(default_factory=list)

    def record_transition(
        self, tick: int, from_dim: Dimension, to_dim: Dimension
    ) -> None:
        """Record a dimension transition event.

        Args:
            tick: Game tick when the transition occurred.
            from_dim: Source dimension.
            to_dim: Destination dimension.
        """
        self.previous = from_dim
        self.current = to_dim
        self.transition_tick = tick
        self.portal_cooldown_ticks = 80  # 4 seconds at 20 tps
        self.transitions.append((tick, from_dim, to_dim))

    def save_position(self, dim: Dimension, pos: tuple[float, float, float]) -> None:
        """Save position for a dimension (for return trips).

        Args:
            dim: Dimension to save position for.
            pos: (x, y, z) position tuple.
        """
        if dim == Dimension.OVERWORLD:
            self.overworld_position = pos
        elif dim == Dimension.NETHER:
            self.nether_position = pos
        elif dim == Dimension.END:
            self.end_position = pos

    def get_saved_position(self, dim: Dimension) -> tuple[float, float, float]:
        """Get the saved position for a dimension.

        Args:
            dim: Dimension to retrieve position for.

        Returns:
            (x, y, z) position tuple.
        """
        if dim == Dimension.OVERWORLD:
            return self.overworld_position
        if dim == Dimension.NETHER:
            return self.nether_position
        return self.end_position

    def tick(self) -> None:
        """Advance cooldown timers by one tick."""
        if self.portal_cooldown_ticks > 0:
            self.portal_cooldown_ticks -= 1

    @property
    def in_cooldown(self) -> bool:
        """Whether the player is in portal cooldown."""
        return self.portal_cooldown_ticks > 0


# =============================================================================
# Observation Space Layout (256 floats)
# =============================================================================


@dataclass(frozen=True)
class ObservationLayout:
    """Defines the layout of the 256-float observation vector.

    The observation space is divided into semantic regions to make it easier
    to understand what information the agent receives at each timestep.

    All values are normalized to [0, 1] or [-1, 1] depending on the field.
    """

    # Player state [0-31]: 32 floats
    PLAYER_START: int = 0
    PLAYER_END: int = 32
    # [0-2]: position x, y, z (normalized to world bounds)
    # [3-5]: velocity x, y, z (normalized to max speed)
    # [6-7]: yaw, pitch (normalized to [-1, 1])
    # [8]: health (0-1, maps to 0-20 HP)
    # [9]: hunger (0-1, maps to 0-20)
    # [10]: saturation (0-1, maps to 0-20)
    # [11-12]: armor points, armor toughness (0-1)
    # [13]: experience level (0-1, maps to 0-30)
    # [14]: experience progress (0-1)
    # [15]: on_ground flag (0 or 1)
    # [16]: in_water flag (0 or 1)
    # [17]: in_lava flag (0 or 1)
    # [18]: sprinting flag (0 or 1)
    # [19]: sneaking flag (0 or 1)
    # [20-23]: active potion effects (speed, strength, fire_resistance, regen)
    # [24-27]: effect durations (normalized)
    # [28]: fall distance (normalized)
    # [29]: air supply (0-1, maps to 0-300 ticks)
    # [30]: fire ticks (0-1, maps to 0-300)
    # [31]: hurt_time (0-1, recent damage indicator)

    # Inventory state [32-63]: 32 floats
    INVENTORY_START: int = 32
    INVENTORY_END: int = 64
    # Key items for speedrun (count normalized to max stack):
    # [32]: wood_logs
    # [33]: planks
    # [34]: sticks
    # [35]: cobblestone
    # [36]: iron_ingot
    # [37]: gold_ingot
    # [38]: diamond
    # [39]: obsidian
    # [40]: blaze_rod
    # [41]: ender_pearl
    # [42]: eye_of_ender
    # [43]: flint
    # [44]: gravel
    # [45]: food_count (any food)
    # [46]: has_crafting_table
    # [47]: has_furnace
    # [48]: has_wooden_pickaxe
    # [49]: has_stone_pickaxe
    # [50]: has_iron_pickaxe
    # [51]: has_diamond_pickaxe
    # [52]: has_sword (any)
    # [53]: sword_material (0=none, 0.25=wood, 0.5=stone, 0.75=iron, 1.0=diamond)
    # [54]: has_bow
    # [55]: arrow_count (normalized)
    # [56]: has_shield
    # [57]: has_bed
    # [58]: has_bucket
    # [59]: bucket_type (0=empty, 0.5=water, 1.0=lava)
    # [60]: hotbar_slot (0-1, maps to 0-8)
    # [61]: has_flint_and_steel
    # [62]: armor_equipped (0-1, count of pieces)
    # [63]: total_inventory_slots_used (0-1)

    # Local environment [64-127]: 64 floats
    ENVIRONMENT_START: int = 64
    ENVIRONMENT_END: int = 128
    # [64-79]: 4x4 block type grid at feet level (flattened)
    # [80-95]: 4x4 block type grid at head level
    # [96-103]: 8 nearest entity types (one-hot encoded major types)
    # [104-111]: 8 nearest entity distances (normalized)
    # [112-119]: 8 nearest entity angles (normalized yaw)
    # [120]: nearest_hostile_distance
    # [121]: nearest_hostile_type (encoded)
    # [122]: nearest_item_distance
    # [123]: nearest_item_type
    # [124]: biome_type (encoded)
    # [125]: light_level (0-1, maps to 0-15)
    # [126]: time_of_day (0-1, maps to 0-24000)
    # [127]: dimension (0=overworld, 0.5=nether, 1.0=end)

    # Goal-specific state [128-191]: 64 floats
    GOAL_START: int = 128
    GOAL_END: int = 192
    # Stage 1 (Basic Survival):
    # [128]: zombies_killed (normalized to objective)
    # [129]: skeletons_killed
    # [130]: wood_mined
    # [131]: survival_time (normalized to objective)
    # Stage 2 (Resource Gathering):
    # [132]: iron_obtained
    # [133]: diamonds_obtained
    # [134]: has_nether_portal
    # Stage 3 (Nether Navigation):
    # [135]: in_nether flag
    # [136]: fortress_found flag
    # [137]: blaze_kills
    # [138]: distance_to_fortress (normalized)
    # Stage 4 (Enderman Hunting):
    # [139]: pearls_obtained
    # [140]: eyes_crafted
    # [141]: enderman_nearby flag
    # Stage 5 (Stronghold Finding):
    # [142]: eye_thrown flag
    # [143]: stronghold_found flag
    # [144]: distance_to_stronghold (normalized)
    # [145]: portal_frame_filled (0-1, maps to 0-12)
    # [146-159]: reserved for additional goal tracking
    # [160-175]: relative direction to current objective (16 cardinal directions)
    # [176-191]: distance buckets to various objectives

    # Dragon state [192-223]: 32 floats (primarily stage 6)
    DRAGON_START: int = 192
    DRAGON_END: int = 224
    # [192]: dragon_health (0-1, maps to 0-200)
    # [193-195]: dragon_position (relative to player, normalized)
    # [196-198]: dragon_velocity (normalized)
    # [199]: dragon_phase (0-1, maps to 0-6)
    # [200]: dragon_target_is_player flag
    # [201]: dragon_distance (normalized)
    # [202]: dragon_angle (yaw to dragon, normalized)
    # [203]: dragon_pitch (pitch to dragon, normalized)
    # [204]: dragon_perching flag
    # [205]: dragon_charging flag
    # [206]: dragon_breath_active flag
    # [207]: can_hit_dragon flag
    # [208]: crystals_remaining (0-1, maps to 0-10)
    # [209-218]: crystal_destroyed flags (10 crystals)
    # [219]: nearest_crystal_distance
    # [220]: nearest_crystal_angle
    # [221]: on_obsidian_pillar flag
    # [222]: exit_portal_active flag
    # [223]: dragon_fight_complete flag

    # Dimension/portal state [224-255]: 32 floats
    PORTAL_START: int = 224
    PORTAL_END: int = 256
    # [224]: near_nether_portal flag
    # [225]: portal_distance (normalized)
    # [226]: portal_alignment (how aligned with portal frame)
    # [227]: in_portal_cooldown flag
    # [228]: near_end_portal flag
    # [229]: end_portal_activated flag
    # [230]: void_below flag (End dimension)
    # [231]: void_distance (normalized)
    # [232-239]: last 8 actions taken (action history)
    # [240-247]: reward history (last 8 rewards, normalized)
    # [248]: episode_progress (0-1, time spent / max time)
    # [249]: stage_progress (0-1, objectives completed)
    # [250]: deaths_this_episode (normalized)
    # [251]: total_reward_this_episode (normalized)
    # [252]: current_stage_id (0-1, maps to 1-6)
    # [253]: is_terminal_state flag
    # [254]: success_flag
    # [255]: reserved


# =============================================================================
# Action Space (32 discrete actions)
# =============================================================================


class SpeedrunAction:
    """Enumeration of all available actions in the speedrun environment.

    The action space is designed to cover all mechanics needed for a full
    speedrun while remaining tractable for RL training.
    """

    NOOP = 0
    FORWARD = 1
    BACK = 2
    LEFT = 3
    RIGHT = 4
    FORWARD_LEFT = 5
    FORWARD_RIGHT = 6
    JUMP = 7
    JUMP_FORWARD = 8
    ATTACK = 9
    ATTACK_FORWARD = 10
    SPRINT_TOGGLE = 11
    LOOK_LEFT = 12  # 5 degrees
    LOOK_RIGHT = 13  # 5 degrees
    LOOK_UP = 14  # 7.5 degrees
    LOOK_DOWN = 15  # 7.5 degrees
    LOOK_LEFT_FAST = 16  # 45 degrees
    LOOK_RIGHT_FAST = 17  # 45 degrees
    LOOK_UP_FAST = 18  # 30 degrees
    LOOK_DOWN_FAST = 19  # 30 degrees
    USE_ITEM = 20  # right click (place, eat, shoot, use portal)
    DROP_ITEM = 21
    HOTBAR_1 = 22
    HOTBAR_2 = 23
    HOTBAR_3 = 24
    HOTBAR_4 = 25
    HOTBAR_5 = 26
    HOTBAR_6 = 27
    HOTBAR_7 = 28
    HOTBAR_8 = 29
    HOTBAR_9 = 30
    CRAFT = 31  # context-sensitive quick craft

    NUM_ACTIONS = 32

    # Action names for debugging
    NAMES = [
        "noop",
        "forward",
        "back",
        "left",
        "right",
        "forward_left",
        "forward_right",
        "jump",
        "jump_forward",
        "attack",
        "attack_forward",
        "sprint_toggle",
        "look_left",
        "look_right",
        "look_up",
        "look_down",
        "look_left_fast",
        "look_right_fast",
        "look_up_fast",
        "look_down_fast",
        "use_item",
        "drop_item",
        "hotbar_1",
        "hotbar_2",
        "hotbar_3",
        "hotbar_4",
        "hotbar_5",
        "hotbar_6",
        "hotbar_7",
        "hotbar_8",
        "hotbar_9",
        "craft",
    ]


# =============================================================================
# Episode Statistics
# =============================================================================


@dataclass
class EpisodeStats:
    """Tracks statistics for a single episode."""

    stage_id: int
    steps: int = 0
    total_reward: float = 0.0
    deaths: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    items_collected: int = 0
    blocks_mined: int = 0
    blocks_placed: int = 0
    mobs_killed: int = 0
    distance_traveled: float = 0.0

    # Stage-specific tracking
    objectives_completed: list[str] = field(default_factory=list)

    # Stage 2 resource progress (peak counts observed during episode)
    iron_count: int = 0
    diamond_count: int = 0
    obsidian_count: int = 0

    # Timing
    real_time_seconds: float = 0.0
    game_ticks: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "stage_id": self.stage_id,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "deaths": self.deaths,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "items_collected": self.items_collected,
            "blocks_mined": self.blocks_mined,
            "blocks_placed": self.blocks_placed,
            "mobs_killed": self.mobs_killed,
            "distance_traveled": self.distance_traveled,
            "objectives_completed": self.objectives_completed,
            "stage2_progress": {
                "iron_count": self.iron_count,
                "diamond_count": self.diamond_count,
                "obsidian_count": self.obsidian_count,
            },
            "real_time_seconds": self.real_time_seconds,
            "game_ticks": self.game_ticks,
        }


# =============================================================================
# Main Environment Class
# =============================================================================


class SpeedrunEnv(gym.Env):
    """Gymnasium environment for the full Minecraft Free The End speedrun.

    This environment supports all 6 curriculum stages, from basic survival to
    defeating the Ender Dragon. It uses an extended observation space (256 floats)
    and action space (32 discrete actions) to handle all speedrun mechanics.

    The environment can be initialized at any stage and optionally supports
    automatic stage progression when the curriculum threshold is met.

    Attributes:
        stage_id: Current curriculum stage (1-6).
        observation_space: Box space with 256 normalized floats.
        action_space: Discrete space with 32 actions.
        curriculum_manager: Manages stage transitions and progress tracking.
        auto_advance: Whether to automatically advance stages.

    Example:
        >>> env = SpeedrunEnv(stage_id=1, auto_advance=True)
        >>> obs, info = env.reset()
        >>>
        >>> for episode in range(1000):
        ...     done = False
        ...     while not done:
        ...         action = agent.select_action(obs)
        ...         obs, reward, terminated, truncated, info = env.step(action)
        ...         done = terminated or truncated
        ...
        ...     obs, info = env.reset()
        ...     if info.get("stage_advanced"):
        ...         print(f"Now training on stage {env.stage_id}")
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Observation and action dimensions
    OBS_DIM = 256
    ACT_DIM = SpeedrunAction.NUM_ACTIONS

    def __init__(
        self,
        stage_id: int = 1,
        shader_dir: str | None = None,
        auto_advance: bool = True,
        curriculum_threshold: float | None = None,
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the speedrun environment.

        Args:
            stage_id: Initial curriculum stage (1-6). Defaults to 1 (Basic Survival).
            shader_dir: Path to shader directory for the simulator. If None, uses
                default location relative to package.
            auto_advance: If True, automatically advance to next stage when
                curriculum_threshold is met. Defaults to True.
            curriculum_threshold: Success rate required to advance stages. If None,
                uses the threshold defined in stage config.
            max_episode_steps: Maximum steps per episode. If None, uses stage config.
            render_mode: Rendering mode ("human" or "rgb_array").
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If stage_id is not in range 1-6.
        """
        super().__init__()

        if not 1 <= stage_id <= 6:
            raise ValueError(f"stage_id must be 1-6, got {stage_id}")

        self._stage_id = stage_id
        self._auto_advance = auto_advance
        self._custom_threshold = curriculum_threshold
        self._custom_max_steps = max_episode_steps
        self._render_mode = render_mode
        self._rng = np.random.default_rng(seed)

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.ACT_DIM)

        # Initialize curriculum manager
        self._curriculum = CurriculumManager()
        self._current_stage: Stage | None = None

        # Resolve shader directory
        if shader_dir is None:
            shader_dir = str(Path(__file__).resolve().parents[2] / "cpp" / "shaders")
        self._shader_dir = shader_dir

        # Initialize simulator (deferred until reset)
        self._sim = None
        self._sim_initialized = False

        # Episode state
        self._step_count = 0
        self._max_steps = 0
        self._episode_reward = 0.0
        self._episode_stats = EpisodeStats(stage_id=stage_id)

        # Observation buffer
        self._obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Action history for observation
        self._action_history: deque[int] = deque(maxlen=8)
        self._reward_history: deque[float] = deque(maxlen=8)

        # Curriculum tracking
        self._recent_episodes: deque[bool] = deque(maxlen=100)  # success/failure

        # Milestone tracking (one-time bonus rewards per episode)
        self._milestones_achieved: set[str] = set()

        # Persistent inventory state across stage transitions.
        # Keys map to observation indices 32-63 (see ObservationLayout.INVENTORY_*).
        # Values are normalized floats [0, 1] representing item counts/flags.
        self._inventory_state: dict[str, float] = {}

        # Dimension transition state management.
        # Tracks current dimension, cooldowns, per-dimension position snapshots,
        # and transition history across the episode.
        self._dimension_state = DimensionState()

        # Load stage configuration
        self._load_stage(stage_id)

    @property
    def stage_id(self) -> int:
        """Current curriculum stage ID (1-6)."""
        return self._stage_id

    @property
    def current_stage(self) -> Stage | None:
        """Current stage configuration."""
        return self._current_stage

    @property
    def curriculum_progress(self) -> dict[str, Any]:
        """Get curriculum progress summary."""
        return self._curriculum.get_training_summary()

    # Observation indices for key inventory items that persist across stages.
    _PERSISTENT_INVENTORY_KEYS: dict[str, int] = {
        "wood_logs": 32,
        "cobblestone": 35,
        "iron_ingot": 36,
        "diamond": 38,
        "obsidian": 39,
        "blaze_rod": 40,
        "ender_pearl": 41,
        "eye_of_ender": 42,
        "food_count": 45,
        "has_stone_pickaxe": 49,
        "has_iron_pickaxe": 50,
        "has_diamond_pickaxe": 51,
        "has_sword": 52,
        "sword_material": 53,
        "has_bow": 54,
        "arrow_count": 55,
        "has_shield": 56,
        "has_bucket": 58,
        "bucket_type": 59,
        "has_flint_and_steel": 61,
        "armor_equipped": 62,
    }

    def _serialize_inventory(self) -> dict[str, float]:
        """Capture current inventory state from the observation buffer.

        Returns:
            Dictionary mapping item names to their normalized observation values.
        """
        state: dict[str, float] = {}
        for name, idx in self._PERSISTENT_INVENTORY_KEYS.items():
            val = float(self._obs[idx])
            if val > 0.0:
                state[name] = val
        return state

    def _restore_inventory(self) -> None:
        """Restore persisted inventory into the observation buffer.

        Called after reset/spawn to inject items carried from prior stage.
        Only writes non-zero values so spawn-granted items are not overwritten
        unless the persisted value is higher.
        """
        for name, idx in self._PERSISTENT_INVENTORY_KEYS.items():
            if name in self._inventory_state:
                self._obs[idx] = max(self._obs[idx], self._inventory_state[name])

    def _load_stage(self, stage_id: int) -> None:
        """Load stage configuration and update environment settings.

        Args:
            stage_id: Stage to load (1-6).
        """
        self._stage_id = stage_id
        stage_enum = StageID(stage_id)

        try:
            self._current_stage = self._curriculum.get_stage(stage_enum)
        except KeyError:
            # Create minimal stage config if not found
            self._current_stage = None

        # Set max steps from stage config or override
        if self._custom_max_steps is not None:
            self._max_steps = self._custom_max_steps
        elif self._current_stage is not None:
            self._max_steps = self._current_stage.termination.max_ticks
        else:
            # Default max steps per stage
            defaults = {
                1: 24000,  # 20 minutes
                2: 36000,  # 30 minutes
                3: 36000,  # 30 minutes
                4: 24000,  # 20 minutes
                5: 36000,  # 30 minutes
                6: 36000,  # 30 minutes
            }
            self._max_steps = defaults.get(stage_id, 36000)

    def _init_simulator(self) -> None:
        """Initialize the MC189Simulator with stage-appropriate settings."""
        if self._sim_initialized:
            return

        try:
            import mc189_core

            config = mc189_core.SimulatorConfig()
            config.num_envs = 1
            config.shader_dir = self._shader_dir

            # Load stage-specific shader if available (set via shader_dir)
            stage_shader = Path(self._shader_dir) / f"stage_{self._stage_id}.spv"
            if stage_shader.exists():
                # Stage-specific shader available
                pass  # shader_dir already points to correct location

            self._sim = mc189_core.MC189Simulator(config)
            self._sim_initialized = True

        except (ImportError, AttributeError, Exception):
            # Simulator not available or config mismatch; use mock mode
            self._sim = None
            self._sim_initialized = True

    def _spawn_for_stage(self) -> None:
        """Configure spawn conditions based on current stage."""
        if self._sim is None or self._current_stage is None:
            return

        _spawn = self._current_stage.spawn  # noqa: F841 - reserved for implementation

        # In a real implementation, this would configure the simulator
        # with the spawn position, inventory, biome, etc.
        # For now, we document what should happen:
        #
        # - Set player position to spawn.position or random in spawn.biome
        # - Set health/hunger from spawn config
        # - Populate inventory from spawn.inventory
        # - Set time of day from spawn.time_of_day
        # - Set weather from spawn.weather

    def _compute_observation(self) -> np.ndarray:
        """Compute the full 256-float observation vector.

        Returns:
            Normalized observation array of shape (256,).
        """
        obs = self._obs
        obs.fill(0.0)

        if self._sim is None:
            # Mock observation for testing without simulator
            obs[248] = min(1.0, self._step_count / max(1, self._max_steps))
            obs[252] = (self._stage_id - 1) / 5.0  # Normalize stage to [0, 1]
            return obs

        # Get raw observation from simulator
        try:
            raw_obs = np.array(self._sim.get_observations(), dtype=np.float32).flatten()
        except Exception:
            return obs

        # Map simulator's 48-float observation to our 256-float layout
        # The simulator provides: player state, dragon state, basic inventory

        # Player state [0-31]
        if len(raw_obs) >= 11:
            obs[0:3] = raw_obs[0:3]  # position
            obs[3:6] = raw_obs[3:6]  # velocity
            obs[6:8] = raw_obs[6:8]  # yaw, pitch
            obs[8] = raw_obs[8]  # health
            obs[10] = raw_obs[10]  # on_ground

        # Stage 4 goal-specific observations [139-145]
        if self._stage_id == 4 and len(raw_obs) >= 48:
            obs[139] = min(1.0, raw_obs[33] / 12.0) if len(raw_obs) > 33 else 0.0  # pearls (norm to 12)
            obs[140] = min(1.0, raw_obs[34] / 12.0) if len(raw_obs) > 34 else 0.0  # eyes crafted
            obs[141] = raw_obs[35] if len(raw_obs) > 35 else 0.0  # enderman_nearby
            obs[142] = raw_obs[36] if len(raw_obs) > 36 else 0.0  # eye_thrown
            obs[143] = raw_obs[37] if len(raw_obs) > 37 else 0.0  # stronghold_found
            obs[144] = raw_obs[38] if len(raw_obs) > 38 else 0.0  # distance_to_stronghold
            obs[145] = min(1.0, raw_obs[39] / 12.0) if len(raw_obs) > 39 else 0.0  # portal frames filled

        # Dragon state [192-223] (from simulator's dragon fields)
        if len(raw_obs) >= 32:
            obs[192] = raw_obs[16]  # dragon_health
            obs[193:196] = raw_obs[17:20]  # dragon position
            obs[199] = raw_obs[24]  # dragon_phase
            obs[201] = raw_obs[25]  # dragon_distance
            obs[207] = raw_obs[28]  # can_hit
            obs[208] = raw_obs[32] / 10.0 if len(raw_obs) > 32 else 0.0  # crystals

        # Action history [232-239]
        for i, action in enumerate(self._action_history):
            obs[232 + i] = action / (self.ACT_DIM - 1)

        # Reward history [240-247]
        for i, reward in enumerate(self._reward_history):
            obs[240 + i] = np.clip(reward / 10.0 + 0.5, 0.0, 1.0)

        # Episode/stage state [248-255]
        obs[248] = min(1.0, self._step_count / max(1, self._max_steps))
        obs[249] = self._compute_stage_progress()
        obs[250] = min(1.0, self._episode_stats.deaths / 10.0)
        obs[251] = np.clip(self._episode_reward / 100.0 + 0.5, 0.0, 1.0)
        obs[252] = (self._stage_id - 1) / 5.0

        return obs

    def _compute_stage_progress(self) -> float:
        """Compute progress toward current stage objectives (0-1)."""
        if self._current_stage is None:
            return 0.0

        # Count completed objectives
        total = len(self._current_stage.objectives)
        completed = len(self._episode_stats.objectives_completed)

        return completed / max(1, total)

    def _compute_reward(self, action: int) -> tuple[float, bool, dict[str, Any]]:
        """Compute reward for the current step.

        Args:
            action: Action that was taken.

        Returns:
            Tuple of (reward, terminated, info_dict).
        """
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}

        if self._sim is None:
            # Mock reward for testing
            reward = -0.001  # Time penalty
            return reward, terminated, info

        # Get reward from simulator
        try:
            rewards = self._sim.get_rewards()
            reward = float(rewards[0]) if len(rewards) > 0 else 0.0

            dones = self._sim.get_dones()
            terminated = bool(dones[0]) if len(dones) > 0 else False
        except Exception:
            pass

        # Apply stage-specific reward shaping
        if self._current_stage is not None:
            reward = self._shape_reward(reward, action, info)

        # Add time penalty
        if self._current_stage is not None:
            reward += self._current_stage.rewards.penalty_per_tick
        else:
            reward -= 0.0001

        return reward, terminated, info

    def _shape_reward(
        self,
        base_reward: float,
        action: int,
        info: dict[str, Any],
    ) -> float:
        """Apply stage-specific reward shaping.

        Args:
            base_reward: Raw reward from simulator.
            action: Action taken.
            info: Info dict to populate with reward breakdown.

        Returns:
            Shaped reward.
        """
        if self._current_stage is None:
            return base_reward

        reward = base_reward
        dense = self._current_stage.rewards.dense_rewards

        # Milestone bonus rewards (one-time per episode)
        milestone_rewards = {
            "entered_nether": 100.0,
            "first_blaze_kill": 50.0,
            "stronghold_found": 75.0,
            "portal_activated": 150.0,
            "entered_end": 200.0,
            "first_dragon_hit": 50.0,
            "dragon_killed": 1000.0,
        }

        for milestone, bonus in milestone_rewards.items():
            if milestone in info and milestone not in self._milestones_achieved:
                self._milestones_achieved.add(milestone)
                reward += bonus
                info[f"milestone_{milestone}"] = bonus

        # Stage 1: Basic Survival
        if self._stage_id == 1:
            if "zombie_killed" in info:
                reward += dense.get("zombie_killed", 0.5)
            if "skeleton_killed" in info:
                reward += dense.get("skeleton_killed", 0.5)
            if "wood_mined" in info:
                reward += dense.get("wood_mined", 0.1)

        # Stage 2: Resource Gathering
        elif self._stage_id == 2:
            if "iron_obtained" in info:
                reward += dense.get("iron_obtained", 0.5)
            if "diamond_obtained" in info:
                reward += dense.get("diamond_obtained", 2.0)

        # Stage 3: Nether Navigation
        elif self._stage_id == 3:
            if "entered_nether" in info:
                reward += dense.get("enter_nether", 5.0)
            if "blaze_killed" in info:
                reward += dense.get("blaze_killed", 1.0)
                if "first_blaze_kill" not in info:
                    # Mark first blaze kill for milestone tracking
                    if self._episode_stats.mobs_killed == 0:
                        info["first_blaze_kill"] = True

        # Stage 4: Enderman Hunting + Stronghold + Portal
        elif self._stage_id == 4:
            if "pearl_obtained" in info:
                pearl_count = info.get("pearl_count", 1)
                reward += dense.get("pearl_obtained", 2.0)
                # Bonus for first pearl
                if "first_pearl" not in self._milestones_achieved and pearl_count >= 1:
                    self._milestones_achieved.add("first_pearl")
                    reward += 5.0
                    info["milestone_first_pearl"] = 5.0
                # Threshold bonuses at 6 and 12 pearls
                if "pearls_6" not in self._milestones_achieved and pearl_count >= 6:
                    self._milestones_achieved.add("pearls_6")
                    reward += 3.0
                if "pearls_12" not in self._milestones_achieved and pearl_count >= 12:
                    self._milestones_achieved.add("pearls_12")
                    reward += 5.0

            if "eye_crafted" in info:
                eye_count = info.get("eye_count", 1)
                reward += dense.get("eye_crafted", 1.5)
                # First eye is a major milestone
                if "first_eye" not in self._milestones_achieved and eye_count >= 1:
                    self._milestones_achieved.add("first_eye")
                    reward += 5.0
                    info["milestone_first_eye"] = 5.0
                # Enough eyes to activate portal
                if "eyes_12" not in self._milestones_achieved and eye_count >= 12:
                    self._milestones_achieved.add("eyes_12")
                    reward += 10.0

            if "eye_thrown" in info:
                reward += dense.get("eye_thrown", 0.5)

            if "stronghold_found" in info:
                if "stronghold_found" not in self._milestones_achieved:
                    self._milestones_achieved.add("stronghold_found")
                    reward += dense.get("stronghold_found", 20.0)
                    info["milestone_stronghold_found"] = 20.0

            if "portal_frame_filled" in info:
                frames_filled = info.get("frames_filled", 1)
                reward += dense.get("portal_frame_filled", 2.0) * frames_filled

            if "portal_activated" in info:
                if "portal_activated" not in self._milestones_achieved:
                    self._milestones_achieved.add("portal_activated")
                    reward += dense.get("portal_activated", 50.0)
                    info["milestone_portal_activated"] = 50.0

            # Distance-based shaping: reward getting closer to stronghold
            if "stronghold_distance_delta" in info:
                delta = info["stronghold_distance_delta"]
                if delta < 0:  # Getting closer
                    reward += dense.get("approach_stronghold", 0.01) * abs(delta)

        # Stage 5: Stronghold Finding (legacy, kept for backward compat)
        elif self._stage_id == 5:
            if "stronghold_found" in info:
                reward += dense.get("stronghold_found", 10.0)
            if "portal_frame_filled" in info:
                reward += dense.get("portal_frame_filled", 1.0)

        # Stage 6: End Fight
        elif self._stage_id == 6:
            if "crystal_destroyed" in info:
                reward += dense.get("destroy_crystal", 3.0)
            if "dragon_damaged" in info:
                damage = info.get("dragon_damage_amount", 0.0)
                reward += dense.get("damage_dragon", 0.5) * damage
                if "first_dragon_hit" not in info:
                    if "first_dragon_hit" not in self._milestones_achieved:
                        info["first_dragon_hit"] = True
            if "dragon_killed" in info:
                reward += dense.get("dragon_killed", 50.0)

        return reward

    def _check_stage_mastery(self, success: bool) -> bool:
        """Check if agent has mastered current stage.

        Args:
            success: Whether current episode was successful.

        Returns:
            True if stage was mastered.
        """
        self._recent_episodes.append(success)

        if len(self._recent_episodes) < 100:
            return False

        success_rate = sum(self._recent_episodes) / len(self._recent_episodes)

        threshold = self._custom_threshold
        if threshold is None and self._current_stage is not None:
            threshold = self._current_stage.curriculum_threshold
        if threshold is None:
            threshold = 0.7

        return success_rate >= threshold

    def _advance_stage(self) -> bool:
        """Attempt to advance to the next curriculum stage.

        Serializes the current inventory so it persists into the next stage.

        Returns:
            True if successfully advanced.
        """
        if self._stage_id >= 6:
            return False

        # Persist inventory before transitioning
        self._inventory_state = self._serialize_inventory()

        next_stage = self._stage_id + 1
        self._load_stage(next_stage)
        self._recent_episodes.clear()

        # Reinitialize simulator for new stage
        self._sim_initialized = False

        return True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode.

        Args:
            seed: Random seed for this episode.
            options: Optional reset options:
                - "stage_id": Force reset to specific stage (1-6).
                - "skip_advance": If True, don't auto-advance even if mastered.

        Returns:
            Tuple of (initial observation, info dict).
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}

        # Handle forced stage change (clears persisted inventory)
        if "stage_id" in options:
            new_stage = int(options["stage_id"])
            if 1 <= new_stage <= 6:
                self._load_stage(new_stage)
                self._sim_initialized = False
                self._inventory_state = {}

        # Initialize simulator if needed
        self._init_simulator()

        # Reset episode state
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode_stats = EpisodeStats(stage_id=self._stage_id)
        self._action_history.clear()
        self._reward_history.clear()
        self._milestones_achieved.clear()

        # Reset dimension state to stage-appropriate starting dimension
        initial_dim = self._get_initial_dimension()
        self._dimension_state = DimensionState(current=initial_dim, previous=initial_dim)

        # Reset simulator
        if self._sim is not None:
            try:
                self._sim.reset()
                # Execute no-op to get initial state
                self._sim.step(np.array([0], dtype=np.int32))
            except Exception:
                pass

        # Configure spawn for current stage
        self._spawn_for_stage()

        # Build info dict
        info: dict[str, Any] = {
            "stage_id": self._stage_id,
            "stage_name": self._current_stage.name
            if self._current_stage
            else f"Stage {self._stage_id}",
            "max_steps": self._max_steps,
            "stage_advanced": False,
            "dimension": self._dimension_state.current.name.lower(),
        }

        # Compute initial observation
        obs = self._compute_observation()

        # Restore persisted inventory from prior stage transition
        if self._inventory_state:
            self._restore_inventory()
            obs = self._obs.copy()
            info["inventory_restored"] = True

        return obs, info

    def step(
        self,
        action: int | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Discrete action index (0-31).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Convert action to int
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        # Validate action
        if not 0 <= action < self.ACT_DIM:
            action = 0  # Default to noop for invalid actions

        self._step_count += 1

        # Record action in history
        self._action_history.append(action)

        # Execute action in simulator
        if self._sim is not None:
            try:
                # Map our 32-action space to simulator's action space
                sim_action = self._map_action(action)
                self._sim.step(np.array([sim_action], dtype=np.int32))
            except Exception:

import logging

logger = logging.getLogger(__name__)

                pass

        # Tick dimension cooldown
        self._dimension_state.tick()

        # Compute reward
        reward, terminated, step_info = self._compute_reward(action)

        # Update episode stats
        self._episode_reward += reward
        self._episode_stats.steps = self._step_count
        self._episode_stats.total_reward = self._episode_reward

        # Update Stage 2 resource progress from observation inventory slots
        if self._stage_id == 2:
            self._update_stage2_resource_counts()

        # Record reward in history
        self._reward_history.append(reward)

        # Check truncation (max steps reached)
        truncated = self._step_count >= self._max_steps and not terminated

        # Check if episode ended successfully
        success = terminated and self._episode_reward > 0

        # Build info dict
        info: dict[str, Any] = {
            "stage_id": self._stage_id,
            "step": self._step_count,
            "episode_reward": self._episode_reward,
            "stage_advanced": False,
            **step_info,
        }

        # Handle episode end
        if terminated or truncated:
            # Record in curriculum
            self._curriculum.record_episode(
                success=success,
                reward=self._episode_reward,
                ticks=self._step_count,
                stage_id=StageID(self._stage_id),
            )

            # Check for stage mastery and auto-advance
            if self._auto_advance and success:
                if self._check_stage_mastery(success):
                    if self._advance_stage():
                        info["stage_advanced"] = True
                        info["new_stage_id"] = self._stage_id

            # Add episode summary to info
            info["episode_stats"] = self._episode_stats.to_dict()
            info["success"] = success

        # Compute observation
        obs = self._compute_observation()

        # Detect dimension transitions from observation vector
        transition_info = self._detect_dimension_transition(obs)
        if transition_info:
            info.update(transition_info)

        # Final resource count update from the terminal observation
        if (terminated or truncated) and self._stage_id == 2:
            self._update_stage2_resource_counts()
            # Re-serialize with final counts
            info["episode_stats"] = self._episode_stats.to_dict()

        return obs, reward, terminated, truncated, info

    def _update_stage2_resource_counts(self) -> None:
        """Update Stage 2 resource peak counts from the observation buffer.

        Reads inventory indices 36 (iron_ingot), 38 (diamond), 39 (obsidian)
        which are normalized to max stack size (64).
        """
        obs = self._obs
        iron = int(round(obs[36] * 64))
        diamond = int(round(obs[38] * 64))
        obsidian = int(round(obs[39] * 64))
        self._episode_stats.iron_count = max(self._episode_stats.iron_count, iron)
        self._episode_stats.diamond_count = max(self._episode_stats.diamond_count, diamond)
        self._episode_stats.obsidian_count = max(self._episode_stats.obsidian_count, obsidian)

    def _map_action(self, action: int) -> int:
        """Map our 32-action space to the simulator's 17-action space.

        Args:
            action: Action from our extended action space (0-31).

        Returns:
            Action for simulator (0-16).
        """
        # Direct mappings for actions 0-16
        if action <= 16:
            return action

        # Map extended actions to simulator actions
        mapping = {
            SpeedrunAction.LOOK_LEFT_FAST: 12,  # LOOK_LEFT (repeated)
            SpeedrunAction.LOOK_RIGHT_FAST: 13,  # LOOK_RIGHT (repeated)
            SpeedrunAction.LOOK_UP_FAST: 15,  # LOOK_UP (repeated)
            SpeedrunAction.LOOK_DOWN_FAST: 16,  # LOOK_DOWN (repeated)
            SpeedrunAction.USE_ITEM: 14,  # SWAP_WEAPON (overloaded)
            SpeedrunAction.DROP_ITEM: 0,  # NOOP (no direct mapping)
            # Hotbar slots map to SWAP_WEAPON
            SpeedrunAction.HOTBAR_1: 14,
            SpeedrunAction.HOTBAR_2: 14,
            SpeedrunAction.HOTBAR_3: 14,
            SpeedrunAction.HOTBAR_4: 14,
            SpeedrunAction.HOTBAR_5: 14,
            SpeedrunAction.HOTBAR_6: 14,
            SpeedrunAction.HOTBAR_7: 14,
            SpeedrunAction.HOTBAR_8: 14,
            SpeedrunAction.HOTBAR_9: 14,
            SpeedrunAction.CRAFT: 0,  # NOOP (craft is context-sensitive)
        }

        return mapping.get(action, 0)

    def render(self) -> np.ndarray | None:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise.
        """
        if self._render_mode == "rgb_array":
            # Would return frame from simulator
            # For now, return black frame
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        """Clean up environment resources."""
        self._sim = None
        self._sim_initialized = False

    def set_stage(self, stage_id: int) -> None:
        """Manually set the curriculum stage.

        Serializes current inventory so items persist into the new stage.

        Args:
            stage_id: Stage to switch to (1-6).

        Raises:
            ValueError: If stage_id is invalid.
        """
        if not 1 <= stage_id <= 6:
            raise ValueError(f"stage_id must be 1-6, got {stage_id}")

        # Persist inventory before transitioning
        self._inventory_state = self._serialize_inventory()

        self._load_stage(stage_id)
        self._sim_initialized = False
        self._recent_episodes.clear()

    def get_stage_config(self) -> dict[str, Any] | None:
        """Get the configuration for the current stage.

        Returns:
            Stage configuration as dict, or None if not available.
        """
        if self._current_stage is None:
            return None
        return self._current_stage.to_dict()

    def _get_initial_dimension(self) -> Dimension:
        """Determine the starting dimension based on current stage.

        Returns:
            Dimension appropriate for the stage start:
            - Stages 1-2: Overworld
            - Stage 3: Overworld (spawns at portal, transitions to Nether)
            - Stage 4: Overworld (hunting endermen)
            - Stage 5: Overworld (finding stronghold)
            - Stage 6: End (spawns on obsidian platform)
        """
        if self._stage_id == 6:
            return Dimension.END
        return Dimension.OVERWORLD

    def _detect_dimension_transition(
        self, obs: np.ndarray
    ) -> dict[str, Any] | None:
        """Detect dimension changes from the observation vector and update state.

        Reads obs[127] (dimension indicator: 0.0=Overworld, 0.5=Nether, 1.0=End)
        and compares against the current tracked dimension. On transition:
        - Saves the player position for the departing dimension
        - Records the transition event
        - Applies milestone rewards via info dict
        - Sets portal cooldown

        Args:
            obs: Current 256-float observation vector.

        Returns:
            Info dict additions if a transition occurred, None otherwise.
        """
        if obs.size <= 127:
            return None

        observed_dim = Dimension.from_obs_value(float(obs[127]))
        dim_state = self._dimension_state

        if observed_dim == dim_state.current:
            return None

        if dim_state.in_cooldown:
            # Ignore rapid flickering during portal animation
            return None

        from_dim = dim_state.current
        to_dim = observed_dim

        # Save position in the departing dimension (obs[0:3] = x, y, z)
        if obs.size >= 3:
            pos = (float(obs[0]), float(obs[1]), float(obs[2]))
            dim_state.save_position(from_dim, pos)

        # Record the transition
        dim_state.record_transition(self._step_count, from_dim, to_dim)

        # Restore saved position for the destination dimension into obs
        saved_pos = dim_state.get_saved_position(to_dim)
        # Only restore if the destination has been visited before (non-default)
        visited_before = any(
            t[2] == to_dim for t in dim_state.transitions[:-1]
        )
        if visited_before:
            obs[0], obs[1], obs[2] = saved_pos

        # Build transition info for reward shaping
        info: dict[str, Any] = {
            "dimension_transition": True,
            "from_dimension": from_dim.name.lower(),
            "to_dimension": to_dim.name.lower(),
            "transition_tick": self._step_count,
        }

        # Map transitions to milestone keys for reward shaping
        if from_dim == Dimension.OVERWORLD and to_dim == Dimension.NETHER:
            info["entered_nether"] = True
        elif from_dim == Dimension.OVERWORLD and to_dim == Dimension.END:
            info["entered_end"] = True
        elif from_dim == Dimension.END and to_dim == Dimension.OVERWORLD:
            info["dragon_fight_complete"] = True

        # Update dimension indicator in the observation buffer
        obs[127] = to_dim.to_obs_value()

        # Update portal state observations
        obs[227] = 1.0  # in_portal_cooldown

        return info

    def save_curriculum_progress(self, path: str | Path) -> None:
        """Save curriculum progress to file.

        Args:
            path: Path to save JSON file.
        """
        self._curriculum.save_progress(path)

    def load_curriculum_progress(self, path: str | Path) -> None:
        """Load curriculum progress from file.

        Args:
            path: Path to JSON file.
        """
        self._curriculum.load_progress(path)

        # Update current stage from loaded progress
        if self._curriculum.current_stage is not None:
            self._load_stage(self._curriculum.current_stage.value)


# =============================================================================
# Factory Functions
# =============================================================================


def make_speedrun_env(
    stage_id: int = 1,
    auto_advance: bool = True,
    **kwargs: Any,
) -> SpeedrunEnv:
    """Create a SpeedrunEnv with default settings.

    Args:
        stage_id: Initial curriculum stage (1-6).
        auto_advance: Whether to auto-advance stages.
        **kwargs: Additional arguments passed to SpeedrunEnv.

    Returns:
        Configured SpeedrunEnv instance.
    """
    return SpeedrunEnv(
        stage_id=stage_id,
        auto_advance=auto_advance,
        **kwargs,
    )


def make_stage_env(stage_id: int, **kwargs: Any) -> SpeedrunEnv:
    """Create an environment locked to a specific stage (no auto-advance).

    Useful for training specialized policies on individual stages.

    Args:
        stage_id: Stage to train on (1-6).
        **kwargs: Additional arguments passed to SpeedrunEnv.

    Returns:
        SpeedrunEnv locked to the specified stage.
    """
    return SpeedrunEnv(
        stage_id=stage_id,
        auto_advance=False,
        **kwargs,
    )
