"""Unified Minecraft 1.8.9 speedrun environment.

This module provides FreeTheEndEnv, a unified Gymnasium environment that supports:
1. Single stage mode: Train on a specific stage (1-6)
2. Curriculum mode: Automatic stage progression based on mastery
3. Full speedrun mode: Complete run from spawn to dragon kill

The environment wraps the mc189_core Vulkan backend and provides:
- Compact 256-dimensional observation space
- 32 discrete actions covering movement, combat, and interaction
- Stage-specific reward shaping for efficient curriculum learning
- Automatic world seeding for reproducibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError as exc:
    raise ImportError("gymnasium is required for FreeTheEndEnv") from exc

from .progression import ProgressTracker, SpeedrunProgress


class StageID(IntEnum):
    """Speedrun stage identifiers matching curriculum."""

    BASIC_SURVIVAL = 1
    RESOURCE_GATHERING = 2
    NETHER_NAVIGATION = 3
    ENDERMAN_HUNTING = 4
    STRONGHOLD_FINDING = 5
    END_FIGHT = 6


@dataclass
class StageConfig:
    """Configuration for a specific stage."""

    stage_id: StageID
    spawn_dimension: int  # 0=overworld, 1=nether, 2=end
    spawn_position: tuple[float, float, float] | None  # None = random
    initial_inventory: dict[str, int] = field(default_factory=dict)
    initial_health: float = 20.0
    initial_hunger: float = 20.0
    max_ticks: int = 36000  # 30 minutes
    time_of_day: int = 0  # 0 = dawn


# Pre-configured stage settings
STAGE_CONFIGS: dict[StageID, StageConfig] = {
    StageID.BASIC_SURVIVAL: StageConfig(
        stage_id=StageID.BASIC_SURVIVAL,
        spawn_dimension=0,
        spawn_position=None,
        initial_inventory={},
        max_ticks=12000,  # 10 minutes
    ),
    StageID.RESOURCE_GATHERING: StageConfig(
        stage_id=StageID.RESOURCE_GATHERING,
        spawn_dimension=0,
        spawn_position=None,
        initial_inventory={
            "wooden_pickaxe": 1,
            "wooden_sword": 1,
            "cobblestone": 32,
            "logs": 16,
            "bread": 8,
        },
        max_ticks=18000,  # 15 minutes
    ),
    StageID.NETHER_NAVIGATION: StageConfig(
        stage_id=StageID.NETHER_NAVIGATION,
        spawn_dimension=0,
        spawn_position=None,
        initial_inventory={
            "iron_pickaxe": 1,
            "iron_sword": 1,
            "bucket": 1,
            "flint_and_steel": 1,
            "obsidian": 10,
            "cobblestone": 64,
            "bread": 16,
        },
        max_ticks=24000,  # 20 minutes
    ),
    StageID.ENDERMAN_HUNTING: StageConfig(
        stage_id=StageID.ENDERMAN_HUNTING,
        spawn_dimension=1,  # Start in nether (warped forest)
        spawn_position=None,
        initial_inventory={
            "diamond_sword": 1,
            "diamond_pickaxe": 1,
            "blaze_rods": 7,
            "golden_boots": 1,  # For piglins
            "gold_ingots": 32,
            "bread": 32,
            "water_bucket": 1,
        },
        max_ticks=18000,  # 15 minutes
    ),
    StageID.STRONGHOLD_FINDING: StageConfig(
        stage_id=StageID.STRONGHOLD_FINDING,
        spawn_dimension=0,
        spawn_position=None,
        initial_inventory={
            "diamond_sword": 1,
            "diamond_pickaxe": 1,
            "ender_pearls": 12,
            "blaze_powder": 12,
            "bread": 32,
            "cobblestone": 64,
        },
        max_ticks=18000,  # 15 minutes
    ),
    StageID.END_FIGHT: StageConfig(
        stage_id=StageID.END_FIGHT,
        spawn_dimension=2,  # Start in the end
        spawn_position=(0.5, 64.0, 0.5),  # On obsidian platform
        initial_inventory={
            "diamond_sword": 1,
            "bow": 1,
            "arrows": 64,
            "golden_apples": 8,
            "beds": 5,  # For bed strat
            "water_bucket": 1,
            "ender_pearls": 16,
            "cobblestone": 64,
        },
        initial_health=20.0,
        initial_hunger=20.0,
        max_ticks=12000,  # 10 minutes
    ),
}


class FreeTheEndEnv(gym.Env):
    """Unified Minecraft 1.8.9 speedrun environment.

    Can operate in:
    1. Single stage mode: Fixed stage for focused training
    2. Curriculum mode: Automatic stage progression
    3. Full speedrun mode: All stages in sequence

    Observation Space: Box(256,) float32
        - Player state (position, velocity, health, hunger): 16 dims
        - Inventory summary (key items for speedrun): 24 dims
        - Entity awareness (nearby mobs, dragon state): 48 dims
        - Progress features (stage completion, timing): 32 dims
        - Raycast distances (navigation): 16 dims
        - Local block summary (compressed voxels): 64 dims
        - Stage one-hot + meta: 56 dims

    Action Space: Discrete(32)
        Core actions for speedrun:
        0-8: Movement (none, forward, back, left, right, diagonals)
        9-10: Jump (no, yes)
        11-12: Sprint (no, yes)
        13-14: Attack (no, yes)
        15-16: Use item (no, yes)
        17-24: Look direction (8 discrete directions)
        25-31: Special (hotbar select, craft, interact)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        stage: StageID | int | None = None,
        curriculum: bool = False,
        full_speedrun: bool = False,
        seed: int | None = None,
        max_episode_steps: int = 36000,  # 30 minutes at 20 TPS
        render_mode: str | None = None,
        shader_dir: str | None = None,
        curriculum_threshold: float = 0.7,
    ) -> None:
        """Initialize FreeTheEndEnv.

        Args:
            stage: Fixed stage (1-6 or StageID), None for curriculum/speedrun
            curriculum: Enable automatic stage progression
            full_speedrun: Single episode from spawn to dragon kill
            seed: World seed for reproducibility
            max_episode_steps: Maximum ticks per episode
            render_mode: "human" or "rgb_array"
            shader_dir: Path to Vulkan shaders (auto-detected if None)
            curriculum_threshold: Success rate required to advance stages
        """
        super().__init__()

        # Validate exclusive mode flags
        mode_count = sum([stage is not None, curriculum, full_speedrun])
        if mode_count > 1:
            raise ValueError("Only one of stage, curriculum, or full_speedrun can be set")

        # Convert int to StageID if needed
        if isinstance(stage, int):
            stage = StageID(stage)

        self.fixed_stage = stage
        self.curriculum_enabled = curriculum
        self.full_speedrun = full_speedrun
        self._seed = seed
        self.max_steps = max_episode_steps
        self.render_mode = render_mode
        self.curriculum_threshold = curriculum_threshold

        # Resolve shader directory
        if shader_dir is None:
            shader_dir = str(Path(__file__).resolve().parents[2] / "cpp" / "shaders")
        self._shader_dir = shader_dir

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(32)

        # Initialize backend
        self._simulator: Any = None
        self._has_simulator = False
        self._init_backend()

        # Episode state
        self.current_stage = stage if stage else StageID.BASIC_SURVIVAL
        self.steps = 0
        self.episode_count = 0
        self.progress = SpeedrunProgress()
        self.tracker = ProgressTracker(progress=self.progress)

        # Curriculum tracking
        self._stage_successes: dict[StageID, int] = dict.fromkeys(StageID, 0)
        self._stage_episodes: dict[StageID, int] = dict.fromkeys(StageID, 0)
        self._stage_rewards: dict[StageID, list[float]] = {s: [] for s in StageID}

        # Episode tracking
        self._episode_reward = 0.0
        self._prev_obs: np.ndarray | None = None
        self._prev_dragon_health = 200.0
        self._prev_player_health = 20.0

    def _init_backend(self) -> None:
        """Initialize the mc189_core Vulkan backend."""
        try:
            import mc189_core

            config = mc189_core.SimulatorConfig()
            config.num_envs = 1
            config.shader_dir = self._shader_dir
            self._simulator = mc189_core.MC189Simulator(config)
            self._has_simulator = True
        except ImportError:

import logging

logger = logging.getLogger(__name__)

            self._simulator = None
            self._has_simulator = False

    def _backend_reset(self, stage: StageID, seed: int | None) -> None:
        """Reset the backend simulator for a specific stage.

        Args:
            stage: Stage to initialize
            seed: World seed (None for random)
        """
        if not self._has_simulator:
            return

        config = STAGE_CONFIGS.get(stage, STAGE_CONFIGS[StageID.BASIC_SURVIVAL])

        # Reset simulator
        self._simulator.reset()

        # Configure spawn
        if hasattr(self._simulator, "set_spawn_config"):
            self._simulator.set_spawn_config(
                dimension=config.spawn_dimension,
                position=config.spawn_position,
                health=config.initial_health,
                hunger=config.initial_hunger,
                time_of_day=config.time_of_day,
            )

        # Set inventory
        if hasattr(self._simulator, "set_inventory") and config.initial_inventory:
            self._simulator.set_inventory(config.initial_inventory)

        # Set seed
        if seed is not None and hasattr(self._simulator, "set_seed"):
            self._simulator.set_seed(seed)

    def _backend_step(self, action: int) -> None:
        """Execute one step in the backend simulator.

        Args:
            action: Discrete action index (0-31)
        """
        if not self._has_simulator:
            return

        # Convert discrete action to simulator format
        self._simulator.step(np.array([action], dtype=np.int32))

    def _get_raw_observation(self) -> dict[str, Any]:
        """Get raw observation dictionary from backend.

        Returns:
            Dictionary with player state, inventory, entities, etc.
        """
        if not self._has_simulator:
            return self._get_mock_observation()

        # Get raw state from simulator
        raw = {}

        # Player state
        if hasattr(self._simulator, "get_player_state"):
            raw["player"] = self._simulator.get_player_state()
        else:
            obs_array = np.array(self._simulator.get_observations(), dtype=np.float32)
            raw["player"] = self._parse_player_from_array(obs_array)

        # Inventory
        if hasattr(self._simulator, "get_inventory"):
            raw["inventory"] = self._simulator.get_inventory()
        else:
            raw["inventory"] = {}

        # Dragon state (if in End)
        if hasattr(self._simulator, "get_dragon_state"):
            raw["dragon"] = self._simulator.get_dragon_state()
        else:
            raw["dragon"] = {"is_active": False, "dragon_health": 200.0, "crystals_remaining": 10}

        # Nearby entities
        if hasattr(self._simulator, "get_nearby_entities"):
            raw["nearby_mobs"] = self._simulator.get_nearby_entities()
        else:
            raw["nearby_mobs"] = []

        # Game tick
        if hasattr(self._simulator, "get_tick"):
            raw["tick_number"] = self._simulator.get_tick()
        else:
            raw["tick_number"] = self.steps

        return raw

    def _parse_player_from_array(self, obs: np.ndarray) -> dict[str, Any]:
        """Parse player state from flat observation array.

        Args:
            obs: Flat observation array from simulator

        Returns:
            Player state dictionary
        """
        # Indices based on mc189_core observation layout
        # This is a fallback when structured API is not available
        return {
            "position": (
                obs[0] if len(obs) > 0 else 0.0,
                obs[1] if len(obs) > 1 else 64.0,
                obs[2] if len(obs) > 2 else 0.0,
            ),
            "velocity": (
                obs[3] if len(obs) > 3 else 0.0,
                obs[4] if len(obs) > 4 else 0.0,
                obs[5] if len(obs) > 5 else 0.0,
            ),
            "health": obs[6] * 20.0 if len(obs) > 6 else 20.0,
            "hunger": obs[7] * 20.0 if len(obs) > 7 else 20.0,
            "yaw": obs[8] * 180.0 if len(obs) > 8 else 0.0,
            "pitch": obs[9] * 90.0 if len(obs) > 9 else 0.0,
            "dimension": int(obs[10] * 2) if len(obs) > 10 else 0,
            "on_ground": bool(obs[11] > 0.5) if len(obs) > 11 else True,
        }

    def _get_mock_observation(self) -> dict[str, Any]:
        """Generate mock observation when backend unavailable.

        Returns:
            Plausible observation dictionary for testing
        """
        return {
            "player": {
                "position": (0.0, 64.0, 0.0),
                "velocity": (0.0, 0.0, 0.0),
                "health": 20.0,
                "max_health": 20.0,
                "hunger": 20.0,
                "saturation": 5.0,
                "exhaustion": 0.0,
                "yaw": 0.0,
                "pitch": 0.0,
                "dimension": 0,
                "on_ground": True,
                "in_water": False,
                "in_lava": False,
            },
            "inventory": {},
            "dragon": {
                "is_active": self.current_stage == StageID.END_FIGHT,
                "dragon_health": 200.0 if self.current_stage == StageID.END_FIGHT else 0.0,
                "crystals_remaining": 10 if self.current_stage == StageID.END_FIGHT else 0,
                "phase": 1 if self.current_stage == StageID.END_FIGHT else 0,
            },
            "nearby_mobs": [],
            "tick_number": self.steps,
        }

    def _get_observation(self) -> np.ndarray:
        """Build the 256-dimensional observation vector.

        Returns:
            Normalized observation array of shape (256,)
        """
        raw = self._get_raw_observation()

        # Update progress tracker
        self.tracker.update_from_observation(raw)

        obs = np.zeros(256, dtype=np.float32)
        idx = 0

        # === Player state (16 dims) ===
        player = raw.get("player", {})
        pos = player.get("position", (0.0, 64.0, 0.0))
        vel = player.get("velocity", (0.0, 0.0, 0.0))

        # Ensure pos and vel are tuples/lists of floats
        if isinstance(pos, dict):
            pos = (pos.get("x", 0.0), pos.get("y", 64.0), pos.get("z", 0.0))
        if isinstance(vel, dict):
            vel = (vel.get("x", 0.0), vel.get("y", 0.0), vel.get("z", 0.0))
        pos = tuple(float(x) for x in pos)
        vel = tuple(float(x) for x in vel)

        # Normalized position (roughly -1 to 1 for typical play area)
        obs[idx] = np.clip(pos[0] / 1000.0, -1, 1)
        obs[idx + 1] = np.clip((pos[1] - 64) / 128.0, -1, 1)
        obs[idx + 2] = np.clip(pos[2] / 1000.0, -1, 1)
        idx += 3

        # Velocity
        obs[idx] = np.clip(vel[0] / 10.0, -1, 1)
        obs[idx + 1] = np.clip(vel[1] / 10.0, -1, 1)
        obs[idx + 2] = np.clip(vel[2] / 10.0, -1, 1)
        idx += 3

        # Health and hunger (0-1)
        obs[idx] = player.get("health", 20.0) / 20.0
        obs[idx + 1] = player.get("hunger", 20.0) / 20.0
        idx += 2

        # Look direction (normalized)
        obs[idx] = player.get("yaw", 0.0) / 180.0
        obs[idx + 1] = player.get("pitch", 0.0) / 90.0
        idx += 2

        # Dimension one-hot (3 dims)
        dim = player.get("dimension", 0)
        if 0 <= dim <= 2:
            obs[idx + dim] = 1.0
        idx += 3

        # Binary flags (3 dims)
        obs[idx] = float(player.get("on_ground", True))
        obs[idx + 1] = float(player.get("in_water", False))
        obs[idx + 2] = float(player.get("in_lava", False))
        idx += 3

        # === Inventory summary (24 dims) ===
        inv = raw.get("inventory", {})
        key_items = [
            "blaze_rods",
            "ender_pearls",
            "eyes_of_ender",
            "obsidian",
            "iron_ingots",
            "diamonds",
            "coal",
            "logs",
            "cobblestone",
            "arrows",
            "bread",
            "golden_apples",
            "potions",
            "lava_buckets",
            "water_buckets",
            "empty_buckets",
            "iron_pickaxe",
            "diamond_pickaxe",
            "iron_sword",
            "diamond_sword",
            "bow",
            "shield",
            "beds",
            "flint_and_steel",
        ]
        for i, item in enumerate(key_items):
            if i >= 24:
                break
            count = inv.get(item, 0)
            # Log-normalize: log(1 + count) / log(65)
            obs[idx + i] = np.log1p(count) / np.log(65)
        idx += 24

        # === Entity awareness (48 dims) ===
        nearby = raw.get("nearby_mobs", [])
        for i in range(8):  # Max 8 nearby entities, 6 dims each
            if i < len(nearby):
                mob = nearby[i]
                player_pos = np.array(pos)
                mob_pos = np.array(mob.get("position", pos))
                delta = mob_pos - player_pos
                dist = np.linalg.norm(delta)
                if dist > 0:
                    direction = delta / dist
                else:
                    direction = np.zeros(3)

                obs[idx + i * 6] = float(mob.get("mob_type", 0)) / 100.0
                obs[idx + i * 6 + 1] = np.clip(dist / 64.0, 0, 1)
                obs[idx + i * 6 + 2 : idx + i * 6 + 5] = direction
                obs[idx + i * 6 + 5] = float(mob.get("is_hostile", False))
        idx += 48

        # === Progress features (32 dims) ===
        progress_vec = self.progress.to_observation()
        obs[idx : idx + 32] = progress_vec
        idx += 32

        # === Raycast distances (16 dims) ===
        # Placeholder: would come from backend ray casting
        raycasts = raw.get("raycasts", np.ones(16))
        if isinstance(raycasts, (list, np.ndarray)):
            obs[idx : idx + 16] = np.clip(np.array(raycasts)[:16] / 64.0, 0, 1)
        else:
            obs[idx : idx + 16] = 1.0  # Max distance (nothing nearby)
        idx += 16

        # === Local block summary (64 dims) ===
        # Compressed representation of nearby blocks
        local_blocks = raw.get("local_blocks", np.zeros(64))
        if isinstance(local_blocks, (list, np.ndarray)):
            obs[idx : idx + 64] = np.clip(np.array(local_blocks)[:64] / 255.0, 0, 1)
        idx += 64

        # === Stage and meta info (56 dims) ===
        # Stage one-hot (6 dims)
        stage_idx = self.current_stage.value - 1
        if 0 <= stage_idx < 6:
            obs[idx + stage_idx] = 1.0
        idx += 6

        # Time remaining (1 dim)
        config = STAGE_CONFIGS.get(self.current_stage, STAGE_CONFIGS[StageID.BASIC_SURVIVAL])
        obs[idx] = 1.0 - (self.steps / config.max_ticks)
        idx += 1

        # Episode count normalized (1 dim)
        obs[idx] = np.clip(self.episode_count / 1000.0, 0, 1)
        idx += 1

        # Dragon state (8 dims) - important for END_FIGHT
        dragon = raw.get("dragon", {})
        obs[idx] = float(dragon.get("is_active", False))
        obs[idx + 1] = dragon.get("dragon_health", 200.0) / 200.0
        obs[idx + 2] = dragon.get("crystals_remaining", 10) / 10.0
        obs[idx + 3] = dragon.get("phase", 0) / 3.0
        # Dragon direction (from progress tracker or raw)
        idx += 8

        # Padding to reach 256
        # Remaining dims are zeros (already initialized)

        return obs

    def _compute_reward(self) -> float:
        """Compute reward based on current stage and progress.

        Returns:
            Scalar reward value
        """
        reward = 0.0
        raw = self._get_raw_observation()

        # Time penalty (small constant to encourage faster completion)
        reward -= 0.0001

        # Stage-specific rewards
        if self.current_stage == StageID.BASIC_SURVIVAL:
            reward += self._reward_basic_survival(raw)
        elif self.current_stage == StageID.RESOURCE_GATHERING:
            reward += self._reward_resource_gathering(raw)
        elif self.current_stage == StageID.NETHER_NAVIGATION:
            reward += self._reward_nether_navigation(raw)
        elif self.current_stage == StageID.ENDERMAN_HUNTING:
            reward += self._reward_enderman_hunting(raw)
        elif self.current_stage == StageID.STRONGHOLD_FINDING:
            reward += self._reward_stronghold_finding(raw)
        elif self.current_stage == StageID.END_FIGHT:
            reward += self._reward_end_fight(raw)

        # Health change penalty
        player = raw.get("player", {})
        health = player.get("health", 20.0)
        if health < self._prev_player_health:
            reward -= 0.1 * (self._prev_player_health - health) / 20.0
        self._prev_player_health = health

        # Death penalty
        if health <= 0:
            reward -= 1.0

        self._episode_reward += reward
        return reward

    def _reward_basic_survival(self, raw: dict[str, Any]) -> float:
        """Reward shaping for Stage 1: Basic Survival."""
        reward = 0.0
        p = self.progress

        # Wood collection
        if p.wood_collected > 0:
            reward += 0.01 * min(p.wood_collected, 16) / 16.0

        # Stone collection
        if p.stone_collected > 0:
            reward += 0.01 * min(p.stone_collected, 32) / 32.0

        # Food
        if p.food_eaten > 0:
            reward += 0.005 * min(p.food_eaten, 10) / 10.0

        # Surviving first night
        if p.first_night_survived:
            reward += 0.1

        return reward

    def _reward_resource_gathering(self, raw: dict[str, Any]) -> float:
        """Reward shaping for Stage 2: Resource Gathering."""
        reward = 0.0
        p = self.progress

        # Iron
        if p.iron_ingots > 0:
            reward += 0.02 * min(p.iron_ingots, 10) / 10.0

        # Diamonds
        if p.diamonds > 0:
            reward += 0.05 * min(p.diamonds, 3) / 3.0

        # Tools
        if p.has_iron_pickaxe:
            reward += 0.1
        if p.has_bucket:
            reward += 0.05

        return reward

    def _reward_nether_navigation(self, raw: dict[str, Any]) -> float:
        """Reward shaping for Stage 3: Nether Navigation."""
        reward = 0.0
        p = self.progress

        # Portal construction
        if p.portal_built:
            reward += 0.1

        # Entering nether
        if p.entered_nether:
            reward += 0.2

        # Finding fortress
        if p.fortress_found:
            reward += 0.2

        # Blaze rods
        if p.blaze_rods > 0:
            reward += 0.05 * min(p.blaze_rods, 7) / 7.0

        return reward

    def _reward_enderman_hunting(self, raw: dict[str, Any]) -> float:
        """Reward shaping for Stage 4: Enderman Hunting."""
        reward = 0.0
        p = self.progress

        # Ender pearls (main objective)
        if p.ender_pearls > 0:
            reward += 0.05 * min(p.ender_pearls, 12) / 12.0

        # Endermen kills
        if p.endermen_killed > 0:
            reward += 0.01 * min(p.endermen_killed, 15) / 15.0

        return reward

    def _reward_stronghold_finding(self, raw: dict[str, Any]) -> float:
        """Reward shaping for Stage 5: Stronghold Finding."""
        reward = 0.0
        p = self.progress

        # Eyes crafted
        if p.eyes_crafted > 0:
            reward += 0.02 * min(p.eyes_crafted, 12) / 12.0

        # Stronghold found
        if p.stronghold_found:
            reward += 0.2

        # Portal room found
        if p.portal_room_found:
            reward += 0.2

        # Eyes placed
        if p.eyes_placed > 0:
            reward += 0.02 * min(p.eyes_placed, 12) / 12.0

        # Portal activated
        if p.portal_activated:
            reward += 0.3

        return reward

    def _reward_end_fight(self, raw: dict[str, Any]) -> float:
        """Reward shaping for Stage 6: End Fight."""
        reward = 0.0
        p = self.progress
        dragon = raw.get("dragon", {})

        # Entered end
        if p.entered_end:
            reward += 0.1

        # Crystal destruction
        if p.crystals_destroyed > 0:
            reward += 0.05 * p.crystals_destroyed / 10.0

        # Dragon damage
        dragon_health = dragon.get("dragon_health", 200.0)
        if dragon_health < self._prev_dragon_health:
            damage = self._prev_dragon_health - dragon_health
            reward += 0.01 * damage / 10.0
        self._prev_dragon_health = dragon_health

        # Dragon killed (massive reward)
        if p.dragon_killed:
            reward += 10.0

        return reward

    def _check_terminated(self) -> bool:
        """Check if episode should terminate.

        Returns:
            True if episode is complete (success or failure)
        """
        # Death
        raw = self._get_raw_observation()
        player = raw.get("player", {})
        if player.get("health", 20.0) <= 0:
            if self.progress.deaths >= 5:
                return True

        # Stage completion
        if self._check_stage_complete():
            if self.full_speedrun:
                if self.current_stage == StageID.END_FIGHT:
                    return True  # Full speedrun complete
                # Advance to next stage
                self._advance_stage_speedrun()
                return False
            return True

        return False

    def _check_stage_complete(self) -> bool:
        """Check if current stage objectives are met.

        Returns:
            True if stage is complete
        """
        p = self.progress

        if self.current_stage == StageID.BASIC_SURVIVAL:
            return p.wood_collected >= 16 and p.stone_collected >= 32

        if self.current_stage == StageID.RESOURCE_GATHERING:
            return p.has_iron_pickaxe and p.has_bucket and p.iron_ingots >= 3

        if self.current_stage == StageID.NETHER_NAVIGATION:
            return p.entered_nether and p.blaze_rods >= 7

        if self.current_stage == StageID.ENDERMAN_HUNTING:
            return p.ender_pearls >= 12

        if self.current_stage == StageID.STRONGHOLD_FINDING:
            return p.portal_room_found and p.portal_activated

        if self.current_stage == StageID.END_FIGHT:
            return p.dragon_killed

        return False

    def _advance_stage_speedrun(self) -> None:
        """Advance to next stage in full speedrun mode."""
        if self.current_stage.value < StageID.END_FIGHT.value:
            self.current_stage = StageID(self.current_stage.value + 1)
            self._backend_reset(self.current_stage, self._seed)

    def _update_curriculum(self, success: bool) -> None:
        """Update curriculum tracking after episode.

        Args:
            success: Whether the episode succeeded
        """
        stage = self.current_stage
        self._stage_episodes[stage] += 1
        if success:
            self._stage_successes[stage] += 1
        self._stage_rewards[stage].append(self._episode_reward)

        # Check if should advance
        if self._stage_episodes[stage] >= 100:  # Min episodes before advancement
            success_rate = self._stage_successes[stage] / self._stage_episodes[stage]
            if success_rate >= self.curriculum_threshold:
                if self.current_stage.value < StageID.END_FIGHT.value:
                    self.current_stage = StageID(self.current_stage.value + 1)
                    # Reset counters for new stage
                    self._stage_successes[self.current_stage] = 0
                    self._stage_episodes[self.current_stage] = 0
                    self._stage_rewards[self.current_stage] = []

    def _get_info(self) -> dict[str, Any]:
        """Build info dictionary for step/reset return.

        Returns:
            Dictionary with episode metadata
        """
        return {
            "stage": self.current_stage.value,
            "stage_name": self.current_stage.name,
            "steps": self.steps,
            "episode_reward": self._episode_reward,
            "progress": {
                "wood": self.progress.wood_collected,
                "stone": self.progress.stone_collected,
                "iron": self.progress.iron_ingots,
                "diamonds": self.progress.diamonds,
                "blaze_rods": self.progress.blaze_rods,
                "ender_pearls": self.progress.ender_pearls,
                "crystals_destroyed": self.progress.crystals_destroyed,
                "dragon_killed": self.progress.dragon_killed,
            },
            "stage_complete": self._check_stage_complete(),
            "curriculum": {
                "successes": self._stage_successes[self.current_stage],
                "episodes": self._stage_episodes[self.current_stage],
                "success_rate": (
                    self._stage_successes[self.current_stage]
                    / max(1, self._stage_episodes[self.current_stage])
                ),
            }
            if self.curriculum_enabled
            else None,
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for world generation
            options: Reset options including:
                - advance_stage: Force curriculum advancement
                - stage: Override stage for this episode

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            self._seed = seed

        # Handle options
        if options:
            if options.get("advance_stage") and self.curriculum_enabled:
                if self.current_stage.value < StageID.END_FIGHT.value:
                    self.current_stage = StageID(self.current_stage.value + 1)
            if "stage" in options:
                stage = options["stage"]
                if isinstance(stage, int):
                    stage = StageID(stage)
                self.current_stage = stage

        # Reset backend with current stage
        self._backend_reset(self.current_stage, self._seed)

        # Reset episode state
        self.steps = 0
        self.episode_count += 1
        self._episode_reward = 0.0
        self.progress = SpeedrunProgress()
        self.tracker = ProgressTracker(progress=self.progress)
        self._prev_dragon_health = 200.0
        self._prev_player_health = 20.0

        obs = self._get_observation()
        info = self._get_info()
        self._prev_obs = obs.copy()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Discrete action index (0-31)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        action = int(action)
        if not 0 <= action < 32:
            raise ValueError(f"Invalid action {action}, must be in [0, 31]")

        # Apply action to backend
        self._backend_step(action)

        self.steps += 1

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        terminated = self._check_terminated()

        # Check truncation (time limit)
        config = STAGE_CONFIGS.get(self.current_stage, STAGE_CONFIGS[StageID.BASIC_SURVIVAL])
        truncated = self.steps >= min(self.max_steps, config.max_ticks)

        # Update curriculum if episode ended
        if terminated or truncated:
            success = self._check_stage_complete()
            if self.curriculum_enabled:
                self._update_curriculum(success)

        info = self._get_info()
        self._prev_obs = obs.copy()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the environment.

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "rgb_array":
            if self._has_simulator and hasattr(self._simulator, "render"):
                return self._simulator.render()
            # Return placeholder image
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return None

    def close(self) -> None:
        """Clean up resources."""
        if self._simulator is not None:
            if hasattr(self._simulator, "close"):
                self._simulator.close()
            self._simulator = None
            self._has_simulator = False

    def get_curriculum_stats(self) -> dict[str, Any]:
        """Get detailed curriculum statistics.

        Returns:
            Dictionary with per-stage training statistics
        """
        stats = {}
        for stage in StageID:
            episodes = self._stage_episodes[stage]
            successes = self._stage_successes[stage]
            rewards = self._stage_rewards[stage]
            stats[stage.name] = {
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / max(1, episodes),
                "mean_reward": np.mean(rewards) if rewards else 0.0,
                "max_reward": np.max(rewards) if rewards else 0.0,
            }
        stats["current_stage"] = self.current_stage.name
        return stats

    def set_stage(self, stage: StageID | int) -> None:
        """Manually set the current stage.

        Args:
            stage: Stage to switch to
        """
        if isinstance(stage, int):
            stage = StageID(stage)
        self.current_stage = stage


# Factory functions for common configurations
def make_single_stage(stage: int, seed: int | None = None, **kwargs: Any) -> FreeTheEndEnv:
    """Create environment for training a single stage.

    Args:
        stage: Stage number (1-6)
        seed: World seed
        **kwargs: Additional FreeTheEndEnv arguments

    Returns:
        Configured FreeTheEndEnv
    """
    return FreeTheEndEnv(stage=StageID(stage), seed=seed, **kwargs)


def make_curriculum(
    seed: int | None = None, threshold: float = 0.7, **kwargs: Any
) -> FreeTheEndEnv:
    """Create environment with curriculum learning.

    Args:
        seed: World seed
        threshold: Success rate required to advance stages
        **kwargs: Additional FreeTheEndEnv arguments

    Returns:
        Configured FreeTheEndEnv in curriculum mode
    """
    return FreeTheEndEnv(curriculum=True, seed=seed, curriculum_threshold=threshold, **kwargs)


def make_speedrun(seed: int | None = None, **kwargs: Any) -> FreeTheEndEnv:
    """Create environment for full speedrun training.

    Args:
        seed: World seed for reproducible runs
        **kwargs: Additional FreeTheEndEnv arguments

    Returns:
        Configured FreeTheEndEnv in full speedrun mode
    """
    return FreeTheEndEnv(full_speedrun=True, seed=seed, **kwargs)
