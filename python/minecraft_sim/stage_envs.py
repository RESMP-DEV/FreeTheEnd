"""Individual environment classes for each Minecraft speedrun stage.

Each environment provides a focused training scenario with stage-specific:
- Observation space (128-192 floats depending on stage complexity)
- Action space (24-28 discrete actions)
- Reward shaping for the stage objectives
- Episode termination conditions
- Configurable difficulty

These environments use the mc189_core backend for simulation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("gymnasium is required for stage_envs") from exc

from minecraft_sim.curriculum import get_shader_set_for_stage
from minecraft_sim.observations import decode_flat_observation
from minecraft_sim.progression import ProgressTracker
from minecraft_sim.reward_shaping import get_reward_shaper_factory
from minecraft_sim.speedrun_env import Dimension, DimensionState
from minecraft_sim.stage_criteria import get_stage_criteria

# Import mc189_core backend
_mc189_core = None
try:
    import mc189_core as _mc189_core_module

    _mc189_core = _mc189_core_module
except ImportError:
    pass

if _mc189_core is None:
    try:
        import importlib.util

        so_files = list(Path(__file__).parent.glob("mc189_core.cpython-*.so"))
        if so_files:
            spec = importlib.util.spec_from_file_location("mc189_core", so_files[0])
            if spec and spec.loader:
                _mc189_core = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_mc189_core)
    except Exception:
        pass


def _require_backend() -> None:
    """Raise ImportError if mc189_core is not available."""
    logger.debug("_require_backend called")
    if _mc189_core is None:
        raise ImportError(
            "mc189_core module not found. Build the C++ extension: cd cpp/build && cmake .. && make"
        )


class Difficulty(IntEnum):
    """Difficulty presets affecting spawn conditions and penalties."""

    EASY = 1
    NORMAL = 2
    HARD = 3
    HARDCORE = 4


@dataclass
class StageConfig:
    """Configuration for a speedrun stage environment."""

    # Timing
    max_episode_ticks: int = 6000  # 5 minutes at 20 tps
    ticks_per_second: int = 20

    # Difficulty modifiers
    difficulty: Difficulty = Difficulty.NORMAL
    death_penalty: float = -1.0
    tick_penalty: float = -0.0001

    # Spawn conditions
    spawn_protection_ticks: int = 100

    # Rewards scaling
    reward_scale: float = 1.0


class BaseStageEnv(gym.Env, ABC):
    """Abstract base class for all speedrun stage environments.

    Provides common functionality for:
    - Simulator initialization
    - Reset/step protocol
    - Episode tracking
    - Reward computation with stage-specific shaping
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human"]}

    # Subclasses must define these
    STAGE_ID: ClassVar[int]
    STAGE_NAME: ClassVar[str]
    DEFAULT_OBS_SIZE: ClassVar[int]
    DEFAULT_ACTION_SIZE: ClassVar[int]
    DEFAULT_MAX_TICKS: ClassVar[int]

    def __init__(
        self,
        config: StageConfig | None = None,
        shader_dir: str | Path | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize the stage environment.

        Args:
            config: Stage configuration. Uses defaults if None.
            shader_dir: Path to shader directory. Uses default if None.
            render_mode: Rendering mode ("human" or None).
        """
        logger.info("BaseStageEnv.__init__: config=%s, shader_dir=%s, render_mode=%s", config, shader_dir, render_mode)
        super().__init__()
        _require_backend()

        self.config = config or StageConfig(max_episode_ticks=self.DEFAULT_MAX_TICKS)
        self.render_mode = render_mode

        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.DEFAULT_OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.DEFAULT_ACTION_SIZE)

        # Resolve shader directory
        if shader_dir is None:
            shader_dir = Path(__file__).resolve().parents[2] / "cpp" / "shaders"

        # Initialize simulator with stage-specific shader set
        sim_config = _mc189_core.SimulatorConfig()
        sim_config.num_envs = 1
        sim_config.shader_dir = str(shader_dir)
        if hasattr(sim_config, "shader_set"):
            sim_config.shader_set = get_shader_set_for_stage(self.STAGE_ID)
        self._sim = _mc189_core.MC189Simulator(sim_config)

        # Episode tracking
        self._step_count = 0
        self._episode_reward = 0.0
        self._deaths = 0

        # Stage-specific state (initialized in reset)
        self._stage_state: dict[str, Any] = {}

        # Dimension transition state management
        initial_dim = self._get_initial_dimension()
        self._dimension_state = DimensionState(current=initial_dim, previous=initial_dim)

        # Progress tracking across episodes
        self._progress_tracker = ProgressTracker()
        self._last_decoded_observation: dict[str, Any] | None = None
        self._last_progress_snapshot: dict[str, Any] | None = None
        self._last_criteria_success = False
        self._last_criteria_partial = 0.0
        self._last_criteria_optional = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Optional reset options (stage-specific).

        Returns:
            Tuple of (observation, info dict).
        """
        logger.debug("BaseStageEnv.reset called")
        super().reset(seed=seed)

        self._step_count = 0
        self._episode_reward = 0.0
        self._deaths = 0
        self._stage_state = self._initialize_stage_state()
        self._progress_tracker.reset()

        # Reset dimension state to stage-appropriate starting dimension
        initial_dim = self._get_initial_dimension()
        self._dimension_state = DimensionState(current=initial_dim, previous=initial_dim)

        self._sim.reset()
        # Execute no-op to get initial observation
        self._sim.step(np.array([0], dtype=np.int32))

        obs = self._get_observation()
        self._update_progress_from_observation(obs)
        info = self._get_reset_info()

        return obs, info

    def step(
        self, action: int | np.ndarray
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Discrete action index.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        logger.debug("BaseStageEnv.step: action=%s", action)
        action_int = int(action) if isinstance(action, (int, np.integer)) else int(action.item())
        self._step_count += 1

        # Tick dimension cooldown
        self._dimension_state.tick()

        # Execute action
        self._sim.step(np.array([action_int], dtype=np.int32))

        # Get raw outputs
        obs = self._get_observation()
        base_reward = float(self._sim.get_rewards()[0])
        raw_done = bool(self._sim.get_dones()[0])

        # Apply stage-specific reward shaping
        shaped_reward = self._shape_reward(base_reward, obs, action_int)

        # Apply time penalty
        shaped_reward += self.config.tick_penalty

        self._episode_reward += shaped_reward

        # Detect dimension transitions
        transition_info = self._detect_dimension_transition(obs)

        # Update progress tracking
        self._update_progress_from_observation(obs)

        # Check termination conditions
        terminated, truncated = self._check_termination(raw_done)

        info = self._get_step_info(action_int, shaped_reward, terminated, truncated)

        if transition_info:
            info.update(transition_info)

        return obs, shaped_reward, terminated, truncated, info

    def close(self) -> None:
        """Clean up resources."""
        logger.info("BaseStageEnv.close called")
        if hasattr(self, "_sim"):
            self._sim = None

    def get_progress_tracker(self) -> ProgressTracker:
        """Return the progress tracker for this environment.

        Subclasses can call this to access cumulative progression data
        across episodes for curriculum-aware reward shaping or logging.
        """
        logger.debug("BaseStageEnv.get_progress_tracker called")
        return self._progress_tracker

    def _get_observation(self) -> NDArray[np.float32]:
        """Get observation from simulator with stage-specific processing."""
        logger.debug("BaseStageEnv._get_observation called")
        obs = np.array(self._sim.get_observations(), dtype=np.float32).flatten()

        # Pad or truncate to expected size
        if obs.size < self.DEFAULT_OBS_SIZE:
            obs = np.pad(obs, (0, self.DEFAULT_OBS_SIZE - obs.size))
        elif obs.size > self.DEFAULT_OBS_SIZE:
            obs = obs[: self.DEFAULT_OBS_SIZE]

        return obs

    def _check_termination(self, raw_done: bool) -> tuple[bool, bool]:
        """Check if episode should end.

        Args:
            raw_done: Done flag from simulator.

        Returns:
            Tuple of (terminated, truncated).
        """
        # Natural termination (goal achieved or death)
        logger.debug("BaseStageEnv._check_termination: raw_done=%s", raw_done)
        terminated = raw_done or self._check_success()

        # Truncation (time limit)
        truncated = not terminated and self._step_count >= self.config.max_episode_ticks

        return terminated, truncated

    def _build_progress_snapshot(self) -> dict[str, Any]:
        """Build a JSON-serializable snapshot of the current stage progress.

        Converts numpy scalars, sets, and other non-serializable types
        to plain Python values.
        """
        logger.debug("BaseStageEnv._build_progress_snapshot called")
        snapshot: dict[str, Any] = {}
        for key, value in self._stage_state.items():
            if isinstance(value, set):
                snapshot[key] = list(value)
            elif isinstance(value, np.generic):
                snapshot[key] = value.item()
            elif isinstance(value, np.ndarray):
                snapshot[key] = value.tolist()
            else:
                snapshot[key] = value
        if self._last_progress_snapshot is not None:
            snapshot["tracker"] = self._last_progress_snapshot
        return snapshot

    def _update_progress_from_observation(self, obs: NDArray[np.float32]) -> None:
        """Decode observations and update the progress tracker."""
        logger.debug("BaseStageEnv._update_progress_from_observation: obs=%s", obs)
        vector = np.asarray(obs, dtype=np.float32).flatten()
        if vector.size < 256:
            vector = np.pad(vector, (0, 256 - vector.size))
        elif vector.size > 256:
            vector = vector[:256]

        decoded = decode_flat_observation(self.STAGE_ID, vector)
        self._last_decoded_observation = decoded
        self._progress_tracker.update_from_observation(decoded)
        self._last_progress_snapshot = self._progress_tracker.to_snapshot()

    def _get_reset_info(self) -> dict[str, Any]:
        """Get info dict for reset."""
        logger.debug("BaseStageEnv._get_reset_info called")
        return {
            "stage_id": self.STAGE_ID,
            "stage_name": self.STAGE_NAME,
            "difficulty": self.config.difficulty.name,
            "max_ticks": self.config.max_episode_ticks,
            "dimension": self._dimension_state.current.name.lower(),
            "progress_snapshot": self._build_progress_snapshot(),
        }

    def _get_step_info(
        self,
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        """Get info dict for step."""
        logger.debug("BaseStageEnv._get_step_info: action=%s, reward=%s, terminated=%s, truncated=%s", action, reward, terminated, truncated)
        progress = self._build_progress_snapshot()

        info: dict[str, Any] = {
            "step_count": self._step_count,
            "episode_reward": self._episode_reward,
            "progress_snapshot": progress,
        }

        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._step_count,
                "success": self._check_success(),
                "progress": progress,
            }
            logger.info(
                "episode_summary",
                extra={
                    "stage_id": self.STAGE_ID,
                    "stage_name": self.STAGE_NAME,
                    "episode_reward": self._episode_reward,
                    "episode_length": self._step_count,
                    "success": self._check_success(),
                    "progress_snapshot": progress,
                },
            )

        return info

    @abstractmethod
    def _initialize_stage_state(self) -> dict[str, Any]:
        """Initialize stage-specific state. Override in subclasses."""
        logger.info("BaseStageEnv._initialize_stage_state called")
        ...

    @abstractmethod
    def _shape_reward(
        self,
        base_reward: float,
        obs: NDArray[np.float32],
        action: int,
    ) -> float:
        """Apply stage-specific reward shaping. Override in subclasses."""
        logger.debug("BaseStageEnv._shape_reward: base_reward=%s, obs=%s, action=%s", base_reward, obs, action)
        ...

    @abstractmethod
    def _check_success(self) -> bool:
        """Check if stage goal is achieved. Override in subclasses."""
        logger.debug("BaseStageEnv._check_success called")
        ...

    def _build_criteria_snapshot(self) -> dict[str, Any]:
        """Build a state dict compatible with StageCriteria evaluation.

        Subclasses override to map their _stage_state into the format
        expected by StageCriteria lambdas (inventory sub-dict, flags, etc.).
        Default returns the raw progress snapshot.
        """
        logger.debug("BaseStageEnv._build_criteria_snapshot called")
        if self._last_decoded_observation is not None:
            return self._last_decoded_observation
        return self._build_progress_snapshot()

    def _evaluate_stage_criteria(self) -> bool:
        """Evaluate success using the StageCriteria for this stage.

        Builds a criteria-compatible snapshot and checks all required
        conditions defined in STAGE_CRITERIA for this stage.

        Returns:
            True if all required criteria are met, False otherwise.
        """
        logger.debug("BaseStageEnv._evaluate_stage_criteria called")
        criteria = get_stage_criteria(self.STAGE_ID)
        if criteria is None:
            self._last_criteria_success = False
            self._last_criteria_partial = 0.0
            self._last_criteria_optional = 0.0
            return False
        snapshot = self._build_criteria_snapshot()
        self._last_criteria_success = criteria.check_success(snapshot)
        self._last_criteria_partial = criteria.get_partial_progress(snapshot)
        self._last_criteria_optional = criteria.get_optional_progress(snapshot)
        return self._last_criteria_success

    def _get_initial_dimension(self) -> Dimension:
        """Determine the starting dimension for this stage.

        Override in subclasses that start in a non-Overworld dimension.
        """
        logger.info("BaseStageEnv._get_initial_dimension called")
        return Dimension.OVERWORLD

    def _on_dimension_entered(self, from_dim: Dimension, to_dim: Dimension) -> None:
        """Hook called after a dimension transition is detected.

        Subclasses override to reset or initialize stage-specific state
        appropriate for the new dimension. Called after DimensionState is
        already updated.

        Args:
            from_dim: Dimension the player just left.
            to_dim: Dimension the player just entered.
        """
logger.debug("BaseStageEnv._on_dimension_entered: from_dim=%s, to_dim=%s", from_dim, to_dim)

    def _detect_dimension_transition(self, obs: NDArray[np.float32]) -> dict[str, Any] | None:
        """Detect dimension changes from observation and update state.

        Uses the dimension indicator in the observation vector. The exact
        index depends on the stage's observation layout:
        - Stages with 192-float obs: index 12 is the dimension indicator
          (>0.5 = Nether for Stage 3)
        - Stage 6 (64-float obs): always in The End

        Subclasses override to provide stage-specific dimension detection.

        Args:
            obs: Current observation vector.

        Returns:
            Info dict additions if a transition occurred, None otherwise.
        """
        logger.debug("BaseStageEnv._detect_dimension_transition: obs=%s", obs)
        return None


# =============================================================================
# Stage 1: Basic Survival
# =============================================================================


class BasicSurvivalEnv(BaseStageEnv):
    """Stage 1: Basic Survival Environment.

    Goal: Survive, gather wood, make tools, kill mobs.

    The agent spawns in the overworld and must:
    - Survive hostile mobs
    - Mine wood logs
    - Craft wooden tools (pickaxe, sword)
    - Kill zombies and skeletons

    Observation Space:
        128-dimensional float vector containing:
        - Player position, velocity, health, hunger (12)
        - Nearby entity information (40)
        - Nearby block information (40)
        - Inventory state (32)
        - Time/weather (4)

    Action Space:
        24 discrete actions (no portal/eye actions):
        - Movement (8): forward, back, left, right, forward-left, etc.
        - Jump (1)
        - Sprint (1)
        - Sneak (1)
        - Attack (1)
        - Use (1)
        - Hotbar slots (9)
        - Open inventory (1)
        - Craft (1)

    Episode Termination:
        - Success: Iron pickaxe crafted
        - Failure: 10 deaths
        - Truncation: 5 minute timeout (6000 ticks)
    """

    STAGE_ID: ClassVar[int] = 1
    STAGE_NAME: ClassVar[str] = "Basic Survival"
    DEFAULT_OBS_SIZE: ClassVar[int] = 128
    DEFAULT_ACTION_SIZE: ClassVar[int] = 24
    DEFAULT_MAX_TICKS: ClassVar[int] = 6000  # 5 minutes

    # Reward shaping weights
    REWARD_ZOMBIE_KILLED: ClassVar[float] = 0.5
    REWARD_SKELETON_KILLED: ClassVar[float] = 0.5
    REWARD_WOOD_MINED: ClassVar[float] = 0.1
    REWARD_WOODEN_PICKAXE: ClassVar[float] = 0.5
    REWARD_WOODEN_SWORD: ClassVar[float] = 0.5
    REWARD_IRON_PICKAXE: ClassVar[float] = 2.0  # Completion bonus
    REWARD_EXPLORATION: ClassVar[float] = 0.01

    def _initialize_stage_state(self) -> dict[str, Any]:
        """Initialize basic survival tracking state."""
        logger.info("BasicSurvivalEnv._initialize_stage_state called")
        return {
            "zombies_killed": 0,
            "skeletons_killed": 0,
            "wood_mined": 0,
            "has_wooden_pickaxe": False,
            "has_wooden_sword": False,
            "has_iron_pickaxe": False,
            "chunks_explored": set(),
            "last_position": None,
        }

    def _shape_reward(
        self,
        base_reward: float,
        obs: NDArray[np.float32],
        action: int,
    ) -> float:
        """Apply basic survival reward shaping."""
        logger.debug("BasicSurvivalEnv._shape_reward: base_reward=%s, obs=%s, action=%s", base_reward, obs, action)
        reward = base_reward

        # Position-based exploration reward
        if obs.size >= 3:
            pos = (int(obs[0] // 16), int(obs[2] // 16))  # Chunk coordinates
            if pos not in self._stage_state["chunks_explored"]:
                self._stage_state["chunks_explored"].add(pos)
                reward += self.REWARD_EXPLORATION

        # Difficulty scaling
        reward *= self.config.reward_scale

        return reward

    def _check_success(self) -> bool:
        """Check if iron pickaxe has been crafted."""
        logger.debug("BasicSurvivalEnv._check_success called")
        return self._stage_state.get("has_iron_pickaxe", False)


# =============================================================================
# Stage 2: Resource Gathering
# =============================================================================


class ResourceGatheringEnv(BaseStageEnv):
    """Stage 2: Resource Gathering Environment.

    Goal: Mine iron, diamonds, make bucket.

    The agent spawns with wood tools and must:
    - Mine cobblestone and ores
    - Build and use furnaces
    - Smelt iron
    - Craft bucket and collect 10 obsidian

    Observation Space:
        128-dimensional float vector containing:
        - Player position, velocity, health, hunger (12)
        - Y-level (important for ore finding) (1)
        - Nearby block information with ore detection (50)
        - Inventory state with ore counts (40)
        - Tool durability (5)
        - Light level, time (4)
        - Shelter status (2)
        - Padding (14)

    Action Space:
        24 discrete actions:
        - Movement (8)
        - Jump (1)
        - Sprint (1)
        - Sneak (1)
        - Attack/mine (1)
        - Use/place (1)
        - Hotbar slots (9)
        - Open inventory (1)

    # Observation indices for inventory resource counts (within 62-101 range)
    OBS_IDX_IRON_ORE: ClassVar[int] = 62
    OBS_IDX_DIAMOND: ClassVar[int] = 63
    OBS_IDX_OBSIDIAN: ClassVar[int] = 64
        - Craft/smelt (1)

    Episode Termination:
        - Success: Has bucket AND 10+ obsidian
        - Failure: 7 deaths
        - Truncation: 10 minute timeout (12000 ticks)
    """

    STAGE_ID: ClassVar[int] = 2
    STAGE_NAME: ClassVar[str] = "Resource Gathering"
    DEFAULT_OBS_SIZE: ClassVar[int] = 128
    DEFAULT_ACTION_SIZE: ClassVar[int] = 24
    DEFAULT_MAX_TICKS: ClassVar[int] = 12000  # 10 minutes

    # Reward shaping weights
    REWARD_COBBLESTONE: ClassVar[float] = 0.02
    REWARD_COAL: ClassVar[float] = 0.1
    REWARD_IRON_ORE: ClassVar[float] = 0.3
    REWARD_DIAMOND: ClassVar[float] = 1.0
    REWARD_IRON_INGOT: ClassVar[float] = 0.2
    REWARD_BUCKET: ClassVar[float] = 1.0
    REWARD_OBSIDIAN: ClassVar[float] = 0.3
    REWARD_VERTICAL_MINING: ClassVar[float] = 0.05

    def _initialize_stage_state(self) -> dict[str, Any]:
        """Initialize resource gathering tracking state."""
        logger.info("ResourceGatheringEnv._initialize_stage_state called")
        return {
            "cobblestone_mined": 0,
            "iron_ore_mined": 0,
            "diamonds_mined": 0,
            "iron_ingots_smelted": 0,
            "has_bucket": False,
            "obsidian_count": 0,
            "lowest_y_reached": 64,
            "last_y": 64,
        }

    def _shape_reward(
        self,
        base_reward: float,
        obs: NDArray[np.float32],
        action: int,
    ) -> float:
        """Apply resource gathering reward shaping."""
        logger.debug("ResourceGatheringEnv._shape_reward: base_reward=%s, obs=%s, action=%s", base_reward, obs, action)
        reward = base_reward

        # Y-level based reward for going deeper (where ores are)
        if obs.size >= 2:
            current_y = obs[1] if obs.size > 1 else 64
            if current_y < self._stage_state["lowest_y_reached"]:
                self._stage_state["lowest_y_reached"] = current_y
                reward += self.REWARD_VERTICAL_MINING

        reward *= self.config.reward_scale
        return reward

    def _check_success(self) -> bool:
        """Check if bucket and 10 obsidian obtained."""
        logger.debug("ResourceGatheringEnv._check_success called")
        return (
            self._stage_state.get("has_bucket", False)
            and self._stage_state.get("obsidian_count", 0) >= 10
        )


# =============================================================================
# Stage 3: Nether Navigation
# =============================================================================


class NetherNavigationEnv(BaseStageEnv):
    """Stage 3: Nether Navigation Environment.

    Goal: Enter nether, find fortress, get blaze rods.

    The agent spawns at a nether portal (overworld side) and must:
    - Enter the nether dimension
    - Navigate dangerous terrain
    - Find a nether fortress
    - Kill blazes for blaze rods

    Observation Space:
        192-dimensional float vector containing:
        - Player position, velocity, health, hunger (12)
        - Dimension indicator (1)
        - Nether-specific hazards: lava, fire, ghasts (20)
        - Fortress direction/distance estimation (8)
        - Nearby entity information (40)
        - Nearby block information (40)
        - Inventory with blaze rod count (40)
        - Fire resistance status (1)
        - Projectile warnings (10)
        - Portal proximity (2)
        - Padding (18)

    Action Space:
        28 discrete actions (includes portal interaction):
        - Movement (8)
        - Jump (1)
        - Sprint (1)
        - Sneak (1)
        - Attack (1)
        - Use/block (2)
        - Hotbar slots (9)
        - Drop item (1)
        - Swap hands (1)
        - Place/break block (2)
        - Enter portal (1)

    Episode Termination:
        - Success: 7+ blaze rods collected
        - Failure: 5 deaths
        - Truncation: 15 minute timeout (18000 ticks)
    """

    STAGE_ID: ClassVar[int] = 3
    STAGE_NAME: ClassVar[str] = "Nether Navigation"
    DEFAULT_OBS_SIZE: ClassVar[int] = 192
    DEFAULT_ACTION_SIZE: ClassVar[int] = 28
    DEFAULT_MAX_TICKS: ClassVar[int] = 18000  # 15 minutes

    # Reward shaping weights
    REWARD_PORTAL_LIT: ClassVar[float] = 3.0
    REWARD_NETHER_ENTERED: ClassVar[float] = 5.0
    REWARD_FORTRESS_FOUND: ClassVar[float] = 5.0
    REWARD_BLAZE_DAMAGED: ClassVar[float] = 0.1
    REWARD_BLAZE_KILLED: ClassVar[float] = 1.0
    REWARD_BLAZE_ROD: ClassVar[float] = 1.5
    REWARD_GHAST_DEFLECT: ClassVar[float] = 0.5
    REWARD_ESCAPED_LAVA: ClassVar[float] = 0.5
    PENALTY_DEATH: ClassVar[float] = -2.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment and instantiate Stage 3 reward shaper."""
        logger.debug("NetherNavigationEnv.reset called")
        factory = get_reward_shaper_factory(3)
        self._reward_shaper = factory()
        return super().reset(seed=seed, options=options)

    def _initialize_stage_state(self) -> dict[str, Any]:
        """Initialize nether navigation tracking state."""
        logger.info("NetherNavigationEnv._initialize_stage_state called")
        return {
            "in_nether": False,
            "portal_lit": False,
            "fortress_found": False,
            "blazes_killed": 0,
            "blaze_rods": 0,
            "ghasts_deflected": 0,
            "lava_escapes": 0,
            "deaths": 0,
        }

    def _build_snapshot(self, obs: NDArray[np.float32]) -> dict[str, Any]:
        """Build a state snapshot from the observation vector for the reward shaper.

        Maps observation indices to semantic keys expected by the Stage 3 shaper:
        - obs[0:3]: player position (x, y, z)
        - obs[6]: health
        - obs[7]: hunger
        - obs[12]: dimension indicator (>0.5 = nether)
        - obs[13]: in_lava flag
        - obs[14]: fire_ticks
        - obs[33]: fortress proximity indicator
        - obs[34]: in_fortress flag
        - obs[35]: distance_to_fortress
        - obs[45]: blaze_seen
        - obs[46]: blaze_spawner_found
        - obs[47]: blazes_killed count
        - obs[90]: blaze_rod count
        - obs[121]: fire_resistance status
        - obs[132]: portal proximity
        """
        logger.debug("NetherNavigationEnv._build_snapshot: obs=%s", obs)
        state = self._stage_state
        snapshot: dict[str, Any] = {
            "health": float(obs[6]) if obs.size > 6 else 20.0,
            "hunger": float(obs[7]) if obs.size > 7 else 20.0,
            "x_position": float(obs[0]) if obs.size > 0 else 0.0,
            "y_position": float(obs[1]) if obs.size > 1 else 64.0,
            "z_position": float(obs[2]) if obs.size > 2 else 0.0,
        }

        # Dimension indicator
        in_nether = bool(obs.size > 12 and obs[12] > 0.5)
        if in_nether and not self._stage_state["in_nether"]:
            self._stage_state["in_nether"] = True
        snapshot["in_nether"] = in_nether or self._stage_state["in_nether"]
        snapshot["entered_nether"] = in_nether or self._stage_state["in_nether"]

        # Fire/lava status
        if obs.size > 13:
            snapshot["in_lava"] = bool(obs[13] > 0.5)
            snapshot["fire_ticks"] = float(obs[14]) if obs.size > 14 else 0.0

        # Fire resistance
        if obs.size > 121:
            snapshot["has_fire_resistance"] = bool(obs[121] > 0.5)

        # Portal lit detection from portal proximity obs
        snapshot["portal_lit"] = state.get("portal_lit", False)
        snapshot["portal_frame_placed"] = state.get("portal_lit", False)
        snapshot["portal_built"] = state.get("portal_lit", False)
        if obs.size > 132 and obs[132] > 0.5 and not state.get("portal_lit", False):
            state["portal_lit"] = True
            snapshot["portal_lit"] = True
            snapshot["portal_frame_placed"] = True
            snapshot["portal_built"] = True

        # Fortress detection
        fortress_found = state.get("fortress_found", False)
        if obs.size > 33 and obs[33] > 0.5:
            fortress_found = True
            state["fortress_found"] = True
        snapshot["fortress_found"] = fortress_found
        snapshot["fortress_visible"] = fortress_found
        snapshot["in_fortress"] = fortress_found and obs.size > 34 and obs[34] > 0.5

        # Fortress distance
        if obs.size > 35:
            snapshot["distance_to_fortress"] = float(obs[35])

        # Blaze rod count from inventory encoding
        blaze_rods = state.get("blaze_rods", 0)
        if obs.size > 90:
            obs_blaze_rods = int(max(0, obs[90]))
            if obs_blaze_rods > blaze_rods:
                blaze_rods = obs_blaze_rods
                state["blaze_rods"] = blaze_rods
        snapshot["inventory"] = {"blaze_rod": blaze_rods, "nether_wart": 0}

        # Blaze sighting and spawner
        if obs.size > 45:
            snapshot["blaze_seen"] = bool(obs[45] > 0.5)
            snapshot["blaze_spawner_found"] = bool(obs[46] > 0.5) if obs.size > 46 else False

        # Blazes killed
        blazes_killed = state.get("blazes_killed", 0)
        if obs.size > 47:
            obs_kills = int(max(0, obs[47]))
            if obs_kills > blazes_killed:
                blazes_killed = obs_kills
                state["blazes_killed"] = blazes_killed
        snapshot["blazes_killed"] = blazes_killed

        return snapshot

    def _shape_reward(
        self,
        base_reward: float,
        obs: NDArray[np.float32],
        action: int,
    ) -> float:
        """Apply nether navigation reward shaping via Stage 3 reward shaper."""
        logger.debug("NetherNavigationEnv._shape_reward: base_reward=%s, obs=%s, action=%s", base_reward, obs, action)
        reward = base_reward

        # Check dimension transition for stage state tracking
        if obs.size >= 13 and not self._stage_state["in_nether"]:
            if obs[12] > 0.5:
                self._stage_state["in_nether"] = True
                reward += self.REWARD_NETHER_ENTERED

        # Build snapshot and feed to reward shaper for milestone tracking
        snapshot = self._build_snapshot(obs)
        shaped = self._reward_shaper(snapshot)
        reward += shaped

        reward *= self.config.reward_scale
        return reward

    def _detect_dimension_transition(self, obs: NDArray[np.float32]) -> dict[str, Any] | None:
        """Detect Overworld-to-Nether and Nether-to-Overworld transitions.

        Uses obs[12] as the dimension indicator (>0.5 = Nether).
        Preserves player position per dimension and manages portal cooldown.

        Args:
            obs: Current 192-float observation vector.

        Returns:
            Info dict if transition occurred, None otherwise.
        """
        logger.debug("NetherNavigationEnv._detect_dimension_transition: obs=%s", obs)
        if obs.size <= 12:
            return None

        dim_state = self._dimension_state
        observed_nether = obs[12] > 0.5
        observed_dim = Dimension.NETHER if observed_nether else Dimension.OVERWORLD

        if observed_dim == dim_state.current:
            return None

        if dim_state.in_cooldown:
            return None

        from_dim = dim_state.current
        to_dim = observed_dim

        # Save position in departing dimension
        if obs.size >= 3:
            pos = (float(obs[0]), float(obs[1]), float(obs[2]))
            dim_state.save_position(from_dim, pos)

        dim_state.record_transition(self._step_count, from_dim, to_dim)

        # Restore position if returning to previously-visited dimension
        visited_before = any(t[2] == to_dim for t in dim_state.transitions[:-1])
        if visited_before:
            saved = dim_state.get_saved_position(to_dim)
            obs[0], obs[1], obs[2] = saved

        # Notify subclass hook
        self._on_dimension_entered(from_dim, to_dim)

        info: dict[str, Any] = {
            "dimension_transition": True,
            "from_dimension": from_dim.name.lower(),
            "to_dimension": to_dim.name.lower(),
            "transition_tick": self._step_count,
        }

        if from_dim == Dimension.OVERWORLD and to_dim == Dimension.NETHER:
            info["entered_nether"] = True
        elif from_dim == Dimension.NETHER and to_dim == Dimension.OVERWORLD:
            info["returned_overworld"] = True

        return info

    def _on_dimension_entered(self, from_dim: Dimension, to_dim: Dimension) -> None:
        """Reset stage state appropriate for the new dimension.

        On entering the Nether: marks in_nether, resets fortress tracking
        if this is a fresh entry. On returning to Overworld: preserves
        collected blaze rods.
        """
        logger.debug("NetherNavigationEnv._on_dimension_entered: from_dim=%s, to_dim=%s", from_dim, to_dim)
        if to_dim == Dimension.NETHER:
            self._stage_state["in_nether"] = True
        elif to_dim == Dimension.OVERWORLD:
            # Returning from Nether; blaze_rods and blazes_killed persist
            self._stage_state["in_nether"] = False

    def _check_success(self) -> bool:
        """Check if 7+ blaze rods collected and fortress found."""
        logger.debug("NetherNavigationEnv._check_success called")
        fortress_found = self._stage_state.get("fortress_found", False)
        blaze_rods = self._stage_state.get("blaze_rods", 0)
        return bool(fortress_found) and blaze_rods >= 7


# =============================================================================
# Stage 4: Enderman Hunting
# =============================================================================


class EndermanHuntingEnv(BaseStageEnv):
    """Stage 4: Enderman Hunting Environment.

    Goal: Collect ender pearls from endermen.

    The agent spawns in overworld at night or warped forest and must:
    - Locate endermen
    - Use safe hunting techniques (water traps, low ceilings)
    - Kill endermen without taking excessive damage
    - Collect 12+ ender pearls

    Observation Space:
        192-dimensional float vector containing:
        - Player position, velocity, health, hunger (12)
        - Enderman detection: count, positions, aggro state (40)
        - Looking direction vs enderman eye-line (8)
        - Water/trap placement nearby (10)
        - Ceiling height (1)
        - Nearby entity information (30)
        - Nearby block information (30)
        - Inventory with pearl count (30)
        - Equipment status (10)
        - Pumpkin helmet status (1)
        - Damage source tracking (10)
        - Padding (10)

    Action Space:
        28 discrete actions:
        - Movement (8)
        - Jump (1)
        - Sprint (1)
        - Sneak (1)
        - Look at enderman (1)
        - Look away (1)
        - Attack (1)
        - Use/block (2)
        - Hotbar slots (9)
        - Equip helmet (1)
        - Place block/water (2)

    Episode Termination:
        - Success: 12+ ender pearls collected
        - Failure: 5 deaths
        - Truncation: 10 minute timeout (12000 ticks)
    """

    STAGE_ID: ClassVar[int] = 4
    STAGE_NAME: ClassVar[str] = "Enderman Hunting"
    DEFAULT_OBS_SIZE: ClassVar[int] = 192
    DEFAULT_ACTION_SIZE: ClassVar[int] = 28
    DEFAULT_MAX_TICKS: ClassVar[int] = 12000  # 10 minutes

    # Reward shaping weights
    REWARD_ENDERMAN_DAMAGED: ClassVar[float] = 0.1
    REWARD_ENDERMAN_KILLED: ClassVar[float] = 1.5
    REWARD_ENDER_PEARL: ClassVar[float] = 2.0
    REWARD_WATER_TRAP: ClassVar[float] = 0.5
    REWARD_LOW_CEILING_TRAP: ClassVar[float] = 0.5
    REWARD_SAFE_ENGAGEMENT: ClassVar[float] = 0.3
    REWARD_PUMPKIN_EQUIPPED: ClassVar[float] = 0.3
    PENALTY_DAMAGE_TAKEN: ClassVar[float] = -0.2

    def _initialize_stage_state(self) -> dict[str, Any]:
        """Initialize enderman hunting tracking state."""
        logger.info("EndermanHuntingEnv._initialize_stage_state called")
        return {
            "endermen_killed": 0,
            "ender_pearls": 0,
            "traps_built": 0,
            "safe_kills": 0,
            "damage_taken_from_endermen": 0,
            "wearing_pumpkin": False,
        }

    def _shape_reward(
        self,
        base_reward: float,
        obs: NDArray[np.float32],
        action: int,
    ) -> float:
        """Apply enderman hunting reward shaping."""
        logger.debug("EndermanHuntingEnv._shape_reward: base_reward=%s, obs=%s, action=%s", base_reward, obs, action)
        reward = base_reward

        # Reward efficiency (pearls per time)
        time_fraction = self._step_count / self.config.max_episode_ticks
        if self._stage_state["ender_pearls"] > 0 and time_fraction > 0:
            efficiency = self._stage_state["ender_pearls"] / time_fraction
            reward += efficiency * 0.01

        reward *= self.config.reward_scale
        return reward

    def _check_success(self) -> bool:
        """Check if 12+ ender pearls collected."""
        logger.debug("EndermanHuntingEnv._check_success called")
        return self._stage_state.get("ender_pearls", 0) >= 12


# =============================================================================
# Stage 5: Stronghold Finding
# =============================================================================


class StrongholdFindingEnv(BaseStageEnv):
    """Stage 5: Stronghold Finding Environment.

    Goal: Find stronghold, activate end portal.

    The agent spawns with eyes of ender and must:
    - Use triangulation to locate stronghold
    - Dig down to stronghold
    - Navigate to portal room
    - Activate end portal with eyes

    Observation Space:
        192-dimensional float vector containing:
        - Player position, velocity, health, hunger (12)
        - Eye trajectory tracking (last throw direction) (6)
        - Triangulation waypoints (12)
        - Estimated stronghold position (3)
        - Y-level (1)
        - Stronghold proximity indicators (8)
        - In stronghold flag (1)
        - Rooms visited flags (8)
        - Portal frame status (12 slots) (12)
        - Nearby block information (40)
        - Nearby entity information (30)
        - Inventory with eye count (30)
        - Equipment status (10)
        - Biome/time (4)
        - Padding (23)

    Action Space:
        28 discrete actions (includes eye throwing):
        - Movement (8)
        - Jump (1)
        - Sprint (1)
        - Sneak (1)
        - Throw eye (1)
        - Track eye trajectory (1)
        - Mark waypoint (1)
        - Attack/break (2)
        - Use/place (2)
        - Hotbar slots (9)
        - Open inventory (1)

    Episode Termination:
        - Success: End portal activated
        - Failure: 3 deaths or run out of eyes
        - Truncation: 15 minute timeout (18000 ticks)
    """

    STAGE_ID: ClassVar[int] = 5
    STAGE_NAME: ClassVar[str] = "Stronghold Finding"
    DEFAULT_OBS_SIZE: ClassVar[int] = 192
    DEFAULT_ACTION_SIZE: ClassVar[int] = 28
    DEFAULT_MAX_TICKS: ClassVar[int] = 18000  # 15 minutes

    # Reward shaping weights
    REWARD_FIRST_EYE_THROW: ClassVar[float] = 1.0
    REWARD_TRIANGULATION_TRAVEL: ClassVar[float] = 0.01
    REWARD_SECOND_EYE_THROW: ClassVar[float] = 1.5
    REWARD_INTERSECTION_FOUND: ClassVar[float] = 2.0
    REWARD_EYE_DROPS: ClassVar[float] = 1.0
    REWARD_STRONGHOLD_ENTERED: ClassVar[float] = 2.0
    REWARD_PORTAL_ROOM_FOUND: ClassVar[float] = 5.0
    REWARD_EYE_PLACED: ClassVar[float] = 0.5
    REWARD_PORTAL_ACTIVE: ClassVar[float] = 10.0
    PENALTY_EYE_LOST: ClassVar[float] = -0.5

    def _initialize_stage_state(self) -> dict[str, Any]:
        """Initialize stronghold finding tracking state."""
        logger.info("StrongholdFindingEnv._initialize_stage_state called")
        return {
            "eyes_thrown": 0,
            "pearls_thrown": 0,
            "eyes_remaining": 12,
            "triangulation_points": [],
            "estimated_stronghold": None,
            "stronghold_distance": float("inf"),
            "in_stronghold": False,
            "portal_room_found": False,
            "eyes_placed": 0,
            "portal_active": False,
        }

    def _shape_reward(
        self,
        base_reward: float,
        obs: NDArray[np.float32],
        action: int,
    ) -> float:
        """Apply stronghold finding reward shaping."""
        logger.debug("StrongholdFindingEnv._shape_reward: base_reward=%s, obs=%s, action=%s", base_reward, obs, action)
        reward = base_reward

        # Reward approaching estimated stronghold position
        if self._stage_state["estimated_stronghold"] is not None:
            # Distance-based shaping would go here
            pass

        reward *= self.config.reward_scale
        # Clamp to policy bounds: death penalty from backend (-50) must not
        # exceed the stage policy bound of -10.0
        reward = max(reward, -10.0)
        return reward

    def _detect_dimension_transition(self, obs: NDArray[np.float32]) -> dict[str, Any] | None:
        """Detect Overworld-to-End transition when the end portal is entered.

        Uses obs[12] as the dimension indicator (values: 0.0=Overworld,
        0.5=Nether, >=0.75=End). The transition only fires after
        portal_active is set in stage state.

        Args:
            obs: Current 192-float observation vector.

        Returns:
            Info dict if transition occurred, None otherwise.
        """
        logger.debug("StrongholdFindingEnv._detect_dimension_transition: obs=%s", obs)
        if obs.size <= 12:
            return None

        dim_state = self._dimension_state
        obs_val = float(obs[12])

        # Map obs value to dimension (same encoding as Stage 3)
        if obs_val >= 0.75:
            observed_dim = Dimension.END
        elif obs_val >= 0.25:
            observed_dim = Dimension.NETHER
        else:
            observed_dim = Dimension.OVERWORLD

        if observed_dim == dim_state.current:
            return None

        if dim_state.in_cooldown:
            return None

        from_dim = dim_state.current
        to_dim = observed_dim

        # Save position in departing dimension
        if obs.size >= 3:
            pos = (float(obs[0]), float(obs[1]), float(obs[2]))
            dim_state.save_position(from_dim, pos)

        dim_state.record_transition(self._step_count, from_dim, to_dim)

        # Restore position if returning to previously-visited dimension
        visited_before = any(t[2] == to_dim for t in dim_state.transitions[:-1])
        if visited_before:
            saved = dim_state.get_saved_position(to_dim)
            obs[0], obs[1], obs[2] = saved

        self._on_dimension_entered(from_dim, to_dim)

        info: dict[str, Any] = {
            "dimension_transition": True,
            "from_dimension": from_dim.name.lower(),
            "to_dimension": to_dim.name.lower(),
            "transition_tick": self._step_count,
        }

        if from_dim == Dimension.OVERWORLD and to_dim == Dimension.END:
            info["entered_end"] = True
        elif from_dim == Dimension.END and to_dim == Dimension.OVERWORLD:
            info["returned_overworld"] = True

        return info

    def _on_dimension_entered(self, from_dim: Dimension, to_dim: Dimension) -> None:
        """Handle state adjustments on dimension entry.

        Entering The End marks the portal as activated (success condition).
        """
        logger.debug("StrongholdFindingEnv._on_dimension_entered: from_dim=%s, to_dim=%s", from_dim, to_dim)
        if to_dim == Dimension.END:
            self._stage_state["portal_active"] = True

    def _check_success(self) -> bool:
        """Check if end portal is activated and entered."""
        logger.debug("StrongholdFindingEnv._check_success called")
        return self._stage_state.get("portal_active", False)


# Alias for curriculum manager compatibility
StrongholdHuntEnv = StrongholdFindingEnv


# =============================================================================
# Stage 6: Dragon Fight (Enhanced)
# =============================================================================


class DragonFightEnv(BaseStageEnv):
    """Stage 6: Dragon Fight Environment (Enhanced).

    Goal: Defeat the Ender Dragon.

    The agent spawns on the obsidian platform in The End and must:
    - Destroy end crystals (prioritize caged ones)
    - Avoid dragon breath and knockback
    - Damage dragon during perching phases
    - Optionally use bed explosion strategy for one-cycle

    Observation Space:
        64-dimensional float vector containing:
        - Player position, velocity, health, hunger (12)
        - Dragon position, health, phase (8)
        - Crystal positions and status (12)
        - Breath cloud positions (6)
        - Void proximity (1)
        - Tower positions for navigation (10)
        - Exit portal status (2)
        - Equipment/inventory (8)
        - Bed placement state (2)
        - Padding (3)

    Action Space:
        20 discrete actions (includes bed placement):
        - Movement (8)
        - Jump (1)
        - Sprint (1)
        - Sneak (1)
        - Attack (1)
        - Shoot bow (1)
        - Aim bow (1)
        - Block (1)
        - Throw pearl (1)
        - Place water (1)
        - Place bed (1) - NEW
        - Use item (1)
        - Pillar up (1)

    Episode Termination:
        - Success: Dragon killed
        - Failure: Death (hardcore) or fell in void
        - Truncation: 30 minute timeout (36000 ticks)

    One-Cycle Strategy:
        Beds explode in The End. 7 beds placed near dragon head during
        perch can kill it in one cycle. This requires precise timing.
    """

    STAGE_ID: ClassVar[int] = 6
    STAGE_NAME: ClassVar[str] = "Dragon Fight"
    DEFAULT_OBS_SIZE: ClassVar[int] = 64
    DEFAULT_ACTION_SIZE: ClassVar[int] = 20
    DEFAULT_MAX_TICKS: ClassVar[int] = 36000  # 30 minutes

    # Reward shaping weights
    REWARD_SURVIVE_SPAWN: ClassVar[float] = 1.0
    REWARD_REACH_MAIN_ISLAND: ClassVar[float] = 2.0
    REWARD_DESTROY_CRYSTAL: ClassVar[float] = 3.0
    REWARD_DESTROY_CAGED_CRYSTAL: ClassVar[float] = 5.0
    REWARD_ALL_CRYSTALS_DESTROYED: ClassVar[float] = 10.0
    REWARD_DAMAGE_DRAGON: ClassVar[float] = 0.5
    REWARD_DAMAGE_DRAGON_PERCHING: ClassVar[float] = 1.0
    REWARD_BED_DAMAGE: ClassVar[float] = 3.0  # Bed explosion damage
    REWARD_DRAGON_KILLED: ClassVar[float] = 50.0
    REWARD_ENTER_EXIT_PORTAL: ClassVar[float] = 10.0
    PENALTY_VOID_DEATH: ClassVar[float] = -10.0
    PENALTY_DEATH: ClassVar[float] = -5.0

    # Dragon phases
    PHASE_CIRCLING: ClassVar[int] = 0
    PHASE_STRAFING: ClassVar[int] = 1
    PHASE_PERCHING: ClassVar[int] = 2
    PHASE_BREATH: ClassVar[int] = 3

    def _get_initial_dimension(self) -> Dimension:
        """Dragon fight always starts in The End."""
        logger.info("DragonFightEnv._get_initial_dimension called")
        return Dimension.END

    def _detect_dimension_transition(self, obs: NDArray[np.float32]) -> dict[str, Any] | None:
        """Detect End-to-Overworld transition (dragon death / exit portal).

        The exit portal activates when dragon_fight_complete (obs index
        within the 64-float layout). When the agent enters the exit portal,
        the episode effectively ends.

        Args:
            obs: Current 64-float observation vector.

        Returns:
            Info dict if dragon killed and portal entered, None otherwise.
        """
        logger.debug("DragonFightEnv._detect_dimension_transition: obs=%s", obs)
        dim_state = self._dimension_state

        # Dragon fight uses a compact 64-float obs. Exit portal detection
        # uses the dragon_killed stage state; if the agent enters the exit
        # portal the simulator will signal done.
        if self._stage_state.get("dragon_killed", False) and dim_state.current == Dimension.END:
            # Transition to overworld when exit portal is used (raw_done handles this)
            # We record the transition for state consistency
            dim_state.record_transition(self._step_count, Dimension.END, Dimension.OVERWORLD)
            self._on_dimension_entered(Dimension.END, Dimension.OVERWORLD)
            return {
                "dimension_transition": True,
                "from_dimension": "end",
                "to_dimension": "overworld",
                "transition_tick": self._step_count,
                "dragon_fight_complete": True,
            }

        return None

    def _on_dimension_entered(self, from_dim: Dimension, to_dim: Dimension) -> None:
        """Handle End-to-Overworld transition after dragon death.

        Marks the fight as complete when the player returns to Overworld
        via the exit portal.
        """
        logger.debug("DragonFightEnv._on_dimension_entered: from_dim=%s, to_dim=%s", from_dim, to_dim)
        if from_dim == Dimension.END and to_dim == Dimension.OVERWORLD:
            self._stage_state["dragon_killed"] = True

    def __init__(
        self,
        config: StageConfig | None = None,
        shader_dir: str | Path | None = None,
        render_mode: str | None = None,
        enable_one_cycle: bool = True,
    ) -> None:
        """Initialize dragon fight environment.

        Args:
            config: Stage configuration.
            shader_dir: Path to shaders.
            render_mode: Rendering mode.
            enable_one_cycle: Enable bed explosion mechanics for one-cycle.
        """
        logger.info("DragonFightEnv.__init__: config=%s, shader_dir=%s, render_mode=%s, enable_one_cycle=%s", config, shader_dir, render_mode, enable_one_cycle)
        super().__init__(config, shader_dir, render_mode)
        self.enable_one_cycle = enable_one_cycle

        # Extend observation space if beds are enabled
        if enable_one_cycle:
            # Add bed inventory and placement state
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(64,),  # Extended for bed mechanics
                dtype=np.float32,
            )

    def _initialize_stage_state(self) -> dict[str, Any]:
        """Initialize dragon fight tracking state."""
        logger.info("DragonFightEnv._initialize_stage_state called")
        return {
            "on_main_island": False,
            "crystals_destroyed": 0,
            "caged_crystals_destroyed": 0,
            "total_crystals": 10,
            "caged_crystals": 2,
            "dragon_damage_dealt": 0.0,
            "dragon_health": 200.0,
            "dragon_phase": self.PHASE_CIRCLING,
            "dragon_killed": False,
            "beds_used": 0,
            "beds_in_inventory": 1,
            "perch_damage_dealt": 0.0,
            "void_proximity_warnings": 0,
        }

    def _shape_reward(
        self,
        base_reward: float,
        obs: NDArray[np.float32],
        action: int,
    ) -> float:
        """Apply dragon fight reward shaping."""
        logger.debug("DragonFightEnv._shape_reward: base_reward=%s, obs=%s, action=%s", base_reward, obs, action)
        reward = base_reward
        state = self._stage_state

        # Reward crystal destruction progress
        if state["crystals_destroyed"] == state["total_crystals"]:
            if not state.get("all_crystals_rewarded", False):
                reward += self.REWARD_ALL_CRYSTALS_DESTROYED
                state["all_crystals_rewarded"] = True

        # Extra reward for damage during perch (optimal strategy)
        if state["dragon_phase"] == self.PHASE_PERCHING:
            if action == 5:  # Attack action (assumed)
                reward += self.REWARD_DAMAGE_DRAGON_PERCHING * 0.1

        # Bed explosion bonus if enabled
        if self.enable_one_cycle and action == 17:  # Place bed action
            if state["dragon_phase"] == self.PHASE_PERCHING:
                reward += self.REWARD_BED_DAMAGE

        reward *= self.config.reward_scale
        return reward

    def _check_success(self) -> bool:
        """Check if dragon is killed."""
        logger.debug("DragonFightEnv._check_success called")
        return self._stage_state.get("dragon_killed", False)


# =============================================================================
# Factory functions
# =============================================================================


def make_stage_env(
    stage: int,
    config: StageConfig | None = None,
    **kwargs: Any,
) -> BaseStageEnv:
    """Factory function to create a stage environment.

    Args:
        stage: Stage number (1-6).
        config: Optional stage configuration.
        **kwargs: Additional arguments passed to environment constructor.

    Returns:
        The stage environment instance.

    Raises:
        ValueError: If stage number is invalid.
    """
    logger.debug("make_stage_env: stage=%s, config=%s", stage, config)
    stage_classes: dict[int, type[BaseStageEnv]] = {
        1: BasicSurvivalEnv,
        2: ResourceGatheringEnv,
        3: NetherNavigationEnv,
        4: EndermanHuntingEnv,
        5: StrongholdFindingEnv,
        6: DragonFightEnv,
    }

    if stage not in stage_classes:
        raise ValueError(f"Invalid stage {stage}. Must be 1-6.")

    return stage_classes[stage](config=config, **kwargs)


def get_stage_info(stage: int) -> dict[str, Any]:
    """Get information about a speedrun stage.

    Args:
        stage: Stage number (1-6).

    Returns:
        Dictionary with stage metadata.
    """
    logger.debug("get_stage_info: stage=%s", stage)
    stage_info: dict[int, dict[str, Any]] = {
        1: {
            "name": "Basic Survival",
            "description": "Survive, gather wood, make tools, kill mobs",
            "obs_size": 128,
            "action_size": 24,
            "max_ticks": 6000,
            "goal": "Craft iron pickaxe",
        },
        2: {
            "name": "Resource Gathering",
            "description": "Mine iron, diamonds, make bucket",
            "obs_size": 128,
            "action_size": 24,
            "max_ticks": 12000,
            "goal": "Bucket + 10 obsidian",
        },
        3: {
            "name": "Nether Navigation",
            "description": "Enter nether, find fortress, get blaze rods",
            "obs_size": 192,
            "action_size": 28,
            "max_ticks": 18000,
            "goal": "7+ blaze rods",
        },
        4: {
            "name": "Enderman Hunting",
            "description": "Collect ender pearls from endermen",
            "obs_size": 192,
            "action_size": 28,
            "max_ticks": 12000,
            "goal": "12+ ender pearls",
        },
        5: {
            "name": "Stronghold Finding",
            "description": "Find stronghold, activate end portal",
            "obs_size": 192,
            "action_size": 28,
            "max_ticks": 18000,
            "goal": "End portal activated",
        },
        6: {
            "name": "Dragon Fight",
            "description": "Defeat the Ender Dragon",
            "obs_size": 64,
            "action_size": 20,
            "max_ticks": 36000,
            "goal": "Dragon killed",
        },
    }

    if stage not in stage_info:
        raise ValueError(f"Invalid stage {stage}. Must be 1-6.")

    return stage_info[stage]


__all__ = [
    # Environments
    "BasicSurvivalEnv",
    "ResourceGatheringEnv",
    "NetherNavigationEnv",
    "EndermanHuntingEnv",
    "StrongholdFindingEnv",
    "StrongholdHuntEnv",
    "DragonFightEnv",
    # Base class and config
    "BaseStageEnv",
    "StageConfig",
    "Difficulty",
    # Factory functions
    "make_stage_env",
    "get_stage_info",
]
