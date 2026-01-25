"""
Minecraft RL Simulator - Fast C++ simulation with Python bindings.

This package provides a Gymnasium-compatible Minecraft-like environment
for reinforcement learning research, with support for vectorized
environments and efficient batch operations.

Example:
    >>> import minecraft_sim
    >>> env = minecraft_sim.DragonFightEnv()
    >>> obs, info = env.reset()
    >>> for _ in range(100):
    ...     action = env.action_space.sample()
    ...     obs, reward, terminated, truncated, info = env.step(action)
    ...     if terminated or truncated:
    ...         obs, info = env.reset()

    >>> # Vectorized environments
    >>> from minecraft_sim import VecDragonFightEnv
    >>> vec_env = VecDragonFightEnv(num_envs=64)
    >>> obs = vec_env.reset()
    >>> actions = np.random.randint(0, 17, size=64)
    >>> obs, rewards, dones, infos = vec_env.step(actions)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# =============================================================================
# Load C++ extension module directly (avoid circular imports with `from . import`)
# =============================================================================

_core = None
_HAS_CPP_MODULE = False

# Try direct import first (when mc189_core.so is in sys.path or PYTHONPATH)
try:
    import mc189_core as _direct_core

    _core = _direct_core
    _HAS_CPP_MODULE = True
except ImportError:
    pass

# Try loading from same directory as this __init__.py
if _core is None:
    try:
        so_files = list(Path(__file__).parent.glob("mc189_core.cpython-*.so"))
        if so_files:
            spec = importlib.util.spec_from_file_location("mc189_core", so_files[0])
            if spec and spec.loader:
                _core = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_core)
                _HAS_CPP_MODULE = True
    except Exception:
        pass

# =============================================================================
# Export C++ classes and constants
# =============================================================================

if _HAS_CPP_MODULE and _core is not None:
    # Core classes
    MC189Simulator = _core.MC189Simulator
    SimulatorConfig = _core.SimulatorConfig
    VulkanContext = _core.VulkanContext
    VulkanContextConfig = _core.VulkanContextConfig
    ComputePipeline = _core.ComputePipeline
    Buffer = _core.Buffer
    BufferManager = _core.BufferManager
    BatchExecutor = _core.BatchExecutor

    # Aliases
    MinecraftSimulator = MC189Simulator
    VecMinecraftSimulator = MC189Simulator

    # Optional attributes
    ActionType = getattr(_core, "ActionType", None)
    Dimension = getattr(_core, "Dimension", None)
    MC189Action = getattr(_core, "MC189Action", None)
    MC189Dimension = getattr(_core, "MC189Dimension", None)
    GymSpaceHelper = getattr(_core, "GymSpaceHelper", None)
    BatchUtils = getattr(_core, "BatchUtils", None)
    RunningMeanStd = getattr(_core, "RunningMeanStd", None)
    discrete_to_input = getattr(_core, "discrete_to_input", None)
    check_gpu = getattr(_core, "check_gpu", None)

    # Constants
    OBSERVATION_SIZE = getattr(_core, "OBSERVATION_SIZE", 48)
    ACTION_SIZE = getattr(_core, "ACTION_SIZE", 17)
    MAX_BATCH_SIZE = getattr(_core, "MAX_BATCH_SIZE", 4096)
    TICKS_PER_SECOND = getattr(_core, "TICKS_PER_SECOND", 20)
    MC189_ACTION_COUNT = getattr(_core, "MC189_ACTION_COUNT", 17)

    # Expose as mc189_core for backward compatibility
    mc189_core = _core
else:
    # Fallback when C++ module not available
    MinecraftSimulator = None
    VecMinecraftSimulator = None
    MC189Simulator = None
    SimulatorConfig = None
    VulkanContext = None
    VulkanContextConfig = None
    ComputePipeline = None
    Buffer = None
    BufferManager = None
    BatchExecutor = None
    ActionType = None
    Dimension = None
    MC189Action = None
    MC189Dimension = None
    GymSpaceHelper = None
    BatchUtils = None
    RunningMeanStd = None
    discrete_to_input = None
    check_gpu = None
    mc189_core = None
    OBSERVATION_SIZE = 48
    ACTION_SIZE = 17
    MAX_BATCH_SIZE = 4096
    TICKS_PER_SECOND = 20
    MC189_ACTION_COUNT = 17

# =============================================================================
# Import Python wrappers
# =============================================================================

try:
    from .env import (
        DragonFightEnv,
        MinecraftEnv,
        VecMinecraftEnv,
        make,
        make_vec,
    )
except ImportError:
    MinecraftEnv = None
    VecMinecraftEnv = None
    DragonFightEnv = None
    make = None
    make_vec = None

try:
    from .vec_env import (
        FreeTheEndEnv,
        SB3VecDragonFightEnv,
        SB3VecFreeTheEndEnv,
        VecDragonFightEnv,
    )
    from .vec_env import (
        VecFreeTheEndEnv as _VecFreeTheEndEnv,
    )
except ImportError:
    VecDragonFightEnv = None
    SB3VecDragonFightEnv = None
    FreeTheEndEnv = None
    _VecFreeTheEndEnv = None
    SB3VecFreeTheEndEnv = None

# Import high-performance vectorized environment (overrides basic version)
try:
    from .vec_free_the_end_env import (
        VecFreeTheEndEnv,
        make_vec_free_the_end_env,
    )
except ImportError:
    # Fall back to basic version if GPU env not available
    VecFreeTheEndEnv = _VecFreeTheEndEnv  # type: ignore[assignment,misc]
    make_vec_free_the_end_env = None

try:
    from .curriculum import Stage, StageID
except ImportError:
    StageID = None
    Stage = None

try:
    from .curriculum_manager import (
        AdvancementEvent,
        StageOverride,
        StageStats,
        VecCurriculumManager,
        create_vec_curriculum_with_stage1_overrides,
    )
except ImportError:
    VecCurriculumManager = None
    AdvancementEvent = None
    StageOverride = None
    StageStats = None
    create_vec_curriculum_with_stage1_overrides = None

try:
    from .triangulation import (
        EyeThrow,
        TriangulationState,
        direction_from_yaw,
        triangulate_stronghold,
    )
except ImportError:
    triangulate_stronghold = None
    TriangulationState = None
    EyeThrow = None
    direction_from_yaw = None

try:
    from .replay import (
        SpeedrunRecorder,
        SpeedrunReplayer,
        TrajectoryAnalyzer,
        load_recording,
        merge_recordings,
        save_recording,
    )
except ImportError:
    SpeedrunRecorder = None
    SpeedrunReplayer = None
    TrajectoryAnalyzer = None
    load_recording = None
    save_recording = None
    merge_recordings = None

try:
    from .progress_watchdog import (
        ProgressWatchdog,
        StallAlert,
        StallAlertConfig,
    )
except ImportError:
    ProgressWatchdog = None
    StallAlert = None
    StallAlertConfig = None

try:
    from .progression import (
        ProgressTracker,
        SpeedrunProgress,
        SpeedrunStage,
        create_progress_observation_space,
        merge_progress,
    )
except ImportError:
    SpeedrunProgress = None
    SpeedrunStage = None
    ProgressTracker = None
    create_progress_observation_space = None
    merge_progress = None

# Unified FreeTheEndEnv from free_the_end_env.py (the comprehensive version)
try:
    from .free_the_end_env import (
        STAGE_CONFIGS,
        make_curriculum,
        make_single_stage,
        make_speedrun,
    )
    from .free_the_end_env import (
        FreeTheEndEnv as UnifiedFreeTheEndEnv,
    )
    from .free_the_end_env import (
        StageConfig as UnifiedStageConfig,
    )
    from .free_the_end_env import (
        StageID as UnifiedStageID,
    )
except ImportError:
    UnifiedFreeTheEndEnv = None
    UnifiedStageID = None
    UnifiedStageConfig = None
    STAGE_CONFIGS = None
    make_single_stage = None
    make_curriculum = None
    make_speedrun = None

try:
    from .world_model import (
        DreamEnv,
        LatentState,
        LatentStateExtractor,
        TransitionBatch,
        TransitionBuffer,
        WorldModelDataLoader,
        collect_transitions,
        collect_vec_transitions,
        make_buffer,
        make_data_loader,
    )
except ImportError:
    TransitionBuffer = None
    TransitionBatch = None
    WorldModelDataLoader = None
    LatentState = None
    LatentStateExtractor = None
    DreamEnv = None
    make_buffer = None
    make_data_loader = None
    collect_transitions = None
    collect_vec_transitions = None

try:
    from .speedrun_env import (
        Dimension as PyDimension,
    )
    from .speedrun_env import (
        DimensionState,
        EpisodeStats,
        ObservationLayout,
        SpeedrunAction,
        SpeedrunEnv,
        make_speedrun_env,
        make_stage_env,
    )

    # Use the Python Dimension as the canonical export when C++ module
    # doesn't provide one (or always prefer the richer Python enum).
    if Dimension is None:
        Dimension = PyDimension  # type: ignore[assignment]
except ImportError:
    SpeedrunEnv = None
    SpeedrunAction = None
    ObservationLayout = None
    EpisodeStats = None
    DimensionState = None
    PyDimension = None
    make_speedrun_env = None
    make_stage_env = None

try:
    from .hierarchical import (
        HierarchicalConfig,
        # Controller
        HierarchicalController,
        # Reward shaping
        HierarchicalRewardShaper,
        # Policy interfaces
        HighLevelPolicy,
        LowLevelPolicy,
        OptionState,
        SubGoal,
        SubGoalCategory,
        # Detection
        SubGoalDetector,
        # Sub-goal definitions
        SubGoalID,
        SubGoalRegistry,
        TerminationCritic,
        TrajectorySegment,
        # Utilities
        create_hierarchical_env_wrapper,
        get_subgoal_embedding,
    )
except ImportError:
    SubGoalID = None
    SubGoalCategory = None
    SubGoal = None
    SubGoalRegistry = None
    HighLevelPolicy = None
    LowLevelPolicy = None
    TerminationCritic = None
    HierarchicalController = None
    HierarchicalConfig = None
    OptionState = None
    SubGoalDetector = None
    TrajectorySegment = None
    HierarchicalRewardShaper = None
    create_hierarchical_env_wrapper = None
    get_subgoal_embedding = None

try:
    from .observations import (
        CompactInventoryState,
        CompactObservationDecoder,
        CompactObservationEncoder,
        # Compact 256-float format
        CompactPlayerState,
        DictObservationSpace,
        DragonState,
        EntityAwareness,
        EntityObservation,
        InventorySummary,
        MinecraftObservation,
        ObservationEncoder,
        ObservationSpace,
        # Full observation format (~4500 floats)
        PlayerState,
        RayCastDistances,
        VoxelGrid,
        create_observation_from_c_struct,
    )
except ImportError:
    PlayerState = None
    InventorySummary = None
    VoxelGrid = None
    RayCastDistances = None
    EntityObservation = None
    EntityAwareness = None
    DragonState = None
    MinecraftObservation = None
    ObservationSpace = None
    DictObservationSpace = None
    ObservationEncoder = None
    create_observation_from_c_struct = None
    CompactPlayerState = None
    CompactInventoryState = None
    CompactObservationDecoder = None
    CompactObservationEncoder = None

try:
    from .stage_envs import (
        # Base class and config
        BaseStageEnv,
        # Individual stage environments
        BasicSurvivalEnv,
        Difficulty,
        EndermanHuntingEnv,
        NetherNavigationEnv,
        ResourceGatheringEnv,
        StageConfig,
        StrongholdFindingEnv,
        get_stage_info,
    )
    from .stage_envs import (
        DragonFightEnv as DragonFightStageEnv,  # Alias to avoid conflict
    )
    from .stage_envs import (
        # Factory functions
        make_stage_env as make_individual_stage_env,
    )
except ImportError:
    BasicSurvivalEnv = None
    ResourceGatheringEnv = None
    NetherNavigationEnv = None
    EndermanHuntingEnv = None
    StrongholdFindingEnv = None
    DragonFightStageEnv = None
    BaseStageEnv = None
    StageConfig = None
    Difficulty = None
    make_individual_stage_env = None
    get_stage_info = None

# =============================================================================
# Module metadata
# =============================================================================

__version__ = "0.1.0"
__all__ = [
    # Gymnasium environments
    "MinecraftEnv",
    "VecMinecraftEnv",
    "DragonFightEnv",
    "VecDragonFightEnv",
    "SB3VecDragonFightEnv",
    # FreeTheEnd (speedrun) environments
    "FreeTheEndEnv",
    "VecFreeTheEndEnv",
    "SB3VecFreeTheEndEnv",
    "make_vec_free_the_end_env",
    # Curriculum
    "StageID",
    "Stage",
    "load_stage_config",
    # Factory functions
    "make",
    "make_vec",
    "make_speedrun_env",
    "make_vec_speedrun_env",
    "make_sb3_speedrun_env",
    # Curriculum management
    "VecCurriculumManager",
    "AdvancementEvent",
    "StageStats",
    # Triangulation
    "triangulate_stronghold",
    "TriangulationState",
    "EyeThrow",
    "direction_from_yaw",
    # Replay
    "SpeedrunRecorder",
    "SpeedrunReplayer",
    "TrajectoryAnalyzer",
    "load_recording",
    "save_recording",
    "merge_recordings",
    # Progression tracking
    "SpeedrunProgress",
    "SpeedrunStage",
    "ProgressTracker",
    "create_progress_observation_space",
    "merge_progress",
    # World model (Dreamer-style training)
    "TransitionBuffer",
    "TransitionBatch",
    "WorldModelDataLoader",
    "LatentState",
    "LatentStateExtractor",
    "DreamEnv",
    "make_buffer",
    "make_data_loader",
    "collect_transitions",
    "collect_vec_transitions",
    # Speedrun environment (curriculum learning)
    "SpeedrunEnv",
    "SpeedrunAction",
    "ObservationLayout",
    "EpisodeStats",
    "make_stage_env",
    # Hierarchical RL (option-critic style)
    "SubGoalID",
    "SubGoalCategory",
    "SubGoal",
    "SubGoalRegistry",
    "HighLevelPolicy",
    "LowLevelPolicy",
    "TerminationCritic",
    "HierarchicalController",
    "HierarchicalConfig",
    "OptionState",
    "SubGoalDetector",
    "TrajectorySegment",
    "HierarchicalRewardShaper",
    "create_hierarchical_env_wrapper",
    "get_subgoal_embedding",
    # Observation encoding (full ~4500 floats)
    "PlayerState",
    "InventorySummary",
    "VoxelGrid",
    "RayCastDistances",
    "EntityObservation",
    "EntityAwareness",
    "DragonState",
    "MinecraftObservation",
    "ObservationSpace",
    "DictObservationSpace",
    "ObservationEncoder",
    "create_observation_from_c_struct",
    # Observation encoding (compact 256 floats)
    "CompactPlayerState",
    "CompactInventoryState",
    "CompactObservationDecoder",
    "CompactObservationEncoder",
    # Individual stage environments
    "BasicSurvivalEnv",
    "ResourceGatheringEnv",
    "NetherNavigationEnv",
    "EndermanHuntingEnv",
    "StrongholdFindingEnv",
    "DragonFightStageEnv",
    "BaseStageEnv",
    "StageConfig",
    "Difficulty",
    "make_individual_stage_env",
    "get_stage_info",
    # Unified FreeTheEndEnv (comprehensive single/curriculum/speedrun)
    "UnifiedFreeTheEndEnv",
    "UnifiedStageID",
    "UnifiedStageConfig",
    "STAGE_CONFIGS",
    "make_single_stage",
    "make_curriculum",
    "make_speedrun",
    # C++ classes
    "MinecraftSimulator",
    "VecMinecraftSimulator",
    "MC189Simulator",
    "SimulatorConfig",
    "mc189_core",
    # Enums
    "ActionType",
    "Dimension",
    "DimensionState",
    "MC189Action",
    "MC189Dimension",
    # Utilities
    "GymSpaceHelper",
    "BatchUtils",
    "RunningMeanStd",
    "discrete_to_input",
    "check_gpu",
    # Constants
    "OBSERVATION_SIZE",
    "ACTION_SIZE",
    "MAX_BATCH_SIZE",
    "TICKS_PER_SECOND",
    "MC189_ACTION_COUNT",
    "__version__",
]


def check_cpp_module() -> bool:
    """Check if C++ extension module is available."""
    return _HAS_CPP_MODULE


def get_version_info() -> dict[str, str]:
    """Get version information for the package and dependencies."""
    info = {
        "minecraft_sim": __version__,
        "cpp_module": "available" if _HAS_CPP_MODULE else "not available",
    }

    try:
        import gymnasium

        info["gymnasium"] = gymnasium.__version__
    except ImportError:
        info["gymnasium"] = "not installed"

    try:
        import numpy

        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = "not installed"

    return info
