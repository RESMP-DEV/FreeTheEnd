# Minecraft Simulator API Reference

Complete API documentation for the Minecraft RL simulation package.

## Table of Contents

- [Environments](#environments)
  - [SpeedrunEnv](#speedrunenv)
  - [VecDragonFightEnv](#vecdragonfightenv)
  - [SB3VecDragonFightEnv](#sb3vecdragonfightenv)
  - [SpeedrunVecEnv](#speedrunvecenv)
  - [DragonFightEnv (Gymnasium)](#dragonfightenv-gymnasium)
- [Stage Environments](#stage-environments)
  - [BaseStageEnv](#basestageenv)
  - [BasicSurvivalEnv](#basicsurvivalenv)
  - [ResourceGatheringEnv](#resourcegatheringenv)
  - [NetherNavigationEnv](#nethernavigationenv)
  - [EndermanHuntingEnv](#endermanhuntingenv)
  - [StrongholdFindingEnv](#strongholdfindingenv)
  - [DragonFightEnv (Stage 6)](#dragonfightenv-stage-6)
- [Configuration Classes](#configuration-classes)
  - [SimulatorConfig (C++)](#simulatorconfig-c)
  - [StageConfig (Python)](#stageconfig-python)
  - [SpawnConfig](#spawnconfig)
  - [RewardConfig](#rewardconfig)
  - [TerminationConfig](#terminationconfig)
  - [Difficulty](#difficulty)
- [Curriculum System](#curriculum-system)
  - [CurriculumManager](#curriculummanager)
  - [VecCurriculumManager](#veccurriculummanager)
  - [StageID](#stageid)
  - [Stage](#stage)
  - [StageOverride](#stageoverride)
  - [StageStats](#stagestats)
  - [AdvancementEvent](#advancementevent)
  - [StageCriteria](#stagecriteria)
- [Progress Tracking](#progress-tracking)
  - [SpeedrunProgress](#speedrunprogress)
  - [ProgressTracker](#progresstracker)
  - [ProgressWatchdog](#progresswatchdog)
  - [StallAlertConfig](#stallalertconfig)
  - [StallAlert](#stallalert)
- [Observation Spaces](#observation-spaces)
  - [48-Float Observation (Dragon Fight)](#48-float-observation-dragon-fight)
  - [256-Float Observation (Full Speedrun)](#256-float-observation-full-speedrun)
  - [Observation Encoding/Decoding](#observation-encodingdecoding)
  - [Compact Observation (256 floats)](#compact-observation-256-floats)
- [Action Spaces](#action-spaces)
  - [17-Action Discrete (Dragon Fight)](#17-action-discrete-dragon-fight)
  - [32-Action Discrete (Speedrun)](#32-action-discrete-speedrun)
  - [Per-Stage Action Spaces](#per-stage-action-spaces)
- [Reward Functions](#reward-functions)
  - [create_reward_shaper](#create_reward_shaper)
  - [CompositeRewardShaper](#compositerewardshaper)
  - [Stage-Specific Rewards](#stage-specific-rewards)
  - [RewardStats](#rewardstats)
- [Low-Level API](#low-level-api)
  - [mc189_core.MC189Simulator](#mc189_coremc189simulator)
  - [Shader Loading](#shader-loading)
  - [Buffer Management](#buffer-management)
  - [C++ Data Structures](#c-data-structures)
- [Factory Functions](#factory-functions)
- [Constants](#constants)

---

## Environments

### SpeedrunEnv

Gymnasium-compatible single environment for the full Minecraft Free The End speedrun with curriculum learning. Supports all 6 stages with an extended 256-float observation space and 32 discrete actions.

```python
from minecraft_sim.speedrun_env import SpeedrunEnv

env = SpeedrunEnv(
    stage_id=1,
    shader_dir=None,
    auto_advance=True,
    curriculum_threshold=None,
    max_episode_steps=None,
    render_mode=None,
    seed=None,
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stage_id` | `int` | `1` | Initial curriculum stage (1-6). |
| `shader_dir` | `str \| None` | `None` | Path to shader directory. Auto-detected if None. |
| `auto_advance` | `bool` | `True` | Automatically advance to next stage when threshold met. |
| `curriculum_threshold` | `float \| None` | `None` | Success rate to advance. Uses stage config default if None. |
| `max_episode_steps` | `int \| None` | `None` | Override max steps per episode. Uses stage config if None. |
| `render_mode` | `str \| None` | `None` | "human" or "rgb_array". |
| `seed` | `int \| None` | `None` | Random seed for reproducibility. |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `observation_space` | `gym.spaces.Box` | Box(0, 1, shape=(256,), float32) |
| `action_space` | `gym.spaces.Discrete` | Discrete(32) |
| `stage_id` | `int` | Current curriculum stage (1-6) |
| `current_stage` | `Stage \| None` | Current stage configuration |
| `curriculum_progress` | `dict[str, Any]` | Curriculum training summary |

#### Methods

##### `reset(*, seed=None, options=None) -> tuple[np.ndarray, dict]`

Reset the environment for a new episode.

```python
obs, info = env.reset(seed=42)
obs, info = env.reset(options={"stage_id": 3})  # Force specific stage
```

**Parameters:**
- `seed` (`int | None`): Random seed for this episode.
- `options` (`dict | None`): Reset options. Supported keys:
  - `"stage_id"` (`int`): Force reset to specific stage (1-6).
  - `"skip_advance"` (`bool`): Don't auto-advance even if mastered.

**Returns:**
- `obs` (`np.ndarray`): Shape (256,), normalized to [0, 1].
- `info` (`dict`): Contains `stage_id`, `stage_name`, `max_steps`, `stage_advanced`, `dimension`.

##### `step(action) -> tuple[np.ndarray, float, bool, bool, dict]`

Execute one environment step.

```python
obs, reward, terminated, truncated, info = env.step(action)
```

**Parameters:**
- `action` (`int | np.ndarray`): Discrete action index (0-31).

**Returns:**
- `obs` (`np.ndarray`): Shape (256,).
- `reward` (`float`): Shaped reward for current stage.
- `terminated` (`bool`): True if objective achieved or death.
- `truncated` (`bool`): True if max steps reached.
- `info` (`dict`): Contains `stage_id`, `step`, `episode_reward`, `stage_advanced`, and stage-specific milestone info.

##### `set_stage(stage_id: int) -> None`

Manually set the curriculum stage. Persists inventory across transitions.

##### `save_curriculum_progress(path: str | Path) -> None`

Save curriculum progress to JSON file.

##### `load_curriculum_progress(path: str | Path) -> None`

Load curriculum progress from JSON file and restore stage.

##### `close() -> None`

Clean up simulator resources.

---

### VecDragonFightEnv

High-performance vectorized Dragon Fight environment. Uses the C++ `MC189Simulator` backend for parallel GPU simulation.

```python
from minecraft_sim import VecDragonFightEnv

env = VecDragonFightEnv(
    num_envs=64,
    shader_dir=None,
    observation_size=48,
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_envs` | `int` | `64` | Number of parallel environments. |
| `shader_dir` | `str \| Path \| None` | `None` | Shader directory path. |
| `observation_size` | `int` | `48` | Observation vector dimension. |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `num_envs` | `int` | Number of parallel environments |
| `observation_size` | `int` | Size of observation vector |
| `sim` | `MC189Simulator` | Underlying C++ simulator |
| `action_space_size` | `int` | Number of discrete actions (17) |

#### Methods

##### `reset() -> NDArray[np.float32]`

Reset all environments.

```python
obs = env.reset()
assert obs.shape == (64, 48)
```

##### `step(actions) -> tuple[NDArray, NDArray, NDArray, list[dict]]`

Step all environments.

```python
actions = np.random.randint(0, 17, size=64)
obs, rewards, dones, infos = env.step(actions)
```

**Returns:**
- `obs`: Shape (num_envs, observation_size), clipped to [0, 1].
- `rewards`: Shape (num_envs,).
- `dones`: Shape (num_envs,).
- `infos`: List of per-environment info dicts.

##### `close() -> None`

Release simulator resources.

---

### SB3VecDragonFightEnv

Stable Baselines 3 compatible vectorized environment. Wraps `VecDragonFightEnv` with the full SB3 `VecEnv` interface including episode statistics tracking.

```python
from minecraft_sim import SB3VecDragonFightEnv
from stable_baselines3 import PPO

env = SB3VecDragonFightEnv(num_envs=64, shader_dir=None)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_envs` | `int` | `64` | Number of parallel environments. |
| `shader_dir` | `str \| Path \| None` | `None` | Shader directory path. |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `num_envs` | `int` | Number of parallel environments |
| `observation_space` | `gym.spaces.Box` | Box(0, 1, shape=(48,), float32) |
| `action_space` | `gym.spaces.Discrete` | Discrete(17) |
| `single_observation_space` | `gym.spaces.Box` | Same as observation_space |
| `single_action_space` | `gym.spaces.Discrete` | Same as action_space |

#### Methods

Implements the full SB3 `VecEnv` interface:

- `reset() -> NDArray`: Reset all environments.
- `step(actions) -> tuple`: Returns (obs, rewards, dones, infos) with episode statistics in `infos[i]["episode"]`.
- `step_async(actions) -> None`: Start async step.
- `step_wait() -> tuple`: Get async step results.
- `seed(seed=None) -> list`: Set seed for all environments.
- `env_is_wrapped(wrapper_class) -> list[bool]`: Wrapper check.
- `env_method(name, *args, **kwargs) -> list`: Call method on sub-envs.
- `get_attr(name) -> list`: Get attribute from sub-envs.
- `set_attr(name, value) -> None`: Set attribute on sub-envs.
- `unwrapped -> self`: Returns self.

Episode statistics in `infos`:
```python
for i, info in enumerate(infos):
    if "episode" in info:
        print(f"Env {i}: reward={info['episode']['r']}, length={info['episode']['l']}")
        terminal_obs = info["terminal_observation"]
```

---

### SpeedrunVecEnv

Vectorized environment for parallel Free The End speedrun training with per-environment curriculum advancement.

```python
from minecraft_sim.speedrun_vec_env import SpeedrunVecEnv, make_speedrun_vec_env
from minecraft_sim.curriculum import StageID

env = SpeedrunVecEnv(
    num_envs=64,
    shader_dir=None,
    observation_size=48,
    initial_stage=StageID.BASIC_SURVIVAL,
    curriculum_manager=None,
    auto_curriculum=True,
    success_threshold=0.7,
    min_episodes_for_advance=100,
    progress_watchdog=None,
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_envs` | `int` | `64` | Number of parallel environments. |
| `shader_dir` | `str \| Path \| None` | `None` | Shader directory path. |
| `observation_size` | `int` | `48` | Observation vector dimension. |
| `initial_stage` | `StageID` | `BASIC_SURVIVAL` | Starting stage for all envs. |
| `curriculum_manager` | `CurriculumManager \| None` | `None` | Pre-configured manager. |
| `auto_curriculum` | `bool` | `True` | Enable automatic curriculum advancement. |
| `success_threshold` | `float` | `0.7` | Success rate required to advance stage. |
| `min_episodes_for_advance` | `int` | `100` | Minimum episodes before stage advance. |
| `progress_watchdog` | `ProgressWatchdog \| StallAlertConfig \| None` | `None` | Stall detection config. |

#### Methods

##### `reset(*, seed=None, options=None) -> NDArray[np.float32]`

Reset all environments.

##### `step(actions) -> tuple[NDArray, NDArray, NDArray, list[dict]]`

Step all environments. Handles per-environment curriculum advancement.

**Returns info keys:**
- `episode`: `{"r": float, "l": int, "stage_id": int, "stage_name": str, "success": bool}` on episode completion.
- `terminal_observation`: Final observation on episode completion.
- `curriculum_advanced`: `True` if environment advanced stages.
- `new_stage_id` / `new_stage_name`: New stage info on advancement.
- `stage_id`: Current stage for this environment.
- `progress_snapshot`: Current speedrun progress.
- `obsidian_stall_alert`: Alert if obsidian collection stalled.

##### `set_stage(env_ids, stage_id) -> None`

Set specific environments to a curriculum stage.

```python
env.set_stage([0, 1, 2], StageID.END_FIGHT)
env.set_stage(0, 6)  # Also accepts int
```

##### `advance_curriculum(env_id: int) -> bool`

Manually advance an environment to the next stage.

##### `get_stage_distribution() -> dict[str, int]`

Get count of environments per stage.

```python
>>> env.get_stage_distribution()
{'BASIC_SURVIVAL': 32, 'RESOURCE_GATHERING': 20, 'END_FIGHT': 12}
```

##### `get_curriculum_stats() -> dict[str, Any]`

Get comprehensive curriculum statistics including per-stage success rates, episode counts, and progress.

##### `get_env_stages() -> NDArray[np.int32]`

Get current stage ID for each environment.

##### `get_progress_snapshots() -> list[dict[str, Any]]`

Get latest progress snapshot for each environment.

---

### DragonFightEnv (Gymnasium)

Simple Gymnasium-compatible single environment for Ender Dragon fight training. Uses the 48-float observation space.

```python
from minecraft_sim import DragonFightEnv

env = DragonFightEnv(render_mode=None, shader_dir=None)
obs, info = env.reset(seed=42)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `render_mode` | `str \| None` | `None` | Only "human" supported. |
| `shader_dir` | `str \| None` | `None` | Path to shader directory. |

#### Spaces

- `observation_space`: Box(0, 1, shape=(48,), float32)
- `action_space`: Discrete(17)

---

## Stage Environments

### BaseStageEnv

Abstract base class for all speedrun stage environments. Provides common simulator initialization, reset/step protocol, episode tracking, and reward shaping hooks.

```python
from minecraft_sim.stage_envs import BaseStageEnv, StageConfig

class MyStageEnv(BaseStageEnv):
    STAGE_ID = 7
    STAGE_NAME = "Custom Stage"
    DEFAULT_OBS_SIZE = 128
    DEFAULT_ACTION_SIZE = 24
    DEFAULT_MAX_TICKS = 6000

    def _initialize_stage_state(self) -> dict: ...
    def _shape_reward(self, base_reward, obs, action) -> float: ...
    def _check_success(self) -> bool: ...
```

#### Class Variables (Must Override)

| Variable | Type | Description |
|----------|------|-------------|
| `STAGE_ID` | `int` | Stage number identifier |
| `STAGE_NAME` | `str` | Human-readable name |
| `DEFAULT_OBS_SIZE` | `int` | Observation vector size |
| `DEFAULT_ACTION_SIZE` | `int` | Number of discrete actions |
| `DEFAULT_MAX_TICKS` | `int` | Default episode length |

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `StageConfig \| None` | `None` | Stage configuration. |
| `shader_dir` | `str \| Path \| None` | `None` | Shader directory path. |
| `render_mode` | `str \| None` | `None` | Rendering mode. |

#### Abstract Methods (Must Implement)

```python
def _initialize_stage_state(self) -> dict[str, Any]:
    """Return initial stage-specific tracking state."""
    ...

def _shape_reward(self, base_reward: float, obs: NDArray, action: int) -> float:
    """Apply stage-specific reward shaping to base reward."""
    ...

def _check_success(self) -> bool:
    """Check if stage goal is achieved."""
    ...
```

#### Provided Methods

- `reset(*, seed=None, options=None)`: Reset with simulator re-init and progress tracker reset.
- `step(action)`: Step with reward shaping, progress tracking, and termination checks.
- `close()`: Clean up simulator.
- `get_progress_tracker() -> ProgressTracker`: Access cumulative progression data.

---

### BasicSurvivalEnv

Stage 1: Basic Survival. Goal: Survive, gather wood, make tools.

```python
from minecraft_sim.stage_envs import BasicSurvivalEnv, StageConfig, Difficulty

env = BasicSurvivalEnv(
    config=StageConfig(difficulty=Difficulty.NORMAL),
)
```

| Property | Value |
|----------|-------|
| Observation size | 128 |
| Action size | 24 |
| Max ticks | 6000 (5 min) |
| Success condition | Iron pickaxe crafted |
| Failure condition | 10 deaths |

**Reward Weights:**

| Reward | Value | Trigger |
|--------|-------|---------|
| `REWARD_ZOMBIE_KILLED` | 0.5 | Zombie killed |
| `REWARD_SKELETON_KILLED` | 0.5 | Skeleton killed |
| `REWARD_WOOD_MINED` | 0.1 | Wood block mined |
| `REWARD_WOODEN_PICKAXE` | 0.5 | Wooden pickaxe crafted |
| `REWARD_WOODEN_SWORD` | 0.5 | Wooden sword crafted |
| `REWARD_IRON_PICKAXE` | 2.0 | Iron pickaxe crafted (completion) |
| `REWARD_EXPLORATION` | 0.01 | New chunk explored |

---

### ResourceGatheringEnv

Stage 2: Resource Gathering. Goal: Mine iron, diamonds, make bucket, collect obsidian.

| Property | Value |
|----------|-------|
| Observation size | 128 |
| Action size | 24 |
| Max ticks | 12000 (10 min) |
| Success condition | Has bucket AND 10+ obsidian |
| Failure condition | 7 deaths |

**Reward Weights:**

| Reward | Value | Trigger |
|--------|-------|---------|
| `REWARD_COBBLESTONE` | 0.02 | Cobblestone mined |
| `REWARD_COAL` | 0.1 | Coal mined |
| `REWARD_IRON_ORE` | 0.3 | Iron ore mined |
| `REWARD_DIAMOND` | 1.0 | Diamond mined |
| `REWARD_IRON_INGOT` | 0.2 | Iron smelted |
| `REWARD_BUCKET` | 1.0 | Bucket crafted |
| `REWARD_OBSIDIAN` | 0.3 | Obsidian collected |
| `REWARD_VERTICAL_MINING` | 0.05 | New lowest Y reached |

---

### NetherNavigationEnv

Stage 3: Nether Navigation. Goal: Enter nether, find fortress, get 7+ blaze rods.

| Property | Value |
|----------|-------|
| Observation size | 192 |
| Action size | 28 |
| Max ticks | 18000 (15 min) |
| Success condition | Fortress found AND 7+ blaze rods |
| Failure condition | 5 deaths |

**Reward Weights:**

| Reward | Value | Trigger |
|--------|-------|---------|
| `REWARD_PORTAL_LIT` | 3.0 | Nether portal lit |
| `REWARD_NETHER_ENTERED` | 5.0 | Entered nether dimension |
| `REWARD_FORTRESS_FOUND` | 5.0 | Nether fortress found |
| `REWARD_BLAZE_DAMAGED` | 0.1 | Blaze damaged |
| `REWARD_BLAZE_KILLED` | 1.0 | Blaze killed |
| `REWARD_BLAZE_ROD` | 1.5 | Blaze rod collected |
| `REWARD_GHAST_DEFLECT` | 0.5 | Ghast fireball deflected |
| `REWARD_ESCAPED_LAVA` | 0.5 | Escaped lava |
| `PENALTY_DEATH` | -2.0 | Player death |

---

### EndermanHuntingEnv

Stage 4: Enderman Hunting. Goal: Collect 12+ ender pearls.

| Property | Value |
|----------|-------|
| Observation size | 192 |
| Action size | 28 |
| Max ticks | 12000 (10 min) |
| Success condition | 12+ ender pearls |
| Failure condition | 5 deaths |

**Reward Weights:**

| Reward | Value | Trigger |
|--------|-------|---------|
| `REWARD_ENDERMAN_DAMAGED` | 0.1 | Enderman hit |
| `REWARD_ENDERMAN_KILLED` | 1.5 | Enderman killed |
| `REWARD_ENDER_PEARL` | 2.0 | Pearl collected |
| `REWARD_WATER_TRAP` | 0.5 | Water trap placed |
| `REWARD_LOW_CEILING_TRAP` | 0.5 | Low ceiling trap built |
| `REWARD_SAFE_ENGAGEMENT` | 0.3 | Kill without damage |
| `REWARD_PUMPKIN_EQUIPPED` | 0.3 | Pumpkin head on |
| `PENALTY_DAMAGE_TAKEN` | -0.2 | Damage from enderman |

---

### StrongholdFindingEnv

Stage 5: Stronghold Finding. Goal: Find stronghold, activate end portal.

| Property | Value |
|----------|-------|
| Observation size | 192 |
| Action size | 28 |
| Max ticks | 18000 (15 min) |
| Success condition | End portal activated |
| Failure condition | 3 deaths or out of eyes |

**Reward Weights:**

| Reward | Value | Trigger |
|--------|-------|---------|
| `REWARD_FIRST_EYE_THROW` | 1.0 | First eye of ender thrown |
| `REWARD_TRIANGULATION_TRAVEL` | 0.01 | Distance covered for triangulation |
| `REWARD_SECOND_EYE_THROW` | 1.5 | Second throw (enables triangulation) |
| `REWARD_INTERSECTION_FOUND` | 2.0 | Triangulation complete |
| `REWARD_EYE_DROPS` | 1.0 | Eye drops toward stronghold |
| `REWARD_STRONGHOLD_ENTERED` | 2.0 | Entered stronghold |
| `REWARD_PORTAL_ROOM_FOUND` | 5.0 | Portal room located |
| `REWARD_EYE_PLACED` | 0.5 | Eye placed in frame |
| `REWARD_PORTAL_ACTIVE` | 10.0 | Portal fully activated |
| `PENALTY_EYE_LOST` | -0.5 | Eye of ender broke |

---

### DragonFightEnv (Stage 6)

Stage 6: Dragon Fight (Enhanced). Goal: Defeat the Ender Dragon.

```python
from minecraft_sim.stage_envs import DragonFightEnv, StageConfig

env = DragonFightEnv(
    config=StageConfig(max_episode_ticks=36000),
    shader_dir=None,
    render_mode=None,
    enable_one_cycle=True,  # Enable bed explosion mechanics
)
```

| Property | Value |
|----------|-------|
| Observation size | 64 |
| Action size | 20 |
| Max ticks | 36000 (30 min) |
| Success condition | Dragon killed |
| Failure condition | Death or void fall |

**Additional Constructor Parameter:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_one_cycle` | `bool` | `True` | Enable bed explosion mechanics for one-cycle strategy. |

**Reward Weights:**

| Reward | Value | Trigger |
|--------|-------|---------|
| `REWARD_SURVIVE_SPAWN` | 1.0 | Survived obsidian platform |
| `REWARD_REACH_MAIN_ISLAND` | 2.0 | Reached main island |
| `REWARD_DESTROY_CRYSTAL` | 3.0 | Crystal destroyed |
| `REWARD_DESTROY_CAGED_CRYSTAL` | 5.0 | Caged crystal destroyed |
| `REWARD_ALL_CRYSTALS_DESTROYED` | 10.0 | All 10 crystals gone |
| `REWARD_DAMAGE_DRAGON` | 0.5 | Dragon hit |
| `REWARD_DAMAGE_DRAGON_PERCHING` | 1.0 | Dragon hit during perch |
| `REWARD_BED_DAMAGE` | 3.0 | Bed explosion damage to dragon |
| `REWARD_DRAGON_KILLED` | 50.0 | Dragon defeated |
| `REWARD_ENTER_EXIT_PORTAL` | 10.0 | Entered exit portal |
| `PENALTY_VOID_DEATH` | -10.0 | Fell into void |
| `PENALTY_DEATH` | -5.0 | Normal death |

**Dragon Phases:**
- `PHASE_CIRCLING` (0): Dragon flies around pillars.
- `PHASE_STRAFING` (1): Dragon strafes toward player.
- `PHASE_PERCHING` (2): Dragon lands on fountain (optimal damage window).
- `PHASE_BREATH` (3): Dragon uses breath attack.

---

## Configuration Classes

### SimulatorConfig (C++)

Configuration for the low-level `MC189Simulator` C++ backend.

```python
import mc189_core

config = mc189_core.SimulatorConfig()
config.num_envs = 64
config.enable_validation = False
config.shader_dir = "path/to/shaders"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_envs` | `uint32_t` | `1` | Number of parallel environments. |
| `enable_validation` | `bool` | `False` | Enable Vulkan validation layers. |
| `shader_dir` | `str` | `"shaders"` | Directory containing .spv shader files. |
| `shader_set` | `list[str]` | `[]` | Stage-specific shader subset to load. If empty, loads monolithic shader. |

---

### StageConfig (Python)

Configuration for individual stage environments.

```python
from minecraft_sim.stage_envs import StageConfig, Difficulty

config = StageConfig(
    max_episode_ticks=6000,
    ticks_per_second=20,
    difficulty=Difficulty.NORMAL,
    death_penalty=-1.0,
    tick_penalty=-0.0001,
    spawn_protection_ticks=100,
    reward_scale=1.0,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_episode_ticks` | `int` | `6000` | Maximum episode length (5 min at 20 tps). |
| `ticks_per_second` | `int` | `20` | Simulation tick rate. |
| `difficulty` | `Difficulty` | `NORMAL` | Difficulty preset affecting spawn/penalties. |
| `death_penalty` | `float` | `-1.0` | Reward penalty on player death. |
| `tick_penalty` | `float` | `-0.0001` | Per-tick penalty (time pressure). |
| `spawn_protection_ticks` | `int` | `100` | Invulnerability after spawn. |
| `reward_scale` | `float` | `1.0` | Global reward multiplier. |

---

### SpawnConfig

Configuration for agent spawn conditions within a curriculum stage.

```python
from minecraft_sim.curriculum import SpawnConfig

spawn = SpawnConfig(
    biome="plains",
    time_of_day=0,
    weather="clear",
    random_position=True,
    position=None,
    inventory={"diamond_sword": 1, "bow": 1},
    health=20.0,
    hunger=20.0,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `biome` | `str` | `"plains"` | Spawn biome (plains, nether_wastes, the_end, etc.). |
| `time_of_day` | `int` | `0` | Game ticks (0-24000, 0=dawn, 13000=dusk). |
| `weather` | `str` | `"clear"` | Weather: "clear", "rain", "thunder". |
| `random_position` | `bool` | `True` | Randomize spawn position within biome. |
| `position` | `tuple[float, float, float] \| None` | `None` | Exact (x, y, z) spawn coordinates. |
| `inventory` | `dict[str, int]` | `{}` | Starting items (item_name: count). |
| `health` | `float` | `20.0` | Starting health (0-20). |
| `hunger` | `float` | `20.0` | Starting hunger (0-20). |

---

### RewardConfig

Reward shaping configuration for a curriculum stage.

```python
from minecraft_sim.curriculum import RewardConfig

rewards = RewardConfig(
    sparse_reward=100.0,
    dense_rewards={
        "destroy_crystal": 3.0,
        "damage_dragon": 0.5,
        "dragon_killed": 50.0,
    },
    penalty_per_death=-5.0,
    penalty_per_tick=-0.0002,
    exploration_bonus=0.01,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sparse_reward` | `float` | `1.0` | Reward given on stage completion. |
| `dense_rewards` | `dict[str, float]` | `{}` | Event-based incremental rewards. |
| `penalty_per_death` | `float` | `-0.5` | Negative reward on player death. |
| `penalty_per_tick` | `float` | `-0.0001` | Per-tick time penalty. |
| `exploration_bonus` | `float` | `0.01` | Reward for visiting new chunks. |

---

### TerminationConfig

Conditions that end an episode.

```python
from minecraft_sim.curriculum import TerminationConfig

termination = TerminationConfig(
    max_ticks=36000,
    max_deaths=1,
    success_conditions=["dragon_killed"],
    failure_conditions=["death_count >= max_deaths", "fell_into_void"],
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_ticks` | `int` | `36000` | Maximum episode length in ticks. |
| `max_deaths` | `int` | `5` | Maximum deaths before failure. |
| `success_conditions` | `list[str]` | `[]` | Conditions that trigger success. |
| `failure_conditions` | `list[str]` | `[]` | Conditions that trigger failure. |

---

### Difficulty

Difficulty presets affecting spawn conditions and reward penalties.

```python
from minecraft_sim.stage_envs import Difficulty

class Difficulty(IntEnum):
    EASY = 1      # Reduced mob spawns, lower penalties
    NORMAL = 2    # Standard Minecraft mechanics
    HARD = 3      # Increased mob damage, faster hunger
    HARDCORE = 4  # Single life, maximum difficulty
```

---

## Curriculum System

### CurriculumManager

Manages progression through curriculum stages with automatic advancement tracking.

```python
from minecraft_sim.curriculum import CurriculumManager, StageID

manager = CurriculumManager(config_dir="/path/to/stage_configs")
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_dir` | `Path \| str \| None` | `None` | Stage config YAML directory. Uses `stage_configs/` if None. |

#### Key Methods

- `get_stage(stage_id: StageID) -> Stage`: Get stage configuration.
- `start_training(stage_id: StageID) -> Stage`: Start training on a stage.
- `record_episode(success, reward, ticks, stage_id=None) -> bool`: Record results, returns True if mastered.
- `should_advance() -> bool`: Check if advancement threshold met.
- `advance_stage() -> Stage | None`: Advance to next stage.
- `regress_stage(stage_id=None) -> Stage | None`: Return to earlier stage.
- `get_training_summary() -> dict`: Get full progress.
- `save_progress(path) / load_progress(path)`: Persistence.
- `on_stage_change(callback)`: Register transition callback.

---

### VecCurriculumManager

Per-environment curriculum manager for vectorized training. Each parallel environment advances independently.

```python
from minecraft_sim.curriculum_manager import (
    VecCurriculumManager,
    StageOverride,
    create_vec_curriculum_with_stage1_overrides,
)

manager = VecCurriculumManager(
    num_envs=64,
    initial_stage=1,
    advancement_threshold=0.7,
    min_episodes_to_advance=100,
    max_stage=6,
    stage_overrides={3: StageOverride(min_metric_value=7.0)},
)
```

#### Methods

- `update(env_id, success, reward, length, ...) -> AdvancementEvent | None`: Record episode and check advancement.
- `get_stage(env_id) -> int`: Get current stage for an environment.
- `get_stages() -> NDArray[np.int32]`: All environment stages.
- `get_stats(stage_id) -> StageStats`: Per-stage aggregate statistics.
- `force_advance(env_id) -> bool`: Manually advance an environment.
- `force_stage(env_id, stage_id) -> None`: Set environment to specific stage.

---

### StageID

Stage identifier enum with progression order.

```python
from minecraft_sim.curriculum import StageID

class StageID(IntEnum):
    BASIC_SURVIVAL = 1       # Survive, gather wood, tools
    RESOURCE_GATHERING = 2   # Iron, diamonds, bucket, obsidian
    NETHER_NAVIGATION = 3    # Portal, fortress, blaze rods
    ENDERMAN_HUNTING = 4     # Ender pearls, eyes of ender
    STRONGHOLD_FINDING = 5   # Triangulation, portal activation
    END_FIGHT = 6            # Crystals, dragon, victory
```

---

### Stage

Curriculum stage configuration dataclass. See [Stage Configuration Format](#stage-configuration-format-yaml) for YAML representation.

```python
from minecraft_sim.curriculum import Stage, StageID, SpawnConfig, RewardConfig, TerminationConfig

stage = Stage(
    id=StageID.BASIC_SURVIVAL,
    name="Basic Survival",
    description="Learn fundamental survival skills",
    objectives=["Kill 3 zombies", "Mine 10 wood", "Craft iron pickaxe"],
    spawn=SpawnConfig(biome="plains", health=20.0),
    rewards=RewardConfig(sparse_reward=10.0),
    termination=TerminationConfig(max_ticks=24000, max_deaths=10),
    prerequisites=[],
    difficulty=1,
    expected_episodes=500,
    curriculum_threshold=0.7,
)
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | `StageID` | Stage identifier enum |
| `name` | `str` | Human-readable name |
| `description` | `str` | Detailed description |
| `objectives` | `list[str]` | Specific objectives |
| `spawn` | `SpawnConfig` | Spawn configuration |
| `rewards` | `RewardConfig` | Reward shaping config |
| `termination` | `TerminationConfig` | Episode end conditions |
| `prerequisites` | `list[StageID]` | Required mastered stages |
| `difficulty` | `int` | Difficulty (1-10) |
| `expected_episodes` | `int` | Estimated episodes to master |
| `action_space` | `list[str]` | Available actions |
| `observation_space` | `list[str]` | Relevant observations |
| `curriculum_threshold` | `float` | Success rate to advance (0-1) |
| `metadata` | `dict` | Additional stage-specific data |

Methods:
- `to_dict() -> dict`: Serialize to dictionary.
- `Stage.from_dict(data) -> Stage`: Deserialize from dictionary.

---

### StageOverride

Per-stage overrides for curriculum advancement parameters in `VecCurriculumManager`.

```python
from minecraft_sim.curriculum_manager import StageOverride

override = StageOverride(
    min_episodes_to_advance=200,
    advancement_threshold=0.8,
    min_metric_value=7.0,          # e.g., minimum blaze rods
    min_dimension_episodes=50,      # min episodes in target dimension
    sustained_windows=3,            # windows of sustained success
    sustained_window_size=50,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_episodes_to_advance` | `int \| None` | `None` | Override min episodes. |
| `advancement_threshold` | `float \| None` | `None` | Override success rate. |
| `min_metric_value` | `float \| None` | `None` | Min tracked metric value. |
| `min_dimension_episodes` | `int \| None` | `None` | Min in-dimension episodes. |
| `sustained_windows` | `int \| None` | `None` | Consecutive success windows required. |
| `sustained_window_size` | `int \| None` | `None` | Episodes per evaluation window. |

---

### StageStats

Per-stage statistics aggregated across all environments.

| Attribute | Type | Description |
|-----------|------|-------------|
| `total_episodes` | `int` | Total episodes at this stage |
| `total_successes` | `int` | Successful episodes |
| `total_reward` | `float` | Cumulative reward |
| `best_reward` | `float` | Best single-episode reward |
| `avg_episode_length` | `float` | Running average episode length |

Properties:
- `success_rate -> float`: Successes / total episodes.
- `avg_reward -> float`: Total reward / total episodes.

---

### AdvancementEvent

Records a curriculum advancement event.

| Attribute | Type | Description |
|-----------|------|-------------|
| `env_id` | `int` | Environment that advanced |
| `old_stage` | `int` | Previous stage ID |
| `new_stage` | `int` | New stage ID |
| `timestamp` | `int` | Episode count when advanced |
| `success_rate` | `float` | Rate that triggered advancement |

---

### StageCriteria

Defines success conditions for each curriculum stage. Used by environments to determine episode success/failure and to compute partial progress.

```python
from minecraft_sim.stage_criteria import StageCriteria, get_stage_criteria, get_all_stages, calculate_total_progress

criteria = get_stage_criteria(stage_id=3)
print(criteria.name)  # "Nether Navigation"
print(criteria.check_success(state))  # True/False
print(criteria.get_partial_progress(state))  # 0.0 - 1.0
```

#### Constructor

```python
StageCriteria(
    stage_id: int,
    name: str,
    description: str,
    required: list[Callable[[dict], bool]],
    optional: list[Callable[[dict], bool]] = [],
    max_ticks: int = 6000,
    min_ticks: int = 0,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stage_id` | `int` | - | Stage identifier (1-6). |
| `name` | `str` | - | Human-readable stage name. |
| `description` | `str` | - | Stage objective description. |
| `required` | `list[Callable[[dict], bool]]` | - | All must be True for success. |
| `optional` | `list[Callable[[dict], bool]]` | `[]` | Bonus conditions (extra rewards). |
| `max_ticks` | `int` | `6000` | Timeout in ticks (episode fails if exceeded). |
| `min_ticks` | `int` | `0` | Minimum ticks required (for time-based stages). |

#### Methods

- `check_success(state: dict) -> bool`: Returns True if all `required` conditions are met.
- `get_partial_progress(state: dict) -> float`: Fraction of required conditions met (0.0-1.0).
- `get_optional_progress(state: dict) -> float`: Fraction of optional conditions met (0.0-1.0).

#### Built-in Stage Criteria

| Stage | Required Conditions | Optional Conditions | Max Ticks |
|-------|---------------------|---------------------|-----------|
| 1 | 4+ oak_log, 8+ planks OR crafted_planks, wooden_pickaxe, 16+ cobblestone | stone_pickaxe, 3+ mob kills, health >= 10 | 6000 |
| 2 | 3+ iron_ingot, iron_pickaxe, bucket | 1+ diamond, 10+ obsidian, 10+ iron_ingot | 12000 |
| 3 | portal_built, entered_nether, fortress_found, 7+ blaze_rod | 10+ blaze_rod, nether_wart | 18000 |
| 4 | 12+ ender_pearl | 16+ ender_pearl, 15+ endermen_killed | 12000 |
| 5 | 12+ eyes_crafted, stronghold_found, end_portal_activated | triangulation_used | 18000 |
| 6 | dragon_killed | one_cycle, 8+ crystals_destroyed, < 6000 ticks | 18000 |

#### Module Functions

```python
from minecraft_sim.stage_criteria import get_stage_criteria, get_all_stages, calculate_total_progress

# Get criteria for a single stage
criteria = get_stage_criteria(3)  # Returns StageCriteria | None

# Get all stages in order
all_stages = get_all_stages()  # Returns list[StageCriteria]

# Calculate overall speedrun progress (0.0 - 1.0)
progress = calculate_total_progress(state, current_stage=3)
# Completed stages = 1.0 each, current stage uses partial progress
```

---

## Progress Tracking

### SpeedrunProgress

Comprehensive dataclass tracking player progress across all 6 speedrun stages. Persists across training episodes for curriculum learning analysis.

```python
from minecraft_sim.progression import SpeedrunProgress, SpeedrunStage

progress = SpeedrunProgress()
```

#### Fields by Stage

**Stage 1: Basic Survival**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wood_collected` | `int` | `0` | Total wood logs collected |
| `stone_collected` | `int` | `0` | Total cobblestone collected |
| `zombies_killed` | `int` | `0` | Zombies killed |
| `first_night_survived` | `bool` | `False` | First night survived |
| `food_eaten` | `int` | `0` | Food items consumed |

**Stage 2: Resource Gathering**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `iron_ore_mined` | `int` | `0` | Iron ore blocks mined |
| `iron_ingots` | `int` | `0` | Smelted iron ingots |
| `diamonds` | `int` | `0` | Diamonds in inventory |
| `has_iron_pickaxe` | `bool` | `False` | Iron pickaxe crafted |
| `has_iron_sword` | `bool` | `False` | Iron sword crafted |
| `has_bucket` | `bool` | `False` | Bucket crafted |
| `has_shield` | `bool` | `False` | Shield crafted |

**Stage 3: Nether Navigation**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `obsidian_collected` | `int` | `0` | Obsidian blocks collected |
| `portal_built` | `bool` | `False` | Nether portal constructed |
| `entered_nether` | `bool` | `False` | Player entered the Nether |
| `fortress_found` | `bool` | `False` | Nether fortress located |
| `blazes_killed` | `int` | `0` | Blazes killed |
| `blaze_rods` | `int` | `0` | Blaze rods in inventory |

**Stage 4: Enderman Hunting**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `endermen_killed` | `int` | `0` | Endermen killed |
| `ender_pearls` | `int` | `0` | Ender pearls in inventory |
| `nether_wart_collected` | `int` | `0` | Nether wart collected |
| `piglins_bartered` | `int` | `0` | Successful piglin barters |

**Stage 5: Stronghold Finding**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `eyes_crafted` | `int` | `0` | Eyes of ender crafted |
| `eyes_used` | `int` | `0` | Eyes thrown for location |
| `stronghold_found` | `bool` | `False` | Stronghold located |
| `stronghold_distance` | `float` | `inf` | Distance to stronghold |
| `portal_room_found` | `bool` | `False` | Portal room found |
| `eyes_placed` | `int` | `0` | Eyes placed in frame (0-12) |
| `portal_activated` | `bool` | `False` | End portal activated |

**Stage 6: Dragon Fight**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `entered_end` | `bool` | `False` | Player entered The End |
| `crystals_destroyed` | `int` | `0` | End crystals destroyed (0-10) |
| `dragon_damage_dealt` | `float` | `0.0` | Total damage to dragon |
| `dragon_phase_changes` | `int` | `0` | Phase transitions observed |
| `dragon_perches` | `int` | `0` | Times dragon perched |
| `dragon_killed` | `bool` | `False` | Dragon defeated |

**Timing and Meta**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `total_ticks` | `int` | `0` | Total game ticks all stages |
| `stage_times` | `dict[int, int]` | `{1:0,...,6:0}` | Ticks per stage |
| `deaths` | `int` | `0` | Total death count |
| `stage_deaths` | `dict[int, int]` | `{1:0,...,6:0}` | Deaths per stage |
| `current_stage` | `int` | `1` | Currently active stage |
| `episode_count` | `int` | `0` | Training episodes |

#### Methods

```python
# Reset per-episode counters, preserve cumulative stats
progress.reset_episode()

# Get stage completion percentage (0.0-1.0)
completion = progress.get_stage_completion(SpeedrunStage.NETHER)

# Get overall speedrun progress across all stages
overall = progress.get_overall_progress()

# Serialize to/from dict
data = progress.to_dict()
restored = SpeedrunProgress.from_dict(data)

# Save/load to disk
progress.save("progress.json")
loaded = SpeedrunProgress.load("progress.json")

# Get current progress snapshot as flat dict
snapshot = progress.get_snapshot()
```

---

### ProgressTracker

Real-time progress tracker that updates from game observations. Wraps `SpeedrunProgress` and provides an observation-driven update interface suitable for environment step loops.

```python
from minecraft_sim.progression import ProgressTracker

tracker = ProgressTracker()
```

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `progress` | `SpeedrunProgress` | `SpeedrunProgress()` | Underlying progress data |
| `prev_health` | `float` | `20.0` | Previous health for death detection |
| `prev_dimension` | `int` | `0` | Previous dimension for portal tracking |
| `prev_dragon_health` | `float` | `200.0` | Previous dragon health for damage tracking |

#### Methods

##### `update_from_observation(obs: dict[str, Any]) -> dict[str, float]`

Update progress from a game observation dictionary. Detects deaths, dimension transitions, resource collection, combat kills, and dragon damage.

```python
rewards = tracker.update_from_observation({
    "health": 18.0,
    "dimension": 1,  # nether
    "inventory": {"blaze_rod": 5},
    "fortress_found": True,
})
# Returns: {"entered_nether": 0.4, "fortress_found": 0.3, "blaze_rod_increment": 0.1}
```

**Parameters:**
- `obs` (`dict[str, Any]`): Observation dictionary with player state, inventory, and game flags.

**Returns:**
- `dict[str, float]`: Achievement reward signals unlocked this tick. Keys are milestone names, values are reward magnitudes.

##### `reset() -> None`

Reset tracker state for a new episode.

##### `get_snapshot() -> dict[str, Any]`

Get current progress as a flat dictionary suitable for info dicts.

---

### ProgressWatchdog

Monitors training progress and detects stalled behavior, particularly Stage 2 obsidian collection stalls.

```python
from minecraft_sim.progress_watchdog import ProgressWatchdog, StallAlertConfig

config = StallAlertConfig(stall_window=50, min_obsidian_delta=1)
watchdog = ProgressWatchdog(config)

# In training loop:
obs, reward, terminated, truncated, info = env.step(action)
watchdog.observe(env_id=0, progress_snapshot=info["progress_snapshot"])
```

#### Methods

- `observe(env_id: int, progress_snapshot: dict) -> StallAlert | None`: Feed a progress snapshot. Returns `StallAlert` if stall detected.
- `reset(env_id: int | None = None) -> None`: Reset tracking for an environment (or all if None).
- `get_active_alerts() -> list[StallAlert]`: Get all currently active alerts.

---

### StallAlertConfig

Configuration for obsidian growth stall detection in `ProgressWatchdog`.

```python
from minecraft_sim.progress_watchdog import StallAlertConfig

config = StallAlertConfig(
    stall_window=50,
    min_obsidian_delta=1,
    cooldown_episodes=20,
    target_stage=2,
    on_stall=lambda alert: print(f"STALL: env {alert.env_id}"),
    alert_level="WARNING",
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stall_window` | `int` | `50` | Consecutive episodes without growth before alert. |
| `min_obsidian_delta` | `int` | `1` | Minimum obsidian increase to reset counter. |
| `cooldown_episodes` | `int` | `20` | Suppress further alerts for this many episodes. |
| `target_stage` | `int` | `2` | Stage ID to monitor. |
| `on_stall` | `Callable[[StallAlert], None] \| None` | `None` | Callback on stall. |
| `alert_level` | `str` | `"WARNING"` | Logging level for alerts. |

---

### StallAlert

Frozen dataclass emitted when obsidian growth stalls.

```python
from minecraft_sim.progress_watchdog import StallAlert
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `env_id` | `int` | Environment that stalled |
| `stage_id` | `int` | Stage where stall detected |
| `episodes_since_growth` | `int` | Episodes without obsidian increase |
| `current_obsidian` | `int` | Current obsidian count |
| `last_growth_obsidian` | `int` | Obsidian at last growth event |
| `wall_time_sec` | `float` | Wall-clock seconds since last growth |
| `snapshot` | `dict[str, Any]` | Progress snapshot that triggered alert |

---

## Observation Spaces

### 48-Float Observation (Dragon Fight)

Used by `DragonFightEnv`, `VecDragonFightEnv`, `SB3VecDragonFightEnv`. All values normalized to [0, 1].

The C++ `Observation` struct provides the raw layout:

| Index | Field | Normalization | Description |
|-------|-------|---------------|-------------|
| **Player (0-15)** | | | |
| 0 | `pos_x` | / world_size | X position |
| 1 | `pos_y` | / 256 | Y position |
| 2 | `pos_z` | / world_size | Z position |
| 3 | `vel_x` | / max_speed | X velocity |
| 4 | `vel_y` | / max_speed | Y velocity |
| 5 | `vel_z` | / max_speed | Z velocity |
| 6 | `yaw` | / 360 | Yaw angle |
| 7 | `pitch` | / 90 | Pitch angle |
| 8 | `health` | / 20 | Player health |
| 9 | `hunger` | / 20 | Player hunger |
| 10 | `on_ground` | {0, 1} | Grounded flag |
| 11 | `attack_ready` | [0, 1] | Attack cooldown progress |
| 12 | `weapon` | / 3 | Weapon slot (0=hand, 1=sword, 2=bow) |
| 13 | `arrows` | / 64 | Arrow count |
| 14 | `arrow_charge` | [0, 1] | Bow charge level |
| 15 | `is_burning` | {0, 1} | On fire flag |
| **Dragon (16-31)** | | | |
| 16 | `dragon_health` | / 200 | Dragon HP |
| 17 | `dragon_x` | relative | Dragon X (relative to player) |
| 18 | `dragon_y` | relative | Dragon Y |
| 19 | `dragon_z` | relative | Dragon Z |
| 20 | `dragon_vel_x` | / max | Dragon velocity X |
| 21 | `dragon_vel_y` | / max | Dragon velocity Y |
| 22 | `dragon_vel_z` | / max | Dragon velocity Z |
| 23 | `dragon_yaw` | / 360 | Dragon yaw |
| 24 | `dragon_phase` | / 6 | Phase enum (0-3 primary) |
| 25 | `dragon_dist` | / max_dist | Distance to dragon |
| 26 | `dragon_dir_x` | [-1, 1] | Direction to dragon X |
| 27 | `dragon_dir_z` | [-1, 1] | Direction to dragon Z |
| 28 | `can_hit_dragon` | {0, 1} | In melee range flag |
| 29 | `dragon_attacking` | {0, 1} | Dragon attacking player |
| 30 | `burn_time_remaining` | / max | Dragon fire ticks |
| 31 | `reserved` | - | Reserved |
| **Environment (32-47)** | | | |
| 32 | `crystals_remaining` | / 10 | Active crystals count |
| 33 | `nearest_crystal_dist` | / max | Distance to nearest crystal |
| 34 | `nearest_crystal_dir_x` | [-1, 1] | Direction to crystal X |
| 35 | `nearest_crystal_dir_z` | [-1, 1] | Direction to crystal Z |
| 36 | `nearest_crystal_y` | / 256 | Crystal Y position |
| 37 | `portal_active` | {0, 1} | Exit portal spawned |
| 38 | `portal_dist` | / max | Distance to portal |
| 39 | `time_remaining` | [0, 1] | Fraction of episode remaining |
| 40 | `total_damage_dealt` | / 200 | Cumulative dragon damage |
| 41-47 | reserved | - | Reserved for future use |

---

### 256-Float Observation (Full Speedrun)

Used by `SpeedrunEnv`. All values normalized to [0, 1] except where noted.

#### Region: Player State [0-31]

| Index | Field | Description |
|-------|-------|-------------|
| 0-2 | position x, y, z | Normalized to world bounds |
| 3-5 | velocity x, y, z | Normalized to max speed |
| 6-7 | yaw, pitch | Normalized to [-1, 1] |
| 8 | health | 0-1 (maps to 0-20 HP) |
| 9 | hunger | 0-1 (maps to 0-20) |
| 10 | saturation | 0-1 (maps to 0-20) |
| 11-12 | armor points, toughness | 0-1 |
| 13 | experience level | 0-1 (maps to 0-30) |
| 14 | experience progress | 0-1 |
| 15 | on_ground | 0 or 1 |
| 16 | in_water | 0 or 1 |
| 17 | in_lava | 0 or 1 |
| 18 | sprinting | 0 or 1 |
| 19 | sneaking | 0 or 1 |
| 20-23 | potion effects | speed, strength, fire_res, regen |
| 24-27 | effect durations | normalized |
| 28 | fall distance | normalized |
| 29 | air supply | 0-1 (maps to 0-300 ticks) |
| 30 | fire ticks | 0-1 (maps to 0-300) |
| 31 | hurt_time | 0-1 (recent damage indicator) |

#### Region: Inventory State [32-63]

| Index | Field | Description |
|-------|-------|-------------|
| 32 | wood_logs | count / 64 |
| 33 | planks | count / 64 |
| 34 | sticks | count / 64 |
| 35 | cobblestone | count / 64 |
| 36 | iron_ingot | count / 64 |
| 37 | gold_ingot | count / 64 |
| 38 | diamond | count / 64 |
| 39 | obsidian | count / 64 |
| 40 | blaze_rod | count / 64 |
| 41 | ender_pearl | count / 16 |
| 42 | eye_of_ender | count / 12 |
| 43 | flint | count / 64 |
| 44 | gravel | count / 64 |
| 45 | food_count | any food, count / 64 |
| 46 | has_crafting_table | 0 or 1 |
| 47 | has_furnace | 0 or 1 |
| 48 | has_wooden_pickaxe | 0 or 1 |
| 49 | has_stone_pickaxe | 0 or 1 |
| 50 | has_iron_pickaxe | 0 or 1 |
| 51 | has_diamond_pickaxe | 0 or 1 |
| 52 | has_sword | 0 or 1 (any) |
| 53 | sword_material | 0/0.25/0.5/0.75/1.0 (none/wood/stone/iron/diamond) |
| 54 | has_bow | 0 or 1 |
| 55 | arrow_count | count / 64 |
| 56 | has_shield | 0 or 1 |
| 57 | has_bed | 0 or 1 |
| 58 | has_bucket | 0 or 1 |
| 59 | bucket_type | 0=empty, 0.5=water, 1.0=lava |
| 60 | hotbar_slot | slot / 8 |
| 61 | has_flint_and_steel | 0 or 1 |
| 62 | armor_equipped | pieces / 4 |
| 63 | total_slots_used | used / 36 |

#### Region: Local Environment [64-127]

| Index | Field | Description |
|-------|-------|-------------|
| 64-79 | block_grid_feet | 4x4 block types at feet level |
| 80-95 | block_grid_head | 4x4 block types at head level |
| 96-103 | entity_types | 8 nearest entity type encodings |
| 104-111 | entity_distances | 8 nearest entity distances |
| 112-119 | entity_angles | 8 nearest entity yaw angles |
| 120 | nearest_hostile_distance | Closest hostile mob distance |
| 121 | nearest_hostile_type | Hostile type encoding |
| 122 | nearest_item_distance | Closest item entity distance |
| 123 | nearest_item_type | Item type encoding |
| 124 | biome_type | Biome encoding |
| 125 | light_level | 0-1 (maps to 0-15) |
| 126 | time_of_day | 0-1 (maps to 0-24000) |
| 127 | dimension | 0=overworld, 0.5=nether, 1.0=end |

#### Region: Goal-Specific State [128-191]

| Index | Field | Stage | Description |
|-------|-------|-------|-------------|
| 128 | zombies_killed | 1 | Kill progress |
| 129 | skeletons_killed | 1 | Kill progress |
| 130 | wood_mined | 1 | Resource progress |
| 131 | survival_time | 1 | Time survived |
| 132 | iron_obtained | 2 | Iron progress |
| 133 | diamonds_obtained | 2 | Diamond progress |
| 134 | has_nether_portal | 2 | Portal built |
| 135 | in_nether | 3 | Dimension flag |
| 136 | fortress_found | 3 | Discovery flag |
| 137 | blaze_kills | 3 | Kill count |
| 138 | distance_to_fortress | 3 | Normalized distance |
| 139 | pearls_obtained | 4 | Pearl count / 12 |
| 140 | eyes_crafted | 4 | Eye count / 12 |
| 141 | enderman_nearby | 4 | Detection flag |
| 142 | eye_thrown | 4-5 | Eye thrown flag |
| 143 | stronghold_found | 4-5 | Discovery flag |
| 144 | distance_to_stronghold | 5 | Normalized distance |
| 145 | portal_frame_filled | 5 | Filled / 12 |
| 160-175 | objective_direction | All | 16 cardinal directions to objective |
| 176-191 | objective_distances | All | Distance buckets to objectives |

#### Region: Dragon State [192-223]

| Index | Field | Description |
|-------|-------|-------------|
| 192 | dragon_health | 0-1 (maps to 0-200) |
| 193-195 | dragon_position | Relative to player, normalized |
| 196-198 | dragon_velocity | Normalized |
| 199 | dragon_phase | 0-1 (maps to phases 0-6) |
| 200 | dragon_target_is_player | Flag |
| 201 | dragon_distance | Normalized |
| 202 | dragon_angle | Yaw to dragon, normalized |
| 203 | dragon_pitch | Pitch to dragon, normalized |
| 204 | dragon_perching | Perching flag |
| 205 | dragon_charging | Charging flag |
| 206 | dragon_breath_active | Breath cloud flag |
| 207 | can_hit_dragon | Melee range flag |
| 208 | crystals_remaining | count / 10 |
| 209-218 | crystal_destroyed[0-9] | Per-crystal status flags |
| 219 | nearest_crystal_distance | Normalized |
| 220 | nearest_crystal_angle | Normalized |
| 221 | on_obsidian_pillar | Position flag |
| 222 | exit_portal_active | Portal spawned flag |
| 223 | dragon_fight_complete | Victory flag |

#### Region: Dimension/Portal State [224-255]

| Index | Field | Description |
|-------|-------|-------------|
| 224 | near_nether_portal | Proximity flag |
| 225 | portal_distance | Normalized |
| 226 | portal_alignment | Frame alignment score |
| 227 | in_portal_cooldown | Cooldown flag |
| 228 | near_end_portal | Proximity flag |
| 229 | end_portal_activated | Activation flag |
| 230 | void_below | End dimension void check |
| 231 | void_distance | Distance to void |
| 232-239 | action_history[0-7] | Last 8 actions (action / 31) |
| 240-247 | reward_history[0-7] | Last 8 rewards (clipped, shifted) |
| 248 | episode_progress | step / max_steps |
| 249 | stage_progress | objectives_done / total |
| 250 | deaths_this_episode | deaths / 10 |
| 251 | total_reward_this_episode | clipped, shifted to [0, 1] |
| 252 | current_stage_id | (stage - 1) / 5 |
| 253 | is_terminal_state | Flag |
| 254 | success_flag | Flag |
| 255 | reserved | - |

---

### Observation Encoding/Decoding

```python
from minecraft_sim.observations import (
    ObservationEncoder,
    MinecraftObservation,
    PlayerState,
    DragonState,
    EntityAwareness,
    InventorySummary,
    VoxelGrid,
    RayCastDistances,
    CompactObservationEncoder,
    CompactObservationDecoder,
    CompactPlayerState,
    CompactInventoryState,
    create_observation_from_c_struct,
    decode_flat_observation,
)

# Encode full observation (~4500 floats)
encoder = ObservationEncoder(include_voxels=False)
obs = MinecraftObservation()
array = encoder.encode(obs)  # Shape (172,) without voxels

# Batch encoding
batch = encoder.encode_batch(observations)  # Shape (N, 172)

# Dict format for multi-head networks
batch_dict = encoder.encode_batch_dict(observations)
# {'continuous': (N, 144), 'inventory': (N, 16), 'voxels': (N, 4096), 'hotbar': (N, 9)}

# Decode flat 256-float vector to semantic dict
decoded = decode_flat_observation(stage_id=4, obs_vector)
# {'player': {...}, 'inventory': {...}, 'dragon': {...}}

# Compact 256-float encoding
compact_encoder = CompactObservationEncoder()
compact_decoder = CompactObservationDecoder()
```

---

### Compact Observation (256 floats)

The `CompactPlayerState` and `CompactInventoryState` dataclasses provide structured access to the 256-float observation:

```python
from minecraft_sim.observations import CompactPlayerState, CompactInventoryState

player = CompactPlayerState(
    x=0.5, y=0.25, z=0.5,
    vx=0.0, vy=0.0, vz=0.0,
    yaw=0.0, pitch=0.0,
    health=1.0, hunger=1.0,
    on_ground=True,
)

inventory = CompactInventoryState(
    wood_logs=0.1,
    iron_ingot=0.05,
    has_iron_pickaxe=True,
)
```

---

## Action Spaces

### 17-Action Discrete (Dragon Fight)

Used by `DragonFightEnv`, `VecDragonFightEnv`, `SB3VecDragonFightEnv`.

| Index | Action | Description |
|-------|--------|-------------|
| 0 | NO_OP | No action |
| 1 | FORWARD | Move forward |
| 2 | BACK | Move backward |
| 3 | LEFT | Strafe left |
| 4 | RIGHT | Strafe right |
| 5 | JUMP | Jump |
| 6 | SPRINT | Sprint forward |
| 7 | SNEAK | Sneak |
| 8 | ATTACK | Attack/left click |
| 9 | USE | Use item/right click |
| 10 | LOOK_UP | Look up |
| 11 | LOOK_DOWN | Look down |
| 12 | LOOK_LEFT | Look left |
| 13 | LOOK_RIGHT | Look right |
| 14 | HOTBAR_NEXT | Next hotbar slot |
| 15 | HOTBAR_PREV | Previous hotbar slot |
| 16 | DROP | Drop item |

---

### 32-Action Discrete (Speedrun)

Used by `SpeedrunEnv`. Defined in `SpeedrunAction` class.

| Index | Action | Delta | Description |
|-------|--------|-------|-------------|
| 0 | NOOP | - | No action |
| 1 | FORWARD | - | Move forward |
| 2 | BACK | - | Move backward |
| 3 | LEFT | - | Strafe left |
| 4 | RIGHT | - | Strafe right |
| 5 | FORWARD_LEFT | - | Diagonal movement |
| 6 | FORWARD_RIGHT | - | Diagonal movement |
| 7 | JUMP | - | Jump |
| 8 | JUMP_FORWARD | - | Jump + forward |
| 9 | ATTACK | - | Attack/left click |
| 10 | ATTACK_FORWARD | - | Attack while moving forward |
| 11 | SPRINT_TOGGLE | - | Toggle sprint |
| 12 | LOOK_LEFT | 5 deg | Turn left (fine) |
| 13 | LOOK_RIGHT | 5 deg | Turn right (fine) |
| 14 | LOOK_UP | 7.5 deg | Look up (fine) |
| 15 | LOOK_DOWN | 7.5 deg | Look down (fine) |
| 16 | LOOK_LEFT_FAST | 45 deg | Turn left (coarse) |
| 17 | LOOK_RIGHT_FAST | 45 deg | Turn right (coarse) |
| 18 | LOOK_UP_FAST | 30 deg | Look up (coarse) |
| 19 | LOOK_DOWN_FAST | 30 deg | Look down (coarse) |
| 20 | USE_ITEM | - | Right click (place, eat, shoot, portal) |
| 21 | DROP_ITEM | - | Drop held item |
| 22-30 | HOTBAR_1-9 | - | Select hotbar slot 1-9 |
| 31 | CRAFT | - | Context-sensitive quick craft |

---

### Per-Stage Action Spaces

| Stage | Actions | Notable Actions |
|-------|---------|-----------------|
| 1: Basic Survival | 24 | Movement, mine, craft |
| 2: Resource Gathering | 24 | Movement, mine, smelt |
| 3: Nether Navigation | 28 | + Portal interaction, block placement |
| 4: Enderman Hunting | 28 | + Look at/away, equip helmet, place water |
| 5: Stronghold Finding | 28 | + Throw eye, track trajectory, mark waypoint |
| 6: Dragon Fight | 20 | + Shoot bow, place bed, throw pearl, pillar up |

---

## Reward Functions

### create_reward_shaper

Factory function that creates a stateful reward shaper for a specific stage.

```python
from minecraft_sim.reward_shaping import create_reward_shaper

shaper = create_reward_shaper(stage_id=1)
reward = shaper(state_dict)
```

**Parameters:**
- `stage_id` (`int`): Stage number (1-6).

**Returns:**
- `Callable[[dict[str, Any]], float]`: Stateful function that takes a state dict and returns shaped reward.

The returned callable has two extra attributes:
- `.stats` (`RewardStats`): Running statistics.
- `.reset()`: Reset internal milestone tracking.

---

### CompositeRewardShaper

Manages transitions between stage shapers for end-to-end training.

```python
from minecraft_sim.reward_shaping import CompositeRewardShaper

composite = CompositeRewardShaper(initial_stage=1)
reward = composite.shape_reward(state)

if state.get("stage_complete"):
    composite.advance_stage()
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `set_stage(stage_id)` | `None` | Set current stage (1-6). |
| `advance_stage()` | `bool` | Advance to next stage. False if at stage 6. |
| `shape_reward(state)` | `float` | Shape reward for current stage. |
| `reset(stage_id=None)` | `None` | Reset all shaper state. |
| `get_stats()` | `RewardStats \| None` | Stats for current stage shaper. |

---

### Stage-Specific Rewards

Each stage shaper provides three categories of rewards:

#### 1. Milestone Rewards (One-Time)

Given once per episode when a condition is first met. Prevents reward hacking.

**Stage 1 Milestones:**
- `first_wood` (0.2), `crafting_table` (0.15), `wooden_pickaxe` (0.3), `stone_pickaxe` (0.3), `first_iron_ore` (0.25)

**Stage 2 Milestones:**
- `first_iron_ingot` (0.2), `iron_pickaxe` (0.35), `bucket` (0.3), `first_diamond` (0.3), `obsidian_x10` (0.25)

**Stage 3 Milestones:**
- `entered_nether` (0.4), `fortress_found` (0.4), `first_blaze_kill` (0.3), `blaze_rod_x7` (0.25)

**Stage 4 Milestones:**
- `first_enderman_kill` (0.25), `pearl_x12` (0.25), `first_eye` (0.2), `eye_x12` (0.25), `portal_activated` (1.5)

**Stage 5 Milestones:**
- `first_eye_throw` (0.15), `stronghold_entered` (0.3), `portal_room_found` (0.35), `portal_activated` (0.5)

**Stage 6 Milestones:**
- `entered_end` (0.3), `all_crystals` (0.3), `dragon_half_health` (0.2), `dragon_killed` (1.0), `one_cycle` (0.5), `fast_kill` (0.3)

#### 2. Progressive Rewards (Incremental)

Scaled rewards proportional to resource collection, with diminishing returns.

```python
# Example: iron bonus caps at 0.3
iron_bonus = min(iron_count * 0.015, 0.3)
# Only the delta from previous step is awarded
reward += iron_bonus - prev_iron_bonus
```

#### 3. Penalties

| Penalty | Per-Stage Range | Description |
|---------|-----------------|-------------|
| Time penalty | -0.0001 to -0.0002 | Per-tick pressure |
| Death penalty | -0.8 to -2.0 | On player death |
| Damage penalty | 0.015-0.025 per HP | Damage taken |
| Fire/lava penalty | 1.5x damage penalty | Environmental damage (Stage 3) |
| Void proximity | -0.1 | Near void (Stage 6) |

#### Configurable Parameters

All reward weights are class variables on the stage environment classes and can be overridden:

```python
env = BasicSurvivalEnv()
env.REWARD_WOOD_MINED = 0.5  # Increase wood reward
```

Or via `StageConfig.reward_scale` for global scaling:

```python
config = StageConfig(reward_scale=2.0)  # Double all rewards
env = BasicSurvivalEnv(config=config)
```

---

### RewardStats

Statistics tracked by each reward shaper for debugging and logging.

```python
from minecraft_sim.reward_shaping import RewardStats

@dataclass
class RewardStats:
    total_reward: float = 0.0
    milestone_rewards: float = 0.0
    progressive_rewards: float = 0.0
    penalties: float = 0.0
    stage_completion_bonus: float = 0.0
    milestones_achieved: list[str] = field(default_factory=list)
```

Access via shaper:
```python
shaper = create_reward_shaper(1)
# ... run episodes ...
print(shaper.stats.milestones_achieved)
print(f"Total: {shaper.stats.total_reward}")
```

---

## Low-Level API

### mc189_core.MC189Simulator

The C++ backend providing GPU-accelerated Minecraft simulation via Vulkan compute shaders.

```python
import mc189_core

config = mc189_core.SimulatorConfig()
config.num_envs = 64
config.shader_dir = "path/to/shaders"

sim = mc189_core.MC189Simulator(config)
```

#### Constructor

```python
MC189Simulator(config: SimulatorConfig)
```

Creates a simulator instance with Vulkan context, loads shaders, allocates GPU buffers.

**Raises:** Runtime error if Vulkan initialization fails or shaders not found.

#### Methods

##### `step(actions: NDArray[np.int32]) -> None`

Execute one simulation tick for all environments.

```python
actions = np.zeros(num_envs, dtype=np.int32)
sim.step(actions)
```

**Parameters:**
- `actions`: Array of shape (num_envs,) with action indices (0-16 for dragon fight).

##### `reset(env_id: int = 0xFFFFFFFF, seed: int = 0) -> None`

Reset environment(s) to initial state.

```python
sim.reset()              # Reset all environments
sim.reset(env_id=5)      # Reset specific environment
sim.reset(seed=42)       # Reset all with specific seed
sim.reset(env_id=0, seed=123)  # Reset env 0 with seed
```

**Parameters:**
- `env_id`: Environment index to reset, or `0xFFFFFFFF` to reset all.
- `seed`: World seed. 0 generates a random seed. Same seed guarantees deterministic outcomes.

##### `get_seed(env_id: int = 0) -> int`

Get the current world seed for an environment.

##### `get_observations() -> NDArray[np.float32]`

Get observations for all environments.

```python
obs = sim.get_observations()  # Shape: (num_envs * 48,) flattened
obs = obs.reshape(num_envs, 48)
```

##### `get_rewards() -> NDArray[np.float32]`

Get rewards for all environments. Shape: (num_envs,).

##### `get_dones() -> NDArray[np.uint8]`

Get done flags for all environments. Shape: (num_envs,).

##### `num_envs() -> int`

Number of parallel environments.

##### `obs_dim() -> int`

Observation dimension (48 for dragon fight shader).

---

### Shader Loading

The simulator loads Vulkan compute shaders (.spv files) from the configured shader directory. Each stage uses a subset of shaders defined in `STAGE_SHADER_SETS`:

```python
from minecraft_sim.curriculum import get_shader_set_for_stage, StageID

shaders = get_shader_set_for_stage(StageID.END_FIGHT)
# ['dragon_fight_mvk', 'dragon_ai_full', 'end_terrain']
```

#### Available Shaders

| Shader | Stage | Description |
|--------|-------|-------------|
| `dragon_fight_mvk.comp` | 6 | Main dragon fight simulation |
| `dragon_ai_full.comp` | 6 | Dragon AI with full phase transitions |
| `overworld_gen.comp` | 1 | Overworld terrain generation |
| `furnace_tick.comp` | 2 | Furnace smelting simulation |
| `item_physics.comp` | 2 | Item drop physics |
| `fortress_structure.comp` | 3 | Nether fortress structure generation |
| `stronghold_gen.comp` | 5 | Stronghold structure generation |
| `village_gen.comp` | 1-2 | Village structure generation |
| `bed_explosion.comp` | 6 | Bed explosion in End dimension |
| `crystal_combat.comp` | 6 | End crystal destruction mechanics |
| `ender_pearl.comp` | 4-5 | Ender pearl teleportation |
| `batch_actions.comp` | All | Batch action decoding |
| `reward_computation.comp` | All | Reward signal computation |
| `game_tick.comp` | All | Core game loop |

#### Stage-Specific Shader Configuration

```python
config = mc189_core.SimulatorConfig()
config.shader_set = ["dragon_fight_mvk", "dragon_ai_full"]  # Only load these
sim = mc189_core.MC189Simulator(config)
```

---

### Buffer Management

The simulator manages GPU buffers corresponding to Vulkan descriptor set bindings:

| Binding | Buffer | C++ Type | Description |
|---------|--------|----------|-------------|
| 0 | `player_buffer_` | `Player[num_envs]` | Player state |
| 1 | `input_buffer_` | `InputState[num_envs]` | Per-tick inputs |
| 2 | `dragon_buffer_` | `Dragon[num_envs]` | Dragon state |
| 3 | `crystal_buffer_` | `Crystal[num_envs * 10]` | Crystal states |
| 4 | `game_state_buffer_` | `GameState[num_envs]` | Game tick state |
| 5 | `observation_buffer_` | `float[num_envs * 48]` | Observations |
| 6 | `reward_buffer_` | `float[num_envs]` | Rewards |
| 7 | `done_buffer_` | `uint8_t[num_envs]` | Done flags |

Buffer classes exposed to Python:

```python
import mc189_core

# Available but typically managed internally
VulkanContext = mc189_core.VulkanContext
VulkanContextConfig = mc189_core.VulkanContextConfig
ComputePipeline = mc189_core.ComputePipeline
Buffer = mc189_core.Buffer
BufferManager = mc189_core.BufferManager
BatchExecutor = mc189_core.BatchExecutor
```

---

### C++ Data Structures

#### Player (64 bytes, 16-byte aligned)

```cpp
struct alignas(16) Player {
    float position[3];           // World position
    float yaw;                   // Facing direction
    float velocity[3];           // Movement velocity
    float pitch;                 // Vertical look angle
    float health;                // 0-20 HP
    float hunger;                // 0-20
    float saturation;            // Hidden hunger buffer
    float exhaustion;            // Hunger drain rate
    uint32_t flags;              // Bit flags (on_ground, sprinting, etc.)
    uint32_t invincibility_timer; // Post-damage invincibility
    uint32_t attack_cooldown;    // Weapon cooldown ticks
    uint32_t weapon_slot;        // 0=hand, 1=sword, 2=bow
    float arrow_charge;          // Bow charge [0, 1]
    uint32_t arrows;             // Arrow count
    uint32_t reserved[2];
};
```

#### Dragon (80 bytes, 16-byte aligned)

```cpp
struct alignas(16) Dragon {
    float position[3];           // World position
    float yaw;
    float velocity[3];
    float pitch;
    float health;                // 0-200 HP
    uint32_t phase;              // 0=circling, 1=strafing, 2=perching, 3=breath
    uint32_t phase_timer;        // Ticks in current phase
    uint32_t target_pillar;      // Current target pillar index
    float target_position[3];    // AI navigation target
    float breath_timer;          // Breath attack duration
    uint32_t perch_timer;        // Time on fountain
    uint32_t attack_cooldown;
    uint32_t flags;
    float circle_angle;          // Current circling angle
    uint32_t reserved[3];
};
```

#### Crystal (16 bytes, 16-byte aligned)

```cpp
struct alignas(16) Crystal {
    float position[3];           // Pillar top position
    float is_alive;              // 1.0 = alive, 0.0 = destroyed
};
```

#### Constants

```cpp
constexpr uint32_t NUM_CRYSTALS = 10;
constexpr float DRAGON_MAX_HEALTH = 200.0f;
constexpr float END_SPAWN_Y = 64.0f;
constexpr float PILLAR_HEIGHT = 76.0f;
constexpr size_t OBSERVATION_SIZE = 48;
```

---

## Factory Functions

```python
from minecraft_sim import make, make_vec
from minecraft_sim.speedrun_env import make_speedrun_env, make_stage_env
from minecraft_sim.speedrun_vec_env import make_speedrun_vec_env
from minecraft_sim.stage_envs import make_stage_env as make_individual_stage_env, get_stage_info

# Simple single environment
env = make(seed=42)

# Vectorized environment
vec_env = make_vec(num_envs=64, base_seed=42)

# Speedrun environment with curriculum
env = make_speedrun_env(stage_id=1, auto_advance=True)

# Stage-locked environment (no advancement)
env = make_stage_env(stage_id=3, shader_dir="/path/to/shaders")

# Vectorized speedrun
vec_env = make_speedrun_vec_env(
    num_envs=64,
    initial_stage="BASIC_SURVIVAL",
    auto_curriculum=True,
)

# Individual stage environment
env = make_individual_stage_env(stage=4, config=StageConfig())

# Stage metadata
info = get_stage_info(stage=6)
# {'name': 'Dragon Fight', 'obs_size': 64, 'action_size': 20, ...}
```

---

## Constants

```python
from minecraft_sim import (
    OBSERVATION_SIZE,      # 48 (dragon fight observation)
    ACTION_SIZE,           # 17 (dragon fight actions)
    MAX_BATCH_SIZE,        # 4096 (max parallel environments)
    TICKS_PER_SECOND,      # 20 (Minecraft standard TPS)
    MC189_ACTION_COUNT,    # 17 (same as ACTION_SIZE)
)

# Version and availability checks
from minecraft_sim import check_cpp_module, get_version_info

if check_cpp_module():
    print("C++ GPU simulator available")

print(get_version_info())
# {'minecraft_sim': '0.1.0', 'cpp_module': 'available',
#  'gymnasium': '1.0.0', 'numpy': '2.0.0'}
```

#### Normalization Constants

| Constant | Value | Usage |
|----------|-------|-------|
| Max player health | 20.0 | `obs[8] = health / 20` |
| Max hunger | 20.0 | `obs[9] = hunger / 20` |
| Dragon max health | 200.0 | `obs[16] = dragon_hp / 200` |
| Max crystals | 10 | `obs[32] = crystals / 10` |
| Max stack size | 64 | Inventory normalization |
| Max pearl stack | 16 | `obs[41] = pearls / 16` |
| Max eyes | 12 | `obs[42] = eyes / 12` |
| Max episode ticks | 36000 | Stage 6 default |
| World height | 256 | Y-coordinate normalization |

---

## Stage Configuration Format (YAML)

Stage configurations are loaded from YAML files in the `stage_configs/` directory:

```yaml
# stage_configs/stage_6_end_fight.yaml
id: 6
name: End Fight
description: Defeat the Ender Dragon

objectives:
  - Enter the End portal
  - Destroy all End crystals
  - Kill the Ender Dragon

spawn:
  biome: the_end
  time_of_day: 0
  weather: clear
  random_position: false
  position: [0.0, 64.0, 0.0]
  inventory:
    diamond_sword: 1
    bow: 1
    arrow: 64
  health: 20.0
  hunger: 20.0

rewards:
  sparse_reward: 100.0
  dense_rewards:
    destroy_crystal: 3.0
    damage_dragon: 0.5
    dragon_killed: 50.0
  penalty_per_death: -5.0
  penalty_per_tick: -0.0002
  exploration_bonus: 0.0

termination:
  max_ticks: 36000
  max_deaths: 1
  success_conditions:
    - dragon_killed
  failure_conditions:
    - death_count >= max_deaths
    - fell_into_void

prerequisites: [5]
difficulty: 8
expected_episodes: 3000
curriculum_threshold: 0.5

action_space:
  - move_forward
  - attack
  - shoot_bow

observation_space:
  - player_position
  - dragon_position
  - dragon_health

metadata:
  dragon_phases:
    circling: "Dragon flies around pillars"
    perching: "Dragon lands on fountain"
  crystal_count: 10
  caged_crystals: 2
```

---

## Examples

### Basic Training Loop

```python
from minecraft_sim.speedrun_env import SpeedrunEnv

env = SpeedrunEnv(stage_id=1, auto_advance=True)
obs, info = env.reset(seed=42)

for episode in range(1000):
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    obs, info = env.reset()
    if info.get("stage_advanced"):
        print(f"Advanced to stage {env.stage_id}")

env.close()
```

### Vectorized Training with SB3

```python
from minecraft_sim import SB3VecDragonFightEnv
from stable_baselines3 import PPO

env = SB3VecDragonFightEnv(num_envs=64)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    verbose=1,
)

model.learn(total_timesteps=10_000_000)
model.save("dragon_ppo")
env.close()
```

### Per-Environment Curriculum Training

```python
from minecraft_sim.speedrun_vec_env import SpeedrunVecEnv
from minecraft_sim.curriculum import StageID
import numpy as np

env = SpeedrunVecEnv(
    num_envs=64,
    initial_stage=StageID.BASIC_SURVIVAL,
    auto_curriculum=True,
    success_threshold=0.7,
    min_episodes_for_advance=100,
)

obs = env.reset()

for step in range(1_000_000):
    actions = np.random.randint(0, 17, size=64)
    obs, rewards, dones, infos = env.step(actions)

    for i, info in enumerate(infos):
        if "curriculum_advanced" in info:
            print(f"Env {i} advanced to {info['new_stage_name']}")

    if step % 10000 == 0:
        stats = env.get_curriculum_stats()
        dist = env.get_stage_distribution()
        print(f"Step {step}: {dist}")

env.close()
```

### Individual Stage Training

```python
from minecraft_sim.stage_envs import (
    NetherNavigationEnv,
    StageConfig,
    Difficulty,
)

config = StageConfig(
    max_episode_ticks=18000,
    difficulty=Difficulty.HARD,
    death_penalty=-2.0,
    reward_scale=1.5,
)

env = NetherNavigationEnv(config=config)
obs, info = env.reset()

for episode in range(500):
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    if info.get("episode", {}).get("success"):
        print(f"Episode {episode}: Blaze rods collected!")

    obs, info = env.reset()

env.close()
```

### Reward Shaping Analysis

```python
from minecraft_sim.reward_shaping import create_reward_shaper, CompositeRewardShaper

# Single stage shaper
shaper = create_reward_shaper(stage_id=3)

state = {
    "health": 18.0,
    "hunger": 15.0,
    "in_nether": True,
    "fortress_found": True,
    "inventory": {"blaze_rod": 5},
    "blazes_killed": 8,
}

reward = shaper(state)
print(f"Reward: {reward:.4f}")
print(f"Milestones: {shaper.stats.milestones_achieved}")
print(f"Progressive: {shaper.stats.progressive_rewards:.4f}")
print(f"Penalties: {shaper.stats.penalties:.4f}")

# Composite shaper for full speedrun
composite = CompositeRewardShaper(initial_stage=1)
composite.set_stage(3)
reward = composite.shape_reward(state)
composite.advance_stage()  # Move to stage 4
```

### Low-Level Simulator Access

```python
import mc189_core
import numpy as np

config = mc189_core.SimulatorConfig()
config.num_envs = 128
config.shader_dir = "/path/to/minecraft_sim/cpp/shaders"

sim = mc189_core.MC189Simulator(config)
sim.reset(seed=42)

# Run 1000 ticks with random actions
for tick in range(1000):
    actions = np.random.randint(0, 17, size=128).astype(np.int32)
    sim.step(actions)

    obs = sim.get_observations().reshape(128, 48)
    rewards = sim.get_rewards()
    dones = sim.get_dones()

    # Check for completed episodes
    done_envs = np.where(dones)[0]
    for env_id in done_envs:
        dragon_health = obs[env_id, 16] * 200.0
        print(f"Env {env_id} done. Dragon health: {dragon_health:.1f}")
```
