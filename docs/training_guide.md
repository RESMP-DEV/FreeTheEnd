# Training Guide

Everything needed to train RL agents on FreeTheEnd, from hardware setup to evaluating a trained model.

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [Software Setup](#2-software-setup)
3. [First Training Run (Stage 1 Only)](#3-first-training-run-stage-1-only)
4. [Curriculum Training (All Stages)](#4-curriculum-training-all-stages)
5. [Full Speedrun Training](#5-full-speedrun-training)
6. [Monitoring with TensorBoard](#6-monitoring-with-tensorboard)
7. [Common Issues and Solutions](#7-common-issues-and-solutions)
8. [Hyperparameter Tuning Tips](#8-hyperparameter-tuning-tips)
9. [Expected Training Curves](#9-expected-training-curves)
10. [From Trained Model to Evaluation](#10-from-trained-model-to-evaluation)

---

## 1. Hardware Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Any Vulkan 1.2 compatible | Apple M1+ or NVIDIA RTX 3060+ |
| VRAM | 4 GB | 8+ GB |
| RAM | 16 GB | 32+ GB |
| Storage | 10 GB free | SSD recommended |
| CPU | 4 cores | 8+ cores |

**Note:** GPU provides best throughput at high environment counts. For CPU-only operation (CI, development), build with `-DCPU_ONLY=ON`.

### GPU Compatibility

The simulator uses Vulkan compute shaders. Supported configurations:

**macOS (MoltenVK)**
- Apple Silicon: M1, M2, M3, M4 (all variants)
- Intel Macs with discrete AMD GPUs

**Linux/Windows**
- NVIDIA: GTX 1060+, RTX series
- AMD: RX 580+, RDNA/RDNA2 series
- Intel: Arc GPUs (limited testing)

### CPU-Only Operation

The per-tick simulation is lightweight (~1300 lines of shader logic, no rendering). Build with `-DCPU_ONLY=ON` for CI pipelines and development without GPU access (see [CPU-Only Build](#cpu-only-build-no-vulkan-required) below).

**Estimated CPU throughput:**

| CPU | Cores | Est. throughput |
|-----|-------|-----------------|
| Apple M1 | 4P | ~1-2M steps/sec |
| Apple M2 | 4P | ~2-3M steps/sec |
| Ryzen 5 5600 | 6 | ~2-4M steps/sec |
| Ryzen 7 7700 | 8 | ~4-6M steps/sec |

For comparison, MineRL achieves ~60 steps/sec. Even a modest CPU is **30,000-100,000× faster** than MineRL.

At small environment counts (16-64), CPU can match GPU performance since shader dispatch overhead dominates. GPU pulls ahead at 256+ environments.

### Throughput by Hardware

| Hardware | Environments | Steps/sec |
|----------|-------------|-----------|
| Apple M1 | 64 | ~25,000 |
| Apple M4 Max | 64 | ~264,000 |
| Apple M4 Max | 256 | ~615,000 |
| RTX 3090 | 64 | ~60,000 |
| RTX 4090 | 256 | ~120,000 |

### Memory Scaling

Each environment consumes approximately 0.5 MB of VRAM:

| Environments | VRAM Usage |
|-------------|------------|
| 64 | ~32 MB |
| 256 | ~128 MB |
| 1024 | ~512 MB |
| 4096 | ~2 GB |
| 8192 | ~4 GB |

---

## 2. Software Setup

### Prerequisites

**macOS**

```bash
# Install Vulkan support via MoltenVK
brew install molten-vk

# Install shader compiler
brew install glslang

# Set Vulkan ICD path
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
```

**Linux (Ubuntu/Debian)**

```bash
# Install Vulkan SDK
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
sudo apt update
sudo apt install vulkan-sdk

# Install NVIDIA drivers if needed
sudo apt install nvidia-driver-535
```

### Python Environment

```bash
# Create virtual environment with Python 3.12
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install numpy gymnasium pyyaml stable-baselines3 tensorboard
```

**Critical:** Always prefix Python commands with `uv run` to ensure Python 3.12:
```bash
uv run python script.py     # Correct
uv run pytest tests/ -v     # Correct
python script.py            # WRONG - may use system Python
```

### Building the C++ Extension

```bash
cd cpp

# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build . --target mc189_core -j$(nproc)

# Copy extension to Python package
cp mc189_core.cpython-*-*.so ../../python/minecraft_sim/
```

### CPU-Only Build (No Vulkan Required)

For CI pipelines or development without GPU access:

```bash
cd cpp
mkdir -p build && cd build

# Configure with CPU_ONLY flag (skips Vulkan dependency)
cmake -DCPU_ONLY=ON -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build . --target mc189_core -j$(nproc)

# Copy extension to Python package
cp mc189_core.cpython-*-*.so ../../python/minecraft_sim/
```

Python usage with CPU backend:
```python
import mc189_core

cfg = mc189_core.SimulatorConfig()
cfg.use_cpu = True
cfg.num_envs = 64

sim = mc189_core.MC189Simulator(cfg)
obs = sim.reset()
sim.step(actions)
```

No `VK_ICD_FILENAMES` or Vulkan SDK needed. The CPU backend runs the same simulation logic without GPU dispatch.

### Verify Installation

```bash
# Set Python path
export PYTHONPATH=$(pwd)/python

# Set Vulkan path (macOS)
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json

# Run core tests
uv run pytest tests/test_discrete_actions.py -v

# Run full test suite
uv run pytest tests/ -v --ignore=tests/test_backend_integration.py
```

Expected output:

```
MC189 GPU Simulator Quick Test
Device: Apple M4 Max
Throughput: 264,000 steps/sec
```

---

## 3. First Training Run (Stage 1 Only)

Start with a simple single-stage training run to verify everything works. Stage 1 (Basic Survival) teaches the agent fundamental Minecraft skills: movement, combat, resource gathering, and crafting.

### Stage 1 Reward Shaping Flow

The reward shaper (`create_stage1_reward_shaper()` in `reward_shaping.py`) provides dense feedback through four signal types:

1. **Penalties (continuous pressure)**
   - Time penalty: `-0.0001` per tick (encourages efficiency)
   - Death: `-1.0` terminal penalty
   - Damage taken: `-0.02 * damage_amount`
   - Hunger loss: `-0.01 * hunger_delta`

2. **Milestone rewards (one-time bonuses)**
   - First wood log collected: `+0.2`
   - 4 wood logs: `+0.1`, 8 logs: `+0.1`
   - First planks crafted: `+0.1`, first sticks: `+0.05`
   - Crafting table: `+0.15`
   - Wooden pickaxe: `+0.3`, wooden sword: `+0.1`, wooden axe: `+0.1`
   - First cobblestone: `+0.15`, 16 cobblestone: `+0.1`
   - Stone pickaxe: `+0.3`, stone sword: `+0.1`
   - Furnace crafted: `+0.2`
   - First coal: `+0.15`, first iron ore: `+0.25`
   - First food item: `+0.1`, first mob kill: `+0.15`

3. **Progressive rewards (diminishing returns per resource)**
   - Wood: `+0.02` per log, capped at `+0.2` total
   - Stone: `+0.005` per block, capped at `+0.2` total
   - Coal: `+0.01` per piece, capped at `+0.15` total
   - Mob kills: `+0.1` per kill (uncapped)
   - Chunk exploration: `+0.01` per new chunk

4. **Stage completion bonus**
   - `+2.0` when `stage_complete` flag is set

### Progress Tracking

The `ProgressTracker` (in `progression.py`) monitors the agent's cumulative progress across episodes:

- **SpeedrunProgress dataclass** stores: `wood_collected`, `stone_collected`, `zombies_killed`, `first_night_survived`, `food_eaten`, plus timing and death counts.
- **`update_from_observation(obs)`** is called each tick, updating inventory counts, detecting deaths, and emitting reward signals for newly achieved milestones.
- **`get_stage_completion(stage=1)`** returns a `[0.0, 1.0]` score based on weighted targets: wood >= 16, stone >= 32, food >= 5, first night survived.
- **`is_stage_complete(stage=1)`** returns `True` when `wood_collected >= 16` AND `stone_collected >= 32`.
- **`to_observation()`** produces a 32-element float32 vector (normalized to [0, 1]) that can augment the policy's input with progress context.

### Success Criteria (from `stage_1_basic_survival.yaml`)

The episode terminates with success when ALL conditions are met:

| Condition | Requirement |
|-----------|-------------|
| Zombies killed | >= 3 |
| Skeletons killed | >= 3 |
| Wood mined | >= 10 logs |
| Wooden pickaxe | In inventory |
| Wooden sword | In inventory |

The episode terminates with failure if:
- Deaths >= 10 (per episode)
- Ticks >= 24,000 (one full Minecraft day)

### Curriculum Advancement

Stage 1 uses a curriculum threshold of `0.7` (70% success rate across recent episodes). Once the agent consistently completes objectives in 70% of episodes, the curriculum manager advances to Stage 2 (Resource Gathering). The minimum episode count before advancement is configurable (default: `expected_episodes = 500`).

### Stage 1 Dense Reward Config (from YAML)

These rewards are defined in the stage config and applied by the environment independently of the reward shaper:

| Signal | Value |
|--------|-------|
| Distance traveled | +0.001 per block |
| New chunk explored | +0.1 |
| Damage dealt | +0.05 per HP |
| Zombie killed | +0.5 |
| Skeleton killed | +0.5 |
| Spider killed | +0.3 |
| Wood mined | +0.1 per log |
| Crafting table crafted | +0.3 |
| Wooden pickaxe crafted | +0.5 |
| Wooden sword crafted | +0.5 |

### Basic PPO Training Script

Save as `train_stage1.py`:

```python
#!/usr/bin/env python3
"""Stage 1: Basic survival training.

Run with: uv run python train_stage1.py
"""
import sys
from pathlib import Path

# Add minecraft_sim to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from minecraft_sim import SB3VecDragonFightEnv
from stable_baselines3 import PPO

# Create environment with 64 parallel instances
env = SB3VecDragonFightEnv(num_envs=64)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./logs/stage1_ppo",
)

# Train for 1M steps
model.learn(total_timesteps=1_000_000)

# Save the model
model.save("stage1_basic_survival")
env.close()
```

### Running the Training

```bash
# Set Vulkan path (macOS)
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json

# Run training
PYTHONPATH=python uv run python train_stage1.py
```

### Expected Output

```
Using cuda device
Wrapping the env with a VecTransposeImage wrapper
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 128      |
|    ep_rew_mean     | -2.34    |
| time/              |          |
|    fps             | 12543    |
|    iterations      | 1        |
|    time_elapsed    | 0        |
|    total_timesteps | 8192     |
---------------------------------
```

---

## 3b. Stage 5: Stronghold Finding

Stage 5 is the most exploration-intensive stage. The agent must craft eyes of ender, use triangulation to locate a stronghold, navigate its corridors to find the portal room, and fill all 12 portal frames to activate the End portal.

### Stronghold Generation

Strongholds are generated by the `stronghold_gen.comp` shader with the following Minecraft 1.8.9 mechanics:

- 3 strongholds per world, arranged in a ring 1408-2688 blocks from the origin
- Spaced 120 degrees apart, with a seed-dependent angular offset
- Y-level ranges from 10 to 40 (underground)
- Each stronghold contains up to 50 rooms: portal room, libraries (up to 2), corridors, prison cells, and fountain rooms
- The portal room always generates at depth 0 (first room in the generation tree)

### Stronghold Tracking via Triangulation

The agent locates the stronghold by throwing eyes of ender and tracking their trajectory. The observation space provides:

| Observation | Description |
|-------------|-------------|
| `eye_trajectory` | Direction vector the thrown eye travels |
| `eye_landed_position` | Where the eye dropped (if it went underground) |
| `triangulation_waypoints` | Previously recorded throw positions and directions |
| `estimated_stronghold_pos` | Computed intersection of two direction lines |

The triangulation procedure is:
1. Throw an eye at position A, record direction vector
2. Travel 200-500 blocks perpendicular to the eye's direction
3. Throw a second eye at position B, record direction vector
4. Compute intersection of the two direction lines to estimate stronghold XZ
5. When the eye descends instead of floating, dig straight down

Speedrun shortcut: the "blind travel" method exploits the stronghold ring geometry. Travel ~1800 blocks from the origin in any direction to land within the first ring (1408-2688 blocks).

### Portal Activation Criteria

The End portal has 12 frames arranged in a 5x5 ring (3 per side: North, South, East, West). Activation requires all 12 frames to contain an eye of ender. On world generation, each frame has a 10% chance of already containing an eye.

The agent must:
1. Locate the portal room within the stronghold (`portal_room_found` flag)
2. Place eyes in all empty frames (`portal_frames_filled` counter, 0-12)
3. Once all 12 are filled, the portal activates (`end_portal_activated` flag)

Failure condition: if the agent has fewer eyes remaining than empty portal frames, the episode terminates (`eyes_of_ender_remaining < portal_eyes_needed`). Eyes also have a 20% chance of breaking when thrown, so the agent must manage its eye supply carefully.

### Stage 5 Reward Shaping

The Stage 5 reward shaper (`create_stage5_reward_shaper()` in `reward_shaping.py`) provides dense signals across three phases: triangulation, navigation, and portal filling.

**1. Penalties (continuous pressure)**

| Signal | Value |
|--------|-------|
| Time penalty | `-0.00012` per tick |
| Death | `-0.8` terminal |
| Damage taken | `-0.015 * damage_amount` |

**2. Milestone rewards (one-time bonuses)**

| Milestone | Condition | Bonus |
|-----------|-----------|-------|
| First eye thrown | `eye_thrown` | +0.15 |
| Second eye thrown | `eyes_thrown >= 2` | +0.10 |
| Triangulation ready | 2+ throw points recorded | +0.20 |
| Stronghold located | `stronghold_located` | +0.25 |
| Stronghold entered | `in_stronghold` | +0.30 |
| Library found | `library_found` | +0.10 |
| Portal room found | `portal_room_found` | +0.35 |
| First frame filled | `portal_frames_filled > 0` | +0.10 |
| 6 frames filled | `portal_frames_filled >= 6` | +0.15 |
| Portal activated | `end_portal_activated` | +0.50 |
| Efficient triangulation | Used <= 4 eyes to locate | +0.20 |
| Fast discovery | Found in < 3000 ticks | +0.15 |

**3. Progressive rewards**

| Signal | Calculation |
|--------|-------------|
| Approach stronghold | `min((prev_dist - curr_dist) * 0.0005, 0.03)` per tick |
| Frame filling | `+0.03` per frame placed |
| Eye conservation | `+0.1` if >= 10 eyes remain after finding stronghold |

**4. Stage completion bonus**: `+2.5`

### Stage 5 Dense Rewards (from YAML)

These rewards are applied by the environment independently of the reward shaper:

| Signal | Value |
|--------|-------|
| Craft blaze powder | +0.2 |
| Craft eye of ender | +0.5 |
| Throw first eye | +1.0 |
| Note direction | +0.3 |
| Travel for triangulation | +0.01 per block |
| Throw second eye | +1.5 |
| Calculate intersection | +2.0 |
| Approach stronghold | +0.02 per block |
| Eye drops down | +1.0 |
| Dig to stronghold | +0.5 |
| Enter stronghold | +2.0 |
| Find library | +0.5 |
| Find prison | +0.3 |
| Find portal room | +5.0 |
| Place eye in frame | +0.5 per eye |
| Portal active | +3.0 |

### Success Criteria (from `stage_5_stronghold_finding.yaml`)

The episode succeeds when the End portal is active (`end_portal_active`).

The episode fails if:
- Deaths >= 3
- Ticks >= 60,000
- Remaining eyes < needed portal eyes (unrecoverable state)

### Curriculum Advancement

Stage 5 uses a threshold of `0.55` (55% success rate). The lower threshold compared to earlier stages reflects the heavy RNG involved in stronghold distance and eye-break probability. Minimum episodes before advancement: `expected_episodes = 1000`.

### Stage 5 Training Tips

- **Observation augmentation**: The `ProgressTracker.to_observation()` 32-element vector includes `eyes_crafted`, `eyes_used`, `stronghold_found`, `portal_room_found`, `eyes_placed`, and `portal_activated` signals, giving the policy direct access to progression state.
- **Network architecture**: Use a larger policy network (`[256, 256, 128]`) since Stage 5 requires planning and spatial reasoning.
- **Discount factor**: Use `gamma=0.999` to allow the agent to reason over long horizons (up to 60,000 ticks).
- **Entropy coefficient**: Keep at `0.01-0.02` to encourage exploration of different triangulation strategies.

---

## 4. Curriculum Training (All Stages)

The full curriculum consists of 6 stages with increasing difficulty:

| Stage | Name | Objectives | Expected Episodes |
|-------|------|------------|-------------------|
| 1 | Basic Survival | Kill mobs, craft tools | 500 |
| 2 | Resource Gathering | Mine iron, get armor | 1000 |
| 3 | Nether Navigation | Enter Nether, find fortress | 1500 |
| 4 | Enderman Hunting | Kill Endermen, get pearls | 2000 |
| 5 | Stronghold Finding | Find portal, activate | 2500 |
| 6 | End Fight | Kill the Ender Dragon | 3000 |

### Using the Curriculum Manager

```python
from minecraft_sim.curriculum import CurriculumManager, StageID

# Create curriculum manager
curriculum = CurriculumManager()

# Start at stage 1
stage = curriculum.start_training(StageID.BASIC_SURVIVAL)
print(f"Starting: {stage.name}")
print(f"Objectives: {stage.objectives}")

# Training loop
for episode in range(10000):
    # Run episode...
    success = run_episode(env, policy)
    reward = get_episode_reward()
    ticks = get_episode_length()

    # Record results
    mastered = curriculum.record_episode(success, reward, ticks)

    # Check for advancement
    if curriculum.should_advance():
        new_stage = curriculum.advance_stage()
        if new_stage:
            print(f"Advanced to: {new_stage.name}")
            env = create_env_for_stage(new_stage)

# Save progress
curriculum.save_progress("training_progress.json")
```

### Full Curriculum Training Script

Save as `train_curriculum.py`:

```python
#!/usr/bin/env python3
"""Full curriculum training from basic survival to dragon fight.

Run with: PYTHONPATH=python uv run python train_curriculum.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "python"))

import numpy as np
from minecraft_sim.curriculum import create_speedrun_curriculum, StageID
from minecraft_sim.vec_env import VecDragonFightEnv

# Configuration
NUM_ENVS = 4096
ROLLOUT_STEPS = 256
TOTAL_EPOCHS = 500

def main():
    # Initialize curriculum
    curriculum = create_speedrun_curriculum()
    curriculum.start_training(StageID.BASIC_SURVIVAL)

    # Create initial environment
    env = VecDragonFightEnv(num_envs=NUM_ENVS)

    # Initialize policy (your RL algorithm here)
    policy = create_policy(obs_dim=48, action_dim=17)

    total_steps = 0
    for epoch in range(TOTAL_EPOCHS):
        # Collect rollout
        obs = env.reset()
        episode_rewards = []

        for step in range(ROLLOUT_STEPS):
            actions = policy.act(obs)
            next_obs, rewards, dones, infos = env.step(actions)

            # Record completed episodes
            for i, done in enumerate(dones):
                if done:
                    reward = rewards[i]
                    success = reward > 500  # Win threshold
                    curriculum.record_episode(success, reward, step)

            obs = next_obs
            total_steps += NUM_ENVS

        # Update policy
        policy.update(...)

        # Check curriculum advancement
        if curriculum.should_advance():
            new_stage = curriculum.advance_stage()
            if new_stage:
                print(f"[Epoch {epoch}] Advanced to: {new_stage.name}")

        # Log progress
        summary = curriculum.get_training_summary()
        print(f"Epoch {epoch}: {summary['stages_mastered']}/{summary['total_stages']} stages mastered")

    # Save final model and progress
    policy.save("speedrun_agent.pt")
    curriculum.save_progress("curriculum_progress.json")

if __name__ == "__main__":
    main()
```

### Curriculum Progress Snapshots

The curriculum system shares real-time stage progress between three layers: `BaseStageEnv` (single-environment training), `SpeedrunVecEnv` (vectorized training), and the monitoring stack (dashboards and alerts). This section explains how progress data flows for Stages 1-4.

**ProgressTracker and SpeedrunProgress**

Each environment owns a `ProgressTracker` instance (from `minecraft_sim.progression`) that wraps a `SpeedrunProgress` dataclass. On every `step()`, the flat observation vector is decoded into a structured dict and fed to `ProgressTracker.update_from_observation()`, which increments per-stage counters:

| Stage | Key Metrics Tracked |
|-------|-------------------|
| 1 (Survival) | `wood_collected`, `stone_collected`, `zombies_killed`, `food_eaten` |
| 2 (Resources) | `iron_ore_mined`, `iron_ingots`, `diamonds`, `has_bucket`, `obsidian_collected` |
| 3 (Nether) | `portal_built`, `entered_nether`, `fortress_found`, `blazes_killed`, `blaze_rods` |
| 4 (Pearls) | `endermen_killed`, `ender_pearls`, `nether_wart_collected` |

**BaseStageEnv Integration**

In single-environment mode, `BaseStageEnv` handles progress as follows:

1. `__init__` instantiates `self._progress_tracker = ProgressTracker()`.
2. `reset()` calls `self._progress_tracker.reset()` and returns an initial `progress_snapshot` in the info dict.
3. `step()` updates internal `_stage_state`, builds a JSON-serializable snapshot via `_build_progress_snapshot()`, and attaches it to `infos["progress_snapshot"]`.
4. On episode termination, `infos["episode"]["progress"]` contains the final snapshot for the completed episode.

Subclasses (e.g. `BasicSurvivalEnv`, `ResourceGatheringEnv`) populate `_stage_state` with stage-specific fields that get serialized into the snapshot. The `get_progress_tracker()` method exposes the tracker for external consumers.

**SpeedrunVecEnv Integration**

In vectorized mode, `SpeedrunVecEnv` maintains one `ProgressTracker` per environment in `self._progress_trackers`:

1. After each batched `sim.step()`, the flat observation for each env is decoded via `_decode_obs_to_dict()` and passed to the corresponding tracker's `update_from_observation()`.
2. Each `infos[i]["progress_snapshot"]` receives the result of `tracker.to_snapshot()`, which includes raw progress fields, per-stage completion percentages, and the overall completion metric.
3. On episode termination (`dones[i] == True`), the tracker for that environment resets, and the final snapshot is included in `infos[i]["episode"]`.
4. `get_progress_snapshots()` returns a list of all current snapshots, suitable for batch monitoring.

Since `SpeedrunVecEnv` groups environments by curriculum stage, it can compute stage distributions and per-stage success rates from the tracked progress.

**Monitoring Stack**

The monitoring layer consumes snapshots to produce dashboard metrics:

1. A metrics module ingests snapshots from either environment type and computes aggregate stage distributions (how many envs at each stage) and per-stage success rates.
2. The dashboard renders per-stage cards showing resource velocities (e.g. iron ingots per episode for Stage 2, blaze rods per minute for Stage 3).
3. A watchdog system emits alerts when stage progress stalls, for example zero wood collection in Stage 1 or no new blaze rods in Stage 3 over a configurable tick window.

**Data Flow Diagram**

```
mc189_core.get_observations()
         |
         v
  [Flat obs vector]
         |
    decode_flat_observation() / _decode_obs_to_dict()
         |
         v
  [Structured dict]
         |
    ProgressTracker.update_from_observation()
         |
         v
  [SpeedrunProgress dataclass]
         |
    ProgressTracker.to_snapshot() / _build_progress_snapshot()
         |
         v
  [JSON-serializable snapshot]
         |
    +--- infos["progress_snapshot"]  --> Training loop / SB3 callbacks
    |
    +--- get_progress_snapshots()    --> Monitoring dashboard
    |
    +--- CurriculumManager           --> Stage advancement decisions
```

**Curriculum Implementation Reference**

The curriculum pipeline implementation spans these task groups:

- `CURR-COM-*`: Framework-level plumbing (tracker embedding, observation decoding, snapshot surfacing, monitoring integration).
- `CURR-S1-*`: Stage 1 reward shaping, progress tracking, and success criteria wiring.
- `CURR-S2-*`: Stage 2 resource tracking and obsidian-based advancement thresholds.
- `CURR-S3-*`: Stage 3 nether navigation, fortress discovery, and blaze rod milestones.
- `CURR-S4-*`: Stage 4 stronghold hunting, eye placement, and portal activation.

### Stage 3: Nether Navigation

Stage 3 teaches the agent to build and activate a Nether portal, navigate the hostile Nether dimension, locate a fortress, and kill blazes to collect blaze rods. The agent spawns in the overworld with iron gear, food, buckets, and a flint and steel, so it can focus on portal construction and Nether exploration rather than early-game resource gathering.

#### Portal Tracking

Portal-related progress is tracked through a sequence of observation flags and milestones. The observation space encodes portal state at indices `[224-231]`:

| Observation Index | Field | Description |
|-------------------|-------|-------------|
| 135 | `in_nether` | Whether the agent is currently in the Nether |
| 224 | `near_nether_portal` | Agent is within proximity of a portal frame |
| 225 | `portal_distance` | Normalized distance to nearest portal |
| 226 | `portal_alignment` | How aligned the agent is with the portal frame |
| 227 | `in_portal_cooldown` | Portal travel cooldown is active |

The reward shaper (`create_stage3_reward_shaper()` in `reward_shaping.py`) awards the following one-time milestone bonuses for portal construction:

| Milestone | Condition | Bonus |
|-----------|-----------|-------|
| `portal_frame_placed` | First obsidian placed in frame pattern | +0.2 |
| `portal_built` | Complete 4x5 obsidian frame detected | +0.35 |
| `portal_lit` | Flint and steel used, portal blocks appear | +0.15 |
| `entered_nether` | Dimension changes to Nether | +0.4 |

The stage config (`stage_3_nether_navigation.yaml`) defines additional dense rewards for the construction process:

| Signal | Value |
|--------|-------|
| `obsidian_mined` | +0.3 per block |
| `lava_found` | +0.2 (one-time) |
| `water_placed` | +0.1 |
| `obsidian_placed` | +0.2 per frame block |
| `portal_frame_complete` | +2.0 |
| `portal_lit` | +3.0 |
| `nether_entered` | +5.0 |

On top of the dense reward, `SpeedrunEnv._shape_reward()` awards a one-time milestone bonus of **+100.0** the first time `entered_nether` appears in the step info. This large signal dominates early training and ensures the policy prioritizes portal construction.

#### Fortress Discovery

Once in the Nether, the agent must locate a Nether fortress. The simulator guarantees a fortress spawns within 500 blocks (`metadata.max_fortress_distance: 500`) and provides a continuous distance signal (`distance_to_fortress`) in the observation and state dicts.

The reward shaper uses a distance-approach heuristic: when `in_nether` is true and `fortress_found` has not yet fired, the shaper computes the per-tick change in `distance_to_fortress`. Moving closer yields a shaped reward of `min((prev_dist - curr_dist) * 0.001, 0.05)` per tick, giving the agent a gradient to follow toward the fortress.

Fortress discovery triggers the following milestones in the reward shaper:

| Milestone | Condition | Bonus |
|-----------|-----------|-------|
| `fortress_visible` | Fortress within render distance | +0.2 |
| `fortress_found` | Agent has detected fortress structure | +0.4 |
| `in_fortress` | Agent is standing inside fortress bounds | +0.2 |
| `blaze_spawner_found` | Blaze spawner block detected nearby | +0.25 |

The YAML config also specifies dense rewards for fortress events:

| Signal | Value |
|--------|-------|
| `nether_fortress_found` | +5.0 |
| `fortress_entered` | +2.0 |
| `nether_distance_traveled` | +0.01 per block |
| `exploration_bonus` | +0.03 per new area |

The observation space includes a `fortress_direction` field (from `observation_space` in the YAML) and `distance_to_fortress` at goal index 138, both normalized to [0, 1], which the policy can use to navigate.

#### Reward Shaping Behaviour

Stage 3 reward shaping combines five signal types:

**1. Penalties (continuous pressure)**

| Penalty | Value | Notes |
|---------|-------|-------|
| Time penalty (shaper) | -0.00012 per tick | Slightly higher than Stage 1/2, reflects Nether danger |
| Time penalty (env layer) | -0.00015 per tick | Applied by `_compute_reward()` from YAML config |
| Death (shaper) | -1.2 per death | Higher than Stage 1 (-1.0); recovery in Nether is costly |
| Death (env layer) | -2.0 per death | From `penalty_per_death` in YAML |
| Damage taken | -0.02 * damage | Standard damage penalty |
| Fire/lava damage | -0.03 * damage | 50% higher multiplier for environmental hazards |
| Lava proximity | -0.01 per tick in lava | Continuous penalty while standing in lava |

**2. Milestone rewards (one-time bonuses, shaper layer)**

The full milestone chain for a successful episode is:

`portal_frame_placed` -> `portal_built` -> `portal_lit` -> `entered_nether` -> `fortress_visible` -> `fortress_found` -> `in_fortress` -> `blaze_spawner_found` -> `first_blaze_kill` -> `blaze_rod_x3` -> `blaze_rod_x5` -> `blaze_rod_x7`

Optional milestones: `nether_wart` (+0.1), `fire_resistance` (+0.15), `ghast_fireball_deflected` (+0.3), `blaze_rod_x10` (+0.15).

**3. Progressive rewards (per-event, uncapped or diminishing)**

| Signal | Value | Cap |
|--------|-------|-----|
| Blaze rod progressive | +0.03 per rod | 0.35 total |
| Blaze kill (per kill) | +0.15 per kill | uncapped |
| Distance to fortress (approach) | +0.001 * blocks closed | 0.05 per tick |

**4. Environment-level milestone bonuses (`SpeedrunEnv._shape_reward`)**

| Event | Bonus | Notes |
|-------|-------|-------|
| `entered_nether` | +100.0 | Fires once per episode |
| `first_blaze_kill` | +50.0 | Fires on first kill when `mobs_killed == 0` |

These large bonuses provide a clear learning signal that dominates early in training, then become irrelevant once the policy reliably enters the Nether and kills its first blaze.

**5. Stage completion**

The shaper awards `+2.5` when `stage_complete` fires. The YAML also defines a sparse reward of `+25.0` applied by the curriculum layer on successful episode termination.

#### Success and Termination Conditions

From the stage config:

| Condition | Value |
|-----------|-------|
| Success: dimension | Must be in Nether |
| Success: blaze rods | >= 7 collected |
| Max ticks | 72,000 (60 minutes game time) |
| Max deaths | 5 per episode |
| Curriculum threshold | 0.6 (60% success rate to advance) |
| Expected episodes | 2,000 |

#### Reward Budget for a Successful Episode

A complete Stage 3 episode (enter Nether, find fortress, kill 7 blazes, collect 7 rods) yields approximately:

| Component | Reward |
|-----------|--------|
| Nether entry (dense) | +5.0 |
| Nether entry (milestone) | +100.0 |
| First blaze kill (milestone) | +50.0 |
| 7 blaze kills (progressive) | +1.05 |
| 7 blaze rods (milestones: x1+x3+x5+x7) | +0.95 |
| 7 blaze rods (progressive) | +0.21 |
| Fortress milestones | +0.85 |
| Portal milestones | +1.1 |
| Stage completion (shaper) | +2.5 |
| Stage completion (sparse) | +25.0 |
| **Total positive** | **~187** |
| Time penalty (~30k ticks at -0.00027) | ~ -8.1 |
| **Net reward** | **~179** |

The net reward is strongly positive, ensuring the value function can clearly distinguish successful trajectories from failed ones.

#### Training Tips for Stage 3

- Use `gamma=0.999` or higher. The portal construction and fortress navigation sequences produce long delays between action and reward, requiring high discount factors to propagate credit.
- The `exploration_bonus: 0.03` and `nether_distance_traveled: 0.01` signals prevent the agent from getting stuck near the portal. If training stalls after Nether entry, check that the distance-to-fortress approach reward is firing.
- Deaths from lava are the primary failure mode. The elevated fire/lava damage penalty (-0.03x multiplier vs standard -0.02x) teaches lava avoidance, but early training benefits from a higher entropy coefficient (0.02-0.05) to encourage exploring alternative paths around lava lakes.
- The fortress spawn guarantee (`fortress_spawn_guarantee: true`) means the agent always has a reachable target. If training on a non-guaranteed config, increase `max_ticks` and `expected_episodes` proportionally.

---

## 5. Full Speedrun Training

For training a complete speedrun agent that can beat the game from scratch:

### Recommended Configuration

```python
# Training hyperparameters for speedrun
CONFIG = {
    # Environment
    "num_envs": 8192,           # Maximize parallelism
    "rollout_steps": 256,       # Longer rollouts for sparse rewards

    # PPO
    "learning_rate": 3e-4,
    "batch_size": 4096,
    "n_epochs": 4,
    "gamma": 0.999,             # High discount for long episodes
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    # Curriculum
    "curriculum_threshold": 0.5, # 50% success rate to advance
    "min_episodes_per_stage": 1000,

    # Training
    "total_timesteps": 1_000_000_000,  # 1B steps
    "checkpoint_freq": 10_000_000,
}
```

### Using free_the_end.py

The included `free_the_end.py` script provides a complete training pipeline:

```bash
# Set Vulkan path (macOS)
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json

# Run full training
PYTHONPATH=python uv run python examples/free_the_end.py
```

### Multi-GPU Training

For large-scale training across multiple GPUs:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    rank = setup_distributed()

    # Each GPU gets its own batch of environments
    envs_per_gpu = 2048
    env = VecDragonFightEnv(num_envs=envs_per_gpu)

    # Wrap policy in DDP
    policy = create_policy().cuda(rank)
    policy = DDP(policy, device_ids=[rank])

    # Training loop with gradient synchronization
    ...
```

Launch with:

```bash
torchrun --nproc_per_node=4 train_distributed.py
```

---

## 6. Monitoring with TensorBoard

### Setting Up Logging

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/dragon_fight_ppo")

# In training loop
writer.add_scalar("train/reward", avg_reward, global_step)
writer.add_scalar("train/episode_length", avg_length, global_step)
writer.add_scalar("train/win_rate", wins / episodes, global_step)
writer.add_scalar("train/fps", steps_per_second, global_step)

# Curriculum metrics
writer.add_scalar("curriculum/stage", current_stage, global_step)
writer.add_scalar("curriculum/success_rate", curriculum.progress[stage].success_rate, global_step)

# Policy metrics
writer.add_scalar("policy/entropy", entropy, global_step)
writer.add_scalar("policy/value_loss", value_loss, global_step)
writer.add_scalar("policy/policy_loss", policy_loss, global_step)

writer.close()
```

### Launching TensorBoard

```bash
# Start TensorBoard server
tensorboard --logdir=runs --port=6006

# Open in browser
open http://localhost:6006
```

### Key Metrics to Monitor

| Metric | Description | Target |
|--------|-------------|--------|
| `train/reward` | Episode reward | Increasing |
| `train/win_rate` | Dragon kills / episodes | >50% on stage 6 |
| `train/fps` | Training throughput | >200k steps/sec |
| `policy/entropy` | Action distribution entropy | 0.5-2.0 |
| `policy/value_loss` | Critic MSE loss | Decreasing |
| `curriculum/stage` | Current curriculum stage | 1 -> 6 |

### Weights & Biases Integration

```python
import wandb

wandb.init(project="minecraft-speedrun", config=CONFIG)

# Log metrics
wandb.log({
    "reward": avg_reward,
    "win_rate": win_rate,
    "stage": current_stage,
    "fps": fps,
})

# Log model checkpoints
wandb.save("checkpoint.pt")
```

---

## 7. Common Issues and Solutions

### Build Issues

**Problem: `mc189_core not found`**

```
ImportError: No module named 'mc189_core'
```

**Solution:**
```bash
# Ensure extension is built and copied
cd cpp/build && cmake --build . --target mc189_core
cp mc189_core.cpython-*-*.so ../../python/minecraft_sim/

# Add to Python path
export PYTHONPATH=$(pwd)/python
```

**Problem: `VK_ICD_FILENAMES` error on macOS**

```
vkCreateInstance failed: VK_ERROR_INCOMPATIBLE_DRIVER
```

**Solution:**
```bash
# Set MoltenVK path
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json

# Verify installation
vulkaninfo | head -20
```

**Problem: Shader compilation fails**

```
glslangValidator: error: could not open input file
```

**Solution:**
```bash
# Recompile shaders
make validate-shaders
make shaders
```

### Training Issues

**Problem: NaN rewards or observations**

**Solution:**
```python
# Add observation clipping
obs = np.clip(obs, 0.0, 1.0)

# Check for NaN and reset
if np.isnan(obs).any():
    obs = env.reset()
```

**Problem: Training doesn't converge**

**Solution:**
- Reduce learning rate: `lr=1e-4`
- Increase batch size: `batch_size=4096`
- Check reward scaling: rewards should be in range [-10, 10]
- Verify environment is deterministic: set seeds

**Problem: OOM (Out of Memory)**

**Solution:**
```python
# Reduce number of environments
env = VecDragonFightEnv(num_envs=1024)  # Instead of 8192

# Use gradient checkpointing
model = PPO(..., policy_kwargs=dict(
    net_arch=[256, 256],  # Smaller network
))
```

### Performance Issues

**Problem: Low throughput (<10k steps/sec)**

**Diagnosis:**
```python
import time

start = time.perf_counter()
for _ in range(1000):
    obs, rewards, dones, infos = env.step(actions)
elapsed = time.perf_counter() - start

print(f"Pure env throughput: {64 * 1000 / elapsed:.0f} steps/sec")
```

**Solutions:**
- Ensure GPU is being used (not CPU fallback)
- Increase `num_envs` to saturate GPU
- Use `np.int32` for actions (not Python ints)
- Batch observations before neural network forward pass

---

## 8. Hyperparameter Tuning Tips

### Learning Rate Schedule

```python
def lr_schedule(progress_remaining):
    """Linear decay from 3e-4 to 1e-5."""
    return 3e-4 * progress_remaining + 1e-5 * (1 - progress_remaining)

model = PPO("MlpPolicy", env, learning_rate=lr_schedule)
```

### Entropy Coefficient

| Training Phase | ent_coef | Reason |
|---------------|----------|--------|
| Early (exploration) | 0.05 | Encourage diverse actions |
| Mid (learning) | 0.01 | Balance exploration/exploitation |
| Late (refinement) | 0.001 | Focus on best actions |

### Batch Size vs Learning Rate

| Batch Size | Learning Rate | Notes |
|------------|---------------|-------|
| 256 | 3e-4 | Default SB3 |
| 1024 | 1e-3 | Scale with sqrt(batch_size) |
| 4096 | 2e-3 | Large batch training |
| 8192 | 3e-3 | Maximum recommended |

### Curriculum Thresholds by Stage

| Stage | Threshold | Rationale |
|-------|-----------|-----------|
| 1 (Survival) | 0.7 | Basic skills should be reliable |
| 2 (Resources) | 0.6 | Some RNG in resource spawns |
| 3 (Nether) | 0.5 | Navigation is challenging |
| 4 (Endermen) | 0.6 | Combat is learnable |
| 5 (Stronghold) | 0.4 | Heavy exploration RNG |
| 6 (Dragon) | 0.3 | Final boss, any wins count |

### Network Architecture

```python
# For stages 1-4 (simpler tasks)
policy_kwargs = dict(
    net_arch=[128, 128],
    activation_fn=nn.ReLU,
)

# For stages 5-6 (complex planning)
policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256, 128],  # Larger policy network
        vf=[256, 256, 128],
    ),
    activation_fn=nn.Tanh,
)
```

---

## 9. Expected Training Curves

### Stage 1: Basic Survival

```
Epochs 1-50:    Reward: -5 -> 0     (learning movement)
Epochs 50-150:  Reward: 0 -> 3      (killing mobs)
Epochs 150-300: Reward: 3 -> 8      (crafting tools)
Epochs 300-500: Reward: 8 -> 10     (consistent completion)

Success rate: 0% -> 70%+ by epoch 500
```

### Stage 6: Dragon Fight

```
Epochs 1-100:     Win rate: 0%      (dying immediately)
Epochs 100-500:   Win rate: 0-5%    (learning survival)
Epochs 500-1500:  Win rate: 5-20%   (damaging dragon)
Epochs 1500-2500: Win rate: 20-40%  (crystal destruction)
Epochs 2500-3000: Win rate: 40-50%+ (consistent kills)

Total training: ~500M-1B steps for reliable dragon kills
```

### Full Curriculum Timeline

| Milestone | Steps | Time (M4 Max) |
|-----------|-------|---------------|
| Stage 1 mastered | ~50M | ~20 min |
| Stage 2 mastered | ~150M | ~1 hour |
| Stage 3 mastered | ~300M | ~2 hours |
| Stage 4 mastered | ~500M | ~3 hours |
| Stage 5 mastered | ~750M | ~4 hours |
| Stage 6 50% win | ~1B | ~6 hours |

### Training Cost Estimates

| Hardware | Time to 1B steps | Electricity |
|----------|-----------------|-------------|
| Apple M4 Max | 6 hours | ~$0.50 |
| RTX 3090 | 4 hours | ~$2.00 |
| RTX 4090 | 2.5 hours | ~$1.50 |
| 4x RTX 4090 | 40 min | ~$1.00 |

---

## 10. From Trained Model to Evaluation

### Saving Checkpoints

```python
# Save during training
if global_step % checkpoint_freq == 0:
    model.save(f"checkpoints/model_step_{global_step}")
    curriculum.save_progress(f"checkpoints/progress_step_{global_step}.json")

# Save final model
model.save("final_model")
np.save("final_weights.npy", {
    "policy": policy.state_dict(),
    "value": value.state_dict(),
})
```

### Loading for Evaluation

```python
from stable_baselines3 import PPO

# Load trained model
model = PPO.load("final_model")

# Create single environment for evaluation
env = DragonFightEnv(render_mode="human")

# Run evaluation episodes
wins = 0
total = 100

for episode in range(total):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    if total_reward > 500:  # Win threshold
        wins += 1

    print(f"Episode {episode + 1}: Reward = {total_reward:.1f}")

print(f"\nWin rate: {wins}/{total} = {100*wins/total:.1f}%")
```

### Benchmarking Performance

```python
def benchmark_policy(model, env, num_episodes=100):
    """Comprehensive policy evaluation."""
    results = {
        "rewards": [],
        "lengths": [],
        "wins": 0,
        "deaths": 0,
        "dragon_damage": [],
        "crystals_destroyed": [],
    }

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        results["rewards"].append(episode_reward)
        results["lengths"].append(steps)
        if episode_reward > 500:
            results["wins"] += 1
        else:
            results["deaths"] += 1

    return {
        "mean_reward": np.mean(results["rewards"]),
        "std_reward": np.std(results["rewards"]),
        "mean_length": np.mean(results["lengths"]),
        "win_rate": results["wins"] / num_episodes,
        "total_episodes": num_episodes,
    }
```

### Exporting for Deployment

```python
import torch

# Export to ONNX for deployment
dummy_input = torch.randn(1, 48)
torch.onnx.export(
    policy_net,
    dummy_input,
    "dragon_policy.onnx",
    input_names=["observation"],
    output_names=["action_logits"],
    dynamic_axes={"observation": {0: "batch_size"}},
)

# Export to TorchScript for C++ inference
scripted = torch.jit.script(policy_net)
scripted.save("dragon_policy.pt")
```

### Recording Videos

```python
from gymnasium.wrappers import RecordVideo

# Wrap environment for recording
env = RecordVideo(
    DragonFightEnv(render_mode="rgb_array"),
    video_folder="videos",
    episode_trigger=lambda x: True,  # Record all episodes
)

# Run episodes
for _ in range(10):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()
```

---

## Appendix: Quick Reference

### Environment API

```python
from minecraft_sim import DragonFightEnv, VecDragonFightEnv, SB3VecDragonFightEnv

# Single environment
env = DragonFightEnv()
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)

# Vectorized (custom training)
env = VecDragonFightEnv(num_envs=64)
obs = env.reset()  # (64, 48)
obs, rewards, dones, infos = env.step(actions)

# Vectorized (SB3 compatible)
env = SB3VecDragonFightEnv(num_envs=64)
# Use with PPO, A2C, etc.
```

### Action Space

| Index | Action | Description |
|-------|--------|-------------|
| 0 | NOOP | No action |
| 1 | FORWARD | Move forward |
| 2 | BACK | Move backward |
| 3 | LEFT | Strafe left |
| 4 | RIGHT | Strafe right |
| 5 | FORWARD_LEFT | Diagonal forward-left |
| 6 | FORWARD_RIGHT | Diagonal forward-right |
| 7 | JUMP | Jump |
| 8 | JUMP_FORWARD | Jump while moving forward |
| 9 | ATTACK | Melee attack |
| 10 | ATTACK_FORWARD | Attack while moving forward |
| 11 | SPRINT | Toggle sprint |
| 12 | LOOK_LEFT | Turn left ~5 degrees |
| 13 | LOOK_RIGHT | Turn right ~5 degrees |
| 14 | SWAP_WEAPON | Switch hotbar slot |
| 15 | LOOK_UP | Pitch up ~7.5 degrees |
| 16 | LOOK_DOWN | Pitch down ~7.5 degrees |

### Observation Layout

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 0-2 | player_pos | [0, 1] | XYZ position (scaled) |
| 3-5 | player_vel | [0, 1] | XYZ velocity |
| 6 | player_yaw | [0, 1] | Horizontal look angle |
| 7 | player_pitch | [0, 1] | Vertical look angle |
| 8 | player_health | [0, 1] | Health (0-20 HP) |
| 16 | dragon_health | [0, 1] | Dragon HP (0-200) |
| 17-19 | dragon_pos | [0, 1] | Dragon XYZ position |
| 24 | dragon_phase | [0, 1] | AI phase (0-6) |
| 25 | dragon_dist | [0, 1] | Distance to dragon |
| 28 | can_hit | 0/1 | Attack will connect |
| 32 | crystal_count | [0, 1] | Crystals remaining |

### Useful Commands

```bash
# Build C++ extension
cd cpp/build
cmake --build . --target mc189_core
cp mc189_core.cpython-*-darwin.so ../../python/minecraft_sim/

# Run tests
uv run pytest tests/test_discrete_actions.py -v

# Run training
PYTHONPATH=python uv run python examples/train_ppo.py

# Monitor
tensorboard --logdir=runs

# Benchmark
PYTHONPATH=python uv run python examples/quick_run.py
```
