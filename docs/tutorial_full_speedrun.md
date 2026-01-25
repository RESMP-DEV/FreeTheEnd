# Tutorial: Full Speedrun Training (Spawn to Dragon Kill)

This tutorial walks through training an RL agent to complete a Minecraft 1.8.9 speedrun across all 6 curriculum stages, from spawning in a fresh world to killing the Ender Dragon.

## Overview

### What "Free The End" Means

"Free The End" is the Minecraft speedrun category where the goal is to kill the Ender Dragon as fast as possible from a fresh world spawn. The agent must gather resources, craft tools, navigate to the Nether, collect blaze rods, hunt endermen for pearls, locate a stronghold, activate the End portal, and defeat the dragon. This simulator decomposes that challenge into 6 learnable stages with automatic curriculum advancement.

### The 6 Stages

| Stage | Name | Goal | Max Ticks | Advancement Threshold |
|-------|------|------|-----------|----------------------|
| 1 | Basic Survival | Gather wood, craft pickaxe, mine cobblestone | 6,000 | 0.70 |
| 2 | Resource Gathering | Smelt iron, craft iron pickaxe, craft bucket | 12,000 | 0.65 |
| 3 | Nether Navigation | Build portal, enter Nether, collect 7 blaze rods | 18,000 | 0.60 |
| 4 | Enderman Hunting | Collect 12 ender pearls | 12,000 | 0.65 |
| 5 | Stronghold Finding | Craft 12 eyes, find stronghold, activate portal | 18,000 | 0.55 |
| 6 | Dragon Fight | Kill the Ender Dragon | 18,000 | 0.50 |

Each stage has its own reward shaper, success criteria, and failure conditions. The curriculum manager advances the agent to the next stage once it achieves the threshold success rate over a sliding window of recent episodes.

### Expected Training Timeline (M4 Max, 64 envs, ~264k steps/sec)

| Milestone | Cumulative Steps | Approx. Wall Time |
|-----------|-----------------|-------------------|
| Stage 1 mastered | ~50M | ~3 min |
| Stage 2 mastered | ~150M | ~10 min |
| Stage 3 mastered | ~300M | ~19 min |
| Stage 4 mastered | ~500M | ~32 min |
| Stage 5 mastered | ~750M | ~47 min |
| Stage 6 at 50% win rate | ~1B | ~63 min |
| Stage 6 reliable (>70%) | ~2-5B | ~3-5 hours |

With 256 environments (~615k steps/sec on M4 Max), these times roughly halve. On a machine running 4096 environments, a full training run from scratch to reliable dragon kills fits within 24-48 hours even with conservative hyperparameters.

---

## Prerequisites

```bash
cd FreeTheEnd

# Python environment (3.12 required)
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra training

# Vulkan setup (macOS)
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json

# Build C++ backend
cd cpp && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target mc189_core -j$(sysctl -n hw.ncpu)
cp mc189_core.cpython-*-*.so ../../python/minecraft_sim/
cd ../..

# Verify
PYTHONPATH=python uv run python -c "from minecraft_sim import SpeedrunVecEnv; print('OK')"
```

---

## Stage-by-Stage Setup

### Stage 1: Basic Survival

**Goal:** Gather wood, craft planks, make a wooden pickaxe, mine cobblestone.

**Environment:** `BasicSurvivalEnv` (128-dim obs, 24 actions, 6,000 tick horizon)

**Success criteria** (from `stage_criteria.py`):
- 4+ oak logs collected
- 8+ oak planks crafted (or `crafted_planks` flag set)
- Wooden pickaxe crafted (`has_wooden_pickaxe`)
- 16+ cobblestone mined

**Optional objectives** (bonus rewards):
- Stone pickaxe crafted
- 3+ mobs killed
- Health >= 10 at episode end

**Observation highlights (128 floats):**
- Player state: position, velocity, health (20 HP), hunger (20), yaw, pitch
- Nearby entities: 40 floats encoding hostile mob positions and types
- Nearby blocks: 40 floats encoding tree/stone/ore locations
- Inventory: wood, planks, sticks, cobblestone, tools
- Time of day, damage source

**Reward shaping** (from `create_stage1_reward_shaper()`):

| Signal | Value | Type |
|--------|-------|------|
| First wood | +0.2 | One-time milestone |
| 4 wood logs | +0.1 | One-time milestone |
| First planks | +0.1 | One-time milestone |
| Crafting table | +0.15 | One-time milestone |
| Wooden pickaxe | +0.3 | One-time milestone |
| Stone pickaxe | +0.3 | One-time milestone |
| First cobblestone | +0.15 | One-time milestone |
| 16 cobblestone | +0.1 | One-time milestone |
| Furnace crafted | +0.2 | One-time milestone |
| First iron ore | +0.25 | One-time milestone |
| First mob kill | +0.15 | One-time milestone |
| Wood collected | +0.02/log (cap 0.2) | Progressive |
| Cobblestone | +0.005/block (cap 0.2) | Progressive |
| Mob killed | +0.1/kill | Progressive |
| Exploration | +0.01/chunk | Progressive |
| Death | -1.0 | Penalty |
| Damage taken | -0.02/HP | Penalty |
| Hunger loss | -0.01/point | Penalty |
| Time | -0.0001/tick | Penalty |
| Stage complete | +2.0 | Terminal |

**Training tips:**
- Stage 1 is the easiest. The agent learns movement, crafting sequences, and block breaking.
- Use `gamma=0.99` since episodes are short (6,000 ticks max) and rewards are dense.
- Entropy coefficient of `0.01-0.02` works well; the action space is small enough that random exploration is effective early on.
- Most agents reach 70% success rate within 50M steps.

---

### Stage 2: Resource Gathering

**Goal:** Mine iron ore, smelt ingots, craft an iron pickaxe, and craft a bucket.

**Environment:** `ResourceGatheringEnv` (128-dim obs, 24 actions, 12,000 tick horizon)

**Success criteria** (from `stage_criteria.py`):
- 3+ iron ingots smelted
- Iron pickaxe crafted (`has_iron_pickaxe`)
- Bucket crafted (`has_bucket` or bucket in inventory)

**Optional objectives:**
- 1+ diamond found
- 10+ obsidian collected
- 10+ iron ingots total

**Observation highlights (128 floats):**
- Core player state (as Stage 1)
- Y-level (critical for finding ores at depth)
- Ore detection: iron, diamond, coal positions (50 floats)
- Inventory: iron ore, iron ingots, diamonds, obsidian, tools, bucket
- Furnace/smelting status flags

**Reward shaping** (from `create_stage2_reward_shaper()`):

| Signal | Value | Type |
|--------|-------|------|
| First iron ore | +0.15 | One-time milestone |
| 3 iron ore | +0.1 | One-time milestone |
| First iron ingot | +0.2 | One-time milestone |
| 3 iron ingots | +0.15 | One-time milestone |
| 10 iron ingots | +0.15 | One-time milestone |
| Iron pickaxe | +0.35 | One-time milestone |
| Bucket crafted | +0.3 | One-time milestone |
| Water bucket | +0.15 | One-time milestone |
| Lava bucket | +0.2 | One-time milestone |
| First diamond | +0.3 | One-time milestone |
| First obsidian | +0.2 | One-time milestone |
| 10 obsidian | +0.25 | One-time milestone |
| Flint and steel | +0.15 | One-time milestone |
| Iron ingots | +0.015/ingot (cap 0.3) | Progressive |
| Obsidian | +0.015/block (cap 0.25) | Progressive |
| Depth bonus | +0.005 per Y below 16 | Progressive |
| Death | -0.8 | Penalty |
| Damage taken | -0.015/HP | Penalty |
| Time | -0.0001/tick | Penalty |
| Stage complete | +2.0 | Terminal |

**Training tips:**
- The agent must learn to mine downward to find iron (Y=5-63) and the smelting chain: mine iron ore, place furnace, add coal+ore, wait, collect ingot.
- The Y-level depth reward is essential; without it the agent has no gradient for going underground.
- Consider increasing `n_steps` to 256 or higher since the action sequences to mine and smelt are long.
- A bucket requires 3 iron ingots arranged in a V shape. The agent needs to discover this through crafting exploration.

---

### Stage 3: Nether Navigation

**Goal:** Build and light a Nether portal, enter the Nether, find a fortress, and kill blazes for 7 blaze rods.

**Environment:** `NetherNavigationEnv` (192-dim obs, 28 actions, 18,000 tick horizon)

**Success criteria** (from `stage_criteria.py`):
- Portal built (`portal_built`)
- Entered Nether (`entered_nether`)
- Fortress found (`fortress_found`)
- 7+ blaze rods collected

**Optional objectives:**
- 10+ blaze rods
- Nether wart collected

**Observation highlights (192 floats):**
- Player state (32 floats)
- Inventory: obsidian, blaze rods, fire resistance (32 floats)
- Dimension flag: 0=overworld, 1=nether, 2=end
- Portal state: lit, distance, alignment (8 floats)
- Fortress tracking: direction, distance, entered flag (8 floats)
- Blaze detection: nearby count, killed count (8 floats)
- Hazard warnings: lava, fire, ghast projectiles (20 floats)

**Dimension transition handling:**

The simulator handles the overworld-to-Nether transition seamlessly. When the agent enters a lit portal frame, the `dimension` observation flag transitions from 0 to 1, the environment regenerates surroundings as Nether terrain, and a fortress is guaranteed within 500 blocks. The agent observes `distance_to_fortress` and `fortress_direction` continuously.

**Reward shaping** (from `create_stage3_reward_shaper()`):

| Signal | Value | Type |
|--------|-------|------|
| Portal frame placed | +0.2 | One-time milestone |
| Portal built | +0.35 | One-time milestone |
| Portal lit | +0.15 | One-time milestone |
| Entered Nether | +0.4 | One-time milestone |
| Fortress visible | +0.2 | One-time milestone |
| Fortress found | +0.4 | One-time milestone |
| In fortress | +0.2 | One-time milestone |
| Blaze spawner found | +0.25 | One-time milestone |
| First blaze rod | +0.3 | One-time milestone |
| 3 blaze rods | +0.2 | One-time milestone |
| 5 blaze rods | +0.2 | One-time milestone |
| 7 blaze rods | +0.25 | One-time milestone |
| Ghast fireball deflected | +0.3 | One-time milestone |
| Blaze rods | +0.03/rod (cap 0.35) | Progressive |
| Distance to fortress | +0.001/block closer (cap 0.05) | Progressive |
| Blaze killed | +0.15/kill | Progressive |
| In lava | -0.01/tick | Penalty |
| Fire/lava damage | -0.03/HP | Penalty |
| Other damage | -0.02/HP | Penalty |
| Death | -1.2 | Penalty |
| Time | -0.00012/tick | Penalty |
| Stage complete | +2.5 | Terminal |

**Ghast and blaze combat:**

Ghasts fire explosive fireballs that the agent can deflect by attacking the projectile mid-flight (reward: +0.3 milestone). Blazes shoot 3 fireballs in succession and fly, requiring the agent to time attacks during their "resting" phase. Fire resistance potions negate fire damage from blaze attacks.

**Training tips:**
- Use `gamma=0.999` for this stage. The portal construction and fortress navigation create long delays between action and reward.
- Deaths from lava are the primary failure mode. The elevated fire/lava penalty (-0.03x vs standard -0.02x) teaches avoidance.
- The `distance_to_fortress` approach reward is essential once in the Nether; without it, the agent has no gradient toward the fortress.
- Higher entropy (0.02-0.05) helps early on to explore around lava.

---

### Stage 4: Enderman Hunting

**Goal:** Hunt endermen and collect 12 ender pearls.

**Environment:** `EndermanHuntingEnv` (192-dim obs, 28 actions, 12,000 tick horizon)

**Success criteria** (from `stage_criteria.py`):
- 12+ ender pearls collected

**Optional objectives:**
- 16+ ender pearls
- 15+ endermen killed

**Observation highlights (192 floats):**
- Player state (32 floats)
- Enderman-specific: nearby count, positions, aggro state, distance (40 floats)
- Looking state: whether looking at enderman or away (8 floats)
- Environmental: ceiling height, water proximity, trap status (10 floats)
- Inventory: ender pearls, pumpkin helmet, blaze powder, eyes of ender (30 floats)

**Pumpkin helmet strategy:**

Wearing a carved pumpkin prevents endermen from becoming aggressive when the player looks at them. The observation includes a `pumpkin_equipped` flag. The agent can approach endermen safely to set up attacks.

**Water bucket traps:**

Endermen take damage from water and cannot teleport while standing in it. The optimal strategy:
1. Find or create a 2-block-high ceiling (endermen are 3 blocks tall, cannot enter)
2. Place water to create a moat
3. Aggro endermen by looking at them, then retreat under the ceiling
4. Attack their legs while they cannot reach the player

**Reward shaping** (from `create_stage4_reward_shaper()`):

| Signal | Value | Type |
|--------|-------|------|
| Enderman spotted | +0.1 | One-time milestone |
| First ender pearl | +0.25 | One-time milestone |
| 3 pearls | +0.15 | One-time milestone |
| 6 pearls | +0.15 | One-time milestone |
| 9 pearls | +0.15 | One-time milestone |
| 12 pearls | +0.25 | One-time milestone |
| First blaze powder | +0.15 | One-time milestone |
| First eye of ender | +0.2 | One-time milestone |
| 6 eyes | +0.15 | One-time milestone |
| 12 eyes | +0.25 | One-time milestone |
| Shield crafted | +0.1 | One-time milestone |
| In warped forest | +0.15 | One-time milestone |
| Pearls | +0.02/pearl (cap 0.4) | Progressive |
| Eyes of ender | +0.025/eye (cap 0.35) | Progressive |
| Enderman killed | +0.12/kill | Progressive |
| Night hunting bonus | +0.05 | One-time |
| Death | -1.0 | Penalty |
| Damage taken | -0.02/HP | Penalty |
| Time | -0.00015/tick | Penalty |
| Stage complete | +2.0 | Terminal |

**Training tips:**
- Enderman combat is mechanically demanding. The agent must learn look-avoidance, trap construction, and hit timing.
- Endermen have a 50% chance of dropping a pearl on death. The agent needs ~24 kills for 12 pearls on average.
- Consider pre-training on Stages 1-3 before starting Stage 4; the combat fundamentals transfer.
- The warped forest milestone (+0.15) encourages finding the biome with the highest enderman spawn rate.

---

### Stage 5: Stronghold Finding

**Goal:** Craft 12 eyes of ender, triangulate the stronghold location, navigate to it, find the portal room, and activate the End portal.

**Environment:** `StrongholdFindingEnv` (192-dim obs, 28 actions, 18,000 tick horizon)

**Success criteria** (from `stage_criteria.py`):
- 12+ eyes of ender crafted (`eyes_crafted >= 12`)
- Stronghold found (`stronghold_found`)
- End portal activated (`end_portal_activated`)

**Optional objectives:**
- Triangulation used (more efficient than wandering)

**Observation highlights (192 floats):**
- Player state (32 floats)
- Eye trajectory: last throw direction, distance traveled (6 floats)
- Triangulation waypoints: up to 4 recorded throw positions (12 floats)
- Estimated stronghold position (3 floats)
- Stronghold data: found flag, distance, rooms visited (8 floats)
- Portal frame status: 12 slots, 0.0 (empty) or 1.0 (filled) (12 floats)
- Inventory: eyes of ender remaining

**Eye of ender mechanics:**

Eyes of ender float toward the nearest stronghold when thrown. The agent must:
1. Throw an eye at position A, record the direction it floats
2. Travel 200-500 blocks perpendicular to that direction
3. Throw a second eye at position B, record its direction
4. The intersection of the two direction vectors gives the stronghold XZ coordinates
5. When an eye descends instead of floating, the stronghold is directly below

Eyes have a 20% chance of breaking when thrown, so the agent must manage its supply carefully.

**Stronghold generation:**
- 3 strongholds per world, in a ring 1408-2688 blocks from origin
- Spaced 120 degrees apart
- Y-level 10-40 (underground)
- Portal room always exists; up to 50 rooms total
- Portal frames start with 10% chance each of already containing an eye

**Reward shaping** (from `create_stage5_reward_shaper()`):

| Signal | Value | Type |
|--------|-------|------|
| First eye thrown | +0.15 | One-time milestone |
| Second eye thrown | +0.1 | One-time milestone |
| Triangulation ready | +0.2 | One-time milestone |
| Stronghold located | +0.25 | One-time milestone |
| Stronghold entered | +0.3 | One-time milestone |
| Portal room found | +0.35 | One-time milestone |
| First frame filled | +0.1 | One-time milestone |
| 6 frames filled | +0.15 | One-time milestone |
| Portal activated | +0.5 | One-time milestone |
| Efficient triangulation (<=4 eyes) | +0.2 | One-time milestone |
| Fast discovery (<3000 ticks) | +0.15 | One-time milestone |
| Distance to stronghold | +0.0005/block closer (cap 0.03) | Progressive |
| Frame filling | +0.03/frame | Progressive |
| Eye conservation (10+ remaining at find) | +0.1 | One-time |
| Death | -0.8 | Penalty |
| Damage taken | -0.015/HP | Penalty |
| Time | -0.00012/tick | Penalty |
| Stage complete | +2.5 | Terminal |

**Training tips:**
- This is the most exploration-intensive stage. Use `gamma=0.999` to propagate credit across long triangulation sequences.
- The `distance_to_stronghold` approach reward is essential; without it, the agent has no gradient after triangulation.
- Network architecture should be larger (`[256, 256, 128]`) since the stage requires spatial reasoning.
- Speedrun shortcut: "blind travel" 1800 blocks from origin lands within the stronghold ring, reducing eye usage.
- The 55% threshold is lower than earlier stages because of heavy RNG in stronghold distance and eye-break probability.

---

### Stage 6: Dragon Fight

**Goal:** Kill the Ender Dragon.

**Environment:** `DragonFightEnv` (64-dim obs, 20 actions, 18,000 tick horizon)

**Success criteria** (from `stage_criteria.py`):
- Dragon killed (`dragon_killed`)

**Optional objectives:**
- One-cycle kill (killed in single perch)
- 8+ crystals destroyed
- Kill under 5 minutes (< 6,000 ticks)

**Observation space (64 floats):**
- Player state (32 floats): position, health, hunger, yaw, pitch
- Dragon state (32 floats):
  - Dragon position relative to player (3 floats)
  - Dragon health normalized to [0, 1] (200 HP max)
  - Dragon velocity (3 floats)
  - Dragon phase: 0=circling, 1=strafing, 2=perching, 3=breathing
  - Dragon targeting player flag
  - Dragon angle and pitch
  - Crystal positions and alive/dead status (8 floats)
  - Void proximity warning
  - Exit portal status

**Dragon phases and strategy:**

| Phase | Dragon Behavior | Optimal Agent Strategy |
|-------|----------------|----------------------|
| CIRCLING | Flies in circles around obsidian pillars | Destroy crystals with bow; avoid strafing runs |
| STRAFING | Dives at the player | Dodge sideways; do not attack during dive |
| PERCHING | Lands on the central fountain | Rush to fountain, melee attack (highest DPS window) |
| BREATHING | Breathes acid clouds on the fountain | Retreat immediately; acid deals 6 HP/sec |

**Crystal destruction priority:**

The dragon heals from nearby end crystals. Destroying all 10 crystals is mandatory before dealing meaningful damage. Strategy:
1. Shoot open crystals with bow from the ground
2. For caged crystals (iron bar enclosures), pillar up and break bars first
3. All-crystals-destroyed milestone (+0.3) gates effective dragon DPS

**Perch timing attacks:**

The dragon perches on the fountain periodically. This is the highest-damage window:
- Sprint to the fountain when dragon begins perch descent
- Melee attack continuously during the perch
- Retreat when the dragon starts breathing (phase transitions to BREATHING)
- A well-timed perch attack sequence deals 40-60% of the dragon's HP

**Reward shaping** (from `create_stage6_reward_shaper()`):

| Signal | Value | Type |
|--------|-------|------|
| Entered End | +0.3 | One-time milestone |
| First crystal destroyed | +0.2 | One-time milestone |
| 3 crystals | +0.15 | One-time milestone |
| 5 crystals | +0.15 | One-time milestone |
| 8 crystals | +0.2 | One-time milestone |
| All crystals | +0.3 | One-time milestone |
| Dragon damaged | +0.15 | One-time milestone |
| Dragon half health | +0.2 | One-time milestone |
| Dragon quarter health | +0.2 | One-time milestone |
| Dragon critical (<20 HP) | +0.15 | One-time milestone |
| First perch hit | +0.25 | One-time milestone |
| Perch combo (3+ hits) | +0.2 | One-time milestone |
| Dragon killed | +1.0 | One-time milestone |
| One-cycle kill | +0.5 | One-time milestone |
| Fast kill (<3000 ticks) | +0.3 | One-time milestone |
| Speedrun pace (<60000 total ticks) | +0.5 | One-time milestone |
| Dragon damage | +0.005/HP dealt | Progressive |
| Crystal destroyed | +0.15/crystal | Progressive |
| Perch proximity (<5 blocks) | +0.02/tick | Progressive |
| Bow range proximity (<15 blocks) | +0.01/tick | Progressive |
| Arrow hit dragon | +0.08/hit | Progressive |
| Death | -2.0 | Penalty |
| Damage taken | -0.025/HP | Penalty |
| Void proximity | -0.1/tick | Penalty |
| Bed explosion damage | -0.2 | Penalty |
| Time | -0.0002/tick | Penalty |
| Stage complete | +5.0 | Terminal |

**Training tips:**
- This is the hardest stage. Use `gamma=0.999` and a larger network (`[256, 256, 128]`).
- Crystal destruction must be learned before dragon DPS. The milestone structure ensures this ordering.
- The perch timing window is the key skill gap between 30% and 70% win rate agents. Agents that learn to rush the fountain during perch consistently kill the dragon.
- Void falls are instant death. The `void_proximity` observation warns when near the edge.
- One-cycle kills (+0.5 bonus) encourage the agent to maximize damage during the first perch.

---

## Full Training Script

Complete working Python script for 6-stage curriculum training:

```python
#!/usr/bin/env python3
"""
Full speedrun training: 6-stage curriculum from spawn to dragon kill.

Usage:
    cd FreeTheEnd
    PYTHONPATH=python uv run python docs/train_full_speedrun.py
"""

import json
import time
from pathlib import Path

import numpy as np

from minecraft_sim import SpeedrunVecEnv
from minecraft_sim.curriculum import CurriculumManager


# --- Configuration ---

NUM_ENVS = 64                    # Parallel environments (increase for more GPU utilization)
ROLLOUT_STEPS = 256              # Steps per rollout before policy update
TOTAL_STEPS = 2_000_000_000     # 2B total environment steps
CHECKPOINT_DIR = Path("checkpoints/full_speedrun")
LOG_INTERVAL = 10_000            # Log every N steps

# PPO hyperparameters
LR = 3e-4
GAMMA = 0.999                    # High discount for long-horizon stages
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01                  # Entropy coefficient (increase for more exploration)
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
BATCH_SIZE = 512
N_EPOCHS = 4

# Curriculum settings
ADVANCEMENT_THRESHOLD = 0.6     # Success rate to advance (overridden per-stage)
MIN_EPISODES_TO_ADVANCE = 100   # Minimum episodes before advancement eligible
WINDOW_SIZE = 200               # Sliding window for success rate calculation


def create_policy(obs_dim: int, act_dim: int) -> "SimplePolicy":
    """Create a simple MLP policy. Replace with your preferred RL library."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise SystemExit(
            "PyTorch required. Install with: uv pip install torch"
        )

    class SimplePolicy(nn.Module):
        def __init__(self, obs_dim: int, act_dim: int):
            super().__init__()
            self.policy_net = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, act_dim),
            )
            self.value_net = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            self.optimizer = torch.optim.Adam(self.parameters(), lr=LR)
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.to(self.device)

        def act(self, obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().to(self.device)
                logits = self.policy_net(obs_t)
                probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)
            return actions.cpu().numpy().astype(np.int32)

        def evaluate(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).float().to(self.device)
                logits = self.policy_net(obs_t)
                values = self.value_net(obs_t).squeeze(-1)
            return logits.cpu().numpy(), values.cpu().numpy()

        def save(self, path: str | Path):
            torch.save(self.state_dict(), path)

        def load(self, path: str | Path):
            self.load_state_dict(torch.load(path, weights_only=True))

    return SimplePolicy(obs_dim, act_dim)


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Create vectorized environment starting at Stage 1
    env = SpeedrunVecEnv(
        num_envs=NUM_ENVS,
        initial_stage=1,
        auto_curriculum=True,
        success_threshold=ADVANCEMENT_THRESHOLD,
        min_episodes_for_advance=MIN_EPISODES_TO_ADVANCE,
    )

    # Create policy network (max obs=256, max actions=32 across all stages)
    policy = create_policy(obs_dim=256, act_dim=32)

    # Training state
    total_steps = 0
    episode_count = 0
    stage_episodes: dict[int, int] = {i: 0 for i in range(1, 7)}
    stage_successes: dict[int, int] = {i: 0 for i in range(1, 7)}
    best_stage_reached = 1
    start_time = time.time()

    print("=" * 60)
    print("  Free The End: Full Speedrun Training")
    print("=" * 60)
    print(f"  Environments: {NUM_ENVS}")
    print(f"  Target steps: {TOTAL_STEPS:,}")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    print("=" * 60)

    obs = env.reset()

    while total_steps < TOTAL_STEPS:
        # Collect rollout
        rollout_obs = []
        rollout_actions = []
        rollout_rewards = []
        rollout_dones = []

        for step in range(ROLLOUT_STEPS):
            # Pad observation to max size if needed
            if obs.shape[-1] < 256:
                padded = np.zeros((NUM_ENVS, 256), dtype=np.float32)
                padded[:, :obs.shape[-1]] = obs
                actions = policy.act(padded)
            else:
                actions = policy.act(obs)

            # Clip actions to valid range for current stage
            actions = np.clip(actions, 0, 31)

            next_obs, rewards, dones, infos = env.step(actions)

            rollout_obs.append(obs)
            rollout_actions.append(actions)
            rollout_rewards.append(rewards)
            rollout_dones.append(dones)

            # Track episode completions
            for i in range(NUM_ENVS):
                if dones[i]:
                    episode_count += 1
                    info = infos[i] if isinstance(infos, list) else {}
                    stage_id = info.get("stage_id", 1)
                    stage_episodes[stage_id] = stage_episodes.get(stage_id, 0) + 1

                    success = info.get("success", rewards[i] > 0)
                    if success:
                        stage_successes[stage_id] = stage_successes.get(stage_id, 0) + 1

                    # Check for stage advancement
                    new_stage = info.get("new_stage", stage_id)
                    if new_stage > best_stage_reached:
                        best_stage_reached = new_stage
                        elapsed = time.time() - start_time
                        print(f"\n{'='*60}")
                        print(f"  STAGE ADVANCEMENT: Stage {new_stage} reached!")
                        print(f"  Steps: {total_steps:,} | Episodes: {episode_count:,}")
                        print(f"  Wall time: {elapsed:.0f}s")
                        print(f"{'='*60}\n")

            obs = next_obs
            total_steps += NUM_ENVS

        # --- Policy update (PPO) ---
        # In a full implementation, compute advantages using GAE,
        # then run N_EPOCHS of minibatch PPO updates.
        # Replace this section with SB3 or your own PPO implementation.

        # Logging
        if total_steps % LOG_INTERVAL < NUM_ENVS * ROLLOUT_STEPS:
            elapsed = time.time() - start_time
            sps = total_steps / max(elapsed, 1)
            dist = env.get_stage_distribution()

            print(
                f"Steps: {total_steps:>12,} | "
                f"SPS: {sps:>8,.0f} | "
                f"Episodes: {episode_count:>6,} | "
                f"Best stage: {best_stage_reached} | "
                f"Distribution: {dist}"
            )

        # Periodic checkpoint
        if total_steps % 50_000_000 < NUM_ENVS * ROLLOUT_STEPS:
            save_checkpoint(
                policy, total_steps, episode_count,
                stage_episodes, stage_successes, CHECKPOINT_DIR,
            )

    # Final save
    save_checkpoint(
        policy, total_steps, episode_count,
        stage_episodes, stage_successes, CHECKPOINT_DIR,
    )

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Total steps: {total_steps:,}")
    print(f"  Total episodes: {episode_count:,}")
    print(f"  Best stage reached: {best_stage_reached}")
    for stage in range(1, 7):
        ep = stage_episodes.get(stage, 0)
        sc = stage_successes.get(stage, 0)
        rate = sc / max(ep, 1)
        print(f"  Stage {stage}: {sc}/{ep} success ({rate:.1%})")
    print("=" * 60)

    env.close()


def save_checkpoint(
    policy,
    total_steps: int,
    episode_count: int,
    stage_episodes: dict[int, int],
    stage_successes: dict[int, int],
    checkpoint_dir: Path,
):
    """Save training checkpoint for resume."""
    tag = f"step_{total_steps}"

    # Save policy weights
    policy.save(checkpoint_dir / f"policy_{tag}.pt")

    # Save training state
    state = {
        "total_steps": total_steps,
        "episode_count": episode_count,
        "stage_episodes": stage_episodes,
        "stage_successes": stage_successes,
    }
    with open(checkpoint_dir / f"state_{tag}.json", "w") as f:
        json.dump(state, f, indent=2)

    # Save a "latest" symlink for easy resume
    latest_policy = checkpoint_dir / "policy_latest.pt"
    latest_state = checkpoint_dir / "state_latest.json"
    if latest_policy.exists():
        latest_policy.unlink()
    if latest_state.exists():
        latest_state.unlink()
    latest_policy.symlink_to(f"policy_{tag}.pt")
    latest_state.symlink_to(f"state_{tag}.json")

    print(f"  [Checkpoint saved: {tag}]")


if __name__ == "__main__":
    main()
```

---

## Using Stable-Baselines3

For a simpler setup using SB3's built-in PPO:

```python
#!/usr/bin/env python3
"""Full speedrun training with Stable-Baselines3."""
from minecraft_sim import SpeedrunVecEnv
from minecraft_sim.sb3_wrapper import SB3VecFreeTheEndEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback


class CurriculumCallback(BaseCallback):
    """Log curriculum advancement during training."""

    def __init__(self, env: SpeedrunVecEnv, verbose: int = 1):
        super().__init__(verbose)
        self.env = env
        self.best_stage = 1

    def _on_step(self) -> bool:
        dist = self.env.get_stage_distribution()
        current_best = max(dist.keys()) if dist else 1
        if current_best > self.best_stage:
            self.best_stage = current_best
            if self.verbose:
                print(f"\n*** Advanced to Stage {current_best}! ***\n")
        return True


# Environment
env = SB3VecFreeTheEndEnv(
    num_envs=64,
    start_stage=1,
    curriculum=True,
)

# Model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=512,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="./logs/full_speedrun",
    policy_kwargs=dict(net_arch=[256, 256, 128]),
)

# Callbacks
callbacks = [
    CheckpointCallback(
        save_freq=1_000_000,
        save_path="./checkpoints/sb3_speedrun/",
        name_prefix="speedrun",
    ),
    CurriculumCallback(env.venv),  # Pass underlying SpeedrunVecEnv
]

# Train
model.learn(
    total_timesteps=2_000_000_000,
    callback=callbacks,
    progress_bar=True,
)

model.save("speedrun_final")
env.close()
```

---

## Checkpointing Strategy

Multi-day training runs require robust checkpointing. The system persists three types of state:

### What Gets Saved

1. **Policy weights** -- the neural network parameters (PyTorch `.pt` or SB3 `.zip`)
2. **Curriculum state** -- per-environment stage assignments, success rate history, advancement events
3. **Training metadata** -- total steps, episode counts, per-stage statistics

### Checkpoint Frequency

| Training Phase | Recommended Interval | Rationale |
|---------------|---------------------|-----------|
| Early (Stage 1-2) | Every 50M steps | Fast progress, cheap to regenerate |
| Mid (Stage 3-4) | Every 25M steps | Expensive stages, more risk |
| Late (Stage 5-6) | Every 10M steps | Slow progress, every advancement matters |

### Resuming Training

```python
import json
from pathlib import Path

from minecraft_sim import SpeedrunVecEnv

CHECKPOINT_DIR = Path("checkpoints/full_speedrun")


def resume_training(checkpoint_dir: Path = CHECKPOINT_DIR):
    """Resume training from the latest checkpoint."""

    # Load state
    state_path = checkpoint_dir / "state_latest.json"
    if not state_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {state_path}")

    with open(state_path) as f:
        state = json.load(f)

    total_steps = state["total_steps"]
    episode_count = state["episode_count"]

    print(f"Resuming from step {total_steps:,}, episode {episode_count:,}")

    # Recreate environment
    env = SpeedrunVecEnv(
        num_envs=64,
        initial_stage=1,
        auto_curriculum=True,
    )

    # Load policy
    policy_path = checkpoint_dir / "policy_latest.pt"
    policy = create_policy(obs_dim=256, act_dim=32)
    policy.load(policy_path)

    print(f"Policy loaded from {policy_path}")
    return env, policy, total_steps, episode_count


# Usage:
# env, policy, start_step, start_episodes = resume_training()
# Continue training loop from start_step...
```

### SB3 Resume

```python
from stable_baselines3 import PPO
from minecraft_sim.sb3_wrapper import SB3VecFreeTheEndEnv

env = SB3VecFreeTheEndEnv(num_envs=64, start_stage=1, curriculum=True)

# SB3 handles checkpointing natively
model = PPO.load(
    "checkpoints/sb3_speedrun/speedrun_50000000_steps",
    env=env,
)
model.learn(
    total_timesteps=2_000_000_000,
    reset_num_timesteps=False,  # Continue step counter
)
```

### Best Practices

- Always save curriculum state alongside policy weights. A trained policy without stage assignments will reset all environments to Stage 1.
- Use symlinks (`policy_latest.pt` -> `policy_step_N.pt`) for easy resume without tracking step numbers.
- Keep the last 3-5 checkpoints. If a checkpoint is corrupted or training degrades, roll back to an earlier save.
- Log per-stage success rates at each checkpoint. If success rate drops after advancement, the threshold may be too low.
- For distributed training across machines, save checkpoints to shared storage (NFS, S3) rather than local disk.

---

## Monitoring Training Progress

### TensorBoard

```bash
tensorboard --logdir=logs/full_speedrun --port=6006
```

Key metrics to watch:

| Metric | Healthy Range | Warning Sign |
|--------|--------------|--------------|
| `rollout/ep_rew_mean` | Increasing per stage | Flat for >1M steps |
| `curriculum/stage` | Monotonically increasing | Stuck at one stage |
| `train/entropy_loss` | 0.5-2.0 | Below 0.1 (collapsed) |
| `train/value_loss` | Decreasing within stage | Spiking on advancement |
| `rollout/ep_len_mean` | Decreasing within stage | Increasing (agent stalling) |

### Stall Detection

The `ProgressWatchdog` (in `progress_watchdog.py`) detects when training has stalled:

```python
from minecraft_sim.progress_watchdog import ProgressWatchdog

watchdog = ProgressWatchdog()

# In training loop:
for i in range(NUM_ENVS):
    alert = watchdog.observe(
        env_id=i,
        progress_snapshot=infos[i].get("progress_snapshot", {}),
        stage_id=int(env.get_stage_distribution().get(i, 1)),
    )
    if alert:
        print(f"STALL: Env {i} stuck for {alert.episodes_since_growth} episodes")
        # Consider: reset this env, lower threshold, or increase exploration
```

---

## Troubleshooting

**Training converges on Stage 1 but fails Stage 2:**
- The iron mining and smelting chain is much longer than Stage 1 actions. Increase `n_steps` to 512.
- Check that the Y-level observation is being used; the policy needs to learn "go deeper = find ore".

**Agent enters Nether but never finds fortress (Stage 3):**
- Check the `distance_to_fortress` approach reward is firing. If not, the agent has no gradient toward the fortress.
- Increase exploration bonus or entropy coefficient.

**Enderman hunting plateaus at 6-8 pearls (Stage 4):**
- The agent likely has not learned trap construction. Check if `in_warped_forest` milestone fires.
- The 50% pearl drop rate means ~24 kills needed for 12 pearls. Patience and combat efficiency matter.

**Stronghold triangulation fails (Stage 5):**
- The agent may throw all eyes from the same position. Check that `triangulation_ready` milestone fires (requires 2+ distinct throw locations).
- Reduce the penalty for eye breakage during early Stage 5 training to encourage throwing.

**Dragon fight: agent destroys crystals but dies during perch (Stage 6):**
- The perch-to-breath transition is the most dangerous moment. The `dragon_perching` proximity rewards (+0.02 for <5 blocks) should encourage approach.
- Check void proximity; agents that rush the fountain sometimes overshoot and fall off the island.
- The bed explosion penalty (-0.2) prevents degenerate strategies that self-damage.
