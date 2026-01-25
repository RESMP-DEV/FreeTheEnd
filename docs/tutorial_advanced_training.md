# Advanced Training Tutorial: Minecraft 1.8.9 Speedrun RL

This tutorial provides a deep dive into the observation space, action space, reward engineering, curriculum learning, hyperparameters, and debugging techniques for training RL agents on the GPU-accelerated Minecraft 1.8.9 simulator. It assumes familiarity with PPO, Gymnasium environments, and basic RL concepts.

---

## 1. Understanding the Observation Space

The simulator provides three observation formats depending on the environment type: a compact 48-float GPU observation from the C++ backend (used by `DragonFightEnv`), a 256-float compact format for full speedrun training, and a rich 4,268-float format with voxel grids. Understanding the tradeoffs between them is essential for designing effective policy networks.

### 1.1 The 48-Float GPU Observation (DragonFightEnv)

This is the raw observation produced by the Vulkan compute shaders for the dragon fight, structured into three 16-float regions. All values are normalized to [0, 1]:

**Player block (indices 0-15):**

| Index | Field | Raw Range | Normalization |
|-------|-------|-----------|---------------|
| 0 | `pos_x` | world coords | scaled to [0, 1] by arena bounds |
| 1 | `pos_y` | 0-256 | divided by 256 |
| 2 | `pos_z` | world coords | scaled to [0, 1] by arena bounds |
| 3 | `vel_x` | blocks/tick | clipped to [-1, 1] |
| 4 | `vel_y` | blocks/tick | clipped to [-1, 1] |
| 5 | `vel_z` | blocks/tick | clipped to [-1, 1] |
| 6 | `yaw` | -180 to 180 deg | divided by 180, range [-1, 1] |
| 7 | `pitch` | -90 to 90 deg | divided by 90, range [-1, 1] |
| 8 | `health` | 0-20 HP | divided by 20, range [0, 1] |
| 9 | `hunger` | 0-20 | divided by 20, range [0, 1] |
| 10 | `on_ground` | boolean | 0.0 or 1.0 |
| 11 | `attack_ready` | boolean | 0.0 or 1.0 (cooldown expired) |
| 12 | `weapon` | 0=hand, 1=sword, 2=bow | divided by 2, range [0, 1] |
| 13 | `arrows` | 0-64 | divided by 64, range [0, 1] |
| 14 | `arrow_charge` | 0.0-1.0 | already normalized |
| 15 | `is_burning` | boolean | 0.0 or 1.0 (player on fire) |

**Dragon block (indices 16-31):**

| Index | Field | Raw Range | Normalization |
|-------|-------|-----------|---------------|
| 16 | `dragon_health` | 0-200 HP | divided by 200, range [0, 1] |
| 17 | `dragon_x` | world coords | scaled by arena bounds |
| 18 | `dragon_y` | 0-256 | divided by 256 |
| 19 | `dragon_z` | world coords | scaled by arena bounds |
| 20 | `dragon_vel_x` | blocks/tick | clipped |
| 21 | `dragon_vel_y` | blocks/tick | clipped |
| 22 | `dragon_vel_z` | blocks/tick | clipped |
| 23 | `dragon_yaw` | -180 to 180 | divided by 180 |
| 24 | `dragon_phase` | 0-3 | divided by 3, range [0, 1] |
| 25 | `dragon_dist` | 0-200+ blocks | divided by 200 |
| 26 | `dragon_dir_x` | -1 to 1 | unit vector component |
| 27 | `dragon_dir_z` | -1 to 1 | unit vector component |
| 28 | `can_hit_dragon` | boolean | 0.0 or 1.0 |
| 29 | `dragon_attacking` | boolean | 0.0 or 1.0 |
| 30 | `burn_time_remaining` | ticks | divided by max burn time |
| 31 | reserved | - | 0.0 |

**Environment block (indices 32-47):**

| Index | Field | Raw Range | Normalization |
|-------|-------|-----------|---------------|
| 32 | `crystals_remaining` | 0-10 | divided by 10, range [0, 1] |
| 33 | `nearest_crystal_dist` | 0-200 blocks | divided by 200 |
| 34 | `nearest_crystal_dir_x` | -1 to 1 | unit vector |
| 35 | `nearest_crystal_dir_z` | -1 to 1 | unit vector |
| 36 | `nearest_crystal_y` | 0-256 | divided by 256 |
| 37 | `portal_active` | boolean | 0.0 or 1.0 |
| 38 | `portal_dist` | 0-200 blocks | divided by 200 |
| 39 | `time_remaining` | 0-36000 ticks | divided by 36000 |
| 40 | `total_damage_dealt` | 0-200 HP | divided by 200 |
| 41-47 | reserved | - | 0.0 |

### 1.2 Interpreting Key Observations

**`dragon_phase` (index 24):** Maps to the `DragonPhase` enum defined in `observations.py`:
- 0 = `NONE` (no fight active)
- 1 = `CIRCLING` (flying around the island, vulnerable to arrows)
- 2 = `STRAFING` (diving to attack the player)
- 3 = `PERCHING` (landed on the fountain, vulnerable to melee)

The perching phase is the primary damage window. When `dragon_phase == 3`, the dragon is stationary and takes full melee damage. A well-timed melee combo during a single perch can deal 50-80 HP if the agent positions correctly.

**`crystals_remaining` (index 32):** The number of end crystals still alive (0-10). Crystals heal the dragon by 1 HP/tick when the dragon is within range. Destroying all crystals before dealing significant dragon damage is the optimal strategy.

**`can_hit_dragon` (index 28):** A precomputed flag indicating whether the player's current position and facing angle would result in a successful attack. This accounts for reach distance (3 blocks for melee, line-of-sight for arrows) and the dragon's hitbox. When this is 1.0 and `attack_ready` (index 11) is 1.0, the policy should strongly prefer the ATTACK action.

### 1.3 The 256-Float Compact Observation

For full speedrun training, the `CompactObservationEncoder` produces a 256-float vector with the following layout:

```
[0-31]    Player state (position, velocity, health, flags)
[32-63]   Inventory (hotbar items/counts, key resources)
[64-127]  Environment (local features, stage-specific data)
[128-191] Goals (curriculum stage one-hot at [128-133], objectives)
[192-223] Dragon state (health, position, phase)
[224-255] Dimension/Portal state
```

Decoding via `CompactObservationDecoder`:

**Player state (0-31):** Position is scaled by 1000 (denormalized from [0,1] to world coords), velocity by 10, pitch by 90, yaw by 180, health/hunger by 20, armor by 20, XP level by 30. Boolean flags (sprinting, sneaking, jumping, on_ground, in_water, in_lava, burning) are thresholded at 0.5.

**Inventory (32-63):** Hotbar item IDs scaled by 512, hotbar counts by 64, selected slot by 9. Key resources (wood, iron, diamond, blaze_rod, ender_pearl, eye_of_ender, food) are scaled by 64 or 16 depending on typical max counts.

**Goals (128-191):** Stage one-hot at indices 128-133 (argmax + 1 = stage number).

**Dragon (192-223):** dragon_exists flag at 192, health scaled by 200, position by 100, is_perched at 197, crystals_remaining scaled by 10.

**Dimension/Portal (224-255):** Dimension one-hot at 224-226, portal proximity flags, portal distance scaled by 64, end portal frame/eye counts scaled by 12.

### 1.4 The Full Observation (172 + 4096 floats)

The `MinecraftObservation` dataclass provides the richest representation:

| Component | Dimensions | Contents |
|-----------|-----------|----------|
| `PlayerState` | 22 | position (3), velocity (3), health/hunger/saturation/exhaustion (5), yaw/pitch (2), dimension (1), equipment (2), flags (5), padding (1) |
| `InventorySummary` | 25 | key item counts (16, log-normalized), hotbar item IDs (9) |
| `RayCastDistances` | 16 | 8 horizontal + 4 upward + 4 downward rays, normalized by 64 |
| `EntityAwareness` | 96 | 8 mobs x 8 features + 4 items x 8 features |
| `DragonState` | 10 | is_active, phase/3, health, crystals/10, dx/dy/dz, distance/200, phase_timer, padding |
| Metadata | 3 | game_tick/36000, terminated, truncated |
| **Total (no voxels)** | **172** | |
| `VoxelGrid` (binary) | 4096 | 16x16x16 solid/not-solid grid, index: y*256 + z*16 + x |
| **Total (with voxels)** | **4268** | |

### 1.5 Decoding Observations in Python

```python
import numpy as np
from minecraft_sim.observations import (
    CompactObservationDecoder,
    MinecraftObservation,
    DragonPhase,
)

# For 48-float GPU observations
def decode_dragon_fight_obs(obs: np.ndarray) -> dict:
    """Decode the 48-float GPU observation into named fields."""
    return {
        "player_pos": obs[0:3],
        "player_vel": obs[3:6],
        "yaw": obs[6] * 180,          # degrees
        "pitch": obs[7] * 90,          # degrees
        "health": obs[8] * 20,         # HP
        "hunger": obs[9] * 20,
        "on_ground": obs[10] > 0.5,
        "attack_ready": obs[11] > 0.5,
        "weapon": int(obs[12] * 2),    # 0=hand, 1=sword, 2=bow
        "arrows": int(obs[13] * 64),
        "arrow_charge": obs[14],
        "dragon_health": obs[16] * 200, # HP out of 200
        "dragon_phase": DragonPhase(int(round(obs[24] * 3))),
        "dragon_dist": obs[25] * 200,   # blocks
        "can_hit": obs[28] > 0.5,
        "crystals": int(obs[32] * 10),
    }

# For 256-float compact observations
def analyze_compact_observation(obs: np.ndarray) -> None:
    """Decode and print key values from 256-float observation."""
    decoder = CompactObservationDecoder(obs)

    player = decoder.get_player_state()
    print(f"Position: ({player.position[0]:.1f}, {player.position[1]:.1f}, {player.position[2]:.1f})")
    print(f"Health: {player.health:.1f}/20")
    print(f"On ground: {player.is_on_ground}, Sprinting: {player.is_sprinting}")

    inv = decoder.get_inventory()
    print(f"Blaze rods: {inv.blaze_rod_count}")
    print(f"Ender pearls: {inv.ender_pearl_count}")
    print(f"Eyes of ender: {inv.eye_of_ender_count}")

    print(f"Stage: {decoder.get_current_stage()}")
    print(f"Dimension: {['Overworld', 'Nether', 'End'][decoder.get_dimension()]}")

    dragon = decoder.get_dragon_state()
    if dragon:
        print(f"Dragon HP: {dragon['health']:.0f}/200")
        print(f"Crystals left: {dragon['crystals_remaining']}")
        print(f"Perching: {dragon['is_perched']}")

    portal = decoder.get_portal_state()
    print(f"Near portal: nether={portal['near_nether_portal']}, end={portal['near_end_portal']}")
```

### 1.6 Observation Normalization Strategy

All observations are normalized to prevent gradient explosion in the policy network. The strategy varies by field type:

- **Positions:** Divided by dimension bounds. Overworld uses +/-30M, Nether uses +/-3.75M, End uses +/-30M. In practice positions stay within a few thousand blocks, so values cluster near 0.
- **Velocities:** Divided by 10 and clipped to [-1, 1]. Normal walking speed is ~0.43 blocks/tick, sprinting is ~0.56.
- **Health/hunger:** Divided by max value (20). Simple [0, 1] range.
- **Item counts:** Log-normalized via `log1p(count) / log(65)`. Maps 0 to 0, 64 to 1. Handles varying stack sizes gracefully.
- **Angles:** Yaw divided by 180 ([-1, 1]), pitch divided by 90 ([-1, 1]).
- **Booleans:** Direct 0.0/1.0 encoding.
- **Distances:** Divided by max expected distance for the context (200 for dragon, 64 for raycasts).
- **Entity features:** type_id/100, distance/64, direction as unit vector, health normalized [0,1].

---

## 2. Action Space Deep Dive

The simulator uses a 32-action discrete space defined by the `Action` enum in `actions_discrete.py`. Each action maps to specific keyboard/mouse inputs via `ACTION_TO_KEYS`.

### 2.1 All 32 Actions

```python
from minecraft_sim.actions_discrete import Action

# Movement (0-6)
Action.NOOP          = 0   # No input
Action.FORWARD       = 1   # key_w=True
Action.BACKWARD      = 2   # key_s=True
Action.LEFT          = 3   # key_a=True
Action.RIGHT         = 4   # key_d=True
Action.FORWARD_LEFT  = 5   # key_w + key_a
Action.FORWARD_RIGHT = 6   # key_w + key_d

# Jump variants (7-8)
Action.JUMP          = 7   # key_space=True
Action.JUMP_FORWARD  = 8   # key_space + key_w

# Combat (9-10)
Action.ATTACK         = 9   # mouse_left=True
Action.ATTACK_FORWARD = 10  # mouse_left + key_w

# Sprint toggle (11)
Action.SPRINT_TOGGLE = 11  # key_ctrl=True

# Look directions (12-19)
Action.LOOK_LEFT      = 12  # mouse_dx=-15.0 degrees
Action.LOOK_RIGHT     = 13  # mouse_dx=+15.0 degrees
Action.LOOK_UP        = 14  # mouse_dy=-15.0 degrees
Action.LOOK_DOWN      = 15  # mouse_dy=+15.0 degrees
Action.LOOK_LEFT_FAST = 16  # mouse_dx=-45.0 degrees
Action.LOOK_RIGHT_FAST= 17  # mouse_dx=+45.0 degrees
Action.LOOK_UP_FAST   = 18  # mouse_dy=-45.0 degrees
Action.LOOK_DOWN_FAST = 19  # mouse_dy=+45.0 degrees

# Item interaction (20-21)
Action.USE_ITEM  = 20  # mouse_right=True
Action.DROP_ITEM = 21  # key_q=True

# Hotbar selection (22-30)
Action.HOTBAR_1 = 22  # hotbar_slot=0
Action.HOTBAR_2 = 23  # hotbar_slot=1
# ... through ...
Action.HOTBAR_9 = 30  # hotbar_slot=8

# Crafting (31)
Action.QUICK_CRAFT = 31  # key_e + special_craft flag
```

**DragonFightEnv uses 17 actions** (a subset: 0-16), matching the `action_space = Discrete(17)` specification.

### 2.2 Action Groups

The actions are organized into functional groups for analysis:

```python
from minecraft_sim.actions_discrete import (
    MOVEMENT_ACTIONS,  # {1, 2, 3, 4, 5, 6}
    JUMP_ACTIONS,      # {7, 8}
    COMBAT_ACTIONS,    # {9, 10}
    LOOK_ACTIONS,      # {12, 13, 14, 15, 16, 17, 18, 19}
    HOTBAR_ACTIONS,    # {22, 23, 24, 25, 26, 27, 28, 29, 30}
)
```

Use `analyze_action_distribution(actions)` to profile your policy's action selection patterns. Healthy training shows a mix of movement and look actions early, transitioning to more combat actions as the policy improves.

### 2.3 Movement Math

Movement is specified in the player's local frame. FORWARD (action 1) moves along the player's facing direction (determined by yaw). When FORWARD combines with yaw rotation:

```
World-space velocity:
  dx = speed * sin(yaw) * forward_component + speed * cos(yaw) * strafe_component
  dz = speed * cos(yaw) * forward_component - speed * sin(yaw) * strafe_component
```

Movement speeds per tick:
- Walking: 4.317 blocks/sec = 0.2159 blocks/tick
- Sprinting: 5.612 blocks/sec = 0.2806 blocks/tick
- Jump-sprint: burst of ~0.35 blocks/tick horizontal

Diagonal movement (FORWARD_LEFT, FORWARD_RIGHT) combines both forward and strafe components at 1/sqrt(2) each, resulting in the same total speed as cardinal movement. This is normalized to prevent diagonal speed exploits.

Look actions rotate the camera by fixed amounts per tick. The 15-degree (normal) and 45-degree (fast) options give the policy coarse and fine aiming control:

```python
# Effective turn rate
LOOK_LEFT:       -15 deg/tick = 300 deg/sec
LOOK_LEFT_FAST:  -45 deg/tick = 900 deg/sec (full 180 in 4 ticks)
```

### 2.4 Attack Timing and Perch Detection

Attack success depends on conditions checked by the C++ backend:

1. **Cooldown:** The attack cooldown is 10 ticks (0.5 seconds). `attack_ready` (obs index 11) must be 1.0 before the next attack deals damage.
2. **Range:** Melee reach is 3 blocks. The player must be within 3 blocks of the target's hitbox.
3. **Facing:** The player's look direction must intersect the target's bounding box.
4. **`can_hit_dragon` flag:** Index 28 precomputes all of the above, giving a single boolean the policy can condition on.

Perch detection strategy for maximum DPS:

```python
def is_perch_opportunity(obs: np.ndarray) -> bool:
    """Check if dragon is perching and we can attack."""
    dragon_phase = int(round(obs[24] * 3))  # DragonPhase enum
    can_hit = obs[28] > 0.5
    attack_ready = obs[11] > 0.5
    dragon_dist = obs[25] * 200  # blocks

    return (
        dragon_phase == 3           # PERCHING
        and dragon_dist < 5.0       # Close enough for melee
        and can_hit
        and attack_ready
    )
```

For bow attacks during the circling phase:
- Hold USE_ITEM (action 20) with bow selected for 20+ ticks to charge
- Release when `arrow_charge` (index 14) >= 1.0
- Lead the target: aim ahead of the dragon's velocity vector
- Optimal pitch for distant targets: -15 to -30 degrees (arcing trajectory)

### 2.5 Composite Actions

JUMP_FORWARD (action 8) simultaneously jumps and moves forward, critical for sprint-jumping which is the fastest overworld traversal. ATTACK_FORWARD (action 10) attacks while maintaining forward pressure, essential during dragon perch combos where you need to close distance and deal damage simultaneously.

```python
# Sprint-jump sequence for maximum speed
sprint_jump_sequence = [
    Action.SPRINT_TOGGLE,     # Enable sprint
    Action.JUMP_FORWARD,      # Sprint-jump
    Action.FORWARD,           # Maintain momentum
    Action.FORWARD,
    Action.JUMP_FORWARD,      # Next sprint-jump
]
# This achieves ~7 blocks/sec sustained horizontal speed

# Dragon perch melee combo
perch_combo = []
for _ in range(5):
    perch_combo.append(Action.ATTACK_FORWARD)  # Hit + approach
    perch_combo.extend([Action.FORWARD] * 9)    # Wait cooldown
# Deals ~50 HP over one perch phase
```

### 2.6 Action Masking

Invalid actions are masked before sampling using `get_action_mask()`:

```python
from minecraft_sim.actions_discrete import get_action_mask

state = {
    "held_item": 0,       # Nothing in hand
    "can_craft": False,   # No craftable recipe
}
mask = get_action_mask(stage=1, state=state)
# mask[20] = False  (USE_ITEM invalid, nothing in hand)
# mask[21] = False  (DROP_ITEM invalid, nothing to drop)
# mask[31] = False  (QUICK_CRAFT invalid, can't craft)
```

Apply the mask to policy logits before sampling:

```python
import torch

logits = policy_network(obs)  # shape: (batch, 32)
mask_tensor = torch.tensor(mask, dtype=torch.bool)
logits[~mask_tensor] = float('-inf')
action_dist = torch.distributions.Categorical(logits=logits)
action = action_dist.sample()
```

---

## 3. Reward Engineering

The reward system uses stage-specific reward shapers from `reward_shaping.py`. Each shaper is a stateful callable that maintains milestone tracking and previous-state comparisons. Understanding the four reward components is critical for modifying training behavior.

### 3.1 Default Reward Structure

Every stage implements four reward components:

1. **Time penalty:** Small negative per tick. Scales with stage difficulty: -0.0001 (Stage 1), -0.00012 (Stages 3, 5), -0.00015 (Stage 4), -0.0002 (Stage 6).
2. **Death/damage penalties:** Death penalties range from -0.8 (Stages 2, 5) to -2.0 (Stage 6). Damage penalties are proportional: 0.015-0.025 per HP lost.
3. **Milestone bonuses:** One-time rewards tracked by name in a `given_rewards` set. Range from 0.05 (minor discoveries) to 1.0 (dragon killed).
4. **Progressive rewards:** Per-unit rewards with diminishing returns via `min(count * rate, cap)`. The delta from previous state is computed each tick.

Stage completion bonuses: +2.0 (Stages 1, 4), +2.5 (Stages 3, 5), +5.0 (Stage 6).

### 3.2 Stage-by-Stage Reward Breakdown

**Stage 1: Basic Survival** (time penalty: -0.0001, death: -1.0)

Key milestones:
```
first_wood: +0.2    wood_x4: +0.1      wood_x8: +0.1
first_planks: +0.1  crafting_table: +0.15
wooden_pickaxe: +0.3  stone_pickaxe: +0.3
first_stone: +0.15  first_coal: +0.15
first_iron_ore: +0.25  first_kill: +0.15
```

Progressive: wood * 0.02 (cap 0.2), stone * 0.005 (cap 0.2), coal * 0.01 (cap 0.15), kills * 0.1 each, chunks * 0.01 each.

**Stage 2: Resource Gathering** (time penalty: -0.0001, death: -0.8)

Key milestones:
```
first_iron_ingot: +0.2  iron_pickaxe: +0.35
bucket: +0.3  first_diamond: +0.3  diamond_pickaxe: +0.3
first_obsidian: +0.2  obsidian_x10: +0.25
flint_and_steel: +0.15
```

Progressive: iron * 0.015 (cap 0.3), obsidian * 0.015 (cap 0.25). Bonus for reaching y < 16 (deeper mining).

**Stage 3: Nether Navigation** (time penalty: -0.00012, death: -1.2)

Key milestones:
```
entered_nether: +0.4  fortress_found: +0.4
first_blaze_kill: +0.3  blaze_rod_x7: +0.25
ghast_fireball_deflected: +0.3
```

Progressive: blaze_rods * 0.03 (cap 0.35), blaze kills * 0.15 each. Distance-to-fortress approach reward: `min((prev_dist - curr_dist) * 0.001, 0.05)`. Extra damage penalty multiplier (0.03 vs 0.02) for fire/lava damage.

**Stage 4: Enderman Hunting** (time penalty: -0.00015, death: -1.0)

Key milestones:
```
first_enderman_kill: +0.25  pearl_x12: +0.25
first_eye: +0.2  eye_x12: +0.25
in_warped_forest: +0.15  portal_activated: +1.5
```

Progressive: pearls * 0.02 (cap 0.4), eyes * 0.025 (cap 0.35), enderman kills * 0.12 each, portal frames * 0.1 each.

**Stage 5: Stronghold Finding** (time penalty: -0.00012, death: -0.8)

Key milestones:
```
stronghold_entered: +0.3  portal_room_found: +0.35
portal_activated: +0.5  efficient_triangulation: +0.2
```

Progressive: approach reward `min((prev_dist - curr_dist) * 0.0005, 0.03)`, frame filling * 0.03 each.

**Stage 6: Dragon Fight** (time penalty: -0.0002, death: -2.0)

```python
# Penalties
time_penalty = -0.0002          # Per tick (highest pressure)
death_penalty = -2.0             # Per death (very costly)
damage_penalty = -0.025 * dmg   # Per HP lost
void_penalty = -0.1             # Near void edge (y < 10)

# Crystal destruction
per_crystal = +0.15             # Each crystal destroyed
first_crystal_milestone = +0.2  # First crystal
all_crystals_milestone = +0.3   # All 10 destroyed

# Dragon damage
per_hp_dealt = +0.005           # Per HP of damage to dragon
half_health = +0.2              # Dragon at 100 HP
quarter_health = +0.2           # Dragon at 50 HP
dragon_critical = +0.15         # Dragon at 20 HP

# Perch attacks
first_perch_hit = +0.25         # First melee during perch
perch_combo = +0.2              # 3+ hits in one perch
per_arrow_hit = +0.08           # Each arrow that connects

# Positioning
close_during_perch = +0.02      # Within 5 blocks when dragon perches
bow_range_during_perch = +0.01  # Within 15 blocks

# Victory
dragon_killed = +1.0            # Kill milestone
one_cycle = +0.5                # Kill in first perch
fast_kill = +0.3                # Kill in < 3000 ticks
speedrun_pace = +0.5            # Total run < 60000 ticks
stage_complete = +5.0           # Final bonus
```

### 3.3 Modifying Rewards

To create a custom reward shaper, wrap or replace the factory:

```python
from minecraft_sim.reward_shaping import create_reward_shaper, RewardStats

def create_crystal_priority_shaper():
    """Modified Stage 6 shaper emphasizing fast crystal destruction."""
    base_shaper = create_reward_shaper(6)
    given_rewards: set[str] = set()

    def custom_shape(state: dict) -> float:
        reward = base_shaper(state)

        # Extra bonus for destroying crystals quickly
        crystals = state.get("crystals_destroyed", 0)
        ticks = state.get("ticks_elapsed", 0)
        if crystals >= 10 and ticks < 1500 and "speed_crystals" not in given_rewards:
            reward += 1.5  # Major bonus for fast crystal clear
            given_rewards.add("speed_crystals")

        # Extra penalty for taking damage while crystals remain
        if crystals < 10:
            hp_loss = state.get("damage_this_tick", 0)
            if hp_loss > 0:
                reward -= hp_loss * 0.01  # Additional crystal-phase damage penalty

        return reward

    custom_shape.reset = lambda: (given_rewards.clear(), base_shaper.reset())
    return custom_shape
```

### 3.4 Reward Shaping for Faster Learning

Key principles for effective reward shaping in this environment:

**Milestone spacing:** Place milestones at regular intervals of difficulty. For blaze rod collection (Stage 3), milestones at 1, 3, 5, 7, and 10 rods create a smooth gradient. Gaps larger than 3-4 units of progress leave the agent without gradient signal.

**Progressive reward caps:** The `min(count * rate, cap)` pattern prevents reward hacking. Without caps, the agent could learn to repeatedly gather and drop items. The cap ensures marginal reward decreases to zero after sufficient collection.

**Penalty calibration:** Death penalties should be 5-10x the typical per-episode positive reward from milestones. At Stage 6, the maximum positive reward from milestones alone is approximately 5.0 (crystals + damage + kill), so the death penalty of -2.0 means two deaths nearly cancels a successful kill.

**Time penalty tuning:** At -0.0002/tick over a 6000-tick dragon fight episode, the maximum time penalty is -1.2. A slow agent that destroys all crystals earns approximately 2.5 (crystals alone), while a fast agent that kills the dragon earns ~8.0 total. The time penalty creates urgency without overwhelming milestone rewards.

### 3.5 Using CompositeRewardShaper

The `CompositeRewardShaper` manages transitions between stages automatically:

```python
from minecraft_sim.reward_shaping import CompositeRewardShaper

shaper = CompositeRewardShaper(initial_stage=1)

# During training loop
reward = shaper.shape_reward(state)

# On stage advancement
if state.get("stage_complete"):
    shaper.advance_stage()
    # New stage shaper starts with empty milestones

# Inspect what was achieved
stats = shaper.get_stats()
if stats:
    print(f"Milestones: {stats.milestones_achieved}")
    print(f"Milestone rewards: {stats.milestone_rewards:.2f}")
    print(f"Progressive rewards: {stats.progressive_rewards:.2f}")
    print(f"Penalties: {stats.penalties:.2f}")
    print(f"Total: {stats.total_reward:.2f}")

# Reset on episode end
shaper.reset(stage_id=shaper.current_stage)
```

Each stage's shaper maintains its own `given_rewards` set and `prev_state` reference, ensuring milestones from previous stages don't interfere with the current stage's reward signal.

---

## 4. Curriculum Learning Setup

### 4.1 VecCurriculumManager Configuration

The `VecCurriculumManager` tracks curriculum progress independently for each environment in a vectorized setup. This enables heterogeneous training where different environments can be at different difficulty levels simultaneously:

```python
from minecraft_sim.curriculum_manager import (
    VecCurriculumManager,
    StageOverride,
)

# Basic configuration
manager = VecCurriculumManager(
    num_envs=64,
    min_stage=1,
    max_stage=6,
    advancement_threshold=0.7,   # 70% success rate to advance
    regression_threshold=0.2,    # 20% success rate to regress
    window_size=100,             # Track last 100 episodes per env
    min_episodes_to_advance=20,  # Wait at least 20 episodes
    min_episodes_to_regress=50,  # Wait at least 50 before regressing
    enable_regression=False,     # Disable regression by default
)
```

### 4.2 Stage Overrides

Per-stage `StageOverride` allows fine-grained control over advancement criteria:

```python
manager = VecCurriculumManager(
    num_envs=64,
    stage_overrides={
        1: StageOverride(
            min_episodes_to_advance=10,      # Fast early advancement
            advancement_threshold=0.7,
        ),
        2: StageOverride(
            min_episodes_to_advance=30,
            advancement_threshold=0.75,      # Higher bar for resources
            min_dimension_episodes=5,        # Must complete 5 episodes flagged as
                                             # in-target-dimension (obsidian mining)
        ),
        3: StageOverride(
            min_episodes_to_advance=40,
            advancement_threshold=0.6,       # Nether is hard
            min_metric_value=5.0,            # Must track metric >= 5 (blaze rods)
            sustained_windows=3,             # Must maintain rate for 3 windows
            sustained_window_size=15,        # Each window is 15 episodes
        ),
        4: StageOverride(
            min_episodes_to_advance=30,
            advancement_threshold=0.65,
        ),
        5: StageOverride(
            min_episodes_to_advance=50,
            advancement_threshold=0.55,      # Stronghold search is unreliable
        ),
        6: StageOverride(
            min_episodes_to_advance=100,
            advancement_threshold=0.3,       # Dragon fight is hard
        ),
    },
)
```

The `StageOverride` fields:
- `min_episodes_to_advance`: Override minimum episodes before advancement check.
- `advancement_threshold`: Override success rate required.
- `min_metric_value`: Minimum tracked metric (reported via `update(..., metric_value=...)`) to allow advancement.
- `min_dimension_episodes`: Minimum episodes flagged `in_target_dimension=True`.
- `sustained_windows`: Number of consecutive evaluation windows above threshold.
- `sustained_window_size`: Episodes per sustained evaluation window.

### 4.3 Stage Promotion Criteria

The advancement check (`_check_advancement`) evaluates conditions in order:

1. **Stage limit:** Cannot advance past `max_stage`.
2. **Minimum episodes:** Must have completed `min_episodes_to_advance` at this stage (per-env).
3. **Success rate:** Rolling success rate over recent episodes must meet `advancement_threshold`.
4. **Metric threshold:** If `min_metric_value` is set, `_env_metric_values[env_id]` must be at or above this value.
5. **Dimension episodes:** If `min_dimension_episodes` is set, the env must have accumulated enough in-target-dimension episodes.
6. **Sustained windows:** If configured, success rate must remain above threshold across N consecutive windows. If the rate drops below threshold during the sustained period, the counter resets to zero.

On advancement: the env's stage increments, episode count resets, success history clears, and all metric/dimension/sustained counters reset.

### 4.4 Multi-Stage Training Loop

A complete training loop with per-environment curriculum and reward shaping:

```python
import numpy as np
from minecraft_sim.env import make_vec
from minecraft_sim.curriculum_manager import VecCurriculumManager, StageOverride
from minecraft_sim.reward_shaping import create_reward_shaper

NUM_ENVS = 64
ROLLOUT_STEPS = 2048

def train_with_curriculum():
    env = make_vec(num_envs=NUM_ENVS, base_seed=42)
    manager = VecCurriculumManager(
        num_envs=NUM_ENVS,
        advancement_threshold=0.7,
        min_episodes_to_advance=20,
        enable_regression=True,
    )

    # Per-env reward shapers (each maintains independent state)
    shapers = [create_reward_shaper(1) for _ in range(NUM_ENVS)]

    obs, infos = env.reset()
    total_steps = 0

    while total_steps < 100_000_000:
        # Collect rollout
        for step in range(ROLLOUT_STEPS):
            actions = policy.act(obs)  # Your policy here
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            # Apply reward shaping per environment
            for i in range(NUM_ENVS):
                state_snapshot = info.get(f"env_{i}_state", {})
                if state_snapshot:
                    shaped = shapers[i](state_snapshot)
                    rewards[i] += shaped

            # Handle episode terminations
            for i in range(NUM_ENVS):
                if dones[i]:
                    success = bool(info.get(f"env_{i}_stage_complete", False))
                    stage = manager.get_stage(i)
                    ep_reward = float(info.get(f"env_{i}_episode_reward", 0.0))
                    ep_length = int(info.get(f"env_{i}_episode_length", 0))

                    advanced = manager.update(
                        env_id=i,
                        success=success,
                        stage=stage,
                        reward=ep_reward,
                        episode_length=ep_length,
                    )

                    if advanced:
                        new_stage = manager.get_stage(i)
                        shapers[i] = create_reward_shaper(new_stage)
                        shapers[i].reset() if hasattr(shapers[i], 'reset') else None

                    # Reset shaper state for new episode
                    if hasattr(shapers[i], 'reset'):
                        shapers[i].reset()

            # Store transition in rollout buffer
            rollout_buffer.add(obs, actions, rewards, dones, next_obs)
            obs = next_obs
            total_steps += NUM_ENVS

        # Update policy
        policy.update(rollout_buffer)

        # Periodic logging
        if total_steps % (NUM_ENVS * ROLLOUT_STEPS * 10) == 0:
            stats = manager.get_stats()
            print(f"\n--- Step {total_steps:,} ---")
            print(f"Stage distribution: {stats['stage_distribution']}")
            print(f"Success rates: {stats['success_rates']}")
            print(f"Total advancements: {stats['total_advancements']}")
            print(manager.get_stage_summary())

    return manager.get_stats()
```

### 4.5 Batch Updates for Efficiency

When multiple environments terminate simultaneously, use `update_batch`:

```python
# Identify terminated environments
done_mask = terminated | truncated
done_ids = np.where(done_mask)[0].astype(np.int32)

if len(done_ids) > 0:
    successes = np.array([info.get(f"env_{i}_stage_complete", False) for i in done_ids])
    rewards_arr = np.array([info.get(f"env_{i}_episode_reward", 0.0) for i in done_ids], dtype=np.float32)

    advanced_mask = manager.update_batch(
        env_ids=done_ids,
        successes=successes,
        rewards=rewards_arr,
    )

    # Update shapers for advanced environments
    for idx, env_id in enumerate(done_ids):
        if advanced_mask[idx]:
            new_stage = manager.get_stage(env_id)
            shapers[env_id] = create_reward_shaper(new_stage)
```

---

## 5. Hyperparameter Reference

### 5.1 PPO Settings That Work

The `TrainingConfig` dataclass in `training/training_config.py` defines validated defaults:

```python
from minecraft_sim.training.training_config import TrainingConfig

config = TrainingConfig()
# config.ppo.learning_rate = 3e-4
# config.ppo.n_steps = 2048        # Steps per rollout per env
# config.ppo.batch_size = 64       # Minibatch size
# config.ppo.n_epochs = 10         # Passes over each rollout
# config.ppo.gamma = 0.99          # Discount factor
# config.ppo.gae_lambda = 0.95     # GAE parameter
# config.ppo.clip_range = 0.2      # PPO clipping
# config.ppo.ent_coef = 0.01       # Entropy coefficient
# config.ppo.vf_coef = 0.5         # Value function coefficient
# config.ppo.max_grad_norm = 0.5   # Gradient clipping
```

For later stages with longer horizons, adjust:

```python
LATE_STAGE_OVERRIDES = {
    "gamma": 0.999,              # Higher discount for long episodes
    "n_steps": 4096,             # Longer rollouts to capture full episodes
    "batch_size": 256,           # Larger batches for stability
    "ent_coef": 0.02,            # More exploration for complex stages
}
```

Load config from YAML for reproducibility:

```yaml
# training_config.yaml
env:
  num_envs: 64
  max_episode_steps: 36000
  start_stage: 1
  seed: 42

ppo:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5

curriculum:
  enabled: true
  stage_thresholds: [0.7, 0.7, 0.7, 0.7, 0.7]
  min_episodes_per_stage: 1000
  success_window: 100

total_timesteps: 100000000
normalize_obs: true
normalize_reward: true
clip_obs: 10.0
clip_reward: 10.0
device: auto
```

```python
config = TrainingConfig.from_yaml("training_config.yaml")
```

### 5.2 Learning Rate Schedules

Linear decay works well for curriculum training where the task difficulty increases:

```python
def linear_schedule(initial_lr: float = 3e-4, final_lr: float = 1e-5):
    """Linear LR decay over training progress."""
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule

def cosine_schedule(initial_lr: float = 3e-4, min_lr: float = 1e-5):
    """Cosine annealing for single-stage fine-tuning."""
    import math
    def schedule(progress_remaining: float) -> float:
        return min_lr + 0.5 * (initial_lr - min_lr) * (
            1 + math.cos(math.pi * (1 - progress_remaining))
        )
    return schedule
```

For curriculum transitions, consider per-stage learning rates:

```python
def curriculum_lr(stage: int) -> float:
    """Per-stage learning rate. Higher for early stages, lower for precision stages."""
    return {
        1: 3e-4,   # Fast learning for basic skills
        2: 2e-4,   # Resource gathering optimization
        3: 2e-4,   # Nether navigation
        4: 1.5e-4, # Combat refinement
        5: 1e-4,   # Long-horizon planning
        6: 1e-4,   # Dragon fight precision
    }[stage]
```

### 5.3 Batch Size vs Throughput Tradeoffs

The effective batch size is `num_envs * n_steps`. The relationship between these and training quality:

| `num_envs` | `n_steps` | Effective batch | Gradient noise | GPU utilization |
|------------|-----------|-----------------|----------------|-----------------|
| 64 | 128 | 8,192 | High | Low |
| 64 | 2048 | 131,072 | Medium | Medium |
| 256 | 512 | 131,072 | Medium | High |
| 1024 | 128 | 131,072 | Medium | High |
| 4096 | 64 | 262,144 | Low | Saturated |

Larger effective batches reduce gradient variance but may require higher learning rates to maintain update magnitude. The square root scaling rule applies: when doubling batch size, multiply LR by ~1.4.

The PPO `batch_size` parameter controls the minibatch size for gradient updates within each epoch. The number of gradient updates per rollout is `(num_envs * n_steps) / batch_size * n_epochs`. For the default config: `(64 * 2048) / 64 * 10 = 20,480` gradient steps per rollout.

### 5.4 Entropy Coefficient Tuning

The entropy coefficient controls exploration vs exploitation:

| Phase | `ent_coef` | Effect |
|-------|-----------|--------|
| Stage 1, early | 0.05 | Agent tries many movement patterns |
| Stage 1, converging | 0.01 | Focuses on successful strategies |
| Stage 3 (Nether) | 0.02-0.05 | Prevents getting stuck near portal |
| Stage 6 (dragon) | 0.005-0.01 | Precise attack timing needed |
| Fine-tuning | 0.001 | Maximum exploitation |

A common failure is entropy collapse: the policy becomes deterministic too early and converges to a local optimum. Monitor policy entropy in TensorBoard. If entropy drops below 0.5 nats before convergence, increase `ent_coef`.

### 5.5 Discount Factor by Stage

```python
STAGE_GAMMA = {
    1: 0.99,    # ~100 tick effective horizon
    2: 0.995,   # ~200 ticks (resource sequences)
    3: 0.999,   # ~1000 ticks (Nether navigation)
    4: 0.997,   # ~333 ticks (enderman hunting)
    5: 0.999,   # ~1000 ticks (stronghold search)
    6: 0.999,   # ~1000 ticks (multi-phase fight)
}
```

The effective planning horizon at 95% weight is approximately `1 / (1 - gamma)` ticks. For Stage 3 where the agent must navigate hundreds of blocks through the Nether, gamma=0.999 allows credit assignment across ~1000 ticks.

### 5.6 Observation and Reward Normalization

The `TrainingConfig` includes normalization settings:

```python
config.normalize_obs = True    # Running mean/std normalization
config.normalize_reward = True # Running reward normalization
config.clip_obs = 10.0         # Clip normalized obs to [-10, 10]
config.clip_reward = 10.0      # Clip normalized rewards to [-10, 10]
```

Running normalization is critical because observation magnitudes vary by dimension (positions near 0 in normalized form, but voxel grids are 0/1). Without it, gradient magnitudes from different observation components are imbalanced.

---

## 6. Debugging Training

### 6.1 TensorBoard Metrics to Watch

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f"runs/{config.run_name}")

# Core training metrics (log every episode)
writer.add_scalar("train/episode_reward", ep_reward, step)
writer.add_scalar("train/episode_length", ep_length, step)
writer.add_scalar("train/success_rate", success_rate, step)

# Curriculum metrics
writer.add_scalar("curriculum/mean_stage", np.mean(manager.env_stages), step)
writer.add_scalar("curriculum/max_stage", np.max(manager.env_stages), step)
writer.add_scalar("curriculum/total_advancements", len(manager.advancement_history), step)
for stage_id, rate in manager.get_stats()["success_rates"].items():
    writer.add_scalar(f"curriculum/stage_{stage_id}_success_rate", rate, step)

# Policy health metrics (log every update)
writer.add_scalar("policy/entropy", entropy, step)
writer.add_scalar("policy/approx_kl", approx_kl, step)
writer.add_scalar("policy/clip_fraction", clip_frac, step)
writer.add_scalar("policy/value_loss", vf_loss, step)
writer.add_scalar("policy/policy_loss", pg_loss, step)
writer.add_scalar("policy/explained_variance", explained_var, step)

# Reward decomposition
writer.add_scalar("reward/milestone_total", stats.milestone_rewards, step)
writer.add_scalar("reward/progressive_total", stats.progressive_rewards, step)
writer.add_scalar("reward/penalties_total", stats.penalties, step)

# Action distribution
from minecraft_sim.actions_discrete import analyze_action_distribution
action_stats = analyze_action_distribution(epoch_actions)
for group, pct in action_stats["percentages"].items():
    writer.add_scalar(f"actions/{group}_pct", pct, step)
```

Key indicators of healthy training:

- **Explained variance > 0.5:** The value function meaningfully predicts returns.
- **Clip fraction 0.05-0.15:** PPO is making moderate updates. Too low means learning too slowly; too high means updates are too large and training may be unstable.
- **Entropy decreasing gradually:** The policy becomes more certain over time without collapsing instantly.
- **Episode length increasing then decreasing:** The agent first learns to survive longer, then completes objectives faster.

### 6.2 Common Failure Modes

**Failure: Agent never moves (NOOP collapse)**

Symptoms: Episode reward is constant, episode length equals max ticks, all actions are NOOP (action 0).

Cause: The death penalty is too strong relative to exploration rewards. The agent learns that NOOP avoids death, achieving a local optimum.

Fix:
```python
# Increase exploration pressure
config.ppo.ent_coef = 0.1  # Much higher initially

# Add movement reward in custom shaper
if velocity_magnitude > 0.1:
    reward += 0.001  # Tiny reward for any movement

# Reduce death penalty for first N epochs
if training_epoch < 100:
    death_penalty_scale = 0.1  # 10% of normal
```

**Failure: Agent explores but never gathers resources**

Symptoms: High entropy, diverse movements, zero milestone achievements.

Cause: The gap between random exploration and the first milestone (e.g., punch tree then collect wood) is too large.

Fix: Add intermediate milestones that bridge the gap:
```python
("near_tree", distance_to_nearest_tree < 3, 0.05),
("facing_tree", facing_tree_block, 0.05),
("attacking_tree", attacking_and_facing_tree, 0.1),
```

**Failure: Training diverges (NaN losses)**

Symptoms: Sudden spike in value loss, NaN in observations or rewards.

Cause: Observation or reward values outside expected range, often from division by zero in normalization or extreme reward spikes.

Fix:
```python
# Clip observations and rewards
obs = np.clip(obs, -config.clip_obs, config.clip_obs)
reward = np.clip(reward, -config.clip_reward, config.clip_reward)

# NaN detection and recovery
if np.any(np.isnan(obs)):
    obs, _ = env.reset()
    print("WARNING: NaN in observations, environment reset")
```

**Failure: Stage 3 stalls after entering Nether**

Symptoms: Agent enters Nether (milestone achieved) but never finds fortress. Episodes time out.

Cause: The distance-to-fortress approach reward (0.001 per block) provides insufficient gradient. The agent wanders randomly.

Fix:
```python
# Increase fortress approach reward 5x
approach_reward = min((prev_dist - curr_dist) * 0.005, 0.1)

# Add directional hint
fortress_dir = state.get("fortress_direction", [0, 0])
movement_alignment = np.dot(movement_vec, fortress_dir)
if movement_alignment > 0.5:
    reward += 0.002  # Reward for moving toward fortress
```

**Failure: Stage 6 agent dies to void immediately**

Symptoms: Episode length is always ~50-100 ticks. Agent walks off the main island.

Cause: The void penalty (-0.1 at y < 10) is applied too late. By the time the agent is at y < 10, it has already fallen off the island and cannot recover.

Fix:
```python
# Add edge proximity penalty at the island boundary
if state.get("y_position", 64) < 50:
    reward -= 0.05  # Penalty for being near edge
if state.get("distance_from_center", 0) > 40:
    reward -= 0.02  # Penalty for being far from center
```

### 6.3 Reward Collapse Diagnosis

Reward collapse occurs when the agent finds a degenerate strategy earning constant reward:

```python
def diagnose_reward_collapse(rewards: list[float], window: int = 100) -> dict:
    """Check for signs of reward collapse."""
    recent = rewards[-window:]
    return {
        "mean": np.mean(recent),
        "std": np.std(recent),
        "min": np.min(recent),
        "max": np.max(recent),
        "is_collapsed": np.std(recent) < 0.01 * abs(np.mean(recent) + 1e-8),
        "is_negative_only": np.max(recent) < 0,
        "is_constant_length": np.std(episode_lengths[-window:]) < 10,
    }

# In training loop
diag = diagnose_reward_collapse(episode_rewards)
if diag["is_collapsed"]:
    print(f"REWARD COLLAPSE: reward={diag['mean']:.3f} +/- {diag['std']:.4f}")
    if diag["is_negative_only"]:
        print("  -> Agent only experiences penalties. Check milestone reachability.")
    if diag["is_constant_length"]:
        print("  -> Constant episode length. Agent is stuck in a loop.")
```

Recovery steps:
1. Increase `ent_coef` by 10x temporarily to break out of the local optimum
2. Reset the learning rate to initial value
3. Add a novelty bonus for visiting new states
4. Check that the environment isn't stuck in a broken state (e.g., impossible milestone condition)

### 6.4 Episode Length Patterns

Episode length is a powerful diagnostic signal. Plot distributions, not just means:

```
Stage 1 healthy progression:
  500 -> 2000 -> 6000 -> 3000
  (dies fast) (survives) (explores to max time) (completes efficiently)

Stage 6 healthy:
  100 -> 1000 -> 5000 -> 3000
  (void death) (survives spawn) (fights dragon) (efficient kill)

Stage 6 stuck (void deaths):
  100 -> 100 -> 100 -> 100
  (never learns island navigation)

Stage 3 stuck (no fortress):
  36000 -> 36000 -> 36000
  (always times out wandering the Nether)
```

A bimodal distribution with peaks at ~100 and the max episode length indicates the agent either dies immediately or times out with no intermediate skill level. This signals a missing curriculum step between "survive" and "complete objective."

### 6.5 Debugging Script

A complete diagnostic that runs random episodes and checks environment health:

```python
#!/usr/bin/env python3
"""Diagnose training issues by running random episodes and checking stats."""
import numpy as np
from minecraft_sim.env import make
from minecraft_sim.reward_shaping import create_reward_shaper
from minecraft_sim.actions_discrete import NUM_ACTIONS, analyze_action_distribution

def run_diagnostics(stage_id: int = 1, num_episodes: int = 50):
    env = make(seed=42)
    shaper = create_reward_shaper(stage_id)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    milestones_per_episode: list[int] = []
    all_actions: list[int] = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        if hasattr(shaper, 'reset'):
            shaper.reset()

        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = np.random.randint(0, NUM_ACTIONS)
            all_actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)

            # Apply reward shaping
            state = info if isinstance(info, dict) else {}
            shaped = shaper(state)
            total_reward += reward + shaped
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if hasattr(shaper, 'stats'):
            milestones_per_episode.append(len(shaper.stats.milestones_achieved))

    print(f"\n{'='*60}")
    print(f"Stage {stage_id} Diagnostics ({num_episodes} random episodes)")
    print(f"{'='*60}")
    print(f"Reward:     {np.mean(episode_rewards):+.3f} +/- {np.std(episode_rewards):.3f}")
    print(f"            min={np.min(episode_rewards):.3f}, max={np.max(episode_rewards):.3f}")
    print(f"Length:     {np.mean(episode_lengths):.0f} +/- {np.std(episode_lengths):.0f}")
    print(f"Milestones: {np.mean(milestones_per_episode):.1f} per episode")

    # Action distribution check
    action_stats = analyze_action_distribution(np.array(all_actions))
    print(f"\nAction groups: {action_stats['percentages']}")

    # Issue detection
    issues = []
    if np.std(episode_rewards) < 0.01:
        issues.append("Near-zero reward variance. Reward shaping may be broken.")
    if np.mean(episode_lengths) >= 35000:
        issues.append("All episodes timing out. Check termination conditions.")
    if np.mean(milestones_per_episode) == 0:
        issues.append("No milestones hit by random policy. Early milestones too hard.")
    if np.std(episode_lengths) < 50:
        issues.append("Constant episode length. Environment may be stuck.")

    if issues:
        print("\nWARNINGS:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo issues detected. Environment appears healthy.")

    env.close()

if __name__ == "__main__":
    for stage in range(1, 7):
        run_diagnostics(stage_id=stage)
```

### 6.6 Monitoring Curriculum Health

Beyond individual episode metrics, track the curriculum advancement rate:

```python
def log_curriculum_health(manager: VecCurriculumManager, step: int, writer):
    """Log comprehensive curriculum metrics."""
    stats = manager.get_stats()

    # Stage distribution histogram
    for stage, count in stats["stage_distribution"].items():
        writer.add_scalar(f"curriculum/envs_at_stage_{stage}", count, step)

    # Advancement velocity (advancements per 1000 episodes)
    recent = manager.get_recent_advancements(n=50)
    if len(recent) >= 2:
        time_span = recent[-1].timestamp - recent[0].timestamp
        if time_span > 0:
            velocity = len(recent) / time_span * 1000
            writer.add_scalar("curriculum/advancement_velocity", velocity, step)

    # Stagnation detection
    for stage_id, stage_stat in stats["stage_stats"].items():
        if stage_stat["episodes"] > 500 and stage_stat["success_rate"] < 0.1:
            print(f"STAGNATION: Stage {stage_id} has {stage_stat['episodes']} episodes "
                  f"but only {stage_stat['success_rate']:.1%} success rate")
```

---

## Summary

The key to successful training on this simulator is understanding the interplay between observation normalization, reward shaping, and curriculum progression. The system is designed so that each stage builds on skills learned in previous stages, and the reward structure provides dense feedback at every learning phase.

Start with Stage 1 alone. Verify that milestones are being hit and episode reward is increasing. Then enable curriculum advancement. Monitor entropy and episode length distributions to catch failure modes early. When in doubt, increase exploration pressure (higher `ent_coef`) and check that the reward shaper's milestone chain has no unreachable gaps.

The `VecCurriculumManager` enables heterogeneous training where fast-learning environments can advance while struggling ones remain at earlier stages. This naturally allocates more training time to harder stages without manual intervention. Combined with per-environment reward shapers and stage overrides for advancement criteria, the system provides a complete framework for training from wood-punching to dragon-slaying.
