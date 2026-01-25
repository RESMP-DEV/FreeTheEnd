# Shader Reference

Complete reference for all 79 compute shaders and 5 shared GLSL includes in the Minecraft 1.8.9 GPU simulation backend. All shaders target Vulkan 1.0 compute via GLSL 450 and are compiled to SPIR-V for dispatch.

## Shader Groups

### Core Simulation (9 shaders)

These shaders implement the primary step/reset/observe loop for parallel RL training across N environments.

| # | Shader | Workgroup | Purpose | Input | Output |
|---|--------|-----------|---------|-------|--------|
| 1 | `batch_step` | 64 | Single-dispatch full game tick (action decode, physics, game tick, reward, obs encode) | actions, state buffer | next_state, obs, reward, done |
| 2 | `batch_reset` | 256 | Reset environments to initial state (world, inventory, stats) | seed, reset_flags | initial_state |
| 3 | `batch_actions` | 256 | Batched action processing with physics and combat for all environments | actions, num_envs | state updates, rewards |
| 4 | `game_tick` | 256 | Master game tick with all subsystems (mobs, blocks, fluids, time) | state buffer | updated state |
| 5 | `game_tick_mvk` | 64 | MoltenVK-compatible game tick, single descriptor set, multi-stage via push constants | stage, state | updated state |
| 6 | `action_decoder` | 64 | Decode 32-action discrete space into InputState movement/look/action | int actions[N] | InputState[N] |
| 7 | `observation_encoder` | 256 | Encode game state into 256-float normalized observation vector for RL agent | game state (6 buffers) | float obs[256] |
| 8 | `reward_computation` | 64 | Compute dense rewards for all 6 curriculum stages with milestone bonuses | state, curriculum_stage | float rewards[N] |
| 9 | `time_cycle` | 256 | Day/night cycle (24000 tick), light levels, mob spawning time rules | tick | time_of_day, light |

### Dragon Fight (11 shaders)

Shaders implementing the Ender Dragon boss fight for the final speedrun stage (Stage 6).

| # | Shader | Workgroup | Purpose |
|---|--------|-----------|---------|
| 10 | `dragon_fight_mvk` | 64 | Full dragon fight (MoltenVK-compatible): physics + AI + combat + obs in single dispatch |
| 11 | `dragon_fight_optimized` | 64 | FP16-optimized dragon fight (<50us/step for 64 envs), shared memory caching |
| 12 | `dragon_ai` | 1 | Dragon AI movement/targeting: circling, strafing, landing, perching, dying phases |
| 13 | `dragon_ai_full` | 64 | Enhanced dragon AI state machine with authentic MC 1.8.9 phase transitions (batched) |
| 14 | `dragon_combat` | 64 | Player damage to dragon, crystal healing, multi-hitbox detection (head 4x/body 1x) |
| 15 | `dragon_knockback` | 64 | Dragon attack knockback physics + void death prevention (wing swipe, charge, breath) |
| 16 | `dragon_breath` | 64 | Lingering damage cloud from perch breath attack (3 dmg/s, 15s duration) |
| 17 | `dragon_death` | 64 | Death sequence: fly to fountain, XP drop (12000), egg spawn, exit portal activation |
| 18 | `crystal_tick` | 10 | End crystal healing beams and rotation animation |
| 19 | `crystal_combat` | 64 | Crystal destruction mechanics, caged crystal handling, observation extraction |
| 20 | `bed_explosion` | 64 | Bed explosion in End/Nether (power 5), one-cycle dragon kill strategy (4x head damage) |

### World Generation (17 shaders)

Shaders for procedural terrain, structures, and biome generation matching MC 1.8.9 algorithms.

| # | Shader | Workgroup | Stage | Description |
|---|--------|-----------|-------|-------------|
| 21 | `overworld_gen` | 4x64x4 | 1-2 | Full overworld: biomes, caves, ores, decorations, villages, LRU chunk cache |
| 22 | `overworld_terrain` | 4x4x4 | 1-2 | Biome-aware heightmap generation with Perlin noise for overworld chunks |
| 23 | `biome_gen` | 16x16x1 | 1-2 | Biome determination from temperature + humidity noise (ocean, plains, forest, etc.) |
| 24 | `cave_generation` | 4x4x4 | 1-2 | Worm caves, ravines, mineshafts, ore distribution (MC 1.8.9 accurate) |
| 25 | `ravine_carver` | 16x1x16 | 1-2 | V-shaped canyon carving: long, narrow, deep cuts exposing ore veins |
| 26 | `decoration` | 16x1x16 | 1-2 | Surface decoration: trees, grass, flowers, cactus (biome-aware) |
| 27 | `nether_gen` | 8x8x8 | 3 | Nether terrain: netherrack, soul sand, lava ocean (y=31), glowstone clusters |
| 28 | `nether_terrain` | 8x8x8 | 3 | Hellish landscape with large caverns, bedrock ceiling at y=127 |
| 29 | `fortress_gen` | 4x4x4 | 3 | Nether fortress placement (432x432 grid), blaze spawners, nether wart rooms |
| 30 | `fortress_structure` | 64 | 3 | BFS piece-expansion fortress building with guaranteed blaze/wart rooms |
| 31 | `stronghold_gen` | 8x8x8 | 5 | Stronghold generation: portal room, libraries, corridors (ring at 1408-2688 blocks) |
| 32 | `find_strongholds` | 256 | 5 | Compute stronghold positions from seed using Java Random LCG (3 in ring 1) |
| 33 | `village_gen` | 64 | 1-2 | Simplified village: houses, farms, chests with loot (plains/savanna) |
| 34 | `end_terrain` | 4x4x4 | 6 | End dimension: main island (~100 block radius), obsidian pillars, crystal cages |
| 35 | `end_fountain` | 8x8x8 | 6 | Bedrock fountain structure with dragon egg perch and exit portal management |
| 36 | `end_spawn_platform` | 5x4x5 | 6 | Obsidian platform at (100,48,0) with 5x3x5 air clearance on End entry |
| 37 | `end_spawn` | 64 | 6 | End dimension spawn: platform generation, invulnerability, safety mechanics |

### Player Mechanics (5 shaders)

Shaders handling player state, survival, and interaction systems.

| # | Shader | Workgroup | Description |
|---|--------|-----------|-------------|
| 38 | `health_regen` | 64 | Natural health regeneration when food >= 18 (fast regen at 20), respects difficulty |
| 39 | `hunger_tick` | 64 | Exhaustion accumulation from actions, saturation/food depletion, starvation damage |
| 40 | `experience` | 64 | XP orb collection, total XP tracking, level computation (0-32+ formula) |
| 41 | `status_effects` | 64 | Effect duration tick-down and periodic application (regen, poison, wither, speed) |
| 42 | `enchantment_effects` | 64 | Looting drop bonus (+1/level, max 3), Fire Aspect burning, Sharpness/Smite damage |

### Inventory and Crafting (5 shaders)

| # | Shader | Workgroup | Description |
|---|--------|-----------|-------------|
| 43 | `inventory_ops` | 64 | Move, swap, drop, use, equip, split operations on inventory slots |
| 44 | `inventory_full` | 64 | Complete 42-slot inventory system (hotbar 0-8, main 9-35, armor 36-39, offhand 40, cursor 41) |
| 45 | `crafting` | 64 | Recipe validation and inventory transformation (47 speedrun-essential recipes) |
| 46 | `quick_craft` | 64 | Context-sensitive auto-craft: picks optimal speedrun recipe based on progression |
| 47 | `furnace_tick` | 64 | Furnace smelting: fuel consumption, 200-tick progress, output stacking |

### Block Interaction (6 shaders)

| # | Shader | Workgroup | Description |
|---|--------|-----------|-------------|
| 48 | `block_breaking` | 64 | Block mining with hardness table (256 entries), break progress accumulation |
| 49 | `block_breaking_full` | 64 | Enhanced block breaking with full tool efficiency system (8 types x 5 materials) |
| 50 | `block_placing` | 64 | Block placement with face detection, rotation, and placement rules |
| 51 | `block_updates` | 64 | Queued world modifications: breaking, placing, physics updates (sand/gravel) |
| 52 | `fire_tick` | 64 | Fire spread on flammable blocks (250/250 base chance), burning, extinguishing |
| 53 | `fluid_mechanics` | 256 | Water/lava flow (7/3 blocks), infinite sources, cobblestone/obsidian, buckets |

### Mob AI (14 shaders)

| # | Shader | Workgroup | Mob Type | Description |
|---|--------|-----------|----------|-------------|
| 54 | `mob_ai_base` | 64 | All | Base AI framework: idle, wander, chase, attack, flee states; gravity 0.08 |
| 55 | `mob_ai_enderman` | 64 | Enderman | Teleportation (32 block range, 10 tick cooldown), stare-triggered aggro |
| 56 | `mob_ai_enderman_full` | 64 | Enderman | Full MC 1.8.9: 40 HP, 7 damage, teleport combos, block pickup, pearl drops |
| 57 | `mob_ai_ghast` | 64 | Ghast | Fireball charging (20 ticks), hover mechanics, retreat on close approach |
| 58 | `mob_ai_blaze` | 64 | Blaze | Fireball volleys (3 burst, 100 tick cooldown) + melee, water damage 5.0 |
| 59 | `mob_ai_creeper` | 64 | Creeper | Fuse countdown (30 ticks), silent approach, charged 2x, 3 block explosion |
| 60 | `mob_ai_silverfish` | 64 | Silverfish | Swarm behavior, call reinforcements (21 block range), hide in stone blocks |
| 61 | `mob_ai_overworld_hostile` | 64 | Multiple | Combined zombie/skeleton/spider/creeper behavior with per-type AI |
| 62 | `mob_targeting` | 64 | All | Target selection: detection ranges (40-48 blocks), line-of-sight (64 step raycast) |
| 63 | `mob_combat` | 64 | All | Attack execution: armor 4%/point, protection EPF 4%, resistance 20%/level |
| 64 | `mob_spawning` | 64 | All | Spawn mechanics: mob cap, light levels, biome restrictions, spawn weights |
| 65 | `mob_spawning_overworld` | 64 | Overworld | Hostile cap 70, spawn radius 24-128 blocks, despawn 128, light <= 7 |
| 66 | `enderman_spawning` | 64 | Enderman | Pearl farming: night-only overworld (10%), End 100%, desert +20% bonus |
| 67 | `spawner_tick` | 64 | Spawner | Spawner block: 16 block activation, 4s cycle (80 ticks), max 4 mobs |

### Physics and Projectiles (5 shaders)

| # | Shader | Workgroup | Description |
|---|--------|-----------|-------------|
| 68 | `aabb_ops` | 256 | AABB construction, offset, expansion, intersection tests for entity collision |
| 69 | `item_physics` | 64 | Dropped item entities: gravity 0.04, pickup radius 1.5 blocks, despawn 5 min |
| 70 | `projectile_physics` | 64 | All projectiles: arrow (velocity^2 damage), fireball, pearl, snowball, egg |
| 71 | `ender_pearl` | 64 | Pearl throw: gravity 0.03, drag 0.99, teleport damage 5.0, cooldown 20 ticks |
| 72 | `fireball_tick` | 64 | Blaze fireball: speed 0.9, gravity 0.02, damage 5.0, deflectable at 1.5x speed |

### Portal and Dimension (6 shaders)

| # | Shader | Workgroup | Description |
|---|--------|-----------|-------------|
| 73 | `portal_tick` | 256 | Portal warmup timer (4s), cooldown (1s), transition trigger detection |
| 74 | `portal_creation` | 64 | Nether portal frame validation (2x3 min), flint & steel activation |
| 75 | `dimension_teleport` | 64 | Dimension change: coordinate scaling 8:1 (overworld:nether), position updates |
| 76 | `end_portal` | 256 | End portal frame: 12-frame ring (3x3), eye insertion, activation check |
| 77 | `eye_of_ender` | 64 | Eye of Ender flight toward stronghold: speed 0.2, upward 0.25, 40 tick max |
| 78 | `eye_of_ender_full` | 64 | Complete eye system: rise 15 ticks, fly 25 ticks, 20% shatter, item drop |

### Observation and Detection (1 shader)

| # | Shader | Workgroup | Description |
|---|--------|-----------|-------------|
| 79 | `resource_detection` | 64 | GPU-accelerated raycast cone (45deg, 64 blocks): trees, stone, caves, ores |

## Shared GLSL Includes

Library headers included by compute shaders via `#include` directives (requires `GL_GOOGLE_include_directive`).

| File | Purpose | Key Contents |
|------|---------|--------------|
| `memory_layout.glsl` | Optimized SoA buffer layout for <50us/step | FP16 packed structs, unified buffer offsets, shared memory caching, observation packing |
| `damage.glsl` | MC 1.8.9 damage pipeline | Armor reduction, Protection EPF, Resistance potion, equipment pack/unpack |
| `perlin_noise.glsl` | Noise functions for terrain generation | 2D/3D Perlin, simplex, Worley/cellular, octave FBM, ridged, domain warping |
| `inventory_buffer.glsl` | GPU inventory data structures | InventorySlot/PlayerInventory structs, item IDs, stack limits, durability, enchantment encoding |
| `crafting_recipes.glsl` | Compile-time recipe definitions | 47 speedrun-essential recipes, shaped/shapeless matching, recipe lookup functions |

## Buffer Binding Conventions

Shaders follow consistent binding patterns across the codebase:

| Binding | Typical Usage | Access |
|---------|---------------|--------|
| 0 | Primary state (PlayerBuffer, ChunkData) | read-write |
| 1 | Input/secondary (InputBuffer, HeightmapCache) | readonly |
| 2 | Game entities (DragonBuffer, BiomeData) | read-write |
| 3 | World data (CrystalBuffer, PermTable) | readonly |
| 4 | Game state (GameStateBuffer) | read-write |
| 5 | Observations (ObservationBuffer) | writeonly |
| 6 | Rewards (RewardBuffer) | read-write |
| 7 | Done flags (DoneBuffer) | read-write |

MoltenVK-compatible shaders (`*_mvk`) use a single descriptor set (set 0) with all bindings. Standard Vulkan shaders may use multiple descriptor sets for logical grouping.

## Push Constants

Common push constant layouts used across shaders:

```glsl
// Standard RL pipeline push constants
layout(push_constant) uniform PushConstants {
    uint num_envs;       // Number of parallel environments
    uint tick;           // Current game tick
    uint random_seed;    // Per-dispatch RNG seed
    uint stage;          // Curriculum stage (1-6)
} pc;

// World generation push constants
layout(push_constant) uniform GenParams {
    ivec3 chunk_pos;     // Chunk coordinates
    uint seed;           // World seed
    float base_height;   // Terrain base height
    float height_scale;  // Terrain amplitude
    uint lod_level;      // Level of detail
    uint generation_flags; // Feature enable flags
} params;
```

## Memory Architecture

The simulation uses a unified buffer design for minimal descriptor bindings and coalesced GPU access:

```
UnifiedEnvBuffer (single binding, SoA layout):
  Header:           64 bytes (num_envs, tick, seed, stage)
  PlayerPosition:   N * 16 bytes (FP16 pos + vel)
  PlayerState:      N * 16 bytes (FP16 health/hunger + packed flags)
  DragonPosition:   N * 16 bytes (FP16 pos + vel)
  DragonState:      N * 32 bytes (FP16 health + phase + target)
  CrystalArray:     N * 8 bytes  (alive bitmask)
  InputState:       N * 16 bytes (FP16 movement + action flags)
  Observation:      N * 72 bytes (FP16 obs vector)
  GameState:        N * 8 bytes  (tick, random, stats)
  Rewards:          N * 4 bytes  (float32)
  Dones:            N * 4 bytes  (uint32 flags)

Total: ~192 bytes/env, 12KB for 64 envs (fits L2 cache)
```

Shared memory per workgroup (~2.3KB):
- Player position cache: 1024 bytes
- Dragon position cache: 512 bytes
- Crystal alive masks: 256 bytes
- Accumulated rewards: 256 bytes
- Done flags: 256 bytes

## Shader Compilation

All shaders compile from `.comp` (GLSL 450) to `.spv` (SPIR-V) using the Vulkan SDK's `glslc`.

### Compile all shaders

```bash
cd cpp
./compile_shaders.sh
```

The script searches for `glslc` in PATH, `/opt/homebrew/bin/`, or `$VULKAN_SDK/bin/`.

### Compile a single shader

```bash
glslc -fshader-stage=compute shaders/my_shader.comp -o shaders/my_shader.spv
```

### With include paths (for shaders using shared headers)

```bash
glslc -fshader-stage=compute -I shaders/ shaders/bed_explosion.comp -o shaders/bed_explosion.spv
```

### MoltenVK target (macOS)

```bash
glslc -fshader-stage=compute --target-env=vulkan1.0 shaders/dragon_fight_mvk.comp -o shaders/dragon_fight_mvk.spv
```

### Required Extensions

| Extension | Used By | Purpose |
|-----------|---------|---------|
| `GL_EXT_shader_atomic_float` | batch_step, combat shaders | Atomic float add for reward accumulation |
| `GL_EXT_shader_16bit_storage` | dragon_fight_optimized | FP16 buffer storage |
| `GL_EXT_shader_explicit_arithmetic_types_float16` | dragon_fight_optimized | FP16 compute |
| `GL_EXT_shader_explicit_arithmetic_types_int16` | dragon_fight_optimized | Int16 packed flags |
| `GL_ARB_gpu_shader_int64` | fortress_gen, stronghold_gen, find_strongholds | 64-bit world seeds |
| `GL_GOOGLE_include_directive` | bed_explosion, portal shaders, damage users | `#include` support |

## Adding New Shaders

### Step 1: Create the shader source

Create `shaders/my_shader.comp` with the standard header:

```glsl
#version 450
#extension GL_EXT_shader_atomic_float : enable  // if needed
#extension GL_GOOGLE_include_directive : enable  // if using includes

// Description of what this shader does
// Input/output buffers and their layouts

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint num_envs;
    uint tick;
    uint random_seed;
    uint stage;
} pc;

// Buffer bindings matching C++ side
layout(std430, binding = 0) buffer StateBuffer {
    // Per-environment state
} state;

void main() {
    uint env_id = gl_GlobalInvocationID.x;
    if (env_id >= pc.num_envs) return;

    // Shader logic here
}
```

### Step 2: Compile to SPIR-V

```bash
glslc -fshader-stage=compute shaders/my_shader.comp -o shaders/my_shader.spv
# Or with includes:
glslc -fshader-stage=compute -I shaders/ shaders/my_shader.comp -o shaders/my_shader.spv
```

### Step 3: Register in C++ backend

Add the shader to the pipeline in `mc189_simulator.cpp`:
- Create compute pipeline with `VkComputePipelineCreateInfo`
- Add descriptor set layout matching your buffer bindings
- Record dispatch in the appropriate command buffer

### Step 4: Choose workgroup size

| Size | Use Case |
|------|----------|
| 64 | Standard per-environment processing (1 thread per env) |
| 256 | High-throughput batch operations (portals, AABB, fluids) |
| 4x4x4 | 3D terrain generation (chunk voxels, 64 threads) |
| 8x8x8 | Large 3D volumes (nether, strongholds, 512 threads) |
| 16x16x1 | 2D map operations (biomes, heightmaps, 256 threads) |
| 16x1x16 | Surface decoration and ravine carving (256 threads) |
| 5x4x5 | Special-purpose (spawn platform matches structure, 100 threads) |
| 10 | Per-crystal processing (10 crystals in vanilla dragon fight) |
| 1 | Single-instance logic (dragon AI state machine, serial) |

### Step 5: Validate

```bash
# Compile check
glslc -fshader-stage=compute shaders/my_shader.comp -o shaders/my_shader.spv

# Run shader test suite
uv run pytest tests/test_shaders.py -v -k "my_shader"
```

## Curriculum Stage Mapping

| Stage | Name | Primary Shaders |
|-------|------|----------------|
| 1 | Basic Survival | overworld_gen, block_breaking, crafting, hunger_tick, time_cycle |
| 2 | Resource Gathering | resource_detection, furnace_tick, inventory_ops, cave_generation |
| 3 | Nether Navigation | nether_gen, fortress_gen, portal_creation, mob_ai_blaze, mob_ai_ghast |
| 4 | Enderman Hunting | enderman_spawning, mob_ai_enderman_full, ender_pearl |
| 5 | Stronghold Finding | find_strongholds, eye_of_ender_full, stronghold_gen, end_portal |
| 6 | Dragon Fight | dragon_fight_optimized, crystal_combat, bed_explosion, dragon_death |

## Physics Constants (MC 1.8.9)

Constants used across multiple shaders for authentic simulation:

| Constant | Value | Used In |
|----------|-------|---------|
| Player gravity | 0.08 blocks/tick^2 | mob_ai_base, batch_step |
| Item gravity | 0.04 blocks/tick^2 | item_physics |
| Projectile gravity | 0.03-0.05 blocks/tick^2 | projectile_physics, ender_pearl |
| Terminal velocity | 3.92 blocks/tick | mob_ai_base |
| Ticks per second | 20 | all time-based shaders |
| Day length | 24000 ticks | time_cycle |
| Furnace cook time | 200 ticks | furnace_tick |
| Portal warmup | 80 ticks (4s) | portal_tick |
| Creeper fuse | 30 ticks | mob_ai_creeper |
| Dragon XP | 12000 | dragon_death |
| Bed explosion power | 5.0 | bed_explosion |
