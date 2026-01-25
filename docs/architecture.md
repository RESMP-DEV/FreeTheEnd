# Minecraft Simulator Architecture

## Overview

The simulator uses Vulkan compute shaders for GPU-accelerated parallel environment simulation. The architecture is designed for massive throughput (615K+ steps/sec on M4 Max) to enable practical RL training on Minecraft speedruns.

## Current State (January 2026)

### Working
- **End Fight Stage**: `dragon_fight_mvk.comp` provides complete dragon fight simulation
- **Physics**: Player movement, jumping, collision
- **Combat**: Melee attacks, dragon AI, crystal mechanics
- **Observations**: 48-dimensional normalized state vector
- **Actions**: 17 discrete actions

### Not Yet Integrated
- Overworld/Nether/End world generation
- Inventory and crafting systems
- Block breaking and placing
- Portal mechanics
- Multi-stage progression

## Directory Structure

```
FreeTheEnd/
├── cpp/
│   ├── shaders/           # 79 Vulkan compute shaders
│   ├── src/               # C++ simulator code
│   ├── include/           # Header files
│   └── build/             # Build artifacts
├── python/minecraft_sim/  # Python package
│   └── *.py               # Python modules
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Shader Architecture

### Current: Monolithic Dragon Fight

```
dragon_fight_mvk.comp (914 lines)
├── Player physics
├── Dragon AI (6 phases)
├── Crystal mechanics
├── Combat resolution
├── Observation encoding
└── Reward computation
```

### Target: Multi-Shader Pipeline

```
Stage 1 - Overworld Spawn:
  overworld_gen.comp → biome_gen.comp → mob_spawning_overworld.comp
  
Stage 2 - Resource Gathering:
  block_breaking.comp → inventory_ops.comp → crafting.comp
  
Stage 3 - Nether:
  portal_creation.comp → nether_gen.comp → fortress_gen.comp → mob_ai_blaze.comp
  
Stage 4 - Enderman Hunting:
  mob_ai_enderman.comp → enderman_spawning.comp
  
Stage 5 - Stronghold:
  eye_of_ender.comp → find_strongholds.comp → stronghold_gen.comp
  
Stage 6 - End Fight:
  dragon_fight_mvk.comp (current working implementation)
```

## Data Structures

### GPU Buffers (in dragon_fight_mvk.comp)

```glsl
// Per-environment player state
struct Player {
    vec3 position;      // World coordinates
    float yaw;          // Horizontal look angle
    vec3 velocity;      // Movement velocity
    float pitch;        // Vertical look angle
    float health;       // 0-20 HP
    float hunger;       // 0-20 hunger points
    float saturation;   // Hunger saturation
    float exhaustion;   // Exhaustion level
    uint flags;         // State flags (on_ground, sprinting, etc)
    uint invincibility_timer;
    uint attack_cooldown;
    uint weapon_slot;   // Active weapon
    float arrow_charge; // Bow charge level
    uint arrows;        // Arrow count
};

// Per-environment dragon state
struct Dragon {
    vec3 position;
    float yaw;
    vec3 velocity;
    float pitch;
    float health;       // 0-200 HP
    uint phase;         // AI phase (0-6)
    uint phase_timer;
    uint target_pillar;
    vec3 target_pos;
    uint flags;
};

// Input actions decoded from discrete action ID
struct InputState {
    float movement_x;   // Strafe input
    float movement_z;   // Forward/back input
    float look_delta_x; // Yaw change (degrees)
    float look_delta_y; // Pitch change (degrees)
    uint flags;         // Jump, sprint, sneak flags
    uint action_type;   // Attack, use, etc
};
```

### Buffer Bindings

```glsl
layout(set = 0, binding = 0) buffer PlayerBuffer { Player players[]; };
layout(set = 0, binding = 1) buffer InputBuffer { InputState inputs[]; };
layout(set = 0, binding = 2) buffer DragonBuffer { Dragon dragons[]; };
layout(set = 0, binding = 3) buffer CrystalBuffer { Crystal crystals[]; };
layout(set = 0, binding = 4) buffer GameStateBuffer { GameState game_states[]; };
layout(set = 0, binding = 5) buffer ObservationBuffer { Observation observations[]; };
layout(set = 0, binding = 6) buffer RewardBuffer { float rewards[]; };
layout(set = 0, binding = 7) buffer DoneBuffer { uint dones[]; };
```

## C++ Simulator

### MC189Simulator Class

```cpp
class MC189Simulator {
public:
    MC189Simulator(SimulatorConfig config);
    
    void reset(uint32_t env_id = 0xFFFFFFFF, uint64_t seed = 0);
    void step(const std::vector<int32_t>& actions);
    
    std::vector<std::vector<float>> get_observations();
    std::vector<float> get_rewards();
    std::vector<int32_t> get_dones();

private:
    void load_shaders();
    void create_buffers();
    void dispatch_compute();
    
    VulkanContext ctx_;
    SimulatorConfig config_;
    // GPU buffers...
};
```

### Current Limitations

1. **Single shader**: Only loads `dragon_fight.spv`
2. **No world state**: No persistent terrain/chunks
3. **No dimension switching**: Stuck in End dimension
4. **No inventory**: Items not tracked across steps

## Python Interface

### Gymnasium Environment

```python
class DragonFightEnv(gym.Env):
    observation_space = Box(low=0, high=1, shape=(48,), dtype=np.float32)
    action_space = Discrete(17)
    
    def __init__(self, shader_dir=None):
        self._sim = mc189_core.MC189Simulator(config)
    
    def reset(self, seed=None, options=None):
        self._sim.reset(seed=seed or 0)
        return self._get_obs(), {}
    
    def step(self, action):
        self._sim.step([int(action)])
        obs = self._get_obs()
        reward = self._sim.get_rewards()[0]
        done = bool(self._sim.get_dones()[0])
        return obs, reward, done, False, {}
```

### Vectorized Environment

```python
class VecDragonFightEnv:
    def __init__(self, num_envs=64, shader_dir=None):
        config = mc189_core.SimulatorConfig()
        config.num_envs = num_envs
        self._sim = mc189_core.MC189Simulator(config)
    
    def step(self, actions):
        self._sim.step(actions.tolist())
        obs = np.array(self._sim.get_observations())
        rewards = np.array(self._sim.get_rewards())
        dones = np.array(self._sim.get_dones(), dtype=bool)
        return obs, rewards, dones, [{} for _ in range(self.num_envs)]
```

## Integration Targets

### 1. Multi-Shader Dispatch

Modify `mc189_simulator.cpp` to:
- Load multiple `.spv` files
- Select shader based on current dimension/stage
- Dispatch appropriate shader per step

```cpp
void MC189Simulator::step_multi_shader(const std::vector<int32_t>& actions) {
    // Determine which shader each env needs
    for (uint32_t env = 0; env < config_.num_envs; ++env) {
        Dimension dim = get_dimension(env);
        switch (dim) {
            case OVERWORLD: dispatch_shader(overworld_shader_, env); break;
            case NETHER: dispatch_shader(nether_shader_, env); break;
            case END: dispatch_shader(dragon_fight_shader_, env); break;
        }
    }
}
```

### 2. World State Management

Add chunk system for persistent terrain:

```cpp
struct Chunk {
    int32_t x, z;           // Chunk coordinates
    uint16_t blocks[16][256][16];  // Block data
    uint8_t light[16][256][16];    // Light levels
};

class WorldState {
    std::unordered_map<ChunkKey, Chunk> loaded_chunks_;
    void load_chunk(int32_t x, int32_t z);
    void unload_distant_chunks(vec3 player_pos);
};
```

### 3. Stage Transitions

Wire Python curriculum to C++ simulator:

```python
class SpeedrunEnv(gym.Env):
    def __init__(self):
        self.curriculum = CurriculumManager()
        self._current_stage = StageID.BASIC_SURVIVAL
    
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
        if self.curriculum.should_advance(info):
            self._current_stage = self.curriculum.next_stage()
            self._sim.set_stage(self._current_stage)
        
        return obs, reward, done, truncated, info
```

## Stronghold Tracking

### Shader: `stronghold_gen.comp`

The stronghold generation shader implements Minecraft 1.8.9 stronghold placement and structure generation on the GPU. It is dispatched once per environment at world initialization to pre-compute stronghold data that the CPU-side simulation queries during gameplay.

### Generation Algorithm

```
Input:  world_seed (int64), stronghold_index (0-2), chunk_offset (ivec3)
Output: StrongholdData struct (portal room position, frame states, room layout)
        BlockData buffer (chunk-level block placement)
```

Stronghold placement follows Java Edition 1.8.9 rules:

1. Seed an LCG RNG from `world_seed`
2. Compute a base angle from the seed, then offset by `index * 120 degrees`
3. Choose a distance within the ring [1408, 2688] blocks from origin
4. Snap XZ to chunk boundaries (multiples of 16)
5. Choose Y-level in range [10, 40]

### GPU Data Structures

```glsl
layout(std430, binding = 0) buffer StrongholdData {
    ivec3 portal_room_pos;         // World position of portal room center
    int portal_frames[12];          // 1 = has eye, 0 = empty (10% chance each)
    int silverfish_spawner_pos[3];  // Spawner below portal
    int library_count;              // 0-2 libraries
    ivec4 library_positions[2];     // Position + type (normal/tall)
    int corridor_count;             // Up to 64 corridors
    ivec4 corridors[64];            // Start pos + (direction << 16 | length)
    int prison_count;               // Up to 8 cells
    ivec4 prison_cells[8];
    int fountain_count;             // Up to 4 fountains
    ivec4 fountains[4];             // Position + fluid type (water/lava)
    int total_rooms;                // Total generated rooms (max 50)
    int generation_complete;        // Set to 1 when done
};
```

### Room Generation

Rooms are generated iteratively (GLSL has no recursion) using an explicit stack with max depth 7 and up to 3 branches per node. The portal room is always placed first (depth 0). Subsequent rooms are selected probabilistically:

| Room Type | Probability | Max Count |
|-----------|-------------|-----------|
| Library | 5% | 2 |
| Fountain | 5% | 4 |
| Prison cell | 10% | 8 |
| Corridor | 80% | 64 |

### Tracking at Runtime

The Python-side `ProgressTracker` maintains stronghold state per environment:

- `stronghold_found: bool` - set when agent enters stronghold bounds
- `portal_room_found: bool` - set when agent reaches portal room coordinates
- `eyes_placed: int` - frames filled (0-12)
- `portal_activated: bool` - all 12 frames filled

The C++ simulator exposes the `StrongholdData` buffer to Python through `get_stronghold_info(env_id)`, providing the agent's observation space with distance and direction signals for the reward shaper's approach bonus.

## Portal Activation

### Portal Frame Layout

The End portal consists of 12 frames arranged in a 5x5 ring pattern with 3 frames per side:

```
    N N N
  W       E
  W       E
  W       E
    S S S
```

Each frame faces inward toward the portal center. Frames are placed at Y+4 relative to the portal room floor, on a raised stone brick platform at Y+3.

### Frame State Machine

```
EMPTY (block ID 120) → agent places eye → FILLED (block ID 120 | (4 << 8))
```

On world generation, each frame independently has a 10% probability of starting filled (pre-placed eye). The simulation tracks this in `portal_frames[12]` within the `StrongholdData` buffer.

### Activation Conditions

The portal activates when all 12 frames contain an eye of ender. The check occurs each tick when the agent is within the portal room:

```python
# Simplified portal check logic
frames_filled = state["portal_frames_filled"]  # 0-12
if frames_filled == 12:
    state["end_portal_activated"] = True
    # Episode transitions to Stage 6 (Dragon Fight)
```

### Eye Management

Eyes of ender are a consumable resource. The agent starts Stage 5 with 14 ender pearls and 7 blaze rods (yielding 14 blaze powder, enough for 14 eyes). Key constraints:

- Crafting: 1 blaze powder + 1 ender pearl = 1 eye of ender
- Throwing: 20% chance the eye breaks on use
- Minimum needed: 12 minus pre-filled frames (typically 10-11 needed)
- Failure detection: if `eyes_remaining < frames_empty`, episode terminates

## Reward Shaping Architecture

### Overview

The reward shaping system (`reward_shaping.py`) provides per-stage dense reward functions that guide the agent through sub-goals within each curriculum stage. Each stage has a dedicated shaper created via factory function.

### Design Pattern

```python
def create_stageN_reward_shaper() -> Callable[[dict[str, Any]], float]:
    """Closure-based reward shaper with persistent state."""
    given_rewards: set[str] = set()      # Tracks one-time milestones
    prev_state: dict[str, Any] = {}      # Previous tick state for deltas
    stats = RewardStats()                 # Logging/debugging stats

    def shape_reward(state: dict[str, Any]) -> float:
        # 1. Penalties (time, death, damage)
        # 2. Milestone checks (one-time bonuses)
        # 3. Progressive rewards (per-resource deltas)
        # 4. Stage completion bonus
        return reward

    shape_reward.stats = stats
    shape_reward.reset = lambda: ...
    return shape_reward
```

### Reward Signal Categories

Each shaper produces four signal types, applied additively:

| Category | Purpose | Typical Range |
|----------|---------|---------------|
| Penalties | Pressure toward efficiency and survival | -2.0 to -0.0001/tick |
| Milestones | One-time bonuses for key achievements | +0.05 to +1.5 |
| Progressive | Diminishing per-resource increments | +0.005 to +0.03/unit |
| Completion | Terminal bonus for finishing the stage | +2.0 to +5.0 |

### Stage-Specific Shaper Summary

| Stage | Key Signals | Completion Bonus |
|-------|-------------|-----------------|
| 1 (Survival) | Wood/stone gathering, tool crafting, mob kills | +2.0 |
| 2 (Resources) | Iron/diamond/obsidian, depth exploration | +2.0 |
| 3 (Nether) | Portal construction, fortress approach, blaze kills | +2.5 |
| 4 (Pearls) | Enderman kills, pearl/eye accumulation, frame filling | +2.0 |
| 5 (Stronghold) | Triangulation, stronghold approach, portal room, activation | +2.5 |
| 6 (Dragon) | Crystal destruction, dragon damage, perch hits, kill | +5.0 |

### Interaction with Environment Rewards

The reward shaper operates independently from the environment's built-in dense rewards (defined in stage YAML configs). Both are summed to produce the final reward signal:

```
total_reward = env_dense_reward + shaper(state)
```

The env rewards are coarser (fewer signals, higher magnitude) while the shaper provides fine-grained shaping. This separation allows tuning either independently.

### CompositeRewardShaper

For end-to-end training across all stages, the `CompositeRewardShaper` class manages transitions:

```python
composite = CompositeRewardShaper(initial_stage=1)
composite.set_stage(current_stage)    # Switch active shaper
reward = composite.shape_reward(state) # Delegates to current stage
composite.advance_stage()              # Move to next stage
composite.get_stats()                  # Current stage RewardStats
```

### Reward Scaling Across Stages

Later stages have higher completion bonuses to maintain gradient signal despite longer episode horizons:

| Stage | Max Ticks | Death Penalty | Time Penalty/Tick | Completion |
|-------|-----------|---------------|-------------------|------------|
| 1 | 24,000 | -1.0 | -0.0001 | +2.0 |
| 2 | 36,000 | -0.8 | -0.0001 | +2.0 |
| 3 | 48,000 | -1.2 | -0.00012 | +2.5 |
| 4 | 48,000 | -1.0 | -0.00015 | +2.0 |
| 5 | 60,000 | -0.8 | -0.00012 | +2.5 |
| 6 | 36,000 | -2.0 | -0.0002 | +5.0 |

## Observation Encoding

### Current (48 floats)

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 0-2 | player_pos | 0-1 | Position / 1000 |
| 3-5 | player_vel | 0-1 | Velocity (clamped) |
| 6 | yaw | 0-1 | yaw / 360 |
| 7 | pitch | 0-1 | (pitch + 90) / 180 |
| 8 | health | 0-1 | health / 20 |
| 9 | hunger | 0-1 | hunger / 20 |
| 10-15 | flags | 0-1 | Boolean flags |
| 16-23 | dragon | 0-1 | Dragon state |
| 24-31 | combat | 0-1 | Combat info |
| 32-47 | misc | 0-1 | Crystals, etc |

### Target (Extended for full speedrun)

Additional observations needed:
- Inventory contents (hotbar + main)
- Nearby blocks (3D grid around player)
- Visible entities
- Portal locations
- Eye of ender trajectory

## Performance Considerations

### Current Throughput (Apple M4 Max, January 2026)
- 64 envs: ~264,000 steps/sec
- 256 envs: ~615,000 steps/sec
- MineRL comparison: 4,400× speedup
- Bottleneck: GPU compute

## Testing Strategy

### Unit Tests
- Per-action behavior (`test_discrete_actions.py`)
- Physics accuracy
- Combat mechanics

### Integration Tests
- Stage transitions
- Dimension changes
- Full speedrun completion

### Performance Tests
- Throughput benchmarks
- Memory usage
- GPU utilization

## File Reference

### Key Shaders
| File | Lines | Purpose |
|------|-------|---------|
| dragon_fight_mvk.comp | 914 | End fight (working) |
| overworld_gen.comp | 1196 | Overworld terrain |
| nether_gen.comp | 643 | Nether terrain |
| stronghold_gen.comp | 520 | Stronghold structures |
| inventory_full.comp | 975 | Inventory system |
| crafting.comp | 557 | Crafting recipes |

### Key C++ Files
| File | Purpose |
|------|---------|
| mc189_simulator.cpp | Main simulator |
| world_seed_impl.cpp | RNG utilities |

### Key Python Files
| File | Purpose |
|------|---------|
| env.py | DragonFightEnv |
| vec_env.py | Vectorized env |
| curriculum.py | Stage management |
| stage_configs/*.yaml | Stage definitions |
