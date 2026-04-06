/*
 * Optimized Memory Layout for Minecraft RL Simulation
 * Target: < 50us per step for 64 environments
 *
 * Optimization strategies:
 * 1. COALESCED ACCESS: Struct-of-Arrays (SoA) layout for parallel env access
 * 2. SHARED MEMORY: Cache hot data per workgroup (player/dragon state)
 * 3. MINIMAL BINDINGS: Single buffer with offsets, packed layouts
 * 4. HALF PRECISION: FP16 for observations, positions, velocities
 * 5. BIT PACKING: Flags, phases, and small integers in uint16
 *
 * Memory layout: All environments are interleaved for coalesced access
 * - Thread i accesses env i, stride = num_envs
 * - Adjacent threads access adjacent memory addresses
 */

#ifndef MEMORY_LAYOUT_GLSL
#define MEMORY_LAYOUT_GLSL

// =============================================================================
// FP16 PACK/UNPACK HELPERS (standard GLSL lacks packHalf4x16/unpackHalf4x16)
// =============================================================================

f16vec4 unpackHalf4x16(uvec2 v) {
    vec2 lo = unpackHalf2x16(v.x);
    vec2 hi = unpackHalf2x16(v.y);
    return f16vec4(float16_t(lo.x), float16_t(lo.y), float16_t(hi.x), float16_t(hi.y));
}

uvec2 packHalf4x16(f16vec4 v) {
    return uvec2(
        packHalf2x16(vec2(float(v.x), float(v.y))),
        packHalf2x16(vec2(float(v.z), float(v.w)))
    );
}

// =============================================================================
// CONSTANTS
// =============================================================================

const uint MAX_ENVS = 256;
const uint WORKGROUP_SIZE = 64;
const uint NUM_CRYSTALS = 10;

// Shared memory sizes (in 16-bit units)
const uint SHARED_PLAYER_SIZE = 32;   // 64 bytes per player
const uint SHARED_DRAGON_SIZE = 16;   // 32 bytes per dragon
const uint SHARED_FLAGS_SIZE = 4;     // 8 bytes for packed flags

// =============================================================================
// PACKED STRUCTURES - SoA Layout for Coalesced Access
// =============================================================================

/*
 * PlayerStateSoA: Struct-of-Arrays for 64 players
 * Each array is contiguous: [env0, env1, ..., env63]
 * Thread i reads position[i], velocity[i], etc. - coalesced!
 *
 * Size per env: 64 bytes (32 x float16 + packed flags)
 * Total for 64 envs: 4KB - fits in L1 cache
 */
struct PlayerPositionPacked {
    f16vec4 pos_vel_xy;   // [pos.x, pos.y, vel.x, vel.y]
    f16vec4 pos_vel_z_yp; // [pos.z, vel.z, yaw, pitch]
};

struct PlayerStatePacked {
    f16vec4 health_hunger;  // [health, hunger, saturation, exhaustion]
    uint16_t flags;         // bit0=ground, bit1=sprint, bit2=sneak, bit3=dead, bit4=won
    uint16_t timers;        // [invincibility:8, attack_cd:8]
    uint16_t weapon_arrows; // [weapon_slot:4, arrows:12]
    float16_t arrow_charge;
};

/*
 * DragonStateSoA: Packed dragon state
 * Size per env: 32 bytes
 */
struct DragonPositionPacked {
    f16vec4 pos_vel_xy;   // [pos.x, pos.y, vel.x, vel.y]
    f16vec4 pos_vel_z_yp; // [pos.z, vel.z, yaw, pitch]
};

struct DragonStatePacked {
    float16_t health;
    uint16_t phase_timer;   // [phase:4, timer:12]
    uint16_t target_breath; // [target_pillar:4, breath_timer:12]
    uint16_t perch_attack;  // [perch_timer:12, attack_cd:4]
    f16vec4 target_pos;     // target position + circle_angle
};

/*
 * CrystalStatePacked: 10 crystals packed into 32 bytes per env
 * Each crystal: 3 bytes (pos index:12, alive:1, reserved:11)
 * We precompute pillar positions, store only index + alive flag
 */
struct CrystalArrayPacked {
    uint crystals_alive_mask;  // bit i = crystal i alive
    uint reserved;
};

/*
 * InputStatePacked: Compact input representation
 * Size per env: 16 bytes
 */
struct InputStatePacked {
    f16vec4 movement_look;  // [move.x, move.y, move.z, look_delta_x]
    float16_t look_delta_y;
    uint16_t action_flags;  // [action:4, flags:12]
    uint16_t action_data;
    uint16_t reserved;
};

/*
 * ObservationPacked: FP16 observations
 * Reduced from 48 floats (192 bytes) to 48 float16 (96 bytes)
 * Further packed where possible
 */
struct ObservationPacked {
    // Player (16 x fp16 = 32 bytes)
    f16vec4 player_pos_vel;      // [pos.x/100, pos.y/50, pos.z/100, vel_magnitude]
    f16vec4 player_orient_state; // [yaw/360, pitch/180, health/20, hunger/20]
    f16vec4 player_flags_weapon; // [on_ground, attack_ready, weapon/2, arrows/64]
    f16vec4 player_reserved;     // [arrow_charge, invincible, dead, won]

    // Dragon (12 x fp16 = 24 bytes)
    f16vec4 dragon_pos_health;   // [x/100, y/50, z/100, health/200]
    f16vec4 dragon_dir_phase;    // [dir_x, dir_z, dist/150, phase/6]
    f16vec4 dragon_flags;        // [can_hit, attacking, vel_magnitude, reserved]

    // Environment (8 x fp16 = 16 bytes)
    f16vec4 env_crystals;        // [crystals_remaining/10, nearest_dist/100, dir_x, dir_z]
    f16vec4 env_progress;        // [portal_active, portal_dist/100, time/24000, total_damage/200]
};

/*
 * GameStatePacked: Minimal game state tracking
 */
struct GameStatePacked {
    uint tick_random;       // [tick:24, random_high:8]
    uint random_low;        // random seed low bits
    uint stats_packed;      // [crystals_destroyed:4, dragon_hits:12, player_deaths:8, reserved:8]
    float16_t best_damage;
    uint16_t flags;         // game flags
};

// =============================================================================
// UNIFIED BUFFER LAYOUT
// =============================================================================

/*
 * Single buffer contains all per-environment data in SoA format.
 * This minimizes descriptor set bindings and enables single buffer
 * argument for compute dispatch.
 *
 * Layout (for N environments):
 *   Offset 0:                    PlayerPositionPacked[N]
 *   Offset N*16:                 PlayerStatePacked[N]
 *   Offset N*32:                 DragonPositionPacked[N]
 *   Offset N*48:                 DragonStatePacked[N]
 *   Offset N*80:                 CrystalArrayPacked[N]
 *   Offset N*88:                 InputStatePacked[N]
 *   Offset N*104:                ObservationPacked[N]
 *   Offset N*176:                GameStatePacked[N]
 *   Offset N*184:                float rewards[N]
 *   Offset N*188:                uint dones[N]
 *
 * Total per env: ~192 bytes
 * Total for 64 envs: ~12KB (fits in L2 cache)
 */

// Buffer binding with explicit offsets
layout(set = 0, binding = 0) buffer UnifiedEnvBuffer {
    // Header (64 bytes, cache line aligned)
    uint num_envs;
    uint current_tick;
    uint base_random_seed;
    uint stage;
    uvec4 reserved_header[3];

    // Data arrays start at offset 64
    // Access via compute_offset() functions below
    uint data[];
};

// =============================================================================
// OFFSET COMPUTATION - Inline for Performance
// =============================================================================

/*
 * Compute byte offset for player position of environment env_id
 * Layout ensures adjacent threads access adjacent memory
 */
uint player_position_offset(uint env_id, uint num_envs) {
    return 64 + env_id * 16;  // 16 bytes per PlayerPositionPacked
}

uint player_state_offset(uint env_id, uint num_envs) {
    return 64 + num_envs * 16 + env_id * 16;
}

uint dragon_position_offset(uint env_id, uint num_envs) {
    return 64 + num_envs * 32 + env_id * 16;
}

uint dragon_state_offset(uint env_id, uint num_envs) {
    return 64 + num_envs * 48 + env_id * 32;
}

uint crystal_offset(uint env_id, uint num_envs) {
    return 64 + num_envs * 80 + env_id * 8;
}

uint input_offset(uint env_id, uint num_envs) {
    return 64 + num_envs * 88 + env_id * 16;
}

uint observation_offset(uint env_id, uint num_envs) {
    return 64 + num_envs * 104 + env_id * 72;  // ObservationPacked = 72 bytes
}

uint game_state_offset(uint env_id, uint num_envs) {
    return 64 + num_envs * 176 + env_id * 8;
}

uint reward_offset(uint env_id, uint num_envs) {
    return 64 + num_envs * 184 + env_id * 4;
}

uint done_offset(uint env_id, uint num_envs) {
    return 64 + num_envs * 188 + env_id * 4;
}

// =============================================================================
// SHARED MEMORY DECLARATIONS
// =============================================================================

/*
 * Shared memory for workgroup-local caching
 * 64 threads per workgroup, each handles 1 environment
 *
 * Cache frequently accessed data during multi-stage processing:
 * - Player position/velocity (modified by physics)
 * - Dragon position (read by all combat calculations)
 * - Crystal alive mask (read by targeting)
 */

// Player position cache: 64 envs * 16 bytes = 1024 bytes
shared f16vec4 shared_player_pos_vel[WORKGROUP_SIZE * 2];

// Dragon position cache: 64 envs * 8 bytes = 512 bytes
shared f16vec4 shared_dragon_pos[WORKGROUP_SIZE];

// Crystal masks: 64 envs * 4 bytes = 256 bytes
shared uint shared_crystal_mask[WORKGROUP_SIZE];

// Accumulated rewards: 64 envs * 4 bytes = 256 bytes
shared float shared_rewards[WORKGROUP_SIZE];

// Done flags: 64 envs * 4 bytes = 256 bytes
shared uint shared_dones[WORKGROUP_SIZE];

// Total shared memory: ~2.3KB (well under 48KB limit)

// =============================================================================
// UTILITY FUNCTIONS - FP16/FP32 CONVERSION
// =============================================================================

// Convert fp32 vec3 to two f16vec4 (pos_vel format)
void pack_position_velocity(vec3 pos, vec3 vel, float yaw, float pitch,
                            out f16vec4 pos_vel_xy, out f16vec4 pos_vel_z_yp) {
    pos_vel_xy = f16vec4(
        float16_t(pos.x),
        float16_t(pos.y),
        float16_t(vel.x),
        float16_t(vel.y)
    );
    pos_vel_z_yp = f16vec4(
        float16_t(pos.z),
        float16_t(vel.z),
        float16_t(yaw),
        float16_t(pitch)
    );
}

// Unpack position/velocity from fp16
void unpack_position_velocity(f16vec4 pos_vel_xy, f16vec4 pos_vel_z_yp,
                              out vec3 pos, out vec3 vel, out float yaw, out float pitch) {
    pos = vec3(float(pos_vel_xy.x), float(pos_vel_xy.y), float(pos_vel_z_yp.x));
    vel = vec3(float(pos_vel_xy.z), float(pos_vel_xy.w), float(pos_vel_z_yp.y));
    yaw = float(pos_vel_z_yp.z);
    pitch = float(pos_vel_z_yp.w);
}

// Pack player flags into uint16
uint16_t pack_player_flags(bool on_ground, bool sprinting, bool sneaking,
                           bool dead, bool won, bool invincible) {
    uint f = 0u;
    if (on_ground)  f |= 1u;
    if (sprinting)  f |= 2u;
    if (sneaking)   f |= 4u;
    if (dead)       f |= 8u;
    if (won)        f |= 16u;
    if (invincible) f |= 32u;
    return uint16_t(f);
}

// Unpack player flags from uint16
void unpack_player_flags(uint16_t flags, out bool on_ground, out bool sprinting,
                         out bool sneaking, out bool dead, out bool won, out bool invincible) {
    on_ground  = (flags & 1u) != 0u;
    sprinting  = (flags & 2u) != 0u;
    sneaking   = (flags & 4u) != 0u;
    dead       = (flags & 8u) != 0u;
    won        = (flags & 16u) != 0u;
    invincible = (flags & 32u) != 0u;
}

// Pack timers: invincibility (8 bits) + attack_cd (8 bits)
uint16_t pack_timers(uint invincibility, uint attack_cd) {
    return uint16_t((invincibility & 0xFFu) | ((attack_cd & 0xFFu) << 8));
}

void unpack_timers(uint16_t packed, out uint invincibility, out uint attack_cd) {
    invincibility = uint(packed) & 0xFFu;
    attack_cd = (uint(packed) >> 8) & 0xFFu;
}

// Pack dragon phase_timer: phase (4 bits) + timer (12 bits)
uint16_t pack_phase_timer(uint phase, uint timer) {
    return uint16_t((phase & 0xFu) | ((timer & 0xFFFu) << 4));
}

void unpack_phase_timer(uint16_t packed, out uint phase, out uint timer) {
    phase = uint(packed) & 0xFu;
    timer = (uint(packed) >> 4) & 0xFFFu;
}

// =============================================================================
// BUFFER ACCESS HELPERS
// =============================================================================

/*
 * Load player position from unified buffer into shared memory
 * Call at start of workgroup processing
 */
void load_player_to_shared(uint local_id, uint env_id) {
    uint offset = player_position_offset(env_id, num_envs);
    uint idx = offset / 4;

    // Load 4 uints (16 bytes) = PlayerPositionPacked
    uvec4 raw = uvec4(data[idx], data[idx+1], data[idx+2], data[idx+3]);

    // Store in shared memory (reinterpret as f16vec4)
    shared_player_pos_vel[local_id * 2] = unpackHalf4x16(raw.xy);
    shared_player_pos_vel[local_id * 2 + 1] = unpackHalf4x16(raw.zw);
}

/*
 * Store player position from shared memory back to unified buffer
 * Call at end of workgroup processing
 */
void store_player_from_shared(uint local_id, uint env_id) {
    uint offset = player_position_offset(env_id, num_envs);
    uint idx = offset / 4;

    uvec2 packed0 = packHalf4x16(shared_player_pos_vel[local_id * 2]);
    uvec2 packed1 = packHalf4x16(shared_player_pos_vel[local_id * 2 + 1]);

    data[idx] = packed0.x;
    data[idx+1] = packed0.y;
    data[idx+2] = packed1.x;
    data[idx+3] = packed1.y;
}

/*
 * Load dragon position to shared memory
 */
void load_dragon_to_shared(uint local_id, uint env_id) {
    uint offset = dragon_position_offset(env_id, num_envs);
    uint idx = offset / 4;

    uvec2 raw = uvec2(data[idx], data[idx+1]);
    shared_dragon_pos[local_id] = unpackHalf4x16(raw);
}

/*
 * Load crystal mask to shared memory
 */
void load_crystals_to_shared(uint local_id, uint env_id) {
    uint offset = crystal_offset(env_id, num_envs);
    shared_crystal_mask[local_id] = data[offset / 4];
}

// =============================================================================
// OBSERVATION PACKING
// =============================================================================

/*
 * Pack full observation into compact FP16 format
 * All values pre-normalized to [0,1] or [-1,1] range
 */
void pack_observation(
    vec3 player_pos, vec3 player_vel, float yaw, float pitch,
    float health, float hunger, bool on_ground, bool attack_ready,
    uint weapon, uint arrows, float arrow_charge,
    vec3 dragon_pos, vec3 dragon_vel, float dragon_health, uint dragon_phase,
    float dragon_dist, vec3 dragon_dir, bool can_hit, bool attacking,
    uint crystals_remaining, float nearest_crystal_dist, vec3 nearest_crystal_dir,
    bool portal_active, float portal_dist, float time_remaining, float total_damage,
    out ObservationPacked obs
) {
    // Player observation (normalized)
    float vel_mag = length(player_vel);
    obs.player_pos_vel = f16vec4(
        float16_t(player_pos.x / 100.0),
        float16_t((player_pos.y - 64.0) / 50.0),
        float16_t(player_pos.z / 100.0),
        float16_t(vel_mag / 2.0)
    );

    obs.player_orient_state = f16vec4(
        float16_t(yaw / 360.0),
        float16_t((pitch + 90.0) / 180.0),
        float16_t(health / 20.0),
        float16_t(hunger / 20.0)
    );

    obs.player_flags_weapon = f16vec4(
        float16_t(on_ground ? 1.0 : 0.0),
        float16_t(attack_ready ? 1.0 : 0.0),
        float16_t(float(weapon) / 2.0),
        float16_t(float(arrows) / 64.0)
    );

    obs.player_reserved = f16vec4(
        float16_t(arrow_charge),
        float16_t(0.0),  // invincible placeholder
        float16_t(0.0),  // dead placeholder
        float16_t(0.0)   // won placeholder
    );

    // Dragon observation
    obs.dragon_pos_health = f16vec4(
        float16_t(dragon_pos.x / 100.0),
        float16_t((dragon_pos.y - 64.0) / 50.0),
        float16_t(dragon_pos.z / 100.0),
        float16_t(dragon_health / 200.0)
    );

    obs.dragon_dir_phase = f16vec4(
        float16_t(dragon_dir.x),
        float16_t(dragon_dir.z),
        float16_t(dragon_dist / 150.0),
        float16_t(float(dragon_phase) / 6.0)
    );

    float dragon_vel_mag = length(dragon_vel);
    obs.dragon_flags = f16vec4(
        float16_t(can_hit ? 1.0 : 0.0),
        float16_t(attacking ? 1.0 : 0.0),
        float16_t(dragon_vel_mag / 3.0),
        float16_t(0.0)
    );

    // Environment observation
    obs.env_crystals = f16vec4(
        float16_t(float(crystals_remaining) / 10.0),
        float16_t(nearest_crystal_dist / 100.0),
        float16_t(nearest_crystal_dir.x),
        float16_t(nearest_crystal_dir.z)
    );

    obs.env_progress = f16vec4(
        float16_t(portal_active ? 1.0 : 0.0),
        float16_t(portal_dist / 100.0),
        float16_t(time_remaining),
        float16_t(total_damage / 200.0)
    );
}

/*
 * Store packed observation to unified buffer
 */
void store_observation(uint env_id, ObservationPacked obs) {
    uint offset = observation_offset(env_id, num_envs);
    uint idx = offset / 4;

    // Pack 18 f16vec4 = 72 bytes = 18 uints
    uvec2 p0 = packHalf4x16(obs.player_pos_vel);
    uvec2 p1 = packHalf4x16(obs.player_orient_state);
    uvec2 p2 = packHalf4x16(obs.player_flags_weapon);
    uvec2 p3 = packHalf4x16(obs.player_reserved);
    uvec2 d0 = packHalf4x16(obs.dragon_pos_health);
    uvec2 d1 = packHalf4x16(obs.dragon_dir_phase);
    uvec2 d2 = packHalf4x16(obs.dragon_flags);
    uvec2 e0 = packHalf4x16(obs.env_crystals);
    uvec2 e1 = packHalf4x16(obs.env_progress);

    data[idx+0] = p0.x; data[idx+1] = p0.y;
    data[idx+2] = p1.x; data[idx+3] = p1.y;
    data[idx+4] = p2.x; data[idx+5] = p2.y;
    data[idx+6] = p3.x; data[idx+7] = p3.y;
    data[idx+8] = d0.x; data[idx+9] = d0.y;
    data[idx+10] = d1.x; data[idx+11] = d1.y;
    data[idx+12] = d2.x; data[idx+13] = d2.y;
    data[idx+14] = e0.x; data[idx+15] = e0.y;
    data[idx+16] = e1.x; data[idx+17] = e1.y;
}

// =============================================================================
// REWARD ACCUMULATION
// =============================================================================

/*
 * Initialize shared reward/done state
 */
void init_shared_reward_done(uint local_id) {
    shared_rewards[local_id] = 0.0;
    shared_dones[local_id] = 0u;
}

/*
 * Accumulate reward in shared memory (faster than global atomics)
 */
void add_reward(uint local_id, float reward) {
    shared_rewards[local_id] += reward;
}

/*
 * Set done flag in shared memory
 */
void set_done(uint local_id) {
    shared_dones[local_id] = 1u;
}

/*
 * Flush shared rewards/dones to global memory
 */
void flush_reward_done(uint local_id, uint env_id) {
    uint reward_idx = reward_offset(env_id, num_envs) / 4;
    uint done_idx = done_offset(env_id, num_envs) / 4;

    // Atomic add for reward (multiple stages may contribute)
    atomicAdd(data[reward_idx], floatBitsToUint(shared_rewards[local_id]));

    // Done is OR-ed (any stage can mark done)
    if (shared_dones[local_id] != 0u) {
        atomicOr(data[done_idx], 1u);
    }
}

// =============================================================================
// PRECOMPUTED CONSTANTS (Store in constant buffer or push constants)
// =============================================================================

// Pillar positions (precomputed for crystal locations)
const vec3 PILLAR_POSITIONS[10] = vec3[10](
    vec3(40.0 * cos(0.0),      77.0, 40.0 * sin(0.0)),
    vec3(55.0 * cos(0.628318), 77.0, 55.0 * sin(0.628318)),
    vec3(70.0 * cos(1.256637), 77.0, 70.0 * sin(1.256637)),
    vec3(40.0 * cos(1.884956), 77.0, 40.0 * sin(1.884956)),
    vec3(55.0 * cos(2.513274), 77.0, 55.0 * sin(2.513274)),
    vec3(70.0 * cos(3.141593), 77.0, 70.0 * sin(3.141593)),
    vec3(40.0 * cos(3.769911), 77.0, 40.0 * sin(3.769911)),
    vec3(55.0 * cos(4.398230), 77.0, 55.0 * sin(4.398230)),
    vec3(70.0 * cos(5.026548), 77.0, 70.0 * sin(5.026548)),
    vec3(40.0 * cos(5.654867), 77.0, 40.0 * sin(5.654867))
);

// Physics constants
const float GRAVITY_FP16 = -0.08;
const float DRAG_AIR_FP16 = 0.98;
const float DRAG_GROUND_FP16 = 0.6;
const float WALK_SPEED_FP16 = 0.1;
const float SPRINT_MULT_FP16 = 1.3;
const float JUMP_VEL_FP16 = 0.42;
const float END_SPAWN_Y = 64.0;
const float END_ISLAND_RADIUS = 100.0;

// =============================================================================
// FAST MATH UTILITIES
// =============================================================================

// Fast inverse square root (for normalization)
float fast_invsqrt(float x) {
    // Use native GLSL inversesqrt which compiles to rsqrt instruction
    return inversesqrt(x);
}

// Fast normalize using rsqrt
vec3 fast_normalize(vec3 v) {
    float len_sq = dot(v, v);
    return v * inversesqrt(max(len_sq, 0.0001));
}

// Fast distance (avoid sqrt when possible)
float fast_distance_sq(vec3 a, vec3 b) {
    vec3 d = a - b;
    return dot(d, d);
}

// =============================================================================
// SYNCHRONIZATION HELPERS
// =============================================================================

/*
 * Workgroup barrier with memory fence
 * Use between stages that share data
 */
void workgroup_sync() {
    memoryBarrierShared();
    barrier();
}

/*
 * Global memory barrier (for buffer writes visible to CPU)
 */
void global_sync() {
    memoryBarrierBuffer();
    barrier();
}

#endif // MEMORY_LAYOUT_GLSL
