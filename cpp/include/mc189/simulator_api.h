// mc189/simulator_api.h - Public C API for Minecraft 1.8.9 RL Simulator
// Target: 500K+ steps/second on Apple Silicon via MoltenVK
// Usage: RL training for "Free The End" speedrun (spawn to dragon kill)

#ifndef MC189_SIMULATOR_API_H
#define MC189_SIMULATOR_API_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Library visibility
#ifdef _WIN32
    #ifdef MC189_EXPORTS
        #define MC189_API __declspec(dllexport)
    #else
        #define MC189_API __declspec(dllimport)
    #endif
#else
    #define MC189_API __attribute__((visibility("default")))
#endif

// ============================================================================
// CONSTANTS
// ============================================================================

#define MC189_MAX_BATCH_SIZE      16384   // Max parallel environments
#define MC189_INVENTORY_SIZE      36      // Main inventory slots
#define MC189_ARMOR_SLOTS         4       // Armor slots
#define MC189_HOTBAR_SIZE         9       // Hotbar slots (subset of inventory)
#define MC189_MAX_MOBS            65536   // Max mobs per world
#define MC189_CHUNK_SIZE          16      // Blocks per chunk dimension
#define MC189_CHUNK_HEIGHT        384     // Blocks height (1.18+ style)
#define MC189_MAX_LOADED_CHUNKS   1024    // Max simultaneously loaded chunks
#define MC189_TICKS_PER_SECOND    20      // Minecraft tick rate
#define MC189_MAX_STATUS_EFFECTS  8       // Max simultaneous effects

// Dimensions
typedef enum {
    MC189_DIM_OVERWORLD = 0,
    MC189_DIM_NETHER    = 1,
    MC189_DIM_END       = 2,
} mc189_dimension_t;

// Action types for discrete action space
typedef enum {
    MC189_ACTION_NONE       = 0,
    MC189_ACTION_FORWARD    = 1,
    MC189_ACTION_BACKWARD   = 2,
    MC189_ACTION_LEFT       = 3,
    MC189_ACTION_RIGHT      = 4,
    MC189_ACTION_JUMP       = 5,
    MC189_ACTION_SNEAK      = 6,
    MC189_ACTION_SPRINT     = 7,
    MC189_ACTION_ATTACK     = 8,
    MC189_ACTION_USE        = 9,
    MC189_ACTION_MINE       = 10,
    MC189_ACTION_PLACE      = 11,
    MC189_ACTION_HOTBAR_0   = 12,
    MC189_ACTION_HOTBAR_1   = 13,
    MC189_ACTION_HOTBAR_2   = 14,
    MC189_ACTION_HOTBAR_3   = 15,
    MC189_ACTION_HOTBAR_4   = 16,
    MC189_ACTION_HOTBAR_5   = 17,
    MC189_ACTION_HOTBAR_6   = 18,
    MC189_ACTION_HOTBAR_7   = 19,
    MC189_ACTION_HOTBAR_8   = 20,
    MC189_ACTION_LOOK_UP    = 21,
    MC189_ACTION_LOOK_DOWN  = 22,
    MC189_ACTION_LOOK_LEFT  = 23,
    MC189_ACTION_LOOK_RIGHT = 24,
    MC189_ACTION_DROP       = 25,
    MC189_ACTION_INVENTORY  = 26,
    MC189_ACTION_CRAFT      = 27,
    MC189_ACTION_LIGHT_PORTAL = 28,  // Use flint and steel on obsidian to light nether portal
    MC189_ACTION_COUNT      = 29,
} mc189_action_type_t;

// Status effect types
typedef enum {
    MC189_EFFECT_NONE            = 0,
    MC189_EFFECT_POISON          = 1,
    MC189_EFFECT_HUNGER          = 2,
    MC189_EFFECT_REGENERATION    = 3,
    MC189_EFFECT_STRENGTH        = 4,
    MC189_EFFECT_WEAKNESS        = 5,
    MC189_EFFECT_FIRE_RESISTANCE = 6,
    MC189_EFFECT_SLOW_FALLING    = 7,
    MC189_EFFECT_SPEED           = 8,
    MC189_EFFECT_SLOWNESS        = 9,
    MC189_EFFECT_HASTE           = 10,
    MC189_EFFECT_MINING_FATIGUE  = 11,
} mc189_effect_type_t;

// Win/loss conditions
typedef enum {
    MC189_GAME_RUNNING        = 0,
    MC189_GAME_WIN_DRAGON     = 1,  // Killed dragon and entered exit portal
    MC189_GAME_LOSS_DEATH     = 2,  // Player died
    MC189_GAME_LOSS_TIMEOUT   = 3,  // Max ticks exceeded
} mc189_game_state_t;

// Error codes
typedef enum {
    MC189_OK                    = 0,
    MC189_ERROR_VULKAN_INIT     = -1,
    MC189_ERROR_DEVICE_LOST     = -2,
    MC189_ERROR_OUT_OF_MEMORY   = -3,
    MC189_ERROR_INVALID_ACTION  = -4,
    MC189_ERROR_INVALID_STATE   = -5,
    MC189_ERROR_BATCH_SIZE      = -6,
    MC189_ERROR_SHADER_COMPILE  = -7,
    MC189_ERROR_NULL_POINTER    = -8,
} mc189_error_t;

// ============================================================================
// STRUCTURES
// ============================================================================

// Agent input for a single tick
typedef struct {
    // Discrete action (primary)
    mc189_action_type_t action;

    // Continuous look delta (radians) - allows fine-grained camera control
    float look_delta_yaw;    // Horizontal rotation
    float look_delta_pitch;  // Vertical rotation (clamped to [-89, 89] degrees)

    // For PLACE/MINE: target block coordinates (from raycasting)
    int32_t target_block[3];
    uint8_t target_face;     // 0-5: -Y, +Y, -Z, +Z, -X, +X

    // For CRAFT: recipe index
    uint16_t recipe_id;

    // Modifier flags (can be combined with movement actions)
    uint8_t flags;           // Bit 0: sprint, Bit 1: sneak, Bit 2: jump
} mc189_action_t;

// Status effect instance
typedef struct {
    mc189_effect_type_t type;
    float duration_ticks;
    uint8_t amplifier;
    uint8_t _pad[3];
} mc189_status_effect_t;

// Player observation
typedef struct {
    // Position and orientation
    float position[3];       // World coordinates
    float velocity[3];       // Blocks per second
    float yaw;               // Horizontal angle (degrees)
    float pitch;             // Vertical angle (degrees)

    // Health and hunger
    float health;            // 0-20 (hearts * 2)
    float max_health;        // Usually 20
    float hunger;            // 0-20
    float saturation;        // 0-20
    float exhaustion;        // 0-4 before hunger tick

    // State flags
    mc189_dimension_t dimension;
    uint8_t on_ground;
    uint8_t in_water;
    uint8_t in_lava;
    uint8_t sprinting;
    uint8_t sneaking;

    // Inventory (item IDs, 0 = empty)
    uint16_t inventory[MC189_INVENTORY_SIZE];
    uint8_t inventory_counts[MC189_INVENTORY_SIZE];
    uint16_t armor[MC189_ARMOR_SLOTS];
    uint16_t offhand;
    uint8_t active_slot;     // 0-8 hotbar index

    // Status effects
    mc189_status_effect_t effects[MC189_MAX_STATUS_EFFECTS];

    // Combat
    uint32_t attack_cooldown_ticks;  // 0 when ready (1.9+ only, 1.8.9 has no cooldown)
    float armor_value;
    float armor_toughness;

    // Experience
    uint32_t experience_level;
    float experience_progress; // 0-1 within current level
    uint32_t total_experience;

    // Target block (raycast result)
    int32_t looking_at_block[3];
    uint8_t looking_at_face;
    uint8_t has_target_block;
    uint16_t target_block_type;
    float target_distance;
} mc189_player_obs_t;

// Nearby mob information
typedef struct {
    uint16_t mob_type;
    float position[3];
    float velocity[3];
    float health;
    float max_health;
    float distance_to_player;
    float yaw;
    uint8_t is_hostile;
    uint8_t is_targeting_player;
    uint8_t _pad[2];
} mc189_mob_obs_t;

// Dragon fight state
typedef struct {
    uint8_t is_active;
    uint8_t phase;           // 1=circling, 2=diving, 3=perching
    uint8_t crystals_remaining;
    uint8_t _pad;
    float dragon_health;
    float dragon_position[3];
    uint32_t perch_timer;
} mc189_dragon_obs_t;

// Full observation for RL
typedef struct {
    mc189_player_obs_t player;
    mc189_dragon_obs_t dragon;

    // Nearby mobs (sorted by distance)
    uint32_t num_nearby_mobs;
    mc189_mob_obs_t nearby_mobs[16];

    // Nearby blocks (for local perception)
    // 7x7x7 cube centered on player, flattened
    uint16_t local_blocks[343];

    // Game state
    mc189_game_state_t game_state;
    uint32_t tick_number;
    uint64_t world_seed;

    // Win/loss info
    uint8_t terminated;      // Episode ended (win or loss)
    uint8_t truncated;       // Max ticks reached
    uint8_t _pad[2];
} mc189_observation_t;

// Step result for single environment
typedef struct {
    mc189_observation_t observation;
    float reward;
    uint8_t terminated;
    uint8_t truncated;
    uint8_t _pad[2];

    // Info dict values
    float episode_return;    // Cumulative reward if episode ended
    uint32_t episode_length; // Ticks if episode ended
    float dragon_damage_dealt;
    uint8_t crystals_destroyed;
    uint8_t portals_entered;
    uint16_t items_crafted;
} mc189_step_result_t;

// Batch step result for vectorized environments
typedef struct {
    uint32_t batch_size;
    mc189_step_result_t* results;  // Array of batch_size results
} mc189_batch_step_result_t;

// Simulator configuration
typedef struct {
    // Vulkan device selection
    int32_t preferred_device_index;  // -1 for auto-select
    bool enable_validation_layers;   // Debug only, impacts performance

    // Simulation settings
    uint32_t max_ticks_per_episode;  // Default: 36000 (30 min at 20 TPS)
    uint32_t world_render_distance;  // Chunks, affects loaded chunk count
    bool deterministic_mode;         // Force deterministic RNG

    // Batching
    uint32_t batch_size;             // Number of parallel environments
    bool async_step;                 // Non-blocking step (requires poll)

    // Reward shaping
    bool enable_reward_shaping;      // Extra rewards for progress
    float dragon_kill_reward;        // Default: 1000.0
    float death_penalty;             // Default: -100.0
    float time_penalty_per_tick;     // Default: -0.001
    float progress_reward_scale;     // Default: 1.0

    // Memory limits
    size_t max_gpu_memory_bytes;     // 0 for auto (use 80% available)
} mc189_config_t;

// Simulator statistics
typedef struct {
    // Performance
    double steps_per_second;
    double last_step_time_ms;
    double avg_step_time_ms;
    uint64_t total_steps;

    // GPU utilization
    double gpu_utilization_percent;
    size_t gpu_memory_used_bytes;
    size_t gpu_memory_total_bytes;

    // Per-shader timing (microseconds)
    double shader_times_us[32];
    const char* shader_names[32];
    uint32_t num_shaders;
} mc189_stats_t;

// ============================================================================
// OPAQUE HANDLES
// ============================================================================

typedef struct mc189_simulator_impl* mc189_simulator_t;

// ============================================================================
// LIFECYCLE FUNCTIONS
// ============================================================================

// Get default configuration
MC189_API mc189_config_t mc189_default_config(void);

// Create simulator with configuration
// Returns MC189_OK on success, error code on failure
MC189_API mc189_error_t mc189_create(
    const mc189_config_t* config,
    mc189_simulator_t* out_simulator
);

// Destroy simulator and release all resources
MC189_API void mc189_destroy(mc189_simulator_t simulator);

// Get last error message (thread-local)
MC189_API const char* mc189_get_error_message(void);

// ============================================================================
// SIMULATION FUNCTIONS
// ============================================================================

// Reset single environment to initial state
// seed: RNG seed for world generation, 0 for random
MC189_API mc189_error_t mc189_reset(
    mc189_simulator_t simulator,
    uint64_t seed,
    mc189_observation_t* out_observation
);

// Reset batch of environments
// seeds: Array of batch_size seeds (or NULL for random)
MC189_API mc189_error_t mc189_reset_batch(
    mc189_simulator_t simulator,
    const uint64_t* seeds,
    mc189_observation_t* out_observations
);

// Step single environment
MC189_API mc189_error_t mc189_step(
    mc189_simulator_t simulator,
    const mc189_action_t* action,
    mc189_step_result_t* out_result
);

// Step batch of environments
// actions: Array of batch_size actions
// results: Array of batch_size results (caller-allocated)
MC189_API mc189_error_t mc189_step_batch(
    mc189_simulator_t simulator,
    const mc189_action_t* actions,
    mc189_step_result_t* out_results
);

// Get current observation without stepping
MC189_API mc189_error_t mc189_get_observation(
    mc189_simulator_t simulator,
    uint32_t env_index,
    mc189_observation_t* out_observation
);

// Step multiple ticks without returning intermediate observations
// Useful for frame skipping in RL
MC189_API mc189_error_t mc189_step_n(
    mc189_simulator_t simulator,
    const mc189_action_t* action,
    uint32_t num_ticks,
    mc189_step_result_t* out_result
);

// ============================================================================
// ASYNC FUNCTIONS (requires async_step=true in config)
// ============================================================================

// Begin async step (non-blocking)
MC189_API mc189_error_t mc189_step_async(
    mc189_simulator_t simulator,
    const mc189_action_t* actions
);

// Poll for async step completion
// Returns MC189_OK when ready, MC189_ERROR_INVALID_STATE if still running
MC189_API mc189_error_t mc189_poll_step(
    mc189_simulator_t simulator,
    mc189_step_result_t* out_results
);

// Wait for async step to complete (blocking)
MC189_API mc189_error_t mc189_wait_step(
    mc189_simulator_t simulator,
    mc189_step_result_t* out_results
);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Get performance statistics
MC189_API mc189_error_t mc189_get_stats(
    mc189_simulator_t simulator,
    mc189_stats_t* out_stats
);

// Set environment to specific state (for debugging/testing)
MC189_API mc189_error_t mc189_set_player_position(
    mc189_simulator_t simulator,
    uint32_t env_index,
    const float position[3]
);

MC189_API mc189_error_t mc189_set_player_health(
    mc189_simulator_t simulator,
    uint32_t env_index,
    float health
);

MC189_API mc189_error_t mc189_set_dimension(
    mc189_simulator_t simulator,
    uint32_t env_index,
    mc189_dimension_t dimension
);

MC189_API mc189_error_t mc189_give_item(
    mc189_simulator_t simulator,
    uint32_t env_index,
    uint16_t item_id,
    uint8_t count,
    uint8_t slot
);

// Teleport player to specific location
MC189_API mc189_error_t mc189_teleport(
    mc189_simulator_t simulator,
    uint32_t env_index,
    mc189_dimension_t dimension,
    const float position[3]
);

// Spawn mob at location
MC189_API mc189_error_t mc189_spawn_mob(
    mc189_simulator_t simulator,
    uint32_t env_index,
    uint16_t mob_type,
    const float position[3]
);

// ============================================================================
// DEBUG FUNCTIONS
// ============================================================================

// Dump full world state to file (for debugging)
MC189_API mc189_error_t mc189_dump_state(
    mc189_simulator_t simulator,
    uint32_t env_index,
    const char* filepath
);

// Load world state from file
MC189_API mc189_error_t mc189_load_state(
    mc189_simulator_t simulator,
    uint32_t env_index,
    const char* filepath
);

// Get version string
MC189_API const char* mc189_version(void);

// Get Vulkan device name
MC189_API const char* mc189_device_name(mc189_simulator_t simulator);

// Check if GPU supports required features
MC189_API bool mc189_check_gpu_support(void);

#ifdef __cplusplus
}
#endif

#endif // MC189_SIMULATOR_API_H
