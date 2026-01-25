// observation_encoder.h - GPU-efficient observation encoding for Minecraft RL
//
// Encodes raw simulation state into compact observation tensors suitable for
// neural network input. Designed for minimal CPU-GPU transfer overhead.
//
// Key design decisions:
// - Fixed-size output buffers to avoid dynamic allocation
// - SIMD-friendly memory layout (aligned, contiguous)
// - Support for batched encoding (all environments in parallel)
// - Optional GPU-side encoding via compute shaders

#ifndef MC189_OBSERVATION_ENCODER_H
#define MC189_OBSERVATION_ENCODER_H

#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>
#include <mc189/simulator_api.h>

namespace mc189 {

// ============================================================================
// CONSTANTS
// ============================================================================

// Observation dimensions
constexpr size_t PLAYER_STATE_DIM = 22;
constexpr size_t INVENTORY_DIM = 25;
constexpr size_t RAYCAST_DIM = 16;
constexpr size_t ENTITY_DIM = 96;      // 8 mobs * 8 + 4 items * 8
constexpr size_t DRAGON_DIM = 10;
constexpr size_t GAME_STATE_DIM = 3;
constexpr size_t CONTINUOUS_DIM = PLAYER_STATE_DIM + INVENTORY_DIM + RAYCAST_DIM +
                                   ENTITY_DIM + DRAGON_DIM + GAME_STATE_DIM;  // 172

// Voxel grid
constexpr size_t VOXEL_GRID_SIZE = 16;
constexpr size_t VOXEL_GRID_TOTAL = VOXEL_GRID_SIZE * VOXEL_GRID_SIZE * VOXEL_GRID_SIZE;  // 4096
constexpr size_t FLAT_OBS_DIM = CONTINUOUS_DIM + VOXEL_GRID_TOTAL;  // 4268

// Raycast configuration
constexpr size_t NUM_RAYCAST_DIRS = 16;
constexpr float MAX_RAYCAST_DIST = 64.0f;

// Entity tracking
constexpr size_t MAX_NEARBY_MOBS = 8;
constexpr size_t MAX_NEARBY_ITEMS = 4;
constexpr size_t ENTITY_FEATURES = 8;

// Key item IDs for inventory summary
constexpr std::array<uint16_t, 16> KEY_ITEM_IDS = {
    369,   // blaze_rod
    368,   // ender_pearl
    381,   // ender_eye
    49,    // obsidian
    263,   // coal
    265,   // iron_ingot
    264,   // diamond
    276,   // diamond_sword
    278,   // diamond_pickaxe
    262,   // arrow
    261,   // bow
    322,   // golden_apple
    373,   // potion
    327,   // lava_bucket
    326,   // water_bucket
    325,   // bucket
};

// Dimension bounds for position normalization
struct DimensionBounds {
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;
};

constexpr DimensionBounds OVERWORLD_BOUNDS = {
    -30000000.0f, 30000000.0f,
    -64.0f, 320.0f,
    -30000000.0f, 30000000.0f
};

constexpr DimensionBounds NETHER_BOUNDS = {
    -3750000.0f, 3750000.0f,
    0.0f, 256.0f,
    -3750000.0f, 3750000.0f
};

constexpr DimensionBounds END_BOUNDS = {
    -30000000.0f, 30000000.0f,
    0.0f, 256.0f,
    -30000000.0f, 30000000.0f
};

// ============================================================================
// ENCODED OBSERVATION STRUCTURES
// ============================================================================

// Player state (22 floats, 16-byte aligned)
struct alignas(16) EncodedPlayerState {
    float position[3];          // Normalized to [-1, 1]
    float velocity[3];          // Scaled velocity
    float health;               // [0, 1]
    float max_health;           // [0, 1]
    float hunger;               // [0, 1]
    float saturation;           // [0, 1]
    float exhaustion;           // [0, 1]
    float yaw;                  // [-1, 1]
    float pitch;                // [-1, 1]
    float dimension;            // [0, 1]
    float equipped_slot;        // [0, 1]
    float equipped_item;        // Normalized item ID
    float flags[6];             // Binary: on_ground, in_water, in_lava, sprinting, sneaking, pad
};
static_assert(sizeof(EncodedPlayerState) == 88, "EncodedPlayerState size mismatch");

// Inventory summary (25 floats)
struct alignas(16) EncodedInventory {
    float key_item_counts[16];  // Log-normalized counts
    float hotbar[9];            // Normalized item IDs
};

// Entity observation (8 floats)
struct alignas(16) EncodedEntity {
    float entity_type;          // Normalized type ID
    float distance;             // Normalized distance
    float direction[3];         // Unit vector to entity
    float health;               // [0, 1]
    float is_hostile;           // Binary
    float is_targeting;         // Binary
};

// Dragon state (10 floats)
struct alignas(16) EncodedDragonState {
    float is_active;
    float phase;                // [0, 1]
    float health;               // [0, 1]
    float crystals;             // [0, 1]
    float direction[3];         // Unit vector to dragon
    float distance;             // Normalized
    float phase_timer;          // [0, 1]
    float _pad;
};

// Full continuous observation (172 floats)
struct alignas(64) EncodedContinuousObs {
    EncodedPlayerState player;
    EncodedInventory inventory;
    float raycasts[NUM_RAYCAST_DIRS];
    EncodedEntity mobs[MAX_NEARBY_MOBS];
    EncodedEntity items[MAX_NEARBY_ITEMS];
    EncodedDragonState dragon;
    float game_tick;
    float terminated;
    float truncated;
};

// Full flat observation (continuous + voxels)
struct alignas(64) EncodedFlatObs {
    EncodedContinuousObs continuous;
    float voxels[VOXEL_GRID_TOTAL];  // Binary: 1.0 = solid, 0.0 = air
};

// ============================================================================
// OBSERVATION ENCODER
// ============================================================================

// Voxel encoding modes
enum class VoxelEncoding {
    Binary,     // 1.0 for solid, 0.0 for air
    BlockIDs,   // Raw block IDs (for embedding lookup)
    OneHot,     // One-hot vectors (large but informative)
    None        // Don't include voxels
};

// Encoder configuration
struct ObservationEncoderConfig {
    VoxelEncoding voxel_encoding = VoxelEncoding::Binary;
    uint32_t num_block_types = 32;      // For one-hot encoding
    bool include_raycast = true;
    bool include_entities = true;
    bool include_dragon = true;
    float raycast_max_dist = MAX_RAYCAST_DIST;
};

class ObservationEncoder {
public:
    explicit ObservationEncoder(const ObservationEncoderConfig& config = {});
    ~ObservationEncoder();

    // Non-copyable, moveable
    ObservationEncoder(const ObservationEncoder&) = delete;
    ObservationEncoder& operator=(const ObservationEncoder&) = delete;
    ObservationEncoder(ObservationEncoder&&) noexcept;
    ObservationEncoder& operator=(ObservationEncoder&&) noexcept;

    // ========================================================================
    // ENCODING FUNCTIONS
    // ========================================================================

    // Encode single observation
    void encode(const mc189_observation_t& raw, float* out_buffer) const;

    // Encode single observation to struct
    void encode(const mc189_observation_t& raw, EncodedFlatObs& out) const;

    // Encode batch of observations
    // out_buffer: pre-allocated buffer of size batch_size * get_obs_dim()
    void encode_batch(
        const mc189_observation_t* raw_batch,
        size_t batch_size,
        float* out_buffer
    ) const;

    // Encode to separate continuous and voxel buffers
    void encode_split(
        const mc189_observation_t& raw,
        float* out_continuous,
        int32_t* out_voxels
    ) const;

    // Encode batch with split outputs
    void encode_batch_split(
        const mc189_observation_t* raw_batch,
        size_t batch_size,
        float* out_continuous,
        int32_t* out_voxels
    ) const;

    // ========================================================================
    // HELPER ENCODING FUNCTIONS
    // ========================================================================

    // Encode player state portion
    void encode_player(const mc189_player_obs_t& player, EncodedPlayerState& out) const;

    // Encode inventory summary
    void encode_inventory(const mc189_player_obs_t& player, EncodedInventory& out) const;

    // Encode nearby entities
    void encode_entities(
        const mc189_mob_obs_t* mobs,
        uint32_t num_mobs,
        const float player_pos[3],
        EncodedEntity* out_mobs,
        EncodedEntity* out_items
    ) const;

    // Encode dragon state
    void encode_dragon(
        const mc189_dragon_obs_t& dragon,
        const float player_pos[3],
        EncodedDragonState& out
    ) const;

    // Encode voxel grid (binary)
    void encode_voxels_binary(const uint16_t* local_blocks, float* out) const;

    // Encode voxel grid (block IDs for embedding)
    void encode_voxels_ids(const uint16_t* local_blocks, int32_t* out) const;

    // ========================================================================
    // RAYCAST
    // ========================================================================

    // Perform raycasts from player position
    // world: block lookup callback
    // results: output buffer of size NUM_RAYCAST_DIRS
    void compute_raycasts(
        const float position[3],
        const float yaw,
        const float pitch,
        const uint16_t* local_blocks,
        float* out_distances
    ) const;

    // ========================================================================
    // PROPERTIES
    // ========================================================================

    // Get total observation dimension
    size_t get_obs_dim() const;

    // Get continuous observation dimension
    size_t get_continuous_dim() const { return CONTINUOUS_DIM; }

    // Get voxel dimension
    size_t get_voxel_dim() const;

    // Get configuration
    const ObservationEncoderConfig& get_config() const { return config_; }

private:
    ObservationEncoderConfig config_;

    // Precomputed raycast directions
    float raycast_dirs_[NUM_RAYCAST_DIRS][3];

    // Initialize raycast directions
    void init_raycast_directions();

    // Normalization helpers
    float normalize_position(float val, float min_val, float max_val) const;
    void get_dimension_bounds(mc189_dimension_t dim, DimensionBounds& bounds) const;
};

// ============================================================================
// BATCH ENCODING UTILITIES
// ============================================================================

// Pre-allocate buffers for batch encoding
class ObservationBuffer {
public:
    explicit ObservationBuffer(size_t batch_size, const ObservationEncoderConfig& config = {});

    // Get buffer pointers
    float* continuous_data() { return continuous_.data(); }
    int32_t* voxel_data() { return voxels_.data(); }
    float* flat_data() { return flat_.data(); }

    // Buffer sizes
    size_t batch_size() const { return batch_size_; }
    size_t continuous_stride() const { return CONTINUOUS_DIM; }
    size_t voxel_stride() const { return VOXEL_GRID_TOTAL; }
    size_t flat_stride() const { return flat_stride_; }

    // Total bytes for GPU transfer
    size_t continuous_bytes() const { return continuous_.size() * sizeof(float); }
    size_t voxel_bytes() const { return voxels_.size() * sizeof(int32_t); }
    size_t flat_bytes() const { return flat_.size() * sizeof(float); }

private:
    size_t batch_size_;
    size_t flat_stride_;
    std::vector<float> continuous_;
    std::vector<int32_t> voxels_;
    std::vector<float> flat_;
};

// ============================================================================
// SIMD UTILITIES
// ============================================================================

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define MC189_HAS_NEON 1
#include <arm_neon.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define MC189_HAS_SSE 1
#include <immintrin.h>
#endif

// Vectorized normalization for batched encoding
void normalize_positions_batch(
    const float* positions,  // [batch_size, 3]
    const DimensionBounds* bounds,  // [batch_size]
    size_t batch_size,
    float* out  // [batch_size, 3]
);

// Vectorized log1p for inventory normalization
void log_normalize_batch(
    const uint8_t* counts,  // [batch_size, 16]
    size_t batch_size,
    float* out  // [batch_size, 16]
);

}  // namespace mc189

#endif  // MC189_OBSERVATION_ENCODER_H
