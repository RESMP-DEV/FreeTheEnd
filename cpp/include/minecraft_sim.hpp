#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace minecraft_sim {

// Constants matching Vulkan shader definitions
constexpr uint32_t MAX_MOBS = 65536;
constexpr uint32_t MAX_LOADED_CHUNKS = 1024;
constexpr uint32_t CHUNK_SIZE = 16;
constexpr uint32_t CHUNK_HEIGHT = 384;
constexpr uint32_t TICKS_PER_SECOND = 20;

// Dimensions
enum class Dimension : uint32_t {
    Overworld = 0,
    Nether = 1,
    End = 2
};

// Block types (subset - full list in shader)
enum class BlockType : uint32_t {
    Air = 0,
    Stone = 1,
    Dirt = 2,
    Grass = 3,
    Sand = 4,
    Gravel = 5,
    Water = 6,
    Lava = 7,
    Bedrock = 8,
    EndPortal = 200,
    EndPortalFrame = 201,
    DragonEgg = 202
};

// Mob types
enum class MobType : uint32_t {
    Zombie = 0,
    Skeleton = 1,
    Creeper = 2,
    Spider = 3,
    Enderman = 4,
    Blaze = 5,
    Ghast = 6,
    EnderDragon = 100
};

// Action types
enum class ActionType : uint32_t {
    None = 0,
    Mine = 1,
    Place = 2,
    Attack = 3,
    UseItem = 4,
    Interact = 5,
    MoveForward = 6,
    MoveBack = 7,
    MoveLeft = 8,
    MoveRight = 9,
    Jump = 10,
    Sprint = 11,
    Sneak = 12,
    LookUp = 13,
    LookDown = 14,
    LookLeft = 15,
    LookRight = 16,
    SelectSlot = 17,
    LightPortal = 18  // Use flint and steel on obsidian to light nether portal
};

// Vector types
struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
};

struct IVec3 {
    int32_t x, y, z;
    IVec3() : x(0), y(0), z(0) {}
    IVec3(int32_t x, int32_t y, int32_t z) : x(x), y(y), z(z) {}
};

// Status effect
struct StatusEffect {
    uint32_t type;
    float duration;
    float amplifier;
    float reserved;
};

// Player state
struct Player {
    Vec3 position;
    float yaw;
    Vec3 velocity;
    float pitch;
    float health;
    float hunger;
    float saturation;
    float exhaustion;
    Dimension dimension;
    uint32_t gamemode;
    uint32_t active_slot;
    uint32_t flags;  // Bit flags: onGround, sprinting, sneaking, flying
    IVec3 target_block;
    uint32_t target_block_face;
    std::array<StatusEffect, 8> status_effects;
};

// Input state
struct InputState {
    Vec3 movement;      // Normalized movement direction
    float look_delta_x; // Mouse X delta
    float look_delta_y; // Mouse Y delta
    ActionType action;
    uint32_t action_data;
    uint32_t flags;     // Jump, sprint, sneak bits
};

// Mob state
struct Mob {
    Vec3 position;
    float yaw;
    Vec3 velocity;
    float pitch;
    float health;
    float max_health;
    MobType mob_type;
    uint32_t state;
    uint32_t target_entity;
    uint32_t flags;
    Vec3 home_position;
    float aggro_range;
    std::array<uint32_t, 4> ai_data;
};

// Dragon fight state
struct DragonFight {
    uint32_t phase;
    uint32_t health;
    uint32_t crystals_remaining;
    uint32_t target_pillar;
    Vec3 circle_center;
    float circle_radius;
    uint32_t perch_timer;
    uint32_t breath_timer;
};

// Observation space dimensions
struct ObservationShape {
    static constexpr size_t PLAYER_STATE = 32;      // Position, velocity, health, etc.
    static constexpr size_t INVENTORY = 36 * 2;     // 36 slots, 2 values each (item_id, count)
    static constexpr size_t NEARBY_BLOCKS = 11 * 11 * 11;  // 11x11x11 cube around player
    static constexpr size_t NEARBY_MOBS = 10 * 8;   // Up to 10 mobs, 8 values each
    static constexpr size_t DRAGON_STATE = 16;      // Dragon fight info
    static constexpr size_t TOTAL = PLAYER_STATE + INVENTORY + NEARBY_BLOCKS + NEARBY_MOBS + DRAGON_STATE;
};

// Action space
struct ActionSpace {
    static constexpr size_t DISCRETE_ACTIONS = 19;  // Number of discrete action types
    static constexpr size_t CONTINUOUS_DIMS = 4;    // Mouse X/Y delta + action data
};

// Step result
struct StepResult {
    bool terminated;
    bool truncated;
    float reward;

    // Reward components for debugging
    float damage_dealt;
    float damage_taken;
    float blocks_mined;
    float items_crafted;
    float distance_traveled;
    float dragon_damage;
};

// Main simulator class
class MinecraftSimulator {
public:
    explicit MinecraftSimulator(uint64_t seed = 0);
    ~MinecraftSimulator();

    // Gymnasium interface
    void reset(uint64_t seed);
    StepResult step(const InputState& input);

    // Observation access (for NumPy interface)
    const float* get_observation_ptr() const;
    size_t get_observation_size() const { return ObservationShape::TOTAL; }

    // State access
    const Player& get_player() const { return player_; }
    const std::vector<Mob>& get_mobs() const { return mobs_; }
    const DragonFight& get_dragon_fight() const { return dragon_fight_; }

    // Environment info
    uint64_t get_tick() const { return tick_; }
    bool is_dragon_dead() const;
    bool is_player_dead() const { return player_.health <= 0.0f; }

    // Action helpers
    static InputState discrete_to_input(int action, float intensity = 1.0f);

private:
    // Internal state
    Player player_;
    std::vector<Mob> mobs_;
    DragonFight dragon_fight_;
    uint64_t tick_;
    uint64_t seed_;
    std::mt19937_64 rng_;

    // Observation buffer (pre-allocated)
    mutable std::vector<float> observation_buffer_;

    // World state (simplified for CPU simulation)
    std::vector<uint8_t> chunk_data_;  // Block types

    // Internal methods
    void initialize_world();
    void update_physics();
    void process_input(const InputState& input);
    void update_mobs();
    void update_dragon();
    void update_observation();
    void compute_reward(StepResult& result, const Player& prev_state);

    Vec3 apply_movement(const Vec3& pos, const Vec3& vel, float dt);
    bool check_collision(const Vec3& pos, const Vec3& size);
    float calculate_fall_damage(float fall_distance);
};

// Vectorized environment for parallel simulation
class VecMinecraftSimulator {
public:
    explicit VecMinecraftSimulator(size_t num_envs, uint64_t base_seed = 0);
    ~VecMinecraftSimulator();

    // Batch operations
    void reset_all();
    void reset(size_t env_idx, uint64_t seed);
    void step_all(const std::vector<InputState>& inputs);
    void step(size_t env_idx, const InputState& input);

    // Batch observation access (contiguous memory for NumPy)
    const float* get_observations_ptr() const;
    const float* get_rewards_ptr() const;
    const bool* get_terminated_ptr() const;
    const bool* get_truncated_ptr() const;

    // Info
    size_t num_envs() const { return envs_.size(); }
    size_t observation_size() const { return ObservationShape::TOTAL; }

    // Access individual environments
    MinecraftSimulator& get_env(size_t idx) { return *envs_[idx]; }
    const MinecraftSimulator& get_env(size_t idx) const { return *envs_[idx]; }

private:
    std::vector<std::unique_ptr<MinecraftSimulator>> envs_;

    // Contiguous buffers for efficient NumPy interface
    mutable std::vector<float> observations_;
    mutable std::vector<float> rewards_;
    mutable std::vector<bool> terminated_;
    mutable std::vector<bool> truncated_;

    void update_buffers();
};

}  // namespace minecraft_sim
