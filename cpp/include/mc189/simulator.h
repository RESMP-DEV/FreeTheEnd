#pragma once

#include "mc189/buffer_manager.h"
#include "mc189/compute_pipeline.h"
#include "mc189/vulkan_context.h"
#include "mc189/world_seed.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace mc189 {

// Match structures from dragon_fight_mvk.comp
struct alignas(16) Player {
  float position[3];
  float yaw;
  float velocity[3];
  float pitch;
  float health;
  float hunger;
  float saturation;
  float exhaustion;
  uint32_t flags;
  uint32_t invincibility_timer;
  uint32_t attack_cooldown;
  uint32_t weapon_slot; // 0=hand, 1=sword, 2=bow
  float arrow_charge;
  uint32_t arrows;
  uint32_t reserved[2];
};

struct alignas(16) Dragon {
  float position[3];
  float yaw;
  float velocity[3];
  float pitch;
  float health;
  uint32_t phase;
  uint32_t phase_timer;
  uint32_t target_pillar;
  float target_position[3];
  float breath_timer;
  uint32_t perch_timer;
  uint32_t attack_cooldown;
  uint32_t flags;
  float circle_angle;
  uint32_t reserved[3];
};

struct alignas(16) Crystal {
  float position[3];
  float is_alive; // 1.0 = alive, 0.0 = destroyed
};

struct InputState {
  float movement[3];
  float lookDeltaX;
  float lookDeltaY;
  uint32_t action; // 0=none, 1=attack, 2=use, 3=swap_weapon
  uint32_t actionData;
  uint32_t flags; // bit0=jump, bit1=sprint, bit2=sneak
};

struct GameState {
  uint32_t tickNumber;
  uint32_t gameFlags;
  uint32_t randomSeed;
  float deltaTime;
  uint32_t crystals_destroyed;
  uint32_t dragon_hits;
  uint32_t player_deaths;
  float best_dragon_damage;
};

// Dragon constants
constexpr uint32_t NUM_CRYSTALS = 10;
constexpr float DRAGON_MAX_HEALTH = 200.0f;
constexpr float END_SPAWN_Y = 64.0f;
constexpr float PILLAR_HEIGHT = 76.0f;
struct Observation {
  // Player (16 floats)
  float pos_x, pos_y, pos_z;
  float vel_x, vel_y, vel_z;
  float yaw, pitch;
  float health, hunger;
  float on_ground, attack_ready;
  float weapon, arrows, arrow_charge;
  float is_burning;  // 1.0 if player is on fire, 0.0 otherwise

  // Dragon (16 floats)
  float dragon_health;
  float dragon_x, dragon_y, dragon_z;
  float dragon_vel_x, dragon_vel_y, dragon_vel_z;
  float dragon_yaw;
  float dragon_phase;
  float dragon_dist;
  float dragon_dir_x, dragon_dir_z;
  float can_hit_dragon;
  float dragon_attacking;
  float burn_time_remaining;  // Ticks remaining on fire (0 = not burning)
  float reserved2;

  // Environment (16 floats)
  float crystals_remaining;
  float nearest_crystal_dist;
  float nearest_crystal_dir_x, nearest_crystal_dir_z;
  float nearest_crystal_y;
  float portal_active;
  float portal_dist;
  float time_remaining;
  float total_damage_dealt;
  float reserved3, reserved4, reserved5;
  float reserved6, reserved7, reserved8, reserved9;
};

// 48 floats for dragon fight shader
static constexpr size_t OBSERVATION_SIZE = 48;

class MC189Simulator {
public:
  struct Config {
    uint32_t num_envs = 1;
    bool enable_validation = false;
    bool use_cpu = false; // Force CPU backend even if GPU available
    std::string shader_dir = "shaders";
    // Stage-specific shader set. If non-empty, only these shaders are loaded
    // and dispatched instead of the monolithic dragon_fight/game_tick shader.
    std::vector<std::string> shader_set;
  };

  explicit MC189Simulator(const Config &config);
  ~MC189Simulator();

  // Main simulation interface
  void step(const int32_t *actions, size_t num_actions);

  /**
   * Reset environment(s) with optional seed.
   * @param env_id Environment index to reset, or 0xFFFFFFFF to reset all
   * @param seed World seed. 0 = generate random seed, otherwise use provided seed.
   *             Same seed guarantees same world layout and deterministic outcomes.
   */
  void reset(uint32_t env_id = 0xFFFFFFFF, uint64_t seed = 0);

  /**
   * Get the current world seed for an environment.
   * Useful for saving/restoring state or reproducing experiments.
   * @param env_id Environment index (default: 0)
   */
  uint64_t get_seed(uint32_t env_id = 0) const;

  // Data access (GPU -> CPU)
  const float *get_observations() const;
  const float *get_rewards() const;
  const uint8_t *get_dones() const;

  // Batch info
  uint32_t num_envs() const { return config_.num_envs; }
  static constexpr size_t obs_dim() { return OBSERVATION_SIZE; }
  bool using_gpu() const { return use_gpu_; }
  bool is_cpu_backend() const { return !use_gpu_; }

private:
  void load_shaders();
  void create_buffers();
  void dispatch_tick();
  void extract_observations();
  void compute_rewards();
  void auto_reset_done_envs();

  Config config_;
  std::unique_ptr<VulkanContext> ctx_;
  std::unique_ptr<BufferManager> buffer_mgr_;

  // Compute pipelines for each stage
  std::unique_ptr<ComputePipeline> setup_pipeline_;
  std::unique_ptr<ComputePipeline> player_pipeline_;
  std::unique_ptr<ComputePipeline> mob_pipeline_;
  std::unique_ptr<ComputePipeline> combat_pipeline_;
  std::unique_ptr<ComputePipeline> block_pipeline_;
  std::unique_ptr<ComputePipeline> world_pipeline_;

  // Stage-specific shader pipelines (keyed by shader name)
  std::unordered_map<std::string, std::unique_ptr<ComputePipeline>>
      shader_pipelines_;

  // GPU buffers (matching dragon_fight_mvk.comp bindings)
  Buffer player_buffer_;      // binding 0
  Buffer input_buffer_;       // binding 1
  Buffer dragon_buffer_;      // binding 2
  Buffer crystal_buffer_;     // binding 3
  Buffer game_state_buffer_;  // binding 4
  Buffer observation_buffer_; // binding 5
  Buffer reward_buffer_;      // binding 6
  Buffer done_buffer_;        // binding 7

  // CPU-side results
  std::vector<float> observations_;
  std::vector<float> rewards_;
  std::vector<uint8_t> dones_;

  // Deterministic world generation
  std::vector<WorldSeed> world_seeds_;  // One per environment

  bool use_gpu_ = false;
  uint32_t tick_number_ = 0;
};

// Chunk data for terrain (16x256x16 blocks = 65536 blocks per chunk)
// Stored as palette + indices for memory efficiency
struct alignas(16) ChunkData {
    uint16_t block_palette[256];    // Block type IDs
    uint8_t block_indices[65536];   // Index into palette
    uint8_t light_levels[65536];    // 4 bits sky, 4 bits block
    uint8_t biome_map[256];         // 16x16 biome at y=64
    int32_t chunk_x, chunk_z;
    uint32_t flags;                 // bit0=generated, bit1=decorated, bit2=lit
    uint32_t tick_updated;
};

// Entity types
enum class EntityType : uint32_t {
    NONE = 0,
    ZOMBIE, SKELETON, CREEPER, SPIDER, ENDERMAN,
    BLAZE, GHAST, WITHER_SKELETON, MAGMA_CUBE, PIGMAN,
    SILVERFISH, ENDER_DRAGON, IRON_GOLEM, VILLAGER,
    PIG, COW, SHEEP, CHICKEN,
    ARROW, FIREBALL, ENDER_PEARL, EYE_OF_ENDER,
    ITEM, XP_ORB, FALLING_BLOCK, TNT,
    MAX_ENTITY_TYPE
};

// Generic entity (mobs, projectiles, items)
struct alignas(16) Entity {
    float position[3];
    float velocity[3];
    float yaw, pitch;
    float health;
    uint32_t type;       // EntityType
    uint32_t flags;      // bit0=on_ground, bit1=hostile, bit2=burning
    uint32_t target_id;  // For AI targeting
    uint32_t spawn_tick;
    float ai_data[4];    // Type-specific AI state
};

// Portal state
struct alignas(16) Portal {
    float position[3];
    uint32_t type;       // 0=nether, 1=end
    uint32_t active;
    uint32_t target_dimension;
    float target_position[3];
};

// Extended game state for full speedrun
struct alignas(16) WorldState {
    uint32_t tick;
    uint32_t dimension;
    uint32_t weather;          // 0=clear, 1=rain, 2=thunder
    uint32_t time_of_day;      // 0-24000
    uint64_t world_seed;
    float stronghold_x, stronghold_z;
    uint32_t stage;            // Current GameStage
    uint32_t stage_progress;   // Bitmask of completed objectives
    uint32_t deaths;
    float total_reward;
    uint32_t reserved[4];
};

// Inventory slot
struct alignas(4) ItemStack {
    uint16_t item_id;
    uint8_t count;
    uint8_t damage;
};

// Extended player with full inventory
struct alignas(64) PlayerFull {
    Player base;                    // Existing 64 bytes
    ItemStack inventory[36];        // Main inventory
    ItemStack armor[4];             // Armor slots
    ItemStack offhand;              // Shield/totem
    uint32_t selected_slot;         // 0-8 hotbar
    uint32_t xp_level;
    uint32_t xp_progress;
    float spawn_x, spawn_y, spawn_z;
    uint32_t bed_set;
    uint32_t achievements;          // Bitmask
};

} // namespace mc189