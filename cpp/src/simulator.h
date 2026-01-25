// simulator.h - Minecraft 1.8.9 RL Simulator Core Implementation
// Target: 500K+ steps/second on Apple Silicon via MoltenVK
// Architecture: Vulkan compute for parallel simulation, batched environments

#pragma once

#include "../include/mc189/simulator_api.h"
#include "vulkan/instance.h"
#include "vulkan/device.h"

#include <vulkan/vulkan.h>
#include <memory>
#include <vector>
#include <array>
#include <random>
#include <atomic>
#include <chrono>
#include <string>

namespace mcsim {

// Forward declarations
class ComputePipeline;
class BufferManager;
class ShaderManager;

/**
 * GPU buffer layout for a single environment's world state.
 * Matches the GLSL uniform buffer layout exactly.
 */
struct alignas(16) GpuPlayerState {
    float position[4];      // xyz + padding
    float velocity[4];      // xyz + padding
    float yaw;
    float pitch;
    float health;
    float max_health;
    float hunger;
    float saturation;
    float exhaustion;
    uint32_t dimension;
    uint32_t on_ground;
    uint32_t in_water;
    uint32_t in_lava;
    uint32_t sprinting;
    uint32_t sneaking;
    uint32_t active_slot;
    uint32_t flags;
    uint32_t attack_cooldown;
    float armor_value;
    float armor_toughness;
    uint32_t experience_level;
    float experience_progress;
    uint32_t total_experience;
    uint32_t _pad[2];
};

struct alignas(16) GpuDragonState {
    float position[4];      // xyz + padding
    float health;
    uint32_t phase;
    uint32_t crystals_remaining;
    uint32_t perch_timer;
    uint32_t is_active;
    uint32_t _pad[3];
};

struct alignas(16) GpuMobState {
    float position[4];
    float velocity[4];
    float health;
    float max_health;
    uint32_t mob_type;
    uint32_t state;
    float yaw;
    float distance_to_player;
    uint32_t is_hostile;
    uint32_t is_targeting_player;
};

struct alignas(16) GpuWorldState {
    GpuPlayerState player;
    GpuDragonState dragon;
    uint32_t tick_number;
    uint32_t game_state;
    uint64_t world_seed;
    uint32_t num_nearby_mobs;
    uint32_t _pad[3];
    GpuMobState mobs[16];
};

struct alignas(16) GpuAction {
    uint32_t action_type;
    float look_delta_yaw;
    float look_delta_pitch;
    int32_t target_block[4];  // xyz + face
    uint32_t recipe_id;
    uint32_t flags;
    uint32_t _pad[2];
};

struct alignas(16) GpuStepResult {
    float reward;
    uint32_t terminated;
    uint32_t truncated;
    uint32_t _pad;
    float episode_return;
    uint32_t episode_length;
    float dragon_damage_dealt;
    uint32_t crystals_destroyed;
    uint32_t portals_entered;
    uint32_t items_crafted;
    uint32_t _pad2[2];
};

/**
 * SimulatorImpl - Internal implementation of the Minecraft RL simulator.
 *
 * Manages Vulkan resources, compute shaders, and parallel environment simulation.
 * Optimized for Apple Silicon (MoltenVK) with Metal backend.
 */
class SimulatorImpl {
public:
    SimulatorImpl();
    ~SimulatorImpl();

    SimulatorImpl(const SimulatorImpl&) = delete;
    SimulatorImpl& operator=(const SimulatorImpl&) = delete;

    // Lifecycle
    mc189_error_t init(const mc189_config_t& config);
    void destroy();

    // Core simulation
    mc189_error_t reset(uint64_t seed, mc189_observation_t* out_obs);
    mc189_error_t reset_batch(const uint64_t* seeds, mc189_observation_t* out_obs);
    mc189_error_t step(const mc189_action_t* action, mc189_step_result_t* out_result);
    mc189_error_t step_batch(const mc189_action_t* actions, mc189_step_result_t* out_results);
    mc189_error_t step_n(const mc189_action_t* action, uint32_t num_ticks,
                         mc189_step_result_t* out_result);
    mc189_error_t get_observation(uint32_t env_index, mc189_observation_t* out_obs);

    // Async operations
    mc189_error_t step_async(const mc189_action_t* actions);
    mc189_error_t poll_step(mc189_step_result_t* out_results);
    mc189_error_t wait_step(mc189_step_result_t* out_results);

    // Utility
    mc189_error_t get_stats(mc189_stats_t* out_stats);
    mc189_error_t set_player_position(uint32_t env_index, const float position[3]);
    mc189_error_t set_player_health(uint32_t env_index, float health);
    mc189_error_t set_dimension(uint32_t env_index, mc189_dimension_t dimension);
    mc189_error_t give_item(uint32_t env_index, uint16_t item_id, uint8_t count, uint8_t slot);
    mc189_error_t teleport(uint32_t env_index, mc189_dimension_t dimension,
                           const float position[3]);
    mc189_error_t spawn_mob(uint32_t env_index, uint16_t mob_type, const float position[3]);
    mc189_error_t dump_state(uint32_t env_index, const char* filepath);
    mc189_error_t load_state(uint32_t env_index, const char* filepath);

    const char* device_name() const { return device_name_.c_str(); }
    bool is_valid() const { return initialized_; }
    uint32_t batch_size() const { return batch_size_; }

private:
    // Vulkan core resources
    VulkanInstance instance_;
    VulkanDevice device_;

    // Command execution
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence compute_fence_ = VK_NULL_HANDLE;
    VkSemaphore compute_semaphore_ = VK_NULL_HANDLE;

    // Descriptor management
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;

    // Compute pipelines for different simulation stages
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline physics_pipeline_ = VK_NULL_HANDLE;
    VkPipeline mob_ai_pipeline_ = VK_NULL_HANDLE;
    VkPipeline dragon_ai_pipeline_ = VK_NULL_HANDLE;
    VkPipeline reward_pipeline_ = VK_NULL_HANDLE;
    VkPipeline observation_pipeline_ = VK_NULL_HANDLE;
    VkShaderModule physics_shader_ = VK_NULL_HANDLE;
    VkShaderModule mob_ai_shader_ = VK_NULL_HANDLE;
    VkShaderModule dragon_ai_shader_ = VK_NULL_HANDLE;
    VkShaderModule reward_shader_ = VK_NULL_HANDLE;
    VkShaderModule observation_shader_ = VK_NULL_HANDLE;

    // GPU buffers
    VkBuffer world_state_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory world_state_memory_ = VK_NULL_HANDLE;
    VkBuffer action_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory action_memory_ = VK_NULL_HANDLE;
    VkBuffer result_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory result_memory_ = VK_NULL_HANDLE;
    VkBuffer staging_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory_ = VK_NULL_HANDLE;
    VkBuffer observation_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory observation_memory_ = VK_NULL_HANDLE;
    VkBuffer chunk_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory chunk_memory_ = VK_NULL_HANDLE;

    // CPU-side state for fast access
    std::vector<GpuWorldState> cpu_world_states_;
    std::vector<GpuAction> cpu_actions_;
    std::vector<GpuStepResult> cpu_results_;
    std::vector<mc189_observation_t> cpu_observations_;

    // Configuration
    mc189_config_t config_{};
    uint32_t batch_size_ = 1;
    bool initialized_ = false;
    bool async_pending_ = false;
    std::string device_name_;
    std::string last_error_;

    // Performance tracking
    struct PerformanceStats {
        std::atomic<uint64_t> total_steps{0};
        std::atomic<uint64_t> total_time_ns{0};
        std::chrono::high_resolution_clock::time_point last_step_start;
        double last_step_time_ms = 0.0;
        double avg_step_time_ms = 0.0;
        std::array<double, 32> shader_times_us{};
        std::array<const char*, 32> shader_names{};
        uint32_t num_shaders = 5;
    };
    PerformanceStats stats_;

    // RNG for world generation
    std::mt19937_64 rng_;

    // Initialization helpers
    bool init_vulkan(const mc189_config_t& config);
    bool create_command_resources();
    bool create_descriptor_resources();
    bool create_pipelines();
    bool create_buffers();
    void cleanup_vulkan();

    // Buffer management
    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags properties,
                       VkBuffer& buffer, VkDeviceMemory& memory);
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
    void copy_to_buffer(VkBuffer dst, const void* data, VkDeviceSize size);
    void copy_from_buffer(VkBuffer src, void* data, VkDeviceSize size);

    // Simulation core
    void record_step_commands(uint32_t num_envs);
    void submit_and_wait();
    void submit_async();
    bool is_async_complete();
    void wait_async();

    // State management
    void reset_environment(uint32_t env_index, uint64_t seed);
    void upload_actions(const mc189_action_t* actions, uint32_t count);
    void download_results(mc189_step_result_t* results, uint32_t count);
    void download_observations(mc189_observation_t* observations, uint32_t count);
    void convert_gpu_to_observation(const GpuWorldState& gpu_state,
                                    mc189_observation_t& obs);

    // Reward computation
    float compute_reward(const GpuWorldState& prev_state, const GpuWorldState& curr_state,
                         const mc189_action_t& action);

    void set_error(const char* msg);
};

// Thread-local error message storage
inline thread_local std::string g_error_message;

}  // namespace mcsim
