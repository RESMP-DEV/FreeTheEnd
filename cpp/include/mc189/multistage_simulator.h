#pragma once

#include "mc189/buffer_manager.h"
#include "mc189/compute_pipeline.h"
#include "mc189/dimension.h"
#include "mc189/game_stage.h"
#include "mc189/simulator.h"
#include "mc189/vulkan_context.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace mc189 {

class MultistageSimulator {
public:
  struct Config {
    uint32_t num_envs = 1;
    GameStage initial_stage = GameStage::BASIC_SURVIVAL;
    bool enable_validation = false;
    std::string shader_dir = "shaders";
    bool auto_advance_stage = true;  // Auto-progress on objective completion
    uint32_t max_entities_per_env = 256;
    uint32_t max_chunks_per_env = 81; // 9x9 chunk area
  };

  explicit MultistageSimulator(const Config &config);
  ~MultistageSimulator();

  // Main interface
  void step(const int32_t *actions, size_t num_actions);
  void reset(uint32_t env_id = 0xFFFFFFFF, uint64_t seed = 0);

  // Stage management
  void set_stage(uint32_t env_id, GameStage stage);
  GameStage get_stage(uint32_t env_id) const;
  uint32_t get_stage_progress(uint32_t env_id) const;

  // Dimension transitions
  void teleport_to_dimension(uint32_t env_id, Dimension dim, float x, float y,
                             float z);
  Dimension get_dimension(uint32_t env_id) const;

  // Data access
  const float *get_observations() const;
  const float *get_rewards() const;
  const uint8_t *get_dones() const;
  const WorldState *get_world_states() const;

  // Extended observations (256 floats per env)
  static constexpr size_t EXTENDED_OBS_SIZE = 256;
  const float *get_extended_observations() const;

  uint32_t num_envs() const { return config_.num_envs; }

private:
  void load_stage_shaders();
  void create_extended_buffers();
  void dispatch_stage_tick(GameStage stage);
  void check_stage_transitions();
  void generate_terrain_around_player(uint32_t env_id);
  void spawn_mobs(uint32_t env_id);
  void update_portals(uint32_t env_id);

  // Initialize environment state for a given stage
  void init_env_for_stage(uint32_t env_id, GameStage stage, uint64_t seed);
  void fill_extended_observations();
  void compute_stage_rewards();

  Config config_;
  std::unique_ptr<VulkanContext> ctx_;
  std::unique_ptr<BufferManager> buffer_mgr_;

  // Per-stage shader pipelines
  std::unordered_map<GameStage, std::unique_ptr<ComputePipeline>>
      stage_pipelines_;

  // Additional pipelines
  std::unique_ptr<ComputePipeline> terrain_gen_pipeline_;
  std::unique_ptr<ComputePipeline> mob_ai_pipeline_;
  std::unique_ptr<ComputePipeline> portal_pipeline_;
  std::unique_ptr<ComputePipeline> observation_pipeline_;

  // Dragon fight delegated to existing simulator
  std::unique_ptr<MC189Simulator> dragon_sim_;

  // Extended buffers
  Buffer player_full_buffer_;
  Buffer entity_buffer_;
  Buffer chunk_buffer_;
  Buffer portal_buffer_;
  Buffer world_state_buffer_;
  Buffer extended_obs_buffer_;

  // Shared buffers
  Buffer input_buffer_;
  Buffer observation_buffer_;
  Buffer reward_buffer_;
  Buffer done_buffer_;

  // CPU state
  std::vector<float> observations_;
  std::vector<float> extended_observations_;
  std::vector<float> rewards_;
  std::vector<uint8_t> dones_;
  std::vector<WorldState> world_states_;
  std::vector<GameStage> current_stages_;
  std::vector<Dimension> current_dimensions_;
  std::vector<uint32_t> stage_ticks_;       // Per-env tick within stage
  std::vector<uint32_t> stage_progress_;    // Per-env bitmask of completed objectives

  // Chunk cache: key = (env_id << 32) | ((chunk_x & 0xFFFF) << 16) | (chunk_z & 0xFFFF)
  std::unordered_map<uint64_t, bool> generated_chunks_;

  uint32_t tick_number_ = 0;
};

} // namespace mc189
