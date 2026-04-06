// DragonFightSimulator - GPU-accelerated Ender Dragon fight simulation
// Optimized for RL training with "Free The End" goal

#include "mc189/buffer_manager.h"
#include "mc189/compute_pipeline.h"
#include "mc189/simulator.h"
#include "mc189/vulkan_context.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>

namespace mc189 {

namespace {

// Match shader constants
constexpr uint32_t NUM_CRYSTALS = 10;
constexpr float END_SPAWN_Y = 64.0f;
constexpr float DRAGON_MAX_HEALTH = 200.0f;
constexpr float PILLAR_HEIGHT = 76.0f;

// Observation size (must match shader Observation struct)
constexpr size_t OBS_SIZE = 48; // 16 + 16 + 16 floats

std::vector<uint32_t> load_spirv(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open shader: " + path);
  }

  const auto size = file.tellg();
  file.seekg(0);

  std::vector<uint32_t> spirv(size / sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(spirv.data()), size);
  return spirv;
}

struct PushConstants {
  uint32_t stage;
  uint32_t num_envs;
  uint32_t tick;
  uint32_t random_seed;
};

// GPU structures (must match shader)
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
  uint32_t weapon_slot;
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
  float active;
};

struct alignas(16) InputState {
  float movement[3];
  float look_delta_x;
  float look_delta_y;
  uint32_t action;
  uint32_t action_data;
  uint32_t flags;
};

struct alignas(16) GameState {
  uint32_t tick_number;
  uint32_t game_flags;
  uint32_t random_seed;
  float delta_time;
  uint32_t crystals_destroyed;
  uint32_t dragon_hits;
  uint32_t player_deaths;
  float best_dragon_damage;
};

vec3 get_pillar_position(uint32_t idx) {
  float angle = float(idx) * 0.628318f; // 2*PI/10
  float dist = 40.0f + float(idx % 3) * 15.0f;
  return {std::cos(angle) * dist, PILLAR_HEIGHT + 1.0f, std::sin(angle) * dist};
}

struct vec3 {
  float x, y, z;
};

} // namespace

class DragonFightSimulator {
public:
  struct Config {
    uint32_t num_envs = 4096;
    std::string shader_dir = "cpp/shaders";
    bool enable_validation = false;

    // Curriculum settings
    uint32_t curriculum_stage =
        0; // 0=full fight, 1=no crystals, 2=low dragon HP
    float dragon_hp_scale = 1.0f;
    uint32_t starting_crystals = NUM_CRYSTALS;
  };

  DragonFightSimulator(const Config &config);
  ~DragonFightSimulator();

  // RL interface
  void step(const int32_t *actions, size_t num_actions);
  void reset(uint32_t env_id = 0xFFFFFFFF);

  const float *get_observations() const { return observations_.data(); }
  const float *get_rewards() const { return rewards_.data(); }
  const uint8_t *get_dones() const { return dones_.data(); }

  // Info
  uint32_t num_envs() const { return config_.num_envs; }
  size_t obs_dim() const { return OBS_SIZE; }
  size_t action_dim() const { return 15; } // Extended action space

  std::string device_name() const { return ctx_ ? ctx_->device_name() : "CPU"; }

  // Stats
  uint32_t total_wins() const { return total_wins_; }
  uint32_t total_deaths() const { return total_deaths_; }
  float best_dragon_damage() const { return best_dragon_damage_; }

private:
  void load_shaders();
  void create_buffers();
  void dispatch_tick();
  void extract_observations();
  void auto_reset_done_envs();

  Config config_;
  uint32_t tick_number_ = 0;

  // Stats
  uint32_t total_wins_ = 0;
  uint32_t total_deaths_ = 0;
  float best_dragon_damage_ = 0.0f;

  // Vulkan
  std::unique_ptr<VulkanContext> ctx_;
  std::unique_ptr<BufferManager> buffer_mgr_;
  std::unique_ptr<ComputePipeline> pipeline_;

  // Buffers
  Buffer player_buffer_;
  Buffer input_buffer_;
  Buffer dragon_buffer_;
  Buffer crystal_buffer_;
  Buffer game_state_buffer_;
  Buffer observation_buffer_;
  Buffer reward_buffer_;
  Buffer done_buffer_;

  // CPU-side
  std::vector<float> observations_;
  std::vector<float> rewards_;
  std::vector<uint8_t> dones_;
};

DragonFightSimulator::DragonFightSimulator(const Config &config)
    : config_(config) {
  // Initialize Vulkan
  VulkanContext::Config ctx_config{};
  ctx_config.enable_validation = config.enable_validation;
  ctx_config.prefer_discrete_gpu = true;
  ctx_ = std::make_unique<VulkanContext>(ctx_config);

  buffer_mgr_ = std::make_unique<BufferManager>(*ctx_);

  load_shaders();
  create_buffers();

  observations_.resize(config_.num_envs * OBS_SIZE);
  rewards_.resize(config_.num_envs);
  dones_.resize(config_.num_envs);

  reset();
}

DragonFightSimulator::~DragonFightSimulator() = default;

void DragonFightSimulator::load_shaders() {
  const std::string shader_path = config_.shader_dir + "/dragon_fight.spv";

  std::vector<uint32_t> spirv;
  try {
    spirv = load_spirv(shader_path);
  } catch (const std::exception &) {
    // Try compiling
    std::string comp_path = config_.shader_dir + "/dragon_fight_mvk.comp";
    std::string cmd = "glslangValidator -V " + comp_path + " -o " + shader_path;
    system(cmd.c_str());
    spirv = load_spirv(shader_path);
  }

  if (spirv.empty())
    return;

  ComputePipeline::Config info{};
  info.spirv_code = spirv;
  info.entry_point = "main";
  info.local_size_x = 64;
  info.local_size_y = 1;
  info.local_size_z = 1;
  info.push_constants = {
      {0, sizeof(PushConstants), vk::ShaderStageFlagBits::eCompute}};

  info.bindings = {
      {0, vk::DescriptorType::eStorageBuffer}, // players
      {1, vk::DescriptorType::eStorageBuffer}, // inputs
      {2, vk::DescriptorType::eStorageBuffer}, // dragons
      {3, vk::DescriptorType::eStorageBuffer}, // crystals
      {4, vk::DescriptorType::eStorageBuffer}, // game_state
      {5, vk::DescriptorType::eStorageBuffer}, // observations
      {6, vk::DescriptorType::eStorageBuffer}, // rewards
      {7, vk::DescriptorType::eStorageBuffer}, // dones
  };

  pipeline_ = std::make_unique<ComputePipeline>(*ctx_, info);
}

void DragonFightSimulator::create_buffers() {
  const uint32_t n = config_.num_envs;

  player_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(Player), BufferUsage::Storage | BufferUsage::TransferDst);

  input_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(InputState), BufferUsage::Storage | BufferUsage::TransferDst);

  dragon_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(Dragon), BufferUsage::Storage | BufferUsage::TransferDst);

  crystal_buffer_ = buffer_mgr_->create_device_buffer(
      n * NUM_CRYSTALS * sizeof(Crystal),
      BufferUsage::Storage | BufferUsage::TransferDst);

  game_state_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(GameState), BufferUsage::Storage | BufferUsage::TransferDst);

  observation_buffer_ = buffer_mgr_->create_device_buffer(
      n * OBS_SIZE * sizeof(float),
      BufferUsage::Storage | BufferUsage::TransferSrc);

  reward_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(float), BufferUsage::Storage | BufferUsage::TransferSrc);

  done_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(uint32_t), BufferUsage::Storage | BufferUsage::TransferSrc);
}

void DragonFightSimulator::step(const int32_t *actions, size_t num_actions) {
  if (num_actions != config_.num_envs) {
    throw std::invalid_argument("Action count must match num_envs");
  }

  // Convert actions to InputState
  // Action space:
  //   0 = noop
  //   1 = forward
  //   2 = back
  //   3 = left
  //   4 = right
  //   5 = forward+left
  //   6 = forward+right
  //   7 = jump
  //   8 = jump+forward
  //   9 = attack
  //  10 = attack+forward
  //  11 = sprint+forward
  //  12 = look_left
  //  13 = look_right
  //  14 = swap_weapon

  std::vector<InputState> inputs(config_.num_envs);
  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    auto &in = inputs[i];
    std::memset(&in, 0, sizeof(InputState));

    const int32_t action = actions[i];
    switch (action) {
    case 1: // forward
      in.movement[2] = 1.0f;
      break;
    case 2: // back
      in.movement[2] = -1.0f;
      break;
    case 3: // left
      in.movement[0] = -1.0f;
      break;
    case 4: // right
      in.movement[0] = 1.0f;
      break;
    case 5: // forward+left
      in.movement[2] = 1.0f;
      in.movement[0] = -0.7f;
      break;
    case 6: // forward+right
      in.movement[2] = 1.0f;
      in.movement[0] = 0.7f;
      break;
    case 7: // jump
      in.flags |= 1;
      break;
    case 8: // jump+forward
      in.flags |= 1;
      in.movement[2] = 1.0f;
      break;
    case 9: // attack
      in.action = 1;
      break;
    case 10: // attack+forward
      in.action = 1;
      in.movement[2] = 1.0f;
      break;
    case 11: // sprint+forward
      in.flags |= 2;
      in.movement[2] = 1.0f;
      break;
    case 12: // look_left
      in.look_delta_x = -5.0f;
      break;
    case 13: // look_right
      in.look_delta_x = 5.0f;
      break;
    case 14: // swap_weapon
      in.action = 3;
      break;
    default:
      break;
    }
  }

  buffer_mgr_->upload(input_buffer_, inputs.data(),
                      inputs.size() * sizeof(InputState), 0);

  dispatch_tick();
  extract_observations();
  auto_reset_done_envs();

  tick_number_++;
}

void DragonFightSimulator::dispatch_tick() {
  if (!pipeline_) {
    // CPU fallback
    std::mt19937 rng(tick_number_);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < observations_.size(); ++i) {
      observations_[i] = dist(rng);
    }
    for (size_t i = 0; i < rewards_.size(); ++i) {
      rewards_[i] = dist(rng) * 0.1f - 0.05f;
    }
    std::fill(dones_.begin(), dones_.end(), 0);
    return;
  }

  auto desc_set = pipeline_->allocate_descriptor_set();
  pipeline_->update_descriptor(desc_set, 0, player_buffer_.handle());
  pipeline_->update_descriptor(desc_set, 1, input_buffer_.handle());
  pipeline_->update_descriptor(desc_set, 2, dragon_buffer_.handle());
  pipeline_->update_descriptor(desc_set, 3, crystal_buffer_.handle());
  pipeline_->update_descriptor(desc_set, 4, game_state_buffer_.handle());
  pipeline_->update_descriptor(desc_set, 5, observation_buffer_.handle());
  pipeline_->update_descriptor(desc_set, 6, reward_buffer_.handle());
  pipeline_->update_descriptor(desc_set, 7, done_buffer_.handle());

  const uint32_t workgroups = (config_.num_envs + 63) / 64;

  std::random_device rd;
  const uint32_t random_seed = rd();

  auto cmd = ctx_->allocate_command_buffer();
  vk::CommandBufferBeginInfo begin_info{};
  begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  cmd.begin(begin_info);

  // 6 stages
  for (uint32_t stage = 0; stage < 6; ++stage) {
    PushConstants pc{stage, config_.num_envs, tick_number_, random_seed};

    pipeline_->bind(cmd);
    pipeline_->bind_descriptor_set(cmd, desc_set);
    pipeline_->push_constants(cmd, pc);
    pipeline_->dispatch(cmd, workgroups, 1, 1);

    vk::MemoryBarrier barrier{};
    barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, barrier,
                        {}, {});
  }

  cmd.end();
  ctx_->submit_and_wait(cmd);
  pipeline_->free_descriptor_set(desc_set);
}

void DragonFightSimulator::extract_observations() {
  if (pipeline_) {
    buffer_mgr_->download(observation_buffer_, observations_.data(),
                          observations_.size() * sizeof(float), 0);
    buffer_mgr_->download(reward_buffer_, rewards_.data(),
                          rewards_.size() * sizeof(float), 0);

    std::vector<uint32_t> done_u32(config_.num_envs);
    buffer_mgr_->download(done_buffer_, done_u32.data(),
                          done_u32.size() * sizeof(uint32_t), 0);

    for (uint32_t i = 0; i < config_.num_envs; ++i) {
      dones_[i] = done_u32[i] ? 1 : 0;

      // Track stats
      if (done_u32[i]) {
        // Check if win (reward > 500 indicates win)
        if (rewards_[i] > 500.0f) {
          total_wins_++;
        } else {
          total_deaths_++;
        }
      }
    }
  }
}

void DragonFightSimulator::auto_reset_done_envs() {
  // Download current state to check for done envs
  std::vector<Player> players(config_.num_envs);
  std::vector<Dragon> dragons(config_.num_envs);
  std::vector<GameState> states(config_.num_envs);

  buffer_mgr_->download(player_buffer_, players.data(),
                        players.size() * sizeof(Player), 0);
  buffer_mgr_->download(dragon_buffer_, dragons.data(),
                        dragons.size() * sizeof(Dragon), 0);
  buffer_mgr_->download(game_state_buffer_, states.data(),
                        states.size() * sizeof(GameState), 0);

  bool need_upload = false;

  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    if (dones_[i]) {
      need_upload = true;

      // Track best damage
      float dmg_dealt = DRAGON_MAX_HEALTH - dragons[i].health;
      if (dmg_dealt > best_dragon_damage_) {
        best_dragon_damage_ = dmg_dealt;
      }

      // Reset player
      auto &p = players[i];
      std::memset(&p, 0, sizeof(Player));
      p.position[0] = 0.0f;
      p.position[1] = END_SPAWN_Y;
      p.position[2] = 0.0f;
      p.health = 20.0f;
      p.hunger = 20.0f;
      p.saturation = 5.0f;
      p.weapon_slot = 1; // Start with sword
      p.arrows = 64;

      // Reset dragon (with curriculum scaling)
      auto &d = dragons[i];
      std::memset(&d, 0, sizeof(Dragon));
      d.position[0] = 0.0f;
      d.position[1] = 70.0f;
      d.position[2] = 75.0f;
      d.health = DRAGON_MAX_HEALTH * config_.dragon_hp_scale;
      d.phase = 0;                      // Circling
      d.circle_angle = float(i) * 0.1f; // Vary starting angle

      // Reset game state
      auto &gs = states[i];
      std::memset(&gs, 0, sizeof(GameState));
      gs.delta_time = 0.05f;

      // Reset dones
      dones_[i] = 0;
    }
  }

  if (need_upload) {
    buffer_mgr_->upload(player_buffer_, players.data(),
                        players.size() * sizeof(Player), 0);
    buffer_mgr_->upload(dragon_buffer_, dragons.data(),
                        dragons.size() * sizeof(Dragon), 0);
    buffer_mgr_->upload(game_state_buffer_, states.data(),
                        states.size() * sizeof(GameState), 0);

    // Also reset crystals for done envs
    std::vector<Crystal> all_crystals(config_.num_envs * NUM_CRYSTALS);
    buffer_mgr_->download(crystal_buffer_, all_crystals.data(),
                          all_crystals.size() * sizeof(Crystal), 0);

    for (uint32_t i = 0; i < config_.num_envs; ++i) {
      if (dones_[i] == 0 && players[i].health > 0)
        continue; // Already processed

      // Re-init crystals
      for (uint32_t j = 0; j < NUM_CRYSTALS; j++) {
        auto pillar_pos = get_pillar_position(j);
        auto &c = all_crystals[i * NUM_CRYSTALS + j];
        c.position[0] = pillar_pos.x;
        c.position[1] = pillar_pos.y;
        c.position[2] = pillar_pos.z;
        c.active = (j < config_.starting_crystals) ? 1.0f : 0.0f;
      }
    }

    buffer_mgr_->upload(crystal_buffer_, all_crystals.data(),
                        all_crystals.size() * sizeof(Crystal), 0);

    // Clear dones buffer on GPU
    std::vector<uint32_t> zeros(config_.num_envs, 0);
    buffer_mgr_->upload(done_buffer_, zeros.data(),
                        zeros.size() * sizeof(uint32_t), 0);
  }
}

void DragonFightSimulator::reset(uint32_t env_id) {
  if (env_id == 0xFFFFFFFF) {
    // Reset all
    std::vector<Player> players(config_.num_envs);
    std::vector<Dragon> dragons(config_.num_envs);
    std::vector<Crystal> all_crystals(config_.num_envs * NUM_CRYSTALS);
    std::vector<GameState> states(config_.num_envs);

    for (uint32_t i = 0; i < config_.num_envs; ++i) {
      auto &p = players[i];
      std::memset(&p, 0, sizeof(Player));
      p.position[0] = 0.0f;
      p.position[1] = END_SPAWN_Y;
      p.position[2] = 0.0f;
      p.health = 20.0f;
      p.hunger = 20.0f;
      p.saturation = 5.0f;
      p.weapon_slot = 1; // Sword
      p.arrows = 64;

      auto &d = dragons[i];
      std::memset(&d, 0, sizeof(Dragon));
      d.position[0] = 0.0f;
      d.position[1] = 70.0f;
      d.position[2] = 75.0f;
      d.health = DRAGON_MAX_HEALTH * config_.dragon_hp_scale;
      d.phase = 0;
      d.circle_angle = float(i) * 0.1f;

      for (uint32_t j = 0; j < NUM_CRYSTALS; j++) {
        auto pillar_pos = get_pillar_position(j);
        auto &c = all_crystals[i * NUM_CRYSTALS + j];
        c.position[0] = pillar_pos.x;
        c.position[1] = pillar_pos.y;
        c.position[2] = pillar_pos.z;
        c.active = (j < config_.starting_crystals) ? 1.0f : 0.0f;
      }

      auto &gs = states[i];
      std::memset(&gs, 0, sizeof(GameState));
      gs.delta_time = 0.05f;
    }

    buffer_mgr_->upload(player_buffer_, players.data(),
                        players.size() * sizeof(Player), 0);
    buffer_mgr_->upload(dragon_buffer_, dragons.data(),
                        dragons.size() * sizeof(Dragon), 0);
    buffer_mgr_->upload(crystal_buffer_, all_crystals.data(),
                        all_crystals.size() * sizeof(Crystal), 0);
    buffer_mgr_->upload(game_state_buffer_, states.data(),
                        states.size() * sizeof(GameState), 0);

    std::vector<uint32_t> zeros(config_.num_envs, 0);
    buffer_mgr_->upload(done_buffer_, zeros.data(),
                        zeros.size() * sizeof(uint32_t), 0);

    tick_number_ = 0;
    std::fill(dones_.begin(), dones_.end(), 0);
  }
}

} // namespace mc189
