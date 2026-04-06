// MC189Simulator - GPU-accelerated Minecraft 1.8.9 simulation
// Dispatches game_tick.comp shader for batched RL environments

#include "mc189/buffer_manager.h"
#include "mc189/compute_pipeline.h"
#include "mc189/simulator.h"
#include "mc189/vulkan_context.h"

#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>

namespace mc189 {

namespace {

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

// Push constant for shader stage selection
struct PushConstants {
  uint32_t stage;
  uint32_t num_envs;
  uint32_t tick;
  uint32_t random_seed;
};

// Helper to get pillar position (matches shader)
void get_pillar_position(uint32_t idx, float *out) {
  float angle = float(idx) * 0.628318f; // 2*PI/10
  float dist = 40.0f + float(idx % 3) * 15.0f;
  out[0] = std::cos(angle) * dist;
  out[1] = PILLAR_HEIGHT + 1.0f;
  out[2] = std::sin(angle) * dist;
}

void fill_cpu_outputs(WorldSeed &seed, float *obs, float &reward,
                      uint8_t &done) {
  auto &rng = seed.tick_rng();
  for (size_t i = 0; i < OBSERVATION_SIZE; ++i) {
    obs[i] = rng.next_float();
  }
  reward = rng.next_float() * 0.1f - 0.05f;
  done = 0;
}

} // namespace

MC189Simulator::MC189Simulator(const Config &config) : config_(config) {
  // Determine backend: GPU if not forced CPU and Vulkan is available
  if (!config.use_cpu) {
    try {
      VulkanContext::Config ctx_config{};
      ctx_config.enable_validation = config.enable_validation;
      ctx_config.prefer_discrete_gpu = true;
      ctx_ = std::make_unique<VulkanContext>(ctx_config);

      buffer_mgr_ = std::make_unique<BufferManager>(*ctx_);

      load_shaders();
      create_buffers();
      use_gpu_ = true;
    } catch (const std::exception &) {
      // Vulkan unavailable or init failed - fall back to CPU
      ctx_.reset();
      buffer_mgr_.reset();
      use_gpu_ = false;
    }
  }

  // Pre-allocate CPU buffers (needed for both backends)
  observations_.resize(config_.num_envs * OBSERVATION_SIZE);
  rewards_.resize(config_.num_envs);
  dones_.resize(config_.num_envs);

  // Initialize world seeds (one per environment)
  world_seeds_.reserve(config_.num_envs);
  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    world_seeds_.emplace_back(0); // Random seed on first construction
  }

  // Initialize all environments with default random seeds
  reset();
}

MC189Simulator::~MC189Simulator() = default;

void MC189Simulator::load_shaders() {
  // Common pipeline configuration shared by all shaders
  auto make_pipeline_config = [&](const std::vector<uint32_t> &spirv)
      -> ComputePipeline::Config {
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
        {2, vk::DescriptorType::eStorageBuffer}, // mobs
        {3, vk::DescriptorType::eStorageBuffer}, // chunks
        {4, vk::DescriptorType::eStorageBuffer}, // game_state
        {5, vk::DescriptorType::eStorageBuffer}, // observations
        {6, vk::DescriptorType::eStorageBuffer}, // rewards
        {7, vk::DescriptorType::eStorageBuffer}, // dones
    };
    return info;
  };

  // Stage-specific shader loading: load only the shaders this stage needs
  if (!config_.shader_set.empty()) {
    for (const auto &shader_name : config_.shader_set) {
      std::string path = config_.shader_dir + "/" + shader_name + ".spv";
      std::vector<uint32_t> spirv;
      try {
        spirv = load_spirv(path);
      } catch (const std::exception &) {
        // Shader not found - skip (allows graceful degradation)
        continue;
      }
      if (spirv.empty()) {
        continue;
      }
      auto cfg = make_pipeline_config(spirv);
      shader_pipelines_[shader_name] =
          std::make_unique<ComputePipeline>(*ctx_, cfg);
    }

    // Use the first successfully loaded shader as setup_pipeline_ for
    // descriptor set allocation and buffer binding compatibility
    if (!shader_pipelines_.empty()) {
      // Point setup_pipeline_ at the first loaded shader so dispatch_tick
      // knows GPU mode is active
      auto &first = shader_pipelines_.begin()->second;
      // Create a duplicate pipeline for setup (descriptor allocation)
      auto first_name = config_.shader_set[0];
      for (const auto &name : config_.shader_set) {
        if (shader_pipelines_.count(name)) {
          first_name = name;
          break;
        }
      }
      std::string path = config_.shader_dir + "/" + first_name + ".spv";
      auto spirv = load_spirv(path);
      setup_pipeline_ = std::make_unique<ComputePipeline>(
          *ctx_, make_pipeline_config(spirv));
    }
    return;
  }

  // Legacy path: monolithic shader (dragon_fight -> game_tick fallback)
  std::string shader_path = config_.shader_dir + "/dragon_fight.spv";
  std::vector<uint32_t> spirv;
  try {
    spirv = load_spirv(shader_path);
  } catch (const std::exception &) {
    shader_path = config_.shader_dir + "/game_tick.spv";
    try {
      spirv = load_spirv(shader_path);
    } catch (const std::exception &) {
      return;
    }
  }

  if (spirv.empty()) {
    return;
  }

  auto info = make_pipeline_config(spirv);
  setup_pipeline_ = std::make_unique<ComputePipeline>(*ctx_, info);
  player_pipeline_ = std::make_unique<ComputePipeline>(*ctx_, info);
  mob_pipeline_ = std::make_unique<ComputePipeline>(*ctx_, info);
  combat_pipeline_ = std::make_unique<ComputePipeline>(*ctx_, info);
  block_pipeline_ = std::make_unique<ComputePipeline>(*ctx_, info);
  world_pipeline_ = std::make_unique<ComputePipeline>(*ctx_, info);
}

void MC189Simulator::create_buffers() {
  const uint32_t n = config_.num_envs;

  // Player buffer: one Player struct per env
  player_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(Player), BufferUsage::Storage | BufferUsage::TransferDst);

  // Input buffer: one InputState per env
  input_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(InputState), BufferUsage::Storage | BufferUsage::TransferDst);

  // Dragon buffer: one Dragon per env
  dragon_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(Dragon), BufferUsage::Storage | BufferUsage::TransferDst);

  // Crystal buffer: NUM_CRYSTALS per env
  crystal_buffer_ = buffer_mgr_->create_device_buffer(
      n * NUM_CRYSTALS * sizeof(Crystal),
      BufferUsage::Storage | BufferUsage::TransferDst);

  // Game state buffer
  game_state_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(GameState), BufferUsage::Storage | BufferUsage::TransferDst);

  // Observation buffer (output)
  observation_buffer_ = buffer_mgr_->create_device_buffer(
      n * OBSERVATION_SIZE * sizeof(float),
      BufferUsage::Storage | BufferUsage::TransferSrc);

  // Reward buffer (output)
  reward_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(float), BufferUsage::Storage | BufferUsage::TransferSrc);

  // Done buffer (output)
  done_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(uint32_t), // Use uint32 for alignment
      BufferUsage::Storage | BufferUsage::TransferSrc);
}

void MC189Simulator::step(const int32_t *actions, size_t num_actions) {
  if (num_actions != config_.num_envs) {
    throw std::invalid_argument("Action count must match num_envs");
  }

  // Clear done flags from previous step (user has seen them)
  std::fill(dones_.begin(), dones_.end(), 0);

  // Convert discrete actions to InputState
  // Dragon fight action space (15 actions):
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
      in.lookDeltaX = -5.0f;
      break;
    case 13: // look_right
      in.lookDeltaX = 5.0f;
      break;
    case 14: // swap_weapon
      in.action = 3;
      break;
    case 15: // look_up
      in.lookDeltaY = -5.0f;
      break;
    case 16: // look_down
      in.lookDeltaY = 5.0f;
      break;
    default:
      break; // noop
    }
  }

  // Upload inputs to GPU (skip in CPU mode)
  if (use_gpu_) {
    buffer_mgr_->upload(input_buffer_, inputs.data(),
                        inputs.size() * sizeof(InputState), 0);
  }

  // Dispatch simulation
  dispatch_tick();

  // Extract results
  extract_observations();
  compute_rewards();

  // Auto-reset done environments
  auto_reset_done_envs();

  tick_number_++;
}

void MC189Simulator::dispatch_tick() {
  if (!use_gpu_ || !setup_pipeline_) {
    // CPU backend - deterministic synthetic outputs driven by world seeds.
    for (uint32_t env_id = 0; env_id < config_.num_envs; ++env_id) {
      float *obs = observations_.data() + env_id * OBSERVATION_SIZE;
      fill_cpu_outputs(world_seeds_[env_id], obs, rewards_[env_id],
                       dones_[env_id]);
    }
    return;
  }

  // Allocate descriptor set and bind buffers
  auto desc_set = setup_pipeline_->allocate_descriptor_set();
  setup_pipeline_->update_descriptor(desc_set, 0, player_buffer_.handle());
  setup_pipeline_->update_descriptor(desc_set, 1, input_buffer_.handle());
  setup_pipeline_->update_descriptor(desc_set, 2, dragon_buffer_.handle());
  setup_pipeline_->update_descriptor(desc_set, 3, crystal_buffer_.handle());
  setup_pipeline_->update_descriptor(desc_set, 4, game_state_buffer_.handle());
  setup_pipeline_->update_descriptor(desc_set, 5, observation_buffer_.handle());
  setup_pipeline_->update_descriptor(desc_set, 6, reward_buffer_.handle());
  setup_pipeline_->update_descriptor(desc_set, 7, done_buffer_.handle());

  const uint32_t workgroups = (config_.num_envs + 63) / 64;

  std::random_device rd;
  const uint32_t random_seed = rd();

  auto cmd = ctx_->allocate_command_buffer();
  vk::CommandBufferBeginInfo begin_info{};
  begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  cmd.begin(begin_info);

  if (!shader_pipelines_.empty()) {
    // Stage-specific dispatch: run each shader in the shader_set sequentially
    uint32_t shader_idx = 0;
    for (const auto &shader_name : config_.shader_set) {
      auto it = shader_pipelines_.find(shader_name);
      if (it == shader_pipelines_.end()) {
        continue;
      }
      auto &pipeline = it->second;

      PushConstants pc{shader_idx, config_.num_envs, tick_number_, random_seed};

      pipeline->bind(cmd);
      pipeline->bind_descriptor_set(cmd, desc_set);
      pipeline->push_constants(cmd, pc);
      pipeline->dispatch(cmd, workgroups, 1, 1);

      // Barrier between shader dispatches
      vk::MemoryBarrier barrier{};
      barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
      cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eComputeShader, {},
                          barrier, {}, {});
      ++shader_idx;
    }
  } else {
    // Legacy path: dispatch monolithic shader with stage push constants
    for (uint32_t stage = 0; stage < 6; ++stage) {
      PushConstants pc{stage, config_.num_envs, tick_number_, random_seed};

      setup_pipeline_->bind(cmd);
      setup_pipeline_->bind_descriptor_set(cmd, desc_set);
      setup_pipeline_->push_constants(cmd, pc);
      setup_pipeline_->dispatch(cmd, workgroups, 1, 1);

      vk::MemoryBarrier barrier{};
      barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
      cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                          vk::PipelineStageFlagBits::eComputeShader, {},
                          barrier, {}, {});
    }
  }

  cmd.end();
  ctx_->submit_and_wait(cmd);
  setup_pipeline_->free_descriptor_set(desc_set);
}

void MC189Simulator::extract_observations() {
  if (use_gpu_ && setup_pipeline_) {
    // Download from GPU
    buffer_mgr_->download(observation_buffer_, observations_.data(),
                          observations_.size() * sizeof(float), 0);
  }
  // CPU backend already filled observations_ in dispatch_tick()
}

void MC189Simulator::compute_rewards() {
  if (use_gpu_ && setup_pipeline_) {
    // Download rewards and dones from GPU
    buffer_mgr_->download(reward_buffer_, rewards_.data(),
                          rewards_.size() * sizeof(float), 0);

    std::vector<uint32_t> done_u32(config_.num_envs);
    buffer_mgr_->download(done_buffer_, done_u32.data(),
                          done_u32.size() * sizeof(uint32_t), 0);

    for (uint32_t i = 0; i < config_.num_envs; ++i) {
      dones_[i] = done_u32[i] ? 1 : 0;
    }
  }
  // CPU backend already set these
}

void MC189Simulator::reset(uint32_t env_id, uint64_t seed) {
  // Handle seed - if 0, generate random seed
  if (seed == 0) {
    std::random_device rd;
    seed = (static_cast<uint64_t>(rd()) << 32) | rd();
  }

  if (env_id == 0xFFFFFFFF) {
    // Reset all environments with the same base seed (varied per env)
    for (uint32_t i = 0; i < config_.num_envs; ++i) {
      world_seeds_[i] = WorldSeed(seed + i);
    }
    // Reset all environments
    std::vector<Player> players(config_.num_envs);
    std::vector<Dragon> dragons(config_.num_envs);
    std::vector<Crystal> all_crystals(config_.num_envs * NUM_CRYSTALS);
    std::vector<GameState> states(config_.num_envs);

    for (uint32_t i = 0; i < config_.num_envs; ++i) {
      // Initialize player
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

      // Initialize dragon
      auto &d = dragons[i];
      std::memset(&d, 0, sizeof(Dragon));
      d.position[0] = 0.0f;
      d.position[1] = 70.0f;
      d.position[2] = 75.0f;
      d.health = DRAGON_MAX_HEALTH;
      d.phase = 0;                      // Circling
      d.circle_angle = float(i) * 0.1f; // Vary starting angle

      // Initialize crystals
      for (uint32_t j = 0; j < NUM_CRYSTALS; ++j) {
        auto &c = all_crystals[i * NUM_CRYSTALS + j];
        get_pillar_position(j, c.position);
        c.is_alive = 1.0f; // All crystals start alive
      }

      // Initialize game state
      auto &gs = states[i];
      std::memset(&gs, 0, sizeof(GameState));
      gs.deltaTime = 0.05f; // 20 TPS
    }

    tick_number_ = 0;
    std::fill(dones_.begin(), dones_.end(), 0);

    if (use_gpu_) {
      buffer_mgr_->upload(player_buffer_, players.data(),
                          players.size() * sizeof(Player), 0);
      buffer_mgr_->upload(dragon_buffer_, dragons.data(),
                          dragons.size() * sizeof(Dragon), 0);
      buffer_mgr_->upload(crystal_buffer_, all_crystals.data(),
                          all_crystals.size() * sizeof(Crystal), 0);
      buffer_mgr_->upload(game_state_buffer_, states.data(),
                          states.size() * sizeof(GameState), 0);

      // Clear dones buffer
      std::vector<uint32_t> zeros(config_.num_envs, 0);
      buffer_mgr_->upload(done_buffer_, zeros.data(),
                          zeros.size() * sizeof(uint32_t), 0);

      // Prime initial observations so reset() returns usable state.
      std::vector<InputState> inputs(config_.num_envs);
      std::memset(inputs.data(), 0, inputs.size() * sizeof(InputState));
      buffer_mgr_->upload(input_buffer_, inputs.data(),
                          inputs.size() * sizeof(InputState), 0);
    }

    dispatch_tick();
    extract_observations();
    compute_rewards();
    tick_number_ = 1;
  } else if (env_id < config_.num_envs) {
    // Reset single environment with seed
    world_seeds_[env_id] = WorldSeed(seed);

    Player p{};
    std::memset(&p, 0, sizeof(Player));
    p.position[1] = END_SPAWN_Y;
    p.health = 20.0f;
    p.hunger = 20.0f;
    p.saturation = 5.0f;
    p.weapon_slot = 1;
    p.arrows = 64;

    Dragon d{};
    std::memset(&d, 0, sizeof(Dragon));
    d.position[1] = 70.0f;
    d.position[2] = 75.0f;
    d.health = DRAGON_MAX_HEALTH;

    std::vector<Crystal> crystals(NUM_CRYSTALS);
    for (uint32_t j = 0; j < NUM_CRYSTALS; ++j) {
      get_pillar_position(j, crystals[j].position);
      crystals[j].is_alive = 1.0f;
    }

    if (use_gpu_) {
      buffer_mgr_->upload(player_buffer_, &p, sizeof(Player),
                          env_id * sizeof(Player));
      buffer_mgr_->upload(dragon_buffer_, &d, sizeof(Dragon),
                          env_id * sizeof(Dragon));
      buffer_mgr_->upload(crystal_buffer_, crystals.data(),
                          crystals.size() * sizeof(Crystal),
                          env_id * NUM_CRYSTALS * sizeof(Crystal));
    } else {
      float *obs = observations_.data() + env_id * OBSERVATION_SIZE;
      fill_cpu_outputs(world_seeds_[env_id], obs, rewards_[env_id],
                       dones_[env_id]);
    }

    dones_[env_id] = 0;
  }
}

const float *MC189Simulator::get_observations() const {
  return observations_.data();
}

const float *MC189Simulator::get_rewards() const { return rewards_.data(); }

const uint8_t *MC189Simulator::get_dones() const { return dones_.data(); }

void MC189Simulator::auto_reset_done_envs() {
  // Check for any done environments and reset them
  bool need_upload = false;

  std::vector<Player> players(config_.num_envs);
  std::vector<Dragon> dragons(config_.num_envs);
  std::vector<Crystal> all_crystals(config_.num_envs * NUM_CRYSTALS);

  // Only download if we have GPU pipeline
  if (use_gpu_ && setup_pipeline_) {
    buffer_mgr_->download(player_buffer_, players.data(),
                          players.size() * sizeof(Player), 0);
    buffer_mgr_->download(dragon_buffer_, dragons.data(),
                          dragons.size() * sizeof(Dragon), 0);
    buffer_mgr_->download(crystal_buffer_, all_crystals.data(),
                          all_crystals.size() * sizeof(Crystal), 0);
  }

  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    if (dones_[i]) {
      need_upload = true;

      // Reset player
      auto &p = players[i];
      std::memset(&p, 0, sizeof(Player));
      p.position[0] = 0.0f;
      p.position[1] = END_SPAWN_Y;
      p.position[2] = 0.0f;
      p.health = 20.0f;
      p.hunger = 20.0f;
      p.saturation = 5.0f;
      p.weapon_slot = 1;
      p.arrows = 64;

      // Reset dragon
      auto &d = dragons[i];
      std::memset(&d, 0, sizeof(Dragon));
      d.position[0] = 0.0f;
      d.position[1] = 70.0f;
      d.position[2] = 75.0f;
      d.health = DRAGON_MAX_HEALTH;
      d.phase = 0;
      d.circle_angle = float(i) * 0.1f;

      // Reset crystals
      for (uint32_t j = 0; j < NUM_CRYSTALS; ++j) {
        auto &c = all_crystals[i * NUM_CRYSTALS + j];
        get_pillar_position(j, c.position);
        c.is_alive = 1.0f;
      }

      // Clear done flag on GPU (will be set again if episode ends)
      // Note: we don't clear dones_[i] here - let the user see it
      // It will be cleared at the start of the next step
      if (use_gpu_) {
        uint32_t zero = 0;
        buffer_mgr_->upload(done_buffer_, &zero, sizeof(uint32_t),
                            i * sizeof(uint32_t));
      }
    }
  }

  if (need_upload && use_gpu_ && setup_pipeline_) {
    buffer_mgr_->upload(player_buffer_, players.data(),
                        players.size() * sizeof(Player), 0);
    buffer_mgr_->upload(dragon_buffer_, dragons.data(),
                        dragons.size() * sizeof(Dragon), 0);
    buffer_mgr_->upload(crystal_buffer_, all_crystals.data(),
                        all_crystals.size() * sizeof(Crystal), 0);

    // Clear dones buffer on GPU
    std::vector<uint32_t> zeros(config_.num_envs, 0);
    buffer_mgr_->upload(done_buffer_, zeros.data(),
                        zeros.size() * sizeof(uint32_t), 0);
  }
}

} // namespace mc189
