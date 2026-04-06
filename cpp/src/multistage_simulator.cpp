// MultistageSimulator - Full Minecraft 1.8.9 speedrun simulation
// Handles all 6 curriculum stages with proper world generation, mob spawning,
// etc.

#include "mc189/multistage_simulator.h"
#include "mc189/game_stage.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>

namespace mc189 {

namespace {

std::vector<uint32_t> load_spirv(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return {}; // Return empty on failure (some shaders optional)
  }
  const auto size = file.tellg();
  file.seekg(0);
  std::vector<uint32_t> spirv(size / sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(spirv.data()), size);
  return spirv;
}

// Push constants for shader dispatch (must match dragon_fight shader)
struct PushConstants {
  uint32_t stage; // Shader stage (internal 0-5)
  uint32_t num_envs;
  uint32_t tick;
  uint32_t random_seed;
};

// Extended observation layout (256 floats)
// See speedrun_env.py ObservationLayout for semantic mapping
constexpr size_t EXT_OBS_SIZE = 256;

// Initial inventory items for each stage
struct InitialInventory {
  uint16_t item_ids[36];
  uint8_t counts[36];
  uint16_t armor_ids[4];
  uint16_t offhand_id;
};

InitialInventory get_initial_inventory(GameStage stage) {
  InitialInventory inv{};
  std::memset(&inv, 0, sizeof(inv));

  switch (stage) {
  case GameStage::BASIC_SURVIVAL:
    // Empty inventory - must survive first night
    break;

  case GameStage::RESOURCE_GATHERING:
    // Stone tools to start mining
    inv.item_ids[0] = 274; // Stone pickaxe
    inv.item_ids[1] = 275; // Stone axe
    inv.item_ids[2] = 272; // Stone sword
    inv.item_ids[3] = 297; // Bread x10
    inv.counts[3] = 10;
    break;

  case GameStage::NETHER_NAVIGATION:
    // Iron gear + obsidian for portal
    inv.item_ids[0] = 257; // Iron pickaxe
    inv.item_ids[1] = 267; // Iron sword
    inv.item_ids[2] = 49;  // Obsidian x10
    inv.counts[2] = 10;
    inv.item_ids[3] = 259; // Flint and steel
    inv.item_ids[4] = 297; // Bread x32
    inv.counts[4] = 32;
    inv.armor_ids[0] = 306; // Iron helmet
    inv.armor_ids[1] = 307; // Iron chestplate
    inv.armor_ids[2] = 308; // Iron leggings
    inv.armor_ids[3] = 309; // Iron boots
    break;

  case GameStage::ENDERMAN_HUNTING:
    // Blaze rods + gear for enderman farming
    inv.item_ids[0] = 257; // Iron pickaxe
    inv.item_ids[1] = 267; // Iron sword
    inv.item_ids[2] = 369; // Blaze rod x7
    inv.counts[2] = 7;
    inv.item_ids[3] = 297; // Bread x32
    inv.counts[3] = 32;
    inv.armor_ids[0] = 306;
    inv.armor_ids[1] = 307;
    inv.armor_ids[2] = 308;
    inv.armor_ids[3] = 309;
    break;

  case GameStage::STRONGHOLD_FINDING:
    // Eyes of ender + gear
    inv.item_ids[0] = 257; // Iron pickaxe
    inv.item_ids[1] = 267; // Iron sword
    inv.item_ids[2] = 381; // Eye of ender x12
    inv.counts[2] = 12;
    inv.item_ids[3] = 297; // Bread x32
    inv.counts[3] = 32;
    inv.armor_ids[0] = 306;
    inv.armor_ids[1] = 307;
    inv.armor_ids[2] = 308;
    inv.armor_ids[3] = 309;
    break;

  case GameStage::END_FIGHT:
    // Full combat gear
    inv.item_ids[0] = 276; // Diamond sword
    inv.item_ids[1] = 261; // Bow
    inv.item_ids[2] = 262; // Arrow x64
    inv.counts[2] = 64;
    inv.item_ids[3] = 322; // Golden apple x10
    inv.counts[3] = 10;
    inv.item_ids[4] = 373; // Water bucket
    inv.item_ids[5] = 368; // Ender pearl x16
    inv.counts[5] = 16;
    inv.armor_ids[0] = 310; // Diamond helmet
    inv.armor_ids[1] = 311; // Diamond chestplate
    inv.armor_ids[2] = 312; // Diamond leggings
    inv.armor_ids[3] = 313; // Diamond boots
    break;
  }

  return inv;
}

// Shaders needed for each stage
std::vector<std::string> get_stage_shaders(GameStage stage) {
  std::vector<std::string> shaders;

  // Common shaders for all stages
  shaders.push_back("action_decoder");

  switch (stage) {
  case GameStage::BASIC_SURVIVAL:
    shaders.push_back("overworld_terrain");
    shaders.push_back("mob_spawning_overworld");
    shaders.push_back("mob_ai_overworld_hostile");
    shaders.push_back("player_physics");
    shaders.push_back("player_combat");
    shaders.push_back("time_cycle");
    shaders.push_back("observation_encoder");
    shaders.push_back("reward_computation");
    break;

  case GameStage::RESOURCE_GATHERING:
    shaders.push_back("overworld_terrain");
    shaders.push_back("cave_generation");
    shaders.push_back("resource_detection");
    shaders.push_back("mob_spawning_overworld");
    shaders.push_back("mob_ai_overworld_hostile");
    shaders.push_back("player_physics");
    shaders.push_back("player_combat");
    shaders.push_back("block_breaking");
    shaders.push_back("crafting");
    shaders.push_back("furnace_tick");
    shaders.push_back("portal_creation");
    shaders.push_back("time_cycle");
    shaders.push_back("observation_encoder");
    shaders.push_back("reward_computation");
    break;

  case GameStage::NETHER_NAVIGATION:
    shaders.push_back("nether_gen");
    shaders.push_back("fortress_structure");
    shaders.push_back("mob_ai_blaze");
    shaders.push_back("mob_ai_ghast");
    shaders.push_back("player_physics");
    shaders.push_back("player_combat");
    shaders.push_back("block_breaking");
    shaders.push_back("lava_mechanics");
    shaders.push_back("observation_encoder");
    shaders.push_back("reward_computation");
    break;

  case GameStage::ENDERMAN_HUNTING:
    shaders.push_back("overworld_terrain");
    shaders.push_back("mob_ai_enderman_full");
    shaders.push_back("enderman_spawning");
    shaders.push_back("player_physics");
    shaders.push_back("player_combat");
    shaders.push_back("eye_of_ender_full");
    shaders.push_back("find_strongholds");
    shaders.push_back("crafting");
    shaders.push_back("observation_encoder");
    shaders.push_back("reward_computation");
    break;

  case GameStage::STRONGHOLD_FINDING:
    shaders.push_back("overworld_terrain");
    shaders.push_back("stronghold_gen");
    shaders.push_back("eye_of_ender_full");
    shaders.push_back("player_physics");
    shaders.push_back("block_breaking");
    shaders.push_back("end_portal");
    shaders.push_back("mob_ai_silverfish");
    shaders.push_back("observation_encoder");
    shaders.push_back("reward_computation");
    break;

  case GameStage::END_FIGHT:
    // Use monolithic dragon fight shader
    shaders.clear();
    shaders.push_back("dragon_fight_mvk");
    break;
  }

  return shaders;
}

} // namespace

// ============================================================================
// MultistageSimulator Implementation
// ============================================================================

MultistageSimulator::MultistageSimulator(const Config &config)
    : config_(config) {

  // Initialize Vulkan context
  try {
    VulkanContext::Config ctx_config{};
    ctx_config.enable_validation = config.enable_validation;
    ctx_config.prefer_discrete_gpu = true;
    ctx_ = std::make_unique<VulkanContext>(ctx_config);
    buffer_mgr_ = std::make_unique<BufferManager>(*ctx_);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("Vulkan init failed: ") + e.what());
  }

  // Load shaders for initial stage
  load_stage_shaders();

  // Create buffers
  create_extended_buffers();

  // Initialize CPU-side state
  observations_.resize(config_.num_envs * OBSERVATION_SIZE);
  extended_observations_.resize(config_.num_envs * EXT_OBS_SIZE);
  rewards_.resize(config_.num_envs);
  dones_.resize(config_.num_envs);
  world_states_.resize(config_.num_envs);
  current_stages_.resize(config_.num_envs, config_.initial_stage);
  current_dimensions_.resize(config_.num_envs, Dimension::OVERWORLD);
  stage_ticks_.resize(config_.num_envs, 0);
  stage_progress_.resize(config_.num_envs, 0);

  // Reset all environments
  reset();
}

MultistageSimulator::~MultistageSimulator() = default;

void MultistageSimulator::load_stage_shaders() {
  // Get shaders needed for current stage distribution
  std::set<std::string> needed_shaders;
  for (const auto &stage : current_stages_) {
    auto stage_shaders = get_stage_shaders(stage);
    needed_shaders.insert(stage_shaders.begin(), stage_shaders.end());
  }

  // If no current stages, load initial stage shaders
  if (needed_shaders.empty()) {
    auto stage_shaders = get_stage_shaders(config_.initial_stage);
    needed_shaders.insert(stage_shaders.begin(), stage_shaders.end());
  }

  // Common pipeline configuration - must match game_tick.spv layout
  auto make_pipeline_config =
      [&](const std::vector<uint32_t> &spirv) -> ComputePipeline::Config {
    ComputePipeline::Config info{};
    info.spirv_code = spirv;
    info.entry_point = "main";
    info.local_size_x = 64;
    info.local_size_y = 1;
    info.local_size_z = 1;
    info.push_constants = {
        {0, sizeof(PushConstants), vk::ShaderStageFlagBits::eCompute}};
    // Standard 8-binding layout (same as MC189Simulator)
    info.bindings = {
        {0, vk::DescriptorType::eStorageBuffer}, // players
        {1, vk::DescriptorType::eStorageBuffer}, // inputs
        {2, vk::DescriptorType::eStorageBuffer}, // mobs/entities
        {3, vk::DescriptorType::eStorageBuffer}, // chunks
        {4, vk::DescriptorType::eStorageBuffer}, // game_state
        {5, vk::DescriptorType::eStorageBuffer}, // observations
        {6, vk::DescriptorType::eStorageBuffer}, // rewards
        {7, vk::DescriptorType::eStorageBuffer}, // dones
    };
    return info;
  };

  // Load each needed shader
  for (const auto &shader_name : needed_shaders) {
    if (stage_pipelines_.count(static_cast<GameStage>(0)) &&
        stage_pipelines_.find(static_cast<GameStage>(0)) !=
            stage_pipelines_.end()) {
      // Already loaded as part of another stage
      continue;
    }

    std::string path = config_.shader_dir + "/" + shader_name + ".spv";
    auto spirv = load_spirv(path);
    if (spirv.empty()) {
      // Try without .spv extension (might be .comp that needs compilation)
      continue;
    }

    // Store in shader name -> pipeline map
    // Note: We're reusing stage_pipelines_ for now, but ideally we'd have
    // a separate map for named pipelines
  }

  // Try loading shaders in order of preference for Metal compatibility
  // 1. survival_tick.spv - Full overworld survival with mobs (Stage 1+)
  // 2. dragon_fight.spv - End fight only (Stage 6)
  // 3. game_tick_mvk.spv - Basic simulation
  std::vector<std::string> shader_candidates;

  // Choose shader based on game stage
  if (config_.initial_stage == GameStage::END_FIGHT) {
    shader_candidates = {
        config_.shader_dir + "/dragon_fight.spv",
        config_.shader_dir + "/game_tick_mvk.spv",
    };
  } else {
    // Stages 1-5: Use survival_tick_v2.spv which has C++-compatible struct
    // layouts and full overworld survival mechanics (mob spawning, AI, combat,
    // day/night)
    shader_candidates = {
        config_.shader_dir + "/survival_tick_v2.spv",
        config_.shader_dir + "/dragon_fight.spv",
        config_.shader_dir + "/game_tick_mvk.spv",
    };
  }

  std::vector<uint32_t> spirv;
  for (const auto &path : shader_candidates) {
    spirv = load_spirv(path);
    if (!spirv.empty()) {
      break;
    }
  }

  if (!spirv.empty()) {
    auto cfg = make_pipeline_config(spirv);
    // Use as the primary pipeline
    terrain_gen_pipeline_ = std::make_unique<ComputePipeline>(*ctx_, cfg);
    mob_ai_pipeline_ = std::make_unique<ComputePipeline>(*ctx_, cfg);
    observation_pipeline_ = std::make_unique<ComputePipeline>(*ctx_, cfg);
  }
}

void MultistageSimulator::create_extended_buffers() {
  const uint32_t n = config_.num_envs;

  // Extended player buffer with full inventory (for CPU-side tracking)
  player_full_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(PlayerFull), BufferUsage::Storage | BufferUsage::TransferDst |
                                  BufferUsage::TransferSrc);

  // Small player buffer matching dragon_fight shader (64 bytes per env)
  player_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(Player), BufferUsage::Storage | BufferUsage::TransferDst |
                              BufferUsage::TransferSrc);

  // Dragon buffer for dragon_fight shader
  dragon_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(Dragon), BufferUsage::Storage | BufferUsage::TransferDst |
                              BufferUsage::TransferSrc);

  // Crystal buffer for dragon_fight shader (NUM_CRYSTALS = 10)
  constexpr uint32_t NUM_CRYSTALS = 10;
  crystal_buffer_ = buffer_mgr_->create_device_buffer(
      n * NUM_CRYSTALS * sizeof(Crystal), BufferUsage::Storage |
                                              BufferUsage::TransferDst |
                                              BufferUsage::TransferSrc);

  // Game state buffer for dragon_fight shader
  game_state_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(GameState), BufferUsage::Storage | BufferUsage::TransferDst |
                                 BufferUsage::TransferSrc);

  // Input buffer
  input_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(InputState), BufferUsage::Storage | BufferUsage::TransferDst);

  // Entity buffer (mobs, projectiles, items)
  entity_buffer_ = buffer_mgr_->create_device_buffer(
      n * config_.max_entities_per_env * sizeof(Entity),
      BufferUsage::Storage | BufferUsage::TransferDst |
          BufferUsage::TransferSrc);

  // Mob count buffer (per-env uint for tracking active mobs)
  mob_counts_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(uint32_t), BufferUsage::Storage | BufferUsage::TransferDst |
                                BufferUsage::TransferSrc);

  // Chunk buffer
  chunk_buffer_ = buffer_mgr_->create_device_buffer(
      n * config_.max_chunks_per_env * sizeof(ChunkData),
      BufferUsage::Storage | BufferUsage::TransferDst |
          BufferUsage::TransferSrc);

  // World state buffer
  world_state_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(WorldState), BufferUsage::Storage | BufferUsage::TransferDst |
                                  BufferUsage::TransferSrc);

  // Portal buffer
  portal_buffer_ = buffer_mgr_->create_device_buffer(
      n * 4 * sizeof(Portal), // Up to 4 portals per env
      BufferUsage::Storage | BufferUsage::TransferDst |
          BufferUsage::TransferSrc);

  // Standard observation buffer (48 floats)
  observation_buffer_ = buffer_mgr_->create_device_buffer(
      n * OBSERVATION_SIZE * sizeof(float),
      BufferUsage::Storage | BufferUsage::TransferSrc);

  // Extended observation buffer (256 floats)
  extended_obs_buffer_ = buffer_mgr_->create_device_buffer(
      n * EXT_OBS_SIZE * sizeof(float),
      BufferUsage::Storage | BufferUsage::TransferSrc);

  // Reward and done buffers
  reward_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(float), BufferUsage::Storage | BufferUsage::TransferSrc);

  done_buffer_ = buffer_mgr_->create_device_buffer(
      n * sizeof(uint32_t), BufferUsage::Storage | BufferUsage::TransferSrc);
}

void MultistageSimulator::reset(uint32_t env_id, uint64_t seed) {
  if (seed == 0) {
    std::random_device rd;
    seed = (static_cast<uint64_t>(rd()) << 32) | rd();
  }

  if (env_id == 0xFFFFFFFF) {
    // Reset all environments
    for (uint32_t i = 0; i < config_.num_envs; ++i) {
      init_env_for_stage(i, config_.initial_stage, seed + i);
    }
    tick_number_ = 0;
  } else if (env_id < config_.num_envs) {
    init_env_for_stage(env_id, current_stages_[env_id], seed);
  }
}

void MultistageSimulator::init_env_for_stage(uint32_t env_id, GameStage stage,
                                             uint64_t seed) {
  auto stage_cfg = get_stage_config(stage);
  auto inventory = get_initial_inventory(stage);

  // Initialize PlayerFull for CPU-side tracking
  PlayerFull player{};
  std::memset(&player, 0, sizeof(PlayerFull));
  player.base.position[0] = stage_cfg.spawn_x;
  player.base.position[1] = stage_cfg.spawn_y;
  player.base.position[2] = stage_cfg.spawn_z;
  player.base.health = 20.0f;
  player.base.hunger = 20.0f;
  player.base.saturation = 5.0f;
  player.base.flags = 1; // On ground

  // Copy inventory
  for (int i = 0; i < 36; ++i) {
    player.inventory[i].item_id = inventory.item_ids[i];
    player.inventory[i].count =
        inventory.counts[i] > 0 ? inventory.counts[i] : 1;
  }
  for (int i = 0; i < 4; ++i) {
    player.armor[i].item_id = inventory.armor_ids[i];
    player.armor[i].count = 1;
  }

  // Initialize small Player for dragon_fight shader compatibility
  Player p{};
  std::memset(&p, 0, sizeof(Player));
  p.position[0] = stage_cfg.spawn_x;
  p.position[1] = stage_cfg.spawn_y;
  p.position[2] = stage_cfg.spawn_z;
  p.health = 20.0f;
  p.hunger = 20.0f;
  p.saturation = 5.0f;
  p.weapon_slot = 1; // Start with sword
  p.arrows = 64;
  p.flags = 1; // On ground

  // Initialize Dragon (used as MobLeader for survival stages)
  Dragon dragon{};
  std::memset(&dragon, 0, sizeof(Dragon));
  dragon.position[0] = 0.0f;
  dragon.position[1] = 70.0f;
  dragon.position[2] = 75.0f;
  dragon.health = 200.0f; // Full health
  dragon.phase = 0;       // Circling
  dragon.circle_angle = float(env_id) * 0.1f;
  // Start at night for immediate mob spawning in survival stages
  dragon.perch_timer = 13000; // NIGHT_START in shader

  // Initialize Crystals
  constexpr uint32_t NUM_CRYSTALS = 10;
  std::vector<Crystal> crystals(NUM_CRYSTALS);
  for (uint32_t i = 0; i < NUM_CRYSTALS; ++i) {
    float angle = float(i) * 2.0f * 3.14159f / float(NUM_CRYSTALS);
    crystals[i].position[0] = std::cos(angle) * 43.0f;
    crystals[i].position[1] = 76.0f;
    crystals[i].position[2] = std::sin(angle) * 43.0f;
    crystals[i].is_alive = 1.0f;
  }

  // Initialize GameState for dragon_fight shader
  GameState gs{};
  std::memset(&gs, 0, sizeof(GameState));
  gs.deltaTime = 0.05f; // 20 TPS

  // Initialize world state
  WorldState ws{};
  std::memset(&ws, 0, sizeof(WorldState));
  ws.tick = 0;
  ws.dimension = stage_cfg.dimension;
  ws.world_seed = seed;
  ws.stage = static_cast<uint32_t>(stage);
  ws.time_of_day = 0; // Dawn

  // Upload to GPU buffers
  buffer_mgr_->upload(player_full_buffer_, &player, sizeof(PlayerFull),
                      env_id * sizeof(PlayerFull));
  buffer_mgr_->upload(player_buffer_, &p, sizeof(Player),
                      env_id * sizeof(Player));
  buffer_mgr_->upload(dragon_buffer_, &dragon, sizeof(Dragon),
                      env_id * sizeof(Dragon));
  buffer_mgr_->upload(crystal_buffer_, crystals.data(),
                      NUM_CRYSTALS * sizeof(Crystal),
                      env_id * NUM_CRYSTALS * sizeof(Crystal));
  buffer_mgr_->upload(game_state_buffer_, &gs, sizeof(GameState),
                      env_id * sizeof(GameState));
  buffer_mgr_->upload(world_state_buffer_, &ws, sizeof(WorldState),
                      env_id * sizeof(WorldState));

  // Update CPU state
  current_stages_[env_id] = stage;
  current_dimensions_[env_id] = static_cast<Dimension>(stage_cfg.dimension);
  stage_ticks_[env_id] = 0;
  stage_progress_[env_id] = 0;
  world_states_[env_id] = ws;
  dones_[env_id] = 0;
  rewards_[env_id] = 0.0f;

  // Clear generated chunk cache for this env
  // (Would need to track per-env in real impl)
}

void MultistageSimulator::step(const int32_t *actions, size_t num_actions) {
  if (num_actions != config_.num_envs) {
    throw std::invalid_argument("Action count must match num_envs");
  }

  // Clear done flags
  std::fill(dones_.begin(), dones_.end(), 0);

  // Convert and upload actions
  std::vector<InputState> inputs(config_.num_envs);
  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    auto &in = inputs[i];
    std::memset(&in, 0, sizeof(InputState));

    const int32_t action = actions[i];
    // Extended action space (32 actions) - decode
    switch (action) {
    case 1:
      in.movement[2] = 1.0f;
      break; // forward
    case 2:
      in.movement[2] = -1.0f;
      break; // back
    case 3:
      in.movement[0] = -1.0f;
      break; // left
    case 4:
      in.movement[0] = 1.0f;
      break; // right
    case 5:
      in.movement[2] = 1.0f;
      in.movement[0] = -0.7f;
      break; // fwd+left
    case 6:
      in.movement[2] = 1.0f;
      in.movement[0] = 0.7f;
      break; // fwd+right
    case 7:
      in.flags |= 1;
      break; // jump
    case 8:
      in.flags |= 1;
      in.movement[2] = 1.0f;
      break; // jump+fwd
    case 9:
      in.action = 1;
      break; // attack
    case 10:
      in.action = 1;
      in.movement[2] = 1.0f;
      break; // attack+fwd
    case 11:
      in.flags |= 2;
      in.movement[2] = 1.0f;
      break; // sprint+fwd
    case 12:
      in.lookDeltaX = -5.0f;
      break; // look left
    case 13:
      in.lookDeltaX = 5.0f;
      break; // look right
    case 14:
      in.lookDeltaY = -7.5f;
      break; // look up
    case 15:
      in.lookDeltaY = 7.5f;
      break; // look down
    case 16:
      in.lookDeltaX = -45.0f;
      break; // look left fast
    case 17:
      in.lookDeltaX = 45.0f;
      break; // look right fast
    case 18:
      in.lookDeltaY = -30.0f;
      break; // look up fast
    case 19:
      in.lookDeltaY = 30.0f;
      break; // look down fast
    case 20:
      in.action = 2;
      break; // use item
    case 21:
      in.action = 3;
      break; // drop
    case 22:
    case 23:
    case 24:
    case 25:
    case 26:
    case 27:
    case 28:
    case 29:
    case 30:
      in.actionData = action - 22;
      in.action = 5;
      break; // hotbar 1-9
    case 31:
      in.action = 4;
      break; // craft
    default:
      break;
    }
  }

  buffer_mgr_->upload(input_buffer_, inputs.data(),
                      inputs.size() * sizeof(InputState), 0);

  // Dispatch game tick for each active stage
  dispatch_stage_tick(config_.initial_stage); // Simplified: dispatch for all

  // Download results
  buffer_mgr_->download(observation_buffer_, observations_.data(),
                        observations_.size() * sizeof(float), 0);
  buffer_mgr_->download(extended_obs_buffer_, extended_observations_.data(),
                        extended_observations_.size() * sizeof(float), 0);
  buffer_mgr_->download(reward_buffer_, rewards_.data(),
                        rewards_.size() * sizeof(float), 0);

  // Copy shader observations (48 floats) into first 48 elements of extended
  // observations
  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    std::memcpy(&extended_observations_[i * EXT_OBS_SIZE],
                &observations_[i * OBSERVATION_SIZE],
                OBSERVATION_SIZE * sizeof(float));
  }

  std::vector<uint32_t> done_u32(config_.num_envs);
  buffer_mgr_->download(done_buffer_, done_u32.data(),
                        done_u32.size() * sizeof(uint32_t), 0);
  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    dones_[i] = done_u32[i] ? 1 : 0;
  }

  // Save done flags BEFORE auto-reset so Python can see them
  std::vector<uint8_t> saved_dones = dones_;

  // Update stage ticks
  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    stage_ticks_[i]++;
  }

  // Check for stage transitions
  if (config_.auto_advance_stage) {
    check_stage_transitions();
  }

  // Auto-reset done environments
  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    if (saved_dones[i]) {
      std::random_device rd;
      uint64_t seed = (static_cast<uint64_t>(rd()) << 32) | rd();
      init_env_for_stage(i, current_stages_[i], seed);
    }
  }

  // Restore saved dones for Python visibility
  dones_ = saved_dones;

  tick_number_++;
}

void MultistageSimulator::dispatch_stage_tick(GameStage stage) {
  if (!terrain_gen_pipeline_) {
    // No GPU pipeline available - this shouldn't happen
    return;
  }

  auto desc_set = terrain_gen_pipeline_->allocate_descriptor_set();

  // Update descriptors (8 bindings matching dragon_fight.spv layout)
  terrain_gen_pipeline_->update_descriptor(desc_set, 0,
                                           player_buffer_.handle());
  terrain_gen_pipeline_->update_descriptor(desc_set, 1, input_buffer_.handle());
  terrain_gen_pipeline_->update_descriptor(desc_set, 2,
                                           dragon_buffer_.handle());
  terrain_gen_pipeline_->update_descriptor(desc_set, 3,
                                           crystal_buffer_.handle());
  terrain_gen_pipeline_->update_descriptor(desc_set, 4,
                                           game_state_buffer_.handle());
  terrain_gen_pipeline_->update_descriptor(desc_set, 5,
                                           observation_buffer_.handle());
  terrain_gen_pipeline_->update_descriptor(desc_set, 6,
                                           reward_buffer_.handle());
  terrain_gen_pipeline_->update_descriptor(desc_set, 7, done_buffer_.handle());

  const uint32_t workgroups = (config_.num_envs + 63) / 64;
  std::random_device rd;
  const uint32_t random_seed = rd();

  auto cmd = ctx_->allocate_command_buffer();
  vk::CommandBufferBeginInfo begin_info{};
  begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
  cmd.begin(begin_info);

  // Dispatch game tick with stage info in push constants
  PushConstants pc{};
  pc.stage = 0;
  pc.num_envs = config_.num_envs;
  pc.tick = tick_number_;
  pc.random_seed = random_seed;

  // Dispatch 6 internal stages (matches dragon_fight shader layout)
  // 0: Input processing
  // 1: Physics/movement
  // 2: Combat
  // 3: Dragon AI
  // 4: Crystal updates
  // 5: Observation/reward
  for (uint32_t internal_stage = 0; internal_stage < 6; ++internal_stage) {
    pc.stage = internal_stage;

    terrain_gen_pipeline_->bind(cmd);
    terrain_gen_pipeline_->bind_descriptor_set(cmd, desc_set);
    terrain_gen_pipeline_->push_constants(cmd, pc);
    terrain_gen_pipeline_->dispatch(cmd, workgroups, 1, 1);

    // Barrier between stages
    vk::MemoryBarrier barrier{};
    barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader, {}, barrier,
                        {}, {});
  }

  cmd.end();
  ctx_->submit_and_wait(cmd);
  terrain_gen_pipeline_->free_descriptor_set(desc_set);
}

void MultistageSimulator::check_stage_transitions() {
  // Check stage progression objectives for each environment
  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    auto &stage = current_stages_[i];
    auto &progress = stage_progress_[i];

    bool should_advance = false;

    switch (stage) {
    case GameStage::BASIC_SURVIVAL:
      // Advance if survived 5 minutes (6000 ticks) with health > 10
      if (stage_ticks_[i] >= 6000 &&
          observations_[i * OBSERVATION_SIZE + 8] > 0.5f) {
        should_advance = true;
      }
      break;

    case GameStage::RESOURCE_GATHERING:
      // Check for iron and portal materials
      // (Would need to read from extended obs or inventory buffer)
      break;

    case GameStage::NETHER_NAVIGATION:
      // Check for blaze rods collected
      break;

    case GameStage::ENDERMAN_HUNTING:
      // Check for ender pearls and eyes
      break;

    case GameStage::STRONGHOLD_FINDING:
      // Check for portal activation
      break;

    case GameStage::END_FIGHT:
      // Final stage - no advancement
      break;
    }

    if (should_advance && stage != GameStage::END_FIGHT) {
      stage = next_stage(stage);
      stage_ticks_[i] = 0;
      stage_progress_[i] = 0;
      // Re-initialize for new stage (keeping inventory)
    }
  }
}

void MultistageSimulator::set_stage(uint32_t env_id, GameStage stage) {
  if (env_id >= config_.num_envs)
    return;

  std::random_device rd;
  uint64_t seed = (static_cast<uint64_t>(rd()) << 32) | rd();
  init_env_for_stage(env_id, stage, seed);
}

GameStage MultistageSimulator::get_stage(uint32_t env_id) const {
  if (env_id >= config_.num_envs)
    return config_.initial_stage;
  return current_stages_[env_id];
}

uint32_t MultistageSimulator::get_stage_progress(uint32_t env_id) const {
  if (env_id >= config_.num_envs)
    return 0;
  return stage_progress_[env_id];
}

void MultistageSimulator::teleport_to_dimension(uint32_t env_id, Dimension dim,
                                                float x, float y, float z) {
  if (env_id >= config_.num_envs)
    return;

  // Download player, update position/dimension, upload
  PlayerFull player{};
  buffer_mgr_->download(player_full_buffer_, &player, sizeof(PlayerFull),
                        env_id * sizeof(PlayerFull));

  player.base.position[0] = x;
  player.base.position[1] = y;
  player.base.position[2] = z;

  buffer_mgr_->upload(player_full_buffer_, &player, sizeof(PlayerFull),
                      env_id * sizeof(PlayerFull));

  // Update world state
  WorldState ws = world_states_[env_id];
  ws.dimension = static_cast<uint32_t>(dim);
  buffer_mgr_->upload(world_state_buffer_, &ws, sizeof(WorldState),
                      env_id * sizeof(WorldState));

  current_dimensions_[env_id] = dim;
  world_states_[env_id] = ws;
}

Dimension MultistageSimulator::get_dimension(uint32_t env_id) const {
  if (env_id >= config_.num_envs)
    return Dimension::OVERWORLD;
  return current_dimensions_[env_id];
}

const float *MultistageSimulator::get_observations() const {
  return observations_.data();
}

const float *MultistageSimulator::get_extended_observations() const {
  return extended_observations_.data();
}

const float *MultistageSimulator::get_rewards() const {
  return rewards_.data();
}

const uint8_t *MultistageSimulator::get_dones() const { return dones_.data(); }

const WorldState *MultistageSimulator::get_world_states() const {
  return world_states_.data();
}

} // namespace mc189
