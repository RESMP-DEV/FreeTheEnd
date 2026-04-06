#include "mc189/portal_manager.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mc189 {

// Minecraft 1.8.9 portal constants
static constexpr float NETHER_SCALE = 8.0f;
static constexpr float PORTAL_COLLISION_RADIUS = 0.9f;
static constexpr float PORTAL_COLLISION_HEIGHT = 2.5f;
static constexpr uint32_t NETHER_PORTAL_WIDTH = 4;
static constexpr uint32_t NETHER_PORTAL_HEIGHT = 5;
static constexpr uint32_t END_PORTAL_FRAME_COUNT = 12;
static constexpr float END_PORTAL_Y = 0.75f;  // End portal frame height
static constexpr float NETHER_SPAWN_Y = 64.0f;
static constexpr float END_SPAWN_X = 100.5f;
static constexpr float END_SPAWN_Y = 49.0f;
static constexpr float END_SPAWN_Z = 0.5f;
static constexpr float OVERWORLD_RETURN_Y = 64.0f;

PortalManager::PortalManager(VulkanContext& ctx, BufferManager& buf_mgr,
                             const std::string& shader_dir, const Config& config)
    : ctx_(ctx), buf_mgr_(buf_mgr), config_(config) {
    portals_.resize(config_.num_envs);
    for (auto& env_portals : portals_) {
        env_portals.reserve(config_.max_portals_per_env);
    }

    // Allocate GPU buffer for portal state (aligned Portal structs)
    vk::DeviceSize buffer_size = config_.num_envs *
                                 config_.max_portals_per_env *
                                 sizeof(Portal);
    portal_buffer_ = buf_mgr_.create_mapped_buffer(
        buffer_size, BufferUsage::Storage | BufferUsage::TransferDst);

    // Zero-initialize the buffer
    auto* ptr = portal_buffer_.map();
    std::memset(ptr, 0, buffer_size);
    portal_buffer_.flush();

    load_shaders(shader_dir);
}

PortalManager::~PortalManager() = default;

void PortalManager::load_shaders(const std::string& shader_dir) {
    std::string spv_path = shader_dir + "/portal_tick.spv";

    // Try loading the shader; if it doesn't exist, portal_tick_pipeline_ stays null
    // and we rely on CPU-side tick logic only
    try {
        auto spirv = ComputePipeline::load_spirv(spv_path);
        if (spirv.empty()) return;

        ComputePipeline::Config cfg;
        cfg.spirv_code = std::move(spirv);
        cfg.entry_point = "main";
        cfg.bindings = {
            {0, vk::DescriptorType::eStorageBuffer},  // Portal buffer
            {1, vk::DescriptorType::eStorageBuffer},   // Player buffer
        };
        cfg.push_constants = {
            {0, sizeof(uint32_t) * 2}  // env_id, max_portals
        };
        cfg.local_size_x = 64;

        portal_tick_pipeline_ = std::make_unique<ComputePipeline>(ctx_, cfg);
    } catch (const std::exception&) {
        // Shader not available; CPU fallback only
    }
}

bool PortalManager::can_build_nether_portal(uint32_t env_id, float x, float y, float z) const {
    if (env_id >= config_.num_envs) return false;

    // Check we haven't exceeded portal limit
    if (portals_[env_id].size() >= config_.max_portals_per_env) return false;

    // In 1.8.9, nether portal requires:
    // - 4-wide, 5-tall obsidian frame (minimum)
    // - Y position between 1 and 126 (in either dimension)
    // - Not inside another portal's bounding box
    if (y < 1.0f || y > 126.0f) return false;

    // Check no existing nether portal overlaps this position
    for (const auto& portal : portals_[env_id]) {
        if (portal.type != PortalType::NETHER) continue;
        float dx = std::abs(portal.position[0] - x);
        float dy = std::abs(portal.position[1] - y);
        float dz = std::abs(portal.position[2] - z);
        if (dx < NETHER_PORTAL_WIDTH && dy < NETHER_PORTAL_HEIGHT && dz < 2.0f) {
            return false;
        }
    }

    return true;
}

bool PortalManager::build_nether_portal(uint32_t env_id, float x, float y, float z) {
    if (!can_build_nether_portal(env_id, x, y, z)) return false;

    uint32_t slot = find_portal_slot(env_id);
    if (slot == UINT32_MAX) return false;

    PortalState state{};
    state.position[0] = x;
    state.position[1] = y;
    state.position[2] = z;
    state.type = PortalType::NETHER;
    state.source_dim = Dimension::OVERWORLD;  // Default; gets set properly on light
    state.target_dim = Dimension::NETHER;
    state.active = false;  // Not active until lit
    state.frame_complete = NETHER_PORTAL_WIDTH * 2 + (NETHER_PORTAL_HEIGHT - 2) * 2;  // Full obsidian frame

    // Calculate target coords
    calculate_nether_coords(x, z, state.target_position[0], state.target_position[2]);
    state.target_position[1] = NETHER_SPAWN_Y;

    if (slot < portals_[env_id].size()) {
        portals_[env_id][slot] = state;
    } else {
        portals_[env_id].push_back(state);
    }

    return true;
}

bool PortalManager::light_nether_portal(uint32_t env_id, float x, float y, float z) {
    if (env_id >= config_.num_envs) return false;

    // Find the unlit nether portal at this position
    for (auto& portal : portals_[env_id]) {
        if (portal.type != PortalType::NETHER) continue;
        if (portal.active) continue;

        float dx = std::abs(portal.position[0] - x);
        float dy = std::abs(portal.position[1] - y);
        float dz = std::abs(portal.position[2] - z);

        if (dx < NETHER_PORTAL_WIDTH && dy < NETHER_PORTAL_HEIGHT && dz < 2.0f) {
            portal.active = true;
            sync_portal_to_gpu(env_id, portal);
            return true;
        }
    }

    // No frame found; build and light in one step (simplified for RL)
    if (build_nether_portal(env_id, x, y, z)) {
        portals_[env_id].back().active = true;
        sync_portal_to_gpu(env_id, portals_[env_id].back());
        return true;
    }

    return false;
}

uint32_t PortalManager::get_end_portal_eyes(uint32_t env_id, float x, float z) const {
    if (env_id >= config_.num_envs) return 0;

    for (const auto& portal : portals_[env_id]) {
        if (portal.type != PortalType::END) continue;
        float dx = std::abs(portal.position[0] - x);
        float dz = std::abs(portal.position[2] - z);
        // End portal frames are within a ~5 block radius
        if (dx < 5.0f && dz < 5.0f) {
            return portal.frame_complete;
        }
    }
    return 0;
}

bool PortalManager::place_eye_of_ender(uint32_t env_id, float x, float y, float z) {
    if (env_id >= config_.num_envs) return false;

    // Find existing end portal frame at this location
    for (auto& portal : portals_[env_id]) {
        if (portal.type != PortalType::END) continue;
        float dx = std::abs(portal.position[0] - x);
        float dz = std::abs(portal.position[2] - z);
        if (dx < 5.0f && dz < 5.0f) {
            if (portal.frame_complete >= END_PORTAL_FRAME_COUNT) return false;  // Already full
            portal.frame_complete++;
            if (portal.frame_complete >= END_PORTAL_FRAME_COUNT) {
                portal.active = true;
                portal.target_position[0] = END_SPAWN_X;
                portal.target_position[1] = END_SPAWN_Y;
                portal.target_position[2] = END_SPAWN_Z;
            }
            sync_portal_to_gpu(env_id, portal);
            return true;
        }
    }

    // No end portal frame exists here; create one with first eye placed
    uint32_t slot = find_portal_slot(env_id);
    if (slot == UINT32_MAX) return false;

    PortalState state{};
    state.position[0] = x;
    state.position[1] = y;
    state.position[2] = z;
    state.type = PortalType::END;
    state.source_dim = Dimension::OVERWORLD;
    state.target_dim = Dimension::END;
    state.active = false;
    state.frame_complete = 1;
    state.target_position[0] = END_SPAWN_X;
    state.target_position[1] = END_SPAWN_Y;
    state.target_position[2] = END_SPAWN_Z;

    if (slot < portals_[env_id].size()) {
        portals_[env_id][slot] = state;
    } else {
        portals_[env_id].push_back(state);
    }

    return true;
}

bool PortalManager::is_end_portal_active(uint32_t env_id, float x, float z) const {
    if (env_id >= config_.num_envs) return false;

    for (const auto& portal : portals_[env_id]) {
        if (portal.type != PortalType::END) continue;
        if (!portal.active) continue;
        float dx = std::abs(portal.position[0] - x);
        float dz = std::abs(portal.position[2] - z);
        if (dx < 5.0f && dz < 5.0f) return true;
    }
    return false;
}

PortalManager::TeleportResult
PortalManager::check_portal_collision(uint32_t env_id,
                                      float player_x, float player_y, float player_z,
                                      Dimension current_dim) {
    TeleportResult result{false, current_dim, player_x, player_y, player_z};
    if (env_id >= config_.num_envs) return result;

    for (const auto& portal : portals_[env_id]) {
        if (!portal.active) continue;

        // Check dimension match: portal source must match current dimension
        if (portal.source_dim != current_dim) continue;

        // AABB collision check
        float dx = std::abs(portal.position[0] - player_x);
        float dy = std::abs(portal.position[1] - player_y);
        float dz = std::abs(portal.position[2] - player_z);

        bool colliding = false;
        if (portal.type == PortalType::NETHER) {
            // Nether portal: thin rectangle, player must be inside the 2x3 interior
            colliding = (dx < PORTAL_COLLISION_RADIUS &&
                         dy < PORTAL_COLLISION_HEIGHT &&
                         dz < PORTAL_COLLISION_RADIUS);
        } else {
            // End portal: 3x3 horizontal surface at frame height
            colliding = (dx < 1.5f &&
                         dy < 1.0f &&
                         dz < 1.5f);
        }

        if (colliding) {
            result.success = true;
            result.new_dimension = portal.target_dim;
            result.new_x = portal.target_position[0];
            result.new_y = portal.target_position[1];
            result.new_z = portal.target_position[2];

            // For nether portals going back, create return portal if needed
            if (portal.type == PortalType::NETHER && current_dim == Dimension::NETHER) {
                // Returning to overworld: scale coords back
                result.new_dimension = Dimension::OVERWORLD;
                calculate_overworld_coords(player_x, player_z,
                                           result.new_x, result.new_z);
                result.new_y = OVERWORLD_RETURN_Y;
            }

            return result;
        }
    }

    return result;
}

void PortalManager::tick(uint32_t env_id, const Player& player, Dimension dim,
                         bool& out_teleport, Dimension& out_target_dim,
                         float& out_x, float& out_y, float& out_z) {
    out_teleport = false;

    auto result = check_portal_collision(env_id,
                                         player.position[0],
                                         player.position[1],
                                         player.position[2],
                                         dim);
    if (result.success) {
        out_teleport = true;
        out_target_dim = result.new_dimension;
        out_x = result.new_x;
        out_y = result.new_y;
        out_z = result.new_z;
    }
}

std::vector<const PortalState*> PortalManager::get_portals(uint32_t env_id) const {
    std::vector<const PortalState*> result;
    if (env_id >= config_.num_envs) return result;

    result.reserve(portals_[env_id].size());
    for (const auto& portal : portals_[env_id]) {
        result.push_back(&portal);
    }
    return result;
}

const PortalState* PortalManager::get_nearest_portal(uint32_t env_id,
                                                     float x, float z,
                                                     PortalType type) const {
    if (env_id >= config_.num_envs) return nullptr;

    const PortalState* nearest = nullptr;
    float best_dist_sq = std::numeric_limits<float>::max();

    for (const auto& portal : portals_[env_id]) {
        if (portal.type != type) continue;
        float dx = portal.position[0] - x;
        float dz = portal.position[2] - z;
        float dist_sq = dx * dx + dz * dz;
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            nearest = &portal;
        }
    }

    return nearest;
}

void PortalManager::clear(uint32_t env_id) {
    if (env_id >= config_.num_envs) return;
    portals_[env_id].clear();

    // Zero out the GPU buffer region for this env
    vk::DeviceSize env_offset = env_id * config_.max_portals_per_env * sizeof(Portal);
    vk::DeviceSize env_size = config_.max_portals_per_env * sizeof(Portal);

    auto* base = static_cast<uint8_t*>(portal_buffer_.map());
    std::memset(base + env_offset, 0, env_size);
    portal_buffer_.flush(env_offset, env_size);
}

uint32_t PortalManager::find_portal_slot(uint32_t env_id) {
    if (env_id >= config_.num_envs) return UINT32_MAX;
    auto& env_portals = portals_[env_id];

    if (env_portals.size() < config_.max_portals_per_env) {
        return static_cast<uint32_t>(env_portals.size());
    }

    // Look for an inactive portal to reuse
    for (uint32_t i = 0; i < env_portals.size(); ++i) {
        if (!env_portals[i].active) return i;
    }

    return UINT32_MAX;  // No slots available
}

void PortalManager::calculate_nether_coords(float overworld_x, float overworld_z,
                                            float& nether_x, float& nether_z) {
    // Overworld -> Nether: divide by 8
    nether_x = overworld_x / NETHER_SCALE;
    nether_z = overworld_z / NETHER_SCALE;

    // Clamp to nether world border (1.8.9: +/- 29,999,872 / 8 = 3,749,984)
    nether_x = std::clamp(nether_x, -3749984.0f, 3749984.0f);
    nether_z = std::clamp(nether_z, -3749984.0f, 3749984.0f);
}

void PortalManager::calculate_overworld_coords(float nether_x, float nether_z,
                                               float& overworld_x, float& overworld_z) {
    // Nether -> Overworld: multiply by 8
    overworld_x = nether_x * NETHER_SCALE;
    overworld_z = nether_z * NETHER_SCALE;
}

// Private helper: sync a single portal state to the GPU buffer
void PortalManager::sync_portal_to_gpu(uint32_t env_id, const PortalState& state) {
    // Find portal index within env
    uint32_t portal_idx = UINT32_MAX;
    for (uint32_t i = 0; i < portals_[env_id].size(); ++i) {
        if (&portals_[env_id][i] == &state) {
            portal_idx = i;
            break;
        }
    }
    if (portal_idx == UINT32_MAX) return;

    // Convert PortalState to GPU Portal struct
    Portal gpu_portal{};
    gpu_portal.position[0] = state.position[0];
    gpu_portal.position[1] = state.position[1];
    gpu_portal.position[2] = state.position[2];
    gpu_portal.type = static_cast<uint32_t>(state.type);
    gpu_portal.active = state.active ? 1u : 0u;
    gpu_portal.target_dimension = static_cast<uint32_t>(static_cast<int32_t>(state.target_dim));
    gpu_portal.target_position[0] = state.target_position[0];
    gpu_portal.target_position[1] = state.target_position[1];
    gpu_portal.target_position[2] = state.target_position[2];

    // Write to mapped buffer
    vk::DeviceSize offset = (env_id * config_.max_portals_per_env + portal_idx) * sizeof(Portal);
    auto* base = static_cast<uint8_t*>(portal_buffer_.map());
    std::memcpy(base + offset, &gpu_portal, sizeof(Portal));
    portal_buffer_.flush(offset, sizeof(Portal));
}

} // namespace mc189
