#pragma once

#include "mc189/buffer_manager.h"
#include "mc189/compute_pipeline.h"
#include "mc189/dimension.h"
#include "mc189/simulator.h"
#include "mc189/vulkan_context.h"

#include <memory>
#include <vector>

namespace mc189 {

enum class PortalType : uint32_t {
    NETHER = 0,
    END = 1
};

struct PortalState {
    float position[3];
    PortalType type;
    Dimension source_dim;
    Dimension target_dim;
    bool active;
    uint32_t frame_complete;  // For end portal (12 eyes needed)
    float target_position[3];
};

class PortalManager {
public:
    struct Config {
        uint32_t num_envs = 1;
        uint32_t max_portals_per_env = 16;
    };

    PortalManager(VulkanContext& ctx, BufferManager& buf_mgr,
                  const std::string& shader_dir, const Config& config);
    ~PortalManager();

    // Nether portal
    bool can_build_nether_portal(uint32_t env_id, float x, float y, float z) const;
    bool build_nether_portal(uint32_t env_id, float x, float y, float z);
    bool light_nether_portal(uint32_t env_id, float x, float y, float z);

    // End portal
    uint32_t get_end_portal_eyes(uint32_t env_id, float x, float z) const;
    bool place_eye_of_ender(uint32_t env_id, float x, float y, float z);
    bool is_end_portal_active(uint32_t env_id, float x, float z) const;

    // Portal interaction
    struct TeleportResult {
        bool success;
        Dimension new_dimension;
        float new_x, new_y, new_z;
    };
    TeleportResult check_portal_collision(uint32_t env_id,
                                          float player_x, float player_y, float player_z,
                                          Dimension current_dim);

    // Tick portals (check for player entry)
    void tick(uint32_t env_id, const Player& player, Dimension dim,
              bool& out_teleport, Dimension& out_target_dim,
              float& out_x, float& out_y, float& out_z);

    // Query
    std::vector<const PortalState*> get_portals(uint32_t env_id) const;
    const PortalState* get_nearest_portal(uint32_t env_id,
                                          float x, float z,
                                          PortalType type) const;

    // Buffer for shaders
    Buffer& get_portal_buffer() { return portal_buffer_; }

    // Reset
    void clear(uint32_t env_id);

private:
    void load_shaders(const std::string& shader_dir);
    uint32_t find_portal_slot(uint32_t env_id);
    void calculate_nether_coords(float overworld_x, float overworld_z,
                                 float& nether_x, float& nether_z);
    void calculate_overworld_coords(float nether_x, float nether_z,
                                    float& overworld_x, float& overworld_z);
    void sync_portal_to_gpu(uint32_t env_id, const PortalState& state);

    VulkanContext& ctx_;
    BufferManager& buf_mgr_;
    Config config_;

    std::unique_ptr<ComputePipeline> portal_tick_pipeline_;

    Buffer portal_buffer_;
    std::vector<std::vector<PortalState>> portals_;
};

} // namespace mc189
