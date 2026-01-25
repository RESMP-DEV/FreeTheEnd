#pragma once
#include "mc189/simulator.h"
#include "mc189/dimension.h"
#include "mc189/vulkan_context.h"
#include "mc189/buffer_manager.h"
#include <vector>
#include <memory>

namespace mc189 {

class EntityManager {
public:
    struct Config {
        uint32_t max_entities_per_env = 256;
        uint32_t num_envs = 1;
        float despawn_distance = 128.0f;
        float spawn_radius = 24.0f;
        uint32_t spawn_cooldown = 400;  // ticks between spawn attempts
    };

    EntityManager(VulkanContext& ctx, BufferManager& buf_mgr,
                 const std::string& shader_dir, const Config& config);
    ~EntityManager();

    // Spawning
    uint32_t spawn_entity(uint32_t env_id, EntityType type,
                         float x, float y, float z);
    void spawn_natural_mobs(uint32_t env_id, Dimension dim,
                           float player_x, float player_y, float player_z,
                           uint32_t tick);

    // Despawning
    void despawn_entity(uint32_t env_id, uint32_t entity_id);
    void despawn_distant(uint32_t env_id, float player_x, float player_z);
    void clear_all(uint32_t env_id);

    // AI tick (dispatches mob_ai shader)
    void tick_ai(uint32_t env_id, const Player& player, uint32_t tick);
    void tick_all_ai(const std::vector<Player>& players, uint32_t tick);

    // Physics tick
    void tick_physics(uint32_t env_id);

    // Combat
    struct HitResult {
        uint32_t entity_id;
        float damage;
        bool killed;
    };
    std::vector<HitResult> player_attack(uint32_t env_id,
                                         float px, float py, float pz,
                                         float yaw, float pitch,
                                         float reach, float damage);

    // Projectiles
    uint32_t shoot_arrow(uint32_t env_id, float x, float y, float z,
                        float vx, float vy, float vz, float damage);
    uint32_t throw_ender_pearl(uint32_t env_id, float x, float y, float z,
                              float vx, float vy, float vz);
    uint32_t throw_eye_of_ender(uint32_t env_id, float x, float y, float z,
                               float target_x, float target_z);

    // Queries
    const Entity* get_entity(uint32_t env_id, uint32_t entity_id) const;
    std::vector<const Entity*> get_nearby(uint32_t env_id,
                                          float x, float y, float z,
                                          float radius) const;
    uint32_t count_entities(uint32_t env_id, EntityType type = EntityType::NONE) const;

    // Buffer access for shader binding
    Buffer& get_entity_buffer() { return entity_buffer_; }

private:
    void load_shaders(const std::string& shader_dir);
    uint32_t find_free_slot(uint32_t env_id);

    VulkanContext& ctx_;
    BufferManager& buf_mgr_;
    Config config_;

    // Shaders
    std::unique_ptr<ComputePipeline> mob_ai_pipeline_;
    std::unique_ptr<ComputePipeline> physics_pipeline_;
    std::unique_ptr<ComputePipeline> projectile_pipeline_;

    // Storage
    Buffer entity_buffer_;
    std::vector<std::vector<Entity>> entity_cache_;  // CPU mirror
    std::vector<std::vector<uint32_t>> free_slots_;
};

} // namespace mc189
