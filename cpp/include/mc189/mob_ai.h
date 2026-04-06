#pragma once
#include "mc189/simulator.h"
#include <cstdint>

namespace mc189 {

// Entity types matching shader MOB_TYPE_* constants
enum class EntityType : uint32_t {
    ZOMBIE = 0,
    SKELETON = 1,
    CREEPER = 2,
    SPIDER = 3,
    ENDERMAN = 4,
    BLAZE = 5,
    GHAST = 6,
    WITHER_SKELETON = 7,
    MAGMA_CUBE = 8,
    PIGMAN = 9,
    SILVERFISH = 10,
    ENDER_DRAGON = 11,
};

// AI behavior flags
enum class AIBehavior : uint32_t {
    IDLE = 0,
    WANDER = 1,
    FLEE = 2,
    CHASE = 3,
    ATTACK_MELEE = 4,
    ATTACK_RANGED = 5,
    SPECIAL = 6,  // Entity-specific (enderman teleport, creeper explode)
};

// CPU-side entity state (mirrors GPU MobState layout for sync)
struct alignas(16) Entity {
    float position[3];
    float _pad0;
    float velocity[3];
    float _pad1;

    EntityType type;
    AIBehavior ai_state;
    uint32_t target_entity_id;
    uint32_t flags;

    float health;
    float max_health;
    uint32_t hurt_time;
    uint32_t no_damage_time;

    float yaw;
    float pitch;
    float move_speed;
    float jump_velocity;

    uint32_t ai_tick_cooldown;
    uint32_t attack_cooldown;
    uint32_t state_timer;
    uint32_t random_seed;

    float path_target[3];
    uint32_t path_state;

    // Type-specific data (creeper fuse, enderman teleport cooldown, etc.)
    uint32_t type_data[6];
};

// AI parameters for each entity type
struct AIParams {
    EntityType type;
    float detection_range;
    float attack_range;
    float move_speed;
    float attack_damage;
    uint32_t attack_cooldown;
    AIBehavior default_behavior;
    bool hostile_to_player;
    bool burns_in_sunlight;
    float knockback_resistance;
};

constexpr AIParams AI_PARAMS[] = {
    {EntityType::ZOMBIE, 40.0f, 2.0f, 0.23f, 3.0f, 20, AIBehavior::CHASE, true, true, 0.0f},
    {EntityType::SKELETON, 16.0f, 15.0f, 0.25f, 2.0f, 20, AIBehavior::ATTACK_RANGED, true, true, 0.0f},
    {EntityType::CREEPER, 16.0f, 3.0f, 0.25f, 0.0f, 0, AIBehavior::CHASE, true, false, 0.0f},
    {EntityType::SPIDER, 16.0f, 2.0f, 0.3f, 2.0f, 20, AIBehavior::CHASE, true, false, 0.0f},
    {EntityType::ENDERMAN, 64.0f, 2.0f, 0.3f, 7.0f, 20, AIBehavior::IDLE, false, false, 0.0f},
    {EntityType::BLAZE, 48.0f, 48.0f, 0.23f, 5.0f, 60, AIBehavior::ATTACK_RANGED, true, false, 0.0f},
    {EntityType::GHAST, 100.0f, 64.0f, 0.1f, 0.0f, 60, AIBehavior::ATTACK_RANGED, true, false, 0.0f},
    {EntityType::WITHER_SKELETON, 16.0f, 2.0f, 0.25f, 8.0f, 20, AIBehavior::CHASE, true, false, 0.0f},
    {EntityType::MAGMA_CUBE, 16.0f, 2.0f, 0.2f, 3.0f, 20, AIBehavior::CHASE, true, false, 0.0f},
    {EntityType::PIGMAN, 32.0f, 2.0f, 0.23f, 5.0f, 20, AIBehavior::IDLE, false, false, 0.5f},
    {EntityType::SILVERFISH, 16.0f, 1.0f, 0.25f, 1.0f, 20, AIBehavior::CHASE, true, false, 0.0f},
    {EntityType::ENDER_DRAGON, 150.0f, 10.0f, 0.5f, 10.0f, 20, AIBehavior::SPECIAL, true, false, 1.0f},
};

const AIParams& get_ai_params(EntityType type);

// AI update functions (called by EntityManager)
void update_zombie_ai(Entity& e, const Player& player, float dt);
void update_skeleton_ai(Entity& e, const Player& player, float dt,
                       uint32_t& out_shoot_arrow);
void update_creeper_ai(Entity& e, const Player& player, float dt,
                      uint32_t& out_explode);
void update_enderman_ai(Entity& e, const Player& player, float dt,
                       bool player_looking, uint32_t& out_teleport);
void update_blaze_ai(Entity& e, const Player& player, float dt,
                    uint32_t& out_shoot_fireball);
void update_ghast_ai(Entity& e, const Player& player, float dt,
                    uint32_t& out_shoot_fireball);

// Pathfinding (simple A* on chunk grid)
bool find_path(float sx, float sy, float sz,
              float tx, float ty, float tz,
              float* out_next_x, float* out_next_z);

} // namespace mc189
