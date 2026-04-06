// Mob AI dispatcher - coordinates AI behavior across entity types
// Simple direct-chase pathfinding, no complex navigation mesh

#include "mc189/mob_ai.h"

#include <algorithm>
#include <cmath>

namespace mc189 {

namespace {

constexpr float PI = 3.14159265358979f;

float distance_sq(const float* a, const float* b) {
    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    float dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

float distance_xz(const float* a, const float* b) {
    float dx = a[0] - b[0];
    float dz = a[2] - b[2];
    return std::sqrt(dx * dx + dz * dz);
}

float normalize_angle(float angle) {
    while (angle > PI) angle -= 2.0f * PI;
    while (angle < -PI) angle += 2.0f * PI;
    return angle;
}

// Move entity toward target position with given speed
void move_toward(Entity& e, float tx, float tz, float speed, float dt) {
    float dx = tx - e.position[0];
    float dz = tz - e.position[2];
    float dist = std::sqrt(dx * dx + dz * dz);
    if (dist < 0.01f) return;

    float nx = dx / dist;
    float nz = dz / dist;

    e.velocity[0] = nx * speed;
    e.velocity[2] = nz * speed;

    // Face movement direction
    e.yaw = std::atan2(-nx, nz);
}

// Check if player is looking at entity (for enderman aggro)
// Uses player yaw/pitch to determine look direction, checks if entity
// is within a narrow cone (5 degrees) of the look vector
bool is_player_looking_at(const Player& player, const Entity& e) {
    float dx = e.position[0] - player.position[0];
    float dy = (e.position[1] + 1.6f) - (player.position[1] + 1.62f); // eye heights
    float dz = e.position[2] - player.position[2];
    float dist_xz = std::sqrt(dx * dx + dz * dz);
    if (dist_xz < 0.1f) return false;

    // Angle from player to entity
    float target_yaw = std::atan2(-dx, dz);
    float target_pitch = -std::atan2(dy, dist_xz);

    // Player look angles (stored in radians)
    float yaw_diff = std::abs(normalize_angle(player.yaw - target_yaw));
    float pitch_diff = std::abs(normalize_angle(player.pitch - target_pitch));

    // Within ~5 degree cone
    constexpr float LOOK_THRESHOLD = 0.087f; // ~5 degrees in radians
    return yaw_diff < LOOK_THRESHOLD && pitch_diff < LOOK_THRESHOLD;
}

// Apply gravity and ground clamping
void apply_physics(Entity& e, float dt) {
    constexpr float GRAVITY = 0.08f;
    constexpr float DRAG = 0.02f;

    e.velocity[1] -= GRAVITY;
    e.velocity[1] *= (1.0f - DRAG);

    e.position[0] += e.velocity[0] * dt * 20.0f; // 20 ticks/sec
    e.position[1] += e.velocity[1] * dt * 20.0f;
    e.position[2] += e.velocity[2] * dt * 20.0f;

    // Simple ground clamp (no real terrain queries on CPU)
    if (e.position[1] < 0.0f) {
        e.position[1] = 0.0f;
        e.velocity[1] = 0.0f;
        e.flags |= 1; // MOB_FLAG_ON_GROUND
    }
}

// Tick down attack cooldown, returns true if can attack
bool tick_cooldown(Entity& e) {
    if (e.attack_cooldown > 0) {
        e.attack_cooldown--;
        return false;
    }
    return true;
}

} // namespace

const AIParams& get_ai_params(EntityType type) {
    uint32_t idx = static_cast<uint32_t>(type);
    constexpr uint32_t count = sizeof(AI_PARAMS) / sizeof(AI_PARAMS[0]);
    if (idx >= count) idx = 0;
    return AI_PARAMS[idx];
}

void update_zombie_ai(Entity& e, const Player& player, float dt) {
    const auto& params = get_ai_params(EntityType::ZOMBIE);
    float dist = distance_xz(e.position, player.position);

    if (dist > params.detection_range) {
        // Wander randomly
        e.ai_state = AIBehavior::WANDER;
        if (e.state_timer == 0) {
            e.path_target[0] = e.position[0] + (float(e.random_seed % 16) - 8.0f);
            e.path_target[2] = e.position[2] + (float((e.random_seed >> 8) % 16) - 8.0f);
            e.state_timer = 60 + (e.random_seed % 60);
            e.random_seed = e.random_seed * 1103515245 + 12345;
        }
        move_toward(e, e.path_target[0], e.path_target[2], params.move_speed * 0.5f, dt);
        e.state_timer--;
    } else if (dist > params.attack_range) {
        // Chase player
        e.ai_state = AIBehavior::CHASE;
        move_toward(e, player.position[0], player.position[2], params.move_speed, dt);
    } else {
        // Attack
        e.ai_state = AIBehavior::ATTACK_MELEE;
        e.velocity[0] = 0.0f;
        e.velocity[2] = 0.0f;
        // Face player
        float dx = player.position[0] - e.position[0];
        float dz = player.position[2] - e.position[2];
        e.yaw = std::atan2(-dx, dz);
    }

    apply_physics(e, dt);
}

void update_skeleton_ai(Entity& e, const Player& player, float dt,
                        uint32_t& out_shoot_arrow) {
    out_shoot_arrow = 0;
    const auto& params = get_ai_params(EntityType::SKELETON);
    float dist = distance_xz(e.position, player.position);

    if (dist > params.detection_range) {
        e.ai_state = AIBehavior::WANDER;
        if (e.state_timer == 0) {
            e.path_target[0] = e.position[0] + (float(e.random_seed % 16) - 8.0f);
            e.path_target[2] = e.position[2] + (float((e.random_seed >> 8) % 16) - 8.0f);
            e.state_timer = 60 + (e.random_seed % 60);
            e.random_seed = e.random_seed * 1103515245 + 12345;
        }
        move_toward(e, e.path_target[0], e.path_target[2], params.move_speed * 0.5f, dt);
        e.state_timer--;
    } else if (dist < 4.0f) {
        // Too close, back away while shooting
        e.ai_state = AIBehavior::FLEE;
        float dx = e.position[0] - player.position[0];
        float dz = e.position[2] - player.position[2];
        float flee_dist = std::sqrt(dx * dx + dz * dz);
        if (flee_dist > 0.01f) {
            move_toward(e, e.position[0] + dx / flee_dist * 5.0f,
                       e.position[2] + dz / flee_dist * 5.0f,
                       params.move_speed, dt);
        }
        if (tick_cooldown(e)) {
            out_shoot_arrow = 1;
            e.attack_cooldown = params.attack_cooldown;
        }
    } else {
        // Strafe and shoot
        e.ai_state = AIBehavior::ATTACK_RANGED;
        // Strafe perpendicular to player
        float dx = player.position[0] - e.position[0];
        float dz = player.position[2] - e.position[2];
        float d = std::sqrt(dx * dx + dz * dz);
        if (d > 0.01f) {
            // Perpendicular direction (rotate 90 degrees)
            float strafe_x = -dz / d;
            float strafe_z = dx / d;
            // Alternate strafe direction based on timer
            if ((e.state_timer / 40) % 2 == 0) {
                strafe_x = -strafe_x;
                strafe_z = -strafe_z;
            }
            e.velocity[0] = strafe_x * params.move_speed * 0.6f;
            e.velocity[2] = strafe_z * params.move_speed * 0.6f;
        }
        // Face player
        e.yaw = std::atan2(-dx, dz);

        if (tick_cooldown(e)) {
            out_shoot_arrow = 1;
            e.attack_cooldown = params.attack_cooldown;
        }
        e.state_timer++;
    }

    apply_physics(e, dt);
}

void update_creeper_ai(Entity& e, const Player& player, float dt,
                       uint32_t& out_explode) {
    out_explode = 0;
    const auto& params = get_ai_params(EntityType::CREEPER);
    float dist = distance_xz(e.position, player.position);

    // type_data[0] = fuse timer (30 ticks = 1.5 sec)
    // type_data[1] = fuse_active flag

    if (dist > params.detection_range) {
        e.ai_state = AIBehavior::WANDER;
        e.type_data[0] = 0;
        e.type_data[1] = 0;
        if (e.state_timer == 0) {
            e.path_target[0] = e.position[0] + (float(e.random_seed % 16) - 8.0f);
            e.path_target[2] = e.position[2] + (float((e.random_seed >> 8) % 16) - 8.0f);
            e.state_timer = 60 + (e.random_seed % 60);
            e.random_seed = e.random_seed * 1103515245 + 12345;
        }
        move_toward(e, e.path_target[0], e.path_target[2], params.move_speed * 0.5f, dt);
        e.state_timer--;
    } else if (dist > params.attack_range) {
        // Chase, reset fuse if player moved away
        e.ai_state = AIBehavior::CHASE;
        if (e.type_data[1] && dist > params.attack_range + 1.0f) {
            // Defuse if player escaped
            e.type_data[0] = 0;
            e.type_data[1] = 0;
        }
        move_toward(e, player.position[0], player.position[2], params.move_speed, dt);
    } else {
        // In range: start/continue fuse
        e.ai_state = AIBehavior::SPECIAL;
        e.velocity[0] = 0.0f;
        e.velocity[2] = 0.0f;
        e.type_data[1] = 1; // fuse active

        e.type_data[0]++;
        if (e.type_data[0] >= 30) {
            out_explode = 1;
            e.health = 0.0f; // Creeper dies on explosion
        }

        // Face player
        float dx = player.position[0] - e.position[0];
        float dz = player.position[2] - e.position[2];
        e.yaw = std::atan2(-dx, dz);
    }

    apply_physics(e, dt);
}

void update_enderman_ai(Entity& e, const Player& player, float dt,
                        bool player_looking, uint32_t& out_teleport) {
    out_teleport = 0;
    const auto& params = get_ai_params(EntityType::ENDERMAN);
    float dist = distance_xz(e.position, player.position);

    // type_data[0] = aggro flag
    // type_data[1] = teleport cooldown
    // type_data[2] = scream timer (freeze before attack)

    // Check if player is looking at us (aggro trigger)
    if (!e.type_data[0] && player_looking && dist < params.detection_range) {
        e.type_data[0] = 1; // Aggro!
        e.type_data[2] = 20; // Freeze for 1 second while screaming
    }

    // Tick teleport cooldown
    if (e.type_data[1] > 0) e.type_data[1]--;

    if (!e.type_data[0]) {
        // Passive: wander aimlessly
        e.ai_state = AIBehavior::WANDER;
        if (e.state_timer == 0) {
            e.path_target[0] = e.position[0] + (float(e.random_seed % 32) - 16.0f);
            e.path_target[2] = e.position[2] + (float((e.random_seed >> 8) % 32) - 16.0f);
            e.state_timer = 100 + (e.random_seed % 100);
            e.random_seed = e.random_seed * 1103515245 + 12345;
        }
        move_toward(e, e.path_target[0], e.path_target[2], params.move_speed * 0.3f, dt);
        e.state_timer--;
    } else if (e.type_data[2] > 0) {
        // Scream/freeze phase
        e.ai_state = AIBehavior::SPECIAL;
        e.velocity[0] = 0.0f;
        e.velocity[2] = 0.0f;
        // Face player during scream
        float dx = player.position[0] - e.position[0];
        float dz = player.position[2] - e.position[2];
        e.yaw = std::atan2(-dx, dz);
        e.type_data[2]--;
    } else if (dist > params.attack_range && dist < params.detection_range) {
        // Aggro'd: teleport toward player if far, else chase
        e.ai_state = AIBehavior::CHASE;

        if (dist > 16.0f && e.type_data[1] == 0) {
            // Teleport closer (within 4-8 blocks of player)
            float angle = std::atan2(e.position[2] - player.position[2],
                                     e.position[0] - player.position[0]);
            float tp_dist = 4.0f + float(e.random_seed % 5);
            e.position[0] = player.position[0] + std::cos(angle) * tp_dist;
            e.position[2] = player.position[2] + std::sin(angle) * tp_dist;
            e.random_seed = e.random_seed * 1103515245 + 12345;
            e.type_data[1] = 40; // 2 second teleport cooldown
            out_teleport = 1;
        } else {
            move_toward(e, player.position[0], player.position[2], params.move_speed, dt);
        }
    } else if (dist <= params.attack_range) {
        // Melee attack
        e.ai_state = AIBehavior::ATTACK_MELEE;
        e.velocity[0] = 0.0f;
        e.velocity[2] = 0.0f;
        float dx = player.position[0] - e.position[0];
        float dz = player.position[2] - e.position[2];
        e.yaw = std::atan2(-dx, dz);
    } else {
        // Lost aggro (player too far)
        e.type_data[0] = 0;
        e.ai_state = AIBehavior::IDLE;
        e.velocity[0] = 0.0f;
        e.velocity[2] = 0.0f;
    }

    // Teleport away if taking damage (hurt_time set externally)
    if (e.hurt_time > 0 && e.type_data[1] == 0) {
        float angle = float(e.random_seed % 628) / 100.0f;
        float tp_dist = 8.0f + float((e.random_seed >> 10) % 16);
        e.position[0] += std::cos(angle) * tp_dist;
        e.position[2] += std::sin(angle) * tp_dist;
        e.random_seed = e.random_seed * 1103515245 + 12345;
        e.type_data[1] = 20;
        out_teleport = 1;
    }

    apply_physics(e, dt);
}

void update_blaze_ai(Entity& e, const Player& player, float dt,
                     uint32_t& out_shoot_fireball) {
    out_shoot_fireball = 0;
    const auto& params = get_ai_params(EntityType::BLAZE);
    float dist = distance_xz(e.position, player.position);

    // Blaze hovers: maintain altitude
    constexpr float HOVER_HEIGHT = 3.0f;
    float target_y = player.position[1] + HOVER_HEIGHT;
    float dy = target_y - e.position[1];
    e.velocity[1] = std::clamp(dy * 0.1f, -0.2f, 0.2f);

    if (dist > params.detection_range) {
        e.ai_state = AIBehavior::IDLE;
        e.velocity[0] = 0.0f;
        e.velocity[2] = 0.0f;
    } else if (dist < 8.0f) {
        // Too close, back away
        e.ai_state = AIBehavior::FLEE;
        float dx = e.position[0] - player.position[0];
        float dz = e.position[2] - player.position[2];
        float d = std::sqrt(dx * dx + dz * dz);
        if (d > 0.01f) {
            e.velocity[0] = (dx / d) * params.move_speed;
            e.velocity[2] = (dz / d) * params.move_speed;
        }
        if (tick_cooldown(e)) {
            out_shoot_fireball = 1;
            e.attack_cooldown = params.attack_cooldown;
        }
    } else {
        // Hover at range and shoot fireballs
        e.ai_state = AIBehavior::ATTACK_RANGED;
        // Slow orbit around player
        float angle = std::atan2(e.position[2] - player.position[2],
                                 e.position[0] - player.position[0]);
        angle += 0.02f * dt * 20.0f;
        float orbit_dist = 16.0f;
        float tx = player.position[0] + std::cos(angle) * orbit_dist;
        float tz = player.position[2] + std::sin(angle) * orbit_dist;
        move_toward(e, tx, tz, params.move_speed * 0.5f, dt);

        // Face player
        float dx = player.position[0] - e.position[0];
        float dz = player.position[2] - e.position[2];
        e.yaw = std::atan2(-dx, dz);

        if (tick_cooldown(e)) {
            out_shoot_fireball = 1;
            e.attack_cooldown = params.attack_cooldown;
        }
    }

    // Blaze doesn't use normal ground physics - override
    e.position[0] += e.velocity[0] * dt * 20.0f;
    e.position[1] += e.velocity[1] * dt * 20.0f;
    e.position[2] += e.velocity[2] * dt * 20.0f;
}

void update_ghast_ai(Entity& e, const Player& player, float dt,
                     uint32_t& out_shoot_fireball) {
    out_shoot_fireball = 0;
    const auto& params = get_ai_params(EntityType::GHAST);
    float dist = distance_xz(e.position, player.position);

    // Ghast floats at high altitude, moves slowly
    constexpr float GHAST_ALTITUDE = 20.0f;
    float target_y = player.position[1] + GHAST_ALTITUDE;
    float dy = target_y - e.position[1];
    e.velocity[1] = std::clamp(dy * 0.05f, -0.1f, 0.1f);

    if (dist > params.detection_range) {
        // Drift randomly
        e.ai_state = AIBehavior::WANDER;
        if (e.state_timer == 0) {
            e.path_target[0] = e.position[0] + (float(e.random_seed % 64) - 32.0f);
            e.path_target[2] = e.position[2] + (float((e.random_seed >> 8) % 64) - 32.0f);
            e.state_timer = 200 + (e.random_seed % 200);
            e.random_seed = e.random_seed * 1103515245 + 12345;
        }
        move_toward(e, e.path_target[0], e.path_target[2], params.move_speed, dt);
        e.state_timer--;
    } else {
        // Face player and shoot fireballs from range
        e.ai_state = AIBehavior::ATTACK_RANGED;
        float dx = player.position[0] - e.position[0];
        float dz = player.position[2] - e.position[2];
        e.yaw = std::atan2(-dx, dz);

        // Maintain distance (30-50 blocks)
        if (dist < 30.0f) {
            float flee_x = e.position[0] - player.position[0];
            float flee_z = e.position[2] - player.position[2];
            float flee_d = std::sqrt(flee_x * flee_x + flee_z * flee_z);
            if (flee_d > 0.01f) {
                e.velocity[0] = (flee_x / flee_d) * params.move_speed;
                e.velocity[2] = (flee_z / flee_d) * params.move_speed;
            }
        } else if (dist > 50.0f) {
            move_toward(e, player.position[0], player.position[2], params.move_speed, dt);
        } else {
            e.velocity[0] *= 0.9f;
            e.velocity[2] *= 0.9f;
        }

        if (tick_cooldown(e)) {
            out_shoot_fireball = 1;
            e.attack_cooldown = params.attack_cooldown;
        }
    }

    // Ghast uses floating physics
    e.position[0] += e.velocity[0] * dt * 20.0f;
    e.position[1] += e.velocity[1] * dt * 20.0f;
    e.position[2] += e.velocity[2] * dt * 20.0f;
}

bool find_path(float sx, float sy, float sz,
              float tx, float ty, float tz,
              float* out_next_x, float* out_next_z) {
    // Simple direct-line pathfinding. No obstacle avoidance since the GPU
    // shaders handle actual collision. This just provides the next waypoint
    // direction for AI movement decisions.
    float dx = tx - sx;
    float dz = tz - sz;
    float dist = std::sqrt(dx * dx + dz * dz);

    if (dist < 0.5f) return false; // Already at target

    // Normalize and step forward by 1 block
    *out_next_x = sx + dx / dist;
    *out_next_z = sz + dz / dist;
    return true;
}

} // namespace mc189
