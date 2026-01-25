// CPU Backend - faithful port of dragon_fight_mvk.comp shader logic
// OpenMP-parallel game tick execution without GPU dependencies

#include "mc189/cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mc189 {

// ============================================================================
// Constants - exact match to dragon_fight_mvk.comp
// ============================================================================

namespace {

// Physics (MC 1.8.9 values)
constexpr float GRAVITY = -0.08f;
constexpr float DRAG_AIR = 0.98f;
constexpr float DRAG_GROUND = 0.6f;
constexpr float WALK_SPEED = 0.1f;
constexpr float SPRINT_MULT = 1.3f;
constexpr float SNEAK_MULT = 0.3f;
constexpr float JUMP_VEL = 0.42f;

// Combat
constexpr float SWORD_DAMAGE = 7.0f;
constexpr float HAND_DAMAGE = 1.0f;
constexpr float BOW_DAMAGE = 9.0f;
constexpr float KNOCKBACK_BASE = 0.4f;
constexpr uint32_t INVINCIBILITY_TICKS = 10;
constexpr uint32_t PLAYER_ATTACK_COOLDOWN_TICKS = 10;
constexpr float CRIT_MULTIPLIER = 1.5f;

// Dragon
constexpr float DRAGON_MELEE_DAMAGE = 10.0f;
constexpr float DRAGON_CHARGE_DAMAGE = 5.0f;
constexpr float DRAGON_BREATH_DAMAGE = 6.0f;
constexpr float DRAGON_CIRCLE_RADIUS = 75.0f;
constexpr float DRAGON_SPEED = 1.5f;

// End dimension
constexpr float END_ISLAND_RADIUS = 100.0f;
constexpr float PILLAR_RADIUS = 3.0f;

// Flags
constexpr uint32_t FLAG_ON_GROUND = 1u;
constexpr uint32_t FLAG_SPRINTING = 2u;
constexpr uint32_t FLAG_SNEAKING = 4u;
constexpr uint32_t FLAG_DEAD = 8u;
constexpr uint32_t FLAG_WON = 16u;

// Dragon phases
constexpr uint32_t DRAGON_CIRCLING = 0u;
constexpr uint32_t DRAGON_CHARGING = 2u;
constexpr uint32_t DRAGON_LANDING = 3u;
constexpr uint32_t DRAGON_PERCHING = 4u;
constexpr uint32_t DRAGON_TAKING_OFF = 5u;
constexpr uint32_t DRAGON_DEAD = 6u;

constexpr float PI = 3.14159265358979323846f;
constexpr float DEG2RAD = PI / 180.0f;
constexpr float RAD2DEG = 180.0f / PI;

} // namespace

// ============================================================================
// Utility functions (match shader utilities)
// ============================================================================

uint32_t CpuBackend::hash(uint32_t x) {
  x ^= x >> 16;
  x *= 0x85ebca6bu;
  x ^= x >> 13;
  x *= 0xc2b2ae35u;
  x ^= x >> 16;
  return x;
}

float CpuBackend::rng(uint32_t &seed) {
  seed = hash(seed);
  return static_cast<float>(seed) / 4294967295.0f;
}

float CpuBackend::get_ground_height(float px, float pz) {
  float dist = std::sqrt(px * px + pz * pz);
  if (dist > END_ISLAND_RADIUS)
    return -64.0f; // Void

  // Check if on pillar
  for (int i = 0; i < 10; i++) {
    float angle = static_cast<float>(i) * 0.628318f; // 2*PI/10
    float pillar_dist = 40.0f + static_cast<float>(i % 3) * 15.0f;
    float pillar_x = std::cos(angle) * pillar_dist;
    float pillar_z = std::sin(angle) * pillar_dist;
    float dx = px - pillar_x;
    float dz = pz - pillar_z;
    if (std::sqrt(dx * dx + dz * dz) < PILLAR_RADIUS) {
      return PILLAR_HEIGHT;
    }
  }

  return END_SPAWN_Y; // Main island surface
}

void CpuBackend::get_pillar_position(uint32_t idx, float *out) {
  float angle = static_cast<float>(idx) * 0.628318f;
  float dist = 40.0f + static_cast<float>(idx % 3) * 15.0f;
  out[0] = std::cos(angle) * dist;
  out[1] = PILLAR_HEIGHT + 1.0f;
  out[2] = std::sin(angle) * dist;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

CpuBackend::CpuBackend(const Config &config) : config_(config) {
#ifdef _OPENMP
  if (config_.num_threads > 0) {
    omp_set_num_threads(static_cast<int>(config_.num_threads));
  }
#endif

  players_.resize(config_.num_envs);
  dragons_.resize(config_.num_envs);
  crystals_.resize(config_.num_envs * NUM_CRYSTALS);
  game_states_.resize(config_.num_envs);
  inputs_.resize(config_.num_envs);
  world_seeds_.reserve(config_.num_envs);
  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    world_seeds_.emplace_back(0);
  }

  observations_.resize(config_.num_envs * OBSERVATION_SIZE, 0.0f);
  rewards_.resize(config_.num_envs, 0.0f);
  dones_.resize(config_.num_envs, 0);

  reset();
}

CpuBackend::~CpuBackend() = default;

// ============================================================================
// Action Decoding (matches mc189_simulator.cpp exactly)
// ============================================================================

void CpuBackend::decode_action(int32_t action, InputState &inp) {
  std::memset(&inp, 0, sizeof(InputState));

  switch (action) {
  case 1: // forward
    inp.movement[2] = 1.0f;
    break;
  case 2: // back
    inp.movement[2] = -1.0f;
    break;
  case 3: // left
    inp.movement[0] = -1.0f;
    break;
  case 4: // right
    inp.movement[0] = 1.0f;
    break;
  case 5: // forward+left
    inp.movement[2] = 1.0f;
    inp.movement[0] = -0.7f;
    break;
  case 6: // forward+right
    inp.movement[2] = 1.0f;
    inp.movement[0] = 0.7f;
    break;
  case 7: // jump
    inp.flags |= 1u;
    break;
  case 8: // jump+forward
    inp.flags |= 1u;
    inp.movement[2] = 1.0f;
    break;
  case 9: // attack
    inp.action = 1;
    break;
  case 10: // attack+forward
    inp.action = 1;
    inp.movement[2] = 1.0f;
    break;
  case 11: // sprint+forward
    inp.flags |= 2u;
    inp.movement[2] = 1.0f;
    break;
  case 12: // look_left
    inp.lookDeltaX = -5.0f;
    break;
  case 13: // look_right
    inp.lookDeltaX = 5.0f;
    break;
  case 14: // swap_weapon
    inp.action = 3;
    break;
  case 15: // look_up
    inp.lookDeltaY = -5.0f;
    break;
  case 16: // look_down
    inp.lookDeltaY = 5.0f;
    break;
  default:
    break; // noop
  }
}

// ============================================================================
// Reset
// ============================================================================

void CpuBackend::reset(uint32_t env_id, uint64_t seed) {
  if (seed == 0) {
    std::random_device rd;
    seed = (static_cast<uint64_t>(rd()) << 32) | rd();
  }

  if (env_id == 0xFFFFFFFF) {
    for (uint32_t i = 0; i < config_.num_envs; ++i) {
      reset_env(i, seed + i);
    }
    tick_number_ = 0;
    std::fill(dones_.begin(), dones_.end(), 0);

    // Prime initial observations (match GPU: dispatch once with noop inputs)
    std::fill(rewards_.begin(), rewards_.end(), 0.0f);
    for (uint32_t i = 0; i < config_.num_envs; ++i) {
      std::memset(&inputs_[i], 0, sizeof(InputState));
      stage_setup(i);
      stage_player_physics(i);
      stage_dragon_ai(i);
      stage_combat(i);
      stage_environment(i);
      stage_observations(i);
    }
    tick_number_ = 1;
  } else if (env_id < config_.num_envs) {
    reset_env(env_id, seed);
    dones_[env_id] = 0;

    // Encode initial observation for the reset env
    std::memset(&inputs_[env_id], 0, sizeof(InputState));
    rewards_[env_id] = 0.0f;
    stage_setup(env_id);
    stage_player_physics(env_id);
    stage_dragon_ai(env_id);
    stage_combat(env_id);
    stage_environment(env_id);
    stage_observations(env_id);
  }
}

void CpuBackend::reset_env(uint32_t env_id, uint64_t seed) {
  world_seeds_[env_id] = WorldSeed(seed);

  auto &p = players_[env_id];
  std::memset(&p, 0, sizeof(Player));
  p.position[0] = 0.0f;
  p.position[1] = END_SPAWN_Y;
  p.position[2] = 0.0f;
  p.health = 20.0f;
  p.hunger = 20.0f;
  p.saturation = 5.0f;
  p.weapon_slot = 1; // Start with sword
  p.arrows = 64;

  auto &d = dragons_[env_id];
  std::memset(&d, 0, sizeof(Dragon));
  d.position[0] = 0.0f;
  d.position[1] = 70.0f;
  d.position[2] = 75.0f;
  d.health = DRAGON_MAX_HEALTH;
  d.phase = 0; // Circling
  d.circle_angle = static_cast<float>(env_id) * 0.1f;

  Crystal *env_crystals = &crystals_[env_id * NUM_CRYSTALS];
  for (uint32_t j = 0; j < NUM_CRYSTALS; ++j) {
    get_pillar_position(j, env_crystals[j].position);
    env_crystals[j].is_alive = 1.0f;
  }

  auto &gs = game_states_[env_id];
  std::memset(&gs, 0, sizeof(GameState));
  gs.deltaTime = 0.05f;
  gs.randomSeed = static_cast<uint32_t>(seed & 0xFFFFFFFF);
}

// ============================================================================
// Main Step
// ============================================================================

void CpuBackend::step(const int32_t *actions, size_t num_actions) {
  if (num_actions != config_.num_envs) {
    throw std::invalid_argument("Action count must match num_envs");
  }

  // Clear previous dones
  std::fill(dones_.begin(), dones_.end(), 0);

  // Decode all actions
  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    decode_action(actions[i], inputs_[i]);
  }

  // Run all stages per environment, parallelized with OpenMP
#pragma omp parallel for schedule(static)
  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    stage_setup(i);
    stage_player_physics(i);
    stage_dragon_ai(i);
    stage_combat(i);
    stage_environment(i);
    stage_observations(i);
  }

  // Auto-reset done environments (sequential, matches GPU behavior)
  for (uint32_t i = 0; i < config_.num_envs; ++i) {
    if (dones_[i]) {
      uint64_t new_seed =
          (static_cast<uint64_t>(tick_number_) << 32) | (i + 1);
      reset_env(i, new_seed);
      // Don't clear dones_[i] - let user see it; cleared at next step()
    }
  }

  tick_number_++;
}

// ============================================================================
// Stage 0: Setup (match shader stage_setup)
// ============================================================================

void CpuBackend::stage_setup(uint32_t env_id) {
  game_states_[env_id].tickNumber = tick_number_;
  game_states_[env_id].randomSeed =
      hash(game_states_[env_id].randomSeed ^ env_id ^ tick_number_);
  rewards_[env_id] = 0.0f;
}

// ============================================================================
// Stage 1: Player Physics & Movement (match shader stage_player_physics)
// ============================================================================

void CpuBackend::stage_player_physics(uint32_t env_id) {
  Player &p = players_[env_id];
  const InputState &inp = inputs_[env_id];

  // Skip if dead
  if ((p.flags & FLAG_DEAD) != 0u)
    return;

  // Look - deltas are already in degrees (from action decoder)
  p.yaw = std::fmod(p.yaw + inp.lookDeltaX, 360.0f);
  if (p.yaw < 0.0f)
    p.yaw += 360.0f;
  p.pitch = std::clamp(p.pitch + inp.lookDeltaY, -90.0f, 90.0f);

  // Movement vectors
  float yaw_rad = p.yaw * DEG2RAD;
  float fwd_x = -std::sin(yaw_rad);
  float fwd_z = std::cos(yaw_rad);
  float right_x = std::cos(yaw_rad);
  float right_z = std::sin(yaw_rad);

  // Input to movement: forward * movement_z + right * movement_x
  float move_x = fwd_x * inp.movement[2] + right_x * inp.movement[0];
  float move_z = fwd_z * inp.movement[2] + right_z * inp.movement[0];

  // Speed modifiers
  float speed = WALK_SPEED;
  bool sprint = (inp.flags & 2u) != 0u;
  bool sneak = (inp.flags & 4u) != 0u;
  bool on_ground = (p.flags & FLAG_ON_GROUND) != 0u;

  if (sprint && !sneak) {
    speed *= SPRINT_MULT;
    p.flags |= FLAG_SPRINTING;
  } else {
    p.flags &= ~FLAG_SPRINTING;
  }

  if (sneak) {
    speed *= SNEAK_MULT;
    p.flags |= FLAG_SNEAKING;
  } else {
    p.flags &= ~FLAG_SNEAKING;
  }

  // Apply movement (additive, like shader)
  float move_len = std::sqrt(move_x * move_x + move_z * move_z);
  if (move_len > 0.001f) {
    float inv_len = speed / move_len;
    p.velocity[0] += move_x * inv_len;
    p.velocity[2] += move_z * inv_len;
  }

  // Jump
  bool jump = (inp.flags & 1u) != 0u;
  if (jump && on_ground) {
    p.velocity[1] = JUMP_VEL;
    p.flags &= ~FLAG_ON_GROUND;
    if (sprint)
      p.exhaustion += 0.2f;
  }

  // Gravity (only when not on ground)
  if (!on_ground) {
    p.velocity[1] += GRAVITY;
  }

  // Drag
  float drag = on_ground ? DRAG_GROUND : DRAG_AIR;
  p.velocity[0] *= drag;
  p.velocity[2] *= drag;
  p.velocity[1] *= DRAG_AIR;

  // Position update
  p.position[0] += p.velocity[0];
  p.position[1] += p.velocity[1];
  p.position[2] += p.velocity[2];

  // Ground collision (uses End island height map)
  float ground = get_ground_height(p.position[0], p.position[2]);
  if (p.position[1] < ground) {
    p.position[1] = ground;
    p.velocity[1] = 0.0f;
    p.flags |= FLAG_ON_GROUND;
  } else if (p.position[1] > ground + 0.1f) {
    p.flags &= ~FLAG_ON_GROUND;
  }

  // Void death - kill if below island or off the edge
  float dist_from_center = std::sqrt(p.position[0] * p.position[0] +
                                     p.position[2] * p.position[2]);
  if (p.position[1] <= -60.0f ||
      (dist_from_center > END_ISLAND_RADIUS &&
       p.position[1] < END_SPAWN_Y - 10.0f)) {
    p.health = 0.0f;
    p.flags |= FLAG_DEAD;
    dones_[env_id] = 1;
    rewards_[env_id] -= 50.0f;
  }

  // Timers
  if (p.invincibility_timer > 0u)
    p.invincibility_timer--;

  // Bow charge (action == 2 = use, with bow equipped and arrows available)
  if (inp.action == 2u && p.arrows > 0u && p.weapon_slot == 2u) {
    p.arrow_charge = std::min(p.arrow_charge + 0.05f, 1.0f);
  }
}

// ============================================================================
// Stage 2: Dragon AI (match shader stage_dragon_ai)
// ============================================================================

void CpuBackend::stage_dragon_ai(uint32_t env_id) {
  Dragon &d = dragons_[env_id];
  const Player &p = players_[env_id];
  GameState &gs = game_states_[env_id];
  uint32_t seed = gs.randomSeed;

  if (d.health <= 0.0f || d.phase == DRAGON_DEAD)
    return;

  // Distance to player
  float to_player_x = p.position[0] - d.position[0];
  float to_player_y = p.position[1] - d.position[1];
  float to_player_z = p.position[2] - d.position[2];
  float player_dist =
      std::sqrt(to_player_x * to_player_x + to_player_y * to_player_y +
                to_player_z * to_player_z);

  d.phase_timer++;

  switch (d.phase) {
  case DRAGON_CIRCLING: {
    // Circle around (0, 70, 0)
    d.circle_angle += 0.02f;
    d.target_position[0] = std::cos(d.circle_angle) * DRAGON_CIRCLE_RADIUS;
    d.target_position[1] = 70.0f + std::sin(d.circle_angle * 2.0f) * 15.0f;
    d.target_position[2] = std::sin(d.circle_angle) * DRAGON_CIRCLE_RADIUS;

    // Transition to charging if player is close
    if (player_dist < 40.0f && rng(seed) < 0.02f) {
      d.phase = DRAGON_CHARGING;
      d.phase_timer = 0u;
      d.target_position[0] = p.position[0];
      d.target_position[1] = p.position[1];
      d.target_position[2] = p.position[2];
    }

    // Transition to perching
    bool all_crystals_destroyed = (gs.crystals_destroyed >= NUM_CRYSTALS);
    bool random_perch = (d.phase_timer > 200u && rng(seed) < 0.005f);

    if (all_crystals_destroyed || random_perch) {
      if (rng(seed) < 0.5f) {
        d.phase = DRAGON_LANDING;
        d.phase_timer = 0u;
        d.target_position[0] = 0.0f;
        d.target_position[1] = END_SPAWN_Y + 4.0f;
        d.target_position[2] = 0.0f;
      }
    }
    break;
  }

  case DRAGON_CHARGING: {
    // Dive toward player
    d.target_position[0] = p.position[0];
    d.target_position[1] = p.position[1] + 2.0f;
    d.target_position[2] = p.position[2];

    // Hit player on contact
    if (player_dist < 8.0f && d.attack_cooldown == 0u) {
      d.attack_cooldown = 20u;
    }

    // Return to circling after charge
    if (d.phase_timer > 60u || player_dist < 5.0f) {
      d.phase = DRAGON_CIRCLING;
      d.phase_timer = 0u;
    }
    break;
  }

  case DRAGON_LANDING: {
    // Move to center perch position
    d.target_position[0] = 0.0f;
    d.target_position[1] = END_SPAWN_Y + 4.0f;
    d.target_position[2] = 0.0f;

    float dx = d.position[0] - d.target_position[0];
    float dy = d.position[1] - d.target_position[1];
    float dz = d.position[2] - d.target_position[2];
    float dist_to_target = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (dist_to_target < 3.0f) {
      d.phase = DRAGON_PERCHING;
      d.phase_timer = 0u;
      d.perch_timer = 200u; // 10 seconds at 20 TPS
    }
    break;
  }

  case DRAGON_PERCHING: {
    // Stay on perch, accumulate breath timer
    d.target_position[0] = 0.0f;
    d.target_position[1] = END_SPAWN_Y + 4.0f;
    d.target_position[2] = 0.0f;
    d.breath_timer += 1.0f;

    d.perch_timer--;
    if (d.perch_timer == 0u) {
      d.phase = DRAGON_TAKING_OFF;
      d.phase_timer = 0u;
      d.breath_timer = 0.0f;
    }
    break;
  }

  case DRAGON_TAKING_OFF: {
    // Rise up
    d.target_position[0] = 0.0f;
    d.target_position[1] = 90.0f;
    d.target_position[2] = 0.0f;

    if (d.position[1] > 80.0f) {
      d.phase = DRAGON_CIRCLING;
      d.phase_timer = 0u;
    }
    break;
  }

  default:
    break;
  }

  // Move toward target
  float to_target_x = d.target_position[0] - d.position[0];
  float to_target_y = d.target_position[1] - d.position[1];
  float to_target_z = d.target_position[2] - d.position[2];
  float dist =
      std::sqrt(to_target_x * to_target_x + to_target_y * to_target_y +
                to_target_z * to_target_z);

  if (dist > 0.5f) {
    float speed =
        (d.phase == DRAGON_CHARGING) ? DRAGON_SPEED * 2.0f : DRAGON_SPEED;
    float inv_dist = speed / dist;
    d.velocity[0] = to_target_x * inv_dist;
    d.velocity[1] = to_target_y * inv_dist;
    d.velocity[2] = to_target_z * inv_dist;
    d.yaw = std::atan2(to_target_x, to_target_z) * RAD2DEG;
    float xz_len = std::sqrt(to_target_x * to_target_x +
                             to_target_z * to_target_z);
    d.pitch = std::atan2(-to_target_y, xz_len) * RAD2DEG;
  } else {
    d.velocity[0] = 0.0f;
    d.velocity[1] = 0.0f;
    d.velocity[2] = 0.0f;
  }

  d.position[0] += d.velocity[0];
  d.position[1] += d.velocity[1];
  d.position[2] += d.velocity[2];

  if (d.attack_cooldown > 0u)
    d.attack_cooldown--;

  gs.randomSeed = seed;
}

// ============================================================================
// Stage 3: Combat (match shader stage_combat)
// ============================================================================

void CpuBackend::stage_combat(uint32_t env_id) {
  Player &p = players_[env_id];
  Dragon &d = dragons_[env_id];
  const InputState &inp = inputs_[env_id];
  GameState &gs = game_states_[env_id];

  if ((p.flags & FLAG_DEAD) != 0u || d.health <= 0.0f)
    return;

  if (p.attack_cooldown > 0u) {
    p.attack_cooldown--;
  }

  // ===== Player attacks =====
  if (inp.action == 1u && p.attack_cooldown == 0u) {
    p.attack_cooldown = PLAYER_ATTACK_COOLDOWN_TICKS;

    // Attack direction from look angles
    float yaw_rad = p.yaw * DEG2RAD;
    float pitch_rad = p.pitch * DEG2RAD;
    float look_x = -std::sin(yaw_rad) * std::cos(pitch_rad);
    float look_y = -std::sin(pitch_rad);
    float look_z = std::cos(yaw_rad) * std::cos(pitch_rad);

    // Check dragon hit (melee)
    float to_dragon_x = d.position[0] - p.position[0];
    float to_dragon_y = d.position[1] - p.position[1];
    float to_dragon_z = d.position[2] - p.position[2];
    float dragon_dist =
        std::sqrt(to_dragon_x * to_dragon_x + to_dragon_y * to_dragon_y +
                  to_dragon_z * to_dragon_z);
    float reach = (p.weapon_slot == 1u) ? 4.5f : 3.5f;

    bool can_hit = false;
    if (dragon_dist <= reach && dragon_dist > 0.01f) {
      float inv_dist = 1.0f / dragon_dist;
      float dot = (to_dragon_x * inv_dist) * look_x +
                  (to_dragon_y * inv_dist) * look_y +
                  (to_dragon_z * inv_dist) * look_z;
      can_hit = dot > 0.5f;
    }

    bool dragon_vulnerable = (d.phase == DRAGON_PERCHING);

    if (can_hit) {
      float damage = (p.weapon_slot == 1u) ? SWORD_DAMAGE : HAND_DAMAGE;

      // Critical hit: falling + not on ground
      bool crit =
          p.velocity[1] < -0.1f && (p.flags & FLAG_ON_GROUND) == 0u;
      if (crit) {
        damage *= CRIT_MULTIPLIER;
        rewards_[env_id] += 2.0f;
      }

      if (dragon_vulnerable) {
        d.health = std::max(0.0f, d.health - damage);
        gs.dragon_hits++;
        gs.best_dragon_damage = std::max(gs.best_dragon_damage, damage);
        rewards_[env_id] += damage * 2.0f;

        // Knockback dragon slightly
        d.velocity[0] += look_x * 0.5f;
        d.velocity[1] += look_y * 0.5f;
        d.velocity[2] += look_z * 0.5f;
      } else {
        rewards_[env_id] += 0.1f; // Small reward for trying
      }
    }

    // Check crystal hit
    Crystal *env_crystals = &crystals_[env_id * NUM_CRYSTALS];
    for (uint32_t i = 0; i < NUM_CRYSTALS; i++) {
      if (env_crystals[i].is_alive < 0.5f)
        continue;

      float to_crystal_x = env_crystals[i].position[0] - p.position[0];
      float to_crystal_y = env_crystals[i].position[1] - p.position[1];
      float to_crystal_z = env_crystals[i].position[2] - p.position[2];
      float crystal_dist =
          std::sqrt(to_crystal_x * to_crystal_x +
                    to_crystal_y * to_crystal_y +
                    to_crystal_z * to_crystal_z);

      if (crystal_dist <= reach && crystal_dist > 0.01f) {
        float inv_dist = 1.0f / crystal_dist;
        float dot = (to_crystal_x * inv_dist) * look_x +
                    (to_crystal_y * inv_dist) * look_y +
                    (to_crystal_z * inv_dist) * look_z;

        if (dot > 0.3f) {
          env_crystals[i].is_alive = 0.0f;
          gs.crystals_destroyed++;
          rewards_[env_id] += 10.0f;
          break;
        }
      }
    }
  }

  // Bow attack
  if (inp.action == 2u && p.weapon_slot == 2u && p.arrows > 0u &&
      p.arrow_charge >= 1.0f) {
    float yaw_rad = p.yaw * DEG2RAD;
    float pitch_rad = p.pitch * DEG2RAD;
    float arrow_dir_x = -std::sin(yaw_rad) * std::cos(pitch_rad);
    float arrow_dir_y = -std::sin(pitch_rad);
    float arrow_dir_z = std::cos(yaw_rad) * std::cos(pitch_rad);

    // Simple raycast
    float arrow_x = p.position[0];
    float arrow_y = p.position[1] + 1.6f;
    float arrow_z = p.position[2];

    for (int t = 0; t < 100; t++) {
      arrow_x += arrow_dir_x * 1.5f;
      arrow_y += arrow_dir_y * 1.5f;
      arrow_z += arrow_dir_z * 1.5f;

      // Check dragon hit
      float dx = arrow_x - d.position[0];
      float dy = arrow_y - d.position[1];
      float dz = arrow_z - d.position[2];
      if (dx * dx + dy * dy + dz * dz < 25.0f) { // 5.0^2
        float damage = BOW_DAMAGE * p.arrow_charge;
        d.health = std::max(0.0f, d.health - damage);
        gs.dragon_hits++;
        rewards_[env_id] += damage * 1.5f;
        break;
      }

      // Check crystal hit
      Crystal *env_crystals = &crystals_[env_id * NUM_CRYSTALS];
      bool hit_crystal = false;
      for (uint32_t i = 0; i < NUM_CRYSTALS; i++) {
        if (env_crystals[i].is_alive < 0.5f)
          continue;

        float cx = arrow_x - env_crystals[i].position[0];
        float cy = arrow_y - env_crystals[i].position[1];
        float cz = arrow_z - env_crystals[i].position[2];
        if (cx * cx + cy * cy + cz * cz < 4.0f) { // 2.0^2
          env_crystals[i].is_alive = 0.0f;
          gs.crystals_destroyed++;
          rewards_[env_id] += 10.0f;
          hit_crystal = true;
          break;
        }
      }
      if (hit_crystal)
        break;

      // Hit ground
      if (arrow_y < get_ground_height(arrow_x, arrow_z))
        break;
    }

    p.arrows--;
    p.arrow_charge = 0.0f;
  }

  // Weapon swap
  if (inp.action == 3u) {
    p.weapon_slot = (p.weapon_slot + 1u) % 3u;
  }

  // ===== Dragon attacks player =====
  float to_player_x = p.position[0] - d.position[0];
  float to_player_z = p.position[2] - d.position[2];
  float to_player_y = p.position[1] - d.position[1];
  float player_dist =
      std::sqrt(to_player_x * to_player_x + to_player_y * to_player_y +
                to_player_z * to_player_z);

  // Contact damage during charge
  if (d.phase == DRAGON_CHARGING && player_dist < 6.0f &&
      p.invincibility_timer == 0u && d.attack_cooldown == 0u) {
    p.health = std::max(0.0f, p.health - DRAGON_CHARGE_DAMAGE);
    p.invincibility_timer = INVINCIBILITY_TICKS;

    // Knockback - MC 1.8.9 formula
    float d0 = d.position[0] - p.position[0];
    float d1 = d.position[2] - p.position[2];
    float f = std::sqrt(d0 * d0 + d1 * d1);
    float knockbackStrength = 1.0f;

    p.velocity[0] /= 2.0f;
    p.velocity[2] /= 2.0f;
    p.velocity[1] /= 2.0f;

    if (f > 0.001f) {
      p.velocity[0] -= (d0 / f) * knockbackStrength * KNOCKBACK_BASE;
      p.velocity[2] -= (d1 / f) * knockbackStrength * KNOCKBACK_BASE;
    }
    p.velocity[1] += knockbackStrength * KNOCKBACK_BASE;
    if (p.velocity[1] > 0.4f)
      p.velocity[1] = 0.4f;

    rewards_[env_id] -= 5.0f;
  }

  // Breath damage during perch
  if (d.phase == DRAGON_PERCHING && d.breath_timer > 0.0f) {
    float dragon_yaw_rad = d.yaw * DEG2RAD;
    float dragon_fwd_x = -std::sin(dragon_yaw_rad);
    float dragon_fwd_z = std::cos(dragon_yaw_rad);

    if (player_dist < 15.0f && p.invincibility_timer == 0u &&
        player_dist > 0.01f) {
      float inv_dist = 1.0f / player_dist;
      float dot = (to_player_x * inv_dist) * dragon_fwd_x +
                  (to_player_z * inv_dist) * dragon_fwd_z;

      if (dot > 0.3f) {
        if (static_cast<int>(d.breath_timer) % 10 == 0) {
          p.health = std::max(0.0f, p.health - DRAGON_BREATH_DAMAGE);
          p.invincibility_timer = INVINCIBILITY_TICKS / 2u;
          rewards_[env_id] -= 3.0f;
        }
      }
    }
  }

  // Melee when close
  if (player_dist < 4.0f && p.invincibility_timer == 0u &&
      d.attack_cooldown == 0u) {
    p.health = std::max(0.0f, p.health - DRAGON_MELEE_DAMAGE);
    p.invincibility_timer = INVINCIBILITY_TICKS;
    d.attack_cooldown = 20u;

    // Knockback - MC 1.8.9 formula (stronger for melee)
    float d0 = d.position[0] - p.position[0];
    float d1 = d.position[2] - p.position[2];
    float f = std::sqrt(d0 * d0 + d1 * d1);
    float knockbackStrength = 2.0f;

    p.velocity[0] /= 2.0f;
    p.velocity[2] /= 2.0f;
    p.velocity[1] /= 2.0f;

    if (f > 0.001f) {
      p.velocity[0] -= (d0 / f) * knockbackStrength * KNOCKBACK_BASE;
      p.velocity[2] -= (d1 / f) * knockbackStrength * KNOCKBACK_BASE;
    }
    p.velocity[1] += knockbackStrength * KNOCKBACK_BASE;
    if (p.velocity[1] > 0.4f)
      p.velocity[1] = 0.4f;

    rewards_[env_id] -= 8.0f;
  }

  // Player death
  if (p.health <= 0.0f) {
    p.flags |= FLAG_DEAD;
    gs.player_deaths++;
    dones_[env_id] = 1;
    rewards_[env_id] -= 50.0f;
  }

  // Dragon death = WIN
  if (d.health <= 0.0f) {
    d.phase = DRAGON_DEAD;
    p.flags |= FLAG_WON;
    dones_[env_id] = 1;
    rewards_[env_id] += 1000.0f;
  }
}

// ============================================================================
// Stage 4: Environment Update (match shader stage_environment)
// ============================================================================

void CpuBackend::stage_environment(uint32_t env_id) {
  Player &p = players_[env_id];
  const GameState &gs = game_states_[env_id];

  // Hunger (simplified, matches shader)
  if (p.exhaustion >= 4.0f) {
    p.exhaustion -= 4.0f;
    if (p.saturation > 0.0f) {
      p.saturation = std::max(0.0f, p.saturation - 1.0f);
    } else {
      p.hunger = std::max(0.0f, p.hunger - 1.0f);
    }
  }

  // Regen
  if (p.hunger >= 18.0f && p.health < 20.0f) {
    p.health = std::min(20.0f, p.health + 0.025f);
    p.exhaustion += 0.05f;
  }

  // Starvation
  if (p.hunger <= 0.0f && gs.tickNumber % 80u == 0u) {
    p.health = std::max(0.0f, p.health - 1.0f);
  }

  // Time penalty (encourage speed)
  rewards_[env_id] -= 0.001f;

  // Small reward for being close to dragon when it's vulnerable
  const Dragon &d = dragons_[env_id];
  if (d.phase == DRAGON_PERCHING) {
    float dx = p.position[0] - d.position[0];
    float dy = p.position[1] - d.position[1];
    float dz = p.position[2] - d.position[2];
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist < 15.0f) {
      rewards_[env_id] += 0.01f;
    }
  }
}

// ============================================================================
// Stage 5: Observation Extraction (match shader stage_observations exactly)
// ============================================================================

void CpuBackend::stage_observations(uint32_t env_id) {
  const Player &p = players_[env_id];
  const Dragon &d = dragons_[env_id];
  const GameState &gs = game_states_[env_id];
  const InputState &inp = inputs_[env_id];

  float *obs = &observations_[env_id * OBSERVATION_SIZE];

  // === Player state (16 floats, normalized) ===
  obs[0] = p.position[0] / 100.0f;                   // pos_x
  obs[1] = (p.position[1] - 64.0f) / 50.0f;          // pos_y
  obs[2] = p.position[2] / 100.0f;                   // pos_z
  obs[3] = p.velocity[0] / 2.0f;                     // vel_x
  obs[4] = p.velocity[1] / 2.0f;                     // vel_y
  obs[5] = p.velocity[2] / 2.0f;                     // vel_z
  obs[6] = p.yaw / 360.0f;                           // yaw
  obs[7] = (p.pitch + 90.0f) / 180.0f;               // pitch
  obs[8] = p.health / 20.0f;                         // health
  obs[9] = p.hunger / 20.0f;                         // hunger
  obs[10] = (p.flags & FLAG_ON_GROUND) ? 1.0f : 0.0f; // on_ground
  obs[11] = (p.attack_cooldown == 0u) ? 1.0f : 0.0f;  // attack_ready
  obs[12] = static_cast<float>(p.weapon_slot) / 2.0f;  // weapon
  obs[13] = static_cast<float>(p.arrows) / 64.0f;      // arrows
  obs[14] = p.arrow_charge;                            // arrow_charge
  obs[15] = static_cast<float>(inp.action);            // reserved0 (debug)

  // === Dragon state (16 floats) ===
  float to_dragon_x = d.position[0] - p.position[0];
  float to_dragon_y = d.position[1] - p.position[1];
  float to_dragon_z = d.position[2] - p.position[2];
  float dragon_dist =
      std::sqrt(to_dragon_x * to_dragon_x + to_dragon_y * to_dragon_y +
                to_dragon_z * to_dragon_z);

  obs[16] = d.health / DRAGON_MAX_HEALTH;              // dragon_health
  obs[17] = d.position[0] / 100.0f;                   // dragon_x
  obs[18] = (d.position[1] - 64.0f) / 50.0f;          // dragon_y
  obs[19] = d.position[2] / 100.0f;                   // dragon_z
  obs[20] = d.velocity[0] / 2.0f;                     // dragon_vel_x
  obs[21] = d.velocity[1] / 2.0f;                     // dragon_vel_y
  obs[22] = d.velocity[2] / 2.0f;                     // dragon_vel_z
  obs[23] = d.yaw / 360.0f;                           // dragon_yaw
  obs[24] = static_cast<float>(d.phase) / 6.0f;        // dragon_phase
  obs[25] = std::clamp(dragon_dist / 150.0f, 0.0f, 1.0f); // dragon_dist
  obs[26] = dragon_dist > 0.1f ? to_dragon_x / dragon_dist : 0.0f; // dir_x
  obs[27] = dragon_dist > 0.1f ? to_dragon_z / dragon_dist : 0.0f; // dir_z
  obs[28] = (d.phase == DRAGON_PERCHING && dragon_dist < 5.0f)
                ? 1.0f : 0.0f;                         // can_hit_dragon
  obs[29] = (d.phase == DRAGON_CHARGING || d.phase == DRAGON_PERCHING)
                ? 1.0f : 0.0f;                         // dragon_attacking
  obs[30] = static_cast<float>(p.attack_cooldown) /
            static_cast<float>(PLAYER_ATTACK_COOLDOWN_TICKS); // reserved1 (debug)
  obs[31] = static_cast<float>(p.flags);               // reserved2 (debug)

  // === Environment state (16 floats) ===
  uint32_t crystals_alive = NUM_CRYSTALS - gs.crystals_destroyed;
  obs[32] = static_cast<float>(crystals_alive) /
            static_cast<float>(NUM_CRYSTALS);           // crystals_remaining

  // Find nearest crystal
  float nearest_dist = 1000.0f;
  float nearest_dir_x = 0.0f;
  float nearest_dir_z = 0.0f;
  float nearest_y = 0.0f;
  const Crystal *env_crystals = &crystals_[env_id * NUM_CRYSTALS];
  for (uint32_t i = 0; i < NUM_CRYSTALS; i++) {
    if (env_crystals[i].is_alive < 0.5f)
      continue;

    float tcx = env_crystals[i].position[0] - p.position[0];
    float tcy = env_crystals[i].position[1] - p.position[1];
    float tcz = env_crystals[i].position[2] - p.position[2];
    float dist = std::sqrt(tcx * tcx + tcy * tcy + tcz * tcz);
    if (dist < nearest_dist) {
      nearest_dist = dist;
      if (dist > 0.01f) {
        nearest_dir_x = tcx / dist;
        nearest_dir_z = tcz / dist;
      }
      nearest_y = env_crystals[i].position[1];
    }
  }

  obs[33] = std::clamp(nearest_dist / 100.0f, 0.0f, 1.0f);  // nearest_crystal_dist
  obs[34] = nearest_dir_x;                                    // nearest_crystal_dir_x
  obs[35] = nearest_dir_z;                                    // nearest_crystal_dir_z
  obs[36] = (nearest_y - 64.0f) / 50.0f;                     // nearest_crystal_y
  obs[37] = d.health <= 0.0f ? 1.0f : 0.0f;                  // portal_active
  float portal_dist_xz = std::sqrt(p.position[0] * p.position[0] +
                                   p.position[2] * p.position[2]);
  obs[38] = std::clamp(portal_dist_xz / 100.0f, 0.0f, 1.0f); // portal_dist
  obs[39] = 1.0f - static_cast<float>(gs.tickNumber) / 24000.0f; // time_remaining
  obs[40] = (DRAGON_MAX_HEALTH - d.health) / DRAGON_MAX_HEALTH;   // total_damage_dealt
  obs[41] = 0.0f; // reserved3
  obs[42] = 0.0f; // reserved4
  obs[43] = 0.0f; // reserved5
  obs[44] = 0.0f; // reserved6
  obs[45] = 0.0f; // reserved7
  obs[46] = 0.0f; // reserved8
  obs[47] = 0.0f; // reserved9
}

// ============================================================================
// Accessors
// ============================================================================

uint64_t CpuBackend::get_seed(uint32_t env_id) const {
  if (env_id >= config_.num_envs) {
    return 0;
  }
  return world_seeds_[env_id].get_seed();
}

const float *CpuBackend::get_observations() const {
  return observations_.data();
}

const float *CpuBackend::get_rewards() const { return rewards_.data(); }

const uint8_t *CpuBackend::get_dones() const { return dones_.data(); }

} // namespace mc189
