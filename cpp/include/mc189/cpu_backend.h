#pragma once

#include "mc189/simulator.h"
#include "mc189/world_seed.h"

#include <cstdint>
#include <vector>

namespace mc189 {

/// CPU backend implementing the dragon_fight_mvk.comp shader logic.
/// Provides the same interface as MC189Simulator but executes game ticks
/// on CPU threads with OpenMP parallelism instead of GPU compute shaders.
///
/// The implementation is a faithful port of all 6 shader stages:
///   0: Setup (tick/seed/reward init)
///   1: Player physics (movement, jumping, ground collision, void death)
///   2: Dragon AI (circling, charging, landing, perching, taking off)
///   3: Combat (melee, bow, crystals, dragon attacks, knockback)
///   4: Environment (hunger, regen, starvation, time penalty)
///   5: Observations (normalized 48-float vector)
class CpuBackend {
public:
  struct Config {
    uint32_t num_envs = 1;
    uint32_t num_threads = 0; // 0 = use OMP default (all cores)
  };

  explicit CpuBackend(const Config &config);
  ~CpuBackend();

  // Main simulation interface (matches MC189Simulator)
  void step(const int32_t *actions, size_t num_actions);
  void reset(uint32_t env_id = 0xFFFFFFFF, uint64_t seed = 0);
  uint64_t get_seed(uint32_t env_id = 0) const;

  const float *get_observations() const;
  const float *get_rewards() const;
  const uint8_t *get_dones() const;

  uint32_t num_envs() const { return config_.num_envs; }
  static constexpr size_t obs_dim() { return OBSERVATION_SIZE; }

private:
  // Per-environment simulation stages (match dragon_fight_mvk.comp exactly)
  void stage_setup(uint32_t env_id);
  void stage_player_physics(uint32_t env_id);
  void stage_dragon_ai(uint32_t env_id);
  void stage_combat(uint32_t env_id);
  void stage_environment(uint32_t env_id);
  void stage_observations(uint32_t env_id);

  void decode_action(int32_t action, InputState &inp);
  void reset_env(uint32_t env_id, uint64_t seed);

  // Shader utility ports
  static uint32_t hash(uint32_t x);
  static float rng(uint32_t &seed);
  static float get_ground_height(float px, float pz);
  static void get_pillar_position(uint32_t idx, float *out);

  Config config_;

  // Per-environment state (matches GPU buffer layout)
  std::vector<Player> players_;
  std::vector<Dragon> dragons_;
  std::vector<Crystal> crystals_; // NUM_CRYSTALS per env
  std::vector<GameState> game_states_;
  std::vector<InputState> inputs_;
  std::vector<WorldSeed> world_seeds_;

  // Output buffers
  std::vector<float> observations_;
  std::vector<float> rewards_;
  std::vector<uint8_t> dones_;

  uint32_t tick_number_ = 0;
};

} // namespace mc189
