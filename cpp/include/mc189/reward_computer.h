#pragma once
#include "mc189/simulator.h"
#include "mc189/game_stage.h"
#include <cstdint>
#include <vector>

namespace mc189 {

struct RewardConfig {
    // Per-stage sparse rewards
    float stage_completion_bonus = 10.0f;
    float dragon_kill_bonus = 100.0f;

    // Dense rewards
    float damage_to_dragon_scale = 0.1f;
    float crystal_destroy_bonus = 2.0f;
    float blaze_kill_bonus = 0.5f;
    float enderman_kill_bonus = 0.3f;
    float ender_pearl_pickup = 0.2f;
    float blaze_rod_pickup = 0.3f;
    float iron_pickup = 0.05f;
    float diamond_pickup = 0.2f;
    float portal_lit_bonus = 1.0f;
    float eye_placed_bonus = 0.5f;
    float stronghold_found_bonus = 2.0f;
    float fortress_found_bonus = 1.5f;

    // Penalties
    float death_penalty = -1.0f;
    float time_penalty_per_tick = -0.0001f;
    float health_loss_penalty = -0.01f;

    // Exploration
    float new_chunk_bonus = 0.001f;
    float progress_toward_goal = 0.001f;  // Getting closer to objective
};

class RewardComputer {
public:
    RewardComputer(uint32_t num_envs, const RewardConfig& config = {});

    // Compute reward for current tick
    float compute(uint32_t env_id,
                 const WorldState& prev_world,
                 const WorldState& curr_world,
                 const PlayerFull& prev_player,
                 const PlayerFull& curr_player,
                 const Dragon* prev_dragon,
                 const Dragon* curr_dragon,
                 bool player_died);

    // Check if episode should terminate
    bool check_done(uint32_t env_id,
                   const WorldState& world,
                   const Dragon* dragon,
                   bool player_died,
                   bool& out_success);

    // Stage-specific objective checking
    bool check_stage_objective(GameStage stage,
                              const WorldState& world,
                              const PlayerFull& player);

    // Reset tracking for new episode
    void reset(uint32_t env_id);

    // Statistics
    float get_total_reward(uint32_t env_id) const;
    float get_episode_length(uint32_t env_id) const;

private:
    float compute_stage_reward(uint32_t env_id, GameStage stage,
                              const WorldState& prev, const WorldState& curr,
                              const PlayerFull& prev_p, const PlayerFull& curr_p);
    float compute_dragon_reward(uint32_t env_id,
                               const Dragon* prev, const Dragon* curr);
    float compute_exploration_reward(uint32_t env_id,
                                    const WorldState& prev, const WorldState& curr);

    uint32_t num_envs_;
    RewardConfig config_;

    // Tracking per environment
    std::vector<float> total_rewards_;
    std::vector<uint32_t> episode_lengths_;
    std::vector<uint32_t> prev_crystals_destroyed_;
    std::vector<float> prev_dragon_health_;
    std::vector<uint32_t> visited_chunks_;  // Bitset or hash
};

} // namespace mc189
