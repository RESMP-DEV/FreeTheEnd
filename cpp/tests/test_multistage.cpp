// test_multistage.cpp - Integration tests for multistage Minecraft 1.8.9 simulator
// Tests stage progression, dimension transitions, extended observations, and performance.
// Run with: ./mc189_multistage_test

#include <gtest/gtest.h>
#include "mc189/multistage_simulator.h"
#include "mc189/game_stage.h"
#include "mc189/dimension.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

using namespace mc189;

// =============================================================================
// MultistageTest fixture
// =============================================================================
class MultistageTest : public ::testing::Test {
protected:
    void SetUp() override {
        MultistageSimulator::Config cfg;
        cfg.num_envs = 4;
        cfg.initial_stage = GameStage::BASIC_SURVIVAL;
        cfg.shader_dir = "shaders";

        sim_ = std::make_unique<MultistageSimulator>(cfg);
    }

    std::unique_ptr<MultistageSimulator> sim_;
};

// =============================================================================
// TEST 1: Initial state is BASIC_SURVIVAL in OVERWORLD
// =============================================================================
TEST_F(MultistageTest, InitializesToBasicSurvival) {
    for (uint32_t i = 0; i < sim_->num_envs(); ++i) {
        EXPECT_EQ(sim_->get_stage(i), GameStage::BASIC_SURVIVAL)
            << "Env " << i << " should start at BASIC_SURVIVAL";
        EXPECT_EQ(sim_->get_dimension(i), Dimension::OVERWORLD)
            << "Env " << i << " should start in OVERWORLD";
    }
}

// =============================================================================
// TEST 2: Reset restores initial state
// =============================================================================
TEST_F(MultistageTest, ResetRestoresInitialState) {
    // Step a few times to advance state
    std::vector<int32_t> actions(sim_->num_envs(), 1);
    for (int i = 0; i < 100; ++i) {
        sim_->step(actions.data(), actions.size());
    }

    // Reset all environments
    sim_->reset();

    // Should be back to initial state
    for (uint32_t i = 0; i < sim_->num_envs(); ++i) {
        EXPECT_EQ(sim_->get_stage(i), GameStage::BASIC_SURVIVAL)
            << "Env " << i << " should be BASIC_SURVIVAL after reset";
    }
}

// =============================================================================
// TEST 3: Stage progression via set_stage
// =============================================================================
TEST_F(MultistageTest, StageProgressionWorks) {
    sim_->set_stage(0, GameStage::NETHER_NAVIGATION);
    EXPECT_EQ(sim_->get_stage(0), GameStage::NETHER_NAVIGATION);

    sim_->set_stage(0, GameStage::END_FIGHT);
    EXPECT_EQ(sim_->get_stage(0), GameStage::END_FIGHT);

    // Other environments unaffected
    EXPECT_EQ(sim_->get_stage(1), GameStage::BASIC_SURVIVAL);
    EXPECT_EQ(sim_->get_stage(2), GameStage::BASIC_SURVIVAL);
}

// =============================================================================
// TEST 4: All stages are settable and queryable
// =============================================================================
TEST_F(MultistageTest, AllStagesSettable) {
    const GameStage stages[] = {
        GameStage::BASIC_SURVIVAL,
        GameStage::RESOURCE_GATHERING,
        GameStage::NETHER_NAVIGATION,
        GameStage::ENDERMAN_HUNTING,
        GameStage::STRONGHOLD_FINDING,
        GameStage::END_FIGHT,
    };

    for (GameStage s : stages) {
        sim_->set_stage(0, s);
        EXPECT_EQ(sim_->get_stage(0), s)
            << "Failed to set stage to " << stage_name(s);
    }
}

// =============================================================================
// TEST 5: Dimension teleport works
// =============================================================================
TEST_F(MultistageTest, DimensionTeleportWorks) {
    sim_->teleport_to_dimension(0, Dimension::NETHER, 0.0f, 64.0f, 0.0f);
    EXPECT_EQ(sim_->get_dimension(0), Dimension::NETHER);

    sim_->teleport_to_dimension(0, Dimension::END, 0.0f, 64.0f, 0.0f);
    EXPECT_EQ(sim_->get_dimension(0), Dimension::END);

    sim_->teleport_to_dimension(0, Dimension::OVERWORLD, 100.0f, 64.0f, 100.0f);
    EXPECT_EQ(sim_->get_dimension(0), Dimension::OVERWORLD);
}

// =============================================================================
// TEST 6: Per-environment dimension independence
// =============================================================================
TEST_F(MultistageTest, PerEnvDimensionIndependent) {
    sim_->teleport_to_dimension(0, Dimension::NETHER, 0, 64, 0);
    sim_->teleport_to_dimension(1, Dimension::END, 0, 64, 0);
    sim_->teleport_to_dimension(2, Dimension::OVERWORLD, 0, 64, 0);

    EXPECT_EQ(sim_->get_dimension(0), Dimension::NETHER);
    EXPECT_EQ(sim_->get_dimension(1), Dimension::END);
    EXPECT_EQ(sim_->get_dimension(2), Dimension::OVERWORLD);
    EXPECT_EQ(sim_->get_dimension(3), Dimension::OVERWORLD); // Untouched
}

// =============================================================================
// TEST 7: Extended observations pointer is non-null and correct size
// =============================================================================
TEST_F(MultistageTest, ExtendedObservationsNotNull) {
    const float *obs = sim_->get_extended_observations();
    ASSERT_NE(obs, nullptr) << "Extended observations should not be null";

    // Size should be num_envs * 256
    EXPECT_EQ(MultistageSimulator::EXTENDED_OBS_SIZE, 256u);
}

// =============================================================================
// TEST 8: Observations normalized after step
// =============================================================================
TEST_F(MultistageTest, ObservationsNormalized) {
    sim_->reset();

    std::vector<int32_t> actions(sim_->num_envs(), 0); // No-op
    sim_->step(actions.data(), actions.size());

    const float *obs = sim_->get_extended_observations();
    ASSERT_NE(obs, nullptr);

    for (uint32_t i = 0; i < sim_->num_envs() * MultistageSimulator::EXTENDED_OBS_SIZE; ++i) {
        EXPECT_GE(obs[i], -1.0f) << "Obs[" << i << "] = " << obs[i] << " is below -1";
        EXPECT_LE(obs[i], 2.0f) << "Obs[" << i << "] = " << obs[i] << " is above 2";
    }
}

// =============================================================================
// TEST 9: Rewards buffer is accessible
// =============================================================================
TEST_F(MultistageTest, RewardsAccessible) {
    sim_->reset();

    std::vector<int32_t> actions(sim_->num_envs(), 0);
    sim_->step(actions.data(), actions.size());

    const float *rewards = sim_->get_rewards();
    ASSERT_NE(rewards, nullptr);

    // Rewards should be finite
    for (uint32_t i = 0; i < sim_->num_envs(); ++i) {
        EXPECT_TRUE(std::isfinite(rewards[i]))
            << "Reward[" << i << "] is not finite: " << rewards[i];
    }
}

// =============================================================================
// TEST 10: Dones buffer is accessible
// =============================================================================
TEST_F(MultistageTest, DonesAccessible) {
    sim_->reset();

    std::vector<int32_t> actions(sim_->num_envs(), 0);
    sim_->step(actions.data(), actions.size());

    const uint8_t *dones = sim_->get_dones();
    ASSERT_NE(dones, nullptr);

    // After 1 step no environment should be done
    for (uint32_t i = 0; i < sim_->num_envs(); ++i) {
        EXPECT_EQ(dones[i], 0u)
            << "Env " << i << " should not be done after 1 step";
    }
}

// =============================================================================
// TEST 11: Stage progress starts at zero
// =============================================================================
TEST_F(MultistageTest, StageProgressInitiallyZero) {
    for (uint32_t i = 0; i < sim_->num_envs(); ++i) {
        EXPECT_EQ(sim_->get_stage_progress(i), 0u)
            << "Env " << i << " progress should be 0 initially";
    }
}

// =============================================================================
// TEST 12: Stage advancement resets progress
// =============================================================================
TEST_F(MultistageTest, StageAdvancementResetsProgress) {
    sim_->set_stage(0, GameStage::RESOURCE_GATHERING);
    EXPECT_EQ(sim_->get_stage_progress(0), 0u)
        << "Progress should reset when stage changes";

    sim_->set_stage(0, GameStage::END_FIGHT);
    EXPECT_EQ(sim_->get_stage_progress(0), 0u)
        << "Progress should reset when stage changes";
}

// =============================================================================
// TEST 13: World state accessible
// =============================================================================
TEST_F(MultistageTest, WorldStateAccessible) {
    const WorldState *ws = sim_->get_world_states();
    ASSERT_NE(ws, nullptr);

    for (uint32_t i = 0; i < sim_->num_envs(); ++i) {
        EXPECT_EQ(ws[i].stage, static_cast<uint32_t>(GameStage::BASIC_SURVIVAL))
            << "World state stage mismatch for env " << i;
    }
}

// =============================================================================
// TEST 14: Single-env reset preserves other envs
// =============================================================================
TEST_F(MultistageTest, SingleEnvResetPreservesOthers) {
    // Advance env 0 and 1 to different stages
    sim_->set_stage(0, GameStage::NETHER_NAVIGATION);
    sim_->set_stage(1, GameStage::END_FIGHT);
    sim_->set_stage(2, GameStage::RESOURCE_GATHERING);

    // Reset only env 0
    sim_->reset(0);

    // Env 0 should be back to initial
    EXPECT_EQ(sim_->get_stage(0), GameStage::BASIC_SURVIVAL);
    // Others unchanged
    EXPECT_EQ(sim_->get_stage(1), GameStage::END_FIGHT);
    EXPECT_EQ(sim_->get_stage(2), GameStage::RESOURCE_GATHERING);
}

// =============================================================================
// TEST 15: next_stage() utility works correctly
// =============================================================================
TEST(GameStageUtil, NextStageProgression) {
    EXPECT_EQ(next_stage(GameStage::BASIC_SURVIVAL), GameStage::RESOURCE_GATHERING);
    EXPECT_EQ(next_stage(GameStage::RESOURCE_GATHERING), GameStage::NETHER_NAVIGATION);
    EXPECT_EQ(next_stage(GameStage::NETHER_NAVIGATION), GameStage::ENDERMAN_HUNTING);
    EXPECT_EQ(next_stage(GameStage::ENDERMAN_HUNTING), GameStage::STRONGHOLD_FINDING);
    EXPECT_EQ(next_stage(GameStage::STRONGHOLD_FINDING), GameStage::END_FIGHT);
    // END_FIGHT is terminal
    EXPECT_EQ(next_stage(GameStage::END_FIGHT), GameStage::END_FIGHT);
}

// =============================================================================
// TEST 16: stage_name() returns non-null strings
// =============================================================================
TEST(GameStageUtil, StageNamesNotNull) {
    const GameStage stages[] = {
        GameStage::BASIC_SURVIVAL,
        GameStage::RESOURCE_GATHERING,
        GameStage::NETHER_NAVIGATION,
        GameStage::ENDERMAN_HUNTING,
        GameStage::STRONGHOLD_FINDING,
        GameStage::END_FIGHT,
    };

    for (GameStage s : stages) {
        const char *name = stage_name(s);
        ASSERT_NE(name, nullptr);
        EXPECT_GT(strlen(name), 0u) << "Stage name should not be empty";
    }
}

// =============================================================================
// TEST 17: DimensionConfig values are physically sensible
// =============================================================================
TEST(DimensionUtil, ConfigPhysicalConstraints) {
    const DimensionConfig &overworld = DIMENSION_CONFIGS[1]; // OVERWORLD index
    const DimensionConfig &nether = DIMENSION_CONFIGS[0];    // NETHER index
    const DimensionConfig &end = DIMENSION_CONFIGS[2];       // END index

    EXPECT_EQ(overworld.id, Dimension::OVERWORLD);
    EXPECT_EQ(nether.id, Dimension::NETHER);
    EXPECT_EQ(end.id, Dimension::END);

    // Gravity should be positive (downward)
    EXPECT_GT(overworld.gravity, 0.0f);
    EXPECT_GT(nether.gravity, 0.0f);
    EXPECT_GT(end.gravity, 0.0f);

    // Nether has ceiling, overworld/end do not
    EXPECT_TRUE(nether.has_ceiling);
    EXPECT_FALSE(overworld.has_ceiling);
    EXPECT_FALSE(end.has_ceiling);

    // Overworld has weather, nether/end do not
    EXPECT_TRUE(overworld.has_weather);
    EXPECT_FALSE(nether.has_weather);
    EXPECT_FALSE(end.has_weather);

    // Y bounds: max > min
    EXPECT_GT(overworld.max_y, overworld.min_y);
    EXPECT_GT(nether.max_y, nether.min_y);
    EXPECT_GT(end.max_y, end.min_y);

    // Nether is height-limited to 128
    EXPECT_EQ(nether.max_y, 128);
}

// =============================================================================
// TEST 18: get_stage_config returns valid configs
// =============================================================================
TEST(GameStageUtil, StageConfigValid) {
    const GameStage stages[] = {
        GameStage::BASIC_SURVIVAL,
        GameStage::RESOURCE_GATHERING,
        GameStage::NETHER_NAVIGATION,
        GameStage::ENDERMAN_HUNTING,
        GameStage::STRONGHOLD_FINDING,
        GameStage::END_FIGHT,
    };

    for (GameStage s : stages) {
        StageConfig cfg = get_stage_config(s);
        EXPECT_EQ(cfg.stage, s);
        EXPECT_GT(cfg.max_ticks, 0u) << "Stage " << stage_name(s) << " has 0 max_ticks";
    }

    // Nether stage should use nether dimension
    StageConfig nether_cfg = get_stage_config(GameStage::NETHER_NAVIGATION);
    EXPECT_EQ(nether_cfg.dimension, static_cast<uint32_t>(-1)); // Nether = -1 cast to uint32

    // End fight should use end dimension
    StageConfig end_cfg = get_stage_config(GameStage::END_FIGHT);
    EXPECT_EQ(end_cfg.dimension, 1u); // End = 1
}

// =============================================================================
// TEST 19: Performance - stepping throughput
// =============================================================================
TEST_F(MultistageTest, PerformanceBasicStep) {
    std::vector<int32_t> actions(sim_->num_envs(), 1);

    auto start = std::chrono::high_resolution_clock::now();
    constexpr int kIterations = 1000;
    for (int i = 0; i < kIterations; ++i) {
        sim_->step(actions.data(), actions.size());
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double total_steps = static_cast<double>(kIterations) * sim_->num_envs();
    double sps = (ms > 0) ? total_steps / (static_cast<double>(ms) / 1000.0) : 0.0;

    std::cout << "[MultistageTest] Performance: " << sps << " steps/second ("
              << sim_->num_envs() << " envs, " << kIterations << " iterations, "
              << ms << " ms)" << std::endl;

    // At least 1000 SPS with 4 envs
    EXPECT_GT(sps, 1000.0) << "Stepping performance below minimum threshold";
}

// =============================================================================
// TEST 20: Reset does not leak (run reset many times)
// =============================================================================
TEST_F(MultistageTest, ResetDoesNotLeak) {
    std::vector<int32_t> actions(sim_->num_envs(), 0);

    for (int i = 0; i < 100; ++i) {
        sim_->reset();
        sim_->step(actions.data(), actions.size());

        const float *obs = sim_->get_extended_observations();
        ASSERT_NE(obs, nullptr) << "Observations null after reset " << i;
    }
}

}  // namespace

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
