#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mc189 {

enum class GameStage : uint32_t {
  BASIC_SURVIVAL = 1,
  RESOURCE_GATHERING = 2,
  NETHER_NAVIGATION = 3,
  ENDERMAN_HUNTING = 4,
  STRONGHOLD_FINDING = 5,
  END_FIGHT = 6,
};

struct StageConfig {
  GameStage stage;
  uint32_t max_ticks;
  float spawn_x, spawn_y, spawn_z;
  uint32_t dimension; // 0=overworld, -1(0xFFFFFFFF)=nether, 1=end
  bool require_portal_build;
  std::vector<uint32_t> initial_inventory;
};

// Constexpr stage metadata (without vector member for compile-time init).
struct StageDefaults {
  GameStage stage;
  uint32_t max_ticks;
  float spawn_x, spawn_y, spawn_z;
  uint32_t dimension;
  bool require_portal_build;
};

inline constexpr std::array<StageDefaults, 6> DEFAULT_STAGE_CONFIGS = {{
    {GameStage::BASIC_SURVIVAL, 6000, 0.0f, 64.0f, 0.0f, 0, false},
    {GameStage::RESOURCE_GATHERING, 12000, 0.0f, 64.0f, 0.0f, 0, true},
    {GameStage::NETHER_NAVIGATION, 12000, 0.0f, 64.0f, 0.0f,
     static_cast<uint32_t>(-1), false},
    {GameStage::ENDERMAN_HUNTING, 12000, 0.0f, 64.0f, 0.0f, 0, false},
    {GameStage::STRONGHOLD_FINDING, 18000, 0.0f, 64.0f, 0.0f, 0, false},
    {GameStage::END_FIGHT, 24000, 0.0f, 64.0f, 0.0f, 1, false},
}};

inline const char *stage_name(GameStage stage) {
  switch (stage) {
  case GameStage::BASIC_SURVIVAL:
    return "Basic Survival";
  case GameStage::RESOURCE_GATHERING:
    return "Resource Gathering";
  case GameStage::NETHER_NAVIGATION:
    return "Nether Navigation";
  case GameStage::ENDERMAN_HUNTING:
    return "Enderman Hunting";
  case GameStage::STRONGHOLD_FINDING:
    return "Stronghold Finding";
  case GameStage::END_FIGHT:
    return "End Fight";
  }
  return "Unknown";
}

inline StageConfig get_stage_config(GameStage stage) {
  uint32_t idx = static_cast<uint32_t>(stage) - 1;
  if (idx >= DEFAULT_STAGE_CONFIGS.size()) {
    throw std::out_of_range("Invalid GameStage value");
  }
  const auto &d = DEFAULT_STAGE_CONFIGS[idx];
  return StageConfig{
      .stage = d.stage,
      .max_ticks = d.max_ticks,
      .spawn_x = d.spawn_x,
      .spawn_y = d.spawn_y,
      .spawn_z = d.spawn_z,
      .dimension = d.dimension,
      .require_portal_build = d.require_portal_build,
      .initial_inventory = {},
  };
}

inline GameStage next_stage(GameStage current) {
  uint32_t val = static_cast<uint32_t>(current);
  if (val >= static_cast<uint32_t>(GameStage::END_FIGHT)) {
    return GameStage::END_FIGHT; // No stage beyond END_FIGHT
  }
  return static_cast<GameStage>(val + 1);
}

} // namespace mc189
