#pragma once

// resource_detection.h - Stage 2 survival resource detection
// Detects trees, exposed stone, caves, and ores via raycast cone sampling

#include <cstdint>
#include <memory>
#include <vector>

namespace mc189 {

// Detection constants
constexpr float RESOURCE_MAX_RANGE = 64.0f;  // Max detection range in blocks
constexpr size_t RESOURCE_OBS_SIZE = 16;     // Observation floats per env
constexpr size_t SURVIVAL_OBS_SIZE = 64;     // Total with base obs (48 + 16)

// Resource detection result for a single environment
struct ResourceDetection {
  // Trees (wood logs)
  float nearest_tree_distance;    // Normalized 0-1 for 0-64 blocks
  float nearest_tree_dir_x;       // Normalized direction X
  float nearest_tree_dir_z;       // Normalized direction Z

  // Exposed stone (surface or shallow caves)
  float nearest_exposed_stone;    // Normalized distance
  float stone_dir_x;
  float stone_dir_z;

  // Cave entrances (darkness + depth indicator)
  float nearest_cave_entrance;    // Normalized distance
  float cave_dir_x;
  float cave_dir_z;

  // Iron ore (visible in caves, y < 64)
  float nearest_iron_ore;         // Normalized distance, 1.0 if not visible
  float iron_dir_x;
  float iron_dir_z;

  // Coal (common, useful for torches/smelting)
  float nearest_coal;
  float coal_dir_x;
  float coal_dir_z;

  // Aggregates
  float any_resource_nearby;      // 1.0 if any resource within 16 blocks
};

// Pack ResourceDetection into float array
inline void pack_detection(const ResourceDetection &det, float *out) {
  out[0] = det.nearest_tree_distance;
  out[1] = det.nearest_tree_dir_x;
  out[2] = det.nearest_tree_dir_z;
  out[3] = det.nearest_exposed_stone;
  out[4] = det.stone_dir_x;
  out[5] = det.stone_dir_z;
  out[6] = det.nearest_cave_entrance;
  out[7] = det.cave_dir_x;
  out[8] = det.cave_dir_z;
  out[9] = det.nearest_iron_ore;
  out[10] = det.iron_dir_x;
  out[11] = det.iron_dir_z;
  out[12] = det.nearest_coal;
  out[13] = det.coal_dir_x;
  out[14] = det.coal_dir_z;
  out[15] = det.any_resource_nearby;
}

// Unpack float array into ResourceDetection
inline ResourceDetection unpack_detection(const float *in) {
  return ResourceDetection{
      in[0],  in[1],  in[2],  in[3],  in[4],  in[5],  in[6],  in[7],
      in[8],  in[9],  in[10], in[11], in[12], in[13], in[14], in[15]};
}

// World query interface for resource detection
// Implement this to provide block/light data from your world representation
class WorldQuery {
public:
  virtual ~WorldQuery() = default;

  // Get block ID at world position
  virtual int get_block(int x, int y, int z) const = 0;

  // Get light level (0-15) at position
  virtual int get_light(int x, int y, int z) const = 0;

  // Check if position is air (passable)
  virtual bool is_air(int x, int y, int z) const = 0;

  // Check if position is solid (blocks movement)
  virtual bool is_solid(int x, int y, int z) const = 0;
};

// Simple procedural world for CPU fallback (seeded Perlin terrain)
class ProceduralWorld : public WorldQuery {
public:
  explicit ProceduralWorld(uint32_t seed);

  int get_block(int x, int y, int z) const override;
  int get_light(int x, int y, int z) const override;
  bool is_air(int x, int y, int z) const override;
  bool is_solid(int x, int y, int z) const override;

private:
  uint32_t seed_;
  uint32_t hash(int x, int y, int z) const;
  float noise(int x, int y, int z) const;
  bool is_cave(int x, int y, int z) const;
  int get_ore(int x, int y, int z) const;
  bool is_tree_trunk(int x, int z) const;
  bool is_tree_canopy(int x, int y, int z) const;
};

// Perform resource detection from player position
// Returns ResourceDetection with normalized distances (0-1 for 0-64 blocks)
// and normalized direction vectors
ResourceDetection detect_resources(const WorldQuery &world,
                                   float player_x, float player_y, float player_z,
                                   float player_yaw, float player_pitch);

// Batch resource detection for vectorized environments
// player_positions: [num_envs, 3] array of (x, y, z)
// player_orientations: [num_envs, 2] array of (yaw, pitch)
// output: [num_envs, 16] array to fill with resource observations
void detect_resources_batch(const std::vector<std::unique_ptr<WorldQuery>> &worlds,
                           const float *player_positions,
                           const float *player_orientations,
                           float *output,
                           uint32_t num_envs);

// Resource detector class for integration with MC189Simulator
// Maintains world state and provides batch update interface
class ResourceDetector {
public:
  explicit ResourceDetector(uint32_t seed = 12345);

  // Update observations with resource detection for a batch of environments
  // observations: [num_envs, SURVIVAL_OBS_SIZE] - will write to indices 48-63
  // player_positions: [num_envs, 3]
  // player_orientations: [num_envs, 2]
  void update_observations(float *observations, uint32_t num_envs,
                          const float *player_positions,
                          const float *player_orientations);

  // Get seed
  uint32_t seed() const { return seed_; }

  // Set seed (regenerates worlds)
  void set_seed(uint32_t seed) { seed_ = seed; }

private:
  uint32_t seed_;
};

// Observation layout for Stage 2 survival mode
// Extends the 48-float dragon fight observation with 16 resource floats
//
// Indices 0-47:  Base observation (player state, dragon, crystals, etc.)
// Indices 48-63: Resource detection
//
// Resource observation layout:
//   48: nearest_tree_distance    - Distance to nearest tree (0-1)
//   49: nearest_tree_dir_x       - Direction X to tree (normalized)
//   50: nearest_tree_dir_z       - Direction Z to tree (normalized)
//   51: nearest_exposed_stone    - Distance to exposed stone
//   52: stone_dir_x              - Direction X to stone
//   53: stone_dir_z              - Direction Z to stone
//   54: nearest_cave_entrance    - Distance to cave entrance
//   55: cave_dir_x               - Direction X to cave
//   56: cave_dir_z               - Direction Z to cave
//   57: nearest_iron_ore         - Distance to visible iron ore
//   58: iron_dir_x               - Direction X to iron
//   59: iron_dir_z               - Direction Z to iron
//   60: nearest_coal             - Distance to coal ore
//   61: coal_dir_x               - Direction X to coal
//   62: coal_dir_z               - Direction Z to coal
//   63: any_resource_nearby      - 1.0 if any resource within 16 blocks

} // namespace mc189
