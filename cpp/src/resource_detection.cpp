// resource_detection.cpp - Stage 2 survival resource detection
// Detects trees, exposed stone, caves, and ores via raycast cone sampling
// For "Free The End" RL progression: wood → stone → iron chain

#include "mc189/simulator.h"

#include <algorithm>
#include <cmath>

namespace mc189 {

// Block IDs (MC 1.8.9)
namespace BlockId {
constexpr int AIR = 0;
constexpr int STONE = 1;
constexpr int GRASS = 2;
constexpr int DIRT = 3;
constexpr int COBBLESTONE = 4;
constexpr int OAK_LOG = 17;
constexpr int BIRCH_LOG = 17;   // Same ID, different data value
constexpr int SPRUCE_LOG = 17;
constexpr int JUNGLE_LOG = 17;
constexpr int OAK_LEAVES = 18;
constexpr int COAL_ORE = 16;
constexpr int IRON_ORE = 15;
constexpr int GOLD_ORE = 14;
constexpr int DIAMOND_ORE = 56;
constexpr int WATER = 9;
constexpr int LAVA = 11;
} // namespace BlockId

// Detection constants
constexpr float MAX_DETECTION_RANGE = 64.0f;
constexpr float CONE_HALF_ANGLE = 0.7854f; // 45 degrees
constexpr int CONE_RAYS = 32;              // Rays per scan
constexpr int VERTICAL_RAYS = 8;           // Vertical divisions
constexpr float TREE_Y_THRESHOLD = 70.0f;  // Trees spawn below this
constexpr float CAVE_DARKNESS_THRESHOLD = 7.0f; // Light level for caves
constexpr float IRON_ORE_MAX_Y = 64.0f;    // Iron spawns y < 64

// Resource detection result for a single environment
struct ResourceDetection {
  // Trees (wood logs)
  float nearest_tree_distance;    // Normalized 0-1 for 0-64 blocks
  float nearest_tree_dir_x;       // Normalized direction
  float nearest_tree_dir_z;

  // Exposed stone (surface or shallow)
  float nearest_exposed_stone;    // Normalized distance
  float stone_dir_x;
  float stone_dir_z;

  // Cave entrances (darkness + depth indicator)
  float nearest_cave_entrance;    // Normalized distance
  float cave_dir_x;
  float cave_dir_z;

  // Iron ore (if visible in caves, y < 64)
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

// World query interface - abstract to support both CPU and GPU backends
class WorldQuery {
public:
  virtual ~WorldQuery() = default;

  // Get block at world position
  virtual int get_block(int x, int y, int z) const = 0;

  // Get light level (0-15)
  virtual int get_light(int x, int y, int z) const = 0;

  // Check if position is air (for raycast)
  virtual bool is_air(int x, int y, int z) const = 0;

  // Check if position is solid (for collision)
  virtual bool is_solid(int x, int y, int z) const = 0;
};

// Simple procedural world for CPU fallback
class ProceduralWorld : public WorldQuery {
public:
  explicit ProceduralWorld(uint32_t seed) : seed_(seed) {}

  int get_block(int x, int y, int z) const override {
    // Ground level at y=64
    if (y < 0) return BlockId::STONE;
    if (y == 0) return BlockId::STONE;

    // Underground
    if (y < 64) {
      // Caves
      if (is_cave(x, y, z)) return BlockId::AIR;

      // Ores
      int ore = get_ore(x, y, z);
      if (ore != BlockId::STONE) return ore;

      return BlockId::STONE;
    }

    // Surface
    if (y == 64) return BlockId::GRASS;
    if (y == 65) {
      // Trees (sparse procedural placement)
      if (is_tree_trunk(x, z)) return BlockId::OAK_LOG;
    }
    if (y >= 66 && y <= 70) {
      // Tree canopy
      if (is_tree_canopy(x, y, z)) return BlockId::OAK_LEAVES;
    }

    return BlockId::AIR;
  }

  int get_light(int x, int y, int z) const override {
    if (y >= 64) return 15; // Full daylight
    if (is_cave(x, y, z)) return 0; // Dark cave
    return 0; // Underground
  }

  bool is_air(int x, int y, int z) const override {
    return get_block(x, y, z) == BlockId::AIR;
  }

  bool is_solid(int x, int y, int z) const override {
    int block = get_block(x, y, z);
    return block != BlockId::AIR && block != BlockId::WATER;
  }

private:
  uint32_t seed_;

  // Simple hash for procedural generation
  uint32_t hash(int x, int y, int z) const {
    uint32_t h = seed_;
    h ^= static_cast<uint32_t>(x) * 374761393u;
    h ^= static_cast<uint32_t>(y) * 668265263u;
    h ^= static_cast<uint32_t>(z) * 2654435761u;
    h ^= h >> 13;
    h *= 1274126177u;
    return h;
  }

  float noise(int x, int y, int z) const {
    return static_cast<float>(hash(x, y, z) & 0xFFFF) / 65535.0f;
  }

  bool is_cave(int x, int y, int z) const {
    // 3D noise for caves
    float n = noise(x / 8, y / 4, z / 8);
    return n > 0.7f && y > 5 && y < 60;
  }

  int get_ore(int x, int y, int z) const {
    float n = noise(x, y, z);

    // Diamond (y 1-16)
    if (y <= 16 && n > 0.995f) return BlockId::DIAMOND_ORE;

    // Gold (y 1-32)
    if (y <= 32 && n > 0.98f) return BlockId::GOLD_ORE;

    // Iron (y 1-64)
    if (y <= 64 && n > 0.94f) return BlockId::IRON_ORE;

    // Coal (y 1-128)
    if (y <= 128 && n > 0.90f) return BlockId::COAL_ORE;

    return BlockId::STONE;
  }

  bool is_tree_trunk(int x, int z) const {
    // Trees at grid points with noise offset
    int gx = (x + 8) / 16;
    int gz = (z + 8) / 16;
    float n = noise(gx * 1000, 0, gz * 1000);
    if (n < 0.3f) return false; // 30% chance per grid cell

    // Tree position within cell
    int tx = gx * 16 + static_cast<int>(noise(gx, 1, gz) * 8.0f);
    int tz = gz * 16 + static_cast<int>(noise(gx, 2, gz) * 8.0f);

    return x == tx && z == tz;
  }

  bool is_tree_canopy(int x, int y, int z) const {
    // Check if near a tree trunk
    for (int dx = -2; dx <= 2; dx++) {
      for (int dz = -2; dz <= 2; dz++) {
        if (is_tree_trunk(x + dx, z + dz)) {
          int dist = std::abs(dx) + std::abs(dz);
          int height = y - 65;
          // Canopy shape: wider at bottom, narrower at top
          int max_dist = 3 - height;
          if (dist <= max_dist && max_dist > 0) return true;
        }
      }
    }
    return false;
  }
};

// Raycast result
struct RayHit {
  bool hit;
  float distance;
  int block_id;
  int x, y, z;
};

// Cast a ray from origin in direction, find first non-air block
RayHit raycast(const WorldQuery &world, float ox, float oy, float oz,
               float dx, float dy, float dz, float max_dist) {
  RayHit result{false, max_dist, BlockId::AIR, 0, 0, 0};

  // Normalize direction
  float len = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (len < 0.0001f) return result;
  dx /= len;
  dy /= len;
  dz /= len;

  // Step through world using DDA-like approach
  float t = 0.0f;
  const float step = 0.5f; // Half-block steps for accuracy

  while (t < max_dist) {
    int bx = static_cast<int>(std::floor(ox + dx * t));
    int by = static_cast<int>(std::floor(oy + dy * t));
    int bz = static_cast<int>(std::floor(oz + dz * t));

    if (!world.is_air(bx, by, bz)) {
      result.hit = true;
      result.distance = t;
      result.block_id = world.get_block(bx, by, bz);
      result.x = bx;
      result.y = by;
      result.z = bz;
      return result;
    }

    t += step;
  }

  return result;
}

// Check if block is a wood log (any type)
bool is_wood_log(int block_id) {
  return block_id == BlockId::OAK_LOG; // All logs share ID 17
}

// Check if block is exposed stone (visible from surface)
bool is_exposed_stone(int block_id) {
  return block_id == BlockId::STONE || block_id == BlockId::COBBLESTONE;
}

// Check if block is iron ore
bool is_iron_ore(int block_id) {
  return block_id == BlockId::IRON_ORE;
}

// Check if block is coal ore
bool is_coal_ore(int block_id) {
  return block_id == BlockId::COAL_ORE;
}

// Perform cone-based resource detection from player position
ResourceDetection detect_resources(const WorldQuery &world,
                                   float player_x, float player_y, float player_z,
                                   float player_yaw, float player_pitch) {
  ResourceDetection result{};

  // Initialize to "not found" (max distance = 1.0 normalized)
  result.nearest_tree_distance = 1.0f;
  result.nearest_exposed_stone = 1.0f;
  result.nearest_cave_entrance = 1.0f;
  result.nearest_iron_ore = 1.0f;
  result.nearest_coal = 1.0f;
  result.any_resource_nearby = 0.0f;

  // Eye position (player eye height is 1.62 blocks)
  float eye_x = player_x;
  float eye_y = player_y + 1.62f;
  float eye_z = player_z;

  // Player look direction
  float yaw_rad = player_yaw * 3.14159f / 180.0f;
  float pitch_rad = player_pitch * 3.14159f / 180.0f;

  float look_x = -std::sin(yaw_rad) * std::cos(pitch_rad);
  float look_y = -std::sin(pitch_rad);
  float look_z = std::cos(yaw_rad) * std::cos(pitch_rad);

  // Generate cone rays around look direction
  // Use spherical coordinates with look direction as pole
  for (int v = 0; v < VERTICAL_RAYS; v++) {
    float v_angle = -CONE_HALF_ANGLE + (2.0f * CONE_HALF_ANGLE * v) / (VERTICAL_RAYS - 1);

    for (int h = 0; h < CONE_RAYS; h++) {
      float h_angle = (2.0f * 3.14159f * h) / CONE_RAYS;

      // Rotate around look direction
      // Using simplified cone: adjust pitch and yaw from look dir
      float ray_yaw = yaw_rad + std::cos(h_angle) * CONE_HALF_ANGLE;
      float ray_pitch = pitch_rad + v_angle + std::sin(h_angle) * CONE_HALF_ANGLE * 0.5f;

      float ray_x = -std::sin(ray_yaw) * std::cos(ray_pitch);
      float ray_y = -std::sin(ray_pitch);
      float ray_z = std::cos(ray_yaw) * std::cos(ray_pitch);

      // Cast ray
      RayHit hit = raycast(world, eye_x, eye_y, eye_z, ray_x, ray_y, ray_z, MAX_DETECTION_RANGE);

      if (!hit.hit) continue;

      // Normalized distance (0-1 for 0-64 blocks)
      float norm_dist = hit.distance / MAX_DETECTION_RANGE;

      // Direction to hit point
      float to_x = hit.x - player_x;
      float to_z = hit.z - player_z;
      float to_len = std::sqrt(to_x * to_x + to_z * to_z);
      if (to_len > 0.001f) {
        to_x /= to_len;
        to_z /= to_len;
      }

      // Check for trees (wood logs, typically y > 64)
      if (is_wood_log(hit.block_id) && hit.y > 64 && hit.y < TREE_Y_THRESHOLD) {
        if (norm_dist < result.nearest_tree_distance) {
          result.nearest_tree_distance = norm_dist;
          result.nearest_tree_dir_x = to_x;
          result.nearest_tree_dir_z = to_z;
        }
      }

      // Check for exposed stone (visible from surface or in shallow caves)
      if (is_exposed_stone(hit.block_id)) {
        // Only count as "exposed" if relatively shallow (y > 50) or in cave
        bool is_shallow = hit.y > 50;
        bool is_cave_stone = hit.y < 60 && world.get_light(hit.x, hit.y + 1, hit.z) < CAVE_DARKNESS_THRESHOLD;

        if (is_shallow || is_cave_stone) {
          if (norm_dist < result.nearest_exposed_stone) {
            result.nearest_exposed_stone = norm_dist;
            result.stone_dir_x = to_x;
            result.stone_dir_z = to_z;
          }
        }
      }

      // Check for cave entrances (dark opening)
      if (hit.block_id == BlockId::AIR || hit.block_id == BlockId::STONE) {
        // A cave entrance is where we see darkness
        int light = world.get_light(hit.x, hit.y, hit.z);
        bool is_dark = light < CAVE_DARKNESS_THRESHOLD;
        bool is_underground = hit.y < 60;

        if (is_dark && is_underground) {
          if (norm_dist < result.nearest_cave_entrance) {
            result.nearest_cave_entrance = norm_dist;
            result.cave_dir_x = to_x;
            result.cave_dir_z = to_z;
          }
        }
      }

      // Check for iron ore (visible in caves, y < 64)
      if (is_iron_ore(hit.block_id) && hit.y < IRON_ORE_MAX_Y) {
        if (norm_dist < result.nearest_iron_ore) {
          result.nearest_iron_ore = norm_dist;
          result.iron_dir_x = to_x;
          result.iron_dir_z = to_z;
        }
      }

      // Check for coal ore
      if (is_coal_ore(hit.block_id)) {
        if (norm_dist < result.nearest_coal) {
          result.nearest_coal = norm_dist;
          result.coal_dir_x = to_x;
          result.coal_dir_z = to_z;
        }
      }
    }
  }

  // Calculate aggregate "any resource nearby" flag
  const float NEARBY_THRESHOLD = 16.0f / MAX_DETECTION_RANGE; // 16 blocks
  if (result.nearest_tree_distance < NEARBY_THRESHOLD ||
      result.nearest_exposed_stone < NEARBY_THRESHOLD ||
      result.nearest_cave_entrance < NEARBY_THRESHOLD ||
      result.nearest_iron_ore < NEARBY_THRESHOLD ||
      result.nearest_coal < NEARBY_THRESHOLD) {
    result.any_resource_nearby = 1.0f;
  }

  return result;
}

// Batch resource detection for vectorized environments
void detect_resources_batch(const std::vector<std::unique_ptr<WorldQuery>> &worlds,
                           const float *player_positions, // [n, 3]
                           const float *player_orientations, // [n, 2] (yaw, pitch)
                           float *output, // [n, 16] output observations
                           uint32_t num_envs) {
  for (uint32_t i = 0; i < num_envs; i++) {
    const WorldQuery &world = *worlds[i];
    float px = player_positions[i * 3 + 0];
    float py = player_positions[i * 3 + 1];
    float pz = player_positions[i * 3 + 2];
    float yaw = player_orientations[i * 2 + 0];
    float pitch = player_orientations[i * 2 + 1];

    ResourceDetection det = detect_resources(world, px, py, pz, yaw, pitch);

    // Pack into output array (16 floats per env)
    float *out = output + i * 16;
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
}

// Extended observation struct for Stage 2 survival mode
// Adds 16 floats to the existing 48-float dragon fight observation
struct SurvivalObservation {
  // Original 48 floats from dragon fight...
  // (inherited from Observation struct)

  // Resource detection (16 floats)
  float nearest_tree_distance;    // Index 48
  float nearest_tree_dir_x;       // Index 49
  float nearest_tree_dir_z;       // Index 50
  float nearest_exposed_stone;    // Index 51
  float stone_dir_x;              // Index 52
  float stone_dir_z;              // Index 53
  float nearest_cave_entrance;    // Index 54
  float cave_dir_x;               // Index 55
  float cave_dir_z;               // Index 56
  float nearest_iron_ore;         // Index 57
  float iron_dir_x;               // Index 58
  float iron_dir_z;               // Index 59
  float nearest_coal;             // Index 60
  float coal_dir_x;               // Index 61
  float coal_dir_z;               // Index 62
  float any_resource_nearby;      // Index 63
};

constexpr size_t SURVIVAL_OBSERVATION_SIZE = 64;

// Resource detector class for integration with MC189Simulator
class ResourceDetector {
public:
  explicit ResourceDetector(uint32_t seed = 12345) : seed_(seed) {}

  // Update observations with resource detection for a batch of environments
  void update_observations(float *observations, uint32_t num_envs,
                          const float *player_positions,
                          const float *player_orientations) {
    // Create procedural worlds for each environment
    std::vector<std::unique_ptr<WorldQuery>> worlds;
    worlds.reserve(num_envs);
    for (uint32_t i = 0; i < num_envs; i++) {
      worlds.push_back(std::make_unique<ProceduralWorld>(seed_ + i));
    }

    // Temporary buffer for resource observations
    std::vector<float> resource_obs(num_envs * 16);

    // Detect resources
    detect_resources_batch(worlds, player_positions, player_orientations,
                          resource_obs.data(), num_envs);

    // Append to existing observations (assumes 48-float base)
    // Or write to reserved slots if observation buffer is pre-allocated
    for (uint32_t i = 0; i < num_envs; i++) {
      float *obs = observations + i * SURVIVAL_OBSERVATION_SIZE;
      float *res = resource_obs.data() + i * 16;

      // Copy resource detection to indices 48-63
      std::copy(res, res + 16, obs + 48);
    }
  }

private:
  uint32_t seed_;
};

} // namespace mc189
