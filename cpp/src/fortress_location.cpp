// Fortress location determination for MC 1.8.9
// Fortresses generate on a 256x256 grid in the Nether
// Each cell contains exactly one fortress at a semi-random position

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace mc189 {

// Simple Vec2i for 2D integer coordinates
struct Vec2i {
  int32_t x;
  int32_t z;

  Vec2i() : x(0), z(0) {}
  Vec2i(int32_t x_, int32_t z_) : x(x_), z(z_) {}
};

// Java-compatible LCG random number generator (matches MC 1.8.9)
class JavaRandom {
public:
  explicit JavaRandom(int64_t seed) { set_seed(seed); }

  void set_seed(int64_t seed) {
    // Java's setSeed implementation
    state_ = (seed ^ 0x5DEECE66DLL) & ((1LL << 48) - 1);
  }

  // Returns [0, bound)
  int32_t next_int(int32_t bound) {
    if (bound <= 0) {
      return 0;
    }

    // If bound is power of 2, use optimized path
    if ((bound & -bound) == bound) {
      return static_cast<int32_t>((bound * static_cast<int64_t>(next(31))) >> 31);
    }

    int32_t bits, val;
    do {
      bits = next(31);
      val = bits % bound;
    } while (bits - val + (bound - 1) < 0);

    return val;
  }

private:
  int64_t state_;

  // Generate n bits of randomness
  int32_t next(int32_t bits) {
    state_ = (state_ * 0x5DEECE66DLL + 0xBLL) & ((1LL << 48) - 1);
    return static_cast<int32_t>(state_ >> (48 - bits));
  }
};

// Constants for fortress generation
constexpr int32_t FORTRESS_GRID_SIZE = 256;    // Nether coordinates
constexpr int32_t FORTRESS_OFFSET_MIN = 28;    // Minimum offset within cell
constexpr int32_t FORTRESS_OFFSET_RANGE = 200; // Range of random offset

// Magic multipliers for seed mixing (from MC decompilation)
constexpr int64_t SEED_MUL_X = 341873128712LL;
constexpr int64_t SEED_MUL_Z = 132897987541LL;

// Get the fortress position within a specific grid cell
// cell_x, cell_z: grid cell coordinates (not block coordinates)
// seed: world seed
Vec2i get_fortress_in_cell(int32_t cell_x, int32_t cell_z, uint64_t seed) {
  // Mix seed with cell coordinates
  int64_t mixed_seed = static_cast<int64_t>(seed);
  mixed_seed ^= (static_cast<int64_t>(cell_x) * SEED_MUL_X);
  mixed_seed ^= (static_cast<int64_t>(cell_z) * SEED_MUL_Z);

  JavaRandom rng(mixed_seed);

  // Calculate position within cell
  // Fortress is placed at cell_origin + offset (28-227 range)
  int32_t x = cell_x * FORTRESS_GRID_SIZE + rng.next_int(FORTRESS_OFFSET_RANGE) + FORTRESS_OFFSET_MIN;
  int32_t z = cell_z * FORTRESS_GRID_SIZE + rng.next_int(FORTRESS_OFFSET_RANGE) + FORTRESS_OFFSET_MIN;

  return Vec2i(x, z);
}

// Calculate squared distance for comparison (avoids sqrt)
inline float distance_squared(float x1, float z1, int32_t x2, int32_t z2) {
  float dx = x1 - static_cast<float>(x2);
  float dz = z1 - static_cast<float>(z2);
  return dx * dx + dz * dz;
}

// Find the nearest fortress to a player position
// player_x, player_z: player's current position in Nether coordinates
// seed: world seed
// Returns: fortress coordinates in Nether blocks
Vec2i find_nearest_fortress(float player_x, float player_z, uint64_t seed) {
  // Determine which grid cell the player is in
  int32_t player_cell_x = static_cast<int32_t>(std::floor(player_x / FORTRESS_GRID_SIZE));
  int32_t player_cell_z = static_cast<int32_t>(std::floor(player_z / FORTRESS_GRID_SIZE));

  Vec2i nearest;
  float min_dist_sq = FLT_MAX;

  // Check 3x3 grid of cells around player
  // This covers all cases where the nearest fortress might not be in the player's cell
  for (int32_t dx = -1; dx <= 1; ++dx) {
    for (int32_t dz = -1; dz <= 1; ++dz) {
      int32_t cell_x = player_cell_x + dx;
      int32_t cell_z = player_cell_z + dz;

      Vec2i fortress = get_fortress_in_cell(cell_x, cell_z, seed);

      float dist_sq = distance_squared(player_x, player_z, fortress.x, fortress.z);
      if (dist_sq < min_dist_sq) {
        min_dist_sq = dist_sq;
        nearest = fortress;
      }
    }
  }

  return nearest;
}

// Find all fortresses within a given radius
// player_x, player_z: player's current position in Nether coordinates
// radius: search radius in blocks
// seed: world seed
// out_fortresses: output buffer for fortress positions
// max_fortresses: maximum number of fortresses to return
// Returns: number of fortresses found
int32_t find_fortresses_in_radius(float player_x, float player_z, float radius, uint64_t seed,
                                   Vec2i* out_fortresses, int32_t max_fortresses) {
  // Calculate the range of cells we need to check
  int32_t cell_radius = static_cast<int32_t>(std::ceil(radius / FORTRESS_GRID_SIZE)) + 1;

  int32_t player_cell_x = static_cast<int32_t>(std::floor(player_x / FORTRESS_GRID_SIZE));
  int32_t player_cell_z = static_cast<int32_t>(std::floor(player_z / FORTRESS_GRID_SIZE));

  float radius_sq = radius * radius;
  int32_t count = 0;

  for (int32_t dx = -cell_radius; dx <= cell_radius && count < max_fortresses; ++dx) {
    for (int32_t dz = -cell_radius; dz <= cell_radius && count < max_fortresses; ++dz) {
      int32_t cell_x = player_cell_x + dx;
      int32_t cell_z = player_cell_z + dz;

      Vec2i fortress = get_fortress_in_cell(cell_x, cell_z, seed);

      float dist_sq = distance_squared(player_x, player_z, fortress.x, fortress.z);
      if (dist_sq <= radius_sq) {
        out_fortresses[count++] = fortress;
      }
    }
  }

  return count;
}

// Check if a position is inside a fortress bounding box
// Fortresses have a variable size, but the main structure is roughly:
// - Main hallway: ~56x56 blocks centered on generation point
// - Extensions can add another ~16-32 blocks in any direction
// This function uses conservative bounds that cover most fortress layouts
bool is_inside_fortress(float pos_x, float pos_z, const Vec2i& fortress_pos) {
  // Fortress generation centers the structure around the generation point
  // Conservative bounds: 64 blocks in each direction from center
  constexpr float FORTRESS_HALF_SIZE = 64.0f;

  float dx = std::abs(pos_x - static_cast<float>(fortress_pos.x));
  float dz = std::abs(pos_z - static_cast<float>(fortress_pos.z));

  return dx <= FORTRESS_HALF_SIZE && dz <= FORTRESS_HALF_SIZE;
}

// Get direction vector from player to fortress (normalized)
// Returns angle in radians, with 0 = +Z (south), PI/2 = +X (west)
float get_fortress_direction(float player_x, float player_z, const Vec2i& fortress_pos) {
  float dx = static_cast<float>(fortress_pos.x) - player_x;
  float dz = static_cast<float>(fortress_pos.z) - player_z;
  return std::atan2(dx, dz);
}

// Get distance to fortress
float get_fortress_distance(float player_x, float player_z, const Vec2i& fortress_pos) {
  float dx = static_cast<float>(fortress_pos.x) - player_x;
  float dz = static_cast<float>(fortress_pos.z) - player_z;
  return std::sqrt(dx * dx + dz * dz);
}

// Convert Overworld coordinates to Nether coordinates
// Overworld 1:8 Nether ratio
inline Vec2i overworld_to_nether(int32_t ow_x, int32_t ow_z) {
  return Vec2i(ow_x >> 3, ow_z >> 3);
}

// Convert Nether coordinates to Overworld coordinates
inline Vec2i nether_to_overworld(int32_t nether_x, int32_t nether_z) {
  return Vec2i(nether_x << 3, nether_z << 3);
}

} // namespace mc189
