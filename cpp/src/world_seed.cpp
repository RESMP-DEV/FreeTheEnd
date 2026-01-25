// world_seed.cpp - Deterministic world generation from seed
// Single seed controls all RNG for reproducible RL research
//
// Guarantees:
// - Same seed = same world layout (terrain, structures)
// - Same seed + actions = same outcome (deterministic simulation)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace mc189 {

/**
 * SplitMix64 - Fast, high-quality PRNG for seeding other generators.
 * Used to derive sub-seeds from the master seed.
 */
class SplitMix64 {
public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}

  uint64_t next() {
    uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

private:
  uint64_t state_;
};

/**
 * Xoshiro256** - Fast, high-quality PRNG with 256-bit state.
 * Main generator for simulation randomness.
 */
class Xoshiro256 {
public:
  explicit Xoshiro256(uint64_t seed) { seed_from(seed); }

  void seed_from(uint64_t seed) {
    SplitMix64 sm(seed);
    for (int i = 0; i < 4; ++i) {
      state_[i] = sm.next();
    }
  }

  uint64_t next() {
    const uint64_t result = rotl(state_[1] * 5, 7) * 9;
    const uint64_t t = state_[1] << 17;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= t;
    state_[3] = rotl(state_[3], 45);
    return result;
  }

  // Uniform double in [0, 1)
  double next_double() {
    return (next() >> 11) * 0x1.0p-53;
  }

  // Uniform int in [0, bound)
  uint64_t next_int(uint64_t bound) {
    // Debiased modulo (Lemire's method)
    uint64_t x = next();
    __uint128_t m = static_cast<__uint128_t>(x) * bound;
    uint64_t l = static_cast<uint64_t>(m);
    if (l < bound) {
      uint64_t t = -bound % bound;
      while (l < t) {
        x = next();
        m = static_cast<__uint128_t>(x) * bound;
        l = static_cast<uint64_t>(m);
      }
    }
    return m >> 64;
  }

  // Uniform float in [0, 1)
  float next_float() {
    return static_cast<float>(next() >> 40) * 0x1.0p-24f;
  }

  // Gaussian with mean 0, std 1
  float next_gaussian() {
    // Box-Muller transform
    float u1 = next_float();
    float u2 = next_float();
    while (u1 < 1e-10f) u1 = next_float();
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
  }

  // Save/restore state for deterministic replay
  std::array<uint64_t, 4> get_state() const { return state_; }
  void set_state(const std::array<uint64_t, 4>& s) { state_ = s; }

private:
  static uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }

  std::array<uint64_t, 4> state_;
};

/**
 * PerlinNoise - Seeded Perlin noise for terrain generation.
 * Deterministic given seed.
 */
class PerlinNoise {
public:
  explicit PerlinNoise(uint64_t seed) {
    // Initialize permutation table from seed
    Xoshiro256 rng(seed);
    for (int i = 0; i < 256; ++i) {
      perm_[i] = i;
    }
    // Fisher-Yates shuffle
    for (int i = 255; i > 0; --i) {
      int j = rng.next_int(i + 1);
      std::swap(perm_[i], perm_[j]);
    }
    // Duplicate for overflow
    for (int i = 0; i < 256; ++i) {
      perm_[256 + i] = perm_[i];
    }
  }

  float noise2d(float x, float z) const {
    int xi = static_cast<int>(std::floor(x)) & 255;
    int zi = static_cast<int>(std::floor(z)) & 255;
    float xf = x - std::floor(x);
    float zf = z - std::floor(z);

    float u = fade(xf);
    float v = fade(zf);

    int aa = perm_[perm_[xi] + zi];
    int ab = perm_[perm_[xi] + zi + 1];
    int ba = perm_[perm_[xi + 1] + zi];
    int bb = perm_[perm_[xi + 1] + zi + 1];

    float x1 = lerp(grad2d(aa, xf, zf), grad2d(ba, xf - 1, zf), u);
    float x2 = lerp(grad2d(ab, xf, zf - 1), grad2d(bb, xf - 1, zf - 1), u);
    return lerp(x1, x2, v);
  }

  float noise3d(float x, float y, float z) const {
    int xi = static_cast<int>(std::floor(x)) & 255;
    int yi = static_cast<int>(std::floor(y)) & 255;
    int zi = static_cast<int>(std::floor(z)) & 255;
    float xf = x - std::floor(x);
    float yf = y - std::floor(y);
    float zf = z - std::floor(z);

    float u = fade(xf);
    float v = fade(yf);
    float w = fade(zf);

    int aaa = perm_[perm_[perm_[xi] + yi] + zi];
    int aba = perm_[perm_[perm_[xi] + yi + 1] + zi];
    int aab = perm_[perm_[perm_[xi] + yi] + zi + 1];
    int abb = perm_[perm_[perm_[xi] + yi + 1] + zi + 1];
    int baa = perm_[perm_[perm_[xi + 1] + yi] + zi];
    int bba = perm_[perm_[perm_[xi + 1] + yi + 1] + zi];
    int bab = perm_[perm_[perm_[xi + 1] + yi] + zi + 1];
    int bbb = perm_[perm_[perm_[xi + 1] + yi + 1] + zi + 1];

    float x1 = lerp(grad3d(aaa, xf, yf, zf), grad3d(baa, xf - 1, yf, zf), u);
    float x2 = lerp(grad3d(aba, xf, yf - 1, zf), grad3d(bba, xf - 1, yf - 1, zf), u);
    float y1 = lerp(x1, x2, v);

    x1 = lerp(grad3d(aab, xf, yf, zf - 1), grad3d(bab, xf - 1, yf, zf - 1), u);
    x2 = lerp(grad3d(abb, xf, yf - 1, zf - 1), grad3d(bbb, xf - 1, yf - 1, zf - 1), u);
    float y2 = lerp(x1, x2, v);

    return lerp(y1, y2, w);
  }

  // Octave (fractal) noise for natural terrain
  float octave2d(float x, float z, int octaves, float persistence = 0.5f) const {
    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float max_value = 0.0f;

    for (int i = 0; i < octaves; ++i) {
      total += noise2d(x * frequency, z * frequency) * amplitude;
      max_value += amplitude;
      amplitude *= persistence;
      frequency *= 2.0f;
    }
    return total / max_value;
  }

private:
  static float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
  static float lerp(float a, float b, float t) { return a + t * (b - a); }

  static float grad2d(int hash, float x, float z) {
    switch (hash & 3) {
    case 0: return x + z;
    case 1: return -x + z;
    case 2: return x - z;
    case 3: return -x - z;
    }
    return 0.0f;
  }

  static float grad3d(int hash, float x, float y, float z) {
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
  }

  std::array<int, 512> perm_;
};

/**
 * WorldSeed - Master seed manager for deterministic world generation.
 *
 * Derives separate sub-seeds for different systems to ensure
 * changes in one system don't affect randomness in others.
 */
class WorldSeed {
public:
  // Sub-seed indices
  enum SubSeedType {
    TERRAIN_OVERWORLD = 0,
    TERRAIN_NETHER = 1,
    TERRAIN_END = 2,
    STRUCTURE_FORTRESS = 3,
    STRUCTURE_STRONGHOLD = 4,
    STRUCTURE_VILLAGE = 5,
    MOB_SPAWNS = 6,
    ITEM_DROPS = 7,
    WEATHER = 8,
    DRAGON_AI = 9,
    NUM_SUBSEEDS = 16
  };

  explicit WorldSeed(uint64_t master_seed = 0)
      : master_seed_(master_seed == 0 ? generate_random_seed() : master_seed) {
    derive_subseeds();
  }

  void reset(uint64_t seed = 0) {
    master_seed_ = (seed == 0) ? generate_random_seed() : seed;
    derive_subseeds();
    tick_rng_.seed_from(sub_seeds_[MOB_SPAWNS]);
  }

  uint64_t get_seed() const { return master_seed_; }

  // Get deterministic RNG for a specific system
  Xoshiro256 get_rng(SubSeedType type) const {
    return Xoshiro256(sub_seeds_[type]);
  }

  // Get terrain noise generator
  PerlinNoise get_terrain_noise(SubSeedType terrain_type) const {
    return PerlinNoise(sub_seeds_[terrain_type]);
  }

  // Per-tick RNG for mob spawns and item drops
  // Advances deterministically each tick
  Xoshiro256& tick_rng() { return tick_rng_; }

  // Coordinate-based seeding for chunk generation
  // Same chunk at same coords always generates the same way
  uint64_t chunk_seed(int32_t chunk_x, int32_t chunk_z, SubSeedType type) const {
    // Mix coordinates with sub-seed
    uint64_t base = sub_seeds_[type];
    uint64_t cx = static_cast<uint64_t>(chunk_x);
    uint64_t cz = static_cast<uint64_t>(chunk_z);
    // Murmur-style mixing
    uint64_t seed = base ^ (cx * 0xcc9e2d51ULL);
    seed = (seed << 15) | (seed >> 49);
    seed ^= cz * 0x1b873593ULL;
    seed = (seed << 13) | (seed >> 51);
    seed = seed * 5 + 0xe6546b64ULL;
    return seed;
  }

  // Structure placement (deterministic per-region)
  bool has_structure_at(int32_t chunk_x, int32_t chunk_z, SubSeedType struct_type,
                        float probability) const {
    uint64_t seed = chunk_seed(chunk_x, chunk_z, struct_type);
    Xoshiro256 rng(seed);
    return rng.next_float() < probability;
  }

  // Stronghold ring generation (3 rings with specific distances)
  struct StrongholdLocation {
    int32_t chunk_x;
    int32_t chunk_z;
  };

  std::vector<StrongholdLocation> generate_strongholds() const {
    std::vector<StrongholdLocation> locations;
    Xoshiro256 rng(sub_seeds_[STRUCTURE_STRONGHOLD]);

    // Ring 1: 3 strongholds, 1280-2816 blocks from origin
    // Ring 2: 6 strongholds, 4352-5888 blocks
    // Ring 3: 10 strongholds, 7424-8960 blocks
    constexpr int counts[] = {3, 6, 10};
    constexpr int min_dist[] = {1280, 4352, 7424};
    constexpr int max_dist[] = {2816, 5888, 8960};

    for (int ring = 0; ring < 3; ++ring) {
      float angle_step = 2.0f * 3.14159265f / counts[ring];
      float start_angle = rng.next_float() * 2.0f * 3.14159265f;

      for (int i = 0; i < counts[ring]; ++i) {
        float angle = start_angle + i * angle_step + (rng.next_float() - 0.5f) * 0.5f;
        float dist = min_dist[ring] + rng.next_float() * (max_dist[ring] - min_dist[ring]);
        locations.push_back({
            static_cast<int32_t>(std::cos(angle) * dist) >> 4,
            static_cast<int32_t>(std::sin(angle) * dist) >> 4
        });
      }
    }
    return locations;
  }

  // Nether fortress generation (grid-based with jitter)
  bool has_fortress_at(int32_t chunk_x, int32_t chunk_z) const {
    // Fortresses spawn in regions of 16x16 chunks
    int32_t region_x = chunk_x >> 4;
    int32_t region_z = chunk_z >> 4;

    uint64_t seed = chunk_seed(region_x, region_z, STRUCTURE_FORTRESS);
    Xoshiro256 rng(seed);

    // ~1/3 chance per region
    if (rng.next_float() > 0.33f) return false;

    // Jitter within region
    int32_t local_x = (region_x << 4) + rng.next_int(16);
    int32_t local_z = (region_z << 4) + rng.next_int(16);

    return chunk_x == local_x && chunk_z == local_z;
  }

  // End pillar generation (always 10 pillars in fixed ring)
  struct PillarInfo {
    float x, y, z;
    float height;
    float radius;
    bool has_cage;
  };

  std::array<PillarInfo, 10> generate_end_pillars() const {
    std::array<PillarInfo, 10> pillars;
    Xoshiro256 rng(sub_seeds_[TERRAIN_END]);

    for (int i = 0; i < 10; ++i) {
      float angle = i * (2.0f * 3.14159265f / 10.0f);
      float dist = 40.0f + (i % 3) * 15.0f;

      pillars[i].x = std::cos(angle) * dist;
      pillars[i].z = std::sin(angle) * dist;
      pillars[i].height = 76.0f + rng.next_int(28);
      pillars[i].y = pillars[i].height;
      pillars[i].radius = 2.0f + rng.next_int(2);
      pillars[i].has_cage = rng.next_float() < 0.4f;
    }
    return pillars;
  }

  // Deterministic mob spawn roll (called each tick)
  struct SpawnRoll {
    float x, y, z;
    uint16_t mob_type;
    uint8_t count;
  };

  SpawnRoll roll_mob_spawn(float player_x, float player_z, int32_t tick) {
    // Advance RNG to this tick
    SpawnRoll roll{};
    uint64_t tick_seed = sub_seeds_[MOB_SPAWNS] ^ static_cast<uint64_t>(tick);
    Xoshiro256 rng(tick_seed);

    // Random offset from player (24-128 blocks)
    float angle = rng.next_float() * 2.0f * 3.14159265f;
    float dist = 24.0f + rng.next_float() * 104.0f;
    roll.x = player_x + std::cos(angle) * dist;
    roll.z = player_z + std::sin(angle) * dist;
    roll.y = 64.0f; // Ground level placeholder

    // Mob type distribution (End dimension)
    float r = rng.next_float();
    if (r < 0.95f) {
      roll.mob_type = 58; // Enderman
      roll.count = 1 + rng.next_int(4);
    } else {
      roll.mob_type = 0; // No spawn
      roll.count = 0;
    }

    return roll;
  }

  // Deterministic item drop (seeded by entity ID and kill tick)
  struct DropRoll {
    uint16_t item_id;
    uint8_t count;
    bool is_rare;
  };

  DropRoll roll_item_drop(uint32_t entity_id, int32_t tick, uint16_t mob_type) const {
    uint64_t drop_seed = sub_seeds_[ITEM_DROPS] ^ (static_cast<uint64_t>(entity_id) << 32) ^
                         static_cast<uint64_t>(tick);
    Xoshiro256 rng(drop_seed);

    DropRoll drop{};
    drop.is_rare = rng.next_float() < 0.025f; // 2.5% rare drop

    // Mob-specific loot tables
    switch (mob_type) {
    case 58: // Enderman
      if (drop.is_rare) {
        drop.item_id = 368; // Ender pearl
        drop.count = 1;
      } else if (rng.next_float() < 0.5f) {
        drop.item_id = 368;
        drop.count = 1;
      } else {
        drop.item_id = 0;
        drop.count = 0;
      }
      break;
    default:
      drop.item_id = 0;
      drop.count = 0;
    }
    return drop;
  }

  // Dragon AI seeding (ensures dragon behavior is deterministic)
  uint32_t dragon_decision_seed(int32_t tick, uint32_t phase) const {
    return static_cast<uint32_t>(sub_seeds_[DRAGON_AI] ^
                                  (static_cast<uint64_t>(tick) << 8) ^ phase);
  }

private:
  static uint64_t generate_random_seed() {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) | rd();
  }

  void derive_subseeds() {
    SplitMix64 sm(master_seed_);
    for (int i = 0; i < NUM_SUBSEEDS; ++i) {
      sub_seeds_[i] = sm.next();
    }
    tick_rng_.seed_from(sub_seeds_[MOB_SPAWNS]);
  }

  uint64_t master_seed_;
  std::array<uint64_t, NUM_SUBSEEDS> sub_seeds_;
  Xoshiro256 tick_rng_{0}; // Updated each tick for mob spawns
};

/**
 * DeterministicTerrainGenerator - Generates terrain chunks deterministically.
 */
class DeterministicTerrainGenerator {
public:
  explicit DeterministicTerrainGenerator(const WorldSeed& seed)
      : seed_(seed),
        overworld_noise_(seed.get_terrain_noise(WorldSeed::TERRAIN_OVERWORLD)),
        nether_noise_(seed.get_terrain_noise(WorldSeed::TERRAIN_NETHER)),
        end_noise_(seed.get_terrain_noise(WorldSeed::TERRAIN_END)) {}

  // Generate height at world coordinate (deterministic)
  float overworld_height(int32_t x, int32_t z) const {
    float scale = 0.02f;
    float base = overworld_noise_.octave2d(x * scale, z * scale, 4, 0.5f);
    return 64.0f + base * 32.0f;
  }

  float nether_ceiling(int32_t x, int32_t z) const {
    float scale = 0.04f;
    float noise = nether_noise_.octave2d(x * scale, z * scale, 3, 0.6f);
    return 100.0f + noise * 28.0f;
  }

  float nether_floor(int32_t x, int32_t z) const {
    float scale = 0.04f;
    float noise = nether_noise_.octave2d(x * scale + 1000.0f, z * scale, 3, 0.6f);
    return 32.0f + noise * 16.0f;
  }

  // End island density (for end city generation)
  float end_island_density(int32_t x, int32_t z) const {
    // Main island (central 100 block radius)
    float dist = std::sqrt(static_cast<float>(x * x + z * z));
    if (dist < 100.0f) {
      return 1.0f - dist / 100.0f;
    }
    // Outer islands (start at ~1000 blocks out)
    if (dist < 1000.0f) return -1.0f;

    float scale = 0.01f;
    return end_noise_.noise2d(x * scale, z * scale);
  }

private:
  const WorldSeed& seed_;
  PerlinNoise overworld_noise_;
  PerlinNoise nether_noise_;
  PerlinNoise end_noise_;
};

} // namespace mc189
