// world_seed_impl.cpp - Implementation of world_seed.h
// Deterministic world generation from seed

#include "mc189/world_seed.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace mc189 {

//==============================================================================
// SplitMix64 - Fast seeding helper
//==============================================================================
namespace {
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

uint64_t generate_random_seed() {
  std::random_device rd;
  return (static_cast<uint64_t>(rd()) << 32) | rd();
}
} // namespace

//==============================================================================
// Xoshiro256 implementation
//==============================================================================
Xoshiro256::Xoshiro256(uint64_t seed) { seed_from(seed); }

void Xoshiro256::seed_from(uint64_t seed) {
  SplitMix64 sm(seed);
  for (int i = 0; i < 4; ++i) {
    state_[i] = sm.next();
  }
}

uint64_t Xoshiro256::next() {
  const uint64_t result = ((state_[1] * 5) << 7 | (state_[1] * 5) >> 57) * 9;
  const uint64_t t = state_[1] << 17;

  state_[2] ^= state_[0];
  state_[3] ^= state_[1];
  state_[1] ^= state_[2];
  state_[0] ^= state_[3];
  state_[2] ^= t;
  state_[3] = (state_[3] << 45) | (state_[3] >> 19);

  return result;
}

double Xoshiro256::next_double() {
  return static_cast<double>(next() >> 11) * 0x1.0p-53;
}

float Xoshiro256::next_float() {
  return static_cast<float>(next() >> 40) * 0x1.0p-24f;
}

uint64_t Xoshiro256::next_int(uint64_t bound) {
  if (bound == 0)
    return 0;
  uint64_t threshold = -bound % bound;
  for (;;) {
    uint64_t r = next();
    if (r >= threshold)
      return r % bound;
  }
}

float Xoshiro256::next_gaussian() {
  // Box-Muller transform
  float u1 = next_float();
  float u2 = next_float();
  if (u1 < 1e-10f)
    u1 = 1e-10f;
  return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
}

std::array<uint64_t, 4> Xoshiro256::get_state() const { return state_; }

void Xoshiro256::set_state(const std::array<uint64_t, 4> &state) {
  state_ = state;
}

//==============================================================================
// PerlinNoise implementation
//==============================================================================
PerlinNoise::PerlinNoise(uint64_t seed) {
  Xoshiro256 rng(seed);

  // Initialize permutation table
  for (int i = 0; i < 256; ++i) {
    perm_[i] = i;
  }

  // Shuffle using Fisher-Yates
  for (int i = 255; i > 0; --i) {
    int j = rng.next_int(i + 1);
    std::swap(perm_[i], perm_[j]);
  }

  // Duplicate for wraparound
  for (int i = 0; i < 256; ++i) {
    perm_[256 + i] = perm_[i];
  }
}

namespace {
inline float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }

inline float lerp(float t, float a, float b) { return a + t * (b - a); }

inline float grad2d(int hash, float x, float z) {
  int h = hash & 3;
  float u = h < 2 ? x : z;
  float v = h < 2 ? z : x;
  return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

inline float grad3d(int hash, float x, float y, float z) {
  int h = hash & 15;
  float u = h < 8 ? x : y;
  float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
  return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}
} // namespace

float PerlinNoise::noise2d(float x, float z) const {
  int X = static_cast<int>(std::floor(x)) & 255;
  int Z = static_cast<int>(std::floor(z)) & 255;

  x -= std::floor(x);
  z -= std::floor(z);

  float u = fade(x);
  float v = fade(z);

  int A = perm_[X] + Z;
  int B = perm_[X + 1] + Z;

  return lerp(v, lerp(u, grad2d(perm_[A], x, z), grad2d(perm_[B], x - 1, z)),
              lerp(u, grad2d(perm_[A + 1], x, z - 1),
                   grad2d(perm_[B + 1], x - 1, z - 1)));
}

float PerlinNoise::noise3d(float x, float y, float z) const {
  int X = static_cast<int>(std::floor(x)) & 255;
  int Y = static_cast<int>(std::floor(y)) & 255;
  int Z = static_cast<int>(std::floor(z)) & 255;

  x -= std::floor(x);
  y -= std::floor(y);
  z -= std::floor(z);

  float u = fade(x);
  float v = fade(y);
  float w = fade(z);

  int A = perm_[X] + Y;
  int AA = perm_[A] + Z;
  int AB = perm_[A + 1] + Z;
  int B = perm_[X + 1] + Y;
  int BA = perm_[B] + Z;
  int BB = perm_[B + 1] + Z;

  return lerp(
      w,
      lerp(v,
           lerp(u, grad3d(perm_[AA], x, y, z), grad3d(perm_[BA], x - 1, y, z)),
           lerp(u, grad3d(perm_[AB], x, y - 1, z),
                grad3d(perm_[BB], x - 1, y - 1, z))),
      lerp(v,
           lerp(u, grad3d(perm_[AA + 1], x, y, z - 1),
                grad3d(perm_[BA + 1], x - 1, y, z - 1)),
           lerp(u, grad3d(perm_[AB + 1], x, y - 1, z - 1),
                grad3d(perm_[BB + 1], x - 1, y - 1, z - 1))));
}

float PerlinNoise::octave2d(float x, float z, int octaves,
                            float persistence) const {
  float total = 0;
  float frequency = 1;
  float amplitude = 1;
  float maxValue = 0;

  for (int i = 0; i < octaves; ++i) {
    total += noise2d(x * frequency, z * frequency) * amplitude;
    maxValue += amplitude;
    amplitude *= persistence;
    frequency *= 2;
  }

  return total / maxValue;
}

//==============================================================================
// WorldSeed implementation
//==============================================================================
WorldSeed::WorldSeed(uint64_t master_seed)
    : master_seed_(master_seed == 0 ? generate_random_seed() : master_seed) {
  // Derive sub-seeds from master seed
  SplitMix64 sm(master_seed_);
  for (int i = 0; i < NUM_SUBSEEDS; ++i) {
    sub_seeds_[i] = sm.next();
  }
}

void WorldSeed::reset(uint64_t seed) {
  master_seed_ = (seed == 0) ? generate_random_seed() : seed;
  SplitMix64 sm(master_seed_);
  for (int i = 0; i < NUM_SUBSEEDS; ++i) {
    sub_seeds_[i] = sm.next();
  }
}

uint64_t WorldSeed::get_seed() const { return master_seed_; }

Xoshiro256 WorldSeed::get_rng(SubSeedType type) const {
  return Xoshiro256(sub_seeds_[type]);
}

PerlinNoise WorldSeed::get_terrain_noise(SubSeedType terrain_type) const {
  return PerlinNoise(sub_seeds_[terrain_type]);
}

Xoshiro256 &WorldSeed::tick_rng() {
  static thread_local Xoshiro256 rng(0);
  static thread_local uint64_t last_seed = 0;
  if (last_seed != sub_seeds_[MOB_SPAWNS]) {
    last_seed = sub_seeds_[MOB_SPAWNS];
    rng.seed_from(last_seed);
  }
  return rng;
}

uint64_t WorldSeed::chunk_seed(int32_t chunk_x, int32_t chunk_z,
                               SubSeedType type) const {
  uint64_t base = sub_seeds_[type];
  uint64_t cx = static_cast<uint64_t>(static_cast<uint32_t>(chunk_x));
  uint64_t cz = static_cast<uint64_t>(static_cast<uint32_t>(chunk_z));
  // Murmur-style mixing
  uint64_t seed = base ^ (cx * 0xcc9e2d51ULL);
  seed = (seed << 15) | (seed >> 49);
  seed ^= cz * 0x1b873593ULL;
  seed = (seed << 13) | (seed >> 51);
  seed = seed * 5 + 0xe6546b64ULL;
  return seed;
}

bool WorldSeed::has_structure_at(int32_t chunk_x, int32_t chunk_z,
                                 SubSeedType struct_type,
                                 float probability) const {
  uint64_t seed = chunk_seed(chunk_x, chunk_z, struct_type);
  Xoshiro256 rng(seed);
  return rng.next_float() < probability;
}

std::vector<WorldSeed::StrongholdLocation>
WorldSeed::generate_strongholds() const {
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
      float angle = start_angle + i * angle_step;
      angle += (rng.next_float() - 0.5f) * angle_step * 0.5f;

      int dist =
          min_dist[ring] + static_cast<int>(rng.next_float() *
                                            (max_dist[ring] - min_dist[ring]));

      StrongholdLocation loc;
      loc.chunk_x = static_cast<int32_t>(std::cos(angle) * dist) / 16;
      loc.chunk_z = static_cast<int32_t>(std::sin(angle) * dist) / 16;
      locations.push_back(loc);
    }
  }

  return locations;
}

bool WorldSeed::has_fortress_at(int32_t chunk_x, int32_t chunk_z) const {
  // Nether fortresses use 16x16 chunk regions
  int region_x = chunk_x >> 4;
  int region_z = chunk_z >> 4;

  uint64_t seed = chunk_seed(region_x, region_z, STRUCTURE_FORTRESS);
  Xoshiro256 rng(seed);

  // ~1 fortress per region, placed at specific chunk
  int fortress_chunk_x = (region_x << 4) + static_cast<int>(rng.next_int(16));
  int fortress_chunk_z = (region_z << 4) + static_cast<int>(rng.next_int(16));

  return chunk_x == fortress_chunk_x && chunk_z == fortress_chunk_z;
}

std::array<WorldSeed::PillarInfo, 10> WorldSeed::generate_end_pillars() const {
  std::array<PillarInfo, 10> pillars;
  Xoshiro256 rng(sub_seeds_[TERRAIN_END]);

  // Pillars arranged in a circle around the fountain
  constexpr float PILLAR_RADIUS = 43.0f;
  constexpr float BASE_HEIGHT = 76.0f;

  for (int i = 0; i < 10; ++i) {
    float angle = (2.0f * 3.14159265f * i) / 10.0f;

    pillars[i].x = std::cos(angle) * PILLAR_RADIUS;
    pillars[i].z = std::sin(angle) * PILLAR_RADIUS;
    pillars[i].y = 0.0f; // Ground level

    // Heights vary: 76, 79, 82, 85, 88, 91, 94, 97, 100, 103
    pillars[i].height = BASE_HEIGHT + (i * 3.0f);

    // Radius alternates: smaller and larger pillars
    pillars[i].radius = (i % 2 == 0) ? 3.0f : 4.0f;

    // Cages on the two tallest pillars (indices 8 and 9)
    pillars[i].has_cage = (i >= 8);
  }

  return pillars;
}

WorldSeed::SpawnRoll WorldSeed::roll_mob_spawn(float player_x, float player_z,
                                               int32_t tick) {
  SpawnRoll roll{};

  uint64_t spawn_seed =
      chunk_seed(static_cast<int32_t>(player_x) >> 4,
                 static_cast<int32_t>(player_z) >> 4, MOB_SPAWNS) ^
      static_cast<uint64_t>(tick);

  Xoshiro256 rng(spawn_seed);

  // Random offset from player
  float angle = rng.next_float() * 2.0f * 3.14159265f;
  float dist = 24.0f + rng.next_float() * 64.0f;

  roll.x = player_x + std::cos(angle) * dist;
  roll.z = player_z + std::sin(angle) * dist;
  roll.y = 64.0f; // Ground level estimate

  // Mob type selection (simplified)
  roll.mob_type = static_cast<uint16_t>(rng.next_int(10)); // 0-9 mob types
  roll.count = static_cast<uint8_t>(1 + rng.next_int(4));  // 1-4 mobs

  return roll;
}

WorldSeed::DropRoll WorldSeed::roll_item_drop(uint32_t entity_id, int32_t tick,
                                              uint16_t mob_type) const {
  DropRoll roll{};

  uint64_t drop_seed = sub_seeds_[ITEM_DROPS] ^
                       (static_cast<uint64_t>(entity_id) << 32) ^
                       static_cast<uint64_t>(tick);

  Xoshiro256 rng(drop_seed);

  // Base drop depends on mob type
  roll.item_id = static_cast<uint16_t>(mob_type * 10 + rng.next_int(5));
  roll.count = static_cast<uint8_t>(1 + rng.next_int(3));
  roll.is_rare = rng.next_float() < 0.05f; // 5% rare drop chance

  return roll;
}

uint32_t WorldSeed::dragon_decision_seed(int32_t tick, uint32_t phase) const {
  uint64_t seed = sub_seeds_[DRAGON_AI] ^ (static_cast<uint64_t>(tick) << 16) ^
                  static_cast<uint64_t>(phase);
  return static_cast<uint32_t>(seed ^ (seed >> 32));
}

//==============================================================================
// DeterministicTerrainGenerator implementation
//==============================================================================
DeterministicTerrainGenerator::DeterministicTerrainGenerator(
    const WorldSeed &seed)
    : seed_(seed),
      overworld_noise_(seed.get_terrain_noise(WorldSeed::TERRAIN_OVERWORLD)),
      nether_noise_(seed.get_terrain_noise(WorldSeed::TERRAIN_NETHER)),
      end_noise_(seed.get_terrain_noise(WorldSeed::TERRAIN_END)) {}

float DeterministicTerrainGenerator::overworld_height(int32_t x,
                                                      int32_t z) const {
  float base = overworld_noise_.octave2d(x * 0.01f, z * 0.01f, 4, 0.5f);
  return 64.0f + base * 32.0f; // Heights from 32 to 96
}

float DeterministicTerrainGenerator::nether_ceiling(int32_t x,
                                                    int32_t z) const {
  float noise = nether_noise_.octave2d(x * 0.02f, z * 0.02f, 3, 0.6f);
  return 120.0f + noise * 8.0f; // Ceiling 112-128
}

float DeterministicTerrainGenerator::nether_floor(int32_t x, int32_t z) const {
  float noise =
      nether_noise_.octave2d(x * 0.02f + 1000, z * 0.02f + 1000, 3, 0.6f);
  return 32.0f + noise * 16.0f; // Floor 16-48
}

float DeterministicTerrainGenerator::end_island_density(int32_t x,
                                                        int32_t z) const {
  // Distance from center affects density
  float dist = std::sqrt(static_cast<float>(x * x + z * z));

  if (dist < 100) {
    return 1.0f; // Main island always solid
  }

  float noise = end_noise_.octave2d(x * 0.05f, z * 0.05f, 4, 0.5f);
  float dist_factor = std::max(0.0f, 1.0f - dist / 1000.0f);

  return noise * dist_factor;
}

} // namespace mc189
