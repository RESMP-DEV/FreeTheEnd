// world_seed.h - Deterministic world generation from seed
// Single seed controls all RNG for reproducible RL research
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace mc189 {

// Forward declarations
class Xoshiro256;
class PerlinNoise;

/**
 * WorldSeed - Master seed manager for deterministic world generation.
 *
 * Guarantees:
 * - Same seed = same world layout (terrain, structures, pillars)
 * - Same seed + same actions = same simulation outcome
 *
 * Usage:
 *   WorldSeed seed(12345);  // Or seed(0) for random
 *   auto pillars = seed.generate_end_pillars();
 *   auto strongholds = seed.generate_strongholds();
 *   auto rng = seed.get_rng(WorldSeed::MOB_SPAWNS);
 */
class WorldSeed {
public:
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

  /**
   * Create a new WorldSeed.
   * @param master_seed The seed value. 0 = generate random seed.
   */
  explicit WorldSeed(uint64_t master_seed = 0);

  /**
   * Reset with a new seed.
   * @param seed The new seed. 0 = generate random seed.
   */
  void reset(uint64_t seed = 0);

  /**
   * Get the current master seed.
   */
  uint64_t get_seed() const;

  /**
   * Get a seeded RNG for a specific system.
   * Each system has independent randomness derived from the master seed.
   */
  Xoshiro256 get_rng(SubSeedType type) const;

  /**
   * Get terrain noise generator for world generation.
   */
  PerlinNoise get_terrain_noise(SubSeedType terrain_type) const;

  /**
   * Get per-tick RNG reference for mob spawns/item drops.
   * Advances deterministically each simulation tick.
   */
  Xoshiro256& tick_rng();

  /**
   * Compute deterministic chunk seed from coordinates.
   * Same chunk always generates identically.
   */
  uint64_t chunk_seed(int32_t chunk_x, int32_t chunk_z, SubSeedType type) const;

  /**
   * Check if a structure should generate at this chunk.
   * @param probability Spawn probability (0.0 to 1.0)
   */
  bool has_structure_at(int32_t chunk_x, int32_t chunk_z, SubSeedType struct_type,
                        float probability) const;

  // Stronghold locations
  struct StrongholdLocation {
    int32_t chunk_x;
    int32_t chunk_z;
  };
  std::vector<StrongholdLocation> generate_strongholds() const;

  // Check for Nether fortress at chunk
  bool has_fortress_at(int32_t chunk_x, int32_t chunk_z) const;

  // End pillar configuration
  struct PillarInfo {
    float x, y, z;
    float height;
    float radius;
    bool has_cage;
  };
  std::array<PillarInfo, 10> generate_end_pillars() const;

  // Deterministic mob spawn roll
  struct SpawnRoll {
    float x, y, z;
    uint16_t mob_type;
    uint8_t count;
  };
  SpawnRoll roll_mob_spawn(float player_x, float player_z, int32_t tick);

  // Deterministic item drop roll
  struct DropRoll {
    uint16_t item_id;
    uint8_t count;
    bool is_rare;
  };
  DropRoll roll_item_drop(uint32_t entity_id, int32_t tick, uint16_t mob_type) const;

  // Dragon AI decision seed for deterministic behavior
  uint32_t dragon_decision_seed(int32_t tick, uint32_t phase) const;

private:
  class Impl;
  uint64_t master_seed_;
  std::array<uint64_t, NUM_SUBSEEDS> sub_seeds_;
  // tick_rng_ stored internally
};

/**
 * Xoshiro256** PRNG - Fast, high-quality generator.
 */
class Xoshiro256 {
public:
  explicit Xoshiro256(uint64_t seed);
  void seed_from(uint64_t seed);

  uint64_t next();
  double next_double();      // [0, 1)
  float next_float();        // [0, 1)
  uint64_t next_int(uint64_t bound); // [0, bound)
  float next_gaussian();     // N(0, 1)

  std::array<uint64_t, 4> get_state() const;
  void set_state(const std::array<uint64_t, 4>& state);

private:
  std::array<uint64_t, 4> state_;
};

/**
 * PerlinNoise - Seeded noise for terrain generation.
 */
class PerlinNoise {
public:
  explicit PerlinNoise(uint64_t seed);

  float noise2d(float x, float z) const;
  float noise3d(float x, float y, float z) const;
  float octave2d(float x, float z, int octaves, float persistence = 0.5f) const;

private:
  std::array<int, 512> perm_;
};

/**
 * DeterministicTerrainGenerator - Chunk-based terrain generation.
 */
class DeterministicTerrainGenerator {
public:
  explicit DeterministicTerrainGenerator(const WorldSeed& seed);

  float overworld_height(int32_t x, int32_t z) const;
  float nether_ceiling(int32_t x, int32_t z) const;
  float nether_floor(int32_t x, int32_t z) const;
  float end_island_density(int32_t x, int32_t z) const;

private:
  const WorldSeed& seed_;
  PerlinNoise overworld_noise_;
  PerlinNoise nether_noise_;
  PerlinNoise end_noise_;
};

} // namespace mc189
