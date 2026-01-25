// spawn_selection.h - MC 1.8.9 spawn point selection API
// Deterministic spawn selection matching vanilla behavior

#pragma once

#include <cstdint>

namespace mc189 {
namespace spawn {

// Result of spawn point selection
struct SpawnResult {
    int32_t x;           // World X coordinate
    int32_t y;           // World Y coordinate (standing position)
    int32_t z;           // World Z coordinate
    bool valid;          // True if a valid spawn was found
    uint32_t biome;      // Biome ID at spawn location
};

// World state after spawn selection
struct WorldSpawnState {
    SpawnResult spawn;   // Selected spawn point
    int64_t worldTime;   // Initial world time (0 = dawn)
    int64_t seed;        // World seed used for generation
};

// Select a spawn point matching MC 1.8.9 behavior.
// Searches in a 256x256 area centered on world origin.
// Prefers plains/forest biomes on solid ground.
// Avoids water, lava, and cliff edges.
// Deterministic for a given seed.
SpawnResult selectSpawnPoint(int64_t worldSeed);

// Get initial world time (always 0 = dawn for new worlds)
int64_t getInitialWorldTime();

// Initialize a new world with spawn selection.
// Combines spawn selection with world time initialization.
WorldSpawnState initializeWorld(int64_t seed);

// Biome IDs for spawn preference checking
namespace biome {
constexpr uint32_t OCEAN = 0;
constexpr uint32_t PLAINS = 1;
constexpr uint32_t DESERT = 2;
constexpr uint32_t EXTREME_HILLS = 3;
constexpr uint32_t FOREST = 4;
constexpr uint32_t TAIGA = 5;
constexpr uint32_t SWAMP = 6;
constexpr uint32_t RIVER = 7;
constexpr uint32_t FROZEN_OCEAN = 10;
constexpr uint32_t ICE_PLAINS = 12;
constexpr uint32_t BEACH = 16;
constexpr uint32_t JUNGLE = 21;
constexpr uint32_t SAVANNA = 35;
}  // namespace biome

}  // namespace spawn
}  // namespace mc189
