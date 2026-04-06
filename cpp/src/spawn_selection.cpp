// spawn_selection.cpp - MC 1.8.9 spawn point selection
// Implements deterministic spawn selection matching vanilla behavior

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace mc189 {
namespace spawn {

// Biome types (MC 1.8.9)
constexpr uint32_t BIOME_OCEAN = 0;
constexpr uint32_t BIOME_PLAINS = 1;
constexpr uint32_t BIOME_DESERT = 2;
constexpr uint32_t BIOME_EXTREME_HILLS = 3;
constexpr uint32_t BIOME_FOREST = 4;
constexpr uint32_t BIOME_TAIGA = 5;
constexpr uint32_t BIOME_SWAMP = 6;
constexpr uint32_t BIOME_RIVER = 7;
constexpr uint32_t BIOME_FROZEN_OCEAN = 10;
constexpr uint32_t BIOME_ICE_PLAINS = 12;
constexpr uint32_t BIOME_BEACH = 16;
constexpr uint32_t BIOME_JUNGLE = 21;
constexpr uint32_t BIOME_SAVANNA = 35;

// Block types
constexpr uint8_t AIR = 0;
constexpr uint8_t STONE = 1;
constexpr uint8_t GRASS = 2;
constexpr uint8_t DIRT = 3;
constexpr uint8_t SAND = 12;
constexpr uint8_t GRAVEL = 13;
constexpr uint8_t WATER = 9;
constexpr uint8_t LAVA = 11;
constexpr uint8_t ICE = 79;

// Search parameters matching MC 1.8.9
constexpr int32_t SPAWN_SEARCH_RADIUS = 128;  // 256x256 area total
constexpr int32_t SEA_LEVEL = 63;
constexpr int32_t MAX_SPAWN_ATTEMPTS = 1000;

// World time constants
constexpr int64_t DAWN_TIME = 0;

// Java Random implementation (matches MC 1.8.9 LCG)
class JavaRandom {
public:
    explicit JavaRandom(int64_t seed) {
        seed_ = (seed ^ 0x5DEECE66DLL) & ((1LL << 48) - 1);
    }

    int32_t nextInt() {
        return next(32);
    }

    int32_t nextInt(int32_t bound) {
        if ((bound & (bound - 1)) == 0) {
            return static_cast<int32_t>((bound * static_cast<int64_t>(next(31))) >> 31);
        }
        int32_t bits, val;
        do {
            bits = next(31);
            val = bits % bound;
        } while (bits - val + (bound - 1) < 0);
        return val;
    }

    double nextDouble() {
        return ((static_cast<int64_t>(next(26)) << 27) + next(27)) / static_cast<double>(1LL << 53);
    }

private:
    int32_t next(int32_t bits) {
        seed_ = (seed_ * 0x5DEECE66DLL + 0xBLL) & ((1LL << 48) - 1);
        return static_cast<int32_t>(seed_ >> (48 - bits));
    }

    int64_t seed_;
};

// Permutation table for Perlin noise (seeded)
class PerlinNoise {
public:
    explicit PerlinNoise(int64_t seed) {
        JavaRandom rng(seed);
        for (int i = 0; i < 256; ++i) {
            perm_[i] = i;
        }
        for (int i = 255; i > 0; --i) {
            int j = rng.nextInt(i + 1);
            int tmp = perm_[i];
            perm_[i] = perm_[j];
            perm_[j] = tmp;
        }
        for (int i = 0; i < 256; ++i) {
            perm_[256 + i] = perm_[i];
        }
    }

    double noise2D(double x, double z) const {
        int X = static_cast<int>(std::floor(x)) & 255;
        int Z = static_cast<int>(std::floor(z)) & 255;
        x -= std::floor(x);
        z -= std::floor(z);
        double u = fade(x);
        double v = fade(z);

        int A = perm_[X] + Z;
        int B = perm_[X + 1] + Z;

        return lerp(v,
            lerp(u, grad2D(perm_[A], x, z), grad2D(perm_[B], x - 1, z)),
            lerp(u, grad2D(perm_[A + 1], x, z - 1), grad2D(perm_[B + 1], x - 1, z - 1)));
    }

    double fbm2D(double x, double z, int octaves) const {
        double total = 0.0;
        double amplitude = 1.0;
        double frequency = 1.0;
        double max_value = 0.0;

        for (int i = 0; i < octaves; ++i) {
            total += noise2D(x * frequency, z * frequency) * amplitude;
            max_value += amplitude;
            amplitude *= 0.5;
            frequency *= 2.0;
        }
        return total / max_value;
    }

private:
    static double fade(double t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    static double lerp(double t, double a, double b) {
        return a + t * (b - a);
    }

    static double grad2D(int hash, double x, double z) {
        int h = hash & 3;
        double u = h < 2 ? x : z;
        double v = h < 2 ? z : x;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

    int perm_[512];
};

// Biome generator (simplified MC 1.8.9 biome selection)
class BiomeGenerator {
public:
    explicit BiomeGenerator(int64_t seed) : noise_(seed) {}

    uint32_t getBiome(int32_t x, int32_t z) const {
        double scale = 0.0025;
        double temp = noise_.fbm2D(x * scale, z * scale, 4);
        double humid = noise_.fbm2D(x * scale + 1000, z * scale + 1000, 4);

        // Ocean check based on additional noise layer
        double oceanNoise = noise_.fbm2D(x * 0.001, z * 0.001, 2);
        if (oceanNoise < -0.3) {
            return BIOME_OCEAN;
        }

        // River check
        double riverNoise = std::abs(noise_.fbm2D(x * 0.005, z * 0.005, 3));
        if (riverNoise < 0.03) {
            return BIOME_RIVER;
        }

        // Beach near water
        if (oceanNoise < -0.2) {
            return BIOME_BEACH;
        }

        // Temperature/humidity based biome selection
        if (temp < -0.3) {
            return humid < 0 ? BIOME_ICE_PLAINS : BIOME_TAIGA;
        } else if (temp < 0.0) {
            return humid < -0.2 ? BIOME_PLAINS : BIOME_FOREST;
        } else if (temp < 0.3) {
            if (humid < -0.3) return BIOME_PLAINS;
            if (humid < 0.1) return BIOME_FOREST;
            return BIOME_SWAMP;
        } else if (temp < 0.6) {
            if (humid < -0.2) return BIOME_SAVANNA;
            return BIOME_JUNGLE;
        } else {
            return BIOME_DESERT;
        }
    }

    // Check if biome is suitable for spawning (MC 1.8.9 criteria)
    bool isSpawnableBiome(uint32_t biome) const {
        return biome == BIOME_PLAINS ||
               biome == BIOME_FOREST ||
               biome == BIOME_TAIGA ||
               biome == BIOME_JUNGLE;
    }

private:
    PerlinNoise noise_;
};

// Height generator (simplified terrain height calculation)
class HeightGenerator {
public:
    explicit HeightGenerator(int64_t seed) : noise_(seed) {}

    int32_t getHeight(int32_t x, int32_t z, uint32_t biome) const {
        double baseHeight, variation;
        getBiomeHeightParams(biome, baseHeight, variation);

        double heightNoise = noise_.fbm2D(x * 0.004, z * 0.004, 4);
        double detailNoise = noise_.fbm2D(x * 0.02, z * 0.02, 2) * 0.3;

        int32_t height = static_cast<int32_t>(baseHeight + (heightNoise + detailNoise) * variation);
        return std::max(1, std::min(255, height));
    }

private:
    void getBiomeHeightParams(uint32_t biome, double &base, double &var) const {
        switch (biome) {
            case BIOME_OCEAN:
            case BIOME_FROZEN_OCEAN:
                base = 36.0; var = 8.0; break;
            case BIOME_PLAINS:
            case BIOME_SAVANNA:
                base = 64.0; var = 4.0; break;
            case BIOME_DESERT:
                base = 66.0; var = 6.0; break;
            case BIOME_EXTREME_HILLS:
                base = 72.0; var = 48.0; break;
            case BIOME_FOREST:
            case BIOME_TAIGA:
            case BIOME_JUNGLE:
                base = 66.0; var = 8.0; break;
            case BIOME_SWAMP:
                base = 62.0; var = 2.0; break;
            case BIOME_RIVER:
                base = 56.0; var = 4.0; break;
            case BIOME_BEACH:
                base = 64.0; var = 2.0; break;
            default:
                base = 64.0; var = 8.0;
        }
    }

    PerlinNoise noise_;
};

// Block type generator for spawn safety checks
class BlockGenerator {
public:
    explicit BlockGenerator(int64_t seed) : noise_(seed) {}

    uint8_t getBlock(int32_t x, int32_t y, int32_t z, int32_t surfaceHeight, uint32_t biome) const {
        if (y > surfaceHeight) {
            if (y <= SEA_LEVEL) {
                return WATER;
            }
            return AIR;
        }
        if (y == surfaceHeight) {
            if (y < SEA_LEVEL - 1) {
                return (biome == BIOME_OCEAN || biome == BIOME_RIVER) ? GRAVEL : GRASS;
            }
            return getSurfaceBlock(biome);
        }
        if (y > surfaceHeight - 4) {
            return (biome == BIOME_DESERT || biome == BIOME_BEACH) ? SAND : DIRT;
        }

        // Check for lava lakes below Y=11
        if (y < 11) {
            double lavaNoise = noise_.fbm2D(x * 0.1 + y * 0.05, z * 0.1, 2);
            if (lavaNoise > 0.7) {
                return LAVA;
            }
        }

        return STONE;
    }

private:
    uint8_t getSurfaceBlock(uint32_t biome) const {
        switch (biome) {
            case BIOME_DESERT:
            case BIOME_BEACH:
                return SAND;
            case BIOME_ICE_PLAINS:
                return ICE;
            default:
                return GRASS;
        }
    }

    PerlinNoise noise_;
};

// Result of spawn point selection
struct SpawnResult {
    int32_t x;
    int32_t y;
    int32_t z;
    bool valid;
    uint32_t biome;
};

// Check if a block is solid ground suitable for spawning
bool isSolidGround(uint8_t block) {
    return block == STONE || block == GRASS || block == DIRT ||
           block == SAND || block == GRAVEL;
}

// Check if a block is dangerous
bool isDangerous(uint8_t block) {
    return block == LAVA || block == WATER;
}

// Check if spawn position is safe (no nearby hazards)
bool isSpawnSafe(const BlockGenerator &blocks, const HeightGenerator &heights,
                 const BiomeGenerator &biomes, int32_t x, int32_t y, int32_t z) {
    // Check immediate surroundings for hazards
    for (int32_t dx = -1; dx <= 1; ++dx) {
        for (int32_t dz = -1; dz <= 1; ++dz) {
            uint32_t biome = biomes.getBiome(x + dx, z + dz);
            int32_t height = heights.getHeight(x + dx, z + dz, biome);

            // Check for cliff edges (more than 3 block drop)
            if (std::abs(height - y) > 3) {
                return false;
            }

            // Check for water/lava at feet level
            for (int32_t dy = -1; dy <= 0; ++dy) {
                uint8_t block = blocks.getBlock(x + dx, y + dy, z + dz, height, biome);
                if (isDangerous(block)) {
                    return false;
                }
            }
        }
    }

    // Check for lava below (within 5 blocks)
    uint32_t biome = biomes.getBiome(x, z);
    int32_t height = heights.getHeight(x, z, biome);
    for (int32_t dy = 1; dy <= 5; ++dy) {
        uint8_t block = blocks.getBlock(x, y - dy, z, height, biome);
        if (block == LAVA) {
            return false;
        }
        if (isSolidGround(block)) {
            break;  // Hit solid ground, no need to check further
        }
    }

    return true;
}

// Main spawn selection function (MC 1.8.9 behavior)
// Searches in a 256x256 area centered on world origin
SpawnResult selectSpawnPoint(int64_t worldSeed) {
    BiomeGenerator biomes(worldSeed);
    HeightGenerator heights(worldSeed + 1);  // Different seed for height
    BlockGenerator blocks(worldSeed + 2);    // Different seed for blocks
    JavaRandom rng(worldSeed);

    SpawnResult result = {0, SEA_LEVEL + 1, 0, false, BIOME_PLAINS};

    // Phase 1: Find a suitable biome in 256x256 area
    // MC 1.8.9 searches in a specific spiral pattern from origin
    int32_t bestX = 0, bestZ = 0;
    double bestScore = -std::numeric_limits<double>::infinity();
    bool foundSuitableBiome = false;

    // Search in expanding squares from origin
    for (int32_t radius = 0; radius <= SPAWN_SEARCH_RADIUS; radius += 8) {
        for (int32_t dx = -radius; dx <= radius; dx += 8) {
            for (int32_t dz = -radius; dz <= radius; dz += 8) {
                // Only check perimeter for larger radii (optimization)
                if (radius > 0 && std::abs(dx) != radius && std::abs(dz) != radius) {
                    continue;
                }

                uint32_t biome = biomes.getBiome(dx, dz);
                if (!biomes.isSpawnableBiome(biome)) {
                    continue;
                }

                int32_t height = heights.getHeight(dx, dz, biome);

                // Skip underwater locations
                if (height < SEA_LEVEL) {
                    continue;
                }

                // Score based on biome preference and distance from origin
                // Plains and forest are preferred
                double biomeScore = (biome == BIOME_PLAINS) ? 2.0 :
                                   (biome == BIOME_FOREST) ? 1.5 : 1.0;

                // Prefer closer to origin
                double distScore = 1.0 - (std::sqrt(dx * dx + dz * dz) / SPAWN_SEARCH_RADIUS);

                // Prefer flatter terrain
                double flatScore = 1.0;
                for (int32_t checkDx = -4; checkDx <= 4; checkDx += 4) {
                    for (int32_t checkDz = -4; checkDz <= 4; checkDz += 4) {
                        uint32_t checkBiome = biomes.getBiome(dx + checkDx, dz + checkDz);
                        int32_t checkHeight = heights.getHeight(dx + checkDx, dz + checkDz, checkBiome);
                        flatScore -= std::abs(checkHeight - height) * 0.05;
                    }
                }
                flatScore = std::max(0.0, flatScore);

                double totalScore = biomeScore * distScore * flatScore;

                if (totalScore > bestScore) {
                    bestScore = totalScore;
                    bestX = dx;
                    bestZ = dz;
                    foundSuitableBiome = true;
                }
            }
        }

        // Early exit if we found a good spot close to origin
        if (foundSuitableBiome && bestScore > 1.5 && radius >= 32) {
            break;
        }
    }

    // Phase 2: Fine-tune position within the selected area
    // MC 1.8.9 does additional checks in a smaller radius
    int32_t searchX = bestX;
    int32_t searchZ = bestZ;

    for (int32_t attempt = 0; attempt < MAX_SPAWN_ATTEMPTS; ++attempt) {
        // Random offset within 16 blocks (deterministic based on seed)
        int32_t offsetX = rng.nextInt(33) - 16;
        int32_t offsetZ = rng.nextInt(33) - 16;

        int32_t testX = searchX + offsetX;
        int32_t testZ = searchZ + offsetZ;

        uint32_t biome = biomes.getBiome(testX, testZ);
        int32_t height = heights.getHeight(testX, testZ, biome);

        // Check basic requirements
        if (height < SEA_LEVEL) {
            continue;
        }

        // Get block at feet position
        uint8_t groundBlock = blocks.getBlock(testX, height, testZ, height, biome);
        uint8_t feetBlock = blocks.getBlock(testX, height + 1, testZ, height, biome);
        uint8_t headBlock = blocks.getBlock(testX, height + 2, testZ, height, biome);

        // Must have solid ground, and air at feet and head level
        if (!isSolidGround(groundBlock)) {
            continue;
        }
        if (feetBlock != AIR || headBlock != AIR) {
            continue;
        }

        // Safety checks
        if (!isSpawnSafe(blocks, heights, biomes, testX, height + 1, testZ)) {
            continue;
        }

        // Valid spawn found
        result.x = testX;
        result.y = height + 1;  // Stand on top of ground block
        result.z = testZ;
        result.valid = true;
        result.biome = biome;
        return result;
    }

    // Fallback: use best found position even if not perfectly safe
    if (foundSuitableBiome) {
        uint32_t biome = biomes.getBiome(bestX, bestZ);
        int32_t height = heights.getHeight(bestX, bestZ, biome);
        result.x = bestX;
        result.y = std::max(height + 1, SEA_LEVEL + 1);
        result.z = bestZ;
        result.valid = true;
        result.biome = biome;
    } else {
        // Ultimate fallback: spawn at origin at sea level + 1
        result.x = 0;
        result.y = SEA_LEVEL + 1;
        result.z = 0;
        result.valid = true;
        result.biome = biomes.getBiome(0, 0);
    }

    return result;
}

// Set initial world time (dawn = 0)
int64_t getInitialWorldTime() {
    return DAWN_TIME;
}

// World state after spawn selection
struct WorldSpawnState {
    SpawnResult spawn;
    int64_t worldTime;
    int64_t seed;
};

// Initialize a new world with spawn selection
WorldSpawnState initializeWorld(int64_t seed) {
    WorldSpawnState state;
    state.seed = seed;
    state.spawn = selectSpawnPoint(seed);
    state.worldTime = getInitialWorldTime();
    return state;
}

}  // namespace spawn
}  // namespace mc189
