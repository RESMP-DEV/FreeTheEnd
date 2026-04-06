// chunk_cache.h - LRU cache for terrain chunks
// Chunks generated on GPU, cached on CPU, uploaded to GPU as needed

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mc189 {

// Chunk dimensions matching shader
constexpr uint32_t kChunkSizeX = 16;
constexpr uint32_t kChunkSizeY = 256;
constexpr uint32_t kChunkSizeZ = 16;
constexpr size_t kBlocksPerChunk = kChunkSizeX * kChunkSizeY * kChunkSizeZ;
constexpr size_t kBiomesPerChunk = kChunkSizeX * kChunkSizeZ;

// Biome IDs matching overworld_terrain.comp
enum class Biome : uint8_t {
    Ocean = 0,
    Plains = 1,
    Desert = 2,
    ExtremeHills = 3,
    Forest = 4,
    Taiga = 5,
    Swamp = 6,
    River = 7,
    FrozenOcean = 10,
    FrozenRiver = 11,
    IcePlains = 12,
    IceMountains = 13,
    MushroomIsland = 14,
    Beach = 16,
    Jungle = 21,
    Savanna = 35,
    Mesa = 37,
};

// Coordinate hashing
inline uint64_t chunk_key(int32_t x, int32_t z) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(z));
}

inline void unpack_key(uint64_t key, int32_t& x, int32_t& z) {
    x = static_cast<int32_t>(key >> 32);
    z = static_cast<int32_t>(key & 0xFFFFFFFF);
}

struct ChunkKeyHash {
    size_t operator()(uint64_t key) const noexcept {
        constexpr uint64_t kFibMult = 11400714819323198485ULL;
        return static_cast<size_t>((key * kFibMult) >> 32);
    }
};

// Single cached chunk
struct CachedChunk {
    int32_t x = 0;
    int32_t z = 0;
    uint64_t seed = 0;
    uint64_t last_access = 0;
    bool valid = false;
    bool dirty = false;

    std::array<uint8_t, kBlocksPerChunk> blocks;
    std::array<uint8_t, kBiomesPerChunk> biomes;
    std::array<int16_t, kBiomesPerChunk> heightmap;

    CachedChunk();

    uint8_t get_block(int lx, int ly, int lz) const;
    void set_block(int lx, int ly, int lz, uint8_t block_id);
    uint8_t get_biome(int lx, int lz) const;
    int16_t get_height(int lx, int lz) const;
};

// LRU chunk cache
class ChunkCache {
public:
    explicit ChunkCache(size_t max_chunks = 256);
    ~ChunkCache() = default;

    ChunkCache(const ChunkCache&) = delete;
    ChunkCache& operator=(const ChunkCache&) = delete;

    CachedChunk* get_or_create(int32_t x, int32_t z, uint64_t seed);
    CachedChunk* get(int32_t x, int32_t z);
    const CachedChunk* get(int32_t x, int32_t z) const;
    bool contains(int32_t x, int32_t z) const;
    void remove(int32_t x, int32_t z);
    void clear();

    std::vector<CachedChunk*> get_dirty_chunks();
    void mark_all_clean();

    size_t size() const;
    size_t capacity() const { return max_chunks_; }
    uint64_t current_tick() const { return global_tick_.load(); }
    size_t memory_bytes() const;

private:
    size_t evict_lru_locked();

    size_t max_chunks_;
    std::atomic<uint64_t> global_tick_;
    mutable std::mutex mutex_;

    std::vector<CachedChunk> cache_;
    std::vector<size_t> free_slots_;
    std::unordered_map<uint64_t, size_t, ChunkKeyHash> index_;
    std::list<uint64_t> lru_list_;
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator, ChunkKeyHash> lru_iters_;
};

// World with cached chunk access
class CachedWorld {
public:
    explicit CachedWorld(uint64_t seed, size_t cache_size = 256);

    uint8_t get_block(int wx, int wy, int wz);
    void set_block(int wx, int wy, int wz, uint8_t block_id);
    int16_t get_height(int wx, int wz);

    CachedChunk* ensure_chunk(int32_t cx, int32_t cz);
    bool has_chunk(int32_t cx, int32_t cz) const;
    CachedChunk* get_chunk(int32_t cx, int32_t cz);
    const CachedChunk* get_chunk(int32_t cx, int32_t cz) const;

    void load_radius(int32_t center_cx, int32_t center_cz, int32_t radius);
    std::vector<CachedChunk*> get_dirty_chunks();
    void mark_all_clean();

    uint64_t seed() const { return seed_; }
    ChunkCache& cache() { return cache_; }
    const ChunkCache& cache() const { return cache_; }

private:
    uint64_t seed_;
    ChunkCache cache_;
};

}  // namespace mc189

// C API
#ifdef __cplusplus
extern "C" {
#endif

typedef struct mc189_chunk_cache_t* mc189_chunk_cache_handle;

mc189_chunk_cache_handle mc189_chunk_cache_create(size_t max_chunks);
void mc189_chunk_cache_destroy(mc189_chunk_cache_handle cache);
const uint8_t* mc189_chunk_cache_get(mc189_chunk_cache_handle cache,
                                      int32_t x, int32_t z);
size_t mc189_chunk_cache_size(mc189_chunk_cache_handle cache);
void mc189_chunk_cache_clear(mc189_chunk_cache_handle cache);

#ifdef __cplusplus
}
#endif
