// chunk_cache.cpp - LRU cache for terrain chunks
// Chunks generated on GPU, cached on CPU, uploaded to GPU as needed
//
// Design:
// - Stable slot-based storage (eviction doesn't invalidate other indices)
// - O(1) lookup via coordinate hash
// - LRU eviction with global tick counter
// - Thread-safe for single-writer / multiple-reader access

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mc189 {

// Chunk dimensions matching shader
constexpr uint32_t kChunkSizeX = 16;
constexpr uint32_t kChunkSizeY = 256;
constexpr uint32_t kChunkSizeZ = 16;
constexpr size_t kBlocksPerChunk = kChunkSizeX * kChunkSizeY * kChunkSizeZ;  // 65536
constexpr size_t kBiomesPerChunk = kChunkSizeX * kChunkSizeZ;  // 256

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

// Packed chunk key: upper 32 bits = x, lower 32 bits = z
inline uint64_t chunk_key(int32_t x, int32_t z) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(z));
}

inline void unpack_key(uint64_t key, int32_t& x, int32_t& z) {
    x = static_cast<int32_t>(key >> 32);
    z = static_cast<int32_t>(key & 0xFFFFFFFF);
}

// Coordinate hash with good distribution for Minecraft chunk patterns
struct ChunkKeyHash {
    size_t operator()(uint64_t key) const noexcept {
        // Fibonacci hashing for spatial coherence
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
    bool dirty = false;  // Needs GPU upload

    // Block data: Y is vertical, indexed as blocks[y * 256 + z * 16 + x]
    // Using uint8_t: 0 = air, 1-255 = block IDs
    std::array<uint8_t, kBlocksPerChunk> blocks;

    // Biome per column
    std::array<uint8_t, kBiomesPerChunk> biomes;

    // Heightmap cache (surface Y for each column)
    std::array<int16_t, kBiomesPerChunk> heightmap;

    CachedChunk() {
        blocks.fill(0);  // Air
        biomes.fill(static_cast<uint8_t>(Biome::Plains));
        heightmap.fill(64);
    }

    // Block access: y is 0-255 (vertical), x/z are 0-15 (horizontal)
    uint8_t get_block(int lx, int ly, int lz) const {
        if (lx < 0 || lx >= 16 || ly < 0 || ly >= 256 || lz < 0 || lz >= 16) {
            return 0;  // Air for out-of-bounds
        }
        return blocks[ly * 256 + lz * 16 + lx];
    }

    void set_block(int lx, int ly, int lz, uint8_t block_id) {
        if (lx < 0 || lx >= 16 || ly < 0 || ly >= 256 || lz < 0 || lz >= 16) {
            return;
        }
        blocks[ly * 256 + lz * 16 + lx] = block_id;
        dirty = true;
    }

    uint8_t get_biome(int lx, int lz) const {
        if (lx < 0 || lx >= 16 || lz < 0 || lz >= 16) {
            return static_cast<uint8_t>(Biome::Plains);
        }
        return biomes[lz * 16 + lx];
    }

    int16_t get_height(int lx, int lz) const {
        if (lx < 0 || lx >= 16 || lz < 0 || lz >= 16) {
            return 64;
        }
        return heightmap[lz * 16 + lx];
    }
};

// LRU chunk cache with stable slot allocation
class ChunkCache {
public:
    explicit ChunkCache(size_t max_chunks = 256)
        : max_chunks_(max_chunks), global_tick_(0) {
        cache_.reserve(max_chunks);
        free_slots_.reserve(max_chunks);
    }

    ~ChunkCache() = default;

    // Non-copyable
    ChunkCache(const ChunkCache&) = delete;
    ChunkCache& operator=(const ChunkCache&) = delete;

    // Get or create chunk at coordinates
    // Returns nullptr if cache is full and no eviction possible (shouldn't happen)
    CachedChunk* get_or_create(int32_t x, int32_t z, uint64_t seed) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t key = chunk_key(x, z);
        uint64_t tick = ++global_tick_;

        // Check if already cached
        auto it = index_.find(key);
        if (it != index_.end()) {
            size_t slot = it->second;
            cache_[slot].last_access = tick;
            return &cache_[slot];
        }

        // Need to create new chunk
        size_t slot;
        if (!free_slots_.empty()) {
            // Reuse freed slot
            slot = free_slots_.back();
            free_slots_.pop_back();
        } else if (cache_.size() < max_chunks_) {
            // Allocate new slot
            slot = cache_.size();
            cache_.emplace_back();
        } else {
            // Evict LRU
            slot = evict_lru_locked();
        }

        // Initialize chunk
        CachedChunk& chunk = cache_[slot];
        chunk.x = x;
        chunk.z = z;
        chunk.seed = seed;
        chunk.last_access = tick;
        chunk.valid = false;
        chunk.dirty = true;
        chunk.blocks.fill(0);
        chunk.biomes.fill(static_cast<uint8_t>(Biome::Plains));
        chunk.heightmap.fill(64);

        index_[key] = slot;
        lru_list_.push_front(key);
        lru_iters_[key] = lru_list_.begin();

        return &chunk;
    }

    // Get existing chunk (returns nullptr if not cached)
    CachedChunk* get(int32_t x, int32_t z) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t key = chunk_key(x, z);

        auto it = index_.find(key);
        if (it == index_.end()) {
            return nullptr;
        }

        size_t slot = it->second;
        cache_[slot].last_access = ++global_tick_;

        // Move to front of LRU
        auto lru_it = lru_iters_.find(key);
        if (lru_it != lru_iters_.end()) {
            lru_list_.erase(lru_it->second);
            lru_list_.push_front(key);
            lru_it->second = lru_list_.begin();
        }

        return &cache_[slot];
    }

    // Get existing chunk (const version)
    const CachedChunk* get(int32_t x, int32_t z) const {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t key = chunk_key(x, z);

        auto it = index_.find(key);
        if (it == index_.end()) {
            return nullptr;
        }
        return &cache_[it->second];
    }

    // Check if chunk is cached
    bool contains(int32_t x, int32_t z) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return index_.find(chunk_key(x, z)) != index_.end();
    }

    // Explicitly remove chunk from cache
    void remove(int32_t x, int32_t z) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t key = chunk_key(x, z);

        auto it = index_.find(key);
        if (it == index_.end()) {
            return;
        }

        size_t slot = it->second;
        cache_[slot].valid = false;
        free_slots_.push_back(slot);
        index_.erase(it);

        auto lru_it = lru_iters_.find(key);
        if (lru_it != lru_iters_.end()) {
            lru_list_.erase(lru_it->second);
            lru_iters_.erase(lru_it);
        }
    }

    // Clear entire cache
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
        index_.clear();
        free_slots_.clear();
        lru_list_.clear();
        lru_iters_.clear();
        global_tick_ = 0;
    }

    // Get all chunks that need GPU upload
    std::vector<CachedChunk*> get_dirty_chunks() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<CachedChunk*> dirty;
        dirty.reserve(16);  // Typical batch size
        for (auto& chunk : cache_) {
            if (chunk.valid && chunk.dirty) {
                dirty.push_back(&chunk);
            }
        }
        return dirty;
    }

    // Mark all dirty chunks as clean (after GPU upload)
    void mark_all_clean() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& chunk : cache_) {
            chunk.dirty = false;
        }
    }

    // Statistics
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return index_.size();
    }

    size_t capacity() const { return max_chunks_; }

    uint64_t current_tick() const { return global_tick_.load(); }

    // Memory usage estimate
    size_t memory_bytes() const {
        return cache_.size() * sizeof(CachedChunk) +
               index_.size() * (sizeof(uint64_t) + sizeof(size_t)) +
               free_slots_.capacity() * sizeof(size_t);
    }

private:
    // Evict least recently used chunk (called with lock held)
    size_t evict_lru_locked() {
        if (lru_list_.empty()) {
            // Fallback: find LRU by scanning
            size_t lru_slot = 0;
            uint64_t min_tick = UINT64_MAX;
            for (size_t i = 0; i < cache_.size(); ++i) {
                if (cache_[i].valid && cache_[i].last_access < min_tick) {
                    min_tick = cache_[i].last_access;
                    lru_slot = i;
                }
            }
            uint64_t key = chunk_key(cache_[lru_slot].x, cache_[lru_slot].z);
            index_.erase(key);
            cache_[lru_slot].valid = false;
            return lru_slot;
        }

        // Pop from back of LRU list
        uint64_t key = lru_list_.back();
        lru_list_.pop_back();
        lru_iters_.erase(key);

        auto it = index_.find(key);
        size_t slot = it->second;
        index_.erase(it);
        cache_[slot].valid = false;

        return slot;
    }

    size_t max_chunks_;
    std::atomic<uint64_t> global_tick_;
    mutable std::mutex mutex_;

    // Chunk storage (stable indices, slots can be reused)
    std::vector<CachedChunk> cache_;
    std::vector<size_t> free_slots_;

    // Key -> slot mapping
    std::unordered_map<uint64_t, size_t, ChunkKeyHash> index_;

    // LRU tracking
    std::list<uint64_t> lru_list_;
    std::unordered_map<uint64_t, std::list<uint64_t>::iterator, ChunkKeyHash> lru_iters_;
};

// Global block access through cache
class CachedWorld {
public:
    explicit CachedWorld(uint64_t seed, size_t cache_size = 256)
        : seed_(seed), cache_(cache_size) {}

    // Get block at world coordinates
    uint8_t get_block(int wx, int wy, int wz) {
        int32_t cx = wx >> 4;  // Divide by 16
        int32_t cz = wz >> 4;
        int lx = wx & 15;  // Mod 16
        int lz = wz & 15;

        const CachedChunk* chunk = cache_.get(cx, cz);
        if (!chunk || !chunk->valid) {
            return 0;  // Air for unloaded chunks
        }
        return chunk->get_block(lx, wy, lz);
    }

    // Set block at world coordinates
    void set_block(int wx, int wy, int wz, uint8_t block_id) {
        int32_t cx = wx >> 4;
        int32_t cz = wz >> 4;
        int lx = wx & 15;
        int lz = wz & 15;

        CachedChunk* chunk = cache_.get_or_create(cx, cz, seed_);
        if (chunk) {
            chunk->set_block(lx, wy, lz, block_id);
        }
    }

    // Get height at world XZ
    int16_t get_height(int wx, int wz) {
        int32_t cx = wx >> 4;
        int32_t cz = wz >> 4;
        int lx = wx & 15;
        int lz = wz & 15;

        const CachedChunk* chunk = cache_.get(cx, cz);
        if (!chunk || !chunk->valid) {
            return 64;
        }
        return chunk->get_height(lx, lz);
    }

    // Ensure chunk is loaded (creates if needed)
    CachedChunk* ensure_chunk(int32_t cx, int32_t cz) {
        return cache_.get_or_create(cx, cz, seed_);
    }

    // Check if chunk exists
    bool has_chunk(int32_t cx, int32_t cz) const {
        return cache_.contains(cx, cz);
    }

    // Get chunk for direct access
    CachedChunk* get_chunk(int32_t cx, int32_t cz) {
        return cache_.get(cx, cz);
    }

    const CachedChunk* get_chunk(int32_t cx, int32_t cz) const {
        return cache_.get(cx, cz);
    }

    // Load chunks in radius around position
    void load_radius(int32_t center_cx, int32_t center_cz, int32_t radius) {
        for (int32_t dz = -radius; dz <= radius; ++dz) {
            for (int32_t dx = -radius; dx <= radius; ++dx) {
                cache_.get_or_create(center_cx + dx, center_cz + dz, seed_);
            }
        }
    }

    // Get chunks needing GPU upload
    std::vector<CachedChunk*> get_dirty_chunks() {
        return cache_.get_dirty_chunks();
    }

    void mark_all_clean() {
        cache_.mark_all_clean();
    }

    uint64_t seed() const { return seed_; }
    ChunkCache& cache() { return cache_; }
    const ChunkCache& cache() const { return cache_; }

private:
    uint64_t seed_;
    ChunkCache cache_;
};

}  // namespace mc189

// Expose C-compatible interface for GPU uploader
extern "C" {

// Opaque handle
typedef struct mc189_chunk_cache_t* mc189_chunk_cache_handle;

mc189_chunk_cache_handle mc189_chunk_cache_create(size_t max_chunks) {
    return reinterpret_cast<mc189_chunk_cache_handle>(
        new mc189::ChunkCache(max_chunks));
}

void mc189_chunk_cache_destroy(mc189_chunk_cache_handle cache) {
    delete reinterpret_cast<mc189::ChunkCache*>(cache);
}

// Returns pointer to chunk data, or nullptr if not found
// Data layout: 65536 uint8_t blocks in Y-Z-X order
const uint8_t* mc189_chunk_cache_get(mc189_chunk_cache_handle cache,
                                      int32_t x, int32_t z) {
    auto* c = reinterpret_cast<mc189::ChunkCache*>(cache);
    const mc189::CachedChunk* chunk = c->get(x, z);
    if (!chunk || !chunk->valid) {
        return nullptr;
    }
    return chunk->blocks.data();
}

size_t mc189_chunk_cache_size(mc189_chunk_cache_handle cache) {
    return reinterpret_cast<mc189::ChunkCache*>(cache)->size();
}

void mc189_chunk_cache_clear(mc189_chunk_cache_handle cache) {
    reinterpret_cast<mc189::ChunkCache*>(cache)->clear();
}

}  // extern "C"
