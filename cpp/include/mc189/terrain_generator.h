#pragma once
#include "mc189/vulkan_context.h"
#include "mc189/buffer_manager.h"
#include "mc189/dimension.h"
#include <cstdint>
#include <unordered_map>
#include <memory>

namespace mc189 {

// 64-bit chunk key from x,z coords
inline uint64_t chunk_key(int32_t cx, int32_t cz) {
    return (uint64_t(uint32_t(cx)) << 32) | uint32_t(cz);
}

struct ChunkMeta {
    int32_t x, z;
    Dimension dimension;
    uint32_t generation_state;  // 0=empty, 1=base, 2=carved, 3=decorated, 4=lit
    uint32_t last_access_tick;
};

class TerrainGenerator {
public:
    struct Config {
        uint32_t max_cached_chunks = 1024;
        uint32_t chunk_load_radius = 4;  // Load 9x9 chunks
        bool enable_caves = true;
        bool enable_decorations = true;
    };

    TerrainGenerator(VulkanContext& ctx, BufferManager& buf_mgr,
                     const std::string& shader_dir, const Config& config);
    ~TerrainGenerator();

    // Generate/load chunk, returns buffer offset
    uint32_t ensure_chunk(int32_t cx, int32_t cz, Dimension dim, uint64_t seed);

    // Batch generate chunks around position
    void generate_around(float x, float z, Dimension dim, uint64_t seed, uint32_t radius);

    // Get block at world position (downloads if needed)
    uint16_t get_block(float x, float y, float z, Dimension dim);

    // Set block (for player building)
    void set_block(float x, float y, float z, uint16_t block_id, Dimension dim);

    // Find structures
    bool find_stronghold(uint64_t seed, float& out_x, float& out_z);
    bool find_fortress(uint64_t seed, float player_x, float player_z,
                       float& out_x, float& out_y, float& out_z);
    bool find_village(uint64_t seed, float player_x, float player_z,
                      float& out_x, float& out_z);

    // Get chunk buffer for shader binding
    Buffer& get_chunk_buffer() { return chunk_buffer_; }

    // Cache management
    void evict_distant_chunks(float player_x, float player_z, uint32_t keep_radius);
    uint32_t cached_chunk_count() const { return chunk_cache_.size(); }

private:
    void load_shaders(const std::string& shader_dir);
    void dispatch_generation(int32_t cx, int32_t cz, Dimension dim, uint64_t seed);
    void dispatch_carving(int32_t cx, int32_t cz);
    void dispatch_decoration(int32_t cx, int32_t cz);
    void dispatch_lighting(int32_t cx, int32_t cz);
    uint32_t allocate_chunk_slot();
    void free_chunk_slot(uint32_t slot);

    VulkanContext& ctx_;
    BufferManager& buf_mgr_;
    Config config_;

    // Shaders
    std::unique_ptr<ComputePipeline> overworld_gen_;
    std::unique_ptr<ComputePipeline> nether_gen_;
    std::unique_ptr<ComputePipeline> end_gen_;
    std::unique_ptr<ComputePipeline> cave_carver_;
    std::unique_ptr<ComputePipeline> decorator_;
    std::unique_ptr<ComputePipeline> lighter_;
    std::unique_ptr<ComputePipeline> stronghold_finder_;
    std::unique_ptr<ComputePipeline> fortress_finder_;

    // Chunk storage
    Buffer chunk_buffer_;  // Array of ChunkData (16*256*16 uint16 per chunk)
    std::unordered_map<uint64_t, ChunkMeta> chunk_cache_;
    std::vector<uint32_t> free_slots_;

    // Permutation table for Perlin noise (uploaded once per seed)
    Buffer perm_table_buffer_;
    uint64_t current_perm_seed_ = 0;

    // Tick counter for LRU
    uint32_t current_tick_ = 0;
};

} // namespace mc189
