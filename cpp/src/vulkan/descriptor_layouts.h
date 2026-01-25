#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mcsim {

/**
 * DescriptorSetLayoutCache - Manages and caches descriptor set layouts
 *
 * Provides pre-defined layouts for all Minecraft simulation shader categories:
 * - World generation (terrain, structures)
 * - Game tick (player, mobs, chunks, combat)
 * - Dragon fight (dragon state, crystals, players)
 * - Inventory and crafting
 * - Physics and block updates
 * - Player mechanics (hunger, health, status effects)
 */
class DescriptorSetLayoutCache {
public:
    /**
     * Layout identifiers for different shader binding patterns.
     * Multiple shaders may share the same layout.
     */
    enum class LayoutType : uint32_t {
        // World generation layouts
        TerrainGen,           // overworld_terrain, nether_terrain, end_terrain

        // Game tick layouts (uses multiple sets)
        GameTickSet0,         // Core game state: player, input, mobs, chunks
        GameTickSet1,         // Update queues: block updates, combat, dragon fight
        GameTickSet2,         // Indirect dispatch and game state
        GameTickSet3,         // Inventory

        // Dragon fight layouts
        DragonAI,             // Dragon state, crystals, players, perches, RNG
        DragonCombat,         // Dragon, crystals, players, attacks, combat output

        // Item/inventory layouts
        InventoryOps,         // Inventory buffer, operations, results, item meta
        Crafting,             // Inventory, recipes, requests, results

        // Physics/world update layouts
        BlockUpdates,         // Modifications, voxel grid, scheduled updates, lighting

        // Player mechanics layouts
        HungerTick,           // Health, max health, food, saturation, exhaustion buffers
        HealthRegen,          // Similar to hunger but for regeneration
        StatusEffects,        // Status effect processing

        // Portal mechanics
        PortalTick,           // Portal state and dimension transitions

        // Mob AI layouts
        MobAIBase,            // Mob buffer, player positions, pathfinding
        MobAIEnderman,        // Extended enderman-specific buffers
        MobAIBlaze,           // Blaze-specific buffers

        // Experience system
        Experience,           // XP orbs and level calculation

        // Utility
        AABBOps,              // Collision detection buffers

        Count
    };

    DescriptorSetLayoutCache() = default;
    ~DescriptorSetLayoutCache();

    DescriptorSetLayoutCache(const DescriptorSetLayoutCache&) = delete;
    DescriptorSetLayoutCache& operator=(const DescriptorSetLayoutCache&) = delete;

    /**
     * Initialize all standard layouts for the given device.
     * @param device Valid Vulkan logical device
     * @return true on success
     */
    bool init(VkDevice device);

    /**
     * Clean up all cached layouts.
     */
    void destroy();

    /**
     * Get a pre-defined layout by type.
     */
    VkDescriptorSetLayout get(LayoutType type) const;

    /**
     * Get or create a custom layout from bindings.
     * Caches the result for future requests with identical bindings.
     */
    VkDescriptorSetLayout get_or_create(
        const std::vector<VkDescriptorSetLayoutBinding>& bindings);

    /**
     * Create a pipeline layout from descriptor set layouts and push constant ranges.
     * The caller is responsible for destroying the returned layout.
     */
    VkPipelineLayout create_pipeline_layout(
        const std::vector<VkDescriptorSetLayout>& set_layouts,
        const std::vector<VkPushConstantRange>& push_constants) const;

    bool is_valid() const { return device_ != VK_NULL_HANDLE; }
    const std::string& error_message() const { return error_msg_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSetLayout, static_cast<size_t>(LayoutType::Count)> layouts_{};
    std::unordered_map<size_t, VkDescriptorSetLayout> custom_layouts_;
    std::string error_msg_;

    bool create_terrain_gen_layout();
    bool create_game_tick_layouts();
    bool create_dragon_layouts();
    bool create_inventory_layouts();
    bool create_block_update_layout();
    bool create_player_mechanics_layouts();
    bool create_mob_ai_layouts();
    bool create_utility_layouts();

    VkDescriptorSetLayout create_layout(
        const std::vector<VkDescriptorSetLayoutBinding>& bindings,
        const char* debug_name = nullptr);

    static size_t hash_bindings(const std::vector<VkDescriptorSetLayoutBinding>& bindings);
};

/**
 * Common push constant structures used by shaders.
 * These match the GLSL layout(push_constant) definitions.
 */
namespace PushConstants {

/**
 * Terrain generation push constants (overworld_terrain.comp)
 */
struct ChunkParams {
    int32_t chunk_pos[3];
    int32_t seed;
    float base_height;
    float height_scale;
    int32_t lod_level;
    uint32_t generation_flags;
};
static_assert(sizeof(ChunkParams) == 32, "ChunkParams must be 32 bytes");

/**
 * Game tick push constants (game_tick.comp)
 */
struct GameTickStage {
    uint32_t stage;
};
static_assert(sizeof(GameTickStage) == 4, "GameTickStage must be 4 bytes");

/**
 * Dragon AI/Combat push constants
 */
struct DragonTick {
    float delta_time;
    float total_time;
    uint32_t tick;
};
static_assert(sizeof(DragonTick) == 12, "DragonTick must be 12 bytes");

/**
 * Inventory operations push constants
 */
struct InventoryOps {
    uint32_t num_agents;
    uint32_t num_operations;
};
static_assert(sizeof(InventoryOps) == 8, "InventoryOps must be 8 bytes");

/**
 * Crafting push constants
 */
struct Crafting {
    uint32_t num_agents;
    uint32_t num_requests;
    uint32_t num_recipes;
    uint32_t use_crafting_table;
};
static_assert(sizeof(Crafting) == 16, "Crafting must be 16 bytes");

/**
 * Block updates push constants
 */
struct BlockUpdates {
    uint32_t max_updates;
    uint32_t max_light_updates;
    uint32_t current_tick;
    uint32_t _pad;
};
static_assert(sizeof(BlockUpdates) == 16, "BlockUpdates must be 16 bytes");

/**
 * Hunger tick push constants
 */
struct HungerTick {
    uint32_t player_count;
    float tick_delta;
    uint32_t game_difficulty;
};
static_assert(sizeof(HungerTick) == 12, "HungerTick must be 12 bytes");

/**
 * Generic count-based push constant
 */
struct CountOnly {
    uint32_t count;
};
static_assert(sizeof(CountOnly) == 4, "CountOnly must be 4 bytes");

}  // namespace PushConstants

/**
 * Specialization constant IDs for shader tuning.
 */
namespace SpecConstants {
    constexpr uint32_t LOCAL_SIZE_X = 0;
    constexpr uint32_t LOCAL_SIZE_Y = 1;
    constexpr uint32_t LOCAL_SIZE_Z = 2;
    constexpr uint32_t MAX_MOBS = 3;
    constexpr uint32_t MAX_CHUNKS = 4;
    constexpr uint32_t CHUNK_SIZE = 5;
    constexpr uint32_t CHUNK_HEIGHT = 6;
}  // namespace SpecConstants

}  // namespace mcsim
