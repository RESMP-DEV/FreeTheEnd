#pragma once

#include "sync_primitives.h"
#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <cstring>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace mcsim::vulkan {

/// Compute shader identifiers for tick execution pipeline
enum class ShaderStage : uint32_t {
    // Input processing
    InputProcessing = 0,

    // Physics (game_tick)
    GameTick,

    // Block updates
    BlockUpdates,
    BlockBreaking,
    BlockPlacing,

    // Mob AI
    MobAIBase,
    MobAIBlaze,
    MobAIEnderman,
    MobSpawning,

    // Combat
    DragonAI,
    DragonCombat,
    CrystalTick,

    // Status effects
    HungerTick,
    HealthRegen,
    StatusEffects,

    // Portals
    PortalTick,
    DimensionTeleport,

    // Output
    OutputCollection,

    Count
};

/// Maps shader stage to its tick phase
[[nodiscard]] constexpr TickStage shaderStageToTickStage(ShaderStage stage) noexcept {
    switch (stage) {
        case ShaderStage::InputProcessing:
            return TickStage::Input;
        case ShaderStage::GameTick:
            return TickStage::Physics;
        case ShaderStage::BlockUpdates:
        case ShaderStage::BlockBreaking:
        case ShaderStage::BlockPlacing:
            return TickStage::BlockUpdate;
        case ShaderStage::MobAIBase:
        case ShaderStage::MobAIBlaze:
        case ShaderStage::MobAIEnderman:
        case ShaderStage::MobSpawning:
            return TickStage::MobAI;
        case ShaderStage::DragonAI:
        case ShaderStage::DragonCombat:
        case ShaderStage::CrystalTick:
            return TickStage::Combat;
        case ShaderStage::HungerTick:
        case ShaderStage::HealthRegen:
        case ShaderStage::StatusEffects:
            return TickStage::StatusEffects;
        case ShaderStage::PortalTick:
        case ShaderStage::DimensionTeleport:
            return TickStage::Portals;
        case ShaderStage::OutputCollection:
            return TickStage::Output;
        default:
            return TickStage::Count;
    }
}

/// Get human-readable name for shader stage
[[nodiscard]] constexpr std::string_view shaderStageName(ShaderStage stage) noexcept {
    switch (stage) {
        case ShaderStage::InputProcessing: return "input_processing";
        case ShaderStage::GameTick: return "game_tick";
        case ShaderStage::BlockUpdates: return "block_updates";
        case ShaderStage::BlockBreaking: return "block_breaking";
        case ShaderStage::BlockPlacing: return "block_placing";
        case ShaderStage::MobAIBase: return "mob_ai_base";
        case ShaderStage::MobAIBlaze: return "mob_ai_blaze";
        case ShaderStage::MobAIEnderman: return "mob_ai_enderman";
        case ShaderStage::MobSpawning: return "mob_spawning";
        case ShaderStage::DragonAI: return "dragon_ai";
        case ShaderStage::DragonCombat: return "dragon_combat";
        case ShaderStage::CrystalTick: return "crystal_tick";
        case ShaderStage::HungerTick: return "hunger_tick";
        case ShaderStage::HealthRegen: return "health_regen";
        case ShaderStage::StatusEffects: return "status_effects";
        case ShaderStage::PortalTick: return "portal_tick";
        case ShaderStage::DimensionTeleport: return "dimension_teleport";
        case ShaderStage::OutputCollection: return "output_collection";
        default: return "unknown";
    }
}

/// Compute dispatch configuration
struct DispatchConfig {
    uint32_t groupCountX = 1;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;

    /// Compute total workgroups
    [[nodiscard]] uint32_t totalGroups() const noexcept {
        return groupCountX * groupCountY * groupCountZ;
    }

    /// Create dispatch for N items with given local size
    [[nodiscard]] static DispatchConfig forItemCount(uint32_t itemCount, uint32_t localSize) noexcept {
        return DispatchConfig{(itemCount + localSize - 1) / localSize, 1, 1};
    }

    /// Create dispatch for 2D grid
    [[nodiscard]] static DispatchConfig for2D(uint32_t width, uint32_t height,
                                               uint32_t localX, uint32_t localY) noexcept {
        return DispatchConfig{
            (width + localX - 1) / localX,
            (height + localY - 1) / localY,
            1
        };
    }
};

/// Buffer binding for compute shader
struct BufferBinding {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
    uint32_t binding;
    bool isReadOnly;
};

/// Shader dispatch descriptor
struct ShaderDispatch {
    ShaderStage stage;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    DispatchConfig config;
    std::vector<BufferBinding> buffers;

    /// Push constant data (up to 128 bytes)
    std::array<uint8_t, 128> pushConstants{};
    uint32_t pushConstantSize = 0;

    template<typename T>
    void setPushConstants(const T& data) {
        static_assert(sizeof(T) <= 128, "Push constants exceed max size");
        std::memcpy(pushConstants.data(), &data, sizeof(T));
        pushConstantSize = sizeof(T);
    }
};

/// World batch for parallel execution
struct WorldBatch {
    uint32_t worldOffset;       // Starting world index
    uint32_t worldCount;        // Number of worlds in batch
    uint32_t maxPlayers;        // Max players per world
    uint32_t maxMobs;           // Max mobs per world
    uint32_t maxBlocks;         // Max active blocks per world

    /// Compute dispatch size for player-based operations
    [[nodiscard]] DispatchConfig playerDispatch(uint32_t localSize = 64) const noexcept {
        return DispatchConfig::forItemCount(worldCount * maxPlayers, localSize);
    }

    /// Compute dispatch size for mob-based operations
    [[nodiscard]] DispatchConfig mobDispatch(uint32_t localSize = 64) const noexcept {
        return DispatchConfig::forItemCount(worldCount * maxMobs, localSize);
    }

    /// Compute dispatch size for block-based operations
    [[nodiscard]] DispatchConfig blockDispatch(uint32_t localSize = 64) const noexcept {
        return DispatchConfig::forItemCount(worldCount * maxBlocks, localSize);
    }

    /// Compute dispatch size for per-world operations
    [[nodiscard]] DispatchConfig worldDispatch(uint32_t localSize = 64) const noexcept {
        return DispatchConfig::forItemCount(worldCount, localSize);
    }
};

/// Entity counts for dynamic dispatch sizing
struct EntityCounts {
    uint32_t activePlayers;
    uint32_t activeMobs;
    uint32_t activeBlocks;
    uint32_t activePortals;
    uint32_t activeDragons;
    uint32_t activeCrystals;
};

/// Command recorder for simulation tick
class CommandRecorder {
public:
    explicit CommandRecorder(VkDevice device);
    ~CommandRecorder() = default;

    CommandRecorder(const CommandRecorder&) = delete;
    CommandRecorder& operator=(const CommandRecorder&) = delete;
    CommandRecorder(CommandRecorder&&) noexcept = default;
    CommandRecorder& operator=(CommandRecorder&&) noexcept = default;

    /// Begin recording a new tick
    void beginTick(VkCommandBuffer cmd);

    /// Record full simulation tick with automatic barriers
    void recordTick(const WorldBatch& batch, const EntityCounts& counts);

    /// Record a single shader dispatch with dependency tracking
    void recordDispatch(const ShaderDispatch& dispatch);

    /// Insert barrier between tick phases
    void insertPhaseBarrier(TickStage from, TickStage to);

    /// End tick recording
    void endTick();

    /// Get number of dispatches recorded
    [[nodiscard]] uint32_t dispatchCount() const noexcept { return dispatchCount_; }

    /// Get number of barriers inserted
    [[nodiscard]] uint32_t barrierCount() const noexcept { return barrierCount_; }

    /// Set pipeline for a shader stage
    void setPipeline(ShaderStage stage, VkPipeline pipeline, VkPipelineLayout layout);

    /// Set buffer bindings for a shader stage
    void setStageBufferBindings(ShaderStage stage, std::vector<BufferBinding> buffers);

    /// Set descriptor set for tick
    void setDescriptorSet(VkDescriptorSet set, uint32_t setIndex = 0);

    /// Configure workgroup local size for stage
    void setLocalSize(ShaderStage stage, uint32_t localX, uint32_t localY = 1, uint32_t localZ = 1);

    /// Enable/disable shader stage
    void setStageEnabled(ShaderStage stage, bool enabled);

    /// Check if stage is enabled
    [[nodiscard]] bool isStageEnabled(ShaderStage stage) const noexcept;

    /// Set custom dispatch callback for dynamic sizing
    using DispatchSizeCallback = std::function<DispatchConfig(ShaderStage, const WorldBatch&, const EntityCounts&)>;
    void setDispatchSizeCallback(DispatchSizeCallback callback);

private:
    VkDevice device_;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;

    // Pipeline state per shader stage
    struct PipelineState {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        uint32_t localSizeX = 64;
        uint32_t localSizeY = 1;
        uint32_t localSizeZ = 1;
        bool enabled = true;
        std::vector<BufferBinding> buffers;
    };
    std::array<PipelineState, static_cast<size_t>(ShaderStage::Count)> pipelineStates_{};

    // Current descriptor set
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    uint32_t descriptorSetIndex_ = 0;

    // Dependency tracking
    StageDependencyTracker dependencyTracker_;
    BarrierBatch barrierBatch_;

    // Statistics
    uint32_t dispatchCount_ = 0;
    uint32_t barrierCount_ = 0;

    // Current tick phase
    TickStage currentPhase_ = TickStage::Input;

    // Custom dispatch sizing
    DispatchSizeCallback dispatchSizeCallback_;

    /// Get default dispatch config for stage
    [[nodiscard]] DispatchConfig getDefaultDispatch(ShaderStage stage, const WorldBatch& batch,
                                                     const EntityCounts& counts) const noexcept;

    /// Record all dispatches for a tick phase
    void recordPhase(TickStage phase, const WorldBatch& batch, const EntityCounts& counts);

    /// Insert compute barrier
    void insertComputeBarrier();

    // Allow TickExecutionPlan to access internal state for optimized recording
    friend class TickExecutionPlan;
};

/// Pre-built tick execution plan for repeated use
class TickExecutionPlan {
public:
    /// Build execution plan for given batch configuration
    void build(const WorldBatch& batch, const EntityCounts& maxCounts);

    /// Record to command buffer using pre-built plan
    void record(VkCommandBuffer cmd, CommandRecorder& recorder,
                const WorldBatch& batch, const EntityCounts& actualCounts) const;

    /// Check if plan needs rebuild for new configuration
    [[nodiscard]] bool needsRebuild(const WorldBatch& batch, const EntityCounts& maxCounts) const noexcept;

    /// Get total number of dispatch commands
    [[nodiscard]] uint32_t totalDispatches() const noexcept { return static_cast<uint32_t>(dispatches_.size()); }

    /// Get total number of barriers
    [[nodiscard]] uint32_t totalBarriers() const noexcept { return totalBarriers_; }

private:
    struct PlannedDispatch {
        ShaderStage stage;
        bool needsBarrierBefore;
    };

    std::vector<PlannedDispatch> dispatches_;
    WorldBatch cachedBatch_{};
    EntityCounts cachedCounts_{};
    uint32_t totalBarriers_ = 0;
};

/// Batch multiple worlds into optimal command buffer
class BatchedCommandBuilder {
public:
    explicit BatchedCommandBuilder(VkDevice device, uint32_t maxWorldsPerBatch = 64);

    /// Add world to current batch
    void addWorld(uint32_t worldId, const EntityCounts& counts);

    /// Build batches and record to command buffers
    void build(std::span<VkCommandBuffer> commandBuffers, CommandRecorder& recorder);

    /// Get number of batches required
    [[nodiscard]] uint32_t batchCount() const noexcept;

    /// Clear all pending worlds
    void clear() noexcept;

private:
    VkDevice device_;
    uint32_t maxWorldsPerBatch_;

    struct WorldEntry {
        uint32_t worldId;
        EntityCounts counts;
    };
    std::vector<WorldEntry> pendingWorlds_;
};

} // namespace mcsim::vulkan
