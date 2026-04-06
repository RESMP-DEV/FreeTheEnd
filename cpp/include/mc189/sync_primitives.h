#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace mcsim::vulkan {

/// Pipeline stage groups for simulation tick phases
enum class TickStage : uint32_t {
    Input = 0,      // Player action processing
    Physics,        // Movement, collisions
    BlockUpdate,    // Block state changes
    MobAI,          // Entity AI decisions
    Combat,         // Dragon, combat resolution
    StatusEffects,  // Hunger, health, buffs
    Portals,        // Dimension transitions
    Output,         // Result collection
    Count
};

/// Memory access patterns for barrier insertion
struct AccessPattern {
    VkPipelineStageFlags2 stageMask;
    VkAccessFlags2 accessMask;
};

/// Pre-defined access patterns for common compute operations
namespace Access {
    constexpr AccessPattern ComputeRead{
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    };
    constexpr AccessPattern ComputeWrite{
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
    };
    constexpr AccessPattern ComputeReadWrite{
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
    };
    constexpr AccessPattern TransferRead{
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT
    };
    constexpr AccessPattern TransferWrite{
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT
    };
    constexpr AccessPattern HostRead{
        VK_PIPELINE_STAGE_2_HOST_BIT,
        VK_ACCESS_2_HOST_READ_BIT
    };
}

/// Buffer memory barrier for synchronization between shader stages
struct BufferBarrier {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize size;
    AccessPattern src;
    AccessPattern dst;
    uint32_t srcQueueFamily = VK_QUEUE_FAMILY_IGNORED;
    uint32_t dstQueueFamily = VK_QUEUE_FAMILY_IGNORED;

    [[nodiscard]] VkBufferMemoryBarrier2 toVkBarrier() const noexcept;
};

/// Global memory barrier (no specific buffer)
struct GlobalBarrier {
    AccessPattern src;
    AccessPattern dst;

    [[nodiscard]] VkMemoryBarrier2 toVkBarrier() const noexcept;
};

/// Batch multiple barriers into a single pipeline barrier
class BarrierBatch {
public:
    void addBuffer(const BufferBarrier& barrier);
    void addGlobal(const GlobalBarrier& barrier);
    void clear() noexcept;

    /// Record barrier command to command buffer
    void record(VkCommandBuffer cmd) const;

    [[nodiscard]] bool empty() const noexcept {
        return bufferBarriers_.empty() && globalBarriers_.empty();
    }

    [[nodiscard]] size_t count() const noexcept {
        return bufferBarriers_.size() + globalBarriers_.size();
    }

private:
    std::vector<VkBufferMemoryBarrier2> bufferBarriers_;
    std::vector<VkMemoryBarrier2> globalBarriers_;
};

/// Stage dependency tracking for automatic barrier insertion
class StageDependencyTracker {
public:
    struct BufferAccessInfo {
        VkBuffer buffer;
        VkDeviceSize offset;
        VkDeviceSize size;
        bool isWrite;
    };

    /// Mark that a buffer is written by a stage
    void markWrite(uint32_t stageIndex, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size);

    /// Mark that a buffer is read by a stage
    void markRead(uint32_t stageIndex, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size);

    /// Generate barriers needed before executing the given stage
    /// Returns barriers for write-after-read and read-after-write hazards
    void generateBarriers(uint32_t stageIndex, BarrierBatch& batch) const;

    /// Add barriers for current access set based on previous accesses
    void addBarriersForAccesses(uint32_t stageIndex,
                                std::span<const BufferAccessInfo> accesses,
                                BarrierBatch& batch) const;

    /// Record current access set for dependency tracking
    void recordAccesses(uint32_t stageIndex, std::span<const BufferAccessInfo> accesses);

    /// Clear all tracking state (call between frames)
    void reset() noexcept;

private:
    struct BufferAccess {
        VkBuffer buffer;
        VkDeviceSize offset;
        VkDeviceSize size;
        uint32_t stageIndex;
        bool isWrite;
    };

    std::vector<BufferAccess> accesses_;

    [[nodiscard]] bool overlaps(const BufferAccess& a, const BufferAccess& b) const noexcept;
};

/// Fence pool for reusing fences across frames
class FencePool {
public:
    explicit FencePool(VkDevice device, uint32_t initialCount = 8);
    ~FencePool();

    FencePool(const FencePool&) = delete;
    FencePool& operator=(const FencePool&) = delete;
    FencePool(FencePool&&) noexcept;
    FencePool& operator=(FencePool&&) noexcept;

    /// Acquire a fence (creates new if none available)
    [[nodiscard]] VkFence acquire();

    /// Return fence to pool after signaled (caller must ensure fence is signaled)
    void release(VkFence fence);

    /// Reset all fences in pool
    void resetAll();

    [[nodiscard]] uint32_t activeCount() const noexcept { return activeCount_; }
    [[nodiscard]] uint32_t totalCount() const noexcept { return static_cast<uint32_t>(fences_.size()); }

private:
    VkDevice device_;
    std::vector<VkFence> fences_;
    std::vector<VkFence> available_;
    uint32_t activeCount_ = 0;
};

/// Semaphore pool for timeline semaphores (cross-queue sync)
class TimelineSemaphorePool {
public:
    explicit TimelineSemaphorePool(VkDevice device, uint32_t initialCount = 4);
    ~TimelineSemaphorePool();

    TimelineSemaphorePool(const TimelineSemaphorePool&) = delete;
    TimelineSemaphorePool& operator=(const TimelineSemaphorePool&) = delete;

    /// Acquire a timeline semaphore with initial value
    [[nodiscard]] VkSemaphore acquire(uint64_t initialValue = 0);

    /// Return semaphore to pool
    void release(VkSemaphore semaphore);

    /// Wait for semaphore to reach value on host
    VkResult waitHost(VkSemaphore semaphore, uint64_t value, uint64_t timeout = UINT64_MAX);

    /// Signal semaphore to value from host
    VkResult signalHost(VkSemaphore semaphore, uint64_t value);

    /// Query current semaphore value
    [[nodiscard]] uint64_t getValue(VkSemaphore semaphore) const;

private:
    VkDevice device_;
    std::vector<VkSemaphore> semaphores_;
    std::vector<VkSemaphore> available_;
};

/// Per-frame synchronization primitives
struct FrameSync {
    VkSemaphore computeFinished;    // Signaled when compute work done
    VkFence frameFence;             // CPU wait for frame completion
    uint64_t timelineValue;         // Current timeline semaphore value

    void wait(VkDevice device, uint64_t timeout = UINT64_MAX) const;
    void reset(VkDevice device) const;
};

/// Create pipeline barrier info for compute-to-compute synchronization
[[nodiscard]] VkDependencyInfo createComputeToComputeDependency(
    std::span<const VkBufferMemoryBarrier2> bufferBarriers,
    std::span<const VkMemoryBarrier2> memoryBarriers = {}
);

/// Utility: Create execution-only barrier (no memory sync, just ordering)
[[nodiscard]] VkMemoryBarrier2 createExecutionBarrier(
    VkPipelineStageFlags2 srcStage,
    VkPipelineStageFlags2 dstStage
);

/// Utility: Create full compute memory barrier
[[nodiscard]] VkMemoryBarrier2 createFullComputeBarrier();

/// Inter-phase barrier configurations for tick execution
namespace PhaseBarriers {
    /// Barrier between Input and Physics phases
    /// Input writes player actions; Physics reads and updates positions
    [[nodiscard]] GlobalBarrier inputToPhysics() noexcept;

    /// Barrier between Physics and BlockUpdate phases
    /// Physics updates positions; BlockUpdate reads positions and modifies blocks
    [[nodiscard]] GlobalBarrier physicsToBlockUpdate() noexcept;

    /// Barrier between BlockUpdate and MobAI phases
    /// BlockUpdate modifies world state; MobAI reads blocks and updates mobs
    [[nodiscard]] GlobalBarrier blockUpdateToMobAI() noexcept;

    /// Barrier between MobAI and Combat phases
    /// MobAI updates mob state; Combat reads positions and resolves damage
    [[nodiscard]] GlobalBarrier mobAIToCombat() noexcept;

    /// Barrier between Combat and StatusEffects phases
    /// Combat applies damage; StatusEffects updates health/hunger
    [[nodiscard]] GlobalBarrier combatToStatusEffects() noexcept;

    /// Barrier between StatusEffects and Portals phases
    /// StatusEffects updates player state; Portals handles teleportation
    [[nodiscard]] GlobalBarrier statusEffectsToPortals() noexcept;

    /// Barrier between Portals and Output phases
    /// Portals finalizes positions; Output collects results
    [[nodiscard]] GlobalBarrier portalsToOutput() noexcept;

    /// Get barrier for transitioning between any two phases
    [[nodiscard]] GlobalBarrier forTransition(TickStage from, TickStage to) noexcept;
}

/// Specialized barrier batch for tick phases with optimized coalescing
class TickPhaseBarrier {
public:
    /// Record optimized barrier for phase transition
    void record(VkCommandBuffer cmd, TickStage from, TickStage to);

    /// Record barriers for all phase transitions in a tick
    void recordFullTick(VkCommandBuffer cmd);

    /// Get total barriers that will be inserted for full tick
    [[nodiscard]] static constexpr uint32_t fullTickBarrierCount() noexcept {
        return static_cast<uint32_t>(TickStage::Count) - 1;
    }

private:
    BarrierBatch batch_;
};

/// Multi-queue synchronization for async compute
class AsyncComputeSync {
public:
    explicit AsyncComputeSync(VkDevice device);
    ~AsyncComputeSync();

    AsyncComputeSync(const AsyncComputeSync&) = delete;
    AsyncComputeSync& operator=(const AsyncComputeSync&) = delete;

    /// Get semaphore for signaling graphics queue completion
    [[nodiscard]] VkSemaphore graphicsCompleteSemaphore() const noexcept { return graphicsComplete_; }

    /// Get semaphore for signaling compute queue completion
    [[nodiscard]] VkSemaphore computeCompleteSemaphore() const noexcept { return computeComplete_; }

    /// Get current timeline value
    [[nodiscard]] uint64_t timelineValue() const noexcept { return timelineValue_; }

    /// Advance timeline and return new value
    [[nodiscard]] uint64_t advanceTimeline() noexcept { return ++timelineValue_; }

    /// Create submit info for graphics queue (waits on compute, signals graphics)
    [[nodiscard]] VkSemaphoreSubmitInfo graphicsWaitInfo() const noexcept;
    [[nodiscard]] VkSemaphoreSubmitInfo graphicsSignalInfo() const noexcept;

    /// Create submit info for compute queue (waits on graphics, signals compute)
    [[nodiscard]] VkSemaphoreSubmitInfo computeWaitInfo() const noexcept;
    [[nodiscard]] VkSemaphoreSubmitInfo computeSignalInfo() const noexcept;

private:
    VkDevice device_;
    VkSemaphore graphicsComplete_;
    VkSemaphore computeComplete_;
    uint64_t timelineValue_ = 0;
};

/// Ring buffer of frame synchronization primitives
class FrameSyncRing {
public:
    static constexpr uint32_t MaxFramesInFlight = 3;

    explicit FrameSyncRing(VkDevice device);
    ~FrameSyncRing();

    FrameSyncRing(const FrameSyncRing&) = delete;
    FrameSyncRing& operator=(const FrameSyncRing&) = delete;

    /// Get sync primitives for current frame
    [[nodiscard]] FrameSync& current() noexcept { return frames_[currentFrame_]; }
    [[nodiscard]] const FrameSync& current() const noexcept { return frames_[currentFrame_]; }

    /// Advance to next frame
    void advance() noexcept { currentFrame_ = (currentFrame_ + 1) % MaxFramesInFlight; }

    /// Get current frame index
    [[nodiscard]] uint32_t frameIndex() const noexcept { return currentFrame_; }

    /// Wait for all frames to complete
    void waitAll(uint64_t timeout = UINT64_MAX) const;

    /// Reset all frame fences
    void resetAll();

private:
    VkDevice device_;
    std::array<FrameSync, MaxFramesInFlight> frames_{};
    uint32_t currentFrame_ = 0;
};

} // namespace mcsim::vulkan
