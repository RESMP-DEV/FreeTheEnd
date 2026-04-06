#include "sync_primitives.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>

namespace mcsim::vulkan {

// ============================================================================
// BufferBarrier
// ============================================================================

VkBufferMemoryBarrier2 BufferBarrier::toVkBarrier() const noexcept {
    return VkBufferMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = src.stageMask,
        .srcAccessMask = src.accessMask,
        .dstStageMask = dst.stageMask,
        .dstAccessMask = dst.accessMask,
        .srcQueueFamilyIndex = srcQueueFamily,
        .dstQueueFamilyIndex = dstQueueFamily,
        .buffer = buffer,
        .offset = offset,
        .size = size
    };
}

// ============================================================================
// GlobalBarrier
// ============================================================================

VkMemoryBarrier2 GlobalBarrier::toVkBarrier() const noexcept {
    return VkMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = src.stageMask,
        .srcAccessMask = src.accessMask,
        .dstStageMask = dst.stageMask,
        .dstAccessMask = dst.accessMask
    };
}

// ============================================================================
// BarrierBatch
// ============================================================================

void BarrierBatch::addBuffer(const BufferBarrier& barrier) {
    bufferBarriers_.push_back(barrier.toVkBarrier());
}

void BarrierBatch::addGlobal(const GlobalBarrier& barrier) {
    globalBarriers_.push_back(barrier.toVkBarrier());
}

void BarrierBatch::clear() noexcept {
    bufferBarriers_.clear();
    globalBarriers_.clear();
}

void BarrierBatch::record(VkCommandBuffer cmd) const {
    if (empty()) return;

    VkDependencyInfo depInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = static_cast<uint32_t>(globalBarriers_.size()),
        .pMemoryBarriers = globalBarriers_.empty() ? nullptr : globalBarriers_.data(),
        .bufferMemoryBarrierCount = static_cast<uint32_t>(bufferBarriers_.size()),
        .pBufferMemoryBarriers = bufferBarriers_.empty() ? nullptr : bufferBarriers_.data(),
        .imageMemoryBarrierCount = 0,
        .pImageMemoryBarriers = nullptr
    };

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

// ============================================================================
// StageDependencyTracker
// ============================================================================

void StageDependencyTracker::markWrite(uint32_t stageIndex, VkBuffer buffer,
                                        VkDeviceSize offset, VkDeviceSize size) {
    accesses_.push_back({buffer, offset, size, stageIndex, true});
}

void StageDependencyTracker::markRead(uint32_t stageIndex, VkBuffer buffer,
                                       VkDeviceSize offset, VkDeviceSize size) {
    accesses_.push_back({buffer, offset, size, stageIndex, false});
}

void StageDependencyTracker::generateBarriers(uint32_t stageIndex, BarrierBatch& batch) const {
    // Find all hazards: previous writes that this stage reads, or previous reads that this stage writes
    for (const auto& prev : accesses_) {
        if (prev.stageIndex >= stageIndex) continue; // Only consider earlier stages

        // Check all accesses we're about to make in this stage
        for (const auto& curr : accesses_) {
            if (curr.stageIndex != stageIndex) continue;
            if (curr.buffer != prev.buffer) continue;
            if (!overlaps(prev, curr)) continue;

            // RAW hazard: previous write, current read
            // WAR hazard: previous read, current write
            // WAW hazard: previous write, current write
            if (prev.isWrite || curr.isWrite) {
                batch.addBuffer(BufferBarrier{
                    .buffer = prev.buffer,
                    .offset = std::min(prev.offset, curr.offset),
                    .size = std::max(prev.offset + prev.size, curr.offset + curr.size) -
                            std::min(prev.offset, curr.offset),
                    .src = prev.isWrite ? Access::ComputeWrite : Access::ComputeRead,
                    .dst = curr.isWrite ? Access::ComputeWrite : Access::ComputeRead
                });
            }
        }
    }
}

void StageDependencyTracker::addBarriersForAccesses(
    uint32_t stageIndex,
    std::span<const BufferAccessInfo> accesses,
    BarrierBatch& batch) const {
    if (accesses.empty()) {
        return;
    }

    for (const auto& curr : accesses) {
        for (const auto& prev : accesses_) {
            if (prev.stageIndex >= stageIndex) {
                continue;
            }
            if (prev.buffer != curr.buffer) {
                continue;
            }
            if (!overlaps(prev, {curr.buffer, curr.offset, curr.size, stageIndex, curr.isWrite})) {
                continue;
            }

            if (prev.isWrite || curr.isWrite) {
                VkDeviceSize rangeStart = std::min(prev.offset, curr.offset);
                VkDeviceSize rangeEnd = std::max(prev.offset + prev.size, curr.offset + curr.size);
                batch.addBuffer(BufferBarrier{
                    .buffer = curr.buffer,
                    .offset = rangeStart,
                    .size = rangeEnd - rangeStart,
                    .src = prev.isWrite ? Access::ComputeWrite : Access::ComputeRead,
                    .dst = curr.isWrite ? Access::ComputeWrite : Access::ComputeRead
                });
            }
        }
    }
}

void StageDependencyTracker::recordAccesses(uint32_t stageIndex,
                                            std::span<const BufferAccessInfo> accesses) {
    for (const auto& access : accesses) {
        if (access.isWrite) {
            markWrite(stageIndex, access.buffer, access.offset, access.size);
        } else {
            markRead(stageIndex, access.buffer, access.offset, access.size);
        }
    }
}

void StageDependencyTracker::reset() noexcept {
    accesses_.clear();
}

bool StageDependencyTracker::overlaps(const BufferAccess& a, const BufferAccess& b) const noexcept {
    VkDeviceSize aEnd = a.offset + a.size;
    VkDeviceSize bEnd = b.offset + b.size;
    return a.offset < bEnd && b.offset < aEnd;
}

// ============================================================================
// FencePool
// ============================================================================

FencePool::FencePool(VkDevice device, uint32_t initialCount)
    : device_(device) {
    fences_.reserve(initialCount);
    available_.reserve(initialCount);

    VkFenceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0
    };

    for (uint32_t i = 0; i < initialCount; ++i) {
        VkFence fence;
        if (vkCreateFence(device_, &createInfo, nullptr, &fence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence");
        }
        fences_.push_back(fence);
        available_.push_back(fence);
    }
}

FencePool::~FencePool() {
    for (VkFence fence : fences_) {
        vkDestroyFence(device_, fence, nullptr);
    }
}

FencePool::FencePool(FencePool&& other) noexcept
    : device_(other.device_)
    , fences_(std::move(other.fences_))
    , available_(std::move(other.available_))
    , activeCount_(other.activeCount_) {
    other.device_ = VK_NULL_HANDLE;
    other.activeCount_ = 0;
}

FencePool& FencePool::operator=(FencePool&& other) noexcept {
    if (this != &other) {
        for (VkFence fence : fences_) {
            vkDestroyFence(device_, fence, nullptr);
        }
        device_ = other.device_;
        fences_ = std::move(other.fences_);
        available_ = std::move(other.available_);
        activeCount_ = other.activeCount_;
        other.device_ = VK_NULL_HANDLE;
        other.activeCount_ = 0;
    }
    return *this;
}

VkFence FencePool::acquire() {
    if (available_.empty()) {
        VkFenceCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0
        };
        VkFence fence;
        if (vkCreateFence(device_, &createInfo, nullptr, &fence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fence");
        }
        fences_.push_back(fence);
        available_.push_back(fence);
    }

    VkFence fence = available_.back();
    available_.pop_back();
    ++activeCount_;
    return fence;
}

void FencePool::release(VkFence fence) {
    assert(activeCount_ > 0);
    vkResetFences(device_, 1, &fence);
    available_.push_back(fence);
    --activeCount_;
}

void FencePool::resetAll() {
    if (!fences_.empty()) {
        vkResetFences(device_, static_cast<uint32_t>(fences_.size()), fences_.data());
    }
    available_ = fences_;
    activeCount_ = 0;
}

// ============================================================================
// TimelineSemaphorePool
// ============================================================================

TimelineSemaphorePool::TimelineSemaphorePool(VkDevice device, uint32_t initialCount)
    : device_(device) {
    semaphores_.reserve(initialCount);
    available_.reserve(initialCount);

    for (uint32_t i = 0; i < initialCount; ++i) {
        VkSemaphoreTypeCreateInfo typeInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .pNext = nullptr,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = 0
        };

        VkSemaphoreCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = &typeInfo,
            .flags = 0
        };

        VkSemaphore semaphore;
        if (vkCreateSemaphore(device_, &createInfo, nullptr, &semaphore) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create timeline semaphore");
        }
        semaphores_.push_back(semaphore);
        available_.push_back(semaphore);
    }
}

TimelineSemaphorePool::~TimelineSemaphorePool() {
    for (VkSemaphore semaphore : semaphores_) {
        vkDestroySemaphore(device_, semaphore, nullptr);
    }
}

VkSemaphore TimelineSemaphorePool::acquire(uint64_t initialValue) {
    if (available_.empty()) {
        VkSemaphoreTypeCreateInfo typeInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .pNext = nullptr,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = initialValue
        };

        VkSemaphoreCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = &typeInfo,
            .flags = 0
        };

        VkSemaphore semaphore;
        if (vkCreateSemaphore(device_, &createInfo, nullptr, &semaphore) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create timeline semaphore");
        }
        semaphores_.push_back(semaphore);
        return semaphore;
    }

    VkSemaphore semaphore = available_.back();
    available_.pop_back();
    return semaphore;
}

void TimelineSemaphorePool::release(VkSemaphore semaphore) {
    available_.push_back(semaphore);
}

VkResult TimelineSemaphorePool::waitHost(VkSemaphore semaphore, uint64_t value, uint64_t timeout) {
    VkSemaphoreWaitInfo waitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext = nullptr,
        .flags = 0,
        .semaphoreCount = 1,
        .pSemaphores = &semaphore,
        .pValues = &value
    };
    return vkWaitSemaphores(device_, &waitInfo, timeout);
}

VkResult TimelineSemaphorePool::signalHost(VkSemaphore semaphore, uint64_t value) {
    VkSemaphoreSignalInfo signalInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
        .pNext = nullptr,
        .semaphore = semaphore,
        .value = value
    };
    return vkSignalSemaphore(device_, &signalInfo);
}

uint64_t TimelineSemaphorePool::getValue(VkSemaphore semaphore) const {
    uint64_t value = 0;
    vkGetSemaphoreCounterValue(device_, semaphore, &value);
    return value;
}

// ============================================================================
// FrameSync
// ============================================================================

void FrameSync::wait(VkDevice device, uint64_t timeout) const {
    vkWaitForFences(device, 1, &frameFence, VK_TRUE, timeout);
}

void FrameSync::reset(VkDevice device) const {
    vkResetFences(device, 1, &frameFence);
}

// ============================================================================
// Utility Functions
// ============================================================================

VkDependencyInfo createComputeToComputeDependency(
    std::span<const VkBufferMemoryBarrier2> bufferBarriers,
    std::span<const VkMemoryBarrier2> memoryBarriers) {
    return VkDependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = static_cast<uint32_t>(memoryBarriers.size()),
        .pMemoryBarriers = memoryBarriers.empty() ? nullptr : memoryBarriers.data(),
        .bufferMemoryBarrierCount = static_cast<uint32_t>(bufferBarriers.size()),
        .pBufferMemoryBarriers = bufferBarriers.empty() ? nullptr : bufferBarriers.data(),
        .imageMemoryBarrierCount = 0,
        .pImageMemoryBarriers = nullptr
    };
}

VkMemoryBarrier2 createExecutionBarrier(VkPipelineStageFlags2 srcStage,
                                         VkPipelineStageFlags2 dstStage) {
    return VkMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = srcStage,
        .srcAccessMask = VK_ACCESS_2_NONE,
        .dstStageMask = dstStage,
        .dstAccessMask = VK_ACCESS_2_NONE
    };
}

VkMemoryBarrier2 createFullComputeBarrier() {
    return VkMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
    };
}

// ============================================================================
// PhaseBarriers
// ============================================================================

namespace PhaseBarriers {
    namespace {
        constexpr GlobalBarrier computeToCompute() {
            return GlobalBarrier{Access::ComputeWrite, Access::ComputeReadWrite};
        }
    }

    GlobalBarrier inputToPhysics() noexcept {
        return computeToCompute();
    }

    GlobalBarrier physicsToBlockUpdate() noexcept {
        return computeToCompute();
    }

    GlobalBarrier blockUpdateToMobAI() noexcept {
        return computeToCompute();
    }

    GlobalBarrier mobAIToCombat() noexcept {
        return computeToCompute();
    }

    GlobalBarrier combatToStatusEffects() noexcept {
        return computeToCompute();
    }

    GlobalBarrier statusEffectsToPortals() noexcept {
        return computeToCompute();
    }

    GlobalBarrier portalsToOutput() noexcept {
        return computeToCompute();
    }

    GlobalBarrier forTransition(TickStage from, TickStage to) noexcept {
        if (from == to) {
            return computeToCompute();
        }
        switch (from) {
            case TickStage::Input:
                return inputToPhysics();
            case TickStage::Physics:
                return physicsToBlockUpdate();
            case TickStage::BlockUpdate:
                return blockUpdateToMobAI();
            case TickStage::MobAI:
                return mobAIToCombat();
            case TickStage::Combat:
                return combatToStatusEffects();
            case TickStage::StatusEffects:
                return statusEffectsToPortals();
            case TickStage::Portals:
                return portalsToOutput();
            case TickStage::Output:
            case TickStage::Count:
                return computeToCompute();
        }
        return computeToCompute();
    }
}

// ============================================================================
// TickPhaseBarrier
// ============================================================================

void TickPhaseBarrier::record(VkCommandBuffer cmd, TickStage from, TickStage to) {
    batch_.clear();
    batch_.addGlobal(PhaseBarriers::forTransition(from, to));
    batch_.record(cmd);
    batch_.clear();
}

void TickPhaseBarrier::recordFullTick(VkCommandBuffer cmd) {
    TickStage stage = TickStage::Input;
    for (uint32_t i = 1; i < static_cast<uint32_t>(TickStage::Count); ++i) {
        TickStage next = static_cast<TickStage>(i);
        record(cmd, stage, next);
        stage = next;
    }
}

// ============================================================================
// AsyncComputeSync
// ============================================================================

namespace {
    VkSemaphore createTimelineSemaphore(VkDevice device, uint64_t initialValue) {
        VkSemaphoreTypeCreateInfo typeInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .pNext = nullptr,
            .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = initialValue
        };

        VkSemaphoreCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = &typeInfo,
            .flags = 0
        };

        VkSemaphore semaphore = VK_NULL_HANDLE;
        if (vkCreateSemaphore(device, &createInfo, nullptr, &semaphore) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create timeline semaphore");
        }
        return semaphore;
    }
}

AsyncComputeSync::AsyncComputeSync(VkDevice device)
    : device_(device)
    , graphicsComplete_(createTimelineSemaphore(device, 0))
    , computeComplete_(createTimelineSemaphore(device, 0)) {}

AsyncComputeSync::~AsyncComputeSync() {
    if (graphicsComplete_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_, graphicsComplete_, nullptr);
    }
    if (computeComplete_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_, computeComplete_, nullptr);
    }
}

VkSemaphoreSubmitInfo AsyncComputeSync::graphicsWaitInfo() const noexcept {
    return VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = computeComplete_,
        .value = timelineValue_,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0
    };
}

VkSemaphoreSubmitInfo AsyncComputeSync::graphicsSignalInfo() const noexcept {
    return VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = graphicsComplete_,
        .value = timelineValue_,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0
    };
}

VkSemaphoreSubmitInfo AsyncComputeSync::computeWaitInfo() const noexcept {
    return VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = graphicsComplete_,
        .value = timelineValue_,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0
    };
}

VkSemaphoreSubmitInfo AsyncComputeSync::computeSignalInfo() const noexcept {
    return VkSemaphoreSubmitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = computeComplete_,
        .value = timelineValue_,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0
    };
}

// ============================================================================
// FrameSyncRing
// ============================================================================

FrameSyncRing::FrameSyncRing(VkDevice device)
    : device_(device) {
    for (auto& frame : frames_) {
        VkSemaphoreCreateInfo semaphoreInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0
        };

        if (vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &frame.computeFinished) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create compute semaphore");
        }

        VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT
        };

        if (vkCreateFence(device_, &fenceInfo, nullptr, &frame.frameFence) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create frame fence");
        }

        frame.timelineValue = 0;
    }
}

FrameSyncRing::~FrameSyncRing() {
    for (auto& frame : frames_) {
        if (frame.computeFinished != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, frame.computeFinished, nullptr);
        }
        if (frame.frameFence != VK_NULL_HANDLE) {
            vkDestroyFence(device_, frame.frameFence, nullptr);
        }
    }
}

void FrameSyncRing::waitAll(uint64_t timeout) const {
    for (const auto& frame : frames_) {
        vkWaitForFences(device_, 1, &frame.frameFence, VK_TRUE, timeout);
    }
}

void FrameSyncRing::resetAll() {
    for (auto& frame : frames_) {
        vkResetFences(device_, 1, &frame.frameFence);
    }
}

} // namespace mcsim::vulkan
