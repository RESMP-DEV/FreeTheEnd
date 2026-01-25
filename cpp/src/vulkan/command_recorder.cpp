#include "command_recorder.h"
#include <algorithm>
#include <cassert>
#include <cstring>

namespace mcsim::vulkan {

// ============================================================================
// CommandRecorder
// ============================================================================

CommandRecorder::CommandRecorder(VkDevice device)
    : device_(device) {
    // Initialize all stages as enabled by default
    for (auto& state : pipelineStates_) {
        state.enabled = true;
        state.localSizeX = 64;
        state.localSizeY = 1;
        state.localSizeZ = 1;
    }
}

void CommandRecorder::beginTick(VkCommandBuffer cmd) {
    cmd_ = cmd;
    dispatchCount_ = 0;
    barrierCount_ = 0;
    currentPhase_ = TickStage::Input;
    dependencyTracker_.reset();
    barrierBatch_.clear();
}

void CommandRecorder::recordTick(const WorldBatch& batch, const EntityCounts& counts) {
    assert(cmd_ != VK_NULL_HANDLE && "Must call beginTick before recordTick");

    // Execute tick phases in order with barriers between dependent phases
    // Phase 1: Input
    recordPhase(TickStage::Input, batch, counts);

    // Phase 2: Physics
    insertPhaseBarrier(TickStage::Input, TickStage::Physics);
    recordPhase(TickStage::Physics, batch, counts);

    // Phase 3: Block Updates
    insertPhaseBarrier(TickStage::Physics, TickStage::BlockUpdate);
    recordPhase(TickStage::BlockUpdate, batch, counts);

    // Phase 4: Mob AI
    insertPhaseBarrier(TickStage::BlockUpdate, TickStage::MobAI);
    recordPhase(TickStage::MobAI, batch, counts);

    // Phase 5: Combat
    insertPhaseBarrier(TickStage::MobAI, TickStage::Combat);
    recordPhase(TickStage::Combat, batch, counts);

    // Phase 6: Status Effects
    insertPhaseBarrier(TickStage::Combat, TickStage::StatusEffects);
    recordPhase(TickStage::StatusEffects, batch, counts);

    // Phase 7: Portals
    insertPhaseBarrier(TickStage::StatusEffects, TickStage::Portals);
    recordPhase(TickStage::Portals, batch, counts);

    // Phase 8: Output
    insertPhaseBarrier(TickStage::Portals, TickStage::Output);
    recordPhase(TickStage::Output, batch, counts);
}

void CommandRecorder::recordDispatch(const ShaderDispatch& dispatch) {
    assert(cmd_ != VK_NULL_HANDLE && "Must call beginTick before recordDispatch");

    const auto& state = pipelineStates_[static_cast<size_t>(dispatch.stage)];
    if (!state.enabled || dispatch.pipeline == VK_NULL_HANDLE) {
        return;
    }

    // Skip empty dispatches
    if (dispatch.config.totalGroups() == 0) {
        return;
    }

    uint32_t stageIndex = static_cast<uint32_t>(dispatch.stage);
    std::vector<StageDependencyTracker::BufferAccessInfo> accessInfos;
    accessInfos.reserve(dispatch.buffers.size());
    for (const auto& binding : dispatch.buffers) {
        accessInfos.push_back(StageDependencyTracker::BufferAccessInfo{
            .buffer = binding.buffer,
            .offset = binding.offset,
            .size = binding.range,
            .isWrite = !binding.isReadOnly
        });
    }

    if (!accessInfos.empty()) {
        dependencyTracker_.addBarriersForAccesses(stageIndex, accessInfos, barrierBatch_);
        if (!barrierBatch_.empty()) {
            barrierBatch_.record(cmd_);
            barrierCount_ += static_cast<uint32_t>(barrierBatch_.count());
            barrierBatch_.clear();
        }
        dependencyTracker_.recordAccesses(stageIndex, accessInfos);
    }

    // Bind pipeline
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, dispatch.pipeline);

    // Bind descriptor set if available
    if (descriptorSet_ != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                dispatch.layout, descriptorSetIndex_, 1,
                                &descriptorSet_, 0, nullptr);
    }

    // Push constants if provided
    if (dispatch.pushConstantSize > 0) {
        vkCmdPushConstants(cmd_, dispatch.layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0, dispatch.pushConstantSize,
                           dispatch.pushConstants.data());
    }

    // Dispatch compute
    vkCmdDispatch(cmd_, dispatch.config.groupCountX,
                  dispatch.config.groupCountY,
                  dispatch.config.groupCountZ);

    ++dispatchCount_;
}

void CommandRecorder::insertPhaseBarrier(TickStage from, TickStage to) {
    if (from == to) {
        return;
    }
    barrierBatch_.addGlobal(PhaseBarriers::forTransition(from, to));
    barrierBatch_.record(cmd_);
    barrierCount_ += static_cast<uint32_t>(barrierBatch_.count());
    barrierBatch_.clear();

    currentPhase_ = to;
}

void CommandRecorder::endTick() {
    // Final barrier to ensure all writes are visible
    insertComputeBarrier();
    cmd_ = VK_NULL_HANDLE;
}

void CommandRecorder::setPipeline(ShaderStage stage, VkPipeline pipeline, VkPipelineLayout layout) {
    auto& state = pipelineStates_[static_cast<size_t>(stage)];
    state.pipeline = pipeline;
    state.layout = layout;
}

void CommandRecorder::setStageBufferBindings(ShaderStage stage, std::vector<BufferBinding> buffers) {
    pipelineStates_[static_cast<size_t>(stage)].buffers = std::move(buffers);
}

void CommandRecorder::setDescriptorSet(VkDescriptorSet set, uint32_t setIndex) {
    descriptorSet_ = set;
    descriptorSetIndex_ = setIndex;
}

void CommandRecorder::setLocalSize(ShaderStage stage, uint32_t localX, uint32_t localY, uint32_t localZ) {
    auto& state = pipelineStates_[static_cast<size_t>(stage)];
    state.localSizeX = localX;
    state.localSizeY = localY;
    state.localSizeZ = localZ;
}

void CommandRecorder::setStageEnabled(ShaderStage stage, bool enabled) {
    pipelineStates_[static_cast<size_t>(stage)].enabled = enabled;
}

bool CommandRecorder::isStageEnabled(ShaderStage stage) const noexcept {
    return pipelineStates_[static_cast<size_t>(stage)].enabled;
}

void CommandRecorder::setDispatchSizeCallback(DispatchSizeCallback callback) {
    dispatchSizeCallback_ = std::move(callback);
}

DispatchConfig CommandRecorder::getDefaultDispatch(ShaderStage stage, const WorldBatch& batch,
                                                    const EntityCounts& counts) const noexcept {
    // Use custom callback if provided
    if (dispatchSizeCallback_) {
        return dispatchSizeCallback_(stage, batch, counts);
    }

    const auto& state = pipelineStates_[static_cast<size_t>(stage)];
    uint32_t localSize = state.localSizeX;

    switch (stage) {
        // Input - per player
        case ShaderStage::InputProcessing:
            return DispatchConfig::forItemCount(counts.activePlayers, localSize);

        // Physics - per player (movement, collisions)
        case ShaderStage::GameTick:
            return DispatchConfig::forItemCount(counts.activePlayers, localSize);

        // Block updates - per active block
        case ShaderStage::BlockUpdates:
        case ShaderStage::BlockBreaking:
        case ShaderStage::BlockPlacing:
            return DispatchConfig::forItemCount(counts.activeBlocks, localSize);

        // Mob AI - per mob
        case ShaderStage::MobAIBase:
        case ShaderStage::MobSpawning:
            return DispatchConfig::forItemCount(counts.activeMobs, localSize);

        // Specialized mob AI - only for specific mob types
        case ShaderStage::MobAIBlaze:
        case ShaderStage::MobAIEnderman:
            return DispatchConfig::forItemCount(counts.activeMobs, localSize);

        // Dragon AI - per dragon
        case ShaderStage::DragonAI:
        case ShaderStage::DragonCombat:
            return DispatchConfig::forItemCount(counts.activeDragons, localSize);

        // Crystal tick - per crystal
        case ShaderStage::CrystalTick:
            return DispatchConfig::forItemCount(counts.activeCrystals, localSize);

        // Status effects - per player
        case ShaderStage::HungerTick:
        case ShaderStage::HealthRegen:
        case ShaderStage::StatusEffects:
            return DispatchConfig::forItemCount(counts.activePlayers, localSize);

        // Portals - per active portal
        case ShaderStage::PortalTick:
        case ShaderStage::DimensionTeleport:
            return DispatchConfig::forItemCount(counts.activePortals, localSize);

        // Output - per world
        case ShaderStage::OutputCollection:
            return batch.worldDispatch(localSize);

        default:
            return DispatchConfig{1, 1, 1};
    }
}

void CommandRecorder::recordPhase(TickStage phase, const WorldBatch& batch, const EntityCounts& counts) {
    // Collect all shader stages belonging to this phase
    bool recordedStage = false;
    for (uint32_t i = 0; i < static_cast<uint32_t>(ShaderStage::Count); ++i) {
        ShaderStage stage = static_cast<ShaderStage>(i);
        if (shaderStageToTickStage(stage) != phase) {
            continue;
        }

        const auto& state = pipelineStates_[i];
        if (!state.enabled || state.pipeline == VK_NULL_HANDLE) {
            continue;
        }

        DispatchConfig config = getDefaultDispatch(stage, batch, counts);
        if (config.totalGroups() == 0) {
            continue;
        }

        if (recordedStage) {
            insertComputeBarrier();
        }

        // Build dispatch descriptor
        ShaderDispatch dispatch{
            .stage = stage,
            .pipeline = state.pipeline,
            .layout = state.layout,
            .config = config,
            .buffers = state.buffers,
            .pushConstants = {},
            .pushConstantSize = 0
        };

        // Set push constants with batch info
        struct PushConstants {
            uint32_t worldOffset;
            uint32_t worldCount;
            uint32_t itemCount;
            uint32_t pad;
        };
        PushConstants pc{
            .worldOffset = batch.worldOffset,
            .worldCount = batch.worldCount,
            .itemCount = config.totalGroups() * state.localSizeX,
            .pad = 0
        };
        dispatch.setPushConstants(pc);

        recordDispatch(dispatch);
        recordedStage = true;
    }
}

void CommandRecorder::insertComputeBarrier() {
    VkMemoryBarrier2 barrier = createFullComputeBarrier();

    VkDependencyInfo depInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = 0,
        .pImageMemoryBarriers = nullptr
    };

    vkCmdPipelineBarrier2(cmd_, &depInfo);
    ++barrierCount_;
}

// ============================================================================
// TickExecutionPlan
// ============================================================================

void TickExecutionPlan::build(const WorldBatch& batch, const EntityCounts& maxCounts) {
    dispatches_.clear();
    totalBarriers_ = 0;

    TickStage prevPhase = TickStage::Input;

    // Build ordered list of dispatches with barrier requirements
    for (uint32_t i = 0; i < static_cast<uint32_t>(ShaderStage::Count); ++i) {
        ShaderStage stage = static_cast<ShaderStage>(i);
        TickStage phase = shaderStageToTickStage(stage);

        bool needsBarrier = (phase != prevPhase);
        if (needsBarrier) {
            ++totalBarriers_;
        }

        dispatches_.push_back(PlannedDispatch{
            .stage = stage,
            .needsBarrierBefore = needsBarrier
        });

        prevPhase = phase;
    }

    cachedBatch_ = batch;
    cachedCounts_ = maxCounts;
}

void TickExecutionPlan::record(VkCommandBuffer cmd, CommandRecorder& recorder,
                                const WorldBatch& batch, const EntityCounts& actualCounts) const {
    recorder.beginTick(cmd);

    TickStage currentPhase = TickStage::Input;
    bool recordedInPhase = false;

    for (const auto& planned : dispatches_) {
        TickStage phase = shaderStageToTickStage(planned.stage);

        if (planned.needsBarrierBefore) {
            recorder.insertPhaseBarrier(currentPhase, phase);
            currentPhase = phase;
            recordedInPhase = false;
        }

        if (!recorder.isStageEnabled(planned.stage)) {
            continue;
        }

        // Record the dispatch using recorder's internal logic
        const auto& state = recorder.pipelineStates_[static_cast<size_t>(planned.stage)];
        if (state.pipeline == VK_NULL_HANDLE) {
            continue;
        }

        DispatchConfig config = recorder.getDefaultDispatch(planned.stage, batch, actualCounts);
        if (config.totalGroups() == 0) {
            continue;
        }

        if (recordedInPhase) {
            recorder.insertComputeBarrier();
        }

        ShaderDispatch dispatch{
            .stage = planned.stage,
            .pipeline = state.pipeline,
            .layout = state.layout,
            .config = config,
            .buffers = state.buffers,
            .pushConstants = {},
            .pushConstantSize = 0
        };

        struct PushConstants {
            uint32_t worldOffset;
            uint32_t worldCount;
            uint32_t itemCount;
            uint32_t pad;
        };
        PushConstants pc{
            .worldOffset = batch.worldOffset,
            .worldCount = batch.worldCount,
            .itemCount = config.totalGroups() * state.localSizeX,
            .pad = 0
        };
        dispatch.setPushConstants(pc);

        recorder.recordDispatch(dispatch);
        recordedInPhase = true;
    }

    recorder.endTick();
}

bool TickExecutionPlan::needsRebuild(const WorldBatch& batch, const EntityCounts& maxCounts) const noexcept {
    // Check if batch configuration changed significantly
    if (batch.worldCount > cachedBatch_.worldCount ||
        batch.maxPlayers > cachedBatch_.maxPlayers ||
        batch.maxMobs > cachedBatch_.maxMobs ||
        batch.maxBlocks > cachedBatch_.maxBlocks) {
        return true;
    }

    // Check if entity counts exceed cached maximums
    if (maxCounts.activePlayers > cachedCounts_.activePlayers ||
        maxCounts.activeMobs > cachedCounts_.activeMobs ||
        maxCounts.activeBlocks > cachedCounts_.activeBlocks ||
        maxCounts.activePortals > cachedCounts_.activePortals ||
        maxCounts.activeDragons > cachedCounts_.activeDragons ||
        maxCounts.activeCrystals > cachedCounts_.activeCrystals) {
        return true;
    }

    return false;
}

// ============================================================================
// BatchedCommandBuilder
// ============================================================================

BatchedCommandBuilder::BatchedCommandBuilder(VkDevice device, uint32_t maxWorldsPerBatch)
    : device_(device)
    , maxWorldsPerBatch_(maxWorldsPerBatch) {
    pendingWorlds_.reserve(maxWorldsPerBatch * 4);
}

void BatchedCommandBuilder::addWorld(uint32_t worldId, const EntityCounts& counts) {
    pendingWorlds_.push_back(WorldEntry{worldId, counts});
}

void BatchedCommandBuilder::build(std::span<VkCommandBuffer> commandBuffers, CommandRecorder& recorder) {
    if (pendingWorlds_.empty() || commandBuffers.empty()) {
        return;
    }

    uint32_t numBatches = batchCount();
    assert(commandBuffers.size() >= numBatches && "Not enough command buffers for batches");

    uint32_t worldIndex = 0;
    for (uint32_t batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
        uint32_t batchStart = worldIndex;
        uint32_t batchSize = std::min(maxWorldsPerBatch_,
                                      static_cast<uint32_t>(pendingWorlds_.size()) - worldIndex);

        // Aggregate entity counts for this batch
        EntityCounts batchCounts{};
        uint32_t maxPlayers = 0, maxMobs = 0, maxBlocks = 0;

        for (uint32_t i = 0; i < batchSize; ++i) {
            const auto& entry = pendingWorlds_[worldIndex + i];
            batchCounts.activePlayers += entry.counts.activePlayers;
            batchCounts.activeMobs += entry.counts.activeMobs;
            batchCounts.activeBlocks += entry.counts.activeBlocks;
            batchCounts.activePortals += entry.counts.activePortals;
            batchCounts.activeDragons += entry.counts.activeDragons;
            batchCounts.activeCrystals += entry.counts.activeCrystals;

            maxPlayers = std::max(maxPlayers, entry.counts.activePlayers);
            maxMobs = std::max(maxMobs, entry.counts.activeMobs);
            maxBlocks = std::max(maxBlocks, entry.counts.activeBlocks);
        }

        WorldBatch batch{
            .worldOffset = batchStart,
            .worldCount = batchSize,
            .maxPlayers = maxPlayers,
            .maxMobs = maxMobs,
            .maxBlocks = maxBlocks
        };

        // Record to command buffer
        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr
        };

        vkBeginCommandBuffer(commandBuffers[batchIdx], &beginInfo);
        recorder.beginTick(commandBuffers[batchIdx]);
        recorder.recordTick(batch, batchCounts);
        recorder.endTick();
        vkEndCommandBuffer(commandBuffers[batchIdx]);

        worldIndex += batchSize;
    }
}

uint32_t BatchedCommandBuilder::batchCount() const noexcept {
    if (pendingWorlds_.empty()) {
        return 0;
    }
    return (static_cast<uint32_t>(pendingWorlds_.size()) + maxWorldsPerBatch_ - 1) / maxWorldsPerBatch_;
}

void BatchedCommandBuilder::clear() noexcept {
    pendingWorlds_.clear();
}

} // namespace mcsim::vulkan
