/*
 * buffer_manager.cpp - Vulkan buffer management implementation
 */

#include "buffer_manager.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cassert>

namespace mc189 {

// -----------------------------------------------------------------------------
// StagingPool Implementation
// -----------------------------------------------------------------------------

Buffer* StagingPool::acquire(VkDeviceSize size, uint64_t current_frame) {
    // First try to find a free buffer of sufficient size
    for (auto& sb : upload_buffers) {
        if (!sb.in_use && sb.buffer.size >= size) {
            sb.in_use = true;
            sb.last_used_frame = current_frame;
            return &sb.buffer;
        }
    }
    return nullptr;  // Caller must create a new staging buffer
}

void StagingPool::release(Buffer* buffer, uint64_t completed_frame) {
    for (auto& sb : upload_buffers) {
        if (&sb.buffer == buffer) {
            sb.in_use = false;
            sb.last_used_frame = completed_frame;
            return;
        }
    }
    for (auto& sb : readback_buffers) {
        if (&sb.buffer == buffer) {
            sb.in_use = false;
            sb.last_used_frame = completed_frame;
            return;
        }
    }
}

void StagingPool::gc(VkDevice device, uint64_t safe_frame) {
    auto cleanup = [&](std::vector<StagingBuffer>& buffers) {
        buffers.erase(
            std::remove_if(buffers.begin(), buffers.end(),
                [&](StagingBuffer& sb) {
                    if (!sb.in_use && sb.last_used_frame < safe_frame) {
                        if (sb.buffer.handle != VK_NULL_HANDLE) {
                            vkDestroyBuffer(device, sb.buffer.handle, nullptr);
                        }
                        if (sb.buffer.memory != VK_NULL_HANDLE) {
                            vkFreeMemory(device, sb.buffer.memory, nullptr);
                        }
                        return true;
                    }
                    return false;
                }),
            buffers.end());
    };
    cleanup(upload_buffers);
    cleanup(readback_buffers);
}

void StagingPool::clear() {
    upload_buffers.clear();
    readback_buffers.clear();
}

// -----------------------------------------------------------------------------
// BufferManager Implementation
// -----------------------------------------------------------------------------

BufferManager::~BufferManager() {
    destroy();
}

BufferManager::BufferManager(BufferManager&& other) noexcept
    : device_(other.device_)
    , physical_device_(other.physical_device_)
    , memory_properties_(other.memory_properties_)
    , config_(other.config_)
    , game_buffers_(std::move(other.game_buffers_))
    , staging_pool_(std::move(other.staging_pool_))
    , current_frame_(other.current_frame_)
    , frame_number_(other.frame_number_)
    , all_buffers_(std::move(other.all_buffers_))
{
    other.device_ = VK_NULL_HANDLE;
    other.physical_device_ = VK_NULL_HANDLE;
}

BufferManager& BufferManager::operator=(BufferManager&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = other.device_;
        physical_device_ = other.physical_device_;
        memory_properties_ = other.memory_properties_;
        config_ = other.config_;
        game_buffers_ = std::move(other.game_buffers_);
        staging_pool_ = std::move(other.staging_pool_);
        current_frame_ = other.current_frame_;
        frame_number_ = other.frame_number_;
        all_buffers_ = std::move(other.all_buffers_);
        other.device_ = VK_NULL_HANDLE;
        other.physical_device_ = VK_NULL_HANDLE;
    }
    return *this;
}

bool BufferManager::initialize(VkDevice device, VkPhysicalDevice physical_device,
                               const Config& config) {
    device_ = device;
    physical_device_ = physical_device;
    config_ = config;

    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);

    if (!create_game_buffers()) {
        return false;
    }

    if (config_.enable_readback && !create_staging_buffers()) {
        return false;
    }

    return true;
}

void BufferManager::destroy() {
    if (device_ == VK_NULL_HANDLE) return;

    // Wait for device idle before cleanup
    vkDeviceWaitIdle(device_);

    // Helper to destroy a buffer
    auto destroy_buf = [this](Buffer& buf) {
        if (buf.handle != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buf.handle, nullptr);
            buf.handle = VK_NULL_HANDLE;
        }
        if (buf.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, buf.memory, nullptr);
            buf.memory = VK_NULL_HANDLE;
        }
        buf.mapped = nullptr;
    };

    // Destroy ring buffers
    auto destroy_ring = [&](RingBuffer& ring) {
        for (auto& buf : ring.frames) {
            destroy_buf(buf);
        }
        ring.frames.clear();
    };

    // Core simulation
    destroy_ring(game_buffers_.player);
    destroy_ring(game_buffers_.input);
    destroy_buf(game_buffers_.mobs);
    destroy_buf(game_buffers_.chunks);

    // Update queues
    destroy_ring(game_buffers_.block_updates);
    destroy_ring(game_buffers_.combat_events);
    destroy_ring(game_buffers_.dragon_fight);

    // Indirect and state
    destroy_ring(game_buffers_.indirect);
    destroy_ring(game_buffers_.game_state);

    // Inventory
    destroy_ring(game_buffers_.inventory);

    // Dragon combat
    destroy_ring(game_buffers_.dragon_state);
    destroy_ring(game_buffers_.crystal_state);
    destroy_ring(game_buffers_.crystal_visuals);

    // Events
    destroy_ring(game_buffers_.events);

    // Staging
    destroy_buf(game_buffers_.player_staging);
    destroy_buf(game_buffers_.input_staging);
    destroy_buf(game_buffers_.game_state_staging);
    destroy_buf(game_buffers_.block_updates_staging);
    destroy_buf(game_buffers_.inventory_staging);
    destroy_buf(game_buffers_.events_readback);
    destroy_buf(game_buffers_.mobs_readback);

    // Staging pool
    for (auto& sb : staging_pool_.upload_buffers) {
        destroy_buf(sb.buffer);
    }
    for (auto& sb : staging_pool_.readback_buffers) {
        destroy_buf(sb.buffer);
    }
    staging_pool_.clear();

    all_buffers_.clear();
    device_ = VK_NULL_HANDLE;
}

void BufferManager::advance_frame() {
    current_frame_ = (current_frame_ + 1) % config_.frames_in_flight;
    frame_number_++;

    // Advance all ring buffers
    game_buffers_.player.advance();
    game_buffers_.input.advance();
    game_buffers_.block_updates.advance();
    game_buffers_.combat_events.advance();
    game_buffers_.dragon_fight.advance();
    game_buffers_.indirect.advance();
    game_buffers_.game_state.advance();
    game_buffers_.inventory.advance();
    game_buffers_.dragon_state.advance();
    game_buffers_.crystal_state.advance();
    game_buffers_.crystal_visuals.advance();
    game_buffers_.events.advance();

    // GC staging pool - keep buffers from last N frames
    uint64_t safe_frame = (frame_number_ > config_.frames_in_flight + 2)
                        ? frame_number_ - config_.frames_in_flight - 2
                        : 0;
    staging_pool_.gc(device_, safe_frame);
}

uint32_t BufferManager::find_memory_type(uint32_t type_filter,
                                          VkMemoryPropertyFlags properties) const {
    for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (memory_properties_.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

VkMemoryPropertyFlags BufferManager::get_memory_properties(MemoryUsage usage) {
    switch (usage) {
        case MemoryUsage::GPU_ONLY:
            return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        case MemoryUsage::CPU_TO_GPU:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        case MemoryUsage::GPU_TO_CPU:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                   VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        case MemoryUsage::CPU_ONLY:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        default:
            return 0;
    }
}

bool BufferManager::allocate_buffer_memory(Buffer& buffer, VkMemoryPropertyFlags properties) {
    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(device_, buffer.handle, &mem_req);

    buffer.alignment = mem_req.alignment;

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_req.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_req.memoryTypeBits, properties);

    if (alloc_info.memoryTypeIndex == UINT32_MAX) {
        // Try without cached bit for GPU_TO_CPU
        if (properties & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
            properties &= ~VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            alloc_info.memoryTypeIndex = find_memory_type(mem_req.memoryTypeBits, properties);
        }
        if (alloc_info.memoryTypeIndex == UINT32_MAX) {
            return false;
        }
    }

    if (vkAllocateMemory(device_, &alloc_info, nullptr, &buffer.memory) != VK_SUCCESS) {
        return false;
    }

    if (vkBindBufferMemory(device_, buffer.handle, buffer.memory, 0) != VK_SUCCESS) {
        vkFreeMemory(device_, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
        return false;
    }

    // Map persistent for host-visible memory
    if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        if (vkMapMemory(device_, buffer.memory, 0, buffer.size, 0, &buffer.mapped) != VK_SUCCESS) {
            buffer.mapped = nullptr;
        }
    }

    return true;
}

Buffer BufferManager::create_buffer(VkDeviceSize size, BufferUsage usage, MemoryUsage memory,
                                    const std::string& debug_name) {
    Buffer buffer{};
    buffer.size = size;
    buffer.memory_usage = memory;
    buffer.buffer_usage = usage;
    buffer.debug_name = debug_name;

    VkBufferCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_info.size = size;
    create_info.usage = static_cast<VkBufferUsageFlags>(usage);
    create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &create_info, nullptr, &buffer.handle) != VK_SUCCESS) {
        return buffer;
    }

    if (!allocate_buffer_memory(buffer, get_memory_properties(memory))) {
        vkDestroyBuffer(device_, buffer.handle, nullptr);
        buffer.handle = VK_NULL_HANDLE;
        return buffer;
    }

    return buffer;
}

RingBuffer BufferManager::create_ring_buffer(VkDeviceSize size, BufferUsage usage,
                                              MemoryUsage memory, const std::string& debug_name) {
    RingBuffer ring{};
    ring.frame_count = config_.frames_in_flight;
    ring.frames.reserve(ring.frame_count);

    for (uint32_t i = 0; i < ring.frame_count; i++) {
        std::string frame_name = debug_name + "[" + std::to_string(i) + "]";
        ring.frames.push_back(create_buffer(size, usage, memory, frame_name));

        if (ring.frames.back().handle == VK_NULL_HANDLE) {
            // Cleanup already created buffers
            for (auto& buf : ring.frames) {
                destroy_buffer(buf);
            }
            ring.frames.clear();
            return ring;
        }
    }

    return ring;
}

void BufferManager::destroy_buffer(Buffer& buffer) {
    if (device_ == VK_NULL_HANDLE) return;

    if (buffer.handle != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, buffer.handle, nullptr);
        buffer.handle = VK_NULL_HANDLE;
    }
    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_, buffer.memory, nullptr);
        buffer.memory = VK_NULL_HANDLE;
    }
    buffer.mapped = nullptr;
}

void BufferManager::destroy_ring_buffer(RingBuffer& ring) {
    for (auto& buf : ring.frames) {
        destroy_buffer(buf);
    }
    ring.frames.clear();
    ring.frame_count = 0;
}

bool BufferManager::create_game_buffers() {
    const auto storage_gpu = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
    const auto storage_ring = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST |
                              BufferUsage::TRANSFER_SRC;

    // Core simulation (Set 0)
    game_buffers_.player = create_ring_buffer(
        PLAYER_BUFFER_SIZE, storage_ring, MemoryUsage::GPU_ONLY, "PlayerBuffer");
    if (game_buffers_.player.frames.empty()) return false;

    game_buffers_.input = create_ring_buffer(
        INPUT_BUFFER_SIZE, storage_ring, MemoryUsage::GPU_ONLY, "InputBuffer");
    if (game_buffers_.input.frames.empty()) return false;

    game_buffers_.mobs = create_buffer(
        MOB_BUFFER_SIZE, storage_gpu, MemoryUsage::GPU_ONLY, "MobBuffer");
    if (game_buffers_.mobs.handle == VK_NULL_HANDLE) return false;

    game_buffers_.chunks = create_buffer(
        CHUNK_BUFFER_SIZE, storage_gpu, MemoryUsage::GPU_ONLY, "ChunkBuffer");
    if (game_buffers_.chunks.handle == VK_NULL_HANDLE) return false;

    // Update queues (Set 1)
    game_buffers_.block_updates = create_ring_buffer(
        BLOCK_UPDATE_QUEUE_SIZE, storage_ring, MemoryUsage::GPU_ONLY, "BlockUpdateQueue");
    if (game_buffers_.block_updates.frames.empty()) return false;

    game_buffers_.combat_events = create_ring_buffer(
        COMBAT_QUEUE_SIZE, storage_ring, MemoryUsage::GPU_ONLY, "CombatQueue");
    if (game_buffers_.combat_events.frames.empty()) return false;

    game_buffers_.dragon_fight = create_ring_buffer(
        DRAGON_FIGHT_BUFFER_SIZE, storage_ring, MemoryUsage::GPU_ONLY, "DragonFightBuffer");
    if (game_buffers_.dragon_fight.frames.empty()) return false;

    // Indirect and state (Set 2)
    const auto indirect_usage = BufferUsage::STORAGE | BufferUsage::INDIRECT |
                                 BufferUsage::TRANSFER_DST;
    game_buffers_.indirect = create_ring_buffer(
        INDIRECT_BUFFER_SIZE, indirect_usage, MemoryUsage::GPU_ONLY, "IndirectBuffer");
    if (game_buffers_.indirect.frames.empty()) return false;

    game_buffers_.game_state = create_ring_buffer(
        GAME_STATE_BUFFER_SIZE, storage_ring, MemoryUsage::GPU_ONLY, "GameStateBuffer");
    if (game_buffers_.game_state.frames.empty()) return false;

    // Inventory (Set 3)
    game_buffers_.inventory = create_ring_buffer(
        INVENTORY_BUFFER_SIZE, storage_ring, MemoryUsage::GPU_ONLY, "InventoryBuffer");
    if (game_buffers_.inventory.frames.empty()) return false;

    // Dragon combat
    game_buffers_.dragon_state = create_ring_buffer(
        DRAGON_STATE_BUFFER_SIZE, storage_ring, MemoryUsage::GPU_ONLY, "DragonStateBuffer");
    if (game_buffers_.dragon_state.frames.empty()) return false;

    game_buffers_.crystal_state = create_ring_buffer(
        CRYSTAL_STATE_BUFFER_SIZE, storage_ring, MemoryUsage::GPU_ONLY, "CrystalStateBuffer");
    if (game_buffers_.crystal_state.frames.empty()) return false;

    game_buffers_.crystal_visuals = create_ring_buffer(
        CRYSTAL_VISUALS_BUFFER_SIZE, storage_ring, MemoryUsage::GPU_ONLY, "CrystalVisualsBuffer");
    if (game_buffers_.crystal_visuals.frames.empty()) return false;

    // Events
    game_buffers_.events = create_ring_buffer(
        EVENT_BUFFER_SIZE, storage_ring, MemoryUsage::GPU_ONLY, "EventBuffer");
    if (game_buffers_.events.frames.empty()) return false;

    return true;
}

bool BufferManager::create_staging_buffers() {
    const auto staging_upload = BufferUsage::TRANSFER_SRC;
    const auto staging_readback = BufferUsage::TRANSFER_DST;

    // Upload staging buffers (persistently mapped)
    game_buffers_.player_staging = create_buffer(
        PLAYER_BUFFER_SIZE, staging_upload, MemoryUsage::CPU_TO_GPU, "PlayerStaging");
    if (game_buffers_.player_staging.handle == VK_NULL_HANDLE) return false;

    game_buffers_.input_staging = create_buffer(
        INPUT_BUFFER_SIZE, staging_upload, MemoryUsage::CPU_TO_GPU, "InputStaging");
    if (game_buffers_.input_staging.handle == VK_NULL_HANDLE) return false;

    game_buffers_.game_state_staging = create_buffer(
        GAME_STATE_BUFFER_SIZE, staging_upload, MemoryUsage::CPU_TO_GPU, "GameStateStaging");
    if (game_buffers_.game_state_staging.handle == VK_NULL_HANDLE) return false;

    game_buffers_.block_updates_staging = create_buffer(
        BLOCK_UPDATE_QUEUE_SIZE, staging_upload, MemoryUsage::CPU_TO_GPU, "BlockUpdatesStaging");
    if (game_buffers_.block_updates_staging.handle == VK_NULL_HANDLE) return false;

    game_buffers_.inventory_staging = create_buffer(
        INVENTORY_BUFFER_SIZE, staging_upload, MemoryUsage::CPU_TO_GPU, "InventoryStaging");
    if (game_buffers_.inventory_staging.handle == VK_NULL_HANDLE) return false;

    // Readback staging buffers
    game_buffers_.events_readback = create_buffer(
        EVENT_BUFFER_SIZE, staging_readback, MemoryUsage::GPU_TO_CPU, "EventsReadback");
    if (game_buffers_.events_readback.handle == VK_NULL_HANDLE) return false;

    // Optional large readback buffer for mobs
    game_buffers_.mobs_readback = create_buffer(
        sizeof(MobBufferHeader) + sizeof(Mob) * 1024,  // Only first 1024 mobs
        staging_readback, MemoryUsage::GPU_TO_CPU, "MobsReadback");
    // Not fatal if this fails

    return true;
}

void BufferManager::upload_to_buffer(VkCommandBuffer cmd, const Buffer& dst,
                                     const void* data, VkDeviceSize size,
                                     VkDeviceSize offset) {
    // Find or create a staging buffer
    Buffer* staging = staging_pool_.acquire(size, frame_number_);

    if (!staging) {
        // Create new staging buffer
        StagingPool::StagingBuffer sb;
        sb.buffer = create_buffer(size, BufferUsage::TRANSFER_SRC,
                                  MemoryUsage::CPU_TO_GPU, "DynamicStaging");
        sb.in_use = true;
        sb.last_used_frame = frame_number_;

        if (sb.buffer.handle == VK_NULL_HANDLE) {
            return;  // Allocation failed
        }

        staging_pool_.upload_buffers.push_back(std::move(sb));
        staging = &staging_pool_.upload_buffers.back().buffer;
    }

    // Copy data to staging buffer
    if (staging->mapped) {
        memcpy(staging->mapped, data, size);
    }

    // Record copy command
    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = offset;
    copy_region.size = size;

    vkCmdCopyBuffer(cmd, staging->handle, dst.handle, 1, &copy_region);
}

void BufferManager::readback_from_buffer(VkCommandBuffer cmd, const Buffer& src,
                                          Buffer& dst_staging, VkDeviceSize size,
                                          VkDeviceSize src_offset) {
    VkBufferCopy copy_region{};
    copy_region.srcOffset = src_offset;
    copy_region.dstOffset = 0;
    copy_region.size = size;

    vkCmdCopyBuffer(cmd, src.handle, dst_staging.handle, 1, &copy_region);
}

std::vector<BufferBinding> BufferManager::get_set0_bindings(uint32_t frame) const {
    return {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, game_buffers_.player[frame].handle,
         0, VK_WHOLE_SIZE, 0},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, game_buffers_.input[frame].handle,
         0, VK_WHOLE_SIZE, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, game_buffers_.mobs.handle,
         0, VK_WHOLE_SIZE, 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, game_buffers_.chunks.handle,
         0, VK_WHOLE_SIZE, 3},
    };
}

std::vector<BufferBinding> BufferManager::get_set1_bindings(uint32_t frame) const {
    return {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, game_buffers_.block_updates[frame].handle,
         0, VK_WHOLE_SIZE, 0},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, game_buffers_.combat_events[frame].handle,
         0, VK_WHOLE_SIZE, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, game_buffers_.dragon_fight[frame].handle,
         0, VK_WHOLE_SIZE, 2},
    };
}

std::vector<BufferBinding> BufferManager::get_set2_bindings(uint32_t frame) const {
    return {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, game_buffers_.indirect[frame].handle,
         0, VK_WHOLE_SIZE, 0},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, game_buffers_.game_state[frame].handle,
         0, VK_WHOLE_SIZE, 1},
    };
}

std::vector<BufferBinding> BufferManager::get_set3_bindings(uint32_t frame) const {
    return {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, game_buffers_.inventory[frame].handle,
         0, VK_WHOLE_SIZE, 0},
    };
}

void BufferManager::upload_player(VkCommandBuffer cmd, const Player& player) {
    game_buffers_.player_staging.write(player);

    VkBufferCopy copy{};
    copy.size = sizeof(Player);
    vkCmdCopyBuffer(cmd, game_buffers_.player_staging.handle,
                    game_buffers_.player.current().handle, 1, &copy);
}

void BufferManager::upload_input(VkCommandBuffer cmd, const InputState& input) {
    game_buffers_.input_staging.write(input);

    VkBufferCopy copy{};
    copy.size = sizeof(InputState);
    vkCmdCopyBuffer(cmd, game_buffers_.input_staging.handle,
                    game_buffers_.input.current().handle, 1, &copy);
}

void BufferManager::upload_game_state(VkCommandBuffer cmd, const GameState& state) {
    game_buffers_.game_state_staging.write(state);

    VkBufferCopy copy{};
    copy.size = sizeof(GameState);
    vkCmdCopyBuffer(cmd, game_buffers_.game_state_staging.handle,
                    game_buffers_.game_state.current().handle, 1, &copy);
}

void BufferManager::upload_block_updates(VkCommandBuffer cmd,
                                         const BlockUpdateQueueHeader& header,
                                         std::span<const BlockUpdate> updates) {
    if (!game_buffers_.block_updates_staging.mapped) {
        return;
    }

    auto* dst = static_cast<uint8_t*>(game_buffers_.block_updates_staging.mapped);
    std::memcpy(dst, &header, sizeof(BlockUpdateQueueHeader));

    VkDeviceSize update_bytes = updates.size_bytes();
    VkDeviceSize max_bytes = BLOCK_UPDATE_QUEUE_SIZE - sizeof(BlockUpdateQueueHeader);
    if (update_bytes > max_bytes) {
        update_bytes = max_bytes;
    }

    if (update_bytes > 0) {
        std::memcpy(dst + sizeof(BlockUpdateQueueHeader), updates.data(), update_bytes);
    }

    VkBufferCopy copy{};
    copy.size = sizeof(BlockUpdateQueueHeader) + update_bytes;
    vkCmdCopyBuffer(cmd, game_buffers_.block_updates_staging.handle,
                    game_buffers_.block_updates.current().handle, 1, &copy);
}

void BufferManager::upload_inventory(VkCommandBuffer cmd, const void* data,
                                     VkDeviceSize size, VkDeviceSize offset) {
    if (!game_buffers_.inventory_staging.mapped || size == 0) {
        return;
    }

    VkDeviceSize max_bytes = INVENTORY_BUFFER_SIZE - offset;
    VkDeviceSize copy_bytes = (size > max_bytes) ? max_bytes : size;
    std::memcpy(static_cast<uint8_t*>(game_buffers_.inventory_staging.mapped) + offset,
                data, copy_bytes);

    VkBufferCopy copy{};
    copy.srcOffset = offset;
    copy.dstOffset = offset;
    copy.size = copy_bytes;
    vkCmdCopyBuffer(cmd, game_buffers_.inventory_staging.handle,
                    game_buffers_.inventory.current().handle, 1, &copy);
}

void BufferManager::readback_events(VkCommandBuffer cmd) {
    readback_from_buffer(cmd, game_buffers_.events.current(),
                         game_buffers_.events_readback, EVENT_BUFFER_SIZE);
}

bool BufferManager::get_events(std::vector<RLEvent>& out_events, uint32_t& out_count) const {
    if (!game_buffers_.events_readback.is_mapped()) {
        return false;
    }

    // Read header
    auto header = game_buffers_.events_readback.read<EventBufferHeader>();
    out_count = std::min(header.event_count, MAX_EVENTS);

    if (out_count == 0) {
        out_events.clear();
        return true;
    }

    out_events.resize(out_count);

    // Read events after header
    const auto* events_ptr = reinterpret_cast<const RLEvent*>(
        static_cast<const uint8_t*>(game_buffers_.events_readback.mapped) +
        sizeof(EventBufferHeader));

    // Handle ring buffer wrap-around
    uint32_t start = header.event_tail;
    for (uint32_t i = 0; i < out_count; i++) {
        uint32_t idx = (start + i) % MAX_EVENTS;
        out_events[i] = events_ptr[idx];
    }

    return true;
}

BufferManager::MemoryStats BufferManager::get_memory_stats() const {
    MemoryStats stats{};

    auto count_buffer = [&](const Buffer& buf) {
        if (buf.handle == VK_NULL_HANDLE) return;
        stats.total_allocated += buf.size;
        stats.buffer_count++;
        if (buf.memory_usage == MemoryUsage::GPU_ONLY) {
            stats.device_local += buf.size;
        } else {
            stats.host_visible += buf.size;
        }
    };

    auto count_ring = [&](const RingBuffer& ring) {
        for (const auto& buf : ring.frames) {
            count_buffer(buf);
        }
    };

    // Count all buffers
    count_ring(game_buffers_.player);
    count_ring(game_buffers_.input);
    count_buffer(game_buffers_.mobs);
    count_buffer(game_buffers_.chunks);
    count_ring(game_buffers_.block_updates);
    count_ring(game_buffers_.combat_events);
    count_ring(game_buffers_.dragon_fight);
    count_ring(game_buffers_.indirect);
    count_ring(game_buffers_.game_state);
    count_ring(game_buffers_.inventory);
    count_ring(game_buffers_.dragon_state);
    count_ring(game_buffers_.crystal_state);
    count_ring(game_buffers_.crystal_visuals);
    count_ring(game_buffers_.events);
    count_buffer(game_buffers_.player_staging);
    count_buffer(game_buffers_.input_staging);
    count_buffer(game_buffers_.game_state_staging);
    count_buffer(game_buffers_.block_updates_staging);
    count_buffer(game_buffers_.inventory_staging);
    count_buffer(game_buffers_.events_readback);
    count_buffer(game_buffers_.mobs_readback);

    return stats;
}

void BufferManager::set_debug_names(VkDevice device) {
#ifdef VK_EXT_debug_utils
    auto set_name = [device](VkBuffer buffer, const std::string& name) {
        if (buffer == VK_NULL_HANDLE || name.empty()) return;

        VkDebugUtilsObjectNameInfoEXT name_info{};
        name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        name_info.objectType = VK_OBJECT_TYPE_BUFFER;
        name_info.objectHandle = reinterpret_cast<uint64_t>(buffer);
        name_info.pObjectName = name.c_str();

        auto vkSetDebugUtilsObjectNameEXT = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
        if (vkSetDebugUtilsObjectNameEXT) {
            vkSetDebugUtilsObjectNameEXT(device, &name_info);
        }
    };

    auto set_ring_names = [&](RingBuffer& ring) {
        for (auto& buf : ring.frames) {
            set_name(buf.handle, buf.debug_name);
        }
    };

    auto set_buf_name = [&](Buffer& buf) {
        set_name(buf.handle, buf.debug_name);
    };

    set_ring_names(game_buffers_.player);
    set_ring_names(game_buffers_.input);
    set_buf_name(game_buffers_.mobs);
    set_buf_name(game_buffers_.chunks);
    set_ring_names(game_buffers_.block_updates);
    set_ring_names(game_buffers_.combat_events);
    set_ring_names(game_buffers_.dragon_fight);
    set_ring_names(game_buffers_.indirect);
    set_ring_names(game_buffers_.game_state);
    set_ring_names(game_buffers_.inventory);
    set_ring_names(game_buffers_.dragon_state);
    set_ring_names(game_buffers_.crystal_state);
    set_ring_names(game_buffers_.crystal_visuals);
    set_ring_names(game_buffers_.events);
    set_buf_name(game_buffers_.player_staging);
    set_buf_name(game_buffers_.input_staging);
    set_buf_name(game_buffers_.game_state_staging);
    set_buf_name(game_buffers_.block_updates_staging);
    set_buf_name(game_buffers_.inventory_staging);
    set_buf_name(game_buffers_.events_readback);
    set_buf_name(game_buffers_.mobs_readback);
#endif
}

} // namespace mc189
