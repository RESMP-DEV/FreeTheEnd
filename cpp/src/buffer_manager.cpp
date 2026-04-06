#include "mc189/buffer_manager.h"
#include "mc189/vulkan_context.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace mc189 {

// Buffer implementation

Buffer::Buffer(const VulkanContext& ctx, vk::DeviceSize size, BufferUsage usage, MemoryLocation loc)
    : ctx_(&ctx), size_(size), location_(loc) {
    // Convert to Vulkan flags
    vk::BufferUsageFlags vk_usage{};
    if (usage & BufferUsage::Storage) {
        vk_usage |= vk::BufferUsageFlagBits::eStorageBuffer;
    }
    if (usage & BufferUsage::Uniform) {
        vk_usage |= vk::BufferUsageFlagBits::eUniformBuffer;
    }
    if (usage & BufferUsage::TransferSrc) {
        vk_usage |= vk::BufferUsageFlagBits::eTransferSrc;
    }
    if (usage & BufferUsage::TransferDst) {
        vk_usage |= vk::BufferUsageFlagBits::eTransferDst;
    }
    if (usage & BufferUsage::Indirect) {
        vk_usage |= vk::BufferUsageFlagBits::eIndirectBuffer;
    }

    vk::BufferCreateInfo buffer_info{};
    buffer_info.size = size;
    buffer_info.usage = vk_usage;
    buffer_info.sharingMode = vk::SharingMode::eExclusive;

    buffer_ = ctx_->device().createBuffer(buffer_info);

    // Get memory requirements
    auto mem_reqs = ctx_->device().getBufferMemoryRequirements(buffer_);

    // Determine memory properties based on location
    vk::MemoryPropertyFlags mem_props{};
    switch (loc) {
        case MemoryLocation::Device:
            mem_props = vk::MemoryPropertyFlagBits::eDeviceLocal;
            break;
        case MemoryLocation::Host:
            mem_props = vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent;
            break;
        case MemoryLocation::Staging:
            mem_props = vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent |
                        vk::MemoryPropertyFlagBits::eHostCached;
            break;
    }

    // Allocate memory
    vk::MemoryAllocateInfo alloc_info{};
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = ctx_->find_memory_type(mem_reqs.memoryTypeBits, mem_props);

    memory_ = ctx_->device().allocateMemory(alloc_info);
    ctx_->device().bindBufferMemory(buffer_, memory_, 0);

    // Persistently map host-visible memory
    if (loc != MemoryLocation::Device) {
        mapped_ = ctx_->device().mapMemory(memory_, 0, size);
    }
}

Buffer::~Buffer() {
    if (ctx_ && buffer_) {
        if (mapped_) {
            ctx_->device().unmapMemory(memory_);
        }
        ctx_->device().destroyBuffer(buffer_);
        ctx_->device().freeMemory(memory_);
    }
}

Buffer::Buffer(Buffer&& other) noexcept
    : ctx_(other.ctx_),
      buffer_(other.buffer_),
      memory_(other.memory_),
      size_(other.size_),
      mapped_(other.mapped_),
      location_(other.location_) {
    other.ctx_ = nullptr;
    other.buffer_ = nullptr;
    other.memory_ = nullptr;
    other.size_ = 0;
    other.mapped_ = nullptr;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        if (ctx_ && buffer_) {
            if (mapped_) {
                ctx_->device().unmapMemory(memory_);
            }
            ctx_->device().destroyBuffer(buffer_);
            ctx_->device().freeMemory(memory_);
        }

        ctx_ = other.ctx_;
        buffer_ = other.buffer_;
        memory_ = other.memory_;
        size_ = other.size_;
        mapped_ = other.mapped_;
        location_ = other.location_;

        other.ctx_ = nullptr;
        other.buffer_ = nullptr;
        other.memory_ = nullptr;
        other.size_ = 0;
        other.mapped_ = nullptr;
    }
    return *this;
}

void* Buffer::map() {
    if (location_ == MemoryLocation::Device) {
        throw std::runtime_error("Cannot map device-local buffer");
    }
    if (!mapped_) {
        mapped_ = ctx_->device().mapMemory(memory_, 0, size_);
    }
    return mapped_;
}

void Buffer::unmap() {
    if (mapped_) {
        ctx_->device().unmapMemory(memory_);
        mapped_ = nullptr;
    }
}

void Buffer::flush(vk::DeviceSize offset, vk::DeviceSize size) {
    vk::MappedMemoryRange range{};
    range.memory = memory_;
    range.offset = offset;
    range.size = size;
    ctx_->device().flushMappedMemoryRanges(range);
}

void Buffer::invalidate(vk::DeviceSize offset, vk::DeviceSize size) {
    vk::MappedMemoryRange range{};
    range.memory = memory_;
    range.offset = offset;
    range.size = size;
    ctx_->device().invalidateMappedMemoryRanges(range);
}

// BufferManager implementation

BufferManager::BufferManager(const VulkanContext& ctx) : ctx_(&ctx) {}

BufferManager::~BufferManager() = default;

Buffer BufferManager::create_buffer(vk::DeviceSize size, BufferUsage usage, MemoryLocation loc) {
    Buffer buf(*ctx_, size, usage, loc);
    total_allocated_ += size;
    if (loc == MemoryLocation::Device) {
        device_allocated_ += size;
    } else {
        host_allocated_ += size;
    }
    return buf;
}

Buffer BufferManager::create_device_buffer(vk::DeviceSize size, BufferUsage usage) {
    return create_buffer(
        size,
        usage | BufferUsage::TransferDst | BufferUsage::TransferSrc,
        MemoryLocation::Device);
}

Buffer BufferManager::create_staging_buffer(vk::DeviceSize size) {
    return create_buffer(
        size,
        BufferUsage::TransferSrc | BufferUsage::TransferDst,
        MemoryLocation::Staging);
}

Buffer BufferManager::create_mapped_buffer(vk::DeviceSize size, BufferUsage usage) {
    auto buf = create_buffer(
        size,
        usage | BufferUsage::TransferSrc | BufferUsage::TransferDst,
        MemoryLocation::Host);
    buf.map();
    return buf;
}

void BufferManager::copy_buffer(
    vk::CommandBuffer cmd,
    const Buffer& src,
    Buffer& dst,
    vk::DeviceSize src_offset,
    vk::DeviceSize dst_offset,
    vk::DeviceSize size) {
    if (size == VK_WHOLE_SIZE) {
        size = std::min(src.size() - src_offset, dst.size() - dst_offset);
    }

    vk::BufferCopy region{};
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = size;

    cmd.copyBuffer(src.handle(), dst.handle(), region);
}

void BufferManager::copy_buffer_sync(
    const Buffer& src,
    Buffer& dst,
    vk::DeviceSize src_offset,
    vk::DeviceSize dst_offset,
    vk::DeviceSize size) {
    auto cmd = ctx_->allocate_command_buffer();

    vk::CommandBufferBeginInfo begin_info{};
    begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cmd.begin(begin_info);

    copy_buffer(cmd, src, dst, src_offset, dst_offset, size);

    cmd.end();
    ctx_->submit_and_wait(cmd);
    ctx_->free_command_buffer(cmd);
}

void BufferManager::upload(
    Buffer& dst, const void* data, vk::DeviceSize size, vk::DeviceSize offset) {
    if (dst.location() != MemoryLocation::Device) {
        // Direct copy for host-visible buffers
        std::memcpy(static_cast<char*>(dst.data<void>()) + offset, data, size);
        return;
    }

    // Need staging buffer for device-local
    if (staging_size_ < size) {
        staging_buffer_ = create_staging_buffer(size);
        staging_size_ = size;
    }

    std::memcpy(staging_buffer_.data<void>(), data, size);
    copy_buffer_sync(staging_buffer_, dst, 0, offset, size);
}

void BufferManager::download(
    const Buffer& src, void* data, vk::DeviceSize size, vk::DeviceSize offset) {
    if (src.location() != MemoryLocation::Device) {
        // Direct copy for host-visible buffers
        std::memcpy(data, static_cast<const char*>(src.data<void>()) + offset, size);
        return;
    }

    // Need staging buffer for device-local
    if (staging_size_ < size) {
        staging_buffer_ = create_staging_buffer(size);
        staging_size_ = size;
    }

    // Can't use copy_buffer_sync because src is const, do it manually
    auto cmd = ctx_->allocate_command_buffer();

    vk::CommandBufferBeginInfo begin_info{};
    begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cmd.begin(begin_info);

    vk::BufferCopy region{};
    region.srcOffset = offset;
    region.dstOffset = 0;
    region.size = size;
    cmd.copyBuffer(src.handle(), staging_buffer_.handle(), region);

    cmd.end();
    ctx_->submit_and_wait(cmd);
    ctx_->free_command_buffer(cmd);

    std::memcpy(data, staging_buffer_.data<void>(), size);
}

void BufferManager::barrier_compute_to_compute(vk::CommandBuffer cmd, const Buffer& buffer) {
    vk::BufferMemoryBarrier barrier{};
    barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer.handle();
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        {},
        {},
        barrier,
        {});
}

void BufferManager::barrier_transfer_to_compute(vk::CommandBuffer cmd, const Buffer& buffer) {
    vk::BufferMemoryBarrier barrier{};
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer.handle();
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eComputeShader,
        {},
        {},
        barrier,
        {});
}

void BufferManager::barrier_compute_to_transfer(vk::CommandBuffer cmd, const Buffer& buffer) {
    vk::BufferMemoryBarrier barrier{};
    barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer.handle();
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eTransfer,
        {},
        {},
        barrier,
        {});
}

vk::BufferUsageFlags BufferManager::to_vulkan_usage(BufferUsage usage) {
    vk::BufferUsageFlags flags{};
    if (usage & BufferUsage::Storage) flags |= vk::BufferUsageFlagBits::eStorageBuffer;
    if (usage & BufferUsage::Uniform) flags |= vk::BufferUsageFlagBits::eUniformBuffer;
    if (usage & BufferUsage::TransferSrc) flags |= vk::BufferUsageFlagBits::eTransferSrc;
    if (usage & BufferUsage::TransferDst) flags |= vk::BufferUsageFlagBits::eTransferDst;
    if (usage & BufferUsage::Indirect) flags |= vk::BufferUsageFlagBits::eIndirectBuffer;
    return flags;
}

vk::MemoryPropertyFlags BufferManager::to_vulkan_properties(MemoryLocation loc) {
    switch (loc) {
        case MemoryLocation::Device:
            return vk::MemoryPropertyFlagBits::eDeviceLocal;
        case MemoryLocation::Host:
            return vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent;
        case MemoryLocation::Staging:
            return vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent |
                   vk::MemoryPropertyFlagBits::eHostCached;
    }
    return {};
}

// RingBuffer implementation

RingBuffer::RingBuffer(const VulkanContext& ctx, vk::DeviceSize size)
    : buffer_(ctx, size,
              BufferUsage::Storage | BufferUsage::Uniform | BufferUsage::TransferSrc,
              MemoryLocation::Host) {
    buffer_.map();
}

RingBuffer::Allocation RingBuffer::allocate(vk::DeviceSize size, vk::DeviceSize alignment) {
    // Align head
    vk::DeviceSize aligned_head = (head_ + alignment - 1) & ~(alignment - 1);

    if (aligned_head + size > buffer_.size()) {
        throw std::runtime_error("Ring buffer overflow");
    }

    Allocation alloc{};
    alloc.offset = aligned_head;
    alloc.size = size;
    alloc.data = static_cast<char*>(buffer_.data<void>()) + aligned_head;

    head_ = aligned_head + size;
    return alloc;
}

void RingBuffer::reset() {
    head_ = 0;
}

// EnvironmentBufferPool implementation

EnvironmentBufferPool::EnvironmentBufferPool(
    const VulkanContext& ctx, BufferManager& mgr, const Config& config)
    : config_(config) {
    uint32_t num_buffers = config.double_buffer ? 2 : 1;

    for (uint32_t i = 0; i < num_buffers; i++) {
        state_buffers_.push_back(mgr.create_device_buffer(
            static_cast<vk::DeviceSize>(config.max_environments) * config.state_size_per_env));

        action_buffers_.push_back(mgr.create_device_buffer(
            static_cast<vk::DeviceSize>(config.max_environments) * config.action_size_per_env));

        reward_buffers_.push_back(mgr.create_device_buffer(
            static_cast<vk::DeviceSize>(config.max_environments) * config.reward_size_per_env));
    }

    // Done flags only need single buffer (uint8 per env)
    done_buffer_ = mgr.create_device_buffer(config.max_environments);
}

}  // namespace mc189
