#pragma once

#include <memory>
#include <ostream>
#include <span>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace mc189 {

class VulkanContext;

enum class BufferUsage : uint32_t {
  Storage = 1 << 0,     // Shader storage buffer
  Uniform = 1 << 1,     // Uniform buffer
  TransferSrc = 1 << 2, // Can be source of transfer
  TransferDst = 1 << 3, // Can be destination of transfer
  Indirect = 1 << 4,    // Indirect command buffer
};

inline BufferUsage operator|(BufferUsage a, BufferUsage b) {
  return static_cast<BufferUsage>(static_cast<uint32_t>(a) |
                                  static_cast<uint32_t>(b));
}

inline bool operator&(BufferUsage a, BufferUsage b) {
  return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
}

enum class MemoryLocation {
  Device,  // GPU-local, fastest for compute
  Host,    // CPU-visible, slower but mappable
  Staging, // CPU-visible, optimal for transfers
};

inline std::ostream &operator<<(std::ostream &os, MemoryLocation loc) {
  switch (loc) {
  case MemoryLocation::Device:
    return os << "Device";
  case MemoryLocation::Host:
    return os << "Host";
  case MemoryLocation::Staging:
    return os << "Staging";
  }
  return os << "Unknown";
}

struct BufferAllocation {
  vk::Buffer buffer;
  vk::DeviceMemory memory;
  vk::DeviceSize size;
  vk::DeviceSize offset;
  void *mapped = nullptr;
  MemoryLocation location;
};

class Buffer {
public:
  Buffer() = default;
  Buffer(const VulkanContext &ctx, vk::DeviceSize size, BufferUsage usage,
         MemoryLocation loc);
  ~Buffer();

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other) noexcept;
  Buffer &operator=(Buffer &&other) noexcept;

  vk::Buffer handle() const { return buffer_; }
  vk::DeviceSize size() const { return size_; }
  MemoryLocation location() const { return location_; }
  bool is_mapped() const { return mapped_ != nullptr; }

  void *map();
  void unmap();
  void flush(vk::DeviceSize offset = 0, vk::DeviceSize size = VK_WHOLE_SIZE);
  void invalidate(vk::DeviceSize offset = 0,
                  vk::DeviceSize size = VK_WHOLE_SIZE);

  // Direct data access (only if mapped)
  template <typename T> T *data() { return static_cast<T *>(mapped_); }

  template <typename T> const T *data() const {
    return static_cast<const T *>(mapped_);
  }

  template <typename T> std::span<T> as_span() {
    return std::span<T>(data<T>(), size_ / sizeof(T));
  }

  template <typename T> std::span<const T> as_span() const {
    return std::span<const T>(data<T>(), size_ / sizeof(T));
  }

private:
  const VulkanContext *ctx_ = nullptr;
  vk::Buffer buffer_;
  vk::DeviceMemory memory_;
  vk::DeviceSize size_ = 0;
  void *mapped_ = nullptr;
  MemoryLocation location_ = MemoryLocation::Device;
};

class BufferManager {
public:
  explicit BufferManager(const VulkanContext &ctx);
  ~BufferManager();

  BufferManager(const BufferManager &) = delete;
  BufferManager &operator=(const BufferManager &) = delete;

  // Create individual buffers
  Buffer create_buffer(vk::DeviceSize size, BufferUsage usage,
                       MemoryLocation loc);

  // Create device-local storage buffer with staging
  Buffer create_device_buffer(vk::DeviceSize size,
                              BufferUsage usage = BufferUsage::Storage);

  // Create host-visible staging buffer
  Buffer create_staging_buffer(vk::DeviceSize size);

  // Create persistently mapped host buffer
  Buffer create_mapped_buffer(vk::DeviceSize size,
                              BufferUsage usage = BufferUsage::Storage);

  // Transfer operations
  void copy_buffer(vk::CommandBuffer cmd, const Buffer &src, Buffer &dst,
                   vk::DeviceSize src_offset = 0, vk::DeviceSize dst_offset = 0,
                   vk::DeviceSize size = VK_WHOLE_SIZE);

  void copy_buffer_sync(const Buffer &src, Buffer &dst,
                        vk::DeviceSize src_offset = 0,
                        vk::DeviceSize dst_offset = 0,
                        vk::DeviceSize size = VK_WHOLE_SIZE);

  // Upload data to device buffer via staging
  void upload(Buffer &dst, const void *data, vk::DeviceSize size,
              vk::DeviceSize offset = 0);

  template <typename T>
  void upload(Buffer &dst, const std::vector<T> &data,
              vk::DeviceSize offset = 0) {
    upload(dst, data.data(), data.size() * sizeof(T), offset);
  }

  template <typename T>
  void upload(Buffer &dst, std::span<const T> data, vk::DeviceSize offset = 0) {
    upload(dst, data.data(), data.size() * sizeof(T), offset);
  }

  // Download data from device buffer via staging
  void download(const Buffer &src, void *data, vk::DeviceSize size,
                vk::DeviceSize offset = 0);

  template <typename T>
  void download(const Buffer &src, std::vector<T> &data,
                vk::DeviceSize offset = 0) {
    download(src, data.data(), data.size() * sizeof(T), offset);
  }

  // Memory barriers for compute synchronization
  static void barrier_compute_to_compute(vk::CommandBuffer cmd,
                                         const Buffer &buffer);
  static void barrier_transfer_to_compute(vk::CommandBuffer cmd,
                                          const Buffer &buffer);
  static void barrier_compute_to_transfer(vk::CommandBuffer cmd,
                                          const Buffer &buffer);

  // Get stats
  vk::DeviceSize total_allocated() const { return total_allocated_; }
  vk::DeviceSize device_allocated() const { return device_allocated_; }
  vk::DeviceSize host_allocated() const { return host_allocated_; }

private:
  vk::BufferUsageFlags to_vulkan_usage(BufferUsage usage);
  vk::MemoryPropertyFlags to_vulkan_properties(MemoryLocation loc);

  const VulkanContext *ctx_;
  Buffer staging_buffer_; // Reusable staging buffer
  vk::DeviceSize staging_size_ = 0;

  vk::DeviceSize total_allocated_ = 0;
  vk::DeviceSize device_allocated_ = 0;
  vk::DeviceSize host_allocated_ = 0;
};

// Ring buffer for streaming data to GPU
class RingBuffer {
public:
  RingBuffer(const VulkanContext &ctx, vk::DeviceSize size);

  // Allocate space from the ring buffer
  struct Allocation {
    vk::DeviceSize offset;
    vk::DeviceSize size;
    void *data;
  };

  Allocation allocate(vk::DeviceSize size, vk::DeviceSize alignment = 256);
  void reset();

  vk::Buffer handle() const { return buffer_.handle(); }
  vk::DeviceSize capacity() const { return buffer_.size(); }
  vk::DeviceSize used() const { return head_; }
  vk::DeviceSize available() const { return capacity() - head_; }

private:
  Buffer buffer_;
  vk::DeviceSize head_ = 0;
};

// Buffer pool for batch simulation environments
class EnvironmentBufferPool {
public:
  struct Config {
    uint32_t max_environments = 32768;
    uint32_t state_size_per_env = 1024; // bytes
    uint32_t action_size_per_env = 64;
    uint32_t reward_size_per_env = 4; // float
    bool double_buffer = true;        // For async updates
  };

  EnvironmentBufferPool(const VulkanContext &ctx, BufferManager &mgr,
                        const Config &config);

  // Buffer accessors
  Buffer &states(uint32_t buffer_index = 0) {
    return state_buffers_[buffer_index];
  }
  Buffer &actions(uint32_t buffer_index = 0) {
    return action_buffers_[buffer_index];
  }
  Buffer &rewards(uint32_t buffer_index = 0) {
    return reward_buffers_[buffer_index];
  }
  Buffer &dones() { return done_buffer_; }

  // Swap double buffers
  void swap_buffers() {
    if (config_.double_buffer) {
      current_buffer_ = 1 - current_buffer_;
    }
  }

  uint32_t current_buffer_index() const { return current_buffer_; }
  uint32_t next_buffer_index() const {
    return config_.double_buffer ? (1 - current_buffer_) : 0;
  }
  uint32_t max_environments() const { return config_.max_environments; }

private:
  Config config_;
  std::vector<Buffer> state_buffers_;
  std::vector<Buffer> action_buffers_;
  std::vector<Buffer> reward_buffers_;
  Buffer done_buffer_;
  uint32_t current_buffer_ = 0;
};

} // namespace mc189
