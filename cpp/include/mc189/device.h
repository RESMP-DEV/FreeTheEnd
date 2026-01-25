#pragma once

#include "instance.h"

#include <vulkan/vulkan.h>
#include <optional>
#include <string>
#include <vector>
#include <memory>

// Forward declare VMA types to avoid including vk_mem_alloc.h in header
struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;
struct VmaAllocation_T;
typedef VmaAllocation_T* VmaAllocation;
struct VmaAllocationInfo;

namespace mcsim::vk {

/// Queue family indices for device creation.
struct QueueFamilyIndices {
    std::optional<uint32_t> compute_family;
    std::optional<uint32_t> transfer_family;

    [[nodiscard]] bool is_complete() const noexcept {
        return compute_family.has_value();
    }
};

/// Physical device capabilities relevant to our compute workloads.
struct DeviceCapabilities {
    bool shader_float16_int8 = false;          // VK_KHR_shader_float16_int8
    bool storage_buffer_storage_class = false; // VK_KHR_storage_buffer_storage_class
    bool int64_atomics = false;                // VK_KHR_shader_atomic_int64
    bool shader_int64 = false;                 // shaderInt64 feature

    uint32_t max_compute_work_group_count[3] = {0, 0, 0};
    uint32_t max_compute_work_group_size[3] = {0, 0, 0};
    uint32_t max_compute_work_group_invocations = 0;
    uint32_t max_compute_shared_memory_size = 0;

    uint64_t max_storage_buffer_range = 0;
    uint64_t total_device_memory = 0;

    std::string device_name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
};

/// Configuration for logical device creation.
struct DeviceConfig {
    bool require_float16_int8 = true;          // Required for int8_t types
    bool require_storage_buffer = true;        // Required for SSBO
    bool require_int64 = true;                 // Required for Java LCG
    bool prefer_dedicated_transfer_queue = true;
    bool verbose = false;
};

/// Vulkan logical device wrapper with VMA integration.
/// Manages device creation, queue acquisition, and memory allocation.
class Device {
public:
    /// Create a device using the given instance and configuration.
    /// Automatically selects the best suitable physical device.
    explicit Device(Instance& instance, const DeviceConfig& config = {});

    /// Create a device using a specific physical device.
    Device(Instance& instance, VkPhysicalDevice physical_device,
           const DeviceConfig& config = {});

    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;

    // Accessors

    /// Get the raw Vulkan logical device handle.
    [[nodiscard]] VkDevice handle() const noexcept { return device_; }

    /// Get the physical device handle.
    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return physical_device_; }

    /// Get the VMA allocator for memory management.
    [[nodiscard]] VmaAllocator allocator() const noexcept { return allocator_; }

    /// Get the compute queue.
    [[nodiscard]] VkQueue compute_queue() const noexcept { return compute_queue_; }

    /// Get the compute queue family index.
    [[nodiscard]] uint32_t compute_queue_family() const noexcept {
        return queue_families_.compute_family.value();
    }

    /// Get the transfer queue (may be same as compute queue).
    [[nodiscard]] VkQueue transfer_queue() const noexcept { return transfer_queue_; }

    /// Get the transfer queue family index.
    [[nodiscard]] uint32_t transfer_queue_family() const noexcept {
        return queue_families_.transfer_family.value_or(
            queue_families_.compute_family.value());
    }

    /// Get device capabilities.
    [[nodiscard]] const DeviceCapabilities& capabilities() const noexcept {
        return capabilities_;
    }

    /// Check if we have a dedicated transfer queue.
    [[nodiscard]] bool has_dedicated_transfer_queue() const noexcept {
        return queue_families_.transfer_family.has_value() &&
               queue_families_.transfer_family != queue_families_.compute_family;
    }

    // Convenience functions

    /// Wait for the device to become idle.
    void wait_idle() const;

    /// Create a command pool for the compute queue.
    [[nodiscard]] VkCommandPool create_compute_command_pool(
        VkCommandPoolCreateFlags flags = 0) const;

    /// Create a command pool for the transfer queue.
    [[nodiscard]] VkCommandPool create_transfer_command_pool(
        VkCommandPoolCreateFlags flags = 0) const;

    /// Submit commands to compute queue and wait for completion.
    void submit_compute_and_wait(VkCommandBuffer cmd) const;

    /// Get physical device properties.
    [[nodiscard]] VkPhysicalDeviceProperties get_properties() const;

    /// Get physical device memory properties.
    [[nodiscard]] VkPhysicalDeviceMemoryProperties get_memory_properties() const;

    /// Find a memory type index suitable for the given requirements.
    [[nodiscard]] uint32_t find_memory_type(uint32_t type_filter,
                                            VkMemoryPropertyFlags properties) const;

    // Static helpers

    /// Rate a physical device for compute suitability.
    [[nodiscard]] static int rate_device(VkPhysicalDevice device, const DeviceConfig& config);

    /// Check if a physical device supports required extensions.
    [[nodiscard]] static bool check_device_extension_support(
        VkPhysicalDevice device, const std::vector<const char*>& extensions);

    /// Get queue family indices for a physical device.
    [[nodiscard]] static QueueFamilyIndices find_queue_families(
        VkPhysicalDevice device, bool prefer_dedicated_transfer);

    /// Query device capabilities.
    [[nodiscard]] static DeviceCapabilities query_capabilities(VkPhysicalDevice device);

private:
    void select_physical_device(Instance& instance, const DeviceConfig& config);
    void create_logical_device(const DeviceConfig& config);
    void create_allocator(Instance& instance);
    void cleanup();

    [[nodiscard]] static std::vector<const char*> get_required_extensions(
        VkPhysicalDevice device, const DeviceConfig& config);

    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;

    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkQueue transfer_queue_ = VK_NULL_HANDLE;

    QueueFamilyIndices queue_families_;
    DeviceCapabilities capabilities_;
};

/// RAII wrapper for VkCommandPool.
class CommandPool {
public:
    CommandPool(VkDevice device, VkCommandPool pool);
    ~CommandPool();

    CommandPool(const CommandPool&) = delete;
    CommandPool& operator=(const CommandPool&) = delete;
    CommandPool(CommandPool&&) noexcept;
    CommandPool& operator=(CommandPool&&) noexcept;

    [[nodiscard]] VkCommandPool handle() const noexcept { return pool_; }
    [[nodiscard]] VkDevice device() const noexcept { return device_; }

    /// Allocate a primary command buffer.
    [[nodiscard]] VkCommandBuffer allocate_command_buffer() const;

    /// Free a command buffer.
    void free_command_buffer(VkCommandBuffer cmd) const;

    /// Reset the pool and all allocated command buffers.
    void reset(VkCommandPoolResetFlags flags = 0) const;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
};

/// RAII wrapper for VkFence.
class Fence {
public:
    explicit Fence(VkDevice device, bool signaled = false);
    ~Fence();

    Fence(const Fence&) = delete;
    Fence& operator=(const Fence&) = delete;
    Fence(Fence&&) noexcept;
    Fence& operator=(Fence&&) noexcept;

    [[nodiscard]] VkFence handle() const noexcept { return fence_; }

    /// Wait for the fence to be signaled.
    void wait(uint64_t timeout_ns = UINT64_MAX) const;

    /// Reset the fence to unsignaled state.
    void reset() const;

    /// Check if the fence is signaled without waiting.
    [[nodiscard]] bool is_signaled() const;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

} // namespace mcsim::vk
