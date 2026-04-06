#pragma once

#include "instance.h"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>
#include <optional>

// Forward declare VMA types if available
struct VmaAllocator_T;
typedef struct VmaAllocator_T* VmaAllocator;

namespace mcsim {

/**
 * VulkanDevice - Manages physical/logical device and compute queue
 *
 * Selects a compute-capable GPU, enables required extensions for shader
 * functionality (int8, float16, 64-bit integers), creates the logical device,
 * and optionally initializes Vulkan Memory Allocator (VMA).
 */
class VulkanDevice {
public:
    /**
     * Required device features for Minecraft simulation shaders.
     * These match the extensions needed by the GLSL shaders.
     */
    struct RequiredFeatures {
        bool shader_int64 = true;          // GL_ARB_gpu_shader_int64 for Java LCG
        bool shader_int16 = false;         // Optional int16
        bool shader_int8 = true;           // VK_KHR_shader_float16_int8
        bool shader_float16 = true;        // VK_KHR_shader_float16_int8
        bool storage_buffer_16bit = false; // 16-bit types in storage buffers
        bool storage_buffer_8bit = true;   // 8-bit types in storage buffers
    };

    struct Config {
        RequiredFeatures features;
        bool prefer_discrete_gpu = true;   // Prefer discrete over integrated
        bool enable_vma = true;            // Use Vulkan Memory Allocator
        uint32_t preferred_device_id = UINT32_MAX;  // Override device selection
    };

    /**
     * Information about device capabilities.
     */
    struct DeviceInfo {
        std::string name;
        uint32_t vendor_id;
        uint32_t device_id;
        VkPhysicalDeviceType type;
        uint32_t api_version;
        uint32_t driver_version;
        uint32_t compute_queue_family;
        uint32_t max_compute_work_group_count[3];
        uint32_t max_compute_work_group_size[3];
        uint32_t max_compute_work_group_invocations;
        VkDeviceSize max_storage_buffer_range;
        VkDeviceSize max_uniform_buffer_range;
        bool supports_shader_int64;
        bool supports_shader_int16;
        bool supports_shader_int8;
        bool supports_shader_float16;
    };

    VulkanDevice() = default;
    ~VulkanDevice();

    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;
    VulkanDevice(VulkanDevice&&) noexcept;
    VulkanDevice& operator=(VulkanDevice&&) noexcept;

    /**
     * Initialize device with the given Vulkan instance.
     * @param instance Valid VulkanInstance (must outlive this device)
     * @param config Device configuration
     * @return true on success, false on failure (check error_message())
     */
    bool init(VulkanInstance& instance, const Config& config = {});

    /**
     * Clean up device resources. Called automatically by destructor.
     */
    void destroy();

    VkDevice handle() const { return device_; }
    VkPhysicalDevice physical_device() const { return physical_device_; }
    VkQueue compute_queue() const { return compute_queue_; }
    uint32_t compute_queue_family() const { return compute_queue_family_; }
    VmaAllocator allocator() const { return allocator_; }

    bool is_valid() const { return device_ != VK_NULL_HANDLE; }
    const std::string& error_message() const { return error_msg_; }
    const DeviceInfo& info() const { return info_; }

    /**
     * Enumerate all physical devices with their capabilities.
     */
    static std::vector<DeviceInfo> enumerate_devices(VkInstance instance);

    /**
     * Check if a physical device supports the required features.
     */
    static bool check_features_support(VkPhysicalDevice device,
                                       const RequiredFeatures& features);

    /**
     * Wait for device to become idle.
     */
    void wait_idle() const;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = nullptr;
    uint32_t compute_queue_family_ = UINT32_MAX;
    DeviceInfo info_{};
    std::string error_msg_;

    std::optional<uint32_t> find_compute_queue_family(VkPhysicalDevice device);
    bool create_logical_device(const Config& config, VkInstance instance);
    bool create_allocator(VkInstance instance);
};

}  // namespace mcsim
