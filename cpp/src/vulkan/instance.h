#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

namespace mcsim {

/**
 * VulkanInstance - Manages Vulkan instance lifecycle with MoltenVK support
 *
 * Configures validation layers for debug builds and enables portability
 * enumeration for MoltenVK on macOS. Provides query methods for available
 * extensions and layers.
 */
class VulkanInstance {
public:
    struct Config {
        std::string app_name = "MinecraftSim";
        uint32_t app_version = VK_MAKE_VERSION(1, 0, 0);
        bool enable_validation = false;
        bool verbose_debug = false;
    };

    VulkanInstance() = default;
    ~VulkanInstance();

    VulkanInstance(const VulkanInstance&) = delete;
    VulkanInstance& operator=(const VulkanInstance&) = delete;
    VulkanInstance(VulkanInstance&&) noexcept;
    VulkanInstance& operator=(VulkanInstance&&) noexcept;

    /**
     * Initialize Vulkan instance with given configuration.
     * @return true on success, false on failure (check error_message())
     */
    bool init(const Config& config = {});

    /**
     * Clean up Vulkan resources. Called automatically by destructor.
     */
    void destroy();

    VkInstance handle() const { return instance_; }
    bool is_valid() const { return instance_ != VK_NULL_HANDLE; }
    const std::string& error_message() const { return error_msg_; }

    /**
     * Check if running on MoltenVK (macOS/iOS).
     */
    bool is_moltenvk() const { return is_moltenvk_; }

    /**
     * Query available instance extensions.
     */
    static std::vector<VkExtensionProperties> enumerate_extensions();

    /**
     * Query available validation layers.
     */
    static std::vector<VkLayerProperties> enumerate_layers();

    /**
     * Check if a specific extension is available.
     */
    static bool is_extension_available(const char* name);

    /**
     * Check if a specific validation layer is available.
     */
    static bool is_layer_available(const char* name);

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    std::string error_msg_;
    bool is_moltenvk_ = false;
    bool validation_enabled_ = false;

    bool setup_debug_messenger();
    void destroy_debug_messenger();

    static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT* data,
        void* user_data);
};

}  // namespace mcsim
