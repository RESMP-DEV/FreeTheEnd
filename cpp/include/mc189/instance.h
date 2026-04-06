#pragma once

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace mcsim::vk {

/// Configuration for Vulkan instance creation.
struct InstanceConfig {
    std::string app_name = "MinecraftSim";
    uint32_t app_version = VK_MAKE_VERSION(1, 0, 0);
    bool enable_validation = false;
    bool verbose = false;
};

/// Vulkan instance wrapper with MoltenVK compatibility.
/// Handles instance creation, extension enumeration, and cleanup.
class Instance {
public:
    explicit Instance(const InstanceConfig& config = {});
    ~Instance();

    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) noexcept;
    Instance& operator=(Instance&&) noexcept;

    /// Get the raw Vulkan instance handle.
    [[nodiscard]] VkInstance handle() const noexcept { return instance_; }

    /// Get available physical devices.
    [[nodiscard]] std::vector<VkPhysicalDevice> enumerate_physical_devices() const;

    /// Check if validation layers are enabled.
    [[nodiscard]] bool validation_enabled() const noexcept { return validation_enabled_; }

    /// Get the instance API version.
    [[nodiscard]] uint32_t api_version() const noexcept { return api_version_; }

private:
    void create_instance(const InstanceConfig& config);
    void setup_debug_messenger();
    void cleanup();

    [[nodiscard]] static std::vector<const char*> get_required_extensions(bool enable_validation);
    [[nodiscard]] static std::vector<const char*> get_validation_layers();
    [[nodiscard]] static bool check_validation_layer_support();
    [[nodiscard]] static bool check_extension_support(const std::vector<const char*>& required);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    bool validation_enabled_ = false;
    uint32_t api_version_ = VK_API_VERSION_1_2;
};

/// Exception type for Vulkan errors.
class VulkanError : public std::runtime_error {
public:
    explicit VulkanError(VkResult result, const std::string& message);
    [[nodiscard]] VkResult result() const noexcept { return result_; }
private:
    VkResult result_;
};

/// Convert VkResult to string for error reporting.
[[nodiscard]] const char* vk_result_to_string(VkResult result);

/// Check VkResult and throw on failure.
inline void vk_check(VkResult result, const char* message) {
    if (result != VK_SUCCESS) {
        throw VulkanError(result, message);
    }
}

} // namespace mcsim::vk
