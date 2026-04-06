#include "instance.h"

#include <cstring>
#include <iostream>
#include <sstream>

namespace mcsim::vk {

namespace {

// Debug callback for validation layers
VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {

    (void)type;
    (void)user_data;

    const char* severity_str = "";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        severity_str = "ERROR";
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        severity_str = "WARNING";
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        severity_str = "INFO";
    } else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        severity_str = "VERBOSE";
    }

    std::cerr << "[Vulkan " << severity_str << "] " << callback_data->pMessage << std::endl;

    return VK_FALSE;
}

VkResult create_debug_utils_messenger(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* create_info,
    const VkAllocationCallbacks* allocator,
    VkDebugUtilsMessengerEXT* messenger) {

    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));

    if (func != nullptr) {
        return func(instance, create_info, allocator, messenger);
    }
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void destroy_debug_utils_messenger(
    VkInstance instance,
    VkDebugUtilsMessengerEXT messenger,
    const VkAllocationCallbacks* allocator) {

    auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));

    if (func != nullptr) {
        func(instance, messenger, allocator);
    }
}

VkDebugUtilsMessengerCreateInfoEXT make_debug_messenger_create_info() {
    VkDebugUtilsMessengerCreateInfoEXT create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = debug_callback;
    create_info.pUserData = nullptr;
    return create_info;
}

} // anonymous namespace

// VulkanError implementation
VulkanError::VulkanError(VkResult result, const std::string& message)
    : std::runtime_error(message + ": " + vk_result_to_string(result))
    , result_(result) {}

const char* vk_result_to_string(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
        case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
        case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
        case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        default: return "VK_UNKNOWN_ERROR";
    }
}

// Instance implementation
Instance::Instance(const InstanceConfig& config) {
    create_instance(config);
    if (validation_enabled_) {
        setup_debug_messenger();
    }
}

Instance::~Instance() {
    cleanup();
}

Instance::Instance(Instance&& other) noexcept
    : instance_(other.instance_)
    , debug_messenger_(other.debug_messenger_)
    , validation_enabled_(other.validation_enabled_)
    , api_version_(other.api_version_) {
    other.instance_ = VK_NULL_HANDLE;
    other.debug_messenger_ = VK_NULL_HANDLE;
}

Instance& Instance::operator=(Instance&& other) noexcept {
    if (this != &other) {
        cleanup();
        instance_ = other.instance_;
        debug_messenger_ = other.debug_messenger_;
        validation_enabled_ = other.validation_enabled_;
        api_version_ = other.api_version_;
        other.instance_ = VK_NULL_HANDLE;
        other.debug_messenger_ = VK_NULL_HANDLE;
    }
    return *this;
}

void Instance::create_instance(const InstanceConfig& config) {
    // Check for validation layer support if requested
    validation_enabled_ = config.enable_validation && check_validation_layer_support();

    if (config.enable_validation && !validation_enabled_) {
        std::cerr << "Warning: Validation layers requested but not available\n";
    }

    // Query supported API version
    uint32_t supported_version = VK_API_VERSION_1_0;
    auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
    if (enumerate_version != nullptr) {
        enumerate_version(&supported_version);
    }

    // Use at least 1.2 for features we need, but cap at supported version
    api_version_ = std::min(VK_API_VERSION_1_2, supported_version);

    if (config.verbose) {
        std::cout << "Vulkan API version: "
                  << VK_VERSION_MAJOR(api_version_) << "."
                  << VK_VERSION_MINOR(api_version_) << "."
                  << VK_VERSION_PATCH(api_version_) << "\n";
    }

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = config.app_name.c_str();
    app_info.applicationVersion = config.app_version;
    app_info.pEngineName = "MinecraftSimEngine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = api_version_;

    auto extensions = get_required_extensions(validation_enabled_);
    if (!check_extension_support(extensions)) {
        throw VulkanError(VK_ERROR_EXTENSION_NOT_PRESENT,
                         "Required instance extensions not available");
    }

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    // MoltenVK portability flag for macOS
#ifdef __APPLE__
    create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
    auto validation_layers = get_validation_layers();

    if (validation_enabled_) {
        create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
        create_info.ppEnabledLayerNames = validation_layers.data();

        // Enable debug messenger during instance creation/destruction
        debug_create_info = make_debug_messenger_create_info();
        create_info.pNext = &debug_create_info;
    } else {
        create_info.enabledLayerCount = 0;
        create_info.pNext = nullptr;
    }

    vk_check(vkCreateInstance(&create_info, nullptr, &instance_),
             "Failed to create Vulkan instance");

    if (config.verbose) {
        std::cout << "Vulkan instance created successfully\n";
        if (validation_enabled_) {
            std::cout << "Validation layers enabled\n";
        }
    }
}

void Instance::setup_debug_messenger() {
    auto create_info = make_debug_messenger_create_info();
    vk_check(create_debug_utils_messenger(instance_, &create_info, nullptr, &debug_messenger_),
             "Failed to set up debug messenger");
}

void Instance::cleanup() {
    if (debug_messenger_ != VK_NULL_HANDLE) {
        destroy_debug_utils_messenger(instance_, debug_messenger_, nullptr);
        debug_messenger_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

std::vector<VkPhysicalDevice> Instance::enumerate_physical_devices() const {
    uint32_t device_count = 0;
    vk_check(vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
             "Failed to enumerate physical devices");

    if (device_count == 0) {
        throw VulkanError(VK_ERROR_INITIALIZATION_FAILED,
                         "No Vulkan-capable GPU found");
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vk_check(vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
             "Failed to enumerate physical devices");

    return devices;
}

std::vector<const char*> Instance::get_required_extensions(bool enable_validation) {
    std::vector<const char*> extensions;

    // MoltenVK portability extension for macOS
#ifdef __APPLE__
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

    if (enable_validation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

std::vector<const char*> Instance::get_validation_layers() {
    return {
        "VK_LAYER_KHRONOS_validation"
    };
}

bool Instance::check_validation_layer_support() {
    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    auto required = get_validation_layers();
    for (const char* layer_name : required) {
        bool found = false;
        for (const auto& layer : available_layers) {
            if (std::strcmp(layer_name, layer.layerName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

bool Instance::check_extension_support(const std::vector<const char*>& required) {
    uint32_t extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);

    std::vector<VkExtensionProperties> available(extension_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, available.data());

    for (const char* ext_name : required) {
        bool found = false;
        for (const auto& ext : available) {
            if (std::strcmp(ext_name, ext.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << "Missing required extension: " << ext_name << "\n";
            return false;
        }
    }
    return true;
}

} // namespace mcsim::vk
