#include "device.h"

// Define VMA implementation in this translation unit
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>

namespace mcsim::vk {

namespace {

// Extension names as constants
constexpr const char* EXT_SHADER_FLOAT16_INT8 = VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME;
constexpr const char* EXT_STORAGE_BUFFER_CLASS = VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME;
constexpr const char* EXT_8BIT_STORAGE = VK_KHR_8BIT_STORAGE_EXTENSION_NAME;
constexpr const char* EXT_16BIT_STORAGE = VK_KHR_16BIT_STORAGE_EXTENSION_NAME;

#ifdef __APPLE__
constexpr const char* EXT_PORTABILITY_SUBSET = "VK_KHR_portability_subset";
#endif

} // anonymous namespace

// Device implementation

Device::Device(Instance& instance, const DeviceConfig& config) {
    select_physical_device(instance, config);
    create_logical_device(config);
    create_allocator(instance);
}

Device::Device(Instance& instance, VkPhysicalDevice physical_device,
               const DeviceConfig& config)
    : physical_device_(physical_device) {

    queue_families_ = find_queue_families(physical_device_,
                                          config.prefer_dedicated_transfer_queue);
    if (!queue_families_.is_complete()) {
        throw VulkanError(VK_ERROR_FEATURE_NOT_PRESENT,
                         "Physical device lacks compute queue family");
    }

    capabilities_ = query_capabilities(physical_device_);
    create_logical_device(config);
    create_allocator(instance);
}

Device::~Device() {
    cleanup();
}

Device::Device(Device&& other) noexcept
    : physical_device_(other.physical_device_)
    , device_(other.device_)
    , allocator_(other.allocator_)
    , compute_queue_(other.compute_queue_)
    , transfer_queue_(other.transfer_queue_)
    , queue_families_(other.queue_families_)
    , capabilities_(std::move(other.capabilities_)) {
    other.physical_device_ = VK_NULL_HANDLE;
    other.device_ = VK_NULL_HANDLE;
    other.allocator_ = VK_NULL_HANDLE;
    other.compute_queue_ = VK_NULL_HANDLE;
    other.transfer_queue_ = VK_NULL_HANDLE;
}

Device& Device::operator=(Device&& other) noexcept {
    if (this != &other) {
        cleanup();
        physical_device_ = other.physical_device_;
        device_ = other.device_;
        allocator_ = other.allocator_;
        compute_queue_ = other.compute_queue_;
        transfer_queue_ = other.transfer_queue_;
        queue_families_ = other.queue_families_;
        capabilities_ = std::move(other.capabilities_);
        other.physical_device_ = VK_NULL_HANDLE;
        other.device_ = VK_NULL_HANDLE;
        other.allocator_ = VK_NULL_HANDLE;
        other.compute_queue_ = VK_NULL_HANDLE;
        other.transfer_queue_ = VK_NULL_HANDLE;
    }
    return *this;
}

void Device::select_physical_device(Instance& instance, const DeviceConfig& config) {
    auto devices = instance.enumerate_physical_devices();

    int best_score = -1;
    VkPhysicalDevice best_device = VK_NULL_HANDLE;

    for (auto device : devices) {
        int score = rate_device(device, config);
        if (score > best_score) {
            best_score = score;
            best_device = device;
        }
    }

    if (best_device == VK_NULL_HANDLE || best_score < 0) {
        throw VulkanError(VK_ERROR_FEATURE_NOT_PRESENT,
                         "No suitable GPU found for compute workloads");
    }

    physical_device_ = best_device;
    queue_families_ = find_queue_families(physical_device_,
                                          config.prefer_dedicated_transfer_queue);
    capabilities_ = query_capabilities(physical_device_);

    if (config.verbose) {
        std::cout << "Selected GPU: " << capabilities_.device_name
                  << " (score: " << best_score << ")\n";
        std::cout << "  Compute work group size: "
                  << capabilities_.max_compute_work_group_size[0] << "x"
                  << capabilities_.max_compute_work_group_size[1] << "x"
                  << capabilities_.max_compute_work_group_size[2] << "\n";
        std::cout << "  Device memory: "
                  << (capabilities_.total_device_memory / (1024 * 1024)) << " MB\n";
    }
}

void Device::create_logical_device(const DeviceConfig& config) {
    // Set up queue create infos
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::set<uint32_t> unique_queue_families;

    unique_queue_families.insert(queue_families_.compute_family.value());
    if (queue_families_.transfer_family.has_value()) {
        unique_queue_families.insert(queue_families_.transfer_family.value());
    }

    float queue_priority = 1.0f;
    for (uint32_t family : unique_queue_families) {
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;
        queue_create_infos.push_back(queue_info);
    }

    // Get required extensions
    auto extensions = get_required_extensions(physical_device_, config);

    // Build feature chain for required capabilities
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    // Enable shaderInt64 for Java LCG
    features2.features.shaderInt64 = VK_TRUE;

    // Float16/Int8 features
    VkPhysicalDeviceShaderFloat16Int8Features float16_int8_features{};
    float16_int8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    float16_int8_features.shaderInt8 = VK_TRUE;

    // 8-bit storage features
    VkPhysicalDevice8BitStorageFeatures storage_8bit_features{};
    storage_8bit_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    storage_8bit_features.storageBuffer8BitAccess = VK_TRUE;
    storage_8bit_features.uniformAndStorageBuffer8BitAccess = VK_TRUE;

    // 16-bit storage features (useful for half-precision math)
    VkPhysicalDevice16BitStorageFeatures storage_16bit_features{};
    storage_16bit_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage_16bit_features.storageBuffer16BitAccess = VK_TRUE;

    // Chain features
    features2.pNext = &float16_int8_features;
    float16_int8_features.pNext = &storage_8bit_features;
    storage_8bit_features.pNext = &storage_16bit_features;

    // Create device
    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pNext = &features2;
    create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
    create_info.pQueueCreateInfos = queue_create_infos.data();
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    create_info.pEnabledFeatures = nullptr;  // Using features2 instead

    vk_check(vkCreateDevice(physical_device_, &create_info, nullptr, &device_),
             "Failed to create logical device");

    // Get queues
    vkGetDeviceQueue(device_, queue_families_.compute_family.value(), 0, &compute_queue_);

    if (queue_families_.transfer_family.has_value()) {
        vkGetDeviceQueue(device_, queue_families_.transfer_family.value(), 0, &transfer_queue_);
    } else {
        transfer_queue_ = compute_queue_;
    }

    if (config.verbose) {
        std::cout << "Logical device created\n";
        std::cout << "  Compute queue family: " << queue_families_.compute_family.value() << "\n";
        if (has_dedicated_transfer_queue()) {
            std::cout << "  Transfer queue family: "
                      << queue_families_.transfer_family.value() << " (dedicated)\n";
        }
    }
}

void Device::create_allocator(Instance& instance) {
    VmaVulkanFunctions vk_funcs{};
    vk_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vk_funcs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo alloc_info{};
    alloc_info.vulkanApiVersion = instance.api_version();
    alloc_info.physicalDevice = physical_device_;
    alloc_info.device = device_;
    alloc_info.instance = instance.handle();
    alloc_info.pVulkanFunctions = &vk_funcs;

    vk_check(static_cast<VkResult>(vmaCreateAllocator(&alloc_info, &allocator_)),
             "Failed to create VMA allocator");
}

void Device::cleanup() {
    if (allocator_ != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator_);
        allocator_ = VK_NULL_HANDLE;
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
}

void Device::wait_idle() const {
    vk_check(vkDeviceWaitIdle(device_), "Failed to wait for device idle");
}

VkCommandPool Device::create_compute_command_pool(VkCommandPoolCreateFlags flags) const {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = flags;
    pool_info.queueFamilyIndex = queue_families_.compute_family.value();

    VkCommandPool pool;
    vk_check(vkCreateCommandPool(device_, &pool_info, nullptr, &pool),
             "Failed to create compute command pool");
    return pool;
}

VkCommandPool Device::create_transfer_command_pool(VkCommandPoolCreateFlags flags) const {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = flags;
    pool_info.queueFamilyIndex = transfer_queue_family();

    VkCommandPool pool;
    vk_check(vkCreateCommandPool(device_, &pool_info, nullptr, &pool),
             "Failed to create transfer command pool");
    return pool;
}

void Device::submit_compute_and_wait(VkCommandBuffer cmd) const {
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    VkFence fence;
    vk_check(vkCreateFence(device_, &fence_info, nullptr, &fence),
             "Failed to create fence");

    vk_check(vkQueueSubmit(compute_queue_, 1, &submit_info, fence),
             "Failed to submit to compute queue");

    vk_check(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX),
             "Failed to wait for fence");

    vkDestroyFence(device_, fence, nullptr);
}

VkPhysicalDeviceProperties Device::get_properties() const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    return props;
}

VkPhysicalDeviceMemoryProperties Device::get_memory_properties() const {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &props);
    return props;
}

uint32_t Device::find_memory_type(uint32_t type_filter,
                                   VkMemoryPropertyFlags properties) const {
    auto mem_props = get_memory_properties();

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw VulkanError(VK_ERROR_FEATURE_NOT_PRESENT,
                     "Failed to find suitable memory type");
}

int Device::rate_device(VkPhysicalDevice device, const DeviceConfig& config) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    // Must have compute queue
    auto families = find_queue_families(device, false);
    if (!families.is_complete()) {
        return -1;
    }

    // Get required extensions and check support
    auto required_extensions = get_required_extensions(device, config);
    if (!check_device_extension_support(device, required_extensions)) {
        return -1;
    }

    // Query features
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceShaderFloat16Int8Features float16_int8{};
    float16_int8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;

    VkPhysicalDevice8BitStorageFeatures storage_8bit{};
    storage_8bit.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;

    features2.pNext = &float16_int8;
    float16_int8.pNext = &storage_8bit;

    vkGetPhysicalDeviceFeatures2(device, &features2);

    // Check required features
    if (config.require_int64 && !features2.features.shaderInt64) {
        return -1;
    }

    if (config.require_float16_int8 && !float16_int8.shaderInt8) {
        return -1;
    }

    if (config.require_storage_buffer && !storage_8bit.storageBuffer8BitAccess) {
        return -1;
    }

    // Score based on device type and capabilities
    int score = 0;

    // Prefer discrete GPUs
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 1000;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        score += 500;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
        score += 250;
    }

    // Score by compute capabilities
    score += props.limits.maxComputeWorkGroupInvocations / 10;
    score += props.limits.maxComputeSharedMemorySize / 1024;

    // Score by memory
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(device, &mem_props);

    uint64_t total_device_memory = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            total_device_memory += mem_props.memoryHeaps[i].size;
        }
    }
    score += static_cast<int>(total_device_memory / (1024 * 1024 * 100)); // Per 100MB

    return score;
}

bool Device::check_device_extension_support(VkPhysicalDevice device,
                                             const std::vector<const char*>& extensions) {
    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

    std::vector<VkExtensionProperties> available(extension_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available.data());

    for (const char* ext_name : extensions) {
        bool found = false;
        for (const auto& ext : available) {
            if (std::strcmp(ext_name, ext.extensionName) == 0) {
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

QueueFamilyIndices Device::find_queue_families(VkPhysicalDevice device,
                                                bool prefer_dedicated_transfer) {
    QueueFamilyIndices indices;

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                              queue_families.data());

    uint32_t best_compute_idx = UINT32_MAX;
    uint32_t best_transfer_idx = UINT32_MAX;

    for (uint32_t i = 0; i < queue_family_count; i++) {
        const auto& family = queue_families[i];

        // Find compute queue (preferably compute-only for async compute)
        if (family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            if (best_compute_idx == UINT32_MAX) {
                best_compute_idx = i;
            }
            // Prefer compute-only queue
            if (!(family.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                best_compute_idx = i;
            }
        }

        // Find dedicated transfer queue
        if (prefer_dedicated_transfer &&
            (family.queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(family.queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(family.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            best_transfer_idx = i;
        }
    }

    if (best_compute_idx != UINT32_MAX) {
        indices.compute_family = best_compute_idx;
    }

    if (best_transfer_idx != UINT32_MAX) {
        indices.transfer_family = best_transfer_idx;
    }

    return indices;
}

DeviceCapabilities Device::query_capabilities(VkPhysicalDevice device) {
    DeviceCapabilities caps;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    caps.device_name = props.deviceName;
    caps.vendor_id = props.vendorID;
    caps.device_id = props.deviceID;

    caps.max_compute_work_group_count[0] = props.limits.maxComputeWorkGroupCount[0];
    caps.max_compute_work_group_count[1] = props.limits.maxComputeWorkGroupCount[1];
    caps.max_compute_work_group_count[2] = props.limits.maxComputeWorkGroupCount[2];

    caps.max_compute_work_group_size[0] = props.limits.maxComputeWorkGroupSize[0];
    caps.max_compute_work_group_size[1] = props.limits.maxComputeWorkGroupSize[1];
    caps.max_compute_work_group_size[2] = props.limits.maxComputeWorkGroupSize[2];

    caps.max_compute_work_group_invocations = props.limits.maxComputeWorkGroupInvocations;
    caps.max_compute_shared_memory_size = props.limits.maxComputeSharedMemorySize;
    caps.max_storage_buffer_range = props.limits.maxStorageBufferRange;

    // Query features
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceShaderFloat16Int8Features float16_int8{};
    float16_int8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;

    VkPhysicalDevice8BitStorageFeatures storage_8bit{};
    storage_8bit.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;

    features2.pNext = &float16_int8;
    float16_int8.pNext = &storage_8bit;

    vkGetPhysicalDeviceFeatures2(device, &features2);

    caps.shader_int64 = features2.features.shaderInt64;
    caps.shader_float16_int8 = float16_int8.shaderInt8 || float16_int8.shaderFloat16;
    caps.storage_buffer_storage_class = storage_8bit.storageBuffer8BitAccess;

    // Get total device memory
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(device, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            caps.total_device_memory += mem_props.memoryHeaps[i].size;
        }
    }

    return caps;
}

std::vector<const char*> Device::get_required_extensions(VkPhysicalDevice device,
                                                          const DeviceConfig& config) {
    std::vector<const char*> extensions;

    // Core compute extensions
    if (config.require_float16_int8) {
        extensions.push_back(EXT_SHADER_FLOAT16_INT8);
        extensions.push_back(EXT_8BIT_STORAGE);
    }

    if (config.require_storage_buffer) {
        extensions.push_back(EXT_STORAGE_BUFFER_CLASS);
        extensions.push_back(EXT_16BIT_STORAGE);
    }

    // MoltenVK portability subset
#ifdef __APPLE__
    // Check if portability subset is available
    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

    std::vector<VkExtensionProperties> available(extension_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available.data());

    for (const auto& ext : available) {
        if (std::strcmp(ext.extensionName, EXT_PORTABILITY_SUBSET) == 0) {
            extensions.push_back(EXT_PORTABILITY_SUBSET);
            break;
        }
    }
#else
    (void)device;
#endif

    return extensions;
}

// CommandPool implementation

CommandPool::CommandPool(VkDevice device, VkCommandPool pool)
    : device_(device), pool_(pool) {}

CommandPool::~CommandPool() {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, pool_, nullptr);
    }
}

CommandPool::CommandPool(CommandPool&& other) noexcept
    : device_(other.device_), pool_(other.pool_) {
    other.pool_ = VK_NULL_HANDLE;
}

CommandPool& CommandPool::operator=(CommandPool&& other) noexcept {
    if (this != &other) {
        if (pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, pool_, nullptr);
        }
        device_ = other.device_;
        pool_ = other.pool_;
        other.pool_ = VK_NULL_HANDLE;
    }
    return *this;
}

VkCommandBuffer CommandPool::allocate_command_buffer() const {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vk_check(vkAllocateCommandBuffers(device_, &alloc_info, &cmd),
             "Failed to allocate command buffer");
    return cmd;
}

void CommandPool::free_command_buffer(VkCommandBuffer cmd) const {
    vkFreeCommandBuffers(device_, pool_, 1, &cmd);
}

void CommandPool::reset(VkCommandPoolResetFlags flags) const {
    vk_check(vkResetCommandPool(device_, pool_, flags),
             "Failed to reset command pool");
}

// Fence implementation

Fence::Fence(VkDevice device, bool signaled)
    : device_(device) {
    VkFenceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) {
        create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    }

    vk_check(vkCreateFence(device_, &create_info, nullptr, &fence_),
             "Failed to create fence");
}

Fence::~Fence() {
    if (fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, fence_, nullptr);
    }
}

Fence::Fence(Fence&& other) noexcept
    : device_(other.device_), fence_(other.fence_) {
    other.fence_ = VK_NULL_HANDLE;
}

Fence& Fence::operator=(Fence&& other) noexcept {
    if (this != &other) {
        if (fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(device_, fence_, nullptr);
        }
        device_ = other.device_;
        fence_ = other.fence_;
        other.fence_ = VK_NULL_HANDLE;
    }
    return *this;
}

void Fence::wait(uint64_t timeout_ns) const {
    vk_check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, timeout_ns),
             "Failed to wait for fence");
}

void Fence::reset() const {
    vk_check(vkResetFences(device_, 1, &fence_),
             "Failed to reset fence");
}

bool Fence::is_signaled() const {
    VkResult result = vkGetFenceStatus(device_, fence_);
    return result == VK_SUCCESS;
}

} // namespace mcsim::vk
