#include "mc189/vulkan_context.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace mc189 {

namespace {

const std::vector<const char *> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char *> kDeviceExtensions = {
    // No extensions required for compute-only workloads
};

// Debug callback using vk:: types for vulkan.hpp compatibility
static VkBool32
vk_debug_callback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                  vk::DebugUtilsMessageTypeFlagsEXT type,
                  const vk::DebugUtilsMessengerCallbackDataEXT *data,
                  void *user_data) {
  (void)type;
  (void)user_data;

  const char *severity_str = "UNKNOWN";
  if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
    severity_str = "ERROR";
  } else if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
    severity_str = "WARNING";
  } else if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo) {
    severity_str = "INFO";
  } else if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose) {
    severity_str = "VERBOSE";
  }

  std::cerr << "[Vulkan " << severity_str << "] " << data->pMessage << "\n";
  return VK_FALSE;
}

} // namespace

VulkanContext::VulkanContext(const Config &config) {
  // Load Vulkan function pointers dynamically
  static vk::detail::DynamicLoader dl;
  auto vkGetInstanceProcAddr =
      dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

  validation_enabled_ =
      config.enable_validation && check_validation_layer_support();
  create_instance(config);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance_);

  if (validation_enabled_) {
    setup_debug_messenger();
  }

  pick_physical_device(config.prefer_discrete_gpu);
  create_logical_device();
  VULKAN_HPP_DEFAULT_DISPATCHER.init(*device_);

  query_capabilities();
  create_command_pool();
}

VulkanContext::~VulkanContext() = default;

VulkanContext::VulkanContext(VulkanContext &&) noexcept = default;
VulkanContext &VulkanContext::operator=(VulkanContext &&) noexcept = default;

void VulkanContext::create_instance(const Config &config) {
  vk::ApplicationInfo app_info{};
  app_info.pApplicationName = config.app_name.c_str();
  app_info.applicationVersion = config.app_version;
  app_info.pEngineName = "mc189";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;

  std::vector<const char *> extensions;

  // MoltenVK portability extension on macOS
#ifdef __APPLE__
  extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  extensions.push_back("VK_KHR_get_physical_device_properties2");
#endif

  if (validation_enabled_) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  vk::InstanceCreateInfo create_info{};
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();

#ifdef __APPLE__
  create_info.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

  vk::DebugUtilsMessengerCreateInfoEXT debug_create_info{};
  if (validation_enabled_) {
    create_info.enabledLayerCount =
        static_cast<uint32_t>(kValidationLayers.size());
    create_info.ppEnabledLayerNames = kValidationLayers.data();

    debug_create_info.messageSeverity =
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning;
    debug_create_info.messageType =
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    debug_create_info.pfnUserCallback = vk_debug_callback;
    create_info.pNext = &debug_create_info;
  }

  instance_ = vk::createInstanceUnique(create_info);
}

void VulkanContext::setup_debug_messenger() {
  vk::DebugUtilsMessengerCreateInfoEXT create_info{};
  create_info.messageSeverity =
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning;
  create_info.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
  create_info.pfnUserCallback = vk_debug_callback;

  debug_messenger_ = instance_->createDebugUtilsMessengerEXTUnique(create_info);
}

void VulkanContext::pick_physical_device(bool prefer_discrete) {
  auto devices = instance_->enumeratePhysicalDevices();
  if (devices.empty()) {
    throw std::runtime_error("No Vulkan-capable GPU found");
  }

  int best_score = -1;
  for (const auto &device : devices) {
    auto indices = find_queue_families(device);
    if (!indices.is_complete())
      continue;

    int score = rate_device(device);
    if (!prefer_discrete) {
      // If not preferring discrete, reduce discrete GPU score
      auto props = device.getProperties();
      if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
        score /= 2;
      }
    }

    if (score > best_score) {
      best_score = score;
      physical_device_ = device;
      queue_indices_ = indices;
    }
  }

  if (!physical_device_) {
    throw std::runtime_error("No suitable GPU found");
  }
}

int VulkanContext::rate_device(vk::PhysicalDevice device) const {
  auto props = device.getProperties();
  auto features = device.getFeatures();

  int score = 0;

  // Discrete GPUs are generally faster
  if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
    score += 1000;
  } else if (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
    score += 500;
  }

  // Prefer higher memory
  auto mem_props = device.getMemoryProperties();
  for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
    if (mem_props.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
      score +=
          static_cast<int>(mem_props.memoryHeaps[i].size / (1024 * 1024 * 100));
    }
  }

  // Bonus for shader features useful in compute
  if (features.shaderInt64)
    score += 50;
  if (features.shaderFloat64)
    score += 50;
  if (features.shaderInt16)
    score += 25;

  return score;
}

QueueFamilyIndices
VulkanContext::find_queue_families(vk::PhysicalDevice device) const {
  QueueFamilyIndices indices;
  auto families = device.getQueueFamilyProperties();

  // Find dedicated compute queue (prefer one without graphics)
  for (uint32_t i = 0; i < families.size(); i++) {
    if ((families[i].queueFlags & vk::QueueFlagBits::eCompute) &&
        !(families[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
      indices.compute = i;
      break;
    }
  }

  // Fallback to any compute-capable queue
  if (!indices.compute) {
    for (uint32_t i = 0; i < families.size(); i++) {
      if (families[i].queueFlags & vk::QueueFlagBits::eCompute) {
        indices.compute = i;
        break;
      }
    }
  }

  // Find dedicated transfer queue
  for (uint32_t i = 0; i < families.size(); i++) {
    if ((families[i].queueFlags & vk::QueueFlagBits::eTransfer) &&
        !(families[i].queueFlags & vk::QueueFlagBits::eCompute) &&
        !(families[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
      indices.transfer = i;
      break;
    }
  }

  return indices;
}

void VulkanContext::create_logical_device() {
  std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
  std::set<uint32_t> unique_families = {queue_indices_.compute.value()};
  if (queue_indices_.transfer) {
    unique_families.insert(queue_indices_.transfer.value());
  }

  float priority = 1.0f;
  for (uint32_t family : unique_families) {
    vk::DeviceQueueCreateInfo queue_info{};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    queue_create_infos.push_back(queue_info);
  }

  // Query available features
  auto available_features = physical_device_.getFeatures();

  // Enable features useful for compute (only if supported)
  vk::PhysicalDeviceFeatures device_features{};
  device_features.shaderInt64 = available_features.shaderInt64;
  device_features.shaderFloat64 = available_features.shaderFloat64;
  device_features.shaderInt16 = available_features.shaderInt16;

  // Enable 16-bit storage if available
  vk::PhysicalDevice16BitStorageFeatures storage_16bit{};
  storage_16bit.storageBuffer16BitAccess =
      VK_FALSE; // Disabled for MoltenVK compatibility

  // Enable 8-bit storage if available
  vk::PhysicalDevice8BitStorageFeatures storage_8bit{};
  storage_8bit.storageBuffer8BitAccess =
      VK_FALSE; // Disabled for MoltenVK compatibility
  storage_8bit.pNext = &storage_16bit;

  std::vector<const char *> extensions = kDeviceExtensions;

#ifdef __APPLE__
  extensions.push_back("VK_KHR_portability_subset");
#endif

  vk::DeviceCreateInfo create_info{};
  create_info.queueCreateInfoCount =
      static_cast<uint32_t>(queue_create_infos.size());
  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.pEnabledFeatures = &device_features;
  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();
  create_info.pNext = &storage_8bit;

  if (validation_enabled_) {
    create_info.enabledLayerCount =
        static_cast<uint32_t>(kValidationLayers.size());
    create_info.ppEnabledLayerNames = kValidationLayers.data();
  }

  device_ = physical_device_.createDeviceUnique(create_info);
  compute_queue_ = device_->getQueue(queue_indices_.compute.value(), 0);
  transfer_queue_ = queue_indices_.transfer
                        ? device_->getQueue(queue_indices_.transfer.value(), 0)
                        : compute_queue_;
}

void VulkanContext::create_command_pool() {
  vk::CommandPoolCreateInfo pool_info{};
  pool_info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  pool_info.queueFamilyIndex = queue_indices_.compute.value();

  command_pool_ = device_->createCommandPoolUnique(pool_info);
}

void VulkanContext::query_capabilities() {
  auto props = physical_device_.getProperties();
  auto limits = props.limits;

  capabilities_.device_name = std::string(props.deviceName.data());
  capabilities_.vendor_id = props.vendorID;

  capabilities_.max_workgroup_size[0] = limits.maxComputeWorkGroupSize[0];
  capabilities_.max_workgroup_size[1] = limits.maxComputeWorkGroupSize[1];
  capabilities_.max_workgroup_size[2] = limits.maxComputeWorkGroupSize[2];

  capabilities_.max_workgroup_count[0] = limits.maxComputeWorkGroupCount[0];
  capabilities_.max_workgroup_count[1] = limits.maxComputeWorkGroupCount[1];
  capabilities_.max_workgroup_count[2] = limits.maxComputeWorkGroupCount[2];

  capabilities_.max_compute_shared_memory = limits.maxComputeSharedMemorySize;
  capabilities_.max_storage_buffer_range = limits.maxStorageBufferRange;
  capabilities_.max_uniform_buffer_range = limits.maxUniformBufferRange;
  capabilities_.max_push_constant_size = limits.maxPushConstantsSize;
  capabilities_.max_bound_descriptor_sets = limits.maxBoundDescriptorSets;

  // Calculate device-local memory
  auto mem_props = physical_device_.getMemoryProperties();
  capabilities_.device_local_memory = 0;
  for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
    if (mem_props.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
      capabilities_.device_local_memory += mem_props.memoryHeaps[i].size;
    }
  }

  // Check for optional features
  vk::PhysicalDeviceFeatures2 features2{};
  vk::PhysicalDevice16BitStorageFeatures storage_16{};
  vk::PhysicalDevice8BitStorageFeatures storage_8{};
  features2.pNext = &storage_16;
  storage_16.pNext = &storage_8;

  physical_device_.getFeatures2(&features2);
  capabilities_.supports_16bit_storage = storage_16.storageBuffer16BitAccess;
  capabilities_.supports_8bit_storage = storage_8.storageBuffer8BitAccess;
}

vk::CommandBuffer
VulkanContext::allocate_command_buffer(vk::CommandBufferLevel level) const {
  vk::CommandBufferAllocateInfo alloc_info{};
  alloc_info.commandPool = *command_pool_;
  alloc_info.level = level;
  alloc_info.commandBufferCount = 1;

  return device_->allocateCommandBuffers(alloc_info)[0];
}

std::vector<vk::CommandBuffer>
VulkanContext::allocate_command_buffers(uint32_t count,
                                        vk::CommandBufferLevel level) const {
  vk::CommandBufferAllocateInfo alloc_info{};
  alloc_info.commandPool = *command_pool_;
  alloc_info.level = level;
  alloc_info.commandBufferCount = count;

  return device_->allocateCommandBuffers(alloc_info);
}

void VulkanContext::free_command_buffer(vk::CommandBuffer buffer) const {
  device_->freeCommandBuffers(*command_pool_, buffer);
}

void VulkanContext::free_command_buffers(
    const std::vector<vk::CommandBuffer> &buffers) const {
  device_->freeCommandBuffers(*command_pool_, buffers);
}

void VulkanContext::submit_and_wait(vk::CommandBuffer cmd) const {
  vk::SubmitInfo submit{};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;

  vk::UniqueFence fence = device_->createFenceUnique({});
  compute_queue_.submit(submit, *fence);
  auto result = device_->waitForFences(*fence, VK_TRUE, UINT64_MAX);
  if (result != vk::Result::eSuccess) {
    throw std::runtime_error("Fence wait failed");
  }
}

void VulkanContext::submit(vk::CommandBuffer cmd, vk::Fence fence) const {
  vk::SubmitInfo submit{};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  compute_queue_.submit(submit, fence);
}

vk::Fence VulkanContext::create_fence(bool signaled) const {
  vk::FenceCreateInfo info{};
  if (signaled) {
    info.flags = vk::FenceCreateFlagBits::eSignaled;
  }
  return device_->createFence(info);
}

void VulkanContext::wait_fence(vk::Fence fence, uint64_t timeout) const {
  auto result = device_->waitForFences(fence, VK_TRUE, timeout);
  if (result == vk::Result::eTimeout) {
    throw std::runtime_error("Fence wait timed out");
  }
}

void VulkanContext::reset_fence(vk::Fence fence) const {
  device_->resetFences(fence);
}

void VulkanContext::destroy_fence(vk::Fence fence) const {
  device_->destroyFence(fence);
}

uint32_t
VulkanContext::find_memory_type(uint32_t type_filter,
                                vk::MemoryPropertyFlags properties) const {
  auto mem_props = physical_device_.getMemoryProperties();

  for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find suitable memory type");
}

std::vector<std::string> VulkanContext::get_available_layers() {
  auto layers = vk::enumerateInstanceLayerProperties();
  std::vector<std::string> names;
  names.reserve(layers.size());
  for (const auto &layer : layers) {
    names.emplace_back(layer.layerName);
  }
  return names;
}

std::vector<std::string> VulkanContext::get_available_extensions() {
  auto extensions = vk::enumerateInstanceExtensionProperties();
  std::vector<std::string> names;
  names.reserve(extensions.size());
  for (const auto &ext : extensions) {
    names.emplace_back(ext.extensionName);
  }
  return names;
}

bool VulkanContext::check_validation_layer_support() {
  auto available = vk::enumerateInstanceLayerProperties();
  for (const char *layer_name : kValidationLayers) {
    bool found = false;
    for (const auto &layer : available) {
      if (std::strcmp(layer_name, layer.layerName) == 0) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

} // namespace mc189
