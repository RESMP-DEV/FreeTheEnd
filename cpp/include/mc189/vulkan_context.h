#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace mc189 {

struct QueueFamilyIndices {
  std::optional<uint32_t> compute;
  std::optional<uint32_t> transfer;

  bool is_complete() const { return compute.has_value(); }
};

struct DeviceCapabilities {
  uint32_t max_workgroup_size[3];
  uint32_t max_workgroup_count[3];
  uint32_t max_compute_shared_memory;
  uint64_t max_storage_buffer_range;
  uint64_t max_uniform_buffer_range;
  uint32_t max_push_constant_size;
  uint32_t max_bound_descriptor_sets;
  uint64_t device_local_memory;
  bool supports_16bit_storage;
  bool supports_8bit_storage;
  std::string device_name;
  uint32_t vendor_id;
};

class VulkanContext {
public:
  struct Config {
    bool enable_validation = false;
    bool prefer_discrete_gpu = true;
    std::string app_name = "mc189";
    uint32_t app_version = VK_MAKE_VERSION(1, 0, 0);
  };

  explicit VulkanContext(const Config &config);
  ~VulkanContext();

  VulkanContext(const VulkanContext &) = delete;
  VulkanContext &operator=(const VulkanContext &) = delete;
  VulkanContext(VulkanContext &&) noexcept;
  VulkanContext &operator=(VulkanContext &&) noexcept;

  vk::Instance instance() const { return *instance_; }
  vk::PhysicalDevice physical_device() const { return physical_device_; }
  vk::Device device() const { return *device_; }
  vk::Queue compute_queue() const { return compute_queue_; }
  vk::Queue transfer_queue() const { return transfer_queue_; }
  uint32_t compute_queue_family() const {
    return queue_indices_.compute.value();
  }
  uint32_t transfer_queue_family() const {
    return queue_indices_.transfer.value_or(queue_indices_.compute.value());
  }
  const DeviceCapabilities &capabilities() const { return capabilities_; }
  vk::CommandPool command_pool() const { return *command_pool_; }

  vk::CommandBuffer allocate_command_buffer(
      vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const;
  std::vector<vk::CommandBuffer> allocate_command_buffers(
      uint32_t count,
      vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary) const;
  void free_command_buffer(vk::CommandBuffer buffer) const;
  void
  free_command_buffers(const std::vector<vk::CommandBuffer> &buffers) const;

  void submit_and_wait(vk::CommandBuffer cmd) const;
  void submit(vk::CommandBuffer cmd, vk::Fence fence = {}) const;

  vk::Fence create_fence(bool signaled = false) const;
  void wait_fence(vk::Fence fence, uint64_t timeout = UINT64_MAX) const;
  void reset_fence(vk::Fence fence) const;
  void destroy_fence(vk::Fence fence) const;

  uint32_t find_memory_type(uint32_t type_filter,
                            vk::MemoryPropertyFlags properties) const;

  static std::vector<std::string> get_available_layers();
  static std::vector<std::string> get_available_extensions();
  static bool check_validation_layer_support();

private:
  void create_instance(const Config &config);
  void setup_debug_messenger();
  void pick_physical_device(bool prefer_discrete);
  void create_logical_device();
  void create_command_pool();
  void query_capabilities();
  QueueFamilyIndices find_queue_families(vk::PhysicalDevice device) const;
  int rate_device(vk::PhysicalDevice device) const;

  vk::UniqueInstance instance_;
  vk::UniqueDebugUtilsMessengerEXT debug_messenger_;
  vk::PhysicalDevice physical_device_;
  vk::UniqueDevice device_;
  vk::Queue compute_queue_;
  vk::Queue transfer_queue_;
  QueueFamilyIndices queue_indices_;
  DeviceCapabilities capabilities_;
  vk::UniqueCommandPool command_pool_;
  bool validation_enabled_ = false;
};

} // namespace mc189
