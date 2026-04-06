#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace mc189 {

class VulkanContext;

struct DescriptorBinding {
  uint32_t binding;
  vk::DescriptorType type;
  uint32_t count = 1;
  vk::ShaderStageFlags stage = vk::ShaderStageFlagBits::eCompute;
};

struct PushConstantRange {
  uint32_t offset;
  uint32_t size;
  vk::ShaderStageFlags stage = vk::ShaderStageFlagBits::eCompute;
};

struct SpecializationConstant {
  uint32_t constant_id;
  uint32_t offset;
  uint32_t size;
};

class ComputePipeline {
public:
  struct Config {
    std::vector<uint32_t> spirv_code;
    std::string entry_point = "main";
    std::vector<DescriptorBinding> bindings;
    std::vector<PushConstantRange> push_constants;
    std::vector<SpecializationConstant> specializations;
    std::vector<uint8_t> specialization_data;
    uint32_t local_size_x = 256;
    uint32_t local_size_y = 1;
    uint32_t local_size_z = 1;
  };

  ComputePipeline(const VulkanContext &ctx, const Config &config);
  ~ComputePipeline();

  ComputePipeline(const ComputePipeline &) = delete;
  ComputePipeline &operator=(const ComputePipeline &) = delete;
  ComputePipeline(ComputePipeline &&) noexcept;
  ComputePipeline &operator=(ComputePipeline &&) noexcept;

  vk::Pipeline pipeline() const { return *pipeline_; }
  vk::PipelineLayout layout() const { return *layout_; }
  vk::DescriptorSetLayout descriptor_layout() const {
    return *descriptor_layout_;
  }

  uint32_t local_size_x() const { return local_size_[0]; }
  uint32_t local_size_y() const { return local_size_[1]; }
  uint32_t local_size_z() const { return local_size_[2]; }

  // Descriptor set management
  vk::DescriptorSet allocate_descriptor_set() const;
  std::vector<vk::DescriptorSet> allocate_descriptor_sets(uint32_t count) const;
  void free_descriptor_set(vk::DescriptorSet set) const;
  void free_descriptor_sets(const std::vector<vk::DescriptorSet> &sets) const;

  void update_descriptor(vk::DescriptorSet set, uint32_t binding,
                         vk::Buffer buffer, vk::DeviceSize offset = 0,
                         vk::DeviceSize range = VK_WHOLE_SIZE) const;

  void update_descriptors(
      vk::DescriptorSet set,
      const std::vector<std::tuple<uint32_t, vk::Buffer, vk::DeviceSize,
                                   vk::DeviceSize>> &updates) const;

  // Command recording helpers
  void bind(vk::CommandBuffer cmd) const;
  void bind_descriptor_set(vk::CommandBuffer cmd, vk::DescriptorSet set,
                           uint32_t index = 0) const;

  template <typename T>
  void push_constants(vk::CommandBuffer cmd, const T &data,
                      uint32_t offset = 0) const {
    cmd.pushConstants(*layout_, vk::ShaderStageFlagBits::eCompute, offset,
                      sizeof(T), &data);
  }

  void dispatch(vk::CommandBuffer cmd, uint32_t x, uint32_t y = 1,
                uint32_t z = 1) const;
  void dispatch_for_count(vk::CommandBuffer cmd, uint32_t count) const;
  void dispatch_for_count_2d(vk::CommandBuffer cmd, uint32_t count_x,
                             uint32_t count_y) const;
  void dispatch_for_count_3d(vk::CommandBuffer cmd, uint32_t count_x,
                             uint32_t count_y, uint32_t count_z) const;

  // Static helpers
  static std::vector<uint32_t> load_spirv(const std::string &path);
  static uint32_t workgroup_count(uint32_t total, uint32_t local_size);

private:
  void create_descriptor_layout(const std::vector<DescriptorBinding> &bindings);
  void
  create_pipeline_layout(const std::vector<PushConstantRange> &push_constants);
  void create_pipeline(const Config &config);
  void create_descriptor_pool();

  const VulkanContext *ctx_;
  vk::UniqueShaderModule shader_module_;
  vk::UniqueDescriptorSetLayout descriptor_layout_;
  vk::UniquePipelineLayout layout_;
  vk::UniquePipeline pipeline_;
  vk::UniqueDescriptorPool descriptor_pool_;
  std::vector<DescriptorBinding> bindings_;
  uint32_t local_size_[3];
};

// Manages multiple pipelines with shared descriptor sets
class PipelineRegistry {
public:
  explicit PipelineRegistry(const VulkanContext &ctx);

  ComputePipeline &add(const std::string &name,
                       const ComputePipeline::Config &config);
  ComputePipeline &get(const std::string &name);
  const ComputePipeline &get(const std::string &name) const;
  bool has(const std::string &name) const;
  void remove(const std::string &name);
  void clear();

  std::vector<std::string> names() const;

private:
  const VulkanContext *ctx_;
  std::unordered_map<std::string, std::unique_ptr<ComputePipeline>> pipelines_;
};

// Batch executor for dispatching compute work across many environments
class BatchExecutor {
public:
  struct Config {
    uint32_t max_batch_size = 32768;
    uint32_t command_buffers = 4; // Double/quad buffering
    bool async_execution = true;
  };

  BatchExecutor(const VulkanContext &ctx, const Config &config);
  ~BatchExecutor();

  BatchExecutor(const BatchExecutor &) = delete;
  BatchExecutor &operator=(const BatchExecutor &) = delete;

  // Record commands for a batch
  vk::CommandBuffer begin_batch(uint32_t batch_index);
  void end_batch(uint32_t batch_index);

  // Execute recorded batch
  void submit_batch(uint32_t batch_index);
  void wait_batch(uint32_t batch_index);
  bool is_batch_complete(uint32_t batch_index) const;

  // Convenience: record, submit, wait
  template <typename Func> void execute(Func &&record_fn) {
    uint32_t idx = current_batch_++ % command_buffers_.size();
    wait_batch(idx);
    auto cmd = begin_batch(idx);
    record_fn(cmd);
    end_batch(idx);
    submit_batch(idx);
    if (!config_.async_execution) {
      wait_batch(idx);
    }
  }

  // Wait for all batches to complete
  void wait_all();

  uint32_t max_batch_size() const { return config_.max_batch_size; }
  uint32_t num_command_buffers() const {
    return static_cast<uint32_t>(command_buffers_.size());
  }

private:
  const VulkanContext *ctx_;
  Config config_;
  std::vector<vk::CommandBuffer> command_buffers_;
  std::vector<vk::Fence> fences_;
  std::vector<bool> fence_submitted_;
  uint32_t current_batch_ = 0;
};

} // namespace mc189
