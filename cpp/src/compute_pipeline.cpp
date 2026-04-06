#include "mc189/compute_pipeline.h"
#include "mc189/vulkan_context.h"
#include <fstream>
#include <stdexcept>

namespace mc189 {

ComputePipeline::ComputePipeline(const VulkanContext& ctx, const Config& config)
    : ctx_(&ctx) {
    local_size_[0] = config.local_size_x;
    local_size_[1] = config.local_size_y;
    local_size_[2] = config.local_size_z;
    bindings_ = config.bindings;

    create_descriptor_layout(config.bindings);
    create_pipeline_layout(config.push_constants);
    create_pipeline(config);
    create_descriptor_pool();
}

ComputePipeline::~ComputePipeline() = default;

ComputePipeline::ComputePipeline(ComputePipeline&&) noexcept = default;
ComputePipeline& ComputePipeline::operator=(ComputePipeline&&) noexcept = default;

void ComputePipeline::create_descriptor_layout(const std::vector<DescriptorBinding>& bindings) {
    std::vector<vk::DescriptorSetLayoutBinding> layout_bindings;
    layout_bindings.reserve(bindings.size());

    for (const auto& binding : bindings) {
        vk::DescriptorSetLayoutBinding layout_binding{};
        layout_binding.binding = binding.binding;
        layout_binding.descriptorType = binding.type;
        layout_binding.descriptorCount = binding.count;
        layout_binding.stageFlags = binding.stage;
        layout_bindings.push_back(layout_binding);
    }

    vk::DescriptorSetLayoutCreateInfo layout_info{};
    layout_info.bindingCount = static_cast<uint32_t>(layout_bindings.size());
    layout_info.pBindings = layout_bindings.data();

    descriptor_layout_ = ctx_->device().createDescriptorSetLayoutUnique(layout_info);
}

void ComputePipeline::create_pipeline_layout(const std::vector<PushConstantRange>& push_constants) {
    std::vector<vk::PushConstantRange> ranges;
    ranges.reserve(push_constants.size());

    for (const auto& pc : push_constants) {
        vk::PushConstantRange range{};
        range.stageFlags = pc.stage;
        range.offset = pc.offset;
        range.size = pc.size;
        ranges.push_back(range);
    }

    vk::PipelineLayoutCreateInfo layout_info{};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &*descriptor_layout_;
    layout_info.pushConstantRangeCount = static_cast<uint32_t>(ranges.size());
    layout_info.pPushConstantRanges = ranges.data();

    layout_ = ctx_->device().createPipelineLayoutUnique(layout_info);
}

void ComputePipeline::create_pipeline(const Config& config) {
    // Create shader module
    vk::ShaderModuleCreateInfo module_info{};
    module_info.codeSize = config.spirv_code.size() * sizeof(uint32_t);
    module_info.pCode = config.spirv_code.data();

    shader_module_ = ctx_->device().createShaderModuleUnique(module_info);

    // Specialization constants
    std::vector<vk::SpecializationMapEntry> spec_entries;
    spec_entries.reserve(config.specializations.size());

    for (const auto& spec : config.specializations) {
        vk::SpecializationMapEntry entry{};
        entry.constantID = spec.constant_id;
        entry.offset = spec.offset;
        entry.size = spec.size;
        spec_entries.push_back(entry);
    }

    vk::SpecializationInfo spec_info{};
    spec_info.mapEntryCount = static_cast<uint32_t>(spec_entries.size());
    spec_info.pMapEntries = spec_entries.data();
    spec_info.dataSize = config.specialization_data.size();
    spec_info.pData = config.specialization_data.data();

    vk::PipelineShaderStageCreateInfo stage_info{};
    stage_info.stage = vk::ShaderStageFlagBits::eCompute;
    stage_info.module = *shader_module_;
    stage_info.pName = config.entry_point.c_str();
    if (!spec_entries.empty()) {
        stage_info.pSpecializationInfo = &spec_info;
    }

    vk::ComputePipelineCreateInfo pipeline_info{};
    pipeline_info.stage = stage_info;
    pipeline_info.layout = *layout_;

    auto result = ctx_->device().createComputePipelineUnique({}, pipeline_info);
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to create compute pipeline");
    }
    pipeline_ = std::move(result.value);
}

void ComputePipeline::create_descriptor_pool() {
    // Pool sizes based on bindings
    std::unordered_map<vk::DescriptorType, uint32_t> type_counts;
    for (const auto& binding : bindings_) {
        type_counts[binding.type] += binding.count * 1024;  // Support up to 1024 sets
    }

    std::vector<vk::DescriptorPoolSize> pool_sizes;
    pool_sizes.reserve(type_counts.size());
    for (const auto& [type, count] : type_counts) {
        pool_sizes.push_back({type, count});
    }

    // Default if no bindings
    if (pool_sizes.empty()) {
        pool_sizes.push_back({vk::DescriptorType::eStorageBuffer, 1024});
    }

    vk::DescriptorPoolCreateInfo pool_info{};
    pool_info.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    pool_info.maxSets = 1024;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();

    descriptor_pool_ = ctx_->device().createDescriptorPoolUnique(pool_info);
}

vk::DescriptorSet ComputePipeline::allocate_descriptor_set() const {
    vk::DescriptorSetAllocateInfo alloc_info{};
    alloc_info.descriptorPool = *descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &*descriptor_layout_;

    return ctx_->device().allocateDescriptorSets(alloc_info)[0];
}

std::vector<vk::DescriptorSet> ComputePipeline::allocate_descriptor_sets(uint32_t count) const {
    std::vector<vk::DescriptorSetLayout> layouts(count, *descriptor_layout_);

    vk::DescriptorSetAllocateInfo alloc_info{};
    alloc_info.descriptorPool = *descriptor_pool_;
    alloc_info.descriptorSetCount = count;
    alloc_info.pSetLayouts = layouts.data();

    return ctx_->device().allocateDescriptorSets(alloc_info);
}

void ComputePipeline::free_descriptor_set(vk::DescriptorSet set) const {
    ctx_->device().freeDescriptorSets(*descriptor_pool_, set);
}

void ComputePipeline::free_descriptor_sets(const std::vector<vk::DescriptorSet>& sets) const {
    ctx_->device().freeDescriptorSets(*descriptor_pool_, sets);
}

void ComputePipeline::update_descriptor(
    vk::DescriptorSet set,
    uint32_t binding,
    vk::Buffer buffer,
    vk::DeviceSize offset,
    vk::DeviceSize range) const {
    // Find binding type
    vk::DescriptorType type = vk::DescriptorType::eStorageBuffer;
    for (const auto& b : bindings_) {
        if (b.binding == binding) {
            type = b.type;
            break;
        }
    }

    vk::DescriptorBufferInfo buffer_info{};
    buffer_info.buffer = buffer;
    buffer_info.offset = offset;
    buffer_info.range = range;

    vk::WriteDescriptorSet write{};
    write.dstSet = set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = type;
    write.descriptorCount = 1;
    write.pBufferInfo = &buffer_info;

    ctx_->device().updateDescriptorSets(write, {});
}

void ComputePipeline::update_descriptors(
    vk::DescriptorSet set,
    const std::vector<std::tuple<uint32_t, vk::Buffer, vk::DeviceSize, vk::DeviceSize>>& updates)
    const {
    std::vector<vk::DescriptorBufferInfo> buffer_infos;
    buffer_infos.reserve(updates.size());

    std::vector<vk::WriteDescriptorSet> writes;
    writes.reserve(updates.size());

    for (const auto& [binding, buffer, offset, range] : updates) {
        // Find binding type
        vk::DescriptorType type = vk::DescriptorType::eStorageBuffer;
        for (const auto& b : bindings_) {
            if (b.binding == binding) {
                type = b.type;
                break;
            }
        }

        buffer_infos.push_back({buffer, offset, range});

        vk::WriteDescriptorSet write{};
        write.dstSet = set;
        write.dstBinding = binding;
        write.dstArrayElement = 0;
        write.descriptorType = type;
        write.descriptorCount = 1;
        write.pBufferInfo = &buffer_infos.back();
        writes.push_back(write);
    }

    ctx_->device().updateDescriptorSets(writes, {});
}

void ComputePipeline::bind(vk::CommandBuffer cmd) const {
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline_);
}

void ComputePipeline::bind_descriptor_set(
    vk::CommandBuffer cmd, vk::DescriptorSet set, uint32_t index) const {
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *layout_, index, set, {});
}

void ComputePipeline::dispatch(vk::CommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) const {
    cmd.dispatch(x, y, z);
}

void ComputePipeline::dispatch_for_count(vk::CommandBuffer cmd, uint32_t count) const {
    uint32_t groups = workgroup_count(count, local_size_[0]);
    cmd.dispatch(groups, 1, 1);
}

void ComputePipeline::dispatch_for_count_2d(
    vk::CommandBuffer cmd, uint32_t count_x, uint32_t count_y) const {
    uint32_t groups_x = workgroup_count(count_x, local_size_[0]);
    uint32_t groups_y = workgroup_count(count_y, local_size_[1]);
    cmd.dispatch(groups_x, groups_y, 1);
}

void ComputePipeline::dispatch_for_count_3d(
    vk::CommandBuffer cmd, uint32_t count_x, uint32_t count_y, uint32_t count_z) const {
    uint32_t groups_x = workgroup_count(count_x, local_size_[0]);
    uint32_t groups_y = workgroup_count(count_y, local_size_[1]);
    uint32_t groups_z = workgroup_count(count_z, local_size_[2]);
    cmd.dispatch(groups_x, groups_y, groups_z);
}

std::vector<uint32_t> ComputePipeline::load_spirv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open SPIR-V file: " + path);
    }

    size_t size = file.tellg();
    if (size % sizeof(uint32_t) != 0) {
        throw std::runtime_error("Invalid SPIR-V file size");
    }

    std::vector<uint32_t> code(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), size);

    return code;
}

uint32_t ComputePipeline::workgroup_count(uint32_t total, uint32_t local_size) {
    return (total + local_size - 1) / local_size;
}

// PipelineRegistry

PipelineRegistry::PipelineRegistry(const VulkanContext& ctx) : ctx_(&ctx) {}

ComputePipeline& PipelineRegistry::add(
    const std::string& name, const ComputePipeline::Config& config) {
    auto [it, inserted] =
        pipelines_.emplace(name, std::make_unique<ComputePipeline>(*ctx_, config));
    if (!inserted) {
        throw std::runtime_error("Pipeline already exists: " + name);
    }
    return *it->second;
}

ComputePipeline& PipelineRegistry::get(const std::string& name) {
    auto it = pipelines_.find(name);
    if (it == pipelines_.end()) {
        throw std::runtime_error("Pipeline not found: " + name);
    }
    return *it->second;
}

const ComputePipeline& PipelineRegistry::get(const std::string& name) const {
    auto it = pipelines_.find(name);
    if (it == pipelines_.end()) {
        throw std::runtime_error("Pipeline not found: " + name);
    }
    return *it->second;
}

bool PipelineRegistry::has(const std::string& name) const {
    return pipelines_.find(name) != pipelines_.end();
}

void PipelineRegistry::remove(const std::string& name) {
    pipelines_.erase(name);
}

void PipelineRegistry::clear() {
    pipelines_.clear();
}

std::vector<std::string> PipelineRegistry::names() const {
    std::vector<std::string> result;
    result.reserve(pipelines_.size());
    for (const auto& [name, _] : pipelines_) {
        result.push_back(name);
    }
    return result;
}

// BatchExecutor

BatchExecutor::BatchExecutor(const VulkanContext& ctx, const Config& config)
    : ctx_(&ctx), config_(config) {
    command_buffers_ = ctx.allocate_command_buffers(config.command_buffers);
    fences_.reserve(config.command_buffers);
    fence_submitted_.resize(config.command_buffers, false);

    for (uint32_t i = 0; i < config.command_buffers; i++) {
        fences_.push_back(ctx.create_fence(false));
    }
}

BatchExecutor::~BatchExecutor() {
    wait_all();
    for (auto fence : fences_) {
        ctx_->destroy_fence(fence);
    }
    ctx_->free_command_buffers(command_buffers_);
}

vk::CommandBuffer BatchExecutor::begin_batch(uint32_t batch_index) {
    auto cmd = command_buffers_[batch_index];
    cmd.reset();

    vk::CommandBufferBeginInfo begin_info{};
    begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cmd.begin(begin_info);

    return cmd;
}

void BatchExecutor::end_batch(uint32_t batch_index) {
    command_buffers_[batch_index].end();
}

void BatchExecutor::submit_batch(uint32_t batch_index) {
    ctx_->reset_fence(fences_[batch_index]);
    ctx_->submit(command_buffers_[batch_index], fences_[batch_index]);
    fence_submitted_[batch_index] = true;
}

void BatchExecutor::wait_batch(uint32_t batch_index) {
    if (fence_submitted_[batch_index]) {
        ctx_->wait_fence(fences_[batch_index]);
        fence_submitted_[batch_index] = false;
    }
}

bool BatchExecutor::is_batch_complete(uint32_t batch_index) const {
    if (!fence_submitted_[batch_index]) {
        return true;
    }
    auto result = ctx_->device().getFenceStatus(fences_[batch_index]);
    return result == vk::Result::eSuccess;
}

void BatchExecutor::wait_all() {
    for (uint32_t i = 0; i < command_buffers_.size(); i++) {
        wait_batch(i);
    }
}

}  // namespace mc189
