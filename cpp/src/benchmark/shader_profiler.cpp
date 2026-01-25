/**
 * shader_profiler.cpp - Individual shader execution timing
 *
 * Profiles each compute shader in the simulation pipeline to identify
 * bottlenecks and optimization opportunities.
 */

#include <vulkan/vulkan.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace mcsim {
namespace benchmark {

//------------------------------------------------------------------------------
// Shader profile data structures
//------------------------------------------------------------------------------

/**
 * Statistics for a single shader invocation.
 */
struct ShaderStats {
    std::string name;
    double mean_ms = 0.0;
    double std_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    uint64_t invocation_count = 0;
    double total_time_ms = 0.0;
    double percentage_of_frame = 0.0;

    // Work dimensions
    uint32_t work_group_size[3] = {0, 0, 0};
    uint32_t dispatch_size[3] = {0, 0, 0};
    uint64_t total_invocations = 0;
};

/**
 * Memory bandwidth metrics for a shader.
 */
struct ShaderMemoryMetrics {
    std::string shader_name;
    VkDeviceSize bytes_read = 0;
    VkDeviceSize bytes_written = 0;
    double bandwidth_gb_s = 0.0;
    double achieved_bandwidth_percent = 0.0;
};

/**
 * GPU occupancy metrics.
 */
struct OccupancyMetrics {
    std::string shader_name;
    uint32_t registers_per_thread = 0;
    uint32_t shared_memory_bytes = 0;
    uint32_t threads_per_block = 0;
    double theoretical_occupancy = 0.0;
    double achieved_occupancy = 0.0;
};

//------------------------------------------------------------------------------
// GPU timestamp query pool
//------------------------------------------------------------------------------

class ShaderTimestampPool {
public:
    static constexpr uint32_t MAX_QUERIES = 1024;
    static constexpr uint32_t QUERIES_PER_SHADER = 2;  // start + end

    bool init(VkDevice device, VkPhysicalDevice phys_device) {
        device_ = device;

        // Get timestamp period (ns per tick)
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys_device, &props);
        timestamp_period_ns_ = props.limits.timestampPeriod;

        if (timestamp_period_ns_ == 0.0f) {
            fprintf(stderr, "[ShaderProfiler] Device does not support timestamps\n");
            return false;
        }

        printf("[ShaderProfiler] Timestamp period: %.3f ns\n", timestamp_period_ns_);

        // Create query pool
        VkQueryPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        pool_info.queryCount = MAX_QUERIES;

        if (vkCreateQueryPool(device, &pool_info, nullptr, &query_pool_) != VK_SUCCESS) {
            fprintf(stderr, "[ShaderProfiler] Failed to create timestamp query pool\n");
            return false;
        }

        timestamps_.resize(MAX_QUERIES);
        return true;
    }

    void destroy() {
        if (query_pool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device_, query_pool_, nullptr);
            query_pool_ = VK_NULL_HANDLE;
        }
    }

    void reset_all(VkCommandBuffer cmd) {
        vkCmdResetQueryPool(cmd, query_pool_, 0, MAX_QUERIES);
        next_query_ = 0;
    }

    /**
     * Begin timing a shader. Returns the query index pair.
     */
    std::pair<uint32_t, uint32_t> begin_shader(VkCommandBuffer cmd,
                                                VkPipelineStageFlagBits stage) {
        if (next_query_ + 2 > MAX_QUERIES) {
            fprintf(stderr, "[ShaderProfiler] Query pool exhausted\n");
            return {UINT32_MAX, UINT32_MAX};
        }

        uint32_t start_query = next_query_++;
        vkCmdWriteTimestamp(cmd, stage, query_pool_, start_query);
        return {start_query, next_query_++};
    }

    void end_shader(VkCommandBuffer cmd, VkPipelineStageFlagBits stage,
                   uint32_t end_query) {
        vkCmdWriteTimestamp(cmd, stage, query_pool_, end_query);
    }

    bool fetch_results() {
        if (next_query_ == 0) return true;

        VkResult result = vkGetQueryPoolResults(
            device_, query_pool_, 0, next_query_,
            next_query_ * sizeof(uint64_t), timestamps_.data(),
            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

        return result == VK_SUCCESS;
    }

    double elapsed_ms(uint32_t start_query, uint32_t end_query) const {
        if (start_query >= timestamps_.size() || end_query >= timestamps_.size()) {
            return 0.0;
        }
        uint64_t start = timestamps_[start_query];
        uint64_t end = timestamps_[end_query];
        return (end - start) * timestamp_period_ns_ / 1e6;
    }

    VkQueryPool pool() const { return query_pool_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    float timestamp_period_ns_ = 0.0f;
    std::vector<uint64_t> timestamps_;
    uint32_t next_query_ = 0;
};

//------------------------------------------------------------------------------
// Pipeline statistics query (if supported)
//------------------------------------------------------------------------------

class PipelineStatsPool {
public:
    bool init(VkDevice device, uint32_t query_count = 64) {
        device_ = device;

        VkQueryPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        pool_info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
        pool_info.queryCount = query_count;
        pool_info.pipelineStatistics =
            VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;

        VkResult result = vkCreateQueryPool(device, &pool_info, nullptr, &query_pool_);
        if (result != VK_SUCCESS) {
            // Pipeline statistics may not be supported on all devices
            printf("[ShaderProfiler] Pipeline statistics not available\n");
            return false;
        }

        stats_.resize(query_count);
        return true;
    }

    void destroy() {
        if (query_pool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device_, query_pool_, nullptr);
            query_pool_ = VK_NULL_HANDLE;
        }
    }

    void begin_query(VkCommandBuffer cmd, uint32_t query) {
        vkCmdResetQueryPool(cmd, query_pool_, query, 1);
        vkCmdBeginQuery(cmd, query_pool_, query, 0);
    }

    void end_query(VkCommandBuffer cmd, uint32_t query) {
        vkCmdEndQuery(cmd, query_pool_, query);
    }

    bool fetch_results(uint32_t query_count) {
        VkResult result = vkGetQueryPoolResults(
            device_, query_pool_, 0, query_count,
            query_count * sizeof(uint64_t), stats_.data(),
            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        return result == VK_SUCCESS;
    }

    uint64_t get_invocations(uint32_t query) const {
        return query < stats_.size() ? stats_[query] : 0;
    }

    bool available() const { return query_pool_ != VK_NULL_HANDLE; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    std::vector<uint64_t> stats_;
};

//------------------------------------------------------------------------------
// Shader profiler
//------------------------------------------------------------------------------

class ShaderProfiler {
public:
    struct ProfileConfig {
        uint32_t warmup_iterations = 10;
        uint32_t profile_iterations = 100;
        bool collect_pipeline_stats = true;
        bool verbose = false;
    };

    struct ShaderInfo {
        std::string name;
        VkPipeline pipeline;
        VkPipelineLayout layout;
        VkDescriptorSet descriptor_set;
        uint32_t dispatch_x;
        uint32_t dispatch_y;
        uint32_t dispatch_z;
    };

    bool init(VkDevice device, VkPhysicalDevice phys_device, VkQueue queue,
              uint32_t queue_family) {
        device_ = device;
        phys_device_ = phys_device;
        queue_ = queue;

        if (!timestamp_pool_.init(device, phys_device)) {
            return false;
        }

        // Pipeline stats are optional
        stats_pool_.init(device);

        // Create command pool
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = queue_family;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(device, &pool_info, nullptr, &cmd_pool_) != VK_SUCCESS) {
            return false;
        }

        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = cmd_pool_;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &alloc_info, &cmd_buffer_) != VK_SUCCESS) {
            return false;
        }

        // Fence
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        if (vkCreateFence(device, &fence_info, nullptr, &fence_) != VK_SUCCESS) {
            return false;
        }

        return true;
    }

    void destroy() {
        if (fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(device_, fence_, nullptr);
        }
        if (cmd_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, cmd_pool_, nullptr);
        }
        stats_pool_.destroy();
        timestamp_pool_.destroy();
    }

    /**
     * Profile a single shader and collect statistics.
     */
    ShaderStats profile_shader(const ShaderInfo& shader, const ProfileConfig& config) {
        printf("[Profile] %s (%ux%ux%u dispatch)\n",
               shader.name.c_str(), shader.dispatch_x, shader.dispatch_y, shader.dispatch_z);

        std::vector<double> samples;
        samples.reserve(config.profile_iterations);

        // Warmup
        for (uint32_t i = 0; i < config.warmup_iterations; ++i) {
            run_shader_once(shader, false);
        }

        // Profiled runs
        for (uint32_t i = 0; i < config.profile_iterations; ++i) {
            double elapsed = run_shader_once(shader, true);
            samples.push_back(elapsed);

            if (config.verbose && i % 10 == 0) {
                printf("  iteration %u: %.4f ms\n", i, elapsed);
            }
        }

        // Compute statistics
        ShaderStats stats;
        stats.name = shader.name;
        stats.invocation_count = config.profile_iterations;
        stats.dispatch_size[0] = shader.dispatch_x;
        stats.dispatch_size[1] = shader.dispatch_y;
        stats.dispatch_size[2] = shader.dispatch_z;

        // Sort for percentiles
        std::sort(samples.begin(), samples.end());

        double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
        stats.mean_ms = sum / samples.size();
        stats.total_time_ms = sum;
        stats.min_ms = samples.front();
        stats.max_ms = samples.back();

        // Standard deviation
        double sq_sum = 0.0;
        for (double s : samples) {
            sq_sum += (s - stats.mean_ms) * (s - stats.mean_ms);
        }
        stats.std_ms = std::sqrt(sq_sum / samples.size());

        // Percentiles
        size_t n = samples.size();
        stats.p50_ms = samples[n / 2];
        stats.p95_ms = samples[static_cast<size_t>(n * 0.95)];
        stats.p99_ms = samples[std::min(static_cast<size_t>(n * 0.99), n - 1)];

        printf("  -> mean: %.4f ms, std: %.4f ms, p99: %.4f ms\n",
               stats.mean_ms, stats.std_ms, stats.p99_ms);

        return stats;
    }

    /**
     * Profile all shaders in the simulation pipeline.
     */
    std::vector<ShaderStats> profile_pipeline(const std::vector<ShaderInfo>& pipeline,
                                               const ProfileConfig& config) {
        std::vector<ShaderStats> results;
        double total_frame_time = 0.0;

        for (const auto& shader : pipeline) {
            ShaderStats stats = profile_shader(shader, config);
            total_frame_time += stats.mean_ms;
            results.push_back(stats);
        }

        // Compute percentage of frame time
        for (auto& stats : results) {
            stats.percentage_of_frame = (stats.mean_ms / total_frame_time) * 100.0;
        }

        return results;
    }

    /**
     * Write profiling results to CSV.
     */
    static void write_csv(const std::string& path, const std::vector<ShaderStats>& results) {
        std::ofstream f(path);
        if (!f.is_open()) {
            fprintf(stderr, "[ShaderProfiler] Failed to open %s\n", path.c_str());
            return;
        }

        f << "shader_name,mean_ms,std_ms,min_ms,max_ms,p50_ms,p95_ms,p99_ms,";
        f << "invocations,total_ms,frame_percent,dispatch_x,dispatch_y,dispatch_z\n";

        for (const auto& s : results) {
            f << s.name << ","
              << s.mean_ms << ","
              << s.std_ms << ","
              << s.min_ms << ","
              << s.max_ms << ","
              << s.p50_ms << ","
              << s.p95_ms << ","
              << s.p99_ms << ","
              << s.invocation_count << ","
              << s.total_time_ms << ","
              << s.percentage_of_frame << ","
              << s.dispatch_size[0] << ","
              << s.dispatch_size[1] << ","
              << s.dispatch_size[2] << "\n";
        }

        printf("[ShaderProfiler] Wrote results to %s\n", path.c_str());
    }

    /**
     * Print a summary table of shader timings.
     */
    static void print_summary(const std::vector<ShaderStats>& results) {
        printf("\n");
        printf("Shader Profile Summary\n");
        printf("============================================================\n");
        printf("%-30s %8s %8s %8s %6s\n", "Shader", "Mean", "Std", "P99", "Frame%");
        printf("------------------------------------------------------------\n");

        double total = 0.0;
        for (const auto& s : results) {
            printf("%-30s %7.3fms %7.3fms %7.3fms %5.1f%%\n",
                   s.name.c_str(), s.mean_ms, s.std_ms, s.p99_ms, s.percentage_of_frame);
            total += s.mean_ms;
        }

        printf("------------------------------------------------------------\n");
        printf("%-30s %7.3fms\n", "TOTAL", total);
        printf("%-30s %7.0f fps\n", "Max FPS", 1000.0 / total);
        printf("============================================================\n\n");
    }

private:
    double run_shader_once(const ShaderInfo& shader, bool profile) {
        vkResetFences(device_, 1, &fence_);

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkResetCommandBuffer(cmd_buffer_, 0);
        vkBeginCommandBuffer(cmd_buffer_, &begin_info);

        // Reset timestamp queries if profiling
        uint32_t start_query = 0, end_query = 0;
        if (profile) {
            timestamp_pool_.reset_all(cmd_buffer_);
            auto queries = timestamp_pool_.begin_shader(
                cmd_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            start_query = queries.first;
            end_query = queries.second;
        }

        // Bind and dispatch
        if (shader.pipeline != VK_NULL_HANDLE) {
            vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);

            if (shader.descriptor_set != VK_NULL_HANDLE) {
                vkCmdBindDescriptorSets(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        shader.layout, 0, 1, &shader.descriptor_set, 0, nullptr);
            }

            vkCmdDispatch(cmd_buffer_, shader.dispatch_x, shader.dispatch_y, shader.dispatch_z);
        } else {
            // Placeholder barrier for testing without actual shader
            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd_buffer_,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // End timestamp
        if (profile) {
            timestamp_pool_.end_shader(cmd_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, end_query);
        }

        vkEndCommandBuffer(cmd_buffer_);

        // Submit
        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &cmd_buffer_;

        vkQueueSubmit(queue_, 1, &submit_info, fence_);
        vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);

        if (profile) {
            timestamp_pool_.fetch_results();
            return timestamp_pool_.elapsed_ms(start_query, end_query);
        }

        return 0.0;
    }

    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    ShaderTimestampPool timestamp_pool_;
    PipelineStatsPool stats_pool_;
};

//------------------------------------------------------------------------------
// Predefined shader profiles for MinecraftSim
//------------------------------------------------------------------------------

namespace shaders {

/**
 * Known shader names in the MinecraftSim pipeline.
 * These would be populated from actual SPIR-V modules.
 */
const std::vector<std::string> PIPELINE_ORDER = {
    "terrain_gen",           // Terrain/chunk generation
    "physics_step",          // Entity physics simulation
    "collision_broad",       // Broad-phase collision detection
    "collision_narrow",      // Narrow-phase collision detection
    "mob_ai",                // Mob AI decision making
    "player_input",          // Player input processing
    "block_updates",         // Block state updates (redstone, etc.)
    "liquid_flow",           // Water/lava flow simulation
    "lighting_update",       // Light propagation
    "random_tick",           // Random tick processing
    "entity_tick",           // Entity tick updates
    "hunger_tick",           // Hunger/saturation updates
    "damage_calc",           // Damage calculation
    "xp_calc",               // Experience calculation
    "inventory_sync",        // Inventory state sync
};

/**
 * Expected timing budgets for 60 FPS (16.67ms frame budget).
 */
const std::map<std::string, double> TIMING_BUDGETS_MS = {
    {"terrain_gen", 2.0},
    {"physics_step", 3.0},
    {"collision_broad", 1.0},
    {"collision_narrow", 2.0},
    {"mob_ai", 2.0},
    {"player_input", 0.5},
    {"block_updates", 1.5},
    {"liquid_flow", 1.0},
    {"lighting_update", 1.5},
    {"random_tick", 0.5},
    {"entity_tick", 1.0},
    {"hunger_tick", 0.1},
    {"damage_calc", 0.2},
    {"xp_calc", 0.1},
    {"inventory_sync", 0.5},
};

}  // namespace shaders

}  // namespace benchmark
}  // namespace mcsim

//------------------------------------------------------------------------------
// Standalone entry point
//------------------------------------------------------------------------------

#ifdef SHADER_PROFILER_STANDALONE

int main(int argc, char** argv) {
    using namespace mcsim::benchmark;

    printf("MinecraftSim Shader Profiler\n");
    printf("============================\n\n");

    ShaderProfiler::ProfileConfig config;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            config.warmup_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            config.profile_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            config.verbose = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --warmup N      Warmup iterations (default: %u)\n",
                   config.warmup_iterations);
            printf("  --iterations N  Profile iterations (default: %u)\n",
                   config.profile_iterations);
            printf("  --verbose       Verbose output\n");
            return 0;
        }
    }

    // Initialize Vulkan
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Shader Profiler";
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;

    VkInstance instance;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create Vulkan instance\n");
        return 1;
    }

    // Find device
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) {
        fprintf(stderr, "No Vulkan devices found\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    VkPhysicalDevice phys_device = VK_NULL_HANDLE;
    uint32_t queue_family = UINT32_MAX;

    for (auto dev : devices) {
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qf_props(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qf_count, qf_props.data());

        for (uint32_t i = 0; i < qf_count; ++i) {
            if ((qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                qf_props[i].timestampValidBits > 0) {
                phys_device = dev;
                queue_family = i;
                break;
            }
        }
        if (phys_device != VK_NULL_HANDLE) break;
    }

    if (phys_device == VK_NULL_HANDLE) {
        fprintf(stderr, "No compute device with timestamp support found\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    // Print device info
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys_device, &props);
    printf("Device: %s\n", props.deviceName);
    printf("Timestamp period: %.3f ns\n\n", props.limits.timestampPeriod);

    // Create logical device with pipeline stats feature
    VkPhysicalDeviceFeatures features{};
    features.pipelineStatisticsQuery = VK_TRUE;

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_info;
    device_create_info.pEnabledFeatures = &features;

    VkDevice device;
    if (vkCreateDevice(phys_device, &device_create_info, nullptr, &device) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create logical device\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    VkQueue queue;
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    // Initialize profiler
    ShaderProfiler profiler;
    if (!profiler.init(device, phys_device, queue, queue_family)) {
        fprintf(stderr, "Failed to initialize shader profiler\n");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    // Create mock shader infos (in production, these would be actual pipelines)
    std::vector<ShaderProfiler::ShaderInfo> pipeline;
    for (const auto& name : shaders::PIPELINE_ORDER) {
        ShaderProfiler::ShaderInfo info;
        info.name = name;
        info.pipeline = VK_NULL_HANDLE;  // Would be actual pipeline
        info.layout = VK_NULL_HANDLE;
        info.descriptor_set = VK_NULL_HANDLE;
        info.dispatch_x = 256;
        info.dispatch_y = 1;
        info.dispatch_z = 1;
        pipeline.push_back(info);
    }

    // Run profiling
    std::vector<ShaderStats> results = profiler.profile_pipeline(pipeline, config);

    // Print and save results
    ShaderProfiler::print_summary(results);
    ShaderProfiler::write_csv("shader_profile_results.csv", results);

    // Cleanup
    profiler.destroy();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return 0;
}

#endif  // SHADER_PROFILER_STANDALONE
