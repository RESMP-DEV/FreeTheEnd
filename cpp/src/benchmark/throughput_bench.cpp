/**
 * throughput_bench.cpp - Simulation throughput benchmarking harness
 *
 * Measures simulation steps/second for single and batched environments.
 * Target: 500K steps/sec minimum throughput.
 *
 * Output:
 *   - throughput_results.csv: batch_size, steps_per_second, latency_ms
 *   - memory_results.csv: GPU utilization metrics
 *   - shader_timings.csv: Per-shader execution times (if profiler enabled)
 */

#include <vulkan/vulkan.h>
#include "shader_profiler.hpp"
#include "../../include/mc189/simulator_api.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task.h>
#endif

namespace mcsim {
namespace benchmark {

//------------------------------------------------------------------------------
// Timing utilities
//------------------------------------------------------------------------------

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

struct TimingStats {
    double mean_ms = 0.0;
    double std_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    uint64_t total_steps = 0;
    double steps_per_second = 0.0;
};

TimingStats compute_stats(std::vector<double>& samples, uint64_t batch_size) {
    if (samples.empty()) return {};

    TimingStats stats;
    std::sort(samples.begin(), samples.end());

    // Basic stats
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.mean_ms = sum / samples.size();
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
    stats.p99_ms = samples[static_cast<size_t>(n * 0.99)];

    // Throughput
    stats.total_steps = samples.size() * batch_size;
    double total_time_sec = sum / 1000.0;
    stats.steps_per_second = stats.total_steps / total_time_sec;

    return stats;
}

//------------------------------------------------------------------------------
// Memory tracking
//------------------------------------------------------------------------------

struct MemoryMetrics {
    VkDeviceSize heap_budget = 0;
    VkDeviceSize heap_usage = 0;
    VkDeviceSize allocation_count = 0;
    VkDeviceSize buffer_memory = 0;
    double utilization_percent = 0.0;
};

MemoryMetrics query_memory_usage(VkPhysicalDevice phys_device, VkDevice device) {
    MemoryMetrics metrics;

    // Get memory properties
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_device, &mem_props);

    // Try to get memory budget if extension available
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_props{};
    budget_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

    VkPhysicalDeviceMemoryProperties2 mem_props2{};
    mem_props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    mem_props2.pNext = &budget_props;

    // This requires VK_EXT_memory_budget extension
    // vkGetPhysicalDeviceMemoryProperties2(phys_device, &mem_props2);

    // For now, use basic heap info
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            metrics.heap_budget += mem_props.memoryHeaps[i].size;
        }
    }

    return metrics;
}

//------------------------------------------------------------------------------
// GPU timestamp queries
//------------------------------------------------------------------------------

class GpuTimer {
public:
    bool init(VkDevice device, VkPhysicalDevice phys_device, uint32_t query_count = 256) {
        device_ = device;
        query_count_ = query_count;

        // Get timestamp period
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys_device, &props);
        timestamp_period_ns_ = props.limits.timestampPeriod;

        if (timestamp_period_ns_ == 0.0f) {
            fprintf(stderr, "[GpuTimer] Device does not support timestamps\n");
            return false;
        }

        // Create query pool
        VkQueryPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        pool_info.queryCount = query_count;

        if (vkCreateQueryPool(device, &pool_info, nullptr, &query_pool_) != VK_SUCCESS) {
            fprintf(stderr, "[GpuTimer] Failed to create query pool\n");
            return false;
        }

        timestamps_.resize(query_count);
        return true;
    }

    void destroy() {
        if (query_pool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device_, query_pool_, nullptr);
            query_pool_ = VK_NULL_HANDLE;
        }
    }

    void reset(VkCommandBuffer cmd, uint32_t first_query, uint32_t count) {
        vkCmdResetQueryPool(cmd, query_pool_, first_query, count);
    }

    void write_timestamp(VkCommandBuffer cmd, VkPipelineStageFlagBits stage, uint32_t query) {
        vkCmdWriteTimestamp(cmd, stage, query_pool_, query);
    }

    bool get_results(uint32_t first_query, uint32_t count) {
        VkResult result = vkGetQueryPoolResults(
            device_, query_pool_, first_query, count,
            count * sizeof(uint64_t), timestamps_.data() + first_query,
            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        return result == VK_SUCCESS;
    }

    double elapsed_ms(uint32_t start_query, uint32_t end_query) const {
        uint64_t start = timestamps_[start_query];
        uint64_t end = timestamps_[end_query];
        return (end - start) * timestamp_period_ns_ / 1e6;
    }

    VkQueryPool pool() const { return query_pool_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    uint32_t query_count_ = 0;
    float timestamp_period_ns_ = 0.0f;
    std::vector<uint64_t> timestamps_;
};

//------------------------------------------------------------------------------
// Benchmark configuration
//------------------------------------------------------------------------------

struct BenchmarkConfig {
    // Batch sizes to test
    std::vector<uint32_t> batch_sizes = {1, 10, 100, 1000};

    // Number of warmup iterations (not counted)
    uint32_t warmup_iterations = 100;

    // Number of measured iterations per batch size
    uint32_t measure_iterations = 1000;

    // Target throughput (steps/sec)
    double target_steps_per_sec = 500000.0;

    // Output file path
    std::string output_csv = "throughput_results.csv";
    std::string memory_csv = "memory_results.csv";
};

//------------------------------------------------------------------------------
// Benchmark results
//------------------------------------------------------------------------------

struct BatchResult {
    uint32_t batch_size;
    TimingStats cpu_timing;
    TimingStats gpu_timing;
    MemoryMetrics memory;
    bool meets_target;
};

struct BenchmarkResults {
    std::vector<BatchResult> batch_results;
    std::string device_name;
    uint64_t device_memory_bytes;
    double timestamp_period_ns;
};

//------------------------------------------------------------------------------
// CSV output
//------------------------------------------------------------------------------

void write_throughput_csv(const std::string& path, const BenchmarkResults& results) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[Benchmark] Failed to open %s for writing\n", path.c_str());
        return;
    }

    f << "batch_size,steps_per_second,latency_ms,latency_std_ms,";
    f << "latency_min_ms,latency_max_ms,p50_ms,p95_ms,p99_ms,";
    f << "gpu_latency_ms,meets_target\n";

    for (const auto& r : results.batch_results) {
        f << r.batch_size << ","
          << r.cpu_timing.steps_per_second << ","
          << r.cpu_timing.mean_ms << ","
          << r.cpu_timing.std_ms << ","
          << r.cpu_timing.min_ms << ","
          << r.cpu_timing.max_ms << ","
          << r.cpu_timing.p50_ms << ","
          << r.cpu_timing.p95_ms << ","
          << r.cpu_timing.p99_ms << ","
          << r.gpu_timing.mean_ms << ","
          << (r.meets_target ? "true" : "false") << "\n";
    }

    printf("[Benchmark] Wrote throughput results to %s\n", path.c_str());
}

void write_memory_csv(const std::string& path, const BenchmarkResults& results) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[Benchmark] Failed to open %s for writing\n", path.c_str());
        return;
    }

    f << "batch_size,heap_budget_mb,heap_usage_mb,buffer_memory_mb,";
    f << "allocation_count,utilization_percent\n";

    for (const auto& r : results.batch_results) {
        double budget_mb = r.memory.heap_budget / (1024.0 * 1024.0);
        double usage_mb = r.memory.heap_usage / (1024.0 * 1024.0);
        double buffer_mb = r.memory.buffer_memory / (1024.0 * 1024.0);

        f << r.batch_size << ","
          << budget_mb << ","
          << usage_mb << ","
          << buffer_mb << ","
          << r.memory.allocation_count << ","
          << r.memory.utilization_percent << "\n";
    }

    printf("[Benchmark] Wrote memory results to %s\n", path.c_str());
}

//------------------------------------------------------------------------------
// Mock simulation step (placeholder for actual shader dispatch)
//------------------------------------------------------------------------------

struct SimulationContext {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice phys_device = VK_NULL_HANDLE;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    // Pipeline for simulation (would contain actual shader)
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

    // Descriptor set for simulation state
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

    // State buffers
    VkBuffer state_buffer = VK_NULL_HANDLE;
    VkDeviceMemory state_memory = VK_NULL_HANDLE;
    VkDeviceSize state_buffer_size = 0;

    GpuTimer gpu_timer;
    uint32_t current_query = 0;
};

bool create_command_resources(SimulationContext& ctx, uint32_t queue_family) {
    // Command pool
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = queue_family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(ctx.device, &pool_info, nullptr, &ctx.cmd_pool) != VK_SUCCESS) {
        return false;
    }

    // Command buffer
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = ctx.cmd_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(ctx.device, &alloc_info, &ctx.cmd_buffer) != VK_SUCCESS) {
        return false;
    }

    // Fence for synchronization
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateFence(ctx.device, &fence_info, nullptr, &ctx.fence) != VK_SUCCESS) {
        return false;
    }

    return true;
}

void destroy_command_resources(SimulationContext& ctx) {
    if (ctx.fence != VK_NULL_HANDLE) {
        vkDestroyFence(ctx.device, ctx.fence, nullptr);
    }
    if (ctx.cmd_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(ctx.device, ctx.cmd_pool, nullptr);
    }
}

/**
 * Simulate a batch of environment steps.
 *
 * This is a placeholder that submits compute work. In production,
 * this would dispatch the actual simulation shaders.
 */
double step_environments_cpu_timed(SimulationContext& ctx, uint32_t batch_size) {
    auto start = Clock::now();

    // Reset fence
    vkResetFences(ctx.device, 1, &ctx.fence);

    // Begin command buffer
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResetCommandBuffer(ctx.cmd_buffer, 0);
    vkBeginCommandBuffer(ctx.cmd_buffer, &begin_info);

    // In real implementation: bind pipeline, descriptor sets, dispatch compute
    // For now, just a memory barrier as placeholder
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        ctx.cmd_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    vkEndCommandBuffer(ctx.cmd_buffer);

    // Submit
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &ctx.cmd_buffer;

    vkQueueSubmit(ctx.compute_queue, 1, &submit_info, ctx.fence);

    // Wait for completion
    vkWaitForFences(ctx.device, 1, &ctx.fence, VK_TRUE, UINT64_MAX);

    auto end = Clock::now();
    return Duration(end - start).count();
}

/**
 * Run throughput benchmark for a single batch size.
 */
BatchResult benchmark_batch_size(
    SimulationContext& ctx,
    const BenchmarkConfig& config,
    uint32_t batch_size) {

    BatchResult result;
    result.batch_size = batch_size;

    printf("[Benchmark] Testing batch_size=%u (%u warmup, %u measured)...\n",
           batch_size, config.warmup_iterations, config.measure_iterations);

    // Warmup
    for (uint32_t i = 0; i < config.warmup_iterations; ++i) {
        step_environments_cpu_timed(ctx, batch_size);
    }

    // Measured iterations
    std::vector<double> cpu_samples;
    cpu_samples.reserve(config.measure_iterations);

    for (uint32_t i = 0; i < config.measure_iterations; ++i) {
        double elapsed = step_environments_cpu_timed(ctx, batch_size);
        cpu_samples.push_back(elapsed);
    }

    // Compute stats
    result.cpu_timing = compute_stats(cpu_samples, batch_size);
    result.meets_target = result.cpu_timing.steps_per_second >= config.target_steps_per_sec;

    // Memory metrics
    result.memory = query_memory_usage(ctx.phys_device, ctx.device);

    printf("  -> %.2f steps/sec (target: %.0f) %s\n",
           result.cpu_timing.steps_per_second,
           config.target_steps_per_sec,
           result.meets_target ? "[PASS]" : "[FAIL]");
    printf("  -> latency: mean=%.3fms, p99=%.3fms\n",
           result.cpu_timing.mean_ms, result.cpu_timing.p99_ms);

    return result;
}

//------------------------------------------------------------------------------
// Main benchmark entry point
//------------------------------------------------------------------------------

class ThroughputBenchmark {
public:
    bool init(VkInstance instance, VkPhysicalDevice phys_device, VkDevice device,
              VkQueue compute_queue, uint32_t queue_family) {
        ctx_.device = device;
        ctx_.phys_device = phys_device;
        ctx_.compute_queue = compute_queue;

        if (!create_command_resources(ctx_, queue_family)) {
            fprintf(stderr, "[Benchmark] Failed to create command resources\n");
            return false;
        }

        if (!ctx_.gpu_timer.init(device, phys_device)) {
            fprintf(stderr, "[Benchmark] Warning: GPU timer not available\n");
        }

        // Get device info
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys_device, &props);
        device_name_ = props.deviceName;

        return true;
    }

    void destroy() {
        ctx_.gpu_timer.destroy();
        destroy_command_resources(ctx_);
    }

    BenchmarkResults run(const BenchmarkConfig& config) {
        BenchmarkResults results;
        results.device_name = device_name_;

        printf("\n========================================\n");
        printf("Throughput Benchmark\n");
        printf("Device: %s\n", device_name_.c_str());
        printf("Target: %.0f steps/sec\n", config.target_steps_per_sec);
        printf("========================================\n\n");

        for (uint32_t batch_size : config.batch_sizes) {
            BatchResult batch_result = benchmark_batch_size(ctx_, config, batch_size);
            results.batch_results.push_back(batch_result);
        }

        // Write results
        write_throughput_csv(config.output_csv, results);
        write_memory_csv(config.memory_csv, results);

        // Summary
        printf("\n========================================\n");
        printf("Summary\n");
        printf("========================================\n");

        bool all_pass = true;
        for (const auto& r : results.batch_results) {
            printf("batch_size=%4u: %10.0f steps/sec %s\n",
                   r.batch_size, r.cpu_timing.steps_per_second,
                   r.meets_target ? "[PASS]" : "[FAIL]");
            all_pass = all_pass && r.meets_target;
        }

        printf("\nOverall: %s\n", all_pass ? "ALL TARGETS MET" : "SOME TARGETS MISSED");

        return results;
    }

private:
    SimulationContext ctx_;
    std::string device_name_;
};

//------------------------------------------------------------------------------
// Process memory utilities
//------------------------------------------------------------------------------

#ifdef __APPLE__
static size_t get_process_memory_bytes() {
    struct task_basic_info info;
    mach_msg_type_number_t size = sizeof(info);
    kern_return_t kerr = task_info(mach_task_self(),
                                   TASK_BASIC_INFO,
                                   (task_info_t)&info,
                                   &size);
    return (kerr == KERN_SUCCESS) ? info.resident_size : 0;
}
#else
static size_t get_process_memory_bytes() {
    std::ifstream statm("/proc/self/statm");
    size_t size = 0, resident = 0;
    if (statm >> size >> resident) {
        return resident * sysconf(_SC_PAGESIZE);
    }
    return 0;
}
#endif

//------------------------------------------------------------------------------
// MC189 API-based benchmark (uses actual simulator)
//------------------------------------------------------------------------------

/**
 * MC189BenchmarkRunner - Benchmark using the actual simulation API
 *
 * This provides the most accurate throughput measurements as it uses
 * the same code path as production training.
 */
class MC189BenchmarkRunner {
public:
    struct Config {
        std::vector<uint32_t> batch_sizes = {1, 10, 100, 1000};
        uint32_t warmup_steps = 1000;
        uint32_t measurement_steps = 10000;
        uint32_t iterations = 3;
        double target_steps_per_sec = 500000.0;
        uint64_t seed = 42;
        std::string output_csv = "throughput_results.csv";
        std::string shader_csv = "shader_timings.csv";
        bool profile_shaders = true;
    };

    struct Result {
        uint32_t batch_size;
        uint64_t total_steps;
        double elapsed_sec;
        double steps_per_second;
        double latency_mean_ms;
        double latency_p50_ms;
        double latency_p95_ms;
        double latency_p99_ms;
        size_t gpu_memory_used;
        size_t gpu_memory_total;
        size_t cpu_memory_delta;
        double bandwidth_gbps;
        bool meets_target;
    };

    bool run(const Config& cfg, std::vector<Result>& out_results) {
        printf("\n");
        printf("================================================================\n");
        printf(" MC189 Throughput Benchmark\n");
        printf(" Target: %.0f steps/second\n", cfg.target_steps_per_sec);
        printf("================================================================\n\n");

        // Check GPU support
        if (!mc189_check_gpu_support()) {
            fprintf(stderr, "[Error] GPU does not support required features\n");
            return false;
        }

        for (uint32_t batch_size : cfg.batch_sizes) {
            printf("Batch size: %u\n", batch_size);

            std::vector<Result> iteration_results;
            for (uint32_t iter = 0; iter < cfg.iterations; ++iter) {
                printf("  Iteration %u/%u... ", iter + 1, cfg.iterations);
                fflush(stdout);

                Result r = run_single(cfg, batch_size);
                iteration_results.push_back(r);

                printf("%.0f steps/sec\n", r.steps_per_second);
            }

            // Average results
            Result avg = average_results(iteration_results);
            avg.meets_target = avg.steps_per_second >= cfg.target_steps_per_sec;
            out_results.push_back(avg);

            printf("  Average: %.0f steps/sec (%.3f ms latency) %s\n",
                   avg.steps_per_second, avg.latency_mean_ms,
                   avg.meets_target ? "[PASS]" : "[FAIL]");
            printf("  GPU: %.1f MB used, %.2f GB/s bandwidth\n",
                   avg.gpu_memory_used / (1024.0 * 1024.0),
                   avg.bandwidth_gbps);
            printf("\n");
        }

        // Write CSV
        write_results_csv(cfg.output_csv, out_results);

        return true;
    }

private:
    Result run_single(const Config& cfg, uint32_t batch_size) {
        Result result{};
        result.batch_size = batch_size;

        // Create simulator config
        mc189_config_t sim_cfg = mc189_default_config();
        sim_cfg.batch_size = batch_size;
        sim_cfg.deterministic_mode = true;
        sim_cfg.enable_validation_layers = false;
        sim_cfg.async_step = false;

        // Create simulator
        mc189_simulator_t sim = nullptr;
        mc189_error_t err = mc189_create(&sim_cfg, &sim);
        if (err != MC189_OK) {
            fprintf(stderr, "Failed to create simulator: %s\n",
                    mc189_get_error_message());
            return result;
        }

        // Allocate buffers
        std::vector<mc189_observation_t> obs(batch_size);
        std::vector<mc189_step_result_t> results(batch_size);
        std::vector<mc189_action_t> actions(batch_size);
        std::vector<uint64_t> seeds(batch_size);

        for (uint32_t i = 0; i < batch_size; ++i) {
            seeds[i] = cfg.seed + i;
            actions[i].action = MC189_ACTION_FORWARD;
            actions[i].flags = 0;
        }

        // Reset
        err = mc189_reset_batch(sim, seeds.data(), obs.data());
        if (err != MC189_OK) {
            fprintf(stderr, "Reset failed: %s\n", mc189_get_error_message());
            mc189_destroy(sim);
            return result;
        }

        size_t mem_baseline = get_process_memory_bytes();

        // Warmup
        for (uint32_t i = 0; i < cfg.warmup_steps; ++i) {
            mc189_step_batch(sim, actions.data(), results.data());
        }

        // Measurement
        std::vector<double> latencies;
        latencies.reserve(cfg.measurement_steps);

        auto start_time = Clock::now();

        for (uint32_t i = 0; i < cfg.measurement_steps; ++i) {
            auto step_start = Clock::now();

            err = mc189_step_batch(sim, actions.data(), results.data());
            if (err != MC189_OK) break;

            auto step_end = Clock::now();
            double step_ms = Duration(step_end - step_start).count();
            latencies.push_back(step_ms);

            // Auto-reset terminated envs
            for (uint32_t j = 0; j < batch_size; ++j) {
                if (results[j].terminated || results[j].truncated) {
                    mc189_observation_t tmp;
                    mc189_reset(sim, seeds[j], &tmp);
                }
            }
        }

        auto end_time = Clock::now();
        double elapsed_sec = Duration(end_time - start_time).count() / 1000.0;

        // Compute latency stats
        std::sort(latencies.begin(), latencies.end());
        size_t n = latencies.size();

        // Get memory stats from API
        mc189_stats_t stats;
        mc189_get_stats(sim, &stats);

        // Fill result
        result.total_steps = cfg.measurement_steps * batch_size;
        result.elapsed_sec = elapsed_sec;
        result.steps_per_second = result.total_steps / elapsed_sec;
        result.latency_mean_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / n;
        result.latency_p50_ms = latencies[n / 2];
        result.latency_p95_ms = latencies[static_cast<size_t>(n * 0.95)];
        result.latency_p99_ms = latencies[static_cast<size_t>(n * 0.99)];
        result.gpu_memory_used = stats.gpu_memory_used_bytes;
        result.gpu_memory_total = stats.gpu_memory_total_bytes;
        result.cpu_memory_delta = get_process_memory_bytes() - mem_baseline;

        // Estimate bandwidth (assume ~1KB state per env per step, read+write)
        double bytes_transferred = batch_size * 1024.0 * cfg.measurement_steps * 2.0;
        result.bandwidth_gbps = (bytes_transferred / elapsed_sec) / (1e9);

        mc189_destroy(sim);
        return result;
    }

    Result average_results(const std::vector<Result>& results) {
        if (results.empty()) return {};

        Result avg{};
        avg.batch_size = results[0].batch_size;

        for (const auto& r : results) {
            avg.total_steps += r.total_steps;
            avg.elapsed_sec += r.elapsed_sec;
            avg.steps_per_second += r.steps_per_second;
            avg.latency_mean_ms += r.latency_mean_ms;
            avg.latency_p50_ms += r.latency_p50_ms;
            avg.latency_p95_ms += r.latency_p95_ms;
            avg.latency_p99_ms += r.latency_p99_ms;
            avg.gpu_memory_used += r.gpu_memory_used;
            avg.gpu_memory_total += r.gpu_memory_total;
            avg.cpu_memory_delta += r.cpu_memory_delta;
            avg.bandwidth_gbps += r.bandwidth_gbps;
        }

        double n = static_cast<double>(results.size());
        avg.total_steps /= results.size();
        avg.elapsed_sec /= n;
        avg.steps_per_second /= n;
        avg.latency_mean_ms /= n;
        avg.latency_p50_ms /= n;
        avg.latency_p95_ms /= n;
        avg.latency_p99_ms /= n;
        avg.gpu_memory_used /= results.size();
        avg.gpu_memory_total /= results.size();
        avg.cpu_memory_delta /= results.size();
        avg.bandwidth_gbps /= n;

        return avg;
    }

    void write_results_csv(const std::string& path, const std::vector<Result>& results) {
        std::ofstream f(path);
        if (!f.is_open()) {
            fprintf(stderr, "Failed to write %s\n", path.c_str());
            return;
        }

        f << "batch_size,steps_per_second,latency_ms,latency_p50_ms,latency_p95_ms,";
        f << "latency_p99_ms,gpu_memory_mb,cpu_memory_mb,bandwidth_gbps,meets_target\n";

        for (const auto& r : results) {
            f << r.batch_size << ","
              << r.steps_per_second << ","
              << r.latency_mean_ms << ","
              << r.latency_p50_ms << ","
              << r.latency_p95_ms << ","
              << r.latency_p99_ms << ","
              << (r.gpu_memory_used / (1024.0 * 1024.0)) << ","
              << (r.cpu_memory_delta / (1024.0 * 1024.0)) << ","
              << r.bandwidth_gbps << ","
              << (r.meets_target ? 1 : 0) << "\n";
        }

        printf("Results written to %s\n", path.c_str());
    }
};

}  // namespace benchmark
}  // namespace mcsim

//------------------------------------------------------------------------------
// Standalone executable entry point
//------------------------------------------------------------------------------

#ifdef BENCHMARK_STANDALONE

int main(int argc, char** argv) {
    using namespace mcsim::benchmark;

    printf("MinecraftSim Throughput Benchmark\n");
    printf("=================================\n\n");

    // Parse arguments
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            config.warmup_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            config.measure_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
            config.target_steps_per_sec = atof(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            config.output_csv = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --warmup N      Warmup iterations (default: %u)\n", config.warmup_iterations);
            printf("  --iterations N  Measured iterations (default: %u)\n", config.measure_iterations);
            printf("  --target N      Target steps/sec (default: %.0f)\n", config.target_steps_per_sec);
            printf("  --output PATH   Output CSV path (default: %s)\n", config.output_csv.c_str());
            return 0;
        }
    }

    // Initialize Vulkan (simplified - would use VulkanInstance/VulkanDevice in production)
    printf("[Init] Creating Vulkan instance...\n");

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "MinecraftSim Benchmark";
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;

    VkInstance instance;
    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
        fprintf(stderr, "[Error] Failed to create Vulkan instance\n");
        return 1;
    }

    // Select physical device
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) {
        fprintf(stderr, "[Error] No Vulkan devices found\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

    // Pick first device with compute queue
    VkPhysicalDevice phys_device = VK_NULL_HANDLE;
    uint32_t queue_family = UINT32_MAX;

    for (auto dev : devices) {
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qf_props(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qf_count, qf_props.data());

        for (uint32_t i = 0; i < qf_count; ++i) {
            if (qf_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                phys_device = dev;
                queue_family = i;
                break;
            }
        }
        if (phys_device != VK_NULL_HANDLE) break;
    }

    if (phys_device == VK_NULL_HANDLE) {
        fprintf(stderr, "[Error] No compute-capable device found\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    // Create logical device
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;

    VkDevice device;
    if (vkCreateDevice(phys_device, &device_info, nullptr, &device) != VK_SUCCESS) {
        fprintf(stderr, "[Error] Failed to create logical device\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    VkQueue compute_queue;
    vkGetDeviceQueue(device, queue_family, 0, &compute_queue);

    // Run benchmark
    ThroughputBenchmark bench;
    if (!bench.init(instance, phys_device, device, compute_queue, queue_family)) {
        fprintf(stderr, "[Error] Failed to initialize benchmark\n");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    BenchmarkResults results = bench.run(config);

    // Cleanup
    bench.destroy();
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    // Return status based on targets
    bool all_pass = true;
    for (const auto& r : results.batch_results) {
        all_pass = all_pass && r.meets_target;
    }
    return all_pass ? 0 : 1;
}

#endif  // BENCHMARK_STANDALONE
