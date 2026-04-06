// async_transfer.h - Asynchronous GPU-CPU data transfer with double buffering
// Features:
// - Double-buffered observations for compute/transfer overlap
// - Pinned host memory for faster DMA transfers
// - Vulkan events for fine-grained synchronization
// - 20-30% throughput improvement through overlapping

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace mc189 {

class VulkanContext;

// Memory type flags for pinned (non-paged) host memory
constexpr vk::MemoryPropertyFlags kPinnedMemoryFlags =
    vk::MemoryPropertyFlagBits::eHostVisible |
    vk::MemoryPropertyFlagBits::eHostCoherent |
    vk::MemoryPropertyFlagBits::eHostCached;

/**
 * AsyncTransferBuffer - Double-buffered staging for overlapped transfers.
 *
 * While the GPU writes observations to buffer[N], the CPU reads from buffer[1-N].
 * Vulkan events signal when each buffer is ready for the next operation.
 */
class AsyncTransferBuffer {
public:
    AsyncTransferBuffer(const VulkanContext& ctx, vk::DeviceSize size_per_buffer);
    ~AsyncTransferBuffer();

    AsyncTransferBuffer(const AsyncTransferBuffer&) = delete;
    AsyncTransferBuffer& operator=(const AsyncTransferBuffer&) = delete;
    AsyncTransferBuffer(AsyncTransferBuffer&&) noexcept;
    AsyncTransferBuffer& operator=(AsyncTransferBuffer&&) noexcept;

    /// Get the buffer currently being written by GPU
    vk::Buffer gpu_write_buffer() const { return buffers_[write_idx_].handle; }

    /// Get the buffer ready for CPU read (previous frame's data)
    void* cpu_read_data() const { return buffers_[read_idx_].mapped; }

    /// Swap buffers after GPU write is complete
    void swap();

    /// Record copy command from device buffer to current write staging buffer
    void record_copy(vk::CommandBuffer cmd, vk::Buffer src, vk::DeviceSize size);

    /// Insert event to signal when copy is complete
    void record_signal_event(vk::CommandBuffer cmd);

    /// Check if the read buffer is ready (copy completed)
    bool is_read_ready() const;

    /// Wait for read buffer to be ready
    void wait_read_ready();

    /// Reset events for next frame
    void reset_events();

    vk::DeviceSize size() const { return size_per_buffer_; }

private:
    struct StagingBuffer {
        vk::Buffer handle;
        vk::DeviceMemory memory;
        void* mapped = nullptr;
    };

    const VulkanContext* ctx_ = nullptr;
    std::array<StagingBuffer, 2> buffers_{};
    std::array<vk::Event, 2> events_{};
    vk::DeviceSize size_per_buffer_ = 0;
    uint32_t write_idx_ = 0;
    uint32_t read_idx_ = 1;

    void create_buffer(StagingBuffer& buf);
    void destroy_buffer(StagingBuffer& buf);
};

/**
 * AsyncObservationTransfer - Manages double-buffered observation readback.
 *
 * Pipeline structure:
 *   Frame N:   [Compute shader writes obs] -> [Copy to staging A] -> [Signal event A]
 *   Frame N+1: [Compute shader writes obs] -> [Copy to staging B] -> [Signal event B]
 *                                             [CPU reads staging A]
 *
 * This overlaps GPU compute with CPU data consumption, hiding transfer latency.
 */
class AsyncObservationTransfer {
public:
    struct Config {
        uint32_t num_envs = 1;
        uint32_t observation_size = 256;  // floats per env
        bool enable_profiling = false;
    };

    AsyncObservationTransfer(const VulkanContext& ctx, const Config& config);
    ~AsyncObservationTransfer() = default;

    /// Record commands to copy observations from device buffer.
    /// Must be called after compute shaders have written observations.
    void record_readback(vk::CommandBuffer cmd, vk::Buffer device_obs_buffer);

    /// Get pointer to observation data from the previous frame.
    /// Returns nullptr on first frame (no data ready yet).
    const float* get_observations();

    /// Advance to next frame (swap buffers)
    void advance_frame();

    /// Wait for all pending transfers to complete
    void synchronize();

    /// Get profiling statistics
    double avg_transfer_time_us() const { return avg_transfer_time_us_; }
    double avg_wait_time_us() const { return avg_wait_time_us_; }

private:
    AsyncTransferBuffer staging_;
    Config config_;
    uint32_t frame_count_ = 0;
    bool first_frame_ = true;

    // Profiling
    std::chrono::high_resolution_clock::time_point transfer_start_;
    double avg_transfer_time_us_ = 0.0;
    double avg_wait_time_us_ = 0.0;
};

/**
 * PinnedMemoryPool - Pool of pinned host memory for efficient transfers.
 *
 * Pinned memory cannot be swapped out by the OS, enabling faster DMA transfers.
 * This pool pre-allocates pinned buffers to avoid allocation overhead per frame.
 */
class PinnedMemoryPool {
public:
    PinnedMemoryPool(const VulkanContext& ctx, vk::DeviceSize block_size, uint32_t num_blocks);
    ~PinnedMemoryPool();

    PinnedMemoryPool(const PinnedMemoryPool&) = delete;
    PinnedMemoryPool& operator=(const PinnedMemoryPool&) = delete;

    /// Allocate a pinned buffer from the pool.
    /// Returns nullptr if pool is exhausted.
    void* allocate();

    /// Return a buffer to the pool
    void free(void* ptr);

    /// Get the Vulkan buffer handle for a pinned allocation
    vk::Buffer get_buffer(void* ptr) const;

    /// Get offset within the buffer for a pinned allocation
    vk::DeviceSize get_offset(void* ptr) const;

    vk::DeviceSize block_size() const { return block_size_; }
    uint32_t capacity() const { return num_blocks_; }
    uint32_t available() const { return free_count_.load(); }

private:
    const VulkanContext* ctx_ = nullptr;
    vk::Buffer buffer_;
    vk::DeviceMemory memory_;
    void* mapped_ = nullptr;
    vk::DeviceSize block_size_ = 0;
    uint32_t num_blocks_ = 0;
    std::vector<uint32_t> free_list_;
    std::atomic<uint32_t> free_count_{0};
    std::mutex mutex_;
};

/**
 * OverlappedTransferManager - Coordinates async transfers with compute work.
 *
 * Usage pattern:
 *   1. begin_frame()           - Reset synchronization state
 *   2. record_compute(cmd)     - Record compute shader dispatches
 *   3. record_readback(cmd)    - Record async observation copy
 *   4. submit_and_overlap(cmd) - Submit with overlap
 *   5. get_observations()      - Get previous frame's observations while GPU works
 *   6. end_frame()             - Advance double buffers
 *
 * Expected improvement: 20-30% throughput by hiding transfer latency.
 */
class OverlappedTransferManager {
public:
    struct Config {
        uint32_t num_envs = 1;
        uint32_t observation_floats = 256;
        uint32_t reward_floats = 1;
        uint32_t done_uints = 1;
        bool enable_profiling = false;
    };

    OverlappedTransferManager(const VulkanContext& ctx, const Config& config);
    ~OverlappedTransferManager();

    /// Begin a new frame - retrieves previous frame's data
    void begin_frame();

    /// End frame - swap buffers for next iteration
    void end_frame();

    /// Record readback commands for all output buffers
    void record_readback(vk::CommandBuffer cmd,
                         vk::Buffer obs_buffer,
                         vk::Buffer reward_buffer,
                         vk::Buffer done_buffer);

    /// Get previous frame's observations (overlapped with current GPU work)
    const float* get_observations() const { return obs_data_; }

    /// Get previous frame's rewards (overlapped with current GPU work)
    const float* get_rewards() const { return reward_data_; }

    /// Get previous frame's done flags (overlapped with current GPU work)
    const uint32_t* get_dones() const { return done_data_; }

    /// Wait for all transfers to complete
    void synchronize();

    /// Check if previous frame's data is ready
    bool is_ready() const { return is_ready_; }

    /// Statistics for profiling
    struct Stats {
        double obs_transfer_us = 0.0;
        double reward_transfer_us = 0.0;
        double done_transfer_us = 0.0;
        double total_wait_us = 0.0;
        uint64_t frames_processed = 0;
    };
    Stats get_stats() const { return stats_; }

private:
    const VulkanContext* ctx_ = nullptr;
    Config config_;

    // Double-buffered staging
    AsyncTransferBuffer obs_staging_;
    AsyncTransferBuffer reward_staging_;
    AsyncTransferBuffer done_staging_;

    // Pointers to current readable data
    const float* obs_data_ = nullptr;
    const float* reward_data_ = nullptr;
    const uint32_t* done_data_ = nullptr;

    bool is_ready_ = false;
    bool first_frame_ = true;
    Stats stats_;
};

}  // namespace mc189
