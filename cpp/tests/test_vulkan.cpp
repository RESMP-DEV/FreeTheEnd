#include "mc189/vulkan_context.h"
#include "mc189/compute_pipeline.h"
#include "mc189/buffer_manager.h"
#include <cassert>
#include <cmath>
#include <iostream>

#define TEST(name) \
    std::cout << "Testing " #name "..." << std::flush; \
    test_##name(); \
    std::cout << " OK\n"

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "\nAssertion failed: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
            std::abort(); \
        } \
    } while (0)

#define ASSERT_EQ(a, b) \
    do { \
        auto _a = (a); \
        auto _b = (b); \
        if (_a != _b) { \
            std::cerr << "\nAssertion failed: " #a " == " #b << "\n"; \
            std::cerr << "  " #a " = " << _a << "\n"; \
            std::cerr << "  " #b " = " << _b << "\n"; \
            std::cerr << "  at " __FILE__ ":" << __LINE__ << "\n"; \
            std::abort(); \
        } \
    } while (0)

#define ASSERT_NEAR(a, b, eps) \
    do { \
        auto _a = (a); \
        auto _b = (b); \
        if (std::abs(_a - _b) > (eps)) { \
            std::cerr << "\nAssertion failed: " #a " ~= " #b << "\n"; \
            std::cerr << "  " #a " = " << _a << "\n"; \
            std::cerr << "  " #b " = " << _b << "\n"; \
            std::cerr << "  diff = " << std::abs(_a - _b) << "\n"; \
            std::cerr << "  at " __FILE__ ":" << __LINE__ << "\n"; \
            std::abort(); \
        } \
    } while (0)

namespace {

mc189::VulkanContext* g_ctx = nullptr;

void test_context_creation() {
    mc189::VulkanContext::Config config;
    config.enable_validation = false;
    config.prefer_discrete_gpu = true;

    mc189::VulkanContext ctx(config);

    ASSERT(ctx.instance());
    ASSERT(ctx.physical_device());
    ASSERT(ctx.device());
    ASSERT(ctx.compute_queue());
    ASSERT(ctx.command_pool());

    const auto& caps = ctx.capabilities();
    ASSERT(!caps.device_name.empty());
    ASSERT(caps.max_workgroup_size[0] >= 64);
    ASSERT(caps.device_local_memory > 0);

    g_ctx = new mc189::VulkanContext(std::move(ctx));
}

void test_buffer_creation() {
    ASSERT(g_ctx != nullptr);
    mc189::BufferManager mgr(*g_ctx);

    // Device buffer
    auto device_buf = mgr.create_device_buffer(1024 * 1024);
    ASSERT(device_buf.handle());
    ASSERT_EQ(device_buf.size(), 1024 * 1024);
    ASSERT_EQ(device_buf.location(), mc189::MemoryLocation::Device);

    // Staging buffer
    auto staging_buf = mgr.create_staging_buffer(1024);
    ASSERT(staging_buf.handle());
    ASSERT(staging_buf.is_mapped());
    ASSERT_EQ(staging_buf.location(), mc189::MemoryLocation::Staging);

    // Host buffer
    auto host_buf = mgr.create_mapped_buffer(2048);
    ASSERT(host_buf.handle());
    ASSERT(host_buf.is_mapped());
    ASSERT(host_buf.data<void>() != nullptr);
}

void test_buffer_transfer() {
    ASSERT(g_ctx != nullptr);
    mc189::BufferManager mgr(*g_ctx);

    constexpr size_t kSize = 1024;
    std::vector<float> input(kSize);
    for (size_t i = 0; i < kSize; i++) {
        input[i] = static_cast<float>(i) * 0.5f;
    }

    // Create device buffer and upload
    auto device_buf = mgr.create_device_buffer(kSize * sizeof(float));
    mgr.upload(device_buf, input);

    // Download and verify
    std::vector<float> output(kSize);
    mgr.download(device_buf, output);

    for (size_t i = 0; i < kSize; i++) {
        ASSERT_NEAR(output[i], input[i], 1e-6f);
    }
}

void test_command_buffer() {
    ASSERT(g_ctx != nullptr);

    auto cmd = g_ctx->allocate_command_buffer();
    ASSERT(cmd);

    vk::CommandBufferBeginInfo begin_info{};
    cmd.begin(begin_info);
    cmd.end();

    g_ctx->submit_and_wait(cmd);
    g_ctx->free_command_buffer(cmd);

    // Multiple command buffers
    auto cmds = g_ctx->allocate_command_buffers(4);
    ASSERT_EQ(cmds.size(), 4);
    g_ctx->free_command_buffers(cmds);
}

void test_fence() {
    ASSERT(g_ctx != nullptr);

    auto fence = g_ctx->create_fence(false);
    ASSERT(fence);

    auto cmd = g_ctx->allocate_command_buffer();
    vk::CommandBufferBeginInfo begin_info{};
    cmd.begin(begin_info);
    cmd.end();

    g_ctx->submit(cmd, fence);
    g_ctx->wait_fence(fence);
    g_ctx->reset_fence(fence);
    g_ctx->destroy_fence(fence);

    g_ctx->free_command_buffer(cmd);

    // Signaled fence
    auto signaled = g_ctx->create_fence(true);
    g_ctx->wait_fence(signaled);  // Should return immediately
    g_ctx->destroy_fence(signaled);
}

void test_ring_buffer() {
    ASSERT(g_ctx != nullptr);

    mc189::RingBuffer ring(*g_ctx, 1024 * 1024);
    ASSERT_EQ(ring.capacity(), 1024 * 1024);
    ASSERT_EQ(ring.used(), 0);

    auto alloc1 = ring.allocate(256);
    ASSERT_EQ(alloc1.offset, 0);
    ASSERT_EQ(alloc1.size, 256);
    ASSERT(alloc1.data != nullptr);
    ASSERT_EQ(ring.used(), 256);

    // Aligned allocation
    auto alloc2 = ring.allocate(512, 256);
    ASSERT_EQ(alloc2.offset, 256);  // Already aligned
    ASSERT_EQ(alloc2.size, 512);

    ring.reset();
    ASSERT_EQ(ring.used(), 0);
}

void test_environment_pool() {
    ASSERT(g_ctx != nullptr);
    mc189::BufferManager mgr(*g_ctx);

    mc189::EnvironmentBufferPool::Config config;
    config.max_environments = 1024;
    config.state_size_per_env = 256;
    config.action_size_per_env = 32;
    config.reward_size_per_env = 4;
    config.double_buffer = true;

    mc189::EnvironmentBufferPool pool(*g_ctx, mgr, config);

    ASSERT_EQ(pool.max_environments(), 1024);
    ASSERT_EQ(pool.current_buffer_index(), 0);

    ASSERT(pool.states(0).handle());
    ASSERT(pool.actions(0).handle());
    ASSERT(pool.rewards(0).handle());
    ASSERT(pool.dones().handle());

    // Double buffer exists
    ASSERT(pool.states(1).handle());

    pool.swap_buffers();
    ASSERT_EQ(pool.current_buffer_index(), 1);

    pool.swap_buffers();
    ASSERT_EQ(pool.current_buffer_index(), 0);
}

void test_batch_executor() {
    ASSERT(g_ctx != nullptr);

    mc189::BatchExecutor::Config config;
    config.max_batch_size = 32768;
    config.command_buffers = 2;
    config.async_execution = false;

    mc189::BatchExecutor executor(*g_ctx, config);
    ASSERT_EQ(executor.max_batch_size(), 32768);
    ASSERT_EQ(executor.num_command_buffers(), 2);

    // Execute empty batch
    executor.execute([](vk::CommandBuffer cmd) {
        // No-op
        (void)cmd;
    });

    executor.wait_all();
}

void test_workgroup_count() {
    ASSERT_EQ(mc189::ComputePipeline::workgroup_count(256, 64), 4);
    ASSERT_EQ(mc189::ComputePipeline::workgroup_count(257, 64), 5);
    ASSERT_EQ(mc189::ComputePipeline::workgroup_count(1, 256), 1);
    ASSERT_EQ(mc189::ComputePipeline::workgroup_count(32768, 256), 128);
}

void test_memory_barriers() {
    ASSERT(g_ctx != nullptr);
    mc189::BufferManager mgr(*g_ctx);

    auto buf = mgr.create_device_buffer(1024);

    auto cmd = g_ctx->allocate_command_buffer();
    vk::CommandBufferBeginInfo begin_info{};
    cmd.begin(begin_info);

    mc189::BufferManager::barrier_compute_to_compute(cmd, buf);
    mc189::BufferManager::barrier_compute_to_transfer(cmd, buf);
    mc189::BufferManager::barrier_transfer_to_compute(cmd, buf);

    cmd.end();
    g_ctx->submit_and_wait(cmd);
    g_ctx->free_command_buffer(cmd);
}

}  // namespace

int main() {
    std::cout << "mc189 Vulkan Tests\n";
    std::cout << "==================\n\n";

    try {
        TEST(context_creation);
        TEST(buffer_creation);
        TEST(buffer_transfer);
        TEST(command_buffer);
        TEST(fence);
        TEST(ring_buffer);
        TEST(environment_pool);
        TEST(batch_executor);
        TEST(workgroup_count);
        TEST(memory_barriers);

        std::cout << "\nAll tests passed!\n";

        delete g_ctx;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << "\n";
        delete g_ctx;
        return 1;
    }
}
