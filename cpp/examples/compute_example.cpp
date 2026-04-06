/**
 * @file compute_example.cpp
 * @brief Example demonstrating Vulkan compute shader execution for Minecraft
 * simulation
 */

#include "mc189/buffer_manager.h"
#include "mc189/compute_pipeline.h"
#include "mc189/vulkan_context.h"
#include <iostream>
#include <vector>

int main() {
  std::cout << "MC189 Vulkan Compute Example\n";
  std::cout << "============================\n\n";

  try {
    // Initialize Vulkan context
    mc189::VulkanContext::Config config{};
    config.enable_validation = false;
    config.prefer_discrete_gpu = true;
    mc189::VulkanContext ctx(config);
    std::cout << "Vulkan context created.\n";
    std::cout << "  Instance: " << (ctx.instance() ? "OK" : "FAILED") << "\n";
    std::cout << "  Physical Device: "
              << (ctx.physical_device() ? "OK" : "FAILED") << "\n";
    std::cout << "  Logical Device: " << (ctx.device() ? "OK" : "FAILED")
              << "\n";

    // Create buffer manager
    std::cout << "\nCreating buffer manager...\n";
    mc189::BufferManager buffers(ctx);

    // Allocate test buffers
    constexpr size_t NUM_ELEMENTS = 1024;
    constexpr size_t BUFFER_SIZE = NUM_ELEMENTS * sizeof(float);

    auto input_buffer = buffers.create_buffer(
        BUFFER_SIZE,
        mc189::BufferUsage::Storage | mc189::BufferUsage::TransferDst,
        mc189::MemoryLocation::Device);

    auto output_buffer = buffers.create_buffer(
        BUFFER_SIZE,
        mc189::BufferUsage::Storage | mc189::BufferUsage::TransferSrc,
        mc189::MemoryLocation::Device);

    std::cout << "  Input buffer: " << BUFFER_SIZE << " bytes\n";
    std::cout << "  Output buffer: " << BUFFER_SIZE << " bytes\n";

    std::cout << "\nVulkan compute backend initialized successfully!\n";
    std::cout << "Ready for Minecraft 1.8.9 simulation.\n";

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
