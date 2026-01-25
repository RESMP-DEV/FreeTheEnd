// Main pybind11 module for mc189

#include <fstream>
#include <stdexcept>
#include <string>

#include "mc189/vulkan_context.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward declarations for sub-module initializers
void init_vulkan_context(py::module_ &m);
void init_buffer_manager(py::module_ &m);
void init_compute_pipeline(py::module_ &m);
void init_simulator(py::module_ &m);

namespace {

std::string read_binary_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open SPIR-V file: " + path);
  }

  file.seekg(0, std::ios::end);
  const std::streamsize size = file.tellg();
  if (size < 0) {
    throw std::runtime_error("Failed to read SPIR-V file size: " + path);
  }
  file.seekg(0, std::ios::beg);

  std::string buffer(static_cast<size_t>(size), '\0');
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Failed to read SPIR-V file: " + path);
  }

  return buffer;
}

} // namespace

PYBIND11_MODULE(mc189_core, m) {
  m.doc() = "Minecraft 1.8.9 Vulkan compute backend";
  m.attr("__version__") = "1.0.0";

  init_vulkan_context(m);
  init_buffer_manager(m);
  init_compute_pipeline(m);
  init_simulator(m);

  m.def(
      "create_context",
      [](bool enable_validation, bool prefer_discrete) {
        mc189::VulkanContext::Config config{};
        config.enable_validation = enable_validation;
        config.prefer_discrete_gpu = prefer_discrete;
        return mc189::VulkanContext(config);
      },
      py::arg("enable_validation") = false, py::arg("prefer_discrete") = true,
      "Create a VulkanContext with optional validation and device preference.");

  m.def(
      "load_shader",
      [](const std::string &path) { return py::bytes(read_binary_file(path)); },
      py::arg("path"), "Load a SPIR-V shader file and return its bytes.");

  m.def(
      "get_device_info",
      []() {
        // Create a temporary context to get device info
        mc189::VulkanContext::Config config{};
        mc189::VulkanContext ctx(config);
        auto caps = ctx.capabilities();
        py::dict info;
        info["device_name"] = caps.device_name;
        info["vendor_id"] = caps.vendor_id;
        info["device_local_memory"] = caps.device_local_memory;
        info["max_workgroup_size"] = caps.max_workgroup_size;
        info["max_compute_shared_memory"] = caps.max_compute_shared_memory;
        info["max_storage_buffer_range"] = caps.max_storage_buffer_range;
        info["supports_16bit_storage"] = caps.supports_16bit_storage;
        info["supports_8bit_storage"] = caps.supports_8bit_storage;
        return info;
      },
      "Return information about available Vulkan devices.");
}
