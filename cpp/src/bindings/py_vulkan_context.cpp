// Target: contrib/minecraft_sim/cpp/src/bindings/py_vulkan_context.cpp
// Bindings for mc189::VulkanContext

#include "mc189/vulkan_context.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_vulkan_context(py::module_ &m) {
  // DeviceCapabilities struct
  py::class_<mc189::DeviceCapabilities>(m, "DeviceCapabilities")
      .def_readonly("device_name", &mc189::DeviceCapabilities::device_name)
      .def_readonly("vendor_id", &mc189::DeviceCapabilities::vendor_id)
      .def_readonly("device_local_memory",
                    &mc189::DeviceCapabilities::device_local_memory)
      .def_readonly("max_workgroup_size",
                    &mc189::DeviceCapabilities::max_workgroup_size)
      .def_readonly("max_compute_shared_memory",
                    &mc189::DeviceCapabilities::max_compute_shared_memory)
      .def_readonly("max_storage_buffer_range",
                    &mc189::DeviceCapabilities::max_storage_buffer_range)
      .def_readonly("supports_16bit_storage",
                    &mc189::DeviceCapabilities::supports_16bit_storage)
      .def_readonly("supports_8bit_storage",
                    &mc189::DeviceCapabilities::supports_8bit_storage);

  // VulkanContext::Config struct
  py::class_<mc189::VulkanContext::Config>(m, "VulkanContextConfig")
      .def(py::init<>())
      .def_readwrite("enable_validation",
                     &mc189::VulkanContext::Config::enable_validation)
      .def_readwrite("prefer_discrete_gpu",
                     &mc189::VulkanContext::Config::prefer_discrete_gpu)
      .def_readwrite("app_name", &mc189::VulkanContext::Config::app_name)
      .def_readwrite("app_version", &mc189::VulkanContext::Config::app_version);

  // VulkanContext class
  py::class_<mc189::VulkanContext>(m, "VulkanContext")
      .def(py::init<const mc189::VulkanContext::Config &>(),
           py::arg("config") = mc189::VulkanContext::Config{},
           "Create a Vulkan context with the given configuration")
      .def("capabilities", &mc189::VulkanContext::capabilities,
           py::return_value_policy::reference_internal,
           "Get device capabilities")
      .def_property_readonly(
          "is_valid",
          [](const mc189::VulkanContext &ctx) {
            return ctx.device() != nullptr;
          },
          "Check if context is valid");
}
