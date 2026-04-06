// Target: contrib/minecraft_sim/cpp/src/bindings/py_compute_pipeline.cpp
// Bindings for mc189::ComputePipeline, BatchExecutor

#include "mc189/compute_pipeline.h"
#include "mc189/vulkan_context.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_compute_pipeline(py::module_ &m) {
  // ComputePipeline::Config
  py::class_<mc189::ComputePipeline::Config>(m, "ComputePipelineConfig")
      .def(py::init<>())
      .def_readwrite("entry_point",
                     &mc189::ComputePipeline::Config::entry_point)
      .def_readwrite("local_size_x",
                     &mc189::ComputePipeline::Config::local_size_x)
      .def_readwrite("local_size_y",
                     &mc189::ComputePipeline::Config::local_size_y)
      .def_readwrite("local_size_z",
                     &mc189::ComputePipeline::Config::local_size_z)
      .def(
          "set_spirv",
          [](mc189::ComputePipeline::Config &cfg, py::bytes spirv_code) {
            std::string spirv_str = spirv_code;
            cfg.spirv_code.assign(
                reinterpret_cast<const uint32_t *>(spirv_str.data()),
                reinterpret_cast<const uint32_t *>(spirv_str.data() +
                                                   spirv_str.size()));
          },
          "Set SPIR-V bytecode from Python bytes");

  // ComputePipeline class
  py::class_<mc189::ComputePipeline>(m, "ComputePipeline")
      .def(py::init<const mc189::VulkanContext &,
                    const mc189::ComputePipeline::Config &>(),
           py::arg("ctx"), py::arg("config"),
           "Create a compute pipeline from configuration")
      .def_static("workgroup_count", &mc189::ComputePipeline::workgroup_count,
                  py::arg("total"), py::arg("local_size"),
                  "Calculate number of workgroups needed");

  // BatchExecutor::Config
  py::class_<mc189::BatchExecutor::Config>(m, "BatchExecutorConfig")
      .def(py::init<>())
      .def_readwrite("max_batch_size",
                     &mc189::BatchExecutor::Config::max_batch_size)
      .def_readwrite("command_buffers",
                     &mc189::BatchExecutor::Config::command_buffers)
      .def_readwrite("async_execution",
                     &mc189::BatchExecutor::Config::async_execution);

  // BatchExecutor class
  py::class_<mc189::BatchExecutor>(m, "BatchExecutor")
      .def(py::init<const mc189::VulkanContext &,
                    const mc189::BatchExecutor::Config &>(),
           py::arg("ctx"), py::arg("config") = mc189::BatchExecutor::Config{})
      .def_property_readonly("max_batch_size",
                             &mc189::BatchExecutor::max_batch_size)
      .def_property_readonly("num_command_buffers",
                             &mc189::BatchExecutor::num_command_buffers)
      .def("wait_all", &mc189::BatchExecutor::wait_all,
           py::call_guard<py::gil_scoped_release>(),
           "Wait for all pending executions to complete");
}
