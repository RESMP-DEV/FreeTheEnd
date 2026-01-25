// Bindings for mc189::BufferManager, Buffer, RingBuffer, EnvironmentBufferPool

#include "mc189/buffer_manager.h"
#include "mc189/vulkan_context.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_buffer_manager(py::module_ &m) {
  // BufferUsage enum
  py::enum_<mc189::BufferUsage>(m, "BufferUsage", py::arithmetic())
      .value("Storage", mc189::BufferUsage::Storage)
      .value("Uniform", mc189::BufferUsage::Uniform)
      .value("TransferSrc", mc189::BufferUsage::TransferSrc)
      .value("TransferDst", mc189::BufferUsage::TransferDst)
      .value("Indirect", mc189::BufferUsage::Indirect)
      .export_values();

  // MemoryLocation enum
  py::enum_<mc189::MemoryLocation>(m, "MemoryLocation")
      .value("Device", mc189::MemoryLocation::Device)
      .value("Host", mc189::MemoryLocation::Host)
      .value("Staging", mc189::MemoryLocation::Staging)
      .export_values();

  // Buffer class
  py::class_<mc189::Buffer>(m, "Buffer")
      .def_property_readonly("handle",
                             [](const mc189::Buffer &b) {
                               return reinterpret_cast<uint64_t>(
                                   static_cast<VkBuffer>(b.handle()));
                             })
      .def_property_readonly("size", &mc189::Buffer::size)
      .def_property_readonly("location", &mc189::Buffer::location)
      .def_property_readonly("is_mapped", &mc189::Buffer::is_mapped)
      .def("map", &mc189::Buffer::map, py::return_value_policy::reference)
      .def("unmap", &mc189::Buffer::unmap)
      .def(
          "as_numpy",
          [](mc189::Buffer &buf, const std::string &dtype) {
            if (!buf.is_mapped()) {
              throw std::runtime_error(
                  "Buffer must be mapped to get numpy view");
            }
            py::dtype dt(dtype);
            size_t itemsize = dt.itemsize();
            size_t count = buf.size() / itemsize;
            return py::array(dt, {count}, {itemsize}, buf.data<void>(),
                             py::cast(buf));
          },
          py::arg("dtype") = "float32");

  // BufferManager class
  // Note: keep_alive<0, 1> ensures VulkanContext outlives BufferManager
  py::class_<mc189::BufferManager>(m, "BufferManager")
      .def(py::init<const mc189::VulkanContext &>(), py::keep_alive<1, 2>())
      .def("create_buffer", &mc189::BufferManager::create_buffer,
           py::arg("size"), py::arg("usage"), py::arg("location"),
           py::keep_alive<0, 1>()) // Buffer outlived by manager
      .def("create_device_buffer", &mc189::BufferManager::create_device_buffer,
           py::arg("size"), py::arg("usage") = mc189::BufferUsage::Storage,
           py::keep_alive<0, 1>())
      .def("create_staging_buffer",
           &mc189::BufferManager::create_staging_buffer, py::arg("size"),
           py::keep_alive<0, 1>())
      .def("create_mapped_buffer", &mc189::BufferManager::create_mapped_buffer,
           py::arg("size"), py::arg("usage") = mc189::BufferUsage::Storage,
           py::keep_alive<0, 1>())
      .def(
          "upload",
          [](mc189::BufferManager &mgr, mc189::Buffer &dst,
             py::array_t<float> data, size_t offset) {
            auto info = data.request();
            mgr.upload(dst, info.ptr, info.size * sizeof(float), offset);
          },
          py::arg("dst"), py::arg("data"), py::arg("offset") = 0)
      .def(
          "download",
          [](mc189::BufferManager &mgr, const mc189::Buffer &src, size_t count,
             size_t offset) {
            std::vector<float> data(count);
            mgr.download(src, data.data(), count * sizeof(float), offset);
            return py::array_t<float>(count, data.data());
          },
          py::arg("src"), py::arg("count"), py::arg("offset") = 0)
      .def_property_readonly("total_allocated",
                             &mc189::BufferManager::total_allocated)
      .def_property_readonly("device_allocated",
                             &mc189::BufferManager::device_allocated)
      .def_property_readonly("host_allocated",
                             &mc189::BufferManager::host_allocated);

  // RingBuffer class
  py::class_<mc189::RingBuffer>(m, "RingBuffer")
      .def(py::init<const mc189::VulkanContext &, vk::DeviceSize>())
      .def_property_readonly("capacity", &mc189::RingBuffer::capacity)
      .def_property_readonly("used", &mc189::RingBuffer::used)
      .def_property_readonly("available", &mc189::RingBuffer::available)
      .def("reset", &mc189::RingBuffer::reset);

  // EnvironmentBufferPool::Config
  py::class_<mc189::EnvironmentBufferPool::Config>(
      m, "EnvironmentBufferPoolConfig")
      .def(py::init<>())
      .def_readwrite("max_environments",
                     &mc189::EnvironmentBufferPool::Config::max_environments)
      .def_readwrite("state_size_per_env",
                     &mc189::EnvironmentBufferPool::Config::state_size_per_env)
      .def_readwrite("action_size_per_env",
                     &mc189::EnvironmentBufferPool::Config::action_size_per_env)
      .def_readwrite("reward_size_per_env",
                     &mc189::EnvironmentBufferPool::Config::reward_size_per_env)
      .def_readwrite("double_buffer",
                     &mc189::EnvironmentBufferPool::Config::double_buffer);

  // EnvironmentBufferPool class
  py::class_<mc189::EnvironmentBufferPool>(m, "EnvironmentBufferPool")
      .def(py::init<const mc189::VulkanContext &, mc189::BufferManager &,
                    const mc189::EnvironmentBufferPool::Config &>())
      .def_property_readonly("max_environments",
                             &mc189::EnvironmentBufferPool::max_environments)
      .def_property_readonly(
          "current_buffer_index",
          &mc189::EnvironmentBufferPool::current_buffer_index)
      .def("swap_buffers", &mc189::EnvironmentBufferPool::swap_buffers);
}
