// Python bindings for MC189Simulator and MultistageSimulator
// Exposes the GPU-accelerated simulators to Python

#include "mc189/dimension.h"
#include "mc189/game_stage.h"
#include "mc189/multistage_simulator.h"
#include "mc189/simulator.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace mc189;

void init_simulator(py::module_ &m) {
  py::class_<MC189Simulator::Config>(m, "SimulatorConfig")
      .def(py::init<>())
      .def_readwrite("num_envs", &MC189Simulator::Config::num_envs)
      .def_readwrite("enable_validation",
                     &MC189Simulator::Config::enable_validation)
      .def_readwrite("shader_dir", &MC189Simulator::Config::shader_dir)
      .def_readwrite("shader_set", &MC189Simulator::Config::shader_set)
      .def_readwrite("use_cpu", &MC189Simulator::Config::use_cpu,
                     "Force CPU backend (no GPU required)");

  py::class_<MC189Simulator>(m, "MC189Simulator")
      .def(py::init<const MC189Simulator::Config &>(), py::arg("config"))
      .def(
          "step",
          [](MC189Simulator &self, py::array_t<int32_t> actions) {
            py::buffer_info buf = actions.request();
            if (buf.ndim != 1) {
              throw std::runtime_error("Actions must be 1D array");
            }
            self.step(static_cast<const int32_t *>(buf.ptr),
                      static_cast<size_t>(buf.shape[0]));
          },
          py::arg("actions"),
          "Execute one simulation step with the given actions")
      .def(
          "reset",
          [](MC189Simulator &self, py::object env_id, py::object seed) {
            uint64_t seed_value = 0;
            if (!seed.is_none()) {
              seed_value = py::cast<uint64_t>(seed);
            }
            if (env_id.is_none()) {
              self.reset(0xFFFFFFFF, seed_value); // Reset all
            } else {
              self.reset(py::cast<uint32_t>(env_id), seed_value);
            }
          },
          py::arg("env_id") = py::none(), py::arg("seed") = py::none(),
          "Reset environment(s). Pass None to reset all, or env index. "
          "Optional seed enables deterministic resets.")
      .def(
          "get_observations",
          [](const MC189Simulator &self) {
            const size_t n = self.num_envs();
            const size_t obs_dim = MC189Simulator::obs_dim();
            py::array_t<float> arr({n, obs_dim});
            std::memcpy(arr.mutable_data(), self.get_observations(),
                        n * obs_dim * sizeof(float));
            return arr;
          },
          "Get observations as numpy array (num_envs, obs_dim)")
      .def(
          "get_rewards",
          [](const MC189Simulator &self) {
            const size_t n = self.num_envs();
            py::array_t<float> arr(n);
            std::memcpy(arr.mutable_data(), self.get_rewards(),
                        n * sizeof(float));
            return arr;
          },
          "Get rewards as numpy array (num_envs,)")
      .def(
          "get_dones",
          [](const MC189Simulator &self) {
            const size_t n = self.num_envs();
            py::array_t<bool> arr(n);
            const uint8_t *src = self.get_dones();
            bool *dst = arr.mutable_data();
            for (size_t i = 0; i < n; ++i) {
              dst[i] = src[i] != 0;
            }
            return arr;
          },
          "Get done flags as numpy array (num_envs,)")
      .def_property_readonly("num_envs", &MC189Simulator::num_envs)
      .def_property_readonly_static(
          "obs_dim", [](py::object) { return MC189Simulator::obs_dim(); })
      .def("is_cpu_backend", &MC189Simulator::is_cpu_backend,
           "Returns True if using CPU backend (no GPU)");

  // GameStage enum
  py::enum_<GameStage>(m, "GameStage")
      .value("BASIC_SURVIVAL", GameStage::BASIC_SURVIVAL)
      .value("RESOURCE_GATHERING", GameStage::RESOURCE_GATHERING)
      .value("NETHER_NAVIGATION", GameStage::NETHER_NAVIGATION)
      .value("ENDERMAN_HUNTING", GameStage::ENDERMAN_HUNTING)
      .value("STRONGHOLD_FINDING", GameStage::STRONGHOLD_FINDING)
      .value("END_FIGHT", GameStage::END_FIGHT);

  // Dimension enum
  py::enum_<Dimension>(m, "Dimension")
      .value("NETHER", Dimension::NETHER)
      .value("OVERWORLD", Dimension::OVERWORLD)
      .value("END", Dimension::END);

  // MultistageSimulator config
  py::class_<MultistageSimulator::Config>(m, "MultistageConfig")
      .def(py::init<>())
      .def_readwrite("num_envs", &MultistageSimulator::Config::num_envs)
      .def_readwrite("initial_stage",
                     &MultistageSimulator::Config::initial_stage)
      .def_readwrite("enable_validation",
                     &MultistageSimulator::Config::enable_validation)
      .def_readwrite("shader_dir", &MultistageSimulator::Config::shader_dir)
      .def_readwrite("auto_advance_stage",
                     &MultistageSimulator::Config::auto_advance_stage)
      .def_readwrite("max_entities_per_env",
                     &MultistageSimulator::Config::max_entities_per_env)
      .def_readwrite("max_chunks_per_env",
                     &MultistageSimulator::Config::max_chunks_per_env);

  // MultistageSimulator
  py::class_<MultistageSimulator>(m, "MultistageSimulator")
      .def(py::init<const MultistageSimulator::Config &>(), py::arg("config"))
      .def(
          "step",
          [](MultistageSimulator &self, py::array_t<int32_t> actions) {
            py::buffer_info buf = actions.request();
            if (buf.ndim != 1) {
              throw std::runtime_error("Actions must be 1D array");
            }
            self.step(static_cast<const int32_t *>(buf.ptr),
                      static_cast<size_t>(buf.shape[0]));
          },
          py::arg("actions"),
          "Execute one simulation step with the given actions")
      .def(
          "reset",
          [](MultistageSimulator &self, py::object env_id, py::object seed) {
            uint64_t seed_value = 0;
            if (!seed.is_none()) {
              seed_value = py::cast<uint64_t>(seed);
            }
            if (env_id.is_none()) {
              self.reset(0xFFFFFFFF, seed_value);
            } else {
              self.reset(py::cast<uint32_t>(env_id), seed_value);
            }
          },
          py::arg("env_id") = py::none(), py::arg("seed") = py::none(),
          "Reset environment(s)")
      .def(
          "get_observations",
          [](const MultistageSimulator &self) {
            const size_t n = self.num_envs();
            constexpr size_t obs_dim = MultistageSimulator::EXTENDED_OBS_SIZE;
            py::array_t<float> arr({n, obs_dim});
            std::memcpy(arr.mutable_data(), self.get_extended_observations(),
                        n * obs_dim * sizeof(float));
            return arr;
          },
          "Get extended observations as numpy array (num_envs, 256)")
      .def(
          "get_rewards",
          [](const MultistageSimulator &self) {
            const size_t n = self.num_envs();
            py::array_t<float> arr(n);
            std::memcpy(arr.mutable_data(), self.get_rewards(),
                        n * sizeof(float));
            return arr;
          },
          "Get rewards as numpy array (num_envs,)")
      .def(
          "get_dones",
          [](const MultistageSimulator &self) {
            const size_t n = self.num_envs();
            py::array_t<bool> arr(n);
            const uint8_t *src = self.get_dones();
            bool *dst = arr.mutable_data();
            for (size_t i = 0; i < n; ++i) {
              dst[i] = src[i] != 0;
            }
            return arr;
          },
          "Get done flags as numpy array (num_envs,)")
      .def("set_stage", &MultistageSimulator::set_stage, py::arg("env_id"),
           py::arg("stage"), "Set the game stage for a specific environment")
      .def("get_stage", &MultistageSimulator::get_stage, py::arg("env_id"),
           "Get the current game stage for an environment")
      .def("get_stage_progress", &MultistageSimulator::get_stage_progress,
           py::arg("env_id"), "Get stage progress bitmask for an environment")
      .def("teleport_to_dimension", &MultistageSimulator::teleport_to_dimension,
           py::arg("env_id"), py::arg("dimension"), py::arg("x"), py::arg("y"),
           py::arg("z"), "Teleport player to a dimension at given coordinates")
      .def("get_dimension", &MultistageSimulator::get_dimension,
           py::arg("env_id"), "Get the current dimension for an environment")
      .def_property_readonly("num_envs", &MultistageSimulator::num_envs)
      .def_property_readonly_static("obs_dim", [](py::object) {
        return MultistageSimulator::EXTENDED_OBS_SIZE;
      });
}
