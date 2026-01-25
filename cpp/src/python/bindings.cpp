// bindings.cpp - pybind11 bindings for Minecraft RL Simulator
// Exposes both C++ class API and high-performance C API with NumPy support

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "minecraft_sim.hpp"
#include "mc189/simulator_api.h"
#include "mc189/multistage_simulator.h"

#include <cstring>
#include <stdexcept>
#include <memory>

namespace py = pybind11;
using namespace minecraft_sim;

void init_gym_wrapper(py::module_& m);

// Custom deleter for mc189_simulator_t
struct SimulatorDeleter {
    void operator()(mc189_simulator_t sim) const {
        if (sim) mc189_destroy(sim);
    }
};
using SimulatorPtr = std::unique_ptr<mc189_simulator_impl, SimulatorDeleter>;

// Helper to check mc189 error codes
static void check_mc189_error(mc189_error_t err) {
    if (err != MC189_OK) {
        const char* msg = mc189_get_error_message();
        throw std::runtime_error(msg ? msg : "Unknown mc189 error");
    }
}

// Convert InputState to numpy-friendly format
static py::array_t<float> input_state_to_array(const InputState& input) {
    py::array_t<float> arr(8);
    auto ptr = arr.mutable_data();
    ptr[0] = input.movement.x;
    ptr[1] = input.movement.y;
    ptr[2] = input.movement.z;
    ptr[3] = input.look_delta_x;
    ptr[4] = input.look_delta_y;
    ptr[5] = static_cast<float>(input.action);
    ptr[6] = static_cast<float>(input.action_data);
    ptr[7] = static_cast<float>(input.flags);
    return arr;
}

// Convert numpy array to InputState
static InputState array_to_input_state(py::array_t<float> arr) {
    if (arr.size() < 8) {
        throw std::invalid_argument("Input array must have at least 8 elements");
    }
    auto ptr = arr.data();
    InputState input;
    input.movement = Vec3(ptr[0], ptr[1], ptr[2]);
    input.look_delta_x = ptr[3];
    input.look_delta_y = ptr[4];
    input.action = static_cast<ActionType>(static_cast<int>(ptr[5]));
    input.action_data = static_cast<uint32_t>(ptr[6]);
    input.flags = static_cast<uint32_t>(ptr[7]);
    return input;
}

// Convert mc189_action_t to numpy array
static py::array_t<float> mc189_action_to_array(const mc189_action_t& action) {
    py::array_t<float> arr(10);
    auto ptr = arr.mutable_data();
    ptr[0] = static_cast<float>(action.action);
    ptr[1] = action.look_delta_yaw;
    ptr[2] = action.look_delta_pitch;
    ptr[3] = static_cast<float>(action.target_block[0]);
    ptr[4] = static_cast<float>(action.target_block[1]);
    ptr[5] = static_cast<float>(action.target_block[2]);
    ptr[6] = static_cast<float>(action.target_face);
    ptr[7] = static_cast<float>(action.recipe_id);
    ptr[8] = static_cast<float>(action.flags);
    ptr[9] = 0.0f; // padding
    return arr;
}

// Convert numpy array to mc189_action_t
static mc189_action_t array_to_mc189_action(py::array_t<float> arr) {
    mc189_action_t action;
    std::memset(&action, 0, sizeof(action));

    auto ptr = arr.data();
    auto size = arr.size();

    action.action = static_cast<mc189_action_type_t>(static_cast<int>(ptr[0]));
    if (size > 1) action.look_delta_yaw = ptr[1];
    if (size > 2) action.look_delta_pitch = ptr[2];
    if (size > 3) action.target_block[0] = static_cast<int32_t>(ptr[3]);
    if (size > 4) action.target_block[1] = static_cast<int32_t>(ptr[4]);
    if (size > 5) action.target_block[2] = static_cast<int32_t>(ptr[5]);
    if (size > 6) action.target_face = static_cast<uint8_t>(ptr[6]);
    if (size > 7) action.recipe_id = static_cast<uint16_t>(ptr[7]);
    if (size > 8) action.flags = static_cast<uint8_t>(ptr[8]);

    return action;
}

// Python wrapper for MinecraftSimulator (C++ API)
class PyMinecraftSimulator {
public:
    explicit PyMinecraftSimulator(uint64_t seed = 0)
        : sim_(seed), obs_array_(ObservationShape::TOTAL) {}

    py::tuple reset(uint64_t seed) {
        sim_.reset(seed);
        return py::make_tuple(get_observation(), py::dict());
    }

    py::tuple step(py::array_t<float> action) {
        InputState input = array_to_input_state(action);
        StepResult result = sim_.step(input);

        py::dict info;
        info["damage_dealt"] = result.damage_dealt;
        info["damage_taken"] = result.damage_taken;
        info["blocks_mined"] = result.blocks_mined;
        info["items_crafted"] = result.items_crafted;
        info["distance_traveled"] = result.distance_traveled;
        info["dragon_damage"] = result.dragon_damage;

        return py::make_tuple(
            get_observation(),
            result.reward,
            result.terminated,
            result.truncated,
            info
        );
    }

    py::tuple step_discrete(int action, float intensity = 1.0f) {
        InputState input = MinecraftSimulator::discrete_to_input(action, intensity);
        StepResult result = sim_.step(input);

        py::dict info;
        info["damage_dealt"] = result.damage_dealt;
        info["damage_taken"] = result.damage_taken;
        info["dragon_damage"] = result.dragon_damage;

        return py::make_tuple(
            get_observation(),
            result.reward,
            result.terminated,
            result.truncated,
            info
        );
    }

    py::array_t<float> get_observation() {
        const float* obs_ptr = sim_.get_observation_ptr();
        size_t size = sim_.get_observation_size();

        // Create numpy array with copy of observation data
        py::array_t<float> arr(size);
        std::memcpy(arr.mutable_data(), obs_ptr, size * sizeof(float));
        return arr;
    }

    py::dict get_player_state() const {
        const Player& p = sim_.get_player();
        py::dict state;
        state["position"] = py::make_tuple(p.position.x, p.position.y, p.position.z);
        state["velocity"] = py::make_tuple(p.velocity.x, p.velocity.y, p.velocity.z);
        state["yaw"] = p.yaw;
        state["pitch"] = p.pitch;
        state["health"] = p.health;
        state["hunger"] = p.hunger;
        state["saturation"] = p.saturation;
        state["dimension"] = static_cast<int>(p.dimension);
        state["active_slot"] = p.active_slot;
        return state;
    }

    py::dict get_dragon_state() const {
        const DragonFight& d = sim_.get_dragon_fight();
        py::dict state;
        state["phase"] = d.phase;
        state["health"] = d.health;
        state["crystals_remaining"] = d.crystals_remaining;
        state["circle_center"] = py::make_tuple(d.circle_center.x, d.circle_center.y, d.circle_center.z);
        state["perch_timer"] = d.perch_timer;
        return state;
    }

    uint64_t tick() const { return sim_.get_tick(); }
    bool is_dragon_dead() const { return sim_.is_dragon_dead(); }
    bool is_player_dead() const { return sim_.is_player_dead(); }

    static size_t observation_size() { return ObservationShape::TOTAL; }
    static size_t action_size() { return ActionSpace::DISCRETE_ACTIONS; }

private:
    MinecraftSimulator sim_;
    py::array_t<float> obs_array_;
};

// Python wrapper for VecMinecraftSimulator (C++ API)
class PyVecMinecraftSimulator {
public:
    explicit PyVecMinecraftSimulator(size_t num_envs, uint64_t base_seed = 0)
        : sim_(num_envs, base_seed), num_envs_(num_envs) {}

    py::tuple reset() {
        sim_.reset_all();
        return py::make_tuple(get_observations(), py::dict());
    }

    void reset_env(size_t env_idx, uint64_t seed) {
        sim_.reset(env_idx, seed);
    }

    py::tuple step(py::array_t<float> actions) {
        // actions shape: (num_envs, action_dim)
        auto buf = actions.request();
        if (buf.ndim != 2 || buf.shape[0] != static_cast<py::ssize_t>(num_envs_)) {
            throw std::invalid_argument("Actions must have shape (num_envs, action_dim)");
        }

        std::vector<InputState> inputs(num_envs_);
        float* data = static_cast<float*>(buf.ptr);
        size_t action_dim = buf.shape[1];

        for (size_t i = 0; i < num_envs_; ++i) {
            py::array_t<float> action_slice(action_dim, data + i * action_dim);
            inputs[i] = array_to_input_state(action_slice);
        }

        sim_.step_all(inputs);

        const float* rewards = sim_.get_rewards_ptr();
        const bool* terminated = sim_.get_terminated_ptr();
        const bool* truncated = sim_.get_truncated_ptr();

        py::array_t<float> rewards_arr(num_envs_);
        py::array_t<bool> terminated_arr(num_envs_);
        py::array_t<bool> truncated_arr(num_envs_);

        std::memcpy(rewards_arr.mutable_data(), rewards, num_envs_ * sizeof(float));
        std::memcpy(terminated_arr.mutable_data(), terminated, num_envs_ * sizeof(bool));
        std::memcpy(truncated_arr.mutable_data(), truncated, num_envs_ * sizeof(bool));

        return py::make_tuple(
            get_observations(),
            rewards_arr,
            terminated_arr,
            truncated_arr,
            py::dict()
        );
    }

    py::array_t<float> get_observations() {
        const float* obs_ptr = sim_.get_observations_ptr();
        size_t obs_size = sim_.observation_size();

        // Create 2D numpy array: (num_envs, obs_size)
        py::array_t<float> arr({num_envs_, obs_size});
        std::memcpy(arr.mutable_data(), obs_ptr, num_envs_ * obs_size * sizeof(float));
        return arr;
    }

    size_t num_envs() const { return num_envs_; }
    size_t observation_size() const { return sim_.observation_size(); }

private:
    VecMinecraftSimulator sim_;
    size_t num_envs_;
};

// Python wrapper for high-performance C API (mc189)
class PyMC189Simulator {
public:
    explicit PyMC189Simulator(const py::dict& config = py::dict()) {
        mc189_config_t cfg = mc189_default_config();

        // Apply config overrides
        if (config.contains("batch_size")) {
            cfg.batch_size = config["batch_size"].cast<uint32_t>();
        }
        if (config.contains("max_ticks")) {
            cfg.max_ticks_per_episode = config["max_ticks"].cast<uint32_t>();
        }
        if (config.contains("deterministic")) {
            cfg.deterministic_mode = config["deterministic"].cast<bool>();
        }
        if (config.contains("async_step")) {
            cfg.async_step = config["async_step"].cast<bool>();
        }
        if (config.contains("dragon_kill_reward")) {
            cfg.dragon_kill_reward = config["dragon_kill_reward"].cast<float>();
        }
        if (config.contains("death_penalty")) {
            cfg.death_penalty = config["death_penalty"].cast<float>();
        }
        if (config.contains("time_penalty")) {
            cfg.time_penalty_per_tick = config["time_penalty"].cast<float>();
        }
        if (config.contains("device_index")) {
            cfg.preferred_device_index = config["device_index"].cast<int32_t>();
        }
        if (config.contains("validation_layers")) {
            cfg.enable_validation_layers = config["validation_layers"].cast<bool>();
        }

        batch_size_ = cfg.batch_size;
        async_mode_ = cfg.async_step;

        mc189_simulator_t raw_sim;
        check_mc189_error(mc189_create(&cfg, &raw_sim));
        sim_.reset(raw_sim);

        // Pre-allocate buffers
        obs_buffer_.resize(batch_size_);
        result_buffer_.resize(batch_size_);
    }

    py::tuple reset(py::object seeds = py::none()) {
        if (seeds.is_none()) {
            check_mc189_error(mc189_reset_batch(sim_.get(), nullptr, obs_buffer_.data()));
        } else {
            py::array_t<uint64_t> seed_arr = seeds.cast<py::array_t<uint64_t>>();
            if (seed_arr.size() != static_cast<py::ssize_t>(batch_size_)) {
                throw std::invalid_argument("Seeds array must match batch size");
            }
            check_mc189_error(mc189_reset_batch(sim_.get(), seed_arr.data(), obs_buffer_.data()));
        }
        return py::make_tuple(observations_to_numpy(), py::dict());
    }

    py::tuple step(py::array_t<float> actions) {
        auto buf = actions.request();

        // Support both (batch_size,) discrete and (batch_size, action_dim) continuous
        std::vector<mc189_action_t> action_vec(batch_size_);

        if (buf.ndim == 1) {
            // Discrete actions only
            float* data = static_cast<float*>(buf.ptr);
            for (size_t i = 0; i < batch_size_; ++i) {
                std::memset(&action_vec[i], 0, sizeof(mc189_action_t));
                action_vec[i].action = static_cast<mc189_action_type_t>(static_cast<int>(data[i]));
            }
        } else if (buf.ndim == 2) {
            // Full action array
            float* data = static_cast<float*>(buf.ptr);
            size_t action_dim = buf.shape[1];
            for (size_t i = 0; i < batch_size_; ++i) {
                py::array_t<float> action_slice(action_dim, data + i * action_dim);
                action_vec[i] = array_to_mc189_action(action_slice);
            }
        } else {
            throw std::invalid_argument("Actions must be 1D or 2D array");
        }

        check_mc189_error(mc189_step_batch(sim_.get(), action_vec.data(), result_buffer_.data()));

        return results_to_tuple();
    }

    py::tuple step_async(py::array_t<float> actions) {
        if (!async_mode_) {
            throw std::runtime_error("Async mode not enabled in config");
        }

        auto buf = actions.request();
        std::vector<mc189_action_t> action_vec(batch_size_);
        float* data = static_cast<float*>(buf.ptr);

        for (size_t i = 0; i < batch_size_; ++i) {
            std::memset(&action_vec[i], 0, sizeof(mc189_action_t));
            action_vec[i].action = static_cast<mc189_action_type_t>(static_cast<int>(data[i]));
        }

        check_mc189_error(mc189_step_async(sim_.get(), action_vec.data()));
        return py::make_tuple(true);
    }

    py::tuple poll() {
        mc189_error_t err = mc189_poll_step(sim_.get(), result_buffer_.data());
        if (err == MC189_ERROR_INVALID_STATE) {
            return py::make_tuple(false, py::none());
        }
        check_mc189_error(err);
        return py::make_tuple(true, results_to_tuple());
    }

    py::tuple wait() {
        check_mc189_error(mc189_wait_step(sim_.get(), result_buffer_.data()));
        return results_to_tuple();
    }

    py::dict get_stats() {
        mc189_stats_t stats;
        check_mc189_error(mc189_get_stats(sim_.get(), &stats));

        py::dict result;
        result["steps_per_second"] = stats.steps_per_second;
        result["last_step_time_ms"] = stats.last_step_time_ms;
        result["avg_step_time_ms"] = stats.avg_step_time_ms;
        result["total_steps"] = stats.total_steps;
        result["gpu_utilization"] = stats.gpu_utilization_percent;
        result["gpu_memory_used"] = stats.gpu_memory_used_bytes;
        result["gpu_memory_total"] = stats.gpu_memory_total_bytes;

        py::dict shader_times;
        for (uint32_t i = 0; i < stats.num_shaders; ++i) {
            shader_times[stats.shader_names[i]] = stats.shader_times_us[i];
        }
        result["shader_times_us"] = shader_times;

        return result;
    }

    void set_player_position(uint32_t env_idx, py::array_t<float> pos) {
        if (pos.size() != 3) {
            throw std::invalid_argument("Position must have 3 elements");
        }
        check_mc189_error(mc189_set_player_position(sim_.get(), env_idx, pos.data()));
    }

    void teleport(uint32_t env_idx, int dimension, py::array_t<float> pos) {
        if (pos.size() != 3) {
            throw std::invalid_argument("Position must have 3 elements");
        }
        check_mc189_error(mc189_teleport(
            sim_.get(), env_idx,
            static_cast<mc189_dimension_t>(dimension),
            pos.data()
        ));
    }

    void give_item(uint32_t env_idx, uint16_t item_id, uint8_t count, uint8_t slot) {
        check_mc189_error(mc189_give_item(sim_.get(), env_idx, item_id, count, slot));
    }

    std::string device_name() const {
        return mc189_device_name(sim_.get());
    }

    uint32_t batch_size() const { return batch_size_; }

    static std::string version() { return mc189_version(); }
    static bool check_gpu_support() { return mc189_check_gpu_support(); }

private:
    SimulatorPtr sim_;
    uint32_t batch_size_;
    bool async_mode_;
    std::vector<mc189_observation_t> obs_buffer_;
    std::vector<mc189_step_result_t> result_buffer_;

    py::array_t<float> observations_to_numpy() {
        // Flatten observations into contiguous float array
        // Layout: player state, dragon state, nearby mobs, local blocks per env
        constexpr size_t obs_floats = sizeof(mc189_observation_t) / sizeof(float);

        py::array_t<float> arr({batch_size_, obs_floats});
        float* out = arr.mutable_data();

        for (size_t i = 0; i < batch_size_; ++i) {
            std::memcpy(out + i * obs_floats, &obs_buffer_[i], sizeof(mc189_observation_t));
        }

        return arr;
    }

    py::tuple results_to_tuple() {
        py::array_t<float> rewards(batch_size_);
        py::array_t<bool> terminated(batch_size_);
        py::array_t<bool> truncated(batch_size_);

        float* r = rewards.mutable_data();
        bool* t = terminated.mutable_data();
        bool* tr = truncated.mutable_data();

        for (size_t i = 0; i < batch_size_; ++i) {
            std::memcpy(&obs_buffer_[i], &result_buffer_[i].observation, sizeof(mc189_observation_t));
            r[i] = result_buffer_[i].reward;
            t[i] = result_buffer_[i].terminated;
            tr[i] = result_buffer_[i].truncated;
        }

        py::dict info;
        py::array_t<float> dragon_damage(batch_size_);
        py::array_t<uint8_t> crystals_destroyed(batch_size_);

        float* dd = dragon_damage.mutable_data();
        uint8_t* cd = crystals_destroyed.mutable_data();

        for (size_t i = 0; i < batch_size_; ++i) {
            dd[i] = result_buffer_[i].dragon_damage_dealt;
            cd[i] = result_buffer_[i].crystals_destroyed;
        }

        info["dragon_damage"] = dragon_damage;
        info["crystals_destroyed"] = crystals_destroyed;

        return py::make_tuple(observations_to_numpy(), rewards, terminated, truncated, info);
    }
};

void init_gym_wrapper(py::module_& m);

// Module definition
PYBIND11_MODULE(_minecraft_sim, m) {
    m.doc() = "Minecraft RL Simulator Python bindings";

    // Enums
    py::enum_<Dimension>(m, "Dimension")
        .value("Overworld", Dimension::Overworld)
        .value("Nether", Dimension::Nether)
        .value("End", Dimension::End);

    py::enum_<ActionType>(m, "ActionType")
        .value("None", ActionType::None)
        .value("Mine", ActionType::Mine)
        .value("Place", ActionType::Place)
        .value("Attack", ActionType::Attack)
        .value("UseItem", ActionType::UseItem)
        .value("Interact", ActionType::Interact)
        .value("MoveForward", ActionType::MoveForward)
        .value("MoveBack", ActionType::MoveBack)
        .value("MoveLeft", ActionType::MoveLeft)
        .value("MoveRight", ActionType::MoveRight)
        .value("Jump", ActionType::Jump)
        .value("Sprint", ActionType::Sprint)
        .value("Sneak", ActionType::Sneak)
        .value("LookUp", ActionType::LookUp)
        .value("LookDown", ActionType::LookDown)
        .value("LookLeft", ActionType::LookLeft)
        .value("LookRight", ActionType::LookRight)
        .value("SelectSlot", ActionType::SelectSlot);

    py::enum_<MobType>(m, "MobType")
        .value("Zombie", MobType::Zombie)
        .value("Skeleton", MobType::Skeleton)
        .value("Creeper", MobType::Creeper)
        .value("Spider", MobType::Spider)
        .value("Enderman", MobType::Enderman)
        .value("Blaze", MobType::Blaze)
        .value("Ghast", MobType::Ghast)
        .value("EnderDragon", MobType::EnderDragon);

    // MC189 Action types
    py::enum_<mc189_action_type_t>(m, "MC189Action")
        .value("NONE", MC189_ACTION_NONE)
        .value("FORWARD", MC189_ACTION_FORWARD)
        .value("BACKWARD", MC189_ACTION_BACKWARD)
        .value("LEFT", MC189_ACTION_LEFT)
        .value("RIGHT", MC189_ACTION_RIGHT)
        .value("JUMP", MC189_ACTION_JUMP)
        .value("SNEAK", MC189_ACTION_SNEAK)
        .value("SPRINT", MC189_ACTION_SPRINT)
        .value("ATTACK", MC189_ACTION_ATTACK)
        .value("USE", MC189_ACTION_USE)
        .value("MINE", MC189_ACTION_MINE)
        .value("PLACE", MC189_ACTION_PLACE)
        .value("HOTBAR_0", MC189_ACTION_HOTBAR_0)
        .value("HOTBAR_1", MC189_ACTION_HOTBAR_1)
        .value("HOTBAR_2", MC189_ACTION_HOTBAR_2)
        .value("HOTBAR_3", MC189_ACTION_HOTBAR_3)
        .value("HOTBAR_4", MC189_ACTION_HOTBAR_4)
        .value("HOTBAR_5", MC189_ACTION_HOTBAR_5)
        .value("HOTBAR_6", MC189_ACTION_HOTBAR_6)
        .value("HOTBAR_7", MC189_ACTION_HOTBAR_7)
        .value("HOTBAR_8", MC189_ACTION_HOTBAR_8)
        .value("LOOK_UP", MC189_ACTION_LOOK_UP)
        .value("LOOK_DOWN", MC189_ACTION_LOOK_DOWN)
        .value("LOOK_LEFT", MC189_ACTION_LOOK_LEFT)
        .value("LOOK_RIGHT", MC189_ACTION_LOOK_RIGHT)
        .value("DROP", MC189_ACTION_DROP)
        .value("INVENTORY", MC189_ACTION_INVENTORY)
        .value("CRAFT", MC189_ACTION_CRAFT);

    py::enum_<mc189_dimension_t>(m, "MC189Dimension")
        .value("OVERWORLD", MC189_DIM_OVERWORLD)
        .value("NETHER", MC189_DIM_NETHER)
        .value("END", MC189_DIM_END);

    // C++ API wrappers
    py::class_<PyMinecraftSimulator>(m, "MinecraftSimulator")
        .def(py::init<uint64_t>(), py::arg("seed") = 0)
        .def("reset", &PyMinecraftSimulator::reset, py::arg("seed") = 0)
        .def("step", &PyMinecraftSimulator::step, py::arg("action"))
        .def("step_discrete", &PyMinecraftSimulator::step_discrete,
             py::arg("action"), py::arg("intensity") = 1.0f)
        .def("get_observation", &PyMinecraftSimulator::get_observation)
        .def("get_player_state", &PyMinecraftSimulator::get_player_state)
        .def("get_dragon_state", &PyMinecraftSimulator::get_dragon_state)
        .def_property_readonly("tick", &PyMinecraftSimulator::tick)
        .def_property_readonly("is_dragon_dead", &PyMinecraftSimulator::is_dragon_dead)
        .def_property_readonly("is_player_dead", &PyMinecraftSimulator::is_player_dead)
        .def_property_readonly_static("observation_size",
            [](py::object) { return PyMinecraftSimulator::observation_size(); })
        .def_property_readonly_static("action_size",
            [](py::object) { return PyMinecraftSimulator::action_size(); });

    py::class_<PyVecMinecraftSimulator>(m, "VecMinecraftSimulator")
        .def(py::init<size_t, uint64_t>(),
             py::arg("num_envs"), py::arg("base_seed") = 0)
        .def("reset", &PyVecMinecraftSimulator::reset)
        .def("reset_env", &PyVecMinecraftSimulator::reset_env,
             py::arg("env_idx"), py::arg("seed"))
        .def("step", &PyVecMinecraftSimulator::step, py::arg("actions"))
        .def("get_observations", &PyVecMinecraftSimulator::get_observations)
        .def_property_readonly("num_envs", &PyVecMinecraftSimulator::num_envs)
        .def_property_readonly("observation_size", &PyVecMinecraftSimulator::observation_size);

    // High-performance C API wrapper
    py::class_<PyMC189Simulator>(m, "MC189Simulator")
        .def(py::init<const py::dict&>(), py::arg("config") = py::dict())
        .def("reset", &PyMC189Simulator::reset, py::arg("seeds") = py::none())
        .def("step", &PyMC189Simulator::step, py::arg("actions"))
        .def("step_async", &PyMC189Simulator::step_async, py::arg("actions"))
        .def("poll", &PyMC189Simulator::poll)
        .def("wait", &PyMC189Simulator::wait)
        .def("get_stats", &PyMC189Simulator::get_stats)
        .def("set_player_position", &PyMC189Simulator::set_player_position,
             py::arg("env_idx"), py::arg("position"))
        .def("teleport", &PyMC189Simulator::teleport,
             py::arg("env_idx"), py::arg("dimension"), py::arg("position"))
        .def("give_item", &PyMC189Simulator::give_item,
             py::arg("env_idx"), py::arg("item_id"), py::arg("count"), py::arg("slot"))
        .def_property_readonly("device_name", &PyMC189Simulator::device_name)
        .def_property_readonly("batch_size", &PyMC189Simulator::batch_size)
        .def_static("version", &PyMC189Simulator::version)
        .def_static("check_gpu_support", &PyMC189Simulator::check_gpu_support);

    // Constants
    m.attr("OBSERVATION_SIZE") = ObservationShape::TOTAL;
    m.attr("ACTION_SIZE") = ActionSpace::DISCRETE_ACTIONS;
    m.attr("MAX_BATCH_SIZE") = MC189_MAX_BATCH_SIZE;
    m.attr("TICKS_PER_SECOND") = MC189_TICKS_PER_SECOND;
    m.attr("MC189_ACTION_COUNT") = MC189_ACTION_COUNT;
    m.attr("STATE_DIM") = 32;
    m.attr("NUM_ACTIONS") = 16;
    m.attr("OBS_WIDTH") = 64;
    m.attr("OBS_HEIGHT") = 64;
    m.attr("OBS_CHANNELS") = 3;

    // Utility functions
    m.def("discrete_to_input", [](int action, float intensity) {
        return input_state_to_array(MinecraftSimulator::discrete_to_input(action, intensity));
    }, py::arg("action"), py::arg("intensity") = 1.0f,
       "Convert discrete action index to InputState array");

    m.def("check_gpu", &mc189_check_gpu_support,
          "Check if GPU supports required Vulkan features");

    // GameStage enum
    py::enum_<mc189::GameStage>(m, "GameStage")
        .value("BASIC_SURVIVAL", mc189::GameStage::BASIC_SURVIVAL)
        .value("RESOURCE_GATHERING", mc189::GameStage::RESOURCE_GATHERING)
        .value("NETHER_NAVIGATION", mc189::GameStage::NETHER_NAVIGATION)
        .value("ENDERMAN_HUNTING", mc189::GameStage::ENDERMAN_HUNTING)
        .value("STRONGHOLD_FINDING", mc189::GameStage::STRONGHOLD_FINDING)
        .value("END_FIGHT", mc189::GameStage::END_FIGHT);

    // mc189::Dimension enum for MultistageSimulator (NETHER=-1, OVERWORLD=0, END=1)
    py::enum_<mc189::Dimension>(m, "StageDimension")
        .value("NETHER", mc189::Dimension::NETHER)
        .value("OVERWORLD", mc189::Dimension::OVERWORLD)
        .value("END", mc189::Dimension::END);

    // MultistageSimulator bindings
    py::class_<mc189::MultistageSimulator>(m, "MultistageSimulator")
        .def(py::init([](uint32_t num_envs, int initial_stage,
                        bool enable_validation, const std::string& shader_dir,
                        bool auto_advance) {
            mc189::MultistageSimulator::Config cfg;
            cfg.num_envs = num_envs;
            cfg.initial_stage = static_cast<mc189::GameStage>(initial_stage);
            cfg.enable_validation = enable_validation;
            cfg.shader_dir = shader_dir;
            cfg.auto_advance_stage = auto_advance;
            return std::make_unique<mc189::MultistageSimulator>(cfg);
        }), py::arg("num_envs") = 1,
            py::arg("initial_stage") = 1,
            py::arg("enable_validation") = false,
            py::arg("shader_dir") = "shaders",
            py::arg("auto_advance") = true)
        .def("step", [](mc189::MultistageSimulator& self, py::array_t<int32_t> actions) {
            auto buf = actions.request();
            self.step(static_cast<int32_t*>(buf.ptr), buf.size);
        })
        .def("reset", &mc189::MultistageSimulator::reset,
             py::arg("env_id") = 0xFFFFFFFF, py::arg("seed") = 0)
        .def("set_stage", &mc189::MultistageSimulator::set_stage,
             py::arg("env_id"), py::arg("stage"))
        .def("get_stage", &mc189::MultistageSimulator::get_stage,
             py::arg("env_id") = 0)
        .def("get_stage_progress", &mc189::MultistageSimulator::get_stage_progress,
             py::arg("env_id") = 0)
        .def("teleport_to_dimension", &mc189::MultistageSimulator::teleport_to_dimension,
             py::arg("env_id"), py::arg("dim"),
             py::arg("x") = 0.0f, py::arg("y") = 64.0f, py::arg("z") = 0.0f)
        .def("get_dimension", &mc189::MultistageSimulator::get_dimension,
             py::arg("env_id") = 0)
        .def("get_observations", [](mc189::MultistageSimulator& self) {
            return py::array_t<float>(
                {static_cast<py::ssize_t>(self.num_envs()),
                 static_cast<py::ssize_t>(mc189::OBSERVATION_SIZE)},
                self.get_observations());
        })
        .def("get_extended_observations", [](mc189::MultistageSimulator& self) {
            return py::array_t<float>(
                {static_cast<py::ssize_t>(self.num_envs()),
                 static_cast<py::ssize_t>(mc189::MultistageSimulator::EXTENDED_OBS_SIZE)},
                self.get_extended_observations());
        })
        .def("get_rewards", [](mc189::MultistageSimulator& self) {
            return py::array_t<float>(
                {static_cast<py::ssize_t>(self.num_envs())},
                self.get_rewards());
        })
        .def("get_dones", [](mc189::MultistageSimulator& self) {
            return py::array_t<uint8_t>(
                {static_cast<py::ssize_t>(self.num_envs())},
                self.get_dones());
        })
        .def("num_envs", &mc189::MultistageSimulator::num_envs)
        .def_property_readonly_static("obs_dim", [](py::object) {
            return mc189::OBSERVATION_SIZE;
        })
        .def_property_readonly_static("extended_obs_dim", [](py::object) {
            return mc189::MultistageSimulator::EXTENDED_OBS_SIZE;
        });

    init_gym_wrapper(m);
}
