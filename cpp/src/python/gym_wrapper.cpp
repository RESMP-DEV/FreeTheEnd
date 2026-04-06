/**
 * Gymnasium-compatible wrapper extensions for MinecraftSimulator.
 *
 * Provides additional C++ utilities for efficient Gymnasium integration,
 * including space definitions and batch operations.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace minecraft_sim {

/**
 * Space information for Gymnasium compatibility.
 */
struct SpaceInfo {
    std::string space_type;  // "discrete", "box", "multi_discrete"
    std::vector<int64_t> shape;
    float low;
    float high;
    int64_t n;  // For discrete spaces
    std::string dtype;
};

/**
 * Gymnasium space helpers implemented in C++ for efficiency.
 */
class GymSpaceHelper {
public:
    static constexpr int STATE_DIM = 32;
    static constexpr int NUM_ACTIONS = 16;
    static constexpr int OBS_WIDTH = 64;
    static constexpr int OBS_HEIGHT = 64;
    static constexpr int OBS_CHANNELS = 3;

    /**
     * Get action space information.
     */
    static py::dict get_action_space_info() {
        py::dict info;
        info["type"] = "discrete";
        info["n"] = NUM_ACTIONS;
        info["dtype"] = "int64";
        return info;
    }

    /**
     * Get observation space information for state vector.
     */
    static py::dict get_observation_space_info() {
        py::dict info;
        info["type"] = "box";
        info["shape"] = py::make_tuple(STATE_DIM);
        info["low"] = -std::numeric_limits<float>::infinity();
        info["high"] = std::numeric_limits<float>::infinity();
        info["dtype"] = "float32";
        return info;
    }

    /**
     * Get observation space information for RGB images.
     */
    static py::dict get_image_observation_space_info() {
        py::dict info;
        info["type"] = "box";
        info["shape"] = py::make_tuple(OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS);
        info["low"] = 0;
        info["high"] = 255;
        info["dtype"] = "uint8";
        return info;
    }

    /**
     * Sample random action.
     */
    static int sample_action(std::mt19937_64& rng) {
        return rng() % NUM_ACTIONS;
    }

    /**
     * Sample batch of random actions.
     */
    static py::array_t<int> sample_actions(int batch_size, uint64_t seed = 0) {
        auto actions = py::array_t<int>(batch_size);
        auto buf = actions.mutable_unchecked<1>();

        std::mt19937_64 rng(seed);
        for (int i = 0; i < batch_size; i++) {
            buf(i) = rng() % NUM_ACTIONS;
        }

        return actions;
    }

    /**
     * Validate action is within bounds.
     */
    static bool is_valid_action(int action) {
        return action >= 0 && action < NUM_ACTIONS;
    }

    /**
     * Validate batch of actions.
     */
    static py::array_t<bool> validate_actions(py::array_t<int> actions) {
        auto actions_buf = actions.unchecked<1>();
        auto result = py::array_t<bool>(actions_buf.size());
        auto result_buf = result.mutable_unchecked<1>();

        for (ssize_t i = 0; i < actions_buf.size(); i++) {
            result_buf(i) = is_valid_action(actions_buf(i));
        }

        return result;
    }

    /**
     * Clip observations to valid range.
     */
    static py::array_t<float> clip_observations(
        py::array_t<float> obs,
        float low = -10.0f,
        float high = 10.0f) {

        auto obs_buf = obs.unchecked<1>();
        auto result = py::array_t<float>(obs_buf.size());
        auto result_buf = result.mutable_unchecked<1>();

        for (ssize_t i = 0; i < obs_buf.size(); i++) {
            result_buf(i) = std::max(low, std::min(high, obs_buf(i)));
        }

        return result;
    }
};

/**
 * Efficient batch utilities for vectorized environments.
 */
class BatchUtils {
public:
    /**
     * Stack observations from multiple environments.
     * Input: list of (obs_dim,) arrays
     * Output: (num_envs, obs_dim) array
     */
    static py::array_t<float> stack_observations(py::list obs_list) {
        if (obs_list.empty()) {
            return py::array_t<float>(std::vector<ssize_t>{0, 0});
        }

        auto first = obs_list[0].cast<py::array_t<float>>();
        auto first_buf = first.unchecked<1>();
        int obs_dim = first_buf.size();
        int num_envs = obs_list.size();

        std::vector<ssize_t> shape = {num_envs, obs_dim};
        auto result = py::array_t<float>(shape);
        auto result_buf = result.mutable_unchecked<2>();

        for (int i = 0; i < num_envs; i++) {
            auto obs = obs_list[i].cast<py::array_t<float>>();
            auto obs_buf = obs.unchecked<1>();
            for (int j = 0; j < obs_dim; j++) {
                result_buf(i, j) = obs_buf(j);
            }
        }

        return result;
    }

    /**
     * Compute advantage estimates using GAE.
     * Args:
     *   rewards: (T, N) rewards
     *   values: (T+1, N) value estimates
     *   dones: (T, N) episode termination flags
     *   gamma: discount factor
     *   gae_lambda: GAE lambda
     * Returns:
     *   advantages: (T, N)
     */
    static py::array_t<float> compute_gae(
        py::array_t<float> rewards,
        py::array_t<float> values,
        py::array_t<bool> dones,
        float gamma = 0.99f,
        float gae_lambda = 0.95f) {

        auto rew_buf = rewards.unchecked<2>();
        auto val_buf = values.unchecked<2>();
        auto done_buf = dones.unchecked<2>();

        int T = rew_buf.shape(0);
        int N = rew_buf.shape(1);

        std::vector<ssize_t> shape = {T, N};
        auto advantages = py::array_t<float>(shape);
        auto adv_buf = advantages.mutable_unchecked<2>();

        // Initialize last advantage
        std::vector<float> last_gae(N, 0.0f);

        // Compute GAE backwards
        for (int t = T - 1; t >= 0; t--) {
            for (int n = 0; n < N; n++) {
                float next_val = val_buf(t + 1, n);
                float delta = rew_buf(t, n) + gamma * next_val * (1.0f - done_buf(t, n)) - val_buf(t, n);
                last_gae[n] = delta + gamma * gae_lambda * (1.0f - done_buf(t, n)) * last_gae[n];
                adv_buf(t, n) = last_gae[n];
            }
        }

        return advantages;
    }

    /**
     * Normalize observations using running statistics.
     */
    static py::array_t<float> normalize_obs(
        py::array_t<float> obs,
        py::array_t<float> mean,
        py::array_t<float> std,
        float clip = 10.0f) {

        auto obs_buf = obs.unchecked<2>();
        auto mean_buf = mean.unchecked<1>();
        auto std_buf = std.unchecked<1>();

        int N = obs_buf.shape(0);
        int D = obs_buf.shape(1);

        std::vector<ssize_t> shape = {N, D};
        auto result = py::array_t<float>(shape);
        auto result_buf = result.mutable_unchecked<2>();

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                float normalized = (obs_buf(i, j) - mean_buf(j)) / (std_buf(j) + 1e-8f);
                result_buf(i, j) = std::max(-clip, std::min(clip, normalized));
            }
        }

        return result;
    }

    /**
     * Compute discounted returns.
     */
    static py::array_t<float> compute_returns(
        py::array_t<float> rewards,
        py::array_t<bool> dones,
        float gamma = 0.99f) {

        auto rew_buf = rewards.unchecked<2>();
        auto done_buf = dones.unchecked<2>();

        int T = rew_buf.shape(0);
        int N = rew_buf.shape(1);

        std::vector<ssize_t> shape = {T, N};
        auto returns = py::array_t<float>(shape);
        auto ret_buf = returns.mutable_unchecked<2>();

        std::vector<float> running_return(N, 0.0f);

        for (int t = T - 1; t >= 0; t--) {
            for (int n = 0; n < N; n++) {
                running_return[n] = rew_buf(t, n) + gamma * running_return[n] * (1.0f - done_buf(t, n));
                ret_buf(t, n) = running_return[n];
            }
        }

        return returns;
    }
};

/**
 * Running mean/std tracker for observation normalization.
 */
class RunningMeanStd {
public:
    explicit RunningMeanStd(int dim, float epsilon = 1e-4f)
        : dim_(dim), epsilon_(epsilon), count_(0) {
        mean_.resize(dim, 0.0);
        var_.resize(dim, 1.0);
    }

    void update(py::array_t<float> batch) {
        auto buf = batch.unchecked<2>();
        int batch_size = buf.shape(0);

        // Compute batch statistics
        std::vector<double> batch_mean(dim_, 0.0);
        std::vector<double> batch_var(dim_, 0.0);

        for (int j = 0; j < dim_; j++) {
            for (int i = 0; i < batch_size; i++) {
                batch_mean[j] += buf(i, j);
            }
            batch_mean[j] /= batch_size;

            for (int i = 0; i < batch_size; i++) {
                double diff = buf(i, j) - batch_mean[j];
                batch_var[j] += diff * diff;
            }
            batch_var[j] /= batch_size;
        }

        // Update running statistics
        double delta, new_mean;
        int64_t total_count = count_ + batch_size;

        for (int j = 0; j < dim_; j++) {
            delta = batch_mean[j] - mean_[j];
            new_mean = mean_[j] + delta * batch_size / total_count;

            double m_a = var_[j] * count_;
            double m_b = batch_var[j] * batch_size;
            double m2 = m_a + m_b + delta * delta * count_ * batch_size / total_count;

            mean_[j] = new_mean;
            var_[j] = m2 / total_count;
        }

        count_ = total_count;
    }

    py::array_t<float> get_mean() const {
        auto result = py::array_t<float>(dim_);
        auto buf = result.mutable_unchecked<1>();
        for (int i = 0; i < dim_; i++) {
            buf(i) = static_cast<float>(mean_[i]);
        }
        return result;
    }

    py::array_t<float> get_std() const {
        auto result = py::array_t<float>(dim_);
        auto buf = result.mutable_unchecked<1>();
        for (int i = 0; i < dim_; i++) {
            buf(i) = static_cast<float>(std::sqrt(var_[i] + epsilon_));
        }
        return result;
    }

    int64_t count() const { return count_; }

private:
    int dim_;
    float epsilon_;
    int64_t count_;
    std::vector<double> mean_;
    std::vector<double> var_;
};

}  // namespace minecraft_sim

/**
 * Additional module bindings for Gymnasium utilities.
 */
void init_gym_wrapper(py::module_& m) {
    using namespace minecraft_sim;

    // GymSpaceHelper as a module-level namespace
    py::class_<GymSpaceHelper>(m, "GymSpaceHelper",
        "Gymnasium space utilities")
        .def_static("get_action_space_info", &GymSpaceHelper::get_action_space_info,
            "Get action space information dict")
        .def_static("get_observation_space_info", &GymSpaceHelper::get_observation_space_info,
            "Get observation space information dict")
        .def_static("get_image_observation_space_info", &GymSpaceHelper::get_image_observation_space_info,
            "Get image observation space information dict")
        .def_static("sample_actions", &GymSpaceHelper::sample_actions,
            py::arg("batch_size"),
            py::arg("seed") = 0,
            "Sample batch of random actions")
        .def_static("is_valid_action", &GymSpaceHelper::is_valid_action,
            py::arg("action"),
            "Check if action is valid")
        .def_static("validate_actions", &GymSpaceHelper::validate_actions,
            py::arg("actions"),
            "Validate batch of actions")
        .def_static("clip_observations", &GymSpaceHelper::clip_observations,
            py::arg("obs"),
            py::arg("low") = -10.0f,
            py::arg("high") = 10.0f,
            "Clip observations to valid range");

    // BatchUtils for efficient batch operations
    py::class_<BatchUtils>(m, "BatchUtils",
        "Batch utilities for vectorized environments")
        .def_static("stack_observations", &BatchUtils::stack_observations,
            py::arg("obs_list"),
            "Stack list of observations into batch array")
        .def_static("compute_gae", &BatchUtils::compute_gae,
            py::arg("rewards"),
            py::arg("values"),
            py::arg("dones"),
            py::arg("gamma") = 0.99f,
            py::arg("gae_lambda") = 0.95f,
            "Compute Generalized Advantage Estimation")
        .def_static("normalize_obs", &BatchUtils::normalize_obs,
            py::arg("obs"),
            py::arg("mean"),
            py::arg("std"),
            py::arg("clip") = 10.0f,
            "Normalize observations using mean/std")
        .def_static("compute_returns", &BatchUtils::compute_returns,
            py::arg("rewards"),
            py::arg("dones"),
            py::arg("gamma") = 0.99f,
            "Compute discounted returns");

    // RunningMeanStd for observation normalization
    py::class_<RunningMeanStd>(m, "RunningMeanStd",
        "Running mean/std tracker for normalization")
        .def(py::init<int, float>(),
            py::arg("dim"),
            py::arg("epsilon") = 1e-4f,
            "Create tracker for given dimension")
        .def("update", &RunningMeanStd::update,
            py::arg("batch"),
            "Update statistics with new batch")
        .def("get_mean", &RunningMeanStd::get_mean,
            "Get current mean")
        .def("get_std", &RunningMeanStd::get_std,
            "Get current standard deviation")
        .def_property_readonly("count", &RunningMeanStd::count,
            "Number of samples seen");
}
