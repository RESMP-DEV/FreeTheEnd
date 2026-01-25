// Performance benchmark suite for Minecraft simulation backend
// Uses Google Benchmark library
// Build: g++ -std=c++17 -O3 -o benchmark_backend benchmark_backend.cpp -lbenchmark -lpthread

#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

// Forward declarations for simulation backend types
// These would normally come from the actual backend headers

namespace minecraft_sim {

// Chunk generation constants
constexpr int CHUNK_SIZE_X = 16;
constexpr int CHUNK_SIZE_Y = 256;
constexpr int CHUNK_SIZE_Z = 16;
constexpr int BLOCKS_PER_CHUNK = CHUNK_SIZE_X * CHUNK_SIZE_Y * CHUNK_SIZE_Z;

// Observation encoding constants
constexpr int OBS_WIDTH = 64;
constexpr int OBS_HEIGHT = 64;
constexpr int OBS_CHANNELS = 3;
constexpr int OBS_SIZE = OBS_WIDTH * OBS_HEIGHT * OBS_CHANNELS;

// Block types for terrain generation
enum class BlockType : uint8_t {
    Air = 0,
    Stone = 1,
    Dirt = 2,
    Grass = 3,
    Water = 4,
    Sand = 5,
    Wood = 6,
    Leaves = 7
};

// Simple 3D position
struct Vec3i {
    int x, y, z;
};

// Chunk data structure
struct Chunk {
    std::vector<BlockType> blocks;
    int chunk_x, chunk_z;

    Chunk() : blocks(BLOCKS_PER_CHUNK, BlockType::Air), chunk_x(0), chunk_z(0) {}

    BlockType get_block(int x, int y, int z) const {
        if (x < 0 || x >= CHUNK_SIZE_X || y < 0 || y >= CHUNK_SIZE_Y || z < 0 || z >= CHUNK_SIZE_Z) {
            return BlockType::Air;
        }
        return blocks[y * CHUNK_SIZE_X * CHUNK_SIZE_Z + z * CHUNK_SIZE_X + x];
    }

    void set_block(int x, int y, int z, BlockType type) {
        if (x >= 0 && x < CHUNK_SIZE_X && y >= 0 && y < CHUNK_SIZE_Y && z >= 0 && z < CHUNK_SIZE_Z) {
            blocks[y * CHUNK_SIZE_X * CHUNK_SIZE_Z + z * CHUNK_SIZE_X + x] = type;
        }
    }
};

// Simple noise-based terrain generator
class ChunkGenerator {
public:
    ChunkGenerator(uint64_t seed) : rng_(seed) {}

    void generate(Chunk& chunk, int chunk_x, int chunk_z) {
        chunk.chunk_x = chunk_x;
        chunk.chunk_z = chunk_z;

        // Simple heightmap-based generation
        for (int x = 0; x < CHUNK_SIZE_X; ++x) {
            for (int z = 0; z < CHUNK_SIZE_Z; ++z) {
                int world_x = chunk_x * CHUNK_SIZE_X + x;
                int world_z = chunk_z * CHUNK_SIZE_Z + z;

                // Pseudo-random height based on position
                int height = 64 + static_cast<int>(
                    10.0 * std::sin(world_x * 0.05) * std::cos(world_z * 0.05) +
                    5.0 * std::sin(world_x * 0.1 + world_z * 0.1)
                );
                height = std::max(1, std::min(height, 200));

                for (int y = 0; y < height; ++y) {
                    BlockType type;
                    if (y < height - 4) {
                        type = BlockType::Stone;
                    } else if (y < height - 1) {
                        type = BlockType::Dirt;
                    } else {
                        type = BlockType::Grass;
                    }
                    chunk.set_block(x, y, z, type);
                }
            }
        }
    }

private:
    std::mt19937_64 rng_;
};

// A* pathfinding implementation
class Pathfinder {
public:
    struct Node {
        Vec3i pos;
        float g_cost;
        float h_cost;
        int parent_idx;

        float f_cost() const { return g_cost + h_cost; }
    };

    std::vector<Vec3i> find_path(const Chunk& chunk, Vec3i start, Vec3i goal) {
        // Simple A* implementation
        std::vector<Node> open_list;
        std::vector<Node> closed_list;
        std::vector<Vec3i> path;

        Node start_node{start, 0.0f, heuristic(start, goal), -1};
        open_list.push_back(start_node);

        int iterations = 0;
        constexpr int MAX_ITERATIONS = 1000;

        while (!open_list.empty() && iterations < MAX_ITERATIONS) {
            ++iterations;

            // Find node with lowest f_cost
            size_t best_idx = 0;
            for (size_t i = 1; i < open_list.size(); ++i) {
                if (open_list[i].f_cost() < open_list[best_idx].f_cost()) {
                    best_idx = i;
                }
            }

            Node current = open_list[best_idx];
            open_list.erase(open_list.begin() + best_idx);
            closed_list.push_back(current);

            // Check if reached goal
            if (current.pos.x == goal.x && current.pos.y == goal.y && current.pos.z == goal.z) {
                // Reconstruct path
                int idx = static_cast<int>(closed_list.size()) - 1;
                while (idx >= 0) {
                    path.push_back(closed_list[idx].pos);
                    idx = closed_list[idx].parent_idx;
                }
                std::reverse(path.begin(), path.end());
                return path;
            }

            // Expand neighbors (6-connected grid)
            static const int dx[] = {1, -1, 0, 0, 0, 0};
            static const int dy[] = {0, 0, 1, -1, 0, 0};
            static const int dz[] = {0, 0, 0, 0, 1, -1};

            for (int i = 0; i < 6; ++i) {
                Vec3i neighbor{current.pos.x + dx[i], current.pos.y + dy[i], current.pos.z + dz[i]};

                // Check bounds and walkability
                if (neighbor.y < 0 || neighbor.y >= CHUNK_SIZE_Y) continue;
                if (neighbor.x < 0 || neighbor.x >= CHUNK_SIZE_X) continue;
                if (neighbor.z < 0 || neighbor.z >= CHUNK_SIZE_Z) continue;
                if (chunk.get_block(neighbor.x, neighbor.y, neighbor.z) != BlockType::Air) continue;

                // Check if in closed list
                bool in_closed = false;
                for (const auto& n : closed_list) {
                    if (n.pos.x == neighbor.x && n.pos.y == neighbor.y && n.pos.z == neighbor.z) {
                        in_closed = true;
                        break;
                    }
                }
                if (in_closed) continue;

                float new_g = current.g_cost + 1.0f;

                // Check if in open list
                bool in_open = false;
                for (auto& n : open_list) {
                    if (n.pos.x == neighbor.x && n.pos.y == neighbor.y && n.pos.z == neighbor.z) {
                        in_open = true;
                        if (new_g < n.g_cost) {
                            n.g_cost = new_g;
                            n.parent_idx = static_cast<int>(closed_list.size()) - 1;
                        }
                        break;
                    }
                }

                if (!in_open) {
                    Node neighbor_node{neighbor, new_g, heuristic(neighbor, goal),
                                       static_cast<int>(closed_list.size()) - 1};
                    open_list.push_back(neighbor_node);
                }
            }
        }

        return path; // Empty if no path found
    }

private:
    float heuristic(const Vec3i& a, const Vec3i& b) {
        return static_cast<float>(std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z));
    }
};

// Observation encoder (simulates rendering to observation tensor)
class ObservationEncoder {
public:
    ObservationEncoder() : buffer_(OBS_SIZE) {}

    void encode(const Chunk& chunk, Vec3i agent_pos, uint8_t* output) {
        // Simulate observation encoding: raycast-based view rendering
        for (int y = 0; y < OBS_HEIGHT; ++y) {
            for (int x = 0; x < OBS_WIDTH; ++x) {
                // Simplified: just sample nearby blocks and encode as RGB
                int sample_x = agent_pos.x + (x - OBS_WIDTH / 2) / 4;
                int sample_y = agent_pos.y + (OBS_HEIGHT / 2 - y) / 4;
                int sample_z = agent_pos.z;

                sample_x = std::max(0, std::min(sample_x, CHUNK_SIZE_X - 1));
                sample_y = std::max(0, std::min(sample_y, CHUNK_SIZE_Y - 1));
                sample_z = std::max(0, std::min(sample_z, CHUNK_SIZE_Z - 1));

                BlockType block = chunk.get_block(sample_x, sample_y, sample_z);

                // Map block type to color
                uint8_t r, g, b;
                switch (block) {
                    case BlockType::Stone: r = 128; g = 128; b = 128; break;
                    case BlockType::Dirt:  r = 139; g = 69;  b = 19;  break;
                    case BlockType::Grass: r = 34;  g = 139; b = 34;  break;
                    case BlockType::Water: r = 0;   g = 0;   b = 255; break;
                    case BlockType::Sand:  r = 238; g = 214; b = 175; break;
                    case BlockType::Wood:  r = 139; g = 90;  b = 43;  break;
                    case BlockType::Leaves:r = 0;   g = 100; b = 0;   break;
                    default:               r = 135; g = 206; b = 235; break; // Sky
                }

                int idx = (y * OBS_WIDTH + x) * OBS_CHANNELS;
                output[idx + 0] = r;
                output[idx + 1] = g;
                output[idx + 2] = b;
            }
        }
    }

private:
    std::vector<uint8_t> buffer_;
};

// Single environment state
struct Environment {
    Chunk chunk;
    Vec3i agent_pos;
    uint64_t step_count;
    bool done;
    std::vector<uint8_t> observation;

    Environment() : agent_pos{8, 100, 8}, step_count(0), done(false), observation(OBS_SIZE) {}
};

// Vectorized environment manager
class VecEnv {
public:
    VecEnv(int num_envs, uint64_t seed)
        : num_envs_(num_envs), generator_(seed), encoder_(), rng_(seed) {
        envs_.resize(num_envs);
        reset_all();
    }

    void reset_all() {
        for (int i = 0; i < num_envs_; ++i) {
            reset_env(i);
        }
    }

    void reset_env(int idx) {
        auto& env = envs_[idx];
        generator_.generate(env.chunk, static_cast<int>(rng_() % 1000), static_cast<int>(rng_() % 1000));
        env.agent_pos = {8, find_spawn_height(env.chunk, 8, 8), 8};
        env.step_count = 0;
        env.done = false;
    }

    void step(const std::vector<int>& actions, std::vector<float>& rewards, std::vector<bool>& dones) {
        for (int i = 0; i < num_envs_; ++i) {
            step_env(i, actions[i], rewards[i], dones[i]);
        }
    }

    void step_env(int idx, int action, float& reward, bool& done) {
        auto& env = envs_[idx];

        // Actions: 0=noop, 1=forward, 2=back, 3=left, 4=right, 5=jump
        int dx = 0, dy = 0, dz = 0;
        switch (action) {
            case 1: dz = 1; break;
            case 2: dz = -1; break;
            case 3: dx = -1; break;
            case 4: dx = 1; break;
            case 5: dy = 1; break;
        }

        // Attempt move
        Vec3i new_pos = {env.agent_pos.x + dx, env.agent_pos.y + dy, env.agent_pos.z + dz};

        // Clamp to chunk bounds
        new_pos.x = std::max(0, std::min(new_pos.x, CHUNK_SIZE_X - 1));
        new_pos.y = std::max(0, std::min(new_pos.y, CHUNK_SIZE_Y - 1));
        new_pos.z = std::max(0, std::min(new_pos.z, CHUNK_SIZE_Z - 1));

        // Check collision
        if (env.chunk.get_block(new_pos.x, new_pos.y, new_pos.z) == BlockType::Air) {
            env.agent_pos = new_pos;
        }

        // Apply gravity
        while (env.agent_pos.y > 0 &&
               env.chunk.get_block(env.agent_pos.x, env.agent_pos.y - 1, env.agent_pos.z) == BlockType::Air) {
            env.agent_pos.y--;
        }

        ++env.step_count;

        // Reward: distance from spawn (exploration reward)
        reward = 0.01f * std::sqrt(
            static_cast<float>((env.agent_pos.x - 8) * (env.agent_pos.x - 8) +
                               (env.agent_pos.z - 8) * (env.agent_pos.z - 8))
        );

        // Episode termination
        done = (env.step_count >= 1000) || (env.agent_pos.y <= 0);
        env.done = done;

        if (done) {
            reset_env(idx);
        }
    }

    void get_observations(std::vector<uint8_t*>& obs_ptrs) {
        for (int i = 0; i < num_envs_; ++i) {
            encoder_.encode(envs_[i].chunk, envs_[i].agent_pos, envs_[i].observation.data());
            obs_ptrs[i] = envs_[i].observation.data();
        }
    }

    int num_envs() const { return num_envs_; }

    size_t memory_per_env() const {
        return sizeof(Environment) + BLOCKS_PER_CHUNK * sizeof(BlockType) + OBS_SIZE;
    }

private:
    int find_spawn_height(const Chunk& chunk, int x, int z) {
        for (int y = CHUNK_SIZE_Y - 1; y >= 0; --y) {
            if (chunk.get_block(x, y, z) != BlockType::Air) {
                return y + 1;
            }
        }
        return 64;
    }

    int num_envs_;
    std::vector<Environment> envs_;
    ChunkGenerator generator_;
    ObservationEncoder encoder_;
    std::mt19937_64 rng_;
};

} // namespace minecraft_sim

// =============================================================================
// Benchmarks
// =============================================================================

static void bench_chunk_generation(benchmark::State& state) {
    minecraft_sim::ChunkGenerator generator(42);
    minecraft_sim::Chunk chunk;
    int chunk_x = 0;

    for (auto _ : state) {
        generator.generate(chunk, chunk_x++, 0);
        benchmark::DoNotOptimize(chunk.blocks.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("chunks/sec");
}
BENCHMARK(bench_chunk_generation);

static void bench_pathfinding(benchmark::State& state) {
    minecraft_sim::ChunkGenerator generator(42);
    minecraft_sim::Chunk chunk;
    generator.generate(chunk, 0, 0);

    minecraft_sim::Pathfinder pathfinder;

    // Find valid start and end positions (air blocks)
    minecraft_sim::Vec3i start{8, 100, 8};
    minecraft_sim::Vec3i goal{12, 100, 12};

    for (auto _ : state) {
        auto path = pathfinder.find_path(chunk, start, goal);
        benchmark::DoNotOptimize(path.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("paths/sec");
}
BENCHMARK(bench_pathfinding);

static void bench_observation_encoding(benchmark::State& state) {
    minecraft_sim::ChunkGenerator generator(42);
    minecraft_sim::Chunk chunk;
    generator.generate(chunk, 0, 0);

    minecraft_sim::ObservationEncoder encoder;
    std::vector<uint8_t> observation(minecraft_sim::OBS_SIZE);
    minecraft_sim::Vec3i agent_pos{8, 100, 8};

    for (auto _ : state) {
        encoder.encode(chunk, agent_pos, observation.data());
        benchmark::DoNotOptimize(observation.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("obs/sec");
}
BENCHMARK(bench_observation_encoding);

static void bench_step_single(benchmark::State& state) {
    minecraft_sim::VecEnv env(1, 42);
    std::vector<int> actions(1);
    std::vector<float> rewards(1);
    std::vector<bool> dones(1);
    std::mt19937 rng(42);

    for (auto _ : state) {
        actions[0] = rng() % 6;
        env.step(actions, rewards, dones);
        benchmark::DoNotOptimize(rewards.data());
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel("SPS");
}
BENCHMARK(bench_step_single);

static void bench_step_batch_64(benchmark::State& state) {
    constexpr int NUM_ENVS = 64;
    minecraft_sim::VecEnv env(NUM_ENVS, 42);
    std::vector<int> actions(NUM_ENVS);
    std::vector<float> rewards(NUM_ENVS);
    std::vector<bool> dones(NUM_ENVS);
    std::mt19937 rng(42);

    for (auto _ : state) {
        for (int i = 0; i < NUM_ENVS; ++i) {
            actions[i] = rng() % 6;
        }
        env.step(actions, rewards, dones);
        benchmark::DoNotOptimize(rewards.data());
    }

    state.SetItemsProcessed(state.iterations() * NUM_ENVS);
    state.SetLabel("SPS");
}
BENCHMARK(bench_step_batch_64);

static void bench_step_batch_256(benchmark::State& state) {
    constexpr int NUM_ENVS = 256;
    minecraft_sim::VecEnv env(NUM_ENVS, 42);
    std::vector<int> actions(NUM_ENVS);
    std::vector<float> rewards(NUM_ENVS);
    std::vector<bool> dones(NUM_ENVS);
    std::mt19937 rng(42);

    for (auto _ : state) {
        for (int i = 0; i < NUM_ENVS; ++i) {
            actions[i] = rng() % 6;
        }
        env.step(actions, rewards, dones);
        benchmark::DoNotOptimize(rewards.data());
    }

    state.SetItemsProcessed(state.iterations() * NUM_ENVS);
    state.SetLabel("SPS");
}
BENCHMARK(bench_step_batch_256);

static void bench_reset_batch(benchmark::State& state) {
    constexpr int NUM_ENVS = 64;
    minecraft_sim::VecEnv env(NUM_ENVS, 42);

    for (auto _ : state) {
        env.reset_all();
        benchmark::DoNotOptimize(&env);
    }

    state.SetItemsProcessed(state.iterations() * NUM_ENVS);
    state.SetLabel("resets/sec");
}
BENCHMARK(bench_reset_batch);

static void bench_memory_usage(benchmark::State& state) {
    minecraft_sim::VecEnv env(1, 42);
    size_t mem_per_env = env.memory_per_env();

    for (auto _ : state) {
        benchmark::DoNotOptimize(mem_per_env);
    }

    double mb_per_env = static_cast<double>(mem_per_env) / (1024.0 * 1024.0);
    state.counters["MB_per_env"] = benchmark::Counter(mb_per_env, benchmark::Counter::kDefaults);
}
BENCHMARK(bench_memory_usage);

// Custom reporter that outputs in the requested format
class MinecraftBenchmarkReporter : public benchmark::ConsoleReporter {
public:
    void ReportRuns(const std::vector<Run>& reports) override {
        // Store results for final summary
        for (const auto& run : reports) {
            results_.push_back(run);
        }
    }

    void Finalize() override {
        std::printf("\nBenchmark Results:\n");
        std::printf("-------------------------------------------------\n");

        for (const auto& run : results_) {
            std::string name = run.benchmark_name();

            // Extract base name without prefix
            size_t pos = name.find("bench_");
            if (pos != std::string::npos) {
                name = name.substr(pos + 6);
            }

            // Format based on benchmark type
            if (name.find("memory") != std::string::npos) {
                auto it = run.counters.find("MB_per_env");
                if (it != run.counters.end()) {
                    std::printf("%-24s%.1f MB\n", "memory_per_env:", it->second);
                }
            } else {
                double items_per_sec = run.items_per_second;
                const char* unit = "";

                if (name.find("chunk") != std::string::npos) {
                    unit = "chunks/sec";
                    name = "chunk_generation:";
                } else if (name.find("pathfind") != std::string::npos) {
                    unit = "paths/sec";
                    name = "pathfinding_a_star:";
                } else if (name.find("observation") != std::string::npos) {
                    unit = "obs/sec";
                    name = "observation_encode:";
                } else if (name.find("step_single") != std::string::npos) {
                    unit = "SPS";
                    name = "step_1_env:";
                } else if (name.find("step_batch_64") != std::string::npos) {
                    unit = "SPS";
                    name = "step_64_envs:";
                } else if (name.find("step_batch_256") != std::string::npos) {
                    unit = "SPS";
                    name = "step_256_envs:";
                } else if (name.find("reset") != std::string::npos) {
                    unit = "resets/sec";
                    name = "reset_64_envs:";
                }

                std::printf("%-24s%'-.0f %s\n", name.c_str(), items_per_sec, unit);
            }
        }

        std::printf("-------------------------------------------------\n");
    }

private:
    std::vector<Run> results_;
};

int main(int argc, char** argv) {
    // Enable locale for thousands separator
    std::setlocale(LC_NUMERIC, "");

    benchmark::Initialize(&argc, argv);

    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    // Use custom reporter for formatted output
    MinecraftBenchmarkReporter reporter;
    benchmark::RunSpecifiedBenchmarks(&reporter);
    reporter.Finalize();

    return 0;
}
