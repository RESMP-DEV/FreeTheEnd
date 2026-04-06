// Minecraft 1.8.9 Simulator - high-throughput tick executor
// Target: 500K+ steps/second on Apple Silicon via MoltenVK

#include "simulator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <vulkan/vulkan.h>

namespace mc189::internal {

namespace {
constexpr uint32_t kChannels = 3;
constexpr float kSkyColor[3] = {0.53f, 0.81f, 0.92f};
constexpr float kGroundColor[3] = {0.25f, 0.61f, 0.22f};
constexpr double kPi = 3.141592653589793;
constexpr uint64_t kNsPerUs = 1000ULL;
}  // namespace

// -----------------------------------------------------------------------------
// VulkanContext
// -----------------------------------------------------------------------------

bool VulkanContext::init() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "mc189_sim";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "mc189";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    std::vector<const char*> extensions;
#ifdef __APPLE__
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#endif

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instanceInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
#ifdef __APPLE__
    instanceInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
        return false;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        shutdown();
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (VkPhysicalDevice dev : devices) {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, families.data());
        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physicalDevice = dev;
                computeQueueFamily = i;
                break;
            }
        }
        if (physicalDevice) break;
    }

    if (!physicalDevice) {
        shutdown();
        return false;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = computeQueueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &queuePriority;

    std::vector<const char*> deviceExtensions;
#ifdef __APPLE__
    deviceExtensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
#endif

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.empty() ? nullptr : deviceExtensions.data();

    if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
        shutdown();
        return false;
    }

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, computeQueueFamily, 0, &queue);
    computeQueue = reinterpret_cast<VkQueue_T*>(queue);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
        shutdown();
        return false;
    }
    commandPool = reinterpret_cast<VkCommandPool_T*>(pool);

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        shutdown();
        return false;
    }
    computeFence = reinterpret_cast<VkFence_T*>(fence);

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    std::snprintf(gpuName, sizeof(gpuName), "%s", props.deviceName);

    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    gpuMemory = 0;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            gpuMemory += memProps.memoryHeaps[i].size;
        }
    }

    return true;
}

void VulkanContext::shutdown() {
    if (device) {
        vkDeviceWaitIdle(device);
    }
    if (computeFence) {
        vkDestroyFence(device, reinterpret_cast<VkFence>(computeFence), nullptr);
        computeFence = nullptr;
    }
    if (commandPool) {
        vkDestroyCommandPool(device, reinterpret_cast<VkCommandPool>(commandPool), nullptr);
        commandPool = nullptr;
    }
    if (device) {
        vkDestroyDevice(device, nullptr);
        device = nullptr;
    }
    if (instance) {
        vkDestroyInstance(instance, nullptr);
        instance = nullptr;
    }
    physicalDevice = nullptr;
    computeQueue = nullptr;
    computeQueueFamily = 0;
}

// -----------------------------------------------------------------------------
// WorldState
// -----------------------------------------------------------------------------

ChunkData* WorldState::getChunk(int32_t cx, int32_t cz) {
    for (auto& chunk : chunks) {
        if (chunk->chunkX == cx && chunk->chunkZ == cz) {
            return chunk.get();
        }
    }
    auto chunk = std::make_unique<ChunkData>();
    chunk->chunkX = cx;
    chunk->chunkZ = cz;
    chunks.emplace_back(std::move(chunk));
    minChunkX = std::min(minChunkX, cx);
    maxChunkX = std::max(maxChunkX, cx);
    minChunkZ = std::min(minChunkZ, cz);
    maxChunkZ = std::max(maxChunkZ, cz);
    return chunks.back().get();
}

const ChunkData* WorldState::getChunk(int32_t cx, int32_t cz) const {
    for (const auto& chunk : chunks) {
        if (chunk->chunkX == cx && chunk->chunkZ == cz) {
            return chunk.get();
        }
    }
    return nullptr;
}

BlockType WorldState::getBlock(int x, int y, int z) const {
    int32_t cx = x >> 4;
    int32_t cz = z >> 4;
    const ChunkData* chunk = getChunk(cx, cz);
    if (!chunk) return BlockType::Air;
    int lx = x & 15;
    int lz = z & 15;
    return chunk->get(lx, y, lz);
}

void WorldState::setBlock(int x, int y, int z, BlockType type) {
    int32_t cx = x >> 4;
    int32_t cz = z >> 4;
    ChunkData* chunk = getChunk(cx, cz);
    int lx = x & 15;
    int lz = z & 15;
    chunk->set(lx, y, lz, type);
}

void WorldState::tick() {
    worldTime += 1;
}

// -----------------------------------------------------------------------------
// PlayerState
// -----------------------------------------------------------------------------

void PlayerState::reset(double spawnX, double spawnY, double spawnZ) {
    posX = spawnX;
    posY = spawnY;
    posZ = spawnZ;
    velX = velY = velZ = 0.0;
    yaw = pitch = 0.0f;
    health = 20.0f;
    food = 20.0f;
    saturation = 5.0f;
    oxygen = 300.0f;
    fallDistance = 0.0f;
    onGround = true;
    inWater = false;
    inLava = false;
    sprinting = false;
    sneaking = false;
    inventoryIds.fill(0);
    inventoryCounts.fill(0);
    selectedSlot = 0;
    hurtTime = 0;
    deathTime = 0;
    dead = false;
}

// -----------------------------------------------------------------------------
// WorldGenerator
// -----------------------------------------------------------------------------

WorldGenerator::WorldGenerator(uint64_t seed)
    : seed_(seed), rng_(seed) {}

void WorldGenerator::generate(WorldState& world, int32_t radiusChunks) {
    for (int32_t cx = -radiusChunks; cx <= radiusChunks; ++cx) {
        for (int32_t cz = -radiusChunks; cz <= radiusChunks; ++cz) {
            ChunkData* chunk = world.getChunk(cx, cz);
            generateChunk(*chunk);
        }
    }
}

float WorldGenerator::noise2D(float x, float z, int octaves) const {
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 0.01f;
    for (int i = 0; i < octaves; ++i) {
        value += amplitude * std::sin(x * frequency) * std::cos(z * frequency);
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    return value;
}

int WorldGenerator::terrainHeight(int worldX, int worldZ) const {
    float base = 64.0f + noise2D(static_cast<float>(worldX), static_cast<float>(worldZ), 3) * 8.0f;
    return static_cast<int>(base);
}

void WorldGenerator::generateChunk(ChunkData& chunk) {
    chunk.blocks.fill(BlockType::Air);
    int worldBaseX = chunk.chunkX * static_cast<int>(CHUNK_SIZE_XZ);
    int worldBaseZ = chunk.chunkZ * static_cast<int>(CHUNK_SIZE_XZ);

    for (int x = 0; x < static_cast<int>(CHUNK_SIZE_XZ); ++x) {
        for (int z = 0; z < static_cast<int>(CHUNK_SIZE_XZ); ++z) {
            int worldX = worldBaseX + x;
            int worldZ = worldBaseZ + z;
            int height = terrainHeight(worldX, worldZ);

            chunk.set(x, 0, z, BlockType::Bedrock);
            for (int y = 1; y < height - 3; ++y) {
                chunk.set(x, y, z, BlockType::Stone);
            }
            for (int y = std::max(1, height - 3); y < height; ++y) {
                chunk.set(x, y, z, BlockType::Dirt);
            }
            if (height > 0 && height < static_cast<int>(CHUNK_HEIGHT)) {
                chunk.set(x, height, z, BlockType::Grass);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// PhysicsEngine
// -----------------------------------------------------------------------------

void PhysicsEngine::applyMovement(PlayerState& player, const mc189_action_t& action,
                                  const WorldState& world, const mc189_config_t& config) {
    (void)world;
    player.yaw += static_cast<float>(action.camera_yaw) * 2.5f;
    player.pitch = std::clamp(player.pitch + static_cast<float>(action.camera_pitch) * 2.0f,
                              -90.0f, 90.0f);

    player.sneaking = (action.movement == 6);
    player.sprinting = (action.movement == 1 && !player.sneaking && player.food > 6.0f);

    float speed = config.player_speed;
    if (player.sprinting) speed *= config.sprint_multiplier;
    if (player.sneaking) speed *= 0.3f;
    if (player.inWater) speed *= 0.5f;

    float yawRad = static_cast<float>(player.yaw * (kPi / 180.0));
    float sinYaw = std::sin(yawRad);
    float cosYaw = std::cos(yawRad);

    double moveX = 0.0;
    double moveZ = 0.0;
    switch (action.movement) {
        case 1:
            moveX = -sinYaw;
            moveZ = cosYaw;
            break;
        case 2:
            moveX = sinYaw;
            moveZ = -cosYaw;
            break;
        case 3:
            moveX = cosYaw;
            moveZ = sinYaw;
            break;
        case 4:
            moveX = -cosYaw;
            moveZ = -sinYaw;
            break;
        default:
            break;
    }

    double accel = player.onGround ? 0.1 : 0.02;
    player.velX += moveX * speed * accel;
    player.velZ += moveZ * speed * accel;

    if (action.movement == 5 && player.onGround && !player.sneaking) {
        player.velY = JUMP_VELOCITY;
        player.onGround = false;
    }
}

void PhysicsEngine::applyGravity(PlayerState& player, const WorldState& world,
                                 const mc189_config_t& config) {
    (void)world;
    if (!player.onGround) {
        player.velY -= config.gravity;
    }
}

bool PhysicsEngine::checkCollision(const PlayerState& player, const WorldState& world) const {
    auto min = getAABBMin(player);
    auto max = getAABBMax(player);
    return aabbIntersectsSolid(min[0], min[1], min[2], max[0], max[1], max[2], world);
}

void PhysicsEngine::resolveCollisions(PlayerState& player, const WorldState& world) {
    double originalX = player.posX;
    double originalY = player.posY;
    double originalZ = player.posZ;

    player.posY += player.velY;
    if (checkCollision(player, world)) {
        player.posY = originalY;
        if (player.velY < 0) {
            player.onGround = true;
        }
        player.velY = 0.0;
    }

    player.posX += player.velX;
    if (checkCollision(player, world)) {
        player.posX = originalX;
        player.velX = 0.0;
    }

    player.posZ += player.velZ;
    if (checkCollision(player, world)) {
        player.posZ = originalZ;
        player.velZ = 0.0;
    }

    double drag = player.onGround ? DRAG_GROUND : (1.0 - DRAG_AIR);
    player.velX *= drag;
    player.velZ *= drag;
    if (!player.onGround) {
        player.velY *= (1.0 - DRAG_AIR);
    }
}

void PhysicsEngine::updateEnvironmentFlags(PlayerState& player, const WorldState& world) {
    int footY = static_cast<int>(std::floor(player.posY));
    int headY = static_cast<int>(std::floor(player.posY + PLAYER_EYE_HEIGHT));
    player.inWater = false;
    player.inLava = false;
    for (int y = footY; y <= headY; ++y) {
        BlockType block = world.getBlock(static_cast<int>(player.posX), y,
                                         static_cast<int>(player.posZ));
        if (isWater(block)) player.inWater = true;
        if (isLava(block)) player.inLava = true;
    }
}

void PhysicsEngine::applyDamage(PlayerState& player, float amount, uint8_t& deathReason) {
    player.health -= amount;
    if (player.health <= 0.0f) {
        player.dead = true;
        deathReason = 5;
    }
}

void PhysicsEngine::applyFallDamage(PlayerState& player, uint8_t& deathReason) {
    if (player.fallDistance > FALL_DAMAGE_THRESHOLD) {
        float damage = (player.fallDistance - FALL_DAMAGE_THRESHOLD) * FALL_DAMAGE_PER_BLOCK;
        applyDamage(player, damage, deathReason);
    }
    player.fallDistance = 0.0f;
}

void PhysicsEngine::updateHunger(PlayerState& player, bool moving, bool sprinting) {
    if (sprinting || moving) {
        player.saturation = std::max(0.0f, player.saturation - 0.01f);
        if (player.saturation == 0.0f) {
            player.food = std::max(0.0f, player.food - 0.005f);
        }
    }
}

std::array<double, 3> PhysicsEngine::getAABBMin(const PlayerState& player) const {
    return {player.posX - PLAYER_WIDTH / 2.0, player.posY,
            player.posZ - PLAYER_WIDTH / 2.0};
}

std::array<double, 3> PhysicsEngine::getAABBMax(const PlayerState& player) const {
    return {player.posX + PLAYER_WIDTH / 2.0, player.posY + PLAYER_HEIGHT,
            player.posZ + PLAYER_WIDTH / 2.0};
}

bool PhysicsEngine::aabbIntersectsSolid(double minX, double minY, double minZ,
                                        double maxX, double maxY, double maxZ,
                                        const WorldState& world) const {
    int ix0 = static_cast<int>(std::floor(minX));
    int iy0 = static_cast<int>(std::floor(minY));
    int iz0 = static_cast<int>(std::floor(minZ));
    int ix1 = static_cast<int>(std::floor(maxX));
    int iy1 = static_cast<int>(std::floor(maxY));
    int iz1 = static_cast<int>(std::floor(maxZ));

    for (int x = ix0; x <= ix1; ++x) {
        for (int y = iy0; y <= iy1; ++y) {
            for (int z = iz0; z <= iz1; ++z) {
                if (isSolid(world.getBlock(x, y, z))) {
                    return true;
                }
            }
        }
    }
    return false;
}

// -----------------------------------------------------------------------------
// RewardComputer
// -----------------------------------------------------------------------------

void RewardComputer::setCallback(mc189_reward_fn_t fn, void* userData) {
    callback_ = fn;
    userData_ = userData;
    useBuiltin_ = false;
}

void RewardComputer::setBuiltin(mc189_reward_type_t type) {
    builtinType_ = type;
    useBuiltin_ = true;
}

float RewardComputer::compute(const mc189_observation_t& prev, const mc189_observation_t& curr,
                              const mc189_action_t& action, bool done) {
    if (!useBuiltin_ && callback_) {
        return callback_(&prev, &curr, &action, done ? 1 : 0, userData_);
    }
    switch (builtinType_) {
        case MC189_REWARD_DIAMOND:
            return diamondReward(prev, curr, done);
        case MC189_REWARD_SURVIVAL:
        default:
            return survivalReward(prev, curr, done);
    }
}

void RewardComputer::reset() {
    prevDiamondCount_ = 0;
}

float RewardComputer::survivalReward(const mc189_observation_t& prev,
                                     const mc189_observation_t& curr,
                                     bool done) {
    (void)prev;
    float reward = 0.001f;
    if (done || curr.health <= 0.0f) {
        reward -= 1.0f;
    }
    return reward;
}

float RewardComputer::diamondReward(const mc189_observation_t& prev,
                                    const mc189_observation_t& curr,
                                    bool done) {
    float reward = survivalReward(prev, curr, done);
    int diamondCount = 0;
    for (size_t i = 0; i < 36; ++i) {
        if (curr.inventory_ids[i] == static_cast<uint16_t>(BlockType::DiamondOre) ||
            curr.inventory_ids[i] == static_cast<uint16_t>(BlockType::DiamondBlock)) {
            diamondCount += curr.inventory_counts[i];
        }
    }
    if (diamondCount > prevDiamondCount_) {
        reward += 5.0f;
    }
    prevDiamondCount_ = diamondCount;
    return reward;
}

// -----------------------------------------------------------------------------
// PerfTracker
// -----------------------------------------------------------------------------

void PerfTracker::recordStep(uint64_t stepNs, uint64_t physicsNs, uint64_t renderNs) {
    totalSteps.fetch_add(1, std::memory_order_relaxed);
    totalStepTimeNs.fetch_add(stepNs, std::memory_order_relaxed);
    totalPhysicsTimeNs.fetch_add(physicsNs, std::memory_order_relaxed);
    totalRenderTimeNs.fetch_add(renderNs, std::memory_order_relaxed);
}

void PerfTracker::reset() {
    totalSteps.store(0, std::memory_order_relaxed);
    totalStepTimeNs.store(0, std::memory_order_relaxed);
    totalPhysicsTimeNs.store(0, std::memory_order_relaxed);
    totalRenderTimeNs.store(0, std::memory_order_relaxed);
}

mc189_perf_stats_t PerfTracker::getStats() const {
    mc189_perf_stats_t stats{};
    uint64_t steps = totalSteps.load(std::memory_order_relaxed);
    uint64_t stepTime = totalStepTimeNs.load(std::memory_order_relaxed);
    uint64_t physicsTime = totalPhysicsTimeNs.load(std::memory_order_relaxed);
    uint64_t renderTime = totalRenderTimeNs.load(std::memory_order_relaxed);
    stats.total_steps = steps;
    if (steps > 0) {
        stats.avg_step_time_us = static_cast<double>(stepTime) / (steps * 1000.0);
        stats.avg_physics_time_us = static_cast<double>(physicsTime) / (steps * 1000.0);
        stats.avg_render_time_us = static_cast<double>(renderTime) / (steps * 1000.0);
        stats.steps_per_second = stepTime > 0 ? static_cast<double>(steps) * 1e9 / stepTime : 0.0;
    }
    return stats;
}

// -----------------------------------------------------------------------------
// SimulatorImpl
// -----------------------------------------------------------------------------

SimulatorImpl::SimulatorImpl(const mc189_config_t& config)
    : config_(config) {
    vulkan_ = std::make_unique<VulkanContext>();
    if (!vulkan_->init()) {
        vulkan_.reset();
    }
    initWorld();
}

SimulatorImpl::~SimulatorImpl() = default;

mc189_error_t SimulatorImpl::reset() {
    return resetWithSeed(config_.seed);
}

mc189_error_t SimulatorImpl::resetWithSeed(uint64_t seed) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.seed = seed;
    initWorld();
    return MC189_OK;
}

mc189_error_t SimulatorImpl::step(const mc189_action_t& action, mc189_step_result_t& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t startNs = nowNs();

    std::memcpy(prevPixelBuffer_.data(), pixelBuffer_.data(), pixelBuffer_.size());
    prevObs_ = obs_;
    prevObs_.pixels = prevPixelBuffer_.data();
    applyAction(action);
    tickWorld(config_.ticks_per_step);

    fillObservation(obs_);

    uint8_t deathReason = 0;
    bool done = checkDone(deathReason);
    episodeSteps_++;
    bool truncated = episodeSteps_ >= config_.max_episode_steps;

    float reward = reward_.compute(prevObs_, obs_, action, done);

    uint64_t endNs = nowNs();
    result.reward = reward;
    result.done = done ? 1 : 0;
    result.truncated = truncated ? 1 : 0;
    result.step_count = ++stepCount_;
    result.step_time_us = static_cast<float>(endNs - startNs) / kNsPerUs;
    result.death_reason = deathReason;

    result.observation = obs_;

    perf_.recordStep(endNs - startNs, endNs - startNs, 0);

    if (done || truncated) {
        episodeSteps_ = 0;
    }

    return MC189_OK;
}

mc189_error_t SimulatorImpl::getObservation(mc189_observation_t& obs) const {
    std::lock_guard<std::mutex> lock(mutex_);
    obs = obs_;
    return MC189_OK;
}

mc189_error_t SimulatorImpl::getBlock(int32_t x, int32_t y, int32_t z, uint8_t& type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    type = static_cast<uint8_t>(world_->getBlock(x, y, z));
    return MC189_OK;
}

mc189_error_t SimulatorImpl::setBlock(int32_t x, int32_t y, int32_t z, uint8_t type) {
    std::lock_guard<std::mutex> lock(mutex_);
    world_->setBlock(x, y, z, static_cast<BlockType>(type));
    return MC189_OK;
}

mc189_error_t SimulatorImpl::setRewardFunction(mc189_reward_fn_t fn, void* userData) {
    reward_.setCallback(fn, userData);
    return MC189_OK;
}

mc189_error_t SimulatorImpl::setBuiltinReward(mc189_reward_type_t type) {
    reward_.setBuiltin(type);
    return MC189_OK;
}

mc189_perf_stats_t SimulatorImpl::getPerfStats() const {
    return perf_.getStats();
}

void SimulatorImpl::resetPerfStats() {
    perf_.reset();
}

void SimulatorImpl::initWorld() {
    world_ = std::make_unique<WorldState>();
    world_->seed = config_.seed;
    worldGen_ = std::make_unique<WorldGenerator>(config_.seed);
    worldGen_->generate(*world_, static_cast<int32_t>(config_.render_distance));
    player_.reset(config_.spawn_x, config_.spawn_y, config_.spawn_z);
    stepCount_ = 0;
    episodeSteps_ = 0;
    obs_.width = config_.render_width;
    obs_.height = config_.render_height;
    obs_.channels = kChannels;
    pixelBuffer_.assign(static_cast<size_t>(obs_.width) * obs_.height * obs_.channels, 0);
    obs_.pixels = pixelBuffer_.data();
    prevPixelBuffer_.assign(pixelBuffer_.size(), 0);
    prevObs_ = obs_;
    prevObs_.pixels = prevPixelBuffer_.data();
    fillObservation(obs_);
    prevObs_ = obs_;
    prevObs_.pixels = prevPixelBuffer_.data();
    std::memcpy(prevPixelBuffer_.data(), pixelBuffer_.data(), pixelBuffer_.size());
    reward_.reset();
    initialized_ = true;
}

void SimulatorImpl::tickWorld(uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        world_->tick();
        if (config_.daylight_cycle == 0) {
            world_->worldTime = 0;
        }
        if (player_.inLava) {
            uint8_t deathReason = 2;
            physics_.applyDamage(player_, 4.0f, deathReason);
        }
        if (player_.inWater && !player_.inLava) {
            player_.oxygen = std::max(0.0f, player_.oxygen - 1.0f);
            if (player_.oxygen <= 0.0f) {
                uint8_t deathReason = 3;
                physics_.applyDamage(player_, 1.0f, deathReason);
            }
        } else {
            player_.oxygen = std::min(300.0f, player_.oxygen + 4.0f);
        }
        physics_.applyGravity(player_, *world_, config_);
        physics_.resolveCollisions(player_, *world_);
        physics_.updateEnvironmentFlags(player_, *world_);
    }
}

void SimulatorImpl::applyAction(const mc189_action_t& action) {
    physics_.applyMovement(player_, action, *world_, config_);
    physics_.updateHunger(player_, action.movement != 0, player_.sprinting);

    if (action.hotbar_slot < 9) {
        player_.selectedSlot = action.hotbar_slot;
    }
}

void SimulatorImpl::fillObservation(mc189_observation_t& obs) const {
    renderFrame(pixelBuffer_.data(), obs.width, obs.height);
    obs.pixels = pixelBuffer_.data();
    obs.health = player_.health;
    obs.food = player_.food;
    obs.saturation = player_.saturation;
    obs.oxygen = player_.oxygen;
    obs.pos_x = player_.posX;
    obs.pos_y = player_.posY;
    obs.pos_z = player_.posZ;
    obs.yaw = player_.yaw;
    obs.pitch = player_.pitch;
    obs.on_ground = player_.onGround ? 1 : 0;
    obs.in_water = player_.inWater ? 1 : 0;
    obs.in_lava = player_.inLava ? 1 : 0;
    std::copy(player_.inventoryIds.begin(), player_.inventoryIds.end(), obs.inventory_ids);
    std::copy(player_.inventoryCounts.begin(), player_.inventoryCounts.end(), obs.inventory_counts);
    obs.selected_slot = player_.selectedSlot;
    obs.nearby_hostile_count = 0;
    obs.nearby_passive_count = 0;
    obs.target_block_id = 0;
    obs.target_block_distance = 0.0f;
    obs.world_time = world_->worldTime;
    obs.day_time = world_->worldTime % 24000;
    obs.current_biome = 1;
    obs.light_level = 15;
}

void SimulatorImpl::renderFrame(uint8_t* pixels, uint32_t width, uint32_t height) const {
    if (!pixels) return;
    for (uint32_t y = 0; y < height; ++y) {
        bool isSky = y < height / 2;
        const float* color = isSky ? kSkyColor : kGroundColor;
        for (uint32_t x = 0; x < width; ++x) {
            size_t idx = (static_cast<size_t>(y) * width + x) * kChannels;
            pixels[idx] = static_cast<uint8_t>(color[0] * 255);
            pixels[idx + 1] = static_cast<uint8_t>(color[1] * 255);
            pixels[idx + 2] = static_cast<uint8_t>(color[2] * 255);
        }
    }
    if (player_.inLava) {
        for (size_t i = 0; i < static_cast<size_t>(width) * height * kChannels; i += kChannels) {
            pixels[i] = 255;
            pixels[i + 1] = 64;
            pixels[i + 2] = 0;
        }
    }
}

bool SimulatorImpl::checkDone(uint8_t& deathReason) const {
    if (player_.health <= 0.0f || player_.dead) {
        deathReason = player_.inLava ? 2 : 5;
        return true;
    }
    if (player_.posY < -64.0) {
        deathReason = 1;
        return true;
    }
    if (player_.oxygen <= 0.0f) {
        deathReason = 3;
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
// VectorizedEnvImpl
// -----------------------------------------------------------------------------

VectorizedEnvImpl::VectorizedEnvImpl(size_t numEnvs, const mc189_config_t& config)
    : config_(config) {
    envs_.reserve(numEnvs);
    for (size_t i = 0; i < numEnvs; ++i) {
        mc189_config_t cfg = config_;
        cfg.seed = config_.seed + static_cast<uint64_t>(i);
        envs_.push_back(std::make_unique<SimulatorImpl>(cfg));
    }
#ifdef __APPLE__
    stepQueue_ = dispatch_queue_create("mc189.vecstep", DISPATCH_QUEUE_CONCURRENT);
    stepGroup_ = dispatch_group_create();
#endif
}

VectorizedEnvImpl::~VectorizedEnvImpl() {
#ifdef __APPLE__
    if (stepGroup_) {
        dispatch_group_wait(stepGroup_, DISPATCH_TIME_FOREVER);
        dispatch_release(stepGroup_);
    }
    if (stepQueue_) dispatch_release(stepQueue_);
#endif
}

mc189_error_t VectorizedEnvImpl::reset() {
    for (auto& env : envs_) {
        env->reset();
    }
    return MC189_OK;
}

mc189_error_t VectorizedEnvImpl::resetWithSeeds(const uint64_t* seeds, size_t count) {
    size_t n = std::min(count, envs_.size());
    for (size_t i = 0; i < n; ++i) {
        envs_[i]->resetWithSeed(seeds[i]);
    }
    return MC189_OK;
}

mc189_error_t VectorizedEnvImpl::step(const mc189_action_t* actions,
                                      mc189_step_result_t* results) {
    size_t n = envs_.size();
#ifdef __APPLE__
    dispatch_apply(n, stepQueue_, ^(size_t i) {
        envs_[i]->step(actions[i], results[i]);
    });
#else
    for (size_t i = 0; i < n; ++i) {
        envs_[i]->step(actions[i], results[i]);
    }
#endif
    return MC189_OK;
}

mc189_error_t VectorizedEnvImpl::stepAsync(const mc189_action_t* actions) {
    if (stepping_.exchange(true)) return MC189_ERROR_INTERNAL;
    pendingActions_.assign(actions, actions + envs_.size());
    pendingResults_.assign(envs_.size(), mc189_step_result_t{});
#ifdef __APPLE__
    dispatch_group_async(stepGroup_, stepQueue_, ^{
        dispatch_apply(envs_.size(), stepQueue_, ^(size_t i) {
            envs_[i]->step(pendingActions_[i], pendingResults_[i]);
        });
        hasResults_.store(true);
    });
#else
    step(pendingActions_.data(), pendingResults_.data());
    hasResults_.store(true);
#endif
    return MC189_OK;
}

mc189_error_t VectorizedEnvImpl::waitStep(mc189_step_result_t* results) {
#ifdef __APPLE__
    dispatch_group_wait(stepGroup_, DISPATCH_TIME_FOREVER);
#endif
    if (!hasResults_.exchange(false)) {
        stepping_.store(false);
        return MC189_ERROR_INTERNAL;
    }
    std::copy(pendingResults_.begin(), pendingResults_.end(), results);
    stepping_.store(false);
    return MC189_OK;
}

}  // namespace mc189::internal

// -----------------------------------------------------------------------------
// C++ API wrapper (mc189::MinecraftSimulator)
// -----------------------------------------------------------------------------

namespace mc189 {

struct MinecraftSimulator::Impl {
    explicit Impl(const mc189_config_t& config) : impl(config) {}
    internal::SimulatorImpl impl;
};

MinecraftSimulator::MinecraftSimulator()
    : impl_(nullptr) {
    mc189_config_t cfg{};
    mc189_default_config(&cfg);
    impl_ = std::make_unique<Impl>(cfg);
}

MinecraftSimulator::MinecraftSimulator(const mc189_config_t& config)
    : impl_(std::make_unique<Impl>(config)) {}

MinecraftSimulator::~MinecraftSimulator() = default;

void MinecraftSimulator::reset(uint64_t seed) {
    impl_->impl.resetWithSeed(seed);
}

StepResult MinecraftSimulator::step(const AgentAction& action) {
    mc189_action_t raw{};
    raw.movement = action.movement;
    raw.camera_yaw = action.cameraYaw;
    raw.camera_pitch = action.cameraPitch;
    raw.interaction = action.interaction;
    raw.hotbar_slot = action.hotbarSlot;
    raw.craft_action = action.craftAction;

    mc189_step_result_t result{};
    impl_->impl.step(raw, result);

    StepResult cppResult;
    cppResult.reward = result.reward;
    cppResult.done = result.done != 0;
    cppResult.truncated = result.truncated != 0;
    cppResult.stepCount = result.step_count;
    cppResult.stepTimeUs = result.step_time_us;
    cppResult.deathReason = result.death_reason;

    Observation obs;
    obs.width = result.observation.width;
    obs.height = result.observation.height;
    obs.channels = result.observation.channels;
    size_t pixelCount = static_cast<size_t>(obs.width) * obs.height * obs.channels;
    obs.pixels.assign(result.observation.pixels, result.observation.pixels + pixelCount);
    obs.health = result.observation.health;
    obs.food = result.observation.food;
    obs.saturation = result.observation.saturation;
    obs.oxygen = result.observation.oxygen;
    obs.posX = result.observation.pos_x;
    obs.posY = result.observation.pos_y;
    obs.posZ = result.observation.pos_z;
    obs.yaw = result.observation.yaw;
    obs.pitch = result.observation.pitch;
    obs.onGround = result.observation.on_ground != 0;
    obs.inWater = result.observation.in_water != 0;
    obs.inLava = result.observation.in_lava != 0;
    std::copy(std::begin(result.observation.inventory_ids),
              std::end(result.observation.inventory_ids),
              obs.inventoryIds);
    std::copy(std::begin(result.observation.inventory_counts),
              std::end(result.observation.inventory_counts),
              obs.inventoryCounts);
    obs.selectedSlot = result.observation.selected_slot;
    obs.nearbyHostileCount = result.observation.nearby_hostile_count;
    obs.nearbyPassiveCount = result.observation.nearby_passive_count;
    obs.targetBlockId = result.observation.target_block_id;
    obs.targetBlockDistance = result.observation.target_block_distance;
    obs.worldTime = result.observation.world_time;
    obs.dayTime = result.observation.day_time;
    obs.currentBiome = result.observation.current_biome;
    obs.lightLevel = result.observation.light_level;

    cppResult.observation = std::move(obs);
    return cppResult;
}

BatchStepResult MinecraftSimulator::batchStep(const std::vector<AgentAction>& actions) {
    BatchStepResult result;
    result.results.reserve(actions.size());
    for (const auto& action : actions) {
        result.results.emplace_back(step(action));
    }
    return result;
}

Observation MinecraftSimulator::getObservation() {
    mc189_observation_t obs{};
    impl_->impl.getObservation(obs);
    Observation cppObs;
    cppObs.width = obs.width;
    cppObs.height = obs.height;
    cppObs.channels = obs.channels;
    size_t pixelCount = static_cast<size_t>(obs.width) * obs.height * obs.channels;
    cppObs.pixels.assign(obs.pixels, obs.pixels + pixelCount);
    cppObs.health = obs.health;
    cppObs.food = obs.food;
    cppObs.saturation = obs.saturation;
    cppObs.oxygen = obs.oxygen;
    cppObs.posX = obs.pos_x;
    cppObs.posY = obs.pos_y;
    cppObs.posZ = obs.pos_z;
    cppObs.yaw = obs.yaw;
    cppObs.pitch = obs.pitch;
    cppObs.onGround = obs.on_ground != 0;
    cppObs.inWater = obs.in_water != 0;
    cppObs.inLava = obs.in_lava != 0;
    std::copy(std::begin(obs.inventory_ids), std::end(obs.inventory_ids), cppObs.inventoryIds);
    std::copy(std::begin(obs.inventory_counts), std::end(obs.inventory_counts), cppObs.inventoryCounts);
    cppObs.selectedSlot = obs.selected_slot;
    cppObs.nearbyHostileCount = obs.nearby_hostile_count;
    cppObs.nearbyPassiveCount = obs.nearby_passive_count;
    cppObs.targetBlockId = obs.target_block_id;
    cppObs.targetBlockDistance = obs.target_block_distance;
    cppObs.worldTime = obs.world_time;
    cppObs.dayTime = obs.day_time;
    cppObs.currentBiome = obs.current_biome;
    cppObs.lightLevel = obs.light_level;
    return cppObs;
}

}  // namespace mc189

// -----------------------------------------------------------------------------
// C API
// -----------------------------------------------------------------------------

extern "C" {

MC189_API mc189_simulator_t mc189_create(void) {
    mc189_config_t config{};
    mc189_default_config(&config);
    return mc189_create_with_config(&config);
}

MC189_API mc189_simulator_t mc189_create_with_config(const mc189_config_t* config) {
    if (!config) return nullptr;
    auto sim = new mc189_simulator();
    sim->impl = std::make_unique<mc189::internal::SimulatorImpl>(*config);
    return sim;
}

MC189_API void mc189_destroy(mc189_simulator_t sim) {
    delete sim;
}

MC189_API void mc189_default_config(mc189_config_t* config) {
    if (!config) return;
    std::memset(config, 0, sizeof(mc189_config_t));
    config->seed = 42;
    config->spawn_x = 0;
    config->spawn_y = 64;
    config->spawn_z = 0;
    config->render_width = 84;
    config->render_height = 84;
    config->render_distance = 4;
    config->fov = 70.0f;
    config->ticks_per_step = 1;
    config->max_episode_steps = 10'000;
    config->daylight_cycle = 1;
    config->mob_spawning = 0;
    config->weather = 0;
    config->gravity = mc189::internal::GRAVITY;
    config->player_speed = mc189::internal::WALK_SPEED;
    config->sprint_multiplier = 1.3f;
    config->reward_death = -1.0f;
    config->reward_damage = -0.1f;
    config->reward_heal = 0.05f;
    config->reward_food = 0.01f;
    config->use_gpu_physics = 1;
    config->physics_batch_size = 256;
    config->verbose = 0;
    config->record_video = 0;
}

MC189_API mc189_error_t mc189_reset(mc189_simulator_t sim) {
    if (!sim) return MC189_ERROR_INVALID_HANDLE;
    return sim->impl->reset();
}

MC189_API mc189_error_t mc189_reset_with_seed(mc189_simulator_t sim, uint64_t seed) {
    if (!sim) return MC189_ERROR_INVALID_HANDLE;
    return sim->impl->resetWithSeed(seed);
}

MC189_API mc189_error_t mc189_step(mc189_simulator_t sim,
                                   const mc189_action_t* action,
                                   mc189_step_result_t* result) {
    if (!sim || !action || !result) return MC189_ERROR_INVALID_ARGUMENT;
    uint8_t* pixelTarget = result->observation.pixels;
    mc189_error_t err = sim->impl->step(*action, *result);
    if (pixelTarget && result->observation.pixels) {
        size_t count = static_cast<size_t>(result->observation.width) *
                       result->observation.height * result->observation.channels;
        std::memcpy(pixelTarget, result->observation.pixels, count);
        result->observation.pixels = pixelTarget;
    }
    return err;
}

MC189_API mc189_error_t mc189_get_observation(mc189_simulator_t sim,
                                              mc189_observation_t* obs) {
    if (!sim || !obs) return MC189_ERROR_INVALID_ARGUMENT;
    uint8_t* pixelTarget = obs->pixels;
    mc189_observation_t tmp{};
    sim->impl->getObservation(tmp);
    *obs = tmp;
    if (pixelTarget && tmp.pixels) {
        size_t count = static_cast<size_t>(tmp.width) * tmp.height * tmp.channels;
        std::memcpy(pixelTarget, tmp.pixels, count);
        obs->pixels = pixelTarget;
    }
    return MC189_OK;
}

MC189_API mc189_error_t mc189_get_block(mc189_simulator_t sim,
                                        int32_t x, int32_t y, int32_t z,
                                        uint8_t* block_type) {
    if (!sim || !block_type) return MC189_ERROR_INVALID_ARGUMENT;
    return sim->impl->getBlock(x, y, z, *block_type);
}

MC189_API mc189_error_t mc189_set_block(mc189_simulator_t sim,
                                        int32_t x, int32_t y, int32_t z,
                                        uint8_t block_type) {
    if (!sim) return MC189_ERROR_INVALID_HANDLE;
    return sim->impl->setBlock(x, y, z, block_type);
}

MC189_API mc189_error_t mc189_get_perf_stats(mc189_simulator_t sim,
                                             mc189_perf_stats_t* stats) {
    if (!sim || !stats) return MC189_ERROR_INVALID_ARGUMENT;
    *stats = sim->impl->getPerfStats();
    return MC189_OK;
}

MC189_API void mc189_reset_perf_stats(mc189_simulator_t sim) {
    if (!sim) return;
    sim->impl->resetPerfStats();
}

MC189_API mc189_vectorized_env_t mc189_vec_create(size_t num_envs,
                                                  const mc189_config_t* config) {
    if (!config) return nullptr;
    auto env = new mc189_vectorized_env();
    env->impl = std::make_unique<mc189::internal::VectorizedEnvImpl>(num_envs, *config);
    return env;
}

MC189_API void mc189_vec_destroy(mc189_vectorized_env_t env) {
    delete env;
}

MC189_API mc189_error_t mc189_vec_reset(mc189_vectorized_env_t env) {
    if (!env) return MC189_ERROR_INVALID_HANDLE;
    return env->impl->reset();
}

MC189_API mc189_error_t mc189_vec_reset_with_seeds(mc189_vectorized_env_t env,
                                                   const uint64_t* seeds,
                                                   size_t num_seeds) {
    if (!env || !seeds) return MC189_ERROR_INVALID_ARGUMENT;
    return env->impl->resetWithSeeds(seeds, num_seeds);
}

MC189_API mc189_error_t mc189_vec_step(mc189_vectorized_env_t env,
                                       const mc189_action_t* actions,
                                       mc189_step_result_t* results) {
    if (!env || !actions || !results) return MC189_ERROR_INVALID_ARGUMENT;
    size_t count = env->impl->numEnvs();
    std::vector<uint8_t*> pixelTargets(count, nullptr);
    for (size_t i = 0; i < count; ++i) {
        pixelTargets[i] = results[i].observation.pixels;
    }
    mc189_error_t err = env->impl->step(actions, results);
    for (size_t i = 0; i < count; ++i) {
        if (pixelTargets[i] && results[i].observation.pixels) {
            size_t byteCount = static_cast<size_t>(results[i].observation.width) *
                               results[i].observation.height *
                               results[i].observation.channels;
            std::memcpy(pixelTargets[i], results[i].observation.pixels, byteCount);
            results[i].observation.pixels = pixelTargets[i];
        }
    }
    return err;
}

MC189_API mc189_error_t mc189_vec_step_async(mc189_vectorized_env_t env,
                                             const mc189_action_t* actions) {
    if (!env || !actions) return MC189_ERROR_INVALID_ARGUMENT;
    return env->impl->stepAsync(actions);
}

MC189_API mc189_error_t mc189_vec_wait_step(mc189_vectorized_env_t env,
                                            mc189_step_result_t* results) {
    if (!env || !results) return MC189_ERROR_INVALID_ARGUMENT;
    size_t count = env->impl->numEnvs();
    std::vector<uint8_t*> pixelTargets(count, nullptr);
    for (size_t i = 0; i < count; ++i) {
        pixelTargets[i] = results[i].observation.pixels;
    }
    mc189_error_t err = env->impl->waitStep(results);
    for (size_t i = 0; i < count; ++i) {
        if (pixelTargets[i] && results[i].observation.pixels) {
            size_t byteCount = static_cast<size_t>(results[i].observation.width) *
                               results[i].observation.height *
                               results[i].observation.channels;
            std::memcpy(pixelTargets[i], results[i].observation.pixels, byteCount);
            results[i].observation.pixels = pixelTargets[i];
        }
    }
    return err;
}

MC189_API size_t mc189_vec_num_envs(mc189_vectorized_env_t env) {
    if (!env) return 0;
    return env->impl->numEnvs();
}

MC189_API const char* mc189_get_error_message(mc189_error_t error) {
    switch (error) {
        case MC189_OK:
            return "Success";
        case MC189_ERROR_INVALID_HANDLE:
            return "Invalid handle";
        case MC189_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case MC189_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case MC189_ERROR_VULKAN_INIT:
            return "Vulkan init failed";
        case MC189_ERROR_VULKAN_DEVICE:
            return "Vulkan device creation failed";
        case MC189_ERROR_SHADER_COMPILE:
            return "Shader compile failed";
        case MC189_ERROR_WORLD_GEN:
            return "World generation failed";
        default:
            return "Internal error";
    }
}

MC189_API const char* mc189_version(void) {
    return "0.1.0";
}

MC189_API int mc189_vulkan_available(void) {
    mc189::internal::VulkanContext ctx;
    bool ok = ctx.init();
    ctx.shutdown();
    return ok ? 1 : 0;
}

MC189_API const char* mc189_get_gpu_name(void) {
    static mc189::internal::VulkanContext ctx;
    static bool initialized = false;
    if (!initialized) {
        initialized = ctx.init();
    }
    return initialized ? ctx.gpuName : "";
}

MC189_API uint64_t mc189_get_gpu_memory(void) {
    static mc189::internal::VulkanContext ctx;
    static bool initialized = false;
    if (!initialized) {
        initialized = ctx.init();
    }
    return initialized ? ctx.gpuMemory : 0;
}

MC189_API uint8_t* mc189_alloc_pixels(uint32_t width, uint32_t height, uint32_t channels) {
    size_t count = static_cast<size_t>(width) * height * channels;
    return static_cast<uint8_t*>(std::calloc(count, sizeof(uint8_t)));
}

MC189_API void mc189_free_pixels(uint8_t* pixels) {
    std::free(pixels);
}

MC189_API mc189_observation_t* mc189_alloc_observations(size_t count,
                                                        uint32_t width,
                                                        uint32_t height) {
    auto* obs = static_cast<mc189_observation_t*>(std::calloc(count, sizeof(mc189_observation_t)));
    if (!obs) return nullptr;
    for (size_t i = 0; i < count; ++i) {
        obs[i].width = width;
        obs[i].height = height;
        obs[i].channels = kChannels;
        obs[i].pixels = mc189_alloc_pixels(width, height, kChannels);
    }
    return obs;
}

MC189_API void mc189_free_observations(mc189_observation_t* obs, size_t count) {
    if (!obs) return;
    for (size_t i = 0; i < count; ++i) {
        mc189_free_pixels(obs[i].pixels);
    }
    std::free(obs);
}

MC189_API mc189_error_t mc189_set_reward_function(mc189_simulator_t sim,
                                                  mc189_reward_fn_t fn,
                                                  void* user_data) {
    if (!sim) return MC189_ERROR_INVALID_HANDLE;
    return sim->impl->setRewardFunction(fn, user_data);
}

MC189_API mc189_error_t mc189_set_builtin_reward(mc189_simulator_t sim,
                                                 mc189_reward_type_t type) {
    if (!sim) return MC189_ERROR_INVALID_HANDLE;
    return sim->impl->setBuiltinReward(type);
}

}  // extern "C"
