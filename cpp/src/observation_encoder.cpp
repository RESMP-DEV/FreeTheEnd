#include "mc189/observation_encoder.h"
#include "mc189/compute_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mc189 {

namespace {

// Normalization constants for 0-1 mapping
constexpr float MAX_POS = 30000.0f;        // World border
constexpr float MAX_VEL = 3.92f;           // Terminal velocity in MC 1.8.9
constexpr float MAX_HEALTH = 20.0f;
constexpr float MAX_HUNGER = 20.0f;
constexpr float MAX_SATURATION = 20.0f;
constexpr float MAX_ARMOR = 20.0f;
constexpr float MAX_XP_LEVEL = 50.0f;
constexpr float MAX_DISTANCE = 1000.0f;    // Cap entity/nav distances
constexpr float MAX_ENTITY_COUNT = 64.0f;
constexpr float MAX_STACK = 64.0f;
constexpr float MAX_ARROWS = 64.0f;
constexpr float MAX_CRYSTALS = 10.0f;
constexpr float MAX_DEATHS = 20.0f;
constexpr float MAX_TIME_MINUTES = 60.0f;  // 1 hour cap
constexpr float MAX_DRAGON_DAMAGE = 200.0f;
constexpr float MAX_HEIGHT = 256.0f;
constexpr float MAX_LIGHT = 15.0f;
constexpr float MAX_STAGES = 8.0f;         // Total speedrun stages
constexpr float TICKS_TO_MINUTES = 1.0f / (20.0f * 60.0f);

// Tool tier encoding: wood=0.2, stone=0.4, iron=0.6, gold=0.5, diamond=1.0
float tool_tier_normalized(ToolMaterial mat) {
    switch (mat) {
        case ToolMaterial::WOOD:    return 0.2f;
        case ToolMaterial::STONE:   return 0.4f;
        case ToolMaterial::IRON:    return 0.6f;
        case ToolMaterial::GOLD:    return 0.5f;
        case ToolMaterial::DIAMOND: return 1.0f;
        default:                    return 0.0f;
    }
}

float clamp01(float x) {
    return std::fmin(std::fmax(x, 0.0f), 1.0f);
}

float normalize(float val, float max_val) {
    return clamp01(val / max_val);
}

// Normalize position: map [-MAX_POS, MAX_POS] to [0, 1]
float normalize_pos(float val) {
    return clamp01((val + MAX_POS) / (2.0f * MAX_POS));
}

// Normalize velocity: map [-MAX_VEL, MAX_VEL] to [0, 1]
float normalize_vel(float val) {
    return clamp01((val + MAX_VEL) / (2.0f * MAX_VEL));
}

// Normalize angle: map [-180, 180] or [0, 360] to [0, 1]
float normalize_angle(float degrees) {
    float wrapped = std::fmod(degrees, 360.0f);
    if (wrapped < 0.0f) wrapped += 360.0f;
    return wrapped / 360.0f;
}

// Normalize pitch: map [-90, 90] to [0, 1]
float normalize_pitch(float degrees) {
    return clamp01((degrees + 90.0f) / 180.0f);
}

// Direction vector component: already in [-1, 1], map to [0, 1]
float normalize_dir(float d) {
    return clamp01((d + 1.0f) * 0.5f);
}

float distance_xz(float x1, float z1, float x2, float z2) {
    float dx = x2 - x1;
    float dz = z2 - z1;
    return std::sqrt(dx * dx + dz * dz);
}

void direction_xz(float from_x, float from_z, float to_x, float to_z,
                  float& dir_x, float& dir_z) {
    float dx = to_x - from_x;
    float dz = to_z - from_z;
    float len = std::sqrt(dx * dx + dz * dz);
    if (len > 0.001f) {
        dir_x = dx / len;
        dir_z = dz / len;
    } else {
        dir_x = 0.0f;
        dir_z = 0.0f;
    }
}

// Count items of a given ID in inventory
uint32_t count_item(const PlayerFull& player, ItemID id) {
    uint32_t total = 0;
    uint16_t raw_id = static_cast<uint16_t>(id);
    for (int i = 0; i < 36; ++i) {
        if (player.inventory[i].item_id == raw_id) {
            total += player.inventory[i].count;
        }
    }
    return total;
}

// Find best tool of a given type in inventory
ToolMaterial find_best_tool(const PlayerFull& player, ToolType type) {
    ToolMaterial best = ToolMaterial::NONE;
    for (int i = 0; i < 36; ++i) {
        if (player.inventory[i].item_id == 0) continue;
        auto id = static_cast<ItemID>(player.inventory[i].item_id);
        const auto& props = get_item_properties(id);
        if (props.tool_type == type && props.material > best) {
            best = props.material;
        }
    }
    return best;
}

// Check if player has item
bool has_item(const PlayerFull& player, ItemID id) {
    return count_item(player, id) > 0;
}

// Count food items in inventory
uint32_t count_food(const PlayerFull& player) {
    uint32_t total = 0;
    for (int i = 0; i < 36; ++i) {
        if (player.inventory[i].item_id == 0) continue;
        auto id = static_cast<ItemID>(player.inventory[i].item_id);
        if (is_food(id)) {
            total += player.inventory[i].count;
        }
    }
    return total;
}

// Compute armor points from equipped armor
float compute_armor_points(const PlayerFull& player) {
    float total = 0.0f;
    for (int i = 0; i < 4; ++i) {
        if (player.armor[i].item_id == 0) continue;
        auto id = static_cast<ItemID>(player.armor[i].item_id);
        const auto& props = get_item_properties(id);
        total += props.armor_points;
    }
    return total;
}

// Determine bucket type: 0=none/empty, 1=water, 2=lava
float get_bucket_type(const PlayerFull& player) {
    // Check for water bucket first (more useful)
    if (has_item(player, ItemID::WATER_BUCKET)) return 1.0f;
    if (has_item(player, ItemID::LAVA_BUCKET)) return 2.0f;
    if (has_item(player, ItemID::BUCKET)) return 0.0f;
    return 0.0f;
}

} // namespace

ObservationEncoder::ObservationEncoder(uint32_t num_envs)
    : num_envs_(num_envs) {}

void ObservationEncoder::encode(uint32_t /*env_id*/,
                                const PlayerFull& player,
                                const WorldState& world,
                                const std::vector<Entity>& entities,
                                const Dragon* dragon,
                                const Crystal* crystals,
                                ExtendedObservation& out) const {
    std::memset(&out, 0, sizeof(ExtendedObservation));

    encode_player(player, out);
    encode_inventory(player, out);
    encode_entities(entities, player.base.position[0], player.base.position[2], out);
    encode_navigation(world, player.base.position[0], player.base.position[2], out);
    encode_progress(world, out);
    encode_dragon(dragon, crystals, player.base.position[0], player.base.position[2], out);
}

void ObservationEncoder::encode_to_buffer(uint32_t env_id,
                                          const PlayerFull& player,
                                          const WorldState& world,
                                          const std::vector<Entity>& entities,
                                          const Dragon* dragon,
                                          const Crystal* crystals,
                                          float* out) const {
    ExtendedObservation obs;
    encode(env_id, player, world, entities, dragon, crystals, obs);
    std::memcpy(out, &obs, sizeof(ExtendedObservation));
}

void ObservationEncoder::dispatch_encode(VulkanContext& ctx,
                                         Buffer& player_buffer,
                                         Buffer& world_buffer,
                                         Buffer& entity_buffer,
                                         Buffer& dragon_buffer,
                                         Buffer& crystal_buffer,
                                         Buffer& output_buffer,
                                         uint32_t num_envs) {
    // Load observation encoder compute shader
    auto spirv = ComputePipeline::load_spirv("shaders/observation_encode.spv");
    if (spirv.empty()) return;

    ComputePipeline::Config cfg;
    cfg.spirv_code = std::move(spirv);
    cfg.entry_point = "main";
    cfg.local_size_x = 256;
    cfg.bindings = {
        {0, vk::DescriptorType::eStorageBuffer},  // players
        {1, vk::DescriptorType::eStorageBuffer},  // world state
        {2, vk::DescriptorType::eStorageBuffer},  // entities
        {3, vk::DescriptorType::eStorageBuffer},  // dragon
        {4, vk::DescriptorType::eStorageBuffer},  // crystals
        {5, vk::DescriptorType::eStorageBuffer},  // output observations
    };
    cfg.push_constants = {{0, sizeof(uint32_t)}};

    ComputePipeline pipeline(ctx, cfg);
    auto desc_set = pipeline.allocate_descriptor_set();

    pipeline.update_descriptors(desc_set, {
        {0, player_buffer.handle(),  0, VK_WHOLE_SIZE},
        {1, world_buffer.handle(),   0, VK_WHOLE_SIZE},
        {2, entity_buffer.handle(),  0, VK_WHOLE_SIZE},
        {3, dragon_buffer.handle(),  0, VK_WHOLE_SIZE},
        {4, crystal_buffer.handle(), 0, VK_WHOLE_SIZE},
        {5, output_buffer.handle(),  0, VK_WHOLE_SIZE},
    });

    auto cmd = ctx.allocate_command_buffer();
    cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    pipeline.bind(cmd);
    pipeline.bind_descriptor_set(cmd, desc_set);
    pipeline.push_constants(cmd, num_envs);
    pipeline.dispatch_for_count(cmd, num_envs);
    cmd.end();

    ctx.submit_and_wait(cmd);
    ctx.free_command_buffer(cmd);
    pipeline.free_descriptor_set(desc_set);
}

void ObservationEncoder::encode_player(const PlayerFull& player,
                                       ExtendedObservation& out) const {
    const auto& base = player.base;

    out.pos_x = normalize_pos(base.position[0]);
    out.pos_y = normalize(base.position[1], MAX_HEIGHT);
    out.pos_z = normalize_pos(base.position[2]);

    out.vel_x = normalize_vel(base.velocity[0]);
    out.vel_y = normalize_vel(base.velocity[1]);
    out.vel_z = normalize_vel(base.velocity[2]);

    out.yaw = normalize_angle(base.yaw);
    out.pitch = normalize_pitch(base.pitch);

    out.health = normalize(base.health, MAX_HEALTH);
    out.max_health = 1.0f;  // Always 20 in vanilla

    out.hunger = normalize(base.hunger, MAX_HUNGER);
    out.saturation = normalize(base.saturation, MAX_SATURATION);

    out.armor_points = normalize(compute_armor_points(player), MAX_ARMOR);

    // Flags: bit0=on_ground from Player::flags
    out.on_ground = (base.flags & 0x1) ? 1.0f : 0.0f;
    out.in_water = (base.flags & 0x2) ? 1.0f : 0.0f;
    out.in_lava = (base.flags & 0x4) ? 1.0f : 0.0f;

    // Dimension: normalize -1/0/1 to 0/0.5/1
    out.dimension = 0.5f;  // Default overworld
    // WorldState has dimension info; we'll get it from the WorldState in encode()
    // For now, use flags if available

    out.light_level = 0.5f;  // Placeholder; would need chunk data access

    out.xp_level = normalize(static_cast<float>(player.xp_level), MAX_XP_LEVEL);
    out.xp_progress = normalize(static_cast<float>(player.xp_progress), 1000.0f);
}

void ObservationEncoder::encode_inventory(const PlayerFull& player,
                                          ExtendedObservation& out) const {
    // Held item
    uint32_t slot = player.selected_slot;
    if (slot < 36 && player.inventory[slot].item_id != 0) {
        auto id = static_cast<ItemID>(player.inventory[slot].item_id);
        const auto& props = get_item_properties(id);
        out.held_item_type = static_cast<float>(player.inventory[slot].item_id) /
                             static_cast<float>(ItemID::MAX_ITEM_ID);
        if (props.max_durability > 0) {
            out.held_item_durability = 1.0f - (static_cast<float>(player.inventory[slot].damage) /
                                               static_cast<float>(props.max_durability));
        } else {
            out.held_item_durability = 1.0f;
        }
    }

    // Best pickaxe
    ToolMaterial best_pick = find_best_tool(player, ToolType::PICKAXE);
    out.has_pickaxe = (best_pick != ToolMaterial::NONE) ? 1.0f : 0.0f;
    out.pickaxe_tier = tool_tier_normalized(best_pick);

    // Best sword
    ToolMaterial best_sword = find_best_tool(player, ToolType::SWORD);
    out.has_sword = (best_sword != ToolMaterial::NONE) ? 1.0f : 0.0f;
    out.sword_tier = tool_tier_normalized(best_sword);

    // Bow and arrows
    out.has_bow = has_item(player, ItemID::BOW) ? 1.0f : 0.0f;
    out.arrow_count = normalize(static_cast<float>(count_item(player, ItemID::ARROW)), MAX_ARROWS);

    // Key item counts
    out.iron_count = normalize(static_cast<float>(count_item(player, ItemID::IRON_INGOT)), MAX_STACK);
    out.diamond_count = normalize(static_cast<float>(count_item(player, ItemID::DIAMOND)), MAX_STACK);
    out.blaze_rod_count = normalize(static_cast<float>(count_item(player, ItemID::BLAZE_ROD)), MAX_STACK);
    out.ender_pearl_count = normalize(static_cast<float>(count_item(player, ItemID::ENDER_PEARL)), 16.0f);
    out.eye_of_ender_count = normalize(static_cast<float>(count_item(player, ItemID::EYE_OF_ENDER)), 12.0f);
    out.obsidian_count = normalize(static_cast<float>(count_item(player, ItemID::OBSIDIAN_ITEM)), MAX_STACK);

    // Food
    out.food_count = normalize(static_cast<float>(count_food(player)), MAX_STACK);

    // Beds (block item ID 26 in MC 1.8.9)
    uint32_t bed_count = 0;
    for (int i = 0; i < 36; ++i) {
        if (player.inventory[i].item_id == 355) {  // Bed item ID
            bed_count += player.inventory[i].count;
        }
    }
    out.bed_count = normalize(static_cast<float>(bed_count), 8.0f);

    // Bucket type
    out.bucket_type = get_bucket_type(player) / 2.0f;  // 0, 0.5, 1.0

    // Flint and steel
    out.flint_steel = has_item(player, ItemID::FLINT_AND_STEEL) ? 1.0f : 0.0f;
}

void ObservationEncoder::encode_entities(const std::vector<Entity>& entities,
                                         float px, float pz,
                                         ExtendedObservation& out) const {
    // Sort entities by distance, take nearest 8
    struct EntityDist {
        size_t index;
        float dist;
    };

    std::vector<EntityDist> sorted;
    sorted.reserve(entities.size());

    for (size_t i = 0; i < entities.size(); ++i) {
        const auto& e = entities[i];
        if (e.type == 0) continue;  // Skip empty slots
        float d = distance_xz(px, pz, e.position[0], e.position[2]);
        sorted.push_back({i, d});
    }

    std::sort(sorted.begin(), sorted.end(),
              [](const EntityDist& a, const EntityDist& b) { return a.dist < b.dist; });

    size_t count = std::min(sorted.size(), size_t(8));
    for (size_t i = 0; i < count; ++i) {
        const auto& e = entities[sorted[i].index];
        size_t base = i * 4;

        // Entity type normalized to [0, 1]
        out.entity_data[base + 0] = static_cast<float>(e.type) /
                                    static_cast<float>(EntityType::MAX_ENTITY_TYPE);

        // Distance normalized
        out.entity_data[base + 1] = normalize(sorted[i].dist, MAX_DISTANCE);

        // Relative direction (normalized to [0, 1] from [-1, 1])
        float dir_x, dir_z;
        direction_xz(px, pz, e.position[0], e.position[2], dir_x, dir_z);
        out.entity_data[base + 2] = normalize_dir(dir_x);
        out.entity_data[base + 3] = normalize_dir(dir_z);
    }

    // Remaining slots stay zero from memset
}

void ObservationEncoder::encode_navigation(const WorldState& world,
                                           float px, float pz,
                                           ExtendedObservation& out) const {
    // Dimension for player state (set here since WorldState has dimension)
    // Map: NETHER(-1) -> 0, OVERWORLD(0) -> 0.5, END(1) -> 1.0
    out.dimension = (static_cast<float>(world.dimension) + 1.0f) / 2.0f;

    // Stronghold navigation
    float sh_dist = distance_xz(px, pz, world.stronghold_x, world.stronghold_z);
    out.stronghold_found = (world.stronghold_x != 0.0f || world.stronghold_z != 0.0f) ? 1.0f : 0.0f;
    out.stronghold_dist = normalize(sh_dist, MAX_DISTANCE);

    float dir_x, dir_z;
    direction_xz(px, pz, world.stronghold_x, world.stronghold_z, dir_x, dir_z);
    out.stronghold_dir_x = normalize_dir(dir_x);
    out.stronghold_dir_z = normalize_dir(dir_z);

    // Portal frame eyes: extract from stage_progress bitmask
    // Bits 0-11 could represent individual eyes placed
    uint32_t eyes = 0;
    uint32_t eye_bits = world.stage_progress & 0xFFF;
    while (eye_bits) {
        eyes += eye_bits & 1;
        eye_bits >>= 1;
    }
    out.portal_frame_eyes = normalize(static_cast<float>(eyes), 12.0f);

    // Nether portal / fortress: these depend on dimension-specific state
    // For now, initialize from world state flags
    out.nether_portal_dist = 0.0f;
    out.nether_portal_dir_x = 0.5f;
    out.dir_z = 0.5f;

    out.fortress_found = 0.0f;
    out.fortress_dir_x = 0.5f;
    out.fortress_dir_z = 0.5f;
    out.fortress_dist = 0.0f;
}

void ObservationEncoder::encode_progress(const WorldState& world,
                                         ExtendedObservation& out) const {
    out.stage = normalize(static_cast<float>(world.stage), MAX_STAGES);

    // Stage progress: count set bits as fraction of expected milestones
    uint32_t progress_bits = world.stage_progress;
    uint32_t set_bits = 0;
    while (progress_bits) {
        set_bits += progress_bits & 1;
        progress_bits >>= 1;
    }
    out.stage_progress = normalize(static_cast<float>(set_bits), 32.0f);

    out.deaths = normalize(static_cast<float>(world.deaths), MAX_DEATHS);
    out.time_elapsed = normalize(static_cast<float>(world.tick) * TICKS_TO_MINUTES, MAX_TIME_MINUTES);

    // These would need additional state tracking; initialize from what's available
    out.crystals_destroyed = 0.0f;
    out.dragon_damage_dealt = 0.0f;
    out.beds_placed = 0.0f;
    out.portals_lit = 0.0f;
}

void ObservationEncoder::encode_dragon(const Dragon* dragon,
                                       const Crystal* crystals,
                                       float px, float pz,
                                       ExtendedObservation& out) const {
    if (!dragon) {
        // Not in End, all dragon fields stay zero from memset
        return;
    }

    out.dragon_health = normalize(dragon->health, DRAGON_MAX_HEALTH);
    out.dragon_x = normalize_pos(dragon->position[0]);
    out.dragon_y = normalize(dragon->position[1], MAX_HEIGHT);
    out.dragon_z = normalize_pos(dragon->position[2]);

    float d_dist = distance_xz(px, pz, dragon->position[0], dragon->position[2]);
    out.dragon_dist = normalize(d_dist, MAX_DISTANCE);

    float dir_x, dir_z;
    direction_xz(px, pz, dragon->position[0], dragon->position[2], dir_x, dir_z);
    out.dragon_dir_x = normalize_dir(dir_x);
    out.dragon_dir_z = normalize_dir(dir_z);

    out.dragon_phase = normalize(static_cast<float>(dragon->phase), 10.0f);

    // Perching: perch_timer > 0 means perching
    out.dragon_perching = (dragon->perch_timer > 0) ? 1.0f : 0.0f;

    // Charging: check flags for attack state
    out.dragon_charging = (dragon->flags & 0x1) ? 1.0f : 0.0f;

    // Can hit: close enough and perching or low enough
    bool reachable = (d_dist < 5.0f) ||
                     (dragon->position[1] < END_SPAWN_Y + 10.0f && d_dist < 8.0f);
    out.can_hit_dragon = reachable ? 1.0f : 0.0f;

    // Crystal state
    if (crystals) {
        uint32_t alive = 0;
        float nearest_dist = MAX_DISTANCE;
        float nearest_x = 0.0f, nearest_z = 0.0f;

        for (uint32_t i = 0; i < NUM_CRYSTALS; ++i) {
            if (crystals[i].is_alive > 0.5f) {
                ++alive;
                float cd = distance_xz(px, pz,
                                       crystals[i].position[0],
                                       crystals[i].position[2]);
                if (cd < nearest_dist) {
                    nearest_dist = cd;
                    nearest_x = crystals[i].position[0];
                    nearest_z = crystals[i].position[2];
                }
            }
        }

        out.crystals_remaining = normalize(static_cast<float>(alive), MAX_CRYSTALS);
        out.nearest_crystal_dist = normalize(nearest_dist, MAX_DISTANCE);

        if (alive > 0) {
            float cx, cz;
            direction_xz(px, pz, nearest_x, nearest_z, cx, cz);
            out.nearest_crystal_dir_x = normalize_dir(cx);
            out.nearest_crystal_dir_z = normalize_dir(cz);
        } else {
            out.nearest_crystal_dir_x = 0.5f;
            out.nearest_crystal_dir_z = 0.5f;
        }
    }
}

} // namespace mc189
