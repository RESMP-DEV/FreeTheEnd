#pragma once
#include "mc189/simulator.h"
#include "mc189/dimension.h"
#include "mc189/items.h"
#include <cstdint>
#include <vector>

namespace mc189 {

// 256-float observation layout
// [0-31]: Player state
// [32-63]: Inventory summary
// [64-95]: Nearby entities
// [96-127]: Terrain/blocks
// [128-159]: Navigation (stronghold, portal, etc.)
// [160-191]: Progress/objectives
// [192-223]: Dragon state (if in End)
// [224-255]: Reserved/misc

constexpr size_t EXTENDED_OBS_SIZE = 256;

struct ExtendedObservation {
    // Player (32 floats)
    float pos_x, pos_y, pos_z;           // 0-2
    float vel_x, vel_y, vel_z;           // 3-5
    float yaw, pitch;                     // 6-7
    float health, max_health;             // 8-9
    float hunger, saturation;             // 10-11
    float armor_points;                   // 12
    float on_ground, in_water, in_lava;   // 13-15
    float dimension;                      // 16: -1/0/1
    float light_level;                    // 17
    float xp_level, xp_progress;          // 18-19
    float player_reserved[12];            // 20-31

    // Inventory (32 floats)
    float held_item_type;                 // 32
    float held_item_durability;           // 33
    float has_pickaxe, pickaxe_tier;      // 34-35
    float has_sword, sword_tier;          // 36-37
    float has_bow, arrow_count;           // 38-39
    float iron_count, diamond_count;      // 40-41
    float blaze_rod_count, ender_pearl_count; // 42-43
    float eye_of_ender_count;             // 44
    float obsidian_count;                 // 45
    float food_count;                     // 46
    float bed_count;                      // 47
    float bucket_type;                    // 48: 0=empty, 1=water, 2=lava
    float flint_steel;                    // 49
    float inv_reserved[14];               // 50-63

    // Entities (32 floats) - 8 nearest entities x 4 floats
    float entity_data[32];                // 64-95: [type, dist, rel_x, rel_z] x 8

    // Terrain (32 floats)
    float ground_height[9];               // 96-104: 3x3 grid around player
    float block_ahead[3];                 // 105-107: block type at eye level
    float liquid_nearby;                  // 108
    float lava_nearby;                    // 109
    float terrain_reserved[22];           // 110-127

    // Navigation (32 floats)
    float stronghold_found;               // 128
    float stronghold_dir_x, stronghold_dir_z; // 129-130
    float stronghold_dist;                // 131
    float portal_frame_eyes;              // 132: 0-12
    float nether_portal_dist;             // 133
    float nether_portal_dir_x, dir_z;     // 134-135
    float fortress_found;                 // 136
    float fortress_dir_x, fortress_dir_z; // 137-138
    float fortress_dist;                  // 139
    float nav_reserved[20];               // 140-159

    // Progress (32 floats)
    float stage;                          // 160
    float stage_progress;                 // 161: 0-1 completion
    float deaths;                         // 162
    float time_elapsed;                   // 163: ticks / 20 / 60 (minutes)
    float crystals_destroyed;             // 164
    float dragon_damage_dealt;            // 165
    float beds_placed;                    // 166
    float portals_lit;                    // 167
    float progress_reserved[24];          // 168-191

    // Dragon (32 floats) - only valid in End
    float dragon_health;                  // 192
    float dragon_x, dragon_y, dragon_z;   // 193-195
    float dragon_dist;                    // 196
    float dragon_dir_x, dragon_dir_z;     // 197-198
    float dragon_phase;                   // 199
    float dragon_perching;                // 200
    float dragon_charging;                // 201
    float can_hit_dragon;                 // 202
    float crystals_remaining;             // 203
    float nearest_crystal_dist;           // 204
    float nearest_crystal_dir_x, nearest_crystal_dir_z; // 205-206
    float dragon_reserved[25];            // 207-223  (note: 25 not 17)

    // Reserved (32 floats)
    float reserved[32];                   // 224-255
};

static_assert(sizeof(ExtendedObservation) == EXTENDED_OBS_SIZE * sizeof(float));

class ObservationEncoder {
public:
    explicit ObservationEncoder(uint32_t num_envs);

    // Encode full observation
    void encode(uint32_t env_id,
               const PlayerFull& player,
               const WorldState& world,
               const std::vector<Entity>& entities,
               const Dragon* dragon,  // nullptr if not in End
               const Crystal* crystals,
               ExtendedObservation& out) const;

    // Encode to float array
    void encode_to_buffer(uint32_t env_id,
                         const PlayerFull& player,
                         const WorldState& world,
                         const std::vector<Entity>& entities,
                         const Dragon* dragon,
                         const Crystal* crystals,
                         float* out) const;

    // GPU observation generation (dispatch shader)
    void dispatch_encode(VulkanContext& ctx,
                        Buffer& player_buffer,
                        Buffer& world_buffer,
                        Buffer& entity_buffer,
                        Buffer& dragon_buffer,
                        Buffer& crystal_buffer,
                        Buffer& output_buffer,
                        uint32_t num_envs);

private:
    void encode_player(const PlayerFull& player, ExtendedObservation& out) const;
    void encode_inventory(const PlayerFull& player, ExtendedObservation& out) const;
    void encode_entities(const std::vector<Entity>& entities,
                        float px, float pz, ExtendedObservation& out) const;
    void encode_navigation(const WorldState& world,
                          float px, float pz, ExtendedObservation& out) const;
    void encode_progress(const WorldState& world, ExtendedObservation& out) const;
    void encode_dragon(const Dragon* dragon, const Crystal* crystals,
                      float px, float pz, ExtendedObservation& out) const;

    uint32_t num_envs_;
};

} // namespace mc189
