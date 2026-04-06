#pragma once

#include <cstdint>

namespace mc189 {

// 2D integer coordinates for fortress positions
struct Vec2i {
  int32_t x;
  int32_t z;

  Vec2i();
  Vec2i(int32_t x_, int32_t z_);
};

// Get the fortress position within a specific grid cell
// cell_x, cell_z: grid cell coordinates (not block coordinates)
// seed: world seed
Vec2i get_fortress_in_cell(int32_t cell_x, int32_t cell_z, uint64_t seed);

// Find the nearest fortress to a player position
// player_x, player_z: player's current position in Nether coordinates
// seed: world seed
// Returns: fortress coordinates in Nether blocks
Vec2i find_nearest_fortress(float player_x, float player_z, uint64_t seed);

// Find all fortresses within a given radius
// player_x, player_z: player's current position in Nether coordinates
// radius: search radius in blocks
// seed: world seed
// out_fortresses: output buffer for fortress positions
// max_fortresses: maximum number of fortresses to return
// Returns: number of fortresses found
int32_t find_fortresses_in_radius(float player_x, float player_z, float radius, uint64_t seed,
                                   Vec2i* out_fortresses, int32_t max_fortresses);

// Check if a position is inside a fortress bounding box (conservative bounds)
bool is_inside_fortress(float pos_x, float pos_z, const Vec2i& fortress_pos);

// Get direction angle from player to fortress
// Returns angle in radians, with 0 = +Z (south), PI/2 = +X (west)
float get_fortress_direction(float player_x, float player_z, const Vec2i& fortress_pos);

// Get distance to fortress in blocks
float get_fortress_distance(float player_x, float player_z, const Vec2i& fortress_pos);

// Coordinate conversion utilities
Vec2i overworld_to_nether(int32_t ow_x, int32_t ow_z);
Vec2i nether_to_overworld(int32_t nether_x, int32_t nether_z);

} // namespace mc189
