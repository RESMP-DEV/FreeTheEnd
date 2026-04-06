#pragma once
#include <cstdint>
#include <array>

namespace mc189 {

enum class Dimension : int32_t {
    NETHER = -1,
    OVERWORLD = 0,
    END = 1
};

struct DimensionConfig {
    Dimension id;
    float gravity;           // blocks/tick^2
    float sky_light;         // 0-15
    bool has_ceiling;
    bool has_weather;
    float ambient_light;
    int32_t min_y;
    int32_t max_y;
    int32_t sea_level;
};

constexpr DimensionConfig DIMENSION_CONFIGS[] = {
    {Dimension::NETHER, 0.08f, 0.0f, true, false, 0.1f, 0, 128, 32},
    {Dimension::OVERWORLD, 0.08f, 15.0f, false, true, 0.0f, 0, 256, 63},
    {Dimension::END, 0.08f, 0.0f, false, false, 0.0f, 0, 256, 0}
};

const DimensionConfig& get_dimension_config(Dimension dim);
const char* dimension_name(Dimension dim);

} // namespace mc189
