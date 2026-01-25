#include "mc189/dimension.h"
#include <stdexcept>

namespace mc189 {

const DimensionConfig& get_dimension_config(Dimension dim) {
    switch (dim) {
        case Dimension::NETHER:    return DIMENSION_CONFIGS[0];
        case Dimension::OVERWORLD: return DIMENSION_CONFIGS[1];
        case Dimension::END:       return DIMENSION_CONFIGS[2];
    }
    throw std::invalid_argument("Unknown dimension");
}

const char* dimension_name(Dimension dim) {
    switch (dim) {
        case Dimension::NETHER:    return "the_nether";
        case Dimension::OVERWORLD: return "overworld";
        case Dimension::END:       return "the_end";
    }
    return "unknown";
}

} // namespace mc189
