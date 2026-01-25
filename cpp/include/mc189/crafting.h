#pragma once
#include "mc189/items.h"
#include "mc189/inventory.h"
#include <vector>
#include <array>

namespace mc189 {

// Shapeless recipe (order doesn't matter)
struct ShapelessRecipe {
    std::vector<ItemStack> inputs;
    ItemStack output;
};

// Shaped recipe (3x3 grid)
struct ShapedRecipe {
    std::array<ItemID, 9> pattern;  // Row-major, AIR = empty
    ItemStack output;
};

class CraftingManager {
public:
    CraftingManager();

    // Check if recipe is craftable with current inventory
    bool can_craft(const InventoryManager& inv, uint32_t env_id,
                  const ShapelessRecipe& recipe) const;
    bool can_craft(const InventoryManager& inv, uint32_t env_id,
                  const ShapedRecipe& recipe) const;

    // Execute craft (consumes materials, adds output)
    bool craft(InventoryManager& inv, uint32_t env_id,
              const ShapelessRecipe& recipe);
    bool craft(InventoryManager& inv, uint32_t env_id,
              const ShapedRecipe& recipe);

    // Quick-craft by output item (finds recipe automatically)
    bool quick_craft(InventoryManager& inv, uint32_t env_id,
                    ItemID output, uint32_t count = 1);

    // Get available recipes for current inventory
    std::vector<ItemID> get_craftable(const InventoryManager& inv,
                                      uint32_t env_id) const;

    // Speedrun-essential recipes
    static const std::vector<ShapedRecipe>& get_shaped_recipes();
    static const std::vector<ShapelessRecipe>& get_shapeless_recipes();

private:
    std::vector<ShapedRecipe> shaped_recipes_;
    std::vector<ShapelessRecipe> shapeless_recipes_;
};

// Key speedrun recipes
void init_speedrun_recipes(std::vector<ShapedRecipe>& shaped,
                          std::vector<ShapelessRecipe>& shapeless);

} // namespace mc189
