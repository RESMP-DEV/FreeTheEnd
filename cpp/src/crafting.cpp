#include "mc189/crafting.h"
#include <algorithm>
#include <unordered_map>

namespace mc189 {

namespace {

constexpr ItemID AIR = ItemID(0);

// Helper: build material count map from shaped recipe pattern
std::unordered_map<uint16_t, uint32_t> shaped_materials(const ShapedRecipe& recipe) {
    std::unordered_map<uint16_t, uint32_t> mats;
    for (auto id : recipe.pattern) {
        if (id != AIR) {
            mats[static_cast<uint16_t>(id)]++;
        }
    }
    return mats;
}

} // namespace

// ============================================================================
// Recipe definitions
// ============================================================================

void init_speedrun_recipes(std::vector<ShapedRecipe>& shaped,
                          std::vector<ShapelessRecipe>& shapeless) {
    shaped.clear();
    shapeless.clear();

    // --- Shapeless recipes ---

    // Planks: 1 log -> 4 planks (shapeless, any log type works)
    shapeless.push_back({
        {{ItemID::LOG_ITEM, 1, 0}},
        {ItemID::PLANKS_ITEM, 4, 0}
    });

    // Blaze powder: 1 blaze rod -> 2 blaze powder
    shapeless.push_back({
        {{ItemID::BLAZE_ROD, 1, 0}},
        {ItemID::BLAZE_POWDER, 2, 0}
    });

    // --- Shaped recipes ---

    // Sticks: 2 planks vertical
    // [  ][P][  ]
    // [  ][P][  ]
    // [  ][  ][  ]
    shaped.push_back({
        {AIR, ItemID::PLANKS_ITEM, AIR,
         AIR, ItemID::PLANKS_ITEM, AIR,
         AIR, AIR, AIR},
        {ItemID::STICK, 4, 0}
    });

    // Crafting table: 4 planks in 2x2
    // [P][P][  ]
    // [P][P][  ]
    // [  ][  ][  ]
    shaped.push_back({
        {ItemID::PLANKS_ITEM, ItemID::PLANKS_ITEM, AIR,
         ItemID::PLANKS_ITEM, ItemID::PLANKS_ITEM, AIR,
         AIR, AIR, AIR},
        {ItemID::CRAFTING_TABLE_ITEM, 1, 0}
    });

    // Wooden pickaxe
    // [P][P][P]
    // [  ][S][  ]
    // [  ][S][  ]
    shaped.push_back({
        {ItemID::PLANKS_ITEM, ItemID::PLANKS_ITEM, ItemID::PLANKS_ITEM,
         AIR, ItemID::STICK, AIR,
         AIR, ItemID::STICK, AIR},
        {ItemID::WOODEN_PICKAXE, 1, 0}
    });

    // Stone pickaxe
    shaped.push_back({
        {ItemID::COBBLESTONE_ITEM, ItemID::COBBLESTONE_ITEM, ItemID::COBBLESTONE_ITEM,
         AIR, ItemID::STICK, AIR,
         AIR, ItemID::STICK, AIR},
        {ItemID::STONE_PICKAXE, 1, 0}
    });

    // Iron pickaxe
    shaped.push_back({
        {ItemID::IRON_INGOT, ItemID::IRON_INGOT, ItemID::IRON_INGOT,
         AIR, ItemID::STICK, AIR,
         AIR, ItemID::STICK, AIR},
        {ItemID::IRON_PICKAXE, 1, 0}
    });

    // Diamond pickaxe
    shaped.push_back({
        {ItemID::DIAMOND, ItemID::DIAMOND, ItemID::DIAMOND,
         AIR, ItemID::STICK, AIR,
         AIR, ItemID::STICK, AIR},
        {ItemID::DIAMOND_PICKAXE, 1, 0}
    });

    // Wooden sword
    // [  ][P][  ]
    // [  ][P][  ]
    // [  ][S][  ]
    shaped.push_back({
        {AIR, ItemID::PLANKS_ITEM, AIR,
         AIR, ItemID::PLANKS_ITEM, AIR,
         AIR, ItemID::STICK, AIR},
        {ItemID::WOODEN_SWORD, 1, 0}
    });

    // Stone sword
    shaped.push_back({
        {AIR, ItemID::COBBLESTONE_ITEM, AIR,
         AIR, ItemID::COBBLESTONE_ITEM, AIR,
         AIR, ItemID::STICK, AIR},
        {ItemID::STONE_SWORD, 1, 0}
    });

    // Iron sword
    shaped.push_back({
        {AIR, ItemID::IRON_INGOT, AIR,
         AIR, ItemID::IRON_INGOT, AIR,
         AIR, ItemID::STICK, AIR},
        {ItemID::IRON_SWORD, 1, 0}
    });

    // Diamond sword
    shaped.push_back({
        {AIR, ItemID::DIAMOND, AIR,
         AIR, ItemID::DIAMOND, AIR,
         AIR, ItemID::STICK, AIR},
        {ItemID::DIAMOND_SWORD, 1, 0}
    });

    // Furnace: 8 cobblestone ring
    // [C][C][C]
    // [C][  ][C]
    // [C][C][C]
    shaped.push_back({
        {ItemID::COBBLESTONE_ITEM, ItemID::COBBLESTONE_ITEM, ItemID::COBBLESTONE_ITEM,
         ItemID::COBBLESTONE_ITEM, AIR, ItemID::COBBLESTONE_ITEM,
         ItemID::COBBLESTONE_ITEM, ItemID::COBBLESTONE_ITEM, ItemID::COBBLESTONE_ITEM},
        {ItemID::FURNACE_ITEM, 1, 0}
    });

    // Bucket: 3 iron ingots in V shape
    // [I][  ][I]
    // [  ][I][  ]
    // [  ][  ][  ]
    shaped.push_back({
        {ItemID::IRON_INGOT, AIR, ItemID::IRON_INGOT,
         AIR, ItemID::IRON_INGOT, AIR,
         AIR, AIR, AIR},
        {ItemID::BUCKET, 1, 0}
    });

    // Flint and steel (shaped 1x2)
    // [  ][  ][  ]
    // [I][  ][  ]
    // [F][  ][  ]
    shaped.push_back({
        {AIR, AIR, AIR,
         ItemID::IRON_INGOT, AIR, AIR,
         ItemID::FLINT, AIR, AIR},
        {ItemID::FLINT_AND_STEEL, 1, 0}
    });

    // Eye of ender: blaze powder + ender pearl (shaped for simplicity)
    // In vanilla this is shapeless, but we put it in shapeless below
    // Actually we do it shapeless:
    shapeless.push_back({
        {{ItemID::BLAZE_POWDER, 1, 0}, {ItemID::ENDER_PEARL, 1, 0}},
        {ItemID::EYE_OF_ENDER, 1, 0}
    });

    // Bed: 3 wool + 3 planks
    // [W][W][W]
    // [P][P][P]
    // [  ][  ][  ]
    shaped.push_back({
        {ItemID::WOOL_ITEM, ItemID::WOOL_ITEM, ItemID::WOOL_ITEM,
         ItemID::PLANKS_ITEM, ItemID::PLANKS_ITEM, ItemID::PLANKS_ITEM,
         AIR, AIR, AIR},
        {ItemID::BED_ITEM, 1, 0}
    });
}

// ============================================================================
// CraftingManager implementation
// ============================================================================

CraftingManager::CraftingManager() {
    init_speedrun_recipes(shaped_recipes_, shapeless_recipes_);
}

bool CraftingManager::can_craft(const InventoryManager& inv, uint32_t env_id,
                                const ShapelessRecipe& recipe) const {
    for (const auto& input : recipe.inputs) {
        if (!inv.has_item(env_id, input.id, input.count)) {
            return false;
        }
    }
    return true;
}

bool CraftingManager::can_craft(const InventoryManager& inv, uint32_t env_id,
                                const ShapedRecipe& recipe) const {
    // Count required materials from pattern
    auto mats = shaped_materials(recipe);
    for (const auto& [item_id, needed] : mats) {
        if (!inv.has_item(env_id, ItemID(item_id), needed)) {
            return false;
        }
    }
    return true;
}

bool CraftingManager::craft(InventoryManager& inv, uint32_t env_id,
                            const ShapelessRecipe& recipe) {
    if (!can_craft(inv, env_id, recipe)) {
        return false;
    }
    // Consume inputs
    for (const auto& input : recipe.inputs) {
        inv.remove_item(env_id, input.id, input.count);
    }
    // Add output
    inv.add_item(env_id, recipe.output);
    return true;
}

bool CraftingManager::craft(InventoryManager& inv, uint32_t env_id,
                            const ShapedRecipe& recipe) {
    if (!can_craft(inv, env_id, recipe)) {
        return false;
    }
    // Consume materials
    auto mats = shaped_materials(recipe);
    for (const auto& [item_id, needed] : mats) {
        inv.remove_item(env_id, ItemID(item_id), needed);
    }
    // Add output
    inv.add_item(env_id, recipe.output);
    return true;
}

bool CraftingManager::quick_craft(InventoryManager& inv, uint32_t env_id,
                                  ItemID output, uint32_t count) {
    // Search shapeless recipes first
    for (const auto& recipe : shapeless_recipes_) {
        if (recipe.output.id == output) {
            for (uint32_t i = 0; i < count; ++i) {
                if (!craft(inv, env_id, recipe)) {
                    return i > 0;  // Partial success
                }
            }
            return true;
        }
    }
    // Search shaped recipes
    for (const auto& recipe : shaped_recipes_) {
        if (recipe.output.id == output) {
            for (uint32_t i = 0; i < count; ++i) {
                if (!craft(inv, env_id, recipe)) {
                    return i > 0;
                }
            }
            return true;
        }
    }
    return false;
}

std::vector<ItemID> CraftingManager::get_craftable(const InventoryManager& inv,
                                                    uint32_t env_id) const {
    std::vector<ItemID> result;
    for (const auto& recipe : shapeless_recipes_) {
        if (can_craft(inv, env_id, recipe)) {
            result.push_back(recipe.output.id);
        }
    }
    for (const auto& recipe : shaped_recipes_) {
        if (can_craft(inv, env_id, recipe)) {
            result.push_back(recipe.output.id);
        }
    }
    return result;
}

const std::vector<ShapedRecipe>& CraftingManager::get_shaped_recipes() {
    static std::vector<ShapedRecipe> shaped;
    static std::vector<ShapelessRecipe> shapeless;
    static bool init = false;
    if (!init) {
        init_speedrun_recipes(shaped, shapeless);
        init = true;
    }
    return shaped;
}

const std::vector<ShapelessRecipe>& CraftingManager::get_shapeless_recipes() {
    static std::vector<ShapedRecipe> shaped;
    static std::vector<ShapelessRecipe> shapeless;
    static bool init = false;
    if (!init) {
        init_speedrun_recipes(shaped, shapeless);
        init = true;
    }
    return shapeless;
}

// ============================================================================
// InventoryManager implementation (defined here to keep a single TU)
// ============================================================================

InventoryManager::InventoryManager(uint32_t num_envs)
    : envs_(num_envs) {}

uint32_t InventoryManager::count_item(uint32_t env_id, ItemID item) const {
    uint32_t total = 0;
    const auto& state = envs_[env_id];
    for (const auto& slot : state.slots) {
        if (slot.id == item) {
            total += slot.count;
        }
    }
    return total;
}

bool InventoryManager::has_item(uint32_t env_id, ItemID item, uint32_t count) const {
    return count_item(env_id, item) >= count;
}

bool InventoryManager::remove_item(uint32_t env_id, ItemID item, uint32_t count) {
    uint32_t remaining = count;
    auto& state = envs_[env_id];
    for (auto& slot : state.slots) {
        if (slot.id == item && remaining > 0) {
            uint32_t take = std::min(static_cast<uint32_t>(slot.count), remaining);
            slot.count -= static_cast<uint8_t>(take);
            remaining -= take;
            if (slot.count == 0) {
                slot.id = ItemID(0);
                slot.damage = 0;
            }
        }
    }
    return remaining == 0;
}

bool InventoryManager::add_item(uint32_t env_id, ItemStack stack) {
    auto& state = envs_[env_id];
    uint32_t to_add = stack.count;

    // First pass: stack into existing matching slots
    for (auto& slot : state.slots) {
        if (to_add == 0) break;
        if (slot.id == stack.id && slot.damage == stack.damage && slot.count < 64) {
            uint32_t space = 64 - slot.count;
            uint32_t add = std::min(space, to_add);
            slot.count += static_cast<uint8_t>(add);
            to_add -= add;
        }
    }

    // Second pass: use empty slots
    for (auto& slot : state.slots) {
        if (to_add == 0) break;
        if (slot.empty()) {
            uint32_t add = std::min(to_add, 64u);
            slot.id = stack.id;
            slot.damage = stack.damage;
            slot.count = static_cast<uint8_t>(add);
            to_add -= add;
        }
    }

    return to_add == 0;
}

void InventoryManager::reset(uint32_t env_id) {
    envs_[env_id] = InventoryState{};
}

} // namespace mc189
