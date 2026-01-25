#pragma once
#include <cstdint>
#include <string_view>

namespace mc189 {

// Item IDs matching Minecraft 1.8.9
enum class ItemID : uint16_t {
    // Tools
    IRON_SHOVEL = 256,
    IRON_PICKAXE = 257,
    IRON_AXE = 258,
    FLINT_AND_STEEL = 259,
    APPLE = 260,
    BOW = 261,
    ARROW = 262,
    COAL = 263,
    DIAMOND = 264,
    IRON_INGOT = 265,
    GOLD_INGOT = 266,
    IRON_SWORD = 267,
    WOODEN_SWORD = 268,
    WOODEN_SHOVEL = 269,
    WOODEN_PICKAXE = 270,
    WOODEN_AXE = 271,
    STONE_SWORD = 272,
    STONE_SHOVEL = 273,
    STONE_PICKAXE = 274,
    STONE_AXE = 275,
    DIAMOND_SWORD = 276,
    DIAMOND_SHOVEL = 277,
    DIAMOND_PICKAXE = 278,
    DIAMOND_AXE = 279,
    STICK = 280,
    BOWL = 281,
    MUSHROOM_STEW = 282,
    GOLDEN_SWORD = 283,
    GOLDEN_SHOVEL = 284,
    GOLDEN_PICKAXE = 285,
    GOLDEN_AXE = 286,
    STRING = 287,
    FEATHER = 288,
    GUNPOWDER = 289,
    WOODEN_HOE = 290,
    STONE_HOE = 291,
    IRON_HOE = 292,
    DIAMOND_HOE = 293,
    GOLDEN_HOE = 294,
    WHEAT_SEEDS = 295,
    WHEAT = 296,
    BREAD = 297,
    LEATHER_HELMET = 298,
    LEATHER_CHESTPLATE = 299,
    LEATHER_LEGGINGS = 300,
    LEATHER_BOOTS = 301,
    CHAINMAIL_HELMET = 302,
    CHAINMAIL_CHESTPLATE = 303,
    CHAINMAIL_LEGGINGS = 304,
    CHAINMAIL_BOOTS = 305,
    IRON_HELMET = 306,
    IRON_CHESTPLATE = 307,
    IRON_LEGGINGS = 308,
    IRON_BOOTS = 309,
    DIAMOND_HELMET = 310,
    DIAMOND_CHESTPLATE = 311,
    DIAMOND_LEGGINGS = 312,
    DIAMOND_BOOTS = 313,
    GOLDEN_HELMET = 314,
    GOLDEN_CHESTPLATE = 315,
    GOLDEN_LEGGINGS = 316,
    GOLDEN_BOOTS = 317,
    FLINT = 318,
    RAW_PORKCHOP = 319,
    COOKED_PORKCHOP = 320,
    BUCKET = 325,
    WATER_BUCKET = 326,
    LAVA_BUCKET = 327,
    ENDER_PEARL = 368,
    BLAZE_ROD = 369,
    BLAZE_POWDER = 377,
    EYE_OF_ENDER = 381,
    ENDER_EYE = 381,  // Alias

    // Block items (same ID as block)
    COBBLESTONE_ITEM = 4,
    PLANKS_ITEM = 5,
    LOG_ITEM = 17,
    WOOL_ITEM = 35,
    CRAFTING_TABLE_ITEM = 58,
    FURNACE_ITEM = 61,
    OBSIDIAN_ITEM = 49,
    BED_ITEM = 355,

    MAX_ITEM_ID = 512
};

enum class ItemCategory : uint8_t {
    NONE,
    TOOL,
    WEAPON,
    ARMOR,
    FOOD,
    BLOCK,
    MATERIAL,
    THROWABLE,
    SPECIAL
};

enum class ToolType : uint8_t {
    NONE,
    SWORD,
    PICKAXE,
    AXE,
    SHOVEL,
    HOE,
    BOW
};

enum class ToolMaterial : uint8_t {
    NONE,
    WOOD,
    STONE,
    IRON,
    GOLD,
    DIAMOND
};

struct ItemProperties {
    ItemID id;
    std::string_view name;
    ItemCategory category;
    uint16_t max_stack;
    uint16_t max_durability;
    ToolType tool_type;
    ToolMaterial material;
    float attack_damage;
    float attack_speed;
    float armor_points;
    float food_hunger;      // Hunger restored
    float food_saturation;  // Saturation restored
};


const ItemProperties& get_item_properties(ItemID id);
bool is_tool(ItemID id);
bool is_weapon(ItemID id);
bool is_armor(ItemID id);
bool is_food(ItemID id);
float get_mining_speed(ItemID tool, uint16_t block_id);

} // namespace mc189
