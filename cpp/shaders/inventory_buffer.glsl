/*
 * inventory_buffer.glsl - GPU-side inventory buffer structures for Minecraft 1.8.9 speedrun sim
 *
 * This header defines the inventory data structures for GPU compute shaders.
 * Include this in any shader that needs to read/write player inventories.
 *
 * Minecraft 1.8.9 inventory layout:
 *   Slots 0-8:   Hotbar (quick select)
 *   Slots 9-35:  Main inventory (27 slots)
 *   Slots 36-39: Armor (helmet, chestplate, leggings, boots)
 *   Slot 40:     Offhand (shield slot, technically 1.9+ but present in layout)
 *   Slot 41:     Cursor (held by mouse in UI)
 */

#ifndef INVENTORY_BUFFER_GLSL
#define INVENTORY_BUFFER_GLSL

// ============================================================================
// INVENTORY SLOT STRUCTURE
// ============================================================================

struct InventorySlot {
    uint item_id;     // 0 = empty slot
    uint count;       // Stack size (1-64 typically)
    uint damage;      // Durability remaining (0 for non-tools)
    uint enchants;    // Encoded enchantments (see enchantment encoding below)
};

// ============================================================================
// PLAYER INVENTORY STRUCTURE
// ============================================================================

struct PlayerInventory {
    InventorySlot hotbar[9];      // Slots 0-8 (quick access bar)
    InventorySlot main[27];       // Slots 9-35 (main storage)
    InventorySlot armor[4];       // Helmet(0), Chest(1), Legs(2), Boots(3)
    InventorySlot offhand;        // Shield/offhand slot
    InventorySlot cursor;         // Item held by cursor in UI
    uint selected_slot;           // Currently selected hotbar slot (0-8)
    uint _padding[3];             // Alignment padding to 64-byte boundary
};

// Size verification: 42 slots * 16 bytes + 4 * 4 bytes = 672 + 16 = 688 bytes

// ============================================================================
// ITEM ID CONSTANTS (Minecraft 1.8.9 numeric IDs)
// ============================================================================

// Empty/Air
const uint ITEM_NONE = 0;
const uint ITEM_AIR = 0;

// Basic blocks
const uint ITEM_STONE = 1;
const uint ITEM_GRASS = 2;
const uint ITEM_DIRT = 3;
const uint ITEM_COBBLESTONE = 4;
const uint ITEM_OAK_PLANKS = 5;
const uint ITEM_SAPLING = 6;
const uint ITEM_BEDROCK = 7;
const uint ITEM_SAND = 12;
const uint ITEM_GRAVEL = 13;
const uint ITEM_OAK_LOG = 17;
const uint ITEM_LEAVES = 18;
const uint ITEM_GLASS = 20;

// Ores and minerals
const uint ITEM_GOLD_ORE = 14;
const uint ITEM_IRON_ORE = 15;
const uint ITEM_COAL_ORE = 16;
const uint ITEM_LAPIS_ORE = 21;
const uint ITEM_DIAMOND_ORE = 56;
const uint ITEM_REDSTONE_ORE = 73;
const uint ITEM_EMERALD_ORE = 129;
const uint ITEM_NETHER_QUARTZ_ORE = 153;

// Crafting/utility blocks
const uint ITEM_CRAFTING_TABLE = 58;
const uint ITEM_FURNACE = 61;
const uint ITEM_CHEST = 54;
const uint ITEM_ENCHANTING_TABLE = 116;
const uint ITEM_ANVIL = 145;
const uint ITEM_BREWING_STAND = 117;

// Nether blocks
const uint ITEM_NETHERRACK = 87;
const uint ITEM_SOUL_SAND = 88;
const uint ITEM_GLOWSTONE = 89;
const uint ITEM_NETHER_BRICK = 112;
const uint ITEM_OBSIDIAN = 49;

// End blocks
const uint ITEM_END_STONE = 121;
const uint ITEM_END_PORTAL_FRAME = 120;

// Tools - Iron
const uint ITEM_IRON_SHOVEL = 256;
const uint ITEM_IRON_PICKAXE = 257;
const uint ITEM_IRON_AXE = 258;
const uint ITEM_FLINT_AND_STEEL = 259;
const uint ITEM_IRON_SWORD = 267;
const uint ITEM_IRON_HOE = 292;

// Tools - Diamond
const uint ITEM_DIAMOND_SHOVEL = 277;
const uint ITEM_DIAMOND_PICKAXE = 278;
const uint ITEM_DIAMOND_AXE = 279;
const uint ITEM_DIAMOND_SWORD = 276;
const uint ITEM_DIAMOND_HOE = 293;

// Tools - Stone
const uint ITEM_STONE_SHOVEL = 273;
const uint ITEM_STONE_PICKAXE = 274;
const uint ITEM_STONE_AXE = 275;
const uint ITEM_STONE_SWORD = 272;
const uint ITEM_STONE_HOE = 291;

// Tools - Wood
const uint ITEM_WOODEN_SHOVEL = 269;
const uint ITEM_WOODEN_PICKAXE = 270;
const uint ITEM_WOODEN_AXE = 271;
const uint ITEM_WOODEN_SWORD = 268;
const uint ITEM_WOODEN_HOE = 290;

// Tools - Gold
const uint ITEM_GOLDEN_SHOVEL = 284;
const uint ITEM_GOLDEN_PICKAXE = 285;
const uint ITEM_GOLDEN_AXE = 286;
const uint ITEM_GOLDEN_SWORD = 283;
const uint ITEM_GOLDEN_HOE = 294;

// Armor - Iron
const uint ITEM_IRON_HELMET = 306;
const uint ITEM_IRON_CHESTPLATE = 307;
const uint ITEM_IRON_LEGGINGS = 308;
const uint ITEM_IRON_BOOTS = 309;

// Armor - Diamond
const uint ITEM_DIAMOND_HELMET = 310;
const uint ITEM_DIAMOND_CHESTPLATE = 311;
const uint ITEM_DIAMOND_LEGGINGS = 312;
const uint ITEM_DIAMOND_BOOTS = 313;

// Armor - Gold
const uint ITEM_GOLDEN_HELMET = 314;
const uint ITEM_GOLDEN_CHESTPLATE = 315;
const uint ITEM_GOLDEN_LEGGINGS = 316;
const uint ITEM_GOLDEN_BOOTS = 317;

// Armor - Leather
const uint ITEM_LEATHER_HELMET = 298;
const uint ITEM_LEATHER_CHESTPLATE = 299;
const uint ITEM_LEATHER_LEGGINGS = 300;
const uint ITEM_LEATHER_BOOTS = 301;

// Armor - Chain
const uint ITEM_CHAINMAIL_HELMET = 302;
const uint ITEM_CHAINMAIL_CHESTPLATE = 303;
const uint ITEM_CHAINMAIL_LEGGINGS = 304;
const uint ITEM_CHAINMAIL_BOOTS = 305;

// Basic materials
const uint ITEM_COAL = 263;
const uint ITEM_DIAMOND = 264;
const uint ITEM_IRON_INGOT = 265;
const uint ITEM_GOLD_INGOT = 266;
const uint ITEM_STICK = 280;
const uint ITEM_STRING = 287;
const uint ITEM_FEATHER = 288;
const uint ITEM_GUNPOWDER = 289;
const uint ITEM_FLINT = 318;
const uint ITEM_LEATHER = 334;
const uint ITEM_BRICK = 336;
const uint ITEM_CLAY_BALL = 337;
const uint ITEM_PAPER = 339;
const uint ITEM_BOOK = 340;
const uint ITEM_SLIMEBALL = 341;
const uint ITEM_GLOWSTONE_DUST = 348;
const uint ITEM_BONE = 352;
const uint ITEM_SUGAR = 353;
const uint ITEM_NETHER_BRICK_ITEM = 405;

// Food
const uint ITEM_APPLE = 260;
const uint ITEM_GOLDEN_APPLE = 322;
const uint ITEM_BREAD = 297;
const uint ITEM_PORKCHOP = 319;
const uint ITEM_COOKED_PORKCHOP = 320;
const uint ITEM_BEEF = 363;
const uint ITEM_COOKED_BEEF = 364;
const uint ITEM_CHICKEN = 365;
const uint ITEM_COOKED_CHICKEN = 366;
const uint ITEM_ROTTEN_FLESH = 367;
const uint ITEM_CARROT = 391;
const uint ITEM_POTATO = 392;
const uint ITEM_BAKED_POTATO = 393;
const uint ITEM_MELON_SLICE = 360;

// Speedrun-critical items
const uint ITEM_BLAZE_ROD = 369;
const uint ITEM_BLAZE_POWDER = 377;
const uint ITEM_ENDER_PEARL = 368;
const uint ITEM_EYE_OF_ENDER = 381;
const uint ITEM_NETHER_WART = 372;
const uint ITEM_GHAST_TEAR = 370;
const uint ITEM_MAGMA_CREAM = 378;
const uint ITEM_ENDER_EYE = 381;  // Alias

// Potions and brewing
const uint ITEM_POTION = 373;
const uint ITEM_GLASS_BOTTLE = 374;
const uint ITEM_SPIDER_EYE = 375;
const uint ITEM_FERMENTED_SPIDER_EYE = 376;
const uint ITEM_GOLDEN_CARROT = 396;
const uint ITEM_GLISTERING_MELON = 382;

// Misc items
const uint ITEM_BOW = 261;
const uint ITEM_ARROW = 262;
const uint ITEM_BUCKET = 325;
const uint ITEM_WATER_BUCKET = 326;
const uint ITEM_LAVA_BUCKET = 327;
const uint ITEM_MILK_BUCKET = 335;
const uint ITEM_EGG = 344;
const uint ITEM_COMPASS = 345;
const uint ITEM_CLOCK = 347;
const uint ITEM_MAP = 358;
const uint ITEM_SHEARS = 359;
const uint ITEM_BED = 355;
const uint ITEM_TORCH = 50;
const uint ITEM_FISHING_ROD = 346;
const uint ITEM_LEAD = 420;
const uint ITEM_NAME_TAG = 421;
const uint ITEM_BOAT = 333;

// End-related
const uint ITEM_DRAGON_EGG = 122;
const uint ITEM_END_CRYSTAL = 426;

// ============================================================================
// STACK SIZE LOOKUP
// ============================================================================

// Returns max stack size for an item
// Most items stack to 64, tools/weapons/armor stack to 1, some special cases
uint get_stack_limit(uint item_id) {
    // Empty slot
    if (item_id == ITEM_NONE) return 0;

    // Tools (stack to 1)
    if (item_id >= ITEM_IRON_SHOVEL && item_id <= ITEM_FLINT_AND_STEEL) return 1;
    if (item_id >= ITEM_WOODEN_SWORD && item_id <= ITEM_WOODEN_HOE) return 1;
    if (item_id >= ITEM_STONE_SWORD && item_id <= ITEM_STONE_HOE) return 1;
    if (item_id >= ITEM_IRON_SWORD && item_id <= ITEM_IRON_HOE) return 1;
    if (item_id >= ITEM_DIAMOND_SWORD && item_id <= ITEM_DIAMOND_HOE) return 1;
    if (item_id >= ITEM_GOLDEN_SWORD && item_id <= ITEM_GOLDEN_HOE) return 1;

    // Weapons
    if (item_id == ITEM_BOW) return 1;
    if (item_id == ITEM_FISHING_ROD) return 1;
    if (item_id == ITEM_SHEARS) return 1;

    // Armor (stack to 1)
    if (item_id >= ITEM_LEATHER_HELMET && item_id <= ITEM_LEATHER_BOOTS) return 1;
    if (item_id >= ITEM_CHAINMAIL_HELMET && item_id <= ITEM_CHAINMAIL_BOOTS) return 1;
    if (item_id >= ITEM_IRON_HELMET && item_id <= ITEM_IRON_BOOTS) return 1;
    if (item_id >= ITEM_DIAMOND_HELMET && item_id <= ITEM_DIAMOND_BOOTS) return 1;
    if (item_id >= ITEM_GOLDEN_HELMET && item_id <= ITEM_GOLDEN_BOOTS) return 1;

    // Buckets (stack to 1 when filled)
    if (item_id == ITEM_WATER_BUCKET) return 1;
    if (item_id == ITEM_LAVA_BUCKET) return 1;
    if (item_id == ITEM_MILK_BUCKET) return 1;

    // Special stacks to 16
    if (item_id == ITEM_ENDER_PEARL) return 16;
    if (item_id == ITEM_EGG) return 16;
    if (item_id == ITEM_SNOWBALL) return 16;
    if (item_id == ITEM_BUCKET) return 16;
    if (item_id == ITEM_SIGN) return 16;

    // Potions stack to 1 in 1.8
    if (item_id == ITEM_POTION) return 1;

    // Default stack size
    return 64;
}

// Additional item constants for completeness
const uint ITEM_SNOWBALL = 332;
const uint ITEM_SIGN = 323;

// ============================================================================
// ITEM CATEGORY HELPERS
// ============================================================================

bool is_tool(uint item_id) {
    // Shovels
    if (item_id == ITEM_WOODEN_SHOVEL || item_id == ITEM_STONE_SHOVEL ||
        item_id == ITEM_IRON_SHOVEL || item_id == ITEM_DIAMOND_SHOVEL ||
        item_id == ITEM_GOLDEN_SHOVEL) return true;

    // Pickaxes
    if (item_id == ITEM_WOODEN_PICKAXE || item_id == ITEM_STONE_PICKAXE ||
        item_id == ITEM_IRON_PICKAXE || item_id == ITEM_DIAMOND_PICKAXE ||
        item_id == ITEM_GOLDEN_PICKAXE) return true;

    // Axes
    if (item_id == ITEM_WOODEN_AXE || item_id == ITEM_STONE_AXE ||
        item_id == ITEM_IRON_AXE || item_id == ITEM_DIAMOND_AXE ||
        item_id == ITEM_GOLDEN_AXE) return true;

    // Hoes
    if (item_id == ITEM_WOODEN_HOE || item_id == ITEM_STONE_HOE ||
        item_id == ITEM_IRON_HOE || item_id == ITEM_DIAMOND_HOE ||
        item_id == ITEM_GOLDEN_HOE) return true;

    // Misc tools
    if (item_id == ITEM_FLINT_AND_STEEL || item_id == ITEM_SHEARS ||
        item_id == ITEM_FISHING_ROD) return true;

    return false;
}

bool is_weapon(uint item_id) {
    // Swords
    if (item_id == ITEM_WOODEN_SWORD || item_id == ITEM_STONE_SWORD ||
        item_id == ITEM_IRON_SWORD || item_id == ITEM_DIAMOND_SWORD ||
        item_id == ITEM_GOLDEN_SWORD) return true;

    // Bow
    if (item_id == ITEM_BOW) return true;

    return false;
}

bool is_armor(uint item_id) {
    // Leather
    if (item_id >= ITEM_LEATHER_HELMET && item_id <= ITEM_LEATHER_BOOTS) return true;
    // Chainmail
    if (item_id >= ITEM_CHAINMAIL_HELMET && item_id <= ITEM_CHAINMAIL_BOOTS) return true;
    // Iron
    if (item_id >= ITEM_IRON_HELMET && item_id <= ITEM_IRON_BOOTS) return true;
    // Diamond
    if (item_id >= ITEM_DIAMOND_HELMET && item_id <= ITEM_DIAMOND_BOOTS) return true;
    // Gold
    if (item_id >= ITEM_GOLDEN_HELMET && item_id <= ITEM_GOLDEN_BOOTS) return true;

    return false;
}

bool is_food(uint item_id) {
    if (item_id == ITEM_APPLE || item_id == ITEM_GOLDEN_APPLE) return true;
    if (item_id == ITEM_BREAD) return true;
    if (item_id == ITEM_PORKCHOP || item_id == ITEM_COOKED_PORKCHOP) return true;
    if (item_id == ITEM_BEEF || item_id == ITEM_COOKED_BEEF) return true;
    if (item_id == ITEM_CHICKEN || item_id == ITEM_COOKED_CHICKEN) return true;
    if (item_id == ITEM_ROTTEN_FLESH) return true;
    if (item_id == ITEM_CARROT || item_id == ITEM_POTATO || item_id == ITEM_BAKED_POTATO) return true;
    if (item_id == ITEM_MELON_SLICE) return true;
    if (item_id == ITEM_GOLDEN_CARROT) return true;
    return false;
}

bool is_block(uint item_id) {
    // Most blocks have IDs < 256 in 1.8
    return item_id > 0 && item_id < 256;
}

// ============================================================================
// DURABILITY CONSTANTS
// ============================================================================

// Tool durability by material
const uint DURABILITY_WOOD = 60;
const uint DURABILITY_STONE = 132;
const uint DURABILITY_IRON = 251;
const uint DURABILITY_DIAMOND = 1562;
const uint DURABILITY_GOLD = 33;

// Armor durability multipliers (base * factor = total)
// Helmets: 11, Chestplates: 16, Leggings: 15, Boots: 13
const uint DURABILITY_LEATHER_BASE = 5;
const uint DURABILITY_CHAIN_BASE = 15;
const uint DURABILITY_IRON_BASE = 15;
const uint DURABILITY_DIAMOND_BASE = 33;
const uint DURABILITY_GOLD_BASE = 7;

uint get_max_durability(uint item_id) {
    // Tools
    if (item_id == ITEM_WOODEN_PICKAXE || item_id == ITEM_WOODEN_AXE ||
        item_id == ITEM_WOODEN_SHOVEL || item_id == ITEM_WOODEN_HOE ||
        item_id == ITEM_WOODEN_SWORD) return DURABILITY_WOOD;

    if (item_id == ITEM_STONE_PICKAXE || item_id == ITEM_STONE_AXE ||
        item_id == ITEM_STONE_SHOVEL || item_id == ITEM_STONE_HOE ||
        item_id == ITEM_STONE_SWORD) return DURABILITY_STONE;

    if (item_id == ITEM_IRON_PICKAXE || item_id == ITEM_IRON_AXE ||
        item_id == ITEM_IRON_SHOVEL || item_id == ITEM_IRON_HOE ||
        item_id == ITEM_IRON_SWORD) return DURABILITY_IRON;

    if (item_id == ITEM_DIAMOND_PICKAXE || item_id == ITEM_DIAMOND_AXE ||
        item_id == ITEM_DIAMOND_SHOVEL || item_id == ITEM_DIAMOND_HOE ||
        item_id == ITEM_DIAMOND_SWORD) return DURABILITY_DIAMOND;

    if (item_id == ITEM_GOLDEN_PICKAXE || item_id == ITEM_GOLDEN_AXE ||
        item_id == ITEM_GOLDEN_SHOVEL || item_id == ITEM_GOLDEN_HOE ||
        item_id == ITEM_GOLDEN_SWORD) return DURABILITY_GOLD;

    // Misc tools
    if (item_id == ITEM_FLINT_AND_STEEL) return 65;
    if (item_id == ITEM_SHEARS) return 238;
    if (item_id == ITEM_BOW) return 385;
    if (item_id == ITEM_FISHING_ROD) return 65;

    // Armor (simplified - actual varies by piece type)
    if (item_id >= ITEM_LEATHER_HELMET && item_id <= ITEM_LEATHER_BOOTS) return DURABILITY_LEATHER_BASE * 11;
    if (item_id >= ITEM_CHAINMAIL_HELMET && item_id <= ITEM_CHAINMAIL_BOOTS) return DURABILITY_CHAIN_BASE * 15;
    if (item_id >= ITEM_IRON_HELMET && item_id <= ITEM_IRON_BOOTS) return DURABILITY_IRON_BASE * 15;
    if (item_id >= ITEM_DIAMOND_HELMET && item_id <= ITEM_DIAMOND_BOOTS) return DURABILITY_DIAMOND_BASE * 33;
    if (item_id >= ITEM_GOLDEN_HELMET && item_id <= ITEM_GOLDEN_BOOTS) return DURABILITY_GOLD_BASE * 7;

    // Non-damageable items
    return 0;
}

// ============================================================================
// ENCHANTMENT ENCODING
// ============================================================================

// Enchantments are packed into a single uint32:
//   Bits 0-4:   Enchantment ID (0-31)
//   Bits 5-7:   Level (0-7)
//   Bits 8-12:  Second enchantment ID
//   Bits 13-15: Second enchantment level
//   Bits 16-20: Third enchantment ID
//   Bits 21-23: Third enchantment level
//   Bits 24-28: Fourth enchantment ID
//   Bits 29-31: Fourth enchantment level

// Speedrun-relevant enchantments
const uint ENCH_PROTECTION = 0;
const uint ENCH_FIRE_PROTECTION = 1;
const uint ENCH_FEATHER_FALLING = 2;
const uint ENCH_BLAST_PROTECTION = 3;
const uint ENCH_PROJECTILE_PROTECTION = 4;
const uint ENCH_RESPIRATION = 5;
const uint ENCH_AQUA_AFFINITY = 6;
const uint ENCH_THORNS = 7;
const uint ENCH_SHARPNESS = 16;
const uint ENCH_SMITE = 17;
const uint ENCH_BANE_OF_ARTHROPODS = 18;
const uint ENCH_KNOCKBACK = 19;
const uint ENCH_FIRE_ASPECT = 20;
const uint ENCH_LOOTING = 21;
const uint ENCH_EFFICIENCY = 32;
const uint ENCH_SILK_TOUCH = 33;
const uint ENCH_UNBREAKING = 34;
const uint ENCH_FORTUNE = 35;
const uint ENCH_POWER = 48;
const uint ENCH_PUNCH = 49;
const uint ENCH_FLAME = 50;
const uint ENCH_INFINITY = 51;

uint encode_enchant(uint ench_id, uint level) {
    return (ench_id & 0x1F) | ((level & 0x7) << 5);
}

uint decode_enchant_id(uint encoded, uint slot) {
    uint shift = slot * 8;
    return (encoded >> shift) & 0x1F;
}

uint decode_enchant_level(uint encoded, uint slot) {
    uint shift = slot * 8 + 5;
    return (encoded >> shift) & 0x7;
}

bool has_enchant(uint encoded, uint ench_id) {
    for (uint i = 0; i < 4; i++) {
        if (decode_enchant_id(encoded, i) == ench_id &&
            decode_enchant_level(encoded, i) > 0) {
            return true;
        }
    }
    return false;
}

uint get_enchant_level(uint encoded, uint ench_id) {
    for (uint i = 0; i < 4; i++) {
        if (decode_enchant_id(encoded, i) == ench_id) {
            return decode_enchant_level(encoded, i);
        }
    }
    return 0;
}

// ============================================================================
// INVENTORY SLOT INDEX CONSTANTS
// ============================================================================

const uint SLOT_HOTBAR_START = 0;
const uint SLOT_HOTBAR_END = 8;
const uint SLOT_MAIN_START = 9;
const uint SLOT_MAIN_END = 35;
const uint SLOT_ARMOR_HELMET = 36;
const uint SLOT_ARMOR_CHEST = 37;
const uint SLOT_ARMOR_LEGS = 38;
const uint SLOT_ARMOR_BOOTS = 39;
const uint SLOT_OFFHAND = 40;
const uint SLOT_CURSOR = 41;
const uint SLOT_COUNT = 42;

// ============================================================================
// INVENTORY HELPER FUNCTIONS
// ============================================================================

// Get slot from PlayerInventory by linear index
InventorySlot get_slot(PlayerInventory inv, uint index) {
    if (index < 9) {
        return inv.hotbar[index];
    } else if (index < 36) {
        return inv.main[index - 9];
    } else if (index < 40) {
        return inv.armor[index - 36];
    } else if (index == 40) {
        return inv.offhand;
    } else {
        return inv.cursor;
    }
}

// Check if slot is empty
bool is_slot_empty(InventorySlot slot) {
    return slot.item_id == ITEM_NONE;
}

// Check if two slots can stack (same item, not full)
bool can_stack(InventorySlot a, InventorySlot b) {
    if (a.item_id != b.item_id) return false;
    if (a.item_id == ITEM_NONE) return false;
    if (a.damage != b.damage) return false;  // Different damage = different NBT
    if (a.enchants != b.enchants) return false;  // Different enchants = different NBT
    uint max_stack = get_stack_limit(a.item_id);
    return b.count < max_stack;
}

// Calculate how many items can be added to a slot
uint space_in_slot(InventorySlot slot) {
    if (slot.item_id == ITEM_NONE) return 64;  // Empty slot, assume max
    uint max_stack = get_stack_limit(slot.item_id);
    return max_stack > slot.count ? max_stack - slot.count : 0;
}

// Get currently held item (selected hotbar slot)
InventorySlot get_held_item(PlayerInventory inv) {
    return inv.hotbar[inv.selected_slot];
}

// Count total of an item across inventory
uint count_item(PlayerInventory inv, uint item_id) {
    uint total = 0;

    // Hotbar
    for (uint i = 0; i < 9; i++) {
        if (inv.hotbar[i].item_id == item_id) {
            total += inv.hotbar[i].count;
        }
    }

    // Main inventory
    for (uint i = 0; i < 27; i++) {
        if (inv.main[i].item_id == item_id) {
            total += inv.main[i].count;
        }
    }

    // Offhand
    if (inv.offhand.item_id == item_id) {
        total += inv.offhand.count;
    }

    return total;
}

// Find first slot containing item (-1 if not found)
int find_item(PlayerInventory inv, uint item_id) {
    // Check hotbar first (faster access)
    for (uint i = 0; i < 9; i++) {
        if (inv.hotbar[i].item_id == item_id) {
            return int(i);
        }
    }

    // Check main inventory
    for (uint i = 0; i < 27; i++) {
        if (inv.main[i].item_id == item_id) {
            return int(i + 9);
        }
    }

    return -1;
}

// Find first empty slot in hotbar (-1 if full)
int find_empty_hotbar(PlayerInventory inv) {
    for (uint i = 0; i < 9; i++) {
        if (inv.hotbar[i].item_id == ITEM_NONE) {
            return int(i);
        }
    }
    return -1;
}

// Find first empty slot in main inventory (-1 if full)
int find_empty_main(PlayerInventory inv) {
    for (uint i = 0; i < 27; i++) {
        if (inv.main[i].item_id == ITEM_NONE) {
            return int(i + 9);
        }
    }
    return -1;
}

// Check if player has enough of an item
bool has_item(PlayerInventory inv, uint item_id, uint required_count) {
    return count_item(inv, item_id) >= required_count;
}

// Check if player has any armor equipped
bool has_armor_equipped(PlayerInventory inv) {
    for (uint i = 0; i < 4; i++) {
        if (inv.armor[i].item_id != ITEM_NONE) {
            return true;
        }
    }
    return false;
}

// Get total armor value (protection points)
uint get_armor_value(PlayerInventory inv) {
    uint total = 0;

    for (uint i = 0; i < 4; i++) {
        uint item = inv.armor[i].item_id;

        // Leather
        if (item == ITEM_LEATHER_HELMET) total += 1;
        else if (item == ITEM_LEATHER_CHESTPLATE) total += 3;
        else if (item == ITEM_LEATHER_LEGGINGS) total += 2;
        else if (item == ITEM_LEATHER_BOOTS) total += 1;

        // Chain
        else if (item == ITEM_CHAINMAIL_HELMET) total += 2;
        else if (item == ITEM_CHAINMAIL_CHESTPLATE) total += 5;
        else if (item == ITEM_CHAINMAIL_LEGGINGS) total += 4;
        else if (item == ITEM_CHAINMAIL_BOOTS) total += 1;

        // Iron
        else if (item == ITEM_IRON_HELMET) total += 2;
        else if (item == ITEM_IRON_CHESTPLATE) total += 6;
        else if (item == ITEM_IRON_LEGGINGS) total += 5;
        else if (item == ITEM_IRON_BOOTS) total += 2;

        // Diamond
        else if (item == ITEM_DIAMOND_HELMET) total += 3;
        else if (item == ITEM_DIAMOND_CHESTPLATE) total += 8;
        else if (item == ITEM_DIAMOND_LEGGINGS) total += 6;
        else if (item == ITEM_DIAMOND_BOOTS) total += 3;

        // Gold
        else if (item == ITEM_GOLDEN_HELMET) total += 2;
        else if (item == ITEM_GOLDEN_CHESTPLATE) total += 5;
        else if (item == ITEM_GOLDEN_LEGGINGS) total += 3;
        else if (item == ITEM_GOLDEN_BOOTS) total += 1;
    }

    return total;
}

// ============================================================================
// SPEEDRUN-SPECIFIC HELPERS
// ============================================================================

// Check if player has materials for eye of ender
bool can_craft_eye_of_ender(PlayerInventory inv) {
    return has_item(inv, ITEM_ENDER_PEARL, 1) && has_item(inv, ITEM_BLAZE_POWDER, 1);
}

// Count eyes of ender (needed: 12 for portal, but often less due to existing frames)
uint count_eyes_of_ender(PlayerInventory inv) {
    uint eyes = count_item(inv, ITEM_EYE_OF_ENDER);
    uint pearls = count_item(inv, ITEM_ENDER_PEARL);
    uint powder = count_item(inv, ITEM_BLAZE_POWDER);
    uint craftable = min(pearls, powder);
    return eyes + craftable;
}

// Check if player has blaze rods (for powder)
uint count_blaze_resources(PlayerInventory inv) {
    uint rods = count_item(inv, ITEM_BLAZE_ROD);
    uint powder = count_item(inv, ITEM_BLAZE_POWDER);
    return rods * 2 + powder;  // Each rod gives 2 powder
}

// Check if player can enter nether (has flint and steel or fire charge)
bool can_light_portal(PlayerInventory inv) {
    return has_item(inv, ITEM_FLINT_AND_STEEL, 1);
}

// Check if player has beds for dragon fight
uint count_beds(PlayerInventory inv) {
    return count_item(inv, ITEM_BED);
}

// Check if player has food
bool has_food(PlayerInventory inv) {
    // Check common food items
    if (count_item(inv, ITEM_BREAD) > 0) return true;
    if (count_item(inv, ITEM_COOKED_BEEF) > 0) return true;
    if (count_item(inv, ITEM_COOKED_PORKCHOP) > 0) return true;
    if (count_item(inv, ITEM_COOKED_CHICKEN) > 0) return true;
    if (count_item(inv, ITEM_APPLE) > 0) return true;
    if (count_item(inv, ITEM_GOLDEN_APPLE) > 0) return true;
    if (count_item(inv, ITEM_BAKED_POTATO) > 0) return true;
    if (count_item(inv, ITEM_CARROT) > 0) return true;
    if (count_item(inv, ITEM_GOLDEN_CARROT) > 0) return true;
    return false;
}

// Check if player has iron pickaxe (needed for obsidian/diamonds)
bool has_iron_or_better_pick(PlayerInventory inv) {
    return has_item(inv, ITEM_IRON_PICKAXE, 1) ||
           has_item(inv, ITEM_DIAMOND_PICKAXE, 1);
}

// Check if player has diamond pickaxe (needed for obsidian)
bool has_diamond_pick(PlayerInventory inv) {
    return has_item(inv, ITEM_DIAMOND_PICKAXE, 1);
}

#endif // INVENTORY_BUFFER_GLSL
