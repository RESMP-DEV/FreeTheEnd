/*
 * crafting_recipes.glsl - Compile-time recipe definitions for Minecraft 1.8.9
 *
 * This file defines crafting recipes as compile-time constants for GPU-based
 * recipe matching. Designed for speedrun-essential items.
 *
 * Usage: #include "crafting_recipes.glsl" in compute shaders
 */

#ifndef CRAFTING_RECIPES_GLSL
#define CRAFTING_RECIPES_GLSL

// ============================================================================
// Item IDs (Minecraft 1.8.9)
// ============================================================================

// Blocks
const uint ITEM_AIR = 0;
const uint ITEM_STONE = 1;
const uint ITEM_GRASS = 2;
const uint ITEM_DIRT = 3;
const uint ITEM_COBBLESTONE = 4;
const uint ITEM_PLANKS = 5;           // Oak planks (all variants use same base ID)
const uint ITEM_SAPLING = 6;
const uint ITEM_BEDROCK = 7;
const uint ITEM_SAND = 12;
const uint ITEM_GRAVEL = 13;
const uint ITEM_GOLD_ORE = 14;
const uint ITEM_IRON_ORE = 15;
const uint ITEM_COAL_ORE = 16;
const uint ITEM_LOG = 17;             // Oak log
const uint ITEM_LEAVES = 18;
const uint ITEM_GLASS = 20;
const uint ITEM_LAPIS_ORE = 21;
const uint ITEM_LAPIS_BLOCK = 22;
const uint ITEM_DISPENSER = 23;
const uint ITEM_SANDSTONE = 24;
const uint ITEM_NOTEBLOCK = 25;
const uint ITEM_BED_BLOCK = 26;
const uint ITEM_GOLDEN_RAIL = 27;
const uint ITEM_DETECTOR_RAIL = 28;
const uint ITEM_STICKY_PISTON = 29;
const uint ITEM_COBWEB = 30;
const uint ITEM_PISTON = 33;
const uint ITEM_WOOL = 35;
const uint ITEM_GOLD_BLOCK = 41;
const uint ITEM_IRON_BLOCK = 42;
const uint ITEM_DOUBLE_SLAB = 43;
const uint ITEM_SLAB = 44;
const uint ITEM_BRICK_BLOCK = 45;
const uint ITEM_TNT = 46;
const uint ITEM_BOOKSHELF = 47;
const uint ITEM_MOSSY_COBBLESTONE = 48;
const uint ITEM_OBSIDIAN = 49;
const uint ITEM_TORCH = 50;
const uint ITEM_FIRE = 51;
const uint ITEM_MOB_SPAWNER = 52;
const uint ITEM_WOODEN_STAIRS = 53;
const uint ITEM_CHEST = 54;
const uint ITEM_REDSTONE_WIRE = 55;
const uint ITEM_DIAMOND_ORE = 56;
const uint ITEM_DIAMOND_BLOCK = 57;
const uint ITEM_CRAFTING_TABLE = 58;
const uint ITEM_WHEAT_BLOCK = 59;
const uint ITEM_FARMLAND = 60;
const uint ITEM_FURNACE = 61;
const uint ITEM_LIT_FURNACE = 62;
const uint ITEM_STANDING_SIGN = 63;
const uint ITEM_WOODEN_DOOR = 64;
const uint ITEM_LADDER = 65;
const uint ITEM_RAIL = 66;
const uint ITEM_COBBLESTONE_STAIRS = 67;
const uint ITEM_WALL_SIGN = 68;
const uint ITEM_LEVER = 69;
const uint ITEM_STONE_PRESSURE_PLATE = 70;
const uint ITEM_IRON_DOOR = 71;
const uint ITEM_WOODEN_PRESSURE_PLATE = 72;
const uint ITEM_REDSTONE_ORE = 73;
const uint ITEM_REDSTONE_TORCH = 76;
const uint ITEM_STONE_BUTTON = 77;
const uint ITEM_SNOW = 78;
const uint ITEM_ICE = 79;
const uint ITEM_SNOW_BLOCK = 80;
const uint ITEM_CACTUS = 81;
const uint ITEM_CLAY_BLOCK = 82;
const uint ITEM_SUGAR_CANE = 83;
const uint ITEM_JUKEBOX = 84;
const uint ITEM_FENCE = 85;
const uint ITEM_PUMPKIN = 86;
const uint ITEM_NETHERRACK = 87;
const uint ITEM_SOUL_SAND = 88;
const uint ITEM_GLOWSTONE = 89;
const uint ITEM_PORTAL = 90;
const uint ITEM_JACK_O_LANTERN = 91;
const uint ITEM_CAKE_BLOCK = 92;
const uint ITEM_TRAPDOOR = 96;
const uint ITEM_STONEBRICK = 98;
const uint ITEM_IRON_BARS = 101;
const uint ITEM_GLASS_PANE = 102;
const uint ITEM_MELON_BLOCK = 103;
const uint ITEM_FENCE_GATE = 107;
const uint ITEM_BRICK_STAIRS = 108;
const uint ITEM_STONEBRICK_STAIRS = 109;
const uint ITEM_NETHER_BRICK = 112;
const uint ITEM_NETHER_BRICK_FENCE = 113;
const uint ITEM_NETHER_BRICK_STAIRS = 114;
const uint ITEM_ENCHANTING_TABLE = 116;
const uint ITEM_BREWING_STAND_BLOCK = 117;
const uint ITEM_CAULDRON_BLOCK = 118;
const uint ITEM_END_PORTAL = 119;
const uint ITEM_END_PORTAL_FRAME = 120;
const uint ITEM_END_STONE = 121;
const uint ITEM_DRAGON_EGG = 122;
const uint ITEM_REDSTONE_LAMP = 123;
const uint ITEM_EMERALD_ORE = 129;
const uint ITEM_ENDER_CHEST = 130;
const uint ITEM_TRIPWIRE_HOOK = 131;
const uint ITEM_EMERALD_BLOCK = 133;
const uint ITEM_BEACON = 138;
const uint ITEM_ANVIL = 145;
const uint ITEM_TRAPPED_CHEST = 146;
const uint ITEM_REDSTONE_BLOCK = 152;
const uint ITEM_QUARTZ_ORE = 153;
const uint ITEM_HOPPER = 154;
const uint ITEM_QUARTZ_BLOCK = 155;
const uint ITEM_DROPPER = 158;

// Items (256+)
const uint ITEM_IRON_SHOVEL = 256;
const uint ITEM_IRON_PICKAXE = 257;
const uint ITEM_IRON_AXE = 258;
const uint ITEM_FLINT_AND_STEEL = 259;
const uint ITEM_APPLE = 260;
const uint ITEM_BOW = 261;
const uint ITEM_ARROW = 262;
const uint ITEM_COAL = 263;
const uint ITEM_DIAMOND = 264;
const uint ITEM_IRON_INGOT = 265;
const uint ITEM_GOLD_INGOT = 266;
const uint ITEM_IRON_SWORD = 267;
const uint ITEM_WOODEN_SWORD = 268;
const uint ITEM_WOODEN_SHOVEL = 269;
const uint ITEM_WOODEN_PICKAXE = 270;
const uint ITEM_WOODEN_AXE = 271;
const uint ITEM_STONE_SWORD = 272;
const uint ITEM_STONE_SHOVEL = 273;
const uint ITEM_STONE_PICKAXE = 274;
const uint ITEM_STONE_AXE = 275;
const uint ITEM_DIAMOND_SWORD = 276;
const uint ITEM_DIAMOND_SHOVEL = 277;
const uint ITEM_DIAMOND_PICKAXE = 278;
const uint ITEM_DIAMOND_AXE = 279;
const uint ITEM_STICK = 280;
const uint ITEM_BOWL = 281;
const uint ITEM_MUSHROOM_STEW = 282;
const uint ITEM_GOLDEN_SWORD = 283;
const uint ITEM_GOLDEN_SHOVEL = 284;
const uint ITEM_GOLDEN_PICKAXE = 285;
const uint ITEM_GOLDEN_AXE = 286;
const uint ITEM_STRING = 287;
const uint ITEM_FEATHER = 288;
const uint ITEM_GUNPOWDER = 289;
const uint ITEM_WOODEN_HOE = 290;
const uint ITEM_STONE_HOE = 291;
const uint ITEM_IRON_HOE = 292;
const uint ITEM_DIAMOND_HOE = 293;
const uint ITEM_GOLDEN_HOE = 294;
const uint ITEM_WHEAT_SEEDS = 295;
const uint ITEM_WHEAT = 296;
const uint ITEM_BREAD = 297;
const uint ITEM_LEATHER_HELMET = 298;
const uint ITEM_LEATHER_CHESTPLATE = 299;
const uint ITEM_LEATHER_LEGGINGS = 300;
const uint ITEM_LEATHER_BOOTS = 301;
const uint ITEM_CHAINMAIL_HELMET = 302;
const uint ITEM_CHAINMAIL_CHESTPLATE = 303;
const uint ITEM_CHAINMAIL_LEGGINGS = 304;
const uint ITEM_CHAINMAIL_BOOTS = 305;
const uint ITEM_IRON_HELMET = 306;
const uint ITEM_IRON_CHESTPLATE = 307;
const uint ITEM_IRON_LEGGINGS = 308;
const uint ITEM_IRON_BOOTS = 309;
const uint ITEM_DIAMOND_HELMET = 310;
const uint ITEM_DIAMOND_CHESTPLATE = 311;
const uint ITEM_DIAMOND_LEGGINGS = 312;
const uint ITEM_DIAMOND_BOOTS = 313;
const uint ITEM_GOLDEN_HELMET = 314;
const uint ITEM_GOLDEN_CHESTPLATE = 315;
const uint ITEM_GOLDEN_LEGGINGS = 316;
const uint ITEM_GOLDEN_BOOTS = 317;
const uint ITEM_FLINT = 318;
const uint ITEM_PORKCHOP = 319;
const uint ITEM_COOKED_PORKCHOP = 320;
const uint ITEM_PAINTING = 321;
const uint ITEM_GOLDEN_APPLE = 322;
const uint ITEM_SIGN = 323;
const uint ITEM_WOODEN_DOOR_ITEM = 324;
const uint ITEM_BUCKET = 325;
const uint ITEM_WATER_BUCKET = 326;
const uint ITEM_LAVA_BUCKET = 327;
const uint ITEM_MINECART = 328;
const uint ITEM_SADDLE = 329;
const uint ITEM_IRON_DOOR_ITEM = 330;
const uint ITEM_REDSTONE = 331;
const uint ITEM_SNOWBALL = 332;
const uint ITEM_BOAT = 333;
const uint ITEM_LEATHER = 334;
const uint ITEM_MILK_BUCKET = 335;
const uint ITEM_BRICK = 336;
const uint ITEM_CLAY_BALL = 337;
const uint ITEM_SUGAR_CANE_ITEM = 338;
const uint ITEM_PAPER = 339;
const uint ITEM_BOOK = 340;
const uint ITEM_SLIME_BALL = 341;
const uint ITEM_CHEST_MINECART = 342;
const uint ITEM_FURNACE_MINECART = 343;
const uint ITEM_EGG = 344;
const uint ITEM_COMPASS = 345;
const uint ITEM_FISHING_ROD = 346;
const uint ITEM_CLOCK = 347;
const uint ITEM_GLOWSTONE_DUST = 348;
const uint ITEM_FISH = 349;
const uint ITEM_COOKED_FISH = 350;
const uint ITEM_DYE = 351;
const uint ITEM_BONE = 352;
const uint ITEM_SUGAR = 353;
const uint ITEM_CAKE = 354;
const uint ITEM_BED = 355;
const uint ITEM_REPEATER = 356;
const uint ITEM_COOKIE = 357;
const uint ITEM_FILLED_MAP = 358;
const uint ITEM_SHEARS = 359;
const uint ITEM_MELON = 360;
const uint ITEM_PUMPKIN_SEEDS = 361;
const uint ITEM_MELON_SEEDS = 362;
const uint ITEM_BEEF = 363;
const uint ITEM_COOKED_BEEF = 364;
const uint ITEM_CHICKEN = 365;
const uint ITEM_COOKED_CHICKEN = 366;
const uint ITEM_ROTTEN_FLESH = 367;
const uint ITEM_ENDER_PEARL = 368;
const uint ITEM_BLAZE_ROD = 369;
const uint ITEM_GHAST_TEAR = 370;
const uint ITEM_GOLD_NUGGET = 371;
const uint ITEM_NETHER_WART = 372;
const uint ITEM_POTION = 373;
const uint ITEM_GLASS_BOTTLE = 374;
const uint ITEM_SPIDER_EYE = 375;
const uint ITEM_FERMENTED_SPIDER_EYE = 376;
const uint ITEM_BLAZE_POWDER = 377;
const uint ITEM_MAGMA_CREAM = 378;
const uint ITEM_BREWING_STAND = 379;
const uint ITEM_CAULDRON = 380;
const uint ITEM_ENDER_EYE = 381;
const uint ITEM_SPECKLED_MELON = 382;
const uint ITEM_SPAWN_EGG = 383;
const uint ITEM_EXP_BOTTLE = 384;
const uint ITEM_FIRE_CHARGE = 385;
const uint ITEM_WRITABLE_BOOK = 386;
const uint ITEM_WRITTEN_BOOK = 387;
const uint ITEM_EMERALD = 388;
const uint ITEM_ITEM_FRAME = 389;
const uint ITEM_FLOWER_POT = 390;
const uint ITEM_CARROT = 391;
const uint ITEM_POTATO = 392;
const uint ITEM_BAKED_POTATO = 393;
const uint ITEM_POISONOUS_POTATO = 394;
const uint ITEM_EMPTY_MAP = 395;
const uint ITEM_GOLDEN_CARROT = 396;
const uint ITEM_SKULL = 397;
const uint ITEM_CARROT_ON_A_STICK = 398;
const uint ITEM_NETHER_STAR = 399;
const uint ITEM_PUMPKIN_PIE = 400;
const uint ITEM_FIREWORKS = 401;
const uint ITEM_FIREWORK_CHARGE = 402;
const uint ITEM_ENCHANTED_BOOK = 403;
const uint ITEM_COMPARATOR = 404;
const uint ITEM_NETHER_BRICK_ITEM = 405;
const uint ITEM_QUARTZ = 406;
const uint ITEM_TNT_MINECART = 407;
const uint ITEM_HOPPER_MINECART = 408;
const uint ITEM_PRISMARINE_SHARD = 409;
const uint ITEM_PRISMARINE_CRYSTALS = 410;
const uint ITEM_RABBIT = 411;
const uint ITEM_COOKED_RABBIT = 412;
const uint ITEM_RABBIT_STEW = 413;
const uint ITEM_RABBIT_FOOT = 414;
const uint ITEM_RABBIT_HIDE = 415;
const uint ITEM_ARMOR_STAND = 416;
const uint ITEM_IRON_HORSE_ARMOR = 417;
const uint ITEM_GOLDEN_HORSE_ARMOR = 418;
const uint ITEM_DIAMOND_HORSE_ARMOR = 419;
const uint ITEM_LEAD = 420;
const uint ITEM_NAME_TAG = 421;
const uint ITEM_COMMAND_BLOCK_MINECART = 422;
const uint ITEM_MUTTON = 423;
const uint ITEM_COOKED_MUTTON = 424;
const uint ITEM_BANNER = 425;

// ============================================================================
// Recipe Structure
// ============================================================================

struct Recipe {
    uint pattern[9];    // Item IDs in 3x3 grid (row-major: 0-2 top, 3-5 middle, 6-8 bottom)
    uint result_id;     // Output item ID
    uint result_count;  // Output stack count
    bool shapeless;     // If true, ingredient positions don't matter
};

// ============================================================================
// Speedrun Essential Recipes
// ============================================================================

// Total recipe count
const uint RECIPE_COUNT = 47;

// Recipe definitions (speedrun priorities)
const Recipe RECIPES[RECIPE_COUNT] = Recipe[](
    // -------------------------------------------------------------------------
    // Basic 2x2 Recipes (can use any corner of 3x3)
    // -------------------------------------------------------------------------

    // Log -> Planks (4) - Most important recipe in speedruns
    Recipe(
        uint[9](ITEM_LOG, 0, 0, 0, 0, 0, 0, 0, 0),
        ITEM_PLANKS, 4, false
    ),

    // Planks -> Sticks (4)
    Recipe(
        uint[9](ITEM_PLANKS, 0, 0, ITEM_PLANKS, 0, 0, 0, 0, 0),
        ITEM_STICK, 4, false
    ),

    // Crafting Table (2x2)
    Recipe(
        uint[9](ITEM_PLANKS, ITEM_PLANKS, 0, ITEM_PLANKS, ITEM_PLANKS, 0, 0, 0, 0),
        ITEM_CRAFTING_TABLE, 1, false
    ),

    // -------------------------------------------------------------------------
    // Wooden Tools (Tier 1 - first 30 seconds)
    // -------------------------------------------------------------------------

    // Wooden Pickaxe
    Recipe(
        uint[9](ITEM_PLANKS, ITEM_PLANKS, ITEM_PLANKS, 0, ITEM_STICK, 0, 0, ITEM_STICK, 0),
        ITEM_WOODEN_PICKAXE, 1, false
    ),

    // Wooden Axe
    Recipe(
        uint[9](ITEM_PLANKS, ITEM_PLANKS, 0, ITEM_PLANKS, ITEM_STICK, 0, 0, ITEM_STICK, 0),
        ITEM_WOODEN_AXE, 1, false
    ),

    // Wooden Sword
    Recipe(
        uint[9](ITEM_PLANKS, 0, 0, ITEM_PLANKS, 0, 0, ITEM_STICK, 0, 0),
        ITEM_WOODEN_SWORD, 1, false
    ),

    // Wooden Shovel
    Recipe(
        uint[9](ITEM_PLANKS, 0, 0, ITEM_STICK, 0, 0, ITEM_STICK, 0, 0),
        ITEM_WOODEN_SHOVEL, 1, false
    ),

    // -------------------------------------------------------------------------
    // Stone Tools (Tier 2 - first 2 minutes)
    // -------------------------------------------------------------------------

    // Stone Pickaxe (critical for iron)
    Recipe(
        uint[9](ITEM_COBBLESTONE, ITEM_COBBLESTONE, ITEM_COBBLESTONE, 0, ITEM_STICK, 0, 0, ITEM_STICK, 0),
        ITEM_STONE_PICKAXE, 1, false
    ),

    // Stone Axe
    Recipe(
        uint[9](ITEM_COBBLESTONE, ITEM_COBBLESTONE, 0, ITEM_COBBLESTONE, ITEM_STICK, 0, 0, ITEM_STICK, 0),
        ITEM_STONE_AXE, 1, false
    ),

    // Stone Sword
    Recipe(
        uint[9](ITEM_COBBLESTONE, 0, 0, ITEM_COBBLESTONE, 0, 0, ITEM_STICK, 0, 0),
        ITEM_STONE_SWORD, 1, false
    ),

    // Stone Shovel
    Recipe(
        uint[9](ITEM_COBBLESTONE, 0, 0, ITEM_STICK, 0, 0, ITEM_STICK, 0, 0),
        ITEM_STONE_SHOVEL, 1, false
    ),

    // -------------------------------------------------------------------------
    // Iron Tools (Tier 3 - critical for diamond)
    // -------------------------------------------------------------------------

    // Iron Pickaxe (required for diamond/obsidian)
    Recipe(
        uint[9](ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, 0, ITEM_STICK, 0, 0, ITEM_STICK, 0),
        ITEM_IRON_PICKAXE, 1, false
    ),

    // Iron Axe
    Recipe(
        uint[9](ITEM_IRON_INGOT, ITEM_IRON_INGOT, 0, ITEM_IRON_INGOT, ITEM_STICK, 0, 0, ITEM_STICK, 0),
        ITEM_IRON_AXE, 1, false
    ),

    // Iron Sword
    Recipe(
        uint[9](ITEM_IRON_INGOT, 0, 0, ITEM_IRON_INGOT, 0, 0, ITEM_STICK, 0, 0),
        ITEM_IRON_SWORD, 1, false
    ),

    // Iron Shovel
    Recipe(
        uint[9](ITEM_IRON_INGOT, 0, 0, ITEM_STICK, 0, 0, ITEM_STICK, 0, 0),
        ITEM_IRON_SHOVEL, 1, false
    ),

    // -------------------------------------------------------------------------
    // Diamond Tools (Tier 4 - endgame)
    // -------------------------------------------------------------------------

    // Diamond Pickaxe
    Recipe(
        uint[9](ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, 0, ITEM_STICK, 0, 0, ITEM_STICK, 0),
        ITEM_DIAMOND_PICKAXE, 1, false
    ),

    // Diamond Axe
    Recipe(
        uint[9](ITEM_DIAMOND, ITEM_DIAMOND, 0, ITEM_DIAMOND, ITEM_STICK, 0, 0, ITEM_STICK, 0),
        ITEM_DIAMOND_AXE, 1, false
    ),

    // Diamond Sword (best weapon for dragon)
    Recipe(
        uint[9](ITEM_DIAMOND, 0, 0, ITEM_DIAMOND, 0, 0, ITEM_STICK, 0, 0),
        ITEM_DIAMOND_SWORD, 1, false
    ),

    // -------------------------------------------------------------------------
    // Essential Utility Items
    // -------------------------------------------------------------------------

    // Furnace (smelting iron)
    Recipe(
        uint[9](ITEM_COBBLESTONE, ITEM_COBBLESTONE, ITEM_COBBLESTONE, ITEM_COBBLESTONE, 0, ITEM_COBBLESTONE, ITEM_COBBLESTONE, ITEM_COBBLESTONE, ITEM_COBBLESTONE),
        ITEM_FURNACE, 1, false
    ),

    // Chest
    Recipe(
        uint[9](ITEM_PLANKS, ITEM_PLANKS, ITEM_PLANKS, ITEM_PLANKS, 0, ITEM_PLANKS, ITEM_PLANKS, ITEM_PLANKS, ITEM_PLANKS),
        ITEM_CHEST, 1, false
    ),

    // Bucket (for lava/water - nether portal)
    Recipe(
        uint[9](ITEM_IRON_INGOT, 0, ITEM_IRON_INGOT, 0, ITEM_IRON_INGOT, 0, 0, 0, 0),
        ITEM_BUCKET, 1, false
    ),

    // Flint and Steel (nether portal ignition)
    Recipe(
        uint[9](ITEM_IRON_INGOT, 0, 0, 0, ITEM_FLINT, 0, 0, 0, 0),
        ITEM_FLINT_AND_STEEL, 1, true  // Shapeless
    ),

    // -------------------------------------------------------------------------
    // Nether/End Items
    // -------------------------------------------------------------------------

    // Blaze Powder (from blaze rod - not craftable but needed for reference)
    // Actually blaze powder IS craftable from blaze rod in crafting grid
    Recipe(
        uint[9](ITEM_BLAZE_ROD, 0, 0, 0, 0, 0, 0, 0, 0),
        ITEM_BLAZE_POWDER, 2, false
    ),

    // Eye of Ender (critical for stronghold)
    Recipe(
        uint[9](ITEM_ENDER_PEARL, 0, 0, ITEM_BLAZE_POWDER, 0, 0, 0, 0, 0),
        ITEM_ENDER_EYE, 1, true  // Shapeless
    ),

    // Ender Chest
    Recipe(
        uint[9](ITEM_OBSIDIAN, ITEM_OBSIDIAN, ITEM_OBSIDIAN, ITEM_OBSIDIAN, ITEM_ENDER_EYE, ITEM_OBSIDIAN, ITEM_OBSIDIAN, ITEM_OBSIDIAN, ITEM_OBSIDIAN),
        ITEM_ENDER_CHEST, 1, false
    ),

    // -------------------------------------------------------------------------
    // Combat Items
    // -------------------------------------------------------------------------

    // Bow
    Recipe(
        uint[9](0, ITEM_STICK, ITEM_STRING, ITEM_STICK, 0, ITEM_STRING, 0, ITEM_STICK, ITEM_STRING),
        ITEM_BOW, 1, false
    ),

    // Arrow (4)
    Recipe(
        uint[9](ITEM_FLINT, 0, 0, ITEM_STICK, 0, 0, ITEM_FEATHER, 0, 0),
        ITEM_ARROW, 4, false
    ),

    // -------------------------------------------------------------------------
    // Iron Armor (useful for dragon fight)
    // -------------------------------------------------------------------------

    // Iron Helmet
    Recipe(
        uint[9](ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, 0, ITEM_IRON_INGOT, 0, 0, 0),
        ITEM_IRON_HELMET, 1, false
    ),

    // Iron Chestplate
    Recipe(
        uint[9](ITEM_IRON_INGOT, 0, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT),
        ITEM_IRON_CHESTPLATE, 1, false
    ),

    // Iron Leggings
    Recipe(
        uint[9](ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, 0, ITEM_IRON_INGOT, ITEM_IRON_INGOT, 0, ITEM_IRON_INGOT),
        ITEM_IRON_LEGGINGS, 1, false
    ),

    // Iron Boots
    Recipe(
        uint[9](ITEM_IRON_INGOT, 0, ITEM_IRON_INGOT, ITEM_IRON_INGOT, 0, ITEM_IRON_INGOT, 0, 0, 0),
        ITEM_IRON_BOOTS, 1, false
    ),

    // -------------------------------------------------------------------------
    // Diamond Armor
    // -------------------------------------------------------------------------

    // Diamond Helmet
    Recipe(
        uint[9](ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, 0, ITEM_DIAMOND, 0, 0, 0),
        ITEM_DIAMOND_HELMET, 1, false
    ),

    // Diamond Chestplate
    Recipe(
        uint[9](ITEM_DIAMOND, 0, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND),
        ITEM_DIAMOND_CHESTPLATE, 1, false
    ),

    // Diamond Leggings
    Recipe(
        uint[9](ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, 0, ITEM_DIAMOND, ITEM_DIAMOND, 0, ITEM_DIAMOND),
        ITEM_DIAMOND_LEGGINGS, 1, false
    ),

    // Diamond Boots
    Recipe(
        uint[9](ITEM_DIAMOND, 0, ITEM_DIAMOND, ITEM_DIAMOND, 0, ITEM_DIAMOND, 0, 0, 0),
        ITEM_DIAMOND_BOOTS, 1, false
    ),

    // -------------------------------------------------------------------------
    // Food & Misc
    // -------------------------------------------------------------------------

    // Bread (3 wheat)
    Recipe(
        uint[9](ITEM_WHEAT, ITEM_WHEAT, ITEM_WHEAT, 0, 0, 0, 0, 0, 0),
        ITEM_BREAD, 1, false
    ),

    // Torch (4)
    Recipe(
        uint[9](ITEM_COAL, 0, 0, ITEM_STICK, 0, 0, 0, 0, 0),
        ITEM_TORCH, 4, false
    ),

    // Ladder (3)
    Recipe(
        uint[9](ITEM_STICK, 0, ITEM_STICK, ITEM_STICK, ITEM_STICK, ITEM_STICK, ITEM_STICK, 0, ITEM_STICK),
        ITEM_LADDER, 3, false
    ),

    // Bed
    Recipe(
        uint[9](ITEM_WOOL, ITEM_WOOL, ITEM_WOOL, ITEM_PLANKS, ITEM_PLANKS, ITEM_PLANKS, 0, 0, 0),
        ITEM_BED, 1, false
    ),

    // Boat (for ocean crossing)
    Recipe(
        uint[9](ITEM_PLANKS, 0, ITEM_PLANKS, ITEM_PLANKS, ITEM_PLANKS, ITEM_PLANKS, 0, 0, 0),
        ITEM_BOAT, 1, false
    ),

    // -------------------------------------------------------------------------
    // Block Conversions
    // -------------------------------------------------------------------------

    // Iron Block (storage)
    Recipe(
        uint[9](ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT, ITEM_IRON_INGOT),
        ITEM_IRON_BLOCK, 1, false
    ),

    // Iron Ingots from Block (9)
    Recipe(
        uint[9](ITEM_IRON_BLOCK, 0, 0, 0, 0, 0, 0, 0, 0),
        ITEM_IRON_INGOT, 9, false
    ),

    // Gold Block
    Recipe(
        uint[9](ITEM_GOLD_INGOT, ITEM_GOLD_INGOT, ITEM_GOLD_INGOT, ITEM_GOLD_INGOT, ITEM_GOLD_INGOT, ITEM_GOLD_INGOT, ITEM_GOLD_INGOT, ITEM_GOLD_INGOT, ITEM_GOLD_INGOT),
        ITEM_GOLD_BLOCK, 1, false
    ),

    // Diamond Block
    Recipe(
        uint[9](ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND, ITEM_DIAMOND),
        ITEM_DIAMOND_BLOCK, 1, false
    ),

    // -------------------------------------------------------------------------
    // Brewing/Potions (useful for dragon fight)
    // -------------------------------------------------------------------------

    // Brewing Stand
    Recipe(
        uint[9](0, ITEM_BLAZE_ROD, 0, ITEM_COBBLESTONE, ITEM_COBBLESTONE, ITEM_COBBLESTONE, 0, 0, 0),
        ITEM_BREWING_STAND, 1, false
    ),

    // Glass Bottle (3)
    Recipe(
        uint[9](ITEM_GLASS, 0, ITEM_GLASS, 0, ITEM_GLASS, 0, 0, 0, 0),
        ITEM_GLASS_BOTTLE, 3, false
    ),

    // Golden Apple (healing during dragon fight)
    Recipe(
        uint[9](ITEM_GOLD_INGOT, ITEM_GOLD_INGOT, ITEM_GOLD_INGOT, ITEM_GOLD_INGOT, ITEM_APPLE, ITEM_GOLD_INGOT, ITEM_GOLD_INGOT, ITEM_GOLD_INGOT, ITEM_GOLD_INGOT),
        ITEM_GOLDEN_APPLE, 1, false
    )
);

// ============================================================================
// Recipe Matching Functions
// ============================================================================

// Check if two patterns match exactly (shaped recipe)
bool patterns_match_exact(uint input[9], uint pattern[9]) {
    for (int i = 0; i < 9; i++) {
        if (input[i] != pattern[i]) {
            return false;
        }
    }
    return true;
}

// Check if pattern matches at any valid offset (for 2x2 recipes in 3x3 grid)
bool patterns_match_with_offset(uint input[9], uint pattern[9]) {
    // Find bounds of recipe pattern
    int min_x = 3, max_x = -1, min_y = 3, max_y = -1;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (pattern[y * 3 + x] != 0) {
                min_x = min(min_x, x);
                max_x = max(max_x, x);
                min_y = min(min_y, y);
                max_y = max(max_y, y);
            }
        }
    }

    // Recipe is empty
    if (max_x < 0) return false;

    int recipe_w = max_x - min_x + 1;
    int recipe_h = max_y - min_y + 1;

    // Find bounds of input pattern
    int in_min_x = 3, in_max_x = -1, in_min_y = 3, in_max_y = -1;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            if (input[y * 3 + x] != 0) {
                in_min_x = min(in_min_x, x);
                in_max_x = max(in_max_x, x);
                in_min_y = min(in_min_y, y);
                in_max_y = max(in_max_y, y);
            }
        }
    }

    // Input is empty - no match unless recipe is also empty
    if (in_max_x < 0) return false;

    int input_w = in_max_x - in_min_x + 1;
    int input_h = in_max_y - in_min_y + 1;

    // Dimensions must match
    if (input_w != recipe_w || input_h != recipe_h) return false;

    // Compare normalized patterns
    for (int y = 0; y < recipe_h; y++) {
        for (int x = 0; x < recipe_w; x++) {
            uint recipe_item = pattern[(min_y + y) * 3 + (min_x + x)];
            uint input_item = input[(in_min_y + y) * 3 + (in_min_x + x)];
            if (recipe_item != input_item) return false;
        }
    }

    return true;
}

// Check shapeless recipe match (ingredient counts, position-independent)
bool patterns_match_shapeless(uint input[9], uint pattern[9]) {
    // Count ingredients in pattern
    uint pattern_counts[512];  // Item ID -> count
    for (int i = 0; i < 512; i++) pattern_counts[i] = 0;

    uint pattern_total = 0;
    for (int i = 0; i < 9; i++) {
        if (pattern[i] != 0) {
            pattern_counts[pattern[i]]++;
            pattern_total++;
        }
    }

    // Count ingredients in input
    uint input_counts[512];
    for (int i = 0; i < 512; i++) input_counts[i] = 0;

    uint input_total = 0;
    for (int i = 0; i < 9; i++) {
        if (input[i] != 0) {
            input_counts[input[i]]++;
            input_total++;
        }
    }

    // Total item counts must match
    if (input_total != pattern_total) return false;

    // Each item count must match
    for (int i = 0; i < 512; i++) {
        if (pattern_counts[i] != input_counts[i]) return false;
    }

    return true;
}

// Main pattern matching entry point
bool matches_pattern(uint input[9], Recipe recipe) {
    if (recipe.shapeless) {
        return patterns_match_shapeless(input, recipe.pattern);
    } else {
        return patterns_match_with_offset(input, recipe.pattern);
    }
}

// Find matching recipe for given crafting grid input
// Returns recipe index, or -1 if no match
int find_matching_recipe(uint input[9]) {
    for (int i = 0; i < int(RECIPE_COUNT); i++) {
        if (matches_pattern(input, RECIPES[i])) {
            return i;
        }
    }
    return -1;
}

// Get recipe by index (with bounds check)
Recipe get_recipe(int index) {
    if (index < 0 || index >= int(RECIPE_COUNT)) {
        // Return empty recipe
        return Recipe(
            uint[9](0, 0, 0, 0, 0, 0, 0, 0, 0),
            0, 0, false
        );
    }
    return RECIPES[index];
}

// Check if a specific recipe can be crafted with given input
bool can_craft_recipe(uint input[9], int recipe_index) {
    if (recipe_index < 0 || recipe_index >= int(RECIPE_COUNT)) {
        return false;
    }
    return matches_pattern(input, RECIPES[recipe_index]);
}

// Find recipe by output item ID (returns first match)
int find_recipe_by_output(uint output_item_id) {
    for (int i = 0; i < int(RECIPE_COUNT); i++) {
        if (RECIPES[i].result_id == output_item_id) {
            return i;
        }
    }
    return -1;
}

#endif // CRAFTING_RECIPES_GLSL
