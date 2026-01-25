"""Block breaking verification for Minecraft block interaction subsystems.

Tests break time calculations for all tool/block combinations and validates
drop tables with fortune and silk touch enchantments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto


class ToolType(Enum):
    """Tool categories that affect block breaking speed."""

    NONE = auto()
    PICKAXE = auto()
    AXE = auto()
    SHOVEL = auto()
    HOE = auto()
    SWORD = auto()
    SHEARS = auto()


class ToolMaterial(Enum):
    """Tool material tiers with mining speed multipliers."""

    HAND = (1.0, 0)
    WOOD = (2.0, 0)
    STONE = (4.0, 1)
    IRON = (6.0, 2)
    DIAMOND = (8.0, 3)
    NETHERITE = (9.0, 4)
    GOLD = (12.0, 0)  # Fast but low harvest level

    def __init__(self, speed: float, harvest_level: int) -> None:
        self.speed = speed
        self.harvest_level = harvest_level


class BlockHarvestLevel(Enum):
    """Block harvest level requirements."""

    ANY = 0
    WOOD = 0
    STONE = 1
    IRON = 2
    DIAMOND = 3


@dataclass
class BlockProperties:
    """Properties that define block breaking behavior."""

    hardness: float
    blast_resistance: float
    preferred_tool: ToolType
    harvest_level: BlockHarvestLevel = BlockHarvestLevel.ANY
    requires_tool: bool = False  # If True, drops nothing without proper tool


@dataclass
class ToolProperties:
    """Properties of a mining tool."""

    tool_type: ToolType
    material: ToolMaterial
    efficiency_level: int = 0
    silk_touch: bool = False
    fortune_level: int = 0


@dataclass
class DropEntry:
    """A single drop from a block."""

    item_id: str
    min_count: int = 1
    max_count: int = 1
    chance: float = 1.0  # 0.0 to 1.0
    fortune_multiplier: float = 0.0  # Additional items per fortune level
    requires_silk_touch: bool = False
    silk_touch_override: str | None = None  # Different item with silk touch


@dataclass
class BlockDropTable:
    """Drop table for a block."""

    entries: list[DropEntry] = field(default_factory=list)
    experience_min: int = 0
    experience_max: int = 0


# Standard Minecraft blocks with accurate properties
BLOCK_REGISTRY: dict[str, BlockProperties] = {
    # Stone variants
    "stone": BlockProperties(1.5, 6.0, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    "cobblestone": BlockProperties(2.0, 6.0, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    "granite": BlockProperties(1.5, 6.0, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    "diorite": BlockProperties(1.5, 6.0, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    "andesite": BlockProperties(1.5, 6.0, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    "deepslate": BlockProperties(3.0, 6.0, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    "calcite": BlockProperties(0.75, 0.75, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    "tuff": BlockProperties(1.5, 6.0, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    "dripstone_block": BlockProperties(1.5, 1.0, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    # Ores
    "coal_ore": BlockProperties(3.0, 3.0, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    "deepslate_coal_ore": BlockProperties(4.5, 3.0, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    "iron_ore": BlockProperties(3.0, 3.0, ToolType.PICKAXE, BlockHarvestLevel.STONE, True),
    "deepslate_iron_ore": BlockProperties(
        4.5, 3.0, ToolType.PICKAXE, BlockHarvestLevel.STONE, True
    ),
    "copper_ore": BlockProperties(3.0, 3.0, ToolType.PICKAXE, BlockHarvestLevel.STONE, True),
    "deepslate_copper_ore": BlockProperties(
        4.5, 3.0, ToolType.PICKAXE, BlockHarvestLevel.STONE, True
    ),
    "gold_ore": BlockProperties(3.0, 3.0, ToolType.PICKAXE, BlockHarvestLevel.IRON, True),
    "deepslate_gold_ore": BlockProperties(4.5, 3.0, ToolType.PICKAXE, BlockHarvestLevel.IRON, True),
    "redstone_ore": BlockProperties(3.0, 3.0, ToolType.PICKAXE, BlockHarvestLevel.IRON, True),
    "deepslate_redstone_ore": BlockProperties(
        4.5, 3.0, ToolType.PICKAXE, BlockHarvestLevel.IRON, True
    ),
    "lapis_ore": BlockProperties(3.0, 3.0, ToolType.PICKAXE, BlockHarvestLevel.STONE, True),
    "deepslate_lapis_ore": BlockProperties(
        4.5, 3.0, ToolType.PICKAXE, BlockHarvestLevel.STONE, True
    ),
    "diamond_ore": BlockProperties(3.0, 3.0, ToolType.PICKAXE, BlockHarvestLevel.IRON, True),
    "deepslate_diamond_ore": BlockProperties(
        4.5, 3.0, ToolType.PICKAXE, BlockHarvestLevel.IRON, True
    ),
    "emerald_ore": BlockProperties(3.0, 3.0, ToolType.PICKAXE, BlockHarvestLevel.IRON, True),
    "deepslate_emerald_ore": BlockProperties(
        4.5, 3.0, ToolType.PICKAXE, BlockHarvestLevel.IRON, True
    ),
    "nether_gold_ore": BlockProperties(3.0, 3.0, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    "nether_quartz_ore": BlockProperties(3.0, 3.0, ToolType.PICKAXE, BlockHarvestLevel.WOOD, True),
    "ancient_debris": BlockProperties(
        30.0, 1200.0, ToolType.PICKAXE, BlockHarvestLevel.DIAMOND, True
    ),
    # Wood
    "oak_log": BlockProperties(2.0, 2.0, ToolType.AXE),
    "spruce_log": BlockProperties(2.0, 2.0, ToolType.AXE),
    "birch_log": BlockProperties(2.0, 2.0, ToolType.AXE),
    "jungle_log": BlockProperties(2.0, 2.0, ToolType.AXE),
    "acacia_log": BlockProperties(2.0, 2.0, ToolType.AXE),
    "dark_oak_log": BlockProperties(2.0, 2.0, ToolType.AXE),
    "mangrove_log": BlockProperties(2.0, 2.0, ToolType.AXE),
    "cherry_log": BlockProperties(2.0, 2.0, ToolType.AXE),
    "crimson_stem": BlockProperties(2.0, 2.0, ToolType.AXE),
    "warped_stem": BlockProperties(2.0, 2.0, ToolType.AXE),
    "oak_planks": BlockProperties(2.0, 3.0, ToolType.AXE),
    # Dirt variants
    "dirt": BlockProperties(0.5, 0.5, ToolType.SHOVEL),
    "grass_block": BlockProperties(0.6, 0.6, ToolType.SHOVEL),
    "podzol": BlockProperties(0.5, 0.5, ToolType.SHOVEL),
    "mycelium": BlockProperties(0.6, 0.6, ToolType.SHOVEL),
    "coarse_dirt": BlockProperties(0.5, 0.5, ToolType.SHOVEL),
    "rooted_dirt": BlockProperties(0.5, 0.5, ToolType.SHOVEL),
    "mud": BlockProperties(0.5, 0.5, ToolType.SHOVEL),
    "clay": BlockProperties(0.6, 0.6, ToolType.SHOVEL),
    "gravel": BlockProperties(0.6, 0.6, ToolType.SHOVEL),
    "sand": BlockProperties(0.5, 0.5, ToolType.SHOVEL),
    "red_sand": BlockProperties(0.5, 0.5, ToolType.SHOVEL),
    "soul_sand": BlockProperties(0.5, 0.5, ToolType.SHOVEL),
    "soul_soil": BlockProperties(0.5, 0.5, ToolType.SHOVEL),
    # Hard blocks
    "obsidian": BlockProperties(50.0, 1200.0, ToolType.PICKAXE, BlockHarvestLevel.DIAMOND, True),
    "crying_obsidian": BlockProperties(
        50.0, 1200.0, ToolType.PICKAXE, BlockHarvestLevel.DIAMOND, True
    ),
    "netherite_block": BlockProperties(
        50.0, 1200.0, ToolType.PICKAXE, BlockHarvestLevel.DIAMOND, True
    ),
    "respawn_anchor": BlockProperties(
        50.0, 1200.0, ToolType.PICKAXE, BlockHarvestLevel.DIAMOND, True
    ),
    # Instant break
    "torch": BlockProperties(0.0, 0.0, ToolType.NONE),
    "redstone_torch": BlockProperties(0.0, 0.0, ToolType.NONE),
    "flower": BlockProperties(0.0, 0.0, ToolType.NONE),
    "tall_grass": BlockProperties(0.0, 0.0, ToolType.NONE),
    "fire": BlockProperties(0.0, 0.0, ToolType.NONE),
    # Leaves
    "oak_leaves": BlockProperties(0.2, 0.2, ToolType.HOE),
    "spruce_leaves": BlockProperties(0.2, 0.2, ToolType.HOE),
    "birch_leaves": BlockProperties(0.2, 0.2, ToolType.HOE),
    "jungle_leaves": BlockProperties(0.2, 0.2, ToolType.HOE),
    "acacia_leaves": BlockProperties(0.2, 0.2, ToolType.HOE),
    "dark_oak_leaves": BlockProperties(0.2, 0.2, ToolType.HOE),
    "azalea_leaves": BlockProperties(0.2, 0.2, ToolType.HOE),
    "flowering_azalea_leaves": BlockProperties(0.2, 0.2, ToolType.HOE),
    "mangrove_leaves": BlockProperties(0.2, 0.2, ToolType.HOE),
    "cherry_leaves": BlockProperties(0.2, 0.2, ToolType.HOE),
    # Glass
    "glass": BlockProperties(0.3, 0.3, ToolType.NONE),
    "glass_pane": BlockProperties(0.3, 0.3, ToolType.NONE),
    "tinted_glass": BlockProperties(0.3, 0.3, ToolType.NONE),
    # Special blocks
    "bedrock": BlockProperties(-1.0, 3600000.0, ToolType.NONE),  # Unbreakable
    "end_portal_frame": BlockProperties(-1.0, 3600000.0, ToolType.NONE),  # Unbreakable
    "barrier": BlockProperties(-1.0, 3600000.8, ToolType.NONE),  # Unbreakable
    "command_block": BlockProperties(-1.0, 3600000.0, ToolType.NONE),  # Unbreakable
    # Wool
    "white_wool": BlockProperties(0.8, 0.8, ToolType.SHEARS),
    "orange_wool": BlockProperties(0.8, 0.8, ToolType.SHEARS),
    "magenta_wool": BlockProperties(0.8, 0.8, ToolType.SHEARS),
    # Crops
    "wheat": BlockProperties(0.0, 0.0, ToolType.NONE),
    "carrots": BlockProperties(0.0, 0.0, ToolType.NONE),
    "potatoes": BlockProperties(0.0, 0.0, ToolType.NONE),
    "beetroots": BlockProperties(0.0, 0.0, ToolType.NONE),
    "nether_wart": BlockProperties(0.0, 0.0, ToolType.NONE),
    # Cobweb
    "cobweb": BlockProperties(4.0, 4.0, ToolType.SWORD),
}

# Drop tables with fortune/silk touch behavior
DROP_TABLES: dict[str, BlockDropTable] = {
    "stone": BlockDropTable(
        [
            DropEntry("cobblestone", silk_touch_override="stone"),
        ]
    ),
    "coal_ore": BlockDropTable(
        [DropEntry("coal", fortune_multiplier=1.0, silk_touch_override="coal_ore")],
        experience_min=0,
        experience_max=2,
    ),
    "deepslate_coal_ore": BlockDropTable(
        [DropEntry("coal", fortune_multiplier=1.0, silk_touch_override="deepslate_coal_ore")],
        experience_min=0,
        experience_max=2,
    ),
    "iron_ore": BlockDropTable(
        [
            DropEntry("raw_iron", silk_touch_override="iron_ore"),
        ]
    ),
    "deepslate_iron_ore": BlockDropTable(
        [
            DropEntry("raw_iron", silk_touch_override="deepslate_iron_ore"),
        ]
    ),
    "copper_ore": BlockDropTable(
        [
            DropEntry(
                "raw_copper",
                min_count=2,
                max_count=5,
                fortune_multiplier=1.0,
                silk_touch_override="copper_ore",
            ),
        ]
    ),
    "deepslate_copper_ore": BlockDropTable(
        [
            DropEntry(
                "raw_copper",
                min_count=2,
                max_count=5,
                fortune_multiplier=1.0,
                silk_touch_override="deepslate_copper_ore",
            ),
        ]
    ),
    "gold_ore": BlockDropTable(
        [
            DropEntry("raw_gold", silk_touch_override="gold_ore"),
        ]
    ),
    "deepslate_gold_ore": BlockDropTable(
        [
            DropEntry("raw_gold", silk_touch_override="deepslate_gold_ore"),
        ]
    ),
    "nether_gold_ore": BlockDropTable(
        [
            DropEntry(
                "gold_nugget",
                min_count=2,
                max_count=6,
                fortune_multiplier=1.0,
                silk_touch_override="nether_gold_ore",
            )
        ],
        experience_min=0,
        experience_max=1,
    ),
    "redstone_ore": BlockDropTable(
        [
            DropEntry(
                "redstone",
                min_count=4,
                max_count=5,
                fortune_multiplier=1.0,
                silk_touch_override="redstone_ore",
            )
        ],
        experience_min=1,
        experience_max=5,
    ),
    "deepslate_redstone_ore": BlockDropTable(
        [
            DropEntry(
                "redstone",
                min_count=4,
                max_count=5,
                fortune_multiplier=1.0,
                silk_touch_override="deepslate_redstone_ore",
            )
        ],
        experience_min=1,
        experience_max=5,
    ),
    "lapis_ore": BlockDropTable(
        [
            DropEntry(
                "lapis_lazuli",
                min_count=4,
                max_count=9,
                fortune_multiplier=1.0,
                silk_touch_override="lapis_ore",
            )
        ],
        experience_min=2,
        experience_max=5,
    ),
    "deepslate_lapis_ore": BlockDropTable(
        [
            DropEntry(
                "lapis_lazuli",
                min_count=4,
                max_count=9,
                fortune_multiplier=1.0,
                silk_touch_override="deepslate_lapis_ore",
            )
        ],
        experience_min=2,
        experience_max=5,
    ),
    "diamond_ore": BlockDropTable(
        [DropEntry("diamond", fortune_multiplier=1.0, silk_touch_override="diamond_ore")],
        experience_min=3,
        experience_max=7,
    ),
    "deepslate_diamond_ore": BlockDropTable(
        [DropEntry("diamond", fortune_multiplier=1.0, silk_touch_override="deepslate_diamond_ore")],
        experience_min=3,
        experience_max=7,
    ),
    "emerald_ore": BlockDropTable(
        [DropEntry("emerald", fortune_multiplier=1.0, silk_touch_override="emerald_ore")],
        experience_min=3,
        experience_max=7,
    ),
    "deepslate_emerald_ore": BlockDropTable(
        [DropEntry("emerald", fortune_multiplier=1.0, silk_touch_override="deepslate_emerald_ore")],
        experience_min=3,
        experience_max=7,
    ),
    "nether_quartz_ore": BlockDropTable(
        [DropEntry("quartz", fortune_multiplier=1.0, silk_touch_override="nether_quartz_ore")],
        experience_min=2,
        experience_max=5,
    ),
    "ancient_debris": BlockDropTable(
        [
            DropEntry("ancient_debris"),  # Silk touch doesn't change drops
        ]
    ),
    "gravel": BlockDropTable(
        [
            DropEntry("gravel", chance=0.9, silk_touch_override="gravel"),
            DropEntry("flint", chance=0.1, fortune_multiplier=0.04),  # 10% + 4% per fortune
        ]
    ),
    "grass_block": BlockDropTable(
        [
            DropEntry("dirt", silk_touch_override="grass_block"),
        ]
    ),
    "podzol": BlockDropTable(
        [
            DropEntry("dirt", silk_touch_override="podzol"),
        ]
    ),
    "mycelium": BlockDropTable(
        [
            DropEntry("dirt", silk_touch_override="mycelium"),
        ]
    ),
    "glass": BlockDropTable(
        [
            DropEntry("glass", requires_silk_touch=True),
        ]
    ),
    "glass_pane": BlockDropTable(
        [
            DropEntry("glass_pane", requires_silk_touch=True),
        ]
    ),
    "oak_leaves": BlockDropTable(
        [
            DropEntry("oak_sapling", chance=0.05, fortune_multiplier=0.0125),
            DropEntry("stick", min_count=1, max_count=2, chance=0.02),
            DropEntry("apple", chance=0.005, fortune_multiplier=0.001),
            DropEntry("oak_leaves", requires_silk_touch=True),
        ]
    ),
    "white_wool": BlockDropTable(
        [
            DropEntry("white_wool"),  # Always drops
        ]
    ),
    "cobweb": BlockDropTable(
        [
            DropEntry("string", silk_touch_override="cobweb"),
        ]
    ),
    "wheat": BlockDropTable(
        [
            DropEntry("wheat", min_count=0, max_count=1),  # Only if mature
            DropEntry("wheat_seeds", min_count=0, max_count=3, fortune_multiplier=1.0),
        ]
    ),
    "clay": BlockDropTable(
        [
            DropEntry("clay_ball", min_count=4, max_count=4, silk_touch_override="clay"),
        ]
    ),
}


def calculate_break_time(
    block_id: str,
    tool: ToolProperties | None = None,
    in_water: bool = False,
    on_ground: bool = True,
    haste_level: int = 0,
    mining_fatigue_level: int = 0,
) -> float:
    """Calculate the time to break a block in seconds.

    Uses the Minecraft break time formula:
    1. Base time = hardness * 1.5 (or 5.0 if tool is required but wrong)
    2. Apply tool speed multiplier if using correct tool
    3. Apply efficiency enchantment
    4. Apply haste/mining fatigue effects
    5. Apply in_water and on_ground penalties

    Args:
        block_id: The block identifier
        tool: Tool properties (None for bare hand)
        in_water: Whether the player is underwater without aqua affinity
        on_ground: Whether the player is on the ground
        haste_level: Haste effect level (0-255)
        mining_fatigue_level: Mining fatigue level (0-255)

    Returns:
        Break time in seconds, or math.inf for unbreakable blocks
    """
    if block_id not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block: {block_id}")

    block = BLOCK_REGISTRY[block_id]

    # Unbreakable blocks
    if block.hardness < 0:
        return math.inf

    # Instant break blocks
    if block.hardness == 0:
        return 0.05  # 1 game tick minimum

    # Determine if we can harvest the block
    can_harvest = True
    speed_multiplier = 1.0

    if tool is None:
        tool = ToolProperties(ToolType.NONE, ToolMaterial.HAND)

    # Check tool requirements
    if block.requires_tool:
        if (
            tool.tool_type != block.preferred_tool
            or tool.material.harvest_level < block.harvest_level.value
        ):
            can_harvest = False

    # Calculate speed multiplier
    if tool.tool_type == block.preferred_tool:
        speed_multiplier = tool.material.speed

        # Efficiency enchantment: adds level^2 + 1
        if tool.efficiency_level > 0:
            speed_multiplier += tool.efficiency_level**2 + 1

    # Special case: shears on wool, leaves, cobweb
    if tool.tool_type == ToolType.SHEARS:
        if block_id in ("cobweb",):
            speed_multiplier = 15.0
        elif "wool" in block_id:
            speed_multiplier = 5.0
        elif "leaves" in block_id:
            speed_multiplier = 15.0

    # Special case: sword on cobweb
    if tool.tool_type == ToolType.SWORD and block_id == "cobweb":
        speed_multiplier = 15.0

    # Apply haste
    if haste_level > 0:
        speed_multiplier *= 1.0 + 0.2 * haste_level

    # Apply mining fatigue
    if mining_fatigue_level > 0:
        if mining_fatigue_level >= 4:
            speed_multiplier *= 0.00027  # Essentially frozen
        else:
            speed_multiplier *= 0.3**mining_fatigue_level

    # Calculate damage per tick
    damage = speed_multiplier / block.hardness

    if can_harvest:
        damage /= 30.0
    else:
        damage /= 100.0

    # Apply penalties
    if in_water:
        damage /= 5.0
    if not on_ground:
        damage /= 5.0

    # Calculate ticks to break
    if damage >= 1.0:
        return 0.05  # Instant break (1 tick)

    ticks = math.ceil(1.0 / damage)
    return ticks * 0.05  # Convert to seconds


def get_drops(
    block_id: str,
    tool: ToolProperties | None = None,
) -> list[tuple[str, int, int]]:
    """Get the drops for breaking a block.

    Args:
        block_id: The block identifier
        tool: Tool properties (None for bare hand)

    Returns:
        List of (item_id, min_count, max_count) tuples
    """
    if block_id not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block: {block_id}")

    block = BLOCK_REGISTRY[block_id]

    # Check if we can harvest
    if tool is None:
        tool = ToolProperties(ToolType.NONE, ToolMaterial.HAND)

    can_harvest = True
    if block.requires_tool:
        if (
            tool.tool_type != block.preferred_tool
            or tool.material.harvest_level < block.harvest_level.value
        ):
            can_harvest = False

    if not can_harvest:
        return []

    # Get drop table
    if block_id not in DROP_TABLES:
        # Default: drop itself
        return [(block_id, 1, 1)]

    table = DROP_TABLES[block_id]
    drops: list[tuple[str, int, int]] = []

    for entry in table.entries:
        # Check silk touch override
        if tool.silk_touch and entry.silk_touch_override:
            drops.append((entry.silk_touch_override, 1, 1))
            continue

        # Check if requires silk touch
        if entry.requires_silk_touch and not tool.silk_touch:
            continue

        # Calculate fortune bonus
        fortune_bonus = int(entry.fortune_multiplier * tool.fortune_level)
        min_count = entry.min_count
        max_count = entry.max_count + fortune_bonus

        drops.append((entry.item_id, min_count, max_count))

    return drops


class BlockBreakingVerifier:
    """Verifier for block breaking mechanics."""

    def __init__(self) -> None:
        self.test_results: list[tuple[str, bool, str]] = []

    def _add_result(self, name: str, passed: bool, message: str = "") -> None:
        self.test_results.append((name, passed, message))

    def verify_instant_break_blocks(self) -> bool:
        """Test that instant-break blocks break in one tick."""
        instant_blocks = [
            "torch",
            "redstone_torch",
            "flower",
            "tall_grass",
            "fire",
            "wheat",
            "carrots",
            "potatoes",
            "beetroots",
            "nether_wart",
        ]

        all_passed = True
        for block_id in instant_blocks:
            if block_id not in BLOCK_REGISTRY:
                continue
            time = calculate_break_time(block_id)
            expected = 0.05  # 1 tick
            passed = abs(time - expected) < 0.001
            self._add_result(
                f"instant_break_{block_id}",
                passed,
                f"Expected {expected}s, got {time}s",
            )
            all_passed = all_passed and passed

        return all_passed

    def verify_unbreakable_blocks(self) -> bool:
        """Test that unbreakable blocks return infinite break time."""
        unbreakable = ["bedrock", "end_portal_frame", "barrier", "command_block"]

        all_passed = True
        for block_id in unbreakable:
            if block_id not in BLOCK_REGISTRY:
                continue
            time = calculate_break_time(block_id)
            passed = math.isinf(time)
            self._add_result(
                f"unbreakable_{block_id}",
                passed,
                f"Expected inf, got {time}s",
            )
            all_passed = all_passed and passed

        return all_passed

    def verify_tool_speeds(self) -> bool:
        """Test that proper tools break blocks faster."""
        test_cases = [
            ("stone", ToolType.PICKAXE, ToolMaterial.WOOD),
            ("oak_log", ToolType.AXE, ToolMaterial.WOOD),
            ("dirt", ToolType.SHOVEL, ToolMaterial.WOOD),
            ("oak_leaves", ToolType.HOE, ToolMaterial.WOOD),
            ("white_wool", ToolType.SHEARS, ToolMaterial.HAND),
        ]

        all_passed = True
        for block_id, tool_type, material in test_cases:
            if block_id not in BLOCK_REGISTRY:
                continue

            hand_time = calculate_break_time(block_id)
            tool = ToolProperties(tool_type, material)
            tool_time = calculate_break_time(block_id, tool)

            passed = tool_time < hand_time
            self._add_result(
                f"tool_speed_{block_id}_{tool_type.name}",
                passed,
                f"Hand: {hand_time:.2f}s, Tool: {tool_time:.2f}s",
            )
            all_passed = all_passed and passed

        return all_passed

    def verify_harvest_levels(self) -> bool:
        """Test that blocks require proper harvest level."""
        test_cases = [
            ("iron_ore", ToolMaterial.WOOD, False),
            ("iron_ore", ToolMaterial.STONE, True),
            ("gold_ore", ToolMaterial.STONE, False),
            ("gold_ore", ToolMaterial.IRON, True),
            ("diamond_ore", ToolMaterial.STONE, False),
            ("diamond_ore", ToolMaterial.IRON, True),
            ("obsidian", ToolMaterial.IRON, False),
            ("obsidian", ToolMaterial.DIAMOND, True),
            ("ancient_debris", ToolMaterial.IRON, False),
            ("ancient_debris", ToolMaterial.DIAMOND, True),
        ]

        all_passed = True
        for block_id, material, expected_drops in test_cases:
            if block_id not in BLOCK_REGISTRY:
                continue

            tool = ToolProperties(ToolType.PICKAXE, material)
            drops = get_drops(block_id, tool)
            has_drops = len(drops) > 0

            passed = has_drops == expected_drops
            self._add_result(
                f"harvest_level_{block_id}_{material.name}",
                passed,
                f"Expected drops={expected_drops}, got {has_drops}",
            )
            all_passed = all_passed and passed

        return all_passed

    def verify_efficiency_enchantment(self) -> bool:
        """Test that efficiency enchantment speeds up mining."""
        all_passed = True

        for eff_level in [1, 2, 3, 4, 5]:
            base_tool = ToolProperties(ToolType.PICKAXE, ToolMaterial.DIAMOND)
            eff_tool = ToolProperties(
                ToolType.PICKAXE, ToolMaterial.DIAMOND, efficiency_level=eff_level
            )

            base_time = calculate_break_time("stone", base_tool)
            eff_time = calculate_break_time("stone", eff_tool)

            passed = eff_time < base_time
            self._add_result(
                f"efficiency_{eff_level}",
                passed,
                f"Base: {base_time:.3f}s, Eff{eff_level}: {eff_time:.3f}s",
            )
            all_passed = all_passed and passed

        return all_passed

    def verify_haste_effect(self) -> bool:
        """Test that haste effect speeds up mining."""
        all_passed = True

        tool = ToolProperties(ToolType.PICKAXE, ToolMaterial.DIAMOND)

        for haste_level in [1, 2]:
            base_time = calculate_break_time("stone", tool)
            haste_time = calculate_break_time("stone", tool, haste_level=haste_level)

            passed = haste_time < base_time
            self._add_result(
                f"haste_{haste_level}",
                passed,
                f"Base: {base_time:.3f}s, Haste{haste_level}: {haste_time:.3f}s",
            )
            all_passed = all_passed and passed

        return all_passed

    def verify_mining_fatigue(self) -> bool:
        """Test that mining fatigue slows mining."""
        all_passed = True

        tool = ToolProperties(ToolType.PICKAXE, ToolMaterial.DIAMOND)

        base_time = calculate_break_time("stone", tool)
        fatigue_time = calculate_break_time("stone", tool, mining_fatigue_level=1)

        passed = fatigue_time > base_time
        self._add_result(
            "mining_fatigue",
            passed,
            f"Base: {base_time:.3f}s, Fatigue: {fatigue_time:.3f}s",
        )
        all_passed = all_passed and passed

        return all_passed

    def verify_water_penalty(self) -> bool:
        """Test that mining underwater is slower."""
        tool = ToolProperties(ToolType.PICKAXE, ToolMaterial.DIAMOND)

        normal_time = calculate_break_time("stone", tool)
        water_time = calculate_break_time("stone", tool, in_water=True)

        passed = water_time > normal_time * 4.9  # Should be 5x slower
        self._add_result(
            "water_penalty",
            passed,
            f"Normal: {normal_time:.3f}s, Water: {water_time:.3f}s",
        )
        return passed

    def verify_air_penalty(self) -> bool:
        """Test that mining in air is slower."""
        tool = ToolProperties(ToolType.PICKAXE, ToolMaterial.DIAMOND)

        ground_time = calculate_break_time("stone", tool, on_ground=True)
        air_time = calculate_break_time("stone", tool, on_ground=False)

        passed = air_time > ground_time * 4.9  # Should be 5x slower
        self._add_result(
            "air_penalty",
            passed,
            f"Ground: {ground_time:.3f}s, Air: {air_time:.3f}s",
        )
        return passed

    def verify_silk_touch_drops(self) -> bool:
        """Test silk touch drop behavior."""
        test_cases = [
            ("stone", "stone"),  # Drops stone instead of cobblestone
            ("grass_block", "grass_block"),  # Drops grass block instead of dirt
            ("glass", "glass"),  # Drops glass (normally nothing)
            ("diamond_ore", "diamond_ore"),  # Drops ore instead of diamond
            ("coal_ore", "coal_ore"),
        ]

        all_passed = True
        for block_id, expected_drop in test_cases:
            if block_id not in BLOCK_REGISTRY:
                continue

            silk_tool = ToolProperties(
                ToolType.PICKAXE
                if BLOCK_REGISTRY[block_id].preferred_tool == ToolType.PICKAXE
                else ToolType.SHOVEL,
                ToolMaterial.DIAMOND,
                silk_touch=True,
            )
            drops = get_drops(block_id, silk_tool)

            has_expected = any(d[0] == expected_drop for d in drops)
            self._add_result(
                f"silk_touch_{block_id}",
                has_expected,
                f"Expected {expected_drop}, got {drops}",
            )
            all_passed = all_passed and has_expected

        return all_passed

    def verify_fortune_drops(self) -> bool:
        """Test fortune enchantment increases drops."""
        ore_blocks = [
            "coal_ore",
            "diamond_ore",
            "lapis_ore",
            "redstone_ore",
            "emerald_ore",
            "nether_quartz_ore",
        ]

        all_passed = True
        for block_id in ore_blocks:
            if block_id not in BLOCK_REGISTRY or block_id not in DROP_TABLES:
                continue

            base_tool = ToolProperties(ToolType.PICKAXE, ToolMaterial.DIAMOND)
            fortune_tool = ToolProperties(ToolType.PICKAXE, ToolMaterial.DIAMOND, fortune_level=3)

            base_drops = get_drops(block_id, base_tool)
            fortune_drops = get_drops(block_id, fortune_tool)

            # Fortune should increase max drops
            base_max = sum(d[2] for d in base_drops)
            fortune_max = sum(d[2] for d in fortune_drops)

            passed = fortune_max > base_max
            self._add_result(
                f"fortune_{block_id}",
                passed,
                f"Base max: {base_max}, Fortune max: {fortune_max}",
            )
            all_passed = all_passed and passed

        return all_passed

    def verify_obsidian_break_time(self) -> bool:
        """Test obsidian break times with different tools."""
        # Obsidian: hardness 50, requires diamond pickaxe

        # Diamond pickaxe: ~9.4 seconds
        diamond_pick = ToolProperties(ToolType.PICKAXE, ToolMaterial.DIAMOND)
        diamond_time = calculate_break_time("obsidian", diamond_pick)

        # Netherite pickaxe: ~8.35 seconds
        netherite_pick = ToolProperties(ToolType.PICKAXE, ToolMaterial.NETHERITE)
        netherite_time = calculate_break_time("obsidian", netherite_pick)

        # Diamond with Eff V: ~2.6 seconds
        diamond_eff5 = ToolProperties(ToolType.PICKAXE, ToolMaterial.DIAMOND, efficiency_level=5)
        eff5_time = calculate_break_time("obsidian", diamond_eff5)

        # Verify relative speeds
        passed = (
            netherite_time < diamond_time
            and eff5_time < diamond_time
            and diamond_time > 9.0
            and diamond_time < 10.0
        )

        self._add_result(
            "obsidian_break_times",
            passed,
            f"Diamond: {diamond_time:.2f}s, Netherite: {netherite_time:.2f}s, "
            f"Diamond+Eff5: {eff5_time:.2f}s",
        )
        return passed

    def verify_shears_special_cases(self) -> bool:
        """Test shears on special blocks (wool, leaves, cobweb)."""
        all_passed = True

        shears = ToolProperties(ToolType.SHEARS, ToolMaterial.HAND)

        # Wool should be instant with shears
        wool_time = calculate_break_time("white_wool", shears)
        hand_wool_time = calculate_break_time("white_wool")

        passed = wool_time < hand_wool_time
        self._add_result(
            "shears_wool",
            passed,
            f"Shears: {wool_time:.3f}s, Hand: {hand_wool_time:.3f}s",
        )
        all_passed = all_passed and passed

        # Leaves should be fast with shears
        leaves_time = calculate_break_time("oak_leaves", shears)
        hand_leaves_time = calculate_break_time("oak_leaves")

        passed = leaves_time < hand_leaves_time
        self._add_result(
            "shears_leaves",
            passed,
            f"Shears: {leaves_time:.3f}s, Hand: {hand_leaves_time:.3f}s",
        )
        all_passed = all_passed and passed

        # Cobweb should be fast with shears or sword
        cobweb_shears = calculate_break_time("cobweb", shears)
        sword = ToolProperties(ToolType.SWORD, ToolMaterial.DIAMOND)
        cobweb_sword = calculate_break_time("cobweb", sword)
        cobweb_hand = calculate_break_time("cobweb")

        passed = cobweb_shears < cobweb_hand and cobweb_sword < cobweb_hand
        self._add_result(
            "shears_sword_cobweb",
            passed,
            f"Shears: {cobweb_shears:.3f}s, Sword: {cobweb_sword:.3f}s, Hand: {cobweb_hand:.3f}s",
        )
        all_passed = all_passed and passed

        return all_passed

    def run_all_tests(self) -> tuple[int, int]:
        """Run all verification tests.

        Returns:
            Tuple of (passed_count, total_count)
        """
        self.test_results.clear()

        self.verify_instant_break_blocks()
        self.verify_unbreakable_blocks()
        self.verify_tool_speeds()
        self.verify_harvest_levels()
        self.verify_efficiency_enchantment()
        self.verify_haste_effect()
        self.verify_mining_fatigue()
        self.verify_water_penalty()
        self.verify_air_penalty()
        self.verify_silk_touch_drops()
        self.verify_fortune_drops()
        self.verify_obsidian_break_time()
        self.verify_shears_special_cases()

        passed = sum(1 for _, p, _ in self.test_results if p)
        total = len(self.test_results)

        return passed, total

    def print_results(self) -> None:
        """Print all test results."""
        passed, total = len(self.test_results), len(self.test_results)
        passed = sum(1 for _, p, _ in self.test_results if p)

        print(f"\nBlock Breaking Verification Results: {passed}/{total} passed\n")
        print("-" * 70)

        for name, success, message in self.test_results:
            status = "PASS" if success else "FAIL"
            print(f"[{status}] {name}")
            if message:
                print(f"       {message}")

        print("-" * 70)
        print(f"Total: {passed}/{total} tests passed")


def main() -> None:
    """Run block breaking verification."""
    verifier = BlockBreakingVerifier()
    passed, total = verifier.run_all_tests()
    verifier.print_results()

    # Exit with error code if tests failed
    if passed < total:
        exit(1)


if __name__ == "__main__":
    main()
