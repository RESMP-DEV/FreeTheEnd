"""Test generator for inventory subsystem verification.

Generates comprehensive test cases for:
1. Item add/remove operations
2. Slot movement and swapping
3. Stack size limits
4. Durability tracking
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ItemCategory(Enum):
    """Item categories affecting stack behavior."""

    BLOCK = auto()
    TOOL = auto()
    WEAPON = auto()
    ARMOR = auto()
    FOOD = auto()
    MATERIAL = auto()
    MISC = auto()


@dataclass
class ItemDefinition:
    """Definition of an item type."""

    item_id: str
    category: ItemCategory
    max_stack: int
    has_durability: bool
    max_durability: int = 0

    def __post_init__(self) -> None:
        if self.has_durability and self.max_durability <= 0:
            raise ValueError(f"Item {self.item_id} has durability but max_durability <= 0")
        if self.has_durability and self.max_stack != 1:
            raise ValueError(f"Item {self.item_id} has durability but max_stack != 1")


@dataclass
class ItemStack:
    """Represents a stack of items in inventory."""

    item_id: str
    count: int
    durability: int | None = None

    def copy(self) -> ItemStack:
        return ItemStack(self.item_id, self.count, self.durability)


@dataclass
class TestCase:
    """A single test case for inventory verification."""

    name: str
    description: str
    operation: str
    inputs: dict[str, Any]
    expected_result: dict[str, Any]
    expected_error: str | None = None


@dataclass
class TestSuite:
    """Collection of test cases for a specific aspect."""

    name: str
    test_cases: list[TestCase] = field(default_factory=list)

    def add_test(self, test: TestCase) -> None:
        self.test_cases.append(test)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "test_count": len(self.test_cases),
            "tests": [
                {
                    "name": tc.name,
                    "description": tc.description,
                    "operation": tc.operation,
                    "inputs": tc.inputs,
                    "expected_result": tc.expected_result,
                    "expected_error": tc.expected_error,
                }
                for tc in self.test_cases
            ],
        }


# Standard Minecraft-style item definitions for speedrun scenarios
SPEEDRUN_ITEMS: dict[str, ItemDefinition] = {
    # Blocks (stack to 64)
    "cobblestone": ItemDefinition("cobblestone", ItemCategory.BLOCK, 64, False),
    "dirt": ItemDefinition("dirt", ItemCategory.BLOCK, 64, False),
    "oak_log": ItemDefinition("oak_log", ItemCategory.BLOCK, 64, False),
    "oak_planks": ItemDefinition("oak_planks", ItemCategory.BLOCK, 64, False),
    "crafting_table": ItemDefinition("crafting_table", ItemCategory.BLOCK, 64, False),
    "furnace": ItemDefinition("furnace", ItemCategory.BLOCK, 64, False),
    "chest": ItemDefinition("chest", ItemCategory.BLOCK, 64, False),
    "obsidian": ItemDefinition("obsidian", ItemCategory.BLOCK, 64, False),
    "gravel": ItemDefinition("gravel", ItemCategory.BLOCK, 64, False),
    "sand": ItemDefinition("sand", ItemCategory.BLOCK, 64, False),
    "glass": ItemDefinition("glass", ItemCategory.BLOCK, 64, False),
    "iron_block": ItemDefinition("iron_block", ItemCategory.BLOCK, 64, False),
    "gold_block": ItemDefinition("gold_block", ItemCategory.BLOCK, 64, False),
    "diamond_block": ItemDefinition("diamond_block", ItemCategory.BLOCK, 64, False),
    # Tools (no stacking, has durability)
    "wooden_pickaxe": ItemDefinition("wooden_pickaxe", ItemCategory.TOOL, 1, True, 60),
    "stone_pickaxe": ItemDefinition("stone_pickaxe", ItemCategory.TOOL, 1, True, 132),
    "iron_pickaxe": ItemDefinition("iron_pickaxe", ItemCategory.TOOL, 1, True, 251),
    "diamond_pickaxe": ItemDefinition("diamond_pickaxe", ItemCategory.TOOL, 1, True, 1562),
    "wooden_axe": ItemDefinition("wooden_axe", ItemCategory.TOOL, 1, True, 60),
    "stone_axe": ItemDefinition("stone_axe", ItemCategory.TOOL, 1, True, 132),
    "iron_axe": ItemDefinition("iron_axe", ItemCategory.TOOL, 1, True, 251),
    "diamond_axe": ItemDefinition("diamond_axe", ItemCategory.TOOL, 1, True, 1562),
    "wooden_shovel": ItemDefinition("wooden_shovel", ItemCategory.TOOL, 1, True, 60),
    "stone_shovel": ItemDefinition("stone_shovel", ItemCategory.TOOL, 1, True, 132),
    "iron_shovel": ItemDefinition("iron_shovel", ItemCategory.TOOL, 1, True, 251),
    "diamond_shovel": ItemDefinition("diamond_shovel", ItemCategory.TOOL, 1, True, 1562),
    "flint_and_steel": ItemDefinition("flint_and_steel", ItemCategory.TOOL, 1, True, 65),
    # Weapons
    "wooden_sword": ItemDefinition("wooden_sword", ItemCategory.WEAPON, 1, True, 60),
    "stone_sword": ItemDefinition("stone_sword", ItemCategory.WEAPON, 1, True, 132),
    "iron_sword": ItemDefinition("iron_sword", ItemCategory.WEAPON, 1, True, 251),
    "diamond_sword": ItemDefinition("diamond_sword", ItemCategory.WEAPON, 1, True, 1562),
    "bow": ItemDefinition("bow", ItemCategory.WEAPON, 1, True, 385),
    # Armor
    "iron_helmet": ItemDefinition("iron_helmet", ItemCategory.ARMOR, 1, True, 166),
    "iron_chestplate": ItemDefinition("iron_chestplate", ItemCategory.ARMOR, 1, True, 241),
    "iron_leggings": ItemDefinition("iron_leggings", ItemCategory.ARMOR, 1, True, 226),
    "iron_boots": ItemDefinition("iron_boots", ItemCategory.ARMOR, 1, True, 196),
    "diamond_helmet": ItemDefinition("diamond_helmet", ItemCategory.ARMOR, 1, True, 364),
    "diamond_chestplate": ItemDefinition("diamond_chestplate", ItemCategory.ARMOR, 1, True, 529),
    "diamond_leggings": ItemDefinition("diamond_leggings", ItemCategory.ARMOR, 1, True, 496),
    "diamond_boots": ItemDefinition("diamond_boots", ItemCategory.ARMOR, 1, True, 430),
    # Materials (stack to 64)
    "stick": ItemDefinition("stick", ItemCategory.MATERIAL, 64, False),
    "coal": ItemDefinition("coal", ItemCategory.MATERIAL, 64, False),
    "iron_ingot": ItemDefinition("iron_ingot", ItemCategory.MATERIAL, 64, False),
    "gold_ingot": ItemDefinition("gold_ingot", ItemCategory.MATERIAL, 64, False),
    "diamond": ItemDefinition("diamond", ItemCategory.MATERIAL, 64, False),
    "string": ItemDefinition("string", ItemCategory.MATERIAL, 64, False),
    "flint": ItemDefinition("flint", ItemCategory.MATERIAL, 64, False),
    "leather": ItemDefinition("leather", ItemCategory.MATERIAL, 64, False),
    "feather": ItemDefinition("feather", ItemCategory.MATERIAL, 64, False),
    "blaze_rod": ItemDefinition("blaze_rod", ItemCategory.MATERIAL, 64, False),
    "blaze_powder": ItemDefinition("blaze_powder", ItemCategory.MATERIAL, 64, False),
    "ender_pearl": ItemDefinition("ender_pearl", ItemCategory.MATERIAL, 16, False),  # Special stack
    "eye_of_ender": ItemDefinition("eye_of_ender", ItemCategory.MATERIAL, 64, False),
    "nether_wart": ItemDefinition("nether_wart", ItemCategory.MATERIAL, 64, False),
    "gunpowder": ItemDefinition("gunpowder", ItemCategory.MATERIAL, 64, False),
    "redstone": ItemDefinition("redstone", ItemCategory.MATERIAL, 64, False),
    "glowstone_dust": ItemDefinition("glowstone_dust", ItemCategory.MATERIAL, 64, False),
    # Food (stack to 64)
    "apple": ItemDefinition("apple", ItemCategory.FOOD, 64, False),
    "bread": ItemDefinition("bread", ItemCategory.FOOD, 64, False),
    "cooked_porkchop": ItemDefinition("cooked_porkchop", ItemCategory.FOOD, 64, False),
    "golden_apple": ItemDefinition("golden_apple", ItemCategory.FOOD, 64, False),
    # Special stacking items
    "bucket": ItemDefinition("bucket", ItemCategory.MISC, 16, False),
    "water_bucket": ItemDefinition("water_bucket", ItemCategory.MISC, 1, False),
    "lava_bucket": ItemDefinition("lava_bucket", ItemCategory.MISC, 1, False),
    "snowball": ItemDefinition("snowball", ItemCategory.MISC, 16, False),
    "egg": ItemDefinition("egg", ItemCategory.MISC, 16, False),
    "arrow": ItemDefinition("arrow", ItemCategory.MISC, 64, False),
    "bed": ItemDefinition("bed", ItemCategory.MISC, 1, False),
}


class InventoryTestGenerator:
    """Generates test cases for inventory subsystem."""

    def __init__(
        self,
        items: dict[str, ItemDefinition] | None = None,
        inventory_size: int = 36,
        hotbar_size: int = 9,
        seed: int | None = None,
    ) -> None:
        """Initialize the test generator.

        Args:
            items: Item definitions to use for testing. Defaults to SPEEDRUN_ITEMS.
            inventory_size: Number of inventory slots.
            hotbar_size: Number of hotbar slots (subset of inventory).
            seed: Random seed for reproducible tests.
        """
        self.items = items or SPEEDRUN_ITEMS
        self.inventory_size = inventory_size
        self.hotbar_size = hotbar_size
        self.rng = random.Random(seed)

    def generate_add_remove_tests(self) -> TestSuite:
        """Generate tests for item add/remove operations."""
        suite = TestSuite("Item Add/Remove Operations")

        # Test 1: Add single item to empty slot
        suite.add_test(
            TestCase(
                name="add_single_item_empty_slot",
                description="Add a single item to an empty inventory slot",
                operation="add_item",
                inputs={
                    "item_id": "cobblestone",
                    "count": 1,
                    "target_slot": 0,
                    "initial_inventory": [None] * self.inventory_size,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "cobblestone", "count": 1, "durability": None},
                    "remaining": 0,
                },
            )
        )

        # Test 2: Add full stack
        suite.add_test(
            TestCase(
                name="add_full_stack",
                description="Add a full stack (64) of blocks",
                operation="add_item",
                inputs={
                    "item_id": "dirt",
                    "count": 64,
                    "target_slot": 0,
                    "initial_inventory": [None] * self.inventory_size,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "dirt", "count": 64, "durability": None},
                    "remaining": 0,
                },
            )
        )

        # Test 3: Add to existing partial stack
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "cobblestone", "count": 32, "durability": None}
        suite.add_test(
            TestCase(
                name="add_to_existing_stack",
                description="Add items to an existing partial stack",
                operation="add_item",
                inputs={
                    "item_id": "cobblestone",
                    "count": 16,
                    "target_slot": 0,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "cobblestone", "count": 48, "durability": None},
                    "remaining": 0,
                },
            )
        )

        # Test 4: Overflow to next slot
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "cobblestone", "count": 50, "durability": None}
        suite.add_test(
            TestCase(
                name="add_overflow_to_next_slot",
                description="Add items that overflow to the next available slot",
                operation="add_item",
                inputs={
                    "item_id": "cobblestone",
                    "count": 20,
                    "auto_stack": True,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "cobblestone", "count": 64, "durability": None},
                    "slot_1": {"item_id": "cobblestone", "count": 6, "durability": None},
                    "remaining": 0,
                },
            )
        )

        # Test 5: Add tool with durability
        suite.add_test(
            TestCase(
                name="add_tool_with_durability",
                description="Add a tool item with full durability",
                operation="add_item",
                inputs={
                    "item_id": "iron_pickaxe",
                    "count": 1,
                    "target_slot": 0,
                    "initial_inventory": [None] * self.inventory_size,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "iron_pickaxe", "count": 1, "durability": 251},
                    "remaining": 0,
                },
            )
        )

        # Test 6: Remove items from stack
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "iron_ingot", "count": 10, "durability": None}
        suite.add_test(
            TestCase(
                name="remove_items_from_stack",
                description="Remove some items from a stack",
                operation="remove_item",
                inputs={
                    "slot": 0,
                    "count": 3,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "removed": {"item_id": "iron_ingot", "count": 3, "durability": None},
                    "slot_0": {"item_id": "iron_ingot", "count": 7, "durability": None},
                },
            )
        )

        # Test 7: Remove entire stack
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "diamond", "count": 5, "durability": None}
        suite.add_test(
            TestCase(
                name="remove_entire_stack",
                description="Remove all items from a slot",
                operation="remove_item",
                inputs={
                    "slot": 0,
                    "count": 5,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "removed": {"item_id": "diamond", "count": 5, "durability": None},
                    "slot_0": None,
                },
            )
        )

        # Test 8: Remove more than available (should fail)
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "diamond", "count": 5, "durability": None}
        suite.add_test(
            TestCase(
                name="remove_more_than_available",
                description="Attempt to remove more items than present in stack",
                operation="remove_item",
                inputs={
                    "slot": 0,
                    "count": 10,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": False,
                    "slot_0": {"item_id": "diamond", "count": 5, "durability": None},
                },
                expected_error="InsufficientItemsError",
            )
        )

        # Test 9: Remove from empty slot
        suite.add_test(
            TestCase(
                name="remove_from_empty_slot",
                description="Attempt to remove items from an empty slot",
                operation="remove_item",
                inputs={
                    "slot": 5,
                    "count": 1,
                    "initial_inventory": [None] * self.inventory_size,
                },
                expected_result={
                    "success": False,
                },
                expected_error="EmptySlotError",
            )
        )

        # Test 10: Add to full inventory (no space)
        full_inv = [
            {"item_id": "cobblestone", "count": 64, "durability": None}
        ] * self.inventory_size
        suite.add_test(
            TestCase(
                name="add_to_full_inventory",
                description="Attempt to add items when inventory is completely full",
                operation="add_item",
                inputs={
                    "item_id": "dirt",
                    "count": 1,
                    "auto_stack": True,
                    "initial_inventory": full_inv,
                },
                expected_result={
                    "success": False,
                    "remaining": 1,
                },
                expected_error="InventoryFullError",
            )
        )

        # Test 11: Add items with special stack limit (ender pearls = 16)
        suite.add_test(
            TestCase(
                name="add_special_stack_limit_item",
                description="Add ender pearls which have a stack limit of 16",
                operation="add_item",
                inputs={
                    "item_id": "ender_pearl",
                    "count": 20,
                    "auto_stack": True,
                    "initial_inventory": [None] * self.inventory_size,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "ender_pearl", "count": 16, "durability": None},
                    "slot_1": {"item_id": "ender_pearl", "count": 4, "durability": None},
                    "remaining": 0,
                },
            )
        )

        # Test 12: Add non-stackable item (lava bucket)
        suite.add_test(
            TestCase(
                name="add_non_stackable_item",
                description="Add a lava bucket which doesn't stack",
                operation="add_item",
                inputs={
                    "item_id": "lava_bucket",
                    "count": 1,
                    "target_slot": 0,
                    "initial_inventory": [None] * self.inventory_size,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "lava_bucket", "count": 1, "durability": None},
                    "remaining": 0,
                },
            )
        )

        # Test 13: Multiple non-stackable items
        suite.add_test(
            TestCase(
                name="add_multiple_non_stackable",
                description="Add multiple beds which require separate slots",
                operation="add_item",
                inputs={
                    "item_id": "bed",
                    "count": 3,
                    "auto_stack": True,
                    "initial_inventory": [None] * self.inventory_size,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "bed", "count": 1, "durability": None},
                    "slot_1": {"item_id": "bed", "count": 1, "durability": None},
                    "slot_2": {"item_id": "bed", "count": 1, "durability": None},
                    "remaining": 0,
                },
            )
        )

        return suite

    def generate_slot_movement_tests(self) -> TestSuite:
        """Generate tests for slot movement and swapping."""
        suite = TestSuite("Slot Movement and Swapping")

        # Test 1: Move item to empty slot
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "iron_ingot", "count": 10, "durability": None}
        suite.add_test(
            TestCase(
                name="move_to_empty_slot",
                description="Move an item stack to an empty slot",
                operation="move_item",
                inputs={
                    "source_slot": 0,
                    "target_slot": 5,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": None,
                    "slot_5": {"item_id": "iron_ingot", "count": 10, "durability": None},
                },
            )
        )

        # Test 2: Swap two item stacks
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "iron_ingot", "count": 10, "durability": None}
        initial_inv[5] = {"item_id": "diamond", "count": 3, "durability": None}
        suite.add_test(
            TestCase(
                name="swap_two_stacks",
                description="Swap two different item stacks",
                operation="swap_items",
                inputs={
                    "slot_a": 0,
                    "slot_b": 5,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "diamond", "count": 3, "durability": None},
                    "slot_5": {"item_id": "iron_ingot", "count": 10, "durability": None},
                },
            )
        )

        # Test 3: Merge compatible stacks
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "cobblestone", "count": 32, "durability": None}
        initial_inv[5] = {"item_id": "cobblestone", "count": 20, "durability": None}
        suite.add_test(
            TestCase(
                name="merge_compatible_stacks",
                description="Move stack onto compatible partial stack (should merge)",
                operation="move_item",
                inputs={
                    "source_slot": 0,
                    "target_slot": 5,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": None,
                    "slot_5": {"item_id": "cobblestone", "count": 52, "durability": None},
                },
            )
        )

        # Test 4: Partial merge with overflow
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "cobblestone", "count": 50, "durability": None}
        initial_inv[5] = {"item_id": "cobblestone", "count": 40, "durability": None}
        suite.add_test(
            TestCase(
                name="partial_merge_overflow",
                description="Merge stacks where total exceeds max stack (overflow stays)",
                operation="move_item",
                inputs={
                    "source_slot": 0,
                    "target_slot": 5,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "cobblestone", "count": 26, "durability": None},
                    "slot_5": {"item_id": "cobblestone", "count": 64, "durability": None},
                },
            )
        )

        # Test 5: Move partial stack
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "iron_ingot", "count": 20, "durability": None}
        suite.add_test(
            TestCase(
                name="move_partial_stack",
                description="Move only part of a stack to another slot",
                operation="move_partial",
                inputs={
                    "source_slot": 0,
                    "target_slot": 5,
                    "count": 8,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "iron_ingot", "count": 12, "durability": None},
                    "slot_5": {"item_id": "iron_ingot", "count": 8, "durability": None},
                },
            )
        )

        # Test 6: Split stack in half
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "diamond", "count": 10, "durability": None}
        suite.add_test(
            TestCase(
                name="split_stack_half",
                description="Split a stack in half (right-click behavior)",
                operation="split_stack",
                inputs={
                    "source_slot": 0,
                    "target_slot": 1,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "diamond", "count": 5, "durability": None},
                    "slot_1": {"item_id": "diamond", "count": 5, "durability": None},
                },
            )
        )

        # Test 7: Split odd stack
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "diamond", "count": 7, "durability": None}
        suite.add_test(
            TestCase(
                name="split_odd_stack",
                description="Split an odd-numbered stack (rounds down)",
                operation="split_stack",
                inputs={
                    "source_slot": 0,
                    "target_slot": 1,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "diamond", "count": 4, "durability": None},
                    "slot_1": {"item_id": "diamond", "count": 3, "durability": None},
                },
            )
        )

        # Test 8: Move tool (preserves durability)
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "iron_pickaxe", "count": 1, "durability": 150}
        suite.add_test(
            TestCase(
                name="move_tool_preserve_durability",
                description="Move a partially used tool, durability should be preserved",
                operation="move_item",
                inputs={
                    "source_slot": 0,
                    "target_slot": 8,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": None,
                    "slot_8": {"item_id": "iron_pickaxe", "count": 1, "durability": 150},
                },
            )
        )

        # Test 9: Swap tools with different durabilities
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "iron_pickaxe", "count": 1, "durability": 100}
        initial_inv[1] = {"item_id": "iron_pickaxe", "count": 1, "durability": 200}
        suite.add_test(
            TestCase(
                name="swap_tools_different_durability",
                description="Swap two pickaxes with different durability values",
                operation="swap_items",
                inputs={
                    "slot_a": 0,
                    "slot_b": 1,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "iron_pickaxe", "count": 1, "durability": 200},
                    "slot_1": {"item_id": "iron_pickaxe", "count": 1, "durability": 100},
                },
            )
        )

        # Test 10: Move from hotbar to main inventory
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "diamond_sword", "count": 1, "durability": 1562}
        suite.add_test(
            TestCase(
                name="move_from_hotbar",
                description="Move item from hotbar (slot 0) to main inventory",
                operation="move_item",
                inputs={
                    "source_slot": 0,
                    "target_slot": 27,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": None,
                    "slot_27": {"item_id": "diamond_sword", "count": 1, "durability": 1562},
                },
            )
        )

        # Test 11: Move empty slot (should fail)
        suite.add_test(
            TestCase(
                name="move_empty_slot",
                description="Attempt to move from an empty slot",
                operation="move_item",
                inputs={
                    "source_slot": 0,
                    "target_slot": 5,
                    "initial_inventory": [None] * self.inventory_size,
                },
                expected_result={
                    "success": False,
                },
                expected_error="EmptySlotError",
            )
        )

        # Test 12: Move to invalid slot index
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "cobblestone", "count": 64, "durability": None}
        suite.add_test(
            TestCase(
                name="move_to_invalid_slot",
                description="Attempt to move to slot index beyond inventory size",
                operation="move_item",
                inputs={
                    "source_slot": 0,
                    "target_slot": 100,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": False,
                },
                expected_error="InvalidSlotError",
            )
        )

        return suite

    def generate_stack_limit_tests(self) -> TestSuite:
        """Generate tests for stack size limit enforcement."""
        suite = TestSuite("Stack Size Limits")

        # Test standard 64-stack items
        for item_id in ["cobblestone", "dirt", "oak_planks", "iron_ingot", "diamond"]:
            item_def = self.items[item_id]
            suite.add_test(
                TestCase(
                    name=f"stack_limit_{item_id}",
                    description=f"Verify {item_id} respects stack limit of {item_def.max_stack}",
                    operation="add_item",
                    inputs={
                        "item_id": item_id,
                        "count": 100,
                        "auto_stack": True,
                        "initial_inventory": [None] * self.inventory_size,
                    },
                    expected_result={
                        "success": True,
                        "slot_0": {"item_id": item_id, "count": 64, "durability": None},
                        "slot_1": {"item_id": item_id, "count": 36, "durability": None},
                        "remaining": 0,
                    },
                )
            )

        # Test 16-stack items
        for item_id in ["ender_pearl", "snowball", "egg", "bucket"]:
            item_def = self.items[item_id]
            suite.add_test(
                TestCase(
                    name=f"stack_limit_16_{item_id}",
                    description=f"Verify {item_id} respects stack limit of {item_def.max_stack}",
                    operation="add_item",
                    inputs={
                        "item_id": item_id,
                        "count": 20,
                        "auto_stack": True,
                        "initial_inventory": [None] * self.inventory_size,
                    },
                    expected_result={
                        "success": True,
                        "slot_0": {"item_id": item_id, "count": 16, "durability": None},
                        "slot_1": {"item_id": item_id, "count": 4, "durability": None},
                        "remaining": 0,
                    },
                )
            )

        # Test non-stackable items
        for item_id in ["water_bucket", "lava_bucket", "bed"]:
            item_def = self.items[item_id]
            suite.add_test(
                TestCase(
                    name=f"stack_limit_1_{item_id}",
                    description=f"Verify {item_id} cannot stack (limit of 1)",
                    operation="add_item",
                    inputs={
                        "item_id": item_id,
                        "count": 3,
                        "auto_stack": True,
                        "initial_inventory": [None] * self.inventory_size,
                    },
                    expected_result={
                        "success": True,
                        "slot_0": {"item_id": item_id, "count": 1, "durability": None},
                        "slot_1": {"item_id": item_id, "count": 1, "durability": None},
                        "slot_2": {"item_id": item_id, "count": 1, "durability": None},
                        "remaining": 0,
                    },
                )
            )

        # Test tools (non-stackable with durability)
        for item_id in ["iron_pickaxe", "diamond_sword", "bow"]:
            item_def = self.items[item_id]
            suite.add_test(
                TestCase(
                    name=f"stack_limit_tool_{item_id}",
                    description=f"Verify {item_id} cannot stack (tool with durability)",
                    operation="add_item",
                    inputs={
                        "item_id": item_id,
                        "count": 2,
                        "auto_stack": True,
                        "initial_inventory": [None] * self.inventory_size,
                    },
                    expected_result={
                        "success": True,
                        "slot_0": {
                            "item_id": item_id,
                            "count": 1,
                            "durability": item_def.max_durability,
                        },
                        "slot_1": {
                            "item_id": item_id,
                            "count": 1,
                            "durability": item_def.max_durability,
                        },
                        "remaining": 0,
                    },
                )
            )

        # Test exceeding stack limit on add to existing
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "ender_pearl", "count": 14, "durability": None}
        suite.add_test(
            TestCase(
                name="exceed_stack_limit_partial",
                description="Add ender pearls to partial stack, overflow to next slot",
                operation="add_item",
                inputs={
                    "item_id": "ender_pearl",
                    "count": 5,
                    "auto_stack": True,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "ender_pearl", "count": 16, "durability": None},
                    "slot_1": {"item_id": "ender_pearl", "count": 3, "durability": None},
                    "remaining": 0,
                },
            )
        )

        return suite

    def generate_durability_tests(self) -> TestSuite:
        """Generate tests for durability tracking."""
        suite = TestSuite("Durability Tracking")

        # Test 1: New tool has full durability
        for item_id, item_def in self.items.items():
            if not item_def.has_durability:
                continue
            suite.add_test(
                TestCase(
                    name=f"new_tool_full_durability_{item_id}",
                    description=f"New {item_id} should have durability {item_def.max_durability}",
                    operation="add_item",
                    inputs={
                        "item_id": item_id,
                        "count": 1,
                        "target_slot": 0,
                        "initial_inventory": [None] * self.inventory_size,
                    },
                    expected_result={
                        "success": True,
                        "slot_0": {
                            "item_id": item_id,
                            "count": 1,
                            "durability": item_def.max_durability,
                        },
                    },
                )
            )

        # Test 2: Reduce durability
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "iron_pickaxe", "count": 1, "durability": 251}
        suite.add_test(
            TestCase(
                name="reduce_durability_single",
                description="Use iron pickaxe once, reduce durability by 1",
                operation="use_item",
                inputs={
                    "slot": 0,
                    "durability_cost": 1,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "iron_pickaxe", "count": 1, "durability": 250},
                },
            )
        )

        # Test 3: Reduce durability by multiple
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "diamond_pickaxe", "count": 1, "durability": 1562}
        suite.add_test(
            TestCase(
                name="reduce_durability_multiple",
                description="Mine 100 blocks with diamond pickaxe",
                operation="use_item",
                inputs={
                    "slot": 0,
                    "durability_cost": 100,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "diamond_pickaxe", "count": 1, "durability": 1462},
                },
            )
        )

        # Test 4: Tool breaks at 0 durability
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "wooden_pickaxe", "count": 1, "durability": 1}
        suite.add_test(
            TestCase(
                name="tool_breaks_at_zero",
                description="Wooden pickaxe with 1 durability breaks after use",
                operation="use_item",
                inputs={
                    "slot": 0,
                    "durability_cost": 1,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": None,
                    "item_broken": True,
                },
            )
        )

        # Test 5: Cannot use broken tool
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "stone_pickaxe", "count": 1, "durability": 0}
        suite.add_test(
            TestCase(
                name="cannot_use_zero_durability",
                description="Tool with 0 durability cannot be used",
                operation="use_item",
                inputs={
                    "slot": 0,
                    "durability_cost": 1,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": False,
                },
                expected_error="BrokenToolError",
            )
        )

        # Test 6: Durability preserved on stack operations
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "diamond_sword", "count": 1, "durability": 1000}
        suite.add_test(
            TestCase(
                name="durability_preserved_on_move",
                description="Moving a tool preserves its durability",
                operation="move_item",
                inputs={
                    "source_slot": 0,
                    "target_slot": 5,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": None,
                    "slot_5": {"item_id": "diamond_sword", "count": 1, "durability": 1000},
                },
            )
        )

        # Test 7: Flint and steel special durability
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "flint_and_steel", "count": 1, "durability": 65}
        suite.add_test(
            TestCase(
                name="flint_and_steel_usage",
                description="Flint and steel uses 1 durability per ignition",
                operation="use_item",
                inputs={
                    "slot": 0,
                    "durability_cost": 1,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "flint_and_steel", "count": 1, "durability": 64},
                },
            )
        )

        # Test 8: Armor durability on hit
        initial_inv = [None] * self.inventory_size
        initial_inv[0] = {"item_id": "iron_chestplate", "count": 1, "durability": 241}
        suite.add_test(
            TestCase(
                name="armor_durability_damage",
                description="Iron chestplate takes damage when hit",
                operation="damage_armor",
                inputs={
                    "slot": 0,
                    "damage_points": 5,
                    "initial_inventory": initial_inv,
                },
                expected_result={
                    "success": True,
                    "slot_0": {"item_id": "iron_chestplate", "count": 1, "durability": 236},
                },
            )
        )

        return suite

    def generate_all_tests(self) -> dict[str, TestSuite]:
        """Generate all test suites."""
        return {
            "add_remove": self.generate_add_remove_tests(),
            "slot_movement": self.generate_slot_movement_tests(),
            "stack_limits": self.generate_stack_limit_tests(),
            "durability": self.generate_durability_tests(),
        }

    def export_tests(self, filepath: str) -> None:
        """Export all tests to JSON file."""
        all_suites = self.generate_all_tests()
        export_data = {suite_name: suite.to_dict() for suite_name, suite in all_suites.items()}
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

    def get_test_count(self) -> dict[str, int]:
        """Get count of tests per suite."""
        all_suites = self.generate_all_tests()
        return {name: len(suite.test_cases) for name, suite in all_suites.items()}


def main() -> None:
    """Run test generation and display summary."""
    generator = InventoryTestGenerator(seed=42)

    print("Inventory Test Generator")
    print("=" * 50)

    test_counts = generator.get_test_count()
    total = sum(test_counts.values())

    for suite_name, count in test_counts.items():
        print(f"  {suite_name}: {count} tests")

    print(f"\nTotal: {total} tests")

    # Export to JSON
    generator.export_tests("inventory_tests.json")
    print("\nTests exported to inventory_tests.json")


if __name__ == "__main__":
    main()
