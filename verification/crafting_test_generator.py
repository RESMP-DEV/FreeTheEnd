"""Test generator for crafting subsystem verification.

Generates comprehensive test cases for:
1. All speedrun-relevant recipes craft correctly
2. Recipe ingredient consumption is exact
3. Shaped vs shapeless recipe handling
4. Multi-output recipes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class RecipeType(Enum):
    """Type of crafting recipe."""

    SHAPED = auto()  # Exact pattern required
    SHAPELESS = auto()  # Any arrangement works
    SMELTING = auto()  # Furnace recipes
    SMITHING = auto()  # Smithing table
    BREWING = auto()  # Brewing stand


@dataclass
class Ingredient:
    """A single ingredient for a recipe."""

    item_id: str
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "count": self.count}


@dataclass
class Recipe:
    """A crafting recipe definition."""

    recipe_id: str
    recipe_type: RecipeType
    output_item: str
    output_count: int
    ingredients: list[Ingredient]
    pattern: list[str] | None = None  # For shaped recipes
    key: dict[str, str] | None = None  # Pattern key mapping

    def total_ingredient_cost(self) -> dict[str, int]:
        """Calculate total ingredient cost."""
        cost: dict[str, int] = {}
        for ing in self.ingredients:
            cost[ing.item_id] = cost.get(ing.item_id, 0) + ing.count
        return cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "recipe_type": self.recipe_type.name,
            "output_item": self.output_item,
            "output_count": self.output_count,
            "ingredients": [ing.to_dict() for ing in self.ingredients],
            "pattern": self.pattern,
            "key": self.key,
            "total_cost": self.total_ingredient_cost(),
        }


@dataclass
class CraftingTestCase:
    """A single crafting test case."""

    name: str
    description: str
    recipe_id: str
    operation: str
    inputs: dict[str, Any]
    expected_result: dict[str, Any]
    expected_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "recipe_id": self.recipe_id,
            "operation": self.operation,
            "inputs": self.inputs,
            "expected_result": self.expected_result,
            "expected_error": self.expected_error,
        }


@dataclass
class CraftingTestSuite:
    """Collection of crafting test cases."""

    name: str
    test_cases: list[CraftingTestCase] = field(default_factory=list)

    def add_test(self, test: CraftingTestCase) -> None:
        self.test_cases.append(test)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "test_count": len(self.test_cases),
            "tests": [tc.to_dict() for tc in self.test_cases],
        }


# Speedrun-relevant recipes - exactly as in Minecraft
SPEEDRUN_RECIPES: dict[str, Recipe] = {
    # === Wood Processing ===
    "oak_planks": Recipe(
        recipe_id="oak_planks",
        recipe_type=RecipeType.SHAPELESS,
        output_item="oak_planks",
        output_count=4,
        ingredients=[Ingredient("oak_log", 1)],
    ),
    "stick": Recipe(
        recipe_id="stick",
        recipe_type=RecipeType.SHAPED,
        output_item="stick",
        output_count=4,
        ingredients=[Ingredient("oak_planks", 2)],
        pattern=["#", "#"],
        key={"#": "oak_planks"},
    ),
    "crafting_table": Recipe(
        recipe_id="crafting_table",
        recipe_type=RecipeType.SHAPED,
        output_item="crafting_table",
        output_count=1,
        ingredients=[Ingredient("oak_planks", 4)],
        pattern=["##", "##"],
        key={"#": "oak_planks"},
    ),
    "chest": Recipe(
        recipe_id="chest",
        recipe_type=RecipeType.SHAPED,
        output_item="chest",
        output_count=1,
        ingredients=[Ingredient("oak_planks", 8)],
        pattern=["###", "# #", "###"],
        key={"#": "oak_planks"},
    ),
    # === Tools - Wooden ===
    "wooden_pickaxe": Recipe(
        recipe_id="wooden_pickaxe",
        recipe_type=RecipeType.SHAPED,
        output_item="wooden_pickaxe",
        output_count=1,
        ingredients=[Ingredient("oak_planks", 3), Ingredient("stick", 2)],
        pattern=["###", " | ", " | "],
        key={"#": "oak_planks", "|": "stick"},
    ),
    "wooden_axe": Recipe(
        recipe_id="wooden_axe",
        recipe_type=RecipeType.SHAPED,
        output_item="wooden_axe",
        output_count=1,
        ingredients=[Ingredient("oak_planks", 3), Ingredient("stick", 2)],
        pattern=["##", "#|", " |"],
        key={"#": "oak_planks", "|": "stick"},
    ),
    "wooden_shovel": Recipe(
        recipe_id="wooden_shovel",
        recipe_type=RecipeType.SHAPED,
        output_item="wooden_shovel",
        output_count=1,
        ingredients=[Ingredient("oak_planks", 1), Ingredient("stick", 2)],
        pattern=["#", "|", "|"],
        key={"#": "oak_planks", "|": "stick"},
    ),
    "wooden_sword": Recipe(
        recipe_id="wooden_sword",
        recipe_type=RecipeType.SHAPED,
        output_item="wooden_sword",
        output_count=1,
        ingredients=[Ingredient("oak_planks", 2), Ingredient("stick", 1)],
        pattern=["#", "#", "|"],
        key={"#": "oak_planks", "|": "stick"},
    ),
    # === Tools - Stone ===
    "stone_pickaxe": Recipe(
        recipe_id="stone_pickaxe",
        recipe_type=RecipeType.SHAPED,
        output_item="stone_pickaxe",
        output_count=1,
        ingredients=[Ingredient("cobblestone", 3), Ingredient("stick", 2)],
        pattern=["###", " | ", " | "],
        key={"#": "cobblestone", "|": "stick"},
    ),
    "stone_axe": Recipe(
        recipe_id="stone_axe",
        recipe_type=RecipeType.SHAPED,
        output_item="stone_axe",
        output_count=1,
        ingredients=[Ingredient("cobblestone", 3), Ingredient("stick", 2)],
        pattern=["##", "#|", " |"],
        key={"#": "cobblestone", "|": "stick"},
    ),
    "stone_shovel": Recipe(
        recipe_id="stone_shovel",
        recipe_type=RecipeType.SHAPED,
        output_item="stone_shovel",
        output_count=1,
        ingredients=[Ingredient("cobblestone", 1), Ingredient("stick", 2)],
        pattern=["#", "|", "|"],
        key={"#": "cobblestone", "|": "stick"},
    ),
    "stone_sword": Recipe(
        recipe_id="stone_sword",
        recipe_type=RecipeType.SHAPED,
        output_item="stone_sword",
        output_count=1,
        ingredients=[Ingredient("cobblestone", 2), Ingredient("stick", 1)],
        pattern=["#", "#", "|"],
        key={"#": "cobblestone", "|": "stick"},
    ),
    # === Tools - Iron ===
    "iron_pickaxe": Recipe(
        recipe_id="iron_pickaxe",
        recipe_type=RecipeType.SHAPED,
        output_item="iron_pickaxe",
        output_count=1,
        ingredients=[Ingredient("iron_ingot", 3), Ingredient("stick", 2)],
        pattern=["###", " | ", " | "],
        key={"#": "iron_ingot", "|": "stick"},
    ),
    "iron_axe": Recipe(
        recipe_id="iron_axe",
        recipe_type=RecipeType.SHAPED,
        output_item="iron_axe",
        output_count=1,
        ingredients=[Ingredient("iron_ingot", 3), Ingredient("stick", 2)],
        pattern=["##", "#|", " |"],
        key={"#": "iron_ingot", "|": "stick"},
    ),
    "iron_shovel": Recipe(
        recipe_id="iron_shovel",
        recipe_type=RecipeType.SHAPED,
        output_item="iron_shovel",
        output_count=1,
        ingredients=[Ingredient("iron_ingot", 1), Ingredient("stick", 2)],
        pattern=["#", "|", "|"],
        key={"#": "iron_ingot", "|": "stick"},
    ),
    "iron_sword": Recipe(
        recipe_id="iron_sword",
        recipe_type=RecipeType.SHAPED,
        output_item="iron_sword",
        output_count=1,
        ingredients=[Ingredient("iron_ingot", 2), Ingredient("stick", 1)],
        pattern=["#", "#", "|"],
        key={"#": "iron_ingot", "|": "stick"},
    ),
    # === Tools - Diamond ===
    "diamond_pickaxe": Recipe(
        recipe_id="diamond_pickaxe",
        recipe_type=RecipeType.SHAPED,
        output_item="diamond_pickaxe",
        output_count=1,
        ingredients=[Ingredient("diamond", 3), Ingredient("stick", 2)],
        pattern=["###", " | ", " | "],
        key={"#": "diamond", "|": "stick"},
    ),
    "diamond_axe": Recipe(
        recipe_id="diamond_axe",
        recipe_type=RecipeType.SHAPED,
        output_item="diamond_axe",
        output_count=1,
        ingredients=[Ingredient("diamond", 3), Ingredient("stick", 2)],
        pattern=["##", "#|", " |"],
        key={"#": "diamond", "|": "stick"},
    ),
    "diamond_sword": Recipe(
        recipe_id="diamond_sword",
        recipe_type=RecipeType.SHAPED,
        output_item="diamond_sword",
        output_count=1,
        ingredients=[Ingredient("diamond", 2), Ingredient("stick", 1)],
        pattern=["#", "#", "|"],
        key={"#": "diamond", "|": "stick"},
    ),
    # === Armor - Iron ===
    "iron_helmet": Recipe(
        recipe_id="iron_helmet",
        recipe_type=RecipeType.SHAPED,
        output_item="iron_helmet",
        output_count=1,
        ingredients=[Ingredient("iron_ingot", 5)],
        pattern=["###", "# #"],
        key={"#": "iron_ingot"},
    ),
    "iron_chestplate": Recipe(
        recipe_id="iron_chestplate",
        recipe_type=RecipeType.SHAPED,
        output_item="iron_chestplate",
        output_count=1,
        ingredients=[Ingredient("iron_ingot", 8)],
        pattern=["# #", "###", "###"],
        key={"#": "iron_ingot"},
    ),
    "iron_leggings": Recipe(
        recipe_id="iron_leggings",
        recipe_type=RecipeType.SHAPED,
        output_item="iron_leggings",
        output_count=1,
        ingredients=[Ingredient("iron_ingot", 7)],
        pattern=["###", "# #", "# #"],
        key={"#": "iron_ingot"},
    ),
    "iron_boots": Recipe(
        recipe_id="iron_boots",
        recipe_type=RecipeType.SHAPED,
        output_item="iron_boots",
        output_count=1,
        ingredients=[Ingredient("iron_ingot", 4)],
        pattern=["# #", "# #"],
        key={"#": "iron_ingot"},
    ),
    # === Furnace & Smelting ===
    "furnace": Recipe(
        recipe_id="furnace",
        recipe_type=RecipeType.SHAPED,
        output_item="furnace",
        output_count=1,
        ingredients=[Ingredient("cobblestone", 8)],
        pattern=["###", "# #", "###"],
        key={"#": "cobblestone"},
    ),
    # === Nether & End Items ===
    "flint_and_steel": Recipe(
        recipe_id="flint_and_steel",
        recipe_type=RecipeType.SHAPELESS,
        output_item="flint_and_steel",
        output_count=1,
        ingredients=[Ingredient("iron_ingot", 1), Ingredient("flint", 1)],
    ),
    "bucket": Recipe(
        recipe_id="bucket",
        recipe_type=RecipeType.SHAPED,
        output_item="bucket",
        output_count=1,
        ingredients=[Ingredient("iron_ingot", 3)],
        pattern=["# #", " # "],
        key={"#": "iron_ingot"},
    ),
    "blaze_powder": Recipe(
        recipe_id="blaze_powder",
        recipe_type=RecipeType.SHAPELESS,
        output_item="blaze_powder",
        output_count=2,
        ingredients=[Ingredient("blaze_rod", 1)],
    ),
    "eye_of_ender": Recipe(
        recipe_id="eye_of_ender",
        recipe_type=RecipeType.SHAPELESS,
        output_item="eye_of_ender",
        output_count=1,
        ingredients=[Ingredient("ender_pearl", 1), Ingredient("blaze_powder", 1)],
    ),
    # === Blocks ===
    "iron_block": Recipe(
        recipe_id="iron_block",
        recipe_type=RecipeType.SHAPED,
        output_item="iron_block",
        output_count=1,
        ingredients=[Ingredient("iron_ingot", 9)],
        pattern=["###", "###", "###"],
        key={"#": "iron_ingot"},
    ),
    "gold_block": Recipe(
        recipe_id="gold_block",
        recipe_type=RecipeType.SHAPED,
        output_item="gold_block",
        output_count=1,
        ingredients=[Ingredient("gold_ingot", 9)],
        pattern=["###", "###", "###"],
        key={"#": "gold_ingot"},
    ),
    "diamond_block": Recipe(
        recipe_id="diamond_block",
        recipe_type=RecipeType.SHAPED,
        output_item="diamond_block",
        output_count=1,
        ingredients=[Ingredient("diamond", 9)],
        pattern=["###", "###", "###"],
        key={"#": "diamond"},
    ),
    # === Reverse crafting (blocks to ingots) ===
    "iron_ingot_from_block": Recipe(
        recipe_id="iron_ingot_from_block",
        recipe_type=RecipeType.SHAPELESS,
        output_item="iron_ingot",
        output_count=9,
        ingredients=[Ingredient("iron_block", 1)],
    ),
    "gold_ingot_from_block": Recipe(
        recipe_id="gold_ingot_from_block",
        recipe_type=RecipeType.SHAPELESS,
        output_item="gold_ingot",
        output_count=9,
        ingredients=[Ingredient("gold_block", 1)],
    ),
    "diamond_from_block": Recipe(
        recipe_id="diamond_from_block",
        recipe_type=RecipeType.SHAPELESS,
        output_item="diamond",
        output_count=9,
        ingredients=[Ingredient("diamond_block", 1)],
    ),
    # === Combat ===
    "bow": Recipe(
        recipe_id="bow",
        recipe_type=RecipeType.SHAPED,
        output_item="bow",
        output_count=1,
        ingredients=[Ingredient("stick", 3), Ingredient("string", 3)],
        pattern=[" #|", "# |", " #|"],
        key={"#": "stick", "|": "string"},
    ),
    "arrow": Recipe(
        recipe_id="arrow",
        recipe_type=RecipeType.SHAPED,
        output_item="arrow",
        output_count=4,
        ingredients=[Ingredient("flint", 1), Ingredient("stick", 1), Ingredient("feather", 1)],
        pattern=["#", "|", "F"],
        key={"#": "flint", "|": "stick", "F": "feather"},
    ),
    # === Misc Speedrun Items ===
    "bed": Recipe(
        recipe_id="bed",
        recipe_type=RecipeType.SHAPED,
        output_item="bed",
        output_count=1,
        ingredients=[Ingredient("wool", 3), Ingredient("oak_planks", 3)],
        pattern=["WWW", "###"],
        key={"W": "wool", "#": "oak_planks"},
    ),
    "boat": Recipe(
        recipe_id="boat",
        recipe_type=RecipeType.SHAPED,
        output_item="boat",
        output_count=1,
        ingredients=[Ingredient("oak_planks", 5)],
        pattern=["# #", "###"],
        key={"#": "oak_planks"},
    ),
}

# Smelting recipes
SMELTING_RECIPES: dict[str, Recipe] = {
    "iron_ingot": Recipe(
        recipe_id="iron_ingot_smelting",
        recipe_type=RecipeType.SMELTING,
        output_item="iron_ingot",
        output_count=1,
        ingredients=[Ingredient("iron_ore", 1)],
    ),
    "gold_ingot": Recipe(
        recipe_id="gold_ingot_smelting",
        recipe_type=RecipeType.SMELTING,
        output_item="gold_ingot",
        output_count=1,
        ingredients=[Ingredient("gold_ore", 1)],
    ),
    "glass": Recipe(
        recipe_id="glass_smelting",
        recipe_type=RecipeType.SMELTING,
        output_item="glass",
        output_count=1,
        ingredients=[Ingredient("sand", 1)],
    ),
    "stone": Recipe(
        recipe_id="stone_smelting",
        recipe_type=RecipeType.SMELTING,
        output_item="stone",
        output_count=1,
        ingredients=[Ingredient("cobblestone", 1)],
    ),
    "cooked_porkchop": Recipe(
        recipe_id="cooked_porkchop_smelting",
        recipe_type=RecipeType.SMELTING,
        output_item="cooked_porkchop",
        output_count=1,
        ingredients=[Ingredient("raw_porkchop", 1)],
    ),
}


class CraftingTestGenerator:
    """Generates test cases for crafting subsystem."""

    def __init__(
        self,
        recipes: dict[str, Recipe] | None = None,
        smelting_recipes: dict[str, Recipe] | None = None,
    ) -> None:
        """Initialize the test generator.

        Args:
            recipes: Crafting recipes to test. Defaults to SPEEDRUN_RECIPES.
            smelting_recipes: Smelting recipes to test. Defaults to SMELTING_RECIPES.
        """
        self.recipes = recipes or SPEEDRUN_RECIPES
        self.smelting_recipes = smelting_recipes or SMELTING_RECIPES

    def generate_recipe_correctness_tests(self) -> CraftingTestSuite:
        """Generate tests verifying all recipes produce correct output."""
        suite = CraftingTestSuite("Recipe Correctness")

        for recipe_id, recipe in self.recipes.items():
            # Build inventory with exact ingredients
            inventory = self._build_inventory_with_ingredients(recipe)

            suite.add_test(
                CraftingTestCase(
                    name=f"craft_{recipe_id}_correct",
                    description=f"Crafting {recipe_id} produces {recipe.output_count}x {recipe.output_item}",
                    recipe_id=recipe_id,
                    operation="craft",
                    inputs={
                        "recipe_id": recipe_id,
                        "count": 1,
                        "initial_inventory": inventory,
                        "crafting_grid": self._build_crafting_grid(recipe),
                    },
                    expected_result={
                        "success": True,
                        "output": {
                            "item_id": recipe.output_item,
                            "count": recipe.output_count,
                        },
                    },
                )
            )

        return suite

    def generate_ingredient_consumption_tests(self) -> CraftingTestSuite:
        """Generate tests verifying exact ingredient consumption."""
        suite = CraftingTestSuite("Ingredient Consumption")

        for recipe_id, recipe in self.recipes.items():
            cost = recipe.total_ingredient_cost()
            inventory = self._build_inventory_with_ingredients(recipe, extra=5)

            expected_remaining: dict[str, int] = {}
            for item_id, needed in cost.items():
                # Each ingredient slot has (needed + 5) items
                expected_remaining[item_id] = 5  # Should have 5 left after crafting

            suite.add_test(
                CraftingTestCase(
                    name=f"consume_{recipe_id}_exact",
                    description=f"Crafting {recipe_id} consumes exactly: {cost}",
                    recipe_id=recipe_id,
                    operation="craft_and_verify_consumption",
                    inputs={
                        "recipe_id": recipe_id,
                        "count": 1,
                        "initial_inventory": inventory,
                        "ingredient_cost": cost,
                    },
                    expected_result={
                        "success": True,
                        "consumed": cost,
                        "remaining_counts": expected_remaining,
                    },
                )
            )

        # Test crafting multiple times
        recipe = self.recipes["stick"]
        suite.add_test(
            CraftingTestCase(
                name="consume_multiple_crafts",
                description="Crafting 3 batches of sticks consumes 6 planks",
                recipe_id="stick",
                operation="craft_and_verify_consumption",
                inputs={
                    "recipe_id": "stick",
                    "count": 3,
                    "initial_inventory": {"oak_planks": 10},
                    "ingredient_cost": {"oak_planks": 6},
                },
                expected_result={
                    "success": True,
                    "consumed": {"oak_planks": 6},
                    "output_total": {"item_id": "stick", "count": 12},
                    "remaining_counts": {"oak_planks": 4},
                },
            )
        )

        return suite

    def generate_insufficient_ingredients_tests(self) -> CraftingTestSuite:
        """Generate tests for crafting with insufficient ingredients."""
        suite = CraftingTestSuite("Insufficient Ingredients")

        # Test each recipe with missing ingredients
        for recipe_id, recipe in self.recipes.items():
            cost = recipe.total_ingredient_cost()

            # Missing all ingredients
            suite.add_test(
                CraftingTestCase(
                    name=f"insufficient_{recipe_id}_none",
                    description=f"Cannot craft {recipe_id} with no ingredients",
                    recipe_id=recipe_id,
                    operation="craft",
                    inputs={
                        "recipe_id": recipe_id,
                        "count": 1,
                        "initial_inventory": {},
                    },
                    expected_result={"success": False},
                    expected_error="InsufficientIngredientsError",
                )
            )

            # Missing one ingredient (for multi-ingredient recipes)
            if len(cost) > 1:
                partial_inv = {}
                first_item = list(cost.keys())[0]
                for item_id, count in cost.items():
                    if item_id != first_item:
                        partial_inv[item_id] = count

                suite.add_test(
                    CraftingTestCase(
                        name=f"insufficient_{recipe_id}_partial",
                        description=f"Cannot craft {recipe_id} missing {first_item}",
                        recipe_id=recipe_id,
                        operation="craft",
                        inputs={
                            "recipe_id": recipe_id,
                            "count": 1,
                            "initial_inventory": partial_inv,
                        },
                        expected_result={"success": False},
                        expected_error="InsufficientIngredientsError",
                    )
                )

            # One short of required
            short_inv = {}
            for item_id, count in cost.items():
                short_inv[item_id] = count - 1 if count > 1 else 0

            suite.add_test(
                CraftingTestCase(
                    name=f"insufficient_{recipe_id}_short",
                    description=f"Cannot craft {recipe_id} with one less of each ingredient",
                    recipe_id=recipe_id,
                    operation="craft",
                    inputs={
                        "recipe_id": recipe_id,
                        "count": 1,
                        "initial_inventory": short_inv,
                    },
                    expected_result={"success": False},
                    expected_error="InsufficientIngredientsError",
                )
            )

        return suite

    def generate_shaped_recipe_tests(self) -> CraftingTestSuite:
        """Generate tests for shaped recipe pattern matching."""
        suite = CraftingTestSuite("Shaped Recipe Patterns")

        # Test correct patterns
        shaped_recipes = {
            k: v for k, v in self.recipes.items() if v.recipe_type == RecipeType.SHAPED
        }

        for recipe_id, recipe in shaped_recipes.items():
            if not recipe.pattern:
                continue

            inventory = self._build_inventory_with_ingredients(recipe)

            # Correct pattern
            suite.add_test(
                CraftingTestCase(
                    name=f"shaped_{recipe_id}_correct_pattern",
                    description=f"Crafting {recipe_id} with correct pattern succeeds",
                    recipe_id=recipe_id,
                    operation="craft_shaped",
                    inputs={
                        "recipe_id": recipe_id,
                        "crafting_grid": self._build_crafting_grid(recipe),
                        "initial_inventory": inventory,
                    },
                    expected_result={
                        "success": True,
                        "output": {
                            "item_id": recipe.output_item,
                            "count": recipe.output_count,
                        },
                    },
                )
            )

            # Wrong pattern (rotated 90 degrees for asymmetric recipes)
            if recipe.pattern and len(recipe.pattern) > 1:
                suite.add_test(
                    CraftingTestCase(
                        name=f"shaped_{recipe_id}_wrong_pattern",
                        description=f"Crafting {recipe_id} with wrong pattern fails",
                        recipe_id=recipe_id,
                        operation="craft_shaped",
                        inputs={
                            "recipe_id": recipe_id,
                            "crafting_grid": self._rotate_pattern(recipe),
                            "initial_inventory": inventory,
                        },
                        expected_result={"success": False},
                        expected_error="PatternMismatchError",
                    )
                )

        # Test pickaxe patterns specifically
        suite.add_test(
            CraftingTestCase(
                name="shaped_pickaxe_pattern_exact",
                description="Iron pickaxe requires exact 3-2 pattern",
                recipe_id="iron_pickaxe",
                operation="craft_shaped",
                inputs={
                    "crafting_grid": [
                        ["iron_ingot", "iron_ingot", "iron_ingot"],
                        [None, "stick", None],
                        [None, "stick", None],
                    ],
                    "initial_inventory": {"iron_ingot": 3, "stick": 2},
                },
                expected_result={
                    "success": True,
                    "output": {"item_id": "iron_pickaxe", "count": 1},
                },
            )
        )

        suite.add_test(
            CraftingTestCase(
                name="shaped_pickaxe_wrong_row",
                description="Iron pickaxe with ingots in wrong row fails",
                recipe_id="iron_pickaxe",
                operation="craft_shaped",
                inputs={
                    "crafting_grid": [
                        [None, "stick", None],
                        ["iron_ingot", "iron_ingot", "iron_ingot"],
                        [None, "stick", None],
                    ],
                    "initial_inventory": {"iron_ingot": 3, "stick": 2},
                },
                expected_result={"success": False},
                expected_error="PatternMismatchError",
            )
        )

        return suite

    def generate_shapeless_recipe_tests(self) -> CraftingTestSuite:
        """Generate tests for shapeless recipe flexibility."""
        suite = CraftingTestSuite("Shapeless Recipe Flexibility")

        shapeless_recipes = {
            k: v for k, v in self.recipes.items() if v.recipe_type == RecipeType.SHAPELESS
        }

        for recipe_id, recipe in shapeless_recipes.items():
            inventory = self._build_inventory_with_ingredients(recipe)

            # Any arrangement should work
            suite.add_test(
                CraftingTestCase(
                    name=f"shapeless_{recipe_id}_any_arrangement",
                    description=f"Shapeless recipe {recipe_id} works in any slot arrangement",
                    recipe_id=recipe_id,
                    operation="craft_shapeless",
                    inputs={
                        "recipe_id": recipe_id,
                        "ingredients_present": [ing.to_dict() for ing in recipe.ingredients],
                        "initial_inventory": inventory,
                    },
                    expected_result={
                        "success": True,
                        "output": {
                            "item_id": recipe.output_item,
                            "count": recipe.output_count,
                        },
                    },
                )
            )

        # Test flint and steel in different arrangements
        suite.add_test(
            CraftingTestCase(
                name="shapeless_flint_steel_arrangement_1",
                description="Flint and steel: iron in slot 0, flint in slot 1",
                recipe_id="flint_and_steel",
                operation="craft_shapeless",
                inputs={
                    "crafting_grid": [
                        ["iron_ingot", "flint", None],
                        [None, None, None],
                        [None, None, None],
                    ],
                    "initial_inventory": {"iron_ingot": 1, "flint": 1},
                },
                expected_result={
                    "success": True,
                    "output": {"item_id": "flint_and_steel", "count": 1},
                },
            )
        )

        suite.add_test(
            CraftingTestCase(
                name="shapeless_flint_steel_arrangement_2",
                description="Flint and steel: flint in slot 4, iron in slot 8",
                recipe_id="flint_and_steel",
                operation="craft_shapeless",
                inputs={
                    "crafting_grid": [
                        [None, None, None],
                        [None, "flint", None],
                        [None, None, "iron_ingot"],
                    ],
                    "initial_inventory": {"iron_ingot": 1, "flint": 1},
                },
                expected_result={
                    "success": True,
                    "output": {"item_id": "flint_and_steel", "count": 1},
                },
            )
        )

        # Test planks from log
        suite.add_test(
            CraftingTestCase(
                name="shapeless_planks_any_slot",
                description="Oak planks: log can be in any slot",
                recipe_id="oak_planks",
                operation="craft_shapeless",
                inputs={
                    "crafting_grid": [
                        [None, None, None],
                        [None, None, None],
                        [None, None, "oak_log"],
                    ],
                    "initial_inventory": {"oak_log": 1},
                },
                expected_result={
                    "success": True,
                    "output": {"item_id": "oak_planks", "count": 4},
                },
            )
        )

        return suite

    def generate_multi_output_tests(self) -> CraftingTestSuite:
        """Generate tests for recipes with multiple output items."""
        suite = CraftingTestSuite("Multi-Output Recipes")

        multi_output_recipes = {k: v for k, v in self.recipes.items() if v.output_count > 1}

        for recipe_id, recipe in multi_output_recipes.items():
            inventory = self._build_inventory_with_ingredients(recipe)

            suite.add_test(
                CraftingTestCase(
                    name=f"multi_output_{recipe_id}",
                    description=f"Crafting {recipe_id} produces {recipe.output_count} items",
                    recipe_id=recipe_id,
                    operation="craft",
                    inputs={
                        "recipe_id": recipe_id,
                        "count": 1,
                        "initial_inventory": inventory,
                    },
                    expected_result={
                        "success": True,
                        "output": {
                            "item_id": recipe.output_item,
                            "count": recipe.output_count,
                        },
                    },
                )
            )

        # Specific tests for common multi-output recipes
        suite.add_test(
            CraftingTestCase(
                name="multi_output_planks_4",
                description="1 oak log produces exactly 4 planks",
                recipe_id="oak_planks",
                operation="craft",
                inputs={
                    "recipe_id": "oak_planks",
                    "count": 1,
                    "initial_inventory": {"oak_log": 1},
                },
                expected_result={
                    "success": True,
                    "output": {"item_id": "oak_planks", "count": 4},
                    "remaining": {"oak_log": 0},
                },
            )
        )

        suite.add_test(
            CraftingTestCase(
                name="multi_output_sticks_4",
                description="2 planks produces exactly 4 sticks",
                recipe_id="stick",
                operation="craft",
                inputs={
                    "recipe_id": "stick",
                    "count": 1,
                    "initial_inventory": {"oak_planks": 2},
                },
                expected_result={
                    "success": True,
                    "output": {"item_id": "stick", "count": 4},
                    "remaining": {"oak_planks": 0},
                },
            )
        )

        suite.add_test(
            CraftingTestCase(
                name="multi_output_blaze_powder_2",
                description="1 blaze rod produces exactly 2 blaze powder",
                recipe_id="blaze_powder",
                operation="craft",
                inputs={
                    "recipe_id": "blaze_powder",
                    "count": 1,
                    "initial_inventory": {"blaze_rod": 1},
                },
                expected_result={
                    "success": True,
                    "output": {"item_id": "blaze_powder", "count": 2},
                    "remaining": {"blaze_rod": 0},
                },
            )
        )

        suite.add_test(
            CraftingTestCase(
                name="multi_output_arrows_4",
                description="1 flint + 1 stick + 1 feather = 4 arrows",
                recipe_id="arrow",
                operation="craft",
                inputs={
                    "recipe_id": "arrow",
                    "count": 1,
                    "initial_inventory": {"flint": 1, "stick": 1, "feather": 1},
                },
                expected_result={
                    "success": True,
                    "output": {"item_id": "arrow", "count": 4},
                    "remaining": {"flint": 0, "stick": 0, "feather": 0},
                },
            )
        )

        suite.add_test(
            CraftingTestCase(
                name="multi_output_iron_from_block_9",
                description="1 iron block produces exactly 9 iron ingots",
                recipe_id="iron_ingot_from_block",
                operation="craft",
                inputs={
                    "recipe_id": "iron_ingot_from_block",
                    "count": 1,
                    "initial_inventory": {"iron_block": 1},
                },
                expected_result={
                    "success": True,
                    "output": {"item_id": "iron_ingot", "count": 9},
                    "remaining": {"iron_block": 0},
                },
            )
        )

        return suite

    def generate_smelting_tests(self) -> CraftingTestSuite:
        """Generate tests for furnace smelting recipes."""
        suite = CraftingTestSuite("Smelting Recipes")

        for recipe_id, recipe in self.smelting_recipes.items():
            suite.add_test(
                CraftingTestCase(
                    name=f"smelt_{recipe_id}",
                    description=f"Smelting produces {recipe.output_count}x {recipe.output_item}",
                    recipe_id=recipe_id,
                    operation="smelt",
                    inputs={
                        "recipe_id": recipe_id,
                        "input_item": recipe.ingredients[0].item_id,
                        "fuel": "coal",
                        "initial_inventory": {
                            recipe.ingredients[0].item_id: 1,
                            "coal": 1,
                        },
                    },
                    expected_result={
                        "success": True,
                        "output": {
                            "item_id": recipe.output_item,
                            "count": recipe.output_count,
                        },
                    },
                )
            )

        # Test batch smelting
        suite.add_test(
            CraftingTestCase(
                name="smelt_batch_iron",
                description="Smelt 8 iron ore produces 8 iron ingots",
                recipe_id="iron_ingot_smelting",
                operation="smelt_batch",
                inputs={
                    "recipe_id": "iron_ingot_smelting",
                    "count": 8,
                    "initial_inventory": {"iron_ore": 8, "coal": 1},
                },
                expected_result={
                    "success": True,
                    "output": {"item_id": "iron_ingot", "count": 8},
                    "ore_consumed": 8,
                },
            )
        )

        return suite

    def generate_all_tests(self) -> dict[str, CraftingTestSuite]:
        """Generate all test suites."""
        return {
            "recipe_correctness": self.generate_recipe_correctness_tests(),
            "ingredient_consumption": self.generate_ingredient_consumption_tests(),
            "insufficient_ingredients": self.generate_insufficient_ingredients_tests(),
            "shaped_patterns": self.generate_shaped_recipe_tests(),
            "shapeless_flexibility": self.generate_shapeless_recipe_tests(),
            "multi_output": self.generate_multi_output_tests(),
            "smelting": self.generate_smelting_tests(),
        }

    def export_tests(self, filepath: str) -> None:
        """Export all tests to JSON file."""
        all_suites = self.generate_all_tests()
        export_data = {suite_name: suite.to_dict() for suite_name, suite in all_suites.items()}

        # Also export recipe definitions
        export_data["recipes"] = {
            recipe_id: recipe.to_dict() for recipe_id, recipe in self.recipes.items()
        }
        export_data["smelting_recipes"] = {
            recipe_id: recipe.to_dict() for recipe_id, recipe in self.smelting_recipes.items()
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

    def get_test_count(self) -> dict[str, int]:
        """Get count of tests per suite."""
        all_suites = self.generate_all_tests()
        return {name: len(suite.test_cases) for name, suite in all_suites.items()}

    def _build_inventory_with_ingredients(self, recipe: Recipe, extra: int = 0) -> dict[str, int]:
        """Build inventory dict with required ingredients."""
        inv: dict[str, int] = {}
        for ing in recipe.ingredients:
            inv[ing.item_id] = inv.get(ing.item_id, 0) + ing.count + extra
        return inv

    def _build_crafting_grid(self, recipe: Recipe) -> list[list[str | None]]:
        """Build a 3x3 crafting grid from recipe pattern."""
        grid: list[list[str | None]] = [[None, None, None] for _ in range(3)]

        if not recipe.pattern or not recipe.key:
            # Shapeless - just put ingredients in order
            slot = 0
            for ing in recipe.ingredients:
                for _ in range(ing.count):
                    row = slot // 3
                    col = slot % 3
                    if row < 3:
                        grid[row][col] = ing.item_id
                    slot += 1
            return grid

        # Shaped - follow pattern
        for row_idx, row in enumerate(recipe.pattern):
            for col_idx, char in enumerate(row):
                if char != " " and char in recipe.key:
                    grid[row_idx][col_idx] = recipe.key[char]

        return grid

    def _rotate_pattern(self, recipe: Recipe) -> list[list[str | None]]:
        """Rotate a crafting grid 90 degrees for testing wrong patterns."""
        original = self._build_crafting_grid(recipe)
        # Rotate 90 degrees clockwise
        rotated: list[list[str | None]] = [[None, None, None] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                rotated[j][2 - i] = original[i][j]
        return rotated


def main() -> None:
    """Run test generation and display summary."""
    generator = CraftingTestGenerator()

    print("Crafting Test Generator")
    print("=" * 50)

    test_counts = generator.get_test_count()
    total = sum(test_counts.values())

    for suite_name, count in test_counts.items():
        print(f"  {suite_name}: {count} tests")

    print(f"\nTotal: {total} tests")
    print(f"Recipes: {len(generator.recipes)}")
    print(f"Smelting recipes: {len(generator.smelting_recipes)}")

    # Export to JSON
    generator.export_tests("crafting_tests.json")
    print("\nTests exported to crafting_tests.json")


if __name__ == "__main__":
    main()
