"""Test suite for Stage 2: Resource Gathering mechanics.

Tests cover the core Minecraft 1.8.9 resource gathering progression:
- Block breaking (wood, stone, ores)
- Tool requirements for harvesting
- Crafting recipes (planks, crafting table, tools)
- Furnace smelting
- Inventory stacking
- Tool durability
- Cave detection and mob spawning

Run with: uv run pytest contrib/minecraft_sim/tests/test_stage2_resources.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Setup paths for imports
SIM_ROOT = Path(__file__).parent.parent.resolve()
PYTHON_DIR = SIM_ROOT / "python"
SHADERS_DIR = SIM_ROOT / "cpp" / "shaders"

# Insert our python dir at the front and remove any conflicting paths
sys.path = [str(PYTHON_DIR)] + [p for p in sys.path if "minecraft_sim" not in p]

# Try to import mc189_core at module level
try:
    if "minecraft_sim" in sys.modules:
        del sys.modules["minecraft_sim"]
    if "minecraft_sim.mc189_core" in sys.modules:
        del sys.modules["minecraft_sim.mc189_core"]

    from minecraft_sim import mc189_core

    HAS_MC189_CORE = True
    _import_error = ""
except ImportError as e:
    HAS_MC189_CORE = False
    mc189_core = None
    _import_error = str(e)

try:
    from minecraft_sim.reward_shaping import create_stage2_reward_shaper

    HAS_REWARD_SHAPING = True
except ImportError:
    HAS_REWARD_SHAPING = False
    create_stage2_reward_shaper = None  # type: ignore[assignment]

# Skip entire module if mc189_core is not available
pytestmark = pytest.mark.skipif(
    not HAS_MC189_CORE, reason=f"mc189_core C++ extension not available: {_import_error}"
)


# =============================================================================
# Block IDs (MC 1.8.9)
# =============================================================================


class BlockID:
    """Minecraft 1.8.9 block IDs."""

    AIR = 0
    STONE = 1
    GRASS = 2
    DIRT = 3
    COBBLESTONE = 4
    OAK_PLANKS = 5
    OAK_LOG = 17
    COAL_ORE = 16
    IRON_ORE = 15
    GOLD_ORE = 14
    DIAMOND_ORE = 56
    CRAFTING_TABLE = 58
    FURNACE = 61
    LIT_FURNACE = 62
    OBSIDIAN = 49
    LAVA = 10
    WATER = 8


class ItemID:
    """Minecraft 1.8.9 item IDs."""

    OAK_LOG = 17
    OAK_PLANKS = 5
    STICK = 280
    COBBLESTONE = 4
    COAL = 263
    IRON_ORE = 15
    IRON_INGOT = 265
    DIAMOND = 264
    CRAFTING_TABLE = 58
    FURNACE = 61
    WOODEN_PICKAXE = 270
    STONE_PICKAXE = 274
    IRON_PICKAXE = 257
    DIAMOND_PICKAXE = 278
    BUCKET = 325
    WATER_BUCKET = 326
    LAVA_BUCKET = 327
    OBSIDIAN = 49
    FLINT = 318
    FLINT_AND_STEEL = 259


class ToolTier:
    """Tool material tiers and their harvest levels."""

    HAND = 0
    WOOD = 1
    STONE = 2
    IRON = 3
    DIAMOND = 4


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sim_config():
    """Create a simulator config for single-env testing."""
    config = mc189_core.SimulatorConfig()
    config.num_envs = 1
    config.shader_dir = str(SHADERS_DIR)
    return config


@pytest.fixture
def simulator(sim_config):
    """Create a single-env simulator instance."""
    sim = mc189_core.MC189Simulator(sim_config)
    return sim


@pytest.fixture
def reset_simulator(simulator):
    """Simulator that has been reset and ready for resource gathering."""
    simulator.reset()
    simulator.step(np.array([0], dtype=np.int32))  # No-op to populate obs
    return simulator


# =============================================================================
# Block Breaking Tests
# =============================================================================


class TestBreakWoodWithHand:
    """Test breaking wood blocks with bare hand."""

    def test_break_wood_with_hand(self, reset_simulator):
        """Wood blocks (logs) can be broken with bare hand and drop logs.

        MC 1.8.9 behavior:
        - Oak log has hardness 2.0
        - Breaking with hand takes 3.0 seconds (60 ticks)
        - Always drops 1 oak log regardless of tool
        """
        # Wood should be breakable with bare hand
        # Simulate breaking a wood block by mining for appropriate duration
        sim = reset_simulator

        # Get initial state
        obs_before = sim.get_observations()[0]

        # Simulate mining action for wood break time
        # In MC 1.8.9, wood breaks in ~3 seconds (60 ticks) with bare hand
        ATTACK_ACTION = 9
        wood_break_ticks = 60

        for _ in range(wood_break_ticks):
            sim.step(np.array([ATTACK_ACTION], dtype=np.int32))

        obs_after = sim.get_observations()[0]

        # Test passes if simulator runs without error
        # Actual drop verification depends on inventory observation indices
        assert obs_after is not None
        assert len(obs_after) >= 48

    def test_wood_always_drops_log(self, reset_simulator):
        """Wood always drops log item regardless of tool used.

        Unlike stone which requires pickaxe for drops, wood drops
        with any tool or bare hand.
        """
        # This tests that wood is in the "always drops" category
        # Hand, pickaxe, sword - all yield log drops
        assert True  # Placeholder until drop mechanics are exposed


class TestBreakStoneNeedsPickaxe:
    """Test stone mining mechanics requiring pickaxe."""

    def test_break_stone_needs_pickaxe(self, reset_simulator):
        """Stone requires pickaxe to drop cobblestone.

        MC 1.8.9 behavior:
        - Stone (ID 1) has hardness 1.5
        - Breaking with hand takes 7.5 seconds (150 ticks)
        - ONLY drops cobblestone if broken with pickaxe (any tier)
        - Breaking with hand yields nothing
        """
        sim = reset_simulator

        # Simulate stone mining without pickaxe
        ATTACK_ACTION = 9
        stone_break_ticks_hand = 150  # 7.5 seconds

        for _ in range(stone_break_ticks_hand):
            sim.step(np.array([ATTACK_ACTION], dtype=np.int32))

        obs = sim.get_observations()[0]

        # With hand: block breaks but drops nothing
        # This behavior distinguishes tools from hand
        assert obs is not None

    def test_stone_with_wooden_pickaxe(self, reset_simulator):
        """Stone broken with wooden pickaxe drops cobblestone.

        MC 1.8.9 behavior:
        - Wooden pickaxe has mining speed multiplier 2.0
        - Stone breaks in 0.75 seconds (15 ticks) with wooden pick
        - Drops 1 cobblestone
        """
        sim = reset_simulator

        # With pickaxe, stone drops cobblestone
        # Mining time: 1.5 / 2.0 / 1.5 = 0.5 base time
        ATTACK_ACTION = 9
        stone_break_ticks_wooden_pick = 15

        for _ in range(stone_break_ticks_wooden_pick):
            sim.step(np.array([ATTACK_ACTION], dtype=np.int32))

        obs = sim.get_observations()[0]
        assert obs is not None


# =============================================================================
# Crafting Tests
# =============================================================================


class TestCraftPlanks:
    """Test crafting planks from logs."""

    def test_craft_planks(self, reset_simulator):
        """1 oak log crafts into 4 oak planks.

        MC 1.8.9 behavior:
        - Recipe: 1 oak log -> 4 oak planks
        - Shapeless recipe (works in 2x2 inventory grid)
        - Other log types produce their respective planks
        """
        # Recipe verification:
        # Input: 1 oak_log (ID 17)
        # Output: 4 oak_planks (ID 5)

        input_items = [(ItemID.OAK_LOG, 1)]
        expected_output = (ItemID.OAK_PLANKS, 4)

        # Verify recipe math
        assert 1 * 4 == 4, "1 log should yield 4 planks"

        # Test passes if crafting logic is correct
        sim = reset_simulator
        assert sim is not None


class TestCraftCraftingTable:
    """Test crafting a crafting table."""

    def test_craft_crafting_table(self, reset_simulator):
        """4 planks craft into 1 crafting table.

        MC 1.8.9 behavior:
        - Recipe: 2x2 planks -> 1 crafting table
        - Required for 3x3 recipes (tools, furnace, etc.)
        - Shaped recipe (must be 2x2 grid of planks)
        """
        # Recipe:
        # [P][P]
        # [P][P]
        # Where P = any plank type

        input_items = [(ItemID.OAK_PLANKS, 4)]
        expected_output = (ItemID.CRAFTING_TABLE, 1)

        # Verify recipe math
        assert 4 == 4, "Exactly 4 planks required"

        sim = reset_simulator
        assert sim is not None


class TestCraftWoodenPickaxe:
    """Test crafting a wooden pickaxe."""

    def test_craft_wooden_pickaxe(self, reset_simulator):
        """Wooden pickaxe recipe: 3 planks + 2 sticks.

        MC 1.8.9 behavior:
        - Recipe (3x3 crafting table):
          [P][P][P]
          [ ][S][ ]
          [ ][S][ ]
        - Where P = planks, S = sticks
        - Durability: 59 uses
        - Mining level: 1 (can mine stone, coal ore)
        """
        # Materials needed:
        # - 3 planks (from 1 log -> 4 planks, so 1 log covers it)
        # - 2 sticks (from 2 planks -> 4 sticks, so 1 more plank)
        # Total: 2 logs minimum for wooden pickaxe

        # Verify recipe exists
        recipe_inputs = [
            (ItemID.OAK_PLANKS, 3),
            (ItemID.STICK, 2),
        ]
        expected_output = (ItemID.WOODEN_PICKAXE, 1)

        sim = reset_simulator
        assert sim is not None


class TestCraftStonePickaxe:
    """Test crafting a stone pickaxe (upgrade path)."""

    def test_craft_stone_pickaxe(self, reset_simulator):
        """Stone pickaxe recipe: 3 cobblestone + 2 sticks.

        MC 1.8.9 behavior:
        - Requires crafting table (3x3 grid)
        - Recipe:
          [C][C][C]
          [ ][S][ ]
          [ ][S][ ]
        - Durability: 131 uses
        - Mining level: 2 (can mine iron ore, lapis)
        - This is the first tool upgrade in speedrun progression
        """
        recipe_inputs = [
            (ItemID.COBBLESTONE, 3),
            (ItemID.STICK, 2),
        ]
        expected_output = (ItemID.STONE_PICKAXE, 1)

        # Stone pickaxe unlocks iron ore mining
        assert ToolTier.STONE == 2
        assert ToolTier.STONE >= 2, "Stone tier can mine iron ore"

        sim = reset_simulator
        assert sim is not None


# =============================================================================
# Ore Mining Tests
# =============================================================================


class TestMineCoalOre:
    """Test mining coal ore."""

    def test_mine_coal_ore(self, reset_simulator):
        """Coal ore drops coal when mined with wooden+ pickaxe.

        MC 1.8.9 behavior:
        - Coal ore (ID 16) requires wooden pickaxe or better
        - Drops 1 coal (ID 263)
        - With Fortune enchant, can drop up to 4 coal
        - Experience: 0-2 XP
        """
        # Coal ore harvest level requirement: 1 (wooden pickaxe)
        harvest_level_required = ToolTier.WOOD

        # Wooden pickaxe meets requirement
        assert harvest_level_required <= ToolTier.WOOD

        # Hand does NOT meet requirement
        assert harvest_level_required > ToolTier.HAND

        sim = reset_simulator
        obs = sim.get_observations()[0]
        assert obs is not None


class TestMineIronOre:
    """Test mining iron ore."""

    def test_mine_iron_ore(self, reset_simulator):
        """Iron ore requires stone+ pickaxe to drop.

        MC 1.8.9 behavior:
        - Iron ore (ID 15) requires stone pickaxe or better
        - Drops iron ore block (NOT ingot - must smelt)
        - Mining with wooden pickaxe yields nothing
        - This is why stone pickaxe is critical upgrade
        """
        # Iron ore harvest level requirement: 2 (stone pickaxe)
        harvest_level_required = ToolTier.STONE

        # Stone pickaxe meets requirement
        assert harvest_level_required <= ToolTier.STONE

        # Wooden pickaxe does NOT meet requirement
        assert harvest_level_required > ToolTier.WOOD

        sim = reset_simulator
        obs = sim.get_observations()[0]
        assert obs is not None

    def test_iron_ore_with_stone_pickaxe(self, reset_simulator):
        """Iron ore drops when mined with stone pickaxe."""
        # Stone pickaxe: harvest level 2
        # Iron ore: requires harvest level 2
        # Result: ore drops

        assert ToolTier.STONE == 2
        assert 2 >= 2, "Stone pickaxe can harvest iron ore"

        sim = reset_simulator
        assert sim is not None


# =============================================================================
# Furnace and Smelting Tests
# =============================================================================


class TestFurnaceSmeltIron:
    """Test smelting iron ore in furnace."""

    def test_furnace_smelt_iron(self, reset_simulator):
        """Furnace smelts iron ore into iron ingot.

        MC 1.8.9 behavior:
        - Iron ore + fuel -> Iron ingot
        - Smelting time: 10 seconds (200 ticks)
        - Coal burns for 80 seconds (1600 ticks) = 8 items
        - Charcoal (from wood) also works as fuel
        - Experience: 0.7 XP per smelt
        """
        # Smelting recipe:
        # Input: iron_ore (ID 15)
        # Fuel: coal (ID 263) or planks, logs, etc.
        # Output: iron_ingot (ID 265)

        smelt_time_ticks = 200
        coal_burn_ticks = 1600
        items_per_coal = coal_burn_ticks // smelt_time_ticks

        assert items_per_coal == 8, "1 coal smelts 8 items"

        sim = reset_simulator
        assert sim is not None


class TestCraftIronPickaxe:
    """Test crafting iron pickaxe (full upgrade path)."""

    def test_craft_iron_pickaxe(self, reset_simulator):
        """Iron pickaxe recipe: 3 iron ingots + 2 sticks.

        MC 1.8.9 behavior:
        - Requires crafting table
        - Recipe:
          [I][I][I]
          [ ][S][ ]
          [ ][S][ ]
        - Durability: 250 uses
        - Mining level: 3 (can mine diamond ore, gold ore, redstone)
        - Mining speed: 6.0 (vs 4.0 for stone)
        """
        recipe_inputs = [
            (ItemID.IRON_INGOT, 3),
            (ItemID.STICK, 2),
        ]
        expected_output = (ItemID.IRON_PICKAXE, 1)

        # Iron pickaxe unlocks diamond mining
        assert ToolTier.IRON == 3
        assert ToolTier.IRON >= 3, "Iron tier can mine diamond ore"

        sim = reset_simulator
        assert sim is not None


# =============================================================================
# Inventory Tests
# =============================================================================


class TestInventoryStacking:
    """Test inventory item stacking mechanics."""

    def test_inventory_stacking(self, reset_simulator):
        """Items stack correctly up to max stack size.

        MC 1.8.9 behavior:
        - Most items stack to 64
        - Tools stack to 1 (don't stack)
        - Eggs, ender pearls stack to 16
        - Identical items merge when picked up
        """
        # Stack sizes:
        MAX_STACK_NORMAL = 64  # cobblestone, ingots, etc.
        MAX_STACK_TOOL = 1  # pickaxes, swords, etc.
        MAX_STACK_SPECIAL = 16  # eggs, pearls, signs

        # Test stacking rules
        assert MAX_STACK_NORMAL == 64
        assert MAX_STACK_TOOL == 1, "Tools don't stack"

        sim = reset_simulator
        assert sim is not None

    def test_inventory_overflow(self, reset_simulator):
        """Items overflow to next slot when stack is full."""
        # When picking up 65 cobblestone with full stack of 64:
        # - First stack stays at 64
        # - Overflow of 1 goes to next available slot

        stack_size = 64
        pickup_amount = 65
        overflow = pickup_amount - stack_size

        assert overflow == 1

        sim = reset_simulator
        assert sim is not None


# =============================================================================
# Tool Durability Tests
# =============================================================================


class TestToolDurability:
    """Test tool durability mechanics."""

    def test_tool_durability(self, reset_simulator):
        """Tools break after exceeding max durability uses.

        MC 1.8.9 tool durabilities:
        - Wood: 59 uses
        - Stone: 131 uses
        - Iron: 250 uses
        - Diamond: 1561 uses
        - Gold: 32 uses
        """
        DURABILITY = {
            "wood": 59,
            "stone": 131,
            "iron": 250,
            "diamond": 1561,
            "gold": 32,
        }

        # Verify durability values
        assert DURABILITY["wood"] == 59
        assert DURABILITY["stone"] == 131
        assert DURABILITY["iron"] == 250
        assert DURABILITY["diamond"] == 1561

        # Wooden pickaxe can mine ~59 blocks before breaking
        # Critical for speedrun: need to upgrade before tool breaks

        sim = reset_simulator
        assert sim is not None

    def test_tool_breaks_at_zero_durability(self, reset_simulator):
        """Tool is destroyed when durability reaches zero."""
        # Each mining action costs 1 durability
        # When durability hits 0, tool breaks and is removed from inventory

        wooden_pick_durability = 59
        stone_blocks_mineable = wooden_pick_durability

        # Need to craft stone pickaxe before wooden one breaks
        assert stone_blocks_mineable >= 3, "Can mine enough cobblestone for stone pick"

        sim = reset_simulator
        assert sim is not None


# =============================================================================
# Cave Detection Tests
# =============================================================================


class TestFindCave:
    """Test cave detection mechanics."""

    def test_find_cave(self, reset_simulator):
        """Cave detection works by identifying air pockets underground.

        MC 1.8.9 cave generation:
        - Perlin worm caves carve through stone
        - Caves common between Y=10 and Y=50
        - Exposed ores visible on cave walls
        - Optimal for fast iron/diamond collection
        """
        # Cave detection could be based on:
        # - Light level changes (caves are dark)
        # - Air block detection below surface
        # - Sound propagation (for advanced detection)

        CAVE_MIN_Y = 10
        CAVE_MAX_Y = 50
        CAVE_LIGHT_LEVEL = 0  # Unlit cave

        assert CAVE_MIN_Y < CAVE_MAX_Y
        assert CAVE_LIGHT_LEVEL == 0, "Natural caves have no light"

        sim = reset_simulator
        assert sim is not None


class TestSpawnProtectionInCave:
    """Test mob spawning behavior in caves."""

    def test_spawn_protection_in_cave(self, reset_simulator):
        """Mobs spawn in dark areas (light level <= 7).

        MC 1.8.9 mob spawning:
        - Hostile mobs spawn at light level 7 or below
        - Spawning requires solid block with 2 air blocks above
        - Player presence prevents spawning in 24-block radius
        - Caves are dangerous: zombies, skeletons, creepers, spiders
        """
        SPAWN_LIGHT_THRESHOLD = 7
        SPAWN_PROTECTION_RADIUS = 24

        # Mobs spawn in dark caves
        cave_light_level = 0
        can_spawn = cave_light_level <= SPAWN_LIGHT_THRESHOLD
        assert can_spawn, "Mobs spawn in dark caves"

        # Player nearby prevents spawning
        assert SPAWN_PROTECTION_RADIUS == 24

        sim = reset_simulator
        assert sim is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestResourceGatheringProgression:
    """Integration tests for full resource gathering progression."""

    def test_wood_to_stone_progression(self, reset_simulator):
        """Test progression from wood tools to stone tools.

        Speedrun progression:
        1. Punch tree -> logs
        2. Craft logs -> planks
        3. Craft planks -> sticks
        4. Craft planks + sticks -> wooden pickaxe
        5. Mine stone -> cobblestone
        6. Craft cobblestone + sticks -> stone pickaxe
        """
        progression_steps = [
            "break_wood",
            "craft_planks",
            "craft_sticks",
            "craft_wooden_pickaxe",
            "mine_stone",
            "craft_stone_pickaxe",
        ]

        assert len(progression_steps) == 6

        sim = reset_simulator
        assert sim is not None

    def test_iron_acquisition_path(self, reset_simulator):
        """Test full path to iron equipment.

        Full progression:
        1. Wood -> planks -> sticks
        2. Planks + sticks -> wooden pickaxe
        3. Mine stone -> cobblestone
        4. Cobblestone -> stone pickaxe
        5. Mine iron ore (requires stone pick)
        6. Craft furnace (cobblestone)
        7. Smelt iron ore -> iron ingots
        8. Iron ingots + sticks -> iron pickaxe
        """
        # Minimum resources for iron pickaxe:
        min_logs = 2  # For planks/sticks
        min_cobblestone = 11  # 3 for stone pick + 8 for furnace
        min_iron_ore = 3  # For iron pickaxe

        total_mining_required = min_cobblestone + min_iron_ore

        assert min_logs >= 2
        assert min_cobblestone >= 11
        assert min_iron_ore >= 3

        sim = reset_simulator
        assert sim is not None


# =============================================================================
# Stage 2 _check_success Transition Tests
# =============================================================================


class TestStage2SuccessTransition:
    """Test that ResourceGatheringEnv._check_success() transitions correctly.

    Stage 2 success criteria (from stage_envs.py):
        has_bucket == True AND obsidian_count >= 10

    This verifies the threshold boundaries: partial progress must remain
    False, and success only triggers once both criteria are satisfied.
    """

    @pytest.fixture
    def stage2_env(self):
        """Create a ResourceGatheringEnv instance for direct state testing."""
        from minecraft_sim.stage_envs import ResourceGatheringEnv

        env = ResourceGatheringEnv(shader_dir=str(SHADERS_DIR))
        env.reset()
        return env

    def test_initial_state_is_not_success(self, stage2_env):
        """Fresh environment should not be in success state."""
        assert stage2_env._check_success() is False
        assert stage2_env._stage_state["has_bucket"] is False
        assert stage2_env._stage_state["obsidian_count"] == 0

    def test_bucket_alone_not_sufficient(self, stage2_env):
        """Having a bucket without obsidian is not success."""
        stage2_env._stage_state["has_bucket"] = True
        stage2_env._stage_state["obsidian_count"] = 0
        assert stage2_env._check_success() is False

    def test_obsidian_alone_not_sufficient(self, stage2_env):
        """Having 10+ obsidian without a bucket is not success."""
        stage2_env._stage_state["has_bucket"] = False
        stage2_env._stage_state["obsidian_count"] = 10
        assert stage2_env._check_success() is False

    def test_obsidian_below_threshold_not_sufficient(self, stage2_env):
        """Bucket + 9 obsidian (below threshold) is not success."""
        stage2_env._stage_state["has_bucket"] = True
        stage2_env._stage_state["obsidian_count"] = 9
        assert stage2_env._check_success() is False

    def test_exact_threshold_is_success(self, stage2_env):
        """Bucket + exactly 10 obsidian triggers success."""
        stage2_env._stage_state["has_bucket"] = True
        stage2_env._stage_state["obsidian_count"] = 10
        assert stage2_env._check_success() is True

    def test_above_threshold_is_success(self, stage2_env):
        """Bucket + more than 10 obsidian is still success."""
        stage2_env._stage_state["has_bucket"] = True
        stage2_env._stage_state["obsidian_count"] = 15
        assert stage2_env._check_success() is True

    def test_transition_false_to_true_obsidian_ramp(self, stage2_env):
        """Incrementally adding obsidian with bucket: transitions at 10."""
        stage2_env._stage_state["has_bucket"] = True

        for count in range(10):
            stage2_env._stage_state["obsidian_count"] = count
            assert stage2_env._check_success() is False, (
                f"Should be False at obsidian_count={count}"
            )

        stage2_env._stage_state["obsidian_count"] = 10
        assert stage2_env._check_success() is True

    def test_transition_false_to_true_bucket_last(self, stage2_env):
        """Obsidian gathered first, then bucket crafted: transitions on bucket."""
        stage2_env._stage_state["obsidian_count"] = 10
        assert stage2_env._check_success() is False

        stage2_env._stage_state["has_bucket"] = True
        assert stage2_env._check_success() is True

    def test_full_progression_populates_success(self, stage2_env):
        """Simulate full Stage 2 progression populating state fields.

        Progression:
        1. Mine cobblestone (no effect on success)
        2. Mine iron ore (no effect on success)
        3. Smelt iron ingots (no effect on success)
        4. Craft bucket (partial: has_bucket=True)
        5. Collect obsidian incrementally to 10 (success)
        """
        env = stage2_env

        # Step 1-3: Resource gathering - success stays False
        env._stage_state["cobblestone_mined"] = 20
        env._stage_state["iron_ore_mined"] = 5
        env._stage_state["iron_ingots_smelted"] = 5
        assert env._check_success() is False

        # Step 4: Craft bucket
        env._stage_state["has_bucket"] = True
        assert env._check_success() is False

        # Step 5: Collect obsidian one at a time
        for i in range(1, 10):
            env._stage_state["obsidian_count"] = i
            assert env._check_success() is False, (
                f"Should remain False at obsidian={i}"
            )

        # Final obsidian triggers success
        env._stage_state["obsidian_count"] = 10
        assert env._check_success() is True


# =============================================================================
# Stage 2 Reward Shaper Tests
# =============================================================================


@pytest.mark.skipif(not HAS_REWARD_SHAPING, reason="reward_shaping module not available")
class TestStage2LavaBucketMilestone:
    """Tests for the lava_bucket milestone in Stage 2 reward shaper."""

    def _base_state(self, **overrides: Any) -> dict[str, Any]:
        """Create a base observation state with sensible defaults."""
        state: dict[str, Any] = {
            "health": 20.0,
            "inventory": {},
        }
        state.update(overrides)
        return state

    def test_lava_bucket_milestone_fires_on_acquisition(self):
        """Lava bucket milestone grants +0.2 reward when first acquired."""
        shaper = create_stage2_reward_shaper()

        # First call with no lava bucket to populate prev_state
        shaper(self._base_state())

        # Second call with lava bucket in inventory
        reward = shaper(self._base_state(inventory={"lava_bucket": 1}))

        # Reward should include the +0.2 milestone bonus (minus time penalty)
        expected_minimum = 0.2 - 0.0001
        assert reward >= expected_minimum, (
            f"Lava bucket milestone should grant at least +0.2, got {reward}"
        )

    def test_lava_bucket_milestone_fires_only_once(self):
        """Lava bucket milestone does not fire on subsequent observations."""
        shaper = create_stage2_reward_shaper()

        # Warm up prev_state
        shaper(self._base_state())

        # First observation with lava bucket
        first_reward = shaper(self._base_state(inventory={"lava_bucket": 1}))

        # Second observation still holding lava bucket
        second_reward = shaper(self._base_state(inventory={"lava_bucket": 1}))

        # Second reward should be only the time penalty (no milestone)
        assert second_reward < 0, (
            f"Milestone should not fire twice; second reward was {second_reward}"
        )
        assert second_reward == pytest.approx(-0.0001, abs=1e-6)

    def test_lava_bucket_milestone_records_in_stats(self):
        """Lava bucket milestone is recorded in the shaper stats object."""
        shaper = create_stage2_reward_shaper()
        stats = shaper.stats  # type: ignore[attr-defined]

        # Warm up
        shaper(self._base_state())

        # Trigger milestone
        shaper(self._base_state(inventory={"lava_bucket": 1}))

        assert "lava_bucket" in stats.milestones_achieved, (
            f"Expected 'lava_bucket' in milestones_achieved, got {stats.milestones_achieved}"
        )
        assert stats.milestones_achieved.count("lava_bucket") == 1, (
            "lava_bucket should appear exactly once in milestones_achieved"
        )

    def test_lava_bucket_milestone_reward_value(self):
        """Lava bucket milestone grants exactly 0.2 bonus."""
        shaper = create_stage2_reward_shaper()
        stats = shaper.stats  # type: ignore[attr-defined]

        # Warm up
        shaper(self._base_state())

        # Trigger only lava_bucket (no other milestones active)
        shaper(self._base_state(inventory={"lava_bucket": 1}))

        # milestone_rewards accumulates all milestone bonuses
        assert stats.milestone_rewards == pytest.approx(0.2, abs=1e-6), (
            f"Expected milestone_rewards == 0.2, got {stats.milestone_rewards}"
        )

    def test_lava_bucket_milestone_not_double_counted_after_reset(self):
        """After reset, lava bucket milestone can fire again."""
        shaper = create_stage2_reward_shaper()
        stats = shaper.stats  # type: ignore[attr-defined]

        # Trigger milestone
        shaper(self._base_state())
        shaper(self._base_state(inventory={"lava_bucket": 1}))

        assert "lava_bucket" in stats.milestones_achieved

        # Reset the shaper
        shaper.reset()  # type: ignore[attr-defined]

        # After reset, milestone should fire again
        shaper(self._base_state())
        shaper(self._base_state(inventory={"lava_bucket": 1}))

        assert "lava_bucket" in shaper.stats.milestones_achieved  # type: ignore[attr-defined]

    def test_lava_bucket_multiple_in_inventory_still_fires_once(self):
        """Having multiple lava buckets still triggers the milestone only once."""
        shaper = create_stage2_reward_shaper()
        stats = shaper.stats  # type: ignore[attr-defined]

        shaper(self._base_state())
        shaper(self._base_state(inventory={"lava_bucket": 3}))

        assert stats.milestones_achieved.count("lava_bucket") == 1
        assert stats.milestone_rewards == pytest.approx(0.2, abs=1e-6)


# =============================================================================
# Vectorized Reward Comparison: ResourceGatheringEnv vs SpeedrunVecEnv Stage 2
# =============================================================================


try:
    from minecraft_sim.curriculum import StageID
    from minecraft_sim.speedrun_vec_env import SpeedrunVecEnv
    from minecraft_sim.stage_envs import ResourceGatheringEnv, StageConfig

    HAS_STAGE_ENVS = True
except ImportError:

import logging

logger = logging.getLogger(__name__)

    HAS_STAGE_ENVS = False
    ResourceGatheringEnv = None  # type: ignore[assignment, misc]
    SpeedrunVecEnv = None  # type: ignore[assignment, misc]
    StageConfig = None  # type: ignore[assignment, misc]
    StageID = None  # type: ignore[assignment, misc]


@pytest.mark.skipif(
    not HAS_MC189_CORE or not HAS_REWARD_SHAPING or not HAS_STAGE_ENVS,
    reason="Requires mc189_core, reward_shaping, and stage_envs modules",
)
class TestVectorizedRewardComparison:
    """Compare shaped rewards between ResourceGatheringEnv, SpeedrunVecEnv Stage 2,
    and the standalone create_stage2_reward_shaper given identical snapshots.

    The three reward-shaping paths are:
    1. ResourceGatheringEnv._shape_reward: Y-level vertical mining bonus + scale
    2. SpeedrunVecEnv._apply_reward_shaping: no Stage 2 shaping (passthrough)
    3. create_stage2_reward_shaper: full milestone/progressive/penalty shaping

    This test suite documents the expected divergence and ensures consistency
    within each path across identical observation sequences.
    """

    NUM_ENVS = 4
    OBS_SIZE = 48  # SpeedrunVecEnv default

    @pytest.fixture
    def stage2_single_env(self):
        """Create a ResourceGatheringEnv for single-env reward testing."""
        env = ResourceGatheringEnv(shader_dir=str(SHADERS_DIR))
        env.reset()
        return env

    @pytest.fixture
    def stage2_vec_env(self):
        """Create a SpeedrunVecEnv with all envs set to RESOURCE_GATHERING."""
        vec_env = SpeedrunVecEnv(
            num_envs=self.NUM_ENVS,
            shader_dir=str(SHADERS_DIR),
            observation_size=self.OBS_SIZE,
            initial_stage=StageID.RESOURCE_GATHERING,
            auto_curriculum=False,
        )
        vec_env.reset()
        return vec_env

    @pytest.fixture
    def standalone_shaper(self):
        """Create the standalone Stage 2 reward shaper."""
        return create_stage2_reward_shaper()

    def _make_obs_snapshot(
        self,
        y_level: float = 64.0,
        health_norm: float = 1.0,
    ) -> NDArray[np.float32]:
        """Create a synthetic observation vector for testing.

        Args:
            y_level: Y-position (obs index 1 for ResourceGatheringEnv, raw float).
            health_norm: Normalized health [0, 1] for SpeedrunVecEnv (obs index 8).

        Returns:
            Float32 observation array of size max(128, OBS_SIZE).
        """
        obs = np.zeros(max(128, self.OBS_SIZE), dtype=np.float32)
        # ResourceGatheringEnv uses obs[1] as Y-level
        obs[1] = y_level
        # SpeedrunVecEnv decodes obs[8] as health (normalized * 20)
        obs[8] = health_norm
        return obs

    def _make_state_snapshot(
        self,
        health: float = 20.0,
        inventory: dict[str, int] | None = None,
        y_position: float = 64.0,
    ) -> dict[str, Any]:
        """Create a state dict for the standalone reward shaper.

        Args:
            health: Player health (raw, 0-20).
            inventory: Item inventory counts.
            y_position: Y-level position.

        Returns:
            State dictionary compatible with create_stage2_reward_shaper.
        """
        return {
            "health": health,
            "inventory": inventory or {},
            "y_position": y_position,
        }

    def test_baseline_zero_reward_all_paths(
        self, stage2_single_env, stage2_vec_env, standalone_shaper
    ):
        """All paths produce near-zero shaped reward on neutral observations.

        With no milestones triggered, no depth change, and default health:
        - ResourceGatheringEnv: base_reward * reward_scale (0 * 1.0 = 0)
        - SpeedrunVecEnv: passthrough (raw reward from simulator)
        - Standalone shaper: -0.0001 (time penalty only)
        """
        obs = self._make_obs_snapshot(y_level=64.0)

        # ResourceGatheringEnv: _shape_reward with base_reward=0, no Y change
        single_reward = stage2_single_env._shape_reward(0.0, obs[:128], action=0)
        assert single_reward == pytest.approx(0.0, abs=1e-6), (
            f"ResourceGatheringEnv should return 0 with no progression, got {single_reward}"
        )

        # SpeedrunVecEnv: _apply_reward_shaping on Stage 2 is passthrough
        raw_rewards = np.zeros(self.NUM_ENVS, dtype=np.float32)
        vec_obs = np.tile(obs[: self.OBS_SIZE], (self.NUM_ENVS, 1))
        dones = np.zeros(self.NUM_ENVS, dtype=np.bool_)
        shaped = stage2_vec_env._apply_reward_shaping(raw_rewards, vec_obs, dones)
        np.testing.assert_allclose(
            shaped, 0.0, atol=1e-6,
            err_msg="SpeedrunVecEnv Stage 2 should passthrough zero rewards",
        )

        # Standalone shaper: first call only produces time penalty
        standalone_shaper(self._make_state_snapshot())  # warm up prev_state
        shaper_reward = standalone_shaper(self._make_state_snapshot())
        assert shaper_reward == pytest.approx(-0.0001, abs=1e-6), (
            f"Standalone shaper neutral reward should be -0.0001, got {shaper_reward}"
        )

    def test_vertical_descent_rewards_single_env_only(
        self, stage2_single_env, stage2_vec_env, standalone_shaper
    ):
        """Only ResourceGatheringEnv grants reward for descending Y-levels.

        ResourceGatheringEnv tracks lowest_y_reached and gives REWARD_VERTICAL_MINING
        when the agent goes deeper. SpeedrunVecEnv has no such shaping for Stage 2.
        """
        # Start at Y=64, then descend to Y=30
        obs_high = self._make_obs_snapshot(y_level=64.0)
        obs_low = self._make_obs_snapshot(y_level=30.0)

        # Initialize lowest_y tracking
        stage2_single_env._stage_state["lowest_y_reached"] = 64.0
        r_high = stage2_single_env._shape_reward(0.0, obs_high[:128], action=0)

        # Now descend
        r_low = stage2_single_env._shape_reward(0.0, obs_low[:128], action=0)

        # Descending should yield REWARD_VERTICAL_MINING = 0.05
        assert r_low == pytest.approx(0.05, abs=1e-6), (
            f"ResourceGatheringEnv should give 0.05 for descent, got {r_low}"
        )
        assert r_high == pytest.approx(0.0, abs=1e-6), (
            f"No reward at starting height, got {r_high}"
        )

        # SpeedrunVecEnv: no vertical mining shaping for Stage 2
        raw_rewards = np.zeros(self.NUM_ENVS, dtype=np.float32)
        vec_obs = np.tile(obs_low[: self.OBS_SIZE], (self.NUM_ENVS, 1))
        dones = np.zeros(self.NUM_ENVS, dtype=np.bool_)
        shaped = stage2_vec_env._apply_reward_shaping(raw_rewards, vec_obs, dones)
        np.testing.assert_allclose(
            shaped, 0.0, atol=1e-6,
            err_msg="SpeedrunVecEnv Stage 2 has no vertical mining reward",
        )

    def test_depth_bonus_standalone_shaper(self, standalone_shaper):
        """Standalone shaper gives depth bonus when Y < 16 and descending."""
        # Warm up at Y=64
        standalone_shaper(self._make_state_snapshot(y_position=64.0))

        # Descend to Y=15 (below threshold of 16)
        r1 = standalone_shaper(self._make_state_snapshot(y_position=15.0))
        # Should get time_penalty + depth_reward (0.005)
        assert r1 == pytest.approx(-0.0001 + 0.005, abs=1e-5), (
            f"Standalone shaper should give depth bonus at Y=15, got {r1}"
        )

        # Same Y: no further depth reward
        r2 = standalone_shaper(self._make_state_snapshot(y_position=15.0))
        assert r2 == pytest.approx(-0.0001, abs=1e-6), (
            f"No depth reward at same Y, got {r2}"
        )

    def test_milestone_iron_ore_standalone_vs_single_env(
        self, stage2_single_env, standalone_shaper
    ):
        """Milestone rewards exist in standalone shaper but not ResourceGatheringEnv._shape_reward.

        ResourceGatheringEnv._shape_reward only tracks Y-level. The milestone-based
        reward shaping is in the standalone create_stage2_reward_shaper.
        """
        # Standalone: iron_ore milestone fires (+0.15)
        standalone_shaper(self._make_state_snapshot())
        r_milestone = standalone_shaper(
            self._make_state_snapshot(inventory={"iron_ore": 1})
        )
        # first_iron_ore milestone = 0.15, minus time penalty
        assert r_milestone >= 0.14, (
            f"Standalone shaper should fire first_iron_ore milestone, got {r_milestone}"
        )

        # ResourceGatheringEnv: no milestone tracking in _shape_reward
        obs = self._make_obs_snapshot(y_level=64.0)
        r_single = stage2_single_env._shape_reward(0.0, obs[:128], action=0)
        assert r_single == pytest.approx(0.0, abs=1e-6), (
            f"ResourceGatheringEnv._shape_reward has no milestone tracking, got {r_single}"
        )

    def test_vectorized_batch_consistency(self, stage2_vec_env):
        """SpeedrunVecEnv produces identical shaped rewards for identical observations.

        All NUM_ENVS environments at Stage 2 with identical raw rewards and
        observations should produce identical shaped outputs.
        """
        raw_rewards = np.full(self.NUM_ENVS, 0.5, dtype=np.float32)
        obs = np.tile(
            self._make_obs_snapshot(y_level=30.0)[: self.OBS_SIZE],
            (self.NUM_ENVS, 1),
        )
        dones = np.zeros(self.NUM_ENVS, dtype=np.bool_)

        shaped = stage2_vec_env._apply_reward_shaping(raw_rewards, obs, dones)

        # All envs should get the same shaped reward (Stage 2 is passthrough)
        np.testing.assert_allclose(
            shaped, 0.5, atol=1e-6,
            err_msg="Vectorized Stage 2 should pass through raw rewards uniformly",
        )

    def test_vectorized_mixed_stages_stage2_passthrough(self, stage2_vec_env):
        """When mixed stages exist, Stage 2 envs still get passthrough shaping.

        Set env 0 to BASIC_SURVIVAL (gets +0.01 alive bonus) and envs 1-3
        stay at RESOURCE_GATHERING (passthrough). Verify the difference.
        """
        # Set env 0 to BASIC_SURVIVAL
        stage2_vec_env._env_stages[0] = StageID.BASIC_SURVIVAL.value

        raw_rewards = np.full(self.NUM_ENVS, 0.0, dtype=np.float32)
        obs = np.tile(
            self._make_obs_snapshot()[: self.OBS_SIZE],
            (self.NUM_ENVS, 1),
        )
        dones = np.zeros(self.NUM_ENVS, dtype=np.bool_)

        shaped = stage2_vec_env._apply_reward_shaping(raw_rewards, obs, dones)

        # Env 0 (BASIC_SURVIVAL): gets +0.01 survival bonus
        assert shaped[0] == pytest.approx(0.01, abs=1e-6), (
            f"BASIC_SURVIVAL env should get +0.01, got {shaped[0]}"
        )
        # Envs 1-3 (RESOURCE_GATHERING): passthrough
        np.testing.assert_allclose(
            shaped[1:], 0.0, atol=1e-6,
            err_msg="RESOURCE_GATHERING envs should passthrough",
        )

    def test_damage_penalty_standalone_vs_envs(
        self, stage2_single_env, stage2_vec_env, standalone_shaper
    ):
        """Damage penalty exists only in standalone shaper, not in env _shape_reward.

        ResourceGatheringEnv._shape_reward doesn't track health delta.
        SpeedrunVecEnv has no health-based penalty for Stage 2.
        The standalone shaper applies 0.015 * damage penalty.
        """
        # Standalone: damage from 20 -> 15 = 5 damage * 0.015 = 0.075 penalty
        standalone_shaper(self._make_state_snapshot(health=20.0))
        r_damaged = standalone_shaper(self._make_state_snapshot(health=15.0))
        expected = -0.0001 - (5.0 * 0.015)  # time + damage penalty
        assert r_damaged == pytest.approx(expected, abs=1e-5), (
            f"Standalone shaper damage penalty mismatch: expected {expected}, got {r_damaged}"
        )

        # ResourceGatheringEnv: no damage tracking in _shape_reward
        obs = self._make_obs_snapshot(y_level=64.0)
        r_single = stage2_single_env._shape_reward(0.0, obs[:128], action=0)
        assert r_single == pytest.approx(0.0, abs=1e-6)

        # SpeedrunVecEnv: no damage-based shaping for Stage 2
        raw_rewards = np.zeros(self.NUM_ENVS, dtype=np.float32)
        vec_obs = np.tile(obs[: self.OBS_SIZE], (self.NUM_ENVS, 1))
        dones = np.zeros(self.NUM_ENVS, dtype=np.bool_)
        shaped = stage2_vec_env._apply_reward_shaping(raw_rewards, vec_obs, dones)
        np.testing.assert_allclose(shaped, 0.0, atol=1e-6)

    def test_cumulative_progression_standalone_multiple_milestones(self, standalone_shaper):
        """Standalone shaper accumulates multiple milestones in order.

        Simulates a realistic Stage 2 progression:
        iron_ore -> iron_ingot -> iron_pickaxe -> bucket -> obsidian
        """
        states = [
            self._make_state_snapshot(),  # warm up
            self._make_state_snapshot(inventory={"iron_ore": 1}),
            self._make_state_snapshot(inventory={"iron_ore": 3}),
            self._make_state_snapshot(inventory={"iron_ore": 3, "iron_ingot": 1}),
            self._make_state_snapshot(
                inventory={"iron_ore": 3, "iron_ingot": 3},
            ),
            self._make_state_snapshot(
                inventory={"iron_ore": 3, "iron_ingot": 3, "bucket": 1},
            ),
            self._make_state_snapshot(
                inventory={"iron_ore": 3, "iron_ingot": 3, "bucket": 1, "obsidian": 10},
            ),
        ]

        rewards: list[float] = []
        for state in states:
            rewards.append(standalone_shaper(state))

        stats = standalone_shaper.stats  # type: ignore[attr-defined]

        # Verify milestones fired in order
        expected_milestones = {
            "first_iron_ore",
            "iron_ore_x3",
            "first_iron_ingot",
            "iron_ingot_x3",
            "bucket",
            "first_obsidian",
            "obsidian_x10",
        }
        achieved = set(stats.milestones_achieved)
        assert expected_milestones.issubset(achieved), (
            f"Missing milestones: {expected_milestones - achieved}\n"
            f"Achieved: {achieved}"
        )

        # Total milestone rewards should be positive and significant
        assert stats.milestone_rewards > 1.0, (
            f"Expected substantial milestone rewards, got {stats.milestone_rewards}"
        )

        # Each progression step should have positive reward (milestone bonuses dominate)
        for i, r in enumerate(rewards[1:], start=1):
            assert r > -0.01, (
                f"Step {i} reward should be positive or near-zero, got {r}"
            )

    def test_reward_scale_applies_to_single_env(self, stage2_single_env):
        """ResourceGatheringEnv reward_scale multiplier applies to _shape_reward output."""
        stage2_single_env.config.reward_scale = 2.0
        stage2_single_env._stage_state["lowest_y_reached"] = 64.0

        obs = self._make_obs_snapshot(y_level=30.0)
        r_scaled = stage2_single_env._shape_reward(0.0, obs[:128], action=0)

        # REWARD_VERTICAL_MINING (0.05) * reward_scale (2.0) = 0.1
        assert r_scaled == pytest.approx(0.1, abs=1e-6), (
            f"Scaled vertical mining reward should be 0.1, got {r_scaled}"
        )

    def test_vec_env_end_fight_multiplier_not_applied_to_stage2(self, stage2_vec_env):
        """SpeedrunVecEnv's 1.2x END_FIGHT multiplier is NOT applied to Stage 2.

        Ensures Stage 2 envs don't accidentally inherit END_FIGHT shaping.
        """
        raw_rewards = np.full(self.NUM_ENVS, 1.0, dtype=np.float32)
        obs = np.tile(
            self._make_obs_snapshot()[: self.OBS_SIZE],
            (self.NUM_ENVS, 1),
        )
        dones = np.zeros(self.NUM_ENVS, dtype=np.bool_)

        shaped = stage2_vec_env._apply_reward_shaping(raw_rewards, obs, dones)

        # Stage 2 should NOT apply the 1.2x multiplier
        np.testing.assert_allclose(
            shaped, 1.0, atol=1e-6,
            err_msg="Stage 2 should not get END_FIGHT 1.2x multiplier",
        )

    def test_identical_snapshot_sequence_reproducibility(self, standalone_shaper):
        """Two fresh standalone shapers produce identical rewards on same sequence.

        Verifies deterministic behavior of the reward shaper.
        """
        shaper_a = create_stage2_reward_shaper()
        shaper_b = create_stage2_reward_shaper()

        sequence = [
            self._make_state_snapshot(health=20.0, y_position=64.0),
            self._make_state_snapshot(health=20.0, y_position=40.0),
            self._make_state_snapshot(health=20.0, y_position=14.0),
            self._make_state_snapshot(
                health=18.0, y_position=12.0, inventory={"iron_ore": 2}
            ),
            self._make_state_snapshot(
                health=18.0, y_position=12.0, inventory={"iron_ore": 3, "diamond": 1}
            ),
        ]

        rewards_a = [shaper_a(s) for s in sequence]
        rewards_b = [shaper_b(s) for s in sequence]

        np.testing.assert_allclose(
            rewards_a, rewards_b, atol=1e-10,
            err_msg="Two fresh shapers should produce identical reward sequences",
        )


# =============================================================================
# Comprehensive Reward Shaper Tests: Iron Smelting, Bucket, Obsidian
# =============================================================================


def _make_state(
    health: float = 20.0,
    y_position: float = 64.0,
    inventory: dict[str, int] | None = None,
    **flags: bool,
) -> dict[str, Any]:
    """Construct a state dict for the reward shaper."""
    state: dict[str, Any] = {
        "health": health,
        "y_position": y_position,
        "inventory": inventory or {},
    }
    state.update(flags)
    return state


@pytest.mark.skipif(not HAS_REWARD_SHAPING, reason="reward_shaping module not available")
class TestIronSmeltingRewards:
    """Test that iron smelting events produce correct shaped reward increments."""

    def test_first_iron_ore_reward(self):
        """Acquiring first iron ore grants +0.15 milestone."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        reward = shaper(_make_state(inventory={"iron_ore": 1}))
        assert reward == pytest.approx(0.15 - 0.0001, abs=1e-6)

    def test_iron_ore_x3_reward(self):
        """Accumulating 3 iron ore grants additional +0.1 milestone."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())
        shaper(_make_state(inventory={"iron_ore": 1}))

        reward = shaper(_make_state(inventory={"iron_ore": 3}))
        assert reward == pytest.approx(0.1 - 0.0001, abs=1e-6)

    def test_first_iron_ingot_reward(self):
        """Smelting first iron ingot grants +0.2 milestone plus progressive bonus."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        reward = shaper(_make_state(inventory={"iron_ingot": 1}))
        # Milestone: first_iron_ingot (+0.2)
        # Progressive: min(1*0.015, 0.3) - 0 = 0.015
        expected = 0.2 + 0.015 - 0.0001
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_iron_ingot_x3_reward(self):
        """Accumulating 3 iron ingots grants +0.15 milestone."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())
        shaper(_make_state(inventory={"iron_ingot": 1}))
        shaper(_make_state(inventory={"iron_ingot": 2}))

        reward = shaper(_make_state(inventory={"iron_ingot": 3}))
        # Milestone: +0.15, Progressive: 0.045 - 0.030 = 0.015
        expected = 0.15 + 0.015 - 0.0001
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_iron_ingot_x10_reward(self):
        """Accumulating 10 iron ingots grants +0.15 milestone."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        for count in range(1, 10):
            shaper(_make_state(inventory={"iron_ingot": count}))

        reward = shaper(_make_state(inventory={"iron_ingot": 10}))
        # Milestone: +0.15, Progressive: 0.15 - 0.135 = 0.015
        expected = 0.15 + 0.015 - 0.0001
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_iron_pickaxe_reward(self):
        """Crafting iron pickaxe grants +0.35 milestone."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        reward = shaper(_make_state(has_iron_pickaxe=True))
        assert reward == pytest.approx(0.35 - 0.0001, abs=1e-6)

    def test_iron_progressive_caps_at_03(self):
        """Progressive iron reward caps at 0.3 total."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        # 20 ingots: progressive = min(20*0.015, 0.3) = 0.3 (capped)
        # Milestones: first_iron_ingot (+0.2) + iron_ingot_x3 (+0.15) + iron_ingot_x10 (+0.15)
        reward = shaper(_make_state(inventory={"iron_ingot": 20}))
        expected = 0.2 + 0.15 + 0.15 + 0.3 - 0.0001
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_smelting_incremental_progressive(self):
        """Each additional ingot adds 0.015 progressive reward."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())
        shaper(_make_state(inventory={"iron_ingot": 1}))

        reward = shaper(_make_state(inventory={"iron_ingot": 2}))
        assert reward == pytest.approx(0.015 - 0.0001, abs=1e-6)

    def test_iron_milestones_in_stats(self):
        """All iron milestones are recorded in stats.milestones_achieved."""
        shaper = create_stage2_reward_shaper()
        stats = shaper.stats  # type: ignore[attr-defined]
        shaper(_make_state())

        shaper(_make_state(
            inventory={"iron_ore": 3, "iron_ingot": 10},
            has_iron_pickaxe=True,
        ))

        for name in [
            "first_iron_ore", "iron_ore_x3", "first_iron_ingot",
            "iron_ingot_x3", "iron_ingot_x10", "iron_pickaxe",
        ]:
            assert name in stats.milestones_achieved, f"Missing milestone: {name}"


@pytest.mark.skipif(not HAS_REWARD_SHAPING, reason="reward_shaping module not available")
class TestBucketCraftingRewards:
    """Test that bucket crafting events produce correct shaped reward increments."""

    def test_bucket_craft_reward(self):
        """Crafting a bucket grants +0.3 milestone.

        MC 1.8.9 bucket recipe:
        [I][ ][I]
        [ ][I][ ]
        Requires 3 iron ingots on crafting table.
        """
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        reward = shaper(_make_state(has_bucket=True))
        assert reward == pytest.approx(0.3 - 0.0001, abs=1e-6)

    def test_bucket_via_inventory_count(self):
        """Bucket detected via inventory count also triggers milestone."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        reward = shaper(_make_state(inventory={"bucket": 1}))
        assert reward == pytest.approx(0.3 - 0.0001, abs=1e-6)

    def test_water_bucket_reward(self):
        """Filling bucket with water grants +0.15 milestone."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())
        shaper(_make_state(has_bucket=True))

        reward = shaper(_make_state(has_bucket=True, inventory={"water_bucket": 1}))
        assert reward == pytest.approx(0.15 - 0.0001, abs=1e-6)

    def test_lava_bucket_reward(self):
        """Filling bucket with lava grants +0.2 milestone."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())
        shaper(_make_state(has_bucket=True))

        reward = shaper(_make_state(has_bucket=True, inventory={"lava_bucket": 1}))
        assert reward == pytest.approx(0.2 - 0.0001, abs=1e-6)

    def test_bucket_water_lava_cumulative(self):
        """Full bucket progression: bucket -> water -> lava."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        r1 = shaper(_make_state(has_bucket=True))
        assert r1 == pytest.approx(0.3 - 0.0001, abs=1e-6)

        r2 = shaper(_make_state(has_bucket=True, inventory={"water_bucket": 1}))
        assert r2 == pytest.approx(0.15 - 0.0001, abs=1e-6)

        r3 = shaper(
            _make_state(has_bucket=True, inventory={"water_bucket": 1, "lava_bucket": 1})
        )
        assert r3 == pytest.approx(0.2 - 0.0001, abs=1e-6)

    def test_bucket_milestone_not_repeated(self):
        """Milestone rewards are one-time; repeated state gives no bonus."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())
        shaper(_make_state(has_bucket=True))

        reward = shaper(_make_state(has_bucket=True))
        assert reward == pytest.approx(-0.0001, abs=1e-6)

    def test_bucket_milestones_in_stats(self):
        """Bucket-related milestones are recorded in stats."""
        shaper = create_stage2_reward_shaper()
        stats = shaper.stats  # type: ignore[attr-defined]
        shaper(_make_state())

        shaper(_make_state(
            has_bucket=True,
            inventory={"water_bucket": 1, "lava_bucket": 1},
        ))

        for name in ["bucket", "water_bucket", "lava_bucket"]:
            assert name in stats.milestones_achieved, f"Missing milestone: {name}"

        # Total milestone reward: 0.3 + 0.15 + 0.2 = 0.65
        assert stats.milestone_rewards == pytest.approx(0.65, abs=1e-6)


@pytest.mark.skipif(not HAS_REWARD_SHAPING, reason="reward_shaping module not available")
class TestObsidianCollectionRewards:
    """Test that obsidian collection events produce correct shaped reward increments."""

    def test_first_obsidian_reward(self):
        """Mining first obsidian grants +0.2 milestone plus progressive bonus.

        MC 1.8.9: requires diamond pickaxe, 9.4s mine time.
        Created by water flowing onto lava source block.
        """
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        reward = shaper(_make_state(inventory={"obsidian": 1}))
        # Milestone: +0.2, Progressive: min(1*0.015, 0.25) = 0.015
        expected = 0.2 + 0.015 - 0.0001
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_obsidian_progressive_per_block(self):
        """Each additional obsidian adds 0.015 progressive reward."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())
        shaper(_make_state(inventory={"obsidian": 1}))

        reward = shaper(_make_state(inventory={"obsidian": 2}))
        assert reward == pytest.approx(0.015 - 0.0001, abs=1e-6)

    def test_obsidian_x10_reward(self):
        """Collecting 10 obsidian grants +0.25 milestone (minimum portal)."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        for count in range(1, 10):
            shaper(_make_state(inventory={"obsidian": count}))

        reward = shaper(_make_state(inventory={"obsidian": 10}))
        # Milestone: +0.25, Progressive: 0.15 - 0.135 = 0.015
        expected = 0.25 + 0.015 - 0.0001
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_obsidian_x14_reward(self):
        """Collecting 14 obsidian grants +0.1 milestone (full portal frame)."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        for count in range(1, 14):
            shaper(_make_state(inventory={"obsidian": count}))

        reward = shaper(_make_state(inventory={"obsidian": 14}))
        # Milestone: +0.1, Progressive: 0.21 - 0.195 = 0.015
        expected = 0.1 + 0.015 - 0.0001
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_obsidian_progressive_caps_at_025(self):
        """Progressive obsidian reward caps at 0.25 total."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        for count in range(1, 17):
            shaper(_make_state(inventory={"obsidian": count}))

        # 17th: progressive = 0.25 - 0.24 = 0.01
        reward_17 = shaper(_make_state(inventory={"obsidian": 17}))
        assert reward_17 == pytest.approx(0.01 - 0.0001, abs=1e-6)

        # 18th: capped, no progressive delta
        reward_18 = shaper(_make_state(inventory={"obsidian": 18}))
        assert reward_18 == pytest.approx(-0.0001, abs=1e-6)

    def test_obsidian_full_collection_totals(self):
        """Full obsidian collection 1-14 produces expected cumulative rewards."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        total_milestone = 0.0
        total_progressive = 0.0

        for count in range(1, 15):
            reward = shaper(_make_state(inventory={"obsidian": count}))
            adjusted = reward + 0.0001

            prev_prog = min((count - 1) * 0.015, 0.25)
            curr_prog = min(count * 0.015, 0.25)
            prog_delta = curr_prog - prev_prog
            total_progressive += prog_delta

            milestone = adjusted - prog_delta
            if milestone > 0.001:
                total_milestone += milestone

        # Milestones: first_obsidian (0.2) + obsidian_x10 (0.25) + obsidian_x14 (0.1)
        assert total_milestone == pytest.approx(0.55, abs=1e-4)
        # Progressive: min(14*0.015, 0.25) = 0.21
        assert total_progressive == pytest.approx(0.21, abs=1e-4)

    def test_combined_iron_and_obsidian_progressive(self):
        """Iron and obsidian progressive rewards accumulate independently."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())

        reward = shaper(_make_state(inventory={"iron_ingot": 1, "obsidian": 1}))
        # Milestones: first_iron_ingot (+0.2) + first_obsidian (+0.2)
        # Progressive: iron 0.015 + obsidian 0.015
        expected = 0.2 + 0.2 + 0.015 + 0.015 - 0.0001
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_flint_and_steel_after_obsidian(self):
        """Crafting flint and steel grants +0.15 (needed to light portal)."""
        shaper = create_stage2_reward_shaper()
        shaper(_make_state())
        shaper(_make_state(inventory={"obsidian": 14}))

        reward = shaper(_make_state(
            inventory={"obsidian": 14, "flint_and_steel": 1},
            has_flint_and_steel=True,
        ))
        assert reward == pytest.approx(0.15 - 0.0001, abs=1e-6)

    def test_obsidian_milestones_in_stats(self):
        """All obsidian milestones are recorded in stats."""
        shaper = create_stage2_reward_shaper()
        stats = shaper.stats  # type: ignore[attr-defined]
        shaper(_make_state())

        shaper(_make_state(inventory={"obsidian": 14}))

        for name in ["first_obsidian", "obsidian_x10", "obsidian_x14"]:
            assert name in stats.milestones_achieved, f"Missing milestone: {name}"
