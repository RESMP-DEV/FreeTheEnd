"""XP system verification for Minecraft-style experience mechanics.

Tests:
- XP level calculation curve (piecewise formula)
- XP orb pickup values
- Enchantment level costs
- Anvil repair XP costs
- XP from mob kills and activities
"""

from dataclasses import dataclass
from enum import Enum


class XPSource(Enum):
    """Sources of experience points."""

    # Passive
    ORB_PICKUP = "orb_pickup"
    BREEDING = "breeding"
    FISHING = "fishing"
    TRADING = "trading"

    # Combat
    HOSTILE_MOB = "hostile_mob"
    BOSS_MOB = "boss_mob"
    PLAYER_KILL = "player_kill"

    # Mining/Smelting
    MINING_ORE = "mining_ore"
    SMELTING = "smelting"

    # Other
    BOTTLE_O_ENCHANTING = "bottle_o_enchanting"
    GRINDSTONE = "grindstone"


@dataclass
class XPOrbValue:
    """XP orb with value and visual size."""

    value: int
    texture_index: int  # 0-10 based on value range

    @staticmethod
    def from_value(value: int) -> "XPOrbValue":
        """Create orb with appropriate texture based on value."""
        # Minecraft orb sizes based on XP value
        if value >= 2477:
            texture = 10
        elif value >= 1237:
            texture = 9
        elif value >= 617:
            texture = 8
        elif value >= 307:
            texture = 7
        elif value >= 149:
            texture = 6
        elif value >= 73:
            texture = 5
        elif value >= 37:
            texture = 4
        elif value >= 17:
            texture = 3
        elif value >= 7:
            texture = 2
        elif value >= 3:
            texture = 1
        else:
            texture = 0
        return XPOrbValue(value=value, texture_index=texture)


# Minecraft XP level formulas (Java Edition)
# Total XP to reach level L:
#   L ∈ [0, 16]:  L² + 6L
#   L ∈ [17, 31]: 2.5L² - 40.5L + 360
#   L ≥ 32:       4.5L² - 162.5L + 2220


def total_xp_for_level(level: int) -> int:
    """Calculate total XP points needed to reach a level from 0."""
    if level <= 0:
        return 0
    elif level <= 16:
        return level * level + 6 * level
    elif level <= 31:
        return int(2.5 * level * level - 40.5 * level + 360)
    else:
        return int(4.5 * level * level - 162.5 * level + 2220)


def xp_for_next_level(current_level: int) -> int:
    """Calculate XP points needed to advance from current level to next."""
    # XP for level L → L+1:
    #   L ∈ [0, 15]:  2L + 7
    #   L ∈ [16, 30]: 5L - 38
    #   L ≥ 31:       9L - 158
    if current_level <= 15:
        return 2 * current_level + 7
    elif current_level <= 30:
        return 5 * current_level - 38
    else:
        return 9 * current_level - 158


def level_from_total_xp(total_xp: int) -> tuple[int, int]:
    """Calculate level and leftover XP from total XP points.

    Returns:
        (level, xp_into_current_level)
    """
    if total_xp <= 0:
        return (0, 0)

    # Binary search for level
    low, high = 0, 21863  # Max level with int32 XP
    while low < high:
        mid = (low + high + 1) // 2
        if total_xp_for_level(mid) <= total_xp:
            low = mid
        else:
            high = mid - 1

    level = low
    xp_into_level = total_xp - total_xp_for_level(level)
    return (level, xp_into_level)


# XP values from various sources
XP_VALUES: dict[str, int | tuple[int, int]] = {
    # Breeding
    "breeding_animal": (1, 7),
    # Fishing (base values, can vary)
    "fishing_fish": (1, 6),
    "fishing_treasure": (1, 6),
    "fishing_junk": (1, 6),
    # Smelting (per item)
    "smelt_iron_ore": 7,  # 0.7 per item, stored and given when extracted
    "smelt_gold_ore": 10,  # 1.0 per item
    "smelt_ancient_debris": 20,  # 2.0 per item
    "smelt_diamond_ore": 10,  # 1.0 per item (from deepslate)
    "smelt_emerald_ore": 10,  # 1.0 per item
    "smelt_lapis_ore": 2,  # 0.2 per item
    "smelt_redstone_ore": 3,  # 0.3 per item (silk touched)
    "smelt_nether_quartz_ore": 2,  # 0.2 per item (silk touched)
    "smelt_coal_ore": 1,  # 0.1 per item (silk touched)
    "smelt_food": 3,  # 0.35 per item (varies)
    # Mining ores (with fortune, silk touch gives 0)
    "mine_coal_ore": (0, 2),
    "mine_diamond_ore": (3, 7),
    "mine_emerald_ore": (3, 7),
    "mine_lapis_ore": (2, 5),
    "mine_nether_quartz_ore": (2, 5),
    "mine_redstone_ore": (1, 5),
    "mine_nether_gold_ore": (0, 1),
    "mine_sculk": 1,
    "mine_sculk_catalyst": 5,
    "mine_spawner": (15, 43),
    # Hostile mobs (base values, some drop more)
    "kill_zombie": 5,
    "kill_skeleton": 5,
    "kill_spider": 5,
    "kill_creeper": 5,
    "kill_enderman": 5,
    "kill_witch": 5,
    "kill_blaze": 10,
    "kill_ghast": 5,
    "kill_magma_cube_large": 4,
    "kill_magma_cube_medium": 2,
    "kill_magma_cube_small": 1,
    "kill_slime_large": 4,
    "kill_slime_medium": 2,
    "kill_slime_small": 1,
    "kill_guardian": 10,
    "kill_elder_guardian": 10,
    "kill_shulker": 5,
    "kill_phantom": 5,
    "kill_piglin": 5,
    "kill_piglin_brute": 20,
    "kill_warden": 5,  # Not recommended to fight
    # Boss mobs
    "kill_ender_dragon": 12000,
    "kill_wither": 50,
    # Player kill (returns some of victim's XP)
    "kill_player": (0, 7),  # Base 0-7 per level of victim (max ~100)
    # Bottle o' Enchanting
    "bottle_enchanting": (3, 11),
    # Trading
    "trade_villager": (3, 6),  # Per trade, some give more
    # Grindstone (disenchanting)
    "grindstone": "formula",  # Returns enchantment XP
}

# Mob XP modifiers
MOB_XP_MODIFIERS = {
    "baby": 0,  # Baby zombies etc give same XP
    "equipment_bonus": 1,  # +1-3 per piece of equipment
    "looting": 0,  # Looting doesn't increase XP
    "player_kill_only": True,  # Must be killed by player for XP
}


@dataclass
class EnchantmentCost:
    """Enchantment table level costs."""

    slot: int  # 1, 2, or 3 (top to bottom)
    min_level: int  # Minimum player level required
    lapis_cost: int  # Lapis lazuli consumed

    # Bookshelves affect enchantment level
    # max_level = base + rand(bookshelves/2 + 1) + rand(bookshelves/2 + 1)


# Enchantment table slot requirements
ENCHANTMENT_SLOTS = [
    EnchantmentCost(slot=1, min_level=1, lapis_cost=1),
    EnchantmentCost(slot=2, min_level=2, lapis_cost=2),
    EnchantmentCost(slot=3, min_level=3, lapis_cost=3),
]

# Max bookshelves = 15 (5x5 ring with corners, one block gap)
MAX_BOOKSHELVES = 15
MAX_ENCHANT_LEVEL = 30  # With 15 bookshelves

# Anvil repair costs
ANVIL_COSTS = {
    "rename": 1,  # Just renaming
    "repair_material": 1,  # Per material (ingot/diamond)
    "repair_combine": "sum",  # Sum of both items' prior work
    "enchant_book": "book_cost",  # Based on enchantments
    "prior_work_penalty": lambda n: 2**n - 1,  # Doubles each time
    "too_expensive": 40,  # Operations > 39 levels show "Too Expensive!"
}


class XPState:
    """Player XP state tracker."""

    def __init__(self, level: int = 0, xp_progress: float = 0.0):
        """Initialize XP state.

        Args:
            level: Current experience level
            xp_progress: Progress toward next level (0.0 to 1.0)
        """
        self.level = level
        self.xp_progress = min(max(xp_progress, 0.0), 0.9999)
        self._total_xp = self._calculate_total()

    def _calculate_total(self) -> int:
        """Calculate total XP from level and progress."""
        base = total_xp_for_level(self.level)
        progress_xp = int(self.xp_progress * xp_for_next_level(self.level))
        return base + progress_xp

    @property
    def total_xp(self) -> int:
        """Total accumulated XP points."""
        return self._total_xp

    def add_xp(self, points: int) -> int:
        """Add XP points. Returns levels gained."""
        if points <= 0:
            return 0

        old_level = self.level
        self._total_xp += points
        self.level, xp_into_level = level_from_total_xp(self._total_xp)
        needed = xp_for_next_level(self.level)
        self.xp_progress = xp_into_level / needed if needed > 0 else 0.0
        return self.level - old_level

    def remove_levels(self, levels: int) -> bool:
        """Remove levels (for enchanting). Returns True if successful."""
        if levels > self.level:
            return False
        self.level -= levels
        # XP progress resets when levels are spent
        self._total_xp = total_xp_for_level(self.level)
        self.xp_progress = 0.0
        return True

    def xp_bar_percent(self) -> float:
        """Get XP bar fill percentage (0-100)."""
        return self.xp_progress * 100


class XPVerifier:
    """Verification suite for XP mechanics."""

    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def verify(self, name: str, condition: bool, message: str = "") -> bool:
        """Record verification result."""
        self.results.append((name, condition, message))
        return condition

    def run_all(self) -> dict[str, bool | int | list[tuple[str, bool, str]]]:
        """Run all verification tests."""
        self.results.clear()

        self._verify_level_curve()
        self._verify_xp_requirements()
        self._verify_level_from_total()
        self._verify_orb_values()
        self._verify_mob_xp()
        self._verify_enchantment_costs()
        self._verify_anvil_costs()
        self._verify_xp_state()
        self._verify_specific_totals()

        passed = sum(1 for _, ok, _ in self.results if ok)
        failed = len(self.results) - passed

        return {
            "passed": passed,
            "failed": failed,
            "total": len(self.results),
            "all_passed": failed == 0,
            "results": self.results,
        }

    def _verify_level_curve(self) -> None:
        """Verify XP level curve matches Minecraft formula."""
        # Test specific level totals (from Minecraft wiki)
        expected_totals = {
            0: 0,
            1: 7,
            5: 55,
            10: 160,
            15: 315,
            16: 352,
            17: 394,
            20: 550,
            25: 910,  # 2.5(25)^2 - 40.5(25) + 360 = 1562.5 - 1012.5 + 360 = 910
            30: 1395,
            31: 1507,
            32: 1628,
            40: 2920,
            50: 5345,
        }

        for level, expected in expected_totals.items():
            actual = total_xp_for_level(level)
            self.verify(
                f"total_xp_level_{level}",
                actual == expected,
                f"Level {level}: expected {expected}, got {actual}",
            )

    def _verify_xp_requirements(self) -> None:
        """Verify XP needed per level transition."""
        # L → L+1 requirements
        expected_requirements = {
            0: 7,  # 0→1
            5: 17,  # 5→6
            10: 27,  # 10→11
            15: 37,  # 15→16
            16: 42,  # 16→17 (formula changes)
            20: 62,  # 20→21
            25: 87,  # 25→26
            30: 112,  # 30→31
            31: 121,  # 31→32 (formula changes)
            40: 202,  # 40→41
            50: 292,  # 50→51
        }

        for level, expected in expected_requirements.items():
            actual = xp_for_next_level(level)
            self.verify(
                f"xp_for_level_{level}_to_{level + 1}",
                actual == expected,
                f"Level {level}→{level + 1}: expected {expected}, got {actual}",
            )

    def _verify_level_from_total(self) -> None:
        """Verify level calculation from total XP."""
        test_cases = [
            (0, 0, 0),  # 0 XP = level 0, 0 into level
            (7, 1, 0),  # 7 XP = level 1, 0 into level
            (10, 1, 3),  # 10 XP = level 1, 3 into level (need 9 for level 2)
            (160, 10, 0),  # 160 XP = level 10
            (352, 16, 0),  # 352 XP = level 16
            (1395, 30, 0),  # 1395 XP = level 30
            (1500, 30, 105),  # 1500 XP = level 30, 105 into level
        ]

        for total, expected_level, expected_into in test_cases:
            level, into = level_from_total_xp(total)
            self.verify(
                f"level_from_{total}_xp",
                level == expected_level and into == expected_into,
                f"{total} XP: expected L{expected_level}+{expected_into}, got L{level}+{into}",
            )

    def _verify_orb_values(self) -> None:
        """Verify XP orb texture/size mapping."""
        orb_tests = [
            (1, 0),
            (3, 1),
            (7, 2),
            (17, 3),
            (37, 4),
            (73, 5),
            (149, 6),
            (307, 7),
            (617, 8),
            (1237, 9),
            (2477, 10),
        ]

        for value, expected_texture in orb_tests:
            orb = XPOrbValue.from_value(value)
            self.verify(
                f"orb_texture_{value}",
                orb.texture_index == expected_texture,
                f"Orb value {value}: expected texture {expected_texture}, got {orb.texture_index}",
            )

    def _verify_mob_xp(self) -> None:
        """Verify mob XP values."""
        # Ender Dragon gives massive XP (first kill)
        dragon_xp = XP_VALUES.get("kill_ender_dragon")
        self.verify(
            "ender_dragon_12000_xp",
            dragon_xp == 12000,
            f"Ender Dragon first kill: expected 12000, got {dragon_xp}",
        )

        # Wither gives 50 XP
        wither_xp = XP_VALUES.get("kill_wither")
        self.verify(
            "wither_50_xp",
            wither_xp == 50,
            f"Wither kill: expected 50, got {wither_xp}",
        )

        # Standard hostile mobs give 5 XP
        standard_mobs = ["zombie", "skeleton", "spider", "creeper", "enderman"]
        for mob in standard_mobs:
            xp = XP_VALUES.get(f"kill_{mob}")
            self.verify(
                f"{mob}_5_xp",
                xp == 5,
                f"{mob.capitalize()} kill: expected 5, got {xp}",
            )

        # Blaze gives 10 XP
        blaze_xp = XP_VALUES.get("kill_blaze")
        self.verify(
            "blaze_10_xp",
            blaze_xp == 10,
            f"Blaze kill: expected 10, got {blaze_xp}",
        )

    def _verify_enchantment_costs(self) -> None:
        """Verify enchantment table costs."""
        # Three slots with level 1, 2, 3 requirements
        self.verify(
            "enchant_slot_1_cost_1_level",
            ENCHANTMENT_SLOTS[0].min_level == 1,
            f"Slot 1 min level: {ENCHANTMENT_SLOTS[0].min_level}",
        )
        self.verify(
            "enchant_slot_2_cost_2_levels",
            ENCHANTMENT_SLOTS[1].min_level == 2,
            f"Slot 2 min level: {ENCHANTMENT_SLOTS[1].min_level}",
        )
        self.verify(
            "enchant_slot_3_cost_3_levels",
            ENCHANTMENT_SLOTS[2].min_level == 3,
            f"Slot 3 min level: {ENCHANTMENT_SLOTS[2].min_level}",
        )

        # Lapis costs match slot number
        for i, slot in enumerate(ENCHANTMENT_SLOTS):
            self.verify(
                f"enchant_slot_{i + 1}_lapis_{i + 1}",
                slot.lapis_cost == i + 1,
                f"Slot {i + 1} lapis cost: {slot.lapis_cost}",
            )

        # Max enchant level with 15 bookshelves
        self.verify(
            "max_enchant_level_30",
            MAX_ENCHANT_LEVEL == 30,
            f"Max enchant level: {MAX_ENCHANT_LEVEL}",
        )

    def _verify_anvil_costs(self) -> None:
        """Verify anvil repair/combine costs."""
        # Rename costs 1 level
        self.verify(
            "anvil_rename_1_level",
            ANVIL_COSTS["rename"] == 1,
            f"Rename cost: {ANVIL_COSTS['rename']} levels",
        )

        # Too expensive threshold at 40
        self.verify(
            "anvil_too_expensive_40",
            ANVIL_COSTS["too_expensive"] == 40,
            f"Too expensive threshold: {ANVIL_COSTS['too_expensive']}",
        )

        # Prior work penalty doubles
        penalty_fn = ANVIL_COSTS["prior_work_penalty"]
        self.verify(
            "anvil_prior_work_0",
            penalty_fn(0) == 0,
            f"Prior work 0: penalty {penalty_fn(0)}",
        )
        self.verify(
            "anvil_prior_work_1",
            penalty_fn(1) == 1,
            f"Prior work 1: penalty {penalty_fn(1)}",
        )
        self.verify(
            "anvil_prior_work_2",
            penalty_fn(2) == 3,
            f"Prior work 2: penalty {penalty_fn(2)}",
        )
        self.verify(
            "anvil_prior_work_3",
            penalty_fn(3) == 7,
            f"Prior work 3: penalty {penalty_fn(3)}",
        )
        self.verify(
            "anvil_prior_work_6",
            penalty_fn(6) == 63,
            f"Prior work 6: penalty {penalty_fn(6)} (practically unrepairable)",
        )

    def _verify_xp_state(self) -> None:
        """Verify XP state management."""
        # Initial state
        state = XPState(level=0, xp_progress=0.0)
        self.verify(
            "initial_state_level_0",
            state.level == 0 and state.total_xp == 0,
            f"Initial: level {state.level}, total {state.total_xp}",
        )

        # Add XP to level up
        levels_gained = state.add_xp(7)
        self.verify(
            "add_7_xp_level_1",
            state.level == 1 and levels_gained == 1,
            f"After +7 XP: level {state.level}, gained {levels_gained}",
        )

        # Add more XP
        levels_gained = state.add_xp(153)  # 7 + 153 = 160 = level 10
        self.verify(
            "add_153_xp_level_10",
            state.level == 10,
            f"After +153 XP: level {state.level} (total {state.total_xp})",
        )

        # Remove levels for enchanting
        state2 = XPState(level=30, xp_progress=0.5)
        success = state2.remove_levels(3)
        self.verify(
            "remove_3_levels",
            success and state2.level == 27,
            f"After spending 3 levels: level {state2.level}",
        )

        # Can't remove more than you have
        state3 = XPState(level=5, xp_progress=0.0)
        success = state3.remove_levels(10)
        self.verify(
            "cannot_remove_more_than_have",
            not success and state3.level == 5,
            f"Cannot remove 10 from level 5: still level {state3.level}",
        )

    def _verify_specific_totals(self) -> None:
        """Verify specific XP milestones."""
        # Level 30 for max enchants
        self.verify(
            "level_30_requires_1395",
            total_xp_for_level(30) == 1395,
            f"Level 30 requires: {total_xp_for_level(30)} XP",
        )

        # How many XP orbs from dragon?
        dragon_xp = 12000
        level, _ = level_from_total_xp(dragon_xp)
        self.verify(
            "dragon_xp_gives_level_64",
            level >= 64,  # Should be ~64
            f"12000 XP (dragon) gives level {level}",
        )

        # Verify increasing cost per level
        for l in range(0, 50):
            current = xp_for_next_level(l)
            next_level = xp_for_next_level(l + 1)
            self.verify(
                f"level_{l + 1}_costs_more_than_{l}",
                next_level > current,
                f"L{l}→L{l + 1}: {current}, L{l + 1}→L{l + 2}: {next_level}",
            )


def verify_xp_mechanics() -> dict:
    """Run full XP mechanics verification."""
    verifier = XPVerifier()
    return verifier.run_all()


if __name__ == "__main__":
    results = verify_xp_mechanics()

    print("=" * 60)
    print("XP SYSTEM VERIFICATION")
    print("=" * 60)
    print(f"Passed: {results['passed']}/{results['total']}")
    print(f"Failed: {results['failed']}")
    print()

    for name, ok, msg in results["results"]:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {name}: {msg}")

    print()
    if results["all_passed"]:
        print("All XP mechanics verified successfully!")
    else:
        print("Some verifications failed - review results above.")

    # Print XP level table
    print()
    print("=" * 60)
    print("XP LEVEL REFERENCE TABLE")
    print("=" * 60)
    print(f"{'Level':<8}{'Total XP':<12}{'XP for Next':<14}{'Formula Range':<20}")
    print("-" * 60)
    for level in [0, 5, 10, 15, 16, 20, 25, 30, 31, 40, 50]:
        total = total_xp_for_level(level)
        next_xp = xp_for_next_level(level)
        if level <= 15:
            formula = "2L + 7"
        elif level <= 30:
            formula = "5L - 38"
        else:
            formula = "9L - 158"
        print(f"{level:<8}{total:<12}{next_xp:<14}{formula:<20}")
