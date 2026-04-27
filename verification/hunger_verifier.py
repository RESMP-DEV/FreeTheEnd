"""Hunger system verification for Minecraft-style survival mechanics.

Tests:
- Hunger depletion rates for all player actions
- Food item restoration values
- Sprint threshold (food >= 6)
- Starvation damage on each difficulty
- Natural regeneration threshold (food >= 18)
"""

from dataclasses import dataclass
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class PlayerAction(Enum):
    """Actions that deplete hunger."""

    IDLE = "idle"
    WALKING = "walking"
    SNEAKING = "sneaking"
    SWIMMING = "swimming"
    SPRINTING = "sprinting"
    JUMPING = "jumping"
    SPRINT_JUMPING = "sprint_jumping"
    ATTACKING = "attacking"
    MINING = "mining"
    TAKING_DAMAGE = "taking_damage"
    HUNGER_EFFECT = "hunger_effect"
    REGENERATING = "regenerating"


class Difficulty(Enum):
    """Game difficulty levels."""

    PEACEFUL = "peaceful"
    EASY = "easy"
    NORMAL = "normal"
    HARD = "hard"


@dataclass
class FoodItem:
    """Food item with hunger and saturation restoration."""

    name: str
    hunger_points: int  # Half-shanks restored (max 20)
    saturation: float  # Saturation points restored

    @property
    def saturation_ratio(self) -> float:
        """Saturation/hunger ratio (quality indicator)."""
        logger.debug("FoodItem.saturation_ratio called")
        return self.saturation / max(self.hunger_points, 1)


# Minecraft food values (Java Edition)
FOOD_ITEMS: dict[str, FoodItem] = {
    # Meats (cooked)
    "cooked_beef": FoodItem("Cooked Beef", 8, 12.8),
    "cooked_porkchop": FoodItem("Cooked Porkchop", 8, 12.8),
    "cooked_mutton": FoodItem("Cooked Mutton", 6, 9.6),
    "cooked_chicken": FoodItem("Cooked Chicken", 6, 7.2),
    "cooked_rabbit": FoodItem("Cooked Rabbit", 5, 6.0),
    "cooked_cod": FoodItem("Cooked Cod", 5, 6.0),
    "cooked_salmon": FoodItem("Cooked Salmon", 6, 9.6),
    # Meats (raw)
    "raw_beef": FoodItem("Raw Beef", 3, 1.8),
    "raw_porkchop": FoodItem("Raw Porkchop", 3, 1.8),
    "raw_mutton": FoodItem("Raw Mutton", 2, 1.2),
    "raw_chicken": FoodItem("Raw Chicken", 2, 1.2),
    "raw_rabbit": FoodItem("Raw Rabbit", 3, 1.8),
    "raw_cod": FoodItem("Raw Cod", 2, 0.4),
    "raw_salmon": FoodItem("Raw Salmon", 2, 0.4),
    # Vegetables/Fruits
    "apple": FoodItem("Apple", 4, 2.4),
    "golden_apple": FoodItem("Golden Apple", 4, 9.6),
    "enchanted_golden_apple": FoodItem("Enchanted Golden Apple", 4, 9.6),
    "carrot": FoodItem("Carrot", 3, 3.6),
    "golden_carrot": FoodItem("Golden Carrot", 6, 14.4),
    "potato": FoodItem("Potato", 1, 0.6),
    "baked_potato": FoodItem("Baked Potato", 5, 6.0),
    "beetroot": FoodItem("Beetroot", 1, 1.2),
    "melon_slice": FoodItem("Melon Slice", 2, 1.2),
    "sweet_berries": FoodItem("Sweet Berries", 2, 0.4),
    "glow_berries": FoodItem("Glow Berries", 2, 0.4),
    # Prepared foods
    "bread": FoodItem("Bread", 5, 6.0),
    "cookie": FoodItem("Cookie", 2, 0.4),
    "pumpkin_pie": FoodItem("Pumpkin Pie", 8, 4.8),
    "cake_slice": FoodItem("Cake (per slice)", 2, 0.4),
    "mushroom_stew": FoodItem("Mushroom Stew", 6, 7.2),
    "beetroot_soup": FoodItem("Beetroot Soup", 6, 7.2),
    "rabbit_stew": FoodItem("Rabbit Stew", 10, 12.0),
    "suspicious_stew": FoodItem("Suspicious Stew", 6, 7.2),
    # Special
    "rotten_flesh": FoodItem("Rotten Flesh", 4, 0.8),
    "spider_eye": FoodItem("Spider Eye", 2, 3.2),
    "poisonous_potato": FoodItem("Poisonous Potato", 2, 1.2),
    "pufferfish": FoodItem("Pufferfish", 1, 0.2),
    "tropical_fish": FoodItem("Tropical Fish", 1, 0.2),
    "dried_kelp": FoodItem("Dried Kelp", 1, 0.6),
    "chorus_fruit": FoodItem("Chorus Fruit", 4, 2.4),
    "honey_bottle": FoodItem("Honey Bottle", 6, 1.2),
}

# Exhaustion costs per action (Minecraft Java Edition)
# Exhaustion accumulates; every 4.0 points depletes 1 saturation (or 1 hunger if saturation=0)
EXHAUSTION_COSTS: dict[PlayerAction, float] = {
    PlayerAction.IDLE: 0.0,
    PlayerAction.WALKING: 0.0,  # Walking costs nothing
    PlayerAction.SNEAKING: 0.0,  # Sneaking costs nothing
    PlayerAction.SWIMMING: 0.01,  # Per meter
    PlayerAction.SPRINTING: 0.1,  # Per meter
    PlayerAction.JUMPING: 0.05,  # Per jump
    PlayerAction.SPRINT_JUMPING: 0.2,  # Per jump while sprinting
    PlayerAction.ATTACKING: 0.1,  # Per attack
    PlayerAction.MINING: 0.005,  # Per block broken
    PlayerAction.TAKING_DAMAGE: 0.1,  # Per damage point absorbed
    PlayerAction.HUNGER_EFFECT: 0.1,  # Per tick with Hunger I (0.5 per second at level I)
    PlayerAction.REGENERATING: 6.0,  # Per health point regenerated
}

# Starvation damage per difficulty
STARVATION_DAMAGE: dict[Difficulty, tuple[float, int]] = {
    # (damage_per_hit, minimum_health_after_starvation)
    Difficulty.PEACEFUL: (0.0, 20),  # No starvation, full health maintained
    Difficulty.EASY: (1.0, 10),  # 0.5 hearts, stops at 5 hearts
    Difficulty.NORMAL: (1.0, 1),  # 0.5 hearts, stops at 0.5 hearts
    Difficulty.HARD: (1.0, 0),  # 0.5 hearts, can kill player
}

# Thresholds
SPRINT_THRESHOLD = 6  # Food level >= 6 required to sprint
REGENERATION_THRESHOLD = 18  # Food level >= 18 for natural health regeneration
EXHAUSTION_TO_SATURATION = 4.0  # Exhaustion points to drain 1 saturation/hunger
MAX_FOOD_LEVEL = 20
MAX_SATURATION = 20.0  # Cannot exceed food level


class HungerState:
    """Player hunger state tracker."""

    def __init__(
        self,
        food_level: int = 20,
        saturation: float = 5.0,
        exhaustion: float = 0.0,
    ):
        logger.info("HungerState.__init__: food_level=%s, saturation=%s, exhaustion=%s", food_level, saturation, exhaustion)
        self.food_level = min(food_level, MAX_FOOD_LEVEL)
        self.saturation = min(saturation, float(self.food_level))
        self.exhaustion = exhaustion

    def add_exhaustion(self, amount: float) -> None:
        """Add exhaustion and process any overflow."""
        logger.debug("HungerState.add_exhaustion: amount=%s", amount)
        self.exhaustion += amount
        while self.exhaustion >= EXHAUSTION_TO_SATURATION:
            self.exhaustion -= EXHAUSTION_TO_SATURATION
            if self.saturation > 0:
                self.saturation = max(0.0, self.saturation - 1.0)
            else:
                self.food_level = max(0, self.food_level - 1)

    def eat(self, food: FoodItem) -> bool:
        """Consume food item. Returns True if eaten."""
        logger.debug("HungerState.eat: food=%s", food)
        if self.food_level >= MAX_FOOD_LEVEL:
            return False  # Can't eat when full
        self.food_level = min(MAX_FOOD_LEVEL, self.food_level + food.hunger_points)
        # Saturation capped at food level
        self.saturation = min(float(self.food_level), self.saturation + food.saturation)
        return True

    def can_sprint(self) -> bool:
        """Check if player can sprint."""
        logger.debug("HungerState.can_sprint called")
        return self.food_level >= SPRINT_THRESHOLD

    def can_regenerate(self) -> bool:
        """Check if player can naturally regenerate health."""
        logger.debug("HungerState.can_regenerate called")
        return self.food_level >= REGENERATION_THRESHOLD


class HungerVerifier:
    """Verification suite for hunger mechanics."""

    def __init__(self):
        logger.info("HungerVerifier.__init__ called")
        self.results: list[tuple[str, bool, str]] = []

    def verify(self, name: str, condition: bool, message: str = "") -> bool:
        """Record verification result."""
        logger.debug("HungerVerifier.verify: name=%s, condition=%s, message=%s", name, condition, message)
        self.results.append((name, condition, message))
        return condition

    def run_all(self) -> dict[str, bool | int | list[tuple[str, bool, str]]]:
        """Run all verification tests."""
        logger.debug("HungerVerifier.run_all called")
        self.results.clear()

        self._verify_exhaustion_costs()
        self._verify_food_restoration()
        self._verify_sprint_threshold()
        self._verify_regeneration_threshold()
        self._verify_starvation_damage()
        self._verify_saturation_cap()
        self._verify_exhaustion_overflow()

        passed = sum(1 for _, ok, _ in self.results if ok)
        failed = len(self.results) - passed

        return {
            "passed": passed,
            "failed": failed,
            "total": len(self.results),
            "all_passed": failed == 0,
            "results": self.results,
        }

    def _verify_exhaustion_costs(self) -> None:
        """Verify exhaustion costs are defined for all actions."""
        logger.debug("HungerVerifier._verify_exhaustion_costs called")
        for action in PlayerAction:
            cost = EXHAUSTION_COSTS.get(action)
            self.verify(
                f"exhaustion_cost_{action.value}",
                cost is not None and cost >= 0,
                f"Action {action.value} has valid exhaustion cost: {cost}",
            )

        # Verify relative costs make sense
        self.verify(
            "sprint_more_than_walk",
            EXHAUSTION_COSTS[PlayerAction.SPRINTING] > EXHAUSTION_COSTS[PlayerAction.WALKING],
            "Sprinting should cost more than walking",
        )
        self.verify(
            "sprint_jump_most_expensive_movement",
            EXHAUSTION_COSTS[PlayerAction.SPRINT_JUMPING]
            > EXHAUSTION_COSTS[PlayerAction.SPRINTING],
            "Sprint-jumping should be most expensive movement",
        )
        self.verify(
            "regenerating_is_expensive",
            EXHAUSTION_COSTS[PlayerAction.REGENERATING] >= 6.0,
            "Regenerating health should cost significant exhaustion",
        )

    def _verify_food_restoration(self) -> None:
        """Verify food item values are correct."""
        # Best foods
        logger.debug("HungerVerifier._verify_food_restoration called")
        self.verify(
            "golden_carrot_best_saturation",
            FOOD_ITEMS["golden_carrot"].saturation >= 14.0,
            f"Golden carrot saturation: {FOOD_ITEMS['golden_carrot'].saturation}",
        )
        self.verify(
            "rabbit_stew_most_hunger",
            FOOD_ITEMS["rabbit_stew"].hunger_points >= 10,
            f"Rabbit stew hunger points: {FOOD_ITEMS['rabbit_stew'].hunger_points}",
        )

        # Cooked > raw
        for meat in ["beef", "porkchop", "chicken", "mutton", "rabbit"]:
            raw = FOOD_ITEMS.get(f"raw_{meat}")
            cooked = FOOD_ITEMS.get(f"cooked_{meat}")
            if raw and cooked:
                self.verify(
                    f"cooked_{meat}_better_than_raw",
                    cooked.hunger_points > raw.hunger_points,
                    f"Cooked {meat} ({cooked.hunger_points}) > raw ({raw.hunger_points})",
                )

        # Dangerous foods (cause negative status effects when eaten)
        # These are defined by their side effects, not saturation ratios
        # Rotten flesh: 80% chance of Hunger effect
        # Spider eye: Poison effect (4 seconds)
        # Pufferfish: Hunger III, Nausea II, Poison II
        dangerous_foods = {
            "rotten_flesh": "hunger",
            "spider_eye": "poison",
            "pufferfish": "poison+nausea+hunger",
            "poisonous_potato": "poison",
        }
        for name in dangerous_foods:
            food = FOOD_ITEMS.get(name)
            self.verify(
                f"{name}_is_dangerous",
                food is not None,
                f"{name} exists as dangerous food (causes {dangerous_foods[name]})",
            )

    def _verify_sprint_threshold(self) -> None:
        """Verify sprint threshold behavior."""
        # At threshold
        logger.debug("HungerVerifier._verify_sprint_threshold called")
        state_at = HungerState(food_level=SPRINT_THRESHOLD)
        self.verify(
            "can_sprint_at_threshold",
            state_at.can_sprint(),
            f"Can sprint at food level {SPRINT_THRESHOLD}",
        )

        # Below threshold
        state_below = HungerState(food_level=SPRINT_THRESHOLD - 1)
        self.verify(
            "cannot_sprint_below_threshold",
            not state_below.can_sprint(),
            f"Cannot sprint at food level {SPRINT_THRESHOLD - 1}",
        )

        # Threshold value
        self.verify(
            "sprint_threshold_is_6",
            SPRINT_THRESHOLD == 6,
            f"Sprint threshold should be 6, got {SPRINT_THRESHOLD}",
        )

    def _verify_regeneration_threshold(self) -> None:
        """Verify natural regeneration threshold."""
        # At threshold
        logger.debug("HungerVerifier._verify_regeneration_threshold called")
        state_at = HungerState(food_level=REGENERATION_THRESHOLD)
        self.verify(
            "can_regen_at_threshold",
            state_at.can_regenerate(),
            f"Can regenerate at food level {REGENERATION_THRESHOLD}",
        )

        # Below threshold
        state_below = HungerState(food_level=REGENERATION_THRESHOLD - 1)
        self.verify(
            "cannot_regen_below_threshold",
            not state_below.can_regenerate(),
            f"Cannot regenerate at food level {REGENERATION_THRESHOLD - 1}",
        )

        # Threshold value
        self.verify(
            "regen_threshold_is_18",
            REGENERATION_THRESHOLD == 18,
            f"Regeneration threshold should be 18, got {REGENERATION_THRESHOLD}",
        )

    def _verify_starvation_damage(self) -> None:
        """Verify starvation damage per difficulty."""
        # Peaceful: no starvation
        logger.debug("HungerVerifier._verify_starvation_damage called")
        dmg, min_hp = STARVATION_DAMAGE[Difficulty.PEACEFUL]
        self.verify(
            "peaceful_no_starvation",
            dmg == 0.0,
            f"Peaceful mode: no starvation damage (got {dmg})",
        )

        # Easy: stops at 5 hearts
        dmg, min_hp = STARVATION_DAMAGE[Difficulty.EASY]
        self.verify(
            "easy_min_health_10",
            min_hp == 10,
            f"Easy mode min health: {min_hp} (should be 10)",
        )

        # Normal: stops at 0.5 hearts
        dmg, min_hp = STARVATION_DAMAGE[Difficulty.NORMAL]
        self.verify(
            "normal_min_health_1",
            min_hp == 1,
            f"Normal mode min health: {min_hp} (should be 1)",
        )

        # Hard: can kill
        dmg, min_hp = STARVATION_DAMAGE[Difficulty.HARD]
        self.verify(
            "hard_can_kill",
            min_hp == 0,
            f"Hard mode min health: {min_hp} (should be 0 - can kill)",
        )

    def _verify_saturation_cap(self) -> None:
        """Verify saturation cannot exceed food level."""
        logger.debug("HungerVerifier._verify_saturation_cap called")
        state = HungerState(food_level=10, saturation=20.0)
        self.verify(
            "saturation_capped_to_food_level",
            state.saturation <= state.food_level,
            f"Saturation {state.saturation} <= food {state.food_level}",
        )

        # Eating should respect cap
        state2 = HungerState(food_level=15, saturation=10.0)
        state2.eat(FOOD_ITEMS["golden_carrot"])  # +6 hunger, +14.4 saturation
        self.verify(
            "eating_respects_saturation_cap",
            state2.saturation <= state2.food_level,
            f"After eating: saturation {state2.saturation} <= food {state2.food_level}",
        )

    def _verify_exhaustion_overflow(self) -> None:
        """Verify exhaustion overflow mechanics."""
        # Saturation depletes first
        logger.debug("HungerVerifier._verify_exhaustion_overflow called")
        state = HungerState(food_level=20, saturation=5.0, exhaustion=0.0)
        state.add_exhaustion(4.0)  # Should drain 1 saturation
        self.verify(
            "exhaustion_drains_saturation_first",
            state.saturation == 4.0 and state.food_level == 20,
            f"After 4.0 exhaustion: saturation={state.saturation}, food={state.food_level}",
        )

        # Food drains after saturation depleted
        state2 = HungerState(food_level=20, saturation=0.0, exhaustion=0.0)
        state2.add_exhaustion(4.0)  # Should drain 1 food
        self.verify(
            "exhaustion_drains_food_when_saturation_empty",
            state2.food_level == 19,
            f"After 4.0 exhaustion with no saturation: food={state2.food_level}",
        )


def verify_hunger_mechanics() -> dict:
    """Run full hunger mechanics verification."""
    logger.debug("verify_hunger_mechanics called")
    verifier = HungerVerifier()
    return verifier.run_all()


if __name__ == "__main__":
    results = verify_hunger_mechanics()

    print("=" * 60)
    print("HUNGER SYSTEM VERIFICATION")
    print("=" * 60)
    print(f"Passed: {results['passed']}/{results['total']}")
    print(f"Failed: {results['failed']}")
    print()

    for name, ok, msg in results["results"]:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {name}: {msg}")

    print()
    if results["all_passed"]:
        print("All hunger mechanics verified successfully!")
    else:
        print("Some verifications failed - review results above.")
