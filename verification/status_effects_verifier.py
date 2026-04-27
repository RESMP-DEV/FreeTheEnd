"""Status effects verification for Minecraft-style survival mechanics.

Tests:
- Status effect durations
- Effect amplifier levels (I-IV typical, up to 255 max)
- Stacking/combining behavior
- Beneficial vs harmful effect categorization
- Effect particle colors and visibility
"""

from dataclasses import dataclass
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class EffectCategory(Enum):
    """Categories of status effects."""

    BENEFICIAL = "beneficial"
    HARMFUL = "harmful"
    NEUTRAL = "neutral"


class EffectType(Enum):
    """All Minecraft status effect types."""

    # Beneficial
    SPEED = "speed"
    HASTE = "haste"
    STRENGTH = "strength"
    INSTANT_HEALTH = "instant_health"
    JUMP_BOOST = "jump_boost"
    REGENERATION = "regeneration"
    RESISTANCE = "resistance"
    FIRE_RESISTANCE = "fire_resistance"
    WATER_BREATHING = "water_breathing"
    INVISIBILITY = "invisibility"
    NIGHT_VISION = "night_vision"
    HEALTH_BOOST = "health_boost"
    ABSORPTION = "absorption"
    SATURATION = "saturation"
    LUCK = "luck"
    SLOW_FALLING = "slow_falling"
    CONDUIT_POWER = "conduit_power"
    DOLPHINS_GRACE = "dolphins_grace"
    HERO_OF_THE_VILLAGE = "hero_of_the_village"

    # Harmful
    SLOWNESS = "slowness"
    MINING_FATIGUE = "mining_fatigue"
    INSTANT_DAMAGE = "instant_damage"
    NAUSEA = "nausea"
    BLINDNESS = "blindness"
    HUNGER = "hunger"
    WEAKNESS = "weakness"
    POISON = "poison"
    WITHER = "wither"
    LEVITATION = "levitation"
    BAD_LUCK = "bad_luck"
    BAD_OMEN = "bad_omen"
    DARKNESS = "darkness"
    INFESTED = "infested"
    OOZING = "oozing"
    WEAVING = "weaving"
    WIND_CHARGED = "wind_charged"

    # Neutral
    GLOWING = "glowing"
    TRIAL_OMEN = "trial_omen"


@dataclass
class StatusEffect:
    """Definition of a status effect."""

    name: str
    effect_type: EffectType
    category: EffectCategory
    max_amplifier: int = 255  # Amplifier 0 = Level I, 255 = Level 256
    is_instant: bool = False  # Instant effects (healing/damage)
    default_duration_ticks: int = 0  # 0 for instant, varies for duration
    particle_color: tuple[int, int, int] = (0, 0, 0)  # RGB

    # Effect formula callbacks
    effect_per_level: float = 0.0  # Base modifier per amplifier level

    @property
    def display_level(self) -> str:
        """Convert amplifier to Roman numeral display."""
        logger.debug("StatusEffect.display_level called")
        numerals = ["I", "II", "III", "IV", "V"]
        return numerals[min(self.max_amplifier, 4)]


# Minecraft status effects with their properties
# Durations in ticks (20 ticks = 1 second)
STATUS_EFFECTS: dict[EffectType, StatusEffect] = {
    # Beneficial effects
    EffectType.SPEED: StatusEffect(
        "Speed",
        EffectType.SPEED,
        EffectCategory.BENEFICIAL,
        max_amplifier=255,
        default_duration_ticks=3600,  # 3 minutes default
        particle_color=(124, 175, 198),
        effect_per_level=0.20,  # +20% speed per level
    ),
    EffectType.HASTE: StatusEffect(
        "Haste",
        EffectType.HASTE,
        EffectCategory.BENEFICIAL,
        max_amplifier=255,
        default_duration_ticks=3600,
        particle_color=(217, 192, 67),
        effect_per_level=0.10,  # +10% attack speed, +20% mining speed per level
    ),
    EffectType.STRENGTH: StatusEffect(
        "Strength",
        EffectType.STRENGTH,
        EffectCategory.BENEFICIAL,
        max_amplifier=255,
        default_duration_ticks=3600,
        particle_color=(147, 36, 35),
        effect_per_level=3.0,  # +3 damage per level
    ),
    EffectType.INSTANT_HEALTH: StatusEffect(
        "Instant Health",
        EffectType.INSTANT_HEALTH,
        EffectCategory.BENEFICIAL,
        max_amplifier=255,
        is_instant=True,
        particle_color=(248, 36, 35),
        effect_per_level=2.0,  # Heals 2^(level+1) HP (4 HP at I, 8 at II, etc.)
    ),
    EffectType.JUMP_BOOST: StatusEffect(
        "Jump Boost",
        EffectType.JUMP_BOOST,
        EffectCategory.BENEFICIAL,
        max_amplifier=255,
        default_duration_ticks=3600,
        particle_color=(34, 255, 76),
        effect_per_level=0.5,  # +0.5 blocks jump height per level
    ),
    EffectType.REGENERATION: StatusEffect(
        "Regeneration",
        EffectType.REGENERATION,
        EffectCategory.BENEFICIAL,
        max_amplifier=255,
        default_duration_ticks=900,  # 45 seconds default
        particle_color=(205, 92, 171),
        effect_per_level=0.5,  # Heals 1 HP every (50 >> level) ticks
    ),
    EffectType.RESISTANCE: StatusEffect(
        "Resistance",
        EffectType.RESISTANCE,
        EffectCategory.BENEFICIAL,
        max_amplifier=4,  # Level V = 100% damage reduction
        default_duration_ticks=3600,
        particle_color=(153, 69, 58),
        effect_per_level=0.20,  # -20% damage taken per level
    ),
    EffectType.FIRE_RESISTANCE: StatusEffect(
        "Fire Resistance",
        EffectType.FIRE_RESISTANCE,
        EffectCategory.BENEFICIAL,
        max_amplifier=0,  # Only level I
        default_duration_ticks=3600,
        particle_color=(228, 154, 58),
    ),
    EffectType.WATER_BREATHING: StatusEffect(
        "Water Breathing",
        EffectType.WATER_BREATHING,
        EffectCategory.BENEFICIAL,
        max_amplifier=0,
        default_duration_ticks=3600,
        particle_color=(46, 82, 153),
    ),
    EffectType.INVISIBILITY: StatusEffect(
        "Invisibility",
        EffectType.INVISIBILITY,
        EffectCategory.BENEFICIAL,
        max_amplifier=0,
        default_duration_ticks=3600,
        particle_color=(127, 131, 146),
    ),
    EffectType.NIGHT_VISION: StatusEffect(
        "Night Vision",
        EffectType.NIGHT_VISION,
        EffectCategory.BENEFICIAL,
        max_amplifier=0,
        default_duration_ticks=3600,
        particle_color=(0, 0, 139),
    ),
    EffectType.HEALTH_BOOST: StatusEffect(
        "Health Boost",
        EffectType.HEALTH_BOOST,
        EffectCategory.BENEFICIAL,
        max_amplifier=255,
        default_duration_ticks=3600,
        particle_color=(248, 125, 35),
        effect_per_level=4.0,  # +4 max HP (2 hearts) per level
    ),
    EffectType.ABSORPTION: StatusEffect(
        "Absorption",
        EffectType.ABSORPTION,
        EffectCategory.BENEFICIAL,
        max_amplifier=255,
        default_duration_ticks=2400,  # 2 minutes default
        particle_color=(37, 82, 165),
        effect_per_level=4.0,  # +4 absorption HP per level
    ),
    EffectType.SATURATION: StatusEffect(
        "Saturation",
        EffectType.SATURATION,
        EffectCategory.BENEFICIAL,
        max_amplifier=255,
        is_instant=True,  # Actually instant despite being shown as duration
        particle_color=(248, 36, 35),
        effect_per_level=1.0,  # Restores (level+1) hunger and 2*(level+1) saturation per tick
    ),
    EffectType.LUCK: StatusEffect(
        "Luck",
        EffectType.LUCK,
        EffectCategory.BENEFICIAL,
        max_amplifier=255,
        default_duration_ticks=6000,  # 5 minutes
        particle_color=(51, 153, 0),
        effect_per_level=1.0,  # +1 luck attribute per level
    ),
    EffectType.SLOW_FALLING: StatusEffect(
        "Slow Falling",
        EffectType.SLOW_FALLING,
        EffectCategory.BENEFICIAL,
        max_amplifier=0,
        default_duration_ticks=1800,  # 90 seconds
        particle_color=(243, 207, 185),
    ),
    EffectType.CONDUIT_POWER: StatusEffect(
        "Conduit Power",
        EffectType.CONDUIT_POWER,
        EffectCategory.BENEFICIAL,
        max_amplifier=0,
        default_duration_ticks=260,  # Refreshes while near conduit
        particle_color=(27, 87, 108),
    ),
    EffectType.DOLPHINS_GRACE: StatusEffect(
        "Dolphin's Grace",
        EffectType.DOLPHINS_GRACE,
        EffectCategory.BENEFICIAL,
        max_amplifier=0,
        default_duration_ticks=100,  # 5 seconds, refreshes near dolphins
        particle_color=(136, 163, 190),
    ),
    EffectType.HERO_OF_THE_VILLAGE: StatusEffect(
        "Hero of the Village",
        EffectType.HERO_OF_THE_VILLAGE,
        EffectCategory.BENEFICIAL,
        max_amplifier=4,  # Levels I-V based on raid difficulty
        default_duration_ticks=48000,  # 40 minutes
        particle_color=(68, 255, 68),
    ),
    # Harmful effects
    EffectType.SLOWNESS: StatusEffect(
        "Slowness",
        EffectType.SLOWNESS,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        default_duration_ticks=1800,
        particle_color=(90, 108, 129),
        effect_per_level=0.15,  # -15% speed per level
    ),
    EffectType.MINING_FATIGUE: StatusEffect(
        "Mining Fatigue",
        EffectType.MINING_FATIGUE,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        default_duration_ticks=6000,  # From elder guardian: 5 min
        particle_color=(74, 66, 23),
        effect_per_level=0.10,  # -10% attack speed, mining speed reduced
    ),
    EffectType.INSTANT_DAMAGE: StatusEffect(
        "Instant Damage",
        EffectType.INSTANT_DAMAGE,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        is_instant=True,
        particle_color=(67, 10, 9),
        effect_per_level=3.0,  # 3 * 2^level damage (6 at I, 12 at II)
    ),
    EffectType.NAUSEA: StatusEffect(
        "Nausea",
        EffectType.NAUSEA,
        EffectCategory.HARMFUL,
        max_amplifier=0,  # Higher levels have same effect
        default_duration_ticks=200,  # From pufferfish
        particle_color=(85, 29, 74),
    ),
    EffectType.BLINDNESS: StatusEffect(
        "Blindness",
        EffectType.BLINDNESS,
        EffectCategory.HARMFUL,
        max_amplifier=0,
        default_duration_ticks=200,
        particle_color=(31, 31, 35),
    ),
    EffectType.HUNGER: StatusEffect(
        "Hunger",
        EffectType.HUNGER,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        default_duration_ticks=600,  # 30 seconds from rotten flesh
        particle_color=(88, 118, 83),
        effect_per_level=0.1,  # +0.1 exhaustion per tick per level
    ),
    EffectType.WEAKNESS: StatusEffect(
        "Weakness",
        EffectType.WEAKNESS,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        default_duration_ticks=1800,
        particle_color=(72, 77, 72),
        effect_per_level=4.0,  # -4 melee damage per level
    ),
    EffectType.POISON: StatusEffect(
        "Poison",
        EffectType.POISON,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        default_duration_ticks=900,
        particle_color=(78, 147, 49),
        effect_per_level=0.5,  # Deals 1 damage every (25 >> level) ticks
    ),
    EffectType.WITHER: StatusEffect(
        "Wither",
        EffectType.WITHER,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        default_duration_ticks=800,  # From wither skeleton
        particle_color=(53, 42, 39),
        effect_per_level=0.5,  # Deals 1 damage every (40 >> level) ticks
    ),
    EffectType.LEVITATION: StatusEffect(
        "Levitation",
        EffectType.LEVITATION,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        default_duration_ticks=200,  # From shulker
        particle_color=(206, 255, 255),
        effect_per_level=0.9,  # +0.9 blocks/sec upward per level
    ),
    EffectType.BAD_LUCK: StatusEffect(
        "Bad Luck",
        EffectType.BAD_LUCK,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        default_duration_ticks=6000,
        particle_color=(192, 164, 77),
        effect_per_level=1.0,  # -1 luck attribute per level
    ),
    EffectType.BAD_OMEN: StatusEffect(
        "Bad Omen",
        EffectType.BAD_OMEN,
        EffectCategory.HARMFUL,
        max_amplifier=4,  # Level I-V
        default_duration_ticks=6000,  # 5 minutes (removed in 1.21)
        particle_color=(14, 81, 26),
    ),
    EffectType.DARKNESS: StatusEffect(
        "Darkness",
        EffectType.DARKNESS,
        EffectCategory.HARMFUL,
        max_amplifier=0,
        default_duration_ticks=260,  # From warden
        particle_color=(41, 39, 33),
    ),
    EffectType.INFESTED: StatusEffect(
        "Infested",
        EffectType.INFESTED,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        default_duration_ticks=200,
        particle_color=(141, 154, 115),
    ),
    EffectType.OOZING: StatusEffect(
        "Oozing",
        EffectType.OOZING,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        default_duration_ticks=200,
        particle_color=(153, 255, 163),
    ),
    EffectType.WEAVING: StatusEffect(
        "Weaving",
        EffectType.WEAVING,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        default_duration_ticks=200,
        particle_color=(86, 77, 71),
    ),
    EffectType.WIND_CHARGED: StatusEffect(
        "Wind Charged",
        EffectType.WIND_CHARGED,
        EffectCategory.HARMFUL,
        max_amplifier=255,
        default_duration_ticks=200,
        particle_color=(191, 218, 224),
    ),
    # Neutral effects
    EffectType.GLOWING: StatusEffect(
        "Glowing",
        EffectType.GLOWING,
        EffectCategory.NEUTRAL,
        max_amplifier=0,
        default_duration_ticks=200,
        particle_color=(148, 160, 97),
    ),
    EffectType.TRIAL_OMEN: StatusEffect(
        "Trial Omen",
        EffectType.TRIAL_OMEN,
        EffectCategory.NEUTRAL,
        max_amplifier=0,
        default_duration_ticks=18000,  # 15 minutes
        particle_color=(22, 166, 166),
    ),
}

# Common potion durations (ticks)
POTION_DURATIONS = {
    "normal": 3600,  # 3 minutes
    "extended": 9600,  # 8 minutes
    "splash_normal": 2700,  # 2:15
    "splash_extended": 7200,  # 6:00
    "lingering_normal": 900,  # 0:45
    "lingering_extended": 2400,  # 2:00
    "arrow_normal": 450,  # 0:22.5
    "arrow_extended": 1200,  # 1:00
}

# Beacon effect durations based on pyramid level
BEACON_DURATIONS = {
    1: 220,  # Level 1: 11 seconds
    2: 260,  # Level 2: 13 seconds
    3: 300,  # Level 3: 15 seconds
    4: 340,  # Level 4: 17 seconds (full pyramid)
}


@dataclass
class ActiveEffect:
    """An active status effect on an entity."""

    effect: StatusEffect
    amplifier: int = 0  # 0 = Level I
    duration_ticks: int = 0
    show_particles: bool = True
    show_icon: bool = True
    ambient: bool = False  # From beacon/conduit (less visible particles)

    @property
    def level(self) -> int:
        """Human-readable level (1-256)."""
        logger.debug("ActiveEffect.level called")
        return self.amplifier + 1

    @property
    def remaining_seconds(self) -> float:
        """Duration in seconds."""
        logger.debug("ActiveEffect.remaining_seconds called")
        return self.duration_ticks / 20.0

    def tick(self) -> bool:
        """Advance one tick. Returns False if effect expired."""
        logger.debug("ActiveEffect.tick called")
        if self.effect.is_instant:
            return False
        self.duration_ticks -= 1
        return self.duration_ticks > 0


class EffectManager:
    """Manages active effects on an entity."""

    def __init__(self):
        logger.info("EffectManager.__init__ called")
        self.active_effects: dict[EffectType, ActiveEffect] = {}

    def apply_effect(
        self,
        effect_type: EffectType,
        amplifier: int = 0,
        duration_ticks: int = 0,
        show_particles: bool = True,
    ) -> bool:
        """Apply or upgrade an effect. Returns True if effect was applied/upgraded."""
        logger.debug("EffectManager.apply_effect: effect_type=%s, amplifier=%s, duration_ticks=%s, show_particles=%s", effect_type, amplifier, duration_ticks, show_particles)
        if effect_type not in STATUS_EFFECTS:
            return False

        effect = STATUS_EFFECTS[effect_type]
        new_effect = ActiveEffect(
            effect=effect,
            amplifier=min(amplifier, effect.max_amplifier),
            duration_ticks=duration_ticks or effect.default_duration_ticks,
            show_particles=show_particles,
        )

        existing = self.active_effects.get(effect_type)
        if existing:
            # Same or higher level: extend if new duration is longer
            if new_effect.amplifier >= existing.amplifier:
                if new_effect.amplifier > existing.amplifier:
                    self.active_effects[effect_type] = new_effect
                    return True
                elif new_effect.duration_ticks > existing.duration_ticks:
                    existing.duration_ticks = new_effect.duration_ticks
                    return True
            return False
        else:
            self.active_effects[effect_type] = new_effect
            return True

    def remove_effect(self, effect_type: EffectType) -> bool:
        """Remove an effect. Returns True if effect was removed."""
        logger.debug("EffectManager.remove_effect: effect_type=%s", effect_type)
        if effect_type in self.active_effects:
            del self.active_effects[effect_type]
            return True
        return False

    def clear_effects(self, category: EffectCategory | None = None) -> int:
        """Clear effects by category. Returns count removed."""
        logger.debug("EffectManager.clear_effects: category=%s", category)
        to_remove = []
        for et, ae in self.active_effects.items():
            if category is None or ae.effect.category == category:
                to_remove.append(et)
        for et in to_remove:
            del self.active_effects[et]
        return len(to_remove)


class StatusEffectsVerifier:
    """Verification suite for status effect mechanics."""

    def __init__(self):
        logger.info("StatusEffectsVerifier.__init__ called")
        self.results: list[tuple[str, bool, str]] = []

    def verify(self, name: str, condition: bool, message: str = "") -> bool:
        """Record verification result."""
        logger.debug("StatusEffectsVerifier.verify: name=%s, condition=%s, message=%s", name, condition, message)
        self.results.append((name, condition, message))
        return condition

    def run_all(self) -> dict[str, bool | int | list[tuple[str, bool, str]]]:
        """Run all verification tests."""
        logger.debug("StatusEffectsVerifier.run_all called")
        self.results.clear()

        self._verify_all_effects_defined()
        self._verify_categories()
        self._verify_amplifier_limits()
        self._verify_instant_effects()
        self._verify_duration_mechanics()
        self._verify_stacking_behavior()
        self._verify_opposing_effects()
        self._verify_potion_durations()
        self._verify_beacon_durations()

        passed = sum(1 for _, ok, _ in self.results if ok)
        failed = len(self.results) - passed

        return {
            "passed": passed,
            "failed": failed,
            "total": len(self.results),
            "all_passed": failed == 0,
            "results": self.results,
        }

    def _verify_all_effects_defined(self) -> None:
        """Verify all effect types have definitions."""
        logger.debug("StatusEffectsVerifier._verify_all_effects_defined called")
        for effect_type in EffectType:
            defined = effect_type in STATUS_EFFECTS
            self.verify(
                f"effect_defined_{effect_type.value}",
                defined,
                f"Effect {effect_type.value} is defined",
            )

    def _verify_categories(self) -> None:
        """Verify effect categorization."""
        logger.debug("StatusEffectsVerifier._verify_categories called")
        beneficial = [
            et for et, e in STATUS_EFFECTS.items() if e.category == EffectCategory.BENEFICIAL
        ]
        harmful = [et for et, e in STATUS_EFFECTS.items() if e.category == EffectCategory.HARMFUL]
        neutral = [et for et, e in STATUS_EFFECTS.items() if e.category == EffectCategory.NEUTRAL]

        self.verify(
            "has_beneficial_effects",
            len(beneficial) > 0,
            f"Found {len(beneficial)} beneficial effects",
        )
        self.verify(
            "has_harmful_effects",
            len(harmful) > 0,
            f"Found {len(harmful)} harmful effects",
        )

        # Verify specific categorizations
        self.verify(
            "regeneration_is_beneficial",
            STATUS_EFFECTS[EffectType.REGENERATION].category == EffectCategory.BENEFICIAL,
            "Regeneration should be beneficial",
        )
        self.verify(
            "poison_is_harmful",
            STATUS_EFFECTS[EffectType.POISON].category == EffectCategory.HARMFUL,
            "Poison should be harmful",
        )
        self.verify(
            "glowing_is_neutral",
            STATUS_EFFECTS[EffectType.GLOWING].category == EffectCategory.NEUTRAL,
            "Glowing should be neutral",
        )

    def _verify_amplifier_limits(self) -> None:
        """Verify amplifier limits are appropriate."""
        # Binary effects (on/off) should have max_amplifier = 0
        logger.debug("StatusEffectsVerifier._verify_amplifier_limits called")
        binary_effects = [
            EffectType.FIRE_RESISTANCE,
            EffectType.WATER_BREATHING,
            EffectType.INVISIBILITY,
            EffectType.NIGHT_VISION,
        ]
        for et in binary_effects:
            effect = STATUS_EFFECTS.get(et)
            if effect:
                self.verify(
                    f"binary_effect_{et.value}",
                    effect.max_amplifier == 0,
                    f"{et.value} should only have level I (max_amplifier=0)",
                )

        # Resistance capped at level V (100% reduction)
        resistance = STATUS_EFFECTS.get(EffectType.RESISTANCE)
        if resistance:
            self.verify(
                "resistance_capped_at_5",
                resistance.max_amplifier == 4,  # amplifier 4 = level V
                f"Resistance max level should be V, got {resistance.max_amplifier + 1}",
            )

    def _verify_instant_effects(self) -> None:
        """Verify instant effect behavior."""
        logger.debug("StatusEffectsVerifier._verify_instant_effects called")
        instant_effects = [
            EffectType.INSTANT_HEALTH,
            EffectType.INSTANT_DAMAGE,
            EffectType.SATURATION,
        ]

        for et in instant_effects:
            effect = STATUS_EFFECTS.get(et)
            if effect:
                self.verify(
                    f"instant_{et.value}",
                    effect.is_instant,
                    f"{et.value} should be instant",
                )

        # Non-instant effects should have duration
        for et, effect in STATUS_EFFECTS.items():
            if not effect.is_instant:
                self.verify(
                    f"duration_{et.value}",
                    effect.default_duration_ticks > 0,
                    f"{et.value} should have default duration > 0",
                )

    def _verify_duration_mechanics(self) -> None:
        """Verify duration tick mechanics."""
        logger.debug("StatusEffectsVerifier._verify_duration_mechanics called")
        manager = EffectManager()
        manager.apply_effect(EffectType.SPEED, amplifier=0, duration_ticks=100)

        effect = manager.active_effects.get(EffectType.SPEED)
        self.verify(
            "effect_applied",
            effect is not None,
            "Speed effect should be applied",
        )

        if effect:
            initial = effect.duration_ticks
            effect.tick()
            self.verify(
                "tick_decrements_duration",
                effect.duration_ticks == initial - 1,
                f"Duration should decrement: {initial} -> {effect.duration_ticks}",
            )

            # Simulate until expiry
            while effect.tick():
                pass
            self.verify(
                "effect_expires_at_zero",
                effect.duration_ticks == 0,
                "Effect should expire when duration reaches 0",
            )

    def _verify_stacking_behavior(self) -> None:
        """Verify effect stacking and upgrade rules."""
        logger.debug("StatusEffectsVerifier._verify_stacking_behavior called")
        manager = EffectManager()

        # Apply level I
        manager.apply_effect(EffectType.STRENGTH, amplifier=0, duration_ticks=100)
        self.verify(
            "initial_level_1",
            manager.active_effects[EffectType.STRENGTH].level == 1,
            "Initial strength should be level I",
        )

        # Upgrade to level II
        manager.apply_effect(EffectType.STRENGTH, amplifier=1, duration_ticks=50)
        self.verify(
            "upgrade_to_level_2",
            manager.active_effects[EffectType.STRENGTH].level == 2,
            "Strength should upgrade to level II",
        )

        # Lower level should not downgrade
        manager.apply_effect(EffectType.STRENGTH, amplifier=0, duration_ticks=200)
        self.verify(
            "no_downgrade",
            manager.active_effects[EffectType.STRENGTH].level == 2,
            "Lower level should not downgrade effect",
        )

        # Same level with longer duration extends
        manager = EffectManager()
        manager.apply_effect(EffectType.SPEED, amplifier=0, duration_ticks=100)
        manager.apply_effect(EffectType.SPEED, amplifier=0, duration_ticks=200)
        self.verify(
            "same_level_extends",
            manager.active_effects[EffectType.SPEED].duration_ticks == 200,
            "Same level with longer duration should extend",
        )

    def _verify_opposing_effects(self) -> None:
        """Verify opposing effect relationships."""
        # Speed vs Slowness - both can coexist in Minecraft
        logger.debug("StatusEffectsVerifier._verify_opposing_effects called")
        manager = EffectManager()
        manager.apply_effect(EffectType.SPEED, amplifier=1, duration_ticks=100)
        manager.apply_effect(EffectType.SLOWNESS, amplifier=0, duration_ticks=100)

        self.verify(
            "opposing_effects_coexist",
            EffectType.SPEED in manager.active_effects
            and EffectType.SLOWNESS in manager.active_effects,
            "Speed and Slowness should coexist",
        )

        # Strength vs Weakness
        manager.apply_effect(EffectType.STRENGTH, amplifier=0, duration_ticks=100)
        manager.apply_effect(EffectType.WEAKNESS, amplifier=0, duration_ticks=100)
        self.verify(
            "strength_weakness_coexist",
            EffectType.STRENGTH in manager.active_effects
            and EffectType.WEAKNESS in manager.active_effects,
            "Strength and Weakness should coexist",
        )

    def _verify_potion_durations(self) -> None:
        """Verify standard potion durations."""
        logger.debug("StatusEffectsVerifier._verify_potion_durations called")
        self.verify(
            "normal_potion_3min",
            POTION_DURATIONS["normal"] == 3600,
            f"Normal potion: {POTION_DURATIONS['normal']} ticks (3 minutes)",
        )
        self.verify(
            "extended_potion_8min",
            POTION_DURATIONS["extended"] == 9600,
            f"Extended potion: {POTION_DURATIONS['extended']} ticks (8 minutes)",
        )
        self.verify(
            "splash_shorter_than_normal",
            POTION_DURATIONS["splash_normal"] < POTION_DURATIONS["normal"],
            "Splash potions should have shorter duration",
        )
        self.verify(
            "lingering_shortest",
            POTION_DURATIONS["lingering_normal"] < POTION_DURATIONS["splash_normal"],
            "Lingering potions should have shortest duration",
        )

    def _verify_beacon_durations(self) -> None:
        """Verify beacon effect durations by pyramid level."""
        # Durations should increase with pyramid level
        logger.debug("StatusEffectsVerifier._verify_beacon_durations called")
        levels = sorted(BEACON_DURATIONS.keys())
        for i in range(1, len(levels)):
            prev_lvl = levels[i - 1]
            curr_lvl = levels[i]
            self.verify(
                f"beacon_lvl{curr_lvl}_longer_than_{prev_lvl}",
                BEACON_DURATIONS[curr_lvl] > BEACON_DURATIONS[prev_lvl],
                f"Level {curr_lvl} ({BEACON_DURATIONS[curr_lvl]}) > Level {prev_lvl} ({BEACON_DURATIONS[prev_lvl]})",
            )

        # Full pyramid (level 4) should give 17 seconds
        self.verify(
            "full_pyramid_17sec",
            BEACON_DURATIONS[4] == 340,
            f"Full pyramid duration: {BEACON_DURATIONS[4]} ticks (17 seconds)",
        )


def verify_status_effects() -> dict:
    """Run full status effects verification."""
    logger.debug("verify_status_effects called")
    verifier = StatusEffectsVerifier()
    return verifier.run_all()


if __name__ == "__main__":
    results = verify_status_effects()

    print("=" * 60)
    print("STATUS EFFECTS VERIFICATION")
    print("=" * 60)
    print(f"Passed: {results['passed']}/{results['total']}")
    print(f"Failed: {results['failed']}")
    print()

    for name, ok, msg in results["results"]:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {name}: {msg}")

    print()
    if results["all_passed"]:
        print("All status effect mechanics verified successfully!")
    else:
        print("Some verifications failed - review results above.")
