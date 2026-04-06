"""Dragon Fight Subsystem Verification Module.

Verifies Minecraft Ender Dragon fight mechanics against Java Edition behavior.

NOTE: This verifier and the dragon AI now target MC 1.8.9 mechanics. In 1.8.9,
the dragon only has CIRCLING, STRAFING, CHARGING, and DYING phases. It never
lands on the fountain, never uses breath attacks, and has no perching phase.
The LANDING_APPROACH, LANDING, TAKEOFF, and HOVERING phases listed below are
only for reference/documentation of 1.9+ behavior.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

# =============================================================================
# Constants - Minecraft Java Edition Dragon Fight Values
# =============================================================================


class DragonPhase(Enum):
    """Dragon behavior phases matching Java Edition."""

    CIRCLING = auto()
    STRAFING = auto()  # Charging player
    LANDING_APPROACH = auto()
    LANDING = auto()  # On fountain perch
    TAKEOFF = auto()
    DYING = auto()
    HOVERING = auto()  # Near portal for breath attack


@dataclass(frozen=True)
class DragonConstants:
    """Java Edition dragon constants."""

    # Movement
    CIRCLING_RADIUS: float = 150.0  # Blocks from center
    CIRCLING_HEIGHT_MIN: float = 70.0  # Y coordinate
    CIRCLING_HEIGHT_MAX: float = 120.0
    CIRCLING_SPEED: float = 10.0  # Blocks per second
    STRAFE_SPEED: float = 25.0  # During charge attack
    LANDING_SPEED: float = 8.0

    # Targeting
    TARGET_RANGE: float = 150.0
    STRAFE_RANGE: float = 64.0  # Distance to initiate strafe

    # Combat
    HEALTH: float = 200.0
    HEAD_DAMAGE_MULTIPLIER: float = 4.0  # 4x damage to head
    BODY_DAMAGE_MULTIPLIER: float = 0.25  # 1/4 damage to body
    WING_DAMAGE_MULTIPLIER: float = 0.0  # Wings immune

    # Timing (ticks, 20 ticks = 1 second)
    PHASE_DURATION_MIN: int = 60  # 3 seconds minimum per phase
    CIRCLING_DURATION: int = 200  # ~10 seconds
    LANDING_DURATION: int = 200  # ~10 seconds on perch
    BREATH_ATTACK_DURATION: int = 100  # 5 seconds

    # Death sequence
    DEATH_ANIMATION_TICKS: int = 200  # 10 seconds
    XP_DROP_FIRST_KILL: int = 12000
    XP_DROP_SUBSEQUENT: int = 500

    # Crystal interaction
    CRYSTAL_HEAL_RATE: float = 1.0  # HP per tick when connected
    CRYSTAL_HEAL_RANGE: float = 32.0  # Max distance to heal
    CRYSTAL_EXPLOSION_POWER: float = 6.0  # TNT-scale explosion
    CRYSTAL_DAMAGE_TO_DRAGON: float = 10.0  # When crystal destroyed while healing


@dataclass(frozen=True)
class CrystalConstants:
    """End crystal constants."""

    EXPLOSION_POWER: float = 6.0  # Same as charged creeper
    EXPLOSION_DAMAGE_AT_CENTER: float = 97.0  # Unarmored at epicenter
    EXPLOSION_RADIUS: float = 12.0  # Damage radius in blocks
    HEAL_RANGE: float = 32.0
    HEAL_RATE_PER_TICK: float = 1.0  # 20 HP/second
    RESPAWN_POSITIONS: int = 10  # Pillars with crystals


@dataclass(frozen=True)
class PortalConstants:
    """Exit portal constants."""

    SPAWN_DELAY_TICKS: int = 20  # After dragon death animation
    CENTER_X: float = 0.5
    CENTER_Z: float = 0.5
    CENTER_Y: float = 64.0  # Bedrock platform Y level
    PORTAL_RADIUS: float = 4.0  # 9x9 portal area
    EGG_SPAWN_FIRST_ONLY: bool = True


# =============================================================================
# Data Classes for Test Scenarios
# =============================================================================


@dataclass
class Vec3:
    """3D vector for positions and velocities."""

    x: float
    y: float
    z: float

    def distance_to(self, other: Vec3) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def horizontal_distance_to(self, other: Vec3) -> float:
        dx = self.x - other.x
        dz = self.z - other.z
        return math.sqrt(dx * dx + dz * dz)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Vec3:
        return cls(d["x"], d["y"], d["z"])


@dataclass
class AABB:
    """Axis-aligned bounding box for hitboxes."""

    min_corner: Vec3
    max_corner: Vec3

    def contains(self, point: Vec3) -> bool:
        return (
            self.min_corner.x <= point.x <= self.max_corner.x
            and self.min_corner.y <= point.y <= self.max_corner.y
            and self.min_corner.z <= point.z <= self.max_corner.z
        )

    def width(self) -> float:
        return self.max_corner.x - self.min_corner.x

    def height(self) -> float:
        return self.max_corner.y - self.min_corner.y

    def depth(self) -> float:
        return self.max_corner.z - self.min_corner.z

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "min": self.min_corner.to_dict(),
            "max": self.max_corner.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, dict[str, float]]) -> AABB:
        return cls(Vec3.from_dict(d["min"]), Vec3.from_dict(d["max"]))


@dataclass
class DragonHitboxes:
    """Dragon multi-part hitboxes.

    Java Edition dragon has multiple hitbox parts with different damage modifiers:
    - Head: Small, takes 4x damage
    - Body: Large, takes 0.25x damage
    - Wings: Immune to damage
    - Tail segments: Takes normal damage
    """

    # Hitbox dimensions from Java decompiled code
    HEAD_WIDTH: float = 1.0
    HEAD_HEIGHT: float = 1.0

    BODY_WIDTH: float = 8.0
    BODY_HEIGHT: float = 4.0

    WING_WIDTH: float = 4.0
    WING_HEIGHT: float = 2.0

    TAIL_SEGMENT_WIDTH: float = 2.0
    TAIL_SEGMENT_HEIGHT: float = 2.0
    TAIL_SEGMENTS: int = 3

    def get_head_hitbox(self, dragon_pos: Vec3, yaw: float) -> AABB:
        """Calculate head hitbox based on dragon position and yaw."""
        # Head is offset forward from body center
        head_offset = 8.0  # Blocks forward from center

        # Calculate head position based on facing direction
        yaw_rad = math.radians(yaw)
        head_x = dragon_pos.x - math.sin(yaw_rad) * head_offset
        head_z = dragon_pos.z + math.cos(yaw_rad) * head_offset
        head_y = dragon_pos.y + 3.0  # Slightly elevated

        half_w = self.HEAD_WIDTH / 2
        half_h = self.HEAD_HEIGHT / 2

        return AABB(
            Vec3(head_x - half_w, head_y - half_h, head_z - half_w),
            Vec3(head_x + half_w, head_y + half_h, head_z + half_w),
        )

    def get_body_hitbox(self, dragon_pos: Vec3) -> AABB:
        """Calculate main body hitbox."""
        half_w = self.BODY_WIDTH / 2
        half_h = self.BODY_HEIGHT / 2

        return AABB(
            Vec3(dragon_pos.x - half_w, dragon_pos.y - half_h, dragon_pos.z - half_w),
            Vec3(dragon_pos.x + half_w, dragon_pos.y + half_h, dragon_pos.z + half_w),
        )

    def get_wing_hitboxes(self, dragon_pos: Vec3, yaw: float) -> list[AABB]:
        """Calculate wing hitboxes (immune to damage)."""
        wing_offset = 4.0  # Perpendicular to body
        yaw_rad = math.radians(yaw)

        # Calculate perpendicular direction
        perp_x = math.cos(yaw_rad)
        perp_z = math.sin(yaw_rad)

        wings = []
        for side in [-1, 1]:  # Left and right wings
            wing_x = dragon_pos.x + perp_x * wing_offset * side
            wing_z = dragon_pos.z + perp_z * wing_offset * side
            wing_y = dragon_pos.y + 1.0

            half_w = self.WING_WIDTH / 2
            half_h = self.WING_HEIGHT / 2

            wings.append(
                AABB(
                    Vec3(wing_x - half_w, wing_y - half_h, wing_z - half_w),
                    Vec3(wing_x + half_w, wing_y + half_h, wing_z + half_w),
                )
            )

        return wings

    def get_tail_hitboxes(self, dragon_pos: Vec3, yaw: float) -> list[AABB]:
        """Calculate tail segment hitboxes."""
        yaw_rad = math.radians(yaw)

        # Tail extends opposite to head direction
        tail_dir_x = math.sin(yaw_rad)
        tail_dir_z = -math.cos(yaw_rad)

        segments = []
        for i in range(self.TAIL_SEGMENTS):
            offset = 4.0 + i * 3.0  # Each segment further back
            seg_x = dragon_pos.x + tail_dir_x * offset
            seg_z = dragon_pos.z + tail_dir_z * offset
            seg_y = dragon_pos.y - 0.5 * i  # Tail droops

            half_w = self.TAIL_SEGMENT_WIDTH / 2
            half_h = self.TAIL_SEGMENT_HEIGHT / 2

            segments.append(
                AABB(
                    Vec3(seg_x - half_w, seg_y - half_h, seg_z - half_w),
                    Vec3(seg_x + half_w, seg_y + half_h, seg_z + half_w),
                )
            )

        return segments


@dataclass
class VerificationResult:
    """Result of a verification check."""

    name: str
    passed: bool
    expected: Any
    actual: Any
    tolerance: float = 0.001
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "tolerance": self.tolerance,
            "message": self.message,
        }


@dataclass
class VerificationReport:
    """Complete verification report."""

    results: list[VerificationResult] = field(default_factory=list)

    def add(self, result: VerificationResult) -> None:
        self.results.append(result)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def summary(self) -> str:
        lines = [
            f"Verification Report: {self.passed}/{self.total} passed ({self.success_rate:.1%})",
            "-" * 60,
        ]

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"[{status}] {r.name}")
            if not r.passed:
                lines.append(f"       Expected: {r.expected}")
                lines.append(f"       Actual:   {r.actual}")
                if r.message:
                    lines.append(f"       Note:     {r.message}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "success_rate": self.success_rate,
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# Verifier Classes
# =============================================================================


class DragonMovementVerifier:
    """Verify dragon movement patterns match Java Edition."""

    def __init__(self) -> None:
        self.constants = DragonConstants()

    def verify_circling_pattern(
        self,
        positions: list[Vec3],
        center: Vec3,
        time_step: float,
    ) -> VerificationResult:
        """Verify dragon circling maintains correct radius and speed."""
        if len(positions) < 2:
            return VerificationResult(
                name="Circling Pattern",
                passed=False,
                expected="At least 2 positions",
                actual=len(positions),
                message="Insufficient data points",
            )

        # Check radius consistency
        radii = [p.horizontal_distance_to(center) for p in positions]
        avg_radius = sum(radii) / len(radii)
        radius_variance = sum((r - avg_radius) ** 2 for r in radii) / len(radii)

        # Check height bounds
        heights = [p.y for p in positions]
        min_height = min(heights)
        max_height = max(heights)

        # Check angular velocity (speed)
        speeds = []
        for i in range(1, len(positions)):
            dist = positions[i].horizontal_distance_to(positions[i - 1])
            speeds.append(dist / time_step)

        avg_speed = sum(speeds) / len(speeds) if speeds else 0

        passed = (
            abs(avg_radius - self.constants.CIRCLING_RADIUS) < 20.0  # 20 block tolerance
            and min_height >= self.constants.CIRCLING_HEIGHT_MIN - 5
            and max_height <= self.constants.CIRCLING_HEIGHT_MAX + 5
            and abs(avg_speed - self.constants.CIRCLING_SPEED) < 5.0
        )

        return VerificationResult(
            name="Circling Pattern",
            passed=passed,
            expected={
                "radius": self.constants.CIRCLING_RADIUS,
                "height_range": [
                    self.constants.CIRCLING_HEIGHT_MIN,
                    self.constants.CIRCLING_HEIGHT_MAX,
                ],
                "speed": self.constants.CIRCLING_SPEED,
            },
            actual={
                "radius": avg_radius,
                "height_range": [min_height, max_height],
                "speed": avg_speed,
                "radius_variance": radius_variance,
            },
        )

    def verify_strafe_speed(
        self,
        start_pos: Vec3,
        end_pos: Vec3,
        duration: float,
    ) -> VerificationResult:
        """Verify strafe attack speed."""
        distance = start_pos.distance_to(end_pos)
        speed = distance / duration if duration > 0 else 0

        passed = abs(speed - self.constants.STRAFE_SPEED) < 5.0

        return VerificationResult(
            name="Strafe Speed",
            passed=passed,
            expected=self.constants.STRAFE_SPEED,
            actual=speed,
            tolerance=5.0,
        )

    def verify_phase_transitions(
        self,
        phase_durations: dict[DragonPhase, int],
    ) -> VerificationResult:
        """Verify phase duration minimums are respected."""
        violations = []

        for phase, duration in phase_durations.items():
            if duration < self.constants.PHASE_DURATION_MIN:
                violations.append(
                    f"{phase.name}: {duration} ticks (min: {self.constants.PHASE_DURATION_MIN})"
                )

        passed = len(violations) == 0

        return VerificationResult(
            name="Phase Duration Minimums",
            passed=passed,
            expected=f">= {self.constants.PHASE_DURATION_MIN} ticks per phase",
            actual=violations if violations else "All phases meet minimum",
            message="; ".join(violations) if violations else "",
        )


class CrystalVerifier:
    """Verify end crystal mechanics."""

    def __init__(self) -> None:
        self.crystal_constants = CrystalConstants()
        self.dragon_constants = DragonConstants()

    def verify_heal_rate(
        self,
        dragon_health_before: float,
        dragon_health_after: float,
        ticks_elapsed: int,
        crystal_connected: bool,
    ) -> VerificationResult:
        """Verify crystal healing rate."""
        if not crystal_connected:
            expected_heal = 0.0
        else:
            expected_heal = self.crystal_constants.HEAL_RATE_PER_TICK * ticks_elapsed

        actual_heal = dragon_health_after - dragon_health_before

        # Clamp expected to max health
        max_possible_heal = self.dragon_constants.HEALTH - dragon_health_before
        expected_heal = min(expected_heal, max_possible_heal)

        passed = abs(actual_heal - expected_heal) < 1.0  # 1 HP tolerance

        return VerificationResult(
            name="Crystal Heal Rate",
            passed=passed,
            expected=expected_heal,
            actual=actual_heal,
            tolerance=1.0,
            message=f"Over {ticks_elapsed} ticks, crystal {'connected' if crystal_connected else 'disconnected'}",
        )

    def verify_heal_range(
        self,
        dragon_pos: Vec3,
        crystal_pos: Vec3,
        is_healing: bool,
    ) -> VerificationResult:
        """Verify crystal heal range cutoff."""
        distance = dragon_pos.distance_to(crystal_pos)
        should_heal = distance <= self.crystal_constants.HEAL_RANGE

        passed = is_healing == should_heal

        return VerificationResult(
            name="Crystal Heal Range",
            passed=passed,
            expected=f"Healing: {should_heal} (distance: {distance:.1f}, max: {self.crystal_constants.HEAL_RANGE})",
            actual=f"Healing: {is_healing}",
            message=f"Distance: {distance:.2f} blocks",
        )

    def verify_explosion_damage(
        self,
        explosion_pos: Vec3,
        entity_pos: Vec3,
        entity_armor: float,
        actual_damage: float,
    ) -> VerificationResult:
        """Verify explosion damage calculation.

        Minecraft explosion damage formula:
        damage = (1 - distance/radius) * 2 * power * 7 + 1
        Then armor reduction applies.
        """
        distance = explosion_pos.distance_to(entity_pos)

        if distance > self.crystal_constants.EXPLOSION_RADIUS:
            expected_damage = 0.0
        else:
            # Base explosion damage
            exposure = 1.0 - (distance / self.crystal_constants.EXPLOSION_RADIUS)
            base_damage = exposure * 2 * self.crystal_constants.EXPLOSION_POWER * 7 + 1

            # Armor reduction (simplified - actual MC uses enchants, etc.)
            armor_reduction = min(20.0, entity_armor) / 25.0
            expected_damage = base_damage * (1 - armor_reduction)

        passed = abs(actual_damage - expected_damage) < 5.0  # 5 HP tolerance

        return VerificationResult(
            name="Crystal Explosion Damage",
            passed=passed,
            expected=expected_damage,
            actual=actual_damage,
            tolerance=5.0,
            message=f"Distance: {distance:.2f}, Armor: {entity_armor}",
        )

    def verify_dragon_damage_on_crystal_destroy(
        self,
        dragon_health_before: float,
        dragon_health_after: float,
        was_connected: bool,
    ) -> VerificationResult:
        """Verify dragon takes damage when connected crystal destroyed."""
        if was_connected:
            expected_damage = self.dragon_constants.CRYSTAL_DAMAGE_TO_DRAGON
        else:
            expected_damage = 0.0

        actual_damage = dragon_health_before - dragon_health_after
        passed = abs(actual_damage - expected_damage) < 1.0

        return VerificationResult(
            name="Dragon Damage on Crystal Destroy",
            passed=passed,
            expected=expected_damage,
            actual=actual_damage,
            message=f"Crystal was {'connected' if was_connected else 'not connected'}",
        )


class DamageVerifier:
    """Verify damage calculations and hitbox interactions."""

    def __init__(self) -> None:
        self.constants = DragonConstants()
        self.hitboxes = DragonHitboxes()

    def verify_head_damage_multiplier(
        self,
        base_damage: float,
        actual_damage: float,
    ) -> VerificationResult:
        """Verify head takes 4x damage."""
        expected = base_damage * self.constants.HEAD_DAMAGE_MULTIPLIER
        passed = abs(actual_damage - expected) < 0.5

        return VerificationResult(
            name="Head Damage Multiplier",
            passed=passed,
            expected=expected,
            actual=actual_damage,
            message=f"Base: {base_damage}, Multiplier: {self.constants.HEAD_DAMAGE_MULTIPLIER}x",
        )

    def verify_body_damage_multiplier(
        self,
        base_damage: float,
        actual_damage: float,
    ) -> VerificationResult:
        """Verify body takes 0.25x damage."""
        expected = base_damage * self.constants.BODY_DAMAGE_MULTIPLIER
        passed = abs(actual_damage - expected) < 0.5

        return VerificationResult(
            name="Body Damage Multiplier",
            passed=passed,
            expected=expected,
            actual=actual_damage,
            message=f"Base: {base_damage}, Multiplier: {self.constants.BODY_DAMAGE_MULTIPLIER}x",
        )

    def verify_wing_immunity(
        self,
        damage_dealt: float,
    ) -> VerificationResult:
        """Verify wings are immune to damage."""
        passed = damage_dealt == 0.0

        return VerificationResult(
            name="Wing Damage Immunity",
            passed=passed,
            expected=0.0,
            actual=damage_dealt,
        )

    def verify_hitbox_positions(
        self,
        dragon_pos: Vec3,
        dragon_yaw: float,
        hit_pos: Vec3,
        reported_part: str,
    ) -> VerificationResult:
        """Verify hit detection identifies correct body part."""
        head_box = self.hitboxes.get_head_hitbox(dragon_pos, dragon_yaw)
        body_box = self.hitboxes.get_body_hitbox(dragon_pos)
        wing_boxes = self.hitboxes.get_wing_hitboxes(dragon_pos, dragon_yaw)
        tail_boxes = self.hitboxes.get_tail_hitboxes(dragon_pos, dragon_yaw)

        actual_part = "none"

        if head_box.contains(hit_pos):
            actual_part = "head"
        elif body_box.contains(hit_pos):
            actual_part = "body"
        elif any(w.contains(hit_pos) for w in wing_boxes):
            actual_part = "wing"
        elif any(t.contains(hit_pos) for t in tail_boxes):
            actual_part = "tail"

        passed = actual_part == reported_part

        return VerificationResult(
            name="Hitbox Part Detection",
            passed=passed,
            expected=reported_part,
            actual=actual_part,
            message=f"Hit at {hit_pos.to_dict()}, dragon at {dragon_pos.to_dict()}, yaw={dragon_yaw}",
        )


class DeathSequenceVerifier:
    """Verify dragon death sequence timing."""

    def __init__(self) -> None:
        self.dragon_constants = DragonConstants()
        self.portal_constants = PortalConstants()

    def verify_death_animation_duration(
        self,
        animation_ticks: int,
    ) -> VerificationResult:
        """Verify death animation lasts exactly 200 ticks."""
        expected = self.dragon_constants.DEATH_ANIMATION_TICKS
        passed = animation_ticks == expected

        return VerificationResult(
            name="Death Animation Duration",
            passed=passed,
            expected=f"{expected} ticks (10 seconds)",
            actual=f"{animation_ticks} ticks ({animation_ticks / 20:.1f} seconds)",
        )

    def verify_xp_drop(
        self,
        xp_dropped: int,
        is_first_kill: bool,
    ) -> VerificationResult:
        """Verify XP drop amount."""
        expected = (
            self.dragon_constants.XP_DROP_FIRST_KILL
            if is_first_kill
            else self.dragon_constants.XP_DROP_SUBSEQUENT
        )

        passed = xp_dropped == expected

        return VerificationResult(
            name="XP Drop Amount",
            passed=passed,
            expected=expected,
            actual=xp_dropped,
            message=f"{'First' if is_first_kill else 'Subsequent'} kill",
        )

    def verify_portal_spawn_timing(
        self,
        death_tick: int,
        portal_spawn_tick: int,
    ) -> VerificationResult:
        """Verify portal spawns after death animation."""
        expected_spawn = (
            death_tick
            + self.dragon_constants.DEATH_ANIMATION_TICKS
            + self.portal_constants.SPAWN_DELAY_TICKS
        )

        passed = portal_spawn_tick >= expected_spawn

        return VerificationResult(
            name="Portal Spawn Timing",
            passed=passed,
            expected=f">= tick {expected_spawn}",
            actual=f"tick {portal_spawn_tick}",
            message=f"Death at tick {death_tick}, animation {self.dragon_constants.DEATH_ANIMATION_TICKS} ticks",
        )

    def verify_portal_position(
        self,
        portal_center: Vec3,
    ) -> VerificationResult:
        """Verify portal spawns at world center."""
        expected = Vec3(
            self.portal_constants.CENTER_X,
            self.portal_constants.CENTER_Y,
            self.portal_constants.CENTER_Z,
        )

        distance = portal_center.horizontal_distance_to(expected)
        passed = distance < 1.0 and abs(portal_center.y - expected.y) < 1.0

        return VerificationResult(
            name="Portal Position",
            passed=passed,
            expected=expected.to_dict(),
            actual=portal_center.to_dict(),
            message=f"Distance from expected: {distance:.2f} blocks",
        )

    def verify_egg_spawn(
        self,
        egg_spawned: bool,
        is_first_kill: bool,
    ) -> VerificationResult:
        """Verify egg only spawns on first kill."""
        expected = is_first_kill and self.portal_constants.EGG_SPAWN_FIRST_ONLY
        passed = egg_spawned == expected

        return VerificationResult(
            name="Dragon Egg Spawn",
            passed=passed,
            expected=expected,
            actual=egg_spawned,
            message=f"{'First' if is_first_kill else 'Subsequent'} kill",
        )


class XPOrbVerifier:
    """Verify XP orb spawning mechanics."""

    def __init__(self) -> None:
        self.dragon_constants = DragonConstants()

    def verify_orb_total(
        self,
        orb_values: list[int],
        is_first_kill: bool,
    ) -> VerificationResult:
        """Verify total XP from all orbs matches expected."""
        total = sum(orb_values)
        expected = (
            self.dragon_constants.XP_DROP_FIRST_KILL
            if is_first_kill
            else self.dragon_constants.XP_DROP_SUBSEQUENT
        )

        passed = total == expected

        return VerificationResult(
            name="Total XP from Orbs",
            passed=passed,
            expected=expected,
            actual=total,
            message=f"{len(orb_values)} orbs spawned",
        )

    def verify_orb_spawn_pattern(
        self,
        orb_positions: list[Vec3],
        dragon_death_pos: Vec3,
        spawn_radius: float = 8.0,
    ) -> VerificationResult:
        """Verify orbs spawn near dragon death position."""
        distances = [p.distance_to(dragon_death_pos) for p in orb_positions]
        max_distance = max(distances) if distances else 0
        avg_distance = sum(distances) / len(distances) if distances else 0

        passed = max_distance <= spawn_radius

        return VerificationResult(
            name="XP Orb Spawn Pattern",
            passed=passed,
            expected=f"All orbs within {spawn_radius} blocks of death position",
            actual=f"Max distance: {max_distance:.2f}, Avg: {avg_distance:.2f}",
            message=f"Death position: {dragon_death_pos.to_dict()}",
        )


# =============================================================================
# Main Verification Runner
# =============================================================================


class DragonFightVerifier:
    """Main verification system for dragon fight subsystems."""

    def __init__(self) -> None:
        self.movement = DragonMovementVerifier()
        self.crystal = CrystalVerifier()
        self.damage = DamageVerifier()
        self.death = DeathSequenceVerifier()
        self.xp = XPOrbVerifier()

    def run_scenario(self, scenario: dict[str, Any]) -> VerificationReport:
        """Run a verification scenario from JSON."""
        report = VerificationReport()

        scenario_type = scenario.get("type", "")

        if scenario_type == "movement":
            self._verify_movement(scenario, report)
        elif scenario_type == "crystal":
            self._verify_crystal(scenario, report)
        elif scenario_type == "damage":
            self._verify_damage(scenario, report)
        elif scenario_type == "death":
            self._verify_death(scenario, report)
        elif scenario_type == "xp":
            self._verify_xp(scenario, report)
        elif scenario_type == "full":
            self._verify_movement(scenario.get("movement", {}), report)
            self._verify_crystal(scenario.get("crystal", {}), report)
            self._verify_damage(scenario.get("damage", {}), report)
            self._verify_death(scenario.get("death", {}), report)
            self._verify_xp(scenario.get("xp", {}), report)

        return report

    def _verify_movement(self, data: dict[str, Any], report: VerificationReport) -> None:
        if "circling" in data:
            c = data["circling"]
            positions = [Vec3.from_dict(p) for p in c.get("positions", [])]
            center = Vec3.from_dict(c.get("center", {"x": 0, "y": 0, "z": 0}))
            time_step = c.get("time_step", 1.0)
            report.add(self.movement.verify_circling_pattern(positions, center, time_step))

        if "strafe" in data:
            s = data["strafe"]
            start = Vec3.from_dict(s["start"])
            end = Vec3.from_dict(s["end"])
            duration = s.get("duration", 1.0)
            report.add(self.movement.verify_strafe_speed(start, end, duration))

        if "phases" in data:
            phases = {DragonPhase[k.upper()]: v for k, v in data["phases"].items()}
            report.add(self.movement.verify_phase_transitions(phases))

    def _verify_crystal(self, data: dict[str, Any], report: VerificationReport) -> None:
        if "heal" in data:
            h = data["heal"]
            report.add(
                self.crystal.verify_heal_rate(
                    h["dragon_health_before"],
                    h["dragon_health_after"],
                    h["ticks_elapsed"],
                    h["crystal_connected"],
                )
            )

        if "heal_range" in data:
            r = data["heal_range"]
            report.add(
                self.crystal.verify_heal_range(
                    Vec3.from_dict(r["dragon_pos"]),
                    Vec3.from_dict(r["crystal_pos"]),
                    r["is_healing"],
                )
            )

        if "explosion" in data:
            e = data["explosion"]
            report.add(
                self.crystal.verify_explosion_damage(
                    Vec3.from_dict(e["explosion_pos"]),
                    Vec3.from_dict(e["entity_pos"]),
                    e.get("entity_armor", 0),
                    e["actual_damage"],
                )
            )

        if "crystal_destroy" in data:
            d = data["crystal_destroy"]
            report.add(
                self.crystal.verify_dragon_damage_on_crystal_destroy(
                    d["dragon_health_before"],
                    d["dragon_health_after"],
                    d["was_connected"],
                )
            )

    def _verify_damage(self, data: dict[str, Any], report: VerificationReport) -> None:
        if "head" in data:
            h = data["head"]
            report.add(
                self.damage.verify_head_damage_multiplier(
                    h["base_damage"],
                    h["actual_damage"],
                )
            )

        if "body" in data:
            b = data["body"]
            report.add(
                self.damage.verify_body_damage_multiplier(
                    b["base_damage"],
                    b["actual_damage"],
                )
            )

        if "wing" in data:
            report.add(self.damage.verify_wing_immunity(data["wing"]["damage_dealt"]))

        if "hitbox" in data:
            hb = data["hitbox"]
            report.add(
                self.damage.verify_hitbox_positions(
                    Vec3.from_dict(hb["dragon_pos"]),
                    hb["dragon_yaw"],
                    Vec3.from_dict(hb["hit_pos"]),
                    hb["reported_part"],
                )
            )

    def _verify_death(self, data: dict[str, Any], report: VerificationReport) -> None:
        if "animation_ticks" in data:
            report.add(self.death.verify_death_animation_duration(data["animation_ticks"]))

        if "xp" in data:
            x = data["xp"]
            report.add(self.death.verify_xp_drop(x["dropped"], x["is_first_kill"]))

        if "portal_timing" in data:
            p = data["portal_timing"]
            report.add(
                self.death.verify_portal_spawn_timing(
                    p["death_tick"],
                    p["portal_spawn_tick"],
                )
            )

        if "portal_position" in data:
            report.add(
                self.death.verify_portal_position(
                    Vec3.from_dict(data["portal_position"]),
                )
            )

        if "egg" in data:
            e = data["egg"]
            report.add(self.death.verify_egg_spawn(e["spawned"], e["is_first_kill"]))

    def _verify_xp(self, data: dict[str, Any], report: VerificationReport) -> None:
        if "orbs" in data:
            o = data["orbs"]
            report.add(self.xp.verify_orb_total(o["values"], o["is_first_kill"]))

        if "orb_positions" in data:
            op = data["orb_positions"]
            positions = [Vec3.from_dict(p) for p in op["positions"]]
            death_pos = Vec3.from_dict(op["dragon_death_pos"])
            report.add(self.xp.verify_orb_spawn_pattern(positions, death_pos))

    def run_all_from_file(self, json_path: Path) -> dict[str, VerificationReport]:
        """Load and run all scenarios from a JSON file."""
        with open(json_path) as f:
            data = json.load(f)

        reports = {}
        for scenario in data.get("scenarios", []):
            name = scenario.get("name", "unnamed")
            reports[name] = self.run_scenario(scenario)

        return reports

    def generate_summary(self, reports: dict[str, VerificationReport]) -> str:
        """Generate a combined summary of all reports."""
        lines = ["=" * 60, "DRAGON FIGHT VERIFICATION SUMMARY", "=" * 60, ""]

        total_passed = sum(r.passed for r in reports.values())
        total_tests = sum(r.total for r in reports.values())

        lines.append(f"Overall: {total_passed}/{total_tests} tests passed")
        lines.append("")

        for name, report in reports.items():
            status = "PASS" if report.failed == 0 else "FAIL"
            lines.append(f"[{status}] {name}: {report.passed}/{report.total}")

        lines.append("")
        lines.append("=" * 60)
        lines.append("DETAILED RESULTS")
        lines.append("=" * 60)

        for name, report in reports.items():
            lines.append(f"\n--- {name} ---")
            lines.append(report.summary())

        return "\n".join(lines)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Run verification from command line."""
    import sys

    verifier = DragonFightVerifier()

    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
        if json_path.exists():
            reports = verifier.run_all_from_file(json_path)
            print(verifier.generate_summary(reports))
        else:
            print(f"File not found: {json_path}")
            sys.exit(1)
    else:
        # Run built-in test cases
        test_dir = Path(__file__).parent / "test_cases"
        default_scenarios = test_dir / "dragon_scenarios.json"

        if default_scenarios.exists():
            reports = verifier.run_all_from_file(default_scenarios)
            print(verifier.generate_summary(reports))
        else:
            print("No test file provided and default scenarios not found.")
            print(f"Usage: python {sys.argv[0]} <scenarios.json>")
            sys.exit(1)


if __name__ == "__main__":
    main()
