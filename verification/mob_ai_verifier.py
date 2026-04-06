"""
Mob AI Verification System

Tests and verifies Minecraft-style mob AI subsystems including:
- Enderman aggro mechanics (eye contact detection)
- Enderman teleportation patterns
- Blaze fireball timing and trajectories
- Zombie/skeleton targeting and pathfinding
- Pigman group aggro propagation
- Spawning rates and location validation
- Drop rates with looting enchantment effects
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobType(Enum):
    """Enumeration of mob types for AI testing."""

    ENDERMAN = auto()
    BLAZE = auto()
    ZOMBIE = auto()
    SKELETON = auto()
    PIGMAN = auto()
    CREEPER = auto()
    SPIDER = auto()
    WITHER_SKELETON = auto()


class AggroState(Enum):
    """Mob aggression state."""

    PASSIVE = auto()
    NEUTRAL = auto()
    HOSTILE = auto()
    FLEEING = auto()


@dataclass
class Vec3:
    """3D vector for positions and directions."""

    x: float
    y: float
    z: float

    def distance_to(self, other: Vec3) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def normalized(self) -> Vec3:
        mag = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if mag == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / mag, self.y / mag, self.z / mag)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)


@dataclass
class Entity:
    """Base entity with position and look direction."""

    position: Vec3
    look_direction: Vec3
    entity_id: str = ""


@dataclass
class Player(Entity):
    """Player entity with inventory for looting tests."""

    looting_level: int = 0
    is_wearing_pumpkin: bool = False


@dataclass
class Mob(Entity):
    """Mob entity with AI state."""

    mob_type: MobType = MobType.ZOMBIE
    aggro_state: AggroState = AggroState.NEUTRAL
    health: float = 20.0
    target: Entity | None = None
    last_teleport_tick: int = 0
    group_id: str = ""


@dataclass
class Projectile:
    """Projectile data for fireball/arrow tests."""

    origin: Vec3
    velocity: Vec3
    damage: float
    tick_fired: int = 0


@dataclass
class VerificationResult:
    """Result of a verification test."""

    test_name: str
    passed: bool
    expected: Any
    actual: Any
    details: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.test_name}: {self.details}"


class MobAIVerifier:
    """
    Comprehensive mob AI verification system.

    Tests various mob behavior subsystems for correctness.
    """

    # Enderman constants
    ENDERMAN_AGGRO_RANGE = 64.0
    ENDERMAN_AGGRO_ANGLE = 5.0  # degrees
    ENDERMAN_TELEPORT_COOLDOWN = 100  # ticks
    ENDERMAN_TELEPORT_MIN_DIST = 8.0
    ENDERMAN_TELEPORT_MAX_DIST = 32.0

    # Blaze constants
    BLAZE_ATTACK_RANGE = 48.0
    BLAZE_FIREBALL_SPEED = 0.4
    BLAZE_ATTACK_COOLDOWN = 60  # ticks (3 seconds)
    BLAZE_BURST_COUNT = 3
    BLAZE_BURST_INTERVAL = 10  # ticks between fireballs in burst

    # Zombie/Skeleton constants
    ZOMBIE_DETECTION_RANGE = 40.0
    SKELETON_DETECTION_RANGE = 16.0
    SKELETON_SHOOT_RANGE = 15.0
    PATHFIND_NODE_SPACING = 1.0

    # Pigman constants
    PIGMAN_AGGRO_RANGE = 32.0
    PIGMAN_GROUP_AGGRO_RANGE = 40.0
    PIGMAN_AGGRO_DURATION = 400  # ticks (20 seconds)

    # Spawning constants
    SPAWN_LIGHT_THRESHOLD = 7
    SPAWN_MIN_DISTANCE_FROM_PLAYER = 24.0
    SPAWN_MAX_DISTANCE_FROM_PLAYER = 128.0

    # Drop rate constants
    BASE_DROP_CHANCE = 0.085  # 8.5% base chance for rare drops
    LOOTING_BONUS_PER_LEVEL = 0.01  # +1% per looting level

    def __init__(self) -> None:
        self.results: list[VerificationResult] = []
        self.current_tick = 0

    def reset(self) -> None:
        """Reset verifier state."""
        self.results.clear()
        self.current_tick = 0

    def add_result(self, result: VerificationResult) -> None:
        """Add a verification result."""
        self.results.append(result)
        logger.info(str(result))

    # ========== ENDERMAN TESTS ==========

    def verify_enderman_eye_contact_aggro(
        self,
        enderman: Mob,
        player: Player,
    ) -> VerificationResult:
        """
        Verify Enderman aggro triggers on eye contact.

        Rules:
        - Player must be looking at Enderman's head hitbox
        - Enderman must be looking at player (angle < 5 degrees)
        - Distance must be <= 64 blocks
        - Pumpkin helmet negates aggro
        """
        test_name = "enderman_eye_contact_aggro"

        distance = enderman.position.distance_to(player.position)
        if distance > self.ENDERMAN_AGGRO_RANGE:
            return VerificationResult(
                test_name=test_name,
                passed=True,
                expected="no aggro (out of range)",
                actual="no aggro",
                details=f"Distance {distance:.1f} > {self.ENDERMAN_AGGRO_RANGE}",
            )

        # Check if player is wearing pumpkin (negates aggro)
        if player.is_wearing_pumpkin:
            expected_aggro = False
            actual_aggro = enderman.aggro_state == AggroState.HOSTILE
            return VerificationResult(
                test_name=test_name,
                passed=not actual_aggro,
                expected="no aggro (pumpkin)",
                actual="aggro" if actual_aggro else "no aggro",
                details="Pumpkin helmet should prevent aggro",
            )

        # Calculate if player is looking at enderman
        to_enderman = (enderman.position - player.position).normalized()
        player_look_angle = math.degrees(
            math.acos(max(-1, min(1, player.look_direction.dot(to_enderman))))
        )

        # Calculate if enderman is looking at player
        to_player = (player.position - enderman.position).normalized()
        enderman_look_angle = math.degrees(
            math.acos(max(-1, min(1, enderman.look_direction.dot(to_player))))
        )

        eye_contact = (
            player_look_angle < self.ENDERMAN_AGGRO_ANGLE
            and enderman_look_angle < self.ENDERMAN_AGGRO_ANGLE
        )

        expected_aggro = eye_contact
        actual_aggro = enderman.aggro_state == AggroState.HOSTILE

        return VerificationResult(
            test_name=test_name,
            passed=expected_aggro == actual_aggro,
            expected="aggro" if expected_aggro else "no aggro",
            actual="aggro" if actual_aggro else "no aggro",
            details=f"Player angle: {player_look_angle:.1f}, Enderman angle: {enderman_look_angle:.1f}",
        )

    def verify_enderman_teleport_pattern(
        self,
        enderman: Mob,
        teleport_target: Vec3,
        current_tick: int,
    ) -> VerificationResult:
        """
        Verify Enderman teleportation follows rules.

        Rules:
        - Cannot teleport more often than every 100 ticks
        - Teleport distance must be 8-32 blocks
        - Cannot teleport into water or lava
        - Cannot teleport onto transparent blocks
        """
        test_name = "enderman_teleport_pattern"

        # Check cooldown
        ticks_since_last = current_tick - enderman.last_teleport_tick
        if ticks_since_last < self.ENDERMAN_TELEPORT_COOLDOWN:
            return VerificationResult(
                test_name=test_name,
                passed=False,
                expected=f">= {self.ENDERMAN_TELEPORT_COOLDOWN} ticks since last",
                actual=f"{ticks_since_last} ticks",
                details="Teleport on cooldown",
            )

        # Check distance
        distance = enderman.position.distance_to(teleport_target)
        valid_distance = (
            self.ENDERMAN_TELEPORT_MIN_DIST <= distance <= self.ENDERMAN_TELEPORT_MAX_DIST
        )

        if not valid_distance:
            return VerificationResult(
                test_name=test_name,
                passed=False,
                expected=f"{self.ENDERMAN_TELEPORT_MIN_DIST}-{self.ENDERMAN_TELEPORT_MAX_DIST} blocks",
                actual=f"{distance:.1f} blocks",
                details="Teleport distance out of range",
            )

        return VerificationResult(
            test_name=test_name,
            passed=True,
            expected="valid teleport",
            actual="valid teleport",
            details=f"Distance: {distance:.1f}, Cooldown: {ticks_since_last} ticks",
        )

    # ========== BLAZE TESTS ==========

    def verify_blaze_fireball_timing(
        self,
        blaze: Mob,
        fireballs: list[Projectile],
        current_tick: int,
    ) -> VerificationResult:
        """
        Verify Blaze fireball attack timing.

        Rules:
        - Fires in bursts of 3 fireballs
        - 0.5 second (10 tick) interval between fireballs in burst
        - 3 second (60 tick) cooldown between bursts
        """
        test_name = "blaze_fireball_timing"

        if len(fireballs) < 2:
            return VerificationResult(
                test_name=test_name,
                passed=True,
                expected="timing check",
                actual="insufficient data",
                details="Need at least 2 fireballs to verify timing",
            )

        # Check intervals between consecutive fireballs
        intervals = []
        for i in range(1, len(fireballs)):
            interval = fireballs[i].tick_fired - fireballs[i - 1].tick_fired
            intervals.append(interval)

        # Verify burst intervals (should be ~10 ticks within burst)
        burst_violations = []
        burst_count = 0
        for i, interval in enumerate(intervals):
            if interval < self.BLAZE_ATTACK_COOLDOWN:
                burst_count += 1
                if abs(interval - self.BLAZE_BURST_INTERVAL) > 2:  # 2 tick tolerance
                    burst_violations.append((i, interval))
            else:
                # This is a between-burst interval
                if interval < self.BLAZE_ATTACK_COOLDOWN:
                    burst_violations.append((i, interval))

        passed = len(burst_violations) == 0
        return VerificationResult(
            test_name=test_name,
            passed=passed,
            expected=f"Burst interval: {self.BLAZE_BURST_INTERVAL}, Cooldown: {self.BLAZE_ATTACK_COOLDOWN}",
            actual=f"Intervals: {intervals}",
            details=f"Violations: {burst_violations}" if burst_violations else "Timing correct",
        )

    def verify_blaze_fireball_trajectory(
        self,
        blaze: Mob,
        target: Entity,
        fireball: Projectile,
    ) -> VerificationResult:
        """
        Verify Blaze fireball trajectory accuracy.

        Rules:
        - Fireball should be aimed at target with some inaccuracy
        - Speed should be approximately 0.4 blocks/tick
        - Should have slight gravity effect
        """
        test_name = "blaze_fireball_trajectory"

        # Expected direction (with leading for moving targets)
        expected_direction = (target.position - fireball.origin).normalized()
        actual_direction = fireball.velocity.normalized()

        # Calculate angle between expected and actual
        dot_product = expected_direction.dot(actual_direction)
        angle = math.degrees(math.acos(max(-1, min(1, dot_product))))

        # Blaze has ~10 degree inaccuracy
        max_inaccuracy = 15.0
        passed = angle <= max_inaccuracy

        # Check speed
        speed = math.sqrt(fireball.velocity.x**2 + fireball.velocity.y**2 + fireball.velocity.z**2)
        speed_correct = abs(speed - self.BLAZE_FIREBALL_SPEED) < 0.1

        return VerificationResult(
            test_name=test_name,
            passed=passed and speed_correct,
            expected=f"Angle < {max_inaccuracy}, Speed ~{self.BLAZE_FIREBALL_SPEED}",
            actual=f"Angle: {angle:.1f}, Speed: {speed:.2f}",
            details="Trajectory within acceptable parameters"
            if passed
            else "Trajectory deviation too high",
        )

    # ========== ZOMBIE/SKELETON TESTS ==========

    def verify_zombie_targeting(
        self,
        zombie: Mob,
        players: list[Player],
        villagers: list[Entity],
    ) -> VerificationResult:
        """
        Verify Zombie target selection.

        Rules:
        - Targets nearest player within 40 blocks
        - Will also target villagers
        - Prefers players over villagers at equal distance
        """
        test_name = "zombie_targeting"

        # Find valid targets
        valid_targets: list[tuple[float, Entity, str]] = []

        for player in players:
            dist = zombie.position.distance_to(player.position)
            if dist <= self.ZOMBIE_DETECTION_RANGE:
                valid_targets.append((dist, player, "player"))

        for villager in villagers:
            dist = zombie.position.distance_to(villager.position)
            if dist <= self.ZOMBIE_DETECTION_RANGE:
                valid_targets.append((dist, villager, "villager"))

        if not valid_targets:
            expected_target = None
        else:
            # Sort by distance, players first at equal distance
            valid_targets.sort(key=lambda x: (x[0], 0 if x[2] == "player" else 1))
            expected_target = valid_targets[0][1]

        actual_target = zombie.target
        passed = expected_target == actual_target

        return VerificationResult(
            test_name=test_name,
            passed=passed,
            expected=f"Target: {expected_target.entity_id if expected_target else 'None'}",
            actual=f"Target: {actual_target.entity_id if actual_target else 'None'}",
            details=f"{len(valid_targets)} valid targets in range",
        )

    def verify_skeleton_shoot_behavior(
        self,
        skeleton: Mob,
        target: Entity,
        arrow_fired: bool,
        current_tick: int,
        last_shot_tick: int,
    ) -> VerificationResult:
        """
        Verify Skeleton shooting behavior.

        Rules:
        - Only shoots when target is within 15 blocks
        - Has line of sight requirement
        - Shoots every ~2 seconds (40 ticks) on Normal difficulty
        """
        test_name = "skeleton_shoot_behavior"

        distance = skeleton.position.distance_to(target.position)
        in_range = distance <= self.SKELETON_SHOOT_RANGE
        cooldown_ready = (current_tick - last_shot_tick) >= 40

        expected_shoot = in_range and cooldown_ready

        return VerificationResult(
            test_name=test_name,
            passed=expected_shoot == arrow_fired,
            expected="shoot" if expected_shoot else "no shoot",
            actual="shoot" if arrow_fired else "no shoot",
            details=f"Distance: {distance:.1f}, Cooldown: {current_tick - last_shot_tick} ticks",
        )

    def verify_pathfinding(
        self,
        mob: Mob,
        target: Vec3,
        path: list[Vec3],
        obstacles: list[Vec3],
    ) -> VerificationResult:
        """
        Verify mob pathfinding.

        Rules:
        - Path should avoid obstacles
        - Path nodes should be at most 1 block apart
        - Path should be reasonably efficient (no backtracking)
        """
        test_name = "pathfinding"

        if not path:
            return VerificationResult(
                test_name=test_name,
                passed=False,
                expected="valid path",
                actual="no path",
                details="Pathfinder failed to find route",
            )

        # Check path connectivity
        for i in range(1, len(path)):
            dist = path[i].distance_to(path[i - 1])
            if dist > self.PATHFIND_NODE_SPACING * 1.5:  # Allow some tolerance
                return VerificationResult(
                    test_name=test_name,
                    passed=False,
                    expected=f"Node spacing <= {self.PATHFIND_NODE_SPACING * 1.5}",
                    actual=f"Gap of {dist:.2f} at node {i}",
                    details="Path has disconnected nodes",
                )

        # Check obstacle avoidance
        for node in path:
            for obstacle in obstacles:
                if node.distance_to(obstacle) < 0.5:
                    return VerificationResult(
                        test_name=test_name,
                        passed=False,
                        expected="path avoids obstacles",
                        actual=f"path intersects obstacle at {obstacle}",
                        details="Path goes through obstacle",
                    )

        # Check path efficiency (should not be more than 2x optimal)
        direct_distance = mob.position.distance_to(target)
        path_length = sum(path[i].distance_to(path[i - 1]) for i in range(1, len(path)))
        efficiency = path_length / max(direct_distance, 0.1)

        passed = efficiency <= 2.0
        return VerificationResult(
            test_name=test_name,
            passed=passed,
            expected="efficiency <= 2.0",
            actual=f"efficiency: {efficiency:.2f}",
            details=f"Path length: {path_length:.1f}, Direct: {direct_distance:.1f}",
        )

    # ========== PIGMAN GROUP AGGRO TESTS ==========

    def verify_pigman_group_aggro(
        self,
        attacked_pigman: Mob,
        nearby_pigmen: list[Mob],
        attacker: Player,
    ) -> VerificationResult:
        """
        Verify Zombie Pigman group aggro propagation.

        Rules:
        - When one pigman is attacked, all within 40 blocks become hostile
        - Aggro lasts 20-40 seconds (400-800 ticks)
        - All aggro'd pigmen target the attacker
        """
        test_name = "pigman_group_aggro"

        should_aggro: list[Mob] = []
        should_not_aggro: list[Mob] = []

        for pigman in nearby_pigmen:
            dist = pigman.position.distance_to(attacked_pigman.position)
            if dist <= self.PIGMAN_GROUP_AGGRO_RANGE:
                should_aggro.append(pigman)
            else:
                should_not_aggro.append(pigman)

        # Verify all in range are hostile and targeting attacker
        aggro_failures = []
        for pigman in should_aggro:
            if pigman.aggro_state != AggroState.HOSTILE:
                aggro_failures.append(f"{pigman.entity_id} not hostile")
            elif pigman.target != attacker:
                aggro_failures.append(f"{pigman.entity_id} wrong target")

        # Verify out of range are not hostile
        false_aggro = []
        for pigman in should_not_aggro:
            if pigman.aggro_state == AggroState.HOSTILE and pigman.target == attacker:
                false_aggro.append(pigman.entity_id)

        passed = len(aggro_failures) == 0 and len(false_aggro) == 0
        return VerificationResult(
            test_name=test_name,
            passed=passed,
            expected=f"{len(should_aggro)} hostile, {len(should_not_aggro)} neutral",
            actual=f"Failures: {aggro_failures}, False aggro: {false_aggro}",
            details=f"Group aggro range: {self.PIGMAN_GROUP_AGGRO_RANGE}",
        )

    # ========== SPAWNING TESTS ==========

    def verify_spawning_conditions(
        self,
        spawn_position: Vec3,
        light_level: int,
        player_positions: list[Vec3],
        block_type: str,
        dimension: str,
    ) -> VerificationResult:
        """
        Verify mob spawning conditions.

        Rules:
        - Light level must be 7 or below for hostile mobs
        - Must be 24-128 blocks from nearest player
        - Must be on solid, non-transparent block
        - Different rules for Nether/End
        """
        test_name = "spawning_conditions"

        # Light level check (doesn't apply in Nether/End for most mobs)
        if dimension == "overworld" and light_level > self.SPAWN_LIGHT_THRESHOLD:
            return VerificationResult(
                test_name=test_name,
                passed=False,
                expected=f"light <= {self.SPAWN_LIGHT_THRESHOLD}",
                actual=f"light = {light_level}",
                details="Too bright for hostile mob spawning",
            )

        # Distance from player check
        nearest_player_dist = (
            min(spawn_position.distance_to(p) for p in player_positions)
            if player_positions
            else float("inf")
        )

        valid_distance = (
            self.SPAWN_MIN_DISTANCE_FROM_PLAYER
            <= nearest_player_dist
            <= self.SPAWN_MAX_DISTANCE_FROM_PLAYER
        )
        if not valid_distance:
            return VerificationResult(
                test_name=test_name,
                passed=False,
                expected=f"{self.SPAWN_MIN_DISTANCE_FROM_PLAYER}-{self.SPAWN_MAX_DISTANCE_FROM_PLAYER} blocks from player",
                actual=f"{nearest_player_dist:.1f} blocks",
                details="Invalid spawn distance from player",
            )

        # Block type check
        valid_blocks = {"stone", "grass_block", "dirt", "netherrack", "soul_sand", "end_stone"}
        if block_type not in valid_blocks:
            return VerificationResult(
                test_name=test_name,
                passed=False,
                expected=f"solid block in {valid_blocks}",
                actual=block_type,
                details="Invalid spawn block type",
            )

        return VerificationResult(
            test_name=test_name,
            passed=True,
            expected="valid spawn conditions",
            actual="all checks passed",
            details=f"Light: {light_level}, Distance: {nearest_player_dist:.1f}, Block: {block_type}",
        )

    def verify_spawn_rates(
        self,
        spawns_per_minute: float,
        mob_type: MobType,
        dimension: str,
    ) -> VerificationResult:
        """
        Verify mob spawn rates are within expected ranges.

        Different mobs have different spawn weights.
        """
        test_name = "spawn_rates"

        # Expected spawn rates (spawns per minute in a chunk)
        expected_rates = {
            MobType.ZOMBIE: (2.0, 8.0),
            MobType.SKELETON: (2.0, 8.0),
            MobType.CREEPER: (1.0, 4.0),
            MobType.SPIDER: (2.0, 8.0),
            MobType.ENDERMAN: (0.5, 2.0),
            MobType.BLAZE: (1.0, 4.0),  # Nether fortress only
            MobType.PIGMAN: (3.0, 12.0),  # Nether
            MobType.WITHER_SKELETON: (0.5, 2.0),  # Nether fortress only
        }

        min_rate, max_rate = expected_rates.get(mob_type, (1.0, 5.0))
        passed = min_rate <= spawns_per_minute <= max_rate

        return VerificationResult(
            test_name=test_name,
            passed=passed,
            expected=f"{min_rate}-{max_rate} spawns/min",
            actual=f"{spawns_per_minute:.2f} spawns/min",
            details=f"{mob_type.name} in {dimension}",
        )

    # ========== DROP RATE TESTS ==========

    def verify_drop_rates(
        self,
        mob_type: MobType,
        looting_level: int,
        observed_drops: dict[str, int],
        total_kills: int,
    ) -> VerificationResult:
        """
        Verify drop rates match expected values with looting enchantment.

        Rules:
        - Base rare drop chance is 8.5%
        - Each looting level adds 1% (up to Looting III = +3%)
        - Common drops are guaranteed with variable quantity
        """
        test_name = "drop_rates"

        # Define expected drops per mob type
        expected_drops: dict[MobType, dict[str, tuple[float, float]]] = {
            MobType.ZOMBIE: {
                "rotten_flesh": (0.8, 1.0),  # 80-100% chance
                "iron_ingot": (0.025, 0.055),  # Rare drop
                "carrot": (0.025, 0.055),
                "potato": (0.025, 0.055),
            },
            MobType.SKELETON: {
                "bone": (0.8, 1.0),
                "arrow": (0.8, 1.0),
            },
            MobType.BLAZE: {
                "blaze_rod": (0.5, 0.7),  # 50-70% with variance
            },
            MobType.ENDERMAN: {
                "ender_pearl": (0.5, 0.7),
            },
            MobType.PIGMAN: {
                "gold_nugget": (0.8, 1.0),
                "gold_ingot": (0.025, 0.055),  # Rare
            },
            MobType.WITHER_SKELETON: {
                "bone": (0.8, 1.0),
                "coal": (0.3, 0.5),
                "wither_skeleton_skull": (0.02, 0.055),  # Very rare
            },
        }

        # Adjust for looting
        looting_bonus = looting_level * self.LOOTING_BONUS_PER_LEVEL

        drops_for_mob = expected_drops.get(mob_type, {})
        failures: list[str] = []

        for item, (min_chance, max_chance) in drops_for_mob.items():
            observed = observed_drops.get(item, 0)
            observed_rate = observed / max(total_kills, 1)

            # Apply looting bonus to rare drops
            if max_chance < 0.1:  # Rare drop
                adjusted_max = min(max_chance + looting_bonus, 0.115)  # Cap at 11.5%
                adjusted_min = min_chance
            else:
                adjusted_min = min_chance
                adjusted_max = max_chance

            # Allow 20% variance for statistical noise
            tolerance = 0.2
            if observed_rate < adjusted_min * (1 - tolerance) or observed_rate > adjusted_max * (
                1 + tolerance
            ):
                failures.append(
                    f"{item}: {observed_rate:.2%} not in [{adjusted_min:.2%}, {adjusted_max:.2%}]"
                )

        passed = len(failures) == 0
        return VerificationResult(
            test_name=test_name,
            passed=passed,
            expected="drop rates within expected ranges",
            actual=f"Failures: {failures}" if failures else "all drops normal",
            details=f"{mob_type.name}, Looting {looting_level}, {total_kills} kills",
        )

    # ========== TEST RUNNER ==========

    def run_scenario(self, scenario: dict[str, Any]) -> list[VerificationResult]:
        """Run a test scenario from JSON configuration."""
        results: list[VerificationResult] = []
        scenario_type = scenario.get("type", "")

        if scenario_type == "enderman_aggro":
            enderman = Mob(
                position=Vec3(**scenario["enderman"]["position"]),
                look_direction=Vec3(**scenario["enderman"]["look_direction"]).normalized(),
                mob_type=MobType.ENDERMAN,
                aggro_state=AggroState[scenario["enderman"].get("aggro_state", "NEUTRAL")],
            )
            player = Player(
                position=Vec3(**scenario["player"]["position"]),
                look_direction=Vec3(**scenario["player"]["look_direction"]).normalized(),
                is_wearing_pumpkin=scenario["player"].get("is_wearing_pumpkin", False),
            )
            results.append(self.verify_enderman_eye_contact_aggro(enderman, player))

        elif scenario_type == "enderman_teleport":
            enderman = Mob(
                position=Vec3(**scenario["enderman"]["position"]),
                look_direction=Vec3(0, 0, 1),
                mob_type=MobType.ENDERMAN,
                last_teleport_tick=scenario["enderman"].get("last_teleport_tick", 0),
            )
            target = Vec3(**scenario["teleport_target"])
            results.append(
                self.verify_enderman_teleport_pattern(
                    enderman, target, scenario.get("current_tick", 200)
                )
            )

        elif scenario_type == "blaze_attack":
            blaze = Mob(
                position=Vec3(**scenario["blaze"]["position"]),
                look_direction=Vec3(0, 0, 1),
                mob_type=MobType.BLAZE,
            )
            target = Entity(
                position=Vec3(**scenario["target"]["position"]),
                look_direction=Vec3(0, 0, 1),
            )
            fireballs = [
                Projectile(
                    origin=Vec3(**fb["origin"]),
                    velocity=Vec3(**fb["velocity"]),
                    damage=fb.get("damage", 5.0),
                    tick_fired=fb.get("tick_fired", 0),
                )
                for fb in scenario.get("fireballs", [])
            ]
            results.append(
                self.verify_blaze_fireball_timing(blaze, fireballs, scenario.get("current_tick", 0))
            )
            if fireballs:
                results.append(self.verify_blaze_fireball_trajectory(blaze, target, fireballs[0]))

        elif scenario_type == "zombie_targeting":
            zombie = Mob(
                position=Vec3(**scenario["zombie"]["position"]),
                look_direction=Vec3(0, 0, 1),
                mob_type=MobType.ZOMBIE,
                target=None,  # Will be set based on expected
            )
            players = [
                Player(
                    position=Vec3(**p["position"]),
                    look_direction=Vec3(0, 0, 1),
                    entity_id=p.get("id", f"player_{i}"),
                )
                for i, p in enumerate(scenario.get("players", []))
            ]
            villagers = [
                Entity(
                    position=Vec3(**v["position"]),
                    look_direction=Vec3(0, 0, 1),
                    entity_id=v.get("id", f"villager_{i}"),
                )
                for i, v in enumerate(scenario.get("villagers", []))
            ]
            # Set zombie's actual target based on scenario
            target_id = scenario["zombie"].get("target_id")
            if target_id:
                for p in players:
                    if p.entity_id == target_id:
                        zombie.target = p
                        break
                for v in villagers:
                    if v.entity_id == target_id:
                        zombie.target = v
                        break
            results.append(self.verify_zombie_targeting(zombie, players, villagers))

        elif scenario_type == "pigman_aggro":
            attacked = Mob(
                position=Vec3(**scenario["attacked_pigman"]["position"]),
                look_direction=Vec3(0, 0, 1),
                mob_type=MobType.PIGMAN,
                entity_id="attacked",
                aggro_state=AggroState.HOSTILE,
            )
            attacker = Player(
                position=Vec3(**scenario["attacker"]["position"]),
                look_direction=Vec3(0, 0, 1),
                entity_id="attacker",
            )
            nearby = [
                Mob(
                    position=Vec3(**p["position"]),
                    look_direction=Vec3(0, 0, 1),
                    mob_type=MobType.PIGMAN,
                    entity_id=p.get("id", f"pigman_{i}"),
                    aggro_state=AggroState[p.get("aggro_state", "NEUTRAL")],
                    target=attacker if p.get("targeting_attacker", False) else None,
                )
                for i, p in enumerate(scenario.get("nearby_pigmen", []))
            ]
            results.append(self.verify_pigman_group_aggro(attacked, nearby, attacker))

        elif scenario_type == "spawning":
            spawn_pos = Vec3(**scenario["spawn_position"])
            player_positions = [Vec3(**p) for p in scenario.get("player_positions", [])]
            results.append(
                self.verify_spawning_conditions(
                    spawn_position=spawn_pos,
                    light_level=scenario.get("light_level", 0),
                    player_positions=player_positions,
                    block_type=scenario.get("block_type", "stone"),
                    dimension=scenario.get("dimension", "overworld"),
                )
            )

        elif scenario_type == "drop_rates":
            results.append(
                self.verify_drop_rates(
                    mob_type=MobType[scenario["mob_type"]],
                    looting_level=scenario.get("looting_level", 0),
                    observed_drops=scenario.get("observed_drops", {}),
                    total_kills=scenario.get("total_kills", 100),
                )
            )

        return results

    def load_and_run_scenarios(self, scenario_file: Path) -> list[VerificationResult]:
        """Load scenarios from JSON and run all tests."""
        with open(scenario_file) as f:
            data = json.load(f)

        all_results: list[VerificationResult] = []
        for scenario in data.get("scenarios", []):
            logger.info(f"Running scenario: {scenario.get('name', 'unnamed')}")
            results = self.run_scenario(scenario)
            all_results.extend(results)
            self.results.extend(results)

        return all_results

    def generate_report(self) -> str:
        """Generate a summary report of all verification results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        report = [
            "=" * 60,
            "MOB AI VERIFICATION REPORT",
            "=" * 60,
            f"Total Tests: {total}",
            f"Passed: {passed}",
            f"Failed: {failed}",
            f"Pass Rate: {passed / max(total, 1) * 100:.1f}%",
            "-" * 60,
        ]

        if failed > 0:
            report.append("FAILURES:")
            for r in self.results:
                if not r.passed:
                    report.append(f"  - {r.test_name}: {r.details}")
                    report.append(f"    Expected: {r.expected}")
                    report.append(f"    Actual: {r.actual}")

        report.append("=" * 60)
        return "\n".join(report)


def main() -> None:
    """Main entry point for verification."""
    import argparse

    parser = argparse.ArgumentParser(description="Mob AI Verification System")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path(__file__).parent / "test_cases" / "mob_scenarios.json",
        help="Path to scenarios JSON file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    verifier = MobAIVerifier()

    if args.scenarios.exists():
        verifier.load_and_run_scenarios(args.scenarios)
    else:
        logger.warning(f"Scenario file not found: {args.scenarios}")
        logger.info("Running built-in test scenarios...")

        # Run some built-in tests
        enderman = Mob(
            position=Vec3(0, 64, 0),
            look_direction=Vec3(1, 0, 0).normalized(),
            mob_type=MobType.ENDERMAN,
            aggro_state=AggroState.HOSTILE,
        )
        player = Player(
            position=Vec3(10, 64, 0),
            look_direction=Vec3(-1, 0, 0).normalized(),
        )
        verifier.add_result(verifier.verify_enderman_eye_contact_aggro(enderman, player))

    print(verifier.generate_report())


if __name__ == "__main__":
    main()
