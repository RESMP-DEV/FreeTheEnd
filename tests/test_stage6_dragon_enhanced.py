"""Enhanced tests for Stage 6: End Dragon Fight.

Tests cover the complete Ender Dragon boss fight sequence including:
- Platform spawn mechanics
- Dragon and crystal interactions
- Combat strategies (bow, bed explosions)
- Death sequence and victory conditions
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


class BlockType(Enum):
    AIR = auto()
    OBSIDIAN = auto()
    END_STONE = auto()
    BEDROCK = auto()
    IRON_BARS = auto()
    END_CRYSTAL = auto()


class ItemType(Enum):
    BOW = auto()
    ARROW = auto()
    BED = auto()


class DragonPhase(Enum):
    CIRCLING = auto()
    STRAFING = auto()
    PERCHING = auto()
    DYING = auto()


@dataclass
class Position:
    x: float
    y: float
    z: float

    def distance_to(self, other: Position) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return (
            abs(self.x - other.x) < 0.01
            and abs(self.y - other.y) < 0.01
            and abs(self.z - other.z) < 0.01
        )


@dataclass
class Entity:
    position: Position
    health: float
    max_health: float
    alive: bool = True
    velocity: Position = field(default_factory=lambda: Position(0, 0, 0))


@dataclass
class Player(Entity):
    xp: int = 0
    inventory: list[ItemType] = field(default_factory=list)


@dataclass
class EnderCrystal(Entity):
    caged: bool = False  # Protected by iron bars
    heal_radius: float = 32.0
    heal_amount: float = 1.0  # HP per tick when dragon in range


@dataclass
class EnderDragon(Entity):
    phase: DragonPhase = DragonPhase.CIRCLING
    perch_timer: float = 0.0
    death_animation_time: float = 0.0
    crystals_alive: int = 10


class EndWorld:
    """Simplified End dimension simulation."""

    SPAWN_PLATFORM = Position(100, 49, 0)
    FOUNTAIN_CENTER = Position(0, 64, 0)
    VOID_Y = -64

    def __init__(self) -> None:
        self.blocks: dict[tuple[int, int, int], BlockType] = {}
        self.crystals: list[EnderCrystal] = []
        self.dragon: EnderDragon | None = None
        self.player: Player | None = None
        self.exit_portal_active = False
        self.victory = False
        self._initialize_end()

    def _initialize_end(self) -> None:
        # Create obsidian spawn platform
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                self.set_block(
                    int(self.SPAWN_PLATFORM.x) + dx,
                    int(self.SPAWN_PLATFORM.y),
                    int(self.SPAWN_PLATFORM.z) + dz,
                    BlockType.OBSIDIAN,
                )

        # Create 10 obsidian pillars with crystals
        pillar_positions = [
            (42, 76),
            (0, 76),
            (-42, 76),
            (42, 0),
            (-42, 0),
            (42, -76),
            (0, -76),
            (-42, -76),
            (21, 38),
            (-21, -38),
        ]
        for i, (px, pz) in enumerate(pillar_positions):
            height = 76 + (i * 3)  # Varying heights
            caged = i in (0, 2, 4)  # Some crystals are caged
            crystal = EnderCrystal(
                position=Position(px, height, pz),
                health=1,
                max_health=1,
                caged=caged,
            )
            self.crystals.append(crystal)
            if caged:
                # Add iron bars around crystal
                for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    self.set_block(px + dx, height, pz + dz, BlockType.IRON_BARS)

        # Create dragon
        self.dragon = EnderDragon(
            position=Position(0, 100, 0),
            health=200,
            max_health=200,
            crystals_alive=len(self.crystals),
        )

    def spawn_player(self, clear_obsidian: bool = False) -> Player:
        spawn = Position(self.SPAWN_PLATFORM.x, self.SPAWN_PLATFORM.y + 1, self.SPAWN_PLATFORM.z)

        if clear_obsidian:
            # Clear any obsidian blocking spawn position
            for dy in range(1, 3):
                pos = (int(spawn.x), int(spawn.y) + dy - 1, int(spawn.z))
                if self.get_block(*pos) == BlockType.OBSIDIAN:
                    self.set_block(*pos, BlockType.AIR)

        self.player = Player(
            position=spawn,
            health=20,
            max_health=20,
            inventory=[ItemType.BOW, ItemType.ARROW, ItemType.BED],
        )
        return self.player

    def set_block(self, x: int, y: int, z: int, block_type: BlockType) -> None:
        self.blocks[(x, y, z)] = block_type

    def get_block(self, x: int, y: int, z: int) -> BlockType:
        return self.blocks.get((x, y, z), BlockType.AIR)

    def tick(self) -> None:
        if not self.dragon or not self.player:
            return

        # Dragon healing from crystals
        if self.dragon.alive and self.dragon.phase != DragonPhase.DYING:
            for crystal in self.crystals:
                if crystal.alive:
                    dist = self.dragon.position.distance_to(crystal.position)
                    if dist <= crystal.heal_radius:
                        self.dragon.health = min(
                            self.dragon.max_health,
                            self.dragon.health + crystal.heal_amount,
                        )

        # Check void death
        if self.player.position.y <= self.VOID_Y:
            self.player.alive = False
            self.player.health = 0

        # Dragon death animation
        if self.dragon.phase == DragonPhase.DYING:
            self.dragon.death_animation_time += 1
            if self.dragon.death_animation_time >= 200:  # 10 seconds
                self._on_dragon_death()

    def _on_dragon_death(self) -> None:
        if not self.dragon or not self.player:
            return
        self.dragon.alive = False
        self.player.xp += 12000
        self.exit_portal_active = True

    def destroy_crystal(self, crystal: EnderCrystal) -> tuple[float, Position]:
        """Destroy a crystal, returns explosion damage and position."""
        crystal.alive = False
        crystal.health = 0
        if self.dragon:
            self.dragon.crystals_alive -= 1
        explosion_damage = 6.0  # Base explosion damage
        return explosion_damage, crystal.position

    def shoot_arrow_at(self, target: Position) -> bool:
        """Player shoots arrow at target. Returns True if hit."""
        if not self.player or ItemType.BOW not in self.player.inventory:
            return False

        # Check if any crystal is at target position
        for crystal in self.crystals:
            if crystal.alive and crystal.position.distance_to(target) < 1.0:
                # Check for iron bars blocking
                if crystal.caged:
                    # Need to check if bars are cleared
                    bars_cleared = all(
                        self.get_block(
                            int(crystal.position.x) + dx,
                            int(crystal.position.y),
                            int(crystal.position.z) + dz,
                        )
                        != BlockType.IRON_BARS
                        for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    )
                    if not bars_cleared:
                        return False
                self.destroy_crystal(crystal)
                return True
        return False

    def use_bed(self, target: Position) -> tuple[float, bool]:
        """Use bed at target position. Beds explode in End. Returns (damage, hit_dragon)."""
        if not self.player or ItemType.BED not in self.player.inventory:
            return 0.0, False

        explosion_damage = 100.0  # Bed explosion damage
        explosion_radius = 5.0

        hit_dragon = False
        if self.dragon and self.dragon.alive:
            dist = self.dragon.position.distance_to(target)
            if dist <= explosion_radius:
                # Damage falls off with distance
                actual_damage = explosion_damage * (1 - dist / explosion_radius)
                self.dragon.health -= actual_damage
                hit_dragon = True
                if self.dragon.health <= 0:
                    self.dragon.phase = DragonPhase.DYING

        # Player takes self-damage if too close
        if self.player.position.distance_to(target) <= explosion_radius:
            player_dist = self.player.position.distance_to(target)
            player_damage = explosion_damage * 0.5 * (1 - player_dist / explosion_radius)
            self.player.health -= player_damage
            if self.player.health <= 0:
                self.player.alive = False

        return explosion_damage, hit_dragon

    def dragon_perch(self) -> None:
        """Make dragon land on the fountain."""
        if self.dragon and self.dragon.alive:
            self.dragon.phase = DragonPhase.PERCHING
            self.dragon.position = Position(0, 65, 0)
            self.dragon.perch_timer = 200  # 10 seconds perch time

    def dragon_knockback(self, target: Player) -> None:
        """Dragon wing attack knocks back player."""
        if not self.dragon or not self.dragon.alive:
            return
        knockback_strength = 5.0
        # Direction away from dragon
        dx = target.position.x - self.dragon.position.x
        dz = target.position.z - self.dragon.position.z
        dist = math.sqrt(dx * dx + dz * dz)
        if dist > 0:
            target.velocity = Position(
                knockback_strength * dx / dist, 3.0, knockback_strength * dz / dist
            )
        # Wing attack deals damage
        target.health -= 5.0

    def enter_portal(self) -> bool:
        """Player enters exit portal. Returns True on victory."""
        if not self.exit_portal_active or not self.player:
            return False
        if self.player.position.distance_to(self.FOUNTAIN_CENTER) <= 5.0:
            self.victory = True
            return True
        return False


# =============================================================================
# TESTS
# =============================================================================


class TestSpawnMechanics:
    """Tests for End spawn platform mechanics."""

    def test_spawn_on_platform(self) -> None:
        """Player spawns at (100, 49, 0) on obsidian platform."""
        world = EndWorld()
        player = world.spawn_player()

        # Check spawn position (player spawns on top of platform)
        assert player.position.x == 100
        assert player.position.y == 50  # One block above platform
        assert player.position.z == 0

    def test_clear_spawn_obsidian(self) -> None:
        """Obsidian blocking spawn position is cleared."""
        world = EndWorld()

        # Place blocking obsidian
        world.set_block(100, 50, 0, BlockType.OBSIDIAN)
        world.set_block(100, 51, 0, BlockType.OBSIDIAN)

        player = world.spawn_player(clear_obsidian=True)

        # Blocking obsidian should be cleared
        assert world.get_block(100, 50, 0) == BlockType.AIR
        assert world.get_block(100, 51, 0) == BlockType.AIR
        assert player.alive


class TestDragonState:
    """Tests for dragon state on entry."""

    def test_dragon_alive_on_enter(self) -> None:
        """Dragon is alive when player enters the End."""
        world = EndWorld()
        world.spawn_player()

        assert world.dragon is not None
        assert world.dragon.alive
        assert world.dragon.health == 200
        assert world.dragon.max_health == 200

    def test_crystals_heal_dragon(self) -> None:
        """Dragon heals when near active crystals."""
        world = EndWorld()
        world.spawn_player()
        assert world.dragon is not None

        # Damage dragon
        world.dragon.health = 100

        # Move dragon near a crystal
        world.dragon.position = Position(
            world.crystals[0].position.x,
            world.crystals[0].position.y + 5,
            world.crystals[0].position.z,
        )

        initial_health = world.dragon.health

        # Tick the world
        for _ in range(10):
            world.tick()

        # Dragon should have healed
        assert world.dragon.health > initial_health


class TestCrystalDestruction:
    """Tests for destroying End crystals."""

    def test_destroy_crystal_arrow(self) -> None:
        """Bow destroys crystals with arrow shots."""
        world = EndWorld()
        world.spawn_player()

        # Get an uncaged crystal
        uncaged_crystal = next(c for c in world.crystals if not c.caged)
        initial_crystals = world.dragon.crystals_alive if world.dragon else 0

        # Shoot at crystal
        hit = world.shoot_arrow_at(uncaged_crystal.position)

        assert hit
        assert not uncaged_crystal.alive
        assert world.dragon is not None
        assert world.dragon.crystals_alive == initial_crystals - 1

    def test_crystal_explosion_damage(self) -> None:
        """Crystal explosion deals damage to nearby entities."""
        world = EndWorld()
        player = world.spawn_player()

        crystal = world.crystals[0]

        # Move player close to crystal
        player.position = Position(crystal.position.x + 2, crystal.position.y, crystal.position.z)

        # Destroy crystal
        damage, pos = world.destroy_crystal(crystal)

        assert damage == 6.0
        assert pos == crystal.position

    def test_caged_crystal_harder(self) -> None:
        """Iron bars block arrow shots to caged crystals."""
        world = EndWorld()
        world.spawn_player()

        # Find a caged crystal
        caged_crystal = next(c for c in world.crystals if c.caged)

        # Try to shoot it - should fail due to iron bars
        hit = world.shoot_arrow_at(caged_crystal.position)

        assert not hit
        assert caged_crystal.alive  # Crystal still alive

        # Remove iron bars
        for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            world.set_block(
                int(caged_crystal.position.x) + dx,
                int(caged_crystal.position.y),
                int(caged_crystal.position.z) + dz,
                BlockType.AIR,
            )

        # Now shooting should work
        hit = world.shoot_arrow_at(caged_crystal.position)
        assert hit
        assert not caged_crystal.alive


class TestDragonCombat:
    """Tests for dragon combat mechanics."""

    def test_dragon_perch_phase(self) -> None:
        """Dragon lands on fountain for perch attacks."""
        world = EndWorld()
        world.spawn_player()
        assert world.dragon is not None

        # Force dragon to perch
        world.dragon_perch()

        assert world.dragon.phase == DragonPhase.PERCHING
        # Dragon should be at fountain
        assert world.dragon.position.y == 65
        assert abs(world.dragon.position.x) < 1
        assert abs(world.dragon.position.z) < 1

    def test_bed_explosion(self) -> None:
        """Beds explode when used in the End."""
        world = EndWorld()
        player = world.spawn_player()

        # Player uses bed far from self
        player.position = Position(0, 65, 20)
        target = Position(0, 65, 0)

        damage, _ = world.use_bed(target)

        # Bed should explode with significant damage
        assert damage == 100.0

    def test_bed_damage_dragon(self) -> None:
        """Bed explosion deals ~100 damage to dragon when perching."""
        world = EndWorld()
        player = world.spawn_player()
        assert world.dragon is not None

        # Make dragon perch
        world.dragon_perch()
        initial_health = world.dragon.health

        # Player approaches and uses bed
        player.position = Position(0, 65, 6)  # Just outside explosion radius
        target = Position(0, 65, 0)  # Right on dragon

        _, hit = world.use_bed(target)

        assert hit
        damage_dealt = initial_health - world.dragon.health
        # Bed should deal close to 100 damage at point blank
        assert damage_dealt >= 80  # Accounting for distance falloff

    def test_one_cycle_possible(self) -> None:
        """3 beds can kill dragon (one-cycle strategy)."""
        world = EndWorld()
        player = world.spawn_player()
        assert world.dragon is not None

        # Dragon perches
        world.dragon_perch()

        # Give player 3 beds
        player.inventory = [ItemType.BED, ItemType.BED, ItemType.BED]
        player.position = Position(0, 65, 6)
        player.health = 20  # Reset health

        beds_used = 0
        while world.dragon.alive and ItemType.BED in player.inventory:
            _, hit = world.use_bed(Position(0, 65, 0))
            if hit:
                beds_used += 1
            player.inventory.remove(ItemType.BED)
            # Heal player between attempts for test purposes
            player.health = 20
            # Tick for death animation if dragon is dying
            if world.dragon.phase == DragonPhase.DYING:
                for _ in range(201):
                    world.tick()

        # One-cycle means 3 or fewer beds kill dragon
        assert beds_used <= 3
        assert world.dragon.health <= 0 or not world.dragon.alive

    def test_dragon_knockback(self) -> None:
        """Dragon wing swipe knocks back player."""
        world = EndWorld()
        player = world.spawn_player()
        assert world.dragon is not None

        # Position dragon and player close together
        world.dragon.position = Position(0, 65, 0)
        player.position = Position(3, 65, 0)
        initial_health = player.health

        world.dragon_knockback(player)

        # Player should have velocity away from dragon
        assert player.velocity.x > 0  # Pushed away from dragon at x=0
        assert player.velocity.y > 0  # Upward knockback
        # Player takes damage
        assert player.health < initial_health


class TestDeathConditions:
    """Tests for death conditions."""

    def test_void_death(self) -> None:
        """Falling into void kills player."""
        world = EndWorld()
        player = world.spawn_player()

        # Move player to void
        player.position = Position(0, -65, 0)

        world.tick()

        assert not player.alive
        assert player.health == 0


class TestVictorySequence:
    """Tests for dragon death and victory."""

    def test_dragon_death_sequence(self) -> None:
        """Dragon has proper death animation."""
        world = EndWorld()
        world.spawn_player()
        assert world.dragon is not None

        # Kill dragon
        world.dragon.health = 0
        world.dragon.phase = DragonPhase.DYING

        # Death animation takes 200 ticks (10 seconds)
        for i in range(199):
            world.tick()
            assert world.dragon.alive  # Still in death animation

        world.tick()  # Tick 200
        assert not world.dragon.alive

    def test_xp_drop(self) -> None:
        """Dragon drops 12000 XP on death."""
        world = EndWorld()
        player = world.spawn_player()
        assert world.dragon is not None

        initial_xp = player.xp

        # Kill dragon
        world.dragon.health = 0
        world.dragon.phase = DragonPhase.DYING

        # Complete death animation
        for _ in range(200):
            world.tick()

        assert player.xp == initial_xp + 12000

    def test_exit_portal_activates(self) -> None:
        """Exit portal appears after dragon death."""
        world = EndWorld()
        world.spawn_player()
        assert world.dragon is not None

        assert not world.exit_portal_active

        # Kill dragon
        world.dragon.health = 0
        world.dragon.phase = DragonPhase.DYING

        # Complete death animation
        for _ in range(200):
            world.tick()

        assert world.exit_portal_active

    def test_enter_exit_victory(self) -> None:
        """Entering exit portal wins the game."""
        world = EndWorld()
        player = world.spawn_player()
        assert world.dragon is not None

        # Kill dragon and activate portal
        world.dragon.health = 0
        world.dragon.phase = DragonPhase.DYING
        for _ in range(200):
            world.tick()

        # Move player to portal
        player.position = Position(0, 65, 0)

        result = world.enter_portal()

        assert result
        assert world.victory


class TestTimedFight:
    """Tests for timed fight completion."""

    def test_full_fight_timed(self) -> None:
        """Complete fight can be done under time limit."""
        world = EndWorld()
        player = world.spawn_player()
        assert world.dragon is not None

        start_time = time.time()
        max_fight_time = 30.0  # 30 seconds for test (simulated fight)

        # Destroy all crystals
        for crystal in world.crystals:
            if crystal.caged:
                # Clear iron bars
                for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    world.set_block(
                        int(crystal.position.x) + dx,
                        int(crystal.position.y),
                        int(crystal.position.z) + dz,
                        BlockType.AIR,
                    )
            world.shoot_arrow_at(crystal.position)

        assert world.dragon.crystals_alive == 0

        # Use bed strategy
        world.dragon_perch()
        player.position = Position(0, 65, 6)
        player.inventory = [ItemType.BED, ItemType.BED, ItemType.BED]

        while world.dragon.alive and ItemType.BED in player.inventory:
            world.use_bed(Position(0, 65, 0))
            if ItemType.BED in player.inventory:
                player.inventory.remove(ItemType.BED)
            player.health = 20  # Reset for test
            for _ in range(201):
                world.tick()

        # Move to portal
        player.position = Position(0, 65, 0)
        world.enter_portal()

        elapsed = time.time() - start_time

        assert world.victory
        assert elapsed < max_fight_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
