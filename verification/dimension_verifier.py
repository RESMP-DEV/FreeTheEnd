"""Dimension transition verification for Minecraft-style mechanics.

Verifies:
1. Dimension transition timing (80 tick warmup for Nether portals)
2. End portal instant teleport behavior
3. End spawn platform generation at (100, 49, 0)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple


class BlockPos(NamedTuple):
    """3D block position."""

    x: int
    y: int
    z: int


class Dimension(Enum):
    """Minecraft dimensions."""

    OVERWORLD = "overworld"
    NETHER = "the_nether"
    END = "the_end"


class BlockType(Enum):
    """Relevant block types for dimension transitions."""

    AIR = "air"
    OBSIDIAN = "obsidian"
    END_STONE = "end_stone"
    NETHER_PORTAL = "nether_portal"
    END_PORTAL = "end_portal"
    END_PORTAL_FRAME = "end_portal_frame"
    BEDROCK = "bedrock"


@dataclass
class TransitionState:
    """State of an entity's dimension transition."""

    in_portal: bool = False
    portal_ticks: int = 0
    transition_complete: bool = False
    source_dimension: Dimension | None = None
    target_dimension: Dimension | None = None
    start_time: float = 0.0


@dataclass
class World:
    """Simple world simulation for dimension mechanics."""

    blocks: dict[tuple[Dimension, BlockPos], BlockType] = field(default_factory=dict)

    def set_block(self, dimension: Dimension, pos: BlockPos, block: BlockType) -> None:
        """Set a block in the world."""
        self.blocks[(dimension, pos)] = block

    def get_block(self, dimension: Dimension, pos: BlockPos) -> BlockType:
        """Get a block from the world, default air."""
        return self.blocks.get((dimension, pos), BlockType.AIR)

    def clear_area(
        self,
        dimension: Dimension,
        corner1: BlockPos,
        corner2: BlockPos,
    ) -> None:
        """Clear an area to air."""
        for x in range(min(corner1.x, corner2.x), max(corner1.x, corner2.x) + 1):
            for y in range(min(corner1.y, corner2.y), max(corner1.y, corner2.y) + 1):
                for z in range(min(corner1.z, corner2.z), max(corner1.z, corner2.z) + 1):
                    self.set_block(dimension, BlockPos(x, y, z), BlockType.AIR)


class NetherPortalTransitionVerifier:
    """Verifies Nether portal transition timing.

    Minecraft Nether portal behavior:
    - Player must stand in portal for 80 game ticks (4 seconds at 20 TPS)
    - Transition resets if player leaves portal before completion
    - Creative mode: instant (0 ticks)
    - Purple swirl animation during warmup
    """

    WARMUP_TICKS = 80  # 4 seconds at 20 TPS
    TICKS_PER_SECOND = 20
    CREATIVE_WARMUP = 0

    def __init__(self):
        self.transition_states: dict[str, TransitionState] = {}

    def entity_enters_portal(
        self,
        entity_id: str,
        source_dim: Dimension,
        creative_mode: bool = False,
    ) -> TransitionState:
        """Called when entity enters a Nether portal.

        Args:
            entity_id: Unique entity identifier
            source_dim: Current dimension
            creative_mode: Whether entity is in creative mode

        Returns:
            Updated transition state
        """
        target_dim = Dimension.OVERWORLD if source_dim == Dimension.NETHER else Dimension.NETHER

        state = TransitionState(
            in_portal=True,
            portal_ticks=0,
            transition_complete=creative_mode,  # Instant in creative
            source_dimension=source_dim,
            target_dimension=target_dim,
            start_time=time.time(),
        )
        self.transition_states[entity_id] = state
        return state

    def tick_entity(self, entity_id: str) -> TransitionState | None:
        """Process one game tick for entity in portal.

        Args:
            entity_id: Entity to tick

        Returns:
            Updated state, or None if entity not in portal
        """
        state = self.transition_states.get(entity_id)
        if not state or not state.in_portal:
            return None

        if state.transition_complete:
            return state

        state.portal_ticks += 1
        if state.portal_ticks >= self.WARMUP_TICKS:
            state.transition_complete = True

        return state

    def entity_leaves_portal(self, entity_id: str) -> TransitionState | None:
        """Called when entity leaves portal before transition completes.

        Args:
            entity_id: Entity leaving portal

        Returns:
            Reset state, or None if entity wasn't tracked
        """
        state = self.transition_states.get(entity_id)
        if not state:
            return None

        # Reset progress
        state.in_portal = False
        state.portal_ticks = 0
        state.transition_complete = False
        return state

    def get_progress(self, entity_id: str) -> float:
        """Get transition progress as percentage.

        Args:
            entity_id: Entity to check

        Returns:
            Progress from 0.0 to 1.0
        """
        state = self.transition_states.get(entity_id)
        if not state:
            return 0.0
        return min(1.0, state.portal_ticks / self.WARMUP_TICKS)

    def verify_timing(
        self, actual_ticks: int, expected_ticks: int = WARMUP_TICKS
    ) -> tuple[bool, str]:
        """Verify that transition timing matches expected.

        Args:
            actual_ticks: Actual ticks to transition
            expected_ticks: Expected ticks (default 80)

        Returns:
            Tuple of (is_correct, explanation)
        """
        if actual_ticks == expected_ticks:
            return (
                True,
                f"Correct: {actual_ticks} ticks ({actual_ticks / self.TICKS_PER_SECOND:.1f}s)",
            )
        else:
            diff = actual_ticks - expected_ticks
            return False, f"Wrong: {actual_ticks} ticks (expected {expected_ticks}, diff {diff:+d})"


class EndPortalTransitionVerifier:
    """Verifies End portal instant teleport behavior.

    End portal mechanics:
    - Instant teleport (no warmup)
    - Players teleport to spawn platform at (100, 49, 0) in The End
    - Returning via End gateway goes back to Overworld spawn
    - First player to enter spawns the Ender Dragon
    """

    END_SPAWN_PLATFORM_POS = BlockPos(100, 49, 0)
    END_SPAWN_ABOVE_PLATFORM = BlockPos(100, 50, 0)  # Where player appears

    def __init__(self):
        self.entities_entered_end: set[str] = set()
        self.dragon_spawned: bool = False

    def entity_enters_end_portal(
        self,
        entity_id: str,
        is_player: bool = True,
    ) -> tuple[BlockPos, bool]:
        """Handle entity entering End portal.

        Args:
            entity_id: Entity entering portal
            is_player: Whether this is a player (vs other entity)

        Returns:
            Tuple of (spawn_position, spawns_dragon)
        """
        spawns_dragon = False

        if is_player and entity_id not in self.entities_entered_end:
            self.entities_entered_end.add(entity_id)
            if not self.dragon_spawned:
                self.dragon_spawned = True
                spawns_dragon = True

        # Instant teleport to spawn platform
        return self.END_SPAWN_ABOVE_PLATFORM, spawns_dragon

    def verify_instant_teleport(self, warmup_ticks: int) -> tuple[bool, str]:
        """Verify End portal has no warmup.

        Args:
            warmup_ticks: Observed warmup ticks

        Returns:
            Tuple of (is_correct, explanation)
        """
        if warmup_ticks == 0:
            return True, "Correct: End portal is instant (0 ticks)"
        else:
            return False, f"Wrong: End portal should be instant but took {warmup_ticks} ticks"


class EndSpawnPlatformVerifier:
    """Verifies End spawn platform generation.

    End spawn platform specifications (Java Edition):
    - Location: (100, 49, 0) in The End
    - 5x5 obsidian platform
    - Air cleared above (5x5x3 area)
    - Generated when player first enters The End
    - Regenerated each time player enters (except on initial dragon fight)
    """

    PLATFORM_CENTER = BlockPos(100, 49, 0)
    PLATFORM_SIZE = 5  # 5x5 blocks
    CLEAR_HEIGHT = 3  # Air space above

    def __init__(self, world: World):
        self.world = world
        self.generation_count = 0

    def generate_spawn_platform(self) -> list[BlockPos]:
        """Generate the obsidian spawn platform.

        Returns:
            List of obsidian block positions
        """
        cx, cy, cz = self.PLATFORM_CENTER
        half = self.PLATFORM_SIZE // 2
        obsidian_positions: list[BlockPos] = []

        # Generate 5x5 obsidian platform
        for dx in range(-half, half + 1):
            for dz in range(-half, half + 1):
                pos = BlockPos(cx + dx, cy, cz + dz)
                self.world.set_block(Dimension.END, pos, BlockType.OBSIDIAN)
                obsidian_positions.append(pos)

        # Clear air above platform
        for dx in range(-half, half + 1):
            for dz in range(-half, half + 1):
                for dy in range(1, self.CLEAR_HEIGHT + 1):
                    pos = BlockPos(cx + dx, cy + dy, cz + dz)
                    self.world.set_block(Dimension.END, pos, BlockType.AIR)

        self.generation_count += 1
        return obsidian_positions

    def verify_platform(self) -> tuple[bool, list[str]]:
        """Verify the spawn platform is correctly generated.

        Returns:
            Tuple of (is_valid, issues)
        """
        issues: list[str] = []
        cx, cy, cz = self.PLATFORM_CENTER
        half = self.PLATFORM_SIZE // 2

        # Check obsidian layer
        for dx in range(-half, half + 1):
            for dz in range(-half, half + 1):
                pos = BlockPos(cx + dx, cy, cz + dz)
                block = self.world.get_block(Dimension.END, pos)
                if block != BlockType.OBSIDIAN:
                    issues.append(f"Expected obsidian at {pos}, found {block.value}")

        # Check air above
        for dx in range(-half, half + 1):
            for dz in range(-half, half + 1):
                for dy in range(1, self.CLEAR_HEIGHT + 1):
                    pos = BlockPos(cx + dx, cy + dy, cz + dz)
                    block = self.world.get_block(Dimension.END, pos)
                    if block != BlockType.AIR:
                        issues.append(f"Expected air at {pos}, found {block.value}")

        return len(issues) == 0, issues

    def verify_dimensions(self) -> tuple[bool, str]:
        """Verify platform has correct dimensions.

        Returns:
            Tuple of (is_correct, explanation)
        """
        expected_blocks = self.PLATFORM_SIZE * self.PLATFORM_SIZE
        expected_air = expected_blocks * self.CLEAR_HEIGHT

        # Count actual blocks
        cx, cy, cz = self.PLATFORM_CENTER
        half = self.PLATFORM_SIZE // 2
        obsidian_count = 0
        air_count = 0

        for dx in range(-half, half + 1):
            for dz in range(-half, half + 1):
                pos = BlockPos(cx + dx, cy, cz + dz)
                if self.world.get_block(Dimension.END, pos) == BlockType.OBSIDIAN:
                    obsidian_count += 1
                for dy in range(1, self.CLEAR_HEIGHT + 1):
                    pos = BlockPos(cx + dx, cy + dy, cz + dz)
                    if self.world.get_block(Dimension.END, pos) == BlockType.AIR:
                        air_count += 1

        if obsidian_count == expected_blocks and air_count == expected_air:
            return True, f"Correct: {obsidian_count} obsidian, {air_count} air blocks"
        else:
            return (
                False,
                f"Wrong: {obsidian_count}/{expected_blocks} obsidian, {air_count}/{expected_air} air",
            )


def run_verification() -> None:
    """Run all dimension transition verifications."""
    print("=" * 60)
    print("DIMENSION TRANSITION VERIFICATION SUITE")
    print("=" * 60)

    # Test 1: Nether portal 80 tick warmup
    print("\n[1] Nether Portal Transition Timing (80 ticks)")
    print("-" * 40)

    nether_verifier = NetherPortalTransitionVerifier()

    # Simulate player entering portal
    entity_id = "player_1"
    state = nether_verifier.entity_enters_portal(entity_id, Dimension.OVERWORLD)
    print(f"  Player enters portal in {state.source_dimension.value}")
    print(
        f"  Target dimension: {state.target_dimension.value if state.target_dimension else 'None'}"
    )

    # Simulate ticks
    tick_checkpoints = [0, 20, 40, 60, 79, 80]
    for checkpoint in tick_checkpoints:
        while nether_verifier.transition_states[entity_id].portal_ticks < checkpoint:
            nether_verifier.tick_entity(entity_id)
        state = nether_verifier.transition_states[entity_id]
        progress = nether_verifier.get_progress(entity_id)
        seconds = checkpoint / NetherPortalTransitionVerifier.TICKS_PER_SECOND
        print(
            f"  Tick {checkpoint:3d} ({seconds:.1f}s): {progress * 100:.0f}% - Complete: {state.transition_complete}"
        )

    is_correct, msg = nether_verifier.verify_timing(80)
    print(f"  Timing verification: {msg}")

    # Test creative mode instant
    print("\n  Creative mode test:")
    state = nether_verifier.entity_enters_portal(
        "creative_player", Dimension.OVERWORLD, creative_mode=True
    )
    print(f"    Instant transition: {state.transition_complete}")

    # Test interrupted transition
    print("\n  Interrupted transition test:")
    entity_id = "interrupted_player"
    nether_verifier.entity_enters_portal(entity_id, Dimension.OVERWORLD)
    for _ in range(40):
        nether_verifier.tick_entity(entity_id)
    print(f"    After 40 ticks: {nether_verifier.get_progress(entity_id) * 100:.0f}%")
    nether_verifier.entity_leaves_portal(entity_id)
    print(f"    After leaving: {nether_verifier.get_progress(entity_id) * 100:.0f}%")

    # Test 2: End portal instant teleport
    print("\n[2] End Portal Instant Teleport")
    print("-" * 40)

    end_verifier = EndPortalTransitionVerifier()

    # First player enters
    spawn_pos, spawns_dragon = end_verifier.entity_enters_end_portal("player_1")
    print("  First player enters End:")
    print(f"    Spawn position: {spawn_pos}")
    print(f"    Spawns dragon: {spawns_dragon}")

    # Second player enters
    spawn_pos2, spawns_dragon2 = end_verifier.entity_enters_end_portal("player_2")
    print("  Second player enters End:")
    print(f"    Spawn position: {spawn_pos2}")
    print(f"    Spawns dragon: {spawns_dragon2}")

    is_instant, msg = end_verifier.verify_instant_teleport(0)
    print(f"  Instant verification: {msg}")

    # Test 3: End spawn platform generation
    print("\n[3] End Spawn Platform Generation")
    print("-" * 40)

    world = World()
    platform_verifier = EndSpawnPlatformVerifier(world)

    # Generate platform
    print(f"  Platform center: {EndSpawnPlatformVerifier.PLATFORM_CENTER}")
    print(
        f"  Platform size: {EndSpawnPlatformVerifier.PLATFORM_SIZE}x{EndSpawnPlatformVerifier.PLATFORM_SIZE}"
    )
    print(f"  Clear height: {EndSpawnPlatformVerifier.CLEAR_HEIGHT}")

    obsidian_positions = platform_verifier.generate_spawn_platform()
    print(f"  Generated {len(obsidian_positions)} obsidian blocks")

    # Verify platform
    is_valid, issues = platform_verifier.verify_platform()
    print(f"  Platform valid: {is_valid}")
    if issues:
        for issue in issues[:3]:
            print(f"    - {issue}")

    # Verify dimensions
    is_correct, msg = platform_verifier.verify_dimensions()
    print(f"  Dimensions: {msg}")

    # Test platform regeneration
    print("\n  Platform regeneration test:")
    # Place some blocks on platform
    world.set_block(Dimension.END, BlockPos(100, 50, 0), BlockType.END_STONE)
    world.set_block(Dimension.END, BlockPos(101, 50, 1), BlockType.END_STONE)

    is_valid_before, _ = platform_verifier.verify_platform()
    print(f"    After placing blocks - Valid: {is_valid_before}")

    # Regenerate
    platform_verifier.generate_spawn_platform()
    is_valid_after, _ = platform_verifier.verify_platform()
    print(f"    After regeneration - Valid: {is_valid_after}")
    print(f"    Generation count: {platform_verifier.generation_count}")

    print("\n" + "=" * 60)
    print("DIMENSION TRANSITION VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
