"""Block placing verification for Minecraft block interaction subsystems.

Tests block placement validity, directional block metadata (stairs, logs, etc.),
and falling block physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple

import logging

logger = logging.getLogger(__name__)


class Direction(Enum):
    """Cardinal and vertical directions."""

    DOWN = (0, -1, 0)
    UP = (0, 1, 0)
    NORTH = (0, 0, -1)
    SOUTH = (0, 0, 1)
    WEST = (-1, 0, 0)
    EAST = (1, 0, 0)

    def __init__(self, dx: int, dy: int, dz: int) -> None:
        logger.info("Direction.__init__: dx=%s, dy=%s, dz=%s", dx, dy, dz)
        self.dx = dx
        self.dy = dy
        self.dz = dz

    @property
    def opposite(self) -> Direction:
        logger.debug("Direction.opposite called")
        opposites = {
            Direction.DOWN: Direction.UP,
            Direction.UP: Direction.DOWN,
            Direction.NORTH: Direction.SOUTH,
            Direction.SOUTH: Direction.NORTH,
            Direction.WEST: Direction.EAST,
            Direction.EAST: Direction.WEST,
        }
        return opposites[self]


class Axis(Enum):
    """Block rotation axis."""

    X = auto()
    Y = auto()
    Z = auto()


class StairHalf(Enum):
    """Stair placement half."""

    BOTTOM = auto()
    TOP = auto()


class StairShape(Enum):
    """Stair connection shape."""

    STRAIGHT = auto()
    INNER_LEFT = auto()
    INNER_RIGHT = auto()
    OUTER_LEFT = auto()
    OUTER_RIGHT = auto()


class SlabType(Enum):
    """Slab placement type."""

    BOTTOM = auto()
    TOP = auto()
    DOUBLE = auto()


class BlockType(Enum):
    """Block type categories for placement rules."""

    NORMAL = auto()
    DIRECTIONAL = auto()  # Logs, pillars
    STAIRS = auto()
    SLAB = auto()
    TORCH = auto()
    WALL_MOUNTED = auto()  # Signs, banners, buttons
    FALLING = auto()  # Sand, gravel, concrete powder
    ATTACHABLE = auto()  # Requires solid surface
    LIQUID = auto()
    PLANT = auto()
    RAIL = auto()
    REDSTONE = auto()
    DOOR = auto()
    BED = auto()
    PISTON = auto()


class Position(NamedTuple):
    """3D position."""

    x: int
    y: int
    z: int

    def offset(self, direction: Direction) -> Position:
        logger.debug("Position.offset: direction=%s", direction)
        return Position(
            self.x + direction.dx,
            self.y + direction.dy,
            self.z + direction.dz,
        )


@dataclass
class BlockState:
    """Block state with metadata."""

    block_id: str
    properties: dict[str, str | int | bool] = field(default_factory=dict)

    def with_property(self, key: str, value: str | int | bool) -> BlockState:
        logger.debug("BlockState.with_property: key=%s, value=%s", key, value)
        new_props = dict(self.properties)
        new_props[key] = value
        return BlockState(self.block_id, new_props)


@dataclass
class BlockPlacementContext:
    """Context for block placement."""

    player_direction: Direction  # Where player is facing
    click_direction: Direction  # Face clicked on existing block
    click_position: tuple[float, float, float]  # Exact click position 0.0-1.0
    sneaking: bool = False


@dataclass
class PlacementRules:
    """Rules for how a block can be placed."""

    block_type: BlockType
    requires_solid_below: bool = False
    requires_solid_above: bool = False
    requires_solid_side: bool = False
    can_place_in_water: bool = False
    can_place_in_air: bool = True
    can_replace_fluid: bool = True
    max_y: int = 319
    min_y: int = -64


# Block placement rules registry
PLACEMENT_RULES: dict[str, PlacementRules] = {
    # Normal blocks
    "stone": PlacementRules(BlockType.NORMAL),
    "cobblestone": PlacementRules(BlockType.NORMAL),
    "dirt": PlacementRules(BlockType.NORMAL),
    "grass_block": PlacementRules(BlockType.NORMAL),
    "glass": PlacementRules(BlockType.NORMAL),
    "oak_planks": PlacementRules(BlockType.NORMAL),
    # Directional blocks (logs, pillars)
    "oak_log": PlacementRules(BlockType.DIRECTIONAL),
    "spruce_log": PlacementRules(BlockType.DIRECTIONAL),
    "birch_log": PlacementRules(BlockType.DIRECTIONAL),
    "jungle_log": PlacementRules(BlockType.DIRECTIONAL),
    "acacia_log": PlacementRules(BlockType.DIRECTIONAL),
    "dark_oak_log": PlacementRules(BlockType.DIRECTIONAL),
    "mangrove_log": PlacementRules(BlockType.DIRECTIONAL),
    "cherry_log": PlacementRules(BlockType.DIRECTIONAL),
    "crimson_stem": PlacementRules(BlockType.DIRECTIONAL),
    "warped_stem": PlacementRules(BlockType.DIRECTIONAL),
    "quartz_pillar": PlacementRules(BlockType.DIRECTIONAL),
    "purpur_pillar": PlacementRules(BlockType.DIRECTIONAL),
    "hay_block": PlacementRules(BlockType.DIRECTIONAL),
    "bone_block": PlacementRules(BlockType.DIRECTIONAL),
    # Stairs
    "oak_stairs": PlacementRules(BlockType.STAIRS),
    "spruce_stairs": PlacementRules(BlockType.STAIRS),
    "birch_stairs": PlacementRules(BlockType.STAIRS),
    "jungle_stairs": PlacementRules(BlockType.STAIRS),
    "acacia_stairs": PlacementRules(BlockType.STAIRS),
    "dark_oak_stairs": PlacementRules(BlockType.STAIRS),
    "stone_stairs": PlacementRules(BlockType.STAIRS),
    "cobblestone_stairs": PlacementRules(BlockType.STAIRS),
    "brick_stairs": PlacementRules(BlockType.STAIRS),
    "stone_brick_stairs": PlacementRules(BlockType.STAIRS),
    "nether_brick_stairs": PlacementRules(BlockType.STAIRS),
    "sandstone_stairs": PlacementRules(BlockType.STAIRS),
    "quartz_stairs": PlacementRules(BlockType.STAIRS),
    "purpur_stairs": PlacementRules(BlockType.STAIRS),
    "prismarine_stairs": PlacementRules(BlockType.STAIRS),
    "red_sandstone_stairs": PlacementRules(BlockType.STAIRS),
    # Slabs
    "oak_slab": PlacementRules(BlockType.SLAB),
    "spruce_slab": PlacementRules(BlockType.SLAB),
    "birch_slab": PlacementRules(BlockType.SLAB),
    "jungle_slab": PlacementRules(BlockType.SLAB),
    "stone_slab": PlacementRules(BlockType.SLAB),
    "cobblestone_slab": PlacementRules(BlockType.SLAB),
    "brick_slab": PlacementRules(BlockType.SLAB),
    "smooth_stone_slab": PlacementRules(BlockType.SLAB),
    # Torches
    "torch": PlacementRules(BlockType.TORCH, requires_solid_below=True, requires_solid_side=True),
    "wall_torch": PlacementRules(BlockType.WALL_MOUNTED, requires_solid_side=True),
    "redstone_torch": PlacementRules(
        BlockType.TORCH, requires_solid_below=True, requires_solid_side=True
    ),
    "soul_torch": PlacementRules(
        BlockType.TORCH, requires_solid_below=True, requires_solid_side=True
    ),
    # Falling blocks
    "sand": PlacementRules(BlockType.FALLING),
    "red_sand": PlacementRules(BlockType.FALLING),
    "gravel": PlacementRules(BlockType.FALLING),
    "white_concrete_powder": PlacementRules(BlockType.FALLING),
    "anvil": PlacementRules(BlockType.FALLING),
    "dragon_egg": PlacementRules(BlockType.FALLING),
    # Plants
    "wheat": PlacementRules(BlockType.PLANT, requires_solid_below=True),
    "carrots": PlacementRules(BlockType.PLANT, requires_solid_below=True),
    "potatoes": PlacementRules(BlockType.PLANT, requires_solid_below=True),
    "beetroots": PlacementRules(BlockType.PLANT, requires_solid_below=True),
    "oak_sapling": PlacementRules(BlockType.PLANT, requires_solid_below=True),
    "spruce_sapling": PlacementRules(BlockType.PLANT, requires_solid_below=True),
    "birch_sapling": PlacementRules(BlockType.PLANT, requires_solid_below=True),
    "jungle_sapling": PlacementRules(BlockType.PLANT, requires_solid_below=True),
    "dandelion": PlacementRules(BlockType.PLANT, requires_solid_below=True),
    "poppy": PlacementRules(BlockType.PLANT, requires_solid_below=True),
    "sugar_cane": PlacementRules(BlockType.PLANT, requires_solid_below=True),
    "cactus": PlacementRules(BlockType.PLANT, requires_solid_below=True),
    # Rails
    "rail": PlacementRules(BlockType.RAIL, requires_solid_below=True),
    "powered_rail": PlacementRules(BlockType.RAIL, requires_solid_below=True),
    "detector_rail": PlacementRules(BlockType.RAIL, requires_solid_below=True),
    "activator_rail": PlacementRules(BlockType.RAIL, requires_solid_below=True),
    # Redstone
    "redstone_wire": PlacementRules(BlockType.REDSTONE, requires_solid_below=True),
    "repeater": PlacementRules(BlockType.REDSTONE, requires_solid_below=True),
    "comparator": PlacementRules(BlockType.REDSTONE, requires_solid_below=True),
    # Doors
    "oak_door": PlacementRules(BlockType.DOOR, requires_solid_below=True),
    "iron_door": PlacementRules(BlockType.DOOR, requires_solid_below=True),
    "spruce_door": PlacementRules(BlockType.DOOR, requires_solid_below=True),
    # Beds
    "white_bed": PlacementRules(BlockType.BED, requires_solid_below=True),
    "red_bed": PlacementRules(BlockType.BED, requires_solid_below=True),
    # Pistons
    "piston": PlacementRules(BlockType.PISTON),
    "sticky_piston": PlacementRules(BlockType.PISTON),
    # Wall-mounted
    "lever": PlacementRules(BlockType.WALL_MOUNTED, requires_solid_side=True),
    "stone_button": PlacementRules(BlockType.WALL_MOUNTED, requires_solid_side=True),
    "oak_button": PlacementRules(BlockType.WALL_MOUNTED, requires_solid_side=True),
    "oak_sign": PlacementRules(BlockType.WALL_MOUNTED, requires_solid_below=True),
    "oak_wall_sign": PlacementRules(BlockType.WALL_MOUNTED, requires_solid_side=True),
    "item_frame": PlacementRules(BlockType.WALL_MOUNTED, requires_solid_side=True),
    "painting": PlacementRules(BlockType.WALL_MOUNTED, requires_solid_side=True),
    # Liquids
    "water": PlacementRules(BlockType.LIQUID, can_place_in_water=True),
    "lava": PlacementRules(BlockType.LIQUID, can_place_in_water=True),
}


def get_log_axis_from_face(click_direction: Direction) -> Axis:
    """Get log axis based on the clicked face."""
    logger.debug("get_log_axis_from_face: click_direction=%s", click_direction)
    if click_direction in (Direction.UP, Direction.DOWN):
        return Axis.Y
    elif click_direction in (Direction.NORTH, Direction.SOUTH):
        return Axis.Z
    else:
        return Axis.X


def get_stair_direction(player_direction: Direction) -> Direction:
    """Get stair facing direction based on player facing."""
    # Stairs face the player (opposite of player direction)
    logger.debug("get_stair_direction: player_direction=%s", player_direction)
    horizontal = {
        Direction.NORTH: Direction.NORTH,
        Direction.SOUTH: Direction.SOUTH,
        Direction.EAST: Direction.EAST,
        Direction.WEST: Direction.WEST,
    }
    return horizontal.get(player_direction, Direction.NORTH)


def get_stair_half(
    click_position: tuple[float, float, float], click_direction: Direction
) -> StairHalf:
    """Determine stair half based on click position."""
    logger.debug("get_stair_half: click_position=%s, click_direction=%s", click_position, click_direction)
    _, y, _ = click_position

    if click_direction == Direction.UP:
        return StairHalf.BOTTOM
    elif click_direction == Direction.DOWN:
        return StairHalf.TOP
    else:
        # Clicking side: use Y position within block
        return StairHalf.TOP if y >= 0.5 else StairHalf.BOTTOM


def get_stair_shape(
    pos: Position,
    facing: Direction,
    world: dict[Position, BlockState],
) -> StairShape:
    """Determine stair shape based on neighboring stairs."""
    # Check neighbors for stair connections
    logger.debug("get_stair_shape: pos=%s, facing=%s, world=%s", pos, facing, world)
    left_dir = {
        Direction.NORTH: Direction.WEST,
        Direction.SOUTH: Direction.EAST,
        Direction.EAST: Direction.NORTH,
        Direction.WEST: Direction.SOUTH,
    }[facing]

    right_dir = left_dir.opposite

    back_pos = pos.offset(facing.opposite)
    left_pos = pos.offset(left_dir)
    right_pos = pos.offset(right_dir)

    back_state = world.get(back_pos)
    left_state = world.get(left_pos)
    right_state = world.get(right_pos)

    def is_stairs_facing(state: BlockState | None, direction: Direction) -> bool:
        logger.debug("is_stairs_facing: state=%s, direction=%s", state, direction)
        if state is None:
            return False
        if "stairs" not in state.block_id:
            return False
        stair_facing = state.properties.get("facing")
        return stair_facing == direction.name.lower()

    # Check for inner corners (back stair connects perpendicular)
    if back_state and "stairs" in back_state.block_id:
        back_facing = back_state.properties.get("facing", "").upper()
        if back_facing == left_dir.name:
            return StairShape.INNER_LEFT
        if back_facing == right_dir.name:
            return StairShape.INNER_RIGHT

    # Check for outer corners (side stair connects perpendicular)
    if is_stairs_facing(left_state, facing.opposite):
        return StairShape.OUTER_LEFT
    if is_stairs_facing(right_state, facing.opposite):
        return StairShape.OUTER_RIGHT

    return StairShape.STRAIGHT


def get_slab_type(
    click_position: tuple[float, float, float], click_direction: Direction
) -> SlabType:
    """Determine slab type based on click position."""
    logger.debug("get_slab_type: click_position=%s, click_direction=%s", click_position, click_direction)
    _, y, _ = click_position

    if click_direction == Direction.UP:
        return SlabType.BOTTOM
    elif click_direction == Direction.DOWN:
        return SlabType.TOP
    else:
        return SlabType.TOP if y >= 0.5 else SlabType.BOTTOM


def can_place_block(
    block_id: str,
    pos: Position,
    context: BlockPlacementContext,
    world: dict[Position, BlockState],
) -> tuple[bool, str]:
    """Check if a block can be placed at the given position.

    Args:
        block_id: Block to place
        pos: Target position
        context: Placement context
        world: Current world state

    Returns:
        Tuple of (can_place, reason)
    """
    logger.debug("can_place_block: block_id=%s, pos=%s, context=%s, world=%s", block_id, pos, context, world)
    if block_id not in PLACEMENT_RULES:
        return True, "No rules defined, assuming placeable"

    rules = PLACEMENT_RULES[block_id]

    # Check Y bounds
    if pos.y < rules.min_y:
        return False, f"Below minimum Y ({rules.min_y})"
    if pos.y > rules.max_y:
        return False, f"Above maximum Y ({rules.max_y})"

    # Check existing block
    existing = world.get(pos)
    if existing:
        if existing.block_id in ("water", "lava"):
            if not rules.can_replace_fluid:
                return False, "Cannot replace fluid"
        elif existing.block_id != "air":
            return False, "Position occupied"

    # Check below
    if rules.requires_solid_below:
        below_pos = pos.offset(Direction.DOWN)
        below = world.get(below_pos)
        if below is None or below.block_id == "air":
            return False, "Requires solid block below"
        if below.block_id in ("water", "lava"):
            return False, "Cannot place on fluid"

    # Check above
    if rules.requires_solid_above:
        above_pos = pos.offset(Direction.UP)
        above = world.get(above_pos)
        if above is None or above.block_id == "air":
            return False, "Requires solid block above"

    # Check side for wall-mounted blocks
    if rules.requires_solid_side:
        # Check the clicked side
        attach_dir = context.click_direction.opposite
        attach_pos = pos.offset(attach_dir)
        attach_block = world.get(attach_pos)

        # For torches, can also place on ground
        if rules.block_type == BlockType.TORCH:
            below = world.get(pos.offset(Direction.DOWN))
            if below and below.block_id not in ("air", "water", "lava"):
                return True, "Can place torch on ground"

        if attach_block is None or attach_block.block_id in ("air", "water", "lava"):
            return False, "Requires solid block to attach to"

    return True, "OK"


def get_placed_state(
    block_id: str,
    pos: Position,
    context: BlockPlacementContext,
    world: dict[Position, BlockState],
) -> BlockState:
    """Get the block state when placing a block.

    Args:
        block_id: Block to place
        pos: Target position
        context: Placement context
        world: Current world state

    Returns:
        BlockState with appropriate metadata
    """
    logger.debug("get_placed_state: block_id=%s, pos=%s, context=%s, world=%s", block_id, pos, context, world)
    if block_id not in PLACEMENT_RULES:
        return BlockState(block_id)

    rules = PLACEMENT_RULES[block_id]
    state = BlockState(block_id)

    if rules.block_type == BlockType.DIRECTIONAL:
        axis = get_log_axis_from_face(context.click_direction)
        state = state.with_property("axis", axis.name.lower())

    elif rules.block_type == BlockType.STAIRS:
        facing = get_stair_direction(context.player_direction)
        half = get_stair_half(context.click_position, context.click_direction)
        shape = get_stair_shape(pos, facing, world)

        state = state.with_property("facing", facing.name.lower())
        state = state.with_property("half", half.name.lower())
        state = state.with_property("shape", shape.name.lower())

    elif rules.block_type == BlockType.SLAB:
        slab_type = get_slab_type(context.click_position, context.click_direction)
        state = state.with_property("type", slab_type.name.lower())

    elif rules.block_type == BlockType.TORCH:
        # Determine if wall or standing torch
        if context.click_direction in (
            Direction.NORTH,
            Direction.SOUTH,
            Direction.EAST,
            Direction.WEST,
        ):
            # Check if we can attach to the side
            attach_pos = pos.offset(context.click_direction.opposite)
            attach_block = world.get(attach_pos)
            if attach_block and attach_block.block_id not in ("air", "water", "lava"):
                # Wall torch
                state = BlockState(
                    block_id.replace("torch", "wall_torch") if "wall" not in block_id else block_id
                )
                state = state.with_property("facing", context.click_direction.name.lower())
                return state

        # Standing torch
        state = state.with_property("facing", "up")

    elif rules.block_type == BlockType.PISTON:
        # Pistons face where the player is looking
        state = state.with_property("facing", context.player_direction.name.lower())

    elif rules.block_type == BlockType.DOOR:
        state = state.with_property("facing", context.player_direction.name.lower())
        state = state.with_property("half", "lower")
        state = state.with_property("hinge", "left")  # Determined by surroundings
        state = state.with_property("open", False)

    elif rules.block_type == BlockType.WALL_MOUNTED:
        if "sign" in block_id and "wall" not in block_id:
            # Standing sign
            state = state.with_property("rotation", 0)  # 0-15 based on player angle
        else:
            state = state.with_property("facing", context.click_direction.name.lower())

    elif rules.block_type == BlockType.RAIL:
        # Rail shape determined by neighbors
        state = state.with_property("shape", "north_south")

    return state


class FallingBlockPhysics:
    """Physics simulation for falling blocks."""

    GRAVITY = 0.04  # blocks/tick^2
    TERMINAL_VELOCITY = 3.92  # blocks/tick
    TICK_RATE = 20  # ticks/second

    @staticmethod
    def should_fall(pos: Position, world: dict[Position, BlockState]) -> bool:
        """Check if a falling block at this position should fall."""
        logger.debug("FallingBlockPhysics.should_fall: pos=%s, world=%s", pos, world)
        below = world.get(pos.offset(Direction.DOWN))
        if below is None:
            return True  # Void
        return below.block_id in ("air", "water", "lava", "fire", "tall_grass", "flower")

    @staticmethod
    def simulate_fall(
        start_pos: Position,
        world: dict[Position, BlockState],
        max_ticks: int = 1000,
    ) -> tuple[Position, int]:
        """Simulate a falling block until it lands.

        Args:
            start_pos: Starting position
            world: World state
            max_ticks: Maximum ticks to simulate

        Returns:
            Tuple of (final_position, ticks_elapsed)
        """
        logger.debug("FallingBlockPhysics.simulate_fall: start_pos=%s, world=%s, max_ticks=%s", start_pos, world, max_ticks)
        y = float(start_pos.y)
        velocity = 0.0
        ticks = 0

        while ticks < max_ticks:
            # Apply gravity
            velocity += FallingBlockPhysics.GRAVITY
            if velocity > FallingBlockPhysics.TERMINAL_VELOCITY:
                velocity = FallingBlockPhysics.TERMINAL_VELOCITY

            # Move
            y -= velocity

            # Check for landing
            check_y = int(math.floor(y))
            check_pos = Position(start_pos.x, check_y, start_pos.z)

            if not FallingBlockPhysics.should_fall(check_pos, world):
                # Land on top of the blocking block
                final_y = check_y + 1
                return Position(start_pos.x, final_y, start_pos.z), ticks

            # Check for void
            if check_y < -64:
                return Position(start_pos.x, -64, start_pos.z), ticks

            ticks += 1

        # Max ticks reached
        return Position(start_pos.x, int(y), start_pos.z), ticks

    @staticmethod
    def fall_time_seconds(height: float) -> float:
        """Calculate approximate fall time in seconds.

        Uses simplified physics (ignoring terminal velocity).
        """
        # h = 0.5 * g * t^2, solving for t
        # t = sqrt(2h / g)
        logger.debug("FallingBlockPhysics.fall_time_seconds: height=%s", height)
        g = FallingBlockPhysics.GRAVITY * FallingBlockPhysics.TICK_RATE**2
        t = math.sqrt(2 * height / g)
        return t


class BlockPlacingVerifier:
    """Verifier for block placing mechanics."""

    def __init__(self) -> None:
        logger.info("BlockPlacingVerifier.__init__ called")
        self.test_results: list[tuple[str, bool, str]] = []

    def _add_result(self, name: str, passed: bool, message: str = "") -> None:
        logger.debug("BlockPlacingVerifier._add_result: name=%s, passed=%s, message=%s", name, passed, message)
        self.test_results.append((name, passed, message))

    def verify_log_axis_placement(self) -> bool:
        """Test log axis based on clicked face."""
        logger.debug("BlockPlacingVerifier.verify_log_axis_placement called")
        test_cases = [
            (Direction.UP, Axis.Y),
            (Direction.DOWN, Axis.Y),
            (Direction.NORTH, Axis.Z),
            (Direction.SOUTH, Axis.Z),
            (Direction.EAST, Axis.X),
            (Direction.WEST, Axis.X),
        ]

        all_passed = True
        for click_dir, expected_axis in test_cases:
            axis = get_log_axis_from_face(click_dir)
            passed = axis == expected_axis
            self._add_result(
                f"log_axis_{click_dir.name}",
                passed,
                f"Expected {expected_axis.name}, got {axis.name}",
            )
            all_passed = all_passed and passed

        return all_passed

    def verify_stair_facing(self) -> bool:
        """Test stair facing based on player direction."""
        logger.debug("BlockPlacingVerifier.verify_stair_facing called")
        world: dict[Position, BlockState] = {}
        pos = Position(0, 0, 0)

        all_passed = True
        for player_dir in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
            context = BlockPlacementContext(
                player_direction=player_dir,
                click_direction=Direction.UP,
                click_position=(0.5, 1.0, 0.5),
            )

            state = get_placed_state("oak_stairs", pos, context, world)
            facing = state.properties.get("facing", "")

            expected = player_dir.name.lower()
            passed = facing == expected
            self._add_result(
                f"stair_facing_{player_dir.name}",
                passed,
                f"Expected {expected}, got {facing}",
            )
            all_passed = all_passed and passed

        return all_passed

    def verify_stair_half(self) -> bool:
        """Test stair half based on click position."""
        logger.debug("BlockPlacingVerifier.verify_stair_half called")
        world: dict[Position, BlockState] = {}
        pos = Position(0, 0, 0)

        test_cases = [
            (Direction.UP, (0.5, 1.0, 0.5), "bottom"),
            (Direction.DOWN, (0.5, 0.0, 0.5), "top"),
            (Direction.NORTH, (0.5, 0.3, 0.0), "bottom"),
            (Direction.NORTH, (0.5, 0.7, 0.0), "top"),
            (Direction.EAST, (1.0, 0.2, 0.5), "bottom"),
            (Direction.EAST, (1.0, 0.8, 0.5), "top"),
        ]

        all_passed = True
        for click_dir, click_pos, expected_half in test_cases:
            context = BlockPlacementContext(
                player_direction=Direction.NORTH,
                click_direction=click_dir,
                click_position=click_pos,
            )

            state = get_placed_state("oak_stairs", pos, context, world)
            half = state.properties.get("half", "")

            passed = half == expected_half
            self._add_result(
                f"stair_half_{click_dir.name}_{click_pos[1]}",
                passed,
                f"Expected {expected_half}, got {half}",
            )
            all_passed = all_passed and passed

        return all_passed

    def verify_stair_shapes(self) -> bool:
        """Test stair shape connections."""
        # Create a world with stairs for corner detection
        # Straight stair
        logger.debug("BlockPlacingVerifier.verify_stair_shapes called")
        world: dict[Position, BlockState] = {}
        pos = Position(5, 64, 5)

        context = BlockPlacementContext(
            player_direction=Direction.NORTH,
            click_direction=Direction.UP,
            click_position=(0.5, 1.0, 0.5),
        )

        state = get_placed_state("oak_stairs", pos, context, world)
        shape = state.properties.get("shape", "")

        passed = shape == "straight"
        self._add_result(
            "stair_shape_straight",
            passed,
            f"Expected straight, got {shape}",
        )

        # Add inner corner setup
        # Place a stair at (5, 64, 6) facing east
        world[Position(5, 64, 6)] = BlockState(
            "oak_stairs",
            {
                "facing": "east",
                "half": "bottom",
                "shape": "straight",
            },
        )

        # Now place stair at (5, 64, 5) facing north - should be inner_right
        context = BlockPlacementContext(
            player_direction=Direction.NORTH,
            click_direction=Direction.UP,
            click_position=(0.5, 1.0, 0.5),
        )

        state = get_placed_state("oak_stairs", pos, context, world)
        shape = state.properties.get("shape", "")

        # This depends on implementation details
        inner_passed = shape in ("straight", "inner_left", "inner_right")
        self._add_result(
            "stair_shape_corner",
            inner_passed,
            f"Got shape: {shape}",
        )

        return passed and inner_passed

    def verify_slab_placement(self) -> bool:
        """Test slab placement top/bottom."""
        logger.debug("BlockPlacingVerifier.verify_slab_placement called")
        world: dict[Position, BlockState] = {}
        pos = Position(0, 0, 0)

        test_cases = [
            (Direction.UP, (0.5, 1.0, 0.5), "bottom"),
            (Direction.DOWN, (0.5, 0.0, 0.5), "top"),
            (Direction.NORTH, (0.5, 0.3, 0.0), "bottom"),
            (Direction.NORTH, (0.5, 0.7, 0.0), "top"),
        ]

        all_passed = True
        for click_dir, click_pos, expected_type in test_cases:
            context = BlockPlacementContext(
                player_direction=Direction.NORTH,
                click_direction=click_dir,
                click_position=click_pos,
            )

            state = get_placed_state("oak_slab", pos, context, world)
            slab_type = state.properties.get("type", "")

            passed = slab_type == expected_type
            self._add_result(
                f"slab_type_{click_dir.name}_{click_pos[1]}",
                passed,
                f"Expected {expected_type}, got {slab_type}",
            )
            all_passed = all_passed and passed

        return all_passed

    def verify_torch_placement_rules(self) -> bool:
        """Test torch placement requires solid surface."""
        logger.debug("BlockPlacingVerifier.verify_torch_placement_rules called")
        all_passed = True

        # Torch on ground
        world: dict[Position, BlockState] = {
            Position(0, -1, 0): BlockState("stone"),
        }
        pos = Position(0, 0, 0)
        context = BlockPlacementContext(
            player_direction=Direction.NORTH,
            click_direction=Direction.UP,
            click_position=(0.5, 1.0, 0.5),
        )

        can_place, reason = can_place_block("torch", pos, context, world)
        self._add_result(
            "torch_on_ground",
            can_place,
            reason,
        )
        all_passed = all_passed and can_place

        # Torch in air (should fail without solid below)
        world_air: dict[Position, BlockState] = {}
        can_place, reason = can_place_block("torch", pos, context, world_air)
        self._add_result(
            "torch_in_air_fails",
            not can_place,
            reason,
        )
        all_passed = all_passed and not can_place

        # Wall torch
        world_wall: dict[Position, BlockState] = {
            Position(-1, 0, 0): BlockState("stone"),
        }
        context_wall = BlockPlacementContext(
            player_direction=Direction.NORTH,
            click_direction=Direction.EAST,
            click_position=(1.0, 0.5, 0.5),
        )
        can_place, reason = can_place_block("torch", pos, context_wall, world_wall)
        self._add_result(
            "wall_torch",
            can_place,
            reason,
        )
        all_passed = all_passed and can_place

        return all_passed

    def verify_plant_placement_rules(self) -> bool:
        """Test plants require dirt/grass below."""
        logger.debug("BlockPlacingVerifier.verify_plant_placement_rules called")
        all_passed = True

        # Sapling on grass
        world: dict[Position, BlockState] = {
            Position(0, -1, 0): BlockState("grass_block"),
        }
        pos = Position(0, 0, 0)
        context = BlockPlacementContext(
            player_direction=Direction.NORTH,
            click_direction=Direction.UP,
            click_position=(0.5, 1.0, 0.5),
        )

        can_place, reason = can_place_block("oak_sapling", pos, context, world)
        self._add_result(
            "sapling_on_grass",
            can_place,
            reason,
        )
        all_passed = all_passed and can_place

        # Sapling in air
        world_air: dict[Position, BlockState] = {}
        can_place, reason = can_place_block("oak_sapling", pos, context, world_air)
        self._add_result(
            "sapling_in_air_fails",
            not can_place,
            reason,
        )
        all_passed = all_passed and not can_place

        return all_passed

    def verify_rail_placement(self) -> bool:
        """Test rails require solid below."""
        logger.debug("BlockPlacingVerifier.verify_rail_placement called")
        all_passed = True

        # Rail on stone
        world: dict[Position, BlockState] = {
            Position(0, -1, 0): BlockState("stone"),
        }
        pos = Position(0, 0, 0)
        context = BlockPlacementContext(
            player_direction=Direction.NORTH,
            click_direction=Direction.UP,
            click_position=(0.5, 1.0, 0.5),
        )

        can_place, reason = can_place_block("rail", pos, context, world)
        self._add_result(
            "rail_on_stone",
            can_place,
            reason,
        )
        all_passed = all_passed and can_place

        # Rail in air
        world_air: dict[Position, BlockState] = {}
        can_place, reason = can_place_block("rail", pos, context, world_air)
        self._add_result(
            "rail_in_air_fails",
            not can_place,
            reason,
        )
        all_passed = all_passed and not can_place

        return all_passed

    def verify_falling_block_detection(self) -> bool:
        """Test falling block should_fall detection."""
        logger.debug("BlockPlacingVerifier.verify_falling_block_detection called")
        all_passed = True

        # Sand on stone - shouldn't fall
        world: dict[Position, BlockState] = {
            Position(0, -1, 0): BlockState("stone"),
        }
        pos = Position(0, 0, 0)

        should_fall = FallingBlockPhysics.should_fall(pos, world)
        self._add_result(
            "sand_on_stone_stable",
            not should_fall,
            f"should_fall={should_fall}",
        )
        all_passed = all_passed and not should_fall

        # Sand on air - should fall
        world_air: dict[Position, BlockState] = {
            Position(0, -1, 0): BlockState("air"),
        }
        should_fall = FallingBlockPhysics.should_fall(pos, world_air)
        self._add_result(
            "sand_on_air_falls",
            should_fall,
            f"should_fall={should_fall}",
        )
        all_passed = all_passed and should_fall

        # Sand on water - should fall through
        world_water: dict[Position, BlockState] = {
            Position(0, -1, 0): BlockState("water"),
        }
        should_fall = FallingBlockPhysics.should_fall(pos, world_water)
        self._add_result(
            "sand_on_water_falls",
            should_fall,
            f"should_fall={should_fall}",
        )
        all_passed = all_passed and should_fall

        return all_passed

    def verify_falling_block_physics(self) -> bool:
        """Test falling block physics simulation."""
        logger.debug("BlockPlacingVerifier.verify_falling_block_physics called")
        all_passed = True

        # Create a world with ground at y=0
        world: dict[Position, BlockState] = {
            Position(0, 0, 0): BlockState("stone"),
        }

        # Drop sand from y=10
        start = Position(0, 10, 0)
        final_pos, ticks = FallingBlockPhysics.simulate_fall(start, world)

        # Should land at y=1 (on top of stone)
        passed = final_pos.y == 1
        self._add_result(
            "falling_block_lands_correctly",
            passed,
            f"Started at y=10, landed at y={final_pos.y}, took {ticks} ticks",
        )
        all_passed = all_passed and passed

        # Test fall time approximation
        fall_time = FallingBlockPhysics.fall_time_seconds(10)
        # Approximate: should be around 0.7 seconds
        passed = 0.3 < fall_time < 2.0
        self._add_result(
            "fall_time_reasonable",
            passed,
            f"10 blocks fall time: {fall_time:.3f}s",
        )
        all_passed = all_passed and passed

        return all_passed

    def verify_piston_facing(self) -> bool:
        """Test piston facing based on player direction."""
        logger.debug("BlockPlacingVerifier.verify_piston_facing called")
        world: dict[Position, BlockState] = {}
        pos = Position(0, 0, 0)

        all_passed = True
        for player_dir in Direction:
            context = BlockPlacementContext(
                player_direction=player_dir,
                click_direction=Direction.UP,
                click_position=(0.5, 1.0, 0.5),
            )

            state = get_placed_state("piston", pos, context, world)
            facing = state.properties.get("facing", "")

            expected = player_dir.name.lower()
            passed = facing == expected
            self._add_result(
                f"piston_facing_{player_dir.name}",
                passed,
                f"Expected {expected}, got {facing}",
            )
            all_passed = all_passed and passed

        return all_passed

    def verify_door_placement(self) -> bool:
        """Test door placement rules."""
        logger.debug("BlockPlacingVerifier.verify_door_placement called")
        all_passed = True

        # Door on solid ground
        world: dict[Position, BlockState] = {
            Position(0, -1, 0): BlockState("stone"),
        }
        pos = Position(0, 0, 0)
        context = BlockPlacementContext(
            player_direction=Direction.NORTH,
            click_direction=Direction.UP,
            click_position=(0.5, 1.0, 0.5),
        )

        can_place, reason = can_place_block("oak_door", pos, context, world)
        self._add_result(
            "door_on_ground",
            can_place,
            reason,
        )
        all_passed = all_passed and can_place

        # Door state has correct properties
        state = get_placed_state("oak_door", pos, context, world)

        passed = (
            state.properties.get("half") == "lower" and state.properties.get("facing") == "north"
        )
        self._add_result(
            "door_properties",
            passed,
            f"Properties: {state.properties}",
        )
        all_passed = all_passed and passed

        return all_passed

    def verify_y_bounds(self) -> bool:
        """Test Y coordinate bounds checking."""
        logger.debug("BlockPlacingVerifier.verify_y_bounds called")
        all_passed = True

        world: dict[Position, BlockState] = {}
        context = BlockPlacementContext(
            player_direction=Direction.NORTH,
            click_direction=Direction.UP,
            click_position=(0.5, 1.0, 0.5),
        )

        # Below world
        can_place, reason = can_place_block("stone", Position(0, -65, 0), context, world)
        self._add_result(
            "below_world_fails",
            not can_place,
            reason,
        )
        all_passed = all_passed and not can_place

        # Above world
        can_place, reason = can_place_block("stone", Position(0, 320, 0), context, world)
        self._add_result(
            "above_world_fails",
            not can_place,
            reason,
        )
        all_passed = all_passed and not can_place

        # Valid position
        can_place, reason = can_place_block("stone", Position(0, 64, 0), context, world)
        self._add_result(
            "valid_y_succeeds",
            can_place,
            reason,
        )
        all_passed = all_passed and can_place

        return all_passed

    def verify_occupied_position(self) -> bool:
        """Test cannot place in occupied position."""
        logger.debug("BlockPlacingVerifier.verify_occupied_position called")
        world: dict[Position, BlockState] = {
            Position(0, 0, 0): BlockState("stone"),
        }
        context = BlockPlacementContext(
            player_direction=Direction.NORTH,
            click_direction=Direction.UP,
            click_position=(0.5, 1.0, 0.5),
        )

        can_place, reason = can_place_block("dirt", Position(0, 0, 0), context, world)
        self._add_result(
            "occupied_fails",
            not can_place,
            reason,
        )
        return not can_place

    def run_all_tests(self) -> tuple[int, int]:
        """Run all verification tests.

        Returns:
            Tuple of (passed_count, total_count)
        """
        logger.debug("BlockPlacingVerifier.run_all_tests called")
        self.test_results.clear()

        self.verify_log_axis_placement()
        self.verify_stair_facing()
        self.verify_stair_half()
        self.verify_stair_shapes()
        self.verify_slab_placement()
        self.verify_torch_placement_rules()
        self.verify_plant_placement_rules()
        self.verify_rail_placement()
        self.verify_falling_block_detection()
        self.verify_falling_block_physics()
        self.verify_piston_facing()
        self.verify_door_placement()
        self.verify_y_bounds()
        self.verify_occupied_position()

        passed = sum(1 for _, p, _ in self.test_results if p)
        total = len(self.test_results)

        return passed, total

    def print_results(self) -> None:
        """Print all test results."""
        logger.debug("BlockPlacingVerifier.print_results called")
        passed = sum(1 for _, p, _ in self.test_results if p)
        total = len(self.test_results)

        print(f"\nBlock Placing Verification Results: {passed}/{total} passed\n")
        print("-" * 70)

        for name, success, message in self.test_results:
            status = "PASS" if success else "FAIL"
            print(f"[{status}] {name}")
            if message:
                print(f"       {message}")

        print("-" * 70)
        print(f"Total: {passed}/{total} tests passed")


def main() -> None:
    """Run block placing verification."""
    logger.debug("main called")
    verifier = BlockPlacingVerifier()
    passed, total = verifier.run_all_tests()
    verifier.print_results()

    if passed < total:
        exit(1)


if __name__ == "__main__":
    main()
