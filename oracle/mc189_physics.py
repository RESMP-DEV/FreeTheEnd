"""MC 1.8.9 physics reference implementations.

Pure Python implementations for gravity, drag, and collision physics
that match the decompiled MC 1.8.9 behavior exactly.

Note: The GPU shaders in cpp/shaders/ sometimes use different units:
- Shaders often use blocks/sec² for gravity (32.0) with deltaTime conversion
- Reference uses blocks/tick² (0.08) which is 32.0 / 400 (20 TPS, squared)

When verifying shader output, convert between units appropriately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from .mc189_constants import (
    AIR_ACCELERATION,
    DRAG_AIR,
    DRAG_GROUND,
    GRAVITY,
    GROUND_ACCELERATION,
    JUMP_VELOCITY,
    SLIPPERINESS_DEFAULT,
    TERMINAL_VELOCITY,
)


@dataclass(frozen=True)
class PhysicsConstants:
    """Container for all MC 1.8.9 physics constants.

    Provides a single source of truth for physics values,
    useful for dependency injection in tests.
    """

    gravity: float = GRAVITY
    drag_air: float = DRAG_AIR
    drag_ground: float = DRAG_GROUND
    jump_velocity: float = JUMP_VELOCITY
    terminal_velocity: float = TERMINAL_VELOCITY
    air_acceleration: float = AIR_ACCELERATION
    ground_acceleration: float = GROUND_ACCELERATION
    slipperiness_default: float = SLIPPERINESS_DEFAULT

    @classmethod
    def default(cls) -> PhysicsConstants:
        """Return default MC 1.8.9 physics constants."""
        return cls()


class Vec3(NamedTuple):
    """3D vector for physics calculations."""

    x: float
    y: float
    z: float

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vec3:
        return self.__mul__(scalar)

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self) -> float:
        return self.length_squared() ** 0.5

    def normalize(self) -> Vec3:
        mag = self.length()
        if mag == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / mag, self.y / mag, self.z / mag)

    def xz_length(self) -> float:
        """Length of horizontal (XZ) component only."""
        return (self.x * self.x + self.z * self.z) ** 0.5

    def with_y(self, y: float) -> Vec3:
        """Return new vector with Y component replaced."""
        return Vec3(self.x, y, self.z)


def apply_gravity(
    velocity: Vec3,
    constants: PhysicsConstants | None = None,
) -> Vec3:
    """Apply one tick of gravity to velocity.

    MC applies gravity BEFORE movement, then drag AFTER.

    Args:
        velocity: Current velocity vector
        constants: Physics constants (default: MC 1.8.9 values)

    Returns:
        New velocity with gravity applied
    """
    if constants is None:
        constants = PhysicsConstants.default()

    return Vec3(velocity.x, velocity.y - constants.gravity, velocity.z)


def apply_drag(
    velocity: Vec3,
    on_ground: bool,
    slipperiness: float = SLIPPERINESS_DEFAULT,
    constants: PhysicsConstants | None = None,
) -> Vec3:
    """Apply one tick of drag to velocity.

    In MC 1.8.9:
    - Air drag: all components multiplied by DRAG_AIR
    - Ground drag: XZ multiplied by (slipperiness * 0.91), Y unchanged

    Args:
        velocity: Current velocity vector
        on_ground: Whether entity is on ground
        slipperiness: Block slipperiness (0.6 for most blocks)
        constants: Physics constants

    Returns:
        New velocity with drag applied
    """
    if constants is None:
        constants = PhysicsConstants.default()

    if on_ground:
        # Ground: horizontal drag based on block slipperiness
        xz_drag = slipperiness * 0.91
        return Vec3(velocity.x * xz_drag, velocity.y, velocity.z * xz_drag)
    else:
        # Air: uniform drag on all axes
        return Vec3(
            velocity.x * constants.drag_air,
            velocity.y * constants.drag_air,
            velocity.z * constants.drag_air,
        )


def physics_tick(
    position: Vec3,
    velocity: Vec3,
    on_ground: bool,
    slipperiness: float = SLIPPERINESS_DEFAULT,
    constants: PhysicsConstants | None = None,
) -> tuple[Vec3, Vec3]:
    """Apply one complete physics tick.

    MC tick order:
    1. Apply gravity (if not on ground and not flying)
    2. Move position by velocity
    3. Apply drag

    Args:
        position: Current position
        velocity: Current velocity
        on_ground: Whether entity is on ground
        slipperiness: Block slipperiness under feet
        constants: Physics constants

    Returns:
        Tuple of (new_position, new_velocity)
    """
    if constants is None:
        constants = PhysicsConstants.default()

    # Step 1: Apply gravity if airborne
    if not on_ground:
        velocity = apply_gravity(velocity, constants)

    # Step 2: Apply velocity to position
    position = position + velocity

    # Step 3: Apply drag
    velocity = apply_drag(velocity, on_ground, slipperiness, constants)

    return position, velocity


def simulate_fall(
    y0: float,
    vy0: float,
    ticks: int,
    constants: PhysicsConstants | None = None,
) -> tuple[float, float]:
    """Simulate N ticks of free fall.

    Args:
        y0: Initial Y position
        vy0: Initial Y velocity (positive = up)
        ticks: Number of ticks to simulate
        constants: Physics constants

    Returns:
        Tuple of (final_y, final_vy)
    """
    if constants is None:
        constants = PhysicsConstants.default()

    y, vy = y0, vy0
    for _ in range(ticks):
        # Gravity first
        vy -= constants.gravity
        # Clamp to terminal velocity
        if vy < -constants.terminal_velocity:
            vy = -constants.terminal_velocity
        # Position update
        y += vy
        # Air drag
        vy *= constants.drag_air

    return y, vy


def ticks_to_ground(
    y0: float,
    vy0: float,
    ground_y: float = 0.0,
    max_ticks: int = 1000,
    constants: PhysicsConstants | None = None,
) -> int:
    """Calculate ticks until entity reaches ground level.

    Args:
        y0: Initial Y position
        vy0: Initial Y velocity
        ground_y: Ground level Y coordinate
        max_ticks: Maximum ticks to simulate
        constants: Physics constants

    Returns:
        Number of ticks, or max_ticks if not reached
    """
    if constants is None:
        constants = PhysicsConstants.default()

    y, vy = y0, vy0
    for tick in range(max_ticks):
        if y <= ground_y:
            return tick
        vy -= constants.gravity
        if vy < -constants.terminal_velocity:
            vy = -constants.terminal_velocity
        y += vy
        vy *= constants.drag_air

    return max_ticks


def jump_apex(constants: PhysicsConstants | None = None) -> tuple[int, float]:
    """Calculate jump apex tick and height.

    Args:
        constants: Physics constants

    Returns:
        Tuple of (apex_tick, apex_height_above_start)
    """
    if constants is None:
        constants = PhysicsConstants.default()

    y = 0.0
    vy = constants.jump_velocity
    max_y = 0.0
    apex_tick = 0

    for tick in range(100):
        vy -= constants.gravity
        y += vy
        vy *= constants.drag_air

        if y > max_y:
            max_y = y
            apex_tick = tick + 1
        elif y < max_y - 0.001:
            break

    return apex_tick, max_y
