"""Minecraft 1.8.9 physics constants - authoritative reference values.

import logging

logger = logging.getLogger(__name__)

These values are decompiled from vanilla MC 1.8.9 Java Edition.
All physics calculations must match these constants exactly.

Movement tick order (per MC tick = 50ms):
1. Apply input -> acceleration
2. Apply gravity (if not on ground)
3. Apply velocity -> position
4. Apply drag (air or ground based on collision state)

The drag application happens AFTER position update, which is critical
for matching exact MC behavior.
"""

# =============================================================================
# GRAVITY AND DRAG
# =============================================================================

# Gravity: applied per tick when airborne (blocks/tick^2)
# In MC, this is subtracted from Y velocity each tick
GRAVITY: float = 0.08

# Air drag: velocity multiplier applied each tick when airborne
# Applied as: vel *= DRAG_AIR (where DRAG_AIR = 0.98)
# Some shaders use (1 - 0.02) = 0.98
DRAG_AIR: float = 0.98

# Ground drag: velocity multiplier for horizontal movement on ground
# Applied as: vel_xz *= slipperiness * 0.91
# Default block slipperiness = 0.6, so effective drag = 0.6 * 0.91 = 0.546
# However for consistency with GPU shaders, we use the simplified value
DRAG_GROUND: float = 0.6

# =============================================================================
# MOVEMENT SPEEDS
# =============================================================================

# Base walking speed (blocks/tick)
WALK_SPEED: float = 0.1

# Sprint multiplier: applied when sprinting
SPRINT_MULTIPLIER: float = 1.3

# Sneak multiplier: applied when sneaking
SNEAK_MULTIPLIER: float = 0.3

# Jump initial velocity (blocks/tick upward)
JUMP_VELOCITY: float = 0.42

# =============================================================================
# BLOCK SLIPPERINESS
# =============================================================================

# Default block slipperiness (most blocks)
SLIPPERINESS_DEFAULT: float = 0.6

# Ice block slipperiness
SLIPPERINESS_ICE: float = 0.98

# Packed ice slipperiness (same as regular ice in 1.8.9)
SLIPPERINESS_PACKED_ICE: float = 0.98

# Slime block slipperiness
SLIPPERINESS_SLIME: float = 0.8

# =============================================================================
# PLAYER DIMENSIONS
# =============================================================================

# Player width (hitbox is a square centered on position)
PLAYER_WIDTH: float = 0.6

# Player height (standing)
PLAYER_HEIGHT: float = 1.8

# Eye height from feet position
PLAYER_EYE_HEIGHT: float = 1.62

# Player height when sneaking
PLAYER_SNEAK_HEIGHT: float = 1.5

# Eye height when sneaking
PLAYER_SNEAK_EYE_HEIGHT: float = 1.27

# =============================================================================
# DERIVED CONSTANTS (for convenience)
# =============================================================================

# Sprint speed = WALK_SPEED * SPRINT_MULTIPLIER
SPRINT_SPEED: float = WALK_SPEED * SPRINT_MULTIPLIER  # 0.13

# Air acceleration (reduced when airborne)
AIR_ACCELERATION: float = 0.02

# Ground acceleration
GROUND_ACCELERATION: float = 0.1

# Maximum fall speed (terminal velocity)
TERMINAL_VELOCITY: float = 3.92  # Approximately -GRAVITY / (1 - DRAG_AIR) per tick

# =============================================================================
# PHYSICS TICK FUNCTIONS
# =============================================================================


def tick_fall_velocity(vy: float) -> float:
    """Apply one tick of falling physics to Y velocity.

    Order: gravity first, then drag.

    Args:
        vy: Current Y velocity (negative = falling)

    Returns:
        New Y velocity after one tick
    """
    logger.debug("tick_fall_velocity: vy=%s", vy)
    vy -= GRAVITY  # Apply gravity
    vy *= DRAG_AIR  # Apply air drag
    return vy


def tick_fall_position(y: float, vy: float) -> tuple[float, float]:
    """Apply one tick of falling physics to position and velocity.

    Args:
        y: Current Y position
        vy: Current Y velocity

    Returns:
        Tuple of (new_y, new_vy) after one tick
    """
    logger.debug("tick_fall_position: y=%s, vy=%s", y, vy)
    vy -= GRAVITY
    vy *= DRAG_AIR
    y += vy
    return y, vy


def simulate_fall(y0: float, vy0: float, ticks: int) -> tuple[float, float]:
    """Simulate N ticks of free fall.

    Args:
        y0: Initial Y position
        vy0: Initial Y velocity
        ticks: Number of ticks to simulate

    Returns:
        Tuple of (final_y, final_vy)
    """
    logger.debug("simulate_fall: y0=%s, vy0=%s, ticks=%s", y0, vy0, ticks)
    y, vy = y0, vy0
    for _ in range(ticks):
        y, vy = tick_fall_position(y, vy)
    return y, vy


def simulate_jump(y0: float, ticks: int) -> tuple[float, float]:
    """Simulate a jump from standing position.

    Args:
        y0: Starting Y position (on ground)
        ticks: Number of ticks to simulate

    Returns:
        Tuple of (final_y, final_vy)
    """
    logger.debug("simulate_jump: y0=%s, ticks=%s", y0, ticks)
    return simulate_fall(y0, JUMP_VELOCITY, ticks)


def find_jump_apex_tick() -> tuple[int, float]:
    """Find the tick at which jump apex is reached.

    Returns:
        Tuple of (apex_tick, apex_height) relative to starting position
    """
    logger.debug("find_jump_apex_tick called")
    y, vy = 0.0, JUMP_VELOCITY
    max_y = 0.0
    apex_tick = 0

    for tick in range(100):  # Jump shouldn't last more than 100 ticks
        y, vy = tick_fall_position(y, vy)
        if y > max_y:
            max_y = y
            apex_tick = tick + 1
        elif y < max_y - 0.001:  # Started descending
            break

    return apex_tick, max_y
