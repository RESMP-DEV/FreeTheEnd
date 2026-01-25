"""Stronghold triangulation from Eye of Ender throws.

Implements geometric triangulation to locate strongholds using two or more
Eye of Ender throws. The algorithm finds the intersection of lines projected
from throw positions in the direction the eyes traveled.

Usage:
    >>> from minecraft_sim.triangulation import triangulate_stronghold, TriangulationState
    >>> state = TriangulationState()
    >>> # First throw at position (100, 200), eye flew toward direction (0.8, 0.6)
    >>> state.add_throw((100.0, 200.0), (0.8, 0.6))
    >>> # Move perpendicular, second throw
    >>> state.add_throw((300.0, 150.0), (0.3, 0.95))
    >>> if state.is_complete:
    ...     pos = state.estimated_position
    ...     print(f"Stronghold at ({pos[0]:.1f}, {pos[1]:.1f})")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def triangulate_stronghold(
    pos1: tuple[float, float],
    dir1: tuple[float, float],
    pos2: tuple[float, float],
    dir2: tuple[float, float],
) -> tuple[float, float]:
    """Calculate stronghold position from two eye throws.

    Uses parametric line intersection to find where two rays meet.
    Each ray is defined by a throw position and the normalized direction
    the Eye of Ender traveled.

    Args:
        pos1: (x, z) coordinates of first throw position.
        dir1: Normalized (dx, dz) direction vector of first eye's travel.
        pos2: (x, z) coordinates of second throw position.
        dir2: Normalized (dx, dz) direction vector of second eye's travel.

    Returns:
        (x, z) estimated stronghold coordinates.

    Raises:
        ValueError: If lines are parallel (no intersection) or directions
            are degenerate (zero-length).

    Note:
        For best accuracy, throws should be made with significant perpendicular
        distance (at least 100 blocks apart) and from different angles relative
        to the stronghold.

    Example:
        >>> pos = triangulate_stronghold(
        ...     pos1=(0.0, 0.0), dir1=(1.0, 0.0),  # East
        ...     pos2=(100.0, 0.0), dir2=(0.0, 1.0)  # North
        ... )
        >>> pos
        (100.0, 0.0)
    """
    x1, z1 = pos1
    dx1, dz1 = dir1
    x2, z2 = pos2
    dx2, dz2 = dir2

    # Validate directions aren't degenerate
    len1_sq = dx1 * dx1 + dz1 * dz1
    len2_sq = dx2 * dx2 + dz2 * dz2
    if len1_sq < 1e-10 or len2_sq < 1e-10:
        raise ValueError("Direction vectors must be non-zero")

    # Parametric form: P1 + t1 * D1 = P2 + t2 * D2
    # Solving for t1:
    # x1 + t1*dx1 = x2 + t2*dx2
    # z1 + t1*dz1 = z2 + t2*dz2
    #
    # Rearranged as matrix:
    # [dx1  -dx2] [t1]   [x2 - x1]
    # [dz1  -dz2] [t2] = [z2 - z1]
    #
    # Using Cramer's rule:
    # det = dx1*(-dz2) - (-dx2)*dz1 = -dx1*dz2 + dx2*dz1 = dx2*dz1 - dx1*dz2

    det = dx2 * dz1 - dx1 * dz2

    if abs(det) < 1e-10:
        raise ValueError("Lines are parallel or nearly parallel (no intersection)")

    # t1 = ((x2-x1)*(-dz2) - (-dx2)*(z2-z1)) / det
    #    = (-(x2-x1)*dz2 + dx2*(z2-z1)) / det
    #    = (dx2*(z2-z1) - dz2*(x2-x1)) / det

    t1 = (dx2 * (z2 - z1) - dz2 * (x2 - x1)) / det

    # Intersection point
    x = x1 + t1 * dx1
    z = z1 + t1 * dz1

    return (x, z)


def _normalize(dx: float, dz: float) -> tuple[float, float]:
    """Normalize a 2D vector."""
    length = math.sqrt(dx * dx + dz * dz)
    if length < 1e-10:
        return (0.0, 0.0)
    return (dx / length, dz / length)


def direction_from_yaw(yaw: float) -> tuple[float, float]:
    """Convert Minecraft yaw to a normalized direction vector.

    Minecraft yaw: 0 = south (+Z), 90 = west (-X), 180 = north (-Z), 270 = east (+X)

    Args:
        yaw: Minecraft yaw angle in degrees.

    Returns:
        Normalized (dx, dz) direction vector.
    """
    # Convert to radians and adjust for Minecraft coordinate system
    rad = math.radians(yaw)
    # In MC: yaw 0 = +Z (south), increases clockwise when viewed from above
    dx = -math.sin(rad)
    dz = math.cos(rad)
    return (dx, dz)


@dataclass
class EyeThrow:
    """Record of a single Eye of Ender throw."""

    position: tuple[float, float]
    """(x, z) world coordinates where the throw was made."""

    direction: tuple[float, float]
    """Normalized (dx, dz) direction the eye traveled."""

    yaw: float = 0.0
    """Original yaw angle (for debugging/display)."""

    @classmethod
    def from_yaw(
        cls,
        x: float,
        z: float,
        yaw: float,
    ) -> EyeThrow:
        """Create from position and yaw angle.

        Args:
            x: World X coordinate.
            z: World Z coordinate.
            yaw: Minecraft yaw in degrees (direction player was facing when eye flew).

        Returns:
            EyeThrow instance.
        """
        return cls(
            position=(x, z),
            direction=direction_from_yaw(yaw),
            yaw=yaw,
        )


@dataclass
class TriangulationState:
    """Stateful tracker for stronghold triangulation.

    Accumulates eye throws and computes intersection when sufficient
    data is available. Designed to integrate with observation/info dicts.

    Example:
        >>> state = TriangulationState()
        >>> state.add_throw((0.0, 0.0), (0.707, 0.707))
        >>> state.is_complete
        False
        >>> state.add_throw((100.0, 0.0), (0.0, 1.0))
        >>> state.is_complete
        True
        >>> state.estimated_position
        (100.0, 100.0)
    """

    throws: list[EyeThrow] = field(default_factory=list)
    """List of recorded eye throws."""

    _cached_position: tuple[float, float] | None = field(default=None, repr=False)
    """Cached triangulation result."""

    _error_estimate: float = field(default=float("inf"), repr=False)
    """Estimated error in blocks (lower is better)."""

    def add_throw(
        self,
        position: tuple[float, float],
        direction: tuple[float, float],
        yaw: float = 0.0,
    ) -> None:
        """Record a new eye throw.

        Args:
            position: (x, z) world coordinates of throw.
            direction: Normalized (dx, dz) direction eye traveled.
            yaw: Optional yaw angle for reference.
        """
        # Normalize direction
        dx, dz = direction
        direction = _normalize(dx, dz)

        self.throws.append(EyeThrow(position=position, direction=direction, yaw=yaw))
        self._cached_position = None  # Invalidate cache

    def add_throw_from_yaw(self, x: float, z: float, yaw: float) -> None:
        """Record throw using yaw angle instead of direction vector.

        Args:
            x: World X coordinate.
            z: World Z coordinate.
            yaw: Minecraft yaw angle in degrees.
        """
        throw = EyeThrow.from_yaw(x, z, yaw)
        self.throws.append(throw)
        self._cached_position = None

    def clear(self) -> None:
        """Reset all throws and cached state."""
        self.throws.clear()
        self._cached_position = None
        self._error_estimate = float("inf")

    @property
    def num_throws(self) -> int:
        """Number of recorded throws."""
        return len(self.throws)

    @property
    def is_complete(self) -> bool:
        """True if we have enough throws for triangulation (2+)."""
        return len(self.throws) >= 2

    @property
    def estimated_position(self) -> tuple[float, float] | None:
        """Estimated stronghold (x, z) or None if insufficient data.

        With 2 throws, returns simple intersection.
        With 3+ throws, returns least-squares best fit.
        """
        if not self.is_complete:
            return None

        if self._cached_position is not None:
            return self._cached_position

        if len(self.throws) == 2:
            # Simple two-line intersection
            try:
                self._cached_position = triangulate_stronghold(
                    self.throws[0].position,
                    self.throws[0].direction,
                    self.throws[1].position,
                    self.throws[1].direction,
                )
                self._error_estimate = self._compute_error()
            except ValueError:
                return None
        else:
            # Multiple throws: use all pairs and average
            self._cached_position = self._multi_throw_estimate()
            self._error_estimate = self._compute_error()

        return self._cached_position

    @property
    def error_estimate(self) -> float:
        """Estimated error in blocks (requires 3+ throws)."""
        if self._cached_position is None:
            _ = self.estimated_position  # Compute if needed
        return self._error_estimate

    def _multi_throw_estimate(self) -> tuple[float, float]:
        """Compute best estimate from 3+ throws using all pairs."""
        # Collect all pairwise intersections
        intersections: list[tuple[float, float]] = []
        weights: list[float] = []

        for i, t1 in enumerate(self.throws):
            for j, t2 in enumerate(self.throws):
                if j <= i:
                    continue
                try:
                    point = triangulate_stronghold(
                        t1.position,
                        t1.direction,
                        t2.position,
                        t2.direction,
                    )
                    # Weight by angle difference (more perpendicular = better)
                    dot = abs(t1.direction[0] * t2.direction[0] + t1.direction[1] * t2.direction[1])
                    # dot=0 means perpendicular (best), dot=1 means parallel (worst)
                    weight = 1.0 - dot + 0.01  # Small epsilon to avoid zero weight
                    intersections.append(point)
                    weights.append(weight)
                except ValueError:
                    # Parallel lines, skip
                    continue

        if not intersections:
            # All pairs parallel, fall back to first two
            return triangulate_stronghold(
                self.throws[0].position,
                self.throws[0].direction,
                self.throws[1].position,
                self.throws[1].direction,
            )

        # Weighted average
        total_weight = sum(weights)
        x = sum(p[0] * w for p, w in zip(intersections, weights)) / total_weight
        z = sum(p[1] * w for p, w in zip(intersections, weights)) / total_weight

        return (x, z)

    def _compute_error(self) -> float:
        """Estimate error based on spread of pairwise intersections."""
        if len(self.throws) < 3 or self._cached_position is None:
            return float("inf")

        # Collect all pairwise intersections
        intersections: list[tuple[float, float]] = []
        for i, t1 in enumerate(self.throws):
            for j, t2 in enumerate(self.throws):
                if j <= i:
                    continue
                try:
                    point = triangulate_stronghold(
                        t1.position,
                        t1.direction,
                        t2.position,
                        t2.direction,
                    )
                    intersections.append(point)
                except ValueError:
                    continue

        if len(intersections) < 2:
            return float("inf")

        # Compute RMS distance from estimated position
        ex, ez = self._cached_position
        distances = [math.sqrt((p[0] - ex) ** 2 + (p[1] - ez) ** 2) for p in intersections]
        return math.sqrt(sum(d * d for d in distances) / len(distances))

    def to_observation_dict(self) -> dict[str, bool | tuple[float, float] | None]:
        """Return observation fields for RL integration.

        Returns:
            Dictionary with:
            - triangulation_complete: bool
            - estimated_stronghold_pos: (x, z) or None
        """
        return {
            "triangulation_complete": self.is_complete,
            "estimated_stronghold_pos": self.estimated_position,
        }


# Observation extension fields for MinecraftObservation
TRIANGULATION_OBS_FIELDS = {
    "triangulation_complete": bool,
    "estimated_stronghold_pos": "tuple[float, float] | None",
}
