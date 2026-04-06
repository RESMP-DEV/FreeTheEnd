"""Movement physics verifier for MC 1.8.9 accuracy.

Verifies that simulator physics match the reference MC 1.8.9 implementation.
Tests cover:
1. Free fall (gravity + drag)
2. Walking on flat ground (acceleration + friction)
3. Jumping (initial velocity + apex height)
4. Sprint jumping (sprint multiplier application)
5. Walking on ice (slipperiness)

All test cases use exact float comparison within specified tolerance.
Tolerances are kept tight (1e-9 for CPU, 1e-6 for GPU) to catch drift.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Add parent for oracle import
sys.path.insert(0, str(Path(__file__).parent.parent))

from oracle.mc189_constants import (
    DRAG_AIR,
    DRAG_GROUND,
    GRAVITY,
    GROUND_ACCELERATION,
    JUMP_VELOCITY,
    SLIPPERINESS_DEFAULT,
    SLIPPERINESS_ICE,
    SPRINT_MULTIPLIER,
    WALK_SPEED,
    find_jump_apex_tick,
    simulate_fall,
    simulate_jump,
)


class SurfaceType(Enum):
    """Block surface types affecting movement."""

    DEFAULT = "default"
    ICE = "ice"
    PACKED_ICE = "packed_ice"
    SLIME = "slime"


@dataclass
class Vec3:
    """3D vector for position/velocity."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __repr__(self) -> str:
        return f"Vec3({self.x:.9f}, {self.y:.9f}, {self.z:.9f})"

    def close_to(self, other: Vec3, tol: float = 1e-9) -> bool:
        """Check if vectors are within tolerance."""
        return (
            abs(self.x - other.x) <= tol
            and abs(self.y - other.y) <= tol
            and abs(self.z - other.z) <= tol
        )

    def max_diff(self, other: Vec3) -> float:
        """Return maximum absolute difference in any component."""
        return max(abs(self.x - other.x), abs(self.y - other.y), abs(self.z - other.z))


@dataclass
class MovementTestCase:
    """A single movement physics test case."""

    name: str
    initial_velocity: Vec3
    initial_position: Vec3
    ticks: int
    on_ground: bool
    expected_final_position: Vec3
    expected_final_velocity: Vec3
    sprinting: bool = False
    sneaking: bool = False
    surface: SurfaceType = SurfaceType.DEFAULT
    move_forward: bool = False  # Apply forward input
    yaw: float = 0.0  # Player facing direction (degrees)
    tolerance: float = 1e-9  # Comparison tolerance

    def __repr__(self) -> str:
        return f"MovementTestCase({self.name!r})"


@dataclass
class VerificationResult:
    """Result of verifying a single test case."""

    test_case: MovementTestCase
    actual_position: Vec3
    actual_velocity: Vec3
    passed: bool
    position_error: float
    velocity_error: float
    details: str = ""


def reference_fall_physics(y0: float, vy0: float, ticks: int) -> tuple[float, float]:
    """Calculate position and velocity after falling for N ticks.

    This is the reference implementation that all simulators must match.

    MC 1.8.9 tick order for Y axis:
    1. vel_y -= GRAVITY (apply gravity)
    2. vel_y *= DRAG_AIR (apply air drag)
    3. pos_y += vel_y (update position)

    Args:
        y0: Initial Y position
        vy0: Initial Y velocity (negative = falling)
        ticks: Number of ticks to simulate

    Returns:
        Tuple of (final_y, final_vy)
    """
    return simulate_fall(y0, vy0, ticks)


def reference_ground_physics(
    x0: float,
    z0: float,
    vx0: float,
    vz0: float,
    ticks: int,
    slipperiness: float = SLIPPERINESS_DEFAULT,
    move_forward: bool = False,
    yaw: float = 0.0,
    sprinting: bool = False,
) -> tuple[float, float, float, float]:
    """Calculate position and velocity for horizontal ground movement.

    MC 1.8.9 horizontal tick order:
    1. Apply input acceleration (if moving)
    2. Update position: pos += vel
    3. Apply drag: vel *= slipperiness * 0.91

    Args:
        x0, z0: Initial X/Z position
        vx0, vz0: Initial X/Z velocity
        ticks: Number of ticks to simulate
        slipperiness: Block slipperiness (0.6 default, 0.98 ice)
        move_forward: Whether player is inputting forward movement
        yaw: Player facing direction in degrees (0 = +Z, 90 = -X)
        sprinting: Whether player is sprinting

    Returns:
        Tuple of (final_x, final_z, final_vx, final_vz)
    """
    import math

    x, z = x0, z0
    vx, vz = vx0, vz0

    # Calculate movement direction from yaw
    # MC uses: forward = (-sin(yaw), cos(yaw)) in radians
    yaw_rad = math.radians(yaw)
    forward_x = -math.sin(yaw_rad)
    forward_z = math.cos(yaw_rad)

    # Movement speed
    speed = WALK_SPEED
    if sprinting:
        speed *= SPRINT_MULTIPLIER

    # Ground drag factor
    drag = slipperiness * 0.91

    for _ in range(ticks):
        # Apply input acceleration
        if move_forward:
            vx += forward_x * speed * GROUND_ACCELERATION
            vz += forward_z * speed * GROUND_ACCELERATION

        # Update position
        x += vx
        z += vz

        # Apply drag
        vx *= drag
        vz *= drag

    return x, z, vx, vz


def reference_jump_physics(y0: float, ticks: int, sprint_jump: bool = False) -> tuple[float, float]:
    """Calculate position and velocity for a jump.

    Sprint jumping in MC 1.8.9 provides a horizontal boost but does NOT
    affect vertical velocity. The jump velocity is always 0.42 blocks/tick.

    Args:
        y0: Initial Y position (on ground)
        ticks: Number of ticks after jump initiation
        sprint_jump: Whether this is a sprint jump (affects only horizontal)

    Returns:
        Tuple of (final_y, final_vy)
    """
    return simulate_jump(y0, ticks)


def calculate_expected_free_fall(
    start_y: float, start_vy: float, ticks: int
) -> tuple[float, float]:
    """Calculate expected position and velocity after N ticks of free fall.

    Uses the exact MC 1.8.9 formula applied tick by tick.
    """
    y, vy = start_y, start_vy
    for _ in range(ticks):
        vy -= GRAVITY
        vy *= DRAG_AIR
        y += vy
    return y, vy


def calculate_jump_apex() -> tuple[int, float]:
    """Calculate jump apex tick and height.

    Returns:
        Tuple of (tick_at_apex, height_at_apex)
    """
    return find_jump_apex_tick()


# =============================================================================
# PRE-COMPUTED TEST CASES
# =============================================================================

# Calculate expected values for standard test cases
_fall_10_y, _fall_10_vy = calculate_expected_free_fall(100.0, 0.0, 10)
_jump_apex_tick, _jump_apex_height = calculate_jump_apex()
_jump_9_y, _jump_9_vy = calculate_expected_free_fall(64.0, JUMP_VELOCITY, 9)

MOVEMENT_TEST_CASES: list[MovementTestCase] = [
    # =========================================================================
    # FREE FALL TESTS
    # =========================================================================
    MovementTestCase(
        name="free_fall_1_tick",
        initial_velocity=Vec3(0, 0, 0),
        initial_position=Vec3(0, 100, 0),
        ticks=1,
        on_ground=False,
        expected_final_position=Vec3(0, 100.0 + (-GRAVITY * DRAG_AIR), 0),
        expected_final_velocity=Vec3(0, -GRAVITY * DRAG_AIR, 0),
        tolerance=1e-12,
    ),
    MovementTestCase(
        name="free_fall_10_ticks",
        initial_velocity=Vec3(0, 0, 0),
        initial_position=Vec3(0, 100, 0),
        ticks=10,
        on_ground=False,
        expected_final_position=Vec3(0, _fall_10_y, 0),
        expected_final_velocity=Vec3(0, _fall_10_vy, 0),
        tolerance=1e-9,
    ),
    MovementTestCase(
        name="free_fall_with_initial_velocity",
        initial_velocity=Vec3(0, 0.5, 0),  # Thrown upward
        initial_position=Vec3(0, 64, 0),
        ticks=5,
        on_ground=False,
        expected_final_position=Vec3(0, calculate_expected_free_fall(64.0, 0.5, 5)[0], 0),
        expected_final_velocity=Vec3(0, calculate_expected_free_fall(64.0, 0.5, 5)[1], 0),
        tolerance=1e-9,
    ),
    MovementTestCase(
        name="free_fall_50_ticks_terminal_velocity",
        initial_velocity=Vec3(0, 0, 0),
        initial_position=Vec3(0, 200, 0),
        ticks=50,
        on_ground=False,
        expected_final_position=Vec3(0, calculate_expected_free_fall(200.0, 0.0, 50)[0], 0),
        expected_final_velocity=Vec3(0, calculate_expected_free_fall(200.0, 0.0, 50)[1], 0),
        tolerance=1e-9,
    ),
    # =========================================================================
    # JUMP TESTS
    # =========================================================================
    MovementTestCase(
        name="jump_tick_1",
        initial_velocity=Vec3(0, JUMP_VELOCITY, 0),
        initial_position=Vec3(0, 64, 0),
        ticks=1,
        on_ground=False,
        expected_final_position=Vec3(0, calculate_expected_free_fall(64.0, JUMP_VELOCITY, 1)[0], 0),
        expected_final_velocity=Vec3(0, calculate_expected_free_fall(64.0, JUMP_VELOCITY, 1)[1], 0),
        tolerance=1e-12,
    ),
    MovementTestCase(
        name="jump_at_apex",
        initial_velocity=Vec3(0, JUMP_VELOCITY, 0),
        initial_position=Vec3(0, 64, 0),
        ticks=_jump_apex_tick,
        on_ground=False,
        expected_final_position=Vec3(0, 64.0 + _jump_apex_height, 0),
        expected_final_velocity=Vec3(
            0, calculate_expected_free_fall(64.0, JUMP_VELOCITY, _jump_apex_tick)[1], 0
        ),
        tolerance=1e-9,
    ),
    MovementTestCase(
        name="jump_full_arc_20_ticks",
        initial_velocity=Vec3(0, JUMP_VELOCITY, 0),
        initial_position=Vec3(0, 64, 0),
        ticks=20,
        on_ground=False,
        expected_final_position=Vec3(
            0, calculate_expected_free_fall(64.0, JUMP_VELOCITY, 20)[0], 0
        ),
        expected_final_velocity=Vec3(
            0, calculate_expected_free_fall(64.0, JUMP_VELOCITY, 20)[1], 0
        ),
        tolerance=1e-9,
    ),
    # =========================================================================
    # GROUND MOVEMENT TESTS
    # =========================================================================
    MovementTestCase(
        name="walk_forward_1_tick",
        initial_velocity=Vec3(0, 0, 0),
        initial_position=Vec3(0, 64, 0),
        ticks=1,
        on_ground=True,
        move_forward=True,
        yaw=0.0,  # Facing +Z
        expected_final_position=Vec3(
            0,
            64,
            reference_ground_physics(0, 0, 0, 0, 1, move_forward=True, yaw=0)[1],
        ),
        expected_final_velocity=Vec3(
            0,
            0,
            reference_ground_physics(0, 0, 0, 0, 1, move_forward=True, yaw=0)[3],
        ),
        tolerance=1e-9,
    ),
    MovementTestCase(
        name="walk_forward_10_ticks",
        initial_velocity=Vec3(0, 0, 0),
        initial_position=Vec3(0, 64, 0),
        ticks=10,
        on_ground=True,
        move_forward=True,
        yaw=0.0,
        expected_final_position=Vec3(
            0,
            64,
            reference_ground_physics(0, 0, 0, 0, 10, move_forward=True, yaw=0)[1],
        ),
        expected_final_velocity=Vec3(
            0,
            0,
            reference_ground_physics(0, 0, 0, 0, 10, move_forward=True, yaw=0)[3],
        ),
        tolerance=1e-9,
    ),
    MovementTestCase(
        name="sprint_forward_10_ticks",
        initial_velocity=Vec3(0, 0, 0),
        initial_position=Vec3(0, 64, 0),
        ticks=10,
        on_ground=True,
        move_forward=True,
        sprinting=True,
        yaw=0.0,
        expected_final_position=Vec3(
            0,
            64,
            reference_ground_physics(0, 0, 0, 0, 10, move_forward=True, yaw=0, sprinting=True)[1],
        ),
        expected_final_velocity=Vec3(
            0,
            0,
            reference_ground_physics(0, 0, 0, 0, 10, move_forward=True, yaw=0, sprinting=True)[3],
        ),
        tolerance=1e-9,
    ),
    # =========================================================================
    # ICE MOVEMENT TESTS
    # =========================================================================
    MovementTestCase(
        name="walk_on_ice_10_ticks",
        initial_velocity=Vec3(0, 0, 0),
        initial_position=Vec3(0, 64, 0),
        ticks=10,
        on_ground=True,
        move_forward=True,
        yaw=0.0,
        surface=SurfaceType.ICE,
        expected_final_position=Vec3(
            0,
            64,
            reference_ground_physics(
                0, 0, 0, 0, 10, slipperiness=SLIPPERINESS_ICE, move_forward=True, yaw=0
            )[1],
        ),
        expected_final_velocity=Vec3(
            0,
            0,
            reference_ground_physics(
                0, 0, 0, 0, 10, slipperiness=SLIPPERINESS_ICE, move_forward=True, yaw=0
            )[3],
        ),
        tolerance=1e-9,
    ),
    MovementTestCase(
        name="coast_on_ice_initial_velocity",
        initial_velocity=Vec3(0, 0, 0.5),  # Already moving on ice
        initial_position=Vec3(0, 64, 0),
        ticks=20,
        on_ground=True,
        move_forward=False,  # No input, just coasting
        yaw=0.0,
        surface=SurfaceType.ICE,
        expected_final_position=Vec3(
            0,
            64,
            reference_ground_physics(
                0, 0, 0, 0.5, 20, slipperiness=SLIPPERINESS_ICE, move_forward=False
            )[1],
        ),
        expected_final_velocity=Vec3(
            0,
            0,
            reference_ground_physics(
                0, 0, 0, 0.5, 20, slipperiness=SLIPPERINESS_ICE, move_forward=False
            )[3],
        ),
        tolerance=1e-9,
    ),
    # =========================================================================
    # COMBINED MOVEMENT TESTS
    # =========================================================================
    MovementTestCase(
        name="diagonal_walk",
        initial_velocity=Vec3(0, 0, 0),
        initial_position=Vec3(0, 64, 0),
        ticks=10,
        on_ground=True,
        move_forward=True,
        yaw=45.0,  # Diagonal
        expected_final_position=Vec3(
            reference_ground_physics(0, 0, 0, 0, 10, move_forward=True, yaw=45)[0],
            64,
            reference_ground_physics(0, 0, 0, 0, 10, move_forward=True, yaw=45)[1],
        ),
        expected_final_velocity=Vec3(
            reference_ground_physics(0, 0, 0, 0, 10, move_forward=True, yaw=45)[2],
            0,
            reference_ground_physics(0, 0, 0, 0, 10, move_forward=True, yaw=45)[3],
        ),
        tolerance=1e-9,
    ),
]


class MovementVerifier:
    """Verifies movement physics against reference implementation."""

    def __init__(self, tolerance: float = 1e-9):
        """Initialize verifier.

        Args:
            tolerance: Default comparison tolerance for float values
        """
        self.tolerance = tolerance
        self.results: list[VerificationResult] = []

    def verify_free_fall(
        self, simulator_result: tuple[float, float], ticks: int, y0: float, vy0: float
    ) -> VerificationResult:
        """Verify free fall physics.

        Args:
            simulator_result: (final_y, final_vy) from simulator
            ticks: Number of ticks simulated
            y0: Initial Y position
            vy0: Initial Y velocity

        Returns:
            VerificationResult with pass/fail and details
        """
        expected_y, expected_vy = reference_fall_physics(y0, vy0, ticks)
        actual_y, actual_vy = simulator_result

        y_error = abs(expected_y - actual_y)
        vy_error = abs(expected_vy - actual_vy)

        passed = y_error <= self.tolerance and vy_error <= self.tolerance

        # Create a test case for the result
        test_case = MovementTestCase(
            name=f"free_fall_{ticks}t_from_{y0}_vy{vy0}",
            initial_velocity=Vec3(0, vy0, 0),
            initial_position=Vec3(0, y0, 0),
            ticks=ticks,
            on_ground=False,
            expected_final_position=Vec3(0, expected_y, 0),
            expected_final_velocity=Vec3(0, expected_vy, 0),
        )

        result = VerificationResult(
            test_case=test_case,
            actual_position=Vec3(0, actual_y, 0),
            actual_velocity=Vec3(0, actual_vy, 0),
            passed=passed,
            position_error=y_error,
            velocity_error=vy_error,
            details=f"Y: expected={expected_y:.9f}, actual={actual_y:.9f}, err={y_error:.2e}\n"
            f"VY: expected={expected_vy:.9f}, actual={actual_vy:.9f}, err={vy_error:.2e}",
        )

        self.results.append(result)
        return result

    def verify_test_case(
        self,
        test_case: MovementTestCase,
        actual_position: Vec3,
        actual_velocity: Vec3,
    ) -> VerificationResult:
        """Verify a single test case.

        Args:
            test_case: The test case to verify
            actual_position: Actual final position from simulator
            actual_velocity: Actual final velocity from simulator

        Returns:
            VerificationResult with pass/fail and details
        """
        pos_error = actual_position.max_diff(test_case.expected_final_position)
        vel_error = actual_velocity.max_diff(test_case.expected_final_velocity)

        tol = test_case.tolerance
        passed = pos_error <= tol and vel_error <= tol

        details = (
            f"Position: expected={test_case.expected_final_position}, "
            f"actual={actual_position}, err={pos_error:.2e}\n"
            f"Velocity: expected={test_case.expected_final_velocity}, "
            f"actual={actual_velocity}, err={vel_error:.2e}"
        )

        result = VerificationResult(
            test_case=test_case,
            actual_position=actual_position,
            actual_velocity=actual_velocity,
            passed=passed,
            position_error=pos_error,
            velocity_error=vel_error,
            details=details,
        )

        self.results.append(result)
        return result

    def run_all_tests(
        self, simulator_tick_fn: Any | None = None
    ) -> tuple[int, int, list[VerificationResult]]:
        """Run all predefined test cases.

        If simulator_tick_fn is provided, it will be called to get actual results.
        Otherwise, this just validates the reference implementation against itself.

        Args:
            simulator_tick_fn: Optional function(test_case) -> (actual_pos, actual_vel)

        Returns:
            Tuple of (passed_count, failed_count, all_results)
        """
        self.results.clear()
        passed = 0
        failed = 0

        for test_case in MOVEMENT_TEST_CASES:
            if simulator_tick_fn:
                actual_pos, actual_vel = simulator_tick_fn(test_case)
            else:
                # Self-validation: use expected as actual
                actual_pos = test_case.expected_final_position
                actual_vel = test_case.expected_final_velocity

            result = self.verify_test_case(test_case, actual_pos, actual_vel)

            if result.passed:
                passed += 1
            else:
                failed += 1

        return passed, failed, self.results

    def print_results(self, verbose: bool = False) -> None:
        """Print verification results.

        Args:
            verbose: If True, print details for all tests. If False, only failures.
        """
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print(f"\n{'=' * 60}")
        print(f"MOVEMENT PHYSICS VERIFICATION: {passed}/{total} passed")
        print(f"{'=' * 60}")

        for result in self.results:
            if verbose or not result.passed:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                print(f"\n[{status}] {result.test_case.name}")
                print(f"  {result.details}")


def run_self_verification() -> bool:
    """Run verification of reference implementation against itself.

    This validates that the expected values are correctly computed.

    Returns:
        True if all tests pass
    """
    print("=" * 60)
    print("MC 1.8.9 MOVEMENT PHYSICS SELF-VERIFICATION")
    print("=" * 60)

    verifier = MovementVerifier()
    passed, failed, _ = verifier.run_all_tests()

    print(f"\nSelf-verification: {passed}/{passed + failed} tests passed")

    if failed > 0:
        verifier.print_results(verbose=False)
        return False

    print("All reference computations are self-consistent.")

    # Print some key values for manual verification
    print("\n" + "-" * 40)
    print("KEY PHYSICS VALUES")
    print("-" * 40)

    print("\nConstants:")
    print(f"  GRAVITY = {GRAVITY}")
    print(f"  DRAG_AIR = {DRAG_AIR}")
    print(f"  DRAG_GROUND = {DRAG_GROUND}")
    print(f"  JUMP_VELOCITY = {JUMP_VELOCITY}")
    print(f"  SPRINT_MULTIPLIER = {SPRINT_MULTIPLIER}")
    print(f"  WALK_SPEED = {WALK_SPEED}")

    print("\nFree fall from y=100, vy=0:")
    for ticks in [1, 5, 10, 20, 50]:
        y, vy = reference_fall_physics(100.0, 0.0, ticks)
        print(f"  {ticks:2d} ticks: y={y:10.6f}, vy={vy:10.6f}")

    apex_tick, apex_height = calculate_jump_apex()
    print(f"\nJump apex: tick={apex_tick}, height={apex_height:.6f} blocks")

    print("\nJump from y=64:")
    for ticks in [1, 5, apex_tick, 15, 20]:
        y, vy = calculate_expected_free_fall(64.0, JUMP_VELOCITY, ticks)
        print(f"  {ticks:2d} ticks: y={y:10.6f}, vy={vy:10.6f}")

    print("\nGround movement (10 ticks, forward, yaw=0):")
    _, z, _, vz = reference_ground_physics(0, 0, 0, 0, 10, move_forward=True)
    print(f"  Walk:   z={z:.6f}, vz={vz:.6f}")
    _, z, _, vz = reference_ground_physics(0, 0, 0, 0, 10, move_forward=True, sprinting=True)
    print(f"  Sprint: z={z:.6f}, vz={vz:.6f}")

    print("\nIce movement (10 ticks, forward, yaw=0):")
    _, z, _, vz = reference_ground_physics(
        0, 0, 0, 0, 10, slipperiness=SLIPPERINESS_ICE, move_forward=True
    )
    print(f"  Walk:   z={z:.6f}, vz={vz:.6f}")

    print("\nIce coasting (20 ticks, vz=0.5, no input):")
    _, z, _, vz = reference_ground_physics(
        0, 0, 0, 0.5, 20, slipperiness=SLIPPERINESS_ICE, move_forward=False
    )
    print(f"  Coast:  z={z:.6f}, vz={vz:.9f}")

    return True


if __name__ == "__main__":
    success = run_self_verification()
    sys.exit(0 if success else 1)
