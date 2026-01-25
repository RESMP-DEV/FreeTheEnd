"""Eye of Ender verification for Minecraft-style mechanics.

Verifies:
1. Eye trajectory towards stronghold
2. Eye shatter probability distribution (20% base chance)
3. Flight mechanics (velocity, arc, duration)
"""

import math
import random
from dataclasses import dataclass
from typing import NamedTuple


class BlockPos(NamedTuple):
    """3D block position (integer coordinates)."""

    x: int
    y: int
    z: int


class Vec3(NamedTuple):
    """3D vector (float coordinates)."""

    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Vec3":
        length = self.length()
        if length == 0:
            return Vec3(0, 0, 0)
        return Vec3(self.x / length, self.y / length, self.z / length)

    def horizontal_length(self) -> float:
        return math.sqrt(self.x**2 + self.z**2)


@dataclass
class StrongholdLocation:
    """Stronghold location data."""

    pos: BlockPos
    portal_room_center: BlockPos


@dataclass
class EyeOfEnderState:
    """State of a thrown Eye of Ender."""

    position: Vec3
    velocity: Vec3
    target: BlockPos
    ticks_alive: int = 0
    rising: bool = True
    falling: bool = False
    stationary_ticks: int = 0
    will_shatter: bool = False
    dropped: bool = False


class EyeTrajectoryVerifier:
    """Verifies Eye of Ender trajectory towards stronghold.

    Minecraft Eye of Ender mechanics:
    - Eye flies toward nearest stronghold (portal room)
    - Initial upward velocity with horizontal direction toward target
    - Rises for ~15 blocks then hovers before falling
    - Horizontal speed based on angle to target
    - When above stronghold, eye moves downward
    """

    INITIAL_UPWARD_VELOCITY = 0.5
    HORIZONTAL_SPEED = 0.4
    GRAVITY = 0.03  # MC 1.8.9 Eye of Ender gravity (NOT 0.08 for entities)
    HOVER_TICKS = 20
    RISE_DISTANCE = 15.0

    def __init__(self, stronghold: StrongholdLocation):
        self.stronghold = stronghold

    def throw_eye(self, throw_pos: Vec3, seed: int | None = None) -> EyeOfEnderState:
        """Simulate throwing an Eye of Ender.

        Args:
            throw_pos: Position where eye is thrown from
            seed: Random seed for shatter determination

        Returns:
            Initial state of the thrown eye
        """
        if seed is not None:
            random.seed(seed)

        # Calculate direction to stronghold (horizontal only)
        target = self.stronghold.portal_room_center
        dx = target.x - throw_pos.x
        dz = target.z - throw_pos.z
        horizontal_dist = math.sqrt(dx**2 + dz**2)

        if horizontal_dist > 0:
            # Normalize horizontal direction
            dir_x = dx / horizontal_dist
            dir_z = dz / horizontal_dist
        else:
            # Already at stronghold, eye goes straight up then down
            dir_x = 0.0
            dir_z = 0.0

        # Initial velocity: upward + horizontal toward target
        velocity = Vec3(
            dir_x * self.HORIZONTAL_SPEED,
            self.INITIAL_UPWARD_VELOCITY,
            dir_z * self.HORIZONTAL_SPEED,
        )

        # 20% base shatter chance (in Java Edition)
        will_shatter = random.random() < 0.2

        return EyeOfEnderState(
            position=throw_pos,
            velocity=velocity,
            target=target,
            will_shatter=will_shatter,
        )

    def tick_eye(self, state: EyeOfEnderState) -> EyeOfEnderState:
        """Process one game tick for the eye.

        Args:
            state: Current eye state

        Returns:
            Updated state
        """
        state.ticks_alive += 1

        # Rising phase
        if state.rising:
            rise_height = state.position.y - state.velocity.y * state.ticks_alive
            if state.position.y > rise_height + self.RISE_DISTANCE:
                state.rising = False
                state.velocity = Vec3(state.velocity.x, 0, state.velocity.z)

        # Apply velocity
        state.position = state.position + state.velocity

        # Hover phase (gradual velocity reduction)
        if not state.rising and not state.falling:
            state.stationary_ticks += 1
            # Reduce horizontal velocity
            state.velocity = Vec3(
                state.velocity.x * 0.9,
                state.velocity.y,
                state.velocity.z * 0.9,
            )
            if state.stationary_ticks >= self.HOVER_TICKS:
                state.falling = True

        # Falling phase
        if state.falling:
            state.velocity = Vec3(
                state.velocity.x,
                state.velocity.y - self.GRAVITY,
                state.velocity.z,
            )

            # Check if reached ground level
            if state.position.y <= 64:  # Approximate ground level
                state.dropped = True

        return state

    def simulate_flight(
        self,
        throw_pos: Vec3,
        max_ticks: int = 200,
        seed: int | None = None,
    ) -> list[Vec3]:
        """Simulate complete eye flight and return trajectory.

        Args:
            throw_pos: Position where eye is thrown
            max_ticks: Maximum ticks to simulate
            seed: Random seed

        Returns:
            List of positions along trajectory
        """
        state = self.throw_eye(throw_pos, seed)
        trajectory: list[Vec3] = [state.position]

        for _ in range(max_ticks):
            state = self.tick_eye(state)
            trajectory.append(state.position)
            if state.dropped:
                break

        return trajectory

    def verify_direction(self, throw_pos: Vec3) -> tuple[bool, float, str]:
        """Verify eye points toward stronghold.

        Args:
            throw_pos: Position eye is thrown from

        Returns:
            Tuple of (is_correct, angle_error, explanation)
        """
        state = self.throw_eye(throw_pos, seed=42)

        # Get direction to target
        target = self.stronghold.portal_room_center
        expected_dx = target.x - throw_pos.x
        expected_dz = target.z - throw_pos.z
        expected_angle = math.atan2(expected_dz, expected_dx)

        # Get eye's initial direction
        actual_angle = math.atan2(state.velocity.z, state.velocity.x)

        angle_error = abs(expected_angle - actual_angle)
        # Normalize to 0-pi range
        if angle_error > math.pi:
            angle_error = 2 * math.pi - angle_error

        is_correct = angle_error < 0.01  # Allow small floating point error
        degrees = math.degrees(angle_error)

        if is_correct:
            return True, degrees, f"Correct direction (error: {degrees:.2f} deg)"
        else:
            return False, degrees, f"Wrong direction (error: {degrees:.2f} deg)"


class EyeShatterVerifier:
    """Verifies Eye of Ender shatter probability.

    Minecraft shatter mechanics:
    - 20% base chance to shatter on drop
    - Eye breaks into particles instead of dropping item
    - Distribution should follow binomial with p=0.2
    """

    SHATTER_PROBABILITY = 0.2
    SURVIVAL_PROBABILITY = 0.8

    def __init__(self):
        self.shatter_count = 0
        self.survival_count = 0
        self.trials: list[bool] = []

    def throw_eye(self, seed: int | None = None) -> bool:
        """Simulate throwing an eye and determine if it shatters.

        Args:
            seed: Random seed for reproducibility

        Returns:
            True if eye shatters, False if it survives
        """
        if seed is not None:
            random.seed(seed)

        shatters = random.random() < self.SHATTER_PROBABILITY
        self.trials.append(shatters)

        if shatters:
            self.shatter_count += 1
        else:
            self.survival_count += 1

        return shatters

    def run_trials(self, n: int, base_seed: int = 0) -> tuple[int, int]:
        """Run multiple eye throws.

        Args:
            n: Number of trials
            base_seed: Starting seed

        Returns:
            Tuple of (shatter_count, survival_count)
        """
        self.reset()
        for i in range(n):
            self.throw_eye(seed=base_seed + i)
        return self.shatter_count, self.survival_count

    def reset(self) -> None:
        """Reset trial statistics."""
        self.shatter_count = 0
        self.survival_count = 0
        self.trials.clear()

    def get_observed_probability(self) -> float:
        """Get observed shatter probability from trials."""
        total = self.shatter_count + self.survival_count
        if total == 0:
            return 0.0
        return self.shatter_count / total

    def verify_distribution(
        self,
        n_trials: int = 1000,
        confidence_level: float = 0.95,
    ) -> tuple[bool, float, str]:
        """Verify shatter distribution matches expected 20%.

        Uses binomial confidence interval.

        Args:
            n_trials: Number of trials to run
            confidence_level: Confidence level for interval

        Returns:
            Tuple of (is_within_expected, observed_prob, explanation)
        """
        self.reset()

        # Use time-based seed for variety
        import time

        base_seed = int(time.time() * 1000) % 1000000

        for i in range(n_trials):
            self.throw_eye(seed=base_seed + i)

        observed_prob = self.get_observed_probability()

        # Calculate binomial confidence interval using normal approximation
        # For large n, binomial approaches normal
        z = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
        std_err = math.sqrt(observed_prob * (1 - observed_prob) / n_trials)

        lower = observed_prob - z * std_err
        upper = observed_prob + z * std_err

        is_valid = lower <= self.SHATTER_PROBABILITY <= upper

        msg = (
            f"Observed: {observed_prob:.3f} ({self.shatter_count}/{n_trials}), "
            f"Expected: {self.SHATTER_PROBABILITY}, "
            f"CI: [{lower:.3f}, {upper:.3f}]"
        )

        return is_valid, observed_prob, msg

    def expected_eyes_to_stronghold(self, distance_blocks: int) -> tuple[float, float]:
        """Calculate expected eyes needed to reach stronghold.

        Args:
            distance_blocks: Distance to stronghold in blocks

        Returns:
            Tuple of (expected_throws, expected_eyes_consumed)
        """
        # Eye travels roughly 12 blocks per throw on average
        throws_per_leg = 12
        expected_throws = distance_blocks / throws_per_leg

        # Each throw has 20% chance to consume the eye
        eyes_consumed = expected_throws * self.SHATTER_PROBABILITY

        return expected_throws, eyes_consumed


def run_verification() -> None:
    """Run all Eye of Ender verifications."""
    print("=" * 60)
    print("EYE OF ENDER VERIFICATION SUITE")
    print("=" * 60)

    # Test 1: Eye trajectory toward stronghold
    print("\n[1] Eye Trajectory Toward Stronghold")
    print("-" * 40)

    stronghold = StrongholdLocation(
        pos=BlockPos(1200, 40, -800),
        portal_room_center=BlockPos(1200, 30, -800),
    )
    trajectory_verifier = EyeTrajectoryVerifier(stronghold)

    # Test from various positions
    test_positions = [
        Vec3(0.0, 70.0, 0.0),
        Vec3(500.0, 64.0, -300.0),
        Vec3(1000.0, 80.0, -750.0),  # Close to stronghold
        Vec3(-200.0, 64.0, 100.0),  # Opposite direction
    ]

    for throw_pos in test_positions:
        is_correct, angle_error, msg = trajectory_verifier.verify_direction(throw_pos)
        dist = math.sqrt(
            (stronghold.pos.x - throw_pos.x) ** 2 + (stronghold.pos.z - throw_pos.z) ** 2
        )
        print(f"  From {throw_pos}:")
        print(f"    Distance to stronghold: {dist:.0f} blocks")
        print(f"    Direction: {msg}")

    # Simulate and show trajectory points
    print("\n  Trajectory simulation from (0, 70, 0):")
    trajectory = trajectory_verifier.simulate_flight(Vec3(0, 70, 0), max_ticks=100, seed=42)
    key_points = [0, 10, 30, 50, min(70, len(trajectory) - 1)]
    for i in key_points:
        if i < len(trajectory):
            pos = trajectory[i]
            print(f"    Tick {i:3d}: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})")

    # Test 2: Eye shatter probability
    print("\n[2] Eye Shatter Probability Distribution")
    print("-" * 40)

    shatter_verifier = EyeShatterVerifier()

    # Run small trials first
    print("  Small sample tests (100 trials each):")
    for trial in range(3):
        shatter, survive = shatter_verifier.run_trials(100, base_seed=trial * 1000)
        prob = shatter / 100
        print(f"    Trial {trial + 1}: {shatter} shattered, {survive} survived ({prob:.0%})")

    # Run large trial for statistical verification
    print("\n  Large sample verification (10000 trials):")
    is_valid, observed, msg = shatter_verifier.verify_distribution(n_trials=10000)
    print(f"    {msg}")
    print(f"    Distribution valid: {is_valid}")

    # Expected eyes to reach stronghold
    print("\n  Expected eyes for various distances:")
    distances = [500, 1000, 2000, 4000]
    for dist in distances:
        throws, consumed = shatter_verifier.expected_eyes_to_stronghold(dist)
        print(f"    {dist} blocks: ~{throws:.1f} throws, ~{consumed:.1f} eyes consumed")

    # Test 3: Specific shatter sequences
    print("\n[3] Shatter Sequence Verification")
    print("-" * 40)

    # Test deterministic sequences with known seeds
    print("  Deterministic sequence tests:")
    test_seeds = [42, 123, 456, 789, 1024]
    for seed in test_seeds:
        shatter_verifier.reset()
        results = [shatter_verifier.throw_eye(seed=seed + i) for i in range(10)]
        shatter_count = sum(results)
        symbols = ["X" if r else "O" for r in results]
        print(f"    Seed {seed}: {''.join(symbols)} ({shatter_count}/10 shattered)")

    # Verify reproducibility
    print("\n  Reproducibility test:")
    sequence1 = [shatter_verifier.throw_eye(seed=42 + i) for i in range(5)]
    shatter_verifier.reset()
    sequence2 = [shatter_verifier.throw_eye(seed=42 + i) for i in range(5)]
    is_reproducible = sequence1 == sequence2
    print(f"    Same seed produces same sequence: {is_reproducible}")

    print("\n" + "=" * 60)
    print("EYE OF ENDER VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_verification()
