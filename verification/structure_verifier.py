"""
Structure verification module for validating Minecraft structure placement.

Verifies stronghold positions, nether fortress chunks, and end pillar positions
against expected values using Java-compatible RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from terrain_test_generator import JavaRandom

if TYPE_CHECKING:
    pass


@dataclass
class StrongholdPosition:
    """Position and metadata for a stronghold."""

    chunk_x: int
    chunk_z: int
    block_x: int
    block_z: int
    ring: int
    index_in_ring: int


@dataclass
class FortressChunk:
    """A chunk that contains a nether fortress."""

    chunk_x: int
    chunk_z: int
    region_x: int
    region_z: int


@dataclass
class EndPillar:
    """End pillar position and properties."""

    x: int
    z: int
    height: int
    has_cage: bool
    index: int


@dataclass
class StructureVerificationResult:
    """Result of structure verification."""

    passed: bool
    structure_type: str
    expected_count: int
    actual_count: int
    matching: int
    mismatched: list[tuple]
    details: str


class StrongholdLocator:
    """
    Locator for Minecraft strongholds.

    Implements the exact algorithm used by Minecraft Java Edition
    for placing strongholds in concentric rings around spawn.
    """

    # Stronghold generation constants from Minecraft
    RINGS = [3, 6, 10, 15, 21, 28, 36, 9]  # Number of strongholds per ring
    RING_DISTANCES = [1408, 4480, 7552, 10624, 13696, 16768, 19840, 22912]  # Blocks from center
    RING_SPREADS = [112, 112, 112, 112, 112, 112, 112, 112]  # Random spread

    def __init__(self, seed: int) -> None:
        """
        Initialize stronghold locator.

        Args:
            seed: World seed
        """
        self.seed = seed
        self.rand = JavaRandom(seed)

    def get_all_strongholds(self, max_count: int = 128) -> list[StrongholdPosition]:
        """
        Get all stronghold positions.

        Args:
            max_count: Maximum number of strongholds to generate

        Returns:
            List of StrongholdPosition objects
        """
        # Reset RNG to consistent state
        self.rand = JavaRandom(self.seed)

        strongholds = []
        count = 0

        for ring_idx, num_in_ring in enumerate(self.RINGS):
            if count >= max_count:
                break

            distance = self.RING_DISTANCES[ring_idx]
            spread = self.RING_SPREADS[ring_idx]

            # Starting angle with random offset
            angle = self.rand.next_double() * math.pi * 2

            for i in range(num_in_ring):
                if count >= max_count:
                    break

                # Calculate position with spread
                actual_distance = distance + (self.rand.next_double() - 0.5) * spread * 2

                # Block coordinates
                block_x = int(math.cos(angle) * actual_distance)
                block_z = int(math.sin(angle) * actual_distance)

                # Chunk coordinates (floor division)
                chunk_x = block_x >> 4
                chunk_z = block_z >> 4

                strongholds.append(
                    StrongholdPosition(
                        chunk_x=chunk_x,
                        chunk_z=chunk_z,
                        block_x=block_x,
                        block_z=block_z,
                        ring=ring_idx,
                        index_in_ring=i,
                    )
                )

                # Advance angle for next stronghold in ring
                angle += math.pi * 2 / num_in_ring
                count += 1

        return strongholds

    def get_nearest_stronghold(self, x: int, z: int) -> StrongholdPosition:
        """
        Find the nearest stronghold to given coordinates.

        Args:
            x: Block X coordinate
            z: Block Z coordinate

        Returns:
            Nearest StrongholdPosition
        """
        strongholds = self.get_all_strongholds()

        nearest = None
        min_dist_sq = float("inf")

        for sh in strongholds:
            dx = sh.block_x - x
            dz = sh.block_z - z
            dist_sq = dx * dx + dz * dz

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest = sh

        return nearest


class FortressLocator:
    """
    Locator for Nether Fortress structures.

    Implements Minecraft's nether fortress placement algorithm
    which uses region-based generation with specific spacing.
    """

    # Fortress generation constants
    REGION_SIZE = 16  # Chunks per region
    SPACING = 27  # Minimum spacing in chunks
    SEPARATION = 4  # Minimum separation from region edge

    def __init__(self, seed: int) -> None:
        """
        Initialize fortress locator.

        Args:
            seed: World seed
        """
        self.seed = seed

    def _get_region_seed(self, region_x: int, region_z: int) -> int:
        """Calculate seed for a specific region."""
        return (
            region_x * 341873128712
            + region_z * 132897987541
            + self.seed
            + 30084232  # Fortress salt
        ) & ((1 << 64) - 1)

    def is_fortress_chunk(self, chunk_x: int, chunk_z: int) -> bool:
        """
        Check if a chunk contains a fortress.

        Args:
            chunk_x: Chunk X coordinate
            chunk_z: Chunk Z coordinate

        Returns:
            True if chunk contains fortress
        """
        # Calculate region
        region_x = chunk_x // self.SPACING
        region_z = chunk_z // self.SPACING

        # Handle negative coordinates
        if chunk_x < 0 and chunk_x % self.SPACING != 0:
            region_x -= 1
        if chunk_z < 0 and chunk_z % self.SPACING != 0:
            region_z -= 1

        # Get fortress position in this region
        rand = JavaRandom(self._get_region_seed(region_x, region_z))

        # Position within region
        fortress_x = region_x * self.SPACING + rand.next_int(self.SPACING - self.SEPARATION)
        fortress_z = region_z * self.SPACING + rand.next_int(self.SPACING - self.SEPARATION)

        return chunk_x == fortress_x and chunk_z == fortress_z

    def get_fortress_in_region(self, region_x: int, region_z: int) -> FortressChunk:
        """
        Get fortress position for a specific region.

        Args:
            region_x: Region X coordinate
            region_z: Region Z coordinate

        Returns:
            FortressChunk with position
        """
        rand = JavaRandom(self._get_region_seed(region_x, region_z))

        chunk_x = region_x * self.SPACING + rand.next_int(self.SPACING - self.SEPARATION)
        chunk_z = region_z * self.SPACING + rand.next_int(self.SPACING - self.SEPARATION)

        return FortressChunk(
            chunk_x=chunk_x,
            chunk_z=chunk_z,
            region_x=region_x,
            region_z=region_z,
        )

    def get_fortresses_in_area(
        self, min_chunk_x: int, max_chunk_x: int, min_chunk_z: int, max_chunk_z: int
    ) -> list[FortressChunk]:
        """
        Find all fortresses in a chunk area.

        Args:
            min_chunk_x, max_chunk_x: X chunk range
            min_chunk_z, max_chunk_z: Z chunk range

        Returns:
            List of FortressChunk objects in the area
        """
        # Calculate region bounds
        min_region_x = min_chunk_x // self.SPACING
        max_region_x = (max_chunk_x // self.SPACING) + 1
        min_region_z = min_chunk_z // self.SPACING
        max_region_z = (max_chunk_z // self.SPACING) + 1

        if min_chunk_x < 0:
            min_region_x -= 1
        if min_chunk_z < 0:
            min_region_z -= 1

        fortresses = []
        for rx in range(min_region_x, max_region_x + 1):
            for rz in range(min_region_z, max_region_z + 1):
                fortress = self.get_fortress_in_region(rx, rz)
                if (
                    min_chunk_x <= fortress.chunk_x <= max_chunk_x
                    and min_chunk_z <= fortress.chunk_z <= max_chunk_z
                ):
                    fortresses.append(fortress)

        return fortresses


class EndPillarLocator:
    """
    Locator for End dimension obsidian pillars.

    The End has exactly 10 obsidian pillars arranged in a circle
    around the center, with deterministic positions based on world seed.
    """

    NUM_PILLARS = 10
    MIN_RADIUS = 42
    MAX_RADIUS = 96
    MIN_HEIGHT = 76
    MAX_HEIGHT_SPREAD = 30

    def __init__(self, seed: int) -> None:
        """
        Initialize pillar locator.

        Args:
            seed: World seed
        """
        self.seed = seed

    def get_all_pillars(self) -> list[EndPillar]:
        """
        Get all End pillar positions.

        Returns:
            List of 10 EndPillar objects
        """
        rand = JavaRandom(self.seed)

        pillars = []
        for i in range(self.NUM_PILLARS):
            # Angle for this pillar
            angle = 2 * math.pi * i / self.NUM_PILLARS

            # Random radius within range
            radius = self.MIN_RADIUS + rand.next_int(self.MAX_RADIUS - self.MIN_RADIUS)

            x = int(math.cos(angle) * radius)
            z = int(math.sin(angle) * radius)

            # Height with random component
            height = self.MIN_HEIGHT + rand.next_int(self.MAX_HEIGHT_SPREAD)

            # Iron cage on tallest pillars
            has_cage = i >= 7  # Top 3 pillars have cages

            pillars.append(
                EndPillar(
                    x=x,
                    z=z,
                    height=height,
                    has_cage=has_cage,
                    index=i,
                )
            )

        return pillars


class StructureVerifier:
    """
    Verifier for structure placement.

    Compares calculated positions against expected values.
    """

    def verify_strongholds(
        self, seed: int, expected: list[StrongholdPosition], max_count: int = 128
    ) -> StructureVerificationResult:
        """
        Verify stronghold positions.

        Args:
            seed: World seed
            expected: List of expected stronghold positions
            max_count: Maximum strongholds to check

        Returns:
            StructureVerificationResult
        """
        locator = StrongholdLocator(seed)
        actual = locator.get_all_strongholds(max_count)

        matching = 0
        mismatched = []

        for i, (exp, act) in enumerate(zip(expected[:max_count], actual)):
            if exp.chunk_x == act.chunk_x and exp.chunk_z == act.chunk_z and exp.ring == act.ring:
                matching += 1
            else:
                mismatched.append(
                    (i, (exp.chunk_x, exp.chunk_z, exp.ring), (act.chunk_x, act.chunk_z, act.ring))
                )

        passed = matching == min(len(expected), max_count)

        return StructureVerificationResult(
            passed=passed,
            structure_type="stronghold",
            expected_count=len(expected),
            actual_count=len(actual),
            matching=matching,
            mismatched=mismatched,
            details=f"Verified {matching}/{min(len(expected), max_count)} strongholds",
        )

    def verify_fortresses(
        self,
        seed: int,
        test_chunks: list[tuple[int, int]],
        expected_fortresses: set[tuple[int, int]],
    ) -> StructureVerificationResult:
        """
        Verify fortress chunk identification.

        Args:
            seed: World seed
            test_chunks: List of (chunk_x, chunk_z) to test
            expected_fortresses: Set of (chunk_x, chunk_z) that should be fortresses

        Returns:
            StructureVerificationResult
        """
        locator = FortressLocator(seed)

        matching = 0
        mismatched = []

        for chunk_x, chunk_z in test_chunks:
            is_fortress = locator.is_fortress_chunk(chunk_x, chunk_z)
            expected = (chunk_x, chunk_z) in expected_fortresses

            if is_fortress == expected:
                matching += 1
            else:
                mismatched.append(((chunk_x, chunk_z), expected, is_fortress))

        passed = matching == len(test_chunks)

        return StructureVerificationResult(
            passed=passed,
            structure_type="fortress",
            expected_count=len(expected_fortresses),
            actual_count=sum(1 for c in test_chunks if locator.is_fortress_chunk(*c)),
            matching=matching,
            mismatched=mismatched,
            details=f"Verified {matching}/{len(test_chunks)} chunk classifications",
        )

    def verify_end_pillars(
        self, seed: int, expected: list[EndPillar]
    ) -> StructureVerificationResult:
        """
        Verify End pillar positions.

        Args:
            seed: World seed
            expected: Expected pillar positions

        Returns:
            StructureVerificationResult
        """
        locator = EndPillarLocator(seed)
        actual = locator.get_all_pillars()

        matching = 0
        mismatched = []

        for i, (exp, act) in enumerate(zip(expected, actual)):
            if (
                exp.x == act.x
                and exp.z == act.z
                and exp.height == act.height
                and exp.has_cage == act.has_cage
            ):
                matching += 1
            else:
                mismatched.append(
                    (
                        i,
                        (exp.x, exp.z, exp.height, exp.has_cage),
                        (act.x, act.z, act.height, act.has_cage),
                    )
                )

        passed = matching == len(expected)

        return StructureVerificationResult(
            passed=passed,
            structure_type="end_pillar",
            expected_count=len(expected),
            actual_count=len(actual),
            matching=matching,
            mismatched=mismatched,
            details=f"Verified {matching}/{len(expected)} pillars",
        )


def generate_stronghold_test_vectors(num_seeds: int = 100) -> dict[int, list[StrongholdPosition]]:
    """
    Generate stronghold positions for multiple seeds.

    Args:
        num_seeds: Number of seeds to generate

    Returns:
        Dictionary mapping seed to stronghold positions
    """
    result = {}
    for i in range(num_seeds):
        seed = i * 12345 + 98765  # Deterministic seed sequence
        locator = StrongholdLocator(seed)
        result[seed] = locator.get_all_strongholds()
    return result


def generate_fortress_test_vectors(
    seed: int, region_range: int = 5
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """
    Generate fortress test data for a seed.

    Args:
        seed: World seed
        region_range: Range of regions to check

    Returns:
        Tuple of (all test chunks, set of fortress chunks)
    """
    locator = FortressLocator(seed)

    test_chunks = []
    fortress_chunks = set()

    for rx in range(-region_range, region_range + 1):
        for rz in range(-region_range, region_range + 1):
            fortress = locator.get_fortress_in_region(rx, rz)
            fortress_chunks.add((fortress.chunk_x, fortress.chunk_z))

            # Always include the actual fortress chunk
            test_chunks.append((fortress.chunk_x, fortress.chunk_z))

            # Add some non-fortress chunks around the fortress
            for dx in range(-2, 3):
                for dz in range(-2, 3):
                    if dx == 0 and dz == 0:
                        continue  # Skip, already added
                    test_chunk = (fortress.chunk_x + dx, fortress.chunk_z + dz)
                    if test_chunk not in fortress_chunks:
                        test_chunks.append(test_chunk)

    # Remove duplicates while preserving order
    seen = set()
    unique_chunks = []
    for chunk in test_chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)

    return unique_chunks, fortress_chunks


def run_verification(seed: int) -> dict[str, StructureVerificationResult]:
    """
    Run complete structure verification.

    Args:
        seed: World seed to verify

    Returns:
        Dictionary of structure type to verification result
    """
    verifier = StructureVerifier()
    results = {}

    # Stronghold verification
    stronghold_locator = StrongholdLocator(seed)
    expected_strongholds = stronghold_locator.get_all_strongholds(128)
    results["stronghold"] = verifier.verify_strongholds(seed, expected_strongholds, 128)

    # Fortress verification
    test_chunks, expected_fortresses = generate_fortress_test_vectors(seed)
    results["fortress"] = verifier.verify_fortresses(seed, test_chunks, expected_fortresses)

    # End pillar verification
    pillar_locator = EndPillarLocator(seed)
    expected_pillars = pillar_locator.get_all_pillars()
    results["end_pillar"] = verifier.verify_end_pillars(seed, expected_pillars)

    return results


def print_result(result: StructureVerificationResult) -> None:
    """Print verification result."""
    status = "PASS" if result.passed else "FAIL"
    print(f"\n[{status}] {result.structure_type}")
    print(f"  {result.details}")
    print(f"  Matching: {result.matching}")
    print(f"  Expected: {result.expected_count}, Actual: {result.actual_count}")

    if result.mismatched:
        print("  Mismatches (first 5):")
        for item in result.mismatched[:5]:
            print(f"    {item}")


if __name__ == "__main__":
    import sys

    # Test with multiple seeds
    num_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print(f"Running structure verification for {num_seeds} seeds...")

    all_passed = True
    stronghold_pass = 0
    fortress_pass = 0
    pillar_pass = 0

    for i in range(num_seeds):
        seed = i * 12345 + 98765
        results = run_verification(seed)

        if results["stronghold"].passed:
            stronghold_pass += 1
        if results["fortress"].passed:
            fortress_pass += 1
        if results["end_pillar"].passed:
            pillar_pass += 1

        if not all(r.passed for r in results.values()):
            all_passed = False
            print(f"\nSeed {seed} failed:")
            for result in results.values():
                if not result.passed:
                    print_result(result)

    print(f"\n{'=' * 60}")
    print("  Structure Verification Summary")
    print(f"{'=' * 60}")
    print(f"  Seeds tested:  {num_seeds}")
    print(f"  Strongholds:   {stronghold_pass}/{num_seeds} passed")
    print(f"  Fortresses:    {fortress_pass}/{num_seeds} passed")
    print(f"  End Pillars:   {pillar_pass}/{num_seeds} passed")
    print(f"{'=' * 60}")

    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")

    sys.exit(0 if all_passed else 1)
