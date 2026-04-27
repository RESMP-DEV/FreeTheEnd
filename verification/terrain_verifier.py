"""
Terrain verification module for validating noise and biome generation.

Compares Python implementations against Java reference values to ensure
bit-exact compatibility with Minecraft's terrain generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from terrain_test_generator import (
    BiomeTestVector,
    JavaRandom,
    NoiseTestVector,
    OctaveNoise,
    PerlinNoise,
    SimplexNoise,
    _select_biome,
)

if TYPE_CHECKING:
    pass


@dataclass
class VerificationResult:
    """Result of a single verification test."""

    passed: bool
    test_name: str
    expected: float | int
    actual: float | int
    error: float
    coords: tuple[float, ...]
    seed: int


@dataclass
class VerificationSummary:
    """Summary of verification results."""

    total_tests: int
    passed: int
    failed: int
    max_error: float
    avg_error: float
    worst_cases: list[VerificationResult]


class NoiseVerifier:
    """
    Verifier for noise function implementations.

    Tests simplex and Perlin noise against expected values
    generated from the same seed using Java-compatible RNG.
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        """
        Initialize verifier.

        Args:
            tolerance: Maximum allowed difference between expected and actual values.
                       Default is 1e-10 for floating point comparison.
        """
        self.tolerance = tolerance

    def verify_simplex_2d(
        self, seed: int, test_vectors: list[NoiseTestVector]
    ) -> VerificationSummary:
        """
        Verify 2D simplex noise against expected values.

        Args:
            seed: World seed
            test_vectors: List of test vectors with expected values

        Returns:
            VerificationSummary with test results
        """
        rand = JavaRandom(seed)
        simplex = SimplexNoise(rand)

        results = []
        errors = []

        for vec in test_vectors:
            if vec.expected_simplex_2d is None:
                continue

            actual = simplex.noise_2d(vec.x, vec.z)
            error = abs(actual - vec.expected_simplex_2d)
            errors.append(error)

            passed = error <= self.tolerance
            results.append(
                VerificationResult(
                    passed=passed,
                    test_name="simplex_2d",
                    expected=vec.expected_simplex_2d,
                    actual=actual,
                    error=error,
                    coords=(vec.x, vec.z),
                    seed=vec.seed,
                )
            )

        return self._summarize(results, errors)

    def verify_perlin_3d(
        self, seed: int, test_vectors: list[NoiseTestVector]
    ) -> VerificationSummary:
        """
        Verify 3D Perlin noise against expected values.

        Args:
            seed: World seed
            test_vectors: List of test vectors with expected values

        Returns:
            VerificationSummary with test results
        """
        rand = JavaRandom(seed)
        perlin = PerlinNoise(rand)

        results = []
        errors = []

        for vec in test_vectors:
            if vec.expected_perlin_3d is None:
                continue

            actual = perlin.noise_3d(vec.x, vec.y, vec.z)
            error = abs(actual - vec.expected_perlin_3d)
            errors.append(error)

            passed = error <= self.tolerance
            results.append(
                VerificationResult(
                    passed=passed,
                    test_name="perlin_3d",
                    expected=vec.expected_perlin_3d,
                    actual=actual,
                    error=error,
                    coords=(vec.x, vec.y, vec.z),
                    seed=vec.seed,
                )
            )

        return self._summarize(results, errors)

    def verify_from_file(self, filepath: str | Path) -> dict[str, VerificationSummary]:
        """
        Verify noise functions from saved test vectors.

        Args:
            filepath: Path to JSON file with test vectors

        Returns:
            Dictionary mapping test name to VerificationSummary
        """
        with open(filepath) as f:
            data = json.load(f)

        vectors = []
        for item in data:
            vectors.append(
                NoiseTestVector(
                    seed=item["seed"],
                    x=item["x"],
                    y=item["y"],
                    z=item["z"],
                    expected_simplex_2d=item.get("simplex_2d"),
                    expected_perlin_3d=item.get("perlin_3d"),
                )
            )

        if not vectors:
            raise ValueError("No test vectors found in file")

        seed = vectors[0].seed

        return {
            "simplex_2d": self.verify_simplex_2d(seed, vectors),
            "perlin_3d": self.verify_perlin_3d(seed, vectors),
        }

    def _summarize(
        self, results: list[VerificationResult], errors: list[float]
    ) -> VerificationSummary:
        """Create summary from verification results."""
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        # Get worst cases (top 10 by error)
        sorted_results = sorted(results, key=lambda r: r.error, reverse=True)
        worst_cases = sorted_results[:10]

        return VerificationSummary(
            total_tests=len(results),
            passed=passed,
            failed=failed,
            max_error=max(errors) if errors else 0.0,
            avg_error=sum(errors) / len(errors) if errors else 0.0,
            worst_cases=worst_cases,
        )


class BiomeVerifier:
    """
    Verifier for biome placement implementations.

    Tests biome selection against expected values for a grid of coordinates.
    """

    def __init__(
        self, temperature_tolerance: float = 1e-10, humidity_tolerance: float = 1e-10
    ) -> None:
        """
        Initialize verifier.

        Args:
            temperature_tolerance: Max error for temperature values
            humidity_tolerance: Max error for humidity values
        """
        self.temp_tolerance = temperature_tolerance
        self.humid_tolerance = humidity_tolerance

    def verify_biome_grid(
        self, seed: int, test_vectors: list[BiomeTestVector]
    ) -> VerificationSummary:
        """
        Verify biome placement against expected values.

        Args:
            seed: World seed
            test_vectors: List of biome test vectors

        Returns:
            VerificationSummary with test results
        """
        temp_rand = JavaRandom(seed * 9871)
        humid_rand = JavaRandom(seed * 39811)

        temp_noise = OctaveNoise(temp_rand, 4)
        humid_noise = OctaveNoise(humid_rand, 4)

        results = []
        errors = []

        for vec in test_vectors:
            temp = temp_noise.noise_3d(vec.x / 8.0, 0.0, vec.z / 8.0)
            humid = humid_noise.noise_3d(vec.x / 8.0, 0.0, vec.z / 8.0)

            temp = (temp + 1) / 2
            humid = (humid + 1) / 2

            actual_biome = _select_biome(temp, humid)

            temp_error = abs(temp - vec.expected_temperature)
            humid_error = abs(humid - vec.expected_humidity)
            biome_match = actual_biome == vec.expected_biome_id

            # Combined error metric
            error = max(temp_error, humid_error)
            if not biome_match:
                error = 1.0  # Maximum penalty for biome mismatch

            errors.append(error)

            passed = (
                temp_error <= self.temp_tolerance
                and humid_error <= self.humid_tolerance
                and biome_match
            )

            results.append(
                VerificationResult(
                    passed=passed,
                    test_name="biome_placement",
                    expected=vec.expected_biome_id,
                    actual=actual_biome,
                    error=error,
                    coords=(vec.x, vec.z),
                    seed=vec.seed,
                )
            )

        return self._summarize(results, errors)

    def verify_from_file(self, filepath: str | Path) -> VerificationSummary:
        """
        Verify biome placement from saved test vectors.

        Args:
            filepath: Path to JSON file with biome test vectors

        Returns:
            VerificationSummary
        """
        with open(filepath) as f:
            data = json.load(f)

        vectors = []
        for item in data:
            vectors.append(
                BiomeTestVector(
                    seed=item["seed"],
                    x=item["x"],
                    z=item["z"],
                    expected_biome_id=item["biome_id"],
                    expected_temperature=item["temperature"],
                    expected_humidity=item["humidity"],
                )
            )

        if not vectors:
            raise ValueError("No test vectors found in file")

        return self.verify_biome_grid(vectors[0].seed, vectors)

    def _summarize(
        self, results: list[VerificationResult], errors: list[float]
    ) -> VerificationSummary:
        """Create summary from verification results."""
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        sorted_results = sorted(results, key=lambda r: r.error, reverse=True)
        worst_cases = sorted_results[:10]

        return VerificationSummary(
            total_tests=len(results),
            passed=passed,
            failed=failed,
            max_error=max(errors) if errors else 0.0,
            avg_error=sum(errors) / len(errors) if errors else 0.0,
            worst_cases=worst_cases,
        )


def run_noise_verification(
    seed: int, num_tests: int = 10000, tolerance: float = 1e-10
) -> dict[str, VerificationSummary]:
    """
    Run complete noise verification for a seed.

    Args:
        seed: World seed to test
        num_tests: Number of random coordinates to test
        tolerance: Maximum allowed error

    Returns:
        Dictionary of test name to VerificationSummary
    """
    from terrain_test_generator import generate_noise_test_vectors

    verifier = NoiseVerifier(tolerance=tolerance)
    vectors = list(generate_noise_test_vectors(seed, count=num_tests))

    return {
        "simplex_2d": verifier.verify_simplex_2d(seed, vectors),
        "perlin_3d": verifier.verify_perlin_3d(seed, vectors),
    }


def run_biome_verification(seed: int, grid_size: int = 500, step: int = 16) -> VerificationSummary:
    """
    Run biome placement verification for a seed.

    Args:
        seed: World seed to test
        grid_size: Half-size of verification grid
        step: Step between test points

    Returns:
        VerificationSummary
    """
    from terrain_test_generator import generate_biome_test_grid

    verifier = BiomeVerifier()
    vectors = list(
        generate_biome_test_grid(
            seed, min_x=-grid_size, max_x=grid_size, min_z=-grid_size, max_z=grid_size, step=step
        )
    )

    return verifier.verify_biome_grid(seed, vectors)


def print_summary(name: str, summary: VerificationSummary) -> None:
    """Print formatted verification summary."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Total tests: {summary.total_tests}")
    print(f"  Passed:      {summary.passed} ({100 * summary.passed / summary.total_tests:.2f}%)")
    print(f"  Failed:      {summary.failed}")
    print(f"  Max error:   {summary.max_error:.2e}")
    print(f"  Avg error:   {summary.avg_error:.2e}")

    if summary.failed > 0:
        print("\n  Worst cases:")
        for i, case in enumerate(summary.worst_cases[:5]):
            print(
                f"    {i + 1}. coords={case.coords} expected={case.expected} "
                f"actual={case.actual} error={case.error:.2e}"
            )


if __name__ == "__main__":
    import sys

import logging

logger = logging.getLogger(__name__)

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 12345

    print(f"Running terrain verification for seed {seed}")

    # Noise verification
    noise_results = run_noise_verification(seed, num_tests=10000)
    for name, summary in noise_results.items():
        print_summary(f"Noise: {name}", summary)

    # Biome verification
    biome_summary = run_biome_verification(seed)
    print_summary("Biome Placement", biome_summary)

    # Overall result
    all_passed = all(s.failed == 0 for s in noise_results.values())
    all_passed = all_passed and biome_summary.failed == 0

    print(f"\n{'=' * 60}")
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print(f"{'=' * 60}")

    sys.exit(0 if all_passed else 1)
