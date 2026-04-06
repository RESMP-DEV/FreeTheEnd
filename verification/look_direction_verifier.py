"""Look direction verification system.

Verifies that a look direction implementation correctly converts yaw/pitch
angles to unit vectors within the specified tolerance (1e-5 per component).
"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Tolerance for floating point comparison
TOLERANCE = 1e-5


class LookDirectionFunction(Protocol):
    """Protocol for look direction conversion functions."""

    def __call__(self, yaw: float, pitch: float) -> tuple[float, float, float]:
        """Convert yaw/pitch to unit vector.

        Args:
            yaw: Yaw angle in degrees.
            pitch: Pitch angle in degrees.

        Returns:
            Tuple (x, y, z) representing unit direction vector.
        """
        ...


@dataclass
class VerificationResult:
    """Result of a single test case verification."""

    passed: bool
    yaw: float
    pitch: float
    expected: tuple[float, float, float]
    actual: tuple[float, float, float]
    error: tuple[float, float, float]
    max_error: float
    category: str
    description: str


@dataclass
class VerificationSummary:
    """Summary of all verification results."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    max_error_seen: float
    worst_case: VerificationResult | None
    failures_by_category: dict[str, int]
    results: list[VerificationResult]


def reference_implementation(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    """Reference implementation for yaw/pitch to unit vector conversion.

    Convention:
    - Yaw 0 = looking along +X axis
    - Yaw 90 = looking along +Z axis
    - Pitch 0 = horizontal
    - Pitch +90 = looking straight up (+Y)
    - Pitch -90 = looking straight down (-Y)

    Args:
        yaw_deg: Yaw angle in degrees.
        pitch_deg: Pitch angle in degrees.

    Returns:
        Unit vector (x, y, z).
    """
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)

    cos_pitch = math.cos(pitch_rad)
    x = cos_pitch * math.cos(yaw_rad)
    y = math.sin(pitch_rad)
    z = cos_pitch * math.sin(yaw_rad)

    return (x, y, z)


def verify_single_case(
    func: LookDirectionFunction,
    yaw: float,
    pitch: float,
    expected: tuple[float, float, float],
    category: str,
    description: str,
) -> VerificationResult:
    """Verify a single test case.

    Args:
        func: Function to test.
        yaw: Input yaw angle.
        pitch: Input pitch angle.
        expected: Expected (x, y, z) output.
        category: Test category for reporting.
        description: Human-readable description.

    Returns:
        VerificationResult with pass/fail status and error details.
    """
    actual = func(yaw, pitch)

    error = (
        abs(actual[0] - expected[0]),
        abs(actual[1] - expected[1]),
        abs(actual[2] - expected[2]),
    )
    max_error = max(error)
    passed = max_error <= TOLERANCE

    return VerificationResult(
        passed=passed,
        yaw=yaw,
        pitch=pitch,
        expected=expected,
        actual=actual,
        error=error,
        max_error=max_error,
        category=category,
        description=description,
    )


def verify_from_json(
    func: LookDirectionFunction,
    json_path: Path,
) -> VerificationSummary:
    """Verify implementation against test cases from JSON file.

    Args:
        func: Function to test.
        json_path: Path to JSON file with test cases.

    Returns:
        VerificationSummary with all results.
    """
    with open(json_path) as f:
        data = json.load(f)

    results = []
    worst_case = None
    max_error_seen = 0.0

    for case in data["test_cases"]:
        result = verify_single_case(
            func=func,
            yaw=case["yaw"],
            pitch=case["pitch"],
            expected=(case["expected"]["x"], case["expected"]["y"], case["expected"]["z"]),
            category=case["category"],
            description=case["description"],
        )
        results.append(result)

        if result.max_error > max_error_seen:
            max_error_seen = result.max_error
            worst_case = result

    passed_cases = sum(1 for r in results if r.passed)
    failed_cases = len(results) - passed_cases

    failures_by_category: dict[str, int] = {}
    for r in results:
        if not r.passed:
            failures_by_category[r.category] = failures_by_category.get(r.category, 0) + 1

    return VerificationSummary(
        total_cases=len(results),
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=passed_cases / len(results) if results else 0.0,
        max_error_seen=max_error_seen,
        worst_case=worst_case,
        failures_by_category=failures_by_category,
        results=results,
    )


def verify_implementation(
    func: LookDirectionFunction,
    num_random: int = 5000,
    seed: int = 42,
) -> VerificationSummary:
    """Verify implementation with generated test cases.

    Args:
        func: Function to test.
        num_random: Number of random test cases.
        seed: Random seed for reproducibility.

    Returns:
        VerificationSummary with all results.
    """
    import random

    random.seed(seed)

    results = []

    # Cardinal directions
    cardinal_cases = [
        (0, 0, "Yaw 0 (East, +X)"),
        (90, 0, "Yaw 90 (South, +Z)"),
        (180, 0, "Yaw 180 (West, -X)"),
        (270, 0, "Yaw 270 (North, -Z)"),
    ]
    for yaw, pitch, desc in cardinal_cases:
        expected = reference_implementation(yaw, pitch)
        result = verify_single_case(func, yaw, pitch, expected, "cardinal", desc)
        results.append(result)

    # Pitch extremes
    pitch_extreme_cases = [
        (0, -90, "Looking straight down"),
        (0, 90, "Looking straight up"),
    ]
    for yaw in [0, 45, 90, 135, 180, 225, 270, 315]:
        pitch_extreme_cases.append((yaw, 90, f"Pitch +90 with yaw {yaw}"))
        pitch_extreme_cases.append((yaw, -90, f"Pitch -90 with yaw {yaw}"))

    for yaw, pitch, desc in pitch_extreme_cases:
        expected = reference_implementation(yaw, pitch)
        result = verify_single_case(func, yaw, pitch, expected, "pitch_extreme", desc)
        results.append(result)

    # Random cases
    for i in range(num_random):
        yaw = random.uniform(0, 360)
        pitch = random.uniform(-90, 90)
        expected = reference_implementation(yaw, pitch)
        result = verify_single_case(func, yaw, pitch, expected, "random", f"Random case {i + 1}")
        results.append(result)

    # Edge cases
    edge_cases = []

    # Yaw wrapping
    for yaw in [360, 450, 720, 810, -90, -180, -270, -360]:
        edge_cases.append((yaw, 0, "edge_yaw_wrap", f"Yaw {yaw}"))

    # Pitch boundaries
    for pitch in [-90, -89.999, -89.9, 89.9, 89.999, 90]:
        edge_cases.append((0, pitch, "edge_pitch_boundary", f"Pitch {pitch}"))

    # Pitch out of range
    for pitch in [-91, -100, -180, 91, 100, 180]:
        edge_cases.append((0, pitch, "edge_pitch_out_of_range", f"Pitch {pitch}"))

    # Combined edge cases
    for yaw in [0, 360, -360, 720]:
        for pitch in [-90, 90, 0]:
            edge_cases.append((yaw, pitch, "edge_combined", f"yaw={yaw}, pitch={pitch}"))

    # Precision edge cases
    for yaw in [0.0000001, 359.9999999, 180.0000001]:
        edge_cases.append((yaw, 0, "edge_precision", f"Yaw {yaw}"))

    for yaw, pitch, category, desc in edge_cases:
        expected = reference_implementation(yaw, pitch)
        result = verify_single_case(func, yaw, pitch, expected, category, desc)
        results.append(result)

    # Compute summary
    passed_cases = sum(1 for r in results if r.passed)
    failed_cases = len(results) - passed_cases

    worst_case = None
    max_error_seen = 0.0
    for r in results:
        if r.max_error > max_error_seen:
            max_error_seen = r.max_error
            worst_case = r

    failures_by_category: dict[str, int] = {}
    for r in results:
        if not r.passed:
            failures_by_category[r.category] = failures_by_category.get(r.category, 0) + 1

    return VerificationSummary(
        total_cases=len(results),
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=passed_cases / len(results) if results else 0.0,
        max_error_seen=max_error_seen,
        worst_case=worst_case,
        failures_by_category=failures_by_category,
        results=results,
    )


def print_summary(summary: VerificationSummary, verbose: bool = False) -> None:
    """Print verification summary.

    Args:
        summary: Verification summary to print.
        verbose: If True, print all failed cases.
    """
    print("=" * 60)
    print("LOOK DIRECTION VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total test cases: {summary.total_cases}")
    print(f"Passed: {summary.passed_cases}")
    print(f"Failed: {summary.failed_cases}")
    print(f"Pass rate: {summary.pass_rate * 100:.4f}%")
    print(f"Maximum error seen: {summary.max_error_seen:.2e}")
    print(f"Tolerance: {TOLERANCE:.0e}")
    print()

    if summary.worst_case:
        wc = summary.worst_case
        print("Worst case:")
        print(f"  Yaw: {wc.yaw}, Pitch: {wc.pitch}")
        print(f"  Expected: ({wc.expected[0]:.10f}, {wc.expected[1]:.10f}, {wc.expected[2]:.10f})")
        print(f"  Actual:   ({wc.actual[0]:.10f}, {wc.actual[1]:.10f}, {wc.actual[2]:.10f})")
        print(f"  Error:    ({wc.error[0]:.2e}, {wc.error[1]:.2e}, {wc.error[2]:.2e})")
        print(f"  Category: {wc.category}")
        print(f"  Description: {wc.description}")
        print()

    if summary.failures_by_category:
        print("Failures by category:")
        for cat, count in sorted(summary.failures_by_category.items()):
            print(f"  {cat}: {count}")
        print()

    if verbose and summary.failed_cases > 0:
        print("All failed cases:")
        for r in summary.results:
            if not r.passed:
                print(f"  [{r.category}] yaw={r.yaw:.6f}, pitch={r.pitch:.6f}")
                print(
                    f"    Expected: ({r.expected[0]:.10f}, {r.expected[1]:.10f}, {r.expected[2]:.10f})"
                )
                print(f"    Actual:   ({r.actual[0]:.10f}, {r.actual[1]:.10f}, {r.actual[2]:.10f})")
                print(f"    Max error: {r.max_error:.2e}")

    print("=" * 60)
    if summary.failed_cases == 0:
        print("VERIFICATION PASSED")
    else:
        print("VERIFICATION FAILED")
    print("=" * 60)


def main() -> int:
    """Run verification against reference implementation.

    Returns:
        0 if all tests pass, 1 otherwise.
    """
    print("Running look direction verification...")
    print("Testing reference implementation against itself (sanity check)")
    print()

    summary = verify_implementation(reference_implementation)
    print_summary(summary)

    return 0 if summary.failed_cases == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
