"""Look direction test case generator.

Generates test cases for verifying yaw/pitch to unit vector conversion.
Includes cardinal directions, pitch extremes, random combinations, and edge cases.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestCase:
    """A single test case for look direction verification."""

    yaw: float  # degrees
    pitch: float  # degrees
    expected_x: float
    expected_y: float
    expected_z: float
    category: str
    description: str


def yaw_pitch_to_unit_vector(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    """Convert yaw/pitch angles to a unit vector.

    Convention:
    - Yaw 0 = looking along +X axis
    - Yaw 90 = looking along +Z axis
    - Pitch 0 = horizontal
    - Pitch +90 = looking straight up (+Y)
    - Pitch -90 = looking straight down (-Y)

    Args:
        yaw_deg: Yaw angle in degrees (horizontal rotation).
        pitch_deg: Pitch angle in degrees (vertical rotation).

    Returns:
        Tuple (x, y, z) representing the unit direction vector.
    """
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)

    cos_pitch = math.cos(pitch_rad)
    x = cos_pitch * math.cos(yaw_rad)
    y = math.sin(pitch_rad)
    z = cos_pitch * math.sin(yaw_rad)

    return (x, y, z)


def generate_cardinal_direction_tests() -> list[TestCase]:
    """Generate tests for cardinal directions (yaw 0, 90, 180, 270 with pitch 0)."""
    cases = []

    # Yaw 0, pitch 0 -> +X direction
    x, y, z = yaw_pitch_to_unit_vector(0, 0)
    cases.append(
        TestCase(
            yaw=0,
            pitch=0,
            expected_x=x,
            expected_y=y,
            expected_z=z,
            category="cardinal",
            description="Yaw 0 (East, +X direction)",
        )
    )

    # Yaw 90, pitch 0 -> +Z direction
    x, y, z = yaw_pitch_to_unit_vector(90, 0)
    cases.append(
        TestCase(
            yaw=90,
            pitch=0,
            expected_x=x,
            expected_y=y,
            expected_z=z,
            category="cardinal",
            description="Yaw 90 (South, +Z direction)",
        )
    )

    # Yaw 180, pitch 0 -> -X direction
    x, y, z = yaw_pitch_to_unit_vector(180, 0)
    cases.append(
        TestCase(
            yaw=180,
            pitch=0,
            expected_x=x,
            expected_y=y,
            expected_z=z,
            category="cardinal",
            description="Yaw 180 (West, -X direction)",
        )
    )

    # Yaw 270, pitch 0 -> -Z direction
    x, y, z = yaw_pitch_to_unit_vector(270, 0)
    cases.append(
        TestCase(
            yaw=270,
            pitch=0,
            expected_x=x,
            expected_y=y,
            expected_z=z,
            category="cardinal",
            description="Yaw 270 (North, -Z direction)",
        )
    )

    return cases


def generate_pitch_extreme_tests() -> list[TestCase]:
    """Generate tests for pitch extremes (-90 and +90)."""
    cases = []

    # Pitch -90 -> looking straight down (-Y)
    x, y, z = yaw_pitch_to_unit_vector(0, -90)
    cases.append(
        TestCase(
            yaw=0,
            pitch=-90,
            expected_x=x,
            expected_y=y,
            expected_z=z,
            category="pitch_extreme",
            description="Pitch -90 (looking straight down)",
        )
    )

    # Pitch +90 -> looking straight up (+Y)
    x, y, z = yaw_pitch_to_unit_vector(0, 90)
    cases.append(
        TestCase(
            yaw=0,
            pitch=90,
            expected_x=x,
            expected_y=y,
            expected_z=z,
            category="pitch_extreme",
            description="Pitch +90 (looking straight up)",
        )
    )

    # Also test with different yaw values (should not matter at pitch extremes)
    for yaw in [45, 90, 135, 180, 225, 270, 315]:
        x_up, y_up, z_up = yaw_pitch_to_unit_vector(yaw, 90)
        cases.append(
            TestCase(
                yaw=yaw,
                pitch=90,
                expected_x=x_up,
                expected_y=y_up,
                expected_z=z_up,
                category="pitch_extreme",
                description=f"Pitch +90 with yaw {yaw} (should point up regardless)",
            )
        )

        x_down, y_down, z_down = yaw_pitch_to_unit_vector(yaw, -90)
        cases.append(
            TestCase(
                yaw=yaw,
                pitch=-90,
                expected_x=x_down,
                expected_y=y_down,
                expected_z=z_down,
                category="pitch_extreme",
                description=f"Pitch -90 with yaw {yaw} (should point down regardless)",
            )
        )

    return cases


def generate_random_tests(count: int = 5000, seed: int = 42) -> list[TestCase]:
    """Generate random yaw/pitch combinations.

    Args:
        count: Number of random test cases to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of random test cases.
    """
    random.seed(seed)
    cases = []

    for i in range(count):
        yaw = random.uniform(0, 360)
        pitch = random.uniform(-90, 90)
        x, y, z = yaw_pitch_to_unit_vector(yaw, pitch)

        cases.append(
            TestCase(
                yaw=yaw,
                pitch=pitch,
                expected_x=x,
                expected_y=y,
                expected_z=z,
                category="random",
                description=f"Random case {i + 1}",
            )
        )

    return cases


def generate_edge_case_tests() -> list[TestCase]:
    """Generate edge case tests (yaw > 360, pitch clamping)."""
    cases = []

    # Yaw > 360 (should wrap)
    for yaw in [360, 450, 720, 810, -90, -180, -270, -360]:
        # Normalize yaw to [0, 360)
        normalized_yaw = yaw % 360
        x, y, z = yaw_pitch_to_unit_vector(yaw, 0)
        cases.append(
            TestCase(
                yaw=yaw,
                pitch=0,
                expected_x=x,
                expected_y=y,
                expected_z=z,
                category="edge_yaw_wrap",
                description=f"Yaw {yaw} (wraps to {normalized_yaw})",
            )
        )

    # Pitch clamping edge cases (values near or at boundaries)
    for pitch in [-90, -89.999, -89.9, 89.9, 89.999, 90]:
        x, y, z = yaw_pitch_to_unit_vector(0, pitch)
        cases.append(
            TestCase(
                yaw=0,
                pitch=pitch,
                expected_x=x,
                expected_y=y,
                expected_z=z,
                category="edge_pitch_boundary",
                description=f"Pitch {pitch} (near/at boundary)",
            )
        )

    # Pitch beyond clamping range (implementation should clamp to [-90, 90])
    # Note: These test what happens with invalid input
    for pitch in [-91, -100, -180, 91, 100, 180]:
        # The math will still work, but results may be unexpected
        # This tests robustness of the implementation
        x, y, z = yaw_pitch_to_unit_vector(0, pitch)
        cases.append(
            TestCase(
                yaw=0,
                pitch=pitch,
                expected_x=x,
                expected_y=y,
                expected_z=z,
                category="edge_pitch_out_of_range",
                description=f"Pitch {pitch} (out of valid range, tests clamping behavior)",
            )
        )

    # Combined edge cases
    edge_yaws = [0, 360, -360, 720]
    edge_pitches = [-90, 90, 0]
    for yaw in edge_yaws:
        for pitch in edge_pitches:
            x, y, z = yaw_pitch_to_unit_vector(yaw, pitch)
            cases.append(
                TestCase(
                    yaw=yaw,
                    pitch=pitch,
                    expected_x=x,
                    expected_y=y,
                    expected_z=z,
                    category="edge_combined",
                    description=f"Combined edge: yaw={yaw}, pitch={pitch}",
                )
            )

    # Floating point precision edge cases
    for yaw in [0.0000001, 359.9999999, 180.0000001]:
        x, y, z = yaw_pitch_to_unit_vector(yaw, 0)
        cases.append(
            TestCase(
                yaw=yaw,
                pitch=0,
                expected_x=x,
                expected_y=y,
                expected_z=z,
                category="edge_precision",
                description=f"Yaw precision test: {yaw}",
            )
        )

    return cases


def generate_all_test_cases() -> list[TestCase]:
    """Generate all test cases."""
    cases = []
    cases.extend(generate_cardinal_direction_tests())
    cases.extend(generate_pitch_extreme_tests())
    cases.extend(generate_random_tests(count=5000))
    cases.extend(generate_edge_case_tests())
    return cases


def save_test_cases(cases: list[TestCase], output_path: Path) -> None:
    """Save test cases to JSON file.

    Args:
        cases: List of test cases to save.
        output_path: Path to output JSON file.
    """
    data = {
        "metadata": {
            "total_cases": len(cases),
            "tolerance": 1e-5,
            "categories": {
                "cardinal": sum(1 for c in cases if c.category == "cardinal"),
                "pitch_extreme": sum(1 for c in cases if c.category == "pitch_extreme"),
                "random": sum(1 for c in cases if c.category == "random"),
                "edge_yaw_wrap": sum(1 for c in cases if c.category == "edge_yaw_wrap"),
                "edge_pitch_boundary": sum(1 for c in cases if c.category == "edge_pitch_boundary"),
                "edge_pitch_out_of_range": sum(
                    1 for c in cases if c.category == "edge_pitch_out_of_range"
                ),
                "edge_combined": sum(1 for c in cases if c.category == "edge_combined"),
                "edge_precision": sum(1 for c in cases if c.category == "edge_precision"),
            },
        },
        "test_cases": [
            {
                "yaw": c.yaw,
                "pitch": c.pitch,
                "expected": {
                    "x": c.expected_x,
                    "y": c.expected_y,
                    "z": c.expected_z,
                },
                "category": c.category,
                "description": c.description,
            }
            for c in cases
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(cases)} test cases to {output_path}")
    print("Categories:")
    for cat, count in data["metadata"]["categories"].items():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    cases = generate_all_test_cases()
    output_path = Path(__file__).parent / "look_direction_test_cases.json"
    save_test_cases(cases, output_path)
