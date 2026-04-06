"""Damage calculation test case generator.

Generates full combinatorial test matrix for damage calculation verification.

Test matrix dimensions:
- Raw damage: [1, 2, 4, 6, 8, 10, 15, 20]
- Armor values: [0, 2, 4, 8, 12, 16, 20]
- Protection levels: [0, 1, 2, 3, 4]
- Resistance levels: [0, 1, 2]

Total combinations: 8 * 7 * 5 * 3 = 840 test cases
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from pathlib import Path

# Test matrix values
RAW_DAMAGE_VALUES = [1, 2, 4, 6, 8, 10, 15, 20]
ARMOR_VALUES = [0, 2, 4, 8, 12, 16, 20]
PROTECTION_LEVELS = [0, 1, 2, 3, 4]
RESISTANCE_LEVELS = [0, 1, 2]


@dataclass(frozen=True)
class DamageTestCase:
    """Single damage calculation test case."""

    raw_damage: int
    armor: int
    protection_level: int
    resistance_level: int

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "raw_damage": self.raw_damage,
            "armor": self.armor,
            "protection_level": self.protection_level,
            "resistance_level": self.resistance_level,
        }

    @property
    def test_id(self) -> str:
        """Unique identifier for this test case."""
        return f"d{self.raw_damage}_a{self.armor}_p{self.protection_level}_r{self.resistance_level}"


def generate_test_matrix() -> Iterator[DamageTestCase]:
    """Generate all combinatorial test cases.

    Yields:
        DamageTestCase for each combination in the test matrix.
    """
    for raw_damage, armor, protection, resistance in product(
        RAW_DAMAGE_VALUES,
        ARMOR_VALUES,
        PROTECTION_LEVELS,
        RESISTANCE_LEVELS,
    ):
        yield DamageTestCase(
            raw_damage=raw_damage,
            armor=armor,
            protection_level=protection,
            resistance_level=resistance,
        )


def count_test_cases() -> int:
    """Calculate total number of test cases in the matrix."""
    return (
        len(RAW_DAMAGE_VALUES) * len(ARMOR_VALUES) * len(PROTECTION_LEVELS) * len(RESISTANCE_LEVELS)
    )


def export_test_cases(output_path: Path | str) -> int:
    """Export all test cases to JSON file.

    Args:
        output_path: Path to write JSON output.

    Returns:
        Number of test cases exported.
    """
    output_path = Path(output_path)
    test_cases = [tc.to_dict() for tc in generate_test_matrix()]

    output_data = {
        "matrix_dimensions": {
            "raw_damage": RAW_DAMAGE_VALUES,
            "armor": ARMOR_VALUES,
            "protection_levels": PROTECTION_LEVELS,
            "resistance_levels": RESISTANCE_LEVELS,
        },
        "total_cases": len(test_cases),
        "test_cases": test_cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    return len(test_cases)


def generate_pytest_parametrize() -> str:
    """Generate pytest parametrize decorator string.

    Returns:
        Python code string with pytest.mark.parametrize decorator.
    """
    cases = list(generate_test_matrix())
    params = ", ".join(
        f"({tc.raw_damage}, {tc.armor}, {tc.protection_level}, {tc.resistance_level})"
        for tc in cases
    )
    return f'@pytest.mark.parametrize("raw_damage, armor, protection, resistance", [{params}])'


if __name__ == "__main__":
    print("Test matrix dimensions:")
    print(f"  Raw damage values: {RAW_DAMAGE_VALUES}")
    print(f"  Armor values: {ARMOR_VALUES}")
    print(f"  Protection levels: {PROTECTION_LEVELS}")
    print(f"  Resistance levels: {RESISTANCE_LEVELS}")
    print(f"\nTotal test cases: {count_test_cases()}")

    output_file = Path(__file__).parent / "test_cases.json"
    exported = export_test_cases(output_file)
    print(f"\nExported {exported} test cases to {output_file}")
