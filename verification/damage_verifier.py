"""Damage calculation verifier with exact float comparison.

Implements damage formula verification against reference implementation.

Damage formula (reference):
    effective_armor = armor * (1 - protection_level * 0.2)
    damage_reduction = effective_armor / (effective_armor + 10)
    base_damage = raw_damage * (1 - damage_reduction)
    resistance_multiplier = 1 - resistance_level * 0.25
    final_damage = base_damage * resistance_multiplier

All comparisons use exact float equality (==), not approximate.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from damage_test_generator import (
    ARMOR_VALUES,
    PROTECTION_LEVELS,
    RAW_DAMAGE_VALUES,
    RESISTANCE_LEVELS,
    DamageTestCase,
    generate_test_matrix,
)


@dataclass
class DamageResult:
    """Result of a damage calculation."""

    effective_armor: float
    damage_reduction: float
    base_damage: float
    resistance_multiplier: float
    final_damage: float


@dataclass
class VerificationResult:
    """Result of verifying a single test case."""

    test_case: DamageTestCase
    expected: DamageResult
    actual: DamageResult
    passed: bool
    error_details: str | None = None


def reference_damage_calculation(
    raw_damage: int,
    armor: int,
    protection_level: int,
    resistance_level: int,
) -> DamageResult:
    """Reference implementation of damage calculation.

    This is the canonical formula that all implementations must match exactly.

    Args:
        raw_damage: Base damage value before any mitigation.
        armor: Target's armor value.
        protection_level: Protection level (0-4), each level reduces armor effectiveness by 20%.
        resistance_level: Resistance level (0-2), each level reduces damage by 25%.

    Returns:
        DamageResult with all intermediate and final values.
    """
    # Step 1: Calculate effective armor (protection reduces armor effectiveness)
    effective_armor = armor * (1.0 - protection_level * 0.2)

    # Step 2: Calculate damage reduction from armor (diminishing returns formula)
    if effective_armor <= 0.0:
        damage_reduction = 0.0
    else:
        damage_reduction = effective_armor / (effective_armor + 10.0)

    # Step 3: Apply armor reduction to get base damage
    base_damage = raw_damage * (1.0 - damage_reduction)

    # Step 4: Calculate resistance multiplier
    resistance_multiplier = 1.0 - resistance_level * 0.25

    # Step 5: Apply resistance to get final damage
    final_damage = base_damage * resistance_multiplier

    return DamageResult(
        effective_armor=effective_armor,
        damage_reduction=damage_reduction,
        base_damage=base_damage,
        resistance_multiplier=resistance_multiplier,
        final_damage=final_damage,
    )


class DamageVerifier:
    """Verifies damage calculation implementations against reference."""

    def __init__(
        self,
        implementation: Callable[[int, int, int, int], DamageResult] | None = None,
    ):
        """Initialize verifier.

        Args:
            implementation: Function to verify. If None, uses reference implementation
                           (useful for generating expected values).
        """
        self.implementation = implementation or reference_damage_calculation
        self.results: list[VerificationResult] = []

    def verify_single(self, test_case: DamageTestCase) -> VerificationResult:
        """Verify a single test case.

        Args:
            test_case: Test case to verify.

        Returns:
            VerificationResult with pass/fail status and details.
        """
        expected = reference_damage_calculation(
            test_case.raw_damage,
            test_case.armor,
            test_case.protection_level,
            test_case.resistance_level,
        )

        actual = self.implementation(
            test_case.raw_damage,
            test_case.armor,
            test_case.protection_level,
            test_case.resistance_level,
        )

        # Exact float comparison
        passed = (
            expected.effective_armor == actual.effective_armor
            and expected.damage_reduction == actual.damage_reduction
            and expected.base_damage == actual.base_damage
            and expected.resistance_multiplier == actual.resistance_multiplier
            and expected.final_damage == actual.final_damage
        )

        error_details = None
        if not passed:
            errors = []
            if expected.effective_armor != actual.effective_armor:
                errors.append(
                    f"effective_armor: expected {expected.effective_armor!r}, "
                    f"got {actual.effective_armor!r}"
                )
            if expected.damage_reduction != actual.damage_reduction:
                errors.append(
                    f"damage_reduction: expected {expected.damage_reduction!r}, "
                    f"got {actual.damage_reduction!r}"
                )
            if expected.base_damage != actual.base_damage:
                errors.append(
                    f"base_damage: expected {expected.base_damage!r}, got {actual.base_damage!r}"
                )
            if expected.resistance_multiplier != actual.resistance_multiplier:
                errors.append(
                    f"resistance_multiplier: expected {expected.resistance_multiplier!r}, "
                    f"got {actual.resistance_multiplier!r}"
                )
            if expected.final_damage != actual.final_damage:
                errors.append(
                    f"final_damage: expected {expected.final_damage!r}, got {actual.final_damage!r}"
                )
            error_details = "; ".join(errors)

        result = VerificationResult(
            test_case=test_case,
            expected=expected,
            actual=actual,
            passed=passed,
            error_details=error_details,
        )
        self.results.append(result)
        return result

    def verify_all(self) -> tuple[int, int]:
        """Run full verification against test matrix.

        Returns:
            Tuple of (passed_count, failed_count).
        """
        self.results.clear()
        passed = 0
        failed = 0

        for test_case in generate_test_matrix():
            result = self.verify_single(test_case)
            if result.passed:
                passed += 1
            else:
                failed += 1

        return passed, failed

    def get_failures(self) -> list[VerificationResult]:
        """Get all failed verification results."""
        return [r for r in self.results if not r.passed]

    def generate_expected_values(self, output_path: Path | str) -> int:
        """Generate JSON file with expected values for all test cases.

        Args:
            output_path: Path to write JSON output.

        Returns:
            Number of expected values generated.
        """
        output_path = Path(output_path)
        expected_values = []

        for test_case in generate_test_matrix():
            result = reference_damage_calculation(
                test_case.raw_damage,
                test_case.armor,
                test_case.protection_level,
                test_case.resistance_level,
            )
            expected_values.append(
                {
                    "test_id": test_case.test_id,
                    "inputs": test_case.to_dict(),
                    "expected": {
                        "effective_armor": result.effective_armor,
                        "damage_reduction": result.damage_reduction,
                        "base_damage": result.base_damage,
                        "resistance_multiplier": result.resistance_multiplier,
                        "final_damage": result.final_damage,
                    },
                }
            )

        output_data = {
            "formula": {
                "effective_armor": "armor * (1 - protection_level * 0.2)",
                "damage_reduction": "effective_armor / (effective_armor + 10) if effective_armor > 0 else 0",
                "base_damage": "raw_damage * (1 - damage_reduction)",
                "resistance_multiplier": "1 - resistance_level * 0.25",
                "final_damage": "base_damage * resistance_multiplier",
            },
            "matrix_dimensions": {
                "raw_damage": RAW_DAMAGE_VALUES,
                "armor": ARMOR_VALUES,
                "protection_levels": PROTECTION_LEVELS,
                "resistance_levels": RESISTANCE_LEVELS,
            },
            "total_cases": len(expected_values),
            "expected_values": expected_values,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        return len(expected_values)


def run_verification_report() -> None:
    """Run full verification and print detailed report."""
    verifier = DamageVerifier()
    passed, failed = verifier.verify_all()

    print("=" * 60)
    print("DAMAGE CALCULATION VERIFICATION REPORT")
    print("=" * 60)
    print("\nTest Matrix:")
    print(f"  Raw damage: {RAW_DAMAGE_VALUES}")
    print(f"  Armor: {ARMOR_VALUES}")
    print(f"  Protection levels: {PROTECTION_LEVELS}")
    print(f"  Resistance levels: {RESISTANCE_LEVELS}")
    print(f"\nTotal combinations: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print(f"\n{'=' * 60}")
        print("FAILURES:")
        print("=" * 60)
        for result in verifier.get_failures():
            tc = result.test_case
            print(f"\n[{tc.test_id}]")
            print(
                f"  Inputs: raw={tc.raw_damage}, armor={tc.armor}, "
                f"prot={tc.protection_level}, resist={tc.resistance_level}"
            )
            print(f"  Error: {result.error_details}")

    print(f"\n{'=' * 60}")
    print(f"RESULT: {'PASS' if failed == 0 else 'FAIL'}")
    print("=" * 60)


def verify_against_json(json_path: Path | str, implementation: Callable) -> tuple[int, int]:
    """Verify an implementation against pre-generated expected values.

    Args:
        json_path: Path to JSON file with expected values.
        implementation: Function to verify.

    Returns:
        Tuple of (passed_count, failed_count).
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)

    passed = 0
    failed = 0
    failures = []

    for entry in data["expected_values"]:
        inputs = entry["inputs"]
        expected = entry["expected"]

        actual = implementation(
            inputs["raw_damage"],
            inputs["armor"],
            inputs["protection_level"],
            inputs["resistance_level"],
        )

        # Exact float comparison
        if (
            expected["effective_armor"] == actual.effective_armor
            and expected["damage_reduction"] == actual.damage_reduction
            and expected["base_damage"] == actual.base_damage
            and expected["resistance_multiplier"] == actual.resistance_multiplier
            and expected["final_damage"] == actual.final_damage
        ):
            passed += 1
        else:
            failed += 1
            failures.append((entry["test_id"], expected, actual))

    return passed, failed


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        output_file = Path(__file__).parent / "expected_values.json"
        verifier = DamageVerifier()
        count = verifier.generate_expected_values(output_file)
        print(f"Generated {count} expected values to {output_file}")
    else:
        run_verification_report()
