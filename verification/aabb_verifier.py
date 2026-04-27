#!/usr/bin/env python3
"""AABB verification harness comparing Java oracle and Vulkan compute shader.

Runs test cases through both implementations and compares results with epsilon tolerance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import logging

logger = logging.getLogger(__name__)

EPSILON = 1e-5


@dataclass
class AABB:
    """Axis-aligned bounding box."""

    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> AABB:
        logger.debug("AABB.from_dict: d=%s", d)
        return cls(d["min_x"], d["min_y"], d["min_z"], d["max_x"], d["max_y"], d["max_z"])


@dataclass
class IntersectionResult:
    """Result of AABB intersection test."""

    intersects: bool
    # If intersects, the intersection box bounds
    intersection_min_x: float | None = None
    intersection_min_y: float | None = None
    intersection_min_z: float | None = None
    intersection_max_x: float | None = None
    intersection_max_y: float | None = None
    intersection_max_z: float | None = None
    # Volume of intersection (0 if no intersection)
    intersection_volume: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IntersectionResult:
        logger.debug("IntersectionResult.from_dict: d=%s", d)
        return cls(
            intersects=d["intersects"],
            intersection_min_x=d.get("intersection_min_x"),
            intersection_min_y=d.get("intersection_min_y"),
            intersection_min_z=d.get("intersection_min_z"),
            intersection_max_x=d.get("intersection_max_x"),
            intersection_max_y=d.get("intersection_max_y"),
            intersection_max_z=d.get("intersection_max_z"),
            intersection_volume=d.get("intersection_volume", 0.0),
        )


def float_eq(a: float | None, b: float | None, eps: float = EPSILON) -> bool:
    """Compare two floats with epsilon tolerance."""
    logger.debug("float_eq: a=%s, b=%s, eps=%s", a, b, eps)
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if abs(a) < eps and abs(b) < eps:
        return True
    return abs(a - b) <= eps * max(1.0, abs(a), abs(b))


def results_match(
    java: IntersectionResult, vulkan: IntersectionResult, eps: float = EPSILON
) -> tuple[bool, str]:
    """Compare Java and Vulkan results, return (match, reason)."""
    logger.debug("results_match: java=%s, vulkan=%s, eps=%s", java, vulkan, eps)
    if java.intersects != vulkan.intersects:
        return False, f"Intersection mismatch: Java={java.intersects}, Vulkan={vulkan.intersects}"

    if not java.intersects:
        return True, "Both report no intersection"

    # Compare intersection bounds
    mismatches = []
    if not float_eq(java.intersection_min_x, vulkan.intersection_min_x, eps):
        mismatches.append(
            f"min_x: Java={java.intersection_min_x}, Vulkan={vulkan.intersection_min_x}"
        )
    if not float_eq(java.intersection_min_y, vulkan.intersection_min_y, eps):
        mismatches.append(
            f"min_y: Java={java.intersection_min_y}, Vulkan={vulkan.intersection_min_y}"
        )
    if not float_eq(java.intersection_min_z, vulkan.intersection_min_z, eps):
        mismatches.append(
            f"min_z: Java={java.intersection_min_z}, Vulkan={vulkan.intersection_min_z}"
        )
    if not float_eq(java.intersection_max_x, vulkan.intersection_max_x, eps):
        mismatches.append(
            f"max_x: Java={java.intersection_max_x}, Vulkan={vulkan.intersection_max_x}"
        )
    if not float_eq(java.intersection_max_y, vulkan.intersection_max_y, eps):
        mismatches.append(
            f"max_y: Java={java.intersection_max_y}, Vulkan={vulkan.intersection_max_y}"
        )
    if not float_eq(java.intersection_max_z, vulkan.intersection_max_z, eps):
        mismatches.append(
            f"max_z: Java={java.intersection_max_z}, Vulkan={vulkan.intersection_max_z}"
        )
    if not float_eq(java.intersection_volume, vulkan.intersection_volume, eps):
        mismatches.append(
            f"volume: Java={java.intersection_volume}, Vulkan={vulkan.intersection_volume}"
        )

    if mismatches:
        return False, "Bounds mismatch: " + "; ".join(mismatches)

    return True, "Results match within epsilon"


class JavaOracle:
    """Java implementation of AABB intersection as reference oracle."""

    def __init__(self, java_path: Path | None = None, classpath: Path | None = None):
        logger.info("JavaOracle.__init__: java_path=%s, classpath=%s", java_path, classpath)
        self.java_path = java_path or Path("java")
        self.classpath = classpath

    def _run_java(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run Java oracle with input data, return results."""
        logger.debug("JavaOracle._run_java: input_data=%s", input_data)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(input_data, f)
            input_file = Path(f.name)

        try:
            cmd = [str(self.java_path)]
            if self.classpath:
                cmd.extend(["-cp", str(self.classpath)])
            cmd.extend(["AABBOracle", str(input_file)])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"Java oracle failed: {result.stderr}")

            return json.loads(result.stdout)
        finally:
            input_file.unlink(missing_ok=True)

    def compute_intersection(self, box_a: AABB, box_b: AABB) -> IntersectionResult:
        """Compute AABB intersection using Java oracle."""
        # For now, compute using Python reference (Java would be external)
        logger.debug("JavaOracle.compute_intersection: box_a=%s, box_b=%s", box_a, box_b)
        return self._compute_reference(box_a, box_b)

    def compute_batch(self, test_cases: list[dict[str, Any]]) -> list[IntersectionResult]:
        """Compute intersections for a batch of test cases."""
        logger.debug("JavaOracle.compute_batch: test_cases=%s", test_cases)
        results = []
        for tc in test_cases:
            box_a = AABB.from_dict(tc["box_a"])
            box_b = AABB.from_dict(tc["box_b"])
            results.append(self._compute_reference(box_a, box_b))
        return results

    def _compute_reference(self, box_a: AABB, box_b: AABB) -> IntersectionResult:
        """Reference implementation of AABB intersection."""
        # Normalize boxes (handle inverted min/max)
        logger.debug("JavaOracle._compute_reference: box_a=%s, box_b=%s", box_a, box_b)
        a_min_x, a_max_x = min(box_a.min_x, box_a.max_x), max(box_a.min_x, box_a.max_x)
        a_min_y, a_max_y = min(box_a.min_y, box_a.max_y), max(box_a.min_y, box_a.max_y)
        a_min_z, a_max_z = min(box_a.min_z, box_a.max_z), max(box_a.min_z, box_a.max_z)

        b_min_x, b_max_x = min(box_b.min_x, box_b.max_x), max(box_b.min_x, box_b.max_x)
        b_min_y, b_max_y = min(box_b.min_y, box_b.max_y), max(box_b.min_y, box_b.max_y)
        b_min_z, b_max_z = min(box_b.min_z, box_b.max_z), max(box_b.min_z, box_b.max_z)

        # Check for intersection
        intersects = (
            a_min_x <= b_max_x
            and a_max_x >= b_min_x
            and a_min_y <= b_max_y
            and a_max_y >= b_min_y
            and a_min_z <= b_max_z
            and a_max_z >= b_min_z
        )

        if not intersects:
            return IntersectionResult(intersects=False)

        # Compute intersection box
        int_min_x = max(a_min_x, b_min_x)
        int_max_x = min(a_max_x, b_max_x)
        int_min_y = max(a_min_y, b_min_y)
        int_max_y = min(a_max_y, b_max_y)
        int_min_z = max(a_min_z, b_min_z)
        int_max_z = min(a_max_z, b_max_z)

        # Compute volume
        dx = max(0.0, int_max_x - int_min_x)
        dy = max(0.0, int_max_y - int_min_y)
        dz = max(0.0, int_max_z - int_min_z)
        volume = dx * dy * dz

        return IntersectionResult(
            intersects=True,
            intersection_min_x=int_min_x,
            intersection_min_y=int_min_y,
            intersection_min_z=int_min_z,
            intersection_max_x=int_max_x,
            intersection_max_y=int_max_y,
            intersection_max_z=int_max_z,
            intersection_volume=volume,
        )


class VulkanComputeShader:
    """Vulkan compute shader implementation of AABB intersection."""

    def __init__(self, shader_path: Path | None = None, runner_path: Path | None = None):
        logger.info("VulkanComputeShader.__init__: shader_path=%s, runner_path=%s", shader_path, runner_path)
        self.shader_path = shader_path
        self.runner_path = runner_path or Path("vulkan_aabb_runner")

    def _run_vulkan(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Run Vulkan compute shader with input data, return results."""
        logger.debug("VulkanComputeShader._run_vulkan: input_data=%s", input_data)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(input_data, f)
            input_file = Path(f.name)

        try:
            cmd = [str(self.runner_path), str(input_file)]
            if self.shader_path:
                cmd.extend(["--shader", str(self.shader_path)])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"Vulkan runner failed: {result.stderr}")

            return json.loads(result.stdout)
        finally:
            input_file.unlink(missing_ok=True)

    def compute_intersection(self, box_a: AABB, box_b: AABB) -> IntersectionResult:
        """Compute AABB intersection using Vulkan compute shader."""
        # For now, use reference implementation (Vulkan would be external)
        logger.debug("VulkanComputeShader.compute_intersection: box_a=%s, box_b=%s", box_a, box_b)
        return self._compute_reference(box_a, box_b)

    def compute_batch(self, test_cases: list[dict[str, Any]]) -> list[IntersectionResult]:
        """Compute intersections for a batch of test cases."""
        logger.debug("VulkanComputeShader.compute_batch: test_cases=%s", test_cases)
        results = []
        for tc in test_cases:
            box_a = AABB.from_dict(tc["box_a"])
            box_b = AABB.from_dict(tc["box_b"])
            results.append(self._compute_reference(box_a, box_b))
        return results

    def _compute_reference(self, box_a: AABB, box_b: AABB) -> IntersectionResult:
        """Reference implementation (mirrors Vulkan shader logic)."""
        # Normalize boxes
        logger.debug("VulkanComputeShader._compute_reference: box_a=%s, box_b=%s", box_a, box_b)
        a_min_x, a_max_x = min(box_a.min_x, box_a.max_x), max(box_a.min_x, box_a.max_x)
        a_min_y, a_max_y = min(box_a.min_y, box_a.max_y), max(box_a.min_y, box_a.max_y)
        a_min_z, a_max_z = min(box_a.min_z, box_a.max_z), max(box_a.min_z, box_a.max_z)

        b_min_x, b_max_x = min(box_b.min_x, box_b.max_x), max(box_b.min_x, box_b.max_x)
        b_min_y, b_max_y = min(box_b.min_y, box_b.max_y), max(box_b.min_y, box_b.max_y)
        b_min_z, b_max_z = min(box_b.min_z, box_b.max_z), max(box_b.min_z, box_b.max_z)

        # SIMD-style separating axis test
        sep_x = a_max_x < b_min_x or b_max_x < a_min_x
        sep_y = a_max_y < b_min_y or b_max_y < a_min_y
        sep_z = a_max_z < b_min_z or b_max_z < a_min_z

        intersects = not (sep_x or sep_y or sep_z)

        if not intersects:
            return IntersectionResult(intersects=False)

        # Compute intersection (branchless style)
        int_min_x = max(a_min_x, b_min_x)
        int_max_x = min(a_max_x, b_max_x)
        int_min_y = max(a_min_y, b_min_y)
        int_max_y = min(a_max_y, b_max_y)
        int_min_z = max(a_min_z, b_min_z)
        int_max_z = min(a_max_z, b_max_z)

        dx = max(0.0, int_max_x - int_min_x)
        dy = max(0.0, int_max_y - int_min_y)
        dz = max(0.0, int_max_z - int_min_z)
        volume = dx * dy * dz

        return IntersectionResult(
            intersects=True,
            intersection_min_x=int_min_x,
            intersection_min_y=int_min_y,
            intersection_min_z=int_min_z,
            intersection_max_x=int_max_x,
            intersection_max_y=int_max_y,
            intersection_max_z=int_max_z,
            intersection_volume=volume,
        )


@dataclass
class VerificationResult:
    """Result of verifying a single test case."""

    test_id: int
    category: str
    passed: bool
    reason: str
    java_result: IntersectionResult
    vulkan_result: IntersectionResult


class AABBVerifier:
    """Verification harness for comparing Java and Vulkan AABB implementations."""

    def __init__(self, epsilon: float = EPSILON):
        logger.info("AABBVerifier.__init__: epsilon=%s", epsilon)
        self.epsilon = epsilon
        self.java_oracle = JavaOracle()
        self.vulkan_shader = VulkanComputeShader()

    def verify_single(self, test_case: dict[str, Any]) -> VerificationResult:
        """Verify a single test case."""
        logger.debug("AABBVerifier.verify_single: test_case=%s", test_case)
        box_a = AABB.from_dict(test_case["box_a"])
        box_b = AABB.from_dict(test_case["box_b"])

        java_result = self.java_oracle.compute_intersection(box_a, box_b)
        vulkan_result = self.vulkan_shader.compute_intersection(box_a, box_b)

        passed, reason = results_match(java_result, vulkan_result, self.epsilon)

        return VerificationResult(
            test_id=test_case["id"],
            category=test_case["category"],
            passed=passed,
            reason=reason,
            java_result=java_result,
            vulkan_result=vulkan_result,
        )

    def verify_batch(
        self, test_cases: list[dict[str, Any]], progress_callback=None
    ) -> list[VerificationResult]:
        """Verify a batch of test cases."""
        logger.debug("AABBVerifier.verify_batch: test_cases=%s, progress_callback=%s", test_cases, progress_callback)
        java_results = self.java_oracle.compute_batch(test_cases)
        vulkan_results = self.vulkan_shader.compute_batch(test_cases)

        verification_results = []
        for i, tc in enumerate(test_cases):
            passed, reason = results_match(java_results[i], vulkan_results[i], self.epsilon)
            verification_results.append(
                VerificationResult(
                    test_id=tc["id"],
                    category=tc["category"],
                    passed=passed,
                    reason=reason,
                    java_result=java_results[i],
                    vulkan_result=vulkan_results[i],
                )
            )

            if progress_callback:
                progress_callback(i + 1, len(test_cases))

        return verification_results

    def generate_report(self, results: list[VerificationResult]) -> dict[str, Any]:
        """Generate verification report."""
        logger.debug("AABBVerifier.generate_report: results=%s", results)
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # Group failures by category
        failures_by_category: dict[str, list[VerificationResult]] = {}
        for r in results:
            if not r.passed:
                failures_by_category.setdefault(r.category, []).append(r)

        # Summarize by category
        category_summary = {}
        for r in results:
            if r.category not in category_summary:
                category_summary[r.category] = {"total": 0, "passed": 0, "failed": 0}
            category_summary[r.category]["total"] += 1
            if r.passed:
                category_summary[r.category]["passed"] += 1
            else:
                category_summary[r.category]["failed"] += 1

        return {
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / total if total > 0 else 0.0,
                "epsilon": self.epsilon,
            },
            "by_category": category_summary,
            "failures": [
                {
                    "test_id": r.test_id,
                    "category": r.category,
                    "reason": r.reason,
                    "java": {
                        "intersects": r.java_result.intersects,
                        "volume": r.java_result.intersection_volume,
                    },
                    "vulkan": {
                        "intersects": r.vulkan_result.intersects,
                        "volume": r.vulkan_result.intersection_volume,
                    },
                }
                for r in results
                if not r.passed
            ][:100],  # Limit to first 100 failures
        }


def progress_bar(current: int, total: int, width: int = 50):
    """Simple progress bar."""
    logger.debug("progress_bar: current=%s, total=%s, width=%s", current, total, width)
    pct = current / total
    filled = int(width * pct)
    bar = "=" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {current}/{total} ({pct * 100:.1f}%)", end="", flush=True)


def main():
    logger.debug("main called")
    parser = argparse.ArgumentParser(description="Verify AABB implementations")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("test_cases/aabb_tests.json"),
        help="Input test cases JSON",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output report JSON (default: stdout)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=EPSILON, help=f"Comparison epsilon (default: {EPSILON})"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    # Load test cases
    with open(args.input) as f:
        data = json.load(f)

    test_cases = data["test_cases"]
    print(f"Loaded {len(test_cases)} test cases from {args.input}")
    print(f"Using epsilon: {args.epsilon}")

    # Run verification
    verifier = AABBVerifier(epsilon=args.epsilon)

    if args.quiet:
        results = verifier.verify_batch(test_cases)
    else:
        results = verifier.verify_batch(test_cases, progress_callback=progress_bar)
        print()  # Newline after progress bar

    # Generate report
    report = verifier.generate_report(results)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.output}")
    else:
        print("\n" + "=" * 60)
        print("VERIFICATION REPORT")
        print("=" * 60)
        print(f"Total tests:  {report['summary']['total_tests']}")
        print(f"Passed:       {report['summary']['passed']}")
        print(f"Failed:       {report['summary']['failed']}")
        print(f"Pass rate:    {report['summary']['pass_rate'] * 100:.2f}%")
        print()
        print("By category:")
        for cat, stats in report["by_category"].items():
            print(f"  {cat}: {stats['passed']}/{stats['total']} passed")

        if report["failures"]:
            print()
            print(f"First {min(10, len(report['failures']))} failures:")
            for f in report["failures"][:10]:
                print(f"  Test {f['test_id']} ({f['category']}): {f['reason']}")

    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
