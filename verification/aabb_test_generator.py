#!/usr/bin/env python3
"""AABB test case generator for verification harness.

Generates 10,000 random test cases covering:
- Normal boxes with valid positive dimensions
- Edge cases: zero-volume boxes (degenerate)
- Edge cases: negative dimensions (invalid)
- Intersection edge cases: touching, overlapping, contained, disjoint
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import logging

logger = logging.getLogger(__name__)


@dataclass
class AABB:
    """Axis-aligned bounding box."""

    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    def to_dict(self) -> dict[str, float]:
        logger.debug("AABB.to_dict called")
        return asdict(self)

    @property
    def volume(self) -> float:
        logger.debug("AABB.volume called")
        dx = self.max_x - self.min_x
        dy = self.max_y - self.min_y
        dz = self.max_z - self.min_z
        return max(0.0, dx * dy * dz)

    def is_valid(self) -> bool:
        logger.debug("AABB.is_valid called")
        return self.max_x >= self.min_x and self.max_y >= self.min_y and self.max_z >= self.min_z


@dataclass
class TestCase:
    """Single AABB test case."""

    id: int
    category: str
    box_a: AABB
    box_b: AABB
    description: str

    def to_dict(self) -> dict[str, Any]:
        logger.debug("TestCase.to_dict called")
        return {
            "id": self.id,
            "category": self.category,
            "box_a": self.box_a.to_dict(),
            "box_b": self.box_b.to_dict(),
            "description": self.description,
        }


class AABBTestGenerator:
    """Generates AABB test cases with controlled randomness."""

    def __init__(self, seed: int = 42):
        logger.info("AABBTestGenerator.__init__: seed=%s", seed)
        self.rng = random.Random(seed)
        self.test_id = 0

    def _next_id(self) -> int:
        logger.debug("AABBTestGenerator._next_id called")
        self.test_id += 1
        return self.test_id

    def _random_float(self, lo: float = -100.0, hi: float = 100.0) -> float:
        logger.debug("AABBTestGenerator._random_float: lo=%s, hi=%s", lo, hi)
        return self.rng.uniform(lo, hi)

    def _random_positive(self, lo: float = 0.1, hi: float = 50.0) -> float:
        logger.debug("AABBTestGenerator._random_positive: lo=%s, hi=%s", lo, hi)
        return self.rng.uniform(lo, hi)

    def _random_normal_box(self) -> AABB:
        """Generate a valid AABB with positive dimensions."""
        logger.debug("AABBTestGenerator._random_normal_box called")
        x = self._random_float()
        y = self._random_float()
        z = self._random_float()
        dx = self._random_positive()
        dy = self._random_positive()
        dz = self._random_positive()
        return AABB(x, y, z, x + dx, y + dy, z + dz)

    def generate_normal_boxes(self, count: int) -> list[TestCase]:
        """Generate test cases with two normal valid boxes."""
        logger.debug("AABBTestGenerator.generate_normal_boxes: count=%s", count)
        cases = []
        for _ in range(count):
            cases.append(
                TestCase(
                    id=self._next_id(),
                    category="normal",
                    box_a=self._random_normal_box(),
                    box_b=self._random_normal_box(),
                    description="Two valid boxes with positive dimensions",
                )
            )
        return cases

    def generate_zero_volume_boxes(self, count: int) -> list[TestCase]:
        """Generate test cases with zero-volume (degenerate) boxes."""
        logger.debug("AABBTestGenerator.generate_zero_volume_boxes: count=%s", count)
        cases = []
        for i in range(count):
            box_a = self._random_normal_box()

            # Create degenerate box (zero in one or more dimensions)
            x = self._random_float()
            y = self._random_float()
            z = self._random_float()

            variant = i % 7
            if variant == 0:
                # Point (zero in all dimensions)
                box_b = AABB(x, y, z, x, y, z)
                desc = "Box B is a point (zero volume in all dims)"
            elif variant == 1:
                # Line segment along X
                dx = self._random_positive()
                box_b = AABB(x, y, z, x + dx, y, z)
                desc = "Box B is a line segment along X"
            elif variant == 2:
                # Line segment along Y
                dy = self._random_positive()
                box_b = AABB(x, y, z, x, y + dy, z)
                desc = "Box B is a line segment along Y"
            elif variant == 3:
                # Line segment along Z
                dz = self._random_positive()
                box_b = AABB(x, y, z, x, y, z + dz)
                desc = "Box B is a line segment along Z"
            elif variant == 4:
                # Flat plane XY
                dx, dy = self._random_positive(), self._random_positive()
                box_b = AABB(x, y, z, x + dx, y + dy, z)
                desc = "Box B is a flat plane in XY"
            elif variant == 5:
                # Flat plane XZ
                dx, dz = self._random_positive(), self._random_positive()
                box_b = AABB(x, y, z, x + dx, y, z + dz)
                desc = "Box B is a flat plane in XZ"
            else:
                # Flat plane YZ
                dy, dz = self._random_positive(), self._random_positive()
                box_b = AABB(x, y, z, x, y + dy, z + dz)
                desc = "Box B is a flat plane in YZ"

            cases.append(
                TestCase(
                    id=self._next_id(),
                    category="zero_volume",
                    box_a=box_a,
                    box_b=box_b,
                    description=desc,
                )
            )
        return cases

    def generate_negative_dimension_boxes(self, count: int) -> list[TestCase]:
        """Generate test cases with inverted min/max (negative dimensions)."""
        logger.debug("AABBTestGenerator.generate_negative_dimension_boxes: count=%s", count)
        cases = []
        for i in range(count):
            box_a = self._random_normal_box()

            # Create box with inverted coordinates
            x1, x2 = self._random_float(), self._random_float()
            y1, y2 = self._random_float(), self._random_float()
            z1, z2 = self._random_float(), self._random_float()

            variant = i % 7
            if variant == 0:
                # Inverted X only
                box_b = AABB(max(x1, x2), y1, z1, min(x1, x2), y2, z2)
                if y1 > y2:
                    box_b.min_y, box_b.max_y = y2, y1
                if z1 > z2:
                    box_b.min_z, box_b.max_z = z2, z1
                box_b.min_x, box_b.max_x = box_b.max_x, box_b.min_x
                desc = "Box B has inverted X dimension"
            elif variant == 1:
                # Inverted Y only
                box_b = AABB(
                    min(x1, x2), max(y1, y2), min(z1, z2), max(x1, x2), min(y1, y2), max(z1, z2)
                )
                desc = "Box B has inverted Y dimension"
            elif variant == 2:
                # Inverted Z only
                box_b = AABB(
                    min(x1, x2), min(y1, y2), max(z1, z2), max(x1, x2), max(y1, y2), min(z1, z2)
                )
                desc = "Box B has inverted Z dimension"
            elif variant == 3:
                # Inverted X and Y
                box_b = AABB(
                    max(x1, x2), max(y1, y2), min(z1, z2), min(x1, x2), min(y1, y2), max(z1, z2)
                )
                desc = "Box B has inverted X and Y dimensions"
            elif variant == 4:
                # Inverted X and Z
                box_b = AABB(
                    max(x1, x2), min(y1, y2), max(z1, z2), min(x1, x2), max(y1, y2), min(z1, z2)
                )
                desc = "Box B has inverted X and Z dimensions"
            elif variant == 5:
                # Inverted Y and Z
                box_b = AABB(
                    min(x1, x2), max(y1, y2), max(z1, z2), max(x1, x2), min(y1, y2), min(z1, z2)
                )
                desc = "Box B has inverted Y and Z dimensions"
            else:
                # All dimensions inverted
                box_b = AABB(
                    max(x1, x2), max(y1, y2), max(z1, z2), min(x1, x2), min(y1, y2), min(z1, z2)
                )
                desc = "Box B has all dimensions inverted"

            cases.append(
                TestCase(
                    id=self._next_id(),
                    category="negative_dimension",
                    box_a=box_a,
                    box_b=box_b,
                    description=desc,
                )
            )
        return cases

    def generate_touching_boxes(self, count: int) -> list[TestCase]:
        """Generate test cases where boxes touch at face/edge/corner."""
        logger.debug("AABBTestGenerator.generate_touching_boxes: count=%s", count)
        cases = []
        for i in range(count):
            box_a = self._random_normal_box()

            variant = i % 6
            if variant == 0:
                # Touch at +X face
                dx = self._random_positive()
                dy = self._random_positive()
                dz = self._random_positive()
                box_b = AABB(
                    box_a.max_x,
                    box_a.min_y,
                    box_a.min_z,
                    box_a.max_x + dx,
                    box_a.max_y,
                    box_a.max_z,
                )
                desc = "Boxes touch at +X face"
            elif variant == 1:
                # Touch at -X face
                dx = self._random_positive()
                box_b = AABB(
                    box_a.min_x - dx,
                    box_a.min_y,
                    box_a.min_z,
                    box_a.min_x,
                    box_a.max_y,
                    box_a.max_z,
                )
                desc = "Boxes touch at -X face"
            elif variant == 2:
                # Touch at +Y face
                dy = self._random_positive()
                box_b = AABB(
                    box_a.min_x,
                    box_a.max_y,
                    box_a.min_z,
                    box_a.max_x,
                    box_a.max_y + dy,
                    box_a.max_z,
                )
                desc = "Boxes touch at +Y face"
            elif variant == 3:
                # Touch at edge (X-Y edge at +X, +Y)
                dx, dy = self._random_positive(), self._random_positive()
                box_b = AABB(
                    box_a.max_x,
                    box_a.max_y,
                    box_a.min_z,
                    box_a.max_x + dx,
                    box_a.max_y + dy,
                    box_a.max_z,
                )
                desc = "Boxes touch at +X+Y edge"
            elif variant == 4:
                # Touch at corner (+X, +Y, +Z)
                dx, dy, dz = (
                    self._random_positive(),
                    self._random_positive(),
                    self._random_positive(),
                )
                box_b = AABB(
                    box_a.max_x,
                    box_a.max_y,
                    box_a.max_z,
                    box_a.max_x + dx,
                    box_a.max_y + dy,
                    box_a.max_z + dz,
                )
                desc = "Boxes touch at +X+Y+Z corner"
            else:
                # Touch at -Z face
                dz = self._random_positive()
                box_b = AABB(
                    box_a.min_x,
                    box_a.min_y,
                    box_a.min_z - dz,
                    box_a.max_x,
                    box_a.max_y,
                    box_a.min_z,
                )
                desc = "Boxes touch at -Z face"

            cases.append(
                TestCase(
                    id=self._next_id(),
                    category="touching",
                    box_a=box_a,
                    box_b=box_b,
                    description=desc,
                )
            )
        return cases

    def generate_overlapping_boxes(self, count: int) -> list[TestCase]:
        """Generate test cases where boxes partially overlap."""
        logger.debug("AABBTestGenerator.generate_overlapping_boxes: count=%s", count)
        cases = []
        for i in range(count):
            box_a = self._random_normal_box()

            # Compute center and half-extents
            cx = (box_a.min_x + box_a.max_x) / 2
            cy = (box_a.min_y + box_a.max_y) / 2
            cz = (box_a.min_z + box_a.max_z) / 2
            hx = (box_a.max_x - box_a.min_x) / 2
            hy = (box_a.max_y - box_a.min_y) / 2
            hz = (box_a.max_z - box_a.min_z) / 2

            # Generate overlapping box with offset
            variant = i % 4
            if variant == 0:
                # Small overlap
                offset_factor = self.rng.uniform(0.7, 0.95)
            elif variant == 1:
                # Medium overlap
                offset_factor = self.rng.uniform(0.3, 0.7)
            elif variant == 2:
                # Large overlap
                offset_factor = self.rng.uniform(0.05, 0.3)
            else:
                # Random overlap
                offset_factor = self.rng.uniform(0.01, 0.99)

            # Random direction of offset
            dir_x = self.rng.choice([-1, 1]) * self.rng.random()
            dir_y = self.rng.choice([-1, 1]) * self.rng.random()
            dir_z = self.rng.choice([-1, 1]) * self.rng.random()

            # New center
            new_cx = cx + dir_x * hx * 2 * offset_factor
            new_cy = cy + dir_y * hy * 2 * offset_factor
            new_cz = cz + dir_z * hz * 2 * offset_factor

            # New half-extents (randomize size a bit)
            new_hx = hx * self.rng.uniform(0.5, 1.5)
            new_hy = hy * self.rng.uniform(0.5, 1.5)
            new_hz = hz * self.rng.uniform(0.5, 1.5)

            box_b = AABB(
                new_cx - new_hx,
                new_cy - new_hy,
                new_cz - new_hz,
                new_cx + new_hx,
                new_cy + new_hy,
                new_cz + new_hz,
            )

            cases.append(
                TestCase(
                    id=self._next_id(),
                    category="overlapping",
                    box_a=box_a,
                    box_b=box_b,
                    description=f"Boxes partially overlap (offset factor ~{offset_factor:.2f})",
                )
            )
        return cases

    def generate_contained_boxes(self, count: int) -> list[TestCase]:
        """Generate test cases where one box fully contains the other."""
        logger.debug("AABBTestGenerator.generate_contained_boxes: count=%s", count)
        cases = []
        for i in range(count):
            outer = self._random_normal_box()

            # Compute inner box that fits inside
            cx = (outer.min_x + outer.max_x) / 2
            cy = (outer.min_y + outer.max_y) / 2
            cz = (outer.min_z + outer.max_z) / 2
            hx = (outer.max_x - outer.min_x) / 2
            hy = (outer.max_y - outer.min_y) / 2
            hz = (outer.max_z - outer.min_z) / 2

            # Inner box with random shrink factor
            shrink = self.rng.uniform(0.1, 0.9)
            inner_hx = hx * shrink
            inner_hy = hy * shrink
            inner_hz = hz * shrink

            # Random offset within bounds
            max_offset_x = hx - inner_hx
            max_offset_y = hy - inner_hy
            max_offset_z = hz - inner_hz

            offset_x = self.rng.uniform(-max_offset_x, max_offset_x)
            offset_y = self.rng.uniform(-max_offset_y, max_offset_y)
            offset_z = self.rng.uniform(-max_offset_z, max_offset_z)

            inner = AABB(
                cx + offset_x - inner_hx,
                cy + offset_y - inner_hy,
                cz + offset_z - inner_hz,
                cx + offset_x + inner_hx,
                cy + offset_y + inner_hy,
                cz + offset_z + inner_hz,
            )

            if i % 2 == 0:
                # Box A contains Box B
                cases.append(
                    TestCase(
                        id=self._next_id(),
                        category="contained",
                        box_a=outer,
                        box_b=inner,
                        description="Box A fully contains Box B",
                    )
                )
            else:
                # Box B contains Box A
                cases.append(
                    TestCase(
                        id=self._next_id(),
                        category="contained",
                        box_a=inner,
                        box_b=outer,
                        description="Box B fully contains Box A",
                    )
                )
        return cases

    def generate_disjoint_boxes(self, count: int) -> list[TestCase]:
        """Generate test cases where boxes do not intersect at all."""
        logger.debug("AABBTestGenerator.generate_disjoint_boxes: count=%s", count)
        cases = []
        for i in range(count):
            box_a = self._random_normal_box()

            # Place box_b completely outside box_a
            gap = self._random_positive(0.01, 10.0)
            variant = i % 6

            if variant == 0:
                # Box B is to the +X side
                dx = self._random_positive()
                dy = self._random_positive()
                dz = self._random_positive()
                box_b = AABB(
                    box_a.max_x + gap,
                    self._random_float(),
                    self._random_float(),
                    box_a.max_x + gap + dx,
                    self._random_float() + dy,
                    self._random_float() + dz,
                )
                # Ensure box_b is valid
                if box_b.min_y > box_b.max_y:
                    box_b.min_y, box_b.max_y = box_b.max_y, box_b.min_y
                if box_b.min_z > box_b.max_z:
                    box_b.min_z, box_b.max_z = box_b.max_z, box_b.min_z
                desc = "Box B is disjoint (+X direction)"
            elif variant == 1:
                # Box B is to the -X side
                dx = self._random_positive()
                box_b = AABB(
                    box_a.min_x - gap - dx,
                    box_a.min_y,
                    box_a.min_z,
                    box_a.min_x - gap,
                    box_a.max_y,
                    box_a.max_z,
                )
                desc = "Box B is disjoint (-X direction)"
            elif variant == 2:
                # Box B is to the +Y side
                dy = self._random_positive()
                box_b = AABB(
                    box_a.min_x,
                    box_a.max_y + gap,
                    box_a.min_z,
                    box_a.max_x,
                    box_a.max_y + gap + dy,
                    box_a.max_z,
                )
                desc = "Box B is disjoint (+Y direction)"
            elif variant == 3:
                # Box B is to the -Y side
                dy = self._random_positive()
                box_b = AABB(
                    box_a.min_x,
                    box_a.min_y - gap - dy,
                    box_a.min_z,
                    box_a.max_x,
                    box_a.min_y - gap,
                    box_a.max_z,
                )
                desc = "Box B is disjoint (-Y direction)"
            elif variant == 4:
                # Box B is to the +Z side
                dz = self._random_positive()
                box_b = AABB(
                    box_a.min_x,
                    box_a.min_y,
                    box_a.max_z + gap,
                    box_a.max_x,
                    box_a.max_y,
                    box_a.max_z + gap + dz,
                )
                desc = "Box B is disjoint (+Z direction)"
            else:
                # Box B is to the -Z side
                dz = self._random_positive()
                box_b = AABB(
                    box_a.min_x,
                    box_a.min_y,
                    box_a.min_z - gap - dz,
                    box_a.max_x,
                    box_a.max_y,
                    box_a.min_z - gap,
                )
                desc = "Box B is disjoint (-Z direction)"

            cases.append(
                TestCase(
                    id=self._next_id(),
                    category="disjoint",
                    box_a=box_a,
                    box_b=box_b,
                    description=desc,
                )
            )
        return cases

    def generate_extreme_values(self, count: int) -> list[TestCase]:
        """Generate test cases with extreme float values."""
        logger.debug("AABBTestGenerator.generate_extreme_values: count=%s", count)
        cases = []
        extremes = [1e-10, 1e-5, 1e5, 1e10, float("inf"), -float("inf")]

        for i in range(count):
            variant = i % 6
            box_a = self._random_normal_box()

            if variant == 0:
                # Very small box
                scale = 1e-8
                box_b = AABB(0, 0, 0, scale, scale, scale)
                desc = "Box B has very small dimensions (~1e-8)"
            elif variant == 1:
                # Very large box
                scale = 1e6
                box_b = AABB(-scale, -scale, -scale, scale, scale, scale)
                desc = "Box B has very large dimensions (~1e6)"
            elif variant == 2:
                # Near-zero dimensions
                eps = 1e-10
                box_b = AABB(0, 0, 0, eps, 1, 1)
                desc = "Box B has near-zero X dimension (~1e-10)"
            elif variant == 3:
                # Large coordinate values
                base = 1e8
                d = self._random_positive()
                box_b = AABB(base, base, base, base + d, base + d, base + d)
                desc = "Box B has large coordinate values (~1e8)"
            elif variant == 4:
                # Mixed extreme scales
                box_b = AABB(1e-8, 1e5, 1e-3, 1e-7, 1e6, 1e-2)
                desc = "Box B has mixed extreme scales"
            else:
                # Negative large values
                base = -1e7
                d = self._random_positive() * 1e3
                box_b = AABB(base, base, base, base + d, base + d, base + d)
                desc = "Box B has large negative coordinate values"

            cases.append(
                TestCase(
                    id=self._next_id(),
                    category="extreme_values",
                    box_a=box_a,
                    box_b=box_b,
                    description=desc,
                )
            )
        return cases

    def generate_all(self, total: int = 10000) -> list[TestCase]:
        """Generate all test cases with balanced distribution."""
        # Distribution: ~40% normal, ~10% each for other categories
        logger.debug("AABBTestGenerator.generate_all: total=%s", total)
        counts = {
            "normal": int(total * 0.30),
            "zero_volume": int(total * 0.10),
            "negative_dimension": int(total * 0.10),
            "touching": int(total * 0.10),
            "overlapping": int(total * 0.15),
            "contained": int(total * 0.10),
            "disjoint": int(total * 0.10),
            "extreme_values": int(total * 0.05),
        }

        # Adjust to hit exact total
        allocated = sum(counts.values())
        counts["normal"] += total - allocated

        all_cases = []
        all_cases.extend(self.generate_normal_boxes(counts["normal"]))
        all_cases.extend(self.generate_zero_volume_boxes(counts["zero_volume"]))
        all_cases.extend(self.generate_negative_dimension_boxes(counts["negative_dimension"]))
        all_cases.extend(self.generate_touching_boxes(counts["touching"]))
        all_cases.extend(self.generate_overlapping_boxes(counts["overlapping"]))
        all_cases.extend(self.generate_contained_boxes(counts["contained"]))
        all_cases.extend(self.generate_disjoint_boxes(counts["disjoint"]))
        all_cases.extend(self.generate_extreme_values(counts["extreme_values"]))

        # Shuffle to mix categories
        self.rng.shuffle(all_cases)

        # Re-assign sequential IDs after shuffle
        for i, case in enumerate(all_cases, 1):
            case.id = i

        return all_cases


def main():
    logger.debug("main called")
    parser = argparse.ArgumentParser(description="Generate AABB test cases")
    parser.add_argument("--count", type=int, default=10000, help="Number of test cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", type=Path, default=Path("test_cases/aabb_tests.json"), help="Output JSON file"
    )
    args = parser.parse_args()

    generator = AABBTestGenerator(seed=args.seed)
    cases = generator.generate_all(args.count)

    # Build output structure
    output = {
        "metadata": {
            "total_cases": len(cases),
            "seed": args.seed,
            "categories": {
                cat: sum(1 for c in cases if c.category == cat)
                for cat in [
                    "normal",
                    "zero_volume",
                    "negative_dimension",
                    "touching",
                    "overlapping",
                    "contained",
                    "disjoint",
                    "extreme_values",
                ]
            },
        },
        "test_cases": [c.to_dict() for c in cases],
    }

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(cases)} test cases to {args.output}")
    print(f"Category distribution: {output['metadata']['categories']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
