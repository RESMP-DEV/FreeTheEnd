"""Runtime verifier for inventory subsystem.

Provides verification logic for:
1. Inventory state consistency
2. Operation result validation
3. Constraint enforcement
4. Test case execution
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

try:
    from .inventory_test_generator import (
        SPEEDRUN_ITEMS,
        ItemDefinition,
        TestCase,
        TestSuite,
    )
except ImportError:
    from inventory_test_generator import (
        SPEEDRUN_ITEMS,
        ItemDefinition,
        TestCase,
        TestSuite,
    )


class VerificationResult(Enum):
    """Result of a verification check."""

    PASS = auto()
    FAIL = auto()
    SKIP = auto()
    ERROR = auto()


@dataclass
class VerificationReport:
    """Report from a single verification check."""

    test_name: str
    result: VerificationResult
    message: str
    expected: Any = None
    actual: Any = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "result": self.result.name,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
            "details": self.details,
        }


@dataclass
class VerificationSummary:
    """Summary of verification results."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    reports: list[VerificationReport] = field(default_factory=list)

    def add_report(self, report: VerificationReport) -> None:
        self.reports.append(report)
        self.total += 1
        if report.result == VerificationResult.PASS:
            self.passed += 1
        elif report.result == VerificationResult.FAIL:
            self.failed += 1
        elif report.result == VerificationResult.SKIP:
            self.skipped += 1
        else:
            self.errors += 1

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "success_rate": f"{self.success_rate:.1f}%",
            "reports": [r.to_dict() for r in self.reports],
        }


# Custom exceptions for inventory operations
class InventoryError(Exception):
    """Base exception for inventory errors."""

    pass


class EmptySlotError(InventoryError):
    """Raised when operating on an empty slot."""

    pass


class InvalidSlotError(InventoryError):
    """Raised when slot index is invalid."""

    pass


class InsufficientItemsError(InventoryError):
    """Raised when not enough items available."""

    pass


class InventoryFullError(InventoryError):
    """Raised when inventory has no space."""

    pass


class BrokenToolError(InventoryError):
    """Raised when trying to use a broken tool."""

    pass


class StackOverflowError(InventoryError):
    """Raised when stack would exceed limit."""

    pass


@dataclass
class InventorySlot:
    """Represents a single inventory slot."""

    item_id: str | None = None
    count: int = 0
    durability: int | None = None

    def is_empty(self) -> bool:
        return self.item_id is None or self.count == 0

    def to_dict(self) -> dict[str, Any] | None:
        if self.is_empty():
            return None
        return {
            "item_id": self.item_id,
            "count": self.count,
            "durability": self.durability,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> InventorySlot:
        if data is None:
            return cls()
        return cls(
            item_id=data.get("item_id"),
            count=data.get("count", 0),
            durability=data.get("durability"),
        )

    def copy(self) -> InventorySlot:
        return InventorySlot(self.item_id, self.count, self.durability)


class Inventory:
    """Simulated inventory for verification."""

    def __init__(
        self,
        size: int = 36,
        items: dict[str, ItemDefinition] | None = None,
    ) -> None:
        """Initialize inventory.

        Args:
            size: Number of inventory slots.
            items: Item definitions for stack limits and durability.
        """
        self.size = size
        self.items = items or SPEEDRUN_ITEMS
        self.slots: list[InventorySlot] = [InventorySlot() for _ in range(size)]

    def get_slot(self, index: int) -> InventorySlot:
        """Get slot at index."""
        if not 0 <= index < self.size:
            raise InvalidSlotError(f"Slot {index} is out of range [0, {self.size})")
        return self.slots[index]

    def set_slot(self, index: int, slot: InventorySlot) -> None:
        """Set slot at index."""
        if not 0 <= index < self.size:
            raise InvalidSlotError(f"Slot {index} is out of range [0, {self.size})")
        self.slots[index] = slot

    def get_max_stack(self, item_id: str) -> int:
        """Get maximum stack size for an item."""
        if item_id in self.items:
            return self.items[item_id].max_stack
        return 64  # Default

    def get_item_def(self, item_id: str) -> ItemDefinition | None:
        """Get item definition."""
        return self.items.get(item_id)

    def add_item(
        self,
        item_id: str,
        count: int,
        target_slot: int | None = None,
        auto_stack: bool = True,
    ) -> int:
        """Add items to inventory.

        Args:
            item_id: Item type to add.
            count: Number to add.
            target_slot: Specific slot to target, or None for auto.
            auto_stack: Whether to auto-stack with existing items.

        Returns:
            Number of items that couldn't be added (overflow).

        Raises:
            InvalidSlotError: If target_slot is invalid.
            InventoryFullError: If no space available.
        """
        remaining = count
        item_def = self.get_item_def(item_id)
        max_stack = self.get_max_stack(item_id)

        # Handle durability for tools
        durability = None
        if item_def and item_def.has_durability:
            durability = item_def.max_durability

        if target_slot is not None:
            # Add to specific slot
            slot = self.get_slot(target_slot)
            if slot.is_empty():
                to_add = min(remaining, max_stack)
                self.slots[target_slot] = InventorySlot(item_id, to_add, durability)
                remaining -= to_add
            elif slot.item_id == item_id and durability is None:
                # Stack with existing (only if no durability)
                space = max_stack - slot.count
                to_add = min(remaining, space)
                slot.count += to_add
                remaining -= to_add
            return remaining

        if auto_stack:
            # First try to stack with existing
            if durability is None:  # Only stack non-durability items
                for i, slot in enumerate(self.slots):
                    if slot.item_id == item_id and slot.count < max_stack:
                        space = max_stack - slot.count
                        to_add = min(remaining, space)
                        slot.count += to_add
                        remaining -= to_add
                        if remaining == 0:
                            return 0

            # Then fill empty slots
            for i, slot in enumerate(self.slots):
                if slot.is_empty():
                    to_add = min(remaining, max_stack)
                    self.slots[i] = InventorySlot(item_id, to_add, durability)
                    remaining -= to_add
                    if remaining == 0:
                        return 0

        if remaining > 0 and remaining == count:
            raise InventoryFullError(f"No space for {item_id}")

        return remaining

    def remove_item(self, slot: int, count: int) -> InventorySlot:
        """Remove items from a slot.

        Args:
            slot: Slot index to remove from.
            count: Number to remove.

        Returns:
            The removed items as a slot.

        Raises:
            EmptySlotError: If slot is empty.
            InsufficientItemsError: If not enough items.
        """
        target = self.get_slot(slot)
        if target.is_empty():
            raise EmptySlotError(f"Slot {slot} is empty")
        if target.count < count:
            raise InsufficientItemsError(f"Slot {slot} has {target.count} items, need {count}")

        removed = InventorySlot(target.item_id, count, target.durability)
        target.count -= count
        if target.count == 0:
            self.slots[slot] = InventorySlot()

        return removed

    def move_item(self, source: int, target: int) -> None:
        """Move entire stack from source to target.

        Args:
            source: Source slot index.
            target: Target slot index.

        Raises:
            EmptySlotError: If source is empty.
            InvalidSlotError: If indices invalid.
        """
        src_slot = self.get_slot(source)
        tgt_slot = self.get_slot(target)

        if src_slot.is_empty():
            raise EmptySlotError(f"Source slot {source} is empty")

        if tgt_slot.is_empty():
            # Simple move
            self.slots[target] = src_slot.copy()
            self.slots[source] = InventorySlot()
        elif tgt_slot.item_id == src_slot.item_id and src_slot.durability is None:
            # Merge stacks
            max_stack = self.get_max_stack(src_slot.item_id)
            space = max_stack - tgt_slot.count
            to_move = min(src_slot.count, space)
            tgt_slot.count += to_move
            src_slot.count -= to_move
            if src_slot.count == 0:
                self.slots[source] = InventorySlot()
        else:
            # Swap
            self.swap_items(source, target)

    def swap_items(self, slot_a: int, slot_b: int) -> None:
        """Swap contents of two slots.

        Args:
            slot_a: First slot index.
            slot_b: Second slot index.
        """
        self.get_slot(slot_a)  # Validate
        self.get_slot(slot_b)  # Validate
        self.slots[slot_a], self.slots[slot_b] = self.slots[slot_b], self.slots[slot_a]

    def move_partial(self, source: int, target: int, count: int) -> None:
        """Move partial stack from source to target.

        Args:
            source: Source slot index.
            target: Target slot index.
            count: Number of items to move.
        """
        src_slot = self.get_slot(source)
        tgt_slot = self.get_slot(target)

        if src_slot.is_empty():
            raise EmptySlotError(f"Source slot {source} is empty")
        if src_slot.count < count:
            raise InsufficientItemsError(f"Slot {source} has {src_slot.count}, need {count}")

        if tgt_slot.is_empty():
            self.slots[target] = InventorySlot(src_slot.item_id, count, src_slot.durability)
            src_slot.count -= count
            if src_slot.count == 0:
                self.slots[source] = InventorySlot()
        elif tgt_slot.item_id == src_slot.item_id and src_slot.durability is None:
            max_stack = self.get_max_stack(src_slot.item_id)
            space = max_stack - tgt_slot.count
            to_move = min(count, space)
            tgt_slot.count += to_move
            src_slot.count -= to_move
            if src_slot.count == 0:
                self.slots[source] = InventorySlot()
        else:
            raise StackOverflowError(f"Cannot merge {src_slot.item_id} with {tgt_slot.item_id}")

    def split_stack(self, source: int, target: int) -> None:
        """Split stack in half, putting half in target.

        Args:
            source: Source slot index.
            target: Target slot index (must be empty).
        """
        src_slot = self.get_slot(source)
        tgt_slot = self.get_slot(target)

        if src_slot.is_empty():
            raise EmptySlotError(f"Source slot {source} is empty")
        if not tgt_slot.is_empty():
            raise StackOverflowError(f"Target slot {target} is not empty")

        half = src_slot.count // 2
        remainder = src_slot.count - half

        self.slots[target] = InventorySlot(src_slot.item_id, half, src_slot.durability)
        src_slot.count = remainder

    def use_item(self, slot: int, durability_cost: int = 1) -> bool:
        """Use an item, reducing durability.

        Args:
            slot: Slot index of item to use.
            durability_cost: Durability to consume.

        Returns:
            True if item broke, False otherwise.

        Raises:
            EmptySlotError: If slot is empty.
            BrokenToolError: If tool has 0 durability.
        """
        item_slot = self.get_slot(slot)
        if item_slot.is_empty():
            raise EmptySlotError(f"Slot {slot} is empty")

        if item_slot.durability is None:
            return False  # Non-durability item, no effect

        if item_slot.durability <= 0:
            raise BrokenToolError(f"Tool in slot {slot} is broken")

        item_slot.durability -= durability_cost
        if item_slot.durability <= 0:
            self.slots[slot] = InventorySlot()
            return True

        return False

    def damage_armor(self, slot: int, damage_points: int) -> bool:
        """Apply damage to armor piece.

        Args:
            slot: Slot index of armor.
            damage_points: Damage to apply.

        Returns:
            True if armor broke, False otherwise.
        """
        return self.use_item(slot, damage_points)

    def to_dict(self) -> list[dict[str, Any] | None]:
        """Convert inventory to list of slot dicts."""
        return [slot.to_dict() for slot in self.slots]

    @classmethod
    def from_list(
        cls,
        data: list[dict[str, Any] | None],
        items: dict[str, ItemDefinition] | None = None,
    ) -> Inventory:
        """Create inventory from list of slot dicts."""
        inv = cls(size=len(data), items=items)
        for i, slot_data in enumerate(data):
            inv.slots[i] = InventorySlot.from_dict(slot_data)
        return inv


class InventoryVerifier:
    """Verifies inventory operations against test cases."""

    def __init__(
        self,
        items: dict[str, ItemDefinition] | None = None,
        inventory_size: int = 36,
    ) -> None:
        """Initialize verifier.

        Args:
            items: Item definitions to use.
            inventory_size: Size of inventory to simulate.
        """
        self.items = items or SPEEDRUN_ITEMS
        self.inventory_size = inventory_size
        self.operation_handlers: dict[str, Callable[[TestCase], VerificationReport]] = {
            "add_item": self._verify_add_item,
            "remove_item": self._verify_remove_item,
            "move_item": self._verify_move_item,
            "swap_items": self._verify_swap_items,
            "move_partial": self._verify_move_partial,
            "split_stack": self._verify_split_stack,
            "use_item": self._verify_use_item,
            "damage_armor": self._verify_damage_armor,
        }

    def verify_test_case(self, test: TestCase) -> VerificationReport:
        """Verify a single test case.

        Args:
            test: Test case to verify.

        Returns:
            Verification report with result.
        """
        handler = self.operation_handlers.get(test.operation)
        if not handler:
            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.SKIP,
                message=f"Unknown operation: {test.operation}",
            )

        try:
            return handler(test)
        except Exception as e:
            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.ERROR,
                message=f"Exception during verification: {type(e).__name__}: {e}",
            )

    def verify_test_suite(self, suite: TestSuite) -> VerificationSummary:
        """Verify all tests in a suite.

        Args:
            suite: Test suite to verify.

        Returns:
            Summary of verification results.
        """
        summary = VerificationSummary()
        for test in suite.test_cases:
            report = self.verify_test_case(test)
            summary.add_report(report)
        return summary

    def _create_inventory(self, initial: list[dict | None] | None) -> Inventory:
        """Create inventory from initial state."""
        if initial is None:
            return Inventory(self.inventory_size, self.items)
        return Inventory.from_list(initial, self.items)

    def _verify_add_item(self, test: TestCase) -> VerificationReport:
        """Verify add_item operation."""
        inputs = test.inputs
        expected = test.expected_result

        inv = self._create_inventory(inputs.get("initial_inventory"))

        try:
            remaining = inv.add_item(
                item_id=inputs["item_id"],
                count=inputs["count"],
                target_slot=inputs.get("target_slot"),
                auto_stack=inputs.get("auto_stack", True),
            )

            if test.expected_error:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.FAIL,
                    message=f"Expected error {test.expected_error} but succeeded",
                    expected={"error": test.expected_error},
                    actual={"remaining": remaining},
                )

            # Verify remaining
            if "remaining" in expected and remaining != expected["remaining"]:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.FAIL,
                    message="Remaining items mismatch",
                    expected=expected["remaining"],
                    actual=remaining,
                )

            # Verify slot states
            for key, exp_slot in expected.items():
                if key.startswith("slot_"):
                    slot_idx = int(key.split("_")[1])
                    actual_slot = inv.get_slot(slot_idx).to_dict()
                    if actual_slot != exp_slot:
                        return VerificationReport(
                            test_name=test.name,
                            result=VerificationResult.FAIL,
                            message=f"Slot {slot_idx} mismatch",
                            expected=exp_slot,
                            actual=actual_slot,
                        )

            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.PASS,
                message="Add item verified successfully",
            )

        except InventoryError as e:
            error_type = type(e).__name__
            if test.expected_error == error_type:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.PASS,
                    message=f"Correctly raised {error_type}",
                )
            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.FAIL,
                message=f"Unexpected error: {error_type}",
                expected=test.expected_error,
                actual=error_type,
            )

    def _verify_remove_item(self, test: TestCase) -> VerificationReport:
        """Verify remove_item operation."""
        inputs = test.inputs
        expected = test.expected_result

        inv = self._create_inventory(inputs.get("initial_inventory"))

        try:
            removed = inv.remove_item(
                slot=inputs["slot"],
                count=inputs["count"],
            )

            if test.expected_error:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.FAIL,
                    message=f"Expected error {test.expected_error} but succeeded",
                )

            # Verify removed items
            if "removed" in expected:
                actual_removed = removed.to_dict()
                if actual_removed != expected["removed"]:
                    return VerificationReport(
                        test_name=test.name,
                        result=VerificationResult.FAIL,
                        message="Removed items mismatch",
                        expected=expected["removed"],
                        actual=actual_removed,
                    )

            # Verify slot states
            for key, exp_slot in expected.items():
                if key.startswith("slot_"):
                    slot_idx = int(key.split("_")[1])
                    actual_slot = inv.get_slot(slot_idx).to_dict()
                    if actual_slot != exp_slot:
                        return VerificationReport(
                            test_name=test.name,
                            result=VerificationResult.FAIL,
                            message=f"Slot {slot_idx} mismatch",
                            expected=exp_slot,
                            actual=actual_slot,
                        )

            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.PASS,
                message="Remove item verified successfully",
            )

        except InventoryError as e:
            error_type = type(e).__name__
            if test.expected_error == error_type:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.PASS,
                    message=f"Correctly raised {error_type}",
                )
            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.FAIL,
                message=f"Unexpected error: {error_type}",
                expected=test.expected_error,
                actual=error_type,
            )

    def _verify_move_item(self, test: TestCase) -> VerificationReport:
        """Verify move_item operation."""
        inputs = test.inputs
        expected = test.expected_result

        inv = self._create_inventory(inputs.get("initial_inventory"))

        try:
            inv.move_item(
                source=inputs["source_slot"],
                target=inputs["target_slot"],
            )

            if test.expected_error:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.FAIL,
                    message=f"Expected error {test.expected_error} but succeeded",
                )

            # Verify slot states
            for key, exp_slot in expected.items():
                if key.startswith("slot_"):
                    slot_idx = int(key.split("_")[1])
                    actual_slot = inv.get_slot(slot_idx).to_dict()
                    if actual_slot != exp_slot:
                        return VerificationReport(
                            test_name=test.name,
                            result=VerificationResult.FAIL,
                            message=f"Slot {slot_idx} mismatch",
                            expected=exp_slot,
                            actual=actual_slot,
                        )

            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.PASS,
                message="Move item verified successfully",
            )

        except InventoryError as e:
            error_type = type(e).__name__
            if test.expected_error == error_type:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.PASS,
                    message=f"Correctly raised {error_type}",
                )
            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.FAIL,
                message=f"Unexpected error: {error_type}",
                expected=test.expected_error,
                actual=error_type,
            )

    def _verify_swap_items(self, test: TestCase) -> VerificationReport:
        """Verify swap_items operation."""
        inputs = test.inputs
        expected = test.expected_result

        inv = self._create_inventory(inputs.get("initial_inventory"))

        try:
            inv.swap_items(
                slot_a=inputs["slot_a"],
                slot_b=inputs["slot_b"],
            )

            if test.expected_error:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.FAIL,
                    message=f"Expected error {test.expected_error} but succeeded",
                )

            # Verify slot states
            for key, exp_slot in expected.items():
                if key.startswith("slot_"):
                    slot_idx = int(key.split("_")[1])
                    actual_slot = inv.get_slot(slot_idx).to_dict()
                    if actual_slot != exp_slot:
                        return VerificationReport(
                            test_name=test.name,
                            result=VerificationResult.FAIL,
                            message=f"Slot {slot_idx} mismatch",
                            expected=exp_slot,
                            actual=actual_slot,
                        )

            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.PASS,
                message="Swap items verified successfully",
            )

        except InventoryError as e:
            error_type = type(e).__name__
            if test.expected_error == error_type:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.PASS,
                    message=f"Correctly raised {error_type}",
                )
            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.FAIL,
                message=f"Unexpected error: {error_type}",
                expected=test.expected_error,
                actual=error_type,
            )

    def _verify_move_partial(self, test: TestCase) -> VerificationReport:
        """Verify move_partial operation."""
        inputs = test.inputs
        expected = test.expected_result

        inv = self._create_inventory(inputs.get("initial_inventory"))

        try:
            inv.move_partial(
                source=inputs["source_slot"],
                target=inputs["target_slot"],
                count=inputs["count"],
            )

            if test.expected_error:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.FAIL,
                    message=f"Expected error {test.expected_error} but succeeded",
                )

            # Verify slot states
            for key, exp_slot in expected.items():
                if key.startswith("slot_"):
                    slot_idx = int(key.split("_")[1])
                    actual_slot = inv.get_slot(slot_idx).to_dict()
                    if actual_slot != exp_slot:
                        return VerificationReport(
                            test_name=test.name,
                            result=VerificationResult.FAIL,
                            message=f"Slot {slot_idx} mismatch",
                            expected=exp_slot,
                            actual=actual_slot,
                        )

            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.PASS,
                message="Move partial verified successfully",
            )

        except InventoryError as e:
            error_type = type(e).__name__
            if test.expected_error == error_type:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.PASS,
                    message=f"Correctly raised {error_type}",
                )
            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.FAIL,
                message=f"Unexpected error: {error_type}",
                expected=test.expected_error,
                actual=error_type,
            )

    def _verify_split_stack(self, test: TestCase) -> VerificationReport:
        """Verify split_stack operation."""
        inputs = test.inputs
        expected = test.expected_result

        inv = self._create_inventory(inputs.get("initial_inventory"))

        try:
            inv.split_stack(
                source=inputs["source_slot"],
                target=inputs["target_slot"],
            )

            if test.expected_error:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.FAIL,
                    message=f"Expected error {test.expected_error} but succeeded",
                )

            # Verify slot states
            for key, exp_slot in expected.items():
                if key.startswith("slot_"):
                    slot_idx = int(key.split("_")[1])
                    actual_slot = inv.get_slot(slot_idx).to_dict()
                    if actual_slot != exp_slot:
                        return VerificationReport(
                            test_name=test.name,
                            result=VerificationResult.FAIL,
                            message=f"Slot {slot_idx} mismatch",
                            expected=exp_slot,
                            actual=actual_slot,
                        )

            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.PASS,
                message="Split stack verified successfully",
            )

        except InventoryError as e:
            error_type = type(e).__name__
            if test.expected_error == error_type:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.PASS,
                    message=f"Correctly raised {error_type}",
                )
            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.FAIL,
                message=f"Unexpected error: {error_type}",
                expected=test.expected_error,
                actual=error_type,
            )

    def _verify_use_item(self, test: TestCase) -> VerificationReport:
        """Verify use_item operation."""
        inputs = test.inputs
        expected = test.expected_result

        inv = self._create_inventory(inputs.get("initial_inventory"))

        try:
            broke = inv.use_item(
                slot=inputs["slot"],
                durability_cost=inputs.get("durability_cost", 1),
            )

            if test.expected_error:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.FAIL,
                    message=f"Expected error {test.expected_error} but succeeded",
                )

            # Check if item broke
            if "item_broken" in expected:
                if broke != expected["item_broken"]:
                    return VerificationReport(
                        test_name=test.name,
                        result=VerificationResult.FAIL,
                        message="Item broken status mismatch",
                        expected=expected["item_broken"],
                        actual=broke,
                    )

            # Verify slot states
            for key, exp_slot in expected.items():
                if key.startswith("slot_"):
                    slot_idx = int(key.split("_")[1])
                    actual_slot = inv.get_slot(slot_idx).to_dict()
                    if actual_slot != exp_slot:
                        return VerificationReport(
                            test_name=test.name,
                            result=VerificationResult.FAIL,
                            message=f"Slot {slot_idx} mismatch",
                            expected=exp_slot,
                            actual=actual_slot,
                        )

            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.PASS,
                message="Use item verified successfully",
            )

        except InventoryError as e:
            error_type = type(e).__name__
            if test.expected_error == error_type:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.PASS,
                    message=f"Correctly raised {error_type}",
                )
            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.FAIL,
                message=f"Unexpected error: {error_type}",
                expected=test.expected_error,
                actual=error_type,
            )

    def _verify_damage_armor(self, test: TestCase) -> VerificationReport:
        """Verify damage_armor operation."""
        inputs = test.inputs
        expected = test.expected_result

        inv = self._create_inventory(inputs.get("initial_inventory"))

        try:
            broke = inv.damage_armor(
                slot=inputs["slot"],
                damage_points=inputs.get("damage_points", 1),
            )

            if test.expected_error:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.FAIL,
                    message=f"Expected error {test.expected_error} but succeeded",
                )

            # Verify slot states
            for key, exp_slot in expected.items():
                if key.startswith("slot_"):
                    slot_idx = int(key.split("_")[1])
                    actual_slot = inv.get_slot(slot_idx).to_dict()
                    if actual_slot != exp_slot:
                        return VerificationReport(
                            test_name=test.name,
                            result=VerificationResult.FAIL,
                            message=f"Slot {slot_idx} mismatch",
                            expected=exp_slot,
                            actual=actual_slot,
                        )

            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.PASS,
                message="Damage armor verified successfully",
            )

        except InventoryError as e:
            error_type = type(e).__name__
            if test.expected_error == error_type:
                return VerificationReport(
                    test_name=test.name,
                    result=VerificationResult.PASS,
                    message=f"Correctly raised {error_type}",
                )
            return VerificationReport(
                test_name=test.name,
                result=VerificationResult.FAIL,
                message=f"Unexpected error: {error_type}",
                expected=test.expected_error,
                actual=error_type,
            )

    def verify_from_json(self, filepath: str | Path) -> dict[str, VerificationSummary]:
        """Load test suites from JSON and verify all.

        Args:
            filepath: Path to JSON test file.

        Returns:
            Dict mapping suite names to verification summaries.
        """
        with open(filepath) as f:
            data = json.load(f)

        results: dict[str, VerificationSummary] = {}

        for suite_name, suite_data in data.items():
            if not isinstance(suite_data, dict) or "tests" not in suite_data:
                continue

            suite = TestSuite(name=suite_data["name"])
            for test_data in suite_data["tests"]:
                suite.add_test(
                    TestCase(
                        name=test_data["name"],
                        description=test_data["description"],
                        operation=test_data["operation"],
                        inputs=test_data["inputs"],
                        expected_result=test_data["expected_result"],
                        expected_error=test_data.get("expected_error"),
                    )
                )

            results[suite_name] = self.verify_test_suite(suite)

        return results

    def run_all_verification(self) -> VerificationSummary:
        """Run verification on all generated tests.

        Returns:
            Combined verification summary.
        """
        from verification.inventory_test_generator import InventoryTestGenerator

        generator = InventoryTestGenerator(items=self.items, seed=42)
        all_suites = generator.generate_all_tests()

        combined = VerificationSummary()
        for suite in all_suites.values():
            summary = self.verify_test_suite(suite)
            for report in summary.reports:
                combined.add_report(report)

        return combined


def main() -> None:
    """Run verification and display results."""
    verifier = InventoryVerifier()

    print("Inventory Verifier")
    print("=" * 50)

    summary = verifier.run_all_verification()

    print("\nResults:")
    print(f"  Total:   {summary.total}")
    print(f"  Passed:  {summary.passed}")
    print(f"  Failed:  {summary.failed}")
    print(f"  Skipped: {summary.skipped}")
    print(f"  Errors:  {summary.errors}")
    print(f"  Success: {summary.success_rate:.1f}%")

    if summary.failed > 0 or summary.errors > 0:
        print("\nFailed/Error tests:")
        for report in summary.reports:
            if report.result in (VerificationResult.FAIL, VerificationResult.ERROR):
                print(f"  - {report.test_name}: {report.message}")
                if report.expected is not None:
                    print(f"    Expected: {report.expected}")
                if report.actual is not None:
                    print(f"    Actual:   {report.actual}")


if __name__ == "__main__":
    main()
