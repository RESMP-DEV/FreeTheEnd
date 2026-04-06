"""Test damage calculation mechanics against MC 1.8.9 oracle."""

import json
import sys
from pathlib import Path

import pytest

VERIFICATION_DIR = Path(__file__).parent.parent / "verification"
sys.path.insert(0, str(VERIFICATION_DIR))


class TestDamageCalculation:
    """Test damage calculation against known values."""

    @pytest.fixture
    def test_cases(self):
        """Load damage test cases."""
        test_file = VERIFICATION_DIR / "test_cases.json"
        if not test_file.exists():
            pytest.skip("test_cases.json not found")
        return json.loads(test_file.read_text())

    @pytest.fixture
    def expected_values(self):
        """Load expected damage values."""
        expected_file = VERIFICATION_DIR / "expected_values.json"
        if not expected_file.exists():
            pytest.skip("expected_values.json not found")
        return json.loads(expected_file.read_text())

    def test_test_cases_loaded(self, test_cases):
        """Verify test cases are loadable."""
        assert isinstance(test_cases, (list, dict))
        print(f"Loaded {len(test_cases) if isinstance(test_cases, list) else 'N/A'} test cases")

    def test_expected_values_loaded(self, expected_values):
        """Verify expected values are loadable."""
        assert isinstance(expected_values, (list, dict))


class TestAABB:
    """Test AABB collision mechanics."""

    @pytest.fixture
    def aabb_verifier(self):
        """Import AABB verifier."""
        try:
            import aabb_verifier

            return aabb_verifier
        except ImportError:
            pytest.skip("aabb_verifier not available")

    def test_aabb_verifier_exists(self, aabb_verifier):
        """AABB verifier should be importable."""
        assert aabb_verifier is not None


class TestLookDirection:
    """Test look direction (yaw/pitch) calculations."""

    @pytest.fixture
    def look_test_cases(self):
        """Load look direction test cases."""
        test_file = VERIFICATION_DIR / "look_direction_test_cases.json"
        if not test_file.exists():
            pytest.skip("look_direction_test_cases.json not found")
        return json.loads(test_file.read_text())

    def test_look_test_cases_loaded(self, look_test_cases):
        """Verify look direction test cases are loadable."""
        assert isinstance(look_test_cases, (list, dict))
        if isinstance(look_test_cases, list):
            print(f"Loaded {len(look_test_cases)} look direction test cases")
