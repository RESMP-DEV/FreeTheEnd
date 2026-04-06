"""Test verifier modules for import and basic functionality."""

import importlib
import sys
from pathlib import Path

import pytest

VERIFICATION_DIR = Path(__file__).parent.parent / "verification"

# Add verification dir to path for imports
sys.path.insert(0, str(VERIFICATION_DIR))


def get_verifier_modules():
    """Get all verifier module names."""
    return [f.stem for f in VERIFICATION_DIR.glob("*_verifier.py")]


def get_test_generator_modules():
    """Get all test generator module names."""
    return [f.stem for f in VERIFICATION_DIR.glob("*_test_generator.py")]


class TestVerifierImports:
    """Test that all verifier modules can be imported."""

    @pytest.mark.parametrize("module_name", get_verifier_modules())
    def test_verifier_imports(self, module_name):
        """Each verifier module should import without errors."""
        try:
            module = importlib.import_module(module_name)
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")

    @pytest.mark.parametrize("module_name", get_test_generator_modules())
    def test_generator_imports(self, module_name):
        """Each test generator module should import without errors."""
        try:
            module = importlib.import_module(module_name)
            assert module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")


class TestVerifierStructure:
    """Test verifier file structure."""

    @pytest.mark.parametrize("module_name", get_verifier_modules())
    def test_verifier_has_verify_function(self, module_name):
        """Verifiers should have a verify() or main verification function."""
        module = importlib.import_module(module_name)

        # Check for common verification function names
        has_verify = any(
            [
                hasattr(module, "verify"),
                hasattr(module, "verify_all"),
                hasattr(module, "run_verification"),
                hasattr(module, "main"),
                hasattr(module, "Verifier"),
                # Also check for *Verifier classes (e.g., HungerVerifier, DamageVerifier)
                any(name.endswith("Verifier") for name in dir(module) if not name.startswith("_")),
            ]
        )

        assert has_verify, f"{module_name} missing verification entry point"

    @pytest.mark.parametrize("module_name", get_test_generator_modules())
    def test_generator_has_generate_function(self, module_name):
        """Test generators should have a generate() function."""
        module = importlib.import_module(module_name)

        has_generate = any(
            [
                hasattr(module, "generate"),
                hasattr(module, "generate_tests"),
                hasattr(module, "generate_test_cases"),
                hasattr(module, "main"),
                hasattr(module, "TestGenerator"),
                # Also check for *Generator or *TestGenerator classes
                any(name.endswith("Generator") for name in dir(module) if not name.startswith("_")),
                # Check for functions containing "generate" in their name
                any(
                    "generate" in name.lower()
                    for name in dir(module)
                    if callable(getattr(module, name, None))
                ),
            ]
        )

        assert has_generate, f"{module_name} missing generation entry point"
