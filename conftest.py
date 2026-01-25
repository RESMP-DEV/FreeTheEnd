"""Configuration for pytest test suite.

Run tests with:
    PYTHONPATH=python uv run pytest tests/ -v

Or install the package in editable mode:
    pip install -e .
"""

import sys
from pathlib import Path

# Configure paths ONLY if not already set via PYTHONPATH or package installation
_SIM_ROOT = Path(__file__).parent
_PYTHON_DIR = _SIM_ROOT / "python"
_CPP_BUILD_DIR = _SIM_ROOT / "cpp" / "build"
_VERIFICATION_DIR = _SIM_ROOT / "verification"
_SHADER_DIR = _SIM_ROOT / "cpp" / "shaders"

# Add paths if minecraft_sim is not already importable
try:
    import minecraft_sim  # noqa: F401
except ImportError:
    if _PYTHON_DIR.exists():
        sys.path.insert(0, str(_PYTHON_DIR))
    if _CPP_BUILD_DIR.exists():
        sys.path.insert(0, str(_CPP_BUILD_DIR))

# Verification module is always local
if _VERIFICATION_DIR.exists() and str(_VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VERIFICATION_DIR))

# Export useful paths for tests to use
SIM_ROOT = _SIM_ROOT
SHADER_DIR = str(_SHADER_DIR)
VERIFICATION_DIR = _VERIFICATION_DIR


# pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU/Vulkan")
    config.addinivalue_line("markers", "oracle: marks tests that use MC oracle for validation")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on path or name."""
    for item in items:
        # Auto-mark shader tests as gpu
        if "shader" in item.nodeid.lower():
            item.add_marker("gpu")

        # Auto-mark oracle tests
        if "oracle" in item.nodeid.lower() or "verifier" in item.nodeid.lower():
            item.add_marker("oracle")
