"""Debug scripts configuration - handles path setup."""

import sys
from pathlib import Path
import logging

# Use project-relative paths

logger = logging.getLogger(__name__)

_SIM_ROOT = Path(__file__).parent.parent.parent
_PYTHON_DIR = _SIM_ROOT / "python"
_CPP_BUILD_DIR = _SIM_ROOT / "cpp" / "build"
_SHADER_DIR = _SIM_ROOT / "cpp" / "shaders"

# Add paths if needed
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
if str(_CPP_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_CPP_BUILD_DIR))

SHADER_DIR = str(_SHADER_DIR)
