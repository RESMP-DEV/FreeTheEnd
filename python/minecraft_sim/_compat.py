"""Compatibility layer for optional dependencies.

This module centralizes handling of optional imports (gymnasium, etc.)
so that type checking and runtime behavior are consistent across the codebase.

Usage:
    from ._compat import spaces, HAS_GYMNASIUM

    if HAS_GYMNASIUM:
        self.observation_space = spaces.Box(...)
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

__all__ = ["HAS_GYMNASIUM", "spaces"]

# Gymnasium is optional - environments work without it but won't have proper spaces
HAS_GYMNASIUM: bool = False
spaces: ModuleType | None = None

try:
    from gymnasium import spaces as _spaces

    spaces = _spaces
    HAS_GYMNASIUM = True
except ImportError:
    pass


# For type checkers: expose the types when gymnasium is available
if TYPE_CHECKING:
    from gymnasium import spaces as spaces  # noqa: F811
