"""Verification module - Python reference implementations and test utilities."""

from pathlib import Path

import logging

logger = logging.getLogger(__name__)

VERIFICATION_DIR = Path(__file__).parent
SHADERS_DIR = VERIFICATION_DIR.parent / "cpp" / "shaders"
ORACLE_DIR = VERIFICATION_DIR.parent / "oracle"


def list_verifiers():
    """List all available verifier modules."""
    logger.debug("list_verifiers called")
    return [f.stem for f in VERIFICATION_DIR.glob("*_verifier.py")]


def list_test_generators():
    """List all available test generator modules."""
    logger.debug("list_test_generators called")
    return [f.stem for f in VERIFICATION_DIR.glob("*_test_generator.py")]


def list_shaders():
    """List all Vulkan compute shaders."""
    logger.debug("list_shaders called")
    return [f.name for f in SHADERS_DIR.glob("*.comp")]
