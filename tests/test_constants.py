"""Tests for the constants module.

These tests verify that:
1. All required constants are defined
2. Constants have expected values
3. Constants are importable from multiple locations
"""

from __future__ import annotations

import pytest


class TestConstantsDefined:
    """Verify all required constants are defined."""

    def test_observation_sizes(self):
        """Observation size constants should be defined."""
        from minecraft_sim.constants import (
            EXTENDED_OBSERVATION_SIZE,
            OBSERVATION_SIZE,
            PROGRESS_OBSERVATION_SIZE,
            SURVIVAL_OBSERVATION_SIZE,
        )

        assert OBSERVATION_SIZE == 48
        assert PROGRESS_OBSERVATION_SIZE == 32
        assert EXTENDED_OBSERVATION_SIZE == 256
        assert SURVIVAL_OBSERVATION_SIZE == 64

    def test_action_sizes(self):
        """Action size constants should be defined."""
        from minecraft_sim.constants import ACTION_SIZE

        assert ACTION_SIZE == 17

    def test_simulation_parameters(self):
        """Simulation parameter constants should be defined."""
        from minecraft_sim.constants import (
            DEFAULT_MAX_EPISODE_STEPS,
            MAX_BATCH_SIZE,
            SPEEDRUN_MAX_EPISODE_STEPS,
            TICKS_PER_SECOND,
        )

        assert TICKS_PER_SECOND == 20
        assert MAX_BATCH_SIZE == 4096
        assert DEFAULT_MAX_EPISODE_STEPS == 6000  # 5 minutes at 20 tps
        assert SPEEDRUN_MAX_EPISODE_STEPS == 36000  # 30 minutes at 20 tps

    def test_curriculum_constants(self):
        """Curriculum constants should be defined."""
        from minecraft_sim.constants import NUM_CURRICULUM_STAGES, STAGE_NAMES

        assert NUM_CURRICULUM_STAGES == 6
        assert len(STAGE_NAMES) == 6
        assert STAGE_NAMES[1] == "survival"
        assert STAGE_NAMES[6] == "dragon"


class TestConstantsConsistency:
    """Verify constants are consistent across the codebase."""

    def test_backend_uses_observation_size(self):
        """VulkanBackend should use OBSERVATION_SIZE constant."""
        from minecraft_sim.backend import VulkanBackend
        from minecraft_sim.constants import OBSERVATION_SIZE

        backend = VulkanBackend(num_envs=4)
        assert backend.obs_dim == OBSERVATION_SIZE

    def test_package_exports_match_constants(self):
        """Package-level exports should match constants module."""
        import minecraft_sim
        from minecraft_sim.constants import (
            ACTION_SIZE,
            MAX_BATCH_SIZE,
            OBSERVATION_SIZE,
            TICKS_PER_SECOND,
        )

        assert minecraft_sim.OBSERVATION_SIZE == OBSERVATION_SIZE
        assert minecraft_sim.ACTION_SIZE == ACTION_SIZE
        assert minecraft_sim.MAX_BATCH_SIZE == MAX_BATCH_SIZE
        assert minecraft_sim.TICKS_PER_SECOND == TICKS_PER_SECOND

    def test_progress_observation_different_from_main(self):
        """Progress observation (32) should be distinct from main observation (48)."""
        from minecraft_sim.constants import OBSERVATION_SIZE, PROGRESS_OBSERVATION_SIZE

        assert PROGRESS_OBSERVATION_SIZE != OBSERVATION_SIZE
        assert PROGRESS_OBSERVATION_SIZE == 32
        assert OBSERVATION_SIZE == 48


class TestConstantsDocumentation:
    """Verify constants have proper documentation."""

    def test_observation_layout_documented(self):
        """The observation layout should be documented in constants.py."""
        import minecraft_sim.constants as constants

        # Check that the module docstring mentions observation layout
        assert constants.__doc__ is not None
        assert "observation" in constants.__doc__.lower()

    def test_all_constants_have_type_hints(self):
        """All constants should have type hints."""
        from minecraft_sim.constants import (
            ACTION_SIZE,
            MAX_BATCH_SIZE,
            OBSERVATION_SIZE,
            TICKS_PER_SECOND,
        )

        # These should be ints (type hints enforce this at runtime in Python 3.10+)
        assert isinstance(OBSERVATION_SIZE, int)
        assert isinstance(ACTION_SIZE, int)
        assert isinstance(MAX_BATCH_SIZE, int)
        assert isinstance(TICKS_PER_SECOND, int)


class TestConstantsImportPaths:
    """Verify constants can be imported from various paths."""

    def test_import_from_constants_module(self):
        """Constants should be importable from constants module."""
        from minecraft_sim.constants import OBSERVATION_SIZE
        assert OBSERVATION_SIZE == 48

    def test_import_from_package(self):
        """Constants should be importable from package root."""
        from minecraft_sim import OBSERVATION_SIZE
        assert OBSERVATION_SIZE == 48

    def test_import_full_path(self):
        """Constants should be accessible via full path."""
        import minecraft_sim.constants
        assert minecraft_sim.constants.OBSERVATION_SIZE == 48
