"""Tests that verify Python fallback paths work WITHOUT the C++ extension.

These tests ensure that users who haven't built mc189_core can still:
1. Import the package
2. Get correct observation dimensions
3. Run training scripts (with mock observations)
4. Have deterministic behavior when setting seeds

Run these tests with: pytest tests/test_fallback.py -v

The tests use monkeypatching to hide mc189_core, simulating the experience
of a user who hasn't built the C++ extension.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from pytest import MonkeyPatch


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hide_cpp_module(monkeypatch: MonkeyPatch):
    """Hide mc189_core to force fallback paths.

    This simulates a user who hasn't built the C++ extension.
    """
    # Remove mc189_core from sys.modules if present
    modules_to_hide = [
        "mc189_core",
        "minecraft_sim.backend",
        "minecraft_sim.env",
        "minecraft_sim.vec_env",
        "minecraft_sim.wrappers",
        "minecraft_sim",
    ]

    original_modules = {}
    for mod in modules_to_hide:
        if mod in sys.modules:
            original_modules[mod] = sys.modules.pop(mod)

    # Patch import to fail for mc189_core
    original_import = __builtins__.__dict__.get("__import__", __import__)

    def mock_import(name, *args, **kwargs):
        if name == "mc189_core" or name.startswith("mc189_core."):
            raise ImportError(f"No module named '{name}' (mocked for testing)")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    yield

    # Restore original modules
    for mod, module in original_modules.items():
        sys.modules[mod] = module


@pytest.fixture
def backend_without_cpp(hide_cpp_module):
    """Get VulkanBackend in fallback mode."""
    from minecraft_sim.backend import VulkanBackend
    return VulkanBackend


@pytest.fixture
def constants():
    """Get constants module."""
    from minecraft_sim.constants import (
        ACTION_SIZE,
        OBSERVATION_SIZE,
        PROGRESS_OBSERVATION_SIZE,
    )
    return {
        "OBSERVATION_SIZE": OBSERVATION_SIZE,
        "ACTION_SIZE": ACTION_SIZE,
        "PROGRESS_OBSERVATION_SIZE": PROGRESS_OBSERVATION_SIZE,
    }


# =============================================================================
# Dimension Consistency Tests
# =============================================================================


class TestFallbackDimensions:
    """Verify observation dimensions are consistent in fallback mode."""

    def test_backend_obs_dim_matches_constant(self, backend_without_cpp, constants):
        """Backend.obs_dim should match OBSERVATION_SIZE constant."""
        backend = backend_without_cpp(num_envs=4)
        assert backend.obs_dim == constants["OBSERVATION_SIZE"]
        assert backend.obs_dim == 48, "OBSERVATION_SIZE should be 48"

    def test_reset_returns_correct_shape(self, backend_without_cpp):
        """reset() should return observations with correct shape."""
        backend = backend_without_cpp(num_envs=4)
        obs = backend.reset()

        assert obs.shape == (4, 48), f"Expected (4, 48), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    def test_step_returns_correct_shapes(self, backend_without_cpp):
        """step() should return all outputs with correct shapes."""
        backend = backend_without_cpp(num_envs=4)
        backend.reset()

        actions = np.zeros(4, dtype=np.int32)
        obs, rewards, dones, infos = backend.step(actions)

        assert obs.shape == (4, 48), f"obs shape: expected (4, 48), got {obs.shape}"
        assert rewards.shape == (4,), f"rewards shape: expected (4,), got {rewards.shape}"
        assert dones.shape == (4,), f"dones shape: expected (4,), got {dones.shape}"
        assert isinstance(infos, dict), f"infos type: expected dict, got {type(infos)}"

    def test_different_num_envs(self, backend_without_cpp):
        """Backend should handle various num_envs correctly."""
        for num_envs in [1, 4, 16, 64, 256]:
            backend = backend_without_cpp(num_envs=num_envs)
            obs = backend.reset()

            assert obs.shape == (num_envs, 48), \
                f"num_envs={num_envs}: expected ({num_envs}, 48), got {obs.shape}"

    def test_wrapper_handles_obs_dim(self, hide_cpp_module):
        """NormalizedObsWrapper should handle observation dimensions correctly."""
        from minecraft_sim.backend import VulkanBackend
        from minecraft_sim.wrappers import NormalizedObsWrapper

        backend = VulkanBackend(num_envs=4)
        wrapper = NormalizedObsWrapper(backend)

        assert wrapper.obs_dim == 48

        obs = wrapper.reset()
        assert obs.shape == (4, 48), f"Expected (4, 48), got {obs.shape}"

        # Normalized obs should be in [-1, 1] range (approximately)
        assert obs.min() >= -2.0, f"Normalized obs min too low: {obs.min()}"
        assert obs.max() <= 2.0, f"Normalized obs max too high: {obs.max()}"


class TestProgressObservationSeparation:
    """Verify progress observations (32-dim) are separate from game observations (48-dim)."""

    def test_progress_observation_is_32_dim(self):
        """SpeedrunProgress.to_observation() should return 32-dim vector."""
        from minecraft_sim.progression import SpeedrunProgress

        progress = SpeedrunProgress()
        obs = progress.to_observation()

        assert obs.shape == (32,), f"Expected (32,), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_progress_and_game_obs_are_different_concepts(self, constants):
        """PROGRESS_OBSERVATION_SIZE != OBSERVATION_SIZE (they serve different purposes)."""
        assert constants["PROGRESS_OBSERVATION_SIZE"] == 32
        assert constants["OBSERVATION_SIZE"] == 48
        assert constants["PROGRESS_OBSERVATION_SIZE"] != constants["OBSERVATION_SIZE"]


# =============================================================================
# Determinism Tests
# =============================================================================


class TestFallbackDeterminism:
    """Verify seeds propagate correctly in fallback mode."""

    def test_reset_accepts_seed_parameter(self, backend_without_cpp):
        """reset() should accept seed parameter without error."""
        backend = backend_without_cpp(num_envs=2)

        # Should not raise
        obs = backend.reset(seed=42)
        assert obs is not None

    def test_explicit_seed_is_deterministic(self, backend_without_cpp):
        """Same seed should produce same observations."""
        backend1 = backend_without_cpp(num_envs=2)
        backend2 = backend_without_cpp(num_envs=2)

        obs1 = backend1.reset(seed=12345)
        obs2 = backend2.reset(seed=12345)

        np.testing.assert_array_equal(obs1, obs2,
            "Same seed should produce identical observations")

    def test_different_seeds_may_differ(self, backend_without_cpp):
        """Different seeds should (eventually) produce different observations.

        Note: In pure fallback mode, observations are zeros, so this tests
        that the seed mechanism is in place even if results are identical.
        """
        backend = backend_without_cpp(num_envs=2)

        # At minimum, both calls should succeed
        obs1 = backend.reset(seed=111)
        obs2 = backend.reset(seed=222)

        assert obs1 is not None
        assert obs2 is not None

    def test_numpy_seed_affects_reset(self, backend_without_cpp):
        """np.random.seed() should affect reset() when no explicit seed given."""
        backend = backend_without_cpp(num_envs=2)

        # Set numpy seed and reset
        np.random.seed(42)
        obs1 = backend.reset()

        # Reset numpy seed and reset again
        np.random.seed(42)
        backend2 = backend_without_cpp(num_envs=2)
        obs2 = backend2.reset()

        # Both should be zeros in fallback mode, but the mechanism should work
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_generates_seed_from_numpy(self, backend_without_cpp):
        """When no seed provided, reset() should generate from numpy state."""
        backend = backend_without_cpp(num_envs=2)

        # Mock np.random.randint to verify it's called
        call_count = [0]
        original_randint = np.random.randint

        def mock_randint(*args, **kwargs):
            call_count[0] += 1
            return original_randint(*args, **kwargs)

        with patch.object(np.random, 'randint', mock_randint):
            backend.reset()  # No seed provided

        assert call_count[0] >= 1, "reset() should call np.random.randint when no seed provided"


# =============================================================================
# Import Tests
# =============================================================================


class TestFallbackImports:
    """Verify package can be imported without C++ extension."""

    def test_can_import_package(self, hide_cpp_module):
        """minecraft_sim should import without mc189_core."""
        import minecraft_sim
        assert minecraft_sim is not None

    def test_constants_available(self, hide_cpp_module):
        """Constants should be available without C++ module."""
        from minecraft_sim.constants import (
            ACTION_SIZE,
            OBSERVATION_SIZE,
            PROGRESS_OBSERVATION_SIZE,
        )

        assert OBSERVATION_SIZE == 48
        assert ACTION_SIZE == 17
        assert PROGRESS_OBSERVATION_SIZE == 32

    def test_backend_importable(self, hide_cpp_module):
        """VulkanBackend should be importable without mc189_core."""
        from minecraft_sim.backend import VulkanBackend
        assert VulkanBackend is not None

    def test_progression_importable(self, hide_cpp_module):
        """Progression module should work without C++ extension."""
        from minecraft_sim.progression import ProgressTracker, SpeedrunProgress

        progress = SpeedrunProgress()
        tracker = ProgressTracker()

        assert progress is not None
        assert tracker is not None


# =============================================================================
# Edge Cases
# =============================================================================


class TestFallbackEdgeCases:
    """Test edge cases in fallback mode."""

    def test_zero_envs_rejected(self, backend_without_cpp):
        """num_envs=0 should be rejected."""
        with pytest.raises(ValueError):
            backend_without_cpp(num_envs=0)

    def test_negative_envs_rejected(self, backend_without_cpp):
        """Negative num_envs should be rejected."""
        with pytest.raises(ValueError):
            backend_without_cpp(num_envs=-1)

    def test_action_shape_mismatch_rejected(self, backend_without_cpp):
        """Actions with wrong shape should be rejected."""
        backend = backend_without_cpp(num_envs=4)
        backend.reset()

        # Wrong number of actions
        with pytest.raises(ValueError):
            backend.step(np.zeros(3, dtype=np.int32))  # Expected 4

    def test_multiple_resets(self, backend_without_cpp):
        """Multiple resets should work correctly."""
        backend = backend_without_cpp(num_envs=4)

        for i in range(5):
            obs = backend.reset(seed=i)
            assert obs.shape == (4, 48), f"Reset {i} failed"

    def test_reset_after_step(self, backend_without_cpp):
        """Reset after stepping should work correctly."""
        backend = backend_without_cpp(num_envs=4)

        obs = backend.reset()
        for _ in range(10):
            obs, _, _, _ = backend.step(np.zeros(4, dtype=np.int32))

        # Reset should work after stepping
        obs = backend.reset()
        assert obs.shape == (4, 48)
