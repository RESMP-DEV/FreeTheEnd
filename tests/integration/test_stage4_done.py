"""Integration test: Stage 4 done transitions on End portal success criteria.

Verifies that EndermanHuntingEnv emits terminated=True when the agent
collects 12+ ender pearls (the Stage 4 success condition that enables
End portal activation in Stage 5).

The test exercises the gymnasium step interface and checks:
1. Environment does NOT emit done before success criteria are met
2. Environment DOES emit terminated=True once ender_pearls >= 12
3. The info dict reports success=True on the terminal step
4. The stage state correctly reflects the pearl count at termination

Run with: uv run pytest contrib/minecraft_sim/tests/integration/test_stage4_done.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

SIM_ROOT = Path(__file__).parent.parent.parent.resolve()
PYTHON_DIR = SIM_ROOT / "python"
SHADERS_DIR = SIM_ROOT / "cpp" / "shaders"

sys.path = [str(PYTHON_DIR)] + [p for p in sys.path if "minecraft_sim" not in p]

try:
    if "minecraft_sim" in sys.modules:
        del sys.modules["minecraft_sim"]
    if "minecraft_sim.stage_envs" in sys.modules:
        del sys.modules["minecraft_sim.stage_envs"]

    from minecraft_sim.stage_envs import EndermanHuntingEnv, StageConfig
    HAS_STAGE_ENVS = True
    _import_error = ""
except ImportError as e:

import logging

logger = logging.getLogger(__name__)

    HAS_STAGE_ENVS = False
    EndermanHuntingEnv = None  # type: ignore[assignment, misc]
    StageConfig = None  # type: ignore[assignment, misc]
    _import_error = str(e)

pytestmark = pytest.mark.skipif(
    not HAS_STAGE_ENVS, reason=f"stage_envs not available: {_import_error}"
)


# =============================================================================
# Helpers
# =============================================================================


def _make_mock_simulator(obs_size: int = 192) -> MagicMock:
    """Create a mock mc189_core simulator for deterministic testing.

    Returns a MagicMock that behaves like MC189Simulator with controllable
    observations, rewards, and dones.
    """
    mock_sim = MagicMock()
    mock_sim.get_observations.return_value = [np.zeros(obs_size, dtype=np.float32)]
    mock_sim.get_rewards.return_value = [0.0]
    mock_sim.get_dones.return_value = [False]
    return mock_sim


def _patch_backend(mock_sim: MagicMock):
    """Context manager to patch mc189_core backend with a mock simulator."""
    mock_module = MagicMock()
    mock_config = MagicMock()
    mock_module.SimulatorConfig.return_value = mock_config
    mock_module.MC189Simulator.return_value = mock_sim

    return patch.dict(
        "minecraft_sim.stage_envs.__dict__",
        {"_mc189_core": mock_module},
    )


# =============================================================================
# Tests
# =============================================================================


class TestStage4DoneTransition:
    """Test that Stage 4 emits done when success criteria are satisfied."""

    def test_no_done_before_12_pearls(self):
        """Environment should NOT terminate before collecting 12 pearls."""
        mock_sim = _make_mock_simulator()

        with _patch_backend(mock_sim):
            env = EndermanHuntingEnv(
                config=StageConfig(max_episode_ticks=100000),
                shader_dir=str(SHADERS_DIR),
            )
            env.reset()

            # Simulate collecting fewer than 12 pearls
            env._stage_state["ender_pearls"] = 11

            _, _, terminated, truncated, info = env.step(0)

            assert not terminated, (
                "Environment should not terminate with only 11 pearls"
            )
            assert not truncated, "Should not be truncated yet"

    def test_done_at_12_pearls(self):
        """Environment should terminate once 12 ender pearls are collected."""
        mock_sim = _make_mock_simulator()

        with _patch_backend(mock_sim):
            env = EndermanHuntingEnv(
                config=StageConfig(max_episode_ticks=100000),
                shader_dir=str(SHADERS_DIR),
            )
            env.reset()

            # Set state to exactly 12 pearls (success threshold)
            env._stage_state["ender_pearls"] = 12

            _, _, terminated, truncated, info = env.step(0)

            assert terminated, (
                "Environment must emit terminated=True at 12 pearls"
            )
            assert not truncated, "Should be terminated, not truncated"

    def test_done_above_12_pearls(self):
        """Environment should terminate with more than 12 pearls."""
        mock_sim = _make_mock_simulator()

        with _patch_backend(mock_sim):
            env = EndermanHuntingEnv(
                config=StageConfig(max_episode_ticks=100000),
                shader_dir=str(SHADERS_DIR),
            )
            env.reset()

            env._stage_state["ender_pearls"] = 16

            _, _, terminated, truncated, info = env.step(0)

            assert terminated, (
                "Environment must emit terminated=True with 16 pearls"
            )

    def test_info_reports_success_on_done(self):
        """Info dict must report success=True on the terminal step."""
        mock_sim = _make_mock_simulator()

        with _patch_backend(mock_sim):
            env = EndermanHuntingEnv(
                config=StageConfig(max_episode_ticks=100000),
                shader_dir=str(SHADERS_DIR),
            )
            env.reset()

            env._stage_state["ender_pearls"] = 12

            _, _, terminated, _, info = env.step(0)

            assert terminated
            assert "episode" in info, "Terminal step must include episode info"
            assert info["episode"]["success"] is True, (
                "Episode info must report success=True when pearls >= 12"
            )

    def test_progress_snapshot_reflects_pearls(self):
        """Progress snapshot in info should reflect the pearl count."""
        mock_sim = _make_mock_simulator()

        with _patch_backend(mock_sim):
            env = EndermanHuntingEnv(
                config=StageConfig(max_episode_ticks=100000),
                shader_dir=str(SHADERS_DIR),
            )
            env.reset()

            env._stage_state["ender_pearls"] = 14
            env._stage_state["endermen_killed"] = 20

            _, _, terminated, _, info = env.step(0)

            assert terminated
            progress = info["episode"]["progress"]
            assert progress["ender_pearls"] == 14
            assert progress["endermen_killed"] == 20

    def test_truncation_before_success(self):
        """Episode truncates at max_ticks if pearls < 12."""
        mock_sim = _make_mock_simulator()

        with _patch_backend(mock_sim):
            max_ticks = 50
            env = EndermanHuntingEnv(
                config=StageConfig(max_episode_ticks=max_ticks),
                shader_dir=str(SHADERS_DIR),
            )
            env.reset()

            env._stage_state["ender_pearls"] = 8

            # Step up to max_ticks
            terminated = False
            truncated = False
            for _ in range(max_ticks):
                _, _, terminated, truncated, info = env.step(0)
                if terminated or truncated:
                    break

            assert truncated, "Should truncate at max_ticks without success"
            assert not terminated, "Should NOT terminate without 12 pearls"
            assert info["episode"]["success"] is False

    def test_raw_done_from_simulator_terminates(self):
        """If the raw simulator signals done (e.g., death), env terminates."""
        mock_sim = _make_mock_simulator()
        # Simulator signals done (player died)
        mock_sim.get_dones.return_value = [True]

        with _patch_backend(mock_sim):
            env = EndermanHuntingEnv(
                config=StageConfig(max_episode_ticks=100000),
                shader_dir=str(SHADERS_DIR),
            )
            env.reset()

            env._stage_state["ender_pearls"] = 5

            _, _, terminated, truncated, info = env.step(0)

            assert terminated, (
                "Raw simulator done should cause termination"
            )
            assert not truncated
            assert info["episode"]["success"] is False, (
                "Death with < 12 pearls is not success"
            )

    def test_success_plus_raw_done(self):
        """Success criteria met AND raw done from sim is still a success."""
        mock_sim = _make_mock_simulator()
        mock_sim.get_dones.return_value = [True]

        with _patch_backend(mock_sim):
            env = EndermanHuntingEnv(
                config=StageConfig(max_episode_ticks=100000),
                shader_dir=str(SHADERS_DIR),
            )
            env.reset()

            env._stage_state["ender_pearls"] = 12

            _, _, terminated, _, info = env.step(0)

            assert terminated
            assert info["episode"]["success"] is True

    def test_stage_id_and_name(self):
        """Verify Stage 4 identity constants."""
        mock_sim = _make_mock_simulator()

        with _patch_backend(mock_sim):
            env = EndermanHuntingEnv(
                config=StageConfig(max_episode_ticks=100000),
                shader_dir=str(SHADERS_DIR),
            )

            assert env.STAGE_ID == 4
            assert env.STAGE_NAME == "Enderman Hunting"

    def test_incremental_pearl_collection(self):
        """Simulate incremental pearl collection until done."""
        mock_sim = _make_mock_simulator()

        with _patch_backend(mock_sim):
            env = EndermanHuntingEnv(
                config=StageConfig(max_episode_ticks=100000),
                shader_dir=str(SHADERS_DIR),
            )
            env.reset()

            # Simulate collecting pearls one at a time
            for pearl_count in range(15):
                env._stage_state["ender_pearls"] = pearl_count
                _, _, terminated, truncated, _ = env.step(0)

                if pearl_count < 12:
                    assert not terminated, (
                        f"Should not terminate at {pearl_count} pearls"
                    )
                else:
                    assert terminated, (
                        f"Must terminate at {pearl_count} pearls"
                    )
                    break

    def test_reset_clears_done_state(self):
        """After reset, environment should not be in done state."""
        mock_sim = _make_mock_simulator()

        with _patch_backend(mock_sim):
            env = EndermanHuntingEnv(
                config=StageConfig(max_episode_ticks=100000),
                shader_dir=str(SHADERS_DIR),
            )

            # First episode: achieve success
            env.reset()
            env._stage_state["ender_pearls"] = 12
            _, _, terminated, _, _ = env.step(0)
            assert terminated

            # Reset and verify clean state
            env.reset()
            assert env._stage_state["ender_pearls"] == 0
            _, _, terminated, truncated, _ = env.step(0)
            assert not terminated
            assert not truncated
