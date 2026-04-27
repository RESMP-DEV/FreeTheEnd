"""Fast smoke tests for CI - each test < 1 second.

These tests verify basic functionality without requiring GPU or heavy computation.
Run with: pytest tests/test_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def _load_minecraft_sim() -> ModuleType:
    """Load minecraft_sim from python/ subdirectory explicitly.

    This avoids import conflicts with the parent directory's __init__.py.
    """
    import sys

    # Add python/ to path so relative imports work
    python_dir = Path(__file__).parent.parent / "python"
    python_dir_str = str(python_dir.resolve())
    if python_dir_str not in sys.path:
        sys.path.insert(0, python_dir_str)

    # Remove any cached wrong module
    for key in list(sys.modules.keys()):
        if key == "minecraft_sim" or key.startswith("minecraft_sim."):
            del sys.modules[key]

    # Now import properly
    import minecraft_sim as ms

    return ms


# Load the module once at import time
_minecraft_sim: ModuleType | None = None
_import_error: str = ""

try:
    _minecraft_sim = _load_minecraft_sim()
except ImportError as e:
    _import_error = str(e)


def _require_module() -> ModuleType:
    """Get the minecraft_sim module or skip the test."""
    if _minecraft_sim is None:
        pytest.skip(f"minecraft_sim not available: {_import_error}")
    return _minecraft_sim


# =============================================================================
# Test 1: All modules import
# =============================================================================


def test_import_all() -> None:
    """All modules import without error."""
    ms = _require_module()

    assert hasattr(ms, "__version__")
    assert hasattr(ms, "OBSERVATION_SIZE")
    assert hasattr(ms, "ACTION_SIZE")

    # Check constants have reasonable values
    assert ms.OBSERVATION_SIZE == 48
    assert ms.ACTION_SIZE == 17


# =============================================================================
# Test 2: Env creates without error
# =============================================================================


@pytest.fixture
def dragon_env():
    """Create a DragonFightEnv instance."""
    ms = _require_module()

    DragonFightEnv = getattr(ms, "DragonFightEnv", None)
    if DragonFightEnv is None:
        pytest.skip("DragonFightEnv not available")

    env = DragonFightEnv()
    yield env
    env.close()


def test_create_env(dragon_env) -> None:
    """Env creates without error."""
    assert dragon_env is not None
    assert hasattr(dragon_env, "observation_space")
    assert hasattr(dragon_env, "action_space")
    assert hasattr(dragon_env, "reset")
    assert hasattr(dragon_env, "step")


# =============================================================================
# Test 3: Reset works
# =============================================================================


def test_reset(dragon_env) -> None:
    """Reset works and returns valid observation."""
    obs, info = dragon_env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


# =============================================================================
# Test 4: Step works
# =============================================================================


def test_step(dragon_env) -> None:
    """Step works and returns valid outputs."""
    dragon_env.reset()

    action = dragon_env.action_space.sample()
    result = dragon_env.step(action)

    assert len(result) == 5
    obs, reward, terminated, truncated, info = result

    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# =============================================================================
# Test 5: Observation shape
# =============================================================================


def test_observation_shape(dragon_env) -> None:
    """Observation has correct shape (48,)."""
    obs, _ = dragon_env.reset()

    assert obs.shape == (48,), f"Expected shape (48,), got {obs.shape}"
    assert dragon_env.observation_space.shape == (48,)


# =============================================================================
# Test 6: Action range
# =============================================================================


def test_action_range(dragon_env) -> None:
    """Valid action range is 0-16 (17 discrete actions)."""
    import gymnasium as gym

    assert isinstance(dragon_env.action_space, gym.spaces.Discrete)
    assert dragon_env.action_space.n == 17

    # All actions should be valid
    dragon_env.reset()
    for action in range(17):
        obs, reward, terminated, truncated, info = dragon_env.step(action)
        assert isinstance(obs, np.ndarray)


# =============================================================================
# Test 7: Reward type
# =============================================================================


def test_reward_type(dragon_env) -> None:
    """Reward is a float."""
    dragon_env.reset()
    _, reward, _, _, _ = dragon_env.step(0)

    # Reward should be convertible to float
    reward_float = float(reward)
    assert isinstance(reward_float, float)
    assert np.isfinite(reward_float)


# =============================================================================
# Test 8: Done type
# =============================================================================


def test_done_type(dragon_env) -> None:
    """Terminated and truncated are booleans."""
    dragon_env.reset()
    _, _, terminated, truncated, _ = dragon_env.step(0)

    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


# =============================================================================
# Test 9: Vec env works
# =============================================================================


def test_vec_env() -> None:
    """Vectorized environment works."""
    ms = _require_module()

    VecDragonFightEnv = getattr(ms, "VecDragonFightEnv", None)
    if VecDragonFightEnv is None:
        pytest.skip("VecDragonFightEnv not available")

    try:
        env = VecDragonFightEnv(num_envs=4)
    except ImportError:
        pytest.skip("mc189_core C++ extension not available")

    try:
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4, 48)
        assert obs.dtype == np.float32

        actions = np.random.randint(0, 17, size=4)
        obs, rewards, dones, infos = env.step(actions)

        assert obs.shape == (4, 48)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4
    finally:
        env.close()


# =============================================================================
# Test 10: SB3 env interface
# =============================================================================


def test_sb3_env() -> None:
    """SB3-compatible vectorized environment has required interface."""
    ms = _require_module()

    SB3VecDragonFightEnv = getattr(ms, "SB3VecDragonFightEnv", None)
    if SB3VecDragonFightEnv is None:
        pytest.skip("SB3VecDragonFightEnv not available")

    try:
        env = SB3VecDragonFightEnv(num_envs=4)
    except ImportError:

import logging

logger = logging.getLogger(__name__)

        pytest.skip("mc189_core C++ extension not available")

    try:
        # Check SB3 required attributes
        assert hasattr(env, "num_envs")
        assert env.num_envs == 4
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")
        assert hasattr(env, "single_observation_space")
        assert hasattr(env, "single_action_space")

        # Check SB3 required methods
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "step_async")
        assert hasattr(env, "step_wait")
        assert hasattr(env, "close")
        assert hasattr(env, "env_is_wrapped")
        assert hasattr(env, "env_method")
        assert hasattr(env, "get_attr")
        assert hasattr(env, "set_attr")
        assert hasattr(env, "seed")

        # Test basic functionality
        obs = env.reset()
        assert obs.shape == (4, 48)

        actions = np.random.randint(0, 17, size=4)
        obs, rewards, dones, infos = env.step(actions)

        assert obs.shape == (4, 48)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4

        # Test async interface
        env.step_async(actions)
        obs2, rewards2, dones2, infos2 = env.step_wait()
        assert obs2.shape == (4, 48)
    finally:
        env.close()


# =============================================================================
# Additional smoke tests for module-level checks
# =============================================================================


def test_check_cpp_module() -> None:
    """check_cpp_module() returns boolean."""
    ms = _require_module()

    check_cpp_module = getattr(ms, "check_cpp_module", None)
    if check_cpp_module is None:
        pytest.skip("check_cpp_module not available")

    result = check_cpp_module()
    assert isinstance(result, bool)


def test_get_version_info() -> None:
    """get_version_info() returns dict with expected keys."""
    ms = _require_module()

    get_version_info = getattr(ms, "get_version_info", None)
    if get_version_info is None:
        pytest.skip("get_version_info not available")

    info = get_version_info()
    assert isinstance(info, dict)
    assert "minecraft_sim" in info
    assert "cpp_module" in info
    assert info["cpp_module"] in ("available", "not available")
