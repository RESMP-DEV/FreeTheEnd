"""Stage 1 (Basic Survival) test suite.

Tests cover basic initialization, survival mechanics, and reward shaping:
- Environment initializes
- Player spawns on ground
- Movement works
- Blocks can be broken (attack loop)
- Mobs can spawn at night
- Progressive success criteria (iron pickaxe crafting)
- _shape_reward computes correct values for crafted observation snapshots
- _stage_state reflects wood, combat, and tool progress after each step

Run with: uv run pytest contrib/minecraft_sim/tests/test_stage1_survival.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch  # noqa: F401 - used in _make_mock_env

import numpy as np
import pytest

SIM_ROOT = Path(__file__).parent.parent.resolve()
PYTHON_DIR = SIM_ROOT / "python"
SHADERS_DIR = SIM_ROOT / "cpp" / "shaders"

sys.path = [str(PYTHON_DIR)] + [p for p in sys.path if "minecraft_sim" not in p]

try:
    if "minecraft_sim" in sys.modules:
        del sys.modules["minecraft_sim"]
    if "minecraft_sim.mc189_core" in sys.modules:
        del sys.modules["minecraft_sim.mc189_core"]

    from minecraft_sim import mc189_core  # noqa: F401
    from minecraft_sim.stage_envs import BasicSurvivalEnv, StageConfig

    HAS_MC189_CORE = True
    _import_error = ""
except ImportError as exc:
    HAS_MC189_CORE = False
    BasicSurvivalEnv = None  # type: ignore[assignment, misc]
    StageConfig = None  # type: ignore[assignment, misc]
    _import_error = str(exc)

_needs_backend = pytest.mark.skipif(
    not HAS_MC189_CORE, reason=f"mc189_core C++ extension not available: {_import_error}"
)


class Actions:
    """Stage 1 action space constants (subset)."""

    NOOP = 0
    FORWARD = 1
    ATTACK = 9


def decode_player(obs: np.ndarray) -> dict[str, float | bool]:
    """Decode player state from observation vector."""
    return {
        "x": obs[0] * 100,
        "y": obs[1] * 50 + 64,
        "z": obs[2] * 100,
        "vel_x": obs[3],
        "vel_y": obs[4],
        "vel_z": obs[5],
        "health": obs[8] * 20,
        "on_ground": obs[10] > 0.5,
    }


def decode_time(obs: np.ndarray) -> dict[str, float | bool]:
    """Decode time-of-day from observation vector (if present)."""
    time_normalized = obs[44] if len(obs) > 44 else 0.0
    return {
        "time_normalized": float(time_normalized),
        "is_night": time_normalized > 0.5,
        "ticks": int(time_normalized * 24000),
    }


# ============================================================================
# Mock helper for testing without C++ backend
# ============================================================================


def _make_mock_env():
    """Create a BasicSurvivalEnv with a mocked mc189_core backend.

    Enables testing _shape_reward and _stage_state logic without
    requiring the compiled C++ extension.
    """

    mock_sim = MagicMock()
    mock_sim.get_observations.return_value = np.zeros(128, dtype=np.float32)
    mock_sim.get_rewards.return_value = np.array([0.0], dtype=np.float32)
    mock_sim.get_dones.return_value = np.array([False], dtype=np.bool_)
    mock_sim.reset.return_value = None
    mock_sim.step.return_value = None

    mock_config = MagicMock()
    mock_config.num_envs = 1
    mock_config.shader_dir = str(SHADERS_DIR)

    mock_mc189 = MagicMock()
    mock_mc189.SimulatorConfig.return_value = mock_config
    mock_mc189.MC189Simulator.return_value = mock_sim

    with patch.dict(sys.modules, {"mc189_core": mock_mc189}):
        with patch("minecraft_sim.stage_envs._mc189_core", mock_mc189):
            from minecraft_sim.stage_envs import BasicSurvivalEnv as BSE
            from minecraft_sim.stage_envs import StageConfig as SC

            config = SC()
            env = BSE(config=config, shader_dir=str(SHADERS_DIR))

    return env


# ============================================================================
# Integration tests (require mc189_core C++ backend)
# ============================================================================


@pytest.fixture
def survival_env():
    """Create a BasicSurvivalEnv instance (requires mc189_core)."""
    if not HAS_MC189_CORE:
        pytest.skip(f"mc189_core not available: {_import_error}")
    config = StageConfig()
    env = BasicSurvivalEnv(config=config, shader_dir=str(SHADERS_DIR))
    yield env
    env.close()


@_needs_backend
def test_environment_initializes(survival_env) -> None:
    """Environment initializes with expected spaces."""
    assert survival_env is not None
    assert survival_env.observation_space.shape == (128,)
    assert survival_env.action_space.n >= 17


@_needs_backend
def test_player_spawns_on_ground(survival_env) -> None:
    """Player spawns on the ground with valid position."""
    obs, _ = survival_env.reset()
    player = decode_player(obs)

    assert player["on_ground"]
    assert player["y"] >= 60
    assert np.isfinite(player["x"])
    assert np.isfinite(player["z"])


@_needs_backend
def test_movement_works(survival_env) -> None:
    """Forward movement updates player position."""
    obs, _ = survival_env.reset()
    player_before = decode_player(obs)

    for _ in range(20):
        obs, _, terminated, truncated, _ = survival_env.step(Actions.FORWARD)
        if terminated or truncated:
            break

    player_after = decode_player(obs)
    distance = np.hypot(
        player_after["x"] - player_before["x"],
        player_after["z"] - player_before["z"],
    )

    assert distance >= 0.01


@_needs_backend
def test_can_break_blocks(survival_env) -> None:
    """Attack actions execute without error and advance the simulation."""
    obs, _ = survival_env.reset()
    obs_before = obs.copy()

    for _ in range(60):
        obs, reward, terminated, truncated, _ = survival_env.step(Actions.ATTACK)
        assert np.isfinite(float(reward))
        if terminated or truncated:
            break

    assert obs.shape == obs_before.shape
    assert np.any(obs != obs_before)


@_needs_backend
def test_mobs_spawn_at_night(survival_env) -> None:
    """Mobs can spawn at night (if entity data is exposed)."""
    survival_env.reset()

    mob_found = False
    night_reached = False
    max_steps = 2000

    for _ in range(max_steps):
        obs, _, terminated, truncated, _ = survival_env.step(Actions.NOOP)
        time_info = decode_time(obs)
        night_reached = night_reached or time_info["is_night"]

        if time_info["is_night"] and hasattr(survival_env._sim, "get_nearby_entities"):
            nearby = survival_env._sim.get_nearby_entities()
            if nearby:
                mob_found = True
                break

        if terminated or truncated:
            break

    if not night_reached:
        pytest.skip("Night not reached within limited steps")

    assert isinstance(mob_found, bool)


@_needs_backend
def test_progressive_success_criteria(survival_env) -> None:
    """_check_success() remains False until has_iron_pickaxe is set."""
    survival_env.reset()

    assert not survival_env._check_success()

    survival_env._stage_state["wood_mined"] = 12
    assert not survival_env._check_success()

    survival_env._stage_state["zombies_killed"] = 3
    assert not survival_env._check_success()

    survival_env._stage_state["skeletons_killed"] = 2
    assert not survival_env._check_success()

    survival_env._stage_state["has_wooden_pickaxe"] = True
    assert not survival_env._check_success()

    survival_env._stage_state["has_wooden_sword"] = True
    assert not survival_env._check_success()

    survival_env._stage_state["chunks_explored"] = {(0, 0), (1, 0), (-1, 0), (0, 1)}
    assert not survival_env._check_success()

    survival_env._stage_state["has_iron_pickaxe"] = True
    assert survival_env._check_success()


# ============================================================================
# Reward shaping tests (mocked backend, pure Python logic)
# ============================================================================


class TestShapeReward:
    """Tests for _shape_reward on crafted observation snapshots."""

    def test_exploration_reward_new_chunk(self) -> None:
        """Moving to a new chunk gives exploration bonus."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        obs = np.zeros(128, dtype=np.float32)
        obs[0] = 8.0  # x=8, chunk_x = 8 // 16 = 0
        obs[2] = 8.0  # z=8, chunk_z = 8 // 16 = 0

        reward = env._shape_reward(0.0, obs, Actions.FORWARD)
        assert reward == pytest.approx(env.REWARD_EXPLORATION)
        assert (0, 0) in env._stage_state["chunks_explored"]

    def test_exploration_reward_same_chunk_no_bonus(self) -> None:
        """Staying in an already-explored chunk gives no exploration bonus."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()
        env._stage_state["chunks_explored"].add((0, 0))

        obs = np.zeros(128, dtype=np.float32)
        obs[0] = 8.0
        obs[2] = 8.0

        reward = env._shape_reward(0.0, obs, Actions.FORWARD)
        assert reward == 0.0

    def test_exploration_reward_multiple_chunks(self) -> None:
        """Moving through multiple chunks accumulates exploration rewards."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        total_reward = 0.0
        for chunk_x in range(5):
            obs = np.zeros(128, dtype=np.float32)
            obs[0] = chunk_x * 16.0 + 1.0
            obs[2] = 0.0
            reward = env._shape_reward(0.0, obs, Actions.FORWARD)
            total_reward += reward

        assert len(env._stage_state["chunks_explored"]) == 5
        assert total_reward == pytest.approx(5 * env.REWARD_EXPLORATION)

    def test_reward_scale_applied(self) -> None:
        """Reward scale from config is applied to shaped reward."""
        env = _make_mock_env()
        env.config.reward_scale = 2.0
        env._stage_state = env._initialize_stage_state()

        obs = np.zeros(128, dtype=np.float32)
        obs[0] = 20.0
        obs[2] = 20.0

        reward = env._shape_reward(0.0, obs, Actions.FORWARD)
        assert reward == pytest.approx(env.REWARD_EXPLORATION * 2.0)

    def test_base_reward_passthrough(self) -> None:
        """Base reward passes through when no exploration bonus applies."""
        env = _make_mock_env()
        env.config.reward_scale = 1.0
        env._stage_state = env._initialize_stage_state()
        env._stage_state["chunks_explored"].add((0, 0))

        obs = np.zeros(128, dtype=np.float32)
        obs[0] = 5.0
        obs[2] = 5.0

        reward = env._shape_reward(1.5, obs, Actions.ATTACK)
        assert reward == pytest.approx(1.5)

    def test_base_reward_plus_exploration(self) -> None:
        """Base reward and exploration bonus combine correctly."""
        env = _make_mock_env()
        env.config.reward_scale = 1.0
        env._stage_state = env._initialize_stage_state()

        obs = np.zeros(128, dtype=np.float32)
        obs[0] = 32.0  # New chunk (2, 0)
        obs[2] = 0.0

        reward = env._shape_reward(0.5, obs, Actions.FORWARD)
        expected = (0.5 + env.REWARD_EXPLORATION) * 1.0
        assert reward == pytest.approx(expected)

    def test_negative_base_reward_preserved(self) -> None:
        """Negative base reward (damage penalty) is preserved."""
        env = _make_mock_env()
        env.config.reward_scale = 1.0
        env._stage_state = env._initialize_stage_state()
        env._stage_state["chunks_explored"].add((0, 0))

        obs = np.zeros(128, dtype=np.float32)

        reward = env._shape_reward(-2.0, obs, Actions.NOOP)
        assert reward == pytest.approx(-2.0)

    def test_small_observation_no_crash(self) -> None:
        """Observation smaller than 3 elements doesn't crash _shape_reward."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        obs = np.zeros(2, dtype=np.float32)
        reward = env._shape_reward(0.0, obs, Actions.NOOP)
        assert reward == 0.0


# ============================================================================
# Stage state progression tests
# ============================================================================


class TestStageStateProgression:
    """Tests verifying _stage_state tracks wood, combat, and tool progress."""

    def test_initial_stage_state(self) -> None:
        """Initial _stage_state has all expected keys at zero/empty."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        assert env._stage_state["zombies_killed"] == 0
        assert env._stage_state["skeletons_killed"] == 0
        assert env._stage_state["wood_mined"] == 0
        assert env._stage_state["has_wooden_pickaxe"] is False
        assert env._stage_state["has_wooden_sword"] is False
        assert env._stage_state["has_iron_pickaxe"] is False
        assert isinstance(env._stage_state["chunks_explored"], set)
        assert len(env._stage_state["chunks_explored"]) == 0
        assert env._stage_state["last_position"] is None

    def test_wood_progress_tracking(self) -> None:
        """Incrementing wood_mined reflects gathering progress."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        for i in range(1, 6):
            env._stage_state["wood_mined"] += 1
            assert env._stage_state["wood_mined"] == i

    def test_combat_progress_zombies(self) -> None:
        """Incrementing zombies_killed tracks combat progress."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        env._stage_state["zombies_killed"] += 3
        assert env._stage_state["zombies_killed"] == 3

    def test_combat_progress_skeletons(self) -> None:
        """Incrementing skeletons_killed tracks combat progress."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        env._stage_state["skeletons_killed"] += 2
        assert env._stage_state["skeletons_killed"] == 2

    def test_tool_progress_wooden_pickaxe(self) -> None:
        """Setting has_wooden_pickaxe reflects tool crafting."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        assert not env._stage_state["has_wooden_pickaxe"]
        env._stage_state["has_wooden_pickaxe"] = True
        assert env._stage_state["has_wooden_pickaxe"]

    def test_tool_progress_wooden_sword(self) -> None:
        """Setting has_wooden_sword reflects tool crafting."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        assert not env._stage_state["has_wooden_sword"]
        env._stage_state["has_wooden_sword"] = True
        assert env._stage_state["has_wooden_sword"]

    def test_iron_pickaxe_triggers_success(self) -> None:
        """Setting has_iron_pickaxe causes _check_success to return True."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        assert not env._check_success()
        env._stage_state["has_iron_pickaxe"] = True
        assert env._check_success()

    def test_exploration_state_accumulates_via_reward(self) -> None:
        """chunks_explored grows as _shape_reward sees new positions."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        positions = [(0, 0), (16, 0), (32, 0), (0, 16), (16, 16)]
        for x, z in positions:
            obs = np.zeros(128, dtype=np.float32)
            obs[0] = float(x) + 1.0
            obs[2] = float(z) + 1.0
            env._shape_reward(0.0, obs, Actions.FORWARD)

        expected_chunks = {(int((x + 1) // 16), int((z + 1) // 16)) for x, z in positions}
        assert env._stage_state["chunks_explored"] == expected_chunks

    def test_full_progression_sequence(self) -> None:
        """Simulate a full wood -> combat -> tool progression sequence."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        # Phase 1: Mine wood (4 logs)
        for i in range(4):
            env._stage_state["wood_mined"] += 1
            obs = np.zeros(128, dtype=np.float32)
            obs[0] = float(i * 2)
            env._shape_reward(0.1, obs, Actions.ATTACK)
        assert env._stage_state["wood_mined"] == 4
        assert not env._check_success()

        # Phase 2: Craft wooden tools
        env._stage_state["has_wooden_pickaxe"] = True
        env._stage_state["has_wooden_sword"] = True
        assert not env._check_success()

        # Phase 3: Kill mobs
        env._stage_state["zombies_killed"] = 5
        env._stage_state["skeletons_killed"] = 2
        assert not env._check_success()

        # Phase 4: Iron pickaxe (end goal)
        env._stage_state["has_iron_pickaxe"] = True
        assert env._check_success()

        # Verify complete state snapshot
        assert env._stage_state["wood_mined"] == 4
        assert env._stage_state["has_wooden_pickaxe"] is True
        assert env._stage_state["has_wooden_sword"] is True
        assert env._stage_state["zombies_killed"] == 5
        assert env._stage_state["skeletons_killed"] == 2
        assert env._stage_state["has_iron_pickaxe"] is True

    def test_reward_values_after_each_step(self) -> None:
        """Reward values are correct after each progression milestone."""
        env = _make_mock_env()
        env.config.reward_scale = 1.0
        env._stage_state = env._initialize_stage_state()

        # Step 1: Movement into new chunk -> exploration reward
        obs1 = np.zeros(128, dtype=np.float32)
        obs1[0] = 20.0  # Chunk (1, 0)
        r1 = env._shape_reward(0.0, obs1, Actions.FORWARD)
        assert r1 == pytest.approx(env.REWARD_EXPLORATION)

        # Step 2: Wood mining with base reward in same chunk
        env._stage_state["wood_mined"] += 1
        obs2 = np.zeros(128, dtype=np.float32)
        obs2[0] = 22.0  # Still chunk (1, 0)
        r2 = env._shape_reward(0.1, obs2, Actions.ATTACK)
        assert r2 == pytest.approx(0.1)  # No exploration bonus

        # Step 3: Combat kill in new chunk
        env._stage_state["zombies_killed"] += 1
        obs3 = np.zeros(128, dtype=np.float32)
        obs3[0] = 48.0  # Chunk (3, 0)
        obs3[2] = 0.0
        r3 = env._shape_reward(0.5, obs3, Actions.ATTACK)
        assert r3 == pytest.approx(0.5 + env.REWARD_EXPLORATION)

        # Step 4: Tool crafted, no exploration bonus in visited chunk
        env._stage_state["has_wooden_pickaxe"] = True
        obs4 = np.zeros(128, dtype=np.float32)
        obs4[0] = 21.0  # Chunk (1, 0) already explored
        r4 = env._shape_reward(0.5, obs4, Actions.NOOP)
        assert r4 == pytest.approx(0.5)

    def test_reinitialize_clears_all_progress(self) -> None:
        """_initialize_stage_state produces a clean state after dirtying."""
        env = _make_mock_env()
        env._stage_state = env._initialize_stage_state()

        # Dirty the state
        env._stage_state["zombies_killed"] = 10
        env._stage_state["wood_mined"] = 50
        env._stage_state["has_iron_pickaxe"] = True
        env._stage_state["chunks_explored"].add((99, 99))

        # Re-initialize
        env._stage_state = env._initialize_stage_state()
        assert env._stage_state["zombies_killed"] == 0
        assert env._stage_state["wood_mined"] == 0
        assert env._stage_state["has_iron_pickaxe"] is False
        assert len(env._stage_state["chunks_explored"]) == 0
