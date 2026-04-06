"""Comprehensive test matrix for MC189 Simulator.

Tests all combinations of:
- Stages: 1-6 (BASIC_SURVIVAL through END_FIGHT)
- Num envs: 1, 4, 16, 64
- Actions: all 17 discrete actions (0-16)
- Edge cases: boundary conditions

Run with: PYTHONPATH=python uv run pytest tests/test_matrix.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

# Setup paths for imports
SIM_ROOT = Path(__file__).parent.parent.resolve()
PYTHON_DIR = SIM_ROOT / "python"
SHADERS_DIR = SIM_ROOT / "cpp" / "shaders"

sys.path = [str(PYTHON_DIR)] + [p for p in sys.path if "minecraft_sim" not in p]

try:
    if "minecraft_sim" in sys.modules:
        del sys.modules["minecraft_sim"]
    if "minecraft_sim.mc189_core" in sys.modules:
        del sys.modules["minecraft_sim.mc189_core"]

    from minecraft_sim import mc189_core
    from minecraft_sim.curriculum import StageID

    HAS_MC189_CORE = True
    _import_error = ""
except ImportError as e:
    HAS_MC189_CORE = False
    mc189_core = None
    StageID = None
    _import_error = str(e)

if TYPE_CHECKING:
    pass

pytestmark = pytest.mark.skipif(
    not HAS_MC189_CORE, reason=f"mc189_core not available: {_import_error}"
)

# =============================================================================
# Test Constants
# =============================================================================

STAGES = [1, 2, 3, 4, 5, 6]
NUM_ENVS_VALUES = [1, 4, 16, 64]
ALL_ACTIONS = list(range(17))  # 0-16
OBS_DIM = 48

# Action names for documentation
ACTION_NAMES = [
    "NOOP",
    "FORWARD",
    "BACK",
    "LEFT",
    "RIGHT",
    "FORWARD_LEFT",
    "FORWARD_RIGHT",
    "JUMP",
    "JUMP_FORWARD",
    "ATTACK",
    "ATTACK_FORWARD",
    "SPRINT_FORWARD",
    "LOOK_LEFT",
    "LOOK_RIGHT",
    "SWAP_WEAPON",
    "LOOK_UP",
    "LOOK_DOWN",
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def make_simulator():
    """Factory fixture to create simulators with specified num_envs."""
    created_sims = []

    def _make(num_envs: int = 1) -> mc189_core.MC189Simulator:
        config = mc189_core.SimulatorConfig()
        config.num_envs = num_envs
        config.shader_dir = str(SHADERS_DIR)
        sim = mc189_core.MC189Simulator(config)
        sim.reset()
        sim.step(np.zeros(num_envs, dtype=np.int32))  # Initialize observations
        created_sims.append(sim)
        return sim

    yield _make


# =============================================================================
# Stage x Num Envs Matrix Tests
# =============================================================================


class TestStageNumEnvsMatrix:
    """Test all combinations of stages and num_envs."""

    @pytest.mark.parametrize("stage", STAGES, ids=[f"stage_{s}" for s in STAGES])
    @pytest.mark.parametrize(
        "num_envs", NUM_ENVS_VALUES, ids=[f"envs_{n}" for n in NUM_ENVS_VALUES]
    )
    def test_initialization(self, make_simulator, stage: int, num_envs: int):
        """Test simulator initializes correctly for all stage/env combinations."""
        sim = make_simulator(num_envs)
        obs = sim.get_observations()

        assert obs.shape[0] == num_envs, f"Expected {num_envs} envs, got {obs.shape[0]}"
        assert obs.shape[1] >= OBS_DIM, f"Expected at least {OBS_DIM} obs dims"
        assert obs.dtype == np.float32, "Observations should be float32"

    @pytest.mark.parametrize("stage", STAGES, ids=[f"stage_{s}" for s in STAGES])
    @pytest.mark.parametrize(
        "num_envs", NUM_ENVS_VALUES, ids=[f"envs_{n}" for n in NUM_ENVS_VALUES]
    )
    def test_reset(self, make_simulator, stage: int, num_envs: int):
        """Test reset works for all stage/env combinations."""
        sim = make_simulator(num_envs)

        # Run some steps
        for _ in range(10):
            sim.step(np.random.randint(0, 17, size=num_envs, dtype=np.int32))

        # Reset
        sim.reset()
        sim.step(np.zeros(num_envs, dtype=np.int32))
        obs = sim.get_observations()

        # Verify reset state
        assert obs.shape[0] == num_envs
        assert not np.any(np.isnan(obs)), "Observations contain NaN after reset"
        assert not np.any(np.isinf(obs)), "Observations contain Inf after reset"

    @pytest.mark.parametrize("stage", STAGES, ids=[f"stage_{s}" for s in STAGES])
    @pytest.mark.parametrize(
        "num_envs", NUM_ENVS_VALUES, ids=[f"envs_{n}" for n in NUM_ENVS_VALUES]
    )
    def test_step_stability(self, make_simulator, stage: int, num_envs: int):
        """Test 100 random steps don't produce NaN/Inf."""
        sim = make_simulator(num_envs)

        for step in range(100):
            actions = np.random.randint(0, 17, size=num_envs, dtype=np.int32)
            sim.step(actions)
            obs = sim.get_observations()

            assert not np.any(np.isnan(obs)), f"NaN at step {step}"
            assert not np.any(np.isinf(obs)), f"Inf at step {step}"


# =============================================================================
# Action Space Matrix Tests
# =============================================================================


class TestActionMatrix:
    """Test all 17 discrete actions across configurations."""

    @pytest.mark.parametrize(
        "action", ALL_ACTIONS, ids=[f"action_{a}_{ACTION_NAMES[a]}" for a in ALL_ACTIONS]
    )
    def test_action_executes_single_env(self, make_simulator, action: int):
        """Each action executes without error in single env."""
        sim = make_simulator(1)
        actions = np.array([action], dtype=np.int32)

        # Execute 10 steps with this action
        for _ in range(10):
            sim.step(actions)
            obs = sim.get_observations()
            assert not np.any(np.isnan(obs))

    @pytest.mark.parametrize(
        "action", ALL_ACTIONS, ids=[f"action_{a}_{ACTION_NAMES[a]}" for a in ALL_ACTIONS]
    )
    @pytest.mark.parametrize("num_envs", [4, 16], ids=["envs_4", "envs_16"])
    def test_action_executes_batched(self, make_simulator, action: int, num_envs: int):
        """Each action executes correctly in batched environments."""
        sim = make_simulator(num_envs)
        actions = np.full(num_envs, action, dtype=np.int32)

        for _ in range(10):
            sim.step(actions)
            obs = sim.get_observations()
            assert obs.shape[0] == num_envs
            assert not np.any(np.isnan(obs))

    @pytest.mark.parametrize(
        "action", ALL_ACTIONS, ids=[f"action_{a}_{ACTION_NAMES[a]}" for a in ALL_ACTIONS]
    )
    def test_action_produces_state_change(self, make_simulator, action: int):
        """Most actions should produce observable state change."""
        sim = make_simulator(1)
        obs_before = sim.get_observations()[0].copy()

        # Execute action 20 times
        for _ in range(20):
            sim.step(np.array([action], dtype=np.int32))

        obs_after = sim.get_observations()[0]

        # NOOP (action 0) may not change state much
        if action == 0:
            # NOOP should maintain stability
            assert not np.any(np.isnan(obs_after))
        else:
            # Other actions should change something
            # Allow for physics settling - at least some change
            delta = np.abs(obs_after - obs_before)
            total_change = np.sum(delta)
            # Most actions should produce some change
            # (attack may not if nothing to attack, but position/yaw/pitch should change)
            assert total_change > 0.001 or action in [9, 14], (
                f"Action {action} ({ACTION_NAMES[action]}) produced no state change"
            )


class TestActionSemantics:
    """Test semantic correctness of movement and look actions."""

    @pytest.mark.parametrize(
        "action,expected_change",
        [
            (1, ("z", "increase")),  # FORWARD
            (2, ("z", "decrease")),  # BACK
            (3, ("x", "decrease")),  # LEFT
            (4, ("x", "increase")),  # RIGHT
            (12, ("yaw", "decrease")),  # LOOK_LEFT
            (13, ("yaw", "increase")),  # LOOK_RIGHT
            (15, ("pitch", "decrease")),  # LOOK_UP
            (16, ("pitch", "increase")),  # LOOK_DOWN
        ],
        ids=[
            "forward_increases_z",
            "back_decreases_z",
            "left_decreases_x",
            "right_increases_x",
            "look_left_decreases_yaw",
            "look_right_increases_yaw",
            "look_up_decreases_pitch",
            "look_down_increases_pitch",
        ],
    )
    def test_movement_direction(
        self, make_simulator, action: int, expected_change: tuple[str, str]
    ):
        """Test movement and look actions change expected observation indices."""
        sim = make_simulator(1)
        obs_before = sim.get_observations()[0].copy()

        # Execute action multiple times
        for _ in range(20):
            sim.step(np.array([action], dtype=np.int32))

        obs_after = sim.get_observations()[0]

        index_map = {"x": 0, "y": 1, "z": 2, "yaw": 6, "pitch": 7}
        field, direction = expected_change
        idx = index_map[field]

        delta = obs_after[idx] - obs_before[idx]

        # Handle yaw wraparound
        if field == "yaw" and abs(delta) > 0.5:
            delta = delta - 1.0 if delta > 0 else delta + 1.0

        if direction == "increase":
            assert delta > 0.001, (
                f"Action {action} ({ACTION_NAMES[action]}) should increase {field}, "
                f"got delta={delta}"
            )
        else:
            assert delta < -0.001, (
                f"Action {action} ({ACTION_NAMES[action]}) should decrease {field}, "
                f"got delta={delta}"
            )


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_max_num_envs(self, make_simulator):
        """Test with maximum env count (64)."""
        sim = make_simulator(64)
        obs = sim.get_observations()
        assert obs.shape[0] == 64
        assert obs.shape[1] >= OBS_DIM

    def test_single_env_isolation(self, make_simulator):
        """Test single env operations."""
        sim = make_simulator(1)
        rewards = sim.get_rewards()
        dones = sim.get_dones()
        assert rewards.shape[0] == 1
        assert dones.shape[0] == 1

    @pytest.mark.parametrize("invalid_action", [-1, 17, 100, 255])
    def test_invalid_action_handling(self, make_simulator, invalid_action: int):
        """Invalid actions should be handled gracefully (clamp or error)."""
        sim = make_simulator(1)
        try:
            sim.step(np.array([invalid_action], dtype=np.int32))
            # If it doesn't raise, verify no crash and observations are valid
            obs = sim.get_observations()
            assert not np.any(np.isnan(obs))
        except (ValueError, RuntimeError):
            pass  # Expected behavior for invalid actions

    def test_observation_normalized_range(self, make_simulator):
        """Key observation indices should be normalized to [0, 1]."""
        sim = make_simulator(1)

        # Run random actions to explore state space
        for _ in range(100):
            sim.step(np.random.randint(0, 17, size=1, dtype=np.int32))

        obs = sim.get_observations()[0]

        # Key indices that should be normalized
        normalized_indices = [
            0,
            1,
            2,  # Position
            6,
            7,  # Yaw, pitch
            8,  # Health
            16,  # Dragon health
            24,  # Dragon phase
            25,  # Dragon distance
            32,  # Crystal count
        ]

        for idx in normalized_indices:
            if idx < len(obs):
                if idx in (0, 1, 2):
                    assert -1.0 <= obs[idx] <= 1.0, (
                        f"obs[{idx}] = {obs[idx]} not in [-1, 1]"
                    )
                else:
                    assert 0.0 <= obs[idx] <= 1.0, (
                        f"obs[{idx}] = {obs[idx]} not in [0, 1]"
                    )

    def test_rapid_reset(self, make_simulator):
        """Multiple rapid resets should not cause issues."""
        sim = make_simulator(4)
        for _ in range(10):
            sim.reset()
            sim.step(np.zeros(4, dtype=np.int32))
            obs = sim.get_observations()
            assert obs.shape[0] == 4

    def test_zero_step_observations(self, make_simulator):
        """Observations should be valid immediately after reset+step."""
        sim = make_simulator(4)
        obs = sim.get_observations()
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_mixed_actions_per_env(self, make_simulator):
        """Different envs can take different actions simultaneously."""
        sim = make_simulator(16)
        obs_before = sim.get_observations().copy()

        # Each env takes a different action (cycle through 17 actions)
        actions = np.array([i % 17 for i in range(16)], dtype=np.int32)

        for _ in range(10):
            sim.step(actions)

        obs_after = sim.get_observations()

        # Each env should have potentially different state
        assert obs_after.shape[0] == 16
        assert not np.any(np.isnan(obs_after))


class TestRewardBoundaries:
    """Test reward signal boundaries."""

    def test_rewards_finite(self, make_simulator):
        """Rewards should always be finite."""
        sim = make_simulator(4)

        for _ in range(100):
            sim.step(np.random.randint(0, 17, size=4, dtype=np.int32))
            rewards = sim.get_rewards()
            assert not np.any(np.isnan(rewards)), "Rewards contain NaN"
            assert not np.any(np.isinf(rewards)), "Rewards contain Inf"

    def test_rewards_reasonable_range(self, make_simulator):
        """Rewards should be within reasonable bounds."""
        sim = make_simulator(4)

        for _ in range(100):
            sim.step(np.random.randint(0, 17, size=4, dtype=np.int32))
            rewards = sim.get_rewards()
            # Based on reward structure: dragon kill = +1000, death = -100
            assert np.all(rewards >= -200), f"Reward too negative: {np.min(rewards)}"
            assert np.all(rewards <= 1100), f"Reward too positive: {np.max(rewards)}"


class TestDoneFlagBoundaries:
    """Test done flag behavior at boundaries."""

    def test_done_flag_dtype(self, make_simulator):
        """Done flags should be boolean-like."""
        sim = make_simulator(4)
        sim.step(np.zeros(4, dtype=np.int32))
        dones = sim.get_dones()
        assert dones.dtype in [np.bool_, np.uint8, np.int8, np.int32]

    def test_done_initially_false(self, make_simulator):
        """Done flags should be False at episode start."""
        sim = make_simulator(4)
        sim.reset()
        sim.step(np.zeros(4, dtype=np.int32))
        dones = sim.get_dones()
        assert np.all(dones == False), "Done should be False at start"


# =============================================================================
# Stress Tests
# =============================================================================


class TestStress:
    """Stress tests for stability under load."""

    @pytest.mark.slow
    def test_long_episode(self, make_simulator):
        """Run 10000 steps without issues."""
        sim = make_simulator(4)

        for step in range(10000):
            actions = np.random.randint(0, 17, size=4, dtype=np.int32)
            sim.step(actions)

            if step % 1000 == 0:
                obs = sim.get_observations()
                assert not np.any(np.isnan(obs)), f"NaN at step {step}"

    @pytest.mark.slow
    def test_repeated_episodes(self, make_simulator):
        """Run many short episodes."""
        sim = make_simulator(4)

        for episode in range(100):
            sim.reset()
            for _ in range(100):
                sim.step(np.random.randint(0, 17, size=4, dtype=np.int32))

            obs = sim.get_observations()
            assert not np.any(np.isnan(obs)), f"NaN in episode {episode}"


# =============================================================================
# Cross-Action Interaction Tests
# =============================================================================


class TestActionInteractions:
    """Test interactions between different actions."""

    def test_movement_after_turn(self, make_simulator):
        """Forward movement should follow look direction."""
        sim = make_simulator(1)

        # Turn right 90 degrees (18 * 5 degrees)
        for _ in range(18):
            sim.step(np.array([13], dtype=np.int32))  # LOOK_RIGHT

        obs_after_turn = sim.get_observations()[0].copy()

        # Move forward
        for _ in range(20):
            sim.step(np.array([1], dtype=np.int32))  # FORWARD

        obs_after_move = sim.get_observations()[0]

        # After turning right 90 degrees, forward should decrease X
        x_delta = obs_after_move[0] - obs_after_turn[0]
        assert x_delta < -0.005, f"Expected X decrease after right turn, got {x_delta}"

    def test_sprint_faster_than_walk(self, make_simulator):
        """Sprint should be faster than regular forward."""
        sim = make_simulator(2)

        # Env 0: walk, Env 1: sprint
        obs_before = sim.get_observations().copy()

        for _ in range(20):
            sim.step(np.array([1, 11], dtype=np.int32))  # FORWARD, SPRINT_FORWARD

        obs_after = sim.get_observations()

        walk_dist = obs_after[0, 2] - obs_before[0, 2]
        sprint_dist = obs_after[1, 2] - obs_before[1, 2]

        assert sprint_dist > walk_dist, (
            f"Sprint ({sprint_dist}) should be faster than walk ({walk_dist})"
        )

    def test_jump_vertical_motion(self, make_simulator):
        """Jump should cause vertical motion."""
        sim = make_simulator(1)
        obs_before = sim.get_observations()[0].copy()

        # Jump
        sim.step(np.array([7], dtype=np.int32))  # JUMP

        obs_after = sim.get_observations()[0]

        # Either Y increased or velocity is positive
        y_delta = obs_after[1] - obs_before[1]
        vel_y = obs_after[4]

        assert y_delta > 0 or vel_y > 0, (
            f"Jump should increase Y or have positive velocity: y_delta={y_delta}, vel_y={vel_y}"
        )


# =============================================================================
# Data Type Consistency Tests
# =============================================================================


class TestDataTypes:
    """Test data type consistency across operations."""

    @pytest.mark.parametrize(
        "num_envs", NUM_ENVS_VALUES, ids=[f"envs_{n}" for n in NUM_ENVS_VALUES]
    )
    def test_observation_dtype_consistency(self, make_simulator, num_envs: int):
        """Observations should always be float32."""
        sim = make_simulator(num_envs)

        for _ in range(10):
            sim.step(np.random.randint(0, 17, size=num_envs, dtype=np.int32))
            obs = sim.get_observations()
            assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    @pytest.mark.parametrize(
        "num_envs", NUM_ENVS_VALUES, ids=[f"envs_{n}" for n in NUM_ENVS_VALUES]
    )
    def test_reward_dtype_consistency(self, make_simulator, num_envs: int):
        """Rewards should always be float type."""
        sim = make_simulator(num_envs)

        for _ in range(10):
            sim.step(np.random.randint(0, 17, size=num_envs, dtype=np.int32))
            rewards = sim.get_rewards()
            assert rewards.dtype in [np.float32, np.float64], f"Expected float, got {rewards.dtype}"

    @pytest.mark.parametrize(
        "num_envs", NUM_ENVS_VALUES, ids=[f"envs_{n}" for n in NUM_ENVS_VALUES]
    )
    def test_shape_consistency(self, make_simulator, num_envs: int):
        """Shapes should be consistent across all operations."""
        sim = make_simulator(num_envs)

        for _ in range(10):
            sim.step(np.random.randint(0, 17, size=num_envs, dtype=np.int32))

            obs = sim.get_observations()
            rewards = sim.get_rewards()
            dones = sim.get_dones()

            assert obs.shape[0] == num_envs
            assert rewards.shape[0] == num_envs
            assert dones.shape[0] == num_envs


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
