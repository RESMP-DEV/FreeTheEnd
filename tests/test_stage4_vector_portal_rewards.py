"""Stage 4 vector environment portal reward alignment tests.

Verifies that Stage 4 (Enderman Hunting) shaped rewards for portal completion
are consistent between single-env and vector configurations. The reward shaper
must produce identical portal-related rewards regardless of how many parallel
environments are running.

Key portal rewards tested:
- portal_frames_filled: 0.1 per eye placed (progressive)
- end_portal_activated: 1.5 milestone bonus
- eye_of_ender crafting milestones (first_eye, eye_x6, eye_x12)

Run with: uv run pytest contrib/minecraft_sim/tests/test_stage4_vector_portal_rewards.py -v
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

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
    if "minecraft_sim.reward_shaping" in sys.modules:
        del sys.modules["minecraft_sim.reward_shaping"]

    from minecraft_sim import mc189_core
    from minecraft_sim.reward_shaping import create_stage4_reward_shaper

    HAS_MC189_CORE = True
    _import_error = ""
except ImportError as e:
    HAS_MC189_CORE = False
    mc189_core = None
    create_stage4_reward_shaper = None  # type: ignore[assignment]
    _import_error = str(e)

pytestmark = pytest.mark.skipif(
    not HAS_MC189_CORE, reason=f"mc189_core C++ extension not available: {_import_error}"
)

# =============================================================================
# Test State Builders
# =============================================================================


def _base_state(
    health: float = 20.0,
    ender_pearls: int = 0,
    eye_of_ender: int = 0,
    eyes_crafted: int = 0,
    portal_frames_filled: int = 0,
    end_portal_activated: bool = False,
    endermen_killed: int = 0,
    blaze_powder: int = 0,
    time_of_day: float = 15000.0,
) -> dict:
    """Build a Stage 4 state dict with portal-related fields."""
    return {
        "health": health,
        "inventory": {
            "ender_pearl": ender_pearls,
            "eye_of_ender": eye_of_ender,
            "blaze_powder": blaze_powder,
        },
        "eyes_crafted": eyes_crafted,
        "portal_frames_filled": portal_frames_filled,
        "end_portal_activated": end_portal_activated,
        "endermen_killed": endermen_killed,
        "time_of_day": time_of_day,
        "armor_equipped": 0,
    }


# =============================================================================
# Single-Env vs Vector Reward Alignment
# =============================================================================


class TestPortalRewardAlignment:
    """Verify portal-related shaped rewards match between single and vector configs."""

    def test_portal_frame_progressive_reward_single_vs_multi(self):
        """Progressive portal frame filling reward must match across N shapers.

        Each independent shaper (one per vector env slot) should produce the
        same reward when given the same state transition sequence.
        """
        num_envs = 4

        # Create N independent shapers (simulates vector env with per-env shapers)
        shapers = [create_stage4_reward_shaper() for _ in range(num_envs)]
        single_shaper = create_stage4_reward_shaper()

        # Initial state: 12 pearls, 12 eyes crafted, portal room found
        prev_state = _base_state(ender_pearls=12, eye_of_ender=12, eyes_crafted=12)

        # Prime all shapers with the same previous state
        single_shaper(prev_state)
        for s in shapers:
            s(prev_state)

        # State transition: place first eye in portal frame
        next_state = _base_state(
            ender_pearls=12,
            eye_of_ender=11,
            eyes_crafted=12,
            portal_frames_filled=1,
        )

        single_reward = single_shaper(next_state)
        vector_rewards = [s(deepcopy(next_state)) for s in shapers]

        # All vector env rewards must match the single-env reward
        for i, vr in enumerate(vector_rewards):
            assert vr == pytest.approx(single_reward, abs=1e-7), (
                f"Env {i} reward {vr} != single-env reward {single_reward} "
                f"for first portal frame placement"
            )

    def test_portal_frame_incremental_filling(self):
        """Incremental portal frame filling (1->12) produces consistent rewards.

        Simulates a full portal fill sequence and verifies that every
        single-env shaper step matches every vector-env shaper step.
        """
        num_envs = 8
        shapers_vec = [create_stage4_reward_shaper() for _ in range(num_envs)]
        shaper_single = create_stage4_reward_shaper()

        # Start with 12 eyes in inventory, portal room ready
        state = _base_state(ender_pearls=0, eye_of_ender=12, eyes_crafted=12)

        # Prime shapers
        shaper_single(state)
        for s in shapers_vec:
            s(deepcopy(state))

        # Fill frames one at a time
        for frame_count in range(1, 13):
            state = _base_state(
                ender_pearls=0,
                eye_of_ender=12 - frame_count,
                eyes_crafted=12,
                portal_frames_filled=frame_count,
            )

            r_single = shaper_single(state)
            r_vec = [s(deepcopy(state)) for s in shapers_vec]

            for i, rv in enumerate(r_vec):
                assert rv == pytest.approx(r_single, abs=1e-7), (
                    f"Frame {frame_count}: env {i} reward {rv} != "
                    f"single reward {r_single}"
                )

    def test_portal_activation_milestone_single_vs_multi(self):
        """Portal activation (1.5 bonus) fires identically across all shapers."""
        num_envs = 16
        shapers_vec = [create_stage4_reward_shaper() for _ in range(num_envs)]
        shaper_single = create_stage4_reward_shaper()

        # State before activation: 12 frames filled, not yet activated
        pre_activation = _base_state(
            ender_pearls=0,
            eye_of_ender=0,
            eyes_crafted=12,
            portal_frames_filled=12,
            end_portal_activated=False,
        )

        shaper_single(pre_activation)
        for s in shapers_vec:
            s(deepcopy(pre_activation))

        # Activate portal
        post_activation = _base_state(
            ender_pearls=0,
            eye_of_ender=0,
            eyes_crafted=12,
            portal_frames_filled=12,
            end_portal_activated=True,
        )

        r_single = shaper_single(post_activation)
        r_vec = [s(deepcopy(post_activation)) for s in shapers_vec]

        for i, rv in enumerate(r_vec):
            assert rv == pytest.approx(r_single, abs=1e-7), (
                f"Portal activation: env {i} reward {rv} != single {r_single}"
            )

        # Verify the activation bonus was included (1.5 milestone)
        # The reward includes time penalty (-0.00015) but no damage/kill deltas
        assert r_single > 1.0, (
            f"Portal activation reward {r_single} should include 1.5 bonus"
        )

    def test_portal_activation_not_double_counted(self):
        """Portal activation milestone fires exactly once per shaper instance."""
        num_envs = 4
        shapers = [create_stage4_reward_shaper() for _ in range(num_envs)]

        pre = _base_state(portal_frames_filled=12, end_portal_activated=False)
        post = _base_state(portal_frames_filled=12, end_portal_activated=True)

        # Prime
        for s in shapers:
            s(deepcopy(pre))

        # First call with activation: should include bonus
        first_rewards = [s(deepcopy(post)) for s in shapers]

        # Second call with same activated state: bonus already given
        second_rewards = [s(deepcopy(post)) for s in shapers]

        for i in range(num_envs):
            assert first_rewards[i] > second_rewards[i], (
                f"Env {i}: first reward {first_rewards[i]} should be > "
                f"second reward {second_rewards[i]} (milestone not one-shot)"
            )

            # The difference should be approximately the 1.5 activation bonus
            delta = first_rewards[i] - second_rewards[i]
            assert delta == pytest.approx(1.5, abs=0.01), (
                f"Env {i}: milestone delta {delta} != expected 1.5"
            )


class TestPortalRewardShapingIsolation:
    """Verify reward shaper state isolation between vector env slots."""

    def test_shaper_state_independence(self):
        """Each vector env shaper tracks milestones independently.

        One env activating the portal should not affect another env's
        milestone tracking.
        """
        shaper_a = create_stage4_reward_shaper()
        shaper_b = create_stage4_reward_shaper()

        # Both start at same state
        base = _base_state(portal_frames_filled=12, end_portal_activated=False)
        shaper_a(deepcopy(base))
        shaper_b(deepcopy(base))

        # Only env A activates portal
        activated = _base_state(portal_frames_filled=12, end_portal_activated=True)
        r_a = shaper_a(deepcopy(activated))

        # Env B stays inactive
        still_inactive = _base_state(portal_frames_filled=12, end_portal_activated=False)
        r_b = shaper_b(deepcopy(still_inactive))

        # A should have the 1.5 bonus, B should not
        assert r_a > r_b, (
            f"Shaper A (activated) reward {r_a} should be > "
            f"shaper B (not activated) {r_b}"
        )
        assert r_a - r_b == pytest.approx(1.5, abs=0.05), (
            f"Difference {r_a - r_b} should be ~1.5 (portal activation bonus)"
        )

    def test_progressive_frame_rewards_independent(self):
        """Progressive frame rewards don't leak between shapers.

        If env 0 places 5 frames and env 1 places 2 frames, their
        progressive rewards must reflect their individual state.
        """
        shaper_0 = create_stage4_reward_shaper()
        shaper_1 = create_stage4_reward_shaper()

        # Same initial state
        init = _base_state(eye_of_ender=12, eyes_crafted=12, portal_frames_filled=0)
        shaper_0(deepcopy(init))
        shaper_1(deepcopy(init))

        # Env 0 places 5 frames, env 1 places 2
        state_0 = _base_state(eye_of_ender=7, eyes_crafted=12, portal_frames_filled=5)
        state_1 = _base_state(eye_of_ender=10, eyes_crafted=12, portal_frames_filled=2)

        r_0 = shaper_0(deepcopy(state_0))
        r_1 = shaper_1(deepcopy(state_1))

        # Env 0 should get more portal frame reward (0.1 * 5 vs 0.1 * 2)
        # Both also get eye_of_ender progressive change (negative since eyes decrease)
        # Frame reward delta: 0.5 - 0.2 = 0.3
        assert r_0 > r_1, (
            f"5-frame reward {r_0} should be > 2-frame reward {r_1}"
        )

    def test_eye_crafting_milestones_fire_per_shaper(self):
        """Eye crafting milestones (first_eye, eye_x6, eye_x12) fire per shaper.

        Verifies that the milestone bonus for crafting eyes of ender
        is correctly given to each shaper independently when the condition
        is met, and that all shapers agree when given identical state.
        """
        num_envs = 4
        shapers = [create_stage4_reward_shaper() for _ in range(num_envs)]

        # No eyes yet
        init = _base_state(ender_pearls=12, blaze_powder=12)
        for s in shapers:
            s(deepcopy(init))

        # Craft first eye
        first_eye = _base_state(
            ender_pearls=11,
            blaze_powder=11,
            eye_of_ender=1,
            eyes_crafted=1,
        )

        rewards = [s(deepcopy(first_eye)) for s in shapers]

        # All should fire first_eye milestone (0.2)
        for i in range(1, num_envs):
            assert rewards[i] == pytest.approx(rewards[0], abs=1e-7), (
                f"Env {i} first_eye reward {rewards[i]} != env 0 {rewards[0]}"
            )

        # Verify milestone was included
        assert rewards[0] > 0.0, "First eye milestone should give positive reward"


class TestVectorEnvPortalRewardShaping:
    """Integration tests using actual simulator vector environments.

    These tests verify that the SpeedrunVecEnv reward pipeline correctly
    applies Stage 4 reward shaping in a multi-environment configuration
    and that portal rewards align with the standalone shaper output.
    """

    @pytest.fixture
    def sim_single(self):
        """Single-env simulator for Stage 4."""
        config = mc189_core.SimulatorConfig()
        config.num_envs = 1
        config.shader_dir = str(SHADERS_DIR)
        sim = mc189_core.MC189Simulator(config)
        sim.reset()
        sim.step(np.array([0], dtype=np.int32))
        return sim

    @pytest.fixture
    def sim_vector(self):
        """4-env simulator for Stage 4."""
        config = mc189_core.SimulatorConfig()
        config.num_envs = 4
        config.shader_dir = str(SHADERS_DIR)
        sim = mc189_core.MC189Simulator(config)
        sim.reset()
        sim.step(np.zeros(4, dtype=np.int32))
        return sim

    def test_raw_rewards_shape_matches_num_envs(self, sim_single, sim_vector):
        """Raw reward arrays have correct shape for both configurations."""
        r1 = sim_single.get_rewards()
        r4 = sim_vector.get_rewards()

        assert r1.shape == (1,), f"Single env rewards shape {r1.shape} != (1,)"
        assert r4.shape == (4,), f"Vector env rewards shape {r4.shape} != (4,)"

    def test_identical_actions_produce_consistent_rewards(self, sim_vector):
        """All envs taking the same action produce finite, consistent rewards.

        When all vector envs take identical actions from the same reset state,
        their raw rewards should be identical (same initial conditions).
        """
        # All envs take NOOP for 10 steps
        for _ in range(10):
            sim_vector.step(np.zeros(4, dtype=np.int32))

        rewards = sim_vector.get_rewards()
        assert not np.any(np.isnan(rewards)), "Rewards contain NaN"
        assert not np.any(np.isinf(rewards)), "Rewards contain Inf"

        # All envs started identically and took same actions: rewards should match
        assert np.all(rewards == rewards[0]), (
            f"Identical-action rewards differ: {rewards}"
        )

    def test_shaped_rewards_additive_with_portal_state(self):
        """Reward shaping adds portal bonuses correctly atop raw rewards.

        Verifies that applying the Stage 4 shaper to a portal activation
        state yields a reward that includes the 1.5 bonus regardless of
        whether it's applied to 1 env or N envs.
        """
        num_envs = 4
        shapers = [create_stage4_reward_shaper() for _ in range(num_envs)]

        # Simulate raw reward from sim (small positive or zero)
        raw_rewards = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32)

        # Apply shaping: pre-activation state
        pre = _base_state(portal_frames_filled=12, end_portal_activated=False)
        for s in shapers:
            s(deepcopy(pre))

        # Apply shaping: post-activation state
        post = _base_state(portal_frames_filled=12, end_portal_activated=True)
        shaped_bonuses = np.array(
            [s(deepcopy(post)) for s in shapers], dtype=np.float32
        )

        # Total shaped reward = raw + shaping
        total = raw_rewards + shaped_bonuses

        # All envs should get the same total
        assert np.all(total == pytest.approx(total[0], abs=1e-6)), (
            f"Shaped totals differ across envs: {total}"
        )

        # The shaping bonus should include the 1.5 activation reward
        assert np.all(shaped_bonuses > 1.0), (
            f"Shaped bonuses {shaped_bonuses} should include 1.5 activation"
        )

    def test_single_env_vs_vector_env_reward_trajectory(self, sim_single, sim_vector):
        """Single-env and vector-env produce same reward trajectory for same actions.

        Steps both configs with identical actions and verifies that env 0
        of the vector config matches the single env at each step.
        """
        num_steps = 50

        single_rewards = []
        vector_env0_rewards = []

        for _ in range(num_steps):
            # Same action for all
            sim_single.step(np.array([0], dtype=np.int32))
            sim_vector.step(np.zeros(4, dtype=np.int32))

            r1 = sim_single.get_rewards()[0]
            r4 = sim_vector.get_rewards()[0]

            single_rewards.append(r1)
            vector_env0_rewards.append(r4)

        single_arr = np.array(single_rewards, dtype=np.float32)
        vector_arr = np.array(vector_env0_rewards, dtype=np.float32)

        # Trajectories should be identical (same initial state, same actions)
        np.testing.assert_allclose(
            single_arr, vector_arr, atol=1e-6,
            err_msg="Single-env vs vector-env[0] reward trajectories diverge",
        )


class TestPortalRewardEdgeCases:
    """Edge cases for portal reward shaping in vector configurations."""

    def test_partial_portal_reset_does_not_carry_milestones(self):
        """Resetting a shaper clears portal milestones."""
        shaper = create_stage4_reward_shaper()

        # Activate portal
        pre = _base_state(portal_frames_filled=12, end_portal_activated=False)
        shaper(pre)
        post = _base_state(portal_frames_filled=12, end_portal_activated=True)
        r1 = shaper(post)
        assert r1 > 1.0  # Got the 1.5 bonus

        # Reset
        shaper.reset()

        # Re-run the same sequence: milestone should fire again
        shaper(deepcopy(pre))
        r2 = shaper(deepcopy(post))
        assert r2 == pytest.approx(r1, abs=1e-7), (
            f"After reset, same sequence should produce same reward: {r2} vs {r1}"
        )

    def test_simultaneous_frame_and_activation(self):
        """Placing final frame and activating portal in same step.

        When the last eye is placed and portal activates simultaneously,
        both the frame progressive reward and activation milestone should fire.
        """
        num_envs = 4
        shapers = [create_stage4_reward_shaper() for _ in range(num_envs)]

        # 11 frames filled, not activated
        pre = _base_state(
            eye_of_ender=1,
            eyes_crafted=12,
            portal_frames_filled=11,
            end_portal_activated=False,
        )
        for s in shapers:
            s(deepcopy(pre))

        # Place final frame AND activate in one transition
        post = _base_state(
            eye_of_ender=0,
            eyes_crafted=12,
            portal_frames_filled=12,
            end_portal_activated=True,
        )

        rewards = [s(deepcopy(post)) for s in shapers]

        # Should get frame reward (0.1 for 1 frame) + activation (1.5)
        # Plus time penalty and progressive eye changes
        for i, r in enumerate(rewards):
            assert r > 1.0, (
                f"Env {i}: combined frame+activation reward {r} too low"
            )

        # All envs should agree
        for i in range(1, num_envs):
            assert rewards[i] == pytest.approx(rewards[0], abs=1e-7), (
                f"Env {i} reward {rewards[i]} != env 0 {rewards[0]}"
            )

    def test_zero_frames_no_portal_reward(self):
        """No portal reward when no frames are filled."""
        shaper = create_stage4_reward_shaper()

        state = _base_state(
            ender_pearls=12,
            eye_of_ender=12,
            eyes_crafted=12,
            portal_frames_filled=0,
        )
        shaper(deepcopy(state))

        # No change in frames
        same_state = deepcopy(state)
        r = shaper(same_state)

        # Should only have time penalty (and possibly night hunting bonus),
        # no portal-related reward
        assert r < 0.1, f"No portal activity should not give large reward: {r}"

    def test_death_state_overrides_portal_reward(self):
        """Death penalty takes precedence even with portal activation."""
        num_envs = 2
        shapers = [create_stage4_reward_shaper() for _ in range(num_envs)]

        pre = _base_state(health=20.0, portal_frames_filled=12)
        for s in shapers:
            s(deepcopy(pre))

        # Env 0: dies while activating portal
        dead_activated = _base_state(
            health=0.0,
            portal_frames_filled=12,
            end_portal_activated=True,
        )

        # Env 1: alive, activates portal normally
        alive_activated = _base_state(
            health=20.0,
            portal_frames_filled=12,
            end_portal_activated=True,
        )

        r_dead = shapers[0](deepcopy(dead_activated))
        r_alive = shapers[1](deepcopy(alive_activated))

        # Dead env gets death penalty (-1.0), early return, no portal bonus
        assert r_dead < 0, f"Dead env should have negative reward: {r_dead}"
        # Alive env gets portal activation bonus
        assert r_alive > 1.0, f"Alive env should get portal bonus: {r_alive}"

    @pytest.mark.parametrize("num_envs", [1, 4, 16, 64])
    def test_portal_reward_scales_correctly_across_env_counts(self, num_envs: int):
        """Portal activation reward magnitude is independent of num_envs.

        The per-env reward for portal activation must be 1.5 regardless
        of how many parallel environments exist.
        """
        shapers = [create_stage4_reward_shaper() for _ in range(num_envs)]

        pre = _base_state(portal_frames_filled=12, end_portal_activated=False)
        for s in shapers:
            s(deepcopy(pre))

        post = _base_state(portal_frames_filled=12, end_portal_activated=True)
        rewards = [s(deepcopy(post)) for s in shapers]

        # All rewards should be identical
        for i in range(1, num_envs):
            assert rewards[i] == pytest.approx(rewards[0], abs=1e-7), (
                f"num_envs={num_envs}: env {i} reward differs from env 0"
            )

        # The reward should contain the 1.5 activation bonus
        # (minus time penalty of 0.00015)
        assert rewards[0] == pytest.approx(1.5 - 0.00015, abs=0.01), (
            f"Portal activation reward {rewards[0]} != expected ~1.5"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
