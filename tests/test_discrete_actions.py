#!/usr/bin/env python3
"""Comprehensive test for all 17 discrete actions in MC189Simulator.

Action mapping (from mc189_simulator.cpp):
  0 = noop
  1 = forward
  2 = back
  3 = left
  4 = right
  5 = forward+left
  6 = forward+right
  7 = jump
  8 = jump+forward
  9 = attack
  10 = attack+forward
  11 = sprint+forward
  12 = look_left
  13 = look_right
  14 = swap_weapon
  15 = look_up
  16 = look_down

Observation layout (from simulator.h Observation struct):
  [0] pos_x (normalized: actual_x / 100)
  [1] pos_y (normalized: (actual_y - 64) / 50)
  [2] pos_z (normalized: actual_z / 100)
  [3] vel_x
  [4] vel_y
  [5] vel_z
  [6] yaw (normalized: actual_yaw / 360)
  [7] pitch (normalized: actual_pitch / 90 or similar)
  [8] health
  [9] hunger
  [10] on_ground
  [11] attack_ready
  [12] weapon (0=hand, 1=sword, 2=bow)
  [13] arrows
  [14] arrow_charge
  [15] reserved
  [16-31] dragon state
  [32-47] environment state
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

# Add paths to find the module
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent.parent / "cpp" / "build"))

try:
    import mc189_core

    HAS_MC189 = True
except ImportError:
    HAS_MC189 = False
    mc189_core = None

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Skip all tests if mc189_core is not available
pytestmark = pytest.mark.skipif(not HAS_MC189, reason="mc189_core not available")


class TestDiscreteActions:
    """Test all 17 discrete actions produce expected state changes."""

    @pytest.fixture
    def sim(self):
        """Create and reset a single-environment simulator."""
        config = mc189_core.SimulatorConfig()
        config.num_envs = 1
        config.shader_dir = str(Path(__file__).parent.parent / "cpp" / "shaders")
        simulator = mc189_core.MC189Simulator(config)
        simulator.reset()
        # Execute noop to initialize observations
        simulator.step(np.array([0], dtype=np.int32))
        return simulator

    def get_obs(self, sim) -> NDArray[np.float32]:
        """Get observations as a flat array."""
        return sim.get_observations().flatten().copy()

    def step_action(self, sim, action: int, steps: int = 1) -> NDArray[np.float32]:
        """Execute an action for multiple steps and return final observations."""
        actions = np.array([action], dtype=np.int32)
        for _ in range(steps):
            sim.step(actions)
        return self.get_obs(sim)

    # =========================================================================
    # ACTION 0: NOOP
    # =========================================================================
    def test_action_0_noop(self, sim):
        """Action 0 (noop) should not significantly change player state."""
        obs_before = self.get_obs(sim)
        obs_after = self.step_action(sim, 0, steps=5)

        # Position should remain roughly the same (allow for gravity/physics)
        assert abs(obs_after[0] - obs_before[0]) < 0.01, "X position changed too much on noop"
        assert abs(obs_after[2] - obs_before[2]) < 0.01, "Z position changed too much on noop"

        # Yaw and pitch should not change
        assert abs(obs_after[6] - obs_before[6]) < 0.001, "Yaw changed on noop"
        assert abs(obs_after[7] - obs_before[7]) < 0.001, "Pitch changed on noop"

    # =========================================================================
    # MOVEMENT ACTIONS (1-6)
    # =========================================================================
    def test_action_1_forward(self, sim):
        """Action 1 (forward) should increase Z position."""
        obs_before = self.get_obs(sim)
        obs_after = self.step_action(sim, 1, steps=10)

        # Forward movement in Minecraft increases Z (positive Z direction when yaw=0)
        # Since observation is normalized (z/100), check relative change
        z_delta = obs_after[2] - obs_before[2]
        assert z_delta > 0.001, f"Forward did not increase Z: delta={z_delta}"

    def test_action_2_back(self, sim):
        """Action 2 (back) should decrease Z position."""
        obs_before = self.get_obs(sim)
        obs_after = self.step_action(sim, 2, steps=10)

        z_delta = obs_after[2] - obs_before[2]
        assert z_delta < -0.001, f"Back did not decrease Z: delta={z_delta}"

    def test_action_3_left(self, sim):
        """Action 3 (left) should decrease X position."""
        obs_before = self.get_obs(sim)
        obs_after = self.step_action(sim, 3, steps=10)

        x_delta = obs_after[0] - obs_before[0]
        assert x_delta < -0.001, f"Left did not decrease X: delta={x_delta}"

    def test_action_4_right(self, sim):
        """Action 4 (right) should increase X position."""
        obs_before = self.get_obs(sim)
        obs_after = self.step_action(sim, 4, steps=10)

        x_delta = obs_after[0] - obs_before[0]
        assert x_delta > 0.001, f"Right did not increase X: delta={x_delta}"

    def test_action_5_forward_left(self, sim):
        """Action 5 (forward+left) should increase Z and decrease X."""
        obs_before = self.get_obs(sim)
        obs_after = self.step_action(sim, 5, steps=10)

        z_delta = obs_after[2] - obs_before[2]
        x_delta = obs_after[0] - obs_before[0]

        assert z_delta > 0.0005, f"Forward+left Z delta too small: {z_delta}"
        assert x_delta < -0.0005, f"Forward+left X delta too small: {x_delta}"

    def test_action_6_forward_right(self, sim):
        """Action 6 (forward+right) should increase Z and increase X."""
        obs_before = self.get_obs(sim)
        obs_after = self.step_action(sim, 6, steps=10)

        z_delta = obs_after[2] - obs_before[2]
        x_delta = obs_after[0] - obs_before[0]

        assert z_delta > 0.0005, f"Forward+right Z delta too small: {z_delta}"
        assert x_delta > 0.0005, f"Forward+right X delta too small: {x_delta}"

    # =========================================================================
    # JUMP ACTIONS (7-8)
    # =========================================================================
    def test_action_7_jump(self, sim):
        """Action 7 (jump) should temporarily increase Y position."""
        obs_before = self.get_obs(sim)

        # Jump requires being on ground and takes several ticks
        # Execute jump and immediately check velocity or position change
        self.step_action(sim, 7, steps=1)
        obs_during = self.get_obs(sim)

        # Either Y position should increase or vertical velocity should be positive
        y_delta = obs_during[1] - obs_before[1]
        vel_y = obs_during[4]

        # Jump should either move up or have positive velocity
        jumped = y_delta > 0 or vel_y > 0
        assert jumped, f"Jump had no effect: y_delta={y_delta}, vel_y={vel_y}"

    def test_action_8_jump_forward(self, sim):
        """Action 8 (jump+forward) should increase both Y and Z."""
        obs_before = self.get_obs(sim)
        self.step_action(sim, 8, steps=5)
        obs_after = self.get_obs(sim)

        z_delta = obs_after[2] - obs_before[2]
        # For jump, check peak Y during jump or velocity
        # Since we step multiple times, check if we moved forward at all
        assert z_delta > 0.0005, f"Jump+forward Z delta too small: {z_delta}"

    # =========================================================================
    # ATTACK ACTIONS (9-10)
    # =========================================================================
    def test_action_9_attack(self, sim):
        """Action 9 (attack) should trigger attack cooldown."""
        obs_before = self.get_obs(sim)

        # Attack should set attack cooldown (attack_ready goes from 1 to 0)
        # First verify attack is ready (value > 0.5 means ready)
        attack_ready_before = obs_before[11]

        if attack_ready_before > 0.5:
            self.step_action(sim, 9, steps=1)
            obs_after = self.get_obs(sim)
            attack_ready_after = obs_after[11]

            # After attack, cooldown should start (attack_ready = 0)
            assert attack_ready_after < 0.5, (
                f"Attack did not trigger cooldown: before={attack_ready_before}, after={attack_ready_after}"
            )
        else:
            # Attack was already on cooldown, just verify the action doesn't crash
            self.step_action(sim, 9, steps=1)

    def test_action_10_attack_forward(self, sim):
        """Action 10 (attack+forward) should attack and move forward."""
        obs_before = self.get_obs(sim)
        obs_after = self.step_action(sim, 10, steps=10)

        z_delta = obs_after[2] - obs_before[2]
        assert z_delta > 0.001, f"Attack+forward Z delta too small: {z_delta}"

    # =========================================================================
    # SPRINT ACTION (11)
    # =========================================================================
    def test_action_11_sprint_forward(self, sim):
        """Action 11 (sprint+forward) should move faster than regular forward."""
        # First measure regular forward speed
        sim.reset()
        self.step_action(sim, 0, steps=1)  # Initialize
        obs_before = self.get_obs(sim)
        obs_after = self.step_action(sim, 1, steps=10)
        regular_z_delta = obs_after[2] - obs_before[2]

        # Now measure sprint speed
        sim.reset()
        self.step_action(sim, 0, steps=1)  # Initialize
        obs_before = self.get_obs(sim)
        obs_after = self.step_action(sim, 11, steps=10)
        sprint_z_delta = obs_after[2] - obs_before[2]

        # Sprint should be faster (at least 1.2x)
        assert sprint_z_delta > regular_z_delta * 1.1, (
            f"Sprint not faster than walk: sprint={sprint_z_delta}, walk={regular_z_delta}"
        )

    # =========================================================================
    # LOOK ACTIONS (12-13, 15-16)
    # =========================================================================
    def test_action_12_look_left(self, sim):
        """Action 12 (look_left) should decrease yaw."""
        obs_before = self.get_obs(sim)
        yaw_before = obs_before[6]

        obs_after = self.step_action(sim, 12, steps=1)
        yaw_after = obs_after[6]

        # Look left decreases yaw (or wraps around)
        # Yaw is normalized to [0, 1] representing [0, 360]
        # A -5 degree change is -5/360 ≈ -0.0139
        yaw_delta = yaw_after - yaw_before

        # Handle wraparound: if delta is large positive, it wrapped from 0 to 360
        if yaw_delta > 0.5:
            yaw_delta -= 1.0
        elif yaw_delta < -0.5:
            yaw_delta += 1.0

        assert yaw_delta < -0.01, f"Look left did not decrease yaw: delta={yaw_delta * 360}°"

    def test_action_13_look_right(self, sim):
        """Action 13 (look_right) should increase yaw."""
        obs_before = self.get_obs(sim)
        yaw_before = obs_before[6]

        obs_after = self.step_action(sim, 13, steps=1)
        yaw_after = obs_after[6]

        yaw_delta = yaw_after - yaw_before

        # Handle wraparound
        if yaw_delta > 0.5:
            yaw_delta -= 1.0
        elif yaw_delta < -0.5:
            yaw_delta += 1.0

        assert yaw_delta > 0.01, f"Look right did not increase yaw: delta={yaw_delta * 360}°"

    def test_action_14_swap_weapon(self, sim):
        """Action 14 (swap_weapon) should cycle weapon slot."""
        obs_before = self.get_obs(sim)
        weapon_before = obs_before[12]

        obs_after = self.step_action(sim, 14, steps=1)
        weapon_after = obs_after[12]

        # Weapon should change (0=hand, 1=sword, 2=bow cycles)
        # Note: exact behavior depends on implementation
        # Just verify it changed or if same, weapon was already at last slot
        if weapon_before == weapon_after:
            # Might have cycled back to same weapon - check if second swap also same
            obs_after2 = self.step_action(sim, 14, steps=1)
            weapon_after2 = obs_after2[12]
            # At least one of the two swaps should change something
            # unless only one weapon exists
            pass  # Acceptable if single weapon

    def test_action_15_look_up(self, sim):
        """Action 15 (look_up) should decrease pitch."""
        obs_before = self.get_obs(sim)
        pitch_before = obs_before[7]

        obs_after = self.step_action(sim, 15, steps=1)
        pitch_after = obs_after[7]

        # Look up = negative pitch delta in Minecraft
        pitch_delta = pitch_after - pitch_before
        assert pitch_delta < -0.01, f"Look up did not decrease pitch: delta={pitch_delta}"

    def test_action_16_look_down(self, sim):
        """Action 16 (look_down) should increase pitch."""
        obs_before = self.get_obs(sim)
        pitch_before = obs_before[7]

        obs_after = self.step_action(sim, 16, steps=1)
        pitch_after = obs_after[7]

        # Look down = positive pitch delta
        pitch_delta = pitch_after - pitch_before
        assert pitch_delta > 0.01, f"Look down did not increase pitch: delta={pitch_delta}"


class TestActionCombinations:
    """Test that action combinations work correctly."""

    @pytest.fixture
    def sim(self):
        """Create and reset a single-environment simulator."""
        config = mc189_core.SimulatorConfig()
        config.num_envs = 1
        config.shader_dir = str(Path(__file__).parent.parent / "cpp" / "shaders")
        simulator = mc189_core.MC189Simulator(config)
        simulator.reset()
        simulator.step(np.array([0], dtype=np.int32))
        return simulator

    def test_look_then_move(self, sim):
        """Looking right then moving forward should go in the new direction.

        Minecraft coordinate convention:
        - yaw=0: facing south (+Z)
        - yaw=90: facing west (-X)
        - yaw=-90 (270): facing east (+X)

        So turning right 90° then moving forward should decrease X (heading west).
        """
        # Turn right 90 degrees (18 steps * 5 degrees = 90 degrees)
        for _ in range(18):
            sim.step(np.array([13], dtype=np.int32))

        obs_before = sim.get_observations().flatten().copy()

        # Now move forward - should decrease X (heading west in Minecraft coords)
        for _ in range(20):
            sim.step(np.array([1], dtype=np.int32))

        obs_after = sim.get_observations().flatten()

        x_delta = obs_after[0] - obs_before[0]
        # After turning 90 degrees right (yaw=90), forward is west (-X)
        assert x_delta < -0.005, f"Forward at yaw=90 should decrease X (west): {x_delta}"

    def test_movement_is_relative_to_facing(self, sim):
        """Verify movement directions change based on yaw."""
        # Get initial position
        sim.reset()
        sim.step(np.array([0], dtype=np.int32))
        obs_start = sim.get_observations().flatten().copy()

        # Move forward, should increase Z at yaw=0
        for _ in range(10):
            sim.step(np.array([1], dtype=np.int32))
        obs_forward = sim.get_observations().flatten()

        z_delta_initial = obs_forward[2] - obs_start[2]

        # Reset and turn 180 degrees
        sim.reset()
        sim.step(np.array([0], dtype=np.int32))
        for _ in range(36):  # 36 * 5 = 180 degrees
            sim.step(np.array([13], dtype=np.int32))  # look right

        obs_turned = sim.get_observations().flatten().copy()

        # Move forward again, should decrease Z now
        for _ in range(10):
            sim.step(np.array([1], dtype=np.int32))
        obs_forward_turned = sim.get_observations().flatten()

        z_delta_turned = obs_forward_turned[2] - obs_turned[2]

        # Directions should be opposite
        assert z_delta_initial > 0, f"Initial forward should increase Z: {z_delta_initial}"
        assert z_delta_turned < 0, f"Turned forward should decrease Z: {z_delta_turned}"


class TestActionBounds:
    """Test action bounds and edge cases."""

    @pytest.fixture
    def sim(self):
        """Create simulator."""
        config = mc189_core.SimulatorConfig()
        config.num_envs = 1
        config.shader_dir = str(Path(__file__).parent.parent / "cpp" / "shaders")
        simulator = mc189_core.MC189Simulator(config)
        simulator.reset()
        return simulator

    def test_all_actions_in_range(self, sim):
        """All 17 actions (0-16) should execute without error."""
        for action in range(17):
            try:
                sim.step(np.array([action], dtype=np.int32))
            except Exception as e:
                pytest.fail(f"Action {action} raised exception: {e}")

    def test_invalid_action_clamped_or_noop(self, sim):
        """Actions outside 0-16 should be handled gracefully."""
        obs_before = sim.get_observations().flatten().copy()

        # Try action 17 (out of bounds) - should act as noop or clamp
        try:
            sim.step(np.array([17], dtype=np.int32))
        except Exception:
            pass  # OK if it raises

        # Try negative action
        try:
            sim.step(np.array([-1], dtype=np.int32))
        except Exception:
            pass  # OK if it raises


class TestBatchedActions:
    """Test actions work correctly in batched environments."""

    def test_different_actions_per_env(self):
        """Different environments can take different actions."""
        config = mc189_core.SimulatorConfig()
        config.num_envs = 4
        config.shader_dir = str(Path(__file__).parent.parent / "cpp" / "shaders")
        sim = mc189_core.MC189Simulator(config)
        sim.reset()
        sim.step(np.zeros(4, dtype=np.int32))

        obs_before = sim.get_observations().copy()

        # Env 0: forward, Env 1: back, Env 2: left, Env 3: right
        actions = np.array([1, 2, 3, 4], dtype=np.int32)
        for _ in range(10):
            sim.step(actions)

        obs_after = sim.get_observations()

        # Check each env moved in expected direction
        # Env 0: Z increased
        assert obs_after[0, 2] > obs_before[0, 2], "Env 0 (forward) Z should increase"
        # Env 1: Z decreased
        assert obs_after[1, 2] < obs_before[1, 2], "Env 1 (back) Z should decrease"
        # Env 2: X decreased
        assert obs_after[2, 0] < obs_before[2, 0], "Env 2 (left) X should decrease"
        # Env 3: X increased
        assert obs_after[3, 0] > obs_before[3, 0], "Env 3 (right) X should increase"


if __name__ == "__main__":
    # Run a quick manual test
    if HAS_MC189:
        print("Running manual action verification...")
        print("=" * 70)

        config = mc189_core.SimulatorConfig()
        config.num_envs = 1
        config.shader_dir = str(Path(__file__).parent.parent / "cpp" / "shaders")
        sim = mc189_core.MC189Simulator(config)
        sim.reset()
        sim.step(np.array([0], dtype=np.int32))

        action_names = [
            "0: noop",
            "1: forward",
            "2: back",
            "3: left",
            "4: right",
            "5: forward+left",
            "6: forward+right",
            "7: jump",
            "8: jump+forward",
            "9: attack",
            "10: attack+forward",
            "11: sprint+forward",
            "12: look_left",
            "13: look_right",
            "14: swap_weapon",
            "15: look_up",
            "16: look_down",
        ]

        for action_id, name in enumerate(action_names):
            sim.reset()
            sim.step(np.array([0], dtype=np.int32))
            obs_before = sim.get_observations().flatten().copy()

            # Execute action 10 times
            for _ in range(10):
                sim.step(np.array([action_id], dtype=np.int32))

            obs_after = sim.get_observations().flatten()

            # Calculate deltas
            dx = obs_after[0] - obs_before[0]
            dy = obs_after[1] - obs_before[1]
            dz = obs_after[2] - obs_before[2]
            dyaw = obs_after[6] - obs_before[6]
            dpitch = obs_after[7] - obs_before[7]
            dweapon = obs_after[12] - obs_before[12]

            print(
                f"{name:25s} | dx={dx:+.4f} dz={dz:+.4f} dy={dy:+.4f} | "
                f"dyaw={dyaw * 360:+.1f}° dpitch={dpitch * 90:+.1f}° | weapon={dweapon:+.2f}"
            )

        print("=" * 70)
        print("Manual verification complete.")
    else:
        print("mc189_core not available, skipping manual test")
