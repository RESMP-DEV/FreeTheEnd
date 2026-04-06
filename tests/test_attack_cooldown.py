#!/usr/bin/env python3
"""Test that attack cooldown prevents spam attacks.

Attack cooldown is 10 ticks (from dragon_fight_mvk.comp line 553).
When player attacks:
1. Cooldown is set to 10
2. Subsequent attacks during cooldown deal no damage
3. After 10 ticks of waiting, attacks work again
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
    pass

# Skip all tests if mc189_core is not available
pytestmark = pytest.mark.skipif(not HAS_MC189, reason="mc189_core not available")

# Action constants (from test_discrete_actions.py)
NOOP = 0
ATTACK = 9

# Observation indices
OBS_ATTACK_READY = 11  # 1.0 if ready, 0.0 if on cooldown

# Cooldown constant from dragon_fight_mvk.comp line 553
ATTACK_COOLDOWN_TICKS = 10


class TestAttackCooldown:
    """Test attack cooldown prevents spam attacks."""

    @pytest.fixture
    def sim(self):
        """Create and reset a single-environment simulator."""
        config = mc189_core.SimulatorConfig()
        config.num_envs = 1
        config.shader_dir = str(Path(__file__).parent.parent / "cpp" / "shaders")
        simulator = mc189_core.MC189Simulator(config)
        simulator.reset()
        # Initialize with a noop
        simulator.step(np.array([NOOP], dtype=np.int32))
        return simulator

    def get_obs(self, sim) -> np.ndarray:
        """Get observations as a flat array."""
        return sim.get_observations().flatten().copy()

    def test_initial_attack_ready(self, sim):
        """Verify attack is ready at the start."""
        obs = self.get_obs(sim)
        attack_ready = obs[OBS_ATTACK_READY]
        assert attack_ready > 0.5, f"Attack should be ready initially, got {attack_ready}"

    def test_attack_triggers_cooldown(self, sim):
        """First attack should trigger cooldown."""
        obs_before = self.get_obs(sim)
        assert obs_before[OBS_ATTACK_READY] > 0.5, "Attack should be ready before first attack"

        # Attack once
        sim.step(np.array([ATTACK], dtype=np.int32))
        obs_after = self.get_obs(sim)

        # Cooldown should now be active (attack_ready = 0)
        assert obs_after[OBS_ATTACK_READY] < 0.5, (
            f"Attack should trigger cooldown: attack_ready={obs_after[OBS_ATTACK_READY]}"
        )

    def test_second_attack_blocked_by_cooldown(self, sim):
        """Second immediate attack should NOT reset cooldown (proving it's blocked)."""
        # First attack
        sim.step(np.array([ATTACK], dtype=np.int32))
        obs_after_first = self.get_obs(sim)
        assert obs_after_first[OBS_ATTACK_READY] < 0.5, "First attack should trigger cooldown"

        # Wait 1 tick doing nothing
        sim.step(np.array([NOOP], dtype=np.int32))
        obs_tick1 = self.get_obs(sim)

        # Still on cooldown (only 1 tick passed, need 10)
        assert obs_tick1[OBS_ATTACK_READY] < 0.5, "Still on cooldown after 1 tick"

        # Try attacking again - should be blocked
        sim.step(np.array([ATTACK], dtype=np.int32))
        obs_after_second = self.get_obs(sim)

        # Still on cooldown - attack was blocked
        assert obs_after_second[OBS_ATTACK_READY] < 0.5, (
            "Second attack during cooldown should be blocked"
        )

    def test_cooldown_recovers_after_10_ticks(self, sim):
        """After waiting 10 ticks, attack should be ready again."""
        # First attack
        sim.step(np.array([ATTACK], dtype=np.int32))
        obs = self.get_obs(sim)
        assert obs[OBS_ATTACK_READY] < 0.5, "Attack should trigger cooldown"

        # Wait 10 ticks (cooldown duration)
        for tick in range(ATTACK_COOLDOWN_TICKS):
            sim.step(np.array([NOOP], dtype=np.int32))
            obs = self.get_obs(sim)

            if tick < ATTACK_COOLDOWN_TICKS - 1:
                # Still on cooldown during first 9 ticks
                assert obs[OBS_ATTACK_READY] < 0.5, (
                    f"Should still be on cooldown at tick {tick + 1}/{ATTACK_COOLDOWN_TICKS}"
                )

        # After 10 ticks, should be ready
        obs_final = self.get_obs(sim)
        assert obs_final[OBS_ATTACK_READY] > 0.5, (
            f"Attack should be ready after {ATTACK_COOLDOWN_TICKS} ticks, "
            f"got attack_ready={obs_final[OBS_ATTACK_READY]}"
        )

    def test_attack_works_after_cooldown(self, sim):
        """Attack, wait for cooldown, attack again - second attack should work."""
        # First attack
        sim.step(np.array([ATTACK], dtype=np.int32))
        assert self.get_obs(sim)[OBS_ATTACK_READY] < 0.5, "First attack should trigger cooldown"

        # Wait exactly 10 ticks
        for _ in range(ATTACK_COOLDOWN_TICKS):
            sim.step(np.array([NOOP], dtype=np.int32))

        obs_after_wait = self.get_obs(sim)
        assert obs_after_wait[OBS_ATTACK_READY] > 0.5, (
            f"Should be ready after {ATTACK_COOLDOWN_TICKS} ticks"
        )

        # Second attack should now work and trigger cooldown again
        sim.step(np.array([ATTACK], dtype=np.int32))
        obs_after_second = self.get_obs(sim)

        assert obs_after_second[OBS_ATTACK_READY] < 0.5, (
            "Second attack after cooldown recovery should trigger new cooldown"
        )

    def test_spam_attacks_do_not_reset_cooldown(self, sim):
        """Spamming attack during cooldown should not reset the cooldown timer."""
        # First attack starts cooldown
        sim.step(np.array([ATTACK], dtype=np.int32))

        # Spam attacks for 5 ticks (half the cooldown)
        for _ in range(5):
            sim.step(np.array([ATTACK], dtype=np.int32))

        obs_after_spam = self.get_obs(sim)
        assert obs_after_spam[OBS_ATTACK_READY] < 0.5, "Still on cooldown during spam"

        # Wait 5 more ticks (total 11 ticks since first attack)
        # If spam reset cooldown, we'd need 15 more ticks (10 from each spam)
        # But spam shouldn't reset, so 5 more should be enough
        for _ in range(5):
            sim.step(np.array([NOOP], dtype=np.int32))

        obs_final = self.get_obs(sim)
        assert obs_final[OBS_ATTACK_READY] > 0.5, (
            "Attack should be ready after original cooldown expires, "
            "spam attacks should not extend cooldown"
        )

    def test_cooldown_prevents_damage_spam(self, sim):
        """Verify rapid attacks don't deal damage every tick (would be OP)."""
        # This test verifies the intent: spam attacks = no extra damage
        # We spam attack for 20 ticks and check attack is blocked most of the time

        attacks_blocked = 0
        attacks_ready = 0

        for _ in range(20):
            obs_before = self.get_obs(sim)
            if obs_before[OBS_ATTACK_READY] > 0.5:
                attacks_ready += 1
            else:
                attacks_blocked += 1

            sim.step(np.array([ATTACK], dtype=np.int32))

        # With 10 tick cooldown, only 2 attacks should land in 20 ticks
        # (tick 0 and tick 10)
        assert attacks_ready <= 3, (
            f"Too many attacks landed ({attacks_ready}/20), cooldown not working"
        )
        assert attacks_blocked >= 17, (
            f"Not enough attacks blocked ({attacks_blocked}/20), cooldown not working"
        )


if __name__ == "__main__":
    # Run a quick manual test
    if HAS_MC189:
        print("Attack Cooldown Verification")
        print("=" * 60)
        print(f"Expected cooldown: {ATTACK_COOLDOWN_TICKS} ticks")
        print()

        config = mc189_core.SimulatorConfig()
        config.num_envs = 1
        config.shader_dir = str(Path(__file__).parent.parent / "cpp" / "shaders")
        sim = mc189_core.MC189Simulator(config)
        sim.reset()
        sim.step(np.array([NOOP], dtype=np.int32))

        print("Test 1: Attack triggers cooldown")
        obs = sim.get_observations().flatten()
        print(f"  Before attack: attack_ready = {obs[OBS_ATTACK_READY]}")

        sim.step(np.array([ATTACK], dtype=np.int32))
        obs = sim.get_observations().flatten()
        print(f"  After attack:  attack_ready = {obs[OBS_ATTACK_READY]}")
        print("  ✓ Cooldown triggered" if obs[OBS_ATTACK_READY] < 0.5 else "  ✗ FAILED")
        print()

        print(f"Test 2: Cooldown lasts {ATTACK_COOLDOWN_TICKS} ticks")
        sim.reset()
        sim.step(np.array([NOOP], dtype=np.int32))
        sim.step(np.array([ATTACK], dtype=np.int32))  # Start cooldown

        for tick in range(ATTACK_COOLDOWN_TICKS + 2):
            obs = sim.get_observations().flatten()
            status = "READY" if obs[OBS_ATTACK_READY] > 0.5 else "cooldown"
            expected = "READY" if tick >= ATTACK_COOLDOWN_TICKS else "cooldown"
            match = "✓" if status == expected else "✗"
            print(f"  Tick {tick:2d}: {status:8s} (expected {expected}) {match}")
            sim.step(np.array([NOOP], dtype=np.int32))
        print()

        print("Test 3: Spam attacks don't extend cooldown")
        sim.reset()
        sim.step(np.array([NOOP], dtype=np.int32))
        sim.step(np.array([ATTACK], dtype=np.int32))  # Start cooldown

        # Spam for 5 ticks
        for _ in range(5):
            sim.step(np.array([ATTACK], dtype=np.int32))

        # Wait 5 more (total 11 since first attack)
        for tick in range(6):
            obs = sim.get_observations().flatten()
            status = "READY" if obs[OBS_ATTACK_READY] > 0.5 else "cooldown"
            print(f"  After spam + {tick} ticks: {status}")
            sim.step(np.array([NOOP], dtype=np.int32))

        print()
        print("=" * 60)
        print("Manual verification complete. Run with pytest for full test suite.")
    else:
        print("mc189_core not available, skipping manual test")
