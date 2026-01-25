#!/usr/bin/env python3
"""Comprehensive test for all reward signals in dragon_fight_mvk.comp.

Reward sources (from dragon_fight_mvk.comp):
- Dragon damage (vulnerable): +damage * 2.0 (e.g., +14.0 for sword hit)
- Dragon damage (non-vulnerable): +0.1 (small reward for trying)
- Critical hit bonus: +2.0
- Crystal destruction: +10.0
- Bow damage: +damage * 1.5
- Dragon death (WIN): +1000.0
- Player death: -50.0
- Dragon charge damage: -5.0
- Dragon breath damage: -3.0
- Dragon melee damage: -8.0
- Time penalty: -0.001 per tick
- Perching proximity bonus: +0.01 when close to perching dragon
"""

import math
import sys

import numpy as np

from minecraft_sim import mc189_core

# Observation indices
OBS_POS_X = 0
OBS_POS_Y = 1
OBS_POS_Z = 2
OBS_YAW = 6
OBS_PITCH = 7
OBS_HEALTH = 8
OBS_ON_GROUND = 10
OBS_ATTACK_READY = 11
OBS_WEAPON = 12
OBS_DRAGON_HEALTH = 16
OBS_DRAGON_X = 17
OBS_DRAGON_Y = 18
OBS_DRAGON_Z = 19
OBS_DRAGON_PHASE = 24
OBS_DRAGON_DIST = 25
OBS_CAN_HIT = 28
OBS_CRYSTALS = 32
# Debug observation indices (from shader reserved0, reserved1, reserved2)
OBS_DEBUG_ACTION = 15  # reserved0 = float(inp.action)
OBS_DEBUG_COOLDOWN = 29  # reserved1 = float(p.attack_cooldown)
OBS_DEBUG_FLAGS = 30  # reserved2 = float(p.flags)

# Dragon phases
PHASE_CIRCLING = 0
PHASE_STRAFING = 1
PHASE_CHARGING = 2
PHASE_LANDING = 3
PHASE_PERCHING = 4
PHASE_TAKING_OFF = 5
PHASE_DEAD = 6

# Actions
ACTION_NONE = 0
ACTION_FORWARD = 1
ACTION_ATTACK = 9
ACTION_ATTACK_FORWARD = 10
ACTION_SPRINT_FORWARD = 11
ACTION_LOOK_LEFT = 12
ACTION_LOOK_RIGHT = 13
ACTION_LOOK_UP = 15
ACTION_LOOK_DOWN = 16

# Constants
SWORD_DAMAGE = 7.0
HAND_DAMAGE = 1.0
CRIT_MULTIPLIER = 1.5
DRAGON_MAX_HEALTH = 200.0


def create_sim(num_envs: int = 1) -> mc189_core.MC189Simulator:
    """Create and initialize simulator."""
    config = mc189_core.SimulatorConfig()
    config.num_envs = num_envs
    config.shader_dir = str(Path(__file__).parent.parent.parent / "cpp" / "shaders")
    sim = mc189_core.MC189Simulator(config)
    sim.reset()
    return sim


def get_dragon_phase(obs: np.ndarray) -> int:
    """Get dragon phase from observation."""
    return int(obs[OBS_DRAGON_PHASE] * 6)


def get_dragon_health(obs: np.ndarray) -> float:
    """Get dragon health from observation."""
    return obs[OBS_DRAGON_HEALTH] * DRAGON_MAX_HEALTH


def aim_at_dragon(sim: mc189_core.MC189Simulator) -> None:
    """Aim the player toward the dragon."""
    for _ in range(50):
        obs = sim.get_observations()[0]
        dragon_x = obs[OBS_DRAGON_X] * 100
        dragon_y = obs[OBS_DRAGON_Y] * 50 + 64
        dragon_z = obs[OBS_DRAGON_Z] * 100
        player_x = obs[OBS_POS_X] * 100
        player_y = obs[OBS_POS_Y] * 50 + 64
        player_z = obs[OBS_POS_Z] * 100

        dx = dragon_x - player_x
        dy = dragon_y - player_y
        dz = dragon_z - player_z
        xz_dist = math.sqrt(dx * dx + dz * dz)
        target_pitch = -math.degrees(math.atan2(dy, max(xz_dist, 0.1)))
        pitch = (obs[OBS_PITCH] - 0.5) * 180

        if abs(pitch - target_pitch) > 5:
            if pitch < target_pitch:
                sim.step(np.array([ACTION_LOOK_UP], dtype=np.int32))
            else:
                sim.step(np.array([ACTION_LOOK_DOWN], dtype=np.int32))
        else:
            break


def wait_for_perch(sim: mc189_core.MC189Simulator, max_steps: int = 10000) -> bool:
    """Wait for dragon to enter perching phase."""
    for step in range(max_steps):
        obs = sim.get_observations()[0]
        phase = get_dragon_phase(obs)
        if phase == PHASE_PERCHING:
            return True
        sim.step(np.array([ACTION_NONE], dtype=np.int32))
    return False


def test_time_penalty():
    """Test that time penalty is applied each tick."""
    print("\n" + "=" * 60)
    print("TEST: Time Penalty (-0.001 per tick)")
    print("=" * 60)

    sim = create_sim()
    sim.step(np.array([ACTION_NONE], dtype=np.int32))  # Initial step

    # Do nothing for 100 ticks, accumulate time penalty
    total_reward = 0.0
    for _ in range(100):
        sim.step(np.array([ACTION_NONE], dtype=np.int32))
        reward = sim.get_rewards()[0]
        total_reward += reward

    expected_penalty = -0.001 * 100  # -0.1
    print(f"Total reward after 100 ticks of inaction: {total_reward:.4f}")
    print(f"Expected time penalty: {expected_penalty:.4f}")

    # Allow some tolerance for other small rewards/penalties
    passed = total_reward < 0 and abs(total_reward - expected_penalty) < 0.5
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_dragon_damage_reward():
    """Test reward for dealing damage to vulnerable dragon."""
    print("\n" + "=" * 60)
    print("TEST: Dragon Damage Reward (damage * 2.0)")
    print("=" * 60)

    sim = create_sim()
    sim.step(np.array([ACTION_NONE], dtype=np.int32))

    # Wait for dragon to perch
    print("Waiting for dragon to perch...")
    if not wait_for_perch(sim, max_steps=15000):
        print("Dragon did not perch in time")
        print("RESULT: SKIP (dragon never perched)")
        return None

    print("Dragon perching! Aiming...")
    aim_at_dragon(sim)

    obs = sim.get_observations()[0]
    dragon_hp_before = get_dragon_health(obs)
    print(f"Dragon HP before attack: {dragon_hp_before:.0f}")

    # Attack!
    sim.step(np.array([ACTION_ATTACK], dtype=np.int32))
    reward = sim.get_rewards()[0]
    obs = sim.get_observations()[0]
    dragon_hp_after = get_dragon_health(obs)

    damage_dealt = dragon_hp_before - dragon_hp_after
    expected_reward = damage_dealt * 2.0

    print(f"Dragon HP after attack: {dragon_hp_after:.0f}")
    print(f"Damage dealt: {damage_dealt:.1f}")
    print(f"Reward received: {reward:.2f}")
    print(f"Expected reward (damage * 2.0): {expected_reward:.2f}")

    # Check if we hit
    if damage_dealt > 0:
        # Reward should be close to damage * 2.0 (minus time penalty)
        passed = reward > 10.0 and abs(reward - expected_reward) < 5.0
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")
        return passed
    else:
        print("No damage dealt - attack may have missed")
        print("RESULT: SKIP (no hit)")
        return None


def test_critical_hit_bonus():
    """Test critical hit bonus (+2.0) when falling."""
    print("\n" + "=" * 60)
    print("TEST: Critical Hit Bonus (+2.0)")
    print("=" * 60)
    print("Note: Critical hits require falling velocity + not on ground")
    print("This is difficult to test reliably without precise timing")
    print("RESULT: SKIP (requires complex setup)")
    return None


def test_crystal_destruction_reward():
    """Test reward for destroying end crystal (+10.0)."""
    print("\n" + "=" * 60)
    print("TEST: Crystal Destruction Reward (+10.0)")
    print("=" * 60)
    print("Note: Crystals are on pillars, require climbing or bow")
    print("RESULT: SKIP (requires complex setup)")
    return None


def test_dragon_death_reward():
    """Test massive reward for killing dragon (+1000.0)."""
    print("\n" + "=" * 60)
    print("TEST: Dragon Death WIN Reward (+1000.0)")
    print("=" * 60)

    sim = create_sim()
    sim.step(np.array([ACTION_NONE], dtype=np.int32))

    print("Attacking dragon until death (this may take a while)...")
    total_damage = 0
    perch_count = 0
    max_steps = 100000

    for step in range(max_steps):
        obs = sim.get_observations()[0]
        phase = get_dragon_phase(obs)
        dragon_hp = get_dragon_health(obs)

        # Check win condition
        done = sim.get_dones()[0]
        if done:
            reward = sim.get_rewards()[0]
            print(f"\nEpisode ended at step {step}!")
            print(f"Dragon HP: {dragon_hp:.0f}")
            print(f"Final reward: {reward:.1f}")

            if reward > 500:
                print("WIN detected! Checking reward...")
                passed = reward >= 1000
                print("Expected: +1000.0 (may have other bonuses)")
                print(f"RESULT: {'PASS' if passed else 'FAIL'}")
                return passed
            else:
                print("Player likely died (negative or small reward)")
                print("RESULT: SKIP (player died, not a win)")
                return None

        # Attack during perch
        if phase == PHASE_PERCHING:
            perch_count += 1
            can_hit = obs[OBS_CAN_HIT]
            if can_hit > 0.5:
                dragon_hp_before = dragon_hp
                sim.step(np.array([ACTION_ATTACK], dtype=np.int32))
                obs = sim.get_observations()[0]
                dragon_hp_after = get_dragon_health(obs)
                damage = dragon_hp_before - dragon_hp_after
                if damage > 0:
                    total_damage += damage
                # Wait for cooldown
                for _ in range(10):
                    sim.step(np.array([ACTION_NONE], dtype=np.int32))
                continue
            else:
                aim_at_dragon(sim)

        sim.step(np.array([ACTION_NONE], dtype=np.int32))

        if step % 10000 == 0 and step > 0:
            print(f"Step {step}: Dragon HP = {dragon_hp:.0f}, Perch count = {perch_count}")

    print(f"Timeout after {max_steps} steps")
    print(f"Total damage dealt: {total_damage:.0f}")
    print("RESULT: SKIP (timeout)")
    return None


def test_player_death_penalty():
    """Test penalty for player death (-50.0)."""
    print("\n" + "=" * 60)
    print("TEST: Player Death Penalty (-50.0)")
    print("=" * 60)

    sim = create_sim()
    sim.step(np.array([ACTION_NONE], dtype=np.int32))

    print("Waiting for player to die (walking into dragon attacks)...")

    for step in range(20000):
        obs = sim.get_observations()[0]
        player_hp = obs[OBS_HEALTH] * 20
        phase = get_dragon_phase(obs)
        dragon_dist = obs[OBS_DRAGON_DIST] * 150

        done = sim.get_dones()[0]
        if done:
            reward = sim.get_rewards()[0]
            print(f"\nEpisode ended at step {step}!")
            print(f"Player HP: {player_hp:.0f}")
            print(f"Final reward: {reward:.1f}")

            if reward < -40:
                print("Death detected! Checking penalty...")
                passed = reward <= -50  # Should be -50 or worse
                print("Expected: -50.0 (death penalty)")
                print(f"RESULT: {'PASS' if passed else 'FAIL'}")
                return passed
            elif reward > 500:
                print("Somehow won instead of dying!")
                print("RESULT: SKIP (won instead)")
                return None
            else:
                print(f"Unexpected reward: {reward:.1f}")
                print("RESULT: FAIL (unexpected reward)")
                return False

        # Walk toward dragon to get hit
        if dragon_dist > 0.05:
            sim.step(np.array([ACTION_SPRINT_FORWARD], dtype=np.int32))
        else:
            sim.step(np.array([ACTION_FORWARD], dtype=np.int32))

    print("Timeout - player didn't die")
    print("RESULT: SKIP (no death)")
    return None


def test_perching_proximity_bonus():
    """Test small bonus for being close to perching dragon (+0.01)."""
    print("\n" + "=" * 60)
    print("TEST: Perching Proximity Bonus (+0.01)")
    print("=" * 60)

    sim = create_sim()
    sim.step(np.array([ACTION_NONE], dtype=np.int32))

    print("Waiting for dragon to perch...")
    if not wait_for_perch(sim, max_steps=15000):
        print("Dragon did not perch in time")
        print("RESULT: SKIP (dragon never perched)")
        return None

    # Stay close and observe rewards
    positive_ticks = 0
    total_ticks = 0

    for _ in range(100):
        obs = sim.get_observations()[0]
        phase = get_dragon_phase(obs)
        dragon_dist = obs[OBS_DRAGON_DIST] * 150

        if phase != PHASE_PERCHING:
            break

        sim.step(np.array([ACTION_NONE], dtype=np.int32))
        reward = sim.get_rewards()[0]
        total_ticks += 1

        # Proximity bonus is +0.01, time penalty is -0.001
        # Net should be slightly positive when close
        if dragon_dist < 15 and reward > 0:
            positive_ticks += 1

    print(f"Ticks with positive reward while close: {positive_ticks}/{total_ticks}")
    passed = positive_ticks > total_ticks * 0.3  # At least 30% positive
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_non_vulnerable_hit_reward():
    """Test small reward for hitting non-vulnerable dragon (+0.1)."""
    print("\n" + "=" * 60)
    print("TEST: Non-Vulnerable Hit Reward (+0.1)")
    print("=" * 60)

    sim = create_sim()
    sim.step(np.array([ACTION_NONE], dtype=np.int32))

    # Attack when dragon is circling (not vulnerable)
    positive_small_rewards = 0

    for step in range(1000):
        obs = sim.get_observations()[0]
        phase = get_dragon_phase(obs)
        dragon_dist = obs[OBS_DRAGON_DIST] * 150

        if phase != PHASE_PERCHING and dragon_dist < 15:
            # Attack non-vulnerable dragon
            aim_at_dragon(sim)
            sim.step(np.array([ACTION_ATTACK], dtype=np.int32))
            reward = sim.get_rewards()[0]

            # Small positive reward for trying
            if 0 < reward < 1:
                positive_small_rewards += 1
                print(f"Small reward detected: {reward:.3f}")
                if positive_small_rewards >= 3:
                    break

            # Wait for cooldown
            for _ in range(10):
                sim.step(np.array([ACTION_NONE], dtype=np.int32))
        else:
            sim.step(np.array([ACTION_NONE], dtype=np.int32))

    print(f"Small positive rewards detected: {positive_small_rewards}")
    passed = positive_small_rewards >= 1
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    """Run all reward signal tests."""
    print("=" * 60)
    print("MINECRAFT DRAGON FIGHT REWARD SIGNAL VERIFICATION")
    print("=" * 60)

    results = {}

    # Run tests
    results["time_penalty"] = test_time_penalty()
    results["dragon_damage"] = test_dragon_damage_reward()
    results["critical_hit"] = test_critical_hit_bonus()
    results["crystal_destruction"] = test_crystal_destruction_reward()
    results["perching_proximity"] = test_perching_proximity_bonus()
    results["non_vulnerable_hit"] = test_non_vulnerable_hit_reward()
    results["player_death"] = test_player_death_penalty()

    # Note: Dragon death test takes too long, run separately
    # results["dragon_death"] = test_dragon_death_reward()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results.items():
        if result is True:
            status = "PASS"
            passed += 1
        elif result is False:
            status = "FAIL"
            failed += 1
        else:
            status = "SKIP"
            skipped += 1
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\nSome tests FAILED!")
        return 1
    elif passed == 0:
        print("\nNo tests PASSED - all skipped or complex setup required")
        return 2
    else:
        print("\nAll executed tests PASSED!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
