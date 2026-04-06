#!/usr/bin/env python3
"""Full dragon kill test - demonstrates combat loop until dragon dies."""

import math
import sys
from pathlib import Path

import numpy as np

from minecraft_sim import mc189_core

config = mc189_core.SimulatorConfig()
config.num_envs = 1
config.shader_dir = str(Path(__file__).parent.parent.parent / "cpp" / "shaders")
sim = mc189_core.MC189Simulator(config)
sim.reset()

# Run one tick to populate observations
sim.step(np.array([0], dtype=np.int32))

# Check initial state
obs = sim.get_observations()[0]
print(f"Initial dragon HP: {obs[16] * 200:.0f}")
print(f"Initial phase: {int(obs[24] * 6)}")
crystals = int(obs[32] * 10)
print(f"Crystals alive: {crystals}")
print("\n⚠️ Note: Crystals heal the dragon! Must destroy crystals first.")

ATTACK = 9
LOOK_LEFT = 12
LOOK_RIGHT = 13
LOOK_UP = 15
FORWARD = 1

print("Full Dragon Kill Test")
print("=" * 60)

total_damage = 0
perch_count = 0
max_steps = 50000  # Increase limit
prev_hp = 200

# First, move player to center (where dragon perches)
print("Moving player to center...")
for move_step in range(100):
    sim.step(np.array([FORWARD], dtype=np.int32))
obs = sim.get_observations()[0]
player_x = obs[0] * 100
player_y = obs[1] * 50 + 64
player_z = obs[2] * 100
dragon_dist = obs[25] * 150
print(
    f"Player pos: ({player_x:.1f}, {player_y:.1f}, {player_z:.1f}), Dragon dist: {dragon_dist:.1f}"
)
print()

for step in range(max_steps):
    obs = sim.get_observations()[0]
    phase = int(obs[24] * 6)
    dragon_hp = obs[16] * 200

    # Check for unexpected reset
    if step > 0 and dragon_hp == 200 and prev_hp < 200:
        done = sim.get_dones()[0]
        player_x = obs[0] * 100
        player_y = obs[1] * 50 + 64
        player_z = obs[2] * 100
        player_hp = obs[8] * 20
        print(f"\n⚠️ [Step {step}] Dragon HP reset from {prev_hp:.0f} to 200!")
        print(
            f"  Done={done}, Player: ({player_x:.0f}, {player_y:.0f}, {player_z:.0f}), HP={player_hp:.0f}"
        )
    prev_hp = dragon_hp

    # Check for win (done flag should be set when dragon dies)
    done = sim.get_dones()[0]
    if done:
        reward = sim.get_rewards()[0]
        print(f"\n🏆 VICTORY! Dragon killed at step {step}!")
        print(f"Final reward: {reward:.1f}")
        print(f"Total damage dealt: {total_damage}")
        print(f"Perch attacks: {perch_count}")
        break

    # Check for dragon HP reaching 0 (backup check)
    if dragon_hp <= 0:
        print(f"\n🏆 Dragon HP = 0 at step {step}!")
        print(f"Total damage dealt: {total_damage}")
        print(f"Perch attacks: {perch_count}")
        break

    # Check for dragon perching
    if phase == 4:  # PERCHING
        perch_count += 1
        player_hp = obs[8] * 20
        dragon_dist = obs[25] * 150
        dragon_x = obs[17] * 100
        dragon_y = obs[18] * 50 + 64
        dragon_z = obs[19] * 100
        can_hit = obs[28]
        print(
            f"\n[Step {step}] Dragon perching at ({dragon_x:.0f}, {dragon_y:.0f}, {dragon_z:.0f})"
        )
        print(f"  Dragon HP: {dragon_hp:.0f}, Player HP: {player_hp:.0f}, Dist: {dragon_dist:.1f}")

        # Get positions
        dragon_x = obs[17] * 100
        dragon_y = obs[18] * 50 + 64
        dragon_z = obs[19] * 100
        player_x = obs[0] * 100
        player_y = obs[1] * 50 + 64
        player_z = obs[2] * 100

        dx = dragon_x - player_x
        dy = dragon_y - player_y
        dz = dragon_z - player_z
        xz_dist = math.sqrt(dx * dx + dz * dz)

        # Adjust pitch to look at dragon
        target_pitch = -math.degrees(math.atan2(dy, max(xz_dist, 0.1)))
        for _ in range(20):
            obs = sim.get_observations()[0]
            pitch = (obs[7] - 0.5) * 180
            if pitch < target_pitch + 5:
                break
            sim.step(np.array([LOOK_UP], dtype=np.int32))

        # Attack loop while dragon is perching
        attacks_this_perch = 0
        for attack_step in range(200):  # Max 200 steps per perch
            obs = sim.get_observations()[0]
            phase = int(obs[24] * 6)
            dragon_hp_before = obs[16] * 200

            if phase != 4:  # No longer perching
                print(f"  Perch ended after {attacks_this_perch} attacks")
                break

            # Update pitch for 3D aiming
            dx = obs[17] * 100 - obs[0] * 100
            dy = (obs[18] * 50 + 64) - (obs[1] * 50 + 64)
            dz = obs[19] * 100 - obs[2] * 100
            xz_dist = math.sqrt(dx * dx + dz * dz)
            target_pitch = -math.degrees(math.atan2(dy, max(xz_dist, 0.1)))

            pitch = (obs[7] - 0.5) * 180
            if abs(pitch - target_pitch) > 10:
                sim.step(np.array([LOOK_UP], dtype=np.int32))
                continue

            # Check if within reach and attack
            dragon_dist = obs[25] * 150
            if dragon_dist <= 4.5 and obs[28] > 0.5:  # can_hit_dragon
                sim.step(np.array([ATTACK], dtype=np.int32))

                # Check for win IMMEDIATELY after attack
                done = sim.get_dones()[0]
                reward = sim.get_rewards()[0]
                obs = sim.get_observations()[0]
                dragon_hp_after = obs[16] * 200
                damage = dragon_hp_before - dragon_hp_after

                if damage > 0:
                    total_damage += damage
                    attacks_this_perch += 1
                    print(f"  Hit! Damage: {damage:.0f}, Dragon HP: {dragon_hp_after:.0f}")

                # Check for dragon death
                if done or dragon_hp_after <= 0:
                    print("\n🏆 VICTORY! Dragon killed!")
                    print(f"  Done flag: {done}, Final HP: {dragon_hp_after:.0f}")
                    print(f"  Reward: {reward:.1f}")
                    print(f"  Total damage dealt: {total_damage}")
                    print(f"  Perch attacks: {perch_count}")
                    import sys

                    sys.exit(0)

                # Wait for cooldown
                for _ in range(10):
                    sim.step(np.array([0], dtype=np.int32))
            else:
                sim.step(np.array([0], dtype=np.int32))
    else:
        # Not perching - just wait
        sim.step(np.array([0], dtype=np.int32))

else:
    print(f"\n⏰ Timeout at {max_steps} steps")
    print(f"Dragon HP remaining: {dragon_hp:.0f}")
    print(f"Total damage dealt: {total_damage}")
    print(f"Perch count: {perch_count}")

print("\n" + "=" * 60)
