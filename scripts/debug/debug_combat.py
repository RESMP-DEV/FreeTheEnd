#!/usr/bin/env python3
"""Debug combat - check all conditions.

Run from contrib/minecraft_sim/:
    PYTHONPATH=python:cpp/build python debug_combat.py
"""

import math

import numpy as np

from minecraft_sim import mc189_core

config = mc189_core.SimulatorConfig()
config.num_envs = 1
config.shader_dir = str(Path(__file__).parent.parent.parent / "cpp" / "shaders")
sim = mc189_core.MC189Simulator(config)
sim.reset()

ATTACK = 9
LOOK_LEFT = 12

print("Debug combat - checking all conditions")
print("=" * 60)

# Wait for perch
for step in range(3000):
    sim.step(np.array([0], dtype=np.int32))
    obs = sim.get_observations()[0]
    phase = int(obs[24] * 6)
    if phase == 4:
        print(f"Dragon perched at step {step}")

        # Get dragon position
        obs = sim.get_observations()[0]
        dragon_x = obs[17] * 100
        dragon_y = obs[18] * 50 + 64
        dragon_z = obs[19] * 100
        player_x = obs[0]
        player_y = obs[1]
        player_z = obs[2]
        print(f"Player position: ({player_x:.1f}, {player_y:.1f}, {player_z:.1f})")
        print(f"Dragon position: ({dragon_x:.1f}, {dragon_y:.1f}, {dragon_z:.1f})")

        # Turn until facing
        for turn in range(50):
            obs = sim.get_observations()[0]
            yaw = obs[6] * 360
            dragon_dir_x = obs[26]
            dragon_dir_z = obs[27]
            look_x = -math.sin(math.radians(yaw))
            look_z = math.cos(math.radians(yaw))
            dot = look_x * dragon_dir_x + look_z * dragon_dir_z

            print(f"  Turn {turn}: yaw={yaw:.1f}, dot={dot:.2f}")

            if dot > 0.6:
                print(f"Facing dragon at turn {turn}, dot={dot:.2f}")
                break
            sim.step(np.array([LOOK_LEFT], dtype=np.int32))

        # Now print everything before attack
        obs = sim.get_observations()[0]
        print("\n=== PRE-ATTACK STATE ===")
        print(f"Player position: ({obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f})")
        print(f"Player yaw: {obs[6] * 360:.1f}")
        print(f"Dragon health: {obs[16] * 200:.1f}")
        print(f"Dragon dist: {obs[25] * 100:.2f}")
        print(f"Dragon phase: {int(obs[24] * 6)} (4=perching)")
        print(f"Can hit dragon (obs): {obs[28]:.2f}")
        print(f"Reserved0 (action): {obs[29]:.2f}")
        print(f"Reserved1: {obs[30]:.2f}")
        print(f"Reserved2: {obs[31]:.2f}")

        # Do attack
        print("\n=== ATTACKING ===")
        sim.step(np.array([ATTACK], dtype=np.int32))

        obs = sim.get_observations()[0]
        reward = sim.get_rewards()[0]

        print("\n=== POST-ATTACK STATE ===")
        print(f"Reward: {reward:.4f}")
        print(f"Dragon health: {obs[16] * 200:.1f}")
        print(f"Reserved0 (action): {obs[29]:.2f}")  # Should be 1.0
        print(f"Reserved1: {obs[30]:.2f}")
        print(f"Reserved2: {obs[31]:.2f}")

        # Check if reward matches debug marker
        if abs(reward - 0.123) < 0.01:
            print("\n✓ Attack block was entered (reward includes 0.123 marker)")
        else:
            print(f"\n? Unexpected reward {reward} - attack block might not be entered")

        break

print("\n" + "=" * 60)
