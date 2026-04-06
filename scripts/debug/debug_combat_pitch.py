#!/usr/bin/env python3
"""Debug combat - with pitch adjustment.

Run from contrib/minecraft_sim/:
    PYTHONPATH=python:cpp/build python debug_combat_pitch.py
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
LOOK_UP = 15  # NEW

print("Debug combat - with pitch adjustment")
print("=" * 60)

# Wait for perch
for step in range(3000):
    sim.step(np.array([0], dtype=np.int32))
    obs = sim.get_observations()[0]
    phase = int(obs[24] * 6)
    if phase == 4:
        print(f"Dragon perched at step {step}")

        # Get raw observation values
        obs = sim.get_observations()[0]
        print("\nRaw observations:")
        print(f"  obs[0] (pos_x): {obs[0]:.6f}")
        print(f"  obs[1] (pos_y): {obs[1]:.6f}")
        print(f"  obs[2] (pos_z): {obs[2]:.6f}")
        print(f"  obs[17] (dragon_x): {obs[17]:.6f}")
        print(f"  obs[18] (dragon_y): {obs[18]:.6f}")
        print(f"  obs[19] (dragon_z): {obs[19]:.6f}")
        print(f"  obs[25] (dragon_dist): {obs[25]:.6f}")
        print(f"  obs[28] (can_hit): {obs[28]:.6f}")

        # Get positions (correct decoding)
        dragon_x = obs[17] * 100
        dragon_y = obs[18] * 50 + 64
        dragon_z = obs[19] * 100
        player_x = obs[0] * 100  # Normalized by 100
        player_y = obs[1] * 50 + 64  # Same encoding as dragon
        player_z = obs[2] * 100  # Normalized by 100
        print("\nDecoded positions:")
        print(f"Player position: ({player_x:.1f}, {player_y:.1f}, {player_z:.1f})")
        print(f"Dragon position: ({dragon_x:.1f}, {dragon_y:.1f}, {dragon_z:.1f})")

        # Calculate 3D direction to dragon
        dx = dragon_x - player_x
        dy = dragon_y - player_y
        dz = dragon_z - player_z
        xz_dist = math.sqrt(dx * dx + dz * dz)

        # Target pitch (negative = look up)
        target_pitch = -math.degrees(math.atan2(dy, max(xz_dist, 0.1)))
        print(f"XZ distance: {xz_dist:.1f}, Y diff: {dy:.1f}")
        print(f"Target pitch: {target_pitch:.1f}° (negative = looking up)")

        # First adjust pitch to look up at dragon
        print("\nAdjusting pitch...")
        for i in range(30):
            obs = sim.get_observations()[0]
            pitch = (obs[7] - 0.5) * 180  # Convert from [0,1] to [-90,90]
            if pitch < target_pitch + 5:  # Close enough
                print(f"  Pitch adjusted to {pitch:.1f}° (target: {target_pitch:.1f}°)")
                break
            sim.step(np.array([LOOK_UP], dtype=np.int32))
            if i % 5 == 0:
                print(f"  Step {i}: pitch={pitch:.1f}°")

        # Now turn yaw to face dragon
        print("\nAdjusting yaw...")
        for turn in range(50):
            obs = sim.get_observations()[0]
            yaw = obs[6] * 360
            pitch = (obs[7] - 0.5) * 180

            # Calculate 3D look direction
            look_x = -math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
            look_y = -math.sin(math.radians(pitch))
            look_z = math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))

            # 3D direction to dragon (from current position)
            dx = obs[17] * 100 - obs[0] * 100
            dy = (obs[18] * 50 + 64) - (obs[1] * 50 + 64)
            dz = obs[19] * 100 - obs[2] * 100
            dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist_3d > 0.1:
                dir_x = dx / dist_3d
                dir_y = dy / dist_3d
                dir_z = dz / dist_3d
            else:
                dir_x = dir_y = dir_z = 0

            # 3D dot product
            dot_3d = look_x * dir_x + look_y * dir_y + look_z * dir_z

            if turn % 5 == 0:
                print(f"  Turn {turn}: yaw={yaw:.1f}, pitch={pitch:.1f}, dot3D={dot_3d:.2f}")

            if dot_3d > 0.5:  # Facing dragon in 3D
                print(f"Facing dragon at turn {turn}, dot3D={dot_3d:.2f}")
                break
            sim.step(np.array([LOOK_LEFT], dtype=np.int32))

        # Now print everything before attack
        obs = sim.get_observations()[0]
        dragon_dist_actual = obs[25] * 150
        phase_now = int(obs[24] * 6)
        print("\n=== PRE-ATTACK STATE ===")
        print(f"Player yaw: {obs[6] * 360:.1f}")
        print(f"Player pitch: {(obs[7] - 0.5) * 180:.1f}")
        print(f"Dragon health: {obs[16] * 200:.1f}")
        print(f"Dragon dist: {dragon_dist_actual:.2f}")
        print(f"Dragon phase: {phase_now} (4=perching)")
        print(f"Can hit (obs): {obs[28]:.2f}")

        if phase_now != 4:
            print("⚠️ WARNING: Dragon is no longer perching!")

        # Do attack
        print("\n=== ATTACKING ===")
        sim.step(np.array([ATTACK], dtype=np.int32))

        obs = sim.get_observations()[0]
        reward = sim.get_rewards()[0]

        print("\n=== POST-ATTACK STATE ===")
        print(f"Reward: {reward:.4f}")
        print(f"Dragon health: {obs[16] * 200:.1f}")

        if obs[16] * 200 < 200:
            print(f"\n✅ SUCCESS! Dragon took {200 - obs[16] * 200:.0f} damage!")
        else:
            print("\n❌ FAILED - Dragon still at full HP")

        break

print("\n" + "=" * 60)
