#!/usr/bin/env python3
"""Test dragon damage with proper facing."""

import math
from pathlib import Path

import numpy as np

from minecraft_sim import mc189_core

import logging

logger = logging.getLogger(__name__)

config = mc189_core.SimulatorConfig()
config.num_envs = 1
config.shader_dir = str(Path(__file__).parent.parent.parent / "cpp" / "shaders")
sim = mc189_core.MC189Simulator(config)
sim.reset()

ATTACK = 9
LOOK_LEFT = 12
LOOK_RIGHT = 13

print("Testing dragon damage with facing correction")
print("=" * 60)

# Wait for perch
for step in range(3000):
    sim.step(np.array([0], dtype=np.int32))
    obs = sim.get_observations()[0]
    phase = int(obs[24] * 6)
    if phase == 4:
        print(f"Dragon perched at step {step}")

        # Get facing info
        dragon_dir_x = obs[26]
        dragon_dir_z = obs[27]
        yaw = obs[6] * 360
        print(f"  Initial yaw: {yaw:.1f}")
        print(f"  Dragon dir: ({dragon_dir_x:.2f}, {dragon_dir_z:.2f})")

        # Calculate required yaw to face dragon
        # dragon_dir is already normalized (to_dragon / dist)
        # yaw 0 = look +Z, yaw 90 = look -X
        target_yaw = math.degrees(math.atan2(-dragon_dir_x, dragon_dir_z))
        if target_yaw < 0:
            target_yaw += 360
        print(f"  Target yaw: {target_yaw:.1f}")

        # Turn toward dragon
        print("\n  Turning toward dragon...")
        for turn in range(100):  # More iterations
            obs = sim.get_observations()[0]
            yaw = obs[6] * 360
            dragon_dir_x = obs[26]
            dragon_dir_z = obs[27]

            # Calculate look direction
            look_x = -math.sin(math.radians(yaw))
            look_z = math.cos(math.radians(yaw))

            # Dot product to check if facing
            dot = look_x * dragon_dir_x + look_z * dragon_dir_z

            if dot > 0.6:  # Facing dragon
                print(f"  Facing dragon at turn {turn}, yaw={yaw:.1f}, dot={dot:.2f}")
                break

            # Just turn left until we face the dragon
            sim.step(np.array([LOOK_LEFT], dtype=np.int32))

        # Check final facing
        obs = sim.get_observations()[0]
        yaw = obs[6] * 360
        dragon_dir_x = obs[26]
        dragon_dir_z = obs[27]
        print(f"  Final yaw: {yaw:.1f}")

        # Calculate look direction
        look_x = -math.sin(math.radians(yaw))
        look_z = math.cos(math.radians(yaw))
        dot = look_x * dragon_dir_x + look_z * dragon_dir_z
        print(f"  Look dir: ({look_x:.2f}, {look_z:.2f})")
        print(f"  Dot product: {dot:.2f} (needs > 0.5)")

        if dot <= 0.5:
            print("\n  ⚠️ Warning: Not facing dragon!")

        # Now attack!
        print("\n  Attacking...")
        initial_hp = obs[16] * 200
        print(f"  Initial dragon HP: {initial_hp:.0f}")

        for i in range(5):
            sim.step(np.array([ATTACK], dtype=np.int32))
            obs = sim.get_observations()[0]
            hp = obs[16] * 200
            reward = sim.get_rewards()[0]
            print(f"  Attack {i + 1}: reward={reward:.2f}, dragon_hp={hp:.0f}")

            # Wait for cooldown (10 ticks)
            for _ in range(10):
                sim.step(np.array([0], dtype=np.int32))

        final_hp = obs[16] * 200
        print(f"\n  Final dragon HP: {final_hp:.0f}")

        if final_hp < initial_hp:
            print(f"\n✅ SUCCESS! Dragon took {initial_hp - final_hp:.0f} damage!")
        else:
            print("\n❌ FAILED - Dragon still at full HP")
        break

print("\n" + "=" * 60)
