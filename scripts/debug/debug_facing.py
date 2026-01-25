#!/usr/bin/env python3
"""Debug facing direction."""


import math

import numpy as np

from minecraft_sim import mc189_core

config = mc189_core.SimulatorConfig()
config.num_envs = 1
config.shader_dir = str(Path(__file__).parent.parent.parent / "cpp" / "shaders")
sim = mc189_core.MC189Simulator(config)
sim.reset()

# Wait for perch
for step in range(3000):
    sim.step(np.array([0], dtype=np.int32))
    obs = sim.get_observations()[0]
    phase = int(obs[24] * 6)
    if phase == 4:
        print(f"Dragon perched at step {step}")
        print(f"Player yaw: {obs[6] * 360:.1f}, pitch: {obs[7] * 180 - 90:.1f}")

        # Calculate direction to dragon
        dx = obs[26]  # dragon_dir_x (already normalized)
        dz = obs[27]  # dragon_dir_z (already normalized)
        print(f"Dragon dir (from obs): ({dx:.2f}, {dz:.2f})")

        # Calculate look direction from player yaw/pitch
        yaw = obs[6] * 360
        pitch = obs[7] * 180 - 90
        look_x = -math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
        look_z = math.cos(math.radians(yaw)) * math.cos(math.radians(pitch))
        print(f"Look dir: ({look_x:.2f}, {look_z:.2f})")

        # Dot product (2D since dragon_dir doesn't have y)
        dot = look_x * dx + look_z * dz
        print(f"Dot product (2D): {dot:.2f} (needs > 0.5 for can_hit)")
        print(f"can_hit_dragon (obs): {obs[28]:.1f}")
        break
