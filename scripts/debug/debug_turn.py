#!/usr/bin/env python3
"""Debug turning."""


import math

import numpy as np

from minecraft_sim import mc189_core

config = mc189_core.SimulatorConfig()
config.num_envs = 1
config.shader_dir = str(Path(__file__).parent.parent.parent / "cpp" / "shaders")
sim = mc189_core.MC189Simulator(config)
sim.reset()

LOOK_LEFT = 12

# Wait for perch
for step in range(3000):
    sim.step(np.array([0], dtype=np.int32))
    obs = sim.get_observations()[0]
    phase = int(obs[24] * 6)
    if phase == 4:
        print(f"Dragon perched at step {step}")

        # Turn left until facing dragon
        for turn in range(50):
            obs = sim.get_observations()[0]
            yaw = obs[6] * 360
            dragon_dir_x = obs[26]
            dragon_dir_z = obs[27]

            look_x = -math.sin(math.radians(yaw))
            look_z = math.cos(math.radians(yaw))

            dot = look_x * dragon_dir_x + look_z * dragon_dir_z
            print(f"Turn {turn}: yaw={yaw:.1f}, dot={dot:.2f}")

            if dot > 0.6:
                print("Facing dragon!")
                break

            sim.step(np.array([LOOK_LEFT], dtype=np.int32))
        break
