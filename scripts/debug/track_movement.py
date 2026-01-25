#!/usr/bin/env python3
"""Track player movement toward dragon.

Run from FreeTheEnd/:
    PYTHONPATH=python:cpp/build python track_movement.py
"""

import mc189_core
import numpy as np

config = mc189_core.SimulatorConfig()
config.num_envs = 16
config.shader_dir = "cpp/shaders"

sim = mc189_core.MC189Simulator(config)
sim.reset()

# Initialize with one step
actions = np.zeros(16, dtype=np.int32)
sim.step(actions)
obs = sim.get_observations()


def decode_pos(obs, idx=0):
    x = obs[idx, 0] * 100
    y = obs[idx, 1] * 50 + 64
    z = obs[idx, 2] * 100
    return x, y, z


def decode_dragon_pos(obs, idx=0):
    x = obs[idx, 17] * 100
    y = obs[idx, 18] * 50 + 64
    z = obs[idx, 19] * 100
    return x, y, z


print("Initial:")
px, py, pz = decode_pos(obs)
dx, dy, dz = decode_dragon_pos(obs)
print(f"  Player pos: ({px:.1f}, {py:.1f}, {pz:.1f})")
print(f"  Dragon pos: ({dx:.1f}, {dy:.1f}, {dz:.1f})")
print(f"  Dragon dist: {obs[0, 25] * 150:.1f} blocks")

# Run 100 steps moving forward
for i in range(100):
    actions = np.ones(16, dtype=np.int32)  # Action 1 = forward
    sim.step(actions)

obs = sim.get_observations()
px, py, pz = decode_pos(obs)
print("\nAfter 100 forward steps:")
print(f"  Player pos: ({px:.1f}, {py:.1f}, {pz:.1f})")
print(f"  Dragon dist: {obs[0, 25] * 150:.1f} blocks")

# Keep going
for i in range(400):
    actions = np.ones(16, dtype=np.int32)
    sim.step(actions)

obs = sim.get_observations()
px, py, pz = decode_pos(obs)
print("\nAfter 500 forward steps:")
print(f"  Player pos: ({px:.1f}, {py:.1f}, {pz:.1f})")
print(f"  Dragon dist: {obs[0, 25] * 150:.1f} blocks")

# Keep going to 1000
for i in range(500):
    actions = np.ones(16, dtype=np.int32)
    sim.step(actions)

obs = sim.get_observations()
px, py, pz = decode_pos(obs)
dx, dy, dz = decode_dragon_pos(obs)
print("\nAfter 1000 forward steps:")
print(f"  Player pos: ({px:.1f}, {py:.1f}, {pz:.1f})")
print(f"  Dragon pos: ({dx:.1f}, {dy:.1f}, {dz:.1f})")
print(f"  Dragon dist: {obs[0, 25] * 150:.1f} blocks")
print(f"  Player health: {obs[0, 8]:.3f}")
print(f"  Dragon phase: {int(obs[0, 24] * 6)}")
print(f"  Can hit dragon: {obs[0, 28]:.1f}")

# Now try sprinting forward
print("\n--- Now sprinting ---")
for i in range(1000):
    actions = np.full(16, 11, dtype=np.int32)  # Action 11 = sprint + forward
    sim.step(actions)

obs = sim.get_observations()
px, py, pz = decode_pos(obs)
print("\nAfter 1000 more sprint steps (2000 total):")
print(f"  Player pos: ({px:.1f}, {py:.1f}, {pz:.1f})")
print(f"  Dragon dist: {obs[0, 25] * 150:.1f} blocks")
print(f"  Player health: {obs[0, 8]:.3f}")
