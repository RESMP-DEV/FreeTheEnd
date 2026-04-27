#!/usr/bin/env python3
"""Debug observation values.

Run from contrib/minecraft_sim/:
    PYTHONPATH=python:cpp/build python debug_obs.py
"""

import mc189_core
import numpy as np

import logging

logger = logging.getLogger(__name__)

# Create smaller simulator for debugging
config = mc189_core.SimulatorConfig()
config.num_envs = 4
config.shader_dir = "cpp/shaders"

sim = mc189_core.MC189Simulator(config)
sim.reset()

# Step once to get observations
actions = np.zeros(4, dtype=np.int32)
sim.step(actions)

obs = sim.get_observations()

# Print all observations for env 0
print("Full observation for env 0 (48 values):")
print()

labels = [
    # Player (0-15)
    "pos_x",
    "pos_y",
    "pos_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "yaw",
    "pitch",
    "health",
    "hunger",
    "on_ground",
    "attack_ready",
    "weapon",
    "arrows",
    "arrow_charge",
    "reserved0",
    # Dragon (16-31)
    "dragon_health",
    "dragon_x",
    "dragon_y",
    "dragon_z",
    "dragon_vel_x",
    "dragon_vel_y",
    "dragon_vel_z",
    "dragon_yaw",
    "dragon_phase",
    "dragon_dist",
    "dragon_dir_x",
    "dragon_dir_z",
    "can_hit",
    "attacking",
    "reserved1",
    "reserved2",
    # Env (32-47)
    "crystals",
    "nearest_crystal_dist",
    "crystal_dir_x",
    "crystal_dir_z",
    "crystal_y",
    "portal_active",
    "portal_dist",
    "time_remaining",
    "total_damage",
    "res3",
    "res4",
    "res5",
    "res6",
    "res7",
    "res8",
    "res9",
]

for i, (label, val) in enumerate(zip(labels, obs[0])):
    print(f"{i:2d}. {label:20s}: {val:10.4f}")

# Decode positions
print("\n--- Decoded positions ---")
print(
    f"Player actual pos: ({obs[0, 0] * 100:.1f}, {obs[0, 1] * 50 + 64:.1f}, {obs[0, 2] * 100:.1f})"
)
print(
    f"Dragon actual pos: ({obs[0, 17] * 100:.1f}, {obs[0, 18] * 50 + 64:.1f}, {obs[0, 19] * 100:.1f})"
)
print(f"Dragon dist (actual): {obs[0, 25] * 150:.1f}")
