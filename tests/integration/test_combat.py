#!/usr/bin/env python3
"""Test combat when dragon perches.

Run from contrib/minecraft_sim/:
    PYTHONPATH=python:cpp/build python test_combat.py
"""

import time

import mc189_core
import numpy as np

config = mc189_core.SimulatorConfig()
config.num_envs = 1024
config.shader_dir = "cpp/shaders"

sim = mc189_core.MC189Simulator(config)
sim.reset()

# Initialize
actions = np.zeros(1024, dtype=np.int32)
sim.step(actions)

print("Testing Dragon Combat")
print("=" * 50)

total_damage = 0
total_deaths = 0
dragon_hits = 0

start = time.perf_counter()

for step in range(5000):
    obs = sim.get_observations()

    # For each env, pick action based on state
    for e in range(1024):
        dragon_phase = obs[e, 24] * 6  # Decode phase (0-6)
        dragon_dist = obs[e, 25] * 150  # Actual distance
        can_hit = obs[e, 28] > 0.5

        if can_hit:
            # Dragon is perching and we're close - ATTACK!
            actions[e] = 9  # attack
        elif dragon_phase >= 3.5:  # Perching (phase 4)
            # Dragon is perching - move toward center (0, 0)
            if dragon_dist > 10:
                # Sprint to center
                # Since we spawn at positive z, moving "back" (action 2) goes toward center
                # Actually need to check which direction to go
                px = obs[e, 0] * 100
                pz = obs[e, 2] * 100
                if pz > 5:
                    actions[e] = 2  # back (negative z)
                elif pz < -5:
                    actions[e] = 1  # forward (positive z)
                elif px > 5:
                    actions[e] = 3  # left (negative x)
                else:
                    actions[e] = 4  # right (positive x)
            else:
                actions[e] = 9  # attack when close
        else:
            # Dragon not perching - stay near center and wait
            px = obs[e, 0] * 100
            pz = obs[e, 2] * 100
            dist_to_center = np.sqrt(px * px + pz * pz)

            if dist_to_center > 20:
                # Move toward center
                if abs(pz) > abs(px):
                    actions[e] = 2 if pz > 0 else 1
                else:
                    actions[e] = 3 if px > 0 else 4
            else:
                actions[e] = 0  # wait

    sim.step(actions)
    rewards = sim.get_rewards()
    dones = sim.get_dones()

    # Track hits
    hits_this_step = ((rewards > 1) & (rewards < 100)).sum()  # Damage rewards 2-14
    dragon_hits += hits_this_step

    deaths_this_step = dones.sum()
    total_deaths += deaths_this_step

    if step % 1000 == 999:
        elapsed = time.perf_counter() - start
        obs = sim.get_observations()
        avg_dragon_health = obs[:, 16].mean()
        damage_dealt = (1.0 - avg_dragon_health) * 200

        print(f"\nStep {step + 1}:")
        print(f"  Dragon hits: {dragon_hits}")
        print(f"  Avg dragon health: {avg_dragon_health:.3f} ({int(avg_dragon_health * 200)} HP)")
        print(f"  Damage dealt: {damage_dealt:.1f}")
        print(f"  Deaths: {total_deaths}")
        print(f"  SPS: {1024 * (step + 1) / elapsed / 1e6:.2f}M")

elapsed = time.perf_counter() - start
obs = sim.get_observations()

print("\n" + "=" * 50)
print("RESULTS:")
print("=" * 50)
print(f"Time: {elapsed:.1f}s")
print(f"SPS: {1024 * 5000 / elapsed / 1e6:.2f}M")
print(f"Dragon hits: {dragon_hits}")
print(f"Total deaths: {total_deaths}")
print(f"Final avg dragon health: {obs[:, 16].mean():.3f}")

if dragon_hits > 0:
    print("\n⚔️ COMBAT WORKING!")
else:
    print("\n😴 No combat detected")
