#!/usr/bin/env python3
"""Quick test of dragon fight simulator.

Run from FreeTheEnd/:
    PYTHONPATH=python:cpp/build python test_dragon.py
"""

import time

import mc189_core
import numpy as np

print("Testing Dragon Fight Simulator - Combat Test")
print("=" * 50)

config = mc189_core.SimulatorConfig()
config.num_envs = 4096
config.shader_dir = "cpp/shaders"

print("Creating simulator...")
sim = mc189_core.MC189Simulator(config)
print("Resetting...")
sim.reset()

# Step once to populate observations
actions = np.zeros(4096, dtype=np.int32)
sim.step(actions)

obs = sim.get_observations()
print("\nInitial state for env 0:")
print(f"  Player: pos=({obs[0, 0] * 100:.1f}, {obs[0, 1] * 50 + 64:.1f}, {obs[0, 2] * 100:.1f})")
print(
    f"  Dragon: pos=({obs[0, 17] * 100:.1f}, {obs[0, 18] * 50 + 64:.1f}, {obs[0, 19] * 100:.1f}), health={obs[0, 16]:.2f}"
)
print(f"  Dragon dist: {obs[0, 25] * 150:.1f}, phase: {int(obs[0, 24] * 6)}")
print(f"  Crystals: {int(obs[0, 32] * 10)}")

# Run with smart actions - move forward toward dragon
print("\n" + "=" * 50)
print("Running 5000 steps - moving toward dragon and attacking")
print("=" * 50)

total_reward = 0
max_reward = -1000
min_reward = 1000
total_dones = 0
damage_rewards = 0
death_count = 0
win_count = 0

start = time.perf_counter()
for i in range(5000):
    obs = sim.get_observations()

    # Smart action selection based on observation
    # dragon_dir is in obs[26] (x) and obs[27] (z)
    # on_ground is obs[10]
    # dragon_dist is obs[25]
    # can_hit is obs[28]

    actions = np.zeros(4096, dtype=np.int32)

    for e in range(4096):
        dragon_dist = obs[e, 25]  # 0-1 normalized
        can_hit = obs[e, 28]
        on_ground = obs[e, 10]
        dragon_dir_z = obs[e, 27]

        if can_hit > 0.5:
            # Can hit dragon - ATTACK!
            actions[e] = 9  # attack
        elif dragon_dist < 0.1:  # Within 15 units
            # Close - attack + forward
            actions[e] = 10  # attack + forward
        elif dragon_dist < 0.5:  # Within 75 units
            # Medium distance - sprint forward
            if on_ground > 0.5:
                actions[e] = 11  # sprint + forward
            else:
                actions[e] = 1  # forward
        else:
            # Far away - sprint forward
            actions[e] = 11  # sprint + forward

    sim.step(actions)
    rewards = sim.get_rewards()
    dones = sim.get_dones()

    # Track stats
    total_reward += rewards.sum()
    max_reward = max(max_reward, rewards.max())
    min_reward = min(min_reward, rewards.min())

    # Count wins and deaths
    wins = (rewards > 500).sum()
    deaths = dones.sum() - wins
    win_count += wins
    death_count += deaths
    total_dones += dones.sum()

    # Count damage rewards (2.0 per hit)
    damage_hits = ((rewards > 1.0) & (rewards < 500)).sum()
    damage_rewards += damage_hits

    if i % 1000 == 999:
        obs = sim.get_observations()
        avg_dragon_health = obs[:, 16].mean()
        avg_player_health = obs[:, 8].mean()
        print(f"\nStep {i + 1}:")
        print(f"  Avg dragon health: {avg_dragon_health:.3f}")
        print(f"  Avg player health: {avg_player_health:.3f}")
        print(f"  Wins: {win_count}, Deaths: {death_count}")
        print(f"  Damage hits: {damage_rewards}")
        print(f"  Reward range: [{min_reward:.2f}, {max_reward:.2f}]")

elapsed = time.perf_counter() - start

print("\n" + "=" * 50)
print("RESULTS after 5000 steps:")
print("=" * 50)
print(f"Time: {elapsed:.3f}s")
print(f"SPS: {4096 * 5000 / elapsed / 1e6:.2f}M")
print(f"Total reward: {total_reward:.2f}")
print(f"Total episode ends: {total_dones}")
print(f"  - Wins: {win_count}")
print(f"  - Deaths: {death_count}")
print(f"Damage rewards: {damage_rewards}")

if win_count > 0:
    print("\n🐲 DRAGON KILLED! 🐲")
elif damage_rewards > 0:
    print(f"\n⚔️  Combat working! {damage_rewards} damage hits detected!")
elif death_count > 0:
    print(f"\n💀 {death_count} deaths detected - combat is happening!")
else:
    print("\n😴 No combat detected - agents not reaching dragon?")
