#!/usr/bin/env python3
"""Full combat test - dragon kill verification."""

import time
from pathlib import Path

import numpy as np

# Setup paths relative to this script
_SIM_ROOT = Path(__file__).parent.parent.parent
import sys

if str(_SIM_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_SIM_ROOT / "python"))
if str(_SIM_ROOT / "cpp/build") not in sys.path:
    sys.path.insert(0, str(_SIM_ROOT / "cpp/build"))

import mc189_core

config = mc189_core.SimulatorConfig()
config.num_envs = 256
config.shader_dir = str(_SIM_ROOT / "cpp/shaders")

sim = mc189_core.MC189Simulator(config)
sim.reset()

print("Dragon Kill Test - 256 environments")
print("=" * 60)

total_damage = 0
dragon_kills = 0
player_deaths = 0
hits = 0

start = time.perf_counter()

for step in range(10000):
    obs = sim.get_observations()
    actions = np.zeros(256, dtype=np.int32)

    for e in range(256):
        can_hit = obs[e, 28] > 0.5
        dragon_phase = int(obs[e, 24] * 6)

        # Stay at center (0, 0) and attack when dragon perches
        px = obs[e, 0] * 100
        pz = obs[e, 2] * 100
        dist_to_center = np.sqrt(px * px + pz * pz)

        if can_hit:
            actions[e] = 9  # Attack!
        elif dist_to_center > 5:
            # Move to center
            if abs(pz) > abs(px):
                actions[e] = 2 if pz > 0 else 1
            else:
                actions[e] = 3 if px > 0 else 4
        else:
            actions[e] = 0  # Wait at center

    prev_dragon_health = obs[:, 16].copy() * 200
    sim.step(actions)

    new_obs = sim.get_observations()
    new_dragon_health = new_obs[:, 16] * 200

    # Check damage
    damage = (prev_dragon_health - new_dragon_health).clip(min=0).sum()
    if damage > 0:
        total_damage += damage
        hits += 1

    # Check kills and deaths via done flags
    dones = sim.get_dones()
    rewards = sim.get_rewards()

    # Dragon kill = big reward + done
    for e in range(256):
        if dones[e]:
            if rewards[e] > 50:  # Win reward is 100
                dragon_kills += 1
            else:
                player_deaths += 1

    if step % 2000 == 1999:
        elapsed = time.perf_counter() - start
        avg_health = new_obs[:, 16].mean() * 200
        print(f"\nStep {step + 1}:")
        print(f"  Total damage: {total_damage:.0f}")
        print(f"  Dragon kills: {dragon_kills}")
        print(f"  Player deaths: {player_deaths}")
        print(f"  Avg dragon health: {avg_health:.0f}/200")
        print(f"  SPS: {256 * (step + 1) / elapsed / 1e6:.2f}M")

elapsed = time.perf_counter() - start
print("\n" + "=" * 60)
print("FINAL RESULTS:")
print("=" * 60)
print("Steps: 10000")
print("Environments: 256")
print(f"Time: {elapsed:.1f}s")
print(f"SPS: {256 * 10000 / elapsed / 1e6:.2f}M")
print()
print(f"Total damage dealt: {total_damage:.0f}")
print(f"Dragon kills: {dragon_kills}")
print(f"Player deaths: {player_deaths}")
print(f"Combat hits: {hits}")

if dragon_kills > 0:
    print("\n🏆 DRAGON SLAIN! FREE THE END ACHIEVED!")
elif total_damage > 0:
    print("\n⚔️ Combat working! Dragon took damage.")
else:
    print("\n❌ No combat occurred")
