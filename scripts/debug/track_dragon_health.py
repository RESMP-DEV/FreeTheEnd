#!/usr/bin/env python3
"""Test dragon taking damage by tracking health.

Run from FreeTheEnd/:
    PYTHONPATH=python:cpp/build python track_dragon_health.py
"""

import mc189_core
import numpy as np

config = mc189_core.SimulatorConfig()
config.num_envs = 1
config.shader_dir = "cpp/shaders"

sim = mc189_core.MC189Simulator(config)
sim.reset()

# Track dragon health over time
actions = np.zeros(1, dtype=np.int32)

print("Tracking dragon health over time (attack when can_hit)")
print("=" * 60)

prev_health = 200.0
total_damage = 0

for step in range(5000):
    obs = sim.get_observations()

    can_hit = obs[0, 28] > 0.5
    dragon_health = obs[0, 16] * 200
    dragon_phase = int(obs[0, 24] * 6)
    phases = ["CIRC", "STRF", "CHRG", "LAND", "PRCH", "TOFF", "DEAD"]

    # Attack if can hit
    actions[0] = 9 if can_hit else 0

    sim.step(actions)
    rewards = sim.get_rewards()

    new_obs = sim.get_observations()
    new_health = new_obs[0, 16] * 200

    # Check for damage
    if new_health < dragon_health - 0.1:
        damage = dragon_health - new_health
        total_damage += damage
        print(f"Step {step}: HIT! Dealt {damage:.0f} damage (health: {new_health:.0f}/200)")

    # Log every 500 steps or on phase 4
    if step % 500 == 0:
        print(
            f"Step {step}: phase={phases[dragon_phase]}, health={dragon_health:.0f}, can_hit={can_hit}"
        )
    elif dragon_phase == 4 and step % 20 == 0:
        print(
            f"Step {step}: PERCHING! health={dragon_health:.0f}, can_hit={can_hit}, reward={rewards[0]:.3f}"
        )

print(f"\n Total damage dealt: {total_damage:.0f}")
