#!/usr/bin/env python3
"""Test getting close to dragon and attacking during perch.

Run from contrib/minecraft_sim/:
    PYTHONPATH=python:cpp/build python test_perch_combat.py
"""

import mc189_core
import numpy as np

import logging

logger = logging.getLogger(__name__)

config = mc189_core.SimulatorConfig()
config.num_envs = 1
config.shader_dir = "cpp/shaders"

sim = mc189_core.MC189Simulator(config)
sim.reset()


def decode_obs(obs_vec):
    logger.debug("decode_obs: obs_vec=%s", obs_vec)
    px, py, pz = obs_vec[0] * 100, obs_vec[1] * 50 + 64, obs_vec[2] * 100
    dragon_x = obs_vec[17] * 100
    dragon_y = obs_vec[18] * 50 + 64
    dragon_z = obs_vec[19] * 100
    dragon_phase = int(obs_vec[24] * 6)
    dragon_dist = obs_vec[25] * 150
    can_hit = obs_vec[28] > 0.5
    dragon_health = obs_vec[16] * 200
    player_health = obs_vec[8] * 20
    return (
        px,
        py,
        pz,
        dragon_x,
        dragon_y,
        dragon_z,
        dragon_phase,
        dragon_dist,
        can_hit,
        dragon_health,
        player_health,
    )


print("Continuous walk toward dragon + attack when possible")
print("=" * 60)

actions = np.array([0], dtype=np.int32)
phases = ["CIRCLING", "STRAFING", "CHARGING", "LANDING", "PERCHING", "TAKING_OFF", "DEAD"]

perch_hits = 0
total_damage = 0
steps_in_perch = 0

for step in range(2000):
    obs = sim.get_observations()
    px, py, pz, dx, dy, dz, phase, dist, can_hit, dhealth, phealth = decode_obs(obs[0])

    # Calculate direction to dragon
    dir_x = dx - px
    dir_z = dz - pz
    mag = np.sqrt(dir_x**2 + dir_z**2)

    # Always try to move toward dragon or center (0, 0)
    # Dragon perches at (0, 68, 0), so moving to center is smart
    target_x, target_z = 0, 0  # Center

    dir_x = target_x - px
    dir_z = target_z - pz
    mag = np.sqrt(dir_x**2 + dir_z**2)

    if phase == 4:  # Dragon perching
        steps_in_perch += 1
        # Move toward dragon at center
        if can_hit:
            actions[0] = 9  # Attack!
        elif dist > 5:
            # Sprint to center
            if abs(dir_z) > abs(dir_x):
                actions[0] = 11 if dir_z > 0 else 2  # Sprint forward or back
            else:
                actions[0] = 4 if dir_x > 0 else 3  # Right or left
        else:
            actions[0] = 9  # Try attack anyway
    else:
        # Dragon not perching - stay at center and wait
        if mag > 5:
            # Move to center
            if abs(dir_z) > abs(dir_x):
                actions[0] = 1 if dir_z > 0 else 2
            else:
                actions[0] = 4 if dir_x > 0 else 3
        else:
            actions[0] = 0  # Wait at center

    prev_health = dhealth
    sim.step(actions)

    obs = sim.get_observations()
    px, py, pz, dx, dy, dz, phase, dist, can_hit, dhealth, phealth = decode_obs(obs[0])
    reward = sim.get_rewards()[0]

    if prev_health > dhealth:
        damage = prev_health - dhealth
        total_damage += damage
        perch_hits += 1
        print(f">>> HIT! Step {step}: {damage:.0f} damage, total={total_damage:.0f}")

    if step % 200 == 0:
        print(
            f"Step {step}: pos=({px:.1f}, {pz:.1f}), phase={phases[phase]}, "
            f"dist={dist:.1f}, can_hit={can_hit}, dragon={dhealth:.0f}/200"
        )

print("\n" + "=" * 60)
print("RESULTS:")
print(f"Steps in perch: {steps_in_perch}")
print(f"Perch hits: {perch_hits}")
print(f"Total damage: {total_damage:.0f}")
print(f"Final dragon health: {dhealth:.0f}/200")

if perch_hits > 0:
    print("\n⚔️ COMBAT IS WORKING!")
else:
    print("\n❌ No combat occurred")
