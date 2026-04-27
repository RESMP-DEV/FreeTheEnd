#!/usr/bin/env python3
"""Test combat by walking toward perching dragon.

Run from contrib/minecraft_sim/:
    PYTHONPATH=python:cpp/build python walk_to_dragon.py
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


print("Walking toward dragon and attacking")
print("=" * 60)

actions = np.array([0], dtype=np.int32)
phases = ["CIRCLING", "STRAFING", "CHARGING", "LANDING", "PERCHING", "TAKING_OFF", "DEAD"]

# Wait for dragon to start perching
print("Waiting for dragon to perch...")
for step in range(1000):
    sim.step(actions)
    obs = sim.get_observations()
    px, py, pz, dx, dy, dz, phase, dist, can_hit, dhealth, phealth = decode_obs(obs[0])

    if phase == 4:  # Perching
        print(f"\nDragon perching at step {step}!")
        print(f"  Player: ({px:.1f}, {py:.1f}, {pz:.1f})")
        print(f"  Dragon: ({dx:.1f}, {dy:.1f}, {dz:.1f})")
        print(f"  Distance: {dist:.1f}")
        break

# Now walk toward dragon while it perches
print("\nWalking toward dragon...")
for walk_step in range(50):
    # Determine direction to dragon
    obs = sim.get_observations()
    px, py, pz, dx, dy, dz, phase, dist, can_hit, dhealth, phealth = decode_obs(obs[0])

    # Calculate direction
    dir_x = dx - px
    dir_z = dz - pz
    mag = np.sqrt(dir_x**2 + dir_z**2)

    if mag > 0.1:
        dir_x /= mag
        dir_z /= mag

    # Choose action based on direction (simplified)
    # Action: 1=forward(+z), 2=back(-z), 3=left(-x), 4=right(+x)
    if abs(dir_z) > abs(dir_x):
        actions[0] = 1 if dir_z > 0 else 2  # Forward or back
    else:
        actions[0] = 4 if dir_x > 0 else 3  # Right or left

    # If close, attack!
    if can_hit:
        actions[0] = 9  # Attack

    sim.step(actions)
    obs = sim.get_observations()
    reward = sim.get_rewards()[0]

    px, py, pz, dx, dy, dz, phase, dist, can_hit, dhealth, phealth = decode_obs(obs[0])

    print(
        f"Step {walk_step}: pos=({px:.1f}, {pz:.1f}), dragon=({dx:.1f}, {dz:.1f}), "
        f"dist={dist:.1f}, phase={phases[phase]}, can_hit={can_hit}, "
        f"dragon_hp={dhealth:.0f}, reward={reward:.1f}"
    )

    if phase != 4:
        print("Dragon stopped perching!")
        break

    if dhealth < 200:
        print(f">>> DRAGON TOOK DAMAGE! Health: {dhealth:.0f}")

print("\n" + "=" * 60)
print("Final state:")
obs = sim.get_observations()
px, py, pz, dx, dy, dz, phase, dist, can_hit, dhealth, phealth = decode_obs(obs[0])
print(f"Player: ({px:.1f}, {py:.1f}, {pz:.1f}), health={phealth:.1f}")
print(f"Dragon: ({dx:.1f}, {dy:.1f}, {dz:.1f}), health={dhealth:.0f}, phase={phases[phase]}")
