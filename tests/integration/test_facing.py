#!/usr/bin/env python3
"""Test dragon combat with proper facing.

Run from contrib/minecraft_sim/:
    PYTHONPATH=python:cpp/build python test_facing.py
"""

import mc189_core
import numpy as np

config = mc189_core.SimulatorConfig()
config.num_envs = 1
config.shader_dir = "cpp/shaders"

sim = mc189_core.MC189Simulator(config)
sim.reset()

actions = np.zeros(1, dtype=np.int32)

print("Testing combat with facing direction")
print("=" * 60)

# Action meanings:
# 9 = attack
# 12 = look_left (-5 degrees)
# 13 = look_right (+5 degrees)

for step in range(3000):
    obs = sim.get_observations()

    # Player state
    px, py, pz = obs[0, 0] * 100, obs[0, 1] * 50 + 64, obs[0, 2] * 100
    yaw = obs[0, 6] * 360  # Player yaw

    # Dragon state
    dragon_phase = int(obs[0, 24] * 6)
    dragon_x = obs[0, 17] * 100
    dragon_y = obs[0, 18] * 50 + 64
    dragon_z = obs[0, 19] * 100
    dragon_health = obs[0, 16] * 200
    can_hit_obs = obs[0, 28] > 0.5  # From observation
    attack_ready = obs[0, 11] > 0.5

    # Calculate direction to dragon
    dx = dragon_x - px
    dz = dragon_z - pz
    dragon_dist = np.sqrt(dx * dx + dz * dz)

    # What yaw should player have to face dragon?
    # look_dir = (-sin(yaw), 0, cos(yaw))
    # to_dragon = (dx, 0, dz)
    # We want: -sin(yaw) = dx/dist, cos(yaw) = dz/dist
    # So yaw = atan2(-dx, dz)
    target_yaw = np.degrees(np.arctan2(-dx, dz))
    if target_yaw < 0:
        target_yaw += 360

    yaw_diff = target_yaw - yaw
    if yaw_diff > 180:
        yaw_diff -= 360
    if yaw_diff < -180:
        yaw_diff += 360

    # Decide action
    phases = ["CIRC", "STRF", "CHRG", "LAND", "PRCH", "TOFF", "DEAD"]

    if dragon_phase == 4 and dragon_dist < 5:  # Perching and close
        if abs(yaw_diff) > 30:
            # Need to turn to face dragon
            actions[0] = 12 if yaw_diff < 0 else 13  # look left or right
            action_name = "TURNING"
        elif attack_ready:
            # Facing dragon, attack!
            actions[0] = 9
            action_name = "ATTACK!"
        else:
            actions[0] = 0  # Wait for cooldown
            action_name = "cooldown"
    else:
        actions[0] = 0
        action_name = "wait"

    prev_health = dragon_health
    sim.step(actions)

    new_obs = sim.get_observations()
    new_health = new_obs[0, 16] * 200
    rewards = sim.get_rewards()

    # Log perching and attacks
    if dragon_phase == 4 or (prev_health - new_health) > 0.1:
        print(
            f"Step {step}: phase={phases[dragon_phase]}, dist={dragon_dist:.1f}, "
            f"yaw={yaw:.0f}, target={target_yaw:.0f}, diff={yaw_diff:.0f}, "
            f"action={action_name}, hp={new_health:.0f}, reward={rewards[0]:.2f}"
        )

        if (prev_health - new_health) > 0.1:
            print(f"  >>> HIT! Dealt {prev_health - new_health:.0f} damage!")

obs = sim.get_observations()
final_health = obs[0, 16] * 200
print(f"\nFinal dragon health: {final_health:.0f}/200")
print(f"Damage dealt: {200 - final_health:.0f}")
