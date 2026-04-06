#!/usr/bin/env python3
"""Debug dragon combat conditions with correct decoding.

Run from contrib/minecraft_sim/:
    PYTHONPATH=python:cpp/build python debug_combat2.py
"""

import mc189_core
import numpy as np

config = mc189_core.SimulatorConfig()
config.num_envs = 8
config.shader_dir = "cpp/shaders"

sim = mc189_core.MC189Simulator(config)
sim.reset()

actions = np.zeros(8, dtype=np.int32)


def decode_obs(obs_vec):
    """Decode observation vector with correct normalizations."""
    # Player (indices 0-15)
    px = obs_vec[0] * 100
    py = obs_vec[1] * 50 + 64
    pz = obs_vec[2] * 100
    pvx, pvy, pvz = obs_vec[3] * 2, obs_vec[4] * 2, obs_vec[5] * 2
    yaw = obs_vec[6] * 360
    pitch = obs_vec[7] * 180 - 90
    health = obs_vec[8] * 20
    hunger = obs_vec[9] * 20
    on_ground = obs_vec[10] > 0.5
    attack_ready = obs_vec[11] > 0.5
    weapon = int(obs_vec[12] * 2)
    arrows = int(obs_vec[13] * 64)
    arrow_charge = obs_vec[14]

    # Dragon (indices 16-29)
    dragon_health = obs_vec[16] * 200
    dragon_x = obs_vec[17] * 100
    dragon_y = obs_vec[18] * 50 + 64
    dragon_z = obs_vec[19] * 100
    dragon_vx = obs_vec[20] * 2
    dragon_vy = obs_vec[21] * 2
    dragon_vz = obs_vec[22] * 2
    dragon_yaw = obs_vec[23] * 360
    dragon_phase = int(obs_vec[24] * 6)
    dragon_dist = obs_vec[25] * 150
    dragon_dir_x = obs_vec[26]
    dragon_dir_z = obs_vec[27]
    can_hit = obs_vec[28] > 0.5
    dragon_attacking = obs_vec[29] > 0.5

    return {
        "player_pos": (px, py, pz),
        "player_vel": (pvx, pvy, pvz),
        "yaw": yaw,
        "pitch": pitch,
        "health": health,
        "hunger": hunger,
        "on_ground": on_ground,
        "attack_ready": attack_ready,
        "weapon": weapon,
        "arrows": arrows,
        "dragon_health": dragon_health,
        "dragon_pos": (dragon_x, dragon_y, dragon_z),
        "dragon_phase": dragon_phase,
        "dragon_dist": dragon_dist,
        "can_hit": can_hit,
        "dragon_attacking": dragon_attacking,
    }


print("Dragon Combat Debug (Correct Decoding)")
print("=" * 60)

# Run until dragon perches
for step in range(500):
    sim.step(actions)
    obs = sim.get_observations()

    d = decode_obs(obs[0])

    # Calculate real distance
    px, py, pz = d["player_pos"]
    dx, dy, dz = d["dragon_pos"]
    real_dist = np.sqrt((px - dx) ** 2 + (py - dy) ** 2 + (pz - dz) ** 2)

    phases = ["CIRCLING", "STRAFING", "CHARGING", "LANDING", "PERCHING", "TAKING_OFF", "DEAD"]
    phase_name = phases[d["dragon_phase"]] if d["dragon_phase"] < len(phases) else "???"

    if d["dragon_phase"] == 4 or step % 100 == 0:  # Perching or periodic
        print(f"\nStep {step}:")
        print(f"  Player:  ({px:.1f}, {py:.1f}, {pz:.1f})")
        print(f"  Dragon:  ({dx:.1f}, {dy:.1f}, {dz:.1f}) [{phase_name}]")
        print(f"  Health:  Player={d['health']:.1f}/20, Dragon={d['dragon_health']:.0f}/200")
        print(f"  Obs dist: {d['dragon_dist']:.1f}, Real dist: {real_dist:.1f}")
        print(f"  Can hit: {d['can_hit']}, Attack ready: {d['attack_ready']}")

        if d["dragon_phase"] == 4:
            print("  >>> Dragon is PERCHING!")
            if real_dist < 10:
                print("  >>> Distance < 10, combat should be possible!")
                if real_dist < 5:
                    print("  >>> Distance < 5, can_hit should be TRUE!")

# Check where dragon perches
print("\n" + "=" * 60)
print("Checking dragon perch positions...")

sim.reset()
perch_positions = []
for step in range(1000):
    sim.step(actions)
    obs = sim.get_observations()
    d = decode_obs(obs[0])

    if d["dragon_phase"] == 4:
        perch_positions.append(d["dragon_pos"])

if perch_positions:
    px, py, pz = perch_positions[0]
    print(f"Dragon perches at: ({px:.1f}, {py:.1f}, {pz:.1f})")
    print("Expected: (0, 68, 0)")
    print(f"Found {len(perch_positions)} perch frames")
else:
    print("No perching observed!")
