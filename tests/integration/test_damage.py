#!/usr/bin/env python3
"""Test if dragon actually takes damage now with reach fix."""

from pathlib import Path

import numpy as np

from minecraft_sim import mc189_core

# Check GPU support
print("Device info:", mc189_core.get_device_info()["device_name"])

# Create simulator using the core module directly
print("\nCreating MC189Simulator...")
config = mc189_core.SimulatorConfig()
config.num_envs = 1
config.shader_dir = str(Path(__file__).parent.parent.parent / "cpp" / "shaders")
sim = mc189_core.MC189Simulator(config)

# Attack action
attack_action = 9  # maps to inp.action = 1

print("\nTesting dragon damage with reach fix (4.5 blocks, <= comparison)")
print("=" * 60)

# Reset and run
sim.reset()

max_steps = 5000
for step in range(max_steps):
    obs = sim.get_observations()[0]  # Get first env's obs (48 floats)

    # Parse observation - using correct indices from shader Observation struct
    # Player: 0-15, Dragon: 16-31, Environment: 32-47
    dragon_hp = obs[16] * 200  # dragon_health (normalized to 0-1)
    dragon_phase = int(obs[24] * 6)  # dragon_phase (normalized to 0-1, multiply by 6 phases)
    can_hit = obs[28]  # can_hit_dragon
    dragon_dist = obs[25] * 150  # dragon_dist (normalized to 0-1)

    if step % 500 == 0:
        print(
            f"Step {step}: dragon_hp={dragon_hp:.0f}, phase={dragon_phase}, "
            f"dist={dragon_dist:.1f}, can_hit={can_hit:.1f}"
        )

    # When perching (phase 4), attack
    if dragon_phase == 4:
        print(f"\n>>> Step {step}: PERCHING - attacking!")
        print(f"    dragon_dist={dragon_dist:.2f}, can_hit={can_hit:.1f}")

        initial_hp = dragon_hp

        # Attack multiple times
        for i in range(20):
            actions = np.array([attack_action], dtype=np.int32)
            sim.step(actions)

            obs = sim.get_observations()[0]
            new_hp = obs[16] * 200  # dragon_health
            reward = sim.get_rewards()[0]
            phase = int(obs[24] * 6)
            dist = obs[25] * 150
            hit = obs[28]

            print(
                f"    Hit {i + 1}: reward={reward:.2f}, hp={new_hp:.0f}, phase={phase}, dist={dist:.1f}, can_hit={hit:.1f}"
            )

        # Check final state
        obs = sim.get_observations()[0]
        final_hp = obs[16] * 200  # dragon_health
        print(f"\n    After 20 attacks: dragon_hp={final_hp:.0f}")

        if final_hp < initial_hp:
            print("\n✅ SUCCESS! Dragon took damage!")
            total_dmg = initial_hp - final_hp
            print(f"   Total damage dealt: {total_dmg:.0f}")
        else:
            print("\n❌ FAILED - Dragon still at full HP")
            print("   Checking can_hit during attack...")
            print(f"   can_hit={obs[28]:.1f}, dragon_phase={int(obs[24] * 6)}")
        break

    # Just idle wait
    actions = np.array([0], dtype=np.int32)
    sim.step(actions)

else:
    print("\n❌ Dragon never perched in 5000 steps")

print("\n" + "=" * 60)
