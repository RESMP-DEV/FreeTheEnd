#!/usr/bin/env python3
"""Test and document the 48-float observation layout from dragon_fight_mvk.comp.

This test verifies that the observation buffer matches the expected layout
from the GLSL Observation struct and prints all values with labels.

Observation Layout (48 floats total):
=====================================

PLAYER (indices 0-15):
  0:  pos_x          - Normalized X position (raw / 100.0)
  1:  pos_y          - Normalized Y position ((raw - 64.0) / 50.0)
  2:  pos_z          - Normalized Z position (raw / 100.0)
  3:  vel_x          - Normalized X velocity (raw / 2.0)
  4:  vel_y          - Normalized Y velocity (raw / 2.0)
  5:  vel_z          - Normalized Z velocity (raw / 2.0)
  6:  yaw            - Normalized yaw (raw / 360.0), range [0, 1)
  7:  pitch          - Normalized pitch ((raw + 90.0) / 180.0), range [0, 1]
  8:  health         - Normalized health (raw / 20.0), range [0, 1]
  9:  hunger         - Normalized hunger (raw / 20.0), range [0, 1]
  10: on_ground      - Boolean: 1.0 if on ground, 0.0 if airborne
  11: attack_ready   - Boolean: 1.0 if cooldown == 0, 0.0 otherwise
  12: weapon         - Weapon slot (0=hand, 1=sword, 2=bow) / 2.0
  13: arrows         - Arrow count / 64.0
  14: arrow_charge   - Bow charge [0, 1]
  15: reserved0      - DEBUG: stores input action (0=none, 1=attack, 2=use, 3=swap)

DRAGON (indices 16-31):
  16: dragon_health  - Normalized health (raw / 200.0), range [0, 1]
  17: dragon_x       - Normalized X position (raw / 100.0)
  18: dragon_y       - Normalized Y position ((raw - 64.0) / 50.0)
  19: dragon_z       - Normalized Z position (raw / 100.0)
  20: dragon_vel_x   - Normalized X velocity (raw / 2.0)
  21: dragon_vel_y   - Normalized Y velocity (raw / 2.0)
  22: dragon_vel_z   - Normalized Z velocity (raw / 2.0)
  23: dragon_yaw     - Normalized yaw (raw / 360.0)
  24: dragon_phase   - Phase (0-6) / 6.0
                       Phases: 0=CIRCLING, 1=STRAFING, 2=CHARGING,
                              3=LANDING, 4=PERCHING, 5=TAKING_OFF, 6=DEAD
  25: dragon_dist    - Distance to player / 150.0, clamped [0, 1]
  26: dragon_dir_x   - X component of direction to dragon (unit vector)
  27: dragon_dir_z   - Z component of direction to dragon (unit vector)
  28: can_hit_dragon - Boolean: 1.0 if dragon perching AND dist < 5.0
  29: dragon_attacking - Boolean: 1.0 if phase is CHARGING or PERCHING
  30: reserved1      - DEBUG: stores attack_cooldown value
  31: reserved2      - DEBUG: stores player flags

ENVIRONMENT (indices 32-47):
  32: crystals_remaining - Remaining crystals / 10.0, range [0, 1]
  33: nearest_crystal_dist - Distance to nearest crystal / 100.0, clamped [0, 1]
  34: nearest_crystal_dir_x - X component of direction to nearest crystal
  35: nearest_crystal_dir_z - Z component of direction to nearest crystal
  36: nearest_crystal_y - Y position of nearest crystal ((raw - 64.0) / 50.0)
  37: portal_active  - Boolean: 1.0 if dragon dead
  38: portal_dist    - Distance to portal / 100.0
  39: time_remaining - 1.0 - (tick / 24000.0), 20 min game limit
  40: total_damage_dealt - Dragon damage / 200.0
  41: reserved3      - Reserved
  42: reserved4      - Reserved
  43: reserved5      - Reserved
  44: reserved6      - Reserved
  45: reserved7      - Reserved
  46: reserved8      - Reserved
  47: reserved9      - Reserved
"""

from pathlib import Path

import numpy as np

# Use the conftest.py for path setup - imports should work via pytest
_SIM_ROOT = Path(__file__).parent.parent.parent

# Observation labels with index, name, description, and denormalization info
OBSERVATION_LAYOUT = [
    # Player (0-15)
    (0, "pos_x", "Player X position", lambda v: v * 100.0),
    (1, "pos_y", "Player Y position", lambda v: v * 50.0 + 64.0),
    (2, "pos_z", "Player Z position", lambda v: v * 100.0),
    (3, "vel_x", "Player X velocity", lambda v: v * 2.0),
    (4, "vel_y", "Player Y velocity", lambda v: v * 2.0),
    (5, "vel_z", "Player Z velocity", lambda v: v * 2.0),
    (6, "yaw", "Player yaw (degrees)", lambda v: v * 360.0),
    (7, "pitch", "Player pitch (degrees)", lambda v: v * 180.0 - 90.0),
    (8, "health", "Player health (0-20)", lambda v: v * 20.0),
    (9, "hunger", "Player hunger (0-20)", lambda v: v * 20.0),
    (10, "on_ground", "On ground flag", lambda v: bool(v > 0.5)),
    (11, "attack_ready", "Attack cooldown ready", lambda v: bool(v > 0.5)),
    (12, "weapon", "Weapon slot (0=hand,1=sword,2=bow)", lambda v: int(v * 2.0)),
    (13, "arrows", "Arrow count", lambda v: int(v * 64.0)),
    (14, "arrow_charge", "Bow charge (0-1)", lambda v: v),
    (15, "reserved0", "DEBUG: last input action", lambda v: int(v)),
    # Dragon (16-31)
    (16, "dragon_health", "Dragon health (0-200)", lambda v: v * 200.0),
    (17, "dragon_x", "Dragon X position", lambda v: v * 100.0),
    (18, "dragon_y", "Dragon Y position", lambda v: v * 50.0 + 64.0),
    (19, "dragon_z", "Dragon Z position", lambda v: v * 100.0),
    (20, "dragon_vel_x", "Dragon X velocity", lambda v: v * 2.0),
    (21, "dragon_vel_y", "Dragon Y velocity", lambda v: v * 2.0),
    (22, "dragon_vel_z", "Dragon Z velocity", lambda v: v * 2.0),
    (23, "dragon_yaw", "Dragon yaw (degrees)", lambda v: v * 360.0),
    (24, "dragon_phase", "Dragon phase (0-6)", lambda v: int(v * 6.0)),
    (25, "dragon_dist", "Distance to dragon", lambda v: v * 150.0),
    (26, "dragon_dir_x", "Direction to dragon X", lambda v: v),
    (27, "dragon_dir_z", "Direction to dragon Z", lambda v: v),
    (28, "can_hit_dragon", "Can hit dragon flag", lambda v: bool(v > 0.5)),
    (29, "dragon_attacking", "Dragon attacking flag", lambda v: bool(v > 0.5)),
    (30, "reserved1", "DEBUG: attack cooldown", lambda v: int(v)),
    (31, "reserved2", "DEBUG: player flags", lambda v: int(v)),
    # Environment (32-47)
    (32, "crystals_remaining", "Crystals alive (0-10)", lambda v: int(v * 10.0)),
    (33, "nearest_crystal_dist", "Nearest crystal distance", lambda v: v * 100.0),
    (34, "nearest_crystal_dir_x", "Nearest crystal dir X", lambda v: v),
    (35, "nearest_crystal_dir_z", "Nearest crystal dir Z", lambda v: v),
    (36, "nearest_crystal_y", "Nearest crystal Y pos", lambda v: v * 50.0 + 64.0),
    (37, "portal_active", "Portal active flag", lambda v: bool(v > 0.5)),
    (38, "portal_dist", "Distance to portal", lambda v: v * 100.0),
    (39, "time_remaining", "Time remaining (0-1)", lambda v: v),
    (40, "total_damage_dealt", "Total dragon damage", lambda v: v * 200.0),
    (41, "reserved3", "Reserved", lambda v: v),
    (42, "reserved4", "Reserved", lambda v: v),
    (43, "reserved5", "Reserved", lambda v: v),
    (44, "reserved6", "Reserved", lambda v: v),
    (45, "reserved7", "Reserved", lambda v: v),
    (46, "reserved8", "Reserved", lambda v: v),
    (47, "reserved9", "Reserved", lambda v: v),
]

# Phase names for decoding
DRAGON_PHASES = [
    "CIRCLING",
    "STRAFING",
    "CHARGING",
    "LANDING",
    "PERCHING",
    "TAKING_OFF",
    "DEAD",
]


def print_observation(obs: np.ndarray, env_idx: int = 0) -> None:
    """Print all observation values with labels and decoded values.

    Args:
        obs: Observation array of shape (num_envs, 48)
        env_idx: Which environment to print
    """
    if obs.ndim == 1:
        obs_env = obs
    else:
        obs_env = obs[env_idx]

    assert len(obs_env) == 48, f"Expected 48 floats, got {len(obs_env)}"

    print(f"\n{'=' * 70}")
    print(f"OBSERVATION LAYOUT (env {env_idx})")
    print(f"{'=' * 70}")

    sections = [
        ("PLAYER (indices 0-15)", 0, 16),
        ("DRAGON (indices 16-31)", 16, 32),
        ("ENVIRONMENT (indices 32-47)", 32, 48),
    ]

    for section_name, start, end in sections:
        print(f"\n{section_name}")
        print("-" * 70)
        print(f"{'Idx':>3} {'Name':20} {'Normalized':>12} {'Decoded':>15}")
        print("-" * 70)

        for idx, name, desc, decode_fn in OBSERVATION_LAYOUT[start:end]:
            norm_val = obs_env[idx]
            decoded = decode_fn(norm_val)

            # Special formatting for phase
            if name == "dragon_phase":
                phase_idx = min(int(norm_val * 6.0), 6)
                decoded_str = f"{phase_idx} ({DRAGON_PHASES[phase_idx]})"
            elif isinstance(decoded, bool):
                decoded_str = "True" if decoded else "False"
            elif isinstance(decoded, float):
                decoded_str = f"{decoded:.2f}"
            else:
                decoded_str = str(decoded)

            print(f"{idx:>3} {name:20} {norm_val:>12.4f} {decoded_str:>15}")


def verify_observation_layout() -> None:
    """Verify observation layout matches expected 48 floats."""
    assert len(OBSERVATION_LAYOUT) == 48, f"Layout has {len(OBSERVATION_LAYOUT)} entries"

    for i, (idx, name, desc, decode_fn) in enumerate(OBSERVATION_LAYOUT):
        assert idx == i, f"Index mismatch at {i}: expected {idx}"

    print("Observation layout verification: PASSED (48 floats)")


def test_observation_layout_with_sim() -> None:
    """Test observation layout with actual simulator."""
    try:
        import mc189_core
    except ImportError:
        print("mc189_core not available, skipping simulation test")
        print("Run 'make' in cpp/ directory to build the simulator")
        return

    config = mc189_core.SimulatorConfig()
    config.num_envs = 4
    config.shader_dir = str(_SIM_ROOT / "cpp/shaders")

    print(f"Creating simulator with {config.num_envs} environments...")
    sim = mc189_core.MC189Simulator(config)
    sim.reset()

    # Step once to populate observations
    actions = np.zeros(config.num_envs, dtype=np.int32)
    sim.step(actions)

    obs = sim.get_observations()
    print(f"Observation shape: {obs.shape}")
    assert obs.shape == (config.num_envs, 48), f"Expected (4, 48), got {obs.shape}"

    # Print observations for each environment
    for env_idx in range(min(2, config.num_envs)):
        print_observation(obs, env_idx)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"Player position: ({obs[0, 0] * 100:.1f}, {obs[0, 1] * 50 + 64:.1f}, {obs[0, 2] * 100:.1f})"
    )
    print(
        f"Dragon position: ({obs[0, 17] * 100:.1f}, {obs[0, 18] * 50 + 64:.1f}, {obs[0, 19] * 100:.1f})"
    )
    print(f"Dragon health: {obs[0, 16] * 200:.0f}/200")
    print(f"Dragon distance: {obs[0, 25] * 150:.1f}")
    phase_idx = min(int(obs[0, 24] * 6), 6)
    print(f"Dragon phase: {DRAGON_PHASES[phase_idx]}")
    print(f"Crystals remaining: {int(obs[0, 32] * 10)}/10")


if __name__ == "__main__":
    verify_observation_layout()
    print()
    test_observation_layout_with_sim()
