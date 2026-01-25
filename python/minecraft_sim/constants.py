"""Canonical constants for the Minecraft simulator.

This module is the single source of truth for dimension constants.
All other modules should import from here rather than hardcoding values.

Usage:
    from minecraft_sim.constants import OBSERVATION_SIZE, ACTION_SIZE

    # In your code:
    obs_space = spaces.Box(low=0, high=1, shape=(OBSERVATION_SIZE,), dtype=np.float32)
"""

from __future__ import annotations

# =============================================================================
# Observation Dimensions
# =============================================================================

# Main game observation size (dragon fight / speedrun)
# This is the primary observation vector returned by the simulator.
OBSERVATION_SIZE: int = 48

# Extended observation size for unified FreeTheEnd environment
# Includes additional context like inventory, nearby entities, etc.
EXTENDED_OBSERVATION_SIZE: int = 256

# Progress tracking observation size (SpeedrunProgress.to_observation())
# This is a SEPARATE vector tracking curriculum progress, not game state.
PROGRESS_OBSERVATION_SIZE: int = 32

# Survival observation size (base observation + resource detection)
SURVIVAL_OBSERVATION_SIZE: int = 64  # 48 + 16 resource features

# =============================================================================
# Action Dimensions
# =============================================================================

# Discrete action space size
ACTION_SIZE: int = 17

# Action components for multi-discrete action space
ACTION_COMPONENTS: dict[str, int] = {
    "forward": 3,      # -1, 0, 1
    "strafe": 3,       # -1, 0, 1
    "jump": 2,         # 0, 1
    "sneak": 2,        # 0, 1
    "sprint": 2,       # 0, 1
    "attack": 2,       # 0, 1
    "use": 2,          # 0, 1
    "yaw": 5,          # -2, -1, 0, 1, 2 (camera turn)
    "pitch": 5,        # -2, -1, 0, 1, 2 (camera look up/down)
}

# =============================================================================
# Simulation Parameters
# =============================================================================

# Game ticks per second (Minecraft standard)
TICKS_PER_SECOND: int = 20

# Maximum batch size for vectorized environments
MAX_BATCH_SIZE: int = 4096

# Default maximum episode length (5 minutes at 20 tps)
DEFAULT_MAX_EPISODE_STEPS: int = 6000

# Speedrun maximum episode length (30 minutes at 20 tps)
SPEEDRUN_MAX_EPISODE_STEPS: int = 36000

# =============================================================================
# Curriculum Stages
# =============================================================================

NUM_CURRICULUM_STAGES: int = 6

STAGE_NAMES: dict[int, str] = {
    1: "survival",
    2: "resources",
    3: "nether",
    4: "pearls",
    5: "stronghold",
    6: "dragon",
}

# =============================================================================
# Observation Layout Documentation
# =============================================================================

# 48-dimensional observation layout (OBSERVATION_SIZE):
#   [0-2]:   position (x, y, z)
#   [3-5]:   velocity (vx, vy, vz)
#   [6-7]:   yaw, pitch (look direction)
#   [8-10]:  health, hunger, saturation
#   [11-14]: flags (on_ground, in_water, in_lava, sprinting)
#   [15]:    dimension (0=overworld, 1=nether, 2=end)
#   [16-19]: nearest mob info (type, distance, dx, dz)
#   [20-23]: dragon state (health, phase, distance, angle)
#   [24-27]: goal info (distance, dx, dz, type)
#   [28-31]: inventory summary
#   [32-35]: crystal info (remaining, nearest_dist, nearest_dx, nearest_dz)
#   [36-39]: portal info (nearest_dist, angle, type, cooldown)
#   [40-43]: combat info (damage_cooldown, can_attack, target_in_range, target_angle)
#   [44-47]: reserved for expansion
