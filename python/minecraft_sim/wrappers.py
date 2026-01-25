import numpy as np

from .backend import VulkanBackend


class NormalizedObsWrapper:
    """Normalize observations for RL training.

    This wrapper normalizes observations to [-1, 1] range for more stable
    neural network training. The normalization bounds are based on the
    48-dimensional observation vector from the MC189 simulator.

    Observation layout (48 floats):
        [0-2]: position (x, y, z)
        [3-5]: velocity (vx, vy, vz)
        [6-7]: yaw, pitch (look direction)
        [8-10]: health, hunger, saturation
        [11-14]: flags (on_ground, in_water, in_lava, sprinting)
        [15]: dimension (0=overworld, 1=nether, 2=end)
        [16-19]: nearest mob info (type, distance, dx, dz)
        [20-23]: dragon state (health, phase, distance, angle)
        [24-27]: goal info (distance, dx, dz, type)
        [28-31]: inventory summary
        [32-35]: crystal info (remaining, nearest_dist, nearest_dx, nearest_dz)
        [36-39]: portal info (nearest_dist, angle, type, cooldown)
        [40-43]: combat info (damage_cooldown, can_attack, target_in_range, target_angle)
        [44-47]: reserved for expansion
    """

    # Observation ranges for 48-dimensional observations from game_tick.comp
    OBS_LOW = np.array(
        [
            # [0-2] Position
            -1000, -64, -1000,
            # [3-5] Velocity
            -80, -80, -80,
            # [6-7] Yaw, Pitch
            0, -90,
            # [8-10] Health, Hunger, Saturation
            0, 0, 0,
            # [11-14] Flags (on_ground, in_water, in_lava, sprinting)
            0, 0, 0, 0,
            # [15] Dimension
            0,
            # [16-19] Nearest mob info
            0, -1, -1, 0,
            # [20-23] Dragon state
            0, 0, 0, -1,
            # [24-27] Goal info
            0, -1, -1, 0,
            # [28-31] Inventory summary
            0, 0, 0, 0,
            # [32-35] Crystal info
            0, 0, -1, -1,
            # [36-39] Portal info
            0, -1, 0, 0,
            # [40-43] Combat info
            0, 0, 0, -1,
            # [44-47] Reserved
            0, 0, 0, 0,
        ],
        dtype=np.float32,
    )

    OBS_HIGH = np.array(
        [
            # [0-2] Position
            1000, 320, 1000,
            # [3-5] Velocity
            80, 80, 80,
            # [6-7] Yaw, Pitch
            360, 90,
            # [8-10] Health, Hunger, Saturation
            20, 20, 20,
            # [11-14] Flags
            1, 1, 1, 1,
            # [15] Dimension
            2,
            # [16-19] Nearest mob info (type, distance, dx, dz)
            100, 1, 1, 20,
            # [20-23] Dragon state (health, phase, distance, angle)
            200, 6, 100, 1,
            # [24-27] Goal info
            1000, 1, 1, 10,
            # [28-31] Inventory summary
            64, 64, 64, 64,
            # [32-35] Crystal info (remaining, nearest_dist, nearest_dx, nearest_dz)
            10, 100, 1, 1,
            # [36-39] Portal info (nearest_dist, angle, type, cooldown)
            100, 1, 3, 1,
            # [40-43] Combat info
            1, 1, 1, 1,
            # [44-47] Reserved
            1, 1, 1, 1,
        ],
        dtype=np.float32,
    )

    def __init__(self, backend: VulkanBackend):
        self.backend = backend
        self.obs_dim = backend.obs_dim

        # Validate dimension consistency
        if len(self.OBS_LOW) != self.obs_dim:
            # Fall back to dynamic bounds if dimensions don't match
            self.OBS_LOW = np.zeros(self.obs_dim, dtype=np.float32)
            self.OBS_HIGH = np.ones(self.obs_dim, dtype=np.float32)

        self.obs_range = self.OBS_HIGH - self.OBS_LOW
        self.obs_range[self.obs_range == 0] = 1  # Avoid division by zero

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize to [-1, 1] range."""
        return 2 * (obs - self.OBS_LOW) / self.obs_range - 1

    def step(self, actions):
        obs, rewards, dones, infos = self.backend.step(actions)
        return self.normalize(obs), rewards, dones, infos

    def reset(self, **kwargs):
        obs = self.backend.reset(**kwargs)
        return self.normalize(obs)
