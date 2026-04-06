import numpy as np

from .backend import VulkanBackend


class NormalizedObsWrapper:
    """Normalize observations for RL training."""

    # Observation ranges from game_tick.comp
    OBS_LOW = np.array(
        [
            -1000,
            -64,
            -1000,  # position
            -80,
            -80,
            -80,  # velocity
            0,
            -90,  # yaw, pitch
            0,
            0,
            0,  # health, hunger, saturation
            0,
            0,
            0,
            0,  # flags
            0,  # dimension
            0,
            -1,
            -1,
            0,  # nearest mob
            0,
            0,
            0,
            0,  # dragon
            0,
            0,
            0,
            -1,
            -1,  # goal
            0,
            0,
            0,  # reserved
        ],
        dtype=np.float32,
    )

    OBS_HIGH = np.array(
        [
            1000,
            320,
            1000,
            80,
            80,
            80,
            360,
            90,
            20,
            20,
            20,
            1,
            1,
            1,
            1,
            2,
            100,
            1,
            1,
            20,
            200,
            4,
            10,
            1,
            1,
            1,
            100,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=np.float32,
    )

    def __init__(self, backend: VulkanBackend):
        self.backend = backend
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
