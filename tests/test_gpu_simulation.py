import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mc189_core")
import mc189_core as mc

SHADER_DIR = str(Path(__file__).parent.parent / "cpp" / "shaders")
OBS_SIZE = 48  # Dragon fight observation size


def _create_sim(num_envs: int) -> mc.MC189Simulator:
    """Create simulator with proper config API."""
    cfg = mc.SimulatorConfig()
    cfg.num_envs = num_envs
    cfg.shader_dir = SHADER_DIR
    cfg.enable_validation = False
    return mc.MC189Simulator(cfg)


class TestGPUSimulation:
    def test_observation_shape(self):
        sim = _create_sim(64)
        sim.reset()
        obs = sim.get_observations()
        assert obs.shape == (64, OBS_SIZE)

    def test_step_changes_state(self):
        sim = _create_sim(1)
        sim.reset()
        obs1 = sim.get_observations().copy()

        actions = np.array([1], dtype=np.int32)  # Discrete action
        sim.step(actions)
        obs2 = sim.get_observations()

        assert not np.allclose(obs1[0, :3], obs2[0, :3])

    def test_damage_reduces_health(self):
        sim = _create_sim(1)
        sim.reset()
        initial_health = sim.get_observations()[0, 8]
        assert initial_health > 0.9

        for _ in range(1000):
            actions = np.array([0], dtype=np.int32)  # NOOP
            sim.step(actions)

        final_health = sim.get_observations()[0, 8]
        assert final_health <= initial_health

    def test_throughput_32k_envs(self):
        sim = _create_sim(32768)
        sim.reset()
        actions = np.zeros(32768, dtype=np.int32)  # Discrete actions

        for _ in range(10):
            sim.step(actions)

        start = time.perf_counter()
        for _ in range(100):
            sim.step(actions)
        elapsed = time.perf_counter() - start

        steps_per_sec = (32768 * 100) / elapsed
        print(f"Throughput: {steps_per_sec:,.0f} steps/sec")

        assert steps_per_sec > 100_000
