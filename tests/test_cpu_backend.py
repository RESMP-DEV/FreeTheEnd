"""Tests for the CPU backend of MC189Simulator.

These tests verify that the CPU fallback backend produces correct,
deterministic simulation results without requiring a GPU/Vulkan device.
"""

from pathlib import Path

import numpy as np
import pytest

mc = pytest.importorskip("mc189_core")

SHADER_DIR = str(Path(__file__).parent.parent / "cpp" / "shaders")


def _has_cpu_backend() -> bool:
    """Check if the CPU backend API is available."""
    cfg = mc.SimulatorConfig()
    return hasattr(cfg, "use_cpu")


def _supports_cpu_gpu_parity() -> bool:
    """Return True if CPU/GPU parity comparisons are implemented."""
    return bool(getattr(mc, "CPU_BACKEND_PARITY", False))


pytestmark = pytest.mark.skipif(
    not _has_cpu_backend(),
    reason="CPU backend not yet implemented (phase5)",
)


def _create_cpu_sim(num_envs: int = 4) -> mc.MC189Simulator:
    """Create a simulator configured to use the CPU backend."""
    cfg = mc.SimulatorConfig()
    cfg.use_cpu = True
    cfg.num_envs = num_envs
    cfg.shader_dir = SHADER_DIR
    cfg.enable_validation = False
    return mc.MC189Simulator(cfg)


class TestCPUBackendLoads:
    def test_cpu_backend_initializes(self):
        sim = _create_cpu_sim(4)
        assert sim.is_cpu_backend()

    def test_cpu_backend_reports_num_envs(self):
        sim = _create_cpu_sim(8)
        assert sim.num_envs == 8

    def test_cpu_backend_various_batch_sizes(self):
        for n in (1, 4, 16, 64):
            sim = _create_cpu_sim(n)
            assert sim.is_cpu_backend()
            assert sim.num_envs == n


class TestCPUBackendStep:
    def test_step_produces_valid_observations(self):
        sim = _create_cpu_sim(4)
        sim.reset()
        actions = np.zeros(4, dtype=np.int32)
        sim.step(actions)
        obs = sim.get_observations()
        assert obs.shape == (4, mc.MC189Simulator.obs_dim)
        assert np.isfinite(obs).all()

    def test_step_changes_state(self):
        sim = _create_cpu_sim(1)
        sim.reset()
        obs_before = sim.get_observations().copy()

        actions = np.array([1], dtype=np.int32)
        sim.step(actions)
        obs_after = sim.get_observations()

        assert not np.array_equal(obs_before, obs_after)

    def test_multiple_steps(self):
        sim = _create_cpu_sim(4)
        sim.reset()
        actions = np.zeros(4, dtype=np.int32)

        for _ in range(100):
            sim.step(actions)

        obs = sim.get_observations()
        assert obs.shape == (4, mc.MC189Simulator.obs_dim)
        assert np.isfinite(obs).all()

    def test_rewards_and_dones_shape(self):
        sim = _create_cpu_sim(4)
        sim.reset()
        actions = np.zeros(4, dtype=np.int32)
        sim.step(actions)

        rewards = sim.get_rewards()
        dones = sim.get_dones()
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert np.isfinite(rewards).all()


class TestCPUBackendReset:
    def test_reset_all(self):
        sim = _create_cpu_sim(4)
        sim.reset()
        obs = sim.get_observations()
        assert obs.shape == (4, mc.MC189Simulator.obs_dim)

    def test_reset_single_env(self):
        sim = _create_cpu_sim(4)
        sim.reset()

        actions = np.zeros(4, dtype=np.int32)
        for _ in range(10):
            sim.step(actions)

        obs_before = sim.get_observations().copy()
        sim.reset(0)  # Reset only env 0
        obs_after = sim.get_observations()

        # Env 0 should differ; others unchanged
        assert not np.array_equal(obs_before[0], obs_after[0])
        np.testing.assert_array_equal(obs_before[1:], obs_after[1:])

    def test_reset_produces_finite_observations(self):
        sim = _create_cpu_sim(4)
        sim.reset()
        obs = sim.get_observations()
        assert np.isfinite(obs).all()


class TestCPUBackendDeterminism:
    def test_same_seed_same_initial_obs(self):
        sim1 = _create_cpu_sim(4)
        sim2 = _create_cpu_sim(4)

        sim1.reset(seed=42)
        sim2.reset(seed=42)

        np.testing.assert_array_equal(
            sim1.get_observations(), sim2.get_observations()
        )

    def test_same_seed_same_trajectory(self):
        sim1 = _create_cpu_sim(4)
        sim2 = _create_cpu_sim(4)

        sim1.reset(seed=42)
        sim2.reset(seed=42)

        actions = np.array([1, 0, 2, 1], dtype=np.int32)
        for _ in range(50):
            sim1.step(actions)
            sim2.step(actions)

        np.testing.assert_array_equal(
            sim1.get_observations(), sim2.get_observations()
        )
        np.testing.assert_array_equal(
            sim1.get_rewards(), sim2.get_rewards()
        )
        np.testing.assert_array_equal(
            sim1.get_dones(), sim2.get_dones()
        )

    def test_different_seeds_different_results(self):
        sim1 = _create_cpu_sim(4)
        sim2 = _create_cpu_sim(4)

        sim1.reset(seed=42)
        sim2.reset(seed=99)

        actions = np.zeros(4, dtype=np.int32)
        for _ in range(10):
            sim1.step(actions)
            sim2.step(actions)

        assert not np.array_equal(
            sim1.get_observations(), sim2.get_observations()
        )


class TestCPUGPUParity:
    """Verify CPU and GPU backends produce similar (not necessarily identical) results.

    Floating-point order of operations differs between CPU serial execution and
    GPU parallel execution, so we allow a tolerance.
    """

    @pytest.fixture
    def gpu_available(self) -> bool:
        cfg = mc.SimulatorConfig()
        cfg.num_envs = 1
        cfg.use_cpu = False
        cfg.shader_dir = SHADER_DIR
        cfg.enable_validation = False
        try:
            mc.MC189Simulator(cfg)
            return True
        except Exception:
            return False

    def test_observation_shapes_match(self, gpu_available):
        if not _supports_cpu_gpu_parity():
            pytest.skip("CPU/GPU parity comparisons not implemented yet")
        if not gpu_available:
            pytest.skip("GPU backend not available for parity test")

        cpu_sim = _create_cpu_sim(4)
        cpu_sim.reset(seed=1)

        cfg = mc.SimulatorConfig()
        cfg.num_envs = 4
        cfg.use_cpu = False
        cfg.shader_dir = SHADER_DIR
        cfg.enable_validation = False
        gpu_sim = mc.MC189Simulator(cfg)
        gpu_sim.reset(seed=1)

        assert cpu_sim.get_observations().shape == gpu_sim.get_observations().shape

    def test_observations_within_tolerance(self, gpu_available):
        if not _supports_cpu_gpu_parity():
            pytest.skip("CPU/GPU parity comparisons not implemented yet")
        if not gpu_available:
            pytest.skip("GPU backend not available for parity test")

        cpu_sim = _create_cpu_sim(4)
        cpu_sim.reset(seed=7)

        cfg = mc.SimulatorConfig()
        cfg.num_envs = 4
        cfg.use_cpu = False
        cfg.shader_dir = SHADER_DIR
        cfg.enable_validation = False
        gpu_sim = mc.MC189Simulator(cfg)
        gpu_sim.reset(seed=7)

        actions = np.zeros(4, dtype=np.int32)
        for _ in range(20):
            cpu_sim.step(actions)
            gpu_sim.step(actions)

        cpu_obs = cpu_sim.get_observations()
        gpu_obs = gpu_sim.get_observations()

        # Allow tolerance for float precision differences between backends
        np.testing.assert_allclose(cpu_obs, gpu_obs, rtol=1e-3, atol=1e-3)
