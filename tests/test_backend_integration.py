# Target: contrib/minecraft_sim/tests/test_backend_integration.py
"""Integration tests for VulkanBackend - requires compiled mc189_core."""

import time

import numpy as np
import pytest

# Skip all tests if backend not available
pytest.importorskip("mc189_core")

from minecraft_sim.backend import VulkanBackend


class TestVulkanBackendIntegration:
    @pytest.fixture
    def backend(self):
        backend = VulkanBackend(num_envs=64, enable_validation=False)
        yield backend
        if hasattr(backend, "close"):
            backend.close()

    def test_initialization(self, backend):
        assert backend.device_name
        assert backend.num_envs == 64

    def test_single_step(self, backend):
        # Discrete actions: shape (num_envs,)
        actions = np.zeros(64, dtype=np.int32)
        obs, rewards, dones, infos = backend.step(actions)
        assert obs.shape == (64, backend.obs_dim)
        assert rewards.shape == (64,)
        assert dones.shape == (64,)

    @pytest.mark.parametrize("num_envs", [1, 64, 1024, 16384])
    def test_batch_sizes(self, num_envs):
        backend = VulkanBackend(num_envs=num_envs, enable_validation=False)
        try:
            # Discrete actions: shape (num_envs,)
            actions = np.zeros(num_envs, dtype=np.int32)
            obs, rewards, dones, infos = backend.step(actions)
            assert obs.shape == (num_envs, backend.obs_dim)
            assert rewards.shape == (num_envs,)
            assert dones.shape == (num_envs,)
        finally:
            if hasattr(backend, "close"):
                backend.close()

    def test_reset(self, backend):
        obs = backend.reset()
        assert obs.shape == (64, backend.obs_dim)

    def test_device_info(self, backend):
        if hasattr(backend, "get_device_info"):
            info = backend.get_device_info()
        elif hasattr(backend, "device_info"):
            info = backend.device_info()
        else:
            pytest.skip("device info API not available")
        assert isinstance(info, dict)
        assert info

    def test_throughput_benchmark(self, backend):
        """Target: 500K steps/sec on Apple M1+."""
        # Discrete actions: shape (num_envs,)
        actions = np.zeros(backend.num_envs, dtype=np.int32)

        warmup_steps = 5
        for _ in range(warmup_steps):
            backend.step(actions)

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            backend.step(actions)
        elapsed = time.perf_counter() - start

        steps_per_sec = (backend.num_envs * iterations) / max(elapsed, 1e-9)
        assert steps_per_sec > 0
