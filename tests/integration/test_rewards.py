#!/usr/bin/env python3
"""Test combat rewards and kills."""

from pathlib import Path

import numpy as np
import pytest

try:
    import mc189_core

    HAS_MC189 = True
except ImportError:

import logging

logger = logging.getLogger(__name__)

    HAS_MC189 = False
    mc189_core = None

pytestmark = pytest.mark.skipif(not HAS_MC189, reason="mc189_core not available")

_SIM_ROOT = Path(__file__).parent.parent.parent


def test_combat_rewards():
    """Test that combat gives rewards."""
    config = mc189_core.SimulatorConfig()
    config.num_envs = 256
    config.shader_dir = str(_SIM_ROOT / "cpp/shaders")

    sim = mc189_core.MC189Simulator(config)
    sim.reset()

    total_reward = 0
    for step in range(2000):
        obs = sim.get_observations()
        actions = np.zeros(256, dtype=np.int32)
        can_hit = obs[:, 28] > 0.5
        actions[can_hit] = 9
        sim.step(actions)
        rewards = sim.get_rewards()
        total_reward += rewards.sum()

    assert total_reward > 0, "Should receive some reward"
