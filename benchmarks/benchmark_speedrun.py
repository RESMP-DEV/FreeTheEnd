#!/usr/bin/env python3
"""Comprehensive performance benchmarks for Minecraft Dragon Fight simulator.

This module benchmarks:
1. Step throughput across various num_envs configurations
2. Reset latency (full reset and per-env reset)
3. Observation transfer time (GPU -> CPU)
4. Per-stage overhead breakdown
5. Memory usage scaling with environment count
6. Comparison with MineRL baseline

Run:
    cd contrib/minecraft_sim
    PYTHONPATH=python:cpp/build python benchmarks/benchmark_speedrun.py

Generate report:
    PYTHONPATH=python:cpp/build python benchmarks/benchmark_speedrun.py --report
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Ensure correct path resolution
_SCRIPT_DIR = Path(__file__).resolve().parent
_SIM_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SIM_ROOT / "python"))
sys.path.insert(0, str(_SIM_ROOT / "cpp" / "build"))

try:
    from minecraft_sim import SB3VecDragonFightEnv, VecDragonFightEnv
    from minecraft_sim.backend import VulkanBackend

    HAS_SIMULATOR = True
except ImportError as e:
    print(f"Warning: Could not import minecraft_sim: {e}")
    HAS_SIMULATOR = False

try:
    from minecraft_sim.curriculum import AutoCurriculumManager, CurriculumConfig, StageID
    from minecraft_sim.progression import ProgressTracker, SpeedrunProgress, SpeedrunStage

    HAS_CURRICULUM = True
except ImportError as e:
    print(f"Warning: Could not import curriculum/progression modules: {e}")
    HAS_CURRICULUM = False

try:
    from minecraft_sim.speedrun_vec_env import SpeedrunVecEnv

    HAS_VEC_ENV = True
except ImportError as e:
    print(f"Warning: Could not import SpeedrunVecEnv: {e}")
    HAS_VEC_ENV = False

try:
    from minecraft_sim.triangulation import TriangulationState, triangulate_stronghold

    HAS_TRIANGULATION = True
except ImportError as e:
    print(f"Warning: Could not import triangulation module: {e}")
    HAS_TRIANGULATION = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# -----------------------------------------------------------------------------
# Benchmark Configuration
# -----------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark suite."""

    # Throughput benchmark
    throughput_env_counts: list[int] = field(
        default_factory=lambda: [1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096]
    )
    throughput_warmup_steps: int = 100
    throughput_benchmark_steps: int = 1000

    # Reset benchmark
    reset_iterations: int = 100
    reset_env_counts: list[int] = field(default_factory=lambda: [1, 64, 256, 1024, 4096])

    # Observation transfer benchmark
    obs_transfer_iterations: int = 1000
    obs_transfer_env_counts: list[int] = field(default_factory=lambda: [64, 256, 1024, 4096])

    # Memory benchmark
    memory_env_counts: list[int] = field(default_factory=lambda: [64, 256, 512, 1024, 2048, 4096])

    # Per-stage overhead
    stage_iterations: int = 500

    # Stage 2 resource gathering benchmark
    stage2_num_agents: int = 50
    stage2_warmup_steps: int = 200
    stage2_benchmark_steps: int = 2000
    stage2_log_interval: int = 100  # Log metrics every N steps

    # Output
    output_dir: Path = field(default_factory=lambda: _SCRIPT_DIR / "results")


@dataclass
class BenchmarkResults:
    """Container for all benchmark results."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    platform_info: dict[str, Any] = field(default_factory=dict)
    throughput: dict[str, Any] = field(default_factory=dict)
    reset_latency: dict[str, Any] = field(default_factory=dict)
    obs_transfer: dict[str, Any] = field(default_factory=dict)
    stage_overhead: dict[str, Any] = field(default_factory=dict)
    memory_scaling: dict[str, Any] = field(default_factory=dict)
    minerl_comparison: dict[str, Any] = field(default_factory=dict)
    stage2_resources: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "platform_info": self.platform_info,
            "throughput": self.throughput,
            "reset_latency": self.reset_latency,
            "obs_transfer": self.obs_transfer,
            "stage_overhead": self.stage_overhead,
            "memory_scaling": self.memory_scaling,
            "minerl_comparison": self.minerl_comparison,
            "stage2_resources": self.stage2_resources,
        }


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def get_platform_info() -> dict[str, Any]:
    """Collect platform and hardware information."""
    info: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # macOS specific
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info["cpu_brand"] = result.stdout.strip()

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info["total_memory_gb"] = int(result.stdout.strip()) / (1024**3)
        except Exception:
            pass

    # GPU info from Vulkan backend
    if HAS_SIMULATOR:
        try:
            backend = VulkanBackend(num_envs=1)
            info["vulkan_device"] = backend.device_name
            del backend
        except Exception as e:
            info["vulkan_device"] = f"Error: {e}"

    return info


def format_sps(sps: float) -> str:
    """Format steps per second with appropriate unit."""
    if sps >= 1e6:
        return f"{sps / 1e6:.2f}M"
    if sps >= 1e3:
        return f"{sps / 1e3:.1f}K"
    return f"{sps:.0f}"


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB.

    Uses psutil if available for accurate cross-platform memory measurement,
    falls back to resource module otherwise.
    """
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    try:
        import resource

        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS returns bytes, Linux returns KB
        if platform.system() == "Darwin":
            return rusage.ru_maxrss / (1024 * 1024)
        return rusage.ru_maxrss / 1024
    except Exception:
        return 0.0


def percentile_stats(data: NDArray[np.float64]) -> dict[str, float]:
    """Compute percentile statistics for latency data."""
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


# -----------------------------------------------------------------------------
# Benchmark Functions
# -----------------------------------------------------------------------------


def benchmark_throughput(config: BenchmarkConfig) -> dict[str, Any]:
    """Benchmark step throughput across various environment counts.

    Measures sustained steps-per-second for different vectorization levels.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Step Throughput")
    print("=" * 70)

    results: dict[str, Any] = {
        "env_counts": [],
        "steps_per_second": [],
        "steps_per_env_per_second": [],
        "total_steps": [],
        "elapsed_seconds": [],
    }

    for num_envs in config.throughput_env_counts:
        print(f"\n  Testing num_envs={num_envs}...")

        try:
            backend = VulkanBackend(num_envs=num_envs)
            backend.reset()

            # Warmup
            actions = np.zeros(num_envs, dtype=np.int32)
            for _ in range(config.throughput_warmup_steps):
                backend.step(actions)

            # Benchmark
            gc.collect()
            start = time.perf_counter()
            for _ in range(config.throughput_benchmark_steps):
                actions = np.random.randint(0, 17, size=num_envs, dtype=np.int32)
                backend.step(actions)
            elapsed = time.perf_counter() - start

            total_steps = num_envs * config.throughput_benchmark_steps
            sps = total_steps / elapsed
            sps_per_env = sps / num_envs

            results["env_counts"].append(num_envs)
            results["steps_per_second"].append(sps)
            results["steps_per_env_per_second"].append(sps_per_env)
            results["total_steps"].append(total_steps)
            results["elapsed_seconds"].append(elapsed)

            print(
                f"    {num_envs:>5} envs: {format_sps(sps):>8} steps/s "
                f"({sps_per_env:.1f} steps/env/s)"
            )

            del backend
            gc.collect()

        except Exception as e:
            print(f"    {num_envs:>5} envs: ERROR - {e}")

    # Find optimal configuration
    if results["steps_per_second"]:
        best_idx = np.argmax(results["steps_per_second"])
        results["optimal_num_envs"] = results["env_counts"][best_idx]
        results["peak_throughput"] = results["steps_per_second"][best_idx]
        print(
            f"\n  Peak: {format_sps(results['peak_throughput'])} steps/s "
            f"at num_envs={results['optimal_num_envs']}"
        )

    return results


def benchmark_reset_latency(config: BenchmarkConfig) -> dict[str, Any]:
    """Benchmark environment reset latency.

    Measures both full reset (all envs) and per-environment reset times.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Reset Latency")
    print("=" * 70)

    results: dict[str, Any] = {
        "full_reset": {},
        "per_env_reset": {},
    }

    for num_envs in config.reset_env_counts:
        print(f"\n  Testing num_envs={num_envs}...")

        try:
            backend = VulkanBackend(num_envs=num_envs)

            # Full reset latency
            latencies = []
            for _ in range(config.reset_iterations):
                gc.collect()
                start = time.perf_counter()
                backend.reset()
                elapsed = time.perf_counter() - start
                latencies.append(elapsed * 1000)  # Convert to ms

            stats = percentile_stats(np.array(latencies))
            results["full_reset"][str(num_envs)] = stats
            print(f"    Full reset: {stats['mean']:.3f}ms (p99: {stats['p99']:.3f}ms)")

            del backend
            gc.collect()

        except Exception as e:
            print(f"    {num_envs:>5} envs: ERROR - {e}")

    return results


def benchmark_obs_transfer(config: BenchmarkConfig) -> dict[str, Any]:
    """Benchmark observation transfer time from GPU to CPU.

    Measures the time to retrieve observations after stepping.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Observation Transfer Time")
    print("=" * 70)

    results: dict[str, Any] = {}

    for num_envs in config.obs_transfer_env_counts:
        print(f"\n  Testing num_envs={num_envs}...")

        try:
            backend = VulkanBackend(num_envs=num_envs)
            backend.reset()
            actions = np.zeros(num_envs, dtype=np.int32)

            # Warmup
            for _ in range(50):
                backend.step(actions)

            # Measure step + get_observations separately (not possible with VulkanBackend)
            # Instead measure full step overhead vs pure compute
            step_latencies = []
            for _ in range(config.obs_transfer_iterations):
                start = time.perf_counter()
                backend.step(actions)
                elapsed = time.perf_counter() - start
                step_latencies.append(elapsed * 1000)

            stats = percentile_stats(np.array(step_latencies))
            bytes_per_step = num_envs * 48 * 4  # 48 float32s per env
            bandwidth_gbps = (bytes_per_step / stats["mean"]) * 1000 / (1024**3)

            results[str(num_envs)] = {
                "step_latency_ms": stats,
                "obs_bytes": bytes_per_step,
                "effective_bandwidth_gbps": bandwidth_gbps,
            }
            print(
                f"    Step latency: {stats['mean']:.3f}ms (p99: {stats['p99']:.3f}ms), "
                f"~{bandwidth_gbps:.2f} GB/s effective"
            )

            del backend
            gc.collect()

        except Exception as e:
            print(f"    {num_envs:>5} envs: ERROR - {e}")

    return results


def benchmark_stage_overhead(config: BenchmarkConfig) -> dict[str, Any]:
    """Benchmark per-stage overhead in the simulation pipeline.

    Breaks down timing for different components of the step cycle.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Per-Stage Overhead")
    print("=" * 70)

    results: dict[str, Any] = {}
    num_envs = 1024  # Fixed for this benchmark

    try:
        backend = VulkanBackend(num_envs=num_envs)
        backend.reset()

        # Measure action preparation time
        action_times = []
        for _ in range(config.stage_iterations):
            start = time.perf_counter()
            actions = np.random.randint(0, 17, size=num_envs, dtype=np.int32)
            elapsed = time.perf_counter() - start
            action_times.append(elapsed * 1000)

        # Measure full step time
        step_times = []
        actions = np.zeros(num_envs, dtype=np.int32)
        for _ in range(config.stage_iterations):
            start = time.perf_counter()
            backend.step(actions)
            elapsed = time.perf_counter() - start
            step_times.append(elapsed * 1000)

        # Measure numpy operations overhead
        numpy_times = []
        obs_template = np.zeros((num_envs, 48), dtype=np.float32)
        for _ in range(config.stage_iterations):
            start = time.perf_counter()
            obs_copy = obs_template.copy()
            rewards = np.zeros(num_envs, dtype=np.float32)
            dones = np.zeros(num_envs, dtype=bool)
            _ = obs_copy, rewards, dones
            elapsed = time.perf_counter() - start
            numpy_times.append(elapsed * 1000)

        results = {
            "num_envs": num_envs,
            "action_prep_ms": percentile_stats(np.array(action_times)),
            "step_total_ms": percentile_stats(np.array(step_times)),
            "numpy_overhead_ms": percentile_stats(np.array(numpy_times)),
        }

        step_mean = results["step_total_ms"]["mean"]
        action_mean = results["action_prep_ms"]["mean"]
        numpy_mean = results["numpy_overhead_ms"]["mean"]
        compute_estimate = step_mean - numpy_mean

        print(f"\n  num_envs = {num_envs}")
        print(f"    Action preparation:  {action_mean:.4f}ms")
        print(f"    NumPy overhead:      {numpy_mean:.4f}ms")
        print(f"    Total step:          {step_mean:.4f}ms")
        print(f"    GPU compute (est):   {compute_estimate:.4f}ms")

        results["gpu_compute_estimate_ms"] = compute_estimate

        del backend
        gc.collect()

    except Exception as e:
        print(f"  ERROR: {e}")

    return results


def benchmark_memory_scaling(config: BenchmarkConfig) -> dict[str, Any]:
    """Benchmark memory usage scaling with environment count.

    Measures peak memory usage for different vectorization levels.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: Memory Usage Scaling")
    print("=" * 70)

    results: dict[str, Any] = {
        "env_counts": [],
        "memory_mb": [],
        "memory_per_env_kb": [],
    }

    # Baseline memory
    gc.collect()
    baseline_memory = get_process_memory_mb()

    for num_envs in config.memory_env_counts:
        print(f"\n  Testing num_envs={num_envs}...")

        try:
            gc.collect()
            pre_memory = get_process_memory_mb()

            backend = VulkanBackend(num_envs=num_envs)
            backend.reset()

            # Do some steps to ensure full allocation
            actions = np.zeros(num_envs, dtype=np.int32)
            for _ in range(10):
                backend.step(actions)

            gc.collect()
            post_memory = get_process_memory_mb()

            memory_used = post_memory - pre_memory
            memory_per_env = (memory_used * 1024) / num_envs  # KB per env

            results["env_counts"].append(num_envs)
            results["memory_mb"].append(memory_used)
            results["memory_per_env_kb"].append(memory_per_env)

            print(f"    Memory: {memory_used:.1f} MB ({memory_per_env:.2f} KB/env)")

            del backend
            gc.collect()

        except Exception as e:
            print(f"    {num_envs:>5} envs: ERROR - {e}")

    results["baseline_memory_mb"] = baseline_memory
    return results


def benchmark_minerl_comparison(config: BenchmarkConfig) -> dict[str, Any]:
    """Generate comparison data with MineRL baseline.

    Uses published MineRL benchmarks as reference since MineRL requires
    a full Minecraft installation and Java runtime.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK: MineRL Comparison")
    print("=" * 70)

    # MineRL published baseline numbers (from their documentation)
    minerl_baseline = {
        "single_env_sps": 60,  # ~60 steps/sec for single env
        "vec_env_sps": 120,  # ~120 steps/sec for 2 envs (limited scaling)
        "notes": "MineRL baselines from published benchmarks (Java-based, CPU-bound)",
    }

    # Run our benchmark at comparable configs
    our_results: dict[str, Any] = {}

    try:
        # Single env comparison
        backend = VulkanBackend(num_envs=1)
        backend.reset()

        actions = np.zeros(1, dtype=np.int32)
        for _ in range(100):  # warmup
            backend.step(actions)

        gc.collect()
        start = time.perf_counter()
        for _ in range(1000):
            backend.step(actions)
        elapsed = time.perf_counter() - start

        our_single_sps = 1000 / elapsed
        our_results["single_env_sps"] = our_single_sps
        speedup_single = our_single_sps / minerl_baseline["single_env_sps"]

        print("\n  Single Environment:")
        print(f"    MineRL:     {minerl_baseline['single_env_sps']:>10} steps/s")
        print(f"    This sim:   {our_single_sps:>10.0f} steps/s")
        print(f"    Speedup:    {speedup_single:>10.0f}x")

        del backend
        gc.collect()

        # 64-env comparison (MineRL doesn't scale well, use their max effective)
        backend = VulkanBackend(num_envs=64)
        backend.reset()

        actions = np.zeros(64, dtype=np.int32)
        for _ in range(100):
            backend.step(actions)

        gc.collect()
        start = time.perf_counter()
        for _ in range(1000):
            backend.step(actions)
        elapsed = time.perf_counter() - start

        our_vec_sps = 64 * 1000 / elapsed
        our_results["vec_64_env_sps"] = our_vec_sps
        speedup_vec = our_vec_sps / minerl_baseline["vec_env_sps"]

        print("\n  Vectorized (64 envs vs MineRL's ~2 env limit):")
        print(f"    MineRL (2):   {minerl_baseline['vec_env_sps']:>10} steps/s")
        print(f"    This sim (64):{our_vec_sps:>10.0f} steps/s")
        print(f"    Speedup:      {speedup_vec:>10.0f}x")

        del backend
        gc.collect()

        # Peak throughput comparison
        backend = VulkanBackend(num_envs=4096)
        backend.reset()

        actions = np.zeros(4096, dtype=np.int32)
        for _ in range(100):
            backend.step(actions)

        gc.collect()
        start = time.perf_counter()
        for _ in range(1000):
            backend.step(actions)
        elapsed = time.perf_counter() - start

        our_peak_sps = 4096 * 1000 / elapsed
        our_results["peak_sps"] = our_peak_sps
        speedup_peak = our_peak_sps / minerl_baseline["single_env_sps"]

        print("\n  Peak Throughput (4096 envs):")
        print(f"    MineRL:       {minerl_baseline['single_env_sps']:>10} steps/s")
        print(f"    This sim:     {our_peak_sps:>10.0f} steps/s")
        print(f"    Speedup:      {speedup_peak:>10.0f}x")

        del backend
        gc.collect()

    except Exception as e:
        print(f"  ERROR: {e}")

    return {
        "minerl_baseline": minerl_baseline,
        "our_results": our_results,
    }


# -----------------------------------------------------------------------------
# Stage 1 Curriculum Benchmark (50 agents)
# -----------------------------------------------------------------------------


@dataclass
class Stage1ProgressMetrics:
    """Aggregate progress metrics for Stage 1 curriculum across all agents."""

    num_agents: int = 50
    total_episodes: int = 0
    total_steps: int = 0
    elapsed_seconds: float = 0.0

    # Per-agent progress summaries
    wood_collected: list[int] = field(default_factory=list)
    stone_collected: list[int] = field(default_factory=list)
    zombies_killed: list[int] = field(default_factory=list)
    food_eaten: list[int] = field(default_factory=list)
    nights_survived: list[bool] = field(default_factory=list)
    stage_completions: list[float] = field(default_factory=list)

    # Curriculum manager stats
    success_rate: float = 0.0
    advancements: int = 0
    mean_episode_reward: float = 0.0
    best_episode_reward: float = 0.0

    # Throughput
    steps_per_second: float = 0.0
    episodes_per_second: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_agents": self.num_agents,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "elapsed_seconds": self.elapsed_seconds,
            "steps_per_second": self.steps_per_second,
            "episodes_per_second": self.episodes_per_second,
            "curriculum": {
                "success_rate": self.success_rate,
                "advancements": self.advancements,
                "mean_episode_reward": self.mean_episode_reward,
                "best_episode_reward": self.best_episode_reward,
            },
            "progress_per_agent": {
                "wood_collected": self.wood_collected,
                "stone_collected": self.stone_collected,
                "zombies_killed": self.zombies_killed,
                "food_eaten": self.food_eaten,
                "nights_survived": self.nights_survived,
                "stage_completions": self.stage_completions,
            },
            "progress_aggregates": {
                "mean_wood": float(np.mean(self.wood_collected)) if self.wood_collected else 0.0,
                "mean_stone": float(np.mean(self.stone_collected)) if self.stone_collected else 0.0,
                "mean_zombies_killed": float(np.mean(self.zombies_killed)) if self.zombies_killed else 0.0,
                "mean_food_eaten": float(np.mean(self.food_eaten)) if self.food_eaten else 0.0,
                "pct_nights_survived": float(np.mean(self.nights_survived)) if self.nights_survived else 0.0,
                "mean_stage_completion": float(np.mean(self.stage_completions)) if self.stage_completions else 0.0,
                "max_stage_completion": float(np.max(self.stage_completions)) if self.stage_completions else 0.0,
            },
        }


def benchmark_stage1_curriculum(
    num_agents: int = 50,
    episodes_per_agent: int = 20,
    steps_per_episode: int = 1200,
) -> Stage1ProgressMetrics:
    """Benchmark Stage 1 curriculum with progress tracking for multiple agents.

    Runs the Stage 1 (Basic Survival) curriculum with `num_agents` parallel
    environments, tracking per-agent SpeedrunProgress and curriculum advancement
    metrics.

    Args:
        num_agents: Number of parallel environments (agents).
        episodes_per_agent: Episodes to run per agent.
        steps_per_episode: Maximum steps per episode before truncation.

    Returns:
        Stage1ProgressMetrics with full per-agent progress and aggregate stats.
    """
    print("\n" + "=" * 70)
    print(f"BENCHMARK: Stage 1 Curriculum ({num_agents} agents)")
    print("=" * 70)

    if not HAS_SIMULATOR:
        print("  ERROR: minecraft_sim module not available.")
        return Stage1ProgressMetrics(num_agents=num_agents)

    if not HAS_CURRICULUM:
        print("  ERROR: curriculum/progression modules not available.")
        return Stage1ProgressMetrics(num_agents=num_agents)

    metrics = Stage1ProgressMetrics(num_agents=num_agents)

    # Configure curriculum to stay on Stage 1 only
    curriculum_config = CurriculumConfig(
        min_episodes_per_stage=10,
        advancement_threshold=0.7,
        allow_regression=False,
        max_episodes_per_stage=episodes_per_agent + 1,  # Prevent forced advancement
        window_size=50,
        min_stage=1,
        max_stage=1,  # Lock to Stage 1 only
    )
    curriculum = AutoCurriculumManager(num_envs=num_agents, config=curriculum_config)

    # Per-agent progress trackers
    trackers = [ProgressTracker() for _ in range(num_agents)]

    # Create backend
    try:
        backend = VulkanBackend(num_envs=num_agents)
        backend.reset()
    except Exception as e:
        print(f"  ERROR creating backend: {e}")
        return metrics

    print(f"  Backend initialized: {num_agents} envs")
    print(f"  Running {episodes_per_agent} episodes/agent, {steps_per_episode} max steps/ep")
    print()

    total_steps = 0
    total_episodes = 0
    episode_rewards = np.zeros(num_agents, dtype=np.float64)
    episode_lengths = np.zeros(num_agents, dtype=np.int32)
    all_episode_rewards: list[float] = []

    gc.collect()
    start_time = time.perf_counter()

    episodes_completed = np.zeros(num_agents, dtype=np.int32)
    active_mask = np.ones(num_agents, dtype=bool)  # All agents active initially

    while np.any(active_mask):
        # Take actions (random policy for benchmark purposes)
        actions = np.random.randint(0, 17, size=num_agents, dtype=np.int32)
        # Zero out actions for inactive agents
        actions[~active_mask] = 0

        backend.step(actions)
        total_steps += int(np.sum(active_mask))

        # Simulate observation dict per active agent for progress tracking
        for i in range(num_agents):
            if not active_mask[i]:
                continue

            episode_lengths[i] += 1

            # Generate simulated progress observations based on step count
            # (In real training, these come from the backend observations)
            step_in_ep = episode_lengths[i]
            obs: dict[str, Any] = {
                "tick_number": trackers[i].progress.total_ticks + 1,
                "player": {
                    "health": max(0.0, 20.0 - np.random.exponential(0.1)),
                    "dimension": 0,
                },
                "inventory": {},
            }

            # Simulate Stage 1 resource gathering based on step progression
            if step_in_ep > 50:
                obs["inventory"]["wood"] = min(
                    trackers[i].progress.wood_collected + int(np.random.poisson(0.3)),
                    64,
                )
            if step_in_ep > 150:
                obs["inventory"]["cobblestone"] = min(
                    trackers[i].progress.stone_collected + int(np.random.poisson(0.2)),
                    128,
                )
            if step_in_ep > 300 and np.random.random() < 0.02:
                obs["inventory"]["food_eaten"] = trackers[i].progress.food_eaten + 1

            # Update progress tracker
            reward_signals = trackers[i].update_from_observation(obs)
            step_reward = sum(reward_signals.values())
            episode_rewards[i] += step_reward

            # Check episode termination
            done = step_in_ep >= steps_per_episode
            success = trackers[i].progress.is_stage_complete(SpeedrunStage.SURVIVAL)

            if done or success:
                # Record in curriculum manager
                curriculum.update(
                    env_id=i,
                    success=success,
                    episode_length=int(episode_lengths[i]),
                    episode_reward=float(episode_rewards[i]),
                )

                all_episode_rewards.append(float(episode_rewards[i]))
                episodes_completed[i] += 1
                total_episodes += 1

                # Log periodic progress
                if total_episodes % (num_agents * 5) == 0:
                    rate = curriculum.get_success_rate(1)
                    print(
                        f"  Episodes: {total_episodes:>6d} | "
                        f"Success rate: {rate:.1%} | "
                        f"Steps: {total_steps:>8d}"
                    )

                # Reset episode state for this agent
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0
                trackers[i].reset()

                # Check if agent has completed all episodes
                if episodes_completed[i] >= episodes_per_agent:
                    active_mask[i] = False

    elapsed = time.perf_counter() - start_time

    # Collect per-agent progress metrics
    for i in range(num_agents):
        p = trackers[i].progress
        metrics.wood_collected.append(p.wood_collected)
        metrics.stone_collected.append(p.stone_collected)
        metrics.zombies_killed.append(p.zombies_killed)
        metrics.food_eaten.append(p.food_eaten)
        metrics.nights_survived.append(p.first_night_survived)
        metrics.stage_completions.append(p.get_stage_completion(SpeedrunStage.SURVIVAL))

    # Curriculum stats
    curriculum_stats = curriculum.get_stats()
    metrics.total_episodes = total_episodes
    metrics.total_steps = total_steps
    metrics.elapsed_seconds = elapsed
    metrics.steps_per_second = total_steps / elapsed if elapsed > 0 else 0.0
    metrics.episodes_per_second = total_episodes / elapsed if elapsed > 0 else 0.0
    metrics.success_rate = curriculum.get_success_rate(1)
    metrics.advancements = curriculum_stats["advancements"]
    metrics.mean_episode_reward = float(np.mean(all_episode_rewards)) if all_episode_rewards else 0.0
    metrics.best_episode_reward = float(np.max(all_episode_rewards)) if all_episode_rewards else 0.0

    # Print summary
    print()
    print("  " + "-" * 60)
    print(f"  STAGE 1 RESULTS ({num_agents} agents)")
    print("  " + "-" * 60)
    print(f"    Total episodes:      {total_episodes:>8d}")
    print(f"    Total steps:         {total_steps:>8d}")
    print(f"    Elapsed:             {elapsed:>8.2f}s")
    print(f"    Steps/sec:           {metrics.steps_per_second:>8.0f}")
    print(f"    Episodes/sec:        {metrics.episodes_per_second:>8.1f}")
    print()
    print(f"    Success rate:        {metrics.success_rate:>8.1%}")
    print(f"    Mean ep reward:      {metrics.mean_episode_reward:>8.2f}")
    print(f"    Best ep reward:      {metrics.best_episode_reward:>8.2f}")
    print()
    agg = metrics.to_dict()["progress_aggregates"]
    print(f"    Mean wood:           {agg['mean_wood']:>8.1f}")
    print(f"    Mean stone:          {agg['mean_stone']:>8.1f}")
    print(f"    Mean zombies killed: {agg['mean_zombies_killed']:>8.1f}")
    print(f"    Mean food eaten:     {agg['mean_food_eaten']:>8.1f}")
    print(f"    Nights survived:     {agg['pct_nights_survived']:>8.1%}")
    print(f"    Mean completion:     {agg['mean_stage_completion']:>8.1%}")
    print(f"    Max completion:      {agg['max_stage_completion']:>8.1%}")

    del backend
    gc.collect()

    return metrics


# -----------------------------------------------------------------------------
# Stage 3 Nether Benchmark (50 agents)
# -----------------------------------------------------------------------------


@dataclass
class Stage3ProgressMetrics:
    """Aggregate progress metrics for Stage 3 (Nether) curriculum across all agents.

    Tracks blaze rod acquisition rate and lava hazard avoidance as primary
    performance indicators for Nether navigation competence.
    """

    num_agents: int = 50
    total_episodes: int = 0
    total_steps: int = 0
    elapsed_seconds: float = 0.0

    # Per-agent Nether progress
    blaze_rods_collected: list[int] = field(default_factory=list)
    blazes_killed: list[int] = field(default_factory=list)
    lava_deaths: list[int] = field(default_factory=list)
    lava_escapes: list[int] = field(default_factory=list)
    fortress_found: list[bool] = field(default_factory=list)
    portal_built: list[bool] = field(default_factory=list)
    entered_nether: list[bool] = field(default_factory=list)
    stage_completions: list[float] = field(default_factory=list)

    # Key metrics
    blaze_rods_per_minute: float = 0.0
    lava_avoidance_rate: float = 0.0

    # Curriculum manager stats
    success_rate: float = 0.0
    advancements: int = 0
    mean_episode_reward: float = 0.0
    best_episode_reward: float = 0.0

    # Throughput
    steps_per_second: float = 0.0
    episodes_per_second: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_agents": self.num_agents,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "elapsed_seconds": self.elapsed_seconds,
            "steps_per_second": self.steps_per_second,
            "episodes_per_second": self.episodes_per_second,
            "key_metrics": {
                "blaze_rods_per_minute": self.blaze_rods_per_minute,
                "lava_avoidance_rate": self.lava_avoidance_rate,
            },
            "curriculum": {
                "success_rate": self.success_rate,
                "advancements": self.advancements,
                "mean_episode_reward": self.mean_episode_reward,
                "best_episode_reward": self.best_episode_reward,
            },
            "progress_per_agent": {
                "blaze_rods_collected": self.blaze_rods_collected,
                "blazes_killed": self.blazes_killed,
                "lava_deaths": self.lava_deaths,
                "lava_escapes": self.lava_escapes,
                "fortress_found": self.fortress_found,
                "portal_built": self.portal_built,
                "entered_nether": self.entered_nether,
                "stage_completions": self.stage_completions,
            },
            "progress_aggregates": {
                "mean_blaze_rods": (
                    float(np.mean(self.blaze_rods_collected)) if self.blaze_rods_collected else 0.0
                ),
                "max_blaze_rods": (
                    int(np.max(self.blaze_rods_collected)) if self.blaze_rods_collected else 0
                ),
                "mean_blazes_killed": (
                    float(np.mean(self.blazes_killed)) if self.blazes_killed else 0.0
                ),
                "mean_lava_deaths": (
                    float(np.mean(self.lava_deaths)) if self.lava_deaths else 0.0
                ),
                "mean_lava_escapes": (
                    float(np.mean(self.lava_escapes)) if self.lava_escapes else 0.0
                ),
                "pct_fortress_found": (
                    float(np.mean(self.fortress_found)) if self.fortress_found else 0.0
                ),
                "pct_entered_nether": (
                    float(np.mean(self.entered_nether)) if self.entered_nether else 0.0
                ),
                "mean_stage_completion": (
                    float(np.mean(self.stage_completions)) if self.stage_completions else 0.0
                ),
                "max_stage_completion": (
                    float(np.max(self.stage_completions)) if self.stage_completions else 0.0
                ),
            },
        }


def benchmark_stage3_nether(
    num_agents: int = 50,
    episodes_per_agent: int = 20,
    steps_per_episode: int = 2400,
) -> Stage3ProgressMetrics:
    """Benchmark Stage 3 (Nether) curriculum measuring blaze rod rate and lava avoidance.

    Runs the Stage 3 (Nether Navigation) curriculum with `num_agents` parallel
    environments. Primary metrics:
    - Blaze rods per minute: acquisition efficiency in the fortress
    - Lava avoidance rate: fraction of lava encounters survived without death

    Args:
        num_agents: Number of parallel environments (agents).
        episodes_per_agent: Episodes to run per agent.
        steps_per_episode: Maximum steps per episode (longer than Stage 1 due to
            Nether traversal complexity).

    Returns:
        Stage3ProgressMetrics with blaze rod rate and lava hazard stats.
    """
    print("\n" + "=" * 70)
    print(f"BENCHMARK: Stage 3 Nether ({num_agents} agents)")
    print("=" * 70)

    if not HAS_SIMULATOR:
        print("  ERROR: minecraft_sim module not available.")
        return Stage3ProgressMetrics(num_agents=num_agents)

    if not HAS_CURRICULUM:
        print("  ERROR: curriculum/progression modules not available.")
        return Stage3ProgressMetrics(num_agents=num_agents)

    metrics = Stage3ProgressMetrics(num_agents=num_agents)

    # Configure curriculum locked to Stage 3 (Nether)
    curriculum_config = CurriculumConfig(
        min_episodes_per_stage=10,
        advancement_threshold=0.7,
        allow_regression=False,
        max_episodes_per_stage=episodes_per_agent + 1,
        window_size=50,
        min_stage=3,
        max_stage=3,  # Lock to Stage 3 only
    )
    curriculum = AutoCurriculumManager(num_envs=num_agents, config=curriculum_config)

    # Per-agent progress trackers
    trackers = [ProgressTracker() for _ in range(num_agents)]

    # Per-agent lava encounter tracking (beyond what ProgressTracker stores)
    lava_encounters = np.zeros(num_agents, dtype=np.int32)
    lava_deaths_count = np.zeros(num_agents, dtype=np.int32)
    lava_escapes_count = np.zeros(num_agents, dtype=np.int32)
    was_in_lava = np.zeros(num_agents, dtype=bool)

    # Create backend
    try:
        backend = VulkanBackend(num_envs=num_agents)
        backend.reset()
    except Exception as e:
        print(f"  ERROR creating backend: {e}")
        return metrics

    print(f"  Backend initialized: {num_agents} envs")
    print(f"  Running {episodes_per_agent} episodes/agent, {steps_per_episode} max steps/ep")
    print("  Key metrics: blaze rods/min, lava avoidance rate")
    print()

    total_steps = 0
    total_episodes = 0
    total_blaze_rods_all = 0
    episode_rewards = np.zeros(num_agents, dtype=np.float64)
    episode_lengths = np.zeros(num_agents, dtype=np.int32)
    all_episode_rewards: list[float] = []

    gc.collect()
    start_time = time.perf_counter()

    episodes_completed = np.zeros(num_agents, dtype=np.int32)
    active_mask = np.ones(num_agents, dtype=bool)

    while np.any(active_mask):
        actions = np.random.randint(0, 17, size=num_agents, dtype=np.int32)
        actions[~active_mask] = 0

        backend.step(actions)
        total_steps += int(np.sum(active_mask))

        for i in range(num_agents):
            if not active_mask[i]:
                continue

            episode_lengths[i] += 1
            step_in_ep = episode_lengths[i]

            # Simulate Nether-stage observations
            health = max(0.0, 20.0 - np.random.exponential(0.3))
            obs: dict[str, Any] = {
                "tick_number": trackers[i].progress.total_ticks + 1,
                "player": {
                    "health": health,
                    "dimension": 0 if step_in_ep < 100 else -1,  # Nether dimension
                    "in_lava": False,
                    "on_fire": False,
                },
                "inventory": {},
            }

            # Portal construction and Nether entry
            if step_in_ep >= 50:
                trackers[i].progress.portal_built = True
                obs["inventory"]["obsidian"] = 10
            if step_in_ep >= 100:
                trackers[i].progress.entered_nether = True

            # Lava hazard encounters in the Nether (~1.5% per step)
            in_lava_now = False
            if step_in_ep > 100 and np.random.random() < 0.015:
                in_lava_now = True
                obs["player"]["in_lava"] = True
                obs["player"]["on_fire"] = True
                lava_encounters[i] += 1
                health = max(0.0, health - np.random.uniform(2.0, 8.0))
                obs["player"]["health"] = health

            # Lava state transitions
            if was_in_lava[i] and not in_lava_now:
                lava_escapes_count[i] += 1
            if in_lava_now and health <= 0:
                lava_deaths_count[i] += 1
            was_in_lava[i] = in_lava_now

            # Fortress finding and blaze rod collection
            if step_in_ep > 400:
                trackers[i].progress.fortress_found = True
            if step_in_ep > 500 and np.random.random() < 0.008:
                # ~0.8% chance per step of obtaining a blaze rod
                current_rods = trackers[i].progress.blaze_rods
                trackers[i].progress.blaze_rods = current_rods + 1
                trackers[i].progress.blazes_killed += 1
                obs["inventory"]["blaze_rod"] = trackers[i].progress.blaze_rods

            # Update progress tracker
            reward_signals = trackers[i].update_from_observation(obs)
            step_reward = sum(reward_signals.values())
            episode_rewards[i] += step_reward

            # Episode termination
            done = step_in_ep >= steps_per_episode
            health_dead = health <= 0
            success = trackers[i].progress.is_stage_complete(SpeedrunStage.NETHER)

            if done or health_dead or success:
                curriculum.update(
                    env_id=i,
                    success=success,
                    episode_length=int(episode_lengths[i]),
                    episode_reward=float(episode_rewards[i]),
                )

                total_blaze_rods_all += trackers[i].progress.blaze_rods
                all_episode_rewards.append(float(episode_rewards[i]))
                episodes_completed[i] += 1
                total_episodes += 1

                if total_episodes % (num_agents * 5) == 0:
                    rate = curriculum.get_success_rate(3)
                    elapsed_so_far = time.perf_counter() - start_time
                    rods_per_min = (
                        (total_blaze_rods_all / (elapsed_so_far / 60.0))
                        if elapsed_so_far > 0
                        else 0.0
                    )
                    print(
                        f"  Episodes: {total_episodes:>6d} | "
                        f"Success: {rate:.1%} | "
                        f"Rods/min: {rods_per_min:.1f} | "
                        f"Steps: {total_steps:>8d}"
                    )

                # Reset episode state
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0
                was_in_lava[i] = False
                trackers[i].reset()

                if episodes_completed[i] >= episodes_per_agent:
                    active_mask[i] = False

    elapsed = time.perf_counter() - start_time

    # Collect per-agent progress metrics
    for i in range(num_agents):
        p = trackers[i].progress
        metrics.blaze_rods_collected.append(p.blaze_rods)
        metrics.blazes_killed.append(p.blazes_killed)
        metrics.lava_deaths.append(int(lava_deaths_count[i]))
        metrics.lava_escapes.append(int(lava_escapes_count[i]))
        metrics.fortress_found.append(p.fortress_found)
        metrics.portal_built.append(p.portal_built)
        metrics.entered_nether.append(p.entered_nether)
        metrics.stage_completions.append(p.get_stage_completion(SpeedrunStage.NETHER))

    # Compute key metrics: blaze rods per minute
    elapsed_minutes = elapsed / 60.0
    metrics.blaze_rods_per_minute = (
        total_blaze_rods_all / elapsed_minutes if elapsed_minutes > 0 else 0.0
    )

    # Compute lava avoidance rate: 1 - (deaths / encounters)
    total_lava_encounters = int(np.sum(lava_encounters))
    total_lava_deaths = int(np.sum(lava_deaths_count))
    metrics.lava_avoidance_rate = (
        1.0 - (total_lava_deaths / total_lava_encounters)
        if total_lava_encounters > 0
        else 1.0
    )

    # Curriculum stats
    curriculum_stats = curriculum.get_stats()
    metrics.total_episodes = total_episodes
    metrics.total_steps = total_steps
    metrics.elapsed_seconds = elapsed
    metrics.steps_per_second = total_steps / elapsed if elapsed > 0 else 0.0
    metrics.episodes_per_second = total_episodes / elapsed if elapsed > 0 else 0.0
    metrics.success_rate = curriculum.get_success_rate(3)
    metrics.advancements = curriculum_stats["advancements"]
    metrics.mean_episode_reward = float(np.mean(all_episode_rewards)) if all_episode_rewards else 0.0
    metrics.best_episode_reward = float(np.max(all_episode_rewards)) if all_episode_rewards else 0.0

    # Print summary
    print()
    print("  " + "-" * 60)
    print(f"  STAGE 3 NETHER RESULTS ({num_agents} agents)")
    print("  " + "-" * 60)
    print(f"    Total episodes:        {total_episodes:>8d}")
    print(f"    Total steps:           {total_steps:>8d}")
    print(f"    Elapsed:               {elapsed:>8.2f}s")
    print(f"    Steps/sec:             {metrics.steps_per_second:>8.0f}")
    print(f"    Episodes/sec:          {metrics.episodes_per_second:>8.1f}")
    print()
    print(f"    Blaze rods/min:        {metrics.blaze_rods_per_minute:>8.2f}")
    print(f"    Lava avoidance rate:   {metrics.lava_avoidance_rate:>8.1%}")
    print(f"    Total lava encounters: {total_lava_encounters:>8d}")
    print(f"    Total lava deaths:     {total_lava_deaths:>8d}")
    print()
    print(f"    Success rate:          {metrics.success_rate:>8.1%}")
    print(f"    Mean ep reward:        {metrics.mean_episode_reward:>8.2f}")
    print(f"    Best ep reward:        {metrics.best_episode_reward:>8.2f}")
    print()
    agg = metrics.to_dict()["progress_aggregates"]
    print(f"    Mean blaze rods:       {agg['mean_blaze_rods']:>8.1f}")
    print(f"    Max blaze rods:        {agg['max_blaze_rods']:>8d}")
    print(f"    Mean blazes killed:    {agg['mean_blazes_killed']:>8.1f}")
    print(f"    Mean lava deaths:      {agg['mean_lava_deaths']:>8.1f}")
    print(f"    Mean lava escapes:     {agg['mean_lava_escapes']:>8.1f}")
    print(f"    Fortress found:        {agg['pct_fortress_found']:>8.1%}")
    print(f"    Entered nether:        {agg['pct_entered_nether']:>8.1%}")
    print(f"    Mean completion:       {agg['mean_stage_completion']:>8.1%}")
    print(f"    Max completion:        {agg['max_stage_completion']:>8.1%}")

    del backend
    gc.collect()

    return metrics


# -----------------------------------------------------------------------------
# Stronghold Triangulation Benchmark (50 agents)
# -----------------------------------------------------------------------------


@dataclass
class StrongholdBenchmarkMetrics:
    """Metrics for stronghold triangulation, eye placement, and portal activation.

    Measures three phases of Stage 5 (Stronghold Finding) across parallel agents:
    1. Triangulation time: How quickly agents compute stronghold position from eye throws.
    2. Eye placement velocity: Rate at which eyes are placed in portal frames.
    3. Portal activation latency: Ticks from first eye placed to portal activation.
    """

    num_agents: int = 50
    total_episodes: int = 0
    total_steps: int = 0
    elapsed_seconds: float = 0.0

    # Triangulation timing (per-agent, in milliseconds)
    triangulation_times_ms: list[float] = field(default_factory=list)
    triangulation_throws_needed: list[int] = field(default_factory=list)
    triangulation_errors_blocks: list[float] = field(default_factory=list)

    # Eye placement velocity (eyes placed per 100 ticks)
    eye_placement_rates: list[float] = field(default_factory=list)
    eyes_placed_total: list[int] = field(default_factory=list)
    eye_placement_ticks: list[int] = field(default_factory=list)

    # Portal activation latency (ticks from first eye placed to portal active)
    portal_activation_latencies: list[int] = field(default_factory=list)
    portals_activated: int = 0

    # Throughput
    steps_per_second: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        tri_times = np.array(self.triangulation_times_ms) if self.triangulation_times_ms else np.array([0.0])
        eye_rates = np.array(self.eye_placement_rates) if self.eye_placement_rates else np.array([0.0])
        portal_lats = np.array(self.portal_activation_latencies) if self.portal_activation_latencies else np.array([0])

        return {
            "num_agents": self.num_agents,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "elapsed_seconds": self.elapsed_seconds,
            "steps_per_second": self.steps_per_second,
            "triangulation": {
                "mean_time_ms": float(np.mean(tri_times)),
                "p50_time_ms": float(np.percentile(tri_times, 50)),
                "p95_time_ms": float(np.percentile(tri_times, 95)),
                "p99_time_ms": float(np.percentile(tri_times, 99)),
                "mean_throws_needed": float(np.mean(self.triangulation_throws_needed)) if self.triangulation_throws_needed else 0.0,
                "mean_error_blocks": float(np.mean(self.triangulation_errors_blocks)) if self.triangulation_errors_blocks else 0.0,
                "all_times_ms": self.triangulation_times_ms,
            },
            "eye_placement": {
                "mean_rate_per_100_ticks": float(np.mean(eye_rates)),
                "p50_rate": float(np.percentile(eye_rates, 50)),
                "p95_rate": float(np.percentile(eye_rates, 95)),
                "mean_total_placed": float(np.mean(self.eyes_placed_total)) if self.eyes_placed_total else 0.0,
                "mean_ticks_to_place_all": float(np.mean(self.eye_placement_ticks)) if self.eye_placement_ticks else 0.0,
            },
            "portal_activation": {
                "mean_latency_ticks": float(np.mean(portal_lats)),
                "p50_latency_ticks": float(np.percentile(portal_lats, 50)),
                "p95_latency_ticks": float(np.percentile(portal_lats, 95)),
                "p99_latency_ticks": float(np.percentile(portal_lats, 99)),
                "portals_activated": self.portals_activated,
                "activation_rate": self.portals_activated / self.num_agents if self.num_agents > 0 else 0.0,
            },
        }


def benchmark_stronghold(
    num_agents: int = 50,
    episodes_per_agent: int = 10,
    steps_per_episode: int = 2400,
) -> StrongholdBenchmarkMetrics:
    """Benchmark Stage 5 stronghold finding with triangulation, eye placement, and portal activation.

    Simulates the stronghold-finding phase for `num_agents` parallel environments.
    Each agent performs:
    1. Eye of Ender throws to triangulate the stronghold position.
    2. Navigation to the stronghold and eye placement in the portal frame.
    3. Portal activation once all 12 eyes are placed.

    The benchmark measures compute performance of triangulation, the rate of eye
    placement across the agent population, and the latency from first eye placed
    to full portal activation.

    Args:
        num_agents: Number of parallel environments (agents).
        episodes_per_agent: Episodes to run per agent.
        steps_per_episode: Maximum steps per episode before truncation.

    Returns:
        StrongholdBenchmarkMetrics with per-agent timing and aggregate stats.
    """
    print("\n" + "=" * 70)
    print(f"BENCHMARK: Stronghold Triangulation & Portal ({num_agents} agents)")
    print("=" * 70)

    if not HAS_SIMULATOR:
        print("  ERROR: minecraft_sim module not available.")
        return StrongholdBenchmarkMetrics(num_agents=num_agents)

    if not HAS_TRIANGULATION:
        print("  ERROR: triangulation module not available.")
        return StrongholdBenchmarkMetrics(num_agents=num_agents)

    if not HAS_CURRICULUM:
        print("  ERROR: curriculum/progression modules not available.")
        return StrongholdBenchmarkMetrics(num_agents=num_agents)

    metrics = StrongholdBenchmarkMetrics(num_agents=num_agents)

    # Configure curriculum locked to Stage 5 (Stronghold)
    curriculum_config = CurriculumConfig(
        min_episodes_per_stage=5,
        advancement_threshold=0.7,
        allow_regression=False,
        max_episodes_per_stage=episodes_per_agent + 1,
        window_size=50,
        min_stage=5,
        max_stage=5,
    )
    curriculum = AutoCurriculumManager(num_envs=num_agents, config=curriculum_config)

    # Per-agent state
    trackers = [ProgressTracker() for _ in range(num_agents)]
    triangulation_states = [TriangulationState() for _ in range(num_agents)]

    # Simulated stronghold positions for each agent (fixed per episode)
    rng = np.random.default_rng(42)
    stronghold_positions = [
        (rng.uniform(800, 2400) * rng.choice([-1, 1]), rng.uniform(800, 2400) * rng.choice([-1, 1]))
        for _ in range(num_agents)
    ]

    # Per-agent episode tracking
    first_eye_tick: list[int] = [0] * num_agents  # Tick when first eye was placed
    eye_place_start_tick: list[int] = [0] * num_agents
    triangulation_done: list[bool] = [False] * num_agents
    agent_positions: list[tuple[float, float]] = [(0.0, 0.0)] * num_agents

    # Create backend
    try:
        backend = VulkanBackend(num_envs=num_agents)
        backend.reset()
    except Exception as e:

import logging

logger = logging.getLogger(__name__)

        print(f"  ERROR creating backend: {e}")
        return metrics

    print(f"  Backend initialized: {num_agents} envs")
    print(f"  Running {episodes_per_agent} episodes/agent, {steps_per_episode} max steps/ep")
    print()

    total_steps = 0
    total_episodes = 0
    episode_lengths = np.zeros(num_agents, dtype=np.int32)
    episodes_completed = np.zeros(num_agents, dtype=np.int32)
    active_mask = np.ones(num_agents, dtype=bool)

    gc.collect()
    start_time = time.perf_counter()

    while np.any(active_mask):
        actions = np.random.randint(0, 17, size=num_agents, dtype=np.int32)
        actions[~active_mask] = 0

        backend.step(actions)
        total_steps += int(np.sum(active_mask))

        for i in range(num_agents):
            if not active_mask[i]:
                continue

            episode_lengths[i] += 1
            step_in_ep = int(episode_lengths[i])
            p = trackers[i].progress
            tri = triangulation_states[i]
            sx, sz = stronghold_positions[i]

            # Phase 1: Eye throws for triangulation (steps 10-100)
            # Simulate throwing eyes at intervals to build triangulation data
            if not triangulation_done[i] and step_in_ep >= 10:
                # Throw every ~30 ticks from different positions, simulate travel
                if step_in_ep % 30 == 0 and tri.num_throws < 4:
                    # Agent moves between throws
                    offset_x = rng.uniform(-200, 200)
                    offset_z = rng.uniform(-200, 200)
                    throw_pos = (offset_x, offset_z)
                    agent_positions[i] = throw_pos

                    # Direction toward stronghold (with noise simulating eye flight)
                    dx = sx - throw_pos[0]
                    dz = sz - throw_pos[1]
                    dist = math.sqrt(dx * dx + dz * dz)
                    if dist > 0:
                        noise_angle = rng.normal(0, 0.03)  # ~1.7 degree noise
                        cos_n = math.cos(noise_angle)
                        sin_n = math.sin(noise_angle)
                        ndx = (dx / dist) * cos_n - (dz / dist) * sin_n
                        ndz = (dx / dist) * sin_n + (dz / dist) * cos_n
                        tri.add_throw(throw_pos, (ndx, ndz))

                        p.eyes_used = tri.num_throws
                        p.eyes_crafted = max(p.eyes_crafted, tri.num_throws)

                # Check if triangulation is ready
                if tri.is_complete and not triangulation_done[i]:
                    t_start = time.perf_counter()
                    estimated_pos = tri.estimated_position
                    t_elapsed = (time.perf_counter() - t_start) * 1000.0

                    metrics.triangulation_times_ms.append(t_elapsed)
                    metrics.triangulation_throws_needed.append(tri.num_throws)

                    if estimated_pos is not None:
                        error = math.sqrt(
                            (estimated_pos[0] - sx) ** 2 + (estimated_pos[1] - sz) ** 2
                        )
                        metrics.triangulation_errors_blocks.append(error)

                        # Mark stronghold found if close enough
                        if error < 100:
                            p.stronghold_found = True
                            p.stronghold_distance = error

                    triangulation_done[i] = True

            # Phase 2: Navigate and place eyes (steps 100-2000)
            if triangulation_done[i] and step_in_ep >= 100:
                # Simulate approaching stronghold and placing eyes
                ticks_since_found = step_in_ep - 100

                if p.eyes_placed == 0 and ticks_since_found > 50:
                    # First eye placement
                    p.portal_room_found = True
                    first_eye_tick[i] = step_in_ep
                    eye_place_start_tick[i] = step_in_ep

                if p.portal_room_found and p.eyes_placed < 12:
                    # Place eyes at a rate influenced by randomness
                    # Average ~1 eye per 20-40 ticks once portal room is found
                    ticks_in_room = step_in_ep - first_eye_tick[i]
                    expected_eyes = min(12, int(ticks_in_room / rng.uniform(20, 40)))
                    if expected_eyes > p.eyes_placed:
                        p.eyes_placed = expected_eyes
                        p.eyes_crafted = max(p.eyes_crafted, 12)

            # Phase 3: Portal activation
            if p.eyes_placed >= 12 and not p.portal_activated:
                p.portal_activated = True
                activation_latency = step_in_ep - first_eye_tick[i]
                metrics.portal_activation_latencies.append(activation_latency)
                metrics.portals_activated += 1

            # Episode termination
            done = step_in_ep >= steps_per_episode
            success = p.portal_activated

            if done or success:
                # Record eye placement metrics for this episode
                if p.eyes_placed > 0 and first_eye_tick[i] > 0:
                    placement_ticks = step_in_ep - first_eye_tick[i]
                    if placement_ticks > 0:
                        rate = (p.eyes_placed / placement_ticks) * 100.0
                        metrics.eye_placement_rates.append(rate)
                        metrics.eyes_placed_total.append(p.eyes_placed)
                        metrics.eye_placement_ticks.append(placement_ticks)

                curriculum.update(
                    env_id=i,
                    success=success,
                    episode_length=step_in_ep,
                    episode_reward=float(p.eyes_placed + int(p.portal_activated) * 20),
                )

                episodes_completed[i] += 1
                total_episodes += 1

                if total_episodes % (num_agents * 2) == 0:
                    rate = curriculum.get_success_rate(5)
                    print(
                        f"  Episodes: {total_episodes:>5d} | "
                        f"Portals: {metrics.portals_activated:>4d} | "
                        f"Success: {rate:.1%} | "
                        f"Steps: {total_steps:>8d}"
                    )

                # Reset episode state
                episode_lengths[i] = 0
                trackers[i].reset()
                triangulation_states[i].clear()
                triangulation_done[i] = False
                first_eye_tick[i] = 0
                eye_place_start_tick[i] = 0
                # New stronghold position for next episode
                stronghold_positions[i] = (
                    rng.uniform(800, 2400) * rng.choice([-1, 1]),
                    rng.uniform(800, 2400) * rng.choice([-1, 1]),
                )

                if episodes_completed[i] >= episodes_per_agent:
                    active_mask[i] = False

    elapsed = time.perf_counter() - start_time
    metrics.total_episodes = total_episodes
    metrics.total_steps = total_steps
    metrics.elapsed_seconds = elapsed
    metrics.steps_per_second = total_steps / elapsed if elapsed > 0 else 0.0

    # Print summary
    result = metrics.to_dict()
    tri_stats = result["triangulation"]
    eye_stats = result["eye_placement"]
    portal_stats = result["portal_activation"]

    print()
    print("  " + "-" * 60)
    print(f"  STRONGHOLD BENCHMARK RESULTS ({num_agents} agents)")
    print("  " + "-" * 60)
    print(f"    Total episodes:         {total_episodes:>8d}")
    print(f"    Total steps:            {total_steps:>8d}")
    print(f"    Elapsed:                {elapsed:>8.2f}s")
    print(f"    Steps/sec:              {metrics.steps_per_second:>8.0f}")
    print()
    print("  Triangulation:")
    print(f"    Mean time:              {tri_stats['mean_time_ms']:>8.4f}ms")
    print(f"    P95 time:               {tri_stats['p95_time_ms']:>8.4f}ms")
    print(f"    P99 time:               {tri_stats['p99_time_ms']:>8.4f}ms")
    print(f"    Mean throws needed:     {tri_stats['mean_throws_needed']:>8.1f}")
    print(f"    Mean error (blocks):    {tri_stats['mean_error_blocks']:>8.1f}")
    print()
    print("  Eye Placement:")
    print(f"    Mean rate (/100 ticks): {eye_stats['mean_rate_per_100_ticks']:>8.2f}")
    print(f"    P95 rate:               {eye_stats['p95_rate']:>8.2f}")
    print(f"    Mean total placed:      {eye_stats['mean_total_placed']:>8.1f}")
    print(f"    Mean ticks to fill:     {eye_stats['mean_ticks_to_place_all']:>8.0f}")
    print()
    print("  Portal Activation:")
    print(f"    Mean latency (ticks):   {portal_stats['mean_latency_ticks']:>8.0f}")
    print(f"    P50 latency:            {portal_stats['p50_latency_ticks']:>8.0f}")
    print(f"    P95 latency:            {portal_stats['p95_latency_ticks']:>8.0f}")
    print(f"    P99 latency:            {portal_stats['p99_latency_ticks']:>8.0f}")
    print(f"    Portals activated:      {portal_stats['portals_activated']:>8d}/{num_agents * episodes_per_agent}")
    print(f"    Activation rate:        {portal_stats['activation_rate']:>8.1%}")

    del backend
    gc.collect()

    return metrics


# -----------------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------------


def generate_report(results: BenchmarkResults, output_dir: Path) -> None:
    """Generate benchmark report with graphs and summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw JSON results
    json_path = output_dir / f"benchmark_results_{results.timestamp.replace(':', '-')}.json"
    with open(json_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\nResults saved to: {json_path}")

    if not HAS_MATPLOTLIB:
        print("Note: matplotlib not available, skipping graphs")
        return

    # Generate graphs
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Minecraft Simulator Performance Benchmarks", fontsize=14, fontweight="bold")

    # 1. Throughput vs num_envs
    ax = axes[0, 0]
    if results.throughput.get("env_counts"):
        env_counts = results.throughput["env_counts"]
        sps = [s / 1e6 for s in results.throughput["steps_per_second"]]  # Convert to millions
        ax.plot(env_counts, sps, "b-o", linewidth=2, markersize=8)
        ax.axhline(y=60 / 1e6, color="r", linestyle="--", label="MineRL baseline")
        ax.set_xlabel("Number of Environments")
        ax.set_ylabel("Steps per Second (millions)")
        ax.set_title("Throughput Scaling")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 2. Efficiency (steps/env/s) vs num_envs
    ax = axes[0, 1]
    if results.throughput.get("env_counts"):
        env_counts = results.throughput["env_counts"]
        sps_per_env = results.throughput["steps_per_env_per_second"]
        ax.plot(env_counts, sps_per_env, "g-o", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Environments")
        ax.set_ylabel("Steps per Environment per Second")
        ax.set_title("Per-Environment Efficiency")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

    # 3. Memory scaling
    ax = axes[1, 0]
    if results.memory_scaling.get("env_counts"):
        env_counts = results.memory_scaling["env_counts"]
        memory = results.memory_scaling["memory_mb"]
        ax.plot(env_counts, memory, "m-o", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Environments")
        ax.set_ylabel("Memory Usage (MB)")
        ax.set_title("Memory Scaling")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

    # 4. MineRL comparison bar chart
    ax = axes[1, 1]
    if results.minerl_comparison.get("our_results"):
        our = results.minerl_comparison["our_results"]
        minerl = results.minerl_comparison["minerl_baseline"]

        categories = ["Single Env", "Peak (4096 envs)"]
        minerl_values = [minerl["single_env_sps"], minerl["single_env_sps"]]
        our_values = [
            our.get("single_env_sps", 0),
            our.get("peak_sps", 0),
        ]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            [v / 1000 for v in minerl_values],
            width,
            label="MineRL",
            color="red",
            alpha=0.7,
        )
        bars2 = ax.bar(
            x + width / 2,
            [v / 1000 for v in our_values],
            width,
            label="This Simulator",
            color="blue",
            alpha=0.7,
        )

        ax.set_ylabel("Steps per Second (thousands)")
        ax.set_title("MineRL Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    graph_path = output_dir / f"benchmark_graphs_{results.timestamp.replace(':', '-')}.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    print(f"Graphs saved to: {graph_path}")
    plt.close()

    # Generate text summary
    summary_path = output_dir / f"benchmark_summary_{results.timestamp.replace(':', '-')}.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("MINECRAFT SIMULATOR PERFORMANCE BENCHMARK SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Timestamp: {results.timestamp}\n\n")

        f.write("PLATFORM INFORMATION\n")
        f.write("-" * 40 + "\n")
        for key, value in results.platform_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        if results.throughput.get("peak_throughput"):
            f.write("THROUGHPUT RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Peak throughput: {results.throughput['peak_throughput']:,.0f} steps/s\n")
            f.write(f"  Optimal num_envs: {results.throughput['optimal_num_envs']}\n")
            f.write("\n  Scaling:\n")
            for i, (env, sps) in enumerate(
                zip(
                    results.throughput["env_counts"],
                    results.throughput["steps_per_second"],
                )
            ):
                f.write(f"    {env:>5} envs: {sps:>12,.0f} steps/s\n")
            f.write("\n")

        if results.minerl_comparison.get("our_results"):
            our = results.minerl_comparison["our_results"]
            minerl = results.minerl_comparison["minerl_baseline"]
            f.write("MINERL COMPARISON\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"  Single env speedup: {our.get('single_env_sps', 0) / minerl['single_env_sps']:.0f}x\n"
            )
            f.write(
                f"  Peak speedup:       {our.get('peak_sps', 0) / minerl['single_env_sps']:.0f}x\n"
            )
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")

    print(f"Summary saved to: {summary_path}")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def run_all_benchmarks(config: BenchmarkConfig) -> BenchmarkResults:
    """Run complete benchmark suite."""
    results = BenchmarkResults()

    print("=" * 70)
    print("MINECRAFT SIMULATOR COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 70)
    print(f"Started: {results.timestamp}")

    # Platform info
    print("\nCollecting platform information...")
    results.platform_info = get_platform_info()
    for key, value in results.platform_info.items():
        print(f"  {key}: {value}")

    if not HAS_SIMULATOR:
        print("\nERROR: minecraft_sim module not available. Cannot run benchmarks.")
        return results

    # Run all benchmarks
    results.throughput = benchmark_throughput(config)
    results.reset_latency = benchmark_reset_latency(config)
    results.obs_transfer = benchmark_obs_transfer(config)
    results.stage_overhead = benchmark_stage_overhead(config)
    results.memory_scaling = benchmark_memory_scaling(config)
    results.minerl_comparison = benchmark_minerl_comparison(config)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    return results


def main() -> None:
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Comprehensive performance benchmarks for Minecraft Dragon Fight simulator"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed report with graphs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with reduced iterations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR / "results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--throughput-only",
        action="store_true",
        help="Run only throughput benchmark",
    )
    parser.add_argument(
        "--stage1",
        action="store_true",
        help="Run Stage 1 curriculum benchmark with 50 agents logging progress metrics",
    )
    parser.add_argument(
        "--stage1-agents",
        type=int,
        default=50,
        help="Number of agents for --stage1 preset (default: 50)",
    )
    parser.add_argument(
        "--stage1-episodes",
        type=int,
        default=20,
        help="Episodes per agent for --stage1 preset (default: 20)",
    )
    parser.add_argument(
        "--stage1-steps",
        type=int,
        default=1200,
        help="Max steps per episode for --stage1 preset (default: 1200)",
    )
    parser.add_argument(
        "--stage3",
        action="store_true",
        help="Run Stage 3 Nether benchmark with 50 agents measuring blaze rod per-minute "
        "and lava hazard avoidance rate",
    )
    parser.add_argument(
        "--stage3-agents",
        type=int,
        default=50,
        help="Number of agents for --stage3 preset (default: 50)",
    )
    parser.add_argument(
        "--stage3-episodes",
        type=int,
        default=20,
        help="Episodes per agent for --stage3 preset (default: 20)",
    )
    parser.add_argument(
        "--stage3-steps",
        type=int,
        default=2400,
        help="Max steps per episode for --stage3 preset (default: 2400)",
    )
    parser.add_argument(
        "--stronghold",
        action="store_true",
        help="Run stronghold triangulation/portal benchmark with 50 agents measuring "
        "triangulation time, eye placement velocity, and portal activation latency",
    )
    parser.add_argument(
        "--stronghold-agents",
        type=int,
        default=50,
        help="Number of agents for --stronghold preset (default: 50)",
    )
    parser.add_argument(
        "--stronghold-episodes",
        type=int,
        default=10,
        help="Episodes per agent for --stronghold preset (default: 10)",
    )
    parser.add_argument(
        "--stronghold-steps",
        type=int,
        default=2400,
        help="Max steps per episode for --stronghold preset (default: 2400)",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(output_dir=args.output_dir)

    if args.quick:
        config.throughput_warmup_steps = 50
        config.throughput_benchmark_steps = 200
        config.reset_iterations = 20
        config.obs_transfer_iterations = 200
        config.stage_iterations = 100
        config.throughput_env_counts = [1, 64, 256, 1024, 4096]

    if args.stronghold:
        stronghold_metrics = benchmark_stronghold(
            num_agents=args.stronghold_agents,
            episodes_per_agent=args.stronghold_episodes,
            steps_per_episode=args.stronghold_steps,
        )
        # Save stronghold results
        config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat().replace(":", "-")
        stronghold_path = config.output_dir / f"stronghold_benchmark_{timestamp}.json"
        with open(stronghold_path, "w") as f:
            json.dump(stronghold_metrics.to_dict(), f, indent=2)
        print(f"\n  Stronghold results saved to: {stronghold_path}")
    elif args.stage3:
        stage3_metrics = benchmark_stage3_nether(
            num_agents=args.stage3_agents,
            episodes_per_agent=args.stage3_episodes,
            steps_per_episode=args.stage3_steps,
        )
        # Save stage3 results
        config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat().replace(":", "-")
        stage3_path = config.output_dir / f"stage3_nether_benchmark_{timestamp}.json"
        with open(stage3_path, "w") as f:
            json.dump(stage3_metrics.to_dict(), f, indent=2)
        print(f"\n  Stage 3 results saved to: {stage3_path}")
    elif args.stage1:
        stage1_metrics = benchmark_stage1_curriculum(
            num_agents=args.stage1_agents,
            episodes_per_agent=args.stage1_episodes,
            steps_per_episode=args.stage1_steps,
        )
        # Save stage1 results
        config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat().replace(":", "-")
        stage1_path = config.output_dir / f"stage1_benchmark_{timestamp}.json"
        with open(stage1_path, "w") as f:
            json.dump(stage1_metrics.to_dict(), f, indent=2)
        print(f"\n  Stage 1 results saved to: {stage1_path}")
    elif args.throughput_only:
        results = BenchmarkResults()
        results.platform_info = get_platform_info()
        results.throughput = benchmark_throughput(config)
        if args.report:
            generate_report(results, config.output_dir)
    else:
        results = run_all_benchmarks(config)
        if args.report:
            generate_report(results, config.output_dir)


if __name__ == "__main__":
    main()
