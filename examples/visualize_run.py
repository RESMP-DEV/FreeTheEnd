#!/usr/bin/env python3
"""Visualize training and evaluation runs for the Minecraft RL simulator.

Features:
    1. Render observation as 2D map
    2. Show player path over time
    3. Mark milestone achievements
    4. Show dragon fight heatmap
    5. Reward over time graph
    6. Action distribution histogram

Output:
    - PNG frames for video creation
    - Interactive matplotlib plot
    - JSON summary for dashboards

Usage:
    # Run a quick episode and visualize
    python visualize_run.py --output-dir ./viz_output

    # Load from existing trajectory file
    python visualize_run.py --trajectory trajectory.npz --output-dir ./viz_output

    # Interactive mode
    python visualize_run.py --interactive

    # Generate video frames
    python visualize_run.py --video-frames --fps 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Add minecraft_sim to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

try:
    import matplotlib
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.gridspec import GridSpec

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# OBSERVATION FIELD INDICES
# Based on the 48-dimensional observation space from DragonFightEnv
# =============================================================================


@dataclass(frozen=True)
class ObsIndices:
    """Indices into the 48-dim observation vector for dragon fight."""

    # Player position (normalized [0, 1])
    PLAYER_X: int = 0
    PLAYER_Y: int = 1
    PLAYER_Z: int = 2

    # Player velocity
    PLAYER_VX: int = 3
    PLAYER_VY: int = 4
    PLAYER_VZ: int = 5

    # Player state
    PLAYER_HEALTH: int = 6
    PLAYER_YAW: int = 7
    PLAYER_PITCH: int = 8
    PLAYER_ON_GROUND: int = 9

    # Dragon state
    DRAGON_ACTIVE: int = 10
    DRAGON_HEALTH: int = 11
    DRAGON_PHASE: int = 12
    DRAGON_DX: int = 13
    DRAGON_DY: int = 14
    DRAGON_DZ: int = 15
    DRAGON_DISTANCE: int = 16

    # Crystal state (10 crystals)
    CRYSTALS_START: int = 17
    CRYSTALS_END: int = 27

    # Additional features (platform, etc.)
    NEAR_PLATFORM: int = 27
    NEAR_FOUNTAIN: int = 28
    IN_COMBAT_RANGE: int = 29

    # Ray cast distances (8 directions)
    RAYCAST_START: int = 30
    RAYCAST_END: int = 38

    # Miscellaneous
    TICK_NORMALIZED: int = 38
    CUMULATIVE_REWARD: int = 39


OBS = ObsIndices()


# =============================================================================
# ACTION NAMES
# =============================================================================

ACTION_NAMES = [
    "No-op",
    "Forward",
    "Back",
    "Left",
    "Right",
    "Jump",
    "Attack",
    "Look Left",
    "Look Right",
    "Look Up",
    "Look Down",
    "Sprint Forward",
    "Sneak",
    "Strafe Left",
    "Strafe Right",
    "Jump Attack",
    "Look at Dragon",
]


# =============================================================================
# MILESTONE DEFINITIONS
# =============================================================================


@dataclass
class Milestone:
    """A milestone achievement during the run."""

    name: str
    step: int
    position: tuple[float, float, float]
    value: float = 0.0  # e.g., dragon health at time of hit


MILESTONE_DEFS = {
    "first_crystal_destroyed": "First End Crystal destroyed",
    "half_crystals_destroyed": "50% of crystals destroyed",
    "all_crystals_destroyed": "All crystals destroyed",
    "first_dragon_hit": "First damage to dragon",
    "dragon_half_health": "Dragon at 50% health",
    "dragon_quarter_health": "Dragon at 25% health",
    "dragon_killed": "Dragon defeated!",
    "player_death": "Player died",
}


# =============================================================================
# TRAJECTORY DATA
# =============================================================================


@dataclass
class TrajectoryData:
    """Stores trajectory data from a run."""

    observations: NDArray[np.float32] = field(default_factory=lambda: np.zeros((0, 48)))
    actions: NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    rewards: NDArray[np.float32] = field(default_factory=lambda: np.zeros(0))
    dones: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    milestones: list[Milestone] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.observations)

    @property
    def player_positions(self) -> NDArray[np.float32]:
        """Extract player X, Z positions (for top-down view)."""
        if len(self.observations) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return self.observations[:, [OBS.PLAYER_X, OBS.PLAYER_Z]]

    @property
    def player_positions_3d(self) -> NDArray[np.float32]:
        """Extract player X, Y, Z positions."""
        if len(self.observations) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return self.observations[:, [OBS.PLAYER_X, OBS.PLAYER_Y, OBS.PLAYER_Z]]

    @property
    def dragon_positions(self) -> NDArray[np.float32]:
        """Extract dragon relative positions where active."""
        if len(self.observations) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        active = self.observations[:, OBS.DRAGON_ACTIVE] > 0.5
        positions = np.zeros((len(self.observations), 3), dtype=np.float32)
        # Dragon position is relative, add to player position
        positions[:, 0] = (
            self.observations[:, OBS.PLAYER_X] + self.observations[:, OBS.DRAGON_DX] * 0.1
        )
        positions[:, 1] = (
            self.observations[:, OBS.PLAYER_Y] + self.observations[:, OBS.DRAGON_DY] * 0.1
        )
        positions[:, 2] = (
            self.observations[:, OBS.PLAYER_Z] + self.observations[:, OBS.DRAGON_DZ] * 0.1
        )
        positions[~active] = np.nan
        return positions

    @property
    def cumulative_rewards(self) -> NDArray[np.float32]:
        """Compute cumulative reward over time."""
        return np.cumsum(self.rewards)

    @property
    def dragon_health_over_time(self) -> NDArray[np.float32]:
        """Get dragon health at each step."""
        if len(self.observations) == 0:
            return np.zeros(0, dtype=np.float32)
        return self.observations[:, OBS.DRAGON_HEALTH]

    @property
    def crystals_remaining(self) -> NDArray[np.int32]:
        """Count crystals remaining at each step."""
        if len(self.observations) == 0:
            return np.zeros(0, dtype=np.int32)
        crystal_states = self.observations[:, OBS.CRYSTALS_START : OBS.CRYSTALS_END]
        return (crystal_states > 0.5).sum(axis=1).astype(np.int32)

    def detect_milestones(self) -> list[Milestone]:
        """Detect milestones from trajectory data."""
        milestones = []

        if len(self.observations) < 2:
            return milestones

        crystals = self.crystals_remaining
        dragon_health = self.dragon_health_over_time
        player_health = self.observations[:, OBS.PLAYER_HEALTH]
        pos_3d = self.player_positions_3d

        initial_crystals = crystals[0] if len(crystals) > 0 else 10
        prev_crystals = initial_crystals
        prev_dragon_health = 1.0
        first_hit_detected = False
        half_health_detected = False
        quarter_health_detected = False
        half_crystals_detected = False
        all_crystals_detected = False

        for i in range(1, len(self.observations)):
            curr_crystals = crystals[i]
            curr_dragon = dragon_health[i]
            curr_player_health = player_health[i]
            pos = tuple(pos_3d[i])

            # First crystal destroyed
            if curr_crystals < initial_crystals and prev_crystals == initial_crystals:
                milestones.append(
                    Milestone(
                        name="first_crystal_destroyed",
                        step=i,
                        position=pos,
                        value=float(curr_crystals),
                    )
                )

            # Half crystals destroyed
            if not half_crystals_detected and curr_crystals <= initial_crystals // 2:
                half_crystals_detected = True
                milestones.append(
                    Milestone(
                        name="half_crystals_destroyed",
                        step=i,
                        position=pos,
                        value=float(curr_crystals),
                    )
                )

            # All crystals destroyed
            if not all_crystals_detected and curr_crystals == 0:
                all_crystals_detected = True
                milestones.append(
                    Milestone(
                        name="all_crystals_destroyed",
                        step=i,
                        position=pos,
                    )
                )

            # First dragon hit
            if not first_hit_detected and curr_dragon < prev_dragon_health:
                first_hit_detected = True
                milestones.append(
                    Milestone(
                        name="first_dragon_hit",
                        step=i,
                        position=pos,
                        value=float(curr_dragon),
                    )
                )

            # Dragon at 50% health
            if not half_health_detected and curr_dragon <= 0.5 and prev_dragon_health > 0.5:
                half_health_detected = True
                milestones.append(
                    Milestone(
                        name="dragon_half_health",
                        step=i,
                        position=pos,
                        value=float(curr_dragon),
                    )
                )

            # Dragon at 25% health
            if not quarter_health_detected and curr_dragon <= 0.25 and prev_dragon_health > 0.25:
                quarter_health_detected = True
                milestones.append(
                    Milestone(
                        name="dragon_quarter_health",
                        step=i,
                        position=pos,
                        value=float(curr_dragon),
                    )
                )

            # Dragon killed
            if curr_dragon <= 0 and prev_dragon_health > 0:
                milestones.append(
                    Milestone(
                        name="dragon_killed",
                        step=i,
                        position=pos,
                    )
                )

            # Player death
            if curr_player_health <= 0 and player_health[i - 1] > 0:
                milestones.append(
                    Milestone(
                        name="player_death",
                        step=i,
                        position=pos,
                    )
                )

            prev_crystals = curr_crystals
            prev_dragon_health = curr_dragon

        return milestones

    def save(self, path: str | Path) -> None:
        """Save trajectory to npz file."""
        path = Path(path)
        milestone_data = [
            {
                "name": m.name,
                "step": int(m.step),
                "position": [float(p) for p in m.position],
                "value": float(m.value),
            }
            for m in self.milestones
        ]
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            milestones=json.dumps(milestone_data),
            metadata=json.dumps(self.metadata),
        )

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryData:
        """Load trajectory from npz file."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        milestone_data = json.loads(str(data["milestones"]))
        milestones = [
            Milestone(
                name=m["name"],
                step=m["step"],
                position=tuple(m["position"]),
                value=m.get("value", 0.0),
            )
            for m in milestone_data
        ]
        return cls(
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
            dones=data["dones"],
            milestones=milestones,
            metadata=json.loads(str(data["metadata"])),
        )


# =============================================================================
# RUN COLLECTOR
# =============================================================================


def collect_trajectory(
    num_steps: int = 1000,
    env_index: int = 0,
    policy: str = "random",
) -> TrajectoryData:
    """Collect a trajectory from the simulator.

    Args:
        num_steps: Maximum number of steps to collect.
        env_index: Which environment to visualize (for vectorized envs).
        policy: Policy to use - "random" or "chase_dragon".

    Returns:
        TrajectoryData with collected observations, actions, rewards.
    """
    try:
        from minecraft_sim.vec_env import VecDragonFightEnv
    except ImportError:
        print("Error: minecraft_sim not available. Using mock data.")
        return _create_mock_trajectory(num_steps)

    try:
        env = VecDragonFightEnv(num_envs=max(1, env_index + 1))
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Using mock data for visualization.")
        return _create_mock_trajectory(num_steps)

    observations = []
    actions = []
    rewards = []
    dones = []

    obs = env.reset()
    observations.append(obs[env_index].copy())

    for step in range(num_steps):
        if policy == "random":
            action = np.random.randint(0, 17, size=env.num_envs)
        elif policy == "chase_dragon":
            action = _chase_dragon_policy(obs)
        else:
            action = np.zeros(env.num_envs, dtype=np.int32)

        obs, reward, done, info = env.step(action)

        actions.append(action[env_index])
        rewards.append(reward[env_index])
        dones.append(done[env_index])

        if done[env_index]:
            observations.append(obs[env_index].copy())
            break

        observations.append(obs[env_index].copy())

    env.close()

    traj = TrajectoryData(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int32),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
        metadata={
            "num_steps": len(observations),
            "policy": policy,
            "env_index": env_index,
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    traj.milestones = traj.detect_milestones()
    return traj


def _chase_dragon_policy(obs: NDArray[np.float32]) -> NDArray[np.int32]:
    """Simple policy that tries to face and attack the dragon."""
    num_envs = obs.shape[0]
    actions = np.zeros(num_envs, dtype=np.int32)

    for i in range(num_envs):
        dragon_active = obs[i, OBS.DRAGON_ACTIVE] > 0.5
        dragon_dx = obs[i, OBS.DRAGON_DX]
        dragon_distance = obs[i, OBS.DRAGON_DISTANCE]
        in_combat = obs[i, OBS.IN_COMBAT_RANGE] > 0.5

        if not dragon_active:
            # No dragon, move forward
            actions[i] = 1  # Forward
        elif in_combat:
            # In range, attack
            actions[i] = 6  # Attack
        elif abs(dragon_dx) > 0.3:
            # Turn toward dragon
            if dragon_dx > 0:
                actions[i] = 8  # Look right
            else:
                actions[i] = 7  # Look left
        else:
            # Move toward dragon
            if dragon_distance > 0.2:
                actions[i] = 11  # Sprint forward
            else:
                actions[i] = 1  # Forward

    return actions


def _create_mock_trajectory(num_steps: int) -> TrajectoryData:
    """Create mock trajectory data for testing visualization."""
    rng = np.random.default_rng(42)

    # Generate spiral path for player
    t = np.linspace(0, 4 * np.pi, num_steps)
    radius = 0.3 + 0.1 * t / (4 * np.pi)
    player_x = 0.5 + radius * np.cos(t) * 0.3
    player_z = 0.5 + radius * np.sin(t) * 0.3
    player_y = 0.3 + 0.1 * np.sin(t * 2)

    # Create observation array
    observations = np.zeros((num_steps, 48), dtype=np.float32)
    observations[:, OBS.PLAYER_X] = player_x
    observations[:, OBS.PLAYER_Y] = player_y
    observations[:, OBS.PLAYER_Z] = player_z
    observations[:, OBS.PLAYER_HEALTH] = np.clip(
        1.0 - 0.3 * rng.random(num_steps).cumsum() / num_steps, 0.1, 1.0
    )

    # Dragon state
    observations[:, OBS.DRAGON_ACTIVE] = 1.0
    observations[:, OBS.DRAGON_HEALTH] = np.clip(1.0 - np.linspace(0, 1.2, num_steps), 0, 1)
    observations[:, OBS.DRAGON_DX] = 0.5 * np.cos(t * 0.5)
    observations[:, OBS.DRAGON_DZ] = 0.5 * np.sin(t * 0.5)
    observations[:, OBS.DRAGON_DISTANCE] = 0.3 + 0.2 * np.abs(np.sin(t))

    # Crystals (destroy over time)
    for c in range(10):
        destroy_step = int(num_steps * (c + 1) / 12)
        observations[:destroy_step, OBS.CRYSTALS_START + c] = 1.0

    # Random actions
    actions = rng.integers(0, 17, size=num_steps)

    # Rewards: small positive for movement, bigger for damage
    rewards = 0.01 * rng.random(num_steps) - 0.005
    damage_steps = np.where(np.diff(observations[:, OBS.DRAGON_HEALTH]) < -0.01)[0]
    rewards[damage_steps] += 1.0
    crystal_steps = np.where(
        np.diff(observations[:, OBS.CRYSTALS_START : OBS.CRYSTALS_END].sum(axis=1)) < -0.5
    )[0]
    rewards[crystal_steps] += 0.5

    dones = np.zeros(num_steps, dtype=bool)
    if observations[-1, OBS.DRAGON_HEALTH] <= 0:
        dones[-1] = True

    traj = TrajectoryData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        metadata={
            "num_steps": num_steps,
            "policy": "mock",
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    traj.milestones = traj.detect_milestones()
    return traj


# =============================================================================
# VISUALIZERS
# =============================================================================


class RunVisualizer:
    """Visualization tools for Minecraft RL training runs."""

    # Color scheme
    COLORS = {
        "background": "#1a1a2e",
        "grid": "#2d2d44",
        "player_path": "#00ff88",
        "player_current": "#ffff00",
        "dragon_path": "#ff4444",
        "dragon_current": "#ff0000",
        "crystal": "#aa00ff",
        "crystal_destroyed": "#440044",
        "fountain": "#4444ff",
        "platform": "#666666",
        "milestone_positive": "#00ff00",
        "milestone_negative": "#ff0000",
        "reward_positive": "#00ff88",
        "reward_negative": "#ff4466",
        "health_high": "#00ff00",
        "health_low": "#ff0000",
    }

    # End dimension layout (simplified)
    END_PLATFORM_CENTER = (0.5, 0.5)
    END_PLATFORM_RADIUS = 0.35
    FOUNTAIN_CENTER = (0.5, 0.5)
    FOUNTAIN_RADIUS = 0.05

    # Crystal positions (10 crystals in a circle)
    CRYSTAL_POSITIONS = [
        (0.5 + 0.3 * np.cos(i * 2 * np.pi / 10), 0.5 + 0.3 * np.sin(i * 2 * np.pi / 10))
        for i in range(10)
    ]

    def __init__(self, trajectory: TrajectoryData, style: str = "dark") -> None:
        """Initialize visualizer with trajectory data.

        Args:
            trajectory: TrajectoryData to visualize.
            style: Plot style - "dark" or "light".
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for visualization")

        self.trajectory = trajectory
        self.style = style

        if style == "dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("seaborn-v0_8-whitegrid")

    def render_2d_map(
        self,
        ax: plt.Axes | None = None,
        show_path: bool = True,
        show_dragon: bool = True,
        show_crystals: bool = True,
        show_milestones: bool = True,
        step: int | None = None,
    ) -> plt.Axes:
        """Render top-down 2D map of the End dimension.

        Args:
            ax: Matplotlib axes to draw on. Creates new if None.
            show_path: Show player path trail.
            show_dragon: Show dragon position/path.
            show_crystals: Show crystal positions.
            show_milestones: Show milestone markers.
            step: If provided, only show up to this step.

        Returns:
            Matplotlib axes with the rendered map.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        ax.set_facecolor(self.COLORS["background"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("X (normalized)")
        ax.set_ylabel("Z (normalized)")
        ax.set_title("End Dimension - Dragon Fight")

        # Draw main platform (circle)
        platform = plt.Circle(
            self.END_PLATFORM_CENTER,
            self.END_PLATFORM_RADIUS,
            fill=True,
            color=self.COLORS["platform"],
            alpha=0.3,
        )
        ax.add_patch(platform)

        # Draw fountain
        fountain = plt.Circle(
            self.FOUNTAIN_CENTER,
            self.FOUNTAIN_RADIUS,
            fill=True,
            color=self.COLORS["fountain"],
            alpha=0.5,
        )
        ax.add_patch(fountain)

        # Draw crystals
        if show_crystals and len(self.trajectory.observations) > 0:
            step_idx = step if step is not None else len(self.trajectory.observations) - 1
            crystal_states = self.trajectory.observations[
                step_idx, OBS.CRYSTALS_START : OBS.CRYSTALS_END
            ]

            for i, (cx, cz) in enumerate(self.CRYSTAL_POSITIONS):
                if i < len(crystal_states):
                    alive = crystal_states[i] > 0.5
                    color = self.COLORS["crystal"] if alive else self.COLORS["crystal_destroyed"]
                    marker = "D" if alive else "x"
                    ax.scatter([cx], [cz], c=color, marker=marker, s=100, zorder=5)

        # Get positions up to step
        end_step = step if step is not None else len(self.trajectory.observations)
        positions = self.trajectory.player_positions[:end_step]

        # Draw player path with color gradient (time)
        if show_path and len(positions) > 1:
            # Create line segments colored by time
            points = positions.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Color by time
            norm = Normalize(vmin=0, vmax=len(segments))
            cmap = LinearSegmentedColormap.from_list(
                "path_cmap",
                ["#004400", self.COLORS["player_path"]],
            )
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2, alpha=0.7)
            lc.set_array(np.arange(len(segments)))
            ax.add_collection(lc)

        # Draw current player position
        if len(positions) > 0:
            ax.scatter(
                [positions[-1, 0]],
                [positions[-1, 1]],
                c=self.COLORS["player_current"],
                marker="o",
                s=150,
                zorder=10,
                edgecolor="white",
                linewidth=2,
            )

        # Draw dragon path and position
        if show_dragon:
            dragon_pos = self.trajectory.dragon_positions[:end_step]
            valid = ~np.isnan(dragon_pos[:, 0])
            if valid.any():
                dragon_xz = dragon_pos[valid][:, [0, 2]]  # X, Z
                if len(dragon_xz) > 1:
                    ax.plot(
                        dragon_xz[:, 0],
                        dragon_xz[:, 1],
                        color=self.COLORS["dragon_path"],
                        alpha=0.4,
                        linewidth=1,
                    )
                if len(dragon_xz) > 0:
                    ax.scatter(
                        [dragon_xz[-1, 0]],
                        [dragon_xz[-1, 1]],
                        c=self.COLORS["dragon_current"],
                        marker="^",
                        s=200,
                        zorder=9,
                        edgecolor="white",
                        linewidth=2,
                    )

        # Draw milestones
        if show_milestones:
            for m in self.trajectory.milestones:
                if step is None or m.step <= step:
                    color = (
                        self.COLORS["milestone_negative"]
                        if m.name == "player_death"
                        else self.COLORS["milestone_positive"]
                    )
                    ax.scatter(
                        [m.position[0]],
                        [m.position[2]],
                        c=color,
                        marker="*",
                        s=300,
                        zorder=11,
                        edgecolor="white",
                        linewidth=1.5,
                    )

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=self.COLORS["player_path"], label="Player Path"),
            plt.scatter([], [], c=self.COLORS["player_current"], marker="o", s=100, label="Player"),
            plt.scatter([], [], c=self.COLORS["dragon_current"], marker="^", s=100, label="Dragon"),
            plt.scatter([], [], c=self.COLORS["crystal"], marker="D", s=50, label="Crystal"),
            plt.scatter(
                [], [], c=self.COLORS["milestone_positive"], marker="*", s=100, label="Milestone"
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        return ax

    def render_heatmap(
        self,
        ax: plt.Axes | None = None,
        data_type: str = "player",
        bins: int = 50,
    ) -> plt.Axes:
        """Render heatmap of player or dragon positions.

        Args:
            ax: Matplotlib axes to draw on.
            data_type: "player" or "dragon".
            bins: Number of bins for histogram.

        Returns:
            Matplotlib axes with heatmap.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        if data_type == "player":
            positions = self.trajectory.player_positions
            title = "Player Position Heatmap"
            cmap = "Greens"
        else:
            dragon_3d = self.trajectory.dragon_positions
            valid = ~np.isnan(dragon_3d[:, 0])
            positions = dragon_3d[valid][:, [0, 2]] if valid.any() else np.zeros((0, 2))
            title = "Dragon Position Heatmap"
            cmap = "Reds"

        if len(positions) > 0:
            h, xedges, yedges = np.histogram2d(
                positions[:, 0],
                positions[:, 1],
                bins=bins,
                range=[[0, 1], [0, 1]],
            )
            im = ax.imshow(
                h.T,
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap=cmap,
                aspect="equal",
                alpha=0.8,
            )
            plt.colorbar(im, ax=ax, label="Visit count")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("X (normalized)")
        ax.set_ylabel("Z (normalized)")
        ax.set_title(title)

        # Overlay platform outline
        platform = plt.Circle(
            self.END_PLATFORM_CENTER,
            self.END_PLATFORM_RADIUS,
            fill=False,
            color="white",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(platform)

        return ax

    def render_rewards_graph(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Render reward over time graph.

        Args:
            ax: Matplotlib axes to draw on.

        Returns:
            Matplotlib axes with reward graph.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        steps = np.arange(len(self.trajectory.rewards))
        rewards = self.trajectory.rewards
        cumulative = self.trajectory.cumulative_rewards

        # Per-step rewards (bar)
        colors = [
            self.COLORS["reward_positive"] if r >= 0 else self.COLORS["reward_negative"]
            for r in rewards
        ]
        ax.bar(steps, rewards, color=colors, alpha=0.5, width=1.0, label="Step Reward")

        # Cumulative rewards (line)
        ax2 = ax.twinx()
        ax2.plot(steps, cumulative, color="#ffffff", linewidth=2, label="Cumulative")
        ax2.fill_between(steps, 0, cumulative, alpha=0.2, color="white")

        # Mark milestones
        for m in self.trajectory.milestones:
            if m.step < len(steps):
                color = (
                    self.COLORS["milestone_negative"]
                    if m.name == "player_death"
                    else self.COLORS["milestone_positive"]
                )
                ax.axvline(m.step, color=color, linestyle="--", alpha=0.7)
                ax.annotate(
                    MILESTONE_DEFS.get(m.name, m.name),
                    xy=(m.step, rewards[m.step] if m.step < len(rewards) else 0),
                    xytext=(5, 10),
                    textcoords="offset points",
                    fontsize=7,
                    rotation=45,
                    color=color,
                )

        ax.set_xlabel("Step")
        ax.set_ylabel("Step Reward", color=self.COLORS["reward_positive"])
        ax2.set_ylabel("Cumulative Reward", color="white")
        ax.set_title("Reward Over Time")

        return ax

    def render_action_histogram(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Render action distribution histogram.

        Args:
            ax: Matplotlib axes to draw on.

        Returns:
            Matplotlib axes with histogram.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        action_counts = np.bincount(self.trajectory.actions, minlength=17)

        colors = plt.cm.viridis(np.linspace(0, 1, 17))
        bars = ax.bar(range(17), action_counts, color=colors)

        ax.set_xticks(range(17))
        ax.set_xticklabels(ACTION_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Action")
        ax.set_ylabel("Count")
        ax.set_title("Action Distribution")

        # Add percentage labels
        total = action_counts.sum()
        if total > 0:
            for i, (bar, count) in enumerate(zip(bars, action_counts)):
                pct = 100 * count / total
                if pct > 2:
                    ax.annotate(
                        f"{pct:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                    )

        return ax

    def render_health_graph(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Render player and dragon health over time.

        Args:
            ax: Matplotlib axes to draw on.

        Returns:
            Matplotlib axes with health graph.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        steps = np.arange(len(self.trajectory.observations))
        player_health = self.trajectory.observations[:, OBS.PLAYER_HEALTH]
        dragon_health = self.trajectory.dragon_health_over_time
        crystals = self.trajectory.crystals_remaining / 10.0  # Normalize to 0-1

        ax.fill_between(steps, 0, player_health, alpha=0.3, color="#00ff88", label="Player Health")
        ax.plot(steps, player_health, color="#00ff88", linewidth=2)

        ax.fill_between(steps, 0, dragon_health, alpha=0.3, color="#ff4444", label="Dragon Health")
        ax.plot(steps, dragon_health, color="#ff4444", linewidth=2)

        ax.plot(
            steps, crystals, color="#aa00ff", linewidth=2, linestyle="--", label="Crystals (x10)"
        )

        ax.set_xlabel("Step")
        ax.set_ylabel("Health / Crystals (normalized)")
        ax.set_title("Health and Crystals Over Time")
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right")

        return ax

    def render_dashboard(
        self,
        figsize: tuple[int, int] = (16, 12),
        step: int | None = None,
    ) -> plt.Figure:
        """Render full dashboard with all visualizations.

        Args:
            figsize: Figure size (width, height).
            step: If provided, show state at this step.

        Returns:
            Matplotlib figure with dashboard.
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 2D Map (large, top-left)
        ax_map = fig.add_subplot(gs[0:2, 0:2])
        self.render_2d_map(ax_map, step=step)

        # Player heatmap (top-right)
        ax_heatmap_player = fig.add_subplot(gs[0, 2])
        self.render_heatmap(ax_heatmap_player, data_type="player", bins=30)

        # Dragon heatmap (middle-right)
        ax_heatmap_dragon = fig.add_subplot(gs[1, 2])
        self.render_heatmap(ax_heatmap_dragon, data_type="dragon", bins=30)

        # Rewards graph (bottom-left)
        ax_rewards = fig.add_subplot(gs[2, 0:2])
        self.render_rewards_graph(ax_rewards)

        # Action histogram (bottom-right)
        ax_actions = fig.add_subplot(gs[2, 2])
        self.render_action_histogram(ax_actions)

        fig.suptitle(
            f"Minecraft RL Run Visualization - {self.trajectory.num_steps} steps",
            fontsize=14,
            fontweight="bold",
        )

        return fig

    def save_frames(
        self,
        output_dir: str | Path,
        frame_interval: int = 10,
        figsize: tuple[int, int] = (16, 12),
    ) -> list[Path]:
        """Save PNG frames for video creation.

        Args:
            output_dir: Directory to save frames.
            frame_interval: Save every N steps.
            figsize: Figure size.

        Returns:
            List of saved frame paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_frames = []
        num_frames = len(self.trajectory.observations) // frame_interval + 1

        print(f"Saving {num_frames} frames to {output_dir}")

        for i, step in enumerate(range(0, len(self.trajectory.observations), frame_interval)):
            fig = self.render_dashboard(figsize=figsize, step=step)

            frame_path = output_dir / f"frame_{i:05d}.png"
            fig.savefig(
                frame_path, dpi=100, bbox_inches="tight", facecolor=self.COLORS["background"]
            )
            plt.close(fig)

            saved_frames.append(frame_path)

            if (i + 1) % 10 == 0:
                print(f"  Saved frame {i + 1}/{num_frames}")

        print(f"Done! Saved {len(saved_frames)} frames.")
        return saved_frames

    def to_json_summary(self) -> dict[str, Any]:
        """Generate JSON summary for dashboards.

        Returns:
            Dictionary with summary statistics.
        """
        traj = self.trajectory

        summary: dict[str, Any] = {
            "metadata": traj.metadata,
            "statistics": {
                "num_steps": traj.num_steps,
                "total_reward": float(traj.rewards.sum()),
                "mean_reward": float(traj.rewards.mean()) if len(traj.rewards) > 0 else 0.0,
                "max_reward": float(traj.rewards.max()) if len(traj.rewards) > 0 else 0.0,
                "min_reward": float(traj.rewards.min()) if len(traj.rewards) > 0 else 0.0,
                "episode_completed": bool(traj.dones.any()),
                "dragon_killed": any(m.name == "dragon_killed" for m in traj.milestones),
                "player_died": any(m.name == "player_death" for m in traj.milestones),
            },
            "dragon_fight": {
                "initial_health": float(traj.dragon_health_over_time[0])
                if len(traj.observations) > 0
                else 1.0,
                "final_health": float(traj.dragon_health_over_time[-1])
                if len(traj.observations) > 0
                else 1.0,
                "damage_dealt": float(1.0 - traj.dragon_health_over_time[-1])
                if len(traj.observations) > 0
                else 0.0,
                "initial_crystals": int(traj.crystals_remaining[0])
                if len(traj.observations) > 0
                else 10,
                "final_crystals": int(traj.crystals_remaining[-1])
                if len(traj.observations) > 0
                else 0,
                "crystals_destroyed": int(traj.crystals_remaining[0] - traj.crystals_remaining[-1])
                if len(traj.observations) > 0
                else 0,
            },
            "milestones": [
                {
                    "name": m.name,
                    "description": MILESTONE_DEFS.get(m.name, m.name),
                    "step": m.step,
                    "position": list(m.position),
                    "value": m.value,
                }
                for m in traj.milestones
            ],
            "action_distribution": {
                ACTION_NAMES[i]: int(count)
                for i, count in enumerate(np.bincount(traj.actions, minlength=17))
            },
            "player_stats": {
                "distance_traveled": float(
                    np.sum(np.linalg.norm(np.diff(traj.player_positions, axis=0), axis=1))
                )
                if len(traj.observations) > 1
                else 0.0,
                "final_health": float(traj.observations[-1, OBS.PLAYER_HEALTH])
                if len(traj.observations) > 0
                else 0.0,
            },
        }

        return summary


# =============================================================================
# INTERACTIVE MODE
# =============================================================================


def run_interactive(trajectory: TrajectoryData) -> None:
    """Run interactive visualization with slider.

    Args:
        trajectory: TrajectoryData to visualize.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for interactive mode")
        return

    from matplotlib.widgets import Slider

    viz = RunVisualizer(trajectory)

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3, height_ratios=[2, 2, 1, 0.2])

    # Map
    ax_map = fig.add_subplot(gs[0:2, 0:2])

    # Heatmaps
    ax_heatmap_player = fig.add_subplot(gs[0, 2])
    ax_heatmap_dragon = fig.add_subplot(gs[1, 2])

    # Rewards and health
    ax_rewards = fig.add_subplot(gs[2, 0:2])
    ax_health = fig.add_subplot(gs[2, 2])

    # Slider
    ax_slider = fig.add_subplot(gs[3, :])

    # Initial render
    def update(step: int) -> None:
        ax_map.clear()
        viz.render_2d_map(ax_map, step=step)

        ax_rewards.clear()
        viz.render_rewards_graph(ax_rewards)
        ax_rewards.axvline(step, color="yellow", linewidth=2)

        ax_health.clear()
        viz.render_health_graph(ax_health)
        ax_health.axvline(step, color="yellow", linewidth=2)

        fig.canvas.draw_idle()

    # Render static parts
    viz.render_heatmap(ax_heatmap_player, data_type="player", bins=30)
    viz.render_heatmap(ax_heatmap_dragon, data_type="dragon", bins=30)

    # Create slider
    max_step = len(trajectory.observations) - 1
    slider = Slider(
        ax_slider,
        "Step",
        0,
        max_step,
        valinit=max_step,
        valstep=1,
    )
    slider.on_changed(lambda val: update(int(val)))

    # Initial update
    update(max_step)

    fig.suptitle("Minecraft RL Run Visualization (Interactive)", fontsize=14, fontweight="bold")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Minecraft RL training/evaluation runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        help="Path to trajectory .npz file to load",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./viz_output",
        help="Output directory for frames and summary",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with step slider",
    )
    parser.add_argument(
        "--video-frames",
        action="store_true",
        help="Generate PNG frames for video creation",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=10,
        help="Save every N steps when generating frames",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of steps to collect (if not loading trajectory)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "chase_dragon"],
        default="random",
        help="Policy to use when collecting trajectory",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="16,12",
        help="Figure size as 'width,height'",
    )

    args = parser.parse_args()

    # Check matplotlib
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        sys.exit(1)

    # Load or collect trajectory
    if args.trajectory:
        print(f"Loading trajectory from {args.trajectory}")
        trajectory = TrajectoryData.load(args.trajectory)
    else:
        print(f"Collecting {args.num_steps} steps with {args.policy} policy...")
        trajectory = collect_trajectory(
            num_steps=args.num_steps,
            policy=args.policy,
        )

    print(f"Trajectory: {trajectory.num_steps} steps, {len(trajectory.milestones)} milestones")

    # Parse figsize
    figsize = tuple(map(int, args.figsize.split(",")))

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizer
    viz = RunVisualizer(trajectory)

    # Interactive mode
    if args.interactive:
        run_interactive(trajectory)
        return

    # Generate video frames
    if args.video_frames:
        frames = viz.save_frames(
            output_dir / "frames",
            frame_interval=args.frame_interval,
            figsize=figsize,
        )
        print("\nTo create video, run:")
        print(
            f"  ffmpeg -framerate 20 -i {output_dir}/frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p run.mp4"
        )

    # Save dashboard
    print("Rendering dashboard...")
    fig = viz.render_dashboard(figsize=figsize)
    dashboard_path = output_dir / "dashboard.png"
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight", facecolor=viz.COLORS["background"])
    plt.close(fig)
    print(f"Saved dashboard to {dashboard_path}")

    # Save JSON summary
    summary = viz.to_json_summary()
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Save trajectory
    trajectory_path = output_dir / "trajectory.npz"
    trajectory.save(trajectory_path)
    print(f"Saved trajectory to {trajectory_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"Steps: {summary['statistics']['num_steps']}")
    print(f"Total reward: {summary['statistics']['total_reward']:.2f}")
    print(f"Dragon damage: {summary['dragon_fight']['damage_dealt'] * 100:.1f}%")
    print(f"Crystals destroyed: {summary['dragon_fight']['crystals_destroyed']}/10")
    print(f"Dragon killed: {summary['statistics']['dragon_killed']}")
    print(f"Player died: {summary['statistics']['player_died']}")
    print("\nMilestones:")
    for m in summary["milestones"]:
        print(f"  Step {m['step']:5d}: {m['description']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
