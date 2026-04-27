"""Progression tracking for Minecraft speedrun curriculum learning.

This module tracks player progress through all stages of a speedrun,
from basic survival to defeating the Ender Dragon. Progress data can
be saved/loaded for curriculum training and analysis.

Each stage has specific metrics that indicate mastery:
- Stage 1 (Survival): Wood, stone, combat basics
- Stage 2 (Resources): Iron, diamonds, tools
- Stage 3 (Nether): Portal, fortress, blaze rods
- Stage 4 (Pearls): Enderman hunting, pearl collection
- Stage 5 (Stronghold): Eye crafting, portal location
- Stage 6 (Dragon): End entry, crystal destruction, dragon kill
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

import logging

logger = logging.getLogger(__name__)


class SpeedrunStage(IntEnum):
    """Speedrun stages matching curriculum StageID."""

    SURVIVAL = 1
    RESOURCES = 2
    NETHER = 3
    PEARLS = 4
    STRONGHOLD = 5
    DRAGON = 6


@dataclass
class SpeedrunProgress:
    """Track player progress across all speedrun stages.

    This dataclass captures all metrics relevant to speedrun progression,
    organized by stage. Values update during gameplay and persist across
    training episodes for curriculum learning.

    Attributes:
        # Stage 1: Basic Survival
        wood_collected: Total wood logs collected.
        stone_collected: Total cobblestone collected.
        zombies_killed: Zombies killed (combat practice).
        first_night_survived: Whether the first night was survived.
        food_eaten: Food items consumed.

        # Stage 2: Resource Gathering
        iron_ore_mined: Iron ore blocks mined.
        iron_ingots: Smelted iron ingots in inventory.
        diamonds: Diamonds in inventory.
        has_iron_pickaxe: True if iron pickaxe crafted.
        has_iron_sword: True if iron sword crafted.
        has_bucket: True if bucket crafted.
        has_shield: True if shield crafted.

        # Stage 3: Nether Navigation
        obsidian_collected: Obsidian blocks collected.
        portal_built: True if nether portal was constructed.
        entered_nether: True if player entered the nether.
        fortress_found: True if nether fortress was located.
        blazes_killed: Number of blazes killed.
        blaze_rods: Blaze rods in inventory.

        # Stage 4: Enderman/Pearl Hunting
        endermen_killed: Number of endermen killed.
        ender_pearls: Ender pearls in inventory.
        nether_wart_collected: Nether wart collected (for potions).
        piglins_bartered: Successful piglin barters.

        # Stage 5: Stronghold Finding
        eyes_crafted: Eyes of ender crafted.
        eyes_used: Eyes thrown to locate stronghold.
        stronghold_found: True if stronghold was located.
        portal_room_found: True if end portal room was found.
        eyes_placed: Eyes placed in portal frame (0-12).
        portal_activated: True if end portal was activated.

        # Stage 6: Dragon Fight
        entered_end: True if player entered The End.
        crystals_destroyed: End crystals destroyed (0-10).
        dragon_damage_dealt: Total damage dealt to dragon.
        dragon_phase_changes: Number of dragon phase transitions observed.
        dragon_perches: Number of times dragon perched on fountain.
        dragon_killed: True if ender dragon was killed.

        # Timing and meta
        total_ticks: Total game ticks across all stages.
        stage_times: Dict mapping stage number to ticks spent.
        deaths: Total death count.
        stage_deaths: Dict mapping stage number to deaths.
        current_stage: Currently active stage (1-6).
        episode_count: Number of training episodes.
    """

    # Stage 1: Basic Survival
    wood_collected: int = 0
    stone_collected: int = 0
    zombies_killed: int = 0
    first_night_survived: bool = False
    food_eaten: int = 0

    # Stage 2: Resource Gathering
    iron_ore_mined: int = 0
    iron_ingots: int = 0
    diamonds: int = 0
    has_iron_pickaxe: bool = False
    has_iron_sword: bool = False
    has_bucket: bool = False
    has_shield: bool = False

    # Stage 3: Nether Navigation
    obsidian_collected: int = 0
    portal_built: bool = False
    entered_nether: bool = False
    fortress_found: bool = False
    blazes_killed: int = 0
    blaze_rods: int = 0

    # Stage 4: Enderman/Pearl Hunting
    endermen_killed: int = 0
    ender_pearls: int = 0
    nether_wart_collected: int = 0
    piglins_bartered: int = 0

    # Stage 5: Stronghold Finding
    eyes_crafted: int = 0
    eyes_used: int = 0
    stronghold_found: bool = False
    stronghold_distance: float = float("inf")
    portal_room_found: bool = False
    eyes_placed: int = 0
    portal_activated: bool = False

    # Stage 6: Dragon Fight
    entered_end: bool = False
    crystals_destroyed: int = 0
    dragon_damage_dealt: float = 0.0
    dragon_phase_changes: int = 0
    dragon_perches: int = 0
    dragon_killed: bool = False

    # Timing and meta
    total_ticks: int = 0
    stage_times: dict[int, int] = field(default_factory=dict)
    deaths: int = 0
    stage_deaths: dict[int, int] = field(default_factory=dict)
    current_stage: int = 1
    episode_count: int = 0

    def __post_init__(self) -> None:
        """Initialize stage tracking dicts if empty."""
        if not self.stage_times:
            self.stage_times = {s.value: 0 for s in SpeedrunStage}
        if not self.stage_deaths:
            self.stage_deaths = {s.value: 0 for s in SpeedrunStage}

    def reset_episode(self) -> None:
        """Reset per-episode counters while preserving cumulative stats."""
        logger.debug("SpeedrunProgress.reset_episode called")
        self.episode_count += 1
        self.current_stage = 1

    def get_stage_completion(self, stage: int | SpeedrunStage) -> float:
        """Calculate completion percentage for a stage.

        Args:
            stage: Stage number (1-6) or SpeedrunStage enum.

        Returns:
            Completion percentage [0.0, 1.0].
        """
        logger.debug("SpeedrunProgress.get_stage_completion: stage=%s", stage)
        stage_val = stage.value if isinstance(stage, SpeedrunStage) else stage

        if stage_val == SpeedrunStage.SURVIVAL:
            targets = [
                (self.wood_collected, 16),
                (self.stone_collected, 32),
                (self.food_eaten, 5),
                (int(self.first_night_survived), 1),
            ]
        elif stage_val == SpeedrunStage.RESOURCES:
            targets = [
                (self.iron_ingots, 10),
                (self.diamonds, 3),
                (int(self.has_iron_pickaxe), 1),
                (int(self.has_bucket), 1),
            ]
        elif stage_val == SpeedrunStage.NETHER:
            targets = [
                (int(self.portal_built), 1),
                (int(self.entered_nether), 1),
                (int(self.fortress_found), 1),
                (self.blaze_rods, 7),
            ]
        elif stage_val == SpeedrunStage.PEARLS:
            targets = [
                (self.ender_pearls, 12),
                (self.endermen_killed, 12),
            ]
        elif stage_val == SpeedrunStage.STRONGHOLD:
            targets = [
                (self.eyes_crafted, 12),
                (int(self.stronghold_found), 1),
                (int(self.portal_room_found), 1),
                (self.eyes_placed, 12),
            ]
        elif stage_val == SpeedrunStage.DRAGON:
            targets = [
                (int(self.entered_end), 1),
                (self.crystals_destroyed, 10),
                (min(self.dragon_damage_dealt / 200.0, 1.0), 1),
                (int(self.dragon_killed), 1),
            ]
        else:
            return 0.0

        total_weight = sum(t[1] for t in targets)
        achieved = sum(min(val / target, 1.0) * target for val, target in targets)
        return achieved / total_weight if total_weight > 0 else 0.0

    def get_overall_completion(self) -> float:
        """Calculate overall speedrun completion percentage.

        Returns:
            Weighted average completion across all stages [0.0, 1.0].
        """
        logger.debug("SpeedrunProgress.get_overall_completion called")
        weights = [1.0, 1.5, 2.0, 2.0, 1.5, 3.0]  # Later stages weighted more
        completions = [self.get_stage_completion(s) for s in SpeedrunStage]
        return sum(c * w for c, w in zip(completions, weights)) / sum(weights)

    def is_stage_complete(self, stage: int | SpeedrunStage) -> bool:
        """Check if a stage's primary objectives are complete.

        Args:
            stage: Stage number (1-6) or SpeedrunStage enum.

        Returns:
            True if stage is considered complete.
        """
        logger.debug("SpeedrunProgress.is_stage_complete: stage=%s", stage)
        stage_val = stage.value if isinstance(stage, SpeedrunStage) else stage

        if stage_val == SpeedrunStage.SURVIVAL:
            return self.wood_collected >= 16 and self.stone_collected >= 32
        if stage_val == SpeedrunStage.RESOURCES:
            return self.has_iron_pickaxe and self.has_bucket and self.iron_ingots >= 3
        if stage_val == SpeedrunStage.NETHER:
            return self.entered_nether and self.blaze_rods >= 7
        if stage_val == SpeedrunStage.PEARLS:
            return self.ender_pearls >= 12
        if stage_val == SpeedrunStage.STRONGHOLD:
            return self.portal_room_found and self.eyes_placed >= 12
        if stage_val == SpeedrunStage.DRAGON:
            return self.dragon_killed
        return False

    def to_observation(self) -> np.ndarray:
        """Convert progress to a normalized observation vector.

        Returns:
            Float32 array of shape (32,) with normalized progress values.
        """
        logger.debug("SpeedrunProgress.to_observation called")
        return np.array(
            [
                # Stage 1 (5 values)
                min(self.wood_collected / 64.0, 1.0),
                min(self.stone_collected / 128.0, 1.0),
                min(self.zombies_killed / 10.0, 1.0),
                float(self.first_night_survived),
                min(self.food_eaten / 20.0, 1.0),
                # Stage 2 (5 values)
                min(self.iron_ingots / 20.0, 1.0),
                min(self.diamonds / 10.0, 1.0),
                float(self.has_iron_pickaxe),
                float(self.has_bucket),
                float(self.has_shield),
                # Stage 3 (5 values)
                float(self.portal_built),
                float(self.entered_nether),
                float(self.fortress_found),
                min(self.blazes_killed / 10.0, 1.0),
                min(self.blaze_rods / 10.0, 1.0),
                # Stage 4 (4 values)
                min(self.endermen_killed / 15.0, 1.0),
                min(self.ender_pearls / 16.0, 1.0),
                min(self.nether_wart_collected / 16.0, 1.0),
                min(self.piglins_bartered / 20.0, 1.0),
                # Stage 5 (5 values)
                min(self.eyes_crafted / 12.0, 1.0),
                float(self.stronghold_found),
                float(self.portal_room_found),
                min(self.eyes_placed / 12.0, 1.0),
                float(self.portal_activated),
                # Stage 6 (5 values)
                float(self.entered_end),
                min(self.crystals_destroyed / 10.0, 1.0),
                min(self.dragon_damage_dealt / 200.0, 1.0),
                min(self.dragon_perches / 5.0, 1.0),
                float(self.dragon_killed),
                # Meta (3 values)
                min(self.total_ticks / 72000.0, 1.0),  # Normalize to 1 hour
                min(self.deaths / 10.0, 1.0),
                (self.current_stage - 1) / 5.0,  # Stage as [0, 1]
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize progress to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        logger.debug("SpeedrunProgress.to_dict called")
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeedrunProgress:
        """Deserialize progress from dictionary.

        Args:
            data: Dictionary with progress fields.

        Returns:
            SpeedrunProgress instance.
        """
        # Handle nested dicts that need int keys
        logger.debug("SpeedrunProgress.from_dict: data=%s", data)
        if "stage_times" in data and isinstance(data["stage_times"], dict):
            data["stage_times"] = {int(k): v for k, v in data["stage_times"].items()}
        if "stage_deaths" in data and isinstance(data["stage_deaths"], dict):
            data["stage_deaths"] = {int(k): v for k, v in data["stage_deaths"].items()}
        return cls(**data)

    def save(self, path: Path | str) -> None:
        """Save progress to JSON file.

        Args:
            path: Path to save file.
        """
        logger.debug("SpeedrunProgress.save: path=%s", path)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> SpeedrunProgress:
        """Load progress from JSON file.

        Args:
            path: Path to progress file.

        Returns:
            Loaded SpeedrunProgress instance.
        """
        logger.info("SpeedrunProgress.load: path=%s", path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ProgressTracker:
    """Real-time progress tracker that updates from observations.

    This class wraps SpeedrunProgress and provides methods to update
    progress based on game observations, making it suitable for
    integration with environment step loops.

    Attributes:
        progress: The underlying SpeedrunProgress instance.
        prev_health: Previous health value for death detection.
        prev_dimension: Previous dimension for portal tracking.
    """

    progress: SpeedrunProgress = field(default_factory=SpeedrunProgress)
    prev_health: float = 20.0
    prev_dimension: int = 0
    prev_dragon_health: float = 200.0

    def update_from_observation(self, obs: dict[str, Any]) -> dict[str, float]:
        """Update progress from a game observation dictionary.

        Args:
            obs: Observation dictionary with player state, inventory, etc.

        Returns:
            Dictionary of reward signals for achievements unlocked this tick.
        """
        logger.debug("ProgressTracker.update_from_observation: obs=%s", obs)
        rewards: dict[str, float] = {}
        p = self.progress

        # Update tick count
        tick = obs.get("tick_number", obs.get("game_tick", p.total_ticks + 1))
        tick_delta = tick - p.total_ticks
        p.total_ticks = tick
        p.stage_times[p.current_stage] = p.stage_times.get(p.current_stage, 0) + tick_delta

        # Detect death (health dropped to 0 or respawn detected)
        player = obs.get("player", {})
        health = player.get("health", 20.0)
        if health <= 0 and self.prev_health > 0:
            p.deaths += 1
            p.stage_deaths[p.current_stage] = p.stage_deaths.get(p.current_stage, 0) + 1
            rewards["death_penalty"] = -1.0
        self.prev_health = health

        # Detect dimension change
        dimension = player.get("dimension", 0)
        if dimension != self.prev_dimension:
            if dimension == 1 and not p.entered_nether:
                p.entered_nether = True
                rewards["entered_nether"] = 10.0
            elif dimension == 2 and not p.entered_end:
                p.entered_end = True
                rewards["entered_end"] = 20.0
        self.prev_dimension = dimension

        # Update Stage 1 cumulative stats (zombies_killed, food_eaten)
        stats = obs.get("stats", {})
        zombies = stats.get("zombies_killed", 0)
        if zombies > p.zombies_killed:
            rewards["zombies_killed"] = 0.2 * (zombies - p.zombies_killed)
            p.zombies_killed = zombies
        food = stats.get("food_eaten", 0)
        if food > p.food_eaten:
            rewards["food_eaten"] = 0.1 * (food - p.food_eaten)
            p.food_eaten = food

        # Update Stage 3 nether snapshot
        nether = obs.get("nether", {})
        if nether:
            if nether.get("portal_built", False) and not p.portal_built:
                p.portal_built = True
                rewards["portal_built"] = 5.0
            if nether.get("entered_nether", nether.get("in_nether", False)) and not p.entered_nether:
                p.entered_nether = True
                rewards["entered_nether"] = 10.0
            if nether.get("fortress_found", False) and not p.fortress_found:
                p.fortress_found = True
                rewards["fortress_found"] = 8.0
            blazes = nether.get("blazes_killed", 0)
            if blazes > p.blazes_killed:
                rewards["blaze_killed"] = 3.0 * (blazes - p.blazes_killed)
                p.blazes_killed = blazes

        # Update inventory-based progress
        inv = obs.get("inventory", {})
        self._update_inventory_progress(inv, rewards)

        # Update stronghold/portal progress (Stage 5)
        stronghold_dist = obs.get("distance_to_stronghold")
        if stronghold_dist is not None:
            p.stronghold_distance = stronghold_dist
            if not p.stronghold_found and stronghold_dist < 50.0:
                p.stronghold_found = True
                rewards["stronghold_found"] = 10.0

        eyes = obs.get("eyes_placed", obs.get("portal_frame_filled", 0))
        if isinstance(eyes, float) and eyes <= 1.0:
            eyes = int(eyes * 12)
        eyes = int(eyes)
        if eyes > p.eyes_placed:
            rewards["eye_placed"] = 1.0 * (eyes - p.eyes_placed)
            p.eyes_placed = eyes

        portal_active = obs.get("end_portal_activated", obs.get("portal_active", False))
        if portal_active and not p.portal_activated:
            p.portal_activated = True
            rewards["portal_activated"] = 20.0

        # Update dragon fight progress
        dragon = obs.get("dragon", {})
        if dragon.get("is_active", False):
            self._update_dragon_progress(dragon, rewards)

        # Auto-advance stage based on completion
        for stage in SpeedrunStage:
            if stage.value == p.current_stage and p.is_stage_complete(stage):
                if stage.value < SpeedrunStage.DRAGON:
                    p.current_stage = stage.value + 1
                    rewards[f"stage_{stage.value}_complete"] = 5.0

        return rewards

    def _update_inventory_progress(self, inv: dict[str, int], rewards: dict[str, float]) -> None:
        """Update progress based on inventory contents."""
        logger.debug("ProgressTracker._update_inventory_progress: inv=%s, rewards=%s", inv, rewards)
        p = self.progress

        # Wood (item IDs 17, 162 for logs, or count from inventory summary)
        wood = inv.get("wood", inv.get("logs", 0))
        if wood > p.wood_collected:
            rewards["wood_collected"] = 0.1 * (wood - p.wood_collected)
            p.wood_collected = wood

        # Stone
        stone = inv.get("cobblestone", inv.get("stone", 0))
        if stone > p.stone_collected:
            rewards["stone_collected"] = 0.05 * (stone - p.stone_collected)
            p.stone_collected = stone

        # Iron ore
        iron_ore = inv.get("iron_ore", 0)
        if iron_ore > p.iron_ore_mined:
            rewards["iron_ore_mined"] = 0.3 * (iron_ore - p.iron_ore_mined)
            p.iron_ore_mined = iron_ore

        # Iron ingots
        iron = inv.get("iron_ingots", inv.get("iron_ingot", 0))
        if iron > p.iron_ingots:
            rewards["iron_collected"] = 0.5 * (iron - p.iron_ingots)
            p.iron_ingots = iron

        # Diamonds
        diamonds = inv.get("diamonds", inv.get("diamond", 0))
        if diamonds > p.diamonds:
            rewards["diamond_collected"] = 2.0 * (diamonds - p.diamonds)
            p.diamonds = diamonds

        # Obsidian
        obsidian = inv.get("obsidian", 0)
        if obsidian > p.obsidian_collected:
            rewards["obsidian_collected"] = 0.5 * (obsidian - p.obsidian_collected)
            p.obsidian_collected = obsidian

        # Tools (check for presence)
        if inv.get("iron_pickaxe", 0) > 0 and not p.has_iron_pickaxe:
            p.has_iron_pickaxe = True
            rewards["iron_pickaxe_crafted"] = 2.0

        if inv.get("bucket", inv.get("empty_buckets", 0)) > 0 and not p.has_bucket:
            p.has_bucket = True
            rewards["bucket_crafted"] = 1.0

        # Water/lava buckets also count as having a bucket
        if (inv.get("water_buckets", 0) > 0 or inv.get("lava_buckets", 0) > 0) and not p.has_bucket:
            p.has_bucket = True
            rewards["bucket_crafted"] = 1.0

        # Blaze rods
        rods = inv.get("blaze_rods", inv.get("blaze_rod", 0))
        if rods > p.blaze_rods:
            rewards["blaze_rod_collected"] = 3.0 * (rods - p.blaze_rods)
            p.blaze_rods = rods

        # Ender pearls
        pearls = inv.get("ender_pearls", inv.get("ender_pearl", 0))
        if pearls > p.ender_pearls:
            rewards["ender_pearl_collected"] = 2.0 * (pearls - p.ender_pearls)
            p.ender_pearls = pearls

        # Eyes of ender
        eyes = inv.get("eyes_of_ender", inv.get("ender_eye", 0))
        if eyes > p.eyes_crafted:
            rewards["eye_crafted"] = 1.0 * (eyes - p.eyes_crafted)
            p.eyes_crafted = eyes

    def _update_dragon_progress(self, dragon: dict[str, Any], rewards: dict[str, float]) -> None:
        """Update progress from dragon fight state."""
        logger.debug("ProgressTracker._update_dragon_progress: dragon=%s, rewards=%s", dragon, rewards)
        p = self.progress

        # Track dragon damage
        dragon_health = dragon.get("dragon_health", dragon.get("health", 200.0))
        if isinstance(dragon_health, float) and dragon_health < 1.0:
            # Normalized health, convert to absolute
            dragon_health = dragon_health * 200.0

        damage = max(0, self.prev_dragon_health - dragon_health)
        if damage > 0:
            p.dragon_damage_dealt += damage
            rewards["dragon_damage"] = damage * 0.1
        self.prev_dragon_health = dragon_health

        # Track crystals
        crystals = dragon.get("crystals_remaining", 10)
        destroyed = 10 - crystals
        if destroyed > p.crystals_destroyed:
            new_destroyed = destroyed - p.crystals_destroyed
            rewards["crystal_destroyed"] = 5.0 * new_destroyed
            p.crystals_destroyed = destroyed

        # Track perches
        phase = dragon.get("phase", 0)
        if phase == 3:  # Perching
            # This is simplified; real tracking would need state machine
            pass

        # Check for dragon kill
        if dragon_health <= 0 and not p.dragon_killed:
            p.dragon_killed = True
            rewards["dragon_killed"] = 100.0

    def to_snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly dict describing the current SpeedrunProgress.

        Combines the raw progress fields from SpeedrunProgress.to_dict() with
        computed completion metrics per stage and overall.

        Returns:
            Dictionary with keys 'progress', 'stage_completions',
            'overall_completion', and 'current_stage'.
        """
        logger.debug("ProgressTracker.to_snapshot called")
        p = self.progress
        return {
            "progress": p.to_dict(),
            "stage_completions": {
                stage.name.lower(): p.get_stage_completion(stage)
                for stage in SpeedrunStage
            },
            "overall_completion": p.get_overall_completion(),
            "current_stage": p.current_stage,
        }

    def reset(self) -> None:
        """Reset tracker for new episode while preserving cumulative stats."""
        logger.debug("ProgressTracker.reset called")
        self.progress.reset_episode()
        self.prev_health = 20.0
        self.prev_dimension = 0
        self.prev_dragon_health = 200.0


def create_progress_observation_space() -> tuple[np.ndarray, np.ndarray]:
    """Create observation space bounds for progress vector.

    Returns:
        Tuple of (low_bounds, high_bounds) arrays of shape (32,).
    """
    logger.info("create_progress_observation_space called")
    low = np.zeros(32, dtype=np.float32)
    high = np.ones(32, dtype=np.float32)
    return low, high


def merge_progress(runs: list[SpeedrunProgress]) -> SpeedrunProgress:
    """Merge progress from multiple runs into aggregate statistics.

    Args:
        runs: List of SpeedrunProgress instances to merge.

    Returns:
        New SpeedrunProgress with summed/maxed values.
    """
    logger.debug("merge_progress: runs=%s", runs)
    if not runs:
        return SpeedrunProgress()

    merged = SpeedrunProgress()

    # Sum cumulative counters
    for r in runs:
        merged.wood_collected += r.wood_collected
        merged.stone_collected += r.stone_collected
        merged.zombies_killed += r.zombies_killed
        merged.food_eaten += r.food_eaten
        merged.iron_ore_mined += r.iron_ore_mined
        merged.iron_ingots += r.iron_ingots
        merged.diamonds += r.diamonds
        merged.obsidian_collected += r.obsidian_collected
        merged.blazes_killed += r.blazes_killed
        merged.blaze_rods += r.blaze_rods
        merged.endermen_killed += r.endermen_killed
        merged.ender_pearls += r.ender_pearls
        merged.nether_wart_collected += r.nether_wart_collected
        merged.piglins_bartered += r.piglins_bartered
        merged.eyes_crafted += r.eyes_crafted
        merged.eyes_used += r.eyes_used
        merged.eyes_placed += r.eyes_placed
        merged.crystals_destroyed += r.crystals_destroyed
        merged.dragon_damage_dealt += r.dragon_damage_dealt
        merged.dragon_phase_changes += r.dragon_phase_changes
        merged.dragon_perches += r.dragon_perches
        merged.total_ticks += r.total_ticks
        merged.deaths += r.deaths
        merged.episode_count += r.episode_count

        for stage in SpeedrunStage:
            merged.stage_times[stage.value] = merged.stage_times.get(
                stage.value, 0
            ) + r.stage_times.get(stage.value, 0)
            merged.stage_deaths[stage.value] = merged.stage_deaths.get(
                stage.value, 0
            ) + r.stage_deaths.get(stage.value, 0)

    # Take any True for boolean achievements
    merged.first_night_survived = any(r.first_night_survived for r in runs)
    merged.has_iron_pickaxe = any(r.has_iron_pickaxe for r in runs)
    merged.has_iron_sword = any(r.has_iron_sword for r in runs)
    merged.has_bucket = any(r.has_bucket for r in runs)
    merged.has_shield = any(r.has_shield for r in runs)
    merged.portal_built = any(r.portal_built for r in runs)
    merged.entered_nether = any(r.entered_nether for r in runs)
    merged.fortress_found = any(r.fortress_found for r in runs)
    merged.stronghold_found = any(r.stronghold_found for r in runs)
    merged.portal_room_found = any(r.portal_room_found for r in runs)
    merged.portal_activated = any(r.portal_activated for r in runs)
    merged.entered_end = any(r.entered_end for r in runs)
    merged.dragon_killed = any(r.dragon_killed for r in runs)

    # Max stage reached
    merged.current_stage = max(r.current_stage for r in runs)

    return merged
