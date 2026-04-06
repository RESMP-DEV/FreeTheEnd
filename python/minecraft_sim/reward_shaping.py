"""Stage-specific reward shaping functions for Minecraft speedrun training.

This module provides dense reward signals for each curriculum stage to guide
the agent toward optimal speedrun behavior. Each stage has a unique reward
shaper that tracks progress and provides shaped rewards based on:

1. Milestone achievements (one-time bonuses for key actions)
2. Progressive rewards (incremental bonuses for gathering resources)
3. Efficiency penalties (time pressure to encourage fast play)
4. Death/damage penalties (survival pressure)
5. Stage completion bonuses

The reward shapers maintain internal state to track which milestones have
been achieved and prevent reward hacking through repeated collection.

Example:
    >>> from minecraft_sim.reward_shaping import create_reward_shaper
    >>> shaper = create_reward_shaper(stage_id=1)
    >>> reward = shaper(observation_state)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardStats:
    """Statistics tracked by reward shapers for logging/debugging."""

    total_reward: float = 0.0
    milestone_rewards: float = 0.0
    progressive_rewards: float = 0.0
    penalties: float = 0.0
    stage_completion_bonus: float = 0.0
    milestones_achieved: list[str] = field(default_factory=list)


def _get_inventory(state: dict[str, Any], item: str, default: int = 0) -> int:
    """Safely get item count from inventory."""
    return state.get("inventory", {}).get(item, default)


def _get_flag(state: dict[str, Any], flag: str) -> bool:
    """Safely get boolean flag from state."""
    return bool(state.get(flag, False))


def _get_value(state: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Safely get numeric value from state."""
    return float(state.get(key, default))


# =============================================================================
# Stage 1: Basic Survival
# =============================================================================


def create_stage1_reward_shaper() -> Callable[[dict[str, Any]], float]:
    """Create reward shaper for Stage 1: Basic Survival.

    Objectives:
    - Survive (avoid damage and death)
    - Gather wood and craft planks
    - Craft a crafting table
    - Mine cobblestone
    - Craft wooden then stone pickaxe
    - Craft a furnace (one-time bonus via progress_snapshot)
    - Find coal and iron ore

    Returns:
        Callable that takes state dict and returns shaped reward.
    """
    given_rewards: set[str] = set()
    prev_state: dict[str, Any] = {}
    stats = RewardStats()

    def shape_reward(state: dict[str, Any]) -> float:
        nonlocal prev_state
        reward = 0.0

        # Time penalty (encourages efficiency) - 0.0001 per tick
        reward -= 0.0001

        # Death penalty - terminal
        if _get_value(state, "health") <= 0:
            reward -= 1.0
            stats.penalties += 1.0
            stats.total_reward += reward
            return reward

        # Damage taken penalty
        if prev_state:
            prev_health = _get_value(prev_state, "health", 20.0)
            curr_health = _get_value(state, "health", 20.0)
            damage = prev_health - curr_health
            if damage > 0:
                penalty = damage * 0.02
                reward -= penalty
                stats.penalties += penalty

        # Hunger penalty
        if prev_state:
            prev_hunger = _get_value(prev_state, "hunger", 20.0)
            curr_hunger = _get_value(state, "hunger", 20.0)
            hunger_loss = prev_hunger - curr_hunger
            if hunger_loss > 0:
                penalty = hunger_loss * 0.01
                reward -= penalty
                stats.penalties += penalty

        # Milestone rewards (one-time)
        milestones = [
            # Wood gathering
            ("first_wood", _get_inventory(state, "oak_log") > 0, 0.2),
            ("wood_x4", _get_inventory(state, "oak_log") >= 4, 0.1),
            ("wood_x8", _get_inventory(state, "oak_log") >= 8, 0.1),
            # Crafting progression
            ("first_planks", _get_inventory(state, "oak_planks") > 0, 0.1),
            ("first_sticks", _get_inventory(state, "stick") > 0, 0.05),
            (
                "crafting_table",
                _get_inventory(state, "crafting_table") > 0
                or _get_flag(state, "has_crafting_table"),
                0.15,
            ),
            # Tool progression
            ("wooden_pickaxe", _get_flag(state, "has_wooden_pickaxe"), 0.3),
            ("wooden_sword", _get_flag(state, "has_wooden_sword"), 0.1),
            ("wooden_axe", _get_flag(state, "has_wooden_axe"), 0.1),
            # Stone age
            ("first_stone", _get_inventory(state, "cobblestone") > 0, 0.15),
            ("stone_x16", _get_inventory(state, "cobblestone") >= 16, 0.1),
            ("stone_pickaxe", _get_flag(state, "has_stone_pickaxe"), 0.3),
            ("stone_sword", _get_flag(state, "has_stone_sword"), 0.1),
            (
                "furnace_crafted",
                state.get("progress_snapshot", {}).get("has_furnace", False),
                0.2,
            ),
            # Resource discovery
            ("first_coal", _get_inventory(state, "coal") > 0, 0.15),
            (
                "first_iron_ore",
                _get_inventory(state, "iron_ore") > 0 or _get_inventory(state, "raw_iron") > 0,
                0.25,
            ),
            ("first_food", _get_value(state, "food_count", 0) > 0, 0.1),
            # Combat achievements
            ("first_kill", _get_value(state, "mobs_killed") > 0, 0.15),
        ]

        for name, condition, bonus in milestones:
            if condition and name not in given_rewards:
                reward += bonus
                given_rewards.add(name)
                stats.milestone_rewards += bonus
                stats.milestones_achieved.append(name)

        # Progressive rewards (diminishing)
        wood_count = _get_inventory(state, "oak_log")
        wood_bonus = min(wood_count * 0.02, 0.2)

        stone_count = _get_inventory(state, "cobblestone")
        stone_bonus = min(stone_count * 0.005, 0.2)

        coal_count = _get_inventory(state, "coal")
        coal_bonus = min(coal_count * 0.01, 0.15)

        # Calculate delta from previous state
        if prev_state:
            prev_wood = min(_get_inventory(prev_state, "oak_log") * 0.02, 0.2)
            prev_stone = min(_get_inventory(prev_state, "cobblestone") * 0.005, 0.2)
            prev_coal = min(_get_inventory(prev_state, "coal") * 0.01, 0.15)

            prog_reward = (
                (wood_bonus - prev_wood) + (stone_bonus - prev_stone) + (coal_bonus - prev_coal)
            )
            if prog_reward > 0:
                reward += prog_reward
                stats.progressive_rewards += prog_reward

        # Mob kill rewards
        mobs_killed = int(_get_value(state, "mobs_killed"))
        prev_mobs = int(_get_value(prev_state, "mobs_killed")) if prev_state else 0
        kill_reward = (mobs_killed - prev_mobs) * 0.1
        if kill_reward > 0:
            reward += kill_reward
            stats.progressive_rewards += kill_reward

        # Exploration bonus (new chunks/areas)
        chunks_explored = int(_get_value(state, "chunks_explored"))
        prev_chunks = int(_get_value(prev_state, "chunks_explored")) if prev_state else 0
        explore_reward = (chunks_explored - prev_chunks) * 0.01
        if explore_reward > 0:
            reward += explore_reward
            stats.progressive_rewards += explore_reward

        # Stage completion bonus
        if _get_flag(state, "stage_complete") and "stage_complete" not in given_rewards:
            reward += 2.0
            given_rewards.add("stage_complete")
            stats.stage_completion_bonus = 2.0

        prev_state.clear()
        prev_state.update(state)
        stats.total_reward += reward
        return reward

    shape_reward.stats = stats  # type: ignore[attr-defined]
    shape_reward.reset = lambda: (given_rewards.clear(), prev_state.clear(), stats.__init__())  # type: ignore[attr-defined]
    return shape_reward


# =============================================================================
# Stage 2: Resource Gathering
# =============================================================================


def create_stage2_reward_shaper() -> Callable[[dict[str, Any]], float]:
    """Create reward shaper for Stage 2: Resource Gathering.

    Objectives:
    - Mine iron ore and smelt ingots
    - Craft iron pickaxe
    - Craft bucket
    - Find diamonds (optional)
    - Gather obsidian
    - Prepare for Nether

    Returns:
        Callable that takes state dict and returns shaped reward.
    """
    given_rewards: set[str] = set()
    prev_state: dict[str, Any] = {}
    stats = RewardStats()

    def shape_reward(state: dict[str, Any]) -> float:
        nonlocal prev_state
        reward = 0.0

        # Time penalty
        reward -= 0.0001

        # Death penalty
        if _get_value(state, "health") <= 0:
            reward -= 0.8
            stats.penalties += 0.8
            stats.total_reward += reward
            return reward

        # Damage penalty
        if prev_state:
            damage = _get_value(prev_state, "health", 20.0) - _get_value(state, "health", 20.0)
            if damage > 0:
                penalty = damage * 0.015
                reward -= penalty
                stats.penalties += penalty

        # Milestone rewards
        milestones = [
            # Iron progression
            (
                "first_iron_ore",
                _get_inventory(state, "iron_ore") > 0 or _get_inventory(state, "raw_iron") > 0,
                0.15,
            ),
            (
                "iron_ore_x3",
                _get_inventory(state, "iron_ore") >= 3 or _get_inventory(state, "raw_iron") >= 3,
                0.1,
            ),
            ("first_iron_ingot", _get_inventory(state, "iron_ingot") > 0, 0.2),
            ("iron_ingot_x3", _get_inventory(state, "iron_ingot") >= 3, 0.15),
            ("iron_ingot_x10", _get_inventory(state, "iron_ingot") >= 10, 0.15),
            ("iron_pickaxe", _get_flag(state, "has_iron_pickaxe"), 0.35),
            ("iron_sword", _get_flag(state, "has_iron_sword"), 0.1),
            # Bucket and water
            ("bucket", _get_flag(state, "has_bucket") or _get_inventory(state, "bucket") > 0, 0.3),
            ("water_bucket", _get_inventory(state, "water_bucket") > 0, 0.15),
            ("lava_bucket", _get_inventory(state, "lava_bucket") > 0, 0.2),
            # Diamond discovery
            ("first_diamond", _get_inventory(state, "diamond") > 0, 0.3),
            ("diamond_x3", _get_inventory(state, "diamond") >= 3, 0.2),
            ("diamond_pickaxe", _get_flag(state, "has_diamond_pickaxe"), 0.3),
            # Obsidian collection
            ("first_obsidian", _get_inventory(state, "obsidian") > 0, 0.2),
            ("obsidian_x10", _get_inventory(state, "obsidian") >= 10, 0.25),
            ("obsidian_x14", _get_inventory(state, "obsidian") >= 14, 0.1),  # Full portal
            # Flint and steel
            ("flint", _get_inventory(state, "flint") > 0, 0.05),
            (
                "flint_and_steel",
                _get_flag(state, "has_flint_and_steel")
                or _get_inventory(state, "flint_and_steel") > 0,
                0.15,
            ),
            # Gold for piglins
            ("first_gold", _get_inventory(state, "gold_ingot") > 0, 0.1),
        ]

        for name, condition, bonus in milestones:
            if condition and name not in given_rewards:
                reward += bonus
                given_rewards.add(name)
                stats.milestone_rewards += bonus
                stats.milestones_achieved.append(name)

        # Progressive rewards
        iron_count = _get_inventory(state, "iron_ingot")
        iron_bonus = min(iron_count * 0.015, 0.3)

        obsidian_count = _get_inventory(state, "obsidian")
        obsidian_bonus = min(obsidian_count * 0.015, 0.25)

        if prev_state:
            prev_iron = min(_get_inventory(prev_state, "iron_ingot") * 0.015, 0.3)
            prev_obsidian = min(_get_inventory(prev_state, "obsidian") * 0.015, 0.25)

            prog_reward = (iron_bonus - prev_iron) + (obsidian_bonus - prev_obsidian)
            if prog_reward > 0:
                reward += prog_reward
                stats.progressive_rewards += prog_reward

        # Y-level bonus (deeper = more valuable ores)
        y_level = _get_value(state, "y_position", 64)
        if y_level < 16 and prev_state:
            prev_y = _get_value(prev_state, "y_position", 64)
            if y_level < prev_y:
                depth_reward = 0.005
                reward += depth_reward
                stats.progressive_rewards += depth_reward

        # Stage completion
        if _get_flag(state, "stage_complete") and "stage_complete" not in given_rewards:
            reward += 2.0
            given_rewards.add("stage_complete")
            stats.stage_completion_bonus = 2.0

        prev_state.clear()
        prev_state.update(state)
        stats.total_reward += reward
        return reward

    shape_reward.stats = stats  # type: ignore[attr-defined]
    shape_reward.reset = lambda: (given_rewards.clear(), prev_state.clear(), stats.__init__())  # type: ignore[attr-defined]
    return shape_reward


# =============================================================================
# Stage 3: Nether Navigation
# =============================================================================


def create_stage3_reward_shaper() -> Callable[[dict[str, Any]], float]:
    """Create reward shaper for Stage 3: Nether Navigation.

    Objectives:
    - Build nether portal
    - Enter the Nether
    - Locate a fortress
    - Kill blazes and collect blaze rods
    - Survive hostile environment

    Returns:
        Callable that takes state dict and returns shaped reward.
    """
    given_rewards: set[str] = set()
    prev_state: dict[str, Any] = {}
    stats = RewardStats()

    def shape_reward(state: dict[str, Any]) -> float:
        nonlocal prev_state
        reward = 0.0

        # Time penalty (slightly higher in Nether due to danger)
        reward -= 0.00012

        # Death penalty (higher in Nether - harder to recover)
        if _get_value(state, "health") <= 0:
            reward -= 1.2
            stats.penalties += 1.2
            stats.total_reward += reward
            return reward

        # Damage penalty (higher for fire/lava)
        if prev_state:
            damage = _get_value(prev_state, "health", 20.0) - _get_value(state, "health", 20.0)
            if damage > 0:
                # Extra penalty for fire/lava damage
                in_lava = _get_flag(state, "in_lava")
                on_fire = _get_value(state, "fire_ticks") > 0
                multiplier = 0.03 if (in_lava or on_fire) else 0.02
                penalty = damage * multiplier
                reward -= penalty
                stats.penalties += penalty

        # Lava proximity warning
        if _get_flag(state, "in_lava"):
            reward -= 0.01

        # Milestone rewards
        milestones = [
            # Portal construction
            ("portal_frame_placed", _get_flag(state, "portal_frame_placed"), 0.2),
            ("portal_built", _get_flag(state, "portal_built"), 0.35),
            ("portal_lit", _get_flag(state, "portal_lit"), 0.15),
            # Nether entry
            (
                "entered_nether",
                _get_flag(state, "entered_nether") or _get_flag(state, "in_nether"),
                0.4,
            ),
            # Fortress discovery
            ("fortress_visible", _get_flag(state, "fortress_visible"), 0.2),
            ("fortress_found", _get_flag(state, "fortress_found"), 0.4),
            ("in_fortress", _get_flag(state, "in_fortress"), 0.2),
            # Blaze hunting
            ("first_blaze_seen", _get_flag(state, "blaze_seen"), 0.1),
            ("blaze_spawner_found", _get_flag(state, "blaze_spawner_found"), 0.25),
            ("first_blaze_kill", _get_inventory(state, "blaze_rod") > 0, 0.3),
            ("blaze_rod_x3", _get_inventory(state, "blaze_rod") >= 3, 0.2),
            ("blaze_rod_x5", _get_inventory(state, "blaze_rod") >= 5, 0.2),
            ("blaze_rod_x7", _get_inventory(state, "blaze_rod") >= 7, 0.25),
            ("blaze_rod_x10", _get_inventory(state, "blaze_rod") >= 10, 0.15),
            # Nether wart (optional but useful)
            ("nether_wart", _get_inventory(state, "nether_wart") > 0, 0.1),
            # Fire resistance
            ("fire_resistance", _get_flag(state, "has_fire_resistance"), 0.15),
            # Ghast fireball deflection
            ("ghast_fireball_deflected", _get_flag(state, "ghast_fireball_deflected"), 0.3),
        ]

        for name, condition, bonus in milestones:
            if condition and name not in given_rewards:
                reward += bonus
                given_rewards.add(name)
                stats.milestone_rewards += bonus
                stats.milestones_achieved.append(name)

        # Progressive blaze rod rewards
        blaze_count = _get_inventory(state, "blaze_rod")
        blaze_bonus = min(blaze_count * 0.03, 0.35)

        if prev_state:
            prev_blaze = min(_get_inventory(prev_state, "blaze_rod") * 0.03, 0.35)
            prog_reward = blaze_bonus - prev_blaze
            if prog_reward > 0:
                reward += prog_reward
                stats.progressive_rewards += prog_reward

        # Distance to fortress reward (when searching)
        if _get_flag(state, "in_nether") and not _get_flag(state, "fortress_found"):
            dist_to_fortress = _get_value(state, "distance_to_fortress", 1000)
            if prev_state and "distance_to_fortress" in prev_state:
                prev_dist = _get_value(prev_state, "distance_to_fortress", 1000)
                # Reward getting closer
                if dist_to_fortress < prev_dist:
                    approach_reward = min((prev_dist - dist_to_fortress) * 0.001, 0.05)
                    reward += approach_reward
                    stats.progressive_rewards += approach_reward

        # Blaze kill rewards (per kill)
        blazes_killed = int(_get_value(state, "blazes_killed"))
        prev_blazes = int(_get_value(prev_state, "blazes_killed")) if prev_state else 0
        kill_reward = (blazes_killed - prev_blazes) * 0.15
        if kill_reward > 0:
            reward += kill_reward
            stats.progressive_rewards += kill_reward

        # Stage completion
        if _get_flag(state, "stage_complete") and "stage_complete" not in given_rewards:
            reward += 2.5
            given_rewards.add("stage_complete")
            stats.stage_completion_bonus = 2.5

        prev_state.clear()
        prev_state.update(state)
        stats.total_reward += reward
        return reward

    shape_reward.stats = stats  # type: ignore[attr-defined]
    shape_reward.reset = lambda: (given_rewards.clear(), prev_state.clear(), stats.__init__())  # type: ignore[attr-defined]
    return shape_reward


# =============================================================================
# Stage 4: Enderman Hunting
# =============================================================================


def create_stage4_reward_shaper() -> Callable[[dict[str, Any]], float]:
    """Create reward shaper for Stage 4: Enderman Hunting.

    Objectives:
    - Hunt endermen for ender pearls
    - Collect 12+ pearls for eyes of ender
    - Craft blaze powder from rods
    - Survive enderman attacks

    Returns:
        Callable that takes state dict and returns shaped reward.
    """
    given_rewards: set[str] = set()
    prev_state: dict[str, Any] = {}
    stats = RewardStats()

    def shape_reward(state: dict[str, Any]) -> float:
        nonlocal prev_state
        reward = 0.0

        # Time penalty
        reward -= 0.00015

        # Death penalty (endermen can be deadly)
        if _get_value(state, "health") <= 0:
            reward -= 1.0
            stats.penalties += 1.0
            stats.total_reward += reward
            return reward

        # Damage penalty
        if prev_state:
            damage = _get_value(prev_state, "health", 20.0) - _get_value(state, "health", 20.0)
            if damage > 0:
                penalty = damage * 0.02
                reward -= penalty
                stats.penalties += penalty

        # Milestone rewards
        milestones = [
            # Enderman hunting
            (
                "enderman_spotted",
                _get_flag(state, "enderman_nearby") or _get_flag(state, "enderman_seen"),
                0.1,
            ),
            ("first_enderman_kill", _get_inventory(state, "ender_pearl") > 0, 0.25),
            ("pearl_x3", _get_inventory(state, "ender_pearl") >= 3, 0.15),
            ("pearl_x6", _get_inventory(state, "ender_pearl") >= 6, 0.15),
            ("pearl_x9", _get_inventory(state, "ender_pearl") >= 9, 0.15),
            ("pearl_x12", _get_inventory(state, "ender_pearl") >= 12, 0.25),
            ("pearl_x16", _get_inventory(state, "ender_pearl") >= 16, 0.15),
            # Blaze powder crafting
            ("first_blaze_powder", _get_inventory(state, "blaze_powder") > 0, 0.15),
            ("blaze_powder_x12", _get_inventory(state, "blaze_powder") >= 12, 0.1),
            # Eye of ender crafting
            (
                "first_eye",
                _get_inventory(state, "eye_of_ender") > 0 or _get_value(state, "eyes_crafted") > 0,
                0.2,
            ),
            (
                "eye_x6",
                _get_inventory(state, "eye_of_ender") >= 6
                or _get_value(state, "eyes_crafted") >= 6,
                0.15,
            ),
            (
                "eye_x12",
                _get_inventory(state, "eye_of_ender") >= 12
                or _get_value(state, "eyes_crafted") >= 12,
                0.25,
            ),
            # Combat gear
            ("has_shield", _get_flag(state, "has_shield"), 0.1),
            ("has_armor", _get_value(state, "armor_equipped") >= 2, 0.1),
            # Warped forest (safe enderman farm location)
            ("in_warped_forest", _get_flag(state, "in_warped_forest"), 0.15),
        ]

        for name, condition, bonus in milestones:
            if condition and name not in given_rewards:
                reward += bonus
                given_rewards.add(name)
                stats.milestone_rewards += bonus
                stats.milestones_achieved.append(name)

        # Progressive pearl rewards
        pearl_count = _get_inventory(state, "ender_pearl")
        pearl_bonus = min(pearl_count * 0.02, 0.4)

        eye_count = _get_inventory(state, "eye_of_ender") + int(_get_value(state, "eyes_crafted"))
        eye_bonus = min(eye_count * 0.025, 0.35)

        if prev_state:
            prev_pearls = min(_get_inventory(prev_state, "ender_pearl") * 0.02, 0.4)
            prev_eyes = min(
                (
                    _get_inventory(prev_state, "eye_of_ender")
                    + int(_get_value(prev_state, "eyes_crafted"))
                )
                * 0.025,
                0.35,
            )

            prog_reward = (pearl_bonus - prev_pearls) + (eye_bonus - prev_eyes)
            if prog_reward > 0:
                reward += prog_reward
                stats.progressive_rewards += prog_reward

        # Enderman kill rewards
        endermen_killed = int(_get_value(state, "endermen_killed"))
        prev_endermen = int(_get_value(prev_state, "endermen_killed")) if prev_state else 0
        kill_reward = (endermen_killed - prev_endermen) * 0.12
        if kill_reward > 0:
            reward += kill_reward
            stats.progressive_rewards += kill_reward

        # Incremental bonus per eye of ender placed in portal frame
        eyes_placed = int(_get_value(state, "portal_frames_filled"))
        if prev_state:
            prev_placed = int(_get_value(prev_state, "portal_frames_filled"))
            placed_delta = eyes_placed - prev_placed
            if placed_delta > 0:
                # 0.1 per eye placed (max 12 eyes = 1.2 total progressive)
                place_reward = placed_delta * 0.1
                reward += place_reward
                stats.progressive_rewards += place_reward

        # Portal activation completion bonus
        if _get_flag(state, "end_portal_activated") and "portal_activated" not in given_rewards:
            reward += 1.5
            given_rewards.add("portal_activated")
            stats.milestone_rewards += 1.5
            stats.milestones_achieved.append("portal_activated")

        # Bonus for nighttime hunting (more endermen spawn)
        time_of_day = _get_value(state, "time_of_day")
        if 13000 <= time_of_day <= 23000:  # Night time
            if "night_hunting" not in given_rewards:
                reward += 0.05
                given_rewards.add("night_hunting")

        # Stage completion
        if _get_flag(state, "stage_complete") and "stage_complete" not in given_rewards:
            reward += 2.0
            given_rewards.add("stage_complete")
            stats.stage_completion_bonus = 2.0

        prev_state.clear()
        prev_state.update(state)
        stats.total_reward += reward
        return reward

    shape_reward.stats = stats  # type: ignore[attr-defined]
    shape_reward.reset = lambda: (given_rewards.clear(), prev_state.clear(), stats.__init__())  # type: ignore[attr-defined]
    return shape_reward


# =============================================================================
# Stage 5: Stronghold Finding
# =============================================================================


def create_stage5_reward_shaper() -> Callable[[dict[str, Any]], float]:
    """Create reward shaper for Stage 5: Stronghold Finding.

    Objectives:
    - Throw eyes of ender to locate stronghold
    - Use triangulation for efficiency
    - Navigate to stronghold
    - Find portal room
    - Activate end portal

    Returns:
        Callable that takes state dict and returns shaped reward.
    """
    given_rewards: set[str] = set()
    prev_state: dict[str, Any] = {}
    stats = RewardStats()
    triangulation_points: list[tuple[float, float]] = []

    def shape_reward(state: dict[str, Any]) -> float:
        nonlocal prev_state
        reward = 0.0

        # Time penalty
        reward -= 0.00012

        # Death penalty
        if _get_value(state, "health") <= 0:
            reward -= 0.8
            stats.penalties += 0.8
            stats.total_reward += reward
            return reward

        # Damage penalty
        if prev_state:
            damage = _get_value(prev_state, "health", 20.0) - _get_value(state, "health", 20.0)
            if damage > 0:
                penalty = damage * 0.015
                reward -= penalty
                stats.penalties += penalty

        # Milestone rewards
        milestones = [
            # Eye throwing
            ("first_eye_throw", _get_flag(state, "eye_thrown"), 0.15),
            ("second_eye_throw", _get_value(state, "eyes_thrown") >= 2, 0.1),
            (
                "triangulation_ready",
                len(triangulation_points) >= 2 or _get_flag(state, "triangulation_used"),
                0.2,
            ),
            # Stronghold discovery
            ("stronghold_located", _get_flag(state, "stronghold_located"), 0.25),
            (
                "stronghold_entered",
                _get_flag(state, "stronghold_found") or _get_flag(state, "in_stronghold"),
                0.3,
            ),
            ("library_found", _get_flag(state, "library_found"), 0.1),
            ("portal_room_found", _get_flag(state, "portal_room_found"), 0.35),
            # Portal activation
            ("first_frame_filled", _get_value(state, "portal_frames_filled") > 0, 0.1),
            ("frames_filled_6", _get_value(state, "portal_frames_filled") >= 6, 0.15),
            ("portal_activated", _get_flag(state, "end_portal_activated"), 0.5),
            # Efficiency bonuses
            (
                "efficient_triangulation",
                _get_flag(state, "triangulation_used") and _get_value(state, "eyes_used") <= 4,
                0.2,
            ),
            (
                "fast_discovery",
                _get_flag(state, "stronghold_found") and _get_value(state, "ticks_elapsed") < 3000,
                0.15,
            ),
        ]

        for name, condition, bonus in milestones:
            if condition and name not in given_rewards:
                reward += bonus
                given_rewards.add(name)
                stats.milestone_rewards += bonus
                stats.milestones_achieved.append(name)

        # Track triangulation points
        if (
            _get_flag(state, "eye_thrown")
            and prev_state
            and not _get_flag(prev_state, "eye_thrown")
        ):
            x = _get_value(state, "x_position")
            z = _get_value(state, "z_position")
            triangulation_points.append((x, z))

        # Distance to stronghold reward (when searching)
        if not _get_flag(state, "stronghold_found"):
            dist_to_stronghold = _get_value(state, "distance_to_stronghold", 2000)
            if prev_state and "distance_to_stronghold" in prev_state:
                prev_dist = _get_value(prev_state, "distance_to_stronghold", 2000)
                if dist_to_stronghold < prev_dist:
                    approach_reward = min((prev_dist - dist_to_stronghold) * 0.0005, 0.03)
                    reward += approach_reward
                    stats.progressive_rewards += approach_reward

        # Portal frame filling progress
        frames_filled = int(_get_value(state, "portal_frames_filled"))

        if prev_state:
            prev_frames = int(_get_value(prev_state, "portal_frames_filled"))
            frame_delta = (frames_filled - prev_frames) * 0.03
            if frame_delta > 0:
                reward += frame_delta
                stats.progressive_rewards += frame_delta

        # Eye conservation bonus
        eyes_remaining = _get_inventory(state, "eye_of_ender")
        if _get_flag(state, "stronghold_found") and eyes_remaining >= 10:
            if "eye_conservation" not in given_rewards:
                reward += 0.1
                given_rewards.add("eye_conservation")

        # Stage completion
        if _get_flag(state, "stage_complete") and "stage_complete" not in given_rewards:
            reward += 2.5
            given_rewards.add("stage_complete")
            stats.stage_completion_bonus = 2.5

        prev_state.clear()
        prev_state.update(state)
        stats.total_reward += reward
        return reward

    shape_reward.stats = stats  # type: ignore[attr-defined]
    shape_reward.reset = lambda: (
        given_rewards.clear(),
        prev_state.clear(),
        triangulation_points.clear(),
        stats.__init__(),
    )  # type: ignore[attr-defined]
    return shape_reward


# =============================================================================
# Stage 6: Dragon Fight
# =============================================================================


def create_stage6_reward_shaper() -> Callable[[dict[str, Any]], float]:
    """Create reward shaper for Stage 6: Ender Dragon Fight.

    Objectives:
    - Enter The End
    - Destroy end crystals
    - Damage the dragon
    - Defeat the dragon
    - Optionally achieve one-cycle kill

    Returns:
        Callable that takes state dict and returns shaped reward.
    """
    given_rewards: set[str] = set()
    prev_state: dict[str, Any] = {}
    stats = RewardStats()
    max_dragon_health_seen = 200.0

    def shape_reward(state: dict[str, Any]) -> float:
        nonlocal prev_state, max_dragon_health_seen
        reward = 0.0

        # Time penalty (high pressure during dragon fight)
        reward -= 0.0002

        # Death penalty (very high - dragon fight deaths are costly)
        if _get_value(state, "health") <= 0:
            reward -= 2.0
            stats.penalties += 2.0
            stats.total_reward += reward
            return reward

        # Damage penalty
        if prev_state:
            damage = _get_value(prev_state, "health", 20.0) - _get_value(state, "health", 20.0)
            if damage > 0:
                # Higher penalty during dragon fight
                penalty = damage * 0.025
                reward -= penalty
                stats.penalties += penalty

        # Void death prevention (heavy penalty for falling)
        if _get_flag(state, "void_below") and _get_value(state, "y_position") < 10:
            reward -= 0.1
            stats.penalties += 0.1

        # Milestone rewards
        milestones = [
            # End entry
            ("entered_end", _get_flag(state, "in_end"), 0.3),
            (
                "dragon_spawned",
                _get_flag(state, "dragon_alive") or _get_value(state, "dragon_health") > 0,
                0.1,
            ),
            # Crystal destruction
            ("first_crystal", _get_value(state, "crystals_destroyed") >= 1, 0.2),
            ("crystals_x3", _get_value(state, "crystals_destroyed") >= 3, 0.15),
            ("crystals_x5", _get_value(state, "crystals_destroyed") >= 5, 0.15),
            ("crystals_x8", _get_value(state, "crystals_destroyed") >= 8, 0.2),
            (
                "all_crystals",
                _get_value(state, "crystals_destroyed") >= 10
                or _get_flag(state, "all_crystals_destroyed"),
                0.3,
            ),
            # Dragon phases
            ("dragon_damaged", _get_value(state, "dragon_health", 200) < 200, 0.15),
            ("dragon_half_health", _get_value(state, "dragon_health", 200) <= 100, 0.2),
            ("dragon_quarter_health", _get_value(state, "dragon_health", 200) <= 50, 0.2),
            ("dragon_critical", _get_value(state, "dragon_health", 200) <= 20, 0.15),
            # Perch attacks
            ("first_perch_hit", _get_flag(state, "hit_dragon_perched"), 0.25),
            ("perch_combo", _get_value(state, "perch_hits") >= 3, 0.2),
            # Victory
            ("dragon_killed", _get_flag(state, "dragon_killed"), 1.0),
            (
                "one_cycle",
                _get_flag(state, "one_cycle")
                or (_get_flag(state, "dragon_killed") and _get_value(state, "perch_count") <= 1),
                0.5,
            ),
            # Speed bonuses
            (
                "fast_kill",
                _get_flag(state, "dragon_killed") and _get_value(state, "fight_ticks") < 3000,
                0.3,
            ),
            (
                "speedrun_pace",
                _get_flag(state, "dragon_killed") and _get_value(state, "total_ticks") < 60000,
                0.5,
            ),
        ]

        for name, condition, bonus in milestones:
            if condition and name not in given_rewards:
                reward += bonus
                given_rewards.add(name)
                stats.milestone_rewards += bonus
                stats.milestones_achieved.append(name)

        # Progressive dragon damage rewards
        dragon_health = _get_value(state, "dragon_health", 200)
        if dragon_health > 0:
            max_dragon_health_seen = max(max_dragon_health_seen, dragon_health)

        if prev_state and "dragon_health" in prev_state:
            prev_dragon_health = _get_value(prev_state, "dragon_health", 200)
            dragon_damage = prev_dragon_health - dragon_health
            if dragon_damage > 0:
                # Scale reward by damage dealt
                damage_reward = dragon_damage * 0.005
                reward += damage_reward
                stats.progressive_rewards += damage_reward

        # Crystal destruction rewards (per crystal)
        crystals_destroyed = int(_get_value(state, "crystals_destroyed"))
        prev_crystals = int(_get_value(prev_state, "crystals_destroyed")) if prev_state else 0
        crystal_reward = (crystals_destroyed - prev_crystals) * 0.15
        if crystal_reward > 0:
            reward += crystal_reward
            stats.progressive_rewards += crystal_reward

        # Positioning rewards
        # Reward being close enough to hit dragon when perching
        if _get_flag(state, "dragon_perching"):
            dragon_dist = _get_value(state, "dragon_distance", 100)
            if dragon_dist < 5:
                reward += 0.02  # Close enough for melee
                stats.progressive_rewards += 0.02
            elif dragon_dist < 15:
                reward += 0.01  # Within bow range
                stats.progressive_rewards += 0.01

        # Penalty for using beds in wrong location (can self-damage)
        if _get_flag(state, "bed_explosion_damage"):
            reward -= 0.2
            stats.penalties += 0.2

        # Bow accuracy reward
        arrows_hit = int(_get_value(state, "arrows_hit_dragon"))
        prev_arrows_hit = int(_get_value(prev_state, "arrows_hit_dragon")) if prev_state else 0
        arrow_reward = (arrows_hit - prev_arrows_hit) * 0.08
        if arrow_reward > 0:
            reward += arrow_reward
            stats.progressive_rewards += arrow_reward

        # Stage completion (victory!)
        if _get_flag(state, "stage_complete") and "stage_complete" not in given_rewards:
            reward += 5.0  # Major reward for winning
            given_rewards.add("stage_complete")
            stats.stage_completion_bonus = 5.0

        prev_state.clear()
        prev_state.update(state)
        stats.total_reward += reward
        return reward

    shape_reward.stats = stats  # type: ignore[attr-defined]
    shape_reward.reset = lambda: (given_rewards.clear(), prev_state.clear(), stats.__init__())  # type: ignore[attr-defined]
    return shape_reward


# =============================================================================
# Factory and Registry
# =============================================================================


REWARD_SHAPERS: dict[int, Callable[[], Callable[[dict[str, Any]], float]]] = {
    1: create_stage1_reward_shaper,
    2: create_stage2_reward_shaper,
    3: create_stage3_reward_shaper,
    4: create_stage4_reward_shaper,
    5: create_stage5_reward_shaper,
    6: create_stage6_reward_shaper,
}


def create_reward_shaper(stage_id: int) -> Callable[[dict[str, Any]], float]:
    """Create a reward shaper for the specified stage.

    Args:
        stage_id: Stage number (1-6).

    Returns:
        Callable that takes state dict and returns shaped reward.

    Raises:
        ValueError: If stage_id is not 1-6.

    Example:
        >>> shaper = create_reward_shaper(1)
        >>> state = {"health": 20, "inventory": {"oak_log": 5}}
        >>> reward = shaper(state)
    """
    if stage_id not in REWARD_SHAPERS:
        raise ValueError(f"Invalid stage_id {stage_id}. Must be 1-6.")
    return REWARD_SHAPERS[stage_id]()


def get_reward_shaper_factory(stage_id: int) -> Callable[[], Callable[[dict[str, Any]], float]]:
    """Get the factory function for a stage's reward shaper.

    Useful when you need to create multiple independent shapers.

    Args:
        stage_id: Stage number (1-6).

    Returns:
        Factory function that creates reward shapers.
    """
    if stage_id not in REWARD_SHAPERS:
        raise ValueError(f"Invalid stage_id {stage_id}. Must be 1-6.")
    return REWARD_SHAPERS[stage_id]


class CompositeRewardShaper:
    """Combines multiple stage shapers for end-to-end training.

    This class manages transitions between stages and provides a unified
    interface for full speedrun training.

    Example:
        >>> composite = CompositeRewardShaper()
        >>> composite.set_stage(1)
        >>> reward = composite.shape_reward(state)
        >>> if state.get("stage_complete"):
        ...     composite.advance_stage()
    """

    def __init__(self, initial_stage: int = 1):
        """Initialize composite reward shaper.

        Args:
            initial_stage: Starting stage (1-6).
        """
        self.current_stage = initial_stage
        self._shapers: dict[int, Callable[[dict[str, Any]], float]] = {}
        self._ensure_shaper(initial_stage)

    def _ensure_shaper(self, stage_id: int) -> None:
        """Ensure a shaper exists for the given stage."""
        if stage_id not in self._shapers:
            self._shapers[stage_id] = create_reward_shaper(stage_id)

    def set_stage(self, stage_id: int) -> None:
        """Set the current stage.

        Args:
            stage_id: Stage to switch to (1-6).
        """
        if stage_id < 1 or stage_id > 6:
            raise ValueError(f"Invalid stage_id {stage_id}. Must be 1-6.")
        self.current_stage = stage_id
        self._ensure_shaper(stage_id)

    def advance_stage(self) -> bool:
        """Advance to the next stage.

        Returns:
            True if advanced, False if already at final stage.
        """
        if self.current_stage >= 6:
            return False
        self.current_stage += 1
        self._ensure_shaper(self.current_stage)
        return True

    def shape_reward(self, state: dict[str, Any]) -> float:
        """Shape reward for current stage.

        Args:
            state: Environment state dictionary.

        Returns:
            Shaped reward value.
        """
        self._ensure_shaper(self.current_stage)
        return self._shapers[self.current_stage](state)

    def reset(self, stage_id: int | None = None) -> None:
        """Reset shaper state.

        Args:
            stage_id: Optionally set stage on reset.
        """
        if stage_id is not None:
            self.set_stage(stage_id)
        # Reset all shapers
        for shaper in self._shapers.values():
            if hasattr(shaper, "reset"):
                shaper.reset()  # type: ignore[attr-defined]

    def get_stats(self) -> RewardStats | None:
        """Get statistics for current stage shaper.

        Returns:
            RewardStats if available, None otherwise.
        """
        shaper = self._shapers.get(self.current_stage)
        if shaper and hasattr(shaper, "stats"):
            return shaper.stats  # type: ignore[attr-defined]
        return None
