"""Success criteria for each curriculum stage in Minecraft speedrun training."""

from collections.abc import Callable
from dataclasses import dataclass, field

import logging

logger = logging.getLogger(__name__)


@dataclass
class StageCriteria:
    """Success criteria for a curriculum stage."""

    stage_id: int
    name: str
    description: str

    # Required conditions (all must be met)
    required: list[Callable[[dict], bool]]

    # Optional conditions (bonus rewards)
    optional: list[Callable[[dict], bool]] = field(default_factory=list)

    # Timeout in ticks (failure if exceeded)
    max_ticks: int = 6000

    # Minimum ticks required (for time-based stages)
    min_ticks: int = 0

    def check_success(self, state: dict) -> bool:
        """Check if all required criteria are met."""
        logger.debug("StageCriteria.check_success: state=%s", state)
        return all(cond(state) for cond in self.required)

    def get_partial_progress(self, state: dict) -> float:
        """Get fraction of required criteria met (0.0 - 1.0)."""
        logger.debug("StageCriteria.get_partial_progress: state=%s", state)
        if not self.required:
            return 1.0
        return sum(1 for cond in self.required if cond(state)) / len(self.required)

    def get_optional_progress(self, state: dict) -> float:
        """Get fraction of optional criteria met (0.0 - 1.0)."""
        logger.debug("StageCriteria.get_optional_progress: state=%s", state)
        if not self.optional:
            return 0.0
        return sum(1 for cond in self.optional if cond(state)) / len(self.optional)


# Helper functions for cleaner lambda definitions
def _has_item(state: dict, item: str, count: int = 1) -> bool:
    logger.debug("_has_item: state=%s, item=%s, count=%s", state, item, count)
    return state.get("inventory", {}).get(item, 0) >= count


def _has_flag(state: dict, flag: str) -> bool:
    logger.debug("_has_flag: state=%s, flag=%s", state, flag)
    return state.get(flag, False)


def _has_count(state: dict, key: str, count: int) -> bool:
    logger.debug("_has_count: state=%s, key=%s, count=%s", state, key, count)
    return state.get(key, 0) >= count


# Stage criteria definitions
STAGE_CRITERIA: dict[int, StageCriteria] = {
    1: StageCriteria(
        stage_id=1,
        name="Basic Survival",
        description="Survive, gather wood, craft basic tools",
        max_ticks=6000,  # 5 minutes
        required=[
            lambda s: _has_item(s, "oak_log", 4),
            lambda s: _has_item(s, "oak_planks", 8) or _has_flag(s, "crafted_planks"),
            lambda s: _has_flag(s, "has_wooden_pickaxe"),
            lambda s: _has_item(s, "cobblestone", 16),
        ],
        optional=[
            lambda s: _has_flag(s, "has_stone_pickaxe"),
            lambda s: _has_count(s, "mobs_killed", 3),
            lambda s: s.get("health", 0) >= 10,
        ],
    ),
    2: StageCriteria(
        stage_id=2,
        name="Resource Gathering",
        description="Mine iron, craft bucket and iron tools",
        max_ticks=12000,  # 10 minutes
        required=[
            lambda s: _has_item(s, "iron_ingot", 3),
            lambda s: _has_flag(s, "has_iron_pickaxe"),
            lambda s: _has_flag(s, "has_bucket") or _has_item(s, "bucket", 1),
        ],
        optional=[
            lambda s: _has_item(s, "diamond", 1),
            lambda s: _has_item(s, "obsidian", 10),
            lambda s: _has_item(s, "iron_ingot", 10),
        ],
    ),
    3: StageCriteria(
        stage_id=3,
        name="Nether Navigation",
        description="Build portal, enter Nether, kill blazes",
        max_ticks=18000,  # 15 minutes
        required=[
            lambda s: _has_flag(s, "portal_built"),
            lambda s: _has_flag(s, "entered_nether"),
            lambda s: _has_flag(s, "fortress_found"),
            lambda s: _has_item(s, "blaze_rod", 7),
        ],
        optional=[
            lambda s: _has_item(s, "blaze_rod", 10),
            lambda s: _has_item(s, "nether_wart", 1),
        ],
    ),
    4: StageCriteria(
        stage_id=4,
        name="Enderman Hunting",
        description="Collect ender pearls for eyes of ender",
        max_ticks=12000,  # 10 minutes
        required=[
            lambda s: _has_item(s, "ender_pearl", 12),
        ],
        optional=[
            lambda s: _has_item(s, "ender_pearl", 16),
            lambda s: _has_count(s, "endermen_killed", 15),
        ],
    ),
    5: StageCriteria(
        stage_id=5,
        name="Stronghold Finding",
        description="Craft eyes, find stronghold, activate portal",
        max_ticks=18000,  # 15 minutes
        required=[
            lambda s: _has_count(s, "eyes_crafted", 12),
            lambda s: _has_flag(s, "stronghold_found"),
            lambda s: _has_flag(s, "end_portal_activated"),
        ],
        optional=[
            lambda s: _has_flag(s, "triangulation_used"),  # Faster than wandering
        ],
    ),
    6: StageCriteria(
        stage_id=6,
        name="Dragon Fight",
        description="Defeat the Ender Dragon",
        max_ticks=18000,  # 15 minutes
        required=[
            lambda s: _has_flag(s, "dragon_killed"),
        ],
        optional=[
            lambda s: _has_flag(s, "one_cycle"),  # Killed in one perch
            lambda s: _has_count(s, "crystals_destroyed", 8),
            lambda s: s.get("ticks_taken", float("inf")) < 6000,  # Under 5 min
        ],
    ),
}


def get_stage_criteria(stage_id: int) -> StageCriteria | None:
    """Get criteria for a specific stage."""
    logger.debug("get_stage_criteria: stage_id=%s", stage_id)
    return STAGE_CRITERIA.get(stage_id)


def get_all_stages() -> list[StageCriteria]:
    """Get all stage criteria in order."""
    logger.debug("get_all_stages called")
    return [STAGE_CRITERIA[i] for i in sorted(STAGE_CRITERIA.keys())]


def calculate_total_progress(state: dict, current_stage: int) -> float:
    """Calculate overall progress across all stages (0.0 - 1.0).

    Completed stages count as 1.0, current stage uses partial progress.
    """
    logger.debug("calculate_total_progress: state=%s, current_stage=%s", state, current_stage)
    total_stages = len(STAGE_CRITERIA)
    completed = current_stage - 1
    current_criteria = STAGE_CRITERIA.get(current_stage)
    current_progress = current_criteria.get_partial_progress(state) if current_criteria else 0.0
    return (completed + current_progress) / total_stages
