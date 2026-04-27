"""Tests for ProgressTracker.update_from_observation with synthetic Stage 1 snapshots.

Verifies that cumulative counters (wood, stone, food, kills) advance correctly
when fed sequential observation dicts representing a Stage 1 survival episode.
"""

from __future__ import annotations

import pytest

from minecraft_sim.progression import ProgressTracker

import logging

logger = logging.getLogger(__name__)


def _make_stage1_obs(
    tick: int,
    *,
    wood: int = 0,
    stone: int = 0,
    zombies_killed: int = 0,
    food_eaten: int = 0,
    health: float = 20.0,
    dimension: int = 0,
    iron_pickaxe: int = 0,
    bucket: int = 0,
) -> dict:
    """Construct a minimal Stage 1 observation dictionary."""
    logger.debug("_make_stage1_obs: tick=%s", tick)
    return {
        "tick_number": tick,
        "player": {"health": health, "dimension": dimension},
        "inventory": {
            "wood": wood,
            "cobblestone": stone,
            "iron_pickaxe": iron_pickaxe,
            "bucket": bucket,
        },
        "stats": {
            "zombies_killed": zombies_killed,
            "food_eaten": food_eaten,
        },
    }


class TestProgressTrackerStage1:
    """Tests for Stage 1 cumulative counter advancement."""

    def test_wood_counter_advances(self) -> None:
        """Wood collected counter increases with inventory growth."""
        logger.debug("TestProgressTrackerStage1.test_wood_counter_advances called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage1_obs(tick=1, wood=4))
        assert tracker.progress.wood_collected == 4

        tracker.update_from_observation(_make_stage1_obs(tick=2, wood=10))
        assert tracker.progress.wood_collected == 10

    def test_wood_counter_does_not_regress(self) -> None:
        """Wood counter holds its peak even if inventory drops."""
        logger.debug("TestProgressTrackerStage1.test_wood_counter_does_not_regress called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage1_obs(tick=1, wood=8))
        tracker.update_from_observation(_make_stage1_obs(tick=2, wood=3))
        assert tracker.progress.wood_collected == 8

    def test_stone_counter_advances(self) -> None:
        """Stone collected counter tracks cobblestone inventory."""
        logger.debug("TestProgressTrackerStage1.test_stone_counter_advances called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage1_obs(tick=1, stone=16))
        assert tracker.progress.stone_collected == 16

        tracker.update_from_observation(_make_stage1_obs(tick=2, stone=40))
        assert tracker.progress.stone_collected == 40

    def test_stone_counter_does_not_regress(self) -> None:
        """Stone counter holds peak value."""
        logger.debug("TestProgressTrackerStage1.test_stone_counter_does_not_regress called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage1_obs(tick=1, stone=32))
        tracker.update_from_observation(_make_stage1_obs(tick=2, stone=10))
        assert tracker.progress.stone_collected == 32

    def test_tick_counter_advances(self) -> None:
        """Total ticks updates from tick_number field."""
        logger.debug("TestProgressTrackerStage1.test_tick_counter_advances called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage1_obs(tick=100))
        assert tracker.progress.total_ticks == 100

        tracker.update_from_observation(_make_stage1_obs(tick=250))
        assert tracker.progress.total_ticks == 250

    def test_stage_time_accumulates(self) -> None:
        """Stage 1 time accumulates across observations."""
        logger.debug("TestProgressTrackerStage1.test_stage_time_accumulates called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage1_obs(tick=50))
        tracker.update_from_observation(_make_stage1_obs(tick=120))
        # First tick delta: 50 - 0 = 50, second: 120 - 50 = 70 => total 120
        assert tracker.progress.stage_times[1] == 120

    def test_death_detection_increments_counter(self) -> None:
        """Deaths detected when health drops to 0."""
        logger.debug("TestProgressTrackerStage1.test_death_detection_increments_counter called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage1_obs(tick=1, health=20.0))
        tracker.update_from_observation(_make_stage1_obs(tick=2, health=0.0))
        assert tracker.progress.deaths == 1
        assert tracker.progress.stage_deaths[1] == 1

    def test_multiple_deaths_accumulate(self) -> None:
        """Multiple deaths accumulate in counter."""
        logger.debug("TestProgressTrackerStage1.test_multiple_deaths_accumulate called")
        tracker = ProgressTracker()
        # First life
        tracker.update_from_observation(_make_stage1_obs(tick=1, health=20.0))
        tracker.update_from_observation(_make_stage1_obs(tick=2, health=0.0))
        # Respawn
        tracker.update_from_observation(_make_stage1_obs(tick=3, health=20.0))
        tracker.update_from_observation(_make_stage1_obs(tick=4, health=0.0))
        assert tracker.progress.deaths == 2

    def test_reward_signals_on_wood_collection(self) -> None:
        """Reward signal emitted when wood counter advances."""
        logger.debug("TestProgressTrackerStage1.test_reward_signals_on_wood_collection called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(_make_stage1_obs(tick=1, wood=5))
        assert "wood_collected" in rewards
        assert rewards["wood_collected"] == pytest.approx(0.5)  # 0.1 * 5

    def test_reward_signals_on_stone_collection(self) -> None:
        """Reward signal emitted when stone counter advances."""
        logger.debug("TestProgressTrackerStage1.test_reward_signals_on_stone_collection called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(_make_stage1_obs(tick=1, stone=10))
        assert "stone_collected" in rewards
        assert rewards["stone_collected"] == pytest.approx(0.5)  # 0.05 * 10

    def test_death_penalty_reward(self) -> None:
        """Death penalty reward emitted on death."""
        logger.debug("TestProgressTrackerStage1.test_death_penalty_reward called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage1_obs(tick=1, health=20.0))
        rewards = tracker.update_from_observation(_make_stage1_obs(tick=2, health=0.0))
        assert rewards["death_penalty"] == -1.0

    def test_no_reward_when_counters_unchanged(self) -> None:
        """No resource rewards when inventory is empty/unchanged."""
        logger.debug("TestProgressTrackerStage1.test_no_reward_when_counters_unchanged called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(_make_stage1_obs(tick=1))
        # Only tick-related updates, no resource rewards
        assert "wood_collected" not in rewards
        assert "stone_collected" not in rewards

    def test_sequential_observations_accumulate(self) -> None:
        """Full sequence of Stage 1 observations accumulates correctly."""
        logger.debug("TestProgressTrackerStage1.test_sequential_observations_accumulate called")
        tracker = ProgressTracker()
        observations = [
            _make_stage1_obs(tick=20, wood=4),
            _make_stage1_obs(tick=60, wood=8, stone=5),
            _make_stage1_obs(tick=100, wood=12, stone=20),
            _make_stage1_obs(tick=200, wood=15, stone=30),
        ]
        for obs in observations:
            tracker.update_from_observation(obs)

        p = tracker.progress
        assert p.wood_collected == 15
        assert p.stone_collected == 30
        assert p.total_ticks == 200
        assert p.deaths == 0
        assert p.current_stage == 1  # Below completion thresholds

    def test_stage_auto_advances_on_completion(self) -> None:
        """Stage advances from 1 to 2 when survival targets met."""
        logger.debug("TestProgressTrackerStage1.test_stage_auto_advances_on_completion called")
        tracker = ProgressTracker()
        # Meet Stage 1 completion: wood >= 16 and stone >= 32
        tracker.update_from_observation(_make_stage1_obs(tick=100, wood=20, stone=40))
        assert tracker.progress.current_stage == 2

    def test_stage_advance_emits_reward(self) -> None:
        """Stage completion emits a reward signal."""
        logger.debug("TestProgressTrackerStage1.test_stage_advance_emits_reward called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(
            _make_stage1_obs(tick=100, wood=20, stone=40)
        )
        assert "stage_1_complete" in rewards
        assert rewards["stage_1_complete"] == 5.0

    def test_iron_pickaxe_tracked(self) -> None:
        """Iron pickaxe boolean set from inventory."""
        logger.debug("TestProgressTrackerStage1.test_iron_pickaxe_tracked called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage1_obs(tick=1, iron_pickaxe=1)
        )
        assert tracker.progress.has_iron_pickaxe is True

    def test_bucket_tracked(self) -> None:
        """Bucket boolean set from inventory."""
        logger.debug("TestProgressTrackerStage1.test_bucket_tracked called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage1_obs(tick=1, bucket=1))
        assert tracker.progress.has_bucket is True

    def test_initial_state_is_zero(self) -> None:
        """Fresh tracker has all counters at zero."""
        logger.info("TestProgressTrackerStage1.test_initial_state_is_zero called")
        tracker = ProgressTracker()
        p = tracker.progress
        assert p.wood_collected == 0
        assert p.stone_collected == 0
        assert p.total_ticks == 0
        assert p.deaths == 0
        assert p.current_stage == 1

    def test_dimension_change_does_not_affect_stage1(self) -> None:
        """Dimension changes tracked but don't interfere with Stage 1 counters."""
        logger.debug("TestProgressTrackerStage1.test_dimension_change_does_not_affect_stage1 called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage1_obs(tick=1, wood=5))
        # Dimension stays 0 (overworld) throughout Stage 1
        assert tracker.prev_dimension == 0
        assert tracker.progress.entered_nether is False


def _make_stage2_obs(
    tick: int,
    *,
    iron_ore: int = 0,
    iron_ingots: int = 0,
    diamonds: int = 0,
    iron_pickaxe: int = 0,
    bucket: int = 0,
    water_buckets: int = 0,
    lava_buckets: int = 0,
    wood: int = 20,
    stone: int = 40,
    health: float = 20.0,
    dimension: int = 0,
) -> dict:
    """Construct a minimal Stage 2 observation dictionary.

    Defaults wood/stone above Stage 1 thresholds so the tracker auto-advances
    to Stage 2 when needed.
    """
    logger.debug("_make_stage2_obs: tick=%s", tick)
    return {
        "tick_number": tick,
        "player": {"health": health, "dimension": dimension},
        "inventory": {
            "wood": wood,
            "cobblestone": stone,
            "iron_ore": iron_ore,
            "iron_ingots": iron_ingots,
            "diamonds": diamonds,
            "iron_pickaxe": iron_pickaxe,
            "bucket": bucket,
            "water_buckets": water_buckets,
            "lava_buckets": lava_buckets,
        },
    }


class TestProgressTrackerStage2:
    """Tests for Stage 2 resource counter and flag advancement."""

    def test_iron_ore_counter_advances(self) -> None:
        """Iron ore mined counter increases with inventory growth."""
        logger.debug("TestProgressTrackerStage2.test_iron_ore_counter_advances called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, iron_ore=3))
        assert tracker.progress.iron_ore_mined == 3

        tracker.update_from_observation(_make_stage2_obs(tick=2, iron_ore=8))
        assert tracker.progress.iron_ore_mined == 8

    def test_iron_ore_counter_does_not_regress(self) -> None:
        """Iron ore counter holds peak value if inventory drops."""
        logger.debug("TestProgressTrackerStage2.test_iron_ore_counter_does_not_regress called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, iron_ore=6))
        tracker.update_from_observation(_make_stage2_obs(tick=2, iron_ore=2))
        assert tracker.progress.iron_ore_mined == 6

    def test_iron_ingots_counter_advances(self) -> None:
        """Iron ingots counter tracks smelted iron in inventory."""
        logger.debug("TestProgressTrackerStage2.test_iron_ingots_counter_advances called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, iron_ingots=4))
        assert tracker.progress.iron_ingots == 4

        tracker.update_from_observation(_make_stage2_obs(tick=2, iron_ingots=10))
        assert tracker.progress.iron_ingots == 10

    def test_iron_ingots_counter_does_not_regress(self) -> None:
        """Iron ingots counter holds peak value."""
        logger.debug("TestProgressTrackerStage2.test_iron_ingots_counter_does_not_regress called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, iron_ingots=7))
        tracker.update_from_observation(_make_stage2_obs(tick=2, iron_ingots=3))
        assert tracker.progress.iron_ingots == 7

    def test_iron_ingot_alternate_key(self) -> None:
        """Iron ingots accepted via 'iron_ingot' key (singular)."""
        logger.debug("TestProgressTrackerStage2.test_iron_ingot_alternate_key called")
        tracker = ProgressTracker()
        obs = {
            "tick_number": 1,
            "player": {"health": 20.0, "dimension": 0},
            "inventory": {"iron_ingot": 5},
        }
        tracker.update_from_observation(obs)
        assert tracker.progress.iron_ingots == 5

    def test_diamonds_counter_advances(self) -> None:
        """Diamond counter tracks diamond inventory."""
        logger.debug("TestProgressTrackerStage2.test_diamonds_counter_advances called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, diamonds=2))
        assert tracker.progress.diamonds == 2

        tracker.update_from_observation(_make_stage2_obs(tick=2, diamonds=5))
        assert tracker.progress.diamonds == 5

    def test_diamonds_counter_does_not_regress(self) -> None:
        """Diamond counter holds peak value."""
        logger.debug("TestProgressTrackerStage2.test_diamonds_counter_does_not_regress called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, diamonds=4))
        tracker.update_from_observation(_make_stage2_obs(tick=2, diamonds=1))
        assert tracker.progress.diamonds == 4

    def test_diamond_alternate_key(self) -> None:
        """Diamonds accepted via 'diamond' key (singular)."""
        logger.debug("TestProgressTrackerStage2.test_diamond_alternate_key called")
        tracker = ProgressTracker()
        obs = {
            "tick_number": 1,
            "player": {"health": 20.0, "dimension": 0},
            "inventory": {"diamond": 3},
        }
        tracker.update_from_observation(obs)
        assert tracker.progress.diamonds == 3

    def test_iron_pickaxe_flag_set(self) -> None:
        """has_iron_pickaxe flag set when iron_pickaxe > 0 in inventory."""
        logger.debug("TestProgressTrackerStage2.test_iron_pickaxe_flag_set called")
        tracker = ProgressTracker()
        assert tracker.progress.has_iron_pickaxe is False
        tracker.update_from_observation(_make_stage2_obs(tick=1, iron_pickaxe=1))
        assert tracker.progress.has_iron_pickaxe is True

    def test_iron_pickaxe_flag_stays_true(self) -> None:
        """has_iron_pickaxe remains True even if pickaxe leaves inventory."""
        logger.debug("TestProgressTrackerStage2.test_iron_pickaxe_flag_stays_true called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, iron_pickaxe=1))
        tracker.update_from_observation(_make_stage2_obs(tick=2, iron_pickaxe=0))
        assert tracker.progress.has_iron_pickaxe is True

    def test_bucket_flag_set_from_bucket_key(self) -> None:
        """has_bucket flag set via 'bucket' inventory key."""
        logger.debug("TestProgressTrackerStage2.test_bucket_flag_set_from_bucket_key called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, bucket=1))
        assert tracker.progress.has_bucket is True

    def test_bucket_flag_set_from_water_bucket(self) -> None:
        """has_bucket flag set via water_buckets inventory key."""
        logger.debug("TestProgressTrackerStage2.test_bucket_flag_set_from_water_bucket called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, water_buckets=1))
        assert tracker.progress.has_bucket is True

    def test_bucket_flag_set_from_lava_bucket(self) -> None:
        """has_bucket flag set via lava_buckets inventory key."""
        logger.debug("TestProgressTrackerStage2.test_bucket_flag_set_from_lava_bucket called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, lava_buckets=1))
        assert tracker.progress.has_bucket is True

    def test_bucket_flag_stays_true(self) -> None:
        """has_bucket remains True even if bucket leaves inventory."""
        logger.debug("TestProgressTrackerStage2.test_bucket_flag_stays_true called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, bucket=1))
        tracker.update_from_observation(_make_stage2_obs(tick=2, bucket=0))
        assert tracker.progress.has_bucket is True

    def test_iron_ore_reward_signal(self) -> None:
        """Reward emitted when iron ore counter advances."""
        logger.debug("TestProgressTrackerStage2.test_iron_ore_reward_signal called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(_make_stage2_obs(tick=1, iron_ore=4))
        assert "iron_ore_mined" in rewards
        assert rewards["iron_ore_mined"] == pytest.approx(1.2)  # 0.3 * 4

    def test_iron_ingots_reward_signal(self) -> None:
        """Reward emitted when iron ingots counter advances."""
        logger.debug("TestProgressTrackerStage2.test_iron_ingots_reward_signal called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(
            _make_stage2_obs(tick=1, iron_ingots=3)
        )
        assert "iron_collected" in rewards
        assert rewards["iron_collected"] == pytest.approx(1.5)  # 0.5 * 3

    def test_diamond_reward_signal(self) -> None:
        """Reward emitted when diamond counter advances."""
        logger.debug("TestProgressTrackerStage2.test_diamond_reward_signal called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(_make_stage2_obs(tick=1, diamonds=2))
        assert "diamond_collected" in rewards
        assert rewards["diamond_collected"] == pytest.approx(4.0)  # 2.0 * 2

    def test_iron_pickaxe_crafted_reward(self) -> None:
        """Reward emitted when iron pickaxe first appears."""
        logger.debug("TestProgressTrackerStage2.test_iron_pickaxe_crafted_reward called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(
            _make_stage2_obs(tick=1, iron_pickaxe=1)
        )
        assert "iron_pickaxe_crafted" in rewards
        assert rewards["iron_pickaxe_crafted"] == 2.0

    def test_iron_pickaxe_no_duplicate_reward(self) -> None:
        """No duplicate reward for iron pickaxe already tracked."""
        logger.debug("TestProgressTrackerStage2.test_iron_pickaxe_no_duplicate_reward called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, iron_pickaxe=1))
        rewards = tracker.update_from_observation(
            _make_stage2_obs(tick=2, iron_pickaxe=1)
        )
        assert "iron_pickaxe_crafted" not in rewards

    def test_bucket_crafted_reward(self) -> None:
        """Reward emitted when bucket first appears."""
        logger.debug("TestProgressTrackerStage2.test_bucket_crafted_reward called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(_make_stage2_obs(tick=1, bucket=1))
        assert "bucket_crafted" in rewards
        assert rewards["bucket_crafted"] == 1.0

    def test_bucket_no_duplicate_reward(self) -> None:
        """No duplicate reward for bucket already tracked."""
        logger.debug("TestProgressTrackerStage2.test_bucket_no_duplicate_reward called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, bucket=1))
        rewards = tracker.update_from_observation(_make_stage2_obs(tick=2, bucket=1))
        assert "bucket_crafted" not in rewards

    def test_incremental_iron_ore_reward(self) -> None:
        """Reward only reflects the delta, not the total."""
        logger.debug("TestProgressTrackerStage2.test_incremental_iron_ore_reward called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage2_obs(tick=1, iron_ore=3))
        rewards = tracker.update_from_observation(_make_stage2_obs(tick=2, iron_ore=5))
        assert rewards["iron_ore_mined"] == pytest.approx(0.6)  # 0.3 * (5-3)

    def test_no_resource_rewards_when_unchanged(self) -> None:
        """No Stage 2 resource rewards when inventory is empty/unchanged."""
        logger.debug("TestProgressTrackerStage2.test_no_resource_rewards_when_unchanged called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(_make_stage2_obs(tick=1))
        assert "iron_ore_mined" not in rewards
        assert "iron_collected" not in rewards
        assert "diamond_collected" not in rewards
        assert "iron_pickaxe_crafted" not in rewards
        assert "bucket_crafted" not in rewards

    def test_stage2_sequential_accumulation(self) -> None:
        """Full sequence of Stage 2 observations accumulates correctly."""
        logger.debug("TestProgressTrackerStage2.test_stage2_sequential_accumulation called")
        tracker = ProgressTracker()
        observations = [
            _make_stage2_obs(tick=100, iron_ore=5),
            _make_stage2_obs(tick=200, iron_ore=8, iron_ingots=3),
            _make_stage2_obs(tick=300, iron_ingots=6, iron_pickaxe=1),
            _make_stage2_obs(tick=400, iron_ingots=10, diamonds=3, bucket=1),
        ]
        for obs in observations:
            tracker.update_from_observation(obs)

        p = tracker.progress
        assert p.iron_ore_mined == 8
        assert p.iron_ingots == 10
        assert p.diamonds == 3
        assert p.has_iron_pickaxe is True
        assert p.has_bucket is True

    def test_stage2_completion_advances_to_stage3(self) -> None:
        """Stage advances from 2 to 3 when resource targets met."""
        logger.debug("TestProgressTrackerStage2.test_stage2_completion_advances_to_stage3 called")
        tracker = ProgressTracker()
        # First meet Stage 1 completion thresholds
        tracker.update_from_observation(_make_stage2_obs(tick=50))
        assert tracker.progress.current_stage == 2

        # Now meet Stage 2 completion: iron_pickaxe + bucket + iron_ingots >= 3
        tracker.update_from_observation(
            _make_stage2_obs(tick=100, iron_ingots=5, iron_pickaxe=1, bucket=1)
        )
        assert tracker.progress.current_stage == 3

    def test_stage2_completion_reward(self) -> None:
        """Stage 2 completion emits reward signal."""
        logger.debug("TestProgressTrackerStage2.test_stage2_completion_reward called")
        tracker = ProgressTracker()
        # Auto-advance to Stage 2
        tracker.update_from_observation(_make_stage2_obs(tick=50))
        # Complete Stage 2
        rewards = tracker.update_from_observation(
            _make_stage2_obs(tick=100, iron_ingots=5, iron_pickaxe=1, bucket=1)
        )
        assert "stage_2_complete" in rewards
        assert rewards["stage_2_complete"] == 5.0

    def test_stage2_death_tracked_in_stage_deaths(self) -> None:
        """Deaths during Stage 2 increment stage_deaths[2]."""
        logger.debug("TestProgressTrackerStage2.test_stage2_death_tracked_in_stage_deaths called")
        tracker = ProgressTracker()
        # Enter Stage 2
        tracker.update_from_observation(_make_stage2_obs(tick=50, health=20.0))
        assert tracker.progress.current_stage == 2

        # Die in Stage 2
        tracker.update_from_observation(_make_stage2_obs(tick=60, health=0.0))
        assert tracker.progress.stage_deaths[2] == 1
        assert tracker.progress.deaths == 1

    def test_stage2_time_accumulates(self) -> None:
        """Time spent in Stage 2 accumulates in stage_times[2]."""
        logger.debug("TestProgressTrackerStage2.test_stage2_time_accumulates called")
        tracker = ProgressTracker()
        # Enter Stage 2 at tick 50
        tracker.update_from_observation(_make_stage2_obs(tick=50))
        assert tracker.progress.current_stage == 2
        # Continue in Stage 2
        tracker.update_from_observation(_make_stage2_obs(tick=150, iron_ore=3))
        assert tracker.progress.stage_times[2] == 100  # 150 - 50


def _make_stage4_obs(
    tick: int,
    *,
    ender_pearls: int = 0,
    eyes_of_ender: int = 0,
    health: float = 20.0,
    dimension: int = 0,
) -> dict:
    """Construct a minimal Stage 4/5 observation dictionary."""
    logger.debug("_make_stage4_obs: tick=%s", tick)
    return {
        "tick_number": tick,
        "player": {"health": health, "dimension": dimension},
        "inventory": {
            "ender_pearls": ender_pearls,
            "eyes_of_ender": eyes_of_ender,
        },
    }


class TestProgressTrackerStage4Observations:
    """Regression tests for Stage 4/5 observation vector updates.

    Verifies that portal activation, eye counts, and ender pearl tracking
    produce correct values in the observation vector returned by to_observation().
    """

    # --- Observation vector index reference ---
    # Stage 4 (indices 15-18): endermen_killed/15, ender_pearls/16,
    #   nether_wart/16, piglins_bartered/20
    # Stage 5 (indices 19-23): eyes_crafted/12, stronghold_found,
    #   portal_room_found, eyes_placed/12, portal_activated

    IDX_ENDERMEN_KILLED = 15
    IDX_ENDER_PEARLS = 16
    IDX_NETHER_WART = 17
    IDX_PIGLINS_BARTERED = 18
    IDX_EYES_CRAFTED = 19
    IDX_STRONGHOLD_FOUND = 20
    IDX_PORTAL_ROOM_FOUND = 21
    IDX_EYES_PLACED = 22
    IDX_PORTAL_ACTIVATED = 23

    def test_ender_pearls_observation_updates(self) -> None:
        """Ender pearl count reflected in observation vector at index 16."""
        logger.debug("TestProgressTrackerStage4Observations.test_ender_pearls_observation_updates called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage4_obs(tick=1, ender_pearls=6))
        obs = tracker.progress.to_observation()
        # 6 / 16.0 = 0.375
        assert obs[self.IDX_ENDER_PEARLS] == pytest.approx(6.0 / 16.0)

    def test_ender_pearls_observation_clamped_at_one(self) -> None:
        """Ender pearl observation saturates at 1.0 when exceeding normalizer."""
        logger.debug("TestProgressTrackerStage4Observations.test_ender_pearls_observation_clamped_at_one called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage4_obs(tick=1, ender_pearls=20))
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_ENDER_PEARLS] == pytest.approx(1.0)

    def test_ender_pearls_observation_zero_initially(self) -> None:
        """Ender pearl observation is 0.0 with no pearls collected."""
        logger.info("TestProgressTrackerStage4Observations.test_ender_pearls_observation_zero_initially called")
        tracker = ProgressTracker()
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_ENDER_PEARLS] == 0.0

    def test_ender_pearls_observation_monotonic(self) -> None:
        """Ender pearl observation never decreases across updates."""
        logger.debug("TestProgressTrackerStage4Observations.test_ender_pearls_observation_monotonic called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage4_obs(tick=1, ender_pearls=8))
        obs1 = tracker.progress.to_observation()

        # Inventory drops but tracker holds peak
        tracker.update_from_observation(_make_stage4_obs(tick=2, ender_pearls=3))
        obs2 = tracker.progress.to_observation()

        assert obs2[self.IDX_ENDER_PEARLS] >= obs1[self.IDX_ENDER_PEARLS]

    def test_eyes_crafted_observation_updates(self) -> None:
        """Eyes of ender crafted reflected in observation at index 19."""
        logger.debug("TestProgressTrackerStage4Observations.test_eyes_crafted_observation_updates called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage4_obs(tick=1, eyes_of_ender=4))
        obs = tracker.progress.to_observation()
        # 4 / 12.0 = 0.333...
        assert obs[self.IDX_EYES_CRAFTED] == pytest.approx(4.0 / 12.0)

    def test_eyes_crafted_observation_full(self) -> None:
        """Eyes crafted observation saturates at 1.0 with 12+ eyes."""
        logger.debug("TestProgressTrackerStage4Observations.test_eyes_crafted_observation_full called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage4_obs(tick=1, eyes_of_ender=12))
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_EYES_CRAFTED] == pytest.approx(1.0)

    def test_eyes_crafted_observation_exceeds_twelve(self) -> None:
        """Eyes crafted observation clamps at 1.0 even with >12 eyes."""
        logger.debug("TestProgressTrackerStage4Observations.test_eyes_crafted_observation_exceeds_twelve called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage4_obs(tick=1, eyes_of_ender=15))
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_EYES_CRAFTED] == pytest.approx(1.0)

    def test_portal_activated_observation_false_initially(self) -> None:
        """Portal activated observation is 0.0 when not activated."""
        logger.info("TestProgressTrackerStage4Observations.test_portal_activated_observation_false_initially called")
        tracker = ProgressTracker()
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_PORTAL_ACTIVATED] == 0.0

    def test_portal_activated_observation_set_directly(self) -> None:
        """Portal activated observation becomes 1.0 when flag set."""
        logger.debug("TestProgressTrackerStage4Observations.test_portal_activated_observation_set_directly called")
        tracker = ProgressTracker()
        tracker.progress.portal_activated = True
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_PORTAL_ACTIVATED] == pytest.approx(1.0)

    def test_eyes_placed_observation_updates(self) -> None:
        """Eyes placed observation reflects normalized count at index 22."""
        logger.debug("TestProgressTrackerStage4Observations.test_eyes_placed_observation_updates called")
        tracker = ProgressTracker()
        tracker.progress.eyes_placed = 7
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_EYES_PLACED] == pytest.approx(7.0 / 12.0)

    def test_eyes_placed_full_activates_portal_completion(self) -> None:
        """12 eyes placed with portal_room_found completes Stage 5."""
        logger.debug("TestProgressTrackerStage4Observations.test_eyes_placed_full_activates_portal_completion called")
        tracker = ProgressTracker()
        tracker.progress.portal_room_found = True
        tracker.progress.eyes_placed = 12
        assert tracker.progress.is_stage_complete(5) is True

    def test_portal_activated_with_full_eyes_in_observation(self) -> None:
        """Full portal activation state reflected across all Stage 5 obs indices."""
        logger.debug("TestProgressTrackerStage4Observations.test_portal_activated_with_full_eyes_in_observation called")
        tracker = ProgressTracker()
        tracker.progress.eyes_crafted = 12
        tracker.progress.stronghold_found = True
        tracker.progress.portal_room_found = True
        tracker.progress.eyes_placed = 12
        tracker.progress.portal_activated = True

        obs = tracker.progress.to_observation()
        assert obs[self.IDX_EYES_CRAFTED] == pytest.approx(1.0)
        assert obs[self.IDX_STRONGHOLD_FOUND] == pytest.approx(1.0)
        assert obs[self.IDX_PORTAL_ROOM_FOUND] == pytest.approx(1.0)
        assert obs[self.IDX_EYES_PLACED] == pytest.approx(1.0)
        assert obs[self.IDX_PORTAL_ACTIVATED] == pytest.approx(1.0)

    def test_observation_vector_shape(self) -> None:
        """Observation vector has expected shape (32,) for Stage 4/5 data."""
        logger.debug("TestProgressTrackerStage4Observations.test_observation_vector_shape called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage4_obs(tick=1, ender_pearls=5, eyes_of_ender=3))
        obs = tracker.progress.to_observation()
        assert obs.shape == (32,)

    def test_stage4_fields_independent_of_stage5(self) -> None:
        """Stage 4 pearl observations don't affect Stage 5 eye/portal observations."""
        logger.debug("TestProgressTrackerStage4Observations.test_stage4_fields_independent_of_stage5 called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage4_obs(tick=1, ender_pearls=12))
        obs = tracker.progress.to_observation()

        # Stage 4 updated
        assert obs[self.IDX_ENDER_PEARLS] == pytest.approx(12.0 / 16.0)
        # Stage 5 unchanged
        assert obs[self.IDX_EYES_CRAFTED] == 0.0
        assert obs[self.IDX_PORTAL_ACTIVATED] == 0.0

    def test_sequential_pearl_then_eye_updates(self) -> None:
        """Sequential observations updating pearls then eyes produce correct vector."""
        logger.debug("TestProgressTrackerStage4Observations.test_sequential_pearl_then_eye_updates called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage4_obs(tick=1, ender_pearls=10))
        tracker.update_from_observation(_make_stage4_obs(tick=2, ender_pearls=10, eyes_of_ender=6))

        obs = tracker.progress.to_observation()
        assert obs[self.IDX_ENDER_PEARLS] == pytest.approx(10.0 / 16.0)
        assert obs[self.IDX_EYES_CRAFTED] == pytest.approx(6.0 / 12.0)

    def test_endermen_killed_observation_updates(self) -> None:
        """Endermen killed reflected in observation when set directly."""
        logger.debug("TestProgressTrackerStage4Observations.test_endermen_killed_observation_updates called")
        tracker = ProgressTracker()
        tracker.progress.endermen_killed = 9
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_ENDERMEN_KILLED] == pytest.approx(9.0 / 15.0)

    def test_endermen_killed_clamped(self) -> None:
        """Endermen killed observation clamped at 1.0."""
        logger.debug("TestProgressTrackerStage4Observations.test_endermen_killed_clamped called")
        tracker = ProgressTracker()
        tracker.progress.endermen_killed = 20
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_ENDERMEN_KILLED] == pytest.approx(1.0)


def _make_stage3_obs(
    tick: int,
    *,
    wood: int = 20,
    stone: int = 40,
    iron_ingots: int = 5,
    iron_pickaxe: int = 1,
    bucket: int = 1,
    obsidian: int = 0,
    blaze_rods: int = 0,
    health: float = 20.0,
    dimension: int = 0,
    portal_built: bool = False,
    entered_nether: bool = False,
    fortress_found: bool = False,
    blazes_killed: int = 0,
) -> dict:
    """Construct a minimal Stage 3 observation dictionary.

    Defaults wood/stone/iron/tools above Stage 1 and 2 thresholds so the
    tracker auto-advances to Stage 3 when needed.
    """
    logger.debug("_make_stage3_obs: tick=%s", tick)
    obs: dict = {
        "tick_number": tick,
        "player": {"health": health, "dimension": dimension},
        "inventory": {
            "wood": wood,
            "cobblestone": stone,
            "iron_ingots": iron_ingots,
            "iron_pickaxe": iron_pickaxe,
            "bucket": bucket,
            "obsidian": obsidian,
            "blaze_rods": blaze_rods,
        },
    }
    if portal_built or entered_nether or fortress_found or blazes_killed > 0:
        obs["nether"] = {
            "portal_built": portal_built,
            "entered_nether": entered_nether,
            "fortress_found": fortress_found,
            "blazes_killed": blazes_killed,
        }
    return obs


class TestProgressTrackerStage3:
    """Tests for Stage 3 nether objective boolean flags and counters."""

    def test_portal_built_flag_set_from_nether_dict(self) -> None:
        """portal_built flag set when nether.portal_built is True."""
        logger.debug("TestProgressTrackerStage3.test_portal_built_flag_set_from_nether_dict called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, portal_built=True)
        )
        assert tracker.progress.portal_built is True

    def test_portal_built_flag_stays_true(self) -> None:
        """portal_built remains True once set."""
        logger.debug("TestProgressTrackerStage3.test_portal_built_flag_stays_true called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, portal_built=True)
        )
        tracker.update_from_observation(_make_stage3_obs(tick=2))
        assert tracker.progress.portal_built is True

    def test_entered_nether_from_nether_dict(self) -> None:
        """entered_nether set from nether.entered_nether field."""
        logger.debug("TestProgressTrackerStage3.test_entered_nether_from_nether_dict called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, entered_nether=True)
        )
        assert tracker.progress.entered_nether is True

    def test_entered_nether_from_dimension_change(self) -> None:
        """entered_nether set when player dimension changes to 1 (nether)."""
        logger.debug("TestProgressTrackerStage3.test_entered_nether_from_dimension_change called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage3_obs(tick=1, dimension=0))
        tracker.update_from_observation(_make_stage3_obs(tick=2, dimension=1))
        assert tracker.progress.entered_nether is True

    def test_entered_nether_stays_true(self) -> None:
        """entered_nether remains True after returning to overworld."""
        logger.debug("TestProgressTrackerStage3.test_entered_nether_stays_true called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage3_obs(tick=1, dimension=0))
        tracker.update_from_observation(_make_stage3_obs(tick=2, dimension=1))
        tracker.update_from_observation(_make_stage3_obs(tick=3, dimension=0))
        assert tracker.progress.entered_nether is True

    def test_fortress_found_flag_set(self) -> None:
        """fortress_found flag set from nether.fortress_found."""
        logger.debug("TestProgressTrackerStage3.test_fortress_found_flag_set called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, fortress_found=True)
        )
        assert tracker.progress.fortress_found is True

    def test_fortress_found_stays_true(self) -> None:
        """fortress_found remains True once set."""
        logger.debug("TestProgressTrackerStage3.test_fortress_found_stays_true called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, fortress_found=True)
        )
        tracker.update_from_observation(_make_stage3_obs(tick=2))
        assert tracker.progress.fortress_found is True

    def test_blazes_killed_counter_advances(self) -> None:
        """blazes_killed counter increases from nether.blazes_killed."""
        logger.debug("TestProgressTrackerStage3.test_blazes_killed_counter_advances called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, blazes_killed=3)
        )
        assert tracker.progress.blazes_killed == 3

        tracker.update_from_observation(
            _make_stage3_obs(tick=2, blazes_killed=7)
        )
        assert tracker.progress.blazes_killed == 7

    def test_blazes_killed_does_not_regress(self) -> None:
        """blazes_killed counter holds peak value."""
        logger.debug("TestProgressTrackerStage3.test_blazes_killed_does_not_regress called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, blazes_killed=5)
        )
        tracker.update_from_observation(
            _make_stage3_obs(tick=2, blazes_killed=2)
        )
        assert tracker.progress.blazes_killed == 5

    def test_blaze_rods_counter_advances(self) -> None:
        """blaze_rods counter increases from inventory."""
        logger.debug("TestProgressTrackerStage3.test_blaze_rods_counter_advances called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, blaze_rods=4)
        )
        assert tracker.progress.blaze_rods == 4

        tracker.update_from_observation(
            _make_stage3_obs(tick=2, blaze_rods=8)
        )
        assert tracker.progress.blaze_rods == 8

    def test_blaze_rods_does_not_regress(self) -> None:
        """blaze_rods counter holds peak value."""
        logger.debug("TestProgressTrackerStage3.test_blaze_rods_does_not_regress called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, blaze_rods=6)
        )
        tracker.update_from_observation(
            _make_stage3_obs(tick=2, blaze_rods=2)
        )
        assert tracker.progress.blaze_rods == 6

    def test_obsidian_counter_advances(self) -> None:
        """obsidian_collected counter increases from inventory."""
        logger.debug("TestProgressTrackerStage3.test_obsidian_counter_advances called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, obsidian=10)
        )
        assert tracker.progress.obsidian_collected == 10

        tracker.update_from_observation(
            _make_stage3_obs(tick=2, obsidian=14)
        )
        assert tracker.progress.obsidian_collected == 14

    def test_obsidian_counter_does_not_regress(self) -> None:
        """obsidian_collected counter holds peak value."""
        logger.debug("TestProgressTrackerStage3.test_obsidian_counter_does_not_regress called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, obsidian=12)
        )
        tracker.update_from_observation(
            _make_stage3_obs(tick=2, obsidian=4)
        )
        assert tracker.progress.obsidian_collected == 12

    def test_portal_built_reward(self) -> None:
        """Reward emitted when portal_built first set."""
        logger.debug("TestProgressTrackerStage3.test_portal_built_reward called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(
            _make_stage3_obs(tick=1, portal_built=True)
        )
        assert "portal_built" in rewards
        assert rewards["portal_built"] == 5.0

    def test_portal_built_no_duplicate_reward(self) -> None:
        """No duplicate portal_built reward on subsequent observations."""
        logger.debug("TestProgressTrackerStage3.test_portal_built_no_duplicate_reward called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, portal_built=True)
        )
        rewards = tracker.update_from_observation(
            _make_stage3_obs(tick=2, portal_built=True)
        )
        assert "portal_built" not in rewards

    def test_entered_nether_reward_from_dimension(self) -> None:
        """Reward emitted when entering nether via dimension change."""
        logger.debug("TestProgressTrackerStage3.test_entered_nether_reward_from_dimension called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage3_obs(tick=1, dimension=0))
        rewards = tracker.update_from_observation(
            _make_stage3_obs(tick=2, dimension=1)
        )
        assert "entered_nether" in rewards
        assert rewards["entered_nether"] == 10.0

    def test_entered_nether_no_duplicate_reward(self) -> None:
        """No duplicate entered_nether reward on re-entry."""
        logger.debug("TestProgressTrackerStage3.test_entered_nether_no_duplicate_reward called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage3_obs(tick=1, dimension=0))
        tracker.update_from_observation(_make_stage3_obs(tick=2, dimension=1))
        # Return to overworld and re-enter
        tracker.update_from_observation(_make_stage3_obs(tick=3, dimension=0))
        rewards = tracker.update_from_observation(
            _make_stage3_obs(tick=4, dimension=1)
        )
        assert "entered_nether" not in rewards

    def test_fortress_found_reward(self) -> None:
        """Reward emitted when fortress_found first set."""
        logger.debug("TestProgressTrackerStage3.test_fortress_found_reward called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(
            _make_stage3_obs(tick=1, fortress_found=True)
        )
        assert "fortress_found" in rewards
        assert rewards["fortress_found"] == 8.0

    def test_blaze_killed_reward(self) -> None:
        """Reward emitted when blazes_killed counter advances."""
        logger.debug("TestProgressTrackerStage3.test_blaze_killed_reward called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(
            _make_stage3_obs(tick=1, blazes_killed=2)
        )
        assert "blaze_killed" in rewards
        assert rewards["blaze_killed"] == pytest.approx(6.0)  # 3.0 * 2

    def test_blaze_killed_incremental_reward(self) -> None:
        """Blaze kill reward reflects only the delta."""
        logger.debug("TestProgressTrackerStage3.test_blaze_killed_incremental_reward called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, blazes_killed=3)
        )
        rewards = tracker.update_from_observation(
            _make_stage3_obs(tick=2, blazes_killed=5)
        )
        assert rewards["blaze_killed"] == pytest.approx(6.0)  # 3.0 * (5-3)

    def test_blaze_rod_collected_reward(self) -> None:
        """Reward emitted when blaze_rods inventory advances."""
        logger.debug("TestProgressTrackerStage3.test_blaze_rod_collected_reward called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(
            _make_stage3_obs(tick=1, blaze_rods=3)
        )
        assert "blaze_rod_collected" in rewards
        assert rewards["blaze_rod_collected"] == pytest.approx(9.0)  # 3.0 * 3

    def test_blaze_rod_incremental_reward(self) -> None:
        """Blaze rod reward reflects only the delta."""
        logger.debug("TestProgressTrackerStage3.test_blaze_rod_incremental_reward called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, blaze_rods=2)
        )
        rewards = tracker.update_from_observation(
            _make_stage3_obs(tick=2, blaze_rods=5)
        )
        assert rewards["blaze_rod_collected"] == pytest.approx(9.0)  # 3.0 * (5-2)

    def test_obsidian_collected_reward(self) -> None:
        """Reward emitted when obsidian inventory advances."""
        logger.debug("TestProgressTrackerStage3.test_obsidian_collected_reward called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(
            _make_stage3_obs(tick=1, obsidian=10)
        )
        assert "obsidian_collected" in rewards
        assert rewards["obsidian_collected"] == pytest.approx(5.0)  # 0.5 * 10

    def test_stage3_completion_condition(self) -> None:
        """Stage 3 complete when entered_nether and blaze_rods >= 7."""
        logger.debug("TestProgressTrackerStage3.test_stage3_completion_condition called")
        tracker = ProgressTracker()
        # Auto-advance to stage 3 (needs stage 1 + 2 complete)
        tracker.update_from_observation(_make_stage3_obs(tick=50))
        assert tracker.progress.current_stage == 3

        # Enter nether but not enough blaze rods
        tracker.update_from_observation(
            _make_stage3_obs(tick=100, dimension=1, blaze_rods=5)
        )
        assert tracker.progress.current_stage == 3

        # Now get enough blaze rods
        tracker.update_from_observation(
            _make_stage3_obs(tick=200, dimension=1, blaze_rods=7)
        )
        assert tracker.progress.current_stage == 4

    def test_stage3_completion_reward(self) -> None:
        """Stage 3 completion emits reward signal."""
        logger.debug("TestProgressTrackerStage3.test_stage3_completion_reward called")
        tracker = ProgressTracker()
        # Auto-advance to stage 3
        tracker.update_from_observation(_make_stage3_obs(tick=50))
        assert tracker.progress.current_stage == 3

        # Complete stage 3
        rewards = tracker.update_from_observation(
            _make_stage3_obs(tick=100, dimension=1, blaze_rods=7)
        )
        assert "stage_3_complete" in rewards
        assert rewards["stage_3_complete"] == 5.0

    def test_stage3_death_tracked(self) -> None:
        """Deaths during Stage 3 increment stage_deaths[3]."""
        logger.debug("TestProgressTrackerStage3.test_stage3_death_tracked called")
        tracker = ProgressTracker()
        # Enter Stage 3
        tracker.update_from_observation(_make_stage3_obs(tick=50, health=20.0))
        assert tracker.progress.current_stage == 3

        # Die in Stage 3
        tracker.update_from_observation(_make_stage3_obs(tick=60, health=0.0))
        assert tracker.progress.stage_deaths[3] == 1
        assert tracker.progress.deaths == 1

    def test_stage3_time_accumulates(self) -> None:
        """Time spent in Stage 3 accumulates in stage_times[3]."""
        logger.debug("TestProgressTrackerStage3.test_stage3_time_accumulates called")
        tracker = ProgressTracker()
        # Enter Stage 3 at tick 50
        tracker.update_from_observation(_make_stage3_obs(tick=50))
        assert tracker.progress.current_stage == 3
        # Continue in Stage 3
        tracker.update_from_observation(_make_stage3_obs(tick=200, obsidian=5))
        assert tracker.progress.stage_times[3] == 150  # 200 - 50

    # --- Observation vector tests for Stage 3 indices ---

    IDX_PORTAL_BUILT = 10
    IDX_ENTERED_NETHER = 11
    IDX_FORTRESS_FOUND = 12
    IDX_BLAZES_KILLED = 13
    IDX_BLAZE_RODS = 14

    def test_observation_portal_built_false(self) -> None:
        """portal_built observation is 0.0 when not built."""
        logger.debug("TestProgressTrackerStage3.test_observation_portal_built_false called")
        tracker = ProgressTracker()
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_PORTAL_BUILT] == 0.0

    def test_observation_portal_built_true(self) -> None:
        """portal_built observation becomes 1.0 when set."""
        logger.debug("TestProgressTrackerStage3.test_observation_portal_built_true called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, portal_built=True)
        )
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_PORTAL_BUILT] == pytest.approx(1.0)

    def test_observation_entered_nether_false(self) -> None:
        """entered_nether observation is 0.0 initially."""
        logger.debug("TestProgressTrackerStage3.test_observation_entered_nether_false called")
        tracker = ProgressTracker()
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_ENTERED_NETHER] == 0.0

    def test_observation_entered_nether_true(self) -> None:
        """entered_nether observation becomes 1.0 on dimension change."""
        logger.debug("TestProgressTrackerStage3.test_observation_entered_nether_true called")
        tracker = ProgressTracker()
        tracker.update_from_observation(_make_stage3_obs(tick=1, dimension=0))
        tracker.update_from_observation(_make_stage3_obs(tick=2, dimension=1))
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_ENTERED_NETHER] == pytest.approx(1.0)

    def test_observation_fortress_found_false(self) -> None:
        """fortress_found observation is 0.0 initially."""
        logger.debug("TestProgressTrackerStage3.test_observation_fortress_found_false called")
        tracker = ProgressTracker()
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_FORTRESS_FOUND] == 0.0

    def test_observation_fortress_found_true(self) -> None:
        """fortress_found observation becomes 1.0 when set."""
        logger.debug("TestProgressTrackerStage3.test_observation_fortress_found_true called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, fortress_found=True)
        )
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_FORTRESS_FOUND] == pytest.approx(1.0)

    def test_observation_blazes_killed_normalized(self) -> None:
        """blazes_killed observation normalized to blazes_killed/10."""
        logger.debug("TestProgressTrackerStage3.test_observation_blazes_killed_normalized called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, blazes_killed=6)
        )
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_BLAZES_KILLED] == pytest.approx(6.0 / 10.0)

    def test_observation_blazes_killed_clamped(self) -> None:
        """blazes_killed observation clamped at 1.0."""
        logger.debug("TestProgressTrackerStage3.test_observation_blazes_killed_clamped called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, blazes_killed=15)
        )
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_BLAZES_KILLED] == pytest.approx(1.0)

    def test_observation_blaze_rods_normalized(self) -> None:
        """blaze_rods observation normalized to blaze_rods/10."""
        logger.debug("TestProgressTrackerStage3.test_observation_blaze_rods_normalized called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, blaze_rods=5)
        )
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_BLAZE_RODS] == pytest.approx(5.0 / 10.0)

    def test_observation_blaze_rods_clamped(self) -> None:
        """blaze_rods observation clamped at 1.0."""
        logger.debug("TestProgressTrackerStage3.test_observation_blaze_rods_clamped called")
        tracker = ProgressTracker()
        tracker.update_from_observation(
            _make_stage3_obs(tick=1, blaze_rods=12)
        )
        obs = tracker.progress.to_observation()
        assert obs[self.IDX_BLAZE_RODS] == pytest.approx(1.0)

    def test_sequential_nether_progression(self) -> None:
        """Full Stage 3 nether progression sequence accumulates correctly."""
        logger.debug("TestProgressTrackerStage3.test_sequential_nether_progression called")
        tracker = ProgressTracker()
        observations = [
            _make_stage3_obs(tick=50, obsidian=10),
            _make_stage3_obs(tick=100, obsidian=14, portal_built=True),
            _make_stage3_obs(tick=150, dimension=1, portal_built=True),
            _make_stage3_obs(
                tick=200, dimension=1, fortress_found=True, portal_built=True
            ),
            _make_stage3_obs(
                tick=300,
                dimension=1,
                fortress_found=True,
                blazes_killed=5,
                blaze_rods=4,
                portal_built=True,
            ),
            _make_stage3_obs(
                tick=400,
                dimension=1,
                fortress_found=True,
                blazes_killed=10,
                blaze_rods=8,
                portal_built=True,
            ),
        ]
        for obs in observations:
            tracker.update_from_observation(obs)

        p = tracker.progress
        assert p.obsidian_collected == 14
        assert p.portal_built is True
        assert p.entered_nether is True
        assert p.fortress_found is True
        assert p.blazes_killed == 10
        assert p.blaze_rods == 8
        # Stage should have advanced past 3
        assert p.current_stage == 4

    def test_no_nether_rewards_when_unchanged(self) -> None:
        """No Stage 3 rewards when nether state is empty/unchanged."""
        logger.debug("TestProgressTrackerStage3.test_no_nether_rewards_when_unchanged called")
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(_make_stage3_obs(tick=1))
        assert "portal_built" not in rewards
        assert "entered_nether" not in rewards
        assert "fortress_found" not in rewards
        assert "blaze_killed" not in rewards
        assert "blaze_rod_collected" not in rewards

    def test_initial_stage3_flags_false(self) -> None:
        """Fresh tracker has all Stage 3 flags at False/zero."""
        logger.info("TestProgressTrackerStage3.test_initial_stage3_flags_false called")
        tracker = ProgressTracker()
        p = tracker.progress
        assert p.portal_built is False
        assert p.entered_nether is False
        assert p.fortress_found is False
        assert p.blazes_killed == 0
        assert p.blaze_rods == 0
        assert p.obsidian_collected == 0
