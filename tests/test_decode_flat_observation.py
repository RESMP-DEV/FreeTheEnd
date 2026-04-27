"""Tests for decode_flat_observation across stages 1-4.

Verifies that decode_flat_observation correctly reconstructs the dict format
expected by ProgressTracker.update_from_observation from compact 256-float
observation vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from minecraft_sim.observations import (
    decode_flat_observation,
)
from minecraft_sim.progression import ProgressTracker

import logging

logger = logging.getLogger(__name__)


def _make_compact_vector(
    *,
    health: float = 20.0,
    hunger: float = 20.0,
    dimension: int = 0,
    position: tuple[float, float, float] = (0.0, 64.0, 0.0),
    on_ground: bool = True,
    in_water: bool = False,
    in_lava: bool = False,
    sprinting: bool = False,
    iron_count: int = 0,
    diamond_count: int = 0,
    blaze_rod_count: int = 0,
    ender_pearl_count: int = 0,
    eye_of_ender_count: int = 0,
    wood_count: int = 0,
    food_count: int = 0,
    hotbar_items: list[int] | None = None,
    stage: int = 1,
    dragon_active: bool = False,
    dragon_health: float = 200.0,
    crystals_remaining: int = 10,
    dragon_perched: bool = False,
) -> np.ndarray:
    """Build a compact 256-float observation vector with specified values."""
    logger.debug("_make_compact_vector called")
    vec = np.zeros(256, dtype=np.float32)

    # Player state (0-31)
    # Position: normalized as pos / 1000
    vec[0] = position[0] / 1000.0
    vec[1] = position[1] / 1000.0
    vec[2] = position[2] / 1000.0
    # Velocity (3-5): leave at 0
    # Pitch (6), Yaw (7): leave at 0
    vec[8] = health / 20.0
    vec[9] = hunger / 20.0
    # Armor (10): 0
    # Flags
    vec[12] = float(sprinting)
    vec[15] = float(on_ground)
    vec[16] = float(in_water)
    vec[17] = float(in_lava)

    # Inventory (32-63)
    if hotbar_items is None:
        hotbar_items = [0] * 9
    for i, item_id in enumerate(hotbar_items[:9]):
        vec[32 + i] = item_id / 512.0
    # Selected slot (32+18)
    vec[32 + 19] = wood_count / 64.0
    vec[32 + 20] = iron_count / 64.0
    vec[32 + 21] = diamond_count / 64.0
    vec[32 + 22] = blaze_rod_count / 64.0
    vec[32 + 23] = ender_pearl_count / 16.0
    vec[32 + 24] = eye_of_ender_count / 16.0
    vec[32 + 25] = food_count / 64.0

    # Stage one-hot (128-133)
    if 1 <= stage <= 6:
        vec[128 + stage - 1] = 1.0

    # Dragon (192-223)
    if dragon_active:
        vec[192] = 1.0
        vec[193] = dragon_health / 200.0
        vec[197] = float(dragon_perched)
        vec[198] = crystals_remaining / 10.0

    # Dimension one-hot (224-226)
    vec[224 + dimension] = 1.0

    return vec


class TestDecodeFlatObservationBasic:
    """Basic decode/encode round-trip and validation tests."""

    def test_invalid_shape_raises(self) -> None:
        """Non-256 vectors must raise ValueError."""
        logger.debug("TestDecodeFlatObservationBasic.test_invalid_shape_raises called")
        with pytest.raises(ValueError, match="shape"):
            decode_flat_observation(1, np.zeros(100, dtype=np.float32))

    def test_invalid_stage_raises(self) -> None:
        """stage_id outside [1, 6] must raise ValueError."""
        logger.debug("TestDecodeFlatObservationBasic.test_invalid_stage_raises called")
        vec = np.zeros(256, dtype=np.float32)
        with pytest.raises(ValueError, match="stage_id"):
            decode_flat_observation(0, vec)
        with pytest.raises(ValueError, match="stage_id"):
            decode_flat_observation(7, vec)

    def test_returns_dict_with_required_keys(self) -> None:
        """Result must contain all keys needed by ProgressTracker."""
        logger.debug("TestDecodeFlatObservationBasic.test_returns_dict_with_required_keys called")
        vec = _make_compact_vector(stage=1)
        result = decode_flat_observation(1, vec)
        assert "player" in result
        assert "inventory" in result
        assert "dimension" in result
        assert "tick_number" in result
        assert "hotbar" in result
        assert "dragon" in result

    def test_player_has_required_fields(self) -> None:
        """Player dict must contain health and dimension."""
        logger.debug("TestDecodeFlatObservationBasic.test_player_has_required_fields called")
        vec = _make_compact_vector(health=15.0, dimension=0)
        result = decode_flat_observation(1, vec)
        player = result["player"]
        assert "health" in player
        assert "dimension" in player
        assert player["dimension"] == 0

    def test_dimension_extraction(self) -> None:
        """Dimension is correctly extracted from one-hot encoding."""
        logger.debug("TestDecodeFlatObservationBasic.test_dimension_extraction called")
        for dim in (0, 1, 2):
            vec = _make_compact_vector(dimension=dim)
            result = decode_flat_observation(1, vec)
            assert result["dimension"] == dim
            assert result["player"]["dimension"] == dim


class TestDecodeFlatObservationStage1:
    """Stage 1 (Survival): wood collection, health, basic state."""

    def test_wood_count_decoded(self) -> None:
        """Wood count is extracted from compact inventory."""
        logger.debug("TestDecodeFlatObservationStage1.test_wood_count_decoded called")
        vec = _make_compact_vector(stage=1, wood_count=16)
        result = decode_flat_observation(1, vec)
        assert result["inventory"]["wood"] == 16

    def test_health_decoded(self) -> None:
        """Player health is correctly denormalized."""
        logger.debug("TestDecodeFlatObservationStage1.test_health_decoded called")
        vec = _make_compact_vector(stage=1, health=14.0)
        result = decode_flat_observation(1, vec)
        # Compact format: health stored as health/20, decoder multiplies by 20
        assert abs(result["player"]["health"] - 14.0) < 1.0

    def test_on_ground_flag(self) -> None:
        """on_ground flag is correctly decoded."""
        logger.debug("TestDecodeFlatObservationStage1.test_on_ground_flag called")
        vec = _make_compact_vector(stage=1, on_ground=True)
        result = decode_flat_observation(1, vec)
        assert result["player"]["on_ground"] is True

    def test_overworld_dimension(self) -> None:
        """Stage 1 should be in overworld (dimension 0)."""
        logger.debug("TestDecodeFlatObservationStage1.test_overworld_dimension called")
        vec = _make_compact_vector(stage=1, dimension=0)
        result = decode_flat_observation(1, vec)
        assert result["dimension"] == 0

    def test_compatible_with_progress_tracker(self) -> None:
        """Decoded Stage 1 obs can be fed to ProgressTracker without error."""
        logger.debug("TestDecodeFlatObservationStage1.test_compatible_with_progress_tracker called")
        vec = _make_compact_vector(stage=1, wood_count=8, health=20.0)
        result = decode_flat_observation(1, vec)
        tracker = ProgressTracker()
        rewards = tracker.update_from_observation(result)
        assert isinstance(rewards, dict)
        assert tracker.progress.wood_collected == 8


class TestDecodeFlatObservationStage2:
    """Stage 2 (Resources): iron, diamonds, tools."""

    def test_iron_count_decoded(self) -> None:
        """Iron ingot count is correctly extracted."""
        logger.debug("TestDecodeFlatObservationStage2.test_iron_count_decoded called")
        vec = _make_compact_vector(stage=2, iron_count=10)
        result = decode_flat_observation(2, vec)
        assert result["inventory"]["iron_ingots"] == 10

    def test_diamond_count_decoded(self) -> None:
        """Diamond count is correctly extracted."""
        logger.debug("TestDecodeFlatObservationStage2.test_diamond_count_decoded called")
        vec = _make_compact_vector(stage=2, diamond_count=3)
        result = decode_flat_observation(2, vec)
        assert result["inventory"]["diamonds"] == 3

    def test_iron_pickaxe_in_hotbar(self) -> None:
        """Iron pickaxe detection from hotbar item IDs."""
        logger.debug("TestDecodeFlatObservationStage2.test_iron_pickaxe_in_hotbar called")
        vec = _make_compact_vector(stage=2, hotbar_items=[257, 0, 0, 0, 0, 0, 0, 0, 0])
        result = decode_flat_observation(2, vec)
        assert result["inventory"]["iron_pickaxe"] >= 1

    def test_bucket_in_hotbar(self) -> None:
        """Bucket detection from hotbar item IDs."""
        logger.debug("TestDecodeFlatObservationStage2.test_bucket_in_hotbar called")
        vec = _make_compact_vector(stage=2, hotbar_items=[0, 325, 0, 0, 0, 0, 0, 0, 0])
        result = decode_flat_observation(2, vec)
        assert result["inventory"]["empty_buckets"] >= 1

    def test_progress_tracker_iron_update(self) -> None:
        """ProgressTracker correctly updates iron from decoded Stage 2."""
        logger.debug("TestDecodeFlatObservationStage2.test_progress_tracker_iron_update called")
        vec = _make_compact_vector(stage=2, iron_count=5)
        result = decode_flat_observation(2, vec)
        tracker = ProgressTracker()
        tracker.update_from_observation(result)
        assert tracker.progress.iron_ingots == 5

    def test_progress_tracker_diamond_update(self) -> None:
        """ProgressTracker correctly updates diamonds from decoded Stage 2."""
        logger.debug("TestDecodeFlatObservationStage2.test_progress_tracker_diamond_update called")
        vec = _make_compact_vector(stage=2, diamond_count=2)
        result = decode_flat_observation(2, vec)
        tracker = ProgressTracker()
        tracker.update_from_observation(result)
        assert tracker.progress.diamonds == 2


class TestDecodeFlatObservationStage3:
    """Stage 3 (Nether): blaze rods, dimension transition."""

    def test_blaze_rod_count_decoded(self) -> None:
        """Blaze rod count is correctly extracted."""
        logger.debug("TestDecodeFlatObservationStage3.test_blaze_rod_count_decoded called")
        vec = _make_compact_vector(stage=3, blaze_rod_count=7, dimension=1)
        result = decode_flat_observation(3, vec)
        assert result["inventory"]["blaze_rods"] == 7

    def test_nether_dimension(self) -> None:
        """Stage 3 in the nether reports dimension=1."""
        logger.debug("TestDecodeFlatObservationStage3.test_nether_dimension called")
        vec = _make_compact_vector(stage=3, dimension=1)
        result = decode_flat_observation(3, vec)
        assert result["dimension"] == 1

    def test_nether_dimension_triggers_entered_nether(self) -> None:
        """Feeding a nether-dimension obs to ProgressTracker sets entered_nether."""
        # First send an overworld obs to establish prev_dimension
        logger.debug("TestDecodeFlatObservationStage3.test_nether_dimension_triggers_entered_nether called")
        vec_ow = _make_compact_vector(stage=3, dimension=0)
        result_ow = decode_flat_observation(3, vec_ow)
        tracker = ProgressTracker()
        tracker.update_from_observation(result_ow)
        assert not tracker.progress.entered_nether

        # Now send nether obs
        vec_nether = _make_compact_vector(stage=3, dimension=1, blaze_rod_count=3)
        result_nether = decode_flat_observation(3, vec_nether)
        rewards = tracker.update_from_observation(result_nether)
        assert tracker.progress.entered_nether
        assert "entered_nether" in rewards

    def test_progress_tracker_blaze_rod_update(self) -> None:
        """ProgressTracker correctly counts blaze rods from decoded Stage 3."""
        logger.debug("TestDecodeFlatObservationStage3.test_progress_tracker_blaze_rod_update called")
        vec = _make_compact_vector(stage=3, blaze_rod_count=4, dimension=1)
        result = decode_flat_observation(3, vec)
        tracker = ProgressTracker()
        tracker.update_from_observation(result)
        assert tracker.progress.blaze_rods == 4


class TestDecodeFlatObservationStage4:
    """Stage 4 (Pearls): ender pearl collection."""

    def test_ender_pearl_count_decoded(self) -> None:
        """Ender pearl count is correctly extracted."""
        logger.debug("TestDecodeFlatObservationStage4.test_ender_pearl_count_decoded called")
        vec = _make_compact_vector(stage=4, ender_pearl_count=12)
        result = decode_flat_observation(4, vec)
        assert result["inventory"]["ender_pearls"] == 12

    def test_eye_of_ender_count_decoded(self) -> None:
        """Eye of ender count is correctly extracted."""
        logger.debug("TestDecodeFlatObservationStage4.test_eye_of_ender_count_decoded called")
        vec = _make_compact_vector(stage=4, eye_of_ender_count=6)
        result = decode_flat_observation(4, vec)
        assert result["inventory"]["eyes_of_ender"] == 6

    def test_progress_tracker_pearl_update(self) -> None:
        """ProgressTracker correctly counts pearls from decoded Stage 4."""
        logger.debug("TestDecodeFlatObservationStage4.test_progress_tracker_pearl_update called")
        vec = _make_compact_vector(stage=4, ender_pearl_count=8)
        result = decode_flat_observation(4, vec)
        tracker = ProgressTracker()
        tracker.update_from_observation(result)
        assert tracker.progress.ender_pearls == 8

    def test_progress_tracker_eye_update(self) -> None:
        """ProgressTracker correctly counts eyes from decoded Stage 4."""
        logger.debug("TestDecodeFlatObservationStage4.test_progress_tracker_eye_update called")
        vec = _make_compact_vector(stage=4, eye_of_ender_count=5)
        result = decode_flat_observation(4, vec)
        tracker = ProgressTracker()
        tracker.update_from_observation(result)
        assert tracker.progress.eyes_crafted == 5

    def test_sequential_pearl_collection(self) -> None:
        """Simulates progressive pearl collection across observations."""
        logger.debug("TestDecodeFlatObservationStage4.test_sequential_pearl_collection called")
        tracker = ProgressTracker()
        for n_pearls in (2, 5, 9, 12):
            vec = _make_compact_vector(stage=4, ender_pearl_count=n_pearls)
            result = decode_flat_observation(4, vec)
            tracker.update_from_observation(result)
        assert tracker.progress.ender_pearls == 12

    def test_hotbar_decoded(self) -> None:
        """Hotbar item IDs are preserved in decoded output."""
        logger.debug("TestDecodeFlatObservationStage4.test_hotbar_decoded called")
        items = [276, 261, 262, 0, 0, 0, 0, 0, 0]  # sword, bow, arrow
        vec = _make_compact_vector(stage=4, hotbar_items=items)
        result = decode_flat_observation(4, vec)
        hotbar = result["hotbar"]
        assert len(hotbar) == 9
        # Values should be approximately correct (int truncation from normalization)
        assert hotbar[0] > 200  # diamond sword (276)
        assert hotbar[1] > 200  # bow (261)


class TestDecodeFlatObservationDragonState:
    """Dragon state decoding (active during stage 6, but verifiable with any stage)."""

    def test_dragon_inactive_by_default(self) -> None:
        """No dragon state when not active."""
        logger.debug("TestDecodeFlatObservationDragonState.test_dragon_inactive_by_default called")
        vec = _make_compact_vector(stage=1)
        result = decode_flat_observation(1, vec)
        assert result["dragon"]["is_active"] is False

    def test_dragon_active_decoded(self) -> None:
        """Active dragon state is decoded correctly."""
        logger.debug("TestDecodeFlatObservationDragonState.test_dragon_active_decoded called")
        vec = _make_compact_vector(
            stage=6,
            dimension=2,
            dragon_active=True,
            dragon_health=150.0,
            crystals_remaining=5,
        )
        result = decode_flat_observation(6, vec)
        dragon = result["dragon"]
        assert dragon["is_active"] is True
        assert abs(dragon["dragon_health"] - 150.0) < 5.0
        assert dragon["crystals_remaining"] == 5

    def test_dragon_perched_phase(self) -> None:
        """Dragon perched state maps to phase 3."""
        logger.debug("TestDecodeFlatObservationDragonState.test_dragon_perched_phase called")
        vec = _make_compact_vector(
            stage=6,
            dimension=2,
            dragon_active=True,
            dragon_health=100.0,
            dragon_perched=True,
        )
        result = decode_flat_observation(6, vec)
        assert result["dragon"]["phase"] == 3


class TestDecodeFlatObservationEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_vector(self) -> None:
        """All-zero vector decodes without errors."""
        logger.debug("TestDecodeFlatObservationEdgeCases.test_zero_vector called")
        vec = np.zeros(256, dtype=np.float32)
        result = decode_flat_observation(1, vec)
        assert result["player"]["health"] == 0.0
        assert result["dimension"] == 0

    def test_accepts_float64_input(self) -> None:
        """Function handles float64 arrays by casting to float32."""
        logger.debug("TestDecodeFlatObservationEdgeCases.test_accepts_float64_input called")
        vec = np.zeros(256, dtype=np.float64)
        vec[224] = 1.0  # overworld
        result = decode_flat_observation(1, vec)
        assert result["dimension"] == 0

    def test_full_inventory_stage2(self) -> None:
        """Full Stage 2 inventory with multiple items."""
        logger.debug("TestDecodeFlatObservationEdgeCases.test_full_inventory_stage2 called")
        vec = _make_compact_vector(
            stage=2,
            iron_count=15,
            diamond_count=5,
            wood_count=32,
            hotbar_items=[257, 256, 325, 0, 0, 0, 0, 0, 0],
        )
        result = decode_flat_observation(2, vec)
        inv = result["inventory"]
        assert inv["iron_ingots"] == 15
        assert inv["diamonds"] == 5
        assert inv["wood"] == 32
        assert inv["iron_pickaxe"] >= 1
        assert inv["empty_buckets"] >= 1
