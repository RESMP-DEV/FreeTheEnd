"""Tests for Minecraft curriculum learning module.

This module tests both the stage-based curriculum system (curriculum.py)
and the per-environment vectorized curriculum manager (curriculum_manager.py).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from minecraft_sim.curriculum import (
    CurriculumManager as GlobalCurriculumManager,
)
from minecraft_sim.curriculum import (
    RewardConfig,
    SpawnConfig,
    Stage,
    StageID,
    StageProgress,
    TerminationConfig,
)
from minecraft_sim.curriculum_manager import (
    AdvancementEvent,
    StageStats,
    VecCurriculumManager,
)

# =============================================================================
# Tests for StageID enum
# =============================================================================


class TestStageID:
    """Tests for StageID enum."""

    def test_stage_order(self) -> None:
        """Test stages are in correct difficulty order."""
        assert StageID.BASIC_SURVIVAL < StageID.RESOURCE_GATHERING
        assert StageID.RESOURCE_GATHERING < StageID.NETHER_NAVIGATION
        assert StageID.NETHER_NAVIGATION < StageID.ENDERMAN_HUNTING
        assert StageID.ENDERMAN_HUNTING < StageID.STRONGHOLD_FINDING
        assert StageID.STRONGHOLD_FINDING < StageID.END_FIGHT

    def test_stage_values(self) -> None:
        """Test stage integer values."""
        assert StageID.BASIC_SURVIVAL == 1
        assert StageID.END_FIGHT == 6

    def test_stage_count(self) -> None:
        """Test total number of stages."""
        assert len(StageID) == 6


# =============================================================================
# Tests for configuration dataclasses
# =============================================================================


class TestSpawnConfig:
    """Tests for SpawnConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default spawn configuration."""
        config = SpawnConfig()
        assert config.biome == "plains"
        assert config.time_of_day == 0
        assert config.weather == "clear"
        assert config.random_position is True
        assert config.position is None
        assert config.inventory == {}
        assert config.health == 20.0
        assert config.hunger == 20.0

    def test_custom_config(self) -> None:
        """Test custom spawn configuration."""
        config = SpawnConfig(
            biome="desert",
            time_of_day=12000,
            weather="rain",
            random_position=False,
            position=(100.0, 64.0, 200.0),
            inventory={"diamond_sword": 1, "golden_apple": 5},
            health=10.0,
            hunger=15.0,
        )
        assert config.biome == "desert"
        assert config.time_of_day == 12000
        assert config.weather == "rain"
        assert config.random_position is False
        assert config.position == (100.0, 64.0, 200.0)
        assert config.inventory == {"diamond_sword": 1, "golden_apple": 5}
        assert config.health == 10.0
        assert config.hunger == 15.0


class TestRewardConfig:
    """Tests for RewardConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default reward configuration."""
        config = RewardConfig()
        assert config.sparse_reward == 1.0
        assert config.dense_rewards == {}
        assert config.penalty_per_death == -0.5
        assert config.penalty_per_tick == -0.0001
        assert config.exploration_bonus == 0.01

    def test_custom_config(self) -> None:
        """Test custom reward configuration."""
        config = RewardConfig(
            sparse_reward=10.0,
            dense_rewards={"kill_zombie": 0.1, "mine_diamond": 1.0},
            penalty_per_death=-1.0,
            penalty_per_tick=-0.001,
            exploration_bonus=0.05,
        )
        assert config.sparse_reward == 10.0
        assert config.dense_rewards["kill_zombie"] == 0.1
        assert config.penalty_per_death == -1.0


class TestTerminationConfig:
    """Tests for TerminationConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default termination configuration."""
        config = TerminationConfig()
        assert config.max_ticks == 36000
        assert config.max_deaths == 5
        assert config.success_conditions == []
        assert config.failure_conditions == []

    def test_custom_config(self) -> None:
        """Test custom termination configuration."""
        config = TerminationConfig(
            max_ticks=72000,
            max_deaths=3,
            success_conditions=["enter_nether"],
            failure_conditions=["fall_in_lava"],
        )
        assert config.max_ticks == 72000
        assert config.max_deaths == 3
        assert "enter_nether" in config.success_conditions


# =============================================================================
# Tests for Stage dataclass
# =============================================================================


class TestStage:
    """Tests for Stage dataclass."""

    @pytest.fixture
    def basic_stage(self) -> Stage:
        """Create a basic stage for testing."""
        return Stage(
            id=StageID.BASIC_SURVIVAL,
            name="Basic Survival",
            description="Learn basic survival skills",
            objectives=["survive", "gather wood"],
            spawn=SpawnConfig(),
            rewards=RewardConfig(),
            termination=TerminationConfig(),
            difficulty=1,
            expected_episodes=1000,
            curriculum_threshold=0.7,
        )

    def test_stage_creation(self, basic_stage: Stage) -> None:
        """Test stage creation with required fields."""
        assert basic_stage.id == StageID.BASIC_SURVIVAL
        assert basic_stage.name == "Basic Survival"
        assert len(basic_stage.objectives) == 2

    def test_to_dict(self, basic_stage: Stage) -> None:
        """Test stage serialization to dict."""
        d = basic_stage.to_dict()
        assert d["id"] == 1
        assert d["name"] == "Basic Survival"
        assert d["difficulty"] == 1
        assert d["curriculum_threshold"] == 0.7
        assert "spawn" in d
        assert "rewards" in d
        assert "termination" in d

    def test_from_dict(self, basic_stage: Stage) -> None:
        """Test stage deserialization from dict."""
        d = basic_stage.to_dict()
        restored = Stage.from_dict(d)
        assert restored.id == basic_stage.id
        assert restored.name == basic_stage.name
        assert restored.objectives == basic_stage.objectives
        assert restored.difficulty == basic_stage.difficulty

    def test_roundtrip(self, basic_stage: Stage) -> None:
        """Test to_dict/from_dict roundtrip."""
        d = basic_stage.to_dict()
        restored = Stage.from_dict(d)
        assert restored.to_dict() == d

    def test_prerequisites(self) -> None:
        """Test stage with prerequisites."""
        stage = Stage(
            id=StageID.NETHER_NAVIGATION,
            name="Nether Navigation",
            description="Navigate the Nether",
            objectives=["enter nether", "find fortress"],
            spawn=SpawnConfig(),
            rewards=RewardConfig(),
            termination=TerminationConfig(),
            prerequisites=[StageID.BASIC_SURVIVAL, StageID.RESOURCE_GATHERING],
        )
        assert len(stage.prerequisites) == 2
        assert StageID.BASIC_SURVIVAL in stage.prerequisites

    def test_stage1_metadata_progress_keys_and_observations(self) -> None:
        """Test Stage 1 YAML loads metadata.progress_keys and observation entries."""
        stage1_yaml = (
            Path(__file__).parent.parent
            / "python/minecraft_sim/stage_configs/stage_1_basic_survival.yaml"
        )
        if not stage1_yaml.exists():
            pytest.skip("stage_1_basic_survival.yaml not found")

        import yaml

        with open(stage1_yaml) as f:
            data = yaml.safe_load(f)
        stage = Stage.from_dict(data)

        # Verify metadata.progress_keys is populated
        assert "progress_keys" in stage.metadata
        progress_keys = stage.metadata["progress_keys"]
        assert isinstance(progress_keys, dict)
        assert len(progress_keys) > 0

        # Verify expected progress key entries for wood, stone, combat
        assert "wood_logs_mined" in progress_keys
        assert "cobblestone_mined" in progress_keys
        assert "zombies_killed" in progress_keys
        assert "skeletons_killed" in progress_keys
        assert "spiders_killed" in progress_keys

        # Verify observation_space includes telemetry counters
        expected_obs = [
            "wood_logs_mined",
            "cobblestone_mined",
            "zombies_killed",
            "skeletons_killed",
            "spiders_killed",
            "total_damage_dealt",
        ]
        for obs in expected_obs:
            assert obs in stage.observation_space, f"Missing observation: {obs}"

        # Verify progress_keys values are human-readable dashboard labels
        for key, label in progress_keys.items():
            assert isinstance(label, str)
            assert len(label) > 0

    def test_spawn_with_position(self) -> None:
        """Test stage with specific spawn position."""
        stage = Stage(
            id=StageID.END_FIGHT,
            name="End Fight",
            description="Defeat the Ender Dragon",
            objectives=["kill ender dragon"],
            spawn=SpawnConfig(
                position=(0.0, 64.0, 0.0),
                random_position=False,
            ),
            rewards=RewardConfig(sparse_reward=100.0),
            termination=TerminationConfig(max_ticks=72000),
        )
        d = stage.to_dict()
        restored = Stage.from_dict(d)
        assert restored.spawn.position == (0.0, 64.0, 0.0)
        assert restored.spawn.random_position is False


# =============================================================================
# Tests for StageProgress dataclass
# =============================================================================


class TestStageProgress:
    """Tests for StageProgress dataclass."""

    def test_defaults(self) -> None:
        """Test default progress values."""
        prog = StageProgress(stage_id=StageID.BASIC_SURVIVAL)
        assert prog.episodes_completed == 0
        assert prog.successes == 0
        assert prog.total_reward == 0.0
        assert prog.best_reward == float("-inf")
        assert prog.average_ticks == 0.0
        assert prog.mastered is False

    def test_success_rate_zero_episodes(self) -> None:
        """Test success rate with no episodes."""
        prog = StageProgress(stage_id=StageID.BASIC_SURVIVAL)
        assert prog.success_rate == 0.0

    def test_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        prog = StageProgress(
            stage_id=StageID.BASIC_SURVIVAL,
            episodes_completed=100,
            successes=70,
        )
        assert prog.success_rate == 0.7

    def test_average_reward_zero_episodes(self) -> None:
        """Test average reward with no episodes."""
        prog = StageProgress(stage_id=StageID.BASIC_SURVIVAL)
        assert prog.average_reward == 0.0

    def test_average_reward_calculation(self) -> None:
        """Test average reward calculation."""
        prog = StageProgress(
            stage_id=StageID.BASIC_SURVIVAL,
            episodes_completed=10,
            total_reward=50.0,
        )
        assert prog.average_reward == 5.0


# =============================================================================
# Tests for GlobalCurriculumManager (curriculum.py)
# =============================================================================


class TestGlobalCurriculumManager:
    """Tests for CurriculumManager class from curriculum.py."""

    @pytest.fixture
    def manager(self) -> GlobalCurriculumManager:
        """Create a curriculum manager with test stages."""
        manager = GlobalCurriculumManager(config_dir=None)
        # Register test stages manually
        for stage_id in StageID:
            stage = Stage(
                id=stage_id,
                name=stage_id.name.replace("_", " ").title(),
                description=f"Stage {stage_id.value}",
                objectives=[f"objective_{stage_id.value}"],
                spawn=SpawnConfig(),
                rewards=RewardConfig(),
                termination=TerminationConfig(),
                prerequisites=[StageID(stage_id.value - 1)] if stage_id.value > 1 else [],
                difficulty=stage_id.value,
                expected_episodes=1000,
                curriculum_threshold=0.7,
            )
            manager.register_stage(stage)
        return manager

    def test_register_stage(self, manager: GlobalCurriculumManager) -> None:
        """Test stage registration."""
        assert len(manager.stages) == 6
        assert StageID.BASIC_SURVIVAL in manager.stages

    def test_get_stage(self, manager: GlobalCurriculumManager) -> None:
        """Test getting stage by ID."""
        stage = manager.get_stage(StageID.BASIC_SURVIVAL)
        assert stage.id == StageID.BASIC_SURVIVAL

    def test_get_stage_not_found(self) -> None:
        """Test getting non-existent stage raises error."""
        manager = GlobalCurriculumManager(config_dir="/nonexistent/path")
        # Clear any auto-loaded stages
        manager.stages.clear()
        with pytest.raises(KeyError):
            manager.get_stage(StageID.BASIC_SURVIVAL)

    def test_start_training(self, manager: GlobalCurriculumManager) -> None:
        """Test starting training on a stage."""
        stage = manager.start_training(StageID.BASIC_SURVIVAL)
        assert stage.id == StageID.BASIC_SURVIVAL
        assert manager.current_stage == StageID.BASIC_SURVIVAL
        assert StageID.BASIC_SURVIVAL in manager.stage_history

    def test_start_training_default(self, manager: GlobalCurriculumManager) -> None:
        """Test starting training without specifying stage."""
        stage = manager.start_training()
        assert stage.id == StageID.BASIC_SURVIVAL

    def test_record_episode_success(self, manager: GlobalCurriculumManager) -> None:
        """Test recording successful episode."""
        manager.start_training(StageID.BASIC_SURVIVAL)
        manager.record_episode(success=True, reward=10.0, ticks=1000)
        prog = manager.progress[StageID.BASIC_SURVIVAL]
        assert prog.episodes_completed == 1
        assert prog.successes == 1
        assert prog.total_reward == 10.0

    def test_record_episode_failure(self, manager: GlobalCurriculumManager) -> None:
        """Test recording failed episode."""
        manager.start_training(StageID.BASIC_SURVIVAL)
        manager.record_episode(success=False, reward=-1.0, ticks=500)
        prog = manager.progress[StageID.BASIC_SURVIVAL]
        assert prog.episodes_completed == 1
        assert prog.successes == 0
        assert prog.total_reward == -1.0

    def test_record_episode_best_reward(self, manager: GlobalCurriculumManager) -> None:
        """Test best reward tracking."""
        manager.start_training(StageID.BASIC_SURVIVAL)
        manager.record_episode(success=True, reward=5.0, ticks=1000)
        manager.record_episode(success=True, reward=15.0, ticks=1000)
        manager.record_episode(success=True, reward=10.0, ticks=1000)
        prog = manager.progress[StageID.BASIC_SURVIVAL]
        assert prog.best_reward == 15.0

    def test_mastery_detection(self, manager: GlobalCurriculumManager) -> None:
        """Test stage mastery detection."""
        manager.start_training(StageID.BASIC_SURVIVAL)
        # Simulate 100+ episodes with 70%+ success rate
        for i in range(150):
            success = i < 110  # 110 successes, 40 failures = 73%
            manager.record_episode(success=success, reward=1.0 if success else 0.0, ticks=1000)
        prog = manager.progress[StageID.BASIC_SURVIVAL]
        assert prog.mastered is True

    def test_stage1_advancement_honors_overridden_min_episodes_and_threshold(self) -> None:
        """Test Stage 1 advancement respects overridden expected_episodes and curriculum_threshold."""
        manager = GlobalCurriculumManager(config_dir=None)

        # Register Stage 1 with custom overrides:
        # expected_episodes=200 -> min_episodes = 200 // 10 = 20
        # curriculum_threshold=0.9 (higher than default 0.7)
        stage1 = Stage(
            id=StageID.BASIC_SURVIVAL,
            name="Basic Survival",
            description="Stage 1",
            objectives=["survive"],
            spawn=SpawnConfig(),
            rewards=RewardConfig(),
            termination=TerminationConfig(),
            expected_episodes=200,
            curriculum_threshold=0.9,
        )
        manager.register_stage(stage1)
        manager.start_training(StageID.BASIC_SURVIVAL)

        # Record 19 successes (100% rate, but below min_episodes=20)
        for _ in range(19):
            manager.record_episode(success=True, reward=1.0, ticks=100)
        prog = manager.progress[StageID.BASIC_SURVIVAL]
        assert prog.mastered is False, "Should not master before min_episodes reached"

        # Episode 20: now at min_episodes but with a failure, dropping rate to 19/20=0.95
        # Wait -- 0.95 >= 0.9, so that would pass. Instead, add failures to drop below 0.9.
        # Reset and redo: 15 successes + 5 failures = 20 episodes, rate=0.75 < 0.9
        manager2 = GlobalCurriculumManager(config_dir=None)
        manager2.register_stage(stage1)
        manager2.start_training(StageID.BASIC_SURVIVAL)

        for _ in range(15):
            manager2.record_episode(success=True, reward=1.0, ticks=100)
        for _ in range(5):
            manager2.record_episode(success=False, reward=0.0, ticks=100)

        prog2 = manager2.progress[StageID.BASIC_SURVIVAL]
        assert prog2.episodes_completed == 20
        assert prog2.success_rate == 0.75
        assert prog2.mastered is False, "Should not master with rate below overridden threshold (0.9)"

        # Now add more successes to push rate above 0.9: need total successes/total >= 0.9
        # Currently 15/20. After N more successes: (15+N)/(20+N) >= 0.9 -> N >= 50
        for _ in range(50):
            manager2.record_episode(success=True, reward=1.0, ticks=100)

        prog2 = manager2.progress[StageID.BASIC_SURVIVAL]
        # 65 successes / 70 episodes = 0.9286
        assert prog2.episodes_completed == 70
        assert prog2.success_rate > 0.9
        assert prog2.mastered is True, "Should master once rate exceeds overridden threshold"

    def test_stage2_advancement_requires_obsidian_and_success_rate(self) -> None:
        """Test Stage 2 advancement waits for configured obsidian requirement and success rate.

        Stage 2 (Resource Gathering) defines:
        - Success condition: has_bucket AND obsidian_count >= 10 (from stage_envs.py)
        - curriculum_threshold: 0.65 (from stage_2_resource_gathering.yaml)
        - expected_episodes: 1000 -> min_episodes = 100

        Advancement should NOT occur until both min_episodes is reached AND the
        success rate (episodes where the obsidian+bucket goal was met) exceeds 0.65.
        """
        manager = GlobalCurriculumManager(config_dir=None)

        # Register Stage 2 with the actual YAML config values
        stage2 = Stage(
            id=StageID.RESOURCE_GATHERING,
            name="Resource Gathering",
            description="Mine resources and collect obsidian",
            objectives=["Mine iron", "Craft bucket", "Collect 10 obsidian"],
            spawn=SpawnConfig(inventory={"wooden_pickaxe": 1, "wooden_sword": 1, "oak_log": 16}),
            rewards=RewardConfig(sparse_reward=15.0),
            termination=TerminationConfig(max_ticks=48000, max_deaths=7),
            prerequisites=[StageID.BASIC_SURVIVAL],
            difficulty=3,
            expected_episodes=1000,  # min_episodes = 1000 // 10 = 100
            curriculum_threshold=0.65,  # From stage_2_resource_gathering.yaml
        )
        manager.register_stage(stage2)
        manager.start_training(StageID.RESOURCE_GATHERING)

        # Phase 1: Record 99 episodes (all successes) -- below min_episodes=100
        for _ in range(99):
            manager.record_episode(success=True, reward=15.0, ticks=5000)
        prog = manager.progress[StageID.RESOURCE_GATHERING]
        assert prog.episodes_completed == 99
        assert prog.success_rate == 1.0
        assert prog.mastered is False, (
            "Should not master before min_episodes (100) even with perfect success rate"
        )

        # Phase 2: At episode 100, mastery is evaluated. Inject failures to push rate below 0.65.
        # Reset and test with a mixed success/failure pattern.
        manager2 = GlobalCurriculumManager(config_dir=None)
        manager2.register_stage(stage2)
        manager2.start_training(StageID.RESOURCE_GATHERING)

        # 60 successes (episodes where obsidian >= 10 AND has_bucket)
        for _ in range(60):
            manager2.record_episode(success=True, reward=15.0, ticks=5000)
        # 40 failures (episodes where obsidian < 10, representing incomplete mining)
        for _ in range(40):
            manager2.record_episode(success=False, reward=2.0, ticks=10000)

        prog2 = manager2.progress[StageID.RESOURCE_GATHERING]
        assert prog2.episodes_completed == 100
        assert prog2.success_rate == 0.6  # 60/100 < 0.65
        assert prog2.mastered is False, (
            "Should not master with success rate 0.6 < configured threshold 0.65"
        )

        # Phase 3: Add more successes to bring rate above 0.65
        # Need (60 + N) / (100 + N) >= 0.65 -> N >= ~15 (approx)
        for _ in range(20):
            manager2.record_episode(success=True, reward=15.0, ticks=5000)

        prog2 = manager2.progress[StageID.RESOURCE_GATHERING]
        # 80 successes / 120 episodes = 0.6667 > 0.65
        assert prog2.episodes_completed == 120
        assert prog2.success_rate > 0.65
        assert prog2.mastered is True, (
            "Should master once success rate exceeds configured threshold (0.65)"
        )

    def test_stage3_advancement_requires_blaze_rods_and_min_episodes(self) -> None:
        """Test Stage 3 advancement waits for configured blaze rod and episode constraints.

        Stage 3 (Nether Navigation) defines:
        - Success condition: fortress_found AND blaze_rods >= 7 (from stage_envs.py)
        - curriculum_threshold: 0.6 (from stage_3_nether_navigation.yaml)
        - expected_episodes: 2000 -> min_episodes = 200

        Advancement should NOT occur until both min_episodes is reached AND the
        success rate (episodes where blaze_rods >= 7 goal was met) exceeds 0.6.
        """
        manager = GlobalCurriculumManager(config_dir=None)

        stage3 = Stage(
            id=StageID.NETHER_NAVIGATION,
            name="Nether Navigation",
            description="Navigate nether, find fortress, collect blaze rods",
            objectives=[
                "Mine 10 obsidian",
                "Build nether portal frame",
                "Light portal with flint and steel",
                "Enter the Nether",
                "Locate a Nether fortress",
                "Kill 7 blazes",
                "Collect 7 blaze rods",
            ],
            spawn=SpawnConfig(
                inventory={
                    "iron_pickaxe": 2,
                    "iron_sword": 1,
                    "cobblestone": 64,
                    "bucket": 2,
                    "flint_and_steel": 1,
                }
            ),
            rewards=RewardConfig(sparse_reward=25.0, penalty_per_death=-2.0),
            termination=TerminationConfig(
                max_ticks=72000,
                max_deaths=5,
                success_conditions=["dimension == nether", "objective:blaze_rods >= 7"],
            ),
            prerequisites=[StageID.RESOURCE_GATHERING],
            difficulty=5,
            expected_episodes=2000,  # min_episodes = 2000 // 10 = 200
            curriculum_threshold=0.6,  # From stage_3_nether_navigation.yaml
        )
        manager.register_stage(stage3)
        manager.start_training(StageID.NETHER_NAVIGATION)

        # Phase 1: Record 199 episodes (all successes) -- below min_episodes=200
        for _ in range(199):
            manager.record_episode(success=True, reward=25.0, ticks=10000)
        prog = manager.progress[StageID.NETHER_NAVIGATION]
        assert prog.episodes_completed == 199
        assert prog.success_rate == 1.0
        assert prog.mastered is False, (
            "Should not master before min_episodes (200) even with perfect success rate"
        )

        # Phase 2: At min_episodes, test that low success rate blocks mastery.
        manager2 = GlobalCurriculumManager(config_dir=None)
        manager2.register_stage(stage3)
        manager2.start_training(StageID.NETHER_NAVIGATION)

        # 100 successes (episodes where blaze_rods >= 7 was achieved)
        for _ in range(100):
            manager2.record_episode(success=True, reward=25.0, ticks=10000)
        # 100 failures (episodes where agent failed to collect 7 blaze rods)
        for _ in range(100):
            manager2.record_episode(success=False, reward=3.0, ticks=20000)

        prog2 = manager2.progress[StageID.NETHER_NAVIGATION]
        assert prog2.episodes_completed == 200
        assert prog2.success_rate == 0.5  # 100/200 < 0.6
        assert prog2.mastered is False, (
            "Should not master with success rate 0.5 < configured threshold 0.6"
        )

        # Phase 3: Add more successes to bring rate above 0.6
        # Need (100+N)/(200+N) >= 0.6 -> 0.4N >= 20 -> N >= 50
        for _ in range(50):
            manager2.record_episode(success=True, reward=25.0, ticks=10000)

        prog2 = manager2.progress[StageID.NETHER_NAVIGATION]
        # 150 successes / 250 episodes = 0.6
        assert prog2.episodes_completed == 250
        assert prog2.success_rate >= 0.6
        assert prog2.mastered is True, (
            "Should master once success rate reaches configured threshold (0.6)"
        )

    def test_stage3_no_advancement_with_insufficient_blaze_rod_episodes(self) -> None:
        """Test that partial blaze rod collection episodes don't count as successes.

        In the actual environment, an episode is only marked as success=True when
        the agent collects >= 7 blaze rods (as defined in _evaluate_stage_criteria).
        This test validates that the curriculum correctly blocks advancement when
        most episodes end with fewer than 7 blaze rods.
        """
        manager = GlobalCurriculumManager(config_dir=None)

        stage3 = Stage(
            id=StageID.NETHER_NAVIGATION,
            name="Nether Navigation",
            description="Navigate nether, find fortress, collect blaze rods",
            objectives=["Collect 7 blaze rods"],
            spawn=SpawnConfig(),
            rewards=RewardConfig(sparse_reward=25.0),
            termination=TerminationConfig(max_ticks=72000, max_deaths=5),
            prerequisites=[StageID.RESOURCE_GATHERING],
            difficulty=5,
            expected_episodes=2000,
            curriculum_threshold=0.6,
        )
        manager.register_stage(stage3)
        manager.start_training(StageID.NETHER_NAVIGATION)

        # Simulate 300 episodes where agent mostly fails (only gets 3-5 blaze rods)
        for i in range(300):
            # Only 30% success rate (agent rarely gets all 7 rods)
            success = i % 10 < 3
            manager.record_episode(
                success=success, reward=5.0 if success else 1.0, ticks=15000
            )

        prog = manager.progress[StageID.NETHER_NAVIGATION]
        assert prog.episodes_completed == 300
        assert prog.success_rate == pytest.approx(0.3, abs=0.01)
        assert prog.mastered is False, (
            "Should not master with 30% success rate "
            "(agents rarely collecting 7 blaze rods)"
        )

    def test_stage3_advancement_uses_yaml_threshold_not_default(self) -> None:
        """Test that Stage 3 uses its YAML-defined threshold (0.6) not the default (0.7).

        This ensures that the curriculum_threshold from the YAML config is correctly
        respected. A success rate between 0.6 and 0.7 should trigger mastery for
        Stage 3 but not for a stage with the default 0.7 threshold.
        """
        # Stage 3 with curriculum_threshold=0.6
        manager_s3 = GlobalCurriculumManager(config_dir=None)
        stage3 = Stage(
            id=StageID.NETHER_NAVIGATION,
            name="Nether Navigation",
            description="Nether stage",
            objectives=["Collect 7 blaze rods"],
            spawn=SpawnConfig(),
            rewards=RewardConfig(),
            termination=TerminationConfig(),
            expected_episodes=100,  # min_episodes = 10
            curriculum_threshold=0.6,
        )
        manager_s3.register_stage(stage3)
        manager_s3.start_training(StageID.NETHER_NAVIGATION)

        # Stage with default threshold (0.7)
        manager_default = GlobalCurriculumManager(config_dir=None)
        stage_default = Stage(
            id=StageID.NETHER_NAVIGATION,
            name="Nether Navigation",
            description="Nether stage",
            objectives=["Collect 7 blaze rods"],
            spawn=SpawnConfig(),
            rewards=RewardConfig(),
            termination=TerminationConfig(),
            expected_episodes=100,  # min_episodes = 10
            curriculum_threshold=0.7,  # default threshold
        )
        manager_default.register_stage(stage_default)
        manager_default.start_training(StageID.NETHER_NAVIGATION)

        # Record 7 failures first, then 13 successes = 65% overall.
        # Mastery is checked after each episode, so ordering matters.
        # With failures up front, rate never exceeds 65% once min_episodes is met.
        for i in range(20):
            success = i >= 7  # first 7 fail, last 13 succeed -> 13/20 = 0.65
            manager_s3.record_episode(success=success, reward=1.0, ticks=1000)
            manager_default.record_episode(success=success, reward=1.0, ticks=1000)

        prog_s3 = manager_s3.progress[StageID.NETHER_NAVIGATION]
        prog_default = manager_default.progress[StageID.NETHER_NAVIGATION]

        assert prog_s3.success_rate == 0.65
        assert prog_s3.mastered is True, (
            "Stage 3 with threshold 0.6 should master at 65% success rate"
        )
        assert prog_default.mastered is False, (
            "Stage with threshold 0.7 should NOT master at 65% success rate"
        )

    def test_should_advance_not_mastered(self, manager: GlobalCurriculumManager) -> None:
        """Test should_advance returns False when not mastered."""
        manager.start_training(StageID.BASIC_SURVIVAL)
        assert manager.should_advance() is False

    def test_should_advance_mastered(self, manager: GlobalCurriculumManager) -> None:
        """Test should_advance returns True when mastered."""
        manager.start_training(StageID.BASIC_SURVIVAL)
        manager.progress[StageID.BASIC_SURVIVAL].mastered = True
        assert manager.should_advance() is True

    def test_advance_stage(self, manager: GlobalCurriculumManager) -> None:
        """Test advancing to next stage."""
        manager.start_training(StageID.BASIC_SURVIVAL)
        manager.progress[StageID.BASIC_SURVIVAL].mastered = True
        new_stage = manager.advance_stage()
        assert new_stage is not None
        assert new_stage.id == StageID.RESOURCE_GATHERING
        assert manager.current_stage == StageID.RESOURCE_GATHERING

    def test_advance_stage_blocked_by_prerequisite(self, manager: GlobalCurriculumManager) -> None:
        """Test advance blocked when prerequisites not met."""
        # Start at stage 2 without mastering stage 1
        manager.start_training(StageID.RESOURCE_GATHERING)
        # Stage 3 requires stage 2 mastered
        result = manager.advance_stage()
        # Should fail because RESOURCE_GATHERING not mastered
        assert result is None

    def test_advance_stage_final(self, manager: GlobalCurriculumManager) -> None:
        """Test advancing from final stage returns None."""
        manager.start_training(StageID.END_FIGHT)
        manager.progress[StageID.END_FIGHT].mastered = True
        result = manager.advance_stage()
        assert result is None

    def test_regress_stage(self, manager: GlobalCurriculumManager) -> None:
        """Test regressing to previous stage."""
        manager.start_training(StageID.RESOURCE_GATHERING)
        prev_stage = manager.regress_stage()
        assert prev_stage is not None
        assert prev_stage.id == StageID.BASIC_SURVIVAL
        assert manager.current_stage == StageID.BASIC_SURVIVAL

    def test_regress_stage_specific(self, manager: GlobalCurriculumManager) -> None:
        """Test regressing to specific stage."""
        manager.start_training(StageID.NETHER_NAVIGATION)
        target = manager.regress_stage(StageID.BASIC_SURVIVAL)
        assert target is not None
        assert target.id == StageID.BASIC_SURVIVAL

    def test_regress_stage_from_first(self, manager: GlobalCurriculumManager) -> None:
        """Test regressing from first stage returns None."""
        manager.start_training(StageID.BASIC_SURVIVAL)
        result = manager.regress_stage()
        assert result is None

    def test_stage_change_callback(self, manager: GlobalCurriculumManager) -> None:
        """Test stage change callbacks are invoked."""
        callback_calls: list[tuple[StageID, StageID]] = []

        def callback(old: StageID, new: StageID) -> None:
            callback_calls.append((old, new))

        manager.on_stage_change(callback)
        manager.start_training(StageID.BASIC_SURVIVAL)
        manager.progress[StageID.BASIC_SURVIVAL].mastered = True
        manager.advance_stage()

        assert len(callback_calls) == 1
        assert callback_calls[0] == (StageID.BASIC_SURVIVAL, StageID.RESOURCE_GATHERING)

    def test_training_summary(self, manager: GlobalCurriculumManager) -> None:
        """Test training summary generation."""
        manager.start_training(StageID.BASIC_SURVIVAL)
        manager.record_episode(success=True, reward=10.0, ticks=1000)

        summary = manager.get_training_summary()
        assert summary["current_stage"] == "BASIC_SURVIVAL"
        assert summary["total_stages"] == 6
        assert summary["total_episodes"] == 1
        assert "BASIC_SURVIVAL" in summary["stage_progress"]

    def test_save_progress(self, manager: GlobalCurriculumManager) -> None:
        """Test saving progress to file."""
        manager.start_training(StageID.BASIC_SURVIVAL)
        manager.record_episode(success=True, reward=5.0, ticks=500)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            manager.save_progress(path)
            assert path.exists()

            with open(path) as f:
                data = json.load(f)
            assert data["current_stage"] == 1
            assert 1 in [int(k) for k in data["progress"].keys()]
        finally:
            path.unlink()

    def test_load_progress(self, manager: GlobalCurriculumManager) -> None:
        """Test loading progress from file."""
        manager.start_training(StageID.BASIC_SURVIVAL)
        for _ in range(10):
            manager.record_episode(success=True, reward=5.0, ticks=500)
        manager.progress[StageID.BASIC_SURVIVAL].mastered = True
        manager.advance_stage()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            manager.save_progress(path)

            # Create new manager and load
            new_manager = GlobalCurriculumManager(config_dir=None)
            for stage_id in StageID:
                new_manager.register_stage(manager.stages[stage_id])
            new_manager.load_progress(path)

            assert new_manager.current_stage == StageID.RESOURCE_GATHERING
            assert new_manager.progress[StageID.BASIC_SURVIVAL].mastered is True
            assert new_manager.progress[StageID.BASIC_SURVIVAL].episodes_completed == 10
        finally:
            path.unlink()


# =============================================================================
# Tests for VecCurriculumManager (curriculum_manager.py)
# =============================================================================


class TestVecCurriculumManager:
    """Tests for VecCurriculumManager (per-environment curriculum)."""

    def test_create_manager(self) -> None:
        """Manager can be created with specified num_envs."""
        manager = VecCurriculumManager(num_envs=8)
        assert manager.num_envs == 8

    def test_initial_stage(self) -> None:
        """Environments start at min_stage (default 1)."""
        manager = VecCurriculumManager(num_envs=8)
        for i in range(8):
            assert manager.get_stage(i) == 1

    def test_initial_stage_custom(self) -> None:
        """Environments start at custom min_stage."""
        manager = VecCurriculumManager(num_envs=4, min_stage=2)
        for i in range(4):
            assert manager.get_stage(i) == 2

    def test_success_tracking(self) -> None:
        """Success rate is tracked correctly per stage."""
        manager = VecCurriculumManager(num_envs=1)

        # Record 10 successes
        for _ in range(10):
            manager.update(0, success=True, stage=1, episode_length=100, reward=1.0)

        stats = manager.get_stats()
        assert stats["success_rates"][1] == 1.0

        # Record 10 failures
        for _ in range(10):
            manager.update(0, success=False, stage=1, episode_length=100, reward=0.0)

        stats = manager.get_stats()
        assert stats["success_rates"][1] == 0.5

    def test_advancement_threshold(self) -> None:
        """Environment advances after reaching threshold."""
        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=5,
            advancement_threshold=0.8,
        )

        # Initial stage
        assert manager.get_stage(0) == 1

        # Record successes (all pass, rate=1.0 > 0.8)
        for _ in range(10):
            advanced = manager.update(0, success=True, stage=1, episode_length=100, reward=1.0)

        # Should have advanced
        assert manager.get_stage(0) == 2

    def test_no_advancement_low_success(self) -> None:
        """Environment doesn't advance with low success rate."""
        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=5,
            advancement_threshold=0.8,
        )

        # Record mostly failures (rate < 0.8)
        for _ in range(20):
            manager.update(0, success=False, stage=1, episode_length=100, reward=0.0)

        # Should still be at stage 1
        assert manager.get_stage(0) == 1

    def test_advancement_requires_min_episodes(self) -> None:
        """Environment doesn't advance until min_episodes reached."""
        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=10,
            advancement_threshold=0.7,
        )

        # Record 5 successes (below min_episodes_to_advance)
        for _ in range(5):
            manager.update(0, success=True, stage=1, episode_length=100, reward=1.0)

        # Should not advance yet
        assert manager.get_stage(0) == 1

        # Record 5 more successes (now at 10 episodes)
        for _ in range(5):
            manager.update(0, success=True, stage=1, episode_length=100, reward=1.0)

        # Now should advance
        assert manager.get_stage(0) == 2

    def test_regression_disabled_by_default(self) -> None:
        """Regression is disabled by default."""
        manager = VecCurriculumManager(num_envs=1)
        manager.set_stage(0, 3)  # Start at stage 3

        # Record many failures
        for _ in range(100):
            manager.update(0, success=False, stage=3, episode_length=100, reward=0.0)

        # Should still be at stage 3 (regression disabled)
        assert manager.get_stage(0) == 3

    def test_regression_enabled(self) -> None:
        """Environment regresses when enabled and below threshold."""
        manager = VecCurriculumManager(
            num_envs=1,
            enable_regression=True,
            regression_threshold=0.2,
            min_episodes_to_regress=10,
        )
        manager.set_stage(0, 3)  # Start at stage 3

        # Record enough failures to trigger regression
        for _ in range(50):
            manager.update(0, success=False, stage=3, episode_length=100, reward=0.0)

        # Should regress to stage 2
        assert manager.get_stage(0) == 2

    def test_get_stats(self) -> None:
        """Statistics are computed correctly."""
        manager = VecCurriculumManager(num_envs=4)

        for i in range(4):
            manager.update(i, success=True, stage=1, episode_length=100, reward=1.0)

        stats = manager.get_stats()

        assert "stage_distribution" in stats
        assert "success_rates" in stats
        assert "stage_stats" in stats
        assert "total_episodes" in stats
        assert stats["total_episodes"] == 4

    def test_batch_update(self) -> None:
        """Batch updates work correctly."""
        manager = VecCurriculumManager(num_envs=4)

        env_ids = np.array([0, 1, 2, 3], dtype=np.int32)
        successes = np.array([True, True, False, True], dtype=np.bool_)

        advanced = manager.update_batch(env_ids, successes)

        assert advanced.shape == (4,)
        stats = manager.get_stats()
        assert stats["total_episodes"] == 4

    def test_get_stages(self) -> None:
        """Get all stages as array."""
        manager = VecCurriculumManager(num_envs=4)
        manager.set_stage(0, 1)
        manager.set_stage(1, 2)
        manager.set_stage(2, 3)
        manager.set_stage(3, 1)

        stages = manager.get_stages()
        assert stages.shape == (4,)
        assert list(stages) == [1, 2, 3, 1]

    def test_set_all_stages(self) -> None:
        """Set all environments to same stage."""
        manager = VecCurriculumManager(num_envs=4)
        manager.set_all_stages(3)

        for i in range(4):
            assert manager.get_stage(i) == 3

    def test_stage_clipping(self) -> None:
        """Stages are clipped to valid range."""
        manager = VecCurriculumManager(num_envs=1, min_stage=1, max_stage=6)

        manager.set_stage(0, 10)  # Above max
        assert manager.get_stage(0) == 6

        manager.set_stage(0, 0)  # Below min
        assert manager.get_stage(0) == 1

    def test_advancement_history(self) -> None:
        """Advancement events are recorded."""
        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=5,
            advancement_threshold=0.8,
        )

        # Trigger advancement
        for _ in range(10):
            manager.update(0, success=True, stage=1, episode_length=100, reward=1.0)

        events = manager.get_recent_advancements()
        assert len(events) >= 1
        assert events[-1].old_stage == 1
        assert events[-1].new_stage == 2
        assert events[-1].env_id == 0

    def test_reset(self) -> None:
        """Reset clears all state."""
        manager = VecCurriculumManager(num_envs=4)

        # Make some progress
        for _ in range(10):
            manager.update(0, success=True, stage=1, episode_length=100, reward=1.0)

        manager.set_stage(0, 3)

        # Reset
        manager.reset()

        assert manager.get_stage(0) == 1
        stats = manager.get_stats()
        assert stats["total_episodes"] == 0

    def test_save_load_state(self) -> None:
        """State can be saved and loaded."""
        manager = VecCurriculumManager(num_envs=4)

        # Make some progress
        manager.set_stage(0, 2)
        manager.set_stage(1, 3)
        for i in range(20):
            manager.update(i % 4, success=i % 2 == 0, stage=manager.get_stage(i % 4))

        # Save state
        state = manager.save_state()

        # Create new manager and load
        new_manager = VecCurriculumManager(num_envs=4)
        new_manager.load_state(state)

        assert new_manager.get_stage(0) == manager.get_stage(0)
        assert new_manager.get_stage(1) == manager.get_stage(1)
        assert new_manager._total_episodes == manager._total_episodes

    def test_stage_summary_string(self) -> None:
        """Stage summary produces readable output."""
        manager = VecCurriculumManager(num_envs=8)
        manager.set_stage(0, 2)
        manager.set_stage(1, 2)

        summary = manager.get_stage_summary()

        assert "Curriculum Summary" in summary
        assert "Stage 1" in summary
        assert "Stage 2" in summary


# =============================================================================
# Tests for StageStats
# =============================================================================


class TestStageStats:
    """Tests for StageStats dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        stats = StageStats()
        assert stats.total_episodes == 0
        assert stats.total_successes == 0
        assert stats.total_reward == 0.0
        assert stats.best_reward == float("-inf")
        assert stats.avg_episode_length == 0.0

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        stats = StageStats(total_episodes=100, total_successes=70)
        assert stats.success_rate == 0.7

    def test_success_rate_zero_episodes(self) -> None:
        """Test success rate with no episodes."""
        stats = StageStats()
        assert stats.success_rate == 0.0

    def test_avg_reward(self) -> None:
        """Test average reward calculation."""
        stats = StageStats(total_episodes=10, total_reward=50.0)
        assert stats.avg_reward == 5.0

    def test_avg_reward_zero_episodes(self) -> None:
        """Test average reward with no episodes."""
        stats = StageStats()
        assert stats.avg_reward == 0.0


# =============================================================================
# Tests for AdvancementEvent
# =============================================================================


class TestAdvancementEvent:
    """Tests for AdvancementEvent dataclass."""

    def test_creation(self) -> None:
        """Test event creation."""
        event = AdvancementEvent(
            env_id=5,
            old_stage=2,
            new_stage=3,
            timestamp=1000,
            success_rate=0.85,
        )
        assert event.env_id == 5
        assert event.old_stage == 2
        assert event.new_stage == 3
        assert event.timestamp == 1000
        assert event.success_rate == 0.85


# =============================================================================
# Integration tests
# =============================================================================


class TestCurriculumIntegration:
    """Integration tests for curriculum system."""

    def test_full_curriculum_progression(self) -> None:
        """Test progressing through entire curriculum."""
        manager = GlobalCurriculumManager(config_dir=None)

        # Register all stages
        for stage_id in StageID:
            stage = Stage(
                id=stage_id,
                name=stage_id.name,
                description=f"Stage {stage_id.value}",
                objectives=["complete"],
                spawn=SpawnConfig(),
                rewards=RewardConfig(),
                termination=TerminationConfig(),
                prerequisites=[StageID(stage_id.value - 1)] if stage_id.value > 1 else [],
                expected_episodes=10,  # Low for testing
                curriculum_threshold=0.6,
            )
            manager.register_stage(stage)

        # Start at first stage
        manager.start_training()
        stages_completed = 0

        # Simulate training through all stages
        for _ in range(100):  # Max iterations
            if manager.current_stage is None:
                break

            # Simulate episodes until mastery
            for _ in range(20):
                manager.record_episode(success=True, reward=1.0, ticks=100)

            if manager.should_advance():
                if manager.advance_stage() is None:
                    stages_completed += 1
                    break
                stages_completed += 1

        assert stages_completed >= 5  # Should complete most stages

    def test_vec_curriculum_multi_env_progression(self) -> None:
        """Test multiple environments progressing at different rates."""
        manager = VecCurriculumManager(
            num_envs=8,
            min_episodes_to_advance=10,
            advancement_threshold=0.7,
        )

        # Simulate training where env 0 always succeeds, env 7 always fails
        for episode in range(100):
            for env_id in range(8):
                # Success rate varies by env_id
                success = np.random.random() < (1.0 - env_id * 0.1)
                manager.update(env_id, success=success, stage=manager.get_stage(env_id))

        # Env 0 should be at higher stage than env 7
        stages = manager.get_stages()
        assert stages[0] >= stages[7]

    def test_curriculum_with_rewards_and_lengths(self) -> None:
        """Test curriculum tracking rewards and episode lengths."""
        manager = VecCurriculumManager(num_envs=4)

        rewards = [10.0, 5.0, 15.0, 8.0]
        lengths = [100, 200, 150, 120]

        for env_id in range(4):
            manager.update(
                env_id,
                success=True,
                stage=1,
                reward=rewards[env_id],
                episode_length=lengths[env_id],
            )

        stats = manager.get_stats()
        stage_stats = stats["stage_stats"][1]

        assert stage_stats["episodes"] == 4
        assert stage_stats["best_reward"] == 15.0
        assert stage_stats["avg_reward"] == np.mean(rewards)


# =============================================================================
# Tests for StageOverride and per-stage configuration
# =============================================================================


class TestStageOverride:
    """Tests for per-stage override functionality in VecCurriculumManager."""

    def test_stage1_reduced_min_episodes(self) -> None:
        """Stage 1 advances with fewer episodes when override is set."""
        from minecraft_sim.curriculum_manager import StageOverride

        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=20,
            advancement_threshold=0.7,
            stage_overrides={1: StageOverride(min_episodes_to_advance=5)},
        )

        # 5 successes should be enough for Stage 1 with override
        for _ in range(5):
            manager.update(0, success=True, stage=1, episode_length=100, reward=1.0)

        assert manager.get_stage(0) == 2

    def test_stage2_keeps_default_min_episodes(self) -> None:
        """Stage 2 still requires the global min_episodes_to_advance."""
        from minecraft_sim.curriculum_manager import StageOverride

        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=20,
            advancement_threshold=0.7,
            stage_overrides={1: StageOverride(min_episodes_to_advance=5)},
        )

        # Advance past stage 1
        for _ in range(5):
            manager.update(0, success=True, stage=1, episode_length=100, reward=1.0)
        assert manager.get_stage(0) == 2

        # 10 successes at stage 2 should NOT be enough (needs 20)
        for _ in range(10):
            manager.update(0, success=True, stage=2, episode_length=100, reward=1.0)
        assert manager.get_stage(0) == 2

        # 10 more (total 20) should advance
        for _ in range(10):
            manager.update(0, success=True, stage=2, episode_length=100, reward=1.0)
        assert manager.get_stage(0) == 3

    def test_stage1_threshold_override(self) -> None:
        """Stage 1 uses overridden advancement threshold."""
        from minecraft_sim.curriculum_manager import StageOverride

        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=10,
            advancement_threshold=0.7,
            stage_overrides={1: StageOverride(advancement_threshold=0.9)},
        )

        # 10 episodes with 80% success: above global 0.7 but below Stage 1's 0.9
        for i in range(10):
            manager.update(0, success=(i < 8), stage=1, episode_length=100, reward=1.0)

        assert manager.get_stage(0) == 1  # Should NOT advance

    def test_stage1_threshold_override_passes(self) -> None:
        """Stage 1 advances when overridden threshold is met."""
        from minecraft_sim.curriculum_manager import StageOverride

        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=10,
            advancement_threshold=0.7,
            stage_overrides={1: StageOverride(advancement_threshold=0.9)},
        )

        # 10 episodes with 100% success: above 0.9
        for _ in range(10):
            manager.update(0, success=True, stage=1, episode_length=100, reward=1.0)

        assert manager.get_stage(0) == 2

    def test_factory_with_metadata_threshold(self) -> None:
        """Factory function derives threshold from Stage 1 metadata."""
        from minecraft_sim.curriculum_manager import (
            create_vec_curriculum_with_stage1_overrides,
        )

        metadata = {"curriculum_threshold": 0.5, "target_completion_time": 12000}
        manager = create_vec_curriculum_with_stage1_overrides(
            num_envs=1,
            stage1_metadata=metadata,
            stage1_min_episodes=5,
            advancement_threshold=0.8,  # global default
        )

        # 5 episodes with 60% success: above metadata threshold (0.5)
        # but below global threshold (0.8)
        for i in range(5):
            manager.update(0, success=(i < 3), stage=1, episode_length=100, reward=1.0)

        assert manager.get_stage(0) == 2  # Uses metadata threshold 0.5

    def test_factory_without_metadata(self) -> None:
        """Factory function with no metadata uses global threshold."""
        from minecraft_sim.curriculum_manager import (
            create_vec_curriculum_with_stage1_overrides,
        )

        manager = create_vec_curriculum_with_stage1_overrides(
            num_envs=1,
            stage1_metadata=None,
            stage1_min_episodes=10,
            advancement_threshold=0.8,
        )

        # 10 episodes with 75% success: below global 0.8
        for i in range(10):
            manager.update(0, success=(i < 7), stage=1, episode_length=100, reward=1.0)

        # Stage 1 has no threshold override, uses global 0.8
        assert manager.get_stage(0) == 1

    def test_factory_reduced_min_episodes(self) -> None:
        """Factory function applies reduced min episodes to Stage 1."""
        from minecraft_sim.curriculum_manager import (
            create_vec_curriculum_with_stage1_overrides,
        )

        manager = create_vec_curriculum_with_stage1_overrides(
            num_envs=1,
            stage1_min_episodes=3,
            min_episodes_to_advance=50,  # high global default
        )

        # Only 3 episodes needed for Stage 1
        for _ in range(3):
            manager.update(0, success=True, stage=1, episode_length=100, reward=1.0)

        assert manager.get_stage(0) == 2

    def test_override_only_affects_specified_stage(self) -> None:
        """Overrides for stage 1 don't leak to other stages."""
        from minecraft_sim.curriculum_manager import StageOverride

        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=20,
            advancement_threshold=0.7,
            stage_overrides={
                1: StageOverride(min_episodes_to_advance=5, advancement_threshold=0.5),
            },
        )

        # Verify internal helper methods
        assert manager._get_stage_min_episodes(1) == 5
        assert manager._get_stage_min_episodes(2) == 20
        assert manager._get_stage_min_episodes(3) == 20
        assert manager._get_stage_threshold(1) == 0.5
        assert manager._get_stage_threshold(2) == 0.7
        assert manager._get_stage_threshold(3) == 0.7


# =============================================================================
# Tests for Stage 4 Portal Success Threshold Advancement
# =============================================================================


class TestStage4PortalSuccessThresholds:
    """Tests that Stage 4 advancement obeys portal success thresholds.

    Stage 4 (Enderman Hunting / Pearl Collection) requires sustained portal
    activation success rates before advancing. These tests verify that
    VecCurriculumManager stage_overrides for Stage 4 are enforced correctly.
    """

    def test_stage4_override_blocks_advancement_below_threshold(self) -> None:
        """Stage 4 should not advance when success rate is below the overridden threshold."""
        from minecraft_sim.curriculum_manager import StageOverride

        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=10,
            advancement_threshold=0.7,
            stage_overrides={
                4: StageOverride(
                    min_episodes_to_advance=20,
                    advancement_threshold=0.85,
                ),
            },
        )
        manager.set_stage(0, 4)

        # Record 20 episodes with 75% success (above default 0.7, below Stage 4's 0.85)
        for i in range(20):
            manager.update(
                0, success=(i % 4 != 0), stage=4, episode_length=100, reward=1.0
            )

        assert manager.get_stage(0) == 4

    def test_stage4_override_blocks_advancement_below_min_episodes(self) -> None:
        """Stage 4 should not advance until min_episodes override is reached."""
        from minecraft_sim.curriculum_manager import StageOverride

        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=5,
            advancement_threshold=0.7,
            stage_overrides={
                4: StageOverride(
                    min_episodes_to_advance=30,
                    advancement_threshold=0.8,
                ),
            },
        )
        manager.set_stage(0, 4)

        # Record 15 all-success episodes (above threshold, but below min_episodes=30)
        for _ in range(15):
            manager.update(0, success=True, stage=4, episode_length=100, reward=1.0)

        assert manager.get_stage(0) == 4, (
            "Should not advance: only 15 episodes, Stage 4 requires 30"
        )

    def test_stage4_advances_when_override_threshold_met(self) -> None:
        """Stage 4 advances once both min_episodes and threshold are satisfied."""
        from minecraft_sim.curriculum_manager import StageOverride

        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=10,
            advancement_threshold=0.7,
            stage_overrides={
                4: StageOverride(
                    min_episodes_to_advance=20,
                    advancement_threshold=0.85,
                ),
            },
        )
        manager.set_stage(0, 4)

        # Record 25 all-success episodes (rate=1.0 >= 0.85, episodes=25 >= 20)
        for _ in range(25):
            manager.update(0, success=True, stage=4, episode_length=100, reward=1.0)

        assert manager.get_stage(0) == 5, (
            "Should advance: 25 episodes with 100% success exceeds Stage 4 thresholds"
        )

    def test_stage4_advancement_event_records_correct_rate(self) -> None:
        """Advancement event for Stage 4 records the success rate at advancement."""
        from minecraft_sim.curriculum_manager import StageOverride

        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=10,
            advancement_threshold=0.7,
            stage_overrides={
                4: StageOverride(
                    min_episodes_to_advance=20,
                    advancement_threshold=0.8,
                ),
            },
        )
        manager.set_stage(0, 4)

        # 18 successes + 2 failures = 90% success rate (>= 0.8 threshold)
        for i in range(20):
            manager.update(
                0, success=(i < 18), stage=4, episode_length=100, reward=1.0
            )

        events = manager.get_recent_advancements()
        assert len(events) >= 1
        last_event = events[-1]
        assert last_event.old_stage == 4
        assert last_event.new_stage == 5
        assert last_event.success_rate >= 0.8

    def test_stage4_override_does_not_affect_other_stages(self) -> None:
        """Stage 4 overrides should not change advancement criteria for other stages."""
        from minecraft_sim.curriculum_manager import StageOverride

        manager = VecCurriculumManager(
            num_envs=2,
            min_episodes_to_advance=10,
            advancement_threshold=0.7,
            stage_overrides={
                4: StageOverride(
                    min_episodes_to_advance=30,
                    advancement_threshold=0.9,
                ),
            },
        )

        # Env 0 at Stage 3 (no override): should advance with default thresholds
        manager.set_stage(0, 3)
        for _ in range(15):
            manager.update(0, success=True, stage=3, episode_length=100, reward=1.0)
        assert manager.get_stage(0) == 4, "Stage 3 should advance with default threshold"

        # Env 1 at Stage 4 (override): same episode count should NOT advance
        manager.set_stage(1, 4)
        for _ in range(15):
            manager.update(1, success=True, stage=4, episode_length=100, reward=1.0)
        assert manager.get_stage(1) == 4, (
            "Stage 4 should NOT advance with only 15 episodes (needs 30)"
        )

    def test_stage4_global_curriculum_threshold_override(self) -> None:
        """GlobalCurriculumManager respects Stage 4 curriculum_threshold for mastery."""
        manager = GlobalCurriculumManager(config_dir=None)

        stage4 = Stage(
            id=StageID.ENDERMAN_HUNTING,
            name="Enderman Hunting",
            description="Stage 4: Collect pearls and activate portal",
            objectives=["collect 12 ender pearls", "activate end portal"],
            spawn=SpawnConfig(),
            rewards=RewardConfig(),
            termination=TerminationConfig(),
            prerequisites=[StageID.NETHER_NAVIGATION],
            expected_episodes=500,  # min_episodes = 500 // 10 = 50
            curriculum_threshold=0.85,
        )
        manager.register_stage(stage4)
        manager.start_training(StageID.ENDERMAN_HUNTING)

        # Record 50 episodes with 80% success rate (below 0.85 threshold)
        for i in range(50):
            manager.record_episode(
                success=(i % 5 != 0),  # 40/50 = 0.8
                reward=1.0,
                ticks=1000,
            )

        prog = manager.progress[StageID.ENDERMAN_HUNTING]
        assert prog.success_rate == 0.8
        assert prog.mastered is False, (
            "Should not master Stage 4 with 80% rate when threshold is 85%"
        )

        # Add more successes to exceed threshold: (40+N)/(50+N) >= 0.85 -> N >= 17
        for _ in range(20):
            manager.record_episode(success=True, reward=1.0, ticks=1000)

        prog = manager.progress[StageID.ENDERMAN_HUNTING]
        assert prog.success_rate > 0.85
        assert prog.mastered is True, (
            "Should master Stage 4 once rate exceeds the portal success threshold"
        )

    def test_stage4_mixed_portal_success_prevents_premature_advancement(self) -> None:
        """Intermittent portal failures keep Stage 4 from advancing prematurely."""
        from minecraft_sim.curriculum_manager import StageOverride

        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=10,
            advancement_threshold=0.7,
            stage_overrides={
                4: StageOverride(
                    min_episodes_to_advance=25,
                    advancement_threshold=0.85,
                ),
            },
        )
        manager.set_stage(0, 4)

        # 20 successes then 5 failures = 80% over 25 episodes (below 0.85)
        for _ in range(20):
            manager.update(0, success=True, stage=4, episode_length=100, reward=1.0)
        for _ in range(5):
            manager.update(0, success=False, stage=4, episode_length=100, reward=0.0)

        assert manager.get_stage(0) == 4, (
            "Should not advance with 80% rate over last 25 episodes"
        )

        # Add sustained success streak to push window above threshold
        for _ in range(30):
            manager.update(0, success=True, stage=4, episode_length=100, reward=1.0)

        assert manager.get_stage(0) == 5, (
            "Should advance after sustained success streak exceeds portal threshold"
        )

    def test_stage4_helper_methods_return_overrides(self) -> None:
        """Internal helper methods return Stage 4 override values correctly."""
        from minecraft_sim.curriculum_manager import StageOverride

        manager = VecCurriculumManager(
            num_envs=1,
            min_episodes_to_advance=10,
            advancement_threshold=0.7,
            stage_overrides={
                4: StageOverride(
                    min_episodes_to_advance=40,
                    advancement_threshold=0.9,
                ),
            },
        )

        assert manager._get_stage_min_episodes(3) == 10
        assert manager._get_stage_min_episodes(4) == 40
        assert manager._get_stage_min_episodes(5) == 10
        assert manager._get_stage_threshold(3) == 0.7
        assert manager._get_stage_threshold(4) == 0.9
        assert manager._get_stage_threshold(5) == 0.7


# =============================================================================
# Tests for Stage 2 metadata and observation_space
# =============================================================================


class TestStage2MetadataAndObservations:
    """Tests that Stage 2 (RESOURCE_GATHERING) exposes metadata and observation_space."""

    @pytest.fixture
    def stage2(self) -> Stage:
        """Load Stage 2 from YAML config."""
        import yaml

        config_path = (
            Path(__file__).parent.parent
            / "python"
            / "minecraft_sim"
            / "stage_configs"
            / "stage_2_resource_gathering.yaml"
        )
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return Stage.from_dict(data)

    def test_stage2_has_metadata(self, stage2: Stage) -> None:
        """Stage 2 exposes a non-empty metadata dict."""
        assert isinstance(stage2.metadata, dict)
        assert len(stage2.metadata) > 0

    def test_stage2_metadata_keys(self, stage2: Stage) -> None:
        """Stage 2 metadata contains expected mining-depth keys."""
        expected_keys = {
            "min_y_for_iron",
            "min_y_for_diamond",
            "optimal_diamond_y",
            "difficulty",
            "target_completion_time",
        }
        assert expected_keys.issubset(stage2.metadata.keys())

    def test_stage2_metadata_values(self, stage2: Stage) -> None:
        """Stage 2 metadata values are correct."""
        assert stage2.metadata["min_y_for_iron"] == 64
        assert stage2.metadata["min_y_for_diamond"] == 16
        assert stage2.metadata["optimal_diamond_y"] == 11
        assert stage2.metadata["difficulty"] == "normal"
        assert stage2.metadata["target_completion_time"] == 36000

    def test_stage2_has_observation_space(self, stage2: Stage) -> None:
        """Stage 2 exposes a non-empty observation_space list."""
        assert isinstance(stage2.observation_space, list)
        assert len(stage2.observation_space) > 0

    def test_stage2_observation_space_core(self, stage2: Stage) -> None:
        """Stage 2 observation_space contains core observations."""
        required_obs = [
            "position",
            "velocity",
            "health",
            "hunger",
            "inventory",
            "nearby_blocks",
            "y_level",
            "equipped_item",
        ]
        for obs in required_obs:
            assert obs in stage2.observation_space, f"Missing observation: {obs}"

    def test_stage2_observation_space_resource_specific(self, stage2: Stage) -> None:
        """Stage 2 observation_space includes resource-gathering-specific observations."""
        resource_obs = [
            "iron_ore_count",
            "iron_ingot_count",
            "diamond_count",
            "iron_ore_nearby",
            "diamond_ore_nearby",
            "ore_vein_direction",
        ]
        for obs in resource_obs:
            assert obs in stage2.observation_space, f"Missing resource observation: {obs}"

    def test_stage2_metadata_roundtrip(self, stage2: Stage) -> None:
        """Metadata survives to_dict/from_dict roundtrip."""
        d = stage2.to_dict()
        assert "metadata" in d
        assert d["metadata"] == stage2.metadata

        restored = Stage.from_dict(d)
        assert restored.metadata == stage2.metadata

    def test_stage2_observation_space_roundtrip(self, stage2: Stage) -> None:
        """observation_space survives to_dict/from_dict roundtrip."""
        d = stage2.to_dict()
        assert "observation_space" in d
        assert d["observation_space"] == stage2.observation_space

        restored = Stage.from_dict(d)
        assert restored.observation_space == stage2.observation_space

    def test_stage2_basic_fields_intact(self, stage2: Stage) -> None:
        """Metadata/observations do not break existing Stage fields."""
        assert stage2.id == StageID.RESOURCE_GATHERING
        assert stage2.name == "Resource Gathering"
        assert stage2.difficulty == 3
        assert stage2.curriculum_threshold == 0.65
        assert stage2.expected_episodes == 1000
        assert StageID.BASIC_SURVIVAL in stage2.prerequisites
        assert len(stage2.objectives) == 8
        assert stage2.spawn.inventory == {
            "wooden_pickaxe": 1,
            "wooden_sword": 1,
            "oak_log": 16,
        }

    def test_stage2_metadata_does_not_pollute_other_fields(self, stage2: Stage) -> None:
        """Metadata keys do not leak into other Stage attributes."""
        for key in stage2.metadata:
            if key == "difficulty":
                # 'difficulty' exists on Stage too but with different type (int vs str)
                assert isinstance(stage2.difficulty, int)
                assert isinstance(stage2.metadata["difficulty"], str)
            else:
                assert not hasattr(stage2, key)


# =============================================================================
# Tests for Stage 3 metadata progress_keys (portal & blaze rod tracking)
# =============================================================================


class TestStage3MetadataProgressKeys:
    """Tests that Stage 3 (NETHER_NAVIGATION) metadata exposes progress_keys for portal and blaze rod tracking."""

    @pytest.fixture
    def stage3(self) -> Stage:
        """Load Stage 3 from YAML config."""
        import yaml

import logging

logger = logging.getLogger(__name__)

        config_path = (
            Path(__file__).parent.parent
            / "python"
            / "minecraft_sim"
            / "stage_configs"
            / "stage_3_nether_navigation.yaml"
        )
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return Stage.from_dict(data)

    def test_stage3_metadata_has_progress_keys(self, stage3: Stage) -> None:
        """Stage 3 metadata contains a progress_keys dict."""
        assert "progress_keys" in stage3.metadata
        progress_keys = stage3.metadata["progress_keys"]
        assert isinstance(progress_keys, dict)
        assert len(progress_keys) > 0

    def test_stage3_progress_keys_portal_tracking(self, stage3: Stage) -> None:
        """Stage 3 progress_keys includes portal construction milestones."""
        progress_keys = stage3.metadata["progress_keys"]
        portal_keys = ["obsidian_mined", "portal_frame_placed", "portal_lit", "nether_entered"]
        for key in portal_keys:
            assert key in progress_keys, f"Missing portal progress key: {key}"

    def test_stage3_progress_keys_blaze_rod_tracking(self, stage3: Stage) -> None:
        """Stage 3 progress_keys includes blaze rod collection milestones."""
        progress_keys = stage3.metadata["progress_keys"]
        blaze_keys = ["blazes_killed", "blaze_rods_collected"]
        for key in blaze_keys:
            assert key in progress_keys, f"Missing blaze rod progress key: {key}"

    def test_stage3_progress_keys_fortress_tracking(self, stage3: Stage) -> None:
        """Stage 3 progress_keys includes fortress discovery."""
        progress_keys = stage3.metadata["progress_keys"]
        assert "fortress_found" in progress_keys

    def test_stage3_progress_keys_are_human_readable_labels(self, stage3: Stage) -> None:
        """Stage 3 progress_keys values are non-empty human-readable strings."""
        progress_keys = stage3.metadata["progress_keys"]
        for key, label in progress_keys.items():
            assert isinstance(label, str), f"progress_keys[{key!r}] is not a string"
            assert len(label) > 0, f"progress_keys[{key!r}] is empty"
            # Labels should be title-case or sentence-case, not snake_case
            assert "_" not in label, f"progress_keys[{key!r}] label looks like a code identifier: {label!r}"

    def test_stage3_progress_keys_roundtrip(self, stage3: Stage) -> None:
        """progress_keys survive to_dict/from_dict roundtrip."""
        d = stage3.to_dict()
        restored = Stage.from_dict(d)
        assert restored.metadata["progress_keys"] == stage3.metadata["progress_keys"]
