#!/usr/bin/env python3
"""Test all major module imports.

Run from FreeTheEnd/:
    PYTHONPATH=python:cpp/build python test_imports.py
"""

errors = []
successes = []


def test_import(name, import_func):
    try:
        import_func()
        successes.append(name)
        print(f"✓ {name}")
    except Exception as e:
        errors.append((name, str(e)))
        print(f"✗ {name}: {e}")


# Core
test_import("mc189_core", lambda: __import__("minecraft_sim").mc189_core)
test_import("MC189Simulator", lambda: __import__("minecraft_sim").MC189Simulator)
test_import("SimulatorConfig", lambda: __import__("minecraft_sim").SimulatorConfig)

# Environments
test_import("FreeTheEndEnv", lambda: __import__("minecraft_sim").FreeTheEndEnv)
test_import("VecFreeTheEndEnv", lambda: __import__("minecraft_sim").VecFreeTheEndEnv)
test_import("DragonFightEnv", lambda: __import__("minecraft_sim").DragonFightEnv)
test_import("VecDragonFightEnv", lambda: __import__("minecraft_sim").VecDragonFightEnv)

# Curriculum
test_import(
    "CurriculumManager", lambda: exec("from minecraft_sim.curriculum import CurriculumManager")
)
test_import("StageID", lambda: exec("from minecraft_sim.curriculum import StageID"))
test_import("VecCurriculumManager", lambda: __import__("minecraft_sim").VecCurriculumManager)

# Stage environments
test_import("BasicSurvivalEnv", lambda: __import__("minecraft_sim").BasicSurvivalEnv)
test_import("ResourceGatheringEnv", lambda: __import__("minecraft_sim").ResourceGatheringEnv)
test_import("NetherNavigationEnv", lambda: __import__("minecraft_sim").NetherNavigationEnv)
test_import("EndermanHuntingEnv", lambda: __import__("minecraft_sim").EndermanHuntingEnv)
test_import("StrongholdFindingEnv", lambda: __import__("minecraft_sim").StrongholdFindingEnv)

# Hierarchical RL
test_import("HierarchicalController", lambda: __import__("minecraft_sim").HierarchicalController)
test_import("SubGoalID", lambda: __import__("minecraft_sim").SubGoalID)
test_import(
    "HierarchicalRewardShaper", lambda: __import__("minecraft_sim").HierarchicalRewardShaper
)

# World model
test_import("TransitionBuffer", lambda: __import__("minecraft_sim").TransitionBuffer)
test_import("DreamEnv", lambda: __import__("minecraft_sim").DreamEnv)
test_import("WorldModelDataLoader", lambda: __import__("minecraft_sim").WorldModelDataLoader)

# Observations
test_import("ObservationEncoder", lambda: __import__("minecraft_sim").ObservationEncoder)
test_import(
    "CompactObservationEncoder", lambda: __import__("minecraft_sim").CompactObservationEncoder
)

# Replay
test_import("SpeedrunRecorder", lambda: __import__("minecraft_sim").SpeedrunRecorder)
test_import("SpeedrunReplayer", lambda: __import__("minecraft_sim").SpeedrunReplayer)

# Triangulation
test_import("triangulate_stronghold", lambda: __import__("minecraft_sim").triangulate_stronghold)

# Training config
test_import(
    "TrainingConfig", lambda: exec("from minecraft_sim.training_config import TrainingConfig")
)

# Reward shaping
test_import("RewardShaper", lambda: exec("from minecraft_sim.reward_shaping import RewardShaper"))

print(f"\n{'=' * 50}")
print(f"Results: {len(successes)} passed, {len(errors)} failed")
if errors:
    print("\nFailed imports:")
    for name, err in errors:
        print(f"  - {name}: {err[:100]}")
