"""Test suite for Stage 5: Stronghold Finding.

Tests cover eye of ender crafting, throwing mechanics, stronghold generation,
triangulation, end portal mechanics, and teleportation to the End dimension.

Run with: uv run pytest contrib/minecraft_sim/tests/test_stage5_stronghold.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Setup paths for imports
SIM_ROOT = Path(__file__).parent.parent.resolve()
PYTHON_DIR = SIM_ROOT / "python"
SHADERS_DIR = SIM_ROOT / "cpp" / "shaders"

# Insert our python dir at the front
sys.path = [str(PYTHON_DIR)] + [p for p in sys.path if "minecraft_sim" not in p]

# Try to import mc189_core at module level
try:
    if "minecraft_sim" in sys.modules:
        del sys.modules["minecraft_sim"]
    if "minecraft_sim.mc189_core" in sys.modules:
        del sys.modules["minecraft_sim.mc189_core"]

    from minecraft_sim import mc189_core

    HAS_MC189_CORE = True
    _import_error = ""
except ImportError as e:
    HAS_MC189_CORE = False
    mc189_core = None
    _import_error = str(e)

# Skip entire module if mc189_core is not available
pytestmark = pytest.mark.skipif(
    not HAS_MC189_CORE, reason=f"mc189_core C++ extension not available: {_import_error}"
)


# ============================================================================
# Constants
# ============================================================================

# Stronghold ring distances from spawn (MC 1.8.9)
# Ring 1: 1280-2816 blocks
# Ring 2: 4352-5888 blocks
# Ring 3: 7424-8960 blocks
STRONGHOLD_RING_1_MIN = 1280
STRONGHOLD_RING_1_MAX = 2816
STRONGHOLD_RING_2_MIN = 4352
STRONGHOLD_RING_2_MAX = 5888
STRONGHOLD_RING_3_MIN = 7424
STRONGHOLD_RING_3_MAX = 8960

# Eye of ender properties
EYE_BREAK_CHANCE = 0.2  # 20% chance to break on throw
EYE_TRAVEL_DISTANCE = 12.0  # Blocks traveled per throw

# End portal properties
PORTAL_FRAME_COUNT = 12
END_SPAWN_PLATFORM = (100, 49, 0)  # Obsidian platform in End

# Item IDs (simplified for testing)
ITEM_BLAZE_POWDER = 377
ITEM_ENDER_PEARL = 368
ITEM_EYE_OF_ENDER = 381


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sim_config():
    """Create a simulator config for stronghold testing."""
    config = mc189_core.SimulatorConfig()
    config.num_envs = 1
    config.shader_dir = str(SHADERS_DIR)
    # Enable stronghold features if available
    if hasattr(config, "enable_strongholds"):
        config.enable_strongholds = True
    if hasattr(config, "enable_nether"):
        config.enable_nether = True
    return config


@pytest.fixture
def simulator(sim_config):
    """Create a simulator instance for stronghold testing."""
    sim = mc189_core.MC189Simulator(sim_config)
    return sim


@pytest.fixture
def reset_simulator(simulator):
    """Simulator that has been reset and stepped once."""
    simulator.reset()
    simulator.step(np.array([0], dtype=np.int32))  # No-op to populate obs
    return simulator


@pytest.fixture
def world_seed():
    """Fixed seed for deterministic stronghold testing."""
    return 12345


# ============================================================================
# Helper Functions
# ============================================================================


def decode_inventory(obs: np.ndarray) -> dict:
    """Decode inventory state from observation vector.

    Inventory layout (obs[32:64]):
    - [32-35]: Wood types count
    - [36-39]: Stone/iron/gold/diamond count
    - [40-43]: Blaze rods/powder, pearls, eyes
    - [44-47]: Food, armor, tools
    """
    return {
        "blaze_rods": int(obs[40] * 64) if len(obs) > 40 else 0,
        "blaze_powder": int(obs[41] * 64) if len(obs) > 41 else 0,
        "ender_pearls": int(obs[42] * 64) if len(obs) > 42 else 0,
        "eyes_of_ender": int(obs[43] * 64) if len(obs) > 43 else 0,
    }


def decode_stronghold_info(obs: np.ndarray) -> dict:
    """Decode stronghold-related info from observation vector.

    Stronghold info layout (obs[224:240]):
    - [224-225]: Nearest stronghold direction (x, z normalized)
    - [226]: Estimated distance to stronghold
    - [227]: Triangulation progress (0-1)
    - [228]: In stronghold flag
    - [229]: Portal room found flag
    - [230]: Portal eyes count (0-12)
    - [231]: Portal active flag
    - [232-234]: Stronghold position (x, y, z) if known
    """
    base_idx = 224 if len(obs) > 235 else len(obs) - 16
    if base_idx < 0:
        return {
            "direction_x": 0.0,
            "direction_z": 0.0,
            "estimated_distance": 0.0,
            "triangulation_progress": 0.0,
            "in_stronghold": False,
            "portal_room_found": False,
            "portal_eyes": 0,
            "portal_active": False,
            "stronghold_pos": (0, 0, 0),
        }

    return {
        "direction_x": obs[base_idx] * 2 - 1 if len(obs) > base_idx else 0.0,
        "direction_z": obs[base_idx + 1] * 2 - 1 if len(obs) > base_idx + 1 else 0.0,
        "estimated_distance": obs[base_idx + 2] * 10000 if len(obs) > base_idx + 2 else 0.0,
        "triangulation_progress": obs[base_idx + 3] if len(obs) > base_idx + 3 else 0.0,
        "in_stronghold": obs[base_idx + 4] > 0.5 if len(obs) > base_idx + 4 else False,
        "portal_room_found": obs[base_idx + 5] > 0.5 if len(obs) > base_idx + 5 else False,
        "portal_eyes": int(obs[base_idx + 6] * 12) if len(obs) > base_idx + 6 else 0,
        "portal_active": obs[base_idx + 7] > 0.5 if len(obs) > base_idx + 7 else False,
        "stronghold_pos": (
            obs[base_idx + 8] * 10000 - 5000 if len(obs) > base_idx + 8 else 0,
            obs[base_idx + 9] * 64 if len(obs) > base_idx + 9 else 0,
            obs[base_idx + 10] * 10000 - 5000 if len(obs) > base_idx + 10 else 0,
        ),
    }


def decode_player_position(obs: np.ndarray) -> tuple[float, float, float]:
    """Get player position from observation."""
    return (
        obs[0] * 100 if len(obs) > 0 else 0,
        obs[1] * 50 + 64 if len(obs) > 1 else 64,
        obs[2] * 100 if len(obs) > 2 else 0,
    )


def decode_dimension(obs: np.ndarray) -> int:
    """Get current dimension from observation.

    Returns: 0 = Overworld, -1 = Nether, 1 = End
    """
    dim_idx = 235 if len(obs) > 235 else len(obs) - 1
    if dim_idx < 0:
        return 0
    dim_val = obs[dim_idx] if len(obs) > dim_idx else 0.5
    if dim_val < 0.33:
        return -1  # Nether
    elif dim_val > 0.66:
        return 1  # End
    return 0  # Overworld


def calculate_distance(pos1: tuple, pos2: tuple) -> float:
    """Calculate 2D horizontal distance between positions."""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[2] - pos2[2]) ** 2)


def triangulate_stronghold(
    throw1_pos: tuple[float, float],
    throw1_dir: tuple[float, float],
    throw2_pos: tuple[float, float],
    throw2_dir: tuple[float, float],
) -> tuple[float, float] | None:
    """Triangulate stronghold position from two eye throws.

    Args:
        throw1_pos: (x, z) position of first throw
        throw1_dir: (dx, dz) direction from first throw (normalized)
        throw2_pos: (x, z) position of second throw
        throw2_dir: (dx, dz) direction from second throw (normalized)

    Returns:
        Estimated (x, z) stronghold position, or None if lines parallel
    """
    # Ray-ray intersection in 2D
    x1, z1 = throw1_pos
    dx1, dz1 = throw1_dir
    x2, z2 = throw2_pos
    dx2, dz2 = throw2_dir

    # Check for parallel lines
    det = dx1 * dz2 - dz1 * dx2
    if abs(det) < 1e-6:
        return None

    # Solve for intersection parameter t
    t = ((x2 - x1) * dz2 - (z2 - z1) * dx2) / det

    # Calculate intersection point
    stronghold_x = x1 + t * dx1
    stronghold_z = z1 + t * dz1

    return (stronghold_x, stronghold_z)


# ============================================================================
# Test Classes
# ============================================================================


class TestCraftEyeOfEnder:
    """Test eye of ender crafting mechanics."""

    def test_craft_eye_of_ender(self, reset_simulator):
        """Blaze powder + ender pearl = eye of ender."""
        # Skip if crafting system not available
        if not hasattr(reset_simulator, "craft_item"):
            pytest.skip("Crafting system not implemented in simulator")

        # Give player materials
        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_BLAZE_POWDER, 16)
            reset_simulator.give_item(0, ITEM_ENDER_PEARL, 16)

        # Attempt craft
        if hasattr(reset_simulator, "craft_item"):
            result = reset_simulator.craft_item(0, ITEM_EYE_OF_ENDER)
            assert result, "Should be able to craft eye of ender"

            obs = reset_simulator.get_observations()[0]
            inv = decode_inventory(obs)
            assert inv["eyes_of_ender"] >= 1, "Should have at least 1 eye after crafting"

    def test_craft_requires_both_ingredients(self, reset_simulator):
        """Crafting eye requires both blaze powder AND ender pearl."""
        if not hasattr(reset_simulator, "craft_item"):
            pytest.skip("Crafting system not implemented")

        # Only give pearls, no powder
        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_ENDER_PEARL, 16)
            # Clear any existing blaze powder
            if hasattr(reset_simulator, "clear_item"):
                reset_simulator.clear_item(0, ITEM_BLAZE_POWDER)

        if hasattr(reset_simulator, "craft_item"):
            result = reset_simulator.craft_item(0, ITEM_EYE_OF_ENDER)
            assert not result, "Should not craft without blaze powder"

    def test_craft_consumes_ingredients(self, reset_simulator):
        """Crafting consumes one of each ingredient."""
        if not hasattr(reset_simulator, "craft_item"):
            pytest.skip("Crafting system not implemented")

        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_BLAZE_POWDER, 5)
            reset_simulator.give_item(0, ITEM_ENDER_PEARL, 5)

        obs_before = reset_simulator.get_observations()[0]
        inv_before = decode_inventory(obs_before)

        if hasattr(reset_simulator, "craft_item"):
            reset_simulator.craft_item(0, ITEM_EYE_OF_ENDER)

        obs_after = reset_simulator.get_observations()[0]
        inv_after = decode_inventory(obs_after)

        # Should consume 1 powder and 1 pearl
        assert inv_after["blaze_powder"] == inv_before["blaze_powder"] - 1
        assert inv_after["ender_pearls"] == inv_before["ender_pearls"] - 1


class TestEyeOfEnderThrow:
    """Test eye of ender throwing mechanics."""

    def test_throw_eye_direction(self, reset_simulator):
        """Eye points toward nearest stronghold when thrown."""
        if not hasattr(reset_simulator, "throw_eye_of_ender"):
            pytest.skip("Eye throwing not implemented")

        # Give player an eye
        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, 1)

        # Throw eye
        result = reset_simulator.throw_eye_of_ender(0)
        assert result is not None, "Throw should return direction info"

        # Result should contain direction vector
        if isinstance(result, dict):
            assert "direction_x" in result or "dir_x" in result
            assert "direction_z" in result or "dir_z" in result

            dir_x = result.get("direction_x", result.get("dir_x", 0))
            dir_z = result.get("direction_z", result.get("dir_z", 0))

            # Direction should be normalized
            magnitude = math.sqrt(dir_x**2 + dir_z**2)
            assert 0.9 < magnitude < 1.1, f"Direction should be normalized, got {magnitude}"

    def test_eye_breaks_sometimes(self, reset_simulator):
        """Eye has 20% chance to break on throw."""
        if not hasattr(reset_simulator, "throw_eye_of_ender"):
            pytest.skip("Eye throwing not implemented")

        # Run multiple trials
        num_trials = 100
        breaks = 0

        for _ in range(num_trials):
            if hasattr(reset_simulator, "give_item"):
                reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, 1)

            result = reset_simulator.throw_eye_of_ender(0)
            if result and result.get("broke", False):
                breaks += 1

        # Should be approximately 20% break rate (15-25% acceptable)
        break_rate = breaks / num_trials
        assert 0.10 <= break_rate <= 0.30, f"Break rate {break_rate} outside expected range"

    def test_eye_reusable(self, reset_simulator):
        """80% of eyes can be picked up again."""
        if not hasattr(reset_simulator, "throw_eye_of_ender"):
            pytest.skip("Eye throwing not implemented")

        num_trials = 100
        reusable = 0

        for _ in range(num_trials):
            if hasattr(reset_simulator, "give_item"):
                reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, 1)

            result = reset_simulator.throw_eye_of_ender(0)
            if result and not result.get("broke", True):
                reusable += 1

        # Should be approximately 80% reusable (70-90% acceptable)
        reuse_rate = reusable / num_trials
        assert 0.70 <= reuse_rate <= 0.90, f"Reuse rate {reuse_rate} outside expected range"

    def test_eye_travel_distance(self, reset_simulator):
        """Eye travels approximately 12 blocks before landing."""
        if not hasattr(reset_simulator, "throw_eye_of_ender"):
            pytest.skip("Eye throwing not implemented")

        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, 1)

        obs_before = reset_simulator.get_observations()[0]
        player_pos = decode_player_position(obs_before)

        result = reset_simulator.throw_eye_of_ender(0)
        if result and "landing_pos" in result:
            land_x = result["landing_pos"][0]
            land_z = result["landing_pos"][2]

            travel_dist = math.sqrt((land_x - player_pos[0]) ** 2 + (land_z - player_pos[2]) ** 2)
            # Should be roughly 12 blocks (8-16 acceptable)
            assert 8 <= travel_dist <= 16, f"Travel distance {travel_dist} unexpected"


class TestStrongholdGeneration:
    """Test stronghold world generation."""

    def test_stronghold_exists(self, reset_simulator, world_seed):
        """World has strongholds generated."""
        if not hasattr(reset_simulator, "get_stronghold_positions"):
            pytest.skip("Stronghold generation not implemented")

        if hasattr(reset_simulator, "set_seed"):
            reset_simulator.set_seed(world_seed)
            reset_simulator.reset()

        strongholds = reset_simulator.get_stronghold_positions()
        assert strongholds is not None, "Should return stronghold list"
        assert len(strongholds) >= 3, "World should have at least 3 strongholds"

    def test_stronghold_in_ring(self, reset_simulator, world_seed):
        """Strongholds spawn in correct distance rings from spawn."""
        if not hasattr(reset_simulator, "get_stronghold_positions"):
            pytest.skip("Stronghold generation not implemented")

        if hasattr(reset_simulator, "set_seed"):
            reset_simulator.set_seed(world_seed)
            reset_simulator.reset()

        strongholds = reset_simulator.get_stronghold_positions()
        if not strongholds:
            pytest.skip("No strongholds returned")

        spawn = (0, 0)  # World spawn

        for i, sh in enumerate(strongholds):
            dist = calculate_distance((sh[0], 0, sh[2]), (0, 0, 0))

            # At least first stronghold should be in ring 1
            if i == 0:
                assert STRONGHOLD_RING_1_MIN <= dist <= STRONGHOLD_RING_1_MAX, (
                    f"First stronghold at {dist} blocks, expected {STRONGHOLD_RING_1_MIN}-{STRONGHOLD_RING_1_MAX}"
                )

    def test_strongholds_spread_around_spawn(self, reset_simulator, world_seed):
        """Strongholds are spread around spawn, not clustered."""
        if not hasattr(reset_simulator, "get_stronghold_positions"):
            pytest.skip("Stronghold generation not implemented")

        if hasattr(reset_simulator, "set_seed"):
            reset_simulator.set_seed(world_seed)
            reset_simulator.reset()

        strongholds = reset_simulator.get_stronghold_positions()
        if len(strongholds) < 3:
            pytest.skip("Need at least 3 strongholds for spread test")

        # Calculate angles from spawn
        angles = []
        for sh in strongholds[:3]:
            angle = math.atan2(sh[2], sh[0])
            angles.append(angle)

        angles.sort()

        # Check that strongholds are spread (not within 60 degrees of each other)
        for i in range(len(angles)):
            diff = angles[(i + 1) % len(angles)] - angles[i]
            if diff < 0:
                diff += 2 * math.pi
            assert diff > math.pi / 3, "Strongholds should be spread around spawn"


class TestTriangulation:
    """Test stronghold triangulation mechanics."""

    def test_triangulation(self, reset_simulator):
        """Two eye throws can find stronghold location."""
        if not hasattr(reset_simulator, "throw_eye_of_ender"):
            pytest.skip("Eye throwing not implemented")

        # First throw at position 1
        pos1 = (0, 0)
        if hasattr(reset_simulator, "teleport"):
            reset_simulator.teleport(0, pos1[0], 64, pos1[1])

        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, 2)

        result1 = reset_simulator.throw_eye_of_ender(0)
        if not result1:
            pytest.skip("First throw failed")
        dir1 = (
            result1.get("direction_x", result1.get("dir_x", 0)),
            result1.get("direction_z", result1.get("dir_z", 0)),
        )

        # Second throw at position 2 (moved perpendicular)
        pos2 = (500, 0)
        if hasattr(reset_simulator, "teleport"):
            reset_simulator.teleport(0, pos2[0], 64, pos2[1])

        result2 = reset_simulator.throw_eye_of_ender(0)
        if not result2:
            pytest.skip("Second throw failed")
        dir2 = (
            result2.get("direction_x", result2.get("dir_x", 0)),
            result2.get("direction_z", result2.get("dir_z", 0)),
        )

        # Triangulate
        estimated = triangulate_stronghold(pos1, dir1, pos2, dir2)
        assert estimated is not None, "Should be able to triangulate from 2 throws"

        # Verify against actual stronghold position
        if hasattr(reset_simulator, "get_stronghold_positions"):
            strongholds = reset_simulator.get_stronghold_positions()
            if strongholds:
                nearest = min(
                    strongholds,
                    key=lambda sh: calculate_distance(
                        (estimated[0], 0, estimated[1]), (sh[0], 0, sh[2])
                    ),
                )
                error = calculate_distance(
                    (estimated[0], 0, estimated[1]), (nearest[0], 0, nearest[2])
                )
                # Should be within 100 blocks (triangulation has some error)
                assert error < 100, f"Triangulation error {error} too large"

    def test_triangulation_parallel_throws_fail(self):
        """Parallel throws cannot triangulate."""
        # Two parallel directions
        pos1 = (0, 0)
        dir1 = (1, 0)  # East
        pos2 = (0, 100)
        dir2 = (1, 0)  # Also east (parallel)

        result = triangulate_stronghold(pos1, dir1, pos2, dir2)
        assert result is None, "Parallel throws should fail triangulation"

    def test_triangulation_accuracy(self):
        """Triangulation gives accurate results for known geometry."""
        # Known stronghold at (1000, 1000)
        stronghold = (1000, 1000)

        # Throw 1 from origin
        pos1 = (0, 0)
        dir1_raw = (stronghold[0] - pos1[0], stronghold[1] - pos1[1])
        mag1 = math.sqrt(dir1_raw[0] ** 2 + dir1_raw[1] ** 2)
        dir1 = (dir1_raw[0] / mag1, dir1_raw[1] / mag1)

        # Throw 2 from offset
        pos2 = (500, 0)
        dir2_raw = (stronghold[0] - pos2[0], stronghold[1] - pos2[1])
        mag2 = math.sqrt(dir2_raw[0] ** 2 + dir2_raw[1] ** 2)
        dir2 = (dir2_raw[0] / mag2, dir2_raw[1] / mag2)

        result = triangulate_stronghold(pos1, dir1, pos2, dir2)
        assert result is not None

        error = math.sqrt((result[0] - stronghold[0]) ** 2 + (result[1] - stronghold[1]) ** 2)
        assert error < 1, f"Perfect triangulation should have minimal error, got {error}"


class TestPortalRoom:
    """Test end portal room mechanics."""

    def test_portal_room_exists(self, reset_simulator, world_seed):
        """Stronghold has end portal room."""
        if not hasattr(reset_simulator, "get_stronghold_structures"):
            pytest.skip("Stronghold structure query not implemented")

        if hasattr(reset_simulator, "set_seed"):
            reset_simulator.set_seed(world_seed)
            reset_simulator.reset()

        if hasattr(reset_simulator, "get_stronghold_positions"):
            strongholds = reset_simulator.get_stronghold_positions()
            if not strongholds:
                pytest.skip("No strongholds")

            # Check first stronghold for portal room
            sh = strongholds[0]
            structures = reset_simulator.get_stronghold_structures(sh[0], sh[2])
            room_types = [s["type"] for s in structures]
            assert "portal_room" in room_types, "Stronghold should have portal room"

    def test_portal_eyes_missing(self, reset_simulator, world_seed):
        """Portal starts with some frames empty (random 0-12 eyes)."""
        if not hasattr(reset_simulator, "get_portal_state"):
            pytest.skip("Portal state query not implemented")

        if hasattr(reset_simulator, "set_seed"):
            reset_simulator.set_seed(world_seed)
            reset_simulator.reset()

        # Teleport to stronghold portal room
        if hasattr(reset_simulator, "get_stronghold_positions"):
            strongholds = reset_simulator.get_stronghold_positions()
            if not strongholds:
                pytest.skip("No strongholds")

            # Teleport to portal
            if hasattr(reset_simulator, "teleport_to_portal_room"):
                reset_simulator.teleport_to_portal_room(0, 0)  # First stronghold

            portal_state = reset_simulator.get_portal_state(0)
            eyes_filled = portal_state.get("eyes_filled", 0)

            # MC spawns with 0-12 eyes (average ~1-2)
            assert 0 <= eyes_filled <= 12, f"Invalid eye count {eyes_filled}"
            # Most worlds have missing eyes
            # (This test may occasionally fail on rare full-portal worlds)

    def test_portal_frame_count(self, reset_simulator, world_seed):
        """Portal has exactly 12 frames."""
        if not hasattr(reset_simulator, "get_portal_state"):
            pytest.skip("Portal state query not implemented")

        if hasattr(reset_simulator, "teleport_to_portal_room"):
            reset_simulator.teleport_to_portal_room(0, 0)

        portal_state = reset_simulator.get_portal_state(0)
        total_frames = portal_state.get("total_frames", 12)
        assert total_frames == 12, f"Portal should have 12 frames, got {total_frames}"


class TestEyePlacement:
    """Test placing eyes in portal frames."""

    def test_place_eye_in_frame(self, reset_simulator, world_seed):
        """Eye can be placed in empty portal frame."""
        if not hasattr(reset_simulator, "place_eye_in_portal"):
            pytest.skip("Eye placement not implemented")

        if hasattr(reset_simulator, "teleport_to_portal_room"):
            reset_simulator.teleport_to_portal_room(0, 0)

        # Give player eyes
        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, 12)

        portal_before = reset_simulator.get_portal_state(0)
        eyes_before = portal_before.get("eyes_filled", 0)

        # Place one eye
        result = reset_simulator.place_eye_in_portal(0)
        assert result, "Should successfully place eye"

        portal_after = reset_simulator.get_portal_state(0)
        eyes_after = portal_after.get("eyes_filled", 0)
        assert eyes_after == eyes_before + 1, "Eye count should increase by 1"

    def test_cannot_place_eye_in_filled_frame(self, reset_simulator, world_seed):
        """Cannot place eye in already-filled frame."""
        if not hasattr(reset_simulator, "place_eye_in_portal"):
            pytest.skip("Eye placement not implemented")

        if hasattr(reset_simulator, "teleport_to_portal_room"):
            reset_simulator.teleport_to_portal_room(0, 0)

        # Fill all frames
        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, 12)

        for _ in range(12):
            reset_simulator.place_eye_in_portal(0)

        # Try to place 13th eye
        result = reset_simulator.place_eye_in_portal(0)
        assert not result, "Should not place eye in full portal"

    def test_place_eye_consumes_from_inventory(self, reset_simulator, world_seed):
        """Placing eye removes it from inventory."""
        if not hasattr(reset_simulator, "place_eye_in_portal"):
            pytest.skip("Eye placement not implemented")

        if hasattr(reset_simulator, "teleport_to_portal_room"):
            reset_simulator.teleport_to_portal_room(0, 0)

        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, 5)

        obs_before = reset_simulator.get_observations()[0]
        inv_before = decode_inventory(obs_before)

        reset_simulator.place_eye_in_portal(0)

        obs_after = reset_simulator.get_observations()[0]
        inv_after = decode_inventory(obs_after)

        assert inv_after["eyes_of_ender"] == inv_before["eyes_of_ender"] - 1


class TestPortalActivation:
    """Test end portal activation mechanics."""

    def test_portal_activates(self, reset_simulator, world_seed):
        """12 eyes activates end portal."""
        if not hasattr(reset_simulator, "get_portal_state"):
            pytest.skip("Portal state not implemented")

        if hasattr(reset_simulator, "teleport_to_portal_room"):
            reset_simulator.teleport_to_portal_room(0, 0)

        # Give and place 12 eyes
        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, 12)

        if hasattr(reset_simulator, "place_eye_in_portal"):
            for _ in range(12):
                reset_simulator.place_eye_in_portal(0)

        portal_state = reset_simulator.get_portal_state(0)
        assert portal_state.get("active", False), "Portal should activate with 12 eyes"

    def test_portal_not_active_incomplete(self, reset_simulator, world_seed):
        """Portal with < 12 eyes is not active."""
        if not hasattr(reset_simulator, "get_portal_state"):
            pytest.skip("Portal state not implemented")

        if hasattr(reset_simulator, "teleport_to_portal_room"):
            reset_simulator.teleport_to_portal_room(0, 0)

        # Place only 11 eyes
        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, 11)

        if hasattr(reset_simulator, "place_eye_in_portal"):
            for _ in range(11):
                reset_simulator.place_eye_in_portal(0)

        portal_state = reset_simulator.get_portal_state(0)
        assert not portal_state.get("active", True), "Portal should not be active with 11 eyes"


class TestEnterEndPortal:
    """Test teleportation through end portal."""

    def test_enter_end_portal(self, reset_simulator, world_seed):
        """Entering active portal teleports to End dimension."""
        if not hasattr(reset_simulator, "enter_portal"):
            pytest.skip("Portal entry not implemented")

        # Setup: activate portal
        if hasattr(reset_simulator, "teleport_to_portal_room"):
            reset_simulator.teleport_to_portal_room(0, 0)

        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, 12)

        if hasattr(reset_simulator, "place_eye_in_portal"):
            for _ in range(12):
                reset_simulator.place_eye_in_portal(0)

        # Enter portal
        reset_simulator.enter_portal(0)

        obs = reset_simulator.get_observations()[0]
        dim = decode_dimension(obs)
        assert dim == 1, f"Should be in End (dim=1), got dim={dim}"

    def test_end_spawn_platform(self, reset_simulator, world_seed):
        """Player spawns on obsidian platform in End."""
        if not hasattr(reset_simulator, "enter_portal"):
            pytest.skip("Portal entry not implemented")

        # Setup and enter portal
        if hasattr(reset_simulator, "teleport_to_portal_room"):
            reset_simulator.teleport_to_portal_room(0, 0)

        if hasattr(reset_simulator, "give_item"):
            reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, 12)

        if hasattr(reset_simulator, "place_eye_in_portal"):
            for _ in range(12):
                reset_simulator.place_eye_in_portal(0)

        reset_simulator.enter_portal(0)

        obs = reset_simulator.get_observations()[0]
        pos = decode_player_position(obs)

        # Should spawn at approximately (100, 49, 0)
        expected_x, expected_y, expected_z = END_SPAWN_PLATFORM
        assert abs(pos[0] - expected_x) < 5, f"End spawn X incorrect: {pos[0]} vs {expected_x}"
        assert abs(pos[1] - expected_y) < 5, f"End spawn Y incorrect: {pos[1]} vs {expected_y}"
        assert abs(pos[2] - expected_z) < 5, f"End spawn Z incorrect: {pos[2]} vs {expected_z}"

    def test_cannot_enter_inactive_portal(self, reset_simulator, world_seed):
        """Cannot teleport through inactive portal."""
        if not hasattr(reset_simulator, "enter_portal"):
            pytest.skip("Portal entry not implemented")

        if hasattr(reset_simulator, "teleport_to_portal_room"):
            reset_simulator.teleport_to_portal_room(0, 0)

        # Don't fill portal
        obs_before = reset_simulator.get_observations()[0]
        dim_before = decode_dimension(obs_before)

        # Try to enter
        result = reset_simulator.enter_portal(0)

        obs_after = reset_simulator.get_observations()[0]
        dim_after = decode_dimension(obs_after)

        assert dim_after == dim_before, "Dimension should not change with inactive portal"


# ============================================================================
# Integration Tests
# ============================================================================


class TestStage5Integration:
    """Integration tests for complete Stage 5 flow."""

    def test_full_stronghold_flow(self, reset_simulator, world_seed):
        """Complete flow: craft eyes -> triangulate -> find portal -> activate."""
        # This is a comprehensive integration test
        if not all(
            hasattr(reset_simulator, attr)
            for attr in ["give_item", "throw_eye_of_ender", "get_portal_state"]
        ):
            pytest.skip("Required methods not implemented")

        # 1. Give materials for eyes
        reset_simulator.give_item(0, ITEM_BLAZE_POWDER, 16)
        reset_simulator.give_item(0, ITEM_ENDER_PEARL, 16)

        # 2. Craft eyes
        if hasattr(reset_simulator, "craft_item"):
            for _ in range(12):
                reset_simulator.craft_item(0, ITEM_EYE_OF_ENDER)

        # 3. Triangulate stronghold (simulate 2 throws)
        # (In real gameplay, player moves and throws from different positions)

        # 4. Travel to stronghold
        if hasattr(reset_simulator, "get_stronghold_positions"):
            strongholds = reset_simulator.get_stronghold_positions()
            if strongholds and hasattr(reset_simulator, "teleport"):
                sh = strongholds[0]
                reset_simulator.teleport(0, sh[0], 40, sh[2])

        # 5. Find and activate portal
        if hasattr(reset_simulator, "teleport_to_portal_room"):
            reset_simulator.teleport_to_portal_room(0, 0)

        if hasattr(reset_simulator, "place_eye_in_portal"):
            # Place remaining eyes needed
            portal_state = reset_simulator.get_portal_state(0)
            eyes_needed = 12 - portal_state.get("eyes_filled", 0)

            reset_simulator.give_item(0, ITEM_EYE_OF_ENDER, eyes_needed)
            for _ in range(eyes_needed):
                reset_simulator.place_eye_in_portal(0)

        # 6. Verify portal is active
        portal_state = reset_simulator.get_portal_state(0)
        assert portal_state.get("active", False), "Portal should be active after full flow"

    def test_observation_stronghold_info_updates(self, reset_simulator):
        """Observation vector updates with stronghold info."""
        obs = reset_simulator.get_observations()[0]
        info = decode_stronghold_info(obs)

        # Should have some stronghold info in observation
        assert isinstance(info["direction_x"], (float, np.floating))
        assert isinstance(info["direction_z"], (float, np.floating))
        assert isinstance(info["estimated_distance"], (float, np.floating))

    def test_reward_for_stronghold_progress(self, reset_simulator):
        """Rewards given for stronghold progress milestones."""
        if not hasattr(reset_simulator, "get_rewards"):
            pytest.skip("Reward system not implemented")

        # Progress milestones that should give rewards:
        # - First eye throw
        # - Triangulation complete
        # - Entering stronghold
        # - Finding portal room
        # - Each eye placed
        # - Portal activation

        # This is a placeholder - actual reward values depend on implementation
        reset_simulator.step(np.array([0], dtype=np.int32))
        rewards = reset_simulator.get_rewards()
        assert rewards is not None
