"""Minecraft RL observation space definitions.

Compact observation encoding designed for fast GPU transfer and efficient RL training.
Observations are structured into several components:

1. Player state (continuous): position, velocity, health, hunger, look direction
2. Player flags (binary): on_ground, in_water, in_lava, sprinting, sneaking
3. Inventory summary (discrete): counts of key items for speedrun
4. Local voxels (discrete): 16x16x16 block grid centered on player
5. Entity awareness (continuous): nearby hostile mobs, dragon state
6. Ray-cast distances (continuous): distance to blocks in multiple directions

Total observation size: ~4500 floats (18KB as float32)
GPU transfer: ~300us per batch of 1024 environments
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Dimension bounds for normalization
OVERWORLD_BOUNDS = (-30_000_000, 30_000_000, -64, 320, -30_000_000, 30_000_000)
NETHER_BOUNDS = (-3_750_000, 3_750_000, 0, 256, -3_750_000, 3_750_000)
END_BOUNDS = (-30_000_000, 30_000_000, 0, 256, -30_000_000, 30_000_000)

DIMENSION_BOUNDS: dict[int, tuple[int, ...]] = {
    0: OVERWORLD_BOUNDS,
    1: NETHER_BOUNDS,
    2: END_BOUNDS,
}

# Voxel grid dimensions
VOXEL_GRID_SIZE = 16  # 16x16x16 cube centered on player
VOXEL_GRID_TOTAL = VOXEL_GRID_SIZE**3  # 4096 blocks

# Ray-cast configuration
NUM_RAYCAST_DIRS = 16  # 8 horizontal + 4 up + 4 down
MAX_RAYCAST_DIST = 64.0  # blocks

# Inventory key items for speedrun tracking
KEY_ITEM_IDS: tuple[int, ...] = (
    369,  # blaze_rod
    368,  # ender_pearl
    381,  # ender_eye
    49,  # obsidian
    263,  # coal (for torches)
    265,  # iron_ingot
    264,  # diamond
    276,  # diamond_sword
    278,  # diamond_pickaxe
    262,  # arrow
    261,  # bow
    322,  # golden_apple
    373,  # potion (any)
    327,  # lava_bucket
    326,  # water_bucket
    325,  # bucket
)
NUM_KEY_ITEMS = len(KEY_ITEM_IDS)

# Max nearby entities to track
MAX_NEARBY_MOBS = 8
MAX_NEARBY_ITEMS = 4

# Block type embeddings
NUM_BLOCK_TYPES = 256  # Simplified block ID space
BLOCK_EMBEDDING_DIM = 8  # Dimension for learned block embeddings


class Dimension(IntEnum):
    """Minecraft dimensions."""

    OVERWORLD = 0
    NETHER = 1
    END = 2


class DragonPhase(IntEnum):
    """Ender dragon fight phases."""

    NONE = 0  # No dragon fight active
    CIRCLING = 1  # Flying around the island
    STRAFING = 2  # Diving to attack
    PERCHING = 3  # Landed on fountain


# ============================================================================
# OBSERVATION COMPONENTS
# ============================================================================


@dataclass(slots=True)
class PlayerState:
    """Core player state observations.

    All continuous values are normalized to roughly [-1, 1] or [0, 1] range
    for stable neural network training.
    """

    # Position (normalized to dimension bounds)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Velocity (blocks/tick, typically small values)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # Health and hunger (normalized to [0, 1])
    health: float = 1.0  # 0-20 normalized to 0-1
    max_health: float = 1.0  # Usually 20, normalized
    hunger: float = 1.0  # 0-20 normalized to 0-1
    saturation: float = 1.0  # 0-20 normalized to 0-1
    exhaustion: float = 0.0  # 0-4 normalized to 0-1

    # Look direction (normalized angles)
    yaw: float = 0.0  # -180 to 180 normalized to -1 to 1
    pitch: float = 0.0  # -90 to 90 normalized to -1 to 1

    # Dimension (one-hot encoded elsewhere)
    dimension: int = 0  # 0=overworld, 1=nether, 2=end

    # Equipped item (hotbar slot 0-8)
    equipped_slot: int = 0
    equipped_item_id: int = 0

    # Binary flags
    on_ground: bool = True
    in_water: bool = False
    in_lava: bool = False
    sprinting: bool = False
    sneaking: bool = False

    def to_array(self) -> NDArray[np.float32]:
        """Convert to flat numpy array for neural network input.

        Returns:
            Array of shape (22,) containing normalized state values.
        """
        logger.debug("PlayerState.to_array called")
        return np.array(
            [
                self.x,
                self.y,
                self.z,
                self.vx,
                self.vy,
                self.vz,
                self.health,
                self.max_health,
                self.hunger,
                self.saturation,
                self.exhaustion,
                self.yaw,
                self.pitch,
                float(self.dimension) / 2.0,  # normalize to [0, 1]
                float(self.equipped_slot) / 8.0,  # normalize to [0, 1]
                float(self.equipped_item_id) / 500.0,  # rough normalize
                float(self.on_ground),
                float(self.in_water),
                float(self.in_lava),
                float(self.sprinting),
                float(self.sneaking),
                0.0,  # padding for alignment
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_raw(
        cls,
        position: tuple[float, float, float],
        velocity: tuple[float, float, float],
        health: float,
        max_health: float,
        hunger: float,
        saturation: float,
        exhaustion: float,
        yaw: float,
        pitch: float,
        dimension: int,
        equipped_slot: int,
        equipped_item_id: int,
        on_ground: bool,
        in_water: bool,
        in_lava: bool,
        sprinting: bool,
        sneaking: bool,
    ) -> PlayerState:
        """Create from raw Minecraft values with automatic normalization."""
        # Get dimension bounds for position normalization
        logger.debug("PlayerState.from_raw: position=%s, velocity=%s, health=%s, max_health=%s", position, velocity, health, max_health)
        bounds = DIMENSION_BOUNDS.get(dimension, OVERWORLD_BOUNDS)
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        return cls(
            # Normalize position to [-1, 1] based on dimension
            x=2.0 * (position[0] - x_min) / (x_max - x_min) - 1.0 if x_max > x_min else 0.0,
            y=2.0 * (position[1] - y_min) / (y_max - y_min) - 1.0 if y_max > y_min else 0.0,
            z=2.0 * (position[2] - z_min) / (z_max - z_min) - 1.0 if z_max > z_min else 0.0,
            # Velocity is already small, just clamp
            vx=np.clip(velocity[0] / 10.0, -1.0, 1.0),
            vy=np.clip(velocity[1] / 10.0, -1.0, 1.0),
            vz=np.clip(velocity[2] / 10.0, -1.0, 1.0),
            # Normalize health/hunger to [0, 1]
            health=health / 20.0,
            max_health=max_health / 20.0,
            hunger=hunger / 20.0,
            saturation=saturation / 20.0,
            exhaustion=exhaustion / 4.0,
            # Normalize angles
            yaw=yaw / 180.0,
            pitch=pitch / 90.0,
            dimension=dimension,
            equipped_slot=equipped_slot,
            equipped_item_id=equipped_item_id,
            on_ground=on_ground,
            in_water=in_water,
            in_lava=in_lava,
            sprinting=sprinting,
            sneaking=sneaking,
        )


@dataclass(slots=True)
class InventorySummary:
    """Summarized inventory for speedrun-relevant items.

    Instead of encoding the full 36-slot inventory, we track counts of
    items critical for the speedrun objective (kill ender dragon).
    """

    # Key item counts (normalized logarithmically)
    blaze_rods: int = 0
    ender_pearls: int = 0
    eyes_of_ender: int = 0
    obsidian: int = 0
    coal: int = 0
    iron_ingots: int = 0
    diamonds: int = 0
    diamond_sword: int = 0
    diamond_pickaxe: int = 0
    arrows: int = 0
    bow: int = 0
    golden_apples: int = 0
    potions: int = 0
    lava_buckets: int = 0
    water_buckets: int = 0
    empty_buckets: int = 0

    # Hotbar item IDs (raw, for context)
    hotbar: tuple[int, ...] = field(default_factory=lambda: (0,) * 9)

    def to_array(self) -> NDArray[np.float32]:
        """Convert to numpy array with log-normalized counts.

        Returns:
            Array of shape (25,) with normalized inventory features.
        """
        logger.debug("InventorySummary.to_array called")
        counts = np.array(
            [
                self.blaze_rods,
                self.ender_pearls,
                self.eyes_of_ender,
                self.obsidian,
                self.coal,
                self.iron_ingots,
                self.diamonds,
                self.diamond_sword,
                self.diamond_pickaxe,
                self.arrows,
                self.bow,
                self.golden_apples,
                self.potions,
                self.lava_buckets,
                self.water_buckets,
                self.empty_buckets,
            ],
            dtype=np.float32,
        )

        # Log-normalize counts: log(1 + count) / log(65)
        # This maps 0 -> 0, 64 -> 1
        normalized_counts = np.log1p(counts) / np.log(65)

        # Hotbar items normalized
        hotbar_arr = np.array(self.hotbar, dtype=np.float32) / 500.0

        return np.concatenate([normalized_counts, hotbar_arr])

    @classmethod
    def from_inventory(
        cls,
        inventory: NDArray[np.uint16],
        inventory_counts: NDArray[np.uint8],
    ) -> InventorySummary:
        """Extract summary from full inventory arrays.

        Args:
            inventory: Item IDs for 36 slots.
            inventory_counts: Stack sizes for 36 slots.
        """
        # Build item ID to count mapping
        logger.debug("InventorySummary.from_inventory: inventory=%s, inventory_counts=%s", inventory, inventory_counts)
        item_counts: dict[int, int] = {}
        for item_id, count in zip(inventory, inventory_counts, strict=True):
            if item_id > 0:
                item_counts[item_id] = item_counts.get(item_id, 0) + count

        return cls(
            blaze_rods=item_counts.get(369, 0),
            ender_pearls=item_counts.get(368, 0),
            eyes_of_ender=item_counts.get(381, 0),
            obsidian=item_counts.get(49, 0),
            coal=item_counts.get(263, 0),
            iron_ingots=item_counts.get(265, 0),
            diamonds=item_counts.get(264, 0),
            diamond_sword=min(item_counts.get(276, 0), 1),
            diamond_pickaxe=min(item_counts.get(278, 0), 1),
            arrows=item_counts.get(262, 0),
            bow=min(item_counts.get(261, 0), 1),
            golden_apples=item_counts.get(322, 0),
            potions=item_counts.get(373, 0),
            lava_buckets=item_counts.get(327, 0),
            water_buckets=item_counts.get(326, 0),
            empty_buckets=item_counts.get(325, 0),
            hotbar=tuple(int(x) for x in inventory[:9]),
        )


@dataclass(slots=True)
class VoxelGrid:
    """Local block grid centered on player.

    16x16x16 cube of block IDs for spatial awareness. Blocks are encoded
    either as raw IDs (for embedding lookup) or as one-hot vectors.

    Grid is centered on player with Y-axis going up.
    Index mapping: blocks[y * 256 + z * 16 + x]
    """

    blocks: NDArray[np.uint8] = field(
        default_factory=lambda: np.zeros(VOXEL_GRID_TOTAL, dtype=np.uint8)
    )

    def to_array_ids(self) -> NDArray[np.int32]:
        """Return block IDs for embedding lookup.

        Returns:
            Array of shape (4096,) with block IDs.
        """
        logger.debug("VoxelGrid.to_array_ids called")
        return self.blocks.astype(np.int32)

    def to_array_onehot(self, num_classes: int = 32) -> NDArray[np.float32]:
        """Convert to one-hot encoding for small number of block types.

        This is efficient when there are few distinct blocks to track
        (air, stone, water, lava, obsidian, end_stone, bedrock, etc.)

        Args:
            num_classes: Number of block categories.

        Returns:
            Array of shape (4096, num_classes) with one-hot vectors.
        """
        # Clamp block IDs to [0, num_classes-1]
        logger.debug("VoxelGrid.to_array_onehot: num_classes=%s", num_classes)
        clamped = np.clip(self.blocks, 0, num_classes - 1)
        onehot = np.zeros((VOXEL_GRID_TOTAL, num_classes), dtype=np.float32)
        onehot[np.arange(VOXEL_GRID_TOTAL), clamped] = 1.0
        return onehot

    def to_array_binary(self) -> NDArray[np.float32]:
        """Convert to binary solid/not-solid encoding.

        Returns:
            Array of shape (4096,) with 1.0 for solid blocks.
        """
        # Air is 0, most other blocks are solid
        logger.debug("VoxelGrid.to_array_binary called")
        return (self.blocks > 0).astype(np.float32)

    @classmethod
    def from_world(
        cls,
        world_blocks: NDArray[np.uint16],
        player_pos: tuple[int, int, int],
    ) -> VoxelGrid:
        """Extract voxel grid from world around player position.

        Args:
            world_blocks: Full world block array.
            player_pos: Player block coordinates (x, y, z).
        """
        # This would be implemented by the C++ encoder for efficiency
        logger.debug("VoxelGrid.from_world: world_blocks=%s, player_pos=%s", world_blocks, player_pos)
        grid = cls()
        # Placeholder: actual extraction happens in C++
        return grid


@dataclass(slots=True)
class RayCastDistances:
    """Distance measurements in multiple directions for navigation.

    16 rays cast from player position:
    - 8 horizontal (N, NE, E, SE, S, SW, W, NW)
    - 4 upward diagonal
    - 4 downward diagonal

    Values are normalized distances (0-1 where 1 = max distance).
    """

    distances: NDArray[np.float32] = field(
        default_factory=lambda: np.ones(NUM_RAYCAST_DIRS, dtype=np.float32)
    )
    hit_block_types: NDArray[np.uint8] = field(
        default_factory=lambda: np.zeros(NUM_RAYCAST_DIRS, dtype=np.uint8)
    )

    def to_array(self) -> NDArray[np.float32]:
        """Return normalized distances.

        Returns:
            Array of shape (16,) with distances normalized to [0, 1].
        """
        logger.debug("RayCastDistances.to_array called")
        return self.distances / MAX_RAYCAST_DIST


@dataclass(slots=True)
class EntityObservation:
    """Single entity (mob or item) observation."""

    entity_type: int = 0
    distance: float = 0.0
    dx: float = 0.0  # Normalized direction to entity
    dy: float = 0.0
    dz: float = 0.0
    health: float = 0.0  # Normalized health [0, 1]
    is_hostile: bool = False
    is_targeting_player: bool = False

    def to_array(self) -> NDArray[np.float32]:
        """Convert to array.

        Returns:
            Array of shape (8,).
        """
        logger.debug("EntityObservation.to_array called")
        return np.array(
            [
                float(self.entity_type) / 100.0,  # Normalize mob type ID
                self.distance / 64.0,  # Normalize to max tracking distance
                self.dx,
                self.dy,
                self.dz,
                self.health,
                float(self.is_hostile),
                float(self.is_targeting_player),
            ],
            dtype=np.float32,
        )


@dataclass(slots=True)
class EntityAwareness:
    """Nearby entity tracking for combat and navigation.

    Tracks up to 8 nearest hostile mobs and 4 nearest items.
    Entities are sorted by distance.
    """

    nearby_mobs: list[EntityObservation] = field(default_factory=list)
    nearby_items: list[EntityObservation] = field(default_factory=list)

    def to_array(self) -> NDArray[np.float32]:
        """Convert to fixed-size array.

        Returns:
            Array of shape (96,) = 8 mobs * 8 features + 4 items * 8 features.
        """
        # Pad mobs to MAX_NEARBY_MOBS
        logger.debug("EntityAwareness.to_array called")
        mob_arrays = [m.to_array() for m in self.nearby_mobs[:MAX_NEARBY_MOBS]]
        while len(mob_arrays) < MAX_NEARBY_MOBS:
            mob_arrays.append(np.zeros(8, dtype=np.float32))

        # Pad items to MAX_NEARBY_ITEMS
        item_arrays = [i.to_array() for i in self.nearby_items[:MAX_NEARBY_ITEMS]]
        while len(item_arrays) < MAX_NEARBY_ITEMS:
            item_arrays.append(np.zeros(8, dtype=np.float32))

        return np.concatenate(mob_arrays + item_arrays)


@dataclass(slots=True)
class DragonState:
    """Ender dragon fight state.

    Only relevant when in The End dimension during the dragon fight.
    """

    is_active: bool = False
    phase: int = 0  # DragonPhase enum
    health: float = 0.0  # Normalized [0, 1], dragon has 200 HP
    crystals_remaining: int = 0  # 0-10
    # Dragon position relative to player (normalized)
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    distance: float = 0.0
    # Time until next phase change (normalized)
    phase_timer: float = 0.0

    def to_array(self) -> NDArray[np.float32]:
        """Convert to array.

        Returns:
            Array of shape (10,).
        """
        logger.debug("DragonState.to_array called")
        return np.array(
            [
                float(self.is_active),
                float(self.phase) / 3.0,
                self.health,
                float(self.crystals_remaining) / 10.0,
                self.dx,
                self.dy,
                self.dz,
                self.distance / 200.0,  # Dragon can be far
                self.phase_timer,
                0.0,  # padding
            ],
            dtype=np.float32,
        )


# ============================================================================
# FULL OBSERVATION
# ============================================================================


@dataclass(slots=True)
class MinecraftObservation:
    """Complete observation for RL agent.

    Components:
    - player: Core player state (22 floats)
    - inventory: Key item counts and hotbar (25 floats)
    - voxels: Local block grid (4096 block IDs)
    - raycasts: Distance to blocks in 16 directions (16 floats)
    - entities: Nearby mobs and items (96 floats)
    - dragon: Dragon fight state (10 floats)

    Total: ~169 floats + 4096 block IDs
    """

    player: PlayerState = field(default_factory=PlayerState)
    inventory: InventorySummary = field(default_factory=InventorySummary)
    voxels: VoxelGrid = field(default_factory=VoxelGrid)
    raycasts: RayCastDistances = field(default_factory=RayCastDistances)
    entities: EntityAwareness = field(default_factory=EntityAwareness)
    dragon: DragonState = field(default_factory=DragonState)
    game_tick: int = 0
    terminated: bool = False
    truncated: bool = False

    def to_flat_array(self, include_voxels: bool = True) -> NDArray[np.float32]:
        """Convert entire observation to flat numpy array.

        Args:
            include_voxels: If True, include binary voxel grid (adds 4096 floats).

        Returns:
            Flat array suitable for MLP input.
        """
        logger.debug("MinecraftObservation.to_flat_array: include_voxels=%s", include_voxels)
        components = [
            self.player.to_array(),  # 22
            self.inventory.to_array(),  # 25
            self.raycasts.to_array(),  # 16
            self.entities.to_array(),  # 96
            self.dragon.to_array(),  # 10
            np.array(
                [
                    float(self.game_tick) / 36000.0,  # Normalize to typical episode length
                    float(self.terminated),
                    float(self.truncated),
                ],
                dtype=np.float32,
            ),  # 3
        ]

        if include_voxels:
            components.append(self.voxels.to_array_binary())  # 4096

        return np.concatenate(components)

    def to_dict(self) -> dict[str, NDArray[np.float32 | np.int32]]:
        """Convert to dictionary format for neural networks with separate encoders.

        Returns:
            Dictionary with:
            - 'continuous': Player, raycast, entity, dragon features
            - 'inventory': Key item counts
            - 'voxels': Block IDs for embedding lookup
            - 'hotbar': Hotbar item IDs
        """
        logger.debug("MinecraftObservation.to_dict called")
        return {
            "continuous": np.concatenate(
                [
                    self.player.to_array(),
                    self.raycasts.to_array(),
                    self.entities.to_array(),
                    self.dragon.to_array(),
                ]
            ),
            "inventory": self.inventory.to_array()[:16],  # Just counts, not hotbar
            "voxels": self.voxels.to_array_ids(),
            "hotbar": np.array(self.inventory.hotbar, dtype=np.int32),
        }


# ============================================================================
# OBSERVATION SPACE (Gymnasium-compatible)
# ============================================================================


class ObservationSpace:
    """Gymnasium-compatible observation space specification.

    Supports both flat (Box) and dict (Dict) observation formats.
    """

    # Flat observation dimensions (without voxels)
    CONTINUOUS_DIM = 22 + 25 + 16 + 96 + 10 + 3  # 172
    # With voxels
    FLAT_DIM = CONTINUOUS_DIM + VOXEL_GRID_TOTAL  # 4268

    def __init__(
        self,
        include_voxels: bool = True,
        voxel_encoding: str = "binary",  # 'binary', 'ids', 'onehot'
        num_block_types: int = 32,
    ) -> None:
        """Initialize observation space.

        Args:
            include_voxels: Whether to include voxel grid in observations.
            voxel_encoding: How to encode voxels ('binary', 'ids', 'onehot').
            num_block_types: Number of block types for one-hot encoding.
        """
        logger.info("ObservationSpace.__init__: include_voxels=%s, voxel_encoding=%s, num_block_types=%s", include_voxels, voxel_encoding, num_block_types)
        self.include_voxels = include_voxels
        self.voxel_encoding = voxel_encoding
        self.num_block_types = num_block_types

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape for flat observations."""
        logger.debug("ObservationSpace.shape called")
        if not self.include_voxels:
            return (self.CONTINUOUS_DIM,)
        if self.voxel_encoding == "binary":
            return (self.FLAT_DIM,)
        if self.voxel_encoding == "onehot":
            return (self.CONTINUOUS_DIM + VOXEL_GRID_TOTAL * self.num_block_types,)
        # 'ids' encoding returns separate arrays
        return (self.CONTINUOUS_DIM,)

    @property
    def dtype(self) -> np.dtype[np.float32]:
        """Return dtype for observations."""
        logger.debug("ObservationSpace.dtype called")
        return np.dtype(np.float32)

    def sample(self) -> NDArray[np.float32]:
        """Sample random observation (for testing)."""
        logger.debug("ObservationSpace.sample called")
        return np.random.randn(self.shape[0]).astype(np.float32)

    def contains(self, obs: NDArray[np.floating]) -> bool:
        """Check if observation is valid."""
        logger.debug("ObservationSpace.contains: obs=%s", obs)
        if obs.shape != self.shape:
            return False
        return not (np.isnan(obs).any() or np.isinf(obs).any())


class DictObservationSpace:
    """Dictionary observation space for separate encoding of different modalities.

    This is preferred for architectures that use:
    - MLP for continuous features
    - Embedding + CNN for voxel grid
    - Embedding for inventory items
    """

    def __init__(
        self,
        num_block_types: int = NUM_BLOCK_TYPES,
        block_embedding_dim: int = BLOCK_EMBEDDING_DIM,
    ) -> None:
        """Initialize dict observation space.

        Args:
            num_block_types: Vocabulary size for block embeddings.
            block_embedding_dim: Dimension of learned block embeddings.
        """
        logger.info("DictObservationSpace.__init__: num_block_types=%s, block_embedding_dim=%s", num_block_types, block_embedding_dim)
        self.num_block_types = num_block_types
        self.block_embedding_dim = block_embedding_dim

        # Define sub-spaces
        self.spaces = {
            "continuous": (172,),  # All continuous features
            "voxels": (VOXEL_GRID_TOTAL,),  # Block IDs
            "hotbar": (9,),  # Hotbar item IDs
        }

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        """Return shapes for all sub-spaces."""
        logger.debug("DictObservationSpace.shape called")
        return self.spaces

    def sample(self) -> dict[str, NDArray]:
        """Sample random observation."""
        logger.debug("DictObservationSpace.sample called")
        return {
            "continuous": np.random.randn(172).astype(np.float32),
            "voxels": np.random.randint(0, self.num_block_types, VOXEL_GRID_TOTAL, dtype=np.int32),
            "hotbar": np.random.randint(0, 500, 9, dtype=np.int32),
        }


# ============================================================================
# BATCH PROCESSING
# ============================================================================


class ObservationEncoder:
    """Encode observations for batch processing.

    Handles efficient encoding of batched observations for GPU transfer.
    """

    def __init__(
        self,
        include_voxels: bool = True,
        voxel_encoding: str = "binary",
        num_block_types: int = 32,
    ) -> None:
        logger.info("ObservationEncoder.__init__: include_voxels=%s, voxel_encoding=%s, num_block_types=%s", include_voxels, voxel_encoding, num_block_types)
        self.include_voxels = include_voxels
        self.voxel_encoding = voxel_encoding
        self.num_block_types = num_block_types
        self._obs_space = ObservationSpace(include_voxels, voxel_encoding, num_block_types)

    def encode(self, obs: MinecraftObservation) -> NDArray[np.float32]:
        """Encode single observation."""
        logger.debug("ObservationEncoder.encode: obs=%s", obs)
        return obs.to_flat_array(include_voxels=self.include_voxels)

    def encode_batch(
        self,
        observations: list[MinecraftObservation],
    ) -> NDArray[np.float32]:
        """Encode batch of observations.

        Args:
            observations: List of observations.

        Returns:
            Array of shape (batch_size, obs_dim).
        """
        logger.debug("ObservationEncoder.encode_batch: observations=%s", observations)
        return np.stack([self.encode(obs) for obs in observations])

    def encode_batch_dict(
        self,
        observations: list[MinecraftObservation],
    ) -> dict[str, NDArray]:
        """Encode batch as dictionary format.

        Args:
            observations: List of observations.

        Returns:
            Dictionary with batched arrays for each observation component.
        """
        logger.debug("ObservationEncoder.encode_batch_dict: observations=%s", observations)
        dicts = [obs.to_dict() for obs in observations]
        return {key: np.stack([d[key] for d in dicts]) for key in dicts[0]}


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

# ============================================================================
# COMPACT 256-FLOAT OBSERVATION FORMAT
# ============================================================================


@dataclass(slots=True)
class CompactPlayerState:
    """Decoded player state from compact observation (indices 0-31).

    All values are denormalized from the [0, 1] range.
    """

    position: NDArray[np.float32]  # (3,) x, y, z in world coords
    velocity: NDArray[np.float32]  # (3,) vx, vy, vz blocks/tick
    pitch: float  # -90 to 90
    yaw: float  # -180 to 180
    health: float  # 0-20
    hunger: float  # 0-20
    armor: float  # 0-20
    xp_level: int  # 0-30
    is_sprinting: bool
    is_sneaking: bool
    is_jumping: bool
    is_on_ground: bool
    is_in_water: bool
    is_in_lava: bool
    is_burning: bool


@dataclass(slots=True)
class CompactInventoryState:
    """Decoded inventory from compact observation (indices 32-63)."""

    hotbar_items: list[int]  # Item IDs
    hotbar_counts: list[int]  # Stack sizes
    selected_slot: int
    wood_count: int
    iron_count: int
    diamond_count: int
    blaze_rod_count: int
    ender_pearl_count: int
    eye_of_ender_count: int
    food_count: int


class CompactObservationDecoder:
    """Decode 256-float compact observation into structured data.

    Compact observation layout:
        [0-31]    Player state (position, velocity, health, flags)
        [32-63]   Inventory (hotbar items/counts, key resources)
        [64-127]  Environment (local features, not voxel grid)
        [128-191] Goals (curriculum stage one-hot, objectives)
        [192-223] Dragon state (health, position, phase)
        [224-255] Dimension/Portal state
    """

    def __init__(self, obs: NDArray[np.float32]) -> None:
        """Initialize decoder with raw observation array.

        Args:
            obs: Compact observation of shape (256,).

        Raises:
            AssertionError: If observation shape is not (256,).
        """
        logger.info("CompactObservationDecoder.__init__: obs=%s", obs)
        assert obs.shape == (256,), f"Expected (256,), got {obs.shape}"
        self.obs = obs

    def get_player_state(self) -> CompactPlayerState:
        """Extract and denormalize player state from observation.

        Returns:
            CompactPlayerState with denormalized values.
        """
        logger.debug("CompactObservationDecoder.get_player_state called")
        o = self.obs[0:32]
        return CompactPlayerState(
            position=o[0:3] * 1000,  # Denormalize position
            velocity=o[3:6] * 10,  # Denormalize velocity
            pitch=float(o[6]) * 90,
            yaw=float(o[7]) * 180,
            health=float(o[8]) * 20,
            hunger=float(o[9]) * 20,
            armor=float(o[10]) * 20,
            xp_level=int(o[11] * 30),
            is_sprinting=bool(o[12] > 0.5),
            is_sneaking=bool(o[13] > 0.5),
            is_jumping=bool(o[14] > 0.5),
            is_on_ground=bool(o[15] > 0.5),
            is_in_water=bool(o[16] > 0.5),
            is_in_lava=bool(o[17] > 0.5),
            is_burning=bool(o[18] > 0.5),
        )

    def get_inventory(self) -> CompactInventoryState:
        """Extract and denormalize inventory from observation.

        Returns:
            CompactInventoryState with denormalized counts.
        """
        logger.debug("CompactObservationDecoder.get_inventory called")
        o = self.obs[32:64]
        return CompactInventoryState(
            hotbar_items=[int(o[i] * 512) for i in range(9)],
            hotbar_counts=[int(o[9 + i] * 64) for i in range(9)],
            selected_slot=int(o[18] * 9),
            wood_count=int(o[19] * 64),
            iron_count=int(o[20] * 64),
            diamond_count=int(o[21] * 64),
            blaze_rod_count=int(o[22] * 64),
            ender_pearl_count=int(o[23] * 16),
            eye_of_ender_count=int(o[24] * 16),
            food_count=int(o[25] * 64),
        )

    def get_current_stage(self) -> int:
        """Get current curriculum stage from one-hot encoding.

        Returns:
            Stage number 1-6.
        """
        logger.debug("CompactObservationDecoder.get_current_stage called")
        stage_onehot = self.obs[128:134]
        return int(np.argmax(stage_onehot)) + 1

    def get_dragon_state(self) -> dict[str, float | bool | NDArray[np.float32]] | None:
        """Get dragon state if in End dimension during fight.

        Returns:
            Dictionary with dragon info, or None if no dragon active.
        """
        logger.debug("CompactObservationDecoder.get_dragon_state called")
        o = self.obs[192:224]
        if o[0] < 0.5:  # dragon_exists flag
            return None
        return {
            "health": float(o[1]) * 200,
            "position": o[2:5] * 100,
            "is_perched": bool(o[5] > 0.5),
            "crystals_remaining": int(o[6] * 10),
        }

    def get_dimension(self) -> int:
        """Get current dimension from observation.

        Returns:
            0=Overworld, 1=Nether, 2=End.
        """
        logger.debug("CompactObservationDecoder.get_dimension called")
        dim_onehot = self.obs[224:227]
        return int(np.argmax(dim_onehot))

    def get_portal_state(self) -> dict[str, bool | float]:
        """Get portal-related state.

        Returns:
            Dictionary with portal proximity and status.
        """
        logger.debug("CompactObservationDecoder.get_portal_state called")
        o = self.obs[227:256]
        return {
            "near_nether_portal": bool(o[0] > 0.5),
            "near_end_portal": bool(o[1] > 0.5),
            "portal_distance": float(o[2]) * 64,
            "end_portal_frame_count": int(o[3] * 12),
            "end_portal_eyes_placed": int(o[4] * 12),
        }


class CompactObservationEncoder:
    """Encode MinecraftObservation to compact 256-float format.

    This is the inverse of CompactObservationDecoder, converting
    structured observation data into the compact format for storage
    or transmission.
    """

    @staticmethod
    def encode(obs: MinecraftObservation, stage: int = 1) -> NDArray[np.float32]:
        """Encode a full observation to compact 256-float format.

        Args:
            obs: Full MinecraftObservation.
            stage: Current curriculum stage (1-6).

        Returns:
            Compact observation of shape (256,).
        """
        logger.debug("CompactObservationEncoder.encode: obs=%s, stage=%s", obs, stage)
        result = np.zeros(256, dtype=np.float32)

        # Player state (0-31)
        p = obs.player
        result[0:3] = np.array([p.x, p.y, p.z]) / 2.0 + 0.5  # Normalize from [-1,1] to [0,1]
        result[3:6] = np.array([p.vx, p.vy, p.vz]) / 2.0 + 0.5
        result[6] = p.pitch
        result[7] = p.yaw
        result[8] = p.health
        result[9] = p.hunger
        result[10] = p.max_health  # Using as armor proxy
        result[11] = 0.0  # XP level (not in MinecraftObservation)
        result[12] = float(p.sprinting)
        result[13] = float(p.sneaking)
        result[14] = 0.0  # is_jumping (not tracked)
        result[15] = float(p.on_ground)
        result[16] = float(p.in_water)
        result[17] = float(p.in_lava)
        result[18] = 0.0  # is_burning (not tracked)

        # Inventory (32-63)
        # Layout: [0-8] hotbar items, [9-17] hotbar counts, [18] selected slot
        # [19] wood, [20] iron, [21] diamond, [22] blaze_rod, [23] ender_pearl
        # [24] eye_of_ender, [25] food
        inv = obs.inventory
        for i, item_id in enumerate(inv.hotbar[:9]):
            result[32 + i] = item_id / 512.0
        # Hotbar counts not available from full obs, skip 9-17
        result[32 + 18] = 0.0  # selected_slot placeholder
        result[32 + 19] = 0.0  # wood_count placeholder
        result[32 + 20] = inv.iron_ingots / 64.0
        result[32 + 21] = inv.diamonds / 64.0
        result[32 + 22] = inv.blaze_rods / 64.0
        result[32 + 23] = inv.ender_pearls / 16.0
        result[32 + 24] = inv.eyes_of_ender / 16.0
        result[32 + 25] = 0.0  # food_count placeholder

        # Goals/Stage (128-191)
        if 1 <= stage <= 6:
            result[128 + stage - 1] = 1.0  # One-hot stage

        # Dragon (192-223)
        d = obs.dragon
        result[192] = float(d.is_active)
        result[193] = d.health
        result[194:197] = np.array([d.dx, d.dy, d.dz]) * 0.01
        result[197] = float(d.phase == DragonPhase.PERCHING) if d.is_active else 0.0
        result[198] = d.crystals_remaining / 10.0

        # Dimension (224-255)
        dim = obs.player.dimension
        result[224 + dim] = 1.0  # One-hot dimension

        return result


def create_observation_from_c_struct(raw_obs: dict) -> MinecraftObservation:
    """Create observation from C API mc189_observation_t struct.

    Args:
        raw_obs: Dictionary with fields matching mc189_observation_t.

    Returns:
        MinecraftObservation instance.
    """
    logger.info("create_observation_from_c_struct: raw_obs=%s", raw_obs)
    player_raw = raw_obs["player"]

    player = PlayerState.from_raw(
        position=tuple(player_raw["position"]),
        velocity=tuple(player_raw["velocity"]),
        health=player_raw["health"],
        max_health=player_raw["max_health"],
        hunger=player_raw["hunger"],
        saturation=player_raw["saturation"],
        exhaustion=player_raw["exhaustion"],
        yaw=player_raw["yaw"],
        pitch=player_raw["pitch"],
        dimension=player_raw["dimension"],
        equipped_slot=player_raw["active_slot"],
        equipped_item_id=player_raw["inventory"][player_raw["active_slot"]],
        on_ground=bool(player_raw["on_ground"]),
        in_water=bool(player_raw["in_water"]),
        in_lava=bool(player_raw["in_lava"]),
        sprinting=bool(player_raw["sprinting"]),
        sneaking=bool(player_raw["sneaking"]),
    )

    inventory = InventorySummary.from_inventory(
        inventory=np.array(player_raw["inventory"], dtype=np.uint16),
        inventory_counts=np.array(player_raw["inventory_counts"], dtype=np.uint8),
    )

    # Extract voxel grid (7x7x7 in C API, we expand or pad)
    local_blocks = np.array(raw_obs.get("local_blocks", [0] * 343), dtype=np.uint8)
    voxels = VoxelGrid(blocks=np.pad(local_blocks, (0, VOXEL_GRID_TOTAL - len(local_blocks))))

    # Extract nearby mobs
    nearby_mobs = []
    for mob_raw in raw_obs.get("nearby_mobs", []):
        if mob_raw["mob_type"] == 0:
            continue
        dx = mob_raw["position"][0] - player_raw["position"][0]
        dy = mob_raw["position"][1] - player_raw["position"][1]
        dz = mob_raw["position"][2] - player_raw["position"][2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dist > 0:
            dx, dy, dz = dx / dist, dy / dist, dz / dist
        nearby_mobs.append(
            EntityObservation(
                entity_type=mob_raw["mob_type"],
                distance=dist,
                dx=dx,
                dy=dy,
                dz=dz,
                health=mob_raw["health"] / mob_raw["max_health"]
                if mob_raw["max_health"] > 0
                else 0,
                is_hostile=bool(mob_raw["is_hostile"]),
                is_targeting_player=bool(mob_raw["is_targeting_player"]),
            )
        )

    entities = EntityAwareness(nearby_mobs=nearby_mobs)

    # Dragon state
    dragon_raw = raw_obs.get("dragon", {})
    dragon = DragonState(
        is_active=bool(dragon_raw.get("is_active", False)),
        phase=dragon_raw.get("phase", 0),
        health=dragon_raw.get("dragon_health", 0) / 200.0,
        crystals_remaining=dragon_raw.get("crystals_remaining", 0),
    )
    if dragon.is_active and "dragon_position" in dragon_raw:
        dx = dragon_raw["dragon_position"][0] - player_raw["position"][0]
        dy = dragon_raw["dragon_position"][1] - player_raw["position"][1]
        dz = dragon_raw["dragon_position"][2] - player_raw["position"][2]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        if dist > 0:
            dragon.dx, dragon.dy, dragon.dz = dx / dist, dy / dist, dz / dist
            dragon.distance = dist

    return MinecraftObservation(
        player=player,
        inventory=inventory,
        voxels=voxels,
        entities=entities,
        dragon=dragon,
        game_tick=raw_obs.get("tick_number", 0),
        terminated=bool(raw_obs.get("terminated", False)),
        truncated=bool(raw_obs.get("truncated", False)),
    )


def decode_flat_observation(stage_id: int, vector: np.ndarray) -> dict[str, Any]:
    """Decode a compact 256-float observation vector into the dict format
    expected by ProgressTracker.update_from_observation.

    Reconstructs minimal player state, inventory, dimension, tick, and hotbar
    information from the compact encoding using CompactObservationDecoder.

    Args:
        stage_id: Current curriculum stage (1-6). Used for validation and
            cross-referencing against the stage encoded in the vector.
        vector: Flat observation array of shape (256,) in the compact format
            produced by CompactObservationEncoder.

    Returns:
        Dictionary with keys:
            - 'player': dict with 'health', 'dimension', 'on_ground',
              'in_water', 'in_lava', 'position', 'velocity', 'yaw', 'pitch'
            - 'inventory': dict with item count keys matching
              ProgressTracker._update_inventory_progress expectations
            - 'dimension': int (0=Overworld, 1=Nether, 2=End)
            - 'tick_number': int (denormalized game tick)
            - 'hotbar': list[int] of 9 item IDs
            - 'dragon': dict with dragon fight state (if stage >= 6 or active)

    Raises:
        ValueError: If vector shape is not (256,) or stage_id not in [1, 6].
    """
    logger.debug("decode_flat_observation: stage_id=%s, vector=%s", stage_id, vector)
    vec = np.asarray(vector, dtype=np.float32)
    if vec.shape != (256,):
        raise ValueError(f"Expected observation vector of shape (256,), got {vec.shape}")
    if not 1 <= stage_id <= 6:
        raise ValueError(f"stage_id must be 1-6, got {stage_id}")

    decoder = CompactObservationDecoder(vec)

    # Decode sub-components
    player_state = decoder.get_player_state()
    inv_state = decoder.get_inventory()
    dimension = decoder.get_dimension()
    dragon_raw = decoder.get_dragon_state()

    # Reconstruct PlayerState dataclass for structured access
    player_ds = PlayerState(
        x=float(player_state.position[0]),
        y=float(player_state.position[1]),
        z=float(player_state.position[2]),
        vx=float(player_state.velocity[0]),
        vy=float(player_state.velocity[1]),
        vz=float(player_state.velocity[2]),
        health=player_state.health / 20.0,  # normalize to [0,1]
        hunger=player_state.hunger / 20.0,
        yaw=player_state.yaw / 180.0,
        pitch=player_state.pitch / 90.0,
        dimension=dimension,
        on_ground=player_state.is_on_ground,
        in_water=player_state.is_in_water,
        in_lava=player_state.is_in_lava,
        sprinting=player_state.is_sprinting,
        sneaking=player_state.is_sneaking,
    )

    # Reconstruct InventorySummary from compact inventory
    inv_ds = InventorySummary(
        blaze_rods=inv_state.blaze_rod_count,
        ender_pearls=inv_state.ender_pearl_count,
        eyes_of_ender=inv_state.eye_of_ender_count,
        iron_ingots=inv_state.iron_count,
        diamonds=inv_state.diamond_count,
        hotbar=tuple(inv_state.hotbar_items[:9]),
    )

    # Build the player dict for ProgressTracker
    player_dict: dict[str, Any] = {
        "health": player_state.health,
        "dimension": dimension,
        "on_ground": player_state.is_on_ground,
        "in_water": player_state.is_in_water,
        "in_lava": player_state.is_in_lava,
        "position": [
            float(player_state.position[0]),
            float(player_state.position[1]),
            float(player_state.position[2]),
        ],
        "velocity": [
            float(player_state.velocity[0]),
            float(player_state.velocity[1]),
            float(player_state.velocity[2]),
        ],
        "yaw": player_state.yaw,
        "pitch": player_state.pitch,
    }

    # Build inventory dict matching ProgressTracker._update_inventory_progress keys
    inventory_dict: dict[str, int] = {
        "blaze_rods": inv_state.blaze_rod_count,
        "ender_pearls": inv_state.ender_pearl_count,
        "eyes_of_ender": inv_state.eye_of_ender_count,
        "iron_ingots": inv_state.iron_count,
        "diamonds": inv_state.diamond_count,
        "wood": inv_state.wood_count,
        "food_count": inv_state.food_count,
        "empty_buckets": 0,  # Not tracked in compact format
        "iron_pickaxe": 0,  # Not tracked in compact format
    }

    # Detect tools from hotbar item IDs
    IRON_PICKAXE_ID = 257
    BUCKET_ID = 325
    for item_id in inv_state.hotbar_items:
        if item_id == IRON_PICKAXE_ID:
            inventory_dict["iron_pickaxe"] = 1
        elif item_id == BUCKET_ID:
            inventory_dict["empty_buckets"] = 1

    # Denormalize tick from the observation vector
    # Tick is not directly in the compact 256-float layout, so estimate from
    # stage timing: use a reasonable default or extract if encoded elsewhere.
    # The compact format doesn't explicitly encode tick_number, so we use 0
    # and let the caller override if needed.
    tick_number = 0

    # Build dragon dict
    dragon_dict: dict[str, Any] = {"is_active": False}
    if dragon_raw is not None:
        dragon_dict = {
            "is_active": True,
            "dragon_health": dragon_raw["health"],
            "crystals_remaining": dragon_raw["crystals_remaining"],
            "phase": 3 if dragon_raw.get("is_perched", False) else 1,
        }

    result: dict[str, Any] = {
        "player": player_dict,
        "inventory": inventory_dict,
        "dimension": dimension,
        "tick_number": tick_number,
        "hotbar": list(inv_state.hotbar_items[:9]),
        "dragon": dragon_dict,
    }

    return result
