"""
Multi-discrete action space for Minecraft RL agents.

Action space components:
1. Movement: 9 options (none, forward, back, left, right, forward-left, forward-right, back-left, back-right)
2. Jump: 2 options (no, yes)
3. Sprint: 2 options (no, yes)
4. Sneak: 2 options (no, yes)
5. Attack: 2 options (no, yes)
6. Use item: 2 options (no, yes)
7. Look yaw delta: 9 options (-90, -45, -15, -5, 0, 5, 15, 45, 90)
8. Look pitch delta: 7 options (-45, -15, -5, 0, 5, 15, 45)
9. Hotbar slot: 9 options (0-8)
10. Special actions: 4 options (none, craft, drop, open_inventory)

Total combinations: 9 * 2 * 2 * 2 * 2 * 2 * 9 * 7 * 9 * 4 = 1,306,368
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

import logging

logger = logging.getLogger(__name__)


class Movement(IntEnum):
    """Movement direction options."""

    NONE = 0
    FORWARD = 1
    BACK = 2
    LEFT = 3
    RIGHT = 4
    FORWARD_LEFT = 5
    FORWARD_RIGHT = 6
    BACK_LEFT = 7
    BACK_RIGHT = 8


class SpecialAction(IntEnum):
    """Special action options."""

    NONE = 0
    CRAFT = 1
    DROP = 2
    OPEN_INVENTORY = 3


# Discretized look deltas
YAW_DELTAS = np.array([-90.0, -45.0, -15.0, -5.0, 0.0, 5.0, 15.0, 45.0, 90.0], dtype=np.float32)
PITCH_DELTAS = np.array([-45.0, -15.0, -5.0, 0.0, 5.0, 15.0, 45.0], dtype=np.float32)


@dataclass(slots=True)
class MinecraftAction:
    """Decoded Minecraft action with all components."""

    movement: Movement = Movement.NONE
    jump: bool = False
    sprint: bool = False
    sneak: bool = False
    attack: bool = False
    use_item: bool = False
    yaw_delta: float = 0.0
    pitch_delta: float = 0.0
    hotbar_slot: int = 0
    special: SpecialAction = SpecialAction.NONE

    def to_dict(self) -> dict[str, float | int | bool]:
        """Convert to dictionary for serialization."""
        logger.debug("MinecraftAction.to_dict called")
        return {
            "movement": int(self.movement),
            "jump": self.jump,
            "sprint": self.sprint,
            "sneak": self.sneak,
            "attack": self.attack,
            "use_item": self.use_item,
            "yaw_delta": self.yaw_delta,
            "pitch_delta": self.pitch_delta,
            "hotbar_slot": self.hotbar_slot,
            "special": int(self.special),
        }

    @classmethod
    def from_dict(cls, d: dict[str, float | int | bool]) -> MinecraftAction:
        """Create from dictionary."""
        logger.debug("MinecraftAction.from_dict: d=%s", d)
        return cls(
            movement=Movement(int(d["movement"])),
            jump=bool(d["jump"]),
            sprint=bool(d["sprint"]),
            sneak=bool(d["sneak"]),
            attack=bool(d["attack"]),
            use_item=bool(d["use_item"]),
            yaw_delta=float(d["yaw_delta"]),
            pitch_delta=float(d["pitch_delta"]),
            hotbar_slot=int(d["hotbar_slot"]),
            special=SpecialAction(int(d["special"])),
        )


@dataclass
class ActionSpace:
    """
    Multi-discrete action space for Minecraft.

    Uses factored representation for efficiency - each component is sampled
    independently rather than enumerating all 1.3M combinations.

    Attributes:
        nvec: Array of dimensions for each action component.
        shape: Shape of the multi-discrete space (10 components).
    """

    # Dimensions for each action component
    nvec: NDArray[np.int32] = field(
        default_factory=lambda: np.array([9, 2, 2, 2, 2, 2, 9, 7, 9, 4], dtype=np.int32)
    )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the action space."""
        logger.debug("ActionSpace.shape called")
        return (len(self.nvec),)

    @property
    def num_components(self) -> int:
        """Number of action components."""
        logger.debug("ActionSpace.num_components called")
        return len(self.nvec)

    @property
    def total_combinations(self) -> int:
        """Total number of possible action combinations."""
        logger.debug("ActionSpace.total_combinations called")
        return int(np.prod(self.nvec))

    def sample(self, rng: np.random.Generator | None = None) -> NDArray[np.int32]:
        """
        Sample a random action.

        Args:
            rng: NumPy random generator. Uses default if None.

        Returns:
            Array of action indices, one per component.
        """
        logger.debug("ActionSpace.sample: rng=%s", rng)
        if rng is None:
            rng = np.random.default_rng()
        return np.array([rng.integers(0, n) for n in self.nvec], dtype=np.int32)

    def contains(self, action: NDArray[np.int32]) -> bool:
        """Check if action is valid."""
        logger.debug("ActionSpace.contains: action=%s", action)
        if len(action) != len(self.nvec):
            return False
        return all(0 <= a < n for a, n in zip(action, self.nvec, strict=True))

    def decode(self, action: NDArray[np.int32]) -> MinecraftAction:
        """
        Decode multi-discrete action array to MinecraftAction.

        Args:
            action: Array of 10 integers representing action indices.

        Returns:
            Decoded MinecraftAction object.
        """
        logger.debug("ActionSpace.decode: action=%s", action)
        return decode_action(action)

    def encode(self, action: MinecraftAction) -> NDArray[np.int32]:
        """
        Encode MinecraftAction to multi-discrete array.

        Args:
            action: MinecraftAction to encode.

        Returns:
            Array of 10 integers representing action indices.
        """
        logger.debug("ActionSpace.encode: action=%s", action)
        return encode_action(action)


def decode_action(action: NDArray[np.int32]) -> MinecraftAction:
    """
    Decode multi-discrete action array to MinecraftAction.

    Args:
        action: Array of 10 integers [movement, jump, sprint, sneak, attack,
                use_item, yaw_idx, pitch_idx, hotbar, special].

    Returns:
        Decoded MinecraftAction with actual values.
    """
    logger.debug("decode_action: action=%s", action)
    return MinecraftAction(
        movement=Movement(action[0]),
        jump=bool(action[1]),
        sprint=bool(action[2]),
        sneak=bool(action[3]),
        attack=bool(action[4]),
        use_item=bool(action[5]),
        yaw_delta=float(YAW_DELTAS[action[6]]),
        pitch_delta=float(PITCH_DELTAS[action[7]]),
        hotbar_slot=int(action[8]),
        special=SpecialAction(action[9]),
    )


def encode_action(action: MinecraftAction) -> NDArray[np.int32]:
    """
    Encode MinecraftAction to multi-discrete array.

    Args:
        action: MinecraftAction to encode.

    Returns:
        Array of 10 integers representing action indices.
    """
    # Find closest yaw/pitch indices
    logger.debug("encode_action: action=%s", action)
    yaw_idx = int(np.argmin(np.abs(YAW_DELTAS - action.yaw_delta)))
    pitch_idx = int(np.argmin(np.abs(PITCH_DELTAS - action.pitch_delta)))

    return np.array(
        [
            int(action.movement),
            int(action.jump),
            int(action.sprint),
            int(action.sneak),
            int(action.attack),
            int(action.use_item),
            yaw_idx,
            pitch_idx,
            action.hotbar_slot,
            int(action.special),
        ],
        dtype=np.int32,
    )


def action_to_keyboard_mouse(action: MinecraftAction) -> dict[str, bool | float]:
    """
    Convert MinecraftAction to keyboard/mouse input representation.

    Useful for sending to actual Minecraft client or simulation.

    Args:
        action: Decoded MinecraftAction.

    Returns:
        Dictionary with key/button states and mouse deltas.
    """
    logger.debug("action_to_keyboard_mouse: action=%s", action)
    keys: dict[str, bool | float] = {
        # Movement keys
        "key_w": action.movement
        in (Movement.FORWARD, Movement.FORWARD_LEFT, Movement.FORWARD_RIGHT),
        "key_s": action.movement in (Movement.BACK, Movement.BACK_LEFT, Movement.BACK_RIGHT),
        "key_a": action.movement in (Movement.LEFT, Movement.FORWARD_LEFT, Movement.BACK_LEFT),
        "key_d": action.movement in (Movement.RIGHT, Movement.FORWARD_RIGHT, Movement.BACK_RIGHT),
        # Action keys
        "key_space": action.jump,
        "key_ctrl": action.sprint,
        "key_shift": action.sneak,
        # Mouse buttons
        "mouse_left": action.attack,
        "mouse_right": action.use_item,
        # Mouse deltas (degrees)
        "mouse_dx": action.yaw_delta,
        "mouse_dy": action.pitch_delta,
        # Hotbar (1-9 keys, 0-indexed internally)
        "hotbar_slot": action.hotbar_slot,
        # Special actions (would need separate handling in game)
        "special_craft": action.special == SpecialAction.CRAFT,
        "special_drop": action.special == SpecialAction.DROP,
        "special_inventory": action.special == SpecialAction.OPEN_INVENTORY,
    }
    return keys


# Hierarchical action space for more efficient exploration
@dataclass
class HierarchicalActionSpace:
    """
    Hierarchical action space grouping related actions.

    Groups:
    - Locomotion: movement + jump + sprint + sneak (9 * 2 * 2 * 2 = 72)
    - Combat: attack + use_item (2 * 2 = 4)
    - Camera: yaw + pitch (9 * 7 = 63)
    - Inventory: hotbar + special (9 * 4 = 36)

    Total: 72 * 4 * 63 * 36 = 653,184 (half of flat space)
    """

    locomotion_nvec: NDArray[np.int32] = field(
        default_factory=lambda: np.array([9, 2, 2, 2], dtype=np.int32)
    )
    combat_nvec: NDArray[np.int32] = field(default_factory=lambda: np.array([2, 2], dtype=np.int32))
    camera_nvec: NDArray[np.int32] = field(default_factory=lambda: np.array([9, 7], dtype=np.int32))
    inventory_nvec: NDArray[np.int32] = field(
        default_factory=lambda: np.array([9, 4], dtype=np.int32)
    )

    @property
    def nvec(self) -> NDArray[np.int32]:
        """Combined nvec for all groups."""
        logger.debug("HierarchicalActionSpace.nvec called")
        return np.concatenate(
            [
                self.locomotion_nvec,
                self.combat_nvec,
                self.camera_nvec,
                self.inventory_nvec,
            ]
        )

    def sample(self, rng: np.random.Generator | None = None) -> NDArray[np.int32]:
        """Sample action from hierarchical space."""
        logger.debug("HierarchicalActionSpace.sample: rng=%s", rng)
        if rng is None:
            rng = np.random.default_rng()
        return np.array([rng.integers(0, n) for n in self.nvec], dtype=np.int32)

    def decode(self, action: NDArray[np.int32]) -> MinecraftAction:
        """Decode hierarchical action to MinecraftAction."""
        # Reorder to match flat action space format
        logger.debug("HierarchicalActionSpace.decode: action=%s", action)
        flat = np.array(
            [
                action[0],  # movement
                action[1],  # jump
                action[2],  # sprint
                action[3],  # sneak
                action[4],  # attack
                action[5],  # use_item
                action[6],  # yaw
                action[7],  # pitch
                action[8],  # hotbar
                action[9],  # special
            ],
            dtype=np.int32,
        )
        return decode_action(flat)


# Action masking support for invalid action combinations
@dataclass
class ActionMask:
    """
    Mask for invalid action combinations.

    Useful for:
    - Preventing sprint while sneaking
    - Preventing attack while inventory open
    - Limiting look angles near pitch limits
    """

    mask: NDArray[np.bool_] = field(default_factory=lambda: np.ones(10, dtype=np.bool_))
    component_masks: list[NDArray[np.bool_]] = field(default_factory=list)

    @classmethod
    def create(cls, nvec: NDArray[np.int32]) -> ActionMask:
        """Create action mask with all actions enabled."""
        logger.info("ActionMask.create: nvec=%s", nvec)
        component_masks = [np.ones(n, dtype=np.bool_) for n in nvec]
        return cls(mask=np.ones(len(nvec), dtype=np.bool_), component_masks=component_masks)

    def disable_sprint_while_sneaking(self, action: NDArray[np.int32]) -> None:
        """Disable sprint if sneak is active."""
        logger.debug("ActionMask.disable_sprint_while_sneaking: action=%s", action)
        if action[3]:  # sneak active
            self.component_masks[2][1] = False  # disable sprint=yes

    def disable_attack_while_inventory(self, action: NDArray[np.int32]) -> None:
        """Disable attack if inventory is open."""
        logger.debug("ActionMask.disable_attack_while_inventory: action=%s", action)
        if action[9] == SpecialAction.OPEN_INVENTORY:
            self.component_masks[4][1] = False  # disable attack=yes

    def apply_pitch_limits(self, current_pitch: float) -> None:
        """Limit pitch deltas near -90/+90."""
        logger.debug("ActionMask.apply_pitch_limits: current_pitch=%s", current_pitch)
        if current_pitch <= -75:
            # Near looking straight down, disable further down
            for i, delta in enumerate(PITCH_DELTAS):
                if delta < 0:
                    self.component_masks[7][i] = False
        elif current_pitch >= 75:
            # Near looking straight up, disable further up
            for i, delta in enumerate(PITCH_DELTAS):
                if delta > 0:
                    self.component_masks[7][i] = False


# Action embedding for neural network input
def get_action_embedding_dims() -> dict[str, int]:
    """
    Get embedding dimensions for each action component.

    Smaller embeddings for binary actions, larger for categorical.
    """
    logger.debug("get_action_embedding_dims called")
    return {
        "movement": 8,  # 9 categories -> 8-dim embedding
        "jump": 2,  # binary -> 2-dim
        "sprint": 2,
        "sneak": 2,
        "attack": 2,
        "use_item": 2,
        "yaw_delta": 8,  # 9 categories -> 8-dim
        "pitch_delta": 6,  # 7 categories -> 6-dim
        "hotbar_slot": 8,  # 9 categories -> 8-dim
        "special": 4,  # 4 categories -> 4-dim
    }


def get_total_embedding_dim() -> int:
    """Total dimension of action embedding."""
    logger.debug("get_total_embedding_dim called")
    return sum(get_action_embedding_dims().values())  # = 44
