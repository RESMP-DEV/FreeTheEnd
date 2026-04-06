"""
Action encoding utilities.

Discrete action space (32 actions):
0:      noop
1-4:    cardinal movement
5-6:    diagonal forward
7-8:    jump variants
9-10:   attack variants
11:     sprint toggle
12-19:  look directions
20:     use item
21:     drop item
22-30:  hotbar slots
31:     quick craft
"""

from enum import IntEnum

import numpy as np


class Action(IntEnum):
    NOOP = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    FORWARD_LEFT = 5
    FORWARD_RIGHT = 6
    JUMP = 7
    JUMP_FORWARD = 8
    ATTACK = 9
    ATTACK_FORWARD = 10
    SPRINT_TOGGLE = 11
    LOOK_LEFT = 12
    LOOK_RIGHT = 13
    LOOK_UP = 14
    LOOK_DOWN = 15
    LOOK_LEFT_FAST = 16
    LOOK_RIGHT_FAST = 17
    LOOK_UP_FAST = 18
    LOOK_DOWN_FAST = 19
    USE_ITEM = 20
    DROP_ITEM = 21
    HOTBAR_1 = 22
    HOTBAR_2 = 23
    HOTBAR_3 = 24
    HOTBAR_4 = 25
    HOTBAR_5 = 26
    HOTBAR_6 = 27
    HOTBAR_7 = 28
    HOTBAR_8 = 29
    HOTBAR_9 = 30
    QUICK_CRAFT = 31


NUM_ACTIONS = 32

# Action groups for analysis
MOVEMENT_ACTIONS = frozenset(
    {
        Action.FORWARD,
        Action.BACKWARD,
        Action.LEFT,
        Action.RIGHT,
        Action.FORWARD_LEFT,
        Action.FORWARD_RIGHT,
    }
)

JUMP_ACTIONS = frozenset({Action.JUMP, Action.JUMP_FORWARD})

COMBAT_ACTIONS = frozenset({Action.ATTACK, Action.ATTACK_FORWARD})

LOOK_ACTIONS = frozenset(
    {
        Action.LOOK_LEFT,
        Action.LOOK_RIGHT,
        Action.LOOK_UP,
        Action.LOOK_DOWN,
        Action.LOOK_LEFT_FAST,
        Action.LOOK_RIGHT_FAST,
        Action.LOOK_UP_FAST,
        Action.LOOK_DOWN_FAST,
    }
)

HOTBAR_ACTIONS = frozenset(
    {
        Action.HOTBAR_1,
        Action.HOTBAR_2,
        Action.HOTBAR_3,
        Action.HOTBAR_4,
        Action.HOTBAR_5,
        Action.HOTBAR_6,
        Action.HOTBAR_7,
        Action.HOTBAR_8,
        Action.HOTBAR_9,
    }
)


def get_valid_actions(stage: int, state: dict) -> set[int]:
    """
    Get set of valid actions given current state.

    Useful for action masking in training.

    Args:
        stage: Current game stage (unused for now, reserved for stage-specific masks).
        state: Game state dict with keys like 'held_item', 'can_craft'.

    Returns:
        Set of valid action indices.
    """
    valid = set(range(NUM_ACTIONS))

    # Can't use item if nothing in hand
    if state.get("held_item", 0) == 0:
        valid.discard(Action.USE_ITEM)

    # Can't drop if nothing to drop
    if state.get("held_item", 0) == 0:
        valid.discard(Action.DROP_ITEM)

    # Can't quick craft if no materials
    if not state.get("can_craft", False):
        valid.discard(Action.QUICK_CRAFT)

    return valid


def get_action_mask(stage: int, state: dict) -> np.ndarray:
    """
    Get action mask as boolean numpy array.

    Args:
        stage: Current game stage.
        state: Game state dict.

    Returns:
        Boolean array of shape (32,) where True means action is valid.
    """
    valid = get_valid_actions(stage, state)
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for a in valid:
        mask[a] = True
    return mask


def action_to_string(action: int) -> str:
    """Convert action ID to human-readable string."""
    try:
        return Action(action).name.lower().replace("_", " ")
    except ValueError:
        return f"unknown({action})"


def analyze_action_distribution(actions: np.ndarray) -> dict:
    """
    Analyze distribution of actions taken.

    Args:
        actions: 1D array of action indices.

    Returns:
        Dict with 'distribution' (action -> count), 'groups' (group -> count),
        and 'percentages' (group -> pct).
    """
    total = len(actions)
    if total == 0:
        return {"distribution": {}, "groups": {}, "percentages": {}}

    unique, counts = np.unique(actions, return_counts=True)
    dist = {int(a): int(c) for a, c in zip(unique, counts, strict=True)}

    # Group analysis
    groups = {
        "movement": sum(dist.get(int(a), 0) for a in MOVEMENT_ACTIONS),
        "jump": sum(dist.get(int(a), 0) for a in JUMP_ACTIONS),
        "combat": sum(dist.get(int(a), 0) for a in COMBAT_ACTIONS),
        "look": sum(dist.get(int(a), 0) for a in LOOK_ACTIONS),
        "hotbar": sum(dist.get(int(a), 0) for a in HOTBAR_ACTIONS),
        "noop": dist.get(0, 0),
        "other": (
            dist.get(int(Action.SPRINT_TOGGLE), 0)
            + dist.get(int(Action.USE_ITEM), 0)
            + dist.get(int(Action.DROP_ITEM), 0)
            + dist.get(int(Action.QUICK_CRAFT), 0)
        ),
    }

    # Convert to percentages
    pcts = {k: v / total * 100 for k, v in groups.items()}

    return {
        "distribution": dist,
        "groups": groups,
        "percentages": pcts,
    }


def one_hot_encode(action: int) -> np.ndarray:
    """
    One-hot encode a discrete action.

    Args:
        action: Action index (0-31).

    Returns:
        One-hot array of shape (32,).
    """
    encoded = np.zeros(NUM_ACTIONS, dtype=np.float32)
    encoded[action] = 1.0
    return encoded


def batch_one_hot_encode(actions: np.ndarray) -> np.ndarray:
    """
    One-hot encode a batch of actions.

    Args:
        actions: Array of action indices, shape (batch_size,).

    Returns:
        One-hot array of shape (batch_size, 32).
    """
    batch_size = len(actions)
    encoded = np.zeros((batch_size, NUM_ACTIONS), dtype=np.float32)
    encoded[np.arange(batch_size), actions] = 1.0
    return encoded


# Action to keyboard/mouse mapping for simulation
ACTION_TO_KEYS: dict[Action, dict[str, bool | float]] = {
    Action.NOOP: {},
    Action.FORWARD: {"key_w": True},
    Action.BACKWARD: {"key_s": True},
    Action.LEFT: {"key_a": True},
    Action.RIGHT: {"key_d": True},
    Action.FORWARD_LEFT: {"key_w": True, "key_a": True},
    Action.FORWARD_RIGHT: {"key_w": True, "key_d": True},
    Action.JUMP: {"key_space": True},
    Action.JUMP_FORWARD: {"key_space": True, "key_w": True},
    Action.ATTACK: {"mouse_left": True},
    Action.ATTACK_FORWARD: {"mouse_left": True, "key_w": True},
    Action.SPRINT_TOGGLE: {"key_ctrl": True},
    Action.LOOK_LEFT: {"mouse_dx": -15.0},
    Action.LOOK_RIGHT: {"mouse_dx": 15.0},
    Action.LOOK_UP: {"mouse_dy": -15.0},
    Action.LOOK_DOWN: {"mouse_dy": 15.0},
    Action.LOOK_LEFT_FAST: {"mouse_dx": -45.0},
    Action.LOOK_RIGHT_FAST: {"mouse_dx": 45.0},
    Action.LOOK_UP_FAST: {"mouse_dy": -45.0},
    Action.LOOK_DOWN_FAST: {"mouse_dy": 45.0},
    Action.USE_ITEM: {"mouse_right": True},
    Action.DROP_ITEM: {"key_q": True},
    Action.HOTBAR_1: {"hotbar_slot": 0},
    Action.HOTBAR_2: {"hotbar_slot": 1},
    Action.HOTBAR_3: {"hotbar_slot": 2},
    Action.HOTBAR_4: {"hotbar_slot": 3},
    Action.HOTBAR_5: {"hotbar_slot": 4},
    Action.HOTBAR_6: {"hotbar_slot": 5},
    Action.HOTBAR_7: {"hotbar_slot": 6},
    Action.HOTBAR_8: {"hotbar_slot": 7},
    Action.HOTBAR_9: {"hotbar_slot": 8},
    Action.QUICK_CRAFT: {"key_e": True, "special_craft": True},
}


def action_to_keyboard_mouse(action: int) -> dict[str, bool | float]:
    """
    Convert action ID to keyboard/mouse input representation.

    Args:
        action: Action index (0-31).

    Returns:
        Dict with key/button states and mouse deltas.
    """
    try:
        return ACTION_TO_KEYS[Action(action)].copy()
    except (ValueError, KeyError):
        return {}
