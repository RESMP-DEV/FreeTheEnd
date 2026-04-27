"""Tests for Minecraft action space module."""

from __future__ import annotations

import numpy as np

from minecraft_sim.actions import (

import logging

logger = logging.getLogger(__name__)

    PITCH_DELTAS,
    YAW_DELTAS,
    ActionMask,
    ActionSpace,
    HierarchicalActionSpace,
    MinecraftAction,
    Movement,
    SpecialAction,
    action_to_keyboard_mouse,
    decode_action,
    encode_action,
    get_action_embedding_dims,
    get_total_embedding_dim,
)


class TestMovementEnum:
    """Tests for Movement enum."""

    def test_movement_values(self) -> None:
        """Test that movement enum has expected values."""
        assert Movement.NONE == 0
        assert Movement.FORWARD == 1
        assert Movement.BACK == 2
        assert Movement.LEFT == 3
        assert Movement.RIGHT == 4
        assert Movement.FORWARD_LEFT == 5
        assert Movement.FORWARD_RIGHT == 6
        assert Movement.BACK_LEFT == 7
        assert Movement.BACK_RIGHT == 8

    def test_movement_count(self) -> None:
        """Test that movement enum has 9 values."""
        assert len(Movement) == 9


class TestSpecialActionEnum:
    """Tests for SpecialAction enum."""

    def test_special_action_values(self) -> None:
        """Test special action enum values."""
        assert SpecialAction.NONE == 0
        assert SpecialAction.CRAFT == 1
        assert SpecialAction.DROP == 2
        assert SpecialAction.OPEN_INVENTORY == 3

    def test_special_action_count(self) -> None:
        """Test special action enum has 4 values."""
        assert len(SpecialAction) == 4


class TestLookDeltas:
    """Tests for discretized look delta arrays."""

    def test_yaw_deltas_shape(self) -> None:
        """Test yaw deltas array shape."""
        assert YAW_DELTAS.shape == (9,)
        assert YAW_DELTAS.dtype == np.float32

    def test_pitch_deltas_shape(self) -> None:
        """Test pitch deltas array shape."""
        assert PITCH_DELTAS.shape == (7,)
        assert PITCH_DELTAS.dtype == np.float32

    def test_yaw_deltas_symmetric(self) -> None:
        """Test yaw deltas are roughly symmetric around 0."""
        assert YAW_DELTAS[4] == 0.0  # Center is zero
        assert np.isclose(YAW_DELTAS[0], -YAW_DELTAS[-1])

    def test_pitch_deltas_symmetric(self) -> None:
        """Test pitch deltas are roughly symmetric around 0."""
        assert PITCH_DELTAS[3] == 0.0  # Center is zero
        assert np.isclose(PITCH_DELTAS[0], -PITCH_DELTAS[-1])


class TestMinecraftAction:
    """Tests for MinecraftAction dataclass."""

    def test_default_action(self) -> None:
        """Test default action values."""
        action = MinecraftAction()
        assert action.movement == Movement.NONE
        assert action.jump is False
        assert action.sprint is False
        assert action.sneak is False
        assert action.attack is False
        assert action.use_item is False
        assert action.yaw_delta == 0.0
        assert action.pitch_delta == 0.0
        assert action.hotbar_slot == 0
        assert action.special == SpecialAction.NONE

    def test_custom_action(self) -> None:
        """Test creating custom action."""
        action = MinecraftAction(
            movement=Movement.FORWARD,
            jump=True,
            sprint=True,
            attack=True,
            yaw_delta=45.0,
            hotbar_slot=5,
        )
        assert action.movement == Movement.FORWARD
        assert action.jump is True
        assert action.sprint is True
        assert action.attack is True
        assert action.yaw_delta == 45.0
        assert action.hotbar_slot == 5

    def test_to_dict(self) -> None:
        """Test action serialization to dict."""
        action = MinecraftAction(
            movement=Movement.FORWARD_RIGHT,
            jump=True,
            special=SpecialAction.CRAFT,
        )
        d = action.to_dict()
        assert d["movement"] == 6  # FORWARD_RIGHT
        assert d["jump"] is True
        assert d["special"] == 1  # CRAFT

    def test_from_dict(self) -> None:
        """Test action deserialization from dict."""
        d = {
            "movement": 2,
            "jump": False,
            "sprint": True,
            "sneak": False,
            "attack": True,
            "use_item": False,
            "yaw_delta": -15.0,
            "pitch_delta": 5.0,
            "hotbar_slot": 3,
            "special": 0,
        }
        action = MinecraftAction.from_dict(d)
        assert action.movement == Movement.BACK
        assert action.sprint is True
        assert action.attack is True
        assert action.yaw_delta == -15.0
        assert action.hotbar_slot == 3

    def test_roundtrip(self) -> None:
        """Test to_dict/from_dict roundtrip."""
        original = MinecraftAction(
            movement=Movement.LEFT,
            jump=True,
            sneak=True,
            yaw_delta=90.0,
            pitch_delta=-45.0,
            hotbar_slot=8,
            special=SpecialAction.DROP,
        )
        restored = MinecraftAction.from_dict(original.to_dict())
        assert restored.movement == original.movement
        assert restored.jump == original.jump
        assert restored.sneak == original.sneak
        assert restored.yaw_delta == original.yaw_delta
        assert restored.pitch_delta == original.pitch_delta
        assert restored.hotbar_slot == original.hotbar_slot
        assert restored.special == original.special


class TestActionSpace:
    """Tests for ActionSpace class."""

    def test_nvec_shape(self) -> None:
        """Test action space has correct dimensions."""
        space = ActionSpace()
        assert len(space.nvec) == 10
        assert space.nvec.dtype == np.int32

    def test_nvec_values(self) -> None:
        """Test action space dimension values."""
        space = ActionSpace()
        expected = [9, 2, 2, 2, 2, 2, 9, 7, 9, 4]
        np.testing.assert_array_equal(space.nvec, expected)

    def test_shape(self) -> None:
        """Test shape property."""
        space = ActionSpace()
        assert space.shape == (10,)

    def test_num_components(self) -> None:
        """Test num_components property."""
        space = ActionSpace()
        assert space.num_components == 10

    def test_total_combinations(self) -> None:
        """Test total combinations calculation."""
        space = ActionSpace()
        expected = 9 * 2 * 2 * 2 * 2 * 2 * 9 * 7 * 9 * 4
        assert space.total_combinations == expected
        # 9*2*2*2*2*2*9*7*9*4 = 653,184 (not 1,306,368)
        assert space.total_combinations == 653_184

    def test_sample(self) -> None:
        """Test action sampling."""
        space = ActionSpace()
        rng = np.random.default_rng(42)
        action = space.sample(rng)
        assert action.shape == (10,)
        assert action.dtype == np.int32
        assert space.contains(action)

    def test_sample_bounds(self) -> None:
        """Test sampled actions are within bounds."""
        space = ActionSpace()
        rng = np.random.default_rng(123)
        for _ in range(100):
            action = space.sample(rng)
            for i, (a, n) in enumerate(zip(action, space.nvec)):
                assert 0 <= a < n, f"Action[{i}]={a} out of bounds [0, {n})"

    def test_contains_valid(self) -> None:
        """Test contains returns True for valid actions."""
        space = ActionSpace()
        valid = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        assert space.contains(valid)
        valid = np.array([8, 1, 1, 1, 1, 1, 8, 6, 8, 3], dtype=np.int32)
        assert space.contains(valid)

    def test_contains_invalid(self) -> None:
        """Test contains returns False for invalid actions."""
        space = ActionSpace()
        invalid = np.array([9, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)  # Movement out of bounds
        assert not space.contains(invalid)
        invalid = np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)  # Jump out of bounds
        assert not space.contains(invalid)
        invalid = np.array([0, 0, 0, 0, 0], dtype=np.int32)  # Wrong length
        assert not space.contains(invalid)

    def test_decode(self) -> None:
        """Test action decoding via ActionSpace."""
        space = ActionSpace()
        action = np.array([1, 1, 0, 1, 0, 0, 4, 3, 2, 1], dtype=np.int32)
        decoded = space.decode(action)
        assert decoded.movement == Movement.FORWARD
        assert decoded.jump is True
        assert decoded.sneak is True
        assert decoded.yaw_delta == 0.0
        assert decoded.pitch_delta == 0.0
        assert decoded.hotbar_slot == 2
        assert decoded.special == SpecialAction.CRAFT

    def test_encode(self) -> None:
        """Test action encoding via ActionSpace."""
        space = ActionSpace()
        action = MinecraftAction(
            movement=Movement.FORWARD,
            jump=True,
            sneak=True,
            yaw_delta=0.0,
            pitch_delta=0.0,
            hotbar_slot=2,
            special=SpecialAction.CRAFT,
        )
        encoded = space.encode(action)
        expected = np.array([1, 1, 0, 1, 0, 0, 4, 3, 2, 1], dtype=np.int32)
        np.testing.assert_array_equal(encoded, expected)


class TestEncodeDecodeAction:
    """Tests for encode_action and decode_action functions."""

    def test_decode_all_zeros(self) -> None:
        """Test decoding all-zero action."""
        action = np.zeros(10, dtype=np.int32)
        decoded = decode_action(action)
        assert decoded.movement == Movement.NONE
        assert decoded.jump is False
        assert decoded.yaw_delta == YAW_DELTAS[0]
        assert decoded.pitch_delta == PITCH_DELTAS[0]

    def test_decode_all_max(self) -> None:
        """Test decoding max-value action."""
        action = np.array([8, 1, 1, 1, 1, 1, 8, 6, 8, 3], dtype=np.int32)
        decoded = decode_action(action)
        assert decoded.movement == Movement.BACK_RIGHT
        assert decoded.jump is True
        assert decoded.sprint is True
        assert decoded.sneak is True
        assert decoded.attack is True
        assert decoded.use_item is True
        assert decoded.yaw_delta == YAW_DELTAS[8]
        assert decoded.pitch_delta == PITCH_DELTAS[6]
        assert decoded.hotbar_slot == 8
        assert decoded.special == SpecialAction.OPEN_INVENTORY

    def test_encode_default(self) -> None:
        """Test encoding default action."""
        action = MinecraftAction()
        # Default yaw_delta=0 maps to index 4, pitch_delta=0 maps to index 3
        encoded = encode_action(action)
        assert encoded[0] == 0  # Movement.NONE
        assert encoded[6] == 4  # yaw_delta=0 is at index 4
        assert encoded[7] == 3  # pitch_delta=0 is at index 3

    def test_encode_finds_closest_yaw(self) -> None:
        """Test encoding finds closest yaw delta."""
        action = MinecraftAction(yaw_delta=10.0)  # Between 5 and 15
        encoded = encode_action(action)
        # Should map to index 5 (value 5.0) or index 6 (value 15.0)
        # 10.0 is equidistant, argmin returns first match
        assert encoded[6] in (5, 6)

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip preserves values."""
        original = MinecraftAction(
            movement=Movement.FORWARD_LEFT,
            jump=True,
            sprint=True,
            attack=True,
            yaw_delta=45.0,
            pitch_delta=-15.0,
            hotbar_slot=4,
            special=SpecialAction.DROP,
        )
        encoded = encode_action(original)
        decoded = decode_action(encoded)

        assert decoded.movement == original.movement
        assert decoded.jump == original.jump
        assert decoded.sprint == original.sprint
        assert decoded.attack == original.attack
        assert decoded.yaw_delta == original.yaw_delta
        assert decoded.pitch_delta == original.pitch_delta
        assert decoded.hotbar_slot == original.hotbar_slot
        assert decoded.special == original.special


class TestActionToKeyboardMouse:
    """Tests for action_to_keyboard_mouse function."""

    def test_forward_movement(self) -> None:
        """Test forward movement maps to W key."""
        action = MinecraftAction(movement=Movement.FORWARD)
        keys = action_to_keyboard_mouse(action)
        assert keys["key_w"] is True
        assert keys["key_s"] is False
        assert keys["key_a"] is False
        assert keys["key_d"] is False

    def test_diagonal_movement(self) -> None:
        """Test diagonal movement maps to two keys."""
        action = MinecraftAction(movement=Movement.FORWARD_LEFT)
        keys = action_to_keyboard_mouse(action)
        assert keys["key_w"] is True
        assert keys["key_a"] is True
        assert keys["key_s"] is False
        assert keys["key_d"] is False

    def test_jump_maps_to_space(self) -> None:
        """Test jump maps to space key."""
        action = MinecraftAction(jump=True)
        keys = action_to_keyboard_mouse(action)
        assert keys["key_space"] is True

    def test_sprint_maps_to_ctrl(self) -> None:
        """Test sprint maps to ctrl key."""
        action = MinecraftAction(sprint=True)
        keys = action_to_keyboard_mouse(action)
        assert keys["key_ctrl"] is True

    def test_sneak_maps_to_shift(self) -> None:
        """Test sneak maps to shift key."""
        action = MinecraftAction(sneak=True)
        keys = action_to_keyboard_mouse(action)
        assert keys["key_shift"] is True

    def test_attack_maps_to_left_mouse(self) -> None:
        """Test attack maps to left mouse button."""
        action = MinecraftAction(attack=True)
        keys = action_to_keyboard_mouse(action)
        assert keys["mouse_left"] is True

    def test_use_item_maps_to_right_mouse(self) -> None:
        """Test use_item maps to right mouse button."""
        action = MinecraftAction(use_item=True)
        keys = action_to_keyboard_mouse(action)
        assert keys["mouse_right"] is True

    def test_mouse_deltas(self) -> None:
        """Test mouse deltas are passed through."""
        action = MinecraftAction(yaw_delta=45.0, pitch_delta=-30.0)
        keys = action_to_keyboard_mouse(action)
        assert keys["mouse_dx"] == 45.0
        assert keys["mouse_dy"] == -30.0

    def test_hotbar_slot(self) -> None:
        """Test hotbar slot is passed through."""
        action = MinecraftAction(hotbar_slot=7)
        keys = action_to_keyboard_mouse(action)
        assert keys["hotbar_slot"] == 7

    def test_special_actions(self) -> None:
        """Test special action flags."""
        action = MinecraftAction(special=SpecialAction.CRAFT)
        keys = action_to_keyboard_mouse(action)
        assert keys["special_craft"] is True
        assert keys["special_drop"] is False
        assert keys["special_inventory"] is False


class TestHierarchicalActionSpace:
    """Tests for HierarchicalActionSpace class."""

    def test_nvec_combined(self) -> None:
        """Test combined nvec matches flat space."""
        hier = HierarchicalActionSpace()
        flat = ActionSpace()
        np.testing.assert_array_equal(hier.nvec, flat.nvec)

    def test_sample(self) -> None:
        """Test hierarchical sampling."""
        hier = HierarchicalActionSpace()
        rng = np.random.default_rng(42)
        action = hier.sample(rng)
        assert len(action) == 10

    def test_decode(self) -> None:
        """Test hierarchical decode matches flat decode."""
        hier = HierarchicalActionSpace()
        flat = ActionSpace()
        action = np.array([2, 1, 0, 1, 1, 0, 3, 4, 5, 2], dtype=np.int32)

        hier_decoded = hier.decode(action)
        flat_decoded = flat.decode(action)

        assert hier_decoded.movement == flat_decoded.movement
        assert hier_decoded.jump == flat_decoded.jump
        assert hier_decoded.sprint == flat_decoded.sprint
        assert hier_decoded.special == flat_decoded.special


class TestActionMask:
    """Tests for ActionMask class."""

    def test_create_all_enabled(self) -> None:
        """Test create method enables all actions."""
        nvec = np.array([9, 2, 2, 2, 2, 2, 9, 7, 9, 4], dtype=np.int32)
        mask = ActionMask.create(nvec)
        assert all(mask.mask)
        for i, n in enumerate(nvec):
            assert len(mask.component_masks[i]) == n
            assert all(mask.component_masks[i])

    def test_disable_sprint_while_sneaking(self) -> None:
        """Test sprint is disabled when sneaking."""
        nvec = np.array([9, 2, 2, 2, 2, 2, 9, 7, 9, 4], dtype=np.int32)
        mask = ActionMask.create(nvec)

        # Action with sneak=1
        action = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        mask.disable_sprint_while_sneaking(action)

        # Sprint component (index 2), value 1 should be disabled
        assert mask.component_masks[2][1] == False  # noqa: E712
        assert mask.component_masks[2][0] == True  # sprint=no still allowed  # noqa: E712

    def test_disable_attack_while_inventory(self) -> None:
        """Test attack is disabled when inventory open."""
        nvec = np.array([9, 2, 2, 2, 2, 2, 9, 7, 9, 4], dtype=np.int32)
        mask = ActionMask.create(nvec)

        # Action with special=OPEN_INVENTORY (3)
        action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 3], dtype=np.int32)
        mask.disable_attack_while_inventory(action)

        # Attack component (index 4), value 1 should be disabled
        assert mask.component_masks[4][1] == False  # noqa: E712
        assert mask.component_masks[4][0] == True  # attack=no still allowed  # noqa: E712

    def test_apply_pitch_limits_looking_down(self) -> None:
        """Test pitch limits near -90 degrees."""
        nvec = np.array([9, 2, 2, 2, 2, 2, 9, 7, 9, 4], dtype=np.int32)
        mask = ActionMask.create(nvec)

        mask.apply_pitch_limits(-80.0)  # Near looking straight down

        # Negative pitch deltas should be disabled
        for i, delta in enumerate(PITCH_DELTAS):
            if delta < 0:
                assert mask.component_masks[7][i] == False  # noqa: E712
            else:
                assert mask.component_masks[7][i] == True  # noqa: E712

    def test_apply_pitch_limits_looking_up(self) -> None:
        """Test pitch limits near +90 degrees."""
        nvec = np.array([9, 2, 2, 2, 2, 2, 9, 7, 9, 4], dtype=np.int32)
        mask = ActionMask.create(nvec)

        mask.apply_pitch_limits(80.0)  # Near looking straight up

        # Positive pitch deltas should be disabled
        for i, delta in enumerate(PITCH_DELTAS):
            if delta > 0:
                assert mask.component_masks[7][i] == False  # noqa: E712
            else:
                assert mask.component_masks[7][i] == True  # noqa: E712


class TestEmbeddingDimensions:
    """Tests for action embedding helpers."""

    def test_embedding_dims_all_present(self) -> None:
        """Test all action components have embeddings."""
        dims = get_action_embedding_dims()
        expected_keys = {
            "movement",
            "jump",
            "sprint",
            "sneak",
            "attack",
            "use_item",
            "yaw_delta",
            "pitch_delta",
            "hotbar_slot",
            "special",
        }
        assert set(dims.keys()) == expected_keys

    def test_embedding_dims_positive(self) -> None:
        """Test all embedding dimensions are positive."""
        dims = get_action_embedding_dims()
        for key, dim in dims.items():
            assert dim > 0, f"{key} has non-positive dimension {dim}"

    def test_total_embedding_dim(self) -> None:
        """Test total embedding dimension calculation."""
        dims = get_action_embedding_dims()
        expected = sum(dims.values())
        assert get_total_embedding_dim() == expected
        assert get_total_embedding_dim() == 44
