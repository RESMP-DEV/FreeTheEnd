// action_decoder.cpp - Implementation of multi-discrete action space decoder
// Converts factored RL actions to simulator API format

#include "action_decoder.h"
#include "../include/mc189/simulator_api.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace mc189 {

// Precomputed multipliers for flat encoding
// kMultipliers[i] = product of kNvec[i+1] * kNvec[i+2] * ... * kNvec[9]
// Example: kMultipliers[0] = 2*2*2*2*2*9*7*9*4 = 181440
//          kMultipliers[9] = 1
constexpr std::array<int64_t, kNumDimensions> ActionDecoder::kMultipliers;

int64_t ActionDecoder::encode_flat(const MultiDiscreteAction& action) noexcept {
    // Convert to flat index: sum(action[i] * multiplier[i])
    int64_t flat = 0;
    flat += action.movement    * kMultipliers[0];
    flat += action.jump        * kMultipliers[1];
    flat += action.sprint      * kMultipliers[2];
    flat += action.sneak       * kMultipliers[3];
    flat += action.attack      * kMultipliers[4];
    flat += action.use_item    * kMultipliers[5];
    flat += action.look_yaw    * kMultipliers[6];
    flat += action.look_pitch  * kMultipliers[7];
    flat += action.hotbar_slot * kMultipliers[8];
    flat += action.special     * kMultipliers[9];
    return flat;
}

MultiDiscreteAction ActionDecoder::decode_flat(int64_t flat_idx) noexcept {
    MultiDiscreteAction action;

    // Extract each dimension by division and modulo
    action.movement    = static_cast<int8_t>(flat_idx / kMultipliers[0]);
    flat_idx %= kMultipliers[0];

    action.jump        = static_cast<int8_t>(flat_idx / kMultipliers[1]);
    flat_idx %= kMultipliers[1];

    action.sprint      = static_cast<int8_t>(flat_idx / kMultipliers[2]);
    flat_idx %= kMultipliers[2];

    action.sneak       = static_cast<int8_t>(flat_idx / kMultipliers[3]);
    flat_idx %= kMultipliers[3];

    action.attack      = static_cast<int8_t>(flat_idx / kMultipliers[4]);
    flat_idx %= kMultipliers[4];

    action.use_item    = static_cast<int8_t>(flat_idx / kMultipliers[5]);
    flat_idx %= kMultipliers[5];

    action.look_yaw    = static_cast<int8_t>(flat_idx / kMultipliers[6]);
    flat_idx %= kMultipliers[6];

    action.look_pitch  = static_cast<int8_t>(flat_idx / kMultipliers[7]);
    flat_idx %= kMultipliers[7];

    action.hotbar_slot = static_cast<int8_t>(flat_idx / kMultipliers[8]);
    flat_idx %= kMultipliers[8];

    action.special     = static_cast<int8_t>(flat_idx);  // kMultipliers[9] = 1

    return action;
}

void ActionDecoder::encode_flat_batch(const MultiDiscreteAction* actions,
                                       int64_t* flat_indices,
                                       int batch_size) noexcept {
    // Process 4 at a time when batch_size % 4 == 0 for cache friendliness
    // (actual SIMD would require platform-specific intrinsics)
    int i = 0;

    // Unrolled loop for better ILP
    for (; i + 3 < batch_size; i += 4) {
        flat_indices[i + 0] = encode_flat(actions[i + 0]);
        flat_indices[i + 1] = encode_flat(actions[i + 1]);
        flat_indices[i + 2] = encode_flat(actions[i + 2]);
        flat_indices[i + 3] = encode_flat(actions[i + 3]);
    }

    // Remainder
    for (; i < batch_size; ++i) {
        flat_indices[i] = encode_flat(actions[i]);
    }
}

void ActionDecoder::decode_flat_batch(const int64_t* flat_indices,
                                       MultiDiscreteAction* actions,
                                       int batch_size) noexcept {
    int i = 0;

    // Unrolled loop
    for (; i + 3 < batch_size; i += 4) {
        actions[i + 0] = decode_flat(flat_indices[i + 0]);
        actions[i + 1] = decode_flat(flat_indices[i + 1]);
        actions[i + 2] = decode_flat(flat_indices[i + 2]);
        actions[i + 3] = decode_flat(flat_indices[i + 3]);
    }

    // Remainder
    for (; i < batch_size; ++i) {
        actions[i] = decode_flat(flat_indices[i]);
    }
}

void ActionDecoder::to_simulator_action(const MultiDiscreteAction& src,
                                         mc189_action_t* dst) noexcept {
    std::memset(dst, 0, sizeof(mc189_action_t));

    // Determine primary action type (priority: attack > use > movement actions)
    if (src.attack) {
        dst->action = MC189_ACTION_ATTACK;
    } else if (src.use_item) {
        dst->action = MC189_ACTION_USE;
    } else if (src.special == static_cast<int8_t>(SpecialAction::kCraft)) {
        dst->action = MC189_ACTION_CRAFT;
    } else if (src.special == static_cast<int8_t>(SpecialAction::kDrop)) {
        dst->action = MC189_ACTION_DROP;
    } else if (src.special == static_cast<int8_t>(SpecialAction::kOpenInventory)) {
        dst->action = MC189_ACTION_INVENTORY;
    } else {
        // Movement-based action
        switch (src.movement) {
        case 1:  // Forward
        case 5:  // Forward-left
        case 6:  // Forward-right
            dst->action = MC189_ACTION_FORWARD;
            break;
        case 2:  // Back
        case 7:  // Back-left
        case 8:  // Back-right
            dst->action = MC189_ACTION_BACKWARD;
            break;
        case 3:  // Left
            dst->action = MC189_ACTION_LEFT;
            break;
        case 4:  // Right
            dst->action = MC189_ACTION_RIGHT;
            break;
        default:
            dst->action = MC189_ACTION_NONE;
            break;
        }
    }

    // Hotbar selection (if changed, this takes priority)
    // The simulator handles hotbar selection via specific action types
    if (src.hotbar_slot >= 0 && src.hotbar_slot <= 8) {
        // Note: In a real impl, you'd track previous slot to detect changes
        // For now, we encode slot in a reserved field or as part of combined action
    }

    // Look deltas (convert degrees to radians for simulator API)
    constexpr float kDegToRad = 3.14159265358979f / 180.0f;
    dst->look_delta_yaw = src.yaw_degrees() * kDegToRad;
    dst->look_delta_pitch = src.pitch_degrees() * kDegToRad;

    // Modifier flags
    dst->flags = 0;
    if (src.sprint) dst->flags |= 0x01;  // Sprint flag
    if (src.sneak)  dst->flags |= 0x02;  // Sneak flag
    if (src.jump)   dst->flags |= 0x04;  // Jump flag
}

void ActionDecoder::to_simulator_action_batch(const MultiDiscreteAction* src,
                                               mc189_action_t* dst,
                                               int batch_size) noexcept {
    for (int i = 0; i < batch_size; ++i) {
        to_simulator_action(src[i], &dst[i]);
    }
}

// ActionMask implementation
constexpr std::array<int, kNumDimensions> ActionMask::kLogitOffsets;

void ActionMask::compute_mask(float* mask_out,
                               bool can_jump,
                               bool can_sprint,
                               bool can_attack,
                               bool can_use_item,
                               bool can_craft) const noexcept {
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();

    // Initialize all to 0 (valid)
    std::fill(mask_out, mask_out + kNumLogits, 0.0f);

    // Mask invalid actions with -inf
    // Jump is at offset 9 (after movement), index 1 is "yes"
    if (!can_jump) {
        mask_out[kLogitOffsets[1] + 1] = kNegInf;  // Jump=yes
    }

    // Sprint is at offset 11 (after movement + jump), index 1 is "yes"
    if (!can_sprint) {
        mask_out[kLogitOffsets[2] + 1] = kNegInf;  // Sprint=yes
    }

    // Attack is at offset 15 (after movement + jump + sprint + sneak), index 1 is "yes"
    if (!can_attack) {
        mask_out[kLogitOffsets[4] + 1] = kNegInf;  // Attack=yes
    }

    // Use item is at offset 17
    if (!can_use_item) {
        mask_out[kLogitOffsets[5] + 1] = kNegInf;  // UseItem=yes
    }

    // Craft is special action at offset 44, index 1 is "craft"
    if (!can_craft) {
        mask_out[kLogitOffsets[9] + static_cast<int>(SpecialAction::kCraft)] = kNegInf;
    }
}

void ActionMask::compute_mask_batch(float* mask_out,
                                     const bool* can_jump,
                                     const bool* can_sprint,
                                     const bool* can_attack,
                                     const bool* can_use_item,
                                     const bool* can_craft,
                                     int batch_size) const noexcept {
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();

    for (int b = 0; b < batch_size; ++b) {
        float* mask = mask_out + b * kNumLogits;
        std::fill(mask, mask + kNumLogits, 0.0f);

        if (can_jump && !can_jump[b]) {
            mask[kLogitOffsets[1] + 1] = kNegInf;
        }
        if (can_sprint && !can_sprint[b]) {
            mask[kLogitOffsets[2] + 1] = kNegInf;
        }
        if (can_attack && !can_attack[b]) {
            mask[kLogitOffsets[4] + 1] = kNegInf;
        }
        if (can_use_item && !can_use_item[b]) {
            mask[kLogitOffsets[5] + 1] = kNegInf;
        }
        if (can_craft && !can_craft[b]) {
            mask[kLogitOffsets[9] + static_cast<int>(SpecialAction::kCraft)] = kNegInf;
        }
    }
}

// ============================================================================
// Speedrun Action Decoder (32 discrete actions)
// ============================================================================

// Look delta magnitude in radians for discrete look actions
constexpr float kLookDeltaRad = 15.0f * (3.14159265358979f / 180.0f);

void decode_speedrun_action(SpeedrunAction action, DecodedAction& out) noexcept {
    // Zero-initialize
    out = DecodedAction{};

    switch (action) {
    case SpeedrunAction::NOOP:
        break;

    // Movement (0-6)
    case SpeedrunAction::FORWARD:
        out.movement[0] = 1.0f;
        break;
    case SpeedrunAction::BACK:
        out.movement[0] = -1.0f;
        break;
    case SpeedrunAction::LEFT:
        out.movement[1] = -1.0f;
        break;
    case SpeedrunAction::RIGHT:
        out.movement[1] = 1.0f;
        break;
    case SpeedrunAction::FORWARD_LEFT:
        out.movement[0] = 0.7071067811865476f;
        out.movement[1] = -0.7071067811865476f;
        break;
    case SpeedrunAction::FORWARD_RIGHT:
        out.movement[0] = 0.7071067811865476f;
        out.movement[1] = 0.7071067811865476f;
        break;

    // Jump combinations (7-10)
    case SpeedrunAction::JUMP:
        out.flags |= 0x01;  // jump
        break;
    case SpeedrunAction::JUMP_FORWARD:
        out.movement[0] = 1.0f;
        out.flags |= 0x01;  // jump
        break;
    case SpeedrunAction::JUMP_SPRINT:
        out.flags |= 0x01 | 0x02;  // jump + sprint
        break;
    case SpeedrunAction::JUMP_SPRINT_FORWARD:
        out.movement[0] = 1.0f;
        out.flags |= 0x01 | 0x02;  // jump + sprint
        break;

    // Combat (11-14)
    case SpeedrunAction::ATTACK:
        out.action_type = 1;
        break;
    case SpeedrunAction::ATTACK_FORWARD:
        out.action_type = 1;
        out.movement[0] = 1.0f;
        break;
    case SpeedrunAction::USE_ITEM:
        out.action_type = 2;
        break;
    case SpeedrunAction::USE_ITEM_FORWARD:
        out.action_type = 2;
        out.movement[0] = 1.0f;
        break;

    // Look (15-18)
    case SpeedrunAction::LOOK_LEFT:
        out.look_delta[0] = -kLookDeltaRad;
        break;
    case SpeedrunAction::LOOK_RIGHT:
        out.look_delta[0] = kLookDeltaRad;
        break;
    case SpeedrunAction::LOOK_UP:
        out.look_delta[1] = -kLookDeltaRad;
        break;
    case SpeedrunAction::LOOK_DOWN:
        out.look_delta[1] = kLookDeltaRad;
        break;

    // Inventory (19-22)
    case SpeedrunAction::SELECT_SLOT_0:
        out.action_type = 3;
        out.action_data = 0;
        break;
    case SpeedrunAction::SELECT_SLOT_1:
        out.action_type = 3;
        out.action_data = 1;
        break;
    case SpeedrunAction::SELECT_SLOT_2:
        out.action_type = 3;
        out.action_data = 2;
        break;
    case SpeedrunAction::SELECT_SLOT_3:
        out.action_type = 3;
        out.action_data = 3;
        break;

    // Special (23-27)
    case SpeedrunAction::SPRINT_TOGGLE:
        out.flags |= 0x02;  // sprint
        break;
    case SpeedrunAction::SNEAK_TOGGLE:
        out.flags |= 0x04;  // sneak
        break;
    case SpeedrunAction::DROP_ITEM:
        out.action_type = 3;
        out.action_data = 10;  // drop sentinel
        break;
    case SpeedrunAction::SWAP_HANDS:
        out.action_type = 3;
        out.action_data = 11;  // swap sentinel
        break;
    case SpeedrunAction::OPEN_INVENTORY:
        out.action_type = 3;
        out.action_data = 12;  // inventory sentinel
        break;

    // Quick actions (28-31)
    case SpeedrunAction::QUICK_CRAFT:
        out.action_type = 3;
        out.action_data = 20;  // craft sentinel
        break;
    case SpeedrunAction::QUICK_EAT:
        out.action_type = 2;   // use item (food)
        out.action_data = 21;  // eat sentinel
        break;
    case SpeedrunAction::THROW_PEARL:
        out.action_type = 2;   // use item
        out.action_data = 22;  // pearl sentinel
        break;
    case SpeedrunAction::THROW_EYE:
        out.action_type = 2;   // use item
        out.action_data = 23;  // eye sentinel
        break;

    default:
        break;
    }
}

void decode_speedrun_action_batch(const int32_t* actions, DecodedAction* out,
                                  int batch_size) noexcept {
    int i = 0;
    for (; i + 3 < batch_size; i += 4) {
        decode_speedrun_action(static_cast<SpeedrunAction>(actions[i + 0]), out[i + 0]);
        decode_speedrun_action(static_cast<SpeedrunAction>(actions[i + 1]), out[i + 1]);
        decode_speedrun_action(static_cast<SpeedrunAction>(actions[i + 2]), out[i + 2]);
        decode_speedrun_action(static_cast<SpeedrunAction>(actions[i + 3]), out[i + 3]);
    }
    for (; i < batch_size; ++i) {
        decode_speedrun_action(static_cast<SpeedrunAction>(actions[i]), out[i]);
    }
}

void decoded_to_simulator_action(const DecodedAction& src,
                                 mc189_action_t* dst) noexcept {
    std::memset(dst, 0, sizeof(mc189_action_t));

    // Map action_type to simulator action enum
    switch (src.action_type) {
    case 1:  // attack
        dst->action = MC189_ACTION_ATTACK;
        break;
    case 2:  // use item
        dst->action = MC189_ACTION_USE;
        break;
    case 3:  // special
        switch (src.action_data) {
        case 0: case 1: case 2: case 3: case 4:
        case 5: case 6: case 7: case 8:
            // Hotbar slot selection
            dst->action = static_cast<mc189_action_type_t>(
                MC189_ACTION_HOTBAR_0 + src.action_data);
            break;
        case 10:  // drop
            dst->action = MC189_ACTION_DROP;
            break;
        case 12:  // inventory
            dst->action = MC189_ACTION_INVENTORY;
            break;
        case 20:  // craft
            dst->action = MC189_ACTION_CRAFT;
            break;
        default:
            dst->action = MC189_ACTION_NONE;
            break;
        }
        break;
    default:
        // Determine from movement
        if (src.movement[0] > 0.5f) {
            dst->action = MC189_ACTION_FORWARD;
        } else if (src.movement[0] < -0.5f) {
            dst->action = MC189_ACTION_BACKWARD;
        } else if (src.movement[1] < -0.5f) {
            dst->action = MC189_ACTION_LEFT;
        } else if (src.movement[1] > 0.5f) {
            dst->action = MC189_ACTION_RIGHT;
        } else {
            dst->action = MC189_ACTION_NONE;
        }
        break;
    }

    // Look deltas (already in radians)
    dst->look_delta_yaw = src.look_delta[0];
    dst->look_delta_pitch = src.look_delta[1];

    // Modifier flags: simulator uses bit0=sprint, bit1=sneak, bit2=jump
    // DecodedAction uses bit0=jump, bit1=sprint, bit2=sneak
    dst->flags = 0;
    if (src.flags & 0x02) dst->flags |= 0x01;  // sprint
    if (src.flags & 0x04) dst->flags |= 0x02;  // sneak
    if (src.flags & 0x01) dst->flags |= 0x04;  // jump
}

MultiDiscreteAction speedrun_to_multi_discrete(SpeedrunAction action) noexcept {
    MultiDiscreteAction mda;  // Default: noop with neutral look

    switch (action) {
    case SpeedrunAction::NOOP:
        break;
    case SpeedrunAction::FORWARD:
        mda.movement = 1;
        break;
    case SpeedrunAction::BACK:
        mda.movement = 2;
        break;
    case SpeedrunAction::LEFT:
        mda.movement = 3;
        break;
    case SpeedrunAction::RIGHT:
        mda.movement = 4;
        break;
    case SpeedrunAction::FORWARD_LEFT:
        mda.movement = 5;
        break;
    case SpeedrunAction::FORWARD_RIGHT:
        mda.movement = 6;
        break;
    case SpeedrunAction::JUMP:
        mda.jump = 1;
        break;
    case SpeedrunAction::JUMP_FORWARD:
        mda.movement = 1;
        mda.jump = 1;
        break;
    case SpeedrunAction::JUMP_SPRINT:
        mda.jump = 1;
        mda.sprint = 1;
        break;
    case SpeedrunAction::JUMP_SPRINT_FORWARD:
        mda.movement = 1;
        mda.jump = 1;
        mda.sprint = 1;
        break;
    case SpeedrunAction::ATTACK:
        mda.attack = 1;
        break;
    case SpeedrunAction::ATTACK_FORWARD:
        mda.movement = 1;
        mda.attack = 1;
        break;
    case SpeedrunAction::USE_ITEM:
        mda.use_item = 1;
        break;
    case SpeedrunAction::USE_ITEM_FORWARD:
        mda.movement = 1;
        mda.use_item = 1;
        break;
    case SpeedrunAction::LOOK_LEFT:
        mda.look_yaw = 2;   // -15 degrees
        break;
    case SpeedrunAction::LOOK_RIGHT:
        mda.look_yaw = 6;   // +15 degrees
        break;
    case SpeedrunAction::LOOK_UP:
        mda.look_pitch = 2;  // -5 degrees (up)
        break;
    case SpeedrunAction::LOOK_DOWN:
        mda.look_pitch = 4;  // +5 degrees (down)
        break;
    case SpeedrunAction::SELECT_SLOT_0:
        mda.hotbar_slot = 0;
        break;
    case SpeedrunAction::SELECT_SLOT_1:
        mda.hotbar_slot = 1;
        break;
    case SpeedrunAction::SELECT_SLOT_2:
        mda.hotbar_slot = 2;
        break;
    case SpeedrunAction::SELECT_SLOT_3:
        mda.hotbar_slot = 3;
        break;
    case SpeedrunAction::SPRINT_TOGGLE:
        mda.sprint = 1;
        break;
    case SpeedrunAction::SNEAK_TOGGLE:
        mda.sneak = 1;
        break;
    case SpeedrunAction::DROP_ITEM:
        mda.special = static_cast<int8_t>(SpecialAction::kDrop);
        break;
    case SpeedrunAction::SWAP_HANDS:
        // No direct equivalent in multi-discrete; treated as noop
        break;
    case SpeedrunAction::OPEN_INVENTORY:
        mda.special = static_cast<int8_t>(SpecialAction::kOpenInventory);
        break;
    case SpeedrunAction::QUICK_CRAFT:
        mda.special = static_cast<int8_t>(SpecialAction::kCraft);
        break;
    case SpeedrunAction::QUICK_EAT:
        mda.use_item = 1;  // Best approximation
        break;
    case SpeedrunAction::THROW_PEARL:
        mda.use_item = 1;
        mda.hotbar_slot = 3;  // Slot 3 = ender pearls
        break;
    case SpeedrunAction::THROW_EYE:
        mda.use_item = 1;
        mda.hotbar_slot = 3;  // Slot 3 = eyes of ender
        break;
    default:
        break;
    }

    return mda;
}

} // namespace mc189
