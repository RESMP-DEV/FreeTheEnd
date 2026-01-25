// action_decoder.h - Multi-discrete action space decoder for Minecraft RL
// Bridges factored action representation (10 dimensions) with simulator API
//
// Action space dimensions:
//   0. Movement (9): none, forward, back, left, right, forward-left, forward-right, back-left, back-right
//   1. Jump (2): no, yes
//   2. Sprint (2): no, yes
//   3. Sneak (2): no, yes
//   4. Attack (2): no, yes
//   5. Use item (2): no, yes
//   6. Look yaw (9): -90, -45, -15, -5, 0, +5, +15, +45, +90 degrees
//   7. Look pitch (7): -45, -15, -5, 0, +5, +15, +45 degrees
//   8. Hotbar slot (9): slots 0-8
//   9. Special (4): none, craft, drop, open_inventory
//
// Total combinations: 9 * 2 * 2 * 2 * 2 * 2 * 9 * 7 * 9 * 4 = 1,632,960

#ifndef MC189_ACTION_DECODER_H
#define MC189_ACTION_DECODER_H

#include <cstdint>
#include <array>
#include <cmath>

namespace mc189 {

// Action space dimension sizes
constexpr int kNumMovement = 9;
constexpr int kNumJump = 2;
constexpr int kNumSprint = 2;
constexpr int kNumSneak = 2;
constexpr int kNumAttack = 2;
constexpr int kNumUseItem = 2;
constexpr int kNumLookYaw = 9;
constexpr int kNumLookPitch = 7;
constexpr int kNumHotbarSlot = 9;
constexpr int kNumSpecial = 4;

constexpr int kNumDimensions = 10;
constexpr std::array<int, kNumDimensions> kNvec = {
    kNumMovement, kNumJump, kNumSprint, kNumSneak, kNumAttack, kNumUseItem,
    kNumLookYaw, kNumLookPitch, kNumHotbarSlot, kNumSpecial};

// Total action space size (for flat representation)
constexpr int64_t kTotalActions = 1632960;

// Total logits for factored policy network (sum of dimensions)
constexpr int kNumLogits = kNumMovement + kNumJump + kNumSprint + kNumSneak +
                           kNumAttack + kNumUseItem + kNumLookYaw +
                           kNumLookPitch + kNumHotbarSlot + kNumSpecial; // 48

// Movement direction enum
enum class Movement : int8_t {
    kNone = 0,
    kForward = 1,
    kBack = 2,
    kLeft = 3,
    kRight = 4,
    kForwardLeft = 5,
    kForwardRight = 6,
    kBackLeft = 7,
    kBackRight = 8,
};

// Special action enum
enum class SpecialAction : int8_t {
    kNone = 0,
    kCraft = 1,
    kDrop = 2,
    kOpenInventory = 3,
};

// Yaw angle lookup table (degrees)
constexpr std::array<float, kNumLookYaw> kYawAngles = {
    -90.0f, -45.0f, -15.0f, -5.0f, 0.0f, 5.0f, 15.0f, 45.0f, 90.0f};

// Pitch angle lookup table (degrees)
constexpr std::array<float, kNumLookPitch> kPitchAngles = {
    -45.0f, -15.0f, -5.0f, 0.0f, 5.0f, 15.0f, 45.0f};

// Multi-discrete action representation (packed into 10 int8 values)
struct alignas(16) MultiDiscreteAction {
    int8_t movement;    // 0-8
    int8_t jump;        // 0-1
    int8_t sprint;      // 0-1
    int8_t sneak;       // 0-1
    int8_t attack;      // 0-1
    int8_t use_item;    // 0-1
    int8_t look_yaw;    // 0-8
    int8_t look_pitch;  // 0-6
    int8_t hotbar_slot; // 0-8
    int8_t special;     // 0-3
    int8_t _pad[6];     // Padding to 16 bytes for SIMD alignment

    // Default constructor (no-op action)
    constexpr MultiDiscreteAction() noexcept
        : movement(0), jump(0), sprint(0), sneak(0), attack(0), use_item(0),
          look_yaw(4), look_pitch(3), hotbar_slot(0), special(0), _pad{} {}

    // Constructor from individual components
    constexpr MultiDiscreteAction(int8_t mov, int8_t jmp, int8_t spr,
                                   int8_t snk, int8_t atk, int8_t use,
                                   int8_t yaw, int8_t pitch, int8_t slot,
                                   int8_t spec) noexcept
        : movement(mov), jump(jmp), sprint(spr), sneak(snk), attack(atk),
          use_item(use), look_yaw(yaw), look_pitch(pitch), hotbar_slot(slot),
          special(spec), _pad{} {}

    // Get yaw delta in degrees
    [[nodiscard]] constexpr float yaw_degrees() const noexcept {
        return kYawAngles[look_yaw];
    }

    // Get pitch delta in degrees
    [[nodiscard]] constexpr float pitch_degrees() const noexcept {
        return kPitchAngles[look_pitch];
    }

    // Get yaw delta in radians
    [[nodiscard]] float yaw_radians() const noexcept {
        return yaw_degrees() * (3.14159265358979f / 180.0f);
    }

    // Get pitch delta in radians
    [[nodiscard]] float pitch_radians() const noexcept {
        return pitch_degrees() * (3.14159265358979f / 180.0f);
    }

    // Check if this is a movement action
    [[nodiscard]] constexpr bool is_moving() const noexcept {
        return movement != 0;
    }

    // Get movement vector (forward, strafe) normalized
    [[nodiscard]] constexpr std::pair<float, float> movement_vector() const noexcept {
        constexpr float kSqrt2Inv = 0.7071067811865476f;
        switch (movement) {
        case 0: return {0.0f, 0.0f};         // None
        case 1: return {1.0f, 0.0f};         // Forward
        case 2: return {-1.0f, 0.0f};        // Back
        case 3: return {0.0f, -1.0f};        // Left
        case 4: return {0.0f, 1.0f};         // Right
        case 5: return {kSqrt2Inv, -kSqrt2Inv};  // Forward-left
        case 6: return {kSqrt2Inv, kSqrt2Inv};   // Forward-right
        case 7: return {-kSqrt2Inv, -kSqrt2Inv}; // Back-left
        case 8: return {-kSqrt2Inv, kSqrt2Inv};  // Back-right
        default: return {0.0f, 0.0f};
        }
    }

    // Validate all indices are within bounds
    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return movement >= 0 && movement < kNumMovement &&
               jump >= 0 && jump < kNumJump &&
               sprint >= 0 && sprint < kNumSprint &&
               sneak >= 0 && sneak < kNumSneak &&
               attack >= 0 && attack < kNumAttack &&
               use_item >= 0 && use_item < kNumUseItem &&
               look_yaw >= 0 && look_yaw < kNumLookYaw &&
               look_pitch >= 0 && look_pitch < kNumLookPitch &&
               hotbar_slot >= 0 && hotbar_slot < kNumHotbarSlot &&
               special >= 0 && special < kNumSpecial;
    }

    // Equality comparison
    [[nodiscard]] constexpr bool operator==(const MultiDiscreteAction& other) const noexcept {
        return movement == other.movement && jump == other.jump &&
               sprint == other.sprint && sneak == other.sneak &&
               attack == other.attack && use_item == other.use_item &&
               look_yaw == other.look_yaw && look_pitch == other.look_pitch &&
               hotbar_slot == other.hotbar_slot && special == other.special;
    }
};

static_assert(sizeof(MultiDiscreteAction) == 16, "MultiDiscreteAction must be 16 bytes");

// Forward declaration for simulator API integration
struct mc189_action_t;

class ActionDecoder {
public:
    // Encode multi-discrete action to flat index
    [[nodiscard]] static int64_t encode_flat(const MultiDiscreteAction& action) noexcept;

    // Decode flat index to multi-discrete action
    [[nodiscard]] static MultiDiscreteAction decode_flat(int64_t flat_idx) noexcept;

    // Batch encode (SIMD optimized for batch_size % 4 == 0)
    static void encode_flat_batch(const MultiDiscreteAction* actions,
                                   int64_t* flat_indices, int batch_size) noexcept;

    // Batch decode (SIMD optimized for batch_size % 4 == 0)
    static void decode_flat_batch(const int64_t* flat_indices,
                                   MultiDiscreteAction* actions, int batch_size) noexcept;

    // Convert multi-discrete action to simulator API action
    // Requires include of mc189/simulator_api.h
    static void to_simulator_action(const MultiDiscreteAction& src,
                                    mc189_action_t* dst) noexcept;

    // Batch convert multi-discrete actions to simulator API actions
    static void to_simulator_action_batch(const MultiDiscreteAction* src,
                                          mc189_action_t* dst, int batch_size) noexcept;

    // Create a no-op action
    [[nodiscard]] static constexpr MultiDiscreteAction noop() noexcept {
        return MultiDiscreteAction{};
    }

    // Compute multipliers for flat encoding (precomputed at compile time)
    [[nodiscard]] static constexpr std::array<int64_t, kNumDimensions> compute_multipliers() noexcept {
        std::array<int64_t, kNumDimensions> mults{};
        mults[kNumDimensions - 1] = 1;
        for (int i = kNumDimensions - 2; i >= 0; --i) {
            mults[i] = mults[i + 1] * kNvec[i + 1];
        }
        return mults;
    }

private:
    // Precomputed multipliers for flat encoding/decoding
    static constexpr std::array<int64_t, kNumDimensions> kMultipliers = compute_multipliers();
};

// Verify multiplier computation produces expected total
static_assert(ActionDecoder::compute_multipliers()[0] * kNvec[0] == kTotalActions,
              "Multiplier computation error");

// ============================================================================
// SPEEDRUN ACTION SPACE (32 discrete actions)
// Single-integer action representation for simpler policy networks.
// Maps to the same simulator API as the multi-discrete space above.
// ============================================================================

enum class SpeedrunAction : int32_t {
    // Movement (0-6)
    NOOP = 0,
    FORWARD = 1,
    BACK = 2,
    LEFT = 3,
    RIGHT = 4,
    FORWARD_LEFT = 5,
    FORWARD_RIGHT = 6,

    // Jump combinations (7-10)
    JUMP = 7,
    JUMP_FORWARD = 8,
    JUMP_SPRINT = 9,
    JUMP_SPRINT_FORWARD = 10,

    // Combat (11-14)
    ATTACK = 11,
    ATTACK_FORWARD = 12,
    USE_ITEM = 13,
    USE_ITEM_FORWARD = 14,

    // Look (15-18)
    LOOK_LEFT = 15,
    LOOK_RIGHT = 16,
    LOOK_UP = 17,
    LOOK_DOWN = 18,

    // Inventory (19-22)
    SELECT_SLOT_0 = 19,
    SELECT_SLOT_1 = 20,
    SELECT_SLOT_2 = 21,
    SELECT_SLOT_3 = 22,

    // Special (23-27)
    SPRINT_TOGGLE = 23,
    SNEAK_TOGGLE = 24,
    DROP_ITEM = 25,
    SWAP_HANDS = 26,
    OPEN_INVENTORY = 27,

    // Quick actions (28-31)
    QUICK_CRAFT = 28,
    QUICK_EAT = 29,
    THROW_PEARL = 30,
    THROW_EYE = 31,

    NUM_ACTIONS = 32,
};

constexpr int32_t kNumSpeedrunActions = static_cast<int32_t>(SpeedrunAction::NUM_ACTIONS);

// Decoded action produced by the speedrun action decoder.
// Provides a uniform interface regardless of whether the source was
// a multi-discrete action or a single discrete speedrun action.
struct DecodedAction {
    float movement[3];      // [forward, strafe, up] each in [-1, 1]
    float look_delta[2];    // [yaw, pitch] change in radians
    uint32_t action_type;   // 0=none, 1=attack, 2=use, 3=special
    uint32_t action_data;   // Slot number, item ID, etc.
    uint32_t flags;         // bit0=jump, bit1=sprint, bit2=sneak

    constexpr DecodedAction() noexcept
        : movement{0.0f, 0.0f, 0.0f}, look_delta{0.0f, 0.0f},
          action_type(0), action_data(0), flags(0) {}
};

// Decode a single speedrun action into the uniform DecodedAction format
void decode_speedrun_action(SpeedrunAction action, DecodedAction& out) noexcept;

// Batch decode speedrun actions
void decode_speedrun_action_batch(const int32_t* actions, DecodedAction* out,
                                  int batch_size) noexcept;

// Convert a DecodedAction to the simulator API action struct
void decoded_to_simulator_action(const DecodedAction& src,
                                 mc189_action_t* dst) noexcept;

// Convert a SpeedrunAction directly to a MultiDiscreteAction (lossy mapping)
MultiDiscreteAction speedrun_to_multi_discrete(SpeedrunAction action) noexcept;

// Action masking for invalid actions (e.g., can't jump while in air)
class ActionMask {
public:
    // Create mask array for policy network logits
    // mask[i] = 0.0f for valid, -inf for invalid
    void compute_mask(float* mask_out,
                      bool can_jump = true,
                      bool can_sprint = true,
                      bool can_attack = true,
                      bool can_use_item = true,
                      bool can_craft = true) const noexcept;

    // Batch version for multiple environments
    void compute_mask_batch(float* mask_out,
                            const bool* can_jump,
                            const bool* can_sprint,
                            const bool* can_attack,
                            const bool* can_use_item,
                            const bool* can_craft,
                            int batch_size) const noexcept;

    // Logit offsets for each dimension (for masking)
    static constexpr std::array<int, kNumDimensions> logit_offsets() noexcept {
        std::array<int, kNumDimensions> offsets{};
        offsets[0] = 0;
        for (int i = 1; i < kNumDimensions; ++i) {
            offsets[i] = offsets[i - 1] + kNvec[i - 1];
        }
        return offsets;
    }

private:
    static constexpr std::array<int, kNumDimensions> kLogitOffsets = logit_offsets();
};

} // namespace mc189

#endif // MC189_ACTION_DECODER_H
