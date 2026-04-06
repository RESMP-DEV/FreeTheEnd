# Physics Constants Audit: MC 1.8.9 Compliance

This document audits physics constants in the shader files against Minecraft 1.8.9 vanilla values.

## Reference Values (MC 1.8.9)

| Constant | Expected Value | Notes |
|----------|----------------|-------|
| GRAVITY | 0.08 | blocks/tick² (player/entity) |
| DRAG_AIR | 0.98 | horizontal velocity multiplier per tick |
| DRAG_GROUND | 0.91 | applied when on ground (slipperiness) |
| JUMP_VELOCITY | 0.42 | initial Y velocity on jump |
| KNOCKBACK_BASE | 0.4 | horizontal knockback factor |
| KNOCKBACK_VERTICAL_CAP | 0.4 | max vertical knockback |
| SPRINT_FACTOR | 1.3 | sprint speed multiplier |
| PEARL_GRAVITY | 0.03 | ender pearl gravity |
| PEARL_DRAG | 0.99 | ender pearl air drag |

## Audit Results

### GRAVITY (Expected: 0.08)

| File | Value | Status |
|------|-------|--------|
| `mob_ai_base.comp:43` | 0.08 | ✅ CORRECT |
| `game_tick_mvk.comp:21` | -0.08 | ✅ CORRECT (signed) |
| `dragon_fight_mvk.comp:22` | -0.08 | ✅ CORRECT (signed) |
| `mob_ai_overworld_hostile.comp:100` | 0.08 | ✅ CORRECT |
| `mob_ai_silverfish.comp:349` | 0.08 | ✅ CORRECT |
| `batch_actions.comp:22` | -0.08 | ✅ CORRECT (signed) |
| `batch_step.comp:23` | -0.08 | ✅ CORRECT (signed) |
| `dragon_ai_full.comp:21` | -0.08 | ✅ CORRECT (signed) |
| `dragon_knockback.comp:51` | 0.08 | ✅ CORRECT |
| `memory_layout.glsl:561` | -0.08 | ✅ CORRECT (signed) |
| `item_physics.comp:13` | 0.04 | ⚠️ DIFFERENT (item physics uses half gravity) |
| `fireball_tick.comp:18` | 0.02 | ⚠️ DIFFERENT (fireballs have custom gravity) |

**Note**: Negative sign indicates subtraction direction in physics calculation. Item and fireball gravity differ intentionally.

### DRAG_AIR (Expected: 0.98)

| File | Value | Status |
|------|-------|--------|
| `game_tick_mvk.comp:22` | 0.98 | ✅ CORRECT |
| `dragon_fight_mvk.comp:23` | 0.98 | ✅ CORRECT |
| `batch_actions.comp:23` | 0.98 | ✅ CORRECT |
| `batch_step.comp:24` | 0.98 | ✅ CORRECT |
| `dragon_ai_full.comp:22` | 0.98 | ✅ CORRECT |
| `memory_layout.glsl:562` | 0.98 | ✅ CORRECT |
| `item_physics.comp:15` | 0.98 | ✅ CORRECT |
| `mob_ai_base.comp:44` | 0.02 | ❌ **MISMATCH** - uses `(1.0 - DRAG_AIR)` pattern |
| `mob_ai_overworld_hostile.comp:103` | 0.02 | ❌ **MISMATCH** - same issue |
| `mob_ai_silverfish.comp:350` | 0.02 | ❌ **MISMATCH** - same issue |

**Critical Issue**: `mob_ai_base.comp`, `mob_ai_overworld_hostile.comp`, and `mob_ai_silverfish.comp` define DRAG_AIR as 0.02 and use `velocity *= (1.0 - DRAG)`. This results in velocity × 0.98, which is mathematically correct but the naming is inverted.

### DRAG_GROUND (Expected: 0.91)

| File | Value | Status |
|------|-------|--------|
| `game_tick.comp:584` | 0.91 | ✅ CORRECT (inline) |
| `game_tick.comp:885` | 0.91 | ✅ CORRECT (mobs, inline) |
| `game_tick_mvk.comp:23` | 0.6 | ❌ **MISMATCH** |
| `dragon_fight_mvk.comp:24` | 0.6 | ❌ **MISMATCH** |
| `batch_actions.comp:24` | 0.6 | ❌ **MISMATCH** |
| `batch_step.comp:25` | 0.6 | ❌ **MISMATCH** |
| `memory_layout.glsl:563` | 0.6 | ❌ **MISMATCH** |
| `item_physics.comp:14` | 0.5 | ❌ **MISMATCH** (should be 0.4 for items) |

**Critical Issue**: Multiple shader files use 0.6 for ground drag instead of 0.91. This causes players to slide ~34% less on the ground than vanilla MC. The value 0.6 is incorrect and likely a typo or misunderstanding of the friction model.

**MC 1.8.9 Ground Friction Model**:
```
horizontal_velocity *= slipperiness * 0.91
```
Where slipperiness is 0.6 for most blocks (hence the 0.6 value), but the TOTAL drag multiplier should be 0.6 × 0.91 = 0.546 for normal blocks. For ice, slipperiness is 0.98.

### JUMP_VELOCITY (Expected: 0.42)

| File | Value | Status |
|------|-------|--------|
| `mob_ai_base.comp:47` | 0.42 | ✅ CORRECT |
| `game_tick_mvk.comp:26` | 0.42 | ✅ CORRECT |
| `dragon_fight_mvk.comp:28` | 0.42 | ✅ CORRECT |
| `mob_ai_overworld_hostile.comp:102` | 0.42 | ✅ CORRECT |
| `mob_spawning_overworld.comp:68-69` | 0.42 | ✅ CORRECT |
| `batch_actions.comp:28` | 0.42 | ✅ CORRECT |
| `batch_step.comp:29` | 0.42 | ✅ CORRECT |
| `memory_layout.glsl:566` | 0.42 | ✅ CORRECT |

All JUMP_VELOCITY values are correct.

### KNOCKBACK (Expected: 0.4 base, 0.4 vertical cap)

| File | Value | Status |
|------|-------|--------|
| `dragon_fight_mvk.comp:36` | 0.4 | ✅ CORRECT |
| `dragon_fight_optimized.comp:44` | 0.4 | ✅ CORRECT |
| `batch_actions.comp:37` | 0.4 | ✅ CORRECT |
| `batch_step.comp:37` | 0.4 | ✅ CORRECT |
| `mob_combat.comp:177-178` | 0.4, 0.4 | ✅ CORRECT (base and vertical) |

All KNOCKBACK_BASE values are correct. Note: `mob_combat.comp` also defines KNOCKBACK_IRON_GOLEM_VERTICAL = 1.0, which is correct for iron golem special attacks.

### SPRINT_FACTOR (Expected: 1.3)

| File | Value | Status |
|------|-------|--------|
| `game_tick_mvk.comp:25` | 1.3 | ✅ CORRECT |
| `dragon_fight_mvk.comp:26` | 1.3 | ✅ CORRECT |
| `batch_actions.comp:26` | 1.3 | ✅ CORRECT |
| `batch_step.comp:27` | 1.3 | ✅ CORRECT |
| `memory_layout.glsl:565` | 1.3 | ✅ CORRECT |

All SPRINT values are correct.

### PEARL_GRAVITY (Expected: 0.03)

| File | Value | Status |
|------|-------|--------|
| `ender_pearl.comp:30` | 0.03 | ✅ CORRECT |
| `eye_of_ender.comp:18` | 0.03 | ✅ CORRECT |
| `eye_of_ender_full.comp:38` | 0.03 | ✅ CORRECT |

All pearl/eye gravity values are correct.

### PEARL_DRAG (Expected: 0.99)

| File | Value | Status |
|------|-------|--------|
| `ender_pearl.comp:31` | 0.99 | ✅ CORRECT |
| `eye_of_ender.comp:19` | 0.99 | ✅ CORRECT |
| `eye_of_ender_full.comp:39` | 0.99 | ✅ CORRECT |

All pearl drag values are correct.

## Summary of Issues

### Critical Mismatches Requiring Fix

1. **DRAG_GROUND = 0.6** in multiple files should be **0.91** (or implement proper slipperiness × 0.91):
   - `game_tick_mvk.comp:23`
   - `dragon_fight_mvk.comp:24`
   - `batch_actions.comp:24`
   - `batch_step.comp:25`
   - `memory_layout.glsl:563`

2. **DRAG_AIR naming inconsistency** - `mob_ai_base.comp`, `mob_ai_overworld_hostile.comp`, `mob_ai_silverfish.comp` use inverted naming (0.02 with subtraction pattern). Consider renaming to `DRAG_FACTOR` to avoid confusion.

### Acceptable Deviations

- `item_physics.comp` uses GRAVITY=0.04 (intentional: items fall slower)
- `fireball_tick.comp` uses FIREBALL_GRAVITY=0.02 (intentional: fireballs have reduced gravity)
- `item_physics.comp` uses DRAG_GROUND=0.5 (item-specific friction)

## Recommended Fixes

```glsl
// In game_tick_mvk.comp, dragon_fight_mvk.comp, batch_actions.comp, batch_step.comp, memory_layout.glsl:
// BEFORE:
const float DRAG_GROUND = 0.6;
// AFTER:
const float DRAG_GROUND = 0.91;

// Or for full accuracy with block slipperiness:
const float BLOCK_SLIPPERINESS = 0.6;  // Default block
const float FRICTION_FACTOR = 0.91;
// Use: velocity *= BLOCK_SLIPPERINESS * FRICTION_FACTOR; // = 0.546
```

---
*Audit performed: 2026-01-20*
*Against: Minecraft Java Edition 1.8.9 decompiled sources*
