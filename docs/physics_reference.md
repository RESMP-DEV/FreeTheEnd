# Physics Reference

This document describes the physics implementation in the Minecraft 1.8.9 simulator, including source verification methods and known deviations from vanilla behavior.

## Source of Truth

The physics constants and formulas are derived from **MCP 9.18** (Mod Coder Pack) decompiled Minecraft Java Edition 1.8.9 source code.

### Key Reference Classes

| Class | Package | Physics Domain |
|-------|---------|----------------|
| `Entity` | `net.minecraft.entity` | Base entity physics, AABB collision |
| `EntityLivingBase` | `net.minecraft.entity` | Health, damage, knockback, death |
| `EntityPlayer` | `net.minecraft.entity.player` | Movement, sprinting, hunger |
| `EntityPlayerMP` | `net.minecraft.entity.player` | Server-side player state |
| `EntityDragon` | `net.minecraft.entity.boss` | Dragon AI, phases, hitboxes |
| `EntityEnderCrystal` | `net.minecraft.entity.item` | Crystal mechanics |
| `AxisAlignedBB` | `net.minecraft.util` | Bounding box math |
| `MovementInput` | `net.minecraft.util` | Input processing |

### MCP Resources

- **MCPBot Issues**: https://github.com/ModCoderPack/MCPBot-Issues
- **MCP Mappings**: https://github.com/ModCoderPack/MCPMappings
- **Decompiled JAR**: Use MCP 9.18 with Minecraft 1.8.9 client JAR

## Core Physics Constants

### Player Movement

These constants are extracted from `EntityLivingBase.moveEntityWithHeading()` and `EntityPlayer.onLivingUpdate()`:

```glsl
// From game_tick_mvk.comp - matches Entity.java
const float GRAVITY = -0.08;         // blocks/tick^2 (applied as vy += GRAVITY)
const float DRAG = 0.98;             // air drag multiplier per tick
const float GROUND_DRAG = 0.6;       // ground friction (slipperiness)
const float WALK_SPEED = 0.1;        // base movement acceleration
const float SPRINT_MULTIPLIER = 1.3; // EntityPlayer.SPRINT_SPEED_BOOST
const float JUMP_VELOCITY = 0.42;    // initial upward velocity
const float PLAYER_HEIGHT = 1.8;     // AABB height in blocks
const float PLAYER_WIDTH = 0.6;      // AABB width in blocks
```

### Mob Physics

From `EntityLivingBase.java` and mob-specific classes:

```glsl
// From mob_ai_base.comp
const float GRAVITY = 0.08;          // positive here, applied as -= GRAVITY * dt
const float DRAG_AIR = 0.02;         // air resistance
const float DRAG_WATER = 0.2;        // water resistance
const float TERMINAL_VELOCITY = 3.92;// max fall speed
const float JUMP_VELOCITY = 0.42;    // matches player
```

### Movement Formula

The core movement update from `Entity.moveEntity()`:

```
// Per-tick velocity update
velocity.y += GRAVITY;                    // Apply gravity

// Apply drag
velocity.x *= DRAG;
velocity.z *= DRAG;
velocity.y *= DRAG;

// Ground friction (only when on_ground)
if (on_ground) {
    velocity.x *= GROUND_DRAG;
    velocity.z *= GROUND_DRAG;
}

// Move and collide
position += velocity;
```

## Dragon Fight Physics

### Dragon Constants

From `EntityDragon.java`:

```python
# From dragon_fight_verifier.py
CIRCLING_RADIUS = 150.0       # blocks from center
CIRCLING_HEIGHT_MIN = 70.0    # Y coordinate
CIRCLING_HEIGHT_MAX = 120.0
CIRCLING_SPEED = 10.0         # blocks/second
STRAFE_SPEED = 25.0           # during charge attack
LANDING_SPEED = 8.0

# Combat
HEALTH = 200.0                    # HP
HEAD_DAMAGE_MULTIPLIER = 4.0      # 4x damage to head
BODY_DAMAGE_MULTIPLIER = 0.25     # 1/4 damage to body
WING_DAMAGE_MULTIPLIER = 0.0      # wings immune

# Timing (20 ticks = 1 second)
PHASE_DURATION_MIN = 60           # 3 seconds minimum
CIRCLING_DURATION = 200           # ~10 seconds
LANDING_DURATION = 200            # ~10 seconds on perch
DEATH_ANIMATION_TICKS = 200       # 10 seconds

# XP
XP_DROP_FIRST_KILL = 12000
XP_DROP_SUBSEQUENT = 500
```

### Dragon Hitboxes

From `EntityDragonPart.java`:

```python
# Multi-part hitbox dimensions
HEAD_WIDTH = 1.0
HEAD_HEIGHT = 1.0
HEAD_OFFSET = 8.0      # blocks forward from center

BODY_WIDTH = 8.0
BODY_HEIGHT = 4.0

WING_WIDTH = 4.0
WING_HEIGHT = 2.0
WING_OFFSET = 4.0      # perpendicular to body

TAIL_SEGMENT_WIDTH = 2.0
TAIL_SEGMENT_HEIGHT = 2.0
TAIL_SEGMENTS = 3
```

### End Crystal Constants

From `EntityEnderCrystal.java`:

```python
EXPLOSION_POWER = 6.0          # same as charged creeper
EXPLOSION_RADIUS = 12.0        # damage radius
HEAL_RANGE = 32.0              # max distance to heal dragon
HEAL_RATE_PER_TICK = 1.0       # 20 HP/second
CRYSTAL_DAMAGE_TO_DRAGON = 10.0 # when destroyed while healing
```

## Damage Calculation

### General Damage Formula

From `EntityLivingBase.applyArmorCalculations()` and `CombatRules.getDamageAfterAbsorb()`:

```python
# Step 1: Calculate effective armor
effective_armor = armor * (1.0 - protection_level * 0.2)

# Step 2: Damage reduction (diminishing returns)
if effective_armor > 0:
    damage_reduction = effective_armor / (effective_armor + 10.0)
else:
    damage_reduction = 0.0

# Step 3: Apply armor reduction
base_damage = raw_damage * (1.0 - damage_reduction)

# Step 4: Resistance effect
resistance_multiplier = 1.0 - resistance_level * 0.25

# Step 5: Final damage
final_damage = base_damage * resistance_multiplier
```

### Explosion Damage

From `Explosion.doExplosionB()`:

```python
# Distance-based damage falloff
exposure = 1.0 - (distance / explosion_radius)
base_damage = exposure * 2 * explosion_power * 7 + 1

# Armor reduction applies separately
```

## Verification Methods

### Running Verification Tests

The verification suite compares simulator output against expected values:

```bash
# Run all verifiers
cd FreeTheEnd
python -m pytest tests/test_verifiers.py -v

# Run specific verifier
python verification/damage_verifier.py
python verification/dragon_fight_verifier.py

# Generate expected values
python verification/damage_verifier.py --generate
```

### Test Matrix Coverage

The damage verifier tests a full matrix of inputs:

```python
RAW_DAMAGE_VALUES = [1, 5, 10, 15, 20]
ARMOR_VALUES = [0, 4, 8, 12, 16, 20]
PROTECTION_LEVELS = [0, 1, 2, 3, 4]
RESISTANCE_LEVELS = [0, 1, 2]
```

Total: 5 x 6 x 5 x 3 = 450 test cases with exact float comparison.

### Verification Against Live Game

To verify against actual Minecraft 1.8.9:

1. **Install MCP 9.18**: Download from MCP archive
2. **Decompile**: Run decompile script with 1.8.9 client JAR
3. **Extract constants**: Search for physics values in decompiled source
4. **Cross-reference**: Compare with `EntityLivingBase.java`, `Entity.java`

Example verification process:

```bash
# Search for gravity in decompiled source
grep -r "0.08" src/minecraft/net/minecraft/entity/

# Find in EntityLivingBase.java:
# this.motionY -= 0.08D;
```

### Verification Scripts

| Script | Purpose |
|--------|---------|
| `damage_verifier.py` | Damage calculation formulas |
| `damage_test_generator.py` | Generate damage test cases |
| `dragon_fight_verifier.py` | Dragon AI, phases, hitboxes |
| `aabb_verifier.py` | Collision detection |
| `hunger_verifier.py` | Hunger and saturation |
| `status_effects_verifier.py` | Potion effects |
| `xp_verifier.py` | Experience calculations |

## Known Deviations

### Intentional Simplifications

| Mechanic | MC 1.8.9 Behavior | Simulator Behavior | Reason |
|----------|-------------------|-------------------|--------|
| Block collision | Full AABB sweep | Simplified ground check | Performance |
| Fluid physics | Complex buoyancy | Binary in_water flag | Complexity |
| Knockback | Direction-dependent | Simplified push | Stability |
| Random tick | Block-level RNG | Approximated | Determinism |

### Performance Tradeoffs

The simulator prioritizes throughput over exact fidelity:

1. **Collision**: Uses simplified ground detection instead of full swept AABB
2. **Mobs**: Reduced pathfinding complexity
3. **Lighting**: Omitted entirely (not needed for RL)
4. **Chunks**: Simplified to End fight arena

### Floating Point Precision

The simulator uses 32-bit floats (GLSL `float`) while Minecraft uses 64-bit doubles. This can cause minor drift over long simulations but is acceptable for RL training.

## Adding New Physics

### Process

1. **Find source**: Locate the mechanic in MCP decompiled source
2. **Extract formula**: Identify constants and calculations
3. **Write verifier**: Create `verification/*_verifier.py`
4. **Implement shader**: Add to appropriate `.comp` file
5. **Test**: Run verification suite

### Example: Adding Swimming

```glsl
// From EntityLivingBase.java
const float WATER_DRAG = 0.8;       // motionX *= 0.800000011920929D
const float WATER_MOVEMENT = 0.02;  // base water movement speed
const float SWIM_UPWARD = 0.03999999910593033; // upward force when swimming

void applyWaterPhysics(inout Player p, InputState inp) {
    if ((p.flags & FLAG_IN_WATER) != 0) {
        // Apply water drag
        p.velocity *= WATER_DRAG;

        // Swimming movement
        if (inp.movement.y > 0) {
            p.velocity.y += SWIM_UPWARD;
        }
    }
}
```

## References

### Primary Sources

- **MCP 9.18**: Mod Coder Pack for 1.8.9 decompilation
- **Minecraft Wiki**: https://minecraft.wiki/ (Java Edition mechanics)
- **DigMinecraft**: https://www.digminecraft.com/ (game mechanics reference)

### GitHub Resources

- **MCPBot Issues**: https://github.com/ModCoderPack/MCPBot-Issues
- **Mapping Viewer**: https://github.com/kashike/MinecraftMappings

### Community Documentation

- **Minecraft Wiki - Physics**: https://minecraft.wiki/w/Entity#Motion_of_entities
- **Minecraft Wiki - Damage**: https://minecraft.wiki/w/Damage
- **Minecraft Wiki - Dragon**: https://minecraft.wiki/w/Ender_Dragon

## Version Notes

This document is accurate for:
- **Minecraft**: Java Edition 1.8.9
- **MCP**: Version 9.18
- **Simulator**: minecraft_sim (January 2026)

Physics may differ in other Minecraft versions. Always verify against the specific version being simulated.
