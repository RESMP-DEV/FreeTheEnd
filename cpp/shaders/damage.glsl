/*
 * Minecraft 1.8.9 Damage Calculation - GLSL Compute Shader
 *
 * This shader implements exact Minecraft 1.8.9 damage mechanics:
 * 1. Armor reduction (armor points, max 20)
 * 2. Protection enchantment reduction (EPF, max 20)
 * 3. Resistance potion effect reduction (amplifier 0-4)
 *
 * Reference: Minecraft Wiki damage mechanics (1.8.9 version)
 *
 * Order of application:
 *   raw_damage -> armor -> protection -> resistance -> final_damage
 */

#ifndef DAMAGE_GLSL
#define DAMAGE_GLSL

/*
 * Calculate armor damage reduction.
 *
 * In Minecraft 1.8.9, armor reduces damage by:
 *   reduction = min(20, armor_points) / 25
 *   damage_after = raw_damage * (1 - reduction)
 *
 * Each armor point provides 4% damage reduction, capping at 80% (20 points).
 *
 * Armor values by piece (full diamond = 20):
 *   Helmet:     Leather=1, Gold=2, Chain=2, Iron=2, Diamond=3
 *   Chestplate: Leather=3, Gold=5, Chain=5, Iron=6, Diamond=8
 *   Leggings:   Leather=2, Gold=3, Chain=4, Iron=5, Diamond=6
 *   Boots:      Leather=1, Gold=1, Chain=1, Iron=2, Diamond=3
 *
 * @param raw_damage    The incoming damage before any reduction
 * @param armor_points  Total armor points (0-20, values >20 are clamped)
 * @return              Damage after armor reduction
 */
float calculate_armor_reduction(float raw_damage, float armor_points) {
    // Clamp armor to valid range [0, 20]
    float clamped_armor = clamp(armor_points, 0.0, 20.0);

    // Each armor point = 4% reduction, max 80%
    float reduction_factor = 1.0 - (clamped_armor / 25.0);

    return raw_damage * reduction_factor;
}

/*
 * Calculate protection enchantment damage reduction.
 *
 * In Minecraft 1.8.9, protection enchantments provide EPF (Enchantment Protection Factor):
 *   - Protection:      EPF = level * 1
 *   - Fire Protection: EPF = level * 2 (fire damage only)
 *   - Blast Protection: EPF = level * 2 (explosion damage only)
 *   - Projectile Protection: EPF = level * 2 (projectile damage only)
 *   - Feather Falling: EPF = level * 3 (fall damage only)
 *
 * Total EPF is summed across all armor pieces and capped at 20.
 * Each EPF point provides 4% damage reduction.
 *   reduction = min(20, total_epf) / 25
 *   damage_after = damage * (1 - reduction)
 *
 * For this function, protection_level represents the already-calculated total EPF.
 *
 * @param damage            Damage after armor reduction
 * @param protection_level  Total EPF from all protection enchantments (0-20)
 * @return                  Damage after protection reduction
 */
float calculate_protection_reduction(float damage, int protection_level) {
    // Clamp EPF to valid range [0, 20]
    float clamped_epf = clamp(float(protection_level), 0.0, 20.0);

    // Each EPF point = 4% reduction, max 80%
    float reduction_factor = 1.0 - (clamped_epf / 25.0);

    return damage * reduction_factor;
}

/*
 * Calculate resistance potion effect damage reduction.
 *
 * In Minecraft 1.8.9, Resistance effect provides:
 *   reduction = 20% * (amplifier + 1)
 *
 * Amplifier values:
 *   Resistance I:   amplifier = 0 -> 20% reduction
 *   Resistance II:  amplifier = 1 -> 40% reduction
 *   Resistance III: amplifier = 2 -> 60% reduction
 *   Resistance IV:  amplifier = 3 -> 80% reduction
 *   Resistance V:   amplifier = 4 -> 100% reduction (immune)
 *
 * Note: amplifier of -1 means no resistance effect (0% reduction).
 *
 * @param damage              Damage after armor and protection
 * @param resistance_amplifier  Resistance amplifier (-1 = none, 0 = Resistance I, etc.)
 * @return                    Final damage after resistance reduction
 */
float calculate_resistance_reduction(float damage, int resistance_amplifier) {
    // No resistance effect
    if (resistance_amplifier < 0) {
        return damage;
    }

    // Calculate reduction: 20% per level (amplifier + 1)
    // Clamp to max amplifier 4 (Resistance V = 100% reduction)
    int clamped_amp = min(resistance_amplifier, 4);
    float reduction = 0.2 * float(clamped_amp + 1);

    // Cap at 100% reduction
    reduction = min(reduction, 1.0);

    return damage * (1.0 - reduction);
}

/*
 * Apply full damage calculation pipeline.
 *
 * Applies all damage reductions in the correct Minecraft 1.8.9 order:
 *   1. Armor reduction
 *   2. Protection enchantment reduction
 *   3. Resistance potion effect reduction
 *
 * @param raw_damage  Raw incoming damage
 * @param armor       Total armor points (0-20)
 * @param protection  Total EPF from protection enchantments (0-20)
 * @param resistance  Resistance amplifier (-1 = none, 0-4 for Resistance I-V)
 * @return            Final damage dealt to entity
 */
float apply_damage(float raw_damage, int armor, int protection, int resistance) {
    // Step 1: Armor reduction
    float after_armor = calculate_armor_reduction(raw_damage, float(armor));

    // Step 2: Protection enchantment reduction
    float after_protection = calculate_protection_reduction(after_armor, protection);

    // Step 3: Resistance potion effect reduction
    float final_damage = calculate_resistance_reduction(after_protection, resistance);

    // Ensure non-negative damage
    return max(final_damage, 0.0);
}

/*
 * Batch damage calculation for compute shader workgroups.
 *
 * Equipment data is encoded as uint32 per entity (see equipment_encoding.h):
 *   bits [0-4]:   armor points (0-20)
 *   bits [5-9]:   protection EPF (0-20)
 *   bits [10-13]: resistance amplifier + 1 (0 = none, 1-5 = Resistance I-V)
 *   bits [14-31]: reserved
 *
 * @param equipment_packed  Packed equipment data
 * @return                  Unpacked struct with armor, protection, resistance
 */
struct EquipmentData {
    int armor;
    int protection;
    int resistance;  // -1 = none, 0-4 = amplifier
};

EquipmentData unpack_equipment(uint equipment_packed) {
    EquipmentData data;

    // Extract armor points (bits 0-4, 5 bits, range 0-31, clamped to 0-20)
    data.armor = int(equipment_packed & 0x1Fu);

    // Extract protection EPF (bits 5-9, 5 bits, range 0-31, clamped to 0-20)
    data.protection = int((equipment_packed >> 5) & 0x1Fu);

    // Extract resistance amplifier + 1 (bits 10-13, 4 bits, range 0-15)
    // Value 0 = no resistance, 1-5 = Resistance I-V
    uint resistance_encoded = (equipment_packed >> 10) & 0xFu;
    data.resistance = int(resistance_encoded) - 1;

    return data;
}

/*
 * Pack equipment data into uint32.
 *
 * @param armor       Armor points (0-20)
 * @param protection  Protection EPF (0-20)
 * @param resistance  Resistance amplifier (-1 = none, 0-4)
 * @return            Packed uint32
 */
uint pack_equipment(int armor, int protection, int resistance) {
    uint packed = 0u;

    // Pack armor (bits 0-4)
    packed |= uint(clamp(armor, 0, 20));

    // Pack protection (bits 5-9)
    packed |= uint(clamp(protection, 0, 20)) << 5;

    // Pack resistance + 1 (bits 10-13), 0 = no effect
    uint resistance_encoded = uint(clamp(resistance + 1, 0, 5));
    packed |= resistance_encoded << 10;

    return packed;
}

/*
 * Compute final damage from packed equipment.
 *
 * Convenience function combining unpack and apply_damage.
 *
 * @param raw_damage        Raw incoming damage
 * @param equipment_packed  Packed equipment uint32
 * @return                  Final damage after all reductions
 */
float compute_damage_packed(float raw_damage, uint equipment_packed) {
    EquipmentData eq = unpack_equipment(equipment_packed);
    return apply_damage(raw_damage, eq.armor, eq.protection, eq.resistance);
}

#endif // DAMAGE_GLSL
