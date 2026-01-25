#pragma once
#include "mc189/items.h"
#include "mc189/simulator.h"
#include <array>
#include <cstdint>
#include <vector>

namespace mc189 {

// Inventory constants
constexpr uint32_t HOTBAR_SIZE = 9;
constexpr uint32_t MAIN_INV_SIZE = 27;
constexpr uint32_t TOTAL_INV_SIZE = 36;
constexpr uint32_t ARMOR_SLOTS = 4;

class InventoryManager {
public:
    explicit InventoryManager(uint32_t num_envs);

    // Item operations
    bool add_item(uint32_t env_id, ItemID item, uint32_t count = 1);
    bool remove_item(uint32_t env_id, ItemID item, uint32_t count = 1);
    bool has_item(uint32_t env_id, ItemID item, uint32_t count = 1) const;
    uint32_t count_item(uint32_t env_id, ItemID item) const;

    // Slot operations
    ItemStack get_slot(uint32_t env_id, uint32_t slot) const;
    void set_slot(uint32_t env_id, uint32_t slot, ItemStack stack);
    void swap_slots(uint32_t env_id, uint32_t slot1, uint32_t slot2);
    void select_slot(uint32_t env_id, uint32_t slot);
    uint32_t get_selected_slot(uint32_t env_id) const;
    ItemStack get_held_item(uint32_t env_id) const;

    // Armor
    void equip_armor(uint32_t env_id, uint32_t armor_slot, ItemStack item);
    ItemStack get_armor(uint32_t env_id, uint32_t armor_slot) const;
    float get_total_armor(uint32_t env_id) const;

    // Durability
    void damage_item(uint32_t env_id, uint32_t slot, uint32_t amount = 1);
    bool is_broken(uint32_t env_id, uint32_t slot) const;

    // Serialization for GPU buffer
    void write_to_buffer(uint32_t env_id, PlayerFull& player) const;
    void read_from_buffer(uint32_t env_id, const PlayerFull& player);

    // Reset
    void clear(uint32_t env_id);
    void set_default_loadout(uint32_t env_id, uint32_t stage);

private:
    uint32_t find_slot_for_item(uint32_t env_id, ItemID item) const;
    uint32_t find_empty_slot(uint32_t env_id) const;

    std::vector<std::array<ItemStack, TOTAL_INV_SIZE>> inventories_;
    std::vector<std::array<ItemStack, ARMOR_SLOTS>> armors_;
    std::vector<uint32_t> selected_slots_;
};

} // namespace mc189
