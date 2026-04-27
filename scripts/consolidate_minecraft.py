#!/usr/bin/env python3
"""Consolidate Minecraft simulation files from scattered task directories."""

import json
import shutil
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

workspace = Path(__file__).parent.parent / "agent_workspace"
mc_dir = workspace / "minecraft_combat_sim"

# Create output directories
(mc_dir / "cpp" / "shaders").mkdir(parents=True, exist_ok=True)
(mc_dir / "oracle").mkdir(parents=True, exist_ok=True)
(mc_dir / "verification").mkdir(parents=True, exist_ok=True)

manifest = {"shaders": [], "verification": [], "oracle": []}
copied = 0

# Minecraft-related keywords
mc_keywords = [
    "aabb",
    "damage",
    "combat",
    "knockback",
    "attack",
    "sword",
    "armor",
    "block",
    "terrain",
    "chunk",
    "world",
    "biome",
    "structure",
    "stronghold",
    "portal",
    "nether",
    "end",
    "dimension",
    "teleport",
    "mob",
    "zombie",
    "skeleton",
    "enderman",
    "blaze",
    "dragon",
    "crystal",
    "inventory",
    "crafting",
    "smelting",
    "furnace",
    "recipe",
    "hunger",
    "health",
    "regen",
    "effect",
    "potion",
    "experience",
    "physics",
    "velocity",
    "collision",
    "movement",
    "gravity",
    "jump",
    "raycast",
    "hitbox",
    "iframe",
    "game_tick",
    "simulation",
    "eye_of_ender",
]


def is_minecraft_related(filepath):
    logger.debug("is_minecraft_related: filepath=%s", filepath)
    name = filepath.name.lower()
    return any(kw in name for kw in mc_keywords)


# Find and copy shaders (.comp files)
for comp_file in workspace.rglob("*.comp"):
    if "minecraft_combat_sim" in str(comp_file):
        continue
    if is_minecraft_related(comp_file):
        dest = mc_dir / "cpp" / "shaders" / comp_file.name
        if not dest.exists():
            shutil.copy2(comp_file, dest)
            manifest["shaders"].append(
                {"file": comp_file.name, "source": str(comp_file.parent.parent.name)}
            )
            copied += 1

# Find and copy GLSL include files
for glsl_file in workspace.rglob("*.glsl"):
    if "minecraft_combat_sim" in str(glsl_file):
        continue
    if is_minecraft_related(glsl_file):
        dest = mc_dir / "cpp" / "shaders" / glsl_file.name
        if not dest.exists():
            shutil.copy2(glsl_file, dest)
            manifest["shaders"].append(
                {"file": glsl_file.name, "source": str(glsl_file.parent.parent.parent.name)}
            )
            copied += 1

# Find and copy verification files
for py_file in workspace.rglob("*_verifier.py"):
    if "minecraft_combat_sim" in str(py_file):
        continue
    dest = mc_dir / "verification" / py_file.name
    if not dest.exists():
        shutil.copy2(py_file, dest)
        manifest["verification"].append(
            {"file": py_file.name, "source": str(py_file.parent.parent.name)}
        )
        copied += 1

# Find test generators
for py_file in workspace.rglob("*_test_generator.py"):
    if "minecraft_combat_sim" in str(py_file):
        continue
    dest = mc_dir / "verification" / py_file.name
    if not dest.exists():
        shutil.copy2(py_file, dest)
        manifest["verification"].append(
            {"file": py_file.name, "source": str(py_file.parent.parent.name)}
        )
        copied += 1

# Find JSON test cases
for json_file in workspace.rglob("*.json"):
    if "minecraft_combat_sim" in str(json_file):
        continue
    name = json_file.name.lower()
    if "test_case" in name or "expected" in name:
        dest = mc_dir / "verification" / json_file.name
        if not dest.exists():
            shutil.copy2(json_file, dest)
            manifest["verification"].append(
                {"file": json_file.name, "source": str(json_file.parent.parent.name)}
            )
            copied += 1

# Write manifest
manifest_path = mc_dir / "consolidation_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Consolidated {copied} files into minecraft_combat_sim/")
print(f"  Shaders:      {len(manifest['shaders'])}")
print(f"  Verification: {len(manifest['verification'])}")
print(f"  Oracle:       {len(manifest['oracle'])}")
print(f"\nManifest: {manifest_path}")
