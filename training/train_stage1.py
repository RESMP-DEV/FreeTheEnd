#!/usr/bin/env python3
"""Stage 1: Basic Survival Training.

Teaches fundamental Minecraft skills: movement, combat, resource gathering, crafting.

Run with:
    cd /path/to/minecraft_sim
    source .venv/bin/activate
    export PYTHONPATH=$(pwd)/python
    export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json  # macOS only
    python training/train_stage1.py

Hyperparameters tuned per docs/training_guide.md:
- net_arch=[128, 128] for Stage 1-4 tasks
- gamma=0.99 (short-horizon Stage 1; use 0.999 for later stages)
- CPU device for MLP (MPS only helps CNNs)
- VecNormalize for reward/observation scaling

NOTE: Using SpeedrunVecEnv which has:
- Curriculum management
- Stage-specific reward shaping
- Progress tracking
"""

import sys
from pathlib import Path

# Add minecraft_sim to path (relative to training/ folder)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))
sys.path.insert(0, str(project_root / "cpp" / "build"))  # For mc189_core module

import os

os.environ.setdefault("VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from minecraft_sim.curriculum import StageID
from minecraft_sim.speedrun_vec_env import make_speedrun_vec_env


def main():
    print("=" * 60)
    print("Stage 1: Basic Survival Training")
    print("=" * 60)

    # Configuration
    num_envs = 64
    total_timesteps = 1_000_000
    output_dir = Path(__file__).parent.parent

    # Create SpeedrunVecEnv - has built-in curriculum and reward shaping
    # start at Stage 1 (BASIC_SURVIVAL), disable auto-advancement for now
    print(f"Creating SpeedrunVecEnv with {num_envs} parallel instances...")
    base_env = make_speedrun_vec_env(
        num_envs=num_envs,
        initial_stage=StageID.BASIC_SURVIVAL,
        auto_curriculum=False,  # Stay at stage 1 for focused training
        max_ticks_per_episode=2000,  # ~100 sec game time, ensures episode completion
        use_multistage_simulator=True,  # Use survival_tick_v2 shader with hostile mobs
    )

    # Add VecNormalize for observation and reward normalization
    # This stabilizes training by normalizing obs to ~N(0,1) and rewards to reasonable scale
    print("Adding VecNormalize wrapper...")
    env = VecNormalize(
        base_env,
        norm_obs=True,  # Normalize observations
        norm_reward=True,  # Normalize rewards (critical for sparse/shaped rewards)
        clip_obs=10.0,  # Clip normalized obs
        clip_reward=10.0,  # Clip normalized reward
        gamma=0.99,  # Same discount as PPO for return normalization
        training=True,  # Update running stats during training
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Checkpoint callback - save every ~200k timesteps
    # Note: save_freq is in env steps, so divide by num_envs
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(200_000 // num_envs, 1),  # ~200k timesteps between saves
        save_path=str(checkpoint_dir),
        name_prefix="stage1_ppo",
    )

    # PPO model with tuned hyperparameters
    # - CPU is 30x faster than MPS for small MLP networks (no GPU dispatch overhead)
    # - net_arch=[128, 128] per guide recommendation for Stage 1-4
    # - High entropy (0.05) to prevent policy collapse and reward hacking
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,  # Steps per env before update
        batch_size=256,  # Minibatch size for SGD
        n_epochs=4,  # Epochs per update
        gamma=0.99,  # Discount factor (short horizon for Stage 1)
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,  # PPO clipping
        ent_coef=0.05,  # Higher entropy to prevent action spam/collapse
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        verbose=1,
        tensorboard_log=str(output_dir / "logs" / "stage1_ppo"),
        device="cpu",  # CPU faster for small MLPs
        policy_kwargs=dict(
            net_arch=[128, 128],  # Guide: [128, 128] for stages 1-4
        ),
    )

    print(f"\nPolicy network: {model.policy}")

    # Train
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("-" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # Save final model
    model_path = output_dir / "models" / "stage1_basic_survival"
    model_path.parent.mkdir(exist_ok=True)
    model.save(str(model_path))
    print(f"\nModel saved to {model_path}.zip")

    # Save VecNormalize statistics (required for evaluation/deployment)
    vecnorm_path = output_dir / "models" / "stage1_vecnormalize.pkl"
    env.save(str(vecnorm_path))
    print(f"VecNormalize stats saved to {vecnorm_path}")

    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
