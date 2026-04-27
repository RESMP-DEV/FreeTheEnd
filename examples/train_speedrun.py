#!/usr/bin/env python3
"""Train PPO agent on Minecraft 1.8.9 speedrun.

Production-ready training script with:
- Curriculum learning for staged progression
- Checkpoint and evaluation callbacks
- VecNormalize for stable training
- WandB integration (optional)
- Graceful interrupt handling
- Resume from checkpoint support

Usage:
    python train_speedrun.py --config configs/default.yaml
    python train_speedrun.py --num-envs 128 --total-timesteps 1e9
    python train_speedrun.py --resume checkpoints/model_10M.zip
    python train_speedrun.py --wandb --wandb-project my-project
"""

from __future__ import annotations

import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from callbacks import CurriculumCallback, MetricsCallback, WandBCallback
from training_config import TrainingConfig

from minecraft_sim import SB3VecFreeTheEndEnv

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv

# Global for graceful shutdown
_shutdown_requested = False


def signal_handler(signum: int, frame: object) -> None:
    """Handle interrupt signals for graceful shutdown."""
    global _shutdown_requested
    if _shutdown_requested:
        print("\n[Train] Force quit requested, exiting immediately...")
        sys.exit(1)
    _shutdown_requested = True
    print("\n[Train] Interrupt received, finishing current update and saving...")


def make_env(
    config: TrainingConfig,
    eval_mode: bool = False,
    vecnorm_path: str | None = None,
) -> VecEnv:
    """Create training or evaluation environment.

    Args:
        config: Training configuration
        eval_mode: If True, create smaller eval environment
        vecnorm_path: Path to load VecNormalize stats from

    Returns:
        Wrapped vectorized environment
    """
    env = SB3VecFreeTheEndEnv(
        num_envs=8 if eval_mode else config.env.num_envs,
        start_stage=config.env.start_stage,
        curriculum=config.curriculum.enabled and not eval_mode,
        max_episode_steps=config.env.max_episode_steps,
    )

    # Apply normalization
    if vecnorm_path and Path(vecnorm_path).exists():
        # Load existing normalization stats
        env = VecNormalize.load(vecnorm_path, env)
        if eval_mode:
            env.training = False
            env.norm_reward = False
    elif config.normalize_obs or config.normalize_reward:
        env = VecNormalize(
            env,
            norm_obs=config.normalize_obs,
            norm_reward=config.normalize_reward and not eval_mode,
            clip_obs=config.clip_obs,
            clip_reward=config.clip_reward,
            training=not eval_mode,
        )

    env = VecMonitor(env)
    return env


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent on Minecraft speedrun",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument("--config", type=str, help="Config YAML file path")

    # Environment
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument(
        "--start-stage", type=int, default=1, help="Starting curriculum stage (1-6)"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=36000, help="Max steps per episode"
    )

    # Training
    parser.add_argument(
        "--total-timesteps", type=float, default=1e8, help="Total training timesteps"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--resume-vecnorm", type=str, help="Path to VecNormalize stats to load")

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")

    # Curriculum
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument(
        "--stage-threshold", type=float, default=0.7, help="Success rate to advance stage"
    )

    # Normalization
    parser.add_argument(
        "--no-normalize-obs", action="store_true", help="Disable observation normalization"
    )
    parser.add_argument(
        "--no-normalize-reward", action="store_true", help="Disable reward normalization"
    )

    # Logging
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--run-name", type=str, default="speedrun", help="Run name prefix")
    parser.add_argument(
        "--checkpoint-freq", type=int, default=500_000, help="Checkpoint save frequency"
    )
    parser.add_argument("--eval-freq", type=int, default=100_000, help="Evaluation frequency")
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Console log interval (updates)"
    )

    # WandB
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb-project", type=str, default="minecraft-speedrun", help="WandB project"
    )
    parser.add_argument("--wandb-name", type=str, help="WandB run name (defaults to run_name)")

    # Hardware
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, mps, cpu)")
    parser.add_argument("--seed", type=int, help="Random seed")

    return parser.parse_args()


def main() -> int:
    """Main training entry point.

    Returns:
        Exit code (0 for success)
    """
    args = parse_args()

    # Load or create config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
        print(f"[Train] Loaded config from {args.config}")
    else:
        config = TrainingConfig()

    # Override from command line
    config.env.num_envs = args.num_envs
    config.env.start_stage = args.start_stage
    config.env.max_episode_steps = args.max_episode_steps
    config.total_timesteps = int(args.total_timesteps)
    config.log_dir = args.log_dir
    config.run_name = args.run_name
    config.checkpoint_freq = args.checkpoint_freq
    config.eval_freq = args.eval_freq
    config.log_interval = args.log_interval
    config.device = args.device

    # PPO overrides
    config.ppo.learning_rate = args.lr
    config.ppo.n_steps = args.n_steps
    config.ppo.batch_size = args.batch_size
    config.ppo.n_epochs = args.n_epochs
    config.ppo.gamma = args.gamma
    config.ppo.ent_coef = args.ent_coef

    # Curriculum overrides
    config.curriculum.enabled = not args.no_curriculum
    if args.stage_threshold != 0.7:
        config.curriculum.stage_thresholds = [args.stage_threshold] * 6

    # Normalization overrides
    config.normalize_obs = not args.no_normalize_obs
    config.normalize_reward = not args.no_normalize_reward

    # Set seed
    if args.seed is not None:
        set_random_seed(args.seed)
        config.env.seed = args.seed
        print(f"[Train] Random seed set to {args.seed}")

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.run_name}_{timestamp}"
    log_path = Path(config.log_dir) / run_name
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "checkpoints").mkdir(exist_ok=True)
    (log_path / "best_model").mkdir(exist_ok=True)

    # Save config
    config_path = log_path / "config.yaml"
    config.to_yaml(str(config_path))
    print(f"[Train] Config saved to {config_path}")

    # Install signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create environments
    print(f"[Train] Creating {config.env.num_envs} training environments...")
    vecnorm_path = args.resume_vecnorm
    if args.resume and not vecnorm_path:
        # Try to find vecnormalize next to checkpoint
        resume_dir = Path(args.resume).parent
        candidate = resume_dir / "vecnormalize.pkl"
        if candidate.exists():
            vecnorm_path = str(candidate)

    train_env = make_env(config, vecnorm_path=vecnorm_path)
    print("[Train] Creating 8 evaluation environments...")
    eval_env = make_env(config, eval_mode=True, vecnorm_path=vecnorm_path)

    # Create or load model
    if args.resume:
        print(f"[Train] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env, device=config.device)
        # Preserve learning rate from command line if specified
        if args.lr != 3e-4:
            model.learning_rate = args.lr
    else:
        print("[Train] Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config.ppo.learning_rate,
            n_steps=config.ppo.n_steps,
            batch_size=config.ppo.batch_size,
            n_epochs=config.ppo.n_epochs,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            clip_range=config.ppo.clip_range,
            ent_coef=config.ppo.ent_coef,
            vf_coef=config.ppo.vf_coef,
            max_grad_norm=config.ppo.max_grad_norm,
            verbose=1,
            tensorboard_log=str(log_path / "tensorboard"),
            device=config.device,
        )

    print(f"[Train] Model device: {model.device}")
    print(f"[Train] Policy: {model.policy.__class__.__name__}")

    # Build callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=max(config.checkpoint_freq // config.env.num_envs, 1),
            save_path=str(log_path / "checkpoints"),
            name_prefix="model",
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            eval_freq=max(config.eval_freq // config.env.num_envs, 1),
            n_eval_episodes=10,
            best_model_save_path=str(log_path / "best_model"),
            deterministic=True,
        ),
        MetricsCallback(log_path, save_freq=10_000),
    ]

    # Add curriculum if enabled
    if config.curriculum.enabled:
        callbacks.append(
            CurriculumCallback(
                train_env,
                stage_thresholds=config.curriculum.stage_thresholds,
                min_episodes=config.curriculum.min_episodes_per_stage,
                success_window=config.curriculum.success_window,
            )
        )

    # Add WandB if requested
    if args.wandb:
        callbacks.append(
            WandBCallback(
                project=args.wandb_project,
                run_name=args.wandb_name or run_name,
                config=config.to_dict(),
            )
        )

    callback_list = CallbackList(callbacks)

    # Print training info
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Total timesteps:     {config.total_timesteps:>15,}")
    print(f"Num environments:    {config.env.num_envs:>15}")
    print(f"Start stage:         {config.env.start_stage:>15}")
    print(f"Curriculum enabled:  {config.curriculum.enabled:>15}")
    print(f"Normalize obs:       {config.normalize_obs:>15}")
    print(f"Normalize reward:    {config.normalize_reward:>15}")
    print(f"Learning rate:       {config.ppo.learning_rate:>15}")
    print(f"Batch size:          {config.ppo.batch_size:>15}")
    print(f"N steps:             {config.ppo.n_steps:>15}")
    print(f"Device:              {str(model.device):>15}")
    print(f"Log path:            {log_path}")
    print("=" * 60 + "\n")

    # Train
    try:
        print("[Train] Starting training...")
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback_list,
            log_interval=config.log_interval,
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )
    except KeyboardInterrupt:

import logging

logger = logging.getLogger(__name__)

        print("\n[Train] Training interrupted")
    finally:
        # Save final model
        final_model_path = log_path / "final_model.zip"
        print(f"[Train] Saving final model to {final_model_path}")
        model.save(str(final_model_path))

        # Save normalization stats
        if isinstance(train_env, VecNormalize) or (
            hasattr(train_env, "venv") and isinstance(train_env.venv, VecNormalize)
        ):
            norm_env = train_env.venv if hasattr(train_env, "venv") else train_env
            norm_path = log_path / "final_vecnormalize.pkl"
            norm_env.save(str(norm_path))
            print(f"[Train] Saved VecNormalize to {norm_path}")

        # Cleanup
        train_env.close()
        eval_env.close()

    print(f"\n[Train] Training complete! Logs saved to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
