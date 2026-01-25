#!/usr/bin/env python3
"""Production PPO training script for Minecraft speedrun with curriculum learning.

This script provides a complete training pipeline with:
- Curriculum-based training with automatic stage progression
- Checkpoint saving/loading with curriculum state preservation
- Wandb and TensorBoard logging
- Multi-GPU support via SB3's SubprocVecEnv or native batch envs
- Comprehensive hyperparameter configuration
- Evaluation callbacks with best model tracking

Usage:
    # Basic training
    python train_ppo_speedrun.py --num-envs 64 --total-timesteps 100_000_000

    # Resume from checkpoint
    python train_ppo_speedrun.py --resume logs/checkpoint_10000000_steps.zip

    # Start from specific curriculum stage
    python train_ppo_speedrun.py --start-stage 3 --num-envs 128

    # Enable wandb logging
    python train_ppo_speedrun.py --wandb --wandb-project minecraft-speedrun

Requirements:
    pip install stable-baselines3 gymnasium numpy
    Optional: pip install wandb torch
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add minecraft_sim to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

# Optional imports with availability flags
HAS_SB3 = False
HAS_TORCH = False
HAS_WANDB = False
HAS_MPS = False
DEVICE_COUNT = 0

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

    HAS_SB3 = True
except ImportError:
    PPO = None  # type: ignore
    BaseCallback = object  # Dummy for class definition
    CallbackList = None  # type: ignore
    CheckpointCallback = None  # type: ignore
    EvalCallback = None  # type: ignore
    VecMonitor = None  # type: ignore
    VecNormalize = None  # type: ignore

try:
    import torch

    HAS_TORCH = True
    # Check CUDA availability (may not exist on Apple Silicon builds)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available"):
        DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # Check MPS (Apple Silicon) availability
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
        HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    torch = None  # type: ignore

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    wandb = None  # type: ignore

from minecraft_sim import SB3VecDragonFightEnv
from minecraft_sim.curriculum import CurriculumManager, StageID


@dataclass
class TrainingConfig:
    """Configuration for PPO training run."""

    # Environment
    num_envs: int = 64
    observation_size: int = 48
    action_size: int = 17

    # Curriculum
    start_stage: int = 1
    auto_curriculum: bool = True
    curriculum_threshold: float = 0.7
    min_episodes_per_stage: int = 100

    # Training
    total_timesteps: int = 100_000_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True

    # Network architecture
    policy_type: str = "MlpPolicy"
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    activation_fn: str = "tanh"

    # Normalization
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    # Checkpointing
    checkpoint_freq: int = 100_000
    keep_checkpoints: int = 5

    # Evaluation
    eval_freq: int = 50_000
    n_eval_episodes: int = 10

    # Logging
    log_dir: str = "logs"
    tensorboard_log: str | None = None
    verbose: int = 1

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "minecraft-speedrun"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)

    # Device
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "num_envs": self.num_envs,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "net_arch": self.net_arch,
            "normalize_obs": self.normalize_obs,
            "normalize_reward": self.normalize_reward,
            "start_stage": self.start_stage,
            "auto_curriculum": self.auto_curriculum,
            "device": self.device,
        }


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning with automatic stage progression."""

    def __init__(
        self,
        curriculum_manager: CurriculumManager,
        env: Any,  # VecNormalize when SB3 available
        config: TrainingConfig,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum = curriculum_manager
        self.env = env
        self.config = config
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.successes: list[bool] = []
        self.stage_start_timestep = 0
        self.stage_episodes = 0

    def _on_step(self) -> bool:
        # Check for episode completion in infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # Determine success: positive reward or dragon killed
                success = ep_reward > 0
                self.successes.append(success)
                self.stage_episodes += 1

                # Record to curriculum manager
                if self.curriculum.current_stage is not None:
                    self.curriculum.record_episode(
                        success=success,
                        reward=ep_reward,
                        ticks=ep_length,
                    )

        # Check for stage advancement
        if self.config.auto_curriculum and self.curriculum.should_advance():
            old_stage = self.curriculum.current_stage
            new_stage_config = self.curriculum.advance_stage()
            if new_stage_config is not None:
                if self.verbose > 0:
                    print(f"\n{'=' * 60}")
                    print(f"CURRICULUM: Stage {old_stage.name} -> {new_stage_config.name}")
                    print(f"  Episodes completed: {self.stage_episodes}")
                    if self.episode_rewards:
                        print(f"  Mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")
                    print(f"{'=' * 60}\n")

                # Reset stage tracking
                self.stage_start_timestep = self.num_timesteps
                self.stage_episodes = 0

                # Log to wandb if enabled
                if HAS_WANDB and self.config.use_wandb and wandb is not None:
                    wandb.log(
                        {
                            "curriculum/stage": new_stage_config.id.value,
                            "curriculum/stage_name": new_stage_config.name,
                            "curriculum/timestep": self.num_timesteps,
                        }
                    )

        return True

    def _on_training_end(self) -> None:
        if self.verbose > 0:
            summary = self.curriculum.get_training_summary()
            print("\nCurriculum Training Summary:")
            print(f"  Stages mastered: {summary['stages_mastered']}/{summary['total_stages']}")
            print(f"  Total episodes: {summary['total_episodes']}")


class MetricsCallback(BaseCallback):
    """Callback for detailed metrics logging to wandb/tensorboard."""

    def __init__(
        self,
        config: TrainingConfig,
        log_freq: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.config = config
        self.log_freq = log_freq
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.start_time = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            elapsed = time.time() - self.start_time
            sps = self.num_timesteps / elapsed

            metrics = {
                "rollout/ep_rew_mean": np.mean(self.episode_rewards[-100:]),
                "rollout/ep_len_mean": np.mean(self.episode_lengths[-100:]),
                "rollout/ep_rew_max": max(self.episode_rewards[-100:]),
                "time/fps": sps,
                "time/total_timesteps": self.num_timesteps,
                "time/iterations": self.n_calls,
            }

            if HAS_WANDB and self.config.use_wandb and wandb is not None:
                wandb.log(metrics, step=self.num_timesteps)

        return True


class ProgressiveCheckpointCallback(BaseCallback):
    """Extended checkpoint callback that saves curriculum state."""

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        curriculum_manager: CurriculumManager | None = None,
        name_prefix: str = "ppo_speedrun",
        keep_checkpoints: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.curriculum = curriculum_manager
        self.name_prefix = name_prefix
        self.keep_checkpoints = keep_checkpoints
        self.saved_checkpoints: list[Path] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save model checkpoint
            model_path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saved checkpoint to {model_path}")

            # Save curriculum progress alongside model checkpoint
            if self.curriculum is not None:
                curriculum_path = (
                    self.save_path / f"{self.name_prefix}_{self.num_timesteps}_curriculum.json"
                )
                self.curriculum.save_progress(curriculum_path)
                self.saved_checkpoints.append(curriculum_path)

                # Clean up old checkpoints
                if len(self.saved_checkpoints) > self.keep_checkpoints:
                    old_checkpoint = self.saved_checkpoints.pop(0)
                    if old_checkpoint.exists():
                        old_checkpoint.unlink()
                    # Also remove corresponding model checkpoint
                    model_checkpoint = old_checkpoint.with_name(
                        old_checkpoint.name.replace("_curriculum.json", "_steps.zip")
                    )
                    if model_checkpoint.exists():
                        model_checkpoint.unlink()

        return True


def create_env(
    config: TrainingConfig,
    curriculum_manager: CurriculumManager | None = None,
) -> Any:
    """Create vectorized environment with normalization wrappers.

    Args:
        config: Training configuration.
        curriculum_manager: Optional curriculum manager for stage-based training.

    Returns:
        Wrapped vectorized environment.
    """
    if not HAS_SB3:
        raise ImportError("stable_baselines3 required for environment creation")

    # Determine shader directory
    shader_dir = Path(__file__).parent.parent / "cpp" / "shaders"
    if not shader_dir.exists():
        shader_dir = None

    # Create base environment
    env = SB3VecDragonFightEnv(
        num_envs=config.num_envs,
        shader_dir=shader_dir,
    )

    # Add monitoring
    env = VecMonitor(env)

    # Add normalization
    if config.normalize_obs or config.normalize_reward:
        env = VecNormalize(
            env,
            norm_obs=config.normalize_obs,
            norm_reward=config.normalize_reward,
            clip_obs=config.clip_obs,
            clip_reward=config.clip_reward,
            gamma=config.gamma,
        )

    return env


def create_eval_env(config: TrainingConfig) -> Any:
    """Create evaluation environment (typically with fewer envs)."""
    eval_config = TrainingConfig(
        num_envs=min(8, config.num_envs),
        normalize_obs=config.normalize_obs,
        normalize_reward=False,  # Don't normalize eval rewards
        clip_obs=config.clip_obs,
        gamma=config.gamma,
    )
    return create_env(eval_config)


def get_device(config: TrainingConfig) -> str:
    """Determine the best available device."""
    if config.device != "auto":
        return config.device

    if HAS_TORCH:
        if DEVICE_COUNT > 0:
            return "cuda"
        if HAS_MPS:
            return "mps"
    return "cpu"


def create_model(
    env: Any,
    config: TrainingConfig,
    resume_path: str | None = None,
) -> Any:
    """Create or load PPO model.

    Args:
        env: Vectorized environment.
        config: Training configuration.
        resume_path: Path to checkpoint to resume from.

    Returns:
        PPO model instance.
    """
    if not HAS_SB3 or PPO is None:
        raise ImportError("stable_baselines3 required for model creation")

    device = get_device(config)

    # Determine policy kwargs
    policy_kwargs: dict[str, Any] = {
        "net_arch": config.net_arch,
    }

    if HAS_TORCH and torch is not None:
        import torch.nn as nn

        fn_map = {"tanh": nn.Tanh, "relu": nn.ReLU}
        policy_kwargs["activation_fn"] = fn_map.get(config.activation_fn, nn.Tanh)

    if resume_path is not None:
        print(f"Loading model from {resume_path}")
        model = PPO.load(resume_path, env=env, device=device)
        # Update learning rate if changed
        model.learning_rate = config.learning_rate
        return model

    # Determine tensorboard log path
    tb_log = config.tensorboard_log or config.log_dir

    return PPO(
        policy=config.policy_type,
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        normalize_advantage=config.normalize_advantage,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log,
        device=device,
        verbose=config.verbose,
    )


def setup_wandb(config: TrainingConfig) -> None:
    """Initialize wandb logging."""
    if not HAS_WANDB or wandb is None:
        print("Warning: wandb not installed. Install with: pip install wandb")
        return

    run_name = config.wandb_run_name or f"ppo_speedrun_{int(time.time())}"

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=run_name,
        config=config.to_dict(),
        tags=config.wandb_tags + [f"stage_{config.start_stage}"],
        sync_tensorboard=True,
    )


def train(config: TrainingConfig, resume_path: str | None = None) -> None:
    """Run PPO training with curriculum learning.

    Args:
        config: Training configuration.
        resume_path: Optional path to checkpoint for resuming.
    """
    if not HAS_SB3:
        print("Error: stable_baselines3 is required for training.")
        print("Install with: pip install stable_baselines3")
        sys.exit(1)

    # Create log directory
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize curriculum
    curriculum = CurriculumManager()
    start_stage = StageID(config.start_stage)
    curriculum.start_training(start_stage)

    # Load curriculum progress if resuming
    if resume_path is not None:
        curriculum_path = Path(resume_path).with_name(
            Path(resume_path).stem.replace("_steps", "_curriculum") + ".json"
        )
        if curriculum_path.exists():
            print(f"Loading curriculum progress from {curriculum_path}")
            curriculum.load_progress(curriculum_path)

    # Create environments
    print(f"Creating {config.num_envs} parallel environments...")
    env = create_env(config, curriculum)
    eval_env = create_eval_env(config)

    # Create model
    model = create_model(env, config, resume_path)

    # Setup wandb
    if config.use_wandb:
        setup_wandb(config)

    # Setup callbacks
    callbacks: list[Any] = []

    # Checkpoint callback with curriculum state
    checkpoint_callback = ProgressiveCheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(log_dir),
        curriculum_manager=curriculum,
        name_prefix="ppo_speedrun",
        keep_checkpoints=config.keep_checkpoints,
        verbose=config.verbose,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        verbose=config.verbose,
    )
    callbacks.append(eval_callback)

    # Curriculum callback
    curriculum_callback = CurriculumCallback(
        curriculum_manager=curriculum,
        env=env,
        config=config,
        verbose=config.verbose,
    )
    callbacks.append(curriculum_callback)

    # Metrics callback
    metrics_callback = MetricsCallback(config=config, verbose=config.verbose)
    callbacks.append(metrics_callback)

    # Print training info
    device = get_device(config)
    print(f"\n{'=' * 60}")
    print("PPO Speedrun Training")
    print(f"{'=' * 60}")
    print(f"  Device:           {device}")
    if HAS_TORCH and DEVICE_COUNT > 1:
        print(f"  GPUs available:   {DEVICE_COUNT}")
    print(f"  Environments:     {config.num_envs}")
    print(f"  Total timesteps:  {config.total_timesteps:,}")
    print(f"  Starting stage:   {start_stage.name}")
    print(f"  Batch size:       {config.batch_size}")
    print(f"  Learning rate:    {config.learning_rate}")
    print(f"  Log directory:    {log_dir}")
    if config.use_wandb:
        print(f"  Wandb project:    {config.wandb_project}")
    print(f"{'=' * 60}\n")

    # Start training
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Save final model and curriculum state
        final_model_path = log_dir / "final_model.zip"
        model.save(final_model_path)
        print(f"Saved final model to {final_model_path}")

        final_curriculum_path = log_dir / "final_curriculum.json"
        curriculum.save_progress(final_curriculum_path)
        print(f"Saved curriculum progress to {final_curriculum_path}")

        # Save normalization stats
        if HAS_SB3 and VecNormalize is not None and isinstance(env, VecNormalize):
            stats_path = log_dir / "vec_normalize.pkl"
            env.save(stats_path)
            print(f"Saved normalization stats to {stats_path}")

        # Close environments
        env.close()
        eval_env.close()

        # Close wandb
        if config.use_wandb and HAS_WANDB and wandb is not None:
            wandb.finish()

    # Print final summary
    elapsed = time.time() - start_time
    summary = curriculum.get_training_summary()

    print(f"\n{'=' * 60}")
    print("Training Complete")
    print(f"{'=' * 60}")
    print(f"  Total time:       {elapsed / 3600:.2f} hours")
    print(f"  Stages mastered:  {summary['stages_mastered']}/{summary['total_stages']}")
    print(f"  Total episodes:   {summary['total_episodes']:,}")
    print(f"  Final stage:      {summary['current_stage']}")
    print(f"{'=' * 60}\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO on Minecraft speedrun simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Number of parallel environments",
    )

    # Curriculum
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help="Starting curriculum stage (1=basic survival, 6=end fight)",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable automatic curriculum progression",
    )

    # Training
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for updates",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
        help="Steps per rollout",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=4,
        help="Epochs per update",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient",
    )

    # Network
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden layer sizes",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100_000,
        help="Checkpoint frequency (timesteps)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Evaluation frequency (timesteps)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs and checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Wandb
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="minecraft-speedrun",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity/team name",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Wandb run name",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for training",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Build configuration from args
    config = TrainingConfig(
        num_envs=args.num_envs,
        start_stage=args.start_stage,
        auto_curriculum=not args.no_curriculum,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        net_arch=args.net_arch,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        log_dir=args.log_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_name,
        device=args.device,
        verbose=args.verbose,
    )

    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
