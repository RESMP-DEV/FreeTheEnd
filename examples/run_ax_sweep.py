#!/usr/bin/env python3
"""Hyperparameter sweep using Ax/BoTorch.

Local Bayesian optimization - no external service costs.
Uses Gaussian Process surrogate with Expected Improvement acquisition.

Usage:
    python run_ax_sweep.py --trials 50
    python run_ax_sweep.py --trials 100 --parallel 4  # Multi-GPU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.instantiation import ObjectiveProperties

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> dict[str, Any]:
    """Load Ax sweep configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_evaluate(parameters: dict[str, Any]) -> dict[str, float]:
    """Run one training trial and return metrics.

    Args:
        parameters: Hyperparameters from Ax

    Returns:
        Dictionary with objective metric(s)
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    from minecraft_sim import SB3VecFreeTheEndEnv

    # Create environment
    env = SB3VecFreeTheEndEnv(
        num_envs=parameters.get("num_envs", 32),
        curriculum=True,
    )
    env = VecNormalize(env)

    # Create model with swept parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=parameters["learning_rate"],
        n_steps=parameters["n_steps"],
        batch_size=parameters["batch_size"],
        n_epochs=parameters["n_epochs"],
        gamma=parameters["gamma"],
        gae_lambda=parameters["gae_lambda"],
        ent_coef=parameters["ent_coef"],
        clip_range=parameters["clip_range"],
        verbose=0,
    )

    # Train for fixed timesteps
    timesteps = parameters.get("eval_timesteps", 100_000)
    model.learn(total_timesteps=timesteps)

    # Evaluate
    eval_env = SB3VecFreeTheEndEnv(num_envs=8, curriculum=False)
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    successes = 0
    episodes = 0
    obs = eval_env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = eval_env.step(action)
        for i, done in enumerate(dones):
            if done:
                episodes += 1
                if infos[i].get("success", False):
                    successes += 1

    success_rate = successes / max(episodes, 1)

    env.close()
    eval_env.close()

    return {"eval/success_rate": success_rate}


def run_sweep(
    config_path: str = "configs/ax_sweep_config.yaml",
    total_trials: int = 50,
    parallel: int = 1,
) -> None:
    """Run Ax/BoTorch hyperparameter sweep.

    Args:
        config_path: Path to sweep config
        total_trials: Total number of trials
        parallel: Max parallel trials (set to num GPUs)
    """
    config = load_config(config_path)

    # Initialize Ax client
    ax_client = AxClient(
        generation_strategy=None,  # Will use config
        verbose_logging=True,
    )

    # Create experiment
    ax_client.create_experiment(
        name=config["experiment"]["name"],
        parameters=config["parameters"],
        objectives={
            config["experiment"]["objective"]: ObjectiveProperties(
                minimize=config["experiment"]["minimize"]
            )
        },
    )

    print(f"Starting Ax sweep: {total_trials} trials, {parallel} parallel")

    # Run optimization loop
    for i in range(total_trials):
        # Get next trial parameters (Bayesian optimization)
        parameters, trial_index = ax_client.get_next_trial()

        print(f"\n=== Trial {trial_index + 1}/{total_trials} ===")
        print(f"Parameters: {parameters}")

        try:
            # Run training and evaluation
            metrics = train_evaluate(parameters)

            # Report results to Ax
            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data=metrics,
            )

            print(f"Result: {metrics}")

        except Exception as e:

import logging

logger = logging.getLogger(__name__)

            print(f"Trial failed: {e}")
            ax_client.log_trial_failure(trial_index=trial_index)

    # Get best parameters
    best_parameters, metrics = ax_client.get_best_parameters()
    print("\n" + "=" * 60)
    print("BEST PARAMETERS:")
    for k, v in best_parameters.items():
        print(f"  {k}: {v}")
    print(f"\nBest {config['experiment']['objective']}: {metrics}")

    # Save results
    ax_client.save_to_json_file("ax_sweep_results.json")
    print("\nResults saved to ax_sweep_results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ax/BoTorch hyperparameter sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ax_sweep_config.yaml",
        help="Path to sweep config",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Total number of trials",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Max parallel trials (set to num GPUs)",
    )

    args = parser.parse_args()
    run_sweep(args.config, args.trials, args.parallel)


if __name__ == "__main__":
    main()
