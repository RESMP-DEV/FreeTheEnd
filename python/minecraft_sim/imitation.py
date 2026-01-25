"""
Imitation learning data collection and algorithms for Minecraft RL.

This module provides:
- Expert trajectory recording from human demonstrations
- Standardized data format (observations, actions, rewards)
- Behavioral cloning training and inference
- DAgger (Dataset Aggregation) implementation

Example:
    >>> from minecraft_sim.imitation import DemonstrationRecorder, BehavioralCloning
    >>>
    >>> # Record demonstrations
    >>> recorder = DemonstrationRecorder(env)
    >>> recorder.start_episode(seed=42)
    >>> obs, action, reward = ...  # from human input
    >>> recorder.record_step(obs, action, reward)
    >>> demo = recorder.end_episode()
    >>>
    >>> # Train BC policy
    >>> bc = BehavioralCloning(obs_dim=48, action_dim=17)
    >>> bc.train(demonstrations=[demo])
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence


@dataclass(slots=True)
class Demonstration:
    """A single expert demonstration trajectory.

    Attributes:
        observations: Observation sequence (T, obs_dim).
        actions: Action sequence (T,) for discrete or (T, action_dim) for continuous.
        rewards: Per-step rewards (T,).
        seed: Random seed used for environment reset.
        completion_time: Number of steps to complete task.
        metadata: Additional information (expert_id, timestamp, etc.).
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    seed: int
    completion_time: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate trajectory shapes."""
        if len(self.observations) != len(self.actions):
            raise ValueError(
                f"Observation and action lengths must match: "
                f"{len(self.observations)} vs {len(self.actions)}"
            )
        if len(self.observations) != len(self.rewards):
            raise ValueError(
                f"Observation and reward lengths must match: "
                f"{len(self.observations)} vs {len(self.rewards)}"
            )

    @property
    def length(self) -> int:
        """Number of timesteps in trajectory."""
        return len(self.observations)

    @property
    def total_reward(self) -> float:
        """Sum of all rewards."""
        return float(np.sum(self.rewards))

    @property
    def obs_dim(self) -> int:
        """Observation dimensionality."""
        return self.observations.shape[1] if self.observations.ndim > 1 else 1

    @property
    def action_dim(self) -> int:
        """Action dimensionality (1 for discrete)."""
        return self.actions.shape[1] if self.actions.ndim > 1 else 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "observations": self.observations.tolist(),
            "actions": self.actions.tolist(),
            "rewards": self.rewards.tolist(),
            "seed": self.seed,
            "completion_time": self.completion_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Demonstration:
        """Create from dictionary."""
        return cls(
            observations=np.array(data["observations"], dtype=np.float32),
            actions=np.array(data["actions"]),
            rewards=np.array(data["rewards"], dtype=np.float32),
            seed=data["seed"],
            completion_time=data["completion_time"],
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        """Save demonstration to file.

        Supports .json (human-readable) and .pkl (efficient).
        """
        path = Path(path)
        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> Demonstration:
        """Load demonstration from file."""
        path = Path(path)
        if path.suffix == ".json":
            with open(path) as f:
                return cls.from_dict(json.load(f))
        else:
            with open(path, "rb") as f:
                return pickle.load(f)


@dataclass
class DemonstrationDataset:
    """Collection of demonstrations with batch sampling."""

    demonstrations: list[Demonstration] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.demonstrations)

    def __getitem__(self, idx: int) -> Demonstration:
        return self.demonstrations[idx]

    def __iter__(self) -> Iterator[Demonstration]:
        return iter(self.demonstrations)

    def add(self, demo: Demonstration) -> None:
        """Add a demonstration to the dataset."""
        self.demonstrations.append(demo)

    @property
    def total_transitions(self) -> int:
        """Total number of (obs, action, reward) tuples."""
        return sum(d.length for d in self.demonstrations)

    @property
    def mean_reward(self) -> float:
        """Mean total reward across demonstrations."""
        if not self.demonstrations:
            return 0.0
        return float(np.mean([d.total_reward for d in self.demonstrations]))

    @property
    def mean_length(self) -> float:
        """Mean episode length."""
        if not self.demonstrations:
            return 0.0
        return float(np.mean([d.length for d in self.demonstrations]))

    def get_all_transitions(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate all transitions into flat arrays.

        Returns:
            (observations, actions, rewards) arrays.
        """
        if not self.demonstrations:
            raise ValueError("Dataset is empty")

        observations = np.concatenate([d.observations for d in self.demonstrations])
        actions = np.concatenate([d.actions for d in self.demonstrations])
        rewards = np.concatenate([d.rewards for d in self.demonstrations])

        return observations, actions, rewards

    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            rng: Random number generator.

        Returns:
            (observations, actions, rewards) batch.
        """
        if rng is None:
            rng = np.random.default_rng()

        obs, act, rew = self.get_all_transitions()
        indices = rng.choice(len(obs), size=batch_size, replace=True)

        return obs[indices], act[indices], rew[indices]

    def save(self, path: str | Path) -> None:
        """Save dataset to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.demonstrations, f)

    @classmethod
    def load(cls, path: str | Path) -> DemonstrationDataset:
        """Load dataset from file."""
        with open(path, "rb") as f:
            demonstrations = pickle.load(f)
        return cls(demonstrations=demonstrations)


@runtime_checkable
class Environment(Protocol):
    """Protocol for Gymnasium-like environments."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]: ...

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]: ...


@runtime_checkable
class ActionProvider(Protocol):
    """Protocol for action providers (policies, human input, etc.)."""

    def get_action(self, observation: np.ndarray) -> int | np.ndarray: ...


class DemonstrationRecorder:
    """Records expert demonstrations from environment interactions.

    Usage:
        recorder = DemonstrationRecorder(env)

        # Start episode
        recorder.start_episode(seed=42)

        # Run episode with expert actions
        obs, _ = env.reset(seed=42)
        while not done:
            action = expert_policy(obs)  # or human input
            next_obs, reward, terminated, truncated, info = env.step(action)
            recorder.record_step(obs, action, reward)
            obs = next_obs
            done = terminated or truncated

        # Finish and get demonstration
        demo = recorder.end_episode()
    """

    def __init__(
        self,
        env: Environment,
        expert_id: str = "unknown",
    ) -> None:
        """Initialize recorder.

        Args:
            env: Environment being recorded.
            expert_id: Identifier for the demonstrator.
        """
        self.env = env
        self.expert_id = expert_id

        # Episode state
        self._recording = False
        self._current_seed: int = 0
        self._observations: list[np.ndarray] = []
        self._actions: list[int | np.ndarray] = []
        self._rewards: list[float] = []
        self._episode_start_time: float = 0.0

    def start_episode(self, seed: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
        """Start recording a new episode.

        Args:
            seed: Seed for environment reset.

        Returns:
            Initial observation and info from env.reset().
        """
        if self._recording:
            raise RuntimeError("Already recording an episode. Call end_episode() first.")

        self._recording = True
        self._current_seed = seed
        self._observations = []
        self._actions = []
        self._rewards = []
        self._episode_start_time = time.time()

        return self.env.reset(seed=seed)

    def record_step(
        self,
        observation: np.ndarray,
        action: int | np.ndarray,
        reward: float,
    ) -> None:
        """Record a single step.

        Args:
            observation: Current observation (before action).
            action: Action taken.
            reward: Reward received.
        """
        if not self._recording:
            raise RuntimeError("Not recording. Call start_episode() first.")

        self._observations.append(np.asarray(observation, dtype=np.float32))
        self._actions.append(action)
        self._rewards.append(float(reward))

    def end_episode(
        self,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Demonstration:
        """End recording and return the demonstration.

        Args:
            extra_metadata: Additional metadata to include.

        Returns:
            Recorded demonstration.
        """
        if not self._recording:
            raise RuntimeError("Not recording. Call start_episode() first.")

        if not self._observations:
            raise ValueError("No steps recorded")

        self._recording = False

        # Build arrays
        observations = np.stack(self._observations)
        actions = np.array(self._actions)
        rewards = np.array(self._rewards, dtype=np.float32)

        # Build metadata
        metadata = {
            "expert_id": self.expert_id,
            "timestamp": time.time(),
            "recording_duration_sec": time.time() - self._episode_start_time,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return Demonstration(
            observations=observations,
            actions=actions,
            rewards=rewards,
            seed=self._current_seed,
            completion_time=len(self._observations),
            metadata=metadata,
        )

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def current_length(self) -> int:
        """Number of steps recorded so far."""
        return len(self._observations)


class AutomaticRecorder:
    """Automatically records demonstrations from a policy.

    Useful for collecting demonstrations from trained agents or oracles.
    """

    def __init__(
        self,
        env: Environment,
        policy: ActionProvider,
        expert_id: str = "automatic",
    ) -> None:
        """Initialize automatic recorder.

        Args:
            env: Environment to run in.
            policy: Policy that provides actions.
            expert_id: Identifier for demonstrations.
        """
        self.env = env
        self.policy = policy
        self.recorder = DemonstrationRecorder(env, expert_id=expert_id)

    def collect_episode(
        self,
        seed: int | None = None,
        max_steps: int = 10000,
    ) -> Demonstration:
        """Collect a single demonstration episode.

        Args:
            seed: Random seed for episode.
            max_steps: Maximum steps before truncation.

        Returns:
            Recorded demonstration.
        """
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))

        obs, _ = self.recorder.start_episode(seed=seed)

        for _ in range(max_steps):
            action = self.policy.get_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.recorder.record_step(obs, action, reward)

            obs = next_obs
            if terminated or truncated:
                break

        return self.recorder.end_episode()

    def collect_dataset(
        self,
        num_episodes: int,
        seeds: Sequence[int] | None = None,
        max_steps: int = 10000,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> DemonstrationDataset:
        """Collect multiple demonstration episodes.

        Args:
            num_episodes: Number of episodes to collect.
            seeds: Optional specific seeds for each episode.
            max_steps: Max steps per episode.
            progress_callback: Called with (current, total) after each episode.

        Returns:
            Dataset of demonstrations.
        """
        if seeds is None:
            rng = np.random.default_rng()
            seeds = [int(rng.integers(0, 2**31)) for _ in range(num_episodes)]

        dataset = DemonstrationDataset()

        for i, seed in enumerate(seeds[:num_episodes]):
            demo = self.collect_episode(seed=seed, max_steps=max_steps)
            dataset.add(demo)

            if progress_callback:
                progress_callback(i + 1, num_episodes)

        return dataset


class BehavioralCloning:
    """Behavioral cloning from expert demonstrations.

    Trains a policy to imitate expert actions via supervised learning.
    Supports both neural network policies (via external frameworks) and
    simple k-NN baseline.

    Example:
        >>> bc = BehavioralCloning(obs_dim=48, action_dim=17)
        >>> bc.train(dataset, epochs=100, batch_size=256)
        >>> action = bc.predict(observation)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        discrete_actions: bool = True,
    ) -> None:
        """Initialize behavioral cloning.

        Args:
            obs_dim: Observation dimensionality.
            action_dim: Action dimensionality (num actions if discrete).
            discrete_actions: Whether actions are discrete.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions

        # Training data (populated by train())
        self._train_obs: np.ndarray | None = None
        self._train_actions: np.ndarray | None = None

        # Simple k-NN model (can be replaced with neural network)
        self._k_neighbors = 5
        self._trained = False

    def train(
        self,
        dataset: DemonstrationDataset | Sequence[Demonstration],
        epochs: int = 1,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        validation_split: float = 0.1,
    ) -> dict[str, float]:
        """Train the BC policy.

        Args:
            dataset: Demonstrations to learn from.
            epochs: Number of training epochs (for gradient-based).
            batch_size: Batch size for training.
            learning_rate: Learning rate (for gradient-based).
            validation_split: Fraction of data for validation.

        Returns:
            Training metrics (loss, accuracy, etc.).
        """
        if isinstance(dataset, DemonstrationDataset):
            obs, actions, _ = dataset.get_all_transitions()
        else:
            obs = np.concatenate([d.observations for d in dataset])
            actions = np.concatenate([d.actions for d in dataset])

        # Store for k-NN inference
        self._train_obs = obs.astype(np.float32)
        self._train_actions = actions
        self._trained = True

        # Compute basic metrics
        metrics: dict[str, float] = {
            "num_transitions": float(len(obs)),
            "num_unique_actions": float(len(np.unique(actions))),
        }

        return metrics

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> int | np.ndarray:
        """Predict action for observation.

        Args:
            observation: Current observation.
            deterministic: If True, return most common action among neighbors.

        Returns:
            Predicted action.
        """
        if not self._trained or self._train_obs is None:
            raise RuntimeError("Model not trained. Call train() first.")

        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)

        # k-NN: find nearest training observations
        distances = np.linalg.norm(self._train_obs - obs, axis=1)
        nearest_indices = np.argsort(distances)[: self._k_neighbors]
        nearest_actions = self._train_actions[nearest_indices]

        if self.discrete_actions:
            # Return most common action among neighbors
            if deterministic:
                values, counts = np.unique(nearest_actions, return_counts=True)
                return int(values[np.argmax(counts)])
            else:
                return int(np.random.choice(nearest_actions))
        else:
            return np.mean(nearest_actions, axis=0)

    def predict_batch(
        self,
        observations: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Predict actions for batch of observations.

        Args:
            observations: Batch of observations (N, obs_dim).
            deterministic: If True, return most common actions.

        Returns:
            Predicted actions (N,) or (N, action_dim).
        """
        return np.array([self.predict(obs, deterministic=deterministic) for obs in observations])

    def get_action(self, observation: np.ndarray) -> int | np.ndarray:
        """ActionProvider protocol implementation."""
        return self.predict(observation, deterministic=True)

    def save(self, path: str | Path) -> None:
        """Save trained model."""
        path = Path(path)
        data = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "discrete_actions": self.discrete_actions,
            "k_neighbors": self._k_neighbors,
            "train_obs": self._train_obs,
            "train_actions": self._train_actions,
            "trained": self._trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> BehavioralCloning:
        """Load trained model."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        bc = cls(
            obs_dim=data["obs_dim"],
            action_dim=data["action_dim"],
            discrete_actions=data["discrete_actions"],
        )
        bc._k_neighbors = data["k_neighbors"]
        bc._train_obs = data["train_obs"]
        bc._train_actions = data["train_actions"]
        bc._trained = data["trained"]

        return bc


class DAgger:
    """Dataset Aggregation (DAgger) for iterative imitation learning.

    DAgger improves upon behavioral cloning by iteratively:
    1. Collect trajectories using current policy
    2. Query expert for correct actions on visited states
    3. Aggregate new data with existing dataset
    4. Retrain policy

    This addresses the distribution shift problem in behavioral cloning.

    Reference:
        Ross, Gordon, Bagnell. "A Reduction of Imitation Learning and
        Structured Prediction to No-Regret Online Learning" (2011)

    Example:
        >>> dagger = DAgger(env, expert_policy, obs_dim=48, action_dim=17)
        >>> for i in range(10):
        ...     metrics = dagger.iterate(
        ...         num_episodes=10,
        ...         beta=max(0.0, 1.0 - i * 0.1),  # Decay expert intervention
        ...     )
        ...     print(f"Iteration {i}: {metrics}")
    """

    def __init__(
        self,
        env: Environment,
        expert: ActionProvider,
        obs_dim: int,
        action_dim: int,
        discrete_actions: bool = True,
    ) -> None:
        """Initialize DAgger.

        Args:
            env: Environment to train in.
            expert: Expert policy for labeling.
            obs_dim: Observation dimensionality.
            action_dim: Action dimensionality.
            discrete_actions: Whether actions are discrete.
        """
        self.env = env
        self.expert = expert

        # Initialize learner policy
        self.policy = BehavioralCloning(
            obs_dim=obs_dim,
            action_dim=action_dim,
            discrete_actions=discrete_actions,
        )

        # Aggregated dataset
        self.dataset = DemonstrationDataset()

        # Iteration tracking
        self.iteration = 0

    def iterate(
        self,
        num_episodes: int,
        beta: float = 0.0,
        max_steps: int = 10000,
        epochs: int = 1,
        batch_size: int = 256,
    ) -> dict[str, float]:
        """Run one DAgger iteration.

        Args:
            num_episodes: Number of episodes to collect.
            beta: Probability of using expert action (vs. learned policy).
                  Set to 1.0 for pure expert, 0.0 for pure learned policy.
            max_steps: Maximum steps per episode.
            epochs: Training epochs after data collection.
            batch_size: Batch size for training.

        Returns:
            Iteration metrics.
        """
        self.iteration += 1
        rng = np.random.default_rng()

        # Collect trajectories with mixed policy
        new_demos: list[Demonstration] = []
        total_reward = 0.0
        total_steps = 0
        expert_queries = 0

        for episode_idx in range(num_episodes):
            seed = int(rng.integers(0, 2**31))
            obs, _ = self.env.reset(seed=seed)

            observations: list[np.ndarray] = []
            actions: list[int | np.ndarray] = []
            rewards: list[float] = []

            for step in range(max_steps):
                # Query expert for correct action (for labeling)
                expert_action = self.expert.get_action(obs)
                expert_queries += 1

                # Execute action: expert with prob beta, else learned
                if rng.random() < beta or not self.policy._trained:
                    exec_action = expert_action
                else:
                    exec_action = self.policy.predict(obs)

                next_obs, reward, terminated, truncated, _ = self.env.step(exec_action)

                # Store transition with EXPERT label (key DAgger insight)
                observations.append(np.asarray(obs, dtype=np.float32))
                actions.append(expert_action)  # Expert label, not executed action
                rewards.append(float(reward))

                obs = next_obs
                total_reward += reward
                total_steps += 1

                if terminated or truncated:
                    break

            if observations:
                demo = Demonstration(
                    observations=np.stack(observations),
                    actions=np.array(actions),
                    rewards=np.array(rewards, dtype=np.float32),
                    seed=seed,
                    completion_time=len(observations),
                    metadata={
                        "dagger_iteration": self.iteration,
                        "beta": beta,
                    },
                )
                new_demos.append(demo)
                self.dataset.add(demo)

        # Retrain policy on aggregated dataset
        train_metrics = self.policy.train(
            self.dataset,
            epochs=epochs,
            batch_size=batch_size,
        )

        metrics = {
            "iteration": float(self.iteration),
            "beta": beta,
            "num_episodes": float(num_episodes),
            "total_transitions": float(total_steps),
            "mean_reward": total_reward / max(num_episodes, 1),
            "mean_episode_length": total_steps / max(num_episodes, 1),
            "expert_queries": float(expert_queries),
            "dataset_size": float(self.dataset.total_transitions),
            **train_metrics,
        }

        return metrics

    def train_full(
        self,
        num_iterations: int,
        episodes_per_iter: int = 10,
        initial_beta: float = 1.0,
        final_beta: float = 0.0,
        max_steps: int = 10000,
        progress_callback: Callable[[int, dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        """Run full DAgger training loop.

        Args:
            num_iterations: Number of DAgger iterations.
            episodes_per_iter: Episodes to collect per iteration.
            initial_beta: Starting expert probability.
            final_beta: Final expert probability.
            max_steps: Max steps per episode.
            progress_callback: Called after each iteration with metrics.

        Returns:
            List of per-iteration metrics.
        """
        all_metrics: list[dict[str, float]] = []

        for i in range(num_iterations):
            # Linear decay of beta
            if num_iterations > 1:
                beta = initial_beta + (final_beta - initial_beta) * i / (num_iterations - 1)
            else:
                beta = initial_beta

            metrics = self.iterate(
                num_episodes=episodes_per_iter,
                beta=beta,
                max_steps=max_steps,
            )
            all_metrics.append(metrics)

            if progress_callback:
                progress_callback(i, metrics)

        return all_metrics

    def get_policy(self) -> BehavioralCloning:
        """Get the trained policy."""
        return self.policy


def filter_demonstrations(
    dataset: DemonstrationDataset,
    min_reward: float | None = None,
    max_length: int | None = None,
    min_length: int | None = None,
) -> DemonstrationDataset:
    """Filter demonstrations by criteria.

    Args:
        dataset: Dataset to filter.
        min_reward: Minimum total reward threshold.
        max_length: Maximum episode length.
        min_length: Minimum episode length.

    Returns:
        Filtered dataset.
    """
    filtered = DemonstrationDataset()

    for demo in dataset:
        if min_reward is not None and demo.total_reward < min_reward:
            continue
        if max_length is not None and demo.length > max_length:
            continue
        if min_length is not None and demo.length < min_length:
            continue
        filtered.add(demo)

    return filtered


def augment_demonstrations(
    dataset: DemonstrationDataset,
    noise_std: float = 0.01,
    num_augmented: int = 1,
    rng: np.random.Generator | None = None,
) -> DemonstrationDataset:
    """Augment demonstrations with observation noise.

    Args:
        dataset: Original dataset.
        noise_std: Standard deviation of Gaussian noise.
        num_augmented: Number of augmented copies per original.
        rng: Random number generator.

    Returns:
        Dataset with original + augmented demonstrations.
    """
    if rng is None:
        rng = np.random.default_rng()

    augmented = DemonstrationDataset()

    for demo in dataset:
        # Keep original
        augmented.add(demo)

        # Add noisy copies
        for _ in range(num_augmented):
            noisy_obs = demo.observations + rng.normal(
                0, noise_std, size=demo.observations.shape
            ).astype(np.float32)

            aug_demo = Demonstration(
                observations=noisy_obs,
                actions=demo.actions.copy(),
                rewards=demo.rewards.copy(),
                seed=demo.seed,
                completion_time=demo.completion_time,
                metadata={**demo.metadata, "augmented": True},
            )
            augmented.add(aug_demo)

    return augmented
