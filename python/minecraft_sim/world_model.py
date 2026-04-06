"""World model training support for DreamerV3-style algorithms.

This module provides infrastructure for training world models (dynamics models)
on Minecraft environment transitions. Key components:

- TransitionBuffer: Efficient circular buffer for storing (s, a, r, d, s') tuples
- WorldModelDataLoader: Batch sampling with sequence chunking for RNN training
- LatentStateExtractor: Helpers for encoding observations to latent space
- DreamEnv: Wrapper for imagination rollouts using a learned world model

Compatible with:
- DreamerV3 (Hafner et al.)
- IRIS (Micheli et al.)
- TWM (Robine et al.)
- Custom world model architectures

Example:
    >>> from minecraft_sim.world_model import TransitionBuffer, WorldModelDataLoader
    >>> buffer = TransitionBuffer(capacity=1_000_000, obs_shape=(48,))
    >>> # Collect experience
    >>> for _ in range(10000):
    ...     buffer.add(obs, action, reward, done, next_obs)
    >>> # Train world model
    >>> loader = WorldModelDataLoader(buffer, batch_size=50, seq_len=50)
    >>> for batch in loader:
    ...     loss = world_model.train_step(batch)
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Type definitions
# =============================================================================

ObsType = NDArray[np.float32]
ActionType = NDArray[np.int32] | NDArray[np.float32]

T = TypeVar("T")


class WorldModelProtocol(Protocol):
    """Protocol for world model implementations."""

    def encode(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Encode observations to latent states."""
        ...

    def decode(self, latent: NDArray[np.float32]) -> NDArray[np.float32]:
        """Decode latent states to observation predictions."""
        ...

    def predict(
        self,
        latent: NDArray[np.float32],
        action: NDArray[np.int32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Predict next latent, reward, and done probability."""
        ...


@dataclass
class TransitionBatch:
    """A batch of transition sequences for world model training.

    Attributes:
        observations: Shape (batch, seq_len, obs_dim)
        actions: Shape (batch, seq_len)
        rewards: Shape (batch, seq_len)
        dones: Shape (batch, seq_len)
        next_observations: Shape (batch, seq_len, obs_dim)
        masks: Shape (batch, seq_len), valid timesteps mask
    """

    observations: NDArray[np.float32]
    actions: NDArray[np.int32]
    rewards: NDArray[np.float32]
    dones: NDArray[np.bool_]
    next_observations: NDArray[np.float32]
    masks: NDArray[np.bool_]

    def to_dict(self) -> dict[str, NDArray[Any]]:
        """Convert to dictionary for framework compatibility."""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "next_observations": self.next_observations,
            "masks": self.masks,
        }

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return self.observations.shape[0]

    @property
    def seq_len(self) -> int:
        """Return sequence length."""
        return self.observations.shape[1]


# =============================================================================
# Transition Buffer
# =============================================================================


class TransitionBuffer:
    """Efficient circular buffer for storing environment transitions.

    Stores (observation, action, reward, done, next_observation) tuples in
    contiguous memory for efficient batch sampling. Supports multi-environment
    collection with episode boundary tracking.

    Args:
        capacity: Maximum number of transitions to store.
        obs_shape: Shape of observations (excluding batch dimension).
        action_dtype: Numpy dtype for actions (int32 for discrete, float32 for continuous).

    Attributes:
        capacity: Maximum buffer capacity.
        size: Current number of stored transitions.
        obs_shape: Shape of individual observations.

    Example:
        >>> buffer = TransitionBuffer(capacity=100_000, obs_shape=(48,))
        >>> for step in range(1000):
        ...     action = policy(obs)
        ...     next_obs, reward, done, _, _ = env.step(action)
        ...     buffer.add(obs, action, reward, done, next_obs)
        ...     obs = next_obs if not done else env.reset()[0]
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_dtype: np.dtype[Any] = np.dtype(np.int32),
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dtype = action_dtype

        # Pre-allocate contiguous arrays
        self._observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=action_dtype)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)
        self._next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)

        # Episode tracking for sequence sampling
        self._episode_starts: list[int] = [0]
        self._episode_lengths: list[int] = []

        self._idx = 0
        self._size = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self._size

    def add(
        self,
        obs: NDArray[np.float32],
        action: int | NDArray[Any],
        reward: float,
        done: bool,
        next_obs: NDArray[np.float32],
    ) -> None:
        """Add a single transition to the buffer.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            done: Whether episode ended.
            next_obs: Resulting observation.
        """
        with self._lock:
            self._observations[self._idx] = obs
            self._actions[self._idx] = action
            self._rewards[self._idx] = reward
            self._dones[self._idx] = done
            self._next_observations[self._idx] = next_obs

            self._idx = (self._idx + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

            # Track episode boundaries
            if done:
                if self._episode_starts:
                    ep_start = self._episode_starts[-1]
                    ep_len = (self._idx - ep_start) % self.capacity
                    if ep_len <= 0:
                        ep_len = self.capacity + ep_len
                    self._episode_lengths.append(ep_len)
                self._episode_starts.append(self._idx)

    def add_batch(
        self,
        obs: NDArray[np.float32],
        actions: NDArray[Any],
        rewards: NDArray[np.float32],
        dones: NDArray[np.bool_],
        next_obs: NDArray[np.float32],
    ) -> None:
        """Add a batch of transitions to the buffer.

        Args:
            obs: Observations, shape (batch, *obs_shape).
            actions: Actions, shape (batch,) or (batch, action_dim).
            rewards: Rewards, shape (batch,).
            dones: Done flags, shape (batch,).
            next_obs: Next observations, shape (batch, *obs_shape).
        """
        batch_size = obs.shape[0]
        with self._lock:
            for i in range(batch_size):
                self._observations[self._idx] = obs[i]
                self._actions[self._idx] = actions[i]
                self._rewards[self._idx] = rewards[i]
                self._dones[self._idx] = dones[i]
                self._next_observations[self._idx] = next_obs[i]

                old_idx = self._idx
                self._idx = (self._idx + 1) % self.capacity
                self._size = min(self._size + 1, self.capacity)

                if dones[i]:
                    if self._episode_starts:
                        ep_start = self._episode_starts[-1]
                        ep_len = (old_idx - ep_start + 1) % self.capacity
                        if ep_len <= 0:
                            ep_len = self.capacity + ep_len
                        self._episode_lengths.append(ep_len)
                    self._episode_starts.append(self._idx)

    def sample(
        self,
        batch_size: int,
        seq_len: int = 1,
    ) -> TransitionBatch:
        """Sample a batch of transition sequences.

        Args:
            batch_size: Number of sequences to sample.
            seq_len: Length of each sequence (for RNN training).

        Returns:
            TransitionBatch containing sampled sequences.

        Raises:
            ValueError: If buffer has fewer transitions than required.
        """
        if self._size < seq_len:
            raise ValueError(f"Buffer has {self._size} transitions, need at least {seq_len}")

        # Sample starting indices ensuring we don't cross episode boundaries
        valid_starts = self._get_valid_sequence_starts(seq_len)
        if len(valid_starts) < batch_size:
            warnings.warn(
                f"Only {len(valid_starts)} valid sequences available, sampling with replacement",
                stacklevel=2,
            )
            starts = np.random.choice(valid_starts, size=batch_size, replace=True)
        else:
            starts = np.random.choice(valid_starts, size=batch_size, replace=False)

        # Extract sequences
        obs_batch = np.zeros((batch_size, seq_len, *self.obs_shape), dtype=np.float32)
        action_batch = np.zeros((batch_size, seq_len), dtype=self.action_dtype)
        reward_batch = np.zeros((batch_size, seq_len), dtype=np.float32)
        done_batch = np.zeros((batch_size, seq_len), dtype=np.bool_)
        next_obs_batch = np.zeros((batch_size, seq_len, *self.obs_shape), dtype=np.float32)
        mask_batch = np.ones((batch_size, seq_len), dtype=np.bool_)

        for b, start in enumerate(starts):
            for t in range(seq_len):
                idx = (start + t) % self.capacity
                obs_batch[b, t] = self._observations[idx]
                action_batch[b, t] = self._actions[idx]
                reward_batch[b, t] = self._rewards[idx]
                done_batch[b, t] = self._dones[idx]
                next_obs_batch[b, t] = self._next_observations[idx]

                # Mask out transitions after episode end
                if t > 0 and done_batch[b, t - 1]:
                    mask_batch[b, t:] = False
                    break

        return TransitionBatch(
            observations=obs_batch,
            actions=action_batch,
            rewards=reward_batch,
            dones=done_batch,
            next_observations=next_obs_batch,
            masks=mask_batch,
        )

    def _get_valid_sequence_starts(self, seq_len: int) -> NDArray[np.int64]:
        """Get valid starting indices for sequences that don't cross buffer boundaries."""
        if self._size == self.capacity:
            # Buffer is full, avoid wraparound region
            invalid_region_start = (self._idx - seq_len + 1) % self.capacity
            invalid_region_end = self._idx

            if invalid_region_start < invalid_region_end:
                valid = np.concatenate(
                    [
                        np.arange(0, invalid_region_start),
                        np.arange(invalid_region_end, self.capacity),
                    ]
                )
            else:
                valid = np.arange(invalid_region_end, invalid_region_start)
        else:
            # Buffer not full, only valid up to current size minus seq_len
            valid = np.arange(0, max(0, self._size - seq_len + 1))

        return valid

    def clear(self) -> None:
        """Clear all stored transitions."""
        with self._lock:
            self._idx = 0
            self._size = 0
            self._episode_starts = [0]
            self._episode_lengths = []

    def get_statistics(self) -> dict[str, Any]:
        """Get buffer statistics."""
        return {
            "size": self._size,
            "capacity": self.capacity,
            "utilization": self._size / self.capacity,
            "num_episodes": len(self._episode_lengths),
            "mean_episode_length": (np.mean(self._episode_lengths) if self._episode_lengths else 0),
            "mean_reward": float(np.mean(self._rewards[: self._size])) if self._size > 0 else 0,
        }


# =============================================================================
# Data Loader
# =============================================================================


class WorldModelDataLoader:
    """Batch data loader for world model training.

    Provides an iterator interface for sampling batches of transition sequences
    from a TransitionBuffer. Supports configurable sequence lengths for
    recurrent world models.

    Args:
        buffer: TransitionBuffer to sample from.
        batch_size: Number of sequences per batch.
        seq_len: Length of each sequence.
        num_batches: Number of batches per epoch (None for infinite).

    Example:
        >>> loader = WorldModelDataLoader(buffer, batch_size=50, seq_len=50)
        >>> for epoch in range(100):
        ...     for batch in loader:
        ...         loss = model.train_step(batch.observations, batch.actions, ...)
    """

    def __init__(
        self,
        buffer: TransitionBuffer,
        batch_size: int = 50,
        seq_len: int = 50,
        num_batches: int | None = None,
    ) -> None:
        self.buffer = buffer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches

    def __iter__(self) -> WorldModelDataLoader:
        self._batch_count = 0
        return self

    def __next__(self) -> TransitionBatch:
        if self.num_batches is not None and self._batch_count >= self.num_batches:
            raise StopIteration

        if self.buffer.size < self.seq_len:
            raise StopIteration

        self._batch_count += 1
        return self.buffer.sample(self.batch_size, self.seq_len)

    def __len__(self) -> int:
        if self.num_batches is not None:
            return self.num_batches
        # Estimate based on buffer size
        return max(1, self.buffer.size // (self.batch_size * self.seq_len))


# =============================================================================
# Latent State Extraction
# =============================================================================


@dataclass
class LatentState:
    """Container for latent state representation.

    Attributes:
        deterministic: Deterministic state (e.g., RNN hidden state).
        stochastic: Stochastic state (e.g., sampled latent).
        mean: Mean of stochastic distribution (for KL computation).
        std: Std of stochastic distribution.
    """

    deterministic: NDArray[np.float32]
    stochastic: NDArray[np.float32]
    mean: NDArray[np.float32] | None = None
    std: NDArray[np.float32] | None = None

    @property
    def combined(self) -> NDArray[np.float32]:
        """Concatenated deterministic and stochastic states."""
        return np.concatenate([self.deterministic, self.stochastic], axis=-1)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of combined state."""
        return self.combined.shape


class LatentStateExtractor(Generic[T]):
    """Helper class for extracting latent states from observations.

    Wraps a world model encoder to provide convenient methods for
    encoding observations, computing prior/posterior distributions,
    and extracting features for actor-critic training.

    Args:
        world_model: World model implementing encode/decode/predict.
        deterministic_size: Size of deterministic state.
        stochastic_size: Size of stochastic state.

    Example:
        >>> extractor = LatentStateExtractor(world_model, det_size=512, stoch_size=32)
        >>> latent = extractor.encode(obs)
        >>> features = latent.combined  # Use for actor-critic
    """

    def __init__(
        self,
        world_model: T,
        deterministic_size: int = 512,
        stochastic_size: int = 32,
    ) -> None:
        self.world_model = world_model
        self.deterministic_size = deterministic_size
        self.stochastic_size = stochastic_size

    def initial_state(self, batch_size: int) -> LatentState:
        """Create initial latent state (zeros).

        Args:
            batch_size: Number of parallel states.

        Returns:
            Zero-initialized LatentState.
        """
        return LatentState(
            deterministic=np.zeros((batch_size, self.deterministic_size), dtype=np.float32),
            stochastic=np.zeros((batch_size, self.stochastic_size), dtype=np.float32),
            mean=np.zeros((batch_size, self.stochastic_size), dtype=np.float32),
            std=np.ones((batch_size, self.stochastic_size), dtype=np.float32),
        )

    def encode(
        self,
        obs: NDArray[np.float32],
        prev_state: LatentState | None = None,
        prev_action: NDArray[np.int32] | None = None,
    ) -> LatentState:
        """Encode observations to latent state (posterior).

        Args:
            obs: Observations, shape (batch, obs_dim).
            prev_state: Previous latent state (optional).
            prev_action: Previous action (optional).

        Returns:
            Posterior LatentState.
        """
        batch_size = obs.shape[0]

        if prev_state is None:
            prev_state = self.initial_state(batch_size)

        # Call world model encoder
        if hasattr(self.world_model, "encode"):
            encoded = self.world_model.encode(obs)
            # Assume encoder returns concatenated [deterministic, stochastic]
            det = encoded[..., : self.deterministic_size]
            stoch = encoded[..., self.deterministic_size :]
            return LatentState(
                deterministic=det,
                stochastic=stoch,
            )

        # Fallback: return observation as "latent" state
        return LatentState(
            deterministic=obs[..., : self.deterministic_size]
            if obs.shape[-1] >= self.deterministic_size
            else np.zeros((batch_size, self.deterministic_size), dtype=np.float32),
            stochastic=np.zeros((batch_size, self.stochastic_size), dtype=np.float32),
        )

    def imagine_step(
        self,
        state: LatentState,
        action: NDArray[np.int32],
    ) -> tuple[LatentState, NDArray[np.float32], NDArray[np.float32]]:
        """Predict next state using world model (prior/imagination).

        Args:
            state: Current latent state.
            action: Action to take.

        Returns:
            Tuple of (next_state, predicted_reward, predicted_done).
        """
        if hasattr(self.world_model, "predict"):
            next_latent, reward, done = self.world_model.predict(state.combined, action)
            next_det = next_latent[..., : self.deterministic_size]
            next_stoch = next_latent[..., self.deterministic_size :]
            next_state = LatentState(
                deterministic=next_det,
                stochastic=next_stoch,
            )
            return next_state, reward, done

        # Fallback: return same state
        batch_size = state.deterministic.shape[0]
        return (
            state,
            np.zeros(batch_size, dtype=np.float32),
            np.zeros(batch_size, dtype=np.float32),
        )


# =============================================================================
# Dream Environment
# =============================================================================


class DreamEnv:
    """Environment wrapper for imagination rollouts using a learned world model.

    Instead of stepping the true environment, DreamEnv uses the world model
    to predict next states, rewards, and terminations. This enables
    training actor-critic on imagined trajectories (Dreamer-style).

    Args:
        world_model: Trained world model for predictions.
        extractor: LatentStateExtractor for encoding/decoding.
        horizon: Maximum imagination horizon.
        num_envs: Number of parallel imagination trajectories.

    Attributes:
        num_envs: Number of parallel environments.
        horizon: Maximum rollout length.

    Example:
        >>> dream_env = DreamEnv(world_model, extractor, horizon=15, num_envs=64)
        >>> # Start imagination from real observations
        >>> dream_obs = dream_env.reset(real_observations)
        >>> for t in range(15):
        ...     actions = actor(dream_obs)
        ...     dream_obs, rewards, dones, _ = dream_env.step(actions)
        ...     # Accumulate imagined returns for actor-critic training
    """

    def __init__(
        self,
        world_model: WorldModelProtocol,
        extractor: LatentStateExtractor[Any],
        horizon: int = 15,
        num_envs: int = 64,
    ) -> None:
        self.world_model = world_model
        self.extractor = extractor
        self.horizon = horizon
        self.num_envs = num_envs

        self._state: LatentState | None = None
        self._step_count = 0

        # Trajectory storage for lambda-return computation
        self._latent_trajectory: list[LatentState] = []
        self._action_trajectory: list[NDArray[np.int32]] = []
        self._reward_trajectory: list[NDArray[np.float32]] = []
        self._done_trajectory: list[NDArray[np.float32]] = []

    def reset(
        self,
        initial_obs: NDArray[np.float32] | None = None,
        initial_state: LatentState | None = None,
    ) -> NDArray[np.float32]:
        """Reset imagination from real observations or latent states.

        Args:
            initial_obs: Real observations to encode, shape (num_envs, obs_dim).
            initial_state: Pre-computed latent state (alternative to obs).

        Returns:
            Decoded observation from initial latent state.
        """
        self._step_count = 0
        self._latent_trajectory = []
        self._action_trajectory = []
        self._reward_trajectory = []
        self._done_trajectory = []

        if initial_state is not None:
            self._state = initial_state
        elif initial_obs is not None:
            self._state = self.extractor.encode(initial_obs)
        else:
            self._state = self.extractor.initial_state(self.num_envs)

        self._latent_trajectory.append(self._state)

        # Decode latent to observation space
        if hasattr(self.world_model, "decode"):
            return self.world_model.decode(self._state.combined)
        return self._state.combined

    def step(
        self,
        actions: NDArray[np.int32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], dict[str, Any]]:
        """Step imagination forward using world model predictions.

        Args:
            actions: Actions to imagine taking, shape (num_envs,).

        Returns:
            Tuple of (observations, rewards, dones, info).
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        self._step_count += 1

        # Predict next state
        next_state, rewards, dones = self.extractor.imagine_step(self._state, actions)

        # Store trajectory
        self._action_trajectory.append(actions)
        self._reward_trajectory.append(rewards)
        self._done_trajectory.append(dones)
        self._latent_trajectory.append(next_state)

        self._state = next_state

        # Decode to observation space
        if hasattr(self.world_model, "decode"):
            obs = self.world_model.decode(next_state.combined)
        else:
            obs = next_state.combined

        # Convert done logits to probabilities
        done_probs = 1.0 / (1.0 + np.exp(-dones))  # sigmoid
        done_flags = done_probs > 0.5

        info: dict[str, Any] = {
            "imagination_step": self._step_count,
            "latent_state": next_state,
        }

        return obs, rewards, done_flags, info

    def get_trajectory(self) -> dict[str, NDArray[Any]]:
        """Get the full imagined trajectory for training.

        Returns:
            Dictionary containing:
                - latent_states: (horizon+1, num_envs, latent_dim)
                - actions: (horizon, num_envs)
                - rewards: (horizon, num_envs)
                - dones: (horizon, num_envs)
        """
        if not self._latent_trajectory:
            raise RuntimeError("No trajectory available. Call reset() and step() first.")

        return {
            "latent_states": np.stack([s.combined for s in self._latent_trajectory], axis=0),
            "actions": np.stack(self._action_trajectory, axis=0)
            if self._action_trajectory
            else np.array([]),
            "rewards": np.stack(self._reward_trajectory, axis=0)
            if self._reward_trajectory
            else np.array([]),
            "dones": np.stack(self._done_trajectory, axis=0)
            if self._done_trajectory
            else np.array([]),
        }

    def compute_lambda_returns(
        self,
        values: NDArray[np.float32],
        gamma: float = 0.99,
        lambda_: float = 0.95,
    ) -> NDArray[np.float32]:
        """Compute lambda-returns (GAE targets) for the imagined trajectory.

        Args:
            values: Value estimates at each step, shape (horizon+1, num_envs).
            gamma: Discount factor.
            lambda_: GAE lambda parameter.

        Returns:
            Lambda-returns, shape (horizon, num_envs).
        """
        if not self._reward_trajectory:
            raise RuntimeError("No trajectory available.")

        horizon = len(self._reward_trajectory)
        rewards = np.stack(self._reward_trajectory, axis=0)
        dones = np.stack(self._done_trajectory, axis=0)

        # Convert done logits to continuation probabilities
        continues = 1.0 - (1.0 / (1.0 + np.exp(-dones)))

        # Compute lambda-returns backward
        returns = np.zeros((horizon, self.num_envs), dtype=np.float32)
        last_value = values[-1]

        for t in reversed(range(horizon)):
            next_value = values[t + 1] if t + 1 < horizon else last_value
            delta = rewards[t] + gamma * continues[t] * next_value - values[t]
            last_gae = delta + gamma * lambda_ * continues[t] * (
                returns[t + 1] if t + 1 < horizon else 0
            )
            returns[t] = last_gae + values[t]

        return returns

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of observations."""
        return (self.extractor.deterministic_size + self.extractor.stochastic_size,)


# =============================================================================
# Convenience factory functions
# =============================================================================


def make_buffer(
    capacity: int = 1_000_000,
    obs_shape: tuple[int, ...] = (48,),
    discrete_actions: bool = True,
) -> TransitionBuffer:
    """Create a transition buffer for Minecraft world model training.

    Args:
        capacity: Maximum transitions to store.
        obs_shape: Observation shape (default: Minecraft's 48-dim obs).
        discrete_actions: If True, use int32 actions; else float32.

    Returns:
        Configured TransitionBuffer.
    """
    action_dtype = np.dtype(np.int32) if discrete_actions else np.dtype(np.float32)
    return TransitionBuffer(capacity, obs_shape, action_dtype)


def make_data_loader(
    buffer: TransitionBuffer,
    batch_size: int = 50,
    seq_len: int = 50,
    batches_per_epoch: int = 100,
) -> WorldModelDataLoader:
    """Create a data loader for world model training.

    Args:
        buffer: TransitionBuffer to sample from.
        batch_size: Sequences per batch.
        seq_len: Sequence length (DreamerV3 uses 50).
        batches_per_epoch: Batches per training epoch.

    Returns:
        Configured WorldModelDataLoader.
    """
    return WorldModelDataLoader(
        buffer,
        batch_size=batch_size,
        seq_len=seq_len,
        num_batches=batches_per_epoch,
    )


def collect_transitions(
    env: Any,
    policy: Any,
    buffer: TransitionBuffer,
    num_steps: int,
    render: bool = False,
) -> dict[str, float]:
    """Collect transitions from environment into buffer.

    Args:
        env: Gymnasium-compatible environment.
        policy: Policy with __call__(obs) -> action method.
        buffer: TransitionBuffer to fill.
        num_steps: Number of steps to collect.
        render: Whether to render environment.

    Returns:
        Collection statistics.
    """
    obs, _ = env.reset()
    episode_rewards: list[float] = []
    current_episode_reward = 0.0
    steps_collected = 0

    for _ in range(num_steps):
        # Get action from policy
        if callable(policy):
            action = policy(obs)
        elif hasattr(policy, "act"):
            action = policy.act(obs)
        else:
            action = env.action_space.sample()

        # Ensure action is numpy array or int
        if hasattr(action, "item"):
            action = action.item()

        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition
        buffer.add(obs, action, float(reward), done, next_obs)
        steps_collected += 1
        current_episode_reward += reward

        if render:
            env.render()

        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

    return {
        "steps_collected": steps_collected,
        "episodes_completed": len(episode_rewards),
        "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
        "buffer_size": buffer.size,
    }


def collect_vec_transitions(
    vec_env: Any,
    policy: Any,
    buffer: TransitionBuffer,
    num_steps: int,
) -> dict[str, float]:
    """Collect transitions from vectorized environment into buffer.

    Args:
        vec_env: Vectorized environment (SB3-style or VecDragonFightEnv).
        policy: Policy with __call__(obs) -> actions method.
        buffer: TransitionBuffer to fill.
        num_steps: Number of steps to collect (per environment).

    Returns:
        Collection statistics.
    """
    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]  # Handle (obs, info) return

    num_envs = obs.shape[0]
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    completed_rewards: list[float] = []
    steps_collected = 0

    for _ in range(num_steps):
        # Get actions
        if callable(policy):
            actions = policy(obs)
        elif hasattr(policy, "act"):
            actions = policy.act(obs)
        else:
            actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])

        # Step environment
        result = vec_env.step(actions)
        if len(result) == 5:
            next_obs, rewards, terminated, truncated, infos = result
            dones = np.logical_or(terminated, truncated)
        else:
            next_obs, rewards, dones, infos = result

        # Store transitions
        buffer.add_batch(obs, actions, rewards, dones, next_obs)
        steps_collected += num_envs

        episode_rewards += rewards
        for i, done in enumerate(dones):
            if done:
                completed_rewards.append(episode_rewards[i])
                episode_rewards[i] = 0.0

        obs = next_obs

    return {
        "steps_collected": steps_collected,
        "episodes_completed": len(completed_rewards),
        "mean_episode_reward": np.mean(completed_rewards) if completed_rewards else 0.0,
        "buffer_size": buffer.size,
    }
