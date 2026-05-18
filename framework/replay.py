"""Replay buffers for off-policy RL algorithms.

ReplayBuffer        — circular buffer of (obs, action_idx, reward, next_obs, done)
MaskedReplayBuffer  — adds a per-transition available-actions boolean mask (used by
                      DQN with action masking, e.g. SC2NeuralDQNPolicy)
"""
from __future__ import annotations

from collections import deque

import numpy as np


class ReplayBuffer:
    """Fixed-size circular buffer of (obs, action_idx, reward, next_obs, done) transitions."""

    def __init__(self, max_size: int) -> None:
        self._buf: deque = deque(maxlen=int(max_size))

    def add(
        self,
        obs: np.ndarray,
        action_idx: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((
            obs.copy(), int(action_idx), float(reward),
            next_obs.copy(), bool(done),
        ))

    # Backward-compat alias — legacy callers use push().
    def push(self, obs, action_idx, reward, next_obs, done) -> None:
        self.add(obs, action_idx, reward, next_obs, done)

    def sample(
        self,
        batch_size: int,
        rng=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        replace = batch_size > len(self._buf)
        if rng is not None:
            idxs = rng.choice(len(self._buf), size=batch_size, replace=replace)
        else:
            idxs = np.random.choice(len(self._buf), size=batch_size, replace=replace)
        batch  = [self._buf[i] for i in idxs]
        obs_b  = np.stack([t[0] for t in batch]).astype(np.float32)
        act_b  = np.array([t[1] for t in batch], dtype=np.int32)
        rew_b  = np.array([t[2] for t in batch], dtype=np.float32)
        next_b = np.stack([t[3] for t in batch]).astype(np.float32)
        done_b = np.array([t[4] for t in batch], dtype=np.float32)
        return obs_b, act_b, rew_b, next_b, done_b

    def __len__(self) -> int:
        return len(self._buf)


class MaskedReplayBuffer(ReplayBuffer):
    """Replay buffer that stores a per-transition available-actions mask.

    Each stored tuple is (obs, action_idx, reward, next_obs, done, mask) where
    *mask* is a boolean array of shape ``(n_actions,)`` indicating which discrete
    actions were legal in the **next** state ``s'``.  Used by
    :class:`~framework.dqn.DQNPolicy` with action masking so the target network
    never bootstraps through Q-values of unavailable actions.
    """

    def add(  # type: ignore[override]
        self,
        obs: np.ndarray,
        action_idx: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        available_mask: np.ndarray | None = None,
    ) -> None:
        if available_mask is None:
            available_mask = np.ones(1, dtype=bool)
        self._buf.append((
            obs.copy(), int(action_idx), float(reward),
            next_obs.copy(), bool(done), available_mask.copy(),
        ))

    def push(  # type: ignore[override]
        self, obs, action_idx, reward, next_obs, done, mask=None
    ) -> None:
        self.add(obs, action_idx, reward, next_obs, done, mask)

    def sample(  # type: ignore[override]
        self,
        batch_size: int,
        rng=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        replace = batch_size > len(self._buf)
        if rng is not None:
            idxs = rng.choice(len(self._buf), size=batch_size, replace=replace)
        else:
            idxs = np.random.choice(len(self._buf), size=batch_size, replace=replace)
        batch  = [self._buf[i] for i in idxs]
        obs_b  = np.stack([t[0] for t in batch]).astype(np.float32)
        act_b  = np.array([t[1] for t in batch], dtype=np.int32)
        rew_b  = np.array([t[2] for t in batch], dtype=np.float32)
        next_b = np.stack([t[3] for t in batch]).astype(np.float32)
        done_b = np.array([t[4] for t in batch], dtype=np.float32)
        mask_b = np.stack([t[5] for t in batch])
        return obs_b, act_b, rew_b, next_b, done_b, mask_b
