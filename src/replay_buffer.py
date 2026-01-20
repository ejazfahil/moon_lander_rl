"""Experience replay buffer for DQN training."""
from __future__ import annotations
from collections import deque
from typing import Tuple, List
import random
import numpy as np


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
