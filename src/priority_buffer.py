"""Prioritized Experience Replay buffer."""
import numpy as np
from src.replay_buffer import ReplayBuffer

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        super().__init__(capacity)
        self.priorities = np.zeros(capacity)
        self.alpha = alpha
        self.pos = 0

    def push(self, state, action, reward, next_state, done) -> None:
        max_priority = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.buffer.maxlen
