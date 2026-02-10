"""Training metrics and episode statistics."""
from __future__ import annotations
from typing import List
import numpy as np


class EpisodeStats:
    """Track episode rewards and compute rolling statistics."""

    def __init__(self, window: int = 100) -> None:
        self.rewards: List[float] = []
        self.window = window

    def record(self, reward: float) -> None:
        self.rewards.append(reward)

    def rolling_mean(self) -> float:
        if not self.rewards:
            return 0.0
        return float(np.mean(self.rewards[-self.window:]))

    def is_solved(self, threshold: float = 200.0) -> bool:
        return len(self.rewards) >= self.window and self.rolling_mean() >= threshold

    def best(self) -> float:
        return max(self.rewards) if self.rewards else float("-inf")
