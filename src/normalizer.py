"""State normalisation for stable RL training."""
import numpy as np

class RunningNormalizer:
    def __init__(self, shape: tuple, epsilon: float = 1e-8) -> None:
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 0
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += delta * (x - self.mean)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var / max(self.count, 1) + self.epsilon)
        return (x - self.mean) / std
