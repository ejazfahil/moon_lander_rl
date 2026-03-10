"""Epsilon scheduling strategies for exploration-exploitation."""
from __future__ import annotations
import math


class ExponentialDecay:
    def __init__(self, start: float, end: float, decay: float) -> None:
        self.start = start
        self.end = end
        self.decay = decay
        self._value = start

    def step(self) -> float:
        self._value = max(self.end, self._value * self.decay)
        return self._value

    @property
    def value(self) -> float:
        return self._value


class LinearDecay:
    def __init__(self, start: float, end: float, steps: int) -> None:
        self.start = start
        self.end = end
        self.steps = steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        fraction = min(self._step / self.steps, 1.0)
        return self.start + fraction * (self.end - self.start)
