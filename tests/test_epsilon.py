import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.epsilon_scheduler import ExponentialDecay, LinearDecay

def test_exponential_decay_decreases():
    sched = ExponentialDecay(1.0, 0.01, 0.99)
    prev = sched.value
    for _ in range(100):
        val = sched.step()
        assert val <= prev or val == sched.end
        prev = val

def test_exponential_decay_respects_minimum():
    sched = ExponentialDecay(1.0, 0.05, 0.5)
    for _ in range(1000):
        sched.step()
    assert sched.value >= 0.05

def test_linear_decay_reaches_end():
    sched = LinearDecay(1.0, 0.1, 10)
    for _ in range(10):
        val = sched.step()
    assert val == 1.0 + 1.0 * (0.1 - 1.0)  # end value after steps
