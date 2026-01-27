import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.replay_buffer import ReplayBuffer
import numpy as np

def test_buffer_push_and_len():
    buf = ReplayBuffer(capacity=100)
    buf.push(np.zeros(8), 0, 1.0, np.zeros(8), False)
    assert len(buf) == 1

def test_buffer_respects_capacity():
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
        buf.push(np.zeros(8), 0, float(i), np.zeros(8), False)
    assert len(buf) == 5

def test_buffer_sample_returns_correct_shapes():
    buf = ReplayBuffer(capacity=100)
    for _ in range(20):
        buf.push(np.random.rand(8), 0, 1.0, np.random.rand(8), False)
    states, actions, rewards, next_states, dones = buf.sample(10)
    assert states.shape == (10, 8)
    assert actions.shape == (10,)
    assert rewards.shape == (10,)
