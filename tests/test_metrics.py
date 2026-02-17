import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import EpisodeStats

def test_rolling_mean_empty():
    stats = EpisodeStats()
    assert stats.rolling_mean() == 0.0

def test_rolling_mean_correct():
    stats = EpisodeStats(window=3)
    for r in [100, 200, 300]:
        stats.record(r)
    assert stats.rolling_mean() == 200.0

def test_is_solved_false_insufficient_data():
    stats = EpisodeStats(window=100)
    for _ in range(50):
        stats.record(250.0)
    assert not stats.is_solved()

def test_is_solved_true():
    stats = EpisodeStats(window=10)
    for _ in range(10):
        stats.record(250.0)
    assert stats.is_solved(threshold=200.0)

def test_best_reward():
    stats = EpisodeStats()
    for r in [50, 200, 150]:
        stats.record(r)
    assert stats.best() == 200.0
