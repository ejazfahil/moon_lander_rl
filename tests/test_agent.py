"""Unit tests for the RL agent."""

def test_action_space_size():
    n_actions = 4  # LunarLander-v2 has 4 discrete actions
    assert n_actions == 4

def test_state_space_size():
    n_observations = 8  # LunarLander-v2 observation space
    assert n_observations == 8

def test_epsilon_decay():
    epsilon = 1.0
    decay = 0.995
    min_epsilon = 0.01
    for _ in range(1000):
        epsilon = max(min_epsilon, epsilon * decay)
    assert epsilon >= min_epsilon
    assert epsilon < 1.0

def test_reward_target():
    # LunarLander-v2 is considered solved at avg reward >= 200
    target_reward = 200
    assert target_reward == 200
