"""Hyperparameter configuration for DQN training."""
from dataclasses import dataclass


@dataclass
class DQNConfig:
    # Environment
    env_name: str = "LunarLander-v2"
    seed: int = 42
    # Network
    hidden_dims: tuple = (256, 256)
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005          # Soft update coefficient
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_capacity: int = 100_000
    min_buffer_size: int = 1_000
    target_update_freq: int = 10
    max_episodes: int = 2_000
    solve_score: float = 200.0   # LunarLander-v2 solve threshold
