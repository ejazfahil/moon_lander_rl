"""Hyperparameter configuration for PPO training.

Algorithm follows Schulman et al. (2017), "Proximal Policy Optimization
Algorithms" (arXiv:1707.06347), with implementation choices corroborated by
Huang et al. (2022), "The 37 Implementation Details of Proximal Policy
Optimization" (ICLR Blog Track) — GAE(lambda) advantage estimation,
advantage normalization, value-loss clipping, a global gradient norm clip,
and linear learning-rate annealing.

Defaults (num_envs, num_steps, gamma, gae_lambda, ent_coef, update_epochs)
are the tuned values published for LunarLander-v3 in RL Baselines3 Zoo
(DLR-RM/rl-baselines3-zoo, hyperparams/ppo.yml, verified against the raw
file on 2026-08-11): n_envs=16, n_steps=1024, batch_size=64, n_epochs=4,
gamma=0.999, gae_lambda=0.98, ent_coef=0.01, n_timesteps=1e6. An initial
run with ad-hoc defaults (gamma=0.99, gae_lambda=0.95, n_steps=128,
n_envs=8) plateaued around avg(100)=+18 without solving; the higher gamma
and much larger rollout/update-batch size are the load-bearing differences
for this specific environment (LunarLander episodes run long, so credit
assignment needs a discount factor closer to 1, and PPO's on-policy update
is noisier with small batches).
"""
from dataclasses import dataclass


@dataclass
class PPOConfig:
    # Environment
    env_name: str = "LunarLander-v3"
    seed: int = 42
    num_envs: int = 16

    # Rollout / batch shape
    num_steps: int = 1024           # steps collected per env, per update
    minibatch_size_target: int = 64  # used to derive num_minibatches
    update_epochs: int = 4

    # Network
    hidden_dims: tuple = (64, 64)

    # Optimization
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    max_grad_norm: float = 0.5

    # PPO objective
    gamma: float = 0.999
    gae_lambda: float = 0.98
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    normalize_advantages: bool = True

    # Training length
    total_timesteps: int = 1_000_000
    solve_score: float = 200.0

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def num_minibatches(self) -> int:
        return max(1, self.batch_size // self.minibatch_size_target)

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches
