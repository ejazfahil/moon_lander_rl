"""PPO (Proximal Policy Optimization) actor-critic agent.

Implements the clipped-surrogate PPO objective of Schulman et al. (2017)
with Generalized Advantage Estimation (Schulman et al., 2016,
arXiv:1506.02438), following the implementation practices documented in
Huang et al. (2022), "The 37 Implementation Details of PPO":
orthogonal init with small policy-head gain, advantage normalization,
value-loss clipping, and global gradient-norm clipping.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.ppo_config import PPOConfig


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Shared-trunk actor-critic for discrete action spaces."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h1)),
            nn.Tanh(),
            layer_init(nn.Linear(h1, h2)),
            nn.Tanh(),
            layer_init(nn.Linear(h2, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h1)),
            nn.Tanh(),
            layer_init(nn.Linear(h1, h2)),
            nn.Tanh(),
            layer_init(nn.Linear(h2, action_dim), std=0.01),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(obs).squeeze(-1)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (Schulman et al., 2016).

    rewards, values, dones: shape (num_steps, num_envs)
    next_value, next_done:  shape (num_envs,) — bootstrap for the step after the rollout
    Returns (advantages, returns), each shape (num_steps, num_envs).
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae_lam = torch.zeros_like(next_value)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - next_done
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam

    returns = advantages + values
    return advantages, returns


class PPOAgent:
    """Owns the policy/value network and the clipped-surrogate update step."""

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.network = ActorCritic(obs_dim, action_dim, config.hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate, eps=1e-5)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        return self.network.get_action_and_value(obs, action)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network.get_value(obs)

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
    ) -> dict:
        cfg = self.config
        batch_size = obs.shape[0]
        b_inds = np.arange(batch_size)

        clipfracs = []
        pg_loss, v_loss, entropy_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, new_logprob, entropy, new_value = self.network.get_action_and_value(
                    obs[mb_inds], actions[mb_inds]
                )
                log_ratio = new_logprob - logprobs[mb_inds]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_advantages = advantages[mb_inds]
                if cfg.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Clipped surrogate policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss, optionally clipped the same way as the policy ratio
                new_value = new_value.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (new_value - returns[mb_inds]) ** 2
                    v_clipped = values[mb_inds] + torch.clamp(
                        new_value - values[mb_inds], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

        return {
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        }

    def act_greedy(self, obs: np.ndarray) -> int:
        """Deterministic (argmax) action for evaluation/inference."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.network.actor(obs_t)
            return int(torch.argmax(logits, dim=-1).item())

    def save(self, path: str) -> None:
        torch.save(
            {"model_state_dict": self.network.state_dict(), "config": self.config},
            path,
        )

    def load(self, path: str) -> None:
        # weights_only=False: checkpoint also stores the PPOConfig dataclass
        # used for training, not just tensors. Safe for this repo's own output.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["model_state_dict"])
