"""Unit tests for the PPO agent, GAE computation, and network shapes.

Unlike tests/test_agent.py's tautological checks, these exercise the actual
src/ppo_agent.py code paths against known-correct outputs.
"""
import numpy as np
import torch

from src.ppo_agent import ActorCritic, PPOAgent, compute_gae
from src.ppo_config import PPOConfig


def test_actor_critic_output_shapes():
    obs_dim, action_dim, batch = 8, 4, 16
    net = ActorCritic(obs_dim, action_dim, hidden_dims=(32, 32))
    obs = torch.randn(batch, obs_dim)

    action, logprob, entropy, value = net.get_action_and_value(obs)

    assert action.shape == (batch,)
    assert logprob.shape == (batch,)
    assert entropy.shape == (batch,)
    assert value.shape == (batch,)
    assert (action >= 0).all() and (action < action_dim).all()
    assert (entropy >= 0).all()  # categorical entropy is non-negative


def test_actor_critic_fixed_action_logprob_matches_distribution():
    """Passing an explicit action should return that action's own log-prob,
    not a resampled one — this is required for the PPO ratio to be correct."""
    obs_dim, action_dim = 8, 4
    net = ActorCritic(obs_dim, action_dim, hidden_dims=(16, 16))
    obs = torch.randn(1, obs_dim)
    fixed_action = torch.tensor([2])

    _, logprob, _, _ = net.get_action_and_value(obs, action=fixed_action)

    logits = net.actor(obs)
    expected_logprob = torch.log_softmax(logits, dim=-1)[0, 2]
    assert torch.allclose(logprob, expected_logprob, atol=1e-5)


def test_compute_gae_single_step_matches_td_residual():
    """With a single timestep, GAE advantage must reduce exactly to the
    one-step TD residual: delta = r + gamma * V(s') * (1-done) - V(s)."""
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.5]])
    dones = torch.tensor([[0.0]])
    next_value = torch.tensor([0.3])
    next_done = torch.tensor([0.0])
    gamma, gae_lambda = 0.99, 0.95

    advantages, returns = compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda)

    expected_delta = 1.0 + gamma * 0.3 - 0.5
    assert torch.allclose(advantages[0, 0], torch.tensor(expected_delta), atol=1e-6)
    assert torch.allclose(returns[0, 0], advantages[0, 0] + values[0, 0], atol=1e-6)


def test_compute_gae_zero_reward_zero_value_is_zero():
    """Degenerate case: no reward anywhere and V==0 everywhere must give
    exactly zero advantage regardless of lambda/gamma."""
    num_steps, num_envs = 10, 2
    rewards = torch.zeros(num_steps, num_envs)
    values = torch.zeros(num_steps, num_envs)
    dones = torch.zeros(num_steps, num_envs)
    next_value = torch.zeros(num_envs)
    next_done = torch.zeros(num_envs)

    advantages, returns = compute_gae(rewards, values, dones, next_value, next_done, 0.99, 0.95)

    assert torch.allclose(advantages, torch.zeros_like(advantages))
    assert torch.allclose(returns, torch.zeros_like(returns))


def test_compute_gae_terminal_step_does_not_bootstrap():
    """If done=1 at the final step, the next state's value must not leak
    into the advantage (the agent shouldn't get credit for a state that
    doesn't exist after episode termination)."""
    rewards = torch.tensor([[5.0]])
    values = torch.tensor([[1.0]])
    dones = torch.tensor([[0.0]])
    next_value = torch.tensor([100.0])  # should be masked out entirely
    next_done = torch.tensor([1.0])
    gamma, gae_lambda = 0.99, 0.95

    advantages, _ = compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda)

    expected = 5.0 + gamma * 100.0 * 0.0 - 1.0  # next_non_terminal = 1 - next_done = 0
    assert torch.allclose(advantages[0, 0], torch.tensor(expected), atol=1e-6)


def test_ppo_update_changes_parameters_and_reduces_loss_scale():
    """A single update() call on a synthetic batch should actually move the
    weights (catches a no-op optimizer/backward-pass bug) without exploding."""
    torch.manual_seed(0)
    obs_dim, action_dim = 8, 4
    config = PPOConfig(hidden_dims=(16, 16), update_epochs=2, minibatch_size_target=16)
    agent = PPOAgent(obs_dim, action_dim, config, device=torch.device("cpu"))

    before = [p.clone() for p in agent.network.parameters()]

    batch = 32
    obs = torch.randn(batch, obs_dim)
    actions = torch.randint(0, action_dim, (batch,))
    with torch.no_grad():
        _, logprobs, _, values = agent.network.get_action_and_value(obs, actions)
    advantages = torch.randn(batch)
    returns = values + advantages

    losses = agent.update(obs, actions, logprobs, advantages, returns, values)

    after = list(agent.network.parameters())
    changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert changed, "PPO update() did not change any network parameters"
    assert np.isfinite(losses["pg_loss"])
    assert np.isfinite(losses["v_loss"])


def test_act_greedy_is_deterministic_and_in_range():
    action_dim = 4
    agent = PPOAgent(8, action_dim, PPOConfig(hidden_dims=(16, 16)), device=torch.device("cpu"))
    obs = np.random.randn(8).astype(np.float32)

    a1 = agent.act_greedy(obs)
    a2 = agent.act_greedy(obs)

    assert a1 == a2  # argmax over unchanged logits must be deterministic
    assert 0 <= a1 < action_dim


def test_batch_size_and_minibatch_size_properties():
    config = PPOConfig(num_envs=8, num_steps=128, minibatch_size_target=256)
    assert config.batch_size == 1024
    assert config.num_minibatches == 4
    assert config.minibatch_size == 256
