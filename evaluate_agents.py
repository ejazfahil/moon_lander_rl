#!/usr/bin/env python3
"""Head-to-head evaluation: original DQN baseline vs. new PPO agent.

Both agents are evaluated under identical conditions: same seeded episodes,
greedy (non-exploratory) action selection, same episode cap. This is
deliberately separate from the training-time rolling average, which for the
DQN includes epsilon-greedy exploration noise and for PPO includes stochastic
sampling — neither is a fair final-performance number on its own.
"""
import argparse
import json
import statistics as stats

import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import DQNAgent
from src.ppo_agent import PPOAgent
from src.ppo_config import PPOConfig


def evaluate_dqn(checkpoint_path: str, env_name: str, episodes: int, seed: int) -> list:
    env = gym.make(env_name)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.load_checkpoint(checkpoint_path)

    returns = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = truncated = False
        total = 0.0
        while not (done or truncated):
            action = agent.act(state, train=False)  # greedy, no epsilon exploration
            state, reward, done, truncated, _ = env.step(action)
            total += reward
        returns.append(total)
    env.close()
    return returns


def evaluate_ppo(checkpoint_path: str, env_name: str, episodes: int, seed: int) -> list:
    env = gym.make(env_name)
    device = torch.device("cpu")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, PPOConfig(), device)
    agent.load(checkpoint_path)

    returns = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = truncated = False
        total = 0.0
        while not (done or truncated):
            action = agent.act_greedy(state)  # argmax over policy logits, no sampling
            state, reward, done, truncated, _ = env.step(action)
            total += reward
        returns.append(total)
    env.close()
    return returns


def summarize(name: str, returns: list, solve_threshold: float = 200.0) -> dict:
    arr = np.array(returns)
    summary = {
        "agent": name,
        "episodes": len(returns),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(stats.median(returns)),
        "solve_rate": float((arr >= solve_threshold).mean()),
    }
    print(
        f"{name:>12} | mean {summary['mean']:7.2f} | std {summary['std']:6.2f} | "
        f"median {summary['median']:7.2f} | min {summary['min']:7.2f} | max {summary['max']:7.2f} | "
        f"solve_rate {summary['solve_rate']*100:5.1f}%"
    )
    return summary


def plot_comparison(dqn_returns, ppo_returns, save_path="agent_comparison.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([dqn_returns, ppo_returns], tick_labels=["DQN (baseline)", "PPO"], showmeans=True)
    ax.axhline(y=200, color="red", linestyle="--", linewidth=1.5, label="Solved threshold")
    ax.set_ylabel("Episode Return (greedy eval)")
    ax.set_title("DQN vs. PPO — LunarLander-v3, 100 greedy eval episodes each")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn-checkpoint", default="checkpoints/final_ep499.pth")
    parser.add_argument("--ppo-checkpoint", default="checkpoints_ppo/ppo_solved_step2506752.pth")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--env-name", default="LunarLander-v3")
    args = parser.parse_args()

    print(f"Evaluating over {args.episodes} greedy episodes each (seed {args.seed}+)...\n")

    dqn_returns = evaluate_dqn(args.dqn_checkpoint, args.env_name, args.episodes, args.seed)
    ppo_returns = evaluate_ppo(args.ppo_checkpoint, args.env_name, args.episodes, args.seed)

    dqn_summary = summarize("DQN", dqn_returns)
    ppo_summary = summarize("PPO", ppo_returns)

    plot_comparison(dqn_returns, ppo_returns)

    with open("agent_comparison.json", "w") as f:
        json.dump(
            {
                "dqn": dqn_summary,
                "ppo": ppo_summary,
                "dqn_returns": dqn_returns,
                "ppo_returns": ppo_returns,
            },
            f,
            indent=2,
        )
    print("\nSaved raw results: agent_comparison.json")


if __name__ == "__main__":
    main()
