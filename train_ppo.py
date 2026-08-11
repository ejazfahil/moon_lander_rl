#!/usr/bin/env python3
"""Moon Lander - Proximal Policy Optimization (PPO) training script.

A second, modern RL agent alongside the repo's original DQN baseline.
Uses vectorized environments (gymnasium.vector.SyncVectorEnv), GAE(lambda)
advantage estimation, and the clipped-surrogate PPO objective — see
src/ppo_agent.py and src/ppo_config.py for algorithm references.
"""
import os
import time
import random
import argparse

import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ppo_config import PPOConfig
from src.ppo_agent import PPOAgent, compute_gae
from src.metrics import EpisodeStats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_name: str, seed: int):
    def thunk():
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return thunk


def train(config: PPOConfig, device: torch.device, checkpoint_dir: str = "checkpoints_ppo"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    set_seed(config.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(config.env_name, config.seed + i) for i in range(config.num_envs)]
    )
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n

    agent = PPOAgent(obs_dim, action_dim, config, device)
    stats = EpisodeStats(window=100)

    num_updates = config.total_timesteps // config.batch_size

    obs_buf = torch.zeros((config.num_steps, config.num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards_buf = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones_buf = torch.zeros((config.num_steps, config.num_envs), device=device)
    values_buf = torch.zeros((config.num_steps, config.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(config.num_envs, device=device)

    avg_scores_log = []
    best_avg = float("-inf")
    solved_at_update = None

    for update in range(1, num_updates + 1):
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            agent.optimizer.param_groups[0]["lr"] = frac * config.learning_rate

        for step in range(config.num_steps):
            global_step += config.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

            if "episode" in infos:
                finished = infos["episode"]["_r"] if "_r" in infos["episode"] else np.ones_like(infos["episode"]["r"], dtype=bool)
                for ep_r, is_final in zip(infos["episode"]["r"], finished):
                    if is_final:
                        stats.record(float(ep_r))

        with torch.no_grad():
            next_value = agent.get_value(next_obs)
        advantages, returns = compute_gae(
            rewards_buf, values_buf, dones_buf, next_value, next_done, config.gamma, config.gae_lambda
        )

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        losses = agent.update(b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values)

        avg_score = stats.rolling_mean()
        avg_scores_log.append((global_step, avg_score))
        best_avg = max(best_avg, avg_score)

        if update % 5 == 0 or update == num_updates:
            elapsed = time.time() - start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0
            print(
                f"update {update:4d}/{num_updates} | step {global_step:8d} | "
                f"avg(100) {avg_score:7.2f} | best {stats.best():7.2f} | "
                f"pg_loss {losses['pg_loss']:.4f} | v_loss {losses['v_loss']:.4f} | "
                f"entropy {losses['entropy']:.3f} | clipfrac {losses['clipfrac']:.3f} | "
                f"{sps} steps/s"
            )

        if update % 20 == 0:
            agent.save(f"{checkpoint_dir}/ppo_step{global_step}.pth")

        if solved_at_update is None and stats.is_solved(config.solve_score):
            solved_at_update = update
            agent.save(f"{checkpoint_dir}/ppo_solved_step{global_step}.pth")
            print(f"\n{'=' * 70}\nSolved at update {update} (step {global_step})! avg(100)={avg_score:.2f}\n{'=' * 70}\n")

    agent.save(f"{checkpoint_dir}/ppo_final_step{global_step}.pth")
    envs.close()

    return stats, avg_scores_log, solved_at_update, best_avg


def plot_progress(avg_scores_log, solve_score: float, save_path: str = "ppo_training_progress.png"):
    steps, avgs = zip(*avg_scores_log)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, avgs, linewidth=2, color="darkgreen", label="Average Score (100 episodes)")
    ax.axhline(y=solve_score, color="red", linestyle="--", linewidth=2, label="Solved Threshold")
    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Moon Lander PPO Training Progress", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    args = parser.parse_args()

    config = PPOConfig()
    if args.total_timesteps:
        config.total_timesteps = args.total_timesteps
    if args.num_envs:
        config.num_envs = args.num_envs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(config)

    stats, avg_scores_log, solved_at_update, best_avg = train(config, device)
    plot_progress(avg_scores_log, config.solve_score)

    print("\nFinal stats:")
    print(f"  Best avg(100): {best_avg:.2f}")
    print(f"  Solved: {'yes, update ' + str(solved_at_update) if solved_at_update else 'no'}")


if __name__ == "__main__":
    main()
