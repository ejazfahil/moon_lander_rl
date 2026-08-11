#!/usr/bin/env python3
"""Record MP4 videos of the trained PPO agent (mirrors record_videos.py's
DQN recording style, applied to the new PPO checkpoint)."""
import os
import torch
import numpy as np
import gymnasium as gym
import imageio

from src.ppo_agent import PPOAgent
from src.ppo_config import PPOConfig


def record_episode(agent: PPOAgent, env, output_path: str, max_steps: int = 1000):
    frames = []
    state, _ = env.reset()
    done = truncated = False
    score = 0.0
    steps = 0

    while not (done or truncated) and steps < max_steps:
        frames.append(env.render())
        action = agent.act_greedy(state)
        state, reward, done, truncated, _ = env.step(action)
        score += reward
        steps += 1

    if frames:
        imageio.mimsave(output_path, frames, fps=30, codec="libx264", quality=8)
    return score, steps


def main():
    checkpoint_path = "checkpoints_ppo/ppo_solved_step2506752.pth"
    output_dir = "videos"
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, PPOConfig(), torch.device("cpu"))
    agent.load(checkpoint_path)

    print(f"Loaded: {checkpoint_path}")
    print("Recording 3 sample runs + searching for best landing over 10 attempts...\n")

    scores = []
    for i in range(3):
        out = os.path.join(output_dir, f"ppo_run_{i + 1}.mp4")
        score, steps = record_episode(agent, env, out)
        scores.append(score)
        status = "LANDED" if score >= 200 else "CRASHED" if score < -100 else "PARTIAL"
        print(f"  run {i + 1}: {status} | score {score:7.2f} | steps {steps:4d} -> {out}")

    best_score, best_frames, best_steps = -float("inf"), None, 0
    all_scores = []
    for i in range(10):
        frames, state = [], env.reset()[0]
        done = truncated = False
        score = steps = 0
        while not (done or truncated) and steps < 1000:
            frames.append(env.render())
            action = agent.act_greedy(state)
            state, reward, done, truncated, _ = env.step(action)
            score += reward
            steps += 1
        all_scores.append(score)
        if score > best_score:
            best_score, best_frames, best_steps = score, frames, steps

    env.close()

    if best_frames:
        out = os.path.join(output_dir, "ppo_best_landing.mp4")
        imageio.mimsave(out, best_frames, fps=30, codec="libx264", quality=8)
        print(f"\nBest landing: score {best_score:.2f} over {best_steps} steps -> {out}")
        print(f"Success rate over 10 attempts: {sum(s >= 200 for s in all_scores)}/10")


if __name__ == "__main__":
    main()
