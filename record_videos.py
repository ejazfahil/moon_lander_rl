#!/usr/bin/env python3
"""
Record MP4 videos of the Moon Lander at different training stages
Shows the learning progression from early training to perfect landing
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import imageio

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN Network (must match training)
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class VideoAgent:
    """Agent for video recording"""
    def __init__(self, state_size, action_size):
        self.qnetwork = DQN(state_size, action_size).to(device)
        self.qnetwork.eval()
    
    def act(self, state):
        """Select best action (no exploration)"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.qnetwork(state)
        return action_values.argmax().item()
    
    def load_checkpoint(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        self.qnetwork.load_state_dict(checkpoint['qnetwork_state_dict'])
        episode = checkpoint['episode']
        avg_score = np.mean(checkpoint['scores'][-100:]) if checkpoint['scores'] else 0
        return episode, avg_score

def record_episode_to_mp4(agent, env, output_path, max_steps=1000):
    """Record a single episode and save as MP4"""
    frames = []
    state, _ = env.reset()
    done = False
    score = 0
    steps = 0
    
    while not done and steps < max_steps:
        # Render and capture frame
        frame = env.render()
        frames.append(frame)
        
        # Take action
        action = agent.act(state)
        state, reward, done, truncated, _ = env.step(action)
        score += reward
        steps += 1
        done = done or truncated
    
    # Save as MP4 with high quality
    if frames:
        imageio.mimsave(output_path, frames, fps=30, codec='libx264', quality=8)
    
    return score, steps, len(frames)

def record_checkpoint_videos(checkpoint_path, episode_num, output_dir, num_episodes=3):
    """Record multiple episodes for a single checkpoint"""
    print(f"\n{'='*70}")
    print(f"Recording videos for Episode {episode_num}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")
    
    # Create environment
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Load agent
    agent = VideoAgent(state_size, action_size)
    episode, avg_score = agent.load_checkpoint(checkpoint_path)
    print(f"Checkpoint Episode: {episode}")
    print(f"Average Score (last 100): {avg_score:.2f}\n")
    
    scores = []
    
    # Record multiple episodes
    for i in range(num_episodes):
        output_file = os.path.join(output_dir, f"episode_{episode_num:03d}_run_{i+1}.mp4")
        print(f"Recording run {i+1}/{num_episodes}...", end=" ")
        
        score, steps, frames = record_episode_to_mp4(agent, env, output_file)
        scores.append(score)
        
        status = "✅ LANDED" if score >= 200 else "⚠️ CRASHED" if score < -100 else "🔶 PARTIAL"
        print(f"{status} | Score: {score:7.2f} | Steps: {steps:3d} | Frames: {frames}")
        print(f"   Saved: {output_file}")
    
    env.close()
    
    avg = np.mean(scores)
    print(f"\nAverage score for this checkpoint: {avg:.2f}")
    
    return scores

def find_best_landing(checkpoint_path, output_dir, num_attempts=10):
    """Find and record the best landing from multiple attempts"""
    print(f"\n{'='*70}")
    print(f"Finding best landing from final checkpoint")
    print(f"{'='*70}")
    
    # Create environment
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Load agent
    agent = VideoAgent(state_size, action_size)
    episode, avg_score = agent.load_checkpoint(checkpoint_path)
    print(f"Final Checkpoint Episode: {episode}")
    print(f"Average Score (last 100): {avg_score:.2f}\n")
    
    print(f"Running {num_attempts} attempts to find best landing...\n")
    
    best_score = -float('inf')
    best_frames = None
    best_steps = 0
    all_scores = []
    
    for i in range(num_attempts):
        frames = []
        state, _ = env.reset()
        done = False
        score = 0
        steps = 0
        
        while not done and steps < 1000:
            frame = env.render()
            frames.append(frame)
            action = agent.act(state)
            state, reward, done, truncated, _ = env.step(action)
            score += reward
            steps += 1
            done = done or truncated
        
        all_scores.append(score)
        status = "✅ LANDED" if score >= 200 else "⚠️ CRASHED" if score < -100 else "🔶 PARTIAL"
        print(f"Attempt {i+1:2d}: {status} | Score: {score:7.2f} | Steps: {steps:3d}")
        
        if score > best_score:
            best_score = score
            best_frames = frames.copy()
            best_steps = steps
            print(f"           ⭐ NEW BEST!")
    
    env.close()
    
    # Save best landing
    output_file = os.path.join(output_dir, "perfect_landing.mp4")
    if best_frames:
        imageio.mimsave(output_file, best_frames, fps=30, codec='libx264', quality=8)
        print(f"\n{'='*70}")
        print(f"✅ BEST LANDING SAVED!")
        print(f"   File: {output_file}")
        print(f"   Score: {best_score:.2f}")
        print(f"   Steps: {best_steps}")
        print(f"   Average of all attempts: {np.mean(all_scores):.2f}")
        print(f"   Success rate: {sum(s >= 200 for s in all_scores)}/{num_attempts} ({100*sum(s >= 200 for s in all_scores)/num_attempts:.0f}%)")
        print(f"{'='*70}")
    
    return best_score, all_scores

def main():
    """Main execution"""
    print("="*70)
    print("MOON LANDER - VIDEO RECORDING AT DIFFERENT TRAINING STAGES")
    print("="*70)
    
    # Setup directories
    checkpoint_dir = "checkpoints"
    output_dir = "videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define checkpoints to record
    checkpoints_to_record = [
        (50, "checkpoints/checkpoint_ep50.pth", "Early Learning"),
        (150, "checkpoints/checkpoint_ep150.pth", "Early-Mid Training"),
        (250, "checkpoints/checkpoint_ep250.pth", "Mid Training"),
        (350, "checkpoints/checkpoint_ep350.pth", "Late Training"),
        (499, "checkpoints/final_ep499.pth", "Final Training"),
    ]
    
    # Check which checkpoints exist
    available_checkpoints = []
    for ep_num, path, description in checkpoints_to_record:
        if os.path.exists(path):
            available_checkpoints.append((ep_num, path, description))
        else:
            print(f"⚠️  Checkpoint not found: {path}")
    
    if not available_checkpoints:
        print("\n❌ No checkpoints found! Please train the model first.")
        return
    
    print(f"\nFound {len(available_checkpoints)} checkpoints to record")
    print(f"Output directory: {output_dir}\n")
    
    # Record videos for each checkpoint
    all_results = {}
    for ep_num, path, description in available_checkpoints:
        print(f"\n📹 {description}")
        scores = record_checkpoint_videos(path, ep_num, output_dir, num_episodes=2)
        all_results[ep_num] = scores
    
    # Find and record best landing
    final_checkpoint = available_checkpoints[-1][1]
    best_score, all_attempts = find_best_landing(final_checkpoint, output_dir, num_attempts=10)
    
    # Summary
    print(f"\n\n{'='*70}")
    print("📊 VIDEO RECORDING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nVideos saved in: {output_dir}/")
    print("\nGenerated videos:")
    
    video_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.mp4')])
    for i, video_file in enumerate(video_files, 1):
        file_path = os.path.join(output_dir, video_file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  {i}. {video_file} ({size_mb:.2f} MB)")
    
    print(f"\n{'='*70}")
    print("✅ SUCCESS! All videos have been generated.")
    print(f"{'='*70}")
    print("\nTraining Progression Summary:")
    for ep_num, scores in all_results.items():
        avg = np.mean(scores)
        print(f"  Episode {ep_num:3d}: Average Score = {avg:7.2f}")
    print(f"\n  Best Landing: Score = {best_score:7.2f}")
    print(f"\nYou can now view the videos to see how the lander learned to land!")

if __name__ == "__main__":
    main()
