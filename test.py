#!/usr/bin/env python3
"""
Test the trained Moon Lander agent
Loads a checkpoint and evaluates performance
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import os

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

class TestAgent:
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
        print(f"✓ Loaded checkpoint from episode {checkpoint['episode']}")
        return checkpoint['episode']

def test_agent(agent, env, episodes=10):
    """Test the agent and return scores"""
    scores = []
    
    print(f"\nTesting agent for {episodes} episodes...\n")
    print("=" * 60)
    
    for episode in range(episodes):
        state, _ = env.reset()
        score = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            score += reward
            steps += 1
            done = done or truncated
        
        scores.append(score)
        status = "✅ LANDED" if score >= 200 else "⚠️  CRASHED" if score < -100 else "🔶 PARTIAL"
        print(f"Episode {episode + 1:2d}: {status} | Score: {score:7.2f} | Steps: {steps:3d}")
    
    print("=" * 60)
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    
    print(f"\nTest Results:")
    print(f"  Average Score: {avg_score:.2f} ± {std_score:.2f}")
    print(f"  Max Score:     {max_score:.2f}")
    print(f"  Min Score:     {min_score:.2f}")
    print(f"  Success Rate:  {sum(s >= 200 for s in scores)}/{episodes} ({100*sum(s >= 200 for s in scores)/episodes:.0f}%)")
    
    return scores

def record_landing(agent, env, filename='videos/trained_landing.gif'):
    """Record a video of the agent landing"""
    os.makedirs('videos', exist_ok=True)
    
    frames = []
    state, _ = env.reset()
    done = False
    score = 0
    
    print(f"\nRecording landing video...")
    
    while not done:
        frames.append(env.render())
        action = agent.act(state)
        state, reward, done, truncated, _ = env.step(action)
        score += reward
        done = done or truncated
    
    # Save as GIF
    imageio.mimsave(filename, frames, fps=30)
    print(f"✓ Video saved: {filename}")
    print(f"  Score: {score:.2f}")
    
    return score

def main():
    print("=" * 60)
    print("MOON LANDER - TESTING TRAINED AGENT")
    print("=" * 60)
    
    # Create environment with visual rendering
    env = gym.make('LunarLander-v3', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = TestAgent(state_size, action_size)
    
    # Find the latest checkpoint
    checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
    if not checkpoints:
        print("Error: No checkpoints found in checkpoints/")
        return
    
    # Use final checkpoint if available, otherwise use latest
    if 'final_ep499.pth' in checkpoints:
        checkpoint_path = 'checkpoints/final_ep499.pth'
    else:
        checkpoint_path = os.path.join('checkpoints', sorted(checkpoints)[-1])
    
    print(f"\nUsing checkpoint: {checkpoint_path}")
    
    agent.load_checkpoint(checkpoint_path)
    
    # Test the agent
    scores = test_agent(agent, env, episodes=10)
    
    # Record a video
    render_env = gym.make('LunarLander-v3', render_mode='rgb_array')
    render_agent = TestAgent(state_size, action_size)
    render_agent.load_checkpoint(checkpoint_path)
    record_landing(render_agent, render_env)
    
    # Cleanup
    env.close()
    render_env.close()
    
    print("\n" + "=" * 60)
    print("✅ TESTING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
