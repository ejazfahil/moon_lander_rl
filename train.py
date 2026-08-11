#!/usr/bin/env python3
"""
Moon Lander - Deep Q-Network (DQN) Training Script
Train an RL agent to land a spacecraft on the moon using Deep Q-Learning
"""

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for visualization
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from tqdm import tqdm
import os
from datetime import datetime

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ==================== DQN Neural Network ====================
class DQN(nn.Module):
    """Deep Q-Network for LunarLander"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ==================== Experience Replay Buffer ====================
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(device)
        actions = torch.LongTensor(np.array([e.action for e in experiences])).to(device)
        rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(device)
        dones = torch.FloatTensor(np.array([e.done for e in experiences])).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

# ==================== DQN Agent ====================
class DQNAgent:
    """DQN Agent with experience replay and target network"""
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-Network and Target Network
        self.qnetwork = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.qnetwork.state_dict())
        
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        
    def act(self, state, train=True):
        """Select action using epsilon-greedy policy"""
        if train and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()
        return action_values.argmax().item()
    
    def step(self, state, action, reward, next_state, done):
        """Save experience and learn from batch"""
        self.memory.add(state, action, reward, next_state, done)
    
    def learn(self, batch_size=64):
        """Update Q-network using batch of experiences"""
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Get current Q values
        current_q = self.qnetwork(states).gather(1, actions.unsqueeze(1))
        
        # Get target Q values
        next_q = self.target_network(next_states).max(1)[0].detach()
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.qnetwork.state_dict())
    
    def save_checkpoint(self, filepath, episode, scores):
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'qnetwork_state_dict': self.qnetwork.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'scores': scores
        }
        torch.save(checkpoint, filepath)
        print(f"\n✓ Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        # weights_only=False: these checkpoints predate PyTorch 2.6's stricter
        # default and contain plain numpy scalars in the scores list, not just
        # tensors. Safe here since the checkpoints are this repo's own output.
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        self.qnetwork.load_state_dict(checkpoint['qnetwork_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"\n✓ Checkpoint loaded: {filepath}")
        return checkpoint['episode'], checkpoint['scores']

# ==================== Training Function ====================
def train_agent(agent, env, episodes=500, load_checkpoint_path=None):
    """Train the DQN agent with periodic checkpointing"""
    
    # Configuration
    MAX_STEPS = 1000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 10
    CHECKPOINT_FREQ = 50  # Checkpoint every 50 episodes for demo
    PRINT_FREQ = 10
    
    scores = []
    avg_scores = []
    start_episode = 0
    
    # Load checkpoint if provided
    if load_checkpoint_path and os.path.exists(load_checkpoint_path):
        start_episode, scores = agent.load_checkpoint(load_checkpoint_path)
        print(f"Resuming from episode {start_episode}\n")
    
    print("🚀 Starting training...\n")
    print(f"Episodes: {episodes}")
    print(f"Checkpoint frequency: Every {CHECKPOINT_FREQ} episodes")
    print(f"Target network update: Every {TARGET_UPDATE_FREQ} episodes\n")
    print("=" * 70)
    
    # Training loop
    for episode in range(start_episode, episodes):
        state, _ = env.reset()
        score = 0
        
        for step in range(MAX_STEPS):
            # Select and perform action
            action = agent.act(state, train=True)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience and learn
            agent.step(state, action, reward, next_state, done or truncated)
            agent.learn(BATCH_SIZE)
            
            state = next_state
            score += reward
            
            if done or truncated:
                break
        
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        # Update target network
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
        
        # Print progress
        if episode % PRINT_FREQ == 0 or episode == episodes - 1:
            print(f"Ep {episode:4d} | Score: {score:7.2f} | Avg(100): {avg_score:7.2f} | ε: {agent.epsilon:.3f} | Buffer: {len(agent.memory)}")
        
        # Save periodic checkpoint
        if episode % CHECKPOINT_FREQ == 0 and episode > 0:
            checkpoint_path = f"checkpoints/checkpoint_ep{episode}.pth"
            agent.save_checkpoint(checkpoint_path, episode, scores)
        
        # Early stopping if solved
        if avg_score >= 200:
            print("\n" + "=" * 70)
            print(f"🎉 Environment solved in {episode} episodes!")
            print(f"Average Score: {avg_score:.2f}")
            print("=" * 70)
            agent.save_checkpoint(f"checkpoints/solved_ep{episode}.pth", episode, scores)
            break
    
    # Save final checkpoint
    print("\n" + "=" * 70)
    print("Training complete!")
    agent.save_checkpoint(f"checkpoints/final_ep{episode}.pth", episode, scores)
    
    return scores, avg_scores

# ==================== Visualization Function ====================
def plot_training_progress(scores, avg_scores, save_path='training_progress.png'):
    """Plot and save training progress"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(scores, alpha=0.3, label='Episode Score', color='steelblue')
    ax.plot(avg_scores, linewidth=2, label='Average Score (100 episodes)', color='darkblue')
    ax.axhline(y=200, color='red', linestyle='--', linewidth=2, label='Solved Threshold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Moon Lander Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training plot saved: {save_path}")
    
    print(f"\nTraining Statistics:")
    print(f"  Best average score: {max(avg_scores):.2f}")
    print(f"  Final average score: {avg_scores[-1]:.2f}")
    print(f"  Total episodes: {len(scores)}")

# ==================== Main Execution ====================
def main():
    """Main execution function"""
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('videos', exist_ok=True)
    
    # Create environment
    print("=" * 70)
    print("MOON LANDER - DEEP Q-NETWORK TRAINING")
    print("=" * 70)
    env = gym.make('LunarLander-v3', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"\nEnvironment: LunarLander-v3")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Actions: 0=nothing, 1=left engine, 2=main engine, 3=right engine")
    print()
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    print("✓ Agent initialized\n")
    
    # Train agent
    scores, avg_scores = train_agent(agent, env, episodes=500)
    
    # Plot results
    plot_training_progress(scores, avg_scores)
    
    # Close environment
    env.close()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nCheckpoints saved in: ./checkpoints/")
    print("Training plot saved: ./training_progress.png")
    print("\nTo test the trained agent, use the test script or notebook.")

if __name__ == "__main__":
    main()
