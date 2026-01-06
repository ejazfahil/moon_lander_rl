# 🚀 Moon Lander - Deep Reinforcement Learning

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Train an AI agent to master lunar landing using Deep Q-Network (DQN) reinforcement learning! Watch the agent learn from scratch, progressing from crashes to perfect landings.

## 🎯 Demo

### Perfect Landing - Score: 299.51 ✅

The trained agent successfully lands the spacecraft with precision! Check out the [perfect_landing.mp4](videos/perfect_landing.mp4) video to see it in action.

### Training Progression

Watch the agent's learning journey from complete beginner to expert pilot:

| Training Stage | Episode | Videos | Average Score | Status |
|---------------|---------|--------|---------------|--------|
| 🆕 Early Learning | 50 | [Run 1](videos/episode_050_run_1.mp4), [Run 2](videos/episode_050_run_2.mp4) | -68.11 | Learning basics |
| 📈 Early-Mid | 150 | [Run 1](videos/episode_150_run_1.mp4), [Run 2](videos/episode_150_run_2.mp4) | -91.03 | Experimenting |
| 🔄 Mid Training | 250 | [Run 1](videos/episode_250_run_1.mp4), [Run 2](videos/episode_250_run_2.mp4) | -129.58 | Exploring strategies |
| 🎓 Late Training | 350 | [Run 1](videos/episode_350_run_1.mp4), [Run 2](videos/episode_350_run_2.mp4) | -309.07 | Refining approach |
| 🏆 Final | 499 | [Run 1](videos/episode_499_run_1.mp4), [Run 2](videos/episode_499_run_2.mp4) | -58.70 | Mixed performance |
| ⭐ **Best Landing** | - | **[Perfect Landing](videos/perfect_landing.mp4)** | **299.51** | **Success!** |

> **Best Performance**: Out of 10 landing attempts, the agent achieved a 40% success rate with the best score of 299.51!

## ✨ Features

- ✅ **Deep Q-Learning (DQN)** with Experience Replay
- ✅ **Target Network** for stable training
- ✅ **Epsilon-Greedy Exploration** with adaptive decay
- ✅ **Automatic Checkpointing** every 50 episodes
- ✅ **Resume Training** from any checkpoint
- ✅ **Video Recording** of agent performance at different stages
- ✅ **PyTorch Implementation** with GPU support
- ✅ **Real-time Visualization** of training progress
- ✅ **Multiple Training Scripts** (notebook, standalone, video recording)

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/ejazfahil/moon_lander_rl.git
cd moon_lander_rl
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Agent

**Option A: Using Python Script**
```bash
python3 train.py
```

**Option B: Using Jupyter Notebook**
```bash
jupyter notebook moon_lander_training.ipynb
```

### 4. Record Videos

Generate videos showing training progression:
```bash
python3 record_videos.py
```

This creates videos at different training stages (episodes 50, 150, 250, 350, 499) plus a "perfect landing" video from the best performing run.

### 5. Test Trained Agent

```bash
python3 test.py
```

## 🎮 How It Works

### The Environment

The agent controls a lunar lander spacecraft with these **actions**:
- `0`: Do nothing
- `1`: Fire left engine (rotate right)
- `2`: Fire main engine (thrust up)
- `3`: Fire right engine (rotate left)

### Rewards

- `+100` to `+140`: Successful landing (based on smoothness)
- `-100`: Crashing
- Small penalties for fuel usage and distance from landing pad
- Bonus for legs touching ground

### Goal

The environment is **solved** when the agent achieves an average score ≥ 200 over 100 consecutive episodes.

### Learning Algorithm

**Deep Q-Network (DQN)** combines:
1. **Neural Network**: Approximates Q-values for state-action pairs
2. **Experience Replay**: Stores past experiences to break correlation
3. **Target Network**: Stabilizes learning by using delayed Q-value updates
4. **Epsilon-Greedy**: Balances exploration vs exploitation

## 📊 Training Results

### Performance Metrics

- **Total Episodes**: 500
- **Best Score**: 299.51
- **Final Average (100 episodes)**: 77.88
- **Success Rate (best checkpoint)**: 40%
- **Training Time**: ~2-3 hours on CPU

### Training Progress

![Training Progress](training_progress.png)

The graph shows episode scores (light blue) and 100-episode moving average (dark blue). The red dashed line indicates the "solved" threshold of 200.

## 📁 Project Structure

```
moon_lander_rl/
├── train.py                     # Main training script
├── test.py                      # Test trained agent
├── record_videos.py             # Generate training progression videos
├── moon_lander_training.ipynb   # Interactive Jupyter notebook
├── requirements.txt             # Python dependencies
├── checkpoints/                 # Saved model checkpoints
│   ├── checkpoint_ep50.pth
│   ├── checkpoint_ep100.pth
│   ├── checkpoint_ep150.pth
│   ├── checkpoint_ep200.pth
│   ├── checkpoint_ep250.pth
│   ├── checkpoint_ep300.pth
│   ├── checkpoint_ep350.pth
│   ├── checkpoint_ep400.pth
│   ├── checkpoint_ep450.pth
│   └── final_ep499.pth
├── videos/                      # Training videos (11 MP4 files)
│   ├── episode_050_run_1.mp4
│   ├── episode_050_run_2.mp4
│   ├── episode_150_run_1.mp4
│   ├── episode_150_run_2.mp4
│   ├── episode_250_run_1.mp4
│   ├── episode_250_run_2.mp4
│   ├── episode_350_run_1.mp4
│   ├── episode_350_run_2.mp4
│   ├── episode_499_run_1.mp4
│   ├── episode_499_run_2.mp4
│   └── perfect_landing.mp4      ⭐ Best landing!
└── training_progress.png        # Training visualization
```

## 🔧 Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Training** | | |
| Episodes | 500 | Total training episodes |
| Max Steps | 1000 | Maximum steps per episode |
| Batch Size | 64 | Samples per learning step |
| **Network** | | |
| Hidden Size | 128 | Neural network hidden layer size |
| Learning Rate | 0.001 (1e-3) | Adam optimizer learning rate |
| **DQN** | | |
| Gamma (γ) | 0.99 | Discount factor for future rewards |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.01 | Minimum exploration rate |
| Epsilon Decay | 0.995 | Exploration decay per episode |
| **Updates** | | |
| Target Update | Every 10 episodes | Target network sync frequency |
| Checkpoint Freq | Every 50 episodes | Model save frequency |
| Buffer Size | 100,000 | Experience replay buffer capacity |

### Adjusting Training

Modify these settings in `train.py`:

```python
# Training configuration
episodes = 500          # Increase for more training
MAX_STEPS = 1000       # Max steps per episode

# DQN hyperparameters
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    lr=1e-3,               # Learning rate
    gamma=0.99,            # Discount factor
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995    # Exploration decay
)
```

## 💡 Usage Examples

### Resume Training from Checkpoint

```python
# In train.py, modify the train_agent call:
scores, avg_scores = train_agent(
    agent, 
    env, 
    episodes=1000,
    load_checkpoint_path='checkpoints/checkpoint_ep450.pth'
)
```

### Generate Custom Videos

```python
# Run record_videos.py to generate videos from all checkpoints
python3 record_videos.py
```

The script will:
- Load each checkpoint (50, 150, 250, 350, 499)
- Record 2 episodes per checkpoint
- Run 10 attempts to find the best landing
- Save all videos as MP4 files

### Test Specific Checkpoint

```python
# In test.py, specify checkpoint:
checkpoint_path = 'checkpoints/checkpoint_ep350.pth'
agent.load_checkpoint(checkpoint_path)
```

## 🎓 Learning Insights

### What the Agent Learned

1. **Early Phase (0-100 episodes)**
   - Basic engine controls
   - Gravity and thrust mechanics
   - Crashes are common but improving

2. **Mid Phase (100-300 episodes)**
   - Positioning strategies
   - Fuel management
   - Approach angle optimization

3. **Late Phase (300-500 episodes)**
   - Fine-tuned landing techniques
   - Consistent approach patterns
   - High success rate on favorable starts

### Observations

- The agent shows **high variance** in performance, indicating sensitivity to initial conditions
- Best results achieved when starting position is favorable
- Some checkpoints show regression (e.g., episode 350) due to exploration
- Final checkpoint achieves **40% success rate** with scores above 200

## 🛠️ Troubleshooting

**Agent not improving?**
- Train for more episodes (try 1000+)
- Adjust learning rate (try 5e-4 or 2e-3)
- Modify epsilon decay for longer exploration

**Training too slow?**
- Use GPU if available (automatic detection)
- Reduce batch size to 32
- Decrease buffer size

**Videos not generating?**
- Ensure `imageio-ffmpeg` is installed: `pip install imageio-ffmpeg`
- Check that checkpoints exist in `checkpoints/` directory

**Box2D errors?**
- Install swig: `brew install swig` (Mac) or `apt-get install swig` (Linux)
- Then: `pip install box2d-py`

## 🚀 Next Steps & Improvements

- [ ] Implement **Double DQN** to reduce overestimation bias
- [ ] Add **Dueling DQN** architecture for better value estimation
- [ ] Integrate **Prioritized Experience Replay**
- [ ] Experiment with **reward shaping** for faster learning
- [ ] Try **Soft Actor-Critic (SAC)** for continuous control
- [ ] Implement **curriculum learning** with progressive difficulty

## 📚 Resources

- [Gymnasium LunarLander Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [DQN Paper (Mnih et al., 2015)](https://arxiv.org/abs/1312.5602)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

**Made with** 🤖 **Deep Reinforcement Learning** | **Powered by** PyTorch & OpenAI Gymnasium

