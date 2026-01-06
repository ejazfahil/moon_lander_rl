# 🚀 Moon Lander Reinforcement Learning

Complete Deep Q-Network (DQN) implementation for training an agent to land a spacecraft on the moon.

## Features

- ✅ Deep Q-Learning (DQN) with Experience Replay
- ✅ Periodic checkpointing every 100 episodes
- ✅ Resume training from any checkpoint
- ✅ Real-time progress visualization
- ✅ Video recording of agent performance
- ✅ PyTorch implementation with GPU support

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the Training Notebook

```bash
jupyter notebook moon_lander_training.ipynb
```

## How It Works

### Training Process

1. **Agent observes** the lunar lander's state (position, velocity, angle, etc.)
2. **Agent selects action** using epsilon-greedy policy
3. **Environment responds** with reward and next state
4. **Agent learns** from experiences using DQN algorithm
5. **Checkpoints automatically saved** every 100 episodes

### Periodic Learning

The agent improves through:
- **Experience Replay**: Stores and reuses past experiences
- **Target Network**: Stabilizes training
- **Epsilon Decay**: Gradually reduces exploration
- **Regular Checkpoints**: Saves progress periodically

### Actions

- `0`: Do nothing
- `1`: Fire left engine
- `2`: Fire main engine
- `3`: Fire right engine

### Rewards

- `+100` for successful landing
- `-100` for crashing
- Penalties for fuel usage and distance from landing pad

### Solved Condition

Environment is considered solved when average score ≥ 200 over 100 consecutive episodes.

## Project Structure

```
moon_lander_rl/
├── moon_lander_training.ipynb   # Main training notebook
├── requirements.txt              # Python dependencies
├── checkpoints/                  # Saved model checkpoints
│   ├── checkpoint_ep100.pth
│   ├── checkpoint_ep200.pth
│   └── final_ep1000.pth
├── videos/                       # Recorded landing videos
│   └── landing.gif
└── training_progress.png         # Training visualization
```

## Usage Examples

### Start Fresh Training

Simply run all cells in the notebook. Checkpoints will be saved automatically.

### Resume from Checkpoint

In cell 8, uncomment and modify:

```python
scores, avg_scores = train_agent(
    agent, 
    env, 
    load_checkpoint_path='checkpoints/checkpoint_ep100.pth'
)
```

### Test Trained Agent

Run cells 10-11 to evaluate the agent's performance and record a video.

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Episodes | 1000 | Total training episodes |
| Batch Size | 64 | Samples per learning step |
| Learning Rate | 0.001 | Optimizer learning rate |
| Gamma | 0.99 | Discount factor |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.01 | Minimum exploration rate |
| Epsilon Decay | 0.995 | Exploration decay rate |
| Target Update | 10 episodes | Target network update frequency |
| Checkpoint Freq | 100 episodes | Save checkpoint frequency |

## Expected Results

- **Episodes 0-100**: Agent learns basic controls, scores around -200 to -100
- **Episodes 100-300**: Consistent improvement, scores reach 0 to 100
- **Episodes 300-500**: Agent masters landing, scores 150-200+
- **Episodes 500+**: Refined performance, consistent 200+ scores

## Tips

1. **GPU Acceleration**: Will automatically use CUDA if available
2. **Adjust Episodes**: Modify `EPISODES` variable based on performance
3. **Checkpoint Management**: Keep important checkpoints to resume training
4. **Hyperparameter Tuning**: Experiment with learning rate and epsilon decay

## Troubleshooting

**Issue**: Agent not improving
- Solution: Train for more episodes or adjust learning rate

**Issue**: Training too slow
- Solution: Reduce batch size or use GPU

**Issue**: Agent too conservative
- Solution: Increase epsilon decay to maintain exploration longer

## Next Steps

- Try different network architectures
- Implement Double DQN or Dueling DQN
- Add prioritized experience replay
- Experiment with different reward shaping

---

Made with 🤖 Deep Reinforcement Learning
