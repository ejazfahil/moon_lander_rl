# 🚀 Moon Lander RL — Deep Q-Network for LunarLander

> Training a Deep Q-Network agent to land a spacecraft in OpenAI Gymnasium's `LunarLander`, with periodic checkpointing, recorded rollouts, and a modular RL component library.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-LunarLander--v3-0081A7)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white)

---

**Status:** Working DQN implementation with a genuine training run committed (`training_progress.png`, 10 checkpoints, 11 rollout videos). The current agent improves substantially over ~500 episodes but does **not** yet cross the LunarLander "solved" threshold (mean reward ≥ 200) within that budget — see [Results](#results). Double/Dueling DQN and Prioritized Experience Replay exist as scaffolding in `src/` and are on the roadmap, not yet wired into the trained agent.

---

## Overview

`LunarLander` is a classic continuous-state, discrete-action control problem: an 8-dimensional observation (position, velocity, angle, leg contacts) and 4 actions (do nothing, fire left / main / right engine). The environment is considered "solved" at a 100-episode average reward of **200**. This project implements a from-scratch **Deep Q-Network (DQN)** in PyTorch to learn a landing policy.

## Methodology

**Algorithm — Deep Q-Network (Mnih et al., 2015):**

- **Q-network:** a 3-layer MLP, `state(8) → FC(128) → ReLU → FC(128) → ReLU → Q-values(4)`.
- **Target network:** a periodically synced copy (every 10 episodes) for stable TD targets.
- **Experience replay:** a uniform `deque`-backed buffer (capacity 100k) sampled in minibatches of 64.
- **Loss:** MSE on the Bellman target `r + γ · maxₐ Qₜₐᵣ𝓰ₑₜ(s′, a) · (1 − done)`.
- **Exploration:** ε-greedy with exponential decay (ε: 1.0 → 0.01, decay 0.995).
- **Optimizer:** Adam, learning rate 1e-3, γ = 0.99.

**Training loop** (`train.py`, mirrored in `moon_lander_training.ipynb`): runs up to 500 episodes, syncs the target network and checkpoints periodically, tracks the rolling 100-episode average, and stops early if that average reaches 200.

```
┌──────────────┐   ε-greedy    ┌──────────────┐   (s,a,r,s′,done)   ┌────────────────┐
│ Environment  │ ───action───► │  DQN Agent   │ ──────store───────► │ Replay Buffer  │
│ LunarLander  │ ◄──reward───  │ (Q + target) │ ◄─────sample──────  │   (100k cap)   │
└──────────────┘               └──────┬───────┘                     └────────────────┘
                                      │ MSE TD-loss → Adam → backprop
                                      ▼
                          target network sync every 10 ep
```

## Tech Stack & Tools

| Area | Tools |
|------|-------|
| RL environment | **Gymnasium** (`LunarLander-v3`, Box2D) |
| Deep learning | **PyTorch** (`nn`, `optim`, autograd) |
| Numerics & viz | **NumPy**, **Matplotlib** |
| Video capture | `imageio`, Gymnasium `rgb_array` rendering |
| Tooling | **pytest**, `tqdm`, **Makefile** |

## Results

The committed `training_progress.png` is from a real 500-episode DQN run:

- The 100-episode **moving average reward rises from roughly −200 (random policy) to about +50 to +75** by episode ~500 — a clear, steady learning trend.
- Individual episodes intermittently exceed the **+200 solved line** in the later stages, but the *running average* has not yet stabilized above 200 within this 500-episode budget, so the environment is **not reported as solved**.

![Training progress](training_progress.png)

Reproducible artifacts in the repo:

- **Checkpoints:** `checkpoints/checkpoint_ep{50…450}.pth` and `checkpoints/final_ep499.pth` (full agent + optimizer state).
- **Rollout videos:** `videos/episode_{050…499}_run_{1,2}.mp4` plus `perfect_landing.mp4`, showing policy evolution across training.

> No "solved-at-episode-N" or fixed final-score figures are claimed beyond what the training curve above supports.

## Project Structure

```
moon_lander_rl/
├── train.py                     # DQN agent + training loop (entry point)
├── test.py                      # evaluate a trained agent
├── record_videos.py             # render rollouts to MP4
├── moon_lander_training.ipynb   # notebook version of the training pipeline
├── training_progress.png        # genuine training reward curve
├── src/                         # modular RL components
│   ├── replay_buffer.py         # uniform experience replay
│   ├── priority_buffer.py       # prioritized replay (scaffolding for PER)
│   ├── epsilon_scheduler.py     # ε-greedy decay schedules
│   ├── normalizer.py            # observation normalization
│   ├── checkpoint.py, metrics.py, config.py, seed_utils.py
├── tests/                       # pytest suite (agent, buffer, epsilon, metrics, checkpoint)
├── checkpoints/                 # saved agent checkpoints (ep50 … ep499)
├── videos/                      # recorded landing rollouts
└── requirements.txt
```

## Key Features

- **From-scratch DQN** with target network and experience replay in clean PyTorch.
- **Resumable training** — checkpoints store network, target, optimizer, ε, and score history.
- **Reproducibility helpers** — `seed_utils.py` and a typed `DQNConfig` dataclass.
- **Visual evidence** — recorded MP4 rollouts at multiple training stages.
- **Tested components** — unit tests for the replay buffer, ε-scheduler, metrics, and checkpointing.

## Getting Started

```bash
git clone https://github.com/ejazfahil/moon_lander_rl.git
cd moon_lander_rl
pip install -r requirements.txt      # plus: pip install gymnasium[box2d] torch matplotlib imageio

# train from scratch (up to 500 episodes, early-stops if solved)
python train.py

# evaluate / record a trained agent
python test.py
python record_videos.py

# run the component tests
pytest tests/
```

## Challenges

- LunarLander reward is high-variance; the 100-episode average is the meaningful signal, and vanilla DQN converges slowly and noisily near the solved threshold.
- Stability hinges on target-network sync cadence and ε-decay — small changes visibly shift the learning curve.

## Future Work

- Wire **Double DQN** and **Dueling DQN** heads into the trained agent to reduce Q-value overestimation.
- Activate **Prioritized Experience Replay** via the existing `priority_buffer.py`.
- Extend the training budget / tune hyperparameters to push the moving average reliably above 200 and report a solved-at-episode figure.
- Add seed-averaged learning curves with confidence bands.

## Conclusion

A clean, well-instrumented DQN baseline for LunarLander with honest, reproducible training evidence — and a modular foundation (PER, schedulers, normalizers) ready for the standard DQN improvements that should carry the agent past the solved threshold.
