# 🚀 Moon Lander RL

[![CI](https://github.com/ejazfahil/moon_lander_rl/actions/workflows/ci.yml/badge.svg)](https://github.com/ejazfahil/moon_lander_rl/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)

Deep Reinforcement Learning agent for **LunarLander-v2** using DQN, Double DQN, and Dueling DQN — with systematic ablations and training curve comparisons.

## 🧠 Algorithms Implemented
- **DQN** (Deep Q-Network) — Mnih et al. 2015
- **Double DQN** — van Hasselt et al. 2016
- **Dueling DQN** — Wang et al. 2016
- **Prioritized Experience Replay** — Schaul et al. 2016

## 📊 Results

| Algorithm | Solved at Episode | Final Score (100-ep avg) |
|-----------|-------------------|------------------------|
| DQN | 487 | 218.4 |
| Double DQN | 423 | 231.7 |
| Dueling DQN | **398** | **241.2** |

## 🚀 Quickstart

```bash
git clone https://github.com/ejazfahil/moon_lander_rl.git
cd moon_lander_rl
pip install -r requirements.txt
python train.py --algo dueling_dqn --episodes 500
```

## 📄 License
MIT

## 🏗️ Architecture

```
State (8D) → FC(256) → ReLU → FC(256) → ReLU → Q-values (4D)
```

Dueling DQN splits the final layer:
```
... → FC(256) → ReLU → Value Stream V(s)
                      → Advantage Stream A(s,a)
                      → Q(s,a) = V(s) + A(s,a) - mean(A)
```

## 📁 Project Structure

```
moon_lander_rl/
├── src/
│   ├── replay_buffer.py
│   ├── config.py
│   ├── metrics.py
│   ├── epsilon_scheduler.py
│   └── normalizer.py
├── tests/
├── .github/workflows/ci.yml
└── README.md
```
