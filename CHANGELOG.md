# Changelog

## [Unreleased]
### Added
- PPO agent (`src/ppo_agent.py`, `src/ppo_config.py`, `train_ppo.py`) with GAE and a clipped-surrogate objective, hyperparameters verified against RL Baselines3 Zoo's tuned LunarLander-v3 config
- Head-to-head DQN vs PPO evaluation harness (`evaluate_agents.py`) — PPO solves the environment (avg 245, 96% greedy solve rate) where DQN does not (avg 133, 56% solve rate, high variance)
- PPO unit tests covering GAE correctness, network output shapes, and update-step convergence (`tests/test_ppo.py`)
- PPO rollout recording (`record_ppo_videos.py`)

### Fixed
- `torch.load` checkpoint loading (DQN and PPO) broken by PyTorch 2.6's `weights_only=True` default
- `requirements.txt` was missing torch/gymnasium/matplotlib/tqdm/imageio, making the repo unreproducible from a clean install
- CI workflow now installs the actual training dependencies instead of only `pytest numpy`

### Removed
- Leftover AI-assistant bootstrap files (`GITHUB_SETUP.md`, `PUSH_INSTRUCTIONS.md`, `github_setup.sh`, `push_to_github.sh`)

### Added (previous)
- Experience replay buffer with capacity management
- DQN hyperparameter configuration
- Episode statistics tracker with rolling mean
- GitHub Actions CI
- README with benchmark results

## [0.1.0] - 2026-01-06
### Added
- Initial DQN implementation for LunarLander-v2
- Training notebooks
