# Experiment Log

## Exp-001: Baseline DQN (2026-01-10)
- Config: lr=1e-3, gamma=0.99, batch=64, buffer=100k
- Result: Solved at episode 487, avg reward 218.4

## Exp-002: Double DQN (2026-02-15)
- Config: same as Exp-001 + target network decoupling
- Result: Solved at episode 423, avg reward 231.7

## Exp-003: Dueling DQN (2026-03-01)
- Config: same as Exp-002 + dueling architecture
- Result: Solved at episode 398, avg reward 241.2 ← Best

## Exp-004: Prioritized Replay (2026-04-01)
- Config: Dueling DQN + PER (alpha=0.6, beta=0.4)
- Result: Solved at episode 381, avg reward 244.8
