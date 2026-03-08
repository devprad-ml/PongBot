# PongBot 🏓

A Pong AI agent trained from scratch using Deep Q-Learning (DQN) in PyTorch. Two agents trained by playing against each other  no human demonstrations, no pre-built environments. Play against it yourself.

Built by [Pradyumnn](https://github.com/devprad-ml)

---

## Demo

| Mode | Controls |
|------|----------|
| Single Player (vs AI) | W / S to move your paddle |
| Two Players | W / S and ↑ / ↓ |

---

## How It Works

### The Problem

Standard Pong has no pre-defined strategy. The agent has to figure out, purely from trial and error, that moving toward the ball is good and missing the ball is bad. This is a classic reinforcement learning problem  the agent learns by interacting with the environment and receiving rewards.

### Deep Q-Learning (DQN)

Q-Learning is an RL algorithm where the agent learns a **Q-value** for every (state, action) pair which in layman's terms mean, "how good is it to take this action in this situation?"

The problem with classic Q-Learning is that it stores Q-values in a table, which breaks down when the state space is continuous (like this game, where the ball and paddles can be at any position). **Deep Q-Learning** solves this by replacing the table with a neural network that approximates Q-values.

The core update rule is the **Bellman equation:**

```
Q(s, a) = reward + γ * max(Q(s', a'))
```

Where:
- `s` = current state
- `a` = action taken
- `reward` = immediate reward received
- `γ` (gamma) = discount factor, i.e how much the agent values future rewards
- `s'` = next state after taking the action

At each training step, the network's prediction is compared to this target and the error (loss) is minimized using backpropagation.

---

## Architecture

### Neural Network (`dqn_agent.py`)

A fully connected feedforward network:

```
Input (6) → Linear(128) → ReLU → Linear(128) → ReLU → Output (3)
```

- **Input:** 6 features (state vector)
- **Hidden layers:** 2 × 128 neurons with ReLU activation
- **Output:** 3 Q-values, one per action

### State Space

The agent observes 6 normalized values at every frame:

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | Left paddle Y position | ÷ screen height |
| 1 | Right paddle Y position | ÷ screen height |
| 2 | Ball X position | ÷ screen width |
| 3 | Ball Y position | ÷ screen height |
| 4 | Ball X velocity | ÷ max speed |
| 5 | Ball Y velocity | ÷ max speed |

Normalization keeps all values between -1 and 1, which helps the neural network train stably.

### Action Space

Each agent has 3 discrete actions:

| Action | Meaning |
|--------|---------|
| 0 | Stay (do nothing) |
| 1 | Move up |
| 2 | Move down |

### Reward Function

| Event | Reward |
|-------|--------|
| Hit the ball | +1 |
| Win a point | +5 |
| Lose a point | -5 |

The reward shaping encourages the agent to both track the ball actively (+1 per hit) and to care about winning points (+5/-5), rather than just passively waiting.

---

## Key Techniques

### Experience Replay

Rather than learning from each frame immediately, transitions `(state, action, reward, next_state, done)` are stored in a **replay buffer** (capacity: 50,000). During training, random batches of 64 transitions are sampled. This breaks the correlation between consecutive frames and stabilizes training.

### Epsilon-Greedy Exploration

The agent uses an **epsilon-greedy policy** to balance exploration and exploitation:

- With probability **ε**: take a random action (explore)
- With probability **1-ε**: take the best known action (exploit)

Epsilon starts at 1.0 (fully random) and decays by a factor of 0.995 per step down to a minimum of 0.05. This means the agent explores heavily early on and gradually shifts to exploiting what it has learned.

### Self-Play Training

Both the left and right agents are trained simultaneously against each other. This creates an **auto-curriculum** as one agent improves, it provides a harder opponent for the other, pushing both to keep improving without any human-designed difficulty progression.

---

## Training

Two agents trained against each other for **258 episodes** using self-play. Training logs (episode rewards, epsilon values, loss, max Q-values, and outcome) are saved to `logs/training_log.csv` for analysis.

Training supports **checkpointing**  if interrupted, it saves both models and resumes from the last episode automatically.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-3 |
| Gamma (discount) | 0.99 |
| Epsilon start | 1.0 |
| Epsilon decay | 0.995 |
| Epsilon min | 0.05 |
| Replay buffer size | 50,000 |
| Batch size | 64 |
| Optimizer | Adam |
| Loss function | MSE (TD error) |

---

## Project Structure

```
PongBot/
├── dqn_agent.py        # DQN neural network and agent (replay buffer, training loop)
├── pong_env.py         # Custom Pong environment (state, actions, rewards, physics)
├── human_ai_pong.py    # Play against the trained AI
├── training.py         # Self-play training loop with checkpointing and CSV logging
├── pongboard.py        # Two-player Pong (no AI, for testing)
├── left_agent.pth      # Trained left agent weights
├── right_agent.pth     # Trained right agent weights (used by human_ai_pong.py)
└── meta.txt            # Last saved training episode (for checkpoint resume)
```

---

## Getting Started

### Prerequisites

```bash
pip install torch pygame numpy
```

### Play Against the AI

```bash
python human_ai_pong.py
```

Press **1** for single player (you vs AI) or **2** for two players.

### Train From Scratch

```bash
python training.py
```

Training saves checkpoints automatically. If interrupted, it resumes from where it left off.

---

## What I Learned

- Implementing DQN from scratch forces you to understand every component  replay buffer, Bellman targets, epsilon decay rather than letting a library abstract it away
- Self-play is a surprisingly effective training signal. The agents only need each other  no human demonstrations or hardcoded strategies
- Reward shaping matters. An early version with only win/lose rewards (+5/-5) trained much slower because the agent got no feedback for the majority of frames where no point was scored. Adding +1 per hit significantly sped up learning
- Normalization of the state vector is not optional  without it, training was unstable and the agents failed to converge
