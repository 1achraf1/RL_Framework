# Agents

This directory contains implementations of various Reinforcement Learning agents, ranging from classical tabular methods to modern deep learning approaches.

All agents are designed to work with the custom Grid World environment and follow consistent interfaces for easy comparison and experimentation.

## 🧠 Agent Implementations

A detailed look at the agents available in this repository.

### Q-Learning (`q_learning/`)

Classical Q-learning implementations for discrete state-action spaces.

#### Tabular Q-Learning (`q_learning.py`)

Standard tabular Q-learning with epsilon-greedy exploration.

**Features:**
- Epsilon-greedy exploration strategy
- Configurable learning rate and discount factor
- Support for decaying exploration
- Training and evaluation modes

### Linear Q-Learning

Q-learning using linear function approximation for state-value estimation.

**Features:**
- Feature engineering for grid world states
- Linear function approximation
- Suitable for larger state spaces
- Gradient-based updates

**Best for:**
- Environments where tabular methods are impractical
- When you want interpretable feature weights
- Bridging tabular and deep learning methods

### Deep Q-Network (DQN) (`dqn/`)

Deep learning-based Q-learning using a neural network to approximate the Q-value function.

**Features:**
- Experience Replay buffer to decorrelate samples
- Separate Target Network for stable updates
- Deep neural network for Q-value approximation
- Handles large/continuous state spaces

**Key Components:**
- Replay buffer for breaking correlation
- Separate target network updated periodically
- Adam optimizer for training

**Best for:**
- Complex environments with large state spaces
- Problems requiring non-linear function approximation
- When feature engineering is difficult

### Monte Carlo Methods (`monte_carlo/`)

Episode-based learning methods that learn from complete returns.

**Features:**
- First-visit and every-visit MC
- Episode-based learning (no bootstrapping)
- On-policy and off-policy variants

**Characteristics:**
- Requires episodic tasks
- High variance, low bias
- Simple and intuitive
- No model required

**Best for:**
- Episodic environments
- When you want unbiased estimates
- Exploring alternatives to temporal difference learning

### Proximal Policy Optimization (PPO) (`ppo/`)

A modern, state-of-the-art policy gradient method known for its stability and sample efficiency.

**Features:**
- Clipped surrogate objective
- Actor-Critic architecture
- Generalized Advantage Estimation (GAE)
- Multiple epochs of training per batch

**Advantages:**
- State-of-the-art performance
- Stable and reliable training
- Works with both discrete and continuous action spaces
- Sample efficient

**Best for:**
- Complex control tasks
- When DQN struggles
- Continuous action spaces
- Production-grade applications

## 📊 Agent Comparison Guide

| Agent | State Space | Action Space | Sample Efficiency | Stability | Complexity |
|-------|-------------|--------------|-------------------|-----------|------------|
| Q-Learning (Tabular) | Small | Discrete | Low | High | Low |
| Q-Learning (Linear) | Medium | Discrete | Medium | High | Low |
| Monte Carlo | Small-Medium | Discrete | Low | Medium | Low |
| DQN | Large | Discrete | Medium | Medium | Medium |
| PPO | Any | Any | High | High | High |

## 🧬 General Agent Interface

Most agents adhere to the following common interface for consistency and easy swapping:

```python
class Agent:
    def __init__(self, env, **hyperparameters):
        """Initialize agent with environment and hyperparameters"""
        pass
        
    def select_action(self, state, eval_mode=False):
        """Select action given current state"""
        pass
        
    def update(self, state, action, reward, next_state, done):
        """Update agent based on experience (for online agents)"""
        pass
        
    def train(self, num_episodes):
        """Train agent for specified episodes"""
        pass
        
    def evaluate(self, num_episodes):
        """Evaluate agent performance"""
        pass
        
    def save(self, path):
        """Save agent parameters/model"""
        pass
        
    def load(self, path):
        """Load agent parameters/model"""
        pass
```

## 🧐 Choosing an Agent

**Start with Q-Learning (Tabular) if:**
- Your state/action space is small
- You're learning RL fundamentals
- You want quick results and easy debugging

**Use Q-Learning (Linear) if:**
- The state space is too large for tabular methods
- You want interpretable features
- The environment has a somewhat linear structure

**Choose DQN if:**
- The state space is very large (e.g., pixel data)
- You need to learn complex, non-linear representations
- Tabular/linear methods don't work

**Try Monte Carlo if:**
- Your tasks are strictly episodic
- You want to compare with TD methods
- You need unbiased estimates of returns

**Go with PPO if:**
- You need state-of-the-art performance
- You are working with continuous action spaces
- You are building production-level systems
- Other methods have failed or are too unstable

## 🛠️ Training & Debugging

### Training Tips

- **Start simple:** Begin with Tabular Q-learning to verify the environment's reward structure and dynamics
- **Tune hyperparameters:** Learning rate and exploration (epsilon) are critical. Start with common values and adjust
- **Monitor metrics:** Track total episode rewards, average reward over time, loss (for DQN/PPO), and exploration rate
- **Visualize learning:** Plot learning curves and, if possible, render the agent's policy in the environment

### Common Issues

**Agent not learning:**
- Check the reward structure. Is it too sparse?
- Verify your state representation
- Adjust the learning rate (try orders of 10)
- Increase exploration or adjust the decay schedule

**Unstable training (DQN/PPO):**
- Reduce the learning rate
- Increase the replay buffer size (DQN)
- Adjust the target network update frequency (DQN)
- Check for exploding/vanishing gradients

**Slow convergence:**
- Tune exploration parameters
- Check the discount factor (gamma)
- Verify the network architecture (DQN/PPO)
- Consider reward shaping (use with caution)

## 📚 References

- Sutton & Barto - *Reinforcement Learning: An Introduction*
- Mnih et al. - *Playing Atari with Deep Reinforcement Learning* (DQN)
- Schulman et al. - *Proximal Policy Optimization Algorithms* (PPO)
