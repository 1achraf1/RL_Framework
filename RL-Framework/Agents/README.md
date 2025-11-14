# Agents

This directory contains implementations of various Reinforcement Learning agents, ranging from classical tabular methods to modern deep learning approaches.

All agents are designed to work with the custom Grid World environment and follow consistent interfaces for easy comparison and experimentation.

## 🧠 Agent Implementations

A detailed look at the agents available in this repository.

### Q-Learning

Classical Q-learning implementations for discrete state-action spaces.

#### Tabular Q-Learning

Standard tabular Q-learning with epsilon-greedy exploration.

**Features:**
- Epsilon-greedy exploration strategy
- Configurable learning rate and discount factor
- Support for decaying exploration
- Training and evaluation modes

### Linear Q-Learning

Q-learning using linear function approximation for state-value estimation.


### Deep Q-Network (DQN) 

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


### Proximal Policy Optimization (PPO)

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


## 📊 Agent Comparison Guide

| Agent | State Space | Action Space | Sample Efficiency | Stability | Complexity |
|-------|-------------|--------------|-------------------|-----------|------------|
| Q-Learning (Tabular) | Small | Discrete | Low | High | Low |
| Q-Learning (Linear) | Medium | Discrete | Medium | High | Low |
| Monte Carlo | Small-Medium | Discrete | Low | Medium | Low |
| DQN | Large | Discrete | Medium | Medium | Medium |
| PPO | Any | Any | High | High | High |


## 📚 References

- Sutton & Barto - *Reinforcement Learning: An Introduction*
- Mnih et al. - *Playing Atari with Deep Reinforcement Learning* (DQN)
- Schulman et al. - *Proximal Policy Optimization Algorithms* (PPO)
