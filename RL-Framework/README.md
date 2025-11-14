# RL-Framework

A comprehensive Reinforcement Learning framework featuring a custom Grid World environment compatible with Gymnasium, along with implementations of classical and modern RL algorithms.

## Overview

This project provides a modular framework for experimenting with reinforcement learning algorithms in a custom-built Grid World environment. The environment follows the Gymnasium API standards, making it easy to integrate with existing RL workflows and libraries.

## Features

- **Custom Grid World Environment**: Fully-featured grid world compatible with Gymnasium interface
- **Multiple RL Agents**: Implementations of various classical and deep RL algorithms
- **Dynamic Programming**: Value and Policy Iteration solvers
- **Educational Focus**: Clear, well-documented code suitable for learning and experimentation

## Project Structure

```
RL-Framework/
├── Agents/                          # RL agent implementations
│   ├── LinearQLearning/             # Linear function approximation Q-learning
│   ├── dqn/                         # Deep Q-Network agent
|   |   ├── DQNAgent.py           
|   |   └── train.py 
│   ├── monte_carlo/                 # Monte Carlo methods
|   |   ├── MCAgent.py
|   |   └── train.py      
│   ├── ppo/                         # Policy proximal optimization
|   |   ├── PPOAgent.py
|   |   └── train.py
│   ├── q_learning/                  # Tabular Q-learning
|   |   ├── Qlearning_Agent.py
|   |   └── train.py 
├── DynamicProgramming/              # DP algorithms for solving MDPs
│   ├── PolicyIteration.py           # Policy iteration algorithm
│   └── ValueIteration.py            # Value iteration algorithm
├── Notebooks/                       # Jupyter notebooks for experiments
├── Utils/                           # Utility functions and helpers
├── env/                             # Custom Grid World environment
├── stable_baseline3_examples/       # Examples using Stable-Baselines3
├── README.md                        # This file
└── __init__.py                      # Package initialization
```

## Environment

The Grid World environment is a custom implementation that:
- Follows the Gymnasium API specification
- Supports configurable grid sizes and obstacle placement
- Provides customizable reward structures
- Includes rendering capabilities for visualization
- Compatible with both tabular and function approximation methods

## Implemented Algorithms

### Tabular Methods
- **Q-Learning**: Standard tabular Q-learning with epsilon-greedy exploration
- **Monte Carlo**: First-visit
- **Dynamic Programming**: Value Iteration and Policy Iteration

### Function Approximation
- **Linear Q-Learning**: Q-learning with linear function approximation
- **Deep Q-Network (DQN)**: Deep learning-based Q-learning

### Policy Gradient Methods
- **PPO**: Proximal Policy Optimization for continuous and discrete action spaces

## Getting Started

### Prerequisites

```bash
pip install gymnasium numpy matplotlib
```

### Basic Usage

```python
import gymnasium as gym
from env import GridWorld

# Create environment
env = GridWorld()

# Reset environment
state, info = env.reset()

# Take action
next_state, reward, terminated, truncated, info = env.step(action)
```

### Training an Agent

```python
from Agents.q_learning import QLearningAgent

# Initialize agent
agent = QLearningAgent(env)

# Train agent
agent.train(num_episodes=1000)

# Evaluate agent
agent.evaluate(num_episodes=100)
```

## Notebooks

The `Notebooks/` directory contains Jupyter notebooks demonstrating:
- Environment usage and visualization
- Training different agents
- Comparing algorithm performance
- Hyperparameter tuning experiments

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Gymnasium (OpenAI Gym) for the environment interface standards
- Sutton & Barto's "Reinforcement Learning: An Introduction" for algorithm references
- Stable-Baselines3 for baseline comparisons

## Contact

For questions or suggestions, please open an issue in the repository.
