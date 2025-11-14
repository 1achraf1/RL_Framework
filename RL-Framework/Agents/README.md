# **Agents**
<br>
This directory contains implementations of various Reinforcement Learning agents, ranging from classical tabular methods to modern deep learning approaches.
Overview
All agents are designed to work with the custom Grid World environment and follow consistent interfaces for easy comparison and experimentation.
Agent Implementations
Q-Learning (q_learning/)
Classical Q-learning implementations for discrete state-action spaces.
Files:

q_learning.py - Standard tabular Q-learning with epsilon-greedy exploration
linear_q.py - Q-learning with linear function approximation

**Features** :

Epsilon-greedy exploration strategy
Configurable learning rate and discount factor
Support for decaying exploration
Training and evaluation modes

**Usage**:

pythonfrom Agents.q_learning.q_learning import QLearningAgent

agent = QLearningAgent(
    env=env,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
)

agent.train(num_episodes=1000)

Linear Q-Learning (LinearQLearning/)


Q-learning using linear function approximation for state-value estimation.
Features:

Feature engineering for grid world states
Linear function approximation
Suitable for larger state spaces
Gradient-based updates

**Best for**:

Environments where tabular methods are impractical
When you want interpretable feature weights
Bridging tabular and deep learning methods


Deep Q-Network (dqn/)


Deep learning-based Q-learning using neural networks.
Features:

Experience replay buffer
Target network for stability
Deep neural network for Q-value approximation
Handles large/continuous state spaces

Key Components:


Replay buffer for breaking correlation
Separate target network updated periodically
Adam optimizer for training
Epsilon-greedy exploration

**Best for**:


Complex environments with large state spaces
Problems requiring non-linear function approximation
When feature engineering is difficult


Monte Carlo Methods (monte_carlo/)
Monte Carlo reinforcement learning algorithms.
Features:



First-visit and every-visit MC
Episode-based learning
No bootstrapping (full returns)
On-policy and off-policy variants

**Characteristics**:


Requires episodic tasks
High variance, low bias
Simple and intuitive
No model required

**Best for**:


Episodic environments
When you want unbiased estimates
Exploring alternative approaches to temporal difference learning


Proximal Policy Optimization (ppo/)
Modern policy gradient method for both discrete and continuous action spaces.


Features:


Clipped surrogate objective
Actor-critic architecture
Generalized Advantage Estimation (GAE)
Multiple epochs per batch


**Advantages**:

State-of-the-art performance
Stable and reliable training
Works with continuous actions
Sample efficient


**Best for**:


Complex control tasks
When DQN struggles
Continuous action spaces
Production-grade applications


**Comparison Guide**
AgentState SpaceAction SpaceSample EfficiencyStabilityComplexityQ-LearningSmallDiscreteLowHighLowLinear QMediumDiscreteMediumHighLowDQNLargeDiscreteMediumMediumMediumMonte CarloSmall-MediumDiscreteLowMediumLowPPOAnyAnyHighHighHigh
General Agent Interface
Most agents follow this common interface:
pythonclass Agent:
    def __init__(self, env, **hyperparameters):
        """Initialize agent with environment and hyperparameters"""
        
    def select_action(self, state, eval_mode=False):
        """Select action given current state"""
        
    def update(self, state, action, reward, next_state, done):
        """Update agent based on experience"""
        
    def train(self, num_episodes):
        """Train agent for specified episodes"""
        
    def evaluate(self, num_episodes):
        """Evaluate agent performance"""
        
    def save(self, path):
        """Save agent parameters"""
        
    def load(self, path):
        """Load agent parameters"""
Choosing an Agent
Start with Q-Learning if:

Your state/action space is small
You're learning RL fundamentals
You want quick results and easy debugging


Use Linear Q-Learning if:


State space is too large for tabular methods
You want interpretable features
Environment has linear structure


Choose DQN if:


State space is very large
You have complex state representations
Tabular methods don't work


Try Monte Carlo if:


Tasks are episodic
You want to compare with TD methods
You need unbiased estimates


Go with PPO if:


You need state-of-the-art performance
Working with continuous actions
Building production systems
Other methods have failed


Training Tips


Start simple: Begin with Q-learning to understand the environment
Tune hyperparameters: Learning rate and exploration are critical
Monitor metrics: Track episode rewards, loss, and exploration rate
Use baselines: Compare against random and optimal policies
Visualize learning: Plot learning curves and policy evolution
Debug systematically: Verify environment rewards and state representations


Common Issues

Agent not learning:


Check reward structure
Verify state representation
Adjust learning rate
Increase exploration


Unstable training:


Reduce learning rate
Increase replay buffer (DQN)
Use target networks
Check for bugs in update logic


Slow convergence:


Tune exploration parameters
Check discount factor
Verify network architecture (DQN/PPO)
Consider reward shaping


References


Sutton & Barto - "Reinforcement Learning: An Introduction"
Mnih et al. - "Playing Atari with Deep Reinforcement Learning" (DQN)
Schulman et al. - "Proximal Policy Optimization Algorithms" (PPO)
