RL-Framework

This repository is a collection of various Reinforcement Learning algorithms implemented from scratch, as well as benchmark implementations using stable-baselines3. All algorithms are tested on a custom Gymnasium-compatible GridWorld environment.

Project Structure

The repository is organized into distinct modules:

RL-Framework/
├── Agents/               # From-scratch model-free agents (MC, Q-Learning, DQN)
│   ├── monte_carlo/
│   ├── q_learning/
│   └── ...
│
├── DP/                   # From-scratch model-based solvers (Dynamic Programming)
│   ├── value_iteration.py
│   └── policy_iteration.py
│
├── env/                  # The custom GridWorld environment
│   └── gridworld.py
│
├── stable_baseline3_exemples/ # Benchmark agent implementations (PPO, A2C, DQN)
│   ├── train_a2c.py
│   ├── train_dqn.py
│   └── train_ppo.py
│
├── Notebooks/            # Jupyter notebooks for experiments and visualization
├── Utils/                # Helper scripts, plotting functions, etc.
│
├── README.md             # You are here
└── requirements.txt      # Project dependencies


🚀 Getting Started

Follow these steps to get the project up and running on your local machine.

Prerequisites

This project requires Python 3.8+ and pip.

Installation

Clone the repo:

git clone [https://github.com/your_username/RL-Framework.git](https://github.com/your_username/RL-Framework.git)
cd RL-Framework


Create a virtual environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install dependencies:

pip install -r requirements.txt


A minimal requirements.txt would look like this:

numpy
matplotlib
gymnasium
stable-baselines3[extra]
tqdm


How to Run

You can run algorithms from any of the three main folders.

1. Dynamic Programming (Model-Based)

These solvers compute the optimal policy directly by using the environment's transition model.

# Run Value Iteration
python DP/value_iteration.py

# Run Policy Iteration
python DP/policy_iteration.py


2. From-Scratch Agents (Model-Free)

These agents learn the optimal policy by interacting with the environment over many episodes.

# Run Monte Carlo
python Agents/monte_carlo/train.py

# Run Q-Learning (example)
python Agents/q_learning/train.py


3. Stable Baselines 3 Examples

These are high-performance implementations used for benchmarking.

# Run PPO
python stable_baseline3_exemples/train_ppo.py

# Run A2C
python stable_baseline3_exemples/train_a2c.py
