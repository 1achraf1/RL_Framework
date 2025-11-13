project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import numpy as np
import sys
import os
from tqdm import tqdm
from env.gridworld import GymGridWorld
from Agents.monte_carlo.MCAgent import MonteCarloAgent

NUM_EPISODES = 50000
GAMMA = 0.99
EPSILON = 0.1
env = GymGridWorld(stochastic=False, max_episode_steps=100)
obs, info = env.reset()

agent = MonteCarloAgent(
    action_space_n=env.action_space.n,
    gamma=GAMMA,
    epsilon=EPSILON
)

print(f"Training Monte Carlo Agent for {NUM_EPISODES} episodes...")

for ep in tqdm(range(NUM_EPISODES)):
    episode_data = []
    obs, info = env.reset()
    state = info['agent_pos']
    done = False

    while not done:
        # Get action from agent
        action = agent.get_action(state)
        
        # Take action in environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Store the experience
        episode_data.append((state, action, reward))
        
        # Update state for next loop
        state = info['agent_pos']
        
    agent.update(episode_data)

print("Training complete.")

policy_grid = agent.get_policy_grid(env)

print("\nRendering Optimal Policy (calculated by Monte Carlo)...")
env.render_policy(policy_grid)
env.close()
