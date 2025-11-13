project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time
import sys
import os
from env.gridworld import GymGridWorld
TRAIN_HEIGHT = 40
TRAIN_WIDTH = 40

env = make_vec_env(
    GymGridWorld,
    n_envs=1,
    env_kwargs=dict(
        height=TRAIN_HEIGHT,
        width=TRAIN_WIDTH,
        max_episode_steps=200  
    )
)

model = PPO("CnnPolicy", env, verbose=1,learning_rate = 0.1,n_steps = 1000, n_epochs = 100,gamma = 0.9)

print("--- Starting Training ---")
model.learn(total_timesteps=50000)
print("--- Training Finished ---")

print("--- Testing Trained Agent ---")

test_env = GymGridWorld(
    height=TRAIN_HEIGHT,
    width=TRAIN_WIDTH,
    max_episode_steps=200,
    render_mode="human"
)
obs, info = test_env.reset()

try:
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        time.sleep(0.05)  # Pause for 50 milliseconds
        if terminated or truncated:
            print("Episode finished. Resetting.")
            obs, info = test_env.reset()  # Reset the single test_env

except KeyboardInterrupt:
    print("\nTest interrupted by user.")
finally:
    test_env.close()
