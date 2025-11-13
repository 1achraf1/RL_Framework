project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
import sys
import os
import time
from env.gridworld import GymGridWorld

def value_iteration(env, gamma=0.99, theta=1e-6):
    
    # Get the environment's transition model
    model = env.get_transition_model()

    states = list(model.keys())
    if not states:
        print("Error: No states found in transition model. Did you call env.reset()?")
        return {}, {}
    V = {} 
    
    print("Running Value Iteration...")
    start_time = time.time()
    while True:
        delta = 0
        for s in states:
            v_old = V.get(s, 0.0)
            action_values = []
            for a in model[s]:
                q_sa = 0.0
                for (prob, next_s, reward, done) in model[s][a]:
                    next_v = V.get(next_s, 0.0) 
                    q_sa += prob * (reward + gamma * next_v)
                action_values.append(q_sa)
            V[s] = max(action_values) if action_values else 0.0
            delta = max(delta, abs(v_old - V[s]))
            
        if delta < theta:
            break
            
    print(f"Value Iteration converged in {time.time() - start_time:.2f}s")

    policy = {}
    for s in states:
        action_values = []
        for a in model[s]:
            q_sa = 0.0
            for (prob, next_s, reward, done) in model[s][a]:
                next_v = V.get(next_s, 0.0)
                q_sa += prob * (reward + gamma * next_v)
            action_values.append(q_sa)

        policy[s] = np.argmax(action_values)
        
    return V, policy


#testing
env = GymGridWorld(stochastic=True)
env.reset()

V, policy_dict = value_iteration(env)

print("\nValue Iteration Complete.")
# print("Optimal Value Function (V):")
# for s, v in sorted(V.items()): print(f"  {s}: {v:.2f}")
# print("\nOptimal Policy (dict):")
# print(policy_dict)

#Visualize the Policy
policy_grid = np.full((env.height, env.width), -1, dtype=int) # -1 for obstacles/goals
for (r, c), action in policy_dict.items():
    policy_grid[r, c] = action
    
print("\nRendering Optimal Policy (calculated by Value Iteration)...")
env.render_policy(policy_grid)
env.close()
