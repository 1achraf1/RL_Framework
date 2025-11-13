project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
import sys
import os
import time
from env.gridworld import GymGridWorld

def policy_evaluation(policy, V, model, states, gamma=0.99, theta=1e-6):
  
    while True:
        delta = 0
        for s in states:
            v_old = V.get(s, 0.0)
          
            a = policy[s]

          
            v_new = 0.0
            if a in model[s]: 
                for (prob, next_s, reward, done) in model[s][a]:
                    next_v = V.get(next_s, 0.0) 
                    v_new += prob * (reward + gamma * next_v)
            
            V[s] = v_new
            delta = max(delta, abs(v_old - V[s]))
          
        if delta < theta:
            break
    return V

def policy_improvement(V, model, states, gamma=0.99):
    
    new_policy = {}
    
    for s in states:
        action_values = []
        for a in model[s]:
            q_sa = 0.0
            for (prob, next_s, reward, done) in model[s][a]:
                next_v = V.get(next_s, 0.0)
                q_sa += prob * (reward + gamma * next_v)
            action_values.append(q_sa)
            
        # Find the best action: argmax_a Q(s, a)
        best_action = np.argmax(action_values)
        new_policy[s] = best_action
        
    return new_policy

def policy_iteration(env, gamma=0.99, theta=1e-6):
    
    

    model = env.get_transition_model()
    states = list(model.keys())
  
    if not states:
        print("Error: No states found in transition model. Did you call env.reset()?")
        return {}, {}
      
    V = {}
    policy = {s: np.random.choice(list(model[s].keys())) for s in states if model[s]}
    
    print("Running Policy Iteration...")
    start_time = time.time()
  
    i = 0
    while True:
        i += 1
        V = policy_evaluation(policy, V, model, states, gamma, theta)
        new_policy = policy_improvement(V, model, states, gamma)
        if policy == new_policy:
            break
        
        policy = new_policy
        
    print(f"Policy Iteration converged in {i} iterations ({time.time() - start_time:.2f}s)")
    
    return V, policy

# testing
env = GymGridWorld(stochastic=True)
env.reset() 

V, policy_dict = policy_iteration(env)

print("\nPolicy Iteration Complete.")
# print("Optimal Value Function (V):")
# for s, v in sorted(V.items()): print(f"  {s}: {v:.2f}")
# print("\nOptimal Policy (dict):")
# print(policy_dict)

policy_grid = np.full((env.height, env.width), -1, dtype=int) # -1 for obstacles/goals
for (r, c), action in policy_dict.items():
    policy_grid[r, c] = action
    
print("\nRendering Optimal Policy (calculated by Policy Iteration)...")
env.render_policy(policy_grid)
env.close()
