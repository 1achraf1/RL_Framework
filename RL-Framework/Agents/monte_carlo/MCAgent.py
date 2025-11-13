import numpy as np
from collections import defaultdict

class MonteCarloAgent:

    def __init__(self, action_space_n, gamma=0.99, epsilon=0.1):
        """
        Args:
            action_space_n (int): The number of possible actions (e.g., 4).
            gamma (float): The discount factor.
            epsilon (float): The exploration rate for epsilon-greedy.
        """
        self.action_space_n = action_space_n
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = defaultdict(lambda: np.zeros(self.action_space_n))
        self.N = defaultdict(lambda: np.zeros(self.action_space_n))
        self.policy = {}

    def get_action(self, state):
       
        # Epsilon-greedy exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_n)
        else:
            # Act greedily
            return np.argmax(self.Q[state])

    def update(self, episode):
        G = 0.0 
        visited = set()
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # Use First-Visit: only update if this (s, a) pair
            # hasn't been seen *earlier* in this episode.
            if (state, action) not in visited:
                # Update counts and Q-value
                self.N[state][action] += 1
                
                # Incremental mean update
                # Q_new = Q_old + (1/N) * (G - Q_old)
                self.Q[state][action] += (1.0 / self.N[state][action]) * (G - self.Q[state][action])
                
                visited.add((state, action))

    def get_policy_grid(self, env):
       
        policy_grid = np.full((env.height, env.width), -1, dtype=int)
        for r in range(env.height):
            for c in range(env.width):
                state = (r, c)
                # Check if state is in Q-table (i.e., has been visited)
                if state in self.Q:
                    policy_grid[r, c] = np.argmax(self.Q[state])
        return policy_grid
