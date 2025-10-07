import numpy as np

class LinearQAgent:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        """
        Linear Q-Learning Agent with feature-based representation.

        Parameters
        ----------
        env : GridWorld
            The environment instance.
        gamma : float
            Discount factor.
        alpha : float
            Learning rate.
        epsilon : float
            Initial exploration probability.
        epsilon_min : float
            Minimum exploration probability.
        epsilon_decay : float
            Multiplicative decay for epsilon after each episode.
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # One-hot features: (state_row * width + state_col) * 4 + action
        self.num_features = env.height * env.width * 4
        self.w = np.zeros(self.num_features)

    def featurize(self, state, action):
        """Return one-hot feature vector for (state, action)."""
        r, c = state
        feat = np.zeros(self.num_features)
        idx = (r * self.env.width + c) * 4 + action
        feat[idx] = 1.0
        return feat

    def q_value(self, state, action):
        """Compute Q-value = w · φ(state, action)."""
        return np.dot(self.w, self.featurize(state, action))

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(4)
        return np.argmax([self.q_value(state, a) for a in range(4)])

    def learn(self, num_episodes=1000, max_steps=100, random_start=False, moving_goal=False, verbose=True):
        """
        Train the agent using linear Q-learning with function approximation.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to train.
        max_steps : int
            Max steps per episode.
        random_start : bool
            If True, randomize starting position each episode.
        moving_goal : bool
            If True, randomize goal each episode (env must support set_random_goal()).
        verbose : bool
            Print training progress.
        """
        for ep in range(num_episodes):
            if random_start:
                while True:
                    rand_pos = (np.random.randint(0, self.env.height),
                                np.random.randint(0, self.env.width))
                    if rand_pos not in self.env.obstacles and rand_pos not in self.env.goals:
                        self.env.set_start_position(rand_pos)
                        break

            if moving_goal and hasattr(self.env, "set_random_goal"):
                self.env.set_random_goal()

            self.env.reset()
            state = self.env.agent.pos
            done, steps = False, 0

            while not done and steps < max_steps:
                action = self.choose_action(state)
                phi = self.featurize(state, action)

                _, reward, done = self.env.step(action)
                next_state = self.env.agent.pos

                # Target = reward + γ max_a' Q(s',a')
                if done:
                    q_next = 0
                else:
                    q_next = max(self.q_value(next_state, a) for a in range(4))

                td_error = (reward + self.gamma * q_next) - self.q_value(state, action)
                self.w += self.alpha * td_error * phi

                state = next_state
                steps += 1

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose and (ep + 1) % max(1, num_episodes // 10) == 0:
                print(f"Episode {ep+1}/{num_episodes}, epsilon={self.epsilon:.3f}")

    def act(self, state):
        """Choose greedy action (no exploration)."""
        return np.argmax([self.q_value(state, a) for a in range(4)])

    def get_q_grid(self):
        """Return Q-values in grid form for visualization."""
        q_grid = np.zeros((self.env.height, self.env.width, 4))
        for r in range(self.env.height):
            for c in range(self.env.width):
                for a in range(4):
                    q_grid[r, c, a] = self.q_value((r, c), a)
        return q_grid
