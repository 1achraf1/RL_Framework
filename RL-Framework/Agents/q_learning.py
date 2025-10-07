import numpy as np

class QLearningAgent:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        """
        Q-Learning Agent for GridWorld-like environments.

        Parameters
        ----------
        env : GridWorld
            Environment instance.
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

        # Initialize Q-table: (height, width, 4 actions)
        self.Q = np.zeros((env.height, env.width, 4))

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        r, c = state
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1, 2, 3])  # Explore
        return np.argmax(self.Q[r, c])  # Exploit

    def learn(self, num_episodes=1000, max_steps=200, random_start=False, verbose=True):
        """
        Train the agent using Q-learning.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to train for.
        max_steps : int
            Maximum steps per episode.
        random_start : bool
            If True, start each episode from a random free cell.
        verbose : bool
            Print training progress.
        """
        for ep in range(num_episodes):
            # Randomize start if requested
            if random_start:
                while True:
                    rand_pos = (np.random.randint(0, self.env.height),
                                np.random.randint(0, self.env.width))
                    if rand_pos not in self.env.obstacles and rand_pos not in self.env.goals:
                        break
                self.env.set_start_position(rand_pos)

            self.env.reset()
            state = self.env.agent.pos
            done, steps = False, 0

            while not done and steps < max_steps:
                r, c = state
                action = self.choose_action(state)
                _, reward, done = self.env.step(action)
                new_state = self.env.agent.pos
                nr, nc = new_state

                # Q-learning update
                best_next = np.max(self.Q[nr, nc])
                td_target = reward + self.gamma * best_next
                td_error = td_target - self.Q[r, c, action]
                self.Q[r, c, action] += self.alpha * td_error

                state = new_state
                steps += 1

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)

            if verbose and (ep + 1) % max(1, num_episodes // 10) == 0:
                print(f"Episode {ep+1}/{num_episodes}, epsilon={self.epsilon:.3f}")

    def act(self, state):
        """Choose the best action (greedy)."""
        r, c = state
        return np.argmax(self.Q[r, c])
