import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, Patch
from IPython.display import HTML
import gymnasium as gym
from gymnasium import spaces

class Agent:
    def __init__(self, start_pos=(0, 0)):
        self.start_pos = tuple(start_pos)
        self.pos = tuple(start_pos)

    def set_position(self, pos):
        self.pos = tuple(pos)

    def reset(self):
        self.pos = self.start_pos

    def move(self, action):
        r, c = self.pos
        if action == 0:  # Up
            return (r - 1, c)
        elif action == 1:  # Right
            return (r, c + 1)
        elif action == 2:  # Down
            return (r + 1, c)
        elif action == 3:  # Left
            return (r, c - 1)
        return (r, c)
class GridWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, height=5, width=5, start=None, goal=None,
                 reward_goal=10, reward_step=-1, obstacles=None, obstacle_penalty=-10,
                 stochastic=True, moving_goal=False, moving_obstacles=False,
                 goal_move_frequency=5, obstacle_move_frequency=3,
                 render_mode=None, max_episode_steps=100):

        super().__init__()

        self.height, self.width = height, width
        self.stochastic = stochastic
        self.moving_goal = moving_goal
        self.moving_obstacles = moving_obstacles
        self.goal_move_frequency = goal_move_frequency
        self.obstacle_move_frequency = obstacle_move_frequency

        # Store initial configurations
        self._start_config = start
        self._goal_config = goal
        self._obstacles_config = obstacles

        
        self.agent = Agent()
        self.start = (0, 0)
        self.goals = []
        self.obstacles = set()
        self.initial_obstacles = []

        # Rewards
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.obstacle_penalty = obstacle_penalty
        self._steps = 0
        self.max_episode_steps = max_episode_steps

        # Visualization Attributes 
        self.cmap = colors.ListedColormap(["white", "gray", "blue", "green", "black"])
        self.norm = colors.BoundaryNorm([0, 1, 2, 3, 4, 5], self.cmap.N)

        #
        self.action_space = spaces.Discrete(4)  # 0:U, 1:R, 2:D, 3:L


        
        self.observation_space = spaces.Box(
            low=0,
            high=255,  
            shape=(self.height, self.width, 1),
            dtype=np.uint8 
        )

        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.fig, self.ax = None, None  # For 'human' mode canvas
        self.agent_circle, self.agent_eye1, self.agent_eye2, self.title_text = None, None, None, None

    def _random_free_cell(self, exclude=[]):
        """Find a random cell not in exclude or obstacles."""
        exclude_set = set(exclude) | self.obstacles
        max_attempts = self.height * self.width
        for _ in range(max_attempts):
            cell = (self.np_random.integers(0, self.height),
                    self.np_random.integers(0, self.width))
            if cell not in exclude_set:
                return cell
        # Fallback
        for r in range(self.height):
            for c in range(self.width):
                cell = (r, c)
                if cell not in exclude_set:
                    return cell
        return (0, 0)

    def _move_goal(self):
        if not self.moving_goal: return
        exclude = [self.agent.pos] + list(self.obstacles) + self.goals
        new_goal = self._random_free_cell(exclude=exclude)
        self.goals = [new_goal]

    def _move_obstacles(self):
        if not self.moving_obstacles or not self.obstacles: return
        new_obstacles = set()
        for obs in self.obstacles:
            adjacent = [(obs[0] - 1, obs[1]), (obs[0], obs[1] + 1),
                        (obs[0] + 1, obs[1]), (obs[0], obs[1] - 1)]
            valid_moves = []
            for new_pos in adjacent:
                r, c = new_pos
                if (0 <= r < self.height and 0 <= c < self.width and
                        new_pos != self.agent.pos and new_pos not in self.goals and
                        new_pos not in new_obstacles):
                    valid_moves.append(new_pos)
            if valid_moves and self.np_random.random() < 0.7:
                new_obstacles.add(valid_moves[self.np_random.choice(len(valid_moves))])
            else:
                new_obstacles.add(obs)
        self.obstacles = new_obstacles

    def _get_obs(self):
        """Get the current grid observation."""
        
        grid = np.full((self.height, self.width), 255, dtype=np.uint8)

        
        for (orow, ocol) in self.obstacles:
            grid[orow, ocol] = 0
        
        for (gr, gc) in self.goals:
            grid[gr, gc] = 64
        
        ar, ac = self.agent.pos
        grid[ar, ac] = 128
        return np.expand_dims(grid, axis=-1)

    def _get_info(self):
        """Get auxiliary information."""
        return {
            "agent_pos": self.agent.pos,
            "goal_pos": self.goals[0] if self.goals else None,
            "steps": self._steps
        }

    def reset(self, seed=None, options=None):
        """Resets the environment to a starting state."""
        super().reset(seed=seed)

        if self._obstacles_config is not None:
            self.initial_obstacles = [tuple(o) for o in self._obstacles_config]
        else:
            num_obstacles = self.np_random.integers(1, 11)
            self.initial_obstacles = []
            self.obstacles = set()
            for _ in range(num_obstacles):
                obs = self._random_free_cell(exclude=self.initial_obstacles)
                self.initial_obstacles.append(obs)
        self.obstacles = set(self.initial_obstacles)

        if self._start_config is not None:
            self.start = tuple(self._start_config)
        else:
            self.start = self._random_free_cell(exclude=list(self.obstacles))

        assert self.start not in self.obstacles, "Start position can't be an obstacle"

        if self._goal_config is None:
            self.initial_goals = [self._random_free_cell(exclude=[self.start] + list(self.obstacles))]
        elif isinstance(self._goal_config, list):
            self.initial_goals = [tuple(g) for g in self._goal_config]
        else:
            self.initial_goals = [tuple(self._goal_config)]
        self.goals = self.initial_goals.copy()

        for g in self.goals:
            assert g not in self.obstacles, f"Goal {g} can't be an obstacle"

        self.agent.set_position(self.start)
        self.agent.start_pos = self.start
        self._steps = 0

        observation = self._get_obs()

        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """Execute one time step in the environment."""
        self._steps += 1

        if self.moving_obstacles and self._steps % self.obstacle_move_frequency == 0:
            self._move_obstacles()

        # Determine actual action
        if self.stochastic:
            actions = [0, 1, 2, 3]  # 0:U, 1:R, 2:D, 3:L
            if action == 0:
                probabilities = [0.8, 0.1, 0, 0.1]
            elif action == 1:
                probabilities = [0.1, 0.8, 0.1, 0]
            elif action == 2:
                probabilities = [0, 0.1, 0.8, 0.1]
            else:
                probabilities = [0.1, 0, 0.1, 0.8]
            actual_action = self.np_random.choice(actions, p=probabilities)
        else:
            actual_action = action

        proposed_pos = self.agent.move(actual_action)
        proposed_r, proposed_c = proposed_pos

        is_valid = (0 <= proposed_r < self.height and
                    0 <= proposed_c < self.width and
                    proposed_pos not in self.obstacles)

        if is_valid:
            self.agent.pos = proposed_pos

        terminated = False
        if not is_valid:
            reward = self.obstacle_penalty
        elif self.agent.pos in self.goals:
            reward = self.reward_goal
            terminated = True
        else:
            reward = self.reward_step

        truncated = self._steps >= self.max_episode_steps

        observation = self._get_obs()
        info = self._get_info()
        info["actual_action"] = actual_action

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    # VISUALIZATION METHODS 

    def _prepare_plot(self):
        """Creates the figure, axis, and static background for visualizations."""
        if self.render_mode == 'human':
            if self.fig is None:  # First render call in human mode
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(10, 10))
            ax = self.ax
            ax.clear()  # Clear dynamic elements, we redraw static ones
        else:  # rgb_array or None
            fig, ax = plt.subplots(figsize=(10, 10))

        # Draw empty cells
        for r in range(self.height):
            for c in range(self.width):
                rect = Rectangle((c, r), 1, 1, linewidth=2.5,
                                 edgecolor='#2c3e50', facecolor='#ecf0f1')
                ax.add_patch(rect)

        # Draw obstacles
        for obs in self.obstacles:
            rect = Rectangle((obs[1], obs[0]), 1, 1, linewidth=2.5,
                             edgecolor='#2c3e50', facecolor='#34495e')
            ax.add_patch(rect)
            ax.plot([obs[1] + 0.2, obs[1] + 0.8], [obs[0] + 0.2, obs[0] + 0.8],
                    'k-', linewidth=2, alpha=0.3)
            ax.plot([obs[1] + 0.2, obs[1] + 0.8], [obs[0] + 0.8, obs[0] + 0.2],
                    'k-', linewidth=2, alpha=0.3)

        # Draw goals
        for goal in self.goals:
            glow = Rectangle((goal[1] - 0.05, goal[0] - 0.05), 1.1, 1.1, linewidth=0,
                             facecolor='#2ecc71', alpha=0.3, zorder=1)
            ax.add_patch(glow)
            rect = Rectangle((goal[1], goal[0]), 1, 1, linewidth=2.5,
                             edgecolor='#27ae60', facecolor='#2ecc71', zorder=2)
            ax.add_patch(rect)
            ax.text(goal[1] + 0.5, goal[0] + 0.5, '*',
                    fontsize=50, ha='center', va='center',
                    color='#f39c12', fontweight='bold', zorder=3)

        # Set axis properties
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match array indexing (0,0 at top-left)
        ax.set_xticks(range(self.width + 1))
        ax.set_yticks(range(self.height + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.set_facecolor('#bdc3c7')

        if self.render_mode == 'human':
            return self.fig, self.ax
        else:
            return fig, ax

    def render(self):

        if self.render_mode is None:
            return

        fig, ax = self._prepare_plot()
        mode = "Stochastic" if self.stochastic else "Deterministic"
        title = f"Steps: {self._steps} | Mode: {mode}"

        pos = self.agent.pos
        center_x = pos[1] + 0.5
        center_y = pos[0] + 0.5

        if self.render_mode == 'human':
            # Create patches once
            if self.agent_circle is None:
                self.agent_circle = Circle((center_x, center_y), 0.4, fc='#3498db', ec='#2c3e50',
                                           linewidth=3, zorder=10)
                self.agent_eye1 = Circle((center_x - 0.1, center_y - 0.1), 0.08, fc='white', zorder=11)
                self.agent_eye2 = Circle((center_x + 0.1, center_y - 0.1), 0.08, fc='white', zorder=11)
                self.title_text = ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

            self.agent_circle.center = (center_x, center_y)
            self.agent_eye1.center = (center_x - 0.1, center_y - 0.1)
            self.agent_eye2.center = (center_x + 0.1, center_y - 0.1)
            self.title_text.set_text(title)

            ax.add_patch(self.agent_circle)
            ax.add_patch(self.agent_eye1)
            ax.add_patch(self.agent_eye2)

            plt.pause(1 / self.metadata["render_fps"])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        elif self.render_mode == 'rgb_array':
            agent_circle = Circle((center_x, center_y), 0.4, fc='#3498db', ec='#2c3e50',
                                  linewidth=3, zorder=10)
            agent_eye1 = Circle((center_x - 0.1, center_y - 0.1), 0.08, fc='white', zorder=11)
            agent_eye2 = Circle((center_x + 0.1, center_y - 0.1), 0.08, fc='white', zorder=11)
            ax.add_patch(agent_circle)
            ax.add_patch(agent_eye1)
            ax.add_patch(agent_eye2)
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data

    def close(self):
        """Close any open resources."""
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig, self.ax = None, None
            self.agent_circle, self.agent_eye1, self.agent_eye2, self.title_text = None, None, None, None

    def render_policy(self, policy):
        """Renders the grid world with arrows indicating the policy."""
        fig, ax = self._prepare_plot()

        action_arrows = {0: (0, -0.35), 1: (0.35, 0), 2: (0, 0.35), 3: (-0.35, 0)}
        action_colors = {0: '#e74c3c', 1: '#3498db', 2: '#9b59b6', 3: '#f39c12'}

        for r in range(self.height):
            for c in range(self.width):
                state = (r, c)
                if state in self.obstacles or state in self.goals:
                    continue
                action = policy[r, c]
                dx, dy = action_arrows.get(action, (0, 0))
                color = action_colors.get(action, 'black')
                ax.arrow(c + 0.5, r + 0.5, dx, dy,
                         head_width=0.25, head_length=0.2, fc=color, ec=color,
                         linewidth=2.5, zorder=5)

        ax.set_title("Agent's Learned Policy", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

    def run_and_animate_episode(self, agent, episode_num=1, jupyter=False):
        """Runs a single episode and creates a smooth animation."""
        fig, ax = self._prepare_plot()
        history = []
        # Use reset() to get the *actual* first observation
        obs, info = self.reset()
        pos = info['agent_pos']
        done = False
        cumulative_reward = 0

        while not done:
            state = pos[0] * self.width + pos[1]
            action = np.argmax(agent.q_table[state])
            obs, reward, terminated, truncated, info = self.step(action)
            done = terminated or truncated  # Gym uses two flags
            cumulative_reward += reward
            next_pos = info['agent_pos']

            history.append({'pos': pos, 'reward': cumulative_reward, 'done': done})
            pos = next_pos

            if done:
                history.append({'pos': pos, 'reward': cumulative_reward, 'done': done})
                break
            if len(history) > self.height * self.width * 2:
                print("Animation stopped: Episode seems to be in an infinite loop.")
                break

        agent_circle = Circle((0.5, 0.5), 0.4, fc='#3498db', ec='#2c3e50',
                              linewidth=3, zorder=10)
        agent_eye1 = Circle((0.4, 0.4), 0.08, fc='white', zorder=11)
        agent_eye2 = Circle((0.6, 0.4), 0.08, fc='white', zorder=11)
        title_text = ax.set_title(f"Episode: {episode_num} | Step: 0 | Reward: 0.0",
                                  fontsize=16, fontweight='bold', pad=20)

        def init():
            ax.add_patch(agent_circle)
            ax.add_patch(agent_eye1)
            ax.add_patch(agent_eye2)
            return agent_circle, agent_eye1, agent_eye2, title_text

        def update(frame_num):
            data = history[frame_num]
            pos, reward, done = data['pos'], data['reward'], data['done']
            center_x = pos[1] + 0.5
            center_y = pos[0] + 0.5
            agent_circle.center = (center_x, center_y)
            agent_eye1.center = (center_x - 0.1, center_y - 0.1)
            agent_eye2.center = (center_x + 0.1, center_y - 0.1)
            if done and frame_num == len(history) - 1:  # Only on last frame
                agent_circle.set_facecolor('#2ecc71')
                title_text.set_text(f"GOAL REACHED! | Steps: {frame_num} | Reward: {reward:.1f}")
                title_text.set_color('#27ae60')
            else:
                title_text.set_text(f"Episode: {episode_num} | Step: {frame_num} | Reward: {reward:.1f}")
            return agent_circle, agent_eye1, agent_eye2, title_text

        anim = FuncAnimation(fig, update, frames=len(history),
                             init_func=init, blit=True, interval=300, repeat=False)
        plt.tight_layout()

        if jupyter:
            html = HTML(anim.to_jshtml())
            plt.close(fig)
            return html
        else:
            plt.show()
            return anim

    def test_agent_live(self, agent, max_steps=100, speed=0.4):
        """Opens a window showing the agent following its policy in real-time."""
        fig, ax = self._prepare_plot()
        agent_circle = Circle((0.5, 0.5), 0.4, fc='#3498db', ec='#2c3e50',
                              linewidth=3, zorder=10)
        agent_eye1 = Circle((0.4, 0.4), 0.08, fc='white', zorder=11)
        agent_eye2 = Circle((0.6, 0.4), 0.08, fc='white', zorder=11)
        ax.add_patch(agent_circle)
        ax.add_patch(agent_eye1)
        ax.add_patch(agent_eye2)
        trail_circles = []

        obs, info = self.reset()
        pos = info['agent_pos']
        agent_circle.center = (pos[1] + 0.5, pos[0] + 0.5)
        agent_eye1.center = (pos[1] + 0.4, pos[0] + 0.4)
        agent_eye2.center = (pos[1] + 0.6, pos[0] + 0.4)

        step, cumulative_reward, terminated, truncated = 0, 0, False, False
        title_text = ax.set_title(f"Testing Agent | Step: {step} | Reward: {cumulative_reward:.1f}",
                                  fontsize=16, fontweight='bold', pad=20,
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.ion()
        plt.show()

        while not (terminated or truncated) and step < max_steps:
            plt.pause(speed)
            state = pos[0] * self.width + pos[1]
            action = np.argmax(agent.q_table[state])

            obs, reward, terminated, truncated, info = self.step(action)
            cumulative_reward += reward
            step += 1
            next_pos = info['agent_pos']

            center_x, center_y = next_pos[1] + 0.5, next_pos[0] + 0.5
            agent_circle.center = (center_x, center_y)
            agent_eye1.center = (center_x - 0.1, center_y - 0.1)
            agent_eye2.center = (center_x + 0.1, center_y - 0.1)
            title_text.set_text(f"Testing Agent | Step: {step} | Reward: {cumulative_reward:.1f}")

            fig.canvas.draw()
            fig.canvas.flush_events()
            pos = next_pos

        if terminated:  # Use 'terminated' for goal celebration
            title_text.set_text(f"GOAL REACHED! | Steps: {step} | Final Reward: {cumulative_reward:.1f}")
        else:  # Truncated or max_steps
            title_text.set_text(f"Max steps reached | Steps: {step} | Reward: {cumulative_reward:.1f}")

        fig.canvas.draw()
        plt.ioff()
        plt.show(block=True)

    def _get_obs_from_pos(self, agent_pos):
        """Get observation for a hypothetical agent position."""
        # Create a grid of 255 (white)
        grid = np.full((self.height, self.width), 255, dtype=np.uint8)
        # Add obstacles (0 = black)
        for (orow, ocol) in self.obstacles:
            grid[orow, ocol] = 0
        # Add goals (64 = dark gray)
        for (gr, gc) in self.goals:
            grid[gr, gc] = 64
        # Add agent (128 = gray)
        ar, ac = agent_pos
        grid[ar, ac] = 128
        # Add a channel dimension to match observation_space
        return np.expand_dims(grid, axis=-1)

if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    
    print("--- Checking Environment Compatibility ---")
    try:
        env_check = GymGridWorld()
        check_env(env_check)
        print("✅ Environment passed compatibility check!")
    except Exception as e:
        print(f"❌ Environment check failed: {e}")
    finally:
        env_check.close()

    print("\n--- Running Random Agent (Standard Gym Loop) ---")
    env = GymGridWorld(
        height=8,
        width=10,
        obstacles=None,
        stochastic=False,
        render_mode='human',
        max_episode_steps=50  # Add a step limit
    )

    try:
        obs, info = env.reset(seed=42)
        for i in range(100):  # Run for 100 steps total
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)

            if i == 0:
                print(f"Observation shape is correct: {obs.shape}")
                print(f"Observation dtype is correct: {obs.dtype}")

            if terminated or truncated:
                print(f"Episode finished after {info['steps']} steps. Resetting.")
                obs, info = env.reset()
    except KeyboardInterrupt:
        print("\nTest run interrupted by user.")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
    finally:
        env.close()
        print("--- Random Agent Test Finished ---")
