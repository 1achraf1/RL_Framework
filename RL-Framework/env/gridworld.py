import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class Agent:
    def __init__(self):
        self.start = None
        self.pos = None

    def set_position(self, start):
        self.start = tuple(start)

    def reset(self):
        self.pos = tuple(self.start)

    def move(self, action, height, width):
        r, c = self.pos
        if action == 0: r -= 1
        elif action == 1: c += 1
        elif action == 2: r += 1
        elif action == 3: c -= 1
        if 0 <= r < height and 0 <= c < width:
            self.pos = (r, c)
        return self.pos


class GridWorld:
    def __init__(self, height=5, width=5, start=None, goal=None,
                 reward_goal=10, reward_step=-1, obstacles=None, obstacle_penalty=-10):
        self.height, self.width = height, width
        self.obstacles = set(tuple(o) for o in obstacles) if obstacles else set()
        self.start = tuple(start) if start else self._random_free_cell(exclude=[])
        if goal is None:
            self.goals = [self._random_free_cell(exclude=[self.start] + list(self.obstacles))]
        elif isinstance(goal, list):
            self.goals = [tuple(g) for g in goal]
        else:
            self.goals = [tuple(goal)]

        self.agent = Agent()
        self.agent.set_position(self.start)

        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.obstacle_penalty = obstacle_penalty
        self._steps = 0

        self.cmap = colors.ListedColormap(["white", "blue", "green", "black"])
        self.norm = colors.BoundaryNorm([0, 2, 3, 4, 5], self.cmap.N)

    def _random_free_cell(self, exclude=[]):
        exclude_set = set(exclude)
        while True:
            cell = (np.random.randint(0, self.height), np.random.randint(0, self.width))
            if cell not in exclude_set and cell not in self.obstacles:
                return cell

    def set_start_position(self, pos):
        assert pos not in self.obstacles
        assert pos not in self.goals
        self.start = pos
        self.agent.set_position(pos)

    def reset(self):
        self.agent.reset()
        self._steps = 0
        return self._get_obs()

    def step(self, action):
        self._steps += 1
        new_pos = self.agent.move(action, self.height, self.width)
        if new_pos in self.goals:
            return self._get_obs(), self.reward_goal, True
        elif new_pos in self.obstacles:
            return self._get_obs(), self.obstacle_penalty, False
        else:
            return self._get_obs(), self.reward_step, False

    def _get_obs(self):
        grid = np.zeros((self.height, self.width), dtype=np.int8)
        for (r, c) in self.obstacles: grid[r, c] = 4
        for (r, c) in self.goals: grid[r, c] = 3
        ar, ac = self.agent.pos
        grid[ar, ac] = 2
        return grid

    def render(self, pause_time=0.3):
        grid = self._get_obs()
        plt.imshow(grid, cmap=self.cmap, norm=self.norm, extent=[0, self.width, self.height, 0])
        plt.xticks(range(self.width+1))
        plt.yticks(range(self.height+1))
        plt.grid(color="black", linewidth=1)
        plt.title(f"Steps: {self._steps}")
        plt.pause(pause_time)
