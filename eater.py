from random import randrange
import gym
from gym import spaces
import numpy as np
import cv2

class Position():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_tuple(self):
        return self.y, self.x
    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        else:
            return False


class EaterEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    # Enums
    AGENT = 1
    TARGET = 2
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    # Rendering Constants
    CELL_SIZE = 10
    AGENT_COLOR = [0, 255, 0]
    TARGET_COLOR = [255, 0, 0]

    def __init__(self):
        super(EaterEnv, self).__init__()
        self.grid_size = 60
        self.grid = np.zeros((self.grid_size,self.grid_size), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                            shape=(4,), dtype=np.uint8)

    # Moves the agent on the grid
    def move_agent(self, action):
        if action == self.UP:
            if self.agent_pos.y != 0:
                self.grid[self.agent_pos.to_tuple()] = 0
                self.agent_pos.y -= 1
                self.grid[self.agent_pos.to_tuple()] = self.AGENT
        elif action == self.DOWN:
            if self.agent_pos.y != self.grid_size - 1:
                self.grid[self.agent_pos.to_tuple()] = 0
                self.agent_pos.y += 1
                self.grid[self.agent_pos.to_tuple()] = self.AGENT
        elif action == self.LEFT:
            if self.agent_pos.x != 0:
                self.grid[self.agent_pos.to_tuple()] = 0
                self.agent_pos.x -= 1
                self.grid[self.agent_pos.to_tuple()] = self.AGENT
        elif action == self.RIGHT:
            if self.agent_pos.x != self.grid_size - 1:
                self.grid[self.agent_pos.to_tuple()] = 0
                self.agent_pos.x += 1
                self.grid[self.agent_pos.to_tuple()] = self.AGENT

    def step(self, action):
        self.moves_left -= 1
        reward = 0
        done = False
        info = {}

        # Move
        self.move_agent(action)

        # Check if the agent got the reward
        if self.agent_pos == self.target_pos:
            reward += self.grid_size * 3
            done = True

        # Out of moves
        elif self.moves_left <= 0:
            reward -= self.grid_size / 5
            done = True

        # Move penalty
        else:
            pass
            # reward -= 0.5

        observation = np.array([*self.agent_pos.to_tuple(), *self.target_pos.to_tuple()], dtype=np.uint8)
        return observation, reward, done, info

    def reset(self):

        self.moves_left = self.grid_size * 3

        # Create the grid
        self.grid = np.zeros((self.grid_size,self.grid_size), dtype=np.uint8)

        # Spawn agent
        agent_x, agent_y = randrange(0, self.grid_size), randrange(0, self.grid_size)
        self.agent_pos = Position(agent_x, agent_y)
        self.grid[self.agent_pos.to_tuple()] = self.AGENT

        # Spawn target
        while True:
            target_x, target_y = randrange(0, self.grid_size), randrange(0, self.grid_size)
            self.target_pos = Position(target_x, target_y)
            if self.agent_pos != self.target_pos:
                self.grid[self.target_pos.to_tuple()] = self.TARGET
                break

        # Return obs
        observation = np.array([*self.agent_pos.to_tuple(), *self.target_pos.to_tuple()], dtype=np.uint8)
        return observation

    def render(self, mode='human'):

        # Create the board
        board = np.zeros([self.grid_size * self.CELL_SIZE, self.grid_size * self.CELL_SIZE, 3], dtype=np.uint8)

        # Draw agent
        y = self.agent_pos.y * self.CELL_SIZE
        x = self.agent_pos.x * self.CELL_SIZE
        board[y:y+self.CELL_SIZE, x:x+self.CELL_SIZE] = self.AGENT_COLOR

        # Draw target
        y = self.target_pos.y * self.CELL_SIZE
        x = self.target_pos.x * self.CELL_SIZE
        board[y:y+self.CELL_SIZE, x:x+self.CELL_SIZE] = self.TARGET_COLOR
        
        cv2.imshow("Eater!", np.uint8(board))
        cv2.waitKey(100)
