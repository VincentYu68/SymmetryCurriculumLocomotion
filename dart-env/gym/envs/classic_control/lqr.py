import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class LQREnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.action_space = spaces.Box(low = np.array([-1.0, -1.0, -1.0]), high = np.array([1.0, 1.0, 1.0]))
        self.observation_space = spaces.Box(np.array([-np.inf, -np.inf, -np.inf]), np.array([np.inf, np.inf, np.inf]))

        self.A = np.array([[1.01, 0.01, 0.0], [0.01, 1.01, 0.01], [0.0, 0.01, 1.01]])
        self.B = np.identity(3)
        self.Q = 1.0 * np.identity(3)
        self.R = np.identity(3)

        self._seed()
        self.viewer = None
        self.state = np.random.normal(0, 1, size=3)
        self.time = 0
        self.alive_bonus = 10.0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, u):
        x = self.state

        cost = np.dot(x, np.dot(self.Q, x)) + np.dot(u, np.dot(self.R, u)) - self.alive_bonus
        new_x = np.dot(self.A, x) + np.dot(self.B, u) + self.np_random.normal(0, 1, 3)

        self.state = new_x

        terminated = False

        if np.any(np.abs(new_x) > 5.0):
            terminated = True

        if self.time > 300:
            terminated = True

        self.time += 1

        return self._get_obs(), - cost, terminated, {}

    def _reset(self):
        self.state = self.np_random.normal(0, 1, size=3)
        self.last_u = None
        self.time = 0

        return self._get_obs()

    def _get_obs(self):
        return self.state

    def _render(self, mode='human', close=False):
        return None
