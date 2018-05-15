import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class PointReachEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):

        self.action_space = spaces.Box(low = np.array([-0.5,-0.5]), high = np.array([0.5,0.5]))
        self.observation_space = spaces.Box(np.array([-10, -10]), np.array([10,10]))

        self._seed()
        self.viewer = None
        self.state = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        action = action.clip(-0.5, 0.5)

        self.state += action
        self.state = self.state.clip(-10, 10)

        reward = -0.05 - np.min([np.linalg.norm(self.state - self.targets[0]), np.linalg.norm(self.state - self.targets[1])])
        if np.linalg.norm(self.state - self.targets[0]) < 0.8:
            reward += 25

        done = False

        self.current_action = np.copy(action)

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))

        self.targets = [np.array([9.5, 1.0]), np.array([-7.0, 1.0])]
        self.current_action = np.ones(2)
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 500
        screen_height = 500

        world_width = 20.0
        scale = screen_width/world_width
        offset = np.array([screen_height / 2.0, screen_width / 2.0])

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = rendering.make_circle(radius=0.2*scale)
            self.agent_transform = rendering.Transform(self.state * scale + offset)
            agent.add_attr(self.agent_transform)
            agent.set_color(0.5, 0.5, 1.0)
            self.viewer.add_geom(agent)

            target1 = rendering.make_circle(radius=0.2*scale)
            target1.add_attr(rendering.Transform(self.targets[0] * scale + offset))
            target1.set_color(0.5, 1.0, 0.5)
            target2 = rendering.make_circle(radius=0.2*scale)
            target2.add_attr(rendering.Transform(self.targets[1] * scale + offset))
            target2.set_color(0.5, 1.0, 0.5)
            self.viewer.add_geom(target1)
            self.viewer.add_geom(target2)

        if self.state is None: return None

        new_pos = self.state * scale + offset
        self.agent_transform.set_translation(new_pos[0], new_pos[1])
        self.actline = self.viewer.draw_line(start=self.state * scale + offset, end=(
                                                                                    self.state + self.current_action / np.linalg.norm(
                                                                                        self.current_action) * 0.5) * scale + offset)
        self.actline.set_color(1.0, 0.5, 0.5)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
