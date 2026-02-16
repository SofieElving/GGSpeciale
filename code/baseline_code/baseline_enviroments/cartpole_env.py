import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ContinuousCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        # Convert continuous → discrete
        discrete_action = 1 if action[0] > 0 else 0
        return self.env.step(discrete_action)


def make_continuous_cartpole():
    env = gym.make("CartPole-v1")
    return ContinuousCartPole(env)