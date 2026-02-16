import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ContinuousMountainCar(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        return self.env.step(action)


def make_continuous_mountaincar():
    env = gym.make("MountainCarContinuous-v0")
    return ContinuousMountainCar(env)

