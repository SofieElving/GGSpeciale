import gymnasium as gym
import numpy as np
from gymnasium import spaces


# class ContinuousCartPole(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

#     def action(self, action):
#         # action may be scalar, (1,), or (1,1)
#         a = float(np.asarray(action).squeeze())
#         return 1 if a > 0.0 else 0

#     def step(self, action):
#         # Convert continuous → discrete
#         discrete_action = 1 if action[0] > 0 else 0
#         return self.env.step(discrete_action)


# def make_continuous_cartpole():
#     env = gym.make("CartPole-v1")
#     return ContinuousCartPole(env)



class ContinuousCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space.n = env.action_space.n

    def step(self, action):
        # action can be scalar, (1,), or (1,1) depending on SB3/VecEnv
        a = float(np.asarray(action).squeeze())
        discrete_action = 1 if a > 0.0 else 0
        return self.env.step(discrete_action)



def make_continuous_cartpole():
    env = gym.make("CartPole-v1")
    return ContinuousCartPole(env)