import numpy as np
import gymnasium as gym
from gymnasium import spaces

# -----------------------------
# Continuous wrapper for Acrobot
# -----------------------------
class ContinuousAcrobot(gym.Wrapper):
    """
    Turns Acrobot-v1 (Discrete(3)) into a pseudo-continuous action environment (Box(1,))
    by binning a continuous action into {0,1,2}.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        a = float(np.asarray(action).reshape(-1)[0])

        # Map continuous scalar -> discrete action {0,1,2}
        if a < -1.0 / 3.0:
            discrete_action = 0
        elif a < 1.0 / 3.0:
            discrete_action = 1
        else:
            discrete_action = 2

        return self.env.step(discrete_action)


def make_continuous_acrobot():
    env = gym.make("Acrobot-v1")
    return ContinuousAcrobot(env)