import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math


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



# class ContinuousCartPole(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
#         self.action_space.n = env.action_space.n

#     def step(self, action):
#         # action can be scalar, (1,), or (1,1) depending on SB3/VecEnv
#         a = float(np.asarray(action).squeeze())
#         discrete_action = 1 if a > 0.0 else 0
#         return self.env.step(discrete_action)



# def make_continuous_cartpole():
#     env = gym.make("CartPole-v1")
#     return ContinuousCartPole(env)





class ContinuousCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, max_episode_steps=500):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # Continuous action: one scalar in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Same observation space as CartPole: [x, x_dot, theta, theta_dot]
        high = np.array([4.8, np.finfo(np.float32).max, 0.418, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Physics constants (standard CartPole values)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * math.pi / 360

        self.state = None
        self.steps_beyond_terminated = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.steps_beyond_terminated = None
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        self.step_count += 1
        a = float(np.asarray(action, dtype=np.float32).squeeze())
        a = float(np.clip(a, -1.0, 1.0))
        force = a * self.force_mag

        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        reward = 1.0
        truncated = self.step_count >= self.max_episode_steps

        return self.state.copy(), reward, terminated, truncated, {}