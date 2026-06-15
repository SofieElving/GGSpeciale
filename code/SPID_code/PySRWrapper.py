from typing import Callable, Optional
import numpy as np
from joblib import load, dump
from pysr import PySRRegressor


ActionTransform = Callable[[np.ndarray], np.ndarray]


def identity_action_transform(actions: np.ndarray) -> np.ndarray:
    return actions


def proposed_insulin_to_raw_action(insulin, max_insulin_action=5):
    """
    Convert predicted insulin dose [0, max_insulin_action]
    back to env action [-1, 1].

    Inverse of:
        insulin = max_insulin_action * exp(4 * (a - 1))
    """
    if max_insulin_action <= 0:
        raise ValueError("max_insulin_action must be positive.")

    u = np.asarray(insulin, dtype=np.float32)

    min_insulin = max_insulin_action * np.exp(-8.0)

    u = np.nan_to_num(
        u,
        nan=min_insulin,
        posinf=max_insulin_action,
        neginf=min_insulin,
    )

    u = np.clip(u, min_insulin, max_insulin_action)
    raw_action = 1.0 + np.log(u / max_insulin_action) / 4.0
    raw_action = np.clip(raw_action, -1.0, 1.0)

    return raw_action.astype(np.float32)

class PySRWrapper:
    def __init__(self, sr: PySRRegressor):
        self.sr = sr

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        a = self.sr.predict(obs)
        a = np.asarray(a, dtype=np.float32).reshape(-1, 1)
        return a

    def fit(self, x, y, weights=None):
        self.sr.fit(x, y, weights=weights)

    @classmethod
    def load(cls, path: str):
        clf = load(path)
        return PySRWrapper(clf)

    def save(self, path: str):
        dump(self.sr, path)

    def print_info(self):
        print("Ain't been done")
        print(self.sr)


class PySRPolicy:
    def __init__(
        self,
        env,
        action_transform: Optional[ActionTransform] = None,
        **kwargs
    ):
        self.shape = env.action_space.shape[0]
        self.action_space = env.action_space

        self.action_transform = action_transform or identity_action_transform

        self.policy_list = [
            PySRWrapper(PySRRegressor(**kwargs))
            for _ in range(self.shape)
        ]

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(obs, dtype=np.float32)

        # Ensure obs is 2D: (n_envs, obs_dim)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        preds = []
        for policy in self.policy_list:
            p = policy.predict(obs)
            p = np.asarray(p, dtype=np.float32).reshape(-1)
            preds.append(p)

        # Shape: (n_envs, action_dim)
        actions = np.stack(preds, axis=1).astype(np.float32)

        # Apply optional post-processing
        actions = self.action_transform(actions)
        actions = np.asarray(actions, dtype=np.float32)

        # Safety: ensure final shape is still valid
        actions = actions.reshape(obs.shape[0], self.shape)

        return actions, state

    def fit(self, x, y, weights=None):
        for policy, actions in zip(self.policy_list, y.T):
            policy.fit(x, actions, weights)

    def print_info(self):
        for i, policy in enumerate(self.policy_list):
            print(f"Action dimension {i}:")
            policy.print_info()

    def save(self, path):
        dump(self, path)

    @classmethod
    def load(cls, path):
        return load(path)


# import numpy as np
# from gymnasium import spaces
# from joblib import dump, load
# from typing import Optional, Tuple

# class PySRWrapper:
#     def __init__(self, sr, action_space: Optional[spaces.Box] = None):
#         self.sr = sr
#         self.action_space = action_space  # pass env.action_space

#     def predict(self, obs, state=None, episode_start=None, deterministic=True):
#         obs = np.asarray(obs, dtype=np.float32)

#         # Ensure 2D: (n_envs, obs_dim)
#         if obs.ndim == 1:
#             obs_2d = obs.reshape(1, -1)
#             single = True
#         elif obs.ndim == 2:
#             obs_2d = obs
#             single = False
#         else:
#             raise ValueError(f"Unexpected obs shape: {obs.shape}")

#         a = self.sr.predict(obs_2d)
#         a = np.asarray(a, dtype=np.float32)

#         # Force (n_envs, 1)
#         a = np.squeeze(a)
#         if a.ndim == 0:
#             a = a.reshape(1, 1)
#         elif a.ndim == 1:
#             a = a.reshape(-1, 1)
#         else:
#             a = a.reshape(a.shape[0], -1)[:, :1]

#         # If SR outputs unbounded values, squash/clip.
#         # (You can swap tanh for clip if you prefer.)
#         a = np.tanh(a)

#         # Scale to env bounds if provided (recommended)
#         if self.action_space is not None:
#             low = np.asarray(self.action_space.low, dtype=np.float32).reshape(1, -1)
#             high = np.asarray(self.action_space.high, dtype=np.float32).reshape(1, -1)
#             a = low + (a + 1.0) * 0.5 * (high - low)
#             a = np.clip(a, low, high)
#         else:
#             a = np.clip(a, -1.0, 1.0)

#         # For non-VecEnv usage, return shape (1,)
#         if single:
#             return a[0], state
#         return a, state

#     @classmethod
#     def load(cls, path: str, action_space: Optional[spaces.Box] = None):
#         sr = load(path)
#         return cls(sr, action_space=action_space)

#     def save(self, path: str):
#         dump(self.sr, path)

#     def print_info(self):
#         print(self.sr)