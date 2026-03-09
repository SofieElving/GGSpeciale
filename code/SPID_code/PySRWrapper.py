from typing import Optional, Tuple

import numpy as np
from joblib import load, dump

from pysr import PySRRegressor 


#Wrapper around our extracted decision tree, mostly so that we can use the sb policy evaluator
class PySRWrapper:
    def __init__(self, sr: PySRRegressor):
        self.sr = sr

    # def predict(
    #         self,
    #         observation: np.ndarray,
    #         state: Optional[Tuple[np.ndarray, ...]] = None,
    #         episode_start: Optional[np.ndarray] = None,
    #         deterministic: bool = False,
    # ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
    #     return self.sr.predict(observation), None

    
    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        a = self.sr.predict(obs)  # could be (n_envs,) or scalar
        a = np.asarray(a, dtype=np.float32).reshape(-1, 1)  # (n_envs, 1)
        a = np.clip(a, -1.0, 1.0)
        return a, state

    @classmethod
    def load(cls, path: str):
        clf = load(path)
        return PySRWrapper(clf)

    def save(self, path: str):
        dump(self.sr, path)

    def print_info(self):
        # TODO: implement info print here. Complexity....
        print("Ain't been done")
        # print(f"Max depth:\t{self..get_depth()}")
        # print(f"# Leaves:\t{self.tree.get_n_leaves()}")
        print(self.sr)




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