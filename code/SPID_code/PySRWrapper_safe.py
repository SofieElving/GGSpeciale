from typing import Optional, Tuple

import numpy as np
from joblib import load, dump

from pysr import PySRRegressor 

#   WOOOOOOOOOOOOOOOO
MAX_INSULIN_ACTION = 5.0


def proposed_insulin_to_raw_action(insulin):
    """
    Convert predicted insulin dose [0, 5] back to env action [-1, 1].
    Inverse of:
        insulin = 5 * exp(4 * (a - 1))
    """
    u = np.asarray(insulin, dtype=np.float32)

    min_insulin = MAX_INSULIN_ACTION * np.exp(-8.0)
    u = np.clip(u, min_insulin, MAX_INSULIN_ACTION)

    raw_action = 1.0 + np.log(u / MAX_INSULIN_ACTION) / 4.0
    raw_action = np.clip(raw_action, -1.0, 1.0)

    return raw_action.astype(np.float32)
#   WOOOOOOOOOOOOOOOO

#Wrapper around our extracted decision tree, mostly so that we can use the sb policy evaluator
class PySRWrapper:
    def __init__(self, sr: PySRRegressor):
        self.sr = sr

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        a = self.sr.predict(obs)  # could be (n_envs,) or scalar
        a = np.asarray(a, dtype=np.float32).reshape(-1, 1)  # (n_envs, 1)
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
        # TODO: implement info print here. Complexity....
        print("Ain't been done")
        print(self.sr)


class PySRPolicy:
    def __init__(self, env, scale=True, **kwargs):
        self.shape = int(np.prod(env.action_space.shape))

        # Save action-space bounds so we can clip before env.step()
        self.action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(1, -1)
        self.action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(1, -1)


        self.policy_list = [
            PySRWrapper(PySRRegressor(**kwargs))
            for _ in range(self.shape)
        ]

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        print(f"obs: {obs}")
        obs = np.asarray(obs)

        # Ensure obs is 2D: (n_envs, obs_dim)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        elif obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)

        preds = []

        for policy in self.policy_list:
            # IMPORTANT:
            # policy.predict(obs) now predicts INSULIN DOSE, not raw PPO action.
            p = policy.predict(obs)
            p = np.asarray(p, dtype=np.float32).reshape(-1)

            if not np.all(np.isfinite(p)):
                policyeq = policy.sr.get_best()["equation"]
                print(
                    f"WARNING: non-finite PySR insulin dose replaced. "
                    f"obs={obs}. eq={policyeq}. raw_pred={p}"
                )

                p = np.nan_to_num(
                    p,
                    nan=0.0,
                    posinf=MAX_INSULIN_ACTION,
                    neginf=0.0,
                )

            # PySR predicts insulin dose, so clip dose to [0, 5]
            p = np.clip(p, 0.0, MAX_INSULIN_ACTION)

            preds.append(p)

        # Shape: (n_envs, action_dim)
        dose_actions = np.stack(preds, axis=1)

        # HARD-CODED TEST:
        # PySR predicts insulin dose [0, 5].
        # Convert it back to raw env action [-1, 1].
        raw_actions = proposed_insulin_to_raw_action(dose_actions)

        print(f"dose action: {dose_actions}")
        print(f"raw : {raw_actions}")

        # Clip to env action space
        raw_actions = np.clip(raw_actions, self.action_low, self.action_high)

        print(f"raw 2: {raw_actions}")

        return raw_actions.astype(np.float32), state

    def fit(self, x, y, weights=None):
        # y must now be teacher insulin dose, not teacher raw PPO action.
        for policy, actions in zip(self.policy_list, y.T):
            policy.fit(x, actions, weights)

    def print_info(self):
        for i, policy in enumerate(self.policy_list):
            print(f"\nAction dimension {i}:")
            policy.print_info()

    def save(self, path):
        dump(self, path)

    def load(path):
        policy = load(path)
        print("Policy loaded")
        return policy

# class PySRPolicy:
#     def __init__(self, env, **kwargs):
#         #self.shape = env.action_space.shape[0]
#         self.shape = int(np.prod(env.action_space.shape))

#         self.policy_list = [
#             PySRWrapper(PySRRegressor(**kwargs))
#             for _ in range(self.shape)
#         ]

#     def predict(self, obs, state=None, episode_start=None, deterministic=True):
#         obs = np.asarray(obs)

#         # Ensure obs is 2D: (n_envs, obs_dim)
#         # if obs.ndim == 1:
#         #     obs = obs.reshape(1, -1)
        
#         if obs.ndim == 1:
#             obs = obs.reshape(1, -1)
#         elif obs.ndim > 2:
#             obs = obs.reshape(obs.shape[0], -1)

#         preds = []
#         for policy in self.policy_list:
#             p = policy.predict(obs)
#             p = np.asarray(p).reshape(-1)   # force shape (n_envs,)
#             preds.append(p)

#             if np.isnan(p).any(): 
#                 policyeq=policy.sr.get_best()["equation"]
#                 print(obs)
#                 raise Exception(f"Action is none. obs:{obs}. eq = {policyeq}")
            
#             if np.isinf(p).any(): 
#                 raise Exception("Action is none")
            


#         # Stack into shape (n_envs, action_dim)
#         actions = np.stack(preds, axis=1)
#         return actions, state

#     def fit(self, x, y, weights=None):
#         for policy, actions in zip(self.policy_list, y.T):
#             policy.fit(x, actions, weights)

#     def print_info(self):
#         # TODO: print model summary
#         pass

#     def save(self, path):
#         # TODO: figure out how to save (and load) model correctly, such that it is each to reload and use
#         # Potentially just do as above, and save model as a PySRPolicy class??
#         dump(self, path)

#     def load(path):
#         policy = load(path)
#         print("Policy loaded")
#         return policy

















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