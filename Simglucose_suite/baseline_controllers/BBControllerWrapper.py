'''
Wrapper of the Simglucose BB-controller, to make it compatible with env_open.
Assumes non-normalized statespace, and returns an action mapped from true insulin dosis to [-1, 1]

Needs meal warning 0

'''

from simglucose.controller.basal_bolus_ctrller import BBController
from collections import namedtuple
import numpy as np

Observation = namedtuple("Observation", ["CGM", "CHO"])

def proposed_insulin_to_raw_action(insulin, max_insulin_action=5.0):
    min_insulin = max_insulin_action * np.exp(-8.0)

    u = np.asarray(insulin, dtype=np.float32)
    u = np.nan_to_num(u, nan=min_insulin, posinf=max_insulin_action, neginf=min_insulin)
    u = np.clip(u, min_insulin, max_insulin_action)

    raw = 1.0 + np.log(u / max_insulin_action) / 4.0
    return np.clip(raw, -1.0, 1.0).astype(np.float32)


class BBPolicy:
    def __init__(
        self,
        env,
        patient_name=None,
        max_insulin_action=5.0,
        normalize=False,
        cgm_index=0,
        meal_warning_index=3,
        meal_size_index=4,
    ):
        self.controller = BBController()
        self.env = env
        self.patient_name = patient_name
        self.max_insulin_action = float(max_insulin_action)
        self.normalize = bool(normalize)

        self.sample_time = float(getattr(env, "sample_time_min", 3.0))
        self.cgm_index = int(cgm_index)
        self.meal_warning_index = int(meal_warning_index)
        self.meal_size_index = int(meal_size_index)

        self.prev_meal_warning = 0.0

        self.patient_name = patient_name or env.unwrapped.env.patient_name

        if self.patient_name is None:
            raise ValueError(
                "Could not infer patient_name. Pass patient_name explicitly to BBPolicy."
            )

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(observation, dtype=np.float32)
        single_obs = obs.ndim == 1
        obs_batch = obs.reshape(1, -1) if single_obs else obs

        actions = []

        for obs_i in obs_batch:
            cgm = float(obs_i[self.cgm_index])
            meal_warning = float(obs_i[self.meal_warning_index])
            meal_size = float(obs_i[self.meal_size_index]) / self.sample_time

            if self.normalize:
                cgm *= 400.0
                # Only do this if your env_open normalizes meal size this way.
                # Otherwise remove this line.
                meal_size *= 120.0

            # Critical: only announce meal once, not during the whole warning window.
            meal = meal_size if meal_warning > 0.0 and self.prev_meal_warning <= 0.0 else 0.0
            self.prev_meal_warning = meal_warning

            bb_obs = Observation(CGM=cgm, CHO=meal_size)

            action_tuple = self.controller.policy(
                observation=bb_obs,
                reward = None,
                done = None,
                sample_time=self.sample_time,
                patient_name=self.patient_name,
                meal=meal,
            )

            insulin = float(action_tuple.basal + action_tuple.bolus)

            #print(f"BB Delivered amount {insulin}")

            raw_action = proposed_insulin_to_raw_action(
                insulin,
                max_insulin_action=self.max_insulin_action,
            )

            actions.append(float(np.asarray(raw_action).reshape(-1)[0]))

        actions = np.asarray(actions, dtype=np.float32).reshape(-1, 1)

        if single_obs:
            actions = actions[0]

        return actions, state
    
    def reset(self):
        self.prev_meal_warning = 0.0