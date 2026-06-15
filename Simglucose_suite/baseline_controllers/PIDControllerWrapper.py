'''
Wrapper of the Simglucose BB-controller, to make it compatible with env_open.
Assumes non-normalized statespace, and returns an action mapped from true insulin dosis to [-1, 1]

Needs meal warning 0

'''

from simglucose.controller.pid_ctrller import PIDController
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


patient_weights = {
    "adult#001": [1.58E-04, 1.00E-07, 1.00E-02],
    "adult#002": [3.98E-04, 1.00E-07, 1.00E-02],
    "adult#003": [4.54E-10, 1.00E-07, 1.00E-02],
    "adult#004": [1.00E-04, 1.00E-07, 3.98E-03],
    "adult#005": [3.02E-04, 1.00E-07, 1.00E-02],
    "adult#006": [2.51E-04, 2.51E-07, 1.00E-02],
    "adult#007": [1.22E-04, 3.49E-07, 2.87E-03],
    "adult#008": [1.00E-04, 1.00E-07, 1.00E-02],
    "adult#009": [1.00E-04, 1.00E-07, 1.00E-02],
    "adult#010": [1.00E-04, 1.00E-07, 1.00E-02],
}

class PIDPolicy:
    def __init__(
        self,
        env,
        patient_name=None,
        max_insulin_action=5.0,
        target_BG=110,
        normalize=False,
        cgm_index=0
    ):
        self.patient_name = patient_name or env.unwrapped.env.patient_name
        self.patient_weights = patient_weights
        P, I, D = patient_weights[self.patient_name]
        self.controller = PIDController(P, I, D, target=target_BG)
        self.env = env
        self.max_insulin_action = float(max_insulin_action)
        self.normalize = bool(normalize)

        self.sample_time = float(getattr(env, "sample_time_min", 3.0))
        self.cgm_index = int(cgm_index)

        

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

            if self.normalize:
                cgm *= 400.0


            bb_obs = Observation(CGM=cgm, CHO=0)

            action_tuple = self.controller.policy(
                observation=bb_obs,
                reward = None,
                done = None,
                sample_time=self.sample_time,
                patient_name=self.patient_name,
                meal=0,
            )


            insulin = float(action_tuple.basal) 

            raw_action = proposed_insulin_to_raw_action(
                insulin,
                max_insulin_action=self.max_insulin_action,
            )

            actions.append(float(np.asarray(raw_action).reshape(-1)[0]))

        actions = np.asarray(actions, dtype=np.float32).reshape(-1, 1)

        if single_obs:
            actions = actions[0]

        return actions, state