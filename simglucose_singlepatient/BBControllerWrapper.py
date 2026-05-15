from simglucose.controller.basal_bolus_ctrller import BBController
from collections import namedtuple

Observation = namedtuple("Observation", ["CGM"])

class BBPolicy():
    def __init__(self, env):

        self.controller = BBController()

        self.env = env
        self.patient_name = env.env.env.env.env.env.patient_name # wtf er det her Elisa
        self.sample_time = env.sample_time_min
        self.cgm_index = env.cgm_index

    def predict(self, obs):
        CGM = obs[self.cgm_index]
        observation = Observation(CGM=CGM)
        #print(observation)
        meal_warning = obs[3]
        meal = obs[4]
        print(observation)
        print(f"meal : {meal*meal_warning}")
        action_tuple = self.controller.policy(observation, 
                               None, 
                               None,
                               sample_time = self.sample_time, 
                               patient_name = self.patient_name, 
                               meal = meal*meal_warning)
        print(action_tuple)
        info = None
        action = action_tuple.basal + action_tuple.bolus
        return action, info