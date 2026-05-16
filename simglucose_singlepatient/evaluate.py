'''
Evaluate any sbs3 compatible policy trained on simglucose environment 
'''
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from env import (
    make_simglucose_spid_env,
    MultiPatientSimglucoseEnv,
    parse_meal_schedule,
    DEFAULT_MEALS
)

def risk_index():
    pass 

class EvalInsulinPolicy(EvalCallback):
    '''
    Evaluation wrapper. Takes compatible policy and env, 
    where env must have a get_history() method.

    Computes the following performance metrics per default: 
     - mean and std TIR, TAR, TBR
     - Critical failure rate
     - Average insulin rate (per minute)
     - ...

    '''
    def __init__(
        self,
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=5,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        deterministic=True,
        render=False,
        verbose=1,
    ):
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
        )

    def _on_step(self) -> bool:
        # Let SB3 run the normal evaluation logic first
        continue_training = super()._on_step()

        # EvalCallback only evaluates every eval_freq calls
        if self.n_calls % self.eval_freq == 0:
            self.after_eval()

        return continue_training
    
    def after_eval(self):
        if len(self.evaluations_results) == 0:
            return
        
        print(self.evaluations_results)
        

