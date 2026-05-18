'''
Evaluate any sbs3 compatible policy trained on simglucose environment 
'''
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

import json
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd


# from env3 import (
#     make_simglucose_spid_env,
#     MultiPatientSimglucoseEnv,
#     parse_meal_schedule,
#     DEFAULT_MEALS
# )

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
        save_history=False,
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

        self.log_dir = Path(log_path)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_index = 0
        self.save_history = save_history

        if best_model_save_path is not None:
            Path(best_model_save_path).mkdir(parents=True, exist_ok=True)
        
        self.history_path = self.log_dir / "history"
        self.metrics_path = self.log_dir / "evaluation"

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.metrics_path.exists():
            self.metrics_path.touch()

    def _on_step(self) -> bool:
        # Let SB3 run the normal evaluation logic first
        continue_training = super()._on_step()

        # EvalCallback only evaluates every eval_freq calls
        if self.n_calls % self.eval_freq == 0:
            self.after_eval()
            self.eval_env.env_method("clear_history")

        return continue_training
    
    def after_eval(self):
        if len(self.evaluations_results) == 0:
            return
             
        history_df = self.eval_env.get_attr("history")[0]

        #print(history_df.tail())

        # TODO: Make it and option to save history (if path is given?)

        if history_df.empty:
            if self.verbose > 0:
                print("[EvalInsulinPolicy] Warning: get_history() returned empty history.")
            return
        
        
        #print(self.history_path)

        if self.save_history:
            write_header = not self.history_path.exists()

            history_df.to_csv(
                self.history_path,
                mode="a",
                header=write_header,
                index=False,
            )

        metrics = compute_scores(history_df)

        # # TODO: Compute metrics 

        row = {
            "eval_index" : int(self.eval_index),
            "num_timesteps": int(self.num_timesteps),
            #"mean_reward": float(np.mean(latest_rewards)),
            #"std_reward": float(np.std(latest_rewards)),
            "n_eval_episodes": int(self.n_eval_episodes),
        }

        self.eval_index += 1

        # # TODO: Append to history log and save metrics 

    
        # if self.verbose > 0:
        #     print(f"[EvalInsulinPolicy] Eval {self.eval_index} at step {self.num_timesteps}")
        #     print(json.dumps(metrics, indent=2))
        
        
def compute_scores(df:pd.DataFrame) -> dict:
    cf_bounds = [0, 54, 250, 999]
    tir_bounds = [0, 70, 180, 999]

    df = df.dropna()
    n = len(df)

    cf = ((df.BG.value_counts(bins=cf_bounds).sort_index()/n)*100)
    tir = ((df.BG.value_counts(bins=tir_bounds).sort_index()/n)*100)

    TBR_II, TIR_II, TAR_II = cf
    TBR_I, TIR_I, TAR_I = tir

    df["insulin2"] =  df.insulin.astype("float")
    df["D"] = df["Time"].dt.date

    #print(df.head())

    # print("========== ")
    # print(df["Time"].max())
    # print(df["Time"].min())

    #print(f"max time {df["Time"].max()}. Min time {df["Time"].min()}")


    daily_insulin = df.groupby(["eval_index", "D"])["insulin2"].sum()


    print(
        df.groupby(["eval_index", "D"])
        .agg({"BG": ["max", "min", "count"], "CGM" : ["max", "min", "count"], "insulin2" : ["max", "mean"]})
    )


    #print(daily_insulin)

    # TODO: compute total daily insulin
    # df has a column insulin 
    # df has column Time which is a datetime for

    metrics = {
        "TBR_II"    : TBR_II,
        "TBR_I"     : TBR_I,
        "TIR"       : TIR_I,
        "TAR_I"     : TAR_I,
        "TAR_II"    : TAR_II,
        "total_daily_insulin" : daily_insulin

    }

    #print(metrics)

    return metrics