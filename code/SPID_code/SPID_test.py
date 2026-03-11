'''
script does: 
* run each of the 5 models: PPO, DDPG SAC, TD3 trained on cartpole
* save: best policy, score, and rewards
'''
import socket
import os 

print("Host:", socket.gethostname())
print("SLURM job:", os.environ.get("SLURM_JOB_ID"))

os.chdir("code")

print(os.getcwd())

from stable_baselines3 import PPO, DDPG, SAC, TD3
from sb3_contrib import TRPO
from importlib import reload 
import SPID_code
reload(SPID_code)
from SPID_code.gmDAGGER import train_spid
import warnings


teacher_path = "/home/ashc/GGSpeciale/ashc_repo/GGSpeciale/code/baseline_code/baseline_models/cartpole"
teachers = ["DDPG_cartpole.zip", "PPO_cartpole.zip", "SAC_cartpole.zip", "TD3_cartpole.zip", "TRPO_cartpole.zip"]

save_folder = "/home/ashc/GGSpeciale/ashc_repo/GGSpeciale/code/SPID_code/cartpole_models"

try: 
    rewards, best_policy, wrapper, run_dir = train_spid(r"C:\GitHub\GGSpeciale\code\baseline_code\baseline_models\cartpole\PPO_cartpole", 
                                            PPO, 
                                            "", 
                                            "CartPole-v1", 
                                            n_iter=5, 
                                            total_timesteps=120, 
                                            verbose=2)
except Exception as e: 
    print(f"training failed")
