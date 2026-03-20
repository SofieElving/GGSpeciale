"""
script does:
* run each of the 5 models: PPO, DDPG, SAC, TD3, TRPO trained on cartpole
* save: best policy, score, and rewards
"""

import socket
import os
import json
import traceback
from pathlib import Path
from importlib import reload
import warnings
import pysr

print("Host:", socket.gethostname())
print("SLURM job:", os.environ.get("SLURM_JOB_ID"))

#os.chdir("SPID_code")
print("run SPID Working dir is:", os.getcwd())

from stable_baselines3 import PPO, DDPG, SAC, TD3
from sb3_contrib import TRPO

from gmDAGGER import train_spid

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
teacher_path = Path("/home/ashc/GGSpeciale/ashc_repo/GGSpeciale/code/baseline_code/baseline_models/cartpole")
save_folder = Path("/home/ashc/GGSpeciale/ashc_repo/GGSpeciale/code/SPID_code/cartpole_GMDAgger_models")

teacher_files = [
    "DDPG_cartpole.zip",
    "PPO_cartpole.zip",
    "SAC_cartpole.zip",
    "TD3_cartpole.zip",
    "TRPO_cartpole.zip",
]

# Map filename prefix -> SB3 class
model_map = {
    "PPO": PPO,
    "DDPG": DDPG,
    "SAC": SAC,
    "TD3": TD3,
    "TRPO": TRPO,
}

# Ensure top-level save folder exists
save_folder.mkdir(parents=True, exist_ok=True)

# Optional training params
N_ITER = 5
TOTAL_TIMESTEPS = 120
ENV_NAME = "CartPole-v1"
VERBOSE = 2
N_EVAL_EPISODES = 100

all_results = []

for teacher_file in teacher_files:
    teacher_stem = Path(teacher_file).stem          # e.g. "PPO_cartpole"
    algo_name = teacher_stem.split("_")[0]          # e.g. "PPO"

    print("\n" + "=" * 80)
    print(f"Running teacher: {teacher_file}")
    print("=" * 80)

    if algo_name not in model_map:
        print(f"Skipping {teacher_file}: unknown algorithm prefix '{algo_name}'")
        continue

    teacher_cls = model_map[algo_name]
    teacher_full_path = teacher_path / teacher_file

    if not teacher_full_path.exists():
        print(f"Skipping {teacher_file}: file not found at {teacher_full_path}")
        all_results.append({
            "teacher": teacher_file,
            "status": "missing_teacher_file",
            "teacher_path": str(teacher_full_path),
        })
        continue

    # Create one subfolder per teacher model
    run_save_folder = save_folder / teacher_stem
    run_save_folder.mkdir(parents=True, exist_ok=True)

    try:
        rewards, best_policy, wrapper, run_dir = train_spid(
            teacher_path=str(teacher_full_path),
            teacher_model=teacher_cls,
            save_folder_path=str(run_save_folder),
            environment=ENV_NAME,
            n_iter=N_ITER,
            total_timesteps=TOTAL_TIMESTEPS,
            save_results=True,
            verbose=VERBOSE,
            n_eval_episodes=N_EVAL_EPISODES,
        )

        result = {
            "teacher": teacher_file,
            "algorithm": algo_name,
            "status": "success",
            "teacher_path": str(teacher_full_path),
            "run_dir": str(run_dir),
            "best_reward_during_search": float(max(rewards)) if len(rewards) > 0 else None,
            "best_iteration": int(rewards.index(max(rewards))) if len(rewards) > 0 else None,
            "n_rewards_logged": len(rewards),
        }

        # Save a lightweight per-run summary
        with open(run_save_folder / "run_summary.json", "w") as f:
            json.dump(result, f, indent=2)

        all_results.append(result)

        print(f"Finished {teacher_file}")
        print(f"Saved outputs to: {run_save_folder}")

    except Exception as e:
        print(f"FAILED for {teacher_file}")
        print(traceback.format_exc())
        error_result = {
            "teacher": teacher_file,
            "algorithm": algo_name,
            "status": "failed",
            "teacher_path": str(teacher_full_path),
            "run_dir": str(run_save_folder),
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }

        with open(run_save_folder / "run_error.json", "w") as f:
            json.dump(error_result, f, indent=2)

        all_results.append(error_result)

        

# Save combined overview across all teachers
with open(save_folder / "all_runs_summary.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 80)
print("All runs complete.")
print(f"Combined summary saved to: {save_folder / 'all_runs_summary.json'}")
print("=" * 80)
