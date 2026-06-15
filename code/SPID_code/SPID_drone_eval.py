import sys
import os
import math
from io import StringIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from PySRWrapper_drone import PySRPolicy

import argparse

DEFAULT_OBS = ObservationType("kin")
DEFAULT_ACT = ActionType("one_d_rpm")
TARGET_POS  = np.array([0, 0, 1])

class HoverActionShapeWrapper(gym.Wrapper):
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        return self.env.step(action)

def collect_actions(policy, seed=0, max_steps=2000):
    env = HoverActionShapeWrapper(
        HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=False, record=False)
    )
    obs, info = env.reset(seed=seed)
    actions      = []
    rewards      = []
    observations = []

    for _ in range(max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        observations.append(obs)
        if terminated or truncated:
            break

    env.close()
    return rewards, np.array(actions).flatten(), np.array(observations)


def analyze_folder(policy_path, teacher_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher = PPO.load(teacher_path)
    base_policy = PySRPolicy.load(policy_path)
    n_expressions = len(base_policy.policy_list[0].sr.equations_)

    teacher_rewards, teacher_actions, teacher_obs = collect_actions(teacher, seed=42)

    teacher_obs_flat  = teacher_obs.reshape(len(teacher_obs), -1)
    teacher_distances = np.linalg.norm(teacher_obs_flat[:, 0:3] - TARGET_POS, axis=1)
    teacher_velocities = np.linalg.norm(teacher_obs_flat[:, 6:9], axis=1)

    # Best model plot
    stdout, stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = sys.stderr = StringIO()
        best_student = PySRPolicy.load(policy_path)
    finally:
        sys.stdout, sys.stderr = stdout, stderr

    best_student_rewards, best_student_actions, best_student_obs = collect_actions(best_student, seed=42)
    best_student_obs_flat   = best_student_obs.reshape(len(best_student_obs), -1)
    best_student_distances  = np.linalg.norm(best_student_obs_flat[:, 0:3] - TARGET_POS, axis=1)
    best_student_velocities = np.linalg.norm(best_student_obs_flat[:, 6:9], axis=1)

    plot_best_model(
        teacher_actions, teacher_rewards, teacher_distances, teacher_velocities,
        best_student_actions, best_student_rewards, best_student_distances, best_student_velocities,
        output_dir
    )

    all_student_actions    = []
    all_complexities       = []
    all_expressions        = []
    all_fidelities         = []
    all_rewards            = []
    all_distances          = []
    all_velocities         = []
    all_performance_gaps   = []

    for idx in range(n_expressions):
        stdout, stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = sys.stderr = StringIO()
            student = PySRPolicy.load_policy_at_index(policy_path, idx)
        finally:
            sys.stdout, sys.stderr = stdout, stderr

        complexity = base_policy.policy_list[0].sr.equations_.iloc[idx]["complexity"]
        expression = base_policy.policy_list[0].sr.equations_.iloc[idx]["equation"]
        fidelity   = base_policy.policy_list[0].sr.equations_.iloc[idx]["fidelity_loss"]
        performance_gap = base_policy.policy_list[0].sr.equations_.iloc[idx]["perf_gap"]

        student_rewards, student_actions, student_obs = collect_actions(student, seed=42)

        student_obs_flat   = student_obs.reshape(len(student_obs), -1)
        student_distances  = np.linalg.norm(student_obs_flat[:, 0:3] - TARGET_POS, axis=1)
        student_velocities = np.linalg.norm(student_obs_flat[:, 6:9], axis=1)

        all_student_actions.append(student_actions)
        all_complexities.append(complexity)
        all_expressions.append(expression)
        all_fidelities.append(fidelity)
        all_rewards.append(student_rewards)
        all_distances.append(student_distances)
        all_velocities.append(student_velocities)
        all_performance_gaps.append(performance_gap)

    n_cols = 2
    n_rows = math.ceil(n_expressions / n_cols)

    # Action plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    for idx, (student_actions, complexity, expression, fidelity) in enumerate(
        zip(all_student_actions, all_complexities, all_expressions, all_fidelities)
    ):
        ax = axes[idx]
        ax.plot(teacher_actions, label="teacher")
        ax.plot(student_actions, label=f"student (complexity={complexity})")
        ax.legend()
        ax.grid()
        ax.set_xlabel("timesteps")
        ax.set_ylabel("action")
        ax.set_title(f"Complexity {complexity} | fidelity={fidelity:.4f}  |  Teacher reward={np.sum(teacher_rewards):.2f}  |  Student reward={np.sum(all_rewards[idx]):.2f}", fontsize=11)
    for idx in range(n_expressions, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.suptitle("Action Comparison Across Student Models", fontsize=16, y=1.03)
    plt.savefig(output_dir / "comparison_actions.png")
    plt.close()

    # Reward plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    for idx, (student_rewards, complexity, expression, fidelity) in enumerate(
        zip(all_rewards, all_complexities, all_expressions, all_fidelities)
    ):
        ax = axes[idx]
        ax.plot(teacher_rewards, label="teacher")
        ax.plot(student_rewards, label=f"student (complexity={complexity})")
        ax.legend()
        ax.grid()
        ax.set_xlabel("timesteps")
        ax.set_ylabel("reward")
        ax.set_title(f"Complexity {complexity} | fidelity={fidelity:.4f}  |  Teacher reward={np.sum(teacher_rewards):.2f}  |  Student reward={np.sum(all_rewards[idx]):.2f}", fontsize=11)
    for idx in range(n_expressions, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.suptitle("Reward Comparison Across Student Models", fontsize=16, y=1.03)
    plt.savefig(output_dir / "comparison_rewards.png")
    plt.close()

    # Distance plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    for idx, (student_distances, complexity, expression, fidelity) in enumerate(
        zip(all_distances, all_complexities, all_expressions, all_fidelities)
    ):
        ax = axes[idx]
        ax.plot(teacher_distances, label="teacher")
        ax.plot(student_distances, label=f"student (complexity={complexity})")
        ax.legend()
        ax.grid()
        ax.set_xlabel("timesteps")
        ax.set_ylabel("distance (m)")
        ax.set_title(f"Complexity {complexity} | fidelity={fidelity:.4f}  |  Teacher reward={np.sum(teacher_rewards):.2f}  |  Student reward={np.sum(all_rewards[idx]):.2f}", fontsize=11)
    for idx in range(n_expressions, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.suptitle("Distance Comparison Across Student Models", fontsize=16, y=1.03)
    plt.savefig(output_dir / "comparison_distances.png")
    plt.close()

    # Velocity plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    for idx, (student_velocities, complexity, expression, fidelity) in enumerate(
        zip(all_velocities, all_complexities, all_expressions, all_fidelities)
    ):
        ax = axes[idx]
        ax.plot(teacher_velocities, label="teacher")
        ax.plot(student_velocities, label=f"student (complexity={complexity})")
        ax.legend()
        ax.grid()
        ax.set_xlabel("timesteps")
        ax.set_ylabel("velocity (m/s)")
        ax.set_title(f"Complexity {complexity} | fidelity={fidelity:.4f}  |  Teacher reward={np.sum(teacher_rewards):.2f}  |  Student reward={np.sum(all_rewards[idx]):.2f}", fontsize=11)
    for idx in range(n_expressions, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.suptitle("Velocity Comparison Across Student Models", fontsize=16, y=1.03)
    plt.savefig(output_dir / "comparison_velocities.png")
    plt.close()

    print(f"Saved plots to {output_dir}")

def plot_best_model(teacher_actions, teacher_rewards, teacher_distances, teacher_velocities,
                    student_actions, student_rewards, student_distances, student_velocities,
                    output_dir):

    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    axes[0].plot(teacher_actions, label="teacher")
    axes[0].plot(student_actions, label="student")
    axes[0].legend()
    axes[0].grid()
    axes[0].set_xlabel("timesteps")
    axes[0].set_ylabel("action")
    axes[0].set_title("Action")

    axes[1].plot(teacher_rewards, label="teacher")
    axes[1].plot(student_rewards, label="student")
    axes[1].legend()
    axes[1].grid()
    axes[1].set_xlabel("timesteps")
    axes[1].set_ylabel("reward")
    axes[1].set_title("Reward per timestep")

    axes[2].plot(teacher_distances, label="teacher")
    axes[2].plot(student_distances, label="student")
    axes[2].legend()
    axes[2].grid()
    axes[2].set_xlabel("timesteps")
    axes[2].set_ylabel("distance (m)")
    axes[2].set_title("Distance to target")

    axes[3].plot(teacher_velocities, label="teacher")
    axes[3].plot(student_velocities, label="student")
    axes[3].legend()
    axes[3].grid()
    axes[3].set_xlabel("timesteps")
    axes[3].set_ylabel("velocity (m/s)")
    axes[3].set_title("Velocity per timestep")

    fig.suptitle(
        f"Total reward={sum(student_rewards):.2f}",
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(output_dir / "best_model.png")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path",  type=str, required=True)
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_folder(args.policy_path, args.teacher_path, args.output_dir)