import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from pysr import PySRRegressor # pip install pysr

# Getting target and features from teacher policy
ENV_ID = "CartPole-v1"
TEACHER_ZIP = "C:\\Users\\sofel\\OneDrive\\Skrivebord\\Speciale\\Git\\GGSpeciale\\code\\baseline_code\\baseline_models\\ppo_cartpole.zip"

n_samples = 100_000
seed = 42

env = gym.make(ENV_ID)
teacher = PPO.load(TEACHER_ZIP, env=None)

X = []
y = []

obs, _ = env.reset(seed=seed)
for _ in range(n_samples):
    # Get action from teacher policy
    action, _ = teacher.predict(obs, deterministic=True)
    X.append(obs)
    #y.append(action)
    # map action {0,1} -> {-1,+1} (often helps symbolic search)
    y.append(-1.0 if int(action) == 0 else 1.0)

    obs, r, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset(seed=seed)

X = np.asarray(X)
y = np.asarray(y)

# Define and fit the symbolic regression model
model = PySRRegressor(
    maxsize=10,
    niterations=50,  # Increase for better results
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
    ], # Custom operator (julia syntax)
    # extra_sympy_mappings={"inv": lambda x: 1 / x}, # Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2", # Custom loss function (julia syntax)
    # batching=True, # For evaluation on random subsets of data
    # batch_size=1024, # Only relevant if batching=True
    temp_equation_file=False,
    delete_tempfiles=True, 
)

# Fit the model to the data
model.fit(X, y)

print(model)
