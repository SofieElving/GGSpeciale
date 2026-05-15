'''
Evaluate any sbs3 compatible policy trained on simglucose environment 
'''

from env import (
    make_simglucose_spid_env,
    MultiPatientSimglucoseEnv,
    parse_meal_schedule,
    DEFAULT_MEALS
)

def risk_index():
    pass 

def evaluate_sim_policy(env, 
                        policy,
                        n_episodes: int = 100):

    obs = env.reset()

    times: list[float] = []
    cgms: list[float] = []
    insulin_actions: list[float] = []
    meals: list[float] = []

    critical_failures: list = [] # BG outside safe values
    TIR_list: list = [] # time in range
    TAR_list: list = [] # time above range
    TBR_list: list = [] # time below range 

    step = 0
    for epsiode in range(n_episodes):
        obs = env.reset()

        # times: list[float] = []
        # cgms: list[float] = []
        # insulin_actions: list[float] = []
        # meals: list[float] = []
        critical_failure = 0 # BG outside safe values
        TIR = 0 # time in range
        TAR = 0 # time above range
        TBR = 0

        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            info0 = info[0]
            cgm = float(info0.get("plot_cgm_raw", float("nan")))
            meal = float(info0.get("plot_meal", 0.0))
            insulin = float(info0.get("plot_insulin_action", float("nan")))

            dt = float(info0.get("sample_time", 3.0))
            # times.append(step * dt)
            # cgms.append(cgm)
            # insulin_actions.append(insulin)
            # meals.append(meal)

            if done[0]:
                break

            step += 1