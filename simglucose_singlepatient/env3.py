from __future__ import annotations

"""
Miljø-wrapper og miljøfabrik til SimGlucose.

Måltidslogik bor i meal_scenarios.py.
"""

from datetime import datetime
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from simglucose.simulation.scenario import CustomScenario

from meal_scenarios import hb_fixed_meal_schedule, SemiRandomHarrisonBenedictScenario
from rewards_smooth import glucose_insulin_reward as reward_smooth
from rewards_steps import glucose_insulin_reward as reward_steps
from rewards_strict import glucose_insulin_reward as reward_strict
from rewards_positive import glucose_insulin_reward as reward_postitve


REWARD_FNS: dict[str, Any] = {
    "default": None,
    "smooth": reward_smooth,
    "strict": reward_strict,
    "steps": reward_steps,
    "positive": reward_postitve,
}


class SimglucoseFeatureWrapper(gym.Wrapper):
    """
    Wrapper som eksponerer observerbare features til PPO:
        [CGM, tid_siden_sidste_måltid, IOB, måltidsvarsel, måltidsstørrelse]

    Politikken handler i normaliseret rum [-1, 1], som mappes til insulin via:
        insulin = I_max * exp(4 * (a - 1))

    Måltidsstørrelse er *kun* synlig inde i warning-vinduet.
    Diagnostics lægges i info, så plotting kan ske uden at kende wrapperens interne state.
    """

    def __init__(
        self,
        env: gym.Env,
        meal_schedule: Sequence[tuple[int, float]] | None,
        sample_time_min: float = 3.0,
        warning_window_min: float = 20.0,
        insulin_tau_min: float = 55.0,
        cgm_index: int = 0,
        normalize: bool = True,
        reward_type: str = "default",
        max_insulin_action: float = 5.0,
    ) -> None:
        super().__init__(env)

        if reward_type not in REWARD_FNS:
            raise ValueError(
                f"Ukendt reward_type={reward_type!r}. Forventede en af: {list(REWARD_FNS)}"
            )
        if max_insulin_action <= 0:
            raise ValueError("max_insulin_action skal være > 0.")

        self.reward_type = reward_type
        self.reward_fn = REWARD_FNS[reward_type]
        self.sample_time_min = float(sample_time_min)
        self.warning_window_min = float(warning_window_min)
        self.insulin_tau_min = float(insulin_tau_min)
        self.cgm_index = int(cgm_index)
        self.normalize = bool(normalize)
        self.max_insulin_action = float(max_insulin_action)
        self.history = []
        self.history_index = 0

        self.meal_schedule: list[tuple[int, float]] = []
        if meal_schedule is not None:
            for minute, carbs in meal_schedule:
                self.meal_schedule.append((int(minute) % 1440, float(carbs)))
            self.meal_schedule.sort(key=lambda item: item[0])

        self.current_minute_of_day = 0.0
        self.last_meal_time_min: float | None = None
        self.iob = 0.0

        if self.normalize:
            low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            high = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
        else:
            low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            high = np.array([600.0, 1440.0, 50.0, 1.0, 200.0], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        new_history = self._get_history()
        self.history.append(new_history)
        #self.history_index += 1

        obs, info = self.env.reset(**kwargs)

        try:
            raw_env = self.env.unwrapped
            inner_env = getattr(raw_env, "env", None)
            live_scenario = getattr(inner_env, "custom_scenario", None)
            if live_scenario is not None and hasattr(live_scenario, "get_meal_schedule"):
                synced_schedule: list[tuple[int, float]] = []
                for minute, carbs in live_scenario.get_meal_schedule():
                    synced_schedule.append((int(minute) % 1440, float(carbs)))
                synced_schedule.sort(key=lambda item: item[0])
                self.meal_schedule = synced_schedule
        except Exception:
            pass

        if "sample_time" in info:
            try:
                self.sample_time_min = float(info["sample_time"])
            except Exception:
                pass

        # Læs minut på dagen fra simulatoren; hvis ikke det findes, antag midnat.
        current_time = info.get("time", None)
        if isinstance(current_time, datetime):
            self.current_minute_of_day = float(current_time.hour * 60 + current_time.minute)
        else:
            self.current_minute_of_day = 0.0

        # Find seneste måltid relativt til nuværende klokkeslæt.
        if not self.meal_schedule:
            self.last_meal_time_min = None
        else:
            past_meals = [
                float(minute)
                for minute, _ in self.meal_schedule
                if minute <= self.current_minute_of_day
            ]
            if past_meals:
                self.last_meal_time_min = max(past_meals)
            else:
                self.last_meal_time_min = float(self.meal_schedule[-1][0] - 1440)

        self.iob = 0.0

        wrapped_obs, features = self._build_obs_and_features(obs)
        self._add_diagnostics_info(
            info=info,
            features=features,
            insulin=0.0,
            raw_action=0.0,
            policy_action=0.0,
        )
        return wrapped_obs, info

    def step(self, action):
        raw_policy_action = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        policy_action = float(np.clip(raw_policy_action, -1.0, 1.0))

        # Politikens normaliserede handling skaleres til fysisk insulin.
        delivered = self.max_insulin_action * np.exp(4.0 * (policy_action - 1.0))
        delivered = float(np.clip(delivered, 0.0, self.max_insulin_action))
        sim_action = np.array([delivered], dtype=np.float32)

        # Enkel IOB-model: eksponentielt henfald + nyt positivt insulinbidrag.
        decay = np.exp(-self.sample_time_min / self.insulin_tau_min)
        self.iob = float(self.iob * decay + max(0.0, delivered))

        obs, reward, terminated, truncated, info = self.env.step(sim_action)

        if "sample_time" in info:
            try:
                self.sample_time_min = float(info["sample_time"])
            except Exception:
                pass

        # Læs klokkeslæt fra info; hvis det mangler, ryk tiden frem med sample_time.
        current_time = info.get("time", None)
        if isinstance(current_time, datetime):
            self.current_minute_of_day = float(current_time.hour * 60 + current_time.minute)
        else:
            self.current_minute_of_day = float(
                (self.current_minute_of_day + self.sample_time_min) % 1440.0
            )

        for minute, _ in self.meal_schedule:
            delta = self.current_minute_of_day - float(minute)
            if delta < 0:
                delta += 1440.0
            if 0.0 <= delta < self.sample_time_min:
                self.last_meal_time_min = float(minute)
                break

        wrapped_obs, features = self._build_obs_and_features(obs)
        raw_cgm = float(features["cgm"])
        original_reward = float(reward)

        if self.reward_fn is not None:
            reward = self.reward_fn(
                bg=raw_cgm,
                insulin=delivered,
                raw_insulin=delivered,
                max_insulin_action=self.max_insulin_action,
            )
        else:
            reward = original_reward

        reward = float(reward)

        if raw_cgm < 40.0 or raw_cgm > 400.0:
            reward = -1000.0
            terminated = True
            info["terminal_reason"] = "cgm_out_of_bounds"

        info["original_reward"] = original_reward
        info["reward_type"] = self.reward_type

        self._add_diagnostics_info(
            info=info,
            features=features,
            insulin=delivered,
            raw_action=raw_policy_action,
            policy_action=policy_action,
        )

        return wrapped_obs, reward, terminated, truncated, info

    def _build_obs_and_features(self, obs: Any) -> tuple[np.ndarray, dict[str, float]]:
        """        
        - rå CGM læses ud af observationen
        - tid siden sidste måltid beregnes
        - næste måltids warning + størrelse findes
        - observationen normaliseres, hvis normalize=True
        """
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        cgm = float(obs_arr[self.cgm_index])

        if self.last_meal_time_min is None:
            time_since_meal = 1440.0
        else:
            time_since_meal = self.current_minute_of_day - self.last_meal_time_min
            if time_since_meal < 0:
                time_since_meal += 1440.0
            time_since_meal = float(time_since_meal)

        meal_warning = 0.0
        meal_size = 0.0
        meal_now = 0.0

        if self.meal_schedule:
            best_dt = float("inf")
            best_carbs = 0.0

            for minute, carbs in self.meal_schedule:
                minute_f = float(minute)

                dt = minute_f - self.current_minute_of_day
                if dt < 0:
                    dt += 1440.0
                if dt < best_dt:
                    best_dt = dt
                    best_carbs = float(carbs)

                since_meal = self.current_minute_of_day - minute_f
                if since_meal < 0:
                    since_meal += 1440.0
                if 0.0 <= since_meal < self.sample_time_min:
                    meal_now = float(carbs)

            # måltidsstørrelsen er kun synlig i warning-vinduet.
            if best_dt <= self.warning_window_min:
                meal_warning = float(np.exp(-best_dt / self.warning_window_min))
                meal_size = float(best_carbs)

        raw_features = np.array(
            [cgm, time_since_meal, self.iob, meal_warning, meal_size],
            dtype=np.float32,
        )

        if self.normalize:
            wrapped_obs = np.array(
                [
                    np.clip(raw_features[0] / 400.0, 0.0, 1.0),
                    np.clip(raw_features[1] / 1440.0, 0.0, 1.0),
                    np.clip(raw_features[2] / 10.0, 0.0, 2.0),
                    np.clip(raw_features[3], 0.0, 1.0),
                    np.clip(raw_features[4] / 120.0, 0.0, 1.0),
                ],
                dtype=np.float32,
            )
        else:
            wrapped_obs = raw_features

        return wrapped_obs, {
            "cgm": cgm,
            "time_since_meal": time_since_meal,
            "meal_warning": meal_warning,
            "meal_size": meal_size,
            "meal_now": meal_now,
            "iob": float(self.iob),
        }

    def _add_diagnostics_info(
        self,
        info: dict[str, Any],
        features: dict[str, float],
        insulin: float,
        raw_action: float,
        policy_action: float,
    ) -> None:
        info["plot_cgm_raw"] = float(features["cgm"])
        info["plot_meal"] = float(info.get("meal", features["meal_now"]))
        info["plot_insulin_action"] = float(insulin)
        info["raw_policy_action"] = float(raw_action)
        info["policy_action"] = float(policy_action)
        info["scaled_action"] = float(insulin)
        info["plot_time_since_meal"] = float(features["time_since_meal"])
        info["plot_meal_warning"] = float(features["meal_warning"])
        info["plot_meal_size"] = float(features["meal_size"])
        info["plot_iob"] = float(features["iob"])
        info["sample_time"] = float(self.sample_time_min)
    
    def _get_history(self):
        return self.unwrapped.env.env.show_history().reset_index()
    
    def clear_history(self):
        print("HISTORY CLEARED")
        self.history = []
        self.history_index = 0


def make_simglucose_spid_env(
    patient_name: str = "adult#010",
    meal_schedule: Sequence[tuple[int, float]] | None = None,
    env_id: str = "simglucose-spid-train-v0",
    max_episode_steps: int = 480,
    normalize: bool = True,
    scenario_mode: str = "fixed",
    seed: int | None = None,
    warning_window_min: float = 20.0,
    insulin_tau_min: float = 55.0,
    sample_time_min: float = 3.0,
    time_std_multiplier: float = 1.0,
    include_snacks: bool = True,
    reward_type: str = "default",
    max_insulin_action: float = 5.0,
) -> gym.Env:
    if reward_type not in REWARD_FNS:
        raise ValueError(
            f"Ukendt reward_type={reward_type!r}. Forventede en af: {list(REWARD_FNS)}"
        )
    if max_insulin_action <= 0:
        raise ValueError("max_insulin_action skal være > 0.")

    start_time = datetime(2018, 1, 1, 0, 0, 0)

    if scenario_mode == "fixed":
        if meal_schedule is None:
            raise ValueError("meal_schedule skal angives når scenario_mode='fixed'.")

        wrapper_schedule = [(int(minute), float(carbs)) for minute, carbs in meal_schedule]
        sim_scenario = CustomScenario(
            start_time=start_time,
            scenario=[(minute / 60.0, float(carbs)) for minute, carbs in wrapper_schedule],
        )

    elif scenario_mode == "fixed_hb":
        wrapper_schedule = hb_fixed_meal_schedule(patient_name)
        sim_scenario = CustomScenario(
            start_time=start_time,
            scenario=[(minute / 60.0, float(carbs)) for minute, carbs in wrapper_schedule],
        )

    elif scenario_mode == "semi_random_hb":
        sim_scenario = SemiRandomHarrisonBenedictScenario(
            patient_name=patient_name,
            start_time=start_time,
            seed=seed,
            time_std_multiplier=time_std_multiplier,
            include_snacks=include_snacks,
        )
        wrapper_schedule = sim_scenario.get_meal_schedule()

    else:
        raise ValueError(
            f"Ukendt scenario_mode={scenario_mode!r}. Forventede en af: "
            "fixed, fixed_hb, semi_random_hb."
        )

 
    if env_id in registry:
        del registry[env_id]

    register(
        id=env_id,
        entry_point="simglucose.envs:T1DSimGymnaisumEnv",
        max_episode_steps=max_episode_steps,
        kwargs={
            "patient_name": patient_name,
            "custom_scenario": sim_scenario,
        },
    )

    env = gym.make(env_id)
    env = SimglucoseFeatureWrapper(
        env=env,
        meal_schedule=wrapper_schedule,
        sample_time_min=sample_time_min,
        warning_window_min=warning_window_min,
        insulin_tau_min=insulin_tau_min,
        cgm_index=0,
        normalize=normalize,
        reward_type=reward_type,
        max_insulin_action=max_insulin_action,
    )
    return env
