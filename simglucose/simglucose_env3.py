from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from simglucose.simulation.scenario import CustomScenario

from meal_scenarios import (
    get_patient_bw_and_kind,
    harris_benedict,
    SemiRandomHarrisonBenedictScenario,
)


DEFAULT_MEALS = [
    (7 * 60, 45.0),
    (12 * 60, 70.0),
    (16 * 60, 15.0),
    (18 * 60, 80.0),
    (23 * 60, 10.0),
]


@dataclass(frozen=True)
class MealEvent:
    minute_of_day: int
    carbs: float


class SimglucoseFeatureWrapper(gym.Wrapper):
    """
    Observation:
        [cgm, time_since_meal, insulin_on_board, meal_warning, meal_size]

    All meal timing is minute-level internally.
    """

    def __init__(
        self,
        env: gym.Env,
        meal_schedule: Sequence[tuple[int, float]] | Sequence[MealEvent],
        sample_time_min: float = 3.0,
        warning_window_min: float = 20.0,
        insulin_tau_min: float = 55.0,
        cgm_index: int = 0,
        normalize: bool = True,
    ):
        super().__init__(env)

        self.sample_time_min = float(sample_time_min)
        self.warning_window_min = float(warning_window_min)
        self.insulin_tau_min = float(insulin_tau_min)
        self.cgm_index = int(cgm_index)
        self.normalize = bool(normalize)

        self.meal_schedule = self._normalize_schedule(meal_schedule)

        self.current_minute_of_day = 0.0
        self.last_meal_time_min: float | None = None
        self.iob = 0.0

        if self.normalize:
            low = np.array([0, 0, 0, 0, 0], dtype=np.float32)
            high = np.array([1, 1, 2, 1, 1], dtype=np.float32)
        else:
            low = np.array([0, 0, 0, 0, 0], dtype=np.float32)
            high = np.array([600, 1440, 50, warning_window_min, 200], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Important: keep action_space from the underlying Gymnasium simglucose env.
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.current_minute_of_day = self._extract_minute_of_day(info)
        self.last_meal_time_min = self._infer_last_meal(self.current_minute_of_day)
        self.iob = 0.0

        raw_cgm = self._extract_cgm(obs)
        time_since_meal = self._time_since_last_meal()
        meal_warning, meal_size = self._next_meal_features()
        wrapped_obs = self._build_obs(obs)

        info["plot_cgm_raw"] = float(raw_cgm)
        info["plot_meal"] = float(info.get("meal", 0.0))
        info["plot_insulin_action"] = 0.0
        info["plot_time_since_meal"] = float(time_since_meal)
        info["plot_meal_warning"] = float(meal_warning)
        info["plot_meal_size"] = float(meal_size)
        info["plot_iob"] = float(self.iob)
        info["sample_time"] = float(self.sample_time_min)

        return wrapped_obs, info

    def step(self, action):
        
        #delivered = self._scalar_action(action)
        #self._update_iob(delivered)

        #obs, reward, terminated, truncated, info = self.env.step(action)

        ### AN HARD LIMIT TO DOSAGE TEST
        raw_action = self._scalar_action(action)

        clipped_action = np.asarray(action, dtype=np.float32).copy()
        clipped_action = np.clip(clipped_action, 0.0, self.max_insulin_action)

        delivered = self._scalar_action(clipped_action)
        self._update_iob(delivered)

        obs, reward, terminated, truncated, info = self.env.step(clipped_action)
        ### END OF TEST

        self.current_minute_of_day = self._extract_minute_of_day(info)
        self._maybe_update_last_meal()

        raw_cgm = self._extract_cgm(obs)
        time_since_meal = self._time_since_last_meal()
        meal_warning, meal_size = self._next_meal_features()
        wrapped_obs = self._build_obs(obs)

        info["plot_cgm_raw"] = float(raw_cgm)
        info["plot_meal"] = float(info.get("meal", 0.0))
        info["plot_insulin_action"] = float(delivered)
        info["plot_time_since_meal"] = float(time_since_meal)
        info["plot_meal_warning"] = float(meal_warning)
        info["plot_meal_size"] = float(meal_size)
        info["plot_iob"] = float(self.iob)
        info["sample_time"] = float(self.sample_time_min)

        return wrapped_obs, reward, terminated, truncated, info

    def _normalize_schedule(
        self,
        meal_schedule: Sequence[tuple[int, float]] | Sequence[MealEvent],
    ) -> list[MealEvent]:
        normalized: list[MealEvent] = []

        for item in meal_schedule:
            if isinstance(item, MealEvent):
                minute = int(item.minute_of_day) % 1440
                carbs = float(item.carbs)
            else:
                minute = int(item[0]) % 1440
                carbs = float(item[1])
            normalized.append(MealEvent(minute_of_day=minute, carbs=carbs))

        normalized.sort(key=lambda x: x.minute_of_day)
        return normalized

    def _build_obs(self, obs) -> np.ndarray:
        cgm = self._extract_cgm(obs)
        time_since_meal = self._time_since_last_meal()
        meal_warning, meal_size = self._next_meal_features()

        x = np.array(
            [cgm, time_since_meal, self.iob, meal_warning, meal_size],
            dtype=np.float32,
        )
        return self._normalize_obs(x) if self.normalize else x

    def _extract_cgm(self, obs) -> float:
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        return float(obs_arr[self.cgm_index])

    def _extract_minute_of_day(self, info: dict) -> float:
        t = info.get("time", None)
        if isinstance(t, datetime):
            return float(t.hour * 60 + t.minute)

        return float((self.current_minute_of_day + self.sample_time_min) % 1440.0)

    def _scalar_action(self, action) -> float:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        return float(arr[0])

    def _update_iob(self, delivered_insulin: float) -> None:
        decay = np.exp(-self.sample_time_min / self.insulin_tau_min)
        self.iob = float(self.iob * decay + max(0.0, delivered_insulin))

    def _infer_last_meal(self, current_minute: float) -> float | None:
        if not self.meal_schedule:
            return None

        past_meals = [
            float(m.minute_of_day)
            for m in self.meal_schedule
            if m.minute_of_day <= current_minute
        ]
        if past_meals:
            return max(past_meals)

        return float(self.meal_schedule[-1].minute_of_day - 1440)

    def _maybe_update_last_meal(self) -> None:
        for meal in self.meal_schedule:
            delta = self.current_minute_of_day - meal.minute_of_day
            if 0 <= delta < self.sample_time_min:
                self.last_meal_time_min = float(meal.minute_of_day)
                return

    def _time_since_last_meal(self) -> float:
        if self.last_meal_time_min is None:
            return 1440.0

        delta = self.current_minute_of_day - self.last_meal_time_min
        if delta < 0:
            delta += 1440.0
        return float(delta)

    def _next_meal_features(self) -> tuple[float, float]:
        if not self.meal_schedule:
            return 0.0, 0.0

        current = self.current_minute_of_day
        best_dt = float("inf")
        best_carbs = 0.0

        for meal in self.meal_schedule:
            meal_min = float(meal.minute_of_day)
            dt = meal_min - current
            if dt < 0:
                dt += 1440.0

            if dt < best_dt:
                best_dt = dt
                best_carbs = meal.carbs

        warning_horizon = self.warning_window_min

        if best_dt <= warning_horizon:
            warning = np.exp(-best_dt / warning_horizon)
            return float(warning), float(best_carbs)

        return 0.0, float(best_carbs)
    
    def _normalize_obs(self, x: np.ndarray) -> np.ndarray:
        cgm = np.clip(x[0] / 400.0, 0.0, 1.0)
        time_since_meal = np.clip(x[1] / 1440.0, 0.0, 1.0)
        iob = np.clip(x[2] / 10.0, 0.0, 2.0)
        meal_warning = np.clip(x[3], 0.0, 1.0)
        meal_size = np.clip(x[4] / 100.0, 0.0, 1.0)

        return np.array(
            [cgm, time_since_meal, iob, meal_warning, meal_size],
            dtype=np.float32,
        )


def parse_meal_schedule(
    text: str | None,
    default: Sequence[tuple[int, float]] = DEFAULT_MEALS,
) -> list[tuple[int, float]]:
    """
    Parses strings like:
        "7:45,12:70,16:15,18:80,23:10"

    Returns minute-level schedule:
        [(420, 45), (720, 70), ...]
    """
    if not text:
        return [(int(m), float(c)) for m, c in default]

    meals: list[tuple[int, float]] = []
    for item in text.split(","):
        hour_str, carbs_str = item.strip().split(":")
        meals.append((int(hour_str) * 60, float(carbs_str)))
    return meals


def hb_fixed_meal_schedule(patient_name: str) -> list[tuple[int, float]]:
    """
    Fixed meal times, Harris-Benedict meal sizes.
    Returns minute-level schedule.
    """
    bw, kind = get_patient_bw_and_kind(patient_name)
    b, l, d, s = harris_benedict(bw, kind)

    return [
        (7 * 60, round(b)),
        (12 * 60, round(l)),
        (16 * 60, round(s)),
        (18 * 60, round(d)),
        (22 * 60, round(s)),
    ]


def build_scenario_and_wrapper_schedule(
    patient_name: str,
    meal_schedule: Sequence[tuple[int, float]] | None,
    scenario_mode: str,
    seed: int | None,
    time_std_multiplier: float,
    include_snacks: bool,
):
    """
    Returns:
        sim_scenario:
            scenario object passed to simglucose
        wrapper_schedule:
            minute-level meal schedule used by SimglucoseFeatureWrapper
    """
    start_time = datetime(2018, 1, 1, 0, 0, 0)

    if scenario_mode == "fixed":
        if meal_schedule is None:
            raise ValueError("meal_schedule must be provided for scenario_mode='fixed'")

        wrapper_schedule = [(int(m), float(c)) for m, c in meal_schedule]
        sim_scenario = CustomScenario(
            start_time=start_time,
            scenario=[(m / 60.0, float(c)) for m, c in wrapper_schedule],
        )

    elif scenario_mode == "fixed_hb":
        wrapper_schedule = hb_fixed_meal_schedule(patient_name)
        sim_scenario = CustomScenario(
            start_time=start_time,
            scenario=[(m / 60.0, float(c)) for m, c in wrapper_schedule],
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
            f"Unknown scenario_mode={scenario_mode}. "
            "Expected one of: fixed, fixed_hb, semi_random_hb"
        )

    return sim_scenario, wrapper_schedule


def register_simglucose_gym_env(
    env_id: str,
    patient_name: str,
    sim_scenario,
    max_episode_steps: int,
) -> None:
    """
    Register a Gymnasium-compatible simglucose env.

    Important:
    - Gym registry is global.
    - For multi-patient training, use unique env_id per reset.
    """
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


def make_simglucose_spid_env(
    patient_name: str = "adult#010",
    meal_schedule: Sequence[tuple[int, float]] | None = DEFAULT_MEALS,
    env_id: str = "simglucose-spid-train-v0",
    max_episode_steps: int = 480,
    normalize: bool = True,
    scenario_mode: str = "fixed",
    seed: int | None = None,
    warning_window_min: float = 10.0,
    insulin_tau_min: float = 55.0,
    sample_time_min: float = 3.0,
    time_std_multiplier: float = 1.0,
    include_snacks: bool = True,
):
    """
    Builds a Gymnasium-compatible SimGlucose env and wraps it with features.

    scenario_mode:
        fixed:
            use provided meal_schedule
        fixed_hb:
            fixed meal times, Harris-Benedict meal sizes
        semi_random_hb:
            semi-random meal times, Harris-Benedict meal sizes
    """
    sim_scenario, wrapper_schedule = build_scenario_and_wrapper_schedule(
        patient_name=patient_name,
        meal_schedule=meal_schedule,
        scenario_mode=scenario_mode,
        seed=seed,
        time_std_multiplier=time_std_multiplier,
        include_snacks=include_snacks,
    )

    register_simglucose_gym_env(
        env_id=env_id,
        patient_name=patient_name,
        sim_scenario=sim_scenario,
        max_episode_steps=max_episode_steps,
    )

    env = gym.make(env_id)

    env = SimglucoseFeatureWrapper(
        env,
        meal_schedule=wrapper_schedule,
        sample_time_min=sample_time_min,
        warning_window_min=warning_window_min,
        insulin_tau_min=insulin_tau_min,
        cgm_index=0,
        normalize=normalize,
    )

    return env


class MultiPatientSimglucoseEnv(gym.Env):
    """
    Samples one patient from patient_names at each reset.

    This keeps your old working simglucose Gym env path:
        gym.make(...) -> T1DSimGymnaisumEnv -> SimglucoseFeatureWrapper

    It avoids using low-level BaseT1DSimEnv directly.
    """

    metadata = {}

    def __init__(
        self,
        patient_names: Sequence[str],
        env_id: str,
        max_episode_steps: int,
        normalize: bool = True,
        meal_schedule: Sequence[tuple[int, float]] | None = DEFAULT_MEALS,
        scenario_mode: str = "fixed",
        seed: int | None = None,
        warning_window_min: float = 10.0,
        insulin_tau_min: float = 55.0,
        sample_time_min: float = 3.0,
        time_std_multiplier: float = 1.0,
        include_snacks: bool = True,
    ):
        super().__init__()

        if len(patient_names) == 0:
            raise ValueError("patient_names must contain at least one patient")

        self.patient_names = list(patient_names)
        self.env_id = env_id
        self.max_episode_steps = int(max_episode_steps)
        self.normalize = bool(normalize)
        self.meal_schedule = meal_schedule
        self.scenario_mode = scenario_mode
        self.warning_window_min = float(warning_window_min)
        self.insulin_tau_min = float(insulin_tau_min)
        self.sample_time_min = float(sample_time_min)
        self.time_std_multiplier = float(time_std_multiplier)
        self.include_snacks = bool(include_snacks)

        self.rng = np.random.RandomState(seed)
        self.base_seed = seed
        self.reset_count = 0
        self.env: gym.Env | None = None
        self.current_patient: str | None = None

        # Probe one env to get action/observation spaces.
        probe_env = make_simglucose_spid_env(
            patient_name=self.patient_names[0],
            meal_schedule=self.meal_schedule,
            env_id=f"{self.env_id}-probe",
            max_episode_steps=self.max_episode_steps,
            normalize=self.normalize,
            scenario_mode=self.scenario_mode,
            seed=self.base_seed,
            warning_window_min=self.warning_window_min,
            insulin_tau_min=self.insulin_tau_min,
            sample_time_min=self.sample_time_min,
            time_std_multiplier=self.time_std_multiplier,
            include_snacks=self.include_snacks,
        )
        self.observation_space = probe_env.observation_space
        self.action_space = probe_env.action_space
        probe_env.close()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)

        if self.env is not None:
            self.env.close()
            self.env = None

        self.reset_count += 1
        self.current_patient = str(self.rng.choice(self.patient_names))

        episode_seed = None
        if self.base_seed is not None:
            episode_seed = int(self.base_seed + self.reset_count)
        if seed is not None:
            episode_seed = int(seed + self.reset_count)

        safe_patient = self.current_patient.replace("#", "-")
        episode_env_id = f"{self.env_id}-{safe_patient}-{self.reset_count}"

        self.env = make_simglucose_spid_env(
            patient_name=self.current_patient,
            meal_schedule=self.meal_schedule,
            env_id=episode_env_id,
            max_episode_steps=self.max_episode_steps,
            normalize=self.normalize,
            scenario_mode=self.scenario_mode,
            seed=episode_seed,
            warning_window_min=self.warning_window_min,
            insulin_tau_min=self.insulin_tau_min,
            sample_time_min=self.sample_time_min,
            time_std_multiplier=self.time_std_multiplier,
            include_snacks=self.include_snacks,
        )

        obs, info = self.env.reset(seed=episode_seed)
        info["patient_name"] = self.current_patient
        info["episode_env_id"] = episode_env_id
        return obs, info

    def step(self, action):
        if self.env is None:
            raise RuntimeError("Environment used before reset().")

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["patient_name"] = self.current_patient
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.env is not None:
            return self.env.render()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None