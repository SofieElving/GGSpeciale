from __future__ import annotations

"""
Closed-loop SimGlucose wrapper using CGM, IOB, and previous actions.
"""

from collections import deque
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from GGSpeciale.GGSpeciale.Thesis_code.envs.env_hybrid_closed import (
    REWARD_FNS,
    SimglucoseFeatureWrapper,
    _make_simglucose_spid_env,
)


class ClosedSimglucoseFeatureWrapper(SimglucoseFeatureWrapper):

    def __init__(self, *args: Any, action_history_len: int = 5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.action_history_len = int(action_history_len)
        self.previous_actions = deque([0.0] * self.action_history_len, maxlen=self.action_history_len)

        action_low = float(np.asarray(self.action_space.low).reshape(-1)[0])
        action_high = float(np.asarray(self.action_space.high).reshape(-1)[0])

        obs_low = [0.0, 0.0] if self.normalize else [0.0, 0.0]
        obs_high = [1.0, 2.0] if self.normalize else [600.0, 50.0]

        low = np.array(obs_low + [action_low] * self.action_history_len, dtype=np.float32)
        high = np.array(obs_high + [action_high] * self.action_history_len, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, *args: Any, **kwargs: Any):
        self.previous_actions = deque([0.0] * self.action_history_len, maxlen=self.action_history_len)
        return super().reset(*args, **kwargs)

    def step(self, action: Any):
        action_low = float(np.asarray(self.action_space.low).reshape(-1)[0])
        action_high = float(np.asarray(self.action_space.high).reshape(-1)[0])

        raw_action = float(np.asarray(action, dtype=np.float32).squeeze())
        raw_action = float(np.clip(raw_action, action_low, action_high))

        self.previous_actions.append(raw_action)

        return super().step(action)

    def _get_meal_now_for_plot(self) -> float:
        if not self.meal_schedule:
            return 0.0

        for minute, carbs in self.meal_schedule:
            since_meal = self.current_minute_of_day - float(minute)
            if since_meal < 0:
                since_meal += 1440.0

            if 0.0 <= since_meal < self.sample_time_min:
                return float(carbs)

        return 0.0

    def _build_obs_and_features(self, obs: Any) -> tuple[np.ndarray, dict[str, float]]:
        cgm = float(np.asarray(obs, dtype=np.float32).reshape(-1)[self.cgm_index])
        action_hist = np.array(self.previous_actions, dtype=np.float32)

        if self.normalize:
            base_obs = np.array(
                [
                    np.clip(cgm / 400.0, 1e-6, 1.0),
                    np.clip(self.iob / 10.0, 1e-6, 2.0),
                ],
                dtype=np.float32,
            )
        else:
            base_obs = np.array([cgm, float(self.iob)], dtype=np.float32)

        wrapped_obs = np.concatenate([base_obs, action_hist]).astype(np.float32)

        return wrapped_obs, {
            "cgm": cgm,
            "iob": float(self.iob),
            "meal_now": float(self._get_meal_now_for_plot()),

            # Kept so existing plotting code does not crash.
            "time_since_meal": 0.0,
            "meal_warning": 0.0,
            "meal_size": 0.0,

            # Optional diagnostics.
            "raw_action_t_minus_5": float(action_hist[0]),
            "raw_action_t_minus_4": float(action_hist[1]),
            "raw_action_t_minus_3": float(action_hist[2]),
            "raw_action_t_minus_2": float(action_hist[3]),
            "raw_action_t_minus_1": float(action_hist[4]),
        }

SimglucoseClosedFeatureWrapper = ClosedSimglucoseFeatureWrapper
SimglucoseFeatureWrapper = ClosedSimglucoseFeatureWrapper


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
    shield_bg_threshold: float = 50.0,
    use_bb_warmup: bool = False,
    amount_noise_std_fraction: float = 0.15,
    actual_time_noise_std_min: float = 0.0,
    actual_time_noise_clip_min: float = 30.0,
) -> gym.Env:
    return _make_simglucose_spid_env(
        patient_name=patient_name,
        meal_schedule=meal_schedule,
        env_id=env_id,
        max_episode_steps=max_episode_steps,
        normalize=normalize,
        scenario_mode=scenario_mode,
        seed=seed,
        warning_window_min=warning_window_min,
        insulin_tau_min=insulin_tau_min,
        sample_time_min=sample_time_min,
        time_std_multiplier=time_std_multiplier,
        include_snacks=include_snacks,
        reward_type=reward_type,
        max_insulin_action=max_insulin_action,
        shield_bg_threshold=shield_bg_threshold,
        use_bb_warmup=use_bb_warmup,
        amount_noise_std_fraction=amount_noise_std_fraction,
        actual_time_noise_std_min=actual_time_noise_std_min,
        actual_time_noise_clip_min=actual_time_noise_clip_min,
        wrapper_cls=ClosedSimglucoseFeatureWrapper,
    )