from __future__ import annotations

"""
Miljø-wrapper og miljøfabrik til SimGlucose.

Måltidslogik bor i meal_scenarios.py.
"""

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.simulation.scenario import CustomScenario

from meal_scenarios import hb_fixed_meal_schedule, SemiRandomHarrisonBenedictScenario
from rewards_smooth import glucose_insulin_reward as reward_smooth
from rewards_steps import glucose_insulin_reward as reward_steps
from rewards_strict import glucose_insulin_reward as reward_strict
from rewards_positive import glucose_insulin_reward as reward_positive


REWARD_FNS: dict[str, Any] = {
    "default": None,
    "smooth": reward_smooth,
    "strict": reward_strict,
    "steps": reward_steps,
    "positive": reward_positive,
}


# Hardcoded BBController warm-up duration.
# Turned on/off from train.py with --bb-warmup.
BB_WARMUP_MINUTES = 180.0


class SimglucoseFeatureWrapper(gym.Wrapper):
    """
    Wrapper som eksponerer observerbare features til PPO:
        [CGM, tid_siden_sidste_måltid, IOB, måltidsvarsel, måltidsstørrelse]

    Politikken handler i normaliseret rum [-1, 1], som mappes til insulin via:
        insulin = I_max * exp(4 * (a - 1))

    Måltidsstørrelse er kun synlig inde i warning-vinduet.

    Shield:
        Hvis den aktuelle CGM-observation, som PPO handler ud fra, er under
        shield_bg_threshold, bliver insulin sat til 0.

    BB warm-up:
        Hvis --bb-warmup bruges, køres BBController skjult efter reset,
        før PPO får første observation. Warm-up indgår ikke i reward/plots.

    History:
        self.history samler SimGlucose show_history() fra afsluttede episoder.
        Historikken gemmes ved næste reset(), fordi simulatorens interne historik
        typisk nulstilles ved reset.
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
        control_max_episode_steps: int | None = None,
        bb_warmup_steps: int = 0,
        use_bb_warmup: bool = False,
        shield_bg_threshold: float = 50.0,
    ) -> None:
        super().__init__(env)

        if reward_type not in REWARD_FNS:
            raise ValueError(
                f"Ukendt reward_type={reward_type!r}. Forventede en af: {list(REWARD_FNS)}"
            )

        if max_insulin_action <= 0:
            raise ValueError("max_insulin_action skal være > 0.")

        if bb_warmup_steps < 0:
            raise ValueError("bb_warmup_steps skal være >= 0.")

        if shield_bg_threshold <= 0:
            raise ValueError("shield_bg_threshold skal være > 0.")

        self.reward_type = reward_type
        self.reward_fn = REWARD_FNS[reward_type]

        self.sample_time_min = float(sample_time_min)
        self.warning_window_min = float(warning_window_min)
        self.insulin_tau_min = float(insulin_tau_min)
        self.cgm_index = int(cgm_index)
        self.normalize = bool(normalize)
        self.max_insulin_action = float(max_insulin_action)
        self.shield_bg_threshold = float(shield_bg_threshold)

        self.control_max_episode_steps = control_max_episode_steps
        self.control_step_count = 0

        self.use_bb_warmup = bool(use_bb_warmup)
        self.bb_warmup_steps = int(bb_warmup_steps)

        # Aggregated SimGlucose history from completed episodes.
        self.history: list[pd.DataFrame] = []
        self.history_index = 0

        # This is the announced meal schedule used for meal_warning and meal_size.
        # In semi_random_hb this can differ from the actual delivered meal schedule.
        self.meal_schedule: list[tuple[int, float]] = []
        if meal_schedule is not None:
            for minute, carbs in meal_schedule:
                self.meal_schedule.append((int(minute) % 1440, float(carbs)))
            self.meal_schedule.sort(key=lambda item: item[0])

        self.current_minute_of_day = 0.0
        self.last_meal_time_min: float | None = None
        self.iob = 0.0

        # Current CGM is the CGM observation PPO is currently acting on.
        # This is not trend and not CGM history.
        self.current_cgm: float | None = None

        if self.normalize:
            low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            high = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
        else:
            low = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            high = np.array([600.0, 1440.0, 50.0, 1.0, 200.0], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        # Store previous simulator trajectory before env.reset() clears it.
        self._store_current_history()
        obs, info = self.env.reset(**kwargs)

        obs = self.unwrapped.env.reset()

        self.control_step_count = 0
        self.iob = 0.0
        self.current_cgm = None
        #RANDOM INIT INSTADT OF BALAS BOLUS INIT
        if self.use_bb_warmup:
            self.iob = float(np.random.default_rng().uniform(0.5, 1.0))
            info["bb_warmup_failed"] = False
            info["bb_warmup_iob_only"] = True
        else:
            self.iob = 0.0
            info["bb_warmup_failed"] = False
            info["bb_warmup_iob_only"] = False

        # Hidden BBController warm-up.
        # PPO does not see these steps. Plotting also starts after this block.
        # if self.use_bb_warmup and self.bb_warmup_steps > 0:
        #     bb_controller = BBController()
        #     bb_warmup_failed = False

        #     last_reward = 0.0
        #     last_done = False

        #     for _ in range(self.bb_warmup_steps):
        #         obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        #         cgm = float(obs_arr[self.cgm_index])

        #         # BBController expects a SimGlucose-like observation with .CGM.
        #         controller_obs = SimpleNamespace(CGM=cgm)

        #         try:
        #             controller_action = bb_controller.policy(
        #                 controller_obs,
        #                 last_reward,
        #                 last_done,
        #                 **info,
        #             )
        #         except TypeError:
        #             controller_action = bb_controller.policy(
        #                 controller_obs,
        #                 last_reward,
        #                 last_done,
        #             )

        #         if hasattr(controller_action, "basal") or hasattr(controller_action, "bolus"):
        #             basal = float(getattr(controller_action, "basal", 0.0))
        #             bolus = float(getattr(controller_action, "bolus", 0.0))

        #             # The Gym wrapper sends one scalar action.
        #             # Basal is used directly. Bolus is spread over one env step.
        #             bb_insulin = basal + bolus / max(self.sample_time_min, 1e-6)
        #         else:
        #             bb_insulin = float(
        #                 np.asarray(controller_action, dtype=np.float32).reshape(-1)[0]
        #             )

        #         bb_insulin = float(np.clip(bb_insulin, 0.0, self.max_insulin_action))
        #         bb_action = np.array([bb_insulin], dtype=np.float32)

        #         decay = np.exp(-self.sample_time_min / self.insulin_tau_min)
        #         self.iob = float(self.iob * decay + max(0.0, bb_insulin))

        #         obs, reward, terminated, truncated, info = self.env.step(bb_action)

        #         last_reward = float(reward)
        #         last_done = bool(terminated or truncated)

        #         if "sample_time" in info:
        #             try:
        #                 self.sample_time_min = float(info["sample_time"])
        #             except Exception:
        #                 pass

        #         if terminated or truncated:
        #             bb_warmup_failed = True
        #             break

        #     if bb_warmup_failed:
        #         obs, info = self.env.reset(**kwargs)
        #         self.iob = 0.0
        #         self.current_cgm = None
        #         info["bb_warmup_failed"] = True
        #     else:
        #         info["bb_warmup_failed"] = False

        # Important for semi_random_hb:
        # This syncs the announced meal schedule, not necessarily the actual delivered one.
        self._sync_live_meal_schedule_if_available()

        if "sample_time" in info:
            try:
                self.sample_time_min = float(info["sample_time"])
            except Exception:
                pass

        current_time = info.get("time", None)
        if isinstance(current_time, datetime):
            self.current_minute_of_day = float(current_time.hour * 60 + current_time.minute)
        else:
            self.current_minute_of_day = 0.0

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

        wrapped_obs, features = self._build_obs_and_features(obs)
        self.current_cgm = float(features["cgm"])

        info["proposed_insulin"] = 0.0
        info["shielded_insulin"] = 0.0
        info["shield_active"] = 0.0
        info["shield_reason"] = "reset"
        info["shield_cgm"] = float(self.current_cgm)
        info["shield_bg_threshold"] = float(self.shield_bg_threshold)

        self._add_diagnostics_info(
            info=info,
            features=features,
            insulin=0.0,
            raw_action=0.0,
            policy_action=0.0,
        )

        info["use_bb_warmup"] = bool(self.use_bb_warmup)
        info["bb_warmup_steps"] = int(self.bb_warmup_steps)
        info["bb_warmup_minutes"] = float(self.bb_warmup_steps * self.sample_time_min)

        return wrapped_obs, info

    def step(self, action):
        raw_policy_action = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        policy_action = float(np.clip(raw_policy_action, -1.0, 1.0))

        proposed_insulin = self.max_insulin_action * np.exp(
            4.0 * (policy_action - 1.0)
        )
        proposed_insulin = float(np.clip(proposed_insulin, 0.0, self.max_insulin_action))

        # Current CGM = den CGM-værdi, PPO lige har observeret og handler ud fra.
        shield_cgm = 120.0 if self.current_cgm is None else float(self.current_cgm)

        shield_active = False
        shield_reason = "no_shield"
        delivered = proposed_insulin

        if shield_cgm < self.shield_bg_threshold:
            delivered = 0.0
            shield_active = True
            shield_reason = f"suspend_bg_below_{self.shield_bg_threshold:g}"

        sim_action = np.array([delivered], dtype=np.float32)

        # Enkel IOB-model: eksponentielt henfald + faktisk leveret insulin.
        decay = np.exp(-self.sample_time_min / self.insulin_tau_min)
        self.iob = float(self.iob * decay + max(0.0, delivered))

        obs, reward, terminated, truncated, info = self.env.step(sim_action)
        print(f"obs: {obs}")

        if "sample_time" in info:
            try:
                self.sample_time_min = float(info["sample_time"])
            except Exception:
                pass

        current_time = info.get("time", None)
        if isinstance(current_time, datetime):
            self.current_minute_of_day = float(current_time.hour * 60 + current_time.minute)
        else:
            self.current_minute_of_day = float(
                (self.current_minute_of_day + self.sample_time_min) % 1440.0
            )

        # For semi_random_hb over multiple days, the scenario may regenerate at midnight.
        # Resync announced meal warnings near day boundary.
        if self.current_minute_of_day < self.sample_time_min:
            self._sync_live_meal_schedule_if_available()

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

        # This is the current CGM for the next action.
        self.current_cgm = raw_cgm

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
            #reward -= 1000.0
            # Keep this commented if you intentionally do not want early termination:
            terminated = True
            info["terminal_reason"] = "cgm_out_of_bounds"

        info["original_reward"] = original_reward
        info["reward_type"] = self.reward_type

        info["proposed_insulin"] = float(proposed_insulin)
        info["shielded_insulin"] = float(delivered)
        info["shield_active"] = float(shield_active)
        info["shield_reason"] = shield_reason
        info["shield_cgm"] = float(shield_cgm)
        info["shield_bg_threshold"] = float(self.shield_bg_threshold)

        self._add_diagnostics_info(
            info=info,
            features=features,
            insulin=delivered,
            raw_action=raw_policy_action,
            policy_action=policy_action,
        )

        self.control_step_count += 1

        if (
            not terminated
            and self.control_max_episode_steps is not None
            and self.control_step_count >= self.control_max_episode_steps
        ):
            truncated = True
            info["terminal_reason"] = info.get("terminal_reason", "control_time_limit")

        info["control_step_count"] = int(self.control_step_count)

        return wrapped_obs, reward, terminated, truncated, info

    def _sync_live_meal_schedule_if_available(self) -> None:
        """
        Sync the wrapper's meal schedule from the scenario.

        For semi_random_hb with noisy actual delivery, this intentionally uses
        get_announced_meal_schedule() when available. That means:
            - PPO gets warning based on announced meals.
            - SimGlucose receives actual noisy meals through scenario.get_action().
        """
        try:
            raw_env = self.env.unwrapped
            inner_env = getattr(raw_env, "env", None)
            live_scenario = getattr(inner_env, "custom_scenario", None)

            if live_scenario is None:
                return

            if hasattr(live_scenario, "get_announced_meal_schedule"):
                schedule = live_scenario.get_announced_meal_schedule()
            elif hasattr(live_scenario, "get_meal_schedule"):
                schedule = live_scenario.get_meal_schedule()
            else:
                return

            synced_schedule: list[tuple[int, float]] = []
            for minute, carbs in schedule:
                synced_schedule.append((int(minute) % 1440, float(carbs)))

            synced_schedule.sort(key=lambda item: item[0])
            self.meal_schedule = synced_schedule

        except Exception:
            pass

    def _build_obs_and_features(self, obs: Any) -> tuple[np.ndarray, dict[str, float]]:
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

            # Måltidsstørrelsen er kun synlig i warning-vinduet.
            # This is based on announced meals.
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

        info["plot_shield_active"] = float(info.get("shield_active", 0.0))
        info["plot_proposed_insulin"] = float(info.get("proposed_insulin", insulin))
        info["plot_shielded_insulin"] = float(info.get("shielded_insulin", insulin))
        info["plot_shield_cgm"] = float(info.get("shield_cgm", features["cgm"]))
        info["plot_shield_bg_threshold"] = float(
            info.get("shield_bg_threshold", self.shield_bg_threshold)
        )

    def _get_history(self) -> pd.DataFrame:
            """
            Return raw SimGlucose simulator history for the current episode.
            """
            return self.unwrapped.env.env.show_history().reset_index()

    def _store_current_history(self) -> None:
        try:
            new_history = self._get_history()
        except Exception:
            return

        if new_history is None or len(new_history) == 0:
            return

        new_history = new_history.copy()

        self.history.append(new_history)
        self.history_index += 1


    def get_history_df(self) -> pd.DataFrame:
        if len(self.history) == 0:
            return pd.DataFrame()

        return pd.concat(
            self.history,
            axis=0,
            keys=range(len(self.history)),
        )


    def clear_history(self) -> None:
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
    shield_bg_threshold: float = 50.0,
    use_bb_warmup: bool = False,
    amount_noise_std_fraction: float = 0.15,
    actual_time_noise_std_min: float = 0.0,
    actual_time_noise_clip_min: float = 30.0,
) -> gym.Env:
    if reward_type not in REWARD_FNS:
        raise ValueError(
            f"Ukendt reward_type={reward_type!r}. Forventede en af: {list(REWARD_FNS)}"
        )

    if max_insulin_action <= 0:
        raise ValueError("max_insulin_action skal være > 0.")

    if shield_bg_threshold <= 0:
        raise ValueError("shield_bg_threshold skal være > 0.")

    if amount_noise_std_fraction < 0:
        raise ValueError("amount_noise_std_fraction skal være >= 0.")

    if actual_time_noise_std_min < 0:
        raise ValueError("actual_time_noise_std_min skal være >= 0.")

    if actual_time_noise_clip_min < 0:
        raise ValueError("actual_time_noise_clip_min skal være >= 0.")

    bb_warmup_steps = 0

    registered_max_episode_steps = int(max_episode_steps)

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
            seed=None,#seed,
            time_std_multiplier=time_std_multiplier,
            include_snacks=include_snacks,
            amount_noise_std_fraction=amount_noise_std_fraction,
            actual_time_noise_std_min=actual_time_noise_std_min,
            actual_time_noise_clip_min=actual_time_noise_clip_min,
        )

        # Important:
        # Wrapper gets announced meals for warning/meal_size.
        # SimGlucose receives actual noisy meals through sim_scenario.get_action().
        if hasattr(sim_scenario, "get_announced_meal_schedule"):
            wrapper_schedule = sim_scenario.get_announced_meal_schedule()
        else:
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
        max_episode_steps=registered_max_episode_steps,
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
        control_max_episode_steps=max_episode_steps,
        bb_warmup_steps=bb_warmup_steps,
        use_bb_warmup=use_bb_warmup,
        shield_bg_threshold=shield_bg_threshold,
    )

    return env