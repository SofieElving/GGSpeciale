from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd
import pkg_resources
from scipy.stats import truncnorm

from simglucose.simulation.scenario import Scenario, Action

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/vpatient_params.csv"
)
QUEST_FILE = pkg_resources.resource_filename(
    "simglucose",
    "params/Quest.csv"
)

'''
havily inspired by https://github.com/girtel/AIML4Diabetes/blob/master/bb-pid/simglucose/simulation/scenario_gen.py
'''

def get_patient_bw_and_kind(
    patient_name: str,
) -> tuple[float, float, str]:

    vpatient_params = pd.read_csv(PATIENT_PARA_FILE)
    quest_params = pd.read_csv(QUEST_FILE)

    bw = (
        vpatient_params
        .query("Name == @patient_name")["BW"]
        .item()
    )

    age = (
        quest_params
        .query("Name == @patient_name")["Age"]
        .item()
    )

    kind = patient_name.split("#")[0]
    return float(bw), float(age), kind

def harris_benedict(
    weight: float,
    age: float,
    kind: str,
) -> tuple[float, float, float, float]:
    
    if kind == "child":
        height = 140.0

    elif kind == "adolescent":
        height = 170.0

    else:
        height = 177.0

    bmr = 66.5 + (13.75 * weight) + (5.003 * height) - (6.755 * age)
    total_carbs = ((1.2 * bmr) * 0.45) / 4.0

    adj = 1.1 + 1.3 + 1.55 + 3 * 0.15
    b_ratio = 1.1 / adj
    l_ratio = 1.3 / adj
    d_ratio = 1.55 / adj
    s_ratio = 0.15 / adj

    return (
        total_carbs * b_ratio,
        total_carbs * l_ratio,
        total_carbs * d_ratio,
        total_carbs * s_ratio,
    )


class SemiRandomHarrisonBenedictScenario(Scenario):
    """
    Semi-random meal scenario with optional mismatch between announced meals
    and actually delivered meals.

    - announced_scenario:
        Used for meal_warning and meal_size.

    - scenario:
        Used for actual SimGlucose meal delivery.

    This allows meal warnings to be imperfect.
    """

    def __init__(
        self,
        patient_name: str,
        start_time=None,
        seed=None,
        time_std_multiplier: float = 1.0,
        include_snacks: bool = True,
        meal_duration_min: int = 1,
        deterministic_meal_size: bool = False,
        deterministic_meal_time: bool = False,
        deterministic_meal_occurrence: bool = False,
        amount_noise_std_fraction: float = 0.15,
        actual_time_noise_std_min: float = 0.0,
        actual_time_noise_clip_min: float = 30.0,
    ):
        """
        Args:
            amount_noise_std_fraction:
                Relative std for actual meal-size noise.
                Example: 0.15 means std = 15% of mean meal size.
                Set to 0.0 for deterministic meal sizes.

            actual_time_noise_std_min:
                Std in minutes added to the actual delivery time after
                the announced time has been sampled.
                Example: 10.0 means actual meal time is usually within
                roughly +/- 20 min of announced time.

            actual_time_noise_clip_min:
                Maximum absolute time mismatch.
                Example: 30.0 clips actual delivery time to announced +/- 30 min.
        """
        super().__init__(start_time=start_time)

        if amount_noise_std_fraction < 0:
            raise ValueError("amount_noise_std_fraction must be >= 0.")

        if actual_time_noise_std_min < 0:
            raise ValueError("actual_time_noise_std_min must be >= 0.")

        if actual_time_noise_clip_min < 0:
            raise ValueError("actual_time_noise_clip_min must be >= 0.")

        self.patient_name = patient_name
        self.seed = seed
        self.time_std_multiplier = float(time_std_multiplier)
        self.include_snacks = bool(include_snacks)
        self.meal_duration_min = int(meal_duration_min)

        self.deterministic_meal_size = bool(deterministic_meal_size)
        self.deterministic_meal_time = bool(deterministic_meal_time)
        self.deterministic_meal_occurrence = bool(deterministic_meal_occurrence)

        self.amount_noise_std_fraction = float(amount_noise_std_fraction)
        self.actual_time_noise_std_min = float(actual_time_noise_std_min)
        self.actual_time_noise_clip_min = float(actual_time_noise_clip_min)

        self.bw, self.age, self.kind = get_patient_bw_and_kind(patient_name)

        self.announced_scenario = {"meal": {"time": [], "amount": []}}
        self.scenario = {"meal": {"time": [], "amount": []}}

        self.reset()

    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
        self.announced_scenario, self.scenario = self.create_scenario()

    def _sample_truncated_normal(self, mean: float, std: float, lb: float, ub: float) -> float:
        if std <= 0:
            return float(np.clip(mean, lb, ub))

        return float(
            truncnorm.rvs(
                a=(lb - mean) / std,
                b=(ub - mean) / std,
                loc=mean,
                scale=std,
                random_state=self.random_gen,
            )
        )

    def _sample_actual_time(self, announced_time: float, lb: float, ub: float) -> int:
        if self.actual_time_noise_std_min <= 0:
            return int(round(announced_time))

        noise = self.random_gen.normal(0.0, self.actual_time_noise_std_min)

        if self.actual_time_noise_clip_min > 0:
            noise = float(
                np.clip(
                    noise,
                    -self.actual_time_noise_clip_min,
                    self.actual_time_noise_clip_min,
                )
            )

        actual_time = announced_time + noise
        actual_time = float(np.clip(actual_time, lb, ub))

        return int(round(actual_time))

    def _sample_amount(self, mean_amount: float) -> float:
        if self.deterministic_meal_size or self.amount_noise_std_fraction <= 0:
            return float(round(mean_amount))

        std = mean_amount * self.amount_noise_std_fraction
        amount = self.random_gen.normal(mean_amount, std)
        return float(max(round(amount), 0.0))

    def create_scenario(self):
        announced = {"meal": {"time": [], "amount": []}}
        actual = {"meal": {"time": [], "amount": []}}

        mu_b, mu_l, mu_d, mu_s = harris_benedict(self.bw, self.age, self.kind)

        # breakfast, snack1, lunch, snack2, dinner, snack3
        probs = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
        if not self.include_snacks:
            probs = [0.95, 0.0, 0.95, 0.0, 0.95, 0.0]

        time_lb = np.array([5, 9, 10, 14, 16, 20], dtype=float) * 60.0
        time_ub = np.array([9, 10, 14, 16, 20, 23], dtype=float) * 60.0
        time_mu = np.array([7, 9.5, 12, 15, 18, 21.5], dtype=float) * 60.0
        time_sigma = (
            np.array([60, 30, 60, 30, 60, 30], dtype=float)
            * self.time_std_multiplier
        )

        amount_mu = np.array([mu_b, mu_s, mu_l, mu_s, mu_d, mu_s], dtype=float)

        for p, tlb, tub, tbar, tsd, mbar in zip(
            probs, time_lb, time_ub, time_mu, time_sigma, amount_mu
        ):
            meal_occurs = self.random_gen.rand() < p or self.deterministic_meal_occurrence

            if not meal_occurs:
                continue

            if self.deterministic_meal_time:
                announced_time = float(round(tbar))
            else:
                announced_time = round(
                    self._sample_truncated_normal(
                        mean=float(tbar),
                        std=float(tsd),
                        lb=float(tlb),
                        ub=float(tub),
                    )
                )

            actual_time = self._sample_actual_time(
                announced_time=float(announced_time),
                lb=float(tlb),
                ub=float(tub),
            )

            announced_amount = float(round(mbar))
            actual_amount = self._sample_amount(float(mbar))

            announced["meal"]["time"].append(int(announced_time))
            announced["meal"]["amount"].append(float(announced_amount))

            actual["meal"]["time"].append(int(actual_time))
            actual["meal"]["amount"].append(float(actual_amount))

        announced_pairs = sorted(zip(announced["meal"]["time"], announced["meal"]["amount"]))
        actual_pairs = sorted(zip(actual["meal"]["time"], actual["meal"]["amount"]))

        announced["meal"]["time"] = [int(t) for t, _ in announced_pairs]
        announced["meal"]["amount"] = [float(a) for _, a in announced_pairs]

        actual["meal"]["time"] = [int(t) for t, _ in actual_pairs]
        actual["meal"]["amount"] = [float(a) for _, a in actual_pairs]

        return announced, actual

    def get_action(self, t):
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()

        # Regenerate each day.
        if t_sec < 1:
            self.announced_scenario, self.scenario = self.create_scenario()

        t_min = int(np.floor(t_sec / 60.0))

        for idx, meal_time in enumerate(self.scenario["meal"]["time"]):
            if meal_time <= t_min < meal_time + self.meal_duration_min:
                meal_amt = self.scenario["meal"]["amount"][idx] / self.meal_duration_min
                return Action(meal=meal_amt)

        return Action(meal=0)

    def get_meal_schedule(self) -> list[tuple[int, float]]:
        return self.get_announced_meal_schedule()

    def get_announced_meal_schedule(self) -> list[tuple[int, float]]:
        return list(
            zip(
                self.announced_scenario["meal"]["time"],
                self.announced_scenario["meal"]["amount"],
            )
        )

    def get_actual_meal_schedule(self) -> list[tuple[int, float]]:
        return list(
            zip(
                self.scenario["meal"]["time"],
                self.scenario["meal"]["amount"],
            )
        )

DEFAULT_MEALS = [
    (7 * 60, 45.0),
    (12 * 60, 70.0),
    (16 * 60, 15.0),
    (18 * 60, 80.0),
    (23 * 60, 10.0),
]


def parse_meal_schedule(
    text: str | None,
    default: list[tuple[int, float]] = DEFAULT_MEALS,
) -> list[tuple[int, float]]:
    """
    Parses:
        "7:45,12:70,16:15,18:80,23:10"

    Meaning:
        hour:grams_carbohydrate

    Returns:
        [(420, 45.0), (720, 70.0), ...]
    """
    if not text:
        return [(int(minute), float(carbs)) for minute, carbs in default]

    meals: list[tuple[int, float]] = []

    for item in text.split(","):
        hour_str, carbs_str = item.strip().split(":")
        meals.append((int(hour_str) * 60, float(carbs_str)))

    return meals


def hb_fixed_meal_schedule(patient_name: str) -> list[tuple[int, float]]:
    
    bw, age, kind = get_patient_bw_and_kind(patient_name)
    b, l, d, s = harris_benedict(bw, age, kind)

    return [
        (7 * 60, float(round(b))),    # breakfast
        (12 * 60, float(round(l))),   # lunch
        (16 * 60, float(round(s))),   # snack
        (18 * 60, float(round(d))),   # dinner
        (22 * 60, float(round(s))),   # late snack
    ]