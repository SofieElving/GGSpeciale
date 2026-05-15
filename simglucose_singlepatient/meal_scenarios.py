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
    """
    Returns mean carb targets for:
        breakfast, lunch, dinner, snack
    following the inspiration you shared.
    """
    if kind == "child":
        height = 140.0

    elif kind == "adolescent":
        height = 170.0

    else:
        height = 177.0

    bmr = 66.5 + (13.75 * weight) + (5.003 * height) - (6.755 * age)
    total_carbs = ((1.2 * bmr) * 0.45) / 4.0

    # Same meal-ratio idea as your inspiration
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
    Generates a daily scenario with semi-random meal times and patient-specific
    meal sizes based on body weight and kind.

    Exposes self.scenario in the same format:
        {
            "meal": {
                "time": [...],    # minute-of-day
                "amount": [...],  # grams of carbs
            }
        }
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
    ):
        super().__init__(start_time=start_time)
        self.patient_name = patient_name
        self.seed = seed
        self.time_std_multiplier = float(time_std_multiplier)
        self.include_snacks = bool(include_snacks)
        self.meal_duration_min = int(meal_duration_min)
        self.deterministic_meal_size = bool(deterministic_meal_size)
        self.deterministic_meal_time = bool(deterministic_meal_time)
        self.deterministic_meal_occurrence = bool(deterministic_meal_occurrence)

        self.bw, self.age, self.kind = get_patient_bw_and_kind(patient_name)
        self.reset()

    def reset(self):
        self.random_gen = np.random.RandomState(self.seed)
        self.scenario = self.create_scenario()

    def create_scenario(self):
        scenario = {"meal": {"time": [], "amount": []}}

        mu_b, mu_l, mu_d, mu_s = harris_benedict(self.bw, self.age, self.kind,)

        # breakfast, snack1, lunch, snack2, dinner, snack3
        probs = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
        if not self.include_snacks:
            probs = [0.95, 0.0, 0.95, 0.0, 0.95, 0.0]

        time_lb = np.array([5, 9, 10, 14, 16, 20]) * 60
        time_ub = np.array([9, 10, 14, 16, 20, 23]) * 60
        time_mu = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60
        time_sigma = np.array([60, 30, 60, 30, 60, 30]) * self.time_std_multiplier

        amount_mu = np.array([mu_b, mu_s, mu_l, mu_s, mu_d, mu_s])
        amount_sigma = amount_mu * 0.15

        for p, tlb, tub, tbar, tsd, mbar, msd in zip(
            probs, time_lb, time_ub, time_mu, time_sigma, amount_mu, amount_sigma
        ):
            if self.random_gen.rand() < p or self.deterministic_meal_occurrence:
                tmeal = np.round(
                    truncnorm.rvs(
                        a=(tlb - tbar) / tsd,
                        b=(tub - tbar) / tsd,
                        loc=tbar,
                        scale=tsd,
                        random_state=self.random_gen,
                    )
                )

                ameal = max(round(self.random_gen.normal(mbar, msd)), 0)

                if self.deterministic_meal_time:
                    tmeal = np.round(tbar)
                if self.deterministic_meal_size:
                    ameal = round(mbar)

                scenario["meal"]["time"].append(int(tmeal))
                scenario["meal"]["amount"].append(float(ameal))

        # sort by time
        pairs = sorted(zip(scenario["meal"]["time"], scenario["meal"]["amount"]))
        scenario["meal"]["time"] = [t for t, _ in pairs]
        scenario["meal"]["amount"] = [a for _, a in pairs]
        return scenario

    def get_action(self, t):
        delta_t = t - datetime.combine(t.date(), datetime.min.time())
        t_sec = delta_t.total_seconds()

        # regenerate each day
        if t_sec < 1:
            self.scenario = self.create_scenario()

        t_min = int(np.floor(t_sec / 60.0))

        for idx, meal_time in enumerate(self.scenario["meal"]["time"]):
            if meal_time <= t_min < meal_time + self.meal_duration_min:
                meal_amt = self.scenario["meal"]["amount"][idx] / self.meal_duration_min
                return Action(meal=meal_amt)

        return Action(meal=0)

    def get_meal_schedule(self) -> list[tuple[int, float]]:
        return list(zip(self.scenario["meal"]["time"], self.scenario["meal"]["amount"]))

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
    """
    Fixed meal times with patient-specific Harris-Benedict meal sizes.

    Times are minute-of-day.
    Amounts are grams carbohydrate.
    """
    bw, age, kind = get_patient_bw_and_kind(patient_name)
    b, l, d, s = harris_benedict(bw, age, kind)

    return [
        (7 * 60, float(round(b))),    # breakfast
        (12 * 60, float(round(l))),   # lunch
        (16 * 60, float(round(s))),   # snack
        (18 * 60, float(round(d))),   # dinner
        (22 * 60, float(round(s))),   # late snack
    ]