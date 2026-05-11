from __future__ import annotations


def glucose_insulin_reward(
    bg: float,
    insulin: float,
    raw_insulin: float,
    max_insulin_action: float = 6.0,
) -> float:
    """
    Custom reward for SimGlucose RL.

    Encourages glucose in range, penalizes red-zone glucose heavily,
    and discourages excessive insulin requests.
    """
    bg = float(bg)
    insulin = float(insulin)
    raw_insulin = float(raw_insulin)

    if 70.0 <= bg <= 180.0:
        glucose_reward = 1.0 - 0.0002 * (bg - 110.0) ** 2

    elif 54.0 <= bg < 70.0:
        glucose_reward = -4.0 - ((70.0 - bg) / 10.0) ** 2

    elif bg < 54.0:
        glucose_reward = -12.0 - ((54.0 - bg) / 10.0) ** 2

    elif 180.0 < bg <= 250.0:
        glucose_reward = -2.0 - ((bg - 180.0) / 35.0) ** 2

    else:
        glucose_reward = -8.0 - ((bg - 250.0) / 40.0) ** 2


    insulin_penalty = 0.001 * insulin

    return float(glucose_reward - insulin_penalty)