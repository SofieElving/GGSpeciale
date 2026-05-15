from __future__ import annotations

import numpy as np

def softplus(z, sharpness=8.0):
    z = np.asarray(z)
    return np.log1p(np.exp(sharpness * z)) / sharpness


def glucose_insulin_reward(
    bg: float,
    insulin: float,
    raw_insulin: float,
    max_insulin_action: float = 5.0,
) -> float:
    bg = float(bg)
    insulin = float(insulin)
    raw_insulin = float(raw_insulin)

    target = 112.0

    if bg <= target:
        target_reward = 40 * np.exp(-((bg - target) / 25.0) ** 2)
    else:
        target_reward = 40 * np.exp(-((bg - target) / 40.0) ** 2)

    # Smoothly increasing penalties
    low_penalty = 3.0 * softplus((70.0 - bg) / 10.0) ** 2
    severe_low_penalty = 6.0 * softplus((54.0 - bg) / 7.0) ** 2

    high_penalty = 2 * softplus((bg - 180.0) / 30.0) ** 2
    severe_high_penalty = 0.7 * softplus((bg - 250.0) / 18.0) ** 2
    

    return float(
        target_reward
        - low_penalty
        - severe_low_penalty
        - high_penalty
        - severe_high_penalty
    )