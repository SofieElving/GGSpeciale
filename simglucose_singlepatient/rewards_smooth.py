from __future__ import annotations

import numpy as np


def softplus(z, sharpness: float = 8.0):
    """
    Smooth approximation of max(0, z).

    Controls how sharply the penalty activates:
        higher sharpness → closer to ReLU
        lower sharpness → smoother transition
    """
    z = np.asarray(z)
    return np.log1p(np.exp(sharpness * z)) / sharpness


# def glucose_insulin_reward(
#     bg: float,
#     insulin: float,
#     raw_insulin: float,
#     max_insulin_action: float = 6.0,
# ) -> float:


#     bg = float(bg)
#     insulin = float(insulin)
#     raw_insulin = float(raw_insulin)

#     # --- Target reward (peak at 110 mg/dL) ---
#     target_reward = 1.2 * (1.0 - ((bg - 110.0) / 50.0) ** 2)
#     # Alternative sharper peak:
#     # target_reward = 1.5 * np.exp(-((bg - 110.0) / 30.0) ** 2)

#     # --- Smooth glucose penalties ---
#     # Hypoglycemia
#     low_penalty = 3.0 * softplus((70.0 - bg) / 10.0) ** 2
#     severe_low_penalty = 6.0 * softplus((54.0 - bg) / 8.0) ** 2

#     # Hyperglycemia
#     high_penalty = 1.5 * softplus((bg - 180.0) / 35.0) ** 2
#     severe_high_penalty = 3.0 * softplus((bg - 250.0) / 40.0) ** 2




#     return float(
#         target_reward
#         - low_penalty
#         - severe_low_penalty
#         - high_penalty
#         - severe_high_penalty
#     )

# def glucose_insulin_reward(
#     bg: float,
#     insulin: float,
#     raw_insulin: float,
#     max_insulin_action: float = 5.0,
# ) -> float:


#     bg = float(bg)
#     insulin = float(insulin)
#     raw_insulin = float(raw_insulin)


#     # Peak at 110 mg/dL
#     #target_reward = 1.2*(1.0 - ((bg - 110.0) / 50.0) ** 2)
#     target_reward = 15 * np.exp(-((bg - 120.0) / 30.0) ** 2)

#     # Smoothly increasing penalties
#     low_penalty = 3.0 * softplus((70.0 - bg) / 3.0) ** 2
#     severe_low_penalty = 6.0 * softplus((54.0 - bg) / 4.0) ** 2

#     high_penalty = 1.5 * softplus((bg - 170.0) / 15.0) ** 2
#     severe_high_penalty = 3.0 * softplus((bg - 250.0) / 30.0) ** 2

#     return float(
#         target_reward
#         - low_penalty
#         - severe_low_penalty
#         - high_penalty
#         - severe_high_penalty
#     )

def glucose_insulin_reward(
    bg: float,
    insulin: float,
    raw_insulin: float,
    max_insulin_action: float = 5.0,
) -> float:


    bg = float(bg)
    insulin = float(insulin)
    raw_insulin = float(raw_insulin)


    # Peak at 110 mg/dL
    #target_reward = 1.2*(1.0 - ((bg - 110.0) / 50.0) ** 2)
    target_reward = 5 * np.exp(-((bg - 115.0) / 30.0) ** 2)

    # Smoothly increasing penalties
    low_penalty = 3.0 * softplus((70.0 - bg) / 9.0) ** 2
    severe_low_penalty = 6.0 * softplus((54.0 - bg) / 6.0) ** 2

    high_penalty = 1.5 * softplus((bg - 180.0) / 22.0) ** 2
    severe_high_penalty = 3.0 * softplus((bg - 250.0) / 15.0) ** 2

    return float(
        target_reward
        - low_penalty
        - severe_low_penalty
        - high_penalty
        - severe_high_penalty
    )