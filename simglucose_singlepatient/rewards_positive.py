from __future__ import annotations

import numpy as np


def glucose_insulin_reward(
    bg: float,
    insulin: float,
    raw_insulin: float,
    max_insulin_action: float = 5.0,
) -> float:

    bg = float(bg)
    insulin = float(insulin)
    raw_insulin = float(raw_insulin)

    if 90.0 < bg < 180.0:
        x = (bg - 90.0) / 45.0 - 1.0
        return float(np.exp(-1.0 / (1.0 - x**2)))

    return 0.0