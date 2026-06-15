from __future__ import annotations

import numpy as np

def risk(BG):
    """
    Risk is a percentage - ranging from 0 to 100%.
    The 20 and 600 mg/dl are just the values to which the risk formula was fit. 
    The aim is to make the risk maximum when it is either 20 or 600.
    The units in the paper below are different (mmol/l), but in our units (mg/dl) these limits are 20 and 600.

    Reference, in particular see appendix for the derivation of risk:
    https://diabetesjournals.org/care/article/20/11/1655/21162/Symmetrization-of-the-Blood-Glucose-Measurement

    """
    MIN_BG = 20.0
    MAX_BG = 600.0
    if BG <= MIN_BG: 
        return (100.0, 0.0, 100.0)
    if BG >= MAX_BG:
        return (0.0, 100.0, 100.0)
    
    U = 1.509 * (np.log(BG)**1.084 - 5.381)

    ri = 10 * U**2

    rl, rh = 0.0, 0.0
    if U <= 0:
        rl = ri
    if U >= 0:
        rh = ri
    return (rl, rh, ri)


def glucose_insulin_reward(
    bg: float,
    insulin: float,
    raw_insulin: float,
    max_insulin_action: float = 5.0,
) -> float:
    
    bg = float(bg)    
    _, _, r = risk(bg)

    r1 = (1 - (r / 180))**2

    if 70.0 <= bg <= 180.0:
        return r1
    elif (50.0 <= bg < 70.0) or (bg > 180):
        return r1*0.5

    return 0.0
