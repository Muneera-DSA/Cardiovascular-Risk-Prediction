"""ACC/AHA Pooled Cohort Equations (Goff et al., Circulation 2014; 2013 ACC/AHA risk guideline).

10-year risk of a first hard ASCVD event (coronary death, non-fatal MI, fatal or non-fatal stroke)
for adults aged 40-79 without cardiovascular disease. Four sex-by-race equations; the guideline uses
the white equations for people of other races.

    risk = 1 - S10 ** exp(sum(coef * x) - mean)

The test suite reproduces the guideline's worked example (55-year-old, total cholesterol 213,
HDL 50, untreated SBP 120, non-smoker, no diabetes: 2.1% / 3.0% / 5.3% / 6.1%).

Note for this project: the PCE predicts fatal AND non-fatal events, while NHANES-NDI records deaths
only. Its absolute risks are therefore expected to be higher than observed CVD death rates. Its
*ranking* (discrimination) is directly comparable, and a recalibrated version is also reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# term order: ln_age, ln_age_sq, ln_tc, ln_age_x_ln_tc, ln_hdl, ln_age_x_ln_hdl,
#             ln_tsbp, ln_age_x_ln_tsbp, ln_usbp, ln_age_x_ln_usbp, smoker, ln_age_x_smoker, diabetes
COEF = {
    ("female", "white"): ([-29.799, 4.884, 13.540, -3.114, -13.578, 3.149, 2.019, 0.0, 1.957, 0.0, 7.574, -1.665, 0.661], -29.18, 0.9665),
    ("female", "black"): ([17.114, 0.0, 0.940, 0.0, -18.920, 4.475, 29.291, -6.432, 27.820, -6.087, 0.691, 0.0, 0.874], 86.61, 0.9533),
    ("male", "white"): ([12.344, 0.0, 11.853, -2.664, -7.990, 1.769, 1.797, 0.0, 1.764, 0.0, 7.837, -1.795, 0.658], 61.18, 0.9144),
    ("male", "black"): ([2.469, 0.0, 0.302, 0.0, -0.307, 0.0, 1.916, 0.0, 1.809, 0.0, 0.549, 0.0, 0.645], 19.54, 0.8954),
}


def _terms(age, tc, hdl, sbp, treated, smoker, diabetes) -> np.ndarray:
    la, ltc, lhdl, lsbp = np.log(age), np.log(tc), np.log(hdl), np.log(sbp)
    t = np.asarray(treated, float)
    s = np.asarray(smoker, float)
    return np.column_stack([la, la ** 2, ltc, la * ltc, lhdl, la * lhdl,
                            lsbp * t, la * lsbp * t, lsbp * (1 - t), la * lsbp * (1 - t),
                            s, la * s, np.asarray(diabetes, float)])


def linear_predictor(df: pd.DataFrame) -> np.ndarray:
    """Individual sum minus the group mean (the 'relative' linear predictor)."""
    X = _terms(df.age, df.tchol, df.hdl, df.sbp, df.bp_treated, df.smoker, df.diabetes)
    out = np.full(len(df), np.nan)
    for (sex, race), (coef, mean, _) in COEF.items():
        m = ((df.female == 1) == (sex == "female")).to_numpy() & ((df.black == 1) == (race == "black")).to_numpy()
        out[m] = X[m] @ np.asarray(coef) - mean
    return out


def risk_10y(df: pd.DataFrame) -> np.ndarray:
    """10-year ASCVD risk (0-1). Inputs outside the validated ranges are clipped as the guideline
    calculator does (TC 130-320, HDL 20-100, SBP 90-200 mg/dL / mmHg)."""
    d = df.copy()
    d["tchol"] = d.tchol.clip(130, 320)
    d["hdl"] = d.hdl.clip(20, 100)
    d["sbp"] = d.sbp.clip(90, 200)
    lp = linear_predictor(d)
    s10 = np.select([(d.female == 1) & (d.black == 0), (d.female == 1) & (d.black == 1), (d.female == 0) & (d.black == 0)],
                    [COEF[("female", "white")][2], COEF[("female", "black")][2], COEF[("male", "white")][2]],
                    COEF[("male", "black")][2])
    return 1 - s10 ** np.exp(lp)
