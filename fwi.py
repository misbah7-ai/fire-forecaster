"""
Canadian Forest Fire Weather Index (FWI) System -- Van Wagner & Pickett (1985) equations in
lean numpy, NO external fire dependency. Validated against the canonical cffdrs single-step
reference vector (temp=17, rh=42, ws=25, rain=0, FFMC/DMC/DC=85/6/15, month=4 ->
FFMC=87.69, DMC=8.55, DC=19.01, ISI=10.85, BUI=8.49, FWI=10.10) -- reproduced to 2 dp.

Used for the AUDIT ONLY (never as a model feature). The three moisture codes (FFMC/DMC/DC)
carry drought memory day to day; ISI/BUI/FWI are the circular ones the dataset's label derives
from. Startup seeding is the FWI-system standard: FFMC 85, DMC 6, DC 15 (persisted in bundle).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FWI_SEEDING = {"FFMC": 85.0, "DMC": 6.0, "DC": 15.0}

# DMC effective day-length by month (northern hemisphere)
_DMC_DAYLENGTH = {1: 6.5, 2: 7.5, 3: 9.0, 4: 12.8, 5: 13.9, 6: 13.9,
                  7: 12.4, 8: 10.9, 9: 9.4, 10: 8.0, 11: 7.0, 12: 6.0}
# DC day-length factor by month (northern hemisphere)
_DC_DAYLENGTH = {1: -1.6, 2: -1.6, 3: -1.6, 4: 0.9, 5: 3.8, 6: 5.8,
                 7: 6.4, 8: 5.0, 9: 2.4, 10: 0.4, 11: -1.6, 12: -1.6}


def ffmc_next(ffmc_prev, temp, rh, wind, rain):
    rh = min(rh, 100.0)
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)
    if rain > 0.5:
        rf = rain - 0.5
        mr = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
        if mo > 150.0:
            mr += 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf)
        mo = min(mr, 250.0)
    ed = (0.942 * rh ** 0.679 + 11.0 * np.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh)))
    if mo > ed:
        ko = 0.424 * (1 - (rh / 100) ** 1.7) + 0.0694 * np.sqrt(wind) * (1 - (rh / 100) ** 8)
        m = ed + (mo - ed) * 10.0 ** (-(ko * 0.581 * np.exp(0.0365 * temp)))
    else:
        ew = (0.618 * rh ** 0.753 + 10.0 * np.exp((rh - 100.0) / 10.0)
              + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh)))
        if mo < ew:
            kl = (0.424 * (1 - ((100 - rh) / 100) ** 1.7)
                  + 0.0694 * np.sqrt(wind) * (1 - ((100 - rh) / 100) ** 8))
            m = ew - (ew - mo) * 10.0 ** (-(kl * 0.581 * np.exp(0.0365 * temp)))
        else:
            m = mo
    return float(min(max(59.5 * (250.0 - m) / (147.2 + m), 0.0), 101.0))


def dmc_next(dmc_prev, temp, rh, rain, month):
    t = max(temp, -1.1)
    rk = 1.894 * (t + 1.1) * (100.0 - rh) * _DMC_DAYLENGTH[month] * 1e-4
    if rain > 1.5:
        re = 0.92 * rain - 1.27
        mo = 20.0 + np.exp(5.6348 - dmc_prev / 43.43)
        if dmc_prev <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65.0:
            b = 14.0 - 1.3 * np.log(dmc_prev)
        else:
            b = 6.2 * np.log(dmc_prev) - 17.2
        mr = mo + 1000.0 * re / (48.77 + b * re)
        dmc_prev = max(244.72 - 43.43 * np.log(mr - 20.0), 0.0)
    return float(dmc_prev + rk)


def dc_next(dc_prev, temp, rain, month):
    t = max(temp, -2.8)
    pe = max((0.36 * (t + 2.8) + _DC_DAYLENGTH[month]) / 2.0, 0.0)
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        qr = 800.0 * np.exp(-dc_prev / 400.0) + 3.937 * rd
        dc_prev = max(400.0 * np.log(800.0 / qr), 0.0)
    return float(dc_prev + pe)


def isi(ffmc, wind):
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * np.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return float(0.208 * np.exp(0.05039 * wind) * ff)


def bui(dmc, dc):
    if dmc == 0 and dc == 0:
        return 0.0
    if dmc <= 0.4 * dc:
        b = 0.8 * dmc * dc / (dmc + 0.4 * dc)
    else:
        b = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
    return float(max(b, 0.0))


def fwi(isi_v, bui_v):
    if bui_v <= 80.0:
        fd = 0.626 * bui_v ** 0.809 + 2.0
    else:
        fd = 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui_v))
    b = 0.1 * isi_v * fd
    return float(np.exp(2.72 * (0.434 * np.log(b)) ** 0.647)) if b > 1.0 else float(b)


def compute_codes(df, temp="temp", rh="RH", wind="Ws", rain="Rain",
                  ffmc0=FWI_SEEDING["FFMC"], dmc0=FWI_SEEDING["DMC"], dc0=FWI_SEEDING["DC"]):
    """Sequentially compute FFMC/DMC/DC/ISI/BUI/FWI for ONE region's daily frame in date order
    (codes carry forward within it). Returns a copy with *_c columns added."""
    d = df.sort_values("date").reset_index(drop=True).copy()
    months = d["date"].dt.month.to_numpy()
    T, H, W, R = (d[temp].to_numpy(float), d[rh].to_numpy(float),
                  d[wind].to_numpy(float), d[rain].to_numpy(float))
    n = len(d)
    cols = {k: np.empty(n) for k in ("FFMC_c", "DMC_c", "DC_c", "ISI_c", "BUI_c", "FWI_c")}
    fp, dp, cp = ffmc0, dmc0, dc0
    for i in range(n):
        fp = ffmc_next(fp, T[i], H[i], W[i], R[i])
        dp = dmc_next(dp, T[i], H[i], R[i], int(months[i]))
        cp = dc_next(cp, T[i], R[i], int(months[i]))
        iv, bv = isi(fp, W[i]), bui(dp, cp)
        cols["FFMC_c"][i], cols["DMC_c"][i], cols["DC_c"][i] = fp, dp, cp
        cols["ISI_c"][i], cols["BUI_c"][i], cols["FWI_c"][i] = iv, bv, fwi(iv, bv)
    for k, v in cols.items():
        d[k] = v
    return d


def _self_test():
    f = ffmc_next(85, 17, 42, 25, 0); d = dmc_next(6, 17, 42, 0, 4); c = dc_next(15, 17, 0, 4)
    iv, bv = isi(f, 25), bui(d, c); wv = fwi(iv, bv)
    exp = dict(FFMC=87.69, DMC=8.55, DC=19.01, ISI=10.85, BUI=8.49, FWI=10.10)
    got = dict(FFMC=f, DMC=d, DC=c, ISI=iv, BUI=bv, FWI=wv)
    ok = all(abs(got[k] - exp[k]) < 0.01 for k in exp)
    for k in exp:
        print(f"  {k:5s} got {got[k]:7.2f}  exp {exp[k]:7.2f}  {'OK' if abs(got[k]-exp[k])<0.01 else 'MISMATCH'}")
    assert ok, "FWI implementation does not match the canonical cffdrs reference vector"
    print("fwi.py self-test PASSED (matches canonical cffdrs vector to 2 dp)")


if __name__ == "__main__":
    _self_test()
