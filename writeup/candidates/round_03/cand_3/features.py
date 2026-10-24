"""Shared movement-profile features, computed by one function on Combine drill reps and NFL snaps (10 Hz x/y/s).

Imported by analysis.py. Features per rep (all from movement onset unless stated):
  d1    yards covered in the first second after onset (burst)
  acc   peak forward acceleration within 0-1.0 s of onset (yd/s^2), from smoothed speed
  bend  hardest sideways push held for 0.5 s (lateral/centripetal acceleration, g) in the window
  top   top speed in the window (yd/s)
Game window: snap to snap + 3.0 s. Drill window: whole clip. Reps already moving at the window start are dropped
for d1/acc (onset undefined) but kept for top/bend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

DT = 0.1
ON, PRE = 1.0, 0.5
W1 = 10                      # first second
GAME_WIN = 30                # 3.0 s after snap
YD_PER_G = 9.80665 / 0.9144
T_CURVE = 21                 # 0-2.0 s speed curve from onset (for figures)


def onset(s: np.ndarray, start: int = 0):
    s = np.asarray(s, float)
    if start >= len(s) or s[start] >= ON:
        return None
    hit = np.flatnonzero(s[start:] >= ON)
    if not len(hit):
        return None
    i = j = start + int(hit[0])
    while j > start and s[j - 1] >= PRE and i - j < 3:
        j -= 1
    return j


def rep_features(x, y, s, start: int = 0, end: int | None = None) -> dict | None:
    x, y, s = (np.asarray(a, float) for a in (x, y, s))
    end = len(x) if end is None else min(end, len(x))
    if end - start < 15:
        return None
    xs, ys = savgol_filter(x, 7, 2), savgol_filter(y, 7, 2)
    vx, vy = savgol_filter(xs, 7, 2, deriv=1, delta=DT), savgol_filter(ys, 7, 2, deriv=1, delta=DT)
    ax, ay = savgol_filter(xs, 7, 2, deriv=2, delta=DT), savgol_filter(ys, 7, 2, deriv=2, delta=DT)
    v = np.hypot(vx, vy)
    alat = np.abs(vx * ay - vy * ax) / np.maximum(v, 0.5)
    roll = pd.Series(alat[start:end]).rolling(5, center=True).mean().to_numpy()
    out = {"bend": float(np.nanmax(roll)) / YD_PER_G if np.isfinite(roll).any() else np.nan,
           "top": float(np.nanmax(s[start:end])), "d1": np.nan, "acc": np.nan}
    o = onset(s, start)
    curve = None
    if o is not None and o + W1 < len(x):
        out["d1"] = float(np.hypot(x[o + W1] - x[o], y[o + W1] - y[o]))
        vs = savgol_filter(s, 5, 2)
        out["acc"] = float(np.max(np.diff(vs[o:o + W1 + 1])) / DT)
        if o + T_CURVE <= len(s):
            curve = s[o:o + T_CURVE].copy()
    out["_curve"] = curve
    return out
