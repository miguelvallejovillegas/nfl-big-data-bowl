"""Kinematic features from 10 Hz x/y tracking.

The provided `dis` under-counts and `a` is an unsigned magnitude, so everything derived here comes
from positions (Savitzky-Golay smoothed) plus the provided speed `s`, which tracks well.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

DT = 0.1
YARD_SPLITS = (10.0, 20.0, 40.0)


def _smooth(v: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    if len(v) < window:
        return v.astype(float)
    return savgol_filter(v, window, poly)


def _deriv(v: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    if len(v) < window:
        return np.gradient(v, DT)
    return savgol_filter(v, window, poly, deriv=1, delta=DT)


def frame_kinematics(x: np.ndarray, y: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame derived columns for one event."""
    xs, ys = _smooth(x), _smooth(y)
    vx, vy = _deriv(xs), _deriv(ys)
    speed_xy = np.hypot(vx, vy)
    accel = _deriv(_smooth(s, 5, 2), 5, 2)  # signed, along path
    heading = (np.degrees(np.arctan2(vx, vy)) + 360) % 360  # NFL convention: 0 = +y, 90 = +x
    return {
        "vx": vx,
        "vy": vy,
        "speed_xy": speed_xy,
        "accel": accel,
        "heading": heading,
    }


def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b) % 360
    return np.where(d > 180, 360 - d, d)


def detect_breaks(heading: np.ndarray, s: np.ndarray, window: int = 5, min_turn: float = 50.0, min_speed: float = 1.5):
    """Indices where heading changes by > min_turn degrees over `window` frames while moving.

    Returns list of (idx, turn_deg). Consecutive detections are merged into the sharpest one.
    """
    n = len(heading)
    if n < 2 * window + 1:
        return []
    turn = np.zeros(n)
    for i in range(window, n - window):
        turn[i] = _angle_diff(heading[i - window], heading[i + window])[()]
    cand = np.where((turn > min_turn) & (s > min_speed))[0]
    breaks: list[tuple[int, float]] = []
    for i in cand:
        if breaks and i - breaks[-1][0] <= 2 * window:
            if turn[i] > breaks[-1][1]:
                breaks[-1] = (int(i), float(turn[i]))
        else:
            breaks.append((int(i), float(turn[i])))
    return breaks


def event_features(x: np.ndarray, y: np.ndarray, s: np.ndarray, drill_type: str) -> dict:
    """Summary features for one drill attempt (PLAYER rows, ordered by time)."""
    n = len(x)
    k = frame_kinematics(x, y, s)
    t = np.arange(n) * DT
    path = float(np.sum(np.hypot(np.diff(x), np.diff(y)))) if n > 1 else 0.0
    disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    top = int(np.argmax(s))
    accel, heading = k["accel"], k["heading"]
    feats = {
        "n_frames": n,
        "duration_s": round(float(t[-1]), 1),
        "path_length_yd": round(path, 2),
        "displacement_yd": round(disp, 2),
        "straightness": round(disp / path, 3) if path > 0 else None,
        "top_speed": round(float(s.max()), 2),
        "time_to_top_speed_s": round(float(t[top]), 1),
        "mean_speed": round(float(s.mean()), 2),
        "start_speed": round(float(s[0]), 2),
        "peak_accel": round(float(np.nanmax(accel)), 2),
        "peak_decel": round(float(-np.nanmin(accel)), 2),
        "time_to_5yps_s": _time_to_speed(t, s, 5.0),
        "time_to_8yps_s": _time_to_speed(t, s, 8.0),
        "n_breaks": 0,
        "sharpest_break_deg": None,
        "break_speed_in": None,
        "break_speed_min": None,
        "break_speed_out": None,
        "break_time_s": None,
        "break_decel": None,
        "break_reaccel": None,
    }
    if drill_type == "FORTY_YARD_DASH":
        d = np.abs(x - x[0])
        for yd in YARD_SPLITS:
            idx = np.argmax(d >= yd) if (d >= yd).any() else None
            feats[f"split_{int(yd)}yd_s"] = round(float(_interp_cross(t, d, yd)), 2) if idx else None
        feats["speed_at_10yd"] = _speed_at(d, s, 10.0)
        feats["speed_at_20yd"] = _speed_at(d, s, 20.0)
        feats["speed_last_10yd"] = round(float(s[d >= 30].mean()), 2) if (d >= 30).any() else None
    else:
        for yd in YARD_SPLITS:
            feats[f"split_{int(yd)}yd_s"] = None
        feats["speed_at_10yd"] = feats["speed_at_20yd"] = feats["speed_last_10yd"] = None

    brks = detect_breaks(heading, s)
    feats["n_breaks"] = len(brks)
    if brks:
        i, turn = max(brks, key=lambda b: b[1])
        w = 10  # 1 s window
        lo, hi = max(0, i - w), min(n, i + w + 1)
        s_in = float(s[lo:i + 1].max())
        s_min = float(s[lo:hi].min())
        s_out = float(s[i:hi].max())
        imin = lo + int(np.argmin(s[lo:hi]))
        feats.update(
            sharpest_break_deg=round(turn, 1),
            break_speed_in=round(s_in, 2),
            break_speed_min=round(s_min, 2),
            break_speed_out=round(s_out, 2),
            break_time_s=round(float(t[i]), 1),
            break_decel=round(float(-np.nanmin(accel[lo:imin + 1])) if imin > lo else 0.0, 2),
            break_reaccel=round(float(np.nanmax(accel[imin:hi])) if hi > imin else 0.0, 2),
        )
    return feats


def _time_to_speed(t: np.ndarray, s: np.ndarray, target: float):
    m = s >= target
    return round(float(t[np.argmax(m)]), 1) if m.any() else None


def _speed_at(d: np.ndarray, s: np.ndarray, yd: float):
    m = d >= yd
    return round(float(s[np.argmax(m)]), 2) if m.any() else None


def _interp_cross(t: np.ndarray, d: np.ndarray, yd: float) -> float:
    i = int(np.argmax(d >= yd))
    if i == 0:
        return 0.0
    d0, d1 = d[i - 1], d[i]
    f = (yd - d0) / (d1 - d0) if d1 != d0 else 0.0
    return float(t[i - 1] + f * DT)
