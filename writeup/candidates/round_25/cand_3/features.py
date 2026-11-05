"""Frame-level features for round 25 cand 3 (Envelope Match). Imported by analysis.py; cached in data/cache/r25c3.

Movement type = the first 3.0 s of a start from a stance: Combine position drills that begin from a stance
(matched per group) vs randomly sampled NFL snaps (from the snap, cut at the throw or sack).
Per frame: speed v (yd/s), forward acceleration dv/dt (yd/s^2) and sideways (centripetal) acceleration v^2/r (g),
from Savitzky-Golay smoothed x, y. A player's envelope = 90th percentile of forward and of sideways acceleration
inside three speed bands (1-3, 3-5, 5-7 yd/s), pooled over all his frames on that side.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[4]
PROC = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "cache" / "r25c3"
CACHE.mkdir(parents=True, exist_ok=True)

DT = 0.1
YD_PER_G = 9.80665 / 0.9144
WIN = 30            # 3.0 s
ON = 1.0            # movement onset in drills: first frame at 1 yd/s
BANDS = [(1, 3), (3, 5), (5, 7)]
ENDS = ["pass_forward", "qb_sack", "qb_strip_sack", "pass_shovel"]

DRILLS = {
    "DL": ["PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL"],
    "OL": ["PASS_RUSH_DROP", "OL_PULL_DRILL_LONG_PULL_RIGHT", "OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT",
           "OL_PULL_DRILL_FOLD_BLOCK_RIGHT"],
    "WR": ["GO_ROUTE_RIGHT", "SLANT_ROUTE_LEFT", "COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT",
           "POST_CORNER_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT", "SPEED_OUT_ROUTE_LEFT", "SLOT_SAIL_ROUTE_RIGHT",
           "SLOT_STRIKE_ROUTE_LEFT", "RED_ZONE_FADE_RIGHT"],
    "TE": ["CORNER_ROUTE", "IN_ROUTE_LEFT", "HOOK_ROUTE_RIGHT", "FLAT_ROUTE_LEFT", "WHEEL_ROUTE_LEFT",
           "END_ZONE_FADE_RIGHT"],
    "DB": ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "W_DRILL",
           "BOX_DRILL", "LINE_DRILL", "TERYL_AUSTIN_DRILL_1", "TERYL_AUSTIN_DRILL_2"],
}


def kin(x, y):
    """speed, forward accel, sideways g from positions (Savitzky-Golay, 7 frames, quadratic)."""
    n = len(x)
    w = 7 if n >= 7 else (n if n % 2 else n - 1)
    xs, ys = savgol_filter(x, w, 2), savgol_filter(y, w, 2)
    vx, vy = savgol_filter(xs, w, 2, deriv=1, delta=DT), savgol_filter(ys, w, 2, deriv=1, delta=DT)
    ax, ay = savgol_filter(xs, w, 2, deriv=2, delta=DT), savgol_filter(ys, w, 2, deriv=2, delta=DT)
    v = np.hypot(vx, vy)
    vs = np.maximum(v, 0.5)
    along = (vx * ax + vy * ay) / vs
    lat = np.abs(vx * ay - vy * ax) / vs / YD_PER_G
    return xs, ys, v, along, lat


def _cached(name, fn):
    f = CACHE / f"{name}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    out = fn()
    pickle.dump(out, open(f, "wb"))
    return out


def drill_frames():
    ev = pd.read_parquet(PROC / "combine_events.parquet")
    keep = pd.concat([ev[(ev.position_group == g) & ev.drill_name.isin(d)] for g, d in DRILLS.items()])
    keep = keep[["event_id", "nfl_id", "draft_year", "position_group", "drill_name", "attempt_no"]]
    fr = pd.read_parquet(PROC / "combine_frames.parquet", filters=[("event_id", "in", list(keep.event_id))],
                         columns=["event_id", "frame", "entity_type", "x", "y", "s"])
    fr = fr[fr.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
    out, paths = [], {}
    for eid, g in fr.groupby("event_id", sort=False):
        s = g.s.to_numpy()
        hit = np.flatnonzero(s >= ON)
        if len(hit) == 0:
            continue
        i0 = max(int(hit[0]) - 2, 0)
        x, y = g.x.to_numpy()[i0:i0 + WIN + 1], g.y.to_numpy()[i0:i0 + WIN + 1]
        if len(x) < 15:
            continue
        xs, ys, v, al, la = kin(x, y)
        out.append(pd.DataFrame({"event_id": eid, "t": np.arange(len(x)) * DT, "v": v, "al": al, "la": la}))
        paths[eid] = (xs, ys, v, al, la)
    df = pd.concat(out, ignore_index=True).merge(keep, on="event_id")
    return df, paths


def game_frames():
    gp = pd.read_parquet(PROC / "game_plays_sample.parquet")
    pl = pd.read_parquet(PROC / "players.parquet", columns=["nfl_id", "position_group"])
    K = ["game_id", "play_id", "nfl_id"]
    gp = gp[gp.sample_kind == "random"].drop_duplicates(K).merge(pl, on="nfl_id")
    gf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "x", "y", "event"])
    gf = gf.merge(gp[K + ["snap_frame", "season"]], on=K)
    ends = gf[gf.event.isin(ENDS)].groupby(K).frame.min().rename("end_frame")
    gf = gf.join(ends, on=K).sort_values(K + ["frame"])
    out, paths = [], {}
    for key, h in gf.groupby(K, sort=False):
        sf = h.snap_frame.iloc[0]
        if pd.isna(sf):
            continue
        e = h.end_frame.iloc[0]
        last = sf + WIN if pd.isna(e) else min(sf + WIN, e)
        w = h[(h.frame >= sf) & (h.frame <= last)]
        if len(w) < 10:
            continue
        xs, ys, v, al, la = kin(w.x.to_numpy(), w.y.to_numpy())
        out.append(pd.DataFrame({"game_id": key[0], "play_id": key[1], "nfl_id": key[2],
                                 "t": np.arange(len(w)) * DT, "v": v, "al": al, "la": la}))
        paths[key] = (xs - xs[0], ys - ys[0], v, al, la)
    df = pd.concat(out, ignore_index=True).merge(gp[K + ["position_group", "season"]], on=K)
    return df, paths


def load():
    return _cached("drill_frames", drill_frames), _cached("game_frames", game_frames)


def envelope(df, by, q=0.9, min_frames=5):
    """90th pct of forward accel and sideways g per speed band. Columns al_b0..2, la_b0..2."""
    rows = []
    for lo_hi_i, (lo, hi) in enumerate(BANDS):
        m = df[(df.v >= lo) & (df.v < hi)]
        g = m.groupby(by)
        a = g.al.quantile(q).rename(f"al_b{lo_hi_i}")
        l_ = g.la.quantile(q).rename(f"la_b{lo_hi_i}")
        n = g.size()
        a[n < min_frames] = np.nan
        l_[n < min_frames] = np.nan
        rows += [a, l_]
    return pd.concat(rows, axis=1)


ELEMS = [f"{k}_b{i}" for k in ["al", "la"] for i in range(3)]
