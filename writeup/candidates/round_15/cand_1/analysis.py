"""Round 15 cand 1: Hoop G day-3 pass-rush shortlist (revision of round 14 cand 1), with the drill-to-game chain for
get-off and bend.

Run from the repo root:
    uv run python writeup/candidates/round_15/cand_1/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (every number the text quotes), frozen_shortlist.json,
board_2023_2025.csv and figures/*.png next to this file, then renders writeup.md from writeup_template.md
({{key}} -> numbers.json value; fails on an unknown key). Frame-level features are cached in data/cache/r15c1
(gitignored). Runtime: about 1 minute.

Design (one primary test, everything else is a check):
  * Metric. Hoop G = hardest 0.5 s of sideways (centripetal) acceleration on the Run-the-Hoop lap, in g, mean over
    runs, z-scored within Combine year. Rush-drill burst = yards covered in the first second after movement onset in
    the pass-rush drill, z within year.
  * Yardstick without tracking. OLS forecast of NFL pressure rate (empirical-Bayes) from log pick, Combine weight,
    NFL EDGE share, NGS athleticism score, official 10-yd split and 3-cone (missing times filled from weight, role,
    NGS). Fit on the 100+ rush linemen of the other two classes (leave-one-class-out).
  * Primary test (frozen in round 13). Day-3/undrafted (pick 103+) AND top third of Hoop G among all hoop runners of
    the class. Hit = NFL pressure rate above the forecast, 30+ rushes. Comparison = the other day-3/undrafted hoop
    runners with 30+ rushes. One row per player, so player-clustered by construction.
  * Checks: time-forward forecast, pick-and-role-only forecast, selection model + inverse-probability weights for
    reaching 30 rushes (new), Mantel-Haenszel class-stratified odds ratio (new), shortlist calibration (new),
    matched-board pairs (new, post hoc), ridge forecast, drop the best rusher, within-class relabelling, Bayesian
    posterior, rush-weighted slope, 14-pair search adjustment, drill-to-game chain, drill-burst equivalence bound (new).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy import stats
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[4]
PROC = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
for _old in FIG.glob("*.png"):
    _old.unlink()
CACHE = ROOT / "data" / "cache" / "r15c1"
CACHE.mkdir(parents=True, exist_ok=True)

NBOOT = int(os.environ.get("BDB_NBOOT", 2000))
NPERM = int(os.environ.get("BDB_NPERM", 5000))
MIN_SNAPS, RATE_MIN, DAY3, UDFA_PICK = 100, 30, 103, 262
DT, W1, T_DEP = 0.1, 10, 15
ON, PRE = 1.0, 0.5
YD_PER_G = 9.80665 / 0.9144
Z90 = 1.6449
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
FIGW = 1600 / 150

NUM: dict = {}
MD: dict = {}


def put(key, v, nd=2):
    if key in NUM:
        raise KeyError(f"duplicate key {key}")
    if isinstance(v, (int, np.integer)):
        NUM[key] = int(v)
        return int(v)
    s = f"{float(v):.{nd}f}"
    if s in ("-0.00", "-0.0", "-0.000", "-0"):
        s = s.lstrip("-")
    NUM[key] = s
    return s


def save(fig, name):
    fig.set_size_inches(FIGW, fig.get_size_inches()[1])
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def cached(name, fn):
    f = CACHE / f"{name}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    out = fn()
    pickle.dump(out, open(f, "wb"))
    return out


def yz(s, year):
    return s.groupby(year).transform(lambda v: (v - v.mean()) / v.std())


def icc(ids, vals):
    d = pd.DataFrame({"i": ids, "v": vals}).dropna()
    d = d[d.groupby("i").v.transform("size") >= 2]
    g = d.groupby("i").v
    n, N = g.ngroups, len(d)
    kk = g.size()
    k0 = (N - (kk ** 2).sum() / N) / (n - 1)
    msb = (kk * (g.mean() - d.v.mean()) ** 2).sum() / (n - 1)
    msw = ((d.v - g.transform("mean")) ** 2).sum() / (N - n)
    return float((msb - msw) / (msb + (k0 - 1) * msw)), int(n)


def sb(r1, k):
    return k * r1 / (1 + (k - 1) * r1)


def boot_sp(x, y, seed, strata=None):
    """Spearman rho and 90% player-bootstrap interval (stratified by class when given)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    st = None if strata is None else np.asarray(strata)[m]
    r = stats.spearmanr(x, y)[0]
    rng = np.random.default_rng(seed)
    if st is None:
        groups = [np.arange(len(x))]
    else:
        groups = [np.flatnonzero(st == c) for c in np.unique(st)]
    b = []
    for _ in range(NBOOT):
        i = np.concatenate([rng.choice(g, len(g)) for g in groups])
        b.append(stats.spearmanr(x[i], y[i])[0])
    lo, hi = np.nanpercentile(b, [5, 95])
    return float(r), float(lo), float(hi), int(len(x)), np.array(b)


def boot_psp(df, x, y, Z, seed):
    """Partial Spearman (ranks residualized on Z) with a class-stratified 90% player bootstrap."""
    d = df[[x, y] + Z + ["draft_year"]].dropna()
    R = d[[x, y] + Z].rank().values
    cl = d.draft_year.values

    def f(ix):
        A = np.c_[np.ones(len(ix)), R[ix, 2:]]
        rx = R[ix, 0] - A @ np.linalg.lstsq(A, R[ix, 0], rcond=None)[0]
        ry = R[ix, 1] - A @ np.linalg.lstsq(A, R[ix, 1], rcond=None)[0]
        return float(np.corrcoef(rx, ry)[0, 1])
    r = f(np.arange(len(d)))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cl == c) for c in np.unique(cl)]
    b = [f(np.concatenate([rng.choice(g_, len(g_)) for g_ in groups])) for _ in range(NBOOT)]
    lo, hi = np.nanpercentile(b, [5, 95])
    return r, float(lo), float(hi), int(len(d)), np.array(b)


def wilson(k, n, z=Z90):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k0, n0):
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    d = p1 - p0
    return d, d - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)


# ============================================================================= load
players = pd.read_parquet(PROC / "players.parquet")
cres = pd.read_parquet(PROC / "combine_results.parquet")
events = pd.read_parquet(PROC / "combine_events.parquet")
seasons = pd.read_parquet(PROC / "player_season_stats.parquet")
align = pd.read_parquet(PROC / "player_season_alignment.parquet")
gp = pd.read_parquet(PROC / "game_plays_sample.parquet")

dl = players[players.position_group == "DL"][["nfl_id", "display_name", "draft_year", "nfl_position",
                                               "draft_overall_pick", "latest_team"]].copy()
dl["pick"] = dl.draft_overall_pick.fillna(UDFA_PICK)
dl["log_pick"] = np.log(dl.pick)
dl = dl.merge(cres.drop(columns=["draft_year", "combine_position"]), on="nfl_id", how="left")
for c in ["combine_weight", "ten_yd_split", "forty", "three_cone", "short_shuttle"]:
    dl[c + "_z"] = yz(dl[c], dl.draft_year)
a = align.merge(dl[["nfl_id"]], on="nfl_id")
a["edge_pl"] = np.where(a.lined_up_position.isin(["EDGE", "OLB"]), a.plays, 0)
es = a.groupby("nfl_id").agg(edge_pl=("edge_pl", "sum"), pl=("plays", "sum"))
dl = dl.merge((es.edge_pl / es.pl).rename("edge_share").reset_index(), on="nfl_id", how="left")
dl["edge_share"] = dl.edge_share.fillna(dl.nfl_position.isin(["DE", "OLB"]).astype(float))
dl["role"] = np.where(dl.edge_share >= 0.5, "EDGE", "interior")
put("n_dl", len(dl))
put("n_three_cone", int(dl.three_cone.notna().sum()))
classes = [2023, 2024, 2025]
# why the yardstick uses the official 10-yd split: the tracked 40 clip starts before the laser
_f40 = events[(events.position_group == "DL") & (events.drill_name == "FORTY_YARD_DASH")].groupby("nfl_id").split_10yd_s.min()
_f40 = dl.set_index("nfl_id").ten_yd_split.to_frame().join(_f40, how="inner").dropna()
put("split_r", np.corrcoef(_f40.ten_yd_split, _f40.split_10yd_s)[0, 1]); put("split_n", len(_f40))
put("split_off", (_f40.split_10yd_s - _f40.ten_yd_split).median(), 2)

# ============================================================================= Combine drill kinematics
ev = events[(events.position_group == "DL") & events.drill_name.isin(["PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL"])]
ev = ev[["event_id", "nfl_id", "draft_year", "drill_name", "attempt_no", "peak_accel", "mean_speed", "duration_s"]]


def onset(s, start=0):
    s = np.asarray(s, float)
    hit = np.flatnonzero(s[start:] >= ON)
    if not len(hit):
        return None
    i = j = start + int(hit[0])
    while j > start and s[j - 1] >= PRE and i - j < 3:
        j -= 1
    return j


def attempt_features(g):
    x, y, s = g.x.to_numpy(), g.y.to_numpy(), g.s.to_numpy()
    n = len(x)
    if n < 15:
        return None
    xs, ys = savgol_filter(x, 7, 2), savgol_filter(y, 7, 2)
    vx, vy = savgol_filter(xs, 7, 2, deriv=1, delta=DT), savgol_filter(ys, 7, 2, deriv=1, delta=DT)
    ax, ay = savgol_filter(xs, 7, 2, deriv=2, delta=DT), savgol_filter(ys, 7, 2, deriv=2, delta=DT)
    v = np.hypot(vx, vy)
    alat = np.abs(vx * ay - vy * ax) / np.maximum(v, 0.5)          # sideways (centripetal) acceleration, yd/s^2
    roll = pd.Series(alat).rolling(5, center=True).mean().to_numpy()  # 0.5 s window
    k = int(np.nanargmax(roll))
    o = onset(s)
    d1 = np.nan
    if o is not None and s[0] < ON and o + W1 < n:
        d1 = float(np.hypot(x[o + W1] - x[o], y[o + W1] - y[o]))     # yards in the first second of movement
    disp = np.hypot(x - x[0], y - y[0])
    t3 = np.argmax(disp >= 3) * DT if disp.max() >= 3 else np.nan
    return {"bend_g": np.nanmax(roll) / YD_PER_G, "d1": d1, "t3": t3, "speed_at_bend": v[k],
            "_alat": alat / YD_PER_G, "_x": xs, "_y": ys, "_k": k}


def drill_features():
    fr = pd.read_parquet(PROC / "combine_frames.parquet", filters=[("event_id", "in", list(ev.event_id))],
                         columns=["event_id", "frame", "entity_type", "x", "y", "s"])
    fr = fr[fr.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
    rows, traces = [], {}
    for eid, g in fr.groupby("event_id", sort=False):
        f = attempt_features(g)
        if f is None:
            continue
        traces[eid] = {k: f.pop(k) for k in ["_alat", "_x", "_y", "_k"]}
        rows.append({"event_id": eid, **f})
    return rows, traces


rows, traces = cached("drill", drill_features)
att = pd.DataFrame(rows).merge(ev, on="event_id")
hoop_att = att[att.drill_name == "RUN_THE_HOOP_DRILL"].copy()
rush_att = att[att.drill_name == "PASS_RUSH_DRILL"].copy()
put("hoop_g_median", hoop_att.bend_g.median())
put("n_hoop_runs", len(hoop_att))

# reliabilities (run to run, among linemen with 2+ attempts; attempts z-scored within Combine year)
hoop_att["z"] = yz(hoop_att.bend_g, hoop_att.draft_year)
rush_att["z"] = yz(rush_att.d1, rush_att.draft_year)
rel_hoop1, n_hrep = icc(hoop_att.nfl_id, hoop_att.z)
rel_d1_1, n_drep = icc(rush_att.nfl_id, rush_att.z)
k_rush = rush_att.dropna(subset=["d1"]).groupby("nfl_id").size().mean()
k_hoop = hoop_att.groupby("nfl_id").size().mean()
rel_d1 = sb(rel_d1_1, k_rush)
rel_hoop = sb(rel_hoop1, k_hoop)
put("rel_hoop1", rel_hoop1); put("rel_hoop2", sb(rel_hoop1, 2)); put("rel_hoop_nrep", n_hrep)
put("rel_d1_1", rel_d1_1); put("rel_d1", rel_d1); put("rel_d1_nrep", n_drep)

hoop = hoop_att.groupby("nfl_id").agg(hoop_g=("bend_g", "mean"), hoop_k=("bend_g", "size"),
                                      hoop_speed=("mean_speed", "mean"), hoop_time=("duration_s", "mean"),
                                      hoop_eid=("event_id", "first"), draft_year=("draft_year", "first")).reset_index()
rush = rush_att.groupby("nfl_id").agg(d1=("d1", "mean"), rush_bend=("bend_g", "mean"), rush_acc=("peak_accel", "mean"),
                                      rush_t3=("t3", "mean"), draft_year=("draft_year", "first")).reset_index()
for c in ["hoop_g", "hoop_speed", "hoop_time"]:
    hoop[c + "_z"] = yz(hoop[c], hoop.draft_year)
for c in ["d1", "rush_bend", "rush_acc", "rush_t3"]:
    rush[c + "_z"] = yz(rush[c], rush.draft_year)
hoop["hoop_time_z"] *= -1
rush["rush_t3_z"] *= -1
hoop = hoop.rename(columns={"hoop_g_z": "hoop_z"})
dl = dl.merge(hoop.drop(columns="draft_year"), on="nfl_id", how="left")
dl = dl.merge(rush.drop(columns="draft_year"), on="nfl_id", how="left")
SCREEN = ["hoop_z", "hoop_speed_z", "hoop_time_z", "rush_bend_z", "rush_acc_z", "rush_t3_z", "d1_z"]
put("n_hoop", int(dl.hoop_z.notna().sum()))
put("n_hoop_no3c", int((dl.hoop_z.notna() & dl.three_cone.isna()).sum()))
put("hoop_sd_g", hoop.hoop_g.groupby(hoop.draft_year).std().mean())
put("pct_one_run", 100 * (dl.hoop_k == 1).sum() / dl.hoop_z.notna().sum(), 0)

# ============================================================================= NFL outcomes (season aggregates)
s = seasons[seasons.position_group == "DL"].copy()
s["w_go"] = s.pass_rush_snaps * s.avg_get_off
agg = s.groupby("nfl_id").agg(rush_snaps=("pass_rush_snaps", "sum"), pressures=("pressures", "sum"),
                              w_go=("w_go", "sum")).reset_index()
agg["get_off"] = agg.w_go / agg.rush_snaps
agg["press_rate"] = agg.pressures / agg.rush_snaps
dl = dl.merge(agg, on="nfl_id", how="left")
dl["rush_snaps"] = dl.rush_snaps.fillna(0)
m = dl.rush_snaps >= 20
n_, k_ = dl.loc[m, "rush_snaps"], dl.loc[m, "pressures"]
mu = k_.sum() / n_.sum()
N_ = n_.sum()
tau2 = ((n_ * (k_ / n_ - mu) ** 2).sum() - (len(n_) - 1) * mu * (1 - mu)) / (N_ - (n_ ** 2).sum() / N_)
Mw = mu * (1 - mu) / tau2 - 1
mm = dl.rush_snaps > 0
dl.loc[mm, "press_eb"] = (dl.loc[mm, "pressures"] + Mw * mu) / (dl.loc[mm, "rush_snaps"] + Mw)
dl["quick"] = -dl.get_off

# ============================================================================= sampled NFL tracking
dlids = set(dl.nfl_id)
K = ["game_id", "play_id", "nfl_id"]
g = gp[gp.nfl_id.isin(dlids)].drop_duplicates(K)
pinfo = g.set_index(K)[["play_direction", "sample_kind", "time_to_pressure", "sack", "lined_up_position",
                        "expected_points_added"]]


def game_features():
    gf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "x", "y", "event"])
    gf = gf[gf.nfl_id.isin(dlids)].merge(g[K + ["snap_frame"]], on=K)
    ends = gf[gf.event.isin(["pass_forward", "qb_sack", "qb_strip_sack", "run", "handoff"])].groupby(K).frame.min()
    gf = gf.join(ends.rename("end_frame"), on=K).sort_values(K + ["frame"])
    out = []
    for key, h in gf.groupby(K, sort=False):
        info = pinfo.loc[key]
        sf, e = h.snap_frame.iloc[0], h.end_frame.iloc[0]
        fr, x_, y_ = h.frame.values, h.x.values, h.y.values
        if not (fr == sf).any():
            continue
        i0 = np.flatnonzero(fr == sf)[0]
        i1 = np.flatnonzero(fr == sf + W1)
        d1 = float(np.hypot(x_[i1[0]] - x_[i0], y_[i1[0]] - y_[i0])) if len(i1) else np.nan
        sg = 1.0 if info.play_direction == "left" else -1.0
        m_ = np.flatnonzero(fr == sf + T_DEP)
        ok = len(m_) and (pd.isna(e) or e >= sf + T_DEP)
        dep = sg * (x_[m_[0]] - x_[i0]) if ok else np.nan
        lat = abs(y_[m_[0]] - y_[i0]) if ok else np.nan
        out.append(dict(zip(K, key), g_d1=d1, dep=dep, lat=lat, kind=info.sample_kind,
                        press=int(pd.notna(info.time_to_pressure) or info.sack == 1)))
    return pd.DataFrame(out)


GF = cached("game", game_features)
GR = GF[GF.kind == "random"].copy()
GR["half"] = GR.groupby("nfl_id").cumcount() % 2
GR["loop"] = (GR.lat >= 2.0).astype(float).where(GR.lat.notna())
put("n_random_rushes", len(GR))


def split_half(df, col, min_n):
    cnt = df.groupby("nfl_id")[col].count()
    keep = cnt[cnt >= min_n].index
    a_ = df[df.nfl_id.isin(keep)].groupby(["nfl_id", "half"])[col].mean().unstack().dropna()
    r_ = stats.spearmanr(a_[0], a_[1])[0]
    return 2 * r_ / (1 + r_), len(a_)


rel_gd1, n_rg = split_half(GR, "g_d1", 8)
rel_dep, n_rdep = split_half(GR, "dep", 10)
rel_prs, _ = split_half(GR, "press", 10)
put("rel_game_d1", rel_gd1); put("rel_game_d1_n", n_rg)
put("rel_dep", rel_dep); put("rel_dep_n", n_rdep); put("rel_prs", rel_prs)
PG = GR.groupby("nfl_id").agg(g_d1=("g_d1", "mean"), g_d1_n=("g_d1", "count"), dep=("dep", "mean"),
                              dep_n=("dep", "count"), loop=("loop", "mean"), loop_n=("loop", "count")).reset_index()
dl = dl.merge(PG, on="nfl_id", how="left")

# motivation: what a pressured snap costs (random sampled NFL pass plays of cohort DL)
_gm = g[(g.sample_kind == "random") & g.expected_points_added.notna()].drop_duplicates(["game_id", "play_id"])
_pr = _gm.time_to_pressure.notna() | (_gm.sack == 1)
e1, e0 = _gm[_pr].expected_points_added.values, _gm[~_pr].expected_points_added.values
_r = np.random.default_rng(101)
_eb = [_r.choice(e0, len(e0)).mean() - _r.choice(e1, len(e1)).mean() for _ in range(NBOOT)]
EPA = e0.mean() - e1.mean()
put("epa_press", EPA); put("epa_press_lo", np.percentile(_eb, 5)); put("epa_press_hi", np.percentile(_eb, 95))
put("epa_n", len(_gm))

# ============================================================================= no-tracking forecast
co = dl[dl.rush_snaps >= MIN_SNAPS].copy()
MAIN = ["log_pick", "combine_weight_z", "edge_share", "ngs_athleticism_score"]
TIMES = ["ten_yd_split_z", "three_cone_z"]
BASE_T = MAIN + TIMES
FILLX = ["combine_weight_z", "edge_share", "ngs_athleticism_score"]
put("n_co", len(co))


def fill_times(tr, te):
    tr, te = tr.copy(), te.copy()
    for c in TIMES:
        mk = tr[c].notna()
        A = np.c_[np.ones(mk.sum()), tr.loc[mk, FILLX].values]
        b = np.linalg.lstsq(A, tr.loc[mk, c].values, rcond=None)[0]
        for d in (tr, te):
            m2 = d[c].isna()
            d.loc[m2, c] = np.c_[np.ones(m2.sum()), d.loc[m2, FILLX].values] @ b
    return tr, te


def ols_pred(tr, te, cols, y):
    A = np.c_[np.ones(len(tr)), tr[cols].values]
    b = np.linalg.lstsq(A, tr[y].values, rcond=None)[0]
    return np.c_[np.ones(len(te)), te[cols].values] @ b


def ols_fc(tr, te, cols=BASE_T):
    tr, te = fill_times(tr, te)
    return ols_pred(tr.dropna(subset=cols + ["press_eb"]), te, cols, "press_eb")


LAMS = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def ridge_fc(tr, te, lam):
    tr, te = fill_times(tr, te)
    mu_, sd_ = tr[BASE_T].mean(), tr[BASE_T].std()
    Xt, Xe = ((tr[BASE_T] - mu_) / sd_).values, ((te[BASE_T] - mu_) / sd_).values
    w_ = tr.rush_snaps.values / tr.rush_snaps.mean()
    y_ = tr.press_eb.values
    ym, xm = np.average(y_, weights=w_), np.average(Xt, axis=0, weights=w_)
    Xc = Xt - xm
    b_ = np.linalg.solve(Xc.T @ (w_[:, None] * Xc) + lam * np.eye(Xc.shape[1]), Xc.T @ (w_ * (y_ - ym)))
    return ym + (Xe - xm) @ b_


def ridge_choose(trc):
    sse = []
    for lam in LAMS:
        e_ = 0.0
        for c2 in trc.draft_year.unique():
            a_, b_ = trc[trc.draft_year != c2], trc[trc.draft_year == c2]
            e_ += float((b_.rush_snaps * (b_.press_eb - ridge_fc(a_, b_, lam)) ** 2).sum())
        sse.append(e_)
    return LAMS[int(np.argmin(sse))]


ba = dl[dl.hoop_z.notna()].copy()
sc = ba.rush_snaps >= RATE_MIN
for col in ["pb", "pb_ridge", "pb_pick", "pb_tf"]:
    ba[col] = np.nan
for c in classes:
    tr_ = co[co.draft_year != c]
    te_ = ba[(ba.draft_year == c) & sc]
    ba.loc[te_.index, "pb"] = ols_fc(tr_, te_)
    ba.loc[te_.index, "pb_ridge"] = ridge_fc(tr_, te_, ridge_choose(tr_))
    ba.loc[te_.index, "pb_pick"] = ols_pred(tr_, te_, ["log_pick", "edge_share"], "press_eb")
    if c > classes[0]:
        tf_ = co[co.draft_year < c]
        ba.loc[te_.index, "pb_tf"] = ols_fc(tf_, te_)
put("n_scored", int(ba.pb.notna().sum()))
# held-out calibration of the forecast itself (rank agreement with NFL pressure rate, 100+ rush scored players)
_cal = ba[ba.pb.notna() & (ba.rush_snaps >= MIN_SNAPS)]
put("fc_rho", stats.spearmanr(_cal.pb, _cal.press_rate)[0]); put("fc_rho_n", len(_cal))

# ============================================================================= the shortlist (primary test)
ba["third"] = ba.groupby("draft_year").hoop_z.transform(
    lambda s_: pd.qcut(s_.rank(method="first"), 3, labels=False)).astype(int)
ba["day3"] = ba.pick >= DAY3
ba["short"] = ba.day3 & (ba.third == 2)
ba["exc450"] = 450 * (ba.pressures - ba.pb * ba.rush_snaps) / ba.rush_snaps
D3 = ba[ba.day3].copy()
put("n_day3_runners", len(D3)); put("n_short_all", int(D3.short.sum()))
put("pct_day3_hoop", 100 * ba.day3.mean(), 0)


def tally(d_, yard):
    d_ = d_[d_[yard].notna()]
    beat = d_.press_rate > d_[yard]
    a_, b_ = beat[d_.short], beat[~d_.short]
    return int(a_.sum()), len(a_), int(b_.sum()), len(b_)


ROB = {}
for tag, yard, sub in [("main", "pb", D3), ("ridge", "pb_ridge", D3), ("tf", "pb_tf", D3),
                       ("pick", "pb_pick", D3)]:
    k1, n1, k0, n0 = tally(sub, yard)
    d, lo, hi = newcombe(k1, n1, k0, n0)
    ROB[tag] = (k1, n1, k0, n0, d, lo, hi)
    put(f"{tag}_k1", k1); put(f"{tag}_n1", n1); put(f"{tag}_k0", k0); put(f"{tag}_n0", n0)
    put(f"{tag}_p1", 100 * k1 / n1, 0); put(f"{tag}_p0", 100 * k0 / n0, 0)
    put(f"{tag}_gap", 100 * d, 0); put(f"{tag}_lo", 100 * lo, 0); put(f"{tag}_hi", 100 * hi, 0)
SKD = D3[D3.short & D3.pb.notna()].sort_values("exc450", ascending=False)
BEST = SKD.display_name.iloc[0]
_d3x = D3[D3.display_name != BEST]
k1, n1, k0, n0 = tally(_d3x, "pb")
d, lo, hi = newcombe(k1, n1, k0, n0)
ROB["drop"] = (k1, n1, k0, n0, d, lo, hi)
put("drop_k1", k1); put("drop_n1", n1); put("drop_k0", k0); put("drop_n0", n0)
put("drop_gap", 100 * d, 0); put("drop_lo", 100 * lo, 0); put("drop_hi", 100 * hi, 0)
NUM["best_name"] = BEST
# day 1-2 runners in the top third, for context (the rule is a day-3 rule)
_e = ba[~ba.day3 & ba.pb.notna()]
put("d12_top_k", int((_e[_e.third == 2].press_rate > _e[_e.third == 2].pb).sum())); put("d12_top_n", int((_e.third == 2).sum()))
put("d12_rest_k", int((_e[_e.third != 2].press_rate > _e[_e.third != 2].pb).sum())); put("d12_rest_n", int((_e.third != 2).sum()))

# Bayesian: independent Beta(1,1) priors on the two hit rates
k1, n1, k0, n0 = ROB["main"][:4]
_rb = np.random.default_rng(7)
post = _rb.beta(k1 + 1, n1 - k1 + 1, 200000) - _rb.beta(k0 + 1, n0 - k0 + 1, 200000)
put("bayes_pgt0", 100 * (post > 0).mean(), 0)
put("bayes_lo", 100 * np.percentile(post, 5), 0); put("bayes_hi", 100 * np.percentile(post, 95), 0)
put("bayes_p10", 100 * (post > 0.10).mean(), 0)

# random relabelling: Hoop G shuffled within class among all hoop runners, thirds and shortlist rebuilt
_rp = np.random.default_rng(1501)
obs_gap = k1 / n1 - k0 / n0
_bix = [np.flatnonzero(ba.draft_year.values == c) for c in classes]
_d3 = ba.day3.values
_scd = ba.pb.notna().values
_bt = (ba.press_rate > ba.pb).values
nul = []
for _ in range(NPERM):
    th_ = np.zeros(len(ba), int)
    for ix in _bix:
        z_ = ba.hoop_z.values[_rp.permutation(ix)]
        th_[ix] = pd.qcut(pd.Series(z_).rank(method="first"), 3, labels=False).values
    s_ = _d3 & (th_ == 2)
    a_, b_ = s_ & _scd, _d3 & ~s_ & _scd
    if a_.sum() == 0 or b_.sum() == 0:
        continue
    nul.append(_bt[a_].mean() - _bt[b_].mean())
nul = np.array(nul)
put("perm_p", (1 + (nul >= obs_gap - 1e-12).sum()) / (1 + len(nul)), 3)

# pressures per 450 rushes over forecast, shortlist and rest of day 3 (player bootstrap)
_sk, _rk = D3[D3.short & D3.pb.notna()], D3[~D3.short & D3.pb.notna()]
ex = lambda q: 450 * (q.pressures.sum() - (q.pb * q.rush_snaps).sum()) / q.rush_snaps.sum()
_rx = np.random.default_rng(31)
_bd = [ex(_sk.iloc[_rx.integers(0, len(_sk), len(_sk))]) - ex(_rk.iloc[_rx.integers(0, len(_rk), len(_rk))])
       for _ in range(NBOOT)]
put("ex_sl", ex(_sk), 1); put("ex_rest", ex(_rk), 1)
put("ex_gap", ex(_sk) - ex(_rk), 1); put("ex_gap_lo", np.percentile(_bd, 5), 1); put("ex_gap_hi", np.percentile(_bd, 95), 1)
put("ex_gap_epa", (ex(_sk) - ex(_rk)) * EPA, 1)

# per class
PERCLS = {}
for c in classes:
    a_, b_ = _sk[_sk.draft_year == c], _rk[_rk.draft_year == c]
    PERCLS[c] = (int((a_.press_rate > a_.pb).sum()), len(a_), int((b_.press_rate > b_.pb).sum()), len(b_))
    put(f"c{c}_k1", PERCLS[c][0]); put(f"c{c}_n1", PERCLS[c][1]); put(f"c{c}_k0", PERCLS[c][2]); put(f"c{c}_n0", PERCLS[c][3])
put("cls_ahead", int(sum(v[0] / max(v[1], 1) > v[2] / max(v[3], 1) for v in PERCLS.values())))

# survival: did shortlisted day-3 rushers reach 30 NFL rushes at the same rate?
put("surv_k1", int((D3.short & (D3.rush_snaps >= RATE_MIN)).sum())); put("surv_n1", int(D3.short.sum()))
put("surv_k0", int((~D3.short & (D3.rush_snaps >= RATE_MIN)).sum())); put("surv_n0", int((~D3.short).sum()))

# same rule on the official 3-cone (day-3 rushers who ran it)
_t3 = ba[ba.three_cone.notna()].copy()
_t3["t3_third"] = _t3.groupby("draft_year").three_cone_z.transform(
    lambda s_: pd.qcut((-s_).rank(method="first"), 3, labels=False))
_t3d = _t3[_t3.day3 & _t3.pb.notna()]
_a3, _b3 = _t3d[_t3d.t3_third == 2], _t3d[_t3d.t3_third != 2]
put("t3_k1", int((_a3.press_rate > _a3.pb).sum())); put("t3_n1", len(_a3))
put("t3_k0", int((_b3.press_rate > _b3.pb).sum())); put("t3_n0", len(_b3))

# ============================================================================= round 15 additions
# (a) Survivorship: who reaches 30 NFL rushes? Logistic selection model on all day-3 hoop runners, then inverse-
#     probability-weighted hit rates (the scored rushers stand in for the unscored ones like them). Bootstrap refits
#     the selection model inside each resample.
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import StratifiedTable

SELX = ["log_pick", "edge_share", "combine_weight_z", "hoop_z"]


def sel_fit(d_):
    X = d_[SELX].copy()
    for c in classes[1:]:
        X[f"y{c}"] = (d_.draft_year == c).astype(float)
    X = sm.add_constant(X.values.astype(float), has_constant="add")
    y = (d_.rush_snaps >= RATE_MIN).astype(float).values
    return sm.GLM(y, X, family=sm.families.Binomial()).fit(), X


def ipw_gap(d_):
    f_, X = sel_fit(d_)
    p_ = np.clip(f_.predict(X), 0.05, 1)
    ok = d_.pb.notna().values
    w = 1 / p_[ok]
    hit = (d_.press_rate > d_.pb).values[ok]
    sh = d_.short.values[ok]
    return np.average(hit[sh], weights=w[sh]) - np.average(hit[~sh], weights=w[~sh]), f_


D3s = D3.copy()
D3s["combine_weight_z"] = D3s.combine_weight_z.fillna(0)
ipw0, selm = ipw_gap(D3s)
put("ipw_gap", 100 * ipw0, 0)
_ri2 = np.random.default_rng(2015)
_d3g = [np.flatnonzero(D3s.draft_year.values == c) for c in classes]
_ib = []
for _ in range(NBOOT):
    try:
        _ib.append(ipw_gap(D3s.iloc[np.concatenate([_ri2.choice(ix, len(ix)) for ix in _d3g])])[0])
    except Exception:
        pass
put("ipw_lo", 100 * np.nanpercentile(_ib, 5), 0); put("ipw_hi", 100 * np.nanpercentile(_ib, 95), 0)
_ci = selm.conf_int(alpha=0.10)
put("sel_or_hoop", np.exp(selm.params[4])); put("sel_or_lo", np.exp(_ci[4, 0])); put("sel_or_hi", np.exp(_ci[4, 1]))
put("sel_or_pick", np.exp(selm.params[1]))
put("n_unscored", int(D3.pb.isna().sum())); put("n_unscored_sl", int((D3.pb.isna() & D3.short).sum()))

# (b) Class-stratified test: Mantel-Haenszel common odds ratio over the three 2x2 class tables (each class is a
#     stratum, so a class where the forecast runs hot or cold cannot drive the gap).
_tabs = [np.array([[v[0], v[1] - v[0]], [v[2], v[3] - v[2]]], float) + 0.5 for v in
         [(int((_a.press_rate > _a.pb).sum()), len(_a), int((_b.press_rate > _b.pb).sum()), len(_b))
          for _a, _b in [(D3[D3.short & D3.pb.notna() & (D3.draft_year == c)],
                          D3[~D3.short & D3.pb.notna() & (D3.draft_year == c)]) for c in classes]]]
_st = StratifiedTable(_tabs)
put("mh_or", _st.oddsratio_pooled, 1)
_mci = _st.oddsratio_pooled_confint(alpha=0.10)
put("mh_lo", _mci[0], 1); put("mh_hi", _mci[1], 1)
put("mh_p", _st.test_null_odds(correction=True).pvalue, 2)
put("mh_homog_p", _st.test_equal_odds().pvalue, 2)

# (c) Calibration: rush-weighted actual vs forecast pressure rate for the shortlist and the rest of day 3
for tag, q in [("sl", D3[D3.short & D3.pb.notna()]), ("re", D3[~D3.short & D3.pb.notna()])]:
    put(f"cal_{tag}_act", 100 * q.pressures.sum() / q.rush_snaps.sum(), 1)
    put(f"cal_{tag}_fc", 100 * (q.pb * q.rush_snaps).sum() / q.rush_snaps.sum(), 1)

# (d) Matched boards: each scored shortlisted rusher vs the nearest-pick unshortlisted rusher of the same class and
#     role (post hoc, reported as context only)
_pairs = []
for _, r_ in D3[D3.short & D3.pb.notna()].iterrows():
    pool = D3[~D3.short & D3.pb.notna() & (D3.draft_year == r_.draft_year) & (D3.role == r_.role)]
    if not len(pool):
        continue
    m_ = pool.iloc[(pool.pick - r_.pick).abs().argsort().values[0]]
    _pairs.append((r_.display_name, m_.display_name, r_.exc450 > m_.exc450))
put("pair_n", len(_pairs)); put("pair_k", int(sum(p[2] for p in _pairs)))
car = pd.read_parquet(PROC / "player_career.parquet")[["nfl_id", "contract_apy_m"]]
ba = ba.merge(car, on="nfl_id", how="left").set_index(ba.index)
for key, nm in [("wcj", "Cedric Johnson"), ("wjj", "Javontae Jean-Baptiste"), ("wkk", "Kyle Kennard"),
                ("wbs", "Barryn Sorrell")]:
    r_ = ba[ba.display_name == nm].iloc[0]
    put(f"{key}_pick", int(r_.pick)); put(f"{key}_hz", r_.hoop_z, 1); put(f"{key}_wt", int(r_.combine_weight))
    put(f"{key}_pr", 100 * r_.press_rate, 1); put(f"{key}_pb", 100 * r_.pb, 1); put(f"{key}_n", int(r_.rush_snaps))
    put(f"{key}_apy", r_.contract_apy_m, 2)

# (e) Drill burst -> pressure as a bound: rush-weighted slope (points per sd) with 90% interval; its upper end is the
#     largest effect a two one-sided test at 5% cannot rule out
SCb = ba[ba.pb.notna() & ba.d1_z.notna()].copy()
SCb["res"] = 100 * (SCb.press_rate - SCb.pb)


def wslope_x(d_, col):
    X = d_[col].values[:, None]
    for c in classes[1:]:
        X = np.c_[X, (d_.draft_year.values == c).astype(float)]
    X = np.c_[np.ones(len(d_)), X]
    w = np.sqrt(d_.rush_snaps.values)
    return np.linalg.lstsq(X * w[:, None], d_.res.values * w, rcond=None)[0][1]


_rb2 = np.random.default_rng(1502)
_g2 = [np.flatnonzero(SCb.draft_year.values == c) for c in classes]
_bb2 = [wslope_x(SCb.iloc[np.concatenate([_rb2.choice(ix, len(ix)) for ix in _g2])], "d1_z") for _ in range(NBOOT)]
put("bslope", wslope_x(SCb, "d1_z")); put("bslope_lo", np.percentile(_bb2, 5)); put("bslope_hi", np.percentile(_bb2, 95))
put("bslope_n", len(SCb))

# (f) Combine protocol: reps needed for 0.5 reliability (Spearman-Brown)
put("burst_reps50", (1 - rel_d1_1) / rel_d1_1, 0)
put("rel_hoop3", sb(rel_hoop1, 3))

# ============================================================================= secondary: continuous slope
SC = ba[ba.pb.notna()].copy()
SC["res"] = 100 * (SC.press_rate - SC.pb)


def wslope(d_):
    X = d_.hoop_z.values[:, None]
    for c in classes[1:]:
        X = np.c_[X, (d_.draft_year.values == c).astype(float)]
    X = np.c_[np.ones(len(d_)), X]
    w = np.sqrt(d_.rush_snaps.values)
    return np.linalg.lstsq(X * w[:, None], d_.res.values * w, rcond=None)[0][1]


b0 = wslope(SC)
_rs = np.random.default_rng(1401)
_gix = [np.flatnonzero(SC.draft_year.values == c) for c in classes]
_bb = np.array([wslope(SC.iloc[np.concatenate([_rs.choice(ix, len(ix)) for ix in _gix])]) for _ in range(NBOOT)])
put("slope", b0); put("slope_lo", np.percentile(_bb, 5)); put("slope_hi", np.percentile(_bb, 95))
put("slope_pos", 100 * (_bb > 0).mean(), 0); put("slope_n", len(SC))
_pc = []
for c in classes:
    d_ = SC[SC.draft_year == c]
    X = np.c_[np.ones(len(d_)), d_.hoop_z.values]
    w = np.sqrt(d_.rush_snaps.values)
    _pc.append(np.linalg.lstsq(X * w[:, None], d_.res.values * w, rcond=None)[0][1])
put("slope_cls_pos", int(sum(v > 0 for v in _pc)))
put("slope_450", 4.5 * b0, 1)

# search adjustment: Hoop G was one of 7 drill measures x 2 outcomes screened in earlier rounds. Statistic =
# partial Spearman (ranks residualized on pick, weight, EDGE share, NGS) in the 100+ rush cohort; null = drill
# block shuffled across linemen within class; p = share of shuffles whose best of 14 beats the observed Hoop G.
def screen(df, cols_perm=None):
    out = {}
    for oc in ["press_eb", "quick"]:
        for col in SCREEN:
            x_ = df[col] if cols_perm is None else cols_perm[col]
            d = pd.DataFrame({"x": x_.values, "y": df[oc].values, **{z: df[z].values for z in MAIN}}).dropna()
            R = d.rank()
            A = np.c_[np.ones(len(d)), R[MAIN].values]
            P = A @ np.linalg.pinv(A)
            rx, ry = R.x.values - P @ R.x.values, R.y.values - P @ R.y.values
            out[(oc, col)] = float(np.corrcoef(rx, ry)[0, 1])
    return out


coS = co.reset_index(drop=True)
obs_scr = screen(coS)
put("scr_hoop", obs_scr[("press_eb", "hoop_z")])
_ri = np.random.default_rng(55)
_cg = [np.flatnonzero(coS.draft_year.values == c) for c in classes]
mx = []
for _ in range(int(os.environ.get("BDB_NSCR", 1000))):
    perm = np.arange(len(coS))
    for ix in _cg:
        perm[ix] = _ri.permutation(ix)
    blk = coS[SCREEN].iloc[perm].reset_index(drop=True)
    mx.append(max(screen(coS, blk).values()))
put("scr_p", (1 + (np.array(mx) >= obs_scr[("press_eb", "hoop_z")] - 1e-12).sum()) / (1 + len(mx)), 2)
put("scr_rank", int(1 + sum(v > obs_scr[("press_eb", "hoop_z")] for v in obs_scr.values())))

# ============================================================================= drill-to-game chain
CH = {}
# (1) rush-drill burst -> first-second distance after the snap on random sampled NFL rushes (8+ such rushes)
d_ = dl[(dl.g_d1_n >= 8)]
RZ = ["combine_weight_z", "edge_share"]
r, lo, hi, n, bs1 = boot_psp(d_, "d1_z", "g_d1", RZ, 11)
den = np.sqrt(rel_d1 * rel_gd1)
CH["burst_game"] = (r, lo, hi, n, r / den, lo / den, hi / den)
put("ch_bg", r); put("ch_bg_lo", lo); put("ch_bg_hi", hi); put("ch_bg_n", n)
put("ch_bg_da", min(r / den, 1)); put("ch_bg_da_lo", lo / den); put("ch_bg_da_hi", min(hi / den, 1))
r_raw = boot_sp(d_.d1_z, d_.g_d1, 19)
put("ch_bg_raw", r_raw[0])
# game burst vs season get-off time (are the two NFL measures the same thing?), 100+ rushes and 8+ sampled rushes
_q = dl[(dl.g_d1_n >= 8) & (dl.rush_snaps >= MIN_SNAPS)]
r, lo, hi, n, _ = boot_psp(_q, "g_d1", "quick", RZ, 20)
put("ch_gq", r); put("ch_gq_lo", lo); put("ch_gq_hi", hi); put("ch_gq_n", n)
CH["gq"] = (r, lo, hi, n)
# (1b) same burst -> season get-off (quicker), 100+ rushes
r, lo, hi, n, _ = boot_psp(co, "d1_z", "quick", RZ, 12)
put("ch_bs", r); put("ch_bs_lo", lo); put("ch_bs_hi", hi); put("ch_bs_n", n)
# (2) Hoop G -> NFL first-second distance (should be ~0: bend is not burst)
r, lo, hi, n, _ = boot_psp(d_, "hoop_z", "g_d1", RZ, 13)
CH["hoop_game"] = (r, lo, hi, n)
put("ch_hg", r); put("ch_hg_lo", lo); put("ch_hg_hi", hi); put("ch_hg_n", n)
# (3) NFL get-off (season, quicker) -> pressure over the no-tracking forecast (30+ rushes)
SC = SC.merge(dl[["nfl_id"]], on="nfl_id")
r, lo, hi, n, _ = boot_sp(SC.quick, SC.res, 14, SC.draft_year)
CH["getoff_press"] = (r, lo, hi, n)
put("ch_gp", r); put("ch_gp_lo", lo); put("ch_gp_hi", hi); put("ch_gp_n", n)
# (4) rush-drill burst -> pressure over forecast
r, lo, hi, n, _ = boot_sp(SC.d1_z, SC.res, 15, SC.draft_year)
CH["burst_press"] = (r, lo, hi, n)
put("ch_bp", r); put("ch_bp_lo", lo); put("ch_bp_hi", hi); put("ch_bp_n", n)
# (5) Hoop G -> pressure over forecast
r, lo, hi, n, _ = boot_sp(SC.hoop_z, SC.res, 16, SC.draft_year)
CH["hoop_press"] = (r, lo, hi, n)
put("ch_hp", r); put("ch_hp_lo", lo); put("ch_hp_hi", hi); put("ch_hp_n", n)
# (6) depth upfield 1.5 s after the snap (random rushes; 10+ timed): forecast LOCO from pick, weight, role, NGS
DD = dl[(dl.dep_n >= 10)].dropna(subset=MAIN + ["dep"]).copy()
DD["dres"] = np.nan
for c in classes:
    tr_, te_ = DD[DD.draft_year != c], DD[DD.draft_year == c]
    DD.loc[te_.index, "dres"] = te_.dep.values - ols_pred(tr_, te_, MAIN, "dep")
r, lo, hi, n, _ = boot_sp(DD.hoop_z, DD.dres, 17, DD.draft_year)
CH["hoop_depth"] = (r, lo, hi, n)
put("ch_hd", r); put("ch_hd_lo", lo); put("ch_hd_hi", hi); put("ch_hd_n", n)
_dv = DD.merge(SC[["nfl_id", "res"]], on="nfl_id")
r, lo, hi, n, _ = boot_sp(_dv.dres, _dv.res, 18, _dv.draft_year)
CH["depth_press"] = (r, lo, hi, n)
put("ch_dp", r); put("ch_dp_lo", lo); put("ch_dp_hi", hi); put("ch_dp_n", n)

put("ch_prod", float(NUM["ch_bg"]) * float(NUM["ch_gp"]))
put("win_s", 0.5, 1); put("dep_s", 1.5, 1)

# ============================================================================= price: do clubs pay for Hoop G?
PR = dl[dl.hoop_z.notna()].copy()
PR["pos_edge"] = PR.nfl_position.isin(["DE", "OLB"]).astype(float)
PCOV = ["combine_weight_z", "ngs_athleticism_score", "ten_yd_split_z", "three_cone_z", "pos_edge"]
for c_ in ["ten_yd_split_z", "three_cone_z", "ngs_athleticism_score"]:
    PR[c_ + "_m"] = PR[c_].isna().astype(float)
    PR[c_] = PR[c_].fillna(PR.groupby("draft_year")[c_].transform("mean"))
PCOV2 = PCOV + ["ten_yd_split_z_m", "three_cone_z_m", "ngs_athleticism_score_m"]


def price_b(d_):
    X = np.c_[np.ones(len(d_)), d_[PCOV2].values, d_.hoop_z.values]
    for c in classes[1:]:
        X = np.c_[X, (d_.draft_year.values == c).astype(float)]
    return np.linalg.lstsq(X, d_.log_pick.values, rcond=None)[0][len(PCOV2) + 1]


_p0 = price_b(PR)
_rq = np.random.default_rng(77)
_pg = [np.flatnonzero(PR.draft_year.values == c) for c in classes]
_pbb = np.array([price_b(PR.iloc[np.concatenate([_rq.choice(ix, len(ix)) for ix in _pg])]) for _ in range(NBOOT)])
put("price_later", 100 * (np.exp(_p0) - 1), 0)
put("price_lo", 100 * (np.exp(np.percentile(_pbb, 5)) - 1), 0)
put("price_hi", 100 * (np.exp(np.percentile(_pbb, 95)) - 1), 0)
put("price_n", len(PR))
put("price_earlier", -int(NUM["price_lo"]))
PK3 = {t: float(np.median(ba[ba.third == t].pick)) for t in range(3)}
for t in range(3):
    put(f"pick_t{t}", int(PK3[t]))
put("short_med_pick", int(D3[D3.short].pick.median()))

# ============================================================================= cases and the 2025 card
CASES = ["Cedric Johnson", "Jaylen Harrell", "Isaiah McGuire", "Jordan Phillips"]
CROWS = {}
for nm in CASES:
    r_ = ba[ba.display_name == nm].iloc[0]
    key = nm.split()[-1].lower().replace("-", "")
    CROWS[nm] = r_
    put(f"{key}_hz", r_.hoop_z, 1); put(f"{key}_pick", int(r_.pick)); put(f"{key}_wt", int(r_.combine_weight))
    put(f"{key}_pr", 100 * r_.press_rate, 1); put(f"{key}_pb", 100 * r_.pb, 1); put(f"{key}_n", int(r_.rush_snaps))
    put(f"{key}_loop", 100 * r_.loop, 0); put(f"{key}_loopn", int(r_.loop_n))
    put(f"{key}_hoopg", r_.hoop_g, 2)
    put(f"{key}_d1", r_.d1_z, 1)
put("int_loop_med", 100 * dl[(dl.role == "interior") & (dl.loop_n >= 10)].loop.median(), 0)


def plan(r_):
    n = 0 if pd.isna(r_.rush_snaps) else int(r_.rush_snaps)
    scd = pd.notna(r_.pb) and n >= MIN_SNAPS
    beat = scd and r_.press_rate > r_.pb
    if n == 0:
        return "Practice-squad look; second hoop run in camp"
    if r_.role == "EDGE":
        if beat:
            return "More third-down snaps from a wide (7/9) alignment"
        if scd:
            return "Hold snaps; grade his wide-arc rushes"
        return "Third-down sub-rusher audition"
    if beat:
        return "Looper on interior stunts"
    if scd:
        return "Early downs; camp test as stunt looper"
    return "Stunt-package audition"


C25 = ba[ba.short & (ba.draft_year == 2025)].sort_values("hoop_z", ascending=False)
G25 = seasons[(seasons.season == 2025) & (seasons.position_group == "DL")].groupby("nfl_id").games.sum()
out = ["| 2025 shortlist (pick, team) | Role | Hoop G (sd) | 2025 games | Rookie pressure vs forecast (rushes; — = under 30) | Plan for 2026 |",
       "|---|---|---|---|---|---|"]
for i, (_, r_) in enumerate(C25.iterrows()):
    pk = "UDFA" if r_.pick >= UDFA_PICK else str(int(r_.pick))
    tm = r_.latest_team if isinstance(r_.latest_team, str) else "FA"
    hz = put(f"c25_{i}_hz", r_.hoop_z, 1)
    if pd.notna(r_.pb):
        res = f"{{{{c25_{i}_pr}}}}% vs {{{{c25_{i}_pb}}}}% ({int(r_.rush_snaps)})"
        put(f"c25_{i}_pr", 100 * r_.press_rate, 1); put(f"c25_{i}_pb", 100 * r_.pb, 1)
    else:
        res = f"— ({int(r_.rush_snaps)})"
    gm = int(G25.get(r_.nfl_id, 0))
    out.append(f"| **{r_.display_name}** ({pk}, {tm}) | {r_.role} | +{{{{c25_{i}_hz}}}} | {gm} | {res} | {plan(r_)} |")
MD["card25"] = "\n".join(out)
put("n_c25", len(C25))
put("c25_one_run", int((C25.hoop_k == 1).sum()))

# board CSV (all hoop runners)
ba[["draft_year", "display_name", "role", "pick", "combine_weight", "hoop_k", "hoop_g", "hoop_z", "third", "day3",
    "short", "rush_snaps", "pressures", "press_rate", "pb", "pb_ridge", "pb_tf", "pb_pick"]].sort_values(
    ["draft_year", "hoop_z"], ascending=[True, False]).round(4).to_csv(OUT / "board_2023_2025.csv", index=False)

# frozen rule for the 2026 class (unchanged from round 13)
SLRULE = {"shortlist": "pick >= 103 or undrafted AND Hoop G in the top third of all Run-the-Hoop runners in his Combine "
                       "year (Hoop G as in frozen_rule.json)",
          "hit": "NFL pressure rate > no-tracking forecast fit on earlier classes (frozen_rule.json forecast)",
          "min_nfl_rushes": RATE_MIN, "base_rate": "other day-3/undrafted hoop runners with 30+ rushes",
          "score_class": 2026, "score_after": "2027 regular season"}
json.dump(SLRULE, open(OUT / "frozen_shortlist.json", "w"), indent=1, sort_keys=True)
FROZEN = {"metric": "Hoop G = max over the Run-the-Hoop lap of the 0.5 s rolling mean of sideways acceleration (g), mean "
                    "over runs, z-scored within Combine year",
          "forecast": {"covariates": BASE_T, "fit_rows": "other (earlier, for 2026: all 2023-25) classes, 100+ rushes",
                       "missing_times": "linear fill from combine_weight_z, edge_share, ngs_athleticism_score",
                       "model": "OLS on empirical-Bayes pressure rate"},
          "min_nfl_rushes": RATE_MIN, "score_class": 2026, "score_after": "2027 regular season"}
json.dump(FROZEN, open(OUT / "frozen_rule.json", "w"), indent=1, sort_keys=True)
NUM["rule_sha"] = hashlib.sha256(json.dumps(SLRULE, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]

# robustness table (5 rows)
RT = [("**Primary: forecast fit on the other two classes**", "main"),
      ("Pick-and-role-only forecast", "pick"),
      ("Time-forward: fit on earlier classes only (2024–25 scored)", "tf")]
out = ["| Yardstick | Shortlist beat it | Rest of day 3 | Gap, points (90%) |", "|---|---|---|---|"]
for lab, t in RT:
    out.append(f"| {lab} | {{{{{t}_k1}}}} of {{{{{t}_n1}}}} | {{{{{t}_k0}}}} of {{{{{t}_n0}}}} | "
               f"+{{{{{t}_gap}}}} ({{{{{t}_lo}}}} to {{{{{t}_hi}}}}) |")
out.append("| Primary, reweighted for who reached 30 rushes | weighted | weighted | "
           "+{{ipw_gap}} ({{ipw_lo}} to {{ipw_hi}}) |")
MD["robust"] = "\n".join(out)

# ============================================================================= figures
C_SL, C_RE, C_D12 = OI["orange"], OI["blue"], OI["grey"]

# ---- Fig 1: (A) every scored day-3 rusher sorted by outcome; (B) hit rates by class and pooled; (C) price
fig = plt.figure(figsize=(FIGW, 9.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], width_ratios=[1.5, 1], hspace=0.32, wspace=0.28)
ax0 = fig.add_subplot(gs[0, :])
q = D3[D3.pb.notna()].copy()
q["res"] = 100 * (q.press_rate - q.pb)
q = q.sort_values("res").reset_index(drop=True)
for i, r_ in q.iterrows():
    col = C_SL if r_.short else C_RE
    ax0.plot([i, i], [0, r_.res], color=col, lw=2.2, alpha=0.9)
    hit = r_.res > 0
    ax0.plot(i, r_.res, "o", ms=8.5, color=col if hit else "white", mec=col, mew=1.8)
    if r_.short:
        nm = r_.display_name.split()[-1]
        yy = r_.res + (0.6 if hit else -0.6)
        ax0.text(i, yy, nm, rotation=90, ha="center", va="bottom" if hit else "top", fontsize=10, color="black")
ax0.axhline(0, color="black", lw=0.9)
ax0.set_xticks([]); ax0.set_xlim(-1, len(q))
ax0.set_ylim(q.res.min() - 5.5, q.res.max() + 5.5)
ax0.set_ylabel("NFL pressure rate\nminus forecast (points)")
ax0.text(0.0, 0.97, "Beat the forecast (filled) ↑", transform=ax0.transAxes, fontsize=11, va="top")
ax0.text(0.0, 0.04, "Below it (open) ↓", transform=ax0.transAxes, fontsize=11, va="bottom")
ax0.set_title(f"A. All {len(q)} scored day-3 / undrafted rushers, sorted by outcome; orange and named = shortlist "
              f"(top-third Hoop G)", loc="left", fontsize=12.5)
ax = fig.add_subplot(gs[1, 0]); bx = fig.add_subplot(gs[1, 1])
labs = [str(c) for c in classes] + ["All three\n(held out)"]
vals = [PERCLS[c] for c in classes] + [ROB["main"][:4]]
for i, (k1, n1, k0, n0) in enumerate(vals):
    for j, (k, n, col, dx) in enumerate([(k1, n1, C_SL, -0.15), (k0, n0, C_RE, 0.15)]):
        p = k / n
        lo, hi = wilson(k, n)
        ms = 11 if i == 3 else 8
        ax.plot([i + dx, i + dx], [100 * lo, 100 * hi], color=col, lw=3 if i == 3 else 2, alpha=0.8)
        ax.plot(i + dx, 100 * p, "o" if j == 0 else "s", color=col, ms=ms, mec="black", mew=0.6)
        ax.text(i + dx + (0.07 if j else -0.07), 100 * p, f"{k}/{n}", ha="left" if j else "right", va="center",
                fontsize=10.5)
ax.axvline(2.5, color="0.8", lw=1)
ax.set_xticks(range(4)); ax.set_xticklabels(labs)
ax.set_xlim(-0.6, 3.6); ax.set_ylim(-5, 100); ax.set_yticks(range(0, 101, 25))
ax.set_ylabel("Beat forecast (%)")
ax.set_title("B. Hit rate, shortlist (●) vs rest (■), 90% intervals", loc="left", fontsize=12)
for t, lab in enumerate(["Bottom", "Middle", "Top"]):
    h_ = ba[ba.third == t].pick.values
    bsm = [np.median(_rq.choice(h_, len(h_))) for _ in range(NBOOT)]
    lo, hi = np.percentile(bsm, [5, 95])
    bx.barh(t, PK3[t], color=C_SL if t == 2 else OI["sky"], alpha=0.9, height=0.6)
    bx.plot([lo, hi], [t, t], color="black", lw=1.5)
    bx.text(8, t, f"{int(PK3[t])}", va="center", ha="left", fontsize=11.5, color="black", fontweight="bold")
bx.set_yticks(range(3)); bx.set_yticklabels(["Bottom\nthird", "Middle\nthird", "Top\nthird"]); bx.set_xlim(0, 265)
bx.tick_params(axis="y", length=0)
bx.set_xlabel("Median draft pick (undrafted = 262)")
bx.set_title("C. Clubs did not pay for Hoop G", loc="left", fontsize=12)
bx.spines["left"].set_visible(False)
fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.09)
save(fig, "fig1_shortlist.png")

# ---- Fig 2: the drill-to-game chain (rho and 90% interval per link)
fig, ax = plt.subplots(figsize=(FIGW, 6.2))
LINKS = [("Rush-drill burst → NFL first-second burst*", "burst_game", OI["green"], 8),
         ("NFL first-second burst → NGS season get-off*", "gq", OI["green"], 7),
         ("NGS season get-off → pressure over forecast", "getoff_press", OI["green"], 6),
         ("End to end: rush-drill burst → pressure", "burst_press", OI["grey"], 5),
         ("Hoop G → NFL first-second burst*", "hoop_game", OI["grey"], 2.5),
         ("Hoop G → NFL depth at 1.5 s over forecast", "hoop_depth", OI["grey"], 1.5),
         ("Hoop G → pressure over forecast", "hoop_press", OI["orange"], 0.5)]
for lab, key, col, yy in LINKS:
    r, lo, hi, n = CH[key][:4]
    ax.plot([lo, hi], [yy, yy], color=col, lw=3.5, solid_capstyle="round")
    ax.plot(r, yy, "o", color=col, ms=11, mec="black", mew=0.7)
    ax.text(1.02, yy, f"ρ {r:+.2f}  (n {n})".replace("-", "\u2212"), va="center", fontsize=11,
            transform=ax.get_yaxis_transform())
    if key == "burst_game":
        da = CH[key][4]
        ax.plot(min(da, 1), yy, "D", color="white", mec=OI["green"], mew=2, ms=10)
        ax.annotate("corrected for noise", (min(da, 1), yy), (0.62, yy + 0.6), fontsize=10, color=OI["green"],
                    ha="center", arrowprops=dict(arrowstyle="-", color=OI["green"]))
ax.set_yticks([l[3] for l in LINKS]); ax.set_yticklabels([l[0] for l in LINKS])
yt = ax.get_yaxis_transform()
ax.text(-0.87, 9.0, f"BURST: each link holds, but only {NUM['ch_bg']} × {NUM['ch_gp']} ≈ {NUM['ch_prod']}\n"
        f"reaches pressure (one drill rep repeats at {NUM['rel_d1_1']})",
        transform=yt, ha="left", fontsize=11.5, color=OI["green"], fontweight="bold", va="center")
ax.text(-0.87, 3.4, "BEND: reaches pressure with no visible path", transform=yt, ha="left", fontsize=11.5,
        color=OI["orange"], fontweight="bold", va="center")
ax.axhline(4.0, color="0.85", lw=1)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-0.35, 1.0); ax.set_ylim(0, 9.8)
ax.set_xlabel("Spearman ρ with 90% player-bootstrap interval (* after weight and EDGE share)")
fig.subplots_adjust(left=0.40, right=0.84, top=0.97, bottom=0.12)
save(fig, "fig4_chain.png")

# ---- Fig 3: four linemen's hoop laps, one colour scale, readouts
fig, axs = plt.subplots(2, 2, figsize=(FIGW, 8.4))
norm = plt.Normalize(0, float(np.nanpercentile(np.concatenate([traces[e]["_alat"] for e in hoop.hoop_eid]), 99)))
cmap = plt.get_cmap("cividis")
mns = lambda v: f"{v:+.1f}".replace("-", "\u2212")
for ax, nm in zip(axs.ravel(), CASES):
    r_ = CROWS[nm]
    eids = hoop_att[hoop_att.nfl_id == r_.nfl_id].sort_values("bend_g", ascending=False).event_id.tolist()
    tr = traces[eids[0]]
    x, y, al = tr["_x"] - np.mean(tr["_x"]), tr["_y"] - np.mean(tr["_y"]), tr["_alat"]
    pts = np.c_[x, y].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=norm, lw=5)
    lc.set_array(al[:-1])
    ax.add_collection(lc)
    ax.plot(x[tr["_k"]], y[tr["_k"]], "o", ms=13, mfc="none", mec=OI["red"], mew=2.8)
    ax.set_xlim(-6.5, 6.5); ax.set_ylim(-5.6, 3.2); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    tag = "Day-3 shortlist" if r_.short else ("Bottom third" if r_.third == 0 else "Middle third")
    res = "beat" if r_.press_rate > r_.pb else "below"
    ax.set_title(f"{nm} ({int(r_.draft_year)}, pick {int(r_.pick)})\n"
                 f"{tag}: Hoop G {mns(r_.hoop_z)} sd ({r_.hoop_g:.2f} g)\n"
                 f"NFL pressure {100 * r_.press_rate:.1f}% vs forecast {100 * r_.pb:.1f}% ({res})",
                 loc="left", fontsize=11.5, linespacing=1.4)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, orientation="horizontal", fraction=0.04,
                  pad=0.03, aspect=50)
cb.set_label("Sideways acceleration on the lap (g); red ring = hardest half-second (Hoop G)")
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.16, hspace=0.45, wspace=0.08)
save(fig, "fig2_laps.png")

# ---- Fig 4: every scored rusher: pressure over forecast vs Hoop G, day 3 highlighted
fig, ax = plt.subplots(figsize=(FIGW, 5.8))
for d3, col, lab in [(False, C_D12, "Day 1–2"), (True, C_RE, "Day 3 / undrafted")]:
    q = SC[(SC.pick >= DAY3) == d3]
    ax.scatter(q.hoop_z, q.res, s=12 + q.rush_snaps / 6, color=col, alpha=0.55 if not d3 else 0.8,
               edgecolor="black" if d3 else "none", lw=0.5, label=lab)
q = SC[SC.short]
ax.scatter(q.hoop_z, q.res, s=12 + q.rush_snaps / 6, color=C_SL, edgecolor="black", lw=0.8, label="Day-3 shortlist")
xs = np.linspace(SC.hoop_z.min(), SC.hoop_z.max(), 50)
ax.plot(xs, b0 * xs + np.average(SC.res, weights=SC.rush_snaps), color="black", lw=1.5)
ax.axhline(0, color="0.6", lw=0.8, ls="--")
for nm in CASES:
    r_ = SC[SC.display_name == nm].iloc[0]
    ax.annotate(nm.split()[-1], (r_.hoop_z, r_.res), (8, 6), textcoords="offset points", fontsize=11)
ax.set_xlabel("Hoop G (sd within Combine class)")
ax.set_ylabel("NFL pressure rate minus forecast (points)")
ax.set_title(f"All {len(SC)} scored hoop runners (marker area = NFL rushes); line = rush-weighted slope", loc="left",
             fontsize=12)
ax.legend(frameon=False, loc="upper left", fontsize=11)
fig.tight_layout()
save(fig, "fig3_scatter.png")

# ============================================================================= render
tpl = (OUT / "writeup_template.md").read_text()
text = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda m_: MD[m_.group(1)], tpl)
used = set()


def sub(m_):
    k = m_.group(1)
    if k not in NUM:
        raise KeyError(f"unknown key {k}")
    used.add(k)
    v = str(NUM[k])
    return "−" + v[1:] if v.startswith("-") else v


(OUT / "writeup.md").write_text(re.sub(r"\{\{(\w+)\}\}", sub, text))
json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1, sort_keys=True)
print(json.dumps({k: NUM[k] for k in sorted(NUM)}, indent=0))
print("unused keys:", sorted(set(NUM) - used))
