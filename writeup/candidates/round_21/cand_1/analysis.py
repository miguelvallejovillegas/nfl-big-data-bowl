"""Round 21 cand 1: DL/EDGE Run-the-Hoop bend (Hoop G) and pass-rush-drill burst vs NFL get-off, bend and pressure.

Run from the repo root:
    uv run python writeup/candidates/round_21/cand_1/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (every number the text quotes), board.csv,
frozen_rule.json and figures/*.png next to this file, runs verify.py (an independent re-implementation of the
headline slope) and quotes its result, then renders writeup.md from writeup_template.md ({{key}} -> numbers.json
value; fails on an unknown key). Frame-level features are cached in data/cache/r21c1 (gitignored). Runtime about
4 minutes.

New against round 20 cand 1 (the same DL line, 8.20):
  * Transfer claims get a no-tracking benchmark: official 3-cone -> NFL bend and official 10-yd split -> NGS get-off
    on the same partial-Spearman design as the drill links.
  * Inverse-probability weighting for being in the analysis set (all six drill measures, 30+ rushes).
  * Day-3 top-third hit gap (forecast + Hoop G minus pick order) with a class-stratified rusher bootstrap and a
    stated bar for board use (+20 points with the 90% lower bound above zero).
  * A count of every hypothesis test the writeup reports.
  * Hero figure: drill lap vs sampled NFL rushes on one yard scale for four named rushers; 2026 player cards with
    sampled-rush splits (wide vs inside alignment, third down) and what flips each call.

Design (one estimand per claim):
  * Drill metrics, z within Combine year, mean over attempts:
      Hoop G  = hardest 0.5 s of sideways (centripetal) acceleration on the Run-the-Hoop lap, g.
      Burst   = yards covered in the first second after movement onset in the pass-rush drill.
  * Game traits (random sampled NFL pass rushes only, self-referenced, no blockers or ball in the data):
      NFL get-off   = NGS season get-off (rush-weighted), sign flipped so higher = quicker.
      NFL burst     = yards covered in the first second after the snap.
      NFL bend      = hardest 0.5 s of sideways acceleration in the first 2.0 s after the snap, g.
  * Outcome: career pass-rush pressure rate 2023-25, empirical-Bayes shrunk, linemen with 30+ rushes.
  * Baseline without tracking: log pick, Combine weight, NFL EDGE share, NGS athleticism score, official 10-yd split
    and 3-cone (missing times filled from weight, role and NGS inside each fold). Leave-one-class-out OLS.
  * One pre-set margin: 0.5 pressure-rate points per sd (2.25 pressures per 450-rush season).
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
CACHE = ROOT / "data" / "cache" / "r21c1"
CACHE.mkdir(parents=True, exist_ok=True)

NBOOT = int(os.environ.get("BDB_NBOOT", 1000))
NPERM = 5000
RATE_MIN, REG_MIN, DAY3, UDFA_PICK = 30, 100, 103, 262
MARGIN = 0.5            # pressure-rate points per sd: smallest effect worth acting on, set before fitting
SEASON = 450            # rushes in a regular's season
PS = SEASON / 100       # pressure-rate points -> pressures per 450-rush season
DT, W1 = 0.1, 10
ON, PRE = 1.0, 0.5
YD_PER_G = 9.80665 / 0.9144
Z90 = 1.6449
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "grey": "#8C8C8C", "black": "#000000", "yellow": "#F0E442"}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
FIGW = 1600 / 150

NUM: dict = {}


def put(key, v, nd=2):
    if key in NUM:
        raise KeyError(f"duplicate key {key}")
    if isinstance(v, (int, np.integer)):
        NUM[key] = int(v)
        return int(v)
    s = f"{float(v):.{nd}f}"
    if re.fullmatch(r"-0\.?0*", s):
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


def wilson(k, n, z=Z90):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot_psp(df, x, y, Z, seed):
    """Partial Spearman of x and y given Z (ranks residualized), class-stratified 90% player bootstrap."""
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
    return r, float(lo), float(hi), int(len(d))


def fm(x, nd=1):
    t_ = f"{x:+.{nd}f}"
    if float(t_) == 0:
        t_ = f"{0:.{nd}f}"
    return t_.replace("-", "−")


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
for c in ["combine_weight", "ten_yd_split", "three_cone"]:
    dl[c + "_z"] = yz(dl[c], dl.draft_year)
a = align.merge(dl[["nfl_id"]], on="nfl_id")
a["edge_pl"] = np.where(a.lined_up_position.isin(["EDGE", "OLB"]), a.plays, 0)
es = a.groupby("nfl_id").agg(edge_pl=("edge_pl", "sum"), pl=("plays", "sum"))
dl = dl.merge((es.edge_pl / es.pl).rename("edge_share").reset_index(), on="nfl_id", how="left")
dl["edge_share"] = dl.edge_share.fillna(dl.nfl_position.isin(["DE", "OLB"]).astype(float))
dl["role"] = np.where(dl.edge_share >= 0.5, "EDGE", "interior")
classes = [2023, 2024, 2025]
put("n_dl", len(dl)); put("margin", MARGIN, 1)
put("margin_season", MARGIN * PS, 2); put("season_rushes", SEASON)
put("n_three_cone", int(dl.three_cone.notna().sum()))
put("pct_day3_dl", 100 * (dl.pick >= DAY3).mean(), 0)

# ============================================================================= Combine drill kinematics
ev = events[(events.position_group == "DL") & events.drill_name.isin(["PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL"])]
ev = ev[["event_id", "nfl_id", "draft_year", "drill_name", "attempt_no", "peak_accel", "top_speed", "duration_s"]]


def onset(s, start=0):
    s = np.asarray(s, float)
    hit = np.flatnonzero(s[start:] >= ON)
    if not len(hit):
        return None
    i = j = start + int(hit[0])
    while j > start and s[j - 1] >= PRE and i - j < 3:
        j -= 1
    return j


def lateral_g(x, y):
    xs, ys = savgol_filter(x, 7, 2), savgol_filter(y, 7, 2)
    vx, vy = savgol_filter(xs, 7, 2, deriv=1, delta=DT), savgol_filter(ys, 7, 2, deriv=1, delta=DT)
    ax, ay = savgol_filter(xs, 7, 2, deriv=2, delta=DT), savgol_filter(ys, 7, 2, deriv=2, delta=DT)
    v = np.hypot(vx, vy)
    alat = np.abs(vx * ay - vy * ax) / np.maximum(v, 0.5) / YD_PER_G
    roll = pd.Series(alat).rolling(5, center=True).mean().to_numpy()
    return alat, roll, xs, ys, v


def attempt_features(g):
    x, y, s = g.x.to_numpy(), g.y.to_numpy(), g.s.to_numpy()
    if len(x) < 15:
        return None
    alat, roll, xs, ys, v = lateral_g(x, y)
    k = int(np.nanargmax(roll))
    o = onset(s)
    d1 = np.nan
    if o is not None and s[0] < ON and o + W1 < len(x):
        d1 = float(np.hypot(x[o + W1] - x[o], y[o + W1] - y[o]))
    return {"bend_g": float(np.nanmax(roll)), "d1": d1, "speed_at_bend": float(v[k]),
            "_alat": alat, "_x": xs, "_y": ys, "_k": k}


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
hoop_att["z"] = yz(hoop_att.bend_g, hoop_att.draft_year)
rush_att["z"] = yz(rush_att.d1, rush_att.draft_year)
rel_hoop1, n_hrep = icc(hoop_att.nfl_id, hoop_att.z)
rel_d1_1, n_drep = icc(rush_att.nfl_id, rush_att.z)
REL = {"hoop_z": sb(rel_hoop1, hoop_att.groupby("nfl_id").size().mean()),
       "burst_z": sb(rel_d1_1, rush_att.dropna(subset=["d1"]).groupby("nfl_id").size().mean())}
put("hoop_g_median", hoop_att.bend_g.median())
put("hoop_speed_median", hoop_att.speed_at_bend.median(), 1)
# coach version: Hoop G = speed^2 / turn radius, so the median runner's radius at his hardest lean
put("hoop_radius_median", (hoop_att.speed_at_bend ** 2 / (hoop_att.bend_g * YD_PER_G)).median(), 1)
put("rel_hoop1", rel_hoop1); put("rel_hoop_nrep", n_hrep); put("rel_hoop2", sb(rel_hoop1, 2))
put("rel_burst1", rel_d1_1); put("rel_burst_nrep", n_drep)
put("burst_median", rush_att.d1.median())

# the pre-listed family: three measures per drill, all mean over attempts, z within Combine year
FAM = [("burst_z", "Rush drill: burst (yd in first 1 s)", rush_att, "d1", 1),
       ("rpa_z", "Rush drill: peak acceleration", rush_att, "peak_accel", 1),
       ("rts_z", "Rush drill: top speed", rush_att, "top_speed", 1),
       ("hoop_z", "Hoop: Hoop G (sideways g)", hoop_att, "bend_g", 1),
       ("hts_z", "Hoop: top speed", hoop_att, "top_speed", 1),
       ("hlap_z", "Hoop: lap time (faster = higher)", hoop_att, "duration_s", -1)]
for key, _, src, col, sgn in FAM:
    pp = src.groupby("nfl_id").agg(v=(col, "mean"), draft_year=("draft_year", "first")).reset_index()
    pp[key] = sgn * yz(pp.v, pp.draft_year)
    dl = dl.merge(pp[["nfl_id", key]], on="nfl_id", how="left")
hk = hoop_att.groupby("nfl_id").agg(hoop_g=("bend_g", "mean"), hoop_k=("bend_g", "size"),
                                    hoop_eid=("event_id", "first")).reset_index()
dl = dl.merge(hk, on="nfl_id", how="left")
put("n_hoop", int(dl.hoop_z.notna().sum())); put("n_burst", int(dl.burst_z.notna().sum()))
put("n_hoop_no3c", int((dl.hoop_z.notna() & dl.three_cone.isna()).sum()))
put("pct_one_run", 100 * (dl.hoop_k == 1).sum() / dl.hoop_z.notna().sum(), 0)
_h3 = dl.dropna(subset=["hoop_z", "three_cone_z"])
put("rho_hoop_3cone", -stats.spearmanr(_h3.hoop_z, _h3.three_cone_z)[0]); put("n_hoop_3cone", len(_h3))
_b10 = dl.dropna(subset=["burst_z", "ten_yd_split_z"])
put("rho_burst_split", -stats.spearmanr(_b10.burst_z, _b10.ten_yd_split_z)[0]); put("n_burst_split", len(_b10))

# ============================================================================= NFL season outcomes
s = seasons[seasons.position_group == "DL"].copy()
s["w_go"] = s.pass_rush_snaps * s.avg_get_off
agg = s.groupby("nfl_id").agg(rush_snaps=("pass_rush_snaps", "sum"), pressures=("pressures", "sum"),
                              w_go=("w_go", "sum")).reset_index()
agg["get_off"] = agg.w_go / agg.rush_snaps
dl = dl.merge(agg, on="nfl_id", how="left")
dl["rush_snaps"] = dl.rush_snaps.fillna(0)
m = dl.rush_snaps >= 20
n_, k_ = dl.loc[m, "rush_snaps"], dl.loc[m, "pressures"]
mu = k_.sum() / n_.sum()
N_ = n_.sum()
tau2 = ((n_ * (k_ / n_ - mu) ** 2).sum() - (len(n_) - 1) * mu * (1 - mu)) / (N_ - (n_ ** 2).sum() / N_)
Mw = mu * (1 - mu) / tau2 - 1
mm = dl.rush_snaps > 0
dl.loc[mm, "press_eb"] = 100 * (dl.loc[mm, "pressures"] + Mw * mu) / (dl.loc[mm, "rush_snaps"] + Mw)
dl["quick"] = -dl.get_off
put("press_mean", 100 * mu, 1); put("eb_prior_rushes", Mw, 0)
s25 = s[s.season == 2025].groupby("nfl_id").agg(r25=("pass_rush_snaps", "sum"), p25=("pressures", "sum"))
dl = dl.merge(s25.reset_index(), on="nfl_id", how="left")

# ============================================================================= sampled NFL tracking
dlids = set(dl.nfl_id)
K = ["game_id", "play_id", "nfl_id"]
g = gp[gp.nfl_id.isin(dlids)].drop_duplicates(K)
pinfo = g.set_index(K)[["sample_kind", "time_to_pressure", "sack", "player_get_off", "expected_points_added"]]


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
        # fixed 2.0 s window, only when the play lasts that long, so pressured snaps are not longer by construction
        w = (fr >= sf) & (fr <= sf + 20)
        bend = np.nan
        if w.sum() == 21 and (pd.isna(e) or e >= sf + 20):
            _, roll, _, _, _ = lateral_g(x_[w], y_[w])
            bend = float(np.nanmax(roll))
        out.append(dict(zip(K, key), g_d1=d1, g_bend=bend, kind=info.sample_kind,
                        press=int(pd.notna(info.time_to_pressure) or info.sack == 1)))
    return pd.DataFrame(out)


GF = cached("game_fixed2s", game_features)
GR = GF[GF.kind == "random"].copy()
GR["half"] = GR.groupby("nfl_id").cumcount() % 2
put("n_random_rushes", len(GR)); put("n_random_players", GR.nfl_id.nunique())


def split_half(df, col, min_n):
    cnt = df.groupby("nfl_id")[col].count()
    keep = cnt[cnt >= min_n].index
    a_ = df[df.nfl_id.isin(keep)].groupby(["nfl_id", "half"])[col].mean().unstack().dropna()
    r_ = stats.spearmanr(a_[0], a_[1])[0]
    return 2 * r_ / (1 + r_), len(a_)


REL["g_d1"], _ = split_half(GR, "g_d1", 8)
REL["g_bend"], n_rb = split_half(GR, "g_bend", 8)
put("rel_game_burst", REL["g_d1"]); put("rel_game_bend", REL["g_bend"]); put("rel_game_n", n_rb)
PG = GR.groupby("nfl_id").agg(g_d1=("g_d1", "mean"), g_bend=("g_bend", "mean"), g_n=("g_bend", "count")).reset_index()
PG = PG[PG.g_n >= 8]
dl = dl.merge(PG, on="nfl_id", how="left")
put("median_random_rushes", PG.g_n.median(), 0)
_s2 = s[s.pass_rush_snaps >= 100].pivot_table(index="nfl_id", columns="season", values="avg_get_off")
_pp = pd.concat([pd.DataFrame({"a": _s2[a_], "b": _s2[b_]}).dropna() for a_, b_ in [(2023, 2024), (2024, 2025)]])
REL["quick"] = stats.spearmanr(_pp.a, _pp.b)[0]
put("rel_getoff_season", REL["quick"]); put("rel_getoff_n", len(_pp))
_s3 = s[s.pass_rush_snaps >= 100].assign(pr=lambda d_: d_.pressures / d_.pass_rush_snaps).pivot_table(
    index="nfl_id", columns="season", values="pr")
_pq = pd.concat([pd.DataFrame({"a": _s3[a_], "b": _s3[b_]}).dropna() for a_, b_ in [(2023, 2024), (2024, 2025)]])
put("rel_press_season", stats.spearmanr(_pq.a, _pq.b)[0]); put("rel_press_n", len(_pq))
# EPA of a pressured snap (random sampled snaps)
_gm = g[(g.sample_kind == "random") & g.expected_points_added.notna()].drop_duplicates(["game_id", "play_id"])
_pr = _gm.time_to_pressure.notna() | (_gm.sack == 1)
e1, e0 = _gm[_pr].expected_points_added.values, _gm[~_pr].expected_points_added.values
_r = np.random.default_rng(101)
_eb = [_r.choice(e0, len(e0)).mean() - _r.choice(e1, len(e1)).mean() for _ in range(NBOOT)]
EPA = e0.mean() - e1.mean()
put("epa_press", EPA); put("epa_press_lo", np.percentile(_eb, 5)); put("epa_press_hi", np.percentile(_eb, 95))
put("epa_n", len(_gm))
put("margin_epa", MARGIN * PS * EPA, 1)

# ============================================================================= the chain: drill -> NFL trait -> pressure
Zc = ["combine_weight_z", "edge_share"]
co30 = dl[dl.rush_snaps >= RATE_MIN].copy()
CHAIN = {}
for dx, gy, seed in [("burst_z", "quick", 11), ("burst_z", "g_d1", 12), ("hoop_z", "g_bend", 13),
                     ("hoop_z", "quick", 14)]:
    r, lo, hi, n = boot_psp(dl, dx, gy, Zc, seed)
    CHAIN[(dx, gy)] = (r, lo, hi, n)
    tag = {"burst_z": "burst", "hoop_z": "hoop"}[dx] + "_" + gy
    put(f"tr_{tag}", r); put(f"tr_{tag}_lo", lo); put(f"tr_{tag}_hi", hi); put(f"tr_{tag}_n", n)
    put(f"tr_{tag}_ceil", np.sqrt(REL[dx] * REL[gy]))
# no-tracking benchmark for each transfer link: the official time that should carry the same trait
dl["three_cone_fast"] = -dl.three_cone_z
dl["split_fast"] = -dl.ten_yd_split_z
for bx, gy, tag, seed in [("three_cone_fast", "g_bend", "3c_bend", 15), ("split_fast", "quick", "split_quick", 16),
                          ("three_cone_fast", "quick", "3c_quick", 17)]:
    r, lo, hi, n = boot_psp(dl, bx, gy, Zc, seed)
    put(f"bm_{tag}", r); put(f"bm_{tag}_lo", lo); put(f"bm_{tag}_hi", hi); put(f"bm_{tag}_n", n)
# Hoop G -> NFL bend on the 3-cone runners only, so both columns use the same linemen
_r = boot_psp(dl[dl.three_cone_z.notna()], "hoop_z", "g_bend", Zc, 18)
put("tr_hoop_bend_3csub", _r[0]); put("tr_hoop_bend_3csub_n", _r[3])
for gy, seed in [("quick", 21), ("g_d1", 22), ("g_bend", 23)]:
    r, lo, hi, n = boot_psp(co30, gy, "press_eb", Zc, seed)
    put(f"gp_{gy}", r); put(f"gp_{gy}_lo", lo); put(f"gp_{gy}_hi", hi); put(f"gp_{gy}_n", n)

# ============================================================================= held-out baseline and increments
MAIN = ["log_pick", "combine_weight_z", "edge_share", "ngs_athleticism_score"]
TIMES = ["ten_yd_split_z", "three_cone_z"]
BASE = MAIN + TIMES
FILLX = ["combine_weight_z", "edge_share", "ngs_athleticism_score"]
dl["ngs_athleticism_score"] = dl.ngs_athleticism_score.fillna(dl.ngs_athleticism_score.median())
co30 = dl[dl.rush_snaps >= RATE_MIN].copy()


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


def ols(tr, te, cols, y="press_eb"):
    A = np.c_[np.ones(len(tr)), tr[cols].values]
    b = np.linalg.lstsq(A, tr[y].values, rcond=None)[0]
    return np.c_[np.ones(len(te)), te[cols].values] @ b, b


def loco(d, cols):
    pred = np.full(len(d), np.nan)
    cl = d.draft_year.values
    for c in classes:
        tr, te = fill_times(d[cl != c], d[cl == c])
        pred[cl == c] = ols(tr, te, cols)[0]
    return pred


def oos(y, p):
    sse = ((y - p) ** 2).sum()
    return 1 - sse / ((y - y.mean()) ** 2).sum(), np.sqrt(sse / len(y)), stats.spearmanr(y, p)[0]


FKEYS = [f[0] for f in FAM]
AN = co30.dropna(subset=FKEYS).reset_index(drop=True)
put("n_an", len(AN))
NUM["ho_n_per_class"] = "/".join(str(int((AN.draft_year == c).sum())) for c in classes)
MODELS = [("Pick + role only", ["log_pick", "edge_share"]), ("No-tracking baseline", BASE),
          ("Baseline + Hoop G", BASE + ["hoop_z"]), ("Baseline + rush-drill burst", BASE + ["burst_z"])]
MK_ = {"Pick + role only": "pick", "No-tracking baseline": "base", "Baseline + Hoop G": "hoop",
       "Baseline + rush-drill burst": "burst"}
y = AN.press_eb.values
PRED = {nm: loco(AN, cols) for nm, cols in MODELS}
RES = y - PRED["No-tracking baseline"]
AN["fc"] = PRED["No-tracking baseline"]
grp = [np.flatnonzero(AN.draft_year.values == c) for c in classes]
rng = np.random.default_rng(2027)
BOOT = {nm: [] for nm, _ in MODELS}
BRES = {k: [] for k in FKEYS}
for _ in range(NBOOT):
    ix = np.concatenate([rng.choice(g_, len(g_)) for g_ in grp])
    d_ = AN.iloc[ix].reset_index(drop=True)
    yb = d_.press_eb.values
    for nm, cols in MODELS:
        p_ = loco(d_, cols)
        BOOT[nm].append(oos(yb, p_)[0])
        if nm == "No-tracking baseline":
            rb = yb - p_
            for k in FKEYS:
                BRES[k].append(np.polyfit(d_[k].values, rb, 1)[0])
for nm, _ in MODELS:
    key = MK_[nm]
    r2, rmse, rho = oos(y, PRED[nm])
    put(f"r2_{key}", r2); put(f"rmse_{key}", rmse); put(f"rho_{key}", rho)
    if key in ("hoop", "burst"):
        dd = np.array(BOOT[nm]) - np.array(BOOT["No-tracking baseline"])
        put(f"dr2_{key}", r2 - oos(y, PRED["No-tracking baseline"])[0])
        put(f"dr2_{key}_lo", np.percentile(dd, 5)); put(f"dr2_{key}_hi", np.percentile(dd, 95))

# ---- headline estimand: held-out slope of (pressure rate - leave-one-class-out baseline) on each drill measure,
# for all six pre-listed measures. Intervals: class-stratified bootstrap refitting every fold. p: within-class
# permutation. Family-wise p: rows of the 6-measure matrix permuted jointly within class (keeps the correlation
# between measures), statistic = max |slope| over the family.
X6 = AN[FKEYS].values
Xc = X6 - X6.mean(0)


def slopes(Xm, r):
    Xm = Xm - Xm.mean(0)
    return (Xm * (r - r.mean())[:, None]).sum(0) / (Xm ** 2).sum(0)


obs = slopes(X6, RES)
prm = np.random.default_rng(77)
PERM = np.empty((NPERM, len(FKEYS)))
for i_ in range(NPERM):
    ix = np.arange(len(AN))
    for g_ in grp:
        ix[g_] = prm.permutation(g_)
    PERM[i_] = slopes(X6[ix], RES)
maxabs = np.abs(PERM).max(1)
FAMR = []
for j, (key, lab, *_rest) in enumerate(FAM):
    bs = np.array(BRES[key])
    lo, hi = np.percentile(bs, [5, 95])
    p_one = (1 + (PERM[:, j] >= obs[j]).sum()) / (NPERM + 1)
    p_two = (1 + (np.abs(PERM[:, j]) >= abs(obs[j])).sum()) / (NPERM + 1)
    p_fw = (1 + (maxabs >= abs(obs[j])).sum()) / (NPERM + 1)
    cls_s = [np.polyfit(X6[g_, j], RES[g_], 1)[0] for g_ in grp]
    FAMR.append(dict(key=key, lab=lab, est=obs[j], lo=lo, hi=hi, p=p_one, p2=p_two, pfw=p_fw, cls=cls_s, bs=bs))
    t = key.replace("_z", "")
    put(f"ho_{t}_season", obs[j] * PS, 1); put(f"ho_{t}_season_lo", lo * PS, 1)
    put(f"ho_{t}_season_hi", hi * PS, 1)
    put(f"ho_{t}_p", p_one); put(f"ho_{t}_p2", p_two); put(f"ho_{t}_pfw", p_fw)
    put(f"ho_{t}_pabove", 100 * (bs > MARGIN).mean(), 0)
    for c, v in zip(classes, cls_s):
        put(f"ho_{t}_{c}", v * PS, 1)
FW95 = np.percentile(maxabs, 95) * PS
put("fw95_season", FW95, 1)
put("n_family", len(FAM))
HO = {f["key"]: f for f in FAMR}
put("ho_hoop_pos_classes", int(sum(v > 0 for v in HO["hoop_z"]["cls"])))
# equivalence (TOST) for burst against +/- margin: both one-sided bootstrap tails
bsb = HO["burst_z"]["bs"]
put("ho_burst_tost", max((1 + (bsb <= -MARGIN).sum()) / (len(bsb) + 1), (1 + (bsb >= MARGIN).sum()) / (len(bsb) + 1)))
put("ho_burst_pinside", 100 * (np.abs(bsb) < MARGIN).mean(), 0)

# ---- inverse-probability weight for being in the analysis set (all six drill measures) among 30+ rush linemen
import statsmodels.api as sm
_ip = co30.copy()
_ip["in_set"] = _ip[FKEYS].notna().all(1).astype(int)
_ipx = sm.add_constant(_ip[["log_pick", "edge_share", "ngs_athleticism_score"]].assign(
    w=_ip.combine_weight_z.fillna(0)))
_ps = sm.Logit(_ip.in_set, _ipx).fit(disp=0).predict(_ipx)
_ip["ipw"] = 1 / _ps
W_IPW = AN.merge(_ip[["nfl_id", "ipw"]], on="nfl_id", how="left").ipw.values
_xh = AN.hoop_z.values
_xm = _xh - np.average(_xh, weights=W_IPW)
IPW_SLOPE = float(np.sum(W_IPW * _xm * (RES - np.average(RES, weights=W_IPW))) / np.sum(W_IPW * _xm ** 2))
put("ipw_hoop_season", IPW_SLOPE * PS, 1); put("n_co30", len(_ip)); put("n_in_set", int(_ip.in_set.sum()))
put("ipw_max", float(W_IPW.max()), 1)

# ---- sensitivity of the Hoop G slope (whole pipeline rerun)


def run_variant(mult=1.0, rmin=RATE_MIN, nperm=2000, seed=5):
    d_ = dl.copy()
    M_ = Mw * mult
    mk = d_.rush_snaps > 0
    d_.loc[mk, "press_eb"] = 100 * (d_.loc[mk, "pressures"] + M_ * mu) / (d_.loc[mk, "rush_snaps"] + M_)
    an_ = d_[d_.rush_snaps >= rmin].dropna(subset=FKEYS).reset_index(drop=True)
    res_ = an_.press_eb.values - loco(an_, BASE)
    xh = an_.hoop_z.values
    sl_ = np.polyfit(xh, res_, 1)[0]
    rg = np.random.default_rng(seed)
    gg = [np.flatnonzero(an_.draft_year.values == c) for c in classes]
    cnt = 0
    for _ in range(nperm):
        xp = xh.copy()
        for g_ in gg:
            xp[g_] = rg.permutation(xp[g_])
        cnt += np.polyfit(xp, res_, 1)[0] >= sl_
    return sl_ * PS, (1 + cnt) / (nperm + 1), len(an_)


SENS = [("prior x0.5", 0.5, RATE_MIN), ("prior x2", 2.0, RATE_MIN), ("20+ rushes", 1.0, 20), ("50+ rushes", 1.0, 50)]
SV = [run_variant(mlt, rmn) for _, mlt, rmn in SENS]
put("sens_slope_min", min(v[0] for v in SV), 1); put("sens_slope_max", max(v[0] for v in SV), 1)
put("sens_p_max", max(v[1] for v in SV)); put("n_sens", len(SENS))

# ============================================================================= day-3 decision test, one row per rusher
# Rankers inside each class's day-3/undrafted rushers (30+ rushes, all six drill measures). Everything is out of
# class: the forecast is the leave-one-class-out baseline, and the Hoop G weight added to it is the held-out
# residual slope estimated on the OTHER two classes. Score = within-class Spearman with shrunk pressure rate,
# pooled across classes (weighted by n). Interval: class-stratified rusher bootstrap of the same day-3 rows.
AN["w_out"] = np.nan
for c in classes:
    o_ = AN.draft_year != c
    AN.loc[AN.draft_year == c, "w_out"] = np.polyfit(AN.loc[o_, "hoop_z"], RES[o_.values], 1)[0]
AN["fc_hoop"] = AN.fc + AN.w_out * AN.hoop_z
AN["neg_pick"] = -AN.pick
D3 = AN[AN.pick >= DAY3].reset_index(drop=True)
put("d3_n", len(D3)); NUM["d3_n_per_class"] = "/".join(str(int((D3.draft_year == c).sum())) for c in classes)
put("d3_udfa", int((D3.pick >= UDFA_PICK).sum()))
RANK = [("neg_pick", "Earlier pick"), ("fc", "No-tracking forecast"), ("fc_hoop", "Forecast + Hoop G"),
        ("hoop_z", "Hoop G alone"), ("burst_z", "Rush-drill burst alone")]
g3 = [np.flatnonzero(D3.draft_year.values == c) for c in classes]


def pooled_rho(d, col, groups):
    num = den = 0.0
    for g_ in groups:
        if len(g_) < 4:
            continue
        r_ = stats.spearmanr(d[col].values[g_], d.press_eb.values[g_])[0]
        if np.isnan(r_):
            continue
        num += r_ * len(g_); den += len(g_)
    return num / den


def top_third_hit(d, col, groups):
    """share of each class's top third (by the ranker) that beat the class's day-3 median pressure rate."""
    k = n = 0
    for g_ in groups:
        v, yv = d[col].values[g_], d.press_eb.values[g_]
        top = np.argsort(-v, kind="stable")[: int(round(len(g_) / 3))]
        k += int((yv[top] > np.median(yv)).sum()); n += len(top)
    return k, n


RR = {col: pooled_rho(D3, col, g3) for col, _ in RANK}
BR = {col: [] for col, _ in RANK}
HITGAP = []
brg = np.random.default_rng(303)
for _ in range(NBOOT):
    ix = np.concatenate([brg.choice(g_, len(g_)) for g_ in g3])
    db = D3.iloc[ix].reset_index(drop=True)
    gb = [np.flatnonzero(db.draft_year.values == c) for c in classes]
    for col, _ in RANK:
        BR[col].append(pooled_rho(db, col, gb))
    ka, na = top_third_hit(db, "fc_hoop", gb)
    kb, nb = top_third_hit(db, "neg_pick", gb)
    HITGAP.append(100 * (ka / na - kb / nb))
BAR_GAP = 20  # points of top-third hit rate over pick order needed before Hoop G may touch board order
DEC = []
for col, lab in RANK:
    t = {"neg_pick": "pick", "fc": "fc", "fc_hoop": "fch", "hoop_z": "hoop", "burst_z": "burst"}[col]
    lo, hi = np.nanpercentile(BR[col], [5, 95])
    k, n = top_third_hit(D3, col, g3)
    wl, wh = wilson(k, n)
    DEC.append(dict(col=col, lab=lab, r=RR[col], lo=lo, hi=hi, k=k, n=n, wl=wl, wh=wh))
    put(f"dr_{t}", RR[col]); put(f"dr_{t}_lo", lo); put(f"dr_{t}_hi", hi)
    put(f"tt_{t}_k", k); put(f"tt_{t}_n", n)
    put(f"tt_{t}_pct", 100 * k / n, 0); put(f"tt_{t}_lo", 100 * wl, 0); put(f"tt_{t}_hi", 100 * wh, 0)
for a_, b_, t in [("fc_hoop", "fc", "fch_fc"), ("fc_hoop", "neg_pick", "fch_pick"), ("hoop_z", "neg_pick", "hoop_pick")]:
    dd = np.array(BR[a_]) - np.array(BR[b_])
    put(f"dd_{t}", RR[a_] - RR[b_]); put(f"dd_{t}_lo", np.nanpercentile(dd, 5))
    put(f"dd_{t}_hi", np.nanpercentile(dd, 95)); put(f"dd_{t}_ppos", 100 * np.nanmean(dd > 0), 0)
_ka, _na = top_third_hit(D3, "fc_hoop", g3)
_kb, _nb = top_third_hit(D3, "neg_pick", g3)
put("hg_gap", 100 * (_ka / _na - _kb / _nb), 0)
put("hg_gap_lo", np.percentile(HITGAP, 5), 0); put("hg_gap_hi", np.percentile(HITGAP, 95), 0)
put("hg_bar", BAR_GAP)
# Hoop G added to the forecast: how often does it change the order, and how far does it move a rusher?
put("w_out_season_min", AN.w_out.min() * PS, 1); put("w_out_season_max", AN.w_out.max() * PS, 1)
# film flag: top third of Hoop G among day-3 rushers vs the rest, beat the no-tracking forecast?
D3["flag"] = D3.groupby("draft_year").hoop_z.transform(lambda v: v.rank(pct=True) > 2 / 3)
D3["beat"] = D3.press_eb > D3.fc
k1, n1 = int(D3[D3.flag].beat.sum()), int(D3.flag.sum())
k0, n0 = int(D3[~D3.flag].beat.sum()), int((~D3.flag).sum())
l1, u1 = wilson(k1, n1); l0, u0 = wilson(k0, n0)
put("fl_k", k1); put("fl_n", n1); put("fl_pct", 100 * k1 / n1, 0); put("fl_lo", 100 * l1, 0); put("fl_hi", 100 * u1, 0)
put("rs_k", k0); put("rs_n", n0); put("rs_pct", 100 * k0 / n0, 0); put("rs_lo", 100 * l0, 0); put("rs_hi", 100 * u0, 0)
put("fl_err", 100 - 100 * k1 / n1, 0); put("fl_err_lo", 100 - 100 * u1, 0); put("fl_err_hi", 100 - 100 * l1, 0)
put("fl_p_fisher", stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")[1])
put("fl_over_season", (D3[D3.flag].press_eb - D3[D3.flag].fc).mean() * PS - (D3[~D3.flag].press_eb -
                                                                             D3[~D3.flag].fc).mean() * PS, 1)
# price: do clubs draft for Hoop G?
_ba = dl[dl.hoop_z.notna()].copy()
_ba["third"] = _ba.groupby("draft_year").hoop_z.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
put("pick_top_third", _ba[_ba.third == 2].pick.median(), 0); put("pick_bot_third", _ba[_ba.third == 0].pick.median(), 0)

# 2026 power for the frozen slope test, one class and three classes
SD_RES = float(np.std(RES)); NCL = round(len(AN) / 3)
prng = np.random.default_rng(26)


def power_slope(beta, ncl, sims=3000):
    hits = 0
    for _ in range(sims):
        x_ = prng.standard_normal(ncl * NCL)
        y_ = beta * x_ + prng.normal(0, SD_RES, len(x_))
        r_ = stats.linregress(x_, y_)
        hits += (r_.slope / r_.stderr) > stats.t.ppf(0.90, len(x_) - 2)
    return hits / sims


B_OBS = float(HO["hoop_z"]["est"])
put("pws_obs_1", 100 * power_slope(B_OBS, 1), 0); put("pws_obs_3", 100 * power_slope(B_OBS, 3), 0)

# ============================================================================= 2026 board: day-3 Hoop G film flags, 2024-25
sc = dl[dl.hoop_z.notna() & (dl.rush_snaps >= RATE_MIN)].copy()
sc["fc"] = np.nan
for c in classes:
    trf, tef = fill_times(co30[co30.draft_year != c].copy(), sc[sc.draft_year == c])
    sc.loc[tef.index, "fc"] = ols(trf, tef, BASE)[0]
ba = dl[dl.hoop_z.notna()].copy()
ba["third"] = ba.groupby("draft_year").hoop_z.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
ba["short"] = (ba.pick >= DAY3) & (ba.third == 2)
brd = ba[ba.draft_year.isin([2024, 2025]) & ba.short].copy().merge(sc[["nfl_id", "fc"]], on="nfl_id", how="left")


def call(r_):
    if r_.rush_snaps < RATE_MIN:
        return "Audition"
    if r_.rush_snaps >= REG_MIN and r_.press_eb - r_.fc >= MARGIN:
        return "Go"
    if r_.rush_snaps >= REG_MIN and r_.press_eb - r_.fc <= -MARGIN:
        return "No-go"
    return "Hold"


def flip_rate(r_):
    """2026 pressure rate (%) that moves the shrunk career rate onto the frozen forecast at his 2025 volume."""
    R = r_.r25 if pd.notna(r_.r25) and r_.r25 > 0 else r_.rush_snaps
    return 100 * ((r_.fc / 100) * (r_.rush_snaps + R + Mw) - r_.pressures - Mw * mu) / R


brd["call"] = brd.apply(call, axis=1)
brd["latest_team"] = brd.latest_team.fillna("FA")
brd["over"] = brd.press_eb - brd.fc
brd["flip"] = [flip_rate(r_) if r_.rush_snaps >= RATE_MIN else np.nan for _, r_ in brd.iterrows()]
_pp_ = brd.press_eb / 100
brd["over_sd"] = 100 * np.sqrt(_pp_ * (1 - _pp_) / (brd.rush_snaps + Mw + 1))
order = {"Go": 0, "No-go": 1, "Hold": 2, "Audition": 3}
brd = brd.sort_values(["call", "over"], key=lambda c_: c_.map(order) if c_.name == "call" else -c_)
brd[["display_name", "draft_year", "pick", "latest_team", "role", "hoop_z", "rush_snaps", "press_eb", "fc", "over",
     "call", "flip"]].to_csv(OUT / "board.csv", index=False)


def flip_txt(r_):
    if r_.call == "Go" and r_.flip <= 0:
        return "cannot flip in one season"
    if r_.call == "Go":
        return f"flips below {r_.flip:.1f}%"
    if r_.call == "No-go":
        return f"flips above {r_.flip:.1f}%"
    if r_.rush_snaps < REG_MIN:
        return f"needs {REG_MIN - int(r_.rush_snaps)} more rushes"
    return f"on par at {r_.flip:.1f}%"


big = brd[brd.call != "Audition"]
AUD = brd[brd.call == "Audition"]
put("n_board", len(brd)); put("n_board_rows", len(big)); put("n_aud", len(AUD))
for cl_ in ("Go", "No-go", "Hold"):
    put(f"n_{cl_.lower().replace('-', '')}", int((brd.call == cl_).sum()))


def nm_team(r_):
    return f"{r_.display_name} ({r_.latest_team})"


GO, NOGO, HOLD = brd[brd.call == "Go"], brd[brd.call == "No-go"], brd[brd.call == "Hold"]
for i_, (_, r_) in enumerate(GO.iterrows()):
    put(f"go{i_}_rush", int(r_.rush_snaps)); put(f"go{i_}_pr", r_.press_eb, 1); put(f"go{i_}_fc", r_.fc, 1)
    put(f"go{i_}_flip", r_.flip, 1); put(f"go{i_}_hoop", r_.hoop_z, 1)
for i_, (_, r_) in enumerate(NOGO.iterrows()):
    put(f"ng{i_}_flip", r_.flip, 1)
put("nogo_flip_min", NOGO.flip.min(), 1)


def join(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


TXT = {"go": join(nm_team(r_) for _, r_ in GO.iterrows()), "nogo": join(nm_team(r_) for _, r_ in NOGO.iterrows()),
       "hold": join(nm_team(r_) for _, r_ in HOLD.iterrows()),
       "aud": ", ".join(f"{r_.display_name} ({int(r_.draft_year)}, "
                        f"{'UDFA' if r_.pick >= UDFA_PICK else int(r_.pick)}, {r_.latest_team})"
                        for _, r_ in AUD.iterrows()),
       "go_names": [r_.display_name for _, r_ in GO.iterrows()],
       "go_teams": list(GO.latest_team), "go_roles": list(GO.role)}

frozen = {"metric": "Hoop G: hardest 0.5 s of sideways acceleration on Run-the-Hoop, z within Combine year",
          "use": f"film flag for day-3/undrafted (pick>={DAY3}) rushers in the class top third of Hoop G; never moves "
                 "the board order on its own",
          "test": "2026 Combine class after its first NFL season: slope of Hoop G on (pressure rate - frozen "
                  "no-tracking forecast), within-class permutation, one-sided p<0.10; slope <= 0 retires the flag",
          "calls_2026_board": {r_.display_name: r_.call for _, r_ in brd.iterrows()},
          "decision_date": "after Week 18 of the 2026 regular season"}
blob = json.dumps(frozen, sort_keys=True).encode()
HASH = hashlib.sha256(blob).hexdigest()[:12]
(OUT / "frozen_rule.json").write_text(json.dumps({**frozen, "sha256": hashlib.sha256(blob).hexdigest()}, indent=1))
TXT["hash"] = HASH

# ============================================================================= figures
C_H, C_B, C_N = OI["orange"], OI["blue"], OI["grey"]
MS = MARGIN * PS

# Fig 1: the screen as a test. Six pre-listed measures, held-out slope + 90% CI, family-wise chance line.
fig, ax = plt.subplots(figsize=(FIGW, 6.0))
ordr = list(range(len(FAMR)))[::-1]
for yy, j in enumerate(ordr):
    f = FAMR[j]
    col = C_H if f["key"] == "hoop_z" else (C_B if f["key"] == "burst_z" else "#555555")
    lw = 6 if f["key"] in ("hoop_z", "burst_z") else 3.5
    ax.plot([f["lo"] * PS, f["hi"] * PS], [yy, yy], color=col, lw=lw, solid_capstyle="butt", alpha=0.9)
    for c, v in zip(classes, f["cls"]):
        ax.plot(v * PS, yy - 0.28, marker="D", color="white", mec="#333", ms=6.5)
    ax.plot(f["est"] * PS, yy, "o", color="white", mec=col, mew=3, ms=12, zorder=5)
    wt = "bold" if f["key"] == "hoop_z" else "normal"
    for xx, txt in [(1.04, f"{f['est'] * PS:+.1f}"), (1.17, f"{f['p2']:.2f}"), (1.34, f"{f['pfw']:.2f}")]:
        ax.text(xx, yy, ("0.0" if txt == "-0.0" else txt).replace("-", "\u2212"), va="center", ha="left", fontsize=12, fontweight=wt,
                transform=ax.get_yaxis_transform())
for xx, txt in [(1.04, "per sd"), (1.17, "p alone"), (1.34, "best-of-6 p")]:
    ax.text(xx, len(FAMR) - 0.45, txt, ha="left", fontsize=11, color="#444", transform=ax.get_yaxis_transform())
ax.axvspan(-MS, MS, color=C_N, alpha=0.22, lw=0)
for sx in (-FW95, FW95):
    ax.axvline(sx, color="#333333", lw=1.6, ls=(0, (2, 2)))
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(FAMR)))
ax.set_yticklabels([FAMR[j]["lab"] for j in ordr])
for t_ in ax.get_yticklabels():
    if "Hoop G" in t_.get_text():
        t_.set_fontweight("bold")
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-0.7, len(FAMR) - 0.3)
ax.set_xlabel("Extra pressures per 450-rush season per sd,\nbeyond the no-tracking forecast (held-out class)")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
hd = [Patch(color=C_N, alpha=0.4, label=f"\u00b1{MS:.2f} action margin"),
      Line2D([], [], color="#333333", lw=1.6, ls=(0, (2, 2)), label=f"chance size of best of {len(FAMR)} (\u00b1{FW95:.1f})"),
      Line2D([], [], marker="D", color="white", mec="#333", ms=6.5, lw=0, label="one held-out class")]
ax.legend(handles=hd, loc="upper left", bbox_to_anchor=(-0.55, -0.27), frameon=False, fontsize=11, ncol=3)
fig.suptitle(f"Six drill measures tested together (n = {len(AN)}): Hoop G leads, but stays inside the range "
             "six\nshuffled drills reach by chance. Burst out of a stance is flat.", x=0.01, ha="left", fontsize=13)
fig.subplots_adjust(left=0.30, right=0.66, top=0.86, bottom=0.24)
save(fig, "fig2_screen.png")

# Fig 1 (hero): the same lineman in the drill and on NFL snaps, one yard scale, one colour scale.
SPD = att.set_index("event_id").speed_at_bend
BI = ba.assign(latest_team=ba.latest_team.fillna("FA")).set_index("nfl_id")
SCI = sc.set_index("nfl_id")
_pg = dl.dropna(subset=["g_bend"])
BENDZ = ((_pg.g_bend - _pg.g_bend.mean()) / _pg.g_bend.std()).set_axis(_pg.nfl_id)
_cand = AN[(AN.hoop_z <= -0.5) & (AN.g_n >= 10)].assign(ov=lambda d_: d_.press_eb - d_.fc)
CTR_ID = int(_cand.sort_values("ov").iloc[-1].nfl_id)
HERO = [("Go", int(BI.index[BI.display_name == GO.display_name.iloc[1]][0])),
        ("Go", int(BI.index[BI.display_name == GO.display_name.iloc[0]][0])),
        ("No-go", int(BI.index[BI.display_name == NOGO.sort_values("hoop_z").display_name.iloc[-1]][0])),
        ("Low Hoop G", CTR_ID)]
HERO_IDS = [p_ for _, p_ in HERO]


def nfl_traces(ids):
    gf_ = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "x", "y"],
                          filters=[("nfl_id", "in", ids)])
    meta = g[(g.sample_kind == "random") & g.nfl_id.isin(ids)][K + ["snap_frame", "play_direction"]]
    gf_ = gf_.merge(meta, on=K).sort_values(K + ["frame"])
    pr_ = GF.set_index(K).press
    out_ = {i: [] for i in ids}
    for key, h in gf_.groupby(K, sort=False):
        sf = h.snap_frame.iloc[0]
        w = h[(h.frame >= sf) & (h.frame <= sf + 20)]
        if len(w) != 21 or key not in pr_.index or key not in GF.set_index(K).dropna(subset=["g_bend"]).index:
            continue
        sgn = -1 if h.play_direction.iloc[0] == "right" else 1
        up = sgn * (w.x.values - w.x.values[0])
        lat = sgn * (w.y.values - w.y.values[0])
        alat, roll, *_ = lateral_g(w.x.values, w.y.values)
        out_[key[2]].append(dict(lat=lat, up=up, alat=alat, k=int(np.nanargmax(roll)), bend=float(np.nanmax(roll)),
                                 press=int(pr_.loc[key])))
    return out_


NT = nfl_traces(HERO_IDS)
fig = plt.figure(figsize=(FIGW, 9.0))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 0.85, 0.3], left=0.02, right=0.9, top=0.84, bottom=0.07, wspace=0.08,
                      hspace=0.30)
norm = plt.Normalize(0, 1.0)
CM = "cividis"
HALF = 5.6
xov = []
for j, (tag, pid) in enumerate(HERO):
    r_ = BI.loc[pid]
    tr_ = traces[r_.hoop_eid]
    ax = fig.add_subplot(gs[0, j])
    x_, y_ = tr_["_x"] - np.mean(tr_["_x"]), tr_["_y"] - np.mean(tr_["_y"])
    pts = np.c_[x_, y_].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=CM, norm=norm, lw=3.2)
    lc.set_array(tr_["_alat"][:-1]); ax.add_collection(lc)
    k_ = tr_["_k"]
    ax.plot(x_[max(k_ - 2, 0):k_ + 3], y_[max(k_ - 2, 0):k_ + 3], color=OI["red"], lw=8, alpha=0.9,
            solid_capstyle="round")
    ax.plot(x_[0], y_[0], "o", color="black", ms=5)
    ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF); ax.set_aspect("equal"); ax.axis("off")
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
    ax.set_title(f"{r_.display_name}\n{r_.latest_team}, {int(r_.draft_year)} {pk}  [{tag}]", fontsize=12.5,
                 fontweight="bold")
    ax.text(0.5, -0.04, f"Drill: Hoop G {fm(r_.hoop_z)} sd ({r_.hoop_g:.2f} g)", transform=ax.transAxes,
            ha="center", va="top", fontsize=11.5)
    ax2 = fig.add_subplot(gs[1, j])
    rs = NT[pid]
    best = int(np.argmax([t["bend"] for t in rs]))
    for i_, t in enumerate(rs):
        ax2.plot(t["lat"], t["up"], color="#B8B8B8", lw=1.1, zorder=1)
        ax2.plot(t["lat"][-1], t["up"][-1], "o", ms=5.5, color=OI["red"] if t["press"] else "#B8B8B8",
                 mec="white", mew=0.5, zorder=3)
    t = rs[best]
    pts = np.c_[t["lat"], t["up"]].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=CM, norm=norm, lw=3.2, zorder=2)
    lc.set_array(t["alat"][:-1]); ax2.add_collection(lc)
    ax2.plot(0, 0, "o", color="black", ms=5, zorder=4)
    ax2.set_xlim(-HALF, HALF); ax2.set_ylim(-0.8, 8.8); ax2.set_aspect("equal"); ax2.axis("off")
    npress = sum(t_["press"] for t_ in rs)
    ax2.text(0.5, -0.04, f"NFL bend {fm(BENDZ.loc[pid])} sd\n{npress} of {len(rs)} pressured",
             transform=ax2.transAxes, ha="center", va="top", fontsize=11.5)
    if j == 0:
        ax.plot([-HALF + 0.2, -HALF + 2.2], [-HALF + 0.3] * 2, color="black", lw=2)
        ax.text(-HALF + 1.2, -HALF + 0.6, "2 yd", ha="center", va="bottom", fontsize=10)
        ax2.annotate("upfield", xy=(-HALF + 0.6, 7.5), xytext=(-HALF + 0.6, 4.5), fontsize=10, ha="center",
                     arrowprops=dict(arrowstyle="->", color="black"))
    s_ = SCI.loc[pid]
    pp_ = s_.press_eb / 100
    sd_ = 100 * np.sqrt(pp_ * (1 - pp_) / (s_.rush_snaps + Mw + 1)) * PS
    xov.append(((s_.press_eb - s_.fc) * PS, sd_, int(s_.rush_snaps)))
# bottom strip: one shared axis for all four
axs_ = fig.add_subplot(gs[2, :])
_ps = axs_.get_position()
axs_.set_position([0.22, _ps.y0, 0.58, _ps.height])
for j, (ov, sd_, nr) in enumerate(xov):
    c_ = [OI["blue"], OI["blue"], OI["red"], OI["purple"]][j]
    axs_.plot([ov - Z90 * sd_, ov + Z90 * sd_], [-j, -j], color=c_, lw=4, alpha=0.6)
    axs_.plot(ov, -j, "o", color=c_, mec="black", ms=10)
    axs_.text(1.01, -j, f"{fm(ov)} ({nr} rushes)", va="center", fontsize=10.5, transform=axs_.get_yaxis_transform())
axs_.axvspan(-MARGIN * PS, MARGIN * PS, color=C_N, alpha=0.22, lw=0)
axs_.axvline(0, color="black", lw=0.8)
axs_.set_yticks([-j for j in range(len(HERO))])
axs_.set_yticklabels([BI.loc[p_].display_name for p_ in HERO_IDS], fontsize=10.5)
axs_.set_xlim(-20, 33); axs_.set_ylim(-3.6, 0.6)
axs_.set_xlabel("Career pressures per 450-rush season above the no-tracking forecast (dot, 90% interval)",
                fontsize=11.5)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CM), cax=fig.add_axes([0.92, 0.36, 0.013, 0.46]))
cb.set_label("Sideways acceleration (g)", fontsize=11.5)
fig.text(0.01, 0.975, "Same lineman, drill lap (top) and sampled NFL rushes (middle), first 2 s after the snap, "
         "on one yard scale", fontsize=13.5, fontweight="bold", va="top")
fig.text(0.01, 0.94, "Vermillion bar = the half-second that defines Hoop G. Middle: grey = every randomly sampled "
         "rush; coloured = his hardest-bending rush;\nvermillion dot = pressured snap. Hoop G ranks the drill lap; "
         "it does not show up reliably in the NFL bend underneath it.", fontsize=11.5, va="top")
save(fig, "fig1_hero.png")
for j, (tag, pid) in enumerate(HERO):
    r_ = BI.loc[pid]
    put(f"h{j}_hoopz", r_.hoop_z, 1); put(f"h{j}_bendz", BENDZ.loc[pid], 1); put(f"h{j}_over", xov[j][0], 1)
    put(f"h{j}_n", len(NT[pid])); put(f"h{j}_np", int(sum(t_["press"] for t_ in NT[pid])))
    TXT[f"h{j}"] = r_.display_name
    TXT[f"h{j}_team"] = r_.latest_team

# Fig 3: day-3 decision, rusher level. A: rank correlation with NFL pressure. B: top-third hit rate.
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.0), sharey=True)
COLS = {"neg_pick": OI["black"], "fc": OI["purple"], "fc_hoop": C_H, "hoop_z": OI["red"], "burst_z": C_B}
MKS = {"neg_pick": "s", "fc": "s", "fc_hoop": "o", "hoop_z": "o", "burst_z": "o"}
for yy, d in enumerate(DEC[::-1]):
    c_ = COLS[d["col"]]
    axs[0].plot([d["lo"], d["hi"]], [yy, yy], color=c_, lw=5, solid_capstyle="butt")
    axs[0].plot(d["r"], yy, MKS[d["col"]], color="white", mec=c_, mew=3, ms=12, zorder=5)
    axs[0].text(d["r"], yy + 0.3, f"{d['r']:+.2f}".replace("-", "−"), ha="center", fontsize=11)
    axs[1].plot([100 * d["wl"], 100 * d["wh"]], [yy, yy], color=c_, lw=5, solid_capstyle="butt")
    axs[1].plot(100 * d["k"] / d["n"], yy, MKS[d["col"]], color="white", mec=c_, mew=3, ms=12, zorder=5)
    axs[1].text(100 * d["k"] / d["n"], yy + 0.3, f"{d['k']} of {d['n']}", ha="center", fontsize=11)
axs[0].axvline(0, color="#333333", lw=1, ls="--")
axs[1].axvline(50, color="black", lw=1, ls="--")
axs[1].text(51, -0.45, "coin flip", fontsize=10.5)
axs[1].text(0.5, 0.99, f"Forecast + Hoop G minus pick order: {NUM['hg_gap'].replace('-', '−')} pts\n[90%: {NUM['hg_gap_lo'].replace('-', '−')}, "
            f"{NUM['hg_gap_hi']}]; bar for board use: +{BAR_GAP}", transform=axs[1].transAxes, ha="center", va="top",
            fontsize=10.5, bbox=dict(fc="white", ec="#333333", lw=0.8))
axs[0].set_yticks(range(len(DEC))); axs[0].set_yticklabels([d["lab"] for d in DEC[::-1]])
axs[0].set_xlabel("Rank correlation with NFL pressure rate\n(within class, 90% interval)", fontsize=12)
axs[1].set_xlabel("% of each ranker's top third who out-rushed\nthe class's day-3 median (90% interval)", fontsize=12)
axs[0].set_xlim(-0.75, 0.9); axs[1].set_xlim(0, 100)
axs[0].set_ylim(-0.6, len(DEC) + 0.55)
axs[0].set_title("A. Ordering the whole day-3 group", loc="left", fontsize=12.5)
axs[1].set_title("B. Picking the top third", loc="left", fontsize=12.5)
fig.suptitle(f"Day-3 and undrafted rushers, scored on the class left out ({len(D3)} rushers, each counted once): "
             "Hoop G\norders them better than draft order (A), but its top third hits no more often (B)", fontsize=12.5,
             x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.subplots_adjust(top=0.80)
save(fig, "fig3_day3.png")

# Fig 4: 2026 player cards for the Go / No-go names, with sampled-rush splits (cell n) and what flips each call.
_meta = g[K + ["lined_up_position", "down"]]
SPL = GF[GF.kind == "random"].merge(_meta, on=K, how="left")
SPL["wide"] = SPL.lined_up_position.isin(["EDGE", "OLB"])


def split_txt(pid):
    d_ = SPL[SPL.nfl_id == pid]
    w_, i_, t_ = d_[d_.wide], d_[~d_.wide], d_[d_.down == 3]
    return (f"wide {int(w_.press.sum())}/{len(w_)}, inside {int(i_.press.sum())}/{len(i_)}, "
            f"3rd down {int(t_.press.sum())}/{len(t_)}"), len(d_)


def action(r_):
    if r_.call == "Go" and r_.role == "interior":
        return "Passing-down looper on\nE-T stunts; keep 2026 rate\nabove the flip line"
    if r_.call == "Go" and r_.flip <= 0:
        return "Third-down snaps from a\nwide-9 alignment; contract\ndecision can use 2025 tape"
    if r_.call == "Go":
        return "Expand third-down wide\nsnaps; re-score after Wk 18"
    return "Run-down rotation only; he\nmust beat the flip rate\nto earn passing downs"


CARDS = pd.concat([GO, NOGO])
fig, axs = plt.subplots(2, 4, figsize=(FIGW, 7.6))
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.03, wspace=0.06, hspace=0.12)
XL = 30
for ax, (_, r_) in zip(axs.flat, CARDS.iterrows()):
    c_ = OI["blue"] if r_.call == "Go" else OI["red"]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(c_); sp.set_linewidth(2)
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
    ax.text(0.04, 0.95, f"{r_.display_name}", fontsize=12.5, fontweight="bold", va="top")
    ax.text(0.04, 0.86, f"{r_.latest_team} · {int(r_.draft_year)} {pk} · {r_.role} · {r_.call.upper()}", fontsize=10.5,
            va="top", color=c_, fontweight="bold")
    sp_, ns_ = split_txt(r_.nfl_id)
    lines = [f"Hoop G {fm(r_.hoop_z)} sd (class top third)",
             f"{int(r_.rush_snaps)} NFL rushes: {r_.press_eb:.1f}% vs {r_.fc:.1f}% fcst",
             f"{flip_txt(r_).replace('cannot flip in one season', 'cannot flip in 2026').capitalize()}",
             f"Sampled rushes pressured (n={ns_}):", f"  {sp_}"]
    for i_, ln in enumerate(lines):
        ax.text(0.04, 0.74 - 0.085 * i_, ln, fontsize=9.6 if i_ == 4 else 10, va="top")
    ax.text(0.04, 0.29, action(r_), fontsize=10, va="top", style="italic")
    ia = ax.inset_axes([0.06, 0.03, 0.88, 0.07])
    ov, sd_ = r_.over * PS, r_.over_sd * PS
    ia.axvspan(-MARGIN * PS, MARGIN * PS, color=C_N, alpha=0.3, lw=0)
    ia.axvline(0, color="black", lw=0.8)
    ia.plot([ov - Z90 * sd_, ov + Z90 * sd_], [0, 0], color=c_, lw=3, alpha=0.6)
    ia.plot(ov, 0, "o", color=c_, mec="black", ms=7)
    ia.set_xlim(-XL, XL); ia.set_yticks([]); ia.tick_params(labelsize=8.5, length=2)
    for s_ in ("left", "right", "top"):
        ia.spines[s_].set_visible(False)
for ax in axs.flat[len(CARDS):]:
    ax.axis("off")
fig.text(0.01, 0.975, f"2026 calls for the {len(CARDS)} day-3 rushers Hoop G flagged who have 100+ NFL rushes "
         "(2024–25 classes)", fontsize=13.5, fontweight="bold", va="top")
fig.text(0.01, 0.935, "Strip = career pressures per 450 rushes above the frozen no-tracking forecast, 90% binomial "
         "interval; grey = action margin.\nSplits are randomly sampled NFL rushes (k pressured / n), wide = EDGE/OLB "
         "alignment. Calls come from NFL pressure, not Hoop G.", fontsize=10.5, va="top")
save(fig, "fig4_cards.png")
put("cards_min_sampled", int(min(split_txt(r_.nfl_id)[1] for _, r_ in CARDS.iterrows())))
put("cards_max_sampled", int(max(split_txt(r_.nfl_id)[1] for _, r_ in CARDS.iterrows())))

# ============================================================================= multiplicity and independent check
TESTS = (["family: " + f[0] for f in FAM]
         + ["chain: burst->get-off", "chain: burst->NFL burst", "chain: hoop->NFL bend", "chain: hoop->get-off",
            "bench: 3-cone->NFL bend", "bench: split->get-off", "bench: 3-cone->get-off",
            "trait->pressure: get-off", "trait->pressure: burst", "trait->pressure: bend",
            "R2 gain: hoop", "R2 gain: burst", "burst equivalence",
            "day3 rho: fc+hoop vs fc", "day3 rho: fc+hoop vs pick", "day3 rho: hoop vs pick",
            "day3 top-third gap", "day3 flag Fisher"])
put("n_tests", len(TESTS))
import subprocess
import sys
subprocess.run([sys.executable, str(OUT / "verify.py")], check=True, capture_output=True)
VJ = json.loads((OUT / "verify.json").read_text())
put("v_n", VJ["v_n"]); put("v_slope", VJ["v_slope"], 1); put("v_lo", VJ["v_lo"], 1); put("v_hi", VJ["v_hi"], 1)
put("v_neg_classes", int(sum(v <= 0 for v in VJ["v_cls"])))

# ============================================================================= render
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
(OUT / "text_bits.json").write_text(json.dumps(TXT, indent=1))
tpl = OUT / "writeup_template.md"
if tpl.exists():
    t = tpl.read_text()

    def sub(m_):
        k_ = m_.group(1)
        if k_.startswith("T:"):
            v_ = TXT[k_[2:].split("[")[0]]
            if "[" in k_:
                v_ = v_[int(k_.split("[")[1].rstrip("]"))]
            return str(v_)
        if k_ not in NUM:
            raise KeyError(k_)
        v_ = str(NUM[k_])
        return v_.replace("-", "−") if v_.startswith("-") else v_
    (OUT / "writeup.md").write_text(re.sub(r"\{\{([^}]+)\}\}", sub, t))
print(json.dumps(NUM, indent=0))
print(json.dumps(TXT, indent=0))
