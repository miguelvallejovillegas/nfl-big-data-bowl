"""Hoop G (Run-the-Hoop bend) vs NFL pass-rush pressure for DL/EDGE: one metric, tested honestly.

Run from the repo root:
    uv run python writeup/candidates/round_06/revised/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (only the values quoted in writeup.md), tables.json,
board_2023_2025.csv and figures/*.png next to this file, then renders writeup.md from writeup_template.md.
Frame-level feature extraction is cached in data/cache/ (gitignored; delete it to recompute from the frames).
Runtime: about 2 minutes cached, about 2.5 minutes cold, one core.

Design (round 6 revised, from round 6 cand_1):
  * Baseline without tracking = log pick + official 10-yd split + 3-cone + NGS athleticism score + weight + NFL EDGE
    share (+ games per season and a missing-time flag in the count model).
  * Headline: Bayesian beta-binomial model of pressures out of pass rushes for every hoop runner with an NFL rush,
    class random intercept, separate EDGE / interior bend slopes, bend measurement error (one-run reliability)
    integrated by Gauss-Hermite quadrature, bend-slope prior scaled to the typical effect of 14 screened pairs.
    Adaptive Metropolis, 4 chains. Missing times: deterministic fill from weight, role and NGS + flag.
  * Sensitivities use the same likelihood and priors with a Laplace approximation (checked against the MCMC):
    complete cases (observed 3-cone; observed 10-yd), multiple imputation (20 stochastic draws, posterior draws
    pooled), wide prior, no-times baseline, leave-one-player-out, and cross-fitted per-class sign consistency.
  * Out of class: fully nested leave-one-class-out loop (impute, screen 14 pairs, pick best, reliability, shrink,
    all inside each fold), class-stratified player bootstrap of the whole loop.
  * Placebo outcome (run-stop proxy) and negative-control drill (back-pedal); path-direction check on NFL snaps.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import textwrap
import time as _time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import optimize, stats
from scipy.signal import savgol_filter
from scipy.special import expit, gammaln
from scipy.special import logit as _logit

warnings.filterwarnings("ignore")


def render(out_dir: Path, num: dict | None = None, md_tables: dict | None = None):
    """Fill writeup_template.md: {{TABLE:name}} -> generated markdown table, {{key}} -> number (fails on unknown keys).
    Returns the set of number keys the text uses (numbers.json keeps exactly these)."""
    tpl = out_dir / "writeup_template.md"
    if not tpl.exists():
        return set()
    num = num if num is not None else json.load(open(out_dir / "numbers.json"))
    md_tables = md_tables if md_tables is not None else json.load(open(out_dir / "tables.json")).get("md", {})
    text = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda m: md_tables[m.group(1)], tpl.read_text())
    used = {k for k in (num or {}) if k.startswith("tb_")}

    def sub(m):
        k = m.group(1)
        if k not in num:
            raise KeyError(f"writeup_template.md uses unknown key {k}")
        used.add(k)
        v = str(num[k])
        return "−" + v[1:] if v.startswith("-") else v
    (out_dir / "writeup.md").write_text(re.sub(r"\{\{(\w+)\}\}", sub, text))
    return used


if "--render-only" in sys.argv:
    render(Path(__file__).resolve().parent)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parents[4]
PROC = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
for old in FIG.glob("*.png"):
    old.unlink()
CACHE = Path(os.environ.get("BDB_CACHE", ROOT / "data" / "cache" / "hoop_r5"))
CACHE.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(2027)
NBOOT = int(os.environ.get("BDB_NBOOT", 1000))
NLOCO = int(os.environ.get("BDB_NLOCO", 200))
NMCMC = int(os.environ.get("BDB_NMCMC", 16000))
MIN_SNAPS = 100
RATE_MIN = 30                 # board shows a pressure rate from 30 rushes on
UDFA_PICK = 262
DT = 0.1
YD_PER_G = 9.80665 / 0.9144
ON, PRE = 1.0, 0.5
W1 = 10
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
FIGW = 1600 / 150
_T0 = _time.time()


def _tick(s):
    print(f"[{_time.time() - _T0:7.1f}s] {s}", flush=True)


NUM: dict = {}
TABLES: dict = {}


def put(key, v, nd=2):
    """Store a number exactly as written in the text."""
    if key in NUM:
        raise KeyError(f"duplicate numbers key {key}")
    if isinstance(v, (int, np.integer)):
        NUM[key] = int(v)
        return int(v)
    s = f"{float(v):.{nd}f}"
    if s in ("-0.00", "-0.000", "-0.0"):
        s = s[1:]
    NUM[key] = s
    return s


def mn(s):
    s = str(s)
    return "−" + s[1:] if s.startswith("-") else s


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


def yz(s: pd.Series, year: pd.Series) -> pd.Series:
    return s.groupby(year).transform(lambda v: (v - v.mean()) / v.std())


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
    alat = np.abs(vx * ay - vy * ax) / np.maximum(v, 0.5)
    roll = pd.Series(alat).rolling(5, center=True).mean().to_numpy()
    k = int(np.nanargmax(roll))
    o = onset(s)
    d1 = np.nan
    if o is not None and s[0] < ON and o + W1 < n:
        d1 = float(np.hypot(x[o + W1] - x[o], y[o + W1] - y[o]))
    disp = np.hypot(x - x[0], y - y[0])
    t3 = np.argmax(disp >= 3) * DT if disp.max() >= 3 else np.nan
    half = n // 2
    return {"bend_g": np.nanmax(roll) / YD_PER_G, "bend_g_h1": np.nanmax(roll[:half]) / YD_PER_G,
            "bend_g_h2": np.nanmax(roll[half:]) / YD_PER_G, "speed_at_bend": v[k], "d1": d1, "t3": t3,
            "_alat": alat / YD_PER_G, "_x": xs, "_y": ys, "_k": k, "_o": o}


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


def sp_ci(x, y):
    m = pd.notna(x) & pd.notna(y)
    x, y = np.asarray(x, float)[m], np.asarray(y, float)[m]
    n = len(x)
    r, p = stats.spearmanr(x, y)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    b = []
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        b.append(stats.spearmanr(x[i], y[i])[0])
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return float(r), float(lo), float(hi), n, float(p)


def _pr(rx, ry, A):
    bx = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    by = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(bx, by)[0, 1])


def partial_sp(df, x, y, Z):
    """Spearman partial correlation: Pearson r of baseline-residualized ranks; bootstrap over players."""
    d = df[[x, y] + Z].dropna()
    R = d[[x, y] + Z].rank()
    A = np.c_[np.ones(len(R)), R[Z].values]
    r = _pr(R[x].values, R[y].values, A)
    n = len(d)
    b = []
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        b.append(_pr(R[x].values[i], R[y].values[i], A[i]))
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return r, float(lo), float(hi), n


def mde(n, k=0, power=0.8, alpha=0.05):
    return float(np.tanh((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(n - 3 - k)))


def power_at(r, n, k=0, alpha=0.05):
    se = 1 / np.sqrt(n - 3 - k)
    return float(stats.norm.sf(stats.norm.ppf(1 - alpha) - np.arctanh(r) / se))


# ============================================================================= load
_tick("load")
players = pd.read_parquet(PROC / "players.parquet")
cres = pd.read_parquet(PROC / "combine_results.parquet")
events = pd.read_parquet(PROC / "combine_events.parquet")
seasons = pd.read_parquet(PROC / "player_season_stats.parquet")
align = pd.read_parquet(PROC / "player_season_alignment.parquet")
gp = pd.read_parquet(PROC / "game_plays_sample.parquet")

dl = players[players.position_group == "DL"][["nfl_id", "display_name", "draft_year", "nfl_position",
                                               "draft_overall_pick"]].copy()
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
dl["edge"] = (dl.edge_share >= 0.5).astype(float)
dl["role"] = np.where(dl.edge == 1, "EDGE", "interior")

# ============================================================================= Combine drills
_tick("Combine drills")
ev = events[(events.position_group == "DL") & events.drill_name.isin(["PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL"])]
ev = ev[["event_id", "nfl_id", "draft_year", "drill_name", "attempt_no", "start_speed", "peak_accel",
         "mean_speed", "duration_s"]]


def drill_features():
    fr = pd.read_parquet(PROC / "combine_frames.parquet", filters=[("event_id", "in", list(ev.event_id))],
                         columns=["event_id", "frame", "entity_type", "x", "y", "s"])
    fr = fr[fr.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
    rows, traces = [], {}
    for eid, g in fr.groupby("event_id", sort=False):
        f = attempt_features(g)
        if f is None:
            continue
        traces[eid] = {k: f.pop(k) for k in ["_alat", "_x", "_y", "_k", "_o"]}
        rows.append({"event_id": eid, **f})
    return rows, traces


rows, traces = cached("drill", drill_features)
att = pd.DataFrame(rows).merge(ev, on="event_id")
hoop_att = att[att.drill_name == "RUN_THE_HOOP_DRILL"].copy()
rush_att = att[att.drill_name == "PASS_RUSH_DRILL"].copy()
hoop_att["hoop_z"] = yz(hoop_att.bend_g, hoop_att.draft_year)
rush_att["d1_z"] = yz(rush_att.d1, rush_att.draft_year)
put("hoop_g_median", hoop_att.bend_g.median())

r1_d1, _ = icc(rush_att.nfl_id, rush_att.d1_z)
k_rush = rush_att.dropna(subset=["d1"]).groupby("nfl_id").size().mean()
rel_d1_player = sb(r1_d1, k_rush)
put("rel_drill_d1_player", rel_d1_player)
h1, h2 = hoop_att.bend_g_h1.to_numpy(), hoop_att.bend_g_h2.to_numpy()
r_sh = stats.spearmanr(h1, h2)[0]
rel_hoop = 2 * r_sh / (1 + r_sh)
put("rel_hoop_sb", rel_hoop)
bs_sh = []
for _ in range(NBOOT):
    i = RNG.integers(0, len(h1), len(h1))
    rr = stats.spearmanr(h1[i], h2[i])[0]
    bs_sh.append(2 * rr / (1 + rr))
put("rel_hoop_sb_lo", np.percentile(bs_sh, 2.5)); put("rel_hoop_sb_hi", np.percentile(bs_sh, 97.5))
put("rel_hoop_3runs", sb(rel_hoop, 3)); put("rel_hoop_4runs", sb(rel_hoop, 4))

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
SCREEN = [("hoop_z", "Hoop G (bend)"), ("hoop_speed_z", "Hoop mean speed"), ("hoop_time_z", "Hoop lap time"),
          ("rush_bend_z", "Rush-drill turn g"), ("rush_acc_z", "Rush-drill peak accel"),
          ("rush_t3_z", "Rush-drill time to 3 yd"), ("d1_z", "Rush-drill first second")]
hstat = hoop.groupby("draft_year").hoop_g.agg(["mean", "std"])

# ============================================================================= NFL outcomes
_tick("NFL outcomes")
s = seasons[seasons.position_group == "DL"].copy()
s["w_go"] = s.pass_rush_snaps * s.avg_get_off
agg = s.groupby("nfl_id").agg(rush_snaps=("pass_rush_snaps", "sum"), pressures=("pressures", "sum"),
                              w_go=("w_go", "sum"), games=("games", "sum")).reset_index()
agg["get_off"] = agg.w_go / agg.rush_snaps
agg["press_rate"] = agg.pressures / agg.rush_snaps
dl = dl.merge(agg, on="nfl_id", how="left")
dl["rush_snaps"] = dl.rush_snaps.fillna(0)
dl["games"] = dl.games.fillna(0)
dl["seasons_avail"] = 2026 - dl.draft_year
dl["games_per_season"] = dl.games / dl.seasons_avail
m = dl.rush_snaps >= 20
n_, k_ = dl.loc[m, "rush_snaps"], dl.loc[m, "pressures"]
p_ = k_ / n_
mu = k_.sum() / n_.sum()
N_ = n_.sum()
tau2 = ((n_ * (p_ - mu) ** 2).sum() - (len(n_) - 1) * mu * (1 - mu)) / (N_ - (n_ ** 2).sum() / N_)
Mw = mu * (1 - mu) / tau2 - 1
mm = dl.rush_snaps > 0
dl.loc[mm, "press_eb"] = (dl.loc[mm, "pressures"] + Mw * mu) / (dl.loc[mm, "rush_snaps"] + Mw)

# ============================================================================= game tracking (sample)
_tick("game tracking")
dlids = set(dl.nfl_id)
g = gp[gp.nfl_id.isin(dlids)]
K = ["game_id", "play_id", "nfl_id"]
gf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "x", "y"])
gf = gf[gf.nfl_id.isin(dlids)].merge(g[K + ["snap_frame"]], on=K)
gf["rel"] = gf.frame - gf.snap_frame
p0 = gf[gf.rel == 0].set_index(K)[["x", "y"]]
p1 = gf[gf.rel == W1].set_index(K)[["x", "y"]]
gd = np.hypot(p1.x - p0.x.reindex(p1.index), p1.y - p0.y.reindex(p1.index)).rename("g_d1").reset_index()
gd = gd.merge(g[K + ["sample_kind"]], on=K).dropna(subset=["g_d1"])
gpl = gd[gd.sample_kind == "random"].groupby("nfl_id").g_d1.mean().rename("g_d1_random").reset_index()
dl = dl.merge(gpl, on="nfl_id", how="left")

gfe = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "event"])
gfe = gfe[gfe.nfl_id.isin(dlids) & gfe.event.isin(["pass_forward", "qb_sack", "qb_strip_sack", "run", "handoff"])]
ends = gfe.groupby(K).frame.min().rename("end_frame")
ARC_CAP, VMIN = 35, 2.0
FIELD_MID, OFF_MIN = 53.3 / 2, 4.0
gfa = gf.sort_values(K + ["frame"]).join(ends, on=K)
pinfo = g.drop_duplicates(K).set_index(K)[["play_direction", "sample_kind", "time_to_pressure", "sack", "down",
                                           "yards_to_go", "lined_up_position"]]


def arc_features():
    arc_rows, arc_tr = [], {}
    for key, h in gfa.groupby(K, sort=False):
        sf = h.snap_frame.iloc[0]
        e = h.end_frame.iloc[0]
        e = sf + ARC_CAP if pd.isna(e) else min(e, sf + ARC_CAP)
        h = h[(h.frame >= sf - 3) & (h.frame <= e + 3)]
        if len(h) < 15 or e - sf < 8:
            continue
        xs, ys = savgol_filter(h.x.values, 7, 2), savgol_filter(h.y.values, 7, 2)
        vx, vy = savgol_filter(xs, 7, 2, deriv=1, delta=DT), savgol_filter(ys, 7, 2, deriv=1, delta=DT)
        ax_, ay_ = savgol_filter(xs, 7, 2, deriv=2, delta=DT), savgol_filter(ys, 7, 2, deriv=2, delta=DT)
        v = np.hypot(vx, vy)
        al = np.abs(vx * ay_ - vy * ax_) / np.maximum(v, 0.5)
        al[v < VMIN] = 0.0
        inw = (h.frame.values >= sf) & (h.frame.values <= e)
        roll = pd.Series(np.where(inw, al, np.nan)).rolling(5, center=True, min_periods=5).mean()
        if roll.notna().sum() == 0:
            continue
        info = pinfo.loc[key]
        fwd = -(xs - xs[0]) if info.play_direction == "right" else (xs - xs[0])
        ii = np.flatnonzero(inw)
        f0 = ii[0]
        off = abs(ys[f0] - FIELD_MID)
        lat = (ys - ys[f0]) * (1.0 if ys[f0] >= FIELD_MID else -1.0)
        fwd = fwd - fwd[f0]
        fw_, la_ = fwd[ii], lat[ii]
        i3 = np.flatnonzero(fw_ >= 3.0)
        l3 = float(la_[i3[0]]) if len(i3) else np.nan
        wide = info.lined_up_position in ("EDGE", "OLB")
        if not (wide and off >= OFF_MIN):
            wtype = "unclassified"
        elif l3 >= 0.5:
            wtype = "outside"
        elif l3 <= -1.0:
            wtype = "inside"
        else:
            wtype = "through"
        arc_rows.append(dict(zip(K, key), arc_g=float(np.nanmax(roll)) / YD_PER_G, dur=(e - sf) * DT,
                             kind=info.sample_kind, down=info.down, ytg=info.yards_to_go, align=info.lined_up_position,
                             press=int(pd.notna(info.time_to_pressure) or info.sack == 1), off=off, l3=l3, wtype=wtype))
        kmax = int(np.nanargmax(roll.values)) - f0
        arc_tr[key] = (la_, fw_, al[ii] / YD_PER_G, kmax, wtype)
    return arc_rows, arc_tr


arc_rows, arc_tr = cached("arc", arc_features)
arcp = pd.DataFrame(arc_rows)
put("arc_snaps_per_player", arcp[arcp.kind == "random"].groupby("nfl_id").size().median(), 0)
dl = dl.merge(arcp[arcp.kind == "random"].groupby("nfl_id").arc_g.mean().rename("arc_random").reset_index(),
              on="nfl_id", how="left")

# ============================================================================= cohort and baselines
_tick("cohort")
co = dl[dl.rush_snaps >= MIN_SNAPS].copy()
co["quick"] = -co.get_off
MAIN = ["log_pick", "combine_weight_z", "edge_share", "ngs_athleticism_score"]
TIMES = ["ten_yd_split_z", "three_cone_z"]
BASE_T = MAIN + TIMES
FILLX = ["combine_weight_z", "edge_share", "ngs_athleticism_score"]


def fill_times(tr, te):
    """Deterministic fill of missing official times from weight, EDGE share and NGS (fit on tr rows; never outcome)."""
    tr, te = tr.copy(), te.copy()
    for c in TIMES:
        mk = tr[c].notna()
        A = np.c_[np.ones(mk.sum()), tr.loc[mk, FILLX].values]
        b = np.linalg.lstsq(A, tr.loc[mk, c].values, rcond=None)[0]
        for d in (tr, te):
            m2 = d[c].isna()
            d.loc[m2, c] = np.c_[np.ones(m2.sum()), d.loc[m2, FILLX].values] @ b
    return tr, te


dH = co.dropna(subset=["hoop_z", "press_eb"]).copy()
put("n_cohort_hoop", len(dH))

# ---- signal map (Fig 2): raw rho with 95% CI; tracked drills also after pick, weight, NGS score, role
MEAS = [("forty_z", "40-yard time", "official", -1),
        ("log_pick", "Earlier draft pick", "official", -1), ("d1_z", "Rush drill: first-second burst", "tracking", 1),
        ("hoop_z", "Run-the-hoop bend", "tracking", 1)]
OUTC = [("press_eb", "NFL pressure rate"), ("quick", "NFL get-off (quicker)"), ("arc_random", "NFL rush-path bend")]
sig = []
for col, lab, kind, sgn in MEAS:
    for oc, olab in OUTC:
        src = dl if oc == "arc_random" else co
        r, lo, hi, n, p = sp_ci(sgn * src[col], src[oc])
        sig.append(dict(col=col, label=lab, kind=kind, outcome=oc, rho=r, lo=lo, hi=hi, n=n, mde=mde(n)))
sig = pd.DataFrame(sig)
AFTER = {}
for x in ["hoop_z", "d1_z"]:
    for y in ["press_eb", "quick"]:
        AFTER[(x, y)] = partial_sp(co, x, y, MAIN)[:3]
    AFTER[(x, "arc_random")] = partial_sp(dl.dropna(subset=[x, "arc_random"]), x, "arc_random",
                                          ["combine_weight_z", "edge_share"])[:3]
put("pr_d1_press", AFTER[("d1_z", "press_eb")][0])
put("pr_hoop_arc", AFTER[("hoop_z", "arc_random")][0])
r, lo, hi, n, p = sp_ci(dl.d1_z, dl.g_d1_random)
put("rho_d1_game", r); put("rho_d1_game_lo", lo); put("rho_d1_game_hi", hi); put("n_d1_game", n)
TABLES["signals"] = sig.round(3).to_dict("records")

# ---- the 14-pair screen (observed effects only): sets the typical effect used as the bend-slope prior
_tick("screen")


def screen_setup(df, outc, base):
    S = {}
    for col, _ in SCREEN:
        d = df[[col, outc] + base + ["draft_year"]].dropna()
        R = d[[col] + base].rank()
        A = np.c_[np.ones(len(d)), R[base].values]
        P = A @ np.linalg.pinv(A)
        S[col] = dict(idx=d.index.values, A=A, rx=R[col].values - P @ R[col].values, P=P)
    return S


def screen_stats(S, yser):
    out = {}
    for col, st in S.items():
        yr = stats.rankdata(yser.loc[st["idx"]].values)
        ry = yr - st["P"] @ yr
        out[col] = float(np.corrcoef(st["rx"], ry)[0, 1])
    return out


S_p, S_g = screen_setup(co, "press_eb", MAIN), screen_setup(co, "quick", MAIN)
obs_p, obs_g = screen_stats(S_p, co.press_eb), screen_stats(S_g, co.quick)
scr = pd.DataFrame([{"col": c, "pr_press": obs_p[c], "pr_getoff": obs_g[c], "n_press": len(S_p[c]["idx"]),
                     "n_getoff": len(S_g[c]["idx"])} for c, _ in SCREEN])
all14 = np.r_[scr.pr_press.abs().values, scr.pr_getoff.abs().values]
z = np.arctanh(np.r_[scr.pr_press.values, scr.pr_getoff.values])
se2 = 1 / (np.r_[scr.n_press.values, scr.n_getoff.values] - 3 - len(MAIN))
mz = np.average(z, weights=1 / se2)
tau2_s = max(0.0, np.var(z, ddof=1) - se2.mean())
z_shr = mz + tau2_s / (tau2_s + se2) * (z - mz)
eb_hoop = float(np.tanh(z_shr[list(scr.col).index("hoop_z")]))
TABLES["screen"] = scr.round(3).to_dict("records")

# ---- OLS on the 100+ cohort (sets the prior scale, as in round 4/5; and the draft-slot price)
dm = dH.assign(pp=lambda d: 100 * d.press_eb)
ofit = smf.ols("pp ~ log_pick + combine_weight_z + edge_share + ngs_athleticism_score + hoop_z", dm).fit()
r_main = partial_sp(co, "hoop_z", "press_eb", MAIN)[0]
pp_per_rho = float(f"{ofit.params['hoop_z']:.2f}") / float(f"{r_main:.2f}")
dHt, _ = fill_times(dH, dH.iloc[:0])
dmt = dHt.assign(pp=lambda d: 100 * d.press_eb)
FMT = "pp ~ log_pick + combine_weight_z + edge_share + ngs_athleticism_score + ten_yd_split_z + three_cone_z"
tfit = smf.ols(FMT + " + hoop_z", dmt).fit()
B_LP, B_LP_SE = -tfit.params["log_pick"], tfit.bse["log_pick"]
dH["res_t"] = 100 * dHt.press_eb - smf.ols(FMT, dmt).fit().fittedvalues

# ---- Fig 3 slopes by ran-40 group, same baseline, class fixed effects, class-stratified bootstrap
dmt["has40"] = dmt.forty_z.notna().astype(int)
FMLI = FMT + " + C(draft_year) + has40 + hoop_z:has40 + hoop_z"
ifit = smf.ols(FMLI, dmt).fit()
put("int_slope_no40", ifit.params["hoop_z"]); put("int_slope_40", ifit.params["hoop_z"] + ifit.params["hoop_z:has40"])
put("int_p", ifit.pvalues["hoop_z:has40"])
gr_ = {c: d for c, d in dmt.groupby("draft_year")}
bi = []
for _ in range(300):
    d_ = pd.concat([g_.iloc[RNG.integers(0, len(g_), len(g_))] for g_ in gr_.values()])
    try:
        f_ = smf.ols(FMLI, d_).fit().params
        bi.append((f_["hoop_z"], f_["hoop_z"] + f_["hoop_z:has40"]))
    except Exception:
        pass
bi = np.array(bi)
for j, tg in enumerate(["no40", "40"]):
    put(f"int_slope_{tg}_lo", np.percentile(bi[:, j], 2.5)); put(f"int_slope_{tg}_hi", np.percentile(bi[:, j], 97.5))
put("n_40", int(dmt.has40.sum())); put("n_no40", int((dmt.has40 == 0).sum()))
dH["has40"] = dmt.has40.values

# ---- placebo outcome and negative-control drill
tfl_ = s.groupby("nfl_id").agg(tfl=("tfl", "sum"), plays=("plays", "sum")).reset_index()
co2 = dH.merge(tfl_, on="nfl_id", how="left")
co2["runstop"] = co2.tfl / (co2.plays - co2.rush_snaps).where(co2.plays - co2.rush_snaps >= 50)
r, lo, hi, n = partial_sp(co2, "hoop_z", "runstop", MAIN)
put("plc_runstop", r); put("plc_runstop_lo", lo); put("plc_runstop_hi", hi); put("plc_runstop_n", n)
bp = events[(events.position_group == "DL") & (events.drill_name == "BACK_PEDAL_AND_REACT")].copy()
bp["bp_z"] = yz(bp.mean_speed, bp.draft_year)
cobp = co.merge(bp.groupby("nfl_id").bp_z.mean().reset_index(), on="nfl_id")
r, lo, hi, n = partial_sp(cobp, "bp_z", "press_eb", MAIN)
put("neg_bp", r); put("neg_bp_lo", lo); put("neg_bp_hi", hi); put("neg_bp_n", n)

# ---- path direction on sampled NFL snaps (EDGE alignment, 4+ yd off the field centre)
wt = arcp[arcp.wtype != "unclassified"].merge(dl[["nfl_id", "hoop_z", "combine_weight_z", "edge_share"]], on="nfl_id")
wt = wt[wt.hoop_z.notna()].copy()
put("wt_n", len(wt))
put("wt_share_outside", 100 * (wt.wtype == "outside").mean(), 0)
wpl = wt.groupby("nfl_id").agg(hoop_z=("hoop_z", "first"), combine_weight_z=("combine_weight_z", "first"),
                               edge_share=("edge_share", "first"), nw=("wtype", "size"),
                               out_share=("wtype", lambda v: (v == "outside").mean()),
                               l3=("l3", "mean")).reset_index()
wpl = wpl[wpl.nw >= 8]
r, lo, hi, n = partial_sp(wpl, "hoop_z", "out_share", ["combine_weight_z", "edge_share"])
put("wt_rho_out", r); put("wt_rho_out_lo", lo); put("wt_rho_out_hi", hi); put("wt_rho_out_n", n)
r, lo, hi, n = partial_sp(wpl, "hoop_z", "l3", ["combine_weight_z", "edge_share"])
put("wt_rho_l3", r); put("wt_rho_l3_lo", lo); put("wt_rho_l3_hi", hi)

# ============================================================================= out of class: fully nested loop
_tick("nested out-of-class")
cand = co.dropna(subset=["press_eb", "quick"]).copy()
cand = cand[cand[[c for c, _ in SCREEN]].notna().any(axis=1)]
classes = sorted(cand.draft_year.unique())
ATTF = {"hoop_z": (hoop_att, "bend_g", 1), "hoop_speed_z": (hoop_att, "mean_speed", 1),
        "hoop_time_z": (hoop_att, "duration_s", -1), "rush_bend_z": (rush_att, "bend_g", 1),
        "rush_acc_z": (rush_att, "peak_accel", 1), "rush_t3_z": (rush_att, "t3", -1), "d1_z": (rush_att, "d1", 1)}
ATTZ = {}
for f_, (tab, col, sg) in ATTF.items():
    a_ = tab[["nfl_id", "draft_year", col]].dropna().copy()
    a_["z"] = sg * yz(a_[col], a_.draft_year)
    ATTZ[f_] = a_
    cand = cand.merge(a_.groupby("nfl_id").size().rename(f_ + "_k").reset_index(), on="nfl_id", how="left")
HSH = hoop_att[["nfl_id", "bend_g_h1", "bend_g_h2"]].dropna()


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ols_pred(tr, te, cols, y):
    A = np.c_[np.ones(len(tr)), tr[cols].values]
    b = np.linalg.lstsq(A, tr[y].values, rcond=None)[0]
    return np.c_[np.ones(len(te)), te[cols].values] @ b


def impute_mi(tr, te, rng, cols, pred):
    """One stochastic draw (parameter + residual noise) for each missing official time, fit on tr rows."""
    tr, te = tr.copy(), te.copy()
    for col in cols:
        mk = tr[col].notna()
        X = np.c_[np.ones(mk.sum()), tr.loc[mk, pred].values]
        y = tr.loc[mk, col].values
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        dof = max(len(y) - X.shape[1], 1)
        s2d = (res @ res / dof) * dof / rng.chisquare(dof)
        bd = rng.multivariate_normal(b, s2d * np.linalg.inv(X.T @ X))
        for d in (tr, te):
            m2 = d[col].isna()
            d.loc[m2, col] = np.c_[np.ones(m2.sum()), d.loc[m2, pred].values] @ bd + rng.normal(0, np.sqrt(s2d), m2.sum())
    return tr, te


def feat_rel(f_, ids):
    if f_ == "hoop_z":
        h_ = HSH[HSH.nfl_id.isin(ids)]
        r_ = stats.spearmanr(h_.bend_g_h1, h_.bend_g_h2)[0]
        rel_ = 2 * r_ / (1 + r_)
    else:
        a_ = ATTZ[f_][ATTZ[f_].nfl_id.isin(ids)]
        try:
            rel_ = icc(a_.nfl_id, a_.z)[0]
        except Exception:
            rel_ = 0.5
    return float(np.clip(np.nan_to_num(rel_, nan=0.5), 0.05, 0.95))


def nested_full(df, base=MAIN, imp_rng=None):
    df = df.reset_index(drop=True)
    res = dict(Y=[], PB=[], PA=[], sel=[])
    for c in classes:
        tr, te = df[df.draft_year != c].copy(), df[df.draft_year == c].copy()
        if imp_rng is not None:
            tr, te = impute_mi(tr, te, imp_rng, TIMES, FILLX)
        Sp, Sg = screen_setup(tr, "press_eb", base), screen_setup(tr, "quick", base)
        st_ = {("press_eb", k): v for k, v in screen_stats(Sp, tr.press_eb).items()}
        st_.update({("quick", k): v for k, v in screen_stats(Sg, tr.quick).items()})
        keys = list(st_)
        r_ = np.array([st_[k] for k in keys])
        nn_ = np.array([len((Sp if k[0] == "press_eb" else Sg)[k[1]]["idx"]) for k in keys])
        zz = np.arctanh(np.clip(r_, -0.99, 0.99)); se2_ = 1 / (nn_ - 3 - len(base))
        mz_ = np.average(zz, weights=1 / se2_); t2_ = max(0.0, np.var(zz, ddof=1) - se2_.mean())
        j = int(np.argmax(np.abs(r_))); oc, best = keys[j]
        fac = float(np.tanh(mz_ + t2_ / (t2_ + se2_[j]) * (zz[j] - mz_)) / r_[j])
        rel_ = feat_rel(best, set(tr.nfl_id))
        tr2, te2 = tr.dropna(subset=[best, oc]).copy(), te.dropna(subset=[best, oc]).copy()
        for d_ in (tr2, te2):
            k_ = d_[best + "_k"].fillna(1).values
            d_["_x"] = k_ * rel_ / (1 + (k_ - 1) * rel_) * d_[best].values
        mu_, sd_ = tr2[oc].mean(), tr2[oc].std()
        tr2["_y"], te2["_y"] = (tr2[oc] - mu_) / sd_, (te2[oc] - mu_) / sd_
        A = np.c_[np.ones(len(tr2)), tr2[base].values, tr2._x.values]
        b = np.linalg.lstsq(A, tr2._y.values, rcond=None)[0]
        b[-1] *= fac
        res["PA"].append(np.c_[np.ones(len(te2)), te2[base].values, te2._x.values] @ b)
        res["PB"].append(ols_pred(tr2, te2, base, "_y"))
        res["Y"].append(te2._y.values); res["sel"].append(f"{best}|{oc}")
    out = {"pooled": r2(np.concatenate(res["Y"]), np.concatenate(res["PA"])) -
           r2(np.concatenate(res["Y"]), np.concatenate(res["PB"])), "sel": res["sel"],
           "n": int(sum(len(v) for v in res["Y"]))}
    for c, y_, pa_, pb_ in zip(classes, res["Y"], res["PA"], res["PB"]):
        out[int(c)] = r2(y_, pa_) - r2(y_, pb_)
        out[f"n{int(c)}"] = len(y_)
    return out


IMP_RNG = np.random.default_rng(7)
NFI = [nested_full(cand, base=BASE_T, imp_rng=IMP_RNG) for _ in range(10)]
put("nf_dr2", np.mean([o_["pooled"] for o_ in NFI]), 3); put("nf_n", NFI[0]["n"])
for c in classes:
    put(f"nf_dr2_{c}", np.mean([o_[int(c)] for o_ in NFI]), 3)
    put(f"nf_n_{c}", NFI[0][f"n{int(c)}"])
SELS = [s_ for o_ in NFI for s_ in o_["sel"]]
put("nf_hoop_sel_pct", 100 * np.mean([s_ == "hoop_z|press_eb" for s_ in SELS]), 0)
gcand = {c: d for c, d in cand.groupby("draft_year")}
BRNG = np.random.default_rng(11)
NFB = []
for _ in range(NLOCO):
    d_ = pd.concat([g_.iloc[BRNG.integers(0, len(g_), len(g_))] for g_ in gcand.values()])
    try:
        o_ = nested_full(d_, base=BASE_T, imp_rng=IMP_RNG)
        NFB.append([o_["pooled"]] + [o_[int(c)] for c in classes])
    except Exception:
        pass
NFB = np.array(NFB)
put("nf_dr2_lo", np.percentile(NFB[:, 0], 2.5), 3); put("nf_dr2_hi", np.percentile(NFB[:, 0], 97.5), 3)
put("nf_pos", 100 * np.mean(NFB[:, 0] > 0), 0); put("nf_nboot", len(NFB))
put("nf4_dr2", nested_full(cand)["pooled"], 3)

# ---- the stated 'official times' baseline tested directly: observed times only, no imputation, same nested loop
for tc, tg in [("ten_yd_split_z", "ob10"), ("three_cone_z", "ob3")]:
    sub_ = cand[cand[tc].notna()].copy()
    base_ = MAIN + [tc]
    o_ = nested_full(sub_, base=base_)
    put(f"{tg}_dr2", o_["pooled"], 3); put(f"{tg}_n", o_["n"])
    gs_ = {c: d for c, d in sub_.groupby("draft_year")}
    bb_ = []
    for _ in range(NLOCO):
        d_ = pd.concat([g_.iloc[BRNG.integers(0, len(g_), len(g_))] for g_ in gs_.values()])
        try:
            bb_.append(nested_full(d_, base=base_)["pooled"])
        except Exception:
            pass
    put(f"{tg}_dr2_lo", np.percentile(bb_, 2.5), 3); put(f"{tg}_dr2_hi", np.percentile(bb_, 97.5), 3)
    TABLES[f"{tg}_boot"] = list(np.round(bb_, 4))

# ============================================================================= headline: Bayesian beta-binomial
_tick("Bayesian headline")
REL = float(NUM["rel_hoop_sb"])
bz = dl[dl.hoop_z.notna() & (dl.rush_snaps > 0)].copy()
bz["pressures"] = bz.pressures.fillna(0)
bz["lgps"] = np.log1p(bz.games_per_season)
for c_ in TIMES:
    bz[c_ + "_m"] = bz[c_].isna().astype(float)
    ok_ = bz[c_].notna()
    A_ = np.c_[np.ones(len(bz)), bz[FILLX].values]
    cf_ = np.linalg.lstsq(A_[ok_.values], bz.loc[ok_, c_].values, rcond=None)[0]
    bz[c_ + "_f"] = np.where(ok_, bz[c_], A_ @ cf_)
put("bayes_n", len(bz)); put("bayes_n_norush", int((dl.hoop_z.notna() & (dl.rush_snaps == 0)).sum()))
put("bayes_n_10yd_obs", int(bz.ten_yd_split_z.notna().sum())); put("bayes_n_3cone_obs", int(bz.three_cone_z.notna().sum()))
ZC4 = ["log_pick", "combine_weight_z", "edge_share", "ngs_athleticism_score", "lgps"]
ZCT = ZC4 + ["ten_yd_split_z_f", "three_cone_z_f", "ten_yd_split_z_m"]
pbar = float(co.press_rate.mean())
rho_typ = float(np.mean(all14))
PRIOR_B = rho_typ * pp_per_rho / (100 * pbar * (1 - pbar)) / np.sqrt(REL)
put("bayes_rho_typ", rho_typ)
ghx, ghw = np.polynomial.hermite_e.hermegauss(15)
lghw = np.log(ghw / ghw.sum())


def make_model(d, cols, ref=None):
    """Arrays for the beta-binomial likelihood on rows d with (standardized) covariates cols.
    ref: frame whose means / sds standardize the covariates (training rows for out-of-class prediction)."""
    ref = d if ref is None else ref
    Zr = d[cols]
    sd = ref[cols].std().replace(0, np.nan)
    keep = sd.notna().values
    Zm = ((Zr - ref[cols].mean()) / sd).values[:, keep]
    y, n = d.pressures.values.astype(float), d.rush_snaps.values.astype(float)
    x, k = d.hoop_z.values, d.hoop_k.values
    lam = REL / (REL + (1 - REL) / k)
    mS, sS = lam * x / np.sqrt(REL), np.sqrt(lam * (1 - REL) / k / REL)
    return dict(Z=Zm, NZ=Zm.shape[1], edge=(d.edge_share >= 0.5).values, cls=pd.Categorical(d.draft_year).codes,
                y=y, n=n, mS=mS, Tn=mS[:, None] + sS[:, None] * ghx[None, :],
                lcb=gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1))


def logpost(t, M, prior_b):
    NZ = M["NZ"]
    a, beta, bi, be = t[0], t[1:1 + NZ], t[1 + NZ], t[2 + NZ]
    zu, lsu, lk = t[3 + NZ:6 + NZ], t[6 + NZ], t[7 + NZ]
    su, kap = np.exp(lsu), np.exp(lk)
    eta = a + su * zu[M["cls"]] + M["Z"] @ beta
    mu_ = expit(eta[:, None] + np.where(M["edge"], be, bi)[:, None] * M["Tn"])
    mu_ = np.clip(mu_, 1e-9, 1 - 1e-9)
    al, bb = mu_ * kap, (1 - mu_) * kap
    yb, nb_ = M["y"], M["n"]
    ll = (M["lcb"][:, None] + gammaln(yb[:, None] + al) + gammaln((nb_ - yb)[:, None] + bb) - gammaln(nb_[:, None] + kap)
          - gammaln(al) - gammaln(bb) + gammaln(kap)) + lghw[None, :]
    m_ = ll.max(1)
    L = (m_ + np.log(np.exp(ll - m_[:, None]).sum(1))).sum()
    lp = -0.5 * (a - _logit(0.08)) ** 2 - 0.5 * (beta ** 2).sum() / 0.25 - 0.5 * (bi ** 2 + be ** 2) / prior_b ** 2
    lp += -0.5 * (zu ** 2).sum() - 0.5 * (su / 0.3) ** 2 + lsu - 0.5 * (lk - np.log(100)) ** 2
    return L + lp


def t0(M):
    return np.r_[_logit(0.08), np.zeros(M["NZ"]), 0, 0, np.zeros(3), np.log(0.1), np.log(100)]


def am_chain(seed, M, prior_b, niter=NMCMC):
    NP_ = M["NZ"] + 8
    rng = np.random.default_rng(seed)
    t = t0(M) + rng.normal(0, 0.05, NP_)
    lp = logpost(t, M, prior_b)
    C = np.eye(NP_) * 0.005
    H_ = np.empty((niter, NP_))
    for i in range(niter):
        if i > 2000 and i % 500 == 0:
            C = np.cov(H_[max(0, i - 20000):i].T) * 2.38 ** 2 / NP_ + np.eye(NP_) * 1e-7
        tp = rng.multivariate_normal(t, C)
        lpp = logpost(tp, M, prior_b)
        if np.log(rng.random()) < lpp - lp:
            t, lp = tp, lpp
        H_[i] = t
    return H_[niter // 2:]


def rhat(v):
    m_ = v.mean(1); W = v.var(1, ddof=1).mean(); B_ = v.shape[1] * m_.var(ddof=1)
    return float(np.sqrt(((v.shape[1] - 1) / v.shape[1] * W + B_ / v.shape[1]) / W))


def effects(smp, M):
    """Extra pressures per 450-rush season for +1 sd of measured bend, averaged over the model's rows."""
    NZ, out = M["NZ"], []
    for t in smp:
        eta = t[0] + M["Z"] @ t[1:1 + NZ]
        sl = np.where(M["edge"], t[2 + NZ], t[1 + NZ])
        d_ = 450 * (expit(eta + sl * (M["mS"] + np.sqrt(REL))) - expit(eta + sl * M["mS"]))
        out.append((d_.mean(), d_[M["edge"]].mean() if M["edge"].any() else np.nan,
                    d_[~M["edge"]].mean() if (~M["edge"]).any() else np.nan))
    return np.array(out)


def bayes_fit(M, prior_b, seed0):
    NP_ = M["NZ"] + 8
    ch = np.array([am_chain(seed0 + s_, M, prior_b) for s_ in range(4)])
    rh = max(rhat(ch[:, ::5, j]) for j in range(NP_))
    return effects(ch.reshape(-1, NP_)[::8], M), rh


def laplace(M, prior_b, ndraw=2000, seed=0, draws=True):
    """Posterior mode + numerical Hessian (same likelihood and priors as the MCMC)."""
    f = lambda t: -logpost(t, M, prior_b)
    o = optimize.minimize(f, t0(M), method="BFGS", options={"gtol": 1e-5, "maxiter": 2000})
    if not draws:
        return effects([o.x], M)[0]
    P = len(o.x); Hm = np.zeros((P, P)); h = 1e-3
    f0 = f(o.x)
    E = np.eye(P) * h
    fp = np.array([f(o.x + E[i]) for i in range(P)]); fm = np.array([f(o.x - E[i]) for i in range(P)])
    for i in range(P):
        Hm[i, i] = (fp[i] - 2 * f0 + fm[i]) / h ** 2
        for j in range(i + 1, P):
            Hm[i, j] = Hm[j, i] = (f(o.x + E[i] + E[j]) - fp[i] - fp[j] + f0) / h ** 2
    w, V = np.linalg.eigh(Hm)
    cov = V @ np.diag(1 / np.clip(w, 1e-6, None)) @ V.T
    smp = np.random.default_rng(seed).multivariate_normal(o.x, cov, ndraw)
    return effects(smp, M)


def summ(v, tag, nd=1):
    put(f"{tag}", np.median(v), nd); put(f"{tag}_lo", np.percentile(v, 5), nd); put(f"{tag}_hi", np.percentile(v, 95), nd)


MH = make_model(bz, ZCT)
BE, BRH = cached(f"mcmc_{NMCMC}_{PRIOR_B:.6f}_{len(bz)}", lambda: bayes_fit(MH, PRIOR_B, 100))
put("bayes_rhat", BRH)
summ(BE[:, 0], "bayes_all")
put("bayes_all_p0", np.mean(BE[:, 0] > 0)); put("bayes_all_p05", np.mean(BE[:, 0] > 0.5))
put("bayes_edge", np.median(BE[:, 1]), 1); put("bayes_int", np.median(BE[:, 2]), 1)
BAYES_DRAWS = BE[:, 0]

# ---- prior dependence: the same MCMC under three prior scales on the bend slope
_tick("prior scales (MCMC)")
PRIORS = [("skeptical", PRIOR_B), ("double", 2 * PRIOR_B), ("wide", 0.25)]
PRIOR_DRAWS = {"skeptical": BAYES_DRAWS}
for tg, pb_ in PRIORS[1:]:
    be_, rh_ = cached(f"mcmc_{NMCMC}_{pb_:.6f}_{len(bz)}", lambda: bayes_fit(MH, pb_, 200))
    PRIOR_DRAWS[tg] = be_[:, 0]
    summ(be_[:, 0], f"pr_{tg}")
    put(f"pr_{tg}_p0", np.mean(be_[:, 0] > 0))
put("prior_skep", PRIOR_B, 3); put("prior_double", 2 * PRIOR_B, 3); put("prior_wide", 0.25)


# ---- out-of-class proper scoring rule: held-out log predictive density, bend model vs the same model without bend
def pointwise_lpd(smp, M):
    """log mean_s p(y_i | theta_s) for each row; a new class gets class effect 0 (the population mean)."""
    NZ, L = M["NZ"], []
    for t in smp:
        a, beta, bi, be = t[0], t[1:1 + NZ], t[1 + NZ], t[2 + NZ]
        kap = np.exp(t[7 + NZ])
        eta = a + M["Z"] @ beta
        mu_ = np.clip(expit(eta[:, None] + np.where(M["edge"], be, bi)[:, None] * M["Tn"]), 1e-9, 1 - 1e-9)
        al, bb = mu_ * kap, (1 - mu_) * kap
        yb, nb_ = M["y"], M["n"]
        ll = (M["lcb"][:, None] + gammaln(yb[:, None] + al) + gammaln((nb_ - yb)[:, None] + bb)
              - gammaln(nb_[:, None] + kap) - gammaln(al) - gammaln(bb) + gammaln(kap)) + lghw[None, :]
        m_ = ll.max(1)
        L.append(m_ + np.log(np.exp(ll - m_[:, None]).sum(1)))
    L = np.array(L)
    mx = L.max(0)
    return mx + np.log(np.exp(L - mx).mean(0))


def laplace_draws(M, prior_b, ndraw, seed):
    f = lambda t: -logpost(t, M, prior_b)
    o = optimize.minimize(f, t0(M), method="BFGS", options={"gtol": 1e-5, "maxiter": 2000})
    P = len(o.x); Hm = np.zeros((P, P)); h = 1e-3; f0 = f(o.x); E = np.eye(P) * h
    fp = np.array([f(o.x + E[i]) for i in range(P)]); fm = np.array([f(o.x - E[i]) for i in range(P)])
    for i in range(P):
        Hm[i, i] = (fp[i] - 2 * f0 + fm[i]) / h ** 2
        for j in range(i + 1, P):
            Hm[i, j] = Hm[j, i] = (f(o.x + E[i] + E[j]) - fp[i] - fp[j] + f0) / h ** 2
    w, V = np.linalg.eigh(Hm)
    return np.random.default_rng(seed).multivariate_normal(o.x, V @ np.diag(1 / np.clip(w, 1e-6, None)) @ V.T, ndraw)


_tick("held-out log predictive density")
ELPD = {}
for tg, pb_ in PRIORS:
    diffs, per_cls = [], {}
    for c in classes:
        tr_, te_ = bz[bz.draft_year != c], bz[bz.draft_year == c]
        Mtr, Mte = make_model(tr_, ZCT), make_model(te_, ZCT, ref=tr_)
        lh = pointwise_lpd(laplace_draws(Mtr, pb_, 800, 40 + int(c) % 10), Mte)
        l0 = pointwise_lpd(laplace_draws(Mtr, 1e-4, 800, 50 + int(c) % 10), Mte)
        diffs.append(lh - l0); per_cls[int(c)] = float((lh - l0).sum())
    dd = np.concatenate(diffs)
    ELPD[tg] = dict(sum=float(dd.sum()), se=float(np.sqrt(len(dd)) * dd.std(ddof=1)), n=len(dd), cls=per_cls,
                    pw=dd)
    put(f"elpd_{tg}", dd.sum(), 2); put(f"elpd_{tg}_se", np.sqrt(len(dd)) * dd.std(ddof=1), 2)
put("elpd_n", ELPD["skeptical"]["n"])
put("elpd_skeptical_pos", int(sum(v > 0 for v in ELPD["skeptical"]["cls"].values())))
TABLES["elpd"] = {k: {kk: vv for kk, vv in v.items() if kk != "pw"} for k, v in ELPD.items()}

# ---- second in-game mechanism test: is the hoop-pressure link stronger on EDGE-aligned (corner) snaps?
_tick("alignment test")
sn = arcp[arcp.kind == "random"].merge(dl[["nfl_id", "hoop_z", "log_pick", "combine_weight_z", "edge_share"]],
                                        on="nfl_id")
sn = sn[sn.hoop_z.notna()].copy()
sn["wide"] = sn["align"].isin(["EDGE", "OLB"]).astype(int)
FA = "press ~ log_pick + combine_weight_z + wide + hoop_z + hoop_z:wide"


def align_fit(d):
    f_ = smf.logit(FA, d).fit(disp=0).params
    return f_["hoop_z"] + f_["hoop_z:wide"], f_["hoop_z"]


aw, ai = align_fit(sn)
gsn = {k: v for k, v in sn.groupby("nfl_id")}
ids_ = np.array(list(gsn))
ab = []
arng = np.random.default_rng(17)
for _ in range(400):
    pk_ = arng.choice(ids_, len(ids_))
    try:
        ab.append(align_fit(pd.concat([gsn[i] for i in pk_], ignore_index=True)))
    except Exception:
        pass
ab = np.array(ab)
put("al_snaps", len(sn)); put("al_players", sn.nfl_id.nunique()); put("al_wide_snaps", int(sn.wide.sum()))
put("al_or_wide", np.exp(aw)); put("al_or_wide_lo", np.exp(np.percentile(ab[:, 0], 5)))
put("al_or_wide_hi", np.exp(np.percentile(ab[:, 0], 95)))
put("al_or_int", np.exp(ai)); put("al_or_int_lo", np.exp(np.percentile(ab[:, 1], 5)))
put("al_or_int_hi", np.exp(np.percentile(ab[:, 1], 95)))
put("hoop_dur_s", hoop_att.duration_s.median(), 1)

# ---- draft-slot price of the headline, with uncertainty (posterior effect x OLS pick slope, same baseline)
_tick("sensitivities (Laplace)")
prng = np.random.default_rng(5)
bl = np.clip(prng.normal(B_LP, B_LP_SE, len(BAYES_DRAWS)), 0.05, None)
peq = 100 * np.exp(-(BAYES_DRAWS / 4.5) / bl)
put("pick_equiv", int(round(100 * np.exp(-(float(NUM["bayes_all"]) / 4.5) / B_LP))))
put("pick_equiv_lo", int(round(np.percentile(peq, 10)))); put("pick_equiv_hi", int(round(np.percentile(peq, 90))))

LP = laplace(MH, PRIOR_B, seed=1)
summ(LP[:, 0], "lap_all")
summ(laplace(MH, 0.25, seed=2)[:, 0], "lap_wide")
summ(laplace(make_model(bz, ZC4), PRIOR_B, seed=3)[:, 0], "lap_4item")

# complete cases on observed stopwatch times: with and without the times, same rows
c3 = bz[bz.three_cone_z.notna()]
c10 = bz[bz.ten_yd_split_z.notna()]
put("cc3_n", len(c3)); put("cc10_n", len(c10))
summ(laplace(make_model(c3, ZC4 + ["three_cone_z"]), PRIOR_B, seed=4)[:, 0], "cc3_t")
summ(laplace(make_model(c3, ZC4), PRIOR_B, seed=5)[:, 0], "cc3_0")
summ(laplace(make_model(c10, ZC4 + ["ten_yd_split_z"]), PRIOR_B, seed=6)[:, 0], "cc10_t")
summ(laplace(make_model(c10, ZC4), PRIOR_B, seed=7)[:, 0], "cc10_0")

# multiple imputation: 20 stochastic draws (imputation model: weight, role, NGS, bend and the pressure rate),
# Laplace posterior draws pooled across imputations
M_MI = 20
bz["lpe"] = _logit(bz.press_eb)
mrng = np.random.default_rng(13)
mi = []
for i in range(M_MI):
    d_, _ = impute_mi(bz, bz.iloc[:0], mrng, TIMES, FILLX + ["hoop_z", "lpe"])
    mi.append(laplace(make_model(d_, ZC4 + TIMES), PRIOR_B, ndraw=500, seed=100 + i)[:, 0])
summ(np.concatenate(mi), "mi")
put("mi_m", M_MI)

# leave one rusher out (posterior-mode plug-in)
full_pt = laplace(MH, PRIOR_B, draws=False)[0]
loo = []
for i in range(len(bz)):
    loo.append(laplace(make_model(bz.drop(bz.index[i]), ZCT), PRIOR_B, draws=False)[0])
loo = np.array(loo)
put("loo_full", full_pt, 1); put("loo_min", loo.min(), 1); put("loo_max", loo.max(), 1)
TABLES["loo_min_player"] = bz.display_name.iloc[int(np.argmin(loo))]
TABLES["loo_max_player"] = bz.display_name.iloc[int(np.argmax(loo))]

# cross-fitted sign consistency: for each class, the effect estimated without it and inside it alone
SIGN = []
for c in classes:
    tr_ = laplace(make_model(bz[bz.draft_year != c], ZCT), PRIOR_B, seed=20 + int(c) % 10)[:, 0]
    te_ = laplace(make_model(bz[bz.draft_year == c], ZCT), PRIOR_B, seed=30 + int(c) % 10)[:, 0]
    put(f"sg_tr_{c}", np.median(tr_), 1); put(f"sg_te_{c}", np.median(te_), 1)
    put(f"sg_te_{c}_lo", np.percentile(te_, 5), 1); put(f"sg_te_{c}_hi", np.percentile(te_, 95), 1)
    put(f"sg_n_{c}", int((bz.draft_year == c).sum()))
    SIGN.append(dict(c=int(c), tr=np.median(tr_), te=np.median(te_)))
put("sg_same", int(sum(np.sign(r_["tr"]) == np.sign(r_["te"]) for r_ in SIGN)))
put("sg_pos_te", int(sum(r_["te"] > 0 for r_ in SIGN)))

# pre-registration power (one-sided 0.05, about 40 rushers, selection-shrunk partial rho, full baseline)
put("power_prereg", power_at(eb_hoop, 40, len(BASE_T)))
put("alpha_prereg", 0.05)

# ============================================================================= board
_tick("board")
ba = dl[dl.hoop_z.notna()].copy()
ba["lam"] = ba.hoop_k * rel_hoop / (1 + (ba.hoop_k - 1) * rel_hoop)
ba["shr"] = ba.lam * ba.hoop_z
ba["shr_sd"] = np.sqrt(ba.lam * (1 - ba.lam))
ba["shr_lo"], ba["shr_hi"] = ba.shr - 1.2816 * ba.shr_sd, ba.shr + 1.2816 * ba.shr_sd
# out-of-class no-tracking forecast for every runner with a rush: fit on the other classes' 100+ cohort
fc = []
for c in classes:
    tr_ = co[co.draft_year != c]
    te_ = ba[(ba.draft_year == c) & (ba.rush_snaps > 0)]
    tr_, te_ = fill_times(tr_, te_)
    fc.append(pd.DataFrame({"nfl_id": te_.nfl_id.values, "pb": ols_pred(tr_, te_, BASE_T, "press_eb")}))
ba = ba.merge(pd.concat(fc), on="nfl_id", how="left")
ba["pr_lo"] = stats.beta.ppf(0.1, ba.pressures + 0.5, ba.rush_snaps - ba.pressures + 0.5)
ba["pr_hi"] = stats.beta.ppf(0.9, ba.pressures + 0.5, ba.rush_snaps - ba.pressures + 0.5)
okr = ba.rush_snaps >= RATE_MIN
ba.loc[~okr, ["pb", "pr_lo", "pr_hi"]] = np.nan
ba["call"] = np.select([ba.shr_lo > 0, ba.shr_hi < 0], ["plus", "minus"], "neutral")
# role split: an interior minus is mostly size (the model already prices weight), so it is not used as a call
put("int_minus_min_wt", int(ba[(ba.call == "minus") & (ba.role == "interior")].combine_weight.min()))
put("edge_minus_n", int(((ba.call == "minus") & (ba.role == "EDGE")).sum()))
ok100 = ba.rush_snaps >= MIN_SNAPS
hi_flag = ok100 & (ba.shr >= 0.4) & (ba.pr_hi < ba.pb)
lo_flag = ok100 & (ba.shr <= -0.4) & (ba.pr_lo > ba.pb)
ba.loc[hi_flag, "call"] = "film (under)"
ba.loc[lo_flag, "call"] = "film (over)"
ba["rank_in_class"] = ba.groupby("draft_year").shr.rank(ascending=False, method="first").astype(int)
ba = ba.sort_values(["draft_year", "rank_in_class"])
ba[["draft_year", "rank_in_class", "display_name", "role", "pick", "combine_weight", "hoop_k", "hoop_g", "hoop_z", "shr",
    "shr_lo", "shr_hi", "rush_snaps", "press_rate", "pr_lo", "pr_hi", "pb", "call"]].round(4).to_csv(
    OUT / "board_2023_2025.csv", index=False)
called = ba[ba.call != "neutral"].copy()
for rl, tg in [("EDGE", "e"), ("interior", "i")]:
    for cl in ["plus", "minus"]:
        put(f"calls_{tg}_{cl}", int(((called.role == rl) & (called.call == cl)).sum()))
put("calls_film", int(called.call.str.startswith("film").sum()))
# forecast vs actual for the plus / minus calls with 100+ rushes (calls use the drill only)
ev_ = called[called.call.isin(["plus", "minus"]) & (called.rush_snaps >= MIN_SNAPS)]
for cl in ["plus", "minus"]:
    e_ = ev_[ev_.call == cl]
    put(f"fa_{cl}_n", len(e_)); put(f"fa_{cl}_above", int((e_.press_rate > e_.pb).sum()))
    put(f"fa_{cl}_gap", 100 * (e_.press_rate - e_.pb).mean(), 1)

ROLE_TXT = {"EDGE": "EDGE", "interior": "Int"}


def call_txt(r_):
    if r_.call.startswith("film"):
        return "film first " + r_.call[5:]
    if r_.call == "neutral":
        return "neutral"
    if r_.role == "EDGE":
        return "**bend +**" if r_.call == "plus" else "**bend −**"
    return "agility note" if r_.call == "plus" else "no call (size)"


def pk_txt(p):
    return "UDFA" if p == UDFA_PICK else str(int(p))


def prs_txt(r_):
    if r_.rush_snaps == 0:
        return "— (no rushes)"
    if pd.isna(r_.pb):
        return f"— ({int(r_.rush_snaps)} rushes)"
    return f"{100 * r_.press_rate:.1f}% vs {100 * r_.pb:.1f}% ({int(r_.rush_snaps)})"


def tie_range(r_):
    """Pick window in which the tie-break applies: picks within 30% of his own (UDFA: day-3 and undrafted)."""
    if r_.role != "EDGE":
        return "—"
    if r_.pick >= 200:
        return "day 3 / UDFA EDGEs"
    return f"EDGEs picked {int(round(r_.pick / 1.3))}–{int(round(r_.pick * 1.3))}"


def use_txt(r_):
    if r_.call.startswith("film"):
        return "film first: rookie year below forecast" if r_.call == "film (under)" else "film first"
    if r_.role == "EDGE":
        return "break ties his way" if r_.call == "plus" else "must win with power; check film"
    return "agility note (stunts)" if r_.call == "plus" else "no call: size, priced by baseline"


bd = ba[ba.draft_year == 2025].sort_values("shr", ascending=False).reset_index(drop=True)
bd["rank"] = np.arange(1, len(bd) + 1)
put("board_n", len(bd))
t25 = bd[bd.call != "neutral"]
head = "| # of " + str(len(bd)) + " | Player | Pick | Role | Bend, sd [80%] | Tie-break range | Use |"
out = [head, "|" + "---|" * 7]
for i, r_ in enumerate(t25.itertuples()):
    k_ = f"tb_b25_{i}"
    s0, s1, s2 = put(k_ + "_s", r_.shr), put(k_ + "_l", r_.shr_lo), put(k_ + "_h", r_.shr_hi)
    out.append(f"| {int(r_.rank)} | {r_.display_name} | {pk_txt(r_.pick)} | {ROLE_TXT[r_.role]} | "
               f"{mn(s0)} [{mn(s1)}, {mn(s2)}] | {tie_range(r_)} | {use_txt(r_)} |")
MD = {"board25": "\n".join(out)}

# ---- head-to-head EDGE pairs: same class, picks within 30%, bend gap >= 0.4 sd, both 100+ rushes
_tick("pairs")
PR_RATIO, PR_GAP = 1.3, 0.4
ep = ba[(ba.role == "EDGE") & (ba.rush_snaps >= MIN_SNAPS) & ba.pb.notna()].copy()
ep["res"] = ep.press_rate - ep.pb


def pair_list(d, shr):
    out_ = []
    for c, g_ in d.groupby("draft_year"):
        ix = g_.index.values
        for a_i in range(len(ix)):
            for b_i in range(a_i + 1, len(ix)):
                i, j = ix[a_i], ix[b_i]
                pi, pj = d.at[i, "pick"], d.at[j, "pick"]
                if max(pi, pj) / min(pi, pj) <= PR_RATIO and abs(shr[i] - shr[j]) >= PR_GAP:
                    hi_, lo_ = (i, j) if shr[i] > shr[j] else (j, i)
                    out_.append((hi_, lo_, max(pi, pj) / min(pi, pj)))
    return out_


def win_rate(d, shr):
    pl_ = pair_list(d, shr)
    return (np.mean([d.at[h, "res"] > d.at[l, "res"] for h, l, _ in pl_]) if pl_ else np.nan), len(pl_)


shr0 = ep.shr.to_dict()
wr, npairs = win_rate(ep, shr0)
put("pair_n", npairs); put("pair_wins", int(round(wr * npairs))); put("pair_win_pct", 100 * wr, 0)
prng2 = np.random.default_rng(23)
perm = []
for _ in range(2000):
    sh_ = {}
    for c, g_ in ep.groupby("draft_year"):
        sh_.update(dict(zip(g_.index, prng2.permutation(g_.shr.values))))
    perm.append(win_rate(ep, sh_)[0])
perm = np.array(perm)
put("pair_null_pct", 100 * np.nanmean(perm), 0)
put("pair_perm_p", np.nanmean(perm >= wr))
bw = []
for _ in range(1000):
    d_ = pd.concat([g_.iloc[prng2.integers(0, len(g_), len(g_))] for _, g_ in ep.groupby("draft_year")])
    d_ = d_.reset_index().rename(columns={"index": "orig"})
    d_ = d_.drop_duplicates("orig")      # a resampled player counts once in pair formation
    v_, _ = win_rate(d_, d_.shr.to_dict())
    bw.append(v_)
put("pair_win_lo", 100 * np.nanpercentile(bw, 5), 0); put("pair_win_hi", 100 * np.nanpercentile(bw, 95), 0)
# named pairs: the three disjoint pairs with the closest picks
used_, named = set(), []
for h, l, rt in sorted(pair_list(ep, shr0), key=lambda z: z[2]):
    if h in used_ or l in used_:
        continue
    used_ |= {h, l}
    named.append((h, l))
    if len(named) == 3:
        break
out = ["| Class | Higher bend (pick, sd) | Lower bend (pick, sd) | Pressure − forecast, pp (rushes) | Tie-break |",
       "|---|---|---|---|---|"]
for i, (h, l) in enumerate(named):
    H, Lw = ep.loc[h], ep.loc[l]
    rh_, rl_ = put(f"tb_pr{i}_rh", 100 * H.res, 1), put(f"tb_pr{i}_rl", 100 * Lw.res, 1)
    sh1, sl1 = put(f"tb_pr{i}_sh", H.shr), put(f"tb_pr{i}_sl", Lw.shr)
    sg = lambda v: ("+" if not str(v).startswith("-") else "") + mn(v)
    out.append(f"| {int(H.draft_year)} | {H.display_name} ({pk_txt(H.pick)}, {sg(sh1)}) | {Lw.display_name} "
               f"({pk_txt(Lw.pick)}, {sg(sl1)}) | {sg(rh_)} ({int(H.rush_snaps)}) vs {sg(rl_)} ({int(Lw.rush_snaps)}) | "
               f"{'right' if H.res > Lw.res else 'wrong'} |")
MD["pairs"] = "\n".join(out)
TABLES["pairs_named"] = [(ep.at[h, "display_name"], ep.at[l, "display_name"]) for h, l in named]

# ---- plus-call tally with a bootstrap interval, a permutation null and the 2025-only (post-selection) subset
pc_ = ev_[ev_.call == "plus"]
above = (pc_.press_rate > pc_.pb).values
bt_ = [np.mean(above[prng2.integers(0, len(above), len(above))]) for _ in range(2000)]
put("fa_plus_lo", 100 * np.percentile(bt_, 5), 0); put("fa_plus_hi", 100 * np.percentile(bt_, 95), 0)
base_ok = ba[(ba.rush_snaps >= MIN_SNAPS) & ba.pb.notna()]
pn_ = []
for _ in range(2000):
    d_ = ba.copy()
    for c, idx_ in d_.groupby("draft_year").groups.items():
        d_.loc[idx_, "hoop_z"] = prng2.permutation(d_.loc[idx_, "hoop_z"].values)
    sh_ = d_.lam * d_.hoop_z
    lo_ = sh_ - 1.2816 * d_.shr_sd
    m_ = (lo_ > 0) & (d_.rush_snaps >= MIN_SNAPS) & d_.pb.notna()
    if m_.sum():
        pn_.append(np.mean(d_.loc[m_, "press_rate"] > d_.loc[m_, "pb"]))
put("fa_plus_null", 100 * np.mean(pn_), 0); put("fa_plus_perm_p", np.mean(np.array(pn_) >= above.mean()))
put("fa_plus_pct", 100 * above.mean(), 0)
p25 = pc_[pc_.draft_year == 2025]
put("fa25_n", len(p25)); put("fa25_above", int((p25.press_rate > p25.pb).sum()))
# clear calls: the 80% interval of the actual rate excludes the forecast
cl80 = ba[(ba.call != "neutral") & ba.pb.notna()]
put("clear_above", int((cl80.pr_lo > cl80.pb).sum())); put("clear_below", int((cl80.pr_hi < cl80.pb).sum()))
put("clear_n", len(cl80))
MD_TXT = {}

# ============================================================================= FIGURES
_tick("figures")


def name_short(n):
    return n.split(" ")[0][0] + ". " + " ".join(n.split(" ")[1:])


# ---- Fig 1 (hero): four named rushers, drill vs Sunday paths (left), drill bend vs game bend for all DL (right)
P4 = ["Chop Robinson", "Donovan Ezeiruaku", "Tuli Tuipulotu", "Jordan Phillips"]
PC = {"outside": OI["blue"], "inside": OI["green"], "through": "#4D4D4D", "unclassified": "#BBBBBB"}
PRC = OI["red"]
HC = [OI["red"], OI["orange"], OI["purple"], OI["blue"]]
cmap = plt.get_cmap("cividis"); norm = plt.Normalize(0, 0.9)
SPAN = 13.0
FH1 = 9.4
coh_l3 = float(wt.l3.mean())
put("coh_l3", coh_l3, 1)
fig = plt.figure(figsize=(FIGW, FH1))
fig.text(0.01, 0.99, "Plus-bend EDGEs do not rush wider on Sundays", fontsize=18, weight="bold", va="top")
fig.text(0.01, 0.948, "Each row: Combine hoop run (colour = sideways g; ring = the half-second that sets Hoop G) | "
         f"{NUM['arc_snaps_per_player']} random NFL rushes,\nsnap ▲ to throw or sack | lateral offset 3 yd upfield, + = outside.\n"
         f"Black bar = his mean; dashed = cohort mean {mn(f'{coh_l3:.1f}')} yd, {NUM['wt_share_outside']}% outside paths.",
         fontsize=12, va="top", color="#333333")
hand = [Line2D([], [], color=PC["outside"], lw=3, label="outside path"),
        Line2D([], [], color=PC["inside"], lw=3, label="inside path"),
        Line2D([], [], color=PC["through"], lw=3, label="straight"),
        Line2D([], [], color=PC["unclassified"], lw=3, label="interior snap"),
        Line2D([], [], marker="o", color=PRC, lw=0, markersize=9, label="pressure")]
fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.005, 0.868), ncol=5, fontsize=12, frameon=False,
           handlelength=1.4, columnspacing=1.0)
F1 = []
ROWH, TOP = 0.188, 0.815
AH = 0.135
AW = AH * FH1 / FIGW
for idx, nm in enumerate(P4):
    r_ = ba[ba.display_name == nm].iloc[0]
    yt = TOP - idx * ROWH
    ybot = yt - 0.03 - AH
    fig.text(0.01, yt, f"{nm}", fontsize=13.5, weight="bold", va="top", color=HC[idx])
    fig.text(0.01 + 0.0095 * len(nm) + 0.012, yt - 0.002,
             f"{'EDGE' if r_.role == 'EDGE' else 'Int'} '{r_.draft_year % 100:02d} #{pk_txt(r_.pick)} · bend "
             f"{'+' if r_.shr > 0 else '−'}{abs(r_.shr):.2f} sd · pressure {100 * r_.press_rate:.1f}% "
             f"(fcst {100 * r_.pb:.1f}%)", fontsize=11.5, va="top", color="#333333")
    ax = fig.add_axes([0.035, ybot, AW, AH])
    tr = traces[r_.hoop_eid]
    xs_, ys_ = tr["_x"] - (tr["_x"].max() + tr["_x"].min()) / 2, tr["_y"] - (tr["_y"].max() + tr["_y"].min()) / 2
    if np.ptp(ys_) < np.ptp(xs_):
        xs_, ys_ = ys_, xs_
    pts = np.c_[xs_, ys_].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=norm, linewidth=3.0)
    lc.set_array(tr["_alat"][:-1]); ax.add_collection(lc)
    ax.scatter([xs_[tr["_k"]]], [ys_[tr["_k"]]], s=120, facecolor="none", edgecolor="k", lw=2.0, zorder=3)
    ax.set_xlim(-SPAN / 2, SPAN / 2); ax.set_ylim(-SPAN / 2, SPAN / 2); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.03, 0.97, f"hoop\n{r_.hoop_g:.2f} g", transform=ax.transAxes, fontsize=11.5, weight="bold", va="top")
    gx = fig.add_axes([0.035 + AW + 0.01, ybot, AW, AH])
    q_ = arcp[(arcp.nfl_id == r_.nfl_id) & (arcp.kind == "random")]
    for kk in q_.itertuples(index=False):
        lat, fwd, al, kmx, wty = arc_tr[(kk.game_id, kk.play_id, kk.nfl_id)]
        gx.plot(lat, fwd, color=PC[wty], lw=1.8 if wty != "unclassified" else 1.2, alpha=0.9, zorder=2)
        if kk.press == 1:
            gx.scatter([lat[-1]], [fwd[-1]], s=40, color=PRC, zorder=4, edgecolor="white", lw=0.6)
    gx.scatter([0], [0], marker="^", color=OI["black"], s=45, zorder=5)
    gx.set_xlim(-8.5, 4.5); gx.set_ylim(-1, 12); gx.set_aspect("equal")
    gx.set_xticks([]); gx.set_yticks([])
    gx.text(0.03, 0.97, f"NFL\n{q_.arc_g.mean():.2f} g", transform=gx.transAxes, fontsize=11.5, weight="bold", va="top")
    ix = fig.add_axes([0.035 + 2 * AW + 0.055, ybot, 0.06, AH])
    cl_ = q_[q_.wtype != "unclassified"].dropna(subset=["l3"])
    ix.axhline(0, color="#BBBBBB", lw=0.8)
    if len(cl_) >= 5:
        ix.axhline(coh_l3, color="#555555", lw=1.4, ls="--", zorder=2)
        jit = np.random.default_rng(idx).uniform(-0.25, 0.25, len(cl_))
        ix.scatter(jit, cl_.l3.clip(-6, 4), c=[PC[w] for w in cl_.wtype], s=26, edgecolor="white", lw=0.4, zorder=3)
        ix.plot([-0.4, 0.4], [cl_.l3.mean()] * 2, color="k", lw=2.4, zorder=4)
        ix.axhspan(0.5, 4, color=OI["blue"], alpha=0.07, lw=0)
        ix.text(1.12, 0.5, f"{100 * (cl_.wtype == 'outside').mean():.0f}%\nout", transform=ix.transAxes, fontsize=12.5,
                weight="bold", color=OI["blue"], va="center")
    else:
        ix.text(0.5, 0.5, "interior:\nnot\nclassified", transform=ix.transAxes, ha="center", va="center", fontsize=11,
                color="#555555")
    ix.set_ylim(-6, 4); ix.set_xlim(-0.6, 0.6); ix.set_xticks([])
    ix.set_yticks([-4, 0, 4]); ix.tick_params(labelsize=11, length=2, pad=1)
    ix.set_ylabel("yd out", fontsize=11, labelpad=0)
    for a_ in (ax, gx, ix):
        for sp_ in a_.spines.values():
            sp_.set_visible(True); sp_.set_color("#BBBBBB")
    key = nm.split(" ")[-1].lower()
    F1.append(dict(nm=nm, hoop=r_.hoop_g, arc=q_.arc_g.mean(), id=r_.nfl_id))
    put(f"f1_{key}_arc", q_.arc_g.mean())
    if len(cl_) >= 5:
        put(f"f1_{key}_out", 100 * (cl_.wtype == "outside").mean(), 0)
cax = fig.add_axes([0.035, 0.035, 0.16, 0.012])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[0, 0.3, 0.6, 0.9])
cb.ax.tick_params(labelsize=11); cax.set_title("hoop sideways g", fontsize=11.5, loc="left", pad=3)
# right: drill bend vs game bend, every DL with both measures
sc = fig.add_axes([0.72, 0.12, 0.265, 0.6])
pool = dl.dropna(subset=["hoop_g", "arc_random"])
put("strip_n", len(pool))
sc.scatter(pool.hoop_g, pool.arc_random, s=26, color="#AAAAAA", edgecolor="white", lw=0.3, zorder=1)
OFFS = [((10, -30), "right"), ((-6, 30), "right"), ((-30, -48), "right"), ((10, 4), "left")]
for f_, hc, (off_, ha_) in zip(F1, HC, OFFS):
    sc.scatter([f_["hoop"]], [f_["arc"]], s=130, color=hc, edgecolor="k", lw=0.8, zorder=3)
    sc.annotate(name_short(f_["nm"]), (f_["hoop"], f_["arc"]), xytext=off_, textcoords="offset points", ha=ha_,
                fontsize=12, color=hc, weight="bold", arrowprops=dict(arrowstyle="-", color=hc, lw=0.8))
sc.set_xlabel("Combine Hoop G (g)", fontsize=12.5); sc.set_ylabel("NFL rush-path bend (g)", fontsize=12.5)
sc.tick_params(labelsize=11.5)
sc.set_title(f"All {len(pool)} DL: drill bend\nbarely predicts\ngame bend", fontsize=13.5, weight="bold", loc="left")
sc.text(0.97, 0.03, f"ρ = {NUM['pr_hoop_arc']}\nafter weight, role", transform=sc.transAxes, fontsize=12.5, ha="right",
        weight="bold", color="#333333")
save(fig, "fig1_four_rushers.png")

# ---- Fig 2: signal map (3-cone dropped: n too small)
FIG2 = [m_[0] for m_ in MEAS]
labs = {m_[0]: m_[1] for m_ in MEAS}
OUTL = dict(OUTC)
fig = plt.figure(figsize=(FIGW, 5.4))
gs2 = fig.add_gridspec(1, 3, left=0.27, right=0.99, top=0.78, bottom=0.3, wspace=0.08)
axes = [fig.add_subplot(gs2[0, i]) for i in range(3)]
yy = np.arange(len(FIG2))[::-1]
for ax, (oc, _) in zip(axes, OUTC):
    d_ = sig[sig.outcome == oc].set_index("col").loc[FIG2]
    for yv, (c, r_) in zip(yy, d_.iterrows()):
        ax.fill_betweenx([yv - 0.42, yv + 0.42], -r_.mde, r_.mde, color="#E6E6E6", lw=0)
        colr = OI["green"] if r_.kind == "tracking" else OI["black"]
        ax.plot([r_.lo, r_.hi], [yv + 0.12, yv + 0.12], color=colr, lw=2)
        ax.scatter([r_.rho], [yv + 0.12], color=colr, s=60, zorder=3)
        ax.text(0.99, yv + 0.3, f"n={int(r_.n)}", transform=ax.get_yaxis_transform(), ha="right", fontsize=10.5,
                color="#555555")
        if (c, oc) in AFTER:
            ra_ = AFTER[(c, oc)][0]
            ax.annotate("", xy=(ra_, yv - 0.25), xytext=(r_.rho, yv - 0.25),
                        arrowprops=dict(arrowstyle="->", color=OI["purple"], lw=1.8))
            ax.scatter([ra_], [yv - 0.25], marker="D", facecolor="white", edgecolor=OI["purple"], s=60, zorder=4, lw=1.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-0.6, 1.0); ax.set_ylim(-0.75, len(FIG2) - 0.4)
    ax.set_title(OUTL[oc], fontsize=12.5, weight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels([labs[c] for c in FIG2] if ax is axes[0] else [], fontsize=12)
    ax.set_xticks([-0.5, 0, 0.5, 1.0]); ax.set_xlabel("Spearman ρ", fontsize=11.5)
hand = [Line2D([], [], color=OI["black"], marker="o", lw=2, label="stopwatch / draft, raw (95% CI)"),
        Line2D([], [], color=OI["green"], marker="o", lw=2, label="tracked drill, raw (95% CI)"),
        Line2D([], [], color=OI["purple"], marker="D", markerfacecolor="white", lw=0,
               label="tracked drill after pick, weight, NGS, role\n(rush-path bend: after weight and role)"),
        Patch(color="#E6E6E6", label="too small to detect at this row's n (80% power)")]
fig.legend(handles=hand, loc="lower center", ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.55, 0.0))
fig.suptitle("Burst carries from drill to game but not to pressure; bend keeps a small pressure link after the baseline",
             fontsize=13.5, weight="bold", x=0.01, ha="left", y=0.97)
fig.text(0.01, 0.9, "Better Combine number → + ρ. Pressure and get-off: DL with 100+ NFL rushes who have the measure; "
         "rush-path bend: all DL with sampled snaps.\nn differs by row: compare each point with its own grey band, "
         "not with the other rows.", fontsize=10.5, color="#333333", va="top")
for c in FIG2:
    put(f"n_sig_{c}", int(sig[(sig.outcome == "press_eb") & (sig.col == c)].n.iloc[0]))
save(fig, "fig2_signal_map.png")

# ---- Fig 3: the headline under three priors | held-out R2 change | held-out log predictive density
fig = plt.figure(figsize=(FIGW, 5.8))
g3 = fig.add_gridspec(1, 3, left=0.185, right=0.985, top=0.73, bottom=0.2, wspace=0.75)
fa, fb, fc_ = (fig.add_subplot(g3[i]) for i in range(3))


def dotint(ax, y, v, colr, pt=None, lw=(7, 2)):
    a95, a80 = np.percentile(v, [2.5, 97.5]), np.percentile(v, [10, 90])
    ax.plot(a95, [y, y], color=colr, lw=lw[1], solid_capstyle="butt")
    ax.plot(a80, [y, y], color=colr, lw=lw[0], alpha=0.45, solid_capstyle="butt")
    ax.scatter([np.median(v) if pt is None else pt], [y], color=colr, s=60, zorder=3, edgecolor="white")


PRL = [("skeptical", "Skeptical prior\n(headline)"), ("double", "2× skeptical"), ("wide", "Wide prior")]
for j, (tg, lab) in enumerate(PRL):
    dotint(fa, 2 - j, PRIOR_DRAWS[tg], OI["blue"] if j == 0 else OI["sky"])
fa.axvline(0, color="k", lw=0.8)
fa.set_yticks([2, 1, 0]); fa.set_yticklabels([l for _, l in PRL], fontsize=11.5)
fa.set_ylim(-0.6, 2.6); fa.set_xlabel("extra pressures per 450-rush\nseason, per sd of bend", fontsize=11.5)
fa.set_title(f"In sample, by prior (n = {NUM['bayes_n']})", fontsize=12.5, weight="bold", loc="left")
labs4 = ["Pooled"] + [str(c) for c in classes]
for j, lab in enumerate(labs4):
    v = NFB[:, j]
    pt = float(NUM["nf_dr2"]) if j == 0 else float(NUM[f"nf_dr2_{classes[j - 1]}"])
    dotint(fb, len(labs4) - 1 - j, v, OI["black"] if j == 0 else OI["green"], pt=pt)
    nn_ = NFI[0]["n"] if j == 0 else NFI[0][f"n{classes[j - 1]}"]
    labs4[j] = f"{lab} (n={nn_})"
lo3 = np.floor(np.percentile(NFB[:, 1:], 2.5, axis=0).min() * 10) / 10
fb.set_xlim(lo3 - 0.05, 0.45)
fb.axvline(0, color="k", lw=0.8)
fb.set_yticks(range(len(labs4))); fb.set_yticklabels(labs4[::-1], fontsize=11.5)
fb.set_ylim(-0.6, len(labs4) - 0.4)
fb.set_xlabel("held-out R² change\n(unbounded below)", fontsize=11.5)
fb.set_title("Out of class: R²", fontsize=12.5, weight="bold", loc="left")
ncl = [int((bz.draft_year == c).sum()) for c in classes]
pw = ELPD["skeptical"]["pw"]
cuts = np.cumsum([0] + ncl)
rowsE = [("Pooled", pw)] + [(str(c), pw[cuts[i]:cuts[i + 1]]) for i, c in enumerate(classes)]
for j, (lab, v) in enumerate(rowsE):
    y = len(rowsE) - 1 - j
    s_, se_ = v.sum(), np.sqrt(len(v)) * v.std(ddof=1)
    colr = OI["black"] if j == 0 else OI["green"]
    fc_.plot([s_ - 1.96 * se_, s_ + 1.96 * se_], [y, y], color=colr, lw=2, solid_capstyle="butt")
    fc_.plot([s_ - 1.28 * se_, s_ + 1.28 * se_], [y, y], color=colr, lw=7, alpha=0.45, solid_capstyle="butt")
    fc_.scatter([s_], [y], color=colr, s=60, zorder=3, edgecolor="white")
fc_.axvline(0, color="k", lw=0.8)
fc_.set_yticks(range(len(rowsE))); fc_.set_yticklabels([f"{l} (n={len(v)})" for l, v in rowsE][::-1], fontsize=11.5)
fc_.set_ylim(-0.6, len(rowsE) - 0.4)
fc_.set_xlabel("held-out log predictive\ndensity gain (nats)", fontsize=11.5)
fc_.set_title("Out of class: log score", fontsize=12.5, weight="bold", loc="left")
fig.suptitle(f"Planning number {NUM['bayes_all']} pressures a season per sd; out of class, no R² gain and a small log-score gain",
             fontsize=14, weight="bold", x=0.01, ha="left", y=0.975)
fig.text(0.01, 0.9, "Dot = estimate; thick bar = 80%, thin = 95%. Left: Bayesian beta-binomial, all hoop runners with an "
         "NFL rush. Middle: fully nested refit (100+ rushes),\nbootstrap of the whole loop. Right: the left model fit on "
         "two classes and scored on the third, minus the same model without bend (skeptical prior).",
         fontsize=10.5, color="#333333", va="top")
save(fig, "fig3_headline.png")

# ---- Fig 4: forecast vs actual for every 2023-25 call, sorted by actual minus forecast
cv = called[called.pb.notna()].copy()
cv["res"] = cv.press_rate - cv.pb
cv["grp"] = np.where(cv.call == "plus", "plus", "other")
panels = [("plus", "Plus calls"), ("other", "Minus (Int) + film first*")]
nrow = max((cv.grp == "plus").sum(), (cv.grp != "plus").sum())
HH = 0.42 * nrow + 3.0
XMAX = float(np.ceil(100 * cv.pr_hi.max() + 1))
fig, axs = plt.subplots(1, 2, figsize=(FIGW, HH), sharex=True)
fig.subplots_adjust(left=0.25, right=0.97, top=1 - 1.6 / HH, bottom=1.25 / HH, wspace=1.15)
for ax, (gk, title) in zip(axs, panels):
    d_ = cv[cv.grp == gk].sort_values("res")
    yy = np.arange(len(d_))
    for yv, r_ in zip(yy, d_.itertuples()):
        clear_up, clear_dn = r_.pr_lo > r_.pb, r_.pr_hi < r_.pb
        c_ = OI["blue"] if clear_up else (OI["red"] if clear_dn else "#999999")
        ax.plot([100 * r_.pr_lo, 100 * r_.pr_hi], [yv, yv], color="#DDDDDD", lw=6, solid_capstyle="round", zorder=1)
        if clear_up or clear_dn:
            ax.annotate("", xy=(100 * r_.press_rate, yv), xytext=(100 * r_.pb, yv),
                        arrowprops=dict(arrowstyle="-|>", color=c_, lw=2.2, mutation_scale=14), zorder=3)
        else:
            ax.plot([100 * r_.pb, 100 * r_.press_rate], [yv, yv], color=c_, lw=2.2, zorder=3)
            ax.scatter([100 * r_.press_rate], [yv], color=c_, s=36, zorder=3)
        ax.scatter([100 * r_.pb], [yv], marker="|", s=170, color="k", lw=2.2, zorder=4)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{name_short(r_.display_name)} '{r_.draft_year % 100:02d} {ROLE_TXT[r_.role]}"
                        + ("*" if r_.call.startswith("film") else "")
                        + f"  {100 * r_.press_rate:.1f}% ({int(r_.rush_snaps)})" for r_ in d_.itertuples()],
                       fontsize=11)
    ax.set_ylim(-0.7, nrow - 0.3); ax.grid(axis="x", color="#EEEEEE", lw=0.6)
    nu, ndn = int((d_.pr_lo > d_.pb).sum()), int((d_.pr_hi < d_.pb).sum())
    ax.set_title(f"{title}\n{nu} clearly above · {ndn} below · {len(d_) - nu - ndn} in noise",
                 fontsize=12.5, weight="bold")
    ax.set_xlim(0, XMAX); ax.set_xlabel("NFL pressure rate, %")
hand = [Line2D([], [], marker="|", color="k", lw=0, markersize=13, markeredgewidth=2.2, label="no-tracking forecast"),
        Line2D([], [], color=OI["blue"], lw=2.2, marker=">", label="clearly above"),
        Line2D([], [], color=OI["red"], lw=2.2, marker=">", label="clearly below"),
        Line2D([], [], color="#999999", lw=2.2, marker="o", label="inside noise"),
        Line2D([], [], color="#DDDDDD", lw=6, label="80% interval of actual")]
fig.legend(handles=hand, frameon=False, fontsize=11, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=5,
           handlelength=1.5, columnspacing=1.0)
fig.suptitle("Every 2023–25 bend call: half land inside noise",
             fontsize=13, weight="bold", x=0.01, ha="left", y=1 - 0.12 / HH)
fig.text(0.01, 1 - 0.55 / HH, "Label: pressure rate (rushes). 'Clearly' = the 80% interval of the actual rate excludes the "
         "out-of-class no-tracking forecast.\nCalls use the drill only; rows sorted by actual minus forecast; 30+ rushes; "
         "2023–24 helped choose the hoop. * = film-first flag.", fontsize=11, color="#333333", va="top")
save(fig, "fig4_calls.png")

# ============================================================================= write
_tick("write")
TABLES["md"] = MD
for k_, v_ in MD_TXT.items():
    MD[k_] = v_
used = render(OUT, NUM, MD)
keep = {k_: NUM[k_] for k_ in sorted(used) if k_ in NUM}
for k_ in list(keep):
    if str(keep[k_]).startswith("-"):
        keep[k_ + "_abs"] = str(keep[k_])[1:]     # check.py reads digits without the sign
json.dump(keep, open(OUT / "numbers.json", "w"), indent=1, sort_keys=True)
json.dump(TABLES, open(OUT / "tables.json", "w"), indent=1, default=str)
if os.environ.get("BDB_ALLNUM"):
    json.dump(NUM, open(os.environ["BDB_ALLNUM"], "w"), indent=1, sort_keys=True)
_tick("done")

