"""Hoop G (Run-the-Hoop bend) vs NFL pass-rush pressure for DL/EDGE: one metric, tested honestly.

Run from the repo root:
    uv run python writeup/candidates/round_07/revised/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (only the values quoted in writeup.md), tables.json,
board_2023_2025.csv and figures/*.png next to this file, then renders writeup.md from writeup_template.md.
Frame-level feature extraction is cached in data/cache/ (gitignored; delete it to recompute from the frames).
Runtime: about 1 minute cached, about 6 minutes cold, one core.

Design (round 7 revised, from round 7 cand_1). New in this revision: within-class label-shuffle negative control
(BDB_NPERM reruns of the whole nested pipeline, 4 processes), rush-weighted nested R² on all 90 as the out-of-class
headline, observed-times-only models on a reduced baseline, run-to-run reliability from repeat runs, a common-n
comparison with the 3-cone and 10-yd split, and film-first flags dropped. Earlier design notes:
  * Baseline without tracking = log pick + official 10-yd split + 3-cone + NGS athleticism score + weight + NFL EDGE
    share (+ games per season and a missing-time flag in the count model).
  * Headline: Bayesian beta-binomial model of pressures out of pass rushes for every hoop runner with an NFL rush,
    class random intercept, separate EDGE / interior Hoop G slopes, Hoop G measurement error (one-run reliability)
    integrated by Gauss-Hermite quadrature, slope prior scaled to the typical effect of 14 screened pairs.
    Adaptive Metropolis, 4 chains. Missing times: deterministic fill from weight, role and NGS + flag.
  * Out of class, two fully nested tests (everything chosen inside each training fold):
      - R² change: impute, screen 14 drill-outcome pairs, pick best, reliability, shrink; least squares and ridge.
      - Log score (new in round 7): the screen, the chosen drill feature, its reliability and the prior scale are all
        set on the two training classes; the beta-binomial with and without that feature is scored on the third.
  * Checks: three prior scales, posterior-predictive check, cross-fitted per-class estimates, observed-times-only,
    multiple imputation, leave-one-out, placebo outcome and drill, two in-game path tests.
  * numbers.json holds exactly the numbers the text uses; claims.json maps each key to the sentence that uses it.
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
    raw = tpl.read_text()
    claims = {}
    for m in re.finditer(r"\{\{TABLE:(\w+)\}\}", raw):
        for k in re.findall(r"\{\{(\w+)\}\}", md_tables[m.group(1)]):
            claims.setdefault(k, f"table/sentence '{m.group(1)}'")
    text = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda m: md_tables[m.group(1)], raw)
    used = set()
    for line in text.splitlines():
        for k in re.findall(r"\{\{(\w+)\}\}", line):
            if k not in claims:
                claims[k] = re.sub(r"\{\{(\w+)\}\}", lambda m: str(num.get(m.group(1), "?")), line).strip()[:160]

    def sub(m):
        k = m.group(1)
        if k not in num:
            raise KeyError(f"writeup_template.md uses unknown key {k}")
        used.add(k)
        v = str(num[k])
        return "−" + v[1:] if v.startswith("-") else v
    (out_dir / "writeup.md").write_text(re.sub(r"\{\{(\w+)\}\}", sub, text))
    json.dump({k: {"value": num[k], "claim": claims.get(k, "")} for k in sorted(used)},
              open(out_dir / "claims.json", "w"), indent=1, ensure_ascii=False)
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
# direct check of the one-run reliability: run-to-run agreement among players with 2+ hoop runs
hoop_att["hz_tmp"] = yz(hoop_att.bend_g, hoop_att.draft_year)
REP = hoop_att[hoop_att.groupby("nfl_id").nfl_id.transform("size") >= 2][["nfl_id", "hz_tmp"]]
rel_icc, n_rep = icc(REP.nfl_id, REP.hz_tmp)
put("rel_icc", rel_icc); put("rel_icc_n", n_rep)
_rid = REP.nfl_id.unique(); _bi = []; _irng = np.random.default_rng(41)
for b_ in range(NBOOT):
    pk_ = _irng.choice(_rid, len(_rid))
    d_ = pd.concat([REP[REP.nfl_id == p_].assign(nfl_id=j_) for j_, p_ in enumerate(pk_)])
    try:
        _bi.append(icc(d_.nfl_id, d_.hz_tmp)[0])
    except Exception:
        pass
put("rel_icc_lo", np.nanpercentile(_bi, 5)); put("rel_icc_hi", np.nanpercentile(_bi, 95))

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
SCREEN = [("hoop_z", "Hoop G"), ("hoop_speed_z", "Hoop mean speed"), ("hoop_time_z", "Hoop lap time"),
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
        ("log_pick", "Earlier draft pick", "official", -1), ("d1_z", "Rush-drill burst", "tracking", 1),
        ("hoop_z", "Hoop G", "tracking", 1)]
OUTC = [("press_eb", "NFL pressure rate"), ("quick", "NFL get-off (quicker)"), ("arc_random", "NFL rush-path sideways g")]
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
put("n_hoop_arc", len(dl.dropna(subset=["hoop_z", "arc_random"])))
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
    dl = dl.merge(a_.groupby("nfl_id").size().rename(f_ + "_k").reset_index(), on="nfl_id", how="left")
HSH =hoop_att[["nfl_id", "bend_g_h1", "bend_g_h2"]].dropna()


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


def _nested_r2():
    imp_rng, brng = np.random.default_rng(7), np.random.default_rng(11)
    nfi = [nested_full(cand, base=BASE_T, imp_rng=imp_rng) for _ in range(10)]
    nfb = []
    for _ in range(NLOCO):
        d_ = pd.concat([g_.iloc[brng.integers(0, len(g_), len(g_))] for g_ in gcand.values()])
        try:
            o_ = nested_full(d_, base=BASE_T, imp_rng=imp_rng)
            nfb.append([o_["pooled"]] + [o_[int(c)] for c in classes])
        except Exception:
            pass
    return nfi, np.array(nfb)


gcand = {c: d for c, d in cand.groupby("draft_year")}
NFI, NFB = cached(f"nested_r2_{NLOCO}", _nested_r2)
put("nf_dr2", np.mean([o_["pooled"] for o_ in NFI]), 3); put("nf_n", NFI[0]["n"])
for c in classes:
    put(f"nf_dr2_{c}", np.mean([o_[int(c)] for o_ in NFI]), 3)
    put(f"nf_n_{c}", NFI[0][f"n{int(c)}"])
SELS = [s_ for o_ in NFI for s_ in o_["sel"]]
put("nf_hoop_sel_pct", 100 * np.mean([s_ == "hoop_z|press_eb" for s_ in SELS]), 0)
BRNG = np.random.default_rng(12)
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

    def _ob(gs_=gs_, base_=base_, seed=len(tg)):
        rng_, bb = np.random.default_rng(100 + seed), []
        for _ in range(NLOCO):
            d_ = pd.concat([g_.iloc[rng_.integers(0, len(g_), len(g_))] for g_ in gs_.values()])
            try:
                bb.append(nested_full(d_, base=base_)["pooled"])
            except Exception:
                pass
        return bb
    bb_ = cached(f"obs_{tg}_{NLOCO}", _ob)
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


def make_model(d, cols, ref=None, feat="hoop_z", kcol="hoop_k", rel=None):
    """Arrays for the beta-binomial likelihood on rows d with (standardized) covariates cols.
    ref: frame whose means / sds standardize the covariates (training rows for out-of-class prediction).
    feat / kcol / rel: the drill feature (year z), its number of runs and its one-run reliability."""
    ref = d if ref is None else ref
    rel = REL if rel is None else rel
    Zr = d[cols]
    sd = ref[cols].std().replace(0, np.nan)
    keep = sd.notna().values
    Zm = ((Zr - ref[cols].mean()) / sd).values[:, keep]
    y, n = d.pressures.values.astype(float), d.rush_snaps.values.astype(float)
    x, k = d[feat].values, d[kcol].fillna(1).values
    lam = rel / (rel + (1 - rel) / k)
    mS, sS = lam * x / np.sqrt(rel), np.sqrt(lam * (1 - rel) / k / rel)
    return dict(Z=Zm, NZ=Zm.shape[1], edge=(d.edge_share >= 0.5).values, cls=pd.Categorical(d.draft_year).codes,
                y=y, n=n, mS=mS, sS=sS, rel=rel, Tn=mS[:, None] + sS[:, None] * ghx[None, :],
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
        d_ = 450 * (expit(eta + sl * (M["mS"] + np.sqrt(M["rel"]))) - expit(eta + sl * M["mS"]))
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


_tick("held-out log score, fully nested")


def prep_rows(d, ref):
    """Count-model rows: pressures, log games per season, official times filled from ref rows (+ missing flag)."""
    d = d.copy()
    d["pressures"] = d.pressures.fillna(0)
    d["lgps"] = np.log1p(d.games_per_season)
    for c_ in TIMES:
        d[c_ + "_m"] = d[c_].isna().astype(float)
        ok_ = ref[c_].notna()
        A_ = np.c_[np.ones(ok_.sum()), ref.loc[ok_, FILLX].values]
        cf_ = np.linalg.lstsq(A_, ref.loc[ok_, c_].values, rcond=None)[0]
        d[c_ + "_f"] = np.where(d[c_].notna(), d[c_], np.c_[np.ones(len(d)), d[FILLX].values] @ cf_)
    return d


def screen_pick(trc):
    """The 14-pair screen on training rows only: best drill feature by |partial rho|, its outcome, typical |rho|."""
    Sp, Sg = screen_setup(trc, "press_eb", MAIN), screen_setup(trc, "quick", MAIN)
    st_ = {("press_eb", k): v for k, v in screen_stats(Sp, trc.press_eb).items()}
    st_.update({("quick", k): v for k, v in screen_stats(Sg, trc.quick).items()})
    keys = list(st_)
    rr_ = np.array([st_[k] for k in keys])
    oc_, best = keys[int(np.argmax(np.abs(rr_)))]
    return best, oc_, float(np.mean(np.abs(rr_)))


def screen_fac(trc):
    """Empirical-Bayes selection shrink factor for the screen winner (same rule as nested_full)."""
    Sp, Sg = screen_setup(trc, "press_eb", MAIN), screen_setup(trc, "quick", MAIN)
    st_ = {("press_eb", k): v for k, v in screen_stats(Sp, trc.press_eb).items()}
    st_.update({("quick", k): v for k, v in screen_stats(Sg, trc.quick).items()})
    keys = list(st_)
    r_ = np.array([st_[k] for k in keys])
    nn_ = np.array([len((Sp if k[0] == "press_eb" else Sg)[k[1]]["idx"]) for k in keys])
    zz = np.arctanh(np.clip(r_, -0.99, 0.99)); se2_ = 1 / (nn_ - 3 - len(MAIN))
    mz_ = np.average(zz, weights=1 / se2_); t2_ = max(0.0, np.var(zz, ddof=1) - se2_.mean())
    j = int(np.argmax(np.abs(r_)))
    return float(np.tanh(mz_ + t2_ / (t2_ + se2_[j]) * (zz[j] - mz_)) / r_[j])


def fold_setup(cand_, pool_all, c):
    """Everything chosen on the two training classes: screen winner, its reliability, the prior scale."""
    trc = cand_[cand_.draft_year != c]
    best, oc_, rho_f = screen_pick(trc)
    pool_ = pool_all[(pool_all.rush_snaps > 0) & pool_all[best].notna()]
    tr_raw, te_raw = pool_[pool_.draft_year != c], pool_[pool_.draft_year == c]
    rel_f = feat_rel(best, set(tr_raw.nfl_id))
    # prior scale from training rows only: typical |rho| x (pp per unit rho) / logit slope factor / sqrt(rel)
    d100 = trc.dropna(subset=[best]).assign(pp=lambda d: 100 * d.press_eb)
    A_ = np.c_[np.ones(len(d100)), d100[MAIN].values]
    ry_ = d100.pp.values - A_ @ np.linalg.lstsq(A_, d100.pp.values, rcond=None)[0]
    rx_ = d100[best].values - A_ @ np.linalg.lstsq(A_, d100[best].values, rcond=None)[0]
    pbar_f = float(trc.press_rate.mean())
    pb_f = rho_f * (ry_.std() / rx_.std()) / (100 * pbar_f * (1 - pbar_f)) / np.sqrt(rel_f)
    return dict(trc=trc, best=best, oc=oc_, rho=rho_f, rel=rel_f, prior=pb_f,
                tr=prep_rows(tr_raw, tr_raw), te=prep_rows(te_raw, tr_raw))


def nested_lpd(cand_, pool_all, cols=ZCT, priors=("skeptical", "double", "wide"), ndraw=800, base_cache=None):
    """Fully nested held-out log predictive density gain (drill model minus the same model without the drill)."""
    out = {tg: dict(pw=[], cls={}, n={}) for tg in priors}
    sel = {}
    for c in classes:
        F = fold_setup(cand_, pool_all, c)
        tr_, te_, best = F["tr"], F["te"], F["best"]
        sel[int(c)] = dict(feat=best, outcome=F["oc"], rel=F["rel"], prior=F["prior"], rho_typ=F["rho"])
        kw = dict(feat=best, kcol=best + "_k", rel=F["rel"])
        Mtr, Mte = make_model(tr_, cols, **kw), make_model(te_, cols, ref=tr_, **kw)
        ck = (int(c), best, len(tr_), len(te_))
        if base_cache is not None and ck in base_cache:
            l0 = base_cache[ck]
        else:
            l0 = pointwise_lpd(laplace_draws(Mtr, 1e-4, ndraw, 50 + int(c) % 10), Mte)
            if base_cache is not None:
                base_cache[ck] = l0
        for tg in priors:
            pbx = {"skeptical": F["prior"], "double": 2 * F["prior"], "wide": 0.25}[tg]
            lh = pointwise_lpd(laplace_draws(Mtr, pbx, ndraw, 40 + int(c) % 10), Mte)
            out[tg]["pw"].append(lh - l0)
            out[tg]["cls"][int(c)] = float((lh - l0).sum())
            out[tg]["n"][int(c)] = len(te_)
    for tg in priors:
        dd = np.concatenate(out[tg]["pw"])
        out[tg]["sum"], out[tg]["se"] = float(dd.sum()), float(np.sqrt(len(dd)) * dd.std(ddof=1))
    return out, sel


def wls_pred(tr, te, cols, xcol=None, fac=1.0):
    """Rush-weighted least squares of the pressure rate; standardized covariates (constant columns dropped)."""
    cc = [c_ for c_ in cols if tr[c_].std() > 0]
    mu_, sd_ = tr[cc].mean(), tr[cc].std()
    X = lambda d: np.c_[np.ones(len(d)), ((d[cc] - mu_) / sd_).values] if xcol is None else \
        np.c_[np.ones(len(d)), ((d[cc] - mu_) / sd_).values, d[xcol].values]
    w = tr.rush_snaps.values
    A = X(tr) * np.sqrt(w)[:, None]
    b = np.linalg.lstsq(A, (tr.pressures / tr.rush_snaps).values * np.sqrt(w), rcond=None)[0]
    if xcol is not None:
        b[-1] *= fac
    return X(te) @ b


def wr2(y, p, w):
    m_ = np.average(y, weights=w)
    return 1 - (w * (y - p) ** 2).sum() / (w * (y - m_) ** 2).sum()


def nested_w(cand_, pool_all, cols=ZCT):
    """Fully nested, rush-weighted R² change on every runner with an NFL rush (same rows as the count model)."""
    Y, W, PA, PB = {}, {}, {}, {}
    for c in classes:
        F = fold_setup(cand_, pool_all, c)
        best, rel_f = F["best"], F["rel"]
        fac = screen_fac(F["trc"])
        tr_, te_ = F["tr"].copy(), F["te"].copy()
        for d_ in (tr_, te_):
            k_ = d_[best + "_k"].fillna(1).values
            d_["_x"] = k_ * rel_f / (1 + (k_ - 1) * rel_f) * d_[best].values
        PA[c] = wls_pred(tr_, te_, cols, "_x", fac)
        PB[c] = wls_pred(tr_, te_, cols)
        Y[c], W[c] = (te_.pressures / te_.rush_snaps).values, te_.rush_snaps.values
    cat = lambda D: np.concatenate([D[c] for c in classes])
    out = {"pooled": wr2(cat(Y), cat(PA), cat(W)) - wr2(cat(Y), cat(PB), cat(W)), "n": int(sum(len(Y[c]) for c in classes))}
    for c in classes:
        out[int(c)] = wr2(Y[c], PA[c], W[c]) - wr2(Y[c], PB[c], W[c])
    out["_arr"] = (Y, W, PA, PB)
    return out


NESTED, NSEL = nested_lpd(cand, dl)
for c in classes:
    _tick(f"  fold {c}: {NSEL[int(c)]['feat']} via {NSEL[int(c)]['outcome']}, rel {NSEL[int(c)]['rel']:.2f}, "
          f"prior {NSEL[int(c)]['prior']:.3f}, gain {NESTED['skeptical']['cls'][int(c)]:+.2f}")
for tg, _ in PRIORS:
    dd = np.concatenate(NESTED[tg]["pw"])
    NESTED[tg]["sum"], NESTED[tg]["se"] = float(dd.sum()), float(np.sqrt(len(dd)) * dd.std(ddof=1))
    put(f"nlpd_{tg}", dd.sum(), 2); put(f"nlpd_{tg}_se", np.sqrt(len(dd)) * dd.std(ddof=1), 2)
    put(f"nlpd_{tg}_lr", np.exp(dd.sum()), 1)
    for c in classes:
        put(f"nlpd_{tg}_{c}", NESTED[tg]["cls"][int(c)], 2)
    put(f"nlpd_{tg}_pos", int(sum(v > 0 for v in NESTED[tg]["cls"].values())))
put("nlpd_n", int(sum(NESTED["skeptical"]["n"].values())))
for c in classes:
    put(f"nlpd_n_{c}", NESTED["skeptical"]["n"][int(c)])
put("nlpd_hoop_folds", int(sum(v["feat"] == "hoop_z" for v in NSEL.values())))
TABLES["nested_lpd"] = {"sel": NSEL, **{k: {kk: vv for kk, vv in v.items() if kk != "pw"} for k, v in NESTED.items()}}
# the global prior scale computed with the same fold rule (residual-sd version), for comparison with PRIOR_B
_A = np.c_[np.ones(len(dH)), dH[MAIN].values]
_ry = 100 * dH.press_eb.values - _A @ np.linalg.lstsq(_A, 100 * dH.press_eb.values, rcond=None)[0]
_rx = dH.hoop_z.values - _A @ np.linalg.lstsq(_A, dH.hoop_z.values, rcond=None)[0]
TABLES["prior_rule_check"] = {"PRIOR_B": PRIOR_B, "resid_sd_rule": rho_typ * (_ry.std() / _rx.std()) /
                              (100 * pbar * (1 - pbar)) / np.sqrt(REL)}
for i_, c in enumerate(classes):
    pw_ = NESTED["skeptical"]["pw"][i_]
    put(f"nlpd_skeptical_{c}_se", np.sqrt(len(pw_)) * pw_.std(ddof=1), 2)

# ---- rush-weighted, fully nested R² change on the same rows as the count model (every runner with a rush)
_tick("weighted nested R2")
WN = nested_w(cand, dl)
put("wr2", WN["pooled"], 3); put("wr2_n", WN["n"])
for c in classes:
    put(f"wr2_{c}", WN[int(c)], 3)
# 90% intervals over held-out rushers (fold models fixed), within class and pooled
_Y, _W, _PA, _PB = WN["_arr"]; _wrng = np.random.default_rng(19); WB = {c: [] for c in list(classes) + ["pooled"]}
for _ in range(2000):
    ii = {c: _wrng.integers(0, len(_Y[c]), len(_Y[c])) for c in classes}
    for c in classes:
        WB[c].append(wr2(_Y[c][ii[c]], _PA[c][ii[c]], _W[c][ii[c]]) - wr2(_Y[c][ii[c]], _PB[c][ii[c]], _W[c][ii[c]]))
    cat_ = lambda D: np.concatenate([D[c][ii[c]] for c in classes])
    WB["pooled"].append(wr2(cat_(_Y), cat_(_PA), cat_(_W)) - wr2(cat_(_Y), cat_(_PB), cat_(_W)))
put("wr2_lo", np.percentile(WB["pooled"], 5), 3); put("wr2_hi", np.percentile(WB["pooled"], 95), 3)
for c in classes:
    put(f"wr2_{c}_lo", np.percentile(WB[c], 5), 3); put(f"wr2_{c}_hi", np.percentile(WB[c], 95), 3)
    put(f"wr2_n_{c}", len(_Y[c]))


# ---- negative control: shuffle Hoop G across players within class, rerun the whole nested pipeline
#      (14-drill screen, chosen drill, its reliability, prior scale) for both the held-out log score and the R²
NPERM = int(os.environ.get("BDB_NPERM", 300))
NDRAW_NULL = 300
HCOLS = ["hoop_z", "hoop_z_k", "hoop_k"]


def permuted(seed):
    rng_ = np.random.default_rng(seed)
    hv = dl[dl.hoop_z.notna()][["nfl_id", "draft_year"] + HCOLS]
    new = []
    for _, g_ in hv.groupby("draft_year"):
        p_ = rng_.permutation(len(g_))
        new.append(pd.DataFrame(g_[HCOLS].values[p_], columns=HCOLS).assign(nfl_id=g_.nfl_id.values))
    new = pd.concat(new).set_index("nfl_id")
    out_ = []
    for d_ in (cand, dl):
        d_ = d_.copy()
        m_ = d_.nfl_id.isin(new.index)
        for h_ in HCOLS:
            d_.loc[m_, h_] = new.loc[d_.loc[m_, "nfl_id"], h_].values
        out_.append(d_)
    return out_


_BC: dict = {}


def null_stat(seed):
    if seed is None:
        c_, d_ = cand, dl
    else:
        c_, d_ = permuted(seed)
    o_, s_ = nested_lpd(c_, d_, priors=("skeptical",), ndraw=NDRAW_NULL, base_cache=_BC)
    return (o_["skeptical"]["sum"], nested_w(c_, d_)["pooled"],
            sum(v["feat"] == "hoop_z" for v in s_.values()))


def _null():
    import multiprocessing as mp
    obs_ = null_stat(None)
    with mp.get_context("fork").Pool(min(4, os.cpu_count() or 1)) as pool:
        res_ = pool.map(null_stat, list(range(1000, 1000 + NPERM)), chunksize=4)
    return obs_, np.array(res_)


_tick(f"label-shuffle null ({NPERM} reruns)")
NULL_OBS, NULL = cached(f"null_{NPERM}_{NDRAW_NULL}", _null)
put("perm_n", NPERM)
put("perm_p_lpd", (1 + (NULL[:, 0] >= NULL_OBS[0]).sum()) / (1 + len(NULL)))
put("perm_p_r2", (1 + (NULL[:, 1] >= NULL_OBS[1]).sum()) / (1 + len(NULL)))
put("perm_lpd_med", np.median(NULL[:, 0])); put("perm_lpd_q95", np.percentile(NULL[:, 0], 95))
put("perm_r2_q95", np.percentile(NULL[:, 1], 95), 3)
put("perm_hoop_pct", 100 * np.mean(NULL[:, 2] >= 1), 0)
_h = NULL[:, 2] >= 1        # shuffles in which the shuffled Hoop G still won the screen in at least one fold
put("perm_h_n", int(_h.sum()))
put("perm_p_r2_h", (1 + (NULL[_h, 1] >= NULL_OBS[1]).sum()) / (1 + _h.sum()))
put("perm_p_lpd_h", (1 + (NULL[_h, 0] >= NULL_OBS[0]).sum()) / (1 + _h.sum()))
put("perm_r2_med", np.median(NULL[:, 1]), 3); put("perm_r2_h_med", np.median(NULL[_h, 1]), 3)
put("perm_lpd_h_med", np.median(NULL[_h, 0]))
TABLES["null"] = dict(obs=list(NULL_OBS), lpd=list(np.round(NULL[:, 0], 4)), r2=list(np.round(NULL[:, 1], 4)))

# ---- observed times only, reduced baseline (pick, weight, role, NGS + the observed time): no filled-in times
_tick("observed-only models")
RED = ["log_pick", "combine_weight_z", "edge_share", "ngs_athleticism_score"]
for tc, tg in [("ten_yd_split_z", "oo10"), ("three_cone_z", "oo3")]:
    c_ = cand[cand[tc].notna()]
    d_ = dl[dl[tc].notna()]
    o_, s_ = nested_lpd(c_, d_, cols=RED + [tc], priors=("skeptical",))
    w_ = nested_w(c_, d_, cols=RED + [tc])
    put(f"{tg}_lpd", o_["skeptical"]["sum"]); put(f"{tg}_lpd_se", o_["skeptical"]["se"])
    put(f"{tg}_wr2", w_["pooled"], 3); put(f"{tg}_n", w_["n"])
    bzo = bz[bz[tc].notna()]
    summ(laplace(make_model(bzo, RED + [tc]), PRIOR_B, seed=8 + len(tg))[:, 0], f"{tg}_t")

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
loo = np.array(cached(f"loo_{len(bz)}", lambda: [laplace(make_model(bz.drop(bz.index[i]), ZCT), PRIOR_B,
                                                                  draws=False)[0] for i in range(len(bz))]))
put("loo_full", full_pt, 1); put("loo_min", loo.min(), 1); put("loo_max", loo.max(), 1)
TABLES["loo_min_player"] = bz.display_name.iloc[int(np.argmin(loo))]
TABLES["loo_max_player"] = bz.display_name.iloc[int(np.argmax(loo))]

# cross-fitted sign consistency: for each class, the effect estimated without it and inside it alone
SIGN = []
SIGN_DRAWS = {}
for c in classes:
    tr_ = laplace(make_model(bz[bz.draft_year != c], ZCT), PRIOR_B, seed=20 + int(c) % 10)[:, 0]
    te_ = laplace(make_model(bz[bz.draft_year == c], ZCT), PRIOR_B, seed=30 + int(c) % 10)[:, 0]
    SIGN_DRAWS[int(c)] = (tr_, te_)
    put(f"sg_tr_{c}", np.median(tr_), 1); put(f"sg_te_{c}", np.median(te_), 1)
    for tg_, v_ in [("tr", tr_), ("te", te_)]:
        put(f"sg_{tg_}_{c}_lo", np.percentile(v_, 5), 1); put(f"sg_{tg_}_{c}_hi", np.percentile(v_, 95), 1)
    put(f"sg_n_{c}", int((bz.draft_year == c).sum()))
    SIGN.append(dict(c=int(c), tr=np.median(tr_), te=np.median(te_)))
put("sg_same", int(sum(np.sign(r_["tr"]) == np.sign(r_["te"]) for r_ in SIGN)))
put("sg_pos_te", int(sum(r_["te"] > 0 for r_ in SIGN)))

# posterior-predictive check: simulate every rusher's pressure count from the fitted model (Laplace draws, skeptical
# prior, class effects included) and ask whether the observed counts and their spread look like the simulations
_tick("posterior-predictive check")
th = laplace_draws(MH, PRIOR_B, 1000, 77)
NZ_ = MH["NZ"]
crng = np.random.default_rng(31)
sims = np.empty((len(th), len(bz)))
for s_i, t in enumerate(th):
    eta = t[0] + np.exp(t[6 + NZ_]) * t[3 + NZ_:6 + NZ_][MH["cls"]] + MH["Z"] @ t[1:1 + NZ_]
    sl = np.where(MH["edge"], t[2 + NZ_], t[1 + NZ_])
    T_ = MH["mS"] + MH["sS"] * crng.standard_normal(len(bz))
    mu_ = np.clip(expit(eta + sl * T_), 1e-6, 1 - 1e-6)
    kap = np.exp(t[7 + NZ_])
    p_ = crng.beta(mu_ * kap, (1 - mu_) * kap)
    sims[s_i] = crng.binomial(MH["n"].astype(int), p_)
lo80, hi80 = np.percentile(sims, [10, 90], axis=0)
put("ppc_cov80", 100 * np.mean((MH["y"] >= lo80) & (MH["y"] <= hi80)), 0)
big = MH["n"] >= MIN_SNAPS
obs_sd = np.std(MH["y"][big] / MH["n"][big])
sim_sd = np.std(sims[:, big] / MH["n"][big], axis=1)
put("ppc_sd_p", np.mean(sim_sd >= obs_sd))

# prior-free alternative on the 100+ cohort: ridge, penalty chosen by inner leave-one-class-out, fully nested
_tick("ridge")
LAMS = [0.1, 0.3, 1, 3, 10, 30, 100]


def ridge_fit(tr, cols, lam):
    X = tr[cols].values
    mx, sx = X.mean(0), X.std(0)
    sx[sx == 0] = 1
    Xs = (X - mx) / sx
    y = tr._y.values
    b = np.linalg.solve(Xs.T @ Xs + lam * np.eye(len(cols)), Xs.T @ (y - y.mean()))
    return lambda te: y.mean() + ((te[cols].values - mx) / sx) @ b


def ridge_loco(df, cols, screen=False):
    """Leave one class out. screen=True: the drill feature is re-chosen by the 14-pair screen on the training classes
    and appended to cols (rows missing it are dropped in that fold)."""
    Y, P = {}, {}
    for c in classes:
        tr, te = df[df.draft_year != c], df[df.draft_year == c]
        cc_ = cols
        if screen:
            best_ = screen_pick(tr)[0]
            cc_ = cols + [best_]
            tr, te = tr.dropna(subset=[best_]), te.dropna(subset=[best_])
        tr, te = fill_times(tr, te)
        mu_, sd_ = tr.press_eb.mean(), tr.press_eb.std()
        tr["_y"], te["_y"] = (tr.press_eb - mu_) / sd_, (te.press_eb - mu_) / sd_
        sc_ = []
        for lam in LAMS:          # inner loop: leave one training class out
            e_ = 0.0
            for c2 in tr.draft_year.unique():
                f_ = ridge_fit(tr[tr.draft_year != c2], cc_, lam)
                e_ += ((tr[tr.draft_year == c2]._y - f_(tr[tr.draft_year == c2])) ** 2).sum()
            sc_.append(e_)
        f_ = ridge_fit(tr, cc_, LAMS[int(np.argmin(sc_))])
        Y[c], P[c] = pd.Series(te._y.values, index=te.index), pd.Series(f_(te), index=te.index)
    return Y, P


def ridge_gain(df, screen=True):
    """R² change on the same held-out rows: ridge baseline vs ridge baseline + drill feature."""
    df = df.reset_index(drop=True)
    _, Pa = ridge_loco(df, BASE_T, screen=screen) if screen else ridge_loco(df, BASE_T + ["hoop_z"])
    Yb, Pb = ridge_loco(df, BASE_T)
    for c in classes:
        Yb[c], Pb[c] = Yb[c].loc[Pa[c].index].values, Pb[c].loc[Pa[c].index].values
        Pa[c] = Pa[c].values
    yy = np.concatenate([Yb[c] for c in classes])
    out = {"pooled": r2(yy, np.concatenate([Pa[c] for c in classes])) - r2(yy, np.concatenate([Pb[c] for c in classes]))}
    for c in classes:
        out[int(c)] = r2(Yb[c], Pa[c]) - r2(Yb[c], Pb[c])
    return out


rd = cand[cand.hoop_z.notna()].copy()
put("ridge_n", len(rd))
grd = {c: d for c, d in rd.groupby("draft_year")}
for tg_, scr_ in [("ridge", True), ("ridgefix", False)]:
    rg = ridge_gain(rd, screen=scr_)
    put(f"{tg_}_dr2", rg["pooled"], 3)
    for c in classes:
        put(f"{tg_}_dr2_{c}", rg[int(c)], 3)

    def _rb(scr_=scr_):
        rng_, out_ = np.random.default_rng(13), []
        for _ in range(NLOCO):
            d_ = pd.concat([g_.iloc[rng_.integers(0, len(g_), len(g_))] for g_ in grd.values()])
            out_.append(ridge_gain(d_, screen=scr_)["pooled"])
        return out_
    rb_ = cached(f"ridge_{tg_}_{NLOCO}", _rb)
    put(f"{tg_}_dr2_lo", np.percentile(rb_, 2.5), 3); put(f"{tg_}_dr2_hi", np.percentile(rb_, 97.5), 3)
# least squares with Hoop G fixed in advance (no screen, no selection shrinkage): the optimistic bound
fx_ = []
for c in classes:
    tr, te = fill_times(rd[rd.draft_year != c], rd[rd.draft_year == c])
    fx_.append((te.press_eb.values, ols_pred(tr, te, BASE_T, "press_eb"), ols_pred(tr, te, BASE_T + ["hoop_z"], "press_eb")))
put("olsfix_dr2", r2(np.concatenate([f[0] for f in fx_]), np.concatenate([f[2] for f in fx_])) -
    r2(np.concatenate([f[0] for f in fx_]), np.concatenate([f[1] for f in fx_])), 3)
put("olsfix_pos", int(sum(r2(f[0], f[2]) > r2(f[0], f[1]) for f in fx_)))

# what a team already has: the stopwatch drills against pressure, same baseline (pick, weight, role, NGS score)
for col_, tg_ in [("three_cone_z", "cmp3"), ("ten_yd_split_z", "cmp10"), ("hoop_z", "cmph")]:
    r, lo, hi, n = partial_sp(co.assign(_x=-co[col_] if col_ != "hoop_z" else co[col_]), "_x", "press_eb", MAIN)
    put(f"{tg_}_r", r); put(f"{tg_}_lo", lo); put(f"{tg_}_hi", hi); put(f"{tg_}_n", n)
put("rho_hoop_wt", stats.spearmanr(dl.hoop_z, dl.combine_weight, nan_policy="omit")[0])
# the same comparison on one common set of rushers (all of: Hoop G, 10-yd split, 3-cone, NGS score)
cm = co.dropna(subset=["hoop_z", "ten_yd_split_z", "three_cone_z", "ngs_athleticism_score", "press_eb"]).copy()
cm["_p"], cm["_t10"], cm["_t3"] = -cm.log_pick, -cm.ten_yd_split_z, -cm.three_cone_z
WR_ = ["combine_weight_z", "edge_share"]
CMP = [("_p", "Earlier pick", WR_), ("ngs_athleticism_score", "NGS athleticism score", WR_),
       ("_t10", "10-yd split (official)", MAIN), ("_t3", "3-cone (official)", MAIN), ("hoop_z", "Hoop G (tracked)", MAIN)]
co_own = co.assign(_p=-co.log_pick, _t10=-co.ten_yd_split_z, _t3=-co.three_cone_z)
CMPR = []
for col_, lab_, ctl_ in CMP:
    r, lo, hi, n = partial_sp(cm, col_, "press_eb", ctl_)
    ro, loo_, hio_, no_ = partial_sp(co_own, col_, "press_eb", ctl_)
    CMPR.append(dict(col=col_, label=lab_, r=r, lo=lo, hi=hi, n=n, r_own=ro, n_own=no_, ctl=len(ctl_)))
put("cmn_n", len(cm))
for d_, tg_ in zip(CMPR, ["pick", "ngs", "t10", "t3", "hoop"]):
    put(f"cmn_{tg_}_r", d_["r"]); put(f"cmn_{tg_}_lo", d_["lo"]); put(f"cmn_{tg_}_hi", d_["hi"])
put("cmn_mde", mde(len(cm), len(MAIN)))
TABLES["common_n"] = CMPR
# what a confirmatory test would need: rushers for 80% power (one-sided 0.05) at the selection-adjusted effect
n80 = ((stats.norm.ppf(0.95) + stats.norm.ppf(0.8)) / np.arctanh(eb_hoop)) ** 2 + 3 + len(BASE_T)
put("n_power80", int(np.ceil(n80 / 10) * 10)); put("eb_hoop", eb_hoop)
put("classes_power80", int(np.ceil(n80 / (NFI[0]["n"] / 3))))
# headline under the run-to-run reliability (players with 2+ runs) instead of the split-half value
summ(laplace(make_model(bz, ZCT, rel=float(np.clip(rel_icc, 0.1, 0.95))), PRIOR_B / np.sqrt(np.clip(rel_icc, 0.1, 0.95) / REL),
             seed=9)[:, 0], "relicc_t")

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
ba["rank_in_class"] = ba.groupby("draft_year").shr.rank(ascending=False, method="first").astype(int)
ba = ba.sort_values(["draft_year", "rank_in_class"])
ba[["draft_year", "rank_in_class", "display_name", "role", "pick", "combine_weight", "hoop_k", "hoop_g", "hoop_z", "shr",
    "shr_lo", "shr_hi", "rush_snaps", "press_rate", "pr_lo", "pr_hi", "pb", "call"]].round(4).to_csv(
    OUT / "board_2023_2025.csv", index=False)
called = ba[ba.call != "neutral"].copy()
for rl, tg in [("EDGE", "e"), ("interior", "i")]:
    for cl in ["plus", "minus"]:
        put(f"calls_{tg}_{cl}", int(((called.role == rl) & (called.call == cl)).sum()))
# forecast vs actual for the plus / minus calls with 100+ rushes (calls use the drill only)
ev_ = called[called.call.isin(["plus", "minus"]) & (called.rush_snaps >= MIN_SNAPS)]
for cl in ["plus", "minus"]:
    e_ = ev_[ev_.call == cl]
    put(f"fa_{cl}_n", len(e_)); put(f"fa_{cl}_above", int((e_.press_rate > e_.pb).sum()))
    put(f"fa_{cl}_gap", 100 * (e_.press_rate - e_.pb).mean(), 1)

ROLE_TXT = {"EDGE": "EDGE", "interior": "Int"}
_nc = ba[(ba.call == "neutral") & ba.pb.notna()]
put("base_above_pct", 100 * (_nc.pr_lo > _nc.pb).mean(), 0); put("base_n", len(_nc))
_ni = _nc[_nc.role == "interior"]
put("base_int_below_pct", 100 * (_ni.pr_hi < _ni.pb).mean(), 0); put("base_int_n", len(_ni))


def ph(key, v, nd=2):
    """put() a number and return its template placeholder (tables are rendered through the same key check)."""
    put(key, v, nd)
    return "{{" + key + "}}"


def pk_txt(p):
    return "UDFA" if p == UDFA_PICK else str(int(p))


def rule_txt(r_):
    if r_.call == "plus":
        return "break ties his way"
    if r_.role == "EDGE":
        return "check film for a power rush"
    return "no call: weight already priced"


def result_txt(r_):
    if r_.pr_lo > r_.pb:
        return "**clearly above**"
    if r_.pr_hi < r_.pb:
        return "**clearly below**"
    return "in noise"


def tie_range(r_):
    """Pick window in which the tie-break applies: same role, picks within 30% of his own (day 3: rounds 6-7, UDFA)."""
    who = "EDGEs" if r_.role == "EDGE" else "interior DL"
    if r_.pick >= 200:
        return f"day-3 / UDFA {who}"
    return f"{who} picked {int(round(r_.pick / 1.3))}–{int(round(r_.pick * 1.3))}"


bd = ba[ba.draft_year == 2025].sort_values("shr", ascending=False).reset_index(drop=True)
bd["rank"] = np.arange(1, len(bd) + 1)
put("board_n", len(bd))
t25 = bd[(bd.call != "neutral") & bd.pb.notna()]
out = ["| # of " + str(len(bd)) + " | Player (pick, role) | Hoop G, sd [80%] | Draft-day rule | Rookie pressure rate vs "
       "forecast (rushes) | Result |", "|" + "---|" * 6]
for i, r_ in enumerate(t25.itertuples()):
    k_ = f"tb_b25_{i}"
    hg = f"{ph(k_ + '_s', r_.shr)} [{ph(k_ + '_l', r_.shr_lo)}, {ph(k_ + '_h', r_.shr_hi)}]"
    prs = f"{ph(k_ + '_pr', 100 * r_.press_rate, 1)}% vs {ph(k_ + '_pb', 100 * r_.pb, 1)}% ({int(r_.rush_snaps)})"
    win = f" vs {tie_range(r_)}" if r_.call == "plus" else ""
    out.append(f"| {int(r_.rank)} | {r_.display_name} ({pk_txt(r_.pick)}, {ROLE_TXT[r_.role]}) | {hg} | "
               f"{rule_txt(r_)}{win} | {prs} | {result_txt(r_)} |")
MD = {"board25": "\n".join(out)}

# ---- head-to-head EDGE pairs: same class, picks within 30%, Hoop G gap >= 0.4 sd, both 100+ rushes
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

# ---- draft-room example: the closest-pick same-class EDGE pair with a 0.4+ sd Hoop G gap
h_, l_, _ = sorted(pair_list(ep, shr0), key=lambda z: z[2])[0]
H, Lw = ep.loc[h_], ep.loc[l_]
dP = float(NUM["bayes_all"]) * (H.shr - Lw.shr) / REL        # shr = lam * z, so (shr / REL) is in measured-run sd
p_ref = max(H.pick, Lw.pick)
put("ex_dpress", dP, 1)
put("ex_pick_from", int(p_ref)); put("ex_pick_to", int(round(p_ref * np.exp(-(dP / 4.5) / B_LP))))
put("ex_gap", H.shr - Lw.shr)
put("ex_pr_h", 100 * H.press_rate, 1); put("ex_pb_h", 100 * H.pb, 1)
put("ex_pr_l", 100 * Lw.press_rate, 1); put("ex_pb_l", 100 * Lw.pb, 1)
MD["ex_names"] = (f"{H.display_name} (pick {pk_txt(H.pick)}) and {Lw.display_name} (pick {pk_txt(Lw.pick)}) "
                  f"in {int(H.draft_year)}")
MD["ex_h"], MD["ex_l"] = H.display_name.split(" ")[-1], Lw.display_name.split(" ")[-1]
MD["ex_right"] = "right" if H.res > Lw.res else "wrong"

# clear calls: the 80% interval of the actual rate excludes the forecast
cl80 = ba[(ba.call != "neutral") & ba.pb.notna()]
put("clear_above", int((cl80.pr_lo > cl80.pb).sum())); put("clear_below", int((cl80.pr_hi < cl80.pb).sum()))
put("clear_n", len(cl80))
_sgn = np.sign(cl80.shr)
_up, _dn = cl80.pr_lo > cl80.pb, cl80.pr_hi < cl80.pb
put("clear_hit", int((((_sgn > 0) & _up) | ((_sgn < 0) & _dn)).sum()))
put("clear_miss", int((((_sgn > 0) & _dn) | ((_sgn < 0) & _up)).sum()))
p24 = cl80[(cl80.call == "plus") & (cl80.draft_year <= 2024)]
grp_ = {"above": p24[p24.pr_lo > p24.pb], "noise": p24[(p24.pr_lo <= p24.pb) & (p24.pr_hi >= p24.pb)],
        "below": p24[p24.pr_hi < p24.pb]}


def names_(d):
    n_ = [r_.display_name for r_ in d.sort_values("pick").itertuples()]
    return "none" if not n_ else (n_[0] if len(n_) == 1 else ", ".join(n_[:-1]) + " and " + n_[-1])


MD["calls_above"], MD["calls_noise"], MD["calls_below"] = (names_(grp_[k]) for k in ["above", "noise", "below"])
put("calls24_n", len(p24))
MD_TXT = {}

# ============================================================================= FIGURES
_tick("figures")


def name_short(n):
    return n.split(" ")[0][0] + ". " + " ".join(n.split(" ")[1:])


# ---- Fig 1 (hero): four named rushers. Hoop path and its sideways-g trace next to his hardest-turning sampled NFL rush
P4 = ["Chop Robinson", "Donovan Ezeiruaku", "Tuli Tuipulotu", "Jordan Phillips"]
cmap = plt.get_cmap("cividis"); norm = plt.Normalize(0, 0.9)
GREY_T = "#C8C8C8"
FS = 13
F1 = {}
for nm in P4:
    r_ = ba[ba.display_name == nm].iloc[0]
    tr = traces[r_.hoop_eid]
    xs_, ys_ = tr["_x"] - (tr["_x"].max() + tr["_x"].min()) / 2, tr["_y"] - (tr["_y"].max() + tr["_y"].min()) / 2
    if np.ptp(ys_) < np.ptp(xs_):
        xs_, ys_ = ys_, xs_
    roll_h = pd.Series(tr["_alat"]).rolling(5, center=True).mean().to_numpy()
    q_ = arcp[(arcp.nfl_id == r_.nfl_id) & (arcp.kind == "random")]
    kk = q_.loc[q_.arc_g.idxmax()]
    F1[nm] = dict(r=r_, hx=xs_, hy=ys_, hal=tr["_alat"], hk=tr["_k"], hpk=np.nanmax(roll_h), q=q_,
                  sel=arc_tr[(kk.game_id, kk.play_id, kk.nfl_id)], sel_g=kk.arc_g, sel_press=kk.press, sel_wt=kk.wtype)
HSPAN = max(max(np.ptp(v["hx"]), np.ptp(v["hy"])) for v in F1.values()) + 1.5
allx = np.concatenate([arc_tr[(k.game_id, k.play_id, k.nfl_id)][0] for v in F1.values() for k in v["q"].itertuples()])
ally = np.concatenate([arc_tr[(k.game_id, k.play_id, k.nfl_id)][1] for v in F1.values() for k in v["q"].itertuples()])
NX = (np.floor(allx.min()) - 0.5, np.ceil(allx.max()) + 0.5)
NY = (-1.0, np.ceil(ally.max()) + 0.8)
put("f1_cohort_arc_med", arcp[arcp.kind == "random"].arc_g.median())
FH1 = 15.0
YS = 12.6 / FH1
YF = lambda f: 1 - (1 - f) * YS
fig = plt.figure(figsize=(FIGW, FH1))
fig.text(0.012, YF(0.992), "What Hoop G sees: four rushers in the hoop and on Sundays", fontsize=18, weight="bold", va="top")
fig.text(0.012, YF(0.962), "Columns 1–2: one Combine hoop run, coloured by sideways g; the ring and shaded band mark the half-second "
         "that sets Hoop G.\nColumns 3–4: his hardest-turning of 20 random NFL rushes (others grey), snap ▲ to "
         "throw or sack; outside is to the right.", fontsize=12.5, va="top", color="#333333")
ROWH, TOP, AH = 0.212 * YS, YF(0.895), 0.15 * YS
AW = AH * FH1 / FIGW
TW = 0.235
X0 = [0.03, 0.03 + AW + 0.07, 0.03 + AW + 0.07 + TW + 0.025]
NW = 0.215
X0.append(X0[2] + NW + 0.035)
_rend = fig.canvas.get_renderer()
for idx, nm in enumerate(P4):
    v = F1[nm]; r_ = v["r"]
    yt = TOP - idx * ROWH
    ybot = yt - 0.035 * YS - AH
    t_nm = fig.text(0.012, yt, nm, fontsize=15, weight="bold", va="top")
    x_info = 0.012 + t_nm.get_window_extent(_rend).width / fig.bbox.width + 0.012
    fig.text(x_info, yt - 0.002,
             f"{'EDGE' if r_.role == 'EDGE' else 'interior'}, {r_.draft_year} pick {pk_txt(r_.pick)} · raw Hoop G "
             f"{'+' if r_.hoop_z > 0 else '−'}{abs(r_.hoop_z):.1f} sd · NFL pressure rate {100 * r_.press_rate:.1f}% "
             f"({int(r_.rush_snaps)} rushes)", fontsize=13, va="top", color="#333333")
    # 1: hoop path
    ax = fig.add_axes([X0[0], ybot, AW, AH])
    pts = np.c_[v["hx"], v["hy"]].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=norm, linewidth=3.2)
    lc.set_array(v["hal"][:-1]); ax.add_collection(lc)
    ax.scatter([v["hx"][v["hk"]]], [v["hy"][v["hk"]]], s=150, facecolor="none", edgecolor="k", lw=2.2, zorder=3)
    ax.set_xlim(-HSPAN / 2, HSPAN / 2); ax.set_ylim(-HSPAN / 2, HSPAN / 2); ax.set_aspect("equal")
    ax.plot([-HSPAN / 2 + 0.5, -HSPAN / 2 + 2.5], [HSPAN / 2 - 1.3] * 2, color="k", lw=2.5)
    ax.text(-HSPAN / 2 + 1.5, HSPAN / 2 - 1.0, "2 yd", ha="center", va="bottom", fontsize=12)
    # 2: hoop sideways g over time
    tx = fig.add_axes([X0[1], ybot, TW, AH])
    t_ = np.arange(len(v["hal"])) * DT
    tx.axvspan((v["hk"] - 2) * DT, (v["hk"] + 2) * DT, color="#BBBBBB", alpha=0.6, lw=0)
    tx.plot(t_, np.clip(v["hal"], 0, 1.2), color=OI["blue"], lw=2)
    tx.axhline(v["hpk"], color="k", lw=1, ls=":")
    tx.text(0.98, 0.95, f"Hoop G {v['hpk']:.2f} g", transform=tx.transAxes, ha="right", va="top", fontsize=13, weight="bold")
    tx.set_ylim(0, 1.2); tx.set_xlim(0, 9); tx.set_yticks([0, 0.5, 1.0]); tx.tick_params(labelsize=12)
    tx.set_ylabel("sideways g", fontsize=12)
    if idx == 3:
        tx.set_xlabel("seconds into the hoop run", fontsize=12.5)
    # 3: one NFL rush (hardest sustained turn), others grey
    gx = fig.add_axes([X0[2], ybot, NW, AH])
    for k in v["q"].itertuples():
        la, fw, *_ = arc_tr[(k.game_id, k.play_id, k.nfl_id)]
        gx.plot(la, fw, color=GREY_T, lw=1.1, zorder=1)
    la, fw, al, kmx, wty = v["sel"]
    pts = np.c_[la, fw].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=norm, linewidth=3.4, zorder=3)
    lc.set_array(al[:-1]); gx.add_collection(lc)
    kc = int(np.clip(kmx, 0, len(la) - 1))
    gx.scatter([la[kc]], [fw[kc]], s=150, facecolor="none", edgecolor="k", lw=2.2, zorder=4)
    gx.scatter([0], [0], marker="^", color="k", s=60, zorder=5)
    gx.set_xlim(*NX); gx.set_ylim(*NY); gx.set_aspect("equal")
    gx.plot([NX[1] - 2.6, NX[1] - 0.6], [NY[0] + 0.6] * 2, color="k", lw=2.5)
    gx.text(NX[1] - 1.6, NY[0] + 0.9, "2 yd", ha="center", va="bottom", fontsize=12)
    # 4: that rush's sideways g over time
    nx_ = fig.add_axes([X0[3], ybot, 1 - X0[3] - 0.012, AH])
    tn = np.arange(len(al)) * DT
    nx_.axvspan((kc - 2) * DT, (kc + 2) * DT, color="#BBBBBB", alpha=0.6, lw=0)
    nx_.plot(tn, np.clip(al, 0, 1.2), color=OI["blue"], lw=2)
    nx_.axhline(v["sel_g"], color="k", lw=1, ls=":")
    nx_.axhline(v["hpk"], color=OI["purple"], lw=2.2, ls="--")
    nx_.text(0.02, v["hpk"] + 0.04, f"his Hoop G {v['hpk']:.2f}", transform=nx_.get_yaxis_transform(), fontsize=12,
             color=OI["purple"], weight="bold", va="bottom")
    nx_.text(0.98, 0.95, f"NFL {v['sel_g']:.2f} g" + (" · pressure" if v["sel_press"] else ""),
             transform=nx_.transAxes, ha="right", va="top", fontsize=13, weight="bold")
    nx_.set_ylim(0, 1.2); nx_.set_xlim(0, ARC_CAP * DT); nx_.set_yticks([0, 0.5, 1.0]); nx_.tick_params(labelsize=12)
    if idx == 3:
        nx_.set_xlabel("seconds after the snap", fontsize=12.5)
    for a_ in (ax, gx):
        a_.set_xticks([]); a_.set_yticks([])
        for sp_ in a_.spines.values():
            sp_.set_visible(True); sp_.set_color("#BBBBBB")
    key = nm.split(" ")[-1].lower()
    put(f"f1_{key}_hoop", v["hpk"]); put(f"f1_{key}_nfl", v["sel_g"])
    q2 = v["q"][v["q"].wtype != "unclassified"].dropna(subset=["l3"])
    if len(q2) >= 5:
        put(f"f1_{key}_out", 100 * (q2.wtype == "outside").mean(), 0)
cax = fig.add_axes([0.03, YF(0.028), 0.2, 0.011 * YS])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[0, 0.3, 0.6, 0.9])
cb.ax.tick_params(labelsize=12); cax.set_title("sideways g (path colour)", fontsize=12.5, loc="left", pad=4)
# overlay: hoop peak vs NFL peaks for all four on one g axis
ov = fig.add_axes([0.36, 0.04, 0.6, 0.12])
for i_, nm in enumerate(P4[::-1]):
    v = F1[nm]
    med_ = v["q"].arc_g.median()
    ov.plot([med_, v["hpk"]], [i_, i_], color="#BBBBBB", lw=3, zorder=1)
    ov.scatter([v["hpk"]], [i_], s=130, color=OI["purple"], zorder=3)
    ov.scatter([v["sel_g"]], [i_], s=110, facecolor="white", edgecolor="k", lw=2, zorder=3)
    ov.scatter([med_], [i_], s=70, color="k", zorder=3)
    F1[nm]["med"] = med_
ov.axvline(float(arcp[arcp.kind == "random"].arc_g.median()), color="#555555", lw=1, ls=":")
ov.set_yticks(range(4)); ov.set_yticklabels([n_.split(" ")[-1] for n_ in P4[::-1]], fontsize=13)
ov.set_xlim(0.1, 0.8); ov.set_ylim(-0.6, 3.6); ov.tick_params(axis="x", labelsize=12)
ov.set_xlabel("peak half-second sideways g", fontsize=12.5)
ov.set_title("All four on one scale: no rusher turns as hard on Sundays as in the hoop", fontsize=13.5, loc="left",
             weight="bold")
ov.legend(handles=[Line2D([], [], marker="o", color=OI["purple"], lw=0, markersize=11, label="Hoop G"),
                   Line2D([], [], marker="o", color="k", markerfacecolor="white", lw=0, markersize=10, mew=2,
                          label="hardest NFL rush"),
                   Line2D([], [], marker="o", color="k", lw=0, markersize=7, label="typical NFL rush (median)"),
                   Line2D([], [], color="#555555", ls=":", label="all-DL typical rush")],
          loc="upper left", bbox_to_anchor=(-0.6, 0.8), fontsize=12, frameon=False)
for nm in P4:
    key = nm.split(" ")[-1].lower(); r_ = F1[nm]["r"]
    put(f"f1_{key}_med", F1[nm]["med"]); put(f"f1_{key}_pr", 100 * r_.press_rate, 1); put(f"f1_{key}_pb", 100 * r_.pb, 1)
save(fig, "fig1_four_rushers.png")
put("f1_nfl_max", max(v["sel_g"] for v in F1.values()))
put("f1_edge_min", min(v["hpk"] for v in F1.values() if v["r"].role == "EDGE"))
put("f1_edge_max", max(v["hpk"] for v in F1.values() if v["r"].role == "EDGE"))
put("f1_hoop_min", min(v["hpk"] for v in F1.values()))

# ---- Fig 2: one outcome (pressure), five measures, one common set of rushers
fig = plt.figure(figsize=(FIGW, 6.2))
ax = fig.add_axes([0.3, 0.16, 0.66, 0.6])
yy = np.arange(len(CMPR))[::-1]
for y_, d_ in zip(yy, CMPR):
    col_ = OI["blue"] if d_["col"] == "hoop_z" else OI["black"]
    ax.plot([d_["lo"], d_["hi"]], [y_, y_], color=col_, lw=3, solid_capstyle="butt")
    ax.scatter([d_["r"]], [y_], color=col_, s=120, zorder=3)
    ax.scatter([d_["r_own"]], [y_ - 0.3], marker="o", facecolor="white", edgecolor=col_, s=60, lw=1.6, zorder=3)
    ax.text(0.99, y_, f"{d_['r']:+.2f}".replace("-", "−"), transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=14, weight="bold", color=col_)
    ax.text(0.99, y_ - 0.3, f"own n={d_['n_own']}: {d_['r_own']:+.2f}".replace("-", "−"), transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=11.5, color="#444444")
ax.axvline(0, color="k", lw=0.9)
_m = float(NUM["cmn_mde"])
ax.axvline(_m, color="#555555", lw=1.4, ls="--")
ax.text(_m + 0.01, yy.max() + 0.55, f"80% power at n={len(cm)}", fontsize=12, color="#444444", va="bottom")
ax.set_yticks(yy)
ax.set_yticklabels([f"{d_['label']}\n" + ("after weight, role" if d_["ctl"] == 2 else "after pick, weight, role, NGS")
                    for d_ in CMPR], fontsize=13)
ax.set_xlim(-0.75, 1.15); ax.set_ylim(-0.7, len(CMPR) - 0.1)
ax.set_xticks([-0.5, 0, 0.5]); ax.tick_params(axis="x", labelsize=12)
ax.set_xlabel("adjusted rank correlation with NFL pressure rate (better number → positive)", fontsize=13)
_ncross = sum(d_["lo"] < 0 < d_["hi"] for d_ in CMPR)
fig.suptitle(f"Pressure vs five draft-day measures on the same {len(cm)} rushers: {_ncross} of 5 intervals include zero",
             fontsize=15, weight="bold", x=0.012, ha="left", y=0.975)
fig.text(0.012, 0.9, f"DL with 100+ NFL rushes who have Hoop G, an official 10-yd split, 3-cone and NGS score (n={len(cm)}). "
         "Filled dot and bar: adjusted\ncorrelation and 95% bootstrap interval on these rushers; open dot: same measure on "
         "every rusher who has it.", fontsize=12.5, color="#333333", va="top")
save(fig, "fig2_signal_map.png")

# ---- Fig 3: planning number and, per held-out class, the class's own estimate next to the held-out accuracy gain
rows3 = [(f"All {len(bz)} · planning number", PRIOR_DRAWS["skeptical"], OI["blue"], None)]
for c in classes:
    rows3.append((f"{c} class alone (n={int((bz.draft_year == c).sum())})", SIGN_DRAWS[int(c)][1], OI["black"], int(c)))
fig = plt.figure(figsize=(FIGW, 6.5))
axl = fig.add_axes([0.24, 0.12, 0.33, 0.56]); axr = fig.add_axes([0.62, 0.12, 0.24, 0.56])
yy = np.array([3.4, 2, 1, 0], float)
for y_, (lab, v, col, cl) in zip(yy, rows3):
    a90, a80 = np.percentile(v, [5, 95]), np.percentile(v, [10, 90])
    axl.plot(a90, [y_, y_], color=col, lw=2.2, solid_capstyle="butt")
    axl.plot(a80, [y_, y_], color=col, lw=9, alpha=0.4, solid_capstyle="butt")
    axl.scatter([np.median(v)], [y_], color=col, s=80, zorder=3, edgecolor="white")
    if cl is None:
        g_, glo_, ghi_ = WN["pooled"], np.percentile(WB["pooled"], 5), np.percentile(WB["pooled"], 95)
        fe_ = f"Hoop G\nin {NUM['nlpd_hoop_folds']} of 3"
        lo_a, hi_a = np.percentile(NULL[:, 1], [5, 95])
        lo_h, hi_h = np.percentile(NULL[NULL[:, 2] >= 1, 1], [5, 95])
        axr.fill_between([lo_a, hi_a], y_ + 0.05, y_ + 0.4, color="#D0D0D0", lw=0, zorder=0)
        axr.fill_between([lo_h, hi_h], y_ - 0.4, y_ - 0.05, color="#A8A8A8", lw=0, zorder=0)
    else:
        g_ = WN[cl]; glo_, ghi_ = np.percentile(WB[cl], 5), np.percentile(WB[cl], 95)
        fe_ = NSEL[cl]["feat"].replace("hoop_z", "Hoop G").replace("hoop_speed_z", "hoop speed")
    axr.plot([glo_, ghi_], [y_, y_], color=col, lw=2.6)
    axr.scatter([g_], [y_], color=col, s=80, zorder=3, edgecolor="white")
    axr.text(1.02, y_, fe_, transform=axr.get_yaxis_transform(), fontsize=12, va="center", color="#333333")
for a_ in (axl, axr):
    a_.axvline(0, color="k", lw=0.9); a_.set_ylim(-0.6, 4.1); a_.tick_params(labelsize=12)
axl.set_yticks(yy); axl.set_yticklabels([r[0] for r in rows3], fontsize=13)
axr.set_yticks(yy); axr.set_yticklabels([])
axl.set_xlabel("extra pressures per 450-rush season, per sd", fontsize=12.5)
axr.set_xlabel("held-out R² change, rush-weighted", fontsize=12.5)
axl.set_title("Effect of 1 sd of Hoop G", fontsize=13.5, weight="bold", loc="left")
axr.set_title("Scored on the class left out", fontsize=13.5, weight="bold", loc="left")
axr.text(1.02, 3.95, "drill the\nfold chose", transform=axr.get_yaxis_transform(), fontsize=11.5, va="bottom",
         weight="bold")
fig.suptitle(f"Scored on unseen classes, Hoop G helps in {sum(WN[int(c)] > 0 for c in classes)} of 3 and beats "
             f"shuffled labels (p = {NUM['perm_p_r2']})", fontsize=14.5, weight="bold", x=0.012, ha="left", y=0.975)
fig.text(0.012, 0.9, "Left: dot = median, thick = 80%, thin = 90% interval. Right: baseline plus the drill the screen "
         "chose on the two\ntraining classes, minus the baseline alone (90% interval). Grey: 5th–95th percentile of "
         f"the same pipeline after\n{NPERM} within-class shuffles of Hoop G; light = all shuffles, dark = shuffles in "
         f"which the shuffled Hoop G won the screen (p = {NUM['perm_p_r2_h']}).", fontsize=12, color="#333333", va="top")
save(fig, "fig3_headline.png")

# ---- Fig 4: every 2023-25 call against its out-of-class no-tracking forecast, plus and minus as two blocks
cv = called[called.pb.notna()].copy()
cv["res"] = cv.press_rate - cv.pb
cv["grp"] = np.where(cv.call == "plus", "plus", "minus")
_mrole = "all interior" if (cv[cv.grp == "minus"].role == "interior").all() else "mixed roles"
GRPS = [("plus", "Plus calls (break ties his way)"), ("minus", f"Minus calls ({_mrole}: no call, weight already priced)")]
NOISE = "#666666"
ylist, rowsd, heads = [], [], []
y = 0.0
for gk, gt in GRPS:
    d_ = cv[cv.grp == gk].sort_values("res", ascending=False)
    heads.append((y, gt, d_)); y += 1.0
    for r_ in d_.itertuples():
        ylist.append(y); rowsd.append(r_); y += 1
    y += 0.4
NR = y
HH = 0.62 * NR + 2.8
fig = plt.figure(figsize=(FIGW, HH))
ax = fig.add_axes([0.47, 1.1 / HH, 0.36, 1 - 2.95 / HH])
ax.spines["left"].set_visible(False)
XMAX = float(np.ceil(100 * cv.pr_hi.max() + 1))
for yv, r_ in zip(ylist, rowsd):
    clear_up, clear_dn = r_.pr_lo > r_.pb, r_.pr_hi < r_.pb
    c_ = OI["blue"] if clear_up else (OI["red"] if clear_dn else NOISE)
    ax.plot([100 * r_.pr_lo, 100 * r_.pr_hi], [yv, yv], color="#D9D9D9", lw=10, solid_capstyle="round", zorder=1)
    ax.annotate("", xy=(100 * r_.press_rate, yv), xytext=(100 * r_.pb, yv),
                arrowprops=dict(arrowstyle="-|>", color=c_, lw=2.6, mutation_scale=18), zorder=3)
    ax.scatter([100 * r_.pb], [yv], marker="|", s=260, color="k", lw=2.6, zorder=4)
    lab_ = "clearly above" if clear_up else ("clearly below" if clear_dn else "in noise")
    ax.text(1.02, yv, lab_, transform=ax.get_yaxis_transform(), fontsize=15, va="center", color=c_,
            weight="bold" if (clear_up or clear_dn) else "normal")
    # sparkline: his hoop run's sideways g on one shared 0-1.2 g scale, Hoop G value printed inside
    tr = traces[r_.hoop_eid]
    sp = ax.inset_axes([-0.36, yv - 0.44, 0.3, 0.88], transform=ax.get_yaxis_transform())
    k_ = tr["_k"]
    sp.axvspan(k_ - 2, k_ + 2, color="#BBBBBB", alpha=0.7, lw=0)
    sp.plot(np.clip(tr["_alat"], 0, 1.2), color=OI["blue"] if r_.call == "plus" else "#333333", lw=1.6)
    sp.set_ylim(0, 1.2); sp.set_xlim(0, 90); sp.set_xticks([]); sp.set_yticks([])
    for sp_ in sp.spines.values():
        sp_.set_visible(True); sp_.set_color("#CCCCCC")
    sp.text(0.03, 0.95, f"{r_.hoop_g:.2f} g", transform=sp.transAxes, fontsize=14, va="top", ha="left", weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
ax.set_yticks(ylist)
ax.set_yticklabels([f"{r_.display_name} '{r_.draft_year % 100:02d}" for r_ in rowsd], fontsize=15)
ax.tick_params(axis="y", pad=0.38 * 0.36 * FIGW * 72 + 6, length=0)
for y0, gt, d_ in heads:
    nu, ndn = int((d_.pr_lo > d_.pb).sum()), int((d_.pr_hi < d_.pb).sum())
    ax.text(-0.98, y0 + 0.1, f"{gt}: {nu} above, {ndn} below, {len(d_) - nu - ndn} in noise",
            transform=ax.get_yaxis_transform(), fontsize=15, weight="bold", va="center")
ax.set_ylim(NR - 0.3, -0.6); ax.grid(axis="x", color="#EEEEEE", lw=0.6)
ax.set_xlim(0, XMAX); ax.set_xlabel("NFL pressure rate, %", fontsize=15)
ax.tick_params(axis="x", labelsize=14)
hand = [Line2D([], [], marker="|", color="k", lw=0, markersize=16, markeredgewidth=2.4, label="no-tracking forecast"),
        Line2D([], [], color=NOISE, lw=2.4, marker=">", label="actual rate"),
        Line2D([], [], color="#D9D9D9", lw=9, label="80% interval of actual")]
fig.legend(handles=hand, frameon=False, fontsize=14, loc="lower center", bbox_to_anchor=(0.62, 0.0), ncol=3)
_pl = cv[cv.grp == "plus"]
_pu, _pd = int((_pl.pr_lo > _pl.pb).sum()), int((_pl.pr_hi < _pl.pb).sum())
fig.suptitle(f"{_pu} of {len(_pl)} plus calls beat their no-tracking forecast clearly; "
             f"{'none' if _pd == 0 else _pd} fell clearly below", fontsize=17, weight="bold", x=0.012, ha="left",
             y=1 - 0.12 / HH)
fig.text(0.012, 1 - 0.62 / HH, "Sparkline: his hoop run, sideways g on one 0–1.2 g scale, grey = the half-second that "
         "sets Hoop G.\n'Clearly' = 80% interval of his actual rate excludes the out-of-class forecast. 2023–24 helped "
         f"choose the hoop.\nFor scale: {NUM['base_above_pct']}% of rushers without a call landed clearly above, and "
         f"{NUM['base_int_below_pct']}% of interior DL without a call clearly below.", fontsize=13.5, color="#333333", va="top")
save(fig, "fig4_calls.png")
put("plus_n", len(_pl)); put("plus_above", _pu); put("plus_below", _pd)
pm_ = cv[cv.grp == "minus"]
put("imin_n", len(pm_)); put("imin_below", int((pm_.pr_hi < pm_.pb).sum()))

# ============================================================================= write
_tick("write")
TABLES["md"] = MD
for k_, v_ in MD_TXT.items():
    MD[k_] = v_
used = render(OUT, NUM, MD)
keep = {k_: NUM[k_] for k_ in sorted(used)}     # exactly the numbers the text uses; claims.json maps each to its sentence
json.dump(keep, open(OUT / "numbers.json", "w"), indent=1, sort_keys=True)
json.dump(TABLES, open(OUT / "tables.json", "w"), indent=1, default=str)
if os.environ.get("BDB_ALLNUM"):
    json.dump(NUM, open(os.environ["BDB_ALLNUM"], "w"), indent=1, sort_keys=True)
_tick("done")

