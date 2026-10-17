"""Hoop G (Run-the-Hoop bend) vs NFL pass-rush pressure for DL/EDGE: one metric, tested honestly.

Run from the repo root:
    uv run python writeup/candidates/round_08/revised/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (only the values quoted in writeup.md), tables.json,
board_2023_2025.csv and figures/*.png next to this file, then renders writeup.md from writeup_template.md.
Frame-level feature extraction is cached in data/cache/ (gitignored; delete it to recompute from the frames).
Runtime: about 1 minute cached, about 6 minutes cold, one core.

Design (round 8 revised, from round 8 cand 1). Revised additions: complete-case 3-cone / no-3-cone held-out comparison, frozen-rule calibration with matched base rate and permutation test, matched-speed within-player mixed model, redrawn figures. New in round 8: role-specific draft-room calls (a low interior
Hoop G is a film flag, not a call) written to the board; Hoop G vs weight within role; the turning circle implied by a
typical Hoop G; within-player check against in-game sideways g with a player bootstrap; a max-statistic shuffle over the
14 screened drill-outcome pairs (multiplicity); class-by-class calibration of the calls with exact binomial intervals and
raw pressure rates by role; a 2025 decision card; a new 4x3 hero figure, a merged validation figure and a calls figure
with hollow 2025 markers and base-rate bands. Earlier design notes:
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
CACHE = Path(os.environ.get("BDB_CACHE", ROOT / "data" / "cache" / "r8c1"))
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

# ---- round 8: draft-room call per role. A low interior Hoop G is not a call (it mostly measures size): film flag only
_tick("round 8 additions")
ba["draft_room_call"] = np.select(
    [ba.call == "plus", (ba.call == "minus") & (ba.role == "EDGE"), (ba.call == "minus") & (ba.role == "interior")],
    ["plus", "minus", "none (low interior: size; film flag only)"], "none")
ba[["draft_year", "rank_in_class", "display_name", "role", "pick", "combine_weight", "three_cone", "hoop_k", "hoop_g",
    "hoop_z", "shr", "shr_lo", "shr_hi", "rush_snaps", "press_rate", "pr_lo", "pr_hi", "pb", "call",
    "draft_room_call"]].round(4).to_csv(OUT / "board_2023_2025.csv", index=False)
put("plus_all_n", int((ba.call == "plus").sum()))
put("plus_scored_n", int(((ba.call == "plus") & ba.pb.notna()).sum()))
put("low_int_n", int((ba.draft_room_call.str.startswith("none (low")).sum()))

# Hoop G vs body weight within each role
for rl, tg in [("interior", "int"), ("EDGE", "edge")]:
    d_ = ba[ba.role == rl][["hoop_z", "combine_weight"]].dropna()
    put(f"rho_wt_{tg}", stats.spearmanr(d_.hoop_z, d_.combine_weight)[0]); put(f"rho_wt_{tg}_n", len(d_))

# what 0.62 g means on the field: the turning circle at that sideways g and the speed he carries through it
_rad = (hoop_att.speed_at_bend ** 2 / (hoop_att.bend_g * YD_PER_G)).replace([np.inf, -np.inf], np.nan)
put("hoop_v_peak", hoop_att.speed_at_bend.median(), 1)
put("hoop_v_mph", hoop_att.speed_at_bend.median() * 0.9144 * 2.236936, 0)
put("hoop_radius", _rad.median(), 1)
put("nfl_arc_med", arcp[arcp.kind == "random"].arc_g.median())

# within-player check: does Hoop G show up in his own NFL rush paths? (mean peak sideways g over his random rushes,
# after weight and role; bootstrap over players)
put("pr_hoop_arc_lo", AFTER[("hoop_z", "arc_random")][1]); put("pr_hoop_arc_hi", AFTER[("hoop_z", "arc_random")][2])

# multiplicity: how often would the best of the 14 screened drill-outcome pairs look as strong as Hoop G -> pressure
# if drills and NFL outcomes were unrelated? Shuffle the NFL outcomes jointly across players within class.
NMULT = 2000
_obs = abs(obs_p["hoop_z"])
put("mult_obs", obs_p["hoop_z"])
_mr = np.random.default_rng(29)
_both = co.quick.notna()
_grp = [g_.index.values for _, g_ in co[_both].groupby("draft_year")]
_mx = []
for _ in range(NMULT):
    yp, yg = co.press_eb.copy(), co.quick.copy()
    for ix in _grp:
        pi_ = _mr.permutation(ix)
        yp.loc[ix], yg.loc[ix] = co.press_eb.loc[pi_].values, co.quick.loc[pi_].values
    a1, a2 = screen_stats(S_p, yp), screen_stats(S_g, yg)
    _mx.append(max(max(abs(v) for v in a1.values()), max(abs(v) for v in a2.values())))
_mx = np.array(_mx)
put("mult_p", (1 + (_mx >= _obs).sum()) / (1 + NMULT)); put("mult_med", np.median(_mx))
put("mult_nperm", NMULT)


# ---- calibration, class by class: each class is forecast by a no-tracking model fit on the other two; calls use the
#      drill alone. Exact (Clopper-Pearson) 90% intervals on the share of rushers who clearly beat their forecast.
def cp(k, n, a=0.10):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


sc = ba[ba.pb.notna()].copy()
sc["up"], sc["dn"] = sc.pr_lo > sc.pb, sc.pr_hi < sc.pb
sc["grp"] = np.select([sc.call == "plus", (sc.call == "minus") & (sc.role == "interior"), sc.call == "minus"],
                      ["plus", "lowint", "minusedge"], "none")
put("cal_n", len(sc))
CAL = [("plus", "EDGE", "Plus call, EDGE", "pe"), ("plus", "interior", "Plus call, interior", "pi"),
       ("none", "EDGE", "No call, EDGE", "ne"), ("none", "interior", "No call, interior", "ni"),
       ("lowint", "interior", "Low Hoop G, interior (no call)", "li")]
out = ["| Group (30+ NFL rushes) | Rushers | Pressure rate (forecast) | Clearly beat forecast [90%] | Clearly below |",
       "|---|---|---|---|---|"]
for g_, rl, lab, tg in CAL:
    d_ = sc[(sc.grp == g_) & (sc.role == rl)]
    if not len(d_):
        continue
    k_, n_ = int(d_.up.sum()), len(d_)
    lo_, hi_ = cp(k_, n_)
    rate = 100 * d_.pressures.sum() / d_.rush_snaps.sum()
    fcst = 100 * np.average(d_.pb, weights=d_.rush_snaps)
    out.append(f"| {lab} | {ph(f'cal_{tg}_n', n_)} | {ph(f'cal_{tg}_rate', rate, 1)}% ({ph(f'cal_{tg}_fc', fcst, 1)}%) | "
               f"{ph(f'cal_{tg}_up', k_)} of {n_} [{ph(f'cal_{tg}_uplo', 100 * lo_, 0)}–{ph(f'cal_{tg}_uphi', 100 * hi_, 0)}%] | "
               f"{ph(f'cal_{tg}_dn', int(d_.dn.sum()))} |")
MD["calib"] = "\n".join(out)
for tg, m_ in [("tune", sc.draft_year <= 2024), ("clean", sc.draft_year == 2025)]:
    d_ = sc[m_ & (sc.grp == "plus")]
    k_, n_ = int(d_.up.sum()), len(d_)
    lo_, hi_ = cp(k_, n_)
    put(f"calp_{tg}_k", k_); put(f"calp_{tg}_n", n_)
    put(f"calp_{tg}_lo", 100 * lo_, 0); put(f"calp_{tg}_hi", 100 * hi_, 0)
    put(f"calp_{tg}_dn", int(d_.dn.sum()))
_nn = sc[sc.grp == "none"]
_lo, _hi = cp(int(_nn.up.sum()), len(_nn))
put("caln_k", int(_nn.up.sum())); put("caln_n", len(_nn)); put("caln_pct", 100 * _nn.up.mean(), 0)
put("caln_lo", 100 * _lo, 0); put("caln_hi", 100 * _hi, 0)
for c in classes:
    d_ = sc[(sc.draft_year == c) & (sc.grp == "plus")]
    e_ = sc[(sc.draft_year == c) & (sc.grp == "none")]
    put(f"calc_{c}_k", int(d_.up.sum())); put(f"calc_{c}_n", len(d_))
    put(f"calc_{c}_base", 100 * e_.up.mean(), 0)
# raw pressure rates, plus vs no call (pooled over roles), and the mean gap to the forecast
for g_, tg in [("plus", "p"), ("none", "n")]:
    d_ = sc[sc.grp == g_]
    put(f"raw_{tg}_rate", 100 * d_.pressures.sum() / d_.rush_snaps.sum(), 1)
    put(f"raw_{tg}_gap", 100 * (d_.pressures.sum() / d_.rush_snaps.sum() - np.average(d_.pb, weights=d_.rush_snaps)), 1)

# ---- decision card, 2025 class: every plus call, the two nearest misses, the lowest EDGE and the lowest interior
d25 = ba[ba.draft_year == 2025].sort_values("shr", ascending=False).reset_index(drop=True)
d25["rank"] = np.arange(1, len(d25) + 1)
pick_ = list(d25[d25.call == "plus"].index)
pick_ += list(d25[(d25.call == "neutral")].index[:2])
for rl in ["EDGE", "interior"]:
    pick_.append(d25[d25.role == rl].index[-1])
card = d25.loc[sorted(set(pick_))]


def card_call(r_):
    tc = pd.notna(r_.three_cone)
    if r_.call == "plus":
        return (("3-cone first; then " if tc else "") + f"break ties his way vs {tie_range(r_)}")
    if r_.call == "minus" and r_.role == "interior":
        return "no call: low Hoop G here mostly means heavy; film flag only"
    if r_.call == "minus":
        return "move down a tie vs EDGEs near his slot"
    if r_.shr > 0:
        return "close, but no call: " + ("use his 3-cone" if tc else "ask for more hoop runs")
    return "no call" + (": use his 3-cone" if tc else "")


out = ["| Player (pick, role) | Hoop G, sd [80%] | 3-cone, s | Draft-day call | Rookie pressure rate vs forecast |",
       "|---|---|---|---|---|"]
for i, r_ in enumerate(card.itertuples()):
    k_ = f"dc_{i}"
    hg = f"{ph(k_ + '_s', r_.shr)} [{ph(k_ + '_l', r_.shr_lo)}, {ph(k_ + '_h', r_.shr_hi)}]"
    tc_ = ph(k_ + "_tc", r_.three_cone) if pd.notna(r_.three_cone) else "none"
    if pd.notna(r_.pb):
        res_ = (f"{ph(k_ + '_pr', 100 * r_.press_rate, 1)}% vs {ph(k_ + '_pb', 100 * r_.pb, 1)}% "
                f"({int(r_.rush_snaps)} rushes, {result_txt(r_).replace('**', '')})")
    else:
        res_ = "no NFL rushes yet" if r_.rush_snaps == 0 else f"{int(r_.rush_snaps)} rushes: too few to score"
    out.append(f"| **{r_.display_name}** ({pk_txt(r_.pick)}, {ROLE_TXT[r_.role]}) | {hg} | {tc_} | {card_call(r_)} | {res_} |")
MD["card"] = "\n".join(out)
put("card_n", len(card))
TABLES["card"] = card[["display_name", "role", "pick", "shr", "three_cone", "call"]].to_dict("records")
put("pct_3cone_25", 100 * d25.three_cone.notna().mean(), 0)
# plain-language out-of-class number: points of pressure-rate variance explained, held-out classes pooled
put("wr2_pts", 100 * WN["pooled"], 1); put("wr2_pts_lo", 100 * np.percentile(WB["pooled"], 5), 1)
put("wr2_pts_hi", 100 * np.percentile(WB["pooled"], 95), 1)
for c in classes:
    put(f"wr2_pts_{c}", 100 * WN[int(c)], 1)

# ============================================================================= revised round 8 additions
_tick("revised: complete-case stopwatch comparison")
# (1) Complete cases, no imputed times. Hoop G fixed in advance (no screen). Leave one class out, rush-weighted least
#     squares of the pressure rate; held-out R² change (percentage points) and held-out beta-binomial log score per
#     rusher; 90% intervals by resampling held-out rushers within class (fold models fixed).
KAP = float(Mw)
CCB = ["log_pick", "combine_weight_z", "edge_share", "ngs_athleticism_score"]


def bb_ll(y, n, p):
    p = np.clip(p, 0.005, 0.6)
    al, bb_ = p * KAP, (1 - p) * KAP
    return (gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1) + gammaln(y + al) + gammaln(n - y + bb_)
            - gammaln(n + KAP) - gammaln(al) - gammaln(bb_) + gammaln(KAP))


def loco_cc(rows, base_cols, add_cols, seed):
    Y, W, P0, P1, L0, L1 = {}, {}, {}, {}, {}, {}
    for c in classes:
        tr, te = rows[rows.draft_year != c], rows[rows.draft_year == c]
        P0[c], P1[c] = wls_pred(tr, te, base_cols), wls_pred(tr, te, base_cols + add_cols)
        Y[c], W[c] = (te.pressures / te.rush_snaps).values, te.rush_snaps.values
        L0[c], L1[c] = bb_ll(te.pressures.values, W[c], P0[c]), bb_ll(te.pressures.values, W[c], P1[c])
    cat = lambda D, ii=None: np.concatenate([D[c] if ii is None else D[c][ii[c]] for c in classes])
    g_ = lambda ii=None: 100 * (wr2(cat(Y, ii), cat(P1, ii), cat(W, ii)) - wr2(cat(Y, ii), cat(P0, ii), cat(W, ii)))
    l_ = lambda ii=None: float(np.mean(cat(L1, ii) - cat(L0, ii)))
    rng_, B = np.random.default_rng(seed), []
    for _ in range(2000):
        ii = {c: rng_.integers(0, len(Y[c]), len(Y[c])) for c in classes}
        B.append((g_(ii), l_(ii)))
    B = np.array(B)
    return dict(n=int(sum(len(Y[c]) for c in classes)), r2=g_(), r2_lo=np.percentile(B[:, 0], 5),
                r2_hi=np.percentile(B[:, 0], 95), ls=l_(), ls_lo=np.percentile(B[:, 1], 5), ls_hi=np.percentile(B[:, 1], 95))


ccr = dl[dl.hoop_z.notna() & (dl.rush_snaps > 0)].dropna(subset=CCB).copy()
ccr["pressures"] = ccr.pressures.fillna(0)
put("cc_all_hoop3", int((dl.hoop_z.notna() & dl.three_cone.notna()).sum()))
c3r = ccr[ccr.three_cone_z.notna()]
n3r = ccr[ccr.three_cone_z.isna()]
CC = {"cc3cone": loco_cc(c3r, CCB, ["three_cone_z"], 61),            # what the 3-cone adds
      "cc3hoop": loco_cc(c3r, CCB + ["three_cone_z"], ["hoop_z"], 62),  # Hoop G on top of an observed 3-cone
      "ccnohoop": loco_cc(n3r, CCB, ["hoop_z"], 63)}                   # Hoop G where the 3-cone was skipped
for tg, d_ in CC.items():
    put(f"{tg}_n", d_["n"])
    for k in ["r2", "r2_lo", "r2_hi"]:
        put(f"{tg}_{k}", d_[k], 1)
    for k in ["ls", "ls_lo", "ls_hi"]:
        put(f"{tg}_{k}", d_[k], 3)
TABLES["complete_case"] = CC

# (2) Frozen decision rule, scored out of class. Rule (fixed since round 6): plus call = 80% interval of the
#     noise-corrected Hoop G clears zero. Hit = 80% interval of his actual pressure rate is above his out-of-class
#     no-tracking forecast. Matched base rate: no-call rushers reweighted to the plus calls' role-by-class mix.
_tick("revised: frozen-rule calibration and permutation")


def hits(frame):
    d_ = frame[frame.pb.notna()]
    return d_, (d_.pr_lo > d_.pb)


def matched_base(sc_, plus_mask):
    nc_ = sc_[~plus_mask]
    up_ = nc_.pr_lo > nc_.pb
    mix = sc_[plus_mask].groupby(["role", "draft_year"]).size()
    rates = up_.groupby([nc_.role, nc_.draft_year]).mean()
    return float(sum(mix[k] * rates.get(k, up_.mean()) for k in mix.index) / mix.sum())


scd, up_all = hits(ba)
plus_m = (scd.call == "plus").values
for tg, m_ in [("all", np.ones(len(scd), bool)), ("tune", (scd.draft_year <= 2024).values),
               ("clean", (scd.draft_year == 2025).values)]:
    k_, n_ = int(up_all[plus_m & m_].sum()), int((plus_m & m_).sum())
    lo_, hi_ = cp(k_, n_)
    put(f"fr_{tg}_k", k_); put(f"fr_{tg}_n", n_)
    put(f"fr_{tg}_lo", 100 * lo_, 0); put(f"fr_{tg}_hi", 100 * hi_, 0)
    mb_ = matched_base(scd[m_], plus_m[m_])
    put(f"fr_{tg}_base", 100 * mb_, 0)
put("fr_edge_tune_hits", int(up_all[plus_m & (scd.role == "EDGE").values & (scd.draft_year <= 2024).values].sum()))
_nc = scd[~plus_m]
lo_, hi_ = cp(int((_nc.pr_lo > _nc.pb).sum()), len(_nc))
put("fr_nc_lo", 100 * lo_, 0); put("fr_nc_hi", 100 * hi_, 0)
gap_obs = up_all[plus_m].mean() - up_all[~plus_m].mean()
put("fr_gap", 100 * gap_obs, 0)
# permutation: shuffle Hoop G (with its run count) across runners within class, re-make the calls, recompute the gap
prm, gaps = np.random.default_rng(71), []
for _ in range(5000):
    sh_ = ba.copy()
    for c in classes:
        ix = sh_.index[sh_.draft_year == c]
        sh_.loc[ix, ["shr_lo"]] = sh_.loc[prm.permutation(ix), "shr_lo"].values
    d_, u_ = hits(sh_)
    pm_ = (d_.shr_lo > 0).values
    if pm_.sum() and (~pm_).sum():
        gaps.append(u_[pm_].mean() - u_[~pm_].mean())
gaps = np.array(gaps)
put("fr_perm_p", (1 + (gaps >= gap_obs - 1e-12).sum()) / (1 + len(gaps)))
put("fr_perm_n", len(gaps))

# (3) Within-player mechanism check at matched speed: sideways g on his random NFL rushes only in half-seconds run at
#     hoop speed (2.5-4.5 yd/s). Mixed model (random player intercept) of snap-level matched sideways g on Hoop G,
#     weight, role and speed; and the drill-NFL correlation corrected for the noise in both measures.
_tick("revised: matched-speed mechanism check")
VLO, VHI = 2.5, 4.5


def matched_features():
    out_ = []
    for key, h in gfa.groupby(K, sort=False):
        if pinfo.loc[key].sample_kind != "random":
            continue
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
        inw = (h.frame.values >= sf) & (h.frame.values <= e)
        ra = pd.Series(np.where(inw, al, np.nan)).rolling(5, center=True, min_periods=5).mean().to_numpy()
        rv = pd.Series(np.where(inw, v, np.nan)).rolling(5, center=True, min_periods=5).mean().to_numpy()
        ok = (rv >= VLO) & (rv <= VHI) & np.isfinite(ra)
        if not ok.any():
            continue
        j = np.flatnonzero(ok)[int(np.argmax(ra[ok]))]
        out_.append(dict(zip(K, key), mg=ra[j] / YD_PER_G, mv=rv[j]))
    return out_


msn = pd.DataFrame(cached("matched_v", matched_features))
msn = msn.merge(dl[["nfl_id", "hoop_z", "hoop_g", "hoop_k", "combine_weight_z", "edge", "display_name"]], on="nfl_id")
msn = msn[msn.hoop_z.notna()].copy()
put("mx_snaps", len(msn)); put("mx_players", msn.nfl_id.nunique())
put("mx_med", msn.mg.median())
mfit = smf.mixedlm("mg ~ hoop_z + combine_weight_z + edge + mv", msn, groups=msn["nfl_id"]).fit(reml=True)
put("mx_b", mfit.params["hoop_z"], 3)
put("mx_b_lo", mfit.conf_int().loc["hoop_z", 0], 3); put("mx_b_hi", mfit.conf_int().loc["hoop_z", 1], 3)
# player level: partial correlation (after weight, role) of Hoop G with his mean matched-speed NFL sideways g,
# divided by sqrt(reliability of Hoop G x reliability of his NFL mean); player bootstrap re-estimates both
mpl = msn.groupby("nfl_id").agg(mg=("mg", "mean"), k=("mg", "size"), hoop_z=("hoop_z", "first"),
                                hoop_g=("hoop_g", "first"), hoop_k=("hoop_k", "first"),
                                combine_weight_z=("combine_weight_z", "first"), edge=("edge", "first"),
                                name=("display_name", "first")).reset_index()
mpl = mpl[mpl.k >= 5].reset_index(drop=True)
msn5 = msn[msn.nfl_id.isin(mpl.nfl_id)]
SNP = {i: g_.mg.values for i, g_ in msn5.groupby("nfl_id")}


def corr_att(P):
    A = np.c_[np.ones(len(P)), P.combine_weight_z.values, P.edge.values]
    r_ = _pr(P.hoop_z.values, P.mg.values, A)
    vals = [SNP[i] for i in P.nfl_id]
    ids = np.repeat(np.arange(len(vals)), [len(v) for v in vals])
    i1, _ = icc(ids, np.concatenate(vals))
    rel_n = float(np.clip(sb(i1, P.k.mean()), 0.05, 1))
    rel_h = float(np.mean(P.hoop_k * rel_hoop / (1 + (P.hoop_k - 1) * rel_hoop)))
    return r_, r_ / np.sqrt(rel_h * rel_n), rel_n, i1


r_m, rc_m, reln_m, icc_m = corr_att(mpl)
mrng, mb_ = np.random.default_rng(83), []
for _ in range(1000):
    mb_.append(corr_att(mpl.iloc[mrng.integers(0, len(mpl), len(mpl))].reset_index(drop=True))[:2])
mb_ = np.array(mb_)
put("mx_n", len(mpl)); put("mx_r", r_m); put("mx_rc", rc_m)
put("mx_rc_lo", np.percentile(mb_[:, 1], 5)); put("mx_rc_hi", np.percentile(mb_[:, 1], 95))
put("mx_rel_nfl", reln_m); put("mx_icc", icc_m)
put("rel_hoop_2runs", sb(rel_hoop, 2))
TABLES["mechanism"] = dict(mixed=str(mfit.summary()), r=r_m, rc=rc_m, boot=mb_.tolist())

# (4) Leave-one-class-out calibration table for the writeup: raw pressure rate per group, exact 90% intervals
_tick("revised: tables")
scd["hit"] = up_all.values
GR = [("Plus call, 2023–24 classes (tuned the rule)", plus_m & (scd.draft_year <= 2024).values, "t"),
      ("Plus call, 2025 class (clean test)", plus_m & (scd.draft_year == 2025).values, "c"),
      ("No call, EDGE", ~plus_m & (scd.role == "EDGE").values, "ne"),
      ("No call, interior", ~plus_m & (scd.role == "interior").values, "ni")]
out = ["| Group (30+ NFL rushes, 2023–25) | Rushers | Pressure rate (forecast) | Beat forecast [exact 90%] |",
       "|---|---|---|---|"]
for lab, m_, tg in GR:
    d_ = scd[m_]
    k_, n_ = int(d_.hit.sum()), len(d_)
    lo_, hi_ = cp(k_, n_)
    rate = 100 * d_.pressures.sum() / d_.rush_snaps.sum()
    fcst = 100 * np.average(d_.pb, weights=d_.rush_snaps)
    out.append(f"| {lab} | {ph(f'lt_{tg}_n', n_)} | {ph(f'lt_{tg}_rate', rate, 1)}% ({ph(f'lt_{tg}_fc', fcst, 1)}%) | "
               f"{ph(f'lt_{tg}_k', k_)} of {n_} [{ph(f'lt_{tg}_lo', 100 * lo_, 0)}–{ph(f'lt_{tg}_hi', 100 * hi_, 0)}%] |")
MD["calib"] = "\n".join(out)

# (5) 2025 decision card: the plus calls and the lowest interior hoop (film flag), one call and one film cue each
CARD_NAMES = list(d25[d25.call == "plus"].display_name) + [d25[d25.role == "interior"].iloc[-1].display_name]
FILM = {("plus", "EDGE"): "Flattens at the top of the rush, or only redirects late?",
        ("plus", "interior"): "Wins laterally on games and stunts?",
        ("minus", "interior"): "Can he still redirect on stunts at his weight?"}


def card_call2(r_):
    if r_.call == "plus":
        return ("3-cone first; if tied, take him" if pd.notna(r_.three_cone) else "Take him in a tie") + \
            f" over {tie_range(r_)}"
    return "No call: low here mostly means heavy"


out = ["| 2025 player (pick, role) | Hoop G, sd [80%] | 3-cone | Draft-day call | Film cue | Rookie year (rushes) |",
       "|---|---|---|---|---|---|"]
for i, nm in enumerate(CARD_NAMES):
    r_ = d25[d25.display_name == nm].iloc[0]
    k_ = f"cd_{i}"
    hg = f"{ph(k_ + '_s', r_.shr)} [{ph(k_ + '_l', r_.shr_lo)}, {ph(k_ + '_h', r_.shr_hi)}]"
    tc_ = ph(k_ + "_tc", r_.three_cone) + " s" if pd.notna(r_.three_cone) else "not run"
    if pd.notna(r_.pb):
        res_ = (f"{ph(k_ + '_pr', 100 * r_.press_rate, 1)}% vs {ph(k_ + '_pb', 100 * r_.pb, 1)}% forecast "
                f"({int(r_.rush_snaps)}; {result_txt(r_).replace('**', '').replace('in noise', 'within noise')})")
    else:
        res_ = f"too few to score ({int(r_.rush_snaps)})"
    out.append(f"| **{r_.display_name}** ({pk_txt(r_.pick)}, {ROLE_TXT[r_.role]}) | {hg} | {tc_} | {card_call2(r_)} | "
               f"{FILM[(r_.call, r_.role)]} | {res_} |")
MD["card"] = "\n".join(out)

# ============================================================================= FIGURES
_tick("figures")


# ---- Fig 1 (hero): two EDGE plus calls (one beat his forecast, one did not) and the top / bottom interior hoop, each with
#      the hoop path; sideways g of the hoop run and of his hardest-turning sampled NFL rush on one time axis (aligned at
#      each peak); NFL pressure rate against the out-of-class forecast. Bottom strip: every runner, drill vs NFL turn.
F1_NAMES = ["Javon Solomon", "Donovan Ezeiruaku", "Adetomiwa Adebawore", "Jordan Phillips"]
F1_TAG = {"Javon Solomon": "Plus call, beat his forecast", "Donovan Ezeiruaku": "Plus call, has not beaten it yet",
          "Adetomiwa Adebawore": "Best interior hoop: on forecast", "Jordan Phillips": "Low interior hoop: film flag only"}
cmap = plt.get_cmap("cividis"); norm = plt.Normalize(0, 0.9)
F1 = {}
for nm in F1_NAMES:
    r_ = ba[ba.display_name == nm].iloc[0]
    tr = traces[r_.hoop_eid]
    xs_, ys_ = tr["_x"] - (tr["_x"].max() + tr["_x"].min()) / 2, tr["_y"] - (tr["_y"].max() + tr["_y"].min()) / 2
    if np.ptp(ys_) < np.ptp(xs_):
        xs_, ys_ = ys_, xs_
    roll_h = pd.Series(tr["_alat"]).rolling(5, center=True).mean().to_numpy()
    q_ = arcp[(arcp.nfl_id == r_.nfl_id) & (arcp.kind == "random")]
    kk = q_.loc[q_.arc_g.idxmax()]
    la, fw, al, kmx, wty = arc_tr[(kk.game_id, kk.play_id, kk.nfl_id)]
    roll_n = pd.Series(al).rolling(5, center=True, min_periods=5).mean().to_numpy()
    F1[nm] = dict(r=r_, hx=xs_, hy=ys_, hal=tr["_alat"], hroll=roll_h, hk=int(np.nanargmax(roll_h)),
                  hpk=float(np.nanmax(roll_h)), nroll=roll_n, nk=int(np.nanargmax(roll_n)), npk=float(np.nanmax(roll_n)),
                  nmed=float(q_.arc_g.median()), nrush=len(q_))
HSPAN = max(max(np.ptp(v["hx"]), np.ptp(v["hy"])) for v in F1.values()) + 1.5
FH1 = 17.2
IN = lambda h: h / FH1          # inches -> figure fraction (vertical)
fig = plt.figure(figsize=(FIGW, FH1))
fig.text(0.012, 1 - IN(0.15), "Hoop G in the drill and on Sundays: two EDGE plus calls, the top and bottom interior hoop",
         fontsize=16.5, weight="bold", va="top")
fig.text(0.012, 1 - IN(0.55), "Middle: half-second sideways g of his hoop run (blue) and of his hardest-turning NFL rush out "
         "of about 20 random ones\n(orange), lined up at each peak; dotted line = his typical NFL rush. Right: NFL pressure "
         "rate, dot = actual with 80% interval,\ntick = no-tracking forecast from a model fit on the other two draft classes.",
         fontsize=12, va="top", color="#333333")
AH_IN, ROW_IN, TOP_IN = 1.75, 2.62, 1.75
AH = IN(AH_IN)
AW = AH * FH1 / FIGW
X1, W1_ = 0.03 + AW + 0.075, 0.42
X2 = X1 + W1_ + 0.06
W2 = 1 - X2 - 0.02
XMAXP = float(np.ceil(100 * max(F1[n_]["r"].pr_hi for n_ in F1_NAMES) + 2))
for idx, nm in enumerate(F1_NAMES):
    v = F1[nm]; r_ = v["r"]
    yt = 1 - IN(TOP_IN + idx * ROW_IN)
    ybot = yt - IN(0.42) - AH
    fig.text(0.012, yt, f"{nm} ({'EDGE' if r_.role == 'EDGE' else 'interior'}). {F1_TAG[nm]}", fontsize=14, weight="bold",
             va="top")
    fig.text(0.012, yt - IN(0.24), f"{r_.draft_year} pick {pk_txt(r_.pick)} · {int(r_.combine_weight)} lb · Hoop G "
             f"{'+' if r_.shr > 0 else '−'}{abs(r_.shr):.1f} sd · 3-cone "
             f"{'%.2f s' % r_.three_cone if pd.notna(r_.three_cone) else 'not run'} · {int(r_.rush_snaps)} NFL rushes",
             fontsize=12, va="top", color="#333333")
    ax = fig.add_axes([0.03, ybot, AW, AH])
    pts = np.c_[v["hx"], v["hy"]].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=norm, linewidth=3.2)
    lc.set_array(v["hal"][:-1]); ax.add_collection(lc)
    ax.scatter([v["hx"][v["hk"]]], [v["hy"][v["hk"]]], s=140, facecolor="none", edgecolor="k", lw=2.2, zorder=3)
    ax.set_xlim(-HSPAN / 2, HSPAN / 2); ax.set_ylim(-HSPAN / 2, HSPAN / 2); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(True); sp_.set_color("#BBBBBB")
    if idx == 0:
        ax.plot([-HSPAN / 2 + 0.5, -HSPAN / 2 + 2.5], [HSPAN / 2 - 1.3] * 2, color="k", lw=2.5)
        ax.text(-HSPAN / 2 + 1.5, HSPAN / 2 - 1.0, "2 yd", ha="center", va="bottom", fontsize=11.5)
    tx = fig.add_axes([X1, ybot, W1_, AH])
    th_ = (np.arange(len(v["hroll"])) - v["hk"]) * DT
    tn_ = (np.arange(len(v["nroll"])) - v["nk"]) * DT
    tx.axhline(v["nmed"], color="#777777", lw=1.4, ls=":")
    mh_, mn_ = np.abs(th_) <= 2.0, np.abs(tn_) <= 2.0
    tx.plot(th_[mh_], v["hroll"][mh_], color=OI["blue"], lw=2.6)
    tx.plot(tn_[mn_], v["nroll"][mn_], color=OI["orange"], lw=2.6)
    tx.text(2.15, v["hpk"], f"hoop {v['hpk']:.2f} g", color=OI["blue"], fontsize=12.5, weight="bold", va="center")
    tx.text(2.15, min(v["npk"], v["hpk"] - 0.15), f"NFL best {v['npk']:.2f} g", color="#B06E00", fontsize=12.5,
            weight="bold", va="center")
    _ny = min(v["npk"], v["hpk"] - 0.15)
    _ty = v["nmed"] + 0.02 if abs(v["nmed"] + 0.05 - _ny) > 0.16 else v["nmed"] - 0.02
    tx.text(2.15, _ty, f"typical rush {v['nmed']:.2f}", color="#555555", fontsize=11.5,
            va="bottom" if _ty > v["nmed"] else "top")
    tx.set_xlim(-2, 3.6); tx.set_ylim(0, 0.95); tx.set_yticks([0, 0.3, 0.6, 0.9]); tx.set_xticks([-2, -1, 0, 1, 2])
    tx.set_ylabel("sideways g", fontsize=12)
    if idx == 3:
        tx.set_xlabel("seconds from his peak (0 = the half-second that sets the number)", fontsize=11.5)
    bx = fig.add_axes([X2, ybot + AH * 0.25, W2, AH * 0.5])
    col_ = OI["blue"] if r_.pr_lo > r_.pb else (OI["red"] if r_.pr_hi < r_.pb else "#555555")
    bx.plot([100 * r_.pr_lo, 100 * r_.pr_hi], [0, 0], color="#D0D0D0", lw=11, solid_capstyle="butt")
    bx.scatter([100 * r_.press_rate], [0], s=150, color=col_, zorder=3)
    bx.scatter([100 * r_.pb], [0], marker="|", s=400, color="k", lw=2.8, zorder=4)
    bx.text(0.0, 1.25, f"{100 * r_.press_rate:.1f}% vs {100 * r_.pb:.1f}% forecast", transform=bx.transAxes,
            fontsize=13, weight="bold", color=col_, va="bottom")
    bx.set_xlim(0, XMAXP); bx.set_ylim(-1, 1); bx.set_yticks([])
    bx.spines["left"].set_visible(False)
    bx.set_xlabel("NFL pressure rate, %", fontsize=11.5)
    key = nm.split(" ")[-1].lower().replace("'", "")
    put(f"f1_{key}_hoop", v["hpk"]); put(f"f1_{key}_nfl", v["npk"]); put(f"f1_{key}_med", v["nmed"])
    put(f"f1_{key}_pr", 100 * r_.press_rate, 1); put(f"f1_{key}_pb", 100 * r_.pb, 1)
    put(f"f1_{key}_wt", int(r_.combine_weight)); put(f"f1_{key}_n", int(r_.rush_snaps))
    put(f"f1_{key}_pick", int(r_.pick))
# bottom strip: every hoop runner with 5+ random NFL rushes, drill Hoop G vs his average matched-speed NFL sideways g
ys0 = 1 - IN(TOP_IN + 4 * ROW_IN + 0.3)
fig.text(0.012, ys0, f"Every hoop runner with 5+ random NFL rushes (n = {len(mpl)}): beyond weight and role, the drill does "
         "not show up", fontsize=14, weight="bold", va="top")
fig.text(0.012, ys0 - IN(0.27), "NFL value = his average peak half-second sideways g while moving 2.5–4.5 yd/s (hoop speed).\n"
         f"After weight and role, noise-corrected r = {NUM['mx_rc']} [90% {mn(NUM['mx_rc_lo'])}, {NUM['mx_rc_hi']}].",
         fontsize=12, va="top", color="#333333")
sx = fig.add_axes([0.09, IN(1.05), 0.58, IN(2.7)])
for rl, c_, lab in [("EDGE", OI["blue"], "EDGE"), ("interior", OI["orange"], "interior")]:
    d_ = mpl[(mpl.edge == 1) == (rl == "EDGE")]
    sx.scatter(d_.hoop_g, d_.mg, s=45, color=c_, alpha=0.85, edgecolor="white", lw=0.6, label=f"{lab} (n={len(d_)})")
for nm in F1_NAMES:
    d_ = mpl[mpl.name == nm]
    if len(d_):
        sx.scatter(d_.hoop_g, d_.mg, s=150, facecolor="none", edgecolor="k", lw=2.0, zorder=4)
        sx.annotate(nm.split(" ")[-1], (d_.hoop_g.iloc[0], d_.mg.iloc[0]), xytext={"Javon Solomon": (0, 12), "Donovan Ezeiruaku": (-10, 8)}.get(nm, (8, 6)),
                    textcoords="offset points", fontsize=12, weight="bold",
                    ha="right" if nm == "Donovan Ezeiruaku" else ("center" if nm == "Javon Solomon" else "left"))
sx.set_xlabel("Hoop G in the drill, g (one run unless two were recorded)", fontsize=12)
sx.set_ylabel("NFL sideways g\nat hoop speed", fontsize=12)
sx.legend(loc="upper left", frameon=False, fontsize=11.5, ncol=2)
sx.set_ylim(mpl.mg.min() - 0.02, mpl.mg.max() + 0.06)
fig.text(0.735, IN(3.6), "How to read it", fontsize=12.5, weight="bold", va="top")
fig.text(0.735, IN(3.3), f"The upward drift is role and size:\nlighter EDGEs turn harder in both.\nWithin role, after weight, "
         f"the\nlink is near zero. One NFL rush is\nmostly noise (per-snap agreement\n{NUM['mx_icc']}), so the orange "
         f"trace is a\nfilm cue, not evidence.",
         fontsize=11.5, va="top", color="#333333")
cax = fig.add_axes([0.735, IN(0.95), 0.2, IN(0.13)])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[0, 0.3, 0.6, 0.9])
cb.ax.tick_params(labelsize=10.5); cax.set_title("hoop path colour: sideways g", fontsize=11, loc="left", pad=3)
fig.legend(handles=[Line2D([], [], color=OI["blue"], lw=2.6, label="hoop run"),
                    Line2D([], [], color=OI["orange"], lw=2.6, label="hardest NFL rush"),
                    Line2D([], [], marker="o", color="#555555", lw=0, markersize=10, label="actual pressure rate"),
                    Line2D([], [], color="#D0D0D0", lw=9, label="its 80% interval"),
                    Line2D([], [], marker="|", color="k", lw=0, markersize=16, markeredgewidth=2.6,
                           label="no-tracking forecast")],
           loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=5, fontsize=11, frameon=False)
save(fig, "fig1_hero.png")

# ---- Fig 2: (a) five draft-day measures on one common set of rushers, sorted; (b) held-out gain, with the shuffle band
fig = plt.figure(figsize=(FIGW, 6.6))
axa = fig.add_axes([0.25, 0.14, 0.27, 0.62]); axb = fig.add_axes([0.765, 0.14, 0.215, 0.62])
CS = sorted(CMPR, key=lambda d_: d_["r"])
yy = np.arange(len(CS))
for y_, d_ in zip(yy, CS):
    col_ = OI["blue"] if d_["col"] == "hoop_z" else OI["black"]
    axa.plot([d_["lo"], d_["hi"]], [y_, y_], color=col_, lw=3, solid_capstyle="butt")
    axa.scatter([d_["r"]], [y_], color=col_, s=110, zorder=3)
    axa.scatter([d_["r_own"]], [y_ - 0.3], marker="o", facecolor="white", edgecolor=col_, s=55, lw=1.6, zorder=3)
axa.axvline(0, color="k", lw=0.9)
axa.set_yticks(yy)
axa.set_yticklabels([f"{d_['label']}\nn = {d_['n']} (all: {d_['n_own']})" for d_ in CS], fontsize=11.5)
axa.set_xlim(-0.75, 0.85); axa.set_ylim(-0.7, len(CS) - 0.4); axa.set_xticks([-0.5, 0, 0.5])
axa.set_xlabel("adjusted correlation with NFL pressure rate", fontsize=12)
axa.set_title("(a) Five draft-day measures, same rushers", fontsize=13.5, weight="bold", loc="left", x=-0.6)
rows3 = [(f"Nested test, all {WN['n']}", WN["pooled"], np.percentile(WB["pooled"], [5, 95]), OI["blue"])]
for c in classes:
    rows3.append((f"  {c} class held out (n={len(_Y[c])})", WN[int(c)], np.percentile(WB[c], [5, 95]), "#444444"))
rows3.append((f"No 3-cone, Hoop G (n={CC['ccnohoop']['n']})", CC["ccnohoop"]["r2"] / 100,
              np.array([CC["ccnohoop"]["r2_lo"], CC["ccnohoop"]["r2_hi"]]) / 100, OI["green"]))
rows3.append((f"Has 3-cone, Hoop G on top (n={CC['cc3hoop']['n']})", CC["cc3hoop"]["r2"] / 100,
              np.array([CC["cc3hoop"]["r2_lo"], CC["cc3hoop"]["r2_hi"]]) / 100, OI["red"]))
yb = np.array([5.6, 4.6, 3.6, 2.6, 1.2, 0.0])
lo_a, hi_a = 100 * np.percentile(NULL[:, 1], [5, 95])
XL = (-30, 35)
axb.axvspan(lo_a, hi_a, color="#DDDDDD", lw=0, zorder=0)
axb.text((lo_a + hi_a) / 2, 6.25, f"shuffled Hoop G\n(nested p = {NUM['perm_p_r2']})", fontsize=10.5, zorder=5,
         bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.9), color="#555555", ha="center", va="bottom")
for y_, (lab, g_, ci_, col) in zip(yb, rows3):
    lo_, hi_ = np.clip(100 * ci_, XL[0], XL[1])
    axb.plot([lo_, hi_], [y_, y_], color=col, lw=2.6)
    if 100 * ci_[0] < XL[0]:
        axb.annotate("", xy=(XL[0], y_), xytext=(XL[0] + 4, y_), arrowprops=dict(arrowstyle="->", color=col, lw=2))
    gx = float(np.clip(100 * g_, XL[0] + 1, XL[1]))
    axb.scatter([gx], [y_], color=col, s=90, zorder=3, edgecolor="white", marker="o" if 100 * g_ > XL[0] else "<")
    axb.text(gx + (5 if 100 * g_ < XL[0] else 0), y_ + 0.3, f"{100 * g_:+.1f}", fontsize=11, ha="center", color=col, weight="bold")
axb.axvline(0, color="k", lw=0.9); axb.set_ylim(-0.6, 7.0); axb.set_xlim(*XL)
axb.set_yticks(yb); axb.set_yticklabels([r[0] for r in rows3], fontsize=11.5)
axb.set_xlabel("change in R-squared,\npercentage points", fontsize=12)
axb.set_title("(b) Scored on a class it never saw", fontsize=13.5, weight="bold",
              loc="left", x=-1.25)
fig.suptitle("Where both exist the 3-cone is the better number; Hoop G leans positive only where it was skipped", fontsize=14,
             weight="bold", x=0.012, ha="left", y=0.975)
fig.text(0.012, 0.915, "(a) Filled dot and 95% bootstrap interval: after pick, weight, role and NGS score (pick and NGS: after "
         "weight and role);\nopen dot: every rusher with that measure. (b) Model with Hoop G minus the no-tracking model, "
         "rush-weighted, 90% intervals.\nGrey strip: middle 90% of nested-test gains when Hoop G is shuffled within class.",
         fontsize=11, color="#333333", va="top")
save(fig, "fig2_validation.png")

# ---- Fig 3: every scored plus call against its out-of-class forecast, split into the tuning classes and 2025
cv = ba[(ba.call == "plus") & ba.pb.notna()].copy()
cv["res"] = cv.press_rate - cv.pb
SECS = [("tune", "2023–24 classes (helped choose the drill)", cv[cv.draft_year <= 2024]),
        ("clean", "2025 class (not used to choose the drill)", cv[cv.draft_year == 2025])]
NOISE = "#666666"
ylist, rowsd, heads = [], [], []
y = 0.0
for tg, gt, d_ in SECS:
    heads.append((y, tg, gt, d_.sort_values("res", ascending=False))); y += 2.0
    for r_ in heads[-1][3].itertuples():
        ylist.append(y); rowsd.append(r_); y += 1.55
    y += 0.4
NR = y
HH = 0.62 * NR + 2.6
fig = plt.figure(figsize=(FIGW, HH))
ax = fig.add_axes([0.52, 0.95 / HH, 0.31, 1 - 2.75 / HH])
ax.spines["left"].set_visible(False)
XMAX = float(np.ceil(100 * cv.pr_hi.max() + 1))
for yv, r_ in zip(ylist, rowsd):
    clear_up, clear_dn = r_.pr_lo > r_.pb, r_.pr_hi < r_.pb
    c_ = OI["blue"] if clear_up else (OI["red"] if clear_dn else NOISE)
    ax.plot([100 * r_.pr_lo, 100 * r_.pr_hi], [yv, yv], color="#D9D9D9", lw=10, solid_capstyle="round", zorder=1)
    ax.plot([100 * r_.pb, 100 * r_.press_rate], [yv, yv], color=c_, lw=2.6, zorder=2)
    ax.scatter([100 * r_.press_rate], [yv], s=150, zorder=4, lw=2.4, color=c_)
    ax.scatter([100 * r_.pb], [yv], marker="|", s=260, color="k", lw=2.6, zorder=3)
    lab_ = "beat forecast" if clear_up else ("below forecast" if clear_dn else "in noise")
    ax.text(1.02, yv, lab_, transform=ax.get_yaxis_transform(), fontsize=13.5, va="center", color=c_,
            weight="bold" if (clear_up or clear_dn) else "normal")
    tr = traces[r_.hoop_eid]
    sp = ax.inset_axes([-0.72, yv - 0.62, 0.6, 1.24], transform=ax.get_yaxis_transform())
    k_ = tr["_k"]
    tt_ = np.arange(len(tr["_alat"])) * DT
    sp.axvspan((k_ - 2) * DT, (k_ + 2) * DT, color="#BBBBBB", alpha=0.7, lw=0)
    sp.axhline(float(NUM["hoop_g_median"]), color="#999999", lw=1, ls=":")
    sp.plot(tt_, np.clip(tr["_alat"], 0, 1.2), color=OI["blue"], lw=1.8)
    sp.set_ylim(0, 1.2); sp.set_xlim(0, 9); sp.set_xticks([])
    sp.set_yticks([0, 0.6, 1.2]); sp.tick_params(axis="y", labelsize=10, length=2, pad=1)
    for sp_ in sp.spines.values():
        sp_.set_visible(True); sp_.set_color("#CCCCCC")
    sp.text(0.98, 0.95, f"{r_.hoop_g:.2f} g", transform=sp.transAxes, fontsize=12.5, va="top", ha="right", weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.85))
ax.set_yticks(ylist)
ax.set_yticklabels([f"{r_.display_name} ({'EDGE' if r_.role == 'EDGE' else 'Int'})" for r_ in rowsd], fontsize=13.5)
ax.tick_params(axis="y", pad=0.86 * 0.31 * FIGW * 72 + 6, length=0)
from matplotlib.transforms import blended_transform_factory
_bt = blended_transform_factory(fig.transFigure, ax.transData)
for y0, tg, gt, d_ in heads:
    k_, n_ = int(NUM[f"fr_{tg}_k"]), int(NUM[f"fr_{tg}_n"])
    lo_, hi_ = cp(k_, n_)
    base_ = float(NUM[f"fr_{tg}_base"])
    ax.text(0.012, y0 - 0.3, gt, transform=_bt, fontsize=14.5, weight="bold", va="center")
    ax.text(0.012, y0 + 0.55, f"{k_} of {n_} beat forecast (blue bar, exact 90% interval); orange: matched no-call rate "
            f"{NUM[f'fr_{tg}_base']}%", transform=_bt, fontsize=11.5, va="center", color="#333333")
    mb = ax.inset_axes([0.05, y0 - 0.65, 0.9, 0.7], transform=ax.get_yaxis_transform())
    mb.axvspan(max(base_ - 1.2, 0), base_ + 1.2, color=OI["orange"], alpha=0.95, lw=0)
    mb.plot([100 * lo_, 100 * hi_], [0, 0], color=OI["blue"], lw=3)
    mb.scatter([100 * k_ / n_], [0], s=80, color=OI["blue"], zorder=3)
    mb.set_xlim(0, 100); mb.set_ylim(-1, 1); mb.set_yticks([]); mb.set_xticks([0, 50, 100])
    mb.set_xticklabels(["0%", "50%", "100%"])
    mb.tick_params(axis="x", labelsize=10, length=2, pad=1)
    for s_ in ["left", "right", "top"]:
        mb.spines[s_].set_visible(False)
ax.set_ylim(NR - 0.3, -0.9); ax.grid(axis="x", color="#EEEEEE", lw=0.6)
ax.set_xlim(0, XMAX); ax.set_xlabel("NFL pressure rate, %", fontsize=13.5)
ax.tick_params(axis="x", labelsize=12.5)
hand = [Line2D([], [], marker="|", color="k", lw=0, markersize=16, markeredgewidth=2.4, label="no-tracking forecast"),
        Line2D([], [], marker="o", color=NOISE, lw=0, markersize=11, label="actual"),
        Line2D([], [], color="#D9D9D9", lw=9, label="80% interval of actual"),
        Line2D([], [], color=OI["blue"], lw=1.8, label="hoop run, sideways g (0–1.2 g)")]
fig.legend(handles=hand, frameon=False, fontsize=12, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4)
fig.suptitle("Every plus hit came in the classes used to tune the rule; 2025 is in the noise so far", fontsize=15.5,
             weight="bold", x=0.012, ha="left", y=1 - 0.12 / HH)
fig.text(0.012, 1 - 0.5 / HH, f"Plus call = 80% interval of noise-corrected Hoop G above zero. Each class is forecast by "
         f"a no-tracking model fit on the other two.\nSparkline: his hoop run, 0–9 s (dotted: median run {NUM['hoop_g_median']} g; grey: the "
         "half-second that sets Hoop G).\n'Beat' = 80% interval of actual rate above forecast.",
         fontsize=11.5, color="#333333", va="top")
save(fig, "fig3_calls.png")

# ============================================================================= write
_tick("write")
TABLES["md"] = MD
if os.environ.get("BDB_ALLNUM"):
    json.dump(NUM, open(os.environ["BDB_ALLNUM"], "w"), indent=1, sort_keys=True)
used = render(OUT, NUM, MD)
keep = {k_: NUM[k_] for k_ in sorted(used)}     # exactly the numbers the text uses; claims.json maps each to its sentence
json.dump(keep, open(OUT / "numbers.json", "w"), indent=1, sort_keys=True)
json.dump(TABLES, open(OUT / "tables.json", "w"), indent=1, default=str)
if os.environ.get("BDB_ALLNUM"):
    json.dump(NUM, open(os.environ["BDB_ALLNUM"], "w"), indent=1, sort_keys=True)
_tick("done")

