"""Hoop G (Run-the-Hoop bend) vs NFL pass-rush pressure for DL/EDGE: one metric, tested honestly.

Run from the repo root:
    uv run python writeup/candidates/round_09/revised/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (only the values quoted in writeup.md), tables.json,
board_2023_2025.csv and figures/*.png next to this file, then renders writeup.md from writeup_template.md.
Frame-level feature extraction is cached in data/cache/ (gitignored; delete it to recompute from the frames).
Runtime: about 1 minute cached, about 6 minutes cold, one core.

Round 9 revised (from round 9 cand 1): adds a section 'round 10' that computes the EPA cost of a pressured snap, 3-cone
coverage on one denominator, a leave-one-class-out partial rho for the 3-cone with a refit bootstrap, one count model on
all 90 rushers with separate Hoop G slopes for 3-cone / no 3-cone runners (shared prior; replaces the 34-rusher stacked
least squares), the plus-call rule re-chosen on two classes and scored on the third (with and without Downs), the 2025
0-of-2 against base rates, matched no-call twins with a shuffle null, EDGE rush direction, a 4-column decision card, a
one-table test summary, and two redrawn figures (paths side by side; twins). Round 9 (cand 1, revising the round 8 winner): leads with the 3-cone and treats Hoop G as its stand-in; adds a
coach table (plus vs no call vs minus flag, by role, with n), a 7-player 2025 decision card with slot swaps, a table
of every test with its null, the matched base-rate computation, an observed-times-only calibration of the call
rule, the 2025 clean test as a posterior predictive probability (model fit on 2023-24), the record with Downs
scored, per-role drill-to-NFL mixed models, and three redrawn figures (3-row hero, one-panel held-out gain, calls
with pooled no-call rows). Earlier rounds: Design (round 8 revised, from round 8 cand 1). Revised additions: complete-case 3-cone / no-3-cone held-out comparison, frozen-rule calibration with matched base rate and permutation test, matched-speed within-player mixed model, redrawn figures. New in round 8: role-specific draft-room calls (a low interior
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


# ============================================================================= round 9 additions
_tick("round 9: motivation, coach table, calibration, posterior predictive, card")
from matplotlib.transforms import blended_transform_factory  # noqa: E402

# (R9-1) motivation and the stopwatch result we lead with
put("r9_hoop_all", int(dl.hoop_z.notna().sum()))
put("r9_dl_all", len(dl))
put("r9_3cone_all_pct", 100 * dl.three_cone.notna().mean(), 0)
put("r9_3cone_25_k", int(d25.three_cone.notna().sum())); put("r9_3cone_25_n", len(d25))
_r, _lo, _hi, _n, _ = sp_ci(-co.three_cone_z, co.press_eb)
put("r9_t3_raw", _r); put("r9_t3_raw_lo", _lo); put("r9_t3_raw_hi", _hi); put("r9_t3_raw_n", _n)
_r, _lo, _hi, _n, _ = sp_ci(co.hoop_z, co.press_eb)
put("r9_hoop_raw", _r); put("r9_hoop_raw_lo", _lo); put("r9_hoop_raw_hi", _hi); put("r9_hoop_raw_n", _n)

# (R9-2) coach table: raw pressure rate by call and role (30+ rushes, out-of-class forecast), with n
scd["below"] = (scd.pr_hi < scd.pb).values
scd["grp9"] = np.select([scd.call == "plus", scd.call == "minus"], ["plus", "minus"], "none")
COACH = [("Plus call, EDGE", "plus", "EDGE", "pe"), ("Plus call, interior", "plus", "interior", "pi"),
         ("No call, EDGE", "none", "EDGE", "ne"), ("No call, interior", "none", "interior", "ni"),
         ("Minus flag, interior (film check)", "minus", "interior", "mi")]
out = ["| Group, 2023–25 (30+ NFL rushes) | Rushers (2025) | Pressure rate (forecast) | Beat forecast [exact 90%] "
       "| Clearly below |", "|---|---|---|---|---|"]
COACH_ROWS = {}
for lab, g_, rl, tg in COACH:
    d_ = scd[(scd.grp9 == g_) & (scd.role == rl)]
    k_, n_ = int(d_.hit.sum()), len(d_)
    lo_, hi_ = cp(k_, n_)
    rate = 100 * d_.pressures.sum() / d_.rush_snaps.sum()
    fcst = 100 * np.average(d_.pb, weights=d_.rush_snaps)
    COACH_ROWS[tg] = dict(n=n_, rate=rate, fc=fcst, k=k_)
    out.append(f"| {lab} | {ph(f'co_{tg}_n', n_)} ({ph(f'co_{tg}_n25', int((d_.draft_year == 2025).sum()))}) | "
               f"{ph(f'co_{tg}_rate', rate, 1)}% ({ph(f'co_{tg}_fc', fcst, 1)}%) | "
               f"{ph(f'co_{tg}_k', k_)} of {n_} [{ph(f'co_{tg}_lo', 100 * lo_, 0)}–{ph(f'co_{tg}_hi', 100 * hi_, 0)}%] | "
               f"{ph(f'co_{tg}_dn', int(d_.below.sum()))} |")
MD["coach"] = "\n".join(out)

# (R9-3) how the matched no-call base rate is computed: no-call hit rate in each role x class cell, weighted by the
#        number of plus calls in that cell
nc_ = scd[~plus_m]
put("r9_nc_k", int(nc_.hit.sum())); put("r9_nc_n", len(nc_)); put("r9_nc_pct", 100 * nc_.hit.mean(), 0)
mix_ = scd[plus_m].groupby(["role", "draft_year"]).size()
parts, wsum = [], 0.0
for (rl, yr), w_ in mix_.items():
    d_ = nc_[(nc_.role == rl) & (nc_.draft_year == yr)]
    parts.append(f"{w_} × {ROLE_TXT[rl]} {yr} ({int(d_.hit.sum())}/{len(d_)})")
    wsum += w_ * d_.hit.mean()
put("r9_matched_check", 100 * wsum / mix_.sum(), 0)
assert NUM["r9_matched_check"] == NUM["fr_all_base"]
MD["matched"] = ", ".join(parts)

# (R9-4) leave-one-class-out calibration of the frozen call rule on an observed-times-only forecast (no filled-in
#        times: each rusher is forecast by a model with exactly the official times he has, fit on the other classes)
def obs_fc(tr, te):
    o_ = pd.Series(np.nan, index=te.index)
    for h10 in (True, False):
        for h3 in (True, False):
            m_ = (te.ten_yd_split_z.notna() == h10) & (te.three_cone_z.notna() == h3)
            cols = MAIN + (["ten_yd_split_z"] if h10 else []) + (["three_cone_z"] if h3 else [])
            te_ = te[m_].dropna(subset=cols)
            if len(te_):
                o_.loc[te_.index] = ols_pred(tr.dropna(subset=cols + ["press_eb"]), te_, cols, "press_eb")
    return o_


ba["pb_obs"] = pd.concat([obs_fc(co[co.draft_year != c], ba[(ba.draft_year == c) & (ba.rush_snaps >= RATE_MIN)])
                          for c in classes])
so_ = ba[ba.pb_obs.notna()].copy()
so_["hit"] = so_.pr_lo > so_.pb_obs
po_ = so_.call == "plus"
for tg, m_ in [("all", po_), ("clean", po_ & (so_.draft_year == 2025)), ("rest", ~po_)]:
    k_, n_ = int(so_[m_].hit.sum()), int(m_.sum())
    lo_, hi_ = cp(k_, n_)
    put(f"ob_{tg}_k", k_); put(f"ob_{tg}_n", n_); put(f"ob_{tg}_lo", 100 * lo_, 0); put(f"ob_{tg}_hi", 100 * hi_, 0)


def oe(d_, col, seed):
    """Observed / forecast pressures, rush-weighted; 90% interval by resampling rushers."""
    y_, e_ = d_.pressures.values, (d_[col] * d_.rush_snaps).values
    r_ = np.random.default_rng(seed)
    B_ = [y_[i].sum() / e_[i].sum() for i in (r_.integers(0, len(y_), len(y_)) for _ in range(4000))]
    return y_.sum() / e_.sum(), np.percentile(B_, 5), np.percentile(B_, 95)


for tg, d_, col in [("pf", scd[plus_m], "pb"), ("nf", scd[~plus_m], "pb"), ("po", so_[po_], "pb_obs"),
                    ("no", so_[~po_], "pb_obs")]:
    v_ = oe(d_, col, 300 + len(tg))
    put(f"oe_{tg}", v_[0]); put(f"oe_{tg}_lo", v_[1]); put(f"oe_{tg}_hi", v_[2])

# (R9-5) the plus-call record with Ethan Downs (7 rushes, below the 30-rush scoring line) scored anyway
dwr = ba[ba.display_name == "Ethan Downs"]
_tr, _te = fill_times(co[co.draft_year != 2025], dwr)
pb_dw = float(ols_pred(_tr, _te, BASE_T, "press_eb")[0])
dw_ = dwr.iloc[0]
lo_dw = stats.beta.ppf(0.1, np.nan_to_num(dw_.pressures) + 0.5, dw_.rush_snaps - np.nan_to_num(dw_.pressures) + 0.5)
hit_dw = int(lo_dw > pb_dw)
put("dw_rush", int(dw_.rush_snaps)); put("dw_pb", 100 * pb_dw, 1)
k_, n_ = int(NUM["fr_all_k"]) + hit_dw, int(NUM["fr_all_n"]) + 1
lo_, hi_ = cp(k_, n_)
put("dw_all_k", k_); put("dw_all_n", n_); put("dw_all_lo", 100 * lo_, 0); put("dw_all_hi", 100 * hi_, 0)
put("dw_clean_k", int(NUM["fr_clean_k"]) + hit_dw); put("dw_clean_n", int(NUM["fr_clean_n"]) + 1)

# (R9-6) the 2025 clean test as a posterior predictive probability: fit the count model on 2023-24 only (skeptical
#        prior, Laplace draws), simulate the two scored 2025 plus calls' rookie pressure counts (new-class effect drawn
#        from its prior, Hoop G measurement error included), and score each simulated season with the same hit rule
bz_tr, bz_te = bz[bz.draft_year != 2025], bz[bz.draft_year == 2025].reset_index(drop=True)
M25tr, M25te = make_model(bz_tr, ZCT), make_model(bz_te, ZCT, ref=bz_tr)
th25 = laplace_draws(M25tr, PRIOR_B, 4000, 91)
cl_ids = list(scd[plus_m & (scd.draft_year == 2025)].nfl_id)
ix25 = np.array([int(np.flatnonzero(bz_te.nfl_id.values == i)[0]) for i in cl_ids])
pb25 = ba.set_index("nfl_id").loc[cl_ids, "pb"].values
NZ25 = M25tr["NZ"]


def ppc25(zero_slope, seed):
    r_, nh = np.random.default_rng(seed), []
    nn = M25te["n"][ix25]
    for t in th25:
        u_ = r_.standard_normal() * np.exp(t[6 + NZ25])
        eta = t[0] + u_ + M25te["Z"][ix25] @ t[1:1 + NZ25]
        sl = 0.0 if zero_slope else np.where(M25te["edge"][ix25], t[2 + NZ25], t[1 + NZ25])
        T_ = M25te["mS"][ix25] + M25te["sS"][ix25] * r_.standard_normal(len(ix25))
        mu_ = np.clip(expit(eta + sl * T_), 1e-6, 1 - 1e-6)
        kap = np.exp(t[7 + NZ25])
        y_ = r_.binomial(nn.astype(int), r_.beta(mu_ * kap, (1 - mu_) * kap))
        nh.append(int((stats.beta.ppf(0.1, y_ + 0.5, nn - y_ + 0.5) > pb25).sum()))
    nh = np.array(nh)
    return float(np.mean(nh == 0)), float(np.mean(nh))


p0_m, eh_m = ppc25(False, 93)
p0_n, eh_n = ppc25(True, 94)
put("pp25_p0", p0_m); put("pp25_eh", eh_m); put("pp25_p0_null", p0_n); put("pp25_eh_null", eh_n)

# (R9-7) drill -> NFL link, by role: same matched-speed mixed model, fit separately for EDGE and interior
for rl, tg in [(1, "e"), (0, "i")]:
    d_ = msn[msn.edge == rl]
    f_ = smf.mixedlm("mg ~ hoop_z + combine_weight_z + mv", d_, groups=d_["nfl_id"]).fit(reml=True)
    put(f"mx9_{tg}", f_.params["hoop_z"], 3)
    ci_ = f_.conf_int(alpha=0.10).loc["hoop_z"]
    put(f"mx9_{tg}_lo", ci_[0], 3); put(f"mx9_{tg}_hi", ci_[1], 3)
    put(f"mx9_{tg}_n", d_.nfl_id.nunique())

# (R9-8) one table of every test, the null it tests, the estimate and p
put("r9_lpd_obs", NULL_OBS[0])
put("r9_hit_plus", 100 * up_all[plus_m].mean(), 0)
P_ROWS = [
    ("Plus calls beat forecast more often than other rushers",
     "Hoop G shuffled within class 5,000 times, calls remade",
     f"{{{{fr_all_k}}}} of {{{{fr_all_n}}}} vs {{{{r9_nc_k}}}} of {{{{r9_nc_n}}}}; gap {{{{fr_gap}}}} pts", "{{fr_perm_p}}"),
    ("Held-out R² gain, 90 rushers, 3 folds",
     "Hoop G shuffled 300 times; screen, fit and forecast all redone",
     "+{{wr2_pts}} pts [{{wr2_pts_lo}}, {{wr2_pts_hi}}]", "{{perm_p_r2}}"),
    ("Held-out log score gain, same reruns", "same shuffle",
     "+{{r9_lpd_obs}} nats in total", "{{perm_p_lpd}}"),
    ("Best of the 14 screened drill–outcome pairs", "NFL outcomes shuffled within class 2,000 times",
     "Hoop G partial ρ {{mult_obs}}", "{{mult_p}}"),
    ("Headline effect (count model)", "none (posterior)", "{{bayes_all}} pressures/season [{{bayes_all_lo}}, "
     "{{bayes_all_hi}}]; P(>0) {{bayes_all_p0}}", "—"),
    ("2025 clean test, 0 of 2", "fitted on 2023–24 only",
     "P(0 of 2) {{pp25_p0}} with the shrunk effect, {{pp25_p0_null}} with none", "—"),
    ("Hoop G → NFL sideways g at hoop speed", "mixed model, player intercept",
     "EDGE {{mx9_e}} g [{{mx9_e_lo}}, {{mx9_e_hi}}]; interior {{mx9_i}} [{{mx9_i_lo}}, {{mx9_i_hi}}]", "—")]
out = ["| Test | Null or model | Result [90%] | p |", "|---|---|---|---|"]
for a_, b_, c_, d_ in P_ROWS:
    out.append(f"| {a_} | {b_} | {c_} | {d_} |")
MD["ptable"] = "\n".join(out)
put("cc3hoop_r2_lo9", CC["cc3hoop"]["r2_lo"], 1); put("cc3hoop_r2_hi9", CC["cc3hoop"]["r2_hi"], 1)

# (R9-9) where the tiebreak applies: three rows, one line each
MD["where"] = "\n".join([
    "| Rushers with an NFL rush | Added to pick, weight, role, NGS score | Result | Takeaway |",
    "|---|---|---|---|",
    "| No 3-cone ({{ccnohoop_n}}) | Hoop G | held out: +{{ccnohoop_r2}} R² pts [90%: {{ccnohoop_r2_lo}}, {{ccnohoop_r2_hi}}] "
    "| Leans positive; this is the use case |",
    "| Ran the 3-cone ({{cc3hoop_n}}) | 3-cone; then Hoop G on top | held out: {{cc3cone_r2}} [{{cc3cone_r2_lo}}, {{cc3cone_r2_hi}}]; "
    "{{cc3hoop_r2}} [{{cc3hoop_r2_lo9}}, {{cc3hoop_r2_hi9}}] | Too few rushers to stack terms; never add Hoop G to a 3-cone |",
    "| Ran both, every measure ({{cmn_n}}) | partial ρ with pressure | in-sample [95%]: | 3-cone {{cmn_t3_r}} [{{cmn_t3_lo}}, {{cmn_t3_hi}}]; "
    "Hoop G {{cmn_hoop_r}} [{{cmn_hoop_lo}}, {{cmn_hoop_hi}}] | The 3-cone outranks Hoop G |"])

# (R9-10) 2025 decision card: the three plus calls, two 'film first' rushers (closest to a plus without a 3-cone; the
#         lowest EDGE), and the two interior minus flags. Tie-break partner = nearest same-role 2025 pick on the other
#         side of him on Hoop G.
def partner(r_, higher):
    p_ = d25[(d25.role == r_.role) & (d25.display_name != r_.display_name) & (d25.call != "plus")]
    p_ = p_[p_.shr > r_.shr] if higher else p_[p_.shr < r_.shr]
    p_ = p_[p_.call == "neutral"] if higher else p_
    return p_.iloc[(p_.pick - r_.pick).abs().argsort()].iloc[0]


ff1 = d25[(d25.call == "neutral") & d25.three_cone.isna() & (d25.rush_snaps >= RATE_MIN)].iloc[0]
ff2 = d25[d25.role == "EDGE"].iloc[-1]
CARD9 = list(d25[d25.call == "plus"].display_name) + [ff1.display_name, ff2.display_name] + \
    list(d25[d25.call == "minus"].sort_values("shr").display_name)
FILM9 = {"plusEDGE": "Flattens at the top of the rush?", "plusinterior": "Wins laterally on games and stunts?",
         "ffEDGE": "Turns the corner, or wins only with power?", "minusinterior": "Still redirects on stunts at his weight?"}
out = ["| 2025 player (pick, role) | Hoop G, sd [80%] | 3-cone | Call and tie-break | Film cue | Rookie pressure vs "
       "forecast (rushes) |", "|---|---|---|---|---|---|"]
CARD_ROWS = []
for i, nm in enumerate(CARD9):
    r_ = d25[d25.display_name == nm].iloc[0]
    k_ = f"c9_{i}"
    hg = f"{ph(k_ + '_s', r_.shr)} [{ph(k_ + '_l', r_.shr_lo)}, {ph(k_ + '_h', r_.shr_hi)}]"
    tc_ = ph(k_ + "_tc", r_.three_cone) + " s" if pd.notna(r_.three_cone) else "not run"
    if r_.call == "plus":
        q_ = partner(r_, higher=False)
        call = (("**Plus.** 3-cone first; if still tied, " if pd.notna(r_.three_cone) else "**Plus.** In a tie, ")
                + f"take him over {q_.display_name} ({pk_txt(q_.pick)})")
        cue = FILM9["plus" + r_.role]
    elif r_.call == "minus":
        q_ = partner(r_, higher=True)
        call = (f"**Minus flag, film check.** Partly size ({int(r_.combine_weight)} lb). If film agrees, "
                f"prefer {q_.display_name} ({pk_txt(q_.pick)})")
        cue = FILM9["minusinterior"]
    elif nm == ff1.display_name:
        call = "**Film first.** One run, just short of a plus, no 3-cone: film decides; ask for a second hoop run"
        cue = FILM9["ffEDGE"] if r_.role == "EDGE" else FILM9["plusinterior"]
    else:
        call = "**Film first.** Lowest EDGE hoop in the class, just short of a minus flag; no swap on Hoop G alone"
        cue = FILM9["ffEDGE"]
    if pd.notna(r_.pb):
        res_ = (f"{ph(k_ + '_pr', 100 * r_.press_rate, 1)}% vs {ph(k_ + '_pb', 100 * r_.pb, 1)}% "
                f"({int(r_.rush_snaps)}; {result_txt(r_).replace('**', '').replace('in noise', 'within noise')})")
    else:
        res_ = f"too few to score ({int(r_.rush_snaps)})"
    out.append(f"| **{r_.display_name}** ({pk_txt(r_.pick)}, {ROLE_TXT[r_.role]}) | {hg} | {tc_} | {call} | {cue} | {res_} |")
    CARD_ROWS.append(dict(name=nm, call=r_.call, pick=r_.pick, shr=r_.shr))
MD["card9"] = "\n".join(out)
TABLES["card9"] = CARD_ROWS
_i25 = d25[(d25.role == "interior") & d25.three_cone.notna()]
assert _i25.three_cone.idxmax() == d25.index[d25.display_name == "Jordan Phillips"][0]  # slowest interior 3-cone, 2025

# ============================================================================= round 10 (revision of round 9 cand 1)
_tick("round 10: motivation, LOCO rule, twins, 3-cone out of class, split count model, EDGE rush direction")

# (R10-1) motivation: what a pressured snap costs the offense, on the sampled random NFL pass plays of cohort DL
_gm = gp[gp.nfl_id.isin(dlids) & (gp.sample_kind == "random")].drop_duplicates(K).copy()
_gm = _gm[_gm.expected_points_added.notna()]
_gm["press"] = _gm.time_to_pressure.notna() | (_gm.sack == 1)
_e1, _e0 = _gm[_gm.press].expected_points_added.values, _gm[~_gm.press].expected_points_added.values
_er = np.random.default_rng(101)
_eb = [_er.choice(_e1, len(_e1)).mean() - _er.choice(_e0, len(_e0)).mean() for _ in range(4000)]
put("r10_epa_diff", _e0.mean() - _e1.mean()); put("r10_epa_lo", -np.percentile(_eb, 95)); put("r10_epa_hi", -np.percentile(_eb, 5))
put("r10_epa_np", len(_e1)); put("r10_epa_n", len(_gm))
# 3-cone coverage, one denominator: every DL in the cohort, official Combine times
put("r10_t3_all_k", int(dl.three_cone.notna().sum())); put("r10_t3_all_n", len(dl))
put("r10_t3_all_pct", 100 * dl.three_cone.notna().mean(), 0)
_d25all = dl[dl.draft_year == 2025]
put("r10_t3_25_k", int(_d25all.three_cone.notna().sum())); put("r10_t3_25_n", len(_d25all))
put("r10_hoop_no3", int((dl.hoop_z.notna() & dl.three_cone.isna()).sum()))

# (R10-2) the 3-cone lead, out of class: leave-one-class-out partial Spearman rho. For each held-out class, pressure
#         and 3-cone are residualized on pick, weight, role and NGS score with fits from the other two classes only;
#         the held-out residual pairs are pooled. 95% interval: resample rushers within class, refit everything.
_c3 = co.dropna(subset=MAIN + ["press_eb"]).copy()


def loco_t3(d):
    rx, ry, per = [], [], {}
    for c in classes:
        tr, te = d[d.draft_year != c], d[(d.draft_year == c) & d.three_cone_z.notna()]
        if len(te) < 3:
            continue
        t3 = tr[tr.three_cone_z.notna()]
        ry_ = te.press_eb.values - ols_pred(tr, te, MAIN, "press_eb")
        rx_ = -te.three_cone_z.values + ols_pred(t3, te, MAIN, "three_cone_z")
        rx.append(rx_); ry.append(ry_); per[int(c)] = stats.spearmanr(rx_, ry_)[0]
    rx, ry = np.concatenate(rx), np.concatenate(ry)
    return float(stats.spearmanr(rx, ry)[0]), len(rx), per


_r3, _n3, _per3 = loco_t3(_c3)
_g3 = {c: x for c, x in _c3.groupby("draft_year")}
_rr3, _b3 = np.random.default_rng(103), []
for _ in range(NBOOT):
    try:
        _b3.append(loco_t3(pd.concat([x.iloc[_rr3.integers(0, len(x), len(x))] for x in _g3.values()]))[0])
    except Exception:
        pass
put("r10_t3_loco", _r3); put("r10_t3_loco_n", _n3)
put("r10_t3_loco_lo", np.nanpercentile(_b3, 2.5)); put("r10_t3_loco_hi", np.nanpercentile(_b3, 97.5))
put("r10_t3_loco_pos", int(sum(v > 0 for v in _per3.values()))); put("r10_t3_loco_k", len(_per3))
put("r10_t3_loco_p0", 100 * np.mean(np.array(_b3) > 0), 0)
TABLES["t3_loco_per_class"] = _per3

# (R10-3) one count model on all 90 rushers, one shared prior: separate Hoop G slopes for rushers who ran the 3-cone
#         and rushers who did not (replaces the 34-rusher stacked least-squares row). Laplace draws, skeptical prior.
M3 = make_model(bz, ZCT)
M3 = dict(M3, edge=bz.three_cone.isna().values)          # 'edge' slot reused as the group flag: True = no 3-cone
_e3 = laplace(M3, PRIOR_B, ndraw=4000, seed=111)
summ(_e3[:, 1], "r10_cm_no3"); summ(_e3[:, 2], "r10_cm_3")
put("r10_cm_no3_n", int(bz.three_cone.isna().sum())); put("r10_cm_3_n", int(bz.three_cone.notna().sum()))
put("r10_cm_no3_p0", np.mean(_e3[:, 1] > 0))

# (R10-4) the plus-call rule, re-chosen out of class. For each held-out class: reliability from the other two classes'
#         hoop runs; interval level (50-90%) picked to maximize the plus vs rest hit-rate gap on the other two classes;
#         then calls are made and scored on the held-out class against its out-of-class forecast. Nothing is fit on
#         the class being scored.
QS = [0.5, 0.6, 0.7, 0.8, 0.9]
_SC = ba[ba.pb.notna()].copy()
_SC["hit"] = _SC.pr_lo > _SC.pb


def _rel_ids(ids):
    h_ = HSH[HSH.nfl_id.isin(ids)]
    r_ = stats.spearmanr(h_.bend_g_h1, h_.bend_g_h2)[0]
    return 2 * r_ / (1 + r_)


def _calls(d, rel, q):
    lam = d.hoop_k * rel / (1 + (d.hoop_k - 1) * rel)
    return (lam * d.hoop_z - stats.norm.ppf(0.5 + q / 2) * np.sqrt(lam * (1 - lam))) > 0


LOCO_RULE, _lk = [], np.zeros(4, int)
for c in classes:
    rel_c = _rel_ids(set(ba[ba.draft_year != c].nfl_id))
    St = _SC[_SC.draft_year != c]
    best = None
    for q in QS:
        p_ = _calls(St, rel_c, q)
        if p_.sum() < 2:
            continue
        gap = St.hit[p_].mean() - St.hit[~p_].mean()
        if best is None or gap > best[1] + 1e-9:
            best = (q, gap)
    Sc = _SC[_SC.draft_year == c]
    p_ = _calls(Sc, rel_c, best[0])
    dwn = ba[(ba.display_name == "Ethan Downs") & (ba.draft_year == c)]
    dw_plus = bool(len(dwn) and _calls(dwn, rel_c, best[0]).iloc[0])
    LOCO_RULE.append(dict(c=int(c), q=best[0], rel=rel_c, plus=list(Sc.display_name[p_]), dw=dw_plus))
    _lk += [p_.sum(), Sc.hit[p_].sum(), (~p_).sum(), Sc.hit[~p_].sum()]
TABLES["loco_rule"] = LOCO_RULE
put("r10_lr_n", int(_lk[0])); put("r10_lr_k", int(_lk[1])); put("r10_lr_rn", int(_lk[2])); put("r10_lr_rk", int(_lk[3]))
put("r10_lr_pct", 100 * _lk[1] / _lk[0], 0); put("r10_lr_rpct", 100 * _lk[3] / _lk[2], 0)
_dwp = any(r_["dw"] for r_ in LOCO_RULE)
put("r10_lr_dw_n", int(_lk[0]) + int(_dwp)); put("r10_lr_dw_k", int(_lk[1]) + (hit_dw if _dwp else 0))

# (R10-5) the 2025 clean test against base rates: probability of 0 hits in 2 under (a) the plus calls' 2023-24 hit rate
#         and (b) the 2025 no-call hit rate (Jeffreys posteriors; beta-binomial predictive)
def p0of2(k, n):
    a_, b_ = k + 0.5, n - k + 0.5
    return b_ * (b_ + 1) / ((a_ + b_) * (a_ + b_ + 1))


_t24 = scd[plus_m & (scd.draft_year <= 2024).values]
_n25 = scd[~plus_m & (scd.draft_year == 2025).values & (scd.call == "neutral").values]
put("r10_pp_plus", p0of2(int(_t24.hit.sum()), len(_t24))); put("r10_pp_nc", p0of2(int(_n25.hit.sum()), len(_n25)))
put("r10_nc25_k", int(_n25.hit.sum())); put("r10_nc25_n", len(_n25))

# (R10-6) matched twins: each scored plus call against the no-call rusher from the same class and role nearest in
#         log pick and Combine weight (greedy in pick order, no reuse). Statistic: mean paired difference in pressure
#         rate minus forecast. Null: Hoop G shuffled within class, calls remade, twins re-matched.
_SC["res"] = 100 * (_SC.press_rate - _SC.pb)
_SC["lpk"] = np.log(_SC.pick)
_sdp, _sdw = _SC.lpk.std(), _SC.combine_weight.std()


def twins(S, plus):
    nc_ = S[~plus & (S.call != "minus")] if "call" in S else S[~plus]
    used, out_ = set(), []
    for r_ in S[plus].sort_values("pick").itertuples():
        c_ = nc_[(nc_.draft_year == r_.draft_year) & (nc_.role == r_.role) & ~nc_.nfl_id.isin(used)]
        if not len(c_):
            continue
        d_ = ((c_.lpk - r_.lpk) / _sdp) ** 2 + ((c_.combine_weight - r_.combine_weight) / _sdw) ** 2
        t_ = c_.loc[d_.idxmin()]
        used.add(t_.nfl_id)
        out_.append((r_.Index, d_.idxmin()))
    return out_


_pl = (_SC.call == "plus").values
TW = twins(_SC, _pl)
_dif = np.array([_SC.res[a] - _SC.res[b] for a, b in TW])
put("r10_tw_n", len(TW)); put("r10_tw_better", int((_dif > 0).sum())); put("r10_tw_diff", _dif.mean(), 1)
put("r10_tw_hits", int(sum(_SC.hit[a] for a, _ in TW))); put("r10_tw_thits", int(sum(_SC.hit[b] for _, b in TW)))
_tr_ = np.random.default_rng(107)
put("r10_tw_lo", np.percentile([_dif[_tr_.integers(0, len(_dif), len(_dif))].mean() for _ in range(4000)], 5), 1)
put("r10_tw_hi", np.percentile([_dif[_tr_.integers(0, len(_dif), len(_dif))].mean() for _ in range(4000)], 95), 1)
_tn, _tp = [], np.random.default_rng(109)
for _ in range(2000):
    sh_ = _SC.copy()
    for c in classes:
        ix = ba.index[ba.draft_year == c]
        perm_ = pd.DataFrame(ba.loc[_tp.permutation(ix), ["shr_lo", "shr_hi"]].values, index=ba.loc[ix, "nfl_id"].values,
                             columns=["shr_lo", "shr_hi"])
        m_ = sh_.draft_year == c
        sh_.loc[m_, ["shr_lo", "shr_hi"]] = perm_.loc[sh_.loc[m_, "nfl_id"]].values
    sh_["call"] = np.select([sh_.shr_lo > 0, sh_.shr_hi < 0], ["plus", "minus"], "neutral")
    p_ = (sh_.shr_lo > 0).values
    tw_ = twins(sh_, p_)
    if tw_:
        _tn.append(np.mean([sh_.res[a] - sh_.res[b] for a, b in tw_]))
_tw_ge = int((np.array(_tn) >= _dif.mean() - 1e-12).sum())
put("r10_tw_nperm", len(_tn))
TW_PTXT = (f"p = {{{{r10_tw_p}}}}" if _tw_ge else "p < {{r10_tw_p}}")
put("r10_tw_p", (1 + _tw_ge) / (1 + len(_tn)), 3) if _tw_ge else put("r10_tw_p", np.ceil(1000 / (1 + len(_tn))) / 1000, 3)
TABLES["twins"] = [dict(plus=_SC.display_name[a], twin=_SC.display_name[b], d=float(_SC.res[a] - _SC.res[b])) for a, b in TW]
_c25tw = [(a, b) for a, b in TW if _SC.draft_year[a] == 2025]
put("r10_tw25_better", int(sum(_SC.res[a] > _SC.res[b] for a, b in _c25tw))); put("r10_tw25_n", len(_c25tw))

# (R10-7) named hit and miss from the 2023-24 classes (largest and smallest rate-over-forecast among plus calls)
_t24s = _SC[_pl & (_SC.draft_year <= 2024).values].sort_values("res")
for tg, r_ in [("hit", _t24s.iloc[-1]), ("miss", _t24s.iloc[0])]:
    TABLES[f"named_{tg}"] = r_.display_name
    put(f"r10_{tg}_pr", 100 * r_.press_rate, 1); put(f"r10_{tg}_pb", 100 * r_.pb, 1); put(f"r10_{tg}_n", int(r_.rush_snaps))
    put(f"r10_{tg}_pick", int(r_.pick)); put(f"r10_{tg}_s", r_.shr)
NAMED = {"hit": _t24s.iloc[-1], "miss": _t24s.iloc[0]}
MD["name_hit"], MD["name_miss"] = NAMED["hit"].display_name, NAMED["miss"].display_name

# (R10-8) EDGE rush direction on sampled wide-aligned NFL rushes (first 3 yards upfield): outside / inside / straight
_wt = arcp[arcp.wtype != "unclassified"].merge(dl[["nfl_id", "display_name", "role"]], on="nfl_id")
_wp = _wt.groupby("nfl_id").agg(n=("wtype", "size"), out=("wtype", lambda v: (v == "outside").mean()),
                                ins=("wtype", lambda v: (v == "inside").mean()),
                                thr=("wtype", lambda v: (v == "through").mean())).reset_index()
_wp = _wp.merge(ba[["nfl_id", "display_name", "role", "shr", "three_cone", "press_rate", "pb", "rush_snaps", "draft_year",
                    "pick"]], on="nfl_id")
_wp = _wp[(_wp.n >= 8) & (_wp.role == "EDGE") & _wp.pb.notna()].sort_values("shr", ascending=False)
put("r10_dir_n", len(_wp)); put("r10_dir_rho", stats.spearmanr(_wp.shr, _wp.out)[0])
put("r10_dir_rho_ins", stats.spearmanr(_wp.shr, _wp.ins)[0])
EDGE_NAMES = ["Javon Solomon", "Donovan Ezeiruaku", "Isaiah McGuire", "Tyler Baron"]
out = ["| EDGE (class, pick) | Hoop G, sd | 3-cone | Sampled wide rushes: outside / inside / straight | Pressure vs forecast |",
       "|---|---|---|---|---|"]
for i, nm in enumerate(EDGE_NAMES):
    r_ = _wp[_wp.display_name == nm].iloc[0]
    k_ = f"r10_ed{i}"
    tc_ = ph(k_ + "_tc", r_.three_cone) + " s" if pd.notna(r_.three_cone) else "not run"
    out.append(f"| {nm} ({int(r_.draft_year)}, {pk_txt(r_.pick)}) | {ph(k_ + '_s', r_.shr)} | {tc_} | "
               f"{ph(k_ + '_o', 100 * r_.out, 0)}% / {ph(k_ + '_i', 100 * r_.ins, 0)}% / {ph(k_ + '_t', 100 * r_.thr, 0)}% "
               f"({int(r_.n)}) | {ph(k_ + '_pr', 100 * r_.press_rate, 1)}% vs {ph(k_ + '_pb', 100 * r_.pb, 1)}% |")
MD["edges"] = "\n".join(out)

# (R10-9) 2025 decision card, scan table (player, call, 3-cone, rookie result) plus a film / tie-break list
out = ["| 2025 player (pick, role) | Call (Hoop G, sd) | 3-cone | Rookie pressure vs forecast (rushes) |",
       "|---|---|---|---|"]
FILM_LIST = []
CALLTXT = {"plus": "Plus", "minus": "Minus flag"}
for i, nm in enumerate(CARD9):
    r_ = d25[d25.display_name == nm].iloc[0]
    k_ = f"r10_c{i}"
    tc_ = ph(k_ + "_tc", r_.three_cone) + " s" if pd.notna(r_.three_cone) else "not run"
    cl_ = CALLTXT.get(r_.call, "Film first")
    if pd.notna(r_.pb):
        res_ = (f"{ph(k_ + '_pr', 100 * r_.press_rate, 1)}% vs {ph(k_ + '_pb', 100 * r_.pb, 1)}% ({int(r_.rush_snaps)}; "
                f"{result_txt(r_).replace('**', '').replace('in noise', 'within noise')})")
    else:
        res_ = f"too few to score ({int(r_.rush_snaps)})"
    out.append(f"| **{nm}** ({pk_txt(r_.pick)}, {ROLE_TXT[r_.role]}) | {cl_} ({ph(k_ + '_s', r_.shr)}) | {tc_} | {res_} |")
MD["card10"] = "\n".join(out)
_p25 = d25[d25.call == "plus"]
_pe = [r_ for r_ in _p25.itertuples() if r_.role == "EDGE"]
_pi = [r_ for r_ in _p25.itertuples() if r_.role == "interior"]
_mi = d25[d25.call == "minus"].sort_values("shr")
_sw = lambda r_, hi: partner(r_, higher=hi)
FILM_LIST.append("- **Plus, EDGE (" + " and ".join(r_.display_name.split()[-1] for r_ in _pe) + "):** does he flatten at the "
                 "top of the rush from a wide-9 or ghost alignment, or only redirect late? If his 3-cone still leaves a tie, "
                 "take him over " + " / ".join(f"{_sw(r_, False).display_name} ({pk_txt(_sw(r_, False).pick)})" for r_ in _pe) + ".")
FILM_LIST.append("- **Plus, interior (" + " and ".join(r_.display_name.split()[-1] for r_ in _pi) + "):** does he win as the "
                 "looper on T-E and E-T stunts, turning tight behind the crash? In a tie, take him over "
                 + " / ".join(f"{_sw(r_, False).display_name} ({pk_txt(_sw(r_, False).pick)})" for r_ in _pi) + ".")
FILM_LIST.append(f"- **Film first ({ff1.display_name.split()[-1]}, {ff2.display_name.split()[-1]}):** does he turn the corner, "
                 "or win only with power? A second hoop run would settle the call.")
FILM_LIST.append("- **Minus flag, interior (" + " and ".join(f"{r_.display_name.split()[-1]}, {int(r_.combine_weight)} lb"
                                                             for r_ in _mi.itertuples()) +
                 "):** partly size. Does he still redirect on games at that weight? If film agrees, prefer "
                 + " / ".join(f"{_sw(r_, True).display_name} ({pk_txt(_sw(r_, True).pick)})" for r_ in _mi.itertuples()) + ".")
MD["film"] = "\n".join(FILM_LIST)

# (R10-10) one compact table: every estimate, its n, and what it is tested against
put("r10_wr2_no3", CC["ccnohoop"]["r2"], 1)
TEST_ROWS = [
    ("3-cone → pressure, after pick, weight, role, NGS ({{cmp3_n}} rushers with 100+ rushes)", "in-sample ρ {{cmp3_r}} [{{cmp3_lo}}, {{cmp3_hi}}]; "
     "fit on two classes, scored on the third: {{r10_t3_loco}} [{{r10_t3_loco_lo}}, {{r10_t3_loco_hi}}], positive in "
     "{{r10_t3_loco_pos}} of {{r10_t3_loco_k}} held-out classes", "95% bootstrap"),
    ("Hoop G → pressures per 450-rush season, per sd (all {{bayes_n}})", "{{bayes_all}} [{{bayes_all_lo}}, {{bayes_all_hi}}]; "
     "no 3-cone ({{r10_cm_no3_n}}) {{r10_cm_no3}} [{{r10_cm_no3_lo}}, {{r10_cm_no3_hi}}], ran it ({{r10_cm_3_n}}) {{r10_cm_3}}", "one count model, shared prior, 90%"),
    ("Hoop G, held-out gain where the 3-cone is missing ({{ccnohoop_n}})", "explains {{r10_wr2_no3}} more points of rusher-to-rusher "
     "spread [{{ccnohoop_r2_lo}}, {{ccnohoop_r2_hi}}]", "fit on two classes, 90%"),
    ("Plus calls beat forecast (frozen rule)", "{{fr_all_k}} of {{fr_all_n}} vs {{r9_nc_k}} of {{r9_nc_n}} without a call",
     "Hoop G shuffled: p = {{fr_perm_p}}"),
    ("Same rule, re-chosen on two classes and scored on the third", "{{r10_lr_k}} of {{r10_lr_n}} vs {{r10_lr_rk}} of {{r10_lr_rn}} "
     "({{r10_lr_dw_k}} of {{r10_lr_dw_n}} with Downs)", "nothing fit on the scored class"),
    ("Plus call vs matched no-call twin", "better in {{r10_tw_better}} of {{r10_tw_n}} pairs; +{{r10_tw_diff}} pts of pressure rate "
     "over forecast [{{r10_tw_lo}}, {{r10_tw_hi}}]", "Hoop G shuffled, twins re-matched: " + TW_PTXT),
    ("Best of 14 drill–outcome pairs we screened", "Hoop G partial ρ {{mult_obs}}", "outcomes shuffled: p = {{mult_p}}"),
    ("2025 clean test", "0 of {{fr_clean_n}} plus calls beat forecast; 0 of {{r10_nc25_n}} without a call did", "P(0 of 2) = "
     "{{r10_pp_plus}} at the 2023–24 plus rate, {{r10_pp_nc}} at the 2025 no-call rate"),
]
out = ["| Claim (rushers) | Estimate | Tested against |", "|---|---|---|"]
for a_, b_, c_ in TEST_ROWS:
    out.append(f"| {a_} | {b_} | {c_} |")
MD["tests10"] = "\n".join(out)
for r_ in out:
    assert r_.count("|") == 4, r_          # every row: exactly three cells


# ============================================================================= FIGURES (round 10)
_tick("figures")
plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
                     "legend.fontsize": 14})
cmap = plt.get_cmap("cividis"); norm = plt.Normalize(0, 0.9)
GREY = "#555555"
ROLEC = {"EDGE": OI["blue"], "interior": OI["orange"]}

# ---- Fig 1 (hero): hoop path and his hardest-turning sampled NFL rush, side by side on one yard scale, both coloured
#      by sideways g; NFL pressure rate vs the out-of-class forecast on one shared axis
F1_NAMES = ["Javon Solomon", "Donovan Ezeiruaku", "Jordan Phillips"]
F1_TAG = {"Javon Solomon": "plus call (2024), beat his forecast",
          "Donovan Ezeiruaku": "plus call (2025 clean class), not yet",
          "Jordan Phillips": "minus flag (2025), below forecast"}
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
    la, fw = np.asarray(la, float), np.asarray(fw, float)
    F1[nm] = dict(r=r_, hx=xs_, hy=ys_, hal=tr["_alat"], hk=int(np.nanargmax(roll_h)), hpk=float(np.nanmax(roll_h)),
                  nx=la - (la.max() + la.min()) / 2, ny=fw - (fw.max() + fw.min()) / 2, nal=np.asarray(al, float),
                  nk=int(np.clip(kmx, 0, len(la) - 1)), npk=float(kk.arc_g), nmed=float(q_.arc_g.median()),
                  n0=(la[0] - (la.max() + la.min()) / 2, fw[0] - (fw.max() + fw.min()) / 2))
SPAN = max(max(np.ptp(v["hx"]), np.ptp(v["hy"]), np.ptp(v["nx"]), np.ptp(v["ny"])) for v in F1.values()) + 1.4
FH1 = 12.6
IN = lambda h: h / FH1
fig = plt.figure(figsize=(FIGW, FH1))
fig.text(0.012, 1 - IN(0.12), "Hoop G in the drill, the hardest turn of his sampled NFL rushes, and his pressure rate",
         fontsize=18, weight="bold", va="top")
fig.text(0.012, 1 - IN(0.52), "Both paths on the same yard scale, coloured by sideways g (circle: the half-second that sets "
         "the peak).\nNFL path: snap at the triangle, upfield is up, outside is right. Right: pressure rate with its 80% "
         "interval\nagainst a no-tracking forecast fit on the other two draft classes.", fontsize=14, va="top", color="#333333")
AH_IN, ROW_IN, TOP_IN = 2.05, 3.3, 1.75
AH = IN(AH_IN); AW = AH * FH1 / FIGW
XA, XB = 0.03, 0.03 + AW + 0.035
XP, WP = 0.665, 0.31
ROWY = []
for idx, nm in enumerate(F1_NAMES):
    v = F1[nm]; r_ = v["r"]
    yt = 1 - IN(TOP_IN + idx * ROW_IN + (0.32 if idx else 0))
    ybot = yt - IN(0.66 + (0.32 if idx == 0 else 0)) - AH
    ROWY.append(ybot + AH / 2)
    fig.text(0.012, yt, f"{nm} ({'EDGE' if r_.role == 'EDGE' else 'interior'}): {F1_TAG[nm]}", fontsize=16,
             weight="bold", va="top")
    fig.text(0.012, yt - IN(0.32), f"pick {pk_txt(r_.pick)} · {int(r_.combine_weight)} lb · Hoop G "
             f"{'+' if r_.shr > 0 else '−'}{abs(r_.shr):.1f} sd · 3-cone "
             f"{'%.2f s' % r_.three_cone if pd.notna(r_.three_cone) else 'not run'} · {int(r_.rush_snaps)} NFL rushes",
             fontsize=14, va="top", color="#333333")
    for x0, px, py, pg, pk, lab in [(XA, v["hx"], v["hy"], v["hal"], v["hk"], f"hoop peak {v['hpk']:.2f} g"),
                                    (XB, v["nx"], v["ny"], v["nal"], v["nk"],
                                     f"NFL peak {v['npk']:.2f} g (typical {v['nmed']:.2f})")]:
        ax = fig.add_axes([x0, ybot, AW, AH])
        pts = np.c_[px, py].reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=norm, linewidth=3.8)
        lc.set_array(np.clip(pg[:-1], 0, 0.9)); ax.add_collection(lc)
        ax.scatter([px[pk]], [py[pk]], s=190, facecolor="none", edgecolor="k", lw=2.4, zorder=3)
        ax.set_xlim(-SPAN / 2, SPAN / 2); ax.set_ylim(-SPAN / 2, SPAN / 2); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(True); sp_.set_color("#BBBBBB")
        ax.text(0.5, -0.03, lab, transform=ax.transAxes, ha="center", va="top", fontsize=14, weight="bold",
                color="#222222")
        if x0 == XB:
            ax.scatter([v["n0"][0]], [v["n0"][1]], marker="^", s=150, color="k", zorder=4)
        if idx == 0:
            ax.set_title("Run-the-Hoop drill" if x0 == XA else "hardest sampled NFL rush", fontsize=14.5, pad=6)
            ax.plot([-SPAN / 2 + 0.4, -SPAN / 2 + 2.4], [-SPAN / 2 + 0.55] * 2, color="k", lw=4)
            ax.text(-SPAN / 2 + 1.4, -SPAN / 2 + 0.8, "2 yd", ha="center", va="bottom", fontsize=15, weight="bold")
# one shared pressure axis
y_lo, y_hi = ROWY[-1] - AH / 2, ROWY[0] + AH / 2
bx = fig.add_axes([XP, y_lo, WP, y_hi - y_lo])
XMAXP = float(np.ceil(100 * max(F1[n_]["r"].pr_hi for n_ in F1_NAMES) + 2))
for yv, nm in zip(ROWY, F1_NAMES):
    r_ = F1[nm]["r"]
    col_ = OI["blue"] if r_.pr_lo > r_.pb else (OI["red"] if r_.pr_hi < r_.pb else GREY)
    bx.plot([100 * r_.pr_lo, 100 * r_.pr_hi], [yv, yv], color="#7A7A7A", lw=9, solid_capstyle="butt", zorder=1)
    bx.scatter([100 * r_.press_rate], [yv], s=220, color=col_, zorder=3, edgecolor="white", lw=1.5)
    bx.plot([100 * r_.pb] * 2, [yv - 0.045, yv + 0.045], color="k", lw=3.5, zorder=4)
    bx.text(0.0, yv + 0.06, f"{100 * r_.press_rate:.1f}% vs forecast {100 * r_.pb:.1f}%", fontsize=15, weight="bold",
            color=col_, va="bottom", transform=blended_transform_factory(bx.transAxes, bx.transData))
bx.set_ylim(y_lo, y_hi); bx.set_yticks([]); bx.spines["left"].set_visible(False)
bx.set_xlim(0, XMAXP); bx.set_xlabel("NFL pressure rate, %"); bx.grid(axis="x", color="#E6E6E6", lw=0.8)
for nm in F1_NAMES:
    r_ = F1[nm]["r"]; v = F1[nm]
    key = nm.split(" ")[-1].lower()
    put(f"f9_{key}_hoop", v["hpk"]); put(f"f9_{key}_pr", 100 * r_.press_rate, 1); put(f"f9_{key}_pb", 100 * r_.pb, 1)
    put(f"f9_{key}_n", int(r_.rush_snaps)); put(f"f9_{key}_pick", int(r_.pick)); put(f"f9_{key}_wt", int(r_.combine_weight))
    put(f"f9_{key}_nfl", v["npk"]); put(f"f9_{key}_nmed", v["nmed"])
cax = fig.add_axes([0.03, IN(0.32), 0.2, IN(0.14)])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[0, 0.3, 0.6, 0.9])
cb.ax.tick_params(labelsize=13.5); cax.set_title("path colour: sideways g", fontsize=14, loc="left", pad=4)
fig.legend(handles=[Line2D([], [], color="#7A7A7A", lw=9, label="80% interval"),
                    Line2D([], [], marker="|", color="k", lw=0, markersize=18, markeredgewidth=3.5, label="forecast")],
           loc="lower left", bbox_to_anchor=(XP - 0.01, -0.005), ncol=2, fontsize=14, frameon=False)
save(fig, "fig1_hero.png")

# ---- Fig 2: every scored plus call beside its matched no-call twin (same class and role, nearest pick and weight).
#      Left: Hoop G against his class. Right: pressure rate minus forecast. Bottom: pooled groups on the same axis.
PAIRS = sorted(TW, key=lambda t: (_SC.draft_year[t[0]], _SC.pick[t[0]]))
POOL = []
for lab, d_, mk, fill in [(f"All plus calls ({int(_pl.sum())})", _SC[_pl], "D", True),
                          (f"Their twins ({len(TW)})", _SC.loc[[b for _, b in TW]], "D", False),
                          (f"All rushers without a call ({int((_SC.call == 'neutral').sum())})", _SC[_SC.call == "neutral"], "D", True),
                          (f"Minus flags, all interior ({int((_SC.call == 'minus').sum())})", _SC[_SC.call == "minus"], "D", True)]:
    rr_ = np.random.default_rng(57)
    st_ = lambda ii: 100 * (d_.pressures.values[ii].sum() / d_.rush_snaps.values[ii].sum()
                            - np.average(d_.pb.values[ii], weights=d_.rush_snaps.values[ii]))
    B_ = [st_(rr_.integers(0, len(d_), len(d_))) for _ in range(4000)]
    POOL.append(dict(lab=lab, v=st_(np.arange(len(d_))), lo=np.percentile(B_, 10), hi=np.percentile(B_, 90), fill=fill,
                     col="#333333" if "Minus" not in lab else OI["red"]))
put("r10_pool_plus", POOL[0]["v"], 1); put("r10_pool_nc", POOL[2]["v"], 1); put("r10_pool_minus", abs(POOL[3]["v"]), 1)
ROWH = 0.58
ys, y = [], 0.0
heads = [(y, "Plus call (filled) and his twin (hollow)")]; y += 0.95
for _ in PAIRS:
    ys.append(y); y += 1.0
y += 0.5; heads.append((y, "Pooled, rush-weighted (diamond; bar: 80% interval)")); y += 0.95
ysp = []
for _ in POOL:
    ysp.append(y); y += 1.0
NR = y
HH = ROWH * NR + 2.6
fig = plt.figure(figsize=(FIGW, HH))
bot = 1.0 / HH; hgt = ROWH * NR / HH
axh = fig.add_axes([0.34, bot, 0.17, hgt]); axr = fig.add_axes([0.56, bot, 0.25, hgt])
cls_sd = {c: ba[ba.draft_year == c].shr.values for c in classes}
for yv, (a, b) in zip(ys, PAIRS):
    ra, rb = _SC.loc[a], _SC.loc[b]
    c_ = ROLEC[ra.role]
    jit = np.random.default_rng(int(ra.nfl_id)).uniform(-0.18, 0.18, len(cls_sd[ra.draft_year]))
    axh.scatter(cls_sd[ra.draft_year], yv + jit, s=10, color="#C8C8C8", zorder=1, lw=0)
    axh.scatter([ra.shr], [yv], s=170, color=c_, zorder=3, edgecolor="white", lw=1)
    axh.scatter([rb.shr], [yv], s=150, facecolor="white", edgecolor=c_, lw=2.4, zorder=3)
    axr.plot([100 * (ra.pr_lo - ra.pb), 100 * (ra.pr_hi - ra.pb)], [yv - 0.14] * 2, color="#7A7A7A", lw=6,
             solid_capstyle="butt", zorder=1)
    axr.plot([100 * (rb.pr_lo - rb.pb), 100 * (rb.pr_hi - rb.pb)], [yv + 0.16] * 2, color="#CFCFCF", lw=6,
             solid_capstyle="butt", zorder=1)
    axr.scatter([ra.res], [yv - 0.14], s=170, color=c_, zorder=3, edgecolor="white", lw=1)
    axr.scatter([rb.res], [yv + 0.16], s=150, facecolor="white", edgecolor=c_, lw=2.4, zorder=3)
    fig.text(0.012, 0, f"{ra.display_name} ({ROLE_TXT[ra.role]}, {int(ra.draft_year)}, {pk_txt(ra.pick)})",
             transform=blended_transform_factory(fig.transFigure, axr.transData), fontsize=14, va="center",
             weight="bold" if ra.hit else "normal").set_y(yv - 0.12)
    fig.text(0.012, 0, f"    twin: {rb.display_name} ({pk_txt(rb.pick)})", fontsize=12.5, color=GREY, va="center",
             transform=blended_transform_factory(fig.transFigure, axr.transData)).set_y(yv + 0.3)
    d_ = ra.res - rb.res
    axr.text(1.03, yv, f"{'+' if d_ >= 0 else '−'}{abs(d_):.1f}", transform=axr.get_yaxis_transform(), fontsize=14.5,
             va="center", color=OI["blue"] if d_ > 0 else OI["red"], weight="bold")
for yv, p_ in zip(ysp, POOL):
    axr.plot([p_["lo"], p_["hi"]], [yv, yv], color="#9A9A9A" if p_["fill"] else "#CFCFCF", lw=14,
             solid_capstyle="butt", zorder=1)
    axr.scatter([p_["v"]], [yv], marker="D", s=170, zorder=3, facecolor=p_["col"] if p_["fill"] else "white",
                edgecolor=p_["col"], lw=2.4)
    fig.text(0.012, 0, p_["lab"], transform=blended_transform_factory(fig.transFigure, axr.transData), fontsize=14,
             va="center").set_y(yv)
    axr.text(1.03, yv, f"{'+' if p_['v'] >= 0 else '−'}{abs(p_['v']):.1f}", transform=axr.get_yaxis_transform(),
             fontsize=14.5, va="center", color=p_["col"])
for y0, t_ in heads:
    fig.text(0.012, 0, t_, transform=blended_transform_factory(fig.transFigure, axr.transData), fontsize=15,
             weight="bold", va="center").set_y(y0)
for a_ in (axh, axr):
    a_.set_ylim(NR - 0.3, -0.6); a_.set_yticks([]); a_.spines["left"].set_visible(False)
axr.axvline(0, color="k", lw=1.2); axh.axvline(0, color="#999999", lw=1, ls=":")
axh.set_xlim(-1.6, 1.6); axh.set_xticks([-1, 0, 1]); axh.set_xlabel("Hoop G, sd\n(grey: his class)")
axr.set_xlim(-8, 10); axr.set_xticks([-5, 0, 5, 10])
axr.set_xlabel("pressure rate minus forecast,\npercentage points")
axr.text(1.03, -0.45, "plus − twin", transform=axr.get_yaxis_transform(), fontsize=13.5, weight="bold", va="center")
fig.text(0.012, 1 - 0.1 / HH, f"Plus calls out-rushed their matched no-call twins in {NUM['r10_tw_better']} of "
         f"{NUM['r10_tw_n']} pairs", fontsize=17.5, weight="bold", va="top")
fig.text(0.012, 1 - 0.56 / HH, "Twin = no-call rusher from the same class and role, nearest in pick and Combine weight. "
         "Blue: EDGE;\norange: interior. Bold name: beat his forecast. Forecast: pick, weight, role, NGS score and official "
         "times,\nfit on the other two classes.", fontsize=13.5, color="#333333", va="top")
fig.legend(handles=[Line2D([], [], marker="o", color=OI["blue"], lw=0, markersize=12, label="EDGE plus call"),
                    Line2D([], [], marker="o", color=OI["orange"], lw=0, markersize=12, label="interior plus call"),
                    Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=GREY, markeredgewidth=2, lw=0,
                           markersize=12, label="twin")],
           loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=1, fontsize=13.5, frameon=False)
save(fig, "fig2_twins.png")

# ============================================================================= write
_tick("write")
TABLES["md"] = MD
used = render(OUT, NUM, MD)
keep = {k_: NUM[k_] for k_ in sorted(used)}     # exactly the numbers the text uses; claims.json maps each to its sentence
json.dump(keep, open(OUT / "numbers.json", "w"), indent=1, sort_keys=True)
json.dump(TABLES, open(OUT / "tables.json", "w"), indent=1, default=str)
if os.environ.get("BDB_ALLNUM"):
    json.dump(NUM, open(os.environ["BDB_ALLNUM"], "w"), indent=1, sort_keys=True)
_tick("done")
