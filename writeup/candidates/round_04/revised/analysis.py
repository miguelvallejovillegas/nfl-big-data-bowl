"""Hoop G (Run-the-Hoop bend) vs NFL pass-rush pressure for DL/EDGE: one metric, tested honestly.

Run from the repo root:
    uv run python writeup/candidates/round_04/revised/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (only the values quoted in writeup.md, one key per claim),
tables.json, board_2023_2025.csv, board_2025.csv and figures/*.png next to this file, then renders writeup.md from
writeup_template.md. Runtime about 10-15 minutes on a laptop core (bootstraps and MCMC).

Design (round 4, revised after judging):
  * Baseline without tracking = log pick + weight + NGS athleticism score + NFL EDGE share (observed for everyone).
  * Primary out-of-class estimate: a fully nested leave-one-class-out loop. Inside each fold, on the two training
    classes only: screen all 14 (drill metric, outcome) pairs, pick the best, estimate that drill's reliability and
    shrink each player's score, refit the empirical-Bayes shrinkage of the 14 effects and apply it to the chosen
    coefficient. Per-class and pooled R2 change, class-stratified player bootstrap of the whole loop.
  * Headline effect: Bayesian beta-binomial model of pressures out of pass rushes for every hoop runner with an NFL
    rush (exposure-weighted, no 100-rush cut), class random intercept, separate EDGE / interior bend slopes, games per
    season as a covariate, and a measurement-error model for bend (one-run reliability 0.41, integrated by
    Gauss-Hermite quadrature). Bend-slope prior scaled to the typical effect among the 14 screened pairs.
    Adaptive-Metropolis, 4 chains, R-hat reported.
  * Placebos: run-stop proxy (tackles for loss per non-rush snap), tackle rate, get-off; negative-control drill:
    back-pedal mean speed.
  * Win type on sampled NFL snaps: EDGE-aligned rushes classified from the rusher's own path (outside arc, inside
    cross, through the blocker), a path proxy for dip/rip vs inside counter vs power. No ball, blockers or film.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.collections import LineCollection
from scipy import stats
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")
import re
import sys


def render(out_dir: Path, num: dict | None = None, md_tables: dict | None = None):
    """Fill writeup_template.md: {{TABLE:name}} -> generated markdown table, {{key}} -> number (fails on unknown keys).
    Returns the set of number keys the text uses (numbers.json keeps exactly these)."""
    tpl = out_dir / "writeup_template.md"
    if not tpl.exists():
        return set()
    num = num if num is not None else json.load(open(out_dir / "numbers.json"))
    md_tables = md_tables if md_tables is not None else json.load(open(out_dir / "tables.json")).get("md", {})
    text = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda m: md_tables[m.group(1)], tpl.read_text())
    used = set()

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

RNG = np.random.default_rng(2027)
import os
NBOOT = int(os.environ.get("BDB_NBOOT", 2000))
NPERM = int(os.environ.get("BDB_NPERM", 2000))
MIN_SNAPS = 100
UDFA_PICK = 262
DT = 0.1
YD_PER_G = 9.80665 / 0.9144
ON, PRE = 1.0, 0.5            # onset threshold and back-off threshold (yd/s)
W1 = 10                       # first second = 10 frames after onset / snap
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
FIGW = 1600 / 150

NUM: dict = {}
TABLES: dict = {}


def put(key, v, nd=2):
    """Store a number exactly as written in the text (and its absolute value, since the checker drops the sign)."""
    if isinstance(v, (int, np.integer)):
        NUM[key] = int(v)
        return int(v)
    s = f"{float(v):.{nd}f}"
    if s in ("-0.00", "-0.000", "-0.0"):
        s = s[1:]
    NUM[key] = s
    if s.startswith("-"):
        NUM[key + "_abs"] = s[1:]
    return s


def save(fig, name):
    fig.set_size_inches(FIGW, fig.get_size_inches()[1])
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def yz(s: pd.Series, year: pd.Series) -> pd.Series:
    return s.groupby(year).transform(lambda v: (v - v.mean()) / v.std())


# ----------------------------------------------------------------------------- drill kinematics
def onset(s: np.ndarray, start: int = 0):
    s = np.asarray(s, float)
    hit = np.flatnonzero(s[start:] >= ON)
    if not len(hit):
        return None
    i = j = start + int(hit[0])
    while j > start and s[j - 1] >= PRE and i - j < 3:
        j -= 1
    return j


def attempt_features(g: pd.DataFrame) -> dict | None:
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
    """One-way random-effects single-rep reliability (players with >= 2 reps)."""
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
    b = [stats.spearmanr(x[i], y[i])[0] for i in (RNG.integers(0, n, n) for _ in range(NBOOT))]
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return float(r), float(lo), float(hi), n, float(p)


def _pr(rx, ry, A):
    bx = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    by = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(bx, by)[0, 1])


def partial_sp(df, x, y, Z, boot=True, strata=None):
    """Spearman partial correlation: Pearson r of baseline-residualized ranks. Bootstrap over players."""
    d = df[[x, y] + Z + (["draft_year"] if strata else [])].dropna()
    R = d[[x, y] + Z].rank()
    A = np.c_[np.ones(len(R)), R[Z].values]
    r = _pr(R[x].values, R[y].values, A)
    n = len(d)
    t = r * np.sqrt((n - 2 - len(Z)) / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), n - 2 - len(Z))
    if not boot:
        return r, np.nan, np.nan, n, float(p)
    b = []
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        b.append(_pr(R[x].values[i], R[y].values[i], A[i]))
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return r, float(lo), float(hi), n, float(p)


def mde(n, k=0, power=0.8, alpha=0.05):
    """Smallest |rho| detectable with given power (Fisher z), k partialled covariates."""
    return float(np.tanh((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(n - 3 - k)))


def power_at(r, n, k=0, alpha=0.05, one_sided=True):
    se = 1 / np.sqrt(n - 3 - k)
    zc = stats.norm.ppf(1 - alpha) if one_sided else stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(zc - np.arctanh(r) / se))


def tost_p(r, n, bound, k=0):
    se = 1 / np.sqrt(n - 3 - k)
    z, zb = np.arctanh(r), np.arctanh(bound)
    return float(max(stats.norm.sf((z + zb) / se), stats.norm.cdf((z - zb) / se)))


# ----------------------------------------------------------------------------- load
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
put("n_dl", len(dl))
for c in ["combine_weight", "ten_yd_split", "forty", "three_cone", "short_shuttle"]:
    dl[c + "_z"] = yz(dl[c], dl.draft_year)

a = align.merge(dl[["nfl_id"]], on="nfl_id")
a["edge_pl"] = np.where(a.lined_up_position.isin(["EDGE", "OLB"]), a.plays, 0)
es = a.groupby("nfl_id").agg(edge_pl=("edge_pl", "sum"), pl=("plays", "sum"))
dl = dl.merge((es.edge_pl / es.pl).rename("edge_share").reset_index(), on="nfl_id", how="left")
dl["edge_share"] = dl.edge_share.fillna(dl.nfl_position.isin(["DE", "OLB"]).astype(float))
dl["edge"] = (dl.edge_share >= 0.5).astype(float)
dl["role"] = np.where(dl.edge == 1, "EDGE", "interior")

# ----------------------------------------------------------------------------- Combine drills
ev = events[(events.position_group == "DL") & events.drill_name.isin(["PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL"])]
ev = ev[["event_id", "nfl_id", "draft_year", "drill_name", "attempt_no", "start_speed", "peak_accel",
         "mean_speed", "duration_s"]]
import pickle
CACHE = os.environ.get("BDB_CACHE")   # optional dev cache for the frame-level feature extraction (never committed)


def cached(name, fn):
    if CACHE:
        f = Path(CACHE) / f"{name}.pkl"
        if f.exists():
            return pickle.load(open(f, "rb"))
        out = fn()
        pickle.dump(out, open(f, "wb"))
        return out
    return fn()


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
put("n_hoop_att", len(hoop_att))
put("hoop_g_median", hoop_att.bend_g.median())
put("hoop_g_p10", hoop_att.bend_g.quantile(0.1))
put("hoop_g_p90", hoop_att.bend_g.quantile(0.9))
put("hoop_speed_at_bend", hoop_att.speed_at_bend.median(), 1)
put("hoop_radius_typ", (hoop_att.speed_at_bend ** 2 / (hoop_att.bend_g * YD_PER_G)).median(), 1)

# ---- reliability, with bootstrap intervals
r1_d1, n_d1pairs = icc(rush_att.nfl_id, rush_att.d1_z)
put("rel1_drill_d1", r1_d1); put("n_drill_d1_pairs", n_d1pairs)
k_rush = rush_att.dropna(subset=["d1"]).groupby("nfl_id").size().mean()
rel_d1_player = sb(r1_d1, k_rush)
put("rel_drill_d1_player", rel_d1_player); put("k_rush_mean", k_rush, 1)
h1, h2 = hoop_att.bend_g_h1.to_numpy(), hoop_att.bend_g_h2.to_numpy()
r_sh = stats.spearmanr(h1, h2)[0]
rel_hoop = 2 * r_sh / (1 + r_sh)
put("rel_hoop_splithalf", r_sh); put("rel_hoop_sb", rel_hoop)
bs_sh = []
for _ in range(NBOOT):
    i = RNG.integers(0, len(h1), len(h1))
    rr = stats.spearmanr(h1[i], h2[i])[0]
    bs_sh.append(2 * rr / (1 + rr))
put("rel_hoop_sb_lo", np.percentile(bs_sh, 2.5)); put("rel_hoop_sb_hi", np.percentile(bs_sh, 97.5))
rep_icc, n_rep = icc(hoop_att.nfl_id, hoop_att.hoop_z)
put("rel_hoop_icc_rep", rep_icc); put("n_hoop_two", n_rep)
rep = hoop_att[hoop_att.groupby("nfl_id").bend_g.transform("size") >= 2]
rid = rep.nfl_id.unique()
bs_icc = []
for _ in range(NBOOT):
    pick_ = RNG.choice(rid, len(rid))
    d_ = pd.concat([rep[rep.nfl_id == p].assign(nfl_id=j) for j, p in enumerate(pick_)])
    try:
        bs_icc.append(icc(d_.nfl_id, d_.hoop_z)[0])
    except Exception:
        pass
put("rel_hoop_icc_lo", np.nanpercentile(bs_icc, 2.5)); put("rel_hoop_icc_hi", np.nanpercentile(bs_icc, 97.5))
put("snaps_for_07_hoop", int(np.ceil(0.7 * (1 - rel_hoop) / (rel_hoop * 0.3))))
put("snaps_for_07_drill", int(np.ceil(0.7 * (1 - r1_d1) / (r1_d1 * 0.3))))

# ---- player-level drill metrics: the seven screened metrics (higher = better after sign)
hoop = hoop_att.groupby("nfl_id").agg(hoop_g=("bend_g", "mean"), hoop_k=("bend_g", "size"),
                                      hoop_speed=("mean_speed", "mean"), hoop_time=("duration_s", "mean"),
                                      hoop_eid=("event_id", "first"), draft_year=("draft_year", "first")).reset_index()
rush = rush_att.groupby("nfl_id").agg(d1=("d1", "mean"), rush_bend=("bend_g", "mean"), rush_acc=("peak_accel", "mean"),
                                      rush_t3=("t3", "mean"), rush_k=("d1", "count"),
                                      draft_year=("draft_year", "first")).reset_index()
for c in ["hoop_g", "hoop_speed", "hoop_time"]:
    hoop[c + "_z"] = yz(hoop[c], hoop.draft_year)
for c in ["d1", "rush_bend", "rush_acc", "rush_t3"]:
    rush[c + "_z"] = yz(rush[c], rush.draft_year)
hoop["hoop_time_z"] *= -1   # faster lap = better
rush["rush_t3_z"] *= -1     # quicker to 3 yd = better
hoop = hoop.rename(columns={"hoop_g_z": "hoop_z"})
dl = dl.merge(hoop.drop(columns="draft_year"), on="nfl_id", how="left")
dl = dl.merge(rush.drop(columns="draft_year"), on="nfl_id", how="left")
put("n_hoop_players", int(dl.hoop_z.notna().sum()))
put("n_rush_players", int(dl.d1_z.notna().sum()))
SCREEN = [("hoop_z", "Hoop G (bend)"), ("hoop_speed_z", "Hoop mean speed"), ("hoop_time_z", "Hoop lap time"),
          ("rush_bend_z", "Rush-drill turn g"), ("rush_acc_z", "Rush-drill peak accel"),
          ("rush_t3_z", "Rush-drill time to 3 yd"), ("d1_z", "Rush-drill first second")]
put("k_screen", len(SCREEN))

# year mean / sd of Hoop G per class (for the shrinkage arithmetic in g)
hstat = hoop.groupby("draft_year").hoop_g.agg(["mean", "std"])

# ----------------------------------------------------------------------------- NFL outcomes
s = seasons[seasons.position_group == "DL"].copy()
s["w_go"] = s.pass_rush_snaps * s.avg_get_off
agg = s.groupby("nfl_id").agg(rush_snaps=("pass_rush_snaps", "sum"), pressures=("pressures", "sum"),
                              w_go=("w_go", "sum"), n_seasons=("season", "count"), games=("games", "sum")).reset_index()
agg["get_off"] = agg.w_go / agg.rush_snaps
agg["press_rate"] = agg.pressures / agg.rush_snaps
dl = dl.merge(agg, on="nfl_id", how="left")
dl["rush_snaps"] = dl.rush_snaps.fillna(0)
dl["games"] = dl.games.fillna(0)
dl["seasons_avail"] = 2026 - dl.draft_year
dl["snaps_per_season"] = dl.rush_snaps / dl.seasons_avail
dl["games_per_season"] = dl.games / dl.seasons_avail

m = dl.rush_snaps >= 20
n_, k_ = dl.loc[m, "rush_snaps"], dl.loc[m, "pressures"]
p_ = k_ / n_
mu = k_.sum() / n_.sum()
N_ = n_.sum()
tau2 = ((n_ * (p_ - mu) ** 2).sum() - (len(n_) - 1) * mu * (1 - mu)) / (N_ - (n_ ** 2).sum() / N_)
M = mu * (1 - mu) / tau2 - 1
put("prior_weight", int(round(M))); put("prior_mean", 100 * mu, 1)
mm = dl.rush_snaps > 0
dl.loc[mm, "press_eb"] = (dl.loc[mm, "pressures"] + M * mu) / (dl.loc[mm, "rush_snaps"] + M)

s2 = s[s.pass_rush_snaps >= MIN_SNAPS][["nfl_id", "season", "pressures", "pass_rush_snaps", "avg_get_off"]].copy()
s2["pr"] = s2.pressures / s2.pass_rush_snaps
pr2 = s2.merge(s2.assign(season=s2.season - 1), on=["nfl_id", "season"], suffixes=("", "_n"))
put("yoy_n", len(pr2))
rel_press_season = stats.spearmanr(pr2.pr, pr2.pr_n)[0]
rel_getoff_season = stats.spearmanr(pr2.avg_get_off, pr2.avg_get_off_n)[0]
put("yoy_press", rel_press_season); put("yoy_getoff", rel_getoff_season)

# ----------------------------------------------------------------------------- game tracking: first second after snap
dlids = set(dl.nfl_id)
g = gp[gp.nfl_id.isin(dlids)]
gf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "x", "y"])
gf = gf[gf.nfl_id.isin(dlids)]
K = ["game_id", "play_id", "nfl_id"]
gf = gf.merge(g[K + ["snap_frame"]], on=K)
gf["rel"] = gf.frame - gf.snap_frame
p0 = gf[gf.rel == 0].set_index(K)[["x", "y"]]
p1 = gf[gf.rel == W1].set_index(K)[["x", "y"]]
gd = np.hypot(p1.x - p0.x.reindex(p1.index), p1.y - p0.y.reindex(p1.index)).rename("g_d1").reset_index()
gd = gd.merge(g[K + ["player_get_off", "sample_kind", "lined_up_position", "play_direction"]], on=K).dropna(
    subset=["g_d1", "player_get_off"])
put("n_game_plays", len(gd))
put("rho_play_d1_getoff", stats.spearmanr(gd.g_d1, gd.player_get_off)[0])
for kind in ["random", "top_epa"]:
    gpl = gd[gd.sample_kind == kind].groupby("nfl_id").g_d1.agg(["mean", "size"]).rename(
        columns={"mean": f"g_d1_{kind}", "size": f"g_n_{kind}"}).reset_index()
    dl = dl.merge(gpl, on="nfl_id", how="left")
put("random_per_player", gd[gd.sample_kind == "random"].groupby("nfl_id").size().median(), 0)

# ---- NFL Arc G: hardest sideways g held 0.5 s from snap to throw/sack (cap 3.5 s), per sampled rush
gfe = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "event"])
gfe = gfe[gfe.nfl_id.isin(dlids) & gfe.event.isin(["pass_forward", "qb_sack", "qb_strip_sack", "run", "handoff"])]
ends = gfe.groupby(K).frame.min().rename("end_frame")
ARC_CAP, VMIN = 35, 2.0
FIELD_MID, OFF_MIN = 53.3 / 2, 4.0   # outside = away from field centre; only trusted 4+ yd off centre (ball on a hash)
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
    al[v < VMIN] = 0.0                     # only turning while actually moving counts
    inw = (h.frame.values >= sf) & (h.frame.values <= e)
    roll = pd.Series(np.where(inw, al, np.nan)).rolling(5, center=True, min_periods=5).mean()
    if roll.notna().sum() == 0:
        continue
    info = pinfo.loc[key]
    # rush path in a defender frame: forward = toward the offense; lateral sign: + = outside (away from field centre)
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
    kmax = int(np.nanargmax(roll.values)) - f0          # centre of the 0.5 s window that sets Arc G
    arc_tr[key] = (la_, fw_, al[ii] / YD_PER_G, kmax, wtype)
  return arc_rows, arc_tr


arc_rows, arc_tr = cached("arc", arc_features)
arcp = pd.DataFrame(arc_rows)
arcp["gz"] = arcp.groupby("nfl_id").arc_g.transform(lambda v: (v - v.mean()) / v.std())
put("n_arc_plays", len(arcp)); put("arc_g_median", arcp.arc_g.median())
put("arc_press_rate_play", 100 * arcp.press.mean(), 1)
ARC = {}
for kind in ["random", "top_epa"]:
    q = arcp[arcp.kind == kind].copy()
    q["half"] = q.groupby("nfl_id").cumcount() % 2
    hh = q.groupby(["nfl_id", "half"]).arc_g.mean().unstack().dropna()
    r_ = stats.spearmanr(hh[0], hh[1])[0]
    ARC[kind] = 2 * r_ / (1 + r_)
    put(f"rel_arc_{kind}", ARC[kind]); put(f"n_rel_arc_{kind}", len(hh))
    dl = dl.merge(q.groupby("nfl_id").arc_g.mean().rename(f"arc_{kind}").reset_index(), on="nfl_id", how="left")
put("arc_snaps_per_player", arcp[arcp.kind == "random"].groupby("nfl_id").size().median(), 0)

# ----------------------------------------------------------------------------- cohort
co = dl[dl.rush_snaps >= MIN_SNAPS].copy()
co["quick"] = -co.get_off
put("n_cohort", len(co))
put("n_cohort_edge", int((co.role == "EDGE").sum()))
put("n_cohort_int", int((co.role == "interior").sum()))
put("cohort_median_snaps", int(co.rush_snaps.median()))
put("n_missing_3cone_cohort", int(co.three_cone_z.isna().sum()))
put("n_missing_10yd_cohort", int(co.ten_yd_split_z.isna().sum()))
put("n_missing_ngs_cohort", int(co.ngs_athleticism_score.isna().sum()))
coh = co[co.hoop_z.notna()]
put("n_missing_10yd_hoop", int(coh.ten_yd_split_z.isna().sum()))
put("n_missing_40_hoop", int(coh.forty_z.isna().sum()))
put("n_full_10yd_hoop", int(coh.ten_yd_split_z.notna().sum()))
put("n_missing_10yd_hoop_edge", int((coh.ten_yd_split_z.isna() & (coh.role == "EDGE")).sum()))
put("n_missing_10yd_hoop_2025", int((coh.ten_yd_split_z.isna() & (coh.draft_year == 2025)).sum()))

# Main baseline: every covariate observed for every cohort player (no imputation anywhere).
MAIN = ["log_pick", "combine_weight_z", "edge_share", "ngs_athleticism_score"]
NOTIME = ["log_pick", "combine_weight_z", "edge_share"]
put("k_base", len(MAIN))

# ---- who drops out of the cohort (injury, depth, rookies without snaps) and is that related to Hoop G?
hr = dl[dl.hoop_z.notna()].copy()
put("n_hoop_runners", len(hr)); put("n_hoop_excluded", int((hr.rush_snaps < MIN_SNAPS).sum()))
r, lo, hi, n, p = sp_ci(hr.hoop_z, hr.snaps_per_season)
put("rho_hoop_snaps_ps", r); put("rho_hoop_snaps_ps_lo", lo); put("rho_hoop_snaps_ps_hi", hi); put("n_hoop_snaps_ps", n)
r, lo, hi, n, p = sp_ci(hr.hoop_z, hr.games_per_season)
put("rho_hoop_games_ps", r); put("rho_hoop_games_ps_lo", lo); put("rho_hoop_games_ps_hi", hi)
inc = hr.rush_snaps >= MIN_SNAPS
put("hoop_z_in_cohort", hr.loc[inc, "hoop_z"].mean()); put("hoop_z_out_cohort", hr.loc[~inc, "hoop_z"].mean())


def impute(tr: pd.DataFrame, te: pd.DataFrame, cols=("ten_yd_split_z", "three_cone_z")):
    """Single deterministic fill of missing official times from weight + EDGE share (training rows only)."""
    tr, te = tr.copy(), te.copy()
    for c in cols:
        mk = tr[c].notna()
        A = np.c_[np.ones(mk.sum()), tr.loc[mk, ["combine_weight_z", "edge_share"]].values]
        b = np.linalg.lstsq(A, tr.loc[mk, c].values, rcond=None)[0]
        for d in (tr, te):
            m2 = d[c].isna()
            d.loc[m2, c] = np.c_[np.ones(m2.sum()), d.loc[m2, ["combine_weight_z", "edge_share"]].values] @ b
    return tr, te


MI_PRED = ["combine_weight_z", "edge_share", "ngs_athleticism_score", "hoop_z"]


def impute_mi(tr: pd.DataFrame, te: pd.DataFrame, rng, cols=("ten_yd_split_z",), pred=None):
    """One proper stochastic draw (parameter + residual noise) for missing official times. Imputation model uses
    weight, EDGE share, NGS athleticism score and Hoop G, fit on training rows; never the outcome."""
    for col in cols:
        tr, te = _impute_mi1(tr, te, rng, col, pred or MI_PRED)
    return tr, te


def _impute_mi1(tr, te, rng, col, MI_PRED=MI_PRED):
    tr, te = tr.copy(), te.copy()
    mk = tr[col].notna()
    X = np.c_[np.ones(mk.sum()), tr.loc[mk, MI_PRED].values]
    y = tr.loc[mk, col].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    dof = max(len(y) - X.shape[1], 1)
    s2 = res @ res / dof
    s2d = s2 * dof / rng.chisquare(dof)
    cov = s2d * np.linalg.inv(X.T @ X)
    bd = rng.multivariate_normal(b, cov)
    for d in (tr, te):
        m2 = d[col].isna()
        Xm = np.c_[np.ones(m2.sum()), d.loc[m2, MI_PRED].values]
        d.loc[m2, col] = Xm @ bd + rng.normal(0, np.sqrt(s2d), m2.sum())
    return tr, te


M_IMP = 20


def partial_draws(df, x, y, Z, nb=NBOOT):
    d = df[[x, y] + Z].dropna()
    R = d[[x, y] + Z].rank()
    A = np.c_[np.ones(len(R)), R[Z].values]
    r = _pr(R[x].values, R[y].values, A)
    b = []
    for _ in range(nb):
        i = RNG.integers(0, len(d), len(d))
        b.append(_pr(R[x].values[i], R[y].values[i], A[i]))
    return r, np.array(b), len(d)


# ---- raw signals (for the signal map): pressure and get-off on the cohort, NFL Arc G on every DL with sampled snaps
MEAS = [("forty_z", "40-yard time", "official", -1), ("ten_yd_split_z", "10-yd split", "official", -1),
        ("three_cone_z", "3-cone", "official", -1), ("short_shuttle_z", "Short shuttle", "official", -1),
        ("combine_weight_z", "Lighter weight", "official", -1), ("ngs_athleticism_score", "NGS athleticism", "official", 1),
        ("log_pick", "Earlier draft pick", "official", -1), ("d1_z", "Rush-drill first second", "tracking", 1),
        ("hoop_z", "Hoop G (bend)", "tracking", 1)]
OUTC = [("press_eb", "NFL pressure rate"), ("quick", "NFL get-off (quicker)"), ("arc_random", "NFL Arc G (game bend)")]
sig = []
for col, lab, kind, sgn in MEAS:
    for oc, olab in OUTC:
        src = dl if oc == "arc_random" else co
        r, lo, hi, n, p = sp_ci(sgn * src[col], src[oc])
        sig.append(dict(col=col, label=lab, kind=kind, outcome=oc, rho=r, lo=lo, hi=hi, n=n, mde=mde(n)))
sig = pd.DataFrame(sig)
for r_ in sig.itertuples():
    put(f"rho_{r_.col}_{r_.outcome}", r_.rho); put(f"n_{r_.col}_{r_.outcome}", int(r_.n))
TABLES["signals"] = sig.round(3).to_dict("records")
put("rho_hoop_weight", stats.spearmanr(dl.hoop_z, dl.combine_weight, nan_policy="omit")[0])
put("rho_hoop_pick", stats.spearmanr(dl.hoop_z, -dl.log_pick, nan_policy="omit")[0])
put("rho_hoop_10yd", stats.spearmanr(dl.hoop_z, -dl.ten_yd_split_z, nan_policy="omit")[0])
put("rho_hoop_3cone", stats.spearmanr(dl.hoop_z, -dl.three_cone_z, nan_policy="omit")[0])
put("rho_hoop_ngs", stats.spearmanr(dl.hoop_z, dl.ngs_athleticism_score, nan_policy="omit")[0])

# ---- primary tests after the MAIN baseline (no imputation)
REL_OUT = {"quick": sb(rel_getoff_season, co.n_seasons.mean()), "press_eb": sb(rel_press_season, co.n_seasons.mean())}
put("rel_press_career", REL_OUT["press_eb"]); put("rel_getoff_career", REL_OUT["quick"])
REL_X = {"d1_z": rel_d1_player, "hoop_z": rel_hoop}
PRIMARY = [("hoop_z", "press_eb"), ("hoop_z", "quick"), ("d1_z", "quick"), ("d1_z", "press_eb")]
prim = {}
for x, y in PRIMARY:
    r, lo, hi, n, p = partial_sp(co, x, y, MAIN)
    m_obs = mde(n, len(MAIN))
    ceil = np.sqrt(REL_X[x] * REL_OUT[y])
    prim[(x, y)] = dict(r=r, lo=lo, hi=hi, n=n, p=p, mde=m_obs, mde_true=min(m_obs / ceil, 1.0))
    key = f"{x}_{y}"
    put(f"pr_{key}", r); put(f"pr_lo_{key}", lo); put(f"pr_hi_{key}", hi); put(f"pr_n_{key}", n)
    put(f"pr_p_{key}", p, 3); put(f"pmde_{key}", m_obs); put(f"mdetrue_{key}", prim[(x, y)]["mde_true"])
H = prim[("hoop_z", "press_eb")]

# ---- disattenuated Hoop G effect, two reliability sources, joint bootstrap
r_obs, r_b, _ = partial_draws(co, "hoop_z", "press_eb", MAIN)
bs_sh = np.array(bs_sh); bs_icc_a = np.array(bs_icc)
for tag, rel_pt, rel_b in [("sh", rel_hoop, bs_sh), ("icc", rep_icc, bs_icc_a)]:
    rel_draw = np.clip(RNG.choice(rel_b, len(r_b)), 0.05, 1.0)
    dis = np.clip(r_b / np.sqrt(rel_draw * REL_OUT["press_eb"]), -1, 1)
    put(f"dis_{tag}", min(r_obs / np.sqrt(max(rel_pt, 0.05) * REL_OUT["press_eb"]), 1.0))
    put(f"dis_{tag}_lo", np.percentile(dis, 2.5)); put(f"dis_{tag}_hi", np.percentile(dis, 97.5))
    put(f"dis_{tag}_capped", float(np.mean(dis >= 0.999)))

# role split (rush type from charted alignment share) for Hoop G
for role in ["EDGE", "interior"]:
    c = co[co.role == role]
    r, lo, hi, n, p = partial_sp(c, "hoop_z", "press_eb", [b for b in MAIN if b != "edge_share"])
    put(f"pr_hoop_{role}", r); put(f"pr_n_hoop_{role}", n); put(f"pr_lo_hoop_{role}", lo); put(f"pr_hi_hoop_{role}", hi)

# ----------------------------------------------------------------------------- does drill bend show up as game bend?
ARCB = ["combine_weight_z", "edge_share"]
r, lo, hi, n, p = sp_ci(dl.hoop_z, dl.arc_random)
put("rho_hoop_arc_raw", r); put("rho_hoop_arc_raw_lo", lo); put("rho_hoop_arc_raw_hi", hi); put("n_hoop_arc", n)
put("rho_arc_weight", stats.spearmanr(dl.arc_random, dl.combine_weight, nan_policy="omit")[0])
put("rho_arc_edge", stats.spearmanr(dl.arc_random, dl.edge_share, nan_policy="omit")[0])
dA = dl.dropna(subset=["hoop_z", "arc_random"] + ARCB)
r, lo, hi, n, p = partial_sp(dA, "hoop_z", "arc_random", ARCB)
put("pr_hoop_arc", r); put("pr_hoop_arc_lo", lo); put("pr_hoop_arc_hi", hi); put("pr_hoop_arc_n", n)
put("mdetrue_hoop_arc", min(mde(n, len(ARCB)) / np.sqrt(rel_hoop * ARC["random"]), 1.0))
put("tost_hoop_arc", tost_p(r, n, 0.30, len(ARCB)), 3)
ARC_AFTER = {"hoop_z": (r, lo, hi)}
r, lo, hi, n, p = partial_sp(dl.dropna(subset=["d1_z", "arc_random"] + ARCB), "d1_z", "arc_random", ARCB)
ARC_AFTER["d1_z"] = (r, lo, hi)
dT = dl.dropna(subset=["hoop_z", "arc_top_epa"] + ARCB)
r, lo, hi, n, p = partial_sp(dT, "hoop_z", "arc_top_epa", ARCB)
put("pr_hoop_arc_top", r); put("pr_hoop_arc_top_n", n)
r, lo, hi, n, p = partial_sp(co, "arc_random", "press_eb", MAIN)
put("pr_arc_press", r); put("pr_arc_press_lo", lo); put("pr_arc_press_hi", hi); put("pr_arc_press_n", n)
r, lo, hi, n, p = partial_sp(co, "hoop_z", "press_eb", MAIN + ["arc_random"])
put("pr_hoop_press_given_arc", r); put("pr_hoop_press_given_arc_lo", lo); put("pr_hoop_press_given_arc_hi", hi)
# within player: are his higher-arc snaps his pressure snaps? (random half, play length controlled)
qa = arcp[(arcp.kind == "random")].dropna(subset=["gz"]).copy()
lfit = smf.logit("press ~ gz + dur", qa).fit(disp=0)
put("within_or", np.exp(lfit.params["gz"])); put("within_n_plays", len(qa)); put("within_n_players", qa.nfl_id.nunique())
ids_ = qa.nfl_id.unique(); grp = {i: d for i, d in qa.groupby("nfl_id")}
bor = []
for _ in range(max(200, NBOOT // 4)):
    d_ = pd.concat([grp[i] for i in RNG.choice(ids_, len(ids_))])
    try:
        bor.append(np.exp(smf.logit("press ~ gz + dur", d_).fit(disp=0).params["gz"]))
    except Exception:
        pass
put("within_or_lo", np.percentile(bor, 2.5)); put("within_or_hi", np.percentile(bor, 97.5))

# ---- rush type and down: on the random sampled snaps, does Hoop G's link sit on EDGE-aligned snaps or
#      on true passing downs (2nd and 8+, 3rd/4th and 5+)? Player's no-tracking expected rate is an offset covariate.
Ab = np.c_[np.ones(len(co)), co[MAIN].values]
co["base_fit"] = np.clip(Ab @ np.linalg.lstsq(Ab, co.press_eb.values, rcond=None)[0], 0.01, 0.5)
sn = qa.merge(co[["nfl_id", "hoop_z", "base_fit", "role"]].dropna(subset=["hoop_z"]), on="nfl_id")
sn["pd"] = (((sn.down == 2) & (sn.ytg >= 8)) | ((sn.down >= 3) & (sn.ytg >= 5))).astype(int)
sn["wide"] = sn["align"].isin(["EDGE", "OLB"]).astype(int)
sn["lb"] = np.log(sn.base_fit / (1 - sn.base_fit))
put("sn_n", len(sn)); put("sn_players", sn.nfl_id.nunique()); put("sn_press", int(sn.press.sum()))
put("sn_pd_share", 100 * sn.pd.mean(), 0)
put("sn_press_pd", 100 * sn[sn.pd == 1].press.mean(), 1); put("sn_press_npd", 100 * sn[sn.pd == 0].press.mean(), 1)
put("sn_press_share_pd", 100 * sn[sn.press == 1].pd.mean(), 0)
# separate cleaner fits for the two splits (each with one interaction) for the reported odds ratios
def snap_or2(d):
    out = {}
    for term in ["pd", "wide"]:
        f = smf.logit(f"press ~ lb + dur + hoop_z * {term}", d).fit(disp=0).params
        out[term + "_1"] = np.exp(f["hoop_z"] + f[f"hoop_z:{term}"]); out[term + "_0"] = np.exp(f["hoop_z"])
    return out


so = snap_or2(sn)
ids2 = sn.nfl_id.unique(); g2 = {i: d for i, d in sn.groupby("nfl_id")}
bso = []
for _ in range(max(200, NBOOT // 4)):
    d_ = pd.concat([g2[i] for i in RNG.choice(ids2, len(ids2))])
    try:
        bso.append(snap_or2(d_))
    except Exception:
        pass
bso = pd.DataFrame(bso)
for k_, v_ in so.items():
    put(f"sor_{k_}", v_); put(f"sor_{k_}_lo", np.percentile(bso[k_], 2.5)); put(f"sor_{k_}_hi", np.percentile(bso[k_], 97.5))
put("sn_wide_share", 100 * sn.wide.mean(), 0)

# ----------------------------------------------------------------------------- selection-honest screen: 7 metrics x 2 outcomes = 14
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


def perm_within_class(df, col):
    y = df[col].copy()
    for c, ix in df.groupby("draft_year").groups.items():
        v = y.loc[ix].values
        y.loc[ix] = v[RNG.permutation(len(v))]
    return y


def perm_pair(df):
    """Shuffle each player's (pressure, get-off) pair jointly within class."""
    yp, yg = df.press_eb.copy(), df.quick.copy()
    for c, ix in df.groupby("draft_year").groups.items():
        ix = np.asarray(ix)
        pi = ix[RNG.permutation(len(ix))]
        yp.loc[ix] = df.press_eb.loc[pi].values; yg.loc[ix] = df.quick.loc[pi].values
    return yp, yg


S_p = screen_setup(co, "press_eb", MAIN)
S_g = screen_setup(co, "quick", MAIN)
obs_p = screen_stats(S_p, co.press_eb)
obs_g = screen_stats(S_g, co.quick)
scr = pd.DataFrame([{"col": c, "label": l, "pr_press": obs_p[c], "pr_getoff": obs_g[c],
                     "n_press": len(S_p[c]["idx"]), "n_getoff": len(S_g[c]["idx"])} for c, l in SCREEN])
all14 = np.r_[scr.pr_press.abs().values, scr.pr_getoff.abs().values]
put("k_screen14", len(all14))
put("screen_best_is_hoop", int(np.argmax(all14) == list(scr.col).index("hoop_z")))
max_obs_p, max_obs_all = scr.pr_press.abs().max(), all14.max()
null_p, null_all, null_hoop = [], [], []
NPERM_SCREEN = 5 * NPERM
for _ in range(NPERM_SCREEN):
    yp, yg = perm_pair(co)
    sp_ = screen_stats(S_p, yp); sg_ = screen_stats(S_g, yg)
    mp = max(abs(v) for v in sp_.values())
    null_p.append(mp); null_hoop.append(abs(sp_["hoop_z"]))
    null_all.append(max(mp, max(abs(v) for v in sg_.values())))
null_p, null_all = np.array(null_p), np.array(null_all)
put("perm_single_hoop", (1 + np.sum(np.array(null_hoop) >= abs(obs_p["hoop_z"]))) / (1 + NPERM_SCREEN), 3)
put("perm_best7", (1 + np.sum(null_p >= max_obs_p)) / (1 + NPERM_SCREEN), 3)
put("perm_best14", (1 + np.sum(null_all >= max_obs_all)) / (1 + NPERM_SCREEN), 3)
put("n_perm_screen", NPERM_SCREEN)
put("perm_best14_pct", 100 * float(NUM["perm_best14"]), 1); put("NBOOT", NBOOT)
put("null_best14_p95", np.percentile(null_all, 95))
put("hoop_obs_abs", abs(obs_p["hoop_z"]))
for r_ in scr.itertuples():
    put(f"scr_{r_.col}", r_.pr_press); put(f"scrg_{r_.col}", r_.pr_getoff)
TABLES["screen"] = scr.round(3).to_dict("records")

# empirical-Bayes shrinkage of all 14 screened effects (Fisher z) -> selection-corrected effect
z = np.arctanh(np.r_[scr.pr_press.values, scr.pr_getoff.values])
se2 = 1 / (np.r_[scr.n_press.values, scr.n_getoff.values] - 3 - len(MAIN))
mz = np.average(z, weights=1 / se2)
tau2_s = max(0.0, np.var(z, ddof=1) - se2.mean())
B = tau2_s / (tau2_s + se2)
z_shr = mz + B * (z - mz)
ih = list(scr.col).index("hoop_z")
put("eb_tau", np.sqrt(tau2_s)); put("eb_hoop", np.tanh(z_shr[ih])); put("eb_mean", np.tanh(mz))
eb_factor = np.tanh(z_shr[ih]) / obs_p["hoop_z"]


# ----------------------------------------------------------------------------- out-of-class machinery
def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ols_pred(tr, te, cols, y):
    A = np.c_[np.ones(len(tr)), tr[cols].values]
    b = np.linalg.lstsq(A, tr[y].values, rcond=None)[0]
    return np.c_[np.ones(len(te)), te[cols].values] @ b


def ridge_pred(tr, te, cols, y, alphas=np.logspace(-2, 3, 30)):
    X = tr[cols].values; mu_, sd_ = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu_) / sd_; yc = tr[y].values - tr[y].values.mean()
    U, sv, Vt = np.linalg.svd(Xs, full_matrices=False)
    best, bb = np.inf, None
    for al in alphas:
        dfac = sv / (sv ** 2 + al)
        beta = Vt.T @ (dfac * (U.T @ yc))
        h = (U ** 2 * (sv ** 2 / (sv ** 2 + al))).sum(1)
        e = ((yc - Xs @ beta) / (1 - h)) ** 2
        if e.mean() < best:
            best, bb = e.mean(), beta
    return tr[y].values.mean() + ((te[cols].values - mu_) / sd_) @ bb


def loco(df, base, extra, y, fit=ols_pred, imp=None, train_one=False, rng=None, imp_cols=("ten_yd_split_z",)):
    """Leave-one-class-out (or train-on-one-class). imp: None | 'single' | 'mi' (M_IMP draws, predictions averaged)."""
    classes = sorted(df.draft_year.unique())
    Y, PB, PA, CL, ID = [], [], [], [], []
    for c in classes:
        tr, te = (df[df.draft_year == c], df[df.draft_year != c]) if train_one else (df[df.draft_year != c], df[df.draft_year == c])
        if imp == "single":
            tr, te = impute(tr, te)
        if imp == "mi":
            pb, pa = np.zeros(len(te)), np.zeros(len(te))
            for _ in range(M_IMP):
                t1, t2 = impute_mi(tr, te, rng if rng is not None else RNG, imp_cols)
                pb += fit(t1, t2, base, y) / M_IMP; pa += fit(t1, t2, base + extra, y) / M_IMP
            PB.append(pb); PA.append(pa)
        else:
            PB.append(fit(tr, te, base, y)); PA.append(fit(tr, te, base + extra, y))
        Y.append(te[y].values); CL.append(np.full(len(te), c)); ID.append(te.nfl_id.values)
    if train_one:   # each player is tested twice: average R2 over the three splits
        rb = [r2(Y[i], PB[i]) for i in range(3)]; ra = [r2(Y[i], PA[i]) for i in range(3)]
        return dict(n=int(np.mean([len(v) for v in Y])), r2_base=np.mean(rb), r2_aug=np.mean(ra),
                    delta=np.mean(ra) - np.mean(rb), split=[a_ - b_ for a_, b_ in zip(ra, rb)])
    Y, PB, PA, CL, ID = map(np.concatenate, (Y, PB, PA, CL, ID))
    out = dict(n=len(Y), r2_base=r2(Y, PB), r2_aug=r2(Y, PA))
    out["delta"] = out["r2_aug"] - out["r2_base"]
    out["pred"] = pd.DataFrame({"nfl_id": ID, "y": Y, "pb": PB, "pa": PA, "cl": CL})
    return out


NLOCO = int(os.environ.get("BDB_NLOCO", 1000))


def boot_loco(df, base, extra, y, nb=NLOCO, draws=False, **kw):
    """Class-stratified player bootstrap of the whole leave-one-class-out loop (refit every time)."""
    groups = {c: d for c, d in df.groupby("draft_year")}
    out = []
    for _ in range(nb):
        d = pd.concat([gg.iloc[RNG.integers(0, len(gg), len(gg))] for gg in groups.values()])
        try:
            out.append(loco(d, base, extra, y, **kw)["delta"])
        except Exception:
            pass
    return np.array(out) if draws else np.percentile(out, [2.5, 97.5])


dH = co.dropna(subset=["hoop_z", "press_eb"]).copy()
LH = loco(dH, MAIN, ["hoop_z"], "press_eb")
put("lo_n", LH["n"]); put("r2b", LH["r2_base"]); put("r2a", LH["r2_aug"]); put("dr2", LH["delta"], 3)
DR2_DRAWS = {"main": boot_loco(dH, MAIN, ["hoop_z"], "press_eb", draws=True)}
lo_, hi_ = np.percentile(DR2_DRAWS["main"], [2.5, 97.5])
put("dr2_lo", lo_, 3); put("dr2_hi", hi_, 3)
put("dr2_pos_share", 100 * np.mean(DR2_DRAWS["main"] > 0), 0); put("dr2_med", np.median(DR2_DRAWS["main"]), 3)
fold = {}
for c, d_ in LH["pred"].groupby("cl"):
    rb, ra = stats.spearmanr(d_.pb, d_.y)[0], stats.spearmanr(d_.pa, d_.y)[0]
    fold[int(c)] = dict(n=len(d_), rb=rb, ra=ra, d=r2(d_.y.values, d_.pa.values) - r2(d_.y.values, d_.pb.values))
    put(f"frho_b_{c}", rb); put(f"frho_a_{c}", ra); put(f"fn_{c}", len(d_)); put(f"fdr2_{c}", fold[int(c)]["d"], 3)
put("folds_up", int(sum(f["ra"] > f["rb"] for f in fold.values())))
put("folds_dr2_up", int(sum(f["d"] > 0 for f in fold.values())))
# train on ONE class, predict the other two (harder out-of-class test)
L2 = loco(dH, MAIN, ["hoop_z"], "press_eb", train_one=True)
put("l2o_dr2", L2["delta"], 3); put("l2o_n", L2["n"])
put("l2o_up", int(sum(v > 0 for v in L2["split"])))
for c, v in zip(sorted(dH.draft_year.unique()), L2["split"]):
    put(f"l2o_dr2_{c}", v, 3)
put("l2o_dr2_min", min(L2["split"]), 3); put("l2o_dr2_max", max(L2["split"]), 3)

# ---- nested leave-one-class-out: re-select the best of all 14 (metric, outcome) pairs on the training classes
cand = co.dropna(subset=["press_eb", "quick"]).copy()
cand = cand[cand[[c for c, _ in SCREEN]].notna().any(axis=1)]
classes = sorted(cand.draft_year.unique())
FOLD = {c: (cand[cand.draft_year != c], cand[cand.draft_year == c],
            screen_setup(cand[cand.draft_year != c], "press_eb", MAIN),
            screen_setup(cand[cand.draft_year != c], "quick", MAIN)) for c in classes}


def nested(yp, yg):
    Y, PB, PA, sel = [], [], [], []
    for c in classes:
        tr, te, Sp, Sg = FOLD[c]
        ys = {"press_eb": yp, "quick": yg}
        st = {("press_eb", k): v for k, v in screen_stats(Sp, yp.loc[tr.index]).items()}
        st.update({("quick", k): v for k, v in screen_stats(Sg, yg.loc[tr.index]).items()})
        oc, best = max(st, key=lambda k: abs(st[k]))
        tr2 = tr.assign(_y=ys[oc].loc[tr.index].values).dropna(subset=[best])
        te2 = te.assign(_y=ys[oc].loc[te.index].values).dropna(subset=[best])
        mu_, sd_ = tr2._y.mean(), tr2._y.std()        # common scale across outcomes, training-fold only
        tr2 = tr2.assign(_y=(tr2._y - mu_) / sd_); te2 = te2.assign(_y=(te2._y - mu_) / sd_)
        PB.append(ols_pred(tr2, te2, MAIN, "_y")); PA.append(ols_pred(tr2, te2, MAIN + [best], "_y"))
        Y.append(te2._y.values); sel.append(f"{best}|{oc}")
    Y, PB, PA = map(np.concatenate, (Y, PB, PA))
    return r2(Y, PA) - r2(Y, PB), sel, len(Y)


nd, nsel, nn = nested(cand.press_eb, cand.quick)
put("nested_dr2", nd, 3); put("nested_n", nn)
put("nested_hoop_folds", int(sum(s_ == "hoop_z|press_eb" for s_ in nsel)))
TABLES["nested_selected"] = dict(zip([int(c) for c in classes], nsel))
null_n = []
for _ in range(NPERM):
    yp, yg = perm_pair(cand)
    null_n.append(nested(yp, yg)[0])
put("nested_perm_p", (1 + np.sum(np.array(null_n) >= nd)) / (1 + NPERM), 3)

# ----------------------------------------------------------------------------- baseline variants for Hoop G (same 63 rows unless noted)
ROB = []
MI_RNG = np.random.default_rng(99)


def mi_partial(df, base, cols=("ten_yd_split_z",)):
    """Rubin-style: average Fisher z of the partial rho over M_IMP imputations; interval from pooled bootstraps."""
    zs, bs = [], []
    for _ in range(M_IMP):
        d1, _ = impute_mi(df, df.iloc[:0], MI_RNG, cols)
        r, b, n = partial_draws(d1, "hoop_z", "press_eb", base, nb=max(100, NBOOT // 10))
        zs.append(np.arctanh(r)); bs.append(b)
    bs = np.concatenate(bs)
    return float(np.tanh(np.mean(zs))), *np.percentile(bs, [2.5, 97.5]), n


def add_var(tag, label, df, base, imp=None, fit=ols_pred, nb=NLOCO, imp_cols=("ten_yd_split_z",)):
    d = df.dropna(subset=["hoop_z", "press_eb"]).copy()
    if imp is None:
        d = d.dropna(subset=base)
    if imp == "mi":
        r, lo, hi, n = mi_partial(d, base, imp_cols)
    elif imp == "single":
        r, lo, hi, n, _ = partial_sp(impute(d, d)[0], "hoop_z", "press_eb", base)
    else:
        r, lo, hi, n, _ = partial_sp(d, "hoop_z", "press_eb", base)
    L = loco(d, base, ["hoop_z"], "press_eb", fit=fit, imp=imp, rng=MI_RNG, imp_cols=imp_cols)
    dr = boot_loco(d, base, ["hoop_z"], "press_eb", nb=nb, fit=fit, imp=imp, rng=MI_RNG, draws=True,
                   imp_cols=imp_cols) if nb else None
    ci = np.percentile(dr, [2.5, 97.5]) if nb else (np.nan, np.nan)
    if nb:
        DR2_DRAWS[tag] = dr
        put(f"v_dr2pos_{tag}", 100 * np.mean(dr > 0), 0)
    row = dict(tag=tag, label=label, n=n, r=r, lo=lo, hi=hi, r2b=L["r2_base"], r2a=L["r2_aug"], dr2=L["delta"],
               dr2_lo=ci[0], dr2_hi=ci[1])
    put(f"v_r_{tag}", r); put(f"v_lo_{tag}", lo); put(f"v_hi_{tag}", hi); put(f"v_n_{tag}", n)
    put(f"v_r2b_{tag}", L["r2_base"], 3); put(f"v_r2a_{tag}", L["r2_aug"], 3); put(f"v_dr2_{tag}", L["delta"], 3)
    if nb:
        put(f"v_dr2lo_{tag}", ci[0], 3); put(f"v_dr2hi_{tag}", ci[1], 3)
    ROB.append(row)
    return row


cc = co[co.ten_yd_split_z.notna()]
add_var("main", "Main: pick, weight, EDGE share, NGS athleticism", co, MAIN)
# one source of truth for the headline interval: alias the primary-test keys to the main-variant bootstrap
for k_ in ["", "lo_", "hi_"]:
    put(f"pr_{k_}hoop_z_press_eb", float(NUM[f"v_{(k_ or 'r_')}main"]))
add_var("notime", "No athletic score at all: pick, weight, EDGE share", co, NOTIME)
add_var("angle", "Pick, 10-yd, 3-cone, NGS (+ weight, EDGE share), multiple imputation", co,
        MAIN + ["ten_yd_split_z", "three_cone_z"], imp="mi", nb=max(100, NLOCO // 3),
        imp_cols=("ten_yd_split_z", "three_cone_z"))
add_var("cc_main", "Main, only the players who ran the 40", cc, MAIN)
add_var("cc_10", "Main + official 10-yd, only the players who ran the 40", cc, MAIN + ["ten_yd_split_z"])
add_var("cc_40", "Main + official 40, only the players who ran the 40", cc, MAIN + ["forty_z"])
add_var("snaps50", "Main, 50+ rushes", dl[dl.rush_snaps >= 50], MAIN, nb=0)
add_var("snaps200", "Main, 200+ rushes", dl[dl.rush_snaps >= 200], MAIN, nb=0)
put("share_r_cc", float(NUM["v_r_cc_10"]) / float(NUM["v_r_main"]) * 100, 0)
# the players without an official 40 / 10-yd split: who are they and what do they carry?
miss = dH[dH.ten_yd_split_z.isna()].copy()
A_ = np.c_[np.ones(len(dH)), dH[MAIN].values]
dH["res_main"] = dH.press_eb - A_ @ np.linalg.lstsq(A_, dH.press_eb.values, rcond=None)[0]
r, lo, hi, n, p = sp_ci(dH.loc[miss.index, "hoop_z"], dH.loc[miss.index, "res_main"])
put("rho_miss", r); put("rho_miss_lo", lo); put("rho_miss_hi", hi); put("n_miss", n)
r, lo, hi, n, p = sp_ci(dH.loc[cc.index.intersection(dH.index), "hoop_z"], dH.loc[cc.index.intersection(dH.index), "res_main"])
put("rho_obs10", r); put("rho_obs10_lo", lo); put("rho_obs10_hi", hi); put("n_obs10", n)
put("miss_hoop_mean", miss.hoop_z.mean()); put("obs_hoop_mean", dH.loc[dH.ten_yd_split_z.notna(), "hoop_z"].mean())
TABLES["robustness"] = pd.DataFrame(ROB).round(3).to_dict("records")

# mixed model with a class random intercept (pressure in pp) and OLS slope
dm = dH.assign(pp=lambda d: 100 * d.press_eb)
FML = "pp ~ log_pick + combine_weight_z + edge_share + ngs_athleticism_score + hoop_z"
mfit = smf.mixedlm(FML, dm, groups=dm.draft_year).fit(reml=True)
ci = mfit.conf_int().loc["hoop_z"]
put("mixed_hoop_pp", mfit.params["hoop_z"]); put("mixed_hoop_lo", ci[0]); put("mixed_hoop_hi", ci[1])
ofit = smf.ols(FML, dm).fit()
oc_ = ofit.conf_int().loc["hoop_z"]
put("slope_hoop_pp", ofit.params["hoop_z"]); put("slope_hoop_pp_lo", oc_[0]); put("slope_hoop_pp_hi", oc_[1])
put("pressures_per_450", 4.5 * ofit.params["hoop_z"], 1)
put("pressures_per_450_eb", 4.5 * ofit.params["hoop_z"] * eb_factor, 1)
put("slope_hoop_pp_eb", ofit.params["hoop_z"] * eb_factor)
put("eb_factor", eb_factor)

# ---- does it matter for a GM? price the effect in draft slots (same OLS, pp of pressure per log pick)
b_lp = -ofit.params["log_pick"]                      # pp gained per unit decrease in log pick
for tag, eff in [("raw", ofit.params["hoop_z"]), ("eb", ofit.params["hoop_z"] * eb_factor)]:
    put(f"pick_equiv_{tag}", int(round(100 * np.exp(-eff / b_lp))))
put("b_logpick_pp", b_lp)

# ---- missing-40 dependence: Hoop G x has-40 interaction, class fixed effects, class-stratified player bootstrap
dH["has40"] = dH.forty_z.notna().astype(int)
dm = dH.assign(pp=lambda d: 100 * d.press_eb)
FMLI = "pp ~ log_pick + combine_weight_z + edge_share + ngs_athleticism_score + C(draft_year) + has40 + hoop_z:has40 + hoop_z"
ifit = smf.ols(FMLI, dm).fit()
b0, b1 = ifit.params["hoop_z"], ifit.params["hoop_z"] + ifit.params["hoop_z:has40"]
put("int_slope_no40", b0); put("int_slope_40", b1); put("int_diff", b1 - b0)
put("int_p", ifit.pvalues["hoop_z:has40"], 2)
gr_ = {c: d for c, d in dm.groupby("draft_year")}
bi = []
for _ in range(NBOOT):
    d_ = pd.concat([g_.iloc[RNG.integers(0, len(g_), len(g_))] for g_ in gr_.values()])
    try:
        f_ = smf.ols(FMLI, d_).fit().params
        bi.append((f_["hoop_z"], f_["hoop_z"] + f_["hoop_z:has40"]))
    except Exception:
        pass
bi = np.array(bi)
put("int_slope_no40_lo", np.percentile(bi[:, 0], 2.5)); put("int_slope_no40_hi", np.percentile(bi[:, 0], 97.5))
put("int_slope_40_lo", np.percentile(bi[:, 1], 2.5)); put("int_slope_40_hi", np.percentile(bi[:, 1], 97.5))
put("n_no40", int((dH.has40 == 0).sum())); put("n_40", int(dH.has40.sum()))
no40 = dH[dH.has40 == 0].sort_values("hoop_z", ascending=False)
put("no40_edge", int((no40.role == "EDGE").sum())); put("no40_2025", int((no40.draft_year == 2025).sum()))
put("no40_pick_med", int(no40.pick.median())); put("has40_pick_med", int(dH[dH.has40 == 1].pick.median()))
put("no40_hoop_mean", no40.hoop_z.mean()); put("has40_hoop_mean", dH[dH.has40 == 1].hoop_z.mean())
top_no40 = no40.head(4)
for i, r_ in enumerate(top_no40.itertuples()):
    put(f"no40_{i}_hoop", r_.hoop_z); put(f"no40_{i}_res", 100 * r_.res_main, 1)
TABLES["no40"] = no40[["display_name", "role", "draft_year", "pick", "combine_weight", "hoop_z", "res_main",
                       "rush_snaps", "press_rate"]].round(4).to_dict("records")
put("no40_top_half_res", 100 * no40[no40.hoop_z >= no40.hoop_z.median()].res_main.mean(), 1)
put("no40_bot_half_res", 100 * no40[no40.hoop_z < no40.hoop_z.median()].res_main.mean(), 1)

# ---- placebo / negative-control outcomes: bend should not buy get-off or tackles
sd_ = s.groupby("nfl_id").agg(tk=("tackles", "sum"), asst=("assists", "sum"), plays=("plays", "sum"),
                               sacks=("sacks", "sum"), qp=("quick_pressures", "sum"),
                               ttp_w=("avg_time_to_pressure", lambda v: np.nan)).reset_index()
s["ttp_w"] = s.avg_time_to_pressure * s.pressures
ttp = s.dropna(subset=["avg_time_to_pressure"]).groupby("nfl_id").agg(ttp_w=("ttp_w", "sum"), pz=("pressures", "sum"))
sd_ = sd_.drop(columns="ttp_w").merge((ttp.ttp_w / ttp.pz).rename("ttp").reset_index(), on="nfl_id", how="left")
co2 = dH.merge(sd_, on="nfl_id", how="left")
co2["tackle_rate"] = (co2.tk + co2.asst) / co2.plays
co2["finish"] = co2.sacks / co2.pressures.replace(0, np.nan)
co2["quick_share"] = co2.qp / co2.pressures.replace(0, np.nan)
for oc in ["tackle_rate", "quick", "finish", "quick_share", "ttp"]:
    r, lo, hi, n, p = partial_sp(co2, "hoop_z", oc, MAIN)
    put(f"plc_{oc}", r); put(f"plc_lo_{oc}", lo); put(f"plc_hi_{oc}", hi); put(f"plc_n_{oc}", n)

# ---- availability (injury / depth) behind the 100+ rush cut
inc = hr.rush_snaps >= MIN_SNAPS
put("gps_in", hr.loc[inc, "games_per_season"].median(), 1); put("gps_out", hr.loc[~inc, "games_per_season"].median(), 1)
r, lo, hi, n, p = partial_sp(dH.assign(gps=dH.games_per_season), "hoop_z", "press_eb", MAIN + ["gps"])
put("pr_hoop_given_games", r); put("pr_hoop_given_games_lo", lo); put("pr_hoop_given_games_hi", hi)

# ---- second placebo outcome (run-stop proxy) and a negative-control drill (back-pedal)
tfl_ = s.groupby("nfl_id").agg(tfl=("tfl", "sum")).reset_index()
co2 = co2.merge(tfl_, on="nfl_id", how="left")
co2["run_snaps"] = co2.plays - co2.rush_snaps
co2["runstop"] = co2.tfl / co2.run_snaps.where(co2.run_snaps >= 50)
r, lo, hi, n, p = partial_sp(co2, "hoop_z", "runstop", MAIN)
put("plc_runstop", r); put("plc_lo_runstop", lo); put("plc_hi_runstop", hi); put("plc_n_runstop", n)
bp = events[(events.position_group == "DL") & (events.drill_name == "BACK_PEDAL_AND_REACT")].copy()
bp["bp_z"] = yz(bp.mean_speed, bp.draft_year)
bpp = bp.groupby("nfl_id").bp_z.mean().reset_index()
cobp = co.merge(bpp, on="nfl_id")
r, lo, hi, n, p = partial_sp(cobp, "bp_z", "press_eb", MAIN)
put("neg_bp", r); put("neg_lo_bp", lo); put("neg_hi_bp", hi); put("neg_n_bp", n)

# ----------------------------------------------------------------------------- win type on sampled NFL snaps (path proxy)
wt = arcp[arcp.wtype != "unclassified"].merge(dl[["nfl_id", "hoop_z", "combine_weight_z", "edge_share"]], on="nfl_id")
wt = wt[wt.hoop_z.notna()].copy()
put("wt_n", len(wt)); put("wt_players", wt.nfl_id.nunique())
for w_ in ["outside", "inside", "through"]:
    put(f"wt_share_{w_}", 100 * (wt.wtype == w_).mean(), 0)
    rr_ = wt[(wt.kind == "random") & (wt.wtype == w_)]
    put(f"wt_press_{w_}", 100 * rr_.press.mean(), 1)
wpl = wt.groupby("nfl_id").agg(hoop_z=("hoop_z", "first"), combine_weight_z=("combine_weight_z", "first"),
                               edge_share=("edge_share", "first"), nw=("wtype", "size"),
                               out_share=("wtype", lambda v: (v == "outside").mean())).reset_index()
wpl = wpl[wpl.nw >= 8]
r, lo, hi, n, p = partial_sp(wpl, "hoop_z", "out_share", ["combine_weight_z", "edge_share"])
put("wt_rho_out", r); put("wt_rho_out_lo", lo); put("wt_rho_out_hi", hi); put("wt_rho_out_n", n)
wp = wt[wt.press == 1].copy()
wp["hi"] = wp.hoop_z > 0
put("wt_npress", len(wp))
put("wt_out_press_hi", 100 * (wp[wp.hi].wtype == "outside").mean(), 0)
put("wt_out_press_lo", 100 * (wp[~wp.hi].wtype == "outside").mean(), 0)
gw_ = {i: d for i, d in wp.groupby("nfl_id")}
idw = np.array(list(gw_))
bw = []
for _ in range(NBOOT):
    d_ = pd.concat([gw_[i] for i in RNG.choice(idw, len(idw))])
    bw.append(100 * ((d_[d_.hi].wtype == "outside").mean() - (d_[~d_.hi].wtype == "outside").mean()))
put("wt_out_press_diff", float(NUM["wt_out_press_hi"]) - float(NUM["wt_out_press_lo"]), 0)
put("wt_out_press_diff_lo", np.nanpercentile(bw, 2.5), 0); put("wt_out_press_diff_hi", np.nanpercentile(bw, 97.5), 0)

# ----------------------------------------------------------------------------- primary out-of-class estimate: fully nested LOCO
# Inside each fold (training classes only): screen 14 pairs -> pick best -> reliability of that drill from the training
# classes' attempts -> shrink each player's score -> EB-shrink the 14 effects and apply the factor to the chosen slope.
ATTF = {"hoop_z": (hoop_att, "bend_g", 1), "hoop_speed_z": (hoop_att, "mean_speed", 1),
        "hoop_time_z": (hoop_att, "duration_s", -1), "rush_bend_z": (rush_att, "bend_g", 1),
        "rush_acc_z": (rush_att, "peak_accel", 1), "rush_t3_z": (rush_att, "t3", -1), "d1_z": (rush_att, "d1", 1)}
ATTZ = {}
for f_, (tab, col, sg) in ATTF.items():
    a_ = tab[["nfl_id", "draft_year", col]].dropna().copy()
    a_["z"] = sg * yz(a_[col], a_.draft_year)
    ATTZ[f_] = a_
    kk_ = a_.groupby("nfl_id").size().rename(f_ + "_k")
    cand = cand.drop(columns=[f_ + "_k"], errors="ignore").merge(kk_.reset_index(), on="nfl_id", how="left")
HSH = hoop_att[["nfl_id", "bend_g_h1", "bend_g_h2"]].dropna()


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
    res = dict(Y=[], PB=[], PA=[], CL=[], sel=[], rel=[], fac=[])
    for c in classes:
        tr, te = df[df.draft_year != c].copy(), df[df.draft_year == c].copy()
        if imp_rng is not None:
            tr, te = impute_mi(tr, te, imp_rng, ("ten_yd_split_z", "three_cone_z"),
                               pred=["combine_weight_z", "edge_share", "ngs_athleticism_score"])
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
        res["Y"].append(te2._y.values); res["CL"].append(np.full(len(te2), c)); res["sel"].append(f"{best}|{oc}")
        res["rel"].append(rel_); res["fac"].append(fac)
    out = {"pooled": r2(np.concatenate(res["Y"]), np.concatenate(res["PA"])) -
           r2(np.concatenate(res["Y"]), np.concatenate(res["PB"])), "sel": res["sel"], "rel": res["rel"],
           "fac": res["fac"], "n": int(sum(len(v) for v in res["Y"]))}
    for c, y_, pa_, pb_ in zip(classes, res["Y"], res["PA"], res["PB"]):
        out[int(c)] = r2(y_, pa_) - r2(y_, pb_)
    return out


NF = nested_full(cand)
put("nf_dr2", NF["pooled"], 3); put("nf_n", NF["n"])
put("nf_hoop_folds", int(sum(s_ == "hoop_z|press_eb" for s_ in NF["sel"])))
for c in classes:
    put(f"nf_dr2_{c}", NF[int(c)], 3)
TABLES["nf_selected"] = dict(zip([int(c) for c in classes], NF["sel"]))
TABLES["nf_rel"] = dict(zip([int(c) for c in classes], [round(v, 3) for v in NF["rel"]]))
TABLES["nf_fac"] = dict(zip([int(c) for c in classes], [round(v, 3) for v in NF["fac"]]))
gcand = {c: d for c, d in cand.groupby("draft_year")}
NFB, NFSEL = [], []
for _ in range(NLOCO):
    d_ = pd.concat([g_.iloc[RNG.integers(0, len(g_), len(g_))] for g_ in gcand.values()])
    try:
        o_ = nested_full(d_)
        NFB.append([o_["pooled"]] + [o_[int(c)] for c in classes]); NFSEL.append(o_["sel"])
    except Exception:
        pass
NFB = np.array(NFB)
put("nf_dr2_lo", np.percentile(NFB[:, 0], 2.5), 3); put("nf_dr2_hi", np.percentile(NFB[:, 0], 97.5), 3)
put("nf_pos", 100 * np.mean(NFB[:, 0] > 0), 0)
for j, c in enumerate(classes):
    put(f"nf_dr2_{c}_lo", np.percentile(NFB[:, j + 1], 2.5), 3); put(f"nf_dr2_{c}_hi", np.percentile(NFB[:, j + 1], 97.5), 3)
put("nf_boot_hoop_share", 100 * np.mean([s_ == "hoop_z|press_eb" for ss in NFSEL for s_ in ss]), 0)
put("nf_nboot", len(NFB))
# same loop with the 10-yd split and 3-cone in the baseline, imputed inside each fold (training rows only)
IMP_RNG = np.random.default_rng(7)
NFI = [nested_full(cand, base=MAIN + ["ten_yd_split_z", "three_cone_z"], imp_rng=IMP_RNG)["pooled"] for _ in range(10)]
put("nf_angle_dr2", np.mean(NFI), 3)
NFIB = []
for _ in range(max(100, NLOCO // 5)):
    d_ = pd.concat([g_.iloc[RNG.integers(0, len(g_), len(g_))] for g_ in gcand.values()])
    try:
        NFIB.append(nested_full(d_, base=MAIN + ["ten_yd_split_z", "three_cone_z"], imp_rng=IMP_RNG)["pooled"])
    except Exception:
        pass
put("nf_angle_lo", np.percentile(NFIB, 2.5), 3); put("nf_angle_hi", np.percentile(NFIB, 97.5), 3)

# ----------------------------------------------------------------------------- headline: Bayesian partial-pooling model
# pressures_i ~ BetaBinomial(rushes_i, mu_i, kappa); logit mu_i = a + u_class + Z_i b + slope_role * T_i
# T_i = true bend in true-sd units; x_i | T_i from one-run reliability REL (k_i runs). T integrated by Gauss-Hermite.
from scipy.special import gammaln, expit, logit as _logit
REL = float(NUM["rel_hoop_sb"])
bz = dl[dl.hoop_z.notna() & (dl.rush_snaps > 0)].copy()
bz["pressures"] = bz.pressures.fillna(0)
bz["lgps"] = np.log1p(bz.games_per_season)
ZC = ["log_pick", "combine_weight_z", "edge_share", "ngs_athleticism_score", "lgps"]
Zb = ((bz[ZC] - bz[ZC].mean()) / bz[ZC].std()).values
edge_b = (bz.edge_share >= 0.5).values
cls_b = pd.Categorical(bz.draft_year).codes
yb, nb_ = bz.pressures.values.astype(float), bz.rush_snaps.values.astype(float)
xb, kb = bz.hoop_z.values, bz.hoop_k.values
lamb = REL / (REL + (1 - REL) / kb)
mS, sS = lamb * xb / np.sqrt(REL), np.sqrt(lamb * (1 - REL) / kb / REL)
ghx, ghw = np.polynomial.hermite_e.hermegauss(15)
lghw = np.log(ghw / ghw.sum())
Tn = mS[:, None] + sS[:, None] * ghx[None, :]
lcb = gammaln(nb_ + 1) - gammaln(yb + 1) - gammaln(nb_ - yb + 1)
put("bayes_n", len(bz)); put("bayes_n_cohort", int((bz.rush_snaps >= MIN_SNAPS).sum()))
put("bayes_n_norush", int((dl.hoop_z.notna() & (dl.rush_snaps == 0)).sum()))
pbar = float(co.press_rate.mean())
rho_typ = float(np.mean(np.abs(all14)))
pp_per_rho = float(NUM["slope_hoop_pp"]) / float(NUM["v_r_main"])
PRIOR_B = rho_typ * pp_per_rho / (100 * pbar * (1 - pbar)) / np.sqrt(REL)
put("bayes_rho_typ", rho_typ); put("bayes_prior_sd", PRIOR_B, 3)
NP_ = 13


def logpost(t, prior_b):
    a, beta, bi, be, zu, lsu, lk = t[0], t[1:6], t[6], t[7], t[8:11], t[11], t[12]
    su, kap = np.exp(lsu), np.exp(lk)
    eta = a + su * zu[cls_b] + Zb @ beta
    mu = expit(eta[:, None] + np.where(edge_b, be, bi)[:, None] * Tn)
    mu = np.clip(mu, 1e-9, 1 - 1e-9)
    al, bb = mu * kap, (1 - mu) * kap
    ll = (lcb[:, None] + gammaln(yb[:, None] + al) + gammaln((nb_ - yb)[:, None] + bb) - gammaln(nb_[:, None] + kap)
          - gammaln(al) - gammaln(bb) + gammaln(kap)) + lghw[None, :]
    m_ = ll.max(1)
    L = (m_ + np.log(np.exp(ll - m_[:, None]).sum(1))).sum()
    lp = -0.5 * (a - _logit(0.08)) ** 2 - 0.5 * (beta ** 2).sum() / 0.25 - 0.5 * (bi ** 2 + be ** 2) / prior_b ** 2
    lp += -0.5 * (zu ** 2).sum() - 0.5 * (su / 0.3) ** 2 + lsu - 0.5 * (lk - np.log(100)) ** 2
    return L + lp


NMCMC = int(os.environ.get("BDB_NMCMC", 30000))


def am_chain(seed, prior_b, niter=NMCMC):
    rng = np.random.default_rng(seed)
    t = np.r_[_logit(0.08), np.zeros(5), 0, 0, np.zeros(3), np.log(0.1), np.log(100)] + rng.normal(0, 0.05, NP_)
    lp = logpost(t, prior_b)
    C = np.eye(NP_) * 0.005
    H_ = np.empty((niter, NP_))
    for i in range(niter):
        if i > 2000 and i % 500 == 0:
            C = np.cov(H_[max(0, i - 20000):i].T) * 2.38 ** 2 / NP_ + np.eye(NP_) * 1e-7
        tp = rng.multivariate_normal(t, C)
        lpp = logpost(tp, prior_b)
        if np.log(rng.random()) < lpp - lp:
            t, lp = tp, lpp
        H_[i] = t
    return H_[niter // 2:]


def rhat(v):
    m_ = v.mean(1); W = v.var(1, ddof=1).mean(); B_ = v.shape[1] * m_.var(ddof=1)
    return float(np.sqrt(((v.shape[1] - 1) / v.shape[1] * W + B_ / v.shape[1]) / W))


def bayes_fit(prior_b, seed0):
    ch = np.array([am_chain(seed0 + s_, prior_b) for s_ in range(4)])
    rh = max(rhat(ch[:, ::5, j]) for j in range(NP_))
    smp = ch.reshape(-1, NP_)[::20]
    effs = []
    for t in smp:
        eta = t[0] + Zb @ t[1:6]
        sl = np.where(edge_b, t[7], t[6])
        d_ = 450 * (expit(eta + sl * (mS + np.sqrt(REL))) - expit(eta + sl * mS))
        effs.append((d_.mean(), d_[edge_b].mean(), d_[~edge_b].mean(),
                     (450 * (expit(eta + sl * (mS + 1)) - expit(eta + sl * mS))).mean()))
    return np.array(effs), rh


BE, BRH = bayes_fit(PRIOR_B, 100)
put("bayes_rhat", BRH)
for j, tag in enumerate(["all", "edge", "int", "true"]):
    v_ = BE[:, j]
    put(f"bayes_{tag}", np.median(v_), 1)
    put(f"bayes_{tag}_lo", np.percentile(v_, 5), 1); put(f"bayes_{tag}_hi", np.percentile(v_, 95), 1)
    put(f"bayes_{tag}_p0", np.mean(v_ > 0), 2); put(f"bayes_{tag}_p05", np.mean(v_ > 0.5), 2)
    put(f"bayes_{tag}_p1", np.mean(v_ > 1.0), 2)
BAYES_DRAWS = BE[:, 0]
BEW, _ = bayes_fit(0.25, 200)          # sensitivity: weakly informative prior on the bend slopes
put("bayes_wide", np.median(BEW[:, 0]), 1); put("bayes_wide_lo", np.percentile(BEW[:, 0], 5), 1)
put("bayes_wide_hi", np.percentile(BEW[:, 0], 95), 1); put("bayes_wide_p05", np.mean(BEW[:, 0] > 0.5), 2)
BAYES_WIDE = BEW[:, 0]
# price the headline in draft slots with the same OLS pick slope used before (pp per log-pick unit)
put("pick_equiv_bayes", int(round(100 * np.exp(-(float(NUM["bayes_all"]) / 4.5) / b_lp))))

# ----------------------------------------------------------------------------- burst: drill vs game first second
for kind in ["random", "top_epa"]:
    r, lo, hi, n, p = sp_ci(dl.d1_z, dl[f"g_d1_{kind}"])
    put(f"rho_d1_game_{kind}", r); put(f"n_d1_game_{kind}", n)

# ----------------------------------------------------------------------------- 2026-27 pre-registration numbers
n26 = 20
put("n_prereg", 2 * n26)
put("power_prereg_obs", power_at(H["r"], 2 * n26, len(MAIN)))
put("power_prereg_eb", power_at(float(NUM["eb_hoop"]), 2 * n26, len(MAIN)))
put("runs_for_07", int(np.ceil(0.7 * (1 - rel_hoop) / (rel_hoop * 0.3))))
put("rel_hoop_3runs", sb(rel_hoop, 3)); put("rel_hoop_4runs", sb(rel_hoop, 4))

# ----------------------------------------------------------------------------- board: every hoop runner, 2023-25
# Decision rule (pre-stated, applied to every row):
#   1. Shrunk Hoop G 80% interval above 0  -> "bend plus: break ties his way"
#   2. Shrunk Hoop G 80% interval below 0  -> "bend minus: must win with power or get-off"
#   3. Otherwise                            -> "neutral: hoop says nothing"
#   Film flag (overrides): 100+ NFL rushes and his 80% pressure interval sits wholly on the other side of the
#   out-of-class no-tracking forecast from where the drill points (|shrunk z| >= 0.4).
lam1 = rel_hoop
ba = dl[dl.hoop_z.notna()].copy()
ba["lam"] = ba.hoop_k * lam1 / (1 + (ba.hoop_k - 1) * lam1)
ba["shr"] = ba.lam * ba.hoop_z
ba["shr_sd"] = np.sqrt(ba.lam * (1 - ba.lam))
ba["shr_lo"], ba["shr_hi"] = ba.shr - 1.2816 * ba.shr_sd, ba.shr + 1.2816 * ba.shr_sd
ba["shr_g"] = ba.draft_year.map(hstat["mean"]) + ba.shr * ba.draft_year.map(hstat["std"])
ba = ba.merge(LH["pred"][["nfl_id", "pb", "pa"]], on="nfl_id", how="left")
ok_ = ba.rush_snaps >= MIN_SNAPS
ba.loc[~ok_, ["pb", "pa"]] = np.nan
ba["pr_lo"] = stats.beta.ppf(0.1, ba.pressures + 0.5, ba.rush_snaps - ba.pressures + 0.5)
ba["pr_hi"] = stats.beta.ppf(0.9, ba.pressures + 0.5, ba.rush_snaps - ba.pressures + 0.5)
ba.loc[~ok_, ["pr_lo", "pr_hi"]] = np.nan
ba["call"] = np.select([ba.shr_lo > 0, ba.shr_hi < 0], ["bend plus: break ties his way",
                                                         "bend minus: must win with power or get-off"],
                       "neutral: hoop says nothing")
hi_flag = ba.pb.notna() & (ba.shr >= 0.4) & (ba.pr_hi < ba.pb)
lo_flag = ba.pb.notna() & (ba.shr <= -0.4) & (ba.pr_lo > ba.pb)
ba["flag"] = np.select([hi_flag, lo_flag], ["film over drill: under-produced", "film over drill: out-produced"], "")
ba["call"] = np.where(ba.flag != "", "film first: " + ba.flag.str.replace("film over drill: ", ""), ba.call)
ba["rank_in_class"] = ba.groupby("draft_year").shr.rank(ascending=False, method="first").astype(int)
ba = ba.sort_values(["draft_year", "rank_in_class"])
put("board_all_n", len(ba))
put("board_all_plus", int(ba.call.str.startswith("bend plus").sum()))
put("board_all_minus", int(ba.call.str.startswith("bend minus").sum()))
put("board_all_flag", int((ba.flag != "").sum()))
put("board_all_withnfl", int(ba.pb.notna().sum()))
ba[["draft_year", "rank_in_class", "display_name", "role", "pick", "combine_weight", "hoop_k", "hoop_g", "hoop_z", "shr",
    "shr_lo", "shr_hi", "rush_snaps", "press_rate", "pr_lo", "pr_hi", "pb", "pa", "flag", "call"]].round(4).to_csv(
    OUT / "board_2023_2025.csv", index=False)
TABLES["flag_all"] = ba[ba.flag != ""][["display_name", "draft_year", "flag", "shr", "press_rate", "pb"]].round(4).to_dict("records")
for r_ in ba[ba.flag != ""].itertuples():
    key = r_.display_name.split(" ")[-1].lower().replace("-", "").replace(".", "")
    put(f"fa_{key}_pr", 100 * r_.press_rate, 1); put(f"fa_{key}_pb", 100 * r_.pb, 1); put(f"fa_{key}_shr", r_.shr)
    put(f"fa_{key}_snaps", int(r_.rush_snaps))
    put(f"fa_{key}_prlo", 100 * r_.pr_lo, 1); put(f"fa_{key}_prhi", 100 * r_.pr_hi, 1)

bd = ba[ba.draft_year == 2025].sort_values("shr", ascending=False).reset_index(drop=True)
bd["rank"] = np.arange(1, len(bd) + 1)
m25, s25 = hstat.loc[2025, "mean"], hstat.loc[2025, "std"]
put("board_n", len(bd)); put("board_mean_g", m25); put("board_sd_g", s25, 3)
put("board_lam1", lam1); put("board_band", 1.2816 * np.sqrt(lam1 * (1 - lam1)))
put("board_n_nfl", int(bd.pb.notna().sum()))
put("board_n_flag", int((bd.flag != "").sum()))
put("board_plus", int(bd.call.str.startswith("bend plus").sum()))
put("board_minus", int(bd.call.str.startswith("bend minus").sum()))
bd[["rank", "display_name", "role", "pick", "combine_weight", "hoop_k", "hoop_g", "hoop_z", "shr", "shr_lo", "shr_hi",
    "shr_g", "rush_snaps", "press_rate", "pr_lo", "pr_hi", "pb", "pa", "flag", "call"]].round(4).to_csv(
    OUT / "board_2025.csv", index=False)
cardp = bd.dropna(subset=["pb"])
closer = (cardp.pa - cardp.press_rate).abs() < (cardp.pb - cardp.press_rate).abs()
put("card_closer", int(closer.sum())); put("card_n", len(cardp))
put("card_binom_p", stats.binomtest(int(closer.sum()), len(cardp), 0.5, alternative="greater").pvalue, 2)
pub = bd.shr.values
rs_rel, rs_true, top5 = [], [], []
for _ in range(NBOOT):
    l1 = np.clip(RNG.choice(bs_sh), 0.05, 0.95)
    lam_ = bd.hoop_k * l1 / (1 + (bd.hoop_k - 1) * l1)
    sh_ = lam_ * bd.hoop_z
    rs_rel.append(stats.spearmanr(sh_, pub)[0])
    tru = sh_ + np.sqrt(lam_ * (1 - lam_)) * RNG.normal(size=len(bd))
    rs_true.append(stats.spearmanr(tru, pub)[0])
    top5.append(np.mean(np.argsort(-tru.values)[:5] < 5))
put("board_rs_true", np.median(rs_true)); put("board_rs_true_lo", np.percentile(rs_true, 2.5))
put("board_rs_true_hi", np.percentile(rs_true, 97.5)); put("board_top5_keep", 100 * np.mean(top5), 0)

# ---- board tables for the text (2025 top 8 + bottom 4; 2023-24 plus / minus / film-first rows)
MD = {}


def short_call(c):
    return {"bend plus": "**plus**", "bend minus": "**minus**", "neutral": "neutral"}.get(
        c.split(":")[0], "film first (" + c.split(": ")[-1].replace("-produced", "") + ")")


def prs_txt(r_):
    if pd.isna(r_.pb):
        return f"{int(r_.rush_snaps)} rushes" if r_.rush_snaps > 0 else "no rushes"
    return f"{100 * r_.press_rate:.1f}% vs {100 * r_.pb:.1f}% ({int(r_.rush_snaps)})"


def board_md(rows, tag, show_class=False):
    head = ("| " + ("Class | " if show_class else "#  | ") + "Player | Pick | Wt | Role | Bend, sd [80%] | "
            "Pressure vs forecast (rushes) | Call |")
    sep = "|" + "---|" * 8
    out = [head, sep]
    for i, r_ in enumerate(rows.itertuples()):
        k_ = f"tb_{tag}_{i}"
        s0, s1, s2 = put(k_ + "_s", r_.shr), put(k_ + "_l", r_.shr_lo), put(k_ + "_h", r_.shr_hi)
        f = lambda v: "−" + v[1:] if v.startswith("-") else v
        out.append(f"| {r_.draft_year if show_class else int(r_.rank)} | {r_.display_name} | {pk_txt(r_.pick)} | "
                   f"{int(r_.combine_weight)} | {'EDGE' if r_.role == 'EDGE' else 'Int'} | {f(s0)} [{f(s1)}, {f(s2)}] | "
                   f"{prs_txt(r_)} | {short_call(r_.call)} |")
    return "\n".join(out)


def pk_txt(p):
    return "UDFA" if p == UDFA_PICK else str(int(p))


t25 = pd.concat([bd.head(8), bd.tail(4)])
MD["board25"] = board_md(t25, "b25")
old = ba[(ba.draft_year < 2025) & (~ba.call.str.startswith("neutral"))].sort_values("shr", ascending=False)
MD["board_old"] = board_md(old, "bold", show_class=True)
put("board_old_n", len(old))
put("board_all_plus_old", int(old.call.str.startswith("bend plus").sum()))
put("board_all_minus_old", int(old.call.str.startswith("bend minus").sum()))
TABLES["md"] = MD


ex = {"top": bd.iloc[0], "bottom": bd.iloc[-1]}
for tag, r_ in ex.items():
    put(f"ex_{tag}_g", r_.hoop_g); put(f"ex_{tag}_z", r_.hoop_z, 1); put(f"ex_{tag}_shr", r_.shr)
    put(f"ex_{tag}_lo", r_.shr_lo); put(f"ex_{tag}_hi", r_.shr_hi)
    put(f"ex_{tag}_wt", int(r_.combine_weight)); put(f"ex_{tag}_snaps", int(r_.rush_snaps))
    TABLES[f"example_{tag}"] = r_.display_name
# named film-check players for Fig 6 (chosen in round 3 by the judges: Ezeiruaku and Phillips)
FILM = {}
for nm in ["Donovan Ezeiruaku", "Jordan Phillips"]:
    r_ = ba[ba.display_name == nm].iloc[0]
    key = nm.split(" ")[-1].lower()
    FILM[nm] = r_
    put(f"film_{key}_shr", r_.shr); put(f"film_{key}_pr", 100 * r_.press_rate, 1); put(f"film_{key}_pb", 100 * r_.pb, 1)
    put(f"film_{key}_prlo", 100 * r_.pr_lo, 1); put(f"film_{key}_prhi", 100 * r_.pr_hi, 1)
    put(f"film_{key}_snaps", int(r_.rush_snaps)); put(f"film_{key}_hoopg", r_.hoop_g)
    put(f"film_{key}_rank", int(bd[bd.display_name == nm]["rank"].iloc[0]))
    q_ = arcp[(arcp.nfl_id == r_.nfl_id) & (arcp.kind == "random")]
    put(f"film_{key}_arc", q_.arc_g.mean()); put(f"film_{key}_nsnap", len(q_))
    put(f"film_{key}_wide_share", 100 * q_["align"].isin(["EDGE", "OLB"]).mean(), 0)
    put(f"film_{key}_arc_pct", stats.percentileofscore(dl.arc_random.dropna(), q_.arc_g.mean()), 0)
    put(f"film_{key}_npress", int(q_.press.sum()))


# ============================================================================= FIGURES
if os.environ.get("BDB_NOFIG"):
    json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1, sort_keys=True)
    json.dump(TABLES, open(OUT / "tables.json", "w"), indent=1, default=str)
    raise SystemExit(0)
ROLEC = {"EDGE": OI["orange"], "interior": OI["blue"]}
HIC, LOC = OI["red"], OI["blue"]


def name_short(n):
    return n.split(" ")[0][0] + ". " + " ".join(n.split(" ")[1:])


def place_labels(ax, xs, ys, labels, fs=11, xmid=None):
    """Labels left of points on the right half, right of points on the left half; then push apart vertically."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    xmid = np.mean(ax.get_xlim()) if xmid is None else xmid
    pts = ax.transData.transform(np.c_[xs, ys])
    T, P = [], []
    for (px_, py_), x, l in zip(pts, xs, labels):
        right = x < xmid
        t = ax.text(0, 0, l, fontsize=fs, ha="left" if right else "right", va="center", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
        T.append(t); P.append([px_ + (14 if right else -14), py_ + 12])
    P = np.array(P, float)
    H_ = np.array([t.get_window_extent(rend).height for t in T]) + 2
    order = np.argsort(P[:, 1])
    for _ in range(200):
        moved = False
        for a_, b_ in zip(order[:-1], order[1:]):
            same_side = (T[a_].get_ha() == T[b_].get_ha()) or abs(P[a_, 0] - P[b_, 0]) < 160
            gap = P[b_, 1] - P[a_, 1]
            need = (H_[a_] + H_[b_]) / 2
            if same_side and gap < need:
                P[a_, 1] -= (need - gap) / 2; P[b_, 1] += (need - gap) / 2; moved = True
        order = np.argsort(P[:, 1])
        if not moved:
            break
    inv = ax.transData.inverted()
    for t, p, x, y in zip(T, P, xs, ys):
        xy = inv.transform(p)
        t.set_position(xy)
        ax.plot([x, xy[0]], [y, xy[1]], color=OI["grey"], lw=0.8, zorder=4)


# ---- Fig 1: four named rushers. Combine hoop run vs 20 random NFL rushes, coloured by win type; pressure snaps marked
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import textwrap
hero_pool = co.dropna(subset=["hoop_z", "arc_random"])
hero_pool = hero_pool[hero_pool.rush_snaps >= 250].sort_values("hoop_z", ascending=False)
put("hero_pool_n", len(hero_pool))
# rule: the two 2025 film checks, plus the highest- and lowest-bend EDGE rushers with 250+ NFL rushes (others)
edge_pool = hero_pool[(hero_pool.role == "EDGE") & ~hero_pool.display_name.isin(["Donovan Ezeiruaku", "Jordan Phillips"])]
P4 = [edge_pool.iloc[0].display_name, "Donovan Ezeiruaku", "Jordan Phillips", edge_pool.iloc[-1].display_name]
WTC = {"outside": OI["blue"], "inside": OI["green"], "through": "#4D4D4D", "unclassified": "#BBBBBB"}
PRC = OI["red"]
cmap = plt.get_cmap("cividis"); norm = plt.Normalize(0, 0.9)
SPAN = 13.0
FH1 = 11.6
fig = plt.figure(figsize=(FIGW, FH1))
fig.text(0.01, 0.99, "Four rushers: the hoop run, and what the same men did on 20 random NFL rushes", fontsize=16,
         weight="bold", va="top")
fig.text(0.01, 0.958, "Left square: Combine hoop run, coloured by sideways g; black ring = the half-second that sets Hoop G. "
         "Right square: 20 random NFL\npass rushes from the snap (triangle) to the throw or sack, outside to the right. "
         "Same 13-yard scale in both squares.\nLine colour = win type from the path (EDGE snaps only); red dot = that rush produced a pressure.",
         fontsize=11.5, va="top")
hand = [Line2D([], [], color=WTC["outside"], lw=2.5, label="outside arc (dip / rip)"),
        Line2D([], [], color=WTC["inside"], lw=2.5, label="inside cross (counter)"),
        Line2D([], [], color=WTC["through"], lw=2.5, label="through the blocker (power)"),
        Line2D([], [], color=WTC["unclassified"], lw=2.5, label="interior snap, not classified"),
        Line2D([], [], marker="o", color=PRC, lw=0, markersize=8, label="pressure"),
        Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="k", lw=0, markersize=11, markeredgewidth=2,
               label="Hoop G window")]
fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.01, 0.885), ncol=3, fontsize=11.5, frameon=False)
cax = fig.add_axes([0.70, 0.835, 0.25, 0.012])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[0, 0.3, 0.6, 0.9])
cb.ax.tick_params(labelsize=10.5); cax.set_title("hoop sideways g", fontsize=11, loc="left", pad=3)
F1 = {}
for idx, nm in enumerate(P4):
    r_ = ba[ba.display_name == nm].iloc[0]
    rr, cc_ = idx // 2, idx % 2
    x0, yt = 0.03 + cc_ * 0.5, 0.79 - rr * 0.395
    sgn_ = "+" if r_.shr > 0 else "−"
    fig.text(x0, yt, f"{nm}   bend {sgn_}{abs(r_.shr):.2f} sd", fontsize=15, weight="bold", va="top", color="black")
    fig.text(x0, yt - 0.028, f"{'EDGE' if r_.role == 'EDGE' else 'Interior'}, {r_.draft_year} pick {pk_txt(r_.pick)}, "
             f"{int(r_.combine_weight)} lb, {int(r_.rush_snaps)} NFL rushes", fontsize=12, va="top", color="#333333")
    w_ax = 0.19
    h_ax = w_ax * FIGW / FH1
    ybot = yt - 0.085 - h_ax
    ax = fig.add_axes([x0 + 0.03, ybot, w_ax, h_ax])
    tr = traces[r_.hoop_eid]
    xs_, ys_ = tr["_x"] - (tr["_x"].max() + tr["_x"].min()) / 2, tr["_y"] - (tr["_y"].max() + tr["_y"].min()) / 2
    if np.ptp(ys_) < np.ptp(xs_):
        xs_, ys_ = ys_, xs_
    pts = np.c_[xs_, ys_].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=norm, linewidth=3.2)
    lc.set_array(tr["_alat"][:-1]); ax.add_collection(lc)
    k = tr["_k"]
    ax.scatter([xs_[k]], [ys_[k]], s=170, facecolor="none", edgecolor="k", lw=2.2, zorder=3)
    ax.set_xlim(-SPAN / 2, SPAN / 2); ax.set_ylim(-SPAN / 2, SPAN / 2); ax.set_aspect("equal")
    ax.set_xticks([-4, 0, 4]); ax.set_yticks([-4, 0, 4]); ax.tick_params(labelsize=10, length=2, pad=1)
    ax.set_xlabel("yards", fontsize=10.5, labelpad=1)
    ax.set_title(f"Combine: {r_.hoop_g:.2f} g", fontsize=13, weight="bold", pad=4)
    gx = fig.add_axes([x0 + 0.03 + w_ax + 0.045, ybot, w_ax, h_ax])
    q_ = arcp[(arcp.nfl_id == r_.nfl_id) & (arcp.kind == "random")]
    for kk in q_.itertuples(index=False):
        lat, fwd, al, kmx, wty = arc_tr[(kk.game_id, kk.play_id, kk.nfl_id)]
        gx.plot(lat, fwd, color=WTC[wty], lw=2.0 if wty != "unclassified" else 1.3, alpha=0.9, zorder=2)
        if kk.press == 1:
            gx.scatter([lat[-1]], [fwd[-1]], s=46, color=PRC, zorder=4, edgecolor="white", lw=0.6)
    gx.scatter([0], [0], marker="^", color=OI["black"], s=50, zorder=5)
    gx.set_xlim(-8.5, 4.5); gx.set_ylim(-1, 12); gx.set_aspect("equal")
    gx.set_xticks([-8, -4, 0, 4]); gx.set_yticks([0, 4, 8]); gx.tick_params(labelsize=10, length=2, pad=1)
    gx.set_xlabel("yards (outside →)", fontsize=10.5, labelpad=1)
    gx.set_title(f"NFL, {len(q_)} rushes: {q_.arc_g.mean():.2f} g", fontsize=13, weight="bold", pad=4)
    for a_ in (ax, gx):
        for sp_ in a_.spines.values():
            sp_.set_visible(True); sp_.set_color("#BBBBBB")
    cl_ = q_[q_.wtype != "unclassified"]
    osh = (f"{100 * (cl_.wtype == 'outside').mean():.0f}% of {len(cl_)} classified EDGE rushes were outside arcs"
           if len(cl_) >= 5 else "interior snaps, win type not classified")
    fc = (f"Career pressure {100 * r_.press_rate:.1f}% vs {100 * r_.pb:.1f}% forecast; {int(q_.press.sum())} of {len(q_)} "
          f"rushes shown were pressures; {osh}.")
    fig.text(x0, ybot - 0.035, "\n".join(textwrap.wrap(fc, 58)), fontsize=11.5, va="top", style="italic")
    key = nm.split(" ")[-1].lower().replace("'", "")
    F1[nm] = dict(key=key, out=(cl_.wtype == "outside").mean() if len(cl_) else np.nan, ncl=len(cl_))
    put(f"f1_{key}_arc", q_.arc_g.mean()); put(f"f1_{key}_hoopg", r_.hoop_g); put(f"f1_{key}_shr", r_.shr)
    put(f"f1_{key}_pr", 100 * r_.press_rate, 1); put(f"f1_{key}_pb", 100 * r_.pb, 1)
    put(f"f1_{key}_npress", int(q_.press.sum())); put(f"f1_{key}_ncl", len(cl_))
    if len(cl_):
        put(f"f1_{key}_out", 100 * (cl_.wtype == "outside").mean(), 0)
    put(f"f1_{key}_arc_pct", stats.percentileofscore(dl.arc_random.dropna(), q_.arc_g.mean()), 0)
    put(f"f1_{key}_rank", int(r_.rank_in_class)); put(f"f1_{key}_classn", int((ba.draft_year == r_.draft_year).sum()))
    put(f"f1_{key}_prlo", 100 * r_.pr_lo, 1); put(f"f1_{key}_prhi", 100 * r_.pr_hi, 1)
TABLES["fig1_players"] = P4
save(fig, "fig1_four_rushers.png")

# ---- Fig 2: signal map, five Combine numbers x three NFL outcomes; tracking metrics raw -> after baseline
FIG2 = ["three_cone_z", "forty_z", "log_pick", "d1_z", "hoop_z"]
labs = {m_[0]: m_[1] for m_ in MEAS}
labs.update({"d1_z": "Rush drill: first-second burst", "hoop_z": "Run-the-hoop bend"})
OUTL = {"press_eb": "NFL pressure rate", "quick": "NFL get-off (quicker)", "arc_random": "NFL rush-path bend"}
fig = plt.figure(figsize=(FIGW, 5.6))
gs2 = fig.add_gridspec(1, 3, left=0.3, right=0.99, top=0.8, bottom=0.27, wspace=0.08)
axes = [fig.add_subplot(gs2[0, i]) for i in range(3)]
after = {("hoop_z", "press_eb"): (H["r"], H["lo"], H["hi"]),
         ("d1_z", "press_eb"): tuple(prim[("d1_z", "press_eb")][k] for k in ("r", "lo", "hi")),
         ("hoop_z", "quick"): tuple(prim[("hoop_z", "quick")][k] for k in ("r", "lo", "hi")),
         ("d1_z", "quick"): tuple(prim[("d1_z", "quick")][k] for k in ("r", "lo", "hi")),
         ("hoop_z", "arc_random"): ARC_AFTER["hoop_z"], ("d1_z", "arc_random"): ARC_AFTER["d1_z"]}
yy = np.arange(len(FIG2))[::-1]
for ax, (oc, _) in zip(axes, OUTC):
    d_ = sig[sig.outcome == oc].set_index("col").loc[FIG2]
    for yv, (c, r_) in zip(yy, d_.iterrows()):
        ax.fill_betweenx([yv - 0.42, yv + 0.42], -r_.mde, r_.mde, color="#E6E6E6", lw=0)
        colr = OI["green"] if r_.kind == "tracking" else OI["black"]
        ax.plot([r_.lo, r_.hi], [yv + 0.12, yv + 0.12], color=colr, lw=2)
        ax.scatter([r_.rho], [yv + 0.12], color=colr, s=60, zorder=3)
        if (c, oc) in after:
            ra_, la_, ha_ = after[(c, oc)]
            ax.annotate("", xy=(ra_, yv - 0.25), xytext=(r_.rho, yv - 0.25),
                        arrowprops=dict(arrowstyle="->", color=OI["purple"], lw=1.8))
            ax.scatter([ra_], [yv - 0.25], marker="D", facecolor="white", edgecolor=OI["purple"], s=60, zorder=4, lw=1.8)
            if c == "hoop_z":
                ax.text(ra_, yv - 0.62, f"after baseline: {ra_:.2f}", color=OI["purple"], fontsize=10.5,
                        va="center", ha="center", weight="bold")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-0.6, 1.0); ax.set_ylim(-0.85, len(FIG2) - 0.4)
    ax.set_title(OUTL[oc], fontsize=12.5, weight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{labs[c]} (n={int(d_.loc[c].n)})" for c in FIG2] if ax is axes[0] else [], fontsize=11.5)
    ax.set_xticks([-0.5, 0, 0.5, 1.0]); ax.set_xlabel("Spearman ρ", fontsize=11)
for c in FIG2:
    put(f"n_sig_{c}", int(sig[(sig.outcome == "press_eb") & (sig.col == c)].n.iloc[0]))
hand = [Line2D([], [], color=OI["black"], marker="o", lw=2, label="stopwatch / draft, raw (95% CI)"),
        Line2D([], [], color=OI["green"], marker="o", lw=2, label="tracked drill, raw (95% CI)"),
        Line2D([], [], color=OI["purple"], marker="D", markerfacecolor="white", lw=0,
               label="tracked drill after the no-tracking baseline\n(rush-path bend: after weight and role only)"),
        Patch(color="#E6E6E6", label="too small to detect at this n (80% power)")]
fig.legend(handles=hand, loc="lower center", ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.55, 0.0))
fig.suptitle("Bend is the one tracked drill number that still points at pressure once draft slot and size are known",
             fontsize=13.5, weight="bold", x=0.01, ha="left", y=0.97)
fig.text(0.01, 0.905, f"Better Combine number → + ρ. n is for the pressure panel (DL with 100+ NFL rushes who have the "
         f"measure);\nrush-path bend uses all DL with sampled snaps (n ≤ {int(sig[sig.outcome == 'arc_random'].n.max())}). "
         "Diamond = what is left after the baseline.", fontsize=10.5, color="#333333", va="top")
save(fig, "fig2_signal_map.png")

# ---- Fig 3: headline scatter, one facet per group (ran the 40 / skipped it)
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 6.0), sharey=True)
fig.subplots_adjust(left=0.09, right=0.98, top=0.76, bottom=0.12, wspace=0.06)
d_ = dH.copy()
d_["res"] = 100 * d_.res_main
xx = np.linspace(-2.3, 2.3, 50)
TXTN = set(P4)
for a1, (h40, colr, mk, lab, sk) in zip(axs, [(1, OI["blue"], "o", f"Ran the 40 (n={NUM['n_40']})", "40"),
                                               (0, OI["orange"], "D", f"Skipped the 40 (n={NUM['n_no40']})", "no40")]):
    gg_ = d_[d_.has40 == h40]
    a1.scatter(gg_.hoop_z, gg_.res, color=colr, s=52, marker=mk, edgecolor="white", lw=0.5, zorder=3)
    bs = np.array([np.polyval(np.polyfit(gg_.hoop_z.values[i], gg_.res.values[i], 1), xx)
                   for i in (RNG.integers(0, len(gg_), len(gg_)) for _ in range(500))])
    a1.fill_between(xx, *np.percentile(bs, [10, 90], axis=0), color=colr, alpha=0.18, lw=0)
    a1.plot(xx, np.polyval(np.polyfit(gg_.hoop_z, gg_.res, 1), xx), color=colr, lw=2.4)
    a1.axhline(0, color="k", lw=0.6)
    a1.set_xlim(-2.6, 2.9); a1.set_ylim(d_.res.min() - 1.0, d_.res.max() + 1.5)
    mn_ = lambda v: str(v).replace("-", "−")
    a1.set_title(f"{lab}: +{NUM['int_slope_' + sk]} pp per sd [{mn_(NUM['int_slope_' + sk + '_lo'])}, "
                 f"{mn_(NUM['int_slope_' + sk + '_hi'])}]", fontsize=12.5, weight="bold", color=colr)
    a1.set_xlabel("Hoop G, sd within Combine year")
    ld = gg_[gg_.display_name.isin(TXTN)]
    if len(ld):
        place_labels(a1, ld.hoop_z.values, ld.res.values, [name_short(n) for n in ld.display_name], fs=11, xmid=0.3)
axs[0].set_ylabel("NFL pressure rate above the\nno-tracking forecast (pp)")
fig.suptitle("Rushers with more bend beat their no-tracking forecast, with or without a 40 time",
             fontsize=14, weight="bold", x=0.01, ha="left", y=0.975)
fig.text(0.01, 0.915, f"{NUM['v_n_main']} DL with 100+ NFL rushes. Line = fit, band = 80% bootstrap. Slopes are "
         f"class-adjusted with 95% intervals; difference p = {NUM['int_p']}.\nForecast = pick, weight, NGS score, EDGE share. "
         "Named: the players in Fig. 1.", fontsize=10.5, color="#333333", va="top")
save(fig, "fig3_hoop_pressure.png")

# ---- Fig 4: the headline posterior (left) and the primary out-of-class estimate (right)
fig = plt.figure(figsize=(FIGW, 6.0))
g4 = fig.add_gridspec(1, 2, left=0.07, right=0.98, top=0.72, bottom=0.14, wspace=0.28, width_ratios=[1, 1.1])
f1, f2 = fig.add_subplot(g4[0]), fig.add_subplot(g4[1])
bins = np.linspace(min(BAYES_WIDE.min(), BAYES_DRAWS.min(), -1), max(np.percentile(BAYES_WIDE, 99.5), 3), 45)
f1.hist(BAYES_DRAWS, bins=bins, density=True, color=OI["blue"], alpha=0.55, label="headline prior (typical drill effect)")
f1.hist(BAYES_WIDE, bins=bins, density=True, histtype="step", color=OI["black"], lw=1.6, label="wide prior (sensitivity)")
f1.axvline(0, color="k", lw=0.8); f1.axvline(0.5, color=OI["red"], lw=1.4, ls="--")
f1.text(0.6, f1.get_ylim()[1] * 0.97, f"P(effect > 0.5) = {NUM['bayes_all_p05']}", color=OI["red"], fontsize=11.5,
        weight="bold", va="top", bbox=dict(facecolor="white", edgecolor="none", pad=1))
f1.set_xlabel("extra pressures per 450-rush season,\nper sd of measured bend"); f1.set_yticks([])
f1.set_title("Posterior effect (all rushers)", fontsize=12.5, weight="bold")
f1.legend(frameon=False, fontsize=10.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))
labs4 = ["Pooled"] + [str(c) for c in classes]
vals = [NFB[:, 0]] + [NFB[:, j + 1] for j in range(len(classes))]
pts4 = [float(NUM["nf_dr2"])] + [float(NUM[f"nf_dr2_{c}"]) for c in classes]
lo4, hi4 = -1.0, 0.6
nclip = int(sum(((v < lo4) | (v > hi4)).sum() for v in vals))
for j, (lab, v, pt) in enumerate(zip(labs4, vals, pts4)):
    yv = len(labs4) - 1 - j
    vp = f2.violinplot([np.clip(v, lo4, hi4)], positions=[yv], vert=False, widths=0.8, showextrema=False)
    colr = OI["black"] if j == 0 else OI["green"]
    for b_ in vp["bodies"]:
        b_.set_facecolor(colr); b_.set_alpha(0.3)
    a_, b2_ = np.percentile(v, [2.5, 97.5])
    f2.plot([a_, b2_], [yv, yv], color=colr, lw=1.6)
    f2.scatter([pt], [yv], color=colr, s=55, zorder=3)
    pout_ = 100 * float(((v < lo4) | (v > hi4)).mean())
    if pout_ >= 1:
        f2.text(lo4 + 0.02, yv + 0.42, f"{pout_:.0f}% of draws beyond the axis, drawn at its edge",
                fontsize=10, color="#555555")
f2.axvline(0, color="k", lw=0.8)
f2.set_yticks(range(len(labs4))); f2.set_yticklabels(labs4[::-1], fontsize=11.5)
f2.set_xlim(lo4, hi4)
f2.set_xlabel("held-out R² change vs no-tracking forecast")
f2.set_title("Fully nested out-of-class test", fontsize=12.5, weight="bold")
put("fig4_nclip", nclip)
fig.suptitle(f"Bend is worth about {NUM['bayes_all']} pressures a season, but it "
             + ("does not improve" if float(NUM["nf_dr2"]) <= 0 else "barely improves") + " a forecast of a new class",
             fontsize=13.5, weight="bold", x=0.01, ha="left", y=0.975)
fig.text(0.01, 0.9, "Left: Bayesian beta-binomial model, every hoop runner with an NFL rush, bend corrected for one-run "
         "reliability.\nRight: each class predicted from the other two, with screening, reliability and shrinkage refit in "
         f"every fold.\nDot = estimate, line = 95% class-stratified player bootstrap ({NUM['nf_nboot']} refits of the whole loop).",
         fontsize=10.5, color="#333333", va="top")
save(fig, "fig4_headline.png")

# ---- Fig 5: compact 2025 board (rows that carry information)
CALLC = {"bend plus": OI["red"], "bend minus": OI["blue"], "neutral": OI["grey"], "film first": OI["purple"]}
bdf = bd.copy()
bdf["ck"] = bdf.call.str.split(":").str[0]
keep_ = (bdf.ck != "neutral") | bdf.nfl_id.isin(bdf[bdf.rush_snaps >= 200].nlargest(5, "rush_snaps").nfl_id)
bdf = bdf[keep_].sort_values("shr", ascending=False)
put("fig5_rows", len(bdf))
HH = 0.42 * len(bdf) + 2.9
fig = plt.figure(figsize=(FIGW, HH))
g5 = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], left=0.37, right=0.86, top=1 - 1.25 / HH, bottom=1.45 / HH,
                      wspace=0.12)
b1, b2 = fig.add_subplot(g5[0]), fig.add_subplot(g5[1])
yy = np.arange(len(bdf))[::-1]
for yv, r_ in zip(yy, bdf.itertuples()):
    c_ = CALLC[r_.ck]
    b1.plot([r_.shr_lo, r_.shr_hi], [yv, yv], color=c_, lw=2.6, alpha=0.6)
    b1.scatter([r_.shr], [yv], color=c_, s=50, zorder=3, marker="o" if r_.role == "EDGE" else "s")
    if pd.notna(r_.pb):
        b2.plot([100 * r_.pr_lo, 100 * r_.pr_hi], [yv, yv], color=OI["black"], lw=1.8, alpha=0.6)
        b2.scatter([100 * r_.press_rate], [yv], s=45, color=OI["black"], zorder=4)
        b2.scatter([100 * r_.pb], [yv], s=90, marker="|", color=OI["orange"], lw=2.8, zorder=5)
    else:
        b2.text(0.3, yv, f"{int(r_.rush_snaps)} NFL rushes" if r_.rush_snaps > 0 else "no NFL rushes",
                va="center", fontsize=10.5, color="#666666")
    b2.text(1.02, yv, r_.ck, transform=b2.get_yaxis_transform(), va="center", fontsize=11, color=c_,
            weight="bold" if r_.ck != "neutral" else "normal")
b1.axvline(0, color="k", lw=0.8)
b1.set_yticks(yy)
b1.set_yticklabels([f"{r_.display_name}  {'EDGE' if r_.role == 'EDGE' else 'Int'}, "
                    f"{'UDFA' if r_.pick == UDFA_PICK else '#' + str(int(r_.pick))}" for r_ in bdf.itertuples()],
                   fontsize=11.5)
b2.set_yticks(yy); b2.set_yticklabels([])
for b_ in (b1, b2):
    b_.set_ylim(-0.7, len(bdf) - 0.3); b_.tick_params(labelsize=11); b_.grid(axis="y", color="#EEEEEE", lw=0.6)
b1.set_xlim(-1.7, 1.7); b2.set_xlim(0, 14); b2.set_xticks([0, 4, 8, 12])
b1.set_xlabel("bend, sd (80% interval)", fontsize=11.5)
b2.set_xlabel("pressure rate, % (80% interval)", fontsize=11.5)
b1.set_title("Combine bend (shrunk)", fontsize=12.5, weight="bold"); b2.set_title("2025 rookie season", fontsize=12.5,
                                                                                  weight="bold")
hand = [Line2D([], [], marker="|", color=OI["orange"], lw=0, markersize=13, markeredgewidth=2.8,
               label="no-tracking forecast"),
        Line2D([], [], marker="o", color="k", lw=1.8, markersize=6, label="actual"),
        Line2D([], [], marker="o", color="#555555", lw=0, label="EDGE"),
        Line2D([], [], marker="s", color="#555555", lw=0, label="interior"),
        Patch(color=CALLC["bend plus"], label="bend plus"), Patch(color=CALLC["bend minus"], label="bend minus"),
        Patch(color=CALLC["film first"], label="film first"), Patch(color=CALLC["neutral"], label="neutral")]
fig.legend(handles=hand, frameon=False, fontsize=11, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4)
fig.suptitle(f"2025 board: the {len(bdf)} rows that carry information", fontsize=14, weight="bold", x=0.01, ha="left",
             y=1 - 0.15 / HH)
fig.text(0.01, 1 - 0.62 / HH, "Plus / minus = 80% bend interval clear of zero. Film first = rookie pressure interval wholly "
         "on the wrong side of the forecast.\nNeutral rows shown are the five with the most rookie rushes.",
         fontsize=10.5, color="#333333", va="top")
save(fig, "fig5_board_2025.png")


# ----------------------------------------------------------------------------- write numbers (one key per quoted claim)
if os.environ.get("BDB_ALLNUM"):    # dev only: full dump (not part of the deliverable)
    json.dump(NUM, open(Path(os.environ["BDB_ALLNUM"]), "w"), indent=1, sort_keys=True)
used = render(OUT, NUM, MD)
keep = {}
for k_ in sorted(used | {k for k in NUM if k.startswith("tb_")}):
    keep[k_] = NUM[k_]
    if str(NUM[k_]).startswith("-"):
        keep[k_ + "_abs"] = str(NUM[k_])[1:]     # check.py reads digits without the sign
json.dump(keep, open(OUT / "numbers.json", "w"), indent=1, sort_keys=True)
json.dump(TABLES, open(OUT / "tables.json", "w"), indent=1, default=str)
print(json.dumps({k: NUM[k] for k in sorted(NUM) if k.startswith(("nf_", "bayes", "wt_", "plc_", "neg_", "f1_"))}, indent=0))
