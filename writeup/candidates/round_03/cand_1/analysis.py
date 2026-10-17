"""Hoop G (Run-the-Hoop bend) vs NFL pressure rate for DL/EDGE, with a selection-honest test.

Run from the repo root:
    uv run python writeup/candidates/round_03/cand_1/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json, tables.json and figures/*.png next to this file.
Every decimal quoted in writeup.md is produced here.

Design (round 3; builds on round_02/revised):
  * New tracking-native outcome, NFL Arc G: the hardest sideways acceleration a rusher holds for 0.5 s between the
    snap and the throw/sack (cap 3.5 s) on his sampled NFL pass rushes, same formula as Hoop G. Self-referenced
    (no ball or opponents needed). Tests whether drill bend shows up as game bend, whether game arc goes with
    pressure within a player, and whether the Hoop G -> pressure link runs through game arc.
Earlier design (round_02/revised):
  * Baseline = log pick + weight + 10-yd split + EDGE share (4 covariates, no tracking). The round-2 six-covariate
    baseline (adds 3-cone and NGS athleticism) is kept as a robustness row, fitted by OLS, ridge and with
    complete cases only (no imputation).
  * Selection: Hoop G was the best of six drill metrics screened in round 1. All seven drill metrics (those six plus
    rush-drill first-second distance) are rescreened (a) in-sample with a max-over-metrics permutation test and
    (b) inside a nested leave-one-class-out loop that re-selects the metric on the training classes only.
  * Empirical-Bayes shrinkage of the seven screened effects gives a selection-corrected effect size.
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
gfa = gf.sort_values(K + ["frame"]).join(ends, on=K)
arc_rows, arc_tr = [], {}
pinfo = g.drop_duplicates(K).set_index(K)[["play_direction", "sample_kind", "time_to_pressure", "sack"]]
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
    arc_rows.append(dict(zip(K, key), arc_g=float(np.nanmax(roll)) / YD_PER_G, dur=(e - sf) * DT,
                         kind=info.sample_kind,
                         press=int(pd.notna(info.time_to_pressure) or info.sack == 1)))
    # rush path in a defender frame: forward = toward the offense, mirrored so the path ends to the right
    fwd = -(xs - xs[0]) if info.play_direction == "right" else (xs - xs[0])
    lat = ys - ys[0]
    ii = np.flatnonzero(inw)
    f0 = ii[0]
    fwd, lat = fwd - fwd[f0], lat - lat[f0]
    if lat[ii[-1]] < 0:
        lat = -lat
    arc_tr[key] = (lat[ii], fwd[ii], al[ii] / YD_PER_G)
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

SMALL = ["log_pick", "combine_weight_z", "ten_yd_split_z", "edge_share"]
FULL = SMALL + ["three_cone_z", "ngs_athleticism_score"]
put("k_base", len(SMALL))


def impute(tr: pd.DataFrame, te: pd.DataFrame, cols=("ten_yd_split_z", "three_cone_z")):
    """Fill missing official times from weight + EDGE share, fit on training rows only (no outcome)."""
    tr, te = tr.copy(), te.copy()
    for c in cols:
        mk = tr[c].notna()
        A = np.c_[np.ones(mk.sum()), tr.loc[mk, ["combine_weight_z", "edge_share"]].values]
        b = np.linalg.lstsq(A, tr.loc[mk, c].values, rcond=None)[0]
        for d in (tr, te):
            m2 = d[c].isna()
            d.loc[m2, c] = np.c_[np.ones(m2.sum()), d.loc[m2, ["combine_weight_z", "edge_share"]].values] @ b
    return tr, te


coi, _ = impute(co, co)

# ---- raw signals (for the signal map)
MEAS = [("forty_z", "40-yard time", "official", -1), ("ten_yd_split_z", "10-yd split", "official", -1),
        ("three_cone_z", "3-cone", "official", -1), ("short_shuttle_z", "Short shuttle", "official", -1),
        ("combine_weight_z", "Lighter weight", "official", -1), ("ngs_athleticism_score", "NGS athleticism", "official", 1),
        ("log_pick", "Earlier draft pick", "official", -1), ("d1_z", "Rush-drill first second", "tracking", 1),
        ("hoop_z", "Hoop G (bend)", "tracking", 1)]
OUTC = [("press_eb", "NFL pressure rate"), ("quick", "NFL get-off (quicker)")]
sig = []
for col, lab, kind, sgn in MEAS:
    for oc, olab in OUTC:
        r, lo, hi, n, p = sp_ci(sgn * co[col], co[oc])
        sig.append(dict(col=col, label=lab, kind=kind, outcome=oc, rho=r, lo=lo, hi=hi, n=n, mde=mde(n)))
sig = pd.DataFrame(sig)
for r_ in sig.itertuples():
    put(f"rho_{r_.col}_{r_.outcome}", r_.rho); put(f"n_{r_.col}_{r_.outcome}", int(r_.n))
TABLES["signals"] = sig.round(3).to_dict("records")
put("rho_hoop_weight", stats.spearmanr(dl.hoop_z, dl.combine_weight, nan_policy="omit")[0])
put("rho_hoop_pick", stats.spearmanr(dl.hoop_z, -dl.log_pick, nan_policy="omit")[0])
put("rho_hoop_10yd", stats.spearmanr(dl.hoop_z, -dl.ten_yd_split_z, nan_policy="omit")[0])

# ---- primary tests after the SMALL baseline
REL_OUT = {"quick": sb(rel_getoff_season, co.n_seasons.mean()), "press_eb": sb(rel_press_season, co.n_seasons.mean())}
put("rel_press_career", REL_OUT["press_eb"]); put("rel_getoff_career", REL_OUT["quick"])
REL_X = {"d1_z": rel_d1_player, "hoop_z": rel_hoop}
PRIMARY = [("hoop_z", "press_eb"), ("hoop_z", "quick"), ("d1_z", "quick"), ("d1_z", "press_eb")]
prim = {}
for x, y in PRIMARY:
    r, lo, hi, n, p = partial_sp(coi, x, y, SMALL)
    m_obs = mde(n, len(SMALL))
    ceil = np.sqrt(REL_X[x] * REL_OUT[y])
    prim[(x, y)] = dict(r=r, lo=lo, hi=hi, n=n, p=p, mde=m_obs, mde_true=min(m_obs / ceil, 1.0))
    key = f"{x}_{y}"
    put(f"pr_{key}", r); put(f"pr_lo_{key}", lo); put(f"pr_hi_{key}", hi); put(f"pr_n_{key}", n)
    put(f"pr_p_{key}", p, 3); put(f"pmde_{key}", m_obs); put(f"mdetrue_{key}", prim[(x, y)]["mde_true"])
    put(f"tost_{key}", tost_p(r, n, 0.30, len(SMALL)), 3)
ps = np.array([prim[k]["p"] for k in PRIMARY])
o = np.argsort(ps)
holm = np.empty(len(ps)); run = 0
for rank, i in enumerate(o):
    run = max(run, (len(ps) - rank) * ps[i]); holm[i] = min(1, run)
for (x, y), h in zip(PRIMARY, holm):
    put(f"holm_{x}_{y}", h, 3)
# smallest effect we could see for Hoop G on the true scale, across the plausible reliability range
H = prim[("hoop_z", "press_eb")]
for tag, rv in [("sh", r_sh), ("sb", rel_hoop), ("sblo", float(NUM["rel_hoop_sb_lo"])), ("sbhi", float(NUM["rel_hoop_sb_hi"]))]:
    put(f"mdetrue_hoop_{tag}", min(H["mde"] / np.sqrt(rv * REL_OUT["press_eb"]), 1.0))

# role split and interaction for Hoop G
for role in ["EDGE", "interior"]:
    c = coi[coi.role == role]
    r, lo, hi, n, p = partial_sp(c, "hoop_z", "press_eb", [b for b in SMALL if b != "edge_share"])
    put(f"pr_hoop_{role}", r); put(f"pr_n_hoop_{role}", n); put(f"pr_lo_hoop_{role}", lo); put(f"pr_hi_hoop_{role}", hi)

# ----------------------------------------------------------------------------- does drill bend show up as game bend?
ARCB = ["combine_weight_z", "edge_share"]
r, lo, hi, n, p = sp_ci(dl.hoop_z, dl.arc_random)
put("rho_hoop_arc_raw", r); put("rho_hoop_arc_raw_lo", lo); put("rho_hoop_arc_raw_hi", hi); put("n_hoop_arc", n)
put("rho_arc_weight", stats.spearmanr(dl.arc_random, dl.combine_weight, nan_policy="omit")[0])
put("rho_arc_edge", stats.spearmanr(dl.arc_random, dl.edge_share, nan_policy="omit")[0])
dA = dl.dropna(subset=["hoop_z", "arc_random"] + ARCB)
r, lo, hi, n, p = partial_sp(dA, "hoop_z", "arc_random", ARCB)
put("pr_hoop_arc", r); put("pr_hoop_arc_lo", lo); put("pr_hoop_arc_hi", hi); put("pr_hoop_arc_n", n); put("pr_hoop_arc_p", p)
m_ = mde(n, len(ARCB)); put("pmde_hoop_arc", m_)
put("mdetrue_hoop_arc", min(m_ / np.sqrt(rel_hoop * ARC["random"]), 1.0))
put("tost_hoop_arc", tost_p(r, n, 0.30, len(ARCB)), 3)
dT = dl.dropna(subset=["hoop_z", "arc_top_epa"] + ARCB)
r, lo, hi, n, p = partial_sp(dT, "hoop_z", "arc_top_epa", ARCB)
put("pr_hoop_arc_top", r); put("pr_hoop_arc_top_lo", lo); put("pr_hoop_arc_top_hi", hi); put("pr_hoop_arc_top_n", n)
dE = dA[dA.role == "EDGE"]
r, lo, hi, n, p = partial_sp(dE, "hoop_z", "arc_random", ["combine_weight_z"])
put("pr_hoop_arc_edge", r); put("pr_hoop_arc_edge_lo", lo); put("pr_hoop_arc_edge_hi", hi); put("pr_hoop_arc_edge_n", n)
# game arc and pressure, player level
r, lo, hi, n, p = sp_ci(co.arc_random, co.press_eb)
put("rho_arc_press_raw", r); put("n_arc_press", n)
r, lo, hi, n, p = partial_sp(coi, "arc_random", "press_eb", SMALL)
put("pr_arc_press", r); put("pr_arc_press_lo", lo); put("pr_arc_press_hi", hi); put("pr_arc_press_n", n)
# mediation: does the Hoop G -> pressure link run through game arc?
r, lo, hi, n, p = partial_sp(coi, "hoop_z", "press_eb", SMALL + ["arc_random"])
put("pr_hoop_press_given_arc", r); put("pr_hoop_press_given_arc_lo", lo); put("pr_hoop_press_given_arc_hi", hi)
put("pr_hoop_press_given_arc_n", n)
# within player: are his higher-arc snaps his pressure snaps? (random half, play length controlled)
qa = arcp[(arcp.kind == "random")].dropna(subset=["gz"]).copy()
lfit = smf.logit("press ~ gz + dur", qa).fit(disp=0)
put("within_or", np.exp(lfit.params["gz"])); put("within_n_plays", len(qa)); put("within_n_players", qa.nfl_id.nunique())
put("within_n_press", int(qa.press.sum()))
ids_ = qa.nfl_id.unique(); grp = {i: d for i, d in qa.groupby("nfl_id")}
bor = []
for _ in range(max(200, NBOOT // 4)):
    d_ = pd.concat([grp[i] for i in RNG.choice(ids_, len(ids_))])
    try:
        bor.append(np.exp(smf.logit("press ~ gz + dur", d_).fit(disp=0).params["gz"]))
    except Exception:
        pass
put("within_or_lo", np.percentile(bor, 2.5)); put("within_or_hi", np.percentile(bor, 97.5))
put("within_gz_press", qa[qa.press == 1].gz.mean()); put("within_gz_nopress", qa[qa.press == 0].gz.mean())

# ----------------------------------------------------------------------------- selection-honest screen
def fast_partial(xr, yr, A):
    return _pr(xr, yr, A)


def screen_setup(df, outc, base):
    """Per metric: rows, rank-design and residualized x-ranks. y ranks recomputed per permutation."""
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


S_p = screen_setup(coi, "press_eb", SMALL)
S_g = screen_setup(coi, "quick", SMALL)
obs_p = screen_stats(S_p, coi.press_eb)
obs_g = screen_stats(S_g, coi.quick)
scr = pd.DataFrame([{"col": c, "label": l, "pr_press": obs_p[c], "pr_getoff": obs_g[c],
                     "n_press": len(S_p[c]["idx"]), "n_getoff": len(S_g[c]["idx"])} for c, l in SCREEN])
best_p = scr.loc[scr.pr_press.abs().idxmax()]
put("screen_best_is_hoop", int(best_p.col == "hoop_z"))
max_obs_p = scr.pr_press.abs().max()
max_obs_all = max(scr.pr_press.abs().max(), scr.pr_getoff.abs().max())
null_p, null_all, null_hoop = [], [], []
NPERM_SCREEN = 5 * NPERM
for _ in range(NPERM_SCREEN):
    i = RNG.permutation(len(coi))
    # permute players' outcome pair jointly within class (keeps pressure-get-off link)
    yp = coi.press_eb.copy(); yg = coi.quick.copy()
    for c, ix in coi.groupby("draft_year").groups.items():
        ix = np.asarray(ix)
        pi = ix[RNG.permutation(len(ix))]
        yp.loc[ix] = coi.press_eb.loc[pi].values; yg.loc[ix] = coi.quick.loc[pi].values
    sp_ = screen_stats(S_p, yp); sg_ = screen_stats(S_g, yg)
    mp = max(abs(v) for v in sp_.values())
    null_p.append(mp); null_hoop.append(abs(sp_["hoop_z"]))
    null_all.append(max(mp, max(abs(v) for v in sg_.values())))
null_p = np.array(null_p)
put("perm_single_hoop", (1 + np.sum(np.array(null_hoop) >= abs(obs_p["hoop_z"]))) / (1 + NPERM_SCREEN), 3)
put("perm_best7", (1 + np.sum(null_p >= max_obs_p)) / (1 + NPERM_SCREEN), 3)
put("n_perm_screen", NPERM_SCREEN)
put("perm_best14", (1 + np.sum(np.array(null_all) >= max_obs_all)) / (1 + NPERM_SCREEN), 3)
put("null_best7_p95", np.percentile(null_p, 95))
for r_ in scr.itertuples():
    put(f"scr_{r_.col}", r_.pr_press); put(f"scrg_{r_.col}", r_.pr_getoff)
TABLES["screen"] = scr.round(3).to_dict("records")

# empirical-Bayes shrinkage of the seven screened pressure effects (Fisher z) -> selection-corrected effect
z = np.arctanh(scr.pr_press.values)
se2 = 1 / (scr.n_press.values - 3 - len(SMALL))
mz = np.average(z, weights=1 / se2)
tau2_s = max(0.0, np.var(z, ddof=1) - se2.mean())
B = tau2_s / (tau2_s + se2)
z_shr = mz + B * (z - mz)
ih = list(scr.col).index("hoop_z")
put("eb_tau", np.sqrt(tau2_s)); put("eb_hoop", np.tanh(z_shr[ih])); put("eb_mean", np.tanh(mz))
put("eb_B_hoop", B[ih])
# shrunk effect in pressure points per sd (scale the slope by the same factor)
eb_factor = np.tanh(z_shr[ih]) / obs_p["hoop_z"]


# ----------------------------------------------------------------------------- leave-one-class-out machinery
def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ols_pred(tr, te, cols, y):
    A = np.c_[np.ones(len(tr)), tr[cols].values]
    b = np.linalg.lstsq(A, tr[y].values, rcond=None)[0]
    return np.c_[np.ones(len(te)), te[cols].values] @ b


def ridge_pred(tr, te, cols, y, alphas=np.logspace(-2, 3, 30)):
    """Ridge on training-standardized covariates, alpha by leave-one-out GCV on the training classes."""
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


def loco(df, base, extra, y, fit=ols_pred, imp=True):
    classes = sorted(df.draft_year.unique())
    Y, PB, PA, CL, ID = [], [], [], [], []
    for c in classes:
        tr, te = df[df.draft_year != c], df[df.draft_year == c]
        if imp:
            tr, te = impute(tr, te)
        PB.append(fit(tr, te, base, y)); PA.append(fit(tr, te, base + extra, y))
        Y.append(te[y].values); CL.append(np.full(len(te), c)); ID.append(te.nfl_id.values)
    Y, PB, PA, CL, ID = map(np.concatenate, (Y, PB, PA, CL, ID))
    out = dict(n=len(Y), r2_base=r2(Y, PB), r2_aug=r2(Y, PA))
    out["delta"] = out["r2_aug"] - out["r2_base"]
    out["pred"] = pd.DataFrame({"nfl_id": ID, "y": Y, "pb": PB, "pa": PA, "cl": CL})
    return out


def boot_loco(df, base, extra, y, nb=1000, imp=True):
    """Refit bootstrap: resample players within class, rerun the whole leave-one-class-out loop."""
    groups = {c: d for c, d in df.groupby("draft_year")}
    out = []
    for _ in range(nb):
        d = pd.concat([gg.iloc[RNG.integers(0, len(gg), len(gg))] for gg in groups.values()])
        try:
            out.append(loco(d, base, extra, y, imp=imp)["delta"])
        except Exception:
            pass
    return np.percentile(out, [2.5, 97.5])


dH = co.dropna(subset=["hoop_z", "press_eb"]).copy()
LH = loco(dH, SMALL, ["hoop_z"], "press_eb")
put("lo_n", LH["n"]); put("r2b", LH["r2_base"]); put("r2a", LH["r2_aug"]); put("dr2", LH["delta"], 3)
lo_, hi_ = boot_loco(dH, SMALL, ["hoop_z"], "press_eb")
put("dr2_lo", lo_, 3); put("dr2_hi", hi_, 3)
# per-class rank rho with bootstrap intervals (held-out players resampled)
fold = {}
for c, d_ in LH["pred"].groupby("cl"):
    rb, ra = stats.spearmanr(d_.pb, d_.y)[0], stats.spearmanr(d_.pa, d_.y)[0]
    bb_, ba_, bd_ = [], [], []
    for _ in range(NBOOT):
        i = RNG.integers(0, len(d_), len(d_))
        x1, x2 = stats.spearmanr(d_.pb.values[i], d_.y.values[i])[0], stats.spearmanr(d_.pa.values[i], d_.y.values[i])[0]
        bb_.append(x1); ba_.append(x2); bd_.append(x2 - x1)
    fold[int(c)] = dict(n=len(d_), rb=rb, ra=ra, rb_ci=np.nanpercentile(bb_, [10, 90]), ra_ci=np.nanpercentile(ba_, [10, 90]),
                        d_ci=np.nanpercentile(bd_, [2.5, 97.5]))
    put(f"frho_b_{c}", rb); put(f"frho_a_{c}", ra); put(f"fn_{c}", len(d_))
put("folds_up", int(sum(f["ra"] > f["rb"] for f in fold.values())))

# ---- nested leave-one-class-out: re-select the best of the seven metrics on the training classes only
cand = co.dropna(subset=["press_eb"]).copy()
cand = cand[cand[[c for c, _ in SCREEN]].notna().any(axis=1)]
classes = sorted(cand.draft_year.unique())
FOLD = {}
for c in classes:
    tr, te = impute(cand[cand.draft_year != c], cand[cand.draft_year == c])
    FOLD[c] = (tr, te, screen_setup(tr, "press_eb", SMALL))


def nested(yser):
    Y, PB, PA, sel = [], [], [], []
    for c in classes:
        tr, te, S = FOLD[c]
        tr = tr.assign(press_eb=yser.loc[tr.index].values); te = te.assign(press_eb=yser.loc[te.index].values)
        st = screen_stats(S, tr.press_eb)
        best = max(st, key=lambda k: abs(st[k]))
        trm, tem = tr.dropna(subset=[best]), te.dropna(subset=[best])
        PB.append(ols_pred(trm, tem, SMALL, "press_eb")); PA.append(ols_pred(trm, tem, SMALL + [best], "press_eb"))
        Y.append(tem.press_eb.values); sel.append(best)
    Y, PB, PA = map(np.concatenate, (Y, PB, PA))
    return r2(Y, PA) - r2(Y, PB), sel, len(Y)


nd, nsel, nn = nested(cand.press_eb)
put("nested_dr2", nd, 3); put("nested_n", nn)
put("nested_hoop_folds", int(sum(s_ == "hoop_z" for s_ in nsel)))
TABLES["nested_selected"] = dict(zip([int(c) for c in classes], nsel))
null_n = np.array([nested(perm_within_class(cand, "press_eb"))[0] for _ in range(NPERM)])
put("nested_perm_p", (1 + np.sum(null_n >= nd)) / (1 + NPERM), 3)
# plain (no reselection) permutation of Hoop G dR2 for reference
null_h = []
for _ in range(NPERM):
    null_h.append(loco(dH.assign(press_eb=perm_within_class(dH, "press_eb")), SMALL, ["hoop_z"], "press_eb")["delta"])
put("perm_dr2_hoop", (1 + np.sum(np.array(null_h) >= LH["delta"])) / (1 + NPERM), 3)

# ----------------------------------------------------------------------------- baseline robustness for Hoop G
ROB = []


def rob_row(tag, label, df, base, imp=True, fit=ols_pred, do_loco=True):
    d = df.dropna(subset=["hoop_z", "press_eb"] + ([] if imp else base)).copy()
    dd = impute(df, df)[0].loc[d.index] if imp else d
    r, lo, hi, n, p = partial_sp(dd, "hoop_z", "press_eb", base)
    row = dict(tag=tag, label=label, n=n, r=r, lo=lo, hi=hi)
    put(f"rob_r_{tag}", r); put(f"rob_lo_{tag}", lo); put(f"rob_hi_{tag}", hi); put(f"rob_n_{tag}", n)
    if do_loco and d.groupby("draft_year").size().min() >= 5:
        L = loco(d, base, ["hoop_z"], "press_eb", fit=fit, imp=imp)
        row["dr2"] = L["delta"]; put(f"rob_dr2_{tag}", L["delta"], 3)
    ROB.append(row)


rob_row("small", "4-covariate baseline, 10-yd imputed (main)", co, SMALL)
rob_row("small_cc", "4-covariate, 10-yd missing dropped", co, SMALL, imp=False)
rob_row("full", "6-covariate (+3-cone, NGS), imputed", co, FULL)
rob_row("full_ridge", "6-covariate, ridge", co, FULL, fit=ridge_pred)
rob_row("full_cc", "6-covariate, 3-cone and 10-yd missing dropped", co, FULL, imp=False, do_loco=False)
rob_row("snaps50", "50+ rushes", dl[dl.rush_snaps >= 50], SMALL)
rob_row("snaps200", "200+ rushes", dl[dl.rush_snaps >= 200], SMALL)
# ridge, full baseline: also with-and-without Hoop G dR2 already in rob_row
# mixed model with a class random intercept (pressure in pp)
dm = impute(dH, dH)[0].assign(pp=lambda d: 100 * d.press_eb)
mfit = smf.mixedlm("pp ~ log_pick + combine_weight_z + ten_yd_split_z + edge_share + hoop_z", dm,
                   groups=dm.draft_year).fit(reml=True)
ci = mfit.conf_int().loc["hoop_z"]
put("mixed_hoop_pp", mfit.params["hoop_z"]); put("mixed_hoop_lo", ci[0]); put("mixed_hoop_hi", ci[1])
ofit = smf.ols("pp ~ log_pick + combine_weight_z + ten_yd_split_z + edge_share + hoop_z", dm).fit()
oc = ofit.conf_int().loc["hoop_z"]
put("slope_hoop_pp", ofit.params["hoop_z"]); put("slope_hoop_pp_lo", oc[0]); put("slope_hoop_pp_hi", oc[1])
put("slope_hoop_eb_pp", ofit.params["hoop_z"] * eb_factor)
put("pressures_per_450", 4.5 * ofit.params["hoop_z"], 1)
put("pressures_per_450_eb", 4.5 * ofit.params["hoop_z"] * eb_factor, 1)
TABLES["robustness"] = pd.DataFrame(ROB).round(3).to_dict("records")

# ----------------------------------------------------------------------------- burst: interaction and top-EPA replication
for kind in ["random", "top_epa"]:
    r, lo, hi, n, p = sp_ci(dl.d1_z, dl[f"g_d1_{kind}"])
    put(f"rho_d1_game_{kind}", r); put(f"lo_d1_game_{kind}", lo); put(f"hi_d1_game_{kind}", hi); put(f"n_d1_game_{kind}", n)
dd = dl.dropna(subset=["d1_z", "g_d1_random"])
ed, it = dd[dd.role == "EDGE"], dd[dd.role == "interior"]
put("rho_d1_game_EDGE", stats.spearmanr(ed.d1_z, ed.g_d1_random)[0]); put("n_d1_game_EDGE", len(ed))
put("rho_d1_game_int", stats.spearmanr(it.d1_z, it.g_d1_random)[0]); put("n_d1_game_int", len(it))
diffs = []
for _ in range(NBOOT):
    a_ = ed.iloc[RNG.integers(0, len(ed), len(ed))]; b_ = it.iloc[RNG.integers(0, len(it), len(it))]
    diffs.append(stats.spearmanr(a_.d1_z, a_.g_d1_random)[0] - stats.spearmanr(b_.d1_z, b_.g_d1_random)[0])
put("d1_edge_minus_int", float(NUM["rho_d1_game_EDGE"]) - float(NUM["rho_d1_game_int"]))
put("d1_edge_minus_int_lo", np.nanpercentile(diffs, 2.5)); put("d1_edge_minus_int_hi", np.nanpercentile(diffs, 97.5))

# ----------------------------------------------------------------------------- 2026 pre-registration numbers
n26 = 20
put("n2026", n26)
put("power2026_eb", power_at(float(NUM["eb_hoop"]), n26, len(SMALL)))
put("power2026_obs", power_at(H["r"], n26, len(SMALL)))
put("n_prereg", 2 * n26)
put("power_prereg_obs", power_at(H["r"], 2 * n26, len(SMALL)))
put("power_prereg_eb", power_at(float(NUM["eb_hoop"]), 2 * n26, len(SMALL)))
k_ = len(SMALL)
n1, z1 = H["n"], np.arctanh(H["r"])
w1, w2 = n1 - 3 - k_, n26 - 3 - k_
put("cmm_r2026", float(np.tanh((1.96 * np.sqrt(w1 + w2) - w1 * z1) / w2)))

# ----------------------------------------------------------------------------- board: 2025 class, every hoop runner
lam1 = rel_hoop
bd = dl[(dl.draft_year == 2025) & dl.hoop_z.notna()].copy()
bd["lam"] = bd.hoop_k * lam1 / (1 + (bd.hoop_k - 1) * lam1)
bd["shr"] = bd.lam * bd.hoop_z
bd["shr_sd"] = np.sqrt(bd.lam * (1 - bd.lam))   # posterior sd of the true score, observed-z units
bd["shr_lo"], bd["shr_hi"] = bd.shr - 1.2816 * bd.shr_sd, bd.shr + 1.2816 * bd.shr_sd
m25, s25 = hstat.loc[2025, "mean"], hstat.loc[2025, "std"]
bd["shr_g"] = m25 + bd.shr * s25
put("board_n", len(bd)); put("board_mean_g", m25); put("board_sd_g", s25, 3)
put("board_lam1", lam1)
put("board_band", 1.2816 * np.sqrt(lam1 * (1 - lam1)))
# NFL rookie check: baseline forecast trained on 2023-24 only (from the main leave-one-class-out run)
p25 = LH["pred"][LH["pred"].cl == 2025][["nfl_id", "pb", "pa", "y"]]
bd = bd.merge(p25, on="nfl_id", how="left")
bd["pr_lo"] = stats.beta.ppf(0.1, bd.pressures + 0.5, bd.rush_snaps - bd.pressures + 0.5)
bd["pr_hi"] = stats.beta.ppf(0.9, bd.pressures + 0.5, bd.rush_snaps - bd.pressures + 0.5)
bd["film"] = bd.press_rate - bd.pb                # rookie pressure above the no-tracking forecast
# NFL Arc G adjusted for weight and EDGE share (fit on every DL with sampled snaps), z units
fa = dl.dropna(subset=["arc_random", "combine_weight_z", "edge_share"])
Aa = np.c_[np.ones(len(fa)), fa[["combine_weight_z", "edge_share"]].values]
ba_ = np.linalg.lstsq(Aa, fa.arc_random.values, rcond=None)[0]
sda = np.std(fa.arc_random.values - Aa @ ba_)
bd["arc_res"] = (bd.arc_random - np.c_[np.ones(len(bd)), bd[["combine_weight_z", "edge_share"]].values] @ ba_) / sda
bd["arcflag"] = np.where(bd.arc_res.notna() & (bd.shr.abs() >= 0.4) & (np.sign(bd.shr) != np.sign(bd.arc_res))
                         & (bd.arc_res.abs() >= 0.5), "arc", "")
put("board_n_arc", int(bd.arc_res.notna().sum())); put("board_n_arcflag", int((bd.arcflag == "arc").sum()))
put("board_rho_shr_arc", stats.spearmanr(bd.shr, bd.arc_res, nan_policy="omit")[0])
bd["flag"] = np.where(bd.pb.notna() & (bd.shr.abs() >= 0.4) & (np.sign(bd.shr) != np.sign(bd.film)), "disagree", "")
bd = bd.sort_values("shr", ascending=False).reset_index(drop=True)
put("board_n_nfl", int(bd.pb.notna().sum()))
put("board_n_flag", int((bd.flag == "disagree").sum()))
cardp = bd.dropna(subset=["pb"])
closer = (cardp.pa - cardp.press_rate).abs() < (cardp.pb - cardp.press_rate).abs()
put("card_closer", int(closer.sum())); put("card_n", len(cardp))
put("card_binom_p", stats.binomtest(int(closer.sum()), len(cardp), 0.5, alternative="greater").pvalue, 2)
put("card_rho_base", stats.spearmanr(cardp.pb, cardp.y)[0]); put("card_rho_aug", stats.spearmanr(cardp.pa, cardp.y)[0])
TABLES["board_2025"] = bd[["display_name", "role", "pick", "combine_weight", "hoop_g", "hoop_k", "hoop_z", "shr", "shr_lo",
                           "shr_hi", "shr_g", "rush_snaps", "press_rate", "pb", "pa", "film", "flag", "arc_random", "arc_res", "arcflag"]].round(4).to_dict("records")
# shrinkage arithmetic for three named prospects (top, bottom, and a disagreement case)
ex = [bd.iloc[0], bd.iloc[-1]]
dis = bd[bd.flag == "disagree"]
if len(dis):
    ex.append(dis.iloc[0])
for i, r_ in enumerate(ex):
    put(f"ex{i}_g", r_.hoop_g); put(f"ex{i}_z", r_.hoop_z, 1); put(f"ex{i}_shr", r_.shr); put(f"ex{i}_shrg", r_.shr_g, 3)
    put(f"ex{i}_lo", r_.shr_lo); put(f"ex{i}_hi", r_.shr_hi); put(f"ex{i}_wt", int(r_.combine_weight))
    put(f"ex{i}_rank", int(r_.name) + 1)
    TABLES[f"example{i}"] = r_.display_name
    if pd.notna(r_.pb):
        put(f"ex{i}_pb", 100 * r_.pb, 1); put(f"ex{i}_pr", 100 * r_.press_rate, 1); put(f"ex{i}_snaps", int(r_.rush_snaps))

# ============================================================================= FIGURES
ROLEC = {"EDGE": OI["orange"], "interior": OI["blue"]}


def name_short(n):
    return n.split(" ")[0][0] + ". " + " ".join(n.split(" ")[1:])


def repel(ax, xs, ys, labels, fs=10, iters=300):
    """Tiny label de-overlap: push text boxes apart in display space, then draw leader lines."""
    fig = ax.figure
    fig.canvas.draw()
    T = [ax.text(x, y, l, fontsize=fs, ha="left", va="bottom") for x, y, l in zip(xs, ys, labels)]
    P = ax.transData.transform(np.c_[xs, ys]) + np.array([6, 6])
    r = fig.canvas.get_renderer()
    W = np.array([[t.get_window_extent(r).width, t.get_window_extent(r).height] for t in T])
    pts = ax.transData.transform(np.c_[xs, ys])
    for _ in range(iters):
        moved = False
        for i in range(len(T)):
            for j in range(len(T)):
                if i == j:
                    continue
                dx = (P[i, 0] + W[i, 0] / 2) - (P[j, 0] + W[j, 0] / 2)
                dy = (P[i, 1] + W[i, 1] / 2) - (P[j, 1] + W[j, 1] / 2)
                ox = (W[i, 0] + W[j, 0]) / 2 - abs(dx)
                oy = (W[i, 1] + W[j, 1]) / 2 - abs(dy)
                if ox > 0 and oy > 0:
                    P[i, 1] += np.sign(dy if dy != 0 else 1) * min(oy, 3) * 0.6
                    moved = True
            for q in pts:  # also keep off the data points
                dx = q[0] - np.clip(q[0], P[i, 0], P[i, 0] + W[i, 0])
                dy = q[1] - np.clip(q[1], P[i, 1], P[i, 1] + W[i, 1])
                if abs(dx) < 3 and abs(dy) < 3:
                    P[i, 1] += 2; moved = True
        if not moved:
            break
    inv = ax.transData.inverted()
    for t, p, x, y in zip(T, P, xs, ys):
        xy = inv.transform(p)
        t.set_position(xy)
        ax.plot([x, xy[0]], [y, xy[1]], color=OI["grey"], lw=0.6, zorder=1)


# ---- Fig 1: drill bend vs game bend, 3 highest and 3 lowest Hoop G among DL with 250+ NFL rushes
hero_pool = co.dropna(subset=["hoop_z", "arc_random"])
hero_pool = hero_pool[hero_pool.rush_snaps >= 250].sort_values("hoop_z", ascending=False)
hero = pd.concat([hero_pool.head(3), hero_pool.tail(3)])
put("hero_pool_n", len(hero_pool))
arc_keys = arcp[arcp.kind == "random"][K]
norm = plt.Normalize(0, 0.9)
cmap = plt.get_cmap("cividis")
fig = plt.figure(figsize=(FIGW, 11.0))
press_max = 16
for idx, r_ in enumerate(hero.itertuples()):
    rr, cc = idx // 3, idx % 3
    x0, yt = 0.03 + cc * 0.305, 0.88 - rr * 0.44
    pk = "UDFA" if r_.pick == UDFA_PICK else f"#{int(r_.pick)}"
    fig.text(x0, yt, f"{name_short(r_.display_name)}", fontsize=12.5, weight="bold", va="top")
    fig.text(x0, yt - 0.024, f"{r_.role}, pick {pk[1:] if pk != 'UDFA' else 'UDFA'}, {int(r_.combine_weight)} lb",
             fontsize=10.5, va="top", color="#333333")
    fig.text(x0, yt - 0.052, f"Hoop G {r_.hoop_g:.2f} g", fontsize=11, va="top", weight="bold", color=OI["red"])
    ax = fig.add_axes([x0, yt - 0.315, 0.095, 0.24])
    tr = traces[r_.hoop_eid]
    xs_, ys_ = tr["_x"] - tr["_x"].mean(), tr["_y"] - tr["_y"].mean()
    if np.ptp(ys_) > np.ptp(xs_):
        xs_, ys_ = ys_, xs_
    xs_, ys_ = ys_, xs_    # draw the figure-eight upright so it fills a portrait cell
    pts = np.c_[xs_, ys_].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=norm, linewidth=3.4)
    lc.set_array(tr["_alat"][:-1]); ax.add_collection(lc)
    k = tr["_k"]
    ax.scatter(xs_[max(k - 2, 0):k + 3], ys_[max(k - 2, 0):k + 3], s=40, facecolor="none", edgecolor=OI["red"], lw=1.8, zorder=3)
    pad = 0.4
    ax.set_xlim(xs_.min() - pad, xs_.max() + pad); ax.set_ylim(ys_.min() - pad, ys_.max() + pad)
    ax.set_aspect("equal", adjustable="datalim"); ax.axis("off")
    gx = fig.add_axes([x0 + 0.12, yt - 0.315, 0.15, 0.24])
    ks = arc_keys[arc_keys.nfl_id == r_.nfl_id]
    for kk in ks.itertuples(index=False):
        lat, fwd, al = arc_tr[tuple(kk)]
        P_ = np.c_[lat, fwd].reshape(-1, 1, 2)
        l2 = LineCollection(np.concatenate([P_[:-1], P_[1:]], axis=1), cmap=cmap, norm=norm, linewidth=1.7, alpha=0.9)
        l2.set_array(al[:-1]); gx.add_collection(l2)
    gx.scatter([0], [0], marker="^", color=OI["black"], s=34, zorder=4)
    gx.set_xlim(-2, 7); gx.set_ylim(-1, 9)
    gx.set_xticks([0, 3, 6]); gx.set_yticks([0, 4, 8]); gx.tick_params(labelsize=9.5)
    gx.set_xlabel("yd sideways (upfield ↑)", fontsize=10, labelpad=1)
    fig.text(x0 + 0.12, yt - 0.052, f"NFL Arc G {r_.arc_random:.2f} g", fontsize=11, va="top",
             weight="bold", color=OI["blue"])
    px = fig.add_axes([x0, yt - 0.39, 0.27, 0.018])
    px.axvline(100 * co.press_rate.median(), color=OI["grey"], lw=1, ls="--")
    px.scatter([100 * r_.press_rate], [0], s=20 + r_.rush_snaps / 8, color=ROLEC[r_.role], edgecolor="k", zorder=3)
    right_ = r_.press_rate < 0.095
    px.text(100 * r_.press_rate + (0.7 if right_ else -0.7), 0.3, f"{100 * r_.press_rate:.1f}% on {int(r_.rush_snaps)} rushes",
            fontsize=9.5, va="center", ha="left" if right_ else "right")
    px.set_xlim(0, press_max); px.set_ylim(-1, 1.5); px.set_yticks([])
    px.spines["left"].set_visible(False); px.tick_params(labelsize=9.5)
    px.set_xlabel("NFL pressure rate (%)", fontsize=10, labelpad=1)
fig.text(0.01, 0.982, "Drill bend and game bend for the 3 highest and 3 lowest Hoop G (DL with 250+ NFL rushes)",
         fontsize=14.5, weight="bold", va="top")
fig.text(0.01, 0.95, "Chosen by rule; top row = highest. Each card: one Combine hoop run and his 20 random NFL pass rushes "
         "(snap at the triangle, path to\nthrow or sack, upfield is up), both coloured by sideways g. Red circles = the 0.5 s "
         "that sets Hoop G. Strip: career pressure rate, dashed = cohort median.",
         fontsize=10.5, va="top")
cax = fig.add_axes([0.955, 0.3, 0.012, 0.4])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[0, 0.3, 0.6, 0.9])
cb.ax.tick_params(labelsize=10); cb.set_label("sideways g", fontsize=10.5)
save(fig, "fig1_drill_vs_game.png")
for i, r_ in enumerate(hero.itertuples()):
    put(f"hero{i}_hoop", r_.hoop_g); put(f"hero{i}_press", 100 * r_.press_rate, 1); put(f"hero{i}_snaps", int(r_.rush_snaps))
    put(f"hero{i}_arc", r_.arc_random)
TABLES["hero"] = hero[["display_name", "role", "pick", "hoop_g", "arc_random", "press_rate", "rush_snaps"]].to_dict("records")

# ---- Fig 2: reliability strip + NGS validation inset + signal map sorted by effect
fig = plt.figure(figsize=(FIGW, 7.0))
gsl = fig.add_gridspec(2, 1, left=0.12, right=0.31, top=0.86, bottom=0.2, hspace=0.75)
gsr = fig.add_gridspec(1, 2, left=0.49, right=0.985, top=0.86, bottom=0.2, wspace=0.06)
a0 = fig.add_subplot(gsl[0, 0])
lad = [("Rush drill", rel_d1_player, None), ("Hoop G", rel_hoop, (float(NUM["rel_hoop_sb_lo"]), float(NUM["rel_hoop_sb_hi"]))),
       ("NFL Arc G", ARC["random"], None), ("NFL pressure", REL_OUT["press_eb"], None), ("NFL get-off", REL_OUT["quick"], None)]
yy = np.arange(len(lad))[::-1]
for yv, (lab, rv, ci_) in zip(yy, lad):
    colr = OI["green"] if yv >= 3 else OI["blue"]
    if ci_:
        a0.plot(ci_, [yv, yv], color=colr, lw=2)
    a0.scatter([rv], [yv], color=colr, s=60, zorder=3)
    a0.text((ci_[1] if ci_ else rv) + 0.04, yv, f"{rv:.2f}", va="center", fontsize=10)
a0.axvline(0.7, color=OI["red"], ls="--", lw=1)
a0.text(0.73, 4.75, "0.7 = ranks players", color=OI["red"], fontsize=10, va="top", ha="left")
a0.set_yticks(yy); a0.set_yticklabels([l for l, *_ in lad], fontsize=10)
a0.set_xlim(0, 1.15); a0.set_ylim(-0.5, 4.9); a0.tick_params(labelsize=10)
a0.set_title("Reliability per player", fontsize=11.5)
a00 = fig.add_subplot(gsl[1, 0])
hb = a00.hexbin(gd.g_d1, gd.player_get_off, gridsize=22, cmap="cividis", mincnt=1, extent=(0, 4, 0.3, 2.0))
a00.set_xlabel("tracked 1st-second yd", fontsize=10); a00.set_ylabel("NGS get-off (s)", fontsize=10)
a00.tick_params(labelsize=10)
a00.set_title(f"NGS get-off vs burst,\nρ {NUM['rho_play_d1_getoff']} ({len(gd):,} snaps)", fontsize=11)
order = sig[sig.outcome == "press_eb"].sort_values("rho").col.tolist()
labs = {m_[0]: m_[1] for m_ in MEAS}
yy = np.arange(len(order))
axes = [fig.add_subplot(gsr[0, 0]), fig.add_subplot(gsr[0, 1])]
for ax, (oc, olab) in zip(axes, OUTC):
    d_ = sig[sig.outcome == oc].set_index("col").loc[order]
    for yv, (c, r_) in zip(yy, d_.iterrows()):
        ax.fill_betweenx([yv - 0.42, yv + 0.42], -r_.mde, r_.mde, color="#E3E3E3", lw=0)
        colr = OI["green"] if r_.kind == "tracking" else OI["black"]
        ax.plot([r_.lo, r_.hi], [yv, yv], color=colr, lw=1.8)
        ax.scatter([r_.rho], [yv], color=colr, s=50, zorder=3)
    for (x, y), pv in prim.items():
        if y == oc:
            yv = order.index(x) - 0.3
            ax.plot([pv["lo"], pv["hi"]], [yv, yv], color=OI["purple"], lw=1.4)
            ax.scatter([pv["r"]], [yv], marker="D", facecolor="white", edgecolor=OI["purple"], s=50, zorder=4, lw=1.6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-0.65, 1.0); ax.set_ylim(-0.7, len(order) + 0.45)
    ax.set_title(olab, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.text(-0.62, len(order) + 0.4, "grey = too small to detect", ha="left", fontsize=9.5, color="#555555", va="top",
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
axes[0].set_yticks(yy); axes[0].set_yticklabels([labs[c] for c in order], fontsize=10.5)
axes[1].set_yticks(yy); axes[1].set_yticklabels([])
axes[0].scatter([], [], color=OI["black"], label="official / draft, raw ρ (95% CI)")
axes[0].scatter([], [], color=OI["green"], label="tracking, raw ρ (95% CI)")
axes[0].scatter([], [], marker="D", facecolor="white", edgecolor=OI["purple"], label="tracking after size, 10-yd, EDGE share, pick")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.text(0.74, 0.11, f"Spearman ρ, + = better Combine number goes with\nbetter NFL result (n ≤ {len(co)} DL with 100+ rushes)",
         ha="center", fontsize=10.5)
fig.suptitle("Times carry most of the DL signal; bend helps pressure, not clearly get-off", fontsize=14, weight="bold",
             x=0.01, ha="left", y=0.975)
save(fig, "fig2_signal_map.png")

# ---- Fig 3: Hoop G vs pressure residual, per held-out class, selection-honest null
fig = plt.figure(figsize=(FIGW, 9.2))
gs3 = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.25, 1], left=0.08, right=0.985, top=0.9,
                       bottom=0.07, hspace=0.5, wspace=0.25)
a1 = fig.add_subplot(gs3[0, 0])
d_ = impute(dH, dH)[0]
A = np.c_[np.ones(len(d_)), d_[SMALL].values]
d_["res"] = 100 * (d_.press_eb - A @ np.linalg.lstsq(A, d_.press_eb.values, rcond=None)[0])
for role, gg_ in d_.groupby("role"):
    a1.scatter(gg_.hoop_z, gg_.res, color=ROLEC[role], s=36, alpha=0.85, label=f"{role} (n={len(gg_)})",
               marker="o" if role == "EDGE" else "s")
xx = np.linspace(-2.5, 2.5, 50)
bs = np.array([np.polyval(np.polyfit(d_.hoop_z.values[i], d_.res.values[i], 1), xx)
               for i in (RNG.integers(0, len(d_), len(d_)) for _ in range(NBOOT))])
a1.fill_between(xx, *np.percentile(bs, [2.5, 97.5], axis=0), color=OI["grey"], alpha=0.25, lw=0)
a1.plot(xx, np.polyval(np.polyfit(d_.hoop_z, d_.res, 1), xx), color="k", lw=1.6)
lab_ids = set(hero.nfl_id) | set(d_.nlargest(2, "res").nfl_id) | set(d_.nsmallest(2, "res").nfl_id)
ld = d_[d_.nfl_id.isin(lab_ids)]
a1.set_xlim(-2.8, 3.0)
a1.set_ylim(d_.res.min() - 1, d_.res.max() + 2.5)
repel(a1, ld.hoop_z.values, ld.res.values, [name_short(n) for n in ld.display_name], fs=10)
a1.axhline(0, color="k", lw=0.6)
a1.set_xlabel("Hoop G, z within Combine year"); a1.set_ylabel("pressure rate above baseline (pp)")
a1.set_title(f"Effect after size, 10-yd, EDGE share and pick: ρ = {NUM['pr_hoop_z_press_eb']} (n = {H['n']})", fontsize=12)
a1.legend(frameon=False, fontsize=10, loc="lower right")
a2 = fig.add_subplot(gs3[0, 1])
for j, c in enumerate(sorted(fold)):
    f = fold[c]
    a2.plot(f["rb_ci"], [j + 0.13] * 2, color=OI["grey"], lw=2); a2.scatter([f["rb"]], [j + 0.13], color="white", edgecolor=OI["grey"], s=60, zorder=3, lw=1.6)
    a2.plot(f["ra_ci"], [j - 0.13] * 2, color=OI["green"], lw=2); a2.scatter([f["ra"]], [j - 0.13], color=OI["green"], s=60, zorder=3)
a2.scatter([], [], color="white", edgecolor=OI["grey"], label="baseline only"); a2.scatter([], [], color=OI["green"], label="+ Hoop G")
a2.axvline(0, color="k", lw=0.8)
a2.set_yticks(range(len(fold))); a2.set_yticklabels([f"{c} held out\n(n={fold[c]['n']})" for c in sorted(fold)], fontsize=10.5)
a2.set_xlabel("rank ρ, forecast vs actual pressure\n(80% bootstrap interval)")
a2.set_xlim(-0.5, 1.0); a2.tick_params(labelsize=10)
a2.legend(frameon=False, fontsize=10, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
a2.set_title(" ", fontsize=6)
a3 = fig.add_subplot(gs3[1, :])
bins = np.linspace(0, max(0.6, null_p.max() + 0.02), 40)
a3.hist(null_p, bins=bins, color=OI["sky"], alpha=0.8, label=f"best of {len(SCREEN)} metrics, outcome shuffled within class")
a3.hist(np.array(null_hoop), bins=bins, histtype="step", color=OI["blue"], lw=1.6, label="Hoop G alone, outcome shuffled")
a3.axvline(max_obs_p, color=OI["red"], lw=2)
a3.text(max_obs_p - 0.008, a3.get_ylim()[1] * 0.97, f"observed best = Hoop G {abs(obs_p['hoop_z']):.2f}\n"
        f"p = {NUM['perm_best7']} after choosing the best of {len(SCREEN)}", color=OI["red"], fontsize=10.5, va="top", ha="right")
a3.set_xlabel("largest |effect after baseline| among the screened drill metrics (pressure)")
a3.set_ylabel("shuffles"); a3.tick_params(labelsize=10)
a3.legend(frameon=False, fontsize=10, loc="upper right")
a3.set_title("Selection check: how often chance produces a lead this large when we take the best drill metric", fontsize=12)
fig.suptitle("Bend beyond the baseline: a lead, borderline once we account for picking the best metric",
             fontsize=14, weight="bold", x=0.01, ha="left")
save(fig, "fig3_hoop_pressure.png")

# ---- Fig 4: 2025 class board, every hoop runner, shrunk Hoop G + rookie check
fig = plt.figure(figsize=(FIGW, max(7.5, 0.27 * len(bd) + 2.2)))
g4 = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], left=0.33, right=0.985, top=0.92, bottom=0.075, wspace=0.08)
b1, b2 = fig.add_subplot(g4[0]), fig.add_subplot(g4[1], sharey=None)
yy = np.arange(len(bd))[::-1]
for yv, r_ in zip(yy, bd.itertuples()):
    c_ = ROLEC[r_.role]
    b1.plot([r_.shr_lo, r_.shr_hi], [yv, yv], color=c_, lw=2.2, alpha=0.6)
    b1.scatter([r_.hoop_z], [yv], marker="|", color=OI["grey"], s=60, zorder=2)
    b1.scatter([r_.shr], [yv], color=c_, s=46, zorder=3, marker="o" if r_.role == "EDGE" else "s")
    if pd.notna(r_.pb):
        b2.plot([100 * r_.pr_lo, 100 * r_.pr_hi], [yv, yv], color=OI["grey"], lw=2, alpha=0.6)
        b2.scatter([100 * r_.pb], [yv], color="white", edgecolor=OI["black"], s=40, zorder=3)
        b2.annotate("", xy=(100 * r_.pa, yv), xytext=(100 * r_.pb, yv), arrowprops=dict(arrowstyle="->", color=OI["green"], lw=1.4))
        b2.scatter([100 * r_.press_rate], [yv], marker="x", color=OI["red"], s=46, zorder=4)
    ftxt = " + ".join(t_ for t_, f_ in [("rookie", r_.flag == "disagree"), ("NFL arc", r_.arcflag == "arc")] if f_)
    if ftxt:
        b2.text(21.8, yv, ftxt + " ≠ drill", fontsize=10, va="center", ha="right", color=OI["purple"], weight="bold")
b1.axvline(0, color="k", lw=0.8)
labels = []
for r_ in bd.itertuples():
    pk = "UDFA" if r_.pick == UDFA_PICK else f"#{int(r_.pick)}"
    labels.append(f"{r_.display_name}  {r_.role[:3]}, {int(r_.combine_weight)} lb, {pk}")
b1.set_yticks(yy); b1.set_yticklabels(labels, fontsize=10)
b2.set_yticks(yy); b2.set_yticklabels([])
b1.set_ylim(-0.8, len(bd) - 0.2); b2.set_ylim(-0.8, len(bd) - 0.2)
b1.set_xlabel("shrunk Hoop G, z within class (bar = 80% interval;\ngrey tick = raw run)", fontsize=10.5)
b2.set_xlabel("NFL 2025 pressure rate (%)", fontsize=10.5)
b1.set_xlim(-2.6, 2.6); b2.set_xlim(0, 22); b2.set_xticks([0, 5, 10, 15]); b1.tick_params(labelsize=10); b2.tick_params(labelsize=10)
b1.set_title("Drill: bend board", fontsize=12)
b2.set_title("Rookie check (100+ rushes) and flags", fontsize=12)
b2.scatter([], [], color="white", edgecolor="k", label="no-tracking forecast (fit on 2023–24)")
b2.plot([], [], color=OI["green"], label="→ + Hoop G")
b2.scatter([], [], marker="x", color=OI["red"], label="actual (grey = 80% interval)")
fig.legend(*b2.get_legend_handles_labels(), frameon=False, fontsize=10, loc="lower center", bbox_to_anchor=(0.6, 0.0), ncol=3)
fig.suptitle(f"2025 DL class: all {len(bd)} hoop runners ranked by shrunk Hoop G.\nBend moves {NUM['card_closer']} of "
             f"{NUM['card_n']} rookie forecasts closer to the actual rate (binomial p = {NUM['card_binom_p']})", fontsize=13, weight="bold", x=0.01, ha="left")
save(fig, "fig4_board_2025.png")


# ---- Fig 5: does drill bend show up as game bend?
fig = plt.figure(figsize=(FIGW, 5.6))
g5 = fig.add_gridspec(1, 3, left=0.075, right=0.985, top=0.8, bottom=0.14, wspace=0.36)
c1, c2, c3 = (fig.add_subplot(g5[i]) for i in range(3))
for role, gg_ in dA.groupby("role"):
    c1.scatter(gg_.hoop_z, gg_.arc_random, color=ROLEC[role], s=26, alpha=0.85, marker="o" if role == "EDGE" else "s",
               label=f"{role} (n={len(gg_)})")
c1.set_xlabel("Hoop G, z within Combine year"); c1.set_ylabel("NFL Arc G (g), 20 random rushes")
c1.set_title(f"Raw: ρ = {NUM['rho_hoop_arc_raw']} (n = {NUM['n_hoop_arc']})", fontsize=12)
c1.legend(frameon=False, fontsize=10, loc="upper left")
Ad = np.c_[np.ones(len(dA)), dA[ARCB].values]
resx = dA.hoop_z.values - Ad @ np.linalg.lstsq(Ad, dA.hoop_z.values, rcond=None)[0]
resy = dA.arc_random.values - Ad @ np.linalg.lstsq(Ad, dA.arc_random.values, rcond=None)[0]
for role in ["EDGE", "interior"]:
    mk_ = (dA.role == role).values
    c2.scatter(resx[mk_], resy[mk_], color=ROLEC[role], s=26, alpha=0.85, marker="o" if role == "EDGE" else "s")
xx = np.linspace(resx.min(), resx.max(), 30)
c2.plot(xx, np.polyval(np.polyfit(resx, resy, 1), xx), color="k", lw=1.4)
c2.axhline(0, color="k", lw=0.6)
c2.set_xlabel("Hoop G beyond weight, EDGE share"); c2.set_ylabel("NFL Arc G beyond size, alignment (g)")
c2.set_title(f"After size and alignment: ρ = {NUM['pr_hoop_arc']}\n[{NUM['pr_hoop_arc_lo']}, {NUM['pr_hoop_arc_hi']}]", fontsize=12)
bins = np.linspace(-2.5, 3.5, 25)
c3.hist(qa[qa.press == 0].gz, bins=bins, density=True, color=OI["grey"], alpha=0.55, label=f"no pressure ({int((qa.press == 0).sum()):,})")
c3.hist(qa[qa.press == 1].gz, bins=bins, density=True, histtype="step", lw=2.2, color=OI["red"], label=f"pressure or sack ({int(qa.press.sum())})")
c3.set_xlabel("snap Arc G, z within player"); c3.set_ylabel("share of snaps (density)")
c3.set_title(f"Within player: pressure odds ×{NUM['within_or']}\nper sd of snap arc [{NUM['within_or_lo']}, {NUM['within_or_hi']}]", fontsize=12)
c3.set_ylim(0, 0.68)
c3.legend(frameon=False, fontsize=10, loc="upper right")
fig.suptitle("Game arc goes with pressure on the snap; drill bend matches game arc only through size and alignment",
             fontsize=13.5, weight="bold", x=0.01, ha="left", y=0.975)
save(fig, "fig5_arc_chain.png")

# ----------------------------------------------------------------------------- write numbers
json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1, sort_keys=True)
json.dump(TABLES, open(OUT / "tables.json", "w"), indent=1, default=str)
print(json.dumps({k: NUM[k] for k in sorted(NUM) if not k.startswith(("rho_", "n_", "hero"))}, indent=0))
print(json.dumps({k: TABLES[k] for k in ["screen", "robustness", "nested_selected"]}, indent=0, default=str))
