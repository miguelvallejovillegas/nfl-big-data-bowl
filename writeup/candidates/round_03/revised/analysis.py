"""Hoop G (Run-the-Hoop bend) vs NFL pressure rate for DL/EDGE: a lead, tested honestly.

Run from the repo root:
    uv run python writeup/candidates/round_03/revised/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json, tables.json, board_2025.csv, board_table.md and
figures/*.png next to this file. Every decimal quoted in writeup.md is produced here.

Design (round 3, revised after judging):
  * Main baseline = log pick + weight + NGS athleticism score + NFL EDGE share. Every covariate is observed for every
    cohort player, so the headline needs no imputation. Variants: no athletic score; + 10-yd split by multiple
    imputation (M=20, stochastic regression, never uses the outcome); the round-3 single fill; and the 49 players who
    ran the 40 (main, + official 10-yd, + official 40) with baseline-alone and baseline+Hoop G R2 on identical rows.
  * Selection: 7 drill metrics x 2 outcomes (pressure, get-off) = 14 screened tests. Headline selection p is the
    best-of-14 within-class permutation; a nested leave-one-class-out loop re-selects among all 14 on training classes.
    Empirical-Bayes shrinkage over the 14 gives the selection-corrected effect. Train-on-one-class / predict-two check.
  * Reliability: disattenuated effect with a joint bootstrap (split-half and 10 repeat-hoop players); board ordering
    stability under reliability resampling and true-score noise.
  * Football context: EDGE vs interior split; snap-level odds on EDGE-aligned vs interior snaps and true passing downs
    (random sampled snaps only); survivorship check for players under 100 rushes.
  * NFL Arc G: the same sideways-g formula on 20 random NFL rushes per player (self-referenced, no ball/opponents).
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
pinfo = g.drop_duplicates(K).set_index(K)[["play_direction", "sample_kind", "time_to_pressure", "sack", "down",
                                           "yards_to_go", "lined_up_position"]]
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
                         kind=info.sample_kind, down=info.down, ytg=info.yards_to_go, align=info.lined_up_position,
                         press=int(pd.notna(info.time_to_pressure) or info.sack == 1)))
    # rush path in a defender frame: forward = toward the offense, mirrored so the path ends to the right
    fwd = -(xs - xs[0]) if info.play_direction == "right" else (xs - xs[0])
    lat = ys - ys[0]
    ii = np.flatnonzero(inw)
    f0 = ii[0]
    fwd, lat = fwd - fwd[f0], lat - lat[f0]
    if lat[ii[-1]] < 0:
        lat = -lat
    kmax = int(np.nanargmax(roll.values)) - f0          # centre of the 0.5 s window that sets Arc G
    arc_tr[key] = (lat[ii], fwd[ii], al[ii] / YD_PER_G, kmax)
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


def impute_mi(tr: pd.DataFrame, te: pd.DataFrame, rng, col="ten_yd_split_z"):
    """One proper stochastic draw (parameter + residual noise) for missing 10-yd splits. Imputation model uses
    weight, EDGE share, NGS athleticism score and Hoop G, fit on training rows; never the outcome."""
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


def loco(df, base, extra, y, fit=ols_pred, imp=None, train_one=False, rng=None):
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
                t1, t2 = impute_mi(tr, te, rng if rng is not None else RNG)
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


def boot_loco(df, base, extra, y, nb=1000, **kw):
    groups = {c: d for c, d in df.groupby("draft_year")}
    out = []
    for _ in range(nb):
        d = pd.concat([gg.iloc[RNG.integers(0, len(gg), len(gg))] for gg in groups.values()])
        try:
            out.append(loco(d, base, extra, y, **kw)["delta"])
        except Exception:
            pass
    return np.percentile(out, [2.5, 97.5])


dH = co.dropna(subset=["hoop_z", "press_eb"]).copy()
LH = loco(dH, MAIN, ["hoop_z"], "press_eb")
put("lo_n", LH["n"]); put("r2b", LH["r2_base"]); put("r2a", LH["r2_aug"]); put("dr2", LH["delta"], 3)
lo_, hi_ = boot_loco(dH, MAIN, ["hoop_z"], "press_eb")
put("dr2_lo", lo_, 3); put("dr2_hi", hi_, 3)
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


def mi_partial(df, base):
    """Rubin-style: average Fisher z of the partial rho over M_IMP imputations; interval from pooled bootstraps."""
    zs, bs = [], []
    for _ in range(M_IMP):
        d1, _ = impute_mi(df, df.iloc[:0], MI_RNG)
        r, b, n = partial_draws(d1, "hoop_z", "press_eb", base, nb=max(100, NBOOT // 10))
        zs.append(np.arctanh(r)); bs.append(b)
    bs = np.concatenate(bs)
    return float(np.tanh(np.mean(zs))), *np.percentile(bs, [2.5, 97.5]), n


def add_var(tag, label, df, base, imp=None, fit=ols_pred, nb=1000):
    d = df.dropna(subset=["hoop_z", "press_eb"]).copy()
    if imp is None:
        d = d.dropna(subset=base)
    if imp == "mi":
        r, lo, hi, n = mi_partial(d, base)
    elif imp == "single":
        r, lo, hi, n, _ = partial_sp(impute(d, d)[0], "hoop_z", "press_eb", base)
    else:
        r, lo, hi, n, _ = partial_sp(d, "hoop_z", "press_eb", base)
    L = loco(d, base, ["hoop_z"], "press_eb", fit=fit, imp=imp, rng=MI_RNG)
    ci = boot_loco(d, base, ["hoop_z"], "press_eb", nb=nb, fit=fit, imp=imp, rng=MI_RNG) if nb else (np.nan, np.nan)
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
add_var("notime", "No athletic score at all: pick, weight, EDGE share", co, NOTIME)
add_var("mi", f"Main + 10-yd split, multiple imputation (M={M_IMP})", co, MAIN + ["ten_yd_split_z"], imp="mi", nb=300)
add_var("single", "Round-3 baseline: pick, weight, 10-yd (single fill), EDGE", co,
        ["log_pick", "combine_weight_z", "ten_yd_split_z", "edge_share"], imp="single")
add_var("cc_main", "Main, only the players who ran the 40", cc, MAIN)
add_var("cc_10", "Main + official 10-yd, only the players who ran the 40", cc, MAIN + ["ten_yd_split_z"])
add_var("cc_40", "Main + official 40, only the players who ran the 40", cc, MAIN + ["forty_z"])
add_var("ridge", "Main, ridge", co, MAIN, fit=ridge_pred, nb=300)
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

# ----------------------------------------------------------------------------- board: 2025 class, every hoop runner
lam1 = rel_hoop
bd = dl[(dl.draft_year == 2025) & dl.hoop_z.notna()].copy()
bd["lam"] = bd.hoop_k * lam1 / (1 + (bd.hoop_k - 1) * lam1)
bd["shr"] = bd.lam * bd.hoop_z
bd["shr_sd"] = np.sqrt(bd.lam * (1 - bd.lam))
bd["shr_lo"], bd["shr_hi"] = bd.shr - 1.2816 * bd.shr_sd, bd.shr + 1.2816 * bd.shr_sd
m25, s25 = hstat.loc[2025, "mean"], hstat.loc[2025, "std"]
bd["shr_g"] = m25 + bd.shr * s25
put("board_n", len(bd)); put("board_mean_g", m25); put("board_sd_g", s25, 3)
put("board_lam1", lam1); put("board_band", 1.2816 * np.sqrt(lam1 * (1 - lam1)))
p25 = LH["pred"][LH["pred"].cl == 2025][["nfl_id", "pb", "pa", "y"]]
bd = bd.merge(p25, on="nfl_id", how="left")
bd["pr_lo"] = stats.beta.ppf(0.1, bd.pressures + 0.5, bd.rush_snaps - bd.pressures + 0.5)
bd["pr_hi"] = stats.beta.ppf(0.9, bd.pressures + 0.5, bd.rush_snaps - bd.pressures + 0.5)
bd["film"] = bd.press_rate - bd.pb
bd["flag"] = np.where(bd.pb.notna() & (bd.shr.abs() >= 0.4) & (np.sign(bd.shr) != np.sign(bd.film)),
                      np.where(bd.shr > 0, "drill too high", "drill too low"), "")
bd = bd.sort_values("shr", ascending=False).reset_index(drop=True)
bd["rank"] = np.arange(1, len(bd) + 1)
put("board_n_nfl", int(bd.pb.notna().sum()))
put("board_n_flag", int((bd.flag != "").sum()))
cardp = bd.dropna(subset=["pb"])
closer = (cardp.pa - cardp.press_rate).abs() < (cardp.pb - cardp.press_rate).abs()
put("card_closer", int(closer.sum())); put("card_n", len(cardp))
put("card_binom_p", stats.binomtest(int(closer.sum()), len(cardp), 0.5, alternative="greater").pvalue, 2)
# ordering stability: (a) reliability resampled from the split-half bootstrap only; (b) plus true-score noise
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
put("board_rs_rel_min", np.min(rs_rel)); put("board_rs_rel_med", np.median(rs_rel))
put("board_rs_true", np.median(rs_true)); put("board_rs_true_lo", np.percentile(rs_true, 2.5))
put("board_rs_true_hi", np.percentile(rs_true, 97.5)); put("board_top5_keep", 100 * np.mean(top5), 0)
bd[["rank", "display_name", "role", "pick", "combine_weight", "hoop_k", "hoop_g", "hoop_z", "shr", "shr_lo", "shr_hi",
    "shr_g", "rush_snaps", "press_rate", "pb", "pa", "film", "flag"]].round(4).to_csv(OUT / "board_2025.csv", index=False)
TABLES["board_2025"] = bd[["display_name", "role", "pick", "combine_weight", "hoop_g", "hoop_k", "shr", "rush_snaps",
                           "press_rate", "pb", "pa", "flag"]].round(4).to_dict("records")


def pk_txt(p):
    return "UDFA" if p == UDFA_PICK else str(int(p))


# board table for the writeup: top 8 and bottom 4 (full board in board_2025.csv)
rows = []
for r_ in pd.concat([bd.head(8), bd.tail(4)]).itertuples():
    i = r_.rank
    put(f"bt{i}_shr", r_.shr); put(f"bt{i}_lo", r_.shr_lo); put(f"bt{i}_hi", r_.shr_hi)
    rook = f"{100 * r_.press_rate:.1f}% on {int(r_.rush_snaps)}" if r_.rush_snaps >= MIN_SNAPS else (
        f"{int(r_.rush_snaps)} rushes" if r_.rush_snaps > 0 else "no NFL rushes")
    if r_.rush_snaps >= MIN_SNAPS:
        put(f"bt{i}_pr", 100 * r_.press_rate, 1); put(f"bt{i}_pb", 100 * r_.pb, 1)
    sgn = lambda v: f"{v:+.2f}".replace("-", "−")
    rows.append(f"| {i} | {r_.display_name} | {'EDGE' if r_.role == 'EDGE' else 'Interior'} | {int(r_.combine_weight)} | "
                f"{pk_txt(r_.pick)} | {int(r_.hoop_k)} | {sgn(r_.shr)} [{sgn(r_.shr_lo)}, {sgn(r_.shr_hi)}] | {rook} |")
TABLES["board_md"] = "\n".join(rows)
(OUT / "board_table.md").write_text("\n".join(rows) + "\n")
# shrinkage arithmetic for named prospects
ex = {"top": bd.iloc[0], "bottom": bd.iloc[-1]}
for tag, r_ in ex.items():
    put(f"ex_{tag}_g", r_.hoop_g); put(f"ex_{tag}_z", r_.hoop_z, 1); put(f"ex_{tag}_shr", r_.shr)
    put(f"ex_{tag}_shrg", r_.shr_g, 3); put(f"ex_{tag}_lo", r_.shr_lo); put(f"ex_{tag}_hi", r_.shr_hi)
    put(f"ex_{tag}_wt", int(r_.combine_weight)); put(f"ex_{tag}_snaps", int(r_.rush_snaps))
    TABLES[f"example_{tag}"] = r_.display_name
    if pd.notna(r_.pb):
        put(f"ex_{tag}_pb", 100 * r_.pb, 1); put(f"ex_{tag}_pr", 100 * r_.press_rate, 1)
for r_ in bd[bd.flag != ""].itertuples():
    key = r_.display_name.split(" ")[-1].lower().replace("-", "")
    put(f"fl_{key}_pr", 100 * r_.press_rate, 1); put(f"fl_{key}_pb", 100 * r_.pb, 1); put(f"fl_{key}_shr", r_.shr)
    put(f"fl_{key}_snaps", int(r_.rush_snaps))
TABLES["flagged"] = bd[bd.flag != ""][["display_name", "flag", "shr", "press_rate", "pb", "rush_snaps"]].to_dict("records")

# ============================================================================= FIGURES
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


# ---- Fig 1: drill bend vs game bend, 3 highest and 3 lowest Hoop G among DL with 250+ NFL rushes
hero_pool = co.dropna(subset=["hoop_z", "arc_random"])
hero_pool = hero_pool[hero_pool.rush_snaps >= 250].sort_values("hoop_z", ascending=False)
hero = pd.concat([hero_pool.head(3).assign(grp="high"), hero_pool.tail(3).assign(grp="low")])
hero = hero.merge(LH["pred"][["nfl_id", "pb"]], on="nfl_id", how="left")
hero["lam"] = hero.hoop_k * rel_hoop / (1 + (hero.hoop_k - 1) * rel_hoop)
hero["shr"] = hero.lam * hero.hoop_z
put("hero_pool_n", len(hero_pool))


def rec(r_):
    a_, f_ = 100 * r_.press_rate, 100 * r_.pb
    vs = f"{a_:.1f}% pressure vs {f_:.1f}% forecast"
    if r_.grp == "high":
        if a_ >= f_:
            return f"Drill and Sundays agree ({vs}). Plus trait: give him passing downs."
        return f"Bend did not become pressure ({vs}). Check hands and finish on film."
    if a_ >= f_:
        return f"Wins without bend ({vs}). A low hoop alone should not drop him."
    return f"Low bend, low pressure ({vs}). Run defender, not a pass-rush bet."


arc_keys = arcp[arcp.kind == "random"][K]
norm = plt.Normalize(0, 0.9)
cmap = plt.get_cmap("cividis")
SPAN = 11.0
FH1 = 12.2
fig = plt.figure(figsize=(FIGW, FH1))
fig.text(0.01, 0.988, "High-bend and low-bend rushers look alike on Sundays", fontsize=16, weight="bold", va="top")
fig.text(0.01, 0.963, "Three highest and three lowest Hoop G among DL with 250+ NFL rushes (chosen by rule). Left square: his "
         "Combine hoop run.\nRight square: his 20 random NFL pass rushes from the snap (triangle, upfield is up) to the "
         "throw or sack. Same 11-yard scale in both\nsquares. Red ring / red dots = the half-second that sets each score.",
         fontsize=10.5, va="top")
cax = fig.add_axes([0.62, 0.872, 0.3, 0.011])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[0, 0.3, 0.6, 0.9])
cb.ax.tick_params(labelsize=10); cax.set_title("sideways g (colour of every path)", fontsize=10.5, loc="left", pad=3)
for idx, r_ in enumerate(hero.itertuples()):
    rr, cc_ = idx // 3, idx % 3
    x0, yt = 0.02 + cc_ * 0.33, 0.85 - rr * 0.305
    gcol = HIC if r_.grp == "high" else LOC
    if cc_ == 0:
        fig.text(0.01, yt + 0.012, "HIGHEST HOOP G" if rr == 0 else "LOWEST HOOP G", fontsize=11, weight="bold", color=gcol)
    fig.text(x0, yt - 0.01, r_.display_name, fontsize=12.5, weight="bold", va="top", color=gcol)
    fig.text(x0, yt - 0.031, f"{'EDGE' if r_.role == 'EDGE' else 'Interior'}, pick {pk_txt(r_.pick)}, "
             f"{int(r_.combine_weight)} lb, {int(r_.rush_snaps)} NFL rushes", fontsize=10, va="top", color="#333333")
    w_ax = 0.145
    h_ax = w_ax * FIGW / FH1
    ax = fig.add_axes([x0, yt - 0.068 - h_ax, w_ax, h_ax])
    tr = traces[r_.hoop_eid]
    xs_, ys_ = tr["_x"] - (tr["_x"].max() + tr["_x"].min()) / 2, tr["_y"] - (tr["_y"].max() + tr["_y"].min()) / 2
    if np.ptp(ys_) < np.ptp(xs_):
        xs_, ys_ = ys_, xs_          # long axis vertical
    pts = np.c_[xs_, ys_].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=norm, linewidth=3.0)
    lc.set_array(tr["_alat"][:-1]); ax.add_collection(lc)
    k = tr["_k"]
    ax.scatter([xs_[k]], [ys_[k]], s=150, facecolor="none", edgecolor=OI["red"], lw=2, zorder=3)
    ax.set_xlim(-SPAN / 2, SPAN / 2); ax.set_ylim(-SPAN / 2, SPAN / 2); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(True); sp_.set_color("#BBBBBB")
    ax.set_title(f"Hoop G {r_.hoop_g:.2f} g", fontsize=11, weight="bold", pad=3)
    gx = fig.add_axes([x0 + w_ax + 0.012, yt - 0.068 - h_ax, w_ax, h_ax])
    for kk in arc_keys[arc_keys.nfl_id == r_.nfl_id].itertuples(index=False):
        lat, fwd, al, kmx = arc_tr[tuple(kk)]
        P_ = np.c_[lat, fwd].reshape(-1, 1, 2)
        l2 = LineCollection(np.concatenate([P_[:-1], P_[1:]], axis=1), cmap=cmap, norm=norm, linewidth=1.6, alpha=0.9)
        l2.set_array(al[:-1]); gx.add_collection(l2)
        if 0 <= kmx < len(lat):
            gx.scatter([lat[kmx]], [fwd[kmx]], s=12, color=OI["red"], zorder=4, lw=0)
    gx.scatter([0], [0], marker="^", color=OI["black"], s=36, zorder=5)
    gx.set_xlim(-2, SPAN - 2); gx.set_ylim(-1, SPAN - 1); gx.set_aspect("equal")
    gx.set_xticks([0, 4, 8]); gx.set_yticks([0, 4, 8]); gx.tick_params(labelsize=9, length=2, pad=1)
    for sp_ in gx.spines.values():
        sp_.set_visible(True); sp_.set_color("#BBBBBB")
    gx.set_title(f"NFL Arc G {r_.arc_random:.2f} g", fontsize=11, weight="bold", pad=3)
    import textwrap
    fig.text(x0, yt - 0.088 - h_ax, "\n".join(textwrap.wrap(rec(r_), 50)), fontsize=10, va="top", style="italic")
# shared pressure strip
px = fig.add_axes([0.2, 0.05, 0.47, 0.18])
yy = np.arange(len(hero))[::-1]
for yv, r_ in zip(yy, hero.itertuples()):
    gcol = HIC if r_.grp == "high" else LOC
    px.plot([100 * r_.pb, 100 * r_.press_rate], [yv, yv], color=gcol, lw=1.5, alpha=0.6)
    px.scatter([100 * r_.pb], [yv], marker="|", s=160, color=OI["black"], lw=2, zorder=3)
    px.scatter([100 * r_.press_rate], [yv], s=70, color=gcol, edgecolor="k", zorder=4,
               marker="o" if r_.role == "EDGE" else "s")
    px.text(15.4, yv, f"{100 * r_.press_rate:.1f}% on {int(r_.rush_snaps)} rushes", va="center", fontsize=10.5)
px.axvline(100 * co.press_rate.median(), color=OI["grey"], lw=1, ls="--")
px.set_yticks(yy)
px.set_yticklabels([f"{name_short(r_.display_name)} (#{pk_txt(r_.pick)})".replace("#UDFA", "UDFA") for r_ in hero.itertuples()],
                   fontsize=10.5)
for t, r_ in zip(px.get_yticklabels(), hero.itertuples()):
    t.set_color(HIC if r_.grp == "high" else LOC)
px.set_xlim(0, 15); px.set_xticks([0, 5, 10, 15]); px.set_ylim(-0.6, len(hero) - 0.4)
px.set_xlabel("NFL career pressure rate (%)   | = no-tracking forecast, dot = actual, dashed = cohort median", fontsize=10.5)
px.tick_params(labelsize=10.5)
save(fig, "fig1_drill_vs_game.png")
hi_pr, lo_pr = 100 * hero[hero.grp == "high"].press_rate, 100 * hero[hero.grp == "low"].press_rate
put("hero_hi_min", hi_pr.min(), 1); put("hero_hi_max", hi_pr.max(), 1)
put("hero_lo_min", lo_pr.min(), 1); put("hero_lo_max", lo_pr.max(), 1)
for i, r_ in enumerate(hero.itertuples()):
    put(f"hero{i}_hoop", r_.hoop_g); put(f"hero{i}_press", 100 * r_.press_rate, 1); put(f"hero{i}_snaps", int(r_.rush_snaps))
    put(f"hero{i}_arc", r_.arc_random); put(f"hero{i}_pb", 100 * r_.pb, 1)
TABLES["hero"] = hero[["display_name", "grp", "role", "pick", "hoop_g", "hoop_k", "shr", "arc_random", "press_rate", "pb",
                       "rush_snaps"]].to_dict("records")
TABLES["hero_rec"] = {r_.display_name: rec(r_) for r_ in hero.itertuples()}

# ---- Fig 2: signal map, three NFL outcomes; tracking metrics shown raw -> after baseline
fig = plt.figure(figsize=(FIGW, 6.6))
gs2 = fig.add_gridspec(1, 3, left=0.2, right=0.99, top=0.83, bottom=0.25, wspace=0.08)
order = sig[sig.outcome == "press_eb"].sort_values("rho").col.tolist()
labs = {m_[0]: m_[1] for m_ in MEAS}
yy = np.arange(len(order))
axes = [fig.add_subplot(gs2[0, i]) for i in range(3)]
after = {("hoop_z", "press_eb"): (prim[("hoop_z", "press_eb")]["r"], prim[("hoop_z", "press_eb")]["lo"], prim[("hoop_z", "press_eb")]["hi"]),
         ("d1_z", "press_eb"): (prim[("d1_z", "press_eb")]["r"], prim[("d1_z", "press_eb")]["lo"], prim[("d1_z", "press_eb")]["hi"]),
         ("hoop_z", "quick"): (prim[("hoop_z", "quick")]["r"], prim[("hoop_z", "quick")]["lo"], prim[("hoop_z", "quick")]["hi"]),
         ("d1_z", "quick"): (prim[("d1_z", "quick")]["r"], prim[("d1_z", "quick")]["lo"], prim[("d1_z", "quick")]["hi"]),
         ("hoop_z", "arc_random"): ARC_AFTER["hoop_z"], ("d1_z", "arc_random"): ARC_AFTER["d1_z"]}
for ax, (oc, olab) in zip(axes, OUTC):
    d_ = sig[sig.outcome == oc].set_index("col").loc[order]
    for yv, (c, r_) in zip(yy, d_.iterrows()):
        ax.fill_betweenx([yv - 0.42, yv + 0.42], -r_.mde, r_.mde, color="#E6E6E6", lw=0)
        colr = OI["green"] if r_.kind == "tracking" else OI["black"]
        ax.plot([r_.lo, r_.hi], [yv, yv], color=colr, lw=1.8)
        ax.scatter([r_.rho], [yv], color=colr, s=55, zorder=3)
        if (c, oc) in after:
            ra_, la_, ha_ = after[(c, oc)]
            ax.annotate("", xy=(ra_, yv - 0.32), xytext=(r_.rho, yv - 0.32),
                        arrowprops=dict(arrowstyle="->", color=OI["purple"], lw=1.6))
            ax.scatter([ra_], [yv - 0.32], marker="D", facecolor="white", edgecolor=OI["purple"], s=55, zorder=4, lw=1.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-0.6, 1.0); ax.set_ylim(-0.8, len(order) - 0.4)
    ax.set_title(olab, fontsize=12.5, weight="bold")
    ax.tick_params(labelsize=11)
    ax.set_yticks(yy); ax.set_yticklabels([labs[c] for c in order] if ax is axes[0] else [], fontsize=11.5)
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
hand = [Line2D([], [], color=OI["black"], marker="o", lw=1.8, label="official / draft, raw ρ (95% CI)"),
        Line2D([], [], color=OI["green"], marker="o", lw=1.8, label="tracking drill metric, raw ρ"),
        Line2D([], [], color=OI["purple"], marker="D", markerfacecolor="white", lw=0,
               label="→ after the no-tracking baseline (Arc G: after weight, EDGE share)"),
        Patch(color="#E6E6E6", label="too small to detect with this n (80% power)")]
fig.legend(handles=hand, loc="lower center", ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.55, 0.0))
fig.text(0.595, 0.145, f"Spearman ρ; + = better Combine number goes with more of the outcome.\nPressure, get-off: "
         f"n ≤ {len(co)} DL (100+ rushes); Arc G: n ≤ {int(sig[sig.outcome == 'arc_random'].n.max())}",
         ha="center", fontsize=10.5)
fig.suptitle("Stopwatch numbers carry most of the signal. Hoop G keeps a pressure link after the baseline;\n"
             "its link to game bend is mostly size and alignment", fontsize=13.5, weight="bold", x=0.01, ha="left", y=0.985)
save(fig, "fig2_signal_map.png")

# ---- Fig 3: headline, baseline variants, selection check
fig = plt.figure(figsize=(FIGW, 9.6))
gs3 = fig.add_gridspec(2, 2, width_ratios=[1.15, 0.8], height_ratios=[1.3, 0.85], left=0.075, right=0.9, top=0.88,
                       bottom=0.075, hspace=0.42, wspace=0.75)
a1 = fig.add_subplot(gs3[0, 0])
d_ = dH.copy()
d_["res"] = 100 * d_.res_main
for role, gg_ in d_.groupby("role"):
    a1.scatter(gg_.hoop_z, gg_.res, color=ROLEC[role], s=38, alpha=0.85, label=f"{role} (n={len(gg_)})",
               marker="o" if role == "EDGE" else "s", edgecolor="white", lw=0.4)
mk_ = d_.ten_yd_split_z.isna()
a1.scatter(d_.hoop_z[mk_], d_.res[mk_], s=95, facecolor="none", edgecolor=OI["black"], lw=1.1,
           label=f"no official 40 / 10-yd (n={int(mk_.sum())})")
xx = np.linspace(-2.5, 2.5, 50)
bs = np.array([np.polyval(np.polyfit(d_.hoop_z.values[i], d_.res.values[i], 1), xx)
               for i in (RNG.integers(0, len(d_), len(d_)) for _ in range(NBOOT))])
a1.fill_between(xx, *np.percentile(bs, [2.5, 97.5], axis=0), color=OI["grey"], alpha=0.25, lw=0)
a1.plot(xx, np.polyval(np.polyfit(d_.hoop_z, d_.res, 1), xx), color="k", lw=1.6)
a1.axhline(0, color="k", lw=0.6)
a1.set_xlim(-2.8, 3.0); a1.set_ylim(d_.res.min() - 1.5, d_.res.max() + 2.5)
a1.set_xlabel("Hoop G, z within Combine year"); a1.set_ylabel("pressure rate above baseline (pp)")
a1.set_title(f"After pick, weight, EDGE share, NGS score: ρ = {NUM['pr_hoop_z_press_eb']} (n = {H['n']})", fontsize=12)
a1.legend(frameon=False, fontsize=10, loc="lower right")
lab_ids = list(hero.nfl_id) + list(d_.nlargest(1, "res").nfl_id) + list(d_.nsmallest(1, "res").nfl_id)
ld = d_[d_.nfl_id.isin(set(lab_ids))]
place_labels(a1, ld.hoop_z.values, ld.res.values, [name_short(n) for n in ld.display_name], fs=10.5, xmid=0.6)
a2 = fig.add_subplot(gs3[0, 1])
VROWS = [("main", "Main (no imputation)"), ("notime", "No athletic score"), ("mi", "+ 10-yd, multiple imputation"),
         ("single", "Round-3: 10-yd single fill"), ("cc_main", "Ran the 40 only: main"),
         ("cc_10", "Ran the 40 only: + 10-yd"), ("cc_40", "Ran the 40 only: + 40")]
rb_ = {r_["tag"]: r_ for r_ in ROB}
for j, (tag, lab) in enumerate(VROWS[::-1]):
    r_ = rb_[tag]
    colr = OI["black"] if tag.startswith("cc") else OI["green"]
    a2.plot([r_["lo"], r_["hi"]], [j, j], color=colr, lw=2)
    a2.scatter([r_["r"]], [j], color=colr, s=55, zorder=3, marker="s" if tag.startswith("cc") else "o")
    a2.text(1.02, j, f"{r_['dr2']:+.3f}".replace("-", "−"), va="center", fontsize=10.5, transform=a2.get_yaxis_transform())
a2.text(1.02, len(VROWS) - 0.35, "ΔR²\nheld out", fontsize=10, transform=a2.get_yaxis_transform(), va="bottom", weight="bold")
a2.axvline(0, color="k", lw=0.8)
a2.set_yticks(range(len(VROWS))); a2.set_yticklabels([l for _, l in VROWS[::-1]], fontsize=10.5)
a2.set_xlim(-0.4, 0.8); a2.set_xlabel("Hoop G effect after baseline, ρ (95% CI)")
a2.set_title(f"Green: all {NUM['v_n_main']}. Black: the {NUM['v_n_cc_main']} who ran the 40", fontsize=11.5)
a3 = fig.add_subplot(gs3[1, :])
bins = np.linspace(0, 0.75, 46)
a3.hist(null_all, bins=bins, color=OI["sky"], alpha=0.85)
a3.axvline(max_obs_all, color=OI["red"], lw=2.2)
a3.set_xlim(0, 0.75)
yl = a3.get_ylim()[1]
a3.text(max_obs_all + 0.015, yl * 0.9, f"observed: Hoop G → pressure, {max_obs_all:.2f}\n"
        f"chance does this well {100 * float(NUM['perm_best14']):.1f}% of the time\n(p = {NUM['perm_best14']}, best of 14)",
        color=OI["red"], fontsize=11, va="top", ha="left")
a3.set_xlabel(f"largest |effect after baseline| among 7 drill metrics × 2 outcomes, outcomes shuffled within class "
              f"({NPERM_SCREEN:,} times)", fontsize=11)
a3.set_ylabel("shuffles"); a3.tick_params(labelsize=10.5)
a3.set_title("Selection check: how often chance produces a lead this large when we keep the best of 14", fontsize=12)
fig.suptitle(f"Bend adds to the pressure forecast, but the gain rests on the {NUM['n_missing_10yd_hoop']} rushers "
             "who skipped the 40,\nand chance alone produces a lead this large about one time in "
             f"{int(round(1 / float(NUM['perm_best14'])))}",
             fontsize=14, weight="bold", x=0.01, ha="left")
save(fig, "fig3_hoop_pressure.png")

# ---- Fig 4: 2025 rookies with 100+ rushes: shrunk Hoop G, and forecast -> actual dumbbell
rk = bd.dropna(subset=["pb"]).copy()
over = rk[rk.flag == "drill too high"].display_name.map(lambda n: n.split(" ")[-1]).tolist()
under = rk[rk.flag == "drill too low"].display_name.map(lambda n: n.split(" ")[-1]).tolist()
fig = plt.figure(figsize=(FIGW, 0.36 * len(rk) + 2.6))
g4 = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.25], left=0.34, right=0.83, top=0.85, bottom=0.14, wspace=0.07)
b1, b2 = fig.add_subplot(g4[0]), fig.add_subplot(g4[1])
yy = np.arange(len(rk))[::-1]
for yv, r_ in zip(yy, rk.itertuples()):
    c_ = ROLEC[r_.role]
    b1.plot([r_.shr_lo, r_.shr_hi], [yv, yv], color=c_, lw=2.4, alpha=0.55)
    b1.scatter([r_.shr], [yv], color=c_, s=55, zorder=3, marker="o" if r_.role == "EDGE" else "s")
    b2.plot([100 * r_.pb, 100 * r_.press_rate], [yv, yv], color=OI["grey"], lw=2.2, zorder=1)
    b2.scatter([100 * r_.pb], [yv], s=60, facecolor="white", edgecolor=OI["black"], zorder=3, lw=1.4)
    b2.scatter([100 * r_.press_rate], [yv], s=60, color=OI["black"], zorder=4)
    b2.plot([100 * r_.pa] * 2, [yv - 0.32, yv + 0.32], color=OI["green"], lw=2.6, zorder=2)
    if r_.flag:
        b2.text(1.02, yv, r_.flag, transform=b2.get_yaxis_transform(), va="center", fontsize=11, weight="bold")
b1.axvline(0, color="k", lw=0.8)
b1.set_yticks(yy)
b1.set_yticklabels([f"{r_.display_name}  {'EDGE' if r_.role == 'EDGE' else 'Int'}, {int(r_.combine_weight)} lb, "
                    f"{'UDFA' if r_.pick == UDFA_PICK else '#' + str(int(r_.pick))}" for r_ in rk.itertuples()], fontsize=11)
b2.set_yticks(yy); b2.set_yticklabels([])
for b_ in (b1, b2):
    b_.set_ylim(-0.7, len(rk) - 0.3); b_.tick_params(labelsize=11)
b1.set_xlabel("shrunk Hoop G, z\n(bar = 80% interval)", fontsize=11)
b2.set_xlabel("2025 pressure rate (%)", fontsize=11)
b1.set_xlim(-1.6, 1.6); b2.set_xlim(0, 12.5); b2.set_xticks([0, 4, 8, 12])
b1.set_title("Drill bend", fontsize=12.5, weight="bold"); b2.set_title("Forecast → actual", fontsize=12.5, weight="bold")
hand = [Line2D([], [], marker="o", color="white", markeredgecolor="k", markersize=8, label="no-tracking forecast (fit on 2023–24)"),
        Line2D([], [], color=OI["green"], lw=2.6, label="+ Hoop G forecast"),
        Line2D([], [], marker="o", color="k", lw=0, markersize=8, label="actual")]
fig.legend(handles=hand, frameon=False, fontsize=11, loc="lower center", bbox_to_anchor=(0.56, 0.0), ncol=3)
ttl = "2025 rookies (100+ rushes): the hoop over-rated " + ", ".join(over) + (";\nit under-rated " + " and ".join(under) if under else "")
fig.suptitle(ttl + ". For the rest, bend barely moved the forecast", fontsize=13.5, weight="bold", x=0.01, ha="left", y=0.985)
save(fig, "fig4_rookies_2025.png")
put("n_over", len(over)); put("n_under", len(under))
TABLES["over"] = over; TABLES["under"] = under

# ----------------------------------------------------------------------------- write numbers
json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1, sort_keys=True)
json.dump(TABLES, open(OUT / "tables.json", "w"), indent=1, default=str)
print(json.dumps({k: NUM[k] for k in sorted(NUM) if not k.startswith(("rho_", "n_", "hero", "bt"))}, indent=0))
print(json.dumps({k: TABLES[k] for k in ["screen", "robustness", "nested_selected", "hero_rec", "flagged"]}, indent=0, default=str))
