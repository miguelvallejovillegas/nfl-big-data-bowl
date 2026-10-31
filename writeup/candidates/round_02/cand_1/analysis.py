"""Bend vs burst for DL/EDGE: Combine pass-rush-drill first-second distance and Run-the-Hoop bend (Hoop G)
against NFL get-off (NGS, every pass-rush snap) and pressure rate, beyond a no-tracking baseline.

Run from the repo root:
    uv run python writeup/candidates/round_02/cand_1/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json, tables.json and figures/*.png next to this file.
Every decimal quoted in writeup.md is produced here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy import stats
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[4]
PROC = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(2027)
NBOOT = 2000
NPERM = 1000
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
    half = n // 2
    return {"bend_g": np.nanmax(roll) / YD_PER_G, "bend_g_h1": np.nanmax(roll[:half]) / YD_PER_G,
            "bend_g_h2": np.nanmax(roll[half:]) / YD_PER_G, "speed_at_bend": v[k], "d1": d1,
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


def partial_sp(df, x, y, Z):
    d = df[[x, y] + Z].dropna()
    R = d.rank()
    A = np.c_[np.ones(len(R)), R[Z].values]
    rx = R[x] - A @ np.linalg.lstsq(A, R[x], rcond=None)[0]
    ry = R[y] - A @ np.linalg.lstsq(A, R[y], rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(d)
    b = []
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        Ai = A[i]
        bx = R[x].values[i] - Ai @ np.linalg.lstsq(Ai, R[x].values[i], rcond=None)[0]
        by = R[y].values[i] - Ai @ np.linalg.lstsq(Ai, R[y].values[i], rcond=None)[0]
        b.append(np.corrcoef(bx, by)[0, 1])
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    t = r * np.sqrt((n - 2 - len(Z)) / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), n - 2 - len(Z))
    return r, float(lo), float(hi), n, float(p)


def mde(n, k=0, power=0.8, alpha=0.05):
    """Smallest |rho| detectable with given power (Fisher z), k partialled covariates."""
    return float(np.tanh((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(n - 3 - k)))


def tost_p(r, n, bound, k=0):
    """Two one-sided tests that |rho| < bound (Fisher z). Small p = effect is smaller than bound."""
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
put("n_3cone", int(dl.three_cone.notna().sum()))
put("n_shuttle", int(dl.short_shuttle.notna().sum()))
put("n_10yd", int(dl.ten_yd_split.notna().sum()))

# EDGE share of NFL snaps (alignment table); fallback to listed position when no snaps
a = align.merge(dl[["nfl_id"]], on="nfl_id")
a["edge_pl"] = np.where(a.lined_up_position.isin(["EDGE", "OLB"]), a.plays, 0)
es = a.groupby("nfl_id").agg(edge_pl=("edge_pl", "sum"), pl=("plays", "sum"))
dl = dl.merge((es.edge_pl / es.pl).rename("edge_share").reset_index(), on="nfl_id", how="left")
dl["edge_share"] = dl.edge_share.fillna(dl.nfl_position.isin(["DE", "OLB"]).astype(float))
dl["edge"] = (dl.edge_share >= 0.5).astype(float)
dl["role"] = np.where(dl.edge == 1, "EDGE", "interior")

# ----------------------------------------------------------------------------- Combine drills
ev = events[(events.position_group == "DL") & events.drill_name.isin(["PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL"])]
ev = ev[["event_id", "nfl_id", "draft_year", "drill_name", "attempt_no", "start_speed", "peak_accel"]]
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
put("n_rush_att", len(rush_att))
put("n_rush_att_d1", int(rush_att.d1.notna().sum()))
put("hoop_g_median", hoop_att.bend_g.median())
put("hoop_g_p10", hoop_att.bend_g.quantile(0.1))
put("hoop_g_p90", hoop_att.bend_g.quantile(0.9))
put("rush_d1_median", rush_att.d1.median(), 1)
yr_med = hoop_att.groupby("draft_year").bend_g.median()
put("hoop_year_range", yr_med.max() - yr_med.min())
yr_d1 = rush_att.groupby("draft_year").d1.median()
put("d1_year_range", yr_d1.max() - yr_d1.min(), 1)

# reliability of each drill measure
r1_d1, n_d1pairs = icc(rush_att.nfl_id, rush_att.d1_z)
put("rel1_drill_d1", r1_d1); put("n_drill_d1_pairs", n_d1pairs)
r1_hoop_sh = stats.spearmanr(hoop_att.bend_g_h1, hoop_att.bend_g_h2)[0]
put("rel_hoop_splithalf", r1_hoop_sh)
rel_hoop = 2 * r1_hoop_sh / (1 + r1_hoop_sh)
put("rel_hoop_sb", rel_hoop)
k_rush = rush_att.dropna(subset=["d1"]).groupby("nfl_id").size().mean()
rel_d1_player = sb(r1_d1, k_rush)
put("rel_drill_d1_player", rel_d1_player)
put("k_rush_mean", k_rush, 1)
put("n_hoop_two", int((hoop_att.groupby("nfl_id").size() >= 2).sum()))

hoop = hoop_att.groupby("nfl_id").agg(hoop_g=("bend_g", "mean"), hoop_z=("hoop_z", "mean"),
                                      hoop_k=("bend_g", "size"), hoop_eid=("event_id", "first")).reset_index()
rush = rush_att.groupby("nfl_id").agg(d1=("d1", "mean"), d1_z=("d1_z", "mean"),
                                      rush_k=("d1", "count")).reset_index()
dl = dl.merge(hoop, on="nfl_id", how="left").merge(rush, on="nfl_id", how="left")
put("n_hoop_players", int(dl.hoop_z.notna().sum()))
put("n_rush_players", int(dl.d1_z.notna().sum()))
# EB-shrunk Hoop z and posterior sd, in true-score sd units (reliability rel_hoop per rep)
lam = dl.hoop_k * rel_hoop / (1 + (dl.hoop_k - 1) * rel_hoop)
dl["hoop_shr"] = lam * dl.hoop_z / np.sqrt(rel_hoop)
dl["hoop_shr_sd"] = np.sqrt(1 - lam)
put("hoop_lambda_1rep", rel_hoop)

# tracked 40 top speed (needed for "does tracking add over the stopwatch")
f40 = events[(events.position_group == "DL") & (events.drill_type == "FORTY_YARD_DASH")]
t40 = f40.groupby("nfl_id").top_speed.max().rename("top40").reset_index()
dl = dl.merge(t40, on="nfl_id", how="left")
dl["top40_z"] = yz(dl.top40, dl.draft_year)

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
put("n_any_snaps", int((dl.rush_snaps > 0).sum()))
put("n_zero_snaps", int((dl.rush_snaps == 0).sum()))
put("total_rush_snaps", int(dl.rush_snaps.sum()))
put("total_pressures", int(dl.pressures.fillna(0).sum()))

# beta-binomial EB pressure rate, ONE pooled DL prior (a role-specific prior would build EDGE into the outcome).
# Method of moments (Kleinman): tau2 = (sum n_i (p_i - mu)^2 - (k - 1) mu (1 - mu)) / (N - sum n_i^2 / N)
m = dl.rush_snaps >= 20
n_, k_ = dl.loc[m, "rush_snaps"], dl.loc[m, "pressures"]
p_ = k_ / n_
mu = k_.sum() / n_.sum()
N_ = n_.sum()
tau2 = ((n_ * (p_ - mu) ** 2).sum() - (len(n_) - 1) * mu * (1 - mu)) / (N_ - (n_ ** 2).sum() / N_)
M = mu * (1 - mu) / tau2 - 1
put("prior_weight", int(round(M)))
put("prior_mean", 100 * mu, 1)
put("press_true_sd_pp", 100 * np.sqrt(tau2), 1)
mm = dl.rush_snaps > 0
dl.loc[mm, "press_eb"] = (dl.loc[mm, "pressures"] + M * mu) / (dl.loc[mm, "rush_snaps"] + M)

# outcome reliability: season to season (>= MIN_SNAPS both seasons)
s2 = s[s.pass_rush_snaps >= MIN_SNAPS][["nfl_id", "season", "pressures", "pass_rush_snaps", "avg_get_off"]].copy()
s2["pr"] = s2.pressures / s2.pass_rush_snaps
pr2 = s2.merge(s2.assign(season=s2.season - 1), on=["nfl_id", "season"], suffixes=("", "_n"))
put("yoy_n", len(pr2))
rel_press_season = stats.spearmanr(pr2.pr, pr2.pr_n)[0]
rel_getoff_season = stats.spearmanr(pr2.avg_get_off, pr2.avg_get_off_n)[0]
put("yoy_press", rel_press_season)
put("yoy_getoff", rel_getoff_season)
put("median_snaps_season", int(s2.pass_rush_snaps.median()))

# ----------------------------------------------------------------------------- game tracking check of get-off
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
gd = gd.merge(g[K + ["player_get_off", "sample_kind", "lined_up_position"]], on=K).dropna()
put("n_game_plays", len(gd))
put("n_game_players", gd.nfl_id.nunique())
put("rho_play_d1_getoff", stats.spearmanr(gd.g_d1, gd.player_get_off)[0])
put("game_d1_median", gd.g_d1.median(), 1)
rnd = gd[gd.sample_kind == "random"].copy()
put("n_random_plays", len(rnd))
put("random_per_player", rnd.groupby("nfl_id").size().median(), 0)
rnd["g_d1_z"] = (rnd.g_d1 - rnd.g_d1.mean()) / rnd.g_d1.std()
r1_game, _ = icc(rnd.nfl_id, rnd.g_d1_z)
put("rel1_game_d1", r1_game)
r1_go, _ = icc(rnd.nfl_id, rnd.player_get_off)
put("rel1_play_getoff", r1_go)
gpl = rnd.groupby("nfl_id").agg(g_d1=("g_d1", "mean")).reset_index()
dl = dl.merge(gpl, on="nfl_id", how="left")

# ----------------------------------------------------------------------------- cohort and raw signals
co = dl[dl.rush_snaps >= MIN_SNAPS].copy()
put("n_cohort", len(co))
put("n_cohort_edge", int((co.role == "EDGE").sum()))
put("n_cohort_int", int((co.role == "interior").sum()))
put("cohort_median_snaps", int(co.rush_snaps.median()))
put("cohort_press_median", 100 * co.press_rate.median(), 1)
put("cohort_getoff_median", co.get_off.median())
put("rho_play_vs_season_getoff", stats.spearmanr(co.g_d1, co.get_off, nan_policy="omit")[0])

# signs: higher = better athlete. get-off: lower seconds = quicker -> use negative
co["quick"] = -co.get_off
MEAS = [  # col, label, kind, sign
    ("forty_z", "40-yard time", "official", -1), ("ten_yd_split_z", "10-yd split", "official", -1),
    ("three_cone_z", "3-cone", "official", -1), ("short_shuttle_z", "Short shuttle", "official", -1),
    ("combine_weight_z", "Lighter weight", "official", -1), ("ngs_athleticism_score", "NGS athleticism", "official", 1),
    ("log_pick", "Draft pick", "official", -1),
    ("top40_z", "40 top speed (tracked)", "tracking", 1),
    ("d1_z", "Rush-drill first-second distance", "tracking", 1),
    ("hoop_z", "Hoop G (bend)", "tracking", 1),
]
OUTC = [("quick", "NFL get-off (quicker)"), ("press_eb", "NFL pressure rate")]
sig = []
for col, lab, kind, sgn in MEAS:
    for oc, olab in OUTC:
        r, lo, hi, n, p = sp_ci(sgn * co[col], co[oc])
        sig.append(dict(col=col, label=lab, kind=kind, outcome=oc, olab=olab, rho=r, lo=lo, hi=hi, n=n, p=p,
                        mde=mde(n)))
sig = pd.DataFrame(sig)
for r_ in sig.itertuples():
    put(f"rho_{r_.col}_{r_.outcome}", r_.rho)
    put(f"lo_{r_.col}_{r_.outcome}", r_.lo)
    put(f"hi_{r_.col}_{r_.outcome}", r_.hi)
    put(f"n_{r_.col}_{r_.outcome}", int(r_.n))
    put(f"mde_{r_.col}_{r_.outcome}", r_.mde)
TABLES["signals"] = sig.round(3).to_dict("records")

# weight is the get-off story
put("rho_weight_getoff_raw", stats.spearmanr(co.combine_weight, co.get_off)[0])
put("rho_d1_weight", stats.spearmanr(dl.d1_z, dl.combine_weight, nan_policy="omit")[0])
put("rho_hoop_weight", stats.spearmanr(dl.hoop_z, dl.combine_weight, nan_policy="omit")[0])
put("rho_hoop_pick", stats.spearmanr(dl.hoop_z, -dl.log_pick, nan_policy="omit")[0])
put("n_hoop_pick", int(dl.hoop_z.notna().sum()))
put("rho_hoop_d1", stats.spearmanr(dl.hoop_z, dl.d1_z, nan_policy="omit")[0])

# ----------------------------------------------------------------------------- partial after baseline + power
BASE = ["combine_weight_z", "edge_share", "ten_yd_split_z", "three_cone_z", "ngs_athleticism_score", "log_pick"]
BASE_NOEDGE = [c for c in BASE if c != "edge_share"]


def impute(tr: pd.DataFrame, te: pd.DataFrame, cols=("ten_yd_split_z", "three_cone_z")):
    """Fill missing official times from weight + EDGE share, fit on training rows only (no outcome)."""
    tr, te = tr.copy(), te.copy()
    for c in cols:
        m = tr[c].notna()
        A = np.c_[np.ones(m.sum()), tr.loc[m, ["combine_weight_z", "edge_share"]].values]
        b = np.linalg.lstsq(A, tr.loc[m, c].values, rcond=None)[0]
        for d in (tr, te):
            mm = d[c].isna()
            d.loc[mm, c] = np.c_[np.ones(mm.sum()), d.loc[mm, ["combine_weight_z", "edge_share"]].values] @ b
    return tr, te


coi, _ = impute(co, co)
put("n_missing_3cone_cohort", int(co.three_cone_z.isna().sum()))
put("n_missing_10yd_cohort", int(co.ten_yd_split_z.isna().sum()))
REL_OUT = {"quick": sb(rel_getoff_season, co.n_seasons.mean()), "press_eb": sb(rel_press_season, co.n_seasons.mean())}
put("rel_getoff_career", REL_OUT["quick"])
put("rel_press_career", REL_OUT["press_eb"])
REL_X = {"d1_z": rel_d1_player, "hoop_z": rel_hoop}
PRIMARY = [("d1_z", "quick"), ("hoop_z", "press_eb"), ("d1_z", "press_eb"), ("hoop_z", "quick")]
prim = {}
for x, y in PRIMARY:
    r, lo, hi, n, p = partial_sp(coi, x, y, BASE)
    m_obs = mde(n, len(BASE))
    ceil = np.sqrt(REL_X[x] * REL_OUT[y])
    m_true = min(m_obs / ceil, 1.0)
    tp = tost_p(r, n, 0.30, len(BASE))
    rn, lon, hin, nn, pn = partial_sp(coi, x, y, BASE_NOEDGE)
    prim[(x, y)] = dict(r=r, lo=lo, hi=hi, n=n, p=p, mde=m_obs, ceil=ceil, mde_true=m_true, tost=tp, r_noedge=rn,
                        p_noedge=pn)
    key = f"{x}_{y}"
    put(f"pr_{key}", r); put(f"pr_lo_{key}", lo); put(f"pr_hi_{key}", hi); put(f"pr_n_{key}", n)
    put(f"pr_p_{key}", p, 3); put(f"pmde_{key}", m_obs); put(f"ceil_{key}", ceil); put(f"mdetrue_{key}", m_true)
    put(f"tost_{key}", tp, 3); put(f"pr_noedge_{key}", rn); put(f"pr_p_noedge_{key}", pn, 3)
# Holm over the four primary tests
ps = np.array([prim[k]["p"] for k in PRIMARY])
o = np.argsort(ps)
holm = np.empty(4)
run = 0
for rank, i in enumerate(o):
    run = max(run, (4 - rank) * ps[i])
    holm[i] = min(1, run)
for (x, y), h in zip(PRIMARY, holm):
    put(f"holm_{x}_{y}", h, 3)

# by role, for the two headline tests
for role in ["EDGE", "interior"]:
    c = coi[coi.role == role]
    for x, y in PRIMARY[:2]:
        r, lo, hi, n, p = partial_sp(c, x, y, [b for b in BASE if b != "edge_share"])
        put(f"pr_{x}_{y}_{role}", r); put(f"pr_n_{x}_{y}_{role}", n)

# snap thresholds sensitivity for Hoop G -> pressure
for thr in [50, 200]:
    c = dl[dl.rush_snaps >= thr].copy()
    c, _ = impute(c, c)
    r, lo, hi, n, p = partial_sp(c, "hoop_z", "press_eb", BASE)
    put(f"pr_hoop_press_snaps{thr}", r); put(f"pr_n_hoop_press_snaps{thr}", n)

# ----------------------------------------------------------------------------- opportunity / survival
oc_ = dl.copy()
for col, lab in [("hoop_z", "hoop"), ("d1_z", "d1"), ("three_cone_z", "3cone")]:
    r, lo, hi, n, p = sp_ci(oc_[col], oc_.snaps_per_season)
    put(f"rho_{lab}_opp", r); put(f"lo_{lab}_opp", lo); put(f"hi_{lab}_opp", hi); put(f"n_{lab}_opp", n)
    r2_, _, _, n2, _ = sp_ci(oc_[col], oc_.games_per_season)
    put(f"rho_{lab}_games", r2_)
put("rho_pick_opp", stats.spearmanr(-oc_.log_pick, oc_.snaps_per_season)[0])
reach = oc_.dropna(subset=["hoop_z"])
put("reach100_hoop_top", 100 * (reach[reach.hoop_z > 0].rush_snaps >= MIN_SNAPS).mean(), 0)
put("reach100_hoop_bot", 100 * (reach[reach.hoop_z <= 0].rush_snaps >= MIN_SNAPS).mean(), 0)


# ----------------------------------------------------------------------------- leave-one-class-out
def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ols_pred(tr, te, cols, y):
    A = np.c_[np.ones(len(tr)), tr[cols].values]
    b = np.linalg.lstsq(A, tr[y].values, rcond=None)[0]
    return np.c_[np.ones(len(te)), te[cols].values] @ b


def loco(df, base, extra, y, perm=True):
    classes = sorted(df.draft_year.unique())

    def run(d):
        Y, PB, PA, CL, ID = [], [], [], [], []
        for c in classes:
            tr, te = impute(d[d.draft_year != c], d[d.draft_year == c])
            PB.append(ols_pred(tr, te, base, y))
            PA.append(ols_pred(tr, te, base + extra, y))
            Y.append(te[y].values); CL.append(np.full(len(te), c)); ID.append(te.nfl_id.values)
        return tuple(map(np.concatenate, (Y, PB, PA, CL, ID)))
    Y, PB, PA, CL, ID = run(df)
    out = dict(n=len(Y), r2_base=r2(Y, PB), r2_aug=r2(Y, PA))
    out["delta"] = out["r2_aug"] - out["r2_base"]
    out["folds"] = {int(c): dict(n=int((CL == c).sum()), base=float(r2(Y[CL == c], PB[CL == c])),
                                 aug=float(r2(Y[CL == c], PA[CL == c]))) for c in classes}
    out["folds_pos"] = int(sum(f["aug"] > f["base"] for f in out["folds"].values()))
    n = len(Y)
    bd = []
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        bd.append(r2(Y[i], PA[i]) - r2(Y[i], PB[i]))
    out["lo"], out["hi"] = (float(v) for v in np.percentile(bd, [2.5, 97.5]))
    if perm:
        null = []
        for _ in range(NPERM):
            dp = df.copy()
            for c in classes:
                m = (dp.draft_year == c).values
                dp.loc[m, extra] = dp.loc[m, extra].values[RNG.permutation(m.sum())]
            a_, b_, c_, _, _ = run(dp)
            null.append(r2(a_, c_) - r2(a_, b_))
        out["perm_p"] = float((1 + np.sum(np.array(null) >= out["delta"])) / (1 + NPERM))
        out["null"] = np.array(null)
    out["pred"] = pd.DataFrame({"nfl_id": ID, "y": Y, "pb": PB, "pa": PA, "cl": CL})
    return out


LO = {}
for x, y in PRIMARY[:2]:
    d = co.dropna(subset=[x, y]).copy()
    for tag, base in [("", BASE), ("_noedge", BASE_NOEDGE)]:
        res = loco(d, base, [x], y, perm=(tag == ""))
        LO[(x, y, tag)] = res
        k = f"{x}_{y}{tag}"
        put(f"lo_n_{k}", res["n"]); put(f"r2b_{k}", res["r2_base"]); put(f"r2a_{k}", res["r2_aug"])
        put(f"dr2_{k}", res["delta"], 3); put(f"dr2lo_{k}", res["lo"], 3); put(f"dr2hi_{k}", res["hi"], 3)
        put(f"foldspos_{k}", res["folds_pos"])
        for c, f in res["folds"].items():
            put(f"r2b_{k}_{c}", f["base"]); put(f"r2a_{k}_{c}", f["aug"]); put(f"fn_{k}_{c}", f["n"])
        if "perm_p" in res:
            put(f"perm_{k}", res["perm_p"], 3)
# baseline-only references (pressure): pick only, pick+edge
d = co.dropna(subset=["hoop_z", "press_eb"]).copy()
put("r2_pick_only_press", loco(d, ["log_pick"], [], "press_eb", perm=False)["r2_base"])
put("r2_pick_edge_press", loco(d, ["log_pick", "edge_share"], [], "press_eb", perm=False)["r2_base"])
d = co.dropna(subset=["d1_z", "quick"]).copy()
put("r2_weight_edge_getoff", loco(d, ["combine_weight_z", "edge_share"], [], "quick", perm=False)["r2_base"])
# does tracked 40 top speed add over the official 40 for pressure?
d = co.dropna(subset=["top40_z", "forty_z", "press_eb"]).copy()
t = loco(d, ["combine_weight_z", "edge_share", "forty_z", "log_pick"], ["top40_z"], "press_eb", perm=False)
put("dr2_top40_over40", t["delta"], 3); put("n_top40_over40", t["n"])

# effect size in pressures: in-sample OLS slope of shrunk pressure on hoop_z with baseline
d = coi.dropna(subset=["hoop_z", "press_eb"])
A = np.c_[np.ones(len(d)), d[BASE + ["hoop_z"]].values]
beta, res_, *_ = np.linalg.lstsq(A, d.press_eb.values, rcond=None)
resid = d.press_eb.values - A @ beta
sig2 = (resid ** 2).sum() / (len(d) - A.shape[1])
cov = sig2 * np.linalg.inv(A.T @ A)
b_h, se_h = beta[-1], np.sqrt(cov[-1, -1])
put("slope_hoop_pp", 100 * b_h); put("slope_hoop_pp_lo", 100 * (b_h - 1.96 * se_h)); put("slope_hoop_pp_hi", 100 * (b_h + 1.96 * se_h))
put("pressures_per_450", 450 * b_h, 1)

# ----------------------------------------------------------------------------- 2025 draft-room card (held-out class)
res = LO[("hoop_z", "press_eb", "")]
pr = res["pred"]
pr = pr[pr.cl == 2025].merge(dl[["nfl_id", "display_name", "pick", "role", "hoop_g", "hoop_shr", "hoop_shr_sd",
                                 "three_cone", "rush_snaps", "press_rate"]], on="nfl_id")
# residual sd from 2023-24 out-of-class residuals -> 80% interval on predicted rate
oth = res["pred"][res["pred"].cl != 2025]
rsd = float(np.std(oth.y - oth.pa))
pr["lo80"], pr["hi80"] = pr.pa - 1.2816 * rsd, pr.pa + 1.2816 * rsd
pr["move"] = 100 * (pr.pa - pr.pb)
pr["rank_b"] = pr.pb.rank(ascending=False).astype(int)
pr["rank_a"] = pr.pa.rank(ascending=False).astype(int)
pr["rank_y"] = pr.y.rank(ascending=False).astype(int)
pr["shift"] = pr.rank_b - pr.rank_a           # + = bend moves him up the board
pr["flag"] = np.where(pr["shift"] >= 3, "bend up", np.where(pr["shift"] <= -3, "bend down", ""))
pr = pr.sort_values("pa", ascending=False).reset_index(drop=True)
put("card_n", len(pr)); put("card_rsd_pp", 100 * rsd, 1)
put("card_rho_base", stats.spearmanr(pr.pb, pr.y)[0]); put("card_rho_aug", stats.spearmanr(pr.pa, pr.y)[0])
fl = pr[pr.flag != ""]
put("card_n_flag", len(fl))
closer = (fl.rank_a - fl.rank_y).abs() < (fl.rank_b - fl.rank_y).abs()
put("card_flag_closer", int(closer.sum()))
TABLES["card_2025"] = pr.round(4).to_dict("records")

# per-fold rank correlation of predictions (robust to the 2025 level shift)
for (x, y) in PRIMARY[:2]:
    res_ = LO[(x, y, "")]["pred"]
    for c, d_ in res_.groupby("cl"):
        put(f"frho_b_{x}_{y}_{c}", stats.spearmanr(d_.pb, d_.y)[0]); put(f"frho_a_{x}_{y}_{c}", stats.spearmanr(d_.pa, d_.y)[0])
yb = LO[("hoop_z", "press_eb", "")]["pred"]
put("mean_pred_2025_pp", 100 * yb[yb.cl == 2025].pa.mean(), 1); put("mean_y_2025_pp", 100 * yb[yb.cl == 2025].y.mean(), 1)

# what would change my mind: 2026 partial rho (n=20) that would pull the pooled 95% CI to include zero
k_ = len(BASE)
n1, z1 = prim[("hoop_z", "press_eb")]["n"], np.arctanh(prim[("hoop_z", "press_eb")]["r"])
n2 = 20
w1, w2 = n1 - 3 - k_, n2 - 3 - k_
z2 = (1.96 * np.sqrt(w1 + w2) - w1 * z1) / w2
put("cmm_n2026", n2); put("cmm_r2026", float(np.tanh(z2)))
# game-side reliability: random snaps needed for 0.7 on per-play NGS get-off
put("snaps_for_07_getoff", int(np.ceil(0.7 * (1 - r1_go) / (r1_go * 0.3))))
put("snaps_for_07_drill", int(np.ceil(0.7 * (1 - r1_d1) / (r1_d1 * 0.3))))
put("snaps_for_07_hoop", int(np.ceil(0.7 * (1 - rel_hoop) / (rel_hoop * 0.3))))
# same-metric translation: drill first-second distance vs tracked game first-second distance (random snaps)
r, lo, hi, n, p = sp_ci(dl.d1_z, dl.g_d1)
put("rho_d1_drill_game", r); put("lo_d1_drill_game", lo); put("hi_d1_drill_game", hi); put("n_d1_drill_game", n)
# exploratory (found after looking): same-metric link in the cohort, after baseline, and by role
r, lo, hi, n, p = partial_sp(coi, "d1_z", "g_d1", BASE)
put("pr_d1_gamed1", r); put("pr_lo_d1_gamed1", lo); put("pr_hi_d1_gamed1", hi); put("pr_n_d1_gamed1", n)
for role in ["EDGE", "interior"]:
    c = dl[dl.role == role]
    r, lo, hi, n, p = sp_ci(c.d1_z, c.g_d1)
    put(f"rho_d1_gamed1_{role}", r); put(f"n_d1_gamed1_{role}", n)
r, lo, hi, n, p = sp_ci(co.g_d1, co.press_eb)
put("rho_gamed1_press", r); put("n_gamed1_press", n)
r, lo, hi, n, p = sp_ci(co.quick, co.press_eb)
put("rho_getoff_press", r)
r, lo, hi, n, p = partial_sp(coi, "quick", "press_eb", BASE)
put("pr_getoff_press", r); put("pr_lo_getoff_press", lo); put("pr_hi_getoff_press", hi)

# ============================================================================= FIGURES
ROLEC = {"EDGE": OI["orange"], "interior": OI["blue"]}


def name_short(n):
    return n.split(" ")[0][0] + ". " + " ".join(n.split(" ")[1:])


# ---- Fig 1: hero cards, chosen by rule: 3 highest and 3 lowest Hoop G among DL with >= 250 rush snaps
hero_pool = co.dropna(subset=["hoop_z"])
hero_pool = hero_pool[hero_pool.rush_snaps >= 250].sort_values("hoop_z", ascending=False)
hero = pd.concat([hero_pool.head(3), hero_pool.tail(3)])
put("hero_pool_n", len(hero_pool))
BARS = [("hoop_z", "Hoop G (bend)", 1), ("d1_z", "Drill first second", 1), ("three_cone_z", "3-cone", -1),
        ("ten_yd_split_z", "10-yd split", -1), ("combine_weight_z", "Lighter weight", -1)]
fig = plt.figure(figsize=(FIGW, 9.6))
gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1, 1.0, 1], hspace=0.95, wspace=0.12, left=0.155, right=0.99,
                      top=0.835, bottom=0.07)
norm = plt.Normalize(0, 0.9)
cmap = plt.get_cmap("cividis")
for idx, r_ in enumerate(hero.itertuples()):
    row, col = 2 * (idx // 3), idx % 3
    ax = fig.add_subplot(gs[row, col])
    tr = traces[r_.hoop_eid]
    xs_, ys_ = tr["_x"] - tr["_x"].mean(), tr["_y"] - tr["_y"].mean()
    if np.ptp(ys_) > np.ptp(xs_):
        xs_, ys_ = ys_, xs_
    pts = np.c_[xs_, ys_].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=norm, linewidth=3.2)
    lc.set_array(tr["_alat"][:-1])
    ax.add_collection(lc)
    k = tr["_k"]
    ax.scatter(xs_[max(k - 2, 0):k + 3], ys_[max(k - 2, 0):k + 3], s=40, facecolor="none", edgecolor=OI["red"], lw=1.6, zorder=3)
    ax.set_xlim(-6.2, 6.2); ax.set_ylim(-2.7, 2.7); ax.set_aspect("equal"); ax.axis("off")
    pk = "UDFA" if r_.pick == UDFA_PICK else f"pick {int(r_.pick)}"
    ax.set_title(f"{r_.display_name}\n{r_.role}, {pk} · Hoop G {r_.hoop_g:.2f} g\nNFL pressure {100 * r_.press_rate:.1f}% on {int(r_.rush_snaps)} rushes",
                 fontsize=10.5)
    bx = fig.add_subplot(gs[row + 1, col])
    vals = [sg * getattr(r_, c) if pd.notna(getattr(r_, c)) else np.nan for c, _, sg in BARS]
    yy = np.arange(len(BARS))[::-1]
    cols = [OI["green"] if c in ("hoop_z", "d1_z") else OI["grey"] for c, _, _ in BARS]
    bx.barh(yy, np.nan_to_num(vals), color=cols, height=0.62)
    for yv, v in zip(yy, vals):
        if np.isnan(v):
            bx.text(0.08, yv, "not run", va="center", fontsize=9.5, color=OI["grey"])
    bx.axvline(0, color="k", lw=0.8)
    bx.set_xlim(-2.6, 2.6)
    bx.set_yticks(yy); bx.set_yticklabels([l for _, l, _ in BARS] if col == 0 else [], fontsize=10)
    bx.tick_params(axis="x", labelsize=9.5)
    if row == 2:
        bx.set_xlabel("z within Combine year (+ = better)", fontsize=10)
fig.text(0.02, 0.968, "The tightest benders (top) out-rushed the widest (bottom) in the NFL",
         fontsize=13.5, weight="bold")
fig.text(0.02, 0.938, "3 highest and 3 lowest Hoop G among DL with 250+ NFL rushes. Trace colour = turning (lateral) acceleration; "
         "red circles = the 0.5 s\nthat defines Hoop G. Bars = Combine numbers as z within year (green = tracking).", fontsize=10.5, va="top")
sm_ = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
cax = fig.add_axes([0.015, 0.035, 0.11, 0.012])
cb = fig.colorbar(sm_, cax=cax, orientation="horizontal", ticks=[0, 0.3, 0.6, 0.9]); cb.ax.tick_params(labelsize=9)
fig.text(0.015, 0.05, "lateral g", fontsize=10)
save(fig, "fig1_hero_cards.png")
TABLES["hero"] = hero[["display_name", "role", "pick", "hoop_g", "hoop_z", "press_rate", "rush_snaps", "get_off"]].round(3).to_dict("records")
for i, r_ in enumerate(hero.itertuples()):
    put(f"hero{i}_hoop", r_.hoop_g); put(f"hero{i}_press", 100 * r_.press_rate, 1); put(f"hero{i}_snaps", int(r_.rush_snaps))

# ---- Fig 2: measurement ladder (left) and per-play validation of NGS get-off (right)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})
lad = [("Drill: first-second distance\n(pass-rush drill)", r1_d1, rel_d1_player, f"{k_rush:.1f} reps"),
       ("Drill: Hoop G\n(run-the-hoop)", r1_hoop_sh, rel_hoop, "1 rep"),
       ("NFL: pressure rate", rel_press_season, REL_OUT["press_eb"], f"{co.n_seasons.mean():.1f} seasons"),
       ("NFL: NGS get-off", rel_getoff_season, REL_OUT["quick"], f"{co.n_seasons.mean():.1f} seasons")]
yy = np.arange(len(lad))[::-1]
for yv, (lab, r1v, rpv, unit) in zip(yy, lad):
    a1.plot([r1v, rpv], [yv, yv], color=OI["grey"], lw=2)
    a1.scatter([r1v], [yv], color="white", edgecolor=OI["black"], s=70, zorder=3)
    a1.scatter([rpv], [yv], color=OI["green"] if "Drill" in lab else OI["blue"], s=90, zorder=3)
    a1.text(rpv + 0.03, yv + 0.18, f"{rpv:.2f} ({unit})", fontsize=10)
a1.axvline(0.7, color=OI["red"], ls="--", lw=1)
a1.text(0.71, yy[-1] - 0.45, "0.7 = usable\nfor ranking", color=OI["red"], fontsize=10)
a1.set_yticks(yy); a1.set_yticklabels([l for l, *_ in lad], fontsize=10.5)
a1.set_xlim(0, 1.12); a1.set_ylim(-0.8, len(lad) - 0.4)
a1.set_xlabel("reliability\n(open = one rep, half-hoop or season; filled = player total)")
a1.set_title("The drills, not the NFL outcomes, are the noisy side", fontsize=12.5)
hb = a2.hexbin(gd.g_d1, gd.player_get_off, gridsize=32, cmap="cividis", mincnt=1, extent=(0, 4, 0.3, 2.0))
a2.set_xlabel("tracked distance, first second\nafter the snap (yd)")
a2.set_ylabel("NGS get-off on the same snap (s)")
a2.set_title(f"NGS get-off = first-second burst\n(ρ = {NUM['rho_play_d1_getoff']}, {len(gd):,} DL pass rushes)", fontsize=12.5)
fig.colorbar(hb, ax=a2, label="snaps")
fig.tight_layout()
save(fig, "fig2_measurement.png")

# ---- Fig 3: signal map
fig, axes = plt.subplots(1, 2, figsize=(FIGW, 6.8), sharey=True)
order = [m[0] for m in MEAS]
labs = {m[0]: m[1] for m in MEAS}
yy = np.arange(len(order))[::-1]
for ax, (oc, olab) in zip(axes, OUTC):
    d_ = sig[sig.outcome == oc].set_index("col").loc[order]
    for yv, (c, r_) in zip(yy, d_.iterrows()):
        ax.fill_betweenx([yv - 0.4, yv + 0.4], -r_.mde, r_.mde, color="#DDDDDD", lw=0)
        colr = OI["green"] if r_.kind == "tracking" else OI["black"]
        ax.plot([r_.lo, r_.hi], [yv, yv], color=colr, lw=1.8)
        ax.scatter([r_.rho], [yv], color=colr, s=55, zorder=3)
        ax.text(0.97, yv, f"n={int(r_.n)}", fontsize=9.5, va="center", ha="right", transform=ax.get_yaxis_transform())
    for (x, y), pv in prim.items():
        if y == oc:
            ax.scatter([pv["r"]], [yy[order.index(x)] - 0.28], marker="D", facecolor="white", edgecolor=OI["green"],
                       s=45, zorder=4)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-0.6, 1.15)
    ax.set_title(olab + f" (n ≤ {len(co)} DL, 100+ rushes)", fontsize=12)
axes[0].set_yticks(yy); axes[0].set_yticklabels([labs[c] for c in order], fontsize=10.5)
axes[1].scatter([], [], color=OI["black"], label="official / draft (raw ρ, 95% CI)")
axes[1].scatter([], [], color=OI["green"], label="tracking (raw ρ, 95% CI)")
axes[1].scatter([], [], marker="D", facecolor="white", edgecolor=OI["green"], label="tracking after baseline")
axes[1].fill_between([], [], color="#DDDDDD", label="too small to detect at this n")
fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=10, frameon=False)
fig.supxlabel("Spearman ρ, signed so + = the better Combine number goes with the better NFL result", y=0.075, fontsize=11.5)
fig.suptitle("Speed and agility times carry the DL signal; of the drills, only bend survives the baseline", fontsize=13.5,
             weight="bold", x=0.02, ha="left")
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
save(fig, "fig3_signal_map.png")

# ---- Fig 4: Hoop G vs pressure (baseline residual) and out-of-class by fold
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 5.4), gridspec_kw={"width_ratios": [1.3, 1]})
d_ = coi.dropna(subset=["hoop_z", "press_eb"]).copy()
A = np.c_[np.ones(len(d_)), d_[BASE].values]
d_["res"] = 100 * (d_.press_eb - A @ np.linalg.lstsq(A, d_.press_eb.values, rcond=None)[0])
for role, gg in d_.groupby("role"):
    a1.scatter(gg.hoop_z, gg.res, color=ROLEC[role], s=38, alpha=0.85, label=f"{role} (n={len(gg)})",
               marker="o" if role == "EDGE" else "s")
xx = np.linspace(-2.5, 2.5, 50)
bs = []
for _ in range(NBOOT):
    i = RNG.integers(0, len(d_), len(d_))
    b = np.polyfit(d_.hoop_z.values[i], d_.res.values[i], 1)
    bs.append(np.polyval(b, xx))
bs = np.array(bs)
a1.fill_between(xx, *np.percentile(bs, [2.5, 97.5], axis=0), color=OI["grey"], alpha=0.25, lw=0)
a1.plot(xx, np.polyval(np.polyfit(d_.hoop_z, d_.res, 1), xx), color="k", lw=1.6)
lab_ids = set(hero.nfl_id) | set(d_.nlargest(2, "res").nfl_id) | set(d_.nsmallest(2, "res").nfl_id)
for r_ in d_[d_.nfl_id.isin(lab_ids)].itertuples():
    a1.annotate(name_short(r_.display_name), (r_.hoop_z, r_.res), xytext=(5, 4), textcoords="offset points", fontsize=9.5)
a1.axhline(0, color="k", lw=0.6)
a1.set_xlabel("Hoop G, z within Combine year")
a1.set_ylabel("pressure rate above baseline expectation (pp)")
a1.set_title(f"Bend beyond what the baseline knows: partial ρ = {NUM['pr_hoop_z_press_eb']}\n(line = OLS fit, band = 95% bootstrap)",
             fontsize=12)
a1.legend(frameon=False, fontsize=10, loc="upper left")
cls = [2023, 2024, 2025]
w = 0.36
fb = [NUM[f"frho_b_hoop_z_press_eb_{c}"] for c in cls]
fa = [NUM[f"frho_a_hoop_z_press_eb_{c}"] for c in cls]
xb = np.arange(3)
a2.bar(xb - w / 2, [float(v) for v in fb], w, color=OI["grey"], label="baseline (no tracking)")
a2.bar(xb + w / 2, [float(v) for v in fa], w, color=OI["green"], label="baseline + Hoop G")
for xv, v1, v2 in zip(xb, fb, fa):
    a2.text(xv - w / 2, float(v1) + 0.02, v1, ha="center", fontsize=9.5)
    a2.text(xv + w / 2, float(v2) + 0.02, v2, ha="center", fontsize=9.5)
a2.set_xticks(xb); a2.set_xticklabels([f"{c} held out\n(n={NUM[f'fn_hoop_z_press_eb_{c}']})" for c in cls])
a2.set_ylim(0, 0.85)
a2.set_ylabel("rank ρ, prediction vs actual pressure")
a2.set_title(f"Out of class, rank ρ better in {sum(float(b_) > float(a_) for a_, b_ in zip(fb, fa))} of 3 classes\n(pooled R² {NUM['r2b_hoop_z_press_eb']} → {NUM['r2a_hoop_z_press_eb']}, "
             f"permutation p = {NUM['perm_hoop_z_press_eb']})", fontsize=12)
a2.legend(frameon=False, fontsize=10, loc="upper left")
fig.tight_layout()
save(fig, "fig4_hoop_pressure.png")

# ---- Fig 5: 2025 draft-room card (trained on 2023-24 only)
c5 = pr.sort_values("pa").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(FIGW, 7.6))
yy = np.arange(len(c5))
ax.hlines(yy, 100 * c5.lo80, 100 * c5.hi80, color=OI["green"], alpha=0.3, lw=6)
for yv, r_ in zip(yy, c5.itertuples()):
    ax.annotate("", xy=(100 * r_.pa, yv), xytext=(100 * r_.pb, yv),
                arrowprops=dict(arrowstyle="->", color=OI["black"], lw=1.1))
ax.scatter(100 * c5.pb, yy, color="white", edgecolor=OI["grey"], s=45, zorder=3, label="baseline forecast")
ax.scatter(100 * c5.pa, yy, color=OI["green"], s=55, zorder=4, label="baseline + Hoop G forecast (80% band)")
ax.scatter(100 * c5.press_rate, yy, marker="x", color=OI["red"], s=55, zorder=5, label="actual 2025 pressure rate (raw)")
labels = []
for r_ in c5.itertuples():
    pk = "UDFA" if r_.pick == UDFA_PICK else f"#{int(r_.pick)}"
    tag = {"bend up": "  ▲ bend up", "bend down": "  ▼ bend down"}.get(r_.flag, "")
    labels.append(f"{r_.display_name} ({pk}, {r_.role[:3]}, {int(r_.rush_snaps)} rushes){tag}")
ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("pressure rate (%)")
ax.set_xlim(0, 13)
ax.legend(frameon=False, fontsize=10, loc="center left")
fig.suptitle(f"2025 DL board built from 2023–24 data only: Hoop G re-ranks {NUM['card_n_flag']} rushers by 3+ spots,\n"
             f"and {NUM['card_flag_closer']} of them move toward where they actually finished", fontsize=12.5, x=0.02, ha="left")
fig.tight_layout()
save(fig, "fig5_draft_card_2025.png")

# ----------------------------------------------------------------------------- write numbers
json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1, sort_keys=True)
json.dump(TABLES, open(OUT / "tables.json", "w"), indent=1, default=str)
print(json.dumps({k: NUM[k] for k in NUM if k.startswith(("pr_", "dr2_", "perm", "mde", "rel", "yoy", "rho_play",
                                                          "card", "slope", "tost", "holm", "r2"))}, indent=0))
