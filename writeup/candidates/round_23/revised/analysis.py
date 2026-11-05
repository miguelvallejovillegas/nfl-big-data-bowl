"""Round 23 revised: DL/EDGE Run-the-Hoop bend (Hoop G) vs NFL bend and pressure.

Run from the repo root:
    uv run python writeup/candidates/round_23/revised/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (every number the text quotes), board.csv,
multiplicity.csv, frozen_rule.json, tape_pull.csv and figures/*.png next to this file, runs verify.py (an independent
re-implementation of the headline slope on the same rusher set and baseline) and quotes its result, then renders
writeup.md from writeup_template.md ({{key}} -> numbers.json value; fails on an unknown key). Frame-level features
are cached in data/cache/r23c1 (gitignored). Runtime about 7 minutes.

New in the round 23 revision (judges' feedback on cand_1):
  * Headline baseline = pick + role (log pick, NFL edge share), the forecast already frozen for the 2026 test, so it
    was fixed before Hoop G was looked at. Every residual, the family screen, the day-3 test and the cards use it.
    A ridge-shrunk full no-tracking baseline (penalty chosen inside the training classes) is the one secondary.
  * Placebo: within-class shuffle of the whole drill-feature matrix (six drill measures plus five alternative Hoop G
    constructions: averaged turns, 0.3 / 0.7 s windows), null distribution of the best slope among all eleven.
    Construction fixed beforehand (harder turn, 0.5 s) as in frozen_rule.json; the averaged-turn variant is the
    primary sensitivity. Negative-control outcome: run-snap tackle rate through the identical pipeline.
  * Mediation: Hoop G -> observed NFL bend -> pressure residual, path coefficients fitted on two classes and applied
    to the third, class-stratified bootstrap; share of the Hoop G slope that runs through NFL bend.
  * EPA filter traced: 1,681 rusher rows -> distinct plays; a play is pressured if any cohort rusher was credited.
  * Fixed Monday rule and its expected count per class; cost of a missed hit in pressures and EPA.
  * Tape pull: clip IDs and tracking facts for the hardest-bend sampled rush of three named rushers.

Earlier rounds:
  * One rusher set (30+ NFL rushes, all six drill measures valid) and one fill rule (training-fold median for a
    missing 10-yd split / 3-cone) in both analysis.py and verify.py.
  * Three baselines side by side (pick only, pick + role, full no-tracking), plus a nested baseline chosen inside each
    training fold, each with the held-out R2 gain of Hoop G (class-stratified player bootstrap that refits every
    fold), and a repeated random-split check (200 random three-way splits).
  * Measurement-error simulation: true slope grid x one-lap reliability (with its bootstrap uncertainty) -> the
    distribution of the observed slope; implied true-slope range, attenuation, power for one lap and four laps.
  * Film check on sampled rushes: corner-turn rate (share of a lineman's sampled rushes whose hardest half-second is
    in the top quarter of all sampled rushes) by Hoop G third, and pressure on corner-turn vs other rushes.
  * Film list in Hoop G order with the NFL result beside each name, split into open-first and missed.
  * Sources of the EPA-per-pressure figure and the rush counts are written to numbers.json.

Design (one estimand per claim):
  * Drill metrics, z within Combine year, mean over attempts:
      Hoop G  = hardest 0.5 s of sideways (centripetal) acceleration on the Run-the-Hoop lap, g.
      Burst   = yards covered in the first second after movement onset in the pass-rush drill.
  * Game traits (random sampled NFL pass rushes only, self-referenced, no blockers or ball in the data):
      NFL get-off   = NGS season get-off (rush-weighted), sign flipped so higher = quicker.
      NFL burst     = yards covered in the first second after the snap.
      NFL bend      = hardest 0.5 s of sideways acceleration in the first 2.0 s after the snap, g.
  * Outcome: career pass-rush pressure rate 2023-25, empirical-Bayes shrunk, linemen with 30+ rushes.
  * Baseline without tracking: pick + role (headline); pick only, ridge-shrunk full, OLS full (log pick, Combine
    weight, NFL EDGE share, NGS athleticism score, official 10-yd split and 3-cone; missing times = training-fold
    median) and a nested choice made inside each training fold are reported in the appendix.
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
CACHE = ROOT / "data" / "cache" / "r23c1"
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
PB: dict = {}   # bootstrap two-sided p-values of every partial-Spearman link, for the multiplicity table


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
    b = np.asarray(b)
    PB[(x, y, len(d))] = float(min(1.0, 2 * min((1 + (b <= 0).sum()), (1 + (b >= 0).sum())) / (len(b) + 1)))
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


# ---- drill reliability as an evidence base for the rep prescription.
# (1) lap-to-lap ICC (few repeaters) with a player bootstrap; (2) a well-powered internal check: the hoop is a
# figure-eight, so every lap has a left-hand and a right-hand turn. Lean on one turn vs lean on the other, z within
# year, across all runners = split-half reliability of one lap (Spearman-Brown to full lap).
def hoop_turns():
    fr_ = pd.read_parquet(PROC / "combine_frames.parquet", filters=[("event_id", "in", list(hoop_att.event_id))],
                          columns=["event_id", "frame", "entity_type", "x", "y"])
    fr_ = fr_[fr_.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
    out_ = []
    for eid, g_ in fr_.groupby("event_id", sort=False):
        x_, y_ = g_.x.to_numpy(), g_.y.to_numpy()
        if len(x_) < 15:
            continue
        xs, ys = savgol_filter(x_, 7, 2), savgol_filter(y_, 7, 2)
        vx, vy = savgol_filter(xs, 7, 2, deriv=1, delta=DT), savgol_filter(ys, 7, 2, deriv=1, delta=DT)
        ax_, ay_ = savgol_filter(xs, 7, 2, deriv=2, delta=DT), savgol_filter(ys, 7, 2, deriv=2, delta=DT)
        sgn_ = (vx * ay_ - vy * ax_) / np.maximum(np.hypot(vx, vy), 0.5) / YD_PER_G
        lft = pd.Series(np.where(sgn_ > 0, sgn_, 0)).rolling(5, center=True).mean().max()
        rgt = pd.Series(np.where(sgn_ < 0, -sgn_, 0)).rolling(5, center=True).mean().max()
        out_.append({"event_id": eid, "left_g": float(lft), "right_g": float(rgt)})
    return pd.DataFrame(out_)


HT = cached("hoop_turns", hoop_turns).merge(hoop_att[["event_id", "nfl_id", "draft_year"]], on="event_id")
HT["lz"], HT["rz"] = yz(HT.left_g, HT.draft_year), yz(HT.right_g, HT.draft_year)
HTP = HT.groupby("nfl_id")[["lz", "rz"]].mean().reset_index()
_rr = np.random.default_rng(404)
r_lr = stats.spearmanr(HTP.lz, HTP.rz)[0]
_blr = [stats.spearmanr(*HTP.iloc[_rr.integers(0, len(HTP), len(HTP))][["lz", "rz"]].values.T)[0] for _ in range(NBOOT)]
REL_TURN = sb(r_lr, 2)
put("rel_turn_half", r_lr); put("rel_turn_n", len(HTP)); put("rel_turn_lap", REL_TURN)
put("rel_turn_lap_lo", sb(np.percentile(_blr, 5), 2)); put("rel_turn_lap_hi", sb(np.percentile(_blr, 95), 2))
BLR_LAP = np.array([sb(max(b_, 0.05), 2) for b_ in _blr])   # bootstrap draws of one-lap reliability
# left vs right asymmetry: Hoop G takes the harder of the two turns of the lap (max over the whole lap)
put("pct_left_harder", 100 * (HT.left_g > HT.right_g).mean(), 0)
put("turn_gap_g", (HT.left_g - HT.right_g).abs().median())
HT["mean_turn"] = (HT.left_g + HT.right_g) / 2


def icc_boot(df_, seed):
    d_ = df_.dropna(subset=["z"])
    ids_ = d_.groupby("nfl_id").size()
    rep = ids_[ids_ >= 2].index.values
    vals = {p_: d_.loc[d_.nfl_id == p_, "z"].to_numpy() for p_ in rep}
    rg_ = np.random.default_rng(seed)
    out_ = []
    for _ in range(NBOOT):
        pick_ = rg_.choice(rep, len(rep))
        ii = np.concatenate([[i_] * len(vals[p_]) for i_, p_ in enumerate(pick_)])
        vv = np.concatenate([vals[p_] for p_ in pick_])
        with np.errstate(all="ignore"):
            out_.append(icc(ii, vv)[0])
    return np.nanpercentile(out_, [5, 95])


ICC_CI = {"hoop": icc_boot(hoop_att, 405), "burst": icc_boot(rush_att.rename(columns={}), 406)}
put("rel_hoop1_lo", ICC_CI["hoop"][0]); put("rel_hoop1_hi", ICC_CI["hoop"][1])
put("rel_burst1_lo", ICC_CI["burst"][0]); put("rel_burst1_hi", ICC_CI["burst"][1])
REL_TARGET = 0.70


def reps_needed(r1, target=REL_TARGET):
    return int(np.ceil(target * (1 - r1) / (r1 * (1 - target))))


put("rel_target", REL_TARGET); put("reps_hoop", reps_needed(rel_hoop1)); put("reps_burst", reps_needed(rel_d1_1))
put("reps_hoop_turn", reps_needed(REL_TURN))
put("n_burst_one_rep", int((rush_att.dropna(subset=["d1"]).groupby("nfl_id").size() == 1).sum()))

# the pre-listed family: three measures per drill, all mean over attempts, z within Combine year
FAM = [("burst_z", "Rush drill: burst (yd in first 1 s)", rush_att, "d1", 1),
       ("rpa_z", "Rush drill: peak acceleration", rush_att, "peak_accel", 1),
       ("rts_z", "Rush drill: top speed", rush_att, "top_speed", 1),
       ("hoop_z", "Hoop: Hoop G (sideways lean)", hoop_att, "bend_g", 1),
       ("hts_z", "Hoop: top speed", hoop_att, "top_speed", 1),
       ("hlap_z", "Hoop: lap time (faster = higher)", hoop_att, "duration_s", -1)]
for key, _, src, col, sgn in FAM:
    pp = src.groupby("nfl_id").agg(v=(col, "mean"), draft_year=("draft_year", "first")).reset_index()
    pp[key] = sgn * yz(pp.v, pp.draft_year)
    dl = dl.merge(pp[["nfl_id", key]], on="nfl_id", how="left")
hoop_att["rad"] = hoop_att.speed_at_bend ** 2 / (hoop_att.bend_g * YD_PER_G)
hk = hoop_att.groupby("nfl_id").agg(hoop_g=("bend_g", "mean"), hoop_k=("bend_g", "size"),
                                    hoop_eid=("event_id", "first"), hoop_rad=("rad", "mean"),
                                    hoop_v=("speed_at_bend", "mean")).reset_index()
dl = dl.merge(hk, on="nfl_id", how="left")
_mt = HT.groupby("nfl_id").agg(v=("mean_turn", "mean"), draft_year=("draft_year", "first")).reset_index()
_mt["meanturn_z"] = yz(_mt.v, _mt.draft_year)
dl = dl.merge(_mt[["nfl_id", "meanturn_z"]], on="nfl_id", how="left")
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
# which linemen the rush count refers to: 8+ randomly sampled rushes that last the full 2.0 s window
put("n_pg", len(PG)); put("n_rush_2s", int(GR.g_bend.notna().sum()))
put("n_pg_co30", int(PG.nfl_id.isin(dl.loc[dl.rush_snaps >= RATE_MIN, "nfl_id"]).sum()))
put("median_random_rushes_co30", PG[PG.nfl_id.isin(dl.loc[dl.rush_snaps >= RATE_MIN, "nfl_id"])].g_n.median(), 0)
put("median_random_any", GR.groupby("nfl_id").size().median(), 0)

# ---- film check on sampled rushes. A "corner turn" is a sampled rush whose hardest half-second of sideways lean
# (first 2.0 s) is in the top quarter of all sampled rushes: the dip-and-flatten a scout looks for, self-referenced.
# Does a high Hoop G lineman turn the corner more often, and does a corner-turn rush end in pressure more often?
CT = float(GR.g_bend.quantile(0.75))
GR["corner"] = (GR.g_bend >= CT).astype(float).where(GR.g_bend.notna())
put("ct_g", CT)
_cr = GR.groupby("nfl_id").agg(corner=("corner", "mean"), c_k=("corner", "sum"), c_n=("corner", "count")).reset_index()
_cr = _cr[_cr.c_n >= 8]
dl = dl.merge(_cr, on="nfl_id", how="left")
_ct = dl.dropna(subset=["corner", "hoop_z"]).copy()
_ct["third"] = _ct.groupby("draft_year").hoop_z.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
_cg = [np.flatnonzero(_ct.draft_year.values == c) for c in classes]
_crng = np.random.default_rng(515)


def _ct_gap(d_):
    return 100 * (d_[d_.third == 2].corner.mean() - d_[d_.third == 0].corner.mean())


_cb = [_ct_gap(_ct.iloc[np.concatenate([_crng.choice(g_, len(g_)) for g_ in _cg])]) for _ in range(NBOOT)]
put("ct_n", len(_ct)); put("ct_top", 100 * _ct[_ct.third == 2].corner.mean(), 0)
put("ct_bot", 100 * _ct[_ct.third == 0].corner.mean(), 0); put("ct_gap", _ct_gap(_ct), 0)
put("ct_gap_lo", np.percentile(_cb, 5), 0); put("ct_gap_hi", np.percentile(_cb, 95), 0)
PB[("film", "ct_gap")] = float(min(1.0, 2 * min(1 + (np.asarray(_cb) <= 0).sum(), 1 + (np.asarray(_cb) >= 0).sum()) / (NBOOT + 1)))
# rush level: pressure on corner-turn rushes vs other rushes, player-cluster bootstrap
_rk = GR.dropna(subset=["corner"])
_rk = _rk[_rk.nfl_id.isin(_cr.nfl_id)]
_ids = _rk.nfl_id.unique()
_byp = {i: _rk[_rk.nfl_id == i][["corner", "press"]].values for i in _ids}


def _pgap(arr):
    c_ = arr[:, 0] == 1
    return 100 * (arr[c_, 1].mean() - arr[~c_, 1].mean())


_all = np.vstack([_byp[i] for i in _ids])
_pb = [_pgap(np.vstack([_byp[i] for i in _crng.choice(_ids, len(_ids))])) for _ in range(NBOOT)]
put("ct_rush_n", len(_all)); put("ct_press_corner", 100 * _all[_all[:, 0] == 1, 1].mean(), 0)
put("ct_press_other", 100 * _all[_all[:, 0] == 0, 1].mean(), 0)
put("ct_press_lo", np.percentile(_pb, 5), 0); put("ct_press_hi", np.percentile(_pb, 95), 0)
PB[("film", "ct_press")] = float(min(1.0, 2 * min(1 + (np.asarray(_pb) <= 0).sum(), 1 + (np.asarray(_pb) >= 0).sum()) / (NBOOT + 1)))


# ---- drill turn vs NFL rush path, in coach units. g = speed^2 / radius on both sides, at the half-second of
# hardest lean: the hoop lap, and each randomly sampled NFL rush (first 2.0 s). Turn Match = his NFL turn radius
# divided by his own hoop turn radius (1 = he turns as tight on Sunday as on the hoop).
def game_turn():
    keys = GR[GR.g_bend.notna()][K]
    gf_ = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "x", "y"],
                          filters=[("nfl_id", "in", list(set(keys.nfl_id)))])
    gf_ = gf_.merge(keys, on=K).merge(g[K + ["snap_frame"]], on=K).sort_values(K + ["frame"])
    out_ = []
    for key, h in gf_.groupby(K, sort=False):
        sf = h.snap_frame.iloc[0]
        w = h[(h.frame >= sf) & (h.frame <= sf + 20)]
        if len(w) != 21:
            continue
        _, roll, _, _, v = lateral_g(w.x.values, w.y.values)
        k_ = int(np.nanargmax(roll))
        out_.append(dict(zip(K, key), t_bend=float(np.nanmax(roll)), t_v=float(v[k_]), t_k=k_))
    return pd.DataFrame(out_)


GT = cached("game_turn", game_turn)
GT["t_rad"] = GT.t_v ** 2 / (GT.t_bend * YD_PER_G)
put("nfl_mph_median", GT.t_v.median() * 3600 / 1760, 1); put("nfl_radius_median", GT.t_rad.median(), 1)
put("nfl_bend_t_median", GT.t_k.median() / 10, 1)
_gt = GT.groupby("nfl_id").agg(nfl_rad=("t_rad", "median"), nfl_v=("t_v", "median"), t_cnt=("t_rad", "size"))
_gt = _gt[_gt.t_cnt >= 8].reset_index()
dl = dl.merge(_gt[["nfl_id", "nfl_rad", "nfl_v"]], on="nfl_id", how="left")
dl["turn_match"] = dl.nfl_rad / dl.hoop_rad
_tm = dl.dropna(subset=["turn_match"])
put("turn_match_median", _tm.turn_match.median(), 1); put("turn_match_n", len(_tm))
put("turn_match_p10", _tm.turn_match.quantile(0.1), 1); put("turn_match_p90", _tm.turn_match.quantile(0.9), 1)
# share of all sampled NFL rushes whose hardest turn is as tight as the same man's own hoop turn
_gm2 = GT.merge(dl[["nfl_id", "hoop_rad"]], on="nfl_id").dropna(subset=["hoop_rad"])
put("pct_rush_hooptight", 100 * (_gm2.t_rad <= _gm2.hoop_rad).mean(), 0)
dl["hoop_rad_tight"] = -dl.hoop_rad
dl["nfl_rad_tight"] = -dl.nfl_rad
_s2 = s[s.pass_rush_snaps >= 100].pivot_table(index="nfl_id", columns="season", values="avg_get_off")
_pp = pd.concat([pd.DataFrame({"a": _s2[a_], "b": _s2[b_]}).dropna() for a_, b_ in [(2023, 2024), (2024, 2025)]])
REL["quick"] = stats.spearmanr(_pp.a, _pp.b)[0]
put("rel_getoff_season", REL["quick"]); put("rel_getoff_n", len(_pp))
_s3 = s[s.pass_rush_snaps >= 100].assign(pr=lambda d_: d_.pressures / d_.pass_rush_snaps).pivot_table(
    index="nfl_id", columns="season", values="pr")
_pq = pd.concat([pd.DataFrame({"a": _s3[a_], "b": _s3[b_]}).dropna() for a_, b_ in [(2023, 2024), (2024, 2025)]])
put("rel_press_season", stats.spearmanr(_pq.a, _pq.b)[0]); put("rel_press_n", len(_pq))
# EPA of a pressured snap (random sampled snaps)
# trace: every random-sample DL rusher row -> distinct plays (two cohort rushers can share a play); a play counts as
# pressured if ANY cohort rusher on it was credited (time_to_pressure non-null or sack). Round 22 kept the first
# rusher row per play, which dropped pressures credited to the second rusher.
_rows = g[g.sample_kind == "random"].copy()
_rows["pr"] = _rows.time_to_pressure.notna() | (_rows.sack == 1)
put("epa_rows", len(_rows)); put("epa_rows_press", int(_rows.pr.sum()))
_first = _rows.drop_duplicates(["game_id", "play_id"])
put("epa_first_press", int(_first.pr.sum()))
_gm = _rows.groupby(["game_id", "play_id"]).agg(pr=("pr", "max"), expected_points_added=("expected_points_added",
                                                                                           "first")).reset_index()
put("epa_shared", int(len(_rows) - len(_gm)))
_gm = _gm[_gm.expected_points_added.notna()]
_pr = _gm.pr.values
e1, e0 = _gm[_pr].expected_points_added.values, _gm[~_pr].expected_points_added.values
_r = np.random.default_rng(101)
_eb = [_r.choice(e0, len(e0)).mean() - _r.choice(e1, len(e1)).mean() for _ in range(NBOOT)]
EPA = e0.mean() - e1.mean()
put("epa_press", EPA); put("epa_press_lo", np.percentile(_eb, 5)); put("epa_press_hi", np.percentile(_eb, 95))
put("epa_n", len(_gm))
# source: play-level time_to_pressure (non-null = the rusher was credited with a pressure) or sack == 1 on the
# randomly sampled DL plays, nflverse expected_points_added; pressure_allowed is a blocker column (empty for DL rows)
put("epa_n_press", int(_pr.sum())); put("pct_dl_pressure_allowed_null", 100 * g.pressure_allowed.isna().mean(), 0)
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
_r = boot_psp(dl, "hoop_rad_tight", "nfl_rad_tight", Zc, 19)
put("tr_rad", _r[0]); put("tr_rad_lo", _r[1]); put("tr_rad_hi", _r[2]); put("tr_rad_n", _r[3])
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


def fill_times(tr, te, how="median"):
    tr, te = tr.copy(), te.copy()
    for c in TIMES:
        mk = tr[c].notna()
        if how == "median":       # the one fill rule (also verify.py's): training-fold median
            med = tr.loc[mk, c].median()
            med = 0.0 if pd.isna(med) else med     # a training class with no times at all: year-z mean
            tr[c], te[c] = tr[c].fillna(med), te[c].fillna(med)
            continue
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


ALPHAS = np.r_[1e-4, np.logspace(-1, 2.5, 8)]   # ridge penalty, in units of training rushers


def ridge(X, yv, Xt, alpha):
    mu_, sd_ = X.mean(0), X.std(0) + 1e-9
    Xs, Xts = (X - mu_) / sd_, (Xt - mu_) / sd_
    b = np.linalg.solve(Xs.T @ Xs + alpha * len(Xs) * np.eye(X.shape[1]), Xs.T @ (yv - yv.mean()))
    return yv.mean() + Xts @ b


def ridge_cv(tr, te, cols):
    """penalty chosen inside the training classes only: fit on one training class, score on the other."""
    X, yv, tcl = tr[cols].values, tr.press_eb.values, tr.draft_year.values
    us = np.unique(tcl)
    err = []
    for a_ in ALPHAS:
        e_ = 0.0
        for c in us:
            m_ = tcl == c
            e_ += ((yv[m_] - ridge(X[~m_], yv[~m_], X[m_], a_)) ** 2).sum()
        err.append(e_)
    return ridge(X, yv, te[cols].values, ALPHAS[int(np.argmin(err))])


def loco(d, cols, kind="ols", how="median"):
    pred = np.full(len(d), np.nan)
    cl = d.draft_year.values
    for c in classes:
        tr, te = fill_times(d[cl != c], d[cl == c], how)
        pred[cl == c] = ols(tr, te, cols)[0] if kind == "ols" else ridge_cv(tr, te, cols)
    return pred


def oos(y, p):
    sse = ((y - p) ** 2).sum()
    return 1 - sse / ((y - y.mean()) ** 2).sum(), np.sqrt(sse / len(y)), stats.spearmanr(y, p)[0]


FKEYS = [f[0] for f in FAM]
AN = co30.dropna(subset=FKEYS).reset_index(drop=True)
put("n_an", len(AN))
NUM["ho_n_per_class"] = "/".join(str(int((AN.draft_year == c).sum())) for c in classes)
BASES = {"pick1": ["log_pick"], "pick": ["log_pick", "edge_share"], "base": BASE}
BLAB = {"pick1": "Pick only", "pick": "Pick + role", "base": "Full no-tracking", "nest": "Chosen inside each fold"}


def loco_nested(d, extra=(), how="median"):
    """outer leave-one-class-out; inside the two training classes, pick the baseline (of the three) that forecasts
    one training class best from the other, then fit it (+ extra columns) on both training classes."""
    pred = np.full(len(d), np.nan)
    cl = d.draft_year.values
    chosen = []
    for c in classes:
        tr0 = d[cl != c]
        tcl = tr0.draft_year.values
        err = {}
        for bk, cols in BASES.items():
            e_ = 0.0
            for ic in np.unique(tcl):
                a_, b_ = fill_times(tr0[tcl != ic], tr0[tcl == ic], how)
                e_ += ((b_.press_eb.values - ols(a_, b_, cols)[0]) ** 2).sum()
            err[bk] = e_
        bk = min(err, key=err.get)
        chosen.append(bk)
        tr, te = fill_times(tr0, d[cl == c], how)
        pred[cl == c] = ols(tr, te, BASES[bk] + list(extra))[0]
    return pred, chosen


HB = "pick"     # headline baseline: pick + role, the forecast frozen for the 2026 test before Hoop G was examined
MODELS = [(bk, BASES[bk], "ols") for bk in BASES] + [(bk + "_h", BASES[bk] + ["hoop_z"], "ols") for bk in BASES] \
    + [("ridge", BASE, "ridge"), ("ridge_h", BASE + ["hoop_z"], "ridge"),
       ("burst_h", BASES[HB] + ["burst_z"], "ols"), ("nest", [], "nest"), ("nest_h", ["hoop_z"], "nest")]


def predict(d, cols, mode):
    if mode == "nest":
        return loco_nested(d, cols)[0]
    return loco(d, cols, kind=mode)


y = AN.press_eb.values
PRED = {nm: predict(AN, cols, nz) for nm, cols, nz in MODELS}
NEST_CHOSEN = loco_nested(AN)[1]
TXT_NEST = {c: BLAB[k] for c, k in zip(classes, NEST_CHOSEN)}
RES = y - PRED[HB]
AN["fc"] = PRED[HB]
grp = [np.flatnonzero(AN.draft_year.values == c) for c in classes]
rng = np.random.default_rng(2027)
BOOT = {nm: [] for nm, *_ in MODELS}
BRES = {k: [] for k in FKEYS}
BSL = {k: [] for k in ["pick1", "pick", "base", "ridge", "nest"]}   # Hoop G slope on each baseline's residual
BX = {k: [] for k in ["cls2023", "cls2024", "cls2025", "EDGE", "interior", "meanturn"]}
AN["rel_i"] = sb(REL_TURN, AN.hoop_k)          # reliability of each lineman's own Hoop G (one or two laps)
NEST_B = []
for _ in range(NBOOT):
    ix = np.concatenate([rng.choice(g_, len(g_)) for g_ in grp])
    d_ = AN.iloc[ix].reset_index(drop=True)
    yb = d_.press_eb.values
    pb_ = {}
    for nm, cols, nz in MODELS:
        if nm == "nest":
            p_, ch_ = loco_nested(d_)
            NEST_B.append(ch_)
        else:
            p_ = predict(d_, cols, nz)
        pb_[nm] = p_
        BOOT[nm].append(oos(yb, p_)[0])
    for bk in BSL:
        BSL[bk].append(np.polyfit(d_.hoop_z.values, yb - pb_[bk], 1)[0])
    rb = yb - pb_[HB]
    for k in FKEYS:
        BRES[k].append(np.polyfit(d_[k].values, rb, 1)[0])
    for c in classes:
        mc = d_.draft_year.values == c
        BX[f"cls{c}"].append(np.polyfit(d_.hoop_z.values[mc], rb[mc], 1)[0])
    for rl in ("EDGE", "interior"):
        mr = d_.role.values == rl
        BX[rl].append(np.polyfit(d_.hoop_z.values[mr], rb[mr], 1)[0])
    BX["meanturn"].append(np.polyfit(d_.meanturn_z.values, rb, 1)[0])
for nm, *_ in MODELS:
    r2, rmse, rho = oos(y, PRED[nm])
    put(f"r2_{nm}", r2); put(f"rmse_{nm}", rmse)
    ref = nm[:-2] if nm.endswith("_h") else None
    if ref:
        ref = HB if nm == "burst_h" else ref
        dd = np.array(BOOT[nm]) - np.array(BOOT[ref])
        put(f"dr2_{nm}", r2 - oos(y, PRED[ref])[0])
        put(f"dr2_{nm}_lo", np.percentile(dd, 5)); put(f"dr2_{nm}_hi", np.percentile(dd, 95))
        PB[("dr2", nm)] = float(min(1.0, 2 * min(1 + (dd <= 0).sum(), 1 + (dd >= 0).sum()) / (len(dd) + 1)))
        if nm == HB + "_h":   # Bonferroni across the six pre-listed drill measures: 1 - 0.10/6 two-sided interval
            put("dr2_hb_alo", np.percentile(dd, 100 * 0.05 / 6)); put("dr2_hb_ahi", np.percentile(dd, 100 - 100 * 0.05 / 6))
            put("dr2_hb_pneg", 100 * (dd <= 0).mean(), 0)
for bk in BSL:
    est_ = np.polyfit(AN.hoop_z.values, y - PRED[bk], 1)[0]
    put(f"bs_{bk}", est_ * PS, 1); put(f"bs_{bk}_lo", np.percentile(BSL[bk], 5) * PS, 1)
    put(f"bs_{bk}_hi", np.percentile(BSL[bk], 95) * PS, 1)
    PB[("slope", "bs_" + bk)] = float(min(1.0, 2 * min(1 + (np.asarray(BSL[bk]) <= 0).sum(),
                                                       1 + (np.asarray(BSL[bk]) >= 0).sum()) / (NBOOT + 1)))
put("sd_y", float(np.std(y)))
_nb = pd.Series([tuple(c_) for c_ in NEST_B]).explode()
put("nest_pct_pickrole", 100 * (_nb == "pick").mean(), 0)
put("nest_pct_full", 100 * (_nb == "base").mean(), 0)
# regression fill instead of the median (round 22's main choice), same set
_rls = y - loco(AN, BASE, how="ls")
put("ls_fill_hoop_season", np.polyfit(AN.hoop_z.values, _rls, 1)[0] * PS, 1)

# repeated random splits: 200 random three-way splits that ignore class (drill z stays within Combine year)
_rr2 = np.random.default_rng(31)
RSPL = []
for _ in range(200):
    fold = _rr2.permutation(np.arange(len(AN)) % 3)
    pr_ = np.full(len(AN), np.nan)
    for f_ in range(3):
        tr, te = fill_times(AN[fold != f_], AN[fold == f_])
        pr_[fold == f_] = ols(tr, te, BASES[HB])[0]
    RSPL.append(np.polyfit(AN.hoop_z.values, y - pr_, 1)[0] * PS)
RSPL = np.asarray(RSPL)
put("rsplit_med", np.median(RSPL), 1); put("rsplit_lo", np.percentile(RSPL, 5), 1)
put("rsplit_hi", np.percentile(RSPL, 95), 1); put("rsplit_pos", 100 * (RSPL > 0).mean(), 0)

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
_bh = HO["hoop_z"]["bs"]
put("ho_hoop_season_alo", np.percentile(_bh, 100 * 0.05 / 6) * PS, 1)
put("ho_hoop_season_ahi", np.percentile(_bh, 100 - 100 * 0.05 / 6) * PS, 1)


def put_slope(tag, est, draws):
    draws = np.asarray(draws)
    put(f"{tag}", est * PS, 1); put(f"{tag}_lo", np.percentile(draws, 5) * PS, 1)
    put(f"{tag}_hi", np.percentile(draws, 95) * PS, 1)
    PB[("slope", tag)] = float(min(1.0, 2 * min(1 + (draws <= 0).sum(), 1 + (draws >= 0).sum()) / (len(draws) + 1)))


# per-class, with a within-class player bootstrap
for j_, c in enumerate(classes):
    put(f"cls{c}_lo", np.percentile(BX[f"cls{c}"], 5) * PS, 1); put(f"cls{c}_hi", np.percentile(BX[f"cls{c}"], 95) * PS, 1)
put("cls2025_ppos", 100 * (np.asarray(BX["cls2025"]) > 0).mean(), 0)
# edge vs interior: same residual, separate slopes (an edge and a 300-lb 3-technique are not held to one bend line)
for rl in ("EDGE", "interior"):
    mr = AN.role.values == rl
    put_slope(f"role_{rl}", np.polyfit(AN.hoop_z.values[mr], RES[mr], 1)[0], BX[rl])
    put(f"role_{rl}_n", int(mr.sum()))
# mean of the two turns instead of the harder one (left/right asymmetry robustness)
put_slope("mt_hoop", np.polyfit(AN.meanturn_z.values, RES, 1)[0], BX["meanturn"])

# ---- measurement-error simulation: how much of the weak result is one-lap noise?
# True bend T ~ N(0,1); a lineman's Hoop G = sqrt(lam_i) T + sqrt(1 - lam_i) e, then standardized, where lam_i is the
# reliability of his own Hoop G (one lap or two), and the one-lap reliability is redrawn every simulation from its
# player bootstrap. Outcome residual = beta T + noise with the observed residual variance. For each true slope beta
# (per sd of true bend) we get the distribution of the slope we would observe at n = 77; the implied true-slope
# range is every beta whose 5-95% band covers the observed slope.
B_OBS = float(HO["hoop_z"]["est"])
VR = float(np.var(RES))
KK = AN.hoop_k.values
NME = len(AN)
me_rng = np.random.default_rng(4343)
BETA = np.round(np.arange(-1.0, 3.01, 0.05), 2)     # rate points per sd of true bend
NSIM = 3000


def sim_obs(beta, laps=None):
    lam1 = BLR_LAP[me_rng.integers(0, len(BLR_LAP), NSIM)]
    k_ = KK if laps is None else np.full(NME, laps)
    lam = sb(lam1[:, None], k_[None, :])
    T = me_rng.standard_normal((NSIM, NME))
    X = np.sqrt(lam) * T + np.sqrt(1 - lam) * me_rng.standard_normal((NSIM, NME))
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    yv = beta * T + np.sqrt(max(VR - beta ** 2, 0.2 * VR)) * me_rng.standard_normal((NSIM, NME))
    yv = yv - yv.mean(1, keepdims=True)
    bh = (X * yv).sum(1) / (X ** 2).sum(1)
    se = np.sqrt(((yv - bh[:, None] * X) ** 2).sum(1) / (NME - 2) / (X ** 2).sum(1))
    return bh, bh / se


MEQ = []
for be in BETA:
    bh_, _ = sim_obs(be)
    MEQ.append(np.percentile(bh_, [5, 50, 95]))
MEQ = np.array(MEQ)
_cov = BETA[(MEQ[:, 0] <= B_OBS) & (MEQ[:, 2] >= B_OBS)]
_nz = BETA > 0.2
ATT = float(np.median(MEQ[_nz, 1] / BETA[_nz]))          # share of the true slope one lap shows, typical
B_TRUE = B_OBS / ATT
put("me_att", 100 * ATT, 0); put("me_true", B_TRUE * PS, 1)
put("me_lo", _cov.min() * PS, 1); put("me_hi", _cov.max() * PS, 1)
TCRIT = stats.t.ppf(0.90, NME - 2)
_b1, _t1 = sim_obs(B_TRUE)
_b4, _t4 = sim_obs(B_TRUE, laps=4)
put("me_pow1", 100 * (_t1 > TCRIT).mean(), 0); put("me_pow4", 100 * (_t4 > TCRIT).mean(), 0)
put("me_exp4", np.median(_b4) * PS, 1); put("me_exp1", np.median(_b1) * PS, 1)
_b0, _ = sim_obs(0.0)
put("me_null_exceed", 100 * (_b0 >= B_OBS).mean(), 0)
put("lam_hoop", float(AN.rel_i.mean()))
put("me_true_obs", B_OBS * PS / ATT, 1)   # shrinkage-corrected slope (observed / attenuation), secondary line only


# ---- placebo: the whole drill-feature matrix shuffled within class. Hoop G's construction was fixed before this run
# (harder of the two turns, 0.5 s window, as in frozen_rule.json); five alternative constructions that an analyst
# could have picked instead (averaged turns; 0.3 s and 0.7 s windows) join the five other drill measures, and the
# null is the best slope among all eleven when no drill measure carries signal.
def hoop_variants():
    fr_ = pd.read_parquet(PROC / "combine_frames.parquet", filters=[("event_id", "in", list(hoop_att.event_id))],
                          columns=["event_id", "frame", "entity_type", "x", "y"])
    fr_ = fr_[fr_.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
    out_ = []
    for eid, g_ in fr_.groupby("event_id", sort=False):
        x_, y_ = g_.x.to_numpy(), g_.y.to_numpy()
        if len(x_) < 15:
            continue
        xs, ys = savgol_filter(x_, 7, 2), savgol_filter(y_, 7, 2)
        vx, vy = savgol_filter(xs, 7, 2, deriv=1, delta=DT), savgol_filter(ys, 7, 2, deriv=1, delta=DT)
        ax_, ay_ = savgol_filter(xs, 7, 2, deriv=2, delta=DT), savgol_filter(ys, 7, 2, deriv=2, delta=DT)
        sg = (vx * ay_ - vy * ax_) / np.maximum(np.hypot(vx, vy), 0.5) / YD_PER_G
        row = {"event_id": eid}
        for w in (3, 5, 7):
            row[f"hard_w{w}"] = float(pd.Series(np.abs(sg)).rolling(w, center=True).mean().max())
            lf = pd.Series(np.where(sg > 0, sg, 0)).rolling(w, center=True).mean().max()
            rt = pd.Series(np.where(sg < 0, -sg, 0)).rolling(w, center=True).mean().max()
            row[f"mean_w{w}"] = float((lf + rt) / 2)
        out_.append(row)
    return pd.DataFrame(out_)


HV = cached("hoop_variants", hoop_variants).merge(hoop_att[["event_id", "nfl_id", "draft_year"]], on="event_id")
VCOLS = ["hard_w3", "hard_w7", "mean_w3", "mean_w5", "mean_w7"]
_hvp = HV.groupby("nfl_id").agg({**{c: "mean" for c in VCOLS}, "draft_year": "first"}).reset_index()
for c in VCOLS:
    _hvp[c + "_z"] = yz(_hvp[c], _hvp.draft_year)
AN = AN.merge(_hvp[["nfl_id"] + [c + "_z" for c in VCOLS]], on="nfl_id", how="left")
for c in VCOLS:
    AN[c + "_z"] = AN[c + "_z"].fillna(0.0)
VLAB = {"hard_w3": "harder turn, 0.3 s", "hard_w7": "harder turn, 0.7 s", "mean_w3": "both turns, 0.3 s",
        "mean_w5": "both turns, 0.5 s", "mean_w7": "both turns, 0.7 s"}
XALL = np.c_[AN[FKEYS].values, AN[[c + "_z" for c in VCOLS]].values]   # 6 drill measures + 5 Hoop G variants
obs_all = slopes(XALL, RES)
for j_, c in enumerate(VCOLS):
    put(f"var_{c}", obs_all[len(FKEYS) + j_] * PS, 1)
put("var_min", obs_all[len(FKEYS):].min() * PS, 1); put("var_max", obs_all[len(FKEYS):].max() * PS, 1)
pl_rng = np.random.default_rng(1212)
NPL = 5000
PMAX = np.empty(NPL)
PSING = np.empty(NPL)
jh = FKEYS.index("hoop_z")
for i_ in range(NPL):
    ix = np.arange(len(AN))
    for g_ in grp:
        ix[g_] = pl_rng.permutation(g_)
    s_ = slopes(XALL[ix], RES)
    PMAX[i_] = np.abs(s_).max()
    PSING[i_] = s_[jh]
put("pl_n_meas", XALL.shape[1])
put("pl_single95", np.percentile(PSING, 95) * PS, 1)
put("pl_max95", np.percentile(PMAX, 95) * PS, 1); put("pl_max90", np.percentile(PMAX, 90) * PS, 1)
put("pl_max_med", np.median(PMAX) * PS, 1)
put("pl_p_single", (1 + (PSING >= B_OBS).sum()) / (NPL + 1))
put("pl_p_max", (1 + (PMAX >= abs(B_OBS)).sum()) / (NPL + 1))
put("pl_pct_reach", 100 * (PMAX >= abs(B_OBS)).mean(), 0)
PLACEBO = dict(single=PSING * PS, best=PMAX * PS)


# ---- negative control outcome: run-snap tackle rate (tackles + assists per 100 non-rush snaps) through the identical
# pipeline (same 77 rushers, same held-out pick + role baseline). Bend on a figure-eight has no reason to show up here.
_sr = s.groupby("nfl_id").agg(plays=("plays", "sum"), prs=("pass_rush_snaps", "sum"), tk=("tackles", "sum"),
                              asst=("assists", "sum")).reset_index()
_sr["run_snaps"] = _sr.plays - _sr.prs
_sr["run_tk"] = 100 * (_sr.tk + _sr.asst) / _sr.run_snaps.where(_sr.run_snaps >= 50)
AN = AN.merge(_sr[["nfl_id", "run_tk"]], on="nfl_id", how="left")


def loco_y(d, cols, yc):
    pred = np.full(len(d), np.nan)
    cl = d.draft_year.values
    for c in classes:
        tr, te = d[cl != c], d[cl == c]
        pred[cl == c] = ols(tr, te, cols, y=yc)[0]
    return pred


NC = AN.dropna(subset=["run_tk"]).reset_index(drop=True)
_ncg = [np.flatnonzero(NC.draft_year.values == c) for c in classes]


def nc_slope(d_, yc, extra=()):
    r_ = d_[yc].values - loco_y(d_, BASES[HB] + list(extra), yc)
    return np.polyfit(d_.hoop_z.values, r_, 1)[0] / d_[yc].std()


NC["combine_weight_z"] = NC.combine_weight_z.fillna(0.0)
NCV = [("nc", "run_tk", ()), ("nc_press", "press_eb", ()), ("ncw", "run_tk", ("combine_weight_z",)),
       ("ncw_press", "press_eb", ("combine_weight_z",))]
nc_rng = np.random.default_rng(616)
NCBS = {k: [] for k, *_ in NCV}
for _ in range(NBOOT):
    db = NC.iloc[np.concatenate([nc_rng.choice(g_, len(g_)) for g_ in _ncg])].reset_index(drop=True)
    for k, yc, ex in NCV:
        NCBS[k].append(nc_slope(db, yc, ex))
put("nc_n", len(NC))
for k, yc, ex in NCV:
    put(f"{k}_sd", nc_slope(NC, yc, ex)); put(f"{k}_sd_lo", np.percentile(NCBS[k], 5))
    put(f"{k}_sd_hi", np.percentile(NCBS[k], 95))
put("rho_hoop_weight", stats.spearmanr(NC.hoop_z, NC.combine_weight_z)[0])
put("nc_run_mean", NC.run_tk.mean(), 1)


# ---- mediation: Hoop G -> observed NFL bend -> pressure beyond pick + role. a = Hoop G -> NFL bend (sd) given weight
# and edge share; b = NFL bend -> held-out residual given Hoop G; c' = direct. Paths fitted on two classes and
# applied to the third (class-weighted average of the three out-of-class estimates); class-stratified bootstrap.
MD = AN.dropna(subset=["g_bend"]).reset_index(drop=True)
MD["res"] = RES[AN.g_bend.notna().values]
MD["bendz"] = (MD.g_bend - MD.g_bend.mean()) / MD.g_bend.std()
_mdg = [np.flatnonzero(MD.draft_year.values == c) for c in classes]


def paths(d_):
    A1 = np.c_[np.ones(len(d_)), d_.hoop_z, d_.combine_weight_z.fillna(0), d_.edge_share]
    a_ = np.linalg.lstsq(A1, d_.bendz.values, rcond=None)[0][1]
    A2 = np.c_[np.ones(len(d_)), d_.hoop_z, d_.bendz]
    _, cp_, b_ = np.linalg.lstsq(A2, d_.res.values, rcond=None)[0]
    tot_ = np.polyfit(d_.hoop_z.values, d_.res.values, 1)[0]
    return a_, b_, cp_, tot_


def med_heldout(d_):
    out_ = np.zeros(4)
    cl_ = d_.draft_year.values
    for c in classes:
        out_ += np.array(paths(d_[cl_ != c])) * (cl_ == c).mean()
    return out_


MED = med_heldout(MD)
MEDB = []
md_rng = np.random.default_rng(717)
for _ in range(NBOOT):
    db = MD.iloc[np.concatenate([md_rng.choice(g_, len(g_)) for g_ in _mdg])].reset_index(drop=True)
    MEDB.append(med_heldout(db))
MEDB = np.array(MEDB)
_ind, _indb = MED[0] * MED[1], MEDB[:, 0] * MEDB[:, 1]
_shr = _ind / (_ind + MED[2])
_shrb = _indb / (_indb + MEDB[:, 2])
put("md_n", len(MD))
put("md_a", MED[0]); put("md_a_lo", np.percentile(MEDB[:, 0], 5)); put("md_a_hi", np.percentile(MEDB[:, 0], 95))
put("md_b", MED[1] * PS, 1); put("md_b_lo", np.percentile(MEDB[:, 1], 5) * PS, 1)
put("md_b_hi", np.percentile(MEDB[:, 1], 95) * PS, 1)
put("md_ind", _ind * PS, 1); put("md_ind_lo", np.percentile(_indb, 5) * PS, 1); put("md_ind_hi", np.percentile(_indb, 95) * PS, 1)
put("md_dir", MED[2] * PS, 1); put("md_dir_lo", np.percentile(MEDB[:, 2], 5) * PS, 1)
put("md_dir_hi", np.percentile(MEDB[:, 2], 95) * PS, 1)
put("md_share", 100 * _shr, 0); put("md_share_lo", 100 * np.percentile(_shrb, 5), 0)
put("md_share_hi", 100 * np.percentile(_shrb, 95), 0)

# leave-one-player-out: drop each rusher, refit the held-out baseline, re-estimate (overall and 2025 class)
LO = []
for i_ in range(len(AN)):
    d_ = AN.drop(index=i_).reset_index(drop=True)
    r_ = d_.press_eb.values - loco(d_, BASES[HB])
    m25 = d_.draft_year.values == 2025
    LO.append((np.polyfit(d_.hoop_z, r_, 1)[0] * PS, np.polyfit(d_.hoop_z[m25], r_[m25], 1)[0] * PS))
LO = np.array(LO)
put("lopo_min", LO[:, 0].min(), 1); put("lopo_max", LO[:, 0].max(), 1)
put("lopo25_min", LO[:, 1].min(), 1); put("lopo25_max", LO[:, 1].max(), 1)
put("lopo25_npos", int((LO[:, 1] > 0).sum()))
_i25 = int(np.argmax(np.abs(LO[:, 1] - HO["hoop_z"]["cls"][2] * PS)))
TXT_LOPO = AN.iloc[_i25].display_name

# one rusher set, one fill rule: verify.py uses the same set (all six drill measures valid) and the median fill.
# Sensitivity: add the one 30+ rush lineman with a hoop lap and a pass-rush clip but no valid burst rep.
RUSH_IDS = set(ev[ev.drill_name == "PASS_RUSH_DRILL"].nfl_id)
AN78 = co30[co30.hoop_z.notna() & co30.nfl_id.isin(RUSH_IDS)].reset_index(drop=True)
put("n_an78", len(AN78))
put("rec_med78", np.polyfit(AN78.hoop_z, AN78.press_eb.values - loco(AN78, BASES[HB]), 1)[0] * PS, 1)
TXT_EXTRA = ", ".join(AN78[~AN78.nfl_id.isin(AN.nfl_id)].display_name)

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
    res_ = an_.press_eb.values - loco(an_, BASES[HB])
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
    dd = dd[~np.isnan(dd)]
    PB[("day3", t)] = float(min(1.0, 2 * min(1 + (dd <= 0).sum(), 1 + (dd >= 0).sum()) / (len(dd) + 1)))
_ka, _na = top_third_hit(D3, "fc_hoop", g3)
_kb, _nb = top_third_hit(D3, "neg_pick", g3)
put("hg_gap", 100 * (_ka / _na - _kb / _nb), 0)
put("hg_gap_lo", np.percentile(HITGAP, 5), 0); put("hg_gap_hi", np.percentile(HITGAP, 95), 0)
put("hg_bar", BAR_GAP)
_hg = np.asarray(HITGAP)
PB[("day3", "hitgap")] = float(min(1.0, 2 * min(1 + (_hg <= 0).sum(), 1 + (_hg >= 0).sum()) / (len(_hg) + 1)))
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
# ---- the fixed Monday rule: pick >= 103 or undrafted AND top third of Hoop G among his Combine class's DL who ran
# the hoop. Expected count per class from 2023-25; outcome of each flag among held-out day-3 rushers.
_bx = dl[dl.hoop_z.notna()].copy()
_bx["third"] = _bx.groupby("draft_year").hoop_z.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
_bx["rule"] = (_bx.pick >= DAY3) & (_bx.third == 2)
_rc = _bx.groupby("draft_year").rule.sum()
put("rule_mean", _rc.mean(), 1); put("rule_min", int(_rc.min())); put("rule_max", int(_rc.max()))
put("rule_hoop_per_class", _bx.groupby("draft_year").size().mean(), 0)
put("rule_30", 100 * _bx[_bx.rule].nfl_id.isin(co30.nfl_id).mean(), 0)
D3["over"] = (D3.press_eb - D3.fc)
_mh = D3[~D3.flag & (D3.over >= MARGIN)]
_fh = D3[D3.flag & (D3.over >= MARGIN)]
_ff = D3[D3.flag & (D3.over <= -MARGIN)]
put("miss_n", len(_mh)); put("miss_over", _mh.over.mean() * PS, 1); put("miss_epa", _mh.over.mean() * PS * EPA, 1)
put("flag_hit_n", len(_fh)); put("flag_false_n", len(_ff))
put("noflag_n", int((~D3.flag).sum()))
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
    sc.loc[tef.index, "fc"] = ols(trf, tef, BASES[HB])[0]
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

frozen = {"metric": "Hoop G: hardest 0.5 s of sideways acceleration on Run-the-Hoop, z within Combine year, mean of "
                    "graded laps",
          "use": f"film flag for day-3/undrafted (pick>={DAY3}) rushers in the class top third of Hoop G; added to the "
                 "film queue beside pick order, never ahead of it; never moves the board",
          "primary_test": "2026 Combine class after Week 18 of its first NFL season (30+ rushes): slope of Hoop G on "
                          "(career pressure rate - pick + role forecast fitted on the 2023-25 classes and frozen here), "
                          "within-class permutation, one-sided p < 0.10 keeps the flag; slope <= 0 retires it",
          "secondary": "same slope against pick only and against the full no-tracking forecast; reported, not decisive",
          "promotion_to_board": "only if, pooled over 2026 and 2027 classes, the day-3 top third by forecast + Hoop G "
                                "beats pick order's top third by 20+ points of hit rate",
          "calls_2026_board": {r_.display_name: r_.call for _, r_ in brd.iterrows()},
          "decision_date": "after Week 18 of the 2026 regular season"}
blob = json.dumps(frozen, sort_keys=True).encode()
HASH = hashlib.sha256(blob).hexdigest()[:12]
(OUT / "frozen_rule.json").write_text(json.dumps({**frozen, "sha256": hashlib.sha256(blob).hexdigest()}, indent=1))
TXT["hash"] = HASH

# ============================================================================= figures
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
C_H, C_B, C_N = OI["orange"], OI["blue"], OI["grey"]
C_D = "#4D4D4D"
MS = MARGIN * PS

# Fig 3: the six pre-listed measures against the headline (pick + role) forecast. One highlight colour (Hoop G).
# The chance line is the placebo: best |slope| among 11 shuffled measures (six drills + five Hoop G constructions).
CHANCE = float(NUM["pl_max95"])
fig, ax = plt.subplots(figsize=(FIGW, 6.6))
ordr = list(range(len(FAMR)))[::-1]
for yy, j in enumerate(ordr):
    f = FAMR[j]
    hi_ = f["key"] == "hoop_z"
    col = C_H if hi_ else C_D
    ax.plot([f["lo"] * PS, f["hi"] * PS], [yy, yy], color=col, lw=6 if hi_ else 3.5, solid_capstyle="butt")
    for c, v in zip(classes, f["cls"]):
        ax.plot(v * PS, yy - 0.28, marker="D", color="white", mec="#333", ms=6.5)
    ax.plot(f["est"] * PS, yy, "o", color="white", mec=col, mew=3, ms=12, zorder=5)
    wt = "bold" if hi_ else "normal"
    p11 = (1 + (PMAX >= abs(f["est"])).sum()) / (NPL + 1)
    f["p11"] = p11
    for xx, txt in [(1.03, fm(f["est"] * PS)), (1.23, f"{f['p2']:.2f}"), (1.43, f"{p11:.2f}")]:
        ax.text(xx, yy, txt, va="center", ha="left", fontsize=13, fontweight=wt, transform=ax.get_yaxis_transform())
for xx, txt in [(1.03, "per sd"), (1.23, "p alone"), (1.43, "p, best of 11")]:
    ax.text(xx, len(FAMR) - 0.45, txt, ha="left", fontsize=12, color="#333", transform=ax.get_yaxis_transform())
ax.axvspan(-MS, MS, color=C_N, alpha=0.22, lw=0)
for sx in (-CHANCE, CHANCE):
    ax.axvline(sx, color="#333333", lw=1.6, ls=(0, (4, 2)))
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(FAMR)))
ax.set_yticklabels([FAMR[j]["lab"] for j in ordr])
for t_ in ax.get_yticklabels():
    if "Hoop G" in t_.get_text():
        t_.set_fontweight("bold"); t_.set_color(C_H)
ax.set_xlim(-6.0, 7.0)
ax.set_ylim(-0.7, len(FAMR) - 0.3)
ax.set_xlabel("Extra pressures per 450-rush season per sd of the drill measure,\nbeyond the pick + role forecast "
              "(held-out class)")
hd = [Line2D([], [], marker="o", color=C_D, mfc="white", mew=2.5, ms=10, lw=3.5, label="estimate, 90% interval"),
      Line2D([], [], marker="D", color="white", mec="#333", ms=6.5, lw=0, label="one held-out class"),
      Patch(color=C_N, alpha=0.35, label=f"action margin ±{MS:.2f}"),
      Line2D([], [], color="#333333", lw=1.6, ls=(0, (4, 2)),
             label=f"chance line ±{CHANCE:.1f}: best of 11 shuffled measures, 95th pct")]
ax.legend(handles=hd, loc="upper left", bbox_to_anchor=(-0.45, -0.2), frameon=False, fontsize=12, ncol=2)
fig.suptitle(f"Six drill measures, n = {len(AN)} rushers: Hoop G leads; burst out of a stance is flat. Hoop G sits on "
             f"the\nchance line, which the best of 11 shuffled measures (six drills, five other Hoop G builds) reaches in "
             f"{NUM['pl_pct_reach']}% of shuffles.", x=0.01, ha="left", fontsize=14)
fig.subplots_adjust(left=0.27, right=0.61, top=0.86, bottom=0.27)
save(fig, "fig3_screen.png")

# Fig 1 (hero). A: the three men's Combine hoop laps overlaid on one axis (marker every 0.5 s, so marker spacing is
# speed; thick segment = hardest half-second). B: every rusher's hardest lean, drill vs Sunday (speed vs turn radius,
# iso-g curves). C: each man's randomly sampled NFL rushes, hardest rush coloured by speed with 0.5 s ticks.
BI = ba.assign(latest_team=ba.latest_team.fillna("FA")).set_index("nfl_id")
SCI = sc.set_index("nfl_id")
DLI = dl.set_index("nfl_id")
_pg = dl.dropna(subset=["g_bend"])
BENDZ = ((_pg.g_bend - _pg.g_bend.mean()) / _pg.g_bend.std()).set_axis(_pg.nfl_id)
MPH = 3600 / 1760
INFO = g.set_index(K)


def nfl_traces(ids):
    gf_ = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "x", "y"],
                          filters=[("nfl_id", "in", ids)])
    meta = g[(g.sample_kind == "random") & g.nfl_id.isin(ids)][K + ["snap_frame", "play_direction"]]
    gf_ = gf_.merge(meta, on=K).sort_values(K + ["frame"])
    pr_ = GF.set_index(K).press
    ok_ = GF.set_index(K).dropna(subset=["g_bend"]).index
    out_ = {i: [] for i in ids}
    for key, h in gf_.groupby(K, sort=False):
        sf = h.snap_frame.iloc[0]
        w = h[(h.frame >= sf) & (h.frame <= sf + 20)]
        if len(w) != 21 or key not in pr_.index or key not in ok_:
            continue
        sgn = -1 if h.play_direction.iloc[0] == "right" else 1
        up = sgn * (w.x.values - w.x.values[0])
        lat = sgn * (w.y.values - w.y.values[0])
        alat, roll, _, _, v = lateral_g(w.x.values, w.y.values)
        bend = float(np.nanmax(roll))
        out_[key[2]].append(dict(key=key, lat=lat, up=up, alat=alat, v=v, k=int(np.nanargmax(roll)), bend=bend,
                                 press=int(pr_.loc[key]), corner=int(bend >= CT)))
    return out_


def scalebar(ax, x0, y0, fs=12):
    ax.plot([x0, x0 + 2], [y0, y0], color="black", lw=4, zorder=6)
    ax.text(x0 + 1, y0 + 0.3, "2 yd", ha="center", va="bottom", fontsize=fs, fontweight="bold", zorder=6)


GO_, NOGO_, HOLD_ = brd[brd.call == "Go"], brd[brd.call == "No-go"], brd[brd.call == "Hold"]
CARDS = pd.concat([GO_, NOGO_]).sort_values("hoop_z", ascending=False)
_hold_pick = HOLD_[HOLD_.rush_snaps >= REG_MIN] if (HOLD_.rush_snaps >= REG_MIN).any() else HOLD_
_ids_all = sorted(set(CARDS.nfl_id.astype(int)) | set(_hold_pick.nfl_id.astype(int)))
NT = nfl_traces(_ids_all)
_hold_pick = _hold_pick.assign(nt=[len(NT[int(p_)]) for p_ in _hold_pick.nfl_id])
_hold_pick = _hold_pick[_hold_pick.nt >= 8].sort_values("hoop_z", ascending=False)
HERO = [("beat", int(GO_.sort_values("hoop_z").nfl_id.iloc[-1])),
        ("below", int(NOGO_.sort_values("hoop_z").nfl_id.iloc[-1])),
        ("par", int(_hold_pick.nfl_id.iloc[0]))]
MCOL = {"beat": OI["blue"], "below": OI["red"], "par": "#222222"}
MMK = {"beat": "o", "below": "s", "par": "^"}
MGL = {"beat": "●", "below": "■", "par": "▲"}
TAGT = {"beat": "beat forecast", "below": "below forecast", "par": "on par (null case)"}
SPN = plt.Normalize(0, 14)
SCM = "cividis"

fig = plt.figure(figsize=(FIGW, 12.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2], left=0.04, right=0.98, top=0.90, bottom=0.14, hspace=0.36,
                      wspace=0.16)
axL = fig.add_subplot(gs[0, :2])
axS = fig.add_subplot(gs[0, 2])
for j, (tag, pid) in enumerate(HERO):
    r_ = BI.loc[pid]
    tr_ = traces[r_.hoop_eid]
    _hl = hoop_att[hoop_att.nfl_id == pid]
    x_, y_ = tr_["_x"] - np.mean(tr_["_x"]), tr_["_y"] - np.mean(tr_["_y"])
    k_ = int(tr_["_k"])
    axL.plot(x_, y_, color=MCOL[tag], lw=2.0, alpha=0.85, zorder=2)
    axL.plot(x_[::5], y_[::5], MMK[tag], color=MCOL[tag], mfc="white", mew=1.8, ms=7, zorder=3)
    axL.plot(x_[max(k_ - 2, 0):k_ + 3], y_[max(k_ - 2, 0):k_ + 3], color=MCOL[tag], lw=8, alpha=0.6,
             solid_capstyle="round", zorder=4)
    axL.text(1.0, 0.30 - 0.11 * j, f"{MGL[tag]} {r_.display_name}: {_hl.bend_g.mean():.2f} g at "
             f"{_hl.speed_at_bend.mean() * MPH:.1f} mph", transform=axL.transAxes, ha="right", va="top",
             fontsize=12.5, fontweight="bold", color=MCOL[tag], zorder=6)
    TXT[f"h{j}"] = r_.display_name
    TXT[f"h{j}_team"] = r_.latest_team
    put(f"h{j}_hoopz", r_.hoop_z, 1); put(f"h{j}_hoopg", _hl.bend_g.mean())
    put(f"h{j}_mph", _hl.speed_at_bend.mean() * MPH, 1); put(f"h{j}_rad", _hl.rad.mean(), 1)
axL.set_aspect("equal"); axL.axis("off")
_xl, _yl = axL.get_xlim(), axL.get_ylim()
scalebar(axL, _xl[0], _yl[0] - 0.6)
axL.set_ylim(_yl[0] - 4.5, _yl[1])
axL.set_title("A  Hoop laps overlaid: marker every 0.5 s,\n    thick stroke = hardest half-second", loc="left",
              fontsize=13)
# B: drill vs Sunday at the hardest lean, every rusher with both
_ds = dl.dropna(subset=["hoop_rad", "nfl_rad"])
for _, r_ in _ds.iterrows():
    axS.plot([r_.hoop_rad, r_.nfl_rad], [r_.hoop_v * MPH, r_.nfl_v * MPH], color="#CFCFCF", lw=0.8, zorder=1)
axS.scatter(_ds.hoop_rad, _ds.hoop_v * MPH, s=16, color=OI["orange"], zorder=2, label="Combine hoop")
axS.scatter(_ds.nfl_rad, _ds.nfl_v * MPH, s=16, color=OI["sky"], zorder=2, label="NFL rushes (median)")
for tag, pid in HERO:
    r_ = DLI.loc[pid]
    axS.plot([r_.hoop_rad, r_.nfl_rad], [r_.hoop_v * MPH, r_.nfl_v * MPH], color=MCOL[tag], lw=2.2, zorder=3)
    axS.plot(r_.hoop_rad, r_.hoop_v * MPH, MMK[tag], color=MCOL[tag], mfc="white", mew=2, ms=9, zorder=4)
    axS.plot(r_.nfl_rad, r_.nfl_v * MPH, MMK[tag], color=MCOL[tag], ms=9, zorder=4)
rr = np.linspace(0.3, 9, 100)
for gg, xt in [(0.2, 8.2), (0.6, 3.2)]:
    axS.plot(rr, np.sqrt(gg * YD_PER_G * rr) * MPH, color="#555555", lw=1, ls=":", zorder=0)
    axS.text(xt, np.sqrt(gg * YD_PER_G * xt) * MPH + 0.3, f"{gg:.1f} g", fontsize=11.5, color="#333", ha="center")
axS.set_xlim(0, 9); axS.set_ylim(0, 14)
axS.set_xlabel("Turn radius at hardest lean (yd)"); axS.set_ylabel("Speed at hardest lean (mph)")
axS.legend(loc="lower right", frameon=False, fontsize=11.5, handletextpad=0.2)
axS.set_title(f"B  Drill vs Sunday ({len(_ds)} rushers):\n    slower, wider NFL turns", loc="left", fontsize=13)
put("fig1_n", len(_ds))
# C: NFL fans
for j, (tag, pid) in enumerate(HERO):
    ax = fig.add_subplot(gs[1, j])
    r_ = BI.loc[pid]
    rs = NT[pid]
    best = int(np.argmax([t["bend"] for t in rs]))
    for t in rs:
        ax.plot(t["lat"], t["up"], color="#C4C4C4", lw=1.5, zorder=1)
        if t["press"]:
            ax.plot(t["lat"][-1], t["up"][-1], "X", ms=10, color="black", mec="white", mew=0.6, zorder=5)
    t = rs[best]
    pts = np.c_[t["lat"], t["up"]].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=SCM, norm=SPN, lw=5, zorder=3)
    lc.set_array(t["v"][:-1] * MPH); ax.add_collection(lc)
    ax.plot(t["lat"][::5], t["up"][::5], "o", ms=6, color="white", mec="black", mew=1.2, zorder=4)
    ax.plot(0, 0, MMK[tag], color=MCOL[tag], ms=11, zorder=6)
    ax.set_xlim(-6.5, 6.5); ax.set_ylim(-2.0, 10.5); ax.set_aspect("equal"); ax.axis("off")
    scalebar(ax, -6.3, -1.6)
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
    ov = (SCI.loc[pid].press_eb - SCI.loc[pid].fc) * PS
    d_ = DLI.loc[pid]
    npress = sum(t_["press"] for t_ in rs)
    ncorner = sum(t_["corner"] for t_ in rs)
    ax.set_title(f"{MGL[tag]} {r_.display_name}", fontsize=15.5, fontweight="bold", color=MCOL[tag], pad=36)
    ax.text(0.5, 1.0, f"{r_.latest_team} · {int(r_.draft_year)} {pk} · {r_.role}\n{TAGT[tag]}: {fm(ov)} a season",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=12.5)
    if j == 0:
        ax.text(-0.02, 1.32, "C", transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
    ax.text(0.5, -0.04, f"Hoop G {fm(r_.hoop_z)} sd · NFL bend {fm(BENDZ.loc[pid])} sd\n"
            f"{ncorner} of {len(rs)} corner turns · {npress} of {len(rs)} pressured", transform=ax.transAxes,
            ha="center", va="top", fontsize=12.5)
    if j == 0:
        ax.annotate("upfield", xy=(-5.4, 9.0), xytext=(-5.4, 5.2), fontsize=12, ha="center",
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    put(f"h{j}_bendz", BENDZ.loc[pid], 1)
    put(f"h{j}_n", len(rs)); put(f"h{j}_np", int(npress)); put(f"h{j}_nc", int(ncorner))
    put(f"h{j}_nflg", d_.g_bend); put(f"h{j}_nflmph", d_.nfl_v * MPH, 1); put(f"h{j}_nflrad", d_.nfl_rad, 1)
    put(f"h{j}_over", ov, 1)
fig.text(0.5, 0.115, "C: each man's randomly sampled NFL rushes, first 2 s, same yard scale. Grey = every rush; "
         "X = pressured snap;\ncoloured = his hardest-bending rush (colour = speed, dot every 0.5 s).", fontsize=12.5,
         va="top", ha="center")
cax = fig.add_axes([0.32, 0.045, 0.36, 0.013])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=SPN, cmap=SCM), cax=cax, orientation="horizontal")
cb.set_label("Speed on his hardest-bending NFL rush (mph)", fontsize=12.5)
fig.text(0.01, 0.995, "The hoop asks every man for the same tight turn; Sunday does not. Hoop G ranks men within a class.",
         fontsize=15, fontweight="bold", va="top")
save(fig, "fig1_hero.png")
put("hoop_mph_median", hoop_att.speed_at_bend.median() * MPH, 1)
put("nfl_bend_median", float(PG.g_bend.median()))
put("n_short_career", int((dl.rush_snaps < RATE_MIN).sum()))
_gm3 = GT.merge(dl[["nfl_id", "hoop_v"]], on="nfl_id").dropna(subset=["hoop_v"])
put("pct_rush_hoopspeed", 100 * (_gm3.t_v >= _gm3.hoop_v).mean(), 0)

# Fig 2 (appendix): (A) reliability of the drill average by reps; (B) what one lap does to a true effect.
fig, (ax, bx) = plt.subplots(1, 2, figsize=(FIGW, 6.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.30, wspace=0.62)
kk = np.arange(1, 21)
_tl, _th = float(NUM["rel_turn_lap_lo"]), float(NUM["rel_turn_lap_hi"])
for r1, lo1, hi1, col, lab in [
        (REL_TURN, _tl, _th, C_H, f"Hoop G, one lap = {REL_TURN:.2f}"),
        (rel_d1_1, *ICC_CI["burst"], C_B, f"Rush-drill burst, one rep = {rel_d1_1:.2f}")]:
    ax.fill_between(kk, sb(max(lo1, 0.005), kk), sb(hi1, kk), color=col, alpha=0.18, lw=0)
    ax.plot(kk, sb(r1, kk), color=col, lw=3.5, label=lab)
    kn = reps_needed(r1)
    ax.plot(kn, sb(r1, kn), "o", color="white", mec=col, mew=3, ms=11, zorder=5)
    ax.annotate(f"{kn} reps", (kn, sb(r1, kn)), xytext=(8, -18), textcoords="offset points", fontsize=12.5,
                fontweight="bold")
ax.plot(kk, sb(rel_hoop1, kk), color=C_H, lw=2.0, ls=(0, (3, 2)),
        label=f"Hoop G, repeat-lap check ({n_hrep} men ran two)")
ax.axhline(REL_TARGET, color="#333333", lw=1.2, ls=":")
ax.text(9.0, REL_TARGET + 0.015, f"target {REL_TARGET:.2f}", ha="left", va="bottom", fontsize=12)
ax.set_xlim(0.5, 20.5); ax.set_ylim(0, 1)
ax.set_xticks([1, 4, 8, 12, 16, 20])
ax.set_xlabel("Graded reps per lineman")
ax.set_ylabel("Reliability of the drill average")
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.17), frameon=False, fontsize=12)
ax.set_title("A: one hoop lap is a weak read", loc="left", fontsize=13.5, fontweight="bold")
_tk = sorted([(REL["g_bend"], "NFL bend"), (REL["quick"], "get-off")])
_ty = [_tk[0][0], max(_tk[1][0], _tk[0][0] + 0.12)]
for (v_, lab_), ty_ in zip(_tk, _ty):
    ax.plot([20.5, 21.3], [v_, v_], color="black", lw=2.5, clip_on=False)
    ax.annotate(f"{lab_} {v_:.2f}", xy=(21.3, v_), xytext=(22.2, ty_), textcoords="data", va="center", fontsize=12,
                annotation_clip=False, arrowprops=dict(arrowstyle="-", color="#555", lw=0.8))
_meq4 = np.array([np.median(sim_obs(be, laps=4)[0]) for be in BETA])
bx.fill_between(BETA * PS, MEQ[:, 0] * PS, MEQ[:, 2] * PS, color=C_H, alpha=0.2, lw=0)
bx.plot(BETA * PS, MEQ[:, 1] * PS, color=C_H, lw=3)
bx.plot(BETA * PS, _meq4 * PS, color=C_B, lw=2.5, ls="--")
bx.plot(BETA * PS, BETA * PS, color="#777777", lw=1, ls=":")
_ie = len(BETA) - 3
_xe = BETA[_ie] * PS
for yv_, lab_, col_ in [(MEQ[_ie, 1] * PS, "one lap", C_H), (_meq4[_ie] * PS, "four laps", C_B),
                        (BETA[_ie] * PS, "no noise", "#555555")]:
    _below = lab_ == "one lap"
    bx.text(_xe, yv_ - 0.6 if _below else yv_ + 0.4, lab_, color=col_, fontsize=12, va="top" if _below else "bottom",
            ha="right", fontweight="bold")
bx.axhline(B_OBS * PS, color="black", lw=1.4)
bx.text(BETA.min() * PS + 0.2, B_OBS * PS + 0.25, f"observed {B_OBS * PS:.1f}", fontsize=12, va="bottom")
bx.axvspan(_cov.min() * PS, _cov.max() * PS, color=C_N, alpha=0.22, lw=0)
bx.text((_cov.min() + _cov.max()) / 2 * PS, -5.2, f"true slopes consistent\nwith {B_OBS * PS:.1f}: "
        f"{_cov.min() * PS:.1f} to {_cov.max() * PS:.1f}", ha="center", va="bottom", fontsize=12)
bx.set_xlim(BETA.min() * PS, BETA.max() * PS); bx.set_ylim(-5.5, 15)
bx.set_xlabel("True slope (pressures a season\nper sd of true bend)")
bx.set_ylabel(f"Slope seen at n = {len(AN)}")
bx.set_title(f"B: one lap shows ~{100 * ATT:.0f}% of a true slope\n(band = 90% of simulations)", loc="left",
             fontsize=13.5, fontweight="bold")
fig.suptitle(f"One hoop lap ({NUM['pct_one_run']}% of linemen ran only one) hides part of any real bend effect; "
             "four laps\nwould match NFL bend's steadiness. Bands: 90% player bootstrap of reliability.",
             x=0.01, ha="left", fontsize=14)
save(fig, "fig2_reliability.png")

# Fig 4: player cards for the flagged day-3 rushers with 100+ NFL rushes, sorted by Hoop G (not by outcome).
# Each card: Hoop G rank inside his Combine class, sampled NFL rush fan, NFL result with a 90% interval.
fig = plt.figure(figsize=(FIGW, 13.0))
outer = fig.add_gridspec(2, 4, left=0.02, right=0.99, top=0.85, bottom=0.08, wspace=0.12, hspace=0.30)
_ivs = [(r_.over * PS - Z90 * r_.over_sd * PS, r_.over * PS + Z90 * r_.over_sd * PS) for _, r_ in CARDS.iterrows()]
XL = float(np.ceil(max(max(abs(a_), abs(b_)) for a_, b_ in _ivs) + 3))
for i_, (_, r_) in enumerate(CARDS.iterrows()):
    pid = int(r_.nfl_id)
    sub = outer[i_ // 4, i_ % 4].subgridspec(3, 1, height_ratios=[0.55, 2.6, 0.95], hspace=0.10)
    axr = fig.add_subplot(sub[0])
    cls_ = ba[ba.draft_year == r_.draft_year].hoop_z.dropna()
    rank_ = int((cls_ > r_.hoop_z).sum()) + 1
    axr.axvspan(np.percentile(cls_, 100 * 2 / 3), 3.3, color=C_H, alpha=0.15, lw=0)
    axr.plot(cls_, np.zeros(len(cls_)), "|", color="#8C8C8C", ms=14, mew=1.4)
    axr.plot(r_.hoop_z, 0, "o", color=C_H, mec="black", ms=12, zorder=5)
    axr.set_xlim(-3.3, 3.3); axr.set_ylim(-1, 1); axr.axis("off")
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
    axr.set_title(f"{r_.display_name}", fontsize=13.5, fontweight="bold", pad=44)
    axr.text(0.5, 1.05, f"{r_.latest_team} · {int(r_.draft_year)} {pk} · {r_.role}\nHoop G {fm(r_.hoop_z)} sd, "
             f"rank {rank_}/{len(cls_)}", transform=axr.transAxes, ha="center", va="bottom", fontsize=11.5)
    axf = fig.add_subplot(sub[1])
    rs = NT[pid]
    best = int(np.argmax([t["bend"] for t in rs]))
    for t in rs:
        axf.plot(t["lat"], t["up"], color="#B5B5B5", lw=1.2, zorder=1)
        if t["press"]:
            axf.plot(t["lat"][-1], t["up"][-1], "X", ms=8, color="black", mec="white", mew=0.5, zorder=4)
    axf.plot(rs[best]["lat"], rs[best]["up"], color=C_H, lw=2.8, zorder=3)
    axf.plot(0, 0, "o", color="black", ms=4)
    axf.set_xlim(-6, 6); axf.set_ylim(-2.6, 9.4); axf.set_aspect("equal"); axf.axis("off")
    axf.plot([-5.8, -3.8], [-0.7, -0.7], color="black", lw=3)
    axf.text(-4.8, -0.55, "2 yd", ha="center", va="bottom", fontsize=10.5)
    npress = sum(t_["press"] for t_ in rs)
    ncorner = sum(t_["corner"] for t_ in rs)
    axf.text(0.5, 0.0, f"{npress}/{len(rs)} pressured · {ncorner}/{len(rs)} corner turns", transform=axf.transAxes,
             ha="center", va="bottom", fontsize=11.5)
    axs_ = fig.add_subplot(sub[2])
    ov = r_.over * PS
    lo_, hi_ = _ivs[i_]
    crosses = (lo_ < MS) and (hi_ > -MS)
    c_ = "#9A9A9A" if crosses else (OI["blue"] if ov > 0 else OI["red"])
    axs_.axvspan(-MS, MS, color=C_N, alpha=0.25, lw=0)
    axs_.axvline(0, color="black", lw=0.9)
    axs_.plot([lo_, hi_], [0, 0], color=c_, lw=5, solid_capstyle="butt")
    axs_.plot(ov, 0, "o", color=c_, mec="black", ms=9)
    axs_.text(-XL, 0.55, f"NFL {fm(ov)}", ha="left", va="center", fontsize=12.5, fontweight="bold",
              color=c_ if c_ != "#9A9A9A" else "#444444")
    axs_.set_xlim(-XL, XL); axs_.set_ylim(-0.6, 0.9); axs_.set_yticks([])
    axs_.set_xticks([-20, 0, 20]); axs_.tick_params(labelsize=10.5, length=3)
    for s_ in ("left", "right", "top"):
        axs_.spines[s_].set_visible(False)
    put(f"f{i_}_ncorner", int(ncorner)); put(f"f{i_}_nrush", len(rs)); put(f"f{i_}_rank", rank_)
    put(f"f{i_}_ncls", len(cls_))
fig.supxlabel("NFL pressures per 450 rushes above the pick + role forecast (90% interval; grey band = action "
              "margin; grey bar = not clear of it)", fontsize=12, y=0.012)
fig.text(0.01, 0.995, f"For film study, not a ranking: {len(CARDS)} day-3 rushers flagged by Hoop G, sorted by Hoop G",
         fontsize=14, fontweight="bold", va="top")
fig.text(0.01, 0.966, "2024–25 classes, 100+ NFL rushes. Top: his Hoop G (orange dot) among his Combine class "
         "(ticks; shaded = top third).\nMiddle: his randomly sampled NFL rushes, first 2 s (orange = hardest-bending "
         "rush; X = pressured snap).\nBottom: NFL result. Blue = clear gain, vermillion = clear shortfall.",
         fontsize=12, va="top")
save(fig, "fig4_cards.png")

# Tape pull: the hardest-bending sampled rush of three named rushers, with the facts tracking can give. The tackle's
# set and whether he won the block are not in the data; those are what the clip must show.
TAPE = []
for nm_ in ["Javon Solomon", "Tyrion Ingram-Dawkins", "Jaylen Harrell"]:
    r_ = CARDS[CARDS.display_name == nm_].iloc[0]
    rs = NT[int(r_.nfl_id)]
    t = rs[int(np.argmax([t_["bend"] for t_ in rs]))]
    inf = INFO.loc[t["key"]]
    TAPE.append(dict(name=nm_, team=r_.latest_team, season=int(inf.season), week=int(inf.week),
                     game_id=int(t["key"][0]), play_id=int(t["key"][1]), quarter=int(inf.quarter),
                     down=int(inf.down), ytg=int(inf.yards_to_go), align=str(inf.lined_up_position),
                     form=str(inf.offense_formation), getoff=float(inf.player_get_off), g=t["bend"],
                     t_lean=t["k"] / 10, depth=float(t["up"][t["k"]]), mph=float(t["v"][t["k"]] * MPH),
                     press=int(t["press"]), over=r_.over * PS))
TP = pd.DataFrame(TAPE)
TP.to_csv(OUT / "tape_pull.csv", index=False)
_rows_md = []
for i_, r_ in TP.iterrows():
    put(f"tp{i_}_getoff", r_.getoff); put(f"tp{i_}_g", r_.g); put(f"tp{i_}_t", r_.t_lean, 1)
    put(f"tp{i_}_depth", r_.depth, 1); put(f"tp{i_}_mph", r_.mph, 1)
    put(f"tp{i_}_medmph", DLI.loc[int(CARDS[CARDS.display_name == r_["name"]].nfl_id.iloc[0])].nfl_v * MPH, 1)
    _dn = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[r_.down]
    _rows_md.append(f"| {r_['name']} ({fm(r_.over)}) | {r_.season} wk {r_.week}, Q{r_.quarter}, play {r_.game_id}-"
                    f"{r_.play_id} | {r_['align']}, {_dn} & {r_.ytg}, {r_.form.lower()} | {r_.getoff:.2f} s | "
                    f"{r_.g:.2f} g at {r_.t_lean:.1f} s, {r_.depth:.1f} yd upfield, {r_.mph:.1f} mph | "
                    f"{'yes' if r_.press else 'no'} |")
TXT["tape_rows"] = "\n".join(_rows_md)

# film list in Hoop G order, split into open-first (beat the forecast) and missed (fell below it)
# reads: hand-written, but every fact in them is computed here (NFL bend sd, NGS get-off, sampled burst, rushes)
CARDS["bendz"] = [BENDZ.loc[int(p_)] for p_ in CARDS.nfl_id]
CARDS["getoff"] = [DLI.loc[int(p_)].get_off for p_ in CARDS.nfl_id]
CARDS["d1g"] = [DLI.loc[int(p_)].g_d1 for p_ in CARDS.nfl_id]
CARDS["nflv"] = [DLI.loc[int(p_)].nfl_v * MPH for p_ in CARDS.nfl_id]
_rk = lambda col, nm, asc: int(CARDS.sort_values(col, ascending=asc).display_name.tolist().index(nm)) + 1
ORD = {1: "", 2: "second-", 3: "third-"}
READ = {
    "Cedric Johnson": "give him third-down snaps from a wide alignment.",
    "Javon Solomon": "top hoop lap of his class; keep the wide passing-down role.",
    "Warren Brinson": "a 315-lb interior man: judge him against interior linemen; loop him on stunts.",
    "Austin Booker": "most rushes of the eight, a steady small plus.",
    "Jack Sawyer": "{bo}hardest NFL bend of the eight; widen his passing-down role.",
    "Eric Watts": "{do}shortest first-second burst of the eight; coach the get-off first.",
    "Tyrion Ingram-Dawkins": "average NFL lean, {go}slowest NGS get-off of the eight; rep dip-and-rip off a live get-off.",
    "Jaylen Harrell": "{bo}hardest NFL bend of the eight, yet well below forecast: the bend is not becoming pressure (Table 3).",
}
_open, _miss = [], []
for i_, (_, r_) in enumerate(CARDS.iterrows()):
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
    ov = r_.over * PS
    nm = r_.display_name
    put(f"f{i_}_over", ov, 1); put(f"f{i_}_hoopz", r_.hoop_z, 1); put(f"f{i_}_bendz", r_.bendz, 1)
    put(f"f{i_}_getoff", r_.getoff); put(f"f{i_}_rush", int(r_.rush_snaps)); put(f"f{i_}_nflmph", r_.nflv, 1)
    rd = READ.get(nm, "").format(bo=ORD.get(_rk("bendz", nm, False), "?"), do=ORD.get(_rk("d1g", nm, True), "?"),
                                 go=ORD.get(_rk("getoff", nm, False), "?"),
                                 vo=ORD.get(_rk("nflv", nm, True), "?"), vm=f"{r_.nflv:.1f}")
    rd = rd[0].upper() + rd[1:]
    ln = (f"- **{nm}** ({r_.latest_team}, {r_.role}, {int(r_.draft_year)} {pk}): Hoop G {fm(r_.hoop_z)} sd, NFL bend "
          f"{fm(r_.bendz)} sd, NFL {fm(ov)}. {rd}")
    (_open if r_.call == "Go" else _miss).append(ln)
assert CARDS.rush_snaps.idxmax() == CARDS[CARDS.display_name == "Austin Booker"].index[0]
TXT["film_open"] = "\n".join(_open)
TXT["film_miss"] = "\n".join(_miss)
put("n_cards", len(CARDS)); put("n_open", len(_open)); put("n_miss", len(_miss))
put("cards_min_sampled", int(min(len(NT[int(p_)]) for p_ in CARDS.nfl_id)))
put("cards_max_sampled", int(max(len(NT[int(p_)]) for p_ in CARDS.nfl_id)))


# ============================================================================= multiplicity and independent check
TESTS = (["family: " + f[0] for f in FAM]
         + ["chain: burst->get-off", "chain: burst->NFL burst", "chain: hoop->NFL bend", "chain: hoop->get-off",
            "bench: 3-cone->NFL bend", "bench: split->get-off", "bench: 3-cone->get-off",
            "trait->pressure: get-off", "trait->pressure: burst", "trait->pressure: bend",
            "R2 gain: full + hoop", "R2 gain: full + burst", "burst equivalence",
            "day3 rho: fc+hoop vs fc", "day3 rho: fc+hoop vs pick", "day3 rho: hoop vs pick",
            "day3 top-third gap", "day3 flag Fisher"])
put("n_tests", len(TESTS))


def pb(x, y_):
    ks = [k for k in PB if k[:2] == (x, y_)]
    return PB[max(ks, key=lambda k: k[2] if len(k) > 2 else 0)]


PV = ([HO[f[0]]["p2"] for f in FAM]
      + [pb("burst_z", "quick"), pb("burst_z", "g_d1"), pb("hoop_z", "g_bend"), pb("hoop_z", "quick"),
         pb("three_cone_fast", "g_bend"), pb("split_fast", "quick"), pb("three_cone_fast", "quick"),
         pb("quick", "press_eb"), pb("g_d1", "press_eb"), pb("g_bend", "press_eb"),
         PB[("dr2", "base_h")], PB[("dr2", "burst_h")], float(NUM["ho_burst_tost"]),
         PB[("day3", "fch_fc")], PB[("day3", "fch_pick")], PB[("day3", "hoop_pick")],
         PB[("day3", "hitgap")], float(NUM["fl_p_fisher"])])
TESTS = TESTS + ["R2 gain: pick + hoop", "R2 gain: pick + role + hoop", "R2 gain: nested + hoop",
                 "hoop slope: pick residual", "hoop slope: pick + role residual", "hoop slope: nested residual",
                 "hoop slope: EDGE only", "hoop slope: interior only", "hoop slope: mean of two turns",
                 "chain: hoop radius->NFL radius", "film: corner-turn rate by Hoop G third",
                 "film: pressure on corner-turn rushes"]
PV = PV + [PB[("dr2", "pick1_h")], PB[("dr2", "pick_h")], PB[("dr2", "nest_h")], PB[("slope", "bs_pick1")],
           PB[("slope", "bs_pick")], PB[("slope", "bs_nest")], PB[("slope", "role_EDGE")],
           PB[("slope", "role_interior")], PB[("slope", "mt_hoop")], pb("hoop_rad_tight", "nfl_rad_tight"),
           PB[("film", "ct_gap")], PB[("film", "ct_press")]]
assert len(PV) == len(TESTS)
NUM["n_tests"] = len(TESTS)
PV = np.asarray(PV)
Q = 0.10
o_ = np.argsort(PV)
m_ = len(PV)
bh_ok = np.zeros(m_, bool)
passed = np.flatnonzero(PV[o_] <= Q * np.arange(1, m_ + 1) / m_)
if len(passed):
    bh_ok[o_[: passed.max() + 1]] = True
holm_ok = np.zeros(m_, bool)
for i_, j_ in enumerate(o_):
    if PV[j_] <= Q / (m_ - i_):
        holm_ok[j_] = True
    else:
        break
MULT = pd.DataFrame({"test": TESTS, "p": PV, "bh_q10": bh_ok, "holm_10": holm_ok})
MULT.to_csv(OUT / "multiplicity.csv", index=False)
put("n_bh", int(bh_ok.sum())); put("n_holm", int(holm_ok.sum())); put("q_fdr", Q)
TXT["bh_tests"] = list(MULT[MULT.bh_q10].test)
TXT["holm_tests"] = list(MULT[MULT.holm_10].test)
put("p_hoop_bh_rank", int(np.flatnonzero(PV[o_] == HO["hoop_z"]["p2"])[0] + 1))
import subprocess
import sys
subprocess.run([sys.executable, str(OUT / "verify.py")], check=True, capture_output=True)
VJ = json.loads((OUT / "verify.json").read_text())
put("v_n", VJ["v_n"]); put("v_slope", VJ["v_slope"], 1); put("v_lo", VJ["v_lo"], 1); put("v_hi", VJ["v_hi"], 1)
put("v_neg_classes", int(sum(v <= 0 for v in VJ["v_cls"])))
put("v_same_set", int(set(VJ["v_ids"]) == set(int(i) for i in AN.nfl_id)))
put("v_set_diff", len(set(VJ["v_ids"]) ^ set(int(i) for i in AN.nfl_id)))
TXT["nest_choice"] = TXT_NEST
TXT["nest_list"] = ", ".join(f"{v.lower()} for {c}" for c, v in TXT_NEST.items())
for c, v in zip(classes, VJ["v_cls"]):
    put(f"v_cls_{c}", v, 1)
TXT["lopo25_name"] = TXT_LOPO
TXT["extra78"] = TXT_EXTRA

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
