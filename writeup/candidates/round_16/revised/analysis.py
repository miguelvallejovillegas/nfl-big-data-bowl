"""Break G: does the Combine route-drill cut predict NFL separation? And what does?

Round 16 revision (of round_16/cand_2; see "round 16 revision (R7)" before the figures). Run from the repo root:
    uv run python writeup/candidates/round_16/revised/analysis.py

Round 16 revision changes: day-3 gain with the shrinkage refitted inside each season fold and in draft-class folds;
the rule against naive raw last-season SOE; matched contrasts with a shuffle null and an interval on the gap over pick
and routes; TOST of the Break G slope at +/-3 in; scout notes (route mix, coverage splits, passer) for the act-on names;
Fig 1 = all 99 receivers plus the two rule-chosen extremes; Fig 3 = pooled top-vs-bottom-third NFL speed traces per
route family (replaces the hand-picked pair); Fig 4 = one primary gain, naive baseline, contrasts on the same 20 pairs.

Reads data/processed/*.parquet only (about eight minutes). Writes, next to this file:
- numbers.json: every number quoted in writeup.md, one key each, and nothing else;
- claims.json: for every key, the claim it backs (writeup/check.py verifies each value appears in the text);
- figure_numbers.json: numbers printed only inside figures;
- shortlist_2026.csv, board.csv, board_full.csv, frozen_rule.json; figures/fig1-4 (1600 px wide, 150 dpi).

Round 15 changes (judges' round-14 feedback), section "round 15 additions" before the figures:
- survivorship: logistic selection model for reaching the NFL sample and for 100+ career routes; primary slope
  re-weighted by 1 / P(selected), selection refitted in every class-stratified resample;
- one table of Break G slopes (plain offset = primary, ridge, no offset, IPW, receiver random-intercept mixed model,
  without Nacua, route-mix and QB adjusted) with a single one-sided test against +3 in;
- permutation tests (within-fold shuffles of last season's SOE) of the 7/13 vs 6/13 gap and of the rank-correlation gain;
  receiver random-intercept model of persistence; calibration of held-out probabilities;
- 2026 list context (team, alignment, games) and a worked decision case: the two 2025 day-3 WR with matching pick and
  routes whose SOE forecasts differ most, with the catch and cap consequence;
- figures: Fig 1 shared scale bar and regression-to-the-mean note; Fig 2C greys slices with NFL reliability < 0.10;
  Fig 3 prints the price and value of each row; Fig 4 sorted by outcome with hit/miss ticks, larger B, calibration C;
- numbers.json keeps only the numbers quoted in writeup.md (others go to unquoted_numbers.json; all_numbers.json has all).

Round 14 revision changes:
- Primary = the single forecast fixed in round 9 (frozen_rule.json): plain least squares on log pick, 40, weight,
  NGS athleticism and TE, fitted on two classes and used as an offset on the third. The ridge forecast is one
  sensitivity row; the plain/ridge average and the sceptical-prior posterior are gone.
- numbers.json rebuilt from scratch (the internal NUM dictionary of earlier rounds is no longer written).
- Day-3 / UDFA evaluation is continuous and leave-one-season-out: held-out rank correlation, RMSE and Brier score of a
  pick-and-routes forecast with and without last season's empirical-Bayes (route-shrunk) SOE, receiver-clustered
  bootstrap of the paired gains. Yes/no rule = shrunk SOE above a cut-off chosen on the other season pair.
  Player-level probabilities of 2026 SOE > 0 for the 13 names on the 2026 list.
- Drill-to-tape: Bayesian errors-in-variables reading (Bayes factor) of the drill-to-NFL correlation on receivers with
  8+ NFL breaks; within-family links (slant, out, hitch) with per-family reliability; alignment and coverage slices.
- Figures redrawn: six receivers chosen by a stated rule; chain with slice forest; price/value forest; day-3 strip
  with the paired held-out gain.

Metric (definition frozen in round 9; identical code on Combine drills and NFL snaps):
- velocity / acceleration: Savitzky-Golay derivatives of 10 Hz x/y (window 7, order 2);
- break: first >= 30 deg heading change after 1.5 yd of travel, entered above 2.5 yd/s;
- Break G = mean acceleration magnitude (g) over the 1.0 s centred on the sharpest point of the turn;
- per rep z-scored within drill x Combine year; player score = mean over route drills, shrunk by reliability.

The sections before "round 14 revision" are the earlier rounds' pipeline (kept so every input is rebuilt from the
parquet files); many intermediate results there go only into the internal NUM dictionary.
"""
from __future__ import annotations

import hashlib
import json
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # small matrices: threads only add overhead
    os.environ.setdefault(_v, "1")
from pathlib import Path

import matplotlib
import matplotlib.patches  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DT = 0.1
G_YD = 10.73  # 1 g in yd/s^2
IN_YD = 36.0
RNG = np.random.default_rng(2027)
NBOOT = int(os.environ.get("BDB_NBOOT", 2000))  # lower only for quick development runs
NPERM = int(os.environ.get("BDB_NPERM", 2000))
LEAD_MIN = 0.2
SESOI_IN = 3.0  # smallest effect of interest, inches of season SOE per sd, fixed in round 9 before the test
CLASSES = (2023, 2024, 2025)
FAMILIES = ["SLANT", "OUT", "HITCH", "IN", "CORNER"]
DEFAULT_SPEC = (30.0, 5, 2.5)  # turn deg, half-window frames (5 -> 1.0 s), entry speed yd/s

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 150,
                     "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.labelsize": 12})
FIGW = 1600 / 150

NUM: dict[str, str] = {}
import sys  # noqa: E402
import time  # noqa: E402

_T0 = time.time()


def tick(label):
    print(f"[{time.time() - _T0:7.1f}s] {label}", file=sys.stderr, flush=True)


def put(key, val, nd=2):
    if key in NUM:
        raise KeyError(f"duplicate numbers.json key {key}")
    if isinstance(val, (int, np.integer)):
        s = str(int(val))
    else:
        s = f"{float(val):.{nd}f}"
        if s.startswith("-") and float(s) == 0:
            s = s[1:]
    NUM[key] = s
    return s


def save(fig, name):
    w, h = fig.get_size_inches()
    assert abs(w - FIGW) < 1e-6, name
    fig.savefig(FIG / name, dpi=150, facecolor="white")
    plt.close(fig)


def zs(s):
    return (s - s.mean()) / s.std(ddof=1)


# ----------------------------------------------------------------------------- kinematics
def kin(x, y):
    w = 7 if len(x) >= 7 else max(3, (len(x) // 2) * 2 - 1)
    vx = savgol_filter(x, w, 2, deriv=1, delta=DT)
    vy = savgol_filter(y, w, 2, deriv=1, delta=DT)
    ax = savgol_filter(vx, w, 2, deriv=1, delta=DT)
    ay = savgol_filter(vy, w, 2, deriv=1, delta=DT)
    return vx, vy, ax, ay


def angdiff(a, b):
    d = np.abs(a - b) % 360
    return np.where(d > 180, 360 - d, d)


def break_mechanics(x, y, i0, i1, min_path=1.5, min_turn=30.0, h=6, entry_speed=2.5, hw=5):
    n = len(x)
    if n < 2 * h + 3:
        return None
    vx, vy, ax, ay = kin(x, y)
    s = np.hypot(vx, vy)
    head = np.degrees(np.arctan2(vx, vy)) % 360
    amag = np.hypot(ax, ay)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    cum = cum - cum[i0]
    turn = np.zeros(n)
    turn[h:n - h] = angdiff(head[:n - 2 * h], head[2 * h:])
    ok = np.zeros(n, bool)
    ok[max(i0 + h, h):min(i1, n - h)] = True
    ok &= cum >= min_path
    sin = np.array([s[max(0, i - h - 3):i - h + 1].max() if i - h >= 0 else 0.0 for i in range(n)])
    ok &= sin > entry_speed
    cand = np.where(ok & (turn >= min_turn))[0]
    if len(cand) == 0:
        return None
    j = cand[0]
    seg = np.arange(j, min(j + 8, n - h))
    seg = seg[ok[seg]] if ok[seg].any() else np.array([j])
    i = int(seg[np.argmax(turn[seg])])
    lo, hi = max(0, i - hw), min(n, i + hw + 1)
    apex = max(0, i - 5) + int(np.argmin(s[max(0, i - 5):min(n, i + 6)]))
    a0 = max(0, apex - 10)
    iin = a0 + int(np.argmax(s[a0:apex + 1]))
    b1 = min(n, apex + 11)
    iout = apex + int(np.argmax(s[apex:b1]))
    pre, post = slice(max(0, i - 8), max(1, i - 3)), slice(min(n - 1, i + 3), min(n, i + 8))
    v1 = np.array([vx[pre].mean(), vy[pre].mean()])
    v2 = np.array([vx[post].mean(), vy[post].mean()])
    cosang = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    # physics: split |a| into tangential (brake / drive along the path) and normal (turning, v^2/r) parts
    sw = np.maximum(s[lo:hi], 0.3)
    a_t = np.abs(vx[lo:hi] * ax[lo:hi] + vy[lo:hi] * ay[lo:hi]) / sw
    a_n = np.abs(vx[lo:hi] * ay[lo:hi] - vy[lo:hi] * ax[lo:hi]) / sw
    # unsmoothed second difference of position, to show how much the 0.7 s smoother flattens the peak
    rax, ray = np.gradient(np.gradient(x, DT), DT), np.gradient(np.gradient(y, DT), DT)
    raw_pk = float(np.hypot(rax, ray)[lo:hi].max() / G_YD)
    return dict(i_break=i, i_apex=apex, depth=float(cum[i]),
                angle=float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))),
                v_in=float(s[iin]), v_apex=float(s[apex]), v_out=float(s[iout]),
                decel=float((s[iin] - s[apex]) / max((apex - iin) * DT, DT)),
                reaccel=float((s[iout] - s[apex]) / max((iout - apex) * DT, DT)),
                bg=float(amag[lo:hi].mean() / G_YD), bg_t=float(a_t.mean() / G_YD), bg_n=float(a_n.mean() / G_YD),
                pk=float(amag[lo:hi].max() / G_YD), raw_pk=raw_pk)


def clusters(cl):
    cl = np.asarray(cl)
    return [np.where(cl == c)[0] for c in pd.unique(cl)]


def boot_rho(a, b, cluster=None, n=NBOOT, q=(5, 95)):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    rho = spearmanr(a, b)[0]
    groups = clusters(np.asarray(cluster)[m]) if cluster is not None else None
    out = []
    for _ in range(n):
        if groups is None:
            ix = RNG.integers(0, len(a), len(a))
        else:
            ix = np.concatenate([groups[k] for k in RNG.integers(0, len(groups), len(groups))])
        out.append(spearmanr(a[ix], b[ix])[0])
    lo, hi = np.nanpercentile(out, q)
    return float(rho), float(lo), float(hi), int(len(a))


def perm_p_rho(a, b, n=NPERM):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r0 = abs(spearmanr(a, b)[0])
    cnt = sum(abs(spearmanr(a, RNG.permutation(b))[0]) >= r0 for _ in range(n))
    return (1 + cnt) / (1 + n)


COMP = ["bg", "decel", "v_apex", "reaccel"]
COMP_LABEL = {"bg": "Break G", "decel": "Decel into break", "v_apex": "Speed through break",
              "reaccel": "Re-accel out of break"}


def icc1(df, col, gcol="nfl_id"):
    k = df.groupby(gcol).size()
    k0 = (len(df) - (k ** 2).sum() / len(df)) / (len(k) - 1)
    gm = df[col].mean()
    msb = (k * (df.groupby(gcol)[col].mean() - gm) ** 2).sum() / (len(k) - 1)
    msw = ((df[col] - df.groupby(gcol)[col].transform("mean")) ** 2).sum() / (len(df) - len(k))
    return (msb - msw) / (msb + (k0 - 1) * msw)


# ----------------------------------------------------------------------------- Combine route drills
players = pd.read_parquet(P / "players.parquet")
cres = pd.read_parquet(P / "combine_results.parquet")
events = pd.read_parquet(P / "combine_events.parquet")
DRILL_FAMILY = {"SLANT_ROUTE_LEFT": "SLANT", "SLOT_STRIKE_ROUTE_LEFT": "SLANT", "SPEED_OUT_ROUTE_LEFT": "OUT",
                "CURL_ROUTE_RIGHT": "HITCH", "COMEBACK_ROUTE_RIGHT": "HITCH", "HOOK_ROUTE_RIGHT": "HITCH",
                "DAGGER_ROUTE_LEFT": "IN", "IN_ROUTE_LEFT": "IN", "SLOT_SAIL_ROUTE_RIGHT": "CORNER",
                "CORNER_ROUTE": "CORNER"}
ev = events[events.drill_name.isin(DRILL_FAMILY) & events.position_group.isin(["WR", "TE"])]
frames = pd.read_parquet(P / "combine_frames.parquet", filters=[("event_id", "in", ev.event_id.tolist())])
REPS = {}
for eid, d in frames.groupby("event_id", sort=False):
    pl_ = d[d.entity_type == "PLAYER"].sort_values("frame")
    bl = d[d.entity_type == "BALL"]
    x, y = pl_.x.to_numpy(), pl_.y.to_numpy()
    i1 = len(x)
    if len(bl):  # stop at the catch so the ball-in-hand phase is not graded
        m = pl_[["frame", "x", "y"]].merge(bl[["frame", "x", "y"]], on="frame", suffixes=("", "_b"))
        c = m.frame[np.hypot(m.x - m.x_b, m.y - m.y_b) < 1.0]
        if len(c):
            i1 = int(np.searchsorted(pl_.frame.to_numpy(), c.iloc[0]))
    REPS[eid] = (x, y, i1)
EVK = ev[["event_id", "nfl_id", "draft_year", "drill_name", "position_group"]]


def combine_reps(spec):
    turn, hw, vin = spec
    rows = []
    for eid, (x, y, i1) in REPS.items():
        r = break_mechanics(x, y, 0, i1, min_turn=turn, entry_speed=vin, hw=hw)
        if r:
            rows.append(dict(event_id=eid, **r))
    cb_ = pd.DataFrame(rows).merge(EVK, on="event_id")
    cb_["family"] = cb_.drill_name.map(DRILL_FAMILY)
    for m_ in COMP:
        cb_[m_ + "_z"] = cb_.groupby(["drill_name", "draft_year"])[m_].transform(zs)
    return cb_


def player_scores(cb_, comps=COMP):
    cpd_ = cb_.groupby(["nfl_id", "drill_name", "family"], as_index=False)[[m_ + "_z" for m_ in COMP] + ["bg"]].mean()
    cpl = cpd_.groupby("nfl_id")[[m_ + "_z" for m_ in COMP]].mean()
    cpl["n_drills"] = cpd_.groupby("nfl_id").size()
    info = {}
    for m_ in comps:
        icc = max(icc1(cpd_, m_ + "_z"), 0.01)
        vt = cpd_[m_ + "_z"].var()
        var_true, var_err = icc * vt, (1 - icc) * vt
        nd = cpl.n_drills
        post_var = 1 / (1 / var_true + nd / var_err)
        shr = post_var * (nd * cpl[m_ + "_z"] / var_err)
        cpl[m_.upper() + "_S"] = shr / np.sqrt(var_true)
        cpl[m_.upper() + "_SD"] = np.sqrt(post_var) / np.sqrt(var_true)
        info[m_] = (icc, np.sqrt(var_true))
    return cpd_, cpl, info


cb = combine_reps(DEFAULT_SPEC)
put("n_drill_reps", len(ev)); put("pct_drill_detected", 100 * len(cb) / len(ev), 0)
put("bg_drill_median", cb.bg.median())
put("min_path_yd", 1.5, 1); put("entry_speed_yds", DEFAULT_SPEC[2], 1); put("window_s", 2 * DEFAULT_SPEC[1] * DT, 1)
cpd, cplayer, INFO = player_scores(cb)
put("n_combine_players", len(cplayer))
put("median_drills_per_player", int(cplayer.n_drills.median()))

# L1: split-half reliability across disjoint drill sets
HALF_A = {"SLANT_ROUTE_LEFT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT", "SPEED_OUT_ROUTE_LEFT", "HOOK_ROUTE_RIGHT"}
REL = {}
for m in COMP:
    ha = cpd[cpd.drill_name.isin(HALF_A)].groupby("nfl_id")[m + "_z"].mean()
    hb = cpd[~cpd.drill_name.isin(HALF_A)].groupby("nfl_id")[m + "_z"].mean()
    hh = pd.concat([ha, hb], axis=1, keys=["a", "b"]).dropna()
    REL[m] = boot_rho(hh.a, hh.b)
    put(f"rel_{m}", REL[m][0])
    if m == "bg":
        HH_BG = hh
        put("rel_bg_lo", REL[m][1]); put("rel_bg_hi", REL[m][2])
put("rel_n", REL["bg"][3])
put("rel_bg_p", perm_p_rho(HH_BG.a, HH_BG.b), 3)
icc_bg = INFO["bg"][0]
put("rel_7drills", 7 * icc_bg / (1 + 6 * icc_bg))
cplayer["BG"] = cplayer.BG_S
cplayer["BG_lo"] = cplayer.BG_S - 1.2816 * cplayer.BG_SD
cplayer["BG_hi"] = cplayer.BG_S + 1.2816 * cplayer.BG_SD
put("g_true_sd", float(INFO["bg"][1] * cb.groupby(["drill_name", "draft_year"]).bg.std().mean()), 3)

# physics of the window: share of |a| that is braking/driving (tangential) vs turning (normal); smoother attenuation
put("drill_at_g", cb.bg_t.median()); put("drill_an_g", cb.bg_n.median())
put("smooth_pk_pct", 100 * (cb.pk / cb.raw_pk).median(), 0)
put("drill_pk_g", cb.pk.median()); put("drill_rawpk_g", cb.raw_pk.median())

# ---- reliability by mixed model (rep level): bg_z ~ drill x year fixed effects + player + player x drill random
# effects + rep noise. The drill main effect is absorbed by the drill x year fixed effects.
import statsmodels.formula.api as smf  # noqa: E402
import warnings  # noqa: E402

att = events.set_index("event_id").attempt_no
cbm = cb.assign(dy=cb.drill_name + "_" + cb.draft_year.astype(str), att=cb.event_id.map(att))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    MM = smf.mixedlm("bg_z ~ 0 + C(dy)", cbm, groups="nfl_id", re_formula="1",
                     vc_formula={"pxd": "0 + C(drill_name)"}).fit(reml=True)
VP, VPD, VE = float(MM.cov_re.iloc[0, 0]), float(MM.vcomp[0]), float(MM.scale)
put("mm_vp", VP); put("mm_vpd", VPD, 3); put("mm_ve", VE)
put("mm_n_repeat_pairs", int((cbm.groupby(["nfl_id", "drill_name"]).size() >= 2).sum()))


def rel_dr(nd, nr):
    return VP / (VP + (VPD + VE / nr) / nd)


put("mm_rel_7x1", rel_dr(7, 1)); put("mm_rel_7x2", rel_dr(7, 2))
# protocol: second rep on the three drills that load best on the player's other drills (item-rest correlation)
cpd_m = cbm.groupby(["nfl_id", "drill_name"]).bg_z.mean().reset_index()
tot_ = cpd_m.groupby("nfl_id").bg_z.agg(["sum", "count"])
cpd_m = cpd_m.join(tot_, on="nfl_id")
cpd_m = cpd_m[cpd_m["count"] >= 3]
cpd_m["rest"] = (cpd_m["sum"] - cpd_m.bg_z) / (cpd_m["count"] - 1)
ITEM = (cpd_m.groupby("drill_name").apply(lambda d: pd.Series(dict(n=len(d), r=spearmanr(d.bg_z, d.rest)[0])),
                                          include_groups=False))
ITEM = ITEM[ITEM.n >= 50].sort_values("r", ascending=False)
TOP3 = list(ITEM.index[:3])
NUM["protocol_drills"] = ", ".join(TOP3)
for i_, d_ in enumerate(TOP3):
    put(f"item_r{i_}", ITEM.loc[d_, "r"])
put("item_r_low", ITEM.r.iloc[-1]); NUM["item_low_drill"] = ITEM.index[-1]
# 7 drills, second rep on the top 3 -> 10 reps; with player x drill variance ~0 the reliability depends on total reps
put("mm_rel_7p3", VP / (VP + 1 / (3 / (VPD + VE / 2) + 4 / (VPD + VE))))
put("mm_rel_gain_per_rep", (rel_dr(7, 2) - rel_dr(7, 1)) / 7, 3)
# empirical-Bayes player score from the mixed model, in sd of the true player effect
nrep = cbm.groupby("nfl_id").size()
sumz = cbm.groupby("nfl_id").bg_z.sum() - cbm.groupby("nfl_id").apply(
    lambda d: sum(MM.fe_params.get(f"C(dy)[{v}]", 0.0) for v in d.dy), include_groups=False)
npd = cbm.groupby("nfl_id").drill_name.nunique()
# with VPD ~ 0 this is the exact BLUP; otherwise an approximation that treats reps as exchangeable
cplayer["BG_MM"] = (VP * sumz / (VE + VPD * nrep / npd + nrep * VP)) / np.sqrt(VP)
cplayer["MM_REL"] = 1 - (VE + VPD * nrep / npd) / (VE + VPD * nrep / npd + nrep * VP)
put("mm_rel_median", cplayer.MM_REL.median())
put("rho_bg_mm", spearmanr(cplayer.BG, cplayer.BG_MM)[0])

tick('combine done')
# ----------------------------------------------------------------------------- baseline table
cr = cres.set_index("nfl_id")
pb = players.set_index("nfl_id")[["draft_overall_pick", "position_group", "draft_year", "display_name"]].join(
    cr[["forty", "combine_weight", "ngs_athleticism_score", "three_cone", "short_shuttle"]])
pb = pb[pb.position_group.isin(["WR", "TE"])].copy()
pb["logpick"] = np.log(pb.draft_overall_pick.fillna(262))
pb["te"] = (pb.position_group == "TE").astype(float)
for c in ("forty", "combine_weight", "ngs_athleticism_score", "three_cone", "short_shuttle"):
    pb[c + "_s"] = pb.groupby(["draft_year", "position_group"])[c].transform(zs)
    pb[c + "_f"] = pb[c + "_s"].fillna(0.0)
BASE = ["logpick", "forty_f", "combine_weight_f", "ngs_athleticism_score_f", "te"]
X = cplayer.join(pb, how="inner")
put("n_all", len(X))
put("rho_bg_forty", spearmanr(X.BG, X.forty, nan_policy="omit")[0])

# ----------------------------------------------------------------------------- season SOE (full-season NGS)
ss = pd.read_parquet(P / "player_season_stats.parquet")
ssa = ss[ss.position_group.isin(["WR", "TE"])].copy()
ss = ssa[ssa.routes.gt(0) & ssa.avg_separation.notna()].copy()
ss["slot"] = (ss.main_alignment == "SLOT_WR").astype(float)
ss["inl"] = (ss.main_alignment == "TE").astype(float)
ss["te_p"] = (ss.position_group == "TE").astype(float)
ss["cush"] = ss.avg_cushion.fillna(ss.avg_cushion.mean())
A_ss = np.column_stack([np.ones(len(ss)), ss.slot, ss.inl, ss.te_p, ss.cush,
                        (ss.season == 2024).astype(float), (ss.season == 2025).astype(float)])
w_ss = np.sqrt(ss.routes.to_numpy(float))
b_ss = np.linalg.lstsq(A_ss * w_ss[:, None], ss.avg_separation.to_numpy() * w_ss, rcond=None)[0]
ss["ssoe"] = ss.avg_separation - A_ss @ b_ss
put("cush_coef", b_ss[4])
put("n_player_seasons", len(ss))
pv = ss.pivot_table(index="nfl_id", columns="season", values="ssoe")
rv = ss.pivot_table(index="nfl_id", columns="season", values="routes")
yy = []
for a_, b_ in ((2023, 2024), (2024, 2025)):
    mm = pv[[a_, b_]].dropna()
    mm = mm[(rv.loc[mm.index, a_] >= 100) & (rv.loc[mm.index, b_] >= 100)]
    yy.append(mm.rename(columns={a_: "y1", b_: "y2"}))
YY = pd.concat(yy)
ry_ = boot_rho(YY.y1, YY.y2)
put("ssoe_stab", ry_[0]); put("ssoe_stab_lo", ry_[1]); put("ssoe_stab_hi", ry_[2]); put("ssoe_stab_n", ry_[3])
agg = ss.groupby("nfl_id").apply(lambda d: pd.Series(dict(
    ssoe=np.average(d.ssoe, weights=d.routes), routes=d.routes.sum(),
    sep=np.average(d.avg_separation, weights=d.routes), tg=d.targets.sum(), yds=d.rec_yards.sum(),
    rec=d.receptions.sum())), include_groups=False)
D = X.join(agg, how="inner")
D["tprr"] = D.tg / D.routes
D["yprr"] = D.yds / D.routes
D["catch_rate"] = D.rec / D.tg.clip(lower=1)
put("n_main", len(D)); put("n_main_wr", int((D.te == 0).sum()))
put("routes_median", int(D.routes.median()))


def wfit(df, cols, ycol, wcol="routes"):
    w = df[wcol].to_numpy(float)
    A = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    W = np.sqrt(w)
    yv = df[ycol].to_numpy(float)
    b = np.linalg.lstsq(A * W[:, None], yv * W, rcond=None)[0]
    r = yv - A @ b
    s2 = (w * r ** 2).sum() / (len(df) - A.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv((A * w[:, None]).T @ A)))
    return b, se


def wpred(trn, tst, cols, ycol, wcol="routes"):
    b, _ = wfit(trn, cols, ycol, wcol)
    return np.column_stack([np.ones(len(tst)), tst[cols].to_numpy(float)]) @ b


def boot_coef(df, cols, ycol, wcol="routes", n=NBOOT, k=-1):
    bs = []
    for _ in range(n):
        ix = RNG.integers(0, len(df), len(df))
        bs.append(wfit(df.iloc[ix], cols, ycol, wcol)[0][k])
    return np.array(bs)


# ---- primary: season SOE ~ baseline + Break G
b, se = wfit(D, BASE + ["BG"], "ssoe")
bs = boot_coef(D, BASE + ["BG"], "ssoe")
coef_in = IN_YD * b[-1]
lo_in, hi_in = IN_YD * np.percentile(bs, 5), IN_YD * np.percentile(bs, 95)
put("prim_in", coef_in, 1); put("prim_lo_in", lo_in, 1); put("prim_hi_in", hi_in, 1)
put("prim_p", 2 * norm.sf(abs(b[-1] / se[-1])), 2)
se_b = IN_YD * bs.std(ddof=1)
p_up = norm.cdf((coef_in - SESOI_IN) / se_b)
p_dn = norm.sf((coef_in + SESOI_IN) / se_b)
put("sesoi_in", SESOI_IN, 0)
put("tost_p", max(p_up, p_dn), 3)
put("mde_in", (1.96 + 0.84) * se_b, 1)
# 2026 prospective power: one more class of the median size
n26 = int(np.median([int((D.draft_year == c).sum()) for c in CLASSES]))
se26 = se_b * np.sqrt(len(D) / (len(D) + n26))
put("n26", n26); put("mde26_in", (1.96 + 0.84) * se26, 1)
put("power26_3in", norm.cdf(SESOI_IN / se26 - 1.96), 2)
D["one"] = 1.0
DW = D[D.te == 0]
bw, _ = wfit(DW, [c for c in BASE if c != "te"] + ["BG"], "ssoe")
put("prim_wr_in", IN_YD * bw[-1], 1)
D100 = D[D.routes >= 100]
b1_, _ = wfit(D100, BASE + ["BG"], "ssoe")
put("prim_r100_in", IN_YD * b1_[-1], 1); put("prim_r100_n", len(D100))
bu, _ = wfit(D, BASE + ["BG"], "ssoe", "one")
put("prim_unw_in", IN_YD * bu[-1], 1)

# ---- out of class: leave one Combine class out
D["fc0"], D["fc1"] = np.nan, np.nan
CLS = []
for c in CLASSES:
    trn, tst = D[D.draft_year != c], D[D.draft_year == c]
    D.loc[tst.index, "fc0"] = wpred(trn, tst, BASE, "ssoe")
    D.loc[tst.index, "fc1"] = wpred(trn, tst, BASE + ["BG"], "ssoe")


# ---- PRIMARY: held-out offset slope (forecast from the other classes is the offset; class intercepts)
D["res0"] = D.ssoe - D.fc0
CLD = np.column_stack([(D.draft_year == c).to_numpy(float) for c in CLASSES])
W_R = np.sqrt(D.routes.to_numpy(float))
RES0 = D.res0.to_numpy(float)


def off_slope(bgv, rows=None):
    rows = np.arange(len(D)) if rows is None else rows
    A = np.column_stack([CLD[rows], bgv[rows]])
    keep = A[:, :-1].sum(0) > 0
    A = np.column_stack([A[:, :-1][:, keep], A[:, -1]])
    w = W_R[rows]
    return np.linalg.lstsq(A * w[:, None], RES0[rows] * w, rcond=None)[0][-1]


BGV = D.BG.to_numpy(float)
ho = IN_YD * off_slope(BGV)
hob_fixed = np.array([IN_YD * off_slope(BGV, RNG.integers(0, len(D), len(D))) for _ in range(NBOOT)])
put("ho_fixed_lo", np.percentile(hob_fixed, 5), 1); put("ho_fixed_hi", np.percentile(hob_fixed, 95), 1)
cls_arr0 = D.draft_year.to_numpy()
XB_ = np.column_stack([np.ones(len(D)), D[BASE].to_numpy(float)])
Y_ = D.ssoe.to_numpy(float)
WT_ = D.routes.to_numpy(float)


def _wls(A, yv, w):
    s_ = np.sqrt(w)
    return np.linalg.lstsq(A * s_[:, None], yv * s_, rcond=None)[0]


def ho_refit(ix, bgv):
    """Held-out slope with the no-tracking forecast refitted inside the resample (carries forecast error)."""
    Xi, yi, wi, ci, bi = XB_[ix], Y_[ix], WT_[ix], cls_arr0[ix], bgv[ix]
    res = np.empty(len(ix))
    for c in CLASSES:
        t = ci == c
        res[t] = yi[t] - Xi[t] @ _wls(Xi[~t], yi[~t], wi[~t])
    A = np.column_stack([(ci == c).astype(float) for c in CLASSES] + [bi])
    return IN_YD * _wls(A, res, wi)[-1]


def strat_ix():
    return np.concatenate([RNG.choice(np.where(cls_arr0 == c)[0], int((cls_arr0 == c).sum())) for c in CLASSES])


assert abs(ho_refit(np.arange(len(D)), BGV) - ho) < 1e-9
hob = np.array([ho_refit(strat_ix(), BGV) for _ in range(NBOOT)])
put("ho_in", ho, 1); put("ho_lo", np.percentile(hob, 5), 1); put("ho_hi", np.percentile(hob, 95), 1)
# mixed-model (empirical-Bayes) score: same test, same refit bootstrap
BGMM = D.BG_MM.to_numpy(float)
ho_mm = ho_refit(np.arange(len(D)), BGMM)
hob_mm = np.array([ho_refit(strat_ix(), BGMM) for _ in range(NBOOT)])
put("ho_mm_in", ho_mm, 1); put("ho_mm_lo", np.percentile(hob_mm, 5), 1); put("ho_mm_hi", np.percentile(hob_mm, 95), 1)
# calibration of the held-out no-tracking forecast, by class (slope of outcome on forecast)
for c in CLASSES:
    t = cls_arr0 == c
    f_ = XB_[t] @ _wls(XB_[~t], Y_[~t], WT_[~t])
    put(f"calib{c}", _wls(np.column_stack([np.ones(t.sum()), f_]), Y_[t], WT_[t])[1])
NULL_HO = []
for _ in range(NPERM):
    pv_ = BGV.copy()
    for c in CLASSES:
        ix = np.where(cls_arr0 == c)[0]
        pv_[ix] = BGV[RNG.permutation(ix)]
    NULL_HO.append(IN_YD * off_slope(pv_))
NULL_HO = np.array(NULL_HO)
put("ho_shuffle_p", (1 + (np.abs(NULL_HO) >= abs(ho)).sum()) / (1 + NPERM), 2)
put("ho_null_q95", np.percentile(np.abs(NULL_HO), 95), 1)
se_ho = hob.std(ddof=1)
put("ho_up_p", norm.cdf((ho - SESOI_IN) / se_ho), 4)
put("ho_tost_p", max(norm.cdf((ho - SESOI_IN) / se_ho), norm.sf((ho + SESOI_IN) / se_ho)), 3)
HO_CLS = []
for c in CLASSES:
    ix = np.where(cls_arr0 == c)[0]
    v = IN_YD * off_slope(BGV, ix)
    vb = np.array([IN_YD * off_slope(BGV, ix[RNG.integers(0, len(ix), len(ix))]) for _ in range(1000)])
    HO_CLS.append(dict(lab=f"Held out: {c} class (n={len(ix)})", b=v, lo=np.percentile(vb, 5),
                       hi=np.percentile(vb, 95), grp="class"))
    put(f"ho{c}_in", v, 1); put(f"ho{c}_lo", np.percentile(vb, 5), 1); put(f"ho{c}_hi", np.percentile(vb, 95), 1)
put("ho_n_neg_cls", int(sum(r["b"] < 0 for r in HO_CLS)))
# frozen rule, hashed so a 2026 re-score can prove nothing moved
RULE = dict(metric="Break G: mean |accel| (g) over 1.0 s centred on sharpest point of first >=30 deg turn after 1.5 yd,"
                   " entry > 2.5 yd/s; z within drill x year; mean over route drills; reliability-shrunk",
            drills=sorted(DRILL_FAMILY), nfl_families=FAMILIES, smoothing="savgol w7 o2",
            outcome="route-weighted season SOE (NGS avg separation adjusted for slot, inline, TE, cushion, season)",
            forecast=BASE, test="held-out offset slope, class intercepts, routes weights, within-class shuffle null",
            margin_in=SESOI_IN, score_class=2026, score_after="2026 regular season")
blob = json.dumps(RULE, sort_keys=True).encode()
NUM["rule_sha"] = hashlib.sha256(blob).hexdigest()[:12]
(OUT / "frozen_rule.json").write_text(json.dumps(dict(rule=RULE, sha256=hashlib.sha256(blob).hexdigest()), indent=1))


def wr2gain(df):
    w, yv = df.routes.to_numpy(), df.ssoe.to_numpy()
    return 1 - (w * (yv - df.fc1.to_numpy()) ** 2).sum() / (w * (yv - df.fc0.to_numpy()) ** 2).sum()


put("loco_gain_pct", 100 * wr2gain(D), 1)
gl = np.array([wr2gain(D.iloc[RNG.integers(0, len(D), len(D))]) for _ in range(NBOOT)])
put("loco_lo_pct", 100 * np.percentile(gl, 5), 1); put("loco_hi_pct", 100 * np.percentile(gl, 95), 1)
for c in CLASSES:
    tst = D[D.draft_year == c]
    # held-out sign: residual of the baseline forecast vs Break G inside the held-out class
    res = tst.ssoe - tst.fc0
    rr = boot_rho(tst.BG, res, n=1000)
    put(f"cls{c}_n", len(tst))
    put(f"cls{c}_rho", rr[0])

# ---- three coach components and max-t family-wise p
SEC = ["DECEL_S", "V_APEX_S", "REACCEL_S"]
cls_arr = D.draft_year.to_numpy()


def tstat(df, xcol):
    bb, se_ = wfit(df, BASE + [xcol], "ssoe")
    return bb[-1] / se_[-1]


COMPEFF = []
for xc, lab in zip(SEC, ["Decel into break", "Speed through break", "Re-accel out of break"]):
    bb, _ = wfit(D, BASE + [xc], "ssoe")
    bq = boot_coef(D, BASE + [xc], "ssoe", n=1000)
    put(f"comp_{xc.lower()[:-2]}_in", IN_YD * bb[-1], 1)
    COMPEFF.append(dict(lab=lab + " (component)", b=IN_YD * bb[-1], lo=IN_YD * np.percentile(bq, 5),
                        hi=IN_YD * np.percentile(bq, 95), grp="comp"))
t_obs = {xc: abs(tstat(D, xc)) for xc in ["BG"] + SEC}
best_x = max(t_obs, key=t_obs.get)
maxt = []
Xm = D[["BG"] + SEC].to_numpy(float)
for _ in range(NPERM):
    perm = np.arange(len(D))
    for c in CLASSES:
        ix = np.where(cls_arr == c)[0]
        perm[ix] = RNG.permutation(ix)
    Dp = D.copy()
    Dp[["BG"] + SEC] = Xm[perm]
    maxt.append(max(abs(tstat(Dp, xc)) for xc in ["BG"] + SEC))
put("sec_fwer_p", (1 + (np.array(maxt) >= t_obs[best_x]).sum()) / (1 + NPERM), 2)

# ---- disclosed secondary: targets per route run (an earlier round's fragile lead; not separation)
bt, set_ = wfit(D, BASE + ["BG"], "tprr")
put("tprr_pts", 100 * bt[-1], 1); put("tprr_t", bt[-1] / set_[-1], 1)
DN = D[D.display_name != "Puka Nacua"]
btn, setn = wfit(DN, BASE + ["BG"], "tprr")
put("tprr_nonacua_pts", 100 * btn[-1], 1); put("tprr_nonacua_t", btn[-1] / setn[-1], 1)
bsoe_tp, _ = wfit(D, BASE + ["BG", "ssoe"], "tprr")
put("tprr_given_soe_pts", 100 * bsoe_tp[-2], 1)
# multiplicity: three secondary usage outcomes were looked at (targets/route, yards/route, catch rate). Family-wise p
# for targets/route = within-class shuffle of Break G, max |t| over the three, with and without Puka Nacua.
SECOND = ["tprr", "yprr", "catch_rate"]


def sec_fwer(df):
    cl_ = df.draft_year.to_numpy()
    tt = {o: abs(wfit(df, BASE + ["BG"], o)[0][-1] / wfit(df, BASE + ["BG"], o)[1][-1]) for o in SECOND}
    mx, bgv_ = [], df.BG.to_numpy(float)
    for _ in range(NPERM // 2):
        pb_ = bgv_.copy()
        for c in CLASSES:
            ix = np.where(cl_ == c)[0]
            pb_[ix] = bgv_[RNG.permutation(ix)]
        dp = df.assign(BG=pb_)
        mx.append(max(abs(wfit(dp, BASE + ["BG"], o)[0][-1] / wfit(dp, BASE + ["BG"], o)[1][-1]) for o in SECOND))
    return (1 + (np.array(mx) >= tt["tprr"]).sum()) / (1 + len(mx))


put("tprr_fwer_p", sec_fwer(D), 3)
put("tprr_nonacua_fwer_p", sec_fwer(DN), 2)

# ----------------------------------------------------------------------------- what clubs pay vs what it buys
# price: log pick on the measure (+ TE), all WR/TE with the measure (UDFA = pick 262); payoff: season SOE inches per
# sd beyond log pick + TE. Each measure in sd units; times are sign-flipped so faster = +.
PRICE = []
REFX = [("BG", "Break G (route drills)", 1), ("forty_s", "Official 40", -1), ("three_cone_s", "3-cone", -1),
        ("short_shuttle_s", "Short shuttle", -1), ("combine_weight_s", "Weight", 1),
        ("ngs_athleticism_score_s", "NGS athleticism score", 1)]
for xc, lab, sgn in REFX:
    xx = X[X[xc].notna()].copy()
    xx["xx"] = sgn * xx[xc] / xx[xc].std()
    xx["one"] = 1.0
    bp, _ = wfit(xx, ["te", "xx"], "logpick", "one")
    bpb = boot_coef(xx, ["te", "xx"], "logpick", "one", n=1000)
    dd = D[D[xc].notna()].copy()
    dd["xx"] = sgn * dd[xc] / dd[xc].std()
    bv, _ = wfit(dd, ["logpick", "te", "xx"], "ssoe")
    bvb = boot_coef(dd, ["logpick", "te", "xx"], "ssoe", n=1000)
    pr_ = -100 * (np.exp(bp[-1]) - 1)  # % earlier pick per sd
    PRICE.append(dict(lab=lab, x=xc, price=pr_, plo=-100 * (np.exp(np.percentile(bpb, 95)) - 1),
                      phi=-100 * (np.exp(np.percentile(bpb, 5)) - 1), pay=IN_YD * bv[-1],
                      ylo=IN_YD * np.percentile(bvb, 5), yhi=IN_YD * np.percentile(bvb, 95), n_price=len(xx),
                      n_pay=len(dd)))
    k = (xc[:-2] if xc.endswith("_s") else xc).lower()
    put(f"price_{k}", pr_, 0); put(f"pay_{k}_in", IN_YD * bv[-1], 1)
    put(f"pay_{k}_lo", IN_YD * np.percentile(bvb, 5), 1); put(f"pay_{k}_hi", IN_YD * np.percentile(bvb, 95), 1)
    put(f"n_pay_{k}", len(dd)); put(f"n_price_{k}", len(xx))
    if xc == "BG":
        put("price_bg_lo", PRICE[-1]["plo"], 0); put("price_bg_hi", PRICE[-1]["phi"], 0)
PRICE = pd.DataFrame(PRICE)
# Break G price beyond the full no-tracking athletic profile (40, weight, NGS, TE)
X["one"] = 1.0
pcols = ["forty_f", "combine_weight_f", "ngs_athleticism_score_f", "te", "BG"]
bpf, sepf = wfit(X, pcols, "logpick", "one")
put("price_bg_full", -100 * (np.exp(bpf[-1]) - 1), 0)
put("price_bg_full_p", 2 * norm.sf(abs(bpf[-1] / sepf[-1])), 2)
EX_PICK = 100
put("ex_pick", EX_PICK); put("ex_pick_moved", int(round(EX_PICK * np.exp(bpf[-1]))))

# ---- separation -> catches: season catch rate on separation when targeted (20+ targets)
s2 = ssa[(ssa.targets >= 20) & ssa.avg_separation_targeted.notna()].copy()
s2["cr"] = s2.receptions / s2.targets
s2["one"] = 1.0
bcr, secr = wfit(s2, ["avg_separation_targeted"], "cr", "targets")
put("cr_per_yd", 100 * bcr[-1], 1); put("cr_n", len(s2))
TG_SEASON = int(ssa[ssa.routes >= 200].targets.median())
put("tg_season", TG_SEASON)
catch_per_in = bcr[-1] / IN_YD * TG_SEASON
put("catch_upper", hi_in * catch_per_in, 1)
put("catch_sesoi", SESOI_IN * catch_per_in, 1)
put("catch_ho_hi", float(NUM["ho_hi"]) * catch_per_in, 2)
put("catch_weight", abs(float(PRICE.loc[PRICE.x == "combine_weight_s", "pay"].iloc[0])) * catch_per_in, 1)

tick('season tests done')
# ----------------------------------------------------------------------------- sensitivity grid over the definition
GRID = []
for turn in (30.0, 45.0):
    for hw in (3, 5, 7):
        for vin in (2.0, 2.5, 3.0):
            spec = (turn, hw, vin)
            if spec == DEFAULT_SPEC:
                cps = cplayer
            else:
                _, cps, _ = player_scores(combine_reps(spec), comps=["bg"])
            dd = D.drop(columns=[c for c in D.columns if c.startswith("BG")]).join(cps[["BG_S"]], how="inner")
            dd = dd.rename(columns={"BG_S": "BG"})
            bb, _ = wfit(dd, BASE + ["BG"], "ssoe")
            bq = boot_coef(dd, BASE + ["BG"], "ssoe", n=500)
            bgv_ = cps.BG_S.reindex(D.index).to_numpy(float)
            rows_ = np.where(~np.isnan(bgv_))[0]
            hog = IN_YD * off_slope(np.nan_to_num(bgv_), rows_)
            GRID.append(dict(turn=turn, win=2 * hw * DT, vin=vin, b=IN_YD * bb[-1], lo=IN_YD * np.percentile(bq, 5),
                             hi=IN_YD * np.percentile(bq, 95), n=len(dd), default=spec == DEFAULT_SPEC, ho=hog))
            put(f"grid_ho_{int(turn)}_{int(round(20 * hw * DT))}_{int(10 * vin)}", hog, 1)
GRID = pd.DataFrame(GRID)
put("grid_ho_min", GRID.ho.min(), 1); put("grid_ho_max", GRID.ho.max(), 1)
put("grid_n", len(GRID))
put("grid_min_in", GRID.b.min(), 1); put("grid_max_in", GRID.b.max(), 1)
put("grid_hi_max_in", GRID.hi.max(), 1)
put("grid_n_inside", int(((GRID.hi < SESOI_IN) & (GRID.lo > -SESOI_IN)).sum()))

tick('grid done')
# ----------------------------------------------------------------------------- NFL snaps: same code
gp = pd.read_parquet(P / "game_plays_sample.parquet")
gp = gp[gp.week.between(1, 18)]
gp = gp.merge(players[["nfl_id", "position_group"]], on="nfl_id")
gp = gp[gp.position_group.isin(["WR", "TE"])].copy()
gf = pd.read_parquet(P / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "x", "y", "event"])
gf = gf.merge(gp[["game_id", "play_id", "nfl_id"]], on=["game_id", "play_id", "nfl_id"])
grows, gtraces = [], {}
for k, d in gf.groupby(["game_id", "play_id", "nfl_id"], sort=False):
    d = d.sort_values("frame")
    e_ = d.event.fillna("").to_numpy()
    if not ((e_ == "ball_snap").any() and (e_ == "pass_forward").any()):
        continue
    i0 = int(np.argmax(e_ == "ball_snap"))
    ipf = int(np.argmax(e_ == "pass_forward"))
    xs_d = d.x.to_numpy()
    base = dict(game_id=k[0], play_id=k[1], nfl_id=k[2], t_throw=(ipf - i0) * DT, ipf=ipf, dx_thr=xs_d[ipf] - xs_d[i0])
    r = break_mechanics(d.x.to_numpy(), d.y.to_numpy(), i0, min(len(d), ipf + 6))
    if r:
        base.update(r)
        gtraces[k] = (d.x.to_numpy(), d.y.to_numpy(), i0, ipf)
    grows.append(base)
gb = pd.DataFrame(grows).merge(gp, on=["game_id", "play_id", "nfl_id"])
gb["sep"] = gb.separation_at_pass_forward
gb["depth_thr"] = np.where(gb.play_direction == "left", -1.0, 1.0) * gb.dx_thr  # yd downfield at the throw
gb["man"] = gb.team_coverage_type.fillna("").str.contains("MAN")
gb["slotp"] = (gb.lined_up_position == "SLOT_WR").astype(float)
gb["inline"] = gb.lined_up_position.isin(["TE", "FB", "RB"]).astype(float)
gb["cush"] = gb.cushion.fillna(gb.cushion.median())
gb["cush_na"] = gb.cushion.isna().astype(float)
gb["pl"] = gb.pass_length.fillna(gb.pass_length.median())
gb["route"] = gb.route_ran.fillna("OTHER")
ROUTES = sorted(gb.route.unique())


def design(df, with_t=True):
    cols = [np.ones(len(df)), df.pl, df.pl ** 2 / 100, df.cush, df.cush_na, df.man.astype(float), df.slotp, df.inline]
    cols += [df.t_throw] if with_t else []
    cols += [df.target.astype(float), df.down, df.yards_to_go]
    cols += [(df.route == r).astype(float) for r in ROUTES[1:]]
    return np.column_stack(cols)


dev = gb.season < 2025
beta_ctx = np.linalg.lstsq(design(gb[dev]), gb.loc[dev, "sep"].to_numpy(), rcond=None)[0]
gb["soe"] = gb.sep - design(gb) @ beta_ctx
# circularity check: snap SOE built without time to throw (cut-to-throw time is part of time to throw)
beta_nt = np.linalg.lstsq(design(gb[dev], False), gb.loc[dev, "sep"].to_numpy(), rcond=None)[0]
gb["soe_nt"] = gb.sep - design(gb, False) @ beta_nt
g = gb[gb.route_ran.isin(FAMILIES) & gb.bg.notna()].copy()
put("n_break_snaps", len(g))
put("bg_game_median", g.bg.median())
for m in COMP:
    g[m + "_gz"] = g.groupby(["route_ran", "position_group"])[m].transform(zs)
g["lead"] = (g.ipf - g.i_apex) * DT
q = g[g.lead >= LEAD_MIN].copy()
put("n_graded", len(q)); put("n_graded_players", q.nfl_id.nunique())
g = g.sort_values(["nfl_id", "game_id", "play_id"])
g["half"] = g.groupby("nfl_id").cumcount() % 2
a_ = g[g.half == 0].groupby("nfl_id").bg_gz.mean()
b2 = g[g.half == 1].groupby("nfl_id").bg_gz.mean()
hh = pd.concat([a_, b2], axis=1).dropna()
grel = boot_rho(hh.iloc[:, 0], hh.iloc[:, 1])
put("grel_bg", grel[0])
put("game_snaps_median", int(g.groupby("nfl_id").size().median()))
rg_full = 2 * grel[0] / (1 + grel[0])
put("ceiling", np.sqrt(max(float(NUM["rel_7drills"]), 0) * max(rg_full, 0)))
cfam = cpd.groupby(["nfl_id", "family"]).bg_z.mean()
gfam = g.groupby(["nfl_id", "route_ran"]).bg_gz.mean()
gfam.index.names = ["nfl_id", "family"]
jf = pd.concat([cfam.rename("d"), gfam.rename("g")], axis=1).dropna().reset_index()
fr = boot_rho(jf.d, jf.g, cluster=jf.nfl_id)
put("fam_rho", fr[0]); put("fam_lo", fr[1]); put("fam_hi", fr[2]); put("fam_n", fr[3])
FAMR = {f: boot_rho(jf[jf.family == f].d, jf[jf.family == f].g, n=500) for f in FAMILIES}
put("fam_max_rho", max(v[0] for v in FAMR.values()))
NUM["fam_max_name"] = max(FAMR, key=lambda f: FAMR[f][0]).lower()
pg = g.groupby("nfl_id").bg_gz.mean().rename("g_bg")
JP = X.join(pg, how="inner")
pr = boot_rho(JP.BG, JP.g_bg)
put("pl_rho", pr[0]); put("pl_lo", pr[1]); put("pl_hi", pr[2]); put("pl_n", pr[3])
put("pl_p", perm_p_rho(JP.BG, JP.g_bg), 2)
# L3 within player
q["kept"] = q.v_apex / q.v_in
dm = q[["nfl_id", "bg_gz", "soe", "soe_nt", "lead", "t_throw", "v_in", "kept"]].dropna().copy()
for c_ in ("bg_gz", "soe", "soe_nt", "t_throw"):
    dm[c_ + "_w"] = dm[c_] - dm.groupby("nfl_id")[c_].transform("mean")
dm["bg_w"], dm["soe_w"] = dm.bg_gz_w, dm.soe_w
sr = boot_rho(dm.bg_w, dm.soe_w, cluster=dm.nfl_id)
put("snap_rho", sr[0]); put("snap_lo", sr[1]); put("snap_hi", sr[2])
put("snap_n_players_med", int(dm.groupby("nfl_id").size().median()))
# the earlier post hoc timing gap (pooled fifths of cut-to-throw time, 2nd minus 5th), three ways
kk = pd.qcut(dm.lead, 5, labels=False).to_numpy()
put("lead_diff_in", IN_YD * (dm.soe_w[kk == 1].mean() - dm.soe_w[kk == 4].mean()), 1)
put("lead_diff_nt_in", IN_YD * (dm.soe_nt_w[kk == 1].mean() - dm.soe_nt_w[kk == 4].mean()), 1)
A_l = np.column_stack([np.ones(len(dm))] + [(kk == k_).astype(float) for k_ in (0, 1, 2, 3)] + [dm.t_throw_w])
put("lead_diff_cov_in", IN_YD * np.linalg.lstsq(A_l, dm.soe_nt_w.to_numpy(), rcond=None)[0][2], 1)
put("corr_lead_tthrow", np.corrcoef(dm.lead - dm.groupby("nfl_id").lead.transform("mean"), dm.t_throw_w)[0, 1])
# all four candidate in-game splits on one footing: fifths of each snap's rank within the player's own snaps; outcome
# = snap SOE without time to throw, demeaned within player. Statistic = spread (max - min) of the five fifth means.
# Null: SOE shuffled within each player (player-clustered), max spread over the four splits -> family-wise p.
SPLITS = {"force": "bg_gz", "time": "lead", "entry": "v_in", "kept": "kept"}
SPLIT_LAB = {"force": "cut force", "time": "cut-to-throw time", "entry": "entry speed", "kept": "speed kept"}
for k_, c_ in SPLITS.items():
    dm["r_" + k_] = np.minimum((dm.groupby("nfl_id")[c_].rank(pct=True, method="average") * 5 - 1e-9).astype(int), 4)
SNT = dm.soe_nt_w.to_numpy()
RK = {k_: dm["r_" + k_].to_numpy() for k_ in SPLITS}


def spreads(sv):
    return {k_: IN_YD * (max(m_ := [sv[RK[k_] == j].mean() for j in range(5)]) - min(m_)) for k_ in SPLITS}


SPR = spreads(SNT)
grp_all = clusters(dm.nfl_id)
NULL_SPR = []
for _ in range(NPERM):
    sp_ = SNT.copy()
    for g_ in grp_all:
        sp_[g_] = SNT[RNG.permutation(g_)]
    NULL_SPR.append(spreads(sp_))
NULL_SPR = pd.DataFrame(NULL_SPR)
mx_ = NULL_SPR.max(axis=1).to_numpy()
for k_ in SPLITS:
    put(f"split_{k_}_in", SPR[k_], 1)
    put(f"split_{k_}_p", (1 + (NULL_SPR[k_] >= SPR[k_]).sum()) / (1 + NPERM), 2)
    put(f"split_{k_}_fwp", (1 + (mx_ >= SPR[k_]).sum()) / (1 + NPERM), 2)
best_split = max(SPR, key=SPR.get)
NUM["best_split"] = SPLIT_LAB[best_split]


def fifths_rank(k_):
    out = []
    for j in range(5):
        sel = RK[k_] == j
        d_ = dm[sel]
        grp = clusters(d_.nfl_id)
        bsq = [SNT[sel][np.concatenate([grp[i] for i in RNG.integers(0, len(grp), len(grp))])].mean()
               for _ in range(1000)]
        out.append(dict(k=j, m=IN_YD * SNT[sel].mean(), lo=IN_YD * np.percentile(bsq, 5),
                        hi=IN_YD * np.percentile(bsq, 95), n=int(sel.sum())))
    return pd.DataFrame(out)


QB, QL = fifths_rank("force"), fifths_rank("time")
put("snap_soe_sd_in", IN_YD * dm.soe_w.std(), 0)

# within-player dose response: each player's own slope (inches of snap SOE per sd of his own cut force) for players
# with 6+ graded breaks (the caterpillar), and the pooled slope on exactly those players and snaps
dm["bgs"] = dm.bg_w / dm.bg_w.std()
CAT = []
for pid_, d_ in dm.groupby("nfl_id"):
    if len(d_) < 6 or d_.bgs.std() == 0:
        continue
    xv, yv = d_.bgs.to_numpy(), IN_YD * d_.soe_w.to_numpy()
    bb_ = np.polyfit(xv, yv, 1)
    rs_ = yv - np.polyval(bb_, xv)
    se_ = np.sqrt((rs_ ** 2).sum() / (len(xv) - 2) / ((xv - xv.mean()) ** 2).sum())
    CAT.append(dict(nfl_id=pid_, b=bb_[0], lo=bb_[0] - 1.645 * se_, hi=bb_[0] + 1.645 * se_, n=len(xv)))
CAT = pd.DataFrame(CAT).sort_values("b").reset_index(drop=True)
dm6 = dm[dm.nfl_id.isin(CAT.nfl_id)]
grp6 = clusters(dm6.nfl_id)
POOL_B = IN_YD * np.polyfit(dm6.bgs, dm6.soe_w, 1)[0]
pb_s = []
for _ in range(NBOOT):
    ix = np.concatenate([grp6[i] for i in RNG.integers(0, len(grp6), len(grp6))])
    pb_s.append(IN_YD * np.polyfit(dm6.bgs.to_numpy()[ix], dm6.soe_w.to_numpy()[ix], 1)[0])
put("wp_slope_in", POOL_B, 1); put("wp_slope_lo", np.percentile(pb_s, 5), 1); put("wp_slope_hi", np.percentile(pb_s, 95), 1)
put("cat_n", len(CAT)); put("cat_snaps", len(dm6)); put("cat_pct_pos", 100 * (CAT.b > 0).mean(), 0)
put("cat_pct_cross", 100 * ((CAT.lo < 0) & (CAT.hi > 0)).mean(), 0)

tick('nfl snaps done')
# ============================================================================= round 12 additions
# ---- (a) quarterback adjustment by a crossed hierarchical model (revised round 12). The passer is named in every
# sampled play description. Snap SOE (context-adjusted, inches) = intercept + receiver effect + QB effect + noise, with
# receiver and QB effects random (REML, own implementation; checked against a dense brute-force likelihood, and it
# reaches a lower -2 log L than statsmodels' MixedLM on the same data). A player-season's QB environment is the mean
# QB BLUP over his sampled snaps that season, subtracted from season SOE. Interval on the QB variance share:
# receiver-cluster bootstrap. The model is refitted on the random half of the sampled snaps (no EPA selection).
import scipy.sparse as sps  # noqa: E402
from scipy.optimize import minimize  # noqa: E402


def crossed_reml(y, ga, gbb, starts=([0.8, 0.1, 0.1], [0.9, 0.05, 0.01]), profile_b=False):
    ia, ua = pd.factorize(ga)
    ib, ub = pd.factorize(gbb)
    n, qa, qb_ = len(y), len(ua), len(ub)
    rows = np.arange(n)
    Z = sps.hstack([sps.csr_matrix((np.ones(n), (rows, ia)), shape=(n, qa)),
                    sps.csr_matrix((np.ones(n), (rows, ib)), shape=(n, qb_))]).tocsr()
    ZtZ = (Z.T @ Z).toarray()
    Zty, Zt1 = Z.T @ y, np.asarray(Z.sum(0)).ravel()
    yy, sy = y @ y, y.sum()

    def parts(th):
        s2, ta, tb = np.exp(th)
        gv = np.r_[np.full(qa, ta), np.full(qb_, tb)]
        L = np.linalg.cholesky(np.diag(1 / gv) + ZtZ / s2)

        def sol(v):
            return np.linalg.solve(L.T, np.linalg.solve(L, v))
        ciy, ci1 = sol(Zty), sol(Zt1)
        a11 = n / s2 - Zt1 @ ci1 / s2 ** 2
        a1y = sy / s2 - Zt1 @ ciy / s2 ** 2
        ayy = yy / s2 - Zty @ ciy / s2 ** 2
        beta = a1y / a11
        ld = n * np.log(s2) + np.log(gv).sum() + 2 * np.log(np.diag(L)).sum()
        return ld + np.log(a11) + ayy - a1y ** 2 / a11, beta, sol, s2

    v0 = np.var(y)
    best = None
    for st in starts:  # keep the start with the lower -2 log L
        r_ = minimize(lambda th: parts(th)[0], np.log(np.array(st) * v0), method="L-BFGS-B",
                      bounds=[(-12, 14)] * 3)
        if best is None or r_.fun < best.fun:
            best = r_
    _, beta, sol, s2 = parts(best.x)
    u = sol(Z.T @ (y - beta)) / s2
    if profile_b:  # profile -2 log L over the second variance (the QB variance), others re-optimised at each value
        grid = np.r_[1e-6, np.geomspace(0.05, 400.0, 60)]
        prof = []
        for tb in grid:
            rr = minimize(lambda t2: parts(np.r_[t2, np.log(tb)])[0], best.x[:2], method="L-BFGS-B",
                          bounds=[(-12, 14)] * 2)
            prof.append((tb, rr.fun - best.fun, np.exp(rr.x).sum() + tb))
        return np.exp(best.x), pd.Series(u[:qa], ua), pd.Series(u[qa:], ub), pd.DataFrame(
            prof, columns=["tb", "d2ll", "tot"])
    return np.exp(best.x), pd.Series(u[:qa], ua), pd.Series(u[qa:], ub)


gq = gb.copy()
gq["qb"] = gq.play_description.str.extract(r"([A-Z][\w\.'-]*\.[A-Z][\w'-]+)\s+(?:pass|sacked|scrambles)")[0]
put("qb_parse_pct", 100 * gq.qb.notna().mean(), 0)
gq = gq[gq.qb.notna() & gq.soe.notna()].copy()
gq["soe_in"] = IN_YD * gq.soe
put("n_qbs", gq.qb.nunique()); put("n_qb_snaps", len(gq)); put("n_qb_rcv", gq.nfl_id.nunique())


def qb_env(df):
    vc_, ur_, uq_ = crossed_reml(df.soe_in.to_numpy(float), df.nfl_id.to_numpy(), df.qb.to_numpy())
    e_ = df.assign(qe=df.qb.map(uq_)).groupby(["nfl_id", "season"]).qe.mean() / IN_YD
    return vc_, e_


VC_QB, env = qb_env(gq)
qb_share = VC_QB[2] / VC_QB.sum()
put("qb_share_pct", 100 * qb_share, 1); put("rcv_share_pct", 100 * VC_QB[1] / VC_QB.sum(), 1)
put("qb_sd_in", np.sqrt(VC_QB[2]), 1); put("rcv_sd_in", np.sqrt(VC_QB[1]), 1)
# 90% interval on the QB variance by profile likelihood (a receiver-cluster bootstrap duplicates snaps within QBs and
# so manufactures QB variance in a crossed design); the lower end is the boundary 0 when the profile allows it
PROF = crossed_reml(gq.soe_in.to_numpy(float), gq.nfl_id.to_numpy(), gq.qb.to_numpy(), profile_b=True)[3]
ok_p = PROF[PROF.d2ll <= 2.706]
put("qb_sd_hi", np.sqrt(ok_p.tb.max()), 1); put("qb_share_hi", 100 * (ok_p.tb / ok_p.tot).max(), 1)
put("qb_share_lo", 100 * (ok_p.tb / ok_p.tot).min(), 1)
put("snap_sd_in", np.sqrt(VC_QB.sum()), 0)
gq_r = gq[gq.sample_kind == "random"]
VC_QBR, env_r = qb_env(gq_r)
put("qb_share_rand_pct", 100 * VC_QBR[2] / VC_QBR.sum(), 1); put("n_qb_snaps_rand", len(gq_r))
ss = ss.join(env.rename("qbenv"), on=["nfl_id", "season"]).join(env_r.rename("qbenv_r"), on=["nfl_id", "season"])
ss["qbenv"] = ss.qbenv.fillna(0.0)
ss["qbenv_r"] = ss.qbenv_r.fillna(0.0)
ss["one"] = 1.0
ss["ssoe_qb"] = ss.ssoe - ss.qbenv
ss["ssoe_qbr"] = ss.ssoe - ss.qbenv_r
put("qbenv_sd_in", IN_YD * np.sqrt(np.cov(ss.qbenv, aweights=ss.routes)), 2)
for col_ in ("ssoe_qb", "ssoe_qbr"):
    aggq = ss.groupby("nfl_id").apply(lambda d: np.average(d[col_], weights=d.routes), include_groups=False)
    D[col_] = aggq.reindex(D.index)
put("rho_soe_soeqb", spearmanr(D.ssoe, D.ssoe_qb)[0])
YQ_ = D.ssoe_qb.to_numpy(float)
YQR_ = D.ssoe_qbr.to_numpy(float)


tick('qb model done')
# ---- (b) forecast variants. ridge: standardized baseline, penalty picked by leave-one-class-out inside the training
# classes; "cov": the held-out forecast enters as a covariate (its calibration slope is estimated, not assumed = 1)
LAMS = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def ridge_fit(A, yv, w, lam):
    w = w / w.mean()  # penalty on the scale of one average player
    mu = np.average(A[:, 1:], axis=0, weights=w)
    sd = A[:, 1:].std(0) + 1e-9
    Z = np.column_stack([np.ones(len(A)), (A[:, 1:] - mu) / sd])
    s_ = np.sqrt(w)
    pen = lam * np.diag([0.0] + [1.0] * (Z.shape[1] - 1)) + 1e-8 * np.eye(Z.shape[1])  # jitter: constant columns
    b_ = np.linalg.solve((Z * w[:, None]).T @ Z + pen, (Z * w[:, None]).T @ yv)
    return lambda Anew: np.column_stack([np.ones(len(Anew)), (Anew[:, 1:] - mu) / sd]) @ b_


def ridge_cv(A, yv, w, ci):
    best, bl = np.inf, 0.0
    for lam in LAMS:
        err = 0.0
        for c in np.unique(ci):
            t = ci == c
            f = ridge_fit(A[~t], yv[~t], w[~t], lam)
            err += (w[t] * (yv[t] - f(A[t])) ** 2).sum()
        if err < best:
            best, bl = err, lam
    return ridge_fit(A, yv, w, bl), bl


def ho_general(ix, bgv, Y=None, mode="offset", fitter="ols", XB=None, W=None):
    Y = Y_ if Y is None else Y
    XB = XB_ if XB is None else XB
    W = WT_ if W is None else W
    Xi, yi, wi, ci, bi = XB[ix], Y[ix], W[ix], cls_arr0[ix], bgv[ix]
    fc = np.empty(len(ix))
    for c in CLASSES:
        t = ci == c
        if fitter == "ols":
            fc[t] = Xi[t] @ _wls(Xi[~t], yi[~t], wi[~t])
        else:
            fc[t] = ridge_cv(Xi[~t], yi[~t], wi[~t], ci[~t])[0](Xi[t])
    dmy = [(ci == c).astype(float) for c in CLASSES]
    if mode == "offset":
        return _wls(np.column_stack(dmy + [bi]), yi - fc, wi)[-1]
    return _wls(np.column_stack(dmy + [fc, bi]), yi, wi)[-1]


assert abs(IN_YD * ho_general(np.arange(len(D)), BGV) - ho) < 1e-9
LAM_SEL = []
for c in CLASSES:
    t = cls_arr0 == c
    f_, lam_ = ridge_cv(XB_[~t], Y_[~t], WT_[~t], cls_arr0[~t])
    LAM_SEL.append(lam_)
    put(f"calib_ridge{c}", _wls(np.column_stack([np.ones(t.sum()), f_(XB_[t])]), Y_[t], WT_[t])[1])
put("ridge_lam_med", float(np.median(LAM_SEL)), 0)

ROB = {}


def rob(key, lab, fn, nb=NBOOT):
    est = IN_YD * fn(np.arange(len(D)))
    bs_ = np.array([IN_YD * fn(strat_ix()) for _ in range(nb)])
    ROB[key] = dict(lab=lab, b=est, lo=np.percentile(bs_, 5), hi=np.percentile(bs_, 95))
    put(f"rob_{key}_in", est, 1); put(f"rob_{key}_lo", np.percentile(bs_, 5), 1)
    put(f"rob_{key}_hi", np.percentile(bs_, 95), 1)


rob("qb", "QB-adjusted season SOE", lambda ix: ho_general(ix, BGV, Y=YQ_))
rob("qbr", "QB-adjusted, random snaps only", lambda ix: ho_general(ix, BGV, Y=YQR_))
rob("ridge", "Ridge (shrunken) forecast", lambda ix: ho_general(ix, BGV, fitter="ridge"), nb=1000)
rob("cov", "Forecast as covariate", lambda ix: ho_general(ix, BGV, mode="cov"))
rob("qbridge", "QB-adjusted + ridge", lambda ix: ho_general(ix, BGV, Y=YQ_, fitter="ridge"), nb=1000)
# leave-one-player-out (forecast refit every time): range and the most influential player
LOPO = np.array([IN_YD * ho_general(np.delete(np.arange(len(D)), i), BGV) for i in range(len(D))])
n_ = len(D)
jse = np.sqrt((n_ - 1) / n_ * ((LOPO - LOPO.mean()) ** 2).sum())
put("lopo_min", LOPO.min(), 1); put("lopo_max", LOPO.max(), 1)
put("jack_lo", ho - 1.645 * jse, 1); put("jack_hi", ho + 1.645 * jse, 1)
NUM["lopo_infl_name"] = D.display_name.iloc[int(np.argmax(np.abs(LOPO - ho)))]
# per-class held-out slope with the QB-adjusted outcome
for c in CLASSES:
    ix = np.where(cls_arr0 == c)[0]
    fq_ = XB_[ix] @ _wls(XB_[cls_arr0 != c], YQ_[cls_arr0 != c], WT_[cls_arr0 != c])
    put(f"ho_qb{c}_in", IN_YD * _wls(np.column_stack([np.ones(len(ix)), BGV[ix]]), YQ_[ix] - fq_, WT_[ix])[-1], 1)

# ---- (c) the two secondary claims, held out the same way.
# price: log pick on Break G beyond the athletic profile (40, weight, NGS, TE), all WR/TE with a score, unweighted
XP = X.copy()
XP_B = np.column_stack([np.ones(len(XP)), XP[["forty_f", "combine_weight_f", "ngs_athleticism_score_f", "te"]]])
cls_save = cls_arr0
cls_arr0 = XP.draft_year.to_numpy()
YP_, WP_, BGP_ = XP.logpick.to_numpy(float), np.ones(len(XP)), XP.BG.to_numpy(float)
hp = ho_general(np.arange(len(XP)), BGP_, Y=YP_, XB=XP_B, W=WP_)
hpb = np.array([ho_general(strat_ix(), BGP_, Y=YP_, XB=XP_B, W=WP_) for _ in range(NBOOT)])
put("price_ho", -100 * (np.exp(hp) - 1), 0)
put("price_ho_lo", -100 * (np.exp(np.percentile(hpb, 95)) - 1), 0)
put("price_ho_hi", -100 * (np.exp(np.percentile(hpb, 5)) - 1), 0)
put("price_ho_n", len(XP))
put("price_ho_p", 2 * min((hpb >= 0).mean(), (hpb <= 0).mean()), 2)
PR_CLS = []
for c in CLASSES:
    ix = np.where(cls_arr0 == c)[0]
    fc_ = XP_B[ix] @ _wls(XP_B[cls_arr0 != c], YP_[cls_arr0 != c], WP_[cls_arr0 != c])
    v = _wls(np.column_stack([np.ones(len(ix)), BGP_[ix]]), YP_[ix] - fc_, WP_[ix])[-1]
    PR_CLS.append(-100 * (np.exp(v) - 1))
    put(f"price_ho{c}", PR_CLS[-1], 0)
put("price_ho_n_earlier", int(sum(v > 0 for v in PR_CLS)))
lp_ = np.array([ho_general(np.delete(np.arange(len(XP)), i), BGP_, Y=YP_, XB=XP_B, W=WP_) for i in range(len(XP))])
put("price_lopo_min", -100 * (np.exp(lp_.max()) - 1), 0); put("price_lopo_max", -100 * (np.exp(lp_.min()) - 1), 0)
put("price_ho_moved", EX_PICK - int(round(EX_PICK * np.exp(hp))))
cls_arr0 = cls_save
# targets per route run: held-out offset slope on the baseline forecast (routes-weighted), with and without Nacua
YT_ = D.tprr.to_numpy(float)
ht = ho_general(np.arange(len(D)), BGV, Y=YT_)
htb = np.array([ho_general(strat_ix(), BGV, Y=YT_) for _ in range(NBOOT)])
put("tprr_ho_pts", 100 * ht, 1); put("tprr_ho_lo", 100 * np.percentile(htb, 5), 1)
put("tprr_ho_hi", 100 * np.percentile(htb, 95), 1)
LT = np.array([100 * ho_general(np.delete(np.arange(len(D)), i), BGV, Y=YT_) for i in range(len(D))])
NUM["tprr_infl_name"] = D.display_name.iloc[int(np.argmin(LT))]
put("tprr_ho_nonacua", LT.min(), 1)
nac = np.where(D.display_name.to_numpy() == "Puka Nacua")[0]
keep_ = np.delete(np.arange(len(D)), nac)
htn_b = []
for _ in range(NBOOT):
    sub = np.concatenate([RNG.choice(keep_[cls_arr0[keep_] == c], int((cls_arr0[keep_] == c).sum())) for c in CLASSES])
    htn_b.append(100 * ho_general(sub, BGV, Y=YT_))
put("tprr_ho_nn_lo", np.percentile(htn_b, 5), 1); put("tprr_ho_nn_hi", np.percentile(htn_b, 95), 1)
put("tprr_ho_nn_p", 2 * min((np.array(htn_b) >= 0).mean(), (np.array(htn_b) <= 0).mean()), 2)
# within-class shuffle p for the held-out targets/route slope (with Nacua)
nt_ = []
for _ in range(NPERM // 2):
    pv_ = BGV.copy()
    for c in CLASSES:
        ix = np.where(cls_arr0 == c)[0]
        pv_[ix] = BGV[RNG.permutation(ix)]
    nt_.append(ho_general(np.arange(len(D)), pv_, Y=YT_))
put("tprr_ho_shuffle_p", (1 + (np.abs(nt_) >= abs(ht)).sum()) / (1 + len(nt_)), 2)

tick('robustness done')
# ============================================================================= revised round 12 additions
# ---- (e) primary test without Puka Nacua (the most influential player), same refit bootstrap
ho_nn = IN_YD * ho_general(keep_, BGV)
hnb = []
for _ in range(NBOOT):
    sub = np.concatenate([RNG.choice(keep_[cls_arr0[keep_] == c], int((cls_arr0[keep_] == c).sum())) for c in CLASSES])
    hnb.append(IN_YD * ho_general(sub, BGV))
put("ho_nn_in", ho_nn, 1); put("ho_nn_lo", np.percentile(hnb, 5), 1); put("ho_nn_hi", np.percentile(hnb, 95), 1)
put("n_nn", len(keep_))

# ---- (f) every draft slot checked against the nflverse draft record (UDFA = not in the record for his class)
dpk = pd.read_csv(ROOT / "data" / "external" / "nflverse" / "draft_picks.csv", low_memory=False,
                  usecols=["season", "pick", "pfr_player_id", "gsis_id"])
pid_map = players.set_index("nfl_id")[["pfr_id", "gsis_id", "draft_year"]]
chk = pid_map.loc[D.index].copy()
chk["ours"] = D.draft_overall_pick.reindex(chk.index)
by_pfr = dpk.dropna(subset=["pfr_player_id"]).set_index(["pfr_player_id", "season"]).pick
chk["record"] = [by_pfr.get((r_.pfr_id, r_.draft_year), np.nan) for r_ in chk.itertuples()]
mism = ((chk.ours.fillna(-1) != chk.record.fillna(-1))).sum()
put("draft_check_n", len(chk)); put("draft_check_mismatch", int(mism))
assert mism == 0, chk[chk.ours.fillna(-1) != chk.record.fillna(-1)]

# ---- (g) draft price as a held-out pick forecast. No-tracking board = 40, weight, NGS athleticism, TE, fitted on two
# classes; scored on the third with and without Break G. Gain = share of held-out squared log-pick error removed.
# p values: within-class shuffles of Break G. Multiplicity: Holm over the three price tests run (in-sample full
# profile, held-out slope, held-out forecast gain) and Bonferroni over every test in the ledger.
XPc = XP.draft_year.to_numpy()
XP_B1 = np.column_stack([XP_B, BGP_])


def pick_gain(ix, bgv):
    A0, yv, ci = XP_B[ix], YP_[ix], XPc[ix]
    A1 = np.column_stack([A0, bgv[ix]])
    e0 = e1 = tot = 0.0
    per = {}
    for c in CLASSES:
        t = ci == c
        r0 = yv[t] - A0[t] @ np.linalg.lstsq(A0[~t], yv[~t], rcond=None)[0]
        r1 = yv[t] - A1[t] @ np.linalg.lstsq(A1[~t], yv[~t], rcond=None)[0]
        tt = ((yv[t] - yv[~t].mean()) ** 2).sum()
        e0, e1, tot = e0 + (r0 ** 2).sum(), e1 + (r1 ** 2).sum(), tot + tt
        per[c] = (1 - (r0 ** 2).sum() / tt, 1 - (r1 ** 2).sum() / tt)
    return 1 - e0 / tot, 1 - e1 / tot, per


ALLP = np.arange(len(XP))
r2_0, r2_1, PER = pick_gain(ALLP, BGP_)
put("pick_r2_base", 100 * r2_0, 1); put("pick_r2_bg", 100 * r2_1, 1); put("pick_r2_gain", 100 * (r2_1 - r2_0), 1)
for c in CLASSES:
    put(f"pick_r2_gain{c}", 100 * (PER[c][1] - PER[c][0]), 1)
put("pick_gain_n_pos", int(sum(PER[c][1] > PER[c][0] for c in CLASSES)))


def strat_xp():
    return np.concatenate([RNG.choice(np.where(XPc == c)[0], int((XPc == c).sum())) for c in CLASSES])


gb_ = []
for _ in range(NBOOT):
    ix = strat_xp()
    a0_, a1_, _ = pick_gain(ix, BGP_)
    gb_.append(100 * (a1_ - a0_))
put("pick_r2_gain_lo", np.percentile(gb_, 5), 1); put("pick_r2_gain_hi", np.percentile(gb_, 95), 1)


def shuffle_xp(v):
    o = v.copy()
    for c in CLASSES:
        ix = np.where(XPc == c)[0]
        o[ix] = v[RNG.permutation(ix)]
    return o


cls_save = cls_arr0
cls_arr0 = XPc
NULL_G, NULL_S = [], []
for _ in range(NPERM):
    pv_ = shuffle_xp(BGP_)
    a0_, a1_, _ = pick_gain(ALLP, pv_)
    NULL_G.append(a1_ - a0_)
    NULL_S.append(ho_general(ALLP, pv_, Y=YP_, XB=XP_B, W=WP_))
cls_arr0 = cls_save
p_gain = (1 + (np.array(NULL_G) >= r2_1 - r2_0).sum()) / (1 + NPERM)
p_slope = (1 + (np.abs(NULL_S) >= abs(hp)).sum()) / (1 + NPERM)
put("pick_gain_p", p_gain, 3); put("price_ho_perm_p", p_slope, 3)
praw = np.array([float(NUM["price_bg_full_p"]), p_slope, p_gain])
order = np.argsort(praw)
holm = np.empty(3)
run = 0.0
for k_, i_ in enumerate(order):
    run = max(run, min(1.0, (3 - k_) * praw[i_]))
    holm[i_] = run
put("price_holm_slope", holm[1], 3); put("price_holm_gain", holm[2], 3); put("price_holm_full", holm[0], 2)

# ---- (h) price versus value for every Combine measure, both held out. Price = % earlier pick per sd (log pick on the
# measure + TE, fitted on two classes, offset slope on the third); value = season SOE inches per sd beyond the same
# held-out log pick + TE forecast. Class-stratified bootstrap, models refitted in every resample.


def ho_any(Ab, yv, wv, cl, mv, ix):
    Ab, yv, wv, cl, mv = Ab[ix], yv[ix], wv[ix], cl[ix], mv[ix]
    res = np.empty(len(ix))
    for c in CLASSES:
        t = cl == c
        res[t] = yv[t] - Ab[t] @ _wls(Ab[~t], yv[~t], wv[~t])
    A = np.column_stack([(cl == c).astype(float) for c in CLASSES] + [mv])
    return _wls(A, res, wv)[-1]


PV = []
for xc, lab, sgn in REFX:
    xx = X[X[xc].notna()]
    mv = (sgn * xx[xc] / xx[xc].std()).to_numpy(float)
    Ab = np.column_stack([np.ones(len(xx)), xx.te])
    cl = xx.draft_year.to_numpy()
    yv, wv = xx.logpick.to_numpy(float), np.ones(len(xx))
    sx = [np.concatenate([RNG.choice(np.where(cl == c)[0], int((cl == c).sum())) for c in CLASSES])
          for _ in range(NBOOT // 2)]
    pr0 = ho_any(Ab, yv, wv, cl, mv, np.arange(len(xx)))
    prb = np.array([ho_any(Ab, yv, wv, cl, mv, ix) for ix in sx])
    dd = D[D[xc].notna()]
    mvd = (sgn * dd[xc] / xx[xc].std()).to_numpy(float)
    Ad = np.column_stack([np.ones(len(dd)), dd.logpick, dd.te])
    cld, yd, wd = dd.draft_year.to_numpy(), dd.ssoe.to_numpy(float), dd.routes.to_numpy(float)
    sd_ = [np.concatenate([RNG.choice(np.where(cld == c)[0], int((cld == c).sum())) for c in CLASSES])
           for _ in range(NBOOT // 2)]
    va0 = IN_YD * ho_any(Ad, yd, wd, cld, mvd, np.arange(len(dd)))
    vab = np.array([IN_YD * ho_any(Ad, yd, wd, cld, mvd, ix) for ix in sd_])
    k = (xc[:-2] if xc.endswith("_s") else xc).lower()
    row = dict(lab=lab, k=k, price=-100 * (np.exp(pr0) - 1), plo=-100 * (np.exp(np.percentile(prb, 95)) - 1),
               phi=-100 * (np.exp(np.percentile(prb, 5)) - 1), val=va0, vlo=np.percentile(vab, 5),
               vhi=np.percentile(vab, 95), n_p=len(xx), n_v=len(dd))
    PV.append(row)
    for f_ in ("price", "plo", "phi"):
        put(f"pv_{k}_{f_}", row[f_], 0)
    for f_ in ("val", "vlo", "vhi"):
        put(f"pv_{k}_{f_}", row[f_], 1)
    put(f"pv_{k}_np", len(xx)); put(f"pv_{k}_nv", len(dd))
PV = pd.DataFrame(PV)

tick('price done')
# ---- (i) are the decision-card splits real? For the qualifying receivers (5+ route drills, 200+ NFL routes, WR), split
# each player's sampled snaps at random into two halves 200 times; correlate the man-minus-zone (and slot-minus-wide,
# and overall level) of snap SOE between halves across players; Spearman-Brown to the full 40. Placebo: half B taken
# from a different receiver (leave-one-player-out pairing). Interval: players resampled inside each split.
cand = D[(D.n_drills >= 5) & (D.routes >= 200) & (D.te == 0)].copy()
GS = gb[gb.nfl_id.isin(cand.index) & gb.soe.notna()].copy()
GS["soe_in"] = IN_YD * GS.soe
GS["wide"] = (GS.slotp == 0) & (GS.inline == 0)
GS["slot"] = GS.slotp == 1
put("sh_n_players", GS.nfl_id.nunique()); put("sh_n_snaps", len(GS))
SH_ITERS = 200


def half_stats(df):
    out = {}
    g_ = df.groupby("nfl_id")
    out["level"] = g_.soe_in.mean()
    for nm_, a_c, b_c in (("manzone", "man", None), ("slotwide", "slot", "wide")):
        A_ = df[df[a_c]].groupby("nfl_id").soe_in.agg(["mean", "size"])
        B_ = (df[~df[a_c]] if b_c is None else df[df[b_c]]).groupby("nfl_id").soe_in.agg(["mean", "size"])
        j_ = A_.join(B_, lsuffix="_a", rsuffix="_b", how="inner")
        j_ = j_[(j_.size_a >= 3) & (j_.size_b >= 3)]
        out[nm_] = j_.mean_a - j_.mean_b
    return out


SH = {k: [] for k in ("level", "manzone", "slotwide")}
SHP = {k: [] for k in SH}
pids_ = np.array(sorted(GS.nfl_id.unique()))
for it in range(SH_ITERS):
    hv = RNG.random(len(GS))
    rk = pd.Series(hv, index=GS.index).groupby(GS.nfl_id).rank()
    nn_ = GS.groupby("nfl_id").nfl_id.transform("size")
    ha = half_stats(GS[rk <= nn_ / 2])
    hb = half_stats(GS[rk > nn_ / 2])
    bs_p = RNG.choice(pids_, len(pids_))
    der = np.roll(RNG.permutation(pids_), 1)
    pmap = dict(zip(RNG.permutation(pids_), der))  # every player paired with a different one
    for k in SH:
        j_ = pd.concat([ha[k].rename("a"), hb[k].rename("b")], axis=1).dropna()
        jb = j_.reindex(bs_p).dropna()
        r_ = np.corrcoef(jb.a, jb.b)[0, 1]
        SH[k].append(dict(r=np.corrcoef(j_.a, j_.b)[0, 1], rb=r_, n=len(j_)))
        bpl = hb[k].reindex([pmap[p_] for p_ in ha[k].index])
        ok_ = bpl.notna().to_numpy()
        SHP[k].append(np.corrcoef(ha[k].to_numpy()[ok_], bpl.to_numpy()[ok_])[0, 1])
SB = {}
for k in SH:
    d_ = pd.DataFrame(SH[k])
    r_ = d_.r.mean()
    sb = 2 * r_ / (1 + r_) if r_ > 0 else r_
    rb = d_.rb.to_numpy()
    sbb = np.where(rb > 0, 2 * rb / (1 + rb), rb)
    SB[k] = max(sb, 0.0)
    put(f"sh_{k}_r", r_); put(f"sh_{k}_full", sb); put(f"sh_{k}_lo", np.percentile(sbb, 5))
    put(f"sh_{k}_hi", np.percentile(sbb, 95)); put(f"sh_{k}_n", int(d_.n.median()))
    put(f"sh_{k}_placebo_hi", np.percentile(SHP[k], 95)); put(f"sh_{k}_placebo_med", np.median(SHP[k]))
# full-sample raw splits for the pool and their reliability-shrunk versions (shrunk toward the pool mean)
FULL = half_stats(GS)
for k in ("manzone", "slotwide"):
    put(f"pool_{k}_sd", FULL[k].std(), 0)

# ----------------------------------------------------------------------------- named players + case study
cnt = q.groupby("nfl_id").size()
cand = D[(D.n_drills >= 5) & (D.routes >= 200) & (D.te == 0)].copy()
cand["fc_res_in"] = IN_YD * (cand.ssoe - cand.fc0)
cand["ngame"] = cnt.reindex(cand.index).fillna(0).astype(int)
cand["g_bg"] = pg.reindex(cand.index)
top = cand.sort_values("BG", ascending=False).head(5)
bot = cand.sort_values("BG").head(5)
board = pd.concat([top, bot.iloc[::-1]])  # ranked board: five hardest and five softest drill cutters, by Break G
SIX = [0, 1, 2, 7, 8, 9]  # the three highest and three lowest, drawn in Figure 2
put("n_card_pool", len(cand))
put("card_hi_beat", int((top.fc_res_in > 0).sum())); put("card_lo_beat", int((bot.fc_res_in > 0).sum()))
put("pool_beat_pct", 100 * (cand.fc_res_in > 0).mean(), 0)
CARD = []
for i_, (pid, r_) in enumerate(board.iterrows()):
    pick = "UDFA" if pd.isna(r_.draft_overall_pick) else str(int(r_.draft_overall_pick))
    CARD.append(dict(name=r_.display_name, year=int(r_.draft_year), pick=pick, drills=int(r_.n_drills),
                     bg=round(float(r_.BG), 1), lo=round(float(r_.BG_lo), 1), hi=round(float(r_.BG_hi), 1),
                     g_bg=None if pd.isna(r_.g_bg) else round(float(r_.g_bg), 1), routes=int(r_.routes),
                     res_in=round(float(r_.fc_res_in), 1), forty=r_.forty))
    put(f"card{i_}_bg", r_.BG, 1); put(f"card{i_}_lo", r_.BG_lo, 1); put(f"card{i_}_hi", r_.BG_hi, 1)
    put(f"card{i_}_res_in", r_.fc_res_in, 1); put(f"card{i_}_routes", int(r_.routes))
    if not pd.isna(r_.forty):
        put(f"card{i_}_forty", r_.forty)
    if not pd.isna(r_.g_bg):
        put(f"card{i_}_gbg", r_.g_bg, 1)
# per-player decision-card context (sampled NFL snaps): alignment, route mix, cut timing, speed kept, interval on
# SOE vs forecast (his sampled targeted-snap SOE spread over his season targets), and the cost of paying for Break G
for i_, (pid, r_) in enumerate(board.iterrows()):
    gpp = gp[gp.nfl_id == pid]
    gbt = gb[(gb.nfl_id == pid) & gb.target.astype(bool)].soe.dropna()
    qq_ = q[q.nfl_id == pid]
    se_s = IN_YD * gbt.std(ddof=1) / np.sqrt(max(r_.tg, 1)) if len(gbt) >= 3 else np.nan
    # add forecast error: refit the no-tracking forecast on bootstrap resamples of the other two classes
    oth = np.where(cls_arr0 != r_.draft_year)[0]
    xi = np.concatenate([[1.0], r_[BASE].to_numpy(float)])
    fcs = [xi @ _wls(XB_[ix_], Y_[ix_], WT_[ix_]) for ix_ in (RNG.choice(oth, len(oth)) for _ in range(500))]
    se_i = float(np.sqrt(se_s ** 2 + (IN_YD * np.std(fcs, ddof=1)) ** 2))
    pick_ = 262 if pd.isna(r_.draft_overall_pick) else int(r_.draft_overall_pick)
    pif = min(262, int(round(pick_ * np.exp(bpf[-1] * r_.BG))))  # 262 = undrafted either way
    gpc = gpp.cushion.dropna()
    CARD[i_].update(tight_pct=round(100 * (gpc <= 3).mean()) if len(gpc) else None,
                    man_pct=round(100 * gpp.team_coverage_type.fillna("").str.contains("MAN").mean()),
                    n_snaps=len(gpp), decel=round(float(r_.DECEL_S), 1), vap=round(float(r_.V_APEX_S), 1),
                    reacc=round(float(r_.REACCEL_S), 1),
                    max_catch=round(abs(float(r_.BG)) * float(NUM["ho_hi"]) * catch_per_in, 2))
    put(f"card{i_}_tight", CARD[i_]["tight_pct"]); put(f"card{i_}_man", CARD[i_]["man_pct"])
    put(f"card{i_}_maxcatch", CARD[i_]["max_catch"], 2); put(f"card{i_}_moved", abs(pick_ - pif))
    CARD[i_].update(slot_pct=round(100 * (gpp.lined_up_position == "SLOT_WR").mean()),
                    go_pct=round(100 * (gpp.route_ran == "GO").mean()), n_breaks=len(qq_),
                    lead=None if not len(qq_) else round(float(qq_.lead.median()), 1),
                    kept=None if not len(qq_) else round(float(100 * (qq_.v_apex / qq_.v_in).median())),
                    targets=int(r_.tg), res_lo=round(float(r_.fc_res_in - 1.645 * se_i), 1),
                    res_hi=round(float(r_.fc_res_in + 1.645 * se_i), 1), pick_if_paid=pif, pick=pick_)
    put(f"card{i_}_res_lo", r_.fc_res_in - 1.645 * se_i, 1); put(f"card{i_}_res_hi", r_.fc_res_in + 1.645 * se_i, 1)
    put(f"card{i_}_pif", pif); put(f"card{i_}_slot", CARD[i_]["slot_pct"]); put(f"card{i_}_go", CARD[i_]["go_pct"])
    if CARD[i_]["lead"] is not None:
        put(f"card{i_}_lead", CARD[i_]["lead"], 1)
    # scheme splits on his sampled snaps (context-adjusted snap SOE, inches): man vs zone, slot vs outside, route
    gpl = gb[(gb.nfl_id == pid) & gb.soe.notna()]
    spl = {}
    for nm_, msk in (("man", gpl.man), ("zone", ~gpl.man), ("slot", gpl.slotp == 1),
                     ("wide", (gpl.slotp == 0) & (gpl.inline == 0))):
        spl[nm_] = (IN_YD * gpl[msk].soe.mean() if msk.sum() else np.nan, int(msk.sum()))
        put(f"card{i_}_soe_{nm_}", spl[nm_][0], 0); put(f"card{i_}_n_{nm_}", spl[nm_][1])
    rt_ = gpl.groupby("route").soe.agg(["mean", "size"])
    rt_ = rt_[rt_["size"] >= 3]
    if len(rt_):
        bst, wst = rt_["mean"].idxmax(), rt_["mean"].idxmin()
        CARD[i_].update(best_route=bst.lower(), worst_route=wst.lower())
        NUM[f"card{i_}_best_route"] = bst.lower(); NUM[f"card{i_}_worst_route"] = wst.lower()
        put(f"card{i_}_best_soe", IN_YD * rt_.loc[bst, "mean"], 0); put(f"card{i_}_best_n", int(rt_.loc[bst, "size"]))
        put(f"card{i_}_worst_soe", IN_YD * rt_.loc[wst, "mean"], 0); put(f"card{i_}_worst_n", int(rt_.loc[wst, "size"]))
    CARD[i_].update({f"soe_{k}": round(float(v[0]), 0) if not np.isnan(v[0]) else None for k, v in spl.items()})
    CARD[i_].update({f"n_{k}": v[1] for k, v in spl.items()})
# reliability-shrunk scheme splits (toward the pool mean) and usage, for the board
for i_, (pid, r_) in enumerate(board.iterrows()):
    for k in ("manzone", "slotwide"):
        raw_ = FULL[k].get(pid, np.nan)
        shr_ = FULL[k].mean() + SB[k] * (raw_ - FULL[k].mean()) if not np.isnan(raw_) else np.nan
        CARD[i_][f"{k}_raw"] = None if np.isnan(raw_) else round(float(raw_))
        CARD[i_][f"{k}_shr"] = None if np.isnan(shr_) else round(float(shr_), 1)
        if not np.isnan(raw_):
            put(f"card{i_}_{k}_raw", raw_, 0); put(f"card{i_}_{k}_shr", shr_, 1)
    gpp = gp[gp.nfl_id == pid]
    CARD[i_]["wide_pct"] = round(100 * ((gpp.lined_up_position != "SLOT_WR") & ~gpp.lined_up_position.isin(
        ["TE", "FB", "RB"])).mean())
    put(f"card{i_}_wide", CARD[i_]["wide_pct"])
put("card_max_abs_shr", max(abs(c_[f"{k}_shr"]) for c_ in CARD for k in ("manzone", "slotwide")
                            if c_[f"{k}_shr"] is not None), 1)
# the discount case, by rule: soft drill cut (Break G < 0), day-3 or undrafted, best season SOE beyond forecast
cand["day3"] = players.set_index("nfl_id").draft_round.reindex(cand.index).fillna(8).ge(4).to_numpy()
dsc = cand[(cand.BG < 0) & cand.day3].sort_values("fc_res_in").iloc[-1]
NUM["disc_name"] = dsc.display_name
put("disc_pick", int(dsc.draft_overall_pick)) if not pd.isna(dsc.draft_overall_pick) else None
put("disc_year", int(dsc.draft_year)); put("disc_bg", dsc.BG, 1); put("disc_res_in", dsc.fc_res_in, 1)
put("disc_routes", int(dsc.routes)); put("disc_sep", dsc.sep)
put("disc_n_soft_day3", int(((cand.BG < 0) & cand.day3).sum()))
put("disc_n_soft_day3_beat", int(((cand.BG < 0) & cand.day3 & (cand.fc_res_in > 0)).sum()))
put("pool_go_pct", 100 * (gp[gp.nfl_id.isin(cand.index)].route_ran == "GO").mean(), 0)
put("pool_slot_pct", 100 * (gp[gp.nfl_id.isin(cand.index)].lined_up_position == "SLOT_WR").mean(), 0)
put("pool_tight_pct", 100 * (gp[gp.nfl_id.isin(cand.index)].cushion.dropna() <= 3).mean(), 0)
put("pool_man_pct", 100 * gp[gp.nfl_id.isin(cand.index)].team_coverage_type.fillna("").str.contains("MAN").mean(), 0)
put("n_card_inside", int(sum((c_["res_lo"] < 0 < c_["res_hi"]) for c_ in CARD)))
pd.DataFrame(CARD).to_csv(OUT / "board.csv", index=False)
D.assign(ssoe_in=IN_YD * D.ssoe, fc0_in=IN_YD * D.fc0)[
    ["display_name", "draft_year", "draft_overall_pick", "position_group", "n_drills", "BG", "BG_lo", "BG_hi",
     "routes", "sep", "ssoe_in", "fc0_in"]].sort_values("BG", ascending=False).to_csv(OUT / "board_full.csv")
# false-positive case study: the hardest drill cutter in the pool, drill vs NFL break mechanics vs league medians
fp = top.index[0]
NUM["fp_name"] = D.loc[fp, "display_name"]
dfp, gfp = cb[cb.nfl_id == fp], q[q.nfl_id == fp]
put("fp_n_reps", len(dfp)); put("fp_n_snaps", len(gfp))
put("fp_drill_g", dfp.bg.median()); put("fp_game_g", gfp.bg.median())
put("fp_drill_vin", dfp.v_in.median(), 1); put("lg_drill_vin", cb.v_in.median(), 1)
put("fp_game_vin", gfp.v_in.median(), 1); put("lg_game_vin", q.v_in.median(), 1)
put("fp_game_kept", 100 * (gfp.v_apex / gfp.v_in).median(), 0); put("lg_game_kept", 100 * (q.v_apex / q.v_in).median(), 0)
put("fp_lead", gfp.lead.median(), 1); put("lg_lead", q.lead.median(), 1)
rr_ = gp[gp.nfl_id == fp].route_ran
put("fp_go_pct", 100 * (rr_ == "GO").mean(), 0)
put("fp_sep", D.loc[fp, "sep"])
put("fp_ssoe_in", IN_YD * D.loc[fp, "ssoe"], 1)
put("fp_pick", int(D.loc[fp, "draft_overall_pick"]))
# cost of the wrong call: where a board that priced Break G at the full-profile rate would have taken him
put("fp_pick_if_priced", int(round(D.loc[fp, "draft_overall_pick"] * np.exp(bpf[-1] * D.loc[fp, "BG"]))))

tick('board done')
# ============================================================================= round 13 additions
# ---- (j) PRIMARY (round 13): ridge no-tracking forecast (penalty by leave-one-class-out inside the two training
# classes), entered on the held-out class as a covariate, so its calibration is estimated, not assumed. Slope of season
# SOE on Break G with class intercepts, routes-weighted. Interval: class-stratified player bootstrap that refits the
# forecast and the penalty in every resample. Null: Break G shuffled within class (forecast fixed, it ignores Break G).
ALL = np.arange(len(D))
NAC = set(np.where(D.display_name.to_numpy() == "Puka Nacua")[0])


def strat_rows(rows):
    rows = np.asarray(rows)
    return np.concatenate([RNG.choice(rows[cls_arr0[rows] == c], int((cls_arr0[rows] == c).sum())) for c in CLASSES])


def ridge_fc(rows, Y, W):
    """Held-out ridge forecast for every row (fitted on the other classes among `rows`)."""
    fc = np.full(len(D), np.nan)
    for c in CLASSES:
        t = rows[cls_arr0[rows] == c]
        o = rows[cls_arr0[rows] != c]
        fc[t] = ridge_cv(XB_[o], Y[o], W[o], cls_arr0[o])[0](XB_[t])
    return fc


def suite(key, Y, W, rows, mult, nb=NBOOT, nperm=NPERM, nd=1):
    """Held-out covariate slope (ridge forecast) with interval, shuffle p and the same without Puka Nacua."""
    rows = np.asarray(rows)
    out = {}
    for tag, rr in (("", rows), ("_nn", np.array([r for r in rows if r not in NAC]))):
        est = mult * ho_general(rr, BGV, Y=Y, W=W, mode="cov", fitter="ridge")
        bs_ = np.array([mult * ho_general(strat_rows(rr), BGV, Y=Y, W=W, mode="cov", fitter="ridge")
                        for _ in range(nb)])
        fc = ridge_fc(rr, Y, W)
        dmy = np.column_stack([(cls_arr0[rr] == c).astype(float) for c in CLASSES])
        nul = []
        for _ in range(nperm):
            pv_ = BGV.copy()
            for c in CLASSES:
                ix = rr[cls_arr0[rr] == c]
                pv_[ix] = BGV[RNG.permutation(ix)]
            nul.append(mult * _wls(np.column_stack([dmy, fc[rr], pv_[rr]]), Y[rr], W[rr])[-1])
        p_ = (1 + (np.abs(nul) >= abs(est)).sum()) / (1 + nperm)
        out[tag] = dict(b=est, lo=np.percentile(bs_, 5), hi=np.percentile(bs_, 95), p=p_, n=len(rr), se=bs_.std(ddof=1))
        put(f"{key}{tag}_b", est, nd); put(f"{key}{tag}_lo", np.percentile(bs_, 5), nd)
        put(f"{key}{tag}_hi", np.percentile(bs_, 95), nd); put(f"{key}{tag}_p", p_, 2); put(f"{key}{tag}_n", len(rr))
        put(f"{key}{tag}_pct", 100 * p_, 0)
    return out


P13 = suite("p13", Y_, WT_, ALL, IN_YD)
pr13 = P13[""]
put("p13_up_p", norm.cdf((pr13["b"] - SESOI_IN) / pr13["se"]), 4)
put("p13_catch_hi", pr13["hi"] * catch_per_in, 2)
# calibration of the ridge forecast entered as a covariate (pooled coefficient on the forecast) and per-class slopes
FC13 = ridge_fc(ALL, Y_, WT_)
dmy_all = np.column_stack([(cls_arr0 == c).astype(float) for c in CLASSES])
cal_ = _wls(np.column_stack([dmy_all, FC13, BGV]), Y_, WT_)
put("p13_calib", cal_[-2])
for c in CLASSES:
    t = cls_arr0 == c
    put(f"p13_{c}", IN_YD * _wls(np.column_stack([np.ones(t.sum()), FC13[t], BGV[t]]), Y_[t], WT_[t])[-1], 1)
put("p13_n_neg_cls", int(sum(float(NUM[f"p13_{c}"]) < 0 for c in CLASSES)))
# old primary (round 12: OLS forecast as an offset) kept as a version
put("r12_b", ho, 1)
tick("primary r13 done")

# ---- (k) route mix and depth at the throw (target quality). On the random half of the sampled snaps (no EPA
# selection), separation at the throw ~ route + depth + depth^2; each player-season's expected separation from his own
# route mix and depth is the mean prediction over his random snaps. It enters the season SOE adjustment.
gr = gb[(gb.sample_kind == "random") & gb.sep.notna() & gb.depth_thr.notna()].copy()
Am = np.column_stack([np.ones(len(gr)), gr.depth_thr, gr.depth_thr ** 2 / 100] +
                     [(gr.route == r).astype(float) for r in ROUTES[1:]])
bm_ = np.linalg.lstsq(Am, gr.sep.to_numpy(), rcond=None)[0]
gr["mixexp"] = Am @ bm_
put("mix_n_snaps", len(gr)); put("mix_r2_pct", 100 * (1 - np.var(gr.sep - gr.mixexp) / np.var(gr.sep)), 0)
mx_ps = gr.groupby(["nfl_id", "season"]).mixexp.mean()
ss = ss.join(mx_ps.rename("mixexp"), on=["nfl_id", "season"])
put("mix_cov_pct", 100 * np.average(ss.mixexp.notna(), weights=ss.routes), 0)
ss["mix_f"] = ss.mixexp.fillna(np.average(ss.mixexp.dropna(), weights=ss.routes[ss.mixexp.notna()]))
A_mx = np.column_stack([A_ss, ss.mix_f])
b_mx = np.linalg.lstsq(A_mx * w_ss[:, None], ss.avg_separation.to_numpy() * w_ss, rcond=None)[0]
ss["ssoe_mix"] = ss.avg_separation - A_mx @ b_mx
put("mix_coef", b_mx[-1])
aggm = ss.groupby("nfl_id").apply(lambda d: np.average(d.ssoe_mix, weights=d.routes), include_groups=False)
D["ssoe_mix"] = aggm.reindex(D.index)
YMIX_ = D.ssoe_mix.to_numpy(float)
put("rho_soe_mix", spearmanr(D.ssoe, D.ssoe_mix)[0])
PMIX = suite("pmix", YMIX_, WT_, ALL, IN_YD, nperm=NPERM // 4)
PQB = suite("pqb", YQ_, WT_, ALL, IN_YD, nperm=NPERM // 4)
tick("mix done")

# ---- (l) outcomes that do not hinge on separation: yards per route run (routes-weighted) and catch rate over
# expected (catches beyond a season model of catch rate on separation when targeted, slot, TE; targets-weighted)
cs = ssa[(ssa.targets > 0) & ssa.avg_separation_targeted.notna()].copy()
cs["cr"] = cs.receptions / cs.targets
A_cr = np.column_stack([np.ones(len(cs)), cs.avg_separation_targeted, (cs.main_alignment == "SLOT_WR").astype(float),
                        (cs.position_group == "TE").astype(float)])
w_cr = np.sqrt(cs.targets.to_numpy(float))
b_cr = np.linalg.lstsq(A_cr * w_cr[:, None], cs.cr.to_numpy() * w_cr, rcond=None)[0]
cs["xrec"] = cs.targets * (A_cr @ b_cr)
coe = cs.groupby("nfl_id").apply(lambda d: pd.Series(dict(coe=(d.receptions - d.xrec).sum() / d.targets.sum(),
                                                          tgt=d.targets.sum())), include_groups=False)
D["coe"] = coe.coe.reindex(D.index)
D["coe_w"] = coe.tgt.reindex(D.index).fillna(0.0)
YCOE_, WCOE_ = D.coe.fillna(0.0).to_numpy(float), D.coe_w.to_numpy(float)
ROWS_COE = np.where(D.coe.notna() & (D.coe_w >= 10))[0]
YPRR_ = D.yprr.to_numpy(float)
PYPRR = suite("pyprr", YPRR_, WT_, ALL, 1.0, nperm=NPERM // 4, nd=2)  # yards per route
PCOE = suite("pcoe", YCOE_, WCOE_, ROWS_COE, 100.0, nperm=NPERM // 4)  # percentage points of catch rate
put("yprr_mean", np.average(D.yprr, weights=D.routes))
put("coe_sd_pts", 100 * np.sqrt(np.cov(D.coe.iloc[ROWS_COE], aweights=WCOE_[ROWS_COE])), 1)
tick("outcomes done")

# ---- (m) the shortlist rule, scored on the next season. Pool: day-3 (round 4+) or undrafted WR/TE with 100+ routes
# in two consecutive seasons. Rule (fixed in advance, nothing fitted): season-1 SOE above zero. Hit: season-2 SOE above
# zero. Compared with the receivers the rule passes over and with the pool base rate. Interval: player-cluster
# bootstrap of the hit-rate gap; p: one-sided Fisher exact test. Break G is then asked to sort the shortlist.
from scipy.stats import fisher_exact  # noqa: E402

pl_r = players.set_index("nfl_id")[["draft_round", "display_name", "draft_year"]]
pvt = ss.pivot_table(index="nfl_id", columns="season", values=["ssoe", "routes"])
PAIRS = []
for a_, b_ in ((2023, 2024), (2024, 2025)):
    if ("ssoe", b_) not in pvt.columns:
        continue
    m_ = pvt[[("ssoe", a_), ("ssoe", b_), ("routes", a_), ("routes", b_)]].dropna()
    m_.columns = ["y1", "y2", "r1", "r2"]
    m_["s1"] = a_
    PAIRS.append(m_)
PAIRS = pd.concat(PAIRS).join(pl_r)
PAIRS["day3"] = PAIRS.draft_round.isna() | (PAIRS.draft_round >= 4)
PAIRS = PAIRS[(PAIRS.r1 >= 100) & (PAIRS.r2 >= 100)].copy()
PAIRS["BG"] = cplayer.BG.reindex(PAIRS.index)
PAIRS["short"] = PAIRS.y1 > 0
PAIRS["hit"] = PAIRS.y2 > 0
PAIRS["pid"] = PAIRS.index


def rates(df):
    s_, r_ = df[df.short], df[~df.short]
    return s_.hit.mean(), r_.hit.mean(), df.hit.mean()


for tag, sub in (("d3", PAIRS[PAIRS.day3]), ("all", PAIRS)):
    hs, hr, base = rates(sub)
    put(f"sl_{tag}_n", len(sub)); put(f"sl_{tag}_players", sub.pid.nunique())
    put(f"sl_{tag}_short_n", int(sub.short.sum())); put(f"sl_{tag}_short_hit", int(sub[sub.short].hit.sum()))
    put(f"sl_{tag}_rest_n", int((~sub.short).sum())); put(f"sl_{tag}_rest_hit", int(sub[~sub.short].hit.sum()))
    put(f"sl_{tag}_short_pct", 100 * hs, 0); put(f"sl_{tag}_rest_pct", 100 * hr, 0); put(f"sl_{tag}_base_pct", 100 * base, 0)
    tab = [[int(sub[sub.short].hit.sum()), int((sub.short & ~sub.hit).sum())],
           [int(sub[~sub.short].hit.sum()), int((~sub.short & ~sub.hit).sum())]]
    put(f"sl_{tag}_fisher_p", fisher_exact(tab, alternative="greater")[1], 3)
    grp_ = clusters(sub.pid)
    gaps, lifts = [], []
    for _ in range(NBOOT):
        bb_ = sub.iloc[np.concatenate([grp_[i] for i in RNG.integers(0, len(grp_), len(grp_))])]
        if bb_.short.all() or (~bb_.short).all():
            continue
        a1, a2, a3 = rates(bb_)
        gaps.append(a1 - a2); lifts.append(a1 - a3)
    put(f"sl_{tag}_gap", 100 * (hs - hr), 0); put(f"sl_{tag}_gap_lo", 100 * np.percentile(gaps, 5), 0)
    put(f"sl_{tag}_gap_hi", 100 * np.percentile(gaps, 95), 0)
    put(f"sl_{tag}_lift", 100 * (hs - base), 0); put(f"sl_{tag}_lift_lo", 100 * np.percentile(lifts, 5), 0)
    put(f"sl_{tag}_lift_hi", 100 * np.percentile(lifts, 95), 0)
SLD = PAIRS[PAIRS.day3]
SLS = SLD[SLD.short & SLD.BG.notna()]
put("sl_soft_n", int((SLS.BG < 0).sum())); put("sl_soft_hit", int(SLS[SLS.BG < 0].hit.sum()))
put("sl_hard_n", int((SLS.BG >= 0).sum())); put("sl_hard_hit", int(SLS[SLS.BG >= 0].hit.sum()))
put("sl_soft_pct", 100 * SLS[SLS.BG < 0].hit.mean(), 0); put("sl_hard_pct", 100 * SLS[SLS.BG >= 0].hit.mean(), 0)
# the latest-season pairs alone (2024 -> 2025): the rule was first written down before these were scored
lat = SLD[SLD.s1 == 2024]
put("sl_late_short_n", int(lat.short.sum())); put("sl_late_short_hit", int(lat[lat.short].hit.sum()))
put("sl_late_base_pct", 100 * lat.hit.mean(), 0)
# the 2026 shortlist: day-3 / undrafted WR/TE whose 2025 season qualifies (100+ routes, SOE above zero)
s25 = ss[(ss.season == 2025) & (ss.routes >= 100)].join(pl_r, on="nfl_id")
s25 = s25[(s25.draft_round.isna() | (s25.draft_round >= 4)) & (s25.ssoe > 0)].copy()
s25["BG"] = cplayer.BG.reindex(s25.nfl_id).to_numpy()
s25 = s25.sort_values("ssoe", ascending=False)
put("sl26_n", len(s25)); put("sl26_soft", int((s25.BG < 0).sum()))
SL26 = []
for i_, r_ in enumerate(s25.itertuples()):
    pk_ = players.set_index("nfl_id").draft_overall_pick.get(r_.nfl_id)
    SL26.append(dict(name=r_.display_name, year=int(r_.draft_year), pick="UDFA" if pd.isna(pk_) else int(pk_),
                     pos=r_.position_group, routes=int(r_.routes), soe_in=round(IN_YD * r_.ssoe, 1),
                     bg=None if pd.isna(r_.BG) else round(float(r_.BG), 1), align=r_.main_alignment))
    put(f"sl26_{i_}_soe", IN_YD * r_.ssoe, 1); put(f"sl26_{i_}_routes", int(r_.routes))
    if not pd.isna(r_.BG):
        put(f"sl26_{i_}_bg", r_.BG, 1)
pd.DataFrame(SL26).to_csv(OUT / "shortlist_2026.csv", index=False)
print(pd.DataFrame(SL26).to_string())
print(PAIRS[PAIRS.day3].sort_values("y1")[["display_name", "s1", "draft_round", "r1", "y1", "y2", "BG"]].to_string())
tick("shortlist done")

# ============================================================================= round 14 additions
# ---- (n) ONE primary estimate. Rounds 12 and 13 each used a different no-tracking forecast (plain least squares as an
# offset; ridge as a covariate) and the switch was made after the first was seen. Round 14 reports the average of the
# two, both refitted inside every class-stratified resample, as the single primary number, with the ridge forecast's
# held-out weight and its interval from the same resamples. Bayesian reading: normal likelihood (bootstrap se) with a
# sceptical normal prior centred at 0 whose sd is the +3 in margin; posterior probability of a gain of +3 in or more.


def combo(ix, Y=None):
    Y = Y_ if Y is None else Y
    Xi, yi, wi, ci, bi = XB_[ix], Y[ix], WT_[ix], cls_arr0[ix], BGV[ix]
    fo, fr = np.empty(len(ix)), np.empty(len(ix))
    for c in CLASSES:
        t = ci == c
        fo[t] = Xi[t] @ _wls(Xi[~t], yi[~t], wi[~t])
        fr[t] = ridge_cv(Xi[~t], yi[~t], wi[~t], ci[~t])[0](Xi[t])
    dmy = [(ci == c).astype(float) for c in CLASSES]
    b_plain = _wls(np.column_stack(dmy + [bi]), yi - fo, wi)[-1]
    cr_ = _wls(np.column_stack(dmy + [fr, bi]), yi, wi)
    return IN_YD * 0.5 * (b_plain + cr_[-1]), IN_YD * b_plain, IN_YD * cr_[-1], cr_[-2]


def strat_sub(rows):
    rows = np.asarray(rows)
    return np.concatenate([RNG.choice(rows[cls_arr0[rows] == c], int((cls_arr0[rows] == c).sum())) for c in CLASSES])


COMBO = {}
NN_ROWS = np.array([r for r in ALL if r not in NAC])
for key_, lab_, Yv, rows_ in (("main", "Primary", Y_, ALL), ("nn", "Without Puka Nacua", Y_, NN_ROWS),
                              ("mix", "Route mix and depth adjusted", YMIX_, ALL),
                              ("qb", "Quarterback adjusted", YQ_, ALL)):
    est_ = combo(rows_, Yv)
    bs_ = np.array([combo(strat_sub(rows_), Yv) for _ in range(NBOOT if key_ == "main" else NBOOT // 2)])
    COMBO[key_] = dict(lab=lab_, b=est_[0], lo=np.percentile(bs_[:, 0], 5), hi=np.percentile(bs_[:, 0], 95),
                       se=bs_[:, 0].std(ddof=1), n=len(rows_), bs=bs_, est=est_)
    put(f"c14_{key_}_b", est_[0], 1); put(f"c14_{key_}_lo", np.percentile(bs_[:, 0], 5), 1)
    put(f"c14_{key_}_hi", np.percentile(bs_[:, 0], 95), 1); put(f"c14_{key_}_n", len(rows_))
    if key_ == "main":
        put("c14_plain_b", est_[1], 1); put("c14_plain_lo", np.percentile(bs_[:, 1], 5), 1)
        put("c14_plain_hi", np.percentile(bs_[:, 1], 95), 1)
        put("c14_ridge_b", est_[2], 1); put("c14_ridge_lo", np.percentile(bs_[:, 2], 5), 1)
        put("c14_ridge_hi", np.percentile(bs_[:, 2], 95), 1)
        put("c14_wt", est_[3]); put("c14_wt_lo", np.percentile(bs_[:, 3], 5)); put("c14_wt_hi", np.percentile(bs_[:, 3], 95))
        put("c14_corr_pr", np.corrcoef(bs_[:, 1], bs_[:, 2])[0, 1])
CM = COMBO["main"]
# within-class shuffle null for the combined estimate (forecasts fixed: they ignore Break G)
fo_all = np.empty(len(D)); fr_all = ridge_fc(ALL, Y_, WT_)
for c in CLASSES:
    t = cls_arr0 == c
    fo_all[t] = XB_[t] @ _wls(XB_[~t], Y_[~t], WT_[~t])
nul_c = []
for _ in range(NPERM):
    pv_ = BGV.copy()
    for c in CLASSES:
        ix = np.where(cls_arr0 == c)[0]
        pv_[ix] = BGV[RNG.permutation(ix)]
    b1_ = _wls(np.column_stack([dmy_all, pv_]), Y_ - fo_all, WT_)[-1]
    b2_ = _wls(np.column_stack([dmy_all, fr_all, pv_]), Y_, WT_)[-1]
    nul_c.append(IN_YD * 0.5 * (b1_ + b2_))
put("c14_shuffle_p", (1 + (np.abs(nul_c) >= abs(CM["b"])).sum()) / (1 + NPERM), 2)
# equivalence (upper end) and Bayesian bound
put("c14_up_p", norm.cdf((CM["b"] - SESOI_IN) / CM["se"]), 4)
put("c14_se", CM["se"], 1)
pri_sd = SESOI_IN
post_var = 1 / (1 / pri_sd ** 2 + 1 / CM["se"] ** 2)
post_m = post_var * CM["b"] / CM["se"] ** 2
post_sd = np.sqrt(post_var)
POST = (post_m, post_sd)
put("bayes_m", post_m, 1); put("bayes_lo", post_m - 1.645 * post_sd, 1); put("bayes_hi", post_m + 1.645 * post_sd, 1)
put("bayes_p3_pct", 100 * norm.sf((SESOI_IN - post_m) / post_sd), 1)
put("bayes_p0_pct", 100 * norm.sf(-post_m / post_sd), 0)
put("bayes_prior_p3_pct", 100 * norm.sf(1.0), 0)
put("flat_p3_pct", 100 * norm.sf((SESOI_IN - CM["b"]) / CM["se"]), 1)
put("c14_catch_hi", CM["hi"] * catch_per_in, 2)
put("c14_catch_sesoi", SESOI_IN * catch_per_in, 2)
tick("r14 primary done")

# ---- (o) disattenuated drill-to-game correlation, players with 8+ NFL breaks on the five matched route families.
# Observed Pearson r between the Combine score and the NFL Break G (both z), divided by the square root of the product
# of the two split-half reliabilities (Spearman-Brown to full length), all recomputed inside a player bootstrap.
gcount = g.groupby("nfl_id").size()
ga_ = g[g.half == 0].groupby("nfl_id").bg_gz.mean()
gb_h = g[g.half == 1].groupby("nfl_id").bg_gz.mean()
P8 = pd.concat([X.BG.rename("d"), pg.rename("gm"), gcount.rename("ng"), HH_BG.a.rename("da"), HH_BG.b.rename("db"),
                ga_.rename("ga"), gb_h.rename("gb")], axis=1).dropna()
P8 = P8[P8.ng >= 8]


def sb_(r):
    return 2 * r / (1 + r)


def disatt(df):
    r_o = np.corrcoef(df.d, df.gm)[0, 1]
    rd = sb_(np.corrcoef(df.da, df.db)[0, 1])
    rg = sb_(np.corrcoef(df.ga, df.gb)[0, 1])
    return r_o, rd, rg, r_o / np.sqrt(max(rd, 0.05) * max(rg, 0.05))


d8 = disatt(P8)
bs8 = np.array([disatt(P8.iloc[RNG.integers(0, len(P8), len(P8))]) for _ in range(NBOOT)])
put("p8_n", len(P8)); put("p8_breaks_med", int(P8.ng.median()))
put("p8_r", d8[0]); put("p8_r_lo", np.percentile(bs8[:, 0], 5)); put("p8_r_hi", np.percentile(bs8[:, 0], 95))
put("p8_reld", d8[1]); put("p8_relg", d8[2])
put("p8_relg_lo", np.percentile(bs8[:, 2], 5)); put("p8_relg_hi", np.percentile(bs8[:, 2], 95))
put("p8_dis", d8[3]); put("p8_dis_lo", np.percentile(bs8[:, 3], 5)); put("p8_dis_hi", np.percentile(bs8[:, 3], 95))
put("p8_relg_neg_pct", 100 * (bs8[:, 2] <= 0.05).mean(), 0)
tick("disattenuation done")

# ---- (p) the shortlist rule against a no-tracking forecast and over time. For every season pair, a forecast of
# next-season SOE from log pick, TE and log routes in season 1 (opportunity) is fitted on the OTHER season pair only
# (time-forward for 2024->25, backward for 2023->24). The pick-and-opportunity rule shortlists the same number of
# receivers per season pair and pool, ranked by that forecast. Persistence net of it: slope of season-2 SOE on
# season-1 SOE with the forecast held fixed; receiver-clustered bootstrap.
pk_all = players.set_index("nfl_id").draft_overall_pick
PAIRS = PAIRS.reset_index(drop=True)
PAIRS["logpick"] = np.log(pk_all.reindex(PAIRS.pid).fillna(262).to_numpy(float))
PAIRS["te"] = players.set_index("nfl_id").position_group.reindex(PAIRS.pid).eq("TE").astype(float).to_numpy()
PAIRS["lr1"] = np.log(PAIRS.r1)
PAIRS["fc_pk"] = np.nan
for s_ in (2023, 2024):
    trn, tst = PAIRS.s1 != s_, PAIRS.s1 == s_
    A_t = np.column_stack([np.ones(trn.sum()), PAIRS.loc[trn, ["logpick", "te", "lr1"]]])
    b_t = np.linalg.lstsq(A_t, PAIRS.loc[trn, "y2"].to_numpy(), rcond=None)[0]
    PAIRS.loc[tst, "fc_pk"] = np.column_stack([np.ones(tst.sum()), PAIRS.loc[tst, ["logpick", "te", "lr1"]]]) @ b_t
PAIRS["short_pk"] = False
for (s_, d3_), sub_ in PAIRS.groupby(["s1", "day3"]):
    k_ = int(sub_.short.sum())
    PAIRS.loc[sub_.sort_values("fc_pk", ascending=False).index[:k_], "short_pk"] = True
for tag, sub in (("d3", PAIRS[PAIRS.day3]), ("all", PAIRS)):
    put(f"pk_{tag}_short_n", int(sub.short_pk.sum())); put(f"pk_{tag}_short_hit", int(sub[sub.short_pk].hit.sum()))
    put(f"pk_{tag}_short_pct", 100 * sub[sub.short_pk].hit.mean(), 0)
    put(f"pk_{tag}_overlap", int((sub.short & sub.short_pk).sum()))
    grp_ = clusters(sub.pid)
    dif = []
    for _ in range(NBOOT):
        bb_ = sub.iloc[np.concatenate([grp_[i] for i in RNG.integers(0, len(grp_), len(grp_))])]
        if bb_.short.sum() == 0 or bb_.short_pk.sum() == 0:
            continue
        dif.append(bb_[bb_.short].hit.mean() - bb_[bb_.short_pk].hit.mean())
    put(f"pk_{tag}_diff", 100 * (sub[sub.short].hit.mean() - sub[sub.short_pk].hit.mean()), 0)
    put(f"pk_{tag}_diff_lo", 100 * np.percentile(dif, 5), 0); put(f"pk_{tag}_diff_hi", 100 * np.percentile(dif, 95), 0)
    # time-forward: the 2024 -> 2025 pairs alone (forecast fitted on 2023 -> 2024 only)
    lt_ = sub[sub.s1 == 2024]
    put(f"tf_{tag}_short_n", int(lt_.short.sum())); put(f"tf_{tag}_short_hit", int(lt_[lt_.short].hit.sum()))
    put(f"tf_{tag}_rest_n", int((~lt_.short).sum())); put(f"tf_{tag}_rest_hit", int(lt_[~lt_.short].hit.sum()))
    put(f"tf_{tag}_pk_hit", int(lt_[lt_.short_pk].hit.sum())); put(f"tf_{tag}_pk_n", int(lt_.short_pk.sum()))
    put(f"tf_{tag}_base_pct", 100 * lt_.hit.mean(), 0)
    e_ = sub[sub.s1 == 2023]
    put(f"bw_{tag}_short_n", int(e_.short.sum())); put(f"bw_{tag}_short_hit", int(e_[e_.short].hit.sum()))
    put(f"bw_{tag}_rest_n", int((~e_.short).sum())); put(f"bw_{tag}_rest_hit", int(e_[~e_.short].hit.sum()))
# persistence beyond pick and opportunity (inches of season-2 SOE per inch of season-1 SOE), all rounds
A_p = np.column_stack([np.ones(len(PAIRS)), PAIRS.fc_pk, PAIRS.y1])
bp_ = np.linalg.lstsq(A_p, PAIRS.y2.to_numpy(), rcond=None)[0]
grpP = clusters(PAIRS.pid)
bpb_ = []
for _ in range(NBOOT):
    ix = np.concatenate([grpP[i] for i in RNG.integers(0, len(grpP), len(grpP))])
    bpb_.append(np.linalg.lstsq(A_p[ix], PAIRS.y2.to_numpy()[ix], rcond=None)[0][2])
put("persist_b", bp_[2]); put("persist_lo", np.percentile(bpb_, 5)); put("persist_hi", np.percentile(bpb_, 95))
put("rho_fc_y2", spearmanr(PAIRS.fc_pk, PAIRS.y2)[0])
put("sl_all_twice", int((PAIRS.groupby("pid").size() == 2).sum()))
put("sl_d3_twice", int((PAIRS[PAIRS.day3].groupby("pid").size() == 2).sum()))
put("sl_d3_players_n", PAIRS[PAIRS.day3].pid.nunique())
# day-3 hit rates as Wilson 90% intervals (written for the text)
for tag, h_k, n_k in (("d3s", "sl_d3_short_hit", "sl_d3_short_n"), ("d3r", "sl_d3_rest_hit", "sl_d3_rest_n")):
    h_, n_ = int(NUM[h_k]), int(NUM[n_k])
    p_ = h_ / n_; z_ = 1.645
    c_ = (p_ + z_ * z_ / (2 * n_)) / (1 + z_ * z_ / n_)
    hw_ = z_ * np.sqrt(p_ * (1 - p_) / n_ + z_ * z_ / (4 * n_ * n_)) / (1 + z_ * z_ / n_)
    put(f"wil_{tag}_lo", 100 * (c_ - hw_), 0); put(f"wil_{tag}_hi", 100 * (c_ + hw_), 0)
print(PAIRS[["display_name", "s1", "draft_round", "r1", "y1", "y2", "fc_pk", "short", "short_pk", "BG", "day3"]]
      .sort_values(["day3", "s1", "y1"]).to_string())

# ---- (q) decision cases for the board: the 2026 shortlist receivers whose Combine cut and NFL tape disagree.
# For each 2026 shortlist receiver with a drill score: 2025 SOE, routes, NFL Break G (z) on matched families, number of
# NFL breaks, and his best route family by snap separation on the sampled snaps (3+ snaps).
s25i = s25.set_index("nfl_id")
DEC = []
for pid_, r_ in s25i.iterrows():
    gq_ = g[g.nfl_id == pid_]
    gs_ = gb[(gb.nfl_id == pid_) & gb.soe.notna() & gb.route_ran.isin(FAMILIES)]
    fam_ = gs_.groupby("route_ran").soe.agg(["mean", "size"])
    fam_ = fam_[fam_["size"] >= 3]
    DEC.append(dict(nfl_id=pid_, name=r_.display_name, bg=r_.BG, soe=IN_YD * r_.ssoe, routes=int(r_.routes),
                    n_nfl=len(gq_), g_nfl=gq_.bg.median() if len(gq_) else np.nan,
                    gz_nfl=gq_.bg_gz.mean() if len(gq_) else np.nan,
                    g_drill=cb[cb.nfl_id == pid_].bg.median() if (cb.nfl_id == pid_).any() else np.nan,
                    best_fam=fam_["mean"].idxmax().lower() if len(fam_) else None,
                    best_fam_soe=IN_YD * fam_["mean"].max() if len(fam_) else np.nan,
                    slot=100 * (gp[gp.nfl_id == pid_].lined_up_position == "SLOT_WR").mean()))
DEC = pd.DataFrame(DEC)
print(DEC.round(2).to_string())
for i_, r_ in DEC.iterrows():
    if not pd.isna(r_.g_nfl):
        put(f"dec{i_}_gnfl", r_.g_nfl); put(f"dec{i_}_nnfl", int(r_.n_nfl))
    if not pd.isna(r_.g_drill):
        put(f"dec{i_}_gdrill", r_.g_drill)
    if not pd.isna(r_.gz_nfl):
        put(f"dec{i_}_gznfl", r_.gz_nfl, 1)
    if r_.best_fam:
        NUM[f"dec{i_}_bestfam"] = r_.best_fam; put(f"dec{i_}_bestfam_soe", r_.best_fam_soe, 0)
    put(f"dec{i_}_slot", r_.slot, 0)
put("bg_game_q75", g.bg.quantile(0.75)); put("bg_game_q25", g.bg.quantile(0.25))
for i_, r_ in DEC.iterrows():
    if r_.best_fam:
        gs_ = gb[(gb.nfl_id == r_.nfl_id) & gb.soe.notna() & (gb.route_ran == r_.best_fam.upper())]
        put(f"dec{i_}_bestfam_n", len(gs_))
# named season-pair cases (inches): where the drill cut and last season's SOE pointed different ways
for nm_, s1_, key_ in (("Malik Washington", 2024, "mw"), ("Trey Palmer", 2023, "tp"), ("Devaughn Vele", 2024, "dv"),
                       ("Kayshon Boutte", 2024, "kb")):
    rr_ = PAIRS[(PAIRS.display_name == nm_) & (PAIRS.s1 == s1_)]
    if len(rr_):
        rr_ = rr_.iloc[0]
        put(f"case_{key_}_y1", IN_YD * rr_.y1, 1); put(f"case_{key_}_y2", IN_YD * rr_.y2, 1)
        put(f"case_{key_}_bg", rr_.BG, 1)
# persistence net of pick and routes inside the day-3 / UDFA pool alone
P3 = PAIRS[PAIRS.day3]
A3 = np.column_stack([np.ones(len(P3)), P3.fc_pk, P3.y1])
b3_ = np.linalg.lstsq(A3, P3.y2.to_numpy(), rcond=None)[0]
g3 = clusters(P3.pid)
b3b = []
for _ in range(NBOOT):
    ix = np.concatenate([g3[i] for i in RNG.integers(0, len(g3), len(g3))])
    b3b.append(np.linalg.lstsq(A3[ix], P3.y2.to_numpy()[ix], rcond=None)[0][2])
put("persist_d3_b", b3_[2]); put("persist_d3_lo", np.percentile(b3b, 5)); put("persist_d3_hi", np.percentile(b3b, 95))
tick("r14 shortlist done")

# ============================================================================= round 14 revision
# Everything below produces the numbers quoted in writeup.md. Only these are written to numbers.json (one key per
# quoted number; claims.json says which sentence uses each key). The intermediate dictionary NUM above is internal.
from scipy.stats import norm as _norm  # noqa: E402

Q: dict[str, str] = {}
CLAIM: dict[str, str] = {}


def fmt(val, nd=1, sign=False):
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    s = f"{float(val):+.{nd}f}" if sign else f"{float(val):.{nd}f}"
    if float(s) == 0:
        s = s.replace("-", "").replace("+", "")
    return s


def qput(key, val, nd=1, claim=""):
    if key in Q:
        raise KeyError(f"duplicate key {key}")
    Q[key] = fmt(val, nd)
    CLAIM[key] = claim
    return Q[key]


def pct(a, p):
    return float(np.percentile(a, p))


# ---------------------------------------------------------------- R1. the pre-specified primary test
# The forecast fixed in round 9 (frozen_rule.json): plain least squares on log pick, 40, weight, NGS athleticism and TE,
# fitted on two classes and used as an offset on the third. That is the primary. The ridge forecast (round 13) and the
# outcome adjustments are sensitivity rows. Every row comes from the same 2,000 (rows: 1,000) class-stratified resamples
# that refit the forecast inside each resample (COMBO, column 1 = plain offset, column 2 = ridge covariate).
qput("n_main", len(D), 0, "116 WR/TE with a drill score and NFL routes")
qput("n_seasons", int(NUM["n_player_seasons"]), 0, "213 player-seasons")
qput("routes_med", int(NUM["routes_median"]), 0, "median 231 career routes")
SENS = [("prim", "Pre-specified forecast (primary)", COMBO["main"], 1),
        ("ridge", "Ridge forecast instead", COMBO["main"], 2),
        ("nn", "Without Puka Nacua, the most influential player", COMBO["nn"], 1),
        ("mix", "SOE adjusted for route mix and depth", COMBO["mix"], 1),
        ("qb", "SOE adjusted for quarterback", COMBO["qb"], 1)]
SROWS = []
for k_, lab_, cm_, col_ in SENS:
    b_, lo_, hi_ = cm_["est"][col_], pct(cm_["bs"][:, col_], 5), pct(cm_["bs"][:, col_], 95)
    SROWS.append((lab_, b_, lo_, hi_, cm_["n"]))
    Q[f"{k_}_b"] = fmt(b_, 1, True); CLAIM[f"{k_}_b"] = f"{lab_}: inches of season SOE per sd of Break G"
    Q[f"{k_}_lo"] = fmt(lo_, 1, True); CLAIM[f"{k_}_lo"] = f"{lab_}: 90% lower"
    Q[f"{k_}_hi"] = fmt(hi_, 1, True); CLAIM[f"{k_}_hi"] = f"{lab_}: 90% upper"
PRIM_B, PRIM_SE = COMBO["main"]["est"][1], COMBO["main"]["bs"][:, 1].std(ddof=1)
qput("nn_n", COMBO["nn"]["n"], 0, "115 receivers without Nacua")
qput("max_hi", max(r[3] for r in SROWS), 1, "highest upper end of any row")
# within-class shuffle of Break G for the primary (forecast fixed: it does not use Break G)
nul_p = []
for _ in range(NPERM):
    pv_ = BGV.copy()
    for c in CLASSES:
        ix = np.where(cls_arr0 == c)[0]
        pv_[ix] = BGV[RNG.permutation(ix)]
    nul_p.append(IN_YD * _wls(np.column_stack([dmy_all, pv_]), Y_ - fo_all, WT_)[-1])
qput("prim_p", (1 + (np.abs(nul_p) >= abs(PRIM_B)).sum()) / (1 + NPERM), 2, "within-class shuffle p, primary")
qput("prim_up_p", _norm.cdf((PRIM_B - SESOI_IN) / PRIM_SE), 3, "one-sided p against a +3 in effect")
qput("sesoi_catch", SESOI_IN * catch_per_in, 2, "+3 in is worth 0.44 catches a season")
qput("cr_per_yd", float(NUM["cr_per_yd"]), 1, "catch-rate points per yard of separation")
qput("tg_season", int(NUM["tg_season"]), 0, "targets for a median 200-route receiver")
qput("power26", float(NUM["power26_3in"]), 2, "power of the 2026 class against +3 in")
tick("R1 primary")

# ---------------------------------------------------------------- R2. last season's SOE as a forecast (day 3 / UDFA)
# (a) Empirical-Bayes SOE: a season's SOE is true skill (variance tau^2) plus noise whose variance is s^2 / routes.
# tau^2 = covariance of the same receiver's SOE in consecutive seasons (WR/TE, 50+ routes in both; skill carries,
# noise does not); s^2 from how the spread around the mean falls with routes, intercept held at tau^2.
# Shrunk SOE = mean + k (SOE - mean), k = tau^2 / (tau^2 + s^2 / routes).
ssm = ss[ss.routes >= 50].copy()
mu_ = np.average(ssm.ssoe, weights=ssm.routes)
pv2 = ssm.pivot_table(index="nfl_id", columns="season", values="ssoe")
cp_ = pd.concat([pv2[[a, b]].dropna().set_axis(["a", "b"], axis=1) for a, b in ((2023, 2024), (2024, 2025))])
TAU2 = float(np.mean((cp_.a - mu_) * (cp_.b - mu_)))
iv_ = 1.0 / ssm.routes.to_numpy(float)
S2 = float(np.sum(((ssm.ssoe.to_numpy() - mu_) ** 2 - TAU2) * iv_) / np.sum(iv_ ** 2))
qput("eb_pairs", len(cp_), 0, "consecutive-season pairs (50+ routes) used for the skill variance")


def shrink(y, r):
    k = TAU2 / (TAU2 + S2 / np.asarray(r, float))
    return mu_ + k * (np.asarray(y, float) - mu_), k


qput("eb_n", len(ssm), 0, "player-seasons used for the shrinkage")
qput("eb_k200", TAU2 / (TAU2 + S2 / 200), 2, "weight kept on last season's SOE at 200 routes")
qput("eb_half_routes", S2 / TAU2, 0, "routes at which half the SOE is kept")
PAIRS["y1s"], PAIRS["k1"] = shrink(PAIRS.y1, PAIRS.r1)

# (b) leave-one-season-out forecasts of next-season SOE. Fold = season pair (2023->24, 2024->25); each model is fitted
# on the other pair. M0 = pick and routes (log pick, TE, log season-1 routes). M1 = M0 + shrunk season-1 SOE.
# Probabilities P(next SOE > 0) = Phi(forecast / residual sd of the training fold).
F0 = ["logpick", "te", "lr1"]
F1 = F0 + ["y1s"]


def fitp(df, cols):
    A = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    b = np.linalg.lstsq(A, df.y2.to_numpy(float), rcond=None)[0]
    return b, float(np.std(df.y2.to_numpy(float) - A @ b, ddof=len(b)))


def predp(df, b, cols):
    return np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)]) @ b


for m_, cols in (("m0", F0), ("m1", F1)):
    PAIRS[f"f_{m_}"] = np.nan
    PAIRS[f"p_{m_}"] = np.nan
    for s_ in (2023, 2024):
        trn, tst = PAIRS[PAIRS.s1 != s_], PAIRS.s1 == s_
        b_, sd_ = fitp(trn, cols)
        PAIRS.loc[tst, f"f_{m_}"] = predp(PAIRS[tst], b_, cols)
        PAIRS.loc[tst, f"p_{m_}"] = _norm.cdf(PAIRS.loc[tst, f"f_{m_}"] / sd_)


def metrics(df):
    out = {}
    for m_ in ("m0", "m1"):
        e_ = df.y2 - df[f"f_{m_}"]
        out[m_] = dict(rmse=IN_YD * np.sqrt(np.mean(e_ ** 2)), rho=spearmanr(df[f"f_{m_}"], df.y2)[0],
                       brier=np.mean((df[f"p_{m_}"] - df.hit.astype(float)) ** 2))
    return out


EVAL = {}
for tag, sub in (("d3", PAIRS[PAIRS.day3]), ("all", PAIRS)):
    mt = metrics(sub)
    grp_ = clusters(sub.pid)
    bs_ = []
    for _ in range(NBOOT):
        bb_ = sub.iloc[np.concatenate([grp_[i] for i in RNG.integers(0, len(grp_), len(grp_))])]
        m2 = metrics(bb_)
        bs_.append([m2["m1"]["rho"] - m2["m0"]["rho"], m2["m0"]["rmse"] - m2["m1"]["rmse"],
                    m2["m0"]["brier"] - m2["m1"]["brier"]])
    bs_ = np.array(bs_)
    EVAL[tag] = dict(mt=mt, bs=bs_, n=len(sub), np=sub.pid.nunique())
    qput(f"ev_{tag}_n", len(sub), 0, f"{tag}: season pairs")
    qput(f"ev_{tag}_np", sub.pid.nunique(), 0, f"{tag}: receivers")
    for m_ in ("m0", "m1"):
        qput(f"ev_{tag}_{m_}_rho", mt[m_]["rho"], 2, f"{tag} {m_}: held-out rank correlation with next-season SOE")
        qput(f"ev_{tag}_{m_}_rmse", mt[m_]["rmse"], 1, f"{tag} {m_}: held-out RMSE, inches")
        qput(f"ev_{tag}_{m_}_brier", mt[m_]["brier"], 3, f"{tag} {m_}: held-out Brier score")
    for j_, nm_ in enumerate(("drho", "drmse", "dbrier")):
        pt_ = [mt["m1"]["rho"] - mt["m0"]["rho"], mt["m0"]["rmse"] - mt["m1"]["rmse"],
               mt["m0"]["brier"] - mt["m1"]["brier"]][j_]
        nd_ = {0: 2, 1: 1, 2: 3}[j_]
        qput(f"ev_{tag}_{nm_}", pt_, nd_, f"{tag}: gain of the SOE forecast over pick and routes ({nm_})")
        qput(f"ev_{tag}_{nm_}_lo", pct(bs_[:, j_], 5), nd_, f"{tag}: {nm_} 90% lower, receiver-clustered")
        qput(f"ev_{tag}_{nm_}_hi", pct(bs_[:, j_], 95), nd_, f"{tag}: {nm_} 90% upper, receiver-clustered")
        qput(f"ev_{tag}_{nm_}_pos", 100 * (bs_[:, j_] > 0).mean(), 0, f"{tag}: % of resamples where SOE forecast wins")
# per fold (graft from the DL shortlist candidate: is the edge there in every fold?)
for tag, sub in (("d3", PAIRS[PAIRS.day3]), ("all", PAIRS)):
    wins = sum(metrics(sub[sub.s1 == s_])["m1"]["rho"] > metrics(sub[sub.s1 == s_])["m0"]["rho"] for s_ in (2023, 2024))
    qput(f"ev_{tag}_folds_won", int(wins), 0, f"{tag}: folds where SOE forecast ranks better")

# (c) the yes/no rule: shortlist if shrunk season-1 SOE is above a threshold chosen on the OTHER season pair (all
# rounds) by Youden's J for next-season SOE > 0. Pick-and-routes lists the same number per fold and pool, ranked by M0.


def youden_t(df):
    best, bt = -9, 0.0
    for t in np.unique(df.y1s):
        pos = df.y1s >= t
        if pos.sum() < 3 or (~pos).sum() < 3:
            continue
        j = (pos & df.hit).sum() / max(df.hit.sum(), 1) + (~pos & ~df.hit).sum() / max((~df.hit).sum(), 1) - 1
        if j > best:
            best, bt = j, t
    return bt


PAIRS["rule"] = False
THR = {}
for s_ in (2023, 2024):
    t_ = youden_t(PAIRS[PAIRS.s1 != s_])
    THR[s_] = t_
    PAIRS.loc[PAIRS.s1 == s_, "rule"] = PAIRS.loc[PAIRS.s1 == s_, "y1s"] >= t_
PAIRS["rule_pk"] = False
for (s_, d3_), sub_ in PAIRS.groupby(["s1", "day3"]):
    k_ = int(sub_.rule.sum())
    PAIRS.loc[sub_.sort_values("f_m0", ascending=False).index[:k_], "rule_pk"] = True
qput("thr_23", IN_YD * THR[2023], 1, "threshold for the 2023->24 fold, chosen on 2024->25 (in)")
qput("thr_24", IN_YD * THR[2024], 1, "threshold for the 2024->25 fold, chosen on 2023->24 (in)")
for tag, sub in (("d3", PAIRS[PAIRS.day3]), ("all", PAIRS)):
    for nm_, msk in (("rule", sub.rule), ("pk", sub.rule_pk), ("rest", ~sub.rule)):
        qput(f"r_{tag}_{nm_}_hit", int(sub[msk].hit.sum()), 0, f"{tag} {nm_}: hits")
        qput(f"r_{tag}_{nm_}_n", int(msk.sum()), 0, f"{tag} {nm_}: listed")
        qput(f"r_{tag}_{nm_}_pct", 100 * sub[msk].hit.mean(), 0, f"{tag} {nm_}: hit rate %")
    grp_ = clusters(sub.pid)
    dif = []
    for _ in range(NBOOT):
        bb_ = sub.iloc[np.concatenate([grp_[i] for i in RNG.integers(0, len(grp_), len(grp_))])]
        if bb_.rule.sum() == 0 or bb_.rule_pk.sum() == 0:
            continue
        dif.append(100 * (bb_[bb_.rule].hit.mean() - bb_[bb_.rule_pk].hit.mean()))
    qput(f"r_{tag}_diff", 100 * (sub[sub.rule].hit.mean() - sub[sub.rule_pk].hit.mean()), 0, f"{tag}: rule minus pk, pts")
    qput(f"r_{tag}_diff_lo", pct(dif, 5), 0, f"{tag}: 90% lower"); qput(f"r_{tag}_diff_hi", pct(dif, 95), 0, f"{tag}: upper")
    qput(f"r_{tag}_base", 100 * sub.hit.mean(), 0, f"{tag}: base rate %")
    qput(f"r_{tag}_overlap", int((sub.rule & sub.rule_pk).sum()), 0, f"{tag}: names on both lists")
qput("twice_all", int((PAIRS.groupby("pid").size() == 2).sum()), 0, "receivers in two pairs, all rounds")
qput("twice_d3", int((PAIRS[PAIRS.day3].groupby("pid").size() == 2).sum()), 0, "receivers in two pairs, day 3")
# persistence slope beyond pick and routes (kept from the previous round, now on shrunk SOE too)
for tag, sub in (("d3", PAIRS[PAIRS.day3]), ("all", PAIRS)):
    A_ = np.column_stack([np.ones(len(sub)), sub[F0].to_numpy(float), sub.y1.to_numpy(float)])
    b_ = np.linalg.lstsq(A_, sub.y2.to_numpy(float), rcond=None)[0][-1]
    grp_ = clusters(sub.pid)
    bb = [np.linalg.lstsq(A_[ix], sub.y2.to_numpy(float)[ix], rcond=None)[0][-1]
          for ix in (np.concatenate([grp_[i] for i in RNG.integers(0, len(grp_), len(grp_))]) for _ in range(NBOOT))]
    qput(f"pers_{tag}", b_, 2, f"{tag}: inches of next SOE per inch of last SOE beyond pick and routes")
    qput(f"pers_{tag}_lo", pct(bb, 5), 2, f"{tag}: 90% lower"); qput(f"pers_{tag}_hi", pct(bb, 95), 2, f"{tag}: upper")

# (d) the 2026 list: day-3 / UDFA WR/TE with 100+ routes and SOE above zero in 2025 (as in the previous round).
# Probability of 2026 SOE > 0 from M1 and from M0, both fitted on all 53 pairs; parameter uncertainty by averaging
# Phi(forecast / sd) over receiver-clustered refits.
gA = clusters(PAIRS.pid)
pk_all = players.set_index("nfl_id").draft_overall_pick
L26 = s25.copy()
L26["logpick"] = np.log(pk_all.reindex(L26.nfl_id).fillna(262).to_numpy(float))
L26["te"] = (L26.position_group == "TE").astype(float).to_numpy()
L26["lr1"] = np.log(L26.routes.to_numpy(float))
L26["y1s"], L26["k1"] = shrink(L26.ssoe, L26.routes)
P26 = {"m0": [], "m1": []}
for bi_ in range(NBOOT // 2):
    bb_ = PAIRS.iloc[np.concatenate([gA[i] for i in RNG.integers(0, len(gA), len(gA))])] if bi_ else PAIRS
    for m_, cols in (("m0", F0), ("m1", F1)):
        b_, sd_ = fitp(bb_, cols)
        P26[m_].append(_norm.cdf(predp(L26, b_, cols) / sd_))
for m_ in P26:
    L26[f"p_{m_}"] = 100 * np.mean(P26[m_], axis=0)
T_ALL = youden_t(PAIRS)
L26["rule"] = L26.y1s >= T_ALL
L26 = L26.sort_values("p_m1", ascending=False).reset_index(drop=True)
qput("l26_n", len(L26), 0, "names on the 2026 list")
qput("l26_rule_n", int(L26.rule.sum()), 0, "names above the threshold fitted on both pairs")
qput("thr_all", IN_YD * T_ALL, 1, "threshold fitted on both pairs (in)")
qput("l26_soft", int((L26.BG < 0).sum()), 0, "soft drill cutters on the 2026 list")
SL26 = []
for i_, r_ in L26.iterrows():
    pk_ = pk_all.get(r_.nfl_id)
    SL26.append(dict(name=r_.display_name, year=int(r_.draft_year), pick="UDFA" if pd.isna(pk_) else int(pk_),
                     pos=r_.position_group, routes=int(r_.routes), soe_in=round(IN_YD * r_.ssoe, 1),
                     soe_shrunk_in=round(IN_YD * r_.y1s, 1), p_soe=round(r_.p_m1), p_pick_routes=round(r_.p_m0),
                     above_threshold=bool(r_.rule), break_g_sd=None if pd.isna(r_.BG) else round(float(r_.BG), 1)))
    tg_ = r_.display_name.split()[-1].lower().replace("'", "").replace(".", "")
    qput(f"l26_{tg_}_routes", int(r_.routes), 0, f"{r_.display_name}: 2025 routes")
    Q[f"l26_{tg_}_soe"] = fmt(IN_YD * r_.ssoe, 1, True); CLAIM[f"l26_{tg_}_soe"] = f"{r_.display_name}: 2025 SOE"
    Q[f"l26_{tg_}_shr"] = fmt(IN_YD * r_.y1s, 1, True); CLAIM[f"l26_{tg_}_shr"] = f"{r_.display_name}: shrunk SOE"
    qput(f"l26_{tg_}_p1", r_.p_m1, 0, f"{r_.display_name}: P(2026 SOE > 0), SOE forecast")
    qput(f"l26_{tg_}_p0", r_.p_m0, 0, f"{r_.display_name}: P(2026 SOE > 0), pick and routes")
    if not pd.isna(r_.BG):
        Q[f"l26_{tg_}_bg"] = fmt(r_.BG, 1, True); CLAIM[f"l26_{tg_}_bg"] = f"{r_.display_name}: Break G sd"
pd.DataFrame(SL26).to_csv(OUT / "shortlist_2026.csv", index=False)
print(pd.DataFrame(SL26).to_string())
print(PAIRS[PAIRS.day3][["display_name", "s1", "r1", "y1", "y1s", "y2", "f_m0", "f_m1", "rule", "rule_pk", "hit"]]
      .assign(y1=lambda d: IN_YD * d.y1, y1s=lambda d: IN_YD * d.y1s, y2=lambda d: IN_YD * d.y2).round(2).to_string())
tick("R2 shortlist")

# ---------------------------------------------------------------- R3. drill to tape
qput("n_reps", int(NUM["n_drill_reps"]), 0, "926 drill reps"); qput("pct_found", int(NUM["pct_drill_detected"]), 0, "96% found")
qput("drills_med", int(NUM["median_drills_per_player"]), 0, "median 7 drills")
qput("rel_split", REL["bg"][0], 2, "five drills predict the other five"); qput("rel_split_lo", REL["bg"][1], 2, "lower")
qput("rel_split_hi", REL["bg"][2], 2, "upper"); qput("rel_split_n", REL["bg"][3], 0, "players")
qput("rho_forty", float(NUM["rho_bg_forty"]), 2, "Break G vs official 40")
qput("pl_rho", pr[0], 2, "Combine vs NFL Break G, all players"); qput("pl_lo", pr[1], 2, "lower"); qput("pl_hi", pr[2], 2, "upper")
qput("pl_n", pr[3], 0, "players with drill score and NFL breaks")
qput("p8_n", len(P8), 0, "players with 8+ NFL breaks"); qput("p8_med", int(P8.ng.median()), 0, "median NFL breaks")
qput("p8_r", d8[0], 2, "observed r, 8+ breaks")
qput("nfl_rel", float(NUM["grel_bg"]), 2, "NFL Break G split-half reliability")
qput("rel_mm1", float(NUM["mm_rel_7x1"]), 2, "reliability, seven drills one rep")
qput("rel_mm3", float(NUM["mm_rel_7p3"]), 2, "reliability, second rep on three drills")
qput("vp_pct", 100 * VP, 0, "share of a single rep that is player")

# (a) Bayesian errors-in-variables reading of the 8+ break correlation. True drill skill and true NFL cut skill
# correlate at rho; each is measured with reliability R_d, R_g (split-half, Spearman-Brown), so the observed r has
# expectation rho * sqrt(R_d R_g). Likelihood: Fisher z of r ~ N(atanh(rho sqrt(R_d R_g)), 1 / (n - 3)), averaged over
# the bootstrap draws of the two reliabilities (their uncertainty). Prior: rho uniform on [-1, 1].
# Bayes factor: H1 "the drill cut carries" (rho uniform on [0, 1]) against H0 (rho = 0).
RG = np.linspace(-1, 1, 2001)
att_ = np.sqrt(np.clip(bs8[:, 1], 0.02, 1) * np.clip(bs8[:, 2], 0.02, 1))
z_obs, se_z = np.arctanh(d8[0]), 1 / np.sqrt(len(P8) - 3)
LIK = np.array([_norm.pdf(z_obs, np.arctanh(np.clip(r_ * att_, -0.999, 0.999)), se_z).mean() for r_ in RG])
post_ = LIK / LIK.sum()
cdf_ = np.cumsum(post_)
l0_ = LIK[np.argmin(np.abs(RG))]
BF10 = LIK[RG >= 0].mean() / l0_
qput("eiv_bf01", 1 / BF10, 1, "Bayes factor for no link over a positive link")
qput("eiv_med", RG[np.searchsorted(cdf_, 0.5)], 2, "posterior median true correlation")
qput("eiv_lo", RG[np.searchsorted(cdf_, 0.05)], 2, "posterior 5%"); qput("eiv_hi", RG[np.searchsorted(cdf_, 0.95)], 2, "95%")
qput("eiv_p05", 100 * post_[RG >= 0.5].sum(), 0, "posterior chance true link >= 0.5")
qput("eiv_p03", 100 * post_[RG >= 0.3].sum(), 0, "posterior chance true link >= 0.3")
qput("eiv_att", float(np.median(att_)), 2, "median attenuation factor")
qput("p8_reld", d8[1], 2, "drill split-half reliability, 8+ players"); qput("p8_relg", d8[2], 2, "NFL split-half, 8+")


# (b) slices: within route family (drill family score vs NFL family score) and by NFL alignment / coverage (overall
# drill score vs NFL Break G on those snaps). Players with 3+ NFL breaks in the slice. Reliability: one-way ICC on
# snaps (NFL) or reps (drill) within the slice, Spearman-Brown to the median count per player.
def sb_k(icc, k):
    return k * icc / (1 + (k - 1) * icc) if icc > 0 else 0.0


SLICES = []
for fam_ in ("SLANT", "OUT", "HITCH"):
    gs_ = g[g.route_ran == fam_]
    ng_ = gs_.groupby("nfl_id").size()
    gm_ = gs_.groupby("nfl_id").bg_gz.mean()[ng_ >= 3]
    cr_ = cb[cb.family == fam_]
    dm_ = cr_.groupby("nfl_id").bg_z.mean()
    jj_ = pd.concat([dm_.rename("d"), gm_.rename("g")], axis=1).dropna()
    r_ = boot_rho(jj_.d, jj_.g)
    rel_g = sb_k(icc1(gs_[gs_.nfl_id.isin(gm_.index)], "bg_gz"), float(ng_[ng_ >= 3].median()))
    nrep_ = cr_.groupby("nfl_id").size()
    rel_d = sb_k(icc1(cr_, "bg_z"), float(nrep_.median()))
    SLICES.append(dict(k=fam_.lower(), lab=f"{fam_.lower()} only (drill {fam_.lower()} vs NFL {fam_.lower()})",
                       rho=r_[0], lo=r_[1], hi=r_[2], n=r_[3], rel_g=rel_g, rel_d=rel_d))
for k_, lab_, msk in (("slot", "NFL snaps from the slot", g.lined_up_position == "SLOT_WR"),
                      ("wide", "NFL snaps lined up wide", (g.slotp == 0) & (g.inline == 0)),
                      ("man", "NFL snaps against man", g.man),
                      ("zone", "NFL snaps against zone", ~g.man)):
    gs_ = g[msk]
    ng_ = gs_.groupby("nfl_id").size()
    gm_ = gs_.groupby("nfl_id").bg_gz.mean()[ng_ >= 3]
    jj_ = pd.concat([X.BG.rename("d"), gm_.rename("g")], axis=1).dropna()
    r_ = boot_rho(jj_.d, jj_.g)
    rel_g = sb_k(icc1(gs_[gs_.nfl_id.isin(gm_.index)], "bg_gz"), float(ng_[ng_ >= 3].median()))
    SLICES.append(dict(k=k_, lab=lab_, rho=r_[0], lo=r_[1], hi=r_[2], n=r_[3], rel_g=rel_g, rel_d=np.nan))
SLICES = pd.DataFrame(SLICES)
print(SLICES.round(2).to_string())
for _, r_ in SLICES.iterrows():
    qput(f"sl_{r_.k}_rho", r_.rho, 2, f"{r_.lab}: rho"); qput(f"sl_{r_.k}_lo", r_.lo, 2, "lower")
    qput(f"sl_{r_.k}_hi", r_.hi, 2, "upper"); qput(f"sl_{r_.k}_n", int(r_.n), 0, "players")
    qput(f"sl_{r_.k}_relg", r_.rel_g, 2, f"{r_.lab}: NFL reliability")
    if not np.isnan(r_.rel_d):
        qput(f"sl_{r_.k}_reld", r_.rel_d, 2, f"{r_.lab}: drill reliability")
qput("slice_max_hi", SLICES.hi.max(), 2, "highest upper end over the seven slices")
# how hard NFL cuts are by alignment and coverage (raw g, receiver-clustered difference)
for k_, a_m, b_m in (("slotwide", g.lined_up_position == "SLOT_WR", (g.slotp == 0) & (g.inline == 0)),
                     ("manzone", g.man, ~g.man)):
    d_ = g.assign(a=a_m.to_numpy(), b=b_m.to_numpy())
    grp_ = clusters(d_.nfl_id)
    est_ = d_[d_.a].bg.mean() - d_[d_.b].bg.mean()
    bb_ = []
    for _ in range(NBOOT):
        e_ = d_.iloc[np.concatenate([grp_[i] for i in RNG.integers(0, len(grp_), len(grp_))])]
        bb_.append(e_[e_.a].bg.mean() - e_[e_.b].bg.mean())
    qput(f"g_{k_}", est_, 2, f"{k_}: difference in NFL Break G, g"); qput(f"g_{k_}_lo", pct(bb_, 5), 2, "lower")
    qput(f"g_{k_}_hi", pct(bb_, 95), 2, "upper")
qput("g_slot", g[g.lined_up_position == "SLOT_WR"].bg.median(), 2, "median NFL Break G from the slot")
qput("g_wide", g[(g.slotp == 0) & (g.inline == 0)].bg.median(), 2, "median NFL Break G wide")
qput("g_drill_med", cb.bg.median(), 2, "median drill Break G"); qput("g_nfl_med", g.bg.median(), 2, "median NFL Break G")
qput("n_nfl_breaks", len(g), 0, "graded NFL breaks on the five families")
tick("R3 drill to tape")

# ---------------------------------------------------------------- R4. where drill and tape disagree (2026 list)
# For the named cases: NFL Break G on the matched families (z, mean over snaps), median g in drill and on NFL snaps,
# NFL breaks graded, slot share, entry speed into NFL breaks vs the league median; two named season-pair cases; price.
DECI = DEC.set_index("name")
qput("lg_vin", g.v_in.median(), 1, "league median entry speed into NFL breaks, yd/s")
for nm_ in ("Kayshon Boutte", "Isaiah Bond", "Arian Smith", "Malik Washington"):
    r_ = DECI.loc[nm_]
    tg_ = nm_.split()[-1].lower()
    Q[f"dec_{tg_}_gz"] = fmt(r_.gz_nfl, 1, True); CLAIM[f"dec_{tg_}_gz"] = f"{nm_}: NFL Break G, z"
    qput(f"dec_{tg_}_gd", r_.g_drill, 2, f"{nm_}: median drill Break G, g")
    qput(f"dec_{tg_}_gn", r_.g_nfl, 2, f"{nm_}: median NFL Break G, g")
    qput(f"dec_{tg_}_nn", int(r_.n_nfl), 0, f"{nm_}: NFL breaks graded")
    qput(f"dec_{tg_}_slot", r_.slot, 0, f"{nm_}: % of sampled snaps from the slot")
    qput(f"dec_{tg_}_vin", g[g.nfl_id == r_.nfl_id].v_in.median(), 1, f"{nm_}: entry speed into NFL breaks, yd/s")
for nm_, s1_, tg_ in (("Malik Washington", 2024, "mw"), ("Trey Palmer", 2023, "tp")):
    rr_ = PAIRS[(PAIRS.display_name == nm_) & (PAIRS.s1 == s1_)].iloc[0]
    Q[f"case_{tg_}_y1"] = fmt(IN_YD * rr_.y1, 1, True); CLAIM[f"case_{tg_}_y1"] = f"{nm_}: season-1 SOE"
    Q[f"case_{tg_}_y2"] = fmt(IN_YD * rr_.y2, 1, True); CLAIM[f"case_{tg_}_y2"] = f"{nm_}: season-2 SOE"
    Q[f"case_{tg_}_bg"] = fmt(rr_.BG, 1, True); CLAIM[f"case_{tg_}_bg"] = f"{nm_}: Break G sd"
PVk = PV.set_index("k")
qput("price_bg", PVk.loc["bg", "price"], 0, "clubs pay % earlier pick per sd of Break G")
qput("price_bg_lo", PVk.loc["bg", "plo"], 0, "lower"); qput("price_bg_hi", PVk.loc["bg", "phi"], 0, "upper")
qput("price_forty", PVk.loc["forty", "price"], 0, "price of the 40")
Q["val_forty"] = fmt(PVk.loc["forty", "val"], 1, True); CLAIM["val_forty"] = "value of the 40, in per sd"
Q["val_weight"] = fmt(PVk.loc["combine_weight", "val"], 1, True); CLAIM["val_weight"] = "value of weight, in per sd"
tick("R4 cases")


# ============================================================================= round 15 additions (cand_2)
# R5 (a) survivorship: who reaches the primary sample? Selection model on every drill-scored WR/TE with a baseline
# (logistic on the five forecast inputs plus Break G), then the primary slope re-weighted by 1 / P(selected).
# R5 (b) one table of Break G slopes: plain offset (primary), ridge, no offset (covariate in one regression), IPW,
# receiver random-intercept mixed model on player-seasons, and the outcome / influence variants; one-sided p against +3.
# R5 (c) day-3 list: permutation tests of the yes/no gap and the rank-correlation gain; receiver random-intercept model
# of persistence; calibration of the held-out probabilities.
# R5 (d) 2026 list context (games, alignment, team) and a worked decision case on two 2025 day-3 receivers.
import warnings  # noqa: E402

import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

warnings.filterwarnings("ignore")
SELC = BASE + ["BG"]
XS = X.copy()
XS["sel"] = XS.index.isin(D.index).astype(int)
XS["sel100"] = XS.index.isin(D.index[D.routes >= 100]).astype(int)
DPOS = pd.Series(np.arange(len(D)), index=D.index)


def sel_fit(df, ycol="sel"):
    return LogisticRegression(C=1e6, max_iter=5000).fit(df[SELC].to_numpy(float), df[ycol].to_numpy(int))


qput("sel_n", len(XS), 0, "drill-scored WR/TE with a baseline")
qput("sel_in", int(XS.sel.sum()), 0, "of them with NFL routes and NGS separation (primary sample)")
qput("sel_out", int((1 - XS.sel).sum()), 0, "excluded: no NFL routes with separation")
qput("sel_100", int(XS.sel100.sum()), 0, "of them with 100+ career routes")
for yc_, tg_ in (("sel", "in"), ("sel100", "r100")):
    m_ = sel_fit(XS, yc_)
    orb = []
    for _ in range(NBOOT // 2):
        ix_ = np.concatenate([RNG.choice(np.where(XS.draft_year.to_numpy() == c)[0], int((XS.draft_year == c).sum()))
                              for c in CLASSES])
        orb.append(np.exp(sel_fit(XS.iloc[ix_], yc_).coef_[0][-1]))
    qput(f"sel_or_{tg_}", np.exp(m_.coef_[0][-1]), 2, f"odds ratio of {yc_} per sd of Break G, given pick and baseline")
    qput(f"sel_or_{tg_}_lo", pct(orb, 5), 2, "lower"); qput(f"sel_or_{tg_}_hi", pct(orb, 95), 2, "upper")
PSEL = np.clip(sel_fit(XS).predict_proba(XS[SELC].to_numpy(float))[:, 1], 0.05, 1)
XS["psel"] = PSEL
IPW_ = 1.0 / XS.psel.reindex(D.index).to_numpy(float)


def prim_w(rows, wm):
    Xi, yi, wi, ci, bi = XB_[rows], Y_[rows], WT_[rows] * wm, cls_arr0[rows], BGV[rows]
    fo = np.empty(len(rows))
    for c in CLASSES:
        t = ci == c
        fo[t] = Xi[t] @ _wls(Xi[~t], yi[~t], wi[~t])
    dmy = [(ci == c).astype(float) for c in CLASSES]
    return IN_YD * _wls(np.column_stack(dmy + [bi]), yi - fo, wi)[-1]


assert abs(prim_w(ALL, np.ones(len(D))) - COMBO["main"]["est"][1]) < 1e-9
ipw_b = prim_w(ALL, IPW_)
ipw_bs = []
xcls = XS.draft_year.to_numpy()
for _ in range(NBOOT // 2):
    ix_ = np.concatenate([RNG.choice(np.where(xcls == c)[0], int((xcls == c).sum())) for c in CLASSES])
    xb_ = XS.iloc[ix_]
    ps_ = np.clip(sel_fit(xb_).predict_proba(xb_[SELC].to_numpy(float))[:, 1], 0.05, 1)
    keep_ = xb_.sel.to_numpy() == 1
    rows_ = DPOS.reindex(xb_.index[keep_]).to_numpy(int)
    ipw_bs.append(prim_w(rows_, 1.0 / ps_[keep_]))
ipw_bs = np.array(ipw_bs)
qput("ipw_wmax", float(IPW_.max()), 1, "largest inverse-probability weight")

# no offset: Break G as a covariate beside the five forecast inputs, one routes-weighted regression on all classes
def nooff(rows):
    A = np.column_stack([XB_[rows], BGV[rows]])
    return IN_YD * _wls(A, Y_[rows], WT_[rows])[-1]


no_b = nooff(ALL)
no_bs = np.array([nooff(strat_sub(ALL)) for _ in range(NBOOT // 2)])

# receiver random-intercept mixed model on player-seasons (50+ routes): season SOE ~ forecast inputs + Break G +
# season + (1 | receiver). MixedLM has no weights, so the 50-route floor stands in for them.
PSX = ss[ss.routes >= 50].join(D[BASE + ["BG", "draft_year"]], on="nfl_id", how="inner").copy()
PSX["y_in"] = IN_YD * PSX.ssoe
fml_ = "y_in ~ " + " + ".join(BASE) + " + BG + C(season)"
mm_ = smf.mixedlm(fml_, PSX, groups=PSX.nfl_id).fit(reml=True)
mm_b, mm_se = float(mm_.params["BG"]), float(mm_.bse["BG"])
qput("mm_ps_n", len(PSX), 0, "player-seasons in the mixed model"); qput("mm_ps_np", PSX.nfl_id.nunique(), 0, "receivers")
qput("mm_icc", float(mm_.cov_re.iloc[0, 0] / (mm_.cov_re.iloc[0, 0] + mm_.scale)), 2,
     "share of season SOE variance that is the receiver (random intercept)")

SL5 = [("plain", "Pre-specified offset forecast (primary)", COMBO["main"]["est"][1], COMBO["main"]["bs"][:, 1], len(D)),
       ("ridge5", "Ridge forecast", COMBO["main"]["est"][2], COMBO["main"]["bs"][:, 2], len(D)),
       ("nooff", "No offset: Break G beside the five inputs", no_b, no_bs, len(D)),
       ("ipw", "Re-weighted for who reaches the NFL sample", ipw_b, ipw_bs, len(D)),
       ("mixed", "Player-seasons, receiver random intercept", mm_b, None, len(PSX)),
       ("nn5", "Without Puka Nacua", COMBO["nn"]["est"][1], COMBO["nn"]["bs"][:, 1], COMBO["nn"]["n"]),
       ("mix5", "SOE adjusted for route mix and depth", COMBO["mix"]["est"][1], COMBO["mix"]["bs"][:, 1], len(D)),
       ("qb5", "SOE adjusted for quarterback", COMBO["qb"]["est"][1], COMBO["qb"]["bs"][:, 1], len(D))]
SLOPES = []
for k_, lab_, b_, bs_, n_ in SL5:
    if bs_ is None:
        lo_, hi_, se_ = b_ - 1.645 * mm_se, b_ + 1.645 * mm_se, mm_se
    else:
        lo_, hi_, se_ = pct(bs_, 5), pct(bs_, 95), float(np.std(bs_, ddof=1))
    p3_ = _norm.cdf((b_ - SESOI_IN) / se_)
    SLOPES.append(dict(k=k_, lab=lab_, b=b_, lo=lo_, hi=hi_, p3=p3_, n=n_))
    if k_ in ("plain", "nn5", "mix5", "qb5", "ridge5"):
        continue  # already quoted under the R1 keys
    Q[f"t5_{k_}_b"] = fmt(b_, 1, True); CLAIM[f"t5_{k_}_b"] = f"{lab_}: slope, in per sd"
    Q[f"t5_{k_}_lo"] = fmt(lo_, 1, True); CLAIM[f"t5_{k_}_lo"] = f"{lab_}: 90% lower"
    Q[f"t5_{k_}_hi"] = fmt(hi_, 1, True); CLAIM[f"t5_{k_}_hi"] = f"{lab_}: 90% upper"
SLOPES = pd.DataFrame(SLOPES)
for _, r_ in SLOPES.iterrows():
    qput(f"t5_{r_.k}_p3", max(r_.p3, 0.001), 3, f"{r_.lab}: one-sided p against +3 in (floor 0.001)")
qput("t5_max_hi", SLOPES.hi.max(), 1, "highest upper end in the slope table")
qput("t5_max_p3", SLOPES.p3.max(), 3, "largest one-sided p against +3 in the slope table")
print(SLOPES.round(3).to_string())
tick("R5 selection and slopes")

# (c) permutation tests on the day-3 list (season-1 SOE shuffled within each season fold, all 53 pairs; thresholds and
# the pick-and-routes list are fixed because they do not use season-1 SOE of the held-out fold)
d3m = PAIRS.day3.to_numpy()
obs_gap = int(PAIRS[PAIRS.day3 & PAIRS.rule].hit.sum()) - int(PAIRS[PAIRS.day3 & PAIRS.rule_pk].hit.sum())


def loso_rho(df):
    f1 = np.empty(len(df))
    for s_ in (2023, 2024):
        t = (df.s1 == s_).to_numpy()
        b_, _ = fitp(df[~t], F1)
        f1[t] = predp(df[t], b_, F1)
    m = df.day3.to_numpy()
    return spearmanr(f1[m], df.y2.to_numpy()[m])[0] - spearmanr(df.f_m0.to_numpy()[m], df.y2.to_numpy()[m])[0]


obs_drho = loso_rho(PAIRS)
assert abs(obs_drho - (EVAL["d3"]["mt"]["m1"]["rho"] - EVAL["d3"]["mt"]["m0"]["rho"])) < 1e-9
pg_, pr_ = [], []
for _ in range(NPERM):
    PP = PAIRS.copy()
    for s_ in (2023, 2024):
        ix_ = np.where(PP.s1.to_numpy() == s_)[0]
        PP.iloc[ix_, PP.columns.get_loc("y1s")] = PP.y1s.to_numpy()[RNG.permutation(ix_)]
    rule_ = np.zeros(len(PP), bool)
    for s_ in (2023, 2024):
        t = PP.s1.to_numpy() == s_
        rule_[t] = PP.y1s.to_numpy()[t] >= THR[s_]
    sub_ = PP[d3m]
    r3_ = rule_[d3m]
    hits_rule = int(sub_.hit.to_numpy()[r3_].sum())
    # pick-and-routes list of the same size per fold
    hits_pk = 0
    for s_ in (2023, 2024):
        t = sub_.s1.to_numpy() == s_
        k_ = int(r3_[t].sum())
        hits_pk += int(sub_[t].sort_values("f_m0", ascending=False).hit.iloc[:k_].sum())
    pg_.append(hits_rule - hits_pk)
    pr_.append(loso_rho(PP))
qput("perm_gap_obs", obs_gap, 0, "observed hit gap, rule minus pick and routes (count)")
qput("perm_gap_p", (1 + (np.array(pg_) >= obs_gap).sum()) / (1 + NPERM), 2, "permutation p of the yes/no gap")
qput("perm_rho_p", (1 + (np.array(pr_) >= obs_drho).sum()) / (1 + NPERM), 2, "permutation p of the rank-correlation gain")

# receiver random-intercept model of persistence (all 53 pairs; 4+ receivers appear twice)
PM = PAIRS.assign(y2i=IN_YD * PAIRS.y2, y1i=IN_YD * PAIRS.y1s)
pm_ = smf.mixedlm("y2i ~ logpick + te + lr1 + y1i", PM, groups=PM.pid).fit(reml=True)
qput("mmp_b", float(pm_.params["y1i"]), 2, "persistence: inches of next SOE per inch of shrunk SOE, mixed model")
qput("mmp_lo", float(pm_.params["y1i"] - 1.645 * pm_.bse["y1i"]), 2, "lower")
qput("mmp_hi", float(pm_.params["y1i"] + 1.645 * pm_.bse["y1i"]), 2, "upper")
pm3_ = smf.mixedlm("y2i ~ logpick + te + lr1 + y1i", PM[PM.day3], groups=PM[PM.day3].pid).fit(reml=True)
qput("mmp3_b", float(pm3_.params["y1i"]), 2, "day-3 persistence, mixed model")
qput("mmp3_lo", float(pm3_.params["y1i"] - 1.645 * pm3_.bse["y1i"]), 2, "lower")
qput("mmp3_hi", float(pm3_.params["y1i"] + 1.645 * pm3_.bse["y1i"]), 2, "upper")

# calibration of the held-out probabilities (all 53 pairs; bins fixed at 0.4 and 0.6)
CAL = []
for lo_, hi_, lab_ in ((0, 0.4, "lo"), (0.4, 0.6, "mid"), (0.6, 1.01, "hi")):
    m = (PAIRS.p_m1 >= lo_) & (PAIRS.p_m1 < hi_)
    CAL.append(dict(k=lab_, n=int(m.sum()), pred=100 * PAIRS.p_m1[m].mean(), obs=100 * PAIRS.hit[m].mean(),
                    hit=int(PAIRS.hit[m].sum())))
    qput(f"cal_{lab_}_n", int(m.sum()), 0, f"pairs with held-out P in bin {lab_}")
    qput(f"cal_{lab_}_pred", 100 * PAIRS.p_m1[m].mean(), 0, f"mean forecast %, bin {lab_}")
    qput(f"cal_{lab_}_hit", int(PAIRS.hit[m].sum()), 0, f"hits, bin {lab_}")
CAL = pd.DataFrame(CAL)
lp_ = np.log(PAIRS.p_m1.clip(0.01, 0.99) / (1 - PAIRS.p_m1.clip(0.01, 0.99)))
cs_ = sm.Logit(PAIRS.hit.astype(float).to_numpy(), sm.add_constant(lp_.to_numpy())).fit(disp=0)
qput("cal_slope", float(cs_.params[1]), 2, "calibration slope of held-out probabilities (1 = calibrated)")
qput("cal_slope_lo", float(cs_.params[1] - 1.645 * cs_.bse[1]), 2, "lower")
qput("cal_slope_hi", float(cs_.params[1] + 1.645 * cs_.bse[1]), 2, "upper")
CAL3 = PAIRS[PAIRS.day3]
qput("cal_d3_pred", 100 * CAL3.p_m1.mean(), 0, "day 3: mean held-out forecast %")
qput("cal_d3_obs", 100 * CAL3.hit.mean(), 0, "day 3: observed %")
tick("R5 permutation, mixed, calibration")

# (d) 2026 list context and the worked decision case. Pool: every 2025 day-3 / UDFA WR/TE with 100+ routes (any SOE),
# probabilities from M1 and M0 fitted on all 53 pairs, averaged over receiver-clustered refits.
B26 = ss[(ss.season == 2025) & (ss.routes >= 100)].join(pl_r, on="nfl_id")
B26 = B26[B26.draft_round.isna() | (B26.draft_round >= 4)].copy()
B26["logpick"] = np.log(pk_all.reindex(B26.nfl_id).fillna(262).to_numpy(float))
B26["te"] = (B26.position_group == "TE").astype(float).to_numpy()
B26["lr1"] = np.log(B26.routes.to_numpy(float))
B26["y1s"], _ = shrink(B26.ssoe, B26.routes)
F26 = {"m0": [], "m1": []}
PB26 = {"m0": [], "m1": []}
for bi_ in range(NBOOT // 2):
    bb_ = PAIRS.iloc[np.concatenate([gA[i] for i in RNG.integers(0, len(gA), len(gA))])] if bi_ else PAIRS
    for m_, cols in (("m0", F0), ("m1", F1)):
        b_, sd_ = fitp(bb_, cols)
        F26[m_].append(predp(B26, b_, cols))
        PB26[m_].append(_norm.cdf(predp(B26, b_, cols) / sd_))
for m_ in F26:
    B26[f"f_{m_}"] = IN_YD * np.mean(F26[m_], axis=0)
    B26[f"p_{m_}"] = 100 * np.mean(PB26[m_], axis=0)
B26["BG"] = cplayer.BG.reindex(B26.nfl_id).to_numpy()
car_ = pd.read_parquet(P / "player_career.parquet").set_index("nfl_id")
B26["apy"] = car_.contract_apy_m.reindex(B26.nfl_id).to_numpy()
B26["pick"] = pk_all.reindex(B26.nfl_id).to_numpy()
qput("b26_n", len(B26), 0, "2025 day-3/UDFA WR/TE with 100+ routes")
# worked case: same position group, picks within 0.25 log units (or both UDFA), routes within 25%; largest gap in P(SOE>0)
# from the SOE forecast among pairs whose pick-and-routes probabilities differ by under 3 points
best_, bgap_ = None, -1
for i_ in range(len(B26)):
    for j_ in range(i_ + 1, len(B26)):
        a_, b_ = B26.iloc[i_], B26.iloc[j_]
        if a_.position_group != b_.position_group or a_.position_group != "WR":
            continue
        if abs(a_.logpick - b_.logpick) > 0.25 or abs(np.log(a_.routes / b_.routes)) > np.log(1.25):
            continue
        if abs(a_.p_m0 - b_.p_m0) >= 3:
            continue
        gap_ = abs(a_.p_m1 - b_.p_m1)
        if gap_ > bgap_:
            best_, bgap_ = (i_, j_), gap_
wa_, wb_ = B26.iloc[best_[0]], B26.iloc[best_[1]]
if wa_.p_m1 < wb_.p_m1:
    wa_, wb_ = wb_, wa_
WC = {}
for tg_, r_ in (("a", wa_), ("b", wb_)):
    WC[tg_] = r_.display_name
    qput(f"wc_{tg_}_pick", 262 if pd.isna(r_.pick) else int(r_.pick), 0, f"{r_.display_name}: pick (262 = UDFA)")
    qput(f"wc_{tg_}_routes", int(r_.routes), 0, f"{r_.display_name}: 2025 routes")
    Q[f"wc_{tg_}_soe"] = fmt(IN_YD * r_.ssoe, 1, True); CLAIM[f"wc_{tg_}_soe"] = f"{r_.display_name}: 2025 SOE"
    qput(f"wc_{tg_}_p1", r_.p_m1, 0, f"{r_.display_name}: P(2026 SOE > 0), SOE forecast")
    qput(f"wc_{tg_}_p0", r_.p_m0, 0, f"{r_.display_name}: P(2026 SOE > 0), pick and routes")
    qput(f"wc_{tg_}_games", int(r_.games), 0, f"{r_.display_name}: 2025 games")
    qput(f"wc_{tg_}_apy", r_.apy, 2, f"{r_.display_name}: contract APY, $M")
    Q[f"wc_{tg_}_f1"] = fmt(r_.f_m1, 1, True); CLAIM[f"wc_{tg_}_f1"] = f"{r_.display_name}: 2026 SOE forecast, in"
    if not pd.isna(r_.BG):
        Q[f"wc_{tg_}_bg"] = fmt(r_.BG, 1, True); CLAIM[f"wc_{tg_}_bg"] = f"{r_.display_name}: Break G sd"
    WC[tg_ + "_align"] = r_.main_alignment
    WC[tg_ + "_team"] = r_.team
dsoe_ = wa_.f_m1 - wb_.f_m1
qput("wc_dsoe", dsoe_, 1, "forecast 2026 SOE gap between the two, in")
qput("wc_dcatch", dsoe_ * catch_per_in, 1, "catches a season the gap is worth")
qput("wc_dpct", wa_.p_m1 - wb_.p_m1, 0, "gap in P(2026 SOE > 0), points")
print("worked case:", WC, wa_[["p_m0", "p_m1", "f_m1", "apy", "games"]].to_dict(), wb_[["p_m0", "p_m1", "f_m1", "apy", "games"]].to_dict())
# context columns for the 13-name list
L26["align"], L26["team25"] = L26.main_alignment, L26.team
for _, r_ in L26.iterrows():
    tg_ = r_.display_name.split()[-1].lower().replace("'", "").replace(".", "")
    qput(f"l26_{tg_}_games", int(r_.games), 0, f"{r_.display_name}: 2025 games")
qput("l26_short_games", int((L26.games <= 13).sum()), 0, "list names who played 13 or fewer games in 2025")
pd.DataFrame(SL26).assign(games_2025=L26.games.astype(int).to_numpy(), align_2025=L26["align"].to_numpy(),
                          team_2025=L26.team25.to_numpy()).to_csv(OUT / "shortlist_2026.csv", index=False)
print(L26[["display_name", "games", "align", "team25", "p_m1"]].to_string())
tick("R5 context and case")

# ============================================================================= round 15 revision additions (R6)
# (a) nested test of the day-3 claim. All 53 season pairs pooled. Receiver random-intercept model with a pick x
#     shrunk-SOE interaction; leave-one-RECEIVER-out forecasts (a receiver's pairs are never in his own training set);
#     gain in rank correlation on day-3 pairs and on all pairs; 90% interval from receiver-clustered resamples that redo
#     the whole leave-one-receiver-out loop; null band from within-season shuffles of last season's SOE.
LPM = float(PAIRS.logpick.mean())
PAIRS["lpy"] = (PAIRS.logpick - LPM) * PAIRS.y1s
PMX = PAIRS.assign(y2i=IN_YD * PAIRS.y2, y1i=IN_YD * PAIRS.y1s, lpc=PAIRS.logpick - LPM)
pmx_ = smf.mixedlm("y2i ~ te + lr1 + lpc * y1i", PMX, groups=PMX.pid).fit(reml=True)
ib_, ise_ = float(pmx_.params["lpc:y1i"]), float(pmx_.bse["lpc:y1i"])
qput("nx_int", ib_, 2, "pick x SOE interaction: change in persistence per log unit of pick (mixed model)")
qput("nx_int_lo", ib_ - 1.645 * ise_, 2, "lower"); qput("nx_int_hi", ib_ + 1.645 * ise_, 2, "upper")
cvx_ = pmx_.cov_params()
for pk_, tg_ in ((40, "p40"), (200, "p200")):
    cvec = np.zeros(len(pmx_.params)); nm_ = list(pmx_.params.index)
    cvec[nm_.index("y1i")] = 1.0; cvec[nm_.index("lpc:y1i")] = np.log(pk_) - LPM
    est_ = float(cvec @ pmx_.params.to_numpy()); se_ = float(np.sqrt(cvec @ cvx_.to_numpy() @ cvec))
    qput(f"nx_{tg_}", est_, 2, f"persistence at pick {pk_}: inches of next SOE per inch of shrunk SOE")
    qput(f"nx_{tg_}_lo", est_ - 1.645 * se_, 2, "lower"); qput(f"nx_{tg_}_hi", est_ + 1.645 * se_, 2, "upper")
qput("nx_np", PAIRS.pid.nunique(), 0, "receivers in the 53 pairs")
LORO_M = {"m0": F0, "m1": F1, "m1x": F1 + ["lpy"]}


def loro_fc(df):
    pids_ = df.pid.to_numpy()
    y_ = df.y2.to_numpy(float)
    A_ = {k: np.column_stack([np.ones(len(df)), df[c].to_numpy(float)]) for k, c in LORO_M.items()}
    out = {k: np.empty(len(df)) for k in LORO_M}
    for p_ in np.unique(pids_):
        t_ = pids_ == p_
        for k in LORO_M:
            out[k][t_] = A_[k][t_] @ np.linalg.lstsq(A_[k][~t_], y_[~t_], rcond=None)[0]
    return out


def loro_gain(df):
    fc_ = loro_fc(df)
    res = []
    for msk in (df.day3.to_numpy(), np.ones(len(df), bool)):
        y_ = df.y2.to_numpy()[msk]
        r0 = spearmanr(fc_["m0"][msk], y_)[0]
        res += [spearmanr(fc_["m1x"][msk], y_)[0] - r0, spearmanr(fc_["m1"][msk], y_)[0] - r0, r0,
                spearmanr(fc_["m1x"][msk], y_)[0]]
    for msk in (df.day3.to_numpy(), np.ones(len(df), bool)):
        y_ = df.y2.to_numpy()[msk]
        res.append(IN_YD * (np.sqrt(np.mean((fc_["m0"][msk] - y_) ** 2)) - np.sqrt(np.mean((fc_["m1x"][msk] - y_) ** 2))))
    return np.array(res)  # d3: gain x, gain plain, rho0, rho1x; all: same (rmse gains appended below)


LG = loro_gain(PAIRS)
lg_bs = []
for _ in range(NBOOT):
    bb_ = PAIRS.iloc[np.concatenate([gA[i] for i in RNG.integers(0, len(gA), len(gA))])]
    if bb_.day3.sum() < 5:
        continue
    lg_bs.append(loro_gain(bb_))
lg_bs = np.array(lg_bs)
lg_null = []
for _ in range(NPERM):
    PP = PAIRS.copy()
    for s_ in (2023, 2024):
        ix_ = np.where(PP.s1.to_numpy() == s_)[0]
        PP.iloc[ix_, PP.columns.get_loc("y1s")] = PP.y1s.to_numpy()[RNG.permutation(ix_)]
    PP["lpy"] = (PP.logpick - LPM) * PP.y1s
    lg_null.append(loro_gain(PP))
lg_null = np.array(lg_null)
for j_, tg_ in ((0, "d3"), (4, "all")):
    qput(f"lo_{tg_}_gain", LG[j_], 2, f"{tg_}: leave-one-receiver-out rank-correlation gain, SOE x pick model")
    qput(f"lo_{tg_}_lo", pct(lg_bs[:, j_], 5), 2, "lower, receiver-clustered")
    qput(f"lo_{tg_}_hi", pct(lg_bs[:, j_], 95), 2, "upper")
    p_lo_ = (1 + (lg_null[:, j_] >= LG[j_]).sum()) / (1 + NPERM)
    qput(f"lo_{tg_}_p", p_lo_, 2 if p_lo_ >= 0.01 else 4, "within-season shuffle p")
    qput(f"lo_{tg_}_gplain", LG[j_ + 1], 2, f"{tg_}: same gain without the interaction")
    qput(f"lo_{tg_}_r0", LG[j_ + 2], 2, f"{tg_}: pick and routes rank correlation, leave one receiver out")
    qput(f"lo_{tg_}_r1", LG[j_ + 3], 2, f"{tg_}: with last season's SOE, leave one receiver out")
    qput(f"lo_{tg_}_pos", 100 * (lg_bs[:, j_] > 0).mean(), 0, f"{tg_}: % of resamples with a positive gain")
for j_, tg_ in ((8, "d3"), (9, "all")):
    qput(f"lo_{tg_}_drmse", LG[j_], 1, f"{tg_}: leave-one-receiver-out RMSE removed, inches")
    qput(f"lo_{tg_}_drmse_lo", pct(lg_bs[:, j_], 5), 1, "lower"); qput(f"lo_{tg_}_drmse_hi", pct(lg_bs[:, j_], 95), 1, "upper")
LO_NULL = {tg_: (pct(lg_null[:, j_], 5), pct(lg_null[:, j_], 95)) for j_, tg_ in ((0, "d3"), (4, "all"))}
LO_CI = {tg_: (LG[j_], pct(lg_bs[:, j_], 5), pct(lg_bs[:, j_], 95)) for j_, tg_ in ((0, "d3"), (4, "all"))}
SF_CI = {tg_: (EVAL[tg_]["mt"]["m1"]["rho"] - EVAL[tg_]["mt"]["m0"]["rho"], pct(EVAL[tg_]["bs"][:, 0], 5),
               pct(EVAL[tg_]["bs"][:, 0], 95)) for tg_ in ("d3", "all")}
tick("R6a nested persistence")


# (b) family-matched errors-in-variables model, receivers with 8+ NFL breaks. Cell = receiver x route family with at
#     least one drill rep and one NFL snap of that family. Observed means carry known noise (pooled within-cell rep / snap
#     variance over n reps / snaps); true drill skill xi ~ N(m_f, s_f^2); true NFL skill eta ~ N(a_f + b_f xi, t_f^2);
#     family slopes b_f ~ N(b0, w^2) (random slopes). Gibbs sampler, all conditionals conjugate. True link per family =
#     correlation of xi and eta = b s / sqrt(b^2 s^2 + t^2); pooled link uses b0 and the mean family variances.
def wvar(df, col):
    grp_ = df.groupby("nfl_id")[col]
    n_, v_ = grp_.size(), grp_.var(ddof=1)
    ok_ = n_ >= 2
    return float(((n_[ok_] - 1) * v_[ok_]).sum() / (n_[ok_] - 1).sum())


P8ID = set(P8.index)
EC = []
SD2, SG2 = np.zeros(len(FAMILIES)), np.zeros(len(FAMILIES))
for fi_, f_ in enumerate(FAMILIES):
    dr_, nf_ = cb[cb.family == f_], g[g.route_ran == f_]
    SD2[fi_], SG2[fi_] = wvar(dr_, "bg_z"), wvar(nf_, "bg_gz")
    xd_ = dr_[dr_.nfl_id.isin(P8ID)].groupby("nfl_id").bg_z.agg(["mean", "size"])
    yg_ = nf_[nf_.nfl_id.isin(P8ID)].groupby("nfl_id").bg_gz.agg(["mean", "size"])
    j_ = xd_.join(yg_, lsuffix="_d", rsuffix="_g", how="inner")
    EC.append(pd.DataFrame(dict(nfl_id=j_.index, f=fi_, xm=j_.mean_d, nd=j_.size_d, ym=j_.mean_g, ng=j_.size_g)))
EC = pd.concat(EC, ignore_index=True)


def eiv_gibbs(ec, n_iter=8000, burn=2000, seed=11):
    rg_ = np.random.default_rng(seed)
    F_ = len(FAMILIES)
    fi = ec.f.to_numpy(int)
    xm, ym = ec.xm.to_numpy(float), ec.ym.to_numpy(float)
    vx, vy = SD2[fi] / ec.nd.to_numpy(float), SG2[fi] / ec.ng.to_numpy(float)
    N_ = len(ec)
    eta = ym.copy()
    al, be, t2, m, s2 = np.zeros(F_), np.zeros(F_), np.full(F_, 0.5), np.zeros(F_), np.full(F_, 0.5)
    b0, w2 = 0.0, 0.05
    keep = []
    for it_ in range(n_iter):
        Pr = 1 / s2[fi] + 1 / vx + be[fi] ** 2 / t2[fi]
        xi = (m[fi] / s2[fi] + xm / vx + be[fi] * (eta - al[fi]) / t2[fi]) / Pr + rg_.standard_normal(N_) / np.sqrt(Pr)
        Pr = 1 / vy + 1 / t2[fi]
        eta = (ym / vy + (al[fi] + be[fi] * xi) / t2[fi]) / Pr + rg_.standard_normal(N_) / np.sqrt(Pr)
        for f in range(F_):
            t_ = fi == f
            n_ = int(t_.sum())
            A_ = np.column_stack([np.ones(n_), xi[t_]])
            pp_ = np.diag([1.0, 1 / w2])
            cov_ = np.linalg.inv(A_.T @ A_ / t2[f] + pp_)
            mu_ = cov_ @ (A_.T @ eta[t_] / t2[f] + pp_ @ np.array([0.0, b0]))
            al[f], be[f] = rg_.multivariate_normal(mu_, cov_)
            r_ = eta[t_] - al[f] - be[f] * xi[t_]
            t2[f] = 1 / rg_.gamma(1 + n_ / 2, 1 / (0.5 + r_ @ r_ / 2))
            pm_ = n_ / s2[f] + 1
            m[f] = rg_.normal(xi[t_].sum() / s2[f] / pm_, 1 / np.sqrt(pm_))
            rr_ = xi[t_] - m[f]
            s2[f] = 1 / rg_.gamma(1 + n_ / 2, 1 / (0.5 + rr_ @ rr_ / 2))
        pb_ = 1 + F_ / w2
        b0 = rg_.normal(be.sum() / w2 / pb_, 1 / np.sqrt(pb_))
        w2 = 1 / rg_.gamma(2 + F_ / 2, 1 / (0.05 + ((be - b0) ** 2).sum() / 2))
        if it_ >= burn:
            rf_ = be * np.sqrt(s2) / np.sqrt(be ** 2 * s2 + t2)
            sb_m, tb_m = np.sqrt(s2.mean()), np.sqrt(t2.mean())
            keep.append(np.r_[rf_, b0 * sb_m / np.sqrt(b0 ** 2 * sb_m ** 2 + tb_m ** 2)])
    return np.array(keep)


EIVF = eiv_gibbs(EC)
FAMPOST = []
for k_, lab_ in enumerate([f.capitalize() for f in FAMILIES] + ["All five (pooled)"]):
    d_ = EIVF[:, k_]
    tg_ = (FAMILIES[k_].lower() if k_ < len(FAMILIES) else "pool")
    nc_ = int((EC.f == k_).sum()) if k_ < len(FAMILIES) else int(EC.nfl_id.nunique())
    FAMPOST.append(dict(k=tg_, lab=lab_, med=np.median(d_), lo=pct(d_, 5), hi=pct(d_, 95), p02=100 * (d_ > 0.2).mean(),
                        n=nc_))
    qput(f"fp_{tg_}_med", np.median(d_), 2, f"{lab_}: posterior median true drill-to-NFL correlation")
    qput(f"fp_{tg_}_lo", pct(d_, 5), 2, "5%"); qput(f"fp_{tg_}_hi", pct(d_, 95), 2, "95%")
    qput(f"fp_{tg_}_p02", 100 * (d_ > 0.2).mean(), 0, f"{lab_}: posterior % that the true link exceeds 0.2")
    qput(f"fp_{tg_}_n", nc_, 0, f"{lab_}: receivers (cells)")
FAMPOST = pd.DataFrame(FAMPOST)
qput("fp_cells", len(EC), 0, "receiver x family cells")
qput("fp_ng_med", int(EC.ng.median()), 0, "median NFL breaks per cell")
qput("fp_max_p02", FAMPOST.p02.iloc[:-1].max(), 0, "highest family posterior % above 0.2")
print(FAMPOST.round(3).to_string())
tick("R6b family EIV")

# (c) recalibration (Platt) of the probabilities, fitted on the 53 held-out season-pair forecasts; cross-fitted check
def logit_(p):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.log(p / (1 - p))


PLATT = {}
for m_ in ("m0", "m1"):
    PLATT[m_] = sm.Logit(PAIRS.hit.astype(float).to_numpy(), sm.add_constant(logit_(PAIRS[f"p_{m_}"]))).fit(disp=0).params
cf_ = np.empty(len(PAIRS))
for s_ in (2023, 2024):
    t_ = (PAIRS.s1 == s_).to_numpy()
    pa_ = sm.Logit(PAIRS.hit.astype(float).to_numpy()[~t_], sm.add_constant(logit_(PAIRS.p_m1[~t_]))).fit(disp=0).params
    cf_[t_] = 1 / (1 + np.exp(-(pa_[0] + pa_[1] * logit_(PAIRS.p_m1[t_]))))
PAIRS["p_cal_cf"] = cf_
d3_ = PAIRS.day3.to_numpy()
qput("rc_d3_pred", 100 * cf_[d3_].mean(), 0, "day 3: mean cross-fitted recalibrated forecast %")
qput("rc_brier_raw", np.mean((PAIRS.p_m1 - PAIRS.hit) ** 2), 3, "all pairs: Brier, raw held-out")
qput("rc_brier_cf", np.mean((cf_ - PAIRS.hit) ** 2), 3, "all pairs: Brier, cross-fitted recalibrated")
qput("rc_a", PLATT["m1"][0], 2, "Platt intercept"); qput("rc_b", PLATT["m1"][1], 2, "Platt slope")


def platt(p100, m_):
    return 100 / (1 + np.exp(-(PLATT[m_][0] + PLATT[m_][1] * logit_(np.asarray(p100, float) / 100))))


for df_ in (L26, B26):
    df_["pc_m1"], df_["pc_m0"] = platt(df_.p_m1, "m1"), platt(df_.p_m0, "m0")
for _, r_ in L26.iterrows():
    tg_ = r_.display_name.split()[-1].lower().replace("'", "").replace(".", "")
    qput(f"l26_{tg_}_pc1", r_.pc_m1, 0, f"{r_.display_name}: recalibrated P(2026 SOE > 0), SOE forecast")
    qput(f"l26_{tg_}_pc0", r_.pc_m0, 0, f"{r_.display_name}: recalibrated P(2026 SOE > 0), pick and routes")
qput("l26_pc_drop", float((L26.p_m1 - L26.pc_m1).mean()), 0, "mean drop from raw to recalibrated, points")
# calibration bins with Jeffreys intervals (for Fig 4C)
from scipy.stats import beta as _beta  # noqa: E402

CAL["lo"] = [100 * _beta.ppf(0.05, h + 0.5, n - h + 0.5) for h, n in zip(CAL.hit, CAL.n)]
CAL["hi"] = [100 * _beta.ppf(0.95, h + 0.5, n - h + 0.5) for h, n in zip(CAL.hit, CAL.n)]
tick("R6c recalibration")

# (d) catch conversion with bounds, and what 1 sd of Break G is in g
cr_lo_, cr_hi_ = bcr[-1] - 1.645 * secr[-1], bcr[-1] + 1.645 * secr[-1]
tg_q = ssa[ssa.routes >= 200].targets
TG_LO, TG_HI = int(tg_q.quantile(0.25)), int(tg_q.quantile(0.75))
qput("cr_lo", 100 * cr_lo_, 1, "catch-rate points per yard, 90% lower"); qput("cr_hi", 100 * cr_hi_, 1, "upper")
qput("tg_lo", TG_LO, 0, "targets, 25th percentile of 200+ route seasons"); qput("tg_hi", TG_HI, 0, "75th percentile")
qput("g_sd", float(NUM["g_true_sd"]), 3, "1 sd of Break G = sd of true receiver Break G, in g")


def catches(d_in):
    return d_in * bcr[-1] / IN_YD * TG_SEASON, d_in * cr_lo_ / IN_YD * TG_LO, d_in * cr_hi_ / IN_YD * TG_HI


qput("sesoi_c_lo", catches(SESOI_IN)[1], 2, "+3 in: catches, low bound"); qput("sesoi_c_hi", catches(SESOI_IN)[2], 2, "high")
qput("best_c", catches(pct(COMBO["main"]["bs"][:, 1], 95))[0], 2, "best case of the primary: catches a season")

# (e) the worked contrast: hard vs soft drill cutter at the same pick (Boutte 187, Washington 184), chosen by drill and
#     pick, not by forecast. Scheme context: full-season man share, main alignment, most frequent passer on sampled
#     2025 snaps, games. And how unusual the earlier rule-chosen pair was: P gaps over every matched 2025 pair.
B26i = B26.set_index("display_name")
qb25 = gq[gq.season == 2025].groupby("nfl_id").qb.agg(lambda s: s.value_counts().index[0])
WK = {}
for nm_, tg_ in (("Kayshon Boutte", "kb"), ("Malik Washington", "mw")):
    r_ = B26i.loc[nm_]
    WK[tg_] = dict(name=nm_, id=int(r_.nfl_id), qb=qb25.get(r_.nfl_id), man=100 * r_.plays_vs_man / r_.routes)
    qput(f"wk_{tg_}_pick", int(r_.pick), 0, f"{nm_}: pick"); qput(f"wk_{tg_}_routes", int(r_.routes), 0, "2025 routes")
    qput(f"wk_{tg_}_games", int(r_.games), 0, "2025 games"); qput(f"wk_{tg_}_man", WK[tg_]["man"], 0, "% routes vs man")
    qput(f"wk_{tg_}_pc1", r_.pc_m1, 0, "recalibrated P, SOE forecast"); qput(f"wk_{tg_}_pc0", r_.pc_m0, 0, "pick+routes")
    qput(f"wk_{tg_}_apy", r_.apy, 2, f"{nm_}: contract APY, $M")
    Q[f"wk_{tg_}_f1"] = fmt(r_.f_m1, 1, True); CLAIM[f"wk_{tg_}_f1"] = f"{nm_}: 2026 SOE forecast, in"
gap_ = B26i.loc["Malik Washington"].f_m1 - B26i.loc["Kayshon Boutte"].f_m1
cc_ = catches(gap_)
qput("wk_dsoe", gap_, 1, "forecast gap, in"); qput("wk_dc", cc_[0], 1, "catches a season, central")
qput("wk_dc_lo", cc_[1], 1, "low bound"); qput("wk_dc_hi", cc_[2], 1, "high bound")
print("worked contrast:", WK)
MGAP = []
for i_ in range(len(B26)):
    for j_ in range(i_ + 1, len(B26)):
        a_, b_ = B26.iloc[i_], B26.iloc[j_]
        if a_.position_group != "WR" or b_.position_group != "WR":
            continue
        if abs(a_.logpick - b_.logpick) > 0.25 or abs(np.log(a_.routes / b_.routes)) > np.log(1.25):
            continue
        if abs(a_.p_m0 - b_.p_m0) >= 3:
            continue
        MGAP.append(abs(a_.pc_m1 - b_.pc_m1))
qput("mp_n", len(MGAP), 0, "matched 2025 day-3 WR pairs"); qput("mp_med", float(np.median(MGAP)), 0, "median gap, pts")
qput("mp_max", float(np.max(MGAP)), 0, "largest gap, points (recalibrated)")
# Combine-discounted mark on the 2026 list: NGS athleticism score below his class-and-position average
ngs_z = pb.ngs_athleticism_score_s
L26["disc"] = ngs_z.reindex(L26.nfl_id).to_numpy() < 0
L26["man"] = 100 * L26.plays_vs_man.to_numpy(float) / L26.routes.to_numpy(float)
L26["qb"] = qb25.reindex(L26.nfl_id).to_numpy()
qput("l26_disc", int(L26.disc.sum()), 0, "Combine-discounted names on the 2026 list")
for _, r_ in L26.iterrows():
    tg_ = r_.display_name.split()[-1].lower().replace("'", "").replace(".", "")
    qput(f"l26_{tg_}_man", r_.man, 0, f"{r_.display_name}: % of 2025 routes against man")
pd.DataFrame(SL26).assign(games_2025=L26.games.astype(int).to_numpy(), align_2025=L26["align"].to_numpy(),
                          team_2025=L26.team25.to_numpy(), man_pct_2025=L26.man.round(0).to_numpy(),
                          qb_2025=L26.qb.to_numpy(), combine_discounted=L26.disc.to_numpy(),
                          p_recal=L26.pc_m1.round(0).to_numpy(), p0_recal=L26.pc_m0.round(0).to_numpy()
                          ).to_csv(OUT / "shortlist_2026.csv", index=False)
print(L26[["display_name", "pc_m1", "pc_m0", "man", "qb", "disc"]].round(1).to_string())

# (f) film use: Boutte vs Washington on the route family both ran most on sampled NFL snaps; entry speed, speed at the
#     apex (gear-down = entry minus apex), Break G; league medians for that family.
gk_, gw_ = g[g.nfl_id == WK["kb"]["id"]], g[g.nfl_id == WK["mw"]["id"]]
fam_cnt = (gk_.route_ran.value_counts().reindex(FAMILIES).fillna(0)
           .combine(gw_.route_ran.value_counts().reindex(FAMILIES).fillna(0), min))
FILM_FAM = fam_cnt.idxmax()
gl_ = g[g.route_ran == FILM_FAM]
FILM = {}
for tg_, df_ in (("kb", gk_[gk_.route_ran == FILM_FAM]), ("mw", gw_[gw_.route_ran == FILM_FAM]), ("lg", gl_)):
    FILM[tg_] = dict(n=len(df_), vin=df_.v_in.median(), vap=df_.v_apex.median(), gd=(df_.v_in - df_.v_apex).median(),
                     bg=df_.bg.median())
    qput(f"film_{tg_}_n", len(df_), 0, f"{tg_}: NFL {FILM_FAM} breaks")
    qput(f"film_{tg_}_vin", FILM[tg_]["vin"], 1, f"{tg_}: median entry speed, yd/s")
    qput(f"film_{tg_}_vap", FILM[tg_]["vap"], 1, f"{tg_}: median speed at the top of the break, yd/s")
    qput(f"film_{tg_}_gd", FILM[tg_]["gd"], 1, f"{tg_}: median gear-down (entry minus apex speed), yd/s")
    qput(f"film_{tg_}_bg", FILM[tg_]["bg"], 2, f"{tg_}: median Break G, g")
NUM["film_fam"] = FILM_FAM.lower()
print("film family", FILM_FAM, FILM)
qput("price_full", float(NUM["price_ho"]), 0, "price beyond the 40, weight and NGS score (held out; earlier rounds)")
tick("R6 done")

# ============================================================================= round 16 additions (judges' round-15 feedback)
# (a) one estimand per claim. Day-3 tiebreaker: the PRIMARY is the season-fold gain (fit on one season pair, forecast the
#     other: the way the rule is used); leave-one-receiver-out is a sensitivity row. Drill-to-NFL: the raw observed rho
#     (all receivers with a drill score and NFL breaks) and the noise-corrected true correlation (8+ NFL breaks) are two
#     estimands, both reported in one appendix table.
# (b) a pre-specified contrast instead of a hand-picked pair: every within-season pair of day-3 receivers; matched =
#     pick-and-routes forecasts closer than the median gap. Share of contrasts where the higher shrunk-SOE receiver had
#     the higher next-season SOE, receiver-clustered bootstrap.
# (c) pre-registration of the 2026 holdout (frozen_rule_2026.json with sha256) and power of the 2026 test.
# (d) shortlist usability: availability flag, act-on names, last list's hits and misses.
from scipy.stats import binom as _binom  # noqa: E402

FREEZE_DATE = "2026-10-24"
D3P = PAIRS[PAIRS.day3].reset_index(drop=True)
CT = []
for s_ in (2023, 2024):
    sub_ = D3P[D3P.s1 == s_]
    ix_ = sub_.index.to_numpy()
    for a_i in range(len(ix_)):
        for b_i in range(a_i + 1, len(ix_)):
            ra_, rb_ = D3P.loc[ix_[a_i]], D3P.loc[ix_[b_i]]
            if ra_.pid == rb_.pid or ra_.y2 == rb_.y2:
                continue
            CT.append(dict(a=ix_[a_i], b=ix_[b_i], pa=ra_.pid, pb=rb_.pid, dpk=abs(ra_.f_m0 - rb_.f_m0),
                           soe_ok=(ra_.y1s - rb_.y1s) * (ra_.y2 - rb_.y2) > 0,
                           pk_ok=(ra_.f_m0 - rb_.f_m0) * (ra_.y2 - rb_.y2) > 0))
CT = pd.DataFrame(CT)
CT_MED = float(CT.dpk.median())
CT["matched"] = CT.dpk < CT_MED


def ct_rates(ct):
    m_ = ct[ct.matched]
    return np.array([100 * m_.soe_ok.mean(), 100 * ct.soe_ok.mean(), 100 * ct.pk_ok.mean()])


CTR = ct_rates(CT)
pids_ct = np.unique(np.r_[CT.pa, CT.pb])
ct_bs = []
for _ in range(NBOOT):
    w_ = pd.Series(RNG.multinomial(len(pids_ct), np.full(len(pids_ct), 1 / len(pids_ct))), index=pids_ct)
    wt_ = w_.reindex(CT.pa).to_numpy() * w_.reindex(CT.pb).to_numpy()
    if wt_[CT.matched.to_numpy()].sum() == 0:
        continue
    mm_ = CT.matched.to_numpy()
    ct_bs.append([100 * np.average(CT.soe_ok[mm_], weights=wt_[mm_]), 100 * np.average(CT.soe_ok, weights=wt_),
                  100 * np.average(CT.pk_ok, weights=wt_)])
ct_bs = np.array(ct_bs)
qput("ct_n", int(CT.matched.sum()), 0, "matched within-season day-3 contrasts (pick-and-routes gap below median)")
qput("ct_all_n", len(CT), 0, "all within-season day-3 contrasts")
qput("ct_med", IN_YD * CT_MED, 1, "median pick-and-routes forecast gap used to define matched, inches")
qput("ct_m", CTR[0], 0, "% of matched contrasts where higher shrunk SOE had higher next SOE")
qput("ct_m_lo", pct(ct_bs[:, 0], 5), 0, "lower, receiver-clustered"); qput("ct_m_hi", pct(ct_bs[:, 0], 95), 0, "upper")
qput("ct_a", CTR[1], 0, "% of all contrasts ordered right by shrunk SOE")
qput("ct_a_lo", pct(ct_bs[:, 1], 5), 0, "lower"); qput("ct_a_hi", pct(ct_bs[:, 1], 95), 0, "upper")
qput("ct_pk", CTR[2], 0, "% of all contrasts ordered right by the held-out pick-and-routes forecast")
print("contrasts", len(CT), int(CT.matched.sum()), CTR, CT_MED * IN_YD)

# (a) reconciliation numbers already exist: ev_d3_* (season folds), lo_d3_* (leave one receiver out), pl_* (raw rho),
# fp_pool_* (true correlation). Here only the sd of each, for the appendix table.
qput("sf_d3_pos", 100 * (EVAL["d3"]["bs"][:, 0] > 0).mean(), 0, "% of resamples with a positive season-fold gain")
qput("pl_n8", len(P8), 0, "receivers with 8+ NFL breaks")

# (c) power of the 2026 test. Expected 2025->26 day-3 pairs = 2025 day-3 pool x historical rate of keeping 100+ routes.
d3ids_ = set(pl_r.index[pl_r.draft_round.isna() | (pl_r.draft_round >= 4)])
keep_ = []
for a_, b_ in ((2023, 2024), (2024, 2025)):
    s_a = ss[(ss.season == a_) & (ss.routes >= 100) & ss.nfl_id.isin(d3ids_)].nfl_id
    s_b = set(ss[(ss.season == b_) & (ss.routes >= 100)].nfl_id)
    keep_ += [i in s_b for i in s_a]
KEEP = float(np.mean(keep_))
N26 = int(round(len(B26) * KEEP))
se_old = float(np.std(EVAL["d3"]["bs"][:, 0], ddof=1))
se_new = se_old * np.sqrt(EVAL["d3"]["n"] / N26)
g_obs = EVAL["d3"]["mt"]["m1"]["rho"] - EVAL["d3"]["mt"]["m0"]["rho"]
qput("pw_keep", 100 * KEEP, 0, "% of day-3 100+ route receivers who log 100+ routes next season")
qput("pw_n26", N26, 0, "expected 2025->26 day-3 pairs")
qput("pw_g", norm.cdf(g_obs / se_new - 1.645), 2, "power of the 2026 rank-gain test if the true gain = the observed")
qput("pw_g02", norm.cdf(0.2 / se_new - 1.645), 2, "power if the true gain = 0.2")
qput("pw_pool", norm.cdf(g_obs / (se_old * np.sqrt(EVAL["d3"]["n"] / (EVAL["d3"]["n"] + N26))) - 1.645), 2,
     "power of the pooled 2023-26 day-3 test")
base_ = float(PAIRS[PAIRS.day3].hit.mean())
n_l = len(L26)
kc_ = next(k for k in range(n_l + 1) if _binom.sf(k - 1, n_l, base_) <= 0.05)
p_rule_ = float(PAIRS[PAIRS.day3 & PAIRS.rule].hit.mean())
qput("pw_hit_k", kc_, 0, "hits of 13 needed to beat the day-3 base rate at 5%")
qput("pw_hit", float(_binom.sf(kc_ - 1, n_l, p_rule_)), 2, "power of the 13-name hit-rate test at the observed rule rate")

# frozen 2026 rule: coefficients on all 53 pairs, threshold, recalibration, names and probabilities, targets
b_all_, sd_all_ = fitp(PAIRS, F1)
b0_all_, sd0_all_ = fitp(PAIRS, F0)
RULE26 = dict(frozen_on=FREEZE_DATE, data_through="2025 regular season (NGS full-season separation)",
              pool="2025 day-3 (round 4+) or undrafted WR/TE with 100+ routes; outcome = 2026 route-weighted SOE",
              forecast_m1=dict(cols=["1"] + F1, coef=[round(float(v), 6) for v in b_all_], resid_sd=round(sd_all_, 6)),
              forecast_m0=dict(cols=["1"] + F0, coef=[round(float(v), 6) for v in b0_all_], resid_sd=round(sd0_all_, 6)),
              shrinkage=dict(tau2=round(TAU2, 8), s2=round(S2, 8), mean=round(float(mu_), 6)),
              platt_m1=[round(float(v), 6) for v in PLATT["m1"]], threshold_in=round(IN_YD * T_ALL, 3),
              list=[dict(name=r_.display_name, nfl_id=int(r_.nfl_id), p_recal=round(float(r_.pc_m1), 1))
                    for _, r_ in L26.iterrows()],
              primary_test="day-3 2025->26 pairs with 100+ routes in both seasons: Spearman(M1 forecast, 2026 SOE) minus "
                           "Spearman(M0 forecast, 2026 SOE) > 0, receiver bootstrap 90% interval",
              secondary="hits (2026 SOE > 0) among the listed names vs the pick-and-routes list of the same size",
              expected_pairs=N26, power_at_observed_gain=float(Q["pw_g"]), power_at_gain_0_2=float(Q["pw_g02"]))
blob26 = json.dumps(RULE26, sort_keys=True).encode()
SHA26 = hashlib.sha256(blob26).hexdigest()
(OUT / "frozen_rule_2026.json").write_text(json.dumps(dict(rule=RULE26, sha256=SHA26), indent=1))
Q["freeze_date"] = "24 October 2026"; CLAIM["freeze_date"] = "date the 2026 rule was frozen"
Q["sha26"] = SHA26[:12]; CLAIM["sha26"] = "first 12 characters of the sha256 of the frozen 2026 rule"

# (d) shortlist usability: availability flag (13 or fewer games in 2025), act-on names (top four recalibrated
# probabilities among names who played 14+ games), and the rule's earlier lists with hits and misses.
L26["avail"] = L26.games > 13
act_ = L26[L26.avail].sort_values("pc_m1", ascending=False).head(4)
L26["act"] = L26.nfl_id.isin(act_.nfl_id)
qput("act_min", float(act_.pc_m1.min()), 0, "lowest recalibrated P among the act-on names")
qput("l26_flag", int((~L26.avail).sum()), 0, "list names flagged for availability (13 or fewer games)")
HIST = PAIRS[PAIRS.day3 & PAIRS.rule].sort_values(["s1", "y2"], ascending=[True, False])
HIST[["display_name", "s1", "hit"]].to_csv(OUT / "rule_history.csv", index=False)
for s_ in (2023, 2024):
    h_ = HIST[HIST.s1 == s_]
    qput(f"hist_{s_}_n", len(h_), 0, f"rule names, {s_} season list"); qput(f"hist_{s_}_hit", int(h_.hit.sum()), 0, "hits")
sl_ = pd.read_csv(OUT / "shortlist_2026.csv")
sl_["avail_flag"] = (~L26.avail).to_numpy()
sl_["act"] = L26.act.to_numpy()
sl_.to_csv(OUT / "shortlist_2026.csv", index=False)
print(L26[["display_name", "games", "pc_m1", "act"]].to_string())
print(HIST[["display_name", "s1", "hit"]].to_string())
tick("R16 additions")

# development checkpoint (off by default): BDB_CKPT=path saves the state here so the sections below can be re-run alone
if os.environ.get("BDB_CKPT"):
    import cloudpickle  # noqa: E402
    import types  # noqa: E402

    _st = {k: v for k, v in globals().items() if not k.startswith("__") and not isinstance(v, types.ModuleType)
           and not type(v).__module__.startswith(("statsmodels", "patsy"))}
    with open(os.environ["BDB_CKPT"], "wb") as _fh:
        cloudpickle.dump(_st, _fh)
# ##### CHECKPOINT #####

# ============================================================================= figures (round 14 revision)
FIGN: dict[str, str] = {}  # numbers printed only inside figures (not quoted in the text)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5, "xtick.labelsize": 10.5,
                     "ytick.labelsize": 10.5, "legend.fontsize": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False})
C_DR, C_NFL, C_REP = OI["blue"], OI["verm"], "#C9C9C9"


MPH = 2.0455  # yd/s -> mph


def oriented(x, y_, i):
    """Path from 1.0 s before to 1.0 s after frame i, shifted to the break and rotated to run upfield, break right."""
    lo, hi = max(0, i - 10), min(len(x), i + 11)
    p = np.column_stack([x[lo:hi] - x[i], y_[lo:hi] - y_[i]])
    a0 = max(0, i - 8)
    d = np.array([x[max(a0, i - 3)] - x[a0], y_[max(a0, i - 3)] - y_[a0]])
    if np.linalg.norm(d) < 1e-6:
        return None
    th = np.arctan2(d[0], d[1])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    qq = p @ R.T
    if qq[-1, 0] < 0:
        qq[:, 0] *= -1
    return qq


def aligned(x, y_, i):
    qq_ = oriented(x, y_, int(i))
    if qq_ is None:
        return None
    out = np.full((21, 2), np.nan)
    lo_ = max(0, int(i) - 10)
    out[lo_ - (int(i) - 10):lo_ - (int(i) - 10) + len(qq_)] = qq_
    return out


def readout3(df):
    return (df.bg.median(), df.decel.median() / G_YD, df.v_apex.median() * MPH, df.reaccel.median() / G_YD)


# ---- speed traces (Fig 1 and Fig 3): speed vs time, aligned at the slowest point of the break (the apex)
def speed_trace(x, y_, iap, w=10):
    vx_, vy_, _, _ = kin(np.asarray(x, float), np.asarray(y_, float))
    s_ = np.hypot(vx_, vy_)
    out = np.full(2 * w + 1, np.nan)
    lo_, hi_ = max(0, int(iap) - w), min(len(s_), int(iap) + w + 1)
    out[lo_ - (int(iap) - w):lo_ - (int(iap) - w) + (hi_ - lo_)] = s_[lo_:hi_]
    return out


def rep_xy(r_, side):
    if side == 0:
        return REPS[r_.event_id][0], REPS[r_.event_id][1]
    return gtraces[(r_.game_id, r_.play_id, r_.nfl_id)][:2]


def phase_times(df):
    tin = ((df.v_in - df.v_apex) / df.decel.where(df.decel > 0)).median()
    tout = ((df.v_out - df.v_apex) / df.reaccel.where(df.reaccel > 0)).median()
    return (0.5 if pd.isna(tin) else float(tin)), (0.5 if pd.isna(tout) else float(tout))


TT = np.arange(-10, 11) * DT
C_BRK, C_THR, C_DRV = "#F6D9B3", "#E3E3E3", "#CDE6F5"


def shade_phases(ax, tin, tout, ymax, labels=False, fs=9.5):
    ax.axvspan(-min(tin, 1.0), -0.1, color=C_BRK, zorder=0, lw=0)
    ax.axvspan(-0.1, 0.1, color=C_THR, zorder=0, lw=0)
    ax.axvspan(0.1, min(tout, 1.0), color=C_DRV, zorder=0, lw=0)
    if labels:
        ax.text(-min(tin, 1.0) / 2 - 0.05, ymax * 0.97, "brake", ha="center", va="top", fontsize=fs, color="#8a5a00")
        ax.text(0, ymax * 0.97, "through", ha="center", va="top", fontsize=fs, color="#444444")
        ax.text(min(tout, 1.0) / 2 + 0.05, ymax * 0.97, "drive", ha="center", va="top", fontsize=fs, color="#1f5f8b")


# ============================================================================= round 16 revision (R7, judges' round-16 feedback)
# (a) the day-3 gain with every fitted piece nested in the folds. Season folds: the shrinkage (tau^2, s^2, mean) is
#     estimated only on the two seasons of the training season pair, then M0 / M1 are fitted there and scored on the other
#     pair. Class folds (sensitivity): train on the pairs and player-seasons of the other draft classes. A fold that is
#     both season-out and class-out leaves no training pairs for the 2023 class in 2024 (2024 rookies have no 2023
#     season), so the two are reported side by side rather than crossed.
def shrink_fit(ssub):
    s_ = ssub[ssub.routes >= 50]
    mu = float(np.average(s_.ssoe, weights=s_.routes))
    pv = s_.pivot_table(index="nfl_id", columns="season", values="ssoe")
    seas = sorted(s_.season.unique())
    cp = pd.concat([pv[[a, a + 1]].dropna().set_axis(["a", "b"], axis=1) for a in seas if a + 1 in seas])
    tau2 = float(np.mean((cp.a - mu) * (cp.b - mu)))
    iv = 1.0 / s_.routes.to_numpy(float)
    s2 = float(np.sum(((s_.ssoe.to_numpy() - mu) ** 2 - tau2) * iv) / np.sum(iv ** 2))
    return mu, tau2, s2


def shr_with(par, y, r):
    mu, tau2, s2 = par
    return mu + tau2 / (tau2 + s2 / np.asarray(r, float)) * (np.asarray(y, float) - mu)


def nested_fc(fold_col, ss_mask_fn):
    """Held-out M0 / M1 forecasts with the shrinkage refitted on the training side of each fold."""
    f0, f1, ys = np.empty(len(PAIRS)), np.empty(len(PAIRS)), np.empty(len(PAIRS))
    for v_ in sorted(PAIRS[fold_col].unique()):
        tst = (PAIRS[fold_col] == v_).to_numpy()
        par = shrink_fit(ss[ss_mask_fn(v_)])
        pp = PAIRS.assign(y1s=shr_with(par, PAIRS.y1, PAIRS.r1))
        trn = pp[~tst]
        if len(trn) < 8:
            raise ValueError(f"fold {v_}: {len(trn)} training pairs")
        b0_, _ = fitp(trn, F0); b1_, _ = fitp(trn, F1)
        f0[tst], f1[tst], ys[tst] = predp(pp[tst], b0_, F0), predp(pp[tst], b1_, F1), pp.y1s.to_numpy()[tst]
    return f0, f1, ys


PAIRS["cls"] = PAIRS.draft_year.astype(int)
NS0, NS1, NSY = nested_fc("s1", lambda v_: ss.season.isin([2024, 2025] if v_ == 2023 else [2023, 2024]))
cls_of = players.set_index("nfl_id").draft_year
NC0, NC1, _ = nested_fc("cls", lambda v_: cls_of.reindex(ss.nfl_id).to_numpy() != v_)
PAIRS["f_n0"], PAIRS["f_n1"], PAIRS["y1n"] = NS0, NS1, NSY
D3M = PAIRS.day3.to_numpy()
Y2 = PAIRS.y2.to_numpy(float)
G3 = clusters(PAIRS[D3M].pid)


def rho3(fc, rows=None):
    m_ = np.where(D3M)[0] if rows is None else rows
    return spearmanr(np.asarray(fc)[m_], Y2[m_])[0]


def gain_boot(fa, fb):
    """day-3 rank-correlation of fa minus that of fb, with a receiver-clustered 90% interval (forecasts fixed)."""
    d3ix = np.where(D3M)[0]
    pt = rho3(fa) - rho3(fb)
    bs = []
    for _ in range(NBOOT):
        ix = d3ix[np.concatenate([G3[i] for i in RNG.integers(0, len(G3), len(G3))])]
        bs.append(rho3(fa, ix) - rho3(fb, ix))
    return pt, pct(bs, 5), pct(bs, 95), 100 * float(np.mean(np.array(bs) > 0))


NEST = {}
for k_, fa_, fb_, lab_ in (("ns", NS1, NS0, "season folds, shrinkage nested"),
                           ("nc", NC1, NC0, "draft-class folds, shrinkage nested"),
                           ("nv", PAIRS.f_m1.to_numpy(), PAIRS.y1.to_numpy(), "rule vs naive raw last-season SOE"),
                           ("nvp", PAIRS.y1.to_numpy(), PAIRS.f_m0.to_numpy(), "naive raw SOE vs pick and routes")):
    NEST[k_] = gain_boot(fa_, fb_)
    Q[f"{k_}_g"] = fmt(NEST[k_][0], 2); CLAIM[f"{k_}_g"] = f"day-3 rank-correlation gain, {lab_}"
    Q[f"{k_}_lo"] = fmt(NEST[k_][1], 2); CLAIM[f"{k_}_lo"] = f"{lab_}: 90% lower, receiver-clustered"
    Q[f"{k_}_hi"] = fmt(NEST[k_][2], 2); CLAIM[f"{k_}_hi"] = f"{lab_}: 90% upper"
    qput(f"{k_}_pos", NEST[k_][3], 0, f"{lab_}: % of resamples with a positive gain")
RHO3 = {"pk": rho3(PAIRS.f_m0), "naive": rho3(PAIRS.y1), "rule": rho3(PAIRS.f_m1), "nest": rho3(NS1)}
for k_, v_ in RHO3.items():
    Q[f"r3_{k_}"] = fmt(v_, 2); CLAIM[f"r3_{k_}"] = f"day-3 held-out rank correlation with next-season SOE: {k_}"
print("nested / naive day-3 gains", {k: np.round(v, 3) for k, v in NEST.items()}, RHO3)

# (b) the matched contrasts with a permutation null and an interval on the gap over pick and routes. Null: last
#     season's shrunk SOE shuffled among the day-3 pairs of the same season (the matched set depends only on the
#     pick-and-routes forecasts and is fixed).
ya_, yb_ = D3P.y1s.to_numpy()[CT.a.to_numpy()], D3P.y1s.to_numpy()[CT.b.to_numpy()]
dy2_ = D3P.y2.to_numpy()[CT.a.to_numpy()] - D3P.y2.to_numpy()[CT.b.to_numpy()]
mm_ = CT.matched.to_numpy()
ct_null = []
for _ in range(NPERM):
    y1p = D3P.y1s.to_numpy().copy()
    for s_ in (2023, 2024):
        ix_ = np.where(D3P.s1.to_numpy() == s_)[0]
        y1p[ix_] = y1p[RNG.permutation(ix_)]
    ok_ = (y1p[CT.a.to_numpy()] - y1p[CT.b.to_numpy()]) * dy2_ > 0
    ct_null.append(100 * ok_[mm_].mean())
ct_null = np.array(ct_null)
qput("ctp_p", (1 + (ct_null >= CTR[0]).sum()) / (1 + NPERM), 2, "shuffle p of the matched-contrast share")
qput("ctp_null", np.median(ct_null), 0, "median matched share under the shuffle, %")
qput("ct_pk_lo", pct(ct_bs[:, 2], 5), 0, "pick-and-routes share of all contrasts, 90% lower")
qput("ct_pk_hi", pct(ct_bs[:, 2], 95), 0, "upper")
dct_ = ct_bs[:, 1] - ct_bs[:, 2]
Q["ct_d"] = fmt(CTR[1] - CTR[2], 0, True); CLAIM["ct_d"] = "SOE minus pick-and-routes share of contrasts ordered right, pts"
Q["ct_d_lo"] = fmt(pct(dct_, 5), 0, True); CLAIM["ct_d_lo"] = "90% lower, receiver-clustered"
Q["ct_d_hi"] = fmt(pct(dct_, 95), 0, True); CLAIM["ct_d_hi"] = "90% upper"
qput("ct_np", len(pids_ct), 0, "receivers in the day-3 contrasts")

# (c) bounded null. Two one-sided tests (TOST) of the primary Break G slope against the pre-specified +/-3 in margin;
#     the family-level true correlation's 90% interval is fp_pool_lo / fp_pool_hi (posterior, 8+ NFL breaks).
p_dn_ = float(_norm.sf((PRIM_B + SESOI_IN) / PRIM_SE))
qput("tost_dn", p_dn_, 2, "one-sided p against a -3 in effect (lower arm of TOST)")
qput("tost_p", max(p_dn_, float(_norm.cdf((PRIM_B - SESOI_IN) / PRIM_SE))), 2, "TOST p at +/-3 in (max of the two arms)")
fpp_ = EIVF[:, -1]
qput("fp_pool_pneg", 100 * (fpp_ < -0.2).mean(), 0, "posterior % that the pooled true link is below -0.2")
qput("fp_pool_pin", 100 * ((fpp_ > -0.2) & (fpp_ < 0.2)).mean(), 0, "posterior % that the pooled true link is within +/-0.2")

# (d) stakes for the 'why receivers' paragraph: what a day-3 receiver costs and how many cuts a Combine scout sees
apy_ = car_.contract_apy_m.reindex(B26.nfl_id).dropna()
qput("why_apy", float(apy_.median()), 2, "median contract APY of 2025 day-3/UDFA WR/TE with 100+ routes, $M")
reps_pl_ = cb.groupby("nfl_id").size()
qput("why_reps", int(reps_pl_.median()), 0, "median graded route-drill reps per receiver at the Combine")
qput("why_single", 100 * float((cb.groupby(["nfl_id", "drill_name"])
                                 .size() == 1).mean()), 0, "% of receiver-drill cells with a single graded rep")

# (e) scout notes for the act-on names: route mix and snap separation by coverage on the sampled 2025 snaps (any
#     route), alignment, passer and games. Coverage splits are film prompts: their split-half reliability is low.
ACT = L26[L26.act].sort_values("pc_m1", ascending=False)
SCOUT = []
for _, r_ in ACT.iterrows():
    tg_ = r_.display_name.split()[-1].lower().replace("'", "").replace(".", "")
    s25_ = gb[(gb.nfl_id == r_.nfl_id) & (gb.season == 2025)]
    rm_ = s25_.route_ran.dropna().str.lower().value_counts(normalize=True)
    sm_ = s25_[s25_.soe.notna()]
    man_, zon_ = sm_[sm_.man], sm_[~sm_.man]
    SCOUT.append(dict(tag=tg_, name=r_.display_name, n=len(s25_), r1=rm_.index[0], r1p=100 * rm_.iloc[0],
                      r2=rm_.index[1], r2p=100 * rm_.iloc[1], nman=len(man_), nzon=len(zon_),
                      sman=IN_YD * man_.soe.mean(), szon=IN_YD * zon_.soe.mean(), qb=r_.qb, games=int(r_.games),
                      slot=100 * float(s25_.slotp.mean())))
    qput(f"sc_{tg_}_n", len(s25_), 0, f"{r_.display_name}: sampled 2025 snaps")
    qput(f"sc_{tg_}_r1p", 100 * rm_.iloc[0], 0, f"{r_.display_name}: share of sampled routes, top route")
    qput(f"sc_{tg_}_r2p", 100 * rm_.iloc[1], 0, f"{r_.display_name}: share of sampled routes, second route")
    qput(f"sc_{tg_}_nman", len(man_), 0, f"{r_.display_name}: sampled snaps vs man")
    Q[f"sc_{tg_}_sman"] = fmt(IN_YD * man_.soe.mean(), 0, True); CLAIM[f"sc_{tg_}_sman"] = "snap SOE vs man, in"
    Q[f"sc_{tg_}_szon"] = fmt(IN_YD * zon_.soe.mean(), 0, True); CLAIM[f"sc_{tg_}_szon"] = "snap SOE vs zone, in"
    qput(f"sc_{tg_}_slot", 100 * float(s25_.slotp.mean()), 0, f"{r_.display_name}: % of sampled 2025 snaps in the slot")
SCOUT = pd.DataFrame(SCOUT)
SCOUT.round(1).to_csv(OUT / "scout_2026.csv", index=False)
qput("sc_rel", float(NUM["sh_manzone_full"]), 2, "split-half reliability of a receiver's man-minus-zone snap SOE")
print(SCOUT.round(1).to_string())
WATCH = L26[~L26.act].sort_values("pc_m1", ascending=False)
tick("R7 nested, contrasts, TOST, scout notes")


# ---- Fig 1 (round 16 revision). A: all receivers with 8+ NFL breaks, Combine Break G and NFL Break G (both z across
# those receivers), one line per receiver; the two shown below are the largest fall and the largest rise. B: those two,
# on the route family with the most NFL breaks that he also ran at the Combine: drill path, NFL path (over the league
# median path for that family) and speed through the cut (his Combine and NFL medians, league NFL median dashed).
P8z = P8.assign(zd=zs(P8.d), zg=zs(P8.gm))
P8z["drop"] = P8z.zd - P8z.zg
SIXP = [P8z.sort_values("drop", ascending=False).index[0], P8z.sort_values("drop").index[0]]
nm_of = players.set_index("nfl_id").display_name
yr_of = players.set_index("nfl_id").draft_year
VMAX = 9.0
REF = {}


def fam_ref(fam):
    if fam in REF:
        return REF[fam]
    out = {}
    for side, df_ in ((0, cb[cb.family == fam]), (1, g[g.route_ran == fam])):
        ps_, sp_ = [], []
        for r_ in df_.itertuples():
            xx_, yy_ = rep_xy(r_, side)
            p_ = aligned(xx_, yy_, r_.i_break)
            if p_ is not None:
                ps_.append(p_)
            sp_.append(speed_trace(xx_, yy_, r_.i_apex))
        out[side] = (np.nanmedian(np.stack(ps_), axis=0), np.nanmedian(np.stack(sp_), axis=0), df_.bg.median())
    REF[fam] = out
    return out


C_FALL, C_RISE = OI["verm"], OI["blue"]
fig = plt.figure(figsize=(FIGW, 11.4))
axS = fig.add_axes([0.11, 0.70, 0.80, 0.185])
rj_ = np.random.default_rng(5)
jd_, jg_ = rj_.uniform(-0.12, 0.12, len(P8z)), rj_.uniform(-0.12, 0.12, len(P8z))
for i_, (pid_, r_) in enumerate(P8z.iterrows()):
    axS.plot([r_.zd, r_.zg], [1 + jd_[i_], jg_[i_]], color="#BBBBBB", lw=0.7, alpha=0.55, zorder=1)
axS.scatter(P8z.zd, 1 + jd_, s=26, color="#8C8C8C", edgecolor="white", lw=0.4, zorder=2)
axS.scatter(P8z.zg, jg_, s=26, color="#8C8C8C", edgecolor="white", lw=0.4, zorder=2)
for pid_, col_, lab_ in ((SIXP[0], C_FALL, "largest fall"), (SIXP[1], C_RISE, "largest rise")):
    i_ = list(P8z.index).index(pid_)
    r_ = P8z.loc[pid_]
    axS.plot([r_.zd, r_.zg], [1 + jd_[i_], jg_[i_]], color=col_, lw=2.6, zorder=3)
    axS.scatter([r_.zd, r_.zg], [1 + jd_[i_], jg_[i_]], s=90, color=col_, edgecolor="black", zorder=4)
    axS.text(r_.zd, 1.24, f"{nm_of[pid_]} ({lab_})", color=col_, fontsize=12, fontweight="bold", ha="center",
             va="bottom")
axS.set_yticks([1, 0], ["Combine\nroute drills", "NFL snaps,\nsame routes"], fontsize=12)
axS.set_ylim(-0.3, 1.45); axS.set_xlim(-3.2, 3.2)
axS.set_xlabel("Break G, z across the receivers (right = harder cut)", fontsize=12)
axS.axvline(0, color="#DDDDDD", lw=1.0, zorder=0)
fig.text(0.01, 0.975, "The Combine cut does not predict the NFL cut", fontsize=16, fontweight="bold", va="top")
fig.text(0.01, 0.94, f"A. All {len(P8z)} receivers with 8+ NFL breaks, one line each. Noise-corrected true correlation "
         f"{FAMPOST.set_index('k').loc['pool', 'med']:.2f} (90%: {FAMPOST.set_index('k').loc['pool', 'lo']:.2f} to "
         f"{FAMPOST.set_index('k').loc['pool', 'hi']:.2f}).", fontsize=12.5, va="top")
fig.text(0.01, 0.912, "The two coloured lines are the extremes, chosen by rule: expect both to regress toward the "
         "middle on a re-test.", fontsize=11.5, va="top", color="#7a2e00")
for j, pid in enumerate(SIXP):
    col_p = C_FALL if j == 0 else C_RISE
    dr_all, nf_all = cb[cb.nfl_id == pid], g[g.nfl_id == pid]
    fams = [f for f in nf_all.route_ran.value_counts().index if (dr_all.family == f).any()]
    fam = fams[0]
    dr, nf = dr_all[dr_all.family == fam], nf_all[nf_all.route_ran == fam]
    ref_ = fam_ref(fam)
    y0_ = 0.365 - 0.275 * j
    fig.text(0.01, y0_ + 0.255, f"B{j + 1}. {nm_of[pid]} ({yr_of[pid]}), {fam.lower()}: Break G {dr.bg.median():.2f} g "
             f"in drills, {nf.bg.median():.2f} g on NFL snaps (league {ref_[0][2]:.2f} / {ref_[1][2]:.2f})",
             fontsize=12.5, fontweight="bold", color=col_p)
    FIGN[f"fig1_{j}"] = nm_of[pid]
    axSp = fig.add_axes([0.60, y0_ + 0.015, 0.38, 0.19])
    axSp.plot(TT, ref_[1][1], color="black", lw=1.6, ls="--", zorder=2)
    for side, (df_, ls_, lab_) in enumerate(((dr, ":", "Combine"), (nf, "-", "NFL"))):
        ax = fig.add_axes([0.03 + 0.27 * side, y0_ + 0.0, 0.25, 0.235])
        rp_ = ref_[side][0]
        ax.plot(rp_[:, 0], rp_[:, 1], color="#9A9A9A", lw=5.0, alpha=0.55, zorder=1, solid_capstyle="round")
        paths, sps = [], []
        for r_ in df_.itertuples():
            xx_, yy_ = rep_xy(r_, side)
            p_ = aligned(xx_, yy_, r_.i_break)
            if p_ is not None:
                paths.append(p_)
                ax.plot(p_[:, 0], p_[:, 1], color=col_p, lw=0.9, alpha=0.3, zorder=2)
            sps.append(speed_trace(xx_, yy_, r_.i_apex))
        if paths:
            med = np.nanmedian(np.stack(paths), axis=0)
            ax.plot(med[:, 0], med[:, 1], color=col_p, lw=3.4, zorder=3, solid_capstyle="round")
        if sps:
            axSp.plot(TT, np.nanmedian(np.stack(sps), axis=0), color=col_p, lw=2.8, ls=ls_, zorder=3, label=lab_)
        ax.plot(0, 0, "o", ms=7, mfc="white", mec="black", mew=1.3, zorder=5)
        ax.set_xlim(-3.0, 6.5); ax.set_ylim(-6.6, 4.4); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(True); sp_.set_color("#DDDDDD")
        ax.text(0.03, 0.97, f"{lab_} (n={len(df_)})", transform=ax.transAxes, fontsize=12, va="top",
                fontweight="bold", color="#222222")
        if j == 0 and side == 0:
            ax.plot([3.6, 5.6], [-6.0, -6.0], color="black", lw=2.6, solid_capstyle="butt", zorder=6)
            ax.text(4.6, -5.7, "2 yd", ha="center", va="bottom", fontsize=11, zorder=6)
    axSp.axvline(0, color="#CCCCCC", lw=1.0, zorder=0)
    axSp.set_xlim(-1.0, 1.0); axSp.set_ylim(0, VMAX)
    axSp.set_xticks([-1, -0.5, 0, 0.5, 1]); axSp.set_yticks([0, 3, 6, 9])
    axSp.set_ylabel("speed, yd/s", fontsize=11.5)
    if j == 1:
        axSp.set_xlabel("seconds from the slowest point of the cut", fontsize=11.5)
    axSp.legend(handles=[plt.Line2D([], [], color=col_p, lw=2.6, ls=":", label="his Combine"),
                         plt.Line2D([], [], color=col_p, lw=2.6, label="his NFL"),
                         plt.Line2D([], [], color="black", lw=1.6, ls="--", label="league NFL")],
                loc="lower center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.98))
fig.text(0.01, 0.008, "Paths: thin = single reps, bold = his median, wide grey = league median for that route; rotated "
         "to run upfield and break right;\nwhite dot = sharpest point. Route family = his most-run NFL family that he "
         "also ran at the Combine. 1 g = his body weight.", fontsize=10.5, color="#333333")
save(fig, "fig1_cards.png")

# ---- Fig 2. A: the drill cut repeats across drills. B: raw drill-vs-NFL rank correlation (any NFL breaks).
# C: same receiver, same route family, measurement noise modelled: posterior true link per family and pooled, with the
# reference of a trait that does carry (official 40 vs NFL top speed on the sampled snaps, raw rank correlation).
vtop_ = {}
for k_, tr_ in gtraces.items():
    vx_, vy_, _, _ = kin(np.asarray(tr_[0], float), np.asarray(tr_[1], float))
    sp_ = np.hypot(vx_, vy_)[tr_[2]:tr_[3] + 1]
    if len(sp_):
        vtop_.setdefault(k_[2], []).append(float(sp_.max()))
vtop_ = pd.Series({k: np.percentile(v, 90) for k, v in vtop_.items() if len(v) >= 8})
f40_ = (X["forty_s"] if "forty_s" in X.columns else X["forty"]).reindex(vtop_.index)
spd_ = boot_rho(-f40_, vtop_)
qput("spd_rho", spd_[0], 2, "official 40 (faster = +) vs NFL top speed on sampled snaps, raw rank correlation")
qput("spd_n", spd_[3], 0, "receivers with 8+ sampled snaps and an official 40")
fig = plt.figure(figsize=(FIGW, 10.0))
LIM = 3.2
xx_ = np.array([-LIM, LIM])
axA = fig.add_axes([0.07, 0.585, 0.36, 0.37]); axB = fig.add_axes([0.56, 0.585, 0.36, 0.37])
axA.scatter(zs(HH_BG.a), zs(HH_BG.b), s=22, color=C_DR, alpha=0.75, edgecolor="none")
axA.plot(xx_, xx_ * REL["bg"][0], color="black", lw=2.0)
axA.set_title(f"A. Drill cut repeats: ρ = {REL['bg'][0]:.2f} ({REL['bg'][3]} players)", loc="left")
axA.set_xlabel("Break G, five route drills (z)"); axA.set_ylabel("Break G, the other five (z)")
zx, zy = zs(JP.BG), zs(JP.g_bg)
m8 = JP.index.isin(P8.index)
axB.scatter(zx[~m8], zy[~m8], s=22, color="#BBBBBB", edgecolor="none", label="fewer than 8 NFL breaks")
axB.scatter(zx[m8], zy[m8], s=32, color=C_NFL, edgecolor="black", lw=0.4, label=f"8+ NFL breaks ({len(P8)})")
axB.plot(xx_, xx_ * pr[0], color="black", lw=2.0)
axB.set_title(f"B. ...not to NFL snaps: raw ρ = {pr[0]:.2f} ({pr[3]} players)", loc="left")
axB.set_xlabel("Combine Break G (z)"); axB.set_ylabel("NFL Break G, same five route families (z)")
axB.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, frameon=False)
for a__ in (axA, axB):
    a__.set_xlim(-LIM, LIM); a__.set_ylim(-LIM, LIM); a__.set_aspect("equal")
axC = fig.add_axes([0.17, 0.15, 0.52, 0.25])
axC.axvspan(0.2, 0.9, color="#EFEFEF", zorder=0)
axC.axvline(0, color="#888888", lw=0.9)
axC.axvline(0.2, color="#555555", lw=1.2, ls="--")
axC.axvline(spd_[0], color=OI["green"], lw=2.2)
axC.text(spd_[0] + 0.015, -1.05, f"reference: official 40 →\nNFL top speed, ρ = {spd_[0]:.2f}", fontsize=11,
         color=OI["green"], va="center", ha="left", fontweight="bold")
axC.text(0.185, -1.05, "0.2: smallest\nuseful link", fontsize=11, color="#333333", va="center", ha="right")
FAMPOST["se"] = (FAMPOST.hi - FAMPOST.lo) / 3.29
FAMPOST["pow"] = norm.cdf(0.2 / FAMPOST.se - 1.645)
axT = fig.add_axes([0.72, 0.15, 0.24, 0.25])
axT.axis("off"); axT.set_ylim(len(FAMPOST) - 0.5, -1.5); axT.set_xlim(0, 1)
axT.text(0.30, -1.05, "P(link\n> 0.2)", ha="center", va="center", fontsize=11.5, fontweight="bold")
axT.text(0.75, -1.05, "power*", ha="center", va="center", fontsize=11.5, fontweight="bold")
for r, r_ in FAMPOST.reset_index(drop=True).iterrows():
    pooled_ = r_.k == "pool"
    low_ = (not pooled_) and r_["pow"] < 0.25
    col_ = "black" if pooled_ else ("#BDBDBD" if low_ else "#555555")
    axC.plot([r_.lo, r_.hi], [r, r], color=col_, lw=3.2 if pooled_ else 2.4)
    axC.plot(r_.med, r, "D" if pooled_ else "o", ms=10 if pooled_ else 8.5, color=col_,
             mec="black" if not low_ else "#999999")
    tc_ = "#999999" if low_ else "black"
    axT.text(0.30, r, f"{r_.p02:.0f}%", ha="center", va="center", fontsize=12, color=tc_,
             fontweight="bold" if pooled_ else "normal")
    axT.text(0.75, r, f"{100 * r_['pow']:.0f}%", ha="center", va="center", fontsize=12, color=tc_)
    FIGN[f"fig2_pow_{r_.k}"] = f"{100 * r_['pow']:.0f}"
axC.set_yticks(range(len(FAMPOST)), [f"{l_} ({int(n_)})" if k_ != "pool" else f"All five, pooled ({int(n_)})"
                                     for l_, k_, n_ in zip(FAMPOST.lab, FAMPOST.k, FAMPOST.n)], fontsize=12)
axC.set_ylim(len(FAMPOST) - 0.5, -1.5); axC.set_xlim(-0.6, 0.9)
axC.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75])
axC.set_xlabel("true drill-to-NFL correlation, same route family (median, 90% interval)", fontsize=12)
fig.text(0.01, 0.445, "C. Same route family, noise modelled: no family's link is distinguishable from zero",
         fontsize=13.5, fontweight="bold", ha="left")
fig.text(0.01, 0.010, f"A, B: one dot per receiver. C: receivers with 8+ NFL breaks in brackets; noise of few reps and few "
         f"snaps modelled ({len(EC)} receiver-family cells).\nGreen: official 40 vs NFL top speed on sampled snaps, raw "
         f"ρ ({spd_[3]} receivers), a trait that does carry. *power = chance a true link of 0.2\nis detected; light "
         f"grey = below 25%. The pooled interval is the bound used in the text.", fontsize=10.5, color="#333333")
save(fig, "fig2_chain.png")

# ---- Fig 3. Pooled film check instead of a hand-picked pair: receivers with a Combine score and NFL breaks, split into
# thirds of Combine Break G; per route family, the median of each receiver's median NFL speed trace through the cut
# (aligned at the slowest point), 90% band from resampling receivers. Speed at the top of the cut: top third minus bottom.
TER = pd.qcut(JP.BG, 3, labels=False)
TOPI, BOTI = set(TER.index[TER == 2]), set(TER.index[TER == 0])
PTR = {}
for r_ in g.itertuples():
    tr_ = gtraces[(r_.game_id, r_.play_id, r_.nfl_id)]
    PTR.setdefault((r_.route_ran, r_.nfl_id), []).append(speed_trace(tr_[0], tr_[1], r_.i_apex))
PMED = {k: np.nanmedian(np.stack(v), axis=0) for k, v in PTR.items()}
VAP = g.groupby(["route_ran", "nfl_id"]).v_apex.median()
FILM3 = {}
fig, axs = plt.subplots(1, 5, figsize=(FIGW, 5.6), sharey=True)
fig.subplots_adjust(left=0.06, right=0.985, top=0.66, bottom=0.17, wspace=0.08)
for fi_, (fam, ax) in enumerate(zip(FAMILIES, axs)):
    res_ = {}
    for tg_, ids_, col_ in (("top", TOPI, OI["verm"]), ("bot", BOTI, OI["blue"])):
        keys_ = [(fam, i) for i in ids_ if (fam, i) in PMED]
        st_ = np.stack([PMED[k] for k in keys_])
        bs_ = np.stack([np.nanmedian(st_[RNG.integers(0, len(st_), len(st_))], axis=0) for _ in range(500)])
        ax.fill_between(TT, np.nanpercentile(bs_, 5, axis=0), np.nanpercentile(bs_, 95, axis=0), color=col_,
                        alpha=0.22, lw=0)
        ax.plot(TT, np.nanmedian(st_, axis=0), color=col_, lw=2.6)
        res_[tg_] = VAP.loc[fam].reindex([k[1] for k in keys_]).to_numpy(float)
    a_, b_ = res_["top"], res_["bot"]
    dd_ = [RNG.choice(a_, len(a_)).mean() - RNG.choice(b_, len(b_)).mean() for _ in range(NBOOT)]
    FILM3[fam] = (a_.mean() - b_.mean(), pct(dd_, 5), pct(dd_, 95), len(a_), len(b_))
    ax.axvline(0, color="#CCCCCC", lw=1.0, zorder=0)
    ax.set_title(f"{fam.capitalize()} ({len(a_)} vs {len(b_)})", fontsize=12.5)
    ax.text(0.5, 0.04, f"gap at the turn\n{FILM3[fam][0]:+.1f} [{FILM3[fam][1]:+.1f}, {FILM3[fam][2]:+.1f}]",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5,
            bbox=dict(fc="white", ec="#CCCCCC", boxstyle="round,pad=0.25"))
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(0, VMAX); ax.set_xticks([-0.5, 0, 0.5])
    if fi_ == 0:
        ax.set_ylabel("speed, yd/s", fontsize=12)
fig.supxlabel("seconds from the slowest point of the cut (NFL snaps)", fontsize=12, y=0.03)
fig.text(0.01, 0.965, "Hard and soft Combine cutters run the same NFL cut", fontsize=15, fontweight="bold", va="top")
fig.text(0.01, 0.885, "Orange = top third of Combine Break G (hard drill cutters), blue = bottom third. Line = median of "
         "each receiver's median NFL\nspeed trace; band = 90% interval over receivers. Box: speed at the slowest point of "
         "the cut, top third minus bottom third, yd/s [90%].\nReceivers per third in brackets (any NFL break of that family; "
         "sampled snaps, half chosen for high EPA).", fontsize=11, va="top", color="#333333")
save(fig, "fig3_film.png")
fh_ = FILM3["HITCH"]
Q["f3_hitch"] = fmt(fh_[0], 1, True); CLAIM["f3_hitch"] = "hitch: speed at the top, top minus bottom third, yd/s"
Q["f3_hitch_lo"] = fmt(fh_[1], 1, True); CLAIM["f3_hitch_lo"] = "90% lower"
Q["f3_hitch_hi"] = fmt(fh_[2], 1, True); CLAIM["f3_hitch_hi"] = "90% upper"
qput("f3_ntop", fh_[3], 0, "top-third receivers with NFL hitches"); qput("f3_nbot", fh_[4], 0, "bottom third")
qput("f3_cover", sum(v[1] < 0 < v[2] for v in FILM3.values()), 0, "families whose top-speed gap interval spans 0")
# film hypothesis: receivers who sink more at the top of NFL hitches (lower speed at the slowest point) separate more?
hv_ = g[g.route_ran == "HITCH"].groupby("nfl_id").v_apex.agg(["median", "size"])
hv_ = hv_[hv_["size"] >= 3].join(D.ssoe, how="inner")
hr_ = boot_rho(-hv_["median"], hv_.ssoe)
qput("f3_h_rho", hr_[0], 2, "rank correlation, sink at the top of NFL hitches vs career SOE (hypothesis)")
qput("f3_h_lo", hr_[1], 2, "lower"); qput("f3_h_hi", hr_[2], 2, "upper"); qput("f3_h_n", hr_[3], 0, "receivers (3+ hitches)")
qput("f3_h_lg", float(g[g.route_ran == "HITCH"].v_apex.median()), 1, "league median speed at the top of a hitch, yd/s")
for fam in FAMILIES:
    qput(f"vap_{fam.lower()}", float(g[g.route_ran == fam].v_apex.median()), 1, f"league median speed at the turn, {fam}")

# ---- Fig 4. A: the 20 day-3 / UDFA season pairs, sorted by next-season SOE, with each list's hits and misses.
# B: held-out rank correlation with next-season SOE on those 20 pairs: pick + routes, last season's raw SOE alone,
# and the rule. C: the same 20 pairs as within-season contrasts: share ordered right, with the shuffle null.
D3 = PAIRS[PAIRS.day3].sort_values(["y2", "y1s", "s1"], kind="stable").reset_index(drop=True)
twice = set(D3.pid[D3.pid.duplicated()])
fig = plt.figure(figsize=(FIGW, 10.0))
axA = fig.add_axes([0.205, 0.185, 0.24, 0.675])
axA.axvline(0, color="#888888", lw=0.9)
for r, r_ in D3.iterrows():
    x1, x2 = IN_YD * r_.y1s, IN_YD * r_.y2
    col_ = C_DR if r_.hit else C_NFL
    axA.plot([x1, x2], [r, r], color="#CCCCCC", lw=1.6, zorder=1)
    axA.plot(x1, r, "o", ms=7.5, mfc="white", mec="black", zorder=3)
    axA.plot(x2, r, "o", ms=8.5, color=col_, mec="black", zorder=4)
    mk_ = "✓" if r_.hit else "✗"
    if r_.rule:
        axA.text(-30.5, r, mk_, color=col_, fontsize=13, fontweight="bold", va="center", ha="center")
    if r_.rule_pk:
        axA.text(-21.0, r, mk_, color=col_, fontsize=13, fontweight="bold", va="center", ha="center")
lab4 = [f"{str(int(r_.s1))[2:]}→{str(int(r_.s1) + 1)[2:]}  {r_.display_name}{' †' if r_.pid in twice else ''}"
        for _, r_ in D3.iterrows()]
axA.set_yticks(range(len(D3)), lab4, fontsize=10.5)
axA.set_xlim(-34, 26); axA.set_ylim(-0.7, len(D3) - 0.3)
nr_, hr4_ = int(D3.rule.sum()), int(D3[D3.rule].hit.sum())
np_, hp_ = int(D3.rule_pk.sum()), int(D3[D3.rule_pk].hit.sum())
axA.text(-30.5, len(D3) - 0.35, f"SOE\nlist\n{hr4_}/{nr_}", ha="center", va="bottom", fontsize=11, fontweight="bold")
axA.text(-20.0, len(D3) - 0.35, f"pick +\nroutes\n{hp_}/{np_}", ha="center", va="bottom", fontsize=11,
         fontweight="bold", color="#a36500")
axA.set_xlabel("separation over expected, inches\n○ last season (shrunk)   ● next season", fontsize=11.5)
fig.text(0.01, 0.925, f"A. All {len(D3)} day-3 / UDFA season pairs\n({D3.pid.nunique()} receivers), sorted by outcome",
         fontsize=13, fontweight="bold")
axA.legend(handles=[plt.Line2D([], [], color=C_DR, marker="o", ls="", ms=9, mec="black", label="✓ stayed above 0"),
                    plt.Line2D([], [], color=C_NFL, marker="o", ls="", ms=9, mec="black", label="✗ fell below 0")],
           loc="upper left", bbox_to_anchor=(-0.25, -0.105), ncol=2, frameon=False, fontsize=11)
axB = fig.add_axes([0.70, 0.64, 0.22, 0.20])
d3ix_ = np.where(D3M)[0]
BR = []
for lab_, fc_, col_ in (("pick + routes", PAIRS.f_m0, "#a36500"), ("last season's\nSOE alone", PAIRS.y1, OI["grey"]),
                        ("rule: pick + routes\n+ shrunk SOE", PAIRS.f_m1, "black")):
    fc_ = np.asarray(fc_, float)
    bs_ = [rho3(fc_, d3ix_[np.concatenate([G3[i] for i in RNG.integers(0, len(G3), len(G3))])]) for _ in range(NBOOT)]
    BR.append((lab_, rho3(fc_), pct(bs_, 5), pct(bs_, 95), col_))
axB.axvline(0, color="#888888", lw=0.9)
for r, (lab_, pt_, lo_, hi_, col_) in enumerate(BR):
    axB.plot([lo_, hi_], [r, r], color=col_, lw=3.0)
    axB.plot(pt_, r, "D", ms=10, color=col_, mec="black", zorder=3)
    axB.text(1.03, r, f"{pt_:+.2f}", va="center", fontsize=11.5, color=col_, fontweight="bold",
             transform=axB.get_yaxis_transform())
    FIGN[f"fig4b_{r}"] = f"{pt_:.2f}"
axB.set_yticks(range(len(BR)), [b_[0] for b_ in BR], fontsize=11)
axB.set_ylim(len(BR) - 0.5, -0.5); axB.set_xlim(-0.7, 0.9)
axB.set_xlabel("held-out rank correlation with next-season SOE", fontsize=11.5)
fig.text(0.53, 0.925, f"B. Ranking the same {len(D3)} pairs, held out by season\nGain of the rule over pick + routes: "
         f"{Q['ev_d3_drho']} [{Q['ev_d3_drho_lo']}, {Q['ev_d3_drho_hi']}]", fontsize=13, fontweight="bold")
axCc = fig.add_axes([0.70, 0.17, 0.20, 0.27])
CROWS = [("SOE, closest\nforecasts (matched)", CTR[0], pct(ct_bs[:, 0], 5), pct(ct_bs[:, 0], 95), "black",
          int(CT.matched.sum())),
         ("SOE, all", CTR[1], pct(ct_bs[:, 1], 5), pct(ct_bs[:, 1], 95), OI["grey"], len(CT)),
         ("pick + routes, all", CTR[2], pct(ct_bs[:, 2], 5), pct(ct_bs[:, 2], 95), "#a36500", len(CT))]
axCc.axvspan(pct(ct_null, 5), pct(ct_null, 95), color="#E6E6E6", lw=0, zorder=0)
axCc.axvline(50, color="#888888", lw=0.9)
for r, (lab_, pt_, lo_, hi_, col_, n_) in enumerate(CROWS):
    axCc.plot([lo_, hi_], [r, r], color=col_, lw=3.0)
    axCc.plot(pt_, r, "o", ms=10, color=col_, mec="black", zorder=3)
    axCc.text(1.03, r, f"{pt_:.0f}%\nof {n_}", va="center", fontsize=11, color=col_,
              transform=axCc.get_yaxis_transform())
axCc.set_yticks(range(len(CROWS)), [c_[0] for c_ in CROWS], fontsize=11)
axCc.set_ylim(len(CROWS) - 0.5, -0.5); axCc.set_xlim(20, 100)
axCc.set_xlabel("% of contrasts ordered right", fontsize=11.5)
fig.text(0.53, 0.50, f"C. The same {len(D3)} pairs as {len(CT)} within-season\ncontrasts: who separates more next "
         "year?", fontsize=13, fontweight="bold")
fig.suptitle("Last season's SOE is a small day-3 tiebreaker, not a proven edge", x=0.01, ha="left", fontsize=15,
             fontweight="bold", y=0.99)
fig.text(0.01, 0.008, "A: a ✓/✗ in a list column = that list took the pair (SOE list: shrunk SOE above a cut-off set on "
         "the other season;\npick + routes: same number, by forecast). † = in two pairs. B, C: bars = 90% intervals, "
         "resampling receivers.\nC: matched = the half of contrasts with the closest pick-and-routes forecasts; grey "
         "band = 90% of shuffles of last season's SOE.", fontsize=10.5, color="#333333")
save(fig, "fig4_shortlist.png")
tick("figures")

# numbers.json holds only the numbers quoted in writeup.md (one key per number); everything else that was computed goes
# to unquoted_numbers.json. A key counts as quoted when its value appears in the text next to a non-digit.
import re  # noqa: E402

WU = OUT / "writeup.md"
TXT = WU.read_text().replace("−", "-") if WU.exists() else None


def quoted(v):
    if TXT is None:
        return True
    s_ = str(v).lstrip("-")  # a "+" sign must appear as written (same rule as writeup/check.py)
    pat = r"(?<![\d.])" + ("" if s_.startswith("+") else "[+-]?") + re.escape(s_) + r"(?![\d]|\.\d)"
    return re.search(pat, TXT) is not None


ALLCLAIM = dict(CLAIM)
UNQ = {k: Q.pop(k) for k in list(Q) if not quoted(Q[k])}
for k in UNQ:
    CLAIM.pop(k, None)
(OUT / "numbers.json").write_text(json.dumps(Q, indent=1))
(OUT / "claims.json").write_text(json.dumps(CLAIM, indent=1))
(OUT / "unquoted_numbers.json").write_text(json.dumps(UNQ, indent=1))
(OUT / "figure_numbers.json").write_text(json.dumps(FIGN, indent=1))
# full set, for writing the text
(OUT / "all_numbers.json").write_text(json.dumps({**Q, **UNQ}, indent=1))
(OUT / "all_claims.json").write_text(json.dumps({**CLAIM, **ALLCLAIM}, indent=1))
# render writeup.md from writeup_template.md (fills {key} from all_numbers.json) and re-filter numbers.json / claims.json
if (OUT / "writeup_template.md").exists():
    import runpy  # noqa: E402

    runpy.run_path(str(OUT / "render.py"))

