"""Break G: does the Combine route-drill cut predict NFL separation? And what does?

Round 18 candidate (revision of round_17/revised). Run from the repo root (about nine minutes):
    uv run python writeup/candidates/round_18/cand_2/analysis.py
Fast check of the headline numbers (seconds): uv run python writeup/candidates/round_18/cand_2/verify.py

Round 18 changes (judges' round-17 feedback), section "round 18" before the figures:
- one pre-specified day-3 test: receiver-clustered bootstrap interval plus a receiver-clustered shuffle that keeps the
  fitted season-fold models and shuffles only the day-3 receivers' own last-season SOE (the earlier all-pairs shuffle
  also destroyed the training signal, which is why it gave p = 0.01); design effect and effective n;
- Break G against the +/-3 in margin as a one-sided benefit test, a TOST and the share of the bootstrap inside +/-3;
- a decision-grade tiebreak rule: same-season day-3 contrasts, SOE vs the earlier pick, by pick-gap threshold, with the
  share ordered right, the hit rate of the chosen receiver and the cost of being wrong (inches and catches);
- per-receiver SOE splits (man / zone, slot / wide) with cell n; posterior intervals of 2026 SOE;
- pairs_inputs.csv, speed_inputs.csv, columns.json and verify.py (replaces reproduce_baseline.py);
- Fig 4 = day-3 tiebreaker (pairs, pick-gap curve, gains); Fig 5 = 2026 calls with intervals and split cells; Fig 2
  shows both speed windows.

Round 17 changes (judges' round-16 feedback): one headline margin (upper 90% bound of the primary slope, in catches);
baseline held-out R^2 / RMSE / rank correlation and the Break G increment as delta R^2 with a class-stratified interval;
speed benchmark on the full-play window (90th percentile of per-snap max speed) with the window stated, the restricted
snap-to-throw number kept for reference; earlier lists vs the base rate with Jeffreys intervals; the 20 day-3 pairs as a
paired contrast of forecast errors, WR and TE separately; simulated power of the 2026 test; GO / HOLD calls.
Fig 1 = five receivers spanning the Combine range against the league median with bootstrap ribbons; Fig 2 = one
full-width family forest with the pooled band; Fig 4 = paired-contrast dot plot and 2026 cards.

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

# ============================================================================= round 17 (judges' round-16 feedback)
# (a) one headline margin: the upper 90% bound of the primary slope (+1.0 in per sd) and what it is worth in catches.
PRIM_HI = pct(COMBO["main"]["bs"][:, 1], 95)
qput("hi_catch", catches(PRIM_HI)[0], 2, "catches a season at the upper 90% bound of the primary Break G slope")
qput("hi_catch_hi", catches(PRIM_HI)[2], 2, "same, high end of the catch conversion (75th pct targets, upper catch rate)")

# (b) baseline without tracking: held-out (leave-one-class-out) R^2, RMSE and rank correlation of the pre-specified
#     forecast (log pick, 40, weight, NGS athleticism, TE) for career SOE, and the increment from adding Break G to it,
#     both refitted inside every class-stratified resample. R^2 against the pooled route-weighted mean.
def ho_r2(ix):
    Xi, yi, wi, ci, bi = XB_[ix], Y_[ix], WT_[ix], cls_arr0[ix], BGV[ix]
    Xb = np.column_stack([Xi, bi])
    f0, f1 = np.empty(len(ix)), np.empty(len(ix))
    for c in CLASSES:
        t = ci == c
        f0[t] = Xi[t] @ _wls(Xi[~t], yi[~t], wi[~t])
        f1[t] = Xb[t] @ _wls(Xb[~t], yi[~t], wi[~t])
    sst = np.sum(wi * (yi - np.average(yi, weights=wi)) ** 2)
    r0, r1 = 1 - np.sum(wi * (yi - f0) ** 2) / sst, 1 - np.sum(wi * (yi - f1) ** 2) / sst
    return np.array([r0, r1, r1 - r0, IN_YD * np.sqrt(np.average((yi - f0) ** 2, weights=wi)),
                     IN_YD * np.sqrt(np.average((yi - f1) ** 2, weights=wi)),
                     spearmanr(f0, yi)[0], spearmanr(f1, yi)[0]])


HR = ho_r2(ALL)
HRB = np.array([ho_r2(strat_sub(ALL)) for _ in range(NBOOT // 2)])
for j_, (k_, nd_, lab_) in enumerate((("bl_r2", 2, "baseline held-out R2"), ("bg_r2", 2, "with Break G, held-out R2"),
                                      ("dr2", 3, "Break G increment, delta R2"), ("bl_rmse", 1, "baseline RMSE, in"),
                                      ("bg_rmse", 1, "with Break G RMSE, in"), ("bl_rho", 2, "baseline rank corr"),
                                      ("bg_rho", 2, "with Break G rank corr"))):
    qput(k_, HR[j_], nd_, f"career SOE, leave-one-class-out: {lab_}")
    qput(f"{k_}_lo", pct(HRB[:, j_], 5), nd_, f"{lab_}: 90% lower, class-stratified bootstrap")
    qput(f"{k_}_hi", pct(HRB[:, j_], 95), nd_, f"{lab_}: 90% upper")
print("held-out R2 / RMSE / rho", np.round(HR, 3))

# (c) speed benchmark with the window stated. Full play = every tracked frame of the sampled snap (the NGS speed
#     column), receiver top speed = 90th percentile of his per-snap maxima (8+ snaps). The earlier 0.34 (spd_rho) used
#     only snap-to-throw frames of snaps with a detected break, which caps speed on short routes.
gfs_ = pd.read_parquet(P / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "s"])
gfs_ = gfs_.merge(gp[["game_id", "play_id", "nfl_id"]].drop_duplicates(), on=["game_id", "play_id", "nfl_id"])
pm_ = gfs_.groupby(["nfl_id", "game_id", "play_id"]).s.max().groupby("nfl_id")
vfull_ = pm_.quantile(0.9)[pm_.size() >= 8]
f40f_ = (X["forty_s"] if "forty_s" in X.columns else X["forty"]).reindex(vfull_.index).dropna()
SPDF = boot_rho(-f40f_, vfull_.reindex(f40f_.index))
qput("spdf_rho", SPDF[0], 2, "official 40 vs NFL top speed, full-play max (90th pct of per-snap max), rank corr")
qput("spdf_lo", SPDF[1], 2, "90% lower"); qput("spdf_hi", SPDF[2], 2, "90% upper")
qput("spdf_n", SPDF[3], 0, "receivers with 8+ sampled snaps and an official 40 (full-play window)")
# the same window with the raw official 40 pooled over WR and TE (the position gap in speed inflates it); the primary
# comparison above uses the 40 standardized within position and Combine year, the scale Break G is on
f40r_ = X["forty"].reindex(vfull_.index).dropna()
SPDR = boot_rho(-f40r_, vfull_.reindex(f40r_.index))
qput("spdr_rho", SPDR[0], 2, "raw official 40 (WR and TE pooled) vs full-play NFL top speed, rank corr")
qput("spdr_n", SPDR[3], 0, "receivers, raw pooled comparison")
del gfs_

# (d) the earlier lists against the base rate, binomial (Jeffreys) 90% intervals; calibration of high forecasts.
d3_ = PAIRS[PAIRS.day3]


def jeff(h, n):
    return 100 * _beta.ppf(0.05, h + 0.5, n - h + 0.5), 100 * _beta.ppf(0.95, h + 0.5, n - h + 0.5)


hk_, hn_ = int(d3_[d3_.rule].hit.sum()), int(d3_.rule.sum())
qput("lh_k", hk_, 0, "day-3 names on the earlier SOE lists who hit"); qput("lh_n", hn_, 0, "day-3 names listed")
qput("lh_pct", 100 * hk_ / hn_, 0, "hit rate of the earlier lists, %")
qput("lh_lo", jeff(hk_, hn_)[0], 0, "Jeffreys 90% lower"); qput("lh_hi", jeff(hk_, hn_)[1], 0, "upper")
bk_, bn_ = int(d3_.hit.sum()), len(d3_)
qput("lb_k", bk_, 0, "day-3 pairs that hit (base)"); qput("lb_pct", 100 * bk_ / bn_, 0, "day-3 base rate, %")
qput("lb_lo", jeff(bk_, bn_)[0], 0, "base rate, 90% lower"); qput("lb_hi", jeff(bk_, bn_)[1], 0, "upper")
pk_k, pk_n = int(d3_[d3_.rule_pk].hit.sum()), int(d3_.rule_pk.sum())
qput("lp_k", pk_k, 0, "pick-and-routes list of the same size: hits"); qput("lp_n", pk_n, 0, "listed")
ch_ = CAL.set_index("k").loc["hi"]
qput("calh_pct", 100 * ch_.hit / ch_.n, 0, "held-out forecasts of 60%+ (all rounds): observed hit rate, %")
qput("calh_lo", jeff(int(ch_.hit), int(ch_.n))[0], 0, "Jeffreys 90% lower")
qput("calh_hi", jeff(int(ch_.hit), int(ch_.n))[1], 0, "upper")

# (e) the day-3 tiebreaker pair by pair: absolute held-out error of the rule vs pick and routes on every one of the 20
#     day-3 pairs, WR and TE separately (receiver-clustered intervals of the mean error reduction, inches).
E0 = IN_YD * np.abs(PAIRS.y2 - PAIRS.f_m0).to_numpy()
E1 = IN_YD * np.abs(PAIRS.y2 - PAIRS.f_m1).to_numpy()
PAIRS["e0"], PAIRS["e1"] = E0, E1
PP = {}
for k_, msk_ in (("d3", D3M), ("wr", D3M & (PAIRS.te == 0).to_numpy()), ("te", D3M & (PAIRS.te == 1).to_numpy())):
    ix_ = np.where(msk_)[0]
    gr_ = clusters(PAIRS.pid.iloc[ix_])
    dd_ = E0[ix_] - E1[ix_]
    bs_ = [dd_[np.concatenate([gr_[i] for i in RNG.integers(0, len(gr_), len(gr_))])].mean() for _ in range(NBOOT)]
    PP[k_] = (len(ix_), int((dd_ > 0).sum()), dd_.mean(), pct(bs_, 5), pct(bs_, 95), PAIRS.pid.iloc[ix_].nunique())
    qput(f"pp_{k_}_n", len(ix_), 0, f"{k_}: day-3 pairs"); qput(f"pp_{k_}_np", PP[k_][5], 0, f"{k_}: receivers")
    qput(f"pp_{k_}_win", PP[k_][1], 0, f"{k_}: pairs where the rule's forecast is closer")
    Q[f"pp_{k_}_d"] = fmt(PP[k_][2], 1, True); CLAIM[f"pp_{k_}_d"] = f"{k_}: mean error cut, rule vs pick+routes, in"
    Q[f"pp_{k_}_lo"] = fmt(PP[k_][3], 1, True); CLAIM[f"pp_{k_}_lo"] = f"{k_}: 90% lower, receiver-clustered"
    Q[f"pp_{k_}_hi"] = fmt(PP[k_][4], 1, True); CLAIM[f"pp_{k_}_hi"] = f"{k_}: 90% upper"
qput("pp_sign_p", float(_binom.sf(PP["d3"][1] - 1, PP["d3"][0], 0.5)), 2, "sign test, rule closer on day-3 pairs")
wr_ix = np.where(D3M & (PAIRS.te == 0).to_numpy())[0]
gw_ = clusters(PAIRS.pid.iloc[wr_ix])
GWR = rho3(PAIRS.f_m1, wr_ix) - rho3(PAIRS.f_m0, wr_ix)
gwr_bs = []
for _ in range(NBOOT):
    ib_ = wr_ix[np.concatenate([gw_[i] for i in RNG.integers(0, len(gw_), len(gw_))])]
    gwr_bs.append(rho3(PAIRS.f_m1, ib_) - rho3(PAIRS.f_m0, ib_))
qput("wr_g", GWR, 2, "WR-only day-3 rank-correlation gain"); qput("wr_g_lo", pct(gwr_bs, 5), 2, "90% lower")
qput("wr_g_hi", pct(gwr_bs, 95), 2, "90% upper")
print("pair-by-pair", PP, GWR)


# (f) 2026 power by simulation. A 2026 season = N26 day-3 pairs drawn from the 20 observed (true gain = observed);
#     the pre-registered test passes when the receiver-bootstrap 90% lower bound of the gain is above 0. Also: how often
#     the 2026 gain alone comes out negative even though the edge is real, and how often it comes out at or above the
#     observed gain when there is no edge (forecasts' SOE term shuffled).
def _rk(a):
    return np.argsort(np.argsort(a, axis=-1), axis=-1).astype(float)


def _rho_rows(F, Y):
    rf, ry = _rk(F), _rk(Y)
    rf -= rf.mean(-1, keepdims=True); ry -= ry.mean(-1, keepdims=True)
    return (rf * ry).sum(-1) / np.sqrt((rf ** 2).sum(-1) * (ry ** 2).sum(-1))


d3ix_all = np.where(D3M)[0]
F0d, F1d, Yd = PAIRS.f_m0.to_numpy()[d3ix_all], PAIRS.f_m1.to_numpy()[d3ix_all], Y2[d3ix_all]
NSIM, NIN = 1000, 300
passes, neg = 0, 0
for _ in range(NSIM):
    s_ = RNG.integers(0, len(d3ix_all), N26)
    inner = s_[RNG.integers(0, N26, (NIN, N26))]
    gi_ = _rho_rows(F1d[inner], Yd[inner]) - _rho_rows(F0d[inner], Yd[inner])
    g1_ = _rho_rows(F1d[s_][None], Yd[s_][None])[0] - _rho_rows(F0d[s_][None], Yd[s_][None])[0]
    passes += np.nanpercentile(gi_, 5) > 0
    neg += g1_ < 0
qput("sim_pow", passes / NSIM, 2, "simulated power of the 2026 test (true gain = observed, N26 pairs)")
qput("sim_neg", 100 * neg / NSIM, 0, "% of simulated 2026 seasons with a negative gain although the edge is real")
qput("sim_n", NSIM, 0, "simulated seasons")
tick("R17 additions")


# ============================================================================= round 17 revision (judges' round-17 feedback)
# (a) plain-language reliability of one NFL season: share of a 200-route season's SOE that is signal (the weight the
#     empirical-Bayes shrinkage keeps), and the rest noise.
SIG200 = 100 * TAU2 / (TAU2 + S2 / 200)
qput("eb_sig", SIG200, 0, "% of a 200-route season's SOE that is signal (weight kept by the shrinkage)")
qput("eb_noise", 100 - round(SIG200), 0, "% of a 200-route season's SOE that is noise")

# (b) the day-3 tiebreaker: estimand = Spearman rho(held-out forecast, next-season SOE) with last season's shrunk SOE
#     minus the same without it; folds = season pair (fit on 2024->25, score 2023->24 and the reverse).
#     - leave-one-pair-out: drop each day-3 pair from fitting and scoring, refit both models, recompute the gain;
#     - season-fold permutation (computed above as perm_rho_p): last season's SOE shuffled within each season, both
#       models refitted; null distribution pr_;
#     - TE-only gain with a receiver-bootstrap interval (WR-only = wr_g above).
def gain_drop(drop):
    df = PAIRS.drop(index=drop)
    f0, f1 = np.empty(len(df)), np.empty(len(df))
    for s_ in (2023, 2024):
        t = (df.s1 == s_).to_numpy()
        b0, _ = fitp(df[~t], F0)
        b1, _ = fitp(df[~t], F1)
        f0[t], f1[t] = predp(df[t], b0, F0), predp(df[t], b1, F1)
    m = df.day3.to_numpy()
    y = df.y2.to_numpy()[m]
    return spearmanr(f1[m], y)[0] - spearmanr(f0[m], y)[0]


G_FULL = gain_drop([])
assert abs(G_FULL - (EVAL["d3"]["mt"]["m1"]["rho"] - EVAL["d3"]["mt"]["m0"]["rho"])) < 1e-9
LOPO = pd.Series({i: gain_drop([i]) for i in np.where(D3M)[0]})
qput("lopo_min", LOPO.min(), 2, "leave-one-pair-out day-3 gain: smallest of the 20")
qput("lopo_max", LOPO.max(), 2, "leave-one-pair-out day-3 gain: largest of the 20")
qput("lopo_pos", int((LOPO > 0).sum()), 0, "leave-one-pair-out gains above zero (of 20)")
PRN = np.asarray(pr_, float)
qput("perm_null_hi", np.nanpercentile(PRN, 95), 2, "season-fold permutation null: 95th percentile of the gain")
te_ix = np.where(D3M & (PAIRS.te == 1).to_numpy())[0]
GTE = rho3(PAIRS.f_m1, te_ix) - rho3(PAIRS.f_m0, te_ix)
gte_ = clusters(PAIRS.pid.iloc[te_ix])
gte_bs = []
for _ in range(NBOOT):
    ib_ = te_ix[np.concatenate([gte_[i] for i in RNG.integers(0, len(gte_), len(gte_))])]
    if PAIRS.y2.iloc[ib_].nunique() > 2 and PAIRS.f_m0.iloc[ib_].nunique() > 2:
        gte_bs.append(rho3(PAIRS.f_m1, ib_) - rho3(PAIRS.f_m0, ib_))
gte_bs = np.array(gte_bs)
gte_bs = gte_bs[np.isfinite(gte_bs)]
qput("te_g", GTE, 2, "TE-only day-3 rank-correlation gain (5 pairs)")
qput("te_g_lo", pct(gte_bs, 5), 2, "TE-only: 90% lower, receiver bootstrap")
qput("te_g_hi", pct(gte_bs, 95), 2, "TE-only: 90% upper")
print("LOPO", LOPO.round(2).to_dict(), "TE", GTE, pct(gte_bs, 5), pct(gte_bs, 95))

# (c) ridge sensitivity of the near-zero baseline R^2: ridge forecast (penalty chosen by inner leave-one-class-out
#     inside the training classes) fitted on two classes, scored on the third; class-stratified bootstrap.
def ho_r2_ridge(ix):
    Xi, yi, wi, ci = XB_[ix], Y_[ix], WT_[ix], cls_arr0[ix]
    fr = np.empty(len(ix))
    for c in CLASSES:
        t = ci == c
        fr[t] = ridge_cv(Xi[~t], yi[~t], wi[~t], ci[~t])[0](Xi[t])
    sst = np.sum(wi * (yi - np.average(yi, weights=wi)) ** 2)
    return 1 - np.sum(wi * (yi - fr) ** 2) / sst, IN_YD * np.sqrt(np.average((yi - fr) ** 2, weights=wi))


RR = ho_r2_ridge(ALL)
RRB = np.array([ho_r2_ridge(strat_sub(ALL)) for _ in range(NBOOT // 4)])
qput("rdg_r2", RR[0], 2, "ridge baseline, held-out R2 (leave one class out)")
qput("rdg_r2_lo", pct(RRB[:, 0], 5), 2, "ridge baseline R2, 90% lower"); qput("rdg_r2_hi", pct(RRB[:, 0], 95), 2, "upper")
qput("rdg_rmse", RR[1], 1, "ridge baseline, held-out RMSE, in")
print("ridge baseline", RR, pct(RRB[:, 0], 5), pct(RRB[:, 0], 95))

# (d) reproducibility: per-receiver inputs of the baseline and the held-out forecasts; reproduce_baseline.py rebuilds
#     season SOE from player_season_stats.parquet and re-derives R^2, RMSE and the Break G slope in seconds.
f0_ = np.empty(len(D))
for c in CLASSES:
    t = cls_arr0 == c
    f0_[t] = XB_[t] @ _wls(XB_[~t], Y_[~t], WT_[~t])
BI = D[["draft_year"] + BASE + ["BG"]].copy()
BI.insert(0, "display_name", players.set_index("nfl_id").display_name.reindex(D.index).to_numpy())
BI["career_soe_in"], BI["routes"], BI["heldout_forecast_in"] = IN_YD * Y_, WT_, IN_YD * f0_
BI.index.name = "nfl_id"
BI.round(4).to_csv(OUT / "baseline_inputs.csv")

# (e) the 2026 film checklist, by a fixed rule on each act-on name's graded NFL breaks (2025 tape first, else 2024):
#     clip 1 = the hitch with the slowest turn (the sink), clip 2 = the slant or out with the fastest turn (speed
#     through the cut), clip 3 = the snap against man coverage (Cover 0 / Cover 1) with the most separation at the throw.
gopp_ = pd.read_parquet(P / "game_plays_sample.parquet", columns=["game_id", "team_abbr"]).drop_duplicates()
HITS_ = set(HIST[HIST.hit].display_name)
L26["go"] = L26.act & L26.display_name.isin(HITS_)
qput("go_n", int(L26.go.sum()), 0, "GO names (listed again after an earlier hit)")
ACT = L26[L26.act].sort_values(["go", "pc_m1"], ascending=False)
COVN = {"COVER_0_MAN": "Cover 0", "COVER_1_MAN": "Cover 1", "COVER_2_ZONE": "Cover 2", "COVER_3_ZONE": "Cover 3",
        "COVER_4_ZONE": "Cover 4", "COVER_6_ZONE": "Cover 6", "2_MAN": "2-man"}
CLIPS = []


def clip_row(tg_, k_, r_, what):
    opp_ = sorted(set(gopp_[gopp_.game_id == r_.game_id].team_abbr) - {r_.team_abbr})
    dn_ = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[int(r_.down)]
    return dict(tag=tg_, k=k_, what=what, season=int(r_.season), week=int(r_.week), opp=opp_[0] if opp_ else "",
                q=int(r_.quarter), clock=str(r_.game_clock), down=dn_, ytg=int(r_.yards_to_go),
                route=str(r_.route_ran).lower(), cov=COVN.get(str(r_.team_coverage_type), str(r_.team_coverage_type)),
                v=float(r_.v_apex), sep=float(r_.sep), align="slot" if r_.lined_up_position == "SLOT_WR" else "wide")


for _, a_ in ACT.iterrows():
    tg_ = a_.display_name.split()[-1].lower().replace("'", "").replace(".", "")
    gg_ = g[g.nfl_id == a_.nfl_id]
    yr_ = 2025 if (gg_.season == 2025).sum() >= 3 else int(gg_.season.max())
    gg_ = gg_[gg_.season == yr_]
    h_ = gg_[gg_.route_ran == "HITCH"].sort_values("v_apex")
    s_ = gg_[gg_.route_ran.isin(["SLANT", "OUT"])].sort_values("v_apex", ascending=False)
    m_ = gg_[gg_.team_coverage_type.isin(["COVER_0_MAN", "COVER_1_MAN", "2_MAN"])].sort_values("sep", ascending=False)
    for k_, df_, what in ((1, h_, "sink"), (2, s_, "speed"), (3, m_, "man")):
        if len(df_):
            CLIPS.append(clip_row(tg_, k_, df_.iloc[0], what))
CLIPS = pd.DataFrame(CLIPS)
CLIPS.round(2).to_csv(OUT / "clips_2026.csv", index=False)
for r_ in CLIPS.itertuples():
    qput(f"clip_{r_.tag}_{r_.k}_v", r_.v, 1, f"{r_.tag} clip {r_.k}: speed at the slowest point of the cut, yd/s")
    qput(f"clip_{r_.tag}_{r_.k}_sep", r_.sep, 1, f"{r_.tag} clip {r_.k}: separation at the throw, yd")
print(CLIPS.to_string())
for _, a_ in ACT.iterrows():
    tg_ = a_.display_name.split()[-1].lower().replace("'", "").replace(".", "")
    qput(f"sl_{tg_}_slot", 100 * float(gb[(gb.nfl_id == a_.nfl_id) & (gb.season == 2025)].slotp.mean()), 0,
         f"{a_.display_name}: % of sampled 2025 snaps lined up in the slot")
# shortlist-level base-rate comparison (identical wording in table and text): earlier day-3 lists hk_/hn_ vs day-3 base
# bk_/bn_ (Jeffreys 90% intervals lh_*, lb_*); do the two intervals overlap?
qput("ov_lists", int(jeff(hk_, hn_)[0] < jeff(bk_, bn_)[1]), 0, "1 if the list and base-rate intervals overlap")
tick("R17 revision numbers")

# ============================================================================= checkpoint (development only)
# BDB_CKPT=path (off by default) saves the state here so the round-18 section and the figures can be re-run alone.
if os.environ.get("BDB_CKPT"):
    import pickle  # noqa: E402
    import types  # noqa: E402

    _keep = {}
    for _k, _v in list(globals().items()):
        if _k.startswith("__") or isinstance(_v, (types.ModuleType, types.FunctionType, type)):
            continue
        try:
            pickle.dumps(_v)
            _keep[_k] = _v
        except Exception:
            pass
    with open(os.environ["BDB_CKPT"], "wb") as _fh:
        pickle.dump(_keep, _fh)
    tick(f"checkpoint: {len(_keep)} objects")

# ============================================================================= round 18 (judges' round-17 feedback)
# (a) ONE pre-specified day-3 test, receiver-clustered. The round-15 shuffle (perm_rho_p) permuted last season's SOE
#     among all 53 pairs, so it also destroyed the all-rounds signal the model is trained on: under that null the SOE
#     coefficient is ~0 and the day-3 gain is ~0, hence p = 0.01. It answers "is there any SOE signal at all", not "does
#     it help on day 3". The day-3 test keeps the fitted season-fold models and shuffles only the day-3 receivers' own
#     last-season SOE at scoring time, whole receivers at a time (a receiver with two day-3 pairs donates both values,
#     so donors are drawn among receivers with the same seasons). Primary interval = receiver-clustered bootstrap; the
#     one-sided p reported with it is this clustered shuffle.
B1F = {}
for s_ in (2023, 2024):
    t_ = (PAIRS.s1 == s_).to_numpy()
    B1F[s_] = fitp(PAIRS[~t_], F1)[0]
d3ix = np.where(D3M)[0]
D3T = PAIRS.iloc[d3ix].copy()
D3T["ord"] = np.arange(len(D3T))
seasets = D3T.groupby("pid").s1.apply(lambda s: tuple(sorted(s)))
strata = {}
for pid_, ss_ in seasets.items():
    strata.setdefault(ss_, []).append(pid_)


def f1_with(y1s_new):
    d_ = D3T.assign(y1s=y1s_new)
    out = np.empty(len(d_))
    for s_ in (2023, 2024):
        t_ = (d_.s1 == s_).to_numpy()
        out[t_] = predp(d_[t_], B1F[s_], F1)
    return out


f1_chk = f1_with(D3T.y1s.to_numpy())
assert np.allclose(f1_chk, PAIRS.f_m1.to_numpy()[d3ix]), "season-fold M1 forecasts not reproduced"
Y2d, F0dd = D3T.y2.to_numpy(), PAIRS.f_m0.to_numpy()[d3ix]
G_OBS = spearmanr(f1_chk, Y2d)[0] - spearmanr(F0dd, Y2d)[0]
key_ = {(r_.pid, r_.s1): r_.y1s for r_ in D3T.itertuples()}
cl_null = []
for _ in range(NPERM):
    mp_ = {}
    for ss_, ids_ in strata.items():
        for a_, b_ in zip(ids_, RNG.permutation(ids_)):
            mp_[a_] = b_
    y1p_ = np.array([key_[(mp_[r_.pid], r_.s1)] for r_ in D3T.itertuples()])
    cl_null.append(spearmanr(f1_with(y1p_), Y2d)[0] - spearmanr(F0dd, Y2d)[0])
cl_null = np.array(cl_null)
qput("d3c_p", (1 + (cl_null >= G_OBS).sum()) / (1 + NPERM), 2,
     "day-3 gain: receiver-clustered shuffle p (models fixed, day-3 receivers' own SOE shuffled), one-sided")
qput("d3c_null_hi", np.percentile(cl_null, 95), 2, "receiver-clustered shuffle null: 95th percentile of the gain")
qput("d3c_strata", len(strata), 0, "exchangeability strata (receivers with the same day-3 seasons)")
qput("d3_boot_p", float(np.mean(EVAL["d3"]["bs"][:, 0] <= 0)), 2,
     "share of receiver-clustered bootstrap resamples with a day-3 gain at or below zero (one-sided)")
print("day-3 clustered shuffle", G_OBS, Q["d3c_p"], Q["d3_boot_p"])

# (b) effective number of independent day-3 observations: 20 pairs from 16 receivers; design effect = variance of the
#     gain under the receiver-clustered bootstrap / under a pair-level bootstrap.
iid_ = []
for _ in range(NBOOT):
    ib_ = RNG.integers(0, len(d3ix), len(d3ix))
    iid_.append(spearmanr(f1_chk[ib_], Y2d[ib_])[0] - spearmanr(F0dd[ib_], Y2d[ib_])[0])
DEFF = float(np.var(EVAL["d3"]["bs"][:, 0]) / np.nanvar(iid_))
qput("d3_deff", DEFF, 2, "design effect of receiver clustering on the day-3 gain (clustered / pair bootstrap variance)")
qput("d3_neff", len(d3ix) / DEFF, 0, "effective independent day-3 pairs (20 / design effect)")

# (c) Break G against the pre-set +/-3 in margin, stated three ways: one-sided test of a benefit as large as +3 (the
#     pricing question), TOST at +/-3 (equivalence), and the bootstrap share of |slope| < 3.
bsb_ = COMBO["main"]["bs"][:, 1]
qput("eq_up_p", float(np.mean(bsb_ >= SESOI_IN)), 3, "bootstrap share of Break G slopes at or above +3 in (benefit arm)")
qput("eq_dn_p", float(np.mean(bsb_ <= -SESOI_IN)), 2, "bootstrap share of Break G slopes at or below -3 in (harm arm)")
qput("eq_tost", max(float(np.mean(bsb_ >= SESOI_IN)), float(np.mean(bsb_ <= -SESOI_IN))), 2,
     "TOST p at +/-3 in (larger of the two bootstrap arms)")
qput("eq_in", 100 * float(np.mean(np.abs(bsb_) < SESOI_IN)), 0, "% of bootstrap slopes inside +/-3 in")
qput("eq_lo95", pct(bsb_, 2.5), 1, "Break G slope, 95% lower (bootstrap)")
print("equivalence", Q["eq_up_p"], Q["eq_dn_p"], Q["eq_tost"], Q["eq_in"])

# (d) a decision-grade tiebreak rule. All same-season contrasts between two day-3 / undrafted receivers (both with 100+
#     routes in both seasons; undrafted = pick 262). Rule: if the pick gap is under GAP picks, take the receiver with the
#     higher shrunk last-season SOE; otherwise take the earlier pick. Compared with "always take the earlier pick" (pick
#     ties broken by more routes). Scored on next-season SOE: share of contrasts ordered right, share of chosen receivers
#     with SOE > 0, and the cost of being wrong = mean next-season SOE shortfall of the chosen receiver when he was the
#     wrong choice. GAP = 32 (one round), fixed before scoring; other gaps shown for sensitivity. Receiver-clustered
#     bootstrap (contrasts weighted by the product of the two receivers' resample counts).
GAP = 32
pkd_ = np.exp(D3T.logpick.to_numpy())
rows_ = []
for s_ in (2023, 2024):
    ii_ = np.where(D3T.s1.to_numpy() == s_)[0]
    for x_ in range(len(ii_)):
        for z_ in range(x_ + 1, len(ii_)):
            a_, b_ = ii_[x_], ii_[z_]
            gap_ = abs(pkd_[a_] - pkd_[b_])
            pick_ch = a_ if (pkd_[a_], -D3T.r1.iloc[a_]) < (pkd_[b_], -D3T.r1.iloc[b_]) else b_
            soe_ch = a_ if D3T.y1s.iloc[a_] > D3T.y1s.iloc[b_] else b_
            rows_.append(dict(a=a_, b=b_, gap=gap_, pick=pick_ch, soe=soe_ch, s1=s_))
CTX = pd.DataFrame(rows_)
y2i_ = IN_YD * Y2d
pid_d3 = D3T.pid.to_numpy()


def rule_stats(ct, gap, w=None):
    w = np.ones(len(ct)) if w is None else w
    other = lambda ch: np.where(ch == ct.a, ct.b, ct.a)  # noqa: E731
    ch_r = np.where(ct.gap < gap, ct.soe, ct.pick)
    ch_p = ct.pick.to_numpy()
    out = {}
    for k_, ch in (("rule", ch_r), ("pick", ch_p)):
        ot = other(ch)
        right = y2i_[ch] > y2i_[ot]
        out[k_ + "_right"] = 100 * np.average(right, weights=w)
        out[k_ + "_hit"] = 100 * np.average(y2i_[ch] > 0, weights=w)
        wr_ = w * (~right)
        out[k_ + "_cost"] = float(np.average(y2i_[ot] - y2i_[ch], weights=wr_)) if wr_.sum() else np.nan
        out[k_ + "_regret"] = float(np.average(np.maximum(y2i_[ot] - y2i_[ch], 0), weights=w))
    sw = (ct.gap < gap).to_numpy() & (ch_r != ch_p)
    out["n_switch"] = int(sw.sum())
    out["d_right"] = out["rule_right"] - out["pick_right"]
    out["d_hit"] = out["rule_hit"] - out["pick_hit"]
    return out


RS = {gp_: rule_stats(CTX, gp_) for gp_ in (8, 16, 32, 64, 999)}
upid_ = np.unique(pid_d3)
RSB = {gp_: [] for gp_ in RS}
for _ in range(NBOOT):
    cnt_ = pd.Series(RNG.choice(upid_, len(upid_))).value_counts()
    m_ = cnt_.reindex(pid_d3).fillna(0).to_numpy()
    w_ = m_[CTX.a.to_numpy()] * m_[CTX.b.to_numpy()]
    if w_.sum() == 0:
        continue
    for gp_ in RS:
        st_ = rule_stats(CTX, gp_, w_)
        RSB[gp_].append([st_["d_right"], st_["d_hit"], st_["rule_cost"], st_["pick_cost"],
                         st_["pick_regret"] - st_["rule_regret"]])
RSB = {k: np.array(v) for k, v in RSB.items()}
r0_ = RS[GAP]
RSG = RSB[GAP]
qput("tb_gap", GAP, 0, "pick-gap threshold of the tiebreak rule (picks), fixed before scoring")
qput("tb_n", len(CTX), 0, "same-season day-3 contrasts")
qput("tb_n_in", int((CTX.gap < GAP).sum()), 0, "contrasts with a pick gap under the threshold")
qput("tb_switch", r0_["n_switch"], 0, "contrasts where the rule overturns the earlier pick")
qput("tb_rule_right", r0_["rule_right"], 0, "rule: % of contrasts ordered right on next-season SOE")
qput("tb_pick_right", r0_["pick_right"], 0, "always earlier pick: % ordered right")
Q["tb_d_right"] = fmt(r0_["d_right"], 0, True); CLAIM["tb_d_right"] = "rule minus pick, % ordered right, pts"
Q["tb_d_right_lo"] = fmt(pct(RSG[:, 0], 5), 0, True); CLAIM["tb_d_right_lo"] = "90% lower, receiver-clustered"
Q["tb_d_right_hi"] = fmt(pct(RSG[:, 0], 95), 0, True); CLAIM["tb_d_right_hi"] = "90% upper"
qput("tb_rule_hit", r0_["rule_hit"], 0, "rule: % of chosen receivers with next-season SOE > 0")
qput("tb_pick_hit", r0_["pick_hit"], 0, "earlier pick: % of chosen receivers with next-season SOE > 0")
Q["tb_d_hit"] = fmt(r0_["d_hit"], 0, True); CLAIM["tb_d_hit"] = "rule minus pick, hit rate of the chosen receiver, pts"
Q["tb_d_hit_lo"] = fmt(pct(RSG[:, 1], 5), 0, True); CLAIM["tb_d_hit_lo"] = "90% lower"
Q["tb_d_hit_hi"] = fmt(pct(RSG[:, 1], 95), 0, True); CLAIM["tb_d_hit_hi"] = "90% upper"
qput("tb_cost", r0_["rule_cost"], 1, "rule: mean next-season SOE shortfall when the rule picks wrong, in")
qput("tb_cost_pick", r0_["pick_cost"], 1, "earlier pick: mean shortfall when wrong, in")
for gp_ in (8, 16, 64, 999):
    lab_ = "all" if gp_ == 999 else str(gp_)
    for j_, nm_ in ((0, "d"), (1, "h")):
        v_ = RS[gp_]["d_right" if j_ == 0 else "d_hit"]
        Q[f"tb_g{lab_}_{nm_}"] = fmt(v_, 0, True)
        Q[f"tb_g{lab_}_{nm_}_lo"] = fmt(pct(RSB[gp_][:, j_], 5), 0, True)
        Q[f"tb_g{lab_}_{nm_}_hi"] = fmt(pct(RSB[gp_][:, j_], 95), 0, True)
        what_ = "% ordered right" if j_ == 0 else "hit rate of the chosen receiver"
        CLAIM[f"tb_g{lab_}_{nm_}"] = f"gap < {gp_}: SOE rule minus earlier pick, {what_}, pts"
        CLAIM[f"tb_g{lab_}_{nm_}_lo"] = "90% lower, receiver-clustered"
        CLAIM[f"tb_g{lab_}_{nm_}_hi"] = "90% upper"
    qput(f"tb_g{lab_}_n", int((CTX.gap < gp_).sum()), 0, f"contrasts with gap < {gp_}")
    qput(f"tb_g{lab_}_sw", RS[gp_]["n_switch"], 0, f"gap < {gp_}: contrasts where SOE overturns the pick")
qput("tb_all_right", RS[999]["rule_right"], 0, "SOE for every day-3 contrast: % ordered right")
qput("tb_all_cost", RS[999]["rule_cost"], 1, "SOE for every contrast: mean shortfall when wrong, in")
qput("tb_all_cost_lo", pct(RSB[999][:, 2], 5), 1, "90% lower"); qput("tb_all_cost_hi", pct(RSB[999][:, 2], 95), 1, "upper")
qput("tb_cost_pick_lo", pct(RSB[999][:, 3], 5), 1, "pick cost, 90% lower")
qput("tb_cost_pick_hi", pct(RSB[999][:, 3], 95), 1, "pick cost, 90% upper")
qput("tb_all_catch", catches(RS[999]["rule_cost"])[0], 1, "SOE rule's cost of being wrong, catches a season")
qput("tb_pick_catch", catches(RS[999]["pick_cost"])[0], 1, "earlier pick's cost of being wrong, catches a season")
qput("tb_reg_rule", RS[999]["rule_regret"], 1, "SOE rule: expected next-season SOE given up per decision, in")
qput("tb_reg_pick", RS[999]["pick_regret"], 1, "earlier pick: expected SOE given up per decision, in")
Q["tb_reg_d"] = fmt(RS[999]["pick_regret"] - RS[999]["rule_regret"], 1)
CLAIM["tb_reg_d"] = "expected SOE saved per decision by the SOE rule, in"
Q["tb_reg_d_lo"] = fmt(pct(RSB[999][:, 4], 5), 1, True); CLAIM["tb_reg_d_lo"] = "90% lower, receiver-clustered"
Q["tb_reg_d_hi"] = fmt(pct(RSB[999][:, 4], 95), 1, True); CLAIM["tb_reg_d_hi"] = "90% upper"
qput("tb_reg_catch", catches(RS[999]["pick_regret"] - RS[999]["rule_regret"])[0], 2,
     "expected catches a season saved per decision by the SOE rule")
qput("tb_np", len(upid_), 0, "receivers in the day-3 contrasts")
TBCURVE = pd.DataFrame([dict(gap=gp_, n=int((CTX.gap < gp_).sum()), d=RS[gp_]["d_right"],
                             lo=pct(RSB[gp_][:, 0], 5), hi=pct(RSB[gp_][:, 0], 95),
                             dh=RS[gp_]["d_hit"], hlo=pct(RSB[gp_][:, 1], 5), hhi=pct(RSB[gp_][:, 1], 95))
                        for gp_ in RS])
print("tiebreak rule", TBCURVE.round(1).to_string(), {k: Q[k] for k in Q if k.startswith("tb_")})
CTX.assign(gap_lt=CTX.gap < GAP).to_csv(OUT / "tiebreak_contrasts.csv", index=False)

# (e) per-receiver SOE splits for the four act-on names on their sampled NFL snaps (all seasons): man vs zone coverage,
#     slot vs wide alignment, with the cell n. Snap SOE = separation at the throw minus its context expectation (pass
#     length, cushion, coverage family, alignment), inches. Cells under 30 snaps are flagged as unreliable (all are).
SPL = []
for _, a_ in ACT.iterrows():
    tg_ = a_.display_name.split()[-1].lower().replace("'", "").replace(".", "")
    sm_ = gb[(gb.nfl_id == a_.nfl_id) & gb.soe.notna()]
    cells = {"man": sm_[sm_.man], "zone": sm_[~sm_.man], "slot": sm_[sm_.slotp == 1],
             "wide": sm_[(sm_.slotp == 0) & (sm_.inline == 0)]}
    row = dict(tag=tg_, name=a_.display_name, n=len(sm_))
    for c_, d_ in cells.items():
        row[f"{c_}_n"] = len(d_)
        row[f"{c_}_soe"] = IN_YD * d_.soe.mean() if len(d_) else np.nan
        qput(f"sp_{tg_}_{c_}_n", len(d_), 0, f"{a_.display_name}: sampled snaps, {c_}")
        if len(d_):
            Q[f"sp_{tg_}_{c_}"] = fmt(IN_YD * d_.soe.mean(), 0, True)
            CLAIM[f"sp_{tg_}_{c_}"] = f"{a_.display_name}: mean snap SOE, {c_}, in"
    SPL.append(row)
SPL = pd.DataFrame(SPL)
SPL.round(1).to_csv(OUT / "splits_2026.csv", index=False)
qput("sp_max_cell", int(SPL[[c + "_n" for c in ("man", "zone", "slot", "wide")]].to_numpy().max()), 0,
     "largest split cell among the act-on names (snaps)")
qput("sp_big_cells", int((SPL[[c + "_n" for c in ("man", "zone", "slot", "wide")]].to_numpy() >= 30).sum()), 0,
     "split cells (of 16) with 30+ snaps")
print(SPL.round(1).to_string())

# (f) the 2026 calls with an interval: posterior of true SOE given 2025 (empirical Bayes: mean shrunk SOE, variance
#     tau^2 * (s^2/r) / (tau^2 + s^2/r)), inches; and the calibrated day-3 hit rate next to each P.
L26["post_sd"] = np.sqrt(TAU2 * (S2 / L26.routes) / (TAU2 + S2 / L26.routes))
L26["post_lo"], L26["post_hi"] = L26.y1s - 1.645 * L26.post_sd, L26.y1s + 1.645 * L26.post_sd
for _, a_ in L26[L26.act].iterrows():
    tg_ = a_.display_name.split()[-1].lower().replace("'", "").replace(".", "")
    Q[f"pi_{tg_}_lo"] = fmt(IN_YD * a_.post_lo, 1, True); CLAIM[f"pi_{tg_}_lo"] = f"{a_.display_name}: 90% lower, true SOE"
    Q[f"pi_{tg_}_hi"] = fmt(IN_YD * a_.post_hi, 1, True); CLAIM[f"pi_{tg_}_hi"] = f"{a_.display_name}: 90% upper"
ACT = L26[L26.act].sort_values(["go", "pc_m1"], ascending=False)

# (g) verification inputs (verify.py recomputes the headline numbers from these in seconds) and their data dictionary.
PI = PAIRS[["pid", "display_name", "s1", "day3", "te", "logpick", "lr1", "y1", "r1", "y1s", "y2", "r2", "f_m0",
            "f_m1"]].copy()
PI.round(5).to_csv(OUT / "pairs_inputs.csv", index=False)
SPI = pd.DataFrame({"forty_std": (X["forty_s"] if "forty_s" in X.columns else X["forty"]),
                    "forty_raw": X["forty"]}).reindex(vfull_.index)
SPI["vtop_full"] = vfull_
vtop_ = {}
for k_, tr_ in gtraces.items():
    vx_, vy_, _, _ = kin(np.asarray(tr_[0], float), np.asarray(tr_[1], float))
    sp_ = np.hypot(vx_, vy_)[tr_[2]:tr_[3] + 1]
    if len(sp_):
        vtop_.setdefault(k_[2], []).append(float(sp_.max()))
vtop_ = pd.Series({k: np.percentile(v, 90) for k, v in vtop_.items() if len(v) >= 8})
SPI = SPI.reindex(SPI.index.union(vtop_.index))
SPI["forty_std"] = (X["forty_s"] if "forty_s" in X.columns else X["forty"]).reindex(SPI.index)
SPI["forty_raw"] = X["forty"].reindex(SPI.index)
SPI["vtop_break"] = vtop_.reindex(SPI.index)
SPI.index.name = "nfl_id"
SPI.round(4).to_csv(OUT / "speed_inputs.csv")
COLS = {
    "baseline_inputs.csv": {
        "nfl_id": "player id (players.parquet)", "display_name": "name", "draft_year": "Combine class = fold label",
        "logpick": "log overall pick (undrafted = 262)",
        "forty_f": "official 40 time, z-scored within Combine class and position, missing set to 0 (the mean)",
        "combine_weight_f": "Combine weight, z within position and class, missing = 0",
        "ngs_athleticism_score_f": "NGS athleticism score, z within position and class, missing = 0",
        "te": "1 = tight end", "BG": "Break G score, z units (shrunk mean of drill-x-year z-scores)",
        "career_soe_in": "route-weighted career SOE, inches (from soe_season.csv)", "routes": "career routes (weight)",
        "heldout_forecast_in": "no-tracking forecast for this receiver, fitted on the other two classes, inches"},
    "pairs_inputs.csv": {
        "pid": "player id", "s1": "first season of the pair (fold label: 2023 -> 2023/24, 2024 -> 2024/25)",
        "day3": "round 4+ or undrafted", "te": "1 = tight end", "logpick": "log pick (undrafted = 262)",
        "lr1": "log routes, season 1", "y1": "season-1 SOE, yd", "r1": "season-1 routes",
        "y1s": "season-1 SOE shrunk by routes (empirical Bayes), yd", "y2": "season-2 SOE, yd (outcome)",
        "r2": "season-2 routes", "f_m0": "held-out pick-and-routes forecast (fitted on the other season pair), yd",
        "f_m1": "held-out forecast with y1s added, yd"},
    "speed_inputs.csv": {
        "forty_std": "official 40, z within position and class", "forty_raw": "official 40, seconds",
        "vtop_full": "NFL top speed: 90th pct of per-snap max over the full sampled play, yd/s (8+ snaps)",
        "vtop_break": "same, snap-to-throw frames of breaking snaps only (8+ snaps), yd/s"},
}
(OUT / "columns.json").write_text(json.dumps(COLS, indent=1))
tick("R18 additions")

# =============================================================================================== figures (revision)
P8z = P8.assign(zd=zs(P8.d), zg=zs(P8.gm))
nm_of = players.set_index("nfl_id").display_name
VMAX = 9.0
REF = {}


def fam_ref(fam):
    if fam in REF:
        return REF[fam]
    out = {}
    for side, df_ in ((0, cb[cb.family == fam]), (1, g[g.route_ran == fam])):
        sp_ = [speed_trace(*rep_xy(r_, side), r_.i_apex) for r_ in df_.itertuples()]
        out[side] = (None, np.nanmedian(np.stack(sp_), axis=0), df_.bg.median())
    REF[fam] = out
    return out


QS = [5, 25, 50, 75, 95]
SPAN = []
for q_ in QS:
    tgt_ = np.percentile(P8z.zd, q_)
    pool_ = P8z.drop(index=[s_ for s_ in SPAN if s_ in P8z.index])
    SPAN.append((pool_.zd - tgt_).abs().idxmin())
C5 = [OI["blue"], OI["sky"], OI["grey"], OI["orange"], OI["verm"]]

# ---- Fig 1. A: scatter of Combine Break G z vs NFL Break G z (99 receivers with 8+ NFL breaks), marginal rugs,
# least-squares line with a 90% bootstrap band, five rule-chosen receivers labelled. B: their speed through the cut on
# one route family, Combine median vs NFL median, with the Combine-minus-NFL gap at the slowest point marked.
fig = plt.figure(figsize=(FIGW, 11.4))
fig.text(0.01, 0.988, "The Combine cut does not predict the NFL cut", fontsize=17, fontweight="bold", va="top")
fig.text(0.01, 0.952, f"Season separation: {Q['prim_b']} in per sd of Break G [90%: {Q['prim_lo']}, {Q['prim_hi']}], "
         f"{Q['n_main']} receivers in unseen classes; pre-set margin ±3 in.\nSame route, Combine vs NFL, noise "
         f"corrected: correlation {Q['fp_pool_med']} [{Q['fp_pool_lo']}, {Q['fp_pool_hi']}], {len(P8z)} receivers. "
         f"Official 40 vs NFL top speed: {Q['spdf_rho']}.".replace("-", "−"), fontsize=12.5, va="top", color="#222222")
axS = fig.add_axes([0.09, 0.515, 0.50, 0.36])
xs_, ys_ = P8z.zd.to_numpy(), P8z.zg.to_numpy()
bl_ = np.polyfit(xs_, ys_, 1)
xg_ = np.linspace(-3, 3, 61)
bsl_ = []
for _ in range(NBOOT):
    i_ = RNG.integers(0, len(xs_), len(xs_))
    bsl_.append(np.polyval(np.polyfit(xs_[i_], ys_[i_], 1), xg_))
bsl_ = np.array(bsl_)
axS.fill_between(xg_, np.percentile(bsl_, 5, axis=0), np.percentile(bsl_, 95, axis=0), color="#BBBBBB", alpha=0.45,
                 lw=0, zorder=1)
axS.plot(xg_, np.polyval(bl_, xg_), color="black", lw=2.0, zorder=2)
axS.plot(xg_, xg_, color="#888888", lw=1.0, ls=":", zorder=1)
axS.text(2.15, 2.45, "perfect\ntransfer", fontsize=10.5, color="#555555", ha="center")
axS.scatter(xs_, ys_, s=30, color="#A0A0A0", edgecolor="white", lw=0.5, zorder=3)
axS.plot(xs_, np.full(len(xs_), -3.15), "|", color="#666666", ms=9, alpha=0.7, clip_on=False)
axS.plot(np.full(len(ys_), -3.15), ys_, "_", color="#666666", ms=9, alpha=0.7, clip_on=False)
LABXY = [(-3.0, 0.85), (-3.0, -1.75), (-1.2, 2.45), (0.85, -2.85), (0.25, -1.4)]  # label anchors, data units
for k_, (pid_, col_) in enumerate(zip(SPAN, C5)):
    r_ = P8z.loc[pid_]
    axS.scatter([r_.zd], [r_.zg], s=120, color=col_, edgecolor="black", lw=1.2, zorder=5)
    axS.annotate(f"{k_ + 1}. {nm_of[pid_]}", (r_.zd, r_.zg), xytext=LABXY[k_], fontsize=11.5, fontweight="bold",
                 color="black", zorder=6, ha="left", va="center",
                 bbox=dict(fc="white", ec=col_, lw=1.5, alpha=0.92, boxstyle="round,pad=0.2"),
                 arrowprops=dict(arrowstyle="-", color="#444444", lw=1.0))
axS.set_xlim(-3.2, 3.2); axS.set_ylim(-3.2, 3.2)
axS.set_xlabel("Combine Break G, z (right = harder drill cut)", fontsize=12)
axS.set_ylabel("NFL Break G, same receiver, z", fontsize=12)
axS.axhline(0, color="#E5E5E5", lw=0.9, zorder=0); axS.axvline(0, color="#E5E5E5", lw=0.9, zorder=0)
FIGN["fig1_slope"] = f"{bl_[0]:.2f}"
fig.text(0.01, 0.895, f"A. All {len(P8z)} receivers with 8+ NFL breaks", fontsize=13, fontweight="bold", va="top")
fig.text(0.63, 0.86, "How to read A", fontsize=12.5, fontweight="bold", va="top")
fig.text(0.63, 0.83, f"Each dot is a receiver. If the drill cut\ncarried to the NFL, dots would follow\nthe "
         f"dotted diagonal. The fitted line\n(black, grey band = 90% interval) is\nflat: slope {bl_[0]:.2f}.\n\n"
         f"Ticks on the axes show where the\nreceivers sit on each scale.\n\nNumbered: the receivers closest to the\n"
         f"5th, 25th, 50th, 75th and 95th\npercentile of Combine Break G,\nshown in B.".replace("-", "−"),
         fontsize=11.5, va="top", color="#333333", linespacing=1.35)
fig.text(0.01, 0.415, "B. The same five: speed through the cut at the Combine (dotted) and in the NFL (solid)",
         fontsize=13, fontweight="bold", va="top")
W5 = 0.172
for k_, (pid, col_) in enumerate(zip(SPAN, C5)):
    dr_all, nf_all = cb[cb.nfl_id == pid], g[g.nfl_id == pid]
    fams = [f for f in nf_all.route_ran.value_counts().index if (dr_all.family == f).any()]
    fam = fams[0] if fams else nf_all.route_ran.value_counts().index[0]
    dr, nf = dr_all[dr_all.family == fam], nf_all[nf_all.route_ran == fam]
    if len(dr) == 0:
        dr = dr_all
    ref_ = fam_ref(fam)
    ax = fig.add_axes([0.06 + k_ * (W5 + 0.016), 0.105, W5, 0.205])
    ax.plot(TT, ref_[1][1], color="black", lw=1.4, ls="--", zorder=2)
    sn_ = np.stack([speed_trace(*rep_xy(r_, 1), r_.i_apex) for r_ in nf.itertuples()])
    bs_ = np.stack([np.nanmedian(sn_[RNG.integers(0, len(sn_), len(sn_))], axis=0) for _ in range(400)])
    ax.fill_between(TT, np.nanpercentile(bs_, 5, axis=0), np.nanpercentile(bs_, 95, axis=0), color=col_, alpha=0.25,
                    lw=0, zorder=1)
    nmed_ = np.nanmedian(sn_, axis=0)
    ax.plot(TT, nmed_, color=col_, lw=2.8, zorder=3)
    sd_ = np.stack([speed_trace(*rep_xy(r_, 0), r_.i_apex) for r_ in dr.itertuples()])
    dmed_ = np.nanmedian(sd_, axis=0)
    ax.plot(TT, dmed_, color=col_, lw=2.4, ls=":", zorder=3)
    i0_ = len(TT) // 2
    gap_ = dmed_[i0_] - nmed_[i0_]
    ax.plot([0.06, 0.06], [nmed_[i0_], dmed_[i0_]], color="black", lw=1.8, marker="_", ms=10, mew=1.8, zorder=5)
    ax.text(0.98, 0.97, f"gap at turn\n{gap_:+.1f} yd/s".replace("-", "−"), transform=ax.transAxes, ha="right",
            va="top", fontsize=11, bbox=dict(fc="white", ec="#BBBBBB", boxstyle="round,pad=0.2"))
    FIGN[f"fig1_gap_{k_}"] = f"{gap_:+.1f}"
    ax.axvline(0, color="#CCCCCC", lw=1.0, zorder=0)
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(0, VMAX); ax.set_xticks([-0.5, 0, 0.5]); ax.set_yticks([0, 3, 6, 9])
    if k_:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("speed, yd/s", fontsize=12)
    r_ = P8z.loc[pid]
    ax.set_title(f"{k_ + 1}. {nm_of[pid]}\n{fam.lower()}: Combine z {r_.zd:+.1f}\n{len(dr)} drill rep"
                 f"{'s' if len(dr) > 1 else ''}, {len(nf)} NFL snaps".replace("-", "−"), fontsize=11, loc="left",
                 fontweight="normal", color="black")
    FIGN[f"fig1_{k_}"] = nm_of[pid]
fig.text(0.5, 0.062, "seconds from the slowest point of the cut", ha="center", fontsize=12)
fig.legend(handles=[plt.Line2D([], [], color="#555555", lw=2.8, label="his NFL median (band: 90%)"),
                    plt.Line2D([], [], color="#555555", lw=2.4, ls=":", label="his Combine median"),
                    plt.Line2D([], [], color="black", lw=1.4, ls="--", label="league NFL median, same route"),
                    plt.Line2D([], [], color="black", lw=1.8, marker="|", ls="", ms=14, mew=1.8,
                               label="gap at the turn")],
           loc="lower left", bbox_to_anchor=(0.01, 0.012), ncol=4, frameon=False, fontsize=11)
save(fig, "fig1_cards.png")

# ---- Fig 2. Family forest: power encoded as marker size (no repeated text column), P(link > 0.2) kept as a column.
# speed benchmark, third window (appendix table): snap-to-throw frames of breaking snaps only
vtop_ = {}
for k_, tr_ in gtraces.items():
    vx_, vy_, _, _ = kin(np.asarray(tr_[0], float), np.asarray(tr_[1], float))
    sp_ = np.hypot(vx_, vy_)[tr_[2]:tr_[3] + 1]
    if len(sp_):
        vtop_.setdefault(k_[2], []).append(float(sp_.max()))
vtop_ = pd.Series({k: np.percentile(v, 90) for k, v in vtop_.items() if len(v) >= 8})
f40_ = (X["forty_s"] if "forty_s" in X.columns else X["forty"]).reindex(vtop_.index)
spd_ = boot_rho(-f40_, vtop_)
qput("spd_rho", spd_[0], 2, "speed benchmark, snap-to-throw window on breaking snaps: official 40 vs NFL top speed")
qput("spd_n", spd_[3], 0, "receivers with 8+ breaking snaps and an official 40")
FAMPOST["se"] = (FAMPOST.hi - FAMPOST.lo) / 3.29
FAMPOST["pow"] = norm.cdf(0.2 / FAMPOST.se - 1.645)
fp_ = FAMPOST.set_index("k").loc["pool"]
fig = plt.figure(figsize=(FIGW, 6.6))
axC = fig.add_axes([0.21, 0.20, 0.62, 0.55])
axC.axvspan(fp_.lo, fp_.hi, color="#DDE8F3", zorder=0, lw=0)
axC.axvspan(SPDF[1], SPDF[2], color="#D5EEE6", zorder=0, lw=0)
axC.axvline(0, color="#888888", lw=0.9)
axC.axvline(0.2, color="#555555", lw=1.2, ls="--")
axC.axvline(SPDF[0], color=OI["green"], lw=2.4)
axC.axvline(spd_[0], color=OI["green"], lw=1.6, ls=(0, (4, 2)))
axC.text(SPDF[2] + 0.02, -0.45, f"dashed: snap-to-throw\nwindow, {spd_[0]:.2f}", fontsize=11.5,
         color="#00664a", va="center", ha="left")
axC.text(0.185, -1.25, "0.2 = smallest\nuseful link", fontsize=12.5, color="#333333", va="center", ha="right")
axC.text(SPDF[2] + 0.02, -1.35, f"official 40 → NFL\ntop speed: {SPDF[0]:.2f}", fontsize=12.5, color="#00664a",
         va="center", ha="left", fontweight="bold")
for r, r_ in FAMPOST.reset_index(drop=True).iterrows():
    pooled_ = r_.k == "pool"
    col_ = "black" if pooled_ else "#444444"
    axC.plot([r_.lo, r_.hi], [r, r], color=col_, lw=3.4 if pooled_ else 2.4, alpha=1.0 if pooled_ else 0.85)
    ms_ = 13 if pooled_ else 5 + 14 * float(r_["pow"])
    axC.plot(r_.med, r, "D" if pooled_ else "o", ms=ms_, color=col_, mec="black", mew=1.2)
    FIGN[f"fig2_pow_{r_.k}"] = f"{100 * r_['pow']:.0f}"
axC.set_yticks(range(len(FAMPOST)), [f"{l_} ({int(n_)})" if k_ != "pool" else f"All five, pooled ({int(n_)})"
                                     for l_, k_, n_ in zip(FAMPOST.lab, FAMPOST.k, FAMPOST.n)], fontsize=12.5)
axC.set_ylim(len(FAMPOST) - 0.5, -1.9); axC.set_xlim(-0.65, 1.0)
axC.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
axC.tick_params(axis="x", labelsize=12)
axC.set_xlabel("drill-to-NFL correlation of Break G, same route family, noise corrected (median, 90% interval)",
               fontsize=12)
axT = fig.add_axes([0.85, 0.20, 0.14, 0.55])
axT.axis("off"); axT.set_ylim(len(FAMPOST) - 0.5, -1.9); axT.set_xlim(0, 1)
axT.text(0.4, -1.2, "chance the\nlink > 0.2", ha="center", va="center", fontsize=12, fontweight="bold")
for r, r_ in FAMPOST.reset_index(drop=True).iterrows():
    axT.text(0.4, r, f"{r_.p02:.0f}%", ha="center", va="center", fontsize=13,
             fontweight="bold" if r_.k == "pool" else "normal")
fig.text(0.01, 0.975, "No route family carries the drill cut to NFL snaps; straight-line speed does carry",
         fontsize=15.5, fontweight="bold", va="top")
fig.text(0.01, 0.915, f"Receivers with 8+ NFL breaks in brackets; noise of few drill reps and few NFL snaps modelled "
         f"({len(EC)} receiver-family cells).\nMarker size = chance a true link of 0.2 would be detected (bigger = more "
         "snaps). Speed: official 40 within position and class\nvs NFL top speed, full play (solid) and snap to throw (dashed).",
         fontsize=12, va="top", color="#333333")
fig.legend(handles=[matplotlib.patches.Patch(color="#DDE8F3", label="pooled 90% interval"),
                    matplotlib.patches.Patch(color="#D5EEE6", label=f"40 vs top speed, 90% ({SPDF[3]} receivers)"),
                    plt.Line2D([], [], color="#444444", marker="o", ls="", ms=5 + 14 * 0.2, mec="black",
                               label="detection chance 20%"),
                    plt.Line2D([], [], color="#444444", marker="o", ls="", ms=5 + 14 * 0.6, mec="black",
                               label="60%")],
           loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=4, frameon=False, fontsize=12)
save(fig, "fig2_chain.png")

# ---- Fig 3. Pooled film check: top vs bottom third of Combine Break G, NFL speed through the cut per family (thin
# ribbons with outlines), and beneath each family the top-minus-bottom difference trace with its 90% band.
TER = pd.qcut(JP.BG, 3, labels=False)
TOPI, BOTI = set(TER.index[TER == 2]), set(TER.index[TER == 0])
PTR = {}
for r_ in g.itertuples():
    tr_ = gtraces[(r_.game_id, r_.play_id, r_.nfl_id)]
    PTR.setdefault((r_.route_ran, r_.nfl_id), []).append(speed_trace(tr_[0], tr_[1], r_.i_apex))
PMED = {k: np.nanmedian(np.stack(v), axis=0) for k, v in PTR.items()}
VAP = g.groupby(["route_ran", "nfl_id"]).v_apex.median()
FILM3 = {}
fig, axs = plt.subplots(2, 5, figsize=(FIGW, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[2.1, 1.0]))
fig.subplots_adjust(left=0.07, right=0.985, top=0.75, bottom=0.12, wspace=0.10, hspace=0.12)
for fi_, fam in enumerate(FAMILIES):
    ax, axd = axs[0, fi_], axs[1, fi_]
    res_, st3_ = {}, {}
    for tg_, ids_, col_ in (("top", TOPI, OI["verm"]), ("bot", BOTI, OI["blue"])):
        keys_ = [(fam, i) for i in ids_ if (fam, i) in PMED]
        st_ = np.stack([PMED[k] for k in keys_])
        st3_[tg_] = st_
        bs_ = np.stack([np.nanmedian(st_[RNG.integers(0, len(st_), len(st_))], axis=0) for _ in range(500)])
        lo_, hi_ = np.nanpercentile(bs_, 5, axis=0), np.nanpercentile(bs_, 95, axis=0)
        ax.fill_between(TT, lo_, hi_, color=col_, alpha=0.12, lw=0)
        ax.plot(TT, lo_, color=col_, lw=0.8, alpha=0.8); ax.plot(TT, hi_, color=col_, lw=0.8, alpha=0.8)
        ax.plot(TT, np.nanmedian(st_, axis=0), color=col_, lw=2.6)
        res_[tg_] = VAP.loc[fam].reindex([k[1] for k in keys_]).to_numpy(float)
    a_, b_ = res_["top"], res_["bot"]
    dd_ = [RNG.choice(a_, len(a_)).mean() - RNG.choice(b_, len(b_)).mean() for _ in range(NBOOT)]
    FILM3[fam] = (a_.mean() - b_.mean(), pct(dd_, 5), pct(dd_, 95), len(a_), len(b_))
    sa_, sb_ = st3_["top"], st3_["bot"]
    dtr_ = np.stack([np.nanmedian(sa_[RNG.integers(0, len(sa_), len(sa_))], axis=0)
                     - np.nanmedian(sb_[RNG.integers(0, len(sb_), len(sb_))], axis=0) for _ in range(500)])
    axd.fill_between(TT, np.nanpercentile(dtr_, 5, axis=0), np.nanpercentile(dtr_, 95, axis=0), color="#999999",
                     alpha=0.35, lw=0)
    axd.plot(TT, np.nanmedian(sa_, axis=0) - np.nanmedian(sb_, axis=0), color="black", lw=2.0)
    axd.axhline(0, color="#666666", lw=0.9)
    axd.set_ylim(-3, 3); axd.set_yticks([-2, 0, 2])
    for a__ in (ax, axd):
        a__.axvline(0, color="#CCCCCC", lw=1.0, zorder=0)
    ax.set_title(f"{fam.capitalize()}\ntop {len(a_)} vs bottom {len(b_)}", fontsize=12.5)
    ax.text(0.5, 0.03, f"gap at the turn\n{FILM3[fam][0]:+.1f} [{FILM3[fam][1]:+.1f}, {FILM3[fam][2]:+.1f}]"
            .replace("-", "−"), transform=ax.transAxes, ha="center", va="bottom", fontsize=11,
            bbox=dict(fc="white", ec="#CCCCCC", boxstyle="round,pad=0.25"))
    ax.set_xlim(-1.0, 1.0); ax.set_ylim(0, VMAX); ax.set_xticks([-0.5, 0, 0.5])
    if fi_ == 0:
        ax.set_ylabel("speed, yd/s", fontsize=12)
        axd.set_ylabel("top − bottom,\nyd/s", fontsize=11.5)
    else:
        ax.set_yticklabels([]); axd.set_yticklabels([])
fig.supxlabel("seconds from the slowest point of the cut (NFL snaps)", fontsize=12, y=0.02)
fig.text(0.01, 0.975, "Hard and soft Combine cutters run the same NFL cut", fontsize=15.5, fontweight="bold", va="top")
fig.text(0.01, 0.915, "Top row: orange = top third of Combine Break G (hard drill cutters), blue = bottom third; line = "
         "median of each receiver's\nmedian NFL speed trace, outlined band = 90% interval over receivers. Bottom row: "
         "difference, top minus bottom third, with its\n90% band, which covers zero at the turn in every family. Receivers "
         "per third in titles (sampled snaps, half chosen for high EPA).", fontsize=11.5, va="top", color="#333333")
save(fig, "fig3_film.png")
fh_ = FILM3["HITCH"]
Q["f3_hitch"] = fmt(fh_[0], 1, True); CLAIM["f3_hitch"] = "hitch: speed at the top, top minus bottom third, yd/s"
Q["f3_hitch_lo"] = fmt(fh_[1], 1, True); CLAIM["f3_hitch_lo"] = "90% lower"
Q["f3_hitch_hi"] = fmt(fh_[2], 1, True); CLAIM["f3_hitch_hi"] = "90% upper"
qput("f3_ntop", fh_[3], 0, "top-third receivers with NFL hitches"); qput("f3_nbot", fh_[4], 0, "bottom third")
qput("f3_cover", sum(v[1] < 0 < v[2] for v in FILM3.values()), 0, "families whose top-speed gap interval spans 0")
for fam in FAMILIES:
    qput(f"vap_{fam.lower()}", float(g[g.route_ran == fam].v_apex.median()), 1, f"league median speed at the turn, {fam}")

# ---- Fig 4. The day-3 tiebreaker (WR and TE labelled). A: paired contrast of held-out forecast errors on all 20
# day-3 pairs. B: the pick-gap rule, SOE minus earlier pick in % of same-season contrasts ordered right, by the largest
# pick gap at which SOE is allowed to overturn the pick. C: rank-correlation gains with 90% receiver-clustered
# intervals (all rounds, day 3, WR, TE).
sl_ = pd.read_csv(OUT / "shortlist_2026.csv")
assert list(sl_.name) == list(L26.display_name)
sl_["go"] = L26.go.to_numpy()
sl_["hist"] = [", ".join(f"{int(s)}→{str(int(s) + 1)[2:]} {'hit' if h else 'miss'}" for s, h in
                         zip(HIST[HIST.display_name == n].s1, HIST[HIST.display_name == n].hit)) for n in sl_.name]
sl_.to_csv(OUT / "shortlist_2026.csv", index=False)
D3 = PAIRS[PAIRS.day3].assign(lopo=LOPO, de=lambda d: d.e0 - d.e1)
D3 = D3.sort_values("de", kind="stable").reset_index(drop=True)
twice = set(D3.pid[D3.pid.duplicated()])
fig = plt.figure(figsize=(FIGW, 9.6))
fig.text(0.01, 0.985, "On day 3, last season's SOE orders receivers better than draft slot, but not provably",
         fontsize=16, fontweight="bold", va="top")
fig.text(0.01, 0.947, f"{len(D3)} day-3 pairs from {Q['ev_d3_np']} receivers (effective n about {Q['d3_neff']}). "
         f"Gain in rank correlation over pick + routes {Q['ev_d3_drho']} [90%: {Q['ev_d3_drho_lo']}, "
         f"{Q['ev_d3_drho_hi']}];\nreceiver-clustered shuffle p = {Q['d3c_p']}, one-sided. All {Q['ev_all_n']} pairs, "
         f"every round: {Q['ev_all_drho']} [{Q['ev_all_drho_lo']}, {Q['ev_all_drho_hi']}].".replace("-", "−"),
         fontsize=12.5, va="top", color="#222222")
axA = fig.add_axes([0.255, 0.115, 0.30, 0.70])
for r, r_ in D3.iterrows():
    win_ = r_.e1 < r_.e0
    axA.plot([r_.e0, r_.e1], [r, r], color=OI["blue"] if win_ else OI["verm"], lw=2.6, zorder=1)
    axA.plot(r_.e0, r, "D", ms=8, color="#E8B04A", mec="black", zorder=3)
    axA.plot(r_.e1, r, "o", ms=8.5, color="black", zorder=4)
lab4 = [f"{str(int(r_.s1))[2:]}→{str(int(r_.s1) + 1)[2:]}  {r_.display_name} ({'TE' if r_.te == 1 else 'WR'})"
        f"{' †' if r_.pid in twice else ''}" for _, r_ in D3.iterrows()]
axA.set_yticks(range(len(D3)), lab4, fontsize=11.5)
for tl_, (_, r_) in zip(axA.get_yticklabels(), D3.iterrows()):
    if r_.te == 1:
        tl_.set_color("#8a4f00"); tl_.set_fontweight("bold")
axA.set_ylim(-0.7, len(D3) - 0.3); axA.set_xlim(0, max(D3.e0.max(), D3.e1.max()) * 1.08)
axA.set_xlabel("held-out error, inches of next-season SOE", fontsize=12)
fig.text(0.01, 0.855, f"A. SOE forecast closer on {PP['d3'][1]} of {PP['d3'][0]} pairs "
         f"(WR {PP['wr'][1]}/{PP['wr'][0]}, TE {PP['te'][1]}/{PP['te'][0]})", fontsize=13, fontweight="bold")
axA.legend(handles=[plt.Line2D([], [], color="#E8B04A", marker="D", ls="", ms=8, mec="black", label="pick + routes"),
                    plt.Line2D([], [], color="black", marker="o", ls="", ms=8.5, label="+ last season's SOE"),
                    plt.Line2D([], [], color=OI["blue"], lw=2.6, label="SOE closer"),
                    plt.Line2D([], [], color=OI["verm"], lw=2.6, label="SOE further")],
           loc="upper left", bbox_to_anchor=(-0.80, -0.075), ncol=4, frameon=False, fontsize=11.5)
fig.text(0.01, 0.012, "† receiver appears in both season pairs", fontsize=11, color="#444444")
# B: pick-gap rule curve
axB = fig.add_axes([0.68, 0.50, 0.30, 0.29])
TBc = TBCURVE.reset_index(drop=True)
xl_ = [f"< {int(gp_)}" if gp_ < 999 else "any" for gp_ in TBc.gap]
for i_, r_ in TBc.iterrows():
    prim_ = int(r_.gap) == GAP
    axB.plot([i_, i_], [r_.lo, r_.hi], color=OI["blue"], lw=3.0 if prim_ else 2.2)
    axB.plot(i_, r_.d, "o", ms=10, color=OI["blue"], mec="black")
    axB.text(i_, r_.hi + 2.5, f"{int(r_.n)}", ha="center", fontsize=11, color="#333333")
axB.axhline(0, color="#888888", lw=0.9)
axB.set_xticks(range(len(TBc)), xl_, fontsize=11.5)
axB.set_xlim(-0.5, len(TBc) - 0.5); axB.set_ylim(min(-15, TBc.lo.min() - 3), TBc.hi.max() + 9)
axB.set_xlabel("pick gap where SOE may overrule the pick", fontsize=12)
axB.set_ylabel("ordered right,\nSOE minus pick, pts", fontsize=12)
fig.text(0.60, 0.855, "B. A pick-gap threshold does not help", fontsize=13, fontweight="bold")
fig.text(0.60, 0.828, f"earlier pick alone orders {Q['tb_pick_right']}% right; numbers = contrasts", fontsize=11.5,
         color="#333333")
# C: gains
axG = fig.add_axes([0.80, 0.115, 0.18, 0.25])
GR = [(f"All rounds ({Q['ev_all_n']})", EVAL["all"]["mt"]["m1"]["rho"] - EVAL["all"]["mt"]["m0"]["rho"],
       pct(EVAL["all"]["bs"][:, 0], 5), pct(EVAL["all"]["bs"][:, 0], 95)),
      (f"Day 3 ({len(D3)})", G_FULL, pct(EVAL["d3"]["bs"][:, 0], 5), pct(EVAL["d3"]["bs"][:, 0], 95)),
      (f"Day 3, WR ({PP['wr'][0]})", GWR, pct(gwr_bs, 5), pct(gwr_bs, 95)),
      (f"Day 3, TE ({PP['te'][0]})", GTE, pct(gte_bs, 5), pct(gte_bs, 95))]
for i_, (l_, m_, lo_, hi_) in enumerate(GR):
    axG.plot([lo_, hi_], [i_, i_], color="black", lw=2.6)
    axG.plot(m_, i_, "o", ms=9, color="black")
axG.axvline(0, color="#888888", lw=0.9)
axG.set_yticks(range(len(GR)), [x[0] for x in GR], fontsize=11.5); axG.set_ylim(len(GR) - 0.5, -0.6)
axG.set_xlabel("gain in rank correlation", fontsize=12)
fig.text(0.60, 0.415, "C. Gains, 90% intervals (pairs)", fontsize=13, fontweight="bold")
save(fig, "fig4_tiebreak.png")

# ---- Fig 5. The 2026 calls. A: every name on the 2026 list, 90% interval of true SOE given 2025 (empirical Bayes),
# coloured by call, with P and the calibrated hit rate of forecasts of 60%+. B: the four act-on names' sampled-snap SOE
# by coverage and alignment, marker area = snaps, bars = +/- 1 standard error.
fig = plt.figure(figsize=(FIGW, 8.4))
fig.text(0.01, 0.98, "2026: two GO names, two HOLD; the coverage and alignment splits are too thin to call", fontsize=16,
         fontweight="bold", va="top")
fig.text(0.01, 0.935, f"P = model chance of 2026 SOE above zero. Read it as an ordering: held-out forecasts of 60%+ came "
         f"true {Q['calh_pct']}% [{Q['calh_lo']}, {Q['calh_hi']}]\nof the time, against a day-3 base rate of "
         f"{Q['lb_pct']}%.",
         fontsize=12.5, va="top", color="#222222")
L5 = L26.sort_values("pc_m1", ascending=True).reset_index(drop=True)
axP = fig.add_axes([0.20, 0.10, 0.29, 0.66])
for r, r_ in L5.iterrows():
    col_ = OI["blue"] if r_.go else (OI["orange"] if r_.act else "#9A9A9A")
    axP.plot([IN_YD * r_.post_lo, IN_YD * r_.post_hi], [r, r], color=col_, lw=3.0 if r_.act else 2.0)
    axP.plot(IN_YD * r_.y1s, r, "o", ms=10 if r_.act else 7, color=col_, mec="black")
    axP.text(IN_YD * L26.post_hi.max() + 2, r, f"{r_.pc_m1:.0f}%", va="center", fontsize=11.5,
             fontweight="bold" if r_.act else "normal")
axP.axvline(0, color="#888888", lw=0.9)
axP.set_yticks(range(len(L5)), [f"{n_} ({'TE' if p_ == 'TE' else 'WR'})" for n_, p_ in
                                zip(L5.display_name, L5.position_group)], fontsize=11.5)
for tl_, (_, r_) in zip(axP.get_yticklabels(), L5.iterrows()):
    if r_.act:
        tl_.set_fontweight("bold")
axP.set_ylim(-0.6, len(L5) - 0.4)
axP.set_xlabel("true SOE given 2025, inches (90% interval)", fontsize=12)
axP.text(IN_YD * L26.post_hi.max() + 2, len(L5) - 0.2, "P", fontsize=12, fontweight="bold", va="bottom")
fig.text(0.01, 0.815, "A. The 2026 list", fontsize=13, fontweight="bold")
axP.legend(handles=[plt.Line2D([], [], color=OI["blue"], marker="o", lw=3, mec="black", label="GO"),
                    plt.Line2D([], [], color=OI["orange"], marker="o", lw=3, mec="black", label="HOLD"),
                    plt.Line2D([], [], color="#9A9A9A", marker="o", lw=2, mec="black", label="watch")],
           loc="upper left", frameon=False, fontsize=11.5)
fig.text(0.60, 0.815, "B. The four: snap SOE by coverage, alignment", fontsize=13, fontweight="bold")
CELLS = [("man", "man"), ("zone", "zone"), ("slot", "slot"), ("wide", "wide")]
for j_, (_, a_) in enumerate(ACT.iterrows()):
    ax = fig.add_axes([0.655 + (j_ % 2) * 0.17, 0.45 - (j_ // 2) * 0.33, 0.155, 0.25])
    sm_ = gb[(gb.nfl_id == a_.nfl_id) & gb.soe.notna()]
    cm_ = {"man": sm_.man, "zone": ~sm_.man, "slot": sm_.slotp == 1, "wide": (sm_.slotp == 0) & (sm_.inline == 0)}
    for i_, (c_, lab_) in enumerate(CELLS):
        v_ = IN_YD * sm_[cm_[c_]].soe
        if len(v_) == 0:
            continue
        se_ = v_.std(ddof=1) / np.sqrt(len(v_)) if len(v_) > 1 else np.nan
        ax.plot([i_, i_], [v_.mean() - se_, v_.mean() + se_], color="#555555", lw=1.8)
        ax.scatter([i_], [v_.mean()], s=12 * len(v_), color=OI["blue"] if a_.go else OI["orange"], edgecolor="black",
                   zorder=3)
        ax.text(i_, -58, f"{len(v_)}", ha="center", fontsize=10.5, color="#333333")
    ax.axhline(0, color="#888888", lw=0.9)
    ax.set_xticks(range(4), [x[1] for x in CELLS], fontsize=10.5)
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(-62, 62); ax.set_yticks([-40, 0, 40])
    if j_ % 2:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("snap SOE, in", fontsize=11.5)
        ax.text(-0.55, -50, "n:", fontsize=10.5, color="#333333")
    ax.set_title(f"{'GO' if a_.go else 'HOLD'}  {a_.display_name.split()[-1]}", fontsize=12, loc="left",
                 color=OI["blue"] if a_.go else "#9a5b00")
fig.text(0.60, 0.012, f"{Q['sp_big_cells']} of 16 cells reach 30 snaps; half the snaps were chosen\nfor high EPA. "
         "Bars = ±1 standard error. Film prompts only.", fontsize=11, color="#333333")
save(fig, "fig5_calls.png")
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
    try:  # verify.py recomputes the headline numbers from the CSVs and exits 1 on any mismatch
        runpy.run_path(str(OUT / "verify.py"), run_name="__main__")
    except SystemExit as e_:
        if e_.code:
            raise

