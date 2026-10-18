"""Break G: does the Combine route-drill cut predict NFL separation? A frozen, held-out test.

Round 13 revised (revision of round_13/cand_2). Run from the repo root:
    uv run python writeup/candidates/round_13/revised/analysis.py

Reads data/processed/*.parquet only (about four minutes). Writes numbers.json, frozen_rule.json (with a SHA-256 of
the rule), board.csv, board_full.csv, card.json and figures/*.png next to this file.

Metric (definition frozen in round 9; identical code on Combine drills and NFL snaps):
- velocity / acceleration: Savitzky-Golay derivatives of 10 Hz x/y (window 7, order 2);
- break: first >= 30 deg heading change after 1.5 yd of travel, entered above 2.5 yd/s;
- Break G = mean acceleration magnitude (g) over the 1.0 s centred on the sharpest point of the turn; it is split into
  tangential (brake / drive) and normal (turning) parts for the physics check;
- per rep z-scored within drill x Combine year; player score = mean over route drills, shrunk by reliability,
  reported in sd of true Break G.

Primary test: leave-one-class-out offset slope. For each draft class, the no-tracking forecast (log pick, 40, weight,
NGS athleticism, TE) is fitted on the other two classes; on the held-out class, season SOE minus that forecast is
regressed on Break G with class intercepts (route-weighted). Interval: 2,000 class-stratified resamples that refit the
forecast inside each resample. Null: Break G shuffled within class. Margin +-3 in per sd, fixed in round 9.
New in this revision: mixed-model reliability (player, player x drill, drill x year) and Combine protocol; held-out
slope with the mixed-model score; held-out slope for all 18 definitions; timing circularity check (SOE without time
to throw, time-to-throw covariate) and a player-clustered family-wise permutation over four in-game splits;
family-wise p for the targets-per-route channel with and without Puka Nacua; forecast-error-inclusive player
intervals; scheme context (slot, tight cushion, man, go) for the decision card.
Round 12: quarterback adjustment of season SOE (passer parsed from every sampled play description, leave-player-out
empirical-Bayes QB effect); ridge forecast with leave-one-class-out penalty and per-class calibration slopes; forecast
as covariate; leave-one-player-out range; held-out tests of the draft-price and targets-per-route claims; per-player
scheme splits (man/zone, slot/wide, route) for the decision card; ledger of every test run. Figures redrawn:
fig1 GM scatter, fig2 six-player cut paths + shared-axis strips, fig3 chain with shuffle band and caterpillar,
fig4 one-strip held-out forest.
Round 12 revised: QB adjustment re-done as a crossed receiver + QB hierarchical model (REML, profile-likelihood
interval, refit on the random half of the sampled snaps); primary test without Puka Nacua; every draft slot checked
against the nflverse draft record; the draft price as a held-out pick forecast with shuffle p values, Holm over the
three price tests and Bonferroni over the whole ledger; price versus value for six Combine measures, both held out;
split-half reliability and a different-player placebo for the decision-card splits, which are then shrunk; a ranked
board of ten receivers and a rule-picked discount case. Figures: fig1 leader-line labels, fig2 drill vs NFL side by
side with median paths and readouts, fig3 two rows with an unclipped caterpillar, fig4 one row per class, fig5 price
versus value.
Round 13: the main test uses a ridge (shrunken) no-tracking forecast entered as a covariate, so its held-out
calibration is estimated rather than assumed; season SOE also adjusted for route mix and depth at the throw (sampled
snaps); Break G tested against yards per route run and catch rate over expected, with and without Puka Nacua; a
shortlist rule for day-3 / undrafted receivers scored on the following season against a base rate; figures
reordered (cut paths first), held-out forest and price-versus-value merged, chain caterpillar clipped and labelled.
"""
from __future__ import annotations

import hashlib
import json
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # small matrices: threads only add overhead
    os.environ.setdefault(_v, "1")
from pathlib import Path

import matplotlib

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
    k = "bg_posonly" if k == "bg" else k  # Break G beyond position only; the headline price is price_ho (full profile)
    put(f"price_{k}", pr_, 0); put(f"pay_{k}_in", IN_YD * bv[-1], 1)
    put(f"pay_{k}_lo", IN_YD * np.percentile(bvb, 5), 1); put(f"pay_{k}_hi", IN_YD * np.percentile(bvb, 95), 1)
    put(f"n_pay_{k}", len(dd)); put(f"n_price_{k}", len(xx))
    if xc == "BG":
        pass
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
    k = "bg_posonly" if k == "bg" else k
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

# ============================================================================= round 13 revised additions
# ---- (n) ONE primary estimate (judges, round 13): season SOE adjusted for target quality (route mix and depth at the
# throw), Break G slope on held-out classes, reported as the average of the plain forecast (OLS, offset) and the ridge
# forecast (covariate), computed on the same class-stratified resamples, so there is no choice between the two.
def ols_fc(rows, Y, W):
    fc = np.full(len(D), np.nan)
    for c in CLASSES:
        t, o = rows[cls_arr0[rows] == c], rows[cls_arr0[rows] != c]
        fc[t] = XB_[t] @ _wls(XB_[o], Y[o], W[o])
    return fc


def two_fc(ix, Y, W):
    return (ho_general(ix, BGV, Y=Y, W=W, mode="offset", fitter="ols"),
            ho_general(ix, BGV, Y=Y, W=W, mode="cov", fitter="ridge"))


COMBO = {}


def combo(key, Y, W, rows, mult=IN_YD, nb=NBOOT, nperm=NPERM // 2, nd=1):
    rows = np.asarray(rows)
    e_ = mult * np.array(two_fc(rows, Y, W))
    bs_ = mult * np.array([two_fc(strat_rows(rows), Y, W) for _ in range(nb)])
    cm_ = bs_.mean(1)
    est = float(e_.mean())
    fo, fr = ols_fc(rows, Y, W), ridge_fc(rows, Y, W)
    dmy = np.column_stack([(cls_arr0[rows] == c).astype(float) for c in CLASSES])
    nul = []
    for _ in range(nperm):
        pv_ = BGV.copy()
        for c in CLASSES:
            ix = rows[cls_arr0[rows] == c]
            pv_[ix] = BGV[RNG.permutation(ix)]
        a1 = _wls(np.column_stack([dmy, pv_[rows]]), Y[rows] - fo[rows], W[rows])[-1]
        a2 = _wls(np.column_stack([dmy, fr[rows], pv_[rows]]), Y[rows], W[rows])[-1]
        nul.append(mult * 0.5 * (a1 + a2))
    se_ = cm_.std(ddof=1)
    out = dict(b=est, lo=np.percentile(cm_, 5), hi=np.percentile(cm_, 95), se=se_, n=len(rows),
               p=(1 + (np.abs(nul) >= abs(est)).sum()) / (1 + nperm),
               ols=e_[0], ols_lo=np.percentile(bs_[:, 0], 5), ols_hi=np.percentile(bs_[:, 0], 95),
               rdg=e_[1], rdg_lo=np.percentile(bs_[:, 1], 5), rdg_hi=np.percentile(bs_[:, 1], 95))
    for k_ in ("b", "lo", "hi", "ols", "ols_lo", "ols_hi", "rdg", "rdg_lo", "rdg_hi"):
        put(f"{key}_{k_}", out[k_], nd)
    put(f"{key}_p", out["p"], 2); put(f"{key}_n", len(rows))
    COMBO[key] = out
    return out


NONAC = np.array([r for r in ALL if r not in NAC])
QP = combo("q", YMIX_, WT_, ALL)  # primary
combo("q_nn", YMIX_, WT_, NONAC)
combo("q_raw", Y_, WT_, ALL)
combo("q_qb", YQ_, WT_, ALL)
combo("q_yprr_nn", YPRR_, WT_, NONAC, mult=1.0, nd=2)
# bounded null, stated as a posterior (flat prior, normal approximation to the bootstrap) and as equivalence
se_q = QP["se"]
put("q_post_gt3_pct", 100 * norm.sf((SESOI_IN - QP["b"]) / se_q), 1)
put("q_post_in3_pct", 100 * (norm.cdf((SESOI_IN - QP["b"]) / se_q) - norm.cdf((-SESOI_IN - QP["b"]) / se_q)), 0)
put("q_post_gt0_pct", 100 * norm.sf(-QP["b"] / se_q), 0)
put("q_post_u95", QP["b"] + 1.645 * se_q, 1)
put("q_tost_p", max(norm.cdf((QP["b"] - SESOI_IN) / se_q), norm.sf((QP["b"] + SESOI_IN) / se_q)), 3)
put("q_catch_hi", QP["hi"] * catch_per_in, 2)
put("q_mde_in", (1.645 + 0.84) * se_q, 1)
# leave-one-class-out weight on the ridge forecast, per held-out class, with a class-stratified bootstrap interval
for c in CLASSES:
    def rw_(ix, c=c):
        ci = cls_arr0[ix]
        t, o = ix[ci == c], ix[ci != c]
        f_ = ridge_cv(XB_[o], YMIX_[o], WT_[o], cls_arr0[o])[0](XB_[t])
        return _wls(np.column_stack([np.ones(len(t)), f_]), YMIX_[t], WT_[t])[1]
    v0 = rw_(ALL)
    vb_ = np.array([rw_(strat_rows(ALL)) for _ in range(NBOOT // 2)])
    put(f"rw{c}", v0); put(f"rw{c}_lo", np.percentile(vb_, 5)); put(f"rw{c}_hi", np.percentile(vb_, 95))
FCM = ridge_fc(ALL, YMIX_, WT_)
put("rw_pool", _wls(np.column_stack([dmy_all, FCM, BGV]), YMIX_, WT_)[-2])
tick("combined primary done")

# ---- (o) drill to game with more NFL reps: players with 8+ graded NFL breaks only; reliabilities (split halves,
# Spearman-Brown) and the reliability-disattenuated correlation, all inside one player bootstrap
nbr_ = g.groupby("nfl_id").size()
ghv = pd.concat([g[g.half == 0].groupby("nfl_id").bg_gz.mean().rename("ga"),
                 g[g.half == 1].groupby("nfl_id").bg_gz.mean().rename("gb")], axis=1)
C8 = JP[["BG", "g_bg"]].join(ghv).join(HH_BG.rename(columns={"a": "da", "b": "db"})).join(nbr_.rename("nb"))
C8 = C8[C8.nb >= 8].dropna()


def sb_(r):
    return 2 * r / (1 + r)


def dis_(df):
    r_ = spearmanr(df.BG, df.g_bg)[0]
    rd_, rg_ = sb_(spearmanr(df.da, df.db)[0]), sb_(spearmanr(df.ga, df.gb)[0])
    return r_, rd_, rg_, float(np.clip(r_ / np.sqrt(max(rd_, 0.05) * max(rg_, 0.05)), -1, 1))


d0 = dis_(C8)
db_ = np.array([dis_(C8.iloc[RNG.integers(0, len(C8), len(C8))]) for _ in range(NBOOT)])
put("c8_n", len(C8)); put("c8_breaks_med", int(C8.nb.median()))
put("c8_rho", d0[0]); put("c8_rho_lo", np.nanpercentile(db_[:, 0], 5)); put("c8_rho_hi", np.nanpercentile(db_[:, 0], 95))
put("c8_rel_d", d0[1]); put("c8_rel_g", d0[2])
put("c8_dis", d0[3]); put("c8_dis_lo", np.nanpercentile(db_[:, 3], 5)); put("c8_dis_hi", np.nanpercentile(db_[:, 3], 95))
put("c8_ceiling", np.sqrt(max(d0[1], 0) * max(d0[2], 0)))
tick("chain 8+ done")

# ---- (p) the shortlist, judged harder. (1) time-forward: each season transition scored on its own (the rule has no
# fitted parts, so this is the leave-one-season-out score). (2) a no-SOE yardstick: next-season SOE forecast from pick,
# season-1 routes and position only, fitted on the other receivers' pairs; it shortlists the same number of receivers.
# (3) net of regression to the mean: hit ~ shortlist + that forecast, receiver-clustered bootstrap. (4) survival, taken
# from the runner-up DL candidate: did shortlisted and passed-over receivers reach 100 routes again at the same rate,
# and the hit rate when a receiver who did not is counted as a miss.
PAIRS["pick"] = players.set_index("nfl_id").draft_overall_pick.reindex(PAIRS.index).fillna(262).to_numpy()
PAIRS["lp"], PAIRS["lr1"] = np.log(PAIRS.pick), np.log(PAIRS.r1)
PAIRS["te"] = (players.set_index("nfl_id").position_group.reindex(PAIRS.index) == "TE").astype(float).to_numpy()
Apk = np.column_stack([np.ones(len(PAIRS)), PAIRS.lp, PAIRS.lr1, PAIRS.te])
fpk = np.empty(len(PAIRS))
pid_arr = PAIRS.pid.to_numpy()
for i_ in range(len(PAIRS)):
    o_ = pid_arr != pid_arr[i_]
    fpk[i_] = Apk[i_] @ np.linalg.lstsq(Apk[o_], PAIRS.y2.to_numpy()[o_], rcond=None)[0]
PAIRS["fpk"] = fpk
for tag, sub in (("d3", PAIRS[PAIRS.day3]), ("all", PAIRS)):
    for s1_ in (2023, 2024):
        t_ = sub[sub.s1 == s1_]
        put(f"tf_{tag}_{s1_}_sh", int(t_[t_.short].hit.sum())); put(f"tf_{tag}_{s1_}_sn", int(t_.short.sum()))
        put(f"tf_{tag}_{s1_}_rh", int(t_[~t_.short].hit.sum())); put(f"tf_{tag}_{s1_}_rn", int((~t_.short).sum()))
        put(f"tf_{tag}_{s1_}_ahead", int(t_[t_.short].hit.mean() > t_[~t_.short].hit.mean()))
    k_ = int(sub.short.sum())
    top_ = sub.sort_values("fpk", ascending=False).head(k_)
    put(f"pk_{tag}_hit", int(top_.hit.sum())); put(f"pk_{tag}_n", k_); put(f"pk_{tag}_pct", 100 * top_.hit.mean(), 0)
    put(f"pk_{tag}_overlap", int((top_.short).sum()))
    gq_ = clusters(sub.pid)

    def net_(df):
        A_ = np.column_stack([np.ones(len(df)), df.short.astype(float), df.fpk])
        return np.linalg.lstsq(A_, df.hit.astype(float).to_numpy(), rcond=None)[0][1]

    def pers_(df):
        A_ = np.column_stack([np.ones(len(df)), df.y1, df.fpk])
        return np.linalg.lstsq(A_, df.y2.to_numpy(), rcond=None)[0][1]

    n0, p0 = net_(sub), pers_(sub)
    nb_, pb2_, gp_ = [], [], []
    for _ in range(NBOOT):
        bb_ = sub.iloc[np.concatenate([gq_[i] for i in RNG.integers(0, len(gq_), len(gq_))])]
        if bb_.short.all() or (~bb_.short).all():
            continue
        nb_.append(net_(bb_)); pb2_.append(pers_(bb_))
        tb_ = bb_.sort_values("fpk", ascending=False).head(int(bb_.short.sum()))
        gp_.append(bb_[bb_.short].hit.mean() - tb_.hit.mean())
    put(f"net_{tag}", 100 * n0, 0); put(f"net_{tag}_lo", 100 * np.percentile(nb_, 5), 0)
    put(f"net_{tag}_hi", 100 * np.percentile(nb_, 95), 0)
    put(f"pers_{tag}", p0); put(f"pers_{tag}_lo", np.percentile(pb2_, 5)); put(f"pers_{tag}_hi", np.percentile(pb2_, 95))
    put(f"vspk_{tag}", 100 * (sub[sub.short].hit.mean() - top_.hit.mean()), 0)
    put(f"vspk_{tag}_lo", 100 * np.percentile(gp_, 5), 0); put(f"vspk_{tag}_hi", 100 * np.percentile(gp_, 95), 0)
# survival: every day-3 / UDFA receiver with 100+ routes in season 1 (2023 or 2024), whatever happened next
SV = []
for a_, b_ in ((2023, 2024), (2024, 2025)):
    s1d = ss[(ss.season == a_) & (ss.routes >= 100)].set_index("nfl_id")
    r2_ = ss[ss.season == b_].set_index("nfl_id")
    for pid_, r_ in s1d.iterrows():
        dr_ = pl_r.draft_round.get(pid_)
        if not (pd.isna(dr_) or dr_ >= 4):
            continue
        rr2 = float(r2_.routes.get(pid_, 0.0)) if pid_ in r2_.index else 0.0
        y2_ = float(r2_.ssoe.get(pid_, np.nan)) if pid_ in r2_.index else np.nan
        SV.append(dict(pid=pid_, s1=a_, short=r_.ssoe > 0, r2=rr2, y2=y2_))
SV = pd.DataFrame(SV)
SV["kept"] = SV.r2 >= 100
SV["itt"] = SV.kept & (SV.y2 > 0)
for nm_, m_ in (("s", SV.short), ("r", ~SV.short)):
    put(f"sv_{nm_}_n", int(m_.sum())); put(f"sv_{nm_}_kept", int(SV[m_].kept.sum()))
    put(f"sv_{nm_}_itt", int(SV[m_].itt.sum())); put(f"sv_{nm_}_itt_pct", 100 * SV[m_].itt.mean(), 0)
put("sv_s_kept_pct", 100 * SV[SV.short].kept.mean(), 0); put("sv_r_kept_pct", 100 * SV[~SV.short].kept.mean(), 0)
put("sv_itt_fisher_p", fisher_exact([[int(SV[SV.short].itt.sum()), int((SV.short & ~SV.itt).sum())],
                                     [int(SV[~SV.short].itt.sum()), int((~SV.short & ~SV.itt).sum())]],
                                    alternative="greater")[1], 2)
# a 2026 table flag: soft cutters are the test cases (does the rule work without a hard cut?)
put("sl26_hard", int((s25.BG >= 0).sum())); put("sl26_nobg", int(s25.BG.isna().sum()))
tick("shortlist checks done")

# ---- (q) route-family coaching for the four board receivers drawn in Figure 1, and the clip a scout pulls for the
# drill-vs-tape case. Per family: median drill Break G (g) vs median NFL Break G (g) on the same family.
FOUR = [0, 1, 8, 9]
FAMLAB = {"SLANT": "slant", "OUT": "out", "HITCH": "hitch/comeback", "IN": "in/dig", "CORNER": "corner"}
for bi in FOUR:
    pid_ = board.index[bi]
    dfm = cb[cb.nfl_id == pid_].groupby("family").bg.agg(["median", "size"])
    nfm = q[q.nfl_id == pid_].groupby("route_ran").bg.agg(["median", "size"])
    jm = dfm.join(nfm, lsuffix="_d", rsuffix="_n", how="inner")
    jm = jm[jm.size_n >= 2] if (jm.size_n >= 2).any() else jm
    put(f"card{bi}_dg", cb[cb.nfl_id == pid_].bg.median()); put(f"card{bi}_ng", q[q.nfl_id == pid_].bg.median())
    put(f"card{bi}_ngn", int((q.nfl_id == pid_).sum()))
    if len(jm):
        jm["drop"] = jm.median_d - jm.median_n
        f_ = jm["drop"].idxmax()
        NUM[f"card{bi}_fam"] = FAMLAB[f_]
        put(f"card{bi}_fam_d", jm.loc[f_, "median_d"]); put(f"card{bi}_fam_n", jm.loc[f_, "median_n"])
        put(f"card{bi}_fam_k", int(jm.loc[f_, "size_n"]))
        CARD[bi].update(fam=FAMLAB[f_], fam_d=round(float(jm.loc[f_, "median_d"]), 2),
                        fam_n=round(float(jm.loc[f_, "median_n"]), 2), fam_k=int(jm.loc[f_, "size_n"]))
    print(NUM[f"card{bi}_fam"] if f"card{bi}_fam" in NUM else "", jm.round(2).to_string())
# the clip: Boutte's softest graded NFL break on an out or hitch/comeback (the routes the coaching point is about)
gfp2 = q[(q.nfl_id == fp) & q.route_ran.isin(["OUT", "HITCH"])].sort_values("bg")
if not len(gfp2):
    gfp2 = q[q.nfl_id == fp].sort_values("bg")
cl_ = gfp2.iloc[0]
put("clip_season", int(cl_.season)); put("clip_week", int(cl_.week)); put("clip_q", int(cl_.quarter))
NUM["clip_clock"] = str(cl_.game_clock); NUM["clip_route"] = str(cl_.route_ran).lower()
put("clip_down", int(cl_.down)); put("clip_ytg", int(cl_.yards_to_go))
put("clip_vin", cl_.v_in, 1); put("clip_vapex", cl_.v_apex, 1); put("clip_kept", 100 * cl_.v_apex / cl_.v_in, 0)
put("clip_bg", cl_.bg); put("clip_sep", cl_.sep, 1)
put("clip_depth", cl_.depth, 1)
NUM["clip_team"] = str(cl_.possession_team)
put("lg_out_kept", 100 * (q[q.route_ran.isin(["OUT", "HITCH"])].v_apex / q[q.route_ran.isin(["OUT", "HITCH"])].v_in).median(), 0)
put("fp_outhitch_kept", 100 * (gfp2.v_apex / gfp2.v_in).median(), 0); put("fp_outhitch_n", len(gfp2))
print("CLIP", cl_[["season", "week", "quarter", "game_clock", "route_ran", "down", "yards_to_go", "v_in", "v_apex", "bg",
                   "sep", "depth", "play_description"]].to_dict())
tick("board extras done")


# ============================================================================= figures
TT = np.arange(-10, 11) * DT


def profile(x, y_, i):
    vx, vy, ax_, ay_ = kin(x, y_)
    a = np.hypot(ax_, ay_) / G_YD
    A = np.full(21, np.nan)
    for k in range(-10, 11):
        if 0 <= i + k < len(x):
            A[k + 10] = a[i + k]
    return A


def oriented(x, y_, i):
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


def gcolor_path(ax, qq, a, cmap, norm_, lw):
    segs = np.stack([qq[:-1], qq[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm_, linewidths=lw)
    lc.set_array(a[:len(segs)])
    ax.add_collection(lc)


MPH = 2.0455  # yd/s -> mph


def label_column(ax, pts, side, x_text, y_lo, y_hi, fs=10.5):
    """Labels stacked in a column outside the data, joined to their points by thin leader lines."""
    pts = sorted(pts, key=lambda t: t[1])
    ys = np.linspace(y_lo, y_hi, len(pts)) if len(pts) > 1 else [0.5 * (y_lo + y_hi)]
    for (x_, y_, txt), yt in zip(pts, ys):
        ax.annotate(txt, (x_, y_), xytext=(x_text, yt), textcoords="data", ha="left" if side == "r" else "right",
                    va="center", fontsize=fs, color="#111111",
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8, shrinkA=2, shrinkB=4))




# ---- Fig 2: drill vs NFL cut, side by side, for the three highest and three lowest Break G on the board
def aligned_path(x, y_, i):
    qq_ = oriented(x, y_, int(i))
    if qq_ is None:
        return None
    out = np.full((21, 2), np.nan)
    lo_ = max(0, int(i) - 10)
    out[lo_ - (int(i) - 10):lo_ - (int(i) - 10) + len(qq_)] = qq_
    return out


def draw_paths(ax, paths, col, readout, title):
    for pth in paths:
        ax.plot(pth[:, 0], pth[:, 1], color=col, lw=0.8, alpha=0.18)
    if paths:
        med = np.nanmedian(np.stack(paths), axis=0)
        ax.plot(med[:, 0], med[:, 1], color=col, lw=3.2)
        ax.plot(med[5:16, 0], med[5:16, 1], color="black", lw=1.2, ls=(0, (2, 1.5)))
    ax.plot(0, 0, "*", ms=12, color="black", zorder=6)
    ax.set_xlim(-2.5, 6.0); ax.set_ylim(-6.5, 4.5); ax.set_aspect("equal")
    ax.set_xticks([0, 4]); ax.set_yticks([-4, 0, 4])
    ax.tick_params(labelsize=10.5)
    ax.set_title(title, loc="left", fontsize=11.5, color=col, fontweight="bold")
    ax.text(0.97, 0.03, readout, transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2), zorder=8)


def readout(df):
    if not len(df):
        return "no graded breaks"
    return (f"in {MPH * df.v_in.median():.1f} mph\nslowest {MPH * df.v_apex.median():.1f}\n"
            f"out {MPH * df.v_out.median():.1f}\nBreak G {df.bg.median():.2f} g")


# ---- Fig 1 (hero): drill vs NFL cut for the two hardest and two softest drill cutters on the board. Faint lines are
# every rep; the thick line is the player's most typical rep (Break G closest to his median), coloured by acceleration
# (g) through the 1.0 s Break G window, so the metric is drawn, not only printed. Bottom: drill vs NFL Break G (sd).
CMAP = plt.get_cmap("viridis")
GNORM = matplotlib.colors.Normalize(0.2, 1.0)


def rep_profile(x, y_, i):
    pth = aligned_path(x, y_, i)
    if pth is None:
        return None, None
    vx_, vy_, ax_, ay_ = kin(x, y_)
    a_ = np.full(21, np.nan)
    for k in range(-10, 11):
        if 0 <= i + k < len(x):
            a_[k + 10] = np.hypot(ax_[i + k], ay_[i + k]) / G_YD
    return pth, a_


def draw_card(ax, reps, col, med_g, sd_, title):
    """reps: list of (path, accel, bg); the typical rep is coloured by accel inside the window."""
    for pth, _, _ in reps:
        ax.plot(pth[:, 0], pth[:, 1], color=col, lw=0.8, alpha=0.25)
    if reps:
        typ = min(reps, key=lambda r_: abs(r_[2] - med_g))
        pth, a_, _ = typ
        ax.plot(pth[:, 0], pth[:, 1], color="#444444", lw=1.6, zorder=3)
        seg = slice(5, 16)
        qq_ = pth[seg]
        ok_ = ~np.isnan(qq_).any(1)
        qq_, aa_ = qq_[ok_], a_[seg][ok_]
        segs = np.stack([qq_[:-1], qq_[1:]], axis=1)
        lc = LineCollection(segs, cmap=CMAP, norm=GNORM, linewidths=6.5, capstyle="round", zorder=4)
        lc.set_array(0.5 * (aa_[:-1] + aa_[1:]))
        ax.add_collection(lc)
    ax.plot(0, 0, "*", ms=13, color="black", zorder=6)
    ax.set_xlim(-2.5, 6.0); ax.set_ylim(-6.5, 4.0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, color=col, fontweight="bold", pad=2)
    ax.text(0.98, 0.02, f"Break G\n{med_g:.2f} g\n({sd_:+.1f} sd)", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=13, fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.0), zorder=8)


fig = plt.figure(figsize=(FIGW, 11.6))
gsP = fig.add_gridspec(2, 4, left=0.01, right=0.99, top=0.835, bottom=0.36, hspace=0.32, wspace=0.04)
CARD4 = [CARD[i] for i in FOUR]
for j, bi in enumerate(FOUR):
    pid = board.index[bi]
    c_ = CARD[bi]
    dr = cb[cb.nfl_id == pid]
    sn = q[q.nfl_id == pid]
    dreps = [(p_, a_, b_) for (p_, a_), b_ in ((rep_profile(REPS[e][0], REPS[e][1], int(i)), bgv_)
                                                for e, i, bgv_ in zip(dr.event_id, dr.i_break, dr.bg)) if p_ is not None]
    nreps = [(p_, a_, b_) for (p_, a_), b_ in ((rep_profile(*gtraces[(a1, b1, c1)][:2], int(i)), bgv_)
                                                for a1, b1, c1, i, bgv_ in zip(sn.game_id, sn.play_id, sn.nfl_id,
                                                                               sn.i_break, sn.bg)) if p_ is not None]
    pk = "UDFA" if c_["pick"] == 262 else f"pick {c_['pick']}"
    axd = fig.add_subplot(gsP[j // 2, 2 * (j % 2)])
    axn = fig.add_subplot(gsP[j // 2, 2 * (j % 2) + 1])
    gsd_ = c_["g_bg"] if c_["g_bg"] is not None else np.nan
    draw_card(axd, dreps, OI["blue"], dr.bg.median(), c_["bg"], f"Combine drill, {len(dr)} reps")
    draw_card(axn, nreps, OI["orange"], sn.bg.median() if len(sn) else np.nan, gsd_, f"NFL, {len(sn)} breaks")
    axd.text(0.0, 1.16, f"{c_['name']} ({c_['year']}, {pk}): {'hard' if j < 2 else 'soft'} drill cutter",
             transform=axd.transAxes, fontsize=13.5, fontweight="bold", ha="left", va="bottom",
             color=OI["purple"] if j < 2 else OI["green"])
    put(f"card{bi}_drill_g", dr.bg.median())
    if len(sn):
        put(f"card{bi}_nfl_g", sn.bg.median())
# colour bar for the window
cax = fig.add_axes([0.06, 0.255, 0.30, 0.022])
cb_ = fig.colorbar(matplotlib.cm.ScalarMappable(norm=GNORM, cmap=CMAP), cax=cax, orientation="horizontal")
cb_.set_label("acceleration in the 1.0 s window (g)", fontsize=11.5)
cb_.ax.tick_params(labelsize=11)
fig.text(0.06, 0.205, "Thick: the player's typical rep, coloured\nthrough the 1.0 s window. Faint: every rep.\n"
         "Star: sharpest point of the turn.\nBreak G = mean of the colour, in g.", fontsize=11.5, va="top")
# dumbbell: drill vs NFL Break G in sd, same four players
axdb = fig.add_axes([0.62, 0.075, 0.35, 0.22])
names4 = [c_["name"] for c_ in CARD4]
for r, c_ in enumerate(CARD4):
    yv_ = len(CARD4) - 1 - r
    gs_ = c_["g_bg"] if c_["g_bg"] is not None else np.nan
    axdb.plot([c_["bg"], gs_], [yv_, yv_], color=OI["grey"], lw=2.5, zorder=1)
    axdb.plot(c_["bg"], yv_, "o", ms=12, color=OI["blue"], mec="black", zorder=3)
    axdb.plot(gs_, yv_, "s", ms=11, color=OI["orange"], mec="black", zorder=3)
axdb.axvline(0, color="black", lw=0.8)
axdb.set_yticks(range(len(CARD4)), names4[::-1], fontsize=12)
axdb.set_xlim(-2.0, 2.3); axdb.set_ylim(-0.6, len(CARD4) - 0.4)
axdb.set_xlabel("Break G (sd of players)", fontsize=12)
axdb.set_title("Drill (●) vs NFL (■): the gap closes", loc="left", fontsize=12.5, x=-0.45)
axdb.tick_params(axis="x", labelsize=11)
fig.suptitle("Same player, same cut: the hardest drill cutters do not stay hard in the NFL", x=0.01, ha="left",
             fontsize=15, fontweight="bold", y=0.99)
fig.text(0.01, 0.958, "Paths rotated so every rep enters straight upfield and breaks to the right (yards). "
         "1 g = his body weight pushed into the ground.\n"
         "Raw g depends on the drill or route; sd compares players on the same drills (Combine) or routes (NFL).",
         fontsize=11.5, color="#333333", va="top")
save(fig, "fig1_cards.png")

# ---- Fig 3: the chain, two panels on identical z axes; panel 2 highlights players with 8+ graded NFL breaks
JPg = JP.g_bg.to_numpy(float)
null_r = np.array([spearmanr(JP.BG, RNG.permutation(JPg))[0] for _ in range(NPERM)])
put("chain_null_lo", np.percentile(null_r, 5)); put("chain_null_hi", np.percentile(null_r, 95))
fig = plt.figure(figsize=(FIGW, 5.9))
gsA = fig.add_gridspec(1, 2, left=0.07, right=0.98, top=0.88, bottom=0.2, wspace=0.28)
LIM = 3.2
for k_, (xa, ya, col, ttl, xl, yl, rr_, cl_) in enumerate((
        (zs(HH_BG.a), zs(HH_BG.b), OI["blue"], "1. The drill cut repeats across drills",
         "Break G, drill set A (z)", "Break G, drill set B (z)", REL["bg"], None),
        (zs(JP.BG), zs(JP.g_bg), OI["orange"], "2. ...but does not carry to NFL routes",
         "Combine Break G (z)", "NFL Break G, same routes (z)", pr, float(NUM["ceiling"])))):
    ax = fig.add_subplot(gsA[0, k_])
    if cl_ is not None:
        xx_ = np.array([-LIM, LIM])
        ax.fill_between(xx_, xx_ * np.percentile(null_r, 5), xx_ * np.percentile(null_r, 95), color=OI["grey"],
                        alpha=0.25, lw=0, label="90% of random pairings")
    ax.scatter(xa, ya, s=22, color=col, alpha=0.7, edgecolor="none")
    ax.plot([-LIM, LIM], [-LIM * rr_[0], LIM * rr_[0]], color=col, lw=2.5, label="observed")
    if cl_ is not None:
        ax.plot([-LIM, LIM], [-LIM * cl_, LIM * cl_], color=OI["black"], lw=1.6, ls="--",
                label="ceiling if one shared skill")
        ax.legend(loc="lower right", fontsize=10.5, frameon=True, framealpha=0.9, edgecolor="none")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    ax.set_xlabel(xl, fontsize=11.5); ax.set_ylabel(yl, fontsize=11.5)
    ax.set_title(ttl, loc="left", fontsize=13)
    txt_ = f"ρ = {rr_[0]:.2f} [{rr_[1]:.2f}, {rr_[2]:.2f}]\nn = {rr_[3]} players"
    if cl_ is not None:
        txt_ += (f"\n{NUM['c8_n']} with 8+ NFL breaks: ρ = {float(NUM['c8_rho']):.2f}\ncorrected for noise: "
                 f"{float(NUM['c8_dis']):.2f} [{float(NUM['c8_dis_lo']):.2f}, {float(NUM['c8_dis_hi']):.2f}]")
    ax.text(0.03, 0.97, txt_, transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5), zorder=10)
fig.text(0.01, 0.02, "Each dot is one player. 90% intervals from player bootstraps. Panel 2: grey band = where 90% of "
         "random pairings fall; dashed = the most the\ntwo could agree given how noisy each measure is.",
         fontsize=10.5, color="#333333")
save(fig, "fig3_chain.png")
put("cat_n_clipped", int(((CAT.hi > 60) | (CAT.lo < -60)).sum()))

# the old held-out forest rows are kept as numbers (appendix), not drawn
RW = [dict(lab="main", b=pr13["b"], lo=pr13["lo"], hi=pr13["hi"], grp="main"),
      dict(lab="nn", **{k: P13["_nn"][k] for k in ("b", "lo", "hi")}, grp="rob"),
      dict(lab="mix", **{k: PMIX[""][k] for k in ("b", "lo", "hi")}, grp="rob"),
      dict(lab="qb", **{k: PQB[""][k] for k in ("b", "lo", "hi")}, grp="rob"),
      dict(lab="r12", b=ho, lo=np.percentile(hob, 5), hi=np.percentile(hob, 95), grp="rob"),
      dict(lab="mm", b=ho_mm, lo=np.percentile(hob_mm, 5), hi=np.percentile(hob_mm, 95), grp="rob"),
      dict(lab="defs", b=GRID.ho.median(), lo=GRID.ho.min(), hi=GRID.ho.max(), grp="range"),
      dict(lab="lopo", b=ho, lo=LOPO.min(), hi=LOPO.max(), grp="range")]
R = pd.DataFrame(RW)
put("fig3_max_hi", R.hi.max(), 1); put("fig3_min_lo", R.lo.min(), 1)
put("fig3_min_b", R[R.grp != "range"].b.min(), 1); put("fig3_max_b", R[R.grp != "range"].b.max(), 1)
put("cq_max_hi", max(v["hi"] for v in COMBO.values() if v is not COMBO["q_yprr_nn"]), 1)


# ---- Fig 2: the day-3 / undrafted shortlist on the next season. A: every qualifying pair; B: hit rates as counts,
# shortlisted vs passed over (each season transition and pooled) and the no-SOE pick forecast
def wilson(h, n, z=1.645):
    p_ = h / n
    c_ = (p_ + z * z / (2 * n)) / (1 + z * z / n)
    hw_ = z * np.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c_ - hw_), 100 * (c_ + hw_)


SLP = PAIRS.copy()
fig = plt.figure(figsize=(FIGW, 7.0))
axL = fig.add_axes([0.075, 0.15, 0.36, 0.72])
axR = fig.add_axes([0.70, 0.15, 0.22, 0.72])
y1i, y2i = IN_YD * SLP.y1, IN_YD * SLP.y2
lim_ = float(np.ceil(max(np.abs(y1i).max(), np.abs(y2i).max()) / 5) * 5 + 2)
axL.axvspan(0, lim_, color=OI["sky"], alpha=0.12, lw=0)
axL.axhline(0, color=OI["grey"], lw=0.9); axL.axvline(0, color=OI["grey"], lw=0.9)
nd3 = ~SLP.day3
axL.scatter(y1i[nd3], y2i[nd3], s=30, color="#BBBBBB", edgecolor="none", label="rounds 1-3", zorder=3)
axL.scatter(y1i[SLP.day3], y2i[SLP.day3], s=80, color=OI["blue"], edgecolor="black", lw=0.6,
            label="day 3 / undrafted", zorder=4)
LABN = {"Malik Washington": (-lim_ + 1, 31), "Demario Douglas": (-lim_ + 1, 25), "A.J. Barner": (-lim_ + 1, 19),
        "Puka Nacua": (4, -25)}
for pid_, r_ in SLP[SLP.day3].iterrows():
    if r_.display_name in LABN and r_.s1 == SLP[SLP.pid == pid_].s1.max():
        axL.annotate(f"{r_.display_name} ({int(r_.s1)}-{str(int(r_.s1) + 1)[2:]})", (IN_YD * r_.y1, IN_YD * r_.y2),
                     xytext=LABN[r_.display_name], textcoords="data", ha="left", va="center", fontsize=10.5,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.4),
                     arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8, shrinkA=0, shrinkB=5))
axL.text(lim_ - 0.8, -lim_ + 1.0, "shortlist:\nseason-1 SOE > 0", ha="right", va="bottom", fontsize=11.5,
         color="#1d4f70")
axL.set_xlim(-lim_, lim_); axL.set_ylim(-lim_, lim_)
axL.set_xlabel("season-1 SOE (in)", fontsize=12); axL.set_ylabel("next-season SOE (in)", fontsize=12)
axL.legend(loc="lower left", fontsize=10.5, frameon=True, framealpha=0.9, edgecolor="none")
axL.set_title(f"A. {len(SLP)} WR/TE season pairs (100+ routes in both)", loc="left", fontsize=12.5)
ROWS4 = [("Shortlisted, pooled", int(NUM["sl_d3_short_hit"]), int(NUM["sl_d3_short_n"]), OI["blue"], True),
         ("  2023 to 2024", int(NUM["tf_d3_2023_sh"]), int(NUM["tf_d3_2023_sn"]), OI["blue"], False),
         ("  2024 to 2025", int(NUM["tf_d3_2024_sh"]), int(NUM["tf_d3_2024_sn"]), OI["blue"], False),
         ("Passed over, pooled", int(NUM["sl_d3_rest_hit"]), int(NUM["sl_d3_rest_n"]), OI["grey"], True),
         ("  2023 to 2024", int(NUM["tf_d3_2023_rh"]), int(NUM["tf_d3_2023_rn"]), OI["grey"], False),
         ("  2024 to 2025", int(NUM["tf_d3_2024_rh"]), int(NUM["tf_d3_2024_rn"]), OI["grey"], False),
         ("Pick + routes forecast,\nsame number picked", int(NUM["pk_d3_hit"]), int(NUM["pk_d3_n"]), OI["orange"], True)]
ypos = [0, 0.8, 1.6, 2.8, 3.6, 4.4, 5.8]
for yp, (lab, h_, n_, col, main_) in zip(ypos, ROWS4):
    if main_:
        lo_, hi_ = wilson(h_, n_)
        axR.plot([lo_, hi_], [yp, yp], color=col, lw=2.0, alpha=0.5, solid_capstyle="round")
    axR.plot(100 * h_ / n_, yp, "o", ms=12 if main_ else 8, color=col, mec="black")
    axR.text(103, yp, f"{h_} of {n_}", va="center", fontsize=12 if main_ else 11,
             fontweight="bold" if main_ else "normal", clip_on=False)
axR.axvline(float(NUM["sl_d3_base_pct"]), color="black", ls="--", lw=1.3)
axR.text(float(NUM["sl_d3_base_pct"]) + 1.5, 6.55, f"base rate {NUM['sl_d3_base_pct']}%", fontsize=11)
axR.set_yticks(ypos, [r_[0] for r_ in ROWS4], fontsize=11.5)
for t_, r_ in zip(axR.get_yticklabels(), ROWS4):
    if r_[4]:
        t_.set_fontweight("bold")
axR.set_xlim(0, 100); axR.set_ylim(6.9, -0.6)
axR.set_xlabel("% with next-season SOE > 0", fontsize=12)
axR.set_title(f"B. Day 3 / undrafted only ({NUM['sl_d3_n']} pairs)", loc="left", fontsize=12.5, x=-0.95)
fig.text(0.01, 0.015, "Rule fixed in advance, nothing fitted: shortlist if season-1 SOE is above zero; the next season "
         "is never seen by the rule.\nB: faint bars = 90% Wilson intervals. "
         f"Inside the day-3 shortlist, soft drill cutters hit {NUM['sl_soft_hit']} of {NUM['sl_soft_n']} and hard "
         f"cutters {NUM['sl_hard_hit']} of {NUM['sl_hard_n']}.",
         fontsize=10.5, color="#333333")
save(fig, "fig2_shortlist.png")


# ---- the full ledger of tests run, for the table next to the headline
# counted as: 1 primary + 8 versions of it (Fig 4 rows that are intervals, incl. in sample) + 18 definitions + Break G
# and 3 coach pieces + 3 usage outcomes + 3 Break G price tests + 10 price/value tests of the other 5 measures + 4
# within-player splits + 3 split-half reliabilities + 3 chain links
LEDGER = {"primary": 1, "robust": 8, "defs": len(GRID), "comps": 4, "usage": 3, "price": 3, "othermeas": 10,
          "splits": 4, "splitrel": 3, "chain": 3,
          # round 13: ridge primary, its no-Nacua, mix and QB versions (each with and without Nacua) = 6; yards per
          # route and catch rate over expected, with and without Nacua = 4; shortlist (all rounds, day 3, soft vs hard) = 3
          "r13_versions": 6, "r13_outcomes": 4, "r13_shortlist": 3}
for k_, v_ in LEDGER.items():
    put(f"ledger_{k_}", v_)
put("ledger_total", sum(LEDGER.values()))
put("rob_max_hi", max(max(r["hi"] for r in ROB.values()), np.percentile(hob_mm, 95), np.percentile(hnb, 95)), 1)
VERS = [ROB[k_] for k_ in ROB] + [dict(b=ho_mm, hi=np.percentile(hob_mm, 95)), dict(b=ho_nn, hi=np.percentile(hnb, 95)),
                                   dict(b=coef_in, hi=hi_in)]
put("ver_n", len(VERS)); put("ver_min_b", min(v["b"] for v in VERS), 1); put("ver_max_b", max(v["b"] for v in VERS), 1)
put("ver_max_hi", max(v["hi"] for v in VERS), 1)
put("pick_gain_p_pct", 100 * p_gain, 1); put("price_ho_perm_p_pct", 100 * p_slope, 1)
put("price_holm_gain_pct", 100 * holm[2], 0); put("price_holm_slope_pct", 100 * holm[1], 0)
put("split_min_fwp", min(float(NUM[f"split_{k_}_fwp"]) for k_ in SPLITS), 2)
# the same p values written as percentages, as quoted in the text
put("split_min_fwp_pct", 100 * float(NUM["split_min_fwp"]), 0); put("sec_fwer_pct", 100 * float(NUM["sec_fwer_p"]), 0)
put("ho_shuffle_pct", 100 * float(NUM["ho_shuffle_p"]), 0); put("mm_vp_pct", 100 * VP, 0)
for j_, c_ in enumerate(CARD):
    put(f"card{j_}_pick", int(c_["pick"]))
put("price_bonf_ledger", min(1.0, sum(LEDGER.values()) * p_gain), 2)
put("price_bonf_ledger_slope", min(1.0, sum(LEDGER.values()) * p_slope), 2)
put("rob_min_b", min(min(r["b"] for r in ROB.values()), ho_mm), 1)
put("rob_max_b", max(max(r["b"] for r in ROB.values()), ho_mm), 1)

# ----------------------------------------------------------------------------- write
for j_, c_ in enumerate(CARD):
    if "n_reach" in c_:
        put(f"card{j_}_reach", c_["n_reach"])
    put(f"card{j_}_nbreaks", c_["n_breaks"])
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
tick("figures done")
(OUT / "card.json").write_text(json.dumps(CARD, indent=1, default=str))
pd.DataFrame(CARD).to_csv(OUT / "board.csv", index=False)
print(pd.DataFrame(CARD).to_string())
print(json.dumps(NUM, indent=0))
print(PRICE.round(2).to_string())
print(GRID.round(2).to_string())
print(QL.round(2).to_string())
print(ITEM.round(2).to_string())
