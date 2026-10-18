"""Break G: does the Combine route-drill cut predict NFL separation? A frozen, held-out test.

Round 11, revised (from round_11/cand_2). Run from the repo root:
    uv run python writeup/candidates/round_11/revised/analysis.py

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
"""
from __future__ import annotations

import hashlib
import json
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
NBOOT = 2000
NPERM = 2000
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
    base = dict(game_id=k[0], play_id=k[1], nfl_id=k[2], t_throw=(ipf - i0) * DT, ipf=ipf)
    r = break_mechanics(d.x.to_numpy(), d.y.to_numpy(), i0, min(len(d), ipf + 6))
    if r:
        base.update(r)
        gtraces[k] = (d.x.to_numpy(), d.y.to_numpy(), i0, ipf)
    grows.append(base)
gb = pd.DataFrame(grows).merge(gp, on=["game_id", "play_id", "nfl_id"])
gb["sep"] = gb.separation_at_pass_forward
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

# ----------------------------------------------------------------------------- named players + case study
cnt = q.groupby("nfl_id").size()
cand = D[(D.n_drills >= 5) & (D.routes >= 200) & (D.te == 0)].copy()
cand["fc_res_in"] = IN_YD * (cand.ssoe - cand.fc0)
cand["ngame"] = cnt.reindex(cand.index).fillna(0).astype(int)
cand["g_bg"] = pg.reindex(cand.index)
top = cand.sort_values("BG", ascending=False).head(3)
bot = cand.sort_values("BG").head(3)
board = pd.concat([top, bot])
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


# ---- Fig 1: anatomy of one named cut (raw vs smoothed acceleration, braking vs turning parts)
fp_reps = cb[cb.nfl_id == fp].sort_values("bg", ascending=False)
r1 = fp_reps.iloc[0]
NUM["fig1_drill"] = r1.drill_name.replace("_", " ").title()
x1, y1, _ = REPS[r1.event_id]
ib = int(r1.i_break)
vx1, vy1, ax1, ay1 = kin(x1, y1)
s1 = np.hypot(vx1, vy1)
am1 = np.hypot(ax1, ay1) / G_YD
sw1 = np.maximum(s1, 0.3)
at1 = (vx1 * ax1 + vy1 * ay1) / sw1 / G_YD  # signed: negative = braking
an1 = np.abs(vx1 * ay1 - vy1 * ax1) / sw1 / G_YD
raw1 = np.hypot(np.gradient(np.gradient(x1, DT), DT), np.gradient(np.gradient(y1, DT), DT)) / G_YD
put("fig1_bg", r1.bg); put("fig1_rawpk", r1.raw_pk); put("fig1_pk", r1.pk)
put("fig1_vin", r1.v_in, 1); put("fig1_vapex", r1.v_apex, 1)
fig, (axA, axB) = plt.subplots(1, 2, figsize=(FIGW, 6.0), gridspec_kw=dict(width_ratios=[0.8, 1.6]))
qq1 = oriented(x1, y1, ib)
lo1 = max(0, ib - 10)
nrm = plt.Normalize(0, 1.0)
gcolor_path(axA, qq1, am1[lo1:lo1 + len(qq1)], "cividis", nrm, 5)
win = slice(ib - 5 - lo1, ib + 6 - lo1)
axA.plot(qq1[win, 0], qq1[win, 1], "o", ms=3.5, color="black", zorder=5)
axA.plot(0, 0, "*", ms=16, color=OI["verm"], zorder=6)
axA.set_aspect("equal"); axA.autoscale()
axA.set_xlabel("yards across"); axA.set_ylabel("yards upfield")
axA.set_title("A. The path, 10 Hz frames", loc="left", fontsize=12.5)
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap="cividis"), ax=axA, fraction=0.06, pad=0.03)
cbar.set_label("smoothed acceleration (g)", fontsize=10.5)
axA.text(0.3, 0.3, "dots: the 11 frames in the\n1.0 s window\nstar: sharpest turn", transform=axA.transAxes,
         fontsize=10, va="bottom")
tt1 = (np.arange(len(x1)) - ib) * DT
m1 = (tt1 >= -1.0) & (tt1 <= 1.0)
axB.axvspan(-0.5, 0.5, color=OI["yellow"], alpha=0.3, lw=0)
axB.plot(tt1[m1], raw1[m1], color=OI["grey"], lw=1.4, ls="--", marker=".", label="raw second difference (noisy)")
axB.plot(tt1[m1], am1[m1], color=OI["black"], lw=3, label=f"smoothed |a|: mean in window = Break G {r1.bg:.2f} g")
axB.plot(tt1[m1], -np.minimum(at1[m1], 0), color=OI["blue"], lw=2, label="braking part (along path)")
axB.plot(tt1[m1], np.maximum(at1[m1], 0), color=OI["green"], lw=2, ls="-.", label="driving part (along path)")
axB.plot(tt1[m1], an1[m1], color=OI["orange"], lw=2, label="turning part (v²/r, across path)")
axB.set_xlim(-1, 1); axB.set_ylim(0, max(1.6, np.nanmax(raw1[m1]) * 1.05))
axB.set_xlabel("s from sharpest point of the turn"); axB.set_ylabel("acceleration (g)")
axB.set_title("B. Smoothing keeps the shape, not the peak", loc="left", fontsize=12.5)
axB.legend(loc="upper left", fontsize=10, frameon=False)
fig.suptitle(f"One named cut: {NUM['fp_name']}, {NUM['fig1_drill']}, entry {r1.v_in:.1f} yd/s, "
             f"{r1.v_apex:.1f} yd/s at the turn", x=0.01, ha="left", fontsize=13.5, fontweight="bold")
fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.12, wspace=0.42)
save(fig, "fig1_one_cut.png")

# ---- Fig 4: six rule-picked player cards. Profiles as a ratio to the player's own drill median Break G
fig = plt.figure(figsize=(FIGW, 10.6))
outer = fig.add_gridspec(2, 3, left=0.09, right=0.985, top=0.845, bottom=0.06, wspace=0.3, hspace=0.28)
for j, (pid, r_) in enumerate(board.iterrows()):
    c_ = CARD[j]
    gs_ = outer[j // 3, j % 3].subgridspec(3, 1, height_ratios=[3.0, 1.0, 1.7], hspace=0.62)
    ax = fig.add_subplot(gs_[0]); axb = fig.add_subplot(gs_[1]); axt = fig.add_subplot(gs_[2]); axt.axis("off")
    dr = cb[cb.nfl_id == pid]
    sn = q[q.nfl_id == pid]
    den = dr.bg.median()
    pdr = np.array([profile(REPS[e][0], REPS[e][1], int(i)) for e, i in zip(dr.event_id, dr.i_break)]) / den
    pgm = (np.array([profile(*gtraces[(a_, b_, c2)][:2], int(i)) for a_, b_, c2, i in
                     zip(sn.game_id, sn.play_id, sn.nfl_id, sn.i_break)]) / den) if len(sn) else np.empty((0, 21))
    ax.axvspan(-0.5, 0.5, color=OI["yellow"], alpha=0.25, lw=0)
    ax.axhline(1, color=OI["grey"], lw=0.8, ls=":")
    md_d = np.nanmedian(pdr, axis=0)
    ax.plot(TT, md_d, color=OI["blue"], lw=3)
    reached = False
    if len(pgm):
        md_g = np.nanmedian(pgm, axis=0)
        ax.plot(TT, md_g, color=OI["orange"], lw=3)
        reached = np.nanmax(md_g) >= np.nanmax(md_d)
        kx = int(np.nanargmax(md_g))
        ax.plot(TT[kx], md_g[kx], "^", ms=10, color=OI["orange"], mec="black",
                mfc=OI["orange"] if reached else "white", zorder=5)
    n_reach = int((sn.pk >= dr.pk.median()).sum())
    CARD[j]["n_reach"] = n_reach
    ax.set_xlim(-1, 1); ax.set_ylim(0, 1.8)
    ax.tick_params(labelsize=10)
    if j % 3 == 0:
        ax.set_ylabel("× his drill median", fontsize=10.5)
    ax.set_xlabel("s from sharpest point of cut", fontsize=10, labelpad=1)
    pk = "UDFA" if c_["pick"] == 262 else f"#{c_['pick']}"
    pif_ = "UDFA" if c_["pick_if_paid"] >= 262 else f"#{c_['pick_if_paid']}"
    ax.set_title(f"{c_['name']}, Break G {c_['bg']:+.1f} sd\n{c_['year']} {pk} · if paid {pif_}", loc="left",
                 fontsize=11)
    comp = [c_["decel"], c_["vap"], c_["reacc"]]
    axb.barh([2, 1, 0], comp, color=[OI["blue"], OI["purple"], OI["green"]], height=0.7)
    axb.axvline(0, color="black", lw=0.8)
    axb.set_xlim(-2, 2); axb.set_yticks([2, 1, 0], ["decel", "speed", "re-acc"], fontsize=10)
    axb.tick_params(axis="x", labelsize=10)
    axb.set_xlabel("component (sd)", fontsize=10, labelpad=1)
    beat = c_["res_in"] > 0
    sym, col = ("▲", OI["blue"]) if beat else ("▼", OI["verm"])
    axt.text(0, 1.0, f"{sym} vs forecast {c_['res_in']:+.1f} in [{c_['res_lo']:+.1f}, {c_['res_hi']:+.1f}]",
             color=col, fontweight="bold", fontsize=10.5, va="top", transform=axt.transAxes)
    axt.text(0, 0.66, f"NFL peak ≥ drill peak on {n_reach} of {len(sn)} breaks\n"
             f"slot {c_['slot_pct']}% · tight cushion {c_['tight_pct']}%\nman {c_['man_pct']}% · go routes {c_['go_pct']}%",
             fontsize=10, va="top", transform=axt.transAxes, linespacing=1.35)
fig.legend(handles=[plt.Line2D([], [], color=OI["blue"], lw=3, label="Combine route drills (median)"),
                    plt.Line2D([], [], color=OI["orange"], lw=3, label="sampled NFL breaks, same families (median)"),
                    plt.Line2D([], [], marker="^", ls="", color=OI["orange"], mec="black", ms=10,
                               label="NFL peak: filled = reached drill peak, open = did not")],
           loc="lower center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Six rule-picked receivers: the hardest drill cutters do not keep the edge on NFL breaks",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=0.985)
fig.text(0.01, 0.955, f"Three highest and three lowest Break G among {NUM['n_card_pool']} WRs with 5+ route drills and "
         "200+ NFL routes. Curves are divided by his own median drill\nBreak G, so shape, not level, is compared. "
         "▲ / ▼: beat / fell short of the no-tracking forecast (90% interval includes forecast error).\n"
         "Bars: Break G's three coach components. Scheme shares from his sampled snaps (tight cushion = 3 yd or less).",
         fontsize=10, color="#333333", va="top")
save(fig, "fig4_cards.png")

# ---- Fig 2: the chain, 1 x 3. Panels 1 and 2 on identical z axes; panel 3 merges two within-player splits
fig, axs = plt.subplots(1, 3, figsize=(FIGW, 5.4), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
LIM = 3.2
for ax, xa, ya, col, ttl, xl, yl, rr_, cl_ in (
        (axs[0], zs(HH_BG.a), zs(HH_BG.b), OI["blue"], "1. Drill trait repeats", "Break G, drill set A (z)",
         "Break G, drill set B (z)", REL["bg"], None),
        (axs[1], zs(JP.BG), zs(JP.g_bg), OI["orange"], "2. ...but not on NFL routes", "Combine Break G (z)",
         "NFL Break G, same routes (z)", pr, float(NUM["ceiling"]))):
    ax.scatter(xa, ya, s=20, color=col, alpha=0.7, edgecolor="none")
    ax.plot([-LIM, LIM], [-LIM, LIM], color=OI["grey"], lw=1, ls=":")
    ax.plot([-LIM, LIM], [-LIM * rr_[0], LIM * rr_[0]], color=col, lw=2.5)
    if cl_ is not None:
        ax.plot([-LIM, LIM], [-LIM * cl_, LIM * cl_], color=OI["black"], lw=1.6, ls="--")
        ax.text(LIM * 0.95, LIM * cl_ * 0.95 + 0.25, "ceiling if\nsame skill", ha="right", fontsize=10)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
    ax.set_title(ttl, loc="left", fontsize=12.5)
    ax.text(0.03, 0.97, f"ρ = {rr_[0]:.2f} [{rr_[1]:.2f}, {rr_[2]:.2f}]\nn = {rr_[3]} players", transform=ax.transAxes,
            va="top", fontsize=10.5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5), zorder=10)
ax = axs[2]
ax.axhline(0, color=OI["grey"], lw=0.8)
for QQ, col, lab, off, mk in ((QB, OI["verm"], f"cut force (fw p = {NUM['split_force_fwp']})", -0.08, "o"),
                              (QL, OI["blue"], f"cut-to-throw time (fw p = {NUM['split_time_fwp']})", 0.08, "s")):
    ax.errorbar(QQ.k + 1 + off, QQ.m, yerr=[QQ.m - QQ.lo, QQ.hi - QQ.m], fmt=mk + "-", color=col, ms=7, capsize=3,
                lw=1.6, label=lab)
ax.set_xticks([1, 2, 3, 4, 5], ["lowest", "2", "3", "4", "highest"])
ax.set_xlabel("fifth of his own NFL breaks (rank within player)", fontsize=11)
ax.set_ylabel("SOE vs his own average (in)", fontsize=11)
ax.set_ylim(-22, 22)
ax.set_title("3. Within player (post hoc splits)", loc="left", fontsize=12.5)
ax.legend(loc="lower left", fontsize=10, frameon=False)
ax.text(0.03, 0.97, f"{len(dm):,} snaps, {dm.nfl_id.nunique()} players",
        transform=ax.transAxes, va="top", fontsize=10)
fig.text(0.01, 0.015, "Dotted: 45° line (perfect agreement). Solid: observed ρ. Dashed: largest ρ the two measures' "
         "reliabilities allow.\nPanel 3: bars are 90% player-clustered bootstrap; snap SOE built without time to throw; "
         "fw p = family-wise p over 4 splits.", fontsize=10, color="#333333")
fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.2, wspace=0.32)
save(fig, "fig2_chain.png")

# ---- Fig 3: the held-out test as one forest
rows = ([dict(lab="Held out, pooled (primary)", b=ho, lo=np.percentile(hob, 5), hi=np.percentile(hob, 95), grp="main")]
        + HO_CLS
        + [dict(lab="Held out, mixed-model score", b=ho_mm, lo=np.percentile(hob_mm, 5), hi=np.percentile(hob_mm, 95),
                grp="mm"),
           dict(lab="In sample, full baseline", b=coef_in, lo=lo_in, hi=hi_in, grp="ins")] + COMPEFF
        + [dict(lab=f"{len(GRID)} alternative definitions", grp="grid")])
R = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
fig, axA = plt.subplots(figsize=(FIGW, 6.2))
axA.axvspan(-SESOI_IN, SESOI_IN, color=OI["sky"], alpha=0.15, lw=0)
axA.axvline(0, color=OI["grey"], lw=0.8)
colmap = {"main": OI["verm"], "class": OI["blue"], "mm": OI["purple"], "ins": OI["black"], "comp": OI["grey"],
          "grid": OI["purple"]}
for i, r_ in R.iterrows():
    c = colmap[r_.grp]
    if r_.grp == "grid":
        axA.fill_betweenx([i - 0.3, i + 0.3], GRID.lo.min(), GRID.hi.max(), color=c, alpha=0.15, lw=0)
        axA.fill_betweenx([i - 0.3, i + 0.3], GRID.ho.min(), GRID.ho.max(), color=c, alpha=0.55, lw=0)
        axA.text(GRID.hi.max() + 0.15, i, "dark: 18 held-out slopes\nlight: their in-sample\n90% intervals",
                 va="center", fontsize=10, color="#333333")
        continue
    axA.plot([r_.lo, r_.hi], [i, i], color=c, lw=3 if r_.grp == "main" else 2)
    axA.plot(r_.b, i, "o", color=c, ms=10 if r_.grp == "main" else 7)
axA.set_yticks(range(len(R)), R.lab, fontsize=10.5)
axA.set_xlabel("season SOE, inches per 1 sd of Break G", fontsize=11)
axA.set_title(f"Primary held-out slope {ho:+.1f} in, upper end {np.percentile(hob, 95):+.1f}: short of the "
              f"+{SESOI_IN:.0f} in that would matter", loc="left", fontsize=13, pad=10)
axA.set_xlim(min(R.lo.min(), GRID.lo.min(), -SESOI_IN) - 0.5, max(R.hi.max(), GRID.hi.max(), SESOI_IN) + 2.2)
sec = axA.secondary_xaxis(-0.2, functions=(lambda v: v * catch_per_in, lambda v: v / catch_per_in))
sec.set_xlabel(f"same scale in catches per {TG_SEASON}-target season", fontsize=10.5)
sec.tick_params(labelsize=10)
fig.text(0.01, 0.012, f"90% bootstrap intervals; pooled held-out rows refit the no-tracking forecast in every resample. "
         f"Shaded: ±{SESOI_IN:.0f} in margin fixed in round 9.", fontsize=10, color="#333333")
fig.subplots_adjust(left=0.27, right=0.985, top=0.92, bottom=0.27)
save(fig, "fig3_heldout.png")

# ----------------------------------------------------------------------------- write
for j_, c_ in enumerate(CARD):
    put(f"card{j_}_reach", c_["n_reach"]); put(f"card{j_}_nbreaks", c_["n_breaks"])
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
(OUT / "card.json").write_text(json.dumps(CARD, indent=1, default=str))
pd.DataFrame(CARD).to_csv(OUT / "board.csv", index=False)
print(pd.DataFrame(CARD).to_string())
print(json.dumps(NUM, indent=0))
print(PRICE.round(2).to_string())
print(GRID.round(2).to_string())
print(QL.round(2).to_string())
print(ITEM.round(2).to_string())
