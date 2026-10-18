"""Break G: what clubs pay for the Combine route-drill cut, and what it buys in NFL separation.

Round 10, candidate 2 (revision of round_09/cand_2). Run from the repo root:
    uv run python writeup/candidates/round_10/cand_2/analysis.py

Reads data/processed/*.parquet only (about two minutes). Writes numbers.json, board.csv, board_full.csv and
figures/*.png next to this file.

Metric (definition frozen in earlier rounds; identical code on Combine drills and NFL snaps):
- velocity / acceleration: Savitzky-Golay derivatives of 10 Hz x/y (window 7, order 2);
- break: first >= 30 deg heading change after 1.5 yd of travel, entered above 2.5 yd/s;
- Break G = mean acceleration magnitude (g) over the 1.0 s centred on the sharpest point of the turn;
- per rep z-scored within drill x Combine year; player score = mean over route drills, shrunk by reliability,
  reported in sd of true Break G.
The three coach components are decel into the break, speed at the apex of the break, re-acceleration out.

Tests (outcome = separation over expected, SOE):
  L1 drill trait   : Break G from one half of the route drills vs the other half.
  L2 same route    : Combine Break G vs NFL Break G on the same route families (SLANT/OUT/HITCH/IN/CORNER).
  L3 snap          : NFL Break G on a snap vs SOE on that snap, within player; plus cut-to-throw timing.
  Primary          : Combine Break G vs season SOE beyond a no-tracking baseline (log pick, 40, weight, NGS
                     athleticism, TE), with a +-3 in equivalence margin fixed in round 9.
New this round: draft price of Break G (what clubs pay) next to its payoff, catches-per-season conversion of the
bound, an 18-spec sensitivity grid over the break definition, per-class held-out signs, a false-positive case study
and a power statement for the 2026 class.
"""
from __future__ import annotations

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
    return dict(i_break=i, i_apex=apex, depth=float(cum[i]),
                angle=float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))),
                v_in=float(s[iin]), v_apex=float(s[apex]), v_out=float(s[iout]),
                decel=float((s[iin] - s[apex]) / max((apex - iin) * DT, DT)),
                reaccel=float((s[iout] - s[apex]) / max((iout - apex) * DT, DT)),
                bg=float(amag[lo:hi].mean() / G_YD))


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
    sep=np.average(d.avg_separation, weights=d.routes), tg=d.targets.sum())), include_groups=False)
D = X.join(agg, how="inner")
D["tprr"] = D.tg / D.routes
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
    bc, _ = wfit(tst, BASE + ["BG"], "ssoe")
    bcs = boot_coef(tst, BASE + ["BG"], "ssoe", n=1000)
    CLS.append(dict(lab=f"{c} class only (n={len(tst)})", b=IN_YD * bc[-1], lo=IN_YD * np.percentile(bcs, 5),
                    hi=IN_YD * np.percentile(bcs, 95), grp="class"))
    put(f"cls{c}_in", IN_YD * bc[-1], 1); put(f"cls{c}_n", len(tst)); put(f"cls{c}_gain", 100 * wr2gain(tst), 1)
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
            GRID.append(dict(turn=turn, win=2 * hw * DT, vin=vin, b=IN_YD * bb[-1], lo=IN_YD * np.percentile(bq, 5),
                             hi=IN_YD * np.percentile(bq, 95), n=len(dd), default=spec == DEFAULT_SPEC))
GRID = pd.DataFrame(GRID)
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


def design(df):
    cols = [np.ones(len(df)), df.pl, df.pl ** 2 / 100, df.cush, df.cush_na, df.man.astype(float), df.slotp, df.inline,
            df.t_throw, df.target.astype(float), df.down, df.yards_to_go]
    cols += [(df.route == r).astype(float) for r in ROUTES[1:]]
    return np.column_stack(cols)


dev = gb.season < 2025
beta_ctx = np.linalg.lstsq(design(gb[dev]), gb.loc[dev, "sep"].to_numpy(), rcond=None)[0]
gb["soe"] = gb.sep - design(gb) @ beta_ctx
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
dm = q[["nfl_id", "bg_gz", "soe", "lead"]].dropna().copy()
dm["bg_w"] = dm.bg_gz - dm.groupby("nfl_id").bg_gz.transform("mean")
dm["soe_w"] = dm.soe - dm.groupby("nfl_id").soe.transform("mean")
sr = boot_rho(dm.bg_w, dm.soe_w, cluster=dm.nfl_id)
put("snap_rho", sr[0]); put("snap_lo", sr[1]); put("snap_hi", sr[2])


def fifths(col):
    dm["k"] = pd.qcut(dm[col], 5, labels=False)
    out = []
    for k_, d_ in dm.groupby("k"):
        grp = clusters(d_.nfl_id)
        bsq = [d_.soe_w.iloc[np.concatenate([grp[i] for i in RNG.integers(0, len(grp), len(grp))])].mean()
               for _ in range(1000)]
        out.append(dict(k=k_, m=IN_YD * d_.soe_w.mean(), lo=IN_YD * np.percentile(bsq, 5),
                        hi=IN_YD * np.percentile(bsq, 95), n=len(d_), med=d_[col].median()))
    return pd.DataFrame(out)


QB = fifths("bg_w")
QL = fifths("lead")  # cut-to-throw time, fifths fixed at quintiles (not hand-picked bins)
put("lead_q1_in", QL.m.iloc[0], 1); put("lead_q2_in", QL.m.iloc[1], 1); put("lead_q5_in", QL.m.iloc[-1], 1)
put("lead_q1_s", QL.med.iloc[0], 1); put("lead_q2_s", QL.med.iloc[1], 1); put("lead_q5_s", QL.med.iloc[-1], 1)
# cluster-bootstrap difference: second fifth minus last fifth
grp_all = clusters(dm.nfl_id)
kk = pd.qcut(dm.lead, 5, labels=False).to_numpy()
sw = dm.soe_w.to_numpy()
dif = []
for _ in range(NBOOT):
    ix = np.concatenate([grp_all[i] for i in RNG.integers(0, len(grp_all), len(grp_all))])
    dif.append(IN_YD * (sw[ix][kk[ix] == 1].mean() - sw[ix][kk[ix] == 4].mean()))
put("lead_diff_in", IN_YD * (sw[kk == 1].mean() - sw[kk == 4].mean()), 1)
put("lead_diff_lo", np.percentile(dif, 5), 1); put("lead_diff_hi", np.percentile(dif, 95), 1)
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
put("lg_go_pct", 100 * (gp[gp.nfl_id.isin(cand.index)].route_ran == "GO").mean(), 0)
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


# ---- Fig 1: hero. Hardest vs softest drill cutter among WRs with 8+ graded NFL breaks.
hero_pool = cand[cand.ngame >= 8]
hero = [hero_pool.BG.idxmax(), hero_pool.BG.idxmin()]
fig = plt.figure(figsize=(FIGW, 8.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.55], left=0.06, right=0.985, top=0.85, bottom=0.17,
                      wspace=0.3, hspace=0.42)
cmap = plt.get_cmap("viridis")
nrm = plt.Normalize(0, 1.0)
for j, pid in enumerate(hero):
    r_ = D.loc[pid]
    dr = cb[cb.nfl_id == pid]
    sn = q[q.nfl_id == pid]
    axA, axB, axC = (fig.add_subplot(gs[j, k]) for k in range(3))
    for src, ax_ in ((zip(dr.event_id, dr.i_break), axA), (zip(zip(sn.game_id, sn.play_id, sn.nfl_id), sn.i_break), axB)):
        for key, i in src:
            x_, y_ = (REPS[key][0], REPS[key][1]) if ax_ is axA else gtraces[key][:2]
            qq = oriented(x_, y_, int(i))
            if qq is None:
                continue
            lo = max(0, int(i) - 10)
            a = profile(x_, y_, int(i))[10 - (int(i) - lo):][:len(qq)]
            gcolor_path(ax_, qq, np.nan_to_num(a), cmap, nrm, 2.0)
    for ax_ in (axA, axB):
        ax_.scatter([0], [0], marker="x", color="black", s=40, zorder=6)
        ax_.set_xlim(-3, 8); ax_.set_ylim(-8, 4); ax_.set_aspect("equal", adjustable="box")
        ax_.set_xticks([0, 5]); ax_.set_yticks([-5, 0])
    pdr = np.array([profile(REPS[e][0], REPS[e][1], int(i)) for e, i in zip(dr.event_id, dr.i_break)])
    pgm = np.array([profile(*gtraces[(a_, b_, c_)][:2], int(i)) for a_, b_, c_, i in
                    zip(sn.game_id, sn.play_id, sn.nfl_id, sn.i_break)])
    axC.axvspan(-0.5, 0.5, color=OI["yellow"], alpha=0.25, lw=0)
    for arr, col in ((pgm, OI["orange"]), (pdr, OI["blue"])):
        for a_ in arr:
            axC.plot(TT, a_, color=col, lw=0.6, alpha=0.25)
        axC.plot(TT, np.nanmedian(arr, axis=0), color=col, lw=3)
    axC.set_xlim(-1, 1); axC.set_ylim(0, 1.0)
    axC.set_ylabel("acceleration (g)")
    dg, gg_ = dr.bg.median(), sn.bg.median()
    put(f"hero{j}_drill_g", dg); put(f"hero{j}_game_g", gg_)
    put(f"hero{j}_ssoe_in", IN_YD * r_.ssoe, 1)
    NUM[f"hero{j}_name"] = r_.display_name
    axC.text(0.02, 0.97, f"Combine drills {dg:.2f} g  ({len(dr)} reps)", color=OI["blue"], transform=axC.transAxes,
             va="top", fontsize=11, fontweight="bold")
    axC.text(0.02, 0.83, f"NFL breaks {gg_:.2f} g  ({len(sn)} sampled snaps)", color="#9A6500",
             transform=axC.transAxes, va="top", fontsize=11, fontweight="bold")
    pick = "UDFA" if pd.isna(r_.draft_overall_pick) else f"pick {int(r_.draft_overall_pick)}"
    lab = "Hardest" if j == 0 else "Softest"
    axA.set_title(f"{lab} drill cutter: {r_.display_name} ({int(r_.draft_year)}, {pick})", loc="left", fontsize=12.5,
                  x=0.0, pad=10)
    axC.text(0.98, 0.97, f"Break G {r_.BG:+.1f} sd\nseason SOE {IN_YD * r_.ssoe:+.1f} in",
             transform=axC.transAxes, ha="right", va="top", fontsize=11)
    axA.text(0.02, 0.98, "Combine drills", transform=axA.transAxes, fontsize=11, va="top", color=OI["blue"],
             fontweight="bold")
    axB.text(0.02, 0.98, "NFL breaks", transform=axB.transAxes, fontsize=11, va="top", color="#9A6500",
             fontweight="bold")
    if j == 1:
        axA.set_xlabel("yd (stem up, cut at x)", fontsize=11)
        axB.set_xlabel("yd (stem up, cut at x)", fontsize=11)
        axC.set_xlabel("seconds from the sharpest point of the cut")
put("hero_drill_gap", float(NUM["hero0_drill_g"]) - float(NUM["hero1_drill_g"]))
put("hero_game_gap", float(NUM["hero0_game_g"]) - float(NUM["hero1_game_g"]))
cax = fig.add_axes([0.06, 0.075, 0.28, 0.02])
cb_ = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cmap), cax=cax, orientation="horizontal")
cb_.set_label("path colour = acceleration (g)", fontsize=11)
fig.legend(handles=[plt.Line2D([], [], color=OI["blue"], lw=3, label="Combine drill reps (median bold)"),
                    plt.Line2D([], [], color=OI["orange"], lw=3, label="NFL slant/out/hitch/in/corner breaks"),
                    plt.Rectangle((0, 0), 1, 1, color=OI["yellow"], alpha=0.4, label="1.0 s Break G window")],
           loc="lower right", ncol=1, frameon=False, fontsize=11, bbox_to_anchor=(0.99, -0.005))
fig.suptitle(f"A {NUM['hero_drill_gap']} g drill gap between two receivers shrinks to {NUM['hero_game_gap']} g on NFL breaks",
             x=0.01, ha="left", fontsize=14, fontweight="bold", y=0.975)
fig.text(0.01, 0.915, f"Each path is ±1 s around one cut, seen from above. Median drill cuts differ by "
         f"{NUM['hero_drill_gap']} g; median NFL cuts by {NUM['hero_game_gap']} g.", fontsize=11.5, ha="left",
         color="#333333")
save(fig, "fig1_hero.png")

# ---- Fig 2: the chain, 2x2. Links 1-2 (does the drill trait reach NFL breaks?) and 3 (what opens a receiver?)
fig, axs = plt.subplots(2, 2, figsize=(FIGW, 8.4))
ax = axs[0, 0]
ax.scatter(HH_BG.a, HH_BG.b, s=22, color=OI["blue"], alpha=0.7, edgecolor="none")
ax.set_xlabel("Break G, drill set A (z)"); ax.set_ylabel("Break G, drill set B (z)")
ax.set_title("1. Drill trait repeats across drills", loc="left", fontsize=12.5)
ax.set_ylim(-2.5, 2.5)
ax.text(0.03, 0.97, f"ρ = {REL['bg'][0]:.2f} [{REL['bg'][1]:.2f}, {REL['bg'][2]:.2f}]\nn = {REL['bg'][3]} players",
        transform=ax.transAxes, va="top", ha="left", fontsize=11,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5), zorder=10)
ax = axs[0, 1]
ax.scatter(JP.BG, JP.g_bg, s=22, color=OI["orange"], alpha=0.8, edgecolor="none")
ax.set_xlabel("Combine Break G (sd)"); ax.set_ylabel("NFL Break G, same routes (z)")
ax.set_title("2. ...but not on the same NFL routes", loc="left", fontsize=12.5)
ax.set_ylim(-1.25, 1.0)
ax.text(0.03, 0.03, f"ρ = {pr[0]:.2f} [{pr[1]:.2f}, {pr[2]:.2f}], n = {pr[3]} players\n"
        f"ceiling if same skill ≈ {NUM['ceiling']}", transform=ax.transAxes, va="bottom", fontsize=11,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5), zorder=10)
for ax, QQ, col, ttl, xl, labs in (
        (axs[1, 0], QB, OI["verm"], "3a. A harder NFL cut does not open him", "his own NFL cuts by force (fifths)",
         ["softest", "2", "3", "4", "hardest"]),
        (axs[1, 1], QL, OI["green"], "3b. Cut-to-throw timing does (exploratory)",
         "time from cut to throw (fifths, median s)", [f"{v:.1f}" for v in QL.med])):
    ax.axhline(0, color=OI["grey"], lw=0.8)
    ax.errorbar(QQ.k + 1, QQ.m, yerr=[QQ.m - QQ.lo, QQ.hi - QQ.m], fmt="o", color=col, ms=7, capsize=3, lw=1.6)
    ax.set_xticks([1, 2, 3, 4, 5], labs)
    ax.set_xlabel(xl); ax.set_ylabel("SOE vs his own average (in)")
    ax.set_ylim(-20, 22)
    ax.set_title(ttl, loc="left", fontsize=12.5)
axs[1, 0].text(0.03, 0.97, f"within-player ρ = {sr[0]:.2f} [{sr[1]:.2f}, {sr[2]:.2f}]\nn = {len(dm):,} snaps, "
               f"{dm.nfl_id.nunique()} players", transform=axs[1, 0].transAxes, va="top", fontsize=11)
axs[1, 1].text(0.03, 0.97, f"2nd minus 5th fifth: {NUM['lead_diff_in']} in [{NUM['lead_diff_lo']}, "
               f"{NUM['lead_diff_hi']}]\nexploratory; same {len(dm):,} snaps", transform=axs[1, 1].transAxes,
               va="top", fontsize=11)
fig.text(0.01, 0.012, "Brackets and bars: 90% bootstrap intervals; links 2 and 3 resample whole players. Ceiling = "
         "largest correlation the two measures' reliabilities allow.", fontsize=10.5, color="#333333")
fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.12, wspace=0.28, hspace=0.5)
save(fig, "fig2_chain.png")

# ---- Fig 3: what clubs pay vs what it buys
fig, ax = plt.subplots(figsize=(FIGW, 6.2))
ax.axhspan(-SESOI_IN, SESOI_IN, color=OI["sky"], alpha=0.15, lw=0)
ax.axhline(0, color=OI["grey"], lw=0.8); ax.axvline(0, color=OI["grey"], lw=0.8)
LABPOS = {"BG": (0.8, -1.0, "left"), "forty_s": (1.0, 0.5, "left"), "three_cone_s": (1.0, 1.05, "left"),
          "ngs_athleticism_score_s": (0.8, -0.95, "left"), "short_shuttle_s": (1.0, -1.45, "left"),
          "combine_weight_s": (-1.0, 0.75, "right")}
for _, r_ in PRICE.iterrows():
    isbg = r_.x == "BG"
    col = OI["verm"] if isbg else OI["blue"]
    ax.errorbar(r_.price, r_.pay, xerr=[[r_.price - r_.plo], [r_.phi - r_.price]],
                yerr=[[r_.pay - r_.ylo], [r_.yhi - r_.pay]], fmt="o", color=col, ms=11 if isbg else 8,
                capsize=3, lw=2.2 if isbg else 1.3, alpha=0.95)
    dx, dy, ha = LABPOS.get(r_.x, (1.2, 0.35, "left"))
    ax.annotate(f"{r_.lab}\nn = {r_.n_price} / {r_.n_pay}", (r_.price, r_.pay), xytext=(r_.price + dx, r_.pay + dy),
                fontsize=11, ha=ha, va="center", color=col, fontweight="bold" if isbg else "normal",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))
ax.set_xlabel("what clubs pay: % earlier draft pick per +1 sd (beyond position)")
ax.set_ylabel("what it buys: season SOE,\ninches per +1 sd (beyond pick)")
ax.text(ax.get_xlim()[1], SESOI_IN - 0.2, f"±{SESOI_IN:.0f} in band: equivalence margin", ha="right", va="top",
        fontsize=10.5, color="#1F5F86")
ax.set_title("Clubs pay a little for a hard route-drill cut; it buys no separation", loc="left", fontsize=13.5)
fig.text(0.01, 0.015, "90% bootstrap intervals. n = players priced / players with NFL routes. Times sign-flipped so "
         "faster = +. Undrafted = pick 262.", fontsize=10.5, color="#333333")
fig.subplots_adjust(left=0.13, right=0.98, top=0.91, bottom=0.17)
save(fig, "fig3_price.png")

# ---- Fig 4: the bound, robust to every choice: (A) model versions + classes, (B) 18 definition specs
fig, (axA, axB) = plt.subplots(1, 2, figsize=(FIGW, 6.0), gridspec_kw=dict(width_ratios=[1.05, 1]))
rows = [dict(lab="All WR/TE, full baseline", b=coef_in, lo=lo_in, hi=hi_in, grp="main")] + CLS + COMPEFF
R = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
for ax in (axA, axB):
    ax.axvspan(-SESOI_IN, SESOI_IN, color=OI["sky"], alpha=0.15, lw=0)
    ax.axvline(0, color=OI["grey"], lw=0.8)
colmap = {"main": OI["verm"], "class": OI["blue"], "comp": OI["grey"]}
for i, r_ in R.iterrows():
    c = colmap[r_.grp]
    axA.plot([r_.lo, r_.hi], [i, i], color=c, lw=3 if r_.grp == "main" else 2)
    axA.plot(r_.b, i, "o", color=c, ms=10 if r_.grp == "main" else 7)
axA.set_yticks(range(len(R)), R.lab, fontsize=11)
axA.set_xlabel("season SOE, inches per 1 sd")
axA.set_title("A. Pooled, each class alone, each component", loc="left", fontsize=12.5)
G2 = GRID.sort_values("b").reset_index(drop=True)
for i, r_ in G2.iterrows():
    c = OI["verm"] if r_.default else OI["purple"]
    axB.plot([r_.lo, r_.hi], [i, i], color=c, lw=2.5 if r_.default else 1.4, alpha=0.9)
    axB.plot(r_.b, i, "o", color=c, ms=8 if r_.default else 5)
axB.set_yticks(range(len(G2)), [f"{int(t)}°, {w:.1f} s, {v:.1f} yd/s" for t, w, v in zip(G2.turn, G2.win, G2.vin)],
               fontsize=9.5)
axB.set_xlabel("season SOE, inches per 1 sd")
axB.set_title(f"B. {len(GRID)} Break G definitions", loc="left", fontsize=12.5)


lim = (min(R.lo.min(), G2.lo.min(), -SESOI_IN) - 0.5, max(R.hi.max(), G2.hi.max(), SESOI_IN) + 0.5)
for ax in (axA, axB):
    ax.set_xlim(*lim)
    sec = ax.secondary_xaxis("top", functions=(lambda v: v * catch_per_in, lambda v: v / catch_per_in))
    sec.set_xlabel(f"catches per {TG_SEASON}-target season", fontsize=10.5)
fig.text(0.01, 0.012, f"90% bootstrap intervals. Shaded band: ±{SESOI_IN:.0f} in equivalence margin. B labels: turn "
         f"threshold, window, entry speed; large dot = frozen definition.\nTop axis converts inches with season catch "
         f"rate (+{NUM['cr_per_yd']} pts per yard of separation when targeted).", fontsize=10.5, color="#333333")
fig.subplots_adjust(left=0.25, right=0.985, top=0.83, bottom=0.18, wspace=0.62)
save(fig, "fig4_bound.png")

# ----------------------------------------------------------------------------- write
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
(OUT / "card.json").write_text(json.dumps(CARD, indent=1, default=str))
print(json.dumps(NUM, indent=0))
print(PRICE.round(2).to_string())
print(GRID.round(2).to_string())
print(QL.round(2).to_string())
