"""Break G: Combine route-drill cut force vs NFL targets per route run (TPRR) and separation over expected.

Round 8, candidate 2. Run from the repo root:
    uv run python writeup/candidates/round_08/cand_2/analysis.py

Reads data/processed/*.parquet only (about one minute). Writes numbers.json, board.csv and figures/*.png next to
this file.

Definitions (identical code on Combine drills and NFL snaps, 10 Hz x/y):
- velocity / acceleration: Savitzky-Golay derivatives of x/y (window 7, order 2);
- break: first >= 30 deg heading change (+-0.6 s) after 1.5 yd of travel, entered above 2.5 yd/s;
- Break G = mean acceleration magnitude (g, 1 g = 10.73 yd/s^2) over the 1.0 s centred on the sharpest turn
  (braking into the cut + sideways force of the turn + drive out of it);
- per rep, z-scored within drill x Combine year; player score = mean over route drills, empirical-Bayes shrunk
  toward 0 with the per-drill reliability, reported in sd of true (error-free) Break G.
Primary outcome (fixed before modelling, a rate to replace round 5's volume outcome): career regular-season targets
per route run (TPRR), weighted by routes, for every WR/TE with at least one NFL route.
Baseline without tracking: log pick, official 40, weight, NGS athleticism (year x position standardized), TE flag.
Separation over expected (SOE): separation at the throw minus a context model fitted on 2023-24 snaps.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DT = 0.1
G_YD = 10.73
RNG = np.random.default_rng(2027)
NBOOT = 2000
NPERM = 2000
LEAD_MIN = 0.2
SEASON_ROUTES = 500  # a full-time receiver season, used only to translate a rate into targets
CLASSES = (2023, 2024, 2025)

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F", "black": "#000000"}
CLS_COL = {2023: OI["blue"], 2024: OI["orange"], 2025: OI["green"]}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 150,
                     "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10})
FIGW = 1600 / 150

NUM: dict[str, str] = {}


def put(key, val, nd=2):
    if key in NUM:
        raise KeyError(f"duplicate key {key}")
    if isinstance(val, (int, np.integer)):
        s = str(int(val))
    else:
        s = f"{float(val):.{nd}f}"
        if s.startswith("-") and float(s) == 0:
            s = s[1:]
    NUM[key] = s
    return s


def save(fig, name):
    fig.set_size_inches(FIGW, fig.get_figheight())
    fig.savefig(FIG / name, dpi=150, facecolor="white")
    plt.close(fig)


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


def break_mechanics(x, y, i0, i1, min_path=1.5, min_turn=30.0, h=6, entry_speed=2.5):
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
    lo, hi = max(0, i - 5), min(n, i + 6)
    apex = lo + int(np.argmin(s[lo:hi]))
    a0 = max(0, apex - 10)
    iin = a0 + int(np.argmax(s[a0:apex + 1]))
    b1 = min(n, apex + 11)
    iout = apex + int(np.argmax(s[apex:b1]))
    pre, post = slice(max(0, i - 8), max(1, i - 3)), slice(min(n - 1, i + 3), min(n, i + 8))
    v1 = np.array([vx[pre].mean(), vy[pre].mean()])
    v2 = np.array([vx[post].mean(), vy[post].mean()])
    cosang = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    c0, tt = cum[apex], np.arange(n) * DT
    bt = float(np.interp(c0 + 4, cum, tt) - np.interp(c0 - 4, cum, tt)) if (c0 - 4 >= cum[0] and cum[-1] >= c0 + 4) \
        else np.nan
    return dict(i_break=i, i_apex=apex, i_in=iin, depth=float(cum[i]),
                angle=float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))),
                v_in=float(s[iin]), v_apex=float(s[apex]), v_out=float(s[iout]),
                decel=float((s[iin] - s[apex]) / max((apex - iin) * DT, DT)),
                reaccel=float((s[iout] - s[apex]) / max((iout - apex) * DT, DT)),
                sk=float(100 * s[apex] / max(s[iin], 1e-6)), bg=float(amag[lo:hi].mean() / G_YD), bt=bt)


def clusters(cl):
    cl = np.asarray(cl)
    return [np.where(cl == c)[0] for c in pd.unique(cl)]


def boot_rho(a, b, cluster=None, n=NBOOT):
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
    lo, hi = np.nanpercentile(out, (2.5, 97.5))
    return float(rho), float(lo), float(hi), int(len(a))


def zs(s):
    return (s - s.mean()) / s.std(ddof=1)


COMP = ["bg", "sk", "v_in", "v_apex", "v_out", "decel", "reaccel", "angle", "bt"]
COMP_LABEL = {"bg": "Break G (force through the cut)", "sk": "Speed kept at apex", "v_in": "Entry speed",
              "v_apex": "Apex speed", "v_out": "Exit speed", "decel": "Decel into break",
              "reaccel": "Re-accel out", "angle": "Break angle", "bt": "Break time (4 yd in + 4 out)"}

# ----------------------------------------------------------------------------- Combine route drills
players = pd.read_parquet(P / "players.parquet")
cres = pd.read_parquet(P / "combine_results.parquet")
events = pd.read_parquet(P / "combine_events.parquet")
DRILL_FAMILY = {"SLANT_ROUTE_LEFT": "SLANT", "SLOT_STRIKE_ROUTE_LEFT": "SLANT", "SPEED_OUT_ROUTE_LEFT": "OUT",
                "CURL_ROUTE_RIGHT": "HITCH", "COMEBACK_ROUTE_RIGHT": "HITCH", "HOOK_ROUTE_RIGHT": "HITCH",
                "DAGGER_ROUTE_LEFT": "IN", "IN_ROUTE_LEFT": "IN", "SLOT_SAIL_ROUTE_RIGHT": "CORNER",
                "CORNER_ROUTE": "CORNER"}
FAMILIES = ["SLANT", "OUT", "HITCH", "IN", "CORNER"]
ev = events[events.drill_name.isin(DRILL_FAMILY) & events.position_group.isin(["WR", "TE"])]
frames = pd.read_parquet(P / "combine_frames.parquet", filters=[("event_id", "in", ev.event_id.tolist())])
rows, dtraces = [], {}
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
    r = break_mechanics(x, y, 0, i1)
    if r:
        rows.append(dict(event_id=eid, **r))
        dtraces[eid] = (x, y)
cb = pd.DataFrame(rows).merge(ev[["event_id", "nfl_id", "draft_year", "drill_name", "position_group"]], on="event_id")
cb["family"] = cb.drill_name.map(DRILL_FAMILY)
put("n_drill_reps", len(ev)); put("pct_drill_detected", 100 * len(cb) / len(ev), 0)
put("bg_drill_median", cb.bg.median())
for m in COMP:
    cb[m + "_z"] = cb.groupby(["drill_name", "draft_year"])[m].transform(zs)
cpd = cb.groupby(["nfl_id", "drill_name", "family"], as_index=False)[[m + "_z" for m in COMP] + ["bg"]].mean()
cplayer = cpd.groupby("nfl_id")[[m + "_z" for m in COMP]].mean()
cplayer["n_drills"] = cpd.groupby("nfl_id").size()
put("n_combine_players", len(cplayer))
put("median_drills_per_player", int(cplayer.n_drills.median()))


def icc1(df, col, gcol="nfl_id"):
    k = df.groupby(gcol).size()
    k0 = (len(df) - (k ** 2).sum() / len(df)) / (len(k) - 1)
    gm = df[col].mean()
    msb = (k * (df.groupby(gcol)[col].mean() - gm) ** 2).sum() / (len(k) - 1)
    msw = ((df[col] - df.groupby(gcol)[col].transform("mean")) ** 2).sum() / (len(df) - len(k))
    return (msb - msw) / (msb + (k0 - 1) * msw)


HALF_A = {"SLANT_ROUTE_LEFT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT", "SPEED_OUT_ROUTE_LEFT", "HOOK_ROUTE_RIGHT"}
rel_rows = []
for m in COMP:
    ha = cpd[cpd.drill_name.isin(HALF_A)].groupby("nfl_id")[m + "_z"].mean()
    hb = cpd[~cpd.drill_name.isin(HALF_A)].groupby("nfl_id")[m + "_z"].mean()
    hh = pd.concat([ha, hb], axis=1, keys=["a", "b"]).dropna()
    rc = boot_rho(hh.a, hh.b)
    rel_rows.append(dict(comp=m, rho=rc[0], lo=rc[1], hi=rc[2], n=rc[3]))
rel = pd.DataFrame(rel_rows).set_index("comp")
put("rel_bg", rel.rho["bg"]); put("rel_bg_lo", rel.lo["bg"]); put("rel_bg_hi", rel.hi["bg"]); put("rel_n", int(rel.n["bg"]))
put("rel_sk", rel.rho["sk"])
icc_bg = icc1(cpd, "bg_z")
put("icc_bg", icc_bg)
var_tot = cpd.bg_z.var()
var_true, var_err = icc_bg * var_tot, (1 - icc_bg) * var_tot
nd = cplayer.n_drills
post_var = 1 / (1 / var_true + nd / var_err)
cplayer["bg_shr"] = post_var * (nd * cplayer.bg_z / var_err)
cplayer["bg_shr_sd"] = np.sqrt(post_var)
put("rel_7drills", 7 * icc_bg / (1 + 6 * icc_bg))
SD_SHR = np.sqrt(var_true)
cplayer["BG"] = cplayer.bg_shr / SD_SHR
cplayer["BG_lo"] = (cplayer.bg_shr - 1.2816 * cplayer.bg_shr_sd) / SD_SHR
cplayer["BG_hi"] = (cplayer.bg_shr + 1.2816 * cplayer.bg_shr_sd) / SD_SHR
cplayer["bt_z"] = cplayer.bt_z.fillna(0.0)
put("g_true_sd", float(np.sqrt(var_true) * cb.groupby(["drill_name", "draft_year"]).bg.std().mean()), 3)

# ----------------------------------------------------------------------------- baseline table
cr = cres.set_index("nfl_id")
pb = players.set_index("nfl_id")[["draft_overall_pick", "position_group", "draft_year", "display_name",
                                  "latest_team"]].join(
    cr[["forty", "combine_weight", "ngs_athleticism_score", "three_cone", "short_shuttle", "ten_yd_split"]])
pb = pb[pb.position_group.isin(["WR", "TE"])].copy()
pb["logpick"] = np.log(pb.draft_overall_pick.fillna(262))
pb["te"] = (pb.position_group == "TE").astype(float)
for c in ("forty", "combine_weight", "ngs_athleticism_score", "three_cone", "short_shuttle", "ten_yd_split"):
    pb[c + "_s"] = pb.groupby(["draft_year", "position_group"])[c].transform(zs).fillna(0.0)
BASE = ["logpick", "forty_s", "combine_weight_s", "ngs_athleticism_score_s", "te"]
X = cplayer.join(pb, how="inner")
put("n_all", len(X))
put("rho_bg_forty", spearmanr(X.BG, X.forty, nan_policy="omit")[0])
put("rho_bg_pick", spearmanr(X.BG, X.draft_overall_pick.fillna(262))[0])

# ----------------------------------------------------------------------------- NFL season outcomes
ss = pd.read_parquet(P / "player_season_stats.parquet")
ss = ss[ss.position_group.isin(["WR", "TE"])].copy()
ss["slot_r"] = (ss.main_alignment == "SLOT_WR").astype(float) * ss.routes
ss["sep_r"] = ss.avg_separation * ss.routes
tot = ss.groupby("nfl_id").agg(tg=("targets", "sum"), routes=("routes", "sum"), yds=("rec_yards", "sum"),
                               slot_r=("slot_r", "sum"), sep_r=("sep_r", "sum"),
                               sep_w=("routes", lambda s: s[ss.loc[s.index, "avg_separation"].notna()].sum()))
X = X.join(tot)
X["routes"] = X.routes.fillna(0.0)
X["any_route"] = (X.routes > 0).astype(float)
put("n_no_route", int((X.routes == 0).sum()))
D = X[X.routes > 0].copy()  # primary analysis set
D["tprr"] = D.tg / D.routes
D["yprr"] = D.yds / D.routes
D["slot"] = D.slot_r / D.routes
D["ngs_sep"] = D.sep_r / D.sep_w.replace(0, np.nan)
put("n_main", len(D)); put("n_main_wr", int((D.te == 0).sum())); put("n_main_te", int((D.te == 1).sum()))
for c in CLASSES:
    put(f"n_main_{c}", int((D.draft_year == c).sum()))
put("tprr_mean", 100 * np.average(D.tprr, weights=D.routes), 1)
put("routes_median", int(D.routes.median()))


def wfit(df, cols, ycol, wcol="routes"):
    """Weighted least squares; returns coef, se."""
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


def wresid(df, cols, v, w):
    A = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    W = np.sqrt(w)
    b = np.linalg.lstsq(A * W[:, None], v * W, rcond=None)[0]
    return v - A @ b


# ---- headline: TPRR ~ baseline + Break G, weighted by routes
b, se = wfit(D, BASE + ["BG"], "tprr")
coef = b[-1]
put("coef_pp", 100 * coef); put("t_main", coef / se[-1])
bs = []
for _ in range(NBOOT):
    ix = RNG.integers(0, len(D), len(D))
    bs.append(wfit(D.iloc[ix], BASE + ["BG"], "tprr")[0][-1])
bs = np.array(bs)
put("coef_pp_lo", 100 * np.percentile(bs, 2.5)); put("coef_pp_hi", 100 * np.percentile(bs, 97.5))
put("coef_pos_pct", 100 * (bs > 0).mean(), 0)

# ---- screen: 9 Combine components x 4 outcomes; family-wise p by within-class permutation
OUTS = {"tprr": "routes", "yprr": "routes", "ngs_sep": "sep_w", "psoe": "n_soe"}
SCREEN_X = [m + "_z" for m in COMP if m != "bg"] + ["BG"]

# player SOE from graded break snaps is computed below from game frames; screen runs after it.

# ---- out-of-class: leave one Combine class out
D["fc0"] = np.nan
D["fc1"] = np.nan
for c in CLASSES:
    trn, tst = D[D.draft_year != c], D[D.draft_year == c]
    D.loc[tst.index, "fc0"] = wpred(trn, tst, BASE, "tprr")
    D.loc[tst.index, "fc1"] = wpred(trn, tst, BASE + ["BG"], "tprr")


def wr2gain(df):
    w, yv = df.routes.to_numpy(), df.tprr.to_numpy()
    return 1 - (w * (yv - df.fc1.to_numpy()) ** 2).sum() / (w * (yv - df.fc0.to_numpy()) ** 2).sum()


put("dr2_pct", 100 * wr2gain(D), 1)
gl = []
for _ in range(NBOOT):
    gl.append(wr2gain(D.iloc[RNG.integers(0, len(D), len(D))]))
gl = np.array(gl)
put("dr2_lo_pct", 100 * np.percentile(gl, 2.5), 1); put("dr2_hi_pct", 100 * np.percentile(gl, 97.5), 1)
put("dr2_pos_pct", 100 * (gl > 0).mean(), 0)
# per held-out class: coefficient fitted on the other two classes, coefficient inside the held-out class, R2 gain
cls_rows = []
for c in CLASSES:
    trn, tst = D[D.draft_year != c], D[D.draft_year == c]
    btr, _ = wfit(trn, BASE + ["BG"], "tprr")
    bte, sete = wfit(tst, BASE + ["BG"], "tprr")
    bb_ = []
    for _ in range(NBOOT):
        ix = RNG.integers(0, len(tst), len(tst))
        bb_.append(wfit(tst.iloc[ix], BASE + ["BG"], "tprr")[0][-1])
    cls_rows.append(dict(cls=c, n=len(tst), b_train=put(f"cls{c}_btrain", 100 * btr[-1]),
                         b_test=put(f"cls{c}_btest", 100 * bte[-1]),
                         lo=put(f"cls{c}_lo", 100 * np.percentile(bb_, 2.5)),
                         hi=put(f"cls{c}_hi", 100 * np.percentile(bb_, 97.5)),
                         gain=put(f"cls{c}_gain_pct", 100 * wr2gain(tst), 1),
                         b=100 * bte[-1], blo=100 * np.percentile(bb_, 2.5), bhi=100 * np.percentile(bb_, 97.5)))
CLS = pd.DataFrame(cls_rows)
put("n_cls_pos", int((CLS.b > 0).sum()))
put("n_cls_ci_pos", int((CLS.blo > 0).sum()))
# power of one ~40-player class to detect the pooled slope (normal approx., two-sided 5%)
se_cls = float(np.mean([(r_.bhi - r_.blo) / 3.92 for _, r_ in CLS.iterrows()]))
from scipy.stats import norm  # noqa: E402
put("power_one_class", norm.cdf(100 * coef / se_cls - 1.96) + norm.cdf(-100 * coef / se_cls - 1.96))

# ---- robustness (appendix)
ROB = []


def rob(label, key, df, cols, ycol="tprr", wcol="routes", x="BG"):
    bb, se_ = wfit(df, cols + [x], ycol, wcol)
    ROB.append((label, put(f"rob_{key}_pp", 100 * bb[-1]), put(f"rob_{key}_t", bb[-1] / se_[-1]), len(df)))
    put(f"rob_{key}_n", len(df))


BASE_NT = [c for c in BASE if c != "te"]
rob("Wide receivers only", "wr", D[D.te == 0], BASE_NT)
rob("Tight ends only", "te", D[D.te == 1], BASE_NT)
D["one"] = 1.0
rob("Unweighted (each player counts once)", "unw", D, BASE, wcol="one")
rob("100+ career routes", "r100", D[D.routes >= 100], BASE)
rob("+ slot share", "slot", D, BASE + ["slot"])
rob("+ entry speed (route-drill speed)", "vin", D, BASE + ["v_in_z"])
BASE2 = BASE + ["three_cone_s", "short_shuttle_s", "ten_yd_split_s"]
rob("+ 3-cone, shuttle, 10-yd split (missing = 0)", "agil", D, BASE2)
cc = D[D.three_cone.notna()]
rob("Complete cases, ran the 3-cone", "cc3", cc, BASE + ["three_cone_s"])
rob("Drafted players only", "drafted", D[D.draft_overall_pick.notna()], BASE)
rob("Unshrunk Break G", "raw", D, BASE, x="bg_z")
top = D.assign(res=D.tprr - D.fc0).sort_values("res").index[-1]
NUM["top_res_name"] = str(D.loc[top, "display_name"])
rob(f"Without the biggest over-performer ({D.loc[top, 'display_name']})", "droptop", D.drop(index=top), BASE)
loo_t = []
for i_ in D.index:
    bb, se_ = wfit(D.drop(index=i_), BASE + ["BG"], "tprr")
    loo_t.append(bb[-1] / se_[-1])
put("t_loo_min", min(loo_t))


# availability: does Break G predict reaching any NFL route? (survivorship check)
def logit(df, cols, yc, iters=50):
    Am = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    yy = df[yc].to_numpy(float)
    bb = np.zeros(Am.shape[1])
    for _ in range(iters):
        pp = 1 / (1 + np.exp(-Am @ bb))
        W = pp * (1 - pp)
        H = Am.T @ (Am * W[:, None])
        bb = bb + np.linalg.solve(H, Am.T @ (yy - pp))
    return bb, np.sqrt(np.diag(np.linalg.inv(H)))


bl_, sl_ = logit(X, BASE + ["BG"], "any_route")
put("reach_t", bl_[-1] / sl_[-1])

# ----------------------------------------------------------------------------- NFL snaps: same metric, SOE
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
put("n_break_snaps", len(g)); NUM["n_break_snaps_fmt"] = f"{len(g):,}"
g["bg_gz"] = g.groupby(["route_ran", "position_group"]).bg.transform(zs)
g["lead"] = (g.ipf - g.i_apex) * DT
g["graded"] = g.lead >= LEAD_MIN
put("pct_not_graded", 100 * (~g.graded).mean(), 0)
put("bg_game_median", g.bg.median())
q = g[g.graded].copy()
put("n_graded", len(q)); put("n_graded_players", q.nfl_id.nunique())
rs = boot_rho(q.bg_gz, q.soe, cluster=q.nfl_id)
put("snap_rho", rs[0]); put("snap_lo", rs[1]); put("snap_hi", rs[2])
# drill family -> same NFL route family
cfam = cpd.groupby(["nfl_id", "family"]).bg_z.mean()
gfam = g.groupby(["nfl_id", "route_ran"]).bg_gz.mean()
gfam.index.names = ["nfl_id", "family"]
jf = pd.concat([cfam.rename("d"), gfam.rename("g")], axis=1).dropna().reset_index()
fr = boot_rho(jf.d, jf.g, cluster=jf.nfl_id)
put("fam_rho", fr[0]); put("fam_lo", fr[1]); put("fam_hi", fr[2]); put("fam_n", fr[3])
# player SOE (graded break snaps): Combine Break G -> SOE in yards, given the baseline
psoe = q.groupby("nfl_id").agg(psoe=("soe", "mean"), n_soe=("soe", "size"))
D = D.join(psoe)
S_ = D.dropna(subset=["psoe"])
bso, seso = wfit(S_, BASE + ["BG"], "psoe", "n_soe")
put("n_soe_players", len(S_))
put("soe_coef_in", 36 * bso[-1], 1)
put("soe_lo_in", 36 * (bso[-1] - 1.96 * seso[-1]), 1); put("soe_hi_in", 36 * (bso[-1] + 1.96 * seso[-1]), 1)
put("soe_sd_in", 36 * float(S_.psoe.std()), 0)
cs = boot_rho(S_.BG, S_.psoe)
put("comb_soe_rho", cs[0]); put("comb_soe_lo", cs[1]); put("comb_soe_hi", cs[2])
# TPRR on the tracked random snaps (an independent read of target-earning from the sample)
R_ = gp[gp.sample_kind == "random"].groupby("nfl_id").agg(n_r=("target", "size"), tg_r=("target", "sum"))
D = D.join(R_)
Rr = D[D.n_r.fillna(0) > 0].copy()
Rr["trr"] = Rr.tg_r / Rr.n_r
rob("Tracked random snaps: target share per route", "rand", Rr, BASE, ycol="trr", wcol="n_r")
put("n_rand_snaps", int(Rr.n_r.sum())); NUM["n_rand_snaps_fmt"] = f"{int(Rr.n_r.sum()):,}"

# ---- screen (now that player SOE exists)
xm_full = D[SCREEN_X].to_numpy(float)
cls_arr = D.draft_year.to_numpy()


def screen_t(xm, DD=None):
    DD = D if DD is None else DD
    out = {}
    for yc, wc in OUTS.items():
        msk = DD[yc].notna().to_numpy() & DD[wc].fillna(0).gt(0).to_numpy()
        df = DD[msk]
        w = df[wc].to_numpy(float)
        ry = wresid(df, BASE, df[yc].to_numpy(float), w)
        tv = []
        for j in range(xm.shape[1]):
            rx = wresid(df, BASE, xm[msk, j], w)
            r = (w * rx * ry).sum() / np.sqrt((w * rx ** 2).sum() * (w * ry ** 2).sum())
            tv.append(r * np.sqrt(len(df) - len(BASE) - 2) / np.sqrt(1 - r ** 2))
        out[yc] = np.array(tv)
    return out


tmat = pd.DataFrame(screen_t(xm_full), index=SCREEN_X)
put("screen_k", int(tmat.size))
put("t_soe", tmat.loc["BG", "psoe"]); put("t_ngs_sep", tmat.loc["BG", "ngs_sep"]); put("t_yprr", tmat.loc["BG", "yprr"])
best = tmat.abs().stack().idxmax()
NUM["screen_best"] = f"{best[0]}|{best[1]}"


def perm_within_class():
    perm = np.arange(len(D))
    for c in CLASSES:
        ix = np.where(cls_arr == c)[0]
        perm[ix] = RNG.permutation(ix)
    return perm


maxt = []
for _ in range(NPERM):
    Tp = screen_t(xm_full[perm_within_class()])
    maxt.append(max(np.abs(v).max() for v in Tp.values()))
put("p_fwer", (1 + (np.array(maxt) >= abs(tmat.loc["BG", "tprr"])).sum()) / (1 + NPERM), 3)
shrink = max(0.0, 1 - 1 / float((tmat.to_numpy() ** 2).mean()))
put("eb_shrink", shrink)
put("plan_pp", 100 * coef * shrink, 1)
put("plan_tg", coef * shrink * SEASON_ROUTES, 0)
put("plan_tg_raw", coef * SEASON_ROUTES, 0)
rank_bg = int((tmat.abs().to_numpy() > abs(tmat.loc["BG", "tprr"])).sum()) + 1
put("rank_bg_full", rank_bg)

# ---- nested screen: run the whole 36-pair screen on two classes, carry the winner to the third
NEST = []
for c in CLASSES:
    trn, tst = D[D.draft_year != c], D[D.draft_year == c]
    tm = pd.DataFrame(screen_t(trn[SCREEN_X].to_numpy(float), trn), index=SCREEN_X)
    xb, yb = tm.abs().stack().idxmax()
    wc = OUTS[yb]
    tr_ = trn[trn[yb].notna() & trn[wc].fillna(0).gt(0)]
    te_ = tst[tst[yb].notna() & tst[wc].fillna(0).gt(0)]
    p0, p1 = wpred(tr_, te_, BASE, yb, wc), wpred(tr_, te_, BASE + [xb], yb, wc)
    w_, y_ = te_[wc].to_numpy(float), te_[yb].to_numpy(float)
    gain = 1 - (w_ * (y_ - p1) ** 2).sum() / (w_ * (y_ - p0) ** 2).sum()
    rk = int((tm.abs().to_numpy() > abs(tm.loc["BG", "tprr"])).sum()) + 1
    NEST.append(dict(cls=c, pick=f"{xb}|{yb}", gain=put(f"nest{c}_gain_pct", 100 * gain, 1),
                     rank=put(f"nest{c}_bg_rank", rk)))
    NUM[f"nest{c}_pick"] = f"{COMP_LABEL[xb.replace('_z', '') if xb != 'BG' else 'bg']} -> {yb}"
put("nest_mean_gain_pct", float(np.mean([float(r_["gain"]) for r_ in NEST])), 1)

# ---- ridge alternative: all nine break measurements together, out of class (alpha chosen inside the fold)
RX = [m + "_z" for m in COMP if m != "bg"] + ["BG"]


def ridge_pred(trn, tst, alpha):
    w = trn.routes.to_numpy(float)
    A = trn[BASE + RX].to_numpy(float)
    mu = np.average(A, axis=0, weights=w)
    A = A - mu
    yv = trn.tprr.to_numpy(float)
    ym = np.average(yv, weights=w)
    pen = np.diag([0.0] * len(BASE) + [alpha] * len(RX))
    bb = np.linalg.solve((A * w[:, None]).T @ A + pen, (A * w[:, None]).T @ (yv - ym))
    return ym + (tst[BASE + RX].to_numpy(float) - mu) @ bb


D["fc_r"] = np.nan
for c in CLASSES:
    trn, tst = D[D.draft_year != c], D[D.draft_year == c]
    best_a, best_l = None, np.inf
    for a_ in [1e2, 1e3, 1e4, 1e5, 1e6]:
        loss = 0.0
        for c2 in trn.draft_year.unique():
            t2, v2 = trn[trn.draft_year != c2], trn[trn.draft_year == c2]
            loss += (v2.routes * (v2.tprr - ridge_pred(t2, v2, a_)) ** 2).sum()
        if loss < best_l:
            best_a, best_l = a_, loss
    D.loc[tst.index, "fc_r"] = ridge_pred(trn, tst, best_a)
w_ = D.routes
put("ridge_gain_pct", 100 * (1 - (w_ * (D.tprr - D.fc_r) ** 2).sum() / (w_ * (D.tprr - D.fc0) ** 2).sum()), 1)

# ---- reconciliation: does separation drive targets here, and does it explain the Break G slope?
S2 = D.dropna(subset=["psoe"]).copy()
ry = wresid(S2, BASE, S2.tprr.to_numpy(float), S2.routes.to_numpy(float))
rho_soe_tprr = boot_rho(S2.psoe, ry)
put("rho_soe_tprr", rho_soe_tprr[0]); put("rho_soe_tprr_lo", rho_soe_tprr[1]); put("rho_soe_tprr_hi", rho_soe_tprr[2])
b_a, se_a = wfit(S2, BASE + ["BG"], "tprr")
b_b, se_b = wfit(S2, BASE + ["psoe", "BG"], "tprr")
put("n_recon", len(S2)); put("recon_without", 100 * b_a[-1]); put("recon_with", 100 * b_b[-1])
# standardized effects on a common set of players: same baseline, outcome in its own player sd
COMMON = D.dropna(subset=["psoe", "ngs_sep"]).copy()
COMMON = COMMON[COMMON.sep_w > 0]
put("n_common", len(COMMON))
EFF_OUT = [("tprr", "routes", "Targets per route (TPRR)"), ("yprr", "routes", "Yards per route"),
           ("ngs_sep", "sep_w", "NGS separation (season)"), ("psoe", "n_soe", "Separation over expected,\ngraded breaks")]
EFF = []
for yc, wc, lab in EFF_OUT:
    sd_y = np.sqrt(np.cov(COMMON[yc], aweights=COMMON[wc]))
    b0 = wfit(COMMON, BASE + ["BG"], yc, wc)[0][-1] / sd_y
    bb_ = []
    for _ in range(NBOOT):
        dd = COMMON.iloc[RNG.integers(0, len(COMMON), len(COMMON))]
        bb_.append(wfit(dd, BASE + ["BG"], yc, wc)[0][-1] / sd_y)
    lo_, hi_ = np.percentile(bb_, [2.5, 97.5])
    EFF.append(dict(y=yc, lab=lab, b=b0, lo=lo_, hi=hi_))
    put(f"eff_{yc}", b0); put(f"eff_{yc}_lo", lo_); put(f"eff_{yc}_hi", hi_)
EFF = pd.DataFrame(EFF)

# ----------------------------------------------------------------------------- board
D["vs_fc"] = D.tprr - D.fc0


def call(r):
    if r.position_group == "TE":
        return "no call (TE)"
    if r.BG_lo > 0:
        return "plus"
    if r.BG_hi < 0:
        return "minus"
    return "neutral"


D["call"] = D.apply(call, axis=1)
D.loc[(D.position_group == "WR") & (D.n_drills < 5), "call"] = "no call (<5 drills)"
from scipy.stats import beta as _beta  # noqa: E402


def cp_int(k, n, a=0.05):
    """Exact (Clopper-Pearson) binomial interval."""
    lo = 0.0 if k == 0 else _beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


CAL = []
for k in ("plus", "neutral", "minus"):
    d = D[D.call == k]
    nb = int((d.vs_fc > 0).sum())
    lo, hi = cp_int(nb, len(d))
    put(f"n_{k}", len(d)); put(f"{k}_n_beat", nb)
    put(f"{k}_beat_pct", 100 * nb / len(d), 0)
    put(f"{k}_beat_lo", 100 * lo, 0); put(f"{k}_beat_hi", 100 * hi, 0)
    put(f"{k}_gap_pp", 100 * np.average(d.vs_fc, weights=d.routes), 1)
    put(f"{k}_tprr", 100 * np.average(d.tprr, weights=d.routes), 1)
    put(f"{k}_fc", 100 * np.average(d.fc0, weights=d.routes), 1)
    by = {}
    for c in CLASSES:
        dc = d[d.draft_year == c]
        by[c] = (int((dc.vs_fc > 0).sum()), len(dc))
    CAL.append(dict(call=k, n=len(d), k=nb, lo=lo, hi=hi, tprr=float(NUM[f"{k}_tprr"]), fc=float(NUM[f"{k}_fc"]),
                    by=by))
CAL = pd.DataFrame(CAL).set_index("call")
# 2025 only (the class with the least development exposure: one season of outcomes)
for k in ("plus", "minus"):
    d = D[(D.call == k) & (D.draft_year == 2025)]
    put(f"{k}25_n", len(d)); put(f"{k}25_beat", int((d.vs_fc > 0).sum()))
# units: what a Break G number means on the field
put("bg_drill_yds", float(NUM["bg_drill_median"]) * G_YD, 1)
put("sd_yds", float(NUM["g_true_sd"]) * G_YD, 2)
put("plus_sd_yds", 1.5 * float(NUM["g_true_sd"]) * G_YD, 1)
D.sort_values("BG", ascending=False)[["display_name", "position_group", "draft_year", "draft_overall_pick", "n_drills",
                                      "BG", "BG_lo", "BG_hi", "routes", "tprr", "fc0", "call"]].to_csv(OUT / "board.csv")
# decision card: 2025 class (newest, one season) chosen by rule: every plus and minus WR, the neutral WR closest to a
# minus call, the WR with fewest drills, and the TE with most routes
c25 = D[D.draft_year == 2025]
w25 = c25[c25.position_group == "WR"]
card_ids = list(w25[w25.call == "plus"].sort_values("BG", ascending=False).index)
card_ids += list(w25[w25.call == "minus"].index)
card_ids.append(w25[w25.call == "neutral"].sort_values("BG_hi").index[0])
card_ids.append(w25[w25.call == "no call (<5 drills)"].sort_values("routes", ascending=False).index[0])
card_ids.append(c25[c25.position_group == "TE"].sort_values("routes", ascending=False).index[0])
PLAIN = {"plus": "Plus. Note it; do not move him up a round",
         "minus": "Minus. Loses a tie with a WR picked near him",
         "neutral": "No call. Grade the tape",
         "no call (<5 drills)": "No call. Ask for all route drills at his pro day",
         "no call (TE)": "No call. Break G is not used for tight ends"}
TAB = []
for i, pid in enumerate(card_ids):
    r_ = D.loc[pid]
    TAB.append(dict(name=r_.display_name, pos=r_.position_group,
                    pick="undrafted" if pd.isna(r_.draft_overall_pick) else str(int(r_.draft_overall_pick)),
                    nd=int(r_.n_drills), bg=put(f"tab{i}_bg", r_.BG, 1), lo=put(f"tab{i}_lo", r_.BG_lo, 1),
                    hi=put(f"tab{i}_hi", r_.BG_hi, 1), routes=int(r_.routes), tprr=put(f"tab{i}_tprr", 100 * r_.tprr, 0),
                    fc=put(f"tab{i}_fc", 100 * r_.fc0, 0), call=PLAIN[r_.call]))
(OUT / "board_table.json").write_text(json.dumps(TAB, indent=1))
b25 = D[(D.draft_year == 2025) & (D.call == "plus")].sort_values("BG", ascending=False)
NUM["b25_plus_names"] = ", ".join(b25.display_name)
put("n_2025_plus", len(b25))


# ---- draft-room example: each minus WR against the nearest-pick non-minus WR of his own class
pairs = []
wr = D[D.position_group == "WR"].copy()
wr["pk"] = wr.draft_overall_pick.fillna(262)
for pid, r_ in wr[wr.call == "minus"].iterrows():
    pool = wr[(wr.draft_year == r_.draft_year) & (wr.call != "minus")]
    if len(pool) == 0:
        continue
    o = (pool.pk - r_.pk).abs().idxmin()
    pairs.append(dict(minus=r_.display_name, other=pool.loc[o, "display_name"], d_pick=abs(pool.loc[o, "pk"] - r_.pk),
                      minus_lower=float(r_.tprr < pool.loc[o, "tprr"])))
PAIRS = pd.DataFrame(pairs)
put("n_pairs", len(PAIRS)); put("n_pairs_minus_lower", int(PAIRS.minus_lower.sum()))
put("pairs_median_dpick", int(PAIRS.d_pick.median()))
ex_m = D[D.display_name == "A.T. Perry"].iloc[0]
ex_o = D[(D.display_name == PAIRS.set_index("minus").loc["A.T. Perry", "other"])].iloc[0]
NUM["ex_other_name"] = ex_o.display_name
put("ex_other_pick", int(ex_o.draft_overall_pick)) if pd.notna(ex_o.draft_overall_pick) else None
put("ex_other_bg", ex_o.BG, 1); put("ex_other_tprr", 100 * ex_o.tprr, 1); put("ex_other_fc", 100 * ex_o.fc0, 1)
put("ex_other_routes", int(ex_o.routes))
put("ex_minus_bg", ex_m.BG, 1); put("ex_minus_tprr", 100 * ex_m.tprr, 1); put("ex_minus_pick", int(ex_m.draft_overall_pick))
put("ex_dpick", int(abs(ex_m.draft_overall_pick - ex_o.draft_overall_pick)))

# ============================================================================= FIGURES
# ---- Fig 1 (hero, tracking-native): four rule-chosen WRs. A: the cut seen from above, drill reps vs NFL breaks;
# B: acceleration through the cut on one time axis; C: NFL targets per route vs the no-tracking forecast
cnt = g.groupby("nfl_id").size()
cand = X[(X.n_drills >= 5) & X.index.isin(cnt[cnt >= 8].index) & X.index.isin(D.index) & (X.position_group == "WR")]
picks = list(cand.sort_values("BG", ascending=False).head(2).index) + list(cand.sort_values("BG").head(2).index)
TT = np.arange(-10, 11) * DT


def profile(x, y_, i):
    vx, vy, ax_, ay_ = kin(x, y_)
    s = np.hypot(vx, vy)
    a = np.hypot(ax_, ay_) / G_YD
    S, A = np.full(21, np.nan), np.full(21, np.nan)
    for k in range(-10, 11):
        if 0 <= i + k < len(x):
            S[k + 10], A[k + 10] = s[i + k], a[i + k]
    return S, A


def oriented(x, y_, i):
    """+-1 s of path around the cut: cut at origin, stem pointing up, turn mirrored to the right."""
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


fig = plt.figure(figsize=(FIGW, 11.2))
gs = fig.add_gridspec(4, 3, width_ratios=[0.95, 1.75, 1.0], left=0.155, right=0.985, top=0.885, bottom=0.105,
                      wspace=0.28, hspace=0.32)
HERO = []
for j, pid in enumerate(picks):
    r_ = X.loc[pid]
    dr = cb[cb.nfl_id == pid]
    sn = g[g.nfl_id == pid]
    gkeys = list(zip(sn.game_id, sn.play_id, sn.nfl_id, sn.i_break))
    axA, axB, axC = (fig.add_subplot(gs[j, k]) for k in range(3))
    # A: overhead
    for (gid, pl, ni, i) in gkeys:
        qq = oriented(*gtraces[(gid, pl, ni)][:2], int(i))
        if qq is not None:
            axA.plot(qq[:, 0], qq[:, 1], color=OI["orange"], lw=0.8, alpha=0.45)
    rep = (dr.bg - dr.bg.median()).abs().idxmin()
    for e, i in zip(dr.event_id, dr.i_break):
        qq = oriented(*dtraces[e], int(i))
        if qq is None:
            continue
        if e == dr.loc[rep, "event_id"]:
            axA.plot(qq[:, 0], qq[:, 1], color=OI["blue"], lw=2.4, zorder=4)
            axA.scatter(qq[:, 0], qq[:, 1], s=9, color=OI["blue"], zorder=5)
        else:
            axA.plot(qq[:, 0], qq[:, 1], color=OI["blue"], lw=0.8, alpha=0.5)
    axA.scatter([0], [0], marker="x", color="black", s=30, zorder=6)
    axA.set_xlim(-4, 9); axA.set_ylim(-8.5, 4.5); axA.set_aspect("equal", adjustable="box")
    axA.set_xticks([0, 5]); axA.set_yticks([-5, 0])
    axA.tick_params(labelsize=8.5)
    call_ = D.loc[pid, "call"] if pid in D.index else ""
    axA.text(-0.2, 0.5, f"{r_.display_name}\n{int(r_.draft_year)}, pick "
             f"{'UDFA' if pd.isna(r_.draft_overall_pick) else int(r_.draft_overall_pick)}\n"
             f"Break G {r_.BG:+.1f} sd\ncall: {call_}", transform=axA.transAxes, ha="right", va="center",
             fontsize=10, fontweight="bold" if j < 2 else "normal")
    # B: acceleration through the cut
    axB.axvspan(-0.5, 0.5, color=OI["yellow"], alpha=0.2, lw=0)
    prof_d = np.array([profile(*dtraces[e], int(i))[1] for e, i in zip(dr.event_id, dr.i_break)])
    prof_g = np.array([profile(*gtraces[(a, b_, c_)][:2], int(i))[1] for a, b_, c_, i in gkeys])
    for arr, col in ((prof_g, OI["orange"]), (prof_d, OI["blue"])):
        for a_ in arr:
            axB.plot(TT, a_, color=col, lw=0.6, alpha=0.22)
        axB.plot(TT, np.nanmedian(arr, axis=0), color=col, lw=2.8)
    axB.axvline(0, color="#555555", lw=0.7, ls=":")
    axB.set_ylim(0, 0.95); axB.set_xlim(-1, 1); axB.set_xticks([-1, -0.5, 0, 0.5, 1])
    dg, gg_ = dr.bg.median(), sn.bg.median()
    put(f"hero{j}_drill_g", dg); put(f"hero{j}_game_g", gg_)
    axB.text(0.02, 0.96, f"drill {dg:.2f} g  (n={len(dr)} reps)", color=OI["blue"], transform=axB.transAxes,
             va="top", fontsize=9.5, fontweight="bold")
    axB.text(0.02, 0.80, f"NFL {gg_:.2f} g  (n={len(sn)} breaks)", color="#A86F00", transform=axB.transAxes,
             va="top", fontsize=9.5, fontweight="bold")
    axB.set_ylabel("accel. (g)", fontsize=9.5)
    axB.tick_params(labelsize=9)
    # C: targets per route vs forecast
    tp, fc = 100 * D.loc[pid, "tprr"], 100 * D.loc[pid, "fc0"]
    axC.barh([1, 0], [tp, fc], color=[OI["green"], OI["grey"]], height=0.6)
    axC.text(tp + 0.6, 1, f"{tp:.0f}%  ({int(D.loc[pid, 'routes'])} routes)", va="center", fontsize=9.5)
    axC.text(fc + 0.6, 0, f"{fc:.0f}%", va="center", fontsize=9.5)
    axC.set_yticks([1, 0], ["NFL TPRR", "forecast"], fontsize=9.5)
    axC.set_xlim(0, 52); axC.set_ylim(-0.6, 1.6)
    axC.tick_params(labelsize=9)
    HERO.append(dict(name=r_.display_name, dg=dg, gg=gg_, tp=tp, fc=fc))
    if j == 0:
        axA.set_title("A. The cut from above (yd)", loc="left", fontsize=11)
        axB.set_title("B. Acceleration through the cut, drill vs NFL", loc="left", fontsize=11)
        axC.set_title("C. Targets per route", loc="left", fontsize=11)
        for k_, lab in ((0.12, "brake"), (0.5, "turn"), (0.86, "drive out")):
            axB.text(k_, 0.03, lab, transform=axB.transAxes, fontsize=9, ha="center", color="#333333")
    if j == 3:
        axB.set_xlabel("seconds from the sharpest point of the turn", fontsize=10)
        axC.set_xlabel("% of routes targeted", fontsize=10)
        axA.set_xlabel("yd; stem up, turn mirrored right", fontsize=9)
hand = [plt.Line2D([], [], color=OI["blue"], lw=2.6, label="Combine route-drill reps (thick: his median rep, dots every 0.1 s)"),
        plt.Line2D([], [], color=OI["orange"], lw=2.6, label="his sampled NFL slant/out/hitch/in/corner breaks"),
        plt.Rectangle((0, 0), 1, 1, color=OI["yellow"], alpha=0.4, label="1.0 s Break G window")]
fig.legend(handles=hand, loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Break G: two hardest and two softest Combine cutters among WRs with 5+ route drills and 8+ NFL breaks",
             x=0.01, ha="left", fontsize=12.5, fontweight="bold")
hit_hi = sum(h["tp"] > h["fc"] for h in HERO[:2])
hit_lo = sum(h["tp"] < h["fc"] for h in HERO[2:])
fig.text(0.01, 0.935, f"Top two rows: hardest drill cuts. Bottom two: softest. The drill gap shrinks on NFL breaks; "
         f"targets go the predicted way for {hit_hi} of 2 hard and {hit_lo} of 2 soft cutters.", fontsize=10, ha="left",
         color="#333333")
fig.savefig(FIG / "fig1_hero.png", dpi=150, facecolor="white")
plt.close(fig)
for j, p in enumerate(picks):
    NUM[f"hero{j}_name"] = X.loc[p, "display_name"]
put("hero_drill_gap", (float(NUM["hero0_drill_g"]) + float(NUM["hero1_drill_g"]) - float(NUM["hero2_drill_g"])
                       - float(NUM["hero3_drill_g"])) / 2)
put("hero_game_gap", (float(NUM["hero0_game_g"]) + float(NUM["hero1_game_g"]) - float(NUM["hero2_game_g"])
                      - float(NUM["hero3_game_g"])) / 2)

# ---- Fig 2: the evidence. A: TPRR slope class by class (held-out R2 change); B: calibration of the calls
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 4.8), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axs[0]
for i, r_ in CLS.iterrows():
    ax.errorbar(r_.b, i, xerr=[[r_.b - r_.blo], [r_.bhi - r_.b]], fmt="o", color=CLS_COL[r_.cls], capsize=4, ms=9, lw=2)
ax.errorbar(100 * coef, 3.3, xerr=[[100 * coef - float(NUM["coef_pp_lo"])], [float(NUM["coef_pp_hi"]) - 100 * coef]],
            fmt="D", color="black", capsize=4, ms=9, lw=2)
ylab = [f"{int(c)} class\nn={int(n)}" for c, n in zip(CLS.cls, CLS.n)] + [f"all three\nn={len(D)}"]
ax.set_yticks([0, 1, 2, 3.3], ylab, fontsize=10.5)
gains = list(CLS.gain) + [NUM["dr2_pct"]]
for yy_, gg in zip([0, 1, 2, 3.3], gains):
    ax.text(1.0, yy_, f"{float(gg):+.1f}%", fontsize=10.5, va="center", ha="right", transform=ax.get_yaxis_transform(),
            color=OI["verm"] if float(gg) < 0 else "#222222")
ax.text(1.0, -0.8, "error removed when\nthis class is held out", fontsize=9, ha="right", va="center",
        transform=ax.get_yaxis_transform(), color="#444444")
ax.axvline(0, color="black", lw=0.9)
ax.set_xlim(-3.5, 9.5)
ax.set_ylim(3.9, -1.3)
ax.set_xlabel("TPRR points per sd of Break G, beyond the baseline (95% CI)", fontsize=10)
ax.set_title("A. Same sign in every class; only 2023 clears zero", loc="left", fontsize=11)
ax = axs[1]
lab_ = {"plus": "plus", "neutral": "no call", "minus": "minus"}
for i, k in enumerate(["plus", "neutral", "minus"]):
    r_ = CAL.loc[k]
    p_ = 100 * r_.k / r_.n
    col = {"plus": OI["blue"], "neutral": OI["grey"], "minus": OI["verm"]}[k]
    ax.errorbar(p_, i, xerr=[[p_ - 100 * r_.lo], [100 * r_.hi - p_]], fmt="o", color=col, capsize=4, ms=10, lw=2.2)
    for jj, c in enumerate(CLASSES):
        kk, nn = r_.by[c]
        if nn:
            ax.scatter(100 * kk / nn, i + 0.28, s=18 + 9 * nn, facecolor="none", edgecolor=CLS_COL[c], lw=1.4)
    ax.text(103, i, f"{int(r_.k)}/{int(r_.n)}\nTPRR {r_.tprr:.1f}% vs {r_.fc:.1f}%", fontsize=9.5, va="center")
ax.axvline(50, color=OI["grey"], lw=0.8, ls="--")
ax.set_yticks([0, 1, 2], [f"{lab_[k]} WRs" for k in ["plus", "neutral", "minus"]], fontsize=10.5)
ax.set_ylim(2.6, -0.5)
ax.set_xlim(0, 100)
ax.set_xlabel("% who beat their held-out-class forecast (exact 95% CI)", fontsize=10)
ax.set_title("B. Calls vs outcomes (hollow: by class, size = n)", loc="left", fontsize=11)
hc = [plt.scatter([], [], s=60, facecolor="none", edgecolor=CLS_COL[c], lw=1.4, label=str(c)) for c in CLASSES]
ax.legend(handles=hc, loc="upper right", fontsize=9, frameon=False, ncol=1, title="class", title_fontsize=9)
fig.subplots_adjust(left=0.09, right=0.83, top=0.9, bottom=0.14, wspace=0.42)
fig.savefig(FIG / "fig2_evidence.png", dpi=150, facecolor="white")
plt.close(fig)
# ---- Fig 3: is it a trait? reliability and link to the 40
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
order = rel.sort_values("rho").index.tolist()
for i, m in enumerate(order):
    r_ = rel.loc[m]
    c = OI["verm"] if m == "bg" else OI["grey"]
    ax.errorbar(r_.rho, i, xerr=[[r_.rho - r_.lo], [r_.hi - r_.rho]], fmt="o", color=c, capsize=3, ms=7)
ax.set_yticks(range(len(order)), [COMP_LABEL[m] for m in order], fontsize=10)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-0.35, 0.65)
ax.set_xlabel(f"split-half rho, 5 route drills vs the other 5 (n={NUM['rel_n']}, 95% CI)", fontsize=10)
ax.set_title("Repeats across drills: only the force measures", loc="left", fontsize=11.5)
ax = axs[1]
f40 = [spearmanr(X[m + "_z"], X.forty, nan_policy="omit")[0] for m in order]
ax.barh(range(len(order)), f40, color=[OI["verm"] if m == "bg" else OI["grey"] for m in order], height=0.6)
ax.set_yticks(range(len(order)), [""] * len(order))
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-0.4, 0.4)
ax.set_xlabel("rho with official 40 (negative = faster)", fontsize=10)
ax.set_title("Not a 40 proxy", loc="left", fontsize=11.5)
fig.tight_layout()
save(fig, "fig3_trait.png")

# ---- Fig 4: why targets and separation disagree
fig, axs = plt.subplots(1, 3, figsize=(FIGW, 4.5))
ax = axs[0]
ax.scatter(jf.d, jf.g, s=12, color=OI["blue"], alpha=0.45)
ax.axhline(0, color=OI["grey"], lw=0.7, ls=":"); ax.axvline(0, color=OI["grey"], lw=0.7, ls=":")
ax.set_xlabel("Combine Break G, drill family (z)", fontsize=10)
ax.set_ylabel("NFL Break G, same route family (z)", fontsize=10)
ax.set_title(f"A. Drill to same NFL route\nrho {NUM['fam_rho']} ({NUM['fam_n']} player-family pairs)", loc="left",
             fontsize=10.5)
ax = axs[1]
qb_ = pd.qcut(q.bg_gz, 5, labels=False)
mm = q.groupby(qb_).agg(x=("bg_gz", "mean"), y=("soe", "mean"), s=("soe", "std"), n=("soe", "size"))
ax.errorbar(mm.x, 36 * mm.y, yerr=36 * 1.96 * mm.s / np.sqrt(mm.n), fmt="o-", color=OI["orange"], capsize=3)
ax.axhline(0, color=OI["grey"], lw=0.7, ls=":")
ax.set_xlabel("NFL Break G on the snap (z, fifths)", fontsize=10)
ax.set_ylabel("separation over expected (in)", fontsize=10)
ax.set_title(f"B. Harder NFL cut, same snap\nrho {NUM['snap_rho']} ({NUM['n_graded']} graded snaps)", loc="left",
             fontsize=10.5)
ax = axs[2]
ax.scatter(36 * S2.psoe, 100 * ry, s=8 + S2.routes / 25, color=OI["green"], alpha=0.6, edgecolor="white", lw=0.5)
ax.axhline(0, color=OI["grey"], lw=0.7, ls=":"); ax.axvline(0, color=OI["grey"], lw=0.7, ls=":")
ax.set_xlabel("player SOE, graded breaks (in)", fontsize=10)
ax.set_ylabel("TPRR beyond baseline (pct pts)", fontsize=10)
ax.set_title(f"C. Getting open vs getting targets\nrho {NUM['rho_soe_tprr']} [{NUM['rho_soe_tprr_lo']}, "
             f"{NUM['rho_soe_tprr_hi']}], n={len(S2)}", loc="left", fontsize=10.5)
fig.tight_layout()
save(fig, "fig4_soe.png")

# ============================================================================= write
put("c_window_s", 1.0, 1); put("c_lead_s", LEAD_MIN, 1); put("c_path_yd", 1.5, 1); put("c_entry", 2.5, 1)
put("c_season_routes", SEASON_ROUTES)
put("tab_beat_gap", float(NUM["minus_gap_pp"]) - float(NUM["plus_gap_pp"]), 1)
MD = {}
TABLE_KEYS = []
put("top_res_tprr", 100 * D.loc[top, "tprr"], 1)


def mn(v):
    v = str(v)
    return "−" + v[1:] if v.startswith("-") else v


def K(key):
    TABLE_KEYS.append(key)
    return mn(NUM[key])


rows = ["| 2025 player (pick) | Route drills | Break G [80% range] | First-season TPRR vs forecast | Call |",
        "|---|---|---|---|---|"]
for i_, t in enumerate(TAB):
    rows.append(f"| {t['name']}, {t['pos']} ({t['pick']}) | {t['nd']} | {K(f'tab{i_}_bg')} [{K(f'tab{i_}_lo')}, "
                f"{K(f'tab{i_}_hi')}] | {K(f'tab{i_}_tprr')}% vs {K(f'tab{i_}_fc')}% ({t['routes']} routes) | "
                f"{t['call']} |")
MD["card"] = "\n".join(rows)
MD["checks"] = "\n".join([
    "| Check | Result | Reading |", "|---|---|---|",
    f"| Break G × TPRR was 1 of {K('screen_k')} measure × outcome pairs tried | chance the best shuffled pair looks "
    f"this strong: {K('p_fwer')} | could be luck of the search |",
    f"| Drop {NUM['top_res_name']} ({int(D.loc[top, 'draft_year'])}, pick {int(D.loc[top, 'draft_overall_pick'])}, "
    f"{K('top_res_tprr')}% TPRR) | slope {K('rob_droptop_pp')} (t = {K('rob_droptop_t')}) | "
    f"one player carries half the slope |",
    f"| Only the {K('rob_cc3_n')} who ran the 3-cone | slope {K('rob_cc3_pp')} (t = {K('rob_cc3_t')}) | "
    f"cannot show it adds to agility times |"])


def render(num, md):
    tpl = OUT / "writeup_template.md"
    text = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda m: md[m.group(1)], tpl.read_text())
    used = []

    def sub(m):
        k = m.group(1)
        if k not in num:
            raise KeyError(f"writeup_template.md uses unknown key {k}")
        used.append(k)
        return mn(num[k])
    out = re.sub(r"\{\{(\w+)\}\}", sub, text)
    (OUT / "writeup.md").write_text(out)
    return used, text


used, tpl_txt = render(NUM, MD)
# numbers.json: exactly the values the text quotes (flat). claims.json: key -> the sentence that quotes it.
claims = {}
for sent in re.split(r"(?<=[.!?])\s+|\n", tpl_txt):
    for k_ in re.findall(r"\{\{(\w+)\}\}", sent):
        if k_ not in claims:
            txt = re.sub(r"\{\{(\w+)\}\}", lambda m: mn(NUM[m.group(1)]), sent).strip()
            claims[k_] = txt[:220]
for k_ in TABLE_KEYS:
    claims.setdefault(k_, "table: " + ("decision card" if k_.startswith("tab") else "checks"))
keep = {k_: NUM[k_] for k_ in sorted(set(used) | set(TABLE_KEYS))}
(OUT / "numbers.json").write_text(json.dumps(keep, indent=1, ensure_ascii=False))
(OUT / "claims.json").write_text(json.dumps({k_: claims[k_] for k_ in keep}, indent=1, ensure_ascii=False))
print(json.dumps(NUM, indent=1))
print(tmat.round(2))
print(CLS)
print(pd.DataFrame(NEST))
print(PAIRS)
print(EFF)
print(pd.DataFrame(ROB, columns=["label", "pp", "t", "n"]).to_string())
print(pd.DataFrame(TAB).to_string())
