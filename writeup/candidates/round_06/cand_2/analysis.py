"""Break G: Combine route-drill cut force vs NFL targets per route run (TPRR) and separation over expected.

Round 6, candidate 2. Run from the repo root:
    uv run python writeup/candidates/round_06/cand_2/analysis.py

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

# ---- robustness (appendix)
ROB = []


def rob(label, key, df, cols, ycol="tprr", wcol="routes", x="BG"):
    bb, se_ = wfit(df, cols + [x], ycol, wcol)
    ROB.append((label, put(f"rob_{key}_pp", 100 * bb[-1]), put(f"rob_{key}_t", bb[-1] / se_[-1]), len(df)))


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
put("n_break_snaps", len(g))
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
put("n_rand_snaps", int(Rr.n_r.sum()))

# ---- screen (now that player SOE exists)
xm_full = D[SCREEN_X].to_numpy(float)
cls_arr = D.draft_year.to_numpy()


def screen_t(xm):
    out = {}
    for yc, wc in OUTS.items():
        msk = D[yc].notna().to_numpy() & D[wc].fillna(0).gt(0).to_numpy()
        df = D[msk]
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
for k in ("plus", "minus", "neutral"):
    d = D[D.call == k]
    put(f"n_{k}", len(d))
    put(f"{k}_beat_pct", 100 * (d.vs_fc > 0).mean(), 0)
    put(f"{k}_gap_pp", 100 * np.average(d.vs_fc, weights=d.routes), 1)
D.sort_values("BG", ascending=False)[["display_name", "position_group", "draft_year", "draft_overall_pick", "n_drills",
                                      "BG", "BG_lo", "BG_hi", "routes", "tprr", "fc0", "call"]].to_csv(OUT / "board.csv")
bw = D[(D.position_group == "WR") & (D.routes >= 100)]
tab = pd.concat([bw.sort_values("BG", ascending=False).head(6), bw.sort_values("BG").head(4).iloc[::-1]])
TAB = []
for i, (pid, r_) in enumerate(tab.iterrows()):
    TAB.append(dict(name=r_.display_name, cls=int(r_.draft_year),
                    pick="UDFA" if pd.isna(r_.draft_overall_pick) else str(int(r_.draft_overall_pick)),
                    bg=put(f"tab{i}_bg", r_.BG, 1), lo=put(f"tab{i}_lo", r_.BG_lo, 1), hi=put(f"tab{i}_hi", r_.BG_hi, 1),
                    routes=int(r_.routes), tprr=put(f"tab{i}_tprr", 100 * r_.tprr, 1),
                    fc=put(f"tab{i}_fc", 100 * r_.fc0, 1), call=r_.call))
(OUT / "board_table.json").write_text(json.dumps(TAB, indent=1))
b25 = D[(D.draft_year == 2025) & (D.call == "plus")].sort_values("BG", ascending=False)
NUM["b25_plus_names"] = ", ".join(b25.display_name)
put("n_2025_plus", len(b25))

# ============================================================================= FIGURES
# ---- Fig 1 (hero): TPRR above the out-of-class forecast vs Break G, plus per-class slopes
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.3), gridspec_kw={"width_ratios": [2.1, 1]})
ax = axs[0]
res_pp = 100 * (D.tprr - D.fc0)
for c in CLASSES:
    d = D[D.draft_year == c]
    ax.scatter(d.BG, res_pp[d.index], s=12 + d.routes / 12, color=CLS_COL[c], alpha=0.75, edgecolor="white", lw=0.6,
               label=f"{c} class (n={len(d)})")
xx = np.linspace(D.BG.min(), D.BG.max(), 50)
band = []
for _ in range(500):
    ix = RNG.integers(0, len(D), len(D))
    dd = D.iloc[ix]
    band.append(np.polyval(np.polyfit(dd.BG, res_pp.iloc[ix], 1, w=np.sqrt(dd.routes)), xx))
band = np.percentile(band, [2.5, 97.5], axis=0)
sl = np.polyfit(D.BG, res_pp, 1, w=np.sqrt(D.routes))
ax.fill_between(xx, band[0], band[1], color=OI["grey"], alpha=0.22, lw=0)
ax.plot(xx, np.polyval(sl, xx), color="black", lw=1.6)
ax.axhline(0, color=OI["grey"], lw=0.8, ls=":")
lab = list(D[D.routes >= 300].assign(a=res_pp.abs()).sort_values("a").index[-3:]) + \
    list(D[D.routes >= 300].sort_values("BG").index[[0, -1]])
for pid in dict.fromkeys(lab):
    ax.annotate(D.loc[pid, "display_name"], (D.loc[pid, "BG"], res_pp[pid]), fontsize=9, xytext=(5, 3),
                textcoords="offset points")
ax.set_xlabel("Combine Break G, shrunk (sd of true Break G)")
ax.set_ylabel("NFL targets per route minus forecast (pct pts)")
ax.set_title("Higher Break G, a few more targets per route than forecast", loc="left", fontsize=11.5)
ax.legend(frameon=False, loc="lower right", fontsize=9, title="dot size = career routes", title_fontsize=9)
ax.set_ylim(np.percentile(res_pp, 0.5) - 3, np.percentile(res_pp, 99.5) + 4)
ax = axs[1]
for i, r_ in CLS.iterrows():
    ax.errorbar(r_.b, i, xerr=[[r_.b - r_.blo], [r_.bhi - r_.b]], fmt="o", color=CLS_COL[r_.cls], capsize=4, ms=8)
    ax.text(r_.bhi + 0.3, i, f"held-out R² {float(r_.gain):+.1f}%", va="center", fontsize=9)
ax.errorbar(100 * coef, 3.2, xerr=[[100 * coef - float(NUM["coef_pp_lo"])], [float(NUM["coef_pp_hi"]) - 100 * coef]],
            fmt="D", color="black", capsize=4, ms=8)
ax.set_yticks([0, 1, 2, 3.2], [f"{c} only\n(n={n})" for c, n in zip(CLS.cls, CLS.n)] + [f"all\n(n={len(D)})"])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(min(-4.5, CLS.blo.min() - 0.5), CLS.bhi.max() + 6.5)
ax.set_xlabel("pct pts of TPRR per sd of Break G (95% CI)")
ax.set_title("Slope by class: all positive", loc="left", fontsize=11)
ax.invert_yaxis()
fig.tight_layout()
save(fig, "fig1_hero.png")

# ---- Fig 2: drill vs NFL cards for four rule-chosen players
FAM1 = "IN"
gpl = g.groupby("nfl_id").size()
cand = X[(X.n_drills >= 5)]
cnt = g[g.route_ran == FAM1].groupby("nfl_id").size()
cand = cand[cand.index.isin(cnt[cnt >= 4].index) & cand.index.isin(cb[cb.family == FAM1].nfl_id)]
picks = list(cand.sort_values("BG", ascending=False).head(2).index) + list(cand.sort_values("BG").head(2).index)
cmap = plt.get_cmap("viridis")
vmax = 0.8


def rot_path(x, y_, apex, pre=12, post=11):
    lo_, hi_ = max(0, apex - pre), min(len(x), apex + post)
    px, py = x[lo_:hi_] - x[apex], y_[lo_:hi_] - y_[apex]
    hx, hy = x[max(lo_, apex - 3)] - x[lo_], y_[max(lo_, apex - 3)] - y_[lo_]
    th = np.arctan2(hx, hy)
    rx = px * np.cos(th) - py * np.sin(th)
    ry = px * np.sin(th) + py * np.cos(th)
    if rx[-1] < 0:
        rx = -rx
    return rx, ry, lo_, hi_


fig, axs = plt.subplots(2, 4, figsize=(FIGW, 7.0))
for j, pid in enumerate(picks):
    r_ = X.loc[pid]
    ax = axs[0, j]
    rep = cb[(cb.nfl_id == pid) & (cb.family == FAM1)].iloc[0]
    x, y_ = dtraces[rep.event_id]
    _, _, ax_, ay_ = kin(x, y_)
    am = np.hypot(ax_, ay_) / G_YD
    rx, ry, lo_, hi_ = rot_path(x, y_, int(rep.i_apex))
    pts = np.column_stack([rx, ry]).reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=plt.Normalize(0, vmax), lw=4)
    lc.set_array(am[lo_:hi_][:-1])
    ax.add_collection(lc)
    ax.plot(0, 0, "o", mfc="white", mec="black", ms=6, zorder=4)
    ax.set_xlim(-2.5, 5.5); ax.set_ylim(-6.5, 3.5); ax.set_aspect("equal")
    tprr_txt = f"NFL TPRR {100 * D.loc[pid, 'tprr']:.0f}% (forecast {100 * D.loc[pid, 'fc0']:.0f}%)" if pid in D.index else ""
    ax.set_title(f"{r_.display_name} ({int(r_.draft_year)})\nBreak G {r_.BG:+.1f} sd\n{tprr_txt}", fontsize=9.5,
                 loc="left")
    ax.text(5.3, -6.2, f"Combine {rep.drill_name.split('_')[0].lower()} rep\n{rep.bg:.2f} g", fontsize=9, ha="right")
    ax2 = axs[1, j]
    sn = g[(g.nfl_id == pid) & (g.route_ran == FAM1)]
    for _, s_ in sn.iterrows():
        xx_, yy_, i0, ipf = gtraces[(s_.game_id, s_.play_id, s_.nfl_id)]
        _, _, gax, gay = kin(xx_, yy_)
        gam = np.hypot(gax, gay) / G_YD
        rx, ry, lo_, hi_ = rot_path(xx_, yy_, int(s_.i_apex))
        pts = np.column_stack([rx, ry]).reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=plt.Normalize(0, vmax),
                            lw=1.8, alpha=0.85)
        lc.set_array(gam[lo_:hi_][:-1])
        ax2.add_collection(lc)
    ax2.plot(0, 0, "o", mfc="white", mec="black", ms=6, zorder=4)
    ax2.set_xlim(-2.5, 5.5); ax2.set_ylim(-6.5, 3.5); ax2.set_aspect("equal")
    ax2.set_title(f"His NFL in routes (n={len(sn)})", fontsize=9.5, loc="left")
    ax2.text(5.3, -6.2, f"median\n{sn.bg.median():.2f} g", fontsize=9, ha="right")
    put(f"fig2_{j}_drill_g", rep.bg); put(f"fig2_{j}_game_g", sn.bg.median())
    for a in (ax, ax2):
        a.set_xticks([-2, 0, 2, 4]); a.set_yticks([-6, -3, 0, 3]); a.tick_params(labelsize=9)
    if j == 0:
        ax.set_ylabel("yd upfield (0 = apex)", fontsize=9.5)
        ax2.set_ylabel("yd upfield (0 = apex)", fontsize=9.5)
    ax2.set_xlabel("yd toward the break", fontsize=9.5)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
cax = fig.add_axes([0.93, 0.18, 0.012, 0.62])
fig.colorbar(sm, cax=cax).set_label("acceleration (g)", fontsize=9.5)
fig.suptitle("Combine in-breaking rep (top) and the same player's NFL in routes (bottom); bright = more force",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.subplots_adjust(left=0.06, right=0.91, top=0.82, bottom=0.08, wspace=0.3, hspace=0.5)
fig.savefig(FIG / "fig2_cards.png", dpi=150, facecolor="white")
plt.close(fig)
NUM["fig2_names"] = ", ".join(X.loc[p, "display_name"] for p in picks)

# ---- Fig 3: is it a trait? reliability and link to the 40
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
order = rel.sort_values("rho").index.tolist()
for i, m in enumerate(order):
    r_ = rel.loc[m]
    c = OI["verm"] if m == "bg" else OI["grey"]
    ax.errorbar(r_.rho, i, xerr=[[r_.rho - r_.lo], [r_.hi - r_.rho]], fmt="o", color=c, capsize=3, ms=7)
ax.set_yticks(range(len(order)), [COMP_LABEL[m] for m in order])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-0.35, 0.65)
ax.set_xlabel(f"split-half rho, 5 route drills vs the other 5 (n={NUM['rel_n']}, 95% CI)", fontsize=10)
ax.set_title("Does a receiver repeat himself across drills?", loc="left", fontsize=11)
ax = axs[1]
f40 = [spearmanr(X[m + "_z"], X.forty, nan_policy="omit")[0] for m in order]
ax.barh(range(len(order)), f40, color=[OI["verm"] if m == "bg" else OI["grey"] for m in order], height=0.6)
ax.set_yticks(range(len(order)), [""] * len(order))
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-0.4, 0.4)
ax.set_xlabel("rho with official 40 (negative = faster)", fontsize=10)
ax.set_title("Is it just the 40?", loc="left", fontsize=11)
fig.tight_layout()
save(fig, "fig3_trait.png")

# ---- Fig 4: separation over expected
fig, axs = plt.subplots(1, 3, figsize=(FIGW, 4.4))
ax = axs[0]
ax.scatter(jf.d, jf.g, s=10, color=OI["blue"], alpha=0.45)
ax.set_xlabel("Combine Break G, drill family (z)")
ax.set_ylabel("NFL Break G, same route family (z)")
ax.set_title(f"Drill to same NFL route\nrho {NUM['fam_rho']} ({NUM['fam_n']} pairs)", loc="left", fontsize=10.5)
ax = axs[1]
qb_ = pd.qcut(q.bg_gz, 5, labels=False)
mm = q.groupby(qb_).agg(x=("bg_gz", "mean"), y=("soe", "mean"), s=("soe", "std"), n=("soe", "size"))
ax.errorbar(mm.x, mm.y, yerr=1.96 * mm.s / np.sqrt(mm.n), fmt="o-", color=OI["orange"], capsize=3)
ax.axhline(0, color=OI["grey"], lw=0.7, ls=":")
ax.set_xlabel("NFL Break G on the snap (z, fifths)")
ax.set_ylabel("separation over expected (yd)")
ax.set_title(f"NFL cut force and SOE, same snap\nrho {NUM['snap_rho']} ({NUM['n_graded']} graded snaps)",
             loc="left", fontsize=10.5)
ax = axs[2]
ax.scatter(S_.BG, S_.psoe, s=10 + 2 * S_.n_soe, color=OI["green"], alpha=0.6, edgecolor="white", lw=0.5)
xx = np.linspace(S_.BG.min(), S_.BG.max(), 20)
ax.plot(xx, np.polyval(np.polyfit(S_.BG, S_.psoe, 1, w=np.sqrt(S_.n_soe)), xx), color="black", lw=1.4)
ax.axhline(0, color=OI["grey"], lw=0.7, ls=":")
ax.set_xlabel("Combine Break G, shrunk (sd)")
ax.set_ylabel("player SOE, graded break snaps (yd)")
ax.set_title(f"Combine Break G and NFL SOE\n{NUM['soe_coef_in']} in per sd [{NUM['soe_lo_in']}, {NUM['soe_hi_in']}]",
             loc="left", fontsize=10.5)
fig.tight_layout()
save(fig, "fig4_soe.png")

# text prints negatives with a Unicode minus; keep the magnitudes of negative decimals so check.py can match them
NUM["neg_abs"] = sorted({v[1:] for v in NUM.values() if isinstance(v, str) and v.startswith("-")})
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
print(json.dumps(NUM, indent=1))
print(tmat.round(2))
print(CLS)
print(pd.DataFrame(ROB, columns=["label", "pp", "t", "n"]).to_string())
print(pd.DataFrame(TAB).to_string())
