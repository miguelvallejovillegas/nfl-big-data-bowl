"""Round 25, candidate 2 (revision of round 24 cand_2): receiver break mechanics, Combine route drill vs the same route family on Sundays.

Changes from round 24: route paths drawn in x-y (Fig 1, Fig 4); the held-out test drawn as data (Fig 3, one dot
per receiver, one fitted line per held-out class) instead of forest plots; team-clustered bootstrap; one-sided
placebo test; a max-statistic correction across the 30 drill-to-NFL cells; play-log evidence for named receivers.

One metric, Break G: the average g-force a receiver puts into the ground over the one second around his first real
cut (braking into the plant, turning, driving out). It is graded by the same code on Combine route drills and on
sampled NFL snaps of the same route family (slant, out, hitch, in, corner), and tested against full-season Next Gen
Stats separation over expected (SOE) on held-out draft classes.

Run from the repo root (about two minutes):
    uv run python writeup/candidates/round_25/cand_2/analysis.py

Reads data/processed/*.parquet only. Writes next to this file: numbers.json (every number writeup.md quotes, as the
strings printed), figures/*.png (1600 px wide, 150 dpi), and writeup.md rendered from writeup_template.md.

The Break G definition (first cut of 30 degrees or more after 1.5 yd of stem, entered faster than 2.5 yd/s; mean
|acceleration| over 1.0 s centred on it; z-scored within drill and Combine year; averaged over drills and shrunk for
few drills) and the +/-3 inch margin are carried unchanged from earlier rounds, so nothing here was tuned on the
outcome.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
P = ROOT / "data" / "processed"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
for _old in FIG.glob("*.png"):
    _old.unlink()

DT = 0.1
G_YD = 10.73          # 1 g in yd/s^2
IN_YD = 36.0
MPH = 3600 / 1760     # 1 yd/s in mph
NBOOT = int(os.environ.get("BDB_NBOOT", 2000))
NSIM = int(os.environ.get("BDB_NSIM", 1000))
MARGIN = 3.0          # inches of SOE per sd, fixed before testing (round 9)
CLASSES = (2023, 2024, 2025)
FAMILIES = ["SLANT", "OUT", "HITCH", "IN", "CORNER"]
FAM_LABEL = {"SLANT": "Slant", "OUT": "Out", "HITCH": "Hitch / curl / comeback", "IN": "In / dagger",
             "CORNER": "Corner / sail"}
DRILL_FAMILY = {"SLANT_ROUTE_LEFT": "SLANT", "SLOT_STRIKE_ROUTE_LEFT": "SLANT", "SPEED_OUT_ROUTE_LEFT": "OUT",
                "CURL_ROUTE_RIGHT": "HITCH", "COMEBACK_ROUTE_RIGHT": "HITCH", "HOOK_ROUTE_RIGHT": "HITCH",
                "DAGGER_ROUTE_LEFT": "IN", "IN_ROUTE_LEFT": "IN", "SLOT_SAIL_ROUTE_RIGHT": "CORNER",
                "CORNER_ROUTE": "CORNER"}
COMP = ["v_in", "decel", "v_apex", "reaccel", "bg"]
COMP_LABEL = {"v_in": "Entry\nspeed", "decel": "Braking into\nthe plant", "v_apex": "Speed at\nthe plant",
              "reaccel": "Drive\nout", "bg": "Break G\n(all three)"}

FIGW = 1600 / 150
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2", "verm": "#D55E00",
      "purple": "#CC79A7", "grey": "#9A9A9A", "black": "#000000", "yellow": "#F0E442"}
DRILL_C, NFL_C = OI["orange"], OI["blue"]
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 150, "savefig.dpi": 150,
                     "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.labelsize": 12})

NUM: dict[str, str] = {}


def put(key, val, nd=2, plus=False):
    """Store a quoted number as the exact string the text prints."""
    if isinstance(val, str):
        s = val
    elif isinstance(val, (int, np.integer)) or nd == 0:
        s = f"{int(round(float(val)))}"
    else:
        s = f"{float(val):+.{nd}f}" if plus else f"{float(val):.{nd}f}"
        if s in ("-0." + "0" * nd,):
            s = "0." + "0" * nd
    NUM[key] = s
    return s


def rng_for(name):
    return np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))


def zs(s):
    return (s - s.mean()) / s.std(ddof=1)


def q(a, p):
    return float(np.nanpercentile(np.asarray(a, float), p))


def wls(A, y, w):
    s = np.sqrt(np.asarray(w, float))
    return np.linalg.lstsq(A * s[:, None], np.asarray(y, float) * s, rcond=None)[0]


def save(fig, name):
    fig.savefig(FIG / name, dpi=150, facecolor="white")
    plt.close(fig)


# ============================================================================ kinematics (frozen definition)
def kin(x, y):
    w = 7 if len(x) >= 7 else max(3, (len(x) // 2) * 2 - 1)
    vx = savgol_filter(x, w, 2, deriv=1, delta=DT)
    vy = savgol_filter(y, w, 2, deriv=1, delta=DT)
    return vx, vy, savgol_filter(vx, w, 2, deriv=1, delta=DT), savgol_filter(vy, w, 2, deriv=1, delta=DT)


def angdiff(a, b):
    d = np.abs(a - b) % 360
    return np.where(d > 180, 360 - d, d)


def break_mechanics(x, y, i0, i1, min_path=1.5, min_turn=30.0, h=6, entry_speed=2.5, hw=5):
    """First real cut and its three speeds; Break G = mean |acceleration| (g) over 1.0 s centred on the cut."""
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
    iout = apex + int(np.argmax(s[apex:min(n, apex + 11)]))
    tr = np.full(21, np.nan)
    tlo, thi = max(0, apex - 10), min(n, apex + 11)
    tr[tlo - (apex - 10):tlo - (apex - 10) + (thi - tlo)] = s[tlo:thi]
    # route path around the plant for drawing only (does not touch any metric): plant at the origin, stem pointing
    # up, break mirrored to the right
    p0, p1 = max(0, apex - 15), min(n, apex + 11)
    k0 = max(0, apex - 8)
    stem = np.array([x[apex] - x[k0], y[apex] - y[k0]])
    ang = np.arctan2(stem[0], stem[1])
    ca, sa = np.cos(ang), np.sin(ang)
    dx, dy = x[p0:p1] - x[apex], y[p0:p1] - y[apex]
    px, py = ca * dx - sa * dy, sa * dx + ca * dy
    if px[apex - p0:].mean() < 0:
        px = -px
    return dict(v_in=float(s[iin]), v_apex=float(s[apex]), v_out=float(s[iout]),
                decel=float((s[iin] - s[apex]) / max((apex - iin) * DT, DT)),
                reaccel=float((s[iout] - s[apex]) / max((iout - apex) * DT, DT)),
                bg=float(amag[lo:hi].mean() / G_YD), trace=tr, apex=int(apex),
                px=px, py=py, ps=s[p0:p1].copy())


def icc1(df, col, gcol="nfl_id"):
    k = df.groupby(gcol).size()
    k0 = (len(df) - (k ** 2).sum() / len(df)) / (len(k) - 1)
    gm = df[col].mean()
    msb = (k * (df.groupby(gcol)[col].mean() - gm) ** 2).sum() / (len(k) - 1)
    msw = ((df[col] - df.groupby(gcol)[col].transform("mean")) ** 2).sum() / (len(df) - len(k))
    return (msb - msw) / (msb + (k0 - 1) * msw)


# ============================================================================ 1. Combine route drills
events = pd.read_parquet(P / "combine_events.parquet")
ev = events[events.drill_name.isin(DRILL_FAMILY) & events.position_group.isin(["WR", "TE"])]
frames = pd.read_parquet(P / "combine_frames.parquet", filters=[("event_id", "in", ev.event_id.tolist())])
rows = []
for eid, d in frames.groupby("event_id", sort=False):
    pl = d[d.entity_type == "PLAYER"].sort_values("frame")
    bl = d[d.entity_type == "BALL"]
    x, y = pl.x.to_numpy(), pl.y.to_numpy()
    i1 = len(x)
    if len(bl):  # stop at the catch so the ball-in-hand phase is not graded
        m = pl[["frame", "x", "y"]].merge(bl[["frame", "x", "y"]], on="frame", suffixes=("", "_b"))
        c = m.frame[np.hypot(m.x - m.x_b, m.y - m.y_b) < 1.0]
        if len(c):
            i1 = int(np.searchsorted(pl.frame.to_numpy(), c.iloc[0]))
    r = break_mechanics(x, y, 0, i1)
    if r:
        rows.append(dict(event_id=eid, **r))
cb = pd.DataFrame(rows).merge(ev[["event_id", "nfl_id", "draft_year", "drill_name", "position_group"]],
                              on="event_id")
cb["family"] = cb.drill_name.map(DRILL_FAMILY)
for m in COMP:
    cb[m + "_z"] = cb.groupby(["drill_name", "draft_year"])[m].transform(zs)
put("n_reps", len(ev), 0)
put("n_cuts", len(cb), 0)

# receiver score: mean z per drill, mean over drills, shrunk by the between-drill reliability (empirical Bayes)
cpd = cb.groupby(["nfl_id", "drill_name", "family"], as_index=False)[[m + "_z" for m in COMP]].mean()
cpl = cpd.groupby("nfl_id")[[m + "_z" for m in COMP]].mean()
cpl["n_drills"] = cpd.groupby("nfl_id").size()
cpl["n_cuts"] = cb.groupby("nfl_id").size()
icc_d = max(icc1(cpd, "bg_z"), 0.01)
vt = cpd.bg_z.var()
var_true, var_err = icc_d * vt, (1 - icc_d) * vt
post_var = 1 / (1 / var_true + cpl.n_drills / var_err)
cpl["BG"] = post_var * (cpl.n_drills * cpl.bg_z / var_err) / np.sqrt(var_true)  # posterior mean, true-sd units
cpl["BG_sd"] = np.sqrt(post_var / var_true)
cpl["rel"] = var_true / (var_true + var_err / cpl.n_drills)
cpl["e_var"] = var_err / (cpl.n_drills * var_true)  # noise variance of the raw mean, true-sd units
# reliability per graded cut (rep-level ICC) and the Spearman-Brown curve
icc_rep = icc1(cb, "bg_z")
med_cuts = float(cpl.n_cuts.median())
sb = lambda k: k * icc_rep / (1 + (k - 1) * icc_rep)  # noqa: E731
cuts70 = int(np.ceil(0.7 * (1 - icc_rep) / (icc_rep * 0.3)))
put("rel_rep", icc_rep)
put("med_cuts", med_cuts, 0)
put("rel_med", sb(med_cuts))
put("cuts70", cuts70, 0)
put("rel70", 0.70)
put("def_path", 1.5, 1)
put("def_entry", 2.5, 1)
put("def_hw", 0.5, 1)
put("clip_offset", 0.3, 1)
put("rel_mean", cpl.rel.mean())

# ============================================================================ 2. players, baselines, season SOE
players = pd.read_parquet(P / "players.parquet")
cr = pd.read_parquet(P / "combine_results.parquet").set_index("nfl_id")
pb = players.set_index("nfl_id")[["draft_overall_pick", "draft_round", "position_group", "draft_year",
                                  "display_name"]].join(cr[["forty", "combine_weight", "ngs_athleticism_score",
                                                            "three_cone", "short_shuttle"]])
pb = pb[pb.position_group.isin(["WR", "TE"])].copy()
pb["logpick"] = np.log(pb.draft_overall_pick.fillna(262))
pb["te"] = (pb.position_group == "TE").astype(float)
for c in ("forty", "combine_weight", "ngs_athleticism_score"):
    pb[c + "_s"] = pb.groupby(["draft_year", "position_group"])[c].transform(zs)
    pb[c + "_f"] = pb[c + "_s"].fillna(0.0)

ss_all = pd.read_parquet(P / "player_season_stats.parquet")
ss = ss_all[ss_all.position_group.isin(["WR", "TE"]) & ss_all.routes.gt(0) & ss_all.avg_separation.notna()].copy()
A = np.column_stack([np.ones(len(ss)), ss.main_alignment == "SLOT_WR", ss.main_alignment == "TE",
                     ss.position_group == "TE", ss.avg_cushion.fillna(ss.avg_cushion.mean()),
                     ss.season == 2024, ss.season == 2025]).astype(float)
ss["ssoe"] = ss.avg_separation - A @ wls(A, ss.avg_separation.to_numpy(float), ss.routes.to_numpy(float))
ss["slot_r"] = (ss.main_alignment == "SLOT_WR") * ss.routes
car = ss.groupby("nfl_id").agg(routes=("routes", "sum"), slot_r=("slot_r", "sum"), seasons=("season", "size"))
car["ssoe"] = ss.groupby("nfl_id").apply(lambda d: np.average(d.ssoe, weights=d.routes), include_groups=False)
car["slot"] = car.slot_r / car.routes
car["cush"] = ss.dropna(subset=["avg_cushion"]).groupby("nfl_id").apply(
    lambda d: np.average(d.avg_cushion, weights=d.routes), include_groups=False)
# noise of a career SOE: shrinkage fit on consecutive seasons (skill variance tau2, noise s2 per route)
s50 = ss[ss.routes >= 50]
mu = float(np.average(s50.ssoe, weights=s50.routes))
pv = s50.pivot_table(index="nfl_id", columns="season", values="ssoe")
cp = pd.concat([pv[[a, a + 1]].dropna().set_axis(["a", "b"], axis=1) for a in (2023, 2024)])
tau2 = float(np.mean((cp.a - mu) * (cp.b - mu)))
iv = 1.0 / s50.routes.to_numpy(float)
s2 = float(np.sum(((s50.ssoe.to_numpy() - mu) ** 2 - tau2) * iv) / np.sum(iv ** 2))
car["soe_se"] = np.sqrt(s2 / car.routes)
put("n_pseason", len(ss[ss.nfl_id.isin(cpl.index) & ss.nfl_id.isin(pb.index)]), 0)

X = cpl.join(pb, how="inner")
put("n_drill", len(X), 0)
D = X.join(car, how="inner").copy()
D = D[D.routes > 0]
put("n_soe", len(D), 0)
put("med_routes", D.routes.median(), 0)
put("sd_soe_in", IN_YD * np.sqrt(np.cov(D.ssoe, aweights=D.routes)), 1)

BASELINES = {"pick": ["logpick"], "role": ["logpick", "te", "slot"],
             "full": ["logpick", "forty_f", "combine_weight_f", "ngs_athleticism_score_f", "te"]}
BL_LABEL = {"pick": "Pick only", "role": "Pick + role (TE, slot share)",
            "full": "Pick + official 40, weight, NGS score, TE"}


def loco(Dd, cols, bgcol="BG", return_res=False):
    """Leave-one-class-out: baseline fitted on two classes, scored on the third (route-weighted).
    Returns held-out R2 of baseline, R2 with Break G added, and the Break G slope on the held-out residual."""
    y, w, cl, bg = (Dd.ssoe.to_numpy(float), Dd.routes.to_numpy(float), Dd.draft_year.to_numpy(),
                    Dd[bgcol].to_numpy(float))
    XB = np.column_stack([np.ones(len(Dd))] + [Dd[c].to_numpy(float) for c in cols])
    Xb = np.column_stack([XB, bg])
    f0, f1, res = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
    for c in CLASSES:
        t = cl == c
        if not t.any():
            continue
        f0[t] = XB[t] @ wls(XB[~t], y[~t], w[~t])
        f1[t] = Xb[t] @ wls(Xb[~t], y[~t], w[~t])
    res = y - f0
    present = [c for c in CLASSES if (cl == c).any()]
    Ac = np.column_stack([(cl == c).astype(float) for c in present] + [bg])
    slope = IN_YD * wls(Ac, res, w)[-1]
    sst = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    r0, r1 = 1 - np.sum(w * (y - f0) ** 2) / sst, 1 - np.sum(w * (y - f1) ** 2) / sst
    # class interaction: the held-out slope inside each class on its own
    per = []
    for c in CLASSES:
        t = cl == c
        per.append(IN_YD * wls(np.column_stack([np.ones(t.sum()), bg[t]]), res[t], w[t])[1] if t.sum() > 3
                   else np.nan)
    if return_res:
        return res
    return np.array([r0, r1, r1 - r0, slope] + per)


TAB = {}
for key, cols in BASELINES.items():
    est = loco(D, cols)
    rng = rng_for("boot_" + key)
    ids = np.arange(len(D))
    bt = []
    for _ in range(NBOOT):  # receiver-level (cluster) bootstrap pooled across classes; LOCO refitted in every draw
        ix = rng.choice(ids, len(ids))
        if len(np.unique(D.draft_year.to_numpy()[ix])) < 3:
            continue
        bt.append(loco(D.iloc[ix], cols))
    bt = np.array(bt)
    TAB[key] = dict(est=est, lo=np.percentile(bt, 5, axis=0), hi=np.percentile(bt, 95, axis=0), boot=bt)
    put(f"{key}_r2", est[0])
    put(f"{key}_dr2", est[2], 3)
    put(f"{key}_dr2_lo", TAB[key]["lo"][2], 3)
    put(f"{key}_dr2_hi", TAB[key]["hi"][2], 3)
    put(f"{key}_b", est[3], 1)
    put(f"{key}_b_lo", TAB[key]["lo"][3], 1)
    put(f"{key}_b_hi", TAB[key]["hi"][3], 1, plus=True)
put("pick_r2_lo", TAB["pick"]["lo"][0])
put("pick_r2_hi", TAB["pick"]["hi"][0])


# ---------------------------------------------------------------------------- ridge-shrunk baseline
RIDGE_COLS = ["logpick", "forty_f", "combine_weight_f", "ngs_athleticism_score_f", "te", "slot"]
LAMS = (0.3, 3.0, 30.0, 300.0)


def ridge_fit(Xt, y, w, lam):
    mu_, sd_ = Xt.mean(0), Xt.std(0) + 1e-9
    Z = (Xt - mu_) / sd_
    s = np.sqrt(w)
    A = np.column_stack([np.ones(len(Z)), Z]) * s[:, None]
    pen = np.diag([0.0] + [lam] * Z.shape[1])
    b = np.linalg.solve(A.T @ A + pen, A.T @ (y * s))
    return lambda Xn: np.column_stack([np.ones(len(Xn)), (Xn - mu_) / sd_]) @ b


def loco_ridge(Dd, bgcol="BG"):
    """As loco(), but the baseline is a ridge over pick, 40, weight, NGS, TE and slot share, its penalty chosen
    inside the two training classes (fit on one, score on the other)."""
    y, w, cl = Dd.ssoe.to_numpy(float), Dd.routes.to_numpy(float), Dd.draft_year.to_numpy()
    XB = np.column_stack([Dd[c].to_numpy(float) for c in RIDGE_COLS])
    bg = Dd[bgcol].to_numpy(float)
    f0, f1 = np.empty(len(y)), np.empty(len(y))
    for c in CLASSES:
        t = cl == c
        tr = np.where(~t)[0]
        inner = [cc for cc in CLASSES if cc != c]
        err = []
        for lam in LAMS:
            e_ = 0.0
            for a in inner:
                ia, ib = tr[cl[tr] == a], tr[cl[tr] != a]
                e_ += np.sum(w[ia] * (y[ia] - ridge_fit(XB[ib], y[ib], w[ib], lam)(XB[ia])) ** 2)
            err.append(e_)
        lam = LAMS[int(np.argmin(err))]
        f0[t] = ridge_fit(XB[~t], y[~t], w[~t], lam)(XB[t])
        Xb = np.column_stack([XB, bg])
        f1[t] = ridge_fit(Xb[~t], y[~t], w[~t], lam)(Xb[t])
    res = y - f0
    Ac = np.column_stack([(cl == c).astype(float) for c in CLASSES] + [bg])
    slope = IN_YD * wls(Ac, res, w)[-1]
    sst = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    r0, r1 = 1 - np.sum(w * (y - f0) ** 2) / sst, 1 - np.sum(w * (y - f1) ** 2) / sst
    return np.array([r0, r1, r1 - r0, slope])


est = loco_ridge(D)
rng = rng_for("boot_ridge")
bt = []
for _ in range(NBOOT):
    ix = rng.choice(np.arange(len(D)), len(D))
    if len(np.unique(D.draft_year.to_numpy()[ix])) == 3:
        bt.append(loco_ridge(D.iloc[ix]))
bt = np.array(bt)
TAB["ridge"] = dict(est=est, lo=np.percentile(bt, 5, axis=0), hi=np.percentile(bt, 95, axis=0), boot=bt)
put("ridge_r2", est[0])
put("ridge_dr2", est[2], 3)
put("ridge_dr2_lo", TAB["ridge"]["lo"][2], 3)
put("ridge_dr2_hi", TAB["ridge"]["hi"][2], 3)
put("ridge_b", est[3], 1)
put("ridge_b_lo", TAB["ridge"]["lo"][3], 1)
put("ridge_b_hi", TAB["ridge"]["hi"][3], 1, plus=True)

# ---------------------------------------------------------------------------- placebo: Break G shuffled within class
rng = rng_for("placebo")
PLAC = []
cls_ = D.draft_year.to_numpy()
for _ in range(NBOOT):
    Dp = D.copy()
    bgp = D.BG.to_numpy().copy()
    for c in CLASSES:
        ii = np.where(cls_ == c)[0]
        bgp[ii] = bgp[rng.permutation(ii)]
    Dp["BG"] = bgp
    PLAC.append(loco(Dp, BASELINES["full"])[3])
PLAC = np.array(PLAC)
put("plac_lo", q(PLAC, 5), 1)
put("plac_hi", q(PLAC, 95), 1, plus=True)
put("n_shuffle", len(PLAC), 0)
put("plac_p", 100 * np.mean(np.abs(PLAC) >= abs(TAB["full"]["est"][3])), 0)

full_boot = TAB["full"]["boot"][:, 3]
put("p_rope", 100 * np.mean(np.abs(full_boot) < MARGIN), 0)
put("p_pos3", 100 * np.mean(full_boot >= MARGIN), 0)
# one-sided placebo test of the claim a club would act on (harder cut -> more open)
put("plac_p1", np.mean(PLAC >= TAB["full"]["est"][3]), 2)

# class interaction: held-out slope inside each class (fit on the other two)
for j, c in enumerate(CLASSES):
    put(f"cls{c}_b", TAB["full"]["est"][4 + j], 1, plus=True)
    put(f"cls{c}_lo", TAB["full"]["lo"][4 + j], 1, plus=True)
    put(f"cls{c}_hi", TAB["full"]["hi"][4 + j], 1, plus=True)
    put(f"cls{c}_n", int((D.draft_year == c).sum()), 0)
RES_FULL = IN_YD * loco(D, BASELINES["full"], return_res=True)

# team-clustered bootstrap: receivers on the same team share a quarterback and scheme, so resample whole teams
team_main = ss.sort_values("routes").groupby("nfl_id").team.last()
D["team"] = team_main.reindex(D.index).fillna("UNK").to_numpy()
tg = {t_: np.where(D.team.to_numpy() == t_)[0] for t_ in pd.unique(D.team)}
tk = list(tg)
rng = rng_for("boot_team")
bt_team = []
for _ in range(NBOOT):
    ix = np.concatenate([tg[tk[k]] for k in rng.integers(0, len(tk), len(tk))])
    if len(np.unique(D.draft_year.to_numpy()[ix])) == 3:
        bt_team.append(loco(D.iloc[ix], BASELINES["full"])[[2, 3]])
bt_team = np.array(bt_team)
put("n_teams", len(tk), 0)
put("team_b_lo", q(bt_team[:, 1], 5), 1)
put("team_b_hi", q(bt_team[:, 1], 95), 1, plus=True)
put("team_dr2_p1", np.mean(bt_team[:, 0] <= 0), 2)

# no-tracking cut benchmarks: official 3-cone and short shuttle (faster = positive), same held-out recipe
for nm, col in (("cone", "three_cone"), ("shut", "short_shuttle")):
    Dt = D[D[col].notna()].copy()
    Dt["OFF"] = -Dt.groupby(["draft_year", "position_group"])[col].transform(zs)
    Dt = Dt[Dt.OFF.notna()]
    e = loco(Dt, BASELINES["full"], "OFF")
    rng = rng_for("boot_" + nm)
    bb = []
    for _ in range(NBOOT):
        ix = rng.choice(np.arange(len(Dt)), len(Dt))
        if len(np.unique(Dt.draft_year.to_numpy()[ix])) == 3:
            bb.append(loco(Dt.iloc[ix], BASELINES["full"], "OFF")[3])
    put(f"{nm}_b", e[3], 1)
    put(f"{nm}_lo", q(bb, 5), 1)
    put(f"{nm}_hi", q(bb, 95), 1, plus=True)
    put(f"{nm}_n", len(Dt), 0)

# braking alone (the one part that carries to Sundays, see transfer grid): same held-out test
Dk = D.copy()
Dk["BRK"] = zs(Dk.decel_z)
eb = loco(Dk, BASELINES["full"], "BRK")
rng = rng_for("boot_brk")
bb = []
for _ in range(NBOOT):
    ix = rng.choice(np.arange(len(Dk)), len(Dk))
    if len(np.unique(Dk.draft_year.to_numpy()[ix])) == 3:
        bb.append(loco(Dk.iloc[ix], BASELINES["full"], "BRK")[3])
put("brk_b", eb[3], 1)
put("brk_lo", q(bb, 5), 1)
put("brk_hi", q(bb, 95), 1, plus=True)

# ---------------------------------------------------------------------------- measurement-error simulation
# Generate a true cut skill T, the drill reading the Combine would have produced for each receiver given his number of
# graded drills (same noise as the real data), the shrunk Break G from it, and a career SOE = real SOE shuffled across
# receivers (keeps real noise and route weights, removes any real link) + beta * T. Then run the same held-out test.
obs_b = TAB["full"]["est"][3]
rng = rng_for("me_sim")
BETAS = [0, 1, 2, 3, 4, 5, 6]
SIM = {}
ev_ = D.e_var.to_numpy(float)
relv = 1 / (1 + ev_)
for b in BETAS:
    out = []
    for _ in range(NSIM):
        T = rng.standard_normal(len(D))
        obs = T + rng.standard_normal(len(D)) * np.sqrt(ev_)
        Ds = D.copy()
        Ds["BG"] = relv * obs
        perm = rng.permutation(len(D))  # outcome moves with its own route weight
        Ds["ssoe"] = D.ssoe.to_numpy()[perm] + b * T / IN_YD
        Ds["routes"] = D.routes.to_numpy()[perm]
        out.append(loco(Ds, BASELINES["full"])[3])
    SIM[b] = np.array(out)
put("sim_p3", 100 * np.mean(SIM[3] <= obs_b), 1)
put("sim_mean3", np.mean(SIM[3]), 1)
put("sim_sd", np.std(SIM[0]), 1)
put("sim_pow3", 100 * np.mean(SIM[3] > 1.645 * np.std(SIM[0])), 0)
ruled = [b for b in BETAS if np.mean(SIM[b] <= obs_b) < 0.05]
put("sim_rule", min(ruled) if ruled else 99, 0)

# ---------------------------------------------------------------------------- what clubs pay for it (draft slot)
Xd = X.copy()
cols = ["BG", "forty_f", "combine_weight_f", "ngs_athleticism_score_f", "te"]
Ad = np.column_stack([np.ones(len(Xd))] + [Xd[c].to_numpy(float) for c in cols])
bd = np.linalg.lstsq(Ad, Xd.logpick.to_numpy(float), rcond=None)[0][1]
rng = rng_for("price")
bb = []
for _ in range(NBOOT):
    ix = rng.integers(0, len(Xd), len(Xd))
    bb.append(np.linalg.lstsq(Ad[ix], Xd.logpick.to_numpy(float)[ix], rcond=None)[0][1])
put("price", 100 * (1 - np.exp(bd)), 0)
put("price_lo", 100 * (1 - np.exp(q(bb, 95))), 0)
put("price_hi", 100 * (1 - np.exp(q(bb, 5))), 0)

# ---------------------------------------------------------------------------- fixed Monday rule: top third in class
Xd["top3"] = Xd.groupby("draft_year").BG.transform(lambda s: s >= s.quantile(2 / 3))
per_class = Xd.groupby("draft_year").top3.sum()
put("t3_per_class", per_class.mean(), 0)
put("t3_class_n", Xd.groupby("draft_year").size().mean(), 0)
gap = Xd.loc[Xd.top3, "BG"].mean() - Xd.loc[~Xd.top3, "BG"].mean()
put("t3_gap_sd", gap, 1)
put("t3_price", 100 * (1 - np.exp(bd * gap)), 0)
med_pick_t3 = float(Xd.loc[Xd.top3 & Xd.draft_overall_pick.notna(), "draft_overall_pick"].median())
put("t3_med_pick", med_pick_t3, 0)
put("t3_slots", med_pick_t3 * (1 - np.exp(bd * gap)), 0)
Dt3 = D.join(Xd[["top3"]], how="left")
Dt3["top3"] = Dt3.top3.fillna(False).astype(bool)


def t3_diff(dd):
    a, b = dd[dd.top3], dd[~dd.top3]
    return IN_YD * (np.average(a.ssoe, weights=a.routes) - np.average(b.ssoe, weights=b.routes))


rng = rng_for("t3")
bb = [t3_diff(Dt3.iloc[rng.integers(0, len(Dt3), len(Dt3))]) for _ in range(NBOOT)]
put("t3_soe", t3_diff(Dt3), 1, plus=True)
put("t3_soe_lo", q(bb, 5), 1, plus=True)
put("t3_soe_hi", q(bb, 95), 1, plus=True)
put("t3_n_soe", int(Dt3.top3.sum()), 0)

# ============================================================================ 3. NFL snaps (sampled), same code
gp = pd.read_parquet(P / "game_plays_sample.parquet")
gp = gp[gp.week.between(1, 18)]
gp = gp.merge(players[["nfl_id", "position_group"]], on="nfl_id")
gp = gp[gp.position_group.isin(["WR", "TE"])].copy()
gf = pd.read_parquet(P / "game_frames_sample.parquet",
                     columns=["game_id", "play_id", "nfl_id", "frame", "x", "y", "s", "event"])
gf = gf.merge(gp[["game_id", "play_id", "nfl_id"]], on=["game_id", "play_id", "nfl_id"])
pm = gf.groupby(["nfl_id", "game_id", "play_id"]).s.max().groupby("nfl_id")
vtop = pm.quantile(0.9)[pm.size() >= 8]
fam_keys = set(map(tuple, gp.loc[gp.route_ran.isin(FAMILIES), ["game_id", "play_id", "nfl_id"]].to_numpy()))
rows = []
for k, d in gf.groupby(["game_id", "play_id", "nfl_id"], sort=False):
    if k not in fam_keys:
        continue
    d = d.sort_values("frame")
    e = d.event.fillna("").to_numpy()
    if not ((e == "ball_snap").any() and (e == "pass_forward").any()):
        continue
    i0, ipf = int(np.argmax(e == "ball_snap")), int(np.argmax(e == "pass_forward"))
    r = break_mechanics(d.x.to_numpy(), d.y.to_numpy(), i0, min(len(d), ipf + 6))
    if r:
        rows.append(dict(game_id=k[0], play_id=k[1], nfl_id=k[2], ipf=ipf, **r))
g = pd.DataFrame(rows).merge(gp[["game_id", "play_id", "nfl_id", "route_ran", "position_group", "sample_kind",
                                 "separation_at_pass_forward", "cushion", "season", "week", "possession_team",
                                 "quarter", "game_clock", "target", "pass_result", "play_description",
                                 "yards_gained"]],
                             on=["game_id", "play_id", "nfl_id"])
g["family"] = g.route_ran
for m in COMP:
    g[m + "_z"] = g.groupby(["family", "position_group"])[m].transform(zs)
g = g.sort_values(["nfl_id", "game_id", "play_id"])
g["half"] = g.groupby("nfl_id").cumcount() % 2
put("n_nfl_breaks", len(g), 0)
put("n_nfl_rec", g.nfl_id.nunique(), 0)
put("share_random", 100 * (g.sample_kind == "random").mean(), 0)

# ---------------------------------------------------------------------------- same-snap check on Sundays
# Does a harder cut go with more separation at the throw on the same snap? Within receiver x route family
# (each man against his own other breaks of that family), controlling for cushion and for how long before the throw
# the cut was finished. Timing placebo: cuts finished after the ball is out cannot create separation at the throw.
g["t_lead"] = (g.ipf - g.apex) * DT
g["sep"] = g.separation_at_pass_forward
put("share_post", 100 * (g.t_lead < 0).mean(), 0)


def same_snap(h, name):
    h = h.copy()
    h["cushion"] = h.cushion.fillna(h.cushion.mean())
    h = h[h.groupby(["nfl_id", "family"]).sep.transform("size") >= 2]
    for c in ("sep", "cushion", "t_lead", "bg_z"):
        h[c + "_w"] = h[c] - h.groupby(["nfl_id", "family"])[c].transform("mean")

    def est(hh):
        A_ = np.column_stack([hh.bg_z_w, hh.cushion_w, hh.t_lead_w])
        return IN_YD * np.linalg.lstsq(A_, hh.sep_w.to_numpy(float), rcond=None)[0][0]

    grp = {i: d_ for i, d_ in h.groupby("nfl_id")}
    ids = np.array(list(grp))
    rng = rng_for("snap_" + name)
    bs = [est(pd.concat([grp[i] for i in rng.choice(ids, len(ids))])) for _ in range(min(NBOOT, 1000))]
    return est(h), q(bs, 5), q(bs, 95), len(h), len(ids)


SNAP = {
    "pre": same_snap(g[g.t_lead >= 0], "pre"),
    "pre_rand": same_snap(g[(g.t_lead >= 0) & (g.sample_kind == "random")], "pre_rand"),
    "pre_untgt": same_snap(g[(g.t_lead >= 0) & ~g.target.astype(bool)], "pre_untgt"),
    "post": same_snap(g[g.t_lead < 0], "post"),
}
for k_, v_ in SNAP.items():
    put(f"snap_{k_}", v_[0], 1, plus=True)
    put(f"snap_{k_}_lo", v_[1], 1, plus=True)
    put(f"snap_{k_}_hi", v_[2], 1, plus=True)
    put(f"snap_{k_}_n", v_[3], 0)
    put(f"snap_{k_}_r", v_[4], 0)

# league entry speed by family: drill vs NFL
ENTRY = {}
for f in FAMILIES:
    ENTRY[f] = (cb.loc[cb.family == f, "v_in"].median(), g.loc[g.family == f, "v_in"].median(),
                cb.loc[cb.family == f, "bg"].median(), g.loc[g.family == f, "bg"].median())
put("hitch_in_drill", ENTRY["HITCH"][0], 1)
put("hitch_in_nfl", ENTRY["HITCH"][1], 1)
put("hitch_in_drill_mph", ENTRY["HITCH"][0] * MPH, 0)
put("hitch_in_nfl_mph", ENTRY["HITCH"][1] * MPH, 0)
put("hitch_bg_drill", ENTRY["HITCH"][2], 2)
put("hitch_bg_nfl", ENTRY["HITCH"][3], 2)
put("n_fam_faster", sum(ENTRY[f][0] > ENTRY[f][1] for f in FAMILIES), 0)
put("share_nfl_at_drill", 100 * np.mean([(g.loc[g.family == f, "v_in"] >= ENTRY[f][0]).mean() for f in FAMILIES]
                                        ), 0)

# ---------------------------------------------------------------------------- transfer grid, same route family
drill_rf = cb.groupby(["nfl_id", "family"])[[m + "_z" for m in COMP]].mean()
nfl_rf = g.groupby(["nfl_id", "family"])[[m + "_z" for m in COMP]].mean()
nfl_rf["nb"] = g.groupby(["nfl_id", "family"]).size()
pairs = drill_rf.join(nfl_rf, lsuffix="_d", rsuffix="_n", how="inner")
pairs = pairs[pairs.nb >= 2].reset_index()


def boot_rho(a, b, name, cl=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = rng_for(name)
    if cl is None:
        groups = [np.array([i]) for i in range(len(a))]
    else:
        cl = np.asarray(cl)
        groups = [np.where(cl == c)[0] for c in pd.unique(cl)]
    out = []
    for _ in range(NBOOT):
        ix = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        out.append(spearmanr(a[ix], b[ix])[0])
    return float(spearmanr(a, b)[0]), q(out, 5), q(out, 95), len(a)


GRID = {}
for f in FAMILIES + ["ALL"]:
    sub = pairs if f == "ALL" else pairs[pairs.family == f]
    for m in COMP:
        if f == "ALL":
            GRID[(f, m)] = boot_rho(sub[m + "_z_d"], sub[m + "_z_n"], f"grid_{f}_{m}", cl=sub.nfl_id)
        else:
            GRID[(f, m)] = (float(spearmanr(sub[m + "_z_d"], sub[m + "_z_n"])[0]), np.nan, np.nan, len(sub))
put("pair_n", len(pairs), 0)
put("pair_rec", pairs.nfl_id.nunique(), 0)
for m in COMP:
    put(f"all_{m}", GRID[("ALL", m)][0])
put("all_decel_lo", GRID[("ALL", "decel")][1])
put("all_decel_hi", GRID[("ALL", "decel")][2])
put("all_bg_lo", GRID[("ALL", "bg")][1])
put("all_bg_hi", GRID[("ALL", "bg")][2])
put("all_vin_lo", GRID[("ALL", "v_in")][1])
put("all_vin_hi", GRID[("ALL", "v_in")][2])
cell_bg = [GRID[(f, "bg")][0] for f in FAMILIES]
put("fam_bg_min", min(cell_bg))
put("fam_bg_max", max(cell_bg))
# multiplicity: max |rho| over all 30 cells when the NFL side is shuffled among receivers within family
NCOL = [m + "_z_n" for m in COMP]
DCOL = [m + "_z_d" for m in COMP]


def grid_max(pp):
    out = []
    for f in FAMILIES + ["ALL"]:
        sub = pp if f == "ALL" else pp[pp.family == f]
        for m in COMP:
            out.append(abs(spearmanr(sub[m + "_z_d"], sub[m + "_z_n"])[0]))
    return np.nanmax(out)


rng = rng_for("grid_max")
fam_idx = {f: np.where(pairs.family.to_numpy() == f)[0] for f in FAMILIES}
MX = []
for _ in range(500):
    pp = pairs.copy()
    nv = pp[NCOL].to_numpy().copy()
    for f, ii in fam_idx.items():
        nv[ii] = nv[rng.permutation(ii)]
    pp[NCOL] = nv
    MX.append(grid_max(pp))
MX = np.array(MX)
put("n_cells", 5 * (len(FAMILIES) + 1), 0)
put("brk_adj_p", np.mean(MX >= abs(GRID[("ALL", "decel")][0])), 2)
put("hitch_bg_rho", GRID[("HITCH", "bg")][0])
put("hitch_n", GRID[("HITCH", "bg")][3], 0)

# entry speed removed on both sides: residualise Break G on entry speed within family, then transfer
cb["bg_r"] = cb.groupby("family", group_keys=False).apply(
    lambda d: d.bg_z - np.polyval(np.polyfit(d.v_in_z, d.bg_z, 1), d.v_in_z), include_groups=False)
g["bg_r"] = g.groupby("family", group_keys=False).apply(
    lambda d: d.bg_z - np.polyval(np.polyfit(d.v_in_z, d.bg_z, 1), d.v_in_z), include_groups=False)
pr2 = cb.groupby(["nfl_id", "family"]).bg_r.mean().to_frame("d").join(
    g.groupby(["nfl_id", "family"]).bg_r.agg(["mean", "size"]), how="inner")
pr2 = pr2[pr2["size"] >= 2].reset_index()
r_res = boot_rho(pr2.d, pr2["mean"], "resid", cl=pr2.nfl_id)
put("res_rho", r_res[0])
put("res_lo", r_res[1])
put("res_hi", r_res[2])
put("rho_bg_vin_drill", spearmanr(cb.v_in, cb.bg)[0])
put("rho_bg_vin_nfl", spearmanr(g.v_in, g.bg)[0])

# receiver level: overall drill Break G vs his NFL Break G (8+ sampled breaks), and the 40 -> top speed control
nfl_rec = g.groupby("nfl_id").agg(nbg=("bg_z", "mean"), nb=("bg_z", "size"), nbg_sd=("bg_z", "std"))
h = g.groupby(["nfl_id", "half"]).bg_z.mean().unstack()
R = X.join(nfl_rec, how="inner")
R = R[R.nb >= 8]
r_bg = boot_rho(R.BG, R.nbg, "rec_bg")
put("rec_rho", r_bg[0])
put("rec_lo", r_bg[1])
put("rec_hi", r_bg[2])
put("rec_n", r_bg[3], 0)
hh = h.loc[R.index].dropna()
r_half = spearmanr(hh[0], hh[1])[0]
rel_nfl = 2 * r_half / (1 + r_half)
ceil_ = np.sqrt(max(rel_nfl, 0) * R.rel.mean())
put("rel_nfl", rel_nfl)
put("ceiling", ceil_)
put("link", 0.5, 1)
put("true05", 0.5 * ceil_)
S = pb[["forty_s", "position_group"]].join(vtop.rename("vtop"), how="inner").dropna()
S = S[S.index.isin(X.index)]
S["vtop_z"] = S.groupby("position_group").vtop.transform(zs)  # within position, like the year-standardised 40
r40 = boot_rho(-S.forty_s, S.vtop_z, "forty")
put("f40_rho", r40[0])
put("f40_lo", r40[1])
put("f40_hi", r40[2])
put("f40_n", r40[3], 0)
# sampling halves: random snaps only
gr = g[g.sample_kind == "random"].groupby("nfl_id").bg_z.agg(["mean", "size"])
Rr = X.join(gr, how="inner")
Rr = Rr[Rr["size"] >= 4]
r_rand = boot_rho(Rr.BG, Rr["mean"], "rand")
put("rand_rho", r_rand[0])
put("rand_lo", r_rand[1])
put("rand_hi", r_rand[2])
put("rand_n", r_rand[3], 0)

# ============================================================================ 4. named receivers
E = D.join(nfl_rec, how="inner")
E = E[(E.position_group == "WR") & (E.routes >= 200) & (E.nb >= 5)].copy()
E = E.sort_values("BG", ascending=False)
E["rank"] = np.arange(1, len(E) + 1)
E["class_rank"] = E.groupby("draft_year").BG.rank(ascending=False).astype(int)
E["class_n"] = E.groupby("draft_year").BG.transform("size").astype(int)
put("n_elig", len(E), 0)
top8, bot8 = E.head(8), E.tail(8)
put("top8_soe", IN_YD * np.average(top8.ssoe, weights=top8.routes), 1, plus=True)
put("bot8_soe", IN_YD * np.average(bot8.ssoe, weights=bot8.routes), 1, plus=True)
put("e_rho_soe", spearmanr(E.BG, E.ssoe)[0])
# receivers' own NFL cut grade vs their career SOE (between receivers, 8+ sampled breaks)
Rn = D.join(nfl_rec, how="inner")
Rn = Rn[Rn.nb >= 8]
r_ns = boot_rho(Rn.nbg, Rn.ssoe, "nfl_soe")
put("nsoe_rho", r_ns[0])
put("nsoe_lo", r_ns[1])
put("nsoe_hi", r_ns[2])
put("nsoe_n", r_ns[3], 0)


for nm, key in (("Arian Smith", "smith"), ("Puka Nacua", "nacua")):
    rr = E[E.display_name == nm].iloc[0]
    put(f"{key}_rank", int(rr["rank"]), 0)
    put(f"{key}_soe", IN_YD * rr.ssoe, 1, plus=True)
    put(f"{key}_routes", rr.routes, 0)


def three_speeds(pid):
    fams = sorted(set(cb.loc[cb.nfl_id == pid, "family"]) & set(g.loc[g.nfl_id == pid, "family"]))
    dd = cb[(cb.nfl_id == pid) & cb.family.isin(fams)]
    nn = g[(g.nfl_id == pid) & g.family.isin(fams)]
    return (dd[["v_in", "v_apex", "v_out", "bg"]].median(), nn[["v_in", "v_apex", "v_out", "bg"]].median(),
            fams, len(dd), len(nn))


mid = E.iloc[[len(E) // 2]]
NAMED = []
ROWS_NAMED = [("top", E.iloc[0]), ("smi", E[E.display_name == "Arian Smith"].iloc[0]),
              ("nac", E[E.display_name == "Puka Nacua"].iloc[0]), ("mid", mid.iloc[0]), ("bot", E.iloc[-1])]
for tag, row in ROWS_NAMED:
    pid = row.name
    dmed, nmed, fams, nd_, nn_ = three_speeds(pid)
    nm = row.display_name
    NAMED.append(dict(tag=tag, pid=pid, name=nm, rank=int(row["rank"]), d=dmed, n=nmed, fams=fams,
                      soe=IN_YD * row.ssoe, nd=nd_, nn=nn_, bgz=row.BG, nbg=row.nbg))
    put(f"{tag}_name", nm)
    put(f"{tag}_rank", int(row["rank"]), 0)
    put(f"{tag}_bgz", row.BG, 1, plus=True)
    put(f"{tag}_nbg", row.nbg, 1, plus=True)
    for side, med in (("d", dmed), ("n", nmed)):
        put(f"{tag}_{side}in", med.v_in, 1)
        put(f"{tag}_{side}ap", med.v_apex, 1)
        put(f"{tag}_{side}out", med.v_out, 1)
        put(f"{tag}_{side}g", med.bg, 2)
        put(f"{tag}_{side}mph", (med.v_in - med.v_apex) * MPH, 1)
    put(f"{tag}_soe", IN_YD * row.ssoe, 1, plus=True)
    put(f"{tag}_nd", nd_, 0)
    put(f"{tag}_nn", nn_, 0)
    # clip to pull: his random-sample, pre-throw break in a shared family closest to his own NFL median Break G
    cand_ = g[(g.nfl_id == pid) & g.family.isin(fams) & (g.t_lead >= 0)]
    cand_ = cand_[cand_.sample_kind == "random"] if (cand_.sample_kind == "random").any() else cand_
    cl_ = cand_.iloc[int(np.argmin(np.abs(cand_.bg - nmed.bg)))]
    put(f"{tag}_clip", f"{int(cl_.season)} wk {int(cl_.week)}, Q{int(cl_.quarter)} {cl_.game_clock}, "
                       f"{FAM_LABEL[cl_.family].split(' /')[0].lower()}")
    put(f"{tag}_clip_g", cl_.bg, 2)
    put(f"{tag}_clip_sep", cl_.sep, 1)
    if bool(cl_.target):
        res_ = (f"caught, {int(cl_.yards_gained)} yd" if cl_.pass_result == "C" else
                "incomplete" if cl_.pass_result == "I" else "intercepted" if cl_.pass_result == "IN" else "sacked")
        put(f"{tag}_clip_log", f"targeted, {res_}")
    else:
        put(f"{tag}_clip_log", "not targeted")
    put(f"{tag}_class_rank", int(row.class_rank), 0)
    put(f"{tag}_class_n", int(row.class_n), 0)
    put(f"{tag}_year", int(row.draft_year), 0)
    put(f"{tag}_pick", "UDFA" if pd.isna(row.draft_overall_pick) else f"pick {int(row.draft_overall_pick)}")
    put(f"{tag}_routes", row.routes, 0)
    put(f"{tag}_cush_pct", 100 * (D.cush < row.cush).mean(), 0)
    NAMED[-1]["clip"] = cl_

# ============================================================================ figures
from matplotlib.collections import LineCollection  # noqa: E402

SPEED_NORM = plt.Normalize(0, 9)
CMAP = plt.get_cmap("viridis")
CLS_C = {2023: OI["purple"], 2024: OI["green"], 2025: OI["verm"]}


def draw_paths(ax, df, k, seed, lw=1.1, alpha=0.75, color=None):
    rng_ = rng_for(seed)
    idx = rng_.choice(len(df), min(k, len(df)), replace=False) if len(df) else []
    for i in idx:
        r = df.iloc[i]
        pts = np.column_stack([r.px, r.py]).reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        if color is None:
            lc = LineCollection(seg, cmap=CMAP, norm=SPEED_NORM, lw=lw, alpha=alpha)
            lc.set_array(0.5 * (r.ps[:-1] + r.ps[1:]))
        else:
            lc = LineCollection(seg, colors=color, lw=lw, alpha=alpha)
        ax.add_collection(lc)
    ax.plot(0, 0, "o", color="black", ms=5, zorder=5)
    return len(idx)


# Fig 1 (hero): route paths around the plant, drill vs NFL, coloured by speed; then the speed traces
fig = plt.figure(figsize=(FIGW, 8.9), layout="constrained")
gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 1, 0.07], height_ratios=[1.15, 1.15, 0.9])
tt = np.arange(-10, 11) * DT
KP = 35
for j, f in enumerate(FAMILIES):
    for i, (src, lab) in enumerate(((cb, "Combine drill"), (g, "NFL snap"))):
        ax = fig.add_subplot(gs[i, j])
        sub = src[src.family == f]
        draw_paths(ax, sub, KP, f"paths_{f}_{i}")
        ax.set_xlim(-3.5, 7.5)
        ax.set_ylim(-9, 5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.add_patch(plt.Rectangle((-3.5, -9), 11, 14, fill=False, ec="#dddddd", lw=0.8))
        med_in = sub.v_in.median()
        ax.text(7.2, -8.6, f"in at {med_in:.1f} yd/s\n{len(sub)} {'cuts' if i == 0 else 'breaks'}",
                ha="right", va="bottom", fontsize=9.5, color="#222222")
        if i == 0:
            ax.set_title(FAM_LABEL[f].replace(" / ", "/"), fontsize=12)
        if j == 0:
            ax.set_ylabel(lab, fontsize=12, fontweight="bold", color=DRILL_C if i == 0 else NFL_C)
            ax.yaxis.label.set_visible(True)
    ax = fig.add_subplot(gs[2, j])
    for src, col in ((cb, DRILL_C), (g, NFL_C)):
        T = np.vstack(src.loc[src.family == f, "trace"].to_numpy())
        med, lo, hi = np.nanmedian(T, 0), np.nanpercentile(T, 25, 0), np.nanpercentile(T, 75, 0)
        thin = np.mean(~np.isnan(T), 0) < 0.95
        med[thin], lo[thin], hi[thin] = np.nan, np.nan, np.nan
        ax.fill_between(tt, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(tt, med, color=col, lw=3)
    ax.axvline(0, color="#444444", lw=0.8, ls=":")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["-1 s", "plant", "+1 s"])
    ax.set_ylim(0, 9.5)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    if j == 0:
        ax.set_ylabel("Speed (yd/s)")
    else:
        ax.set_yticklabels([])
    if j == 1:
        ax.text(0, 7.4, "Combine", color=DRILL_C, ha="center", fontsize=11, fontweight="bold")
        ax.text(0, 2.4, "NFL", color=NFL_C, ha="center", fontsize=11, fontweight="bold")
cax = fig.add_subplot(gs[0:2, 5])
cb_ = fig.colorbar(plt.cm.ScalarMappable(norm=SPEED_NORM, cmap=CMAP), cax=cax)
cb_.set_label("Speed (yd/s)")
fig.suptitle(f"Same route, two worlds: Combine receivers hit a scripted cut hot; on Sundays they arrive "
             f"slower and break where coverage allows\n{KP} random paths per panel, stem up, break to the right, plant = black dot; "
             f"bottom row: median speed and middle half of reps", fontsize=12.5, fontweight="bold")
save(fig, "fig1_route_paths.png")

# Fig 2: transfer, one dot per receiver: the 40 carries to Sunday speed; the cut grade does not
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 5.3), layout="constrained")
a1.scatter(-S.forty_s, S.vtop_z, s=36, color=OI["green"], alpha=0.75, edgecolor="white", lw=0.5)
a1.set_xlabel("Official 40, sd faster than his class")
a1.set_ylabel("NFL top speed in sampled snaps (sd)")
a1.set_title(f"A. Speed carries over\nrank corr. {r40[0]:.2f} (90%: {r40[1]:.2f} to {r40[2]:.2f}), "
             f"{r40[3]} receivers", loc="left", fontsize=12)
a2.scatter(R.BG, R.nbg, s=36, color=OI["verm"], alpha=0.75, edgecolor="white", lw=0.5)
a2.set_xlabel("Combine Break G (sd, shrunk for few drills)")
a2.set_ylabel("NFL Break G, same code (sd)")
a2.set_title(f"B. The cut grade does not\nrank corr. {r_bg[0]:.2f} (90%: {r_bg[1]:.2f} to {r_bg[2]:.2f}), "
             f"{r_bg[3]} receivers", loc="left", fontsize=12)
for ax, xv, yv in ((a1, -S.forty_s, S.vtop_z), (a2, R.BG, R.nbg)):
    b_ = np.polyfit(xv, yv, 1)
    xx = np.linspace(np.nanmin(xv), np.nanmax(xv), 20)
    ax.plot(xx, np.polyval(b_, xx), color="#333333", lw=1.5)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
for nd in NAMED:
    if nd["pid"] in R.index:
        rr = R.loc[nd["pid"]]
        a2.annotate(nd["name"], (rr.BG, rr.nbg), xytext=(6, 6), textcoords="offset points", fontsize=9.5,
                    color="#222222", arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6))
fig.suptitle("One dot per receiver: the 40 shows up on Sundays, the Combine cut does not", fontsize=13,
             fontweight="bold")
save(fig, "fig2_transfer.png")

# Fig 3: held-out test as data: one dot per receiver, residual vs the no-tracking forecast, line per class
fig = plt.figure(figsize=(FIGW, 5.8), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1])
ax = fig.add_subplot(gs[0])
xr = np.linspace(D.BG.min() - 0.1, D.BG.max() + 0.1, 50)
YL = 25
ax.fill_between(xr, -MARGIN * xr, MARGIN * xr, color=OI["sky"], alpha=0.2, lw=0)
yplot = np.clip(RES_FULL, -YL + 1, YL - 1)
off = np.abs(RES_FULL) > YL - 1
cls_handles = []
for c in CLASSES:
    m_ = (D.draft_year == c).to_numpy()
    ax.scatter(D.BG[m_ & ~off], yplot[m_ & ~off], s=10 + 0.12 * D.routes[m_ & ~off], color=CLS_C[c], alpha=0.6,
               edgecolor="white", lw=0.5)
    ax.scatter(D.BG[m_ & off], yplot[m_ & off], s=40, marker="^", color=CLS_C[c], alpha=0.6)
    bc = float(NUM[f"cls{c}_b"])
    xc = np.linspace(D.BG[m_].min(), D.BG[m_].max(), 10)
    a0_ = np.average(RES_FULL[m_], weights=D.routes[m_]) - bc * np.average(D.BG[m_], weights=D.routes[m_])
    ax.plot(xc, a0_ + bc * xc, color=CLS_C[c], lw=2.5)
    cls_handles.append(Line2D([], [], color=CLS_C[c], lw=2.5, marker="o",
                              label=f"{c} class, {int(m_.sum())} receivers: {bc:+.1f} in per sd"))
ax.legend(handles=cls_handles, loc="upper left", frameon=False, fontsize=10, title="Held-out slope by class",
          title_fontsize=10.5, alignment="left")
ax.text(xr[-1], MARGIN * xr[-1] + 1.5, "±3 in per sd: smallest\neffect worth acting on", color=OI["blue"],
        fontsize=9.5, va="bottom", ha="right")
NOFF = {"Kayshon Boutte": (-10, 34), "Arian Smith": (-20, 20), "Puka Nacua": (-5, -26), "Ryan Flournoy": (5, -28),
        "Jake Bobo": (5, 20)}
for nd in NAMED:
    if nd["pid"] in D.index:
        k_ = D.index.get_loc(nd["pid"])
        ax.annotate(nd["name"], (D.BG.iloc[k_], yplot[k_]), xytext=NOFF.get(nd["name"], (8, 8)),
                    textcoords="offset points", fontsize=9.5, ha="center",
                    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.7))
ax.axhline(0, color="#999999", lw=0.8)
ax.set_ylim(-YL, YL)
ax.set_xlim(xr[0], xr[-1] + 0.05)
ax.set_xlabel("Combine Break G (sd within class)")
ax.set_ylabel("Career SOE minus no-tracking forecast (in)")
ax.set_title(f"A. Each class scored by a model fit on the other two\n({len(D)} receivers; dot size = NFL routes; "
             f"triangles = off scale, {int(off.sum())} low-route receivers)", loc="left", fontsize=12)
a2 = fig.add_subplot(gs[1])
a2.hist(PLAC, bins=36, color=OI["sky"], alpha=0.85, edgecolor="white", orientation="vertical")
a2.axvline(TAB["full"]["est"][3], color=OI["black"], lw=2.5)
a2.text(TAB["full"]["est"][3] - 0.2, a2.get_ylim()[1] * 0.95, f"real\nBreak G\n{TAB['full']['est'][3]:.1f}",
        ha="right", va="top", fontsize=10.5)
a2.axvline(q(PLAC, 5), color="#555555", lw=1, ls=":")
a2.axvline(q(PLAC, 95), color="#555555", lw=1, ls=":")
a2.set_yticks([])
a2.set_xlabel("Pooled effect, inches per sd")
a2.set_title(f"B. Break G shuffled within\nclass, {len(PLAC)} times", loc="left", fontsize=12)
fig.suptitle("Harder Combine cutters are not more open on Sundays than draft slot and testing already predict",
             fontsize=13, fontweight="bold")
save(fig, "fig3_heldout.png")

# Fig 4: named receivers, his own drill cuts vs his own NFL breaks of the same family
fig, axs = plt.subplots(1, len(NAMED), figsize=(FIGW, 4.6), layout="constrained")
for ax, nd in zip(axs, NAMED):
    pid = nd["pid"]
    fams = nd["fams"]
    cnt = g[(g.nfl_id == pid) & g.family.isin(fams)].family.value_counts()
    f = cnt.index[0]
    dd = cb[(cb.nfl_id == pid) & (cb.family == f)]
    nn = g[(g.nfl_id == pid) & (g.family == f)]
    draw_paths(ax, nn, 12, f"n4_{pid}", lw=1.4, alpha=0.6, color=NFL_C)
    draw_paths(ax, dd, 12, f"d4_{pid}", lw=2.8, alpha=0.95, color=DRILL_C)
    ax.set_xlim(-3.5, 7.5)
    ax.set_ylim(-9, 5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(plt.Rectangle((-3.5, -9), 11, 14, fill=False, ec="#dddddd", lw=0.8))
    ax.set_title(f"{nd['name']}\n#{nd['rank']} of {len(E)} drill cutters", fontsize=10.5)
    ax.set_xlabel(f"{FAM_LABEL[f].split(' /')[0]}: {len(dd)} drill, {len(nn)} NFL\n"
                  f"Break G {dd.bg.median():.2f} g drill, {nn.bg.median():.2f} g NFL\ncareer SOE {nd['soe']:+.1f} in",
                  fontsize=10)
fig.legend(handles=[Line2D([], [], color=DRILL_C, lw=2.8, label="his Combine drill cuts"),
                    Line2D([], [], color=NFL_C, lw=1.4, label="his NFL breaks, same route family")],
           loc="outside lower center", ncols=2, frameon=False, fontsize=10.5)
fig.suptitle("Same man, same route: all five plant softer on Sundays, and drill rank (left to right) does not "
             "sort who gets open", fontsize=12.5, fontweight="bold")
save(fig, "fig4_named_paths.png")


# ============================================================================ write numbers and render
NUM_SORTED = dict(sorted(NUM.items()))
(OUT / "numbers.json").write_text(json.dumps(NUM_SORTED, indent=1) + "\n")
tpl_path = OUT / "writeup_template.md"
if tpl_path.exists():
    tpl = tpl_path.read_text()
    missing = sorted(set(re.findall(r"\{(\w+)\}", tpl)) - set(NUM))
    if missing:
        raise SystemExit(f"template keys missing from numbers: {missing}")
    txt = re.sub(r"\{(\w+)\}", lambda m_: re.sub(r"^-", "−", NUM[m_.group(1)]), tpl)
    (OUT / "writeup.md").write_text(txt)
print(json.dumps(NUM_SORTED, indent=1))
