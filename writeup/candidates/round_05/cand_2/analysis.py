"""Break G: WR/TE break force in Combine route drills vs NFL targets and separation.

Round 5, candidate 2. Run from the repo root:
    uv run python writeup/candidates/round_05/cand_2/analysis.py

Reads data/processed/*.parquet only (about two minutes). Writes numbers.json, board.csv and figures/*.png next to
this file.

Definitions (identical code on drills and NFL snaps, 10 Hz x/y):
- velocity / acceleration: Savitzky-Golay derivatives of x/y (window 7, order 2);
- break: first >= 30 deg heading change (+-0.6 s) after 1.5 yd of travel, entered above 2.5 yd/s;
- Break G = mean acceleration magnitude (in g, 1 g = 10.73 yd/s^2) over the 1.0 s centred on the sharpest turn.
  It sums the braking into the cut, the sideways force of the turn and the push out of it;
- per rep, z-scored within drill x Combine year; player score = mean over drills, then empirical-Bayes shrunk
  toward the mean with the per-drill reliability (ICC), and reported in sd units across players.
Outcome: route-weighted NFL target rate (targets per route run) over seasons with 100+ routes (player_season_stats).
Baseline without tracking: log pick, official 40, weight, NGS athleticism (all year x position standardized), TE flag.
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
G_YD = 10.73  # 1 g in yd/s^2
RNG = np.random.default_rng(2027)
NBOOT = 2000
NPERM = 2000
LEAD_MIN = 0.2
MIN_ROUTES = 100

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 150,
                     "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10})
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
    # break time: seconds to cover 4 yd of path into and 4 yd out of the apex
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

# ----------------------------------------------------------------------------- Combine drills
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
    if len(bl):
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
put("n_drill_reps", len(ev)); put("n_drill_breaks", len(cb)); put("pct_drill_detected", 100 * len(cb) / len(ev), 0)
put("bg_drill_median", cb.bg.median()); put("bg_drill_p10", cb.bg.quantile(0.1)); put("bg_drill_p90", cb.bg.quantile(0.9))
for m in COMP:
    cb[m + "_z"] = cb.groupby(["drill_name", "draft_year"])[m].transform(zs)
cpd = cb.groupby(["nfl_id", "drill_name", "family"], as_index=False)[[m + "_z" for m in COMP] + ["bg"]].mean()
cplayer = cpd.groupby("nfl_id")[[m + "_z" for m in COMP]].mean()
cplayer["n_drills"] = cpd.groupby("nfl_id").size()
cplayer["bg_raw"] = cpd.groupby("nfl_id").bg.mean()
put("n_combine_players", len(cplayer))
put("median_drills_per_player", int(cplayer.n_drills.median()))


def icc1(df, col, gcol="nfl_id"):
    k = df.groupby(gcol).size()
    k0 = (len(df) - (k ** 2).sum() / len(df)) / (len(k) - 1)
    gm = df[col].mean()
    msb = (k * (df.groupby(gcol)[col].mean() - gm) ** 2).sum() / (len(k) - 1)
    msw = ((df[col] - df.groupby(gcol)[col].transform("mean")) ** 2).sum() / (len(df) - len(k))
    return (msb - msw) / (msb + (k0 - 1) * msw)


# reliability: split-half across disjoint drill sets, and per-drill ICC
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
put("rel_next_best", rel.drop("bg").rho.max())
NUM["rel_next_best_name"] = str(rel.drop("bg").rho.idxmax())
put("rel_sk", rel.rho["sk"])
icc_bg = icc1(cpd, "bg_z")
put("icc_bg", icc_bg)
# Empirical-Bayes shrinkage of the player mean toward 0 using per-drill ICC
var_tot = cpd.bg_z.var()
var_true, var_err = icc_bg * var_tot, (1 - icc_bg) * var_tot
nd = cplayer.n_drills
post_var = 1 / (1 / var_true + nd / var_err)
cplayer["bg_shr"] = post_var * (nd * cplayer.bg_z / var_err)
cplayer["bg_shr_sd"] = np.sqrt(post_var)
rel7 = 7 * icc_bg / (1 + 6 * icc_bg)
put("rel_7drills", rel7)
SD_SHR = np.sqrt(var_true)  # sd of the true (error-free) player Break G
cplayer["BG"] = cplayer.bg_shr / SD_SHR  # shrunk estimate in true-sd units
cplayer["bt_z"] = cplayer.bt_z.fillna(0.0)
cplayer["BG_lo"] = (cplayer.bg_shr - 1.2816 * cplayer.bg_shr_sd) / SD_SHR
cplayer["BG_hi"] = (cplayer.bg_shr + 1.2816 * cplayer.bg_shr_sd) / SD_SHR
# raw g per player sd of BG (for a coach unit)
# one sd of true Break G in g: sqrt(true-score variance) x pooled within drill-year sd of raw Break G
put("g_true_sd", float(np.sqrt(var_true) * cb.groupby(["drill_name", "draft_year"]).bg.std().mean()), 3)

# ----------------------------------------------------------------------------- baseline table
cr = cres.set_index("nfl_id")
pb = players.set_index("nfl_id")[["draft_overall_pick", "position_group", "draft_year", "display_name"]].join(
    cr[["forty", "combine_weight", "ngs_athleticism_score", "three_cone", "short_shuttle", "ten_yd_split"]])
pb = pb[pb.position_group.isin(["WR", "TE"])].copy()
pb["logpick"] = np.log(pb.draft_overall_pick.fillna(262))
pb["te"] = (pb.position_group == "TE").astype(float)
for c in ("forty", "combine_weight", "ngs_athleticism_score", "three_cone", "short_shuttle", "ten_yd_split"):
    pb[c + "_s"] = pb.groupby(["draft_year", "position_group"])[c].transform(zs).fillna(0.0)
BASE = ["logpick", "forty_s", "combine_weight_s", "ngs_athleticism_score_s", "te"]
X = cplayer.join(pb, how="inner")
put("rho_bg_forty", spearmanr(X.BG, X.forty, nan_policy="omit")[0])
put("rho_bg_pick", spearmanr(X.BG, X.draft_overall_pick.fillna(262))[0])
put("rho_bg_weight", spearmanr(X.BG, X.combine_weight, nan_policy="omit")[0])
put("rho_bg_3cone", spearmanr(X.BG, X.three_cone, nan_policy="omit")[0])
put("n_3cone", int(X.three_cone.notna().sum()))
put("rho_bg_vin", spearmanr(X.BG, X.v_in_z)[0])

# ----------------------------------------------------------------------------- NFL season outcomes
ss = pd.read_parquet(P / "player_season_stats.parquet")
ss = ss[ss.position_group.isin(["WR", "TE"])].copy()
ss["slot"] = (ss.main_alignment == "SLOT_WR").astype(float)
q100 = ss[ss.routes >= MIN_ROUTES]
put("rho_vin_forty", spearmanr(X.v_in_z, X.forty, nan_policy="omit")[0])
# primary outcome: regular-season targets per season since drafted, all 149 (0 if never targeted)
tot = ss.groupby("nfl_id").agg(t=("targets", "sum"), r=("routes", "sum"))
X = X.join(tot)
X["t"] = X.t.fillna(0.0)
X["tps"] = X.t / (2026 - X.draft_year)
X["tps_z"] = X.groupby("draft_year").tps.transform(zs)
TPS_SD = float(X.groupby("draft_year").tps.std().mean())
put("n_all", len(X)); put("tps_mean", X.tps.mean(), 1); put("tps_sd", TPS_SD, 1)
put("n_zero_targets", int((X.t == 0).sum()))
for cls in (2023, 2024, 2025):
    put(f"n_all_{cls}", int((X.draft_year == cls).sum()))


def season_agg(d):
    w = d.routes
    return pd.Series({"tr": np.average(d.target_rate, weights=w), "yprr": np.average(d.yards_per_route, weights=w),
                      "sep": np.average(d.avg_separation, weights=w), "slot": np.average(d.slot, weights=w),
                      "ypr": d.rec_yards.sum() / max(d.receptions.sum(), 1),
                      "epat": np.average(d.epa_per_target.fillna(0), weights=w),
                      "routes": d.routes.sum()})


agg = q100.groupby("nfl_id").apply(season_agg, include_groups=False)
Y = X.join(agg, how="inner").copy()
put("n_out", len(Y)); put("n_below_cut", len(X) - len(Y))
posY = X.index.get_indexer(Y.index)


def ols(df, cols, ycol):
    A = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    b, *_ = np.linalg.lstsq(A, df[ycol].to_numpy(float), rcond=None)
    res = df[ycol].to_numpy(float) - A @ b
    s2 = res @ res / (len(df) - A.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    return b, se


def resid_on(df, cols, v):
    A = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    b, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ b


# ---- screen: 9 Combine components x 4 outcomes, partial t of the component given the baseline
OUTCOMES = {"tps_z": "Targets per season", "tr": "Target rate", "yprr": "Yards per route", "sep": "NGS separation"}
SCREEN_X = [m + "_z" for m in COMP if m != "bg"] + ["BG"]
RY = {"tps_z": resid_on(X, BASE, X.tps_z.to_numpy(float))}
for yc in ("tr", "yprr", "sep"):
    RY[yc] = resid_on(Y, BASE, Y[yc].to_numpy(float))


def screen_t(xmat):
    """xmat: 149 x k component matrix aligned to X; returns dict outcome -> t vector"""
    out = {}
    for yc in OUTCOMES:
        df, xm_ = (X, xmat) if yc == "tps_z" else (Y, xmat[posY])
        rx = np.column_stack([resid_on(df, BASE, xm_[:, j]) for j in range(xm_.shape[1])])
        ry = RY[yc]
        r = (rx * ry[:, None]).sum(0) / np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum())
        out[yc] = r * np.sqrt(len(df) - len(BASE) - 2) / np.sqrt(1 - r ** 2)
    return out


xm = X[SCREEN_X].to_numpy(float)
tmat = pd.DataFrame(screen_t(xm), index=SCREEN_X)
best = tmat.abs().stack().idxmax()
put("screen_k", int(tmat.size))
put("t_main", tmat.loc["BG", "tps_z"]); put("t_tr", tmat.loc["BG", "tr"])
put("t_yprr", tmat.loc["BG", "yprr"]); put("t_sep", tmat.loc["BG", "sep"])
put("screen_best_is_main", int(best == ("BG", "tps_z")))
NUM["screen_best"] = f"{best[0]}|{best[1]}"
put("screen_mean_t2", float((tmat.to_numpy() ** 2).mean()))
put("screen_n_abs_t2", int((tmat.abs().to_numpy() > 2).sum()))
cls_arr = X.draft_year.to_numpy()


def perm_within_class():
    perm = np.arange(len(X))
    for c in (2023, 2024, 2025):
        ix = np.where(cls_arr == c)[0]
        perm[ix] = RNG.permutation(ix)
    return perm


maxt, single = [], []
for _ in range(NPERM):
    Tp = screen_t(xm[perm_within_class()])
    maxt.append(max(np.abs(v).max() for v in Tp.values()))
    single.append(abs(Tp["tps_z"][-1]))
put("p_fwer", (1 + (np.array(maxt) >= abs(tmat.loc["BG", "tps_z"])).sum()) / (1 + NPERM))
put("p_single", (1 + (np.array(single) >= abs(tmat.loc["BG", "tps_z"])).sum()) / (1 + NPERM), 3)

# ---- main effect: targets per season (z within class) ~ baseline + BG
b, se = ols(X, BASE + ["BG"], "tps_z")
coef = b[-1]
bs = []
for _ in range(NBOOT):
    ix = RNG.integers(0, len(X), len(X))
    bb, _ = ols(X.iloc[ix], BASE + ["BG"], "tps_z")
    bs.append(bb[-1])
bs = np.array(bs)
put("coef_z", coef); put("coef_z_lo", np.percentile(bs, 2.5)); put("coef_z_hi", np.percentile(bs, 97.5))
put("tgt_raw", coef * TPS_SD, 1); put("tgt_raw_lo", np.percentile(bs, 2.5) * TPS_SD, 1)
put("tgt_raw_hi", np.percentile(bs, 97.5) * TPS_SD, 1)
# selection-adjusted planning number: empirical-Bayes shrink over the 36 screened t values
shrink = max(0.0, 1 - 1 / float((tmat.to_numpy() ** 2).mean()))
put("eb_shrink", shrink)
put("tgt_plan", coef * TPS_SD * shrink, 1)
# yardstick: targets per season for halving the pick number (e.g. pick 100 -> 50), same model
put("tgt_per_pick_halving", -b[1] * np.log(2) * TPS_SD, 1)
put("plan_share_of_halving", 100 * coef * shrink / (-b[1] * np.log(2)), 0)
# secondary: target rate on 100+ route players
b2, _ = ols(Y, BASE + ["BG"], "tr")
put("coef_tr_pp", 100 * b2[-1], 2)
put("tr_mean", 100 * Y.tr.mean(), 1)

# per-class replication and WR / TE
X["tps_res"] = resid_on(X, BASE, X.tps_z.to_numpy(float))
for cls in (2023, 2024, 2025):
    d = X[X.draft_year == cls]
    put(f"rho_cls_{cls}", spearmanr(d.BG, d.tps_res)[0])
put("n_cls_positive", int(sum(spearmanr(X[X.draft_year == c].BG, X[X.draft_year == c].tps_res)[0] > 0
                             for c in (2023, 2024, 2025))))
for pos in ("WR", "TE"):
    d = X[X.position_group == pos]
    bb, se_ = ols(d, [c for c in BASE if c != "te"] + ["BG"], "tps_z")
    put(f"t_{pos.lower()}", bb[-1] / se_[-1]); put(f"n_{pos.lower()}", len(d))


# ---- out-of-class (leave one Combine class out) R^2
def loco(df, cols, ycol):
    pred = pd.Series(np.nan, index=df.index)
    for c in (2023, 2024, 2025):
        trn, tst = df[df.draft_year != c], df[df.draft_year == c]
        A = np.column_stack([np.ones(len(trn)), trn[cols].to_numpy(float)])
        bb, *_ = np.linalg.lstsq(A, trn[ycol].to_numpy(float), rcond=None)
        pred[tst.index] = np.column_stack([np.ones(len(tst)), tst[cols].to_numpy(float)]) @ bb
    return pred


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


p0, p1 = loco(X, BASE, "tps_z"), loco(X, BASE + ["BG"], "tps_z")
X["tps_fc"] = X.groupby("draft_year").tps.transform("mean") + p0 * X.groupby("draft_year").tps.transform("std")
put("r2_base", r2(X.tps_z, p0)); put("r2_plus", r2(X.tps_z, p1)); put("dr2", r2(X.tps_z, p1) - r2(X.tps_z, p0), 3)
yv, a0_, a1_ = X.tps_z.to_numpy(), p0.to_numpy(), p1.to_numpy()
dl = []
for _ in range(NBOOT):
    ix = RNG.integers(0, len(X), len(X))
    dl.append(r2(yv[ix], a1_[ix]) - r2(yv[ix], a0_[ix]))
dl = np.array(dl)
put("dr2_lo", np.percentile(dl, 2.5), 3); put("dr2_hi", np.percentile(dl, 97.5), 3)
put("dr2_pos_pct", 100 * (dl > 0).mean(), 0)
put("dr2_pts", 100 * (r2(X.tps_z, p1) - r2(X.tps_z, p0)), 1)
for c in (2023, 2024, 2025):
    m_ = (X.draft_year == c).to_numpy()
    put(f"dr2_cls_{c}", r2(yv[m_], a1_[m_]) - r2(yv[m_], a0_[m_]), 3)
BASE2 = BASE + ["three_cone_s", "short_shuttle_s", "ten_yd_split_s"]
q0, q1 = loco(X, BASE2, "tps_z"), loco(X, BASE2 + ["BG"], "tps_z")
put("dr2_base2", r2(X.tps_z, q1) - r2(X.tps_z, q0), 3)

# ---- nested: re-run the whole 36-pair screen inside each training fold, test the winner on the held-out class
nest_rows = []
for c in (2023, 2024, 2025):
    trX = X[X.draft_year != c]
    bestv, bestk = 0, None
    for yc in OUTCOMES:
        df = trX if yc == "tps_z" else trX[trX.index.isin(Y.index)].join(Y[["tr", "yprr", "sep"]])
        ry_ = resid_on(df, BASE, df[yc].to_numpy(float))
        for s_ in SCREEN_X:
            rx_ = resid_on(df, BASE, df[s_].to_numpy(float))
            rr = (rx_ * ry_).sum() / np.sqrt((rx_ ** 2).sum() * (ry_ ** 2).sum())
            tt = abs(rr * np.sqrt(len(df) - len(BASE) - 2) / np.sqrt(1 - rr ** 2))
            if tt > bestv:
                bestv, bestk = tt, (s_, yc)
    xs, yc = bestk
    full = X if yc == "tps_z" else X.join(Y[["tr", "yprr", "sep"]], how="inner")
    trn, tst = full[full.draft_year != c], full[full.draft_year == c]
    A0 = np.column_stack([np.ones(len(trn)), trn[BASE].to_numpy(float)])
    A1 = np.column_stack([A0, trn[xs].to_numpy(float)])
    B0 = np.column_stack([np.ones(len(tst)), tst[BASE].to_numpy(float)])
    B1 = np.column_stack([B0, tst[xs].to_numpy(float)])
    e0 = tst[yc].to_numpy() - B0 @ np.linalg.lstsq(A0, trn[yc].to_numpy(float), rcond=None)[0]
    e1 = tst[yc].to_numpy() - B1 @ np.linalg.lstsq(A1, trn[yc].to_numpy(float), rcond=None)[0]
    nest_rows.append(dict(cls=c, pick=f"{xs}|{yc}", gain=float(1 - (e1 ** 2).sum() / (e0 ** 2).sum())))
nest = pd.DataFrame(nest_rows)
put("nest_n_pick_main", int((nest.pick == "BG|tps_z").sum()))
put("nest_mean_gain_pct", 100 * nest.gain.mean(), 1)


# ---- robustness and placebos
def tcoef(df, cols, yc, x="BG"):
    bb, se_ = ols(df, cols + [x], yc)
    return bb[-1], bb[-1] / se_[-1]


put("t_vin", tcoef(X, BASE + ["v_in_z"], "tps_z")[1])
put("t_drills", tcoef(X, BASE2, "tps_z")[1])
put("t_rawbg", tcoef(X, BASE, "tps_z", "bg_z")[1])
put("t_drafted", tcoef(X[X.draft_overall_pick.notna()], BASE, "tps_z")[1])
put("n_drafted", int(X.draft_overall_pick.notna().sum()))
put("t_drills5", tcoef(X[X.n_drills >= 5], BASE, "tps_z")[1]); put("n_drills5", int((X.n_drills >= 5).sum()))
put("t_slot_tr", tcoef(Y, BASE + ["slot"], "tr")[1])
# influence: drop the single biggest over-performer, and a rank (percentile within class) outcome
top_res = X.tps_res.idxmax()
NUM["top_res_name"] = str(X.loc[top_res, "display_name"])
put("t_drop_top", tcoef(X.drop(index=top_res), BASE, "tps_z")[1])
X["tps_rank"] = X.groupby("draft_year").tps.rank(pct=True)
put("t_rank", tcoef(X, BASE, "tps_rank")[1])
loo_t = [tcoef(X.drop(index=i_), BASE, "tps_z")[1] for i_ in X.index]
put("t_loo_min", min(loo_t))
put("t_ypr", tcoef(Y, BASE, "ypr")[1]); put("t_epat", tcoef(Y, BASE, "epat")[1])
# availability: reaching a 100-route season (logistic, class dummies)
A_ = X.copy()
A_["reached"] = A_.index.isin(Y.index).astype(float)
A_["c24"] = (A_.draft_year == 2024).astype(float)
A_["c25"] = (A_.draft_year == 2025).astype(float)


def logit(df, cols, yc, iters=50):
    Am = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    yy = df[yc].to_numpy(float)
    bb = np.zeros(Am.shape[1])
    for _ in range(iters):
        pp = 1 / (1 + np.exp(-Am @ bb))
        W = pp * (1 - pp)
        H = Am.T @ (Am * W[:, None])
        bb = bb + np.linalg.solve(H, Am.T @ (yy - pp))
    se_ = np.sqrt(np.diag(np.linalg.inv(H)))
    return bb, se_


bl_, sl_ = logit(A_, BASE + ["c24", "c25", "BG"], "reached")
put("reach_or", float(np.exp(bl_[-1]))); put("reach_t", bl_[-1] / sl_[-1])
put("reach_pct_top", 100 * A_[A_.BG > A_.BG.quantile(2 / 3)].reached.mean(), 0)
put("reach_pct_bot", 100 * A_[A_.BG < A_.BG.quantile(1 / 3)].reached.mean(), 0)

# ----------------------------------------------------------------------------- NFL snaps: same metric, same family
gp = pd.read_parquet(P / "game_plays_sample.parquet")
gp = gp[gp.week.between(1, 18)]
gp = gp.merge(players[["nfl_id", "position_group", "draft_year", "display_name"]], on="nfl_id")
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
# separation over expected: context model fitted on 2023-24 seasons
gb["sep"] = gb.separation_at_pass_forward
gb["man"] = gb.team_coverage_type.fillna("").str.contains("MAN")
gb["slot"] = (gb.lined_up_position == "SLOT_WR").astype(float)
gb["inline"] = gb.lined_up_position.isin(["TE", "FB", "RB"]).astype(float)
gb["cush"] = gb.cushion.fillna(gb.cushion.median())
gb["cush_na"] = gb.cushion.isna().astype(float)
gb["pl"] = gb.pass_length.fillna(gb.pass_length.median())
gb["route"] = gb.route_ran.fillna("OTHER")
ROUTES = sorted(gb.route.unique())


def design(df):
    cols = [np.ones(len(df)), df.pl, df.pl ** 2 / 100, df.cush, df.cush_na, df.man.astype(float), df.slot, df.inline,
            df.t_throw, df.target.astype(float), df.down, df.yards_to_go]
    cols += [(df.route == r).astype(float) for r in ROUTES[1:]]
    return np.column_stack(cols)


dev = gb.season < 2025
beta_ctx = np.linalg.lstsq(design(gb[dev]), gb.loc[dev, "sep"].to_numpy(), rcond=None)[0]
gb["soe"] = gb.sep - design(gb) @ beta_ctx
g = gb[gb.route_ran.isin(FAMILIES) & gb.bg.notna()].copy()
put("n_break_snaps", len(g)); put("n_break_players", g.nfl_id.nunique())
g["bg_gz"] = g.groupby(["route_ran", "position_group"]).bg.transform(zs)
g["lead"] = (g.ipf - g.i_apex) * DT
g["graded"] = g.lead >= LEAD_MIN
put("pct_not_graded", 100 * (~g.graded).mean(), 0)
put("bg_game_median", g.bg.median())
q = g[g.graded].copy()
put("n_graded", len(q)); put("n_graded_players", q.nfl_id.nunique())
rs = boot_rho(q.bg_gz, q.soe, cluster=q.nfl_id)
put("snap_rho", rs[0]); put("snap_lo", rs[1]); put("snap_hi", rs[2])
# NFL reliability of Break G (alternate snaps)
g = g.sort_values(["nfl_id", "game_id", "play_id"])
g["half"] = g.groupby("nfl_id").cumcount() % 2
gh = g.groupby(["nfl_id", "half"]).bg_gz.mean().unstack().dropna()
rg = boot_rho(gh[0], gh[1])
put("rel_game", rg[0]); put("rel_game_lo", rg[1]); put("rel_game_hi", rg[2])
# family-matched translation: Combine drill family vs NFL route family
cfam = cpd.groupby(["nfl_id", "family"]).bg_z.mean()
gfam = g.groupby(["nfl_id", "route_ran"]).bg_gz.mean()
gfam.index.names = ["nfl_id", "family"]
jf = pd.concat([cfam.rename("d"), gfam.rename("g")], axis=1).dropna().reset_index()
fr = boot_rho(jf.d, jf.g, cluster=jf.nfl_id)
put("fam_rho", fr[0]); put("fam_lo", fr[1]); put("fam_hi", fr[2]); put("fam_n", fr[3])
gpl = g.groupby("nfl_id").agg(bg_game=("bg_gz", "mean"), n_snaps=("bg", "size"))
J = cplayer.join(gpl, how="inner")
tr_ = boot_rho(J.BG, J.bg_game)
put("tr_rho", tr_[0]); put("tr_lo", tr_[1]); put("tr_hi", tr_[2]); put("tr_n", tr_[3])
psoe = q.groupby("nfl_id").soe.mean()
cs = boot_rho(cplayer.BG.reindex(psoe.index), psoe)
put("comb_soe_rho", cs[0]); put("comb_soe_lo", cs[1]); put("comb_soe_hi", cs[2]); put("comb_soe_n", cs[3])
# timing: is a high-BG receiver thrown to sooner after his break? (graded and targeted snaps)
qt = q[q.target == 1]
lead_p = qt.groupby("nfl_id").lead.mean()
lr = boot_rho(cplayer.BG.reindex(lead_p.index), lead_p)
put("lead_rho", lr[0]); put("lead_lo", lr[1]); put("lead_hi", lr[2]); put("lead_n", lr[3])
# NFL in-game Break G (player mean) vs targets per season, given baseline
Z = X.join(gpl, how="inner")
put("t_gamebg_tps", tcoef(Z, BASE, "tps_z", "bg_game")[1]); put("n_gamebg", len(Z))

# ----------------------------------------------------------------------------- board
board = X[["display_name", "position_group", "draft_year", "draft_overall_pick", "combine_weight", "n_drills",
           "BG", "BG_lo", "BG_hi", "tps", "tps_fc"]].copy()
board["vs_fc"] = board.tps - board.tps_fc


def call(r):
    if r.position_group == "TE":
        return "no call (TE)"
    if r.BG_lo > 0:
        return "plus"
    if r.BG_hi < 0:
        return "minus"
    return "neutral"


board["call"] = board.apply(call, axis=1)
for k in ("plus", "minus", "neutral"):
    d = board[board.call == k]
    put(f"n_{k}", len(d)); put(f"{k}_vs_fc", d.vs_fc.mean(), 1)
    put(f"{k}_beat_pct", 100 * (d.vs_fc > 0).mean(), 0)
    put(f"{k}_vs_fc_med", d.vs_fc.median(), 1)
board.sort_values("BG", ascending=False).to_csv(OUT / "board.csv")
bw = board[board.position_group == "WR"]
tab = pd.concat([bw.sort_values("BG", ascending=False).head(7), bw.sort_values("BG").head(5).iloc[::-1]])
TAB = []
for i, (pid, r_) in enumerate(tab.iterrows()):
    TAB.append(dict(name=r_.display_name, pos=r_.position_group, cls=int(r_.draft_year),
                    pick="UDFA" if pd.isna(r_.draft_overall_pick) else str(int(r_.draft_overall_pick)),
                    drills=int(r_.n_drills), bg=put(f"tab{i}_bg", r_.BG, 1), lo=put(f"tab{i}_lo", r_.BG_lo, 1),
                    hi=put(f"tab{i}_hi", r_.BG_hi, 1), tps=put(f"tab{i}_tps", r_.tps, 1),
                    fc=put(f"tab{i}_fc", r_.tps_fc, 1), call=r_.call))
(OUT / "board_table.json").write_text(json.dumps(TAB, indent=1))
b25 = board[board.draft_year == 2025].sort_values("BG", ascending=False)
put("n_2025_plus", int((b25.call == "plus").sum())); put("n_2025_minus", int((b25.call == "minus").sum()))
NUM["b25_plus_names"] = ", ".join(b25[b25.call == "plus"].display_name)

# ============================================================================= FIGURES
C_D, C_G = OI["blue"], OI["orange"]


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


# ---- Fig 1: drill vs NFL cards for four rule-chosen players (2 highest, 2 lowest BG among those with >= 5 drills
#      and >= 4 NFL in-breaking snaps in the most common drill family "IN")
FAM1 = "IN"
cand = X[(X.n_drills >= 5)].join(gpl, how="inner")
cnt = g[g.route_ran == FAM1].groupby("nfl_id").size()
cand = cand[cand.index.isin(cnt[cnt >= 4].index) & cand.index.isin(cb[cb.family == FAM1].nfl_id)]
hi2 = cand.sort_values("BG", ascending=False).head(2)
lo2 = cand.sort_values("BG").head(2)
picks = list(hi2.index) + list(lo2.index)
cmap = plt.get_cmap("viridis")
vmax = 0.8
fig, axs = plt.subplots(2, 4, figsize=(FIGW, 6.4))
for j, pid in enumerate(picks):
    r_ = X.loc[pid]
    ax = axs[0, j]
    reps = cb[(cb.nfl_id == pid) & (cb.family == FAM1)]
    rep = reps.iloc[0]
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
    tag = "high" if j < 2 else "low"
    ax.set_title(f"{r_.display_name} ({r_.position_group} {int(r_.draft_year)})\nBreak G {r_.BG:+.1f} sd ({tag})",
                 fontsize=10.5, loc="left")
    ax.text(5.3, -6.2, f"Combine {rep.drill_name.split('_')[0].lower()} rep\n{rep.bg:.2f} g", fontsize=9, ha="right")
    ax2 = axs[1, j]
    sn = g[(g.nfl_id == pid) & (g.route_ran == FAM1)]
    for _, s_ in sn.iterrows():
        xx, yy, i0, ipf = gtraces[(s_.game_id, s_.play_id, s_.nfl_id)]
        _, _, gax, gay = kin(xx, yy)
        gam = np.hypot(gax, gay) / G_YD
        rx, ry, lo_, hi_ = rot_path(xx, yy, int(s_.i_apex))
        pts = np.column_stack([rx, ry]).reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=plt.Normalize(0, vmax),
                            lw=1.8, alpha=0.85)
        lc.set_array(gam[lo_:hi_][:-1])
        ax2.add_collection(lc)
    ax2.plot(0, 0, "o", mfc="white", mec="black", ms=6, zorder=4)
    ax2.set_xlim(-2.5, 5.5); ax2.set_ylim(-6.5, 3.5); ax2.set_aspect("equal")
    ax2.set_title(f"His NFL in routes (n={len(sn)})", fontsize=10, loc="left")
    ax2.text(5.3, -6.2, f"median\n{sn.bg.median():.2f} g", fontsize=9, ha="right")
    put(f"fig1_{j}_drill_g", rep.bg); put(f"fig1_{j}_game_g", sn.bg.median()); put(f"fig1_{j}_bg", r_.BG, 1)
    for a in (ax, ax2):
        a.set_xticks([-2, 0, 2, 4]); a.set_yticks([-6, -3, 0, 3])
        a.tick_params(labelsize=8)
    if j == 0:
        ax.set_ylabel("yards upfield (0 = apex)", fontsize=9)
        ax2.set_ylabel("yards upfield (0 = apex)", fontsize=9)
    ax2.set_xlabel("yards toward the break", fontsize=9)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
cax = fig.add_axes([0.93, 0.18, 0.012, 0.62])
fig.colorbar(sm, cax=cax).set_label("acceleration (g)", fontsize=9)
fig.suptitle("Top: one Combine in-breaking rep. Bottom: the same player's NFL in routes. Bright = more force.",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.subplots_adjust(left=0.06, right=0.91, top=0.86, bottom=0.08, wspace=0.25, hspace=0.45)
fig.savefig(FIG / "fig1_cards.png", dpi=150, facecolor="white")
plt.close(fig)
NUM["fig1_names"] = ", ".join(X.loc[p, "display_name"] for p in picks)

# ---- Fig 2: reliability of the nine route-drill components, and their link to the 40
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
ax.set_xlabel(f"split-half rho, 5 drills vs the other 5 (n={NUM['rel_n']}, 95% CI)", fontsize=9.5)
ax.set_title("Does the player repeat himself across drills?", loc="left", fontsize=11)
ax = axs[1]
f40 = [spearmanr(X[m + "_z"], X.forty, nan_policy="omit")[0] for m in order]
ax.barh(range(len(order)), f40, color=[OI["verm"] if m == "bg" else OI["grey"] for m in order], height=0.6)
ax.set_yticks(range(len(order)), [""] * len(order))
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-0.4, 0.4)
ax.set_xlabel("rho with official 40 time (negative = faster)", fontsize=9.5)
ax.set_title("Is it just the 40?", loc="left", fontsize=11)
fig.suptitle("Break G repeats across drills about as well as entry speed, and it is not the 40",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "fig2_reliability.png")

# ---- Fig 3: main result
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 4.9), gridspec_kw={"width_ratios": [1.55, 1]})
ax = axs[0]
cols_cls = {2023: OI["blue"], 2024: OI["orange"], 2025: OI["green"]}
res_t = X.tps - X.tps_fc
for c in (2023, 2024, 2025):
    d = X[X.draft_year == c]
    ax.scatter(d.BG, res_t[d.index], s=24, color=cols_cls[c], alpha=0.8, edgecolor="white", lw=0.5,
               label=f"{c} class (n={len(d)}, rho {NUM['rho_cls_' + str(c)]})")
xx = np.linspace(X.BG.min(), X.BG.max(), 50)
sl = np.polyfit(X.BG, res_t, 1)
band = []
for _ in range(500):
    ix = RNG.integers(0, len(X), len(X))
    band.append(np.polyval(np.polyfit(X.BG.iloc[ix], res_t.iloc[ix], 1), xx))
band = np.percentile(band, [2.5, 97.5], axis=0)
ax.fill_between(xx, band[0], band[1], color=OI["grey"], alpha=0.2, lw=0)
ax.plot(xx, np.polyval(sl, xx), color="black", lw=1.5)
lab = res_t.abs().sort_values(ascending=False).index[:3].tolist() + X.BG.sort_values().index[:1].tolist() + \
    X.BG.sort_values().index[-1:].tolist()
for pid in dict.fromkeys(lab):
    ax.annotate(X.loc[pid, "display_name"], (X.loc[pid, "BG"], res_t[pid]), fontsize=8, xytext=(4, 3),
                textcoords="offset points")
ax.axhline(0, color=OI["grey"], lw=0.7, ls=":")
ax.set_xlabel("Break G, shrunk (sd of true Break G)")
ax.set_ylabel("NFL targets per season minus baseline forecast")
ax.set_title(f"All {NUM['n_all']} receivers: targets beyond the pick-and-40 forecast", loc="left", fontsize=11)
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax = axs[1]
ax.hist(100 * dl, bins=40, color=OI["sky"], edgecolor="white")
ax.axvline(0, color="black", lw=1)
ax.axvline(float(NUM["dr2_pts"]), color=OI["verm"], lw=2)
ax.set_xlabel("out-of-class R² gain from adding Break G (pts)")
ax.set_ylabel("bootstrap resamples")
ax.set_title(f"Held-out gain +{NUM['dr2_pts']} pts, above 0 in {NUM['dr2_pos_pct']}%", loc="left", fontsize=11)
fig.tight_layout()
save(fig, "fig3_targets.png")

# ---- Fig 4: game-side chain
fig, axs = plt.subplots(1, 3, figsize=(FIGW, 4.3))
ax = axs[0]
ax.scatter(jf.d, jf.g, s=10, color=OI["blue"], alpha=0.45)
ax.set_xlabel("Combine Break G, drill family (z)")
ax.set_ylabel("NFL Break G, same route family (z)")
ax.set_title(f"Drill to game, same family\nrho {NUM['fam_rho']} ({NUM['fam_n']} player-family pairs)", loc="left",
             fontsize=10.5)
ax = axs[1]
qb_ = pd.qcut(q.bg_gz, 5, labels=False)
mm = q.groupby(qb_).agg(x=("bg_gz", "mean"), y=("soe", "mean"), s=("soe", "std"), n=("soe", "size"))
ax.errorbar(mm.x, mm.y, yerr=1.96 * mm.s / np.sqrt(mm.n), fmt="o-", color=OI["orange"], capsize=3)
ax.axhline(0, color=OI["grey"], lw=0.7, ls=":")
ax.set_xlabel("NFL Break G on the snap (z, fifths)")
ax.set_ylabel("separation over expected (yd)")
ax.set_title(f"Harder NFL break, more separation?\nrho {NUM['snap_rho']} ({NUM['n_graded']} graded snaps)", loc="left",
             fontsize=10.5)
ax = axs[2]
ax.scatter(Y.BG, Y.sep, s=16, color=OI["green"], alpha=0.7)
ax.set_xlabel("Combine Break G, shrunk (sd)")
ax.set_ylabel("NGS season separation (yd)")
ax.set_title(f"Combine Break G vs season separation\nt = {NUM['t_sep']} given baseline (n={NUM['n_out']})", loc="left",
             fontsize=10.5)
fig.suptitle("The extra targets do not arrive through separation or a copied NFL break", x=0.01, ha="left",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "fig4_chain.png")

NUM["nest_picks"] = "; ".join(f"{r.cls}: {r.pick}" for r in nest.itertuples())
NUM["abs_values"] = sorted({v.lstrip("-") for k, v in NUM.items() if isinstance(v, str) and k != "abs_values"})
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
print(json.dumps({k: v for k, v in NUM.items() if k != "abs_values"}, indent=1))
print(tmat.round(2))
print(nest)
print(pd.DataFrame(TAB).to_string())
print(b25[b25.call.isin(["plus", "minus"])].to_string())
