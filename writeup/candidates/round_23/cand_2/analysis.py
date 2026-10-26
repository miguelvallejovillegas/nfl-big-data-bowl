"""Round 23, candidate 2: receiver break mechanics, Combine route drill vs the same route family on Sundays.

One metric, Break G: the average g-force a receiver puts into the ground over the one second around his first real
cut (braking into the plant, turning, driving out). It is graded by the same code on Combine route drills and on
sampled NFL snaps of the same route family (slant, out, hitch, in, corner), and tested against full-season Next Gen
Stats separation over expected (SOE) on held-out draft classes.

Run from the repo root (about two minutes):
    uv run python writeup/candidates/round_23/cand_2/analysis.py

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
    return dict(v_in=float(s[iin]), v_apex=float(s[apex]), v_out=float(s[iout]),
                decel=float((s[iin] - s[apex]) / max((apex - iin) * DT, DT)),
                reaccel=float((s[iout] - s[apex]) / max((iout - apex) * DT, DT)),
                bg=float(amag[lo:hi].mean() / G_YD), trace=tr)


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


def loco(Dd, cols, bgcol="BG"):
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
    return np.array([r0, r1, r1 - r0, slope])


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
full_boot = TAB["full"]["boot"][:, 3]
put("p_rope", 100 * np.mean(np.abs(full_boot) < MARGIN), 0)
put("p_pos3", 100 * np.mean(full_boot >= MARGIN), 0)

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
        rows.append(dict(game_id=k[0], play_id=k[1], nfl_id=k[2], **r))
g = pd.DataFrame(rows).merge(gp[["game_id", "play_id", "nfl_id", "route_ran", "position_group", "sample_kind"]],
                             on=["game_id", "play_id", "nfl_id"])
g["family"] = g.route_ran
for m in COMP:
    g[m + "_z"] = g.groupby(["family", "position_group"])[m].transform(zs)
g = g.sort_values(["nfl_id", "game_id", "play_id"])
g["half"] = g.groupby("nfl_id").cumcount() % 2
put("n_nfl_breaks", len(g), 0)
put("n_nfl_rec", g.nfl_id.nunique(), 0)
put("share_random", 100 * (g.sample_kind == "random").mean(), 0)

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
put("n_elig", len(E), 0)
top8, bot8 = E.head(8), E.tail(8)
put("top8_soe", IN_YD * np.average(top8.ssoe, weights=top8.routes), 1, plus=True)
put("bot8_soe", IN_YD * np.average(bot8.ssoe, weights=bot8.routes), 1, plus=True)
put("e_rho_soe", spearmanr(E.BG, E.ssoe)[0])


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
for tag, row in (("top", E.iloc[0]), ("mid", mid.iloc[0]), ("bot", E.iloc[-1])):
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

# ============================================================================ figures
# Fig 1 (hero): league speed through the cut, drill vs NFL, by route family
fig, axs = plt.subplots(1, 5, figsize=(FIGW, 4.3), sharey=True, constrained_layout=True)
tt = np.arange(-10, 11) * DT
for ax, f in zip(axs, FAMILIES):
    for src, col, lab in ((cb, DRILL_C, "Combine drill"), (g, NFL_C, "NFL snap")):
        T = np.vstack(src.loc[src.family == f, "trace"].to_numpy())
        med, lo, hi = np.nanmedian(T, 0), np.nanpercentile(T, 25, 0), np.nanpercentile(T, 75, 0)
        thin = np.mean(~np.isnan(T), 0) < 0.95  # hide time points most reps have not reached yet
        med[thin], lo[thin], hi[thin] = np.nan, np.nan, np.nan
        ax.fill_between(tt, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(tt, med, color=col, lw=3, label=lab)
    nd_, nn_ = (cb.family == f).sum(), (g.family == f).sum()
    ax.set_title(f"{FAM_LABEL[f].replace(' / ', '/')}\n", fontsize=12)
    ax.text(0.5, 1.02, f"drill {nd_} cuts, NFL {nn_} breaks", transform=ax.transAxes, fontsize=10,
            color="#333333", ha="center", va="bottom")
    ax.axvline(0, color="#444444", lw=0.8, ls=":")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["-1 s", "plant", "+1 s"])
    ax.set_ylim(0, 9.5)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
axs[0].set_ylabel("Speed (yd/s)")
axs[0].legend(loc="lower right", frameon=False, fontsize=10.5)
fig.suptitle(f"The Combine asks for a faster cut than Sundays do: hitch-family entry "
             f"{ENTRY['HITCH'][0]:.1f} vs {ENTRY['HITCH'][1]:.1f} yd/s", fontsize=13.5, fontweight="bold")
save(fig, "fig1_family_traces.png")

# Fig 2: transfer grid (same family) + receiver-level transfer vs the 40 control
fig = plt.figure(figsize=(FIGW, 6.0), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1])
ax = fig.add_subplot(gs[0])
rowsk = FAMILIES + ["ALL"]
M = np.array([[GRID[(f, m)][0] for m in COMP] for f in rowsk])
norm = TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
im = ax.imshow(M, cmap="PuOr_r", norm=norm, aspect="auto")
for i, f in enumerate(rowsk):
    for j, m in enumerate(COMP):
        rho, lo, hi, n = GRID[(f, m)]
        txt = f"{rho:+.2f}\nn={n}" if f != "ALL" else f"{rho:+.2f}\n({lo:+.2f}, {hi:+.2f})".replace("0.", ".").replace("-.00", ".00")
        ax.text(j, i, txt, ha="center", va="center", fontsize=10.5 if f != "ALL" else 9.5,
                color="white" if abs(rho) > 0.4 else "black", fontweight="bold" if f == "ALL" else None)
ax.set_xticks(range(len(COMP)))
ax.set_xticklabels([COMP_LABEL[m] for m in COMP], fontsize=10.5)
ax.set_yticks(range(len(rowsk)))
ax.set_yticklabels([FAM_LABEL.get(f, "All families\n(90% interval)").replace(" / ", "/") for f in rowsk],
                   fontsize=10.5)
ax.axhline(len(FAMILIES) - 0.5, color="black", lw=1.5)
ax.spines[:].set_visible(False)
ax.set_title("A. Same receiver, same route family:\ndrill vs NFL, rank correlation", loc="left", fontsize=12)
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
cbar.set_label("Rank correlation", fontsize=10.5)
ax2 = fig.add_subplot(gs[1])
items = [("Official 40 →\nNFL top speed", r40, OI["green"]),
         ("Drill entry speed →\nNFL entry speed*", GRID[("ALL", "v_in")], OI["grey"]),
         ("Drill Break G →\nNFL Break G", r_bg, OI["verm"]),
         ("Break G, entry speed\nremoved both sides*", r_res, OI["grey"]),
         ("Break G, random\nsnaps only", r_rand, OI["grey"])]
for i, (lab, r, col) in enumerate(items):
    ax2.errorbar(r[0], i, xerr=[[r[0] - r[1]], [r[2] - r[0]]], fmt="o", color=col, ms=9, lw=2.5, capsize=4)
    ax2.text(r[2] + 0.03, i, f"n={r[3]}", fontsize=10.5, va="center", ha="left")
ax2.axvline(0, color="black", lw=0.8)
ax2.axvline(0.5 * ceil_, color=OI["purple"], lw=1.5, ls="--")
ax2.text(0.5 * ceil_ + 0.02, 1.5, f"a true 0.5 cut link\nwould read {0.5 * ceil_:.2f}\nthrough the noise",
         fontsize=10, color=OI["purple"], va="center")
ax2.set_yticks(range(len(items)))
ax2.set_yticklabels([it[0] for it in items], fontsize=10.5)
ax2.invert_yaxis()
ax2.set_xlim(-0.35, 0.9)
ax2.set_xlabel("Rank correlation, 90% interval\n* receiver-family pairs, clustered", fontsize=11)
ax2.set_title("B. Speed carries;\nthe cut does not", loc="left", fontsize=12)
save(fig, "fig2_transfer.png")

# Fig 3: (A) Break G slope under three baselines vs the margin; (B) measurement-error simulation
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 5.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})
keys = ["pick", "role", "full"]
rowsA = [(BL_LABEL[k], TAB[k]["est"][3], TAB[k]["lo"][3], TAB[k]["hi"][3], OI["verm"],
          f"R² {TAB[k]['est'][0]:.2f} → {TAB[k]['est'][1]:.2f}, n={len(D)}") for k in keys]
rowsA += [("Official 3-cone instead\n(no tracking)", float(NUM["cone_b"]), float(NUM["cone_lo"]),
           float(NUM["cone_hi"]), OI["grey"], f"n={NUM['cone_n']}"),
          ("Official shuttle instead\n(no tracking)", float(NUM["shut_b"]), float(NUM["shut_lo"]),
           float(NUM["shut_hi"]), OI["grey"], f"n={NUM['shut_n']}")]
for i, (lab, e_, lo, hi, col, note) in enumerate(rowsA):
    a1.errorbar(e_, i, xerr=[[e_ - lo], [hi - e_]], fmt="o", color=col, ms=9, lw=2.5, capsize=4)
    a1.text(6.3, i, note, fontsize=10, va="center", ha="left", color="#333333")
a1.axvspan(-MARGIN, MARGIN, color=OI["sky"], alpha=0.18, lw=0)
a1.axvline(0, color="black", lw=0.8)
a1.axhline(2.5, color="#999999", lw=0.8, ls="--")
a1.set_yticks(range(len(rowsA)))
a1.set_yticklabels(["Pick only", "Pick + role\n(TE, slot share)", "Pick + official 40,\nweight, NGS score, TE"]
                   + [r[0] for r in rowsA[3:]], fontsize=10.5)
a1.set_xlim(-7, 13.5)
a1.set_xticks([-6, -3, 0, 3, 6])
a1.set_ylim(len(rowsA) - 0.4, -0.9)
a1.text(0, -0.65, "±3 in margin", ha="center", fontsize=10, color=OI["blue"])
a1.set_xlabel("Effect on career SOE, inches per sd (90%)")
a1.set_title("A. Break G added to three baselines, held-out classes", loc="left", fontsize=12)
bp = a2.boxplot([SIM[b] for b in BETAS], positions=BETAS, widths=0.55, showfliers=False, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor(OI["sky"])
    patch.set_alpha(0.6)
for med in bp["medians"]:
    med.set_color("black")
a2.axhline(obs_b, color=OI["verm"], lw=2)
a2.text(6.4, obs_b - 0.4, f"observed {obs_b:.1f}", color=OI["verm"], fontsize=10.5, ha="right", va="top")
a2.plot(BETAS, BETAS, color="#555555", lw=1, ls=":")
a2.set_xlabel("True effect of cut skill (inches per sd)")
a2.set_ylabel("Slope the test would report")
a2.set_title(f"B. If the truth were +3 in, {NUM['sim_p3']}% of samples\nwould look this bad", loc="left",
             fontsize=12)
save(fig, "fig3_baselines_sim.png")

# Fig 4: player-card grid, 8 hardest and 8 softest drill cutters (eligible WRs), greyed when interval crosses 0
G4 = pd.concat([top8, bot8])
fig, axs = plt.subplots(1, 3, figsize=(FIGW, 6.8), sharey=True, layout="constrained")
panels = [("BG", "BG_sd", 1, "A. Combine Break G (sd)", None),
          ("nbg", None, 1, "B. NFL Break G, same code (sd)", None),
          ("ssoe", "soe_se", IN_YD, "C. Career SOE (inches)", None)]
yy = np.arange(len(G4))
for ax, (col, secol, sc, title, _) in zip(axs, panels):
    v = G4[col].to_numpy(float) * sc
    if col == "nbg":
        se = (G4.nbg_sd / np.sqrt(G4.nb)).to_numpy(float)
    else:
        se = G4[secol].to_numpy(float) * sc
    lo, hi = v - 1.645 * se, v + 1.645 * se
    for i in range(len(G4)):
        crosses = lo[i] < 0 < hi[i]
        base = DRILL_C if i < 8 else NFL_C
        c = "#BBBBBB" if crosses else base
        ax.plot([lo[i], hi[i]], [i, i], color=c, lw=3, solid_capstyle="round")
        ax.plot(v[i], i, "o", color=c if crosses else base, ms=9, mec="white", mew=1)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title, loc="left", fontsize=12)
    ax.grid(axis="x", color="#eeeeee")
    ax.axhline(7.5, color="#666666", lw=0.8, ls="--")
axs[0].set_yticks(yy)
axs[0].set_yticklabels([f"{r.display_name} (#{int(r['rank'])})" for _, r in G4.iterrows()], fontsize=10.5)
axs[0].invert_yaxis()
for i, tl in enumerate(axs[0].get_yticklabels()):
    tl.set_color(DRILL_C if i < 8 else NFL_C)
axs[2].text(0.99, 0.905, f"hardest 8 average {NUM['top8_soe']} in", transform=axs[2].transAxes, ha="right",
            va="bottom", fontsize=10, color=DRILL_C)
axs[2].text(0.99, 0.895, f"softest 8 average {NUM['bot8_soe']} in", transform=axs[2].transAxes, ha="right",
            va="top", fontsize=10, color=NFL_C)
axs[2].set_xlim(-22, 42)
fig.legend(handles=[Line2D([], [], color=DRILL_C, lw=3, marker="o", label="8 hardest drill cutters"),
                    Line2D([], [], color=NFL_C, lw=3, marker="o", label="8 softest drill cutters"),
                    Line2D([], [], color="#BBBBBB", lw=3, marker="o", label="90% interval crosses zero")],
           loc="outside lower center", ncols=3, frameon=False, fontsize=10.5)
save(fig, "fig4_cards.png")

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
