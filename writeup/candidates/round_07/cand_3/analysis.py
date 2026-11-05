"""Film Equivalent: what a Combine movement is worth in NFL snaps of the same movement, and whether the
drill-to-game translation explains production beyond draft slot.

Round 7, candidate 3. Run from the repo root:
    uv run python writeup/candidates/round_07/cand_3/analysis.py

Reads data/processed/*.parquet only (about 1-2 minutes, one core). Writes next to this file:
  numbers.json  - every number quoted in writeup.md (only those; each with the claim it supports)
  board.csv     - the DL translation board
  figures/*.png - every figure (1600 px wide, 150 dpi)
  writeup.md    - rendered from writeup_template.md ({{key}} -> numbers.json, {{TABLE:x}} -> generated table)

Design
  * One function turns a 10 Hz speed trace into a start profile: Savitzky-Golay (5, 2) smoothed speed; Combine
    reps aligned to movement onset (no snap), NFL snaps aligned to the ball snap with the player at rest.
    Launch = peak gain in speed over any 0.1 s inside the first 1.0 s (yd/s^2). Top speed = max smoothed speed.
  * Five pre-specified matched movements (one per group): DL 40 launch <-> pass-rush launch; WR 40 top speed <->
    vertical-route (GO/POST/CORNER) top speed before the ball arrives; TE 40 top speed <-> route top speed;
    OL pass-set drill speed at 0.5 s <-> NFL pass-set speed at 0.5 s; DB back-pedal drill launch <-> coverage
    launch. NFL side uses the random half of game_plays_sample only.
  * Film Equivalent (FE): per-snap noise-to-signal ratio k = sigma^2 / tau^2 (one-way variance components on NFL
    snaps); r_true = r(Combine, NFL mean) / sqrt(reliability of the NFL mean); FE = k * r_true^2 / (1 - r_true^2)
    = how many extra NFL snaps the Combine measurement is worth when estimating the same movement on Sundays.
  * Production test (DL, pressure rate): no-tracking baseline = log pick + weight + EDGE share (+ class);
    partial Spearman, class-stratified bootstrap; leave-one-class-out R^2 change with every fitted piece (variance
    components, Combine->NFL map, shrinkage) refit inside the training classes; learning curve over k rushes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
from scipy.stats import norm, pearsonr, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2027)
NBOOT = 2000
UDFA_PICK = 262
MIN_RUSH = 8
REL_MIN = 0.05  # below this one-snap reliability the NFL variance split is too unstable to price

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2", "verm": "#D55E00",
      "purple": "#CC79A7", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "figure.dpi": 150, "savefig.dpi": 150,
                     "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.labelsize": 11})
FIGW = 1600 / 150
NUM: dict[str, str] = {}
CLAIM: dict[str, str] = {}
TABLES: dict[str, str] = {}


def put(key, val, nd=2, claim=""):
    if key in NUM:
        raise KeyError(f"duplicate key {key}")
    if isinstance(val, str):
        s = val
    elif isinstance(val, (int, np.integer)):
        s = str(int(val))
    else:
        s = f"{float(val):.{nd}f}"
        if s.startswith("-") and float(s) == 0:
            s = s[1:]
    NUM[key] = s
    CLAIM[key] = claim
    return s


def save(fig, name):
    fig.set_size_inches(FIGW, fig.get_figheight())
    fig.savefig(FIG / name, dpi=150, facecolor="white")
    plt.close(fig)


# ============================================================================ data
pl = pd.read_parquet(P / "players.parquet").set_index("nfl_id")
cr = pd.read_parquet(P / "combine_results.parquet").set_index("nfl_id")
ev = pd.read_parquet(P / "combine_events.parquet")
cf = pd.read_parquet(P / "combine_frames.parquet", columns=["event_id", "frame", "entity_type", "s"])
cf = cf[cf.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
gp = pd.read_parquet(P / "game_plays_sample.parquet")
gf = pd.read_parquet(P / "game_frames_sample.parquet",
                     columns=["game_id", "play_id", "nfl_id", "frame", "s", "event"])
pv = pd.read_parquet(P / "player_value.parquet")
al = pd.read_parquet(P / "player_season_alignment.parquet")
ss = pd.read_parquet(P / "player_season_stats.parquet")

TW = 16  # frames kept for traces (0-1.5 s)


def start_feats(sm, o):
    w = sm[o:o + 11]
    if len(w) < 11:
        return None
    return dict(launch=float(np.max(np.diff(w)) * 10), v05=float(w[5]), v10=float(w[10]))


def onset(sm, start=0, thr=0.8):
    hit = np.flatnonzero(sm[start:] >= thr)
    if not len(hit):
        return None
    return start + max(int(hit[0]) - 2, 0)


DRILLS = {"FORTY_YARD_DASH", "PASS_RUSH_DRILL", "PASS_RUSH_DROP", "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION",
          "LINE_DRILL"}
ex = ev.set_index("event_id")
keep = set(ev[ev.drill_name.isin(DRILLS)].event_id)
rows, traces_d = [], {}
for eid, r in cf[cf.event_id.isin(keep)].groupby("event_id", sort=False):
    s = r.s.values
    if len(s) < 15:
        continue
    sm = savgol_filter(s, 5, 2)
    rest = np.flatnonzero(sm < 0.8)
    if not len(rest) or rest[0] > 20:
        continue
    o = onset(sm, rest[0])
    if o is None:
        continue
    f = start_feats(sm, o)
    if f is None:
        continue
    e = ex.loc[eid]
    rows.append(dict(event_id=eid, nfl_id=e.nfl_id, grp=e.position_group, drill=e.drill_name,
                     yr=int(e.draft_year), top=float(sm[o:].max()), **f))
    if e.drill_name == "FORTY_YARD_DASH" and e.position_group == "DL" and o + TW <= len(sm):
        traces_d[eid] = np.clip(sm[o:o + TW], 0, None)
DR = pd.DataFrame(rows)
for c in ["launch", "v05", "v10", "top"]:
    DR[c + "_z"] = DR.groupby(["drill", "yr"])[c].transform(lambda v: (v - v.mean()) / v.std())

g_ = gp[gp.pass_result.notna() & (gp.sample_kind == "random")].merge(
    pl[["position_group"]].reset_index(), on="nfl_id")
meta = g_.set_index(["nfl_id", "game_id", "play_id"])
f_ = gf.merge(g_[["game_id", "play_id", "nfl_id"]], on=["game_id", "play_id", "nfl_id"]).sort_values(
    ["nfl_id", "game_id", "play_id", "frame"])
END = {"pass_arrived", "pass_outcome_incomplete", "pass_outcome_caught", "qb_sack", "qb_strip_sack",
       "pass_outcome_interception", "out_of_bounds", "tackle", "fumble", "run"}
rows, traces_g = [], {}
for k, r in f_.groupby(["nfl_id", "game_id", "play_id"], sort=False):
    m = meta.loc[k]
    if not np.isfinite(m.snap_frame):
        continue
    sf = int(m.snap_frame)
    s = r.s.values
    if len(s) < sf + 12:
        continue
    sm = savgol_filter(s, 5, 2)
    evs = r.event.values
    endi = [i for i in range(sf + 1, len(evs)) if evs[i] in END]
    e_ = min(endi[0] if endi else len(sm), sf + 50)
    at_rest = bool(sm[sf] < 1.0)
    d = dict(nfl_id=k[0], game_id=k[1], play_id=k[2], grp=m.position_group, align=m.lined_up_position,
             route=m.route_ran, season=int(m.season), ngs=m.player_get_off, rest=at_rest,
             press=bool(pd.notna(m.time_to_pressure)), top=float(np.max(sm[sf:max(e_, sf + 2)])))
    if at_rest:
        d.update({"s_" + a: b for a, b in start_feats(sm, sf).items()})
        oo = onset(sm[:e_], sf)
        if oo is not None and oo + 11 <= len(sm):
            d.update({"o_" + a: b for a, b in start_feats(sm, oo).items()})
        if m.position_group == "DL" and sf + TW <= len(sm):
            traces_g[k] = np.clip(sm[sf:sf + TW], 0, None)
    rows.append(d)
GA = pd.DataFrame(rows)

# roles
sh = al.pivot_table(index="nfl_id", columns="lined_up_position", values="plays", aggfunc="sum").fillna(0)
sh = sh.div(sh.sum(axis=1), axis=0)
EDGE = sh["EDGE"].rename("edge")


# ============================================================================ helpers
def resid(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ b


def prho(y, x, Z):
    return spearmanr(resid(y, Z), resid(x, Z))[0]


def strat_idx(strata):
    return np.concatenate([RNG.choice(np.flatnonzero(strata == s), (strata == s).sum()) for s in np.unique(strata)])


def boot(fn, n, nb=NBOOT, strata=None):
    out = []
    for _ in range(nb):
        idx = RNG.integers(0, n, n) if strata is None else strat_idx(strata)
        try:
            out.append(fn(idx))
        except Exception:
            pass
    return np.array(out, float)


def ci(a, lo=2.5, hi=97.5):
    return np.nanpercentile(a, lo), np.nanpercentile(a, hi)


def snap_stats(G, col, adj):
    """adjust per snap for alignment/route, return per-player n, mean, var"""
    G = G.dropna(subset=[col]).copy()
    G[col] = G[col] - G.groupby(adj)[col].transform("mean") + G[col].mean()
    a = G.groupby("nfl_id")[col].agg(n="size", m="mean", v="var")
    return a, G


def varcomp(a):
    """one-way random-effects variance components from per-player n, mean, var"""
    a2 = a[a.n >= 2]
    s2 = np.average(a2.v, weights=a2.n - 1)
    t2 = a2.m.var() - s2 * (1 / a2.n).mean()
    return s2, max(t2, 1e-6 * s2)


def fe_from(r, s2, t2, nbar):
    rel = t2 / (t2 + s2 / nbar)
    rt = min(abs(r) / np.sqrt(rel), 0.99) * np.sign(r)
    k = s2 / t2
    return k * rt ** 2 / (1 - rt ** 2), rt, k, rel


def outcome(grp):
    tgt = "opportunity" if grp == "TE" else "production"
    v = pv[(pv.position_group == grp) & (pv.target == tgt)].set_index("nfl_id").actual
    return -v if grp == "OL" else v


VERT = {"GO", "POST", "CORNER"}
CELLS = {
    "DL": dict(drill=["FORTY_YARD_DASH"], dcol="launch", gcol="s_launch", adj="align", minn=MIN_RUSH,
               sel=lambda g: g.rest.values, label="40 launch vs pass-rush launch", unit="rushes",
               out="pressure rate"),
    "WR": dict(drill=["FORTY_YARD_DASH"], dcol="top", gcol="top", adj="route", minn=3,
               sel=lambda g: g.route.isin(VERT).values, label="40 top speed vs go/post/corner top speed",
               unit="vertical routes", out="yards per route"),
    "TE": dict(drill=["FORTY_YARD_DASH"], dcol="top", gcol="top", adj="route", minn=5,
               sel=lambda g: g.route.notna().values, label="40 top speed vs route top speed", unit="routes",
               out="snaps per season"),
    "OL": dict(drill=["PASS_RUSH_DROP"], dcol="v05", gcol="o_v05", adj="align", minn=MIN_RUSH,
               sel=lambda g: g.rest.values, label="pass-set drill vs NFL pass set (speed at 0.5 s)",
               unit="pass sets", out="pressures avoided"),
    "DB": dict(drill=["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "LINE_DRILL"], dcol="launch",
               gcol="o_launch", adj="align", minn=5, sel=lambda g: g.rest.values,
               label="back-pedal drill launch vs coverage launch", unit="coverage snaps", out="solo tackles/100"),
}
GROUPS = ["DL", "WR", "OL", "DB", "TE"]


def cell_data(grp):
    c = CELLS[grp]
    G = GA[GA.grp == grp]
    G = G[c["sel"](G)]
    a, Gadj = snap_stats(G, c["gcol"], c["adj"])
    d = DR[(DR.grp == grp) & DR.drill.isin(c["drill"])].groupby("nfl_id")[c["dcol"] + "_z"].mean().rename("c")
    j = a[a.n >= c["minn"]].join(d, how="inner")
    return a, j, Gadj


# ============================================================================ 1. five-group screen
SCR = []
for grp in GROUPS:
    a, j, _ = cell_data(grp)
    s2, t2 = varcomp(a)
    r = pearsonr(j.c, j.m)[0]
    fe, rt, k, rel = fe_from(r, s2, t2, j.n.mean())
    cvals = j.c.values

    def fe_b(ix):
        # resample matched players for r, and all NFL players for the variance components
        sub = a.iloc[RNG.integers(0, len(a), len(a))]
        s2b, t2b = varcomp(sub)
        rb = pearsonr(cvals[ix], j.m.values[ix])[0]
        return fe_from(rb, s2b, t2b, j.n.values[ix].mean())[0]
    fb = boot(fe_b, len(j), nb=1000)
    rb = boot(lambda ix: pearsonr(cvals[ix], j.m.values[ix])[0], len(j), nb=1000)
    # production link of the NFL movement (beyond pick, weight, class)
    y = outcome(grp).rename("y")
    q = j.join(y, how="inner").join(pl[["draft_overall_pick", "combine_weight", "draft_year"]])
    q["lp"] = np.log(q.draft_overall_pick.fillna(UDFA_PICK))
    q = q.dropna(subset=["combine_weight"])
    Zq = np.column_stack([q.lp, q.combine_weight, q.draft_year == 2023, q.draft_year == 2024]).astype(float)
    pr = prho(q.y.values, q.m.values, Zq)
    pb = boot(lambda ix: prho(q.y.values[ix], q.m.values[ix], Zq[ix]), len(q), nb=1000, strata=q.draft_year.values)
    SCR.append(dict(grp=grp, n=len(j), r=r, r_lo=ci(rb)[0], r_hi=ci(rb)[1], rel1=t2 / (t2 + s2), k=k, rt=rt,
                    fe=fe, fe_lo=np.nanpercentile(fb, 10), fe_hi=np.nanpercentile(fb, 90),
                    nsnap=int(j.n.median()), n_y=len(q), py=pr, py_lo=ci(pb)[0], py_hi=ci(pb)[1]))
SC = pd.DataFrame(SCR).set_index("grp")
print(SC.round(3).to_string())
for grp in GROUPS:
    s = SC.loc[grp]
    put(f"sc_{grp}_r", s.r, claim=f"{grp} Combine-to-NFL correlation of matched movement")
    put(f"sc_{grp}_fe", s.fe, 1, claim=f"{grp} film equivalent")
    put(f"sc_{grp}_n", int(s.n))
    put(f"sc_{grp}_py", s.py, claim=f"{grp} NFL movement partial rho with production")
put("rel_min", REL_MIN, claim="pricing threshold for one-snap reliability")
put("rel_noisy_max", SC.loc[["OL", "DB", "TE"], "rel1"].max(), claim="highest one-snap reliability among unpriced groups")
put("sc_WR_rlo", SC.loc["WR", "r_lo"])
put("sc_WR_rhi", SC.loc["WR", "r_hi"])
put("sc_WR_fe_lo", SC.loc["WR", "fe_lo"], 1)
put("sc_WR_fe_hi", SC.loc["WR", "fe_hi"], 1)
put("sc_WR_pylo", SC.loc["WR", "py_lo"])
put("sc_WR_pyhi", SC.loc["WR", "py_hi"])
put("sc_WR_nsnap", int(SC.loc["WR", "nsnap"]))


# ============================================================================ 2. DL deep dive
c = CELLS["DL"]
aDL, jDL, GDL = cell_data("DL")
s2, t2 = varcomp(aDL)
tau = np.sqrt(t2)
put("dl_n_rushes", f"{int(jDL.n.sum()):,}", claim="random at-rest NFL rushes for DL with a tracked 40")
put("dl_n", len(jDL))
put("dl_rush_med", int(jDL.n.median()))
put("dl_rel1", t2 / (t2 + s2), claim="one-rush reliability of NFL launch")
put("dl_k", s2 / t2, 1, claim="noise-to-signal ratio, rushes")
rel20 = t2 / (t2 + s2 / jDL.n.mean())
put("dl_rel20", rel20)
# NFL launch validity vs official NGS get-off (play level)
pp = GDL[GDL.nfl_id.isin(jDL.index)].dropna(subset=["ngs", "s_launch"])
put("val_play_rho", spearmanr(pp.s_launch, pp.ngs)[0], claim="per-rush launch vs NGS get-off")
put("val_play_n", f"{len(pp):,}")
# medians in yd/s^2
DL40 = DR[(DR.grp == "DL") & (DR.drill == "FORTY_YARD_DASH")]
put("dl_c_med", DL40.launch.median(), 1)
put("dl_g_med", GDL.s_launch.median(), 1)
put("dl_tau", tau, 2, claim="sd of true NFL launch across players")
# translation r, true-score r, FE with bootstrap
r40 = pearsonr(jDL.c, jDL.m)[0]
fe40, rt40, k40, _ = fe_from(r40, s2, t2, jDL.n.mean())
put("dl_r", r40)
put("dl_rs", spearmanr(jDL.c, jDL.m)[0])
put("dl_rt", rt40)
put("dl_r2", min(rt40 ** 2, 0.95), claim="share of true NFL launch variance the 40 explains")
put("dl_fe", fe40, 1, claim="DL film equivalent of the 40 launch")
for key in ("r_lo", "r_hi", "fe_lo", "fe_hi"):
    put(f"dl_{key}", SC.loc["DL", key], 1 if "fe" in key else 2)
# rep-to-rep reliability of 40 launch
dd = DL40.sort_values(["nfl_id", "event_id"]).copy()
dd["kk"] = dd.groupby("nfl_id").cumcount()
w2 = dd[dd.kk < 2].pivot(index="nfl_id", columns="kk", values="launch_z").dropna()
put("dl_c_rep", pearsonr(w2[0], w2[1])[0], claim="40 launch rep-to-rep")
put("dl_c_rep_n", len(w2))

# baselines without tracking and the other tracked drill: their r with NFL launch mean and FE
ALT = {}
for lab, ser, sign in [("Official 40 time", cr.forty, -1), ("Official 10-yd split", cr.ten_yd_split, -1)]:
    q = jDL.join(ser.rename("x"), how="inner").dropna(subset=["x"])
    rr = sign * pearsonr(q.x, q.m)[0]
    ALT[lab] = (rr, len(q), fe_from(rr, s2, t2, q.n.mean())[0])
prd = DR[(DR.grp == "DL") & (DR.drill == "PASS_RUSH_DRILL")].groupby("nfl_id").launch_z.mean().rename("x")
q = aDL[aDL.n >= MIN_RUSH].join(prd, how="inner")
rr = pearsonr(q.x, q.m)[0]
ALT["Pass-rush drill launch (tracked)"] = (rr, len(q), fe_from(rr, s2, t2, q.n.mean())[0])
put("alt_40_r", ALT["Official 40 time"][0])
put("alt_40_fe", ALT["Official 40 time"][2], 1)
put("alt_40_n", ALT["Official 40 time"][1])
put("alt_10_r", ALT["Official 10-yd split"][0])
put("alt_10_fe", ALT["Official 10-yd split"][2], 1)
put("alt_10_n", ALT["Official 10-yd split"][1])
put("alt_prd_r", ALT["Pass-rush drill launch (tracked)"][0])
put("alt_prd_fe", ALT["Pass-rush drill launch (tracked)"][2], 1)
put("alt_prd_n", ALT["Pass-rush drill launch (tracked)"][1])
# tracked 40 launch beyond official 40 time (does tracking add to the stopwatch?)
q = jDL.join(cr[["forty"]]).join(pl[["combine_weight"]]).dropna()
put("dl_beyond_forty", prho(q.m.values, q.c.values, np.column_stack([q.forty, q.combine_weight])),
    claim="tracked 40 launch vs NFL launch beyond official 40 and weight")
put("dl_beyond_forty_n", len(q))

# ---------------------------------------------------------------- production cohort
y = outcome("DL").rename("y")
D = jDL.join(y, how="inner").join(pl[["display_name", "draft_overall_pick", "combine_weight", "draft_year"]])
D = D.join(EDGE).join(cr[["forty", "ten_yd_split"]])
D["lp"] = np.log(D.draft_overall_pick.fillna(UDFA_PICK))
D = D.dropna(subset=["combine_weight", "edge"])
D["role"] = np.where(D.edge >= 0.5, "EDGE", "interior")
n = len(D)
put("pr_n", n, claim="DL with tracked 40, 8+ random rushes and a pressure rate")
put("pr_med", 100 * D.y.median(), 1)


def posterior(Dt, Dp, s2_, t2_):
    """Combine-informed shrinkage. Fit the Combine->NFL map and R^2 on Dt, apply to Dp.
    returns (nfl_pop: shrunk to population mean, prior mean m, posterior, gap, posterior sd)"""
    b = np.polyfit(Dt.c, Dt.m, 1)
    rr = pearsonr(Dt.c, Dt.m)[0]
    rel = t2_ / (t2_ + s2_ / Dt.n.mean())
    R2 = min(rr ** 2 / rel, 0.95)
    mu = Dt.m.mean()
    lam = t2_ / (t2_ + s2_ / Dp.n)
    pop = mu + lam * (Dp.m - mu)
    m0 = np.polyval(b, Dp.c)
    v0 = t2_ * (1 - R2)
    w = v0 / (v0 + s2_ / Dp.n)
    post = m0 + w * (Dp.m - m0)
    return pop.values, m0, post.values, (post - m0).values, np.sqrt(w * s2_ / Dp.n).values


pop, m0, post, gap, psd = posterior(jDL, D, s2, t2)
D["nfl"] = pop
D["prior"] = m0
D["post"] = post
D["gap"] = gap
D["gap_sd"] = gap / tau
D["psd_sd"] = psd / tau
Z = np.column_stack([D.lp, D.combine_weight, D.edge, D.draft_year == 2023, D.draft_year == 2024]).astype(float)
yy = D.y.values
strata = D.draft_year.values
PROD = {}
for lab, col in [("Combine 40 launch (draft-time)", "c"), ("NFL launch, 20 rushes", "nfl"),
                 ("Translation gap", "gap"), ("Official 40 time", "forty"), ("Official 10-yd split", "ten_yd_split")]:
    dq = D.dropna(subset=[col])
    Zx = np.column_stack([dq.lp, dq.combine_weight, dq.edge, dq.draft_year == 2023, dq.draft_year == 2024]).astype(float)
    sign = -1 if col in ("forty", "ten_yd_split") else 1
    x = sign * dq[col].values
    r0 = prho(dq.y.values, x, Zx)
    b_ = boot(lambda ix: prho(dq.y.values[ix], x[ix], Zx[ix]), len(dq), strata=dq.draft_year.values)
    PROD[lab] = dict(rho=r0, lo=ci(b_)[0], hi=ci(b_)[1], n=len(dq), ppos=float((b_ > 0).mean()))
PR = pd.DataFrame(PROD).T
print(PR)
def build_screen_table():
    tb = ["| Group | Combine vs NFL movement | n | r [95%] | FE [80%] | NFL movement vs production |",
          "|---|---|---|---|---|---|"]
    for grp in GROUPS:
        s = SC.loc[grp]
        c = CELLS[grp]
        feh = f"{s.fe_hi:.1f}" if s.fe_hi < 99 else "99+"
        fes = (f"{s.fe:.1f} {c['unit']} [{s.fe_lo:.1f}, {feh}]" if s.rel1 >= REL_MIN else
               "too noisy")
        tb.append(f"| {grp} | {c['label']} | {int(s.n)} | {s.r:.2f} [{s.r_lo:.2f}, {s.r_hi:.2f}] | "
                  f"{fes} | {s.py:.2f} [{s.py_lo:.2f}, {s.py_hi:.2f}] "
                  f"({c['out']}) |")
    TABLES["screen"] = "\n".join(tb).replace("-0.", "−0.").replace("[-", "[−")


# DL row of the screen = the headline test (same players, same baseline incl. EDGE share, same bootstrap)
SC.loc["DL", ["py", "py_lo", "py_hi"]] = PR.loc["NFL launch, 20 rushes", ["rho", "lo", "hi"]].values.astype(float)
build_screen_table()
put("p_c", PR.loc["Combine 40 launch (draft-time)", "rho"], claim="Combine launch vs pressure beyond baseline")
put("p_c_lo", PR.loc["Combine 40 launch (draft-time)", "lo"])
put("p_c_hi", PR.loc["Combine 40 launch (draft-time)", "hi"])
put("p_g", PR.loc["NFL launch, 20 rushes", "rho"], claim="NFL launch vs pressure beyond baseline")
put("p_g_lo", PR.loc["NFL launch, 20 rushes", "lo"])
put("p_g_hi", PR.loc["NFL launch, 20 rushes", "hi"])
put("p_g_ppos", int(round(100 * PR.loc["NFL launch, 20 rushes", "ppos"])))
put("p_gap", PR.loc["Translation gap", "rho"], claim="translation gap vs pressure beyond baseline")
put("p_gap_lo", PR.loc["Translation gap", "lo"])
put("p_gap_hi", PR.loc["Translation gap", "hi"])
put("p_40", PR.loc["Official 40 time", "rho"])
put("p_10", PR.loc["Official 10-yd split", "rho"])
put("p_10_n", int(PR.loc["Official 10-yd split", "n"]))
put("p_gap_given_nfl", prho(yy, D.gap.values, np.column_stack([Z, D.nfl])), claim="gap beyond NFL launch")
put("p_c_given_nfl", prho(yy, D.c.values, np.column_stack([Z, D.nfl])), claim="Combine launch beyond NFL launch")

# mediation arithmetic: what the draft-time signal can be at most
implied = rt40 * PR.loc["NFL launch, 20 rushes", "rho"] / np.sqrt(rel20)
put("implied", implied, claim="implied Combine->pressure partial rho via translation")
n80 = ((norm.ppf(0.975) + norm.ppf(0.8)) / np.arctanh(implied)) ** 2 + 3
put("implied_n80", int(round(n80, -1)), claim="players needed for 80% power")
put("pr_rt_true", rt40)

# ---------------------------------------------------------------- out of class, everything refit in the fold
def loco(Dx, feats, ix=None):
    Dd = Dx if ix is None else Dx.iloc[ix]
    res = {f: [] for f in ["base", *feats]}
    yys, per = [], {}
    for yr in (2023, 2024, 2025):
        tr, te = Dd[Dd.draft_year != yr], Dd[Dd.draft_year == yr]
        if len(te) < 5:
            continue
        # variance components and Combine map from training-class players only (all their rushes)
        a_tr = aDL.loc[aDL.index.isin(jDL.index[pl.loc[jDL.index, "draft_year"].values != yr])]
        s2f, t2f = varcomp(a_tr)
        jtr = jDL.loc[jDL.index.isin(a_tr.index)]
        out = {}
        for part, Dp in (("tr", tr), ("te", te)):
            p_, m_, po_, g_, _ = posterior(jtr, Dp, s2f, t2f)
            out[part] = {"nfl": p_, "gap": g_, "c": Dp.c.values}
        Ztr = np.column_stack([np.ones(len(tr)), tr.lp, tr.combine_weight, tr.edge])
        Zte = np.column_stack([np.ones(len(te)), te.lp, te.combine_weight, te.edge])
        yt = te.y.values - te.y.mean()
        yys.append(yt)
        b0, *_ = np.linalg.lstsq(Ztr, tr.y.values, rcond=None)
        p0 = Zte @ b0
        res["base"].append(p0 - p0.mean())
        per[yr] = {"n": len(te), "base": np.mean((yt - (p0 - p0.mean())) ** 2)}
        for f in feats:
            b1, *_ = np.linalg.lstsq(np.column_stack([Ztr, out["tr"][f]]), tr.y.values, rcond=None)
            p1 = np.column_stack([Zte, out["te"][f]]) @ b1
            res[f].append(p1 - p1.mean())
            per[yr][f] = np.mean((yt - (p1 - p1.mean())) ** 2)
    Y = np.concatenate(yys)
    sst = (Y ** 2).sum()
    r2 = {f: 1 - ((Y - np.concatenate(v)) ** 2).sum() / sst for f, v in res.items()}
    return r2, per


FEATS = ["c", "nfl", "gap"]
r2, per = loco(D, FEATS)
LB = {f: [] for f in FEATS}
for _ in range(1000):
    ix = strat_idx(strata)
    try:
        r2b, _ = loco(D, FEATS, ix)
    except Exception:
        continue
    for f in FEATS:
        LB[f].append(r2b[f] - r2b["base"])
put("oc_base", r2["base"], 3, claim="baseline held-out R2")
for f in FEATS:
    lo, hi = ci(np.array(LB[f]))
    put(f"oc_{f}", r2[f] - r2["base"], 3, claim=f"held-out R2 change, {f}")
    put(f"oc_{f}_lo", lo, 3)
    put(f"oc_{f}_hi", hi, 3)
    put(f"oc_{f}_ppos", int(round(100 * np.mean(np.array(LB[f]) > 0))))
    put(f"oc_{f}_better", int(sum(per[yr][f] < per[yr]["base"] for yr in per)))
for yr in per:
    put(f"oc_n_{yr}", per[yr]["n"])
    for f in ("nfl", "gap"):
        put(f"oc_{f}_{yr}", 1 - per[yr][f] / per[yr]["base"], 3, claim=f"held-out MSE reduction share {f} {yr}")

# ---------------------------------------------------------------- planning number
rush_med = ss[(ss.position_group == "DL") & (ss.pass_rush_snaps > 0) & ss.nfl_id.isin(D.index)].pass_rush_snaps.median()
put("rush_season", int(round(rush_med)))
zn = (D.nfl - D.nfl.mean()) / D.nfl.std()


def slope(ix):
    X = np.column_stack([np.ones(len(ix)), zn.values[ix], Z[ix]])
    b, *_ = np.linalg.lstsq(X, yy[ix], rcond=None)
    return b


bfull = slope(np.arange(n))
bs = np.array([slope(strat_idx(strata)) for _ in range(NBOOT)])
put("eff_pp", 100 * bfull[1], 1, claim="pressure-rate points per sd of NFL launch")
put("eff_pp_lo", 100 * np.percentile(bs[:, 1], 5), 1)
put("eff_pp_hi", 100 * np.percentile(bs[:, 1], 95), 1)
put("eff_press", bfull[1] * rush_med, 1, claim="pressures per season per sd NFL launch")
put("eff_press_lo", np.percentile(bs[:, 1], 5) * rush_med, 1)
put("eff_press_hi", np.percentile(bs[:, 1], 95) * rush_med, 1)
put("pick_80_100", 100 * bfull[2] * (np.log(80) - np.log(100)), 1, claim="pressure-rate pts from pick 100 to 80")
put("pick_80_100_press", bfull[2] * (np.log(80) - np.log(100)) * rush_med, 1)
put("eff_ratio", bfull[1] / (bfull[2] * (np.log(80) - np.log(100))), 0, claim="launch sd effect / pick 100->80 effect")
put("k_prime", (s2 / t2) / (1 - min(rt40 ** 2, 0.95)), 0, claim="rushes at which film outweighs the 40 prior")
put("nfl_sd_raw", D.nfl.std(), 2, claim="sd of shrunk NFL launch across cohort, yd/s2")

# ---------------------------------------------------------------- learning curve: k rushes, film vs Combine+film
KS = [1, 2, 3, 5, 8, 12]
LC_MIN = 12
Dlc = D[D.n >= LC_MIN].copy()
put("lc_n", len(Dlc), claim="players in the learning curve (12+ rushes)")
Zl = np.column_stack([Dlc.lp, Dlc.combine_weight, Dlc.edge, Dlc.draft_year == 2023,
                      Dlc.draft_year == 2024]).astype(float)
rush_by = {nid: g.s_launch.values for nid, g in GDL.groupby("nfl_id")}
# out-of-class Combine prior for every player: map and variance components fit on the other two classes
PRI = {}
for yr in (2023, 2024, 2025):
    tr_ids = jDL.index[pl.loc[jDL.index, "draft_year"].values != yr]
    s2f, t2f = varcomp(aDL.loc[aDL.index.isin(tr_ids)])
    jtr = jDL.loc[tr_ids]
    b = np.polyfit(jtr.c, jtr.m, 1)
    rr = pearsonr(jtr.c, jtr.m)[0]
    R2 = min(rr ** 2 / (t2f / (t2f + s2f / jtr.n.mean())), 0.95)
    PRI[yr] = (b, s2f, t2f, R2)
LC = {"film": {}, "blend": {}}
REPS = 300
for kk in KS:
    vf, vb = [], []
    for _ in range(REPS):
        xf, xb = [], []
        for nid, r in Dlc.iterrows():
            b, s2f, t2f, R2 = PRI[r.draft_year]
            xs = RNG.choice(rush_by[nid], kk, replace=False).mean()
            m0_ = np.polyval(b, r.c)
            v0 = t2f * (1 - R2)
            w = v0 / (v0 + s2f / kk)
            xf.append(xs)
            xb.append(m0_ + w * (xs - m0_))
        vf.append(prho(Dlc.y.values, np.array(xf), Zl))
        vb.append(prho(Dlc.y.values, np.array(xb), Zl))
    LC["film"][kk] = (np.mean(vf), np.percentile(vf, 10), np.percentile(vf, 90))
    LC["blend"][kk] = (np.mean(vb), np.percentile(vb, 10), np.percentile(vb, 90))
c_only = np.array([np.polyval(PRI[r.draft_year][0], r.c) for _, r in Dlc.iterrows()])
LC0 = prho(Dlc.y.values, c_only, Zl)
LCall = prho(Dlc.y.values, Dlc.m.values, Zl)
print("LC", LC, LC0, LCall)
put("lc_c0", LC0, claim="Combine-only prior vs pressure (k=0)")
put("lc_all", LCall)
for kk in (2, 5, 12):
    put(f"lc_film_{kk}", LC["film"][kk][0])
    put(f"lc_blend_{kk}", LC["blend"][kk][0])

# ---------------------------------------------------------------- EDGE vs interior
X0 = np.column_stack([np.ones(n), Z])
b0, *_ = np.linalg.lstsq(X0, yy, rcond=None)
D["fc"] = X0 @ b0
ROLE_ROWS = []
for role in ("EDGE", "interior"):
    q = D[D.role == role].copy()
    Zr = np.column_stack([q.lp, q.combine_weight, q.draft_year == 2023, q.draft_year == 2024]).astype(float)
    t3 = pd.qcut(q.nfl.rank(method="first"), 3, labels=["low", "mid", "high"])
    rr_c = pearsonr(q.c, q.m)[0]
    pr_ = prho(q.y.values, q.nfl.values, Zr)
    b_ = boot(lambda ix: prho(q.y.values[ix], q.nfl.values[ix], Zr[ix]), len(q), nb=1000,
              strata=q.draft_year.values)
    hi_, lo_ = q[t3 == "high"], q[t3 == "low"]
    row = dict(role=role, n=len(q), r_c=rr_c, pr=pr_, pr_lo=ci(b_)[0], pr_hi=ci(b_)[1],
               press_hi=100 * hi_.y.mean(), press_lo=100 * lo_.y.mean(),
               beat_hi=int(round(100 * (hi_.y > hi_.fc).mean())), beat_lo=int(round(100 * (lo_.y > lo_.fc).mean())),
               cut_hi=q.nfl.quantile(2 / 3), cut_lo=q.nfl.quantile(1 / 3), med_press=100 * q.y.median())
    ROLE_ROWS.append(row)
    tag = "e" if role == "EDGE" else "i"
    put(f"role_{tag}_n", len(q))
    put(f"role_{tag}_pr", pr_)
    put(f"role_{tag}_rc", rr_c)
    put(f"role_{tag}_hi", row["press_hi"], 1)
    put(f"role_{tag}_lo", row["press_lo"], 1)
RR = pd.DataFrame(ROLE_ROWS).set_index("role")
tb = ["| Role | n | 40 vs NFL launch, r | NFL launch vs pressure (95%) | "
      "Pressure rate, top vs bottom third of NFL launch | Beat forecast, top vs bottom third |",
      "|---|---|---|---|---|---|"]
for role, r in RR.iterrows():
    tb.append(f"| {role} | {int(r.n)} | {r.r_c:.2f} | {r.pr:.2f} [{r.pr_lo:.2f}, {r.pr_hi:.2f}] | "
              f"{r.press_hi:.1f}% vs {r.press_lo:.1f}% | {int(r.beat_hi)}% vs {int(r.beat_lo)}% |")
TABLES["roles"] = "\n".join(tb).replace("-0.", "−0.").replace("[-", "[−")

# ---------------------------------------------------------------- named board
D["pick_s"] = D.draft_overall_pick.apply(lambda v: "UDFA" if pd.isna(v) else str(int(v)))
c_raw = DL40.groupby("nfl_id").launch.mean()
D["c_raw"] = c_raw.reindex(D.index)
D["g_raw"] = D.m
D["ci90"] = 1.645 * D.psd_sd
D["sure"] = (D.gap_sd.abs() > D.ci90)
board = D.sort_values("gap_sd", ascending=False)
board[["display_name", "draft_year", "pick_s", "role", "n", "c_raw", "g_raw", "c", "prior", "post", "gap_sd",
       "psd_sd", "y", "fc"]].to_csv(OUT / "board.csv")
put("gap_sd_spread", D.gap_sd.std(), 2, claim="sd of gap across players, in sd of true launch")
put("gap_psd_med", D.psd_sd.median(), 2, claim="typical posterior sd of a player's gap")
put("n_sure", int(D.sure.sum()), claim="players whose 90% interval for gap excludes zero")
top, bot = board.head(4), board.tail(4)
rows = ["| Rusher (class, pick, role) | 40 launch | NFL launch (rushes) | Gap, sd (90%) | "
        "Pressure rate vs forecast |", "|---|---|---|---|---|"]
for nid, r in pd.concat([top, bot]).iterrows():
    gs = f"{r.gap_sd:+.2f} ± {r.ci90:.2f}".replace("-", "−")
    rows.append(f"| {r.display_name} ({r.draft_year}, {r.pick_s}, {r.role}) | {r.c_raw:.1f} | "
                f"{r.g_raw:.1f} ({int(r.n)}) | {gs} | {100 * r.y:.1f}% vs {100 * r.fc:.1f}% |")
TABLES["board"] = "\n".join(rows)
top_hi = pd.concat([top, bot])
for nid, r in top_hi.iterrows():
    tag = r.display_name.split()[-1].lower().replace(".", "").replace("'", "")
    put(f"pl_{tag}_c", r.c_raw, 1)
    put(f"pl_{tag}_g", r.g_raw, 1)
    put(f"pl_{tag}_gap", r.gap_sd, 2)
    put(f"pl_{tag}_y", 100 * r.y, 1)
    put(f"pl_{tag}_fc", 100 * r.fc, 1)
put("top4_beat", int((top.y > top.fc).sum()))
put("bot4_beat", int((bot.y > bot.fc).sum()))
# tertiles of the gap: share beating forecast
tg = pd.qcut(D.gap.rank(method="first"), 3, labels=["low", "mid", "high"])
put("gap_hi_beat", int(round(100 * (D[tg == "high"].y > D[tg == "high"].fc).mean())))
put("gap_lo_beat", int(round(100 * (D[tg == "low"].y > D[tg == "low"].fc).mean())))
put("gap_hi_press", 100 * D[tg == "high"].y.mean(), 1)
put("gap_lo_press", 100 * D[tg == "low"].y.mean(), 1)
put("gap_ter_n", int((tg == "high").sum()))
# film-first flag: 40 launch above the cohort median and translation gap in the bottom third
D["flag"] = (D.c > D.c.median()) & (tg == "low")
put("flag_n", int(D.flag.sum()), claim="film-first flags")
put("flag_beat", int((D[D.flag].y > D[D.flag].fc).sum()), claim="flags beating forecast")
put("flag_press", 100 * D[D.flag].y.mean(), 1)
put("flag_fc", 100 * D[D.flag].fc.mean(), 1)
print("FLAG", D[D.flag][["display_name", "role", "gap_sd", "y", "fc"]])
print(board[["display_name", "role", "n", "c_raw", "g_raw", "gap_sd", "ci90", "y", "fc"]].to_string())

# ============================================================================ figures
# Fig 1 (hero): information curve + learning curve
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 5.0), gridspec_kw={"width_ratios": [1, 1]})
k_ = s2 / t2
R2_ = min(rt40 ** 2, 0.95)
nn = np.linspace(0, 40, 401)
film = nn / (nn + k_)
blend = R2_ + (1 - R2_) * nn / (nn + k_ / (1 - R2_))
ax[0].plot(nn, film, color=OI["grey"], lw=2.6, label="NFL film only")
ax[0].plot(nn, blend, color=OI["blue"], lw=2.6, label="Combine 40 + NFL film")
lo_fe, hi_fe = float(SC.loc["DL", "fe_lo"]), float(SC.loc["DL", "fe_hi"])
ax[0].fill_between(nn, film, (nn + hi_fe) / (nn + hi_fe + k_), where=nn >= 0, color=OI["blue"], alpha=0.12,
                   lw=0, label=f"80% interval of the Combine's worth")
yv = 0.5
n_film = yv * k_ / (1 - yv)
ax[0].annotate("", xy=(n_film - fe40, yv), xytext=(n_film, yv),
               arrowprops=dict(arrowstyle="<->", color=OI["verm"], lw=2.2, shrinkA=0, shrinkB=0))
ax[0].text(n_film + 0.6, yv - 0.012, f"the 40 saves {fe40:.1f} rushes", color=OI["verm"], ha="left", va="center",
           fontsize=11, fontweight="bold")
ax[0].scatter([0], [R2_], color=OI["blue"], s=50, zorder=5, clip_on=False)
ax[0].annotate(f"40 alone: {R2_:.2f}", (0, R2_), xytext=(2.2, 0.05), fontsize=10, color=OI["blue"],
               arrowprops=dict(arrowstyle="-", color=OI["blue"], lw=1))
ax[0].set_xlabel("Random NFL pass rushes observed")
ax[0].set_ylabel("Share of true NFL launch known\n(reliability)")
ax[0].set_ylim(0, 1)
ax[0].set_xlim(0, 30)
ax[0].legend(loc="lower right", frameon=False)
ax[0].set_title("A. How well we know his NFL launch", loc="left")
ks = np.array(KS)
for lab, col, key in [("NFL film only", OI["grey"], "film"), ("Combine 40 + NFL film", OI["blue"], "blend")]:
    m_ = np.array([LC[key][k][0] for k in KS])
    lo_ = np.array([LC[key][k][1] for k in KS])
    hi_ = np.array([LC[key][k][2] for k in KS])
    ax[1].plot(ks, m_, "-o", color=col, lw=2.4, ms=6, label=lab)
    ax[1].fill_between(ks, lo_, hi_, color=col, alpha=0.15)
ax[1].scatter([0], [LC0], color=OI["blue"], s=60, marker="D", zorder=5)
ax[1].annotate("40 alone", (0, LC0), xytext=(0.35, LC0 + 0.012), fontsize=10, color=OI["blue"])
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xlabel("Random NFL pass rushes observed")
ax[1].set_ylabel("Partial rank corr. with pressure rate\nbeyond pick, weight, EDGE share")
ax[1].set_xlim(-0.5, 12.5)
ax[1].set_title("B. Predicting his pressure rate", loc="left")
ax[1].legend(loc="upper left", frameon=False)
fig.text(0.01, 0.01, f"DL, 2023-25 classes. A: variance split from {int(jDL.n.sum())} random rushes by {len(jDL)} DL "
         f"with a tracked 40.\nB: {len(Dlc)} DL with 12+ rushes; rushes resampled {REPS} times (band = 10th-90th "
         "percentile); the Combine prior is fit on the other two classes.", fontsize=9.5, color="#333333")
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, "fig1_exchange.png")

# Fig 2: five-group screen, common axes
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 4.8), sharey=True, gridspec_kw={"width_ratios": [1, 1]})
order = GROUPS[::-1]
for i, grp in enumerate(order):
    s = SC.loc[grp]
    col = OI["blue"] if grp == "DL" else (OI["orange"] if grp == "WR" else OI["grey"])
    ax[0].errorbar(s.r, i, xerr=[[s.r - s.r_lo], [s.r_hi - s.r]], fmt="o", ms=8, capsize=4, color=col, lw=2)
    ax[1].errorbar(s.py, i, xerr=[[s.py - s.py_lo], [s.py_hi - s.py]], fmt="s", ms=8, capsize=4, color=col, lw=2)
    ax[0].text(s.r_hi + 0.04, i, f"FE {s.fe:.1f} {CELLS[grp]['unit']}" if s.rel1 >= REL_MIN else "FE: too noisy",
               va="center", fontsize=9.5, color=col)
    ax[1].text(s.py_hi + 0.04, i, f"{CELLS[grp]['out']}, n={int(s.n_y)}", va="center", fontsize=9.5, color=col)
for a_ in ax:
    a_.axvline(0, color="k", lw=0.8)
    a_.set_xlim(-0.7, 1.25)
ax[0].set_yticks(range(len(order)), [f"{g}: {CELLS[g]['label'].replace(' vs ', chr(10) + 'vs ')}\n(n = {int(SC.loc[g, 'n'])})"
                                    for g in order], fontsize=9.5)
ax[0].set_ylim(-0.6, len(order) - 0.4)
ax[0].set_xlabel("Combine vs NFL player means, r (95% CI)")
ax[1].set_xlabel("NFL movement vs production, partial rho (95% CI)")
ax[0].set_title("A. Combine to NFL, same movement", loc="left", fontsize=11.5)
ax[1].set_title("B. NFL movement to production, beyond pick", loc="left", fontsize=11.5)
fig.tight_layout(w_pad=1.5)
save(fig, "fig2_groups.png")

# Fig 3: DL production forest, common n, in sample + out of class
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
labs = ["Combine 40 launch (draft-time)", "Translation gap", "NFL launch, 20 rushes", "Official 40 time",
        "Official 10-yd split"]
cols = [OI["blue"], OI["purple"], OI["green"], OI["grey"], OI["grey"]]
for i, (lab, col) in enumerate(zip(labs[::-1], cols[::-1])):
    r = PR.loc[lab]
    ax[0].errorbar(r.rho, i, xerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt="o", ms=8, capsize=4, color=col, lw=2)
    ax[0].text(0.78, i, f"n={int(r.n)}", va="center", fontsize=9.5)
ax[0].set_yticks(range(len(labs)), labs[::-1])
ax[0].axvline(0, color="k", lw=0.8)
ax[0].set_xlim(-0.5, 0.9)
ax[0].set_xlabel("Partial rank corr. with pressure rate (95% CI)")
ax[0].set_title("A. In sample, beyond pick, weight, EDGE share", loc="left", fontsize=11)
olabs = [("c", "Combine 40 launch", OI["blue"]), ("gap", "Translation gap", OI["purple"]),
         ("nfl", "NFL launch", OI["green"])]
XL = (-0.3, 0.3)
for i, (f, lab, col) in enumerate(olabs[::-1]):
    v = float(NUM[f"oc_{f}"])
    lo, hi = float(NUM[f"oc_{f}_lo"]), float(NUM[f"oc_{f}_hi"])
    lo_c, hi_c = max(lo, XL[0]), min(hi, XL[1])
    ax[1].plot([lo_c, hi_c], [i, i], color=col, lw=2)
    ax[1].plot(v, i, "o", color=col, ms=8)
    if lo < XL[0]:
        ax[1].annotate("", xy=(XL[0], i), xytext=(XL[0] + 0.04, i),
                       arrowprops=dict(arrowstyle="->", color=col, lw=2))
        ax[1].text(XL[0] + 0.005, i - 0.3, f"to {lo:.2f}".replace("-", "−"), fontsize=9.5, color=col)
    if hi > XL[1]:
        ax[1].annotate("", xy=(XL[1], i), xytext=(XL[1] - 0.04, i), arrowprops=dict(arrowstyle="->", color=col, lw=2))
ax[1].set_yticks(range(3), [o[1] for o in olabs[::-1]])
ax[1].set_ylim(-0.6, 2.5)
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_xlim(*XL)
ax[1].set_xlabel("Held-out R² change vs baseline (95% CI)")
ax[1].set_title(f"B. Out of class (baseline R² {NUM['oc_base']})", loc="left", fontsize=11)
fig.tight_layout(w_pad=2.0)
save(fig, "fig3_pressure.png")

# Fig 4: named rushers, traces
sel = list(top.index[:3]) + list(bot.index[-3:])
tg_ = np.arange(TW) / 10
dl_ev = DL40.set_index("event_id")
fig, axs = plt.subplots(2, 3, figsize=(FIGW, 6.4), sharex=True, sharey=True)
ymax = 0
for ax_, nid in zip(axs.flat, sel):
    r = D.loc[nid]
    col = OI["blue"] if r.gap > 0 else OI["verm"]
    for eid in dl_ev.index[dl_ev.nfl_id == nid]:
        if eid in traces_d:
            ax_.plot(tg_, traces_d[eid], ls="--", color="k", lw=1.6)
            ymax = max(ymax, traces_d[eid].max())
    tr_ = [v for kk_, v in traces_g.items() if kk_[0] == nid]
    for v in tr_:
        ax_.plot(tg_, v, color=col, lw=0.6, alpha=0.3)
        ymax = max(ymax, v.max())
    ax_.plot(tg_, np.mean(tr_, 0), color=col, lw=3)
    ax_.set_title(f"{r.display_name} ({r.role}, pick {r.pick_s})\ngap {r.gap_sd:+.2f} sd; pressure "
                  f"{100 * r.y:.1f}% vs {100 * r.fc:.1f}% forecast".replace("-", "−"), fontsize=10.5)
    if r.flag:
        ax_.text(0.97, 0.05, "FILM-FIRST FLAG", transform=ax_.transAxes, ha="right", fontsize=10,
                 fontweight="bold", color=OI["verm"], bbox=dict(fc="white", ec=OI["verm"], lw=1.2))
for ax_ in axs.flat:
    ax_.set_ylim(0, ymax * 1.05)
for ax_ in axs[1]:
    ax_.set_xlabel("Seconds from onset (40) or snap (NFL)")
for ax_ in axs[:, 0]:
    ax_.set_ylabel("Speed (yd/s)")
fig.text(0.01, 0.005, "Top row: largest positive gaps (launch harder on Sundays than the 40 predicts). Bottom row: "
         "largest negative gaps.\nDashed black = Combine 40 reps; thin = single random NFL rushes; thick = NFL mean. "
         "Flag = 40 launch above median and gap in the bottom third.", fontsize=9.5, color="#333333")
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "fig4_rushers.png")


# ============================================================================ render + numbers
def render():
    tpl = OUT / "writeup_template.md"
    if not tpl.exists():
        return set()
    text = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda m: TABLES[m.group(1)], tpl.read_text())
    used = set()

    def sub(m):
        k = m.group(1)
        if k not in NUM:
            raise KeyError(f"template uses unknown key {k}")
        used.add(k)
        v = NUM[k]
        return "−" + v[1:] if v.startswith("-") else v
    (OUT / "writeup.md").write_text(re.sub(r"\{\{(\w+)\}\}", sub, text))
    return used


used = render()
keys = used if used else set(NUM)
out = {}
for k in sorted(keys):
    v = NUM[k]
    e = {"text": v.replace("-", "−")}
    if v.startswith("-"):
        e["magnitude"] = v[1:]
    if CLAIM.get(k):
        e["claim"] = CLAIM[k]
    out[k] = e
# decimals that appear inside generated tables (each comes from a computed value in this script)
tab_txt = "\n".join(TABLES.values())
out["_table_values"] = sorted(set(re.findall(r"(?<![\w.])(\d+\.\d{1,3})(?![\w.])", tab_txt)))
orphans = sorted(set(NUM) - keys)
(OUT / "numbers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
(OUT / "tables.json").write_text(json.dumps(TABLES, indent=1, ensure_ascii=False))
print("unused keys (not written):", len(orphans))
if "--dump" in sys.argv:
    print(json.dumps(NUM, indent=1))
