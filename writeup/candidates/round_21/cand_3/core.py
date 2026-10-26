"""Shared building blocks for round-21 cand_3 (Break G + Cut Match translation; imported by analysis.py and appendix.py).

Nothing here writes files. Every table is rebuilt from data/processed/*.parquet:
- Combine route-drill reps -> Break G per rep and per receiver (definition frozen in round 9, unchanged);
- full-season NGS separation -> season and career separation over expected (SOE);
- sampled NFL snaps -> the same Break G code on slants, outs, hitches, ins and corners, snap SOE, top speed;
- consecutive-season pairs for the day-3 test.

Break G, in one sentence a coach can repeat: on the receiver's first real cut (30 degrees or more after 1.5 yd of
stem, entered faster than 2.5 yd/s), the average push he puts into the ground over the one second around the cut,
braking plus turning plus driving out, as a share of his body weight (g). Per rep it is z-scored within drill and
Combine year; the receiver score averages his route drills and is shrunk toward the mean for few reps.
"""
from __future__ import annotations

import hashlib
import os
import warnings

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import beta as _beta
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data" / "processed"
DT = 0.1
G_YD = 10.73  # 1 g in yd/s^2
IN_YD = 36.0
NBOOT = int(os.environ.get("BDB_NBOOT", 2000))
NPERM = int(os.environ.get("BDB_NPERM", 2000))
SESOI_IN = 3.0  # smallest effect of interest (inches of season SOE per sd), fixed in round 9 before the test
CLASSES = (2023, 2024, 2025)
FAMILIES = ["SLANT", "OUT", "HITCH", "IN", "CORNER"]
BASE = ["logpick", "forty_f", "combine_weight_f", "ngs_athleticism_score_f", "te"]
GAP = 32  # pick-gap threshold of the override rule, fixed in round 18 before scoring
DRILL_FAMILY = {"SLANT_ROUTE_LEFT": "SLANT", "SLOT_STRIKE_ROUTE_LEFT": "SLANT", "SPEED_OUT_ROUTE_LEFT": "OUT",
                "CURL_ROUTE_RIGHT": "HITCH", "COMEBACK_ROUTE_RIGHT": "HITCH", "HOOK_ROUTE_RIGHT": "HITCH",
                "DAGGER_ROUTE_LEFT": "IN", "IN_ROUTE_LEFT": "IN", "SLOT_SAIL_ROUTE_RIGHT": "CORNER",
                "CORNER_ROUTE": "CORNER"}
HALF_A = {"SLANT_ROUTE_LEFT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT", "SPEED_OUT_ROUTE_LEFT", "HOOK_ROUTE_RIGHT"}
COMP = ["bg", "decel", "v_apex", "reaccel"]


def rng_for(name: str) -> np.random.Generator:
    """One seeded generator per statistic, so verify.py can redraw exactly the same resamples."""
    return np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))


def zs(s):
    return (s - s.mean()) / s.std(ddof=1)


def pct(a, q):
    return float(np.nanpercentile(np.asarray(a, float), q))


def jeff(k, n):
    """Jeffreys 90% interval for k of n, in %."""
    return 100 * _beta.ppf(0.05, k + 0.5, n - k + 0.5), 100 * _beta.ppf(0.95, k + 0.5, n - k + 0.5)


def wls(A, y, w):
    s = np.sqrt(np.asarray(w, float))
    return np.linalg.lstsq(A * s[:, None], np.asarray(y, float) * s, rcond=None)[0]


def clusters(cl):
    cl = np.asarray(cl)
    return [np.where(cl == c)[0] for c in pd.unique(cl)]


def boot_rho(a, b, name, n=NBOOT, cluster=None):
    """Spearman rho with a 90% bootstrap interval (receiver-clustered if cluster is given)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    rng = rng_for(name)
    groups = clusters(np.asarray(cluster)[m]) if cluster is not None else None
    out = []
    for _ in range(n):
        if groups is None:
            ix = rng.integers(0, len(a), len(a))
        else:
            ix = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        out.append(spearmanr(a[ix], b[ix])[0])
    return float(spearmanr(a, b)[0]), pct(out, 5), pct(out, 95), int(len(a))


def icc1(df, col, gcol="nfl_id"):
    k = df.groupby(gcol).size()
    k0 = (len(df) - (k ** 2).sum() / len(df)) / (len(k) - 1)
    gm = df[col].mean()
    msb = (k * (df.groupby(gcol)[col].mean() - gm) ** 2).sum() / (len(k) - 1)
    msw = ((df[col] - df.groupby(gcol)[col].transform("mean")) ** 2).sum() / (len(df) - len(k))
    return (msb - msw) / (msb + (k0 - 1) * msw)


# ----------------------------------------------------------------------------- kinematics (frozen definition)
def kin(x, y):
    w = 7 if len(x) >= 7 else max(3, (len(x) // 2) * 2 - 1)
    vx = savgol_filter(x, w, 2, deriv=1, delta=DT)
    vy = savgol_filter(y, w, 2, deriv=1, delta=DT)
    return vx, vy, savgol_filter(vx, w, 2, deriv=1, delta=DT), savgol_filter(vy, w, 2, deriv=1, delta=DT)


def angdiff(a, b):
    d = np.abs(a - b) % 360
    return np.where(d > 180, 360 - d, d)


def break_mechanics(x, y, i0, i1, min_path=1.5, min_turn=30.0, h=6, entry_speed=2.5, hw=5):
    """First real cut of a route and its mechanics. Break G = mean |acceleration| (g) over 1.0 s centred on it."""
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
    return dict(i_break=i, i_apex=apex, v_in=float(s[iin]), v_apex=float(s[apex]), v_out=float(s[iout]),
                decel=float((s[iin] - s[apex]) / max((apex - iin) * DT, DT)),
                reaccel=float((s[iout] - s[apex]) / max((iout - apex) * DT, DT)),
                bg=float(amag[lo:hi].mean() / G_YD))


def speed_trace(x, y, iap, w=10):
    """Speed (yd/s) from 1.0 s before to 1.0 s after the slowest point of the cut."""
    vx, vy, _, _ = kin(np.asarray(x, float), np.asarray(y, float))
    s = np.hypot(vx, vy)
    out = np.full(2 * w + 1, np.nan)
    lo, hi = max(0, int(iap) - w), min(len(s), int(iap) + w + 1)
    out[lo - (int(iap) - w):lo - (int(iap) - w) + (hi - lo)] = s[lo:hi]
    return out


# ----------------------------------------------------------------------------- Combine
def build_combine():
    events = pd.read_parquet(P / "combine_events.parquet")
    ev = events[events.drill_name.isin(DRILL_FAMILY) & events.position_group.isin(["WR", "TE"])]
    frames = pd.read_parquet(P / "combine_frames.parquet", filters=[("event_id", "in", ev.event_id.tolist())])
    reps = {}
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
        reps[eid] = (x, y, i1)
    rows = []
    for eid, (x, y, i1) in reps.items():
        r = break_mechanics(x, y, 0, i1)
        if r:
            rows.append(dict(event_id=eid, **r))
    cb = pd.DataFrame(rows).merge(ev[["event_id", "nfl_id", "draft_year", "drill_name", "attempt_no"]], on="event_id")
    cb["family"] = cb.drill_name.map(DRILL_FAMILY)
    for m in COMP:
        cb[m + "_z"] = cb.groupby(["drill_name", "draft_year"])[m].transform(zs)
    # receiver score: mean z per drill, mean over drills, shrunk by the between-drill reliability (empirical Bayes)
    cpd = cb.groupby(["nfl_id", "drill_name", "family"], as_index=False)[[m + "_z" for m in COMP] + ["bg"]].mean()
    cpl = cpd.groupby("nfl_id")[[m + "_z" for m in COMP]].mean()
    cpl["n_drills"] = cpd.groupby("nfl_id").size()
    info = {}
    for m in COMP:
        icc = max(icc1(cpd, m + "_z"), 0.01)
        vt = cpd[m + "_z"].var()
        var_true, var_err = icc * vt, (1 - icc) * vt
        post_var = 1 / (1 / var_true + cpl.n_drills / var_err)
        cpl[m.upper() + "_S"] = post_var * (cpl.n_drills * cpl[m + "_z"] / var_err) / np.sqrt(var_true)
        info[m] = (icc, np.sqrt(var_true))
    cpl["BG"] = cpl.BG_S
    return dict(ev=ev, reps=reps, cb=cb, cpd=cpd, cpl=cpl, info=info, n_reps=len(ev))


# ----------------------------------------------------------------------------- players, baseline, season SOE
def build_players(cpl):
    players = pd.read_parquet(P / "players.parquet")
    cr = pd.read_parquet(P / "combine_results.parquet").set_index("nfl_id")
    pb = players.set_index("nfl_id")[["draft_overall_pick", "draft_round", "position_group", "draft_year",
                                      "display_name"]].join(cr[["forty", "combine_weight", "ngs_athleticism_score"]])
    pb = pb[pb.position_group.isin(["WR", "TE"])].copy()
    pb["logpick"] = np.log(pb.draft_overall_pick.fillna(262))
    pb["te"] = (pb.position_group == "TE").astype(float)
    for c in ("forty", "combine_weight", "ngs_athleticism_score"):
        pb[c + "_s"] = pb.groupby(["draft_year", "position_group"])[c].transform(zs)
        pb[c + "_f"] = pb[c + "_s"].fillna(0.0)
    X = cpl.join(pb, how="inner")
    return players, pb, X


def season_soe(ss_all, cushion=True):
    """Season SOE (yd): NGS average separation minus its route-weighted expectation from alignment, TE, cushion, season."""
    ss = ss_all[ss_all.position_group.isin(["WR", "TE"]) & ss_all.routes.gt(0) & ss_all.avg_separation.notna()].copy()
    cols = [np.ones(len(ss)), (ss.main_alignment == "SLOT_WR"), (ss.main_alignment == "TE"),
            (ss.position_group == "TE")]
    if cushion:
        cols.append(ss.avg_cushion.fillna(ss.avg_cushion.mean()))
    cols += [ss.season == 2024, ss.season == 2025]
    A = np.column_stack(cols).astype(float)
    ss["ssoe"] = ss.avg_separation - A @ wls(A, ss.avg_separation.to_numpy(float), ss.routes.to_numpy(float))
    return ss


def career(ss, min_routes=0):
    s = ss[ss.routes >= min_routes]
    return s.groupby("nfl_id").apply(lambda d: pd.Series(dict(ssoe=np.average(d.ssoe, weights=d.routes),
                                                              routes=d.routes.sum())), include_groups=False)


# ----------------------------------------------------------------------------- held-out tests (leave one class out)
def loco_parts(D, bgcol="BG", ycol="ssoe"):
    XB = np.column_stack([np.ones(len(D)), D[BASE].to_numpy(float)])
    return XB, D[ycol].to_numpy(float), D.routes.to_numpy(float), D.draft_year.to_numpy(), D[bgcol].to_numpy(float)


def ho_slope(XB, y, w, cl, bg):
    """Held-out Break G slope (inches per sd): the no-tracking forecast fitted on two classes is the offset on the third;
    the residual is regressed on Break G with class intercepts, route-weighted."""
    res = np.empty(len(y))
    for c in CLASSES:
        t = cl == c
        if t.any():
            res[t] = y[t] - XB[t] @ wls(XB[~t], y[~t], w[~t])
    present = [c for c in CLASSES if (cl == c).any()]
    A = np.column_stack([(cl == c).astype(float) for c in present] + [bg])
    return IN_YD * wls(A, res, w)[-1]


def ho_r2(XB, y, w, cl, bg):
    Xb = np.column_stack([XB, bg])
    f0, f1 = np.empty(len(y)), np.empty(len(y))
    for c in CLASSES:
        t = cl == c
        if t.any():
            f0[t] = XB[t] @ wls(XB[~t], y[~t], w[~t])
            f1[t] = Xb[t] @ wls(Xb[~t], y[~t], w[~t])
    sst = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    r0, r1 = 1 - np.sum(w * (y - f0) ** 2) / sst, 1 - np.sum(w * (y - f1) ** 2) / sst
    return np.array([r0, r1, r1 - r0, IN_YD * np.sqrt(np.average((y - f0) ** 2, weights=w))])


def strat_ix(cl, rng):
    return np.concatenate([rng.choice(np.where(cl == c)[0], int((cl == c).sum())) for c in CLASSES if (cl == c).any()])


# ----------------------------------------------------------------------------- NFL snaps (sampled)
def build_nfl(players):
    gp = pd.read_parquet(P / "game_plays_sample.parquet")
    gp = gp[gp.week.between(1, 18)]
    gp = gp.merge(players[["nfl_id", "position_group"]], on="nfl_id")
    gp = gp[gp.position_group.isin(["WR", "TE"])].copy()
    gf = pd.read_parquet(P / "game_frames_sample.parquet",
                         columns=["game_id", "play_id", "nfl_id", "frame", "x", "y", "s", "event"])
    gf = gf.merge(gp[["game_id", "play_id", "nfl_id"]], on=["game_id", "play_id", "nfl_id"])
    # NFL top speed: 90th percentile of the receiver's per-snap maximum speed over the whole sampled play (8+ snaps)
    pm = gf.groupby(["nfl_id", "game_id", "play_id"]).s.max().groupby("nfl_id")
    vtop = pm.quantile(0.9)[pm.size() >= 8]
    rows, traces = [], {}
    for k, d in gf.groupby(["game_id", "play_id", "nfl_id"], sort=False):
        d = d.sort_values("frame")
        e = d.event.fillna("").to_numpy()
        if not ((e == "ball_snap").any() and (e == "pass_forward").any()):
            continue
        i0, ipf = int(np.argmax(e == "ball_snap")), int(np.argmax(e == "pass_forward"))
        base = dict(game_id=k[0], play_id=k[1], nfl_id=k[2], t_throw=(ipf - i0) * DT)
        r = break_mechanics(d.x.to_numpy(), d.y.to_numpy(), i0, min(len(d), ipf + 6))
        if r:
            base.update(r)
            traces[k] = (d.x.to_numpy(), d.y.to_numpy())
        rows.append(base)
    gb = pd.DataFrame(rows).merge(gp, on=["game_id", "play_id", "nfl_id"])
    gb["sep"] = gb.separation_at_pass_forward
    gb["man"] = gb.team_coverage_type.fillna("").str.contains("MAN")
    gb["slotp"] = (gb.lined_up_position == "SLOT_WR").astype(float)
    gb["inline"] = gb.lined_up_position.isin(["TE", "FB", "RB"]).astype(float)
    gb["cush"] = gb.cushion.fillna(gb.cushion.median())
    gb["cush_na"] = gb.cushion.isna().astype(float)
    gb["pl"] = gb.pass_length.fillna(gb.pass_length.median())
    gb["route"] = gb.route_ran.fillna("OTHER")
    routes = sorted(gb.route.unique())

    def design(df):
        cols = [np.ones(len(df)), df.pl, df.pl ** 2 / 100, df.cush, df.cush_na, df.man.astype(float), df.slotp,
                df.inline, df.t_throw, df.target.astype(float), df.down, df.yards_to_go]
        cols += [(df.route == r).astype(float) for r in routes[1:]]
        return np.column_stack(cols)

    # snap SOE: separation at the throw minus its expectation for the same coverage family, alignment, depth, cushion
    # and route (fitted on 2023-24 snaps), so a man or slot cell is read against the league in that cell
    ok = gb.sep.notna()
    dev = ok & (gb.season < 2025)
    beta = np.linalg.lstsq(design(gb[dev]), gb.loc[dev, "sep"].to_numpy(), rcond=None)[0]
    gb["soe"] = np.where(ok, gb.sep - design(gb) @ beta, np.nan)
    g = gb[gb.route_ran.isin(FAMILIES) & gb.bg.notna()].copy()
    g["bg_gz"] = g.groupby(["route_ran", "position_group"]).bg.transform(zs)
    g["v_lg"] = g.v_apex - g.groupby("route_ran").v_apex.transform("median")
    g = g.sort_values(["nfl_id", "game_id", "play_id"])
    g["half"] = g.groupby("nfl_id").cumcount() % 2
    return dict(gp=gp, gb=gb, g=g, traces=traces, vtop=vtop)


# ----------------------------------------------------------------------------- consecutive-season pairs (day 3)
def shrink_fit(ssub):
    """Empirical-Bayes shrinkage of season SOE toward the mean for few routes: (mean, skill variance, noise x routes)."""
    s = ssub[ssub.routes >= 50]
    mu = float(np.average(s.ssoe, weights=s.routes))
    pv = s.pivot_table(index="nfl_id", columns="season", values="ssoe")
    seas = sorted(s.season.unique())
    cp = pd.concat([pv[[a, a + 1]].dropna().set_axis(["a", "b"], axis=1) for a in seas if a + 1 in seas])
    tau2 = float(np.mean((cp.a - mu) * (cp.b - mu)))
    iv = 1.0 / s.routes.to_numpy(float)
    s2 = float(np.sum(((s.ssoe.to_numpy() - mu) ** 2 - tau2) * iv) / np.sum(iv ** 2))
    return mu, tau2, s2


def shr_with(par, y, r):
    mu, tau2, s2 = par
    return mu + tau2 / (tau2 + s2 / np.asarray(r, float)) * (np.asarray(y, float) - mu)


def build_pairs(ss, pb, min_routes=100):
    rows = []
    for a, b in ((2023, 2024), (2024, 2025)):
        s1 = ss[ss.season == a].set_index("nfl_id")
        s2 = ss[ss.season == b].set_index("nfl_id")
        j = s1[["ssoe", "routes", "games", "team"]].join(s2[["ssoe", "routes"]], rsuffix="_2", how="inner")
        j = j.rename(columns={"ssoe": "y1", "routes": "r1", "games": "g1", "ssoe_2": "y2", "routes_2": "r2"})
        j["s1"] = a
        rows.append(j)
    pr = pd.concat(rows).join(pb[["draft_round", "draft_overall_pick", "display_name", "draft_year", "logpick", "te"]])
    pr = pr[(pr.r1 >= min_routes) & (pr.r2 >= min_routes)].copy()
    pr["pid"] = pr.index
    pr["day3"] = pr.draft_round.isna() | (pr.draft_round >= 4)
    pr["lr1"] = np.log(pr.r1)
    pr["hit"] = pr.y2 > 0
    pr["pick"] = pr.draft_overall_pick.fillna(262)
    return pr.reset_index(drop=True)


F0 = ["logpick", "te", "lr1"]
F1 = F0 + ["y1s"]


def fit_ols(df, cols, ycol="y2"):
    A = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    b = np.linalg.lstsq(A, df[ycol].to_numpy(float), rcond=None)[0]
    return b, float(np.std(df[ycol].to_numpy(float) - A @ b, ddof=len(b)))


def pred(df, b, cols):
    return np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)]) @ b


def loco_forecasts(pr, ss, cls_of):
    """Leave-one-draft-class-out forecasts of next-season SOE: shrinkage, M0 (pick, TE, log routes) and M1 (+ shrunk
    last-season SOE) all fitted on the other classes' player-seasons and pairs."""
    out = pr.copy()
    out["f0"], out["f1"], out["y1s"] = np.nan, np.nan, np.nan
    for c in sorted(out.draft_year.unique()):
        tst = (out.draft_year == c).to_numpy()
        par = shrink_fit(ss[cls_of.reindex(ss.nfl_id).to_numpy() != c])
        pp = out.assign(y1s=shr_with(par, out.y1, out.r1))
        trn = pp[~tst]
        b0, _ = fit_ols(trn, F0)
        b1, _ = fit_ols(trn, F1)
        out.loc[tst, "f0"] = pred(pp[tst], b0, F0)
        out.loc[tst, "f1"] = pred(pp[tst], b1, F1)
        out.loc[tst, "y1s"] = pp.y1s.to_numpy()[tst]
    return out


def half_lists(d3, fcol="f1"):
    """Each season, the top half of eligible day-3 receivers by the SOE forecast and by pick order (earlier pick;
    undrafted ties broken by more routes). Returns boolean columns top_soe, top_pick."""
    d = d3.copy()
    n_half = d.groupby("s1").s1.transform("size") // 2
    d["top_soe"] = d.groupby("s1")[fcol].rank(ascending=False, method="first") <= n_half
    d["pk_key"] = d.pick * 1e4 - d.r1
    d["top_pick"] = d.groupby("s1").pk_key.rank(method="first") <= n_half
    return d


def override_contrasts(d3w):
    """Same-season contrasts between two day-3 wide receivers: gap, the pick-order choice and the SOE choice."""
    rows = []
    d = d3w.reset_index(drop=True)
    for s in sorted(d.s1.unique()):
        ii = np.where(d.s1.to_numpy() == s)[0]
        for x in range(len(ii)):
            for z in range(x + 1, len(ii)):
                a, b = ii[x], ii[z]
                gap = abs(d.pick.iloc[a] - d.pick.iloc[b])
                pick_ch = a if (d.pick.iloc[a], -d.r1.iloc[a]) < (d.pick.iloc[b], -d.r1.iloc[b]) else b
                soe_ch = a if d.y1s.iloc[a] > d.y1s.iloc[b] else b
                rows.append(dict(a=a, b=b, gap=gap, pick=pick_ch, soe=soe_ch, s1=s))
    return d, pd.DataFrame(rows)


def override_stats(ct, y2in, gap, w=None):
    """Overrides = contrasts with pick gap < gap where the higher last-season SOE is not the earlier pick.
    wrong = % of overrides where the SOE choice ends with the lower next-season SOE; gain/loss in inches;
    break-even wrong rate = gain / (gain + loss); net = mean next-season SOE gained per override."""
    m = ((ct.gap < gap) & (ct.soe != ct.pick)).to_numpy()
    w = np.ones(len(ct)) if w is None else np.asarray(w, float)
    m = m & (w > 0)
    if m.sum() == 0:
        return dict(n=0, wrong=np.nan, be=np.nan, net=np.nan)
    gs = y2in[ct.soe.to_numpy()[m]] - y2in[ct.pick.to_numpy()[m]]
    ww = w[m]
    rt, wr = gs > 0, gs < 0
    gain = np.average(gs[rt], weights=ww[rt]) if ww[rt].sum() else np.nan
    loss = -np.average(gs[wr], weights=ww[wr]) if ww[wr].sum() else np.nan
    return dict(n=int(m.sum()), wrong=100 * np.average(wr, weights=ww), be=100 * gain / (gain + loss),
                net=np.average(gs, weights=ww), k=int(wr[w[m] > 0].sum()))


def receiver_weights(pids, rng):
    """Receiver-cluster bootstrap for contrasts: each receiver's resample count; a contrast is weighted by the product."""
    u = np.unique(pids)
    cnt = pd.Series(rng.choice(u, len(u))).value_counts()
    return cnt.reindex(pids).fillna(0).to_numpy()


# ----------------------------------------------------------------------------- Cut Match (per-receiver translation)
def cut_match(cb, g, reps, traces, min_reps=2, min_breaks=4):
    """Per receiver, on the route families he ran in BOTH settings: his speed-through-the-cut curve (speed every 0.1 s
    from 1.0 s before to 1.0 s after the slowest point) minus the league median curve for that family, averaged over
    his drill reps and, separately, over his NFL breaks. Each setting is read against its own league, so the
    systematic drill-vs-game entry-speed gap cancels and only HIS deviation has to carry over.
    gap  = root-mean-square difference of the two deviation curves (yd/s; 0 = cuts on Sundays as he did in drills);
    dm, nm = his mean deviation in the drill and in the NFL (speed through the cut vs league, yd/s);
    gap_a / gap_b = the gap with only odd / even NFL breaks (reliability of the gap)."""
    cb, g = cb.copy(), g.copy()
    cb["tr"] = [speed_trace(*reps[e][:2], ia) for e, ia in zip(cb.event_id, cb.i_apex)]
    g["tr"] = [speed_trace(*traces[(r.game_id, r.play_id, r.nfl_id)], r.i_apex) for r in g.itertuples()]
    dl = {f: np.nanmedian(np.stack(cb[cb.family == f].tr.to_list()), 0) for f in FAMILIES}
    nl = {f: np.nanmedian(np.stack(g[g.route_ran == f].tr.to_list()), 0) for f in FAMILIES}
    cb["dv"] = [t - dl[f] for t, f in zip(cb.tr, cb.family)]
    g["dv"] = [t - nl[f] for t, f in zip(g.tr, g.route_ran)]
    rows = []
    gby = dict(tuple(g.groupby("nfl_id")))
    for pid, r in cb.groupby("nfl_id"):
        if pid not in gby:
            continue
        s = gby[pid]
        fams = sorted(set(r.family) & set(s.route_ran))
        r, s = r[r.family.isin(fams)], s[s.route_ran.isin(fams)]
        if len(r) < min_reps or len(s) < min_breaks:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = np.nanmean(np.stack(r.dv.to_list()), 0)
            b = np.nanmean(np.stack(s.dv.to_list()), 0)
            b0 = np.nanmean(np.stack(s.dv.iloc[::2].to_list()), 0)
            b1 = np.nanmean(np.stack(s.dv.iloc[1::2].to_list()), 0)
        rows.append(dict(nfl_id=pid, gap=float(np.sqrt(np.nanmean((a - b) ** 2))),
                         gap_a=float(np.sqrt(np.nanmean((a - b0) ** 2))), gap_b=float(np.sqrt(np.nanmean((a - b1) ** 2))),
                         dm=float(np.nanmean(a)), nm=float(np.nanmean(b)), nr=len(r), ns=len(s), fams=",".join(fams),
                         d_in=float(r.v_in.median()), d_ap=float(r.v_apex.median()), d_out=float(r.v_out.median()),
                         n_in=float(s.v_in.median()), n_ap=float(s.v_apex.median()), n_out=float(s.v_out.median()),
                         n_bg=float(s.bg_gz.mean())))
    T = pd.DataFrame(rows).set_index("nfl_id")
    # Cut Match score: the gap net of what his rep and break counts alone predict (few reps -> noisier curve -> bigger
    # gap), sign flipped so that higher = closer match, z-scored
    A = np.column_stack([np.ones(len(T)), np.log(T.nr), np.log(T.ns)])
    T["gap_res"] = T.gap - A @ np.linalg.lstsq(A, T.gap.to_numpy(float), rcond=None)[0]
    T["CM"] = zs(-T.gap_res)
    return T, cb, g, dl, nl


def quiet_mixedlm(*a, **k):
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(*a, **k).fit(reml=True)
