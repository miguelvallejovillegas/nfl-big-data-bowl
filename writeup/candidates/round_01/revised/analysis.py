"""Burst Translation, revised (round 1 revision of cand_3).

Question: does a defensive lineman's first second of movement in the Combine pass-rush drill show up on his NFL
pass-rush snaps, and does the gap explain pressure beyond what a front office already has? Positive control: the 40's
top speed vs NFL peak speed, validated class by class.

Changes vs cand_3: random-snap half is the primary game sample; rep-level burst is modelled with fixed effects
(drill x year; alignment, man/zone and press/off) and player random effects (empirical-Bayes posterior means);
true-score (disattenuated) correlations with bootstrap CIs; pressure rate is beta-binomial shrunk by rush snaps
(no hard cut); leave-one-class-out with repeated bootstrap draws, a within-class permutation test and Holm/BH
correction across every reported test; OL/TE burst composites (negative reliability) dropped.

Reads data/processed/*.parquet (career outcomes through analysis/insights.py:load). Writes numbers.json, tables.json
and figures/*.png next to this file.

Usage: uv run python writeup/candidates/round_01/revised/analysis.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROC = ROOT / "data" / "processed"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "analysis"))
import insights  # noqa: E402

RNG = np.random.default_rng(2027)
DT = 0.1
T = 16                 # frames kept after onset (0.0-1.5 s)
BURST = 11             # first-second window: frames 0..10 (0.0-1.0 s)
PRECONTACT = 6         # pre-contact window: frames 0..5 (0.0-0.5 s)
ON, PRE = 1.0, 0.5     # onset threshold (yd/s) and back-off threshold
GROUPS = ["DL", "OL", "WR", "TE", "DB"]
BURST_GROUPS = ["DL", "WR", "DB"]     # OL and TE drill composites have negative reliability (dropped)
NBOOT = 2000
NREP = 500
NPERM = 1000
PRESS_CUSHION = 3.0    # yards: cushion at the snap at or below this = press / tight
MATCHED = {
    "DL": ["PASS_RUSH_DRILL"],
    "OL": ["PASS_RUSH_DROP", "PASS_PRO_MIRROR_DRILL"],
    "WR": ["GO_ROUTE_RIGHT", "SLANT_ROUTE_LEFT", "COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT",
           "SPEED_OUT_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT", "SLOT_SAIL_ROUTE_RIGHT"],
    "TE": ["CORNER_ROUTE", "IN_ROUTE_LEFT", "HOOK_ROUTE_RIGHT", "FLAT_ROUTE_LEFT", "WHEEL_ROUTE_LEFT"],
    "DB": ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "LINE_DRILL"],
}
GAME_TYPE = {"DL": "pass rushes", "OL": "pass sets", "WR": "routes", "TE": "routes", "DB": "coverage snaps"}
DRILL_LABEL = {"DL": "pass-rush drill", "OL": "drop + mirror drills", "WR": "8 route drills", "TE": "5 route drills",
               "DB": "3 backpedal drills"}
# Okabe-Ito
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F", "black": "#000000"}
GCOL = {"DL": OI["red"], "OL": OI["orange"], "WR": OI["blue"], "TE": OI["green"], "DB": OI["purple"]}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
FIGW = 1600 / 150

NUM: dict = {}


def put(key: str, value, fmt: str = "{:.2f}") -> str:
    s = fmt.format(value) if not isinstance(value, str) else value
    NUM[key] = s
    return s


def log(m: str) -> None:
    print(m, flush=True)


# ----------------------------------------------------------------------------- curves
def onset_curve(s: np.ndarray, start: int = 0):
    """Speed for T frames from movement onset (first frame >= ON yd/s, backed up <=3 frames while >= PRE)."""
    s = np.asarray(s, float)[start:]
    if not len(s) or s[0] >= ON:
        return None
    hit = np.flatnonzero(s >= ON)
    if not len(hit):
        return None
    i = j = int(hit[0])
    while j > 0 and s[j - 1] >= PRE and i - j < 3:
        j -= 1
    c = s[j:j + T]
    return c if len(c) == T else None


def zscore(v: pd.Series) -> pd.Series:
    sd = v.std()
    return (v - v.mean()) / sd if sd and sd > 0 else v * np.nan


def build_curves(p: pd.DataFrame):
    ev = pd.read_parquet(PROC / "combine_events.parquet")
    cf = pd.read_parquet(PROC / "combine_frames.parquet", columns=["event_id", "frame", "entity_type", "s"])
    cf = cf[cf.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
    sp = dict(tuple(cf.groupby("event_id").s))
    rows = []
    for g, drills in MATCHED.items():
        e = ev[(ev.position_group == g) & ev.drill_name.isin(drills)]
        for r in e.itertuples():
            c = onset_curve(sp[r.event_id].values)
            if c is not None:
                rows.append((g, r.nfl_id, r.drill_name, r.draft_year, r.attempt_no, *c))
    D = pd.DataFrame(rows, columns=["grp", "nfl_id", "drill", "yr", "att"] + [f"c{i}" for i in range(T)])
    f40 = ev[ev.drill_type == "FORTY_YARD_DASH"]
    top40 = f40.groupby(["nfl_id", "draft_year"]).top_speed.max().reset_index()
    top40["top40_z"] = top40.groupby("draft_year").top_speed.transform(zscore)
    t40_pairs = f40.sort_values("attempt_no").groupby("nfl_id").top_speed.apply(
        lambda v: v.values[:2] if len(v) >= 2 else None).dropna()

    gp = pd.read_parquet(PROC / "game_plays_sample.parquet")
    gf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "s"])
    gf = gf.sort_values(["game_id", "play_id", "nfl_id", "frame"])
    k = ["game_id", "play_id", "nfl_id"]
    snapf = gp.set_index(k).snap_frame
    grows, peaks = [], []
    for key, d in gf.groupby(k, sort=False):
        sf = snapf.get(key, np.nan)
        if np.isnan(sf):
            continue
        s = d.s.values
        st = int(np.searchsorted(d.frame.values, sf))
        peaks.append((*key, float(s[st:].max()) if len(s) > st else np.nan))
        c = onset_curve(s, st)
        if c is not None:
            grows.append((*key, *c))
    G = pd.DataFrame(grows, columns=k + [f"c{i}" for i in range(T)])
    PK = pd.DataFrame(peaks, columns=k + ["peak"])
    meta = gp[k + ["season", "lined_up_position", "expected_points_added", "team_coverage_type", "cushion"]].copy()
    # The ETL took each player's 6 highest-|EPA| snaps, then 6 random snaps from the rest. Every random snap therefore
    # has |EPA| <= the 6th-highest, so the 6 lowest-|EPA| of the 12 are exactly the random draw.
    meta["ae"] = meta.expected_points_added.abs()
    meta["rk"] = meta.groupby("nfl_id").ae.rank(ascending=False, method="first")
    meta["nn"] = meta.groupby("nfl_id").ae.transform("size")
    meta["random_half"] = (meta.nn < 12) | (meta.rk > 6)
    meta["man"] = meta.team_coverage_type.fillna("").str.contains("MAN").map({True: "man", False: "zone"})
    meta["press"] = np.where(meta.cushion.isna(), "na", np.where(meta.cushion <= PRESS_CUSHION, "press", "off"))
    meta = meta.merge(p[["nfl_id", "position_group"]], on="nfl_id")
    G = G.merge(meta, on=k)
    PK = PK.merge(meta, on=k)
    G["grp"] = G.position_group
    PK["grp"] = PK.position_group
    return D, G, PK, top40, t40_pairs, gp


# ----------------------------------------------------------------------------- stats helpers
def residualize(df: pd.DataFrame, val: str, fe: list[str]) -> pd.Series:
    """Remove fixed effects (categorical) by OLS; return residual scaled to unit SD."""
    X = pd.get_dummies(df[fe].astype(str), drop_first=True).astype(float)
    A = np.c_[np.ones(len(df)), X.values]
    y = df[val].values
    r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return pd.Series(r / r.std(), index=df.index)


def var_components(ids: np.ndarray, v: np.ndarray):
    """One-way random-effects ANOVA: between-player tau2 and within sigma2 (players with >=2 reps)."""
    d = pd.DataFrame({"i": ids, "v": v}).dropna()
    k = d.groupby("i").v.transform("size")
    d = d[k >= 2]
    g = d.groupby("i").v
    n, N = g.ngroups, len(d)
    kk = g.size()
    k0 = (N - (kk ** 2).sum() / N) / (n - 1)
    grand = d.v.mean()
    msb = (kk * (g.mean() - grand) ** 2).sum() / (n - 1)
    msw = ((d.v - g.transform("mean")) ** 2).sum() / (N - n)
    tau2 = (msb - msw) / k0
    return float(tau2), float(msw), float(tau2 / (tau2 + msw))


def eb_means(df: pd.DataFrame, val: str):
    """Empirical-Bayes posterior player means (random intercept model), in true-score SD units."""
    tau2, s2, r1 = var_components(df.nfl_id.values, df[val].values)
    g = df.groupby("nfl_id")[val].agg(["mean", "size"])
    grand = df[val].mean()
    lam = tau2 / (tau2 + s2 / g["size"]) if tau2 > 0 else g["mean"] * np.nan
    post = grand + lam * (g["mean"] - grand)
    out = pd.DataFrame({"raw": g["mean"], "k": g["size"], "lam": lam,
                        "true": (post - grand) / np.sqrt(tau2) if tau2 > 0 else np.nan})
    return out, dict(tau2=tau2, s2=s2, r1=r1, kbar=float(g["size"].mean()))


def sb(r1: float, k: float) -> float:
    return k * r1 / (1 + (k - 1) * r1)


def reps_for(r1: float, target: float) -> float:
    return target * (1 - r1) / (r1 * (1 - target))


def spearman_ci(x, y):
    m = pd.notna(x) & pd.notna(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    r, pv = stats.spearmanr(x, y)
    n = len(x)
    b = [stats.spearmanr(x[i], y[i])[0] for i in (RNG.integers(0, n, n) for _ in range(NBOOT))]
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return float(r), float(lo), float(hi), int(n), float(pv)


def true_corr(Dg: pd.DataFrame, Gg: pd.DataFrame, ids: np.ndarray):
    """Disattenuated Pearson correlation between drill and game player means; player bootstrap recomputes reliability."""
    def agg(R):
        a = R.groupby("nfl_id").z.agg(["sum", "size", lambda v: ((v - v.mean()) ** 2).sum()])
        a.columns = ["s", "k", "ss"]
        return a
    ad, ag = agg(Dg).reindex(ids), agg(Gg).reindex(ids)

    def comp(a):
        a = a.dropna()
        m = a.s / a.k
        a2 = a[a.k >= 2]
        N, n = a2.k.sum(), len(a2)
        k0 = (N - (a2.k ** 2).sum() / N) / (n - 1)
        m2 = a2.s / a2.k
        grand = a2.s.sum() / N
        msb = (a2.k * (m2 - grand) ** 2).sum() / (n - 1)
        msw = a2.ss.sum() / (N - n)
        r1 = (msb - msw) / (msb + (k0 - 1) * msw)
        return m, sb(r1, a.k.mean())

    def one(idx):
        md, rd = comp(ad.iloc[idx].reset_index(drop=True))
        mg, rg = comp(ag.iloc[idx].reset_index(drop=True))
        ok = md.index.intersection(mg.index)
        r = np.corrcoef(md[ok], mg[ok])[0, 1]
        return r, r / np.sqrt(max(rd, 1e-3) * max(rg, 1e-3)), rd, rg
    n = len(ids)
    r, rt, rd, rg = one(np.arange(n))
    bs = np.array([one(RNG.integers(0, n, n))[1] for _ in range(1000)])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(r), float(rt), float(lo), float(hi), float(rd), float(rg)


def partial_spearman(x, y, Z: pd.DataFrame):
    df = pd.concat([pd.Series(np.asarray(x), name="x", index=Z.index), pd.Series(np.asarray(y), name="y", index=Z.index), Z], axis=1).dropna()
    R = df.rank()
    zc = [c for c in R.columns if c not in ("x", "y")]
    A = np.c_[np.ones(len(R)), R[zc].values]
    rx = R.x - A @ np.linalg.lstsq(A, R.x, rcond=None)[0]
    ry = R.y - A @ np.linalg.lstsq(A, R.y, rcond=None)[0]
    r, pv = stats.pearsonr(rx, ry)
    return float(r), float(pv), int(len(R))


def r2(yy, pp):
    return 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


def fit_pred(tr, te, cols, y, w):
    A = np.c_[np.ones(len(tr)), tr[cols].values]
    sw = np.sqrt(tr[w].values)
    beta = np.linalg.lstsq(A * sw[:, None], tr[y].values * sw, rcond=None)[0]
    return np.c_[np.ones(len(te)), te[cols].values] @ beta


def loco(df: pd.DataFrame, base: list[str], extra: list[str], y: str, w: str = "w", perm: bool = True):
    """Leave-one-class-out (weighted LS fit, unweighted scoring). Returns pooled and fold-wise R2/RMSE, repeated-draw
    interval (bootstrap of the training classes), player-bootstrap CI and within-class permutation p for the gain."""
    classes = sorted(df.draft_year.unique())

    def run(d, boot_train=False):
        yv, pb, pa, cl = [], [], [], []
        for c in classes:
            tr, te = d[d.draft_year != c], d[d.draft_year == c]
            if boot_train:
                tr = tr.iloc[RNG.integers(0, len(tr), len(tr))]
            pb.append(fit_pred(tr, te, base, y, w))
            pa.append(fit_pred(tr, te, base + extra, y, w))
            yv.append(te[y].values)
            cl.append(np.full(len(te), c))
        return tuple(map(np.concatenate, (yv, pb, pa, cl)))
    yv, pb, pa, cl = run(df)
    out = dict(n=len(yv), r2_base=r2(yv, pb), r2_aug=r2(yv, pa))
    out["delta"] = out["r2_aug"] - out["r2_base"]
    out["folds"] = {int(c): dict(n=int((cl == c).sum()), r2_base=float(r2(yv[cl == c], pb[cl == c])),
                                 r2_aug=float(r2(yv[cl == c], pa[cl == c])),
                                 rmse_base=float(np.sqrt(((yv - pb)[cl == c] ** 2).mean())),
                                 rmse_aug=float(np.sqrt(((yv - pa)[cl == c] ** 2).mean()))) for c in classes}
    out["folds_pos"] = sum(f["r2_aug"] > f["r2_base"] for f in out["folds"].values())
    n = len(yv)
    bd = []
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        bd.append(r2(yv[i], pa[i]) - r2(yv[i], pb[i]))
    out["lo"], out["hi"] = np.percentile(bd, [2.5, 97.5])
    rep = []
    for _ in range(NREP):
        a, b_, c_, _ = run(df, boot_train=True)
        rep.append(r2(a, c_) - r2(a, b_))
    out["rep_med"], out["rep_lo"], out["rep_hi"] = np.percentile(rep, [50, 2.5, 97.5])
    if perm:
        null = []
        for _ in range(NPERM):
            dp = df.copy()
            for c in classes:
                m = (dp.draft_year == c).values
                dp.loc[m, extra] = dp.loc[m, extra].values[RNG.permutation(m.sum())]
            a, b_, c_, _ = run(dp)
            null.append(r2(a, c_) - r2(a, b_))
        out["perm_p"] = float((1 + np.sum(np.array(null) >= out["delta"])) / (1 + NPERM))
    return out


def holm_bh(pvals: dict):
    keys = list(pvals)
    p = np.array([pvals[k] for k in keys])
    m = len(p)
    o = np.argsort(p)
    holm = np.empty(m)
    run_max = 0
    for rank, i in enumerate(o):
        run_max = max(run_max, (m - rank) * p[i])
        holm[i] = min(1, run_max)
    bh = np.empty(m)
    prev = 1
    for rank in range(m - 1, -1, -1):
        i = o[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        bh[i] = prev
    return {k: dict(p=float(pvals[k]), holm=float(h), bh=float(b)) for k, h, b in zip(keys, holm, bh)}


# ----------------------------------------------------------------------------- main
def main() -> None:
    d = insights.load()
    p = d["p"].copy()
    D, G, PK, top40, t40_pairs, gp = build_curves(p)
    cb = [f"c{i}" for i in range(BURST)]
    cp = [f"c{i}" for i in range(PRECONTACT)]
    cc = [f"c{i}" for i in range(T)]
    tt = np.arange(T) * DT
    for X in (D, G):
        X["burst"] = X[cb].mean(axis=1)
        X["pre"] = X[cp].mean(axis=1)
    # Drill fixed effect: drill x Combine year
    D["z"] = D.groupby(["grp", "drill", "yr"]).burst.transform(zscore)
    D["zpre"] = D.groupby(["grp", "drill", "yr"]).pre.transform(zscore)
    # Game fixed effects: alignment (+ man/zone + press/off for WR, TE, DB); primary sample = random snaps
    GR = G[G.random_half].copy()
    PKR = PK[PK.random_half].copy()
    fe = {"DL": ["lined_up_position"], "OL": ["lined_up_position"], "WR": ["lined_up_position", "man", "press"],
          "TE": ["lined_up_position", "man", "press"], "DB": ["lined_up_position", "man", "press"]}
    for X in (GR, G):
        X["z"] = np.nan
        X["zpre"] = np.nan
        for g in GROUPS:
            m = X.grp == g
            X.loc[m, "z"] = residualize(X[m], "burst", fe[g])
            X.loc[m, "zpre"] = residualize(X[m], "pre", fe[g])
    for g in GROUPS:
        m = PKR.grp == g
        PKR.loc[m, "peak_z"] = residualize(PKR[m].dropna(subset=["peak"]), "peak", fe[g])

    put("n_players", len(p), "{:d}")
    put("n_game_plays", len(gp), "{:,}")
    put("n_drill_reps", len(D), "{:,}")
    put("n_game_curves_rand", len(GR), "{:,}")
    put("n_game_curves_all", len(G), "{:,}")

    # alignment share from season alignment table (DL: EDGE share)
    al = pd.read_parquet(PROC / "player_season_alignment.parquet")
    tot = al.groupby("nfl_id").plays.sum()
    edge = al[al.lined_up_position.isin(["EDGE", "OLB"])].groupby("nfl_id").plays.sum().reindex(tot.index).fillna(0)
    p = p.merge((edge / tot).rename("edge_share").reset_index(), on="nfl_id", how="left")

    # ------------------------------------------------------------ per-group translation (table 1, fig 3)
    rows, rel, EB = [], {}, {}
    for g in GROUPS:
        Dg, Gg = D[D.grp == g], GR[GR.grp == g]
        ebd, cd = eb_means(Dg, "z")
        ebg, cg = eb_means(Gg, "z")
        EB[g] = (ebd, ebg)
        rd, rg = sb(cd["r1"], cd["kbar"]), sb(cg["r1"], cg["kbar"])
        rel[g] = dict(r1_d=cd["r1"], r1_g=cg["r1"], k_d=cd["kbar"], k_g=cg["kbar"], rel_d=rd, rel_g=rg)
        ids = np.intersect1d(ebd.index, ebg.index)
        rho, lo, hi, n, pv = spearman_ci(ebd.raw.reindex(ids).values, ebg.raw.reindex(ids).values)
        sub = p[p.position_group == g].merge(top40[["nfl_id", "top40_z"]], on="nfl_id")
        pk = PKR[PKR.grp == g].groupby("nfl_id").peak_z.mean()
        sub = sub.merge(pk.rename("peak_z").reset_index(), on="nfl_id")
        ts, tlo, thi, tn, tp = spearman_ci(sub.top40_z.values, sub.peak_z.values)
        row = dict(group=g, n=n, rel_d=rd, rel_g=rg, r1_d=cd["r1"], r1_g=cg["r1"], k_d=cd["kbar"], k_g=cg["kbar"],
                   rho=rho, lo=lo, hi=hi, p=pv, ceiling=float(np.sqrt(max(rd, 0) * max(rg, 0))),
                   top=ts, top_lo=tlo, top_hi=thi, top_n=tn, top_p=tp)
        if g in BURST_GROUPS:
            put(f"reps07_d_{g}", reps_for(cd["r1"], 0.7), "{:.0f}")
            put(f"reps07_g_{g}", reps_for(cg["r1"], 0.7), "{:.0f}")
            Dz = Dg[["nfl_id", "z"]]
            Gz = Gg[["nfl_id", "z"]]
            r_, rt, tlo_, thi_, _, _ = true_corr(Dz, Gz, ids)
            row.update(true=rt, true_lo=tlo_, true_hi=thi_)
        rows.append(row)
        for kk, v in row.items():
            if kk == "group":
                continue
            fmt = "{:d}" if kk in ("n", "top_n") else ("{:.1f}" if kk in ("k_d", "k_g") else ("{:.3f}" if kk in ("p", "top_p") else "{:.2f}"))
            put(f"{kk}_{g}", v, fmt)
    T1 = pd.DataFrame(rows)
    # DL by alignment: EDGE vs interior (players' modal sampled alignment)
    ebd, ebg = EB["DL"]
    modal = GR[GR.grp == "DL"].groupby("nfl_id").lined_up_position.agg(lambda v: v.mode().iloc[0])
    for al_, lab in (("EDGE", "edge"), ("INTERIOR_LINE", "int")):
        ids = np.intersect1d(ebd.index, modal[modal == al_].index)
        rho, lo, hi, n, pv = spearman_ci(ebd.raw.reindex(ids).values, ebg.raw.reindex(ids).values)
        put(f"rho_{lab}", rho); put(f"lo_{lab}", lo); put(f"hi_{lab}", hi); put(f"n_{lab}", n, "{:d}"); put(f"p_{lab}", pv, "{:.3f}")
        rows.append(dict(group=f"DL-{lab}", rho=rho, lo=lo, hi=hi, n=n, p=pv))
    # pre-contact window (0-0.5 s): construct-mismatch check for DL
    ebdp, _ = eb_means(D[D.grp == "DL"], "zpre")
    ebgp, _ = eb_means(GR[GR.grp == "DL"], "zpre")
    ids = np.intersect1d(ebdp.index, ebgp.index)
    rho, lo, hi, n, pv = spearman_ci(ebdp.raw.reindex(ids).values, ebgp.raw.reindex(ids).values)
    put("rho_pre", rho); put("lo_pre", lo); put("hi_pre", hi); put("n_pre", n, "{:d}"); put("p_pre", pv, "{:.3f}")
    # all 12 snaps (sensitivity)
    Ga = G[G.grp == "DL"]
    _, eba = eb_means(D[D.grp == "DL"], "z")
    ebga, _ = eb_means(Ga, "z")
    ids = np.intersect1d(ebd.index, ebga.index)
    put("rho_all12", stats.spearmanr(ebd.raw.reindex(ids), ebga.raw.reindex(ids))[0])
    # 40 repeatability; reps needed
    t40 = np.vstack(t40_pairs.values)
    put("top40_retest", stats.pearsonr(t40[:, 0], t40[:, 1])[0])
    put("top40_retest_n", len(t40), "{:d}")
    rf = rel["DL"]
    put("reps07_drill_DL", reps_for(rf["r1_d"], 0.7), "{:.0f}")
    put("reps07_game_DL", reps_for(rf["r1_g"], 0.7), "{:.0f}")
    put("reps05_drill_DL", reps_for(rf["r1_d"], 0.5), "{:.0f}")
    put("rel_d3_DL", sb(rf["r1_d"], 3))

    # ------------------------------------------------------------ DL player table + outcome
    x = p[p.position_group == "DL"].copy()
    x = x.merge(ebd.add_prefix("d_"), left_on="nfl_id", right_index=True).merge(ebg.add_prefix("g_"), left_on="nfl_id", right_index=True)
    x["bt"] = x.g_true - x.d_true
    x["bt_raw"] = zscore(x.g_raw) - zscore(x.d_raw)
    x["d_rawz"] = zscore(x.d_raw)
    x["g_rawz"] = zscore(x.g_raw)
    x["is_edge"] = (x.edge_share >= 0.5).astype(float)
    # beta-binomial EB pressure rate, prior by EDGE vs interior (method of moments)
    x["eb_press"] = np.nan
    x["press_w"] = np.nan
    for e in (0.0, 1.0):
        m = (x.is_edge == e) & (x.pass_rush_snaps >= 1)
        n_, k_ = x.loc[m, "pass_rush_snaps"], x.loc[m, "pressures"]
        rate = k_ / n_
        mu = (k_.sum() / n_.sum())
        var_obs = np.average((rate - mu) ** 2, weights=n_)
        var_bin = mu * (1 - mu) * len(n_) / n_.sum()
        tau2 = max(var_obs - var_bin, 1e-5)
        ab = mu * (1 - mu) / tau2 - 1
        x.loc[m, "eb_press"] = (k_ + mu * ab) / (n_ + ab)
        x.loc[m, "press_w"] = n_ / (n_ + ab)
        put(f"prior_ab_{int(e)}", ab, "{:.0f}")
    x["y"] = 100 * x.eb_press
    x["w"] = x.press_w
    x["forty_z"] = x.groupby("draft_year").forty.transform(zscore).fillna(0.0)
    x["ngs_z"] = x.groupby("draft_year").ngs_athleticism_score.transform(zscore).fillna(0.0)
    x["weight_z"] = zscore(x.combine_weight)
    put("dl_n", len(x), "{:d}")
    xo = x.dropna(subset=["y"]).copy()
    put("dl_n_out", len(xo), "{:d}")
    put("dl_true_rush_snaps_med", xo.pass_rush_snaps.median(), "{:.0f}")

    # partial correlations (beyond pick, edge share, seasons)
    Z = xo[["log_pick", "edge_share", "seasons_available"]]
    pc = {}
    for lab, col in (("game", "g_true"), ("drill", "d_true"), ("bt", "bt")):
        r_, pv, n_ = partial_spearman(xo[col], xo.y, Z)
        pc[lab] = (r_, pv)
        put(f"pc_{lab}", r_); put(f"pc_{lab}_p", pv, "{:.3f}"); put("pc_n", n_, "{:d}")
    r_, pv, n_ = partial_spearman(xo.avg_get_off, xo.y, Z)
    put("pc_getoff", -r_); put("pc_getoff_p", pv, "{:.3f}")
    put("rho_game_getoff", stats.spearmanr(x.g_true, x.avg_get_off, nan_policy="omit")[0])
    put("rho_weight_getoff", stats.spearmanr(x.combine_weight, x.avg_get_off, nan_policy="omit")[0])

    # leave-one-class-out
    base = ["log_pick", "edge_share", "weight_z", "forty_z", "ngs_z"]
    lo_rows = {}
    xl = xo.dropna(subset=base + ["g_true", "d_true", "bt"]).copy()
    pick_only = loco(xl, ["log_pick"], ["edge_share"], "y", perm=False)
    put("r2_pick_only", pick_only["r2_base"]); put("r2_pick_edge", pick_only["r2_aug"])
    for lab, extra in (("drill", ["d_true"]), ("bt", ["bt"]), ("game", ["g_true"])):
        lo_rows[lab] = loco(xl, base, extra, "y")
    xg = xl.dropna(subset=["avg_get_off"])
    lo_rows["getoff"] = loco(xg, base, ["avg_get_off"], "y")
    lo_rows["game_over_getoff"] = loco(xg, base + ["avg_get_off"], ["g_true"], "y")
    for lab, r in lo_rows.items():
        for kk in ("delta", "lo", "hi", "rep_lo", "rep_hi", "rep_med", "r2_base", "r2_aug"):
            put(f"loco_{lab}_{kk}", r[kk], "{:.3f}")
        put(f"loco_{lab}_p", r["perm_p"], "{:.3f}")
        put(f"loco_{lab}_n", r["n"], "{:d}")
        put(f"loco_{lab}_folds_pos", r["folds_pos"], "{:d}")
        for c, f in r["folds"].items():
            for kk, v in f.items():
                put(f"loco_{lab}_{c}_{kk}", v, "{:d}" if kk == "n" else "{:.3f}")
    put("loco_base_r2", lo_rows["bt"]["r2_base"])
    put("loco_n", lo_rows["bt"]["n"], "{:d}")
    # sensitivity: hard 100-snap cut and raw rates
    xs = xl[xl.pass_rush_snaps >= 100].assign(y=lambda v: 100 * v.pressure_rate, w=1.0)
    s100 = loco(xs, base, ["bt"], "y", perm=False)
    put("loco_bt_100_delta", s100["delta"], "{:.3f}"); put("loco_bt_100_n", s100["n"], "{:d}")

    # multiplicity: every inferential test in the writeup
    pv = {f"burst_{g}": float(T1.set_index("group").loc[g, "p"]) for g in BURST_GROUPS}
    pv.update({f"top_{g}": float(T1.set_index("group").loc[g, "top_p"]) for g in GROUPS})
    pv.update({"burst_edge": float(NUM["p_edge"]), "burst_int": float(NUM["p_int"]), "burst_pre": float(NUM["p_pre"])})
    pv.update({f"pc_{k}": v[1] for k, v in pc.items()})
    pv.update({f"loco_{k}": lo_rows[k]["perm_p"] for k in ("drill", "bt", "game", "game_over_getoff")})
    MT = holm_bh(pv)
    put("n_tests", len(MT), "{:d}")
    for k, v in MT.items():
        put(f"holm_{k}", v["holm"], "{:.3f}"); put(f"bh_{k}", v["bh"], "{:.3f}")
    put("n_survive_bh", sum(v["bh"] < 0.05 for v in MT.values()), "{:d}")

    # quartiles of BT vs EB pressure
    xo["btq"] = pd.qcut(xo.bt, 4, labels=False)
    qrows = []
    for q in range(4):
        v = xo[xo.btq == q].y.values
        bs = [v[RNG.integers(0, len(v), len(v))].mean() for _ in range(NBOOT)]
        qrows.append((q, v.mean(), *np.percentile(bs, [2.5, 97.5]), len(v)))
        put(f"q{q + 1}_press", v.mean(), "{:.1f}"); put(f"q{q + 1}_lo", np.percentile(bs, 2.5), "{:.1f}"); put(f"q{q + 1}_hi", np.percentile(bs, 97.5), "{:.1f}")
    put("q_n", len(v), "{:d}")

    # ------------------------------------------------------------ ranked DL table
    xo = xo.sort_values("bt", ascending=False)
    rank_cols = ["display_name", "pick", "is_edge", "d_rawz", "d_true", "g_rawz", "g_true", "bt", "y", "pass_rush_snaps"]
    top5, bot5 = xo.head(5), xo.tail(5)
    rank_tab = pd.concat([top5, bot5])[rank_cols]
    tab_out = rank_tab.round(2).to_dict("records")
    for i, r in enumerate(rank_tab.itertuples()):
        put(f"rk{i}_dz", r.d_rawz, "{:+.1f}"); put(f"rk{i}_dt", r.d_true, "{:+.1f}")
        put(f"rk{i}_gz", r.g_rawz, "{:+.1f}"); put(f"rk{i}_gt", r.g_true, "{:+.1f}")
        put(f"rk{i}_bt", r.bt, "{:+.1f}"); put(f"rk{i}_y", r.y, "{:.1f}"); put(f"rk{i}_pick", int(r.pick), "{:d}")
        put(f"rk{i}_snaps", int(r.pass_rush_snaps), "{:d}")
    put("lam_d_med", xo.d_lam.median()); put("lam_g_med", xo.g_lam.median())
    put("sd_bt_raw", xo.bt_raw.std()); put("sd_bt_eb", xo.bt.std())

    # ------------------------------------------------------------ fig 1: hero player cards (2 x 3)
    hi_bt = list(xo.head(2).nfl_id)
    lo_bt = list(xo.tail(2).nfl_id)
    rest = xo[~xo.nfl_id.isin(hi_bt + lo_bt)].sort_values("y", ascending=False)
    hi_press = list(rest.head(2).nfl_id)
    cards = hi_bt + lo_bt + hi_press
    roles = ["Quicker on Sundays"] * 2 + ["Quicker in Indianapolis"] * 2 + ["Top pressure, other 4"] * 2
    dl_avg = D[D.grp == "DL"][cc].mean().values
    fig, axes = plt.subplots(2, 3, figsize=(FIGW, 7.6), sharey=True, sharex=True)
    card_out = []
    for ax, nid, role in zip(axes.ravel(), cards, roles):
        r = xo.set_index("nfl_id").loc[nid]
        dc = D[(D.nfl_id == nid)][cc]
        gc = GR[(GR.nfl_id == nid)][cc]
        ax.axvspan(0, 1.0, color=OI["yellow"], alpha=.18, lw=0)
        for _, row in gc.iterrows():
            ax.plot(tt, row.values, color=OI["blue"], alpha=.25, lw=1)
        ax.plot(tt, gc.mean().values, color=OI["blue"], lw=2.8, label="NFL pass rushes (mean; thin = each snap)")
        ax.plot(tt, dc.mean().values, "--", color=OI["red"], lw=2.8, label="his Combine pass-rush drill")
        ax.plot(tt, dl_avg, ":", color=OI["grey"], lw=1.8, label="DL drill average")
        ax.set_title(f"{r.display_name} ({'EDGE' if r.is_edge else 'interior'}, pick {int(r.pick)})\n{role}", fontsize=12)
        txt = (f"drill {r.d_rawz:+.1f} → {r.d_true:+.1f}\nNFL   {r.g_rawz:+.1f} → {r.g_true:+.1f}\n"
               f"BT {r.bt:+.1f}   pressure {r.y:.1f}%")
        ax.text(0.98, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5, family="monospace",
                bbox=dict(facecolor="white", edgecolor=OI["grey"], alpha=.9, boxstyle="round,pad=0.3"))
        ci = len(card_out)
        put(f"card{ci}_bt", r.bt, "{:+.1f}"); put(f"card{ci}_y", r.y, "{:.1f}"); put(f"card{ci}_pick", int(r.pick), "{:d}")
        put(f"card{ci}_gt", r.g_true, "{:+.1f}"); put(f"card{ci}_dt", r.d_true, "{:+.1f}")
        card_out.append(dict(name=r.display_name, role=role, pick=int(r.pick), d_raw=round(r.d_rawz, 2), d_true=round(r.d_true, 2),
                             g_raw=round(r.g_rawz, 2), g_true=round(r.g_true, 2), bt=round(r.bt, 2), press=round(r.y, 1)))
    for ax in axes[1]:
        ax.set_xlabel("seconds after first movement")
    for ax in axes[:, 0]:
        ax.set_ylabel("speed (yd/s)")
    axes[0, 0].legend(loc="upper left", fontsize=10, frameon=False)
    fig.suptitle("Six DL: Combine drill (dashed) vs NFL rushes (blue); box = raw z → shrunk z; yellow = 1-s burst window",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_player_cards.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ fig 2: population profiles (2 x 3)
    fig, axes = plt.subplots(2, 3, figsize=(FIGW, 7.0), sharey=True)
    pop = {}
    for ax, g in zip(axes.ravel(), GROUPS):
        dc = D[D.grp == g][cc]
        gc = GR[GR.grp == g][cc]
        ax.axvspan(0, 1.0, color=OI["yellow"], alpha=.18, lw=0)
        for c, lab, ls in ((dc, "Combine drill", "--"), (gc, "NFL snaps (random)", "-")):
            ax.plot(tt, c.mean().values, ls, color=GCOL[g], lw=2.6, label=lab)
            ax.fill_between(tt, c.quantile(.25).values, c.quantile(.75).values, color=GCOL[g], alpha=.14)
        ax.set_title(f"{g}: {DRILL_LABEL[g]} vs {GAME_TYPE[g]}", fontsize=11.5)
        ax.set_xlabel("s after first movement")
        pop[g] = (float(dc[cb].mean(axis=1).mean()), float(gc[cb].mean(axis=1).mean()))
        ax.legend(frameon=False, fontsize=10, loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("speed (yd/s)")
    ax = axes[1, 2]
    ax.axis("off")
    ax.text(0.02, 0.5, "Dashed = Combine drill\nSolid = NFL snaps (random draw)\nBand = middle 50% of reps\n"
            "Yellow = 1-s burst window\n\nDL flatten at contact;\nWR and TE curves cross\nthe drill after 1 s.",
            fontsize=11.5, va="center")
    fig.suptitle("Average speed from first movement: Combine drill vs NFL snaps", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_profiles.png", dpi=150)
    plt.close(fig)
    for g in GROUPS:
        put(f"pop_drill_{g}", pop[g][0]); put(f"pop_game_{g}", pop[g][1]); put(f"pop_ratio_{g}", 100 * pop[g][1] / pop[g][0], "{:.0f}")
    gcm = GR[GR.grp == "DL"][cc].mean().values
    dcm = D[D.grp == "DL"][cc].mean().values
    put("dl_game_peak_t", tt[int(np.argmax(gcm))], "{:.1f}")
    put("dl_game_peak_v", gcm.max()); put("dl_drill_v15", dcm[-1])

    # ------------------------------------------------------------ fig 3: group translation dot plot
    fig, ax = plt.subplots(figsize=(FIGW, 5.2))
    lab_rows = [("DL", "DL (all)"), ("DL-edge", "DL: EDGE"), ("DL-int", "DL: interior"), ("WR", "WR"), ("DB", "DB"),
                ("OL", "OL"), ("TE", "TE")]
    R = pd.DataFrame(rows).set_index("group")
    yp = np.arange(len(lab_rows))[::-1]
    for yv, (gk, lab) in zip(yp, lab_rows):
        base_g = gk.split("-")[0]
        col = GCOL[base_g]
        if gk in R.index and not np.isnan(R.loc[gk, "rho"]) and (gk not in ("OL", "TE")):
            r = R.loc[gk]
            ax.plot([r.lo, r.hi], [yv + .17] * 2, color=col, lw=3)
            ax.plot(r.rho, yv + .17, "o", color=col, ms=10, label="First-second burst, drill vs NFL" if gk == "DL" else None)
            if gk in BURST_GROUPS:
                ax.plot(r.ceiling, yv + .17, "|", color=OI["black"], ms=20, mew=3, label="Most a noisy pair could show" if gk == "DL" else None)
        if gk in GROUPS:
            r = R.loc[gk]
            ax.plot([r.top_lo, r.top_hi], [yv - .17] * 2, color=OI["grey"], lw=3)
            ax.plot(r.top, yv - .17, "s", color=OI["black"], ms=9, label="Top speed: 40 vs NFL peak" if gk == "DL" else None)
        if gk in ("OL", "TE"):
            ax.text(-0.98, yv + .17, "burst drill unreliable: not tested", fontsize=10, color=OI["grey"], va="center")
        n_ = int(R.loc[gk, "n"]) if gk in R.index else 0
        ax.text(1.01, yv, f"n={n_}", transform=ax.get_yaxis_transform(), fontsize=11, va="center")
    ax.axvline(0, color=OI["black"], lw=.8)
    ax.set_yticks(yp, [l for _, l in lab_rows])
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Spearman ρ across players (95% bootstrap CI)")
    ax.set_title("Burst ranks do not carry over from drill to NFL; the 40's top speed does for TE")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_translation.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ fig 4: DL scatter + BT quartile readout
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 5.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    sx = x.copy()
    for e, mk, lab in ((1.0, "o", "EDGE"), (0.0, "^", "interior")):
        s_ = sx[sx.is_edge == e]
        a1.scatter(s_.d_true, s_.g_true, marker=mk, s=46, color=OI["red"] if e else OI["sky"], edgecolor="k", lw=.4, label=lab, alpha=.85)
    lim = max(sx.d_true.abs().max(), sx.g_true.abs().max()) * 1.1
    a1.plot([-lim, lim], [-lim, lim], ":", color=OI["grey"])
    for r in xo.head(2).itertuples():
        a1.annotate(r.display_name, (r.d_true, r.g_true), xytext=(5, 5), textcoords="offset points", fontsize=10)
    for r in xo.tail(2).itertuples():
        a1.annotate(r.display_name, (r.d_true, r.g_true), xytext=(5, -12), textcoords="offset points", fontsize=10)
    a1.set_xlabel("Combine drill burst (shrunk, true-score z)")
    a1.set_ylabel("NFL rush burst (shrunk, true-score z)")
    a1.set_title(f"DL drill vs NFL burst: ρ = {T1.set_index('group').loc['DL', 'rho']:.2f}, n = {int(T1.set_index('group').loc['DL', 'n'])}")
    a1.legend(frameon=False, loc="upper left")
    for q, m_, lo_, hi_, n_ in qrows:
        a2.errorbar(q + 1, m_, yerr=[[m_ - lo_], [hi_ - m_]], fmt="o", color=OI["black"], ms=10, capsize=5, lw=2)
    a2.axhline(xo.y.mean(), color=OI["grey"], ls=":")
    a2.set_xticks([1, 2, 3, 4], ["Q1\n(drill > NFL)", "Q2", "Q3", "Q4\n(NFL > drill)"])
    a2.set_ylabel("shrunk career pressure rate (%)")
    a2.set_title(f"Pressure by BT quartile (n = {len(xo)})")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_dl_scatter.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ fig 5: forest plot of out-of-class gains
    fig, ax = plt.subplots(figsize=(FIGW, 4.6))
    flab = [("getoff", "NGS get-off (reference)"), ("drill", "+ Combine drill burst"), ("bt", "+ Burst Translation"),
            ("game", "+ NFL burst"), ("game_over_getoff", "+ NFL burst, after get-off")]
    ypf = np.arange(len(flab))[::-1]
    for yv, (k_, lab) in zip(ypf, flab):
        r = lo_rows[k_]
        ax.plot([r["rep_lo"], r["rep_hi"]], [yv] * 2, color=OI["blue"] if k_ != "getoff" else OI["grey"], lw=3)
        ax.plot(r["delta"], yv, "D", color=OI["black"], ms=9)
        for c, f in r["folds"].items():
            ax.plot(f["r2_aug"] - f["r2_base"], yv + .22, "|", color=OI["orange"], ms=12, mew=2.5)
        ax.text(1.01, yv, f"perm p={r['perm_p']:.2f}", transform=ax.get_yaxis_transform(), fontsize=11, va="center")
    ax.axvline(0, color=OI["black"], lw=1.2)
    ax.set_yticks(ypf, [l for _, l in flab])
    ax.set_xlabel("change in out-of-class R² for shrunk pressure rate vs pick + alignment + weight + 40 + NGS\n"
                  "(diamond = pooled; bar = 95% of 500 repeated draws; orange ticks = each held-out class)")
    ax.set_title("No burst measure beats the no-tracking baseline out of class")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_forest.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ top speed: positive control, class by class
    TS = []
    for g in GROUPS:
        sub = p[p.position_group == g].merge(top40[["nfl_id", "top40_z", "top_speed"]], on="nfl_id")
        pk = PKR[PKR.grp == g].groupby("nfl_id").agg(peak_z=("peak_z", "mean"), peak=("peak", "mean"), k=("peak", "size"))
        sub = sub.merge(pk.reset_index(), on="nfl_id").dropna(subset=["top40_z", "peak_z"])
        sub["forty_off_z"] = -sub.groupby("draft_year").forty.transform(zscore)
        TS.append(sub)
    TS = pd.concat(TS)
    te = TS[TS.position_group == "TE"].copy()
    for c in sorted(te.draft_year.unique()):
        s_ = te[te.draft_year == c]
        put(f"te_rho_{c}", stats.spearmanr(s_.top40_z, s_.peak_z)[0]); put(f"te_n_{c}", len(s_), "{:d}")
    # out-of-class: calibrate peak_z ~ top40_z on two classes, predict the third (pooled across all groups, group FE via z)
    def oc(df, col):
        ys, ps = [], []
        for c in sorted(df.draft_year.unique()):
            tr, tst = df[df.draft_year != c], df[df.draft_year == c]
            b = np.polyfit(tr[col], tr.peak_z, 1)
            ys.append(tst.peak_z.values); ps.append(np.polyval(b, tst[col].values))
        return r2(np.concatenate(ys), np.concatenate(ps))
    for g in ("TE", "DL"):
        s_ = TS[TS.position_group == g].dropna(subset=["forty_off_z"])
        put(f"oc_r2_top_{g}", oc(s_, "top40_z"), "{:.2f}"); put(f"oc_r2_forty_{g}", oc(s_, "forty_off_z"), "{:.2f}")
        put(f"rho_top_forty_{g}", stats.spearmanr(s_.top40_z, s_.forty_off_z)[0])
        put(f"oc_n_{g}", len(s_), "{:d}")
        reps = []
        for _ in range(NBOOT):
            i = RNG.integers(0, len(s_), len(s_))
            reps.append(oc(s_.iloc[i].reset_index(drop=True), "top40_z"))
        put(f"oc_r2_top_{g}_lo", np.percentile(reps, 2.5), "{:.2f}"); put(f"oc_r2_top_{g}_hi", np.percentile(reps, 97.5), "{:.2f}")
    te_named = te.sort_values("top_speed", ascending=False)
    te_pick = pd.concat([te_named.head(2), te_named.tail(2)])
    for i, r in enumerate(te_pick.itertuples()):
        put(f"te{i}_top", r.top_speed * 0.0 + r.top_speed, "{:.1f}")
        put(f"te{i}_peak", r.peak, "{:.1f}")
        put(f"te{i}_pick", int(r.pick), "{:d}")
    te_named_out = te_pick[["display_name", "pick", "top_speed", "peak", "peak_z", "top40_z"]].round(2).to_dict("records")

    fig, ax = plt.subplots(figsize=(FIGW, 5.0))
    mk = {2023: "o", 2024: "s", 2025: "^"}
    cl = {2023: OI["blue"], 2024: OI["orange"], 2025: OI["green"]}
    for c in sorted(te.draft_year.unique()):
        s_ = te[te.draft_year == c]
        ax.scatter(s_.top_speed, s_.peak, marker=mk[c], color=cl[c], s=70, edgecolor="k", lw=.4,
                   label=f"{c} class: ρ = {stats.spearmanr(s_.top40_z, s_.peak_z)[0]:.2f} (n = {len(s_)})")
    for r in te_pick.itertuples():
        ax.annotate(r.display_name, (r.top_speed, r.peak), xytext=(6, 4), textcoords="offset points", fontsize=10.5)
    ax.set_xlabel("Combine 40 top speed (yd/s, tracked)")
    ax.set_ylabel("mean NFL peak speed on random snaps (yd/s)")
    ax.set_title("TE: the 40's top speed reappears on NFL snaps in every class")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_te_topspeed.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ WR / DB named examples (shrunk BT extremes)
    named_wd = {}
    for g in ("WR", "DB"):
        ebd_, ebg_ = EB[g]
        s_ = p[p.position_group == g].merge(ebd_.add_prefix("d_"), left_on="nfl_id", right_index=True).merge(
            ebg_.add_prefix("g_"), left_on="nfl_id", right_index=True)
        s_["bt"] = s_.g_true - s_.d_true
        s_ = s_.sort_values("bt")
        named_wd[g] = s_[["display_name", "pick", "d_true", "g_true", "bt"]].iloc[[0, -1]].round(2).to_dict("records")
        for j, r in enumerate(s_.iloc[[0, -1]].itertuples()):
            put(f"{g}{j}_dt", r.d_true, "{:+.1f}"); put(f"{g}{j}_gt", r.g_true, "{:+.1f}"); put(f"{g}{j}_pick", int(r.pick), "{:d}")

    T1.round(3).to_json(HERE / "tables.json", orient="records", indent=1)
    out = dict(NUM)
    out["_abs_values"] = sorted({v.lstrip("-+") for v in NUM.values()})   # text writes negatives with a unicode minus
    out["_cards"] = card_out
    out["_rank_table"] = tab_out
    out["_te_named"] = te_named_out
    out["_wr_db_named"] = named_wd
    out["_multiplicity"] = MT
    out["_loco"] = {k: {kk: (vv if not isinstance(vv, np.floating) else float(vv)) for kk, vv in v.items()} for k, v in lo_rows.items()}
    (HERE / "numbers.json").write_text(json.dumps(out, indent=1, default=float))
    log(T1.round(2).to_string())
    log(json.dumps({k: v for k, v in NUM.items() if not k.startswith(("loco_", "holm_", "bh_", "rk", "q"))}, indent=0))
    log(json.dumps({k: v for k, v in NUM.items() if k.startswith(("loco_bt_", "loco_game_", "loco_drill_", "loco_getoff_")) and "_20" not in k}, indent=0))
    log(json.dumps(card_out, indent=0)); log(json.dumps(tab_out, indent=0)); log(json.dumps(te_named_out)); log(json.dumps(named_wd))
    log(json.dumps(MT, indent=0))


if __name__ == "__main__":
    main()
