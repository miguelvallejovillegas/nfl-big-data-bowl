"""Launch Carry: does a pass rusher's Combine acceleration show up in his NFL rushes, and does that matter?

Movement type: straight-line acceleration from a stance, 1-5 yd/s (the first two to three yards).
  Combine envelope : for each speed band (1-2, 2-3, 3-4, 4-5 yd/s), the 90th-percentile forward
                     acceleration over straight frames of ALL his Combine drills (combine_frames).
  NFL envelope     : the same statistic over the first 3 s after the snap of his random sampled
                     NFL snaps (game_frames_sample, sample_kind == 'random').
  Launch Carry     : median over bands of NFL / Combine (1.00 = shows his full Combine launch).
Group choice: five groups screened; the rule (written before any production model was fitted) is
"highest lower bound of the drill-to-NFL envelope rank correlation, among groups with an individual
production stat". Outcome for DL: career pressure rate (pressures per 100 pass-rush snaps, >=100 snaps).
Validation: leave-one-draft-class-out against pick-only, pick+role, full and ridge baselines,
class-stratified player bootstrap (refit every draw), within-class permutation placebo,
non-DL negative controls, and a mediation split (Combine envelope -> NFL envelope -> pressure).
Run: uv run python writeup/candidates/round_24/cand_3/analysis.py   (about 1 minute)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
D = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(2024)
NB = 1000
NPERM = 1000
BANDS = [1, 2, 3, 4]
# Okabe-Ito
C_EDGE, C_INT, C_GREY, C_HI, C_DARK = "#0072B2", "#E69F00", "#999999", "#D55E00", "#222222"
C_GREEN, C_PURP, C_SKY = "#009E73", "#CC79A7", "#56B4E9"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 13, "axes.titleweight": "bold"})
N: dict[str, str] = {}


def put(k, v, nd=2):
    if isinstance(v, (int, np.integer)) or nd == 0:
        N[k] = str(int(round(float(v))))
    else:
        N[k] = f"{float(v):.{nd}f}"
    if N[k] in ("-0", "-0.0", "-0.00"):
        N[k] = N[k][1:]
    return N[k]


def save(fig, name):
    fig.savefig(FIG / name, dpi=150)  # figsize width 10.667 in -> 1600 px
    plt.close(fig)


# ----------------------------------------------------------------------------- load
pl = pd.read_parquet(D / "players.parquet").set_index("nfl_id")
cr = pd.read_parquet(D / "combine_results.parquet").set_index("nfl_id")
ev = pd.read_parquet(D / "combine_events.parquet")
cf = pd.read_parquet(D / "combine_frames.parquet", columns=["event_id", "frame", "entity_type", "s", "dir"])
gp = pd.read_parquet(D / "game_plays_sample.parquet")
gf = pd.read_parquet(D / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "s", "dir"])
ss = pd.read_parquet(D / "player_season_stats.parquet")
grp = pl.position_group


def accel_frames(df, keys):
    df = df.sort_values(keys + ["frame"]).copy()
    g = df.groupby(keys, sort=False)
    df["al"] = (g.s.shift(-1) - g.s.shift(1)) / 0.2  # forward (tangential) acceleration, yd/s^2
    df["turn"] = ((g.dir.shift(-2) - g.dir.shift(2) + 180) % 360 - 180).abs()
    f = df[(df.turn < 20) & (df.al > 0) & (df.s >= 1) & (df.s < 5)].copy()
    f["b"] = np.floor(f.s).astype(int)
    return f


def envelope(f, key="nfl_id", minf=5):
    q = f.groupby([key, "b"]).al.agg(lambda x: np.quantile(x, 0.9) if len(x) >= minf else np.nan)
    return q.unstack("b").reindex(columns=BANDS)


# Combine side: every drill
cf = cf[cf.entity_type == "PLAYER"].merge(ev[["event_id", "nfl_id", "drill_name", "draft_year", "attempt_no"]])
cf["eidx"] = cf.groupby("nfl_id").event_id.transform(lambda x: pd.factorize(x)[0])
CFa = accel_frames(cf, ["event_id"])
CE = envelope(CFa)
CE_h = [envelope(CFa[CFa.eidx % 2 == h]) for h in (0, 1)]
put("n_combine_events", ev.event_id.nunique(), 0)
put("n_drills", ev.drill_name.nunique(), 0)

# NFL side: random snaps, first 3 s after the snap
rp = gp[(gp.sample_kind == "random") & gp.snap_frame.notna()]
gg = gf.merge(rp[["game_id", "play_id", "nfl_id", "snap_frame"]])
gg = gg[(gg.frame >= gg.snap_frame) & (gg.frame <= gg.snap_frame + 30)]
NFa = accel_frames(gg, ["game_id", "play_id", "nfl_id"])
NFa["pidx"] = NFa.groupby("nfl_id").play_id.transform(lambda x: pd.factorize(x)[0])
NE = envelope(NFa)
NE_h = [envelope(NFa[NFa.pidx % 2 == h], minf=3) for h in (0, 1)]
nsnaps = rp.groupby("nfl_id").play_id.nunique()

# role: share of sampled snaps at the group's outside spot
role_pos = {"DL": ["EDGE", "OLB"], "OL": ["T"], "WR": ["WR"], "TE": ["TE"], "DB": ["CB"]}
lp_ = gp[gp.sample_kind == "random"].copy()
lp_["grp"] = lp_.nfl_id.map(grp)
role = lp_.groupby("nfl_id").apply(lambda d: d.lined_up_position.isin(role_pos[d.grp.iloc[0]]).mean())


def zscore_in(df, cols, by):
    return df.groupby(by)[cols].transform(lambda x: (x - x.mean()) / x.std())


def env_score(E, ids, by_year=True):
    w = E.reindex(ids)
    if by_year:
        w = w.join(pl.draft_year)
        z = zscore_in(w, BANDS, "draft_year")
    else:
        z = (w - w.mean()) / w.std()
    return z[BANDS].mean(axis=1, skipna=True).where(z[BANDS].notna().sum(axis=1) >= 2)


def resid(y, X):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ b


def boot_rho(x, y, strat, nb=NB):
    x, y, strat = map(np.asarray, (x, y, strat))
    out = []
    for _ in range(nb):
        idx = np.concatenate([RNG.choice(np.where(strat == s)[0], (strat == s).sum()) for s in np.unique(strat)])
        out.append(stats.spearmanr(x[idx], y[idx])[0])
    return np.nanquantile(out, [0.05, 0.95])


# ----------------------------------------------------------------------------- five-group screen
screen = []
for G in ["DL", "OL", "WR", "TE", "DB"]:
    ids = [i for i in pl.index[grp == G] if i in CE.index and i in NE.index]
    cs, ns = env_score(CE, ids), env_score(NE, ids, by_year=False)
    d = pd.DataFrame({"cs": cs, "ns": ns}).dropna().join(pl[["draft_year", "combine_weight"]]).join(role.rename("role"))
    d["role"] = d.role.fillna(d.role.median())
    r_raw = stats.spearmanr(d.cs, d.ns)[0]
    lo, hi = boot_rho(d.cs, d.ns, d.draft_year)
    rc = resid(d.cs.values, [d.combine_weight, d.role])
    rn = resid(d.ns.values, [d.combine_weight, d.role])
    r_par = stats.spearmanr(rc, rn)[0]
    plo, phi = boot_rho(rc, rn, d.draft_year)
    # split-half reliability (Spearman-Brown) of each side
    def sh(Eh, by_year):
        a, b = env_score(Eh[0], d.index, by_year), env_score(Eh[1], d.index, by_year)
        m = a.notna() & b.notna()
        r = stats.spearmanr(a[m], b[m])[0]
        return 2 * r / (1 + r)
    relc, reln = sh(CE_h, True), sh(NE_h, False)
    screen.append(dict(G=G, n=len(d), r=r_raw, lo=lo, hi=hi, rp=r_par, plo=plo, phi=phi, relc=relc, reln=reln,
                       ceil=np.sqrt(max(relc, 0) * max(reln, 0))))
S = pd.DataFrame(screen).set_index("G")
for G, r in S.iterrows():
    for k in ["r", "lo", "hi", "rp", "plo", "phi", "relc", "reln", "ceil"]:
        put(f"scr_{G}_{k}", r[k])
    put(f"scr_{G}_n", r["n"], 0)
has_stat = ["DL", "OL", "WR", "TE"]  # DB: no individual coverage stat in player_season_stats
chosen = S.loc[has_stat, "lo"].idxmax()
assert chosen == "DL", chosen
N["chosen"] = chosen

# ----------------------------------------------------------------------------- DL table
ids = [i for i in pl.index[grp == "DL"] if i in CE.index and i in NE.index]
dl = pd.DataFrame({"cs": env_score(CE, ids), "ns": env_score(NE, ids, by_year=False)})
ratio = (NE.reindex(ids) / CE.reindex(ids))
dl["carry"] = ratio.median(axis=1, skipna=True).where(ratio.notna().sum(axis=1) >= 2)
dl = dl.dropna()
dl = dl.join(pl[["display_name", "draft_year", "draft_overall_pick", "combine_weight", "is_udfa"]])
dl = dl.join(cr[["forty", "three_cone", "ngs_athleticism_score"]]).join(role.rename("edge")).join(nsnaps.rename("nsnap"))
dl["edge"] = dl.edge.fillna(0)
dl["lp"] = np.log(dl.draft_overall_pick.fillna(262))
dl["carry_z"] = zscore_in(dl, ["carry"], "draft_year")["carry"]
# NFL envelope relative to what Combine envelope + weight + role predict = per-player translation residual
dl["trans"] = resid(dl.ns.values, [dl.cs, dl.combine_weight, dl.edge])
dl["trans"] = zscore_in(dl, ["trans"], "draft_year")["trans"]
a = ss[ss.position_group == "DL"].groupby("nfl_id")[["pressures", "pass_rush_snaps"]].sum()
dl = dl.join((100 * a.pressures / a.pass_rush_snaps).where(a.pass_rush_snaps >= 100).rename("pr"))
dl = dl.join(a.pass_rush_snaps.rename("prs"))
put("dl_n_env", len(dl), 0)
put("dl_n_snaps_med", dl.nsnap.median(), 0)
put("carry_med", dl.carry.median())
put("carry_p10", dl.carry.quantile(0.1))
put("carry_p90", dl.carry.quantile(0.9))
put("carry_edge", dl[dl.edge >= 0.5].carry.median())
put("carry_int", dl[dl.edge < 0.5].carry.median())
put("n_edge", (dl.edge >= 0.5).sum(), 0)
put("n_int", (dl.edge < 0.5).sum(), 0)
# reliability of Carry
rh = [(NE_h[h].reindex(dl.index) / CE_h[h].reindex(dl.index)).median(axis=1) for h in (0, 1)]
m = rh[0].notna() & rh[1].notna()
r = stats.spearmanr(rh[0][m], rh[1][m])[0]
put("carry_rel", 2 * r / (1 + r))
put("carry_rel_n", m.sum(), 0)
# how much of Carry is role
put("rho_carry_edge", stats.spearmanr(dl.carry, dl.edge)[0])
put("rho_ns_edge", stats.spearmanr(dl.ns, dl.edge)[0])
put("rho_cs_edge", stats.spearmanr(dl.cs, dl.edge)[0])
# within-role translation
for nm, sub in [("edge", dl[dl.edge >= 0.5]), ("int", dl[dl.edge < 0.5])]:
    rc = resid(sub.cs.values, [sub.combine_weight]); rn = resid(sub.ns.values, [sub.combine_weight])
    put(f"tr_{nm}", stats.spearmanr(rc, rn)[0])
    lo, hi = boot_rho(rc, rn, sub.draft_year)
    put(f"tr_{nm}_lo", lo); put(f"tr_{nm}_hi", hi)

D_ = dl.dropna(subset=["pr"]).copy()
D_["forty_f"] = D_.forty.fillna(D_.forty.median())
D_["ngs_f"] = D_.ngs_athleticism_score.fillna(D_.ngs_athleticism_score.median())
put("dl_n", len(D_), 0)
put("pr_med", D_.pr.median(), 1)
put("prs_med", D_.prs.median(), 0)
put("n_forty_missing", D_.forty.isna().sum(), 0)
put("share_day3", 100 * ((D_.draft_overall_pick > 100) | D_.draft_overall_pick.isna()).mean(), 0)
for k in ["cs", "ns"]:
    D_[k + "_z"] = zscore_in(D_, [k], "draft_year")[k]

BASES = {"pick": ["lp"], "pickrole": ["lp", "edge"], "full": ["lp", "edge", "combine_weight", "forty_f", "ngs_f"]}


def design(d, cols):
    X = d[cols].to_numpy(float)
    return X


def fit_pred(Xtr, ytr, Xte, alpha=0.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z, Zt = (Xtr - mu) / sd, (Xte - mu) / sd
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    P = np.eye(Z1.shape[1]) * alpha; P[0, 0] = 0
    b = np.linalg.solve(Z1.T @ Z1 + P, Z1.T @ (ytr - 0))
    return np.column_stack([np.ones(len(Zt)), Zt]) @ b


def ridge_alpha(d, cols, y):
    grid = [0, 1, 3, 10, 30, 100]
    best, ba = 1e18, 0
    yrs = d.draft_year.unique()
    for al in grid:
        sse = 0
        for yr in yrs:
            tr, te = d.draft_year != yr, d.draft_year == yr
            p = fit_pred(design(d[tr], cols), y[tr.values], design(d[te], cols), al)
            sse += ((y[te.values] - p) ** 2).sum()
        if sse < best:
            best, ba = sse, al
    return ba


def loco(d, base, extra=None, ridge=False):
    """Leave-one-class-out: returns held-out preds for base and base+extra, and the held-out slope of extra.
    Slope = coefficient of extra (per sd within class) fitted on training classes, averaged over folds
    weighted by test size; the R2 gain is computed on pooled held-out predictions."""
    y = d.pr.to_numpy(float)
    pb, pf, slopes, w = np.zeros(len(d)), np.zeros(len(d)), [], []
    for yr in sorted(d.draft_year.unique()):
        tr, te = (d.draft_year != yr).values, (d.draft_year == yr).values
        al = ridge_alpha(d[tr], base, y[tr]) if ridge else 0.0
        pb[te] = fit_pred(design(d[tr], base), y[tr], design(d[te], base), al)
        if extra is not None:
            cols = base + [extra]
            Xtr = design(d[tr], cols)
            pf[te] = fit_pred(Xtr, y[tr], design(d[te], cols), al)
            Z1 = np.column_stack([np.ones(tr.sum()), Xtr])
            b = np.linalg.lstsq(Z1, y[tr], rcond=None)[0]
            slopes.append(b[-1] * d[tr][extra].std()); w.append(te.sum())
    r2b = 1 - ((y - pb) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    if extra is None:
        return r2b
    r2f = 1 - ((y - pf) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return r2b, r2f, np.average(slopes, weights=w)


def boot_loco(d, base, extra, nb=NB, ridge=False):
    res = []
    yrs = d.draft_year.values
    for _ in range(nb):
        idx = np.concatenate([RNG.choice(np.where(yrs == y)[0], (yrs == y).sum()) for y in np.unique(yrs)])
        res.append(loco(d.iloc[idx].reset_index(drop=True), base, extra, ridge))
    res = np.array(res)
    return res


METRICS = {"carry_z": "Launch Carry", "cs_z": "Combine launch envelope", "ns_z": "NFL launch envelope",
           "trans": "Translation residual"}
rows = []
for base in ["pick", "pickrole", "full", "ridge"]:
    cols = BASES["full"] if base == "ridge" else BASES[base]
    put(f"r2_{base}", loco(D_, cols, None, ridge=False) if base != "ridge" else loco(D_, cols, "cs_z", True)[0])
    for mk in METRICS:
        nb = NB if base == "pickrole" else 300
        r2b, r2f, sl = loco(D_, cols, mk, ridge=(base == "ridge"))
        bt = boot_loco(D_, cols, mk, nb=nb, ridge=(base == "ridge"))
        rows.append(dict(base=base, m=mk, slope=sl, lo=np.quantile(bt[:, 2], 0.05), hi=np.quantile(bt[:, 2], 0.95),
                         gain=r2f - r2b, glo=np.quantile(bt[:, 1] - bt[:, 0], 0.05),
                         ghi=np.quantile(bt[:, 1] - bt[:, 0], 0.95), pneg=(bt[:, 2] <= 0).mean()))
R = pd.DataFrame(rows)
for _, r in R.iterrows():
    for k in ["slope", "lo", "hi", "gain", "glo", "ghi", "pneg"]:
        put(f"{r.m}_{r.base}_{k}", r[k], 3 if k in ("gain", "glo", "ghi") else 2)

# per-class slope for the headline (pick + role)
for yr in [2023, 2024, 2025]:
    s = D_[D_.draft_year == yr]
    for mk in ["carry_z", "cs_z"]:
        X = np.column_stack([np.ones(len(s)), s.lp, s.edge, s[mk]])
        put(f"{mk}_cls{yr}", np.linalg.lstsq(X, s.pr.values, rcond=None)[0][-1])
    put(f"n_cls{yr}", len(s), 0)

# ----------------------------------------------------------------------------- placebo: within-class permutation
perm = {"carry_z": [], "cs_z": []}
P_ = D_.copy()
for _ in range(NPERM):
    for mk in perm:
        P_[mk] = P_.groupby("draft_year")[mk].transform(lambda x: RNG.permutation(x.values))
        perm[mk].append(loco(P_, BASES["pickrole"], mk)[2])
        P_[mk] = D_[mk]
for mk in perm:
    pa = np.array(perm[mk])
    obs = R[(R.m == mk) & (R.base == "pickrole")].slope.iloc[0]
    put(f"perm_{mk}_lo", np.quantile(pa, 0.05)); put(f"perm_{mk}_hi", np.quantile(pa, 0.95))
    put(f"perm_{mk}_p", (np.abs(pa) >= abs(obs)).mean())
PERM = perm

# ----------------------------------------------------------------------------- negative controls: same Carry in other groups
ctrl = []
out_stats = {"OL": ("pressures_allowed", "pass_blocks", -1), "WR": ("rec_yards", "routes", 1), "TE": ("rec_yards", "routes", 1)}
for G, (num, den, sign) in out_stats.items():
    ids_g = [i for i in pl.index[grp == G] if i in CE.index and i in NE.index]
    rat = (NE.reindex(ids_g) / CE.reindex(ids_g))
    g_ = pd.DataFrame({"carry": rat.median(axis=1).where(rat.notna().sum(axis=1) >= 2)}).dropna()
    g_ = g_.join(pl[["draft_year", "draft_overall_pick"]]).join(role.rename("edge"))
    g_["edge"] = g_.edge.fillna(g_.edge.median())
    a_ = ss[ss.position_group == G].groupby("nfl_id")[[num, den]].sum()
    mult = 100 if G == "OL" else 1
    g_["pr"] = (mult * a_[num] / a_[den]).where(a_[den] >= 100)
    g_ = g_.dropna(subset=["pr"])
    g_["pr"] = (g_.pr - g_.pr.mean()) / g_.pr.std() * sign  # sd units, higher = better
    g_["lp"] = np.log(g_.draft_overall_pick.fillna(262))
    g_["carry_z"] = zscore_in(g_, ["carry"], "draft_year")["carry"]
    _, _, sl = loco(g_, ["lp", "edge"], "carry_z")
    bt = boot_loco(g_, ["lp", "edge"], "carry_z", nb=300)
    ctrl.append((G, len(g_), sl, np.quantile(bt[:, 2], 0.05), np.quantile(bt[:, 2], 0.95)))
# DL in sd units for comparison
sdpr = D_.pr.std()
put("pr_sd", sdpr)
for G, n, sl, lo, hi in ctrl:
    put(f"ctrl_{G}_n", n, 0); put(f"ctrl_{G}_slope", sl); put(f"ctrl_{G}_lo", lo); put(f"ctrl_{G}_hi", hi)
CTRL = ctrl

# ----------------------------------------------------------------------------- mediation: Combine envelope -> NFL envelope -> pressure
def mediation(d):
    X = np.column_stack([np.ones(len(d)), d.cs_z, d.lp, d.edge, d.combine_weight])
    a_ = np.linalg.lstsq(X, d.ns_z.values, rcond=None)[0][1]
    X2 = np.column_stack([X, d.ns_z])
    b2 = np.linalg.lstsq(X2, d.pr.values, rcond=None)[0]
    c = np.linalg.lstsq(X, d.pr.values, rcond=None)[0][1]
    return c, b2[1], a_, b2[-1], a_ * b2[-1]


med = mediation(D_)
bt = []
yrs = D_.draft_year.values
for _ in range(NB):
    idx = np.concatenate([RNG.choice(np.where(yrs == y)[0], (yrs == y).sum()) for y in np.unique(yrs)])
    bt.append(mediation(D_.iloc[idx]))
bt = np.array(bt)
for i, k in enumerate(["total", "direct", "a", "b", "indirect"]):
    put(f"med_{k}", med[i]); put(f"med_{k}_lo", np.quantile(bt[:, i], 0.05)); put(f"med_{k}_hi", np.quantile(bt[:, i], 0.95))
put("med_share", 100 * med[4] / med[0], 0)
MED = (med, bt)

# ----------------------------------------------------------------------------- what a pressure is worth (traced filter)
dlr = gp[(gp.sample_kind == "random") & gp.nfl_id.isin(pl.index[grp == "DL"])].copy()
dlr = dlr[dlr.expected_points_added.notna()]
dlr["pressured"] = dlr.time_to_pressure.notna() | (dlr.sack.fillna(0) > 0)
put("epa_n", len(dlr), 0)
put("epa_npress", dlr.pressured.sum(), 0)
gap = dlr[~dlr.pressured].expected_points_added.mean() - dlr[dlr.pressured].expected_points_added.mean()
bb = []
for _ in range(NB):
    s = dlr.sample(len(dlr), replace=True, random_state=int(RNG.integers(1e9)))
    bb.append(s[~s.pressured].expected_points_added.mean() - s[s.pressured].expected_points_added.mean())
put("epa_gap", gap); put("epa_gap_lo", np.quantile(bb, 0.05)); put("epa_gap_hi", np.quantile(bb, 0.95))

# ----------------------------------------------------------------------------- Monday rule: top third of Combine envelope in class
D_["cs_top"] = D_.groupby("draft_year").cs_z.transform(lambda x: x >= x.quantile(2 / 3))
X = np.column_stack([np.ones(len(D_)), D_.lp, D_.edge])
D_["fc"] = np.zeros(len(D_))
for yr in D_.draft_year.unique():
    tr, te = (D_.draft_year != yr).values, (D_.draft_year == yr).values
    b = np.linalg.lstsq(X[tr], D_.pr.values[tr], rcond=None)[0]
    D_.loc[te, "fc"] = X[te] @ b
D_["beat"] = D_.pr > D_.fc
top = D_[D_.cs_top]
put("rule_n", len(top), 0)
put("rule_beat", top.beat.sum(), 0)
put("rest_n", (~D_.cs_top).sum(), 0)
put("rest_beat", D_[~D_.cs_top].beat.sum(), 0)
put("rule_gap", (top.pr - top.fc).mean(), 1)
bb = []
for _ in range(NB):
    s = D_.sample(len(D_), replace=True, random_state=int(RNG.integers(1e9)))
    bb.append((s[s.cs_top].pr - s[s.cs_top].fc).mean() - (s[~s.cs_top].pr - s[~s.cs_top].fc).mean())
put("rule_diff", (top.pr - top.fc).mean() - (D_[~D_.cs_top].pr - D_[~D_.cs_top].fc).mean(), 1)
put("rule_diff_lo", np.quantile(bb, 0.05), 1); put("rule_diff_hi", np.quantile(bb, 0.95), 1)
put("rule_per_class", len(pl[(grp == "DL")]) / 3 / 3, 0)
put("rule_day3_share", 100 * ((top.draft_overall_pick > 100) | top.draft_overall_pick.isna()).mean(), 0)
put("rule_pick_med", top.draft_overall_pick.fillna(262).median(), 0)

# ----------------------------------------------------------------------------- named players
D_["carry_rank"] = D_.groupby("draft_year").carry.rank(ascending=False).astype(int)
D_["cs_rank"] = D_.groupby("draft_year").cs_z.rank(ascending=False).astype(int)
D_["n_class"] = D_.groupby("draft_year").carry.transform("size")
D_["gap_fc"] = D_.pr - D_.fc
# pick named cases: highest Carry, lowest Carry, and the top Combine envelope day-3 rusher
hi_c = D_.sort_values("carry").iloc[-1]
lo_c = D_.sort_values("carry").iloc[0]
d3pool = D_[(D_.draft_overall_pick.fillna(262) > 100) & D_.cs_top]
d3 = d3pool.sort_values("cs_z").iloc[-1]
d3hit = d3pool.sort_values("gap_fc").iloc[-1]
put("d3pool_n", len(d3pool), 0)
put("d3pool_beat", d3pool.beat.sum(), 0)
named = []
for tag, r in [("high", hi_c), ("low", lo_c), ("d3hit", d3hit), ("day3", d3)]:
    nid = r.name
    # hardest NFL launch: random snap with the highest mean forward accel 1-5 yd/s
    f = NFa[NFa.nfl_id == nid].groupby(["game_id", "play_id"]).al.mean().idxmax()
    pp = gp[(gp.game_id == f[0]) & (gp.play_id == f[1]) & (gp.nfl_id == nid)].iloc[0]
    clip = f"{pp.season} wk {pp.week}, Q{pp.quarter} {pp.game_clock}"
    named.append(dict(tag=tag, id=nid, name=r.display_name, year=int(r.draft_year),
                      pick="UDFA" if pd.isna(r.draft_overall_pick) else str(int(r.draft_overall_pick)),
                      role="edge" if r.edge >= 0.5 else "interior", carry=r.carry, crank=int(r.carry_rank),
                      csrank=int(r.cs_rank), ncls=int(r.n_class), pr=r.pr, fc=r.fc, clip=clip))
    put(f"nm_{tag}_carry", r.carry); put(f"nm_{tag}_pr", r.pr, 1); put(f"nm_{tag}_fc", r.fc, 1)
    N[f"nm_{tag}_name"] = r.display_name
NAMED = named

# Combine vs NFL envelope in physical units for the figure players
put("ce_med_b2", CE.reindex(D_.index)[2].median()); put("ne_med_b2", NE.reindex(D_.index)[2].median())

sl = float(N["cs_z_pickrole_slope"])
put("cs_per400", 4 * sl, 1)
put("cs_epa400", 4 * sl * gap, 1)
put("carry_pp_pick_minus_role", float(N["carry_z_pick_slope"]) - float(N["carry_z_pickrole_slope"]))
put("med_a_x100", 100 * med[2], 0)
put("carry_sd_units", sl_c := float(N["carry_z_pickrole_slope"]) / sdpr)
put("ns_edge_gap", dl[dl.edge >= 0.5].ns.mean() - dl[dl.edge < 0.5].ns.mean(), 1)

# ============================================================================= FIGURES
W = 1600 / 150

# Fig 1: hero, three rushers' launch envelopes
fig, axes = plt.subplots(1, 3, figsize=(W, 4.6), sharey=True)
mid = np.array(BANDS) + 0.5
for ax, nm in zip(axes, NAMED[:3]):
    ce, ne = CE.loc[nm["id"]], NE.loc[nm["id"]]
    ax.fill_between(mid, ne.values, ce.values, color=C_GREY, alpha=0.25, lw=0)
    ax.plot(mid, ce.values, "-o", color=C_DARK, lw=2.5, ms=7)
    ax.plot(mid, ne.values, "-s", color=C_EDGE if nm["role"] == "edge" else C_INT, lw=2.5, ms=7)
    colr = C_EDGE if nm["role"] == "edge" else C_INT
    ic = np.where(np.isfinite(ce.values))[0][-1]; i_n = np.where(np.isfinite(ne.values))[0][-1]
    yc, yn = ce.values[ic], ne.values[i_n]
    if abs(yc - yn) < 0.5:
        yc, yn = (yc + 0.25, yn - 0.25) if yc >= yn else (yc - 0.25, yn + 0.25)
    ax.text(mid[ic] + 0.15, yc, "Combine", color=C_DARK, fontsize=11, va="center")
    ax.text(mid[i_n] + 0.15, yn, "NFL", color=colr, fontsize=11, va="center")
    ax.set_title(f"{nm['name']} ({nm['role']}, pick {nm['pick']})", fontsize=12)
    ax.text(0.03, 0.97, f"Launch Carry {nm['carry']:.2f}\nclass rank {nm['crank']} of {nm['ncls']}\n"
            f"pressure {nm['pr']:.1f} vs {nm['fc']:.1f} forecast", transform=ax.transAxes, fontsize=10.5,
            va="top", bbox=dict(fc="white", ec=C_GREY))
    ax.set_xticks(mid, ["1-2", "2-3", "3-4", "4-5"])
    ax.set_xlim(1.2, 5.5)
    ax.set_xlabel("speed band (yd/s)")
axes[0].set_ylabel("hardest forward acceleration\n(90th pct, yd/s²)")
axes[0].set_ylim(0, 8.9)
fig.suptitle("Launch Carry = NFL rush acceleration / Combine drill acceleration at the same speed", fontsize=13,
             fontweight="bold")
fig.tight_layout()
save(fig, "fig1_launch_envelopes.png")

# Fig 2: translation is role. (A) scatter coloured by role; (B) group screen raw vs within-role
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 5.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
e = dl.edge >= 0.5
a1.scatter(dl.cs[e], dl.ns[e], color=C_EDGE, marker="o", s=38, alpha=0.85)
a1.scatter(dl.cs[~e], dl.ns[~e], color=C_INT, marker="^", s=42, alpha=0.85)
for sub, col in [(dl[e], C_EDGE), (dl[~e], C_INT)]:
    b = np.polyfit(sub.cs, sub.ns, 1); xx = np.linspace(sub.cs.min(), sub.cs.max(), 10)
    a1.plot(xx, np.polyval(b, xx), color=col, lw=2)
a1.text(0.02, 0.97, f"edge (n={N['n_edge']}): within-role ρ {N['tr_edge']}", color=C_EDGE, transform=a1.transAxes,
        va="top", fontsize=11)
a1.text(0.02, 0.90, f"interior (n={N['n_int']}): within-role ρ {N['tr_int']}", color=C_INT, transform=a1.transAxes,
        va="top", fontsize=11)
a1.text(0.02, 0.83, f"all {N['dl_n_env']} pooled: ρ {N['scr_DL_r']}", color=C_DARK, transform=a1.transAxes,
        va="top", fontsize=11)
a1.set_xlabel("Combine launch envelope (z within class)")
a1.set_ylabel("NFL launch envelope (z)")
a1.set_title("A. DL: the NFL launch sorts by alignment")
order = ["DL", "OL", "WR", "TE", "DB"]
yy = np.arange(len(order))[::-1]
for y_, G in zip(yy, order):
    r = S.loc[G]
    a2.plot([r.lo, r.hi], [y_ + 0.15] * 2, color=C_DARK, lw=2); a2.plot(r.r, y_ + 0.15, "o", color=C_DARK, ms=8)
    a2.plot([r.plo, r.phi], [y_ - 0.15] * 2, color=C_SKY, lw=2); a2.plot(r.rp, y_ - 0.15, "D", color=C_SKY, ms=7)
    a2.text(1.02, y_, f"n={int(r.n)}  ceiling {r.ceil:.2f}", va="center", fontsize=10, transform=a2.get_yaxis_transform())
a2.axvline(0, color=C_DARK, lw=1)
a2.set_yticks(yy, order)
a2.set_xlim(-0.5, 0.8)
a2.set_xlabel("drill-to-NFL rank correlation (90% interval)")
a2.text(0.02, 1.0, "● pooled   ", color=C_DARK, transform=a2.transAxes, fontsize=11, va="bottom")
a2.text(0.30, 1.0, "◆ weight and role held fixed", color=C_SKY, transform=a2.transAxes, fontsize=11, va="bottom")
a2.set_title("B. Five groups, same metric", pad=22)
fig.tight_layout()
fig.subplots_adjust(right=0.86)
save(fig, "fig2_translation_screen.png")

# Fig 3: held-out slopes with placebo and controls; mediation panel
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 5.0), gridspec_kw=dict(width_ratios=[1.3, 1]))
lab = [("carry_z", "Launch Carry"), ("trans", "Translation residual"), ("ns_z", "NFL launch envelope"),
       ("cs_z", "Combine launch envelope")]
yy = np.arange(len(lab))
for y_, (mk, nm) in zip(yy, lab):
    sub = R[(R.m == mk) & (R.base == "pickrole")].iloc[0]
    col = C_HI if mk == "cs_z" else C_DARK
    a1.plot([sub.lo, sub.hi], [y_, y_], color=col, lw=2.5); a1.plot(sub.slope, y_, "o", color=col, ms=9)
    a1.text(sub.hi + 0.08, y_, f"{sub.slope:+.2f}", va="center", color=col, fontsize=11)
    f_ = R[(R.m == mk) & (R.base == "full")].iloc[0]
    a1.plot(f_.slope, y_ - 0.22, "|", color=C_GREY, ms=14, mew=2.5)
pl_lo, pl_hi = float(N["perm_carry_z_lo"]), float(N["perm_carry_z_hi"])
a1.axvspan(pl_lo, pl_hi, color=C_GREY, alpha=0.2, lw=0)
a1.text((pl_lo + pl_hi) / 2, len(lab) - 0.62, "grey band: 90% of shuffled slopes", ha="center", va="bottom", fontsize=10, color="#444444")
a1.axvline(0, color=C_DARK, lw=1)
a1.set_yticks(yy, [x[1] for x in lab])
a1.set_ylim(-0.5, len(lab) - 0.25)
a1.set_xlabel("pressure-rate points per sd, unseen class\n(beyond pick + role; grey tick = full baseline)")
a1.set_title(f"A. Held out by draft class, n={N['dl_n']} rushers")
labs = ["total", "direct", "via NFL\nlaunch"]
vals = [MED[0][0], MED[0][1], MED[0][4]]
los = [np.quantile(MED[1][:, i], 0.05) for i in (0, 1, 4)]
his = [np.quantile(MED[1][:, i], 0.95) for i in (0, 1, 4)]
cols = [C_HI, C_PURP, C_GREEN]
for i in range(3):
    a2.bar(i, vals[i], color=cols[i], width=0.6)
    a2.plot([i, i], [los[i], his[i]], color=C_DARK, lw=2)
    a2.text(i + 0.33, vals[i], f"{vals[i]:+.2f}", ha="left", va="center", fontsize=11)
a2.axhline(0, color=C_DARK, lw=1)
a2.set_xticks(range(3), labs)
a2.set_xlim(-0.5, 2.85)
a2.set_ylabel("pressure-rate points per sd\nof Combine envelope")
a2.set_title("B. Where the Combine effect goes")
fig.tight_layout()
save(fig, "fig3_heldout_mediation.png")

# Fig 4: player grid, Combine envelope rank vs pressure over forecast, marker = Carry
fig, ax = plt.subplots(figsize=(W, 5.2))
D_["csr"] = D_.groupby("draft_year").cs_z.rank(pct=True)
sc = ax.scatter(D_.csr, D_.gap_fc, c=D_.carry, cmap="cividis", s=60, edgecolor=C_DARK, lw=0.5,
                marker="o")
ax.axvline(2 / 3, color=C_DARK, lw=1, ls="--")
ax.axhline(0, color=C_DARK, lw=1)
yb = ax.get_ylim()[0]
ax.text(0.68, yb + 0.2, f"top third: {N['rule_beat']} of {N['rule_n']}\nabove forecast", fontsize=11, color=C_DARK, va="bottom")
ax.text(0.30, yb + 0.2, f"rest: {N['rest_beat']} of {N['rest_n']} above forecast", fontsize=11, color=C_DARK, va="bottom")
for nm in NAMED:
    r = D_.loc[nm["id"]]
    ax.annotate(nm["name"], (r.csr, r.gap_fc), xytext=(-14 if r.csr > 0.9 else 8, -16 if r.gap_fc > 0 else (-26 if r.csr > 0.9 else 12)),
                ha="right" if r.csr > 0.9 else "left",
                textcoords="offset points", fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=C_GREY))
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label("Launch Carry")
ax.set_xlabel("Combine launch envelope, percentile within draft class")
ax.set_ylabel("career pressure rate minus\nheld-out pick + role forecast (pts)")
ax.set_title(f"Combine launch (x) leans with pressure over forecast; Launch Carry (colour) does not ({N['dl_n']} DL)",
             fontsize=12.5)
fig.tight_layout()
save(fig, "fig4_board.png")

# ============================================================================= numbers + text
N["table_named"] = "\n".join(
    f"| {n['name']} | {n['year']}, {n['pick']} | {n['role']} | {n['carry']:.2f} ({n['crank']}/{n['ncls']}) | "
    f"{n['csrank']}/{n['ncls']} | {n['pr']:.1f} / {n['fc']:.1f} | {n['clip']} |" for n in NAMED)
for n in NAMED:
    put(f"nm_{n['tag']}_carry", n["carry"]); put(f"nm_{n['tag']}_pr", n["pr"], 1); put(f"nm_{n['tag']}_fc", n["fc"], 1)
(OUT / "numbers.json").write_text(json.dumps(N, indent=1))
D_[["display_name", "draft_year", "draft_overall_pick", "edge", "carry", "cs_z", "ns_z", "pr", "fc"]].round(3).to_csv(
    OUT / "board.csv")

tpl = OUT / "writeup_template.md"
if tpl.exists():
    txt = tpl.read_text()
    miss = set(re.findall(r"\{(\w+)\}", txt)) - set(N)
    assert not miss, miss
    (OUT / "writeup.md").write_text(re.sub(r"\{(\w+)\}", lambda m: N[m.group(1)].replace("-", "−")
                                           if re.fullmatch(r"-?\d+(\.\d+)?", N[m.group(1)]) else N[m.group(1)], txt))
print(json.dumps({k: v for k, v in N.items() if not k.startswith("table")}, indent=0)[:6000])
