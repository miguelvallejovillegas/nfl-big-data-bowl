"""Launch Translation for DL: how much of a pass rusher's Combine first step shows up on random NFL rushes, and does
the drill-to-game match explain pressure beyond draft slot?

One feature function (features.py) runs on Combine drill reps and NFL snaps. Steps:
 1. Screen: 5 groups x 2 drill sources (matched position drill, 40) x 4 profile features, drill vs game player means
    on the 20 *random* NFL snaps per player; single-rep reliability both sides; best-of-k permutation p.
 2. DL deep dive on launch (peak acceleration in the first second after first movement): empirical-Bayes player
    means with posterior sd, true-score correlation, validation against NGS get-off.
 3. Per-player translation: Launch Translation gap (shrunk NFL z - shrunk drill z), Launch Retention % and
    profile-shape similarity; partial rho with pressure beyond a 3-covariate no-tracking baseline, best-of-5
    within-class permutation, leave-one-class-out dR2 with player bootstrap and training-set resampling.
 4. The chain: Combine launch -> NFL launch -> pressure.

Reads data/processed/*.parquet. Writes numbers.json, tables.json, figures/*.png next to this file.
Usage: uv run python writeup/candidates/round_03/cand_3/analysis.py
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
sys.path.insert(0, str(HERE))
import insights  # noqa: E402
from features import GAME_WIN, T_CURVE, rep_features  # noqa: E402

RNG = np.random.default_rng(2027)
NBOOT = 2000
NPERM = 1000
NREP = 300
MIN_SNAPS = 100
GROUPS = ["DL", "OL", "WR", "TE", "DB"]
FEATS = ["acc", "d1", "top", "bend"]
FEAT_LABEL = {"acc": "Launch\n(peak accel, 1st s)", "d1": "Burst\n(yards, 1st s)", "top": "Top speed", "bend": "Bend\n(lateral g)"}
MATCHED = {
    "DL": ["PASS_RUSH_DRILL"],
    "OL": ["PASS_RUSH_DROP", "PASS_PRO_MIRROR_DRILL"],
    "WR": ["GO_ROUTE_RIGHT", "SLANT_ROUTE_LEFT", "COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT",
           "SPEED_OUT_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT", "SLOT_SAIL_ROUTE_RIGHT", "SLOT_STRIKE_ROUTE_LEFT"],
    "TE": ["CORNER_ROUTE", "IN_ROUTE_LEFT", "HOOK_ROUTE_RIGHT", "FLAT_ROUTE_LEFT", "WHEEL_ROUTE_LEFT"],
    "DB": ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "LINE_DRILL"],
}
SOURCES = {"pos": "position drill", "forty": "40-yard dash"}
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
FIGW = 1600 / 150
NUM: dict = {}
TABLES: dict = {}


def put(key, v, nd=2):
    """Store a number exactly as printed in the text (plus |value|, since the checker drops the sign)."""
    if isinstance(v, (int, np.integer)):
        NUM[key] = int(v)
        return str(int(v))
    s = f"{float(v):.{nd}f}"
    NUM[key] = s
    if s.startswith("-"):
        NUM[key + "_abs"] = s[1:]
    return s


def save(fig, name):
    fig.set_size_inches(FIGW, fig.get_size_inches()[1])
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def zs(v: pd.Series) -> pd.Series:
    sd = v.std()
    return (v - v.mean()) / sd if sd and sd > 0 else v * np.nan


def resid(df: pd.DataFrame, col: str, fe: list[str]) -> np.ndarray:
    X = pd.get_dummies(df[fe].astype(str), drop_first=True).astype(float)
    A = np.c_[np.ones(len(df)), X.values]
    y = df[col].to_numpy(float)
    m = np.isfinite(y)
    r = np.full(len(y), np.nan)
    r[m] = y[m] - A[m] @ np.linalg.lstsq(A[m], y[m], rcond=None)[0]
    return r / np.nanstd(r)


def icc(ids, v):
    """One-way random-effects single-rep reliability (players with >= 2 reps)."""
    d = pd.DataFrame({"i": np.asarray(ids), "v": np.asarray(v, float)}).dropna()
    d = d[d.groupby("i").v.transform("size") >= 2]
    g = d.groupby("i").v
    n, N = g.ngroups, len(d)
    if n < 5:
        return np.nan, 0.0, 0.0
    kk = g.size()
    k0 = (N - (kk ** 2).sum() / N) / (n - 1)
    msb = (kk * (g.mean() - d.v.mean()) ** 2).sum() / (n - 1)
    msw = ((d.v - g.transform("mean")) ** 2).sum() / (N - n)
    tau2 = (msb - msw) / k0
    return float((msb - msw) / (msb + (k0 - 1) * msw)), float(tau2), float(msw)


def sb(r1, k):
    return k * r1 / (1 + (k - 1) * r1)


def reps_for(r1, target):
    return target * (1 - r1) / (r1 * (1 - target))


def sp_ci(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    r, p = stats.spearmanr(x, y)
    b = [stats.spearmanr(x[i], y[i])[0] for i in (RNG.integers(0, n, n) for _ in range(NBOOT))]
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return float(r), float(lo), float(hi), n, float(p)


def partial_sp(df, x, y, Z, boot=True):
    d = df[[x, y] + Z].dropna()
    R = d.rank()
    A = np.c_[np.ones(len(R)), R[Z].values]

    def one(i):
        Ai = A[i]
        bx = R[x].values[i] - Ai @ np.linalg.lstsq(Ai, R[x].values[i], rcond=None)[0]
        by = R[y].values[i] - Ai @ np.linalg.lstsq(Ai, R[y].values[i], rcond=None)[0]
        return float(np.corrcoef(bx, by)[0, 1])
    n = len(d)
    r = one(np.arange(n))
    t = r * np.sqrt((n - 2 - len(Z)) / (1 - r ** 2))
    p = float(2 * stats.t.sf(abs(t), n - 2 - len(Z)))
    if not boot:
        return r, np.nan, np.nan, n, p
    b = [one(RNG.integers(0, n, n)) for _ in range(NBOOT)]
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return r, float(lo), float(hi), n, p


def mde(n, k=0, power=0.8, alpha=0.05):
    return float(np.tanh((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(n - 3 - k)))


def power_at(r, n, k=0, alpha=0.05):
    se = 1 / np.sqrt(n - 3 - k)
    return float(stats.norm.sf(stats.norm.ppf(1 - alpha / 2) - np.arctanh(r) / se))


def r2(yy, pp):
    return 1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()


def fit_pred(tr, te, cols, y, w):
    A = np.c_[np.ones(len(tr)), tr[cols].values]
    sw = np.sqrt(tr[w].values)
    beta = np.linalg.lstsq(A * sw[:, None], tr[y].values * sw, rcond=None)[0]
    return np.c_[np.ones(len(te)), te[cols].values] @ beta


def loco(df, base, extra, y="y", w="w"):
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
    out["folds"] = {int(c): float(r2(yv[cl == c], pa[cl == c]) - r2(yv[cl == c], pb[cl == c])) for c in classes}
    out["folds_pos"] = int(sum(v > 0 for v in out["folds"].values()))
    n = len(yv)
    bd = []
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        bd.append(r2(yv[i], pa[i]) - r2(yv[i], pb[i]))
    out["lo"], out["hi"] = (float(v) for v in np.percentile(bd, [2.5, 97.5]))
    rep = []
    for _ in range(NREP):
        a, b_, c_, _ = run(df, boot_train=True)
        rep.append(r2(a, c_) - r2(a, b_))
    out["rep_lo"], out["rep_hi"] = (float(v) for v in np.percentile(rep, [2.5, 97.5]))
    return out


# ============================================================================= 1. features
def build():
    ev = pd.read_parquet(PROC / "combine_events.parquet")
    keep = ev.drill_name.isin(sum(MATCHED.values(), [])) | (ev.drill_type == "FORTY_YARD_DASH")
    ev = ev[keep & ev.position_group.isin(GROUPS)]
    cf = pd.read_parquet(PROC / "combine_frames.parquet", columns=["event_id", "frame", "entity_type", "x", "y", "s"])
    cf = cf[(cf.entity_type == "PLAYER") & cf.event_id.isin(ev.event_id)].sort_values(["event_id", "frame"])
    rows = []
    for eid, g in cf.groupby("event_id", sort=False):
        f = rep_features(g.x.values, g.y.values, g.s.values)
        if f is None:
            continue
        c = f.pop("_curve")
        rows.append({"event_id": eid, **f, **({f"c{i}": v for i, v in enumerate(c)} if c is not None else {})})
    D = pd.DataFrame(rows).merge(ev[["event_id", "nfl_id", "draft_year", "drill_name", "drill_type", "attempt_no",
                                     "position_group"]], on="event_id")
    D["src"] = np.where(D.drill_type == "FORTY_YARD_DASH", "forty", "pos")
    gp = pd.read_parquet(PROC / "game_plays_sample.parquet")
    gf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "x", "y", "s"])
    gf = gf.sort_values(["game_id", "play_id", "nfl_id", "frame"])
    k = ["game_id", "play_id", "nfl_id"]
    snap = gp.set_index(k).snap_frame
    rows = []
    for key, g in gf.groupby(k, sort=False):
        sf = snap.get(key, np.nan)
        if not np.isfinite(sf):
            continue
        st = int(np.searchsorted(g.frame.values, sf))
        f = rep_features(g.x.values, g.y.values, g.s.values, st, st + GAME_WIN)
        if f is None:
            continue
        c = f.pop("_curve")
        rows.append({**dict(zip(k, key)), **f, **({f"c{i}": v for i, v in enumerate(c)} if c is not None else {})})
    G = pd.DataFrame(rows).merge(gp, on=k)
    return D, G, gp


def main():
    d = insights.load()
    p = d["p"].copy()
    D, G, gp = build()
    G = G.merge(p[["nfl_id", "position_group"]], on="nfl_id")
    put("n_players", len(p))
    put("n_game_plays", len(gp))
    put("n_drill_reps", len(D))
    put("n_game_reps", len(G))
    GR = G[G.sample_kind == "random"].copy()
    put("n_game_random", len(GR))
    put("rand_per_player_DL", GR[GR.position_group == "DL"].groupby("nfl_id").size().median(), 0)
    # drill z within drill x Combine year; game z residualized on alignment, separately for random and top-EPA halves
    for f in FEATS:
        D[f + "_z"] = D.groupby(["drill_name", "draft_year"])[f].transform(zs)
    GT = G[G.sample_kind == "top_epa"].copy()
    for X in (GR, GT):
        for f in FEATS:
            X[f + "_z"] = np.nan
            for g in GROUPS:
                m = (X.position_group == g).values
                X.loc[m, f + "_z"] = resid(X[m], f, ["lined_up_position"])

    # ------------------------------------------------------------------ screen (fig 2)
    cells, means = [], {}
    for g in GROUPS:
        for src in SOURCES:
            Dg = D[(D.position_group == g) & (D.src == src) & ((D.src == "forty") | D.drill_name.isin(MATCHED[g]))]
            Gg = GR[GR.position_group == g]
            for f in FEATS:
                md, mg = Dg.groupby("nfl_id")[f + "_z"].mean(), Gg.groupby("nfl_id")[f + "_z"].mean()
                ids = md.dropna().index.intersection(mg.dropna().index)
                r, pv = stats.spearmanr(md[ids], mg[ids])
                r1d, _, _ = icc(Dg.nfl_id, Dg[f + "_z"])
                r1g, _, _ = icc(Gg.nfl_id, Gg[f + "_z"])
                kd = Dg.groupby("nfl_id")[f + "_z"].count().reindex(ids).mean()
                kg = Gg.groupby("nfl_id")[f + "_z"].count().reindex(ids).mean()
                Rd = sb(r1d, kd) if np.isfinite(r1d) else np.nan
                Rg = sb(r1g, kg)
                ceil = float(np.sqrt(max(Rd, 0) * max(Rg, 0))) if np.isfinite(Rd) else np.nan
                cells.append(dict(group=g, src=src, feat=f, rho=float(r), p=float(pv), n=len(ids), r1d=r1d, r1g=r1g,
                                  kd=kd, kg=kg, Rd=Rd, Rg=Rg, ceil=ceil))
                means[(g, src, f)] = (md[ids].values, mg[ids].values)
    C = pd.DataFrame(cells)
    put("n_cells", len(C))
    # best-of-k permutation: shuffle game means across players within group, take max rho over all cells
    null = []
    for _ in range(NPERM):
        mx = -1
        perm_cache = {}
        for (g, src, f), (a, b) in means.items():
            mx = max(mx, stats.spearmanr(a, RNG.permutation(b))[0])
        null.append(mx)
    null = np.array(null)
    C["p_best"] = [(1 + (null >= r).sum()) / (1 + NPERM) for r in C.rho]
    put("null_max_med", np.median(null))
    put("null_max_95", np.percentile(null, 95))
    for r in C.itertuples():
        key = f"{r.group}_{r.src}_{r.feat}"
        put(f"rho_{key}", r.rho); put(f"p_{key}", r.p, 3); put(f"n_{key}", int(r.n)); put(f"pbest_{key}", r.p_best, 3)
        put(f"r1d_{key}", r.r1d); put(f"r1g_{key}", r.r1g); put(f"ceil_{key}", r.ceil); put(f"Rg_{key}", r.Rg)
    put("n_cells_p05", int((C.p < 0.05).sum()))
    put("n_cells_pbest05", int((C.p_best < 0.05).sum()))
    put("n_cells_exp05", len(C) * 0.05, 1)
    TABLES["screen"] = C.round(3).to_dict("records")

    # ------------------------------------------------------------------ DL deep dive
    dl = p[p.position_group == "DL"].copy()
    al = pd.read_parquet(PROC / "player_season_alignment.parquet")
    tot = al.groupby("nfl_id").plays.sum()
    edge = al[al.lined_up_position.isin(["EDGE", "OLB"])].groupby("nfl_id").plays.sum().reindex(tot.index).fillna(0)
    dl = dl.merge((edge / tot).rename("edge_share").reset_index(), on="nfl_id", how="left")
    dl["edge_share"] = dl.edge_share.fillna(dl.nfl_position.isin(["DE", "OLB"]).astype(float))
    dl["role"] = np.where(dl.edge_share >= 0.5, "EDGE", "interior")
    put("n_dl", len(dl))
    Dd = D[(D.position_group == "DL") & (D.drill_name == "PASS_RUSH_DRILL")].dropna(subset=["acc_z"])
    Df = D[(D.position_group == "DL") & (D.src == "forty")].dropna(subset=["acc_z"])
    Gd = GR[GR.position_group == "DL"].dropna(subset=["acc_z"])
    put("n_dl_drill_reps", len(Dd)); put("n_dl_forty_reps", len(Df)); put("n_dl_game_reps", len(Gd))

    def eb(R, col="acc_z", raw="acc"):
        r1, tau2, s2 = icc(R.nfl_id, R[col])
        g = R.groupby("nfl_id").agg(m=(col, "mean"), k=(col, "size"), raw=(raw, "mean"))
        grand, graw = R[col].mean(), R[raw].mean()
        lam = tau2 / (tau2 + s2 / g.k)
        g["true"] = lam * (g.m - grand) / np.sqrt(tau2)          # posterior mean, true-score sd units
        g["sd"] = np.sqrt(1 - lam)                               # posterior sd, same units
        sd_raw = R[raw].std()
        g["raw_shr"] = graw + lam * (g.raw - graw)               # shrunk yd/s^2
        g["lam"] = lam
        return g, dict(r1=r1, tau2=tau2, s2=s2, k=float(g.k.mean()), rel=sb(r1, g.k.mean()), sd_raw=sd_raw)
    ebd, cd = eb(Dd)
    ebf, cf_ = eb(Df)
    ebg, cg = eb(Gd)
    for lab, c in (("drill", cd), ("forty", cf_), ("game", cg)):
        put(f"r1_{lab}", c["r1"]); put(f"k_{lab}", c["k"], 1); put(f"rel_{lab}", c["rel"])
    put("reps07_drill", reps_for(cd["r1"], 0.7), 0); put("reps07_game", reps_for(cg["r1"], 0.7), 0)
    put("rel_game_40", sb(cg["r1"], 40)); put("rel_game_6", sb(cg["r1"], 6))
    put("med_acc_drill", Dd.acc.median(), 1); put("med_acc_game", Gd.acc.median(), 1); put("med_acc_forty", Df.acc.median(), 1)
    put("med_d1_drill", Dd.d1.median()); put("med_d1_game", Gd.d1.median())
    put("game_over_drill_pct", 100 * Gd.acc.median() / Dd.acc.median(), 0)
    dl = dl.merge(ebd.add_prefix("d_"), left_on="nfl_id", right_index=True, how="left")
    dl = dl.merge(ebf.add_prefix("f_"), left_on="nfl_id", right_index=True, how="left")
    dl = dl.merge(ebg.add_prefix("g_"), left_on="nfl_id", right_index=True, how="left")
    # validation of the game-side metric against NGS get-off (lower = faster)
    v = GR[(GR.position_group == "DL")].dropna(subset=["acc", "player_get_off"])
    r, pv = stats.spearmanr(v.acc, v.player_get_off)
    put("val_play_rho", r); put("val_play_n", len(v))
    r, pv = stats.spearmanr(v.d1, v.player_get_off)
    put("val_play_rho_d1", r)
    vv = dl.dropna(subset=["g_m", "avg_get_off"])
    put("val_player_rho", stats.spearmanr(vv.g_m, vv.avg_get_off)[0]); put("val_player_n", len(vv))

    # true-score correlation drill -> game (and 40 -> game), player bootstrap recomputes reliabilities
    def true_corr(R1, R2, ids):
        a1 = R1[R1.nfl_id.isin(ids)][["nfl_id", "acc_z"]]
        a2 = R2[R2.nfl_id.isin(ids)][["nfl_id", "acc_z"]]
        g1, g2 = dict(tuple(a1.groupby("nfl_id").acc_z)), dict(tuple(a2.groupby("nfl_id").acc_z))
        ids = np.array([i for i in ids if i in g1 and i in g2])

        def one(sel):
            rr1 = pd.concat([pd.DataFrame({"i": j, "v": g1[i].values}) for j, i in enumerate(sel)])
            rr2 = pd.concat([pd.DataFrame({"i": j, "v": g2[i].values}) for j, i in enumerate(sel)])
            m1, m2 = rr1.groupby("i").v.mean(), rr2.groupby("i").v.mean()
            q1, _, _ = icc(rr1.i, rr1.v)
            q2, _, _ = icc(rr2.i, rr2.v)
            R1_, R2_ = sb(q1, rr1.groupby("i").size().mean()), sb(q2, rr2.groupby("i").size().mean())
            r = np.corrcoef(m1, m2)[0, 1]
            return r, r / np.sqrt(max(R1_, 0.05) * max(R2_, 0.05))
        r, rt = one(ids)
        bs = np.array([one(RNG.choice(ids, len(ids)))[1] for _ in range(500)])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        return float(r), float(rt), float(lo), float(hi), len(ids)
    ids_dg = np.intersect1d(ebd.index, ebg.index)
    r, rt, lo, hi, n = true_corr(Dd, Gd, ids_dg)
    put("true_dg", rt); put("true_dg_lo", lo); put("true_dg_hi", hi); put("pear_dg", r)
    ids_fg = np.intersect1d(ebf.index, ebg.index)
    r, rt, lo, hi, n = true_corr(Df, Gd, ids_fg)
    put("true_fg", rt); put("true_fg_lo", lo); put("true_fg_hi", hi)
    # Spearman with CI for the two Combine sources
    rr = sp_ci(dl.d_m, dl.g_m); put("sp_dg", rr[0]); put("sp_dg_lo", rr[1]); put("sp_dg_hi", rr[2]); put("sp_dg_n", rr[3]); put("sp_dg_p", rr[4], 3)
    rr = sp_ci(dl.f_m, dl.g_m); put("sp_fg", rr[0]); put("sp_fg_lo", rr[1]); put("sp_fg_hi", rr[2]); put("sp_fg_n", rr[3]); put("sp_fg_p", rr[4], 3)
    put("mde_dg", mde(int(NUM["sp_dg_n"])))
    # top-EPA half replication of drill -> game launch
    tg = GT[GT.position_group == "DL"].groupby("nfl_id").acc_z.mean()
    ii = dl.set_index("nfl_id").d_m.dropna().index.intersection(tg.dropna().index)
    put("sp_dg_top", stats.spearmanr(dl.set_index("nfl_id").d_m[ii], tg[ii])[0]); put("sp_dg_top_n", len(ii))
    ii = dl.set_index("nfl_id").f_m.dropna().index.intersection(tg.dropna().index)
    put("sp_fg_top", stats.spearmanr(dl.set_index("nfl_id").f_m[ii], tg[ii])[0])
    # what the stopwatch already has: official 40 and 10-yd split vs NFL launch; tracked 40 launch beyond them
    for c in ["forty", "ten_yd_split", "combine_weight"]:
        dl[c + "_z"] = dl.groupby("draft_year")[c].transform(zs) if c != "combine_weight" else zs(dl[c])
    rr = sp_ci(-dl.forty_z, dl.g_m); put("sp_40time_g", rr[0]); put("sp_40time_g_n", rr[3])
    rr = sp_ci(-dl.ten_yd_split_z, dl.g_m); put("sp_10yd_g", rr[0]); put("sp_10yd_g_n", rr[3])
    rr = sp_ci(-dl.combine_weight_z, dl.g_m); put("sp_wt_g", rr[0])
    pr_ = partial_sp(dl, "f_m", "g_m", ["forty_z", "combine_weight_z"]); put("pc_fg_40wt", pr_[0]); put("pc_fg_40wt_lo", pr_[1]); put("pc_fg_40wt_hi", pr_[2]); put("pc_fg_40wt_n", pr_[3])
    pr_ = partial_sp(dl, "d_m", "g_m", ["forty_z", "combine_weight_z"]); put("pc_dg_40wt", pr_[0]); put("pc_dg_40wt_lo", pr_[1]); put("pc_dg_40wt_hi", pr_[2])
    # EDGE vs interior translation
    for role in ("EDGE", "interior"):
        s_ = dl[dl.role == role]
        rr = sp_ci(s_.d_m, s_.g_m); put(f"sp_dg_{role}", rr[0]); put(f"sp_dg_{role}_lo", rr[1]); put(f"sp_dg_{role}_hi", rr[2]); put(f"sp_dg_{role}_n", rr[3])

    # ------------------------------------------------------------------ per-player translation metrics
    dl["ltg"] = dl.g_true - dl.d_true                                        # Launch Translation gap
    dl["ltg_sd"] = np.sqrt(dl.g_sd ** 2 + dl.d_sd ** 2)
    dl["retention"] = 100 * dl.g_raw_shr / dl.d_raw_shr                     # Launch Retention %
    cc = [f"c{i}" for i in range(11)]
    dcur = Dd.dropna(subset=cc).groupby("nfl_id")[cc].mean()
    gcur = Gd.dropna(subset=cc).groupby("nfl_id")[cc].mean()
    ii = dcur.index.intersection(gcur.index)
    sim = pd.Series([np.corrcoef(np.diff(dcur.loc[i].values), np.diff(gcur.loc[i].values))[0, 1] for i in ii], index=ii)
    dl = dl.merge(sim.rename("sim").reset_index(), on="nfl_id", how="left")
    put("ret_med", dl.retention.median(), 0); put("ret_p10", dl.retention.quantile(0.1), 0); put("ret_p90", dl.retention.quantile(0.9), 0)
    put("lt_sd_spread", dl.ltg.std()); put("lt_post_sd_med", dl.ltg_sd.median())
    put("sim_med", dl.sim.median())
    put("n_lt", int(dl.ltg.notna().sum()))
    put("lam_drill_1rep", float(dl.loc[dl.d_k == 1, "d_lam"].median())); put("lam_drill_2rep", float(dl.loc[dl.d_k == 2, "d_lam"].median()))
    put("lam_game_med", float(dl.g_lam.median()))

    # pressure: beta-binomial EB by role
    dl["y"] = np.nan
    dl["w"] = np.nan
    for role in ("EDGE", "interior"):
        m = (dl.role == role) & (dl.pass_rush_snaps >= 1)
        n_, k_ = dl.loc[m, "pass_rush_snaps"], dl.loc[m, "pressures"]
        mu = k_.sum() / n_.sum()
        var_obs = np.average((k_ / n_ - mu) ** 2, weights=n_)
        tau2 = max(var_obs - mu * (1 - mu) * len(n_) / n_.sum(), 1e-5)
        ab = mu * (1 - mu) / tau2 - 1
        dl.loc[m, "y"] = 100 * (k_ + mu * ab) / (n_ + ab)
        dl.loc[m, "w"] = n_ / (n_ + ab)
        put(f"prior_ab_{role}", ab, 0)
    co = dl[(dl.pass_rush_snaps >= MIN_SNAPS)].dropna(subset=["y", "g_m"]).copy()
    put("n_cohort", len(co))
    put("cohort_snaps_med", co.pass_rush_snaps.median(), 0)
    co["neg_log_pick"] = -co.log_pick
    BASE = ["log_pick", "combine_weight_z", "edge_share"]
    METRICS = [("ltg", "Launch Translation gap"), ("sim", "Profile similarity"), ("g_true", "NFL launch (level)"),
               ("d_true", "Combine pass-rush-drill launch"), ("f_true", "Combine 40 launch")]
    res = {}
    for col, lab in METRICS:
        r_, lo, hi, n, pv = partial_sp(co, col, "y", BASE)
        rr = sp_ci(co[col], co.y)
        res[col] = dict(label=lab, pr=r_, lo=lo, hi=hi, n=n, p=pv, raw=rr[0], raw_lo=rr[1], raw_hi=rr[2])
        put(f"pr_{col}", r_); put(f"pr_{col}_lo", lo); put(f"pr_{col}_hi", hi); put(f"pr_{col}_n", n); put(f"pr_{col}_p", pv, 3)
        put(f"raw_{col}", rr[0]); put(f"raw_{col}_lo", rr[1]); put(f"raw_{col}_hi", rr[2]); put(f"raw_{col}_n", rr[3])
    # best-of-5 within-class permutation of the max partial rho
    cm = co.dropna(subset=[c for c, _ in METRICS[:4]]).copy()
    put("n_perm_set", len(cm))
    obs = {c: partial_sp(cm, c, "y", BASE, boot=False)[0] for c, _ in METRICS[:4]}
    nul = []
    for _ in range(NPERM):
        cp = cm.copy()
        for yr in cp.draft_year.unique():
            m = (cp.draft_year == yr).values
            cp.loc[m, "y"] = cp.loc[m, "y"].values[RNG.permutation(m.sum())]
        nul.append(max(abs(partial_sp(cp, c, "y", BASE, boot=False)[0]) for c, _ in METRICS[:4]))
    nul = np.array(nul)
    for c in obs:
        put(f"pbest4_{c}", (1 + (nul >= abs(obs[c])).sum()) / (1 + NPERM), 3)
    # robustness for the NFL launch level
    r_ = partial_sp(co, "g_true", "y", BASE + ["avg_get_off"]); put("pr_g_getoff", r_[0]); put("pr_g_getoff_lo", r_[1]); put("pr_g_getoff_hi", r_[2]); put("pr_g_getoff_n", r_[3])
    r_ = partial_sp(co, "g_true", "y", BASE + ["forty_z"]); put("pr_g_40", r_[0]); put("pr_g_40_n", r_[3])
    r_ = partial_sp(co, "ltg", "y", BASE + ["d_true"]); put("pr_lt_givendrill", r_[0]); put("pr_lt_givendrill_lo", r_[1]); put("pr_lt_givendrill_hi", r_[2])
    r_ = partial_sp(co, "ltg", "y", BASE + ["g_true"]); put("pr_lt_givengame", r_[0]); put("pr_lt_givengame_lo", r_[1]); put("pr_lt_givengame_hi", r_[2])
    tgm = tg.rename("g_top").reset_index()
    co = co.merge(tgm, on="nfl_id", how="left")
    r_ = partial_sp(co, "g_top", "y", BASE); put("pr_g_topepa", r_[0]); put("pr_g_topepa_n", r_[3])
    for role in ("EDGE", "interior"):
        s_ = co[co.role == role]
        r_ = partial_sp(s_, "g_true", "y", ["log_pick", "combine_weight_z"]); put(f"pr_g_{role}", r_[0]); put(f"pr_g_{role}_lo", r_[1]); put(f"pr_g_{role}_hi", r_[2]); put(f"pr_g_{role}_n", r_[3])
    put("rho_g_getoff_player", stats.spearmanr(co.g_true, co.avg_get_off)[0])
    put("rho_getoff_y", stats.spearmanr(co.avg_get_off, co.y)[0])
    put("rho_g_weight", stats.spearmanr(co.g_true, co.combine_weight)[0])
    put("rho_lt_pick", stats.spearmanr(co.ltg, co.log_pick, nan_policy="omit")[0])
    put("rho_lt_g", stats.spearmanr(co.ltg, co.g_true, nan_policy="omit")[0])
    put("rho_lt_d", stats.spearmanr(co.ltg, co.d_true, nan_policy="omit")[0])
    put("pow_g", power_at(abs(res["g_true"]["pr"]), len(co), 3))
    put("mde_press", mde(len(co), 3))
    put("ltg_excl0", int(((dl.ltg.abs() - 1.645 * dl.ltg_sd) > 0).sum()))
    # chain: drill->game x game->pressure vs drill->pressure
    put("chain_pred_d", float(NUM["sp_dg"]) * res["g_true"]["raw"]); put("chain_pred_f", float(NUM["sp_fg"]) * res["g_true"]["raw"])
    # LOCO dR2 (complete cases per metric)
    LO = {}
    for col, lab in METRICS:
        dd = co.dropna(subset=[col] + BASE + ["y", "w"])
        o = loco(dd, BASE, [col])
        LO[col] = o
        put(f"loco_{col}_d", o["delta"], 3); put(f"loco_{col}_lo", o["lo"], 3); put(f"loco_{col}_hi", o["hi"], 3)
        put(f"loco_{col}_rlo", o["rep_lo"], 3); put(f"loco_{col}_rhi", o["rep_hi"], 3); put(f"loco_{col}_pos", o["folds_pos"])
        put(f"loco_{col}_base", o["r2_base"]); put(f"loco_{col}_n", o["n"])
        for c, v in o["folds"].items():
            put(f"loco_{col}_f{c}", v, 3)
    dd = co.dropna(subset=["g_true", "avg_get_off"] + BASE)
    o = loco(dd, BASE + ["avg_get_off"], ["g_true"]); put("loco_g_over_getoff", o["delta"], 3); put("loco_g_over_getoff_lo", o["lo"], 3); put("loco_g_over_getoff_hi", o["hi"], 3)
    o = loco(dd, BASE, ["avg_get_off"]); put("loco_getoff", o["delta"], 3)
    # effect size: OLS slope of shrunk pressure on NFL launch (true-score sd) with baseline
    A = np.c_[np.ones(len(co)), co[BASE + ["g_true"]].values]
    beta = np.linalg.lstsq(A, co.y.values, rcond=None)[0]
    put("slope_g_pp", beta[-1]); put("press_med", co.y.median(), 1)
    put("press_per450", beta[-1] / 100 * 450, 1)
    bs = []
    for _ in range(NBOOT):
        i = RNG.integers(0, len(co), len(co))
        bs.append(np.linalg.lstsq(A[i], co.y.values[i], rcond=None)[0][-1])
    put("slope_g_lo", np.percentile(bs, 2.5)); put("slope_g_hi", np.percentile(bs, 97.5))

    # ------------------------------------------------------------------ named players (table 1)
    tb = co.dropna(subset=["ltg"]).sort_values("ltg", ascending=False)
    sel = pd.concat([tb.head(5), tb.tail(5)])
    rows = []
    for r in sel.itertuples():
        pick = "UDFA" if pd.isna(r.draft_overall_pick) else str(int(r.draft_overall_pick))
        rows.append(dict(name=r.display_name, pick=pick, role=r.role, yr=int(r.draft_year),
                         drill=f"{r.d_true:+.1f} ± {r.d_sd:.1f}", game=f"{r.g_true:+.1f} ± {r.g_sd:.1f}",
                         lt=f"{r.ltg:+.1f}", ret=f"{r.retention:.0f}", press=f"{r.y:.1f}", snaps=int(r.pass_rush_snaps)))
        put(f"tab_{r.nfl_id}_press", r.y, 1)
    TABLES["named"] = rows
    put("top_lt_mean_press", tb.head(5).y.mean(), 1); put("bot_lt_mean_press", tb.tail(5).y.mean(), 1)
    # level beats gap: among the 10 named, does game level or gap rank pressure better
    put("named_rho_g", stats.spearmanr(sel.g_true, sel.y)[0]); put("named_rho_lt", stats.spearmanr(sel.ltg, sel.y)[0])

    # ============================================================== figures
    tt = np.arange(T_CURVE) * 0.1
    ccur = [f"c{i}" for i in range(T_CURVE)]
    # ---- fig 1: drill-vs-game cards, 3 highest and 3 lowest LT in cohort (rule-chosen)
    pick6 = pd.concat([tb.head(3), tb.tail(3)])
    fig, axes = plt.subplots(2, 3, figsize=(FIGW, 7.8), sharex=True, sharey=True)
    for ax, r in zip(axes.ravel(), pick6.itertuples()):
        gc = Gd[Gd.nfl_id == r.nfl_id].dropna(subset=ccur)
        for row in gc[ccur].values:
            ax.plot(tt, row, color=OI["sky"], lw=0.7, alpha=0.45)
        if len(gc):
            ax.plot(tt, gc[ccur].mean().values, color=OI["blue"], lw=2.8, label=f"NFL rushes (n={len(gc)}) mean")
        dc = Dd[Dd.nfl_id == r.nfl_id].dropna(subset=ccur)
        for j, row in enumerate(dc[ccur].values):
            ax.plot(tt, row, color=OI["red"], lw=2.4, ls="--", label="Combine pass-rush drill" if j == 0 else None)
        ax.axvspan(0, 1.0, color=OI["yellow"], alpha=0.18, lw=0)
        pk = "UDFA" if pd.isna(r.draft_overall_pick) else f"pick {int(r.draft_overall_pick)}"
        ax.set_title(f"{r.display_name} ({r.role}, {pk})", fontsize=11.5)
        ax.text(0.03, 0.97, f"drill {r.d_true:+.1f}±{r.d_sd:.1f}  NFL {r.g_true:+.1f}±{r.g_sd:.1f}\n"
                            f"LT {r.ltg:+.1f}   retention {r.retention:.0f}%\npressure {r.y:.1f}% ({int(r.pass_rush_snaps)} rushes)",
                transform=ax.transAxes, va="top", fontsize=9.6,
                bbox=dict(boxstyle="round", fc="white", ec=OI["grey"], alpha=0.92))
    for ax in axes[1]:
        ax.set_xlabel("seconds from first movement")
    for ax in axes[:, 0]:
        ax.set_ylabel("speed (yd/s)")
    axes[0, 0].set_ylim(0, 8.5)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[:2], ["NFL random rushes, mean (thin = single snaps)", "Combine pass-rush drill rep"], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Top row: three highest Launch Translation (LT) in the cohort; bottom row: three lowest. "
                 "Yellow = first second.\nPressure does not sort with LT (career pressure, shrunk, in each box)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig1_cards.png")

    # ---- fig 2: screen heatmap (groups x features) for each source, with reliability text
    fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.4))
    for ax, src in zip(axs, SOURCES):
        M = C[C.src == src].pivot(index="group", columns="feat", values="rho").reindex(index=GROUPS, columns=FEATS)
        PB = C[C.src == src].pivot(index="group", columns="feat", values="p").reindex(index=GROUPS, columns=FEATS)
        CE = C[C.src == src].pivot(index="group", columns="feat", values="ceil").reindex(index=GROUPS, columns=FEATS)
        im = ax.imshow(M.values, cmap="PuOr_r", vmin=-0.5, vmax=0.5, aspect="auto")
        for i in range(len(GROUPS)):
            for j in range(len(FEATS)):
                rv, pv_, ce = M.values[i, j], PB.values[i, j], CE.values[i, j]
                star = "*" if pv_ < 0.05 else ""
                ces = f"\nceil {ce:.2f}" if (np.isfinite(ce) and ce > 0) else "\nceil n/a"
                ax.text(j, i, f"{rv:+.2f}{star}{ces}", ha="center", va="center", fontsize=9.5,
                        color="black", fontweight="bold" if (GROUPS[i] == "DL" and FEATS[j] == "acc") else "normal")
        ax.set_xticks(range(len(FEATS)), [FEAT_LABEL[f] for f in FEATS], fontsize=9.8)
        ax.set_yticks(range(len(GROUPS)), GROUPS)
        ax.set_title(f"Combine source: {SOURCES[src]}")
        ax.spines[:].set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.80, bottom=0.15, wspace=0.16)
    cax = fig.add_axes([0.905, 0.2, 0.015, 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Spearman ρ, Combine vs NFL")
    fig.suptitle(f"Translation screen, 40 cells on 20 random NFL snaps per player. * nominal p<0.05 "
                 f"({NUM['n_cells_p05']} of 40; {NUM['n_cells_exp05']} expected by chance).\n"
                 f"ceil = largest ρ that measurement noise allows (n/a = reliability ≤ 0). None clears the best-of-40 bar ({NUM['null_max_95']})",
                 fontsize=11.5)
    save(fig, "fig2_screen.png")

    # ---- fig 3: the chain (Combine launch -> NFL launch -> pressure)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(FIGW, 5.3), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    cmap = {"EDGE": OI["orange"], "interior": OI["blue"]}
    for role in ("EDGE", "interior"):
        s_ = dl[dl.role == role]
        a1.errorbar(s_.f_true, s_.g_true, xerr=s_.f_sd, yerr=s_.g_sd, fmt="o", ms=4.5, color=cmap[role], alpha=0.65,
                    elinewidth=0.6, label=role)
        a2.errorbar(s_.d_true, s_.g_true, xerr=s_.d_sd, yerr=s_.g_sd, fmt="o", ms=4.5, color=cmap[role], alpha=0.65,
                    elinewidth=0.6)
        c_ = co[co.role == role]
        a3.scatter(c_.g_true, c_.y, s=14 + c_.pass_rush_snaps / 25, color=cmap[role], alpha=0.7, edgecolor="white", lw=0.5)
    a1.set_title(f"A. 40-yd launch → NFL launch\nρ = {NUM['sp_fg']} [{NUM['sp_fg_lo']}, {NUM['sp_fg_hi']}], n = {NUM['sp_fg_n']}")
    a2.set_title(f"B. Rush-drill launch → NFL launch\nρ = {NUM['sp_dg']} [{NUM['sp_dg_lo']}, {NUM['sp_dg_hi']}], n = {NUM['sp_dg_n']}")
    a3.set_title(f"C. NFL launch → pressure rate\nρ = {NUM['raw_g_true']}; {NUM['pr_g_true']} beyond baseline, n = {NUM['n_cohort']}")
    a1.set_xlabel("Combine 40 launch (shrunk, sd)"); a2.set_xlabel("Combine drill launch (shrunk, sd)")
    a1.set_ylabel("NFL launch, 20 random rushes (shrunk, sd)")
    a3.set_xlabel("NFL launch (shrunk, sd)"); a3.set_ylabel("pressure rate % (shrunk)")
    a1.legend(loc="upper left", frameon=False)
    for ax in (a1, a2):
        ax.axhline(0, color=OI["grey"], lw=0.6); ax.axvline(0, color=OI["grey"], lw=0.6)
    fig.suptitle("Combine launch shows up in NFL launch (A, B); NFL launch tracks pressure mostly through weight and role (C).\n"
                 f"Product of the links: {NUM['chain_pred_d']} via the drill, {NUM['chain_pred_f']} via the 40. "
                 "Bars = posterior sd; dot size = rush snaps", fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig3_chain.png")

    # ---- fig 4: forest: partial rho + LOCO dR2 for the five candidates
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(FIGW, 5.0), sharey=True)
    ys = np.arange(len(METRICS))[::-1]
    for yv, (col, lab) in zip(ys, METRICS):
        rr, o = res[col], LO[col]
        colr = OI["red"] if col in ("ltg", "sim") else (OI["blue"] if col == "g_true" else OI["green"])
        b1.plot([rr["lo"], rr["hi"]], [yv, yv], color=colr, lw=2.4)
        b1.plot(rr["pr"], yv, "D", color=colr, ms=8)
        b1.text(rr["pr"], yv + 0.25, f"n={rr['n']}", va="bottom", ha="center", fontsize=9.5)
        b2.plot([o["rep_lo"], o["rep_hi"]], [yv, yv], color=colr, lw=2.4)
        b2.plot(o["delta"], yv, "D", color=colr, ms=8)
        for c_, v_ in o["folds"].items():
            b2.plot(v_, yv + 0.22, "|", color=OI["black"], ms=9)
    b1.axvline(0, color="black", lw=0.8); b2.axvline(0, color="black", lw=0.8)
    b1.set_yticks(ys, [lab for _, lab in METRICS])
    b1.set_xlabel("partial ρ with pressure rate, 95% CI")
    b2.set_xlabel("ΔR², bar = training redraws; ticks = classes")
    b1.set_title("In sample"); b2.set_title("Out of class")
    fig.suptitle("Beyond log pick + weight + EDGE share, no launch measure adds out of class.\n"
                 "Red = translation (drill-to-game match); blue = NFL launch level; green = Combine only", fontsize=12)
    fig.tight_layout()
    save(fig, "fig4_forest.png")

    (HERE / "numbers.json").write_text(json.dumps(NUM, indent=1))
    (HERE / "tables.json").write_text(json.dumps(TABLES, indent=1, default=float))
    print(json.dumps({k: NUM[k] for k in list(NUM)[:0]}))
    print(f"{len(NUM)} numbers written")


if __name__ == "__main__":
    main()
