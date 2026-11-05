"""Round 4 / cand 3. Launch translation for DL: does a pass rusher's Combine first step show up on random NFL rushes,
is that carry-over personal or body type, and does the drill-to-game match explain pressure beyond draft slot?

One feature function (features.py) runs on Combine drill reps and NFL snaps. Steps:
 1. Screen: 5 groups x 2 Combine sources (matched position drill, 40) x 4 movement features, Combine vs NFL player
    means on the 20 *random* NFL snaps per player; best-of-40 permutation p.
 2. DL launch (peak acceleration in the first second after first movement): empirical-Bayes player means, validation
    against NGS get-off, speed retention by phase of the rush, and a lookalike placebo (same role, nearest weight).
 3. Production: gap, curve-shape match, NFL launch level, drill and 40 launch vs shrunk pressure rate beyond log pick
    + weight + EDGE share; best-of-4 permutation; leave-one-class-out R2 gain; nested whole-pipeline LOCO (screen and
    measure choice redone per fold); run-stop placebo outcome; gap x EDGE interaction; pressures-per-season and
    draft-slot equivalents; selection into the 100-rush cohort.
 4. Figures, tables, and writeup.md rendered from writeup_template.md with every number taken from numbers.json.

Reads data/processed/*.parquet. Writes numbers.json, tables.json, figures/*.png, writeup.md next to this file.
Usage: uv run python writeup/candidates/round_04/cand_3/analysis.py   (about 4 minutes)
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

import os  # noqa: E402
import pickle  # noqa: E402

RNG = np.random.default_rng(2027)
FAST = os.environ.get("BDB_FAST") == "1"          # dev only: fewer resamples; final numbers use the full counts
NBOOT = 200 if FAST else 2000
NPERM = 100 if FAST else 1000
NREP = 30 if FAST else 300
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
MEANS_IDS: dict = {}
OUT: dict = {}
TABLES_MD: dict = {}
TABLES: dict = {}


def put(key, v, nd=2):
    """Store a number exactly as printed in the text (plus |value|, since the checker drops the sign)."""
    if isinstance(v, (int, np.integer)):
        NUM[key] = int(v)
        return str(int(v))
    s = f"{float(v):.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
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


def render():
    """Fill writeup_template.md with numbers.json values ({key}) and generated tables ({TABLE_x}) -> writeup.md."""
    import re
    tpl = HERE / "writeup_template.md"
    if not tpl.exists():
        return
    missing = []

    def sub(m):
        k = m.group(1)
        if k in TABLES_MD:
            return TABLES_MD[k]
        if k not in NUM:
            missing.append(k)
            return "??"
        v = NUM[k]
        v = f"{v:,}" if isinstance(v, int) else str(v)
        return "−" + v[1:] if v.startswith("-") else v
    (HERE / "writeup.md").write_text(re.sub(r"\{([A-Za-z][A-Za-z0-9_]*)\}", sub, tpl.read_text()))
    if missing:
        print("MISSING KEYS:", sorted(set(missing)))


# ============================================================================= round-4 additions
def extras(dl, co, D, Dd, Df, Gd, C, BASE, METRICS):
    """Lookalike placebo, nested whole-pipeline LOCO, placebo outcome, role interaction, pick-slot cost,
    selection into the 100-rush cohort, and launch retention by phase of the rush."""
    # --- (a) lookalike placebo: is a player's Combine launch closer to HIS NFL launch than to a same-role,
    # nearest-weight teammate-alike's NFL launch?
    # same inputs as the headline translation rho (unshrunk player z-means), so the "own" rho equals it exactly
    for src, xcol in (("drill", "d_m"), ("forty", "f_m")):
        s_ = dl.dropna(subset=[xcol, "g_m", "combine_weight"]).reset_index(drop=True)
        look = np.full(len(s_), np.nan)
        for i, r in s_.iterrows():
            c_ = s_[(s_.role == r.role) & (s_.nfl_id != r.nfl_id)]
            look[i] = c_.g_m.values[np.argmin(np.abs(c_.combine_weight.values - r.combine_weight))]
        x, gs = s_[xcol].values, s_.g_m.values
        closer = np.abs(x - gs) < np.abs(x - look)
        rs, rl = stats.spearmanr(x, gs)[0], stats.spearmanr(x, look)[0]
        bd, bc, bsf, blk = [], [], [], []
        for _ in range(NBOOT):
            i = RNG.integers(0, len(x), len(x))
            a_, b_ = stats.spearmanr(x[i], gs[i])[0], stats.spearmanr(x[i], look[i])[0]
            bd.append(a_ - b_); bsf.append(a_); blk.append(b_)
            bc.append(closer[i].mean())
        OUT[f"look_{src}"] = dict(self=(rs, *np.percentile(bsf, [2.5, 97.5])), look=(rl, *np.percentile(blk, [2.5, 97.5])))
        put(f"look_{src}_look_lo", np.percentile(blk, 2.5)); put(f"look_{src}_look_hi", np.percentile(blk, 97.5))
        put(f"look_{src}_self_lo", np.percentile(bsf, 2.5)); put(f"look_{src}_self_hi", np.percentile(bsf, 97.5))
        put(f"look_{src}_self", rs); put(f"look_{src}_look", rl); put(f"look_{src}_diff", rs - rl)
        put(f"look_{src}_diff_lo", np.percentile(bd, 2.5)); put(f"look_{src}_diff_hi", np.percentile(bd, 97.5))
        put(f"look_{src}_closer", 100 * closer.mean(), 0); put(f"look_{src}_closer_lo", 100 * np.percentile(bc, 2.5), 0)
        put(f"look_{src}_closer_hi", 100 * np.percentile(bc, 97.5), 0); put(f"look_{src}_n", len(x))

    # --- (b) symmetric nested leave-one-class-out of the whole pipeline: in each fold, the 40-cell screen and the
    # choice among the five production measures are redone on the two training classes only.
    yr = dl.set_index("nfl_id").draft_year
    yr_all = pd.read_parquet(PROC / "players.parquet").set_index("nfl_id").draft_year
    classes = sorted(co.draft_year.unique())
    cols = [c for c, _ in METRICS]

    def pipeline(cod, train_ids_all=None, boot=False):
        yv, pb, pa, cl, picks, cells = [], [], [], [], {}, {}
        for c in classes:
            tr, te = cod[cod.draft_year != c], cod[cod.draft_year == c]
            if boot:
                tr = tr.iloc[RNG.integers(0, len(tr), len(tr))]
            if not boot:
                best, bk = -9, None
                for key, (md, mg) in MEANS_IDS.items():
                    keep = yr_all.reindex(md.index).values != c
                    if keep.sum() >= 15:
                        rr_ = stats.spearmanr(md.values[keep], mg.values[keep])[0]
                        if rr_ > best:
                            best, bk = rr_, key
                cells[int(c)] = "/".join(bk)
            sc = {m: abs(partial_sp(tr, m, "y", BASE, boot=False)[0]) for m in cols}
            m = max(sc, key=sc.get)
            picks[int(c)] = m
            ok = te.dropna(subset=[m])
            tr2 = tr.dropna(subset=[m])
            pb.append(fit_pred(tr2, ok, BASE, "y", "w")); pa.append(fit_pred(tr2, ok, BASE + [m], "y", "w"))
            yv.append(ok.y.values); cl.append(np.full(len(ok), c))
        yv, pb, pa, cl = map(np.concatenate, (yv, pb, pa, cl))
        folds = {int(c): float(r2(yv[cl == c], pa[cl == c]) - r2(yv[cl == c], pb[cl == c])) for c in classes}
        return r2(yv, pa) - r2(yv, pb), folds, picks, cells
    cod = co.dropna(subset=BASE + ["y", "w"]).copy()
    d0, folds, picks, cells = pipeline(cod)
    put("nest_n", len(cod))
    put("nest_d", d0, 3)
    for c, v in folds.items():
        put(f"nest_f{c}", v, 3)
    put("nest_pos", int(sum(v > 0 for v in folds.values())))
    TABLES["nested"] = dict(picks=picks, cells=cells, folds=folds)
    put("nest_cell_dl_acc", int(sum(v.startswith("DL/") and v.endswith("/acc") for v in cells.values())))
    put("nest_pick_ltg", int(sum(v == "ltg" for v in picks.values())))
    dist = np.array([pipeline(cod, boot=True)[0] for _ in range(NREP)])
    put("nest_lo", np.percentile(dist, 2.5), 3); put("nest_hi", np.percentile(dist, 97.5), 3)
    put("nest_med", np.median(dist), 3); put("nest_share_pos", 100 * (dist > 0).mean(), 0)
    TABLES["nest_dist"] = dist.round(4).tolist()
    OUT["nest"] = dict(d=d0, folds=folds, dist=dist)

    # --- (c) placebo outcome: run-stop rate (tackles + assists per 100 non-rush snaps), same baseline
    ss = pd.read_parquet(PROC / "player_season_stats.parquet")
    ss = ss[ss.position_group == "DL"]
    ag = ss.groupby("nfl_id").agg(plays=("plays", "sum"), prs=("pass_rush_snaps", "sum"), tk=("tackles", "sum"),
                                  ast=("assists", "sum"), games=("games", "sum"), seasons=("season", "nunique"))
    ag["run_snaps"] = ag.plays - ag.prs
    ag["run_stop"] = 100 * (ag.tk + ag.ast) / ag.run_snaps.where(ag.run_snaps >= 50)
    ag["gps"] = ag.games / ag.seasons
    co2 = co.merge(ag[["run_stop", "gps"]].reset_index(), on="nfl_id", how="left")
    for col in ("ltg", "g_true"):
        r_ = partial_sp(co2, col, "run_stop", BASE)
        put(f"plac_{col}", r_[0]); put(f"plac_{col}_lo", r_[1]); put(f"plac_{col}_hi", r_[2]); put(f"plac_{col}_n", r_[3])
        OUT[f"plac_{col}"] = r_
    put("cohort_gps_med", co2.gps.median(), 1)

    # --- (d) does translation matter for one role? LT x EDGE-share interaction (OLS, bootstrap CI)
    di = co.dropna(subset=["ltg"] + BASE).copy()
    di["ltg_x_edge"] = di.ltg * di.edge_share
    Xc = BASE + ["ltg", "ltg_x_edge"]

    def ols(d_):
        A = np.c_[np.ones(len(d_)), d_[Xc].values]
        return np.linalg.lstsq(A, d_.y.values, rcond=None)[0]
    b = ols(di)
    bs = np.array([ols(di.iloc[RNG.integers(0, len(di), len(di))]) for _ in range(NBOOT)])
    put("int_lt_edge", b[-1]); put("int_lt_edge_lo", np.percentile(bs[:, -1], 2.5)); put("int_lt_edge_hi", np.percentile(bs[:, -1], 97.5))
    for role in ("EDGE", "interior"):
        r_ = partial_sp(co[co.role == role], "ltg", "y", ["log_pick", "combine_weight_z"])
        put(f"pr_lt_{role}", r_[0]); put(f"pr_lt_{role}_lo", r_[1]); put(f"pr_lt_{role}_hi", r_[2]); put(f"pr_lt_{role}_n", r_[3])

    # --- (e) what it is worth: slopes in pressures per season, and the draft-slot equivalent
    co3 = co.merge(ag[["prs", "seasons"]].reset_index(), on="nfl_id", how="left")
    rps = float((co3.prs / co3.seasons).median())
    put("rush_per_season_med", rps, 0)
    for col in ("ltg", "g_true"):
        dd = co.dropna(subset=[col] + BASE)
        A = np.c_[np.ones(len(dd)), dd[BASE + [col]].values]
        beta = np.linalg.lstsq(A, dd.y.values, rcond=None)[0]
        bs = np.array([np.linalg.lstsq(A[i], dd.y.values[i], rcond=None)[0] for i in
                       (RNG.integers(0, len(dd), len(dd)) for _ in range(NBOOT))])
        per = beta[-1] / 100 * rps
        put(f"ps_{col}", per, 1); put(f"ps_{col}_lo", np.percentile(bs[:, -1], 2.5) / 100 * rps, 1)
        put(f"ps_{col}_hi", np.percentile(bs[:, -1], 97.5) / 100 * rps, 1)
        bpick = beta[1]                                                      # pp per log-pick unit (negative)
        put(f"pickeq_{col}", 100 * np.exp(beta[-1] / bpick), 0)              # pick 100 -> this pick for +1 sd
        put("ps_per_logpick", -bpick / 100 * rps, 1)
        put("ps_pick100_to_50", -bpick * np.log(2) / 100 * rps, 1)

    # --- (f) selection into the 100-rush cohort
    has = dl.dropna(subset=["g_m"]).copy()
    has["in_cohort"] = (has.pass_rush_snaps >= MIN_SNAPS).astype(float)
    put("n_with_game", len(has)); put("n_out_cohort", int((has.in_cohort == 0).sum()))
    for col in ("d_true", "f_true", "ltg"):
        r_ = partial_sp(has, col, "in_cohort", ["log_pick"], boot=False)
        put(f"sel_{col}", r_[0]); put(f"sel_{col}_p", r_[4], 3)

    # --- (g) launch retention by phase of the rush (NFL mean curve / drill mean curve, per player)
    cc = [f"c{i}" for i in range(T_CURVE)]
    dcur = Dd.dropna(subset=cc).groupby("nfl_id")[cc].mean()
    gcur = Gd.dropna(subset=cc).groupby("nfl_id")[cc].mean()
    ii = dcur.index.intersection(gcur.index)
    ratio = 100 * gcur.loc[ii] / dcur.loc[ii]
    for t_, c in ((0.5, "c5"), (1.0, "c10"), (1.5, "c15"), (2.0, "c20")):
        put(f"spd_ret_{int(t_ * 10)}", ratio[c].median(), 0)
        put(f"spd_drill_{int(t_ * 10)}", dcur.loc[ii, c].median(), 1); put(f"spd_game_{int(t_ * 10)}", gcur.loc[ii, c].median(), 1)
    put("n_curve", len(ii))
    OUT["curves"] = (dcur.loc[ii], gcur.loc[ii])
    TABLES["curve_ratio_med"] = ratio.median().round(1).tolist()


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
    cache = os.environ.get("BDB_BUILD_CACHE")     # dev only: pickled build() output
    if cache and Path(cache).exists():
        D, G, gp = pickle.load(open(cache, "rb"))
    else:
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
                MEANS_IDS[(g, src, f)] = (md[ids], mg[ids])
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
    rr = sp_ci(-dl.forty_z, dl.g_m); put("sp_40time_g", rr[0]); put("sp_40time_g_lo", rr[1]); put("sp_40time_g_hi", rr[2]); put("sp_40time_g_n", rr[3])
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

    extras(dl, co, D, Dd, Df, Gd, C, BASE, METRICS)

    # ------------------------------------------------------------------ named players (table 1)
    tb = co.dropna(subset=["ltg"]).sort_values("ltg", ascending=False)
    sel = pd.concat([tb.head(5), tb.tail(5)])
    pmed = float(co.y.median())
    put("press_cohort_med", pmed, 1)

    def read(r):
        hi_l, hi_p = r.g_true >= 0.5, r.y >= pmed
        if hi_l and hi_p:
            return "Launch and finish: no flag"
        if hi_l:
            return "Launches, does not finish: hands, counters"
        if hi_p:
            return "Wins without launch: power, length"
        return "Below on both: rotational"
    rows = []
    md = ["| Player (pick, role, class) | Combine drill launch | NFL launch | Gap | Pressure % (rushes) | Read |",
          "|---|---|---|---|---|---|"]
    for r in sel.itertuples():
        pick = "UDFA" if pd.isna(r.draft_overall_pick) else str(int(r.draft_overall_pick))
        row = dict(name=r.display_name, pick=pick, role=r.role, yr=int(r.draft_year),
                   drill=f"{r.d_true:+.1f} ± {r.d_sd:.1f}", game=f"{r.g_true:+.1f} ± {r.g_sd:.1f}",
                   lt=f"{r.ltg:+.1f}", press=f"{r.y:.1f}", snaps=int(r.pass_rush_snaps), read=read(r))
        rows.append(row)
        md.append(f"| {row['name']} ({pick}, {row['role']}, {row['yr']}) | {row['drill'].replace('-', '−')} | "
                  f"{row['game'].replace('-', '−')} | {row['lt'].replace('-', '−')} | {row['press']} ({row['snaps']:,}) | {row['read']} |")
    TABLES["named"] = rows
    TABLES_MD["TABLE_named"] = "\n".join(md)
    put("top_lt_mean_press", tb.head(5).y.mean(), 1); put("bot_lt_mean_press", tb.tail(5).y.mean(), 1)
    put("named_rho_g", stats.spearmanr(sel.g_true, sel.y)[0]); put("named_rho_lt", stats.spearmanr(sel.ltg, sel.y)[0])

    # ============================================================== figures
    tt = np.arange(T_CURVE) * 0.1
    ccur = [f"c{i}" for i in range(T_CURVE)]
    cmap = {"EDGE": OI["orange"], "interior": OI["blue"]}

    # ---- fig 1: the cohort's speed curve, drill vs NFL, by phase of the rush
    dc, gc = OUT["curves"]
    fig, ax = plt.subplots(figsize=(FIGW, 5.4))
    for M, col, lab in ((dc, OI["red"], "Combine pass-rush drill"), (gc, OI["blue"], "NFL random rushes (snap to +3 s)")):
        lo, mdn, hi = M.quantile([0.25, 0.5, 0.75]).values
        ax.fill_between(tt, lo, hi, color=col, alpha=0.16, lw=0)
        ax.plot(tt, mdn, color=col, lw=3, label=f"{lab}, median and middle 50%")
    gm = gc.median().values
    for k in (5, 10, 15, 20):
        ax.annotate(f"NFL speed =\n{NUM[f'spd_ret_{k}']}% of drill", xy=(k / 10, gm[k]), xytext=(k / 10, gm[k] - 1.35),
                    ha="center", fontsize=10, color=OI["blue"],
                    arrowprops=dict(arrowstyle="-", color=OI["grey"], lw=0.8))
    ax.axvspan(0, 0.5, color=OI["yellow"], alpha=0.22, lw=0)
    ax.text(0.25, 6.3, "first\nhalf-second", ha="center", fontsize=10)
    ax.set_xlim(0, 2.0); ax.set_ylim(0, 6.9)
    ax.set_xlabel("seconds from first movement"); ax.set_ylabel("speed (yd/s)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.16, 1.0), frameon=False)
    ax.set_title(f"The first step is the same in Indianapolis and on Sundays; the rush splits after it "
                 f"({NUM['n_curve']} DL, per-player mean curves)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig1_phase.png")

    # ---- fig 2: drill-vs-game cards, 3 largest and 3 smallest gaps in the cohort (rule-chosen)
    pick6 = pd.concat([tb.head(3), tb.tail(3)])
    fig, axes = plt.subplots(2, 3, figsize=(FIGW, 7.6), sharex=True, sharey=True)
    for ax, r in zip(axes.ravel(), pick6.itertuples()):
        g_ = Gd[Gd.nfl_id == r.nfl_id].dropna(subset=ccur)
        for row in g_[ccur].values:
            ax.plot(tt, row, color=OI["sky"], lw=0.7, alpha=0.45)
        if len(g_):
            ax.plot(tt, g_[ccur].mean().values, color=OI["blue"], lw=2.8)
        d_ = Dd[Dd.nfl_id == r.nfl_id].dropna(subset=ccur)
        for row in d_[ccur].values:
            ax.plot(tt, row, color=OI["red"], lw=2.4, ls="--")
        ax.axvspan(0, 1.0, color=OI["yellow"], alpha=0.18, lw=0)
        pk = "UDFA" if pd.isna(r.draft_overall_pick) else f"pick {int(r.draft_overall_pick)}"
        ax.set_title(f"{r.display_name} ({r.role}, {pk})", fontsize=11)
        ax.text(0.03, 0.97, f"{len(d_)} drill rep(s), {len(g_)} NFL rushes\n"
                            f"Combine launch {r.d_true:+.1f}, NFL {r.g_true:+.1f} (sd)\n"
                            f"pressure {r.y:.1f}% on {int(r.pass_rush_snaps):,} rushes",
                transform=ax.transAxes, va="top", fontsize=9.5,
                bbox=dict(boxstyle="round", fc="white", ec=OI["grey"], alpha=0.92))
    for ax in axes[1]:
        ax.set_xlabel("seconds from first movement")
    for ax in axes[:, 0]:
        ax.set_ylabel("speed (yd/s)")
    axes[0, 0].set_ylim(0, 8.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=OI["red"], lw=2.4, ls="--"), Line2D([], [], color=OI["blue"], lw=2.8),
         Line2D([], [], color=OI["sky"], lw=1)]
    fig.legend(h, ["Combine pass-rush drill rep", "NFL rushes, mean", "single NFL rush"], loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Top row: played furthest above their Combine first step. Bottom row: furthest below.\n"
                 "Both rows hold good and poor pass rushers (yellow = first second)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig2_cards.png")

    # ---- fig 3: translation screen as one heatmap
    rows_lab, Mv, Pv = [], [], []
    for g in GROUPS:
        for src in SOURCES:
            sub = C[(C.group == g) & (C.src == src)].set_index("feat").reindex(FEATS)
            rows_lab.append(f"{g}, {SOURCES[src]}"); Mv.append(sub.rho.values); Pv.append(sub.p.values)
    Mv, Pv = np.array(Mv), np.array(Pv)
    fig, ax = plt.subplots(figsize=(FIGW, 6.4))
    im = ax.imshow(Mv, cmap="PuOr_r", vmin=-0.45, vmax=0.45, aspect="auto")
    for i in range(Mv.shape[0]):
        for j in range(Mv.shape[1]):
            ax.text(j, i, f"{Mv[i, j]:+.2f}{'*' if Pv[i, j] < 0.05 else ''}", ha="center", va="center", fontsize=10.5,
                    fontweight="bold" if (rows_lab[i].startswith("DL") and FEATS[j] == "acc") else "normal")
    for i in (0, 1):
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=2))
    ax.set_xticks(range(len(FEATS)), [FEAT_LABEL[f].replace("\n", " ") for f in FEATS])
    ax.set_yticks(range(len(rows_lab)), rows_lab)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Spearman ρ, Combine vs NFL player means")
    ax.set_title(f"Only DL launch (boxed) is positive from both Combine sources. * = nominal p < 0.05: "
                 f"{NUM['n_cells_p05']} of {NUM['n_cells']} cells,\n{NUM['n_cells_exp05']} expected by chance; "
                 f"no cell clears the best-of-{NUM['n_cells']} bar of {NUM['null_max_95']}", fontsize=12)
    fig.tight_layout()
    save(fig, "fig3_screen.png")

    # ---- fig 4: Combine launch -> NFL launch, own vs lookalike
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIGW, 5.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for role in ("EDGE", "interior"):
        s_ = dl[dl.role == role]
        a1.errorbar(s_.f_true, s_.g_true, xerr=s_.f_sd, yerr=s_.g_sd, fmt="o", ms=5, color=cmap[role], alpha=0.65,
                    elinewidth=0.6, label=role)
    a1.axhline(0, color=OI["grey"], lw=0.6); a1.axvline(0, color=OI["grey"], lw=0.6)
    a1.set_xlabel("Combine 40-yard launch (shrunk, sd)"); a1.set_ylabel("NFL launch, 20 random rushes (shrunk, sd)")
    a1.set_title(f"A. 40 launch vs NFL launch: ρ = {NUM['sp_fg']}, n = {NUM['sp_fg_n']}\nbars = posterior sd")
    a1.legend(loc="upper left", frameon=False)
    items = [("Rush drill → his own NFL launch", OUT["look_drill"]["self"], OI["red"]),
             ("Rush drill → lookalike's NFL launch", OUT["look_drill"]["look"], OI["grey"]),
             ("40 → his own NFL launch", OUT["look_forty"]["self"], OI["red"]),
             ("40 → lookalike's NFL launch", OUT["look_forty"]["look"], OI["grey"]),
             ("Official 40 time → his own NFL launch", (float(NUM["sp_40time_g"]), float(NUM["sp_40time_g_lo"]),
                                                       float(NUM["sp_40time_g_hi"])), OI["green"])]
    for k, (lab, (r_, lo, hi), col) in enumerate(items[::-1]):
        a2.plot([lo, hi], [k, k], color=col, lw=2.6); a2.plot(r_, k, "D", color=col, ms=8)
        a2.text(hi + 0.02, k, f"{r_:+.2f}", va="center", fontsize=10)
    a2.set_yticks(range(len(items)), [i[0] for i in items[::-1]], fontsize=10)
    a2.axvline(0, color="black", lw=0.8); a2.set_xlim(-0.35, 0.85)
    a2.set_xlabel("Spearman ρ, 95% bootstrap CI")
    a2.set_title("B. Lookalike = same role, nearest weight\n(another player)")
    fig.suptitle("Combine launch previews NFL launch, but no better than it previews a same-weight lookalike's",
                 fontsize=12.5)
    fig.tight_layout()
    save(fig, "fig4_translation.png")

    # ---- fig 5: does it pay? partial rho and held-out dR2
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(FIGW, 5.8))
    lab_short = {"ltg": "Gap (NFL minus Combine)", "sim": "Curve-shape match", "g_true": "NFL launch level",
                 "d_true": "Combine rush-drill launch", "f_true": "Combine 40 launch"}
    ys = np.arange(len(METRICS) + 1)[::-1]
    for yv, (col, _) in zip(ys[:-1], METRICS):
        rr, o = res[col], LO[col]
        colr = OI["red"] if col in ("ltg", "sim") else (OI["blue"] if col == "g_true" else OI["green"])
        b1.plot([rr["lo"], rr["hi"]], [yv, yv], color=colr, lw=2.4); b1.plot(rr["pr"], yv, "D", color=colr, ms=8)
        b2.plot([o["rep_lo"], o["rep_hi"]], [yv, yv], color=colr, lw=2.4); b2.plot(o["delta"], yv, "D", color=colr, ms=8)
        for c_, v_ in o["folds"].items():
            b2.plot(v_, yv + 0.25, "|", color=OI["black"], ms=9)
    pl = OUT["plac_ltg"]
    b1.plot([pl[1], pl[2]], [ys[-1], ys[-1]], color=OI["purple"], lw=2.4); b1.plot(pl[0], ys[-1], "s", color=OI["purple"], ms=8)
    ne = OUT["nest"]
    b2.plot(np.percentile(ne["dist"], [2.5, 97.5]), [ys[-1], ys[-1]], color=OI["purple"], lw=2.4)
    b2.plot(ne["d"], ys[-1], "s", color=OI["purple"], ms=8)
    for v_ in ne["folds"].values():
        b2.plot(v_, ys[-1] + 0.25, "|", color=OI["black"], ms=9)
    b1.set_yticks(ys, [lab_short[c] for c, _ in METRICS] + ["Placebo outcome:\ngap → run-stop rate"])
    b2.set_yticks(ys, [""] * len(METRICS) + ["Whole pipeline re-run\n(screen + choice of measure)"])
    b2.yaxis.tick_right()
    for b in (b1, b2):
        b.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    for b in (b1, b2):
        b.axvline(0, color="black", lw=0.8)
    b1.set_xlabel("partial ρ beyond the baseline\n(95% bootstrap CI)")
    b2.set_xlabel("held-out R² gain\n(bar = 95% of training redraws; ticks = classes)")
    b1.set_title(f"In sample, n = {NUM['n_cohort']} DL with {MIN_SNAPS}+ rushes")
    b2.set_title("Out of class")
    fig.suptitle("No launch measure, and no measure of the drill-to-game match, adds to draft slot, weight and role",
                 fontsize=12.5)
    fig.tight_layout()
    save(fig, "fig5_pay.png")
    (HERE / "numbers.json").write_text(json.dumps(NUM, indent=1))
    (HERE / "tables.json").write_text(json.dumps(TABLES, indent=1, default=float))
    render()
    print(f"{len(NUM)} numbers written")


if __name__ == "__main__":
    main()
