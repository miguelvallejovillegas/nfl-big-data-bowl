"""Does he play to his 40? Speed Realization: Combine 40 speed profile vs the same player's NFL snaps.

Round 2, candidate 3 (revision of round-1 winner "Burst Translation").
Question: for each player, how much of the speed profile he shows in the Combine 40 (first-second distance,
2.5-s distance, top speed) reappears on his NFL snaps of the matching movement, and does that translation explain
production beyond draft pick? We test every position group and phase, pick the clearest one (WR top end) and check
whether the production link survives once ball-in-hand running is removed.

Game side = data/processed/game_frames_sample (up to 40 plays per player; `sample_kind` random vs top_epa).
Primary sample = the random half; the top_epa half is the replication.

Reads data/processed/*.parquet (player table through analysis/insights.py:load). Writes numbers.json, tables.json and
figures/*.png next to this file.

Usage: uv run python writeup/candidates/round_02/cand_3/analysis.py
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
T = 41                  # frames kept after first movement (0.0-4.0 s)
F1, F25 = 10, 25        # first-second and 2.5-s windows (frames)
ON, PRE = 1.0, 0.5      # onset threshold and back-off threshold (yd/s)
TOPK = 3                # top end = mean of a player's 3 fastest snaps
MIN_SNAPS = 10          # players need >= 10 random snaps of the movement
END_EVENTS = {"pass_forward", "qb_sack", "qb_strip_sack", "handoff", "run", "pass_shovel", "lateral", "qb_spike",
              "qb_kneel", "qb_sneak", "run_attempt", "snap_direct"}
ARRIVE = {"pass_arrived", "pass_outcome_caught", "pass_outcome_incomplete", "pass_outcome_interception",
          "pass_outcome_touchdown", "autoevent_passinterrupted", "pass_tipped"} | (END_EVENTS - {"pass_forward"})
CAP = 50                # if no end event: window = 5.0 s after the snap
GROUPS = ["WR", "TE", "DL", "DB", "OL"]
PHASES = [("d1", "start: first-second distance"), ("d25", "drive: 2.5-s distance"), ("top", "top end: top speed")]
GAME_TYPE = {"DL": "pass rushes", "OL": "snaps", "WR": "routes", "TE": "routes", "DB": "coverage snaps"}
NBOOT = 1000
NPERM = 1000
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F", "black": "#000000"}
GCOL = {"DL": OI["red"], "OL": OI["orange"], "WR": OI["blue"], "TE": OI["green"], "DB": OI["purple"]}
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
FIGW = 1600 / 150
NUM: dict = {}


def put(key: str, value, fmt: str = "{:.2f}") -> str:
    s = fmt.format(value) if not isinstance(value, str) else value
    NUM[key] = s
    return s


def log(m: str) -> None:
    print(m, flush=True)


def save(fig, name):
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- curves
def onset_curve(s: np.ndarray):
    """Speed for T frames from first movement (first frame >= ON, backed up <= 3 frames while >= PRE); NaN-padded."""
    s = np.asarray(s, float)
    if not len(s) or s[0] >= ON:
        return None
    hit = np.flatnonzero(s >= ON)
    if not len(hit):
        return None
    i = j = int(hit[0])
    while j > 0 and s[j - 1] >= PRE and i - j < 3:
        j -= 1
    c = s[j:j + T]
    return np.r_[c, np.full(T - len(c), np.nan)]


def dist(c: np.ndarray, n: int) -> float:
    """Yards covered in the first n frames after onset (NaN if the window is cut short)."""
    seg = c[:n]
    return float(seg.sum() * DT) if np.isfinite(seg).all() else np.nan


def zyear(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby(["position_group", "draft_year"])[col].transform(lambda v: (v - v.mean()) / v.std())


def combine_40(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cf = pd.read_parquet(PROC / "combine_frames.parquet", columns=["event_id", "frame", "entity_type", "s"])
    f40 = ev[ev.drill_type == "FORTY_YARD_DASH"][["event_id", "nfl_id", "attempt_no"]]
    cf = cf[(cf.entity_type == "PLAYER") & cf.event_id.isin(f40.event_id)].sort_values(["event_id", "frame"])
    rows, curves = [], []
    for eid, d in cf.groupby("event_id"):
        c = onset_curve(d.s.values)
        if c is None:
            continue
        rows.append((eid, dist(c, F1), dist(c, F25), float(d.s.max())))
        curves.append((eid, *c))
    A = pd.DataFrame(rows, columns=["event_id", "d1", "d25", "top"]).merge(f40, on="event_id")
    Cc = pd.DataFrame(curves, columns=["event_id"] + list(range(T))).merge(f40, on="event_id")
    return A, Cc


def game_snaps(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gp = pd.read_parquet(PROC / "game_plays_sample.parquet")
    gf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "s", "event"])
    gf = gf.sort_values(["game_id", "play_id", "nfl_id", "frame"])
    k = ["game_id", "play_id", "nfl_id"]
    snapf = gp.set_index(k).snap_frame
    rows, curves = [], []
    for key, d in gf.groupby(k, sort=False):
        sf = snapf.get(key, np.nan)
        if not np.isfinite(sf):
            continue
        f, s = d.frame.values, d.s.values
        post = f >= sf
        if not post.any():
            continue
        endf = d.frame[(d.frame >= sf) & d.event.isin(END_EVENTS)]
        end = int(endf.iloc[0]) if len(endf) else int(sf + CAP)
        arrf = d.frame[(d.frame >= sf) & d.event.isin(ARRIVE)]
        arr = int(arrf.iloc[0]) if len(arrf) else int(sf + CAP)
        pre = post & (f <= end)
        upto = post & (f <= arr)
        c = onset_curve(s[pre])
        rows.append((*key, dist(c, F1) if c is not None else np.nan, dist(c, F25) if c is not None else np.nan,
                     float(s[pre].max()) if pre.any() else np.nan, float(s[post].max()),
                     float(s[upto].max()) if upto.any() else np.nan, (end - sf) * DT))
        if c is not None:
            curves.append((*key, *c))
    G = pd.DataFrame(rows, columns=k + ["d1", "d25", "top", "top_all", "top_arr", "t_end"])
    Gc = pd.DataFrame(curves, columns=k + list(range(T)))
    meta = gp[k + ["sample_kind", "season", "lined_up_position", "route_ran", "target", "time_to_pressure"]].copy()
    meta["route"] = meta.route_ran.fillna("NONE")
    meta["pressure_play"] = meta.time_to_pressure.notna()
    meta["target"] = meta.target.fillna(False).astype(bool)
    G = G.merge(meta, on=k).merge(p[["nfl_id", "position_group"]], on="nfl_id")
    G = G.rename(columns={"position_group": "grp"})
    return G, Gc.merge(meta[k + ["sample_kind", "target"]], on=k), gp


# ----------------------------------------------------------------------------- stats helpers
def adjust(G: pd.DataFrame, col: str, cells: list[str]) -> pd.Series:
    """Remove situation effects: value minus its (group x cell) mean, plus the group mean."""
    return G[col] - G.groupby(["grp"] + cells)[col].transform("mean") + G.groupby("grp")[col].transform("mean")


def topk(v: np.ndarray) -> float:
    v = np.sort(v[np.isfinite(v)])
    return float(v[-TOPK:].mean()) if len(v) >= TOPK else np.nan


AGG = {"d1": lambda v: float(np.nanmean(v)) if np.isfinite(v).sum() else np.nan,
       "d25": lambda v: float(np.nanmean(v)) if np.isfinite(v).sum() else np.nan,
       "top": topk, "top_all": topk, "top_arr": topk}


def player_summary(R: pd.DataFrame, col: str, agg: str) -> pd.Series:
    return R.groupby("nfl_id")[col].apply(lambda v: AGG[agg](v.values))


def split_half(R: pd.DataFrame, col: str, agg: str) -> float:
    """Odd/even-snap split-half Spearman, Spearman-Brown stepped up to the full sample."""
    R = R.copy()
    R["h"] = R.groupby("nfl_id").cumcount() % 2
    hh = R.groupby(["nfl_id", "h"])[col].apply(lambda v: AGG[agg](v.values)).unstack()
    r = stats.spearmanr(hh[0], hh[1], nan_policy="omit")[0]
    return float(2 * r / (1 + r)) if r > -1 else np.nan


def rank_resid(df: pd.DataFrame, col: str, Z: list[str]) -> np.ndarray:
    R = df[[col] + Z].rank()
    A = np.c_[np.ones(len(R)), R[Z].values] if Z else np.ones((len(R), 1))
    return (R[col] - A @ np.linalg.lstsq(A, R[col].values, rcond=None)[0]).values


def partial_rho(df: pd.DataFrame, a: str, b: str, Z: list[str], boot: bool = True):
    d = df[[a, b] + Z].dropna()
    n = len(d)

    def one(dd):
        return np.corrcoef(rank_resid(dd, a, Z), rank_resid(dd, b, Z))[0, 1]
    r = one(d)
    pv = float(stats.pearsonr(rank_resid(d, a, Z), rank_resid(d, b, Z))[1])
    if not boot:
        return float(r), np.nan, np.nan, n, pv
    bs = [one(d.iloc[RNG.integers(0, n, n)]) for _ in range(NBOOT)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(r), float(lo), float(hi), n, pv


def mde(n: int, power: float = 0.8) -> float:
    """Smallest correlation detectable at two-sided alpha 0.05 with the given power (Fisher z)."""
    return float(np.tanh((stats.norm.ppf(0.975) + stats.norm.ppf(power)) / np.sqrt(n - 3 - 0)))


def tost(r: float, n: int, bound: float = 0.3) -> float:
    """Two one-sided tests for |rho| < bound (Fisher z); returns the larger one-sided p."""
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    p1 = 1 - stats.norm.cdf((z + np.arctanh(bound)) / se)
    p2 = stats.norm.cdf((z - np.arctanh(bound)) / se)
    return float(max(p1, p2))


def r2(y, p_):
    return float(1 - ((y - p_) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def wls(tr, cols, y, w):
    A = np.c_[np.ones(len(tr)), tr[cols].values]
    sw = np.sqrt(tr[w].values) if w else np.ones(len(tr))
    return np.linalg.lstsq(A * sw[:, None], tr[y].values * sw, rcond=None)[0]


def loco(df: pd.DataFrame, base: list[str], extra: list[str], y: str, w: str | None = None, perm: bool = True):
    classes = sorted(df.draft_year.unique())

    def run(d):
        ys, pb, pa, cl = [], [], [], []
        for c in classes:
            tr, te = d[d.draft_year != c], d[d.draft_year == c]
            for cols, store in ((base, pb), (base + extra, pa)):
                b = wls(tr, cols, y, w)
                store.append(np.c_[np.ones(len(te)), te[cols].values] @ b)
            ys.append(te[y].values)
            cl.append(np.full(len(te), c))
        return tuple(map(np.concatenate, (ys, pb, pa, cl)))
    yv, pb, pa, cl = run(df)
    out = dict(n=len(yv), r2_base=r2(yv, pb), r2_aug=r2(yv, pa))
    out["delta"] = out["r2_aug"] - out["r2_base"]
    out["folds"] = {int(c): dict(n=int((cl == c).sum()), r2_base=r2(yv[cl == c], pb[cl == c]),
                                 r2_aug=r2(yv[cl == c], pa[cl == c])) for c in classes}
    out["folds_pos"] = int(sum(f["r2_aug"] > f["r2_base"] for f in out["folds"].values()))
    n = len(yv)
    bd = []
    for _ in range(NBOOT):
        i = RNG.integers(0, n, n)
        bd.append(r2(yv[i], pa[i]) - r2(yv[i], pb[i]))
    out["lo"], out["hi"] = (float(v) for v in np.percentile(bd, [2.5, 97.5]))
    if perm:
        null = []
        for _ in range(NPERM // 2):
            dp = df.copy()
            for c in classes:
                m = (dp.draft_year == c).values
                dp.loc[m, extra] = dp.loc[m, extra].values[RNG.permutation(m.sum())]
            a, b_, c_, _ = run(dp)
            null.append(r2(a, c_) - r2(a, b_))
        out["perm_p"] = float((1 + np.sum(np.array(null) >= out["delta"])) / (1 + len(null)))
    return out


def holm_bh(pvals: dict):
    keys = list(pvals)
    p = np.array([pvals[k] for k in keys])
    m = len(p)
    o = np.argsort(p)
    holm, run_max = np.empty(m), 0
    for rank, i in enumerate(o):
        run_max = max(run_max, (m - rank) * p[i])
        holm[i] = min(1, run_max)
    bh, prev = np.empty(m), 1
    for rank in range(m - 1, -1, -1):
        i = o[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        bh[i] = prev
    return {k: dict(p=float(pvals[k]), holm=float(h), bh=float(b)) for k, h, b in zip(keys, holm, bh)}


def eb_rate(num: pd.Series, den: pd.Series):
    """Normal-normal EB shrinkage of a per-opportunity rate (num/den) by opportunity count."""
    m = den > 0
    rate = (num / den).where(m)
    mu = num[m].sum() / den[m].sum()
    # var(rate_i) = tau2 + s2/den_i  -> weighted regression of squared deviation on 1/den
    dev2 = (rate[m] - mu) ** 2
    A = np.c_[np.ones(m.sum()), 1 / den[m].values]
    tau2, s2 = np.linalg.lstsq(A * np.sqrt(den[m].values)[:, None], dev2.values * np.sqrt(den[m].values), rcond=None)[0]
    tau2 = max(tau2, 1e-6)
    lam = tau2 / (tau2 + s2 / den)
    return (mu + lam * (rate - mu)).where(m), lam.where(m), float(tau2), float(s2)


# ----------------------------------------------------------------------------- main
def main() -> None:
    p = insights.load()["p"].copy()
    p["weight_z"] = zyear(p, "combine_weight")
    p["forty_z"] = -zyear(p, "forty")          # positive = faster
    p["ngs_z"] = zyear(p, "ngs_athleticism_score")
    ev = pd.read_parquet(PROC / "combine_events.parquet")
    A40, C40 = combine_40(ev)
    G, Gc, gp = game_snaps(p)
    put("n_players", len(p), "{:d}")
    put("n_game_plays", len(gp), "{:,}")
    put("n_random_plays", int((gp.sample_kind == "random").sum()), "{:,}")
    put("n_40_reps", len(A40), "{:,}")
    put("n_players_40", A40.nfl_id.nunique(), "{:d}")

    # 40 profile per player (mean over attempts) and its retest reliability
    c40 = A40.groupby("nfl_id")[["d1", "d25", "top"]].mean()
    c40["n40"] = A40.groupby("nfl_id").size()
    c40 = c40.join(p.set_index("nfl_id")[["position_group", "draft_year"]], how="inner")
    for ph in ("d1", "d25", "top"):
        c40[ph + "_unadj"] = c40[ph]
        gstat = c40.groupby("position_group")[ph].agg(["mean", "std"])
        z = c40.groupby(["position_group", "draft_year"])[ph].transform(lambda v: (v - v.mean()) / v.std())
        c40[ph] = gstat["mean"].reindex(c40.position_group).values + z * gstat["std"].reindex(c40.position_group).values
    c40 = c40.drop(columns=["position_group", "draft_year"])
    rel40 = {}
    for ph, _ in PHASES:
        pairs = A40.sort_values("attempt_no").groupby("nfl_id")[ph].apply(lambda v: v.values[:2] if len(v) >= 2 else None).dropna()
        pr = np.vstack(pairs.values)
        r1 = float(stats.pearsonr(pr[:, 0], pr[:, 1])[0])
        k = float(c40.n40.mean())
        rel40[ph] = k * r1 / (1 + (k - 1) * r1)
        put(f"retest40_{ph}", r1)
        put(f"retest40_n_{ph}", len(pr), "{:d}")
    put("n40_mean", float(c40.n40.mean()), "{:.1f}")

    # game: situation adjustment (alignment; + route for receivers)
    G["cell"] = np.where(G.grp.isin(["WR", "TE"]), G.lined_up_position.fillna("NA") + "|" + G.route, G.lined_up_position.fillna("NA"))
    for col in ("d1", "d25", "top", "top_all", "top_arr"):
        G[col + "_a"] = adjust(G, col, ["cell"])
    put("median_t_end", G[G.sample_kind == "random"].t_end.median(), "{:.1f}")
    RND = G[G.sample_kind == "random"]
    TOP = G[G.sample_kind == "top_epa"]
    put("snaps_per_player_rand", float(RND.groupby("nfl_id").size().median()), "{:.0f}")

    # ------------------------------------------------------------ translation grid (all groups x phases)
    grid, keep = [], {}
    for g in GROUPS:
        Rg, Tg = RND[RND.grp == g], TOP[TOP.grp == g]
        x = p[p.position_group == g].set_index("nfl_id").join(c40, how="inner")
        nsn = Rg.groupby("nfl_id").size()
        x = x[nsn.reindex(x.index).fillna(0) >= MIN_SNAPS]
        for ph, lab in PHASES:
            x[f"g_{ph}"] = player_summary(Rg, ph + "_a", ph)
            x[f"t_{ph}"] = player_summary(Tg, ph + "_a", ph)
            x[f"graw_{ph}"] = player_summary(Rg, ph, ph)
            relg = split_half(Rg[Rg.nfl_id.isin(x.index)], ph + "_a", ph)
            raw = partial_rho(x.reset_index(), ph, f"graw_{ph}", [], boot=False)
            r, lo, hi, n, pv = partial_rho(x.reset_index(), ph, f"g_{ph}", ["weight_z"])
            rt = partial_rho(x.reset_index(), ph, f"t_{ph}", ["weight_z"], boot=False)
            ceil = float(np.sqrt(max(rel40[ph], 0) * max(relg, 0)))
            grid.append(dict(group=g, phase=ph, rho_raw=raw[0], rho=r, lo=lo, hi=hi, n=n, p=pv, rho_top=rt[0],
                             rel_g=relg, rel_c=rel40[ph], ceiling=ceil, mde=mde(n)))
            for kk, v in grid[-1].items():
                if kk in ("group", "phase"):
                    continue
                put(f"grid_{g}_{ph}_{kk}", v, "{:d}" if kk == "n" else ("{:.3f}" if kk == "p" else "{:.2f}"))
        keep[g] = x
    GRID = pd.DataFrame(grid)
    # pick rule: clearest translation among groups that have a production outcome tied to the movement and n >= 40
    # (DB has no coverage outcome in the data; TE n is too small)
    elig_g = GRID[GRID.group.isin(["WR", "DL", "OL"]) & (GRID.n >= 40)]
    best = elig_g.sort_values("rho", ascending=False).iloc[0]
    put("best_group", best.group, "{}")
    put("min_snaps", MIN_SNAPS, "{:d}")
    put("best_phase", best.phase, "{}")

    # DL contrast: drive distance before / after alignment + size
    x = keep["DL"]
    al = pd.read_parquet(PROC / "player_season_alignment.parquet")
    tot = al.groupby("nfl_id").plays.sum()
    edge = al[al.lined_up_position.isin(["EDGE", "OLB"])].groupby("nfl_id").plays.sum().reindex(tot.index).fillna(0) / tot
    x["edge_share"] = edge.reindex(x.index)
    x["graw_d25"] = player_summary(RND[RND.grp == "DL"], "d25", "d25")
    put("dl_d25_raw", partial_rho(x.reset_index(), "d25", "graw_d25", [], boot=False)[0])
    put("dl_d25_adj", partial_rho(x.reset_index(), "d25", "g_d25", [], boot=False)[0])
    put("dl_d25_partial", partial_rho(x.reset_index(), "d25", "g_d25", ["weight_z", "edge_share"], boot=False)[0])
    put("dl_w_d25", partial_rho(x.reset_index(), "combine_weight", "d25", [], boot=False)[0])
    # DL: NFL rush drive distance vs pressure beyond baseline (and without pressure snaps)
    x["g_d25_np"] = player_summary(RND[(RND.grp == "DL") & ~RND.pressure_play], "d25_a", "d25")
    x["eb_press"], _, _, _ = eb_rate(x.pressures.fillna(0), x.pass_rush_snaps.fillna(0))
    x["eb_press"] *= 100
    dlb = ["log_pick", "edge_share", "weight_z", "forty_z"]
    for lab, col in (("all", "g_d25"), ("np", "g_d25_np"), ("comb", "d25")):
        r, lo, hi, n, pv = partial_rho(x.reset_index(), col, "eb_press", dlb)
        put(f"dl_press_{lab}", r); put(f"dl_press_{lab}_lo", lo); put(f"dl_press_{lab}_hi", hi)
        put(f"dl_press_{lab}_n", n, "{:d}"); put(f"dl_press_{lab}_p", pv, "{:.3f}")

    # ------------------------------------------------------------ WR focus: Speed Realization
    w = keep["WR"].copy()
    Rw, Tw = RND[RND.grp == "WR"], TOP[TOP.grp == "WR"]
    w["n_routes"] = Rw.groupby("nfl_id").size().reindex(w.index)
    w["route_top"] = w.g_top                                    # pre-throw, route x alignment adjusted, top-3 mean
    w["route_top_all"] = player_summary(Rw, "top_all_a", "top_all")             # whole play incl. after catch
    w["route_top_untgt"] = player_summary(Rw[~Rw.target], "top_all_a", "top_all")  # whole play, untargeted routes
    w["route_top_t"] = w.t_top                                  # replication: top_epa half, pre-throw
    w["route_top_all_t"] = player_summary(Tw, "top_all_a", "top_all")
    w["route_top_arr"] = player_summary(Rw, "top_arr_a", "top_arr")             # snap to ball arrival (no YAC)
    w["route_top_arr_t"] = player_summary(Tw, "top_arr_a", "top_arr")
    for c_, s_ in (("route_top", "sr"), ("route_top_all", "sr_all"), ("route_top_untgt", "sr_untgt"),
                   ("route_top_t", "sr_t"), ("route_top_all_t", "sr_all_t"), ("route_top_arr", "sr_arr"),
                   ("route_top_arr_t", "sr_arr_t")):
        w[s_] = 100 * w[c_] / w.top
    w = w[w.route_top.notna() & (w.n_routes >= MIN_SNAPS)].copy()
    put("wr_n", len(w), "{:d}")
    put("wr_routes_med", w.n_routes.median(), "{:.0f}")
    put("wr_top40_med", w.top.median(), "{:.1f}")
    put("wr_route_top_med", w.route_top.median(), "{:.1f}")
    put("wr_sr_med", w.sr.median(), "{:.0f}")
    put("wr_sr_p10", w.sr.quantile(.1), "{:.0f}")
    put("wr_sr_p90", w.sr.quantile(.9), "{:.0f}")
    put("wr_rel_route_top", split_half(Rw[Rw.nfl_id.isin(w.index)], "top_a", "top"))
    put("wr_rel_sr", NUM["wr_rel_route_top"])
    k_ = float(w.n_routes.median())
    rr = float(NUM["wr_rel_route_top"])
    r1_ = rr / (k_ - (k_ - 1) * rr)                     # implied single-route reliability (Spearman-Brown inverse)
    put("wr_reps07", 0.7 * (1 - r1_) / (r1_ * 0.3), "{:.0f}")
    # translation: plain, partial on weight, by class, tracked top vs official 40
    r, lo, hi, n, pv = partial_rho(w.reset_index(), "top", "route_top", [])
    put("wr_tr_rho", r); put("wr_tr_lo", lo); put("wr_tr_hi", hi); put("wr_tr_n", n, "{:d}"); put("wr_tr_p", pv, "{:.4f}")
    r, lo, hi, n, pv = partial_rho(w.reset_index(), "top", "route_top", ["weight_z"])
    put("wr_trw_rho", r); put("wr_trw_lo", lo); put("wr_trw_hi", hi); put("wr_trw_p", pv, "{:.4f}")
    r, lo, hi, n, pv = partial_rho(w.reset_index(), "top", "route_top_t", ["weight_z"])
    put("wr_trt_rho", r); put("wr_trt_lo", lo); put("wr_trt_hi", hi); put("wr_trt_n", n, "{:d}")
    for lab_, col_ in (("arr", "route_top_arr"), ("arr_t", "route_top_arr_t"), ("all", "route_top_all")):
        put(f"wr_trw_{lab_}", partial_rho(w.reset_index(), "top", col_, ["weight_z"], boot=False)[0])
    r, _, _, _, _ = partial_rho(w.reset_index(), "forty_z", "route_top", ["weight_z"], boot=False)
    put("wr_tr_official", r)
    r, _, _, n_, pv_ = partial_rho(w.reset_index(), "top", "route_top", ["weight_z", "forty_z"], boot=False)
    put("wr_tr_over_official", r); put("wr_tr_over_official_p", pv_, "{:.3f}")
    for c in sorted(w.draft_year.unique()):
        s_ = w[w.draft_year == c]
        put(f"wr_tr_{c}", stats.spearmanr(s_.top, s_.route_top)[0]); put(f"wr_tr_n_{c}", len(s_), "{:d}")
    # within-class permutation p for the translation correlation
    obs = stats.spearmanr(w.top, w.route_top)[0]
    null = []
    for _ in range(NPERM):
        yy = w.route_top.copy()
        for c in w.draft_year.unique():
            m = (w.draft_year == c).values
            yy.iloc[m] = yy.iloc[m].values[RNG.permutation(m.sum())]
        null.append(stats.spearmanr(w.top, yy)[0])
    put("wr_tr_perm_p", (1 + np.sum(np.array(null) >= obs)) / (1 + NPERM), "{:.4f}")
    # out of class: calibrate route_top ~ 40 top on two classes, predict the third
    w["w1"] = 1.0
    oc = loco(w.reset_index(), [], ["top"], "route_top", perm=False)
    put("wr_oc_r2", oc["r2_aug"]); put("wr_oc_lo", oc["lo"]); put("wr_oc_hi", oc["hi"])
    for c, f in oc["folds"].items():
        put(f"wr_oc_r2_{c}", f["r2_aug"])
    oc2 = loco(w.dropna(subset=["forty_z"]).reset_index(), [], ["forty_z"], "route_top", perm=False)
    put("wr_oc_r2_official", oc2["r2_aug"])
    oc3 = loco(w.dropna(subset=["forty_z"]).reset_index(), ["forty_z"], ["top"], "route_top")
    put("wr_oc_top_over_official", oc3["delta"], "{:.3f}"); put("wr_oc_top_over_official_p", oc3["perm_p"], "{:.3f}")
    slopes = {}
    for c in sorted(w.draft_year.unique()):
        tr = w[w.draft_year != c]
        slopes[c] = float(np.polyfit(tr.top, tr.route_top, 1)[0])
        put(f"wr_slope_{c}", slopes[c])
    b_all = np.polyfit(w.top, w.route_top, 1)
    put("wr_slope", b_all[0])
    bs = [np.polyfit(*w[["top", "route_top"]].iloc[RNG.integers(0, len(w), len(w))].values.T, 1)[0] for _ in range(NBOOT)]
    put("wr_slope_lo", np.percentile(bs, 2.5)); put("wr_slope_hi", np.percentile(bs, 97.5))

    # production: EB-shrunk yards per route run (by routes), targets per route
    w["routes_c"] = w.routes.fillna(0)
    w["eb_ypr"], w["ypr_lam"], tau2, s2 = eb_rate(w.rec_yards.fillna(0), w.routes_c)
    w["eb_tpr"], _, _, _ = eb_rate(w.targets.fillna(0), w.routes_c)
    put("ypr_prior_routes", s2 / tau2, "{:.0f}")
    # outcome reliability by seasons (YPRR season t vs t+1, >= 100 routes both)
    ss = pd.read_parquet(PROC / "player_season_stats.parquet")
    sw = ss[(ss.position_group == "WR") & (ss.routes >= 100)][["nfl_id", "season", "yards_per_route"]]
    pair = sw.merge(sw.assign(season=sw.season - 1), on=["nfl_id", "season"], suffixes=("", "_next"))
    put("ypr_s2s", stats.pearsonr(pair.yards_per_route, pair.yards_per_route_next)[0]); put("ypr_s2s_n", len(pair), "{:d}")
    slot = al[al.lined_up_position == "SLOT_WR"].groupby("nfl_id").plays.sum().reindex(tot.index).fillna(0) / tot
    w["slot_share"] = slot.reindex(w.index).fillna(0)
    wb = ["log_pick", "weight_z", "forty_z"]
    defs = [("all", "sr_all", "whole play, random routes"), ("untgt", "sr_untgt", "untargeted routes only"),
            ("arr", "sr_arr", "until the ball arrives"),
            ("pre", "sr", "snap to throw (primary)"), ("pre_t", "sr_t", "snap to throw, high-|EPA| half"),
            ("all_t", "sr_all_t", "whole play, high-|EPA| half")]
    prod = []
    for lab, col, desc in defs:
        r, lo, hi, n, pv = partial_rho(w.reset_index(), col, "eb_ypr", wb)
        rt, _, _, _, pvt = partial_rho(w.reset_index(), col, "eb_tpr", wb, boot=False)
        prod.append(dict(lab=lab, desc=desc, rho=r, lo=lo, hi=hi, n=n, p=pv, rho_tpr=rt, p_tpr=pvt, mde=mde(n),
                         tost=tost(r, n)))
        for kk, v in prod[-1].items():
            if kk in ("lab", "desc"):
                continue
            put(f"prod_{lab}_{kk}", v, "{:d}" if kk == "n" else ("{:.3f}" if kk.startswith(("p", "tost")) else "{:.2f}"))
    PROD = pd.DataFrame(prod)
    # where does the contamination come from? share of whole-play top-3 snaps that were targets
    Rw2 = Rw[Rw.nfl_id.isin(w.index)].copy()
    Rw2["rk"] = Rw2.groupby("nfl_id").top_all_a.rank(ascending=False, method="first")
    put("tgt_share_top3_all", 100 * Rw2[Rw2.rk <= TOPK].target.mean(), "{:.0f}")
    Rw2["rk2"] = Rw2.groupby("nfl_id").top_a.rank(ascending=False, method="first")
    put("tgt_share_top3_pre", 100 * Rw2[Rw2.rk2 <= TOPK].target.mean(), "{:.0f}")
    put("tgt_share_all", 100 * Rw2.target.mean(), "{:.0f}")

    # out-of-class gain on YPRR (LOCO, reliability-weighted)
    wl = w.dropna(subset=wb + ["ngs_z", "sr", "sr_all", "eb_ypr"]).reset_index().copy()
    wl["w"] = wl.ypr_lam
    base_full = wb + ["ngs_z", "slot_share"]
    LO = {}
    for lab, extra, base in (("all", ["sr_all"], base_full), ("pre", ["sr"], base_full), ("pre_noslot", ["sr"], wb + ["ngs_z"]),
                             ("untgt", ["sr_untgt"], base_full)):
        LO[lab] = loco(wl.dropna(subset=extra), base, extra, "eb_ypr", w="w")
        r_ = LO[lab]
        for kk in ("delta", "lo", "hi", "r2_base", "r2_aug", "perm_p"):
            put(f"loco_{lab}_{kk}", r_[kk], "{:.3f}")
        put(f"loco_{lab}_n", r_["n"], "{:d}"); put(f"loco_{lab}_folds_pos", r_["folds_pos"], "{:d}")
        for c, f in r_["folds"].items():
            put(f"loco_{lab}_{c}_d", f["r2_aug"] - f["r2_base"], "{:.3f}")
            put(f"loco_{lab}_{c}_base", f["r2_base"], "{:.3f}")
    pk = loco(wl, ["log_pick"], ["slot_share"], "eb_ypr", w="w", perm=False)
    put("loco_pick_only", pk["r2_base"], "{:.3f}")

    # multiplicity across every inferential test reported
    pv = {f"grid_{r.group}_{r.phase}": r.p for r in GRID.itertuples()}
    pv.update({f"prod_{r.lab}": r.p for r in PROD.itertuples()})
    pv.update({"dl_press_all": float(NUM["dl_press_all_p"]), "dl_press_np": float(NUM["dl_press_np_p"]),
               "wr_tr_over_official": float(NUM["wr_tr_over_official_p"]),
               "loco_all": LO["all"]["perm_p"], "loco_pre": LO["pre"]["perm_p"], "loco_untgt": LO["untgt"]["perm_p"]})
    MT = holm_bh(pv)
    put("n_tests", len(MT), "{:d}")
    for k_, v in MT.items():
        put(f"holm_{k_}", v["holm"], "{:.3f}"); put(f"bh_{k_}", v["bh"], "{:.3f}")
    put("n_survive_holm", sum(v["holm"] < 0.05 for v in MT.values()), "{:d}")
    survivors = sorted([k_ for k_, v in MT.items() if v["holm"] < 0.05])

    # ------------------------------------------------------------ figure 1: hero WR cards
    elig = w[w.n_routes >= 12].sort_values("sr", ascending=False)
    put("card_min_routes", 12, "{:d}"); put("card_pool", len(elig), "{:d}")
    picks = list(elig.head(3).index) + list(elig.tail(3).index)[::-1]
    tags = ["plays to his 40"] * 3 + ["plays below his 40"] * 3
    tt = np.arange(T) * DT
    pop40 = C40.merge(p[["nfl_id", "position_group"]], on="nfl_id")
    pop40 = pop40[pop40.position_group == "WR"][list(range(T))].mean().values
    fig, axes = plt.subplots(2, 3, figsize=(FIGW, 8.0), sharex=True, sharey=True)
    cards = []
    wr_sd_top, wr_sd_route = w.top.std(), w.route_top.std()
    for i, (ax, nid, tag) in enumerate(zip(axes.ravel(), picks, tags)):
        r = w.loc[nid]
        curv = Gc[(Gc.nfl_id == nid) & (Gc.sample_kind == "random")]
        for _, row in curv.iterrows():
            ax.plot(tt, row[list(range(T))].values.astype(float), color=OI["blue"], alpha=.22, lw=1)
        c_ = C40[C40.nfl_id == nid][list(range(T))].mean().values
        ax.plot(tt, c_, "--", color=OI["black"], lw=2.4, label="his Combine 40")
        ax.axhline(r.top, color=OI["black"], lw=1, ls=":")
        ax.axhline(r.route_top, color=OI["blue"], lw=2.2)
        ax.fill_between([3.55, 4.0], r.route_top, r.top, color=OI["orange"], alpha=.45, lw=0)
        ax.set_ylim(0, 15.5)
        ax.set_xlim(0, 4)
        ax.set_title(f"{r.display_name} (pick {int(r.pick)}): {tag}", fontsize=11)
        ax.text(0.97, 0.05, f"40 top {r.top:.1f} yd/s\nroutes top-3 {r.route_top:.1f}\nRealization {r.sr:.0f}%\n"
                f"{int(r.n_routes)} random routes\nYPRR {r.eb_ypr:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, bbox=dict(facecolor="white", edgecolor=OI["grey"], alpha=.92, boxstyle="round,pad=0.3"))
        # mini z-bars
        ins = ax.inset_axes([0.17, 0.865, 0.36, 0.11])
        zz = [(r.top - w.top.mean()) / wr_sd_top, (r.route_top - w.route_top.mean()) / wr_sd_route]
        ins.barh([1, 0], zz, color=[OI["black"], OI["blue"]], height=.7)
        ins.axvline(0, color=OI["grey"], lw=.8)
        ins.set_xlim(-3, 3)
        ins.set_yticks([1, 0], ["40 z", "NFL z"], fontsize=9)
        ins.tick_params(axis="x", labelsize=9)
        for s_ in ("top", "right"):
            ins.spines[s_].set_visible(False)
        cards.append(dict(name=r.display_name, pick=int(r.pick), top40=round(r.top, 2), route_top=round(r.route_top, 2),
                          sr=round(r.sr, 1), routes=int(r.n_routes), ypr=round(r.eb_ypr, 2)))
        put(f"card{i}_sr", r.sr, "{:.0f}"); put(f"card{i}_top", r.top, "{:.1f}"); put(f"card{i}_rt", r.route_top, "{:.1f}")
        put(f"card{i}_pick", int(r.pick), "{:d}"); put(f"card{i}_ypr", r.eb_ypr); put(f"card{i}_n", int(r.n_routes), "{:d}")
    for ax in axes[1]:
        ax.set_xlabel("seconds after first movement")
    for ax in axes[:, 0]:
        ax.set_ylabel("speed (yd/s)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hand = [Line2D([], [], color=OI["black"], ls="--", lw=2.4, label="his Combine 40 (dotted = its top speed)"),
            Line2D([], [], color=OI["blue"], lw=1, alpha=.5, label="each random NFL route, snap to throw"),
            Line2D([], [], color=OI["blue"], lw=2.2, label="NFL top end (mean of 3 fastest routes)"),
            Patch(color=OI["orange"], alpha=.45, label="gap between 40 top speed and NFL top end")]
    fig.legend(handles=hand, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f"Speed Realization: the 3 highest and 3 lowest of {len(elig)} WRs with 12+ random routes "
                 f"(bars: 40 top speed and NFL top end as z-scores)", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    save(fig, "fig1_wr_cards.png")

    # ------------------------------------------------------------ figure 2: translation map (groups x phases)
    fig, axes = plt.subplots(1, 3, figsize=(FIGW, 4.8), sharey=True)
    order = GROUPS[::-1]
    for ax, (ph, lab) in zip(axes, PHASES):
        for yv, g in enumerate(order):
            r = GRID[(GRID.group == g) & (GRID.phase == ph)].iloc[0]
            ax.axvspan(-r.mde, r.mde, ymin=(yv + .1) / len(order), ymax=(yv + .9) / len(order), color=OI["grey"], alpha=.13, lw=0)
            ax.plot([r.lo, r.hi], [yv, yv], color=GCOL[g], lw=3)
            ax.plot(r.rho, yv, "o", color=GCOL[g], ms=10, mec="k", mew=.6)
            ax.plot(r.rho_raw, yv, "x", color=OI["black"], ms=8, mew=1.6)
            ax.plot(r.ceiling, yv, "|", color=OI["black"], ms=16, mew=2.5)
            ax.text(1.0, yv, f"n={int(r.n)}", fontsize=9.5, va="center", ha="right")
        ax.axvline(0, color=OI["black"], lw=.8)
        ax.set_xlim(-0.7, 1.0)
        ax.set_title(lab)
        ax.set_xlabel("ρ, Combine 40 vs NFL snaps")
    axes[0].set_yticks(range(len(order)), [f"{g} ({GAME_TYPE[g]})" for g in order])
    hand = [Line2D([], [], marker="o", color=OI["grey"], ls="", ms=9, mec="k", label="adjusted for situation and weight (95% CI)"),
            Line2D([], [], marker="x", color="k", ls="", ms=8, label="raw, no adjustment"),
            Line2D([], [], marker="|", color="k", ls="", ms=14, mew=2.5, label="noise ceiling"),
            Patch(color=OI["grey"], alpha=.2, label="too small to detect at this n (80% power)")]
    fig.legend(handles=hand, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("The 40 reappears only in top speed (WR, DB); the DL's raw link is size and alignment", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save(fig, "fig2_translation_map.png")

    # ------------------------------------------------------------ figure 3: WR 40 top speed vs NFL route top end
    fig, ax = plt.subplots(figsize=(FIGW, 5.4))
    mk = {2023: "o", 2024: "s", 2025: "^"}
    cl = {2023: OI["blue"], 2024: OI["orange"], 2025: OI["green"]}
    xs = np.linspace(w.top.min(), w.top.max(), 50)
    band = np.array([np.polyval(np.polyfit(*w[["top", "route_top"]].iloc[RNG.integers(0, len(w), len(w))].values.T, 1), xs)
                     for _ in range(1000)])
    ax.fill_between(xs, *np.percentile(band, [2.5, 97.5], axis=0), color=OI["grey"], alpha=.22, lw=0, label="fit, 95% bootstrap band")
    ax.plot(xs, np.polyval(b_all, xs), color=OI["black"], lw=2)
    for c in sorted(w.draft_year.unique()):
        s_ = w[w.draft_year == c]
        ax.scatter(s_.top, s_.route_top, marker=mk[c], color=cl[c], s=60, edgecolor="k", lw=.4,
                   label=f"{c} class: ρ = {float(NUM[f'wr_tr_{c}']):.2f} (n = {len(s_)})")
    for nid in picks:
        r = w.loc[nid]
        ax.annotate(r.display_name, (r.top, r.route_top), xytext=(5, 4), textcoords="offset points", fontsize=9.5)
    ax.set_xlabel("Combine 40 top speed (yd/s, tracked)")
    ax.set_ylabel("NFL route top end (yd/s)")
    ax.set_title(f"WR: each 1 yd/s of 40 top speed is worth {b_all[0]:.2f} yd/s on NFL routes "
                 f"(n = {len(w)} WRs, {int(Rw[Rw.nfl_id.isin(w.index)].shape[0]):,} random routes)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_wr_translation.png")
    put("wr_n_routes_total", int(Rw[Rw.nfl_id.isin(w.index)].shape[0]), "{:,}")

    # ------------------------------------------------------------ figure 4: the ball trap (production link by definition)
    fig, ax = plt.subplots(figsize=(FIGW, 4.4))
    yp = np.arange(len(PROD))[::-1]
    for yv, r in zip(yp, PROD.itertuples()):
        col = OI["red"] if r.lab.startswith("all") else OI["blue"]
        ax.axvspan(-r.mde, r.mde, ymin=(yv + .15) / len(PROD), ymax=(yv + .85) / len(PROD), color=OI["grey"], alpha=.13, lw=0)
        ax.plot([r.lo, r.hi], [yv, yv], color=col, lw=3)
        ax.plot(r.rho, yv, "D", color=col, ms=9, mec="k", mew=.6)
        ax.text(0.80, yv, f"ρ = {r.rho:.2f} (n = {r.n})", fontsize=10, va="center")
    ax.axvline(0, color=OI["black"], lw=.8)
    ax.set_yticks(yp, list(PROD.desc))
    ax.set_xlim(-0.6, 1.05)
    ax.set_xlabel("partial ρ, Speed Realization vs shrunk yards per route run (beyond pick, weight, official 40; 95% CI)")
    ax.set_title("Realization predicts production only when the top speed includes running after the catch")
    hand = [Line2D([], [], marker="D", color=OI["red"], ls="", ms=8, label="includes ball-in-hand running"),
            Line2D([], [], marker="D", color=OI["blue"], ls="", ms=8, label="route running only"),
            Patch(color=OI["grey"], alpha=.2, label="undetectable at this n (80% power)")]
    fig.legend(handles=hand, frameon=False, loc="lower center", ncol=3, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig4_ball_trap.png")

    # ------------------------------------------------------------ figure 5: blind draft-room card (train 2023-24, apply to 2025)
    tr, te = w[w.draft_year < 2025], w[w.draft_year == 2025].copy()
    b = np.polyfit(tr.top, tr.route_top, 1)
    res = tr.route_top - np.polyval(b, tr.top)
    sd_res = res.std(ddof=2)
    te["pred"] = np.polyval(b, te.top)
    q = stats.t.ppf(0.9, len(tr) - 2)
    te["lo"], te["hi"] = te.pred - q * sd_res, te.pred + q * sd_res
    te["flag"] = np.where(te.route_top < te.lo, "below", np.where(te.route_top > te.hi, "above", "inside"))
    put("card25_n", len(te), "{:d}"); put("card25_n_train", len(tr), "{:d}")
    put("card25_inside", 100 * (te.flag == "inside").mean(), "{:.0f}")
    put("card25_rho", stats.spearmanr(te.pred, te.route_top)[0])
    put("card25_pi_halfwidth", q * sd_res)
    te = te.sort_values("pred")
    fig, ax = plt.subplots(figsize=(FIGW, 6.6))
    yv = np.arange(len(te))
    ax.hlines(yv, te.lo, te.hi, color=OI["grey"], lw=5, alpha=.45, label="80% band from his 40 (fit on 2023-24)")
    ax.plot(te.pred, yv, "|", color=OI["black"], ms=12, mew=2, label="projection from his 40")
    colmap = {"below": OI["orange"], "above": OI["sky"], "inside": OI["blue"]}
    for f_, lab in (("inside", "NFL top end inside band"), ("below", "below band: check film"), ("above", "above band")):
        s_ = te[te.flag == f_]
        ax.scatter(s_.route_top, yv[te.flag.values == f_], color=colmap[f_], s=55, edgecolor="k", lw=.5, zorder=3, label=lab)
    ax.set_yticks(yv, [f"{n} ({int(pk_)}; {int(k_)} rts)" for n, pk_, k_ in zip(te.display_name, te.pick, te.n_routes)], fontsize=9)
    ax.set_xlabel("NFL route top end (yd/s)")
    ax.set_title(f"Blind test on the 2025 WR class: {NUM['card25_inside']}% land inside the band projected from the 40")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="lower center", ncol=3, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig5_draft_card.png")
    card25 = te[["display_name", "pick", "top", "pred", "lo", "hi", "route_top", "flag", "n_routes"]].round(2).to_dict("records")
    for f_ in ("below", "above"):
        s_ = te[te.flag == f_].sort_values("pick")
        put(f"card25_{f_}_names", ", ".join(f"{n} ({int(k)})" for n, k in zip(s_.display_name, s_.pick)), "{}")
        put(f"card25_{f_}_n", len(s_), "{:d}")

    # ------------------------------------------------------------ table rows for the writeup
    T1 = GRID.round(3).to_dict("records")
    for g in GROUPS:
        r = GRID[(GRID.group == g) & (GRID.phase == "top")].iloc[0]
        put(f"t1_{g}_top", r.rho); put(f"t1_{g}_rel", r.rel_g)
    (HERE / "tables.json").write_text(json.dumps(dict(grid=T1, prod=PROD.round(3).to_dict("records"), card25=card25,
                                                      cards=cards, survivors=survivors), indent=1, default=float))
    out = dict(NUM)
    out["_abs_values"] = sorted({v.lstrip("-+") for v in NUM.values()})
    out["_multiplicity"] = MT
    (HERE / "numbers.json").write_text(json.dumps(out, indent=1, default=float))
    log(GRID.round(2).to_string())
    log(PROD.round(3).to_string())
    log(json.dumps(cards, indent=0))
    log(json.dumps({k_: v for k_, v in NUM.items() if not k_.startswith(("grid_", "holm_", "bh_", "card"))}, indent=0))
    log("survivors " + str(survivors))


if __name__ == "__main__":
    main()
