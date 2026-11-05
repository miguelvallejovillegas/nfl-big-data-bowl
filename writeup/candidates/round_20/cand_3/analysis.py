"""Movement translation: does a player move on Sundays the way he moved in the matching Combine drill, and does that
match ("Carry-over") explain NFL production beyond draft slot?

Run from the repo root:
    uv run python writeup/candidates/round_20/cand_3/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (every value quoted in writeup.md), claims.json,
players.csv, figures/*.png, and renders writeup.md from writeup_template.md. Frame-level curve extraction is cached
in data/cache/r20c3 (gitignored). Runtime about 1-2 minutes on one core.

Design (fixed before any outcome was looked at, except where the text says "screened")
  * Movement = the first 1.5 s of a start from a stance, as speed and acceleration every 0.1 s from first movement
    (smoothed speed first reaches 1 yd/s, minus 0.2 s). The Combine side is the position drill that starts from the
    same stance as the NFL snap: DL pass-rush drill, OL pass-set ("pass rush drop") drill, WR go route, TE in route,
    DB backpedal-and-break; the 40 is a second, straight-line drill for every group. The NFL side is every *random*
    sampled pass snap (game_frames_sample, sample_kind == random) on which he is set at the snap; the top-EPA half
    of the sample is never used, because it over-represents pressures and big plays.
  * Both curves are year-standardized (Combine) and expressed in sd after weight and NFL role (EDGE share, tackle
    share, slot share, inline share, outside-CB share), so "fast" means fast for his size and job.
  * Carry-over = minus the mean absolute gap between his Combine and Sunday curves over the 31 points (16 speed,
    15 acceleration), in sd. 0 = he moves on Sundays exactly as he tested; more negative = less alike.
  * Production: DL career pressure rate (pressures per pass rush, 100+ rushes, shrunk toward the mean); OL minus
    pressure rate allowed (100+ pass blocks); WR/TE yards per route (100+ routes); DB solo tackles per 100 snaps.
  * Baseline without tracking: log draft pick (UDFA = 262), weight, role, official 40 time and NGS athleticism score
    (each year-standardized; missing -> 0 with a flag). Out of class: fit on two draft classes, score the third.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import time as _time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
PROC = ROOT / "data" / "processed"
FIG = OUT / "figures"
CACHE = ROOT / "data" / "cache" / "r20c3"
FIG.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
for old in FIG.glob("*.png"):
    old.unlink()

RNG = np.random.default_rng(2027)
NBOOT = int(os.environ.get("BDB_NBOOT", 1000))
UDFA_PICK = 262
TW = 16                      # 0.0-1.5 s from first movement
FS = [3, 4, 5]               # first step: 0.3-0.5 s
MIN_SNAPS = 8                # random NFL starts per player
MIN_EXPO = 100               # rushes / pass blocks / routes for a production rate
MARGIN = 1.0                 # equivalence margin, pressure-rate points per sd
DRILL = {"DL": "PASS_RUSH_DRILL", "OL": "PASS_RUSH_DROP", "WR": "GO_ROUTE_RIGHT", "TE": "IN_ROUTE_LEFT",
         "DB": "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL"}
DRILL_LABEL = {"PASS_RUSH_DRILL": "pass-rush drill", "PASS_RUSH_DROP": "pass-set drill", "GO_ROUTE_RIGHT": "go route",
               "IN_ROUTE_LEFT": "in route", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL": "backpedal-and-break",
               "FORTY_YARD_DASH": "40-yard dash"}
ROLE = {"DL": ["EDGE", "OLB"], "OL": ["T"], "WR": ["SLOT_WR"], "TE": ["TE"], "DB": ["CB", "SLOT_CB"]}
PROD_LABEL = {"DL": "pressure rate", "OL": "pressure rate allowed (sign flipped)", "WR": "yards per route",
              "TE": "yards per route", "DB": "solo tackles per 100 snaps"}
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "grey": "#8C8C8C", "black": "#000000"}
plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "legend.fontsize": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
FIGW = 1600 / 150
T0 = _time.time()
NUM: dict = {}


def tick(s):
    print(f"[{_time.time() - T0:6.1f}s] {s}", flush=True)


def fmt(v, nd=2):
    s = f"{float(v):.{nd}f}"
    if re.fullmatch(r"-0\.?0*", s):
        s = s[1:]
    return s


def put(key, v, nd=2):
    if key in NUM:
        raise KeyError(f"duplicate key {key}")
    if isinstance(v, (int, np.integer)):
        NUM[key] = int(v)
    elif isinstance(v, str):
        NUM[key] = v
    else:
        NUM[key] = fmt(v, nd)
    return NUM[key]


def save(fig, name):
    fig.set_size_inches(FIGW, fig.get_size_inches()[1])
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def onset(sm, start):
    hit = np.flatnonzero(sm[start:] >= 1.0)
    return None if not len(hit) else start + max(int(hit[0]) - 2, 0)


def _res(v, A):
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]


def partial(df, x, y, Z, nb=NBOOT):
    """Spearman partial correlation (Pearson r of rank residuals) with a 90% player bootstrap interval."""
    d = df[[x, y] + Z].dropna()
    R = d.rank().to_numpy(float)
    A = np.c_[np.ones(len(R)), R[:, 2:]]
    r = float(np.corrcoef(_res(R[:, 0], A), _res(R[:, 1], A))[0, 1])
    n = len(d)
    b = np.empty(nb)
    for k in range(nb):
        i = RNG.integers(0, n, n)
        b[k] = np.corrcoef(_res(R[i, 0], A[i]), _res(R[i, 1], A[i]))[0, 1]
    lo, hi = np.nanpercentile(b, [5, 95])
    return r, float(lo), float(hi), n


def perm_p(df, x, y, Z, nperm=2000, by="draft_year"):
    """Two-sided permutation p for the partial Spearman, x shuffled within draft class."""
    d = df[[x, y] + Z + [by]].dropna()
    R = d[[x, y] + Z].rank().to_numpy(float)
    A = np.c_[np.ones(len(R)), R[:, 2:]]
    ry = _res(R[:, 1], A)
    obs = abs(np.corrcoef(_res(R[:, 0], A), ry)[0, 1])
    grp = d[by].to_numpy()
    cnt = 0
    for _ in range(nperm):
        xx = R[:, 0].copy()
        for c_ in np.unique(grp):
            ii = np.flatnonzero(grp == c_)
            xx[ii] = xx[RNG.permutation(ii)]
        cnt += abs(np.corrcoef(_res(xx, A), ry)[0, 1]) >= obs
    return (cnt + 1) / (nperm + 1)


def sb(r, k):
    """Spearman-Brown step-up of a reliability r to k times the data."""
    return k * r / (1 + (k - 1) * r)


# ============================================================================ load
tick("load")
players = pd.read_parquet(PROC / "players.parquet")
cres = pd.read_parquet(PROC / "combine_results.parquet")
events = pd.read_parquet(PROC / "combine_events.parquet")
seasons = pd.read_parquet(PROC / "player_season_stats.parquet")
align = pd.read_parquet(PROC / "player_season_alignment.parquet")
gp = pd.read_parquet(PROC / "game_plays_sample.parquet")
pv = pd.read_parquet(PROC / "player_value.parquet")

pl = players[["nfl_id", "display_name", "draft_year", "position_group", "draft_overall_pick", "draft_round",
              "latest_team"]].merge(cres[["nfl_id", "combine_weight", "forty", "ngs_athleticism_score"]], on="nfl_id",
                                    how="left")
pl["pick"] = pl.draft_overall_pick.fillna(UDFA_PICK)
pl["lp"] = np.log(pl.pick)
pl["w"] = pl.combine_weight.astype(float)
for c, sign in [("forty", -1), ("ngs_athleticism_score", 1)]:
    z = sign * pl.groupby(["position_group", "draft_year"])[c].transform(lambda v: (v - v.mean()) / v.std())
    pl[c + "_miss"] = z.isna().astype(float)
    pl[c + "_z"] = z.fillna(0.0)

# role share per player from NFL alignment counts
al = align.merge(pl[["nfl_id", "position_group"]], on="nfl_id")
al["in_role"] = [p if lp in ROLE[g] else 0 for p, lp, g in zip(al.plays, al.lined_up_position, al.position_group)]
rs = al.groupby("nfl_id").agg(r=("in_role", "sum"), p=("plays", "sum"))
pl = pl.merge((rs.r / rs.p).rename("role").reset_index(), on="nfl_id", how="left")
pl["role"] = pl.role.fillna(pl.groupby("position_group").role.transform("mean"))

# production per player (career, regular season aggregates)
s = seasons.copy()
s["pp_w"] = s.avg_peak_pressure_prob * s.pass_blocks
ag = s.groupby("nfl_id").agg(rush=("pass_rush_snaps", "sum"), prs=("pressures", "sum"), qp=("quick_pressures", "sum"),
                             blk=("pass_blocks", "sum"), pa=("pressures_allowed", "sum"),
                             routes=("routes", "sum"), yds=("rec_yards", "sum")).reset_index()
pl = pl.merge(ag, on="nfl_id", how="left")
pl = pl.merge(pv[pv.target == "production"][["nfl_id", "actual"]].rename(columns={"actual": "db_prod"}),
              on="nfl_id", how="left")


def eb_rate(k, n, min_n=20):
    """Beta-binomial (method of moments) shrinkage of k/n toward the pooled mean, in percentage points."""
    m = n >= min_n
    nn, kk = n[m], k[m]
    mu = kk.sum() / nn.sum()
    N = nn.sum()
    tau2 = ((nn * (kk / nn - mu) ** 2).sum() - (len(nn) - 1) * mu * (1 - mu)) / (N - (nn ** 2).sum() / N)
    M = mu * (1 - mu) / max(tau2, 1e-6) - 1
    return 100 * (k + M * mu) / (n + M), M, mu


dlm = pl.position_group == "DL"
pl.loc[dlm, "y"], M_DL, MU_DL = eb_rate(pl.loc[dlm, "prs"].fillna(0), pl.loc[dlm, "rush"].fillna(0))
pl.loc[dlm, "expo"] = pl.loc[dlm, "rush"]
olm = pl.position_group == "OL"
pl.loc[olm, "y"] = -eb_rate(pl.loc[olm, "pa"].fillna(0), pl.loc[olm, "blk"].fillna(0))[0]
pl.loc[olm, "expo"] = pl.loc[olm, "blk"]
rcm = pl.position_group.isin(["WR", "TE"])
pl.loc[rcm, "y"] = pl.loc[rcm, "yds"] / pl.loc[rcm, "routes"]
pl.loc[rcm, "expo"] = pl.loc[rcm, "routes"]
dbm = pl.position_group == "DB"
pl.loc[dbm, "y"] = pl.loc[dbm, "db_prod"]
pl.loc[dbm, "expo"] = np.where(pl.loc[dbm, "db_prod"].notna(), 1e4, 0)
pl.loc[pl.expo.fillna(0) < MIN_EXPO, "y"] = np.nan
put("dl_mean_prate", 100 * MU_DL, 1)

# ============================================================================ curves (cached)
K = ["game_id", "play_id", "nfl_id"]


def extract():
    names = set(DRILL.values()) | {"FORTY_YARD_DASH"}
    ev = events[events.drill_name.isin(names)]
    ev = ev[(ev.drill_name == "FORTY_YARD_DASH") | (ev.drill_name == ev.position_group.map(DRILL))]
    fr = pd.read_parquet(PROC / "combine_frames.parquet", filters=[("event_id", "in", list(ev.event_id))],
                         columns=["event_id", "frame", "entity_type", "s"])
    fr = fr[fr.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
    crv = {}
    for eid, g in fr.groupby("event_id", sort=False):
        sp = g.s.to_numpy()
        if len(sp) < 20:
            continue
        sm = savgol_filter(sp, 5, 2)
        rest = np.flatnonzero(sm < 0.8)
        if not len(rest) or rest[0] > 20:
            continue
        o = onset(sm, rest[0])
        if o is None or o + TW > len(sm):
            continue
        crv[eid] = np.clip(sm[o:o + TW], 0, None)
    DC = pd.DataFrame({"event_id": list(crv), "curve": list(crv.values())}).merge(
        ev[["event_id", "nfl_id", "draft_year", "drill_name", "position_group", "attempt_no"]], on="event_id")
    g = gp[gp.snap_frame.notna() & gp.pass_result.notna() & (gp.sample_kind == "random")]
    gf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "s"])
    gf = gf.merge(g[K + ["snap_frame"]], on=K).sort_values(K + ["frame"])
    rows = []
    for k, h in gf.groupby(K, sort=False):
        sf = h.snap_frame.iloc[0]
        sp, f_ = h.s.to_numpy(), h.frame.to_numpy()
        i0 = int(np.searchsorted(f_, sf))
        if len(sp) < i0 + TW + 3:
            continue
        sm = savgol_filter(sp, 5, 2)
        if sm[i0] >= 0.8:
            continue
        o = onset(sm, i0)
        if o is None or o + TW > len(sm) or o - i0 > 10:
            continue
        rows.append(dict(zip(K, k), curve=np.clip(sm[o:o + TW], 0, None)))
    GC = pd.DataFrame(rows).merge(g[K + ["season", "lined_up_position", "player_get_off", "time_to_pressure",
                                         "sack"]], on=K)
    return DC, GC


cf = CACHE / "curves_v1.pkl"
if cf.exists():
    DC, GC = pickle.load(open(cf, "rb"))
else:
    tick("extract curves from frames")
    DC, GC = extract()
    pickle.dump((DC, GC), open(cf, "wb"))
GC = GC.merge(pl[["nfl_id", "position_group"]], on="nfl_id")
SV = [f"v{t}" for t in range(TW)]
AV = [f"a{t}" for t in range(TW - 1)]
PROF = SV + AV
for D in (DC, GC):
    M = np.vstack(D.curve.to_numpy())
    A = np.diff(M, axis=1) * 10
    for t in range(TW):
        D[f"v{t}"] = M[:, t]
    for t in range(TW - 1):
        D[f"a{t}"] = A[:, t]
    D["fs"] = M[:, FS].mean(1)
    D["mean_v"] = M.mean(1)
# NFL: alignment adjustment per snap within group (a tackle's set is not a guard's)
for c in PROF + ["fs", "mean_v"]:
    GC[c + "_adj"] = GC[c] - GC.groupby(["position_group", "lined_up_position"])[c].transform("mean") + \
        GC.groupby("position_group")[c].transform("mean")
put("n_snaps_all", len(GC), 0)
put("n_drill_reps_all", len(DC), 0)


# ============================================================================ per-group tables
def resid_z(T, cols, Z):
    A = np.column_stack([np.ones(len(T))] + [T[z].to_numpy(float) for z in Z])
    for c in cols:
        v = T[c].to_numpy(float)
        r = _res(v, A)
        T[c] = r / r.std()


def build(grp, drill):
    g = GC[GC.position_group == grp]
    q = g.groupby("nfl_id").agg(n_g=("v0", "size"), **{c: (c + "_adj", "mean") for c in PROF + ["fs", "mean_v"]})
    q = q[q.n_g >= MIN_SNAPS]
    d = DC[(DC.position_group == grp) & (DC.drill_name == drill)]
    dm = d.groupby(["nfl_id", "draft_year"]).agg(n_c=("v0", "size"), **{c: (c, "mean") for c in PROF + ["fs", "mean_v"]})
    dm = dm.reset_index()
    dm["fs_raw"], dm["mean_v_raw"] = dm.fs, dm.mean_v
    for c in PROF + ["fs", "mean_v"]:
        dm[c] = dm.groupby("draft_year")[c].transform(lambda v: (v - v.mean()) / v.std())
    T = pl[pl.position_group == grp].merge(dm.drop(columns="draft_year").set_index("nfl_id").add_prefix("c_"),
                                           left_on="nfl_id", right_index=True)
    T = T.merge(q.add_prefix("g_"), left_on="nfl_id", right_index=True)
    T["g_fs_raw"], T["g_mean_v_raw"] = T.g_fs, T.g_mean_v
    cols = [f"c_{c}" for c in PROF + ["fs", "mean_v"]] + [f"g_{c}" for c in PROF + ["fs", "mean_v"]]
    resid_z(T, cols, ["w", "role"])
    Cz, Gz = T[[f"c_{c}" for c in PROF]].to_numpy(), T[[f"g_{c}" for c in PROF]].to_numpy()
    T["carry"] = -np.abs(Cz - Gz).mean(1)
    T["carry_speed"] = -np.abs(Cz[:, :TW] - Gz[:, :TW]).mean(1)
    T["carry_shape"] = [np.corrcoef(a, b)[0, 1] for a, b in zip(Cz, Gz)]
    T["gap_fs"] = T.g_fs - T.c_fs
    T["carry_ratio"] = 100 * T.g_mean_v_raw / T.c_mean_v_raw
    return T.reset_index(drop=True), d, g[g.nfl_id.isin(q.index)]


def reliab(d, g, col_c="mean_v", col_g="mean_v_adj"):
    """Drill: first rep vs last rep. Sunday: odd vs even random snaps, stepped up to all snaps."""
    d = d.sort_values(["nfl_id", "attempt_no"])
    f = d.groupby("nfl_id").head(1).set_index("nfl_id")[col_c]
    l_ = d[d.groupby("nfl_id").nfl_id.transform("size") >= 2].groupby("nfl_id").tail(1).set_index("nfl_id")[col_c]
    rc = stats.spearmanr(f.loc[l_.index], l_)[0] if len(l_) > 10 else np.nan
    reps = d.groupby("nfl_id").size().mean()
    gg = g.copy()
    gg["h"] = gg.groupby("nfl_id").cumcount() % 2
    h = gg.groupby(["nfl_id", "h"])[col_g].mean().unstack().dropna()
    rg = sb(max(stats.spearmanr(h[0], h[1])[0], 0), 2)
    return (sb(max(rc, 0), reps) if not np.isnan(rc) else np.nan), rg, len(l_)


tick("screen")
SCREEN = []
TAB = {}
for grp in ["DL", "OL", "WR", "TE", "DB"]:
    for drill in [DRILL[grp], "FORTY_YARD_DASH"]:
        T, d, g = build(grp, drill)
        TAB[(grp, drill)] = T
        rc, rg, nrep = reliab(d, g)
        tr = partial(T, "c_mean_v", "g_mean_v", [])
        Z = ["lp", "w", "role"]
        py = partial(T, "carry", "y", Z)
        p_ = perm_p(T, "carry", "y", Z, nperm=1000)
        SCREEN.append(dict(grp=grp, drill=drill, n=len(T), rel_c=rc, rel_g=rg, nrep=nrep,
                           ceil=np.sqrt(max(rc, 0) * rg) if not np.isnan(rc) else np.nan,
                           tr=tr[0], tr_lo=tr[1], tr_hi=tr[2], pay=py[0], pay_lo=py[1], pay_hi=py[2], n_y=py[3],
                           p=p_, matched=drill != "FORTY_YARD_DASH"))
SCR = pd.DataFrame(SCREEN)
print(SCR.round(2).to_string())
SCR.round(3).to_csv(OUT / "screen.csv", index=False)
m = SCR[SCR.matched & (SCR.grp != "TE")]
put("scr_tr_max", m.tr.max())
put("scr_tr_min", m.tr.min())
put("scr_n_cells", len(SCR), 0)
put("scr_n_sig", int((SCR.p < 0.05).sum()), 0)
put("scr_pmin", SCR.p.min())
put("scr_pay_maxabs", SCR[SCR.grp != "TE"].pay.abs().max())
for _, r in SCR.iterrows():
    tag = f"{r.grp.lower()}_{'m' if r.matched else '40'}"
    put(f"scr_{tag}_n", int(r.n), 0)
    put(f"scr_{tag}_tr", r.tr)
    put(f"scr_{tag}_pay", r.pay)
    put(f"scr_{tag}_ny", int(r.n_y), 0)
    put(f"scr_{tag}_relc", r.rel_c if not np.isnan(r.rel_c) else 0.0)
    put(f"scr_{tag}_relg", r.rel_g)
    put(f"scr_{tag}_ceil", r.ceil if not np.isnan(r.ceil) else 0.0)
    put(f"scr_{tag}_p", r.p)

# ============================================================================ DL deep dive (matched drill)
tick("DL")
DLT = TAB[("DL", DRILL["DL"])].copy()
D40 = TAB[("DL", "FORTY_YARD_DASH")]
DLT = DLT.merge(D40[["nfl_id", "carry", "c_fs", "c_mean_v"]].rename(
    columns={"carry": "carry40", "c_fs": "c40_fs", "c_mean_v": "c40_mean_v"}), on="nfl_id", how="left")
pool = DLT[DLT.y.notna()].copy()
put("dl_n", len(DLT), 0)
put("dl_pool", len(pool), 0)
put("dl_pool40", int(pool.carry40.notna().sum()), 0)
gdl = GC[(GC.position_group == "DL") & GC.nfl_id.isin(DLT.nfl_id)]
put("dl_snaps", len(gdl), 0)
put("dl_snaps_med", int(DLT.g_n_g.median()), 0)
put("dl_fs_sunday", DLT.g_fs_raw.median(), 1)
put("dl_fs_drill", DLT.c_fs_raw.median(), 1)
put("dl_rush_med", int(pool.rush.median()), 0)
# reliability of the first step on both sides
_, d_dl, g_dl = build("DL", DRILL["DL"])
rc_fs, rg_fs, _ = reliab(d_dl, g_dl, "fs", "fs_adj")
put("dl_rel_fs_drill", rc_fs)
put("dl_rel_fs_sunday", rg_fs)
r1 = rg_fs / (DLT.g_n_g.median() - (DLT.g_n_g.median() - 1) * rg_fs)
put("dl_snaps_for_07", int(np.ceil(0.7 * (1 - r1) / (r1 * 0.3))), 0)
sn = gdl.dropna(subset=["player_get_off"])
put("dl_rho_fs_ngs", -stats.spearmanr(sn.fs_adj, sn.player_get_off)[0])
put("dl_n_ngs", len(sn), 0)
# Carry-over reliability: split each player's Sunday snaps in half and recompute
gg = g_dl.copy()
gg["h"] = gg.groupby("nfl_id").cumcount() % 2
halves = []
for hh in (0, 1):
    q = gg[gg.h == hh].groupby("nfl_id")[[c + "_adj" for c in PROF]].mean()
    q.columns = PROF
    halves.append(q)
ix = DLT.set_index("nfl_id").index.intersection(halves[0].index).intersection(halves[1].index)
carry_h = []
for q in halves:
    Tq = DLT.set_index("nfl_id").loc[ix].copy()
    for c in PROF:
        Tq[f"g_{c}"] = q.loc[ix, c]
    Tq = Tq.reset_index()
    resid_z(Tq, [f"g_{c}" for c in PROF], ["w", "role"])
    carry_h.append(-np.abs(Tq[[f"c_{c}" for c in PROF]].to_numpy() - Tq[[f"g_{c}" for c in PROF]].to_numpy()).mean(1))
put("dl_rel_carry", sb(max(stats.spearmanr(carry_h[0], carry_h[1])[0], 0), 2))

# translation and payoff by tenth of a second
CUR = {"tr": [], "pay_g": [], "pay_c": []}
for t in range(TW):
    CUR["tr"].append(partial(DLT, f"c_v{t}", f"g_v{t}", [], nb=300))
    CUR["pay_g"].append(partial(pool, f"g_v{t}", "y", ["lp", "w", "role"], nb=300))
    CUR["pay_c"].append(partial(pool, f"c_v{t}", "y", ["lp", "w", "role"], nb=300))
put("dl_tr_absmax", max(abs(c[0]) for c in CUR["tr"]))
put("dl_payg_max", max(c[0] for c in CUR["pay_g"]))
put("dl_payc_absmax", max(abs(c[0]) for c in CUR["pay_c"]))
tr_fs = partial(DLT, "c_fs", "g_fs", [])
put("dl_tr_fs", tr_fs[0]); put("dl_tr_fs_lo", tr_fs[1]); put("dl_tr_fs_hi", tr_fs[2])
ceil_fs = np.sqrt(max(rc_fs, 0) * rg_fs)
put("dl_ceil_fs", ceil_fs)
tr40 = partial(DLT, "c40_fs", "g_fs", [])
put("dl_tr40_fs", tr40[0]); put("dl_tr40_n", tr40[3], 0)
put("dl_tr40_lo", tr40[1]); put("dl_tr40_hi", tr40[2])
# raw (unadjusted) first-step translation: size and role make it look real
raw = stats.spearmanr(DLT.c_fs_raw, DLT.g_fs_raw)[0]
put("dl_tr_fs_rawrho", raw)
put("dl_tr_mean_rawrho", stats.spearmanr(DLT.c_mean_v_raw, DLT.g_mean_v_raw)[0])
put("dl_ratio_med", DLT.carry_ratio.median(), 0)
put("dl_ratio_p10", DLT.carry_ratio.quantile(0.1), 0)
put("dl_ratio_p90", DLT.carry_ratio.quantile(0.9), 0)
put("dl_mv_sunday", DLT.g_mean_v_raw.median(), 1)
put("dl_mv_drill", DLT.c_mean_v_raw.median(), 1)

# ============================================================================ held-out test
tick("held-out")
BASE = ["lp", "w", "role", "forty_z", "forty_miss", "ngs_athleticism_score_z", "ngs_athleticism_score_miss"]
MEAS = [("carry", "Carry-over, pass-rush drill (speed + accel)"),
        ("carry_ratio", "Carry-over %, Sunday / drill speed, 0-1.5 s"),
        ("carry_speed", "Carry-over, speed curve only"),
        ("carry_shape", "Carry-over, curve shape only"),
        ("carry40", "Carry-over, 40-yard dash"),
        ("c_fs", "Combine first step, pass-rush drill"),
        ("g_fs", "Sunday first step (game tracking)")]


def design(d, cols):
    X = d[cols].to_numpy(float)
    return np.c_[np.ones(len(X)), X]


def fold_stats(d, x):
    """Per held-out class: slope of (y - baseline forecast) on x (sd from the training classes); plus the held-out
    squared errors with and without x."""
    out = []
    for yr in sorted(d.draft_year.unique()):
        tr, te = d[d.draft_year != yr], d[d.draft_year == yr]
        bb = np.linalg.lstsq(design(tr, BASE), tr.y.to_numpy(), rcond=None)[0]
        f_te = design(te, BASE) @ bb
        mu, sd = tr[x].mean(), tr[x].std()
        xz_tr, xz_te = (tr[x] - mu) / sd, (te[x] - mu) / sd
        bx = np.linalg.lstsq(np.c_[design(tr, BASE), xz_tr], tr.y.to_numpy(), rcond=None)[0]
        f2_te = np.c_[design(te, BASE), xz_te] @ bx
        e = te.y.to_numpy() - f_te
        sl = np.polyfit(xz_te.to_numpy(), e, 1)[0]
        out.append(dict(yr=yr, n=len(te), slope=sl, sse0=((te.y - f_te) ** 2).sum(), sse1=((te.y - f2_te) ** 2).sum(),
                        sst=((te.y - tr.y.mean()) ** 2).sum()))
    return out


def summarize(fs):
    n = np.array([f["n"] for f in fs])
    slope = float(np.sum(n * np.array([f["slope"] for f in fs])) / n.sum())
    sse0, sse1, sst = (sum(f[k] for f in fs) for k in ("sse0", "sse1", "sst"))
    return slope, (sse0 - sse1) / sst, 1 - sse0 / sst


def held_out(d, x, nb=NBOOT):
    d = d.dropna(subset=[x, "y"])
    sl, dr2, r2b = summarize(fold_stats(d, x))
    per = [f["slope"] for f in fold_stats(d, x)]
    bs = []
    groups = {yr: g.index.to_numpy() for yr, g in d.groupby("draft_year")}
    for _ in range(nb):
        idx = np.concatenate([RNG.choice(v, len(v)) for v in groups.values()])
        bs.append(summarize(fold_stats(d.loc[idx].reset_index(drop=True), x))[:2])
    bs = np.array(bs)
    se = bs[:, 0].std()
    p_tost = max(1 - stats.norm.cdf((sl + MARGIN) / se), stats.norm.cdf((sl - MARGIN) / se))
    return dict(slope=sl, lo=np.percentile(bs[:, 0], 5), hi=np.percentile(bs[:, 0], 95), dr2=dr2,
                dr2_lo=np.percentile(bs[:, 1], 5), dr2_hi=np.percentile(bs[:, 1], 95), r2_base=r2b,
                p_tost=p_tost, n=len(d), per=per)


HO = {}
for x, lab in MEAS:
    HO[x] = held_out(pool, x)
    h = HO[x]
    print(f"{lab:45s} n={h['n']:3d} slope {h['slope']:+.2f} [{h['lo']:+.2f},{h['hi']:+.2f}] dR2 {h['dr2']:+.3f} "
          f"[{h['dr2_lo']:+.3f},{h['dr2_hi']:+.3f}] TOST p {h['p_tost']:.3f} per-class {np.round(h['per'], 2)}")
    put(f"ho_{x}_n", h["n"], 0)
    put(f"ho_{x}_b", h["slope"], 2)
    put(f"ho_{x}_lo", h["lo"], 2)
    put(f"ho_{x}_hi", h["hi"], 2)
    put(f"ho_{x}_dr2", h["dr2"], 3)
    put(f"ho_{x}_dr2lo", h["dr2_lo"], 3)
    put(f"ho_{x}_dr2hi", h["dr2_hi"], 3)
    put(f"ho_{x}_tost", h["p_tost"], 3)
put("ho_r2_base", HO["carry"]["r2_base"])
bb = np.linalg.lstsq(design(pool, BASE), pool.y.to_numpy(), rcond=None)[0]
put("pts_pick32_100", bb[1] * (np.log(32) - np.log(100)), 1)
put("y_sd", pool.y.std(), 1)
put("n_equiv", int(sum(HO[x]["p_tost"] < 0.05 for x, _ in MEAS[:6])), 0)
put("n_meas_combine", 6, 0)
# official 40 time alone, the no-tracking contrast: slope of y-forecast(without forty) on forty_z
BASE_SAVE = BASE
BASE = ["lp", "w", "role", "ngs_athleticism_score_z", "ngs_athleticism_score_miss"]
p40 = pool[pool.forty_miss == 0].copy()
p40["forty_sd"] = p40.forty_z
HO["forty"] = held_out(p40, "forty_sd")
BASE = BASE_SAVE
h = HO["forty"]
print("official 40", h["n"], round(h["slope"], 2), round(h["lo"], 2), round(h["hi"], 2), round(h["dr2"], 3))
put("ho_forty_n", h["n"], 0); put("ho_forty_b", h["slope"], 2)
put("ho_forty_lo", h["lo"], 2); put("ho_forty_hi", h["hi"], 2)

# decomposition: does Carry-over add anything once both of its halves are in the model?
dec = partial(pool, "carry", "y", ["lp", "w", "role", "g_fs", "c_fs", "g_mean_v", "c_mean_v"])
put("dec_carry", dec[0]); put("dec_lo", dec[1]); put("dec_hi", dec[2])
pg = partial(pool, "g_fs", "y", ["lp", "w", "role"])
put("pay_gfs", pg[0]); put("pay_gfs_lo", pg[1]); put("pay_gfs_hi", pg[2])
put("pay_gfs_p", perm_p(pool, "g_fs", "y", ["lp", "w", "role"]), 3)
pc = partial(pool, "carry", "y", ["lp", "w", "role"])
put("pay_carry", pc[0]); put("pay_carry_lo", pc[1]); put("pay_carry_hi", pc[2])

# role split (cell n stated)
for e, tag in [(True, "edge"), (False, "int")]:
    sub_ = pool[(pool.role >= 0.5) == e]
    r_ = partial(sub_, "carry", "y", ["lp", "w"])
    put(f"split_{tag}_r", r_[0]); put(f"split_{tag}_lo", r_[1]); put(f"split_{tag}_hi", r_[2])
    put(f"split_{tag}_n", r_[3], 0)

# ============================================================================ the decision: rank rushers within class
# Within each held-out class, all same-role pairs (EDGE vs interior by majority alignment): who out-rushes whom on
# career pressure rate? Rule A: the earlier pick. Rule B: the higher out-of-class forecast that adds the Sunday first
# step. Rule C: the higher out-of-class forecast that adds Carry-over. Pair-level accuracy, rusher-clustered bootstrap.
tick("pairs")


def oof_forecast(d, extra):
    f = pd.Series(index=d.index, dtype=float)
    for yr in sorted(d.draft_year.unique()):
        tr, te = d[d.draft_year != yr], d[d.draft_year == yr]
        cols = BASE + extra
        mu, sd = tr[extra].mean(), tr[extra].std()
        Xtr, Xte = tr[cols].copy(), te[cols].copy()
        if extra:
            Xtr[extra] = (Xtr[extra] - mu) / sd
            Xte[extra] = (Xte[extra] - mu) / sd
        b = np.linalg.lstsq(np.c_[np.ones(len(Xtr)), Xtr.to_numpy(float)], tr.y.to_numpy(), rcond=None)[0]
        f.loc[te.index] = np.c_[np.ones(len(Xte)), Xte.to_numpy(float)] @ b
    return f


pool["f_base"] = oof_forecast(pool, [])
pool["f_sun"] = oof_forecast(pool, ["g_fs"])
pool["f_carry"] = oof_forecast(pool, ["carry"])
pool["edge"] = pool.role >= 0.5
rows = []
for (yr, e), g in pool.groupby(["draft_year", "edge"]):
    g = g.reset_index(drop=True)
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a, b = g.loc[i], g.loc[j]
            if a.y == b.y:
                continue
            win = a.y > b.y
            rows.append(dict(i=a.nfl_id, j=b.nfl_id, pick=(a.pick < b.pick) == win if a.pick != b.pick else np.nan,
                             base=(a.f_base > b.f_base) == win, sun=(a.f_sun > b.f_sun) == win,
                             carry=(a.f_carry > b.f_carry) == win,
                             gap=abs(a.pick - b.pick)))
PR = pd.DataFrame(rows)
ids = np.unique(np.r_[PR.i, PR.j])
put("pairs_n", len(PR), 0)


def pair_acc(PR, col, nb=NBOOT):
    v = PR[col].dropna().astype(float)
    acc = v.mean()
    pos = {k: n for n, k in enumerate(ids)}
    ii, jj = PR.i.map(pos).to_numpy(), PR.j.map(pos).to_numpy()
    vals = PR[col].astype(float).to_numpy()
    bs = []
    for _ in range(nb):
        w = RNG.multinomial(len(ids), np.ones(len(ids)) / len(ids))
        ww = w[ii] * w[jj]
        ok = ~np.isnan(vals)
        bs.append(np.sum(ww[ok] * vals[ok]) / np.sum(ww[ok]))
    return acc, np.percentile(bs, 5), np.percentile(bs, 95), bs


ACC = {}
for col in ["pick", "base", "sun", "carry"]:
    ACC[col] = pair_acc(PR, col)
    put(f"acc_{col}", 100 * ACC[col][0], 0)
    put(f"acc_{col}_lo", 100 * ACC[col][1], 0)
    put(f"acc_{col}_hi", 100 * ACC[col][2], 0)
    print(col, np.round(ACC[col][:3], 3))
# paired differences (same bootstrap seeds are not shared, so redo jointly)
pos = {k: n for n, k in enumerate(ids)}
ii, jj = PR.i.map(pos).to_numpy(), PR.j.map(pos).to_numpy()
V_ = {c: PR[c].astype(float).to_numpy() for c in ["pick", "base", "sun", "carry"]}
okp = ~np.isnan(V_["pick"])
diffs = {"sun_base": [], "carry_base": [], "sun_pick": []}
for _ in range(NBOOT):
    w = RNG.multinomial(len(ids), np.ones(len(ids)) / len(ids))
    ww = w[ii] * w[jj]
    a = {c: np.sum(ww[okp] * V_[c][okp]) / np.sum(ww[okp]) for c in V_}
    diffs["sun_base"].append(a["sun"] - a["base"])
    diffs["carry_base"].append(a["carry"] - a["base"])
    diffs["sun_pick"].append(a["sun"] - a["pick"])
for k, v in diffs.items():
    pt = {"sun_base": V_["sun"][okp].mean() - V_["base"][okp].mean(),
          "carry_base": V_["carry"][okp].mean() - V_["base"][okp].mean(),
          "sun_pick": V_["sun"][okp].mean() - V_["pick"][okp].mean()}[k]
    put(f"d_{k}", 100 * pt, 0)
    put(f"d_{k}_lo", 100 * np.percentile(v, 5), 0)
    put(f"d_{k}_hi", 100 * np.percentile(v, 95), 0)
    put(f"d_{k}_pgt0", float(np.mean(np.array(v) > 0)), 2)
    print(k, round(pt, 3), np.round(np.percentile(v, [5, 95]), 3))

# ============================================================================ named players
tick("names")
pool["resid"] = pool.y - pool.f_base
pool["carry_pct"] = pool.carry.rank(pct=True) * 100
pool["gfs_pct"] = pool.g_fs.rank(pct=True) * 100
pool["cfs_pct"] = pool.c_fs.rank(pct=True) * 100


def pick_txt(r):
    return "UDFA" if pd.isna(r.draft_overall_pick) else f"{int(r.draft_overall_pick)}"


cols = ["display_name", "draft_year", "draft_overall_pick", "latest_team", "role", "rush", "y", "f_base", "resid",
        "carry", "carry_pct", "c_fs", "g_fs", "cfs_pct", "gfs_pct", "g_n_g"]
pool.sort_values("carry", ascending=False)[cols].round(3).to_csv(OUT / "players.csv", index=False)
top = pool.sort_values("carry", ascending=False).head(3)
bot = pool.sort_values("carry").head(3)
put("top3_resid", top.resid.mean(), 1)
put("bot3_resid", bot.resid.mean(), 1)
q = pool.carry.quantile([1 / 3, 2 / 3]).to_numpy()
hi_, lo_ = pool[pool.carry >= q[1]], pool[pool.carry <= q[0]]
put("carry_top3rd_resid", hi_.resid.mean(), 1)
put("carry_bot3rd_resid", lo_.resid.mean(), 1)
put("carry_third_n", len(hi_), 0)
qg = pool.g_fs.quantile([1 / 3, 2 / 3]).to_numpy()
put("gfs_top3rd_resid", pool[pool.g_fs >= qg[1]].resid.mean(), 1)
put("gfs_bot3rd_resid", pool[pool.g_fs <= qg[0]].resid.mean(), 1)
# rushes per season -> pressures: 1 point of pressure rate over a 300-rush season
put("pts_to_pressures", 3, 0)


def name_rows(df):
    out = []
    for _, r in df.iterrows():
        role = "edge" if r.role >= 0.5 else "interior"
        out.append(f"| {r.display_name} ({r.draft_year}, {pick_txt(r)}; {r.latest_team}) | {role} | "
                   f"{fmt(-r.carry, 2)} ({r.carry_pct:.0f}) | {r.cfs_pct:.0f} / {r.gfs_pct:.0f} | "
                   f"{fmt(r.y, 1)}% vs {fmt(r.f_base, 1)}% ({int(r.rush)}) |")
    return out


TABLE_NAMES = "\n".join(
    ["| Rusher (class, pick; team) | Role | Curve gap, sd (Carry-over pct) | Drill / Sunday first step, pct | "
     "Pressure vs forecast (rushes) |", "|---|---|---|---|---|",
     "| ***Most alike: plays as he tested*** | | | | |"] + name_rows(top) +
    ["| ***Least alike*** | | | | |"] + name_rows(bot))
for i, (_, r) in enumerate(top.iterrows()):
    put(f"top{i}_name", r.display_name)
    put(f"top{i}_resid", r.resid, 1)
for i, (_, r) in enumerate(bot.iterrows()):
    put(f"bot{i}_name", r.display_name)
    put(f"bot{i}_resid", r.resid, 1)

# day-3 / UDFA rushers whose Sunday first step is top third and who already beat the forecast: film list
d3 = pool[(pool.pick >= 103) & (pool.g_fs >= qg[1])].sort_values("g_fs", ascending=False)
put("rel_target", 0.7, 1)
put("dl_excluded", len(DLT) - len(pool), 0)
put("d3_list_n", len(d3), 0)
put("d3_list_beat", int((d3.resid > 0).sum()), 0)
d3all = pool[pool.pick >= 103]
put("d3_all_n", len(d3all), 0)
put("d3_all_beat", int((d3all.resid > 0).sum()), 0)
TABLE_D3 = "\n".join(
    ["| Day-3 / UDFA rusher (class, pick; team) | Sunday first step, pct (starts) | Pass-rush drill first step, pct | "
     "Pressure vs forecast (rushes) |", "|---|---|---|---|"] +
    [f"| {r.display_name} ({r.draft_year}, {pick_txt(r)}; {r.latest_team}) | {r.gfs_pct:.0f} ({int(r.g_n_g)}) | "
     f"{r.cfs_pct:.0f} | {fmt(r.y, 1)}% vs {fmt(r.f_base, 1)}% ({int(r.rush)}) |" for _, r in d3.head(6).iterrows()])
print(d3[cols].round(2).to_string())

# ============================================================================ figures
tick("figures")
tt = np.arange(TW) / 10
show = [top.iloc[0], top.iloc[1], bot.iloc[1], bot.iloc[0]]
fig, axs = plt.subplots(1, 4, figsize=(FIGW, 4.9), sharey=True)
DLi = DLT.set_index("nfl_id")
for ax, r in zip(axs, show):
    cz = DLi.loc[r.nfl_id, [f"c_v{t}" for t in range(TW)]].to_numpy(float)
    gz = DLi.loc[r.nfl_id, [f"g_v{t}" for t in range(TW)]].to_numpy(float)
    ax.axhspan(-0.67, 0.67, color=OI["grey"], alpha=0.18, lw=0, label="middle half of DL peers")
    ax.fill_between(tt, cz, gz, color=OI["purple"], alpha=0.25, lw=0, label="the gap Carry-over measures")
    ax.plot(tt, gz, color=OI["blue"], lw=3, label=f"his NFL rushes (mean)")
    ax.plot(tt, cz, color=OI["orange"], lw=3, ls="--", label="his pass-rush drill")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{r.display_name}\ngap {fmt(-r.carry, 2)} sd (pct {r.carry_pct:.0f})", fontsize=13)
    ax.text(0.03, 0.97, f"pressure {fmt(r.y, 1)}%, forecast {fmt(r.f_base, 1)}%", transform=ax.transAxes,
            va="top", fontsize=11)
    ax.set_xlabel("seconds from first move")
    ax.set_ylim(-3.5, 3.6)
axs[0].set_ylabel("speed vs peers, sd")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, fontsize=12)
fig.suptitle(f"Most alike (left) and least alike (right) of {len(pool)} rushers: the match does not sort who beats his forecast",
             fontsize=15, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.09, 1, 0.94))
save(fig, "fig1_curves.png")

# Fig 2: the screen
S = SCR.copy()
S["lab"] = [f"{g}, {DRILL_LABEL[d]}{'' if not np.isnan(cl) else '*'} (n={n})"
            for g, d, n, cl in zip(S.grp, S.drill, S.n, S.ceil)]
S = S.iloc[::-1].reset_index(drop=True)
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 6.6), sharey=True)
yy = np.arange(len(S))
for ax, (c, lo, hi, ttl, xl) in zip(axs, [
        ("tr", "tr_lo", "tr_hi", "A. Drill curve vs NFL curve", "rho after weight and role"),
        ("pay", "pay_lo", "pay_hi", "B. Carry-over vs production", "partial rho beyond pick")]):
    col = [OI["blue"] if m else OI["grey"] for m in S.matched]
    ax.hlines(yy, S[lo], S[hi], color=col, lw=3)
    ax.scatter(S[c], yy, color=col, s=70, zorder=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_title(ttl, fontsize=14, loc="left")
    ax.set_xlim(-0.7, 0.85)
    ax.set_xlabel(xl + " [90%]", fontsize=13)
    if c == "tr":
        ax.scatter(S.ceil, yy, marker="|", s=300, color=OI["red"], lw=3, zorder=4,
                   label="most a perfect link could show (noise ceiling)")
axs[0].set_yticks(yy)
axs[0].set_yticklabels(S.lab, fontsize=12)
axs[0].scatter([], [], color=OI["blue"], s=70, label="matched position drill")
axs[0].scatter([], [], color=OI["grey"], s=70, label="40-yard dash")
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=12)
fig.suptitle("Ten drill-group pairs: no start carries from Combine to NFL, and Carry-over does not pay",
             fontsize=15, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
save(fig, "fig2_screen.png")

# Fig 3: DL by tenth of a second
fig, ax = plt.subplots(figsize=(FIGW, 5.4))
for key, col, lab, mk in [("tr", OI["orange"], "drill speed vs NFL speed (translation)", "s"),
                          ("pay_c", OI["grey"], "drill speed vs pressure, beyond pick", "D"),
                          ("pay_g", OI["blue"], "NFL speed vs pressure, beyond pick", "o")]:
    r = np.array([c[0] for c in CUR[key]])
    lo = np.array([c[1] for c in CUR[key]])
    hi = np.array([c[2] for c in CUR[key]])
    ax.fill_between(tt, lo, hi, color=col, alpha=0.16, lw=0)
    ax.plot(tt, r, color=col, lw=3, marker=mk, label=lab)
ax.axhline(0, color="black", lw=1)
ax.axvspan(0.25, 0.55, color=OI["sky"], alpha=0.15, lw=0)
ax.text(0.27, 0.45, "first step", fontsize=12)
ax.set_ylim(-0.32, 0.5)
ax.set_xlabel("seconds from first move")
ax.set_ylabel("rho [90% band]")
ax.legend(loc="lower right", frameon=False, fontsize=12)
ax.set_title(f"Pass rushers, tenth by tenth (n={len(DLT)}; {len(pool)} with 100+ rushes): only NFL speed leans "
             "toward pressure", fontsize=14, loc="left")
fig.tight_layout()
save(fig, "fig3_by_tenth.png")

# Fig 4: held-out slopes forest
fig, ax = plt.subplots(figsize=(FIGW, 6.0))
labs = [lab for _, lab in MEAS] + ["Official 40 time (stopwatch, contrast)"]
keys = [x for x, _ in MEAS] + ["forty"]
yy = np.arange(len(keys))[::-1]
ax.axvspan(-MARGIN, MARGIN, color=OI["grey"], alpha=0.15, lw=0, label=f"±{MARGIN:.0f} pt margin (about 3 pressures a season)")
for y_, k in zip(yy, keys):
    h = HO[k]
    col = OI["blue"] if k in ("g_fs", "forty") else OI["orange"]
    ax.hlines(y_, h["lo"], h["hi"], color=col, lw=3)
    ax.scatter(h["slope"], y_, color=col, s=90, zorder=3)
    for pc_, mk in zip(h["per"], ["^", "v", "s"]):
        ax.scatter(pc_, y_, facecolor="white", edgecolor="black", marker=mk, s=45, zorder=4)
for mk, yr in zip(["^", "v", "s"], [2023, 2024, 2025]):
    ax.scatter([], [], facecolor="white", edgecolor="black", marker=mk, label=f"{yr} class held out")
ax.axvline(0, color="black", lw=1)
ax.set_xlim(-1.25, 1.25)
ax.set_yticks(yy)
ax.set_yticklabels([f"{l} (n={HO[k]['n']})" for l, k in zip(labs, keys)], fontsize=12)
ax.set_xlabel("pressure-rate points per sd, beyond the no-tracking forecast [90%]", fontsize=13)
h_, l_ = ax.get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, fontsize=11)
ax.set_title("Unseen classes: every Carry-over version sits inside ±1 point", fontsize=15, loc="left")
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fig4_heldout.png")

# ============================================================================ render
tick("render")
TABLES = {"names": TABLE_NAMES}


def render():
    tpl = OUT / "writeup_template.md"
    if not tpl.exists():
        return
    text = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda m: TABLES[m.group(1)], tpl.read_text())
    claims, used = {}, set()
    for line in text.splitlines():
        for k in re.findall(r"\{\{(\w+)\}\}", line):
            claims.setdefault(k, re.sub(r"\{\{(\w+)\}\}", lambda m: str(NUM.get(m.group(1), "?")), line)
                              .strip()[:200])

    def sub(m):
        k = m.group(1)
        if k not in NUM:
            raise KeyError(f"template uses unknown key {k}")
        used.add(k)
        v = str(NUM[k])
        return "−" + v[1:] if v.startswith("-") else v

    (OUT / "writeup.md").write_text(re.sub(r"\{\{(\w+)\}\}", sub, text))
    # numbers.json holds exactly the values the text quotes, plus every decimal in the rendered tables
    quoted = {k: NUM[k] for k in sorted(used)}
    tab_dec = sorted(set(re.findall(r"(?<![\w.])-?\d+\.\d+(?![\w.])", "\n".join(TABLES.values()))))
    for i, v in enumerate(tab_dec):
        quoted[f"table_value_{i:02d}"] = v
        claims[f"table_value_{i:02d}"] = "table cell"
    json.dump(quoted, open(OUT / "numbers.json", "w"), indent=1, ensure_ascii=False)
    json.dump({k: claims.get(k, "") for k in quoted}, open(OUT / "claims.json", "w"), indent=1, ensure_ascii=False)


json.dump(NUM, open(OUT / "numbers_all.json", "w"), indent=1, ensure_ascii=False)
render()
tick("done")
