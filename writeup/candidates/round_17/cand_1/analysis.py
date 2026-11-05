"""Round 17 cand 1: DL/EDGE pass-rush drill and Run-the-Hoop kinematics vs NFL get-off, NFL bend and pressure.

Run from the repo root:
    uv run python writeup/candidates/round_17/cand_1/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (every number the text quotes), board.csv and
figures/*.png next to this file, then renders writeup.md from writeup_template.md ({{key}} -> numbers.json value;
fails on an unknown key). Frame-level features are cached in data/cache/r17c1 (gitignored). Runtime ~6 minutes (2,000 bootstrap refits).

Design (one estimand per claim):
  * Drill metrics, z within Combine year:
      Hoop G  = hardest 0.5 s of sideways (centripetal) acceleration on the Run-the-Hoop lap, g, mean over runs.
      Burst   = yards covered in the first second after movement onset in the pass-rush drill, mean over reps.
  * Game traits (random sampled NFL pass rushes only, ~20 per lineman, self-referenced):
      NFL get-off   = NGS season get-off (rush-weighted), sign flipped so higher = quicker.
      NFL burst     = yards covered in the first second after the snap.
      NFL bend      = hardest 0.5 s of sideways acceleration between snap and throw/sack (max 3 s), g.
  * Outcome: career pass-rush pressure rate, empirical-Bayes shrunk, linemen with 30+ rushes.
  * Baseline without tracking: log pick, Combine weight, NFL EDGE share, NGS athleticism score, official 10-yd split
    and 3-cone (missing times filled from weight, role and NGS). OLS fit on the other two classes, scored on the
    held-out class (leave-one-class-out). Held-out R2, RMSE and Spearman; increments as delta-R2 with a
    class-stratified player bootstrap that refits every fold.
  * One pre-set margin: 0.5 pressure-rate points per sd (one pressure per 200 rushes; 2.25 per 450-rush season).
  * Decision use: day-3/undrafted (pick 103+) top third of Hoop G within class = film shortlist; hit = pressure rate
    above the held-out forecast; reported against the base rate with Wilson intervals; 2026 power simulated.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.collections import LineCollection
from scipy import stats
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[4]
PROC = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
for _old in FIG.glob("*.png"):
    _old.unlink()
CACHE = ROOT / "data" / "cache" / "r17c1"
CACHE.mkdir(parents=True, exist_ok=True)

NBOOT = int(os.environ.get("BDB_NBOOT", 2000))
RATE_MIN, REG_MIN, DAY3, UDFA_PICK = 30, 100, 103, 262
MARGIN = 0.5            # pressure-rate points per sd: smallest effect worth acting on, set before fitting
SEASON = 450            # rushes in a regular's season
DT, W1 = 0.1, 10
ON, PRE = 1.0, 0.5
YD_PER_G = 9.80665 / 0.9144
Z90 = 1.6449
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "grey": "#8C8C8C", "black": "#000000", "yellow": "#F0E442"}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
FIGW = 1600 / 150

NUM: dict = {}


def put(key, v, nd=2):
    if key in NUM:
        raise KeyError(f"duplicate key {key}")
    if isinstance(v, (int, np.integer)):
        NUM[key] = int(v)
        return int(v)
    s = f"{float(v):.{nd}f}"
    if re.fullmatch(r"-0\.?0*", s):
        s = s.lstrip("-")
    NUM[key] = s
    return s


def save(fig, name):
    fig.set_size_inches(FIGW, fig.get_size_inches()[1])
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def cached(name, fn):
    f = CACHE / f"{name}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    out = fn()
    pickle.dump(out, open(f, "wb"))
    return out


def yz(s, year):
    return s.groupby(year).transform(lambda v: (v - v.mean()) / v.std())


def icc(ids, vals):
    d = pd.DataFrame({"i": ids, "v": vals}).dropna()
    d = d[d.groupby("i").v.transform("size") >= 2]
    g = d.groupby("i").v
    n, N = g.ngroups, len(d)
    kk = g.size()
    k0 = (N - (kk ** 2).sum() / N) / (n - 1)
    msb = (kk * (g.mean() - d.v.mean()) ** 2).sum() / (n - 1)
    msw = ((d.v - g.transform("mean")) ** 2).sum() / (N - n)
    return float((msb - msw) / (msb + (k0 - 1) * msw)), int(n)


def sb(r1, k):
    return k * r1 / (1 + (k - 1) * r1)


def wilson(k, n, z=Z90):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k0, n0):
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    d = p1 - p0
    return d, d - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2), d + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)


def boot_psp(df, x, y, Z, seed):
    """Partial Spearman of x and y given Z (ranks residualized), class-stratified 90% player bootstrap."""
    d = df[[x, y] + Z + ["draft_year"]].dropna()
    R = d[[x, y] + Z].rank().values
    cl = d.draft_year.values

    def f(ix):
        A = np.c_[np.ones(len(ix)), R[ix, 2:]]
        rx = R[ix, 0] - A @ np.linalg.lstsq(A, R[ix, 0], rcond=None)[0]
        ry = R[ix, 1] - A @ np.linalg.lstsq(A, R[ix, 1], rcond=None)[0]
        return float(np.corrcoef(rx, ry)[0, 1])
    r = f(np.arange(len(d)))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cl == c) for c in np.unique(cl)]
    b = [f(np.concatenate([rng.choice(g_, len(g_)) for g_ in groups])) for _ in range(NBOOT)]
    lo, hi = np.nanpercentile(b, [5, 95])
    return r, float(lo), float(hi), int(len(d))


# ============================================================================= load
players = pd.read_parquet(PROC / "players.parquet")
cres = pd.read_parquet(PROC / "combine_results.parquet")
events = pd.read_parquet(PROC / "combine_events.parquet")
seasons = pd.read_parquet(PROC / "player_season_stats.parquet")
align = pd.read_parquet(PROC / "player_season_alignment.parquet")
gp = pd.read_parquet(PROC / "game_plays_sample.parquet")
career = pd.read_parquet(PROC / "player_career.parquet")

dl = players[players.position_group == "DL"][["nfl_id", "display_name", "draft_year", "nfl_position",
                                               "draft_overall_pick", "latest_team"]].copy()
dl["pick"] = dl.draft_overall_pick.fillna(UDFA_PICK)
dl["log_pick"] = np.log(dl.pick)
dl = dl.merge(cres.drop(columns=["draft_year", "combine_position"]), on="nfl_id", how="left")
for c in ["combine_weight", "ten_yd_split", "three_cone"]:
    dl[c + "_z"] = yz(dl[c], dl.draft_year)
a = align.merge(dl[["nfl_id"]], on="nfl_id")
a["edge_pl"] = np.where(a.lined_up_position.isin(["EDGE", "OLB"]), a.plays, 0)
es = a.groupby("nfl_id").agg(edge_pl=("edge_pl", "sum"), pl=("plays", "sum"))
dl = dl.merge((es.edge_pl / es.pl).rename("edge_share").reset_index(), on="nfl_id", how="left")
dl["edge_share"] = dl.edge_share.fillna(dl.nfl_position.isin(["DE", "OLB"]).astype(float))
dl["role"] = np.where(dl.edge_share >= 0.5, "EDGE", "interior")
dl = dl.merge(career[["nfl_id", "contract_apy_m"]], on="nfl_id", how="left")
classes = [2023, 2024, 2025]
put("n_dl", len(dl)); put("pick_day3", DAY3); put("margin", MARGIN, 1)
put("margin_season", MARGIN / 100 * SEASON, 2); put("season_rushes", SEASON)
put("n_three_cone", int(dl.three_cone.notna().sum()))

# ============================================================================= Combine drill kinematics
ev = events[(events.position_group == "DL") & events.drill_name.isin(["PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL"])]
ev = ev[["event_id", "nfl_id", "draft_year", "drill_name", "attempt_no"]]


def onset(s, start=0):
    s = np.asarray(s, float)
    hit = np.flatnonzero(s[start:] >= ON)
    if not len(hit):
        return None
    i = j = start + int(hit[0])
    while j > start and s[j - 1] >= PRE and i - j < 3:
        j -= 1
    return j


def lateral_g(x, y):
    xs, ys = savgol_filter(x, 7, 2), savgol_filter(y, 7, 2)
    vx, vy = savgol_filter(xs, 7, 2, deriv=1, delta=DT), savgol_filter(ys, 7, 2, deriv=1, delta=DT)
    ax, ay = savgol_filter(xs, 7, 2, deriv=2, delta=DT), savgol_filter(ys, 7, 2, deriv=2, delta=DT)
    v = np.hypot(vx, vy)
    alat = np.abs(vx * ay - vy * ax) / np.maximum(v, 0.5) / YD_PER_G
    roll = pd.Series(alat).rolling(5, center=True).mean().to_numpy()
    return alat, roll, xs, ys, v


def attempt_features(g):
    x, y, s = g.x.to_numpy(), g.y.to_numpy(), g.s.to_numpy()
    if len(x) < 15:
        return None
    alat, roll, xs, ys, v = lateral_g(x, y)
    k = int(np.nanargmax(roll))
    o = onset(s)
    d1 = np.nan
    if o is not None and s[0] < ON and o + W1 < len(x):
        d1 = float(np.hypot(x[o + W1] - x[o], y[o + W1] - y[o]))
    return {"bend_g": float(np.nanmax(roll)), "d1": d1, "speed_at_bend": float(v[k]),
            "_alat": alat, "_x": xs, "_y": ys, "_k": k}


def drill_features():
    fr = pd.read_parquet(PROC / "combine_frames.parquet", filters=[("event_id", "in", list(ev.event_id))],
                         columns=["event_id", "frame", "entity_type", "x", "y", "s"])
    fr = fr[fr.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
    rows, traces = [], {}
    for eid, g in fr.groupby("event_id", sort=False):
        f = attempt_features(g)
        if f is None:
            continue
        traces[eid] = {k: f.pop(k) for k in ["_alat", "_x", "_y", "_k"]}
        rows.append({"event_id": eid, **f})
    return rows, traces


rows, traces = cached("drill", drill_features)
att = pd.DataFrame(rows).merge(ev, on="event_id")
hoop_att = att[att.drill_name == "RUN_THE_HOOP_DRILL"].copy()
rush_att = att[att.drill_name == "PASS_RUSH_DRILL"].dropna(subset=["d1"]).copy()
hoop_att["z"] = yz(hoop_att.bend_g, hoop_att.draft_year)
rush_att["z"] = yz(rush_att.d1, rush_att.draft_year)
rel_hoop1, n_hrep = icc(hoop_att.nfl_id, hoop_att.z)
rel_d1_1, n_drep = icc(rush_att.nfl_id, rush_att.z)
k_hoop = hoop_att.groupby("nfl_id").size().mean()
k_rush = rush_att.groupby("nfl_id").size().mean()
REL = {"hoop_z": sb(rel_hoop1, k_hoop), "burst_z": sb(rel_d1_1, k_rush)}
put("hoop_g_median", hoop_att.bend_g.median())
put("rel_hoop1", rel_hoop1); put("rel_hoop_nrep", n_hrep); put("rel_hoop2", sb(rel_hoop1, 2))
put("rel_burst1", rel_d1_1); put("rel_burst_nrep", n_drep); put("rel_burst", REL["burst_z"])
put("burst_median", rush_att.d1.median())

hoop = hoop_att.groupby("nfl_id").agg(hoop_g=("bend_g", "mean"), hoop_k=("bend_g", "size"),
                                      hoop_eid=("event_id", "first"), draft_year=("draft_year", "first")).reset_index()
hoop["hoop_z"] = yz(hoop.hoop_g, hoop.draft_year)
rush = rush_att.groupby("nfl_id").agg(burst=("d1", "mean"), burst_k=("d1", "size"),
                                      draft_year=("draft_year", "first")).reset_index()
rush["burst_z"] = yz(rush.burst, rush.draft_year)
dl = dl.merge(hoop.drop(columns="draft_year"), on="nfl_id", how="left")
dl = dl.merge(rush.drop(columns="draft_year"), on="nfl_id", how="left")
put("n_hoop", int(dl.hoop_z.notna().sum())); put("n_burst", int(dl.burst_z.notna().sum()))
put("n_hoop_no3c", int((dl.hoop_z.notna() & dl.three_cone.isna()).sum()))
put("pct_one_run", 100 * (dl.hoop_k == 1).sum() / dl.hoop_z.notna().sum(), 0)
put("rho_hoop_burst", stats.spearmanr(dl.hoop_z, dl.burst_z, nan_policy="omit")[0])
_h3 = dl.dropna(subset=["hoop_z", "three_cone_z"])
put("rho_hoop_3cone", -stats.spearmanr(_h3.hoop_z, _h3.three_cone_z)[0]); put("n_hoop_3cone", len(_h3))
_b10 = dl.dropna(subset=["burst_z", "ten_yd_split_z"])
put("rho_burst_split", -stats.spearmanr(_b10.burst_z, _b10.ten_yd_split_z)[0]); put("n_burst_split", len(_b10))

# ============================================================================= NFL season outcomes
s = seasons[seasons.position_group == "DL"].copy()
s["w_go"] = s.pass_rush_snaps * s.avg_get_off
agg = s.groupby("nfl_id").agg(rush_snaps=("pass_rush_snaps", "sum"), pressures=("pressures", "sum"),
                              w_go=("w_go", "sum")).reset_index()
agg["get_off"] = agg.w_go / agg.rush_snaps
agg["press_rate"] = agg.pressures / agg.rush_snaps
dl = dl.merge(agg, on="nfl_id", how="left")
dl["rush_snaps"] = dl.rush_snaps.fillna(0)
m = dl.rush_snaps >= 20
n_, k_ = dl.loc[m, "rush_snaps"], dl.loc[m, "pressures"]
mu = k_.sum() / n_.sum()
N_ = n_.sum()
tau2 = ((n_ * (k_ / n_ - mu) ** 2).sum() - (len(n_) - 1) * mu * (1 - mu)) / (N_ - (n_ ** 2).sum() / N_)
Mw = mu * (1 - mu) / tau2 - 1
mm = dl.rush_snaps > 0
dl.loc[mm, "press_eb"] = 100 * (dl.loc[mm, "pressures"] + Mw * mu) / (dl.loc[mm, "rush_snaps"] + Mw)
dl["quick"] = -dl.get_off
put("press_mean", 100 * mu, 1); put("eb_prior_rushes", Mw, 0)
# season-level 2025 pressure for the 2025 cards
s25 = s[s.season == 2025].groupby("nfl_id").agg(r25=("pass_rush_snaps", "sum"), p25=("pressures", "sum"))
dl = dl.merge(s25.reset_index(), on="nfl_id", how="left")

# ============================================================================= sampled NFL tracking
dlids = set(dl.nfl_id)
K = ["game_id", "play_id", "nfl_id"]
g = gp[gp.nfl_id.isin(dlids)].drop_duplicates(K)
pinfo = g.set_index(K)[["play_direction", "sample_kind", "time_to_pressure", "sack", "player_get_off",
                        "lined_up_position", "down", "expected_points_added"]]


def game_features():
    gf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "x", "y", "event"])
    gf = gf[gf.nfl_id.isin(dlids)].merge(g[K + ["snap_frame"]], on=K)
    ends = gf[gf.event.isin(["pass_forward", "qb_sack", "qb_strip_sack", "run", "handoff"])].groupby(K).frame.min()
    gf = gf.join(ends.rename("end_frame"), on=K).sort_values(K + ["frame"])
    out, paths = [], {}
    for key, h in gf.groupby(K, sort=False):
        info = pinfo.loc[key]
        sf, e = h.snap_frame.iloc[0], h.end_frame.iloc[0]
        fr, x_, y_ = h.frame.values, h.x.values, h.y.values
        if not (fr == sf).any():
            continue
        i0 = np.flatnonzero(fr == sf)[0]
        i1 = np.flatnonzero(fr == sf + W1)
        d1 = float(np.hypot(x_[i1[0]] - x_[i0], y_[i1[0]] - y_[i0])) if len(i1) else np.nan
        # fixed 2.0 s window, only when the play lasts that long, so pressured snaps are not longer by construction
        w = (fr >= sf) & (fr <= sf + 20)
        bend = np.nan
        if w.sum() == 21 and (pd.isna(e) or e >= sf + 20):
            _, roll, xs, ys, _ = lateral_g(x_[w], y_[w])
            bend = float(np.nanmax(roll))
            if info.sample_kind == "random":
                sg = 1.0 if info.play_direction == "left" else -1.0
                paths[key] = (sg * (xs - xs[0]), ys - ys[0], int(np.nanargmax(roll)))
        out.append(dict(zip(K, key), g_d1=d1, g_bend=bend, kind=info.sample_kind,
                        get_off_snap=info.player_get_off, down=info.down, edge=int(info.lined_up_position == "EDGE"),
                        press=int(pd.notna(info.time_to_pressure) or info.sack == 1)))
    return pd.DataFrame(out), paths


GF, PATHS = cached("game_fixed2s", game_features)
GR = GF[GF.kind == "random"].copy()
GR["half"] = GR.groupby("nfl_id").cumcount() % 2
put("n_random_rushes", len(GR)); put("n_random_players", GR.nfl_id.nunique())
put("game_bend_median", GR.g_bend.median())


def split_half(df, col, min_n):
    cnt = df.groupby("nfl_id")[col].count()
    keep = cnt[cnt >= min_n].index
    a_ = df[df.nfl_id.isin(keep)].groupby(["nfl_id", "half"])[col].mean().unstack().dropna()
    r_ = stats.spearmanr(a_[0], a_[1])[0]
    return 2 * r_ / (1 + r_), len(a_)


REL["g_d1"], _ = split_half(GR, "g_d1", 8)
REL["g_bend"], n_rb = split_half(GR, "g_bend", 8)
REL["press_eb"] = np.nan
put("rel_game_burst", REL["g_d1"]); put("rel_game_bend", REL["g_bend"]); put("rel_game_n", n_rb)
PG = GR.groupby("nfl_id").agg(g_d1=("g_d1", "mean"), g_bend=("g_bend", "mean"), g_n=("g_bend", "count")).reset_index()
PG = PG[PG.g_n >= 8]
dl = dl.merge(PG, on="nfl_id", how="left")
put("n_game_traits", int(dl.g_bend.notna().sum()))
put("median_random_rushes", PG.g_n.median(), 0)
# season NGS get-off reliability: 2-season players, season-to-season correlation
_s2 = s[s.pass_rush_snaps >= 100].pivot_table(index="nfl_id", columns="season", values="avg_get_off")
_pairs = [(_s2[a_], _s2[b_]) for a_, b_ in [(2023, 2024), (2024, 2025)] if a_ in _s2 and b_ in _s2]
_pp = pd.concat([pd.DataFrame({"a": p_[0], "b": p_[1]}).dropna() for p_ in _pairs])
REL["quick"] = stats.spearmanr(_pp.a, _pp.b)[0]
put("rel_getoff_season", REL["quick"]); put("rel_getoff_n", len(_pp))
# EPA of a pressured snap (random sampled snaps)
_gm = g[(g.sample_kind == "random") & g.expected_points_added.notna()].drop_duplicates(["game_id", "play_id"])
_pr = _gm.time_to_pressure.notna() | (_gm.sack == 1)
e1, e0 = _gm[_pr].expected_points_added.values, _gm[~_pr].expected_points_added.values
_r = np.random.default_rng(101)
_eb = [_r.choice(e0, len(e0)).mean() - _r.choice(e1, len(e1)).mean() for _ in range(NBOOT)]
EPA = e0.mean() - e1.mean()
put("epa_press", EPA); put("epa_press_lo", np.percentile(_eb, 5)); put("epa_press_hi", np.percentile(_eb, 95))
put("epa_n", len(_gm))
put("margin_epa", MARGIN / 100 * SEASON * EPA, 1)

# ============================================================================= drill -> game transfer
Zc = ["combine_weight_z", "edge_share"]
DRILLS = [("burst_z", "Rush-drill burst"), ("hoop_z", "Hoop G")]
GAMES = [("quick", "NFL get-off\n(NGS season)"), ("g_d1", "NFL burst\n(first second)"),
         ("g_bend", "NFL bend\n(sideways g)"), ("press_eb", "Pressure rate")]
TR = {}
co30 = dl[dl.rush_snaps >= RATE_MIN].copy()
for i, (dx, _) in enumerate(DRILLS):
    for j, (gy, _) in enumerate(GAMES):
        src = co30 if gy == "press_eb" else dl
        r, lo, hi, n = boot_psp(src, dx, gy, Zc, 10 + 10 * i + j)
        ceil = np.sqrt(REL[dx] * REL[gy]) if not np.isnan(REL[gy]) else np.nan
        TR[(dx, gy)] = (r, lo, hi, n, ceil)
        tag = {"burst_z": "burst", "hoop_z": "hoop"}[dx] + "_" + gy
        put(f"tr_{tag}", r); put(f"tr_{tag}_lo", lo); put(f"tr_{tag}_hi", hi); put(f"tr_{tag}_n", n)
        if not np.isnan(ceil):
            put(f"tr_{tag}_ceil", ceil)
# game trait -> pressure (does the trait matter?)
for gy in ["quick", "g_d1", "g_bend"]:
    r, lo, hi, n = boot_psp(co30, gy, "press_eb", Zc, 77 + len(gy))
    TR[(gy, "press_eb")] = (r, lo, hi, n, np.nan)
    put(f"gp_{gy}", r); put(f"gp_{gy}_lo", lo); put(f"gp_{gy}_hi", hi); put(f"gp_{gy}_n", n)

# ============================================================================= held-out baseline and increments
MAIN = ["log_pick", "combine_weight_z", "edge_share", "ngs_athleticism_score"]
TIMES = ["ten_yd_split_z", "three_cone_z"]
BASE = MAIN + TIMES
FILLX = ["combine_weight_z", "edge_share", "ngs_athleticism_score"]
dl["ngs_athleticism_score"] = dl.ngs_athleticism_score.fillna(dl.ngs_athleticism_score.median())
put("n_ngs_missing", int(cres.set_index("nfl_id").loc[dl.nfl_id, "ngs_athleticism_score"].isna().sum()))


def fill_times(tr, te):
    tr, te = tr.copy(), te.copy()
    for c in TIMES:
        mk = tr[c].notna()
        A = np.c_[np.ones(mk.sum()), tr.loc[mk, FILLX].values]
        b = np.linalg.lstsq(A, tr.loc[mk, c].values, rcond=None)[0]
        for d in (tr, te):
            m2 = d[c].isna()
            d.loc[m2, c] = np.c_[np.ones(m2.sum()), d.loc[m2, FILLX].values] @ b
    return tr, te


def ols(tr, te, cols, y="press_eb"):
    A = np.c_[np.ones(len(tr)), tr[cols].values]
    b = np.linalg.lstsq(A, tr[y].values, rcond=None)[0]
    return np.c_[np.ones(len(te)), te[cols].values] @ b, b


def loco(d, cols):
    """Leave-one-class-out predictions; returns array aligned with d and the last-fold coefficient vector."""
    pred = np.full(len(d), np.nan)
    coefs = []
    cl = d.draft_year.values
    for c in classes:
        tr, te = d[cl != c], d[cl == c]
        tr, te = fill_times(tr, te)
        p, b = ols(tr, te, cols)
        pred[cl == c] = p
        coefs.append(b)
    return pred, np.mean(coefs, axis=0)


def oos(y, p):
    sse = ((y - p) ** 2).sum()
    return 1 - sse / ((y - y.mean()) ** 2).sum(), np.sqrt(sse / len(y)), stats.spearmanr(y, p)[0]


AN = co30.dropna(subset=["hoop_z", "burst_z"]).copy()
AN["ngs_athleticism_score"] = AN.ngs_athleticism_score.fillna(dl.ngs_athleticism_score.median())
for c in TIMES:
    pass
put("n_an", len(AN)); put("n_an_reg", int((AN.rush_snaps >= REG_MIN).sum()))
MODELS = [("Pick only", ["log_pick", "edge_share"]), ("Baseline", BASE),
          ("+ Hoop G", BASE + ["hoop_z"]), ("+ Rush-drill burst", BASE + ["burst_z"]),
          ("+ both drills", BASE + ["hoop_z", "burst_z"])]
y = AN.press_eb.values
PRED = {nm: loco(AN, cols)[0] for nm, cols in MODELS}
rng = np.random.default_rng(2027)
grp = [np.flatnonzero(AN.draft_year.values == c) for c in classes]
BOOT = {nm: [] for nm, _ in MODELS}
BSLOPE = {"hoop_z": [], "burst_z": []}
for _ in range(NBOOT):
    ix = np.concatenate([rng.choice(g_, len(g_)) for g_ in grp])
    d_ = AN.iloc[ix].reset_index(drop=True)
    yb = d_.press_eb.values
    for nm, cols in MODELS:
        p_, b_ = loco(d_, cols)
        BOOT[nm].append(oos(yb, p_)[0])
        if nm == "+ Hoop G":
            BSLOPE["hoop_z"].append(b_[-1])
        if nm == "+ Rush-drill burst":
            BSLOPE["burst_z"].append(b_[-1])
TAB1 = []
for nm, cols in MODELS:
    r2, rmse, rho = oos(y, PRED[nm])
    key = {"Pick only": "pick", "Baseline": "base", "+ Hoop G": "hoop", "+ Rush-drill burst": "burst",
           "+ both drills": "both"}[nm]
    put(f"r2_{key}", r2); put(f"rmse_{key}", rmse); put(f"rho_{key}", rho)
    lo, hi = np.percentile(BOOT[nm], [5, 95])
    put(f"r2_{key}_lo", lo); put(f"r2_{key}_hi", hi)
    row = {"model": nm, "key": key, "r2": r2, "rmse": rmse, "rho": rho, "lo": lo, "hi": hi}
    if key in ("hoop", "burst", "both"):
        dd = np.array(BOOT[nm]) - np.array(BOOT["Baseline"])
        put(f"dr2_{key}", r2 - oos(y, PRED["Baseline"])[0]); put(f"dr2_{key}_lo", np.percentile(dd, 5))
        put(f"dr2_{key}_hi", np.percentile(dd, 95)); put(f"dr2_{key}_ppos", 100 * (dd > 0).mean(), 0)
        row.update(dr=r2 - oos(y, PRED["Baseline"])[0], dlo=np.percentile(dd, 5), dhi=np.percentile(dd, 95),
                   dboot=dd)
    TAB1.append(row)
RES = AN.press_eb.values - PRED["Baseline"]
for k_, tag, sd_ in [("hoop_z", "hoop", 31), ("burst_z", "burst", 32)]:
    _d = AN.assign(res=RES)
    r, lo, hi, n = boot_psp(_d, k_, "res", [], sd_)
    put(f"res_{tag}", r); put(f"res_{tag}_lo", lo); put(f"res_{tag}_hi", hi)
for k_, tag in [("hoop_z", "hoop"), ("burst_z", "burst")]:
    _, b_full = loco(AN, BASE + [k_])
    bs = np.array(BSLOPE[k_])
    put(f"slope_{tag}", b_full[-1]); put(f"slope_{tag}_lo", np.percentile(bs, 5))
    put(f"slope_{tag}_hi", np.percentile(bs, 95))
    put(f"slope_{tag}_season", b_full[-1] / 100 * SEASON, 1)
    put(f"slope_{tag}_season_lo", np.percentile(bs, 5) / 100 * SEASON, 1)
    put(f"slope_{tag}_season_hi", np.percentile(bs, 95) / 100 * SEASON, 1)
    put(f"p_{tag}_above_margin", 100 * (bs > MARGIN).mean(), 0)
    put(f"p_{tag}_within_margin", 100 * (np.abs(bs) < MARGIN).mean(), 0)
    # per-class sign
    pos = 0
    for c in classes:
        d_c = AN[AN.draft_year == c]
        r_c = stats.spearmanr(d_c[k_], d_c.press_eb - PRED["Baseline"][AN.draft_year.values == c])[0]
        pos += int(r_c > 0)
    put(f"{tag}_pos_classes", pos)
SLOPE = {k_: (float(NUM[f"slope_{t}"]), np.array(BSLOPE[k_])) for k_, t in [("hoop_z", "hoop"), ("burst_z", "burst")]}
AN["fc"] = PRED["Baseline"]

# ============================================================================= day-3 film shortlist
ba = dl[dl.hoop_z.notna()].copy()
ba["third"] = ba.groupby("draft_year").hoop_z.transform(
    lambda s_: pd.qcut(s_.rank(method="first"), 3, labels=False)).astype(int)
ba["day3"] = ba.pick >= DAY3
ba["short"] = ba.day3 & (ba.third == 2)
sc = ba[ba.rush_snaps >= RATE_MIN].copy()
sc["fc"] = np.nan
for c in classes:
    tr_ = co30[co30.draft_year != c].copy()
    te_ = sc[sc.draft_year == c]
    trf, tef = fill_times(tr_, te_)
    sc.loc[te_.index, "fc"] = ols(trf, tef, BASE)[0]
sc["hit"] = sc.press_eb > sc.fc
d3 = sc[sc.day3]
k1, n1 = int(d3[d3.short].hit.sum()), int(d3.short.sum())
k0, n0 = int(d3[~d3.short].hit.sum()), int((~d3.short).sum())
kall, nall = k1 + k0, n1 + n0
put("sl_k", k1); put("sl_n", n1); put("rest_k", k0); put("rest_n", n0); put("d3_k", kall); put("d3_n", nall)
put("sl_rate", 100 * k1 / n1, 0); put("rest_rate", 100 * k0 / n0, 0); put("d3_rate", 100 * kall / nall, 0)
l1, u1 = wilson(k1, n1); l0, u0 = wilson(k0, n0); la, ua = wilson(kall, nall)
put("sl_lo", 100 * l1, 0); put("sl_hi", 100 * u1, 0); put("rest_lo", 100 * l0, 0); put("rest_hi", 100 * u0, 0)
put("d3_lo", 100 * la, 0); put("d3_hi", 100 * ua, 0)
gap, glo, ghi = newcombe(k1, n1, k0, n0)
put("gap", 100 * gap, 0); put("gap_lo", 100 * glo, 0); put("gap_hi", 100 * ghi, 0)
put("p_fisher", stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")[1])
put("p_binom_base", stats.binomtest(k1, n1, kall / nall, alternative="greater").pvalue)
# day 1-2 top third (out-of-rule check) and per-class
d12 = sc[~sc.day3]
put("d12_top_k", int(d12[d12.third == 2].hit.sum())); put("d12_top_n", int((d12.third == 2).sum()))
put("d12_rest_k", int(d12[d12.third < 2].hit.sum())); put("d12_rest_n", int((d12.third < 2).sum()))
CLS = []
for c in classes:
    dc = d3[d3.draft_year == c]
    CLS.append((c, int(dc[dc.short].hit.sum()), int(dc.short.sum()), int(dc[~dc.short].hit.sum()),
                int((~dc.short).sum())))
put("sl_classes_ahead", sum(int(a_[1] / max(a_[2], 1) > a_[3] / max(a_[4], 1)) for a_ in CLS))
# price: do clubs draft for Hoop G?
put("pick_top_third", ba[ba.third == 2].pick.median(), 0); put("pick_bot_third", ba[ba.third == 0].pick.median(), 0)
put("pct_day3_dl", 100 * (dl.pick >= DAY3).mean(), 0)
# shortlist pressure in a season vs rest (pressures per 450 rushes over forecast)
ov1 = (d3[d3.short].press_eb - d3[d3.short].fc).mean()
ov0 = (d3[~d3.short].press_eb - d3[~d3.short].fc).mean()
put("sl_over_season", (ov1 - ov0) / 100 * SEASON, 1)
# survivorship: did the shortlist reach 30 rushes as often?
alld3 = ba[ba.day3]
put("reach_sl_k", int((alld3[alld3.short].rush_snaps >= RATE_MIN).sum())); put("reach_sl_n", int(alld3.short.sum()))
put("reach_rest_k", int((alld3[~alld3.short].rush_snaps >= RATE_MIN).sum()))
put("reach_rest_n", int((~alld3.short).sum()))

# 2026 power: one new class, and three, at the observed and half gap (one-sided Fisher at 0.10)
prng = np.random.default_rng(26)
n1c, n0c = round(n1 / 3), round(n0 / 3)
put("pw_n1", n1c); put("pw_n0", n0c)


def power(p1, p0, nn1, nn0, sims=4000):
    hits = 0
    for _ in range(sims):
        a1, a0 = prng.binomial(nn1, p1), prng.binomial(nn0, p0)
        hits += stats.fisher_exact([[a1, nn1 - a1], [a0, nn0 - a0]], alternative="greater")[1] < 0.10
    return hits / sims


p1o, p0o = k1 / n1, k0 / n0
POW = {}
for lab, pp1 in [("obs", p1o), ("half", p0o + (p1o - p0o) / 2)]:
    for ncl in (1, 3):
        POW[(lab, ncl)] = power(pp1, p0o, n1c * ncl, n0c * ncl)
        put(f"pow_{lab}_{ncl}", 100 * POW[(lab, ncl)], 0)
SD_RES = float(np.std(AN.press_eb.values - PRED["+ Hoop G"]))
NCL = round(len(AN) / 3)
put("pw_slope_n", NCL); put("sd_res", SD_RES)


def power_slope(beta, ncl, sims=3000):
    hits = 0
    for _ in range(sims):
        x_ = prng.standard_normal(ncl * NCL)
        y_ = beta * x_ + prng.normal(0, SD_RES, len(x_))
        r_ = stats.linregress(x_, y_)
        hits += (r_.slope / r_.stderr) > stats.t.ppf(0.90, len(x_) - 2)
    return hits / sims


B_OBS = SLOPE_EST = float(NUM["slope_hoop"])
for ncl in (1, 3):
    put(f"pws_obs_{ncl}", 100 * power_slope(B_OBS, ncl), 0)
    put(f"pws_margin_{ncl}", 100 * power_slope(MARGIN, ncl), 0)
POWS = [(k_, power_slope(MARGIN, k_, 1500), power_slope(B_OBS, k_, 1500)) for k_ in range(1, 7)]
POWC = [(k_, power(p0o + (p1o - p0o) / 2, p0o, n1c * k_, n0c * k_, 1500),
         power(p1o, p0o, n1c * k_, n0c * k_, 1500)) for k_ in range(1, 7)]

# ============================================================================= snap-level: what wins on a snap
SN = GR.merge(dl[["nfl_id", "role"]], on="nfl_id")
SN = SN.dropna(subset=["get_off_snap", "g_bend"])
SN = SN[SN.groupby("nfl_id").press.transform("size") >= 8].copy()
for c in ["get_off_snap", "g_bend"]:
    SN[c + "_w"] = SN.groupby("nfl_id")[c].transform(lambda v: (v - v.mean()) / v.std())
SN["quick_w"] = -SN.get_off_snap_w
X = pd.get_dummies(SN.nfl_id.astype(str), drop_first=True).astype(float)
X["quick_w"] = SN.quick_w.values
X["bend_w"] = SN.g_bend_w.values
X["third_down"] = (SN.down == 3).astype(float).values
X["edge"] = SN.edge.values
X = sm.add_constant(X)
fit = sm.GLM(SN.press.values, X, family=sm.families.Binomial()).fit()
ci = fit.conf_int(alpha=0.10)
for c, tag in [("quick_w", "quick"), ("bend_w", "bend")]:
    put(f"or_{tag}", np.exp(fit.params[c])); put(f"or_{tag}_lo", np.exp(ci.loc[c, 0]))
    put(f"or_{tag}_hi", np.exp(ci.loc[c, 1]))
put("sn_n", len(SN)); put("sn_players", SN.nfl_id.nunique())
SN["qq"] = SN.groupby("nfl_id").quick_w.transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))
SN["bq"] = SN.groupby("nfl_id").g_bend_w.transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))
QQ = SN.groupby("qq").press.agg(["mean", "size"])
BQ = SN.groupby("bq").press.agg(["mean", "size"])
put("press_q_fast", 100 * QQ["mean"].iloc[4], 1); put("press_q_slow", 100 * QQ["mean"].iloc[0], 1)
put("press_b_hi", 100 * BQ["mean"].iloc[4], 1); put("press_b_lo", 100 * BQ["mean"].iloc[0], 1)

# ============================================================================= 2025 cards and named players
c25 = sc[(sc.draft_year == 2025) & sc.short].copy()
all25 = ba[(ba.draft_year == 2025) & ba.short].copy().merge(
    sc[["nfl_id", "fc", "hit"]], on="nfl_id", how="left")


def call(r_):
    if r_.rush_snaps >= REG_MIN and r_.press_eb > r_.fc:
        return "Go"
    if r_.rush_snaps >= REG_MIN:
        return "No-go"
    if r_.rush_snaps >= RATE_MIN:
        return "Watch"
    return "Audition"


all25["call"] = all25.apply(call, axis=1)
all25["latest_team"] = all25.latest_team.fillna("FA")
all25 = all25.sort_values(["rush_snaps"], ascending=False)
board = sc[["display_name", "draft_year", "pick", "latest_team", "role", "hoop_z", "burst_z", "rush_snaps",
            "press_eb", "fc", "hit", "short", "day3"]].copy()
board.to_csv(OUT / "board.csv", index=False)


def card_row(r_):
    pk = "UDFA" if r_.pick >= UDFA_PICK else str(int(r_.pick))
    pr = "under 30 rushes" if pd.isna(r_.fc) else f"{r_.press_eb:.1f}% vs {r_.fc:.1f}%"
    return (f"| **{r_.display_name}** ({pk}, {r_.latest_team}) | {r_.role} | {r_.hoop_z:+.1f} | "
            f"{int(r_.rush_snaps)} | {pr} | {r_.call} |")


CARDS = "\n".join(card_row(r_) for _, r_ in all25.iterrows())
for i_, (_, r_) in enumerate(all25.head(3).iterrows()):
    put(f"c{i_}_hoop", r_.hoop_z, 1); put(f"c{i_}_rush", int(r_.rush_snaps))
    if pd.notna(r_.fc):
        put(f"c{i_}_pr", r_.press_eb, 1); put(f"c{i_}_fc", r_.fc, 1)
_ng = all25[all25.call == "No-go"].iloc[0]
put("ng_hoop", _ng.hoop_z, 1); put("ng_rush", int(_ng.rush_snaps)); put("ng_pr", _ng.press_eb, 1); put("ng_fc", _ng.fc, 1)
NOGO = _ng.display_name
NOGO_TEAM = _ng.latest_team
NAMES = list(all25.head(3).display_name)
CALLS = list(all25.head(3).call)
# named contrasts: 2023-24 shortlist outcomes with 100+ rushes
hist = sc[sc.short & sc.draft_year.isin([2023, 2024])].sort_values("press_eb", ascending=False)
HIST_HIT = ", ".join(hist[hist.hit].display_name.iloc[1:])
HIST_MISS = ", ".join(hist[~hist.hit].display_name)
best = hist.iloc[0]
put("best_pr", best.press_eb, 1); put("best_fc", best.fc, 1); put("best_rush", int(best.rush_snaps))
put("best_hoop", best.hoop_z, 1); put("best_pick", int(best.pick))
BEST = best.display_name

# frozen rule hash (rule text + 2025 shortlist)
frozen = {"rule": f"pick>={DAY3} or UDFA AND top third of year-standardized Hoop G among DL hoop runners of the class",
          "hit": "career EB pressure rate above leave-one-class-out baseline forecast (pick, weight, EDGE share, "
                 "NGS athleticism, official 10-yd split, 3-cone); 30+ rushes",
          "test": "primary: OLS slope of year-z Hoop G on (EB pressure rate - frozen baseline forecast) in the 2026 "
                  "class after the 2026 season, one-sided p<0.10; slope <= 0 drops Hoop G as a tiebreaker; secondary: "
                  "shortlist vs rest of day 3 hit rate, one-sided Fisher",
          "shortlist_2025": sorted(all25.display_name)}
blob = json.dumps(frozen, sort_keys=True).encode()
(OUT / "frozen_rule.json").write_text(json.dumps({**frozen, "sha256": hashlib.sha256(blob).hexdigest()}, indent=1))
HASH = hashlib.sha256(blob).hexdigest()[:12]

# ============================================================================= figures
C_H, C_B, C_N = OI["orange"], OI["blue"], OI["grey"]

# Fig 1: headline: slope per sd vs the pre-set margin, and delta R2
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axs[0]
for i_, (k_, lab, col) in enumerate([("hoop_z", "Hoop G (bend)", C_H), ("burst_z", "Rush-drill burst", C_B)]):
    est, bs = SLOPE[k_]
    yy = 1 - i_
    parts = ax.violinplot(bs, positions=[yy], vert=False, widths=0.7, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(col); pc.set_alpha(0.35); pc.set_edgecolor("none")
    lo, hi = np.percentile(bs, [5, 95])
    ax.plot([lo, hi], [yy, yy], color=col, lw=4)
    ax.plot(est, yy, "o", color="white", mec=col, mew=2.5, ms=12)
    ax.text(est, yy + 0.42, f"{est:+.2f} [{lo:+.2f}, {hi:+.2f}]", ha="center", fontsize=12, color="black")
ax.axvspan(-MARGIN, MARGIN, color=C_N, alpha=0.18, lw=0)
ax.axvline(0, color="black", lw=0.8)
ax.text(0, -0.75, f"too small to act on\n(|effect| < {MARGIN} pt per sd)", ha="center", fontsize=11, color="#444",
        bbox=dict(facecolor="#EDEDED", edgecolor="none", pad=1))
ax.set_yticks([1, 0]); ax.set_yticklabels(["Hoop G\n(bend)", "Rush-drill\nburst"])
ax.set_ylim(-1.0, 1.7)
ax.set_xlabel("Pressure-rate points per sd, beyond the no-tracking baseline")
ax.set_title("A. Slope, fitted on two classes at a time", loc="left")
ax = axs[1]
rows_ = [r_ for r_ in TAB1 if "dr" in r_]
for i_, r_ in enumerate(rows_):
    col = {"hoop": C_H, "burst": C_B, "both": OI["green"]}[r_["key"]]
    yy = len(rows_) - 1 - i_
    ax.plot([r_["dlo"], r_["dhi"]], [yy, yy], color=col, lw=4)
    ax.plot(r_["dr"], yy, "o", color="white", mec=col, mew=2.5, ms=12)
    ax.text(r_["dr"], yy + 0.22, f"{r_['dr']:+.2f} [{r_['dlo']:+.2f}, {r_['dhi']:+.2f}]", ha="center", fontsize=11.5)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(rows_))[::-1]); ax.set_yticklabels([r_["model"] for r_ in rows_])
ax.set_ylim(-0.5, len(rows_) - 0.4)
ax.set_xlabel("Scored on the class left out")
ax.set_title("B. Held-out R² gain over baseline", loc="left")
fig.suptitle(f"Bend adds a little, burst adds nothing: Hoop G {NUM['slope_hoop']} pressure-rate pts per sd, "
             f"burst {NUM['slope_burst']}\n{NUM['n_an']} linemen with 30+ NFL rushes, 90% class-stratified bootstrap",
             fontsize=14, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig1_headline.png")

# Fig 2: transfer matrix
fig, ax = plt.subplots(figsize=(FIGW, 5.6))
cols_x = [gy for gy, _ in GAMES]
for i_, (dx, dlab) in enumerate(DRILLS):
    col = C_B if dx == "burst_z" else C_H
    for j_, gy in enumerate(cols_x):
        r, lo, hi, n, ceil = TR[(dx, gy)]
        xx = j_ + (-0.13 if dx == "burst_z" else 0.13)
        ax.plot([xx, xx], [lo, hi], color=col, lw=4, solid_capstyle="butt")
        ax.plot(xx, r, "o", color="white", mec=col, mew=2.5, ms=11)
        if not np.isnan(ceil):
            ax.plot(xx, ceil, marker="_", color=col, ms=22, mew=2.5)
        ax.text(xx + (-0.07 if dx == "burst_z" else 0.07), r, f"{r:+.2f}", fontsize=11.5, va="center",
                ha="right" if dx == "burst_z" else "left")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(cols_x))); ax.set_xticklabels([lab for _, lab in GAMES])
ax.set_ylabel("Partial Spearman ρ\n(after weight and role)")
ax.set_xlim(-0.5, len(cols_x) - 0.5)
ax.plot([], [], "o", color="white", mec=C_B, mew=2.5, ms=10, label="Rush-drill burst")
ax.plot([], [], "o", color="white", mec=C_H, mew=2.5, ms=10, label="Hoop G (bend)")
ax.plot([], [], "_", color="black", ms=18, mew=2.5, label="noise ceiling (both reliabilities)")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
for j_ in range(len(cols_x) - 1):
    ax.axvline(j_ + 0.5, color="#DDDDDD", lw=1)
ax.set_title("Burst reaches the NFL first second, never pressure; Hoop G reaches pressure, not via a trait we see",
             loc="left", fontsize=13)
fig.tight_layout()
save(fig, "fig3_transfer.png")

# Fig 3: hoop laps spanning the Hoop G range + an NFL rush path for the same player
pick_ids = []
hr = ba.dropna(subset=["hoop_eid"]).sort_values("hoop_z")
for q in (0.05, 0.5, 0.95):
    pick_ids.append(hr.iloc[int(q * (len(hr) - 1))].nfl_id)
fig, axs = plt.subplots(1, 3, figsize=(FIGW, 3.9))
norm = plt.Normalize(0, 1.0)
for ax, pid in zip(axs, pick_ids):
    r_ = ba.set_index("nfl_id").loc[pid]
    tr_ = traces[r_.hoop_eid]
    x_, y_, al = tr_["_x"], tr_["_y"], tr_["_alat"]
    x_, y_ = x_ - x_.mean(), y_ - y_.mean()
    pts = np.c_[x_, y_].reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap="cividis", norm=norm, lw=4)
    lc.set_array(al[:-1])
    ax.add_collection(lc)
    k_ = tr_["_k"]
    ax.plot(x_[max(k_ - 2, 0):k_ + 3], y_[max(k_ - 2, 0):k_ + 3], color=OI["red"], lw=8, alpha=0.9)
    ax.set_aspect("equal"); ax.autoscale()
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{r_.display_name}\nHoop G {r_.hoop_g:.2f} g ({r_.hoop_z:+.1f} sd)", fontsize=12.5)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="cividis"), ax=axs, fraction=0.025, pad=0.02)
cb.set_label("Sideways acceleration (g)")
fig.suptitle(f"Run-the-Hoop laps at the 5th, 50th and 95th percentile of Hoop G (median {NUM['hoop_g_median']} g)\n"
             "red band = the half-second of hardest sideways push that defines Hoop G", fontsize=13, x=0.01, ha="left")
save(fig, "fig2_laps.png")
NUM_LAPS = [ba.set_index("nfl_id").loc[p_].display_name for p_ in pick_ids]

# Fig 4: decision: hit rate vs base rate + power
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.0), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axs[0]
bars = [("Day-3 shortlist\n(top-third Hoop G)", k1, n1, C_H), ("Rest of day 3", k0, n0, C_B)]
for i_, (lab, kk, nn, col) in enumerate(bars):
    lo, hi = wilson(kk, nn)
    ax.bar(i_, 100 * kk / nn, color=col, alpha=0.85, width=0.6)
    ax.plot([i_, i_], [100 * lo, 100 * hi], color="black", lw=2)
    ax.text(i_ + 0.08, 100 * hi + 1.5, f"{kk} of {nn}", ha="left", fontsize=12)
for j_, (c, a1, b1, a0, b0) in enumerate(CLS):
    for i_, (aa, bb) in enumerate([(a1, b1), (a0, b0)]):
        if bb:
            ax.plot(i_ - 0.2 + 0.1 * j_, 100 * aa / bb, "D", color="white", mec="#333", ms=7)
ax.axhline(100 * kall / nall, color="black", ls="--", lw=1.2)
ax.text(0.5, 100 * kall / nall + 1.5, f"day-3 base rate {100 * kall / nall:.0f}%", ha="center", fontsize=11)
ax.set_xticks([0, 1]); ax.set_xticklabels([b[0] for b in bars])
ax.set_ylabel("% beating the no-tracking forecast")
ax.set_ylim(0, 100)
ax.plot([], [], "D", color="white", mec="#333", ms=7, label="one class (2023, 24, 25)")
ax.legend(loc="upper right", frameon=False, fontsize=11)
ax.set_title(f"A. Hit rate, 90% Wilson intervals (gap +{NUM['gap']} pts)", loc="left")
ax = axs[1]
kk_ = [p_[0] for p_ in POWC]
ax.axhspan(0, 25, color=C_N, alpha=0.15, lw=0)
ax.text(6.1, 12, "under 25%:\nlikely miss", ha="right", fontsize=11, color="#444")
ax.plot(kk_, [100 * p_[2] for p_ in POWS], "-o", color=C_H, lw=2.5, label=f"slope test, true slope {NUM['slope_hoop']}")
ax.plot(kk_, [100 * p_[1] for p_ in POWS], "--s", color=OI["purple"], lw=2.5, label=f"slope test, true slope {MARGIN}")
ax.plot(kk_, [100 * p_[2] for p_ in POWC], ":^", color=C_N, lw=2, label="shortlist hit-rate test")
ax.axhline(80, color="black", lw=0.8, ls=":")
ax.text(1, 82, "80% power", fontsize=11)
ax.set_xlabel("New Combine classes scored (from 2026)")
ax.set_ylabel("Power at one-sided 0.10 (%)")
ax.set_ylim(0, 100); ax.set_xticks(kk_)
ax.legend(loc="center right", frameon=False, fontsize=10.5)
ax.set_title("B. Simulated power of the frozen 2026 test", loc="left")
fig.tight_layout()
save(fig, "fig5_decision.png")

# Fig 5: snap level, what wins
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 4.6), sharey=True)
for ax, Q, col, lab, tag in [(axs[0], QQ, C_B, "Get-off fifth (his own snaps, slow to quick)", "quick"),
                             (axs[1], BQ, C_H, "Bend fifth (his own snaps, least to most sideways g)", "bend")]:
    pp = Q["mean"].values * 100
    se = np.sqrt(Q["mean"] * (1 - Q["mean"]) / Q["size"]).values * 100 * Z90
    ax.bar(range(1, 6), pp, color=col, alpha=0.85)
    ax.errorbar(range(1, 6), pp, yerr=se, fmt="none", ecolor="black", lw=1.5)
    ax.set_xlabel(lab)
    ax.set_title(f"Odds ratio per within-player sd: {NUM['or_' + tag]} [{NUM['or_' + tag + '_lo']}, "
                 f"{NUM['or_' + tag + '_hi']}]", loc="left", fontsize=12.5)
axs[0].set_ylabel("Pressure rate on the snap (%)")
fig.suptitle(f"Same rusher, different snap: {NUM['sn_n']} random sampled rushes by {NUM['sn_players']} linemen, "
             "player fixed effects, 90% intervals", fontsize=13, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92))
save(fig, "fig4_snaps.png")

# ============================================================================= render
TXT = {"nogo": NOGO, "nogo_team": NOGO_TEAM, "teams": list(all25.head(3).latest_team), "names": NAMES, "calls": CALLS, "cards": CARDS, "hist_hit": HIST_HIT, "hist_miss": HIST_MISS, "best": BEST,
       "hash": HASH, "laps": NUM_LAPS}
TAB1_MD = "\n".join(
    f"| {r_['model']} | {NUM['r2_' + r_['key']]} | "
    f"{NUM['rmse_' + r_['key']]} | {NUM['rho_' + r_['key']]} | "
    + (f"{NUM['dr2_' + r_['key']]} [{NUM['dr2_' + r_['key'] + '_lo']}, {NUM['dr2_' + r_['key'] + '_hi']}]"
       if "dr" in r_ else "—") + " |" for r_ in TAB1)
TAB1_MD = re.sub(r"-(?=\d)", "−", TAB1_MD)
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
(OUT / "text_bits.json").write_text(json.dumps(TXT, indent=1))
tpl = (OUT / "writeup_template.md")
if tpl.exists():
    t = tpl.read_text()

    def sub(m_):
        k_ = m_.group(1)
        if k_ == "TAB1":
            return TAB1_MD
        if k_ in ("CARDS",):
            return CARDS
        if k_.startswith("T:"):
            v_ = TXT[k_[2:].split("[")[0]]
            if "[" in k_:
                v_ = v_[int(k_.split("[")[1].rstrip("]"))]
            return str(v_)
        if k_ not in NUM:
            raise KeyError(k_)
        v_ = str(NUM[k_])
        return v_.replace("-", "−") if v_.startswith("-") else v_
    (OUT / "writeup.md").write_text(re.sub(r"\{\{([^}]+)\}\}", sub, t))
print(json.dumps(NUM, indent=0))
print(json.dumps(TXT, indent=0))
