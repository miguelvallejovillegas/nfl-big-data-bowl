"""Round 17 cand 1: DL/EDGE pass-rush drill and Run-the-Hoop kinematics vs NFL get-off, NFL bend and pressure.

Run from the repo root:
    uv run python writeup/candidates/round_18/cand_1/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (every number the text quotes), board.csv and
figures/*.png next to this file, then renders writeup.md from writeup_template.md ({{key}} -> numbers.json value;
fails on an unknown key). Frame-level features are cached in data/cache/r18c1 (gitignored). Runtime ~6 minutes (2,000 bootstrap refits).

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
CACHE = ROOT / "data" / "cache" / "r18c1"
CACHE.mkdir(parents=True, exist_ok=True)

NBOOT = int(os.environ.get("BDB_NBOOT", 1000))
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
BRES = {"hoop_z": [], "burst_z": []}
for _ in range(NBOOT):
    ix = np.concatenate([rng.choice(g_, len(g_)) for g_ in grp])
    d_ = AN.iloc[ix].reset_index(drop=True)
    yb = d_.press_eb.values
    for nm, cols in MODELS:
        p_, b_ = loco(d_, cols)
        BOOT[nm].append(oos(yb, p_)[0])
        if nm == "Baseline":
            for k_ in ("hoop_z", "burst_z"):
                BRES[k_].append(stats.linregress(d_[k_].values, yb - p_).slope)
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

# ---- headline estimand (pre-specified): out-of-class slope = OLS slope of the held-out residual
# (pressure rate minus the leave-one-class-out baseline forecast) on the year-z drill measure.
# Interval: class-stratified player bootstrap that refits the baseline every fold (players are the
# independent units: effective n = len(AN), 3 classes). p: within-class permutation of the drill measure
# (the baseline does not use the drill, so residuals stay fixed). TOST against the +/-MARGIN band.
HO = {}
prm = np.random.default_rng(77)
NPERM = 5000
for k_, tag in [("hoop_z", "hoop"), ("burst_z", "burst")]:
    x_ = AN[k_].values
    est = stats.linregress(x_, RES).slope
    bs = np.array(BRES[k_])
    lo, hi = np.percentile(bs, [5, 95])
    perm = np.empty(NPERM)
    for i_ in range(NPERM):
        xp = x_.copy()
        for g_ in grp:
            xp[g_] = prm.permutation(xp[g_])
        perm[i_] = stats.linregress(xp, RES).slope
    p_one = (1 + (perm >= est).sum()) / (NPERM + 1)
    # TOST at 0.05 per side == 90% interval inside +/-MARGIN; bootstrap p for each one-sided test
    p_low = (1 + (bs <= -MARGIN).sum()) / (len(bs) + 1)
    p_up = (1 + (bs >= MARGIN).sum()) / (len(bs) + 1)
    HO[k_] = dict(est=est, lo=lo, hi=hi, bs=bs, p=p_one)
    put(f"ho_{tag}", est); put(f"ho_{tag}_lo", lo); put(f"ho_{tag}_hi", hi)
    put(f"ho_{tag}_season", est / 100 * SEASON, 1)
    put(f"ho_{tag}_season_lo", lo / 100 * SEASON, 1); put(f"ho_{tag}_season_hi", hi / 100 * SEASON, 1)
    put(f"ho_{tag}_p", p_one); put(f"ho_{tag}_tost", max(p_low, p_up))
    put(f"ho_{tag}_pabove", 100 * (bs > MARGIN).mean(), 0)
    put(f"ho_{tag}_pinside", 100 * (np.abs(bs) < MARGIN).mean(), 0)
    put(f"ho_{tag}_ppos", 100 * (bs > 0).mean(), 0)
    cls_s = []
    for c, g_ in zip(classes, grp):
        cls_s.append(stats.linregress(x_[g_], RES[g_]).slope)
    HO[k_]["cls"] = cls_s
    put(f"ho_{tag}_pos_classes", int(sum(v > 0 for v in cls_s)))
    for c, v in zip(classes, cls_s):
        put(f"ho_{tag}_{c}", v)
NUM["ho_n_per_class"] = "/".join(str(len(g_)) for g_ in grp)

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

# ---- the decision test: day-3 ties. Two day-3 rushers of the same class whose no-tracking forecasts sit
# within the pre-set margin (TIE pts) are a tie the baseline cannot break. Who rushes better in the NFL?
# Each tiebreaker picks one; "win" = the picked rusher has the higher career (shrunk) pressure rate.
# Inference clusters on players: class-stratified player bootstrap that rebuilds the pairs, and a
# within-class permutation of the tiebreaker for p. Effective n = distinct rushers, not pairs.
TIE = MARGIN
T3 = sc[sc.day3].copy().reset_index(drop=True)
T3["neg_pick"] = -T3.pick
T3["neg_3c"] = -T3.three_cone_z
BREAKERS = [("hoop_z", "Hoop G"), ("neg_pick", "Earlier pick"), ("ngs_athleticism_score", "NGS athleticism"),
            ("neg_3c", "Faster official 3-cone"), ("burst_z", "Rush-drill burst")]


def tie_pairs(t, gap=TIE):
    """all same-class pairs (i < j) of distinct rushers whose forecasts differ by at most `gap` points."""
    fc_, cl_, id_ = t.fc.values, t.draft_year.values, t.nfl_id.values
    I, J = np.triu_indices(len(t), 1)
    ok = (cl_[I] == cl_[J]) & (id_[I] != id_[J]) & (np.abs(fc_[I] - fc_[J]) <= gap)
    return np.c_[I[ok], J[ok]]


def tie_score(t, P, col):
    """win rate (ties in the breaker count half) and mean pressure-rate gain of the picked rusher, pts."""
    if not len(P):
        return np.nan, np.nan, 0
    xa, xb = t[col].values[P[:, 0]], t[col].values[P[:, 1]]
    ok = ~(np.isnan(xa) | np.isnan(xb))
    dy = t.press_eb.values[P[ok, 0]] - t.press_eb.values[P[ok, 1]]
    sg = np.sign(xa[ok] - xb[ok])
    win = np.where(sg == 0, 0.5, (sg * dy > 0).astype(float))
    return float(win.mean()), float((sg * dy).mean()), int(ok.sum())


PAIRS = tie_pairs(T3)
ids_in = np.unique(np.r_[T3.nfl_id.values[PAIRS[:, 0]], T3.nfl_id.values[PAIRS[:, 1]]])
put("tie_pairs", len(PAIRS)); put("tie_players", len(ids_in)); put("tie_pool", len(T3))
put("tie_same_role", 100 * (T3.role.values[PAIRS[:, 0]] == T3.role.values[PAIRS[:, 1]]).mean(), 0)
put("tie_pick_gap_med", np.median(np.abs(T3.pick.values[PAIRS[:, 0]] - T3.pick.values[PAIRS[:, 1]])), 0)
g3 = [np.flatnonzero(T3.draft_year.values == c) for c in classes]
TIEB = {}
trng = np.random.default_rng(303)
NB_T = max(NBOOT, 1000)
for col, lab in BREAKERS:
    w, gain, npair = tie_score(T3, PAIRS, col)
    bw, bg = [], []
    for _ in range(NB_T):
        ix = np.concatenate([trng.choice(g_, len(g_)) for g_ in g3])
        tb = T3.iloc[ix].reset_index(drop=True)
        Pb = tie_pairs(tb)
        a_, b_ = tie_score(tb, Pb, col)[:2]
        bw.append(a_); bg.append(b_)
    TIEB[col] = dict(lab=lab, w=w, gain=gain, n=npair, wlo=np.nanpercentile(bw, 5), whi=np.nanpercentile(bw, 95),
                     glo=np.nanpercentile(bg, 5), ghi=np.nanpercentile(bg, 95))
    tg = {"hoop_z": "hoop", "neg_pick": "pick", "ngs_athleticism_score": "ngs", "neg_3c": "cone",
          "burst_z": "burst"}[col]
    put(f"tie_{tg}_win", 100 * w, 0); put(f"tie_{tg}_lo", 100 * TIEB[col]["wlo"], 0)
    put(f"tie_{tg}_hi", 100 * TIEB[col]["whi"], 0); put(f"tie_{tg}_n", npair)
    put(f"tie_{tg}_gain", gain / 100 * SEASON, 1)
    put(f"tie_{tg}_gain_lo", TIEB[col]["glo"] / 100 * SEASON, 1)
    put(f"tie_{tg}_gain_hi", TIEB[col]["ghi"] / 100 * SEASON, 1)
# permutation p for Hoop G win rate (shuffle Hoop G among day-3 rushers within class)
w_obs = TIEB["hoop_z"]["w"]
pw_, pg_ = [], []
for _ in range(NPERM):
    tp = T3.copy()
    hz = tp.hoop_z.values.copy()
    for g_ in g3:
        hz[g_] = trng.permutation(hz[g_])
    tp["hoop_z"] = hz
    pw_.append(tie_score(tp, PAIRS, "hoop_z")[0])
    pg_.append(tie_score(tp, PAIRS, "hoop_z")[1])
pw_, pg_ = np.array(pw_), np.array(pg_)
put("tie_hoop_p", (1 + (pw_ >= w_obs).sum()) / (NPERM + 1))
put("tie_hoop_gain_p", (1 + (pg_ >= TIEB["hoop_z"]["gain"]).sum()) / (NPERM + 1))
# cost of being wrong: in tied pairs where Hoop G picks the worse rusher, pressures per season given up,
# and the forecast edge it overrides (worst case if Hoop G is pure noise)
xa, xb = T3.hoop_z.values[PAIRS[:, 0]], T3.hoop_z.values[PAIRS[:, 1]]
dy = T3.press_eb.values[PAIRS[:, 0]] - T3.press_eb.values[PAIRS[:, 1]]
sg = np.sign(xa - xb)
gn = sg * dy
put("tie_cost_wrong", -gn[gn < 0].mean() / 100 * SEASON, 1)
put("tie_gain_right", gn[gn > 0].mean() / 100 * SEASON, 1)
dfc = sg * (T3.fc.values[PAIRS[:, 0]] - T3.fc.values[PAIRS[:, 1]])
put("tie_fc_overridden", -dfc[dfc < 0].mean() / 100 * SEASON, 1)
put("tie_fc_override_share", 100 * (dfc < 0).mean(), 0)
put("tie_worst_cost", (dfc < 0).mean() * -dfc[dfc < 0].mean() / 100 * SEASON, 1)
# same-role subset and per-class
same = T3.role.values[PAIRS[:, 0]] == T3.role.values[PAIRS[:, 1]]
put("tie_hoop_win_same", 100 * tie_score(T3, PAIRS[same], "hoop_z")[0], 0); put("tie_n_same", int(same.sum()))
TIE_CLS = []
for c in classes:
    mc = T3.draft_year.values[PAIRS[:, 0]] == c
    TIE_CLS.append(tie_score(T3, PAIRS[mc], "hoop_z")[0])
    put(f"tie_hoop_{c}", 100 * TIE_CLS[-1], 0); put(f"tie_n_{c}", int(mc.sum()))
put("tie_hoop_pos_classes", int(sum(v > 0.5 for v in TIE_CLS)))
# sweep of the tie width (decision threshold): Hoop G vs the forecast itself as tiebreaker
SWEEP = []
T3["fcv"] = T3.fc
for gw in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
    Pg = tie_pairs(T3, gw)
    SWEEP.append((gw, tie_score(T3, Pg, "hoop_z")[0], tie_score(T3, Pg, "fcv")[0], len(Pg)))
put("sweep_wide_hoop", 100 * SWEEP[-1][1], 0); put("sweep_wide_fc", 100 * SWEEP[-1][2], 0)
put("sweep_wide_n", SWEEP[-1][3]); put("sweep_wide_gap", SWEEP[-1][0], 1)

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
SD_RES = float(np.std(RES))
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


B_OBS = float(HO["hoop_z"]["est"])
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
all25 = ba[(ba.draft_year == 2025) & ba.short].copy().merge(sc[["nfl_id", "fc", "hit"]], on="nfl_id", how="left")


def call(r_):
    if r_.rush_snaps < RATE_MIN:
        return "Audition"
    if r_.rush_snaps >= REG_MIN and r_.press_eb - r_.fc >= MARGIN:
        return "Go"
    if r_.rush_snaps >= REG_MIN and r_.press_eb - r_.fc <= -MARGIN:
        return "No-go"
    return "Hold"


def flip_rate(r_):
    """2026 pressure rate (%) that moves the shrunk career rate onto the frozen forecast, if he rushes as
    often in 2026 as in 2025."""
    R = r_.rush_snaps
    return 100 * ((r_.fc / 100) * (r_.rush_snaps + R + Mw) - r_.pressures - Mw * mu) / R


all25["call"] = all25.apply(call, axis=1)
all25["latest_team"] = all25.latest_team.fillna("FA")
all25["over"] = all25.press_eb - all25.fc
all25["flip"] = [flip_rate(r_) if r_.rush_snaps >= REG_MIN else np.nan for _, r_ in all25.iterrows()]
all25["raw"] = 100 * all25.pressures / all25.rush_snaps
order = {"Go": 0, "No-go": 1, "Hold": 2, "Audition": 3}
all25 = all25.sort_values(["call", "rush_snaps"], key=lambda c_: c_.map(order) if c_.name == "call" else -c_)
board = sc[["display_name", "draft_year", "pick", "latest_team", "role", "hoop_z", "burst_z", "rush_snaps",
            "press_eb", "fc", "hit", "short", "day3"]].copy()
board.to_csv(OUT / "board.csv", index=False)


def card_row(r_):
    pk = "UDFA" if r_.pick >= UDFA_PICK else str(int(r_.pick))
    pr = f"{r_.press_eb:.1f}% vs {r_.fc:.1f}%" if r_.rush_snaps >= RATE_MIN else "under 30 rushes"
    fl = f"{r_.flip:.1f}%" if pd.notna(r_.flip) else "needs 100 rushes"
    return (f"| **{r_.display_name}** ({pk}, {r_.latest_team}) | {r_.role} | {r_.hoop_z:+.1f} | "
            f"{int(r_.rush_snaps)} | {pr} | **{r_.call}** | {fl} |")


big = all25[all25.call != "Audition"]
CARDS = "\n".join(card_row(r_) for _, r_ in big.iterrows())
AUD = all25[all25.call == "Audition"]
AUD_TXT = ", ".join(f"{r_.display_name} ({'UDFA' if r_.pick >= UDFA_PICK else int(r_.pick)}, {r_.latest_team})"
                    for _, r_ in AUD.iterrows())
put("n_short25", len(all25)); put("n_aud25", len(AUD))
GO = all25[all25.call == "Go"]
for i_, (_, r_) in enumerate(GO.iterrows()):
    put(f"go{i_}_hoop", r_.hoop_z, 1); put(f"go{i_}_rush", int(r_.rush_snaps)); put(f"go{i_}_pr", r_.press_eb, 1)
    put(f"go{i_}_fc", r_.fc, 1); put(f"go{i_}_flip", r_.flip, 1); put(f"go{i_}_raw", r_.raw, 1)
    put(f"go{i_}_over", r_.over / 100 * SEASON, 1)
_ng = all25[all25.call == "No-go"].iloc[0]
put("ng_hoop", _ng.hoop_z, 1); put("ng_rush", int(_ng.rush_snaps)); put("ng_pr", _ng.press_eb, 1)
put("ng_fc", _ng.fc, 1); put("ng_flip", _ng.flip, 1)
NOGO, NOGO_TEAM = _ng.display_name, _ng.latest_team
NAMES, TEAMS = list(GO.display_name), list(GO.latest_team)
# named history: 2023-24 shortlist outcomes
hist = sc[sc.short & sc.draft_year.isin([2023, 2024])].sort_values("press_eb", ascending=False)
HIST_HIT = ", ".join(hist[hist.hit].display_name.iloc[1:4])
HIST_MISS = ", ".join(hist[~hist.hit].display_name.iloc[:3])
best = hist.iloc[0]
put("best_pr", best.press_eb, 1); put("best_fc", best.fc, 1); put("best_rush", int(best.rush_snaps))
put("best_hoop", best.hoop_z, 1); put("best_pick", int(best.pick))
BEST = best.display_name

# frozen rule hash (rule text + 2025 calls)
frozen = {"rule": f"Day 3 (pick>={DAY3}) or UDFA: when two rushers' no-tracking forecasts are within {TIE} "
                  "pressure-rate points, prefer the higher year-standardized Hoop G; otherwise follow the forecast",
          "forecast": "OLS of shrunk career pressure rate on log pick, weight, EDGE share, NGS athleticism, official "
                      "10-yd split, 3-cone, fit on other classes",
          "test": "primary: 2026 Combine class after its first NFL season, slope of Hoop G on (pressure rate - frozen "
                  "forecast), within-class permutation, one-sided p<0.10; slope <= 0 drops Hoop G. Secondary: tied-pair "
                  "win rate of Hoop G vs 50%",
          "calls_2025": {r_.display_name: r_.call for _, r_ in all25.iterrows()},
          "recheck": "after the 2026 regular season with 2026 pressure counts; flip rates in board"}
blob = json.dumps(frozen, sort_keys=True).encode()
(OUT / "frozen_rule.json").write_text(json.dumps({**frozen, "sha256": hashlib.sha256(blob).hexdigest()}, indent=1))
HASH = hashlib.sha256(blob).hexdigest()[:12]

# ============================================================================= verification inputs
INPUT_COLS = {
    "nfl_id": "player id", "display_name": "player", "draft_year": "Combine class (fold id for leave-one-class-out)",
    "pick": "overall pick, undrafted = 262", "log_pick": "ln(pick)",
    "combine_weight_z": "Combine weight, z within class", "edge_share": "share of NFL snaps at EDGE/OLB (0-1)",
    "ngs_athleticism_score": "NGS athleticism score, missing filled with the DL median",
    "ten_yd_split_z": "official 10-yd split, z within class, blank if not run (filled inside each fold)",
    "three_cone_z": "official 3-cone, z within class, blank if not run (filled inside each fold)",
    "hoop_z": "Hoop G, z within class", "burst_z": "rush-drill burst, z within class (blank if no clean rep)",
    "rush_snaps": "career NFL pass-rush snaps 2023-25", "pressures": "career NFL pressures 2023-25",
    "press_eb": f"pressure rate %, shrunk toward {100 * mu:.2f}% with {Mw:.1f} prior rushes",
    "fc": "leave-one-class-out baseline forecast of press_eb (%), fit on all rows of the other two classes; "
          "blank for linemen without a hoop lap", "day3": "pick >= 103 or undrafted",
    "in_an": "in the model set (both drills, 30+ rushes)"}
inp = co30.merge(sc[["nfl_id", "fc"]], on="nfl_id", how="left")
inp["day3"] = inp.pick >= DAY3
inp["in_an"] = inp.nfl_id.isin(AN.nfl_id)
inp[list(INPUT_COLS)].to_csv(OUT / "inputs.csv", index=False)
(OUT / "inputs_schema.json").write_text(json.dumps({"columns": INPUT_COLS, "margin_pts": MARGIN,
                                                     "season_rushes": SEASON, "tie_width_pts": TIE}, indent=1))

# ============================================================================= figures
C_H, C_B, C_N = OI["orange"], OI["blue"], OI["grey"]
PS = SEASON / 100   # pressure-rate points -> pressures per 450-rush season
MS = MARGIN * PS

# Fig 1: headline, out-of-class effect per sd in pressures per season, vs the pre-set margin
fig, ax = plt.subplots(figsize=(FIGW, 4.4))
for i_, (k_, lab, col) in enumerate([("hoop_z", "Hoop G\n(bend)", C_H), ("burst_z", "Rush-drill\nburst", C_B)]):
    h = HO[k_]
    yy = 1 - i_
    parts = ax.violinplot(h["bs"] * PS, positions=[yy], vert=False, widths=0.6, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(col); pc.set_alpha(0.30); pc.set_edgecolor("none")
    ax.plot([h["lo"] * PS, h["hi"] * PS], [yy, yy], color=col, lw=5, solid_capstyle="butt")
    for j_, (c, v) in enumerate(zip(classes, h["cls"])):
        ax.plot(v * PS, yy - 0.30, marker="D", color="white", mec="#333", ms=7)
        ax.text(v * PS, yy - 0.47, str(c), ha="center", fontsize=9.5, color="#333")
    ax.plot(h["est"] * PS, yy, "o", color="white", mec=col, mew=3, ms=13, zorder=5)
    ax.text(h["hi"] * PS + 0.25, yy + 0.05, f"{h['est'] * PS:+.1f}  [{h['lo'] * PS:+.1f}, {h['hi'] * PS:+.1f}]",
            ha="left", va="center", fontsize=12.5)
ax.axvspan(-MS, MS, color=C_N, alpha=0.16, lw=0)
ax.axvline(0, color="black", lw=0.8)
ax.text(-MS + 0.1, 1.62, f"too small to act on (within ±{MS:.2f})", ha="left", fontsize=11, color="#444")
ax.set_yticks([1, 0]); ax.set_yticklabels(["Hoop G\n(bend)", "Rush-drill\nburst"])
ax.set_ylim(-0.65, 1.8)
ax.set_xlim(-6.5, 8.5)
ax.set_xlabel("Extra pressures per 450-rush season, per sd of the drill, beyond the no-tracking forecast")
ax.plot([], [], "D", color="white", mec="#333", ms=7, label="one held-out class")
ax.legend(loc="lower right", frameon=False, fontsize=11)
ax.set_title(f"Scored on the class left out (n = {NUM['n_an']}): burst cannot reach the action line; Hoop G is "
             "positive but may sit below it", loc="left", fontsize=13)
fig.tight_layout()
save(fig, "fig1_headline.png")

# Fig 2: hoop laps spanning the Hoop G range
pick_ids = []
hr = ba.dropna(subset=["hoop_eid"]).sort_values("hoop_z")
for q in (0.05, 0.5, 0.95):
    pick_ids.append(hr.iloc[int(q * (len(hr) - 1))].nfl_id)
fig, axs = plt.subplots(1, 3, figsize=(FIGW, 3.3))
fig.subplots_adjust(left=0.01, right=0.86, top=0.74, bottom=0.02, wspace=0.05)
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
    ax.plot(x_[max(k_ - 2, 0):k_ + 3], y_[max(k_ - 2, 0):k_ + 3], color=OI["red"], lw=9, alpha=0.9,
            solid_capstyle="round")
    ax.plot(x_[0], y_[0], "o", color="black", ms=6)
    ax.set_aspect("equal"); ax.autoscale()
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{r_.display_name}\nHoop G {r_.hoop_g:.2f} g ({r_.hoop_z:+.1f} sd)", fontsize=12.5)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="cividis"), cax=fig.add_axes([0.875, 0.08, 0.014, 0.6]))
cb.set_label("Sideways acceleration (g)")
fig.suptitle(f"Run-the-Hoop laps at the 5th, 50th and 95th percentile of Hoop G (median {NUM['hoop_g_median']} g)\n"
             "black dot = start of the lap; vermillion band = the half-second of hardest sideways push that defines "
             "Hoop G", fontsize=12.5, x=0.01, ha="left")
save(fig, "fig2_laps.png")
NUM_LAPS = [ba.set_index("nfl_id").loc[p_].display_name for p_ in pick_ids]

# Fig 3: transfer matrix
fig, ax = plt.subplots(figsize=(FIGW, 5.4))
cols_x = [gy for gy, _ in GAMES]
for i_, (dx, dlab) in enumerate(DRILLS):
    col = C_B if dx == "burst_z" else C_H
    for j_, gy in enumerate(cols_x):
        r, lo, hi, n, ceil = TR[(dx, gy)]
        xx = j_ + (-0.14 if dx == "burst_z" else 0.14)
        ax.plot([xx, xx], [lo, hi], color=col, lw=5, solid_capstyle="butt")
        ax.plot(xx, r, "o", color="white", mec=col, mew=2.5, ms=11)
        if not np.isnan(ceil):
            ax.plot(xx, ceil, marker="_", color=col, ms=24, mew=3)
        ax.text(xx + (-0.08 if dx == "burst_z" else 0.08), r, f"{r:+.2f}", fontsize=11.5, va="center",
                ha="right" if dx == "burst_z" else "left")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(cols_x))); ax.set_xticklabels([lab for _, lab in GAMES])
ax.set_ylabel("Partial Spearman ρ\n(after weight and role)")
ax.set_xlim(-0.5, len(cols_x) - 0.5)
ax.set_ylim(-0.35, 0.72)
ax.plot([], [], "o", color="white", mec=C_B, mew=2.5, ms=10, label="Rush-drill burst")
ax.plot([], [], "o", color="white", mec=C_H, mew=2.5, ms=10, label="Hoop G (bend)")
ax.plot([], [], "_", color="black", ms=18, mew=3, label="most the link could show (noise ceiling)")
ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=11)
for j_ in range(len(cols_x) - 1):
    ax.axvline(j_ + 0.5, color="#DDDDDD", lw=1)
ax.set_title("Burst carries to the NFL first second but not to pressure; Hoop G reaches pressure through no "
             "trait we can see", loc="left", fontsize=13)
fig.tight_layout()
save(fig, "fig3_transfer.png")

# Fig 4: decision. A: tied day-3 pairs, win rate by tiebreaker. B: pressures gained per decision
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 4.9), sharey=True)
order_b = ["hoop_z", "neg_pick", "ngs_athleticism_score", "burst_z", "neg_3c"]
cols_b = {"hoop_z": C_H, "neg_pick": OI["black"], "ngs_athleticism_score": OI["purple"], "neg_3c": OI["green"],
          "burst_z": C_B}
labs_b = {"hoop_z": "Higher Hoop G", "neg_pick": "Earlier pick", "ngs_athleticism_score": "Higher NGS score",
          "neg_3c": "Faster 3-cone\n(only 13 pairs ran it)", "burst_z": "More rush-drill\nburst"}
for ax, key, lo_, hi_, sc_, ref, xl, xlim in [
        (axs[0], "w", "wlo", "whi", 100, 50, "% of tied pairs where the pick rushed better in the NFL", (25, 100)),
        (axs[1], "gain", "glo", "ghi", PS, 0, "Pressures per 450-rush season gained by the pick", (-4.5, 9.5))]:
    for i_, col in enumerate(order_b):
        t_ = TIEB[col]
        yy = len(order_b) - 1 - i_
        alpha = 0.45 if col == "neg_3c" else 1
        ax.plot([sc_ * t_[lo_], min(sc_ * t_[hi_], xlim[1])], [yy, yy], color=cols_b[col], lw=5,
                solid_capstyle="butt", alpha=alpha)
        ax.plot(sc_ * t_[key], yy, "o", color="white", mec=cols_b[col], mew=3, ms=12, zorder=5, alpha=alpha)
        txt = f"{sc_ * t_[key]:.0f}%" if key == "w" else f"{sc_ * t_[key]:+.1f}"
        ax.text(sc_ * t_[key], yy + 0.27, txt, ha="center", fontsize=11.5)
    ax.axvline(ref, color="black", lw=1, ls="--")
    ax.set_xlim(*xlim)
    ax.set_xlabel(xl, fontsize=12)
    ax.set_ylim(-0.6, len(order_b) - 0.4)
axs[0].set_yticks(range(len(order_b))[::-1]); axs[0].set_yticklabels([labs_b[c] for c in order_b])
axs[0].text(50.8, -0.5, "coin flip", fontsize=11)
axs[0].set_title("A. How often the tiebreaker is right", loc="left", fontsize=12.5)
axs[1].set_title("B. How much it is worth when used", loc="left", fontsize=12.5)
fig.suptitle(f"Day-3 ties ({NUM['tie_pairs']} pairs of same-class rushers whose forecasts sit within {TIE} pt; "
             f"{NUM['tie_players']} rushers, 90% player-clustered intervals)", fontsize=13, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig4_tiebreak.png")

# ============================================================================= render
TXT = {"nogo": NOGO, "nogo_team": NOGO_TEAM, "teams": TEAMS, "names": NAMES, "cards": CARDS, "aud": AUD_TXT,
       "hist_hit": HIST_HIT, "hist_miss": HIST_MISS, "best": BEST, "hash": HASH, "laps": NUM_LAPS}
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
        if k_ == "CARDS":
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
