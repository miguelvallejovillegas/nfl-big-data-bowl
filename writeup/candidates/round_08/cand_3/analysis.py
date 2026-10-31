"""Movement translation for pass rushers: does a rusher's Combine start show up in his NFL start, and does the
translation explain pressure beyond draft slot?

Run from the repo root:
    uv run python writeup/candidates/round_08/cand_3/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (only the values quoted in writeup.md), claims.json
(key -> sentence that uses it), tables.json, board.csv and figures/*.png next to this file, then renders writeup.md
from writeup_template.md. Frame-level curve extraction is cached in data/cache/r8c3 (gitignored).
Runtime: about 1 minute cold, one core.

Design
  * Same movement on both sides: the first 1.5 s of a start from a stance, as a speed-vs-time curve sampled every
    0.1 s from first movement (speed first reaches 1 yd/s, minus 0.2 s). Combine side: the 40 (tracked, every
    group) and, for DL, the pass-rush drill. NFL side: random sampled snaps (game_frames_sample, sample_kind ==
    random, pass plays) where the player is set at the snap (< 0.8 yd/s); the top-EPA half is never used.
  * First step = mean speed 0.3-0.5 s after first movement; drive = 1.3-1.5 s. Sunday speeds are adjusted per snap
    for alignment (EDGE / interior / OLB) before averaging per player.
  * Translation(t) = partial Spearman of Combine speed at t with Sunday speed at t, after weight (and NFL EDGE share
    for DL). Payoff(t) = partial Spearman of Sunday speed at t with pressure rate after log pick, weight, EDGE share.
  * Per-player translation = Carry-over: Sunday first step minus Combine first step, each in sd after weight and
    role. Five per-player similarity definitions are reported (none chosen on outcome).
  * Out of class: fit on two classes, score the third; rush-weighted R^2 change over the no-tracking baseline
    (log pick, weight, EDGE share), bootstrap over held-out rushers for the pooled interval.
  * Calibration: share of rushers beating their out-of-class no-tracking forecast, exact (Clopper-Pearson) 90%
    intervals.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
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
CACHE = ROOT / "data" / "cache" / "r8c3"


# ============================================================================ rendering
def render(out_dir: Path, num: dict, md_tables: dict):
    tpl = out_dir / "writeup_template.md"
    if not tpl.exists():
        return set()
    raw = tpl.read_text()
    text = re.sub(r"\{\{TABLE:(\w+)\}\}", lambda m: md_tables[m.group(1)], raw)
    claims, used = {}, set()
    for line in text.splitlines():
        for k in re.findall(r"\{\{(\w+)\}\}", line):
            claims.setdefault(k, re.sub(r"\{\{(\w+)\}\}", lambda m: str(num.get(m.group(1), "?")), line)
                              .strip()[:180])

    def sub(m):
        k = m.group(1)
        if k not in num:
            raise KeyError(f"writeup_template.md uses unknown key {k}")
        used.add(k)
        v = str(num[k])
        return "−" + v[1:] if v.startswith("-") else v

    (out_dir / "writeup.md").write_text(re.sub(r"\{\{(\w+)\}\}", sub, text))
    json.dump({k: {"value": num[k], "claim": claims.get(k, "")} for k in sorted(used)},
              open(out_dir / "claims.json", "w"), indent=1, ensure_ascii=False)
    return used


FIG.mkdir(parents=True, exist_ok=True)
for old in FIG.glob("*.png"):
    old.unlink()
CACHE.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(2027)
NBOOT = int(os.environ.get("BDB_NBOOT", 1000))
UDFA_PICK = 262
TW = 16                      # 0.0-1.5 s
FS = slice(3, 6)             # first step: 0.3-0.5 s
DR = slice(13, 16)           # drive: 1.3-1.5 s
MIN_SNAPS_G = 8              # random NFL starts per player
MIN_RUSH = 100               # career pass rushes for a pressure rate
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
FIGW = 1600 / 150
T0 = _time.time()
NUM: dict = {}
TABLES: dict = {"md": {}}


def tick(s):
    print(f"[{_time.time() - T0:6.1f}s] {s}", flush=True)


def put(key, v, nd=2):
    if key in NUM:
        raise KeyError(f"duplicate key {key}")
    if isinstance(v, (int, np.integer)):
        NUM[key] = int(v)
        return NUM[key]
    s = f"{float(v):.{nd}f}"
    if re.fullmatch(r"-0\.?0*", s):
        s = s[1:]
    NUM[key] = s
    return s


def fmt(v, nd=2):
    s = f"{float(v):.{nd}f}"
    if re.fullmatch(r"-0\.?0*", s):
        s = s[1:]
    return s


def ordn(v):
    v = int(round(v))
    suf = "th" if 10 <= v % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(v % 10, "th")
    return f"{v}{suf}"


def save(fig, name):
    fig.set_size_inches(FIGW, fig.get_size_inches()[1])
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def yz(s, year):
    return s.groupby(year).transform(lambda v: (v - v.mean()) / v.std())


def onset(sm, start=0, thr=1.0):
    hit = np.flatnonzero(sm[start:] >= thr)
    if not len(hit):
        return None
    return start + max(int(hit[0]) - 2, 0)


def _res(v, A):
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]


def partial(df, x, y, Z, nb=NBOOT, ci=90):
    """Spearman partial correlation (Pearson r of baseline-residualized ranks) + percentile bootstrap by player."""
    d = df[[x, y] + Z].dropna()
    R = d.rank().to_numpy(float)
    A = np.c_[np.ones(len(R)), R[:, 2:]]
    r = float(np.corrcoef(_res(R[:, 0], A), _res(R[:, 1], A))[0, 1])
    n = len(d)
    b = np.empty(nb)
    for k in range(nb):
        i = RNG.integers(0, n, n)
        b[k] = np.corrcoef(_res(R[i, 0], A[i]), _res(R[i, 1], A[i]))[0, 1]
    lo, hi = np.nanpercentile(b, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return r, float(lo), float(hi), n, float(np.nanmean(b > 0))


def cp(k, n, conf=0.90):
    a = 1 - conf
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


# ============================================================================ load
tick("load")
players = pd.read_parquet(PROC / "players.parquet")
cres = pd.read_parquet(PROC / "combine_results.parquet")
events = pd.read_parquet(PROC / "combine_events.parquet")
seasons = pd.read_parquet(PROC / "player_season_stats.parquet")
align = pd.read_parquet(PROC / "player_season_alignment.parquet")
gp = pd.read_parquet(PROC / "game_plays_sample.parquet")
pv = pd.read_parquet(PROC / "player_value.parquet")

pl = players[["nfl_id", "display_name", "draft_year", "position_group", "draft_overall_pick"]].merge(
    cres[["nfl_id", "combine_weight", "forty", "ten_yd_split"]], on="nfl_id", how="left")
pl["pick"] = pl.draft_overall_pick.fillna(UDFA_PICK)
pl["log_pick"] = np.log(pl.pick)
pl["w_z"] = pl.groupby(["position_group", "draft_year"]).combine_weight.transform(lambda v: (v - v.mean()) / v.std())
pl["forty_z"] = -pl.groupby(["position_group", "draft_year"]).forty.transform(lambda v: (v - v.mean()) / v.std())

# ============================================================================ curves (cached)
K = ["game_id", "play_id", "nfl_id"]


def extract():
    ev = events[(events.drill_name == "FORTY_YARD_DASH") |
                ((events.drill_name == "PASS_RUSH_DRILL") & (events.position_group == "DL"))]
    fr = pd.read_parquet(PROC / "combine_frames.parquet", filters=[("event_id", "in", list(ev.event_id))],
                         columns=["event_id", "frame", "entity_type", "s"])
    fr = fr[fr.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
    crv = {}
    for eid, g in fr.groupby("event_id", sort=False):
        s = g.s.to_numpy()
        if len(s) < 20:
            continue
        sm = savgol_filter(s, 5, 2)
        rest = np.flatnonzero(sm < 0.8)
        if not len(rest) or rest[0] > 20:
            continue
        o = onset(sm, rest[0])
        if o is None or o + TW > len(sm):
            continue
        crv[eid] = np.clip(sm[o:o + TW], 0, None)
    DC = pd.DataFrame({"event_id": list(crv), "curve": list(crv.values())}).merge(
        ev[["event_id", "nfl_id", "draft_year", "drill_name", "position_group", "attempt_no"]], on="event_id")
    g = gp[(gp.sample_kind == "random") & gp.snap_frame.notna()]
    gf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "s"])
    gf = gf.merge(g[K + ["snap_frame"]], on=K).sort_values(K + ["frame"])
    rows = []
    for k, h in gf.groupby(K, sort=False):
        sf = h.snap_frame.iloc[0]
        s, fr_ = h.s.to_numpy(), h.frame.to_numpy()
        i0 = int(np.searchsorted(fr_, sf))
        if len(s) < i0 + TW + 3:
            continue
        sm = savgol_filter(s, 5, 2)
        if sm[i0] >= 0.8:
            continue
        o = onset(sm, i0)
        if o is None or o + TW > len(sm) or o - i0 > 10:
            continue
        rows.append(dict(zip(K, k), react=(o - i0) / 10, curve=np.clip(sm[o:o + TW], 0, None)))
    GC = pd.DataFrame(rows).merge(g[K + ["season", "lined_up_position", "pass_result", "player_get_off"]], on=K)
    return DC, GC


cf = CACHE / "curves.pkl"
if cf.exists():
    DC, GC = pickle.load(open(cf, "rb"))
else:
    tick("extract curves from frames")
    DC, GC = extract()
    pickle.dump((DC, GC), open(cf, "wb"))
GC = GC.merge(pl[["nfl_id", "position_group"]], on="nfl_id")
MG = np.vstack(GC.curve.to_numpy())
GC["fs"], GC["dr"] = MG[:, FS].mean(1), MG[:, DR].mean(1)
for t in range(TW):
    GC[f"v{t}"] = MG[:, t]
VC = [f"v{t}" for t in range(TW)] + ["fs", "dr"]
# alignment adjustment per snap (within position group)
for c in VC:
    GC[c + "_a"] = GC[c] - GC.groupby(["position_group", "lined_up_position"])[c].transform("mean") + \
        GC.groupby("position_group")[c].transform("mean")
MD = np.vstack(DC.curve.to_numpy())
DC["fs"], DC["dr"] = MD[:, FS].mean(1), MD[:, DR].mean(1)
for t in range(TW):
    DC[f"v{t}"] = MD[:, t]

# ============================================================================ DL player table
tick("DL table")
dl = pl[pl.position_group == "DL"].copy()
a = align.merge(dl[["nfl_id"]], on="nfl_id")
a["edge_pl"] = np.where(a.lined_up_position.isin(["EDGE", "OLB"]), a.plays, 0)
es = a.groupby("nfl_id").agg(edge_pl=("edge_pl", "sum"), pl_=("plays", "sum"))
dl = dl.merge((es.edge_pl / es.pl_).rename("edge").reset_index(), on="nfl_id", how="left")
dl["edge"] = dl.edge.fillna(0.0)
dl["role"] = np.where(dl.edge >= 0.5, "EDGE", "interior")
s = seasons[seasons.position_group == "DL"].copy()
s["w_go"] = s.pass_rush_snaps * s.avg_get_off
ag = s.groupby("nfl_id").agg(rs=("pass_rush_snaps", "sum"), pr=("pressures", "sum"), w_go=("w_go", "sum")).reset_index()
ag["getoff"] = ag.w_go / ag.rs
dl = dl.merge(ag, on="nfl_id", how="left")
dl["rs"] = dl.rs.fillna(0)
m = dl.rs >= 20
n_, k_ = dl.loc[m, "rs"], dl.loc[m, "pr"]
mu = k_.sum() / n_.sum()
N_ = n_.sum()
tau2 = ((n_ * (k_ / n_ - mu) ** 2).sum() - (len(n_) - 1) * mu * (1 - mu)) / (N_ - (n_ ** 2).sum() / N_)
Mw = mu * (1 - mu) / tau2 - 1
dl["prate"] = dl.pr / dl.rs.where(dl.rs > 0)
dl["peb"] = (dl.pr + Mw * mu) / (dl.rs + Mw)

gd = GC[(GC.position_group == "DL") & GC.pass_result.notna()]
gplayer = gd.groupby("nfl_id").agg(n_g=("fs", "size"), react=("react", "mean"),
                                   **{c: (c + "_a", "mean") for c in VC},
                                   **{c + "_u": (c, "mean") for c in VC}).reset_index()
gplayer = gplayer[gplayer.n_g >= MIN_SNAPS_G]
d40 = DC[(DC.drill_name == "FORTY_YARD_DASH") & (DC.position_group == "DL")]
c40 = d40.groupby("nfl_id")[VC].mean()
c40 = c40.add_prefix("c_").reset_index()
dl = dl.merge(gplayer, on="nfl_id", how="left").merge(c40, on="nfl_id", how="left")
for c in [f"c_{v}" for v in VC]:
    dl[c] = yz(dl[c], dl.draft_year)

put("n_snaps_dl", len(gd[gd.nfl_id.isin(gplayer.nfl_id)]), 0)
put("snaps_med", int(gplayer.n_g.median()))
put("fs_med_sunday", gplayer.fs.median(), 1)
put("fs_med_40", d40.groupby("nfl_id").fs.mean().median(), 1)
put("dr_med_sunday", gplayer.dr.median(), 1)
put("dr_med_40", d40.groupby("nfl_id").dr.mean().median(), 1)

# ---- measurement checks
gdd = gd[gd.nfl_id.isin(gplayer.nfl_id)]
rel_t = []
for c in [f"v{t}_a" for t in range(TW)] + ["fs_a"]:
    q = gdd.groupby("nfl_id")[c].agg(["size", "mean", "var"])
    s2 = np.average(q["var"], weights=q["size"] - 1)
    t2 = q["mean"].var() - s2 * (1 / q["size"]).mean()
    rel_t.append(t2 / (t2 + s2 / q["size"].median()))
put("rel_fs_game", rel_t[-1])
put("rel_fs_1snap", rel_t[-1] / (q["size"].median() - (q["size"].median() - 1) * rel_t[-1]))
rep = d40[d40.groupby("nfl_id").nfl_id.transform("size") >= 2].sort_values(["nfl_id", "attempt_no"])
r1 = rep.groupby("nfl_id").nth(0).set_index("nfl_id")
r2 = rep.groupby("nfl_id").nth(1).set_index("nfl_id").reindex(r1.index)
rel40_t = [float(np.corrcoef(r1[f"v{t}"], r2[f"v{t}"])[0, 1]) for t in range(TW)]
put("rel40_fs", np.corrcoef(r1.fs, r2.fs)[0, 1]); put("rel40_dr", np.corrcoef(r1.dr, r2.dr)[0, 1])
put("rel40_n", len(r1))
sn = gdd.dropna(subset=["player_get_off"])
put("rho_fs_ngs_snap", -stats.spearmanr(sn.fs_a, sn.player_get_off)[0]); put("n_ngs_snap", len(sn))
pg = dl.dropna(subset=["fs", "getoff"])
put("rho_fs_ngs_player", -stats.spearmanr(pg.fs, pg.getoff)[0]); put("n_ngs_player", len(pg))

# ============================================================================ translation and payoff curves (DL)
tick("curves")
tr_set = dl.dropna(subset=["fs", "c_fs", "w_z"]).copy()
pay_set = dl[(dl.rs >= MIN_RUSH)].dropna(subset=["fs", "w_z", "peb"]).copy()
both = pay_set.dropna(subset=["c_fs"]).copy()
put("n_tr", len(tr_set)); put("n_pay", len(pay_set)); put("n_both", len(both))
ZB = ["log_pick", "w_z", "edge"]
ZS = ["w_z", "edge"]
CUR = {"raw": [], "adj": [], "off": [], "pay": [], "pay40": []}
for t in range(TW):
    v, cv = f"v{t}", f"c_v{t}"
    CUR["raw"].append(partial(tr_set, cv, v + "_u", [], nb=400))
    CUR["adj"].append(partial(tr_set, cv, v, ZS, nb=400))
    CUR["off"].append(partial(tr_set, "forty_z", v, ZS, nb=400))
    CUR["pay"].append(partial(pay_set, v, "peb", ZB, nb=400))
    CUR["pay40"].append(partial(both, cv, "peb", ZB, nb=400))
for nm, key in [("raw", "tr_raw"), ("adj", "tr_adj"), ("off", "tr_off"), ("pay", "pay")]:
    for lab, t in [("03", 3), ("15", 15)]:
        r, lo, hi, n, _ = CUR[nm][t]
        put(f"{key}_{lab}", r); put(f"{key}_{lab}_lo", lo); put(f"{key}_{lab}_hi", hi)
put("tr_raw_max", max(c[0] for c in CUR["raw"]))
put("tr_adj_absmax", max(abs(c[0]) for c in CUR["adj"][3:]))
put("pay_max", max(c[0] for c in CUR["pay"][3:]))
put("pay40_absmax", max(abs(c[0]) for c in CUR["pay40"][3:]))
put("rho_w_dr_sunday", -stats.spearmanr(tr_set.combine_weight, tr_set.dr_u)[0])
put("rho_w_dr_40", -stats.spearmanr(tr_set.combine_weight, tr_set.c_dr)[0])
put("rho_w_40", stats.spearmanr(tr_set.combine_weight, tr_set.forty)[0])
# noise ceiling on the size-adjusted first-step translation
r, lo, hi, n, _ = partial(tr_set, "c_fs", "fs", ZS)
put("tr_fs", r); put("tr_fs_lo", lo); put("tr_fs_hi", hi)
ceil = hi / np.sqrt(float(NUM["rel40_fs"]) * float(NUM["rel_fs_game"]))
put("tr_fs_ceiling", ceil)

prd = DC[(DC.drill_name == "PASS_RUSH_DRILL")].groupby("nfl_id").fs.mean().rename("prd_fs").reset_index()
tp = dl.merge(prd, on="nfl_id").dropna(subset=["fs", "w_z"])
tp["prd_fs"] = yz(tp.prd_fs, tp.draft_year)
r, lo, hi, n, _ = partial(tp, "prd_fs", "fs", ZS)
put("tr_prd", r); put("tr_prd_lo", lo); put("tr_prd_hi", hi); put("tr_prd_n", n)

# max-over-time permutation test for the payoff curve (the first-step window was read off this curve)
Rk = pay_set[["peb"] + ZB].rank().to_numpy(float)
Av = np.c_[np.ones(len(Rk)), Rk[:, 1:]]
Vt = np.column_stack([_res(pay_set[f"v{t}"].rank().to_numpy(float), Av) for t in range(3, TW)])
Vt = (Vt - Vt.mean(0)) / Vt.std(0)


def maxr(yv):
    ry = _res(yv, Av)
    ry = (ry - ry.mean()) / ry.std()
    return np.max(Vt.T @ ry / len(ry))


obs_max = maxr(Rk[:, 0])
yrs = pay_set.draft_year.to_numpy()
nulls = []
for _ in range(2000):
    yy = Rk[:, 0].copy()
    for c_ in np.unique(yrs):
        ii = np.flatnonzero(yrs == c_)
        yy[ii] = yy[RNG.permutation(ii)]
    nulls.append(maxr(yy))
put("p_maxt", float(np.mean(np.array(nulls) >= obs_max)))
put("n_windows", TW - 3)

# ============================================================================ per-player translation (Carry-over)
tick("carry-over")


def adj_z(df, col, Z):
    A = np.c_[np.ones(len(df)), df[Z].to_numpy(float)]
    rr = _res(df[col].to_numpy(float), A)
    return (rr - rr.mean()) / rr.std()


tr_set["cfs_adj"] = adj_z(tr_set, "c_fs", ZS)
tr_set["sfs_adj"] = adj_z(tr_set, "fs", ZS)
tr_set["carry"] = tr_set.sfs_adj - tr_set.cfs_adj
M40 = tr_set[[f"c_v{t}" for t in range(TW)]].to_numpy()
MSN = tr_set[[f"v{t}" for t in range(TW)]].to_numpy()
ZD = (M40 - M40.mean(0)) / M40.std(0)
ZG = (MSN - MSN.mean(0)) / MSN.std(0)
tr_set["shape"] = [np.corrcoef(ZD[i, 3:], ZG[i, 3:])[0, 1] for i in range(len(tr_set))]
tr_set["dist"] = -np.abs(ZD[:, 3:] - ZG[:, 3:]).mean(1)
tr_set["ratio"] = tr_set.fs / tr_set.c_fs.rank()   # placeholder replaced below
c40raw = d40.groupby("nfl_id").fs.mean()
tr_set["ratio"] = tr_set.fs / tr_set.nfl_id.map(c40raw)
c40dr = d40.groupby("nfl_id").dr.mean()
tr_set["ratio_dr"] = tr_set.dr / tr_set.nfl_id.map(c40dr)
put("ratio_fs_med", 100 * tr_set.ratio.median(), 0)
put("ratio_dr_med", 100 * tr_set.ratio_dr.median(), 0)
bt = both.merge(tr_set[["nfl_id", "cfs_adj", "sfs_adj", "carry", "shape", "dist", "ratio", "ratio_dr"]], on="nfl_id")
SIM = [("carry", "Carry-over (Sunday minus Combine first step, sd)"), ("ratio", "Speed ratio, first step"),
       ("ratio_dr", "Speed ratio, drive"), ("shape", "Curve-shape correlation"), ("dist", "Curve distance (negative)")]
simres = {}
for col, lab in SIM:
    simres[col] = partial(bt, col, "peb", ZB)
for col in ["carry", "shape", "ratio"]:
    put(f"sim_{col}", simres[col][0])
put("sim_carry_lo", simres["carry"][1]); put("sim_carry_hi", simres["carry"][2])
put("sim_absmax", max(abs(v[0]) for v in simres.values()))
r, lo, hi, n, ppos = partial(bt, "sfs_adj", "peb", ZB)
put("pay_sfs", r); put("pay_sfs_lo", lo); put("pay_sfs_hi", hi); put("pay_sfs_pos", 100 * ppos, 0)
r, lo, hi, n, _ = partial(bt, "cfs_adj", "peb", ZB)
put("pay_cfs", r); put("pay_cfs_lo", lo); put("pay_cfs_hi", hi)
# all rushers with a Sunday first step (no 40 needed)
pay_set["sfs_adj"] = adj_z(pay_set, "fs", ZS)
r, lo, hi, n, ppos = partial(pay_set, "sfs_adj", "peb", ZB)
PAY_ALL = (r, lo, hi, n)
put("pay_all", r); put("pay_all_lo", lo); put("pay_all_hi", hi)
pay_set["ngo"] = -pay_set.getoff
r, lo, hi, n, _ = partial(pay_set.dropna(subset=["ngo"]), "ngo", "peb", ZB)
put("pay_ngs", r); put("pay_ngs_lo", lo); put("pay_ngs_hi", hi)
r, *_ = partial(pay_set.dropna(subset=["ngo"]), "sfs_adj", "peb", ZB + ["ngo"], nb=10)
put("pay_beyond_ngs", r)

# ---- planning number: pressure-rate points per sd of Sunday first step, rush-weighted least squares
X = np.c_[np.ones(len(pay_set)), pay_set[ZB + ["sfs_adj"]].to_numpy(float)]
W = pay_set.rs.to_numpy(float)
y = 100 * pay_set.peb.to_numpy()


def wls(Xm, yv, w):
    sw = np.sqrt(w)
    return np.linalg.lstsq(Xm * sw[:, None], yv * sw, rcond=None)[0]


b = wls(X, y, W)
bb = []
for _ in range(NBOOT):
    i = RNG.integers(0, len(y), len(y))
    bb.append(wls(X[i], y[i], W[i])[-1])
put("pp_per_sd", b[-1], 1); put("pp_per_sd_lo", np.percentile(bb, 5), 1); put("pp_per_sd_hi", np.percentile(bb, 95), 1)
RUSH_SEASON = float(np.median(seasons[(seasons.position_group == "DL") & (seasons.pass_rush_snaps >= 100)]
                              .pass_rush_snaps))
put("rush_season", RUSH_SEASON, 0)
put("pr_per_season", b[-1] / 100 * RUSH_SEASON, 1)
put("pr_per_season_lo", np.percentile(bb, 5) / 100 * RUSH_SEASON, 1)
put("pr_per_season_hi", np.percentile(bb, 95) / 100 * RUSH_SEASON, 1)

# ============================================================================ out of class
tick("out of class")
CLASSES = [2023, 2024, 2025]


def loco(df, add):
    rows = []
    for yr in CLASSES:
        tr, te = df[df.draft_year != yr], df[df.draft_year == yr]

        def pred(cols):
            bb_ = wls(np.c_[np.ones(len(tr)), tr[cols].to_numpy(float)], tr.peb.to_numpy(), tr.rs.to_numpy(float))
            return np.c_[np.ones(len(te)), te[cols].to_numpy(float)] @ bb_
        ybar = np.average(tr.peb, weights=tr.rs)
        rows.append(pd.DataFrame({"yr": yr, "y": te.peb.to_numpy(), "w": te.rs.to_numpy(float),
                                  "p0": pred(ZB), "p1": pred(ZB + add) if add else pred(ZB), "ybar": ybar,
                                  "nfl_id": te.nfl_id.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def dr2(L):
    sst = np.sum(L.w * (L.y - L.ybar) ** 2)
    return (np.sum(L.w * (L.y - L.p0) ** 2) - np.sum(L.w * (L.y - L.p1) ** 2)) / sst


def dr2_ci(L, nb=NBOOT):
    out = []
    for _ in range(nb):
        parts = [g.iloc[RNG.integers(0, len(g), len(g))] for _, g in L.groupby("yr")]
        out.append(dr2(pd.concat(parts)))
    return np.percentile(out, [5, 95])


bt2 = bt.copy()
# the adjusted z-scores are refit inside each fold to avoid leakage of the scaling: do it explicitly
OOC = {}
for nm, col, base in [("sunday", "sfs_adj", bt2), ("combine", "cfs_adj", bt2), ("carry", "carry", bt2),
                      ("sunday_all", "sfs_adj", pay_set)]:
    L = loco(base, [col])
    lo, hi = dr2_ci(L)
    per = {yr: dr2(L[L.yr == yr]) for yr in CLASSES}
    nper = {yr: int((L.yr == yr).sum()) for yr in CLASSES}
    OOC[nm] = dict(d=dr2(L), lo=lo, hi=hi, per=per, nper=nper, L=L)
    put(f"ooc_{nm}", OOC[nm]["d"], 3); put(f"ooc_{nm}_lo", lo, 3); put(f"ooc_{nm}_hi", hi, 3)
    for yr in CLASSES:
        put(f"ooc_{nm}_{yr}", per[yr], 3)
base_r2 = 1 - np.sum(OOC["sunday"]["L"].w * (OOC["sunday"]["L"].y - OOC["sunday"]["L"].p0) ** 2) / \
    np.sum(OOC["sunday"]["L"].w * (OOC["sunday"]["L"].y - OOC["sunday"]["L"].ybar) ** 2)
put("ooc_base_r2", base_r2, 3)
for yr in CLASSES:
    put(f"n_ooc_{yr}", OOC["sunday_all"]["nper"][yr])

# ============================================================================ calibration and quadrants
tick("calibration")
Lb = loco(pay_set, [])
fc = Lb.set_index("nfl_id").p0
pay_set["forecast"] = pay_set.nfl_id.map(fc)
pay_set["beat"] = (pay_set.prate > pay_set.forecast).astype(int)
bt = bt.merge(pay_set[["nfl_id", "forecast", "beat"]], on="nfl_id")
put("beat_base", 100 * pay_set.beat.mean(), 0)
q_hi, q_lo = pay_set.sfs_adj.quantile(2 / 3), pay_set.sfs_adj.quantile(1 / 3)
top, bot = pay_set[pay_set.sfs_adj >= q_hi], pay_set[pay_set.sfs_adj <= q_lo]
for nm, g_ in [("top", top), ("bot", bot)]:
    k, n = int(g_.beat.sum()), len(g_)
    lo, hi = cp(k, n)
    put(f"cal_{nm}_k", k); put(f"cal_{nm}_n", n)
    put(f"cal_{nm}_pct", 100 * k / n, 0); put(f"cal_{nm}_lo", 100 * lo, 0); put(f"cal_{nm}_hi", 100 * hi, 0)
    put(f"cal_{nm}_rate", 100 * g_.pr.sum() / g_.rs.sum(), 1)
    put(f"cal_{nm}_fc", 100 * np.average(g_.forecast, weights=g_.rs), 1)
# 2025-only line (fit on 2023-24 only, never tuned on 2025 outcomes)
t25 = pay_set[pay_set.draft_year == 2025]
q25 = t25[t25.sfs_adj >= q_hi]
k, n = int(q25.beat.sum()), len(q25)
lo, hi = cp(k, n)
put("cal25_k", k); put("cal25_n", n); put("cal25_lo", 100 * lo, 0); put("cal25_hi", 100 * hi, 0)

# quadrants on the rushers with both a 40 and a Sunday first step
bt["cq"] = np.where(bt.cfs_adj >= 0, "fast", "slow")
bt["sq"] = np.where(bt.sfs_adj >= 0, "fast", "slow")
QROWS = []
for cq, sq, lab in [("fast", "fast", "Fast in both"), ("slow", "fast", "Sunday-only fast"),
                    ("fast", "slow", "Combine-only fast"), ("slow", "slow", "Slow in both")]:
    g_ = bt[(bt.cq == cq) & (bt.sq == sq)]
    k, n = int(g_.beat.sum()), len(g_)
    lo, hi = cp(k, n)
    row = dict(label=lab, n=n, edge=int((g_.role == "EDGE").sum()), rate=100 * g_.pr.sum() / g_.rs.sum(),
               fc=100 * np.average(g_.forecast, weights=g_.rs), beat=100 * k / n, lo=100 * lo, hi=100 * hi)
    QROWS.append(row)
    tag = {"Fast in both": "ff", "Sunday-only fast": "sf", "Combine-only fast": "cf", "Slow in both": "ss"}[lab]
    put(f"q_{tag}_n", n); put(f"q_{tag}_rate", row["rate"], 1); put(f"q_{tag}_fc", row["fc"], 1)
    put(f"q_{tag}_beat", row["beat"], 0); put(f"q_{tag}_lo", row["lo"], 0); put(f"q_{tag}_hi", row["hi"], 0)
put("q_agree", 100 * (bt.cq == bt.sq).mean(), 0)
for tag, g_ in [("up", bt[bt.sq == "fast"]), ("dn", bt[bt.sq == "slow"]), ("cr", bt[bt.cq == "fast"]),
                ("cl", bt[bt.cq == "slow"])]:
    put(f"half_{tag}_beat", 100 * g_.beat.mean(), 0); put(f"half_{tag}_n", len(g_))

# EDGE vs interior
for role in ["EDGE", "interior"]:
    tag = "e" if role == "EDGE" else "i"
    tt = tr_set[tr_set.role == role]
    r, lo, hi, n, _ = partial(tt, "c_fs", "fs", ["w_z"])
    put(f"role_{tag}_tr", r); put(f"role_{tag}_tr_n", n)
    pp = pay_set[pay_set.role == role]
    r, lo, hi, n, _ = partial(pp, "sfs_adj", "peb", ["log_pick", "w_z"])
    put(f"role_{tag}_pay", r); put(f"role_{tag}_pay_lo", lo); put(f"role_{tag}_pay_hi", hi); put(f"role_{tag}_n", n)
    th, tl = pp[pp.sfs_adj >= pp.sfs_adj.quantile(2 / 3)], pp[pp.sfs_adj <= pp.sfs_adj.quantile(1 / 3)]
    put(f"role_{tag}_top", 100 * th.pr.sum() / th.rs.sum(), 1); put(f"role_{tag}_bot", 100 * tl.pr.sum() / tl.rs.sum(), 1)

# ============================================================================ group screen (Table 1)
tick("group screen")
PROD = {"WR": ("Yards per route", 1), "OL": ("Fewer pressures allowed", -1),
        "DB": ("Solo tackles per 100 snaps", 1), "TE": ("Snaps per season", 1)}
GROWS = []
for grp in ["DL", "WR", "OL", "DB", "TE"]:
    gg = GC[(GC.position_group == grp)]
    gpp = gg.groupby("nfl_id").agg(n_g=("fs", "size"), fs=("fs_a", "mean"), fs_u=("fs", "mean")).reset_index()
    gpp = gpp[gpp.n_g >= MIN_SNAPS_G]
    c4 = DC[(DC.drill_name == "FORTY_YARD_DASH") & (DC.position_group == grp)].groupby("nfl_id").fs.mean() \
        .rename("c_fs").reset_index()
    P = pl[pl.position_group == grp].merge(gpp, on="nfl_id").merge(c4, on="nfl_id", how="left")
    P["c_fs"] = yz(P.c_fs, P.draft_year)
    if grp == "DL":
        P = P.merge(dl[["nfl_id", "edge", "peb", "rs"]], on="nfl_id")
        Zs, Zb, ycol = ["w_z", "edge"], ["log_pick", "w_z", "edge"], "peb"
        Pp = P[P.rs >= MIN_RUSH]
        lab = "Pressure rate"
    else:
        tgt = "opportunity" if grp == "TE" else "production"
        v = pv[(pv.position_group == grp) & (pv.target == tgt)][["nfl_id", "actual"]]
        P = P.merge(v, on="nfl_id", how="left")
        P["y"] = PROD[grp][1] * P.actual
        Zs, Zb, ycol = ["w_z"], ["log_pick", "w_z"], "y"
        Pp = P
        lab = PROD[grp][0]
    tr_raw = partial(P, "c_fs", "fs_u", [])
    tr_adj = partial(P, "c_fs", "fs", Zs)
    pay = PAY_ALL if grp == "DL" else partial(Pp, "fs", ycol, Zb)
    q = gg[gg.nfl_id.isin(gpp.nfl_id)].groupby("nfl_id").fs_a.agg(["size", "mean", "var"])
    s2 = np.average(q["var"], weights=q["size"] - 1)
    t2 = q["mean"].var() - s2 * (1 / q["size"]).mean()
    rel = t2 / (t2 + s2 / q["size"].median())
    GROWS.append(dict(grp=grp, n_tr=tr_adj[3], tr_raw=tr_raw[0], tr_adj=tr_adj[0], tr_lo=tr_adj[1], tr_hi=tr_adj[2],
                      rel=rel, n_pay=pay[3], pay=pay[0], pay_lo=pay[1], pay_hi=pay[2], lab=lab))
GT = pd.DataFrame(GROWS)
TABLES["groups"] = GT.round(3).to_dict("records")
md = ["| Group | Players (40 + 8 NFL starts) | Sunday first step, reliability | Combine → Sunday, raw | Combine → Sunday, after size | Sunday first step → production beyond pick [90%] (n) |",
      "|---|---|---|---|---|---|"]
for r_ in GROWS:
    g = r_["grp"]
    put(f"g_{g}_rel", r_["rel"]); put(f"g_{g}_trraw", r_["tr_raw"]); put(f"g_{g}_tradj", r_["tr_adj"])
    put(f"g_{g}_pay", r_["pay"]); put(f"g_{g}_paylo", r_["pay_lo"]); put(f"g_{g}_payhi", r_["pay_hi"])
    md.append(f"| {'**DL**' if g == 'DL' else g} | {r_['n_tr']} | {{{{g_{g}_rel}}}} | {{{{g_{g}_trraw}}}} | "
              f"{{{{g_{g}_tradj}}}} | {'**' if g == 'DL' else ''}{{{{g_{g}_pay}}}}{'**' if g == 'DL' else ''} "
              f"[{{{{g_{g}_paylo}}}}, {{{{g_{g}_payhi}}}}] ({r_['n_pay']}, {r_['lab'].lower()}) |")
TABLES["md"]["groups"] = "\n".join(md)

# ============================================================================ decision card
tick("board")
dl_ = dl.copy()
board = pay_set[["nfl_id", "display_name", "draft_year", "pick", "role", "rs", "pr", "prate", "forecast", "sfs_adj",
                 "fs", "n_g"]].merge(tr_set[["nfl_id", "cfs_adj", "carry"]], on="nfl_id", how="left")
board["s_pct"] = 100 * board.sfs_adj.rank(pct=True)
board["c_pct"] = 100 * board.cfs_adj.rank(pct=True)


def call(r_):
    if r_.s_pct >= 200 / 3:
        if pd.notna(r_.c_pct) and r_.c_pct < 50:
            return "Trust the tape over the 40"
        return "Credit the start as a tie-break"
    if r_.s_pct <= 100 / 3:
        if pd.notna(r_.c_pct) and r_.c_pct >= 50:
            return "Film check: his 40 start is not on tape"
        return "No start edge: grade hands and power"
    return "No call"


board["call"] = board.apply(call, axis=1)
board = board.sort_values("sfs_adj", ascending=False)
board.to_csv(OUT / "board.csv", index=False)
put("n_board", len(board))


def pick_txt(p):
    return "UDFA" if p >= UDFA_PICK else str(int(p))


# pick 7 named: 2025 class top/bottom first steps + 2023-24 clearest quadrant examples
b25 = board[board.draft_year == 2025]
sel = list(b25.head(2).nfl_id) + list(b25.tail(1).nfl_id)
old = board[board.draft_year < 2025]
sf_ = old[(old.c_pct < 50) & (old.s_pct >= 200 / 3)].head(1)
cf_ = old[(old.c_pct >= 50) & (old.s_pct <= 100 / 3)].sort_values("carry").head(2)
ff_ = old[(old.c_pct >= 75) & (old.s_pct >= 75)].head(1)
sel += list(sf_.nfl_id) + list(cf_.nfl_id) + list(ff_.nfl_id)
sel += list(old.head(1).nfl_id)
sel = list(dict.fromkeys(sel))
card = board[board.nfl_id.isin(sel)].copy()
card["cls"] = np.where(card.draft_year == 2025, 0, 1)
card = card.sort_values(["cls", "sfs_adj"], ascending=[True, False])
md = ["| Rusher (class, pick, role) | Combine first step, pct | Sunday first step, pct (starts) | Pressure rate vs forecast | Call |",
      "|---|---|---|---|---|"]
NAMED = {}
for _, r_ in card.iterrows():
    key = re.sub(r"[^a-z]", "", r_.display_name.split()[-1].lower())
    NAMED[key] = r_
    put(f"b_{key}_c", r_.c_pct, 0) if pd.notna(r_.c_pct) else None
    put(f"b_{key}_s", r_.s_pct, 0); put(f"b_{key}_rate", 100 * r_.prate, 1); put(f"b_{key}_fc", 100 * r_.forecast, 1)
    ctxt = f"{{{{b_{key}_c}}}}" if pd.notna(r_.c_pct) else "no 40"
    md.append(f"| {r_.display_name} ({r_.draft_year}, {pick_txt(r_.pick)}, {r_.role}) | {ctxt} | "
              f"{{{{b_{key}_s}}}} ({int(r_.n_g)}) | {{{{b_{key}_rate}}}}% vs {{{{b_{key}_fc}}}}% | {r_.call} |")
TABLES["md"]["card"] = "\n".join(md)
TABLES["card_names"] = list(card.display_name)
print(card[["display_name", "draft_year", "pick", "role", "c_pct", "s_pct", "prate", "forecast", "call"]])

# ============================================================================ figures
tick("figures")
# ---- Fig 1: hero, four rushers, Combine 40 start vs Sunday starts on one time axis, plus pressure bar
fig_names = [n for n in ["fast_both", "sunday_only", "combine_only", "slow"]]
cands = {"fast_both": board[(board.c_pct >= 60) & (board.s_pct >= 80)],
         "sunday_only": board[(board.c_pct < 40) & (board.s_pct >= 80)],
         "combine_only": board[(board.c_pct >= 60) & (board.s_pct <= 20) & (board.rs >= 200)]
         .sort_values("carry").head(1).assign(rs=1e9),
         "slow": board[(board.c_pct < 40) & (board.s_pct <= 20)]}
hero = []
for k_ in fig_names:
    c_ = cands[k_].dropna(subset=["c_pct"])
    c_ = c_.sort_values("rs", ascending=False)
    if len(c_):
        hero.append((k_, c_.iloc[0]))
TT = np.arange(TW) / 10
fig = plt.figure(figsize=(FIGW, 2.55 * len(hero) + 0.9))
gs = fig.add_gridspec(len(hero), 2, width_ratios=[3.2, 1.25], hspace=0.55, wspace=0.28,
                      left=0.075, right=0.97, top=0.9, bottom=0.08)
TITLES = {"fast_both": "Fast in both", "sunday_only": "Sunday-only fast", "combine_only": "Combine-only fast",
          "slow": "Slow in both"}
gmed = np.median(np.vstack(gd.curve.to_numpy()), axis=0)
cmed = np.median(np.vstack(d40.curve.to_numpy()), axis=0)
for row, (k_, r_) in enumerate(hero):
    ax = fig.add_subplot(gs[row, 0])
    ax.axvspan(0.3, 0.5, color=OI["yellow"], alpha=0.35, lw=0)
    snaps = gd[gd.nfl_id == r_.nfl_id]
    for cv in snaps.curve:
        ax.plot(TT, cv, color=OI["blue"], alpha=0.18, lw=0.9)
    ax.plot(TT, np.mean(np.vstack(snaps.curve.to_numpy()), 0), color=OI["blue"], lw=2.6, label="his NFL starts (mean)")
    reps = d40[d40.nfl_id == r_.nfl_id]
    ax.plot(TT, np.mean(np.vstack(reps.curve.to_numpy()), 0), color=OI["orange"], lw=2.6, ls="--",
            label="his Combine 40 start")
    ax.plot(TT, gmed, color=OI["grey"], lw=1.2, ls=":", label="median DL, NFL")
    ax.set_xlim(0, 1.5); ax.set_ylim(0, 8.5)
    ax.set_ylabel("speed (yd/s)")
    ax.set_title(f"{TITLES[k_]}: {r_.display_name} ({r_.draft_year}, {r_.role}) — Combine first step "
                 f"{ordn(r_.c_pct)} pct, NFL {ordn(r_.s_pct)} pct", loc="left", fontsize=11.5)
    if row == len(hero) - 1:
        ax.set_xlabel("seconds from first movement")
    if row == 0:
        ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0, 1.42))
        ax.text(0.4, 8.0, "first step", ha="center", fontsize=10, color="#6b5d00")
    bx = fig.add_subplot(gs[row, 1])
    bx.barh([1, 0], [100 * r_.prate, 100 * r_.forecast], color=[OI["blue"], OI["grey"]], height=0.6)
    for yy, vv in zip([1, 0], [100 * r_.prate, 100 * r_.forecast]):
        bx.text(vv + 0.3, yy, f"{vv:.1f}%", va="center", fontsize=10.5)
    bx.set_yticks([1, 0]); bx.set_yticklabels(["pressure rate", "forecast"])
    bx.set_xlim(0, 16)
    if row == len(hero) - 1:
        bx.set_xlabel("% of pass rushes")
fig.suptitle("Combine 40 start (dashed) vs the same player's NFL pass-rush starts (blue), same time axis",
             x=0.075, ha="left", fontsize=13.5, y=0.985)
save(fig, "fig1_four_rushers.png")
HERO = {k_: r_ for k_, r_ in hero}
for k_, r_ in hero:
    put(f"h_{k_}_rate", 100 * r_.prate, 1); put(f"h_{k_}_fc", 100 * r_.forecast, 1)
    put(f"h_{k_}_c", r_.c_pct, 0); put(f"h_{k_}_s", r_.s_pct, 0)
TABLES["hero"] = {k_: r_.display_name for k_, r_ in hero}

# ---- Fig 2: translation and payoff, tenth by tenth
fig, ax = plt.subplots(figsize=(FIGW, 5.6))
fig.subplots_adjust(left=0.08, right=0.78, top=0.88, bottom=0.12)
ts = TT[3:]


def band(series, color, label, ls="-", lw=2.4, fill=True):
    r_ = np.array([c[0] for c in CUR[series][3:]])
    lo = np.array([c[1] for c in CUR[series][3:]])
    hi = np.array([c[2] for c in CUR[series][3:]])
    if fill:
        ax.fill_between(ts, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(ts, r_, color=color, ls=ls, lw=lw)
    ax.text(1.52, r_[-1], label, color=color, va="center", fontsize=11)


ax.axhline(0, color="black", lw=0.8)
ax.axvspan(0.3, 0.5, color=OI["yellow"], alpha=0.3, lw=0)
band("raw", OI["grey"], "Combine 40 → NFL speed,\nraw (mostly size)", ls="--", lw=1.8, fill=False)
band("adj", OI["orange"], "Combine 40 → NFL speed,\nafter weight and role")
band("pay", OI["blue"], "NFL speed → pressure rate,\nafter pick, weight, role")
ax.set_xlim(0.3, 1.5); ax.set_ylim(-0.45, 0.8)
ax.set_xlabel("seconds from first movement")
ax.set_ylabel("rank correlation (90% band)")
ax.text(0.4, 0.74, "first step", ha="center", fontsize=10.5, color="#6b5d00")
ax.set_title(f"DL: {NUM['n_tr']} rushers for the Combine lines, {NUM['n_pay']} for the pressure line",
             loc="left", fontsize=12)
fig.suptitle("Raw, the 40 previews NFL speed; after weight and role it previews nothing",
             x=0.08, ha="left", fontsize=13.5)
save(fig, "fig2_translation_payoff.png")

# ---- Fig 3: carry-over map
fig, ax = plt.subplots(figsize=(FIGW, 7.2))
fig.subplots_adjust(left=0.09, right=0.97, top=0.9, bottom=0.1)
bt_ = bt.copy()
bt_["res"] = 100 * (bt_.prate - bt_.forecast)
vmax = 6
sc = ax.scatter(bt_.cfs_adj, bt_.sfs_adj, c=bt_.res.clip(-vmax, vmax), cmap="PuOr", vmin=-vmax, vmax=vmax,
                s=34 + 0.12 * bt_.rs.clip(0, 900), edgecolor="black", lw=0.6,
                marker="o")
ax.axhline(0, color=OI["grey"], lw=0.9); ax.axvline(0, color=OI["grey"], lw=0.9)
lim = 2.9
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim + 0.6)
ax.plot([-lim, lim], [-lim, lim], color=OI["grey"], lw=0.8, ls=":")
ax.text(lim - 0.1, -lim + 0.15, "dotted line: perfect carry-over", ha="right", fontsize=10, color=OI["grey"])
for (cq, sq, tag), (xx, yy, ha) in zip([("fast", "fast", "ff"), ("slow", "fast", "sf"), ("fast", "slow", "cf"),
                                        ("slow", "slow", "ss")],
                                       [(lim - 0.1, lim + 0.5, "right"), (-lim + 0.1, lim + 0.5, "left"),
                                        (lim - 0.1, -lim + 0.35, "right"), (-lim + 0.1, -lim + 0.35, "left")]):
    ax.text(xx, yy, f"n = {NUM[f'q_{tag}_n']}: pressure {NUM[f'q_{tag}_rate']}% vs {NUM[f'q_{tag}_fc']}% forecast\n"
            f"beat forecast: {NUM[f'q_{tag}_beat']}%",
            ha=ha, va="top" if yy > 0 else "bottom", fontsize=10.5, fontweight="bold")
for key, r_ in NAMED.items():
    if r_.nfl_id in set(bt_.nfl_id):
        rr = bt_[bt_.nfl_id == r_.nfl_id].iloc[0]
        right = rr.cfs_adj > 1.8
        ax.annotate(r_.display_name, (rr.cfs_adj, rr.sfs_adj), xytext=(-6 if right else 6, 5),
                    textcoords="offset points", fontsize=10, ha="right" if right else "left")
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
cb.set_label("pressure rate minus forecast (points)")
ax.set_xlabel("Combine 40 first step (sd, after weight and role)")
ax.set_ylabel("NFL first step (sd, after weight and role)")
ax.set_title(f"{NUM['n_both']} rushers with a 40, 8+ random NFL starts and 100+ rushes; dot size = rushes",
             loc="left", fontsize=12)
fig.suptitle("Carry-over map: rushers above the line beat their forecast more often; left or right does not matter",
             x=0.09, ha="left",
             fontsize=13.5)
save(fig, "fig3_carryover_map.png")

# ---- Fig 4: out of class
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.3), gridspec_kw={"width_ratios": [1.0, 1.25]})
fig.subplots_adjust(left=0.2, right=0.97, top=0.84, bottom=0.24, wspace=0.55)
ax = axs[0]
rows_ = [("sunday_all", f"NFL first step\n(all {NUM['n_pay']})", OI["blue"]),
         ("sunday", f"NFL first step\n({NUM['n_both']} with a 40)", OI["sky"]),
         ("combine", "Combine first step", OI["orange"]), ("carry", "Carry-over", OI["purple"])]
for i, (nm, lab, col) in enumerate(rows_):
    o = OOC[nm]
    ax.plot([o["lo"], o["hi"]], [i, i], color=col, lw=3)
    ax.plot(o["d"], i, "o", color=col, ms=9, mec="black")
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(rows_))); ax.set_yticklabels([r[1] for r in rows_])
ax.invert_yaxis()
ax.set_xlabel("R² change vs no-tracking baseline")
ax.set_title("Pooled, 90% interval", loc="left", fontsize=12)
ax = axs[1]
for j, yr in enumerate(CLASSES):
    for i, (nm, lab, col) in enumerate(rows_):
        o = OOC[nm]
        ax.plot(o["per"][yr], j + (i - 1.5) * 0.17, "o", color=col, ms=8, mec="black")
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(3))
ax.set_yticklabels([f"held-out {yr}\n(n = {OOC['sunday_all']['nper'][yr]} / {OOC['sunday']['nper'][yr]})"
                    for yr in CLASSES])
ax.invert_yaxis()
ax.set_xlabel("R² change in the held-out class")
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c_, mec="black", ms=8, label=l_.replace("\n", " "))
                    for _, l_, c_ in rows_], loc="lower center", ncol=4, fontsize=10.5, frameon=False)
ax.set_title("By held-out class (all / with a 40)", loc="left", fontsize=12)
fig.suptitle("Fit on two classes, score the third: no translation measure beats draft slot, weight and role",
             x=0.02, ha="left", fontsize=13.5)
save(fig, "fig4_out_of_class.png")

# ============================================================================ write
tick("write")
json.dump(TABLES, open(OUT / "tables.json", "w"), indent=1, default=str)
used = render(OUT, NUM, TABLES["md"])
if used:
    json.dump({k: NUM[k] for k in sorted(used)}, open(OUT / "numbers.json", "w"), indent=1)
else:
    json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1)
json.dump(NUM, open(OUT / "numbers_all.json", "w"), indent=1)
tick("done")
