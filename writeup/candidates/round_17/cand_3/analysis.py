"""Movement translation for pass rushers: does a rusher's Combine start show up in his NFL start, and does the
translation explain pressure beyond draft slot?

Run from the repo root:
    uv run python writeup/candidates/round_17/cand_3/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (only the values quoted in writeup.md), claims.json
(key -> sentence that uses it), tables.json, board.csv and figures/*.png next to this file, then renders writeup.md
from writeup_template.md. Frame-level curve extraction is cached in data/cache/r12c3 (gitignored).
Runtime: about 2 minutes, one core.

Round 16 changes: every pick-matched pair instead of one hand-picked case (each Both rusher vs the nearest-pick
non-Both rusher of the same class and role, career and frozen-2024 -> 2025; Fig 5, bootstrap and sign test); 2026
list with role, availability flag, the 2024 call's 2025 result by name and a pre-specified act-on rule; pre-registered
2026 test (rule, list, targets, power) hashed into frozen_rule.json; Fig 1 league IQR band; Fig 3 legend outside.

Round 15 changes: exclusions and a selection model (P(100+ rushes) from pick, weight, 40, class) with
inverse-probability-weighted reruns; rusher random-intercept binomial model on player-seasons (Gauss-Hermite);
shortlist calibration and Brier score; measurement protocol (reps, sampled starts needed); worked decision case
(Cedric Johnson vs Javontae Jean-Baptiste, frozen after 2024, OTC rookie and veteran edge APY); 2026 list with
alignment and games missed; Fig 2C five-group screen with low-reliability groups greyed; Fig 3 title/p labels.

Round 14 changes: one pool stated once; time-forward test (call frozen from 2023-24 starts, scored on 2025 against a
persistence forecast that knows each rusher's own past pressure) with a no-tracking persistence shortlist; one 5-row
robustness table with rusher-resampled gap intervals; disattenuated translation and a flat-prior posterior bound;
2026 shortlist table (shortlist_2026.csv); Fig 1 rebuilt as a 2x2 of four rushers; Fig 3 labels and panel B
reduced to the table rows. Runtime about 3 minutes, one core.

Round 13 changes: the positive use leads. The two-signal call (official 40 time x NFL first step) now takes its
cut-offs from the other two classes only, and the "Both" shortlist is scored against the base rate of every other
rusher (within-class label shuffle, player bootstrap of the gap, drop-one), under a shrunken forecast, a forecast that
already knows the 40 time and NGS get-off, an outcome with every sampled snap removed, quick pressures and sacks.
Fig 3 is the shortlist map + hit-rate panel; the old Fig 5 is cut (its numbers stay in the text).

Round 12 changes: a second, stronger no-tracking baseline (official 40 time + season NGS get-off) for every measure;
the official 40 time run through the same held-out test; a draft-price model (log pick on 40 time and tracked start);
a ridge forecast with leave-one-class-out penalty and held-out calibration slopes per class; a player-cluster
bootstrap and leave-one-player-out rerun of the whole first-step test (forecast refit); a two-signal decision card
(40 time x NFL first step) with per-player actions; Fig 2B shuffle-null bands; new Fig 5 (robustness).

Round 11 changes: one continuous held-out test (leave-one-class-out quasi-binomial slope with the no-tracking forecast
as offset, random-effects pooling, within-class shuffle null and a best-of-windows adjustment) replaces the hit counts;
a within-rusher snap-level mechanism test (pressure on the same sampled snap); the rule frozen in frozen_rule.json with
a sha256 prefix; a decision card with per-player actions, sideways (scheme) percentile and pressures at stake.

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
CACHE = ROOT / "data" / "cache" / "r16c3"


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
plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 14, "axes.spines.top": False,
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


def perm_p(df, x, y, Z, nperm=2000, by="draft_year"):
    """Two-sided permutation p for the partial Spearman: x shuffled within Combine class."""
    d = df[[x, y] + Z + [by]].dropna()
    R = d[[x, y] + Z].rank().to_numpy(float)
    A = np.c_[np.ones(len(R)), R[:, 2:]]
    rx, ry = _res(R[:, 0], A), _res(R[:, 1], A)
    obs = abs(np.corrcoef(rx, ry)[0, 1])
    grp = d[by].to_numpy()
    cnt = 0
    for _ in range(nperm):
        xx = R[:, 0].copy()
        for c_ in np.unique(grp):
            ii = np.flatnonzero(grp == c_)
            xx[ii] = xx[RNG.permutation(ii)]
        cnt += abs(np.corrcoef(_res(xx, A), ry)[0, 1]) >= obs
    return (cnt + 1) / (nperm + 1)


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
put("snaps_med", int(gplayer.n_g.median())); put("n_gplayer", len(gplayer))
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
put("p_tr_fs", perm_p(tr_set, "c_fs", "fs", ZS), 2)
r, lo, hi, n, _ = partial(tr_set, "forty_z", "fs", ZS)
put("tr_offfs", r); put("tr_offfs_lo", lo); put("tr_offfs_hi", hi)
put("p_tr_offfs", perm_p(tr_set, "forty_z", "fs", ZS), 2)

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
put("p_carry", perm_p(bt, "carry", "peb", ZB), 2); put("n_carry", simres["carry"][3])
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
put("p_pay_all", perm_p(pay_set, "sfs_adj", "peb", ZB), 2)
pay_set["ngo"] = -pay_set.getoff
r, lo, hi, n, _ = partial(pay_set.dropna(subset=["ngo"]), "ngo", "peb", ZB)
put("pay_ngs", r); put("pay_ngs_lo", lo); put("pay_ngs_hi", hi)
r, *_ = partial(pay_set.dropna(subset=["ngo"]), "sfs_adj", "peb", ZB + ["ngo"], nb=10)
put("pay_beyond_ngs", r)
pn = pay_set.dropna(subset=["ngo"])
put("p_beyond_ngs", perm_p(pn, "sfs_adj", "peb", ZB + ["ngo"]), 2); put("n_ngs_pay", len(pn))

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
put("beat_base", 100 * pay_set.beat.mean(), 0); put("beat_k", int(pay_set.beat.sum()))
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
# Fisher exact test, top vs bottom third beating the forecast
kt, nt, kb, nb_ = int(top.beat.sum()), len(top), int(bot.beat.sum()), len(bot)
put("p_fisher", stats.fisher_exact([[kt, nt - kt], [kb, nb_ - kb]])[1], 3)
# posterior predictive for the clean 2025 check: top-third rate learned on 2023-24 only
t2324 = top[top.draft_year < 2025]
k0, n0 = int(t2324.beat.sum()), len(t2324)
put("pp_k0", k0); put("pp_n0", n0)
put("pp25", stats.betabinom.cdf(k, n, 1 + k0, 1 + n0 - k0), 2)
put("pp25_base", stats.binom.cdf(k, n, pay_set.beat.mean()), 2)

# leave-one-class-out choice of the window: the 3-tenth window is picked on two classes, scored on the third
WIN = list(range(3, TW - 2))
for t0 in WIN:
    pay_set[f"w{t0}"] = adj_z(pay_set.assign(_w=pay_set[[f"v{t}" for t in range(t0, t0 + 3)]].mean(1)), "_w", ZS)
LW = []
for yr in CLASSES:
    trn, tst = pay_set[pay_set.draft_year != yr], pay_set[pay_set.draft_year == yr]
    best = max(WIN, key=lambda t0: partial(trn, f"w{t0}", "peb", ZB, nb=1)[0])
    hi_, lo_ = trn[f"w{best}"].quantile(2 / 3), trn[f"w{best}"].quantile(1 / 3)
    LW.append(dict(yr=yr, best=best / 10, kt=int(tst[tst[f"w{best}"] >= hi_].beat.sum()),
                   nt=int((tst[f"w{best}"] >= hi_).sum()), kb=int(tst[tst[f"w{best}"] <= lo_].beat.sum()),
                   nb=int((tst[f"w{best}"] <= lo_).sum())))
LW = pd.DataFrame(LW)
print(LW)
for nm in ["t", "b"]:
    k, n = int(LW[f"k{nm}"].sum()), int(LW[f"n{nm}"].sum())
    lo, hi = cp(k, n)
    put(f"lw_{nm}_k", k); put(f"lw_{nm}_n", n); put(f"lw_{nm}_pct", 100 * k / n, 0)
    put(f"lw_{nm}_lo", 100 * lo, 0); put(f"lw_{nm}_hi", 100 * hi, 0)
put("lw_same", int(sum(abs(b_ - 0.3) < 1e-9 for b_ in LW.best)))
put("p_lw_fisher", stats.fisher_exact([[int(LW.kt.sum()), int(LW.nt.sum() - LW.kt.sum())],
                                       [int(LW.kb.sum()), int(LW.nb.sum() - LW.kb.sum())]])[1], 3)
TABLES["loco_window"] = LW.to_dict("records")

# raw pressure by call third and role (the plus-call vs no-call table)
pay_set["third"] = np.where(pay_set.sfs_adj >= q_hi, "top", np.where(pay_set.sfs_adj <= q_lo, "bot", "mid"))
md = ["| NFL first step (third) | EDGE: n, pressure vs forecast, beat | Interior: n, pressure vs forecast, beat |",
      "|---|---|---|"]
for th, lab in [("top", "Top third (credit)"), ("mid", "Middle (no call)"), ("bot", "Bottom third (no credit)")]:
    cells = []
    for role, tg in [("EDGE", "e"), ("interior", "i")]:
        g_ = pay_set[(pay_set.third == th) & (pay_set.role == role)]
        key = f"rt_{th}_{tg}"
        put(f"{key}_rate", 100 * g_.pr.sum() / g_.rs.sum(), 1)
        put(f"{key}_fc", 100 * np.average(g_.forecast, weights=g_.rs), 1)
        put(f"{key}_beat", 100 * g_.beat.mean(), 0)
        cells.append(f"{len(g_)}: {{{{{key}_rate}}}}% vs {{{{{key}_fc}}}}%, {int(g_.beat.sum())} of {len(g_)}")
    md.append(f"| {lab} | {cells[0]} | {cells[1]} |")
TABLES["md"]["roles"] = "\n".join(md)

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

# ============================================================================ round-10 additions
tick("per-class hit rates")
import statsmodels.api as sm  # noqa: E402


def midp(kt, nt, kb, nb_):
    """One-sided mid-p (top beats more often than bottom) from the hypergeometric of the 2x2 table."""
    rv = stats.hypergeom(nt + nb_, kt + kb, nt)
    return float(rv.sf(kt) + 0.5 * rv.pmf(kt))


def fisher1(kt, nt, kb, nb_):
    return float(stats.fisher_exact([[kt, nt - kt], [kb, nb_ - kb]], alternative="greater")[1])


PC = []
for yr in CLASSES:
    tt, bb_ = top[top.draft_year == yr], bot[bot.draft_year == yr]
    row = dict(yr=yr, kt=int(tt.beat.sum()), nt=len(tt), kb=int(bb_.beat.sum()), nb=len(bb_))
    row["fisher"], row["midp"] = fisher1(row["kt"], row["nt"], row["kb"], row["nb"]), midp(row["kt"], row["nt"], row["kb"], row["nb"])
    PC.append(row)
    for k_ in ["kt", "nt", "kb", "nb"]:
        put(f"pc{yr}_{k_}", row[k_])
    put(f"pc{yr}_fisher", row["fisher"], 2); put(f"pc{yr}_midp", row["midp"], 2)
put("p_fisher1", fisher1(kt, nt, kb, nb_), 3); put("p_midp", midp(kt, nt, kb, nb_), 3)
TABLES["per_class"] = PC

# ---- exposure model: quasi-binomial pressures out of rushes (rushes are the exposure), with rotation share and a
# pick x rotation term; slopes for 100+ and 30-99 rush rushers from one model
tick("exposure model")
sg = seasons[seasons.position_group == "DL"].groupby("nfl_id").agg(gm=("games", "sum"), qp=("quick_pressures", "sum"))
EX = dl.merge(sg.reset_index(), on="nfl_id", how="left")
EX = EX[EX.rs >= 30].dropna(subset=["fs", "w_z", "pr", "gm"]).copy()
EX["sfs_adj"] = adj_z(EX, "fs", ZS)
EX["rot"] = EX.rs / EX.gm
EX["rot_z"] = (EX.rot - EX.rot.mean()) / EX.rot.std()
EX["lp_c"] = EX.log_pick - EX.log_pick.mean()
EX["pick_x_rot"] = EX.lp_c * EX.rot_z
EX["low"] = (EX.rs < MIN_RUSH).astype(float)
EX["sfs_low"] = EX.sfs_adj * EX.low
XE = ["lp_c", "w_z", "edge", "rot_z", "pick_x_rot"]


def qglm(d, cols):
    X = sm.add_constant(d[cols].astype(float), has_constant="add")
    return sm.GLM((d.pr / d.rs).to_numpy(float), X, family=sm.families.Binomial(),
                  var_weights=d.rs.to_numpy(float)).fit(scale="X2")


Z90 = stats.norm.ppf(0.95)
hi_ex = EX[EX.rs >= MIN_RUSH]
m_hi = qglm(hi_ex, XE + ["sfs_adj"])
mu_hi = hi_ex.pr.sum() / hi_ex.rs.sum()
dpp = 100 * mu_hi * (1 - mu_hi)          # pressure-rate points per unit log-odds near the mean
b_, se_ = m_hi.params.sfs_adj, m_hi.bse.sfs_adj
put("ex_n", len(hi_ex)); put("ex_pp", b_ * dpp, 1); put("ex_pp_lo", (b_ - Z90 * se_) * dpp, 1)
put("ex_pp_hi", (b_ + Z90 * se_) * dpp, 1); put("ex_p", m_hi.pvalues.sfs_adj, 2); put("ex_disp", m_hi.scale, 1)
put("ex_pxr", m_hi.params.pick_x_rot, 2); put("ex_pxr_p", m_hi.pvalues.pick_x_rot, 2)
put("ex_pr_season", b_ * dpp / 100 * RUSH_SEASON, 1)
m_all = qglm(EX, XE + ["low", "sfs_adj", "sfs_low"])
put("ex_all_n", len(EX)); put("ex_low_n", int(EX.low.sum()))
b_hi = m_all.params.sfs_adj
b_lo = m_all.params.sfs_adj + m_all.params.sfs_low
V = m_all.cov_params()
se_lo = np.sqrt(V.loc["sfs_adj", "sfs_adj"] + V.loc["sfs_low", "sfs_low"] + 2 * V.loc["sfs_adj", "sfs_low"])
mu_all = EX.pr.sum() / EX.rs.sum()
put("ex_s100", b_hi * dpp, 1); put("ex_s30", b_lo * 100 * mu_all * (1 - mu_all), 1)
put("ex_s30_lo", (b_lo - Z90 * se_lo) * 100 * mu_all * (1 - mu_all), 1)
put("ex_s30_hi", (b_lo + Z90 * se_lo) * 100 * mu_all * (1 - mu_all), 1)
put("ex_int_p", m_all.pvalues.sfs_low, 2)
# out of class: fit on two classes, held-out binomial log-likelihood per 1,000 rushes, with and without first step
EXO = []
for yr in CLASSES:
    tr_, te_ = hi_ex[hi_ex.draft_year != yr], hi_ex[hi_ex.draft_year == yr]
    ll = []
    for cols in [XE, XE + ["sfs_adj"]]:
        m_ = qglm(tr_, cols)
        p_ = m_.predict(sm.add_constant(te_[cols].astype(float), has_constant="add"))
        ll.append(np.sum(te_.pr * np.log(p_) + (te_.rs - te_.pr) * np.log(1 - p_)) / te_.rs.sum() * 1000)
    bt_coef = qglm(tr_, XE + ["sfs_adj"]).params.sfs_adj
    EXO.append(dict(yr=yr, dll=ll[1] - ll[0], coef=bt_coef, n=len(te_)))
    put(f"exo{yr}_dll", ll[1] - ll[0], 2); put(f"exo{yr}_n", len(te_))
put("exo_pos", int(sum(e["dll"] > 0 for e in EXO)))
TABLES["exposure_ooc"] = EXO

# ---- mechanism check: quick (early) vs late pressures, same partial correlation
tick("mechanism")
mq = pay_set.merge(sg.reset_index()[["nfl_id", "qp"]], on="nfl_id", how="left")
mq["q_rate"] = (mq.qp + 1) / (mq.rs + 20)
mq["l_rate"] = (mq.pr - mq.qp + 1) / (mq.rs + 20)
for col, tag in [("q_rate", "quick"), ("l_rate", "late")]:
    r, lo, hi, n, _ = partial(mq, "sfs_adj", col, ZB)
    put(f"mech_{tag}", r); put(f"mech_{tag}_lo", lo); put(f"mech_{tag}_hi", hi)
put("quick_share", 100 * mq.qp.sum() / mq.pr.sum(), 0)

# ---- sensitivity grid: cut-off x minimum rushes x forecast spec
tick("sensitivity grid")
dlx = dl.merge(sg.reset_index()[["nfl_id", "gm"]], on="nfl_id", how="left")
dlx["rot_z"] = dlx.rs / dlx.gm


def oof(d, cols):
    out = pd.Series(index=d.index, dtype=float)
    for yr in CLASSES:
        tr_, te_ = d[d.draft_year != yr], d[d.draft_year == yr]
        bb_ = wls(np.c_[np.ones(len(tr_)), tr_[cols].to_numpy(float)], tr_.peb.to_numpy(), tr_.rs.to_numpy(float))
        out.loc[te_.index] = np.c_[np.ones(len(te_)), te_[cols].to_numpy(float)] @ bb_
    return out


SPECS = [("pick only", ["log_pick"]), ("pick, weight, role", ZB), ("+ rotation", ZB + ["rot_z"])]
CUTS = [("halves", 1 / 2), ("thirds", 1 / 3), ("quartiles", 1 / 4)]
MINS = [50, 100, 150]
GRID = []
for mr in MINS:
    for sname, scols in SPECS:
        d = dlx[dlx.rs >= mr].dropna(subset=["fs", "w_z", "peb", "rot_z"]).copy()
        d["rot_z"] = (d.rot_z - d.rot_z.mean()) / d.rot_z.std()
        d["sfs_adj"] = adj_z(d, "fs", ZS)
        d["beat"] = (d.prate > oof(d, scols)).astype(int)
        for cname, fr_ in CUTS:
            t_, b2 = d[d.sfs_adj >= d.sfs_adj.quantile(1 - fr_)], d[d.sfs_adj <= d.sfs_adj.quantile(fr_)]
            GRID.append(dict(min_rush=mr, spec=sname, cut=cname, n=len(d), nt=len(t_),
                             diff=100 * (t_.beat.mean() - b2.beat.mean()),
                             p=fisher1(int(t_.beat.sum()), len(t_), int(b2.beat.sum()), len(b2))))
GR = pd.DataFrame(GRID)
TABLES["grid"] = GR.round(3).to_dict("records")
print(GR)
put("grid_n", len(GR)); put("grid_pos", int((GR["diff"] > 0).sum()))
put("grid_min", GR["diff"].min(), 0); put("grid_max", GR["diff"].max(), 0)
put("grid_sig", int((GR.p < 0.05).sum()))

# ---- power for the 2026 prospective test (one new class; per-arm n = median per-class third size)
tick("power")
n_arm = int(np.median([r_["nt"] for r_ in PC]))
p_t, p_b = kt / nt, kb / nb_
mid_, half_ = (p_t + p_b) / 2, (p_t - p_b) / 4
PRNG = np.random.default_rng(7)


def power(n, a, b, nsim=1500):
    hits = 0
    for _ in range(nsim):
        x, y_ = PRNG.binomial(n, a), PRNG.binomial(n, b)
        hits += fisher1(x, n, y_, n) < 0.05
    return hits / nsim


put("pow_narm", n_arm)
put("pow_obs", 100 * power(n_arm, p_t, p_b), 0)
put("pow_half", 100 * power(n_arm, mid_ + half_, mid_ - half_), 0)
put("pow_half_pt", 100 * (mid_ + half_), 0); put("pow_half_pb", 100 * (mid_ - half_), 0)
need = next((n for n in range(8, 200, 2) if power(n, mid_ + half_, mid_ - half_, 600) >= 0.8), 200)
put("pow_need", need); put("pow_need_classes", int(np.ceil(need / n_arm)))

# ---- false-positive case study: Maason Smith and the snap check (post hoc)
tick("case study")
s_go = seasons[seasons.position_group == "DL"].copy()
s_go["wgo"] = s_go.avg_get_off * s_go.pass_rush_snaps
gom = (s_go.groupby("nfl_id").wgo.sum() / s_go.groupby("nfl_id").pass_rush_snaps.sum()).rename("go")
pay_set["go_pct"] = 100 * (-pay_set.nfl_id.map(gom)).rank(pct=True)
rxm = gd.groupby("nfl_id").react.mean()
pay_set["react"] = pay_set.nfl_id.map(rxm)
put("rho_react_go", stats.spearmanr(pay_set.react, pay_set.nfl_id.map(gom), nan_policy="omit")[0])
tp_ = pay_set[pay_set.third == "top"]
fast_go, slow_go = tp_[tp_.go_pct >= 50], tp_[tp_.go_pct < 50]
put("snap_fast_k", int(fast_go.beat.sum())); put("snap_fast_n", len(fast_go))
put("snap_slow_k", int(slow_go.beat.sum())); put("snap_slow_n", len(slow_go))
fp = pay_set[(pay_set.third == "top") & (pay_set.prate < pay_set.forecast)].sort_values("sfs_adj").iloc[-1]
fpq = sg.loc[fp.nfl_id]
fpa = align[align.nfl_id == fp.nfl_id]
put("fp_rs", int(fp.rs)); put("fp_wt", int(fp.combine_weight)) if "combine_weight" in fp else put("fp_wt", int(pl.set_index("nfl_id").combine_weight[fp.nfl_id])); put("fp_pr", int(fp.pr)); put("fp_qp", int(fpq.qp))
put("fp_quick", 100 * fpq.qp / fp.pr, 0)
put("fp_go_pct", fp.go_pct, 0); put("fp_go", gom.loc[fp.nfl_id], 2)
put("go_med", gom.loc[pay_set.nfl_id].median(), 2)
put("fp_react", fp.react, 2); put("react_med", pay_set.react.median(), 2)
put("fp_interior", 100 * fpa[fpa.lined_up_position == "INTERIOR_LINE"].plays.sum() / fpa.plays.sum(), 0)
put("fp_lost", (fp.forecast - fp.prate) * fp.rs, 0)
put("fp_lost_season", (fp.forecast - fp.prate) * RUSH_SEASON, 0)
TABLES["false_positive"] = fp.display_name
# scheme check: is a fast start a stunt or slant? lateral share of the first 1.5 s after the snap (|dy| / net move)
lf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "x", "y"]).merge(
    gd[K].merge(gp[K + ["snap_frame"]], on=K), on=K)
lf = lf[(lf.frame >= lf.snap_frame) & (lf.frame <= lf.snap_frame + 15)].sort_values(K + ["frame"])
ends = lf.groupby(K).agg(x0=("x", "first"), x1=("x", "last"), y0=("y", "first"), y1=("y", "last")).reset_index()
ends["lat"] = (ends.y1 - ends.y0).abs() / np.hypot(ends.x1 - ends.x0, ends.y1 - ends.y0).clip(lower=1e-6)
latp = ends.groupby("nfl_id").lat.mean()
pay_set["lat"] = pay_set.nfl_id.map(latp)
put("rho_lat_fs", stats.spearmanr(pay_set.lat, pay_set.sfs_adj, nan_policy="omit")[0])
put("fp_lat_pct", 100 * pay_set.lat.rank(pct=True)[pay_set.nfl_id == fp.nfl_id].iloc[0], 0)
tl_ = pay_set[(pay_set.third == "top")]
hl, ll_ = tl_[tl_.lat >= pay_set.lat.median()], tl_[tl_.lat < pay_set.lat.median()]
put("lat_hi_k", int(hl.beat.sum())); put("lat_hi_n", len(hl)); put("lat_lo_k", int(ll_.beat.sum())); put("lat_lo_n", len(ll_))
print("false positive:", fp.display_name)

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

# ============================================================================ round-11: one clean held-out test
# For each class held out in turn: the no-tracking forecast is fit on the other two classes and enters as an offset
# (logit scale) in a binomial model of the held-out class's pressures out of rushes. The per-class slope on a
# measure is then partially pooled across the three classes (normal-normal random effects, DerSimonian-Laird tau^2).
# Null: the measure shuffled within class, 2,000 times; the max-over-windows null repeats the shuffle for all 13
# first-step windows and keeps the largest pooled slope (the window was read off the payoff curve).
tick("held-out slope test")
NPERM = int(os.environ.get("BDB_NPERM", 2000))


def class_slope(d, x):
    """Quasi-binomial slope (intercept + x) with logit(forecast) offset; numpy IRLS, Pearson-X2 dispersion."""
    off = np.log(d.forecast / (1 - d.forecast)).to_numpy(float)
    y = (d.pr / d.rs).to_numpy(float)
    w = d.rs.to_numpy(float)
    X = np.c_[np.ones(len(d)), d[x].to_numpy(float)]
    beta = np.zeros(2)
    for _ in range(25):
        eta = off + X @ beta
        mu_ = 1 / (1 + np.exp(-eta))
        v = mu_ * (1 - mu_)
        H = X.T @ (X * (w * v)[:, None])
        step = np.linalg.solve(H, X.T @ (w * (y - mu_)))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu_ = 1 / (1 + np.exp(-(off + X @ beta)))
    v = mu_ * (1 - mu_)
    phi = np.sum(w * (y - mu_) ** 2 / v) / (len(y) - 2)
    cov = phi * np.linalg.inv(X.T @ (X * (w * v)[:, None]))
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def pool(b, se):
    w = 1 / se ** 2
    mu0 = (w * b).sum() / w.sum()
    Q = (w * (b - mu0) ** 2).sum()
    t2 = max(0.0, (Q - (len(b) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + t2)
    mu_ = (ws * b).sum() / ws.sum()
    shrunk = mu_ + (t2 / (t2 + se ** 2)) * (b - mu_) if t2 > 0 else np.full(len(b), mu_)
    return float(mu_), float(np.sqrt(1 / ws.sum())), t2, shrunk


def held_out(d, x):
    est = [class_slope(d[d.draft_year == yr], x) for yr in CLASSES]
    b = np.array([e[0] for e in est]); se = np.array([e[1] for e in est])
    mu_, s_, t2, sh = pool(b, se)
    return dict(b=b, se=se, mu=mu_, s=s_, t2=t2, shrunk=sh)


def shuffled(d, cols):
    yrs_ = d.draft_year.to_numpy()
    out = d.copy()
    perm = np.arange(len(d))
    for c_ in np.unique(yrs_):
        ii = np.flatnonzero(yrs_ == c_)
        perm[ii] = RNG.permutation(ii)
    for c_ in cols:
        out[c_] = d[c_].to_numpy()[perm]
    return out


bt = bt.merge(pay_set[["nfl_id", "sfs_adj"]].rename(columns={"sfs_adj": "sfs_all"}), on="nfl_id", how="left")
HT, HT_D = {}, {}
for tag, d, x in [("c40", bt, "cfs_adj"), ("carry", bt, "carry"), ("nfl", pay_set, "sfs_adj")]:
    H = held_out(d, x)
    mu_p = d.pr.sum() / d.rs.sum()
    dpp_ = 100 * mu_p * (1 - mu_p)
    nul = np.array([held_out(shuffled(d, [x]), x)["mu"] for _ in range(NPERM)])
    H["p"] = float((np.sum(np.abs(nul) >= abs(H["mu"])) + 1) / (NPERM + 1))
    H["n"], H["dpp"] = len(d), dpp_
    HT[tag] = H
    HT_D[tag] = d
    put(f"ht_{tag}_n", len(d))
    put(f"ht_{tag}_pp", H["mu"] * dpp_, 1)
    put(f"ht_{tag}_pp_lo", (H["mu"] - Z90 * H["s"]) * dpp_, 1); put(f"ht_{tag}_pp_hi", (H["mu"] + Z90 * H["s"]) * dpp_, 1)
    put(f"ht_{tag}_p", H["p"], 3 if H["p"] < 0.01 else 2)
    put(f"ht_{tag}_pos", int(np.sum(H["b"] > 0)))
    for i_, yr in enumerate(CLASSES):
        put(f"ht_{tag}_{yr}", H["b"][i_] * dpp_, 1)
        put(f"ht_{tag}_{yr}_lo", (H["b"][i_] - Z90 * H["se"][i_]) * dpp_, 1)
        put(f"ht_{tag}_{yr}_hi", (H["b"][i_] + Z90 * H["se"][i_]) * dpp_, 1)
        put(f"ht_{tag}_{yr}_n", int((d.draft_year == yr).sum()))
put("ht_nfl_season", float(NUM["ht_nfl_pp"]) / 100 * RUSH_SEASON, 1)
put("ht_nfl_season_lo", float(NUM["ht_nfl_pp_lo"]) / 100 * RUSH_SEASON, 1)
put("ht_nfl_season_hi", float(NUM["ht_nfl_pp_hi"]) / 100 * RUSH_SEASON, 1)
# multiplicity: the largest pooled slope over the 13 windows, under the same within-class shuffle
WCOLS = [f"w{t0}" for t0 in WIN]
obs_w = {c_: held_out(pay_set, c_)["mu"] for c_ in WCOLS}
put("ht_nwin", len(WCOLS))
put("ht_best_win", int(np.argmax(list(obs_w.values())) + 3))
nmax = []
for _ in range(max(NPERM // 4, 200)):
    sh_ = shuffled(pay_set, WCOLS)
    nmax.append(max(held_out(sh_, c_)["mu"] for c_ in WCOLS))
put("ht_nfl_p_adj", float((np.sum(np.array(nmax) >= HT["nfl"]["mu"]) + 1) / (len(nmax) + 1)), 2)
# 2025 only, never used to choose anything: one-sided p for its own slope
z25 = HT["nfl"]["b"][2] / HT["nfl"]["se"][2]
put("ht_nfl_2025_p1", 1 - stats.norm.cdf(z25), 2)
# power of one new class at the pooled slope, using the median per-class standard error
se_med = float(np.median(HT["nfl"]["se"]))
put("pow_cls", 100 * (1 - stats.norm.cdf(stats.norm.ppf(0.95) - HT["nfl"]["mu"] / se_med)), 0)
need_c = next(k for k in range(1, 60) if 1 - stats.norm.cdf(stats.norm.ppf(0.95) - HT["nfl"]["mu"] / (se_med / np.sqrt(k))) >= 0.8)
put("pow_cls_need", need_c)
# held-out calibration: forecast vs actual by first-step third (rush-weighted), for the figure
TABLES["held_out"] = {k_: {"b": list(v["b"]), "se": list(v["se"]), "mu": v["mu"], "s": v["s"], "p": v["p"]}
                      for k_, v in HT.items()}

# ============================================================================ round-11: mechanism snap by snap
# Within rusher: on random sampled pass snaps, is a faster first step than his own norm followed by more pressure on
# that same snap? Pressure on a sampled snap = NGS time-to-pressure recorded or a sack.
tick("snap-level mechanism")
import statsmodels.formula.api as smf  # noqa: E402

SN = gd[gd.nfl_id.isin(pay_set.nfl_id)].merge(
    gp[K + ["time_to_pressure", "sack", "down", "yards_to_go"]], on=K, how="left").copy()
SN["press"] = (SN.time_to_pressure.notna() | (SN.sack == 1)).astype(int)
SN["fs_w"] = SN.fs_a - SN.groupby("nfl_id").fs_a.transform("mean")
SN["fs_wz"] = SN.fs_w / SN.fs_w.std()
SN = SN.merge(ends[K + ["lat"]], on=K, how="left")
SN["lat_w"] = SN.lat - SN.groupby("nfl_id").lat.transform("mean")
SN["lat_w"] = SN.lat_w / SN.lat_w.std()
SN["pdown"] = ((SN.down >= 3) | ((SN.down == 2) & (SN.yards_to_go >= 8))).astype(int)
SN["go_w"] = -(SN.player_get_off - SN.groupby("nfl_id").player_get_off.transform("mean"))
SN["go_wz"] = SN.go_w / SN.go_w.std()
SN = SN.merge(pay_set[["nfl_id", "draft_year", "role"]], on="nfl_id")


def clog(f, d):
    return smf.glm(f, data=d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d.nfl_id})


m0 = clog("press ~ fs_wz", SN)
put("sn_n", len(SN)); put("sn_k", int(SN.press.sum())); put("sn_players", SN.nfl_id.nunique())
put("sn_rate", 100 * SN.press.mean(), 1)
put("sn_or", np.exp(m0.params.fs_wz)); put("sn_or_lo", np.exp(m0.params.fs_wz - Z90 * m0.bse.fs_wz))
put("sn_or_hi", np.exp(m0.params.fs_wz + Z90 * m0.bse.fs_wz))
nul = []
for _ in range(500):
    d_ = SN.copy()
    d_["fs_wz"] = d_.groupby("nfl_id").fs_wz.transform(lambda v: RNG.permutation(v.to_numpy()))
    nul.append(smf.glm("press ~ fs_wz", data=d_, family=sm.families.Binomial()).fit().params.fs_wz)
put("sn_p", float((np.sum(np.abs(nul) >= abs(m0.params.fs_wz)) + 1) / 501), 3)
SNG = SN.dropna(subset=["go_wz", "lat_w"])
m1 = clog("press ~ fs_wz + go_wz + lat_w + pdown + C(lined_up_position)", SNG)
put("sn_or_ctl", np.exp(m1.params.fs_wz)); put("sn_ctl_n", len(SNG))
put("sn_or_go", np.exp(m1.params.go_wz)); put("sn_p_go", m1.pvalues.go_wz, 2)
put("rho_sn_go", stats.spearmanr(SNG.fs_w, SNG.go_w)[0])
SNC = []
for yr in CLASSES:
    m_ = clog("press ~ fs_wz", SN[SN.draft_year == yr])
    SNC.append(dict(yr=yr, b=m_.params.fs_wz, se=m_.bse.fs_wz, n=int((SN.draft_year == yr).sum())))
    put(f"sn_or_{yr}", np.exp(m_.params.fs_wz))
put("sn_pos", int(sum(r_["b"] > 0 for r_ in SNC)))
for role in ["EDGE", "interior"]:
    m_ = clog("press ~ fs_wz", SN[SN.role == role])
    put(f"sn_or_{role[:1].lower()}", np.exp(m_.params.fs_wz)); put(f"sn_n_{role[:1].lower()}", int((SN.role == role).sum()))
SN["fifth"] = pd.qcut(SN.fs_wz, 5, labels=False)
FIFTH = SN.groupby("fifth").press.agg(["sum", "size"]).reset_index()
put("sn_top5", 100 * FIFTH["sum"].iloc[-1] / FIFTH["size"].iloc[-1], 1)
put("sn_bot5", 100 * FIFTH["sum"].iloc[0] / FIFTH["size"].iloc[0], 1)
# quick (< 2.5 s) vs later pressures on the same snaps
SN["qk"] = (SN.time_to_pressure < 2.5).astype(int)
put("sn_qk_k", int(SN.qk.sum()))
# season pressure vs sampled-snap pressure (sanity: the sample measures the same thing)
pp_ = SN.groupby("nfl_id").press.mean()
put("rho_sample_season", stats.spearmanr(pp_, pay_set.set_index("nfl_id").prate.reindex(pp_.index))[0])

# ============================================================================ round-12: stronger baseline, robustness
# (1) A shrunken (ridge) no-tracking forecast, its penalty chosen inside the two training classes only; held-out
#     calibration slope per class for both forecasts (1 = perfectly calibrated).
# (2) The same held-out first-step test with the ridge forecast as offset, and with the incumbent in-game metric
#     (season NGS get-off) added to the forecast.
# (3) Cluster (player) bootstrap that refits the forecast in every resample; leave-one-player-out range.
# (4) Price: do clubs draft fast 40 starts earlier, after weight, role and the official 40 time?
tick("round-12 robustness")
FC = pay_set.copy()
FC["ngo_z"] = (FC.ngo - FC.ngo.mean()) / FC.ngo.std()
FC["ngo_z"] = FC.ngo_z.fillna(0.0)


def _std(tr, te, cols):
    m_, s_ = tr[cols].mean(), tr[cols].std()
    return ((tr[cols] - m_) / s_).to_numpy(float), ((te[cols] - m_) / s_).to_numpy(float)


def ridge_fit(Xs, y_, w_, lam):
    sw = np.sqrt(w_ / w_.mean())
    A = np.c_[np.ones(len(Xs)), Xs] * sw[:, None]
    pen = lam * np.eye(A.shape[1]); pen[0, 0] = 0
    return np.linalg.solve(A.T @ A + pen, A.T @ (y_ * sw))


LAMS = [0, 1, 3, 10, 30, 100, 300]


def forecast_oof(d, cols, kind="ols"):
    """Out-of-class forecast of shrunk pressure rate. ridge: penalty picked by leave-one-class-out inside train."""
    out = pd.Series(index=d.index, dtype=float)
    lam_used = {}
    for yr in CLASSES:
        tr_, te_ = d[d.draft_year != yr], d[d.draft_year == yr]
        lam = 0
        if kind == "ridge":
            err = []
            for lm in LAMS:
                e_ = 0.0
                for y2 in tr_.draft_year.unique():
                    a_, b2_ = tr_[tr_.draft_year != y2], tr_[tr_.draft_year == y2]
                    Xa, Xb = _std(a_, b2_, cols)
                    bb_ = ridge_fit(Xa, a_.peb.to_numpy(), a_.rs.to_numpy(float), lm)
                    e_ += np.sum(b2_.rs * (b2_.peb - np.c_[np.ones(len(Xb)), Xb] @ bb_) ** 2)
                err.append(e_)
            lam = LAMS[int(np.argmin(err))]
        Xa, Xb = _std(tr_, te_, cols)
        bb_ = ridge_fit(Xa, tr_.peb.to_numpy(), tr_.rs.to_numpy(float), lam)
        out.loc[te_.index] = np.clip(np.c_[np.ones(len(Xb)), Xb] @ bb_, 0.01, 0.3)
        lam_used[yr] = lam
    return out, lam_used


def calib(d, fcol):
    """Held-out calibration slope per class: rush-weighted regression of raw pressure rate on the forecast."""
    res = {}
    for yr in CLASSES:
        g_ = d[d.draft_year == yr]
        A = np.c_[np.ones(len(g_)), g_[fcol].to_numpy(float)]
        res[yr] = float(wls(A, g_.prate.to_numpy(float), g_.rs.to_numpy(float))[1])
    return res


FC["f_ols"], _ = forecast_oof(FC, ZB, "ols")
FC["f_ridge"], LAM_R = forecast_oof(FC, ZB, "ridge")
FC["f_ngs"], LAM_N = forecast_oof(FC, ZB + ["ngo_z"], "ridge")
print("ridge lambdas", LAM_R, LAM_N)
CAL = {k_: calib(FC, k_) for k_ in ["f_ols", "f_ridge", "f_ngs"]}
print("calibration", CAL)
for k_, tag in [("f_ols", "ols"), ("f_ridge", "ridge"), ("f_ngs", "ngs")]:
    for yr in CLASSES:
        put(f"cal_{tag}_{yr}", CAL[k_][yr])
put("lam_ridge_med", int(np.median(list(LAM_R.values()))))
# out-of-class R^2 of each forecast (rush-weighted, against the training-class mean)
def oos_r2(d, fcol):
    ss_r = np.sum(d.rs * (d.prate - d[fcol]) ** 2)
    ybar_ = d.groupby("draft_year").apply(lambda g_: np.average(d[d.draft_year != g_.name].prate,
                                                                 weights=d[d.draft_year != g_.name].rs))
    ss_t = np.sum(d.rs * (d.prate - d.draft_year.map(ybar_)) ** 2)
    return 1 - ss_r / ss_t


for k_, tag in [("f_ols", "ols"), ("f_ridge", "ridge"), ("f_ngs", "ngs")]:
    put(f"r2_{tag}", oos_r2(FC, k_), 2)
# full no-tracking baseline: pick, weight, role + official 40 time (missing -> 0 with a flag) + season NGS get-off
FC["f40z"] = FC.forty_z.fillna(0.0)
FC["no40"] = FC.forty_z.isna().astype(float)
ZF = ZB + ["f40z", "no40", "ngo_z"]
FC["f_full"], _ = forecast_oof(FC, ZF, "ols")
FC["f_pick40"], _ = forecast_oof(FC, ZB + ["f40z", "no40"], "ols")
CAL["f_full"] = calib(FC, "f_full")
for yr in CLASSES:
    put(f"cal_full_{yr}", CAL["f_full"][yr])
put("r2_full", oos_r2(FC, "f_full"), 2)
DPP = HT["nfl"]["dpp"]


def ht(d, x, fcol, tag, nperm=NPERM // 2):
    d_ = d.assign(forecast=d[fcol]).dropna(subset=[x, "forecast"])
    H = held_out(d_, x)
    nul = np.array([held_out(shuffled(d_, [x]), x)["mu"] for _ in range(nperm)])
    H["p"] = float((np.sum(np.abs(nul) >= abs(H["mu"])) + 1) / (len(nul) + 1))
    H["null"] = np.percentile(nul, [5, 95]) * DPP
    H["n"] = len(d_)
    put(f"{tag}_pp", H["mu"] * DPP, 1)
    put(f"{tag}_lo", (H["mu"] - Z90 * H["s"]) * DPP, 1)
    put(f"{tag}_hi", (H["mu"] + Z90 * H["s"]) * DPP, 1)
    put(f"{tag}_p", H["p"], 3 if H["p"] < 0.01 else 2)
    put(f"{tag}_pos", int(np.sum(H["b"] > 0)))
    put(f"{tag}_n", len(d_))
    return H


FC = FC.merge(bt[["nfl_id", "cfs_adj", "carry"]], on="nfl_id", how="left")
FC["f40_adj"] = np.nan
m40 = FC.forty_z.notna()
FC.loc[m40, "f40_adj"] = adj_z(FC[m40], "forty_z", ZS)
RT = {}
RT["ridge"] = ht(FC, "sfs_adj", "f_ridge", "x_ridge")
RT["nfl_full"] = ht(FC, "sfs_adj", "f_full", "x_nfl_full")
RT["c40_full"] = ht(FC, "cfs_adj", "f_full", "x_c40_full")
RT["carry_full"] = ht(FC, "carry", "f_full", "x_carry_full")
RT["f40_pick"] = ht(FC, "f40_adj", "f_ols", "x_f40_pick")
RT["f40_ngs"] = ht(FC, "f40_adj", "f_ngs", "x_f40_ngs")
RT["nfl_pick40"] = ht(FC, "sfs_adj", "f_pick40", "x_nfl_pick40")
# what a sd of official 40 is worth in draft slots: the pick-only forecast's slope on log pick, at pick 64
bpk = wls(np.c_[np.ones(len(FC)), FC[ZB].to_numpy(float)], 100 * FC.peb.to_numpy(), FC.rs.to_numpy(float))[1]
lp_shift = float(NUM["x_f40_pick_pp"]) / -bpk          # log-pick change with the same forecast effect
put("pick_equiv_from", 64); put("pick_equiv_to", int(round(64 * np.exp(-lp_shift))))
put("pp_per_logpick", -bpk, 1)
put("f40_season", float(NUM["x_f40_pick_pp"]) / 100 * RUSH_SEASON, 1)
# shuffle-null band for the headline (for the figure and text)
d_ = pay_set
nul = np.array([held_out(shuffled(d_, ["sfs_adj"]), "sfs_adj")["mu"] for _ in range(NPERM // 2)])
HT["nfl"]["null"] = np.percentile(nul, [5, 95]) * HT["nfl"]["dpp"]
put("null_hi", HT["nfl"]["null"][1], 1)
for tag in ["c40", "carry"]:
    d_ = HT_D[tag]
    x_ = {"c40": "cfs_adj", "carry": "carry"}[tag]
    nul = np.array([held_out(shuffled(d_, [x_]), x_)["mu"] for _ in range(NPERM // 4)])
    HT[tag]["null"] = np.percentile(nul, [5, 95]) * HT[tag]["dpp"]


# cluster bootstrap: resample players within class, refit the out-of-class forecast, rerun the held-out test
def full_test(d):
    d = d.copy()
    d["sfs_adj"] = adj_z(d, "fs", ZS)
    d["forecast"], _ = forecast_oof(d, ZB, "ols")
    return held_out(d, "sfs_adj")["mu"]


BCL = []
base_cols = ["nfl_id", "draft_year", "fs", "w_z", "edge", "log_pick", "peb", "rs", "pr", "prate"]
B0 = pay_set[base_cols].reset_index(drop=True)
for _ in range(NBOOT):
    parts = [g_.iloc[RNG.integers(0, len(g_), len(g_))] for _, g_ in B0.groupby("draft_year")]
    BCL.append(full_test(pd.concat(parts, ignore_index=True)))
BCL = np.array(BCL) * HT["nfl"]["dpp"]
put("cb_lo", np.percentile(BCL, 5), 1); put("cb_hi", np.percentile(BCL, 95), 1)
put("cb_pos", 100 * np.mean(BCL > 0), 0)
LOO = []
for i_ in range(len(B0)):
    LOO.append((B0.nfl_id.iloc[i_], full_test(B0.drop(index=i_)) * HT["nfl"]["dpp"]))
LOO = pd.DataFrame(LOO, columns=["nfl_id", "slope"]).merge(pl[["nfl_id", "display_name"]], on="nfl_id")
LOO = LOO.sort_values("slope").reset_index(drop=True)
put("loo_min", LOO.slope.min(), 1); put("loo_max", LOO.slope.max(), 1)
put("loo_full", full_test(B0) * HT["nfl"]["dpp"], 1)
TABLES["loo_min_name"] = LOO.display_name.iloc[0]
TABLES["loo_max_name"] = LOO.display_name.iloc[-1]
print("LOO extremes", LOO.head(3).to_dict("records"), LOO.tail(3).to_dict("records"))

# price: log pick on the tracked 40 start after weight, role and the official 40 time
PR_ = dl.merge(c40[["nfl_id", "c_fs"]].rename(columns={"c_fs": "cfs_raw"}), on="nfl_id", how="left")
PR_ = PR_.dropna(subset=["c_fs", "w_z", "forty_z"])
mP = sm.OLS(PR_.log_pick, sm.add_constant(PR_[["w_z", "edge", "forty_z", "c_fs"]])).fit()
put("price_n", len(PR_))
put("price_40", abs(100 * (np.exp(mP.params.forty_z) - 1)), 0); put("price_40_p", mP.pvalues.forty_z, 3)
put("price_40_p_lt", max(0.001, float(np.ceil(mP.pvalues.forty_z * 1000) / 1000)), 3)
put("price_cfs", 100 * (np.exp(mP.params.c_fs) - 1), 0); put("price_cfs_p", mP.pvalues.c_fs, 2)
# forward-in-time check (rookie-year first step -> later pressure beyond rookie pressure): too small, reported only
GY = gd.merge(pl[["nfl_id", "draft_year"]], on="nfl_id")
GY = GY[GY.season == GY.draft_year]
r0 = GY.groupby("nfl_id").fs_a.agg(["mean", "size"]).rename(columns={"mean": "fs0", "size": "n0"})
sy = seasons[seasons.position_group == "DL"].merge(pl[["nfl_id", "draft_year"]], on="nfl_id")
y0 = sy[sy.season == sy.draft_year].groupby("nfl_id").agg(rs0=("pass_rush_snaps", "sum"), pr0=("pressures", "sum"))
y1 = sy[sy.season > sy.draft_year].groupby("nfl_id").agg(rs1=("pass_rush_snaps", "sum"), pr1=("pressures", "sum"))
FW = dl[["nfl_id", "log_pick", "w_z", "edge"]].merge(r0.reset_index(), on="nfl_id").merge(
    y0.reset_index(), on="nfl_id").merge(y1.reset_index(), on="nfl_id")
FW = FW[(FW.n0 >= 5) & (FW.rs1 >= 50) & (FW.rs0 >= 20)].dropna().copy()
FW["fz"] = adj_z(FW, "fs0", ZS)
FW["r0"] = np.log((FW.pr0 + 1) / (FW.rs0 - FW.pr0 + 1))
mF = sm.GLM((FW.pr1 / FW.rs1).to_numpy(float), sm.add_constant(FW[ZB + ["r0", "fz"]].astype(float)),
            family=sm.families.Binomial(), var_weights=FW.rs1.to_numpy(float)).fit(scale="X2")
put("fw_n", len(FW)); put("fw_p", mF.pvalues.fz, 2)

# ledger of every test run on this angle (body table). Effects in pressure-rate pts per sd, held-out classes.
def eff(tag):
    return f"{{{{{tag}_pp}}}} [{{{{{tag}_lo}}}}, {{{{{tag}_hi}}}}]"


def dr(tag):
    return f"{{{{{tag}_r2}}}} [{{{{{tag}_lo}}}}, {{{{{tag}_hi}}}}]"


put("ht_nfl_lo", float(NUM["ht_nfl_pp_lo"]), 1); put("ht_nfl_hi", float(NUM["ht_nfl_pp_hi"]), 1)
put("ht_c40_lo", float(NUM["ht_c40_pp_lo"]), 1); put("ht_c40_hi", float(NUM["ht_c40_pp_hi"]), 1)
put("ht_carry_lo", float(NUM["ht_carry_pp_lo"]), 1); put("ht_carry_hi", float(NUM["ht_carry_pp_hi"]), 1)
LEDGER = [
    ("Tracked 40 start (Combine clip)", NUM["ht_c40_n"], eff("ht_c40"), eff("x_c40_full"), dr("dc40")),
    ("Carry-over (NFL start minus 40 start)", NUM["ht_carry_n"], eff("ht_carry"), eff("x_carry_full"), dr("dcarry")),
    ("Official 40 time (stopwatch)", NUM["x_f40_pick_n"], eff("x_f40_pick"), "in baseline", dr("d40")),
    ("**NFL first step (game tracking)**", NUM["ht_nfl_n"], "**" + eff("ht_nfl") + "**", "**" + eff("x_nfl_full") + "**",
     dr("dnfl")),
    ("Both flag (rule, 0/1)", NUM["ht_nfl_n"], "Table 2", "Table 2", dr("dboth")),
]
md = ["| Measure, one sd faster (DL) | n | Pts beyond pick, weight, role | Pts beyond that + 40 + NGS get-off | "
      "Held-out ΔR² |", "|---|---|---|---|---|"]
for q_, n_, e1, e2, e3 in LEDGER:
    md.append(f"| {q_} | {n_} | {e1} | {e2} | {e3} |")
TABLES["md"]["ledger"] = "\n".join(md)

# ============================================================================ round-11: frozen rule (hashed)
import hashlib  # noqa: E402

RULE = {"cohort": "DL, 8+ random NFL pass-rush starts set at the snap, 100+ career rushes",
        "first_step": "mean speed 0.3-0.5 s after first movement (speed first >= 1 yd/s minus 0.2 s), alignment-adjusted",
        "stopwatch": "official 40 time, year-standardized, residual on weight z and EDGE share",
        "adjust": "residual on Combine weight z and EDGE share, standardized",
        "forecast": "rush-weighted least squares of shrunk pressure rate on log pick (UDFA 262), weight z, EDGE share; fit on prior classes",
        "test": "binomial slope with logit(forecast) offset in the new class; one-sided p < 0.05, for the 40 time and the first step",
        "cutoffs": "set on the previous classes only (top third of first step, top half of 40 time)",
        "both": "first step top third and 40 time top half", "start_only": "first step top third, 40 time bottom half or none",
        "speed_only": "first step bottom third, 40 time top half", "neither": "first step bottom third, 40 time bottom half or none"}
blob = json.dumps(RULE, sort_keys=True).encode()
RULE_HASH = hashlib.sha256(blob).hexdigest()[:12]
json.dump({"rule": RULE, "sha256_12": RULE_HASH, "frozen_before": "2026 regular season data"},
          open(OUT / "frozen_rule.json", "w"), indent=1)
TABLES["rule_hash"] = RULE_HASH
NUM["rule_hash"] = RULE_HASH

# ============================================================================ decision card
tick("board")
board = pay_set[["nfl_id", "display_name", "draft_year", "pick", "role", "rs", "pr", "prate", "forecast", "sfs_adj",
                 "fs", "n_g", "go_pct", "react"]].merge(tr_set[["nfl_id", "cfs_adj", "carry"]], on="nfl_id", how="left")
board = board.merge(FC[["nfl_id", "f40_adj"]], on="nfl_id", how="left").merge(
    sg.reset_index()[["nfl_id", "qp"]], on="nfl_id", how="left")
board["s_pct"] = 100 * board.sfs_adj.rank(pct=True)
board["c_pct"] = 100 * board.cfs_adj.rank(pct=True)
board["t_pct"] = 100 * board.f40_adj.rank(pct=True)
board["q_share"] = 100 * board.qp / board.pr


# round 13: the cut-offs for each class come from the other two classes only (top third of NFL first step, top half
# of the official 40 time), so every call below is a held-out call
TH = {}
for yr in CLASSES:
    tr_ = board[board.draft_year != yr]
    TH[yr] = (tr_.sfs_adj.quantile(2 / 3), tr_.sfs_adj.quantile(1 / 3), tr_.f40_adj.median())


def call(r_):
    hi_, lo_, m40 = TH[r_.draft_year]
    fast40 = pd.notna(r_.f40_adj) and r_.f40_adj >= m40
    if r_.sfs_adj >= hi_:
        return "both" if fast40 else "start"
    if r_.sfs_adj <= lo_:
        return "speed" if fast40 else "neither"
    return "none"


board["call"] = board.apply(call, axis=1)
TEAM = seasons[seasons.position_group == "DL"].sort_values("season").groupby("nfl_id").team.agg(
    lambda v: "/".join(dict.fromkeys(v)))
board["team"] = board.nfl_id.map(TEAM).str.split("/").str[-1]
board["lat_pct"] = board.nfl_id.map(100 * pay_set.set_index("nfl_id").lat.rank(pct=True))
board["gap_season"] = (board.prate - board.forecast) * RUSH_SEASON
board = board.sort_values("sfs_adj", ascending=False)
board.to_csv(OUT / "board.csv", index=False)
put("n_board", len(board))
for c_ in ["both", "start", "speed", "neither"]:
    g_ = board[board.call == c_]
    put(f"call_{c_}_n", len(g_))
    put(f"call_{c_}_rate", 100 * g_.pr.sum() / g_.rs.sum(), 1)
    put(f"call_{c_}_fc", 100 * np.average(g_.forecast, weights=g_.rs), 1)
    put(f"call_{c_}_beat", int((g_.prate > g_.forecast).sum()))


def pick_txt(p):
    return "UDFA" if p >= UDFA_PICK else str(int(p))


# per-player actions: written for the player's role, team front and what the tape must answer (text only, no numbers)
ACTIONS = {
    "Byron Young": "Extension-grade: wide on every passing down",
    "Cedric Johnson": "Day-3 hit: third-down reps before paying a veteran",
    "Jordan Burch": "Miss so far: keep passing-down snaps, re-score at midseason",
    "Maason Smith": "Rule and tape disagree: grade hands and a counter first",
    "Laiatu Latu": "Meets forecast: keep him wide; do not project a jump",
    "Eric Watts": "Ignore the stopwatch: early-down edge-setter until the start shows",
}
CARD_NAMES = list(ACTIONS)
card = board[board.display_name.isin(CARD_NAMES)].copy()
card["order"] = card.display_name.map({n: i for i, n in enumerate(CARD_NAMES)})
card = card.sort_values("order")
LAB = {"both": "Both", "start": "Start only", "speed": "Speed only", "neither": "Neither", "none": "Middle"}
md = ["| Rusher (class, pick, team, role) | 40 time | 40 start | NFL first step | Pressure vs forecast | Call: action |",
      "|---|---|---|---|---|---|"]
NAMED = {}
for _, r_ in card.iterrows():
    key = re.sub(r"[^a-z]", "", r_.display_name.split()[-1].lower())
    NAMED[key] = r_
    if pd.notna(r_.c_pct):
        put(f"b_{key}_c", r_.c_pct, 0)
    if pd.notna(r_.t_pct):
        put(f"b_{key}_t", r_.t_pct, 0)
    put(f"b_{key}_s", r_.s_pct, 0); put(f"b_{key}_rate", 100 * r_.prate, 1); put(f"b_{key}_fc", 100 * r_.forecast, 1)
    put(f"b_{key}_lat", r_.lat_pct, 0); put(f"b_{key}_q", r_.q_share, 0); put(f"b_{key}_go", r_.go_pct, 0)
    put(f"b_{key}_gap", r_.gap_season, 0)
    ctxt = f"{{{{b_{key}_c}}}}" if pd.notna(r_.c_pct) else "–"
    ttxt = f"{{{{b_{key}_t}}}}" if pd.notna(r_.t_pct) else "no time"
    md.append(f"| {r_.display_name} ({r_.draft_year}{'*' if r_.draft_year == 2025 else ''}, {pick_txt(r_.pick)}, "
              f"{r_.team}, {r_.role}) | {ttxt} | {ctxt} | {{{{b_{key}_s}}}} | "
              f"{{{{b_{key}_rate}}}}% vs {{{{b_{key}_fc}}}}% | **{LAB[r_.call]}:** {ACTIONS[r_.display_name]} |")
TABLES["md"]["card"] = "\n".join(md)
TABLES["card_names"] = list(card.display_name)
print(card[["display_name", "draft_year", "pick", "team", "role", "t_pct", "c_pct", "s_pct", "prate", "forecast",
            "lat_pct", "q_share", "go_pct", "call"]].round(2).to_string())

# ============================================================================ round-13: the shortlist, held out
# "Both" = the held-out call above: the speed shows up on the stopwatch AND in the first ~20 random NFL starts.
# Hit = career pressure rate above the out-of-class pick forecast. Compared with every other rusher (base rate).
# Checks: shrunken (ridge) forecast; forecast that already knows the 40 time and NGS get-off; outcome with every
# sampled snap removed (so the first-step snaps cannot score their own pressures); quick pressures; sacks.
tick("shortlist")
SLc = ["nfl_id", "display_name", "draft_year", "pick", "role", "rs", "pr", "prate", "sfs_adj", "f40_adj", "log_pick",
       "w_z", "edge"]
SL = board[["nfl_id", "call", "forecast"]].merge(FC[SLc + ["f_ridge", "f_full"]], on="nfl_id")
SL["short"] = SL.call == "both"
gx = gp[gp.nfl_id.isin(SL.nfl_id)].assign(press=lambda d: (d.time_to_pressure.notna() | (d.sack == 1)).astype(int))
ex_ = gx.groupby("nfl_id").agg(n_s=("press", "size"), p_s=("press", "sum")).reset_index()
SL = SL.merge(ex_, on="nfl_id", how="left").fillna({"n_s": 0, "p_s": 0})
SL["rs_x"], SL["pr_x"] = SL.rs - SL.n_s, SL.pr - SL.p_s
SL["prate_x"] = SL.pr_x / SL.rs_x
put("sl_snaps_removed", int(SL.n_s.median()))
X_ = SL.assign(peb=(SL.pr_x + Mw * mu) / (SL.rs_x + Mw), rs=SL.rs_x)
SL["f_x"], _ = forecast_oof(X_, ZB, "ols")
sk_ = seasons[seasons.position_group == "DL"].groupby("nfl_id").agg(sacks=("sacks", "sum"), qp=("quick_pressures", "sum"))
SL = SL.merge(sk_.reset_index(), on="nfl_id", how="left")
SL["srate"], SL["qrate"] = SL.sacks / SL.rs, SL.qp / SL.rs
X_ = SL.assign(peb=(SL.sacks + 5) / (SL.rs + 500), rs=SL.rs)
SL["f_s"], _ = forecast_oof(X_, ZB, "ols")
X_ = SL.assign(peb=(SL.qp + 5) / (SL.rs + 200), rs=SL.rs)
SL["f_q"], _ = forecast_oof(X_, ZB, "ols")


def sl_test(y, f, tag, nperm=NPERM):
    h = (SL[y] > SL[f]).to_numpy()
    s_ = SL.short.to_numpy()
    ks, ns, ko, no = int(h[s_].sum()), int(s_.sum()), int(h[~s_].sum()), int((~s_).sum())
    obs = ks / ns - ko / no
    yrs_ = SL.draft_year.to_numpy()
    cnt = 0
    for _ in range(nperm):
        ss = s_.copy()
        for c_ in CLASSES:
            ii = np.flatnonzero(yrs_ == c_)
            ss[ii] = s_[RNG.permutation(ii)]
        cnt += (h[ss].mean() - h[~ss].mean()) >= obs - 1e-12
    p_ = (cnt + 1) / (nperm + 1)
    put(f"{tag}_ks", ks); put(f"{tag}_ns", ns); put(f"{tag}_ko", ko); put(f"{tag}_no", no)
    put(f"{tag}_ps", 100 * ks / ns, 0); put(f"{tag}_po", 100 * ko / no, 0); put(f"{tag}_base", 100 * h.mean(), 0)
    put(f"{tag}_p", p_, 3 if p_ < 0.01 else 2)
    lo_s, hi_s = cp(ks, ns); lo_o, hi_o = cp(ko, no)
    return dict(tag=tag, ks=ks, ns=ns, ko=ko, no=no, base=h.mean(), p=p_, ls=lo_s, hs=hi_s, lo=lo_o, ho=hi_o)


SLR = [sl_test("prate", "forecast", "sl"),
       sl_test("prate", "f_ridge", "sl_ridge"),
       sl_test("prate", "f_full", "sl_full"),
       sl_test("prate_x", "f_x", "sl_x"),
       sl_test("qrate", "f_q", "sl_q"),
       sl_test("srate", "f_s", "sl_s")]
# per class (main)
for yr in CLASSES:
    g_ = SL[(SL.draft_year == yr)]
    h_ = g_.prate > g_.forecast
    put(f"sl_{yr}_ks", int(h_[g_.short].sum())); put(f"sl_{yr}_ns", int(g_.short.sum()))
    put(f"sl_{yr}_po", 100 * h_[~g_.short].mean(), 0)
# bootstrap of the hit-rate gap (players resampled within class, labels fixed)
gap_b = []
for _ in range(NBOOT):
    parts = [g_.iloc[RNG.integers(0, len(g_), len(g_))] for _, g_ in SL.groupby("draft_year")]
    B_ = pd.concat(parts)
    hb = B_.prate > B_.forecast
    if B_.short.sum() == 0:
        continue
    gap_b.append(100 * (hb[B_.short].mean() - hb[~B_.short].mean()))
put("sl_gap", float(NUM["sl_ps"]) - float(NUM["sl_po"]), 0)
put("sl_gap_lo", np.percentile(gap_b, 5), 0); put("sl_gap_hi", np.percentile(gap_b, 95), 0)
# drop each shortlisted rusher in turn: lowest remaining hit rate
loo_ = []
for i_ in SL.index[SL.short]:
    g_ = SL.drop(index=i_)
    h_ = g_.prate > g_.forecast
    loo_.append(h_[g_.short].mean())
put("sl_loo_min", 100 * min(loo_), 0)
# pressures at stake: rush-weighted excess over forecast, per season of rushes
for tag, m_ in [("s", SL.short), ("o", ~SL.short)]:
    g_ = SL[m_]
    put(f"sl_ex_{tag}", np.average(100 * (g_.prate - g_.forecast), weights=g_.rs), 1)
    put(f"sl_exs_{tag}", np.average(g_.prate - g_.forecast, weights=g_.rs) * RUSH_SEASON, 1)
put("sl_day3", int((SL.short & (SL.pick >= 103)).sum()))
TABLES["shortlist"] = SL[SL.short].sort_values("pick")[["display_name", "draft_year", "pick", "role"]].to_dict("records")
print(SL[SL.short][["display_name", "draft_year", "pick", "role", "prate", "forecast", "prate_x", "f_x"]].round(3))
print(pd.DataFrame(SLR).round(3))

# ============================================================================ round-14 additions
# (1) Bounds on the translation null: disattenuated point estimate, a flat-prior posterior on the Fisher-z scale,
#     and a restriction to rushers with 15+ sampled starts (higher NFL reliability).
# (2) Time-forward test: the call is frozen at the end of the 2024 season (first step from 2023-24 snaps only,
#     cut-offs from that pool, no 2025 data anywhere) and scored on 2025 pressure against a persistence forecast
#     (2023-24 pressure rate shrunk toward a pick/weight/role expectation fit on 2023-24 only). A no-tracking
#     persistence shortlist (top third of shrunk 2023-24 pressure) is scored on the same outcome.
# (3) One robustness table for the shortlist (gap interval = players resampled within class, labels refit).
# (4) The 2026 list: the rule with cut-offs from all three classes, applied to every rusher with 8+ starts.
tick("round-14")
r_tr, n_tr_ = float(NUM["tr_fs"]), int(NUM["n_tr"])
rel_prod = float(NUM["rel40_fs"]) * float(NUM["rel_fs_game"])
put("tr_fs_dis", r_tr / np.sqrt(rel_prod))
se_z = 1 / np.sqrt(n_tr_ - 3 - len(ZS))
# observed-scale correlation implied by a true-score translation of 0.5
r_obs05 = 0.5 * np.sqrt(rel_prod)
put("tr_post05", 1 - stats.norm.cdf((np.arctanh(r_obs05) - np.arctanh(r_tr)) / se_z), 2)
put("tr_post03", 1 - stats.norm.cdf((np.arctanh(0.3) - np.arctanh(r_tr)) / se_z), 2)
t15 = tr_set[tr_set.n_g >= 15]
r, lo, hi, n, _ = partial(t15, "c_fs", "fs", ZS)
put("tr15", r); put("tr15_lo", lo); put("tr15_hi", hi); put("tr15_n", n)

# ---- (2) time-forward
FRZ = 2024
gpre = gd[gd.season <= FRZ]
gpre_p = gpre.groupby("nfl_id").agg(n_pre=("fs", "size"), fs_pre=("fs_a", "mean")).reset_index()
gpre_p = gpre_p[gpre_p.n_pre >= MIN_SNAPS_G]
apre = align[(align.season <= FRZ)].merge(dl[["nfl_id"]], on="nfl_id")
apre["e_"] = np.where(apre.lined_up_position.isin(["EDGE", "OLB"]), apre.plays, 0)
epre = (apre.groupby("nfl_id").e_.sum() / apre.groupby("nfl_id").plays.sum()).rename("edge_pre")
sdl = seasons[seasons.position_group == "DL"]
pre = sdl[sdl.season <= FRZ].groupby("nfl_id").agg(rs_pre=("pass_rush_snaps", "sum"), pr_pre=("pressures", "sum"))
post = sdl[sdl.season == FRZ + 1].groupby("nfl_id").agg(rs_post=("pass_rush_snaps", "sum"),
                                                          pr_post=("pressures", "sum"),
                                                          sk_post=("sacks", "sum"))
TF = dl[["nfl_id", "display_name", "draft_year", "pick", "log_pick", "w_z"]].merge(gpre_p, on="nfl_id") \
    .merge(epre.reset_index(), on="nfl_id").merge(pre.reset_index(), on="nfl_id") \
    .merge(post.reset_index(), on="nfl_id").merge(FC[["nfl_id", "f40_adj"]], on="nfl_id", how="left")
TF = TF[(TF.draft_year <= FRZ) & (TF.rs_pre >= 50) & (TF.rs_post >= MIN_RUSH)].dropna(subset=["w_z"]).copy()
TF["edge"] = TF.edge_pre
TF["sfs"] = adj_z(TF, "fs_pre", ["w_z", "edge"])
mu_pre = TF.pr_pre.sum() / TF.rs_pre.sum()
TF["peb_pre"] = (TF.pr_pre + Mw * mu_pre) / (TF.rs_pre + Mw)
bpre = wls(np.c_[np.ones(len(TF)), TF[ZB].to_numpy(float)], TF.peb_pre.to_numpy(), TF.rs_pre.to_numpy(float))
TF["f_pick"] = np.c_[np.ones(len(TF)), TF[ZB].to_numpy(float)] @ bpre
TF["forecast"] = (TF.pr_pre + Mw * TF.f_pick) / (TF.rs_pre + Mw)       # persistence: own record shrunk to pick
TF["prate"] = TF.pr_post / TF.rs_post
TF["pr"], TF["rs"] = TF.pr_post, TF.rs_post
hi_tf, m40_tf = TF.sfs.quantile(2 / 3), TF.f40_adj.median()
TF["short"] = (TF.sfs >= hi_tf) & (TF.f40_adj >= m40_tf)
TF["persist"] = TF.forecast >= TF.forecast.quantile(2 / 3)
TF["hit"] = TF.prate > TF.forecast
TF["hit_pick"] = TF.prate > TF.f_pick
put("tf_n", len(TF)); put("tf_snaps_med", int(TF.n_pre.median()))
for tag, col, hcol in [("tf", "short", "hit"), ("tfp", "persist", "hit"), ("tfb", "short", "hit_pick"),
                       ("tfq", "persist", "hit_pick")]:
    s_ = TF[col]
    put(f"{tag}_ks", int(TF.loc[s_, hcol].sum())); put(f"{tag}_ns", int(s_.sum()))
    put(f"{tag}_ko", int(TF.loc[~s_, hcol].sum())); put(f"{tag}_no", int((~s_).sum()))
put("tf_base", 100 * TF.hit.mean(), 0)
# continuous slope: binomial with logit(persistence forecast) offset, per sd of the frozen first step
b_tf, se_tf = class_slope(TF.assign(forecast=TF.forecast.clip(0.01, 0.3)), "sfs")
mu_tf = TF.pr.sum() / TF.rs.sum()
dpp_tf = 100 * mu_tf * (1 - mu_tf)
put("tf_pp", b_tf * dpp_tf, 1); put("tf_pp_lo", (b_tf - Z90 * se_tf) * dpp_tf, 1)
put("tf_pp_hi", (b_tf + Z90 * se_tf) * dpp_tf, 1)
print(TF[TF.short][["display_name", "draft_year", "pick", "prate", "forecast", "f_pick"]].round(3))


def gap_ci(d, hcol, scol, by="draft_year", nb=NBOOT):
    out = []
    for _ in range(nb):
        parts = [g_.iloc[RNG.integers(0, len(g_), len(g_))] for _, g_ in d.groupby(by)]
        B_ = pd.concat(parts)
        if B_[scol].sum() == 0 or (~B_[scol]).sum() == 0:
            continue
        out.append(100 * (B_.loc[B_[scol], hcol].mean() - B_.loc[~B_[scol], hcol].mean()))
    return np.percentile(out, [5, 95])


def label_p(d, hcol, scol, by="draft_year", nperm=NPERM):
    h, s_ = d[hcol].to_numpy(), d[scol].to_numpy()
    obs = h[s_].mean() - h[~s_].mean()
    g_ = d[by].to_numpy()
    cnt = 0
    for _ in range(nperm):
        ss = s_.copy()
        for c_ in np.unique(g_):
            ii = np.flatnonzero(g_ == c_)
            ss[ii] = s_[RNG.permutation(ii)]
        cnt += (h[ss].mean() - h[~ss].mean()) >= obs - 1e-12
    return (cnt + 1) / (nperm + 1)


lo_, hi_ = gap_ci(TF.assign(one=1), "hit", "short", by="one")
put("tf_gap", 100 * (TF.loc[TF.short, "hit"].mean() - TF.loc[~TF.short, "hit"].mean()), 0)
put("tf_gap_lo", lo_, 0); put("tf_gap_hi", hi_, 0)
put("tf_p", label_p(TF.assign(one=1), "hit", "short", by="one"), 2)

# ---- (3) robustness table for the shortlist
SL["hit"] = SL.prate > SL.forecast
SL["hit_full"] = SL.prate > SL.f_full
SL["hit_x"] = SL.prate_x > SL.f_x
SL["hit_s"] = SL.srate > SL.f_s
D3 = SL[SL.pick >= 103].copy()
ROB = []
for tag, lab, d_, hcol in [("rb_main", "Main: pick, weight, role forecast", SL, "hit"),
                           ("rb_full", "Forecast adds official 40 time, NGS get-off", SL, "hit_full"),
                           ("rb_x", "Sampled snaps removed from the outcome", SL, "hit_x"),
                           ("rb_d3", "Day 3 and UDFA only (pick 103+)", D3, "hit"),
                           ("rb_tf", "Frozen after 2024, scored on 2025 vs own past pressure", TF, "hit")]:
    sc = "short"
    by = "one" if tag == "rb_tf" else "draft_year"
    dd = d_.assign(one=1)
    ks, ns = int(dd.loc[dd[sc], hcol].sum()), int(dd[sc].sum())
    ko, no = int(dd.loc[~dd[sc], hcol].sum()), int((~dd[sc]).sum())
    lo_, hi_ = gap_ci(dd, hcol, sc, by=by)
    p_ = label_p(dd, hcol, sc, by=by)
    gap = 100 * (ks / ns - ko / no)
    put(f"{tag}_gap", gap, 0); put(f"{tag}_lo", lo_, 0); put(f"{tag}_hi", hi_, 0)
    put(f"{tag}_p", p_, 3 if p_ < 0.01 else 2); put(f"{tag}_base", 100 * dd[hcol].mean(), 0)
    ROB.append(dict(tag=tag, lab=lab, ks=ks, ns=ns, ko=ko, no=no, gap=gap, lo=lo_, hi=hi_, p=p_,
                    base=dd[hcol].mean()))
md = ["| Check | Shortlist beat forecast [90%] | Everyone else [90%] | Gap, pts [90%] |", "|---|---|---|---|"]
for r_ in ROB:
    t_ = r_["tag"]
    for side, k_, n_ in [("s", r_["ks"], r_["ns"]), ("o", r_["ko"], r_["no"])]:
        put(f"{t_}_j{side}_lo", 100 * stats.beta.ppf(0.05, k_ + 0.5, n_ - k_ + 0.5), 0)
        put(f"{t_}_j{side}_hi", 100 * stats.beta.ppf(0.95, k_ + 0.5, n_ - k_ + 0.5), 0)
    md.append(f"| {r_['lab']} | {r_['ks']} of {r_['ns']} [{{{{{t_}_js_lo}}}}–{{{{{t_}_js_hi}}}}%] | "
              f"{r_['ko']} of {r_['no']} [{{{{{t_}_jo_lo}}}}–{{{{{t_}_jo_hi}}}}%] | "
              f"{{{{{t_}_gap}}}} [{{{{{t_}_lo}}}}, {{{{{t_}_hi}}}}] |")
TABLES["md"]["robust"] = "\n".join(md)
TABLES["robust"] = ROB
print(pd.DataFrame(ROB).drop(columns="lab").round(3))
# sacks as a secondary outcome (text)
put("sl_s_ks2", int(SL.loc[SL.short, "hit_s"].sum()))

# ---- (4) the 2026 list: frozen rule, cut-offs from all three classes, every rusher with 8+ sampled starts
ALL = dl.dropna(subset=["fs", "w_z"]).copy()
ALL["pr"] = ALL.pr.fillna(0)
Aps = np.c_[np.ones(len(pay_set)), pay_set[ZS].to_numpy(float)]
bres = np.linalg.lstsq(Aps, pay_set.fs.to_numpy(float), rcond=None)[0]
res_ps = pay_set.fs.to_numpy(float) - Aps @ bres
ALL["s26"] = (ALL.fs - np.c_[np.ones(len(ALL)), ALL[ZS].to_numpy(float)] @ bres - res_ps.mean()) / res_ps.std()
ALL = ALL.merge(FC[["nfl_id", "f40_adj"]], on="nfl_id", how="left")
m40a = ALL.forty_z.notna() & ~ALL.nfl_id.isin(FC.nfl_id)
if m40a.any():
    A40 = np.c_[np.ones(m40.sum()), FC.loc[m40, ZS].to_numpy(float)]
    b40 = np.linalg.lstsq(A40, FC.loc[m40, "forty_z"].to_numpy(float), rcond=None)[0]
    r40 = FC.loc[m40, "forty_z"].to_numpy(float) - A40 @ b40
    ALL.loc[m40a, "f40_adj"] = (ALL.loc[m40a, "forty_z"] - np.c_[np.ones(m40a.sum()), ALL.loc[m40a, ZS].to_numpy(float)] @ b40
                                - r40.mean()) / r40.std()
hi26, m40_26 = board.sfs_adj.quantile(2 / 3), board.f40_adj.median()
ALL["both26"] = (ALL.s26 >= hi26) & (ALL.f40_adj >= m40_26)
bfc = wls(np.c_[np.ones(len(pay_set)), pay_set[ZB].to_numpy(float)], pay_set.peb.to_numpy(), pay_set.rs.to_numpy(float))
ALL["fc26"] = np.c_[np.ones(len(ALL)), ALL[ZB].to_numpy(float)] @ bfc
# round 16: one forecast per player everywhere in the text -- the held-out one (fit on the other two classes), which
# is what Tables 2 and Figures 1/3 use; for pool members it equals board.forecast exactly
for yr in CLASSES:
    tr_ = pay_set[pay_set.draft_year != yr]
    b_ = wls(np.c_[np.ones(len(tr_)), tr_[ZB].to_numpy(float)], tr_.peb.to_numpy(), tr_.rs.to_numpy(float))
    m_ = ALL.draft_year == yr
    ALL.loc[m_, "fc26"] = np.c_[np.ones(m_.sum()), ALL.loc[m_, ZB].to_numpy(float)] @ b_
_chk = ALL.merge(board[["nfl_id", "forecast"]], on="nfl_id")
assert np.allclose(_chk.fc26, _chk.forecast), "held-out forecast mismatch"
ALL["s_pct"] = 100 * stats.percentileofscore(board.sfs_adj, ALL.s26, kind="weak") / 100
ALL["t_pct"] = [100 * stats.percentileofscore(board.f40_adj.dropna(), v, kind="weak") / 100 if pd.notna(v) else np.nan
                for v in ALL.f40_adj]
ALL["team"] = ALL.nfl_id.map(TEAM).fillna("").str.split("/").str[-1]
L26 = ALL[ALL.both26].sort_values("s26", ascending=False).copy()
L26.to_csv(OUT / "shortlist_2026.csv", index=False)
put("l26_n", len(L26)); put("l26_new", int((L26.rs < MIN_RUSH).sum()))
put("l26_cls25", int((L26.draft_year == 2025).sum()))
# round 15: scheme (hand-down end / stand-up OLB / interior, from NFL alignment counts) and availability (games
# without a recorded snap since the draft, out of 17 a season through 2025: injury, inactive or cut)
AL_ = align.pivot_table(index="nfl_id", columns="lined_up_position", values="plays", aggfunc="sum").fillna(0)
for c_ in ["EDGE", "OLB", "INTERIOR_LINE"]:
    if c_ not in AL_:
        AL_[c_] = 0.0


def scheme(nid):
    if nid not in AL_.index:
        return "–"
    r_ = AL_.loc[nid]
    if r_.INTERIOR_LINE >= r_.EDGE + r_.OLB:
        return "interior"
    return "stand-up OLB" if r_.OLB > r_.EDGE else "hand-down end"


GMS = seasons[seasons.position_group == "DL"].groupby("nfl_id").games.sum()


def missed(nid, yr):
    return int(17 * (2025 - yr + 1) - GMS.get(nid, 0))


L26["scheme"] = L26.nfl_id.map(scheme)
L26["missed"] = [missed(n_, y_) for n_, y_ in zip(L26.nfl_id, L26.draft_year)]
L26["possible"] = 17 * (2025 - L26.draft_year + 1)
L26.to_csv(OUT / "shortlist_2026.csv", index=False)
md = ["| Rusher (class, pick, team) | Alignment | Games missed / possible | 40 time pct | NFL first step pct | "
      "Pressure to date vs forecast (rushes) |",
      "|---|---|---|---|---|---|"]
for _, r_ in L26.iterrows():
    key = re.sub(r"[^a-z]", "", r_.display_name.lower())[:14]
    put(f"l26_{key}_rate", 100 * r_.pr / r_.rs if r_.rs > 0 else 0.0, 1); put(f"l26_{key}_fc", 100 * r_.fc26, 1)
    md.append(f"| {r_.display_name} ({r_.draft_year}, {pick_txt(r_.pick)}, {r_.team}) | {r_.scheme} | "
              f"{r_.missed} / {r_.possible} | {int(round(r_.t_pct))} | "
              f"{min(int(round(r_.s_pct)), 100)} | {{{{l26_{key}_rate}}}}% vs {{{{l26_{key}_fc}}}}% "
              f"({int(r_.rs)}) |")
TABLES["md"]["list26"] = "\n".join(md)
put("l26_missed_med", int(L26.missed.median()))
put("l26_olb", int((L26.scheme == "stand-up OLB").sum())); put("l26_int", int((L26.scheme == "interior").sum()))
print(L26[["display_name", "draft_year", "pick", "team", "rs", "pr", "fc26", "s_pct", "t_pct", "scheme", "missed"]]
      .round(3).to_string())

# ============================================================================ round-15 additions
# (1) Exclusions and survivorship: who never reaches the pool, a selection model P(100+ rushes and 8+ starts) from
#     pick, weight, 40 time and class, and inverse-probability-weighted reruns of the held-out slope and shortlist.
# (2) Rusher random-intercept binomial model on player-seasons (Gauss-Hermite), logit(forecast) offset.
# (3) Shortlist calibration: the hit probability each class would have been told (from the other two classes).
# (4) Measurement protocol: reliability of the tracked 40 start with 1 and 2 reps; NFL first step by sampled starts.
# (5) Worked decision case: two 2024 day-3 edge rushers, one call apart, with contract context (OTC via nflverse).
# (6) Group screen for Figure 2C.
tick("round-15")
# ---- (1) exclusions
put("x_all", len(dl))
put("x_nostarts", int(dl.n_g.isna().sum()))
put("x_lowrush", int((dl.n_g.notna() & (dl.rs < MIN_RUSH)).sum()))
put("x_pool", len(pay_set))
SEL = dl.dropna(subset=["w_z"]).copy()
SEL["sel"] = SEL.nfl_id.isin(pay_set.nfl_id).astype(float)
SEL["f40z"] = SEL.forty_z.fillna(0.0)
SEL["no40"] = SEL.forty_z.isna().astype(float)
SEL["c24"] = (SEL.draft_year == 2024).astype(float)
SEL["c25"] = (SEL.draft_year == 2025).astype(float)
XS = ["log_pick", "w_z", "f40z", "no40", "c24", "c25"]
mS = sm.GLM(SEL.sel, sm.add_constant(SEL[XS]), family=sm.families.Binomial()).fit()
SEL["p_sel"] = mS.predict(sm.add_constant(SEL[XS]))
put("sel_n", len(SEL)); put("sel_k", int(SEL.sel.sum()))
put("sel_auc", float(np.mean([(a_ > b_) + 0.5 * (a_ == b_) for a_ in SEL.p_sel[SEL.sel == 1]
                              for b_ in SEL.p_sel[SEL.sel == 0]])))
put("sel_or_pick", np.exp(-mS.params.log_pick * np.log(2)))     # odds of making the pool, pick number halved
# does the NFL first step itself predict making the pool, among rushers with 8+ starts?
S8 = SEL[SEL.n_g.notna()].copy()
S8["sfs8"] = adj_z(S8, "fs", ZS)
mS8 = sm.GLM(S8.sel, sm.add_constant(S8[["log_pick", "w_z", "sfs8"]]), family=sm.families.Binomial()).fit()
put("sel_fs_or", np.exp(mS8.params.sfs8)); put("sel_fs_p", mS8.pvalues.sfs8, 2); put("sel8_n", len(S8))
ipw = (SEL.sel.mean() / SEL.p_sel).clip(upper=float((SEL.sel.mean() / SEL.p_sel).quantile(0.95)))
IPW = dict(zip(SEL.nfl_id, ipw))
put("ipw_max", max(IPW[n_] for n_ in pay_set.nfl_id), 1)


def class_slope_w(d, x, mult):
    """class_slope with rush weights multiplied by an inverse-probability weight."""
    off = np.log(d.forecast / (1 - d.forecast)).to_numpy(float)
    y = (d.pr / d.rs).to_numpy(float)
    w = d.rs.to_numpy(float) * mult
    X = np.c_[np.ones(len(d)), d[x].to_numpy(float)]
    beta = np.zeros(2)
    for _ in range(25):
        mu_ = 1 / (1 + np.exp(-(off + X @ beta)))
        H = X.T @ (X * (w * mu_ * (1 - mu_))[:, None])
        step = np.linalg.solve(H, X.T @ (w * (y - mu_)))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu_ = 1 / (1 + np.exp(-(off + X @ beta)))
    v = mu_ * (1 - mu_)
    phi = np.sum(w * (y - mu_) ** 2 / v) / (len(y) - 2) / np.mean(mult)
    cov = phi * np.linalg.inv(X.T @ (X * (w * v)[:, None]))
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


PI = pay_set.assign(ipw=pay_set.nfl_id.map(IPW))
est = [class_slope_w(PI[PI.draft_year == yr], "sfs_adj", PI[PI.draft_year == yr].ipw.to_numpy()) for yr in CLASSES]
mu_i, s_i, _, _ = pool(np.array([e[0] for e in est]), np.array([e[1] for e in est]))
put("ipw_pp", mu_i * DPP, 1); put("ipw_lo", (mu_i - Z90 * s_i) * DPP, 1); put("ipw_hi", (mu_i + Z90 * s_i) * DPP, 1)
SLI = SL.assign(ipw=SL.nfl_id.map(IPW))
hs_ = np.average(SLI.hit[SLI.short], weights=SLI.ipw[SLI.short])
ho_ = np.average(SLI.hit[~SLI.short], weights=SLI.ipw[~SLI.short])
put("ipw_hs", 100 * hs_, 0); put("ipw_ho", 100 * ho_, 0)
gb_ = []
for _ in range(NBOOT):
    B_ = pd.concat([g_.iloc[RNG.integers(0, len(g_), len(g_))] for _, g_ in SLI.groupby("draft_year")])
    if B_.short.sum() == 0:
        continue
    gb_.append(100 * (np.average(B_.hit[B_.short], weights=B_.ipw[B_.short]) -
                      np.average(B_.hit[~B_.short], weights=B_.ipw[~B_.short])))
put("ipw_gap", 100 * (hs_ - ho_), 0); put("ipw_gap_lo", np.percentile(gb_, 5), 0)
put("ipw_gap_hi", np.percentile(gb_, 95), 0)

# ---- (2) random-intercept binomial model on player-seasons
SEAS = seasons[(seasons.position_group == "DL") & seasons.nfl_id.isin(pay_set.nfl_id) & (seasons.pass_rush_snaps > 0)]
SEAS = SEAS[["nfl_id", "season", "pass_rush_snaps", "pressures"]].merge(
    pay_set[["nfl_id", "draft_year", "sfs_adj", "forecast"]], on="nfl_id")
SEAS["rookie"] = (SEAS.season == SEAS.draft_year).astype(float)
GHx, GHw = np.polynomial.hermite_e.hermegauss(40)
GHw = GHw / GHw.sum()
ids_ = pd.factorize(SEAS.nfl_id)[0]
kS, nS = SEAS.pressures.to_numpy(float), SEAS.pass_rush_snaps.to_numpy(float)
offS = np.log(SEAS.forecast / (1 - SEAS.forecast)).to_numpy(float)
XM = np.c_[np.ones(len(SEAS)), SEAS.rookie.to_numpy(float), SEAS.sfs_adj.to_numpy(float)]


def glmm_nll(th):
    b_, sg_ = th[:3], np.exp(th[3])
    eta = (offS + XM @ b_)[:, None] + sg_ * GHx[None, :]
    ll = kS[:, None] * eta - nS[:, None] * np.logaddexp(0, eta)            # per season x node
    tot = np.zeros((ids_.max() + 1, len(GHx)))
    np.add.at(tot, ids_, ll)
    m_ = tot.max(1, keepdims=True)
    return -np.sum(m_[:, 0] + np.log(np.exp(tot - m_) @ GHw))


from scipy.optimize import minimize  # noqa: E402

opt = minimize(glmm_nll, np.array([0.0, 0.0, 0.0, np.log(0.2)]), method="Nelder-Mead",
               options=dict(maxiter=4000, xatol=1e-7, fatol=1e-9))
opt = minimize(glmm_nll, opt.x, method="BFGS")
th_ = opt.x
hh = 1e-4
Hs = np.zeros((4, 4))
for i_ in range(4):
    for j_ in range(4):
        e_i, e_j = np.eye(4)[i_] * hh, np.eye(4)[j_] * hh
        Hs[i_, j_] = (glmm_nll(th_ + e_i + e_j) - glmm_nll(th_ + e_i - e_j) - glmm_nll(th_ - e_i + e_j)
                      + glmm_nll(th_ - e_i - e_j)) / (4 * hh * hh)
se_m = np.sqrt(np.diag(np.linalg.inv(Hs)))
put("mm_seasons", len(SEAS)); put("mm_rushers", SEAS.nfl_id.nunique())
put("mm_pp", th_[2] * DPP, 1); put("mm_lo", (th_[2] - Z90 * se_m[2]) * DPP, 1)
put("mm_hi", (th_[2] + Z90 * se_m[2]) * DPP, 1); put("mm_sd", np.exp(th_[3]), 2)

# ---- (3) shortlist calibration: what each held-out class would have been told
CALR = []
for yr in CLASSES:
    tr_, te_ = SL[SL.draft_year != yr], SL[SL.draft_year == yr]
    ps_ = (tr_.hit[tr_.short].sum() + 0.5) / (tr_.short.sum() + 1)          # Jeffreys
    po_ = (tr_.hit[~tr_.short].sum() + 0.5) / ((~tr_.short).sum() + 1)
    pb_ = tr_.hit.mean()
    for _, r_ in te_.iterrows():
        CALR.append(dict(yr=yr, short=r_.short, hit=float(r_.hit), p=ps_ if r_.short else po_, pb=pb_))
CALR = pd.DataFrame(CALR)
put("calib_ps", 100 * CALR.p[CALR.short].mean(), 0); put("calib_os", 100 * CALR.hit[CALR.short].mean(), 0)
put("calib_po", 100 * CALR.p[~CALR.short].mean(), 0); put("calib_oo", 100 * CALR.hit[~CALR.short].mean(), 0)
put("brier_rule", np.mean((CALR.p - CALR.hit) ** 2), 3); put("brier_base", np.mean((CALR.pb - CALR.hit) ** 2), 3)

# ---- (4) protocol
r40 = float(NUM["rel40_fs"])
put("rel40_2rep", 2 * r40 / (1 + r40))
put("dl40_2rep", 100 * float((events[(events.drill_name == "FORTY_YARD_DASH") & (events.position_group == "DL")]
                              .groupby("nfl_id").size() >= 2).mean()), 0)
r1s = float(NUM["rel_fs_1snap"])
put("snaps_for_07", int(np.ceil(0.7 * (1 - r1s) / (r1s * 0.3))))
put("rel_8snaps", 8 * r1s / (1 + 7 * r1s))
# translation ceiling even with two 40 reps: the most the observed rho could be if the true link were perfect
put("ceil_2rep", np.sqrt(float(NUM["rel40_2rep"]) * float(NUM["rel_fs_game"])))

# ---- (5) worked decision case: two 2024 day-3 edge rushers
WC = ["Cedric Johnson", "Javontae Jean-Baptiste"]
con = pd.read_parquet(ROOT / "data" / "external" / "nflverse" / "historical_contracts.parquet")
gsis = players.set_index("display_name").gsis_id
for nm in WC:
    r_ = board[board.display_name == nm].iloc[0]
    key = "cj" if nm.startswith("Cedric") else "jjb"
    put(f"wc_{key}_pick", int(r_.pick)); put(f"wc_{key}_rs", int(r_.rs)); put(f"wc_{key}_pr", int(r_.pr))
    put(f"wc_{key}_t", r_.t_pct, 0); put(f"wc_{key}_s", r_.s_pct, 0)
    put(f"wc_{key}_rate", 100 * r_.prate, 1); put(f"wc_{key}_fc", 100 * r_.forecast, 1)
    print(nm, LAB[r_.call], scheme(r_.nfl_id))
    put(f"wc_{key}_missed", missed(r_.nfl_id, int(r_.draft_year)))
    cc = con[(con.gsis_id == gsis[nm]) & (con.year_signed == int(r_.draft_year))]
    put(f"wc_{key}_apy", float(cc.apy.iloc[0]), 2)
# the same two at the decision point (end of 2024): first step from 2024 starts only, the frozen time-forward
# cut-offs, 2024 pressure, then what 2025 showed
Atf = np.c_[np.ones(len(TF)), TF[["w_z", "edge"]].to_numpy(float)]
btf = np.linalg.lstsq(Atf, TF.fs_pre.to_numpy(float), rcond=None)[0]
rtf = TF.fs_pre.to_numpy(float) - Atf @ btf
for nm in WC:
    key = "cj" if nm.startswith("Cedric") else "jjb"
    nid = int(board[board.display_name == nm].nfl_id.iloc[0])
    gq = gpre_p[gpre_p.nfl_id == nid].iloc[0]
    e_ = float(epre.get(nid, 0.0))
    wz_ = float(dl.loc[dl.nfl_id == nid, "w_z"].iloc[0])
    z_ = (gq.fs_pre - np.array([1.0, wz_, e_]) @ btf - rtf.mean()) / rtf.std()
    f40_ = float(FC.loc[FC.nfl_id == nid, "f40_adj"].iloc[0])
    print(nm, "frozen-2024 call:", "Both" if (z_ >= hi_tf and f40_ >= m40_tf) else
          ("Start only" if z_ >= hi_tf else "other"), round(z_, 2), round(hi_tf, 2), round(f40_, 2), round(m40_tf, 2))
    put(f"wc_{key}_n24", int(gq.n_pre))
    p24 = pre.loc[nid]; p25 = post.loc[nid] if nid in post.index else None
    put(f"wc_{key}_rs24", int(p24.rs_pre)); put(f"wc_{key}_rate24", 100 * p24.pr_pre / p24.rs_pre, 1)
    put(f"wc_{key}_rs25", int(p25.rs_post)); put(f"wc_{key}_pr25", int(p25.pr_post))
    put(f"wc_{key}_rate25", 100 * p25.pr_post / p25.rs_post, 1)
    g25 = seasons[(seasons.nfl_id == nid) & (seasons.season == 2025)].games
    put(f"wc_{key}_g25", int(g25.iloc[0]) if len(g25) else 0)
put("wc_gap_season", (float(NUM["wc_cj_rate"]) - float(NUM["wc_jjb_rate"])) / 100 * RUSH_SEASON, 0)
put("wc_rule_season", float(NUM["sl_exs_s"]) - float(NUM["sl_exs_o"]), 1)
vet = con[(con.position == "ED") & (con.year_signed == 2025) & (con.draft_year < 2022)].sort_values("apy", ascending=False)
put("vet_apy", float(vet.apy.iloc[20:40].median()), 1)
put("vet_lo", float(vet.apy.iloc[39]), 1); put("vet_hi", float(vet.apy.iloc[20]), 1)
print(board[board.display_name.isin(WC)].T)

# ---- (6) group screen: figure 2C uses GROWS; the text quotes the g_* keys

# ============================================================================ round-16 additions
# (1) Every pick-matched pair, not one hand-picked case: each "Both" rusher against the non-Both rusher of the same
#     class and role (EDGE / interior) nearest in log pick (twins may repeat). Outcome = raw pressure rate, so the
#     comparison needs no model. Done twice: career (held-out call) and frozen after 2024, scored on 2025.
#     Share of pairs won, mean gap, 90% bootstrap over pairs, exact one-sided sign test.
# (2) Track record by name for the 2026 list (frozen-2024 call scored on 2025), availability flag, act-on rule.
# (3) Pre-registered 2026 test: rule + list + targets hashed into frozen_rule.json, with power.
tick("round-16")


def match_pairs(d, scol, ycol, rcol):
    rows = []
    for _, r_ in d[d[scol]].iterrows():
        pool_ = d[(~d[scol]) & (d.draft_year == r_.draft_year) & (d[rcol] == r_[rcol])]
        if not len(pool_):
            continue
        j = (np.log(pool_.pick) - np.log(r_.pick)).abs().idxmin()
        t_ = d.loc[j]
        rows.append(dict(name=r_.display_name, twin=t_.display_name, yr=int(r_.draft_year), role=r_[rcol],
                         pick=r_.pick, tpick=t_.pick, y=100 * r_[ycol], ty=100 * t_[ycol],
                         rs=int(r_.rs), trs=int(t_.rs)))
    P_ = pd.DataFrame(rows)
    P_["win"] = P_.y > P_.ty
    P_["diff"] = P_.y - P_.ty
    return P_


def pair_stats(P_, tag):
    n_, w_ = len(P_), int(P_.win.sum())
    bw, bd = [], []
    for _ in range(NBOOT):
        i_ = RNG.integers(0, n_, n_)
        bw.append(100 * P_.win.to_numpy()[i_].mean()); bd.append(P_["diff"].to_numpy()[i_].mean())
    put(f"{tag}_n", n_); put(f"{tag}_w", w_)
    put(f"{tag}_diff", P_["diff"].mean(), 1)
    put(f"{tag}_diff_lo", np.percentile(bd, 5), 1); put(f"{tag}_diff_hi", np.percentile(bd, 95), 1)
    put(f"{tag}_wlo", np.percentile(bw, 5), 0); put(f"{tag}_whi", np.percentile(bw, 95), 0)
    put(f"{tag}_p", stats.binomtest(w_, n_, 0.5, alternative="greater").pvalue, 2)
    put(f"{tag}_pickgap", float(np.median(np.abs(P_.pick - P_.tpick))), 0)
    put(f"{tag}_twins", P_.twin.nunique())


SLm = SL.merge(dl[["nfl_id", "role"]].rename(columns={"role": "role_"}), on="nfl_id", how="left")
PAIRS = match_pairs(SLm, "short", "prate", "role_")
pair_stats(PAIRS, "mp")
TFm = TF.assign(role_=np.where(TF.edge >= 0.5, "EDGE", "interior"))
PAIRS_TF = match_pairs(TFm.assign(prate=TFm.pr_post / TFm.rs_post), "short", "prate", "role_")
pair_stats(PAIRS_TF, "mpf")
print(PAIRS.round(1).to_string()); print(PAIRS_TF.round(1).to_string())
PAIRS.to_csv(OUT / "pairs_career.csv", index=False); PAIRS_TF.to_csv(OUT / "pairs_frozen2024.csv", index=False)
# the Johnson / Jean-Baptiste illustration is one of the career pairs (checked, not chosen)
put("wc_in_pairs", int(((PAIRS.name == "Cedric Johnson") & (PAIRS.twin == "Javontae Jean-Baptiste")).any()))

# ---- (2) track record, availability, act-on rule for the 2026 list
TFi = TF.set_index("nfl_id")


def record(nid, yr, key):
    if yr == 2025:
        return "2025 class"
    if nid not in TFi.index:
        return "untested (<100 rushes)"
    r_ = TFi.loc[nid]
    if not r_.short:
        return "not on 2024 list"
    put(f"rec_{key}_a", 100 * r_.prate, 1); put(f"rec_{key}_b", 100 * r_.forecast, 1)
    return ("**hit**" if r_.hit else "**miss**") + f" ({{{{rec_{key}_a}}}}% vs {{{{rec_{key}_b}}}}%)"


L26["key"] = [re.sub(r"[^a-z]", "", n_.lower())[:14] for n_ in L26.display_name]
L26["prate26"] = np.where(L26.rs > 0, L26.pr / L26.rs.where(L26.rs > 0), 0.0)
L26["edge_pts"] = 100 * (L26.prate26 - L26.fc26)
L26["avail"] = L26.missed / L26.possible
L26["record"] = [record(n_, y_, k_) for n_, y_, k_ in zip(L26.nfl_id, L26.draft_year, L26.key)]
L26["act"] = (L26.rs >= 300) & (L26.edge_pts >= 1.0) & (L26.avail < 0.25)
put("act_n", int(L26.act.sum())); put("act_minrush", 300); put("avail_cut", 25); put("alpha", 0.05)
put("l26_flag", int((L26.avail >= 0.25).sum()))
put("rec_hit", int(L26.record.str.startswith("**hit").sum())); put("rec_miss", int(L26.record.str.startswith("**miss").sum()))
md = ["| Rusher (class, pick, team) | Role | Availability (games missed) | 40 / first step pct | "
      "Pressure vs forecast (rushes) | 2024 call scored on 2025 |",
      "|---|---|---|---|---|---|"]
L26 = L26.sort_values(["act", "edge_pts"], ascending=[False, False])
for _, r_ in L26.iterrows():
    key = r_.key
    nm_ = f"**{r_.display_name}**" if r_.act else r_.display_name
    flag = f"flag ({r_.missed}/{r_.possible})" if r_.avail >= 0.25 else f"ok ({r_.missed}/{r_.possible})"
    md.append(f"| {nm_} ({r_.draft_year}, {pick_txt(r_.pick)}, {r_.team}) | {r_.scheme} | {flag} | "
              f"{int(round(r_.t_pct))} / {min(int(round(r_.s_pct)), 100)} | "
              f"{{{{l26_{key}_rate}}}}% vs {{{{l26_{key}_fc}}}}% ({int(r_.rs)}) | {r_.record} |")
TABLES["md"]["list26"] = "\n".join(md)
L26.to_csv(OUT / "shortlist_2026.csv", index=False)
print(L26[["display_name", "draft_year", "team", "rs", "edge_pts", "avail", "record", "act"]].to_string())

# ---- (3) pre-registered 2026 test with power
# Arms: the 2026 list vs every other pool rusher with 100+ 2026 rushes; hit = 2026 pressure rate above a
# persistence forecast (own 2023-25 pressure shrunk to the pick/weight/role forecast). One-sided Fisher p < 0.05.
n_list = len(L26)
n_rest = int(NUM["x_pool"]) - int(L26.nfl_id.isin(pay_set.nfl_id).sum())
pa_, pb_ = int(NUM["tf_ks"]) / int(NUM["tf_ns"]), int(NUM["tf_ko"]) / int(NUM["tf_no"])
PR2 = np.random.default_rng(11)


def power2(n1, n2, a, b, nsim=3000):
    x_ = PR2.binomial(n1, a, nsim); y_ = PR2.binomial(n2, b, nsim)
    cache = {}
    hits = 0
    for xx, yy_ in zip(x_, y_):
        k_ = (int(xx), int(yy_))
        if k_ not in cache:
            cache[k_] = fisher1(k_[0], n1, k_[1], n2) < 0.05
        hits += cache[k_]
    return hits / nsim


put("pre_nlist", n_list); put("pre_nrest", n_rest)
put("pre_pa", 100 * pa_, 0); put("pre_pb", 100 * pb_, 0)
put("pre_pow_obs", 100 * power2(n_list, n_rest, pa_, pb_), 0)
mid2, half2 = (pa_ + pb_) / 2, (pa_ - pb_) / 4
put("pre_pow_half", 100 * power2(n_list, n_rest, mid2 + half2, mid2 - half2), 0)
put("pre_half_pa", 100 * (mid2 + half2), 0); put("pre_half_pb", 100 * (mid2 - half2), 0)
FROZEN = {"rule": RULE, "date_frozen": "2026-10-24", "data_used": "2023-2025 regular seasons only; no 2026 data on disk",
          "list_2026": sorted(L26.display_name.tolist()), "act_on": sorted(L26[L26.act].display_name.tolist()),
          "test": "2026 season, rushers with 100+ 2026 rushes; hit = 2026 pressure rate above persistence forecast "
                  "(own 2023-25 pressure shrunk to pick/weight/role forecast); list vs rest of pool; one-sided "
                  "Fisher exact p < 0.05; secondary: frozen first-step slope with persistence offset, one-sided p < 0.05",
          "power": {"at_time_forward_rates": NUM["pre_pow_obs"], "at_half_gap": NUM["pre_pow_half"]},
          "matched_pairs_2026": "each list member vs nearest-pick non-list rusher of same class and role; "
                                "share of pairs with higher 2026 pressure rate > 50%"}
blob = json.dumps(FROZEN, sort_keys=True).encode()
RULE_HASH = hashlib.sha256(blob).hexdigest()
json.dump({**FROZEN, "sha256": RULE_HASH}, open(OUT / "frozen_rule.json", "w"), indent=1)
NUM["rule_hash"] = RULE_HASH[:12]

# ============================================================================ round-17 additions
# (1) Baselines without tracking, held out by class: R^2, RMSE (pressure-rate pts, rush-weighted), rank correlation.
# (2) Continuous increments as delta R^2 with class-stratified bootstrap intervals (forecasts refit per fold).
# (3) Hit rates vs base rate with Jeffreys intervals; what 2026 result would change the call.
tick("round-17")
FC = FC.merge(SL[["nfl_id", "short"]], on="nfl_id", how="left")
FC["both_d"] = FC.short.fillna(False).astype(float)


def r2_parts(d, fcol):
    ybar_ = {yr: np.average(d[d.draft_year != yr].prate, weights=d[d.draft_year != yr].rs) for yr in CLASSES}
    yb = d.draft_year.map(ybar_)
    return np.sum(d.rs * (d.prate - d[fcol]) ** 2), np.sum(d.rs * (d.prate - yb) ** 2)


def r2_of(d, fcol):
    a_, b_ = r2_parts(d, fcol)
    return 1 - a_ / b_


def dr2_boot(d, f0, f1, nb=NBOOT):
    pt = r2_of(d, f1) - r2_of(d, f0)
    out = []
    for _ in range(nb):
        parts = [g_.iloc[RNG.integers(0, len(g_), len(g_))] for _, g_ in d.groupby("draft_year")]
        b_ = pd.concat(parts)
        out.append(r2_of(b_, f1) - r2_of(b_, f0))
    return pt, np.percentile(out, 5), np.percentile(out, 95)


def r2_boot(d, f0, nb=NBOOT):
    out = []
    for _ in range(nb):
        parts = [g_.iloc[RNG.integers(0, len(g_), len(g_))] for _, g_ in d.groupby("draft_year")]
        out.append(r2_of(pd.concat(parts), f0))
    return np.percentile(out, [5, 95])


def base_metrics(d, fcol, tag):
    put(f"bl_{tag}_r2", r2_of(d, fcol), 2)
    lo_, hi_ = r2_boot(d, fcol)
    put(f"bl_{tag}_r2_lo", lo_, 2); put(f"bl_{tag}_r2_hi", hi_, 2)
    put(f"bl_{tag}_rmse", 100 * np.sqrt(np.average((d.prate - d[fcol]) ** 2, weights=d.rs)), 1)
    put(f"bl_{tag}_rho", stats.spearmanr(d[fcol], d.prate)[0], 2)
    put(f"bl_{tag}_n", len(d))


FC["f_nfl"], _ = forecast_oof(FC, ZB + ["sfs_adj"], "ols")
FC["f_both"], _ = forecast_oof(FC, ZB + ["both_d"], "ols")
FC["f_full_nfl"], _ = forecast_oof(FC, ZF + ["sfs_adj"], "ols")
FC["f_full_both"], _ = forecast_oof(FC, ZF + ["both_d"], "ols")
base_metrics(FC, "f_ols", "pick")
base_metrics(FC, "f_full", "full")
put("bl_mean_rmse", 100 * np.sqrt(np.average((FC.prate - FC.draft_year.map(
    {yr: np.average(FC[FC.draft_year != yr].prate, weights=FC[FC.draft_year != yr].rs) for yr in CLASSES})) ** 2,
    weights=FC.rs)), 1)
for f0, f1, tag in [("f_ols", "f_pick40", "d40"), ("f_ols", "f_nfl", "dnfl"), ("f_ols", "f_both", "dboth"),
                    ("f_full", "f_full_nfl", "dfnfl"), ("f_full", "f_full_both", "dfboth")]:
    pt, lo_, hi_ = dr2_boot(FC, f0, f1)
    put(f"{tag}_r2", pt, 2); put(f"{tag}_lo", lo_, 2); put(f"{tag}_hi", hi_, 2)
# tracked 40 start: only rushers with a usable tracked 40 (subset), same comparison
FCc = FC.dropna(subset=["cfs_adj"]).copy()
FCc["f_ols_c"], _ = forecast_oof(FCc, ZB, "ols")
FCc["f_c40"], _ = forecast_oof(FCc, ZB + ["cfs_adj"], "ols")
FCc["f_carry"], _ = forecast_oof(FCc, ZB + ["carry"], "ols")
for f1, tag in [("f_c40", "dc40"), ("f_carry", "dcarry")]:
    pt, lo_, hi_ = dr2_boot(FCc, "f_ols_c", f1)
    put(f"{tag}_r2", pt, 2); put(f"{tag}_lo", lo_, 2); put(f"{tag}_hi", hi_, 2)
put("dc40_n", len(FCc))
print("baselines/delta R2", {k: NUM[k] for k in NUM if k.startswith(("bl_", "d40", "dnfl", "dboth", "dfnfl",
                                                                       "dfboth", "dc40", "dcarry"))})


def jeff(k, n):
    return 100 * stats.beta.ppf(0.05, k + 0.5, n - k + 0.5), 100 * stats.beta.ppf(0.95, k + 0.5, n - k + 0.5)


for tag, k_, n_ in [("js", int(NUM["sl_ks"]), int(NUM["sl_ns"])), ("jo", int(NUM["sl_ko"]), int(NUM["sl_no"])),
                    ("jb", int(NUM["beat_k"]), int(NUM["sl_ns"]) + int(NUM["sl_no"])),
                    ("jtfs", int(NUM["tf_ks"]), int(NUM["tf_ns"])), ("jtfo", int(NUM["tf_ko"]), int(NUM["tf_no"]))]:
    lo_, hi_ = jeff(k_, n_)
    put(f"{tag}_pct", 100 * k_ / n_, 0); put(f"{tag}_lo", lo_, 0); put(f"{tag}_hi", hi_, 0)
# what would change the call: chance that the 2026 list does no better than the rest (share of hits <= rest's)
PR3 = np.random.default_rng(17)


def p_nobetter(a, b, nsim=20000):
    x_ = PR3.binomial(n_list, a, nsim) / n_list; y_ = PR3.binomial(n_rest, b, nsim) / n_rest
    return 100 * np.mean(x_ <= y_)


put("nb_obs", p_nobetter(pa_, pb_), 0)
put("nb_half", p_nobetter(mid2 + half2, mid2 - half2), 0)
put("nb_null", p_nobetter(mid2, mid2), 0)
# the smallest number of 2026 list hits (of 11) that would pass the pre-registered test if the rest hit at 38%
k_rest = int(round(pb_ * n_rest))
need = next((k_ for k_ in range(n_list + 1) if fisher1(k_, n_list, k_rest, n_rest) < 0.05), None)
put("need_k", need); put("need_rest_k", k_rest)

# ============================================================================ figures
tick("figures")
TT = np.arange(TW) / 10
gmed = np.median(np.vstack(gd.curve.to_numpy()), axis=0)
gq25, gq75 = np.percentile(np.vstack(gd.curve.to_numpy()), [25, 75], axis=0)
# ---- Fig 1: four rushers, 2x2, one scale, readouts in the panel
FOUR = [("Byron Young", "Both"), ("Cedric Johnson", "Both"), ("Maason Smith", "Start only"), ("Eric Watts", "Speed only")]
CCOL = {"Both": OI["blue"], "Start only": OI["purple"], "Speed only": OI["red"], "Neither": OI["grey"]}
fig, axes = plt.subplots(2, 2, figsize=(FIGW, 9.4), sharex=True, sharey=True)
fig.subplots_adjust(left=0.075, right=0.985, top=0.775, bottom=0.085, hspace=0.42, wspace=0.08)
for ax, (nm, lab) in zip(axes.ravel(), FOUR):
    r_ = board[board.display_name == nm].iloc[0]
    ax.axvspan(0.3, 0.5, color=OI["yellow"], alpha=0.35, lw=0)
    sn_ = gd[gd.nfl_id == r_.nfl_id]
    for cv in sn_.curve:
        ax.plot(TT, cv, color=OI["blue"], alpha=0.15, lw=0.8)
    ax.plot(TT, np.mean(np.vstack(sn_.curve.to_numpy()), 0), color=OI["blue"], lw=3)
    reps = d40[d40.nfl_id == r_.nfl_id]
    if len(reps):
        ax.plot(TT, np.mean(np.vstack(reps.curve.to_numpy()), 0), color=OI["orange"], lw=3, ls="--")
    ax.fill_between(TT, gq25, gq75, color=OI["grey"], alpha=0.22, lw=0)
    ax.plot(TT, gmed, color=OI["black"], lw=1.6, ls=":")
    ax.set_xlim(0, 1.5); ax.set_ylim(0, 8.6); ax.set_yticks([0, 2, 4, 6, 8]); ax.set_xticks([0, 0.5, 1.0, 1.5])
    t_txt = ordn(r_.t_pct) if pd.notna(r_.t_pct) else "none"
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"pick {int(r_.pick)}"
    ax.set_title(f"{nm} ({r_.draft_year}, {pk}): {lab}", loc="left", fontsize=15, color=CCOL[lab],
                 fontweight="bold")
    ax.text(0.03, 0.97, f"40 time: {t_txt} pct\nNFL first step: {ordn(r_.s_pct)} pct\n"
            f"Pressure {100 * r_.prate:.1f}% vs {100 * r_.forecast:.1f}% forecast",
            transform=ax.transAxes, va="top", fontsize=13, color="#222222",
            bbox=dict(facecolor="white", edgecolor="#BBBBBB", pad=4))
for ax in axes[:, 0]:
    ax.set_ylabel("speed, yd/s")
for ax in axes[1]:
    ax.set_xlabel("seconds from first movement")
fig.suptitle(f"His Combine 40 start (orange) does not predict his NFL starts (blue): ρ = {NUM['tr_fs']} after size.\n"
             f"Fast on both the stopwatch and Sundays: {NUM['sl_ks']} of {NUM['sl_ns']} beat forecast "
             f"(rest {NUM['sl_ko']} of {NUM['sl_no']})", x=0.075, ha="left",
             fontsize=16, y=0.985)
fig.text(0.075, 0.855, f"Yellow = first step (0.3-0.5 s). Dotted line and grey band = median and middle half of all NFL lineman starts.\nPercentiles: after weight and role, "
         f"in the {NUM['n_pay']}-rusher pool.", fontsize=13, color="#333333")
save(fig, "fig1_four_rushers.png")

# ---- Fig 2: (A) translation and payoff tenth by tenth; (B) one clean held-out test, per class
fig = plt.figure(figsize=(FIGW, 13.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.62], wspace=0.30, hspace=0.42,
                      left=0.075, right=0.975, top=0.865, bottom=0.06)
gs2 = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.62], hspace=0.42, left=0.14, right=0.975, top=0.865, bottom=0.06)
ax = fig.add_subplot(gs[0, 0])
ts = TT[3:]
LABPOS = {"raw": (0.95, 0.66), "adj": (1.02, -0.3), "pay": (0.62, 0.52)}
for series, color, label, ls, fill in [("raw", OI["black"], "40 → NFL speed, raw\n(mostly size)", "--", False),
                                       ("adj", OI["orange"], "40 → NFL speed,\nafter weight, role", "-", True),
                                       ("pay", OI["blue"], "NFL speed → pressure,\nbeyond pick", "-", True)]:
    rr = np.array([c[0] for c in CUR[series][3:]])
    if fill:
        ax.fill_between(ts, [c[1] for c in CUR[series][3:]], [c[2] for c in CUR[series][3:]], color=color, alpha=0.15, lw=0)
    ax.plot(ts, rr, color=color, ls=ls, lw=2.4)
    ax.text(LABPOS[series][0], LABPOS[series][1], label, color=color, va="center", fontsize=12.5)
ax.axhline(0, color="black", lw=0.8); ax.axvspan(0.3, 0.5, color=OI["yellow"], alpha=0.3, lw=0)
ax.set_xlim(0.3, 1.5); ax.set_ylim(-0.45, 0.8)
ax.set_xlabel("seconds from first movement"); ax.set_ylabel("rank correlation (90% band)")
ax.set_title(f"A. Tenth by tenth (n = {NUM['n_tr']} / {NUM['n_pay']})", loc="left", fontsize=14)
bx = fig.add_subplot(gs[0, 1])
ROWS = [("Tracked 40 start", HT["c40"], OI["orange"]), ("Carry-over (NFL − 40)", HT["carry"], OI["purple"]),
        ("Official 40 time", RT["f40_pick"], OI["sky"]), ("NFL first step", HT["nfl"], OI["blue"]),
        ("NFL first step, + 40 & NGS", RT["nfl_full"], OI["green"])]
yy, yl, yt = 0.0, [], []
for lab, H, col in ROWS:
    bx.text(-2.45, yy - 0.8, lab,
            fontsize=12, color=col, va="center", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    for i_, yr in enumerate(CLASSES):
        b_, s_ = H["b"][i_] * DPP, H["se"][i_] * DPP
        bx.plot([b_ - Z90 * s_, b_ + Z90 * s_], [yy, yy], color=col, lw=2.2)
        bx.plot(b_, yy, "o", ms=7.5, color="white" if yr == 2025 else col, mec=col, mew=1.8)
        yl.append(str(yr)); yt.append(yy); yy += 0.85
    nlo, nhi = H["null"]
    bx.fill_betweenx([yy - 0.4, yy + 0.4], nlo, nhi, color=OI["grey"], alpha=0.35, lw=0)
    b_, s_ = H["mu"] * DPP, H["s"] * DPP
    bx.plot([b_ - Z90 * s_, b_ + Z90 * s_], [yy, yy], color=col, lw=4.5)
    bx.plot(b_, yy, "D", ms=9, color=col, mec="black")
    yl.append("pooled"); yt.append(yy)
    yy += 2.1
bx.axvline(0, color="black", lw=1)
bx.set_yticks(yt); bx.set_yticklabels(yl, fontsize=12)
bx.set_ylim(yy - 1.6, -1.7); bx.set_xlim(-2.5, 2.6)
bx.set_xlabel("pressure-rate pts per sd vs forecast (90%)")
bx.set_title("B. Each class held out in turn", loc="left", fontsize=14)
cx = fig.add_subplot(gs2[1, 0])
GORD = ["DL", "WR", "OL", "DB", "TE"]
GR_ = {r_["grp"]: dict(r_) for r_ in GROWS}
# DL row = the main analysis numbers (pass-rush snaps only, same pool as the text)
GR_["DL"].update(tr_adj=float(NUM["tr_fs"]), tr_lo=float(NUM["tr_fs_lo"]), tr_hi=float(NUM["tr_fs_hi"]),
                 rel=float(NUM["rel_fs_game"]), n_tr=int(NUM["n_tr"]))
for i_, g in enumerate(GORD):
    r_ = GR_[g]
    weak = r_["rel"] < 0.3
    for off_, (k_, lo_k, hi_k, col) in enumerate([("tr_adj", "tr_lo", "tr_hi", OI["orange"]),
                                                  ("pay", "pay_lo", "pay_hi", OI["blue"])]):
        c_ = "#BBBBBB" if weak else col
        y_ = i_ + (off_ - 0.5) * 0.34
        cx.plot([r_[lo_k], r_[hi_k]], [y_, y_], color=c_, lw=3, solid_capstyle="butt")
        cx.plot(r_[k_], y_, "o", ms=9, color=c_)
    cx.text(0.58, i_, f"NFL first-step reliability {r_['rel']:.2f}" + (" (too noisy)" if weak else ""),
            va="center", fontsize=12.5, color="#888888" if weak else "#222222")
cx.axvline(0, color="black", lw=1)
cx.set_yticks(range(len(GORD))); cx.set_yticklabels([f"{g} (n = {GR_[g]['n_tr']})" for g in GORD], fontsize=13)
cx.set_ylim(len(GORD) - 0.45, -0.6); cx.set_xlim(-0.65, 1.2); cx.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
cx.set_xlabel("partial rank correlation, 90% interval")
cx.set_title("C. Five groups screened. Orange: 40 start → NFL start, after size.\n"
             "Blue: NFL start → production beyond pick", loc="left", fontsize=13.5)
fig.suptitle("The 40 start does not carry over; the NFL first step adds a small edge that varies by class",
             x=0.075, ha="left", fontsize=17)
fig.text(0.075, 0.9, "B: hollow = 2025 class; diamond = pooled over classes; grey band = 90% of pooled slopes\n"
         "when the measure is shuffled within class, i.e. what no signal looks like. C: grey = reliability below 0.3.",
         fontsize=12.5, color="#333333")
save(fig, "fig2_translation_heldout.png")

# ---- Fig 3: the shortlist (held-out call) and its hit rate against the base rate
fig = plt.figure(figsize=(FIGW, 8.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.14, left=0.075, right=0.975, top=0.79, bottom=0.2)
ax = fig.add_subplot(gs[0, 0])
S3 = SL.dropna(subset=["f40_adj"]).copy()
S3["hit"] = S3.prate > S3.forecast
# each rusher is placed relative to his own class's held-out cut-offs, so the shortlist is exactly the top-right
S3["f40_adj"] = S3.f40_adj - S3.draft_year.map({y: TH[y][2] for y in CLASSES})
S3["sfs_adj"] = S3.sfs_adj - S3.draft_year.map({y: TH[y][0] for y in CLASSES})
ax.add_patch(plt.Rectangle((0, 0), 4, 4, color=OI["sky"], alpha=0.22, lw=0))
ax.axhline(0, color=OI["grey"], lw=1, ls="--"); ax.axvline(0, color=OI["grey"], lw=1, ls="--")
for hit, mk in [(True, dict(color=OI["blue"], mec="black", mew=0.6)),
                (False, dict(color="white", mec=OI["red"], mew=1.8))]:
    g_ = S3[S3.hit == hit]
    ax.plot(g_.f40_adj, g_.sfs_adj, "o", ms=9, **mk, label="beat forecast" if hit else "below forecast", ls="none")
# label positions in data coordinates (text anchor), with a thin leader line to the point
LBL = {"Maason Smith": (-0.25, 2.35, "right"), "Chop Robinson": (1.75, 2.0, "left"),
       "Byron Young": (1.75, 1.55, "left"), "YaYa Diaby": (1.75, 1.1, "left"), "Ruke Orhorhoro": (1.0, 2.4, "left"),
       "Cedric Johnson": (-0.25, 1.55, "right"), "Jordan Burch": (1.75, 0.75, "left"),
       "Adetomiwa Adebawore": (2.75, -0.45, "right"), "Laiatu Latu": (1.1, -1.45, "left"),
       "Eric Watts": (1.1, -3.45, "left")}
SHORT = {"Adetomiwa Adebawore": "Adebawore"}
for nm, (tx, ty, ha) in LBL.items():
    r_ = S3[S3.display_name == nm]
    if len(r_):
        r_ = r_.iloc[0]
        if tx is None:
            tx, ty = r_.f40_adj - 0.3, r_.sfs_adj
        ax.annotate(SHORT.get(nm, nm), (r_.f40_adj, r_.sfs_adj), xytext=(tx, ty), textcoords="data", fontsize=12.5,
                    ha=ha, va="center", arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8, shrinkB=5))
ax.text(2.85, -2.75, f"Both: {NUM['sl_ks']} of {NUM['sl_ns']}\nbeat forecast", ha="right", va="bottom", fontsize=13,
        color=OI["blue"], fontweight="bold")
ax.set_xlim(-2.6, 2.9); ax.set_ylim(min(-3.4, S3.sfs_adj.min() - 0.3), 2.7)
ax.set_xlabel("official 40 time vs cut-off (sd faster)")
ax.set_ylabel("NFL first step vs cut-off (sd)")
ax.set_title(f"A. {len(S3)} rushers with a 40, after weight and role.\nCut-offs (dashed) set on the other two classes", loc="left", fontsize=14)
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.15), ncol=2, fontsize=13, frameon=False, handletextpad=0.2)
bx = fig.add_subplot(gs[0, 1])
ROWS = [("rb_main", "1. Main"), ("rb_full", "2. Forecast knows 40 + NGS"), ("rb_x", "3. Sampled snaps removed"),
        ("rb_d3", "4. Day-3 / UDFA only"), ("rb_tf", "5. Frozen 2024, scored 2025 vs own past")]
R_ = {r["tag"]: r for r in ROB}
for i_, (tag, lab) in enumerate(ROWS):
    r = R_[tag]; y0 = -i_
    lo_s, hi_s = [v_ / 100 for v_ in jeff(r["ks"], r["ns"])]; lo_o, hi_o = [v_ / 100 for v_ in jeff(r["ko"], r["no"])]
    bx.plot([100 * lo_o, 100 * hi_o], [y0 - 0.15] * 2, color=OI["grey"], lw=3, solid_capstyle="butt")
    bx.plot(100 * r["ko"] / r["no"], y0 - 0.15, "o", color=OI["grey"], ms=9)
    bx.plot([100 * lo_s, 100 * hi_s], [y0 + 0.15] * 2, color=OI["blue"], lw=3, solid_capstyle="butt")
    bx.plot(100 * r["ks"] / r["ns"], y0 + 0.15, "o", color=OI["blue"], ms=10)
    bx.plot([100 * r["base"]] * 2, [y0 - 0.3, y0 + 0.28], color="black", lw=1.4)
    bx.text(1, y0 + 0.45, f"{lab}  (gap {NUM[tag + '_gap']} pts)", ha="left", va="center", fontsize=13,
            fontweight="bold")
    bx.text(101, y0 + 0.15, f"{r['ks']}/{r['ns']}", va="center", fontsize=12, color=OI["blue"])
    bx.text(101, y0 - 0.15, f"{r['ko']}/{r['no']}", va="center", fontsize=12, color=OI["grey"])
bx.set_xlim(0, 113); bx.set_ylim(-len(ROWS) + 0.55, 0.7); bx.set_xticks([0, 25, 50, 75, 100])
bx.set_yticks([]); bx.spines["left"].set_visible(False)
bx.set_xlabel("% of rushers beating forecast (90% Jeffreys)")
bx.set_title("B. Shortlist (blue) vs rest of pool (grey);\nblack tick = base rate", loc="left", fontsize=14)
fig.suptitle(f"Fast on the stopwatch and in NFL starts: {NUM['sl_ks']} of {NUM['sl_ns']} beat their held-out\npick/weight/role forecast. "
             "The edge narrows in every stricter check; unproven for day-3 picks", x=0.075, ha="left", fontsize=16)
save(fig, "fig3_shortlist.png")

# ---- Fig 4: snap by snap, within rusher
fig = plt.figure(figsize=(FIGW, 6.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.42, left=0.08, right=0.975, top=0.76, bottom=0.2)
ax = fig.add_subplot(gs[0, 0])
for i_, r_ in FIFTH.iterrows():
    lo, hi = cp(int(r_["sum"]), int(r_["size"]))
    ax.plot([i_, i_], [100 * lo, 100 * hi], color=OI["blue"], lw=3)
    ax.plot(i_, 100 * r_["sum"] / r_["size"], "o", ms=11, color=OI["blue"])
    ax.text(i_ + 0.1, 100 * hi + 0.4, f"{int(r_['sum'])}/{int(r_['size'])}", fontsize=11.5, ha="center")
ax.axhline(100 * SN.press.mean(), color=OI["grey"], ls=":", lw=1.4)
ax.set_xticks(range(5)); ax.set_xticklabels(["slowest\nfifth", "", "", "", "fastest\nfifth"])
ax.set_ylim(0, 22); ax.set_ylabel("% of sampled snaps with a pressure")
ax.set_xlabel("first step on that snap vs his own average")
ax.set_title("A. Same rusher, faster snap → more pressure", loc="left", fontsize=14)
bx = fig.add_subplot(gs[0, 1])
RO = [("All snaps", m0.params.fs_wz, m0.bse.fs_wz), ("With controls*", m1.params.fs_wz, m1.bse.fs_wz)]
RO += [(f"{r_['yr']} class", r_["b"], r_["se"]) for r_ in SNC]
for role in ["EDGE", "interior"]:
    m_ = clog("press ~ fs_wz", SN[SN.role == role])
    RO.append((f"{role}", m_.params.fs_wz, m_.bse.fs_wz))
for i_, (lab, b_, s_) in enumerate(RO):
    bx.plot(np.exp([b_ - Z90 * s_, b_ + Z90 * s_]), [i_, i_], color=OI["green"], lw=3)
    bx.plot(np.exp(b_), i_, "o", ms=10, color=OI["green"] if "2025" not in lab else "white", mec=OI["green"], mew=2)
bx.axvline(1, color="black", lw=1); bx.set_xscale("log"); bx.set_xlim(0.7, 3.2)
bx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); bx.set_xticks([0.8, 1, 1.5, 2, 3])
bx.set_xticklabels(["0.8", "1", "1.5", "2", "3"])
bx.set_yticks(range(len(RO))); bx.set_yticklabels([r[0] for r in RO], fontsize=12); bx.set_ylim(len(RO) - 0.5, -0.5)
bx.set_xlabel("odds ratio per sd of snap first step (90%)")
bx.set_title("B. Holds across controls, classes, roles", loc="left", fontsize=14)
fig.suptitle(f"Snap by snap: {NUM['sn_n']} random NFL pass rushes by {NUM['sn_players']} linemen, {NUM['sn_k']} pressures",
             x=0.08, ha="left", fontsize=16)
fig.text(0.08, 0.83, "Each rusher compared with himself; errors clustered by rusher; hollow = 2025.\n*Controls: that "
         "snap's NGS get-off, sideways share, passing down, alignment.", fontsize=12, color="#333333")
save(fig, "fig5_snap_mechanism.png")


# ---- Fig 4: every pick-matched pair, career (A) and frozen 2024 -> 2025 (B)
fig = plt.figure(figsize=(FIGW, 11.0))
gs = fig.add_gridspec(2, 1, height_ratios=[len(PAIRS), len(PAIRS_TF)], hspace=0.32, left=0.38, right=0.97,
                      top=0.815, bottom=0.07)
for k_, (P_, tag, ttl) in enumerate([(PAIRS, "mp", "A. Career, held-out call"),
                                       (PAIRS_TF, "mpf", "B. Frozen 2024 call, 2025 vs twin")]):
    ax = fig.add_subplot(gs[k_, 0])
    P_ = P_.sort_values("pick").reset_index(drop=True)
    for i_, r_ in P_.iterrows():
        col = OI["blue"] if r_.win else OI["red"]
        ax.plot([r_.ty, r_.y], [i_, i_], color=col, lw=2.6, alpha=0.8)
        ax.plot(r_.ty, i_, "o", ms=10, color="white", mec=OI["grey"], mew=2)
        ax.plot(r_.y, i_, "o" if r_.win else "X", ms=11, color=col, mec="black", mew=0.6)
    lab = [f"{r_['name']} ({pick_txt(r_.pick)}) v {r_.twin.split()[-1]} ({pick_txt(r_.tpick)})"
           for _, r_ in P_.iterrows()]
    ax.set_yticks(range(len(P_))); ax.set_yticklabels(lab, fontsize=13); ax.set_ylim(len(P_) - 0.5, -0.6)
    ax.set_xlim(0, 16); ax.grid(axis="x", color="#DDDDDD", lw=0.8)
    if k_ == 1:
        ax.set_xlabel("pressure rate, % (draft pick in brackets)")
    ax.set_title(f"{ttl}: {NUM[tag + '_w']} of {NUM[tag + '_n']} won, mean +{NUM[tag + '_diff']} pts "
                 f"[{NUM[tag + '_diff_lo']}, {NUM[tag + '_diff_hi']}]", loc="left", fontsize=14.5, x=-0.55)
fig.suptitle("Every Both rusher vs his draft-slot twin (same class and role, nearest pick)", x=0.02, ha="left",
             fontsize=17, y=0.975)
fig.text(0.02, 0.865, "Filled = Both rusher (blue dot: out-rushed his twin; vermillion X: did not). "
         "Hollow grey = twin; twins can repeat.\nB: twins need 100+ 2025 rushes, so some differ from A.\nAgainst their own past "
         f"pressure (Fig 3B, row 5) the same {NUM['tf_ns']} calls hit {NUM['tf_ks']} times.", fontsize=13, color="#333333")
save(fig, "fig4_matched_pairs.png")

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
