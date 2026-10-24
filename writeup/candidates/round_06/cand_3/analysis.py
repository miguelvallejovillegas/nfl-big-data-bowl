"""Get-Off Translation: how much of his Combine 40 start a defensive lineman brings to an NFL snap.

Round 6, candidate 3. Run from the repo root:
    uv run python writeup/candidates/round_06/cand_3/analysis.py

Reads data/processed/*.parquet only (about one minute). Writes numbers.json, board.csv and figures/*.png next to
this file.

Movement profile = smoothed speed curve over the first 1.5 s of a start, built by one function on Combine clips
(combine_frames) and NFL snaps (game_frames_sample). The headline summary is yards covered in the first 1.0 s.
  Combine side: the matched drill rep, aligned to movement onset (the 40 has no snap), year-adjusted.
  NFL side: random-sample pass plays only, aligned to the ball snap, player at rest at the snap, alignment-adjusted.
Matched movement per group (screen):
  DL  40-yard-dash start (3-point stance, straight ahead) <-> NFL pass rush
  WR  40-yard-dash start                                   <-> NFL route release
  TE  40-yard-dash start                                   <-> NFL route release
  OL  pass-rush-drop (kick-slide) drill                    <-> NFL pass set
  DB  four back-pedal drills                               <-> NFL coverage start
Get-Off Translation (yards) = NFL yards in 1.0 s - yards predicted from his Combine yards in 1.0 s (within-group OLS).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2027)
NBOOT = 2000
NPERM = 2000
T = 16          # frames in the profile (0-1.5 s)
W = 10          # headline window: first 1.0 s
MIN_GAME = 8
UDFA_PICK = 262

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 150,
                     "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10})
FIGW = 1600 / 150
NUM: dict[str, str] = {}


def put(key, val, nd=2):
    if key in NUM:
        raise KeyError(f"duplicate key {key}")
    if isinstance(val, (int, np.integer)):
        s = str(int(val))
    else:
        s = f"{float(val):.{nd}f}"
        if s.startswith("-") and float(s) == 0:
            s = s[1:]
    NUM[key] = s
    return s


def save(fig, name):
    fig.set_size_inches(FIGW, fig.get_figheight())
    fig.savefig(FIG / name, dpi=150, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- data
pl = pd.read_parquet(P / "players.parquet")
cr = pd.read_parquet(P / "combine_results.parquet")
ev = pd.read_parquet(P / "combine_events.parquet")
cf = pd.read_parquet(P / "combine_frames.parquet", columns=["event_id", "frame", "entity_type", "s"])
cf = cf[cf.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
gp = pd.read_parquet(P / "game_plays_sample.parquet")
gf = pd.read_parquet(P / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "s"])
pv = pd.read_parquet(P / "player_value.parquet")
al = pd.read_parquet(P / "player_season_alignment.parquet")
ss = pd.read_parquet(P / "player_season_stats.parquet")

GROUPS = ["DL", "WR", "TE", "OL", "DB"]
DRILLS = {"DL": ["FORTY_YARD_DASH"], "WR": ["FORTY_YARD_DASH"], "TE": ["FORTY_YARD_DASH"], "OL": ["PASS_RUSH_DROP"],
          "DB": ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL",
                 "LINE_DRILL", "W_DRILL"]}
MOVE = {"DL": "40 start vs pass rush", "WR": "40 start vs release", "TE": "40 start vs release",
        "OL": "kick-slide vs pass set", "DB": "back-pedal vs coverage start"}


def drill_curve(s):
    """Onset-aligned profile: player near rest, first frame >= 0.8 yd/s, backed up 0.2 s."""
    s = np.asarray(s, float)
    if len(s) < T + 2:
        return None
    sm = savgol_filter(s, 5, 2)
    rest = np.flatnonzero(sm < 0.8)
    if not len(rest) or rest[0] > 20:
        return None
    hit = np.flatnonzero(sm[rest[0]:] >= 0.8)
    if not len(hit):
        return None
    o = rest[0] + max(int(hit[0]) - 2, 0)
    if o + T > len(sm):
        return None
    return np.clip(sm[o:o + T], 0, None)


def game_curve(s, sf):
    """Snap-aligned profile: player at rest at the snap (< 1 yd/s), next 1.5 s."""
    s = np.asarray(s, float)
    if len(s) < sf + T or len(s) < 7:
        return None
    sm = savgol_filter(s, 5, 2)
    if sm[sf] > 1.0:
        return None
    return np.clip(sm[sf:sf + T], 0, None)


V = [f"v{i}" for i in range(T)]
# Combine reps
evx = ev.set_index("event_id")
want = {(g, d) for g, ds in DRILLS.items() for d in ds}
keep = ev[[(g, d) in want for g, d in zip(ev.position_group, ev.drill_name)]].event_id
rows = []
for eid, r in cf[cf.event_id.isin(set(keep))].groupby("event_id", sort=False):
    c = drill_curve(r.s.values)
    if c is None:
        continue
    e = evx.loc[eid]
    rows.append((e.nfl_id, e.position_group, e.drill_name, int(e.draft_year), int(e.attempt_no), *c))
DR = pd.DataFrame(rows, columns=["nfl_id", "grp", "drill", "yr", "att", *V])


def yards(df, w=W):
    return df[V[:w]].sum(axis=1) * 0.1


for w_ in (5, 10, 15):
    DR[f"y{w_}"] = yards(DR, w_)
    DR[f"y{w_}"] = DR[f"y{w_}"] / DR.groupby(["drill", "yr"])[f"y{w_}"].transform("median") * \
        DR.groupby("drill")[f"y{w_}"].transform("median")

# pass-rush drill (comparison only)
prk = set(ev[ev.drill_name == "PASS_RUSH_DRILL"].event_id)
rows = []
for eid, r in cf[cf.event_id.isin(prk)].groupby("event_id", sort=False):
    c = drill_curve(r.s.values)
    if c is None:
        continue
    e = evx.loc[eid]
    rows.append((e.nfl_id, int(e.draft_year), c[:W].sum() * 0.1))
PRD = pd.DataFrame(rows, columns=["nfl_id", "yr", "y10"])
PRD["y10"] = PRD.y10 / PRD.groupby("yr").y10.transform("median") * PRD.y10.median()

# NFL plays: random half, pass plays; WR/TE must have run a route
g = gp[gp.pass_result.notna()].merge(pl[["nfl_id", "position_group"]], on="nfl_id")
g = g[~g.position_group.isin(["WR", "TE"]) | g.route_ran.notna()]
f = gf.merge(g[["game_id", "play_id", "nfl_id"]], on=["game_id", "play_id", "nfl_id"]).sort_values(
    ["nfl_id", "game_id", "play_id", "frame"])
meta = g.set_index(["nfl_id", "game_id", "play_id"])
rows = []
for k, r in f.groupby(["nfl_id", "game_id", "play_id"], sort=False):
    m = meta.loc[k]
    if not np.isfinite(m.snap_frame):
        continue
    c = game_curve(r.s.values, int(m.snap_frame))
    if c is None:
        continue
    rows.append((*k, m.position_group, m.sample_kind, m.lined_up_position, int(m.season),
                 bool(pd.notna(m.time_to_pressure)), m.player_get_off, *c))
GA = pd.DataFrame(rows, columns=["nfl_id", "game_id", "play_id", "grp", "kind", "align", "season", "press",
                                 "ngs_getoff", *V])
for w_ in (5, 10, 15):
    GA[f"y{w_}"] = yards(GA, w_)
    GA[f"y{w_}"] = GA[f"y{w_}"] - GA.groupby(["grp", "align"])[f"y{w_}"].transform("median") + \
        GA.groupby("grp")[f"y{w_}"].transform("median")
GR = GA[GA.kind == "random"].copy()
put("n_drill_reps", int(DR[DR.grp == "DL"].shape[0]))
put("n_nfl_plays_all", len(GR))

# role shares
sh = al.pivot_table(index="nfl_id", columns="lined_up_position", values="plays", aggfunc="sum").fillna(0)
sh = sh.div(sh.sum(axis=1), axis=0)
role = pd.DataFrame({"edge": sh["EDGE"], "tsh": sh["T"], "csh": sh["C"], "cb": sh["CB"],
                     "slot": sh["SLOT_CB"] + sh["SLOT_WR"]})
ROLE = {"OL": ["tsh", "csh"], "DL": ["edge"], "DB": ["cb", "slot"], "WR": ["slot"], "TE": ["slot"]}


def player_table(G, w=W, drill=None, min_game=MIN_GAME):
    drill = DR if drill is None else drill
    cnt = G.groupby("nfl_id").size()
    yg = G.groupby("nfl_id")[f"y{w}"].mean()[cnt >= min_game].rename("g")
    yd = drill.groupby("nfl_id")[f"y{w}"].mean().rename("d")
    Q = pd.concat([yg, yd], axis=1, join="inner")
    Q["n_game"] = cnt.reindex(Q.index)
    Q = Q.join(pl.set_index("nfl_id")[["display_name", "position_group", "draft_overall_pick", "draft_year",
                                       "combine_weight", "nfl_position"]]).join(role)
    Q = Q.join(cr.set_index("nfl_id")[["forty", "ten_yd_split", "ngs_athleticism_score"]])
    Q["lpick"] = np.log(Q.draft_overall_pick.fillna(UDFA_PICK))
    Q["tr"] = np.nan
    for grp, q in Q.groupby("position_group"):
        b = np.polyfit(q.d, q.g, 1)
        Q.loc[q.index, "tr"] = q.g - np.polyval(b, q.d)
    return Q


PT = player_table(GR)


# ----------------------------------------------------------------------------- stats helpers
def resid(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ b


def prho(y, x, Z):
    return spearmanr(resid(y, Z), resid(x, Z))[0]


def boot(fn, n, nb=NBOOT, strata=None):
    vals = []
    for _ in range(nb):
        if strata is None:
            idx = RNG.integers(0, n, n)
        else:
            idx = np.concatenate([RNG.choice(np.flatnonzero(strata == s), (strata == s).sum())
                                  for s in np.unique(strata)])
        try:
            vals.append(fn(idx))
        except Exception:
            pass
    return np.array(vals)


def outcome(grp):
    tgt = "opportunity" if grp == "TE" else "production"
    v = pv[(pv.position_group == grp) & (pv.target == tgt)][["nfl_id", "actual"]].set_index("nfl_id").actual
    return -v if grp == "OL" else v      # fewer pressures allowed = better


def design(grp, Q=None, metric="tr", extra=()):
    Q = PT if Q is None else Q
    cols = ["lpick", "combine_weight", *ROLE[grp], *extra]
    d = Q[Q.position_group == grp].join(outcome(grp).rename("y"), how="inner")
    d = d.dropna(subset=cols + [metric, "y"]).copy()
    Z = np.column_stack([d[cols].values, d.draft_year.values == 2023, d.draft_year.values == 2024]).astype(float)
    return d, Z


OUTLAB = {"DL": "pressure rate", "WR": "yards/route", "TE": "snaps/season", "OL": "pressure avoided",
          "DB": "solo tackles/100"}

# ----------------------------------------------------------------------------- 1. screen over five groups
screen = []
cache = {}
for grp in GROUPS:
    d, Z = design(grp)
    cache[grp] = (d, Z)
    r = prho(d.y.values, d.tr.values, Z)
    bs = boot(lambda ix: prho(d.y.values[ix], d.tr.values[ix], Z[ix]), len(d), nb=1000,
              strata=d.draft_year.values)
    screen.append(dict(grp=grp, n=len(d), rho=r, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5)))
SC = pd.DataFrame(screen).set_index("grp")
maxr = []
for _ in range(NPERM):
    best = 0
    for grp in GROUPS:
        d, Z = cache[grp]
        y = d.y.values.copy()
        for yr in (2023, 2024, 2025):
            ix = np.flatnonzero(d.draft_year.values == yr)
            y[ix] = y[RNG.permutation(ix)]
        best = max(best, abs(prho(y, d.tr.values, Z)))
    maxr.append(best)
maxr = np.array(maxr)
for grp in GROUPS:
    put(f"sc_{grp}_rho", SC.loc[grp, "rho"])
    put(f"sc_{grp}_n", int(SC.loc[grp, "n"]))
put("sc_fw_p", float((maxr >= abs(SC.loc["DL", "rho"])).mean()), 2)

# ----------------------------------------------------------------------------- 2. DL: validity and travel
DLall = PT[PT.position_group == "DL"].copy()
put("dl_n_profile", len(DLall))
gd = GR[(GR.grp == "DL") & GR.nfl_id.isin(DLall.index)].sort_values(["nfl_id", "game_id", "play_id"]).copy()
put("dl_n_rushes", len(gd))
put("dl_rush_med", int(gd.groupby("nfl_id").size().median()))
# per-play agreement with NGS get-off (official, full tracking)
pp = gd.dropna(subset=["ngs_getoff"])
put("val_play_rho", spearmanr(pp.y10, pp.ngs_getoff)[0])
put("val_play_n", len(pp))
# split-half reliability of the NFL half
gd["h"] = gd.groupby("nfl_id").cumcount() % 2
s2 = gd.groupby(["nfl_id", "h"]).y10.mean().unstack().dropna()
r_ = spearmanr(s2[0], s2[1])[0]
rel_g = 2 * r_ / (1 + r_)
put("rel_nfl", rel_g)
# drill rep-to-rep reliability (first two 40 reps), Spearman-Brown to the mean rep count
dd = DR[(DR.grp == "DL")].copy()
dd["k"] = dd.groupby("nfl_id").cumcount()
w2 = dd[dd.k < 2].pivot(index="nfl_id", columns="k", values="y10").dropna()
r1 = spearmanr(w2[0], w2[1])[0]
kbar = dd.groupby("nfl_id").size().reindex(DLall.index).mean()
rel_d = kbar * r1 / (1 + (kbar - 1) * r1)
put("rel_drill_rep", r1)
put("rel_drill_n", len(w2))
put("rel_drill", rel_d)
prd_rel = PRD.copy()
prd_rel["k"] = prd_rel.groupby("nfl_id").cumcount()
w3 = prd_rel[prd_rel.k < 2].pivot(index="nfl_id", columns="k", values="y10").dropna()
put("rel_prd_rep", spearmanr(w3[0], w3[1])[0])
# season NGS get-off (snap-weighted)
sea = ss[(ss.position_group == "DL") & (ss.pass_rush_snaps > 0) & ss.avg_get_off.notna()]
sgo = sea.groupby("nfl_id").apply(lambda q: np.average(q.avg_get_off, weights=q.pass_rush_snaps),
                                  include_groups=False).rename("sgo")
DLall = DLall.join(sgo)
q = DLall.dropna(subset=["sgo"])
put("val_season_rho", spearmanr(q.g, q.sgo)[0])
put("val_season_n", len(q))
# travel: Combine start -> NFL start
travel = {}
for lab, x, sign in [("40 start (tracked)", "d", 1), ("Official 10-yd split", "ten_yd_split", -1)]:
    qq = DLall.dropna(subset=[x])
    rr = sign * spearmanr(qq[x], qq.g)[0]
    bs = boot(lambda ix: sign * spearmanr(qq[x].values[ix], qq.g.values[ix])[0], len(qq), nb=1000)
    travel[lab] = (rr, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), len(qq))
qq = DLall.join(PRD.groupby("nfl_id").y10.mean().rename("prd"), how="inner")
bs = boot(lambda ix: spearmanr(qq.prd.values[ix], qq.g.values[ix])[0], len(qq), nb=1000)
travel["Pass-rush drill (tracked)"] = (spearmanr(qq.prd, qq.g)[0], np.nanpercentile(bs, 2.5),
                                       np.nanpercentile(bs, 97.5), len(qq))
put("tr40_rho", travel["40 start (tracked)"][0])
put("tr40_lo", travel["40 start (tracked)"][1])
put("tr40_hi", travel["40 start (tracked)"][2])
put("tr10_rho", travel["Official 10-yd split"][0])
put("trprd_rho", travel["Pass-rush drill (tracked)"][0])
ceil = np.sqrt(rel_g * rel_d)
put("travel_ceiling", ceil)
put("dl_d_med", DLall.d.median())
put("dl_g_med", DLall.g.median())
put("dl_d_sd", DLall.d.std())
put("dl_g_sd", DLall.g.std())
# shape similarity (the literal "same profile" measure): correlation of the two 16-point curves
gc = gd.groupby("nfl_id")[V].mean()
dc = DR[DR.grp == "DL"].groupby("nfl_id")[V].mean()
DLall["shape"] = [np.corrcoef(gc.loc[i], dc.loc[i])[0, 1] for i in DLall.index]
put("shape_med", DLall["shape"].median(), 3)
put("shape_p10", np.percentile(DLall["shape"], 10), 3)

# ----------------------------------------------------------------------------- 3. DL headline
d, Z = cache["DL"]
d = d.join(DLall[["shape", "sgo"]])
n = len(d)
put("dl_n", n)
n_rush_ok = GR[GR.grp == "DL"].groupby("nfl_id").size().loc[lambda s_: s_ >= MIN_GAME].index
put("dl_n_noforty", int(pv[(pv.position_group == "DL") & (pv.target == "production")].nfl_id.isin(n_rush_ok).sum()) - n)
y = d.y.values
r0 = prho(y, d.tr.values, Z)
bs = boot(lambda ix: prho(y[ix], d.tr.values[ix], Z[ix]), n, strata=d.draft_year.values)
put("dl_rho", r0)
put("dl_rho_lo", np.nanpercentile(bs, 2.5))
put("dl_rho_hi", np.nanpercentile(bs, 97.5))
put("dl_rho_ppos", float((bs > 0).mean()), 2)
put("dl_tr_sd", d.tr.std(), 2)
put("dl_tr_p10", np.percentile(d.tr, 10), 2)
put("dl_tr_p90", np.percentile(d.tr, 90), 2)
put("dl_pr_med", 100 * np.median(y), 1)

# planning number: OLS slope per sd of translation, x median rushes per season
dls = ss[(ss.position_group == "DL") & (ss.pass_rush_snaps > 0) & ss.nfl_id.isin(d.index)]
rush_med = dls.pass_rush_snaps.median()
put("dl_rushes_season", int(round(rush_med)))
zt = (d.tr.values - d.tr.mean()) / d.tr.std()


def fit_slope(ix):
    X = np.column_stack([np.ones(len(ix)), zt[ix], Z[ix]])
    b, *_ = np.linalg.lstsq(X, y[ix], rcond=None)
    return b


b_full = fit_slope(np.arange(n))
slope = b_full[1]
bl = np.array([fit_slope(ix)[1] for ix in [np.concatenate([RNG.choice(np.flatnonzero(d.draft_year.values == s),
                                                                        (d.draft_year.values == s).sum())
                                                             for s in (2023, 2024, 2025)]) for _ in range(NBOOT)]])
put("eff_pp", 100 * slope, 1)
put("eff_press", slope * rush_med, 1)
put("eff_press_lo", np.percentile(bl, 2.5) * rush_med, 1)
put("eff_press_hi", np.percentile(bl, 97.5) * rush_med, 1)
put("eff_press_lo80", np.percentile(bl, 10) * rush_med, 1)
put("eff_press_hi80", np.percentile(bl, 90) * rush_med, 1)
b_lpick = b_full[2]
put("pick_30_100", (b_lpick * (np.log(30) - np.log(100))) * rush_med, 1)

# forecast without tracking and tertiles
X0 = np.column_stack([np.ones(n), Z])
b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
d["fc"] = X0 @ b0
d["beat"] = d.y > d.fc
ter = pd.qcut(d.tr, 3, labels=["low", "mid", "high"])
d["ter"] = ter
put("ter_high_beat", int(round(100 * d[d.ter == "high"].beat.mean())))
put("ter_low_beat", int(round(100 * d[d.ter == "low"].beat.mean())))
put("ter_n", int((d.ter == "high").sum()))
put("ter_high_cut", d.tr.quantile(2 / 3), 2)
put("ter_low_cut", d.tr.quantile(1 / 3), 2)

# out of class: fit on two classes, predict the third
def loco(ix=None):
    dd_ = d if ix is None else d.iloc[ix]
    Zc = np.column_stack([dd_.lpick, dd_.combine_weight, dd_.edge]).astype(float)
    pr0, pr1, yy = [], [], []
    per = {}
    for yr in (2023, 2024, 2025):
        tr_ = dd_.draft_year.values != yr
        te_ = ~tr_
        if te_.sum() < 4:
            continue
        A0 = np.column_stack([np.ones(tr_.sum()), Zc[tr_]])
        A1 = np.column_stack([A0, dd_.tr.values[tr_]])
        c0, *_ = np.linalg.lstsq(A0, dd_.y.values[tr_], rcond=None)
        c1, *_ = np.linalg.lstsq(A1, dd_.y.values[tr_], rcond=None)
        p0 = np.column_stack([np.ones(te_.sum()), Zc[te_]]) @ c0
        p1 = np.column_stack([np.ones(te_.sum()), Zc[te_], dd_.tr.values[te_]]) @ c1
        yt = dd_.y.values[te_]
        pr0.append(p0 - p0.mean()); pr1.append(p1 - p1.mean()); yy.append(yt - yt.mean())
        per[yr] = (np.mean((yt - yt.mean() - (p0 - p0.mean())) ** 2), np.mean((yt - yt.mean() - (p1 - p1.mean())) ** 2))
    pr0, pr1, yy = map(np.concatenate, (pr0, pr1, yy))
    sst = (yy ** 2).sum()
    return 1 - ((yy - pr0) ** 2).sum() / sst, 1 - ((yy - pr1) ** 2).sum() / sst, per


r2b, r2t, per = loco()
put("loco_base", r2b, 3)
put("loco_gain", r2t - r2b, 3)
lb = boot(lambda ix: (lambda a: a[1] - a[0])(loco(ix)), n, nb=1000, strata=d.draft_year.values)
put("loco_gain_lo", np.nanpercentile(lb, 2.5), 3)
put("loco_gain_hi", np.nanpercentile(lb, 97.5), 3)
put("loco_gain_ppos", int(round(100 * (lb > 0).mean())))
per_class = []
for yr in (2023, 2024, 2025):
    q_ = d[d.draft_year == yr]
    Zy = np.column_stack([q_.lpick, q_.combine_weight, q_.edge]).astype(float)
    rr = prho(q_.y.values, q_.tr.values, Zy)
    put(f"cls_{yr}_rho", rr)
    put(f"cls_{yr}_n", len(q_))
    per_class.append((yr, len(q_), rr, per[yr][1] < per[yr][0]))
put("cls_mse_better", int(sum(p[3] for p in per_class)))

# decomposition: which half carries it
put("dec_nfl", prho(y, d.g.values, Z))
put("dec_drill", prho(y, d.d.values, Z))
put("dec_tr_beyond_nfl", prho(y, d.tr.values, np.column_stack([Z, d.g.values])))
put("dec_drill_beyond_nfl", prho(y, d.d.values, np.column_stack([Z, d.g.values])))
put("shape_rho", prho(y, d["shape"].values, Z))

# variants
VAR = []


def var(label, dd_, Z_, x="tr", key=None):
    r = prho(dd_.y.values, dd_[x].values, Z_)
    b_ = boot(lambda ix: prho(dd_.y.values[ix], dd_[x].values[ix], Z_[ix]), len(dd_), nb=1000,
              strata=dd_.draft_year.values)
    VAR.append(dict(label=label, n=len(dd_), rho=r, lo=np.nanpercentile(b_, 2.5), hi=np.nanpercentile(b_, 97.5)))
    if key:
        put(f"v_{key}", r)
        put(f"v_{key}_n", len(dd_))
    return r


var("Headline (pick, weight, EDGE share, class)", d, Z)
dv, Zv = design("DL", extra=("forty", "ten_yd_split"))
var("+ official 40 and 10-yd split", dv, Zv, key="offic")
dv, Zv = design("DL", extra=("ngs_athleticism_score",))
var("+ NGS athleticism score", dv, Zv, key="ngs")
for w_, lab in [(5, "0.5 s"), (15, "1.5 s")]:
    dv, Zv = design("DL", Q=player_table(GR, w=w_))
    var(f"Window {lab}", dv, Zv, key=f"w{w_}")
dv, Zv = design("DL", Q=player_table(GR[~GR.press]))
var("NFL rushes without pressure only", dv, Zv, key="nopress")
dv, Zv = design("DL", Q=player_table(GA))
var("Random + top-EPA rushes", dv, Zv, key="allplays")
PRDq = PRD.rename(columns={"y10": "y10"}).assign(drill="PRD")
dv, Zv = design("DL", Q=player_table(GR, drill=PRDq))
var("Drill side = pass-rush drill", dv, Zv, key="prd")
var("NFL half alone", d, Z, x="g")
var("Combine half alone", d, Z, x="d")
var("Curve-shape similarity", d, Z, x="shape")
VA = pd.DataFrame(VAR)

# ----------------------------------------------------------------------------- 4. named players
d["name"] = d.display_name
d["pick_s"] = d.draft_overall_pick.apply(lambda v: "UDFA" if pd.isna(v) else str(int(v)))
d["role"] = np.where(d.edge >= 0.5, "EDGE", "interior")
board = d.sort_values("tr", ascending=False)
board_out = board[["name", "draft_year", "pick_s", "role", "combine_weight", "d", "g", "tr", "y", "fc", "n_game"]]
board_out.to_csv(OUT / "board.csv")
put("rush_min", int(d.n_game.min()))
put("rush_max", int(d.n_game.max()))
top5, bot5 = board.head(5), board.tail(5)
for i, (nid, r) in enumerate(pd.concat([top5, bot5]).iterrows()):
    tag = f"b{i}"
    put(f"{tag}_d", r.d)
    put(f"{tag}_g", r.g)
    put(f"{tag}_tr", r.tr)
    put(f"{tag}_y", 100 * r.y, 1)
    put(f"{tag}_fc", 100 * r.fc, 1)
board_rows = []
for i, (nid, r) in enumerate(pd.concat([top5, bot5]).iterrows()):
    board_rows.append(f"| {r['name']} ({r.draft_year}, {r.pick_s}, {r.role}) | {NUM[f'b{i}_d']} | {NUM[f'b{i}_g']} | "
                      f"{'+' if r.tr > 0 else ''}{NUM[f'b{i}_tr']} | {NUM[f'b{i}_y']} vs {NUM[f'b{i}_fc']} |")
(OUT / "board_rows.md").write_text("\n".join(board_rows) + "\n")
# fast-40 under-translators
fast = d[(d.d > d.d.median()) & (d.ter == "low")]
put("fastlow_n", len(fast))
put("fastlow_beat", int(fast.beat.sum()))
for nm, tag in [("Laiatu Latu", "latu"), ("Braden Fiske", "fiske")]:
    r = d[d.display_name == nm].iloc[0]
    put(f"{tag}_d", r.d)
    put(f"{tag}_tr", r.tr)
    put(f"{tag}_y", 100 * r.y, 1)
    put(f"{tag}_fc", 100 * r.fc, 1)
print("FASTLOW", fast[["display_name", "d", "tr", "y", "fc"]])

# ----------------------------------------------------------------------------- figures
tgrid = np.arange(T) / 10
# Fig 1 hero: profiles of high vs low translators + pressure vs forecast
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 4.9), gridspec_kw={"width_ratios": [1.35, 1]})
for lab, col in [("high", OI["blue"]), ("low", OI["verm"])]:
    ids = d.index[d.ter == lab]
    dcv = dc.loc[ids].values
    gcv = gc.loc[ids].values
    ax[0].plot(tgrid, np.median(dcv, 0), ls="--", color=col, lw=2)
    ax[0].plot(tgrid, np.median(gcv, 0), ls="-", color=col, lw=2.6)
    ax[0].fill_between(tgrid, np.percentile(gcv, 25, 0), np.percentile(gcv, 75, 0), color=col, alpha=0.15)
ax[0].axvspan(0, 1.0, color=OI["grey"], alpha=0.07)
ax[0].text(0.03, 0.96, "first 1.0 s = Get-Off window", transform=ax[0].transAxes, fontsize=9.5, va="top",
           color=OI["grey"])
ax[0].plot([], [], color=OI["blue"], lw=2.6, label=f"Top third of translation (n={NUM['ter_n']})")
ax[0].plot([], [], color=OI["verm"], lw=2.6, label=f"Bottom third (n={NUM['ter_n']})")
ax[0].plot([], [], color="k", ls="--", lw=2, label="Combine 40 start (from onset)")
ax[0].plot([], [], color="k", ls="-", lw=2.6, label="NFL pass rush (from snap), IQR band")
ax[0].legend(loc="lower right", frameon=False, fontsize=9)
ax[0].set_xlabel("Seconds")
ax[0].set_ylabel("Speed (yd/s), median player")
ax[0].set_title("Same 40 start, different Sunday get-off", fontsize=12, loc="left")
xr = d.tr.values
yr_ = 100 * (d.y - d.fc).values
ax[1].axhline(0, color="k", lw=0.8)
ax[1].axvline(0, color="k", lw=0.8)
ax[1].scatter(xr, yr_, s=34, c=np.where(xr > 0, OI["blue"], OI["verm"]), alpha=0.8, edgecolor="white")
bfit = np.polyfit(xr, yr_, 1)
xx = np.linspace(xr.min(), xr.max(), 10)
ax[1].plot(xx, np.polyval(bfit, xx), color=OI["grey"], lw=1.6, ls="--")
lab_ids = list(board.index[:3]) + list(board.index[-3:]) + [i for i in d.index if d.loc[i, "display_name"] in
                                                             ("Laiatu Latu", "Braden Fiske")]
for i in lab_ids:
    r = d.loc[i]
    ax[1].annotate(r.display_name.split()[-1], (r.tr, 100 * (r.y - r.fc)), fontsize=9, xytext=(4, 3),
                   textcoords="offset points")
ax[1].set_xlabel("Get-Off Translation (yd in first 1.0 s)")
ax[1].set_ylabel("Pressure rate minus forecast (pts)")
ax[1].set_title(f"Beyond draft slot: rho = {NUM['dl_rho']} [{NUM['dl_rho_lo']}, {NUM['dl_rho_hi']}]",
                fontsize=12, loc="left")
fig.text(0.01, 0.01, f"n = {n} DL with a tracked 40 and 8+ random NFL rushes (2023-25). Forecast = pick, weight, "
         f"EDGE share, class (no tracking). Interval = 95% class-stratified bootstrap.", fontsize=9, color="#333333")
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig1_hero.png")

# Fig 2: validity and travel
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 4.5))
q = DLall.dropna(subset=["sgo"])
ax[0].scatter(q.g, q.sgo, s=30, color=OI["blue"], alpha=0.75, edgecolor="white")
ax[0].set_xlabel("Yards in first 1.0 s, our 20-rush sample")
ax[0].set_ylabel("NGS season get-off (s, lower = faster)")
ax[0].set_title(f"NFL half is real: rho = {NUM['val_season_rho']} (n = {NUM['val_season_n']})", fontsize=11.5,
                loc="left")
labs = list(travel)
for i, lab in enumerate(labs):
    rr, lo, hi, nn = travel[lab]
    ax[1].errorbar(rr, i, xerr=[[rr - lo], [hi - rr]], fmt="o", ms=9, capsize=4,
                   color=[OI["blue"], OI["grey"], OI["orange"]][i], lw=2)
    ax[1].text(hi + 0.03, i, f"n={nn}", va="center", fontsize=9.5)
ax[1].axvline(0, color="k", lw=0.8)
ax[1].axvline(ceil, color=OI["green"], ls=":", lw=2)
ax[1].text(ceil + 0.02, 2.35, "reliability ceiling", color=OI["green"], fontsize=9.5)
ax[1].set_yticks(range(3), labs)
ax[1].set_ylim(-0.6, 2.6)
ax[1].set_xlim(-0.45, 0.95)
ax[1].set_xlabel("Rank correlation with NFL yards in 1.0 s (faster = +)")
ax[1].set_title("Combine start barely travels to Sundays", fontsize=11.5, loc="left")
fig.tight_layout()
save(fig, "fig2_validity.png")

# Fig 3: screen + out-of-class
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
order = GROUPS[::-1]
for i, grp in enumerate(order):
    r = SC.loc[grp]
    col = OI["blue"] if grp == "DL" else OI["grey"]
    ax[0].errorbar(r.rho, i, xerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt="o", ms=9, capsize=4, color=col, lw=2)
ax[0].set_yticks(range(len(order)), [f"{g_}: {MOVE[g_]}\n-> {OUTLAB[g_]} (n={int(SC.loc[g_, 'n'])})"
                                    for g_ in order], fontsize=9.5)
ax[0].axvline(0, color="k", lw=0.8)
ax[0].set_xlabel("Partial rank correlation beyond pick, weight, role, class")
ax[0].set_title(f"Five-group screen (family-wise p = {NUM['sc_fw_p']})", fontsize=11.5, loc="left")
for i, (yr, nn, rr, better) in enumerate(per_class):
    ax[1].bar(i, rr, color=OI["blue"] if rr > 0 else OI["verm"], width=0.6)
    ax[1].text(i, rr + 0.02 if rr > 0 else rr - 0.06, f"{rr:.2f}\nn={nn}", ha="center", fontsize=9.5,
               va="bottom" if rr > 0 else "top")
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xticks(range(3), ["2023", "2024", "2025"])
ax[1].set_ylim(-0.4, 0.95)
ax[1].set_ylabel("DL partial rho within class")
ax[1].set_title(f"Out of class: R2 gain {NUM['loco_gain']}", fontsize=11.5, loc="left")
fig.tight_layout()
save(fig, "fig3_screen.png")

# Fig 4: named players small multiples
sel = list(top5.index[:3]) + list(bot5.index[-3:])
fig, axs = plt.subplots(2, 3, figsize=(FIGW, 6.2), sharex=True, sharey=True)
for ax_, nid in zip(axs.flat, sel):
    r = d.loc[nid]
    col = OI["blue"] if r.tr > 0 else OI["verm"]
    for _, rep in DR[(DR.nfl_id == nid)].iterrows():
        ax_.plot(tgrid, rep[V].values.astype(float), ls="--", color="k", lw=1.6)
    pls = gd[gd.nfl_id == nid]
    for _, pr in pls.iterrows():
        ax_.plot(tgrid, pr[V].values.astype(float), color=col, lw=0.6, alpha=0.35)
    ax_.plot(tgrid, pls[V].mean().values, color=col, lw=2.8)
    ax_.axvline(1.0, color=OI["grey"], lw=0.8, ls=":")
    ax_.set_title(f"{r['name']} ({r.draft_year}, {r.role})\nTranslation {r.tr:+.2f} yd", fontsize=10.5)
for ax_ in axs[1]:
    ax_.set_xlabel("Seconds")
for ax_ in axs[:, 0]:
    ax_.set_ylabel("Speed (yd/s)")
fig.text(0.01, 0.005, "Dashed black = Combine 40 reps (from onset). Thin = single random NFL rushes (from snap). "
         "Thick = NFL mean.", fontsize=9.5, color="#333333")
fig.tight_layout(rect=(0, 0.03, 1, 1))
save(fig, "fig4_players.png")

# Fig 5: robustness
fig, ax = plt.subplots(figsize=(FIGW, 4.8))
VA2 = VA.iloc[::-1].reset_index(drop=True)
for i, r in VA2.iterrows():
    grey = r.label in ("NFL half alone", "Combine half alone", "Curve-shape similarity")
    col = OI["grey"] if grey else (OI["blue"] if i == len(VA2) - 1 else OI["sky"])
    ax.errorbar(r.rho, i, xerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt="o", ms=8, capsize=4, color=col, lw=2)
    ax.text(0.78, i, f"n={r.n}", va="center", fontsize=9.5)
ax.set_yticks(range(len(VA2)), VA2.label, fontsize=9.5)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlim(-0.5, 0.88)
ax.set_xlabel("Partial rank correlation with NFL pressure rate (95% CI)")
ax.set_title("Positive in every variant (blue); the NFL half carries it (grey)", fontsize=11.5, loc="left")
fig.tight_layout()
save(fig, "fig5_checks.png")

# the checker reads unsigned decimals (the text uses a typographic minus), so list magnitudes of negatives too
NUM["abs_of_negatives"] = sorted({v.lstrip("-") for v in NUM.values() if isinstance(v, str) and v.startswith("-")})
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
print(json.dumps(NUM, indent=1))
print(SC)
print(VA)
print(board_out.to_string())
