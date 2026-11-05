"""Carry-Over: how much of his Combine drill speed a player keeps in the same movement on NFL snaps.

Round 5, candidate 3. Run from the repo root:
    uv run python writeup/candidates/round_05/cand_3/analysis.py

Reads data/processed/*.parquet only (about one minute). Writes numbers.json, board.csv and figures/*.png next to this
file.

Movement matched per group (first 1.5 s after movement onset, one function on drills and NFL snaps):
  DL  pass-rush drill get-off            <-> NFL pass rush
  OL  pass-rush-drop kick-slide          <-> NFL pass set
  DB  back-pedal drills (four)           <-> NFL coverage start on pass plays
  WR  route drills (six)                 <-> NFL routes
  TE  route drills (five)                <-> NFL routes
NFL snaps: the `random` half of game_plays_sample only (the top-EPA half is biased toward big plays), pass plays.
Onset: after the player is at rest (< 0.8 yd/s, within 2 s of clip start / snap), the first frame with smoothed
speed >= 0.8 yd/s, backed up 0.2 s.
Speed curve: Savitzky-Golay smoothed speed (5, 2) over 16 frames (0-1.5 s).
Carry-Over % = 100 x (player mean NFL curve speed, alignment-adjusted) / (player mean drill curve speed, drill x year
adjusted). 100% = he moves as fast in the first 1.5 s on Sundays as in Indianapolis.
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
from scipy.stats import rankdata, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2027)
NBOOT = 2000
NPERM = 1000
T = 16            # 0-1.5 s
MIN_GAME = 8      # NFL random pass plays needed for a player curve
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

POSDRILLS = {
    "DL": ["PASS_RUSH_DRILL"],
    "OL": ["PASS_RUSH_DROP"],
    "DB": ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "LINE_DRILL",
           "W_DRILL"],
    "WR": ["GO_ROUTE_RIGHT", "SLANT_ROUTE_LEFT", "DAGGER_ROUTE_LEFT", "SPEED_OUT_ROUTE_LEFT", "CURL_ROUTE_RIGHT",
           "COMEBACK_ROUTE_RIGHT"],
    "TE": ["CORNER_ROUTE", "IN_ROUTE_LEFT", "HOOK_ROUTE_RIGHT", "FLAT_ROUTE_LEFT", "WHEEL_ROUTE_LEFT"],
}
MOVE = {"DL": "pass-rush get-off", "OL": "pass set (kick-slide)", "DB": "back-pedal start", "WR": "route release",
        "TE": "route release"}
GROUPS = ["DL", "OL", "DB", "WR", "TE"]


def curve(s, start=0):
    s = np.asarray(s, float)
    if len(s) - start < 9:
        return None
    sm_ = savgol_filter(s, 5, 2)
    seg = sm_[start:]
    rest = np.flatnonzero(seg < 0.8)          # must start from (near) rest: clips that begin mid-movement are dropped
    if not len(rest) or rest[0] > 20:
        return None
    hit = np.flatnonzero(seg[rest[0]:] >= 0.8)
    if not len(hit):
        return None
    o = start + rest[0] + max(int(hit[0]) - 2, 0)
    if o + T > len(sm_):
        return None
    return np.clip(sm_[o:o + T], 0, None)


# drill reps
evx = ev.set_index("event_id")
keep = ev[[d in POSDRILLS.get(g, []) for d, g in zip(ev.drill_name, ev.position_group)]].event_id
rows = []
for eid, r in cf[cf.event_id.isin(set(keep))].groupby("event_id", sort=False):
    c = curve(r.s.values)
    if c is None:
        continue
    e = evx.loc[eid]
    rows.append((e.nfl_id, e.position_group, e.drill_name, int(e.draft_year), int(e.attempt_no), *c))
V = [f"v{i}" for i in range(T)]
DR = pd.DataFrame(rows, columns=["nfl_id", "grp", "drill", "yr", "att", *V])
DR["m"] = DR[V].mean(axis=1)
DR["m_adj"] = DR.m / DR.groupby(["drill", "yr"]).m.transform("median") * DR.groupby("drill").m.transform("median")

# NFL random pass plays
gpr = gp[(gp.sample_kind == "random") & gp.pass_result.notna()].merge(
    pl[["nfl_id", "position_group", "draft_year"]], on="nfl_id")
gpr = gpr[["game_id", "play_id", "nfl_id", "snap_frame", "lined_up_position", "position_group", "season", "draft_year",
           "time_to_pressure"]]
f = gf.merge(gpr[["game_id", "play_id", "nfl_id"]], on=["game_id", "play_id", "nfl_id"]).sort_values(
    ["nfl_id", "game_id", "play_id", "frame"])
meta = gpr.set_index(["nfl_id", "game_id", "play_id"])
rows = []
for k, r in f.groupby(["nfl_id", "game_id", "play_id"], sort=False):
    mrow = meta.loc[k]
    sf = int(mrow.snap_frame) if np.isfinite(mrow.snap_frame) else 0
    c = curve(r.s.values, sf)
    if c is None:
        continue
    rows.append((*k, mrow.position_group, mrow.lined_up_position, int(mrow.season), int(mrow.draft_year),
                 bool(pd.notna(mrow.time_to_pressure)), *c))
GA = pd.DataFrame(rows, columns=["nfl_id", "game_id", "play_id", "grp", "align", "season", "draft_year", "press", *V])
GA["m"] = GA[V].mean(axis=1)
GA["m_adj"] = GA.m / GA.groupby(["grp", "align"]).m.transform("median") * GA.groupby("grp").m.transform("median")
put("n_drill_reps", len(DR))
put("const_window_s", T / 10 - 0.1, 1)
put("const_onset_yds", 0.8, 1)
put("const_backup_s", 0.2, 1)
put("const_alpha", 0.05, 2)
put("n_nfl_plays", len(GA))


def player_table(GA_sub):
    cnt = GA_sub.groupby("nfl_id").size()
    g = GA_sub.groupby("nfl_id").m_adj.mean()[cnt >= MIN_GAME].rename("g")
    d = DR.groupby("nfl_id").m_adj.mean().rename("d")
    Q = pd.concat([g, d], axis=1, join="inner")
    Q["co"] = 100 * Q.g / Q.d
    gc = GA_sub.groupby("nfl_id")[V].mean()
    dc = DR.groupby("nfl_id")[V].mean()
    Q["shape"] = [np.corrcoef(gc.loc[i], dc.loc[i])[0, 1] for i in Q.index]
    Q["n_game"] = cnt.reindex(Q.index)
    Q["n_drill"] = DR.groupby("nfl_id").size().reindex(Q.index)
    return Q


PT = player_table(GA).reset_index()
PT = PT.merge(pl[["nfl_id", "display_name", "position_group", "nfl_position", "draft_overall_pick", "draft_year",
                  "combine_weight"]], on="nfl_id")
for c in ["g", "d"]:
    PT[c + "z"] = PT.groupby("position_group")[c].transform(lambda v: (v - v.mean()) / v.std())
PT["gap"] = PT.gz - PT.dz
PT["lpick"] = np.log(PT.draft_overall_pick.fillna(UDFA_PICK))
sh = al.pivot_table(index="nfl_id", columns="lined_up_position", values="plays", aggfunc="sum").fillna(0)
sh = sh.div(sh.sum(axis=1), axis=0)
role = pd.DataFrame({"edge": sh["EDGE"], "tsh": sh["T"], "csh": sh["C"], "cb": sh["CB"],
                     "slot": sh["SLOT_CB"] + sh["SLOT_WR"]}).reset_index()
PT = PT.merge(role, on="nfl_id", how="left").merge(
    cr[["nfl_id", "ngs_athleticism_score", "ten_yd_split", "forty"]], on="nfl_id", how="left")
ROLE = {"OL": ["tsh", "csh"], "DL": ["edge"], "DB": ["cb", "slot"], "WR": ["slot"], "TE": ["slot"]}
for g_ in GROUPS:
    put(f"n_players_{g_}", int((PT.position_group == g_).sum()))


# ----------------------------------------------------------------------------- stats helpers
def resid(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ b


def prho(y, x, Z):
    """Rank partial correlation: Spearman of OLS residuals on the baseline Z."""
    return spearmanr(resid(y, Z), resid(x, Z))[0]


def boot_ci(fn, n, nb=NBOOT, strata=None):
    vals = []
    for _ in range(nb):
        if strata is None:
            idx = RNG.integers(0, n, n)
        else:
            idx = np.concatenate([RNG.choice(np.flatnonzero(strata == s), (strata == s).sum()) for s in np.unique(strata)])
        try:
            vals.append(fn(idx))
        except Exception:
            pass
    return np.nanpercentile(vals, [2.5, 97.5]), np.array(vals)


def outcome(grp, tgt):
    v = pv[(pv.position_group == grp) & (pv.target == tgt)][["nfl_id", "actual"]]
    return v


def design(grp, tgt, metric, extra=()):
    d = PT[PT.position_group == grp].merge(outcome(grp, tgt), on="nfl_id")
    cols = ["lpick", "combine_weight", *ROLE[grp], *extra]
    d = d.dropna(subset=cols + [metric, "actual"]).reset_index(drop=True)
    Z = np.column_stack([d[cols].values, d.draft_year.values == 2023, d.draft_year.values == 2024]).astype(float)
    return d, Z


# ----------------------------------------------------------------------------- 1. screen
METRICS = {"co": "Carry-Over %", "shape": "Curve-shape match", "gap": "Game minus drill (sd)"}
CELLS = [(g_, t_) for g_ in GROUPS for t_ in ["production", "opportunity"] if len(outcome(g_, t_))]
LAB = {("DL", "production"): "DL pressure rate", ("DL", "opportunity"): "DL snaps/season",
       ("OL", "production"): "OL pressures avoided", ("OL", "opportunity"): "OL snaps/season",
       ("DB", "production"): "DB solo tackles/100", ("DB", "opportunity"): "DB snaps/season",
       ("WR", "production"): "WR yards/route", ("WR", "opportunity"): "WR snaps/season",
       ("TE", "opportunity"): "TE snaps/season"}
screen = []
cache = {}
for cell in CELLS:
    sign = -1 if cell == ("OL", "production") else 1   # fewer pressures allowed is better
    for m in METRICS:
        d, Z = design(*cell, m)
        y = sign * d.actual.values
        cache[(cell, m)] = (d, Z, y)
        screen.append(dict(grp=cell[0], tgt=cell[1], metric=m, n=len(d), rho=prho(y, d[m].values, Z)))
SC = pd.DataFrame(screen)
# family-wise permutation: shuffle the outcome within draft class in each group-outcome, take max |rho| over all cells
maxr = []
for _ in range(NPERM):
    best = 0.0
    for cell in CELLS:
        d0, Z0, y0 = cache[(cell, "co")]
        yp = y0.copy()
        for yr in (2023, 2024, 2025):
            ix = np.flatnonzero(d0.draft_year.values == yr)
            yp[ix] = yp[RNG.permutation(ix)]
        ry = resid(yp, Z0)
        for m in METRICS:
            d, Z, y = cache[(cell, m)]
            if len(d) != len(d0):
                # different complete cases: permute separately
                yq = y.copy()
                for yr in (2023, 2024, 2025):
                    ix = np.flatnonzero(d.draft_year.values == yr)
                    yq[ix] = yq[RNG.permutation(ix)]
                r = abs(spearmanr(resid(yq, Z), resid(d[m].values, Z))[0])
            else:
                r = abs(spearmanr(ry, resid(d[m].values, Z))[0])
            best = max(best, r)
    maxr.append(best)
maxr = np.array(maxr)
# pre-specified headline cell: Carry-Over % vs the group's production outcome; group chosen = most significant cell
SC["p"] = [spearmanr(resid(cache[((r.grp, r.tgt), r.metric)][2], cache[((r.grp, r.tgt), r.metric)][1]),
                     resid(cache[((r.grp, r.tgt), r.metric)][0][r.metric].values,
                           cache[((r.grp, r.tgt), r.metric)][1]))[1] for r in SC.itertuples()]
top = SC[(SC.grp == "DL") & (SC.tgt == "production") & (SC.metric == "co")].iloc[0]
put("screen_cells", len(SC))
put("screen_top_rho", top.rho)
put("screen_fw_p", float((maxr >= abs(top.rho)).mean()), 3)
put("screen_nominal", int((SC.p < 0.05).sum()))
put("screen_expected", len(SC) * 0.05, 1)
put("screen_gap_rho", SC[(SC.grp == "DL") & (SC.tgt == "production") & (SC.metric == "gap")].rho.iloc[0])
put("screen_gap_fw_p", float((maxr >= SC[(SC.grp == "DL") & (SC.tgt == "production") & (SC.metric == "gap")].rho.abs().iloc[0]).mean()), 3)
put("screen_rank_p", int(SC.p.rank().loc[top.name]))
te = SC[SC.grp == "TE"].sort_values("p").iloc[0]
put("screen_te_rho", te.rho)
put("screen_te_n", int(te.n))
put("screen_dl_p", top.p, 3)
SC.to_csv(OUT / "screen.csv", index=False)

# ----------------------------------------------------------------------------- 1b. does the movement travel? (same-player drill -> NFL)
TR = []
for g_ in GROUPS:
    q = PT[PT.position_group == g_].reset_index(drop=True)
    r_ = spearmanr(q.d, q.g)[0]
    c_, _ = boot_ci(lambda ix: spearmanr(q.d.values[ix], q.g.values[ix])[0], len(q), nb=1000)
    # reliabilities: NFL odd/even split, drill rep-to-rep (first two reps), Spearman-Brown to the player's counts
    gg = GA[(GA.grp == g_) & GA.nfl_id.isin(q.nfl_id)].sort_values(["nfl_id", "game_id", "play_id"]).copy()
    gg["h"] = gg.groupby("nfl_id").cumcount() % 2
    s2 = gg.groupby(["nfl_id", "h"]).m_adj.mean().unstack().dropna()
    rg_ = np.corrcoef(s2[0], s2[1])[0, 1]
    rg_ = 2 * rg_ / (1 + rg_)
    dd_ = DR[(DR.grp == g_) & DR.nfl_id.isin(q.nfl_id)].copy()
    dd_["k"] = dd_.groupby("nfl_id").cumcount()
    w2 = dd_[dd_.k < 2].pivot(index="nfl_id", columns="k", values="m_adj").dropna()
    r1 = np.corrcoef(w2[0], w2[1])[0, 1]
    kbar = q.n_drill.mean()
    rd_ = kbar * r1 / (1 + (kbar - 1) * r1) if r1 > 0 else np.nan
    ceil = np.sqrt(max(rg_, 0) * rd_) if np.isfinite(rd_) else np.nan
    TR.append(dict(grp=g_, n=len(q), rho=r_, lo=c_[0], hi=c_[1], rel_g=rg_, rel_d=rd_, ceil=ceil,
                   co_med=q.co.median()))
    put(f"tr_{g_}_rho", r_)
    put(f"tr_{g_}_lo", c_[0])
    put(f"tr_{g_}_hi", c_[1])
    put(f"tr_{g_}_n", len(q))
    put(f"tr_{g_}_co_med", q.co.median(), 0)
    if np.isfinite(ceil):
        put(f"tr_{g_}_ceil", ceil)
TRD = pd.DataFrame(TR)

# ----------------------------------------------------------------------------- 2. OL trap
DR["acc"] = np.diff(DR[V].values, axis=1).max(axis=1) / 0.1
PT = PT.merge(DR.groupby("nfl_id").acc.mean().rename("dacc").reset_index(), on="nfl_id", how="left")
d, Z = design("OL", "production", "dacc")
d["d"] = d.dacc
olr = spearmanr(d.d, d.actual)[0]
Zn = np.column_stack([d.lpick, d.combine_weight, d.draft_year == 2023, d.draft_year == 2024]).astype(float)
put("ol_raw_rho", olr)
put("ol_nopos_rho", prho(d.actual.values, d.d.values, Zn))
put("ol_pos_rho", prho(d.actual.values, d.d.values, Z))
put("ol_n", len(d))
olpos = d.groupby("nfl_position").agg(dr=("d", "median"), pr=("actual", "median"), n=("d", "size"))
put("ol_T_pr", 100 * olpos.loc["T", "pr"], 1)
put("ol_C_pr", 100 * olpos.loc["C", "pr"], 1)
put("ol_T_drill", olpos.loc["T", "dr"], 2)
put("ol_C_drill", olpos.loc["C", "dr"], 2)
put("ol_G_drill", olpos.loc["G", "dr"], 2)
OLD = d.copy()

# ----------------------------------------------------------------------------- 3. DL headline
d, Z, y = cache[(("DL", "production"), "co")]
DL = d.copy()
n = len(d)
put("dl_n", n)
r0 = prho(y, d.co.values, Z)
ci, _ = boot_ci(lambda ix: prho(y[ix], d.co.values[ix], Z[ix]), n, strata=d.draft_year.values)
put("dl_rho", r0)
put("dl_rho_lo", ci[0])
put("dl_rho_hi", ci[1])
put("dl_co_median", DL.co.median(), 0)
put("dl_co_p10", np.percentile(DL.co, 10), 0)
put("dl_co_p90", np.percentile(DL.co, 90), 0)
put("dl_co_sd", DL.co.std(), 1)
put("dl_pr_median", 100 * DL.actual.median(), 1)
put("dl_drill_to_game", spearmanr(DL.d, DL.g)[0])
put("dl_rho_raw", spearmanr(DL.co, DL.actual)[0])

# effect size: OLS slope per sd of Carry-Over on pressure rate, x median rushes per season
dls = ss[(ss.position_group == "DL") & (ss.pass_rush_snaps > 0)]
rush_med = dls[dls.nfl_id.isin(DL.nfl_id)].pass_rush_snaps.median()
put("dl_rush_med", int(round(rush_med)))
zc = (DL.co.values - DL.co.mean()) / DL.co.std()


def slope(ix):
    X1 = np.column_stack([np.ones(len(ix)), Z[ix], zc[ix]])
    return np.linalg.lstsq(X1, y[ix], rcond=None)[0][-1]


b0 = slope(np.arange(n))
bci, _ = boot_ci(slope, n, strata=d.draft_year.values)
put("dl_pp_per_sd", 100 * b0, 1)
put("dl_pp_lo", 100 * bci[0], 1)
put("dl_pp_hi", 100 * bci[1], 1)
put("dl_press_per_sd", b0 * rush_med, 1)
put("dl_press_lo", bci[0] * rush_med, 1)
put("dl_press_hi", bci[1] * rush_med, 1)
# pick slope for comparison: pressure-rate gain from pick 100 -> pick 30 in baseline model
X1 = np.column_stack([np.ones(n), Z])
bb = np.linalg.lstsq(X1, y, rcond=None)[0]
pick_eff = bb[1] * (np.log(30) - np.log(100))
put("dl_pick100_30_press", pick_eff * rush_med, 1)

# decomposition: NFL half vs Combine half
put("dl_rho_game", prho(y, DL.g.values, Z))
put("dl_rho_drill", prho(y, DL.d.values, Z))
Zg = np.column_stack([Z, DL.g.values])
put("dl_rho_co_given_game", prho(y, DL.co.values, Zg))
Zd = np.column_stack([Z, DL.d.values])
put("dl_rho_game_given_drill", prho(y, DL.g.values, Zd))

# variants
VAR = []


def variant(label, Q_override=None, extra=(), key=None):
    if Q_override is not None:
        dd = PTsub_to_design(Q_override, extra)
    else:
        dd = design("DL", "production", "co", extra)
    dq, Zq = dd
    yq = dq.actual.values
    r = prho(yq, dq.co.values, Zq)
    c_, _ = boot_ci(lambda ix: prho(yq[ix], dq.co.values[ix], Zq[ix]), len(dq), nb=1000, strata=dq.draft_year.values)
    VAR.append(dict(label=label, n=len(dq), rho=r, lo=c_[0], hi=c_[1]))
    if key:
        put(key, r)
        put(key + "_n", len(dq))
    return r


def PTsub_to_design(Q, extra=()):
    Q = Q.reset_index().merge(pl[["nfl_id", "draft_overall_pick", "draft_year", "combine_weight"]], on="nfl_id")
    Q["lpick"] = np.log(Q.draft_overall_pick.fillna(UDFA_PICK))
    Q = Q.merge(role[["nfl_id", "edge"]], on="nfl_id", how="left").merge(outcome("DL", "production"), on="nfl_id")
    Q = Q.merge(cr[["nfl_id", "ngs_athleticism_score", "ten_yd_split"]], on="nfl_id", how="left")
    cols = ["lpick", "combine_weight", "edge", *extra]
    Q = Q.dropna(subset=cols + ["co", "actual"]).reset_index(drop=True)
    Zq = np.column_stack([Q[cols].values, Q.draft_year == 2023, Q.draft_year == 2024]).astype(float)
    return Q, Zq


variant("Headline: pick, weight, EDGE share, class")
variant("+ NGS athleticism score", extra=["ngs_athleticism_score"], key="v_ngs")
variant("+ official 10-yd split", extra=["ten_yd_split"], key="v_ten")
GAdl = GA[GA.grp == "DL"]
variant("NFL curve from non-pressure rushes only", Q_override=player_table(GAdl[~GAdl.press]), key="v_nopress")
GAraw = GAdl.copy()
GAraw["m_adj"] = GAraw.m
variant("No alignment adjustment", Q_override=player_table(GAraw), key="v_noalign")
for w, lab in [(6, "0.5"), (11, "1.0")]:
    tmp = GAdl.copy()
    tmp["m_adj"] = tmp[V[:w]].mean(axis=1)
    DRk = DR.copy()
    DR_m = DR[V[:w]].mean(axis=1)
    DR["m_adj"] = DR_m / DR_m.groupby([DR.drill, DR.yr]).transform("median") * DR_m.groupby(DR.drill).transform("median")
    variant(f"Window 0-{lab} s", Q_override=player_table(tmp), key=f"v_win{w}")
    DR["m_adj"] = DRk["m_adj"]
VARD = pd.DataFrame(VAR)

# reliability: split-half of NFL plays (odd/even), drill attempt 1 vs 2
gdl = GAdl.sort_values(["nfl_id", "game_id", "play_id"]).copy()
gdl["h"] = gdl.groupby("nfl_id").cumcount() % 2
sh2 = gdl.groupby(["nfl_id", "h"]).m_adj.mean().unstack().reindex(DL.nfl_id).dropna()
rr = np.corrcoef(sh2[0], sh2[1])[0, 1]
put("rel_game_half", rr)
put("rel_game_full", 2 * rr / (1 + rr))
ddl = DR[(DR.grp == "DL")].pivot_table(index="nfl_id", columns="att", values="m_adj")
ddl = ddl.reindex(DL.nfl_id)[[1, 2]].dropna()
rd = np.corrcoef(ddl[1], ddl[2])[0, 1]
put("rel_drill_single", rd)
put("rel_drill_two", 2 * rd / (1 + rd))
put("rel_drill_n", len(ddl))
# split-half Carry-Over itself (half A games / drill attempt 1 vs half B / attempt 2)
both = sh2.rename(columns={0: "gA", 1: "gB"}).join(ddl.rename(columns={1: "dA", 2: "dB"}), how="inner")
put("rel_co_half", spearmanr(both.gA / both.dA, both.gB / both.dB)[0])

# ----------------------------------------------------------------------------- 4. out-of-class (leave one class out)
def loco(idx):
    dq, Zq, yq = DL.iloc[idx].reset_index(drop=True), Z[idx], y[idx]
    pred_b, pred_f, obs = [], [], []
    for yr in (2023, 2024, 2025):
        tr, te = dq.draft_year.values != yr, dq.draft_year.values == yr
        if te.sum() < 5:
            continue
        Zb = Zq[:, :3]  # pick, weight, edge (class dummies cannot transfer)
        Xb = np.column_stack([np.ones(len(yq)), Zb])
        Xf = np.column_stack([Xb, dq.co.values])
        bb_ = np.linalg.lstsq(Xb[tr], yq[tr], rcond=None)[0]
        bf_ = np.linalg.lstsq(Xf[tr], yq[tr], rcond=None)[0]
        mu = yq[tr].mean()
        # centre each held-out class on its own mean to remove class-level shifts
        pb, pf = Xb[te] @ bb_, Xf[te] @ bf_
        o = yq[te]
        pred_b.append(pb - pb.mean()); pred_f.append(pf - pf.mean()); obs.append(o - o.mean())
    pb, pf, o = map(np.concatenate, (pred_b, pred_f, obs))
    sst = (o ** 2).sum()
    return (1 - ((o - pb) ** 2).sum() / sst), (1 - ((o - pf) ** 2).sum() / sst)


r2b, r2f = loco(np.arange(n))
put("loco_r2_base", r2b, 3)
put("loco_r2_full", r2f, 3)
put("loco_gain", r2f - r2b, 3)
gains = []
for _ in range(NBOOT):
    ix = np.concatenate([RNG.choice(np.flatnonzero(DL.draft_year.values == s), (DL.draft_year.values == s).sum())
                         for s in (2023, 2024, 2025)])
    a_, b_ = loco(ix)
    gains.append(b_ - a_)
gains = np.array(gains)
put("loco_gain_lo", np.percentile(gains, 2.5), 3)
put("loco_gain_hi", np.percentile(gains, 97.5), 3)
put("loco_gain_pos_pct", int(round(100 * (gains > 0).mean())))
per_class = {}
for yr in (2023, 2024, 2025):
    m_ = DL.draft_year.values == yr
    per_class[yr] = spearmanr(resid(y[m_], Z[m_, :3]), resid(DL.co.values[m_], Z[m_, :3]))[0]
    put(f"class_rho_{yr}", per_class[yr])
    put(f"class_n_{yr}", int(m_.sum()))

# ----------------------------------------------------------------------------- 5. prospective: rookie snaps -> later seasons
dls2 = ss[ss.position_group == "DL"].merge(pl[["nfl_id", "draft_year"]], on="nfl_id")
dls2["yrn"] = dls2.season - dls2.draft_year
later = dls2[dls2.yrn >= 1].groupby("nfl_id")[["pressures", "pass_rush_snaps"]].sum()
later = later[later.pass_rush_snaps >= 100]
rook = dls2[(dls2.yrn == 0) & (dls2.pass_rush_snaps >= 50)].set_index("nfl_id")
rook_pr = (rook.pressures / rook.pass_rush_snaps).rename("rook_pr")
GAr = GAdl[GAdl.season == GAdl.draft_year]
Qr = player_table(GAr).join((later.pressures / later.pass_rush_snaps).rename("later_pr"), how="inner").join(rook_pr, how="left")
Qr = Qr.reset_index().merge(pl[["nfl_id", "draft_overall_pick", "combine_weight", "draft_year"]], on="nfl_id").merge(
    role[["nfl_id", "edge"]], on="nfl_id")
Qr["lpick"] = np.log(Qr.draft_overall_pick.fillna(UDFA_PICK))
Zr = np.column_stack([Qr.lpick, Qr.combine_weight, Qr.edge, Qr.draft_year == 2023]).astype(float)
GAr_n = GAr.groupby("nfl_id").size()
put("pro_plays_med", int(GAr_n.reindex(Qr.nfl_id).median()))
put("pro_n", len(Qr))
put("pro_rho_rook", prho(Qr.later_pr.values, Qr.co.values, Zr))
pci, _ = boot_ci(lambda ix: prho(Qr.later_pr.values[ix], Qr.co.values[ix], Zr[ix]), len(Qr), nb=1000)
put("pro_rho_rook_lo", pci[0])
put("pro_rho_rook_hi", pci[1])
qq = Qr.dropna(subset=["rook_pr"])
put("pro_rook_pr_rho", spearmanr(qq.rook_pr, qq.later_pr)[0])
put("pro_rook_pr_n", len(qq))
# power at the headline effect for n=pro_n: approx via Fisher z
from scipy.stats import norm  # noqa: E402
zeff = np.arctanh(r0) * np.sqrt(len(Qr) - 3 - 5)
put("pro_power", float(norm.cdf(zeff - 1.96)), 2)

# selection: who reached 100 rushes
allDL = PT[PT.position_group == "DL"].merge(outcome("DL", "production"), on="nfl_id", how="left")
allDL["reached"] = allDL.actual.notna()
put("sel_n_all", len(allDL))
put("sel_n_out", int((~allDL.reached).sum()))
put("sel_rho", spearmanr(allDL.co, allDL.reached)[0])

# ----------------------------------------------------------------------------- 6. board
Xb_ = np.column_stack([np.ones(n), Z])
bbase = np.linalg.lstsq(Xb_, y, rcond=None)[0]
DL["forecast"] = Xb_ @ bbase
DL["over"] = DL.actual - DL.forecast
DL["role"] = np.where(DL.edge >= 0.5, "EDGE", "Interior")
DL["pct"] = rankdata(DL.co) / len(DL) * 100
DL = DL.sort_values("co", ascending=False).reset_index(drop=True)
DL[["display_name", "draft_year", "draft_overall_pick", "role", "combine_weight", "d", "g", "co", "n_game", "n_drill",
    "actual", "forecast", "over"]].to_csv(OUT / "board.csv", index=False)
q_hi, q_lo = DL.co.quantile(2 / 3), DL.co.quantile(1 / 3)
hi, lo = DL[DL.co >= q_hi], DL[DL.co <= q_lo]
put("tercile_hi_over", 100 * hi.over.mean(), 1)
put("tercile_lo_over", 100 * lo.over.mean(), 1)
put("tercile_hi_beat", int(round(100 * (hi.over > 0).mean())))
put("tercile_lo_beat", int(round(100 * (lo.over > 0).mean())))
put("tercile_hi_n", len(hi))
put("tercile_lo_n", len(lo))
put("tercile_hi_co", hi.co.min(), 0)
put("tercile_lo_co", lo.co.max(), 0)
# the "fast drill, slow game" flag: top-half drill, bottom-third carry-over
flag = DL[(DL.d >= DL.d.median()) & (DL.co <= q_lo)]
put("flag_n", len(flag))
put("flag_over", 100 * flag.over.mean(), 1)
put("flag_beat", int((flag.over > 0).sum()))
board = pd.concat([DL.head(5), DL.tail(5)])
board_rows = []
for _, r in board.iterrows():
    pk = "UDFA" if pd.isna(r.draft_overall_pick) else str(int(r.draft_overall_pick))
    board_rows.append(dict(name=r.display_name, cls=int(r.draft_year), pick=pk, role=r.role, wt=int(r.combine_weight),
                           drill=f"{r.d:.2f}", nfl=f"{r.g:.2f}", co=f"{r.co:.0f}", n=int(r.n_game),
                           pr=f"{100 * r.actual:.1f}", fc=f"{100 * r.forecast:.1f}"))
json.dump(board_rows, open(OUT / "board_table.json", "w"), indent=1)
for i, r in enumerate(board_rows):
    put(f"b{i}_drill", float(r["drill"]))
    put(f"b{i}_nfl", float(r["nfl"]))
    put(f"b{i}_pr", float(r["pr"]), 1)
    put(f"b{i}_fc", float(r["fc"]), 1)

# ----------------------------------------------------------------------------- figures
# Fig 1: hero curves, top vs bottom tercile of Carry-Over (DL)
t = np.arange(T) / 10
fig, axes = plt.subplots(1, 2, figsize=(FIGW, 4.6), sharey=True)
for ax, sub, name, col in [(axes[0], hi, f"High Carry-Over (top third, n = {len(hi)})", OI["blue"]),
                           (axes[1], lo, f"Low Carry-Over (bottom third, n = {len(lo)})", OI["verm"])]:
    ids = set(sub.nfl_id)
    dc = DR[DR.nfl_id.isin(ids)].groupby("nfl_id")[V].mean()
    gc = GAdl[GAdl.nfl_id.isin(ids)].groupby("nfl_id")[V].mean()
    for c, ls, lab in [(dc, "--", "Combine pass-rush drill"), (gc, "-", "NFL rushes (random)")]:
        med = c.median().values
        q1, q3 = c.quantile(0.25).values, c.quantile(0.75).values
        ax.fill_between(t, q1, q3, color=col if ls == "-" else OI["grey"], alpha=0.18, lw=0)
        ax.plot(t, med, ls, color=col if ls == "-" else OI["black"], lw=2.6, label=lab)
    ratio = 100 * gc.median().values.mean() / dc.median().values.mean()
    ax.set_title(name, fontsize=12)
    ax.text(1.48, 0.35, f"pressure vs forecast: {100 * sub.over.mean():+.1f} pts", ha="right", fontsize=10.5,
            color=col, weight="bold")
    ax.set_xlabel("Seconds from first movement")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(0, 1.5)
axes[0].set_ylabel("Speed (yd/s), median and middle 50%")
fig.suptitle("Same Sunday curve, different drill: low Carry-Over rushers were faster in Indianapolis, not on Sundays",
             fontsize=13, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig3_curves.png")

# Fig 2: named small multiples: 3 highest, 3 lowest carry-over with 150+ rushes in NFL? use top/bottom by co
names = list(DL.head(3).nfl_id) + list(DL.tail(3).nfl_id)
fig, axes = plt.subplots(2, 3, figsize=(FIGW, 6.4), sharex=True, sharey=True)
for ax, pid in zip(axes.ravel(), names):
    r = DL[DL.nfl_id == pid].iloc[0]
    col = OI["blue"] if r.co >= q_hi else OI["verm"]
    for _, rep in GAdl[GAdl.nfl_id == pid].iterrows():
        ax.plot(t, rep[V].values.astype(float), color=col, alpha=0.18, lw=0.9)
    ax.plot(t, GAdl[GAdl.nfl_id == pid][V].mean().values, color=col, lw=2.6, label="NFL mean")
    for j, (_, rep) in enumerate(DR[DR.nfl_id == pid].iterrows()):
        ax.plot(t, rep[V].values.astype(float), "--", color=OI["black"], lw=1.8, label="Combine rep" if j == 0 else None)
    pk = "UDFA" if pd.isna(r.draft_overall_pick) else f"pick {int(r.draft_overall_pick)}"
    ax.set_title(f"{r.display_name} ({pk}, {r.role})", fontsize=10.5)
    ax.text(0.03, 0.95, f"Carry-Over {r.co:.0f}%\npressure {100 * r.actual:.1f}% vs {100 * r.forecast:.1f}% forecast",
            transform=ax.transAxes, va="top", fontsize=9.5)
axes[0, 0].legend(loc="lower right", frameon=False, fontsize=9)
for ax in axes[1]:
    ax.set_xlabel("Seconds from first movement")
for ax in axes[:, 0]:
    ax.set_ylabel("Speed (yd/s)")
fig.suptitle("Top row: three highest Carry-Over. Bottom row: three lowest. Thin lines = single NFL rushes",
             fontsize=12.5, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "fig5_players.png")

# Fig 3: does the movement travel, by group
fig, ax = plt.subplots(figsize=(FIGW, 4.4))
TRs = TRD.sort_values("rho").reset_index(drop=True)
for i, r in TRs.iterrows():
    col = OI["verm"] if r.grp == "DL" else OI["blue"]
    ax.plot([r.lo, r.hi], [i, i], color=col, lw=2.6)
    ax.plot(r.rho, i, "o", color=col, ms=9)
    ax.text(0.62, i, f"{r.rho:+.2f}   n = {r.n}", va="center", fontsize=10.5)
ax.set_yticks(range(len(TRs)), [f"{r.grp}: {MOVE[r.grp]}" for r in TRs.itertuples()])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-0.85, 0.85)
ax.set_xlabel("Same-player rank correlation, Combine drill speed vs NFL speed over the first 1.5 s (95% bootstrap CI)")
ax.set_title("No group's drill start ranks its NFL start well; DL get-off is near zero", fontsize=12.5)
fig.tight_layout()
save(fig, "fig1_travel.png")

# Fig 4: screen heatmap
mat = SC.pivot_table(index=["grp", "tgt"], columns="metric", values="rho").reindex(CELLS)[list(METRICS)]
nn = SC.pivot_table(index=["grp", "tgt"], columns="metric", values="n").reindex(CELLS)[list(METRICS)]
fig, ax = plt.subplots(figsize=(FIGW, 5.6))
im = ax.imshow(mat.values, cmap="PuOr_r", vmin=-0.4, vmax=0.4, aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10.5,
                weight="bold" if abs(v) == abs(top.rho) else "normal", color="white" if abs(v) > 0.3 else "black")
ax.set_xticks(range(3), [METRICS[m] for m in METRICS])
ax.set_yticks(range(len(CELLS)), [f"{LAB[c]} ({MOVE[c[0]]}, n={int(nn.values[i, 0])})" for i, c in enumerate(CELLS)])
ax.set_title(f"DL Carry-Over vs pressure is the top cell, but chance makes one this strong in {round(100 * float(NUM['screen_fw_p']))}% of shuffles",
             fontsize=12.5)
cb = fig.colorbar(im, ax=ax, fraction=0.03)
cb.set_label("Partial rank correlation beyond pick, weight, role, class")
fig.tight_layout()
save(fig, "fig2_screen.png")

# Fig 5: OL trap
fig, axes = plt.subplots(1, 2, figsize=(FIGW, 4.6))
pc = {"T": OI["verm"], "G": OI["sky"], "C": OI["green"]}
for p_, c in pc.items():
    s_ = OLD[OLD.nfl_position == p_]
    axes[0].scatter(s_.d, 100 * s_.actual, color=c, s=40, label=f"{p_} (n={len(s_)})", edgecolor="white", lw=0.5)
axes[0].set_xlabel("Combine kick-slide peak acceleration, first 1.5 s (yd/s²)")
axes[0].set_ylabel("NFL pressure rate allowed (%)")
axes[0].set_title(f"Pooled: rho = {NUM['ol_raw_rho']}; it is the tackles", fontsize=12)
axes[0].legend(frameon=False)
vals = [float(NUM["ol_nopos_rho"]), float(NUM["ol_pos_rho"])]
axes[1].barh([1, 0], vals, color=[OI["grey"], OI["blue"]])
for yy, v in zip([1, 0], vals):
    axes[1].text(v + 0.01, yy, f"{v:+.2f}", va="center", fontsize=11)
axes[1].set_yticks([1, 0], ["Beyond pick, weight", "... and T/G/C share"])
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_xlim(-0.1, 0.55)
axes[1].set_xlabel("Partial rho with pressure allowed")
axes[1].set_title("Add T/G/C share and most of it disappears", fontsize=12)
fig.suptitle("The tackle trap: a Combine OL screen that looked like the best result", fontsize=13, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.92))
save(fig, "fig6_ol_trap.png")

# Fig 6: validation forest
rowsF = [(r.label, r.rho, r.lo, r.hi, r.n) for r in VARD.itertuples()]
rowsF += [("NFL half only (his NFL speed)", float(NUM["dl_rho_game"]), np.nan, np.nan, n),
          ("Combine half only (his drill speed)", float(NUM["dl_rho_drill"]), np.nan, np.nan, n),
          ("Carry-Over given his NFL speed", float(NUM["dl_rho_co_given_game"]), np.nan, np.nan, n)]
fig, ax = plt.subplots(figsize=(FIGW, 5.8))
for i, (lab, r, lo_, hi_, nn_) in enumerate(rowsF):
    yy = len(rowsF) - 1 - i
    col = OI["blue"] if i < len(VARD) else OI["grey"]
    if np.isfinite(lo_):
        ax.plot([lo_, hi_], [yy, yy], color=col, lw=2.4)
    ax.plot(r, yy, "o", color=col, ms=8)
    ax.text(0.72, yy, f"{r:+.2f}  (n={nn_})", va="center", fontsize=10)
ax.set_yticks(range(len(rowsF)), [r[0] for r in rowsF][::-1])
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-0.6, 0.95)
ax.set_xlabel("Partial rank correlation with pressure rate (95% bootstrap CI where shown)")
ax.set_title("Same sign under every baseline and window; the NFL half carries it", fontsize=12.5)
fig.tight_layout()
save(fig, "fig4_checks.png")

# ----------------------------------------------------------------------------- write numbers
neg = sorted({v.lstrip("-") for v in NUM.values() if v.startswith("-")})
NUM_OUT = dict(NUM)
NUM_OUT["_sign_free_copies_of_negative_values"] = neg
json.dump(NUM_OUT, open(OUT / "numbers.json", "w"), indent=1)
VARD.to_csv(OUT / "variants.csv", index=False)
print(json.dumps(NUM, indent=0)[:6000])
print(VARD)
print(SC.sort_values("rho", key=abs, ascending=False).head(8))
