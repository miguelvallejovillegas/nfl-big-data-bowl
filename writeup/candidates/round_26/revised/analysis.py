"""Round 26 revised (revision of round 26 cand 1): DL/EDGE Run-the-Hoop bend (Hoop G) vs NFL bend and pressure.

Run from the repo root:  uv run python writeup/candidates/round_26/revised/analysis.py   (about 13 minutes)

Round 27 revision (judges' round-26 feedback): no-selection ridge over all six drill measures (nested penalty,
held-out R2 gain, bootstrap, whole-procedure shuffle); one joint hierarchical logistic model (career counts + sampled
rushes, rusher random intercept, man/zone coverage and long-yardage per rush, Hoop G measurement-error corrected with
the reliability uncertainty propagated), fitted leave-one-class-out; class-heterogeneity posterior and a 2026
prediction interval; within-class-and-role placebo for the day-3 rank and Brier scores; plain-word tape cues with a
clip per name; one tracking read of a named rep; rebuilt figures.


New in round 26 (round 25 judges' top_changes):
  * Honest selection: the best-of-six measure choice is rerun inside each leave-one-class-out training fold
    (held-out R2 gain of the whole procedure, bootstrap 95%, shuffle p with the selection rerun on every shuffle).
  * Direct drill-to-play test on all sampled rushes: linear-probability mixed models (rusher random intercept) of
    pressure on self-normalized rush lean and on hoop-tight rushes (radius within 2x his own hoop radius), each
    interacted with Hoop G; rusher-cluster bootstrap check; Hoop G vs two alternative per-man NFL bend measures.
  * Stance (listed OLB), snap load (rushes per game) and club pass-rush environment added to the forecast.
  * Day 3 on all 47 held-out day-3 men: pooled rank correlation and out-of-class Brier score, 95% bootstrap.
  * The 7 names the flag adds to pick order's top third, ranked, with a rule-based tape cue (Table 2).
  * Figures: Fig 1 hoop lap vs NFL rushes for four of the added names; Fig 2 eight rows with a colour legend;
    Fig 3 day-3 evaluation + added names (95%); Fig 4 mechanism; Fig A1 merges the screen and both placebos.

Earlier notes, round 25 (round 24 judges' top_changes):
  * Joint model: Hoop G and wave-drill lean together on the held-out pick + role residual (bootstrap 95%, shuffle p,
    held-out R2 gain of each beyond the other).
  * Team-clustered bootstrap of the pick + role R2 gain with a one-sided p; best-of-six adjusted one-sided p for the
    slope and for the R2 gain (six-measure rows shuffled jointly within class and role).
  * Why 2025 is flat: hierarchical model of the class slopes (half-normal prior on the class spread, grid posterior):
    shrunk 2025 slope and predictive 2026 slope; Hoop G x class F test; rookie-season-only and later-season outcomes.
  * Day 3: names the flag adds beyond pick order's top third, simulated 2026 race against pick order, expected 2026
    flag count; play-log verdict table for four rule-chosen names, verdict rules fixed in code.
  * Figures: Fig 1 overlaid hoop laps + NFL small multiples with pressure counts and a takeaway title; Fig 2 one
    full-width 95% forest with n column (mediation panel demoted to appendix text); Fig 3 the ranked film queue with
    beat/par/miss counts, larger sparklines and a 2026 preview strip.

Earlier notes (round 24 revised):

Revision of round 24 cand 1 (judges' round-24 feedback):
  * Headline = the single test written down for 2026 (Hoop G vs pick + role), 95% interval; held-out R2 gain with a
    permutation p from the same within-class-and-role shuffles as the slope placebo (Fig A2).
  * Prospective check: forecast and Hoop G weight fitted on 2023+2024, tested on 2025 (slope, 95%, R2).
  * Controls through the same pipeline: five straight-line DL measures, the same lean computation on the DL wave drill,
    and on the OL long-pull drill against OL pressures allowed (beyond pick + tackle share).
  * Measurement-error model for the Hoop G -> NFL bend -> pressure split: regression calibration with shrunk
    random-effects estimates (lap-to-lap reliability; per-rush NFL bend variance components).
  * Four named players chosen by fixed rules, a tape confirm/kill line each, clip locations (season/week/Q/clock).
  * Figures: Fig 1 small multiples coloured by lean with each path's peak marked; Fig 2 tests + mediation share;
    Fig 3 sparklines and one colour rule shared with Fig 1; placebo moved to Fig A2.
  * BDB_RENDER_ONLY=1 re-renders writeup.md from the last run's numbers.json without recomputing.

Earlier notes (round 24 cand 1):

New in round 24 (round 23 judges' top_changes):
  * Headline comparator = pick + edge role (the forecast frozen for the 2026 test); every residual analysis (six-measure
    screen, classes, roles, simulation, leave-one-out, variants, day-3 test, board) is scored against it. Beside it:
    pick only and a ridge-shrunk full no-tracking forecast (penalty chosen inside the training classes). The plain-OLS
    full forecast and the fold-chosen baseline move to the appendix.
  * Team-cluster bootstrap (resample NFL teams) of the held-out R2 gain, next to the within-class player bootstrap.
  * Placebo: Hoop G shuffled within class, and within class x role (5,000 shuffles each): null slope distribution.
  * Mediation on the same rushers: Hoop G -> NFL bend -> pressure residual (indirect a*b, direct c'), class bootstrap.
  * Play-log check (not film) for the flagged men: alignment, down, formation, play-by-play text of corner turns.
  * Monday rule numbers: expected 2026 flags per class, miss share and the average shortfall of a miss.
  * EPA filter traced: 1,681 sampled rusher-rows -> 1,655 plays (26 plays had two sampled linemen).
  * Figures: hero overlays the three hoop laps; Fig 2 = placebo + mediation; film-list chart without clipping.

Run from the repo root:
    uv run python writeup/candidates/round_24/revised/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json (every number the text quotes), board.csv,
multiplicity.csv, frozen_rule.json and figures/*.png next to this file, runs verify.py (an independent
re-implementation of the headline slope on the same rusher set and fill rule) and quotes its result, then renders
writeup.md from writeup_template.md ({{key}} -> numbers.json value; fails on an unknown key). Frame-level features
are cached in data/cache/r24c1 (gitignored). Runtime about 6 minutes.

New in this round (round 22 judges' top_changes):
  * One rusher set (30+ NFL rushes, all six drill measures valid) and one fill rule (training-fold median for a
    missing 10-yd split / 3-cone) in both analysis.py and verify.py.
  * Three baselines side by side (pick only, pick + role, full no-tracking), plus a nested baseline chosen inside each
    training fold, each with the held-out R2 gain of Hoop G (class-stratified player bootstrap that refits every
    fold), and a repeated random-split check (200 random three-way splits).
  * Measurement-error simulation: true slope grid x one-lap reliability (with its bootstrap uncertainty) -> the
    distribution of the observed slope; implied true-slope range, attenuation, power for one lap and four laps.
  * Film check on sampled rushes: corner-turn rate (share of a lineman's sampled rushes whose hardest half-second is
    in the top quarter of all sampled rushes) by Hoop G third, and pressure on corner-turn vs other rushes.
  * Film list in Hoop G order with the NFL result beside each name, split into open-first and missed.
  * Sources of the EPA-per-pressure figure and the rush counts are written to numbers.json.

Design (one estimand per claim):
  * Drill metrics, z within Combine year, mean over attempts:
      Hoop G  = hardest 0.5 s of sideways (centripetal) acceleration on the Run-the-Hoop lap, g.
      Burst   = yards covered in the first second after movement onset in the pass-rush drill.
  * Game traits (random sampled NFL pass rushes only, self-referenced, no blockers or ball in the data):
      NFL get-off   = NGS season get-off (rush-weighted), sign flipped so higher = quicker.
      NFL burst     = yards covered in the first second after the snap.
      NFL bend      = hardest 0.5 s of sideways acceleration in the first 2.0 s after the snap, g.
  * Outcome: career pass-rush pressure rate 2023-25, empirical-Bayes shrunk, linemen with 30+ rushes.
  * Baseline without tracking (pre-specified in the project rules): log pick, Combine weight, NFL EDGE share, NGS
    athleticism score, official 10-yd split and 3-cone (missing times = training-fold median). Leave-one-class-out
    OLS. Reported beside pick only, pick + role, and a nested choice among the three made inside each training fold.
  * One pre-set margin: 0.5 pressure-rate points per sd (2.25 pressures per 450-rush season).
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
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy import stats
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[4]
PROC = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
for _old in ([] if os.environ.get("BDB_RENDER_ONLY") else FIG.glob("*.png")):
    _old.unlink()
CACHE = ROOT / "data" / "cache" / "r24c1"    # same frame features as round 24 cand 1 (gitignored)
CACHE.mkdir(parents=True, exist_ok=True)

NBOOT = int(os.environ.get("BDB_NBOOT", 1000))
NPERM = 5000
RATE_MIN, REG_MIN, DAY3, UDFA_PICK = 30, 100, 103, 262
MARGIN = 0.5            # pressure-rate points per sd: smallest effect worth acting on, set before fitting
SEASON = 450            # rushes in a regular's season
PS = SEASON / 100       # pressure-rate points -> pressures per 450-rush season
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


def render(num, txt):
    """writeup_template.md -> writeup.md: {{key}} = numbers.json value, {{T:key}} = text_bits.json value."""
    tpl = OUT / "writeup_template.md"
    if not tpl.exists():
        return

    def sub(m_):
        k_ = m_.group(1)
        if k_.startswith("T:"):
            v_ = txt[k_[2:].split("[")[0]]
            if "[" in k_:
                v_ = v_[int(k_.split("[")[1].rstrip("]"))]
            return str(v_)
        if k_ not in num:
            raise KeyError(k_)
        v_ = str(num[k_])
        return v_.replace("-", "−") if v_.startswith("-") else v_
    (OUT / "writeup.md").write_text(re.sub(r"\{\{([^}]+)\}\}", sub, tpl.read_text()))


if os.environ.get("BDB_RENDER_ONLY"):      # re-render the text from the last full run's numbers, no recomputation
    render(json.loads((OUT / "numbers.json").read_text()), json.loads((OUT / "text_bits.json").read_text()))
    raise SystemExit(0)

NUM: dict = {}
PB: dict = {}   # bootstrap two-sided p-values of every partial-Spearman link, for the multiplicity table


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
    b = np.asarray(b)
    PB[(x, y, len(d))] = float(min(1.0, 2 * min((1 + (b <= 0).sum()), (1 + (b >= 0).sum())) / (len(b) + 1)))
    return r, float(lo), float(hi), int(len(d))


def fm(x, nd=1):
    t_ = f"{x:+.{nd}f}"
    if float(t_) == 0:
        t_ = f"{0:.{nd}f}"
    return t_.replace("-", "−")


# ============================================================================= load
players = pd.read_parquet(PROC / "players.parquet")
cres = pd.read_parquet(PROC / "combine_results.parquet")
events = pd.read_parquet(PROC / "combine_events.parquet")
seasons = pd.read_parquet(PROC / "player_season_stats.parquet")
align = pd.read_parquet(PROC / "player_season_alignment.parquet")
gp = pd.read_parquet(PROC / "game_plays_sample.parquet")

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
classes = [2023, 2024, 2025]
put("n_dl", len(dl)); put("margin", MARGIN, 1)
put("margin_season", MARGIN * PS, 2); put("season_rushes", SEASON)
put("n_three_cone", int(dl.three_cone.notna().sum()))
put("pct_day3_dl", 100 * (dl.pick >= DAY3).mean(), 0)

# ============================================================================= Combine drill kinematics
ev = events[(events.position_group == "DL") & events.drill_name.isin(["PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL"])]
ev = ev[["event_id", "nfl_id", "draft_year", "drill_name", "attempt_no", "peak_accel", "top_speed", "duration_s"]]


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
rush_att = att[att.drill_name == "PASS_RUSH_DRILL"].copy()
hoop_att["z"] = yz(hoop_att.bend_g, hoop_att.draft_year)
rush_att["z"] = yz(rush_att.d1, rush_att.draft_year)
rel_hoop1, n_hrep = icc(hoop_att.nfl_id, hoop_att.z)
rel_d1_1, n_drep = icc(rush_att.nfl_id, rush_att.z)
REL = {"hoop_z": sb(rel_hoop1, hoop_att.groupby("nfl_id").size().mean()),
       "burst_z": sb(rel_d1_1, rush_att.dropna(subset=["d1"]).groupby("nfl_id").size().mean())}
put("hoop_g_median", hoop_att.bend_g.median())
put("hoop_speed_median", hoop_att.speed_at_bend.median(), 1)
# coach version: Hoop G = speed^2 / turn radius, so the median runner's radius at his hardest lean
put("hoop_radius_median", (hoop_att.speed_at_bend ** 2 / (hoop_att.bend_g * YD_PER_G)).median(), 1)
put("rel_hoop1", rel_hoop1); put("rel_hoop_nrep", n_hrep); put("rel_hoop2", sb(rel_hoop1, 2))
put("rel_burst1", rel_d1_1); put("rel_burst_nrep", n_drep)
put("burst_median", rush_att.d1.median())


# ---- drill reliability as an evidence base for the rep prescription.
# (1) lap-to-lap ICC (few repeaters) with a player bootstrap; (2) a well-powered internal check: the hoop is a
# figure-eight, so every lap has a left-hand and a right-hand turn. Lean on one turn vs lean on the other, z within
# year, across all runners = split-half reliability of one lap (Spearman-Brown to full lap).
def hoop_turns():
    fr_ = pd.read_parquet(PROC / "combine_frames.parquet", filters=[("event_id", "in", list(hoop_att.event_id))],
                          columns=["event_id", "frame", "entity_type", "x", "y"])
    fr_ = fr_[fr_.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
    out_ = []
    for eid, g_ in fr_.groupby("event_id", sort=False):
        x_, y_ = g_.x.to_numpy(), g_.y.to_numpy()
        if len(x_) < 15:
            continue
        xs, ys = savgol_filter(x_, 7, 2), savgol_filter(y_, 7, 2)
        vx, vy = savgol_filter(xs, 7, 2, deriv=1, delta=DT), savgol_filter(ys, 7, 2, deriv=1, delta=DT)
        ax_, ay_ = savgol_filter(xs, 7, 2, deriv=2, delta=DT), savgol_filter(ys, 7, 2, deriv=2, delta=DT)
        sgn_ = (vx * ay_ - vy * ax_) / np.maximum(np.hypot(vx, vy), 0.5) / YD_PER_G
        lft = pd.Series(np.where(sgn_ > 0, sgn_, 0)).rolling(5, center=True).mean().max()
        rgt = pd.Series(np.where(sgn_ < 0, -sgn_, 0)).rolling(5, center=True).mean().max()
        out_.append({"event_id": eid, "left_g": float(lft), "right_g": float(rgt)})
    return pd.DataFrame(out_)


HT = cached("hoop_turns", hoop_turns).merge(hoop_att[["event_id", "nfl_id", "draft_year"]], on="event_id")
HT["lz"], HT["rz"] = yz(HT.left_g, HT.draft_year), yz(HT.right_g, HT.draft_year)
HTP = HT.groupby("nfl_id")[["lz", "rz"]].mean().reset_index()
_rr = np.random.default_rng(404)
r_lr = stats.spearmanr(HTP.lz, HTP.rz)[0]
_blr = [stats.spearmanr(*HTP.iloc[_rr.integers(0, len(HTP), len(HTP))][["lz", "rz"]].values.T)[0] for _ in range(NBOOT)]
REL_TURN = sb(r_lr, 2)
put("rel_turn_half", r_lr); put("rel_turn_n", len(HTP)); put("rel_turn_lap", REL_TURN)
put("rel_turn_lap_lo", sb(np.percentile(_blr, 5), 2)); put("rel_turn_lap_hi", sb(np.percentile(_blr, 95), 2))
BLR_LAP = np.array([sb(max(b_, 0.05), 2) for b_ in _blr])   # bootstrap draws of one-lap reliability
# left vs right asymmetry: Hoop G takes the harder of the two turns of the lap (max over the whole lap)
put("pct_left_harder", 100 * (HT.left_g > HT.right_g).mean(), 0)
put("turn_gap_g", (HT.left_g - HT.right_g).abs().median())
HT["mean_turn"] = (HT.left_g + HT.right_g) / 2


def icc_boot(df_, seed):
    d_ = df_.dropna(subset=["z"])
    ids_ = d_.groupby("nfl_id").size()
    rep = ids_[ids_ >= 2].index.values
    vals = {p_: d_.loc[d_.nfl_id == p_, "z"].to_numpy() for p_ in rep}
    rg_ = np.random.default_rng(seed)
    out_ = []
    for _ in range(NBOOT):
        pick_ = rg_.choice(rep, len(rep))
        ii = np.concatenate([[i_] * len(vals[p_]) for i_, p_ in enumerate(pick_)])
        vv = np.concatenate([vals[p_] for p_ in pick_])
        with np.errstate(all="ignore"):
            out_.append(icc(ii, vv)[0])
    return np.nanpercentile(out_, [5, 95])


ICC_CI = {"hoop": icc_boot(hoop_att, 405), "burst": icc_boot(rush_att.rename(columns={}), 406)}
put("rel_hoop1_lo", ICC_CI["hoop"][0]); put("rel_hoop1_hi", ICC_CI["hoop"][1])
put("rel_burst1_lo", ICC_CI["burst"][0]); put("rel_burst1_hi", ICC_CI["burst"][1])
REL_TARGET = 0.70


def reps_needed(r1, target=REL_TARGET):
    return int(np.ceil(target * (1 - r1) / (r1 * (1 - target))))


put("rel_target", REL_TARGET); put("reps_hoop", reps_needed(rel_hoop1)); put("reps_burst", reps_needed(rel_d1_1))
put("reps_hoop_turn", reps_needed(REL_TURN))
put("n_burst_one_rep", int((rush_att.dropna(subset=["d1"]).groupby("nfl_id").size() == 1).sum()))

# the pre-listed family: three measures per drill, all mean over attempts, z within Combine year
FAM = [("burst_z", "Rush drill: burst (yd in first 1 s)", rush_att, "d1", 1),
       ("rpa_z", "Rush drill: peak acceleration", rush_att, "peak_accel", 1),
       ("rts_z", "Rush drill: top speed", rush_att, "top_speed", 1),
       ("hoop_z", "Hoop: Hoop G (sideways lean)", hoop_att, "bend_g", 1),
       ("hts_z", "Hoop: top speed", hoop_att, "top_speed", 1),
       ("hlap_z", "Hoop: lap time (faster = higher)", hoop_att, "duration_s", -1)]
for key, _, src, col, sgn in FAM:
    pp = src.groupby("nfl_id").agg(v=(col, "mean"), draft_year=("draft_year", "first")).reset_index()
    pp[key] = sgn * yz(pp.v, pp.draft_year)
    dl = dl.merge(pp[["nfl_id", key]], on="nfl_id", how="left")
hoop_att["rad"] = hoop_att.speed_at_bend ** 2 / (hoop_att.bend_g * YD_PER_G)
hk = hoop_att.groupby("nfl_id").agg(hoop_g=("bend_g", "mean"), hoop_k=("bend_g", "size"),
                                    hoop_eid=("event_id", "first"), hoop_rad=("rad", "mean"),
                                    hoop_v=("speed_at_bend", "mean")).reset_index()
dl = dl.merge(hk, on="nfl_id", how="left")
_mt = HT.groupby("nfl_id").agg(v=("mean_turn", "mean"), draft_year=("draft_year", "first")).reset_index()
_mt["meanturn_z"] = yz(_mt.v, _mt.draft_year)
dl = dl.merge(_mt[["nfl_id", "meanturn_z"]], on="nfl_id", how="left")
put("n_hoop", int(dl.hoop_z.notna().sum())); put("n_burst", int(dl.burst_z.notna().sum()))
put("n_hoop_no3c", int((dl.hoop_z.notna() & dl.three_cone.isna()).sum()))
put("pct_one_run", 100 * (dl.hoop_k == 1).sum() / dl.hoop_z.notna().sum(), 0)
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
s25 = s[s.season == 2025].groupby("nfl_id").agg(r25=("pass_rush_snaps", "sum"), p25=("pressures", "sum"))
dl = dl.merge(s25.reset_index(), on="nfl_id", how="left")

# ============================================================================= sampled NFL tracking
dlids = set(dl.nfl_id)
K = ["game_id", "play_id", "nfl_id"]
g = gp[gp.nfl_id.isin(dlids)].drop_duplicates(K)
pinfo = g.set_index(K)[["sample_kind", "time_to_pressure", "sack", "player_get_off", "expected_points_added"]]


def game_features():
    gf = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "x", "y", "event"])
    gf = gf[gf.nfl_id.isin(dlids)].merge(g[K + ["snap_frame"]], on=K)
    ends = gf[gf.event.isin(["pass_forward", "qb_sack", "qb_strip_sack", "run", "handoff"])].groupby(K).frame.min()
    gf = gf.join(ends.rename("end_frame"), on=K).sort_values(K + ["frame"])
    out = []
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
            _, roll, _, _, _ = lateral_g(x_[w], y_[w])
            bend = float(np.nanmax(roll))
        out.append(dict(zip(K, key), g_d1=d1, g_bend=bend, kind=info.sample_kind,
                        press=int(pd.notna(info.time_to_pressure) or info.sack == 1)))
    return pd.DataFrame(out)


GF = cached("game_fixed2s", game_features)
GR = GF[GF.kind == "random"].copy()
GR["half"] = GR.groupby("nfl_id").cumcount() % 2
put("n_random_rushes", len(GR)); put("n_random_players", GR.nfl_id.nunique())


def split_half(df, col, min_n):
    cnt = df.groupby("nfl_id")[col].count()
    keep = cnt[cnt >= min_n].index
    a_ = df[df.nfl_id.isin(keep)].groupby(["nfl_id", "half"])[col].mean().unstack().dropna()
    r_ = stats.spearmanr(a_[0], a_[1])[0]
    return 2 * r_ / (1 + r_), len(a_)


REL["g_d1"], _ = split_half(GR, "g_d1", 8)
REL["g_bend"], n_rb = split_half(GR, "g_bend", 8)
put("rel_game_burst", REL["g_d1"]); put("rel_game_bend", REL["g_bend"]); put("rel_game_n", n_rb)
PG = GR.groupby("nfl_id").agg(g_d1=("g_d1", "mean"), g_bend=("g_bend", "mean"), g_n=("g_bend", "count")).reset_index()
PG = PG[PG.g_n >= 8]
dl = dl.merge(PG, on="nfl_id", how="left")
put("median_random_rushes", PG.g_n.median(), 0)
# which linemen the rush count refers to: 8+ randomly sampled rushes that last the full 2.0 s window
put("n_pg", len(PG)); put("n_rush_2s", int(GR.g_bend.notna().sum()))
put("n_pg_co30", int(PG.nfl_id.isin(dl.loc[dl.rush_snaps >= RATE_MIN, "nfl_id"]).sum()))
put("median_random_rushes_co30", PG[PG.nfl_id.isin(dl.loc[dl.rush_snaps >= RATE_MIN, "nfl_id"])].g_n.median(), 0)
put("median_random_any", GR.groupby("nfl_id").size().median(), 0)

# ---- film check on sampled rushes. A "corner turn" is a sampled rush whose hardest half-second of sideways lean
# (first 2.0 s) is in the top quarter of all sampled rushes: the dip-and-flatten a scout looks for, self-referenced.
# Does a high Hoop G lineman turn the corner more often, and does a corner-turn rush end in pressure more often?
CT = float(GR.g_bend.quantile(0.75))
GR["corner"] = (GR.g_bend >= CT).astype(float).where(GR.g_bend.notna())
put("ct_g", CT)
_cr = GR.groupby("nfl_id").agg(corner=("corner", "mean"), c_k=("corner", "sum"), c_n=("corner", "count")).reset_index()
_cr = _cr[_cr.c_n >= 8]
dl = dl.merge(_cr, on="nfl_id", how="left")
_ct = dl.dropna(subset=["corner", "hoop_z"]).copy()
_ct["third"] = _ct.groupby("draft_year").hoop_z.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
_cg = [np.flatnonzero(_ct.draft_year.values == c) for c in classes]
_crng = np.random.default_rng(515)


def _ct_gap(d_):
    return 100 * (d_[d_.third == 2].corner.mean() - d_[d_.third == 0].corner.mean())


_cb = [_ct_gap(_ct.iloc[np.concatenate([_crng.choice(g_, len(g_)) for g_ in _cg])]) for _ in range(NBOOT)]
put("ct_n", len(_ct)); put("ct_top", 100 * _ct[_ct.third == 2].corner.mean(), 0)
put("ct_bot", 100 * _ct[_ct.third == 0].corner.mean(), 0); put("ct_gap", _ct_gap(_ct), 0)
put("ct_gap_lo", np.percentile(_cb, 5), 0); put("ct_gap_hi", np.percentile(_cb, 95), 0)
PB[("film", "ct_gap")] = float(min(1.0, 2 * min(1 + (np.asarray(_cb) <= 0).sum(), 1 + (np.asarray(_cb) >= 0).sum()) / (NBOOT + 1)))
# rush level: pressure on corner-turn rushes vs other rushes, player-cluster bootstrap
_rk = GR.dropna(subset=["corner"])
_rk = _rk[_rk.nfl_id.isin(_cr.nfl_id)]
_ids = _rk.nfl_id.unique()
_byp = {i: _rk[_rk.nfl_id == i][["corner", "press"]].values for i in _ids}


def _pgap(arr):
    c_ = arr[:, 0] == 1
    return 100 * (arr[c_, 1].mean() - arr[~c_, 1].mean())


_all = np.vstack([_byp[i] for i in _ids])
_pb = [_pgap(np.vstack([_byp[i] for i in _crng.choice(_ids, len(_ids))])) for _ in range(NBOOT)]
put("ct_rush_n", len(_all)); put("ct_press_corner", 100 * _all[_all[:, 0] == 1, 1].mean(), 0)
put("ct_press_other", 100 * _all[_all[:, 0] == 0, 1].mean(), 0)
put("ct_press_lo", np.percentile(_pb, 5), 0); put("ct_press_hi", np.percentile(_pb, 95), 0)
PB[("film", "ct_press")] = float(min(1.0, 2 * min(1 + (np.asarray(_pb) <= 0).sum(), 1 + (np.asarray(_pb) >= 0).sum()) / (NBOOT + 1)))


# ---- drill turn vs NFL rush path, in coach units. g = speed^2 / radius on both sides, at the half-second of
# hardest lean: the hoop lap, and each randomly sampled NFL rush (first 2.0 s). Turn Match = his NFL turn radius
# divided by his own hoop turn radius (1 = he turns as tight on Sunday as on the hoop).
def game_turn():
    keys = GR[GR.g_bend.notna()][K]
    gf_ = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "x", "y"],
                          filters=[("nfl_id", "in", list(set(keys.nfl_id)))])
    gf_ = gf_.merge(keys, on=K).merge(g[K + ["snap_frame"]], on=K).sort_values(K + ["frame"])
    out_ = []
    for key, h in gf_.groupby(K, sort=False):
        sf = h.snap_frame.iloc[0]
        w = h[(h.frame >= sf) & (h.frame <= sf + 20)]
        if len(w) != 21:
            continue
        _, roll, _, _, v = lateral_g(w.x.values, w.y.values)
        k_ = int(np.nanargmax(roll))
        out_.append(dict(zip(K, key), t_bend=float(np.nanmax(roll)), t_v=float(v[k_]), t_k=k_))
    return pd.DataFrame(out_)


GT = cached("game_turn", game_turn)
GT["t_rad"] = GT.t_v ** 2 / (GT.t_bend * YD_PER_G)
put("nfl_mph_median", GT.t_v.median() * 3600 / 1760, 1); put("nfl_radius_median", GT.t_rad.median(), 1)
put("nfl_bend_t_median", GT.t_k.median() / 10, 1)
_gt = GT.groupby("nfl_id").agg(nfl_rad=("t_rad", "median"), nfl_v=("t_v", "median"), t_cnt=("t_rad", "size"))
_gt = _gt[_gt.t_cnt >= 8].reset_index()
dl = dl.merge(_gt[["nfl_id", "nfl_rad", "nfl_v"]], on="nfl_id", how="left")
dl["turn_match"] = dl.nfl_rad / dl.hoop_rad
_tm = dl.dropna(subset=["turn_match"])
put("turn_match_median", _tm.turn_match.median(), 1); put("turn_match_n", len(_tm))
put("turn_match_p10", _tm.turn_match.quantile(0.1), 1); put("turn_match_p90", _tm.turn_match.quantile(0.9), 1)
# share of all sampled NFL rushes whose hardest turn is as tight as the same man's own hoop turn
_gm2 = GT.merge(dl[["nfl_id", "hoop_rad"]], on="nfl_id").dropna(subset=["hoop_rad"])
put("pct_rush_hooptight", 100 * (_gm2.t_rad <= _gm2.hoop_rad).mean(), 0)
dl["hoop_rad_tight"] = -dl.hoop_rad
dl["nfl_rad_tight"] = -dl.nfl_rad
_s2 = s[s.pass_rush_snaps >= 100].pivot_table(index="nfl_id", columns="season", values="avg_get_off")
_pp = pd.concat([pd.DataFrame({"a": _s2[a_], "b": _s2[b_]}).dropna() for a_, b_ in [(2023, 2024), (2024, 2025)]])
REL["quick"] = stats.spearmanr(_pp.a, _pp.b)[0]
put("rel_getoff_season", REL["quick"]); put("rel_getoff_n", len(_pp))
_s3 = s[s.pass_rush_snaps >= 100].assign(pr=lambda d_: d_.pressures / d_.pass_rush_snaps).pivot_table(
    index="nfl_id", columns="season", values="pr")
_pq = pd.concat([pd.DataFrame({"a": _s3[a_], "b": _s3[b_]}).dropna() for a_, b_ in [(2023, 2024), (2024, 2025)]])
put("rel_press_season", stats.spearmanr(_pq.a, _pq.b)[0]); put("rel_press_n", len(_pq))
# EPA of a pressured snap (random sampled snaps)
_gm = g[(g.sample_kind == "random") & g.expected_points_added.notna()].drop_duplicates(["game_id", "play_id"])
_pr = _gm.time_to_pressure.notna() | (_gm.sack == 1)
e1, e0 = _gm[_pr].expected_points_added.values, _gm[~_pr].expected_points_added.values
_r = np.random.default_rng(101)
_eb = [_r.choice(e0, len(e0)).mean() - _r.choice(e1, len(e1)).mean() for _ in range(NBOOT)]
EPA = e0.mean() - e1.mean()
put("epa_press", EPA); put("epa_press_lo", np.percentile(_eb, 5)); put("epa_press_hi", np.percentile(_eb, 95))
put("epa_n", len(_gm))
# source: play-level time_to_pressure (non-null = the rusher was credited with a pressure) or sack == 1 on the
# randomly sampled DL plays, nflverse expected_points_added; pressure_allowed is a blocker column (empty for DL rows)
put("epa_n_press", int(_pr.sum())); put("pct_dl_pressure_allowed_null", 100 * g.pressure_allowed.isna().mean(), 0)
# filter trace: rusher-rows before the one-row-per-play step, and the any-rusher pressure flag per play
_gr = g[(g.sample_kind == "random") & g.expected_points_added.notna()]
_grp = _gr.time_to_pressure.notna() | (_gr.sack == 1)
put("epa_rows", len(_gr)); put("epa_rows_press", int(_grp.sum()))
put("epa_dup_plays", len(_gr) - len(_gm))
_any = _gr.assign(p_=_grp).groupby(["game_id", "play_id"]).agg(p_=("p_", "max"), e_=("expected_points_added", "first"))
put("epa_any_press", int(_any.p_.sum()))
put("epa_press_any", _any[~_any.p_].e_.mean() - _any[_any.p_].e_.mean())
put("epa_press_rows", _gr[~_grp].expected_points_added.mean() - _gr[_grp].expected_points_added.mean())
put("margin_epa", MARGIN * PS * EPA, 1)

# ============================================================================= the chain: drill -> NFL trait -> pressure
Zc = ["combine_weight_z", "edge_share"]
co30 = dl[dl.rush_snaps >= RATE_MIN].copy()
CHAIN = {}
for dx, gy, seed in [("burst_z", "quick", 11), ("burst_z", "g_d1", 12), ("hoop_z", "g_bend", 13),
                     ("hoop_z", "quick", 14)]:
    r, lo, hi, n = boot_psp(dl, dx, gy, Zc, seed)
    CHAIN[(dx, gy)] = (r, lo, hi, n)
    tag = {"burst_z": "burst", "hoop_z": "hoop"}[dx] + "_" + gy
    put(f"tr_{tag}", r); put(f"tr_{tag}_lo", lo); put(f"tr_{tag}_hi", hi); put(f"tr_{tag}_n", n)
    put(f"tr_{tag}_ceil", np.sqrt(REL[dx] * REL[gy]))
# no-tracking benchmark for each transfer link: the official time that should carry the same trait
dl["three_cone_fast"] = -dl.three_cone_z
dl["split_fast"] = -dl.ten_yd_split_z
for bx, gy, tag, seed in [("three_cone_fast", "g_bend", "3c_bend", 15), ("split_fast", "quick", "split_quick", 16),
                          ("three_cone_fast", "quick", "3c_quick", 17)]:
    r, lo, hi, n = boot_psp(dl, bx, gy, Zc, seed)
    put(f"bm_{tag}", r); put(f"bm_{tag}_lo", lo); put(f"bm_{tag}_hi", hi); put(f"bm_{tag}_n", n)
# Hoop G -> NFL bend on the 3-cone runners only, so both columns use the same linemen
_r = boot_psp(dl[dl.three_cone_z.notna()], "hoop_z", "g_bend", Zc, 18)
put("tr_hoop_bend_3csub", _r[0]); put("tr_hoop_bend_3csub_n", _r[3])
_r = boot_psp(dl, "hoop_rad_tight", "nfl_rad_tight", Zc, 19)
put("tr_rad", _r[0]); put("tr_rad_lo", _r[1]); put("tr_rad_hi", _r[2]); put("tr_rad_n", _r[3])
for gy, seed in [("quick", 21), ("g_d1", 22), ("g_bend", 23)]:
    r, lo, hi, n = boot_psp(co30, gy, "press_eb", Zc, seed)
    put(f"gp_{gy}", r); put(f"gp_{gy}_lo", lo); put(f"gp_{gy}_hi", hi); put(f"gp_{gy}_n", n)

# ============================================================================= held-out baseline and increments
MAIN = ["log_pick", "combine_weight_z", "edge_share", "ngs_athleticism_score"]
TIMES = ["ten_yd_split_z", "three_cone_z"]
BASE = MAIN + TIMES
FILLX = ["combine_weight_z", "edge_share", "ngs_athleticism_score"]
dl["ngs_athleticism_score"] = dl.ngs_athleticism_score.fillna(dl.ngs_athleticism_score.median())
co30 = dl[dl.rush_snaps >= RATE_MIN].copy()


def fill_times(tr, te, how="median"):
    tr, te = tr.copy(), te.copy()
    for c in TIMES:
        mk = tr[c].notna()
        if how == "median":       # the one fill rule (also verify.py's): training-fold median
            med = tr.loc[mk, c].median()
            med = 0.0 if pd.isna(med) else med     # a training class with no times at all: year-z mean
            tr[c], te[c] = tr[c].fillna(med), te[c].fillna(med)
            continue
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


ALPHAS = np.r_[1e-4, np.logspace(-1, 2.5, 8)]   # ridge penalty, in units of training rushers


def ridge(X, yv, Xt, alpha):
    mu_, sd_ = X.mean(0), X.std(0) + 1e-9
    Xs, Xts = (X - mu_) / sd_, (Xt - mu_) / sd_
    b = np.linalg.solve(Xs.T @ Xs + alpha * len(Xs) * np.eye(X.shape[1]), Xs.T @ (yv - yv.mean()))
    return yv.mean() + Xts @ b


def ridge_cv(tr, te, cols):
    """penalty chosen inside the training classes only: fit on one training class, score on the other."""
    X, yv, tcl = tr[cols].values, tr.press_eb.values, tr.draft_year.values
    us = np.unique(tcl)
    err = []
    for a_ in ALPHAS:
        e_ = 0.0
        for c in us:
            m_ = tcl == c
            e_ += ((yv[m_] - ridge(X[~m_], yv[~m_], X[m_], a_)) ** 2).sum()
        err.append(e_)
    return ridge(X, yv, te[cols].values, ALPHAS[int(np.argmin(err))])


def loco(d, cols, kind="ols", how="median"):
    pred = np.full(len(d), np.nan)
    cl = d.draft_year.values
    for c in classes:
        tr, te = fill_times(d[cl != c], d[cl == c], how)
        pred[cl == c] = ols(tr, te, cols)[0] if kind == "ols" else ridge_cv(tr, te, cols)
    return pred


def oos(y, p):
    sse = ((y - p) ** 2).sum()
    return 1 - sse / ((y - y.mean()) ** 2).sum(), np.sqrt(sse / len(y)), stats.spearmanr(y, p)[0]


FKEYS = [f[0] for f in FAM]
AN = co30.dropna(subset=FKEYS).reset_index(drop=True)
put("n_an", len(AN))
NUM["ho_n_per_class"] = "/".join(str(int((AN.draft_year == c).sum())) for c in classes)
BASES = {"pick1": ["log_pick"], "pick": ["log_pick", "edge_share"], "base": BASE}
BLAB = {"pick1": "Pick only", "pick": "Pick + role", "base": "Full no-tracking", "nest": "Chosen inside each fold"}


def loco_nested(d, extra=(), how="median"):
    """outer leave-one-class-out; inside the two training classes, pick the baseline (of the three) that forecasts
    one training class best from the other, then fit it (+ extra columns) on both training classes."""
    pred = np.full(len(d), np.nan)
    cl = d.draft_year.values
    chosen = []
    for c in classes:
        tr0 = d[cl != c]
        tcl = tr0.draft_year.values
        err = {}
        for bk, cols in BASES.items():
            e_ = 0.0
            for ic in np.unique(tcl):
                a_, b_ = fill_times(tr0[tcl != ic], tr0[tcl == ic], how)
                e_ += ((b_.press_eb.values - ols(a_, b_, cols)[0]) ** 2).sum()
            err[bk] = e_
        bk = min(err, key=err.get)
        chosen.append(bk)
        tr, te = fill_times(tr0, d[cl == c], how)
        pred[cl == c] = ols(tr, te, BASES[bk] + list(extra))[0]
    return pred, chosen


HEADK = "pick"                       # headline comparator: pick + edge role (frozen for the 2026 test in round 23)
HEADC = BASES[HEADK]
MODELS = [(bk, BASES[bk], False) for bk in BASES] + [(bk + "_h", BASES[bk] + ["hoop_z"], False) for bk in BASES] \
    + [("burst_h", BASE + ["burst_z"], False), ("nest", [], True), ("nest_h", ["hoop_z"], True),
       ("ridge", BASE, "ridge"), ("ridge_h", BASE + ["hoop_z"], "ridge")]


def predict(d, cols, nested):
    if nested == "ridge":
        return loco(d, cols, kind="ridge")
    return loco_nested(d, cols)[0] if nested else loco(d, cols)


y = AN.press_eb.values
PRED = {nm: predict(AN, cols, nz) for nm, cols, nz in MODELS}
NEST_CHOSEN = loco_nested(AN)[1]
TXT_NEST = {c: BLAB[k] for c, k in zip(classes, NEST_CHOSEN)}
RES = y - PRED[HEADK]
AN["fc"] = PRED[HEADK]
grp = [np.flatnonzero(AN.draft_year.values == c) for c in classes]
rng = np.random.default_rng(2027)
BOOT = {nm: [] for nm, *_ in MODELS}
BRES = {k: [] for k in FKEYS}
BSL = {k: [] for k in ["pick1", "pick", "base", "nest", "ridge"]}     # Hoop G slope on each baseline's residual
BX = {k: [] for k in ["cls2023", "cls2024", "cls2025", "EDGE", "interior", "meanturn"]}
AN["rel_i"] = sb(REL_TURN, AN.hoop_k)          # reliability of each lineman's own Hoop G (one or two laps)
NEST_B = []
for _ in range(NBOOT):
    ix = np.concatenate([rng.choice(g_, len(g_)) for g_ in grp])
    d_ = AN.iloc[ix].reset_index(drop=True)
    yb = d_.press_eb.values
    pb_ = {}
    for nm, cols, nz in MODELS:
        if nm == "nest":
            p_, ch_ = loco_nested(d_)
            NEST_B.append(ch_)
        else:
            p_ = predict(d_, cols, nz)
        pb_[nm] = p_
        BOOT[nm].append(oos(yb, p_)[0])
    for bk in BSL:
        BSL[bk].append(np.polyfit(d_.hoop_z.values, yb - pb_[bk], 1)[0])
    rb = yb - pb_[HEADK]
    for k in FKEYS:
        BRES[k].append(np.polyfit(d_[k].values, rb, 1)[0])
    for c in classes:
        mc = d_.draft_year.values == c
        BX[f"cls{c}"].append(np.polyfit(d_.hoop_z.values[mc], rb[mc], 1)[0])
    for rl in ("EDGE", "interior"):
        mr = d_.role.values == rl
        BX[rl].append(np.polyfit(d_.hoop_z.values[mr], rb[mr], 1)[0])
    BX["meanturn"].append(np.polyfit(d_.meanturn_z.values, rb, 1)[0])
for nm, *_ in MODELS:
    r2, rmse, rho = oos(y, PRED[nm])
    put(f"r2_{nm}", r2); put(f"rmse_{nm}", rmse)
    ref = nm[:-2] if nm.endswith("_h") else None
    if ref:
        ref = "base" if nm == "burst_h" else ref
        dd = np.array(BOOT[nm]) - np.array(BOOT[ref])
        put(f"dr2_{nm}", r2 - oos(y, PRED[ref])[0])
        put(f"dr2_{nm}_lo", np.percentile(dd, 5)); put(f"dr2_{nm}_hi", np.percentile(dd, 95))
        PB[("dr2", nm)] = float(min(1.0, 2 * min(1 + (dd <= 0).sum(), 1 + (dd >= 0).sum()) / (len(dd) + 1)))
        if nm == "base_h":   # Bonferroni across the six pre-listed drill measures: 1 - 0.10/6 two-sided interval
            put("dr2_base_h_alo", np.percentile(dd, 100 * 0.05 / 6)); put("dr2_base_h_ahi", np.percentile(dd, 100 - 100 * 0.05 / 6))
for bk in BSL:
    est_ = np.polyfit(AN.hoop_z.values, y - PRED[bk], 1)[0]
    put(f"bs_{bk}", est_ * PS, 1); put(f"bs_{bk}_lo", np.percentile(BSL[bk], 5) * PS, 1)
    put(f"bs_{bk}_hi", np.percentile(BSL[bk], 95) * PS, 1)
    put(f"bs_{bk}_lo95", np.percentile(BSL[bk], 2.5) * PS, 1)
    put(f"bs_{bk}_hi95", np.percentile(BSL[bk], 97.5) * PS, 1)
    PB[("slope", "bs_" + bk)] = float(min(1.0, 2 * min(1 + (np.asarray(BSL[bk]) <= 0).sum(),
                                                       1 + (np.asarray(BSL[bk]) >= 0).sum()) / (NBOOT + 1)))
put("sd_y", float(np.std(y)))
_nb = pd.Series([tuple(c_) for c_ in NEST_B]).explode()
put("nest_pct_pickrole", 100 * (_nb == "pick").mean(), 0)
put("nest_pct_full", 100 * (_nb == "base").mean(), 0)
# regression fill instead of the median (round 22's main choice), same set
_rls = y - loco(AN, BASE, how="ls")
put("ls_fill_hoop_season", np.polyfit(AN.hoop_z.values, _rls, 1)[0] * PS, 1)

# ---- team-cluster bootstrap of the held-out R2 gain: resample NFL teams (latest club; free agents one cluster each),
# keep every rusher of a drawn team, refit every leave-one-class-out fold. Rushers on one club share scheme and coaches.
AN["team_cl"] = np.where(AN.latest_team.isna(), "FA_" + AN.nfl_id.astype(str), AN.latest_team.astype(str))
TEAMS = AN.team_cl.unique()
TIX = {t_: np.flatnonzero(AN.team_cl.values == t_) for t_ in TEAMS}
crng = np.random.default_rng(909)
CB = {k: [] for k in ["pick1", "pick", "ridge"]}
CBS = []
for _ in range(NBOOT):
    ix = np.concatenate([TIX[t_] for t_ in crng.choice(TEAMS, len(TEAMS))])
    d_ = AN.iloc[ix].reset_index(drop=True)
    if min((d_.draft_year == c).sum() for c in classes) < 8:
        continue
    yb = d_.press_eb.values
    for bk in CB:
        kind_ = "ridge" if bk == "ridge" else False
        p0, p1 = predict(d_, BASES.get(bk, BASE), kind_), predict(d_, BASES.get(bk, BASE) + ["hoop_z"], kind_)
        CB[bk].append(oos(yb, p1)[0] - oos(yb, p0)[0])
        if bk == HEADK:
            CBS.append(np.polyfit(d_.hoop_z.values, yb - p0, 1)[0])
for bk in CB:
    put(f"cdr2_{bk}_lo", np.percentile(CB[bk], 5)); put(f"cdr2_{bk}_hi", np.percentile(CB[bk], 95))
put("n_teams", len(TEAMS)); put("n_cboot", len(CBS))
put("cbs_lo", np.percentile(CBS, 5) * PS, 1); put("cbs_hi", np.percentile(CBS, 95) * PS, 1)

# repeated random splits: 200 random three-way splits that ignore class (drill z stays within Combine year)
_rr2 = np.random.default_rng(31)
RSPL = []
for _ in range(200):
    fold = _rr2.permutation(np.arange(len(AN)) % 3)
    pr_ = np.full(len(AN), np.nan)
    for f_ in range(3):
        tr, te = fill_times(AN[fold != f_], AN[fold == f_])
        pr_[fold == f_] = ols(tr, te, HEADC)[0]
    RSPL.append(np.polyfit(AN.hoop_z.values, y - pr_, 1)[0] * PS)
RSPL = np.asarray(RSPL)
put("rsplit_med", np.median(RSPL), 1); put("rsplit_lo", np.percentile(RSPL, 5), 1)
put("rsplit_hi", np.percentile(RSPL, 95), 1); put("rsplit_pos", 100 * (RSPL > 0).mean(), 0)

# ---- headline estimand: held-out slope of (pressure rate - leave-one-class-out baseline) on each drill measure,
# for all six pre-listed measures. Intervals: class-stratified bootstrap refitting every fold. p: within-class
# permutation. Family-wise p: rows of the 6-measure matrix permuted jointly within class (keeps the correlation
# between measures), statistic = max |slope| over the family.
X6 = AN[FKEYS].values
Xc = X6 - X6.mean(0)


def slopes(Xm, r):
    Xm = Xm - Xm.mean(0)
    return (Xm * (r - r.mean())[:, None]).sum(0) / (Xm ** 2).sum(0)


obs = slopes(X6, RES)
prm = np.random.default_rng(77)
PERM = np.empty((NPERM, len(FKEYS)))
for i_ in range(NPERM):
    ix = np.arange(len(AN))
    for g_ in grp:
        ix[g_] = prm.permutation(g_)
    PERM[i_] = slopes(X6[ix], RES)
maxabs = np.abs(PERM).max(1)
FAMR = []
for j, (key, lab, *_rest) in enumerate(FAM):
    bs = np.array(BRES[key])
    lo, hi = np.percentile(bs, [5, 95])
    p_one = (1 + (PERM[:, j] >= obs[j]).sum()) / (NPERM + 1)
    p_two = (1 + (np.abs(PERM[:, j]) >= abs(obs[j])).sum()) / (NPERM + 1)
    p_fw = (1 + (maxabs >= abs(obs[j])).sum()) / (NPERM + 1)
    cls_s = [np.polyfit(X6[g_, j], RES[g_], 1)[0] for g_ in grp]
    FAMR.append(dict(key=key, lab=lab, est=obs[j], lo=lo, hi=hi, p=p_one, p2=p_two, pfw=p_fw, cls=cls_s, bs=bs))
    t = key.replace("_z", "")
    put(f"ho_{t}_season", obs[j] * PS, 1); put(f"ho_{t}_season_lo", lo * PS, 1)
    put(f"ho_{t}_season_hi", hi * PS, 1)
    put(f"ho_{t}_p", p_one); put(f"ho_{t}_p2", p_two); put(f"ho_{t}_pfw", p_fw)
    put(f"ho_{t}_pabove", 100 * (bs > MARGIN).mean(), 0)
    for c, v in zip(classes, cls_s):
        put(f"ho_{t}_{c}", v * PS, 1)
FW95 = np.percentile(maxabs, 95) * PS
put("fw95_season", FW95, 1)
put("n_family", len(FAM))
HO = {f["key"]: f for f in FAMR}
put("ho_hoop_pos_classes", int(sum(v > 0 for v in HO["hoop_z"]["cls"])))
# equivalence (TOST) for burst against +/- margin: both one-sided bootstrap tails
bsb = HO["burst_z"]["bs"]
put("ho_burst_tost", max((1 + (bsb <= -MARGIN).sum()) / (len(bsb) + 1), (1 + (bsb >= MARGIN).sum()) / (len(bsb) + 1)))
put("ho_burst_pinside", 100 * (np.abs(bsb) < MARGIN).mean(), 0)
_bh = HO["hoop_z"]["bs"]
put("ho_hoop_season_alo", np.percentile(_bh, 100 * 0.05 / 6) * PS, 1)
put("ho_hoop_season_ahi", np.percentile(_bh, 100 - 100 * 0.05 / 6) * PS, 1)

# ---- placebo: Hoop G shuffled among linemen of the same class (PERM column) and of the same class AND role.
JH = FKEYS.index("hoop_z")
NULL_C = PERM[:, JH] * PS
prr = np.random.default_rng(78)
cr_groups = [np.flatnonzero((AN.draft_year.values == c) & (AN.role.values == rl)) for c in classes
             for rl in ("EDGE", "interior")]
NULL_CR = np.empty(NPERM)
for i_ in range(NPERM):
    ix = np.arange(len(AN))
    for g_ in cr_groups:
        ix[g_] = prr.permutation(g_)
    NULL_CR[i_] = slopes(X6[ix][:, [JH]], RES)[0] * PS
OBS_H = obs[JH] * PS
for tag, nl in [("c", NULL_C), ("cr", NULL_CR)]:
    put(f"pl_{tag}_p95", np.percentile(nl, 95), 1); put(f"pl_{tag}_sd", np.std(nl), 1)
    put(f"pl_{tag}_p", (1 + (nl >= OBS_H).sum()) / (NPERM + 1), 3)
    put(f"pl_{tag}_pct", 100 * (nl < OBS_H).mean(), 0)
put("n_perm", NPERM)

# ---- mediation on the same rushers: Hoop G -> NFL bend (a, controls weight and edge share) -> pressure residual
# (b, holding Hoop G). Indirect = a*b, direct = c', total = c (all per sd, on the rushers with 8+ sampled 2-s rushes).
MD = AN.dropna(subset=["g_bend"]).reset_index(drop=True)
MD["m_z"] = (MD.g_bend - MD.g_bend.mean()) / MD.g_bend.std()
MD["r_"] = RES[AN.g_bend.notna().values]


def mediate(d_):
    A1 = np.c_[np.ones(len(d_)), d_.hoop_z, d_.combine_weight_z.fillna(0), d_.edge_share]
    a1 = np.linalg.lstsq(A1, d_.m_z.values, rcond=None)[0][1]
    A2 = np.c_[np.ones(len(d_)), d_.hoop_z, d_.m_z]
    b2 = np.linalg.lstsq(A2, d_.r_.values, rcond=None)[0]
    c_ = np.polyfit(d_.hoop_z.values, d_.r_.values, 1)[0]
    return np.array([c_, b2[1], a1, b2[2], a1 * b2[2]])     # total, direct, a, b, indirect


MED = mediate(MD)
mrng = np.random.default_rng(616)
mg = [np.flatnonzero(MD.draft_year.values == c) for c in classes]
MEDB = np.array([mediate(MD.iloc[np.concatenate([mrng.choice(g_, len(g_)) for g_ in mg])]) for _ in range(NBOOT)])
put("med_n", len(MD))
for j_, nm_ in enumerate(["tot", "dir", "a", "b", "ind"]):
    sc_ = 1 if nm_ == "a" else PS
    put(f"med_{nm_}", MED[j_] * sc_, 2 if nm_ == "a" else 1)
    put(f"med_{nm_}_lo", np.percentile(MEDB[:, j_], 5) * sc_, 2 if nm_ == "a" else 1)
    put(f"med_{nm_}_hi", np.percentile(MEDB[:, j_], 95) * sc_, 2 if nm_ == "a" else 1)
put("med_share", 100 * MED[4] / MED[0], 0)
put("med_share_lo", np.percentile(100 * MEDB[:, 4] / MEDB[:, 0], 5), 0)
put("med_share_hi", np.percentile(100 * MEDB[:, 4] / MEDB[:, 0], 95), 0)

# ============================================================================= round-24 revision: added tests
# (1) the single test fixed for 2026 (Hoop G against pick + role), restated with a 95% interval (bs_pick_lo95/hi95),
#     plus the held-out R2 gain with a permutation p from the SAME within-class-and-role shuffles as the slope placebo.
CLV = AN.draft_year.values


def loco_fast(Xv, yv, cl=None):
    """leave-one-class-out OLS for columns with no missing values (pick, role, drill z)."""
    cl = CLV if cl is None else cl
    pred = np.empty(len(yv))
    for c in np.unique(cl):
        m_ = cl == c
        A = np.c_[np.ones((~m_).sum()), Xv[~m_]]
        b = np.linalg.lstsq(A, yv[~m_], rcond=None)[0]
        pred[m_] = np.c_[np.ones(m_.sum()), Xv[m_]] @ b
    return pred


XB = AN[HEADC].values
R2_0 = oos(y, loco_fast(XB, y))[0]
GAIN_OBS = oos(y, loco_fast(np.c_[XB, AN.hoop_z.values], y))[0] - R2_0
assert abs(GAIN_OBS - float(NUM["dr2_pick_h"])) < 0.006
gr_ = np.random.default_rng(79)
NULL_GAIN = np.empty(NPERM)
HZ = AN.hoop_z.values
for i_ in range(NPERM):
    ix = np.arange(len(AN))
    for g_ in cr_groups:
        ix[g_] = gr_.permutation(g_)
    NULL_GAIN[i_] = oos(y, loco_fast(np.c_[XB, HZ[ix]], y))[0] - R2_0
put("gain_obs", GAIN_OBS); put("gain_p", (1 + (NULL_GAIN >= GAIN_OBS).sum()) / (NPERM + 1), 3)
put("gain_null_med", np.median(NULL_GAIN)); put("gain_null_p95", np.percentile(NULL_GAIN, 95))
put("gain_pct", 100 * (NULL_GAIN < GAIN_OBS).mean(), 0)

# (2) prospective check: fit the pick + role forecast AND the Hoop G weight on 2023+2024 only, test on 2025.
TR_ = AN[AN.draft_year < 2025].reset_index(drop=True)
TE_ = AN[AN.draft_year == 2025].reset_index(drop=True)


def fit_trte(tr, te):
    A = np.c_[np.ones(len(tr)), tr[HEADC].values]
    b = np.linalg.lstsq(A, tr.press_eb.values, rcond=None)[0]
    w = np.polyfit(tr.hoop_z.values, tr.press_eb.values - A @ b, 1)
    p_te = np.c_[np.ones(len(te)), te[HEADC].values] @ b
    r_te = te.press_eb.values - p_te
    s_te = np.polyfit(te.hoop_z.values, r_te, 1)[0]
    return w, p_te, r_te, s_te


W_TR, P_TE, R_TE, S_TE = fit_trte(TR_, TE_)
yte = TE_.press_eb.values
sst_ = ((yte - yte.mean()) ** 2).sum()
r2te0 = 1 - ((yte - P_TE) ** 2).sum() / sst_
r2te1 = 1 - ((yte - P_TE - np.polyval(W_TR, TE_.hoop_z.values)) ** 2).sum() / sst_
trg = [np.flatnonzero(TR_.draft_year.values == c) for c in (2023, 2024)]
pr_rng = np.random.default_rng(2025)
PRO = []
for _ in range(NBOOT):
    tb = TR_.iloc[np.concatenate([pr_rng.choice(g_, len(g_)) for g_ in trg])]
    eb = TE_.iloc[pr_rng.choice(len(TE_), len(TE_))]
    w_, _, _, s_ = fit_trte(tb, eb)
    PRO.append((w_[0], s_))
PRO = np.array(PRO)
put("pro_n_tr", len(TR_)); put("pro_n_te", len(TE_))
put("pro_wtr", W_TR[0] * PS, 1); put("pro_wtr_lo95", np.percentile(PRO[:, 0], 2.5) * PS, 1)
put("pro_wtr_hi95", np.percentile(PRO[:, 0], 97.5) * PS, 1)
put("pro_s25", S_TE * PS, 1); put("pro_s25_lo95", np.percentile(PRO[:, 1], 2.5) * PS, 1)
put("pro_s25_hi95", np.percentile(PRO[:, 1], 97.5) * PS, 1)
_dd = PRO[:, 1] - PRO[:, 0]
put("pro_diff_p", min(1.0, 2 * min((_dd <= 0).mean(), (_dd >= 0).mean())), 2)
put("pro_r2_0", r2te0); put("pro_r2_1", r2te1)

# (3) measurement-error model for the Hoop G -> NFL bend -> pressure split (regression calibration with shrunk
# random-effects estimates). Hoop G: each man's value is shrunk toward the class mean by its own reliability (one-lap
# reliability from the two turns of each lap, Spearman-Brown to his lap count). NFL bend: one-way random-effects model
# on the per-rush hardest half-second (between-man variance tau2, rush-to-rush variance sw2), each man's mean shrunk by
# tau2 / (tau2 + sw2 / his rushes). Shrunk values replace the noisy ones as predictors; NFL bend as the outcome of the
# first regression stays unshrunk (noise in an outcome does not bias a slope). Bend is put in sd of TRUE bend.
_gw = GR.dropna(subset=["g_bend"])
_gw = _gw[_gw.nfl_id.isin(PG.nfl_id)]
_gw_ids = _gw.nfl_id.unique()
_gw_by = {i: _gw.g_bend.values[_gw.nfl_id.values == i] for i in _gw_ids}


def re_var(ids):
    """one-way random effects (method of moments) on per-rush NFL bend for the given linemen: tau2, sw2."""
    vals = [_gw_by[i] for i in ids]
    means = np.array([v.mean() for v in vals]); ns = np.array([len(v) for v in vals])
    sw2 = sum(((v - v.mean()) ** 2).sum() for v in vals) / (ns.sum() - len(vals))
    return max(means.var(ddof=1) - (sw2 / ns).mean(), 1e-6), sw2


TAU2, SW2 = re_var(_gw_ids)
put("re_n", len(_gw_ids)); put("lam_bend", float((TAU2 / (TAU2 + SW2 / PG.g_n)).mean()))
put("lam_hoop_md", MD.rel_i.mean())


def mediate_rc(d_, tau2, sw2):
    gm = d_.g_bend.mean()
    lam_m = tau2 / (tau2 + sw2 / d_.g_n.values)
    m_obs = (d_.g_bend.values - gm) / np.sqrt(tau2)
    m_rc = lam_m * m_obs
    x_rc = d_.rel_i.values * d_.hoop_z.values
    r_ = d_.r_.values
    A1 = np.c_[np.ones(len(d_)), x_rc, d_.combine_weight_z.fillna(0).values, d_.edge_share.values]
    a1 = np.linalg.lstsq(A1, m_obs, rcond=None)[0][1]
    b2 = np.linalg.lstsq(np.c_[np.ones(len(d_)), x_rc, m_rc], r_, rcond=None)[0]
    c_ = np.polyfit(x_rc, r_, 1)[0]
    return np.array([c_, b2[1], a1, b2[2], a1 * b2[2]])     # total, direct, a, b, indirect (per sd of true values)


MEDC = mediate_rc(MD, TAU2, SW2)
_mrc = np.random.default_rng(617)
MEDCB = []
for _ in range(NBOOT):
    d_ = MD.iloc[np.concatenate([_mrc.choice(g_, len(g_)) for g_ in mg])]
    t2_, s2_ = re_var(_mrc.choice(_gw_ids, len(_gw_ids)))
    MEDCB.append(mediate_rc(d_, t2_, s2_))
MEDCB = np.array(MEDCB)
for j_, nm_ in enumerate(["tot", "dir", "a", "b", "ind"]):
    sc_ = 1 if nm_ == "a" else PS
    put(f"mc_{nm_}", MEDC[j_] * sc_, 2 if nm_ == "a" else 1)
    put(f"mc_{nm_}_lo", np.percentile(MEDCB[:, j_], 5) * sc_, 2 if nm_ == "a" else 1)
    put(f"mc_{nm_}_hi", np.percentile(MEDCB[:, j_], 95) * sc_, 2 if nm_ == "a" else 1)
put("mc_share", 100 * MEDC[4] / MEDC[0], 0)
_shb = MEDCB[:, 4] / MEDCB[:, 0]
put("mc_share_lo", 100 * np.percentile(_shb, 5), 0); put("mc_share_hi", 100 * np.percentile(_shb, 95), 0)
# same model with the split-half reliability of NFL bend (rank-based, higher) instead of the variance components
_lsh = float(REL["g_bend"])
_t2sh = _lsh / (1 - _lsh) * SW2 / np.median(PG.g_n)
MEDSH = mediate_rc(MD, _t2sh, SW2)
put("ms_share", 100 * MEDSH[4] / MEDSH[0], 0); put("ms_ind", MEDSH[4] * PS, 1)


# (4) negative controls, same pipeline (held-out pick + role residual, class-stratified refit bootstrap, 95%, and
# the within-class-and-role shuffle): (a) the same lean computation on a DL drill with no rush arc (the wave drill);
# (b) the five non-hoop DL measures already in the six-measure family; (c) a group with no hoop: the same lean
# computation on the OL long-pull drill (an arc) against OL pressures allowed, beyond pick + tackle share.
def lean_feature(group, drill):
    ev_ = events[(events.position_group == group) & (events.drill_name == drill)][["event_id", "nfl_id", "draft_year"]]
    fr_ = pd.read_parquet(PROC / "combine_frames.parquet", filters=[("event_id", "in", list(ev_.event_id))],
                          columns=["event_id", "frame", "entity_type", "x", "y", "s"])
    fr_ = fr_[fr_.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
    out_ = []
    for eid, g_ in fr_.groupby("event_id", sort=False):
        f_ = attempt_features(g_)
        if f_ is not None:
            out_.append({"event_id": eid, "lean": f_["bend_g"]})
    d_ = pd.DataFrame(out_).merge(ev_, on="event_id").groupby("nfl_id").agg(
        lean=("lean", "mean"), draft_year=("draft_year", "first")).reset_index()
    d_["lean_z"] = yz(d_.lean, d_.draft_year)
    return d_[["nfl_id", "lean_z"]]


def control_test(d_, xcol, ycol, bcols, rolecol, seed):
    d_ = d_.dropna(subset=[xcol, ycol] + bcols).reset_index(drop=True)
    cl = d_.draft_year.values
    yv = d_[ycol].values
    xv = d_[xcol].values
    res = yv - loco_fast(d_[bcols].values, yv, cl)
    est = np.polyfit(xv, res, 1)[0]
    rg = np.random.default_rng(seed)
    gg = [np.flatnonzero(cl == c) for c in classes]
    bs = []
    for _ in range(NBOOT):
        ix = np.concatenate([rg.choice(g_, len(g_)) for g_ in gg])
        yb = yv[ix]
        bs.append(np.polyfit(xv[ix], yb - loco_fast(d_[bcols].values[ix], yb, cl[ix]), 1)[0])
    cg = [np.flatnonzero((cl == c) & (d_[rolecol].values == r)) for c in classes for r in np.unique(d_[rolecol])]
    nl = np.empty(2000)
    for i_ in range(2000):
        ix = np.arange(len(d_))
        for g_ in cg:
            ix[g_] = rg.permutation(g_)
        nl[i_] = np.polyfit(xv[ix], res, 1)[0]
    p2 = (1 + (np.abs(nl) >= abs(est)).sum()) / (2000 + 1)
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), p2, len(d_)


NC = []
WAVE = cached("wave_lean", lambda: lean_feature("DL", "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION"))
ANW = AN.merge(WAVE, on="nfl_id", how="left")
put("rho_hoop_wave", stats.spearmanr(ANW.dropna(subset=["lean_z"]).hoop_z, ANW.dropna(subset=["lean_z"]).lean_z)[0])
NC.append(("wave", "DL lean, wave drill (no arc)") + control_test(ANW, "lean_z", "press_eb", HEADC, "role", 81))
NC.append(("hoop", "Hoop G, same test") + control_test(AN, "hoop_z", "press_eb", HEADC, "role", 80))
for k_, lab_ in [("burst_z", "DL rush drill: burst"), ("rpa_z", "DL rush drill: peak accel"),
                 ("rts_z", "DL rush drill: top speed"), ("hts_z", "Hoop: top speed"), ("hlap_z", "Hoop: lap time")]:
    NC.append((k_.replace("_z", ""), lab_) + control_test(AN, k_, "press_eb", HEADC, "role", 82))
# OL: same lean computation on the long-pull drill; outcome = shrunk pressures allowed per 100 pass blocks, sign
# flipped (positive = fewer pressures allowed), so a working control would show the same sign as Hoop G.
ol = players[players.position_group == "OL"][["nfl_id", "draft_year", "draft_overall_pick", "nfl_position"]].copy()
ol["log_pick"] = np.log(ol.draft_overall_pick.fillna(UDFA_PICK))
ao = align[align.nfl_id.isin(ol.nfl_id)].copy()
ao["t_pl"] = np.where(ao.lined_up_position == "T", ao.plays, 0)
_to = ao.groupby("nfl_id").agg(t_pl=("t_pl", "sum"), pl=("plays", "sum"))
ol = ol.merge((_to.t_pl / _to.pl).rename("t_share").reset_index(), on="nfl_id", how="left")
ol["t_share"] = ol.t_share.fillna((ol.nfl_position == "T").astype(float))
ol["role"] = np.where(ol.t_share >= 0.5, "T", "IOL")
so = seasons[seasons.position_group == "OL"].groupby("nfl_id")[["pass_blocks", "pressures_allowed"]].sum().reset_index()
ol = ol.merge(so, on="nfl_id", how="left")
mo = ol.pass_blocks >= RATE_MIN
no_, ko_ = ol.loc[mo, "pass_blocks"], ol.loc[mo, "pressures_allowed"]
muo = ko_.sum() / no_.sum(); No_ = no_.sum()
t2o = ((no_ * (ko_ / no_ - muo) ** 2).sum() - (len(no_) - 1) * muo * (1 - muo)) / (No_ - (no_ ** 2).sum() / No_)
Mo = muo * (1 - muo) / max(t2o, 1e-6) - 1
ol.loc[mo, "prot"] = -100 * (ol.loc[mo, "pressures_allowed"] + Mo * muo) / (ol.loc[mo, "pass_blocks"] + Mo)
OLL = cached("ol_pull_lean", lambda: lean_feature("OL", "OL_PULL_DRILL_LONG_PULL_RIGHT"))
ol = ol.merge(OLL, on="nfl_id", how="left")
NC.append(("ol", "OL lean, long-pull drill") + control_test(ol, "lean_z", "prot", ["log_pick", "t_share"], "role", 83))
for tag_, lab_, est_, lo_, hi_, p_, n_ in NC:
    put(f"nc_{tag_}", est_ * PS, 1); put(f"nc_{tag_}_lo", lo_ * PS, 1); put(f"nc_{tag_}_hi", hi_ * PS, 1)
    put(f"nc_{tag_}_p", p_, 2); put(f"nc_{tag_}_n", int(n_))
put("ol_press_mean", 100 * muo, 1)
_nc_other = [r_ for r_ in NC if r_[0] not in ("hoop", "ol", "wave")]   # straight-line / speed measures
put("nc_dl_n_ctrl", len(_nc_other))
put("nc_dl_maxabs", max(abs(r_[2]) for r_ in _nc_other) * PS, 1)
put("nc_dl_min_p", min(r_[5] for r_ in _nc_other), 2)


def put_slope(tag, est, draws):
    draws = np.asarray(draws)
    put(f"{tag}", est * PS, 1); put(f"{tag}_lo", np.percentile(draws, 5) * PS, 1)
    put(f"{tag}_hi", np.percentile(draws, 95) * PS, 1)
    PB[("slope", tag)] = float(min(1.0, 2 * min(1 + (draws <= 0).sum(), 1 + (draws >= 0).sum()) / (len(draws) + 1)))


# per-class, with a within-class player bootstrap
for j_, c in enumerate(classes):
    put(f"cls{c}_lo", np.percentile(BX[f"cls{c}"], 5) * PS, 1); put(f"cls{c}_hi", np.percentile(BX[f"cls{c}"], 95) * PS, 1)
put("cls2025_ppos", 100 * (np.asarray(BX["cls2025"]) > 0).mean(), 0)
# edge vs interior: same residual, separate slopes (an edge and a 300-lb 3-technique are not held to one bend line)
for rl in ("EDGE", "interior"):
    mr = AN.role.values == rl
    put_slope(f"role_{rl}", np.polyfit(AN.hoop_z.values[mr], RES[mr], 1)[0], BX[rl])
    put(f"role_{rl}_n", int(mr.sum()))
# mean of the two turns instead of the harder one (left/right asymmetry robustness)
put_slope("mt_hoop", np.polyfit(AN.meanturn_z.values, RES, 1)[0], BX["meanturn"])

# ---- measurement-error simulation: how much of the weak result is one-lap noise?
# True bend T ~ N(0,1); a lineman's Hoop G = sqrt(lam_i) T + sqrt(1 - lam_i) e, then standardized, where lam_i is the
# reliability of his own Hoop G (one lap or two), and the one-lap reliability is redrawn every simulation from its
# player bootstrap. Outcome residual = beta T + noise with the observed residual variance. For each true slope beta
# (per sd of true bend) we get the distribution of the slope we would observe at n = 77; the implied true-slope
# range is every beta whose 5-95% band covers the observed slope.
B_OBS = float(HO["hoop_z"]["est"])
VR = float(np.var(RES))
KK = AN.hoop_k.values
NME = len(AN)
me_rng = np.random.default_rng(4343)
BETA = np.round(np.arange(-1.0, 3.01, 0.05), 2)     # rate points per sd of true bend
NSIM = 3000


def sim_obs(beta, laps=None):
    lam1 = BLR_LAP[me_rng.integers(0, len(BLR_LAP), NSIM)]
    k_ = KK if laps is None else np.full(NME, laps)
    lam = sb(lam1[:, None], k_[None, :])
    T = me_rng.standard_normal((NSIM, NME))
    X = np.sqrt(lam) * T + np.sqrt(1 - lam) * me_rng.standard_normal((NSIM, NME))
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    yv = beta * T + np.sqrt(max(VR - beta ** 2, 0.2 * VR)) * me_rng.standard_normal((NSIM, NME))
    yv = yv - yv.mean(1, keepdims=True)
    bh = (X * yv).sum(1) / (X ** 2).sum(1)
    se = np.sqrt(((yv - bh[:, None] * X) ** 2).sum(1) / (NME - 2) / (X ** 2).sum(1))
    return bh, bh / se


MEQ = []
for be in BETA:
    bh_, _ = sim_obs(be)
    MEQ.append(np.percentile(bh_, [5, 50, 95]))
MEQ = np.array(MEQ)
_cov = BETA[(MEQ[:, 0] <= B_OBS) & (MEQ[:, 2] >= B_OBS)]
_nz = BETA > 0.2
ATT = float(np.median(MEQ[_nz, 1] / BETA[_nz]))          # share of the true slope one lap shows, typical
B_TRUE = B_OBS / ATT
put("me_att", 100 * ATT, 0); put("me_true", B_TRUE * PS, 1)
put("me_lo", _cov.min() * PS, 1); put("me_hi", _cov.max() * PS, 1)
TCRIT = stats.t.ppf(0.90, NME - 2)
_b1, _t1 = sim_obs(B_TRUE)
_b4, _t4 = sim_obs(B_TRUE, laps=4)
put("me_pow1", 100 * (_t1 > TCRIT).mean(), 0); put("me_pow4", 100 * (_t4 > TCRIT).mean(), 0)
put("me_exp4", np.median(_b4) * PS, 1); put("me_exp1", np.median(_b1) * PS, 1)
_b0, _ = sim_obs(0.0)
put("me_null_exceed", 100 * (_b0 >= B_OBS).mean(), 0)
put("lam_hoop", float(AN.rel_i.mean()))

# leave-one-player-out: drop each rusher, refit the held-out baseline, re-estimate (overall and 2025 class)
LO = []
for i_ in range(len(AN)):
    d_ = AN.drop(index=i_).reset_index(drop=True)
    r_ = d_.press_eb.values - loco(d_, HEADC)
    m25 = d_.draft_year.values == 2025
    LO.append((np.polyfit(d_.hoop_z, r_, 1)[0] * PS, np.polyfit(d_.hoop_z[m25], r_[m25], 1)[0] * PS))
LO = np.array(LO)
put("lopo_min", LO[:, 0].min(), 1); put("lopo_max", LO[:, 0].max(), 1)
put("lopo25_min", LO[:, 1].min(), 1); put("lopo25_max", LO[:, 1].max(), 1)
put("lopo25_npos", int((LO[:, 1] > 0).sum()))
_i25 = int(np.argmax(np.abs(LO[:, 1] - HO["hoop_z"]["cls"][2] * PS)))
TXT_LOPO = AN.iloc[_i25].display_name

# one rusher set, one fill rule: verify.py uses the same set (all six drill measures valid) and the median fill.
# Sensitivity: add the one 30+ rush lineman with a hoop lap and a pass-rush clip but no valid burst rep.
RUSH_IDS = set(ev[ev.drill_name == "PASS_RUSH_DRILL"].nfl_id)
AN78 = co30[co30.hoop_z.notna() & co30.nfl_id.isin(RUSH_IDS)].reset_index(drop=True)
put("n_an78", len(AN78))
put("rec_med78", np.polyfit(AN78.hoop_z, AN78.press_eb.values - loco(AN78, HEADC), 1)[0] * PS, 1)
TXT_EXTRA = ", ".join(AN78[~AN78.nfl_id.isin(AN.nfl_id)].display_name)

# ---- inverse-probability weight for being in the analysis set (all six drill measures) among 30+ rush linemen
import statsmodels.api as sm
_ip = co30.copy()
_ip["in_set"] = _ip[FKEYS].notna().all(1).astype(int)
_ipx = sm.add_constant(_ip[["log_pick", "edge_share", "ngs_athleticism_score"]].assign(
    w=_ip.combine_weight_z.fillna(0)))
_ps = sm.Logit(_ip.in_set, _ipx).fit(disp=0).predict(_ipx)
_ip["ipw"] = 1 / _ps
W_IPW = AN.merge(_ip[["nfl_id", "ipw"]], on="nfl_id", how="left").ipw.values
_xh = AN.hoop_z.values
_xm = _xh - np.average(_xh, weights=W_IPW)
IPW_SLOPE = float(np.sum(W_IPW * _xm * (RES - np.average(RES, weights=W_IPW))) / np.sum(W_IPW * _xm ** 2))
put("ipw_hoop_season", IPW_SLOPE * PS, 1); put("n_co30", len(_ip)); put("n_in_set", int(_ip.in_set.sum()))
put("ipw_max", float(W_IPW.max()), 1)

# ---- sensitivity of the Hoop G slope (whole pipeline rerun)


def run_variant(mult=1.0, rmin=RATE_MIN, nperm=2000, seed=5):
    d_ = dl.copy()
    M_ = Mw * mult
    mk = d_.rush_snaps > 0
    d_.loc[mk, "press_eb"] = 100 * (d_.loc[mk, "pressures"] + M_ * mu) / (d_.loc[mk, "rush_snaps"] + M_)
    an_ = d_[d_.rush_snaps >= rmin].dropna(subset=FKEYS).reset_index(drop=True)
    res_ = an_.press_eb.values - loco(an_, HEADC)
    xh = an_.hoop_z.values
    sl_ = np.polyfit(xh, res_, 1)[0]
    rg = np.random.default_rng(seed)
    gg = [np.flatnonzero(an_.draft_year.values == c) for c in classes]
    cnt = 0
    for _ in range(nperm):
        xp = xh.copy()
        for g_ in gg:
            xp[g_] = rg.permutation(xp[g_])
        cnt += np.polyfit(xp, res_, 1)[0] >= sl_
    return sl_ * PS, (1 + cnt) / (nperm + 1), len(an_)


SENS = [("prior x0.5", 0.5, RATE_MIN), ("prior x2", 2.0, RATE_MIN), ("20+ rushes", 1.0, 20), ("50+ rushes", 1.0, 50)]
SV = [run_variant(mlt, rmn) for _, mlt, rmn in SENS]
put("sens_slope_min", min(v[0] for v in SV), 1); put("sens_slope_max", max(v[0] for v in SV), 1)
put("sens_p_max", max(v[1] for v in SV)); put("n_sens", len(SENS))

# ============================================================================= day-3 decision test, one row per rusher
# Rankers inside each class's day-3/undrafted rushers (30+ rushes, all six drill measures). Everything is out of
# class: the forecast is the leave-one-class-out baseline, and the Hoop G weight added to it is the held-out
# residual slope estimated on the OTHER two classes. Score = within-class Spearman with shrunk pressure rate,
# pooled across classes (weighted by n). Interval: class-stratified rusher bootstrap of the same day-3 rows.
AN["w_out"] = np.nan
for c in classes:
    o_ = AN.draft_year != c
    AN.loc[AN.draft_year == c, "w_out"] = np.polyfit(AN.loc[o_, "hoop_z"], RES[o_.values], 1)[0]
AN["fc_hoop"] = AN.fc + AN.w_out * AN.hoop_z
AN["neg_pick"] = -AN.pick
D3 = AN[AN.pick >= DAY3].reset_index(drop=True)
put("d3_n", len(D3)); NUM["d3_n_per_class"] = "/".join(str(int((D3.draft_year == c).sum())) for c in classes)
put("d3_udfa", int((D3.pick >= UDFA_PICK).sum()))
RANK = [("neg_pick", "Earlier pick"), ("fc", "No-tracking forecast"), ("fc_hoop", "Forecast + Hoop G"),
        ("hoop_z", "Hoop G alone"), ("burst_z", "Rush-drill burst alone")]
g3 = [np.flatnonzero(D3.draft_year.values == c) for c in classes]


def pooled_rho(d, col, groups):
    num = den = 0.0
    for g_ in groups:
        if len(g_) < 4:
            continue
        r_ = stats.spearmanr(d[col].values[g_], d.press_eb.values[g_])[0]
        if np.isnan(r_):
            continue
        num += r_ * len(g_); den += len(g_)
    return num / den


def top_third_hit(d, col, groups):
    """share of each class's top third (by the ranker) that beat the class's day-3 median pressure rate."""
    k = n = 0
    for g_ in groups:
        v, yv = d[col].values[g_], d.press_eb.values[g_]
        top = np.argsort(-v, kind="stable")[: int(round(len(g_) / 3))]
        k += int((yv[top] > np.median(yv)).sum()); n += len(top)
    return k, n


RR = {col: pooled_rho(D3, col, g3) for col, _ in RANK}
BR = {col: [] for col, _ in RANK}
HITGAP = []
brg = np.random.default_rng(303)
for _ in range(NBOOT):
    ix = np.concatenate([brg.choice(g_, len(g_)) for g_ in g3])
    db = D3.iloc[ix].reset_index(drop=True)
    gb = [np.flatnonzero(db.draft_year.values == c) for c in classes]
    for col, _ in RANK:
        BR[col].append(pooled_rho(db, col, gb))
    ka, na = top_third_hit(db, "fc_hoop", gb)
    kb, nb = top_third_hit(db, "neg_pick", gb)
    HITGAP.append(100 * (ka / na - kb / nb))
BAR_GAP = 20  # points of top-third hit rate over pick order needed before Hoop G may touch board order
DEC = []
for col, lab in RANK:
    t = {"neg_pick": "pick", "fc": "fc", "fc_hoop": "fch", "hoop_z": "hoop", "burst_z": "burst"}[col]
    lo, hi = np.nanpercentile(BR[col], [5, 95])
    k, n = top_third_hit(D3, col, g3)
    wl, wh = wilson(k, n)
    DEC.append(dict(col=col, lab=lab, r=RR[col], lo=lo, hi=hi, k=k, n=n, wl=wl, wh=wh))
    put(f"dr_{t}", RR[col]); put(f"dr_{t}_lo", lo); put(f"dr_{t}_hi", hi)
    put(f"tt_{t}_k", k); put(f"tt_{t}_n", n)
    put(f"tt_{t}_pct", 100 * k / n, 0); put(f"tt_{t}_lo", 100 * wl, 0); put(f"tt_{t}_hi", 100 * wh, 0)
for a_, b_, t in [("fc_hoop", "fc", "fch_fc"), ("fc_hoop", "neg_pick", "fch_pick"), ("hoop_z", "neg_pick", "hoop_pick")]:
    dd = np.array(BR[a_]) - np.array(BR[b_])
    put(f"dd_{t}", RR[a_] - RR[b_]); put(f"dd_{t}_lo", np.nanpercentile(dd, 5))
    put(f"dd_{t}_hi", np.nanpercentile(dd, 95)); put(f"dd_{t}_ppos", 100 * np.nanmean(dd > 0), 0)
    dd = dd[~np.isnan(dd)]
    PB[("day3", t)] = float(min(1.0, 2 * min(1 + (dd <= 0).sum(), 1 + (dd >= 0).sum()) / (len(dd) + 1)))
_ka, _na = top_third_hit(D3, "fc_hoop", g3)
_kb, _nb = top_third_hit(D3, "neg_pick", g3)
put("hg_gap", 100 * (_ka / _na - _kb / _nb), 0)
put("hg_gap_lo", np.percentile(HITGAP, 5), 0); put("hg_gap_hi", np.percentile(HITGAP, 95), 0)
put("hg_bar", BAR_GAP)
_hg = np.asarray(HITGAP)
PB[("day3", "hitgap")] = float(min(1.0, 2 * min(1 + (_hg <= 0).sum(), 1 + (_hg >= 0).sum()) / (len(_hg) + 1)))
# Hoop G added to the forecast: how often does it change the order, and how far does it move a rusher?
put("w_out_season_min", AN.w_out.min() * PS, 1); put("w_out_season_max", AN.w_out.max() * PS, 1)
# film flag: top third of Hoop G among day-3 rushers vs the rest, beat the no-tracking forecast?
D3["flag"] = D3.groupby("draft_year").hoop_z.transform(lambda v: v.rank(pct=True) > 2 / 3)
D3["beat"] = D3.press_eb > D3.fc
k1, n1 = int(D3[D3.flag].beat.sum()), int(D3.flag.sum())
k0, n0 = int(D3[~D3.flag].beat.sum()), int((~D3.flag).sum())
l1, u1 = wilson(k1, n1); l0, u0 = wilson(k0, n0)
put("fl_k", k1); put("fl_n", n1); put("fl_pct", 100 * k1 / n1, 0); put("fl_lo", 100 * l1, 0); put("fl_hi", 100 * u1, 0)
put("rs_k", k0); put("rs_n", n0); put("rs_pct", 100 * k0 / n0, 0); put("rs_lo", 100 * l0, 0); put("rs_hi", 100 * u0, 0)
put("fl_err", 100 - 100 * k1 / n1, 0); put("fl_err_lo", 100 - 100 * u1, 0); put("fl_err_hi", 100 - 100 * l1, 0)
put("fl_p_fisher", stats.fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="greater")[1])
put("fl_over_season", (D3[D3.flag].press_eb - D3[D3.flag].fc).mean() * PS - (D3[~D3.flag].press_eb -
                                                                             D3[~D3.flag].fc).mean() * PS, 1)
# Monday rule in numbers: how many names a class produces, and what a miss costs
_fm = D3[D3.flag & ~D3.beat]
put("miss_short", -(_fm.press_eb - _fm.fc).mean() * PS, 1)
put("hit_gain", (D3[D3.flag & D3.beat].press_eb - D3[D3.flag & D3.beat].fc).mean() * PS, 1)
# price: do clubs draft for Hoop G?
_ba = dl[dl.hoop_z.notna()].copy()
_ba["third"] = _ba.groupby("draft_year").hoop_z.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
put("pick_top_third", _ba[_ba.third == 2].pick.median(), 0); put("pick_bot_third", _ba[_ba.third == 0].pick.median(), 0)

# 2026 power for the frozen slope test, one class and three classes
SD_RES = float(np.std(RES)); NCL = round(len(AN) / 3)
prng = np.random.default_rng(26)


def power_slope(beta, ncl, sims=3000):
    hits = 0
    for _ in range(sims):
        x_ = prng.standard_normal(ncl * NCL)
        y_ = beta * x_ + prng.normal(0, SD_RES, len(x_))
        r_ = stats.linregress(x_, y_)
        hits += (r_.slope / r_.stderr) > stats.t.ppf(0.90, len(x_) - 2)
    return hits / sims


B_OBS = float(HO["hoop_z"]["est"])
put("pws_obs_1", 100 * power_slope(B_OBS, 1), 0); put("pws_obs_3", 100 * power_slope(B_OBS, 3), 0)

# ============================================================================= 2026 board: day-3 Hoop G film flags, 2024-25
sc = dl[dl.hoop_z.notna() & (dl.rush_snaps >= RATE_MIN)].copy()
sc["fc"] = np.nan
for c in classes:
    trf, tef = fill_times(co30[co30.draft_year != c].copy(), sc[sc.draft_year == c])
    sc.loc[tef.index, "fc"] = ols(trf, tef, HEADC)[0]
ba = dl[dl.hoop_z.notna()].copy()
ba["third"] = ba.groupby("draft_year").hoop_z.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False))
ba["short"] = (ba.pick >= DAY3) & (ba.third == 2)
ba["cls_rank"] = ba.groupby("draft_year").hoop_z.rank(ascending=False, method="min").astype(int)
ba["cls_n"] = ba.groupby("draft_year").hoop_z.transform("size").astype(int)
_fl = ba.groupby("draft_year").short.sum()
put("flags_per_class", _fl.mean(), 0); put("flags_min", int(_fl.min())); put("flags_max", int(_fl.max()))
put("hoop_per_class", ba.groupby("draft_year").size().mean(), 0)
brd =ba[ba.draft_year.isin([2024, 2025]) & ba.short].copy().merge(sc[["nfl_id", "fc"]], on="nfl_id", how="left")


def call(r_):
    if r_.rush_snaps < RATE_MIN:
        return "Audition"
    if r_.rush_snaps >= REG_MIN and r_.press_eb - r_.fc >= MARGIN:
        return "Go"
    if r_.rush_snaps >= REG_MIN and r_.press_eb - r_.fc <= -MARGIN:
        return "No-go"
    return "Hold"


def flip_rate(r_):
    """2026 pressure rate (%) that moves the shrunk career rate onto the frozen forecast at his 2025 volume."""
    R = r_.r25 if pd.notna(r_.r25) and r_.r25 > 0 else r_.rush_snaps
    return 100 * ((r_.fc / 100) * (r_.rush_snaps + R + Mw) - r_.pressures - Mw * mu) / R


brd["call"] = brd.apply(call, axis=1)
brd["latest_team"] = brd.latest_team.fillna("FA")
brd["over"] = brd.press_eb - brd.fc
brd["flip"] = [flip_rate(r_) if r_.rush_snaps >= RATE_MIN else np.nan for _, r_ in brd.iterrows()]
_pp_ = brd.press_eb / 100
brd["over_sd"] = 100 * np.sqrt(_pp_ * (1 - _pp_) / (brd.rush_snaps + Mw + 1))
order = {"Go": 0, "No-go": 1, "Hold": 2, "Audition": 3}
brd = brd.sort_values(["call", "over"], key=lambda c_: c_.map(order) if c_.name == "call" else -c_)
brd[["display_name", "draft_year", "pick", "latest_team", "role", "hoop_z", "rush_snaps", "press_eb", "fc", "over",
     "call", "flip"]].to_csv(OUT / "board.csv", index=False)


def flip_txt(r_):
    if r_.call == "Go" and r_.flip <= 0:
        return "cannot flip in one season"
    if r_.call == "Go":
        return f"flips below {r_.flip:.1f}%"
    if r_.call == "No-go":
        return f"flips above {r_.flip:.1f}%"
    if r_.rush_snaps < REG_MIN:
        return f"needs {REG_MIN - int(r_.rush_snaps)} more rushes"
    return f"on par at {r_.flip:.1f}%"


big = brd[brd.call != "Audition"]
AUD = brd[brd.call == "Audition"]
put("n_board", len(brd)); put("n_board_rows", len(big)); put("n_aud", len(AUD))
for cl_ in ("Go", "No-go", "Hold"):
    put(f"n_{cl_.lower().replace('-', '')}", int((brd.call == cl_).sum()))


def nm_team(r_):
    return f"{r_.display_name} ({r_.latest_team})"


GO, NOGO, HOLD = brd[brd.call == "Go"], brd[brd.call == "No-go"], brd[brd.call == "Hold"]
for i_, (_, r_) in enumerate(GO.iterrows()):
    put(f"go{i_}_rush", int(r_.rush_snaps)); put(f"go{i_}_pr", r_.press_eb, 1); put(f"go{i_}_fc", r_.fc, 1)
    put(f"go{i_}_flip", r_.flip, 1); put(f"go{i_}_hoop", r_.hoop_z, 1)
for i_, (_, r_) in enumerate(NOGO.iterrows()):
    put(f"ng{i_}_flip", r_.flip, 1)
put("nogo_flip_min", NOGO.flip.min(), 1)


def join(names):
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


TXT = {"go": join(nm_team(r_) for _, r_ in GO.iterrows()), "nogo": join(nm_team(r_) for _, r_ in NOGO.iterrows()),
       "hold": join(nm_team(r_) for _, r_ in HOLD.iterrows()),
       "aud": ", ".join(f"{r_.display_name} ({int(r_.draft_year)}, "
                        f"{'UDFA' if r_.pick >= UDFA_PICK else int(r_.pick)}, {r_.latest_team})"
                        for _, r_ in AUD.iterrows()),
       "go_names": [r_.display_name for _, r_ in GO.iterrows()],
       "go_teams": list(GO.latest_team), "go_roles": list(GO.role)}

frozen = {"metric": "Hoop G: hardest 0.5 s of sideways acceleration on Run-the-Hoop, z within Combine year, mean of "
                    "graded laps",
          "use": f"film flag for day-3/undrafted (pick>={DAY3}) rushers in the class top third of Hoop G; added to the "
                 "film queue beside pick order, never ahead of it; never moves the board",
          "primary_test": "2026 Combine class after Week 18 of its first NFL season (30+ rushes): slope of Hoop G on "
                          "(career pressure rate - pick + role forecast fitted on the 2023-25 classes and frozen here), "
                          "within-class permutation, one-sided p < 0.10 keeps the flag; slope <= 0 retires it",
          "secondary": "same slope against pick only and against the full no-tracking forecast; reported, not decisive",
          "promotion_to_board": "only if, pooled over 2026 and 2027 classes, the day-3 top third by forecast + Hoop G "
                                "beats pick order's top third by 20+ points of hit rate",
          "calls_2026_board": {r_.display_name: r_.call for _, r_ in brd.iterrows()},
          "decision_date": "after Week 18 of the 2026 regular season"}
blob = json.dumps(frozen, sort_keys=True).encode()
HASH = hashlib.sha256(blob).hexdigest()[:12]
(OUT / "frozen_rule.json").write_text(json.dumps({**frozen, "sha256": hashlib.sha256(blob).hexdigest()}, indent=1))
TXT["hash"] = HASH

# ============================================================================= round-25 additions (judges' top_changes)
# (R1) Joint model: Hoop G and wave-drill lean together, on the held-out pick + role residual. Separates lean on the
#      hoop arc from general lateral body control. Coefficients per sd, class-stratified refit bootstrap (95%), and
#      a within-class-and-role shuffle of one measure holding the other fixed. Held-out R2 gain of each beyond the other.
JW = ANW.dropna(subset=["lean_z"]).reset_index(drop=True)
CLJ = JW.draft_year.values
YJ = JW.press_eb.values
XBJ = JW[HEADC].values
HJ, WJ = JW.hoop_z.values, JW.lean_z.values


def joint_coef(yv, xb, h_, w_, cl):
    r_ = yv - loco_fast(xb, yv, cl)
    A = np.c_[np.ones(len(r_)), h_, w_]
    return np.linalg.lstsq(A, r_, rcond=None)[0][1:]


JC = joint_coef(YJ, XBJ, HJ, WJ, CLJ)
_jg = [np.flatnonzero(CLJ == c) for c in classes]
_jr = np.random.default_rng(2501)
JB = []
for _ in range(NBOOT):
    ix = np.concatenate([_jr.choice(g_, len(g_)) for g_ in _jg])
    JB.append(joint_coef(YJ[ix], XBJ[ix], HJ[ix], WJ[ix], CLJ[ix]))
JB = np.array(JB)
_jcg = [np.flatnonzero((CLJ == c) & (JW.role.values == rl)) for c in classes for rl in ("EDGE", "interior")]
_jres = YJ - loco_fast(XBJ, YJ, CLJ)
JN = np.empty((NPERM, 2))
for i_ in range(NPERM):
    ix = np.arange(len(JW))
    for g_ in _jcg:
        ix[g_] = _jr.permutation(g_)
    JN[i_, 0] = np.linalg.lstsq(np.c_[np.ones(len(JW)), HJ[ix], WJ], _jres, rcond=None)[0][1]
    JN[i_, 1] = np.linalg.lstsq(np.c_[np.ones(len(JW)), HJ, WJ[ix]], _jres, rcond=None)[0][2]
put("jw_n", len(JW))
for j_, t_ in enumerate(["hoop", "wave"]):
    put(f"jw_{t_}", JC[j_] * PS, 1)
    put(f"jw_{t_}_lo95", np.percentile(JB[:, j_], 2.5) * PS, 1); put(f"jw_{t_}_hi95", np.percentile(JB[:, j_], 97.5) * PS, 1)
    put(f"jw_{t_}_p", (1 + (JN[:, j_] >= JC[j_]).sum()) / (NPERM + 1), 3)
_r2 = {}
for nm_, cols_ in [("b", []), ("h", [HJ]), ("w", [WJ]), ("hw", [HJ, WJ])]:
    _r2[nm_] = oos(YJ, loco_fast(np.column_stack([XBJ] + cols_), YJ, CLJ))[0]
put("jw_gain_h_given_w", _r2["hw"] - _r2["w"]); put("jw_gain_w_given_h", _r2["hw"] - _r2["h"])
put("jw_r2_hw", _r2["hw"]); put("jw_r2_b", _r2["b"])
put("jw_hoop_ppos", 100 * (JB[:, 0] > 0).mean(), 0)
put("jw_diff_ppos", 100 * (JB[:, 0] > JB[:, 1]).mean(), 0)
put("rho_hoop_wave_jw", stats.spearmanr(HJ, WJ)[0])

# (R2) Team-clustered bootstrap of the pick + role R2 gain, one-sided (share of team-resampled draws with no gain),
#      and the best-of-six adjusted p for the R2 gain: rows of the six-measure matrix shuffled jointly within class and
#      role, statistic = the largest held-out R2 gain among the six.
_cbp = np.asarray(CB["pick"])
put("cdr2_pick_lo95", np.percentile(_cbp, 2.5)); put("cdr2_pick_hi95", np.percentile(_cbp, 97.5))
put("cdr2_pick_p1", (1 + (_cbp <= 0).sum()) / (len(_cbp) + 1), 3)
put("cdr2_pick_ppos", 100 * (_cbp > 0).mean(), 0)
_cbs = np.asarray(CBS)
put("cbs_lo95", np.percentile(_cbs, 2.5) * PS, 1); put("cbs_hi95", np.percentile(_cbs, 97.5) * PS, 1)
put("cbs_p1", (1 + (_cbs <= 0).sum()) / (len(_cbs) + 1), 3)
GAIN6 = np.array([oos(y, loco_fast(np.c_[XB, X6[:, j_]], y))[0] - R2_0 for j_ in range(len(FKEYS))])
_gr6 = np.random.default_rng(2502)
MAXG = np.empty(NPERM)
for i_ in range(NPERM):
    ix = np.arange(len(AN))
    for g_ in cr_groups:
        ix[g_] = _gr6.permutation(g_)
    MAXG[i_] = max(oos(y, loco_fast(np.c_[XB, X6[ix, j_]], y))[0] - R2_0 for j_ in range(len(FKEYS)))
put("gain_pfw", (1 + (MAXG >= GAIN6[JH]).sum()) / (NPERM + 1), 3)
put("gain_best_is_hoop", int(np.argmax(GAIN6) == JH))
# best-of-six one-sided slope p from the same class-and-role shuffles (max slope over the six, rows permuted jointly)
_sr6 = np.random.default_rng(2503)
MAXS = np.empty(NPERM)
for i_ in range(NPERM):
    ix = np.arange(len(AN))
    for g_ in cr_groups:
        ix[g_] = _sr6.permutation(g_)
    MAXS[i_] = slopes(X6[ix], RES).max()
put("slope_pfw1", (1 + (MAXS >= obs[JH]).sum()) / (NPERM + 1), 3)

# (R3) Why is 2025 flat? (a) Hierarchical model of the class slopes: slope_c ~ N(mu, tau^2), observed with its
#      bootstrap variance; flat prior on mu, half-normal prior on tau (scale = the 2.25 action margin), grid posterior.
#      Gives the shrunk 2025 slope, tau, and the predictive slope for a new (2026) class. (b) Interaction test:
#      residual ~ Hoop G x class (F test). (c) Outcome timing: 2025 men have one (rookie) season. Same test with
#      ROOKIE-season pressure rate for every class, and with years 2+ for the 2023-24 classes.
BC = np.array([np.polyfit(AN.hoop_z.values[g_], RES[g_], 1)[0] for g_ in grp]) * PS
VC = np.array([np.var(np.asarray(BX[f"cls{c}"]) * PS) for c in classes])
TAU = np.linspace(0.001, 8, 800)
_lp = []
for t_ in TAU:
    w_ = 1 / (VC + t_ ** 2)
    m_ = (w_ * BC).sum() / w_.sum()
    _lp.append(0.5 * np.log(w_).sum() - 0.5 * np.log(w_.sum()) - 0.5 * (w_ * (BC - m_) ** 2).sum()
               - 0.5 * (t_ / (MARGIN * PS)) ** 2)
_lp = np.asarray(_lp)
_pt = np.exp(_lp - _lp.max()); _pt /= _pt.sum()
_hr = np.random.default_rng(2504)
_ts = _hr.choice(TAU, 20000, p=_pt)
_w = 1 / (VC[None, :] + _ts[:, None] ** 2)
_mus = (_w * BC).sum(1) / _w.sum(1) + _hr.standard_normal(len(_ts)) / np.sqrt(_w.sum(1))
_shr = _ts ** 2 / (_ts ** 2 + VC[2])
_b25 = _mus + _shr * (BC[2] - _mus) + _hr.standard_normal(len(_ts)) * np.sqrt(_shr * VC[2])
_new = _mus + _hr.standard_normal(len(_ts)) * _ts
put("hm_mu", np.median(_mus), 1); put("hm_mu_lo95", np.percentile(_mus, 2.5), 1)
put("hm_mu_hi95", np.percentile(_mus, 97.5), 1)
put("hm_tau", np.median(_ts), 1); put("hm_tau_hi95", np.percentile(_ts, 97.5), 1)
put("hm_b25", np.median(_b25), 1); put("hm_b25_lo95", np.percentile(_b25, 2.5), 1)
put("hm_b25_hi95", np.percentile(_b25, 97.5), 1)
put("hm_new", np.median(_new), 1); put("hm_new_lo95", np.percentile(_new, 2.5), 1)
put("hm_new_hi95", np.percentile(_new, 97.5), 1); put("hm_new_ppos", 100 * (_new > 0).mean(), 0)
for c, b_ in zip(classes, BC):
    put(f"hm_raw_{c}", b_, 1)
_Dc = np.column_stack([(AN.draft_year.values == c).astype(float) for c in classes[1:]])
_A0 = np.c_[np.ones(len(AN)), _Dc, AN.hoop_z.values]
_A1 = np.c_[_A0, _Dc * AN.hoop_z.values[:, None]]
_s0 = ((RES - _A0 @ np.linalg.lstsq(_A0, RES, rcond=None)[0]) ** 2).sum()
_s1 = ((RES - _A1 @ np.linalg.lstsq(_A1, RES, rcond=None)[0]) ** 2).sum()
_F = ((_s0 - _s1) / 2) / (_s1 / (len(AN) - _A1.shape[1]))
put("int_F", _F); put("int_p", stats.f.sf(_F, 2, len(AN) - _A1.shape[1]))
# (c) outcome timing
_sd = s.merge(dl[["nfl_id", "draft_year"]], on="nfl_id")
_rk_ = _sd[_sd.season == _sd.draft_year].groupby("nfl_id")[["pass_rush_snaps", "pressures"]].sum()
_lt_ = _sd[_sd.season > _sd.draft_year].groupby("nfl_id")[["pass_rush_snaps", "pressures"]].sum()


def eb_rate(t_):
    return (100 * (t_.pressures + Mw * mu) / (t_.pass_rush_snaps + Mw)).where(t_.pass_rush_snaps >= RATE_MIN)


RK = AN[["nfl_id", "draft_year", "hoop_z", "role"] + HEADC].merge(
    eb_rate(_rk_).rename("rk").reset_index(), on="nfl_id", how="left").merge(
    eb_rate(_lt_).rename("lt").reset_index(), on="nfl_id", how="left")


def timing_slope(d_, col, cl_set, seed):
    d_ = d_[d_.draft_year.isin(cl_set)].dropna(subset=[col]).reset_index(drop=True)
    cl_ = d_.draft_year.values
    yv = d_[col].values
    xb = d_[HEADC].values
    hz = d_.hoop_z.values

    def f(ix):
        r_ = yv[ix] - loco_fast(xb[ix], yv[ix], cl_[ix])
        return np.polyfit(hz[ix], r_, 1)[0]
    est = f(np.arange(len(d_)))
    rg = np.random.default_rng(seed)
    gg = [np.flatnonzero(cl_ == c) for c in np.unique(cl_)]
    bs = np.array([f(np.concatenate([rg.choice(g_, len(g_)) for g_ in gg])) for _ in range(NBOOT)])
    return est * PS, np.percentile(bs, 2.5) * PS, np.percentile(bs, 97.5) * PS, len(d_)


for tag_, col_, cs_, sd_ in [("rk2324", "rk", [2023, 2024], 2505), ("lt2324", "lt", [2023, 2024], 2506),
                              ("cr2324", None, [2023, 2024], 2507)]:
    if col_ is None:
        RK["cr"] = AN.set_index("nfl_id").loc[RK.nfl_id, "press_eb"].values
        col_ = "cr"
    e_, l_, h_, n_ = timing_slope(RK, col_, cs_, sd_)
    put(f"tm_{tag_}", e_, 1); put(f"tm_{tag_}_lo95", l_, 1); put(f"tm_{tag_}_hi95", h_, 1); put(f"tm_{tag_}_n", n_)
put("rk_rush_med_2025", float(AN[AN.draft_year == 2025].rush_snaps.median()), 0)
put("rk_rush_med_2324", float(AN[AN.draft_year < 2025].rush_snaps.median()), 0)

# (R4) Day 3: what the flag adds beyond pick order (unique names), and the expected 2026 race against pick order.
OVL = []
for g_ in g3:
    d_ = D3.iloc[g_]
    k_ = int(round(len(g_) / 3))
    th = set(d_.sort_values("fc_hoop", ascending=False, kind="stable").nfl_id.head(k_))
    tp = set(d_.sort_values("neg_pick", ascending=False, kind="stable").nfl_id.head(k_))
    med_ = d_.press_eb.median()
    hit = set(d_[d_.press_eb > med_].nfl_id)
    OVL.append((len(th & tp), len(th - tp), len(th - tp & hit), len(tp - th), len(tp - th & hit), k_))
OVL = np.array(OVL).sum(0)
put("ov_shared", int(OVL[0])); put("ov_add", int(OVL[1])); put("ov_add_hit", int(OVL[2]))
put("ov_drop", int(OVL[3])); put("ov_drop_hit", int(OVL[4])); put("ov_top", int(OVL[5]))
# 2026 race: simulated day-3 classes drawn from the 47 held-out day-3 rushers (class size = mean per class);
# within each, top third by forecast + Hoop G vs top third by pick, hit = above the simulated class's day-3 median.
NSIM26 = 5000
_r26 = np.random.default_rng(2026)
NC26 = int(round(len(D3) / 3))
_v1, _v2, _yy = D3.fc_hoop.values, D3.neg_pick.values, D3.press_eb.values
_k26 = int(round(NC26 / 3))
_race = np.empty(NSIM26)
for i_ in range(NSIM26):
    ix = _r26.choice(len(D3), NC26)
    med_ = np.median(_yy[ix])
    a_ = (_yy[ix][np.argsort(-_v1[ix], kind="stable")[:_k26]] > med_).sum()
    b_ = (_yy[ix][np.argsort(-_v2[ix] + 1e-9 * _r26.standard_normal(NC26), kind="stable")[:_k26]] > med_).sum()
    _race[i_] = a_ - b_
put("r26_better", 100 * (_race > 0).mean(), 0); put("r26_tie", 100 * (_race == 0).mean(), 0)
put("r26_worse", 100 * (_race < 0).mean(), 0); put("r26_k", _k26); put("r26_n", NC26)
# expected 2026 flag count: Poisson-binomial over the observed day-3 share among hoop runners by class
_fl_cnt = ba.groupby("draft_year").short.sum().values
put("flags_2026_lo", int(_fl_cnt.min())); put("flags_2026_hi", int(_fl_cnt.max()))
_fl_rate = float((ba.short).mean())
put("flags_rate_pct", 100 * _fl_rate, 0)
_sim_fl = _r26.binomial(int(round(ba.groupby("draft_year").size().mean())), _fl_rate, NSIM26)
put("flags_2026_p10", int(np.percentile(_sim_fl, 10))); put("flags_2026_p90", int(np.percentile(_sim_fl, 90)))
# beat / par / miss count among the flagged 2024-25 day-3 rushers with 100+ NFL rushes (the Figure 3 rows)
put("n_beat", int((brd.call == "Go").sum())); put("n_fellbelow", int((brd.call == "No-go").sum()))
put("n_par", int((brd.call == "Hold").sum()))

GA = GF.copy()
GA["corner"] = (GA.g_bend >= CT).astype(float).where(GA.g_bend.notna())
GA = GA.merge(g.drop_duplicates(K)[K + ["player_get_off", "lined_up_position"]], on=K, how="left")
# ============================================================================= round-26 additions (judges' top_changes)
# (N1) Honest selection: the best-of-six choice is repeated INSIDE each leave-one-class-out training fold. In the two
#      training classes, every one of the six measures is scored by inner leave-one-class-out (fit on one training
#      class, forecast the other) on top of pick + role; the measure with the lower inner error is fitted on both
#      training classes and forecasts the held-out class. Reported: held-out R2 gain of that whole procedure over pick +
#      role, which measure each fold chose, a class-stratified bootstrap (95%) and a within-class-and-role shuffle p
#      (the six-measure rows shuffled jointly, the whole selection rerun every time).
import statsmodels.formula.api as smf


def nested_pick(Xb, X6m, yv, cl):
    pred = np.empty(len(yv))
    pick_ = []
    for c in np.unique(cl):
        tr = cl != c
        err = [((yv[tr] - loco_fast(np.c_[Xb[tr], X6m[tr, j_]], yv[tr], cl[tr])) ** 2).sum()
               for j_ in range(X6m.shape[1])]
        j_ = int(np.argmin(err))
        pick_.append(j_)
        A = np.c_[np.ones(tr.sum()), Xb[tr], X6m[tr, j_]]
        b = np.linalg.lstsq(A, yv[tr], rcond=None)[0]
        pred[~tr] = np.c_[np.ones((~tr).sum()), Xb[~tr], X6m[~tr, j_]] @ b
    return pred, pick_


NSP, NSCH = nested_pick(XB, X6, y, CLV)
NS_GAIN = oos(y, NSP)[0] - R2_0
put("ns_gain", NS_GAIN); put("ns_r2", oos(y, NSP)[0]); put("ns_r2_0", R2_0)
put("ns_hoop_folds", int(sum(j_ == JH for j_ in NSCH)))
put("ns_hoopdrill_folds", int(sum(FAM[j_][1].startswith("Hoop") for j_ in NSCH)))
TXT["ns_choice"] = ", ".join(f"{c}: {FAM[j_][1]}" for c, j_ in zip(classes, NSCH))
# held-out slope of the measure each fold chose, on the pick + role residual (same units as the headline)
_xch = np.empty(len(AN))
for c, j_ in zip(classes, NSCH):
    _xch[CLV == c] = X6[CLV == c, j_]
NS_SLOPE = np.polyfit(_xch, RES, 1)[0] * PS
_nr = np.random.default_rng(2601)
NSB, NSBS, NSBH = [], [], []
for _ in range(NBOOT):
    ix = np.concatenate([_nr.choice(g_, len(g_)) for g_ in grp])
    yb, xb_, x6_, cl_ = y[ix], XB[ix], X6[ix], CLV[ix]
    p1, ch_ = nested_pick(xb_, x6_, yb, cl_)
    p0 = loco_fast(xb_, yb, cl_)
    NSB.append(oos(yb, p1)[0] - oos(yb, p0)[0])
    xc_ = np.empty(len(ix))
    for c, j_ in zip(np.unique(cl_), ch_):
        xc_[cl_ == c] = x6_[cl_ == c, j_]
    NSBS.append(np.polyfit(xc_, yb - p0, 1)[0] * PS)
    NSBH.append(np.mean([j_ == JH for j_ in ch_]))
NSB, NSBS = np.asarray(NSB), np.asarray(NSBS)
put("ns_gain_lo95", np.percentile(NSB, 2.5)); put("ns_gain_hi95", np.percentile(NSB, 97.5))
put("ns_gain_ppos", 100 * (NSB > 0).mean(), 0)
put("ns_slope", NS_SLOPE, 1); put("ns_slope_lo95", np.percentile(NSBS, 2.5), 1)
put("ns_slope_hi95", np.percentile(NSBS, 97.5), 1)
put("ns_hoop_share_boot", 100 * np.mean(NSBH), 0)
NNS = int(os.environ.get("BDB_NNS", 2000))
_np_ = np.random.default_rng(2602)
NS_NULL = np.empty(NNS)
for i_ in range(NNS):
    ix = np.arange(len(AN))
    for g_ in cr_groups:
        ix[g_] = _np_.permutation(g_)
    NS_NULL[i_] = oos(y, nested_pick(XB, X6[ix], y, CLV)[0])[0] - R2_0
put("ns_p", (1 + (NS_NULL >= NS_GAIN).sum()) / (NNS + 1), 3); put("n_ns_perm", NNS)

# (N2) Scheme, stance and snap-load controls added to the forecast (leave-one-class-out, same rushers):
#      stand-up proxy = listed as OLB (3-4 stand-up edge) rather than DE/DT; snap load = NFL pass rushes per game
#      played (log); club pass-rush environment = shrunk pressure rate of the OTHER cohort linemen on his latest club
#      (leave-self-out; league mean when he is alone). Snap load is partly an outcome (good rushers play more), so this
#      row is a stress test, not the headline.
_gm_ = seasons[seasons.position_group == "DL"].groupby("nfl_id").agg(gms=("games", "sum"), prs=("pass_rush_snaps", "sum"))
AN["standup"] = AN.nfl_position.eq("OLB").astype(float)
AN["load"] = np.log1p(AN.nfl_id.map(_gm_.prs / _gm_.gms.clip(lower=1)).fillna(0))
_tm_ = co30.assign(t_=co30.latest_team.fillna("FA"))
_tsum = _tm_.groupby("t_").press_eb.agg(["sum", "count"])
_t_ = AN.latest_team.fillna("FA")
_s_ = _t_.map(_tsum["sum"]) - AN.press_eb
_n_ = _t_.map(_tsum["count"]) - 1
AN["club_env"] = np.where((_n_ > 0) & (_t_ != "FA"), _s_ / _n_.clip(lower=1), np.nan)
AN["club_env"] = AN.club_env.fillna(float(co30.press_eb.mean()))
CTRL = HEADC + ["standup", "load", "club_env"]
XCT = AN[CTRL].values
PCT = loco_fast(XCT, y)
RCT = y - PCT
put("ct_r2", oos(y, PCT)[0]); put("ct_r2_h", oos(y, loco_fast(np.c_[XCT, AN.hoop_z.values], y))[0])
put("ct_slope", np.polyfit(AN.hoop_z.values, RCT, 1)[0] * PS, 1)
_cr_ = np.random.default_rng(2603)
_cb_ = []
for _ in range(NBOOT):
    ix = np.concatenate([_cr_.choice(g_, len(g_)) for g_ in grp])
    _cb_.append(np.polyfit(AN.hoop_z.values[ix], y[ix] - loco_fast(XCT[ix], y[ix], CLV[ix]), 1)[0] * PS)
put("ct_slope_lo95", np.percentile(_cb_, 2.5), 1); put("ct_slope_hi95", np.percentile(_cb_, 97.5), 1)
put("n_standup", int(AN.standup.sum())); put("n_club_env", int(((_n_ > 0) & (_t_ != "FA")).sum()))
# headline slope with a 95% bootstrap interval for every Fig 2 row in one place
put("ho_hoop_lo95", np.percentile(HO["hoop_z"]["bs"], 2.5) * PS, 1)
put("ho_hoop_hi95", np.percentile(HO["hoop_z"]["bs"], 97.5) * PS, 1)
# wave-lean joint model: R2 gain of Hoop G beyond wave lean, class-stratified bootstrap 95%
_jr2 = np.random.default_rng(2604)
_jgb = []
for _ in range(NBOOT):
    ix = np.concatenate([_jr2.choice(g_, len(g_)) for g_ in _jg])
    yb = YJ[ix]
    _jgb.append(oos(yb, loco_fast(np.c_[XBJ[ix], HJ[ix], WJ[ix]], yb, CLJ[ix]))[0]
                - oos(yb, loco_fast(np.c_[XBJ[ix], WJ[ix]], yb, CLJ[ix]))[0])
put("jw_gain_lo95", np.percentile(_jgb, 2.5)); put("jw_gain_hi95", np.percentile(_jgb, 97.5))

# (N3) Direct drill-to-play test on every sampled rush (random half; 2-s window). Bend on a rush is self-normalized:
#      his hardest half-second of lean on that rush, as a z-score within HIS OWN sampled rushes (so a man's size, role
#      and blockers drop out). A "hoop-tight" rush is one whose turn radius is within twice his own hoop radius
#      (tightness matched to his drill). Linear-probability mixed models with a random intercept per rusher:
#        M1  pressure ~ own-lean z
#        M2  pressure ~ own-lean z * Hoop G + edge share         (does Hoop G mark men whose lean ends in pressure?)
#        M3  pressure ~ hoop-tight rush * Hoop G + edge share
#      95% intervals: rusher-cluster bootstrap of the same fits (fixed effects by within-rusher OLS for speed).
RU = GR[GR.nfl_id.isin(AN.nfl_id) & GR.g_bend.notna()].merge(GT[K + ["t_rad"]], on=K, how="left")
RU = RU.merge(AN[["nfl_id", "hoop_z", "edge_share", "draft_year"]], on="nfl_id").merge(
    dl[["nfl_id", "hoop_rad"]], on="nfl_id", how="left")
RU = RU[RU.groupby("nfl_id").g_bend.transform("count") >= 5].reset_index(drop=True)
RU["wz"] = RU.groupby("nfl_id").g_bend.transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
RU["tight"] = (RU.t_rad <= 2 * RU.hoop_rad).astype(float)
RU["press100"] = 100 * RU.press
put("mm_n_rush", len(RU)); put("mm_n_men", RU.nfl_id.nunique())
put("mm_rush_med", float(RU.groupby("nfl_id").size().median()), 0)
put("mm_tight_pct", 100 * RU.tight.mean(), 0)
put("mm_press_pct", RU.press100.mean(), 1)
MM, MMK = {}, {}
for nm_, f_ in [("m1", "press100 ~ wz"), ("m2", "press100 ~ wz * hoop_z + edge_share"),
                ("m3", "press100 ~ tight * hoop_z + edge_share")]:
    try:
        r_ = smf.mixedlm(f_, RU, groups=RU.nfl_id).fit(reml=True, method=["powell", "lbfgs"])
        MMK[nm_] = "mixed"
    except np.linalg.LinAlgError:     # random-intercept variance at zero: same fixed effects, rusher-clustered errors
        r_ = smf.ols(f_, RU).fit(cov_type="cluster", cov_kwds={"groups": RU.nfl_id})
        MMK[nm_] = "ols-cluster"
    MM[nm_] = r_
    ci_ = r_.conf_int()
    for t_ in r_.fe_params.index:
        if t_ in ("Intercept", "edge_share"):
            continue
        k_ = f"mm_{nm_}_" + t_.replace(":", "x").replace("_", "")
        put(k_, r_.fe_params[t_], 1); put(k_ + "_lo", ci_.loc[t_, 0], 1); put(k_ + "_hi", ci_.loc[t_, 1], 1)
        put(k_ + "_p", r_.pvalues[t_], 3)
    if MMK[nm_] == "mixed":
        put(f"mm_{nm_}_re_sd", float(np.sqrt(max(r_.cov_re.iloc[0, 0], 0))), 1)
# rusher-cluster bootstrap of the two interaction terms (within-rusher OLS: same estimand as the random-intercept
# model's within part; robust to the linear-probability error structure)
_ids_ = RU.nfl_id.unique()
_by_ = {i: RU.index[RU.nfl_id == i].values for i in _ids_}


def _within(d_, x_):
    dm = d_[[x_, "press100"]].values - d_.groupby("nfl_id")[[x_, "press100"]].transform("mean").values
    X = np.c_[dm[:, 0], dm[:, 0] * d_.hoop_z.values]
    return np.linalg.lstsq(X, dm[:, 1], rcond=None)[0]


_mr = np.random.default_rng(2605)
MMB = {"wz": [], "tight": []}
for _ in range(NBOOT):
    pick_ = _mr.choice(_ids_, len(_ids_))
    d_ = pd.concat([RU.loc[_by_[i]].assign(nfl_id=j) for j, i in enumerate(pick_)])
    for x_ in MMB:
        MMB[x_].append(_within(d_, x_))
for x_ in MMB:
    b_ = np.asarray(MMB[x_])
    e_ = _within(RU, x_)
    put(f"cb_{x_}_main", e_[0], 1); put(f"cb_{x_}_int", e_[1], 1)
    put(f"cb_{x_}_int_lo", np.percentile(b_[:, 1], 2.5), 1); put(f"cb_{x_}_int_hi", np.percentile(b_[:, 1], 97.5), 1)
    put(f"cb_{x_}_main_lo", np.percentile(b_[:, 0], 2.5), 1); put(f"cb_{x_}_main_hi", np.percentile(b_[:, 0], 97.5), 1)
# for Fig 4: pressure rate by own-lean quarter, Hoop G top third vs the rest, rusher-cluster bootstrap 95%
RU["q"] = RU.groupby("nfl_id").wz.transform(lambda v: pd.qcut(v.rank(method="first"), 4, labels=False))
RU["ht"] = RU.groupby("draft_year").hoop_z.transform(lambda v: v.rank(pct=True) > 2 / 3)
F4 = {}
for ht_ in (True, False):
    sub_ = RU[RU.ht == ht_]
    ids_ = sub_.nfl_id.unique()
    tab_ = sub_.groupby(["nfl_id", "q"]).press100.agg(["sum", "count"]).unstack(fill_value=0)
    est_ = tab_["sum"].sum() / tab_["count"].sum()
    bs_ = []
    for _ in range(NBOOT):
        ii = _mr.choice(len(tab_), len(tab_))
        bs_.append(tab_["sum"].values[ii].sum(0) / tab_["count"].values[ii].sum(0))
    bs_ = np.asarray(bs_)
    F4[ht_] = (est_.values, np.percentile(bs_, 2.5, 0), np.percentile(bs_, 97.5, 0), len(ids_), int(len(sub_)))
    t_ = "top" if ht_ else "rest"
    put(f"f4_{t_}_q4", est_.values[3], 0); put(f"f4_{t_}_q1", est_.values[0], 0)
    put(f"f4_{t_}_n", len(ids_)); put(f"f4_{t_}_nr", int(len(sub_)))
# between-man: does Hoop G predict an alternative NFL bend measure (his 90th-percentile rush lean, and his share of
# hoop-tight rushes)? Partial Spearman given weight and edge share, class-stratified bootstrap.
_alt = RU.groupby("nfl_id").agg(bend90=("g_bend", lambda v: np.percentile(v, 90)), tight_sh=("tight", "mean")).reset_index()
_dla = dl.merge(_alt, on="nfl_id", how="inner")
for col_, sd_ in [("bend90", 2606), ("tight_sh", 2607)]:
    r_, lo_, hi_, n_ = boot_psp(_dla, "hoop_z", col_, Zc, sd_)
    put(f"alt_{col_}", r_); put(f"alt_{col_}_lo", lo_); put(f"alt_{col_}_hi", hi_); put(f"alt_{col_}_n", n_)
# noise ceiling for the per-man NFL bend mean: split-half reliability on these rushes (already REL["g_bend"])
put("alt_ceiling", np.sqrt(REL["hoop_z"] * REL["g_bend"]))

# (N4) Day 3, all 47 held-out day-3/undrafted rushers: pooled within-class rank correlation with pressure rate and an
#      out-of-class Brier score for "beats his class's day-3 median" (logistic calibration of each ranker's within-class
#      percentile, fitted on the other two classes). Pick order vs forecast + Hoop G, class-stratified bootstrap 95%.
D3["hit"] = (D3.press_eb > D3.groupby("draft_year").press_eb.transform("median")).astype(float)


def _logit_fit(x_, y_):
    b = np.zeros(2)
    X = np.c_[np.ones(len(x_)), x_]
    for _ in range(50):
        p_ = 1 / (1 + np.exp(-X @ b))
        W = p_ * (1 - p_) + 1e-9
        b = b + np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(2), X.T @ (y_ - p_))
    return b


def brier(d_, col_):
    pct_ = d_.groupby("draft_year")[col_].rank(pct=True).values
    out_ = np.empty(len(d_))
    cl_ = d_.draft_year.values
    for c in np.unique(cl_):
        m_ = cl_ == c
        b = _logit_fit(pct_[~m_], d_.hit.values[~m_])
        out_[m_] = 1 / (1 + np.exp(-(b[0] + b[1] * pct_[m_])))
    return float(np.mean((out_ - d_.hit.values) ** 2))


BRI = {c_: brier(D3, c_) for c_ in ["neg_pick", "fc", "fc_hoop"]}
_br3 = np.random.default_rng(2608)
BRB = {c_: [] for c_ in BRI}
RHB = {c_: [] for c_ in BRI}
for _ in range(NBOOT):
    ix = np.concatenate([_br3.choice(g_, len(g_)) for g_ in g3])
    db = D3.iloc[ix].reset_index(drop=True)
    db["hit"] = (db.press_eb > db.groupby("draft_year").press_eb.transform("median")).astype(float)
    gb = [np.flatnonzero(db.draft_year.values == c) for c in classes]
    for c_ in BRI:
        BRB[c_].append(brier(db, c_))
        RHB[c_].append(pooled_rho(db, c_, gb))
for c_, t_ in [("neg_pick", "pick"), ("fc", "fc"), ("fc_hoop", "fch")]:
    put(f"bri_{t_}", BRI[c_], 3)
    put(f"bri_{t_}_lo", np.percentile(BRB[c_], 2.5), 3); put(f"bri_{t_}_hi", np.percentile(BRB[c_], 97.5), 3)
    put(f"rho95_{t_}_lo", np.nanpercentile(RHB[c_], 2.5)); put(f"rho95_{t_}_hi", np.nanpercentile(RHB[c_], 97.5))
_dbr = np.asarray(BRB["neg_pick"]) - np.asarray(BRB["fc_hoop"])        # positive = Hoop G forecast better
put("bri_gain", BRI["neg_pick"] - BRI["fc_hoop"], 3)
put("bri_gain_lo", np.percentile(_dbr, 2.5), 3); put("bri_gain_hi", np.percentile(_dbr, 97.5), 3)
put("bri_gain_ppos", 100 * (_dbr > 0).mean(), 0)
_drh = np.asarray(RHB["fc_hoop"]) - np.asarray(RHB["neg_pick"])
put("rho_gain_lo95", np.nanpercentile(_drh, 2.5)); put("rho_gain_hi95", np.nanpercentile(_drh, 97.5))
put("rho_gain_ppos", 100 * np.nanmean(_drh > 0), 0)
put("bri_coin", 0.25)

# (N5) The names the flag adds beyond pick order's top third, ranked, with a tape cue chosen by fixed rules from the
#      play log (no film). Rules: bends on Sunday = corner-turn rate on random snaps >= 25%; wins from bend = at least
#      half of his pressured snaps (all sampled) were corner turns; quick = median NGS get-off below the sampled-lineman
#      median. Cue: bends + wins from bend -> "dip and flatten: confirm the finish at the top of the arc";
#      bends, not from bend -> "bends but finishes elsewhere: check hands at the top"; no bend + quick ->
#      "wins off the ball, not around it: get-off and first step"; otherwise -> "no bend, no get-off edge: power and
#      effort only, lowest priority".
_ga_med = float(GA.player_get_off.median())
ADD = []
for g_ in g3:
    d_ = D3.iloc[g_].copy()
    k_ = int(round(len(g_) / 3))
    d_["r_fch"] = d_.fc_hoop.rank(ascending=False, method="first")
    th = set(d_.sort_values("fc_hoop", ascending=False, kind="stable").nfl_id.head(k_))
    tp = set(d_.sort_values("neg_pick", ascending=False, kind="stable").nfl_id.head(k_))
    ADD.append(d_[d_.nfl_id.isin(th - tp)].assign(cls_n=len(g_)))
ADD = pd.concat(ADD)
ADD["pct_fch"] = ADD.r_fch / ADD.cls_n
_rows7 = []
for _, r_ in ADD.iterrows():
    ra = GR[GR.nfl_id == r_.nfl_id].dropna(subset=["corner"])
    aa = GA[GA.nfl_id == r_.nfl_id]
    pp = aa[(aa.press == 1) & aa.corner.notna()]
    cr_ = 100 * ra.corner.mean() if len(ra) else np.nan
    fb_ = len(pp) > 0 and pp.corner.sum() / len(pp) >= 0.5
    go_ = float(aa.player_get_off.median())
    if cr_ >= 25 and fb_:
        cue = "dip and flatten: confirm the finish at the top of the arc"
    elif cr_ >= 25:
        cue = "bends but finishes elsewhere: check hands at the top"
    elif go_ < _ga_med:
        cue = "wins off the ball, not around it: get-off and first step"
    else:
        cue = "no bend, no get-off edge: power and effort, lowest priority"
    ov_ = (r_.press_eb - r_.fc) * PS
    p_ = r_.press_eb / 100
    se_ = 100 * np.sqrt(p_ * (1 - p_) / (r_.rush_snaps + Mw + 1)) * PS
    _rows7.append(dict(nfl_id=int(r_.nfl_id), name=r_.display_name, team=r_.latest_team if pd.notna(r_.latest_team)
                       else "FA", yr=int(r_.draft_year), pick=r_.pick, role=r_.role, hz=r_.hoop_z, ov=ov_,
                       lo=ov_ - 1.96 * se_, hi=ov_ + 1.96 * se_, hit=int(r_.hit), ct=int(ra.corner.sum()),
                       ctn=len(ra), pc=int(pp.corner.sum()), pn=len(pp), go=go_, cue=cue, pct=r_.pct_fch))
T7 = pd.DataFrame(_rows7).sort_values(["pct", "hz"], ascending=[True, False]).reset_index(drop=True)
assert len(T7) == int(NUM["ov_add"])
put("t7_hits", int(T7.hit.sum())); put("t7_n", len(T7))
put("t7_bend_cue", int(T7.cue.str.startswith("dip").sum()))
_t7 = []
for i_, r_ in T7.iterrows():
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
    _t7.append(f"| {i_ + 1}. {r_['name']} ({r_.team}), {r_.yr} {pk} | {fm(r_.hz)} | {fm(r_.ov)} | "
               f"{'yes' if r_.hit else 'no'} | {r_.ct} of {r_.ctn} | {r_.cue} |")
    put(f"t7_{i_}_hz", r_.hz, 1); put(f"t7_{i_}_ov", r_.ov, 1)
    put(f"t7_{i_}_lo", r_.lo, 1); put(f"t7_{i_}_hi", r_.hi, 1)
    put(f"t7_{i_}_ct", int(r_.ct)); put(f"t7_{i_}_ctn", int(r_.ctn)); put(f"t7_{i_}_go", r_.go)
    TXT[f"t7_{i_}_pick"] = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
TXT["t7_rows"] = "\n".join(_t7)
TXT["t7_names"] = list(T7.name)
_ib, _iw = int(T7.ov.idxmax()), int(T7.ov.idxmin())
TXT["t7_best"], TXT["t7_worst"] = T7.name[_ib], T7.name[_iw]
TXT["t7_best_pick"] = "UDFA" if T7.pick[_ib] >= UDFA_PICK else f"#{int(T7.pick[_ib])}"
put("t7_best_ov", T7.ov[_ib], 1); put("t7_worst_ov", T7.ov[_iw], 1)
put("t7_best_ct", int(T7.ct[_ib])); put("t7_best_ctn", int(T7.ctn[_ib]))
put("t7_worst_go", T7.go[_iw]); TXT["t7_worst_cue"] = T7.cue[_iw]
put("getoff_med_all", _ga_med)
# 2026: the Combine tracking for the 2026 class is not in this data, so no 2026 names can be listed yet.
put("n_class_2026_tracked", int((players.draft_year == 2026).sum()))
assert NUM["n_class_2026_tracked"] == 0     # the template says no 2026 names can be listed yet
assert T7.cue[int(T7.ov.idxmin())].startswith("no bend")     # prose: "lowest priority"
assert "elsewhere" in T7.cue[int(T7.ov.idxmax())]     # prose: "check where his finishes come from"

# ============================================================================= round-27 additions (judges' top_changes)
# (S1) No-selection headline: all six pre-listed drill measures enter one ridge model on top of pick + role. The drill
#      coefficients are penalized (pick + role are not); the penalty is chosen inside the two training classes by inner
#      leave-one-class-out; the held-out class is then forecast. Reported: held-out R2 gain over pick + role, a
#      class-stratified bootstrap (95%) that reruns the penalty choice, and a within-class-and-role shuffle p (the six
#      measures shuffled jointly, whole procedure rerun). Also the pre-registered equal-weight version (mean of the six
#      sign-aligned z-scores, no fitted drill weights). The share of the pooled ridge weight on the three hoop measures
#      says whether the hoop carries the signal without anyone choosing it.
from scipy.special import expit

ALPH6 = np.r_[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 1e3]


def r6_fit(Xb, Xd, yv, alpha):
    mu_, sd_ = Xd.mean(0), Xd.std(0) + 1e-9
    A = np.c_[np.ones(len(yv)), Xb, (Xd - mu_) / sd_]
    P = np.zeros(A.shape[1]); P[1 + Xb.shape[1]:] = alpha * len(yv)
    return np.linalg.solve(A.T @ A + np.diag(P), A.T @ yv), mu_, sd_


def r6_pred(f_, Xb, Xd):
    b, mu_, sd_ = f_
    return np.c_[np.ones(len(Xb)), Xb, (Xd - mu_) / sd_] @ b


def nested_ridge6(Xb, Xd, yv, cl):
    pred, drill, al = np.empty(len(yv)), np.empty(len(yv)), []
    k0 = 1 + Xb.shape[1]
    for c in np.unique(cl):
        tr = cl != c
        Xbt, Xdt, yt, tcl = Xb[tr], Xd[tr], yv[tr], cl[tr]
        err = []
        for a_ in ALPH6:
            e_ = 0.0
            for ic in np.unique(tcl):
                m_ = tcl == ic
                e_ += ((yt[m_] - r6_pred(r6_fit(Xbt[~m_], Xdt[~m_], yt[~m_], a_), Xbt[m_], Xdt[m_])) ** 2).sum()
            err.append(e_)
        a_ = ALPH6[int(np.argmin(err))]
        al.append(a_)
        f_ = r6_fit(Xbt, Xdt, yt, a_)
        pred[~tr] = r6_pred(f_, Xb[~tr], Xd[~tr])
        drill[~tr] = ((Xd[~tr] - f_[1]) / f_[2]) @ f_[0][k0:]
    return pred, drill, al


R6P, R6D, R6A = nested_ridge6(XB, X6, y, CLV)
R6_GAIN = oos(y, R6P)[0] - R2_0
EW = X6.mean(1)                                   # equal-weight composite, z within class again
EWZ = np.empty(len(EW))
for g_ in grp:
    EWZ[g_] = (EW[g_] - EW[g_].mean()) / EW[g_].std()
EW_GAIN = oos(y, loco_fast(np.c_[XB, EWZ], y))[0] - R2_0
# drill score of the ridge model in sd within class, and its held-out slope in pressures per season
R6Z = np.empty(len(R6D))
for g_ in grp:
    R6Z[g_] = (R6D[g_] - R6D[g_].mean()) / (R6D[g_].std() + 1e-9)
put("r6_gain", R6_GAIN); put("r6_r2", oos(y, R6P)[0]); put("ew_gain", EW_GAIN)
put("r6_slope", np.polyfit(R6Z, RES, 1)[0] * PS, 1); put("ew_slope", np.polyfit(EWZ, RES, 1)[0] * PS, 1)
TXT["r6_alphas"] = [float(a_) for a_ in R6A]
# pooled ridge fit (penalty = median of the fold choices): share of absolute standardized weight on hoop measures
_fpool = r6_fit(XB, X6, y, float(np.median(R6A)))
_w6 = _fpool[0][1 + XB.shape[1]:]
HOOPJ = [j_ for j_, f_ in enumerate(FAM) if f_[1].startswith("Hoop")]
put("r6_hoop_share", 100 * np.abs(_w6[HOOPJ]).sum() / np.abs(_w6).sum(), 0)
put("r6_hoopg_share", 100 * abs(_w6[JH]) / np.abs(_w6).sum(), 0)
for j_, f_ in enumerate(FAM):
    put(f"r6_w_{f_[0].replace('_z', '')}", _w6[j_] * PS, 1)
_r6r = np.random.default_rng(2701)
R6B, EWB, R6SB, EWSB = [], [], [], []
for _ in range(NBOOT):
    ix = np.concatenate([_r6r.choice(g_, len(g_)) for g_ in grp])
    yb, xb_, x6_, cl_ = y[ix], XB[ix], X6[ix], CLV[ix]
    p0 = loco_fast(xb_, yb, cl_)
    r20 = oos(yb, p0)[0]
    p1, d1, _a = nested_ridge6(xb_, x6_, yb, cl_)
    R6B.append(oos(yb, p1)[0] - r20)
    ew_ = EWZ[ix]
    EWB.append(oos(yb, loco_fast(np.c_[xb_, ew_], yb, cl_))[0] - r20)
    dz = np.empty(len(ix))
    for c in np.unique(cl_):
        m_ = cl_ == c
        dz[m_] = (d1[m_] - d1[m_].mean()) / (d1[m_].std() + 1e-9)
    R6SB.append(np.polyfit(dz, yb - p0, 1)[0] * PS)
    EWSB.append(np.polyfit(ew_, yb - p0, 1)[0] * PS)
for t_, b_ in [("r6_gain", R6B), ("ew_gain", EWB), ("r6_slope", R6SB), ("ew_slope", EWSB)]:
    b_ = np.asarray(b_)
    nd_ = 1 if "slope" in t_ else 2
    put(f"{t_}_lo95", np.percentile(b_, 2.5), nd_); put(f"{t_}_hi95", np.percentile(b_, 97.5), nd_)
    put(f"{t_}_ppos", 100 * (b_ > 0).mean(), 0)
NR6 = int(os.environ.get("BDB_NR6", 2000))
_r6p = np.random.default_rng(2702)
R6_NULL, EW_NULL = np.empty(NR6), np.empty(NR6)
for i_ in range(NR6):
    ix = np.arange(len(AN))
    for g_ in cr_groups:
        ix[g_] = _r6p.permutation(g_)
    R6_NULL[i_] = oos(y, nested_ridge6(XB, X6[ix], y, CLV)[0])[0] - R2_0
    EW_NULL[i_] = oos(y, loco_fast(np.c_[XB, EWZ[ix]], y))[0] - R2_0
put("r6_p", (1 + (R6_NULL >= R6_GAIN).sum()) / (NR6 + 1), 3)
put("ew_p", (1 + (EW_NULL >= EW_GAIN).sum()) / (NR6 + 1), 3); put("n_r6_perm", NR6)

# (S2) One joint hierarchical model for pressure (logistic, Laplace approximation, rusher random intercept).
#      Two kinds of binomial rows share each rusher's intercept u_i ~ N(0, tau^2):
#        career rows: his season-stat pressures out of his rushes, minus the sampled rushes below (no double count);
#        rush rows:   every randomly sampled rush (pressured or not), with his own-rush lean z, its product with
#                     Hoop G, and the defence's coverage on that snap (man vs zone) and a long-yardage flag.
#      Shared fixed effects: log pick, edge share, Hoop G. Separate intercepts per row kind.
#      Hoop G enters measurement-error corrected (regression calibration): E[true | observed] = rel_i x observed,
#      rel_i = one-lap reliability (two turns of each lap) stepped up to his lap count, put in sd of TRUE Hoop G.
#      The one-lap reliability's own bootstrap draws are propagated into the interval.
#      tau: Laplace marginal likelihood over a grid. Leave-one-class-out: fit on two classes, forecast the held-out
#      class's career rate with u integrated out; held-out R2 gain over the same model without the Hoop G terms.
GJ = g.drop_duplicates(K)[K + ["team_coverage_type", "down", "yards_to_go"]]
RJ = RU.merge(GJ, on=K, how="left")
RJ["man"] = RJ.team_coverage_type.fillna("").str.contains("MAN").astype(float)
RJ["long"] = ((RJ.down >= 3) & (RJ.yards_to_go >= 5)).astype(float)
put("jm_man_pct", 100 * RJ.man.mean(), 0); put("jm_cov_known", 100 * RJ.team_coverage_type.notna().mean(), 0)
_samp = RJ.groupby("nfl_id").agg(sk=("press", "sum"), sn=("press", "size"))
JA = AN.merge(_samp, left_on="nfl_id", right_index=True, how="left").fillna({"sk": 0, "sn": 0})
JA["kc"] = (JA.pressures - JA.sk).clip(lower=0)
JA["nc"] = (JA.rush_snaps - JA.sn).clip(lower=JA.kc + 1)
REL_BAR = float(AN.rel_i.mean())


def hoop_rc(d_, r1=None):
    rel_ = d_.rel_i.values if r1 is None else sb(r1, d_.hoop_k.values)
    rb_ = REL_BAR if r1 is None else float(np.mean(rel_))
    return rel_ * d_.hoop_z.values / np.sqrt(rb_)


def jm_design(ja, rj, hz, hoop=True):
    """rows: career (one per man) then rush. columns: c_int, r_int, log_pick, edge, [hoop], wz, [wz x hoop], man, long"""
    idm = {v: i for i, v in enumerate(ja.nfl_id.values)}
    hr = hz[[idm[i] for i in rj.nfl_id.values]]
    nC, nR = len(ja), len(rj)
    Xc = [np.ones(nC), np.zeros(nC), ja.log_pick.values, ja.edge_share.values]
    Xr = [np.zeros(nR), np.ones(nR), rj.nfl_id.map(ja.set_index("nfl_id").log_pick).values, rj.edge_share.values]
    if hoop:
        Xc.append(hz); Xr.append(hr)
    Xc.append(np.zeros(nC)); Xr.append(rj.wz.values)
    if hoop:
        Xc.append(np.zeros(nC)); Xr.append(rj.wz.values * hr)
    Xc += [np.zeros(nC), np.zeros(nC)]; Xr += [rj.man.values, rj.long.values]
    X = np.r_[np.column_stack(Xc), np.column_stack(Xr)]
    ids = np.r_[np.arange(nC), np.array([idm[i] for i in rj.nfl_id.values])]
    kk = np.r_[ja.kc.values, rj.press.values]
    nn = np.r_[ja.nc.values, np.ones(nR)]
    return X, ids, kk, nn, nC


def glmm(X, ids, kk, nn, nmen, tau, th0=None):
    p = X.shape[1]
    D = np.c_[X, np.eye(nmen)[ids]]
    th = np.zeros(p + nmen) if th0 is None else th0.copy()
    pri = np.r_[np.full(p, 1 / 100.0), np.full(nmen, 1 / tau ** 2)]
    for _ in range(100):
        pr_ = expit(D @ th)
        gr_ = D.T @ (kk - nn * pr_) - pri * th
        H = D.T @ (D * (nn * pr_ * (1 - pr_))[:, None]) + np.diag(pri)
        st = np.linalg.solve(H, gr_)
        th = th + st
        if np.abs(st).max() < 1e-7:
            break
    pr_ = np.clip(expit(D @ th), 1e-12, 1 - 1e-12)
    ll = (kk * np.log(pr_) + (nn - kk) * np.log(1 - pr_)).sum()
    lm = ll - 0.5 * (th[p:] ** 2).sum() / tau ** 2 - nmen * np.log(tau) - 0.5 * np.linalg.slogdet(H)[1]
    return th, H, lm


TAUS = np.r_[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]


def glmm_tau(X, ids, kk, nn, nmen):
    best = None
    for t_ in TAUS:
        th, H, lm = glmm(X, ids, kk, nn, nmen, t_)
        if best is None or lm > best[3]:
            best = (th, H, t_, lm)
    return best


HZ_RC = hoop_rc(JA)
JX, JI, JK, JN, JNC = jm_design(JA, RJ, HZ_RC)
JTH, JH_, JTAU, _ = glmm_tau(JX, JI, JK, JN, len(JA))
JCOV = np.linalg.inv(JH_)[:JX.shape[1], :JX.shape[1]]
JNAMES = ["c_int", "r_int", "log_pick", "edge", "hoop", "wz", "wzxhoop", "man", "long"]
JB = dict(zip(JNAMES, JTH[:JX.shape[1]])); JSE = dict(zip(JNAMES, np.sqrt(np.diag(JCOV))))
# uncorrected Hoop G for comparison (same model, observed z)
_Xu = jm_design(JA, RJ, JA.hoop_z.values)
JTHU = glmm(*_Xu[:4], len(JA), JTAU)[0]
PC_ = float(JA.kc.sum() / JA.nc.sum()); PR_ = float(RJ.press.mean())
DP_C = PC_ * (1 - PC_) * SEASON           # logit -> pressures per 450-rush season, at the career mean rate
DP_R = PR_ * (1 - PR_) * 100              # logit -> points of pressure % on a sampled rush
# propagate the reliability uncertainty: refit at bootstrap draws of the one-lap reliability, mix with the Laplace se
_jr = np.random.default_rng(2703)
_mix = []
for r1 in _jr.choice(BLR_LAP, 200):
    hz_ = hoop_rc(JA, r1)
    th_ = glmm(*jm_design(JA, RJ, hz_)[:4], len(JA), JTAU, th0=JTH)[0]
    _mix.append(th_[[4, 6]] + _jr.standard_normal(2) * np.array([JSE["hoop"], JSE["wzxhoop"]]))
_mix = np.asarray(_mix)
put("jm_tau", JTAU); put("jm_n_men", len(JA)); put("jm_n_rush", len(RJ)); put("jm_rel_bar", REL_BAR)
put("jm_hoop", JB["hoop"] * DP_C, 1)
put("jm_hoop_lo", np.percentile(_mix[:, 0], 2.5) * DP_C, 1); put("jm_hoop_hi", np.percentile(_mix[:, 0], 97.5) * DP_C, 1)
put("jm_hoop_ppos", 100 * (_mix[:, 0] > 0).mean(), 0)
put("jm_hoop_raw", JTHU[4] * DP_C, 1)
put("jm_wz", JB["wz"] * DP_R, 1)
put("jm_wz_lo", (JB["wz"] - 1.96 * JSE["wz"]) * DP_R, 1); put("jm_wz_hi", (JB["wz"] + 1.96 * JSE["wz"]) * DP_R, 1)
put("jm_int", JB["wzxhoop"] * DP_R, 1)
put("jm_int_lo", np.percentile(_mix[:, 1], 2.5) * DP_R, 1); put("jm_int_hi", np.percentile(_mix[:, 1], 97.5) * DP_R, 1)
put("jm_int_ppos", 100 * (_mix[:, 1] > 0).mean(), 0)
put("jm_man", JB["man"] * DP_R, 1); put("jm_man_lo", (JB["man"] - 1.96 * JSE["man"]) * DP_R, 1)
put("jm_man_hi", (JB["man"] + 1.96 * JSE["man"]) * DP_R, 1)


def jm_loco(ja, rj, hz, cl, tau):
    """held-out career-rate forecast (% points) with and without the Hoop G terms."""
    out = {True: np.empty(len(ja)), False: np.empty(len(ja))}
    for c in np.unique(cl):
        mtr = cl != c
        jtr, jte = ja[mtr], ja[~mtr]
        rtr = rj[rj.nfl_id.isin(set(jtr.nfl_id))]
        for hp in (True, False):
            X, ids, kk, nn, _ = jm_design(jtr, rtr, hz[mtr], hp)
            th = glmm(X, ids, kk, nn, len(jtr), tau)[0]
            b = th[:X.shape[1]]
            eta = b[0] + b[2] * jte.log_pick.values + b[3] * jte.edge_share.values + (b[4] * hz[~mtr] if hp else 0)
            out[hp][~mtr] = 100 * expit(eta / np.sqrt(1 + np.pi * tau ** 2 / 8))
    return out


# scored against his own raw career rate on the non-sampled rushes (the model's outcome; the shrunk rate used
# elsewhere would penalize an unshrunk forecast for its scale)
JRAW = 100 * JA.kc.values / JA.nc.values
_jl = jm_loco(JA, RJ, HZ_RC, JA.draft_year.values, JTAU)
put("jm_r2_0", oos(JRAW, _jl[False])[0]); put("jm_r2_1", oos(JRAW, _jl[True])[0])
put("jm_gain", oos(JRAW, _jl[True])[0] - oos(JRAW, _jl[False])[0])
# class-stratified bootstrap of the held-out gain (tau fixed at the full-sample value; duplicated men get new ids)
_jb = np.random.default_rng(2704)
JGB = []
NJB = int(os.environ.get("BDB_NJB", max(200, NBOOT // 2)))
_rjby = {i: RJ[RJ.nfl_id == i] for i in JA.nfl_id}
for _ in range(NJB):
    ix = np.concatenate([_jb.choice(g_, len(g_)) for g_ in grp])
    ja_ = JA.iloc[ix].reset_index(drop=True).copy()
    ja_["nfl_id"] = np.arange(len(ja_))
    rj_ = pd.concat([_rjby[i].assign(nfl_id=j) for j, i in enumerate(JA.nfl_id.values[ix])], ignore_index=True)
    o_ = jm_loco(ja_, rj_, HZ_RC[ix], ja_.draft_year.values, JTAU)
    yb = JRAW[ix]
    JGB.append(oos(yb, o_[True])[0] - oos(yb, o_[False])[0])
put("jm_gain_lo95", np.percentile(JGB, 2.5)); put("jm_gain_hi95", np.percentile(JGB, 97.5))
put("jm_gain_ppos", 100 * np.mean(np.asarray(JGB) > 0), 0); put("n_jm_boot", NJB)

# (S3) Class heterogeneity and the 2026 class: the frozen test's slope in each class (pick + role residual) with its
#      within-class bootstrap se; normal-normal hierarchical model, flat prior on the mean, half-normal prior on the
#      between-class sd (scale = the 2.25-pressure action margin), grid posterior. Predictive distribution of the
#      2026 class's observed slope (between-class spread + a typical class's sampling error).
CS = np.array(HO["hoop_z"]["cls"]) * PS
CSE = np.array([np.std(BX[f"cls{c}"]) for c in classes]) * PS
MU_G = np.linspace(-8, 12, 801); TAU_G = np.linspace(0.01, 10, 400)
_M, _T = np.meshgrid(MU_G, TAU_G, indexing="ij")
_lp = stats.halfnorm.logpdf(_T, scale=MS if "MS" in globals() else MARGIN * PS)
for s_, e_ in zip(CS, CSE):
    _lp = _lp + stats.norm.logpdf(s_, _M, np.sqrt(_T ** 2 + e_ ** 2))
_po = np.exp(_lp - _lp.max()); _po /= _po.sum()
_hr = np.random.default_rng(2705)
_ii = _hr.choice(_po.size, 20000, p=_po.ravel())
_mu_s, _tau_s = _M.ravel()[_ii], _T.ravel()[_ii]
_se_new = float(np.median(CSE))
_new = _hr.normal(_mu_s, np.sqrt(_tau_s ** 2 + _se_new ** 2))
_true_new = _hr.normal(_mu_s, _tau_s)
put("het_mu", np.median(_mu_s), 1); put("het_mu_lo", np.percentile(_mu_s, 2.5), 1)
put("het_mu_hi", np.percentile(_mu_s, 97.5), 1)
put("het_tau", np.median(_tau_s), 1); put("het_tau_hi", np.percentile(_tau_s, 97.5), 1)
put("het_new_lo", np.percentile(_new, 2.5), 1); put("het_new_hi", np.percentile(_new, 97.5), 1)
put("het_new_med", np.median(_new), 1)
put("het_true_pos", 100 * (_true_new > 0).mean(), 0); put("het_obs_pos", 100 * (_new > 0).mean(), 0)
put("het_se_new", _se_new, 1)
for c, s_, e_ in zip(classes, CS, CSE):
    put(f"het_cls_{c}", s_, 1); put(f"het_se_{c}", e_, 1)

# (S4) Day-3 placebo: Hoop G shuffled within class and role among the 77, the out-of-class Hoop G weight refitted, and
#      the two day-3 scores recomputed on the 47 day-3 men (rank correlation gain over pick order, Brier gain).
D3M = (AN.pick >= DAY3).values
_d3 = AN[D3M].reset_index(drop=True).copy()
_d3["hit"] = (_d3.press_eb > _d3.groupby("draft_year").press_eb.transform("median")).astype(float)
_g3 = [np.flatnonzero(_d3.draft_year.values == c) for c in classes]
RHO_PICK = pooled_rho(_d3, "neg_pick", _g3); BRI_PICK = brier(_d3, "neg_pick")


def d3_scores(hz):
    w_ = np.empty(len(AN))
    for c in classes:
        o_ = CLV != c
        w_[CLV == c] = np.polyfit(hz[o_], RES[o_], 1)[0]
    _d3["fc_hoop"] = (AN.fc.values + w_ * hz)[D3M]
    pcl = [stats.spearmanr(_d3.fc_hoop.values[g_], _d3.press_eb.values[g_])[0] - stats.spearmanr(
        _d3.neg_pick.values[g_], _d3.press_eb.values[g_])[0] for g_ in _g3]
    return pooled_rho(_d3, "fc_hoop", _g3) - RHO_PICK, BRI_PICK - brier(_d3, "fc_hoop"), pcl


D3_OBS = d3_scores(AN.hoop_z.values)
NPL3 = int(os.environ.get("BDB_NPL3", 2000))
_pl3 = np.random.default_rng(2706)
D3_NULL = np.empty((NPL3, 2 + len(classes)))
for i_ in range(NPL3):
    ix = np.arange(len(AN))
    for g_ in cr_groups:
        ix[g_] = _pl3.permutation(g_)
    r_ = d3_scores(AN.hoop_z.values[ix])
    D3_NULL[i_] = [r_[0], r_[1], *r_[2]]
put("pl3_rho_obs", D3_OBS[0]); put("pl3_bri_obs", D3_OBS[1], 3)
put("pl3_rho_p", (1 + (D3_NULL[:, 0] >= D3_OBS[0]).sum()) / (NPL3 + 1), 3)
put("pl3_bri_p", (1 + (D3_NULL[:, 1] >= D3_OBS[1]).sum()) / (NPL3 + 1), 3)
for j_, c in enumerate(classes):
    put(f"pl3_rho_{c}", D3_OBS[2][j_])
    put(f"pl3_rho_{c}_p", (1 + (D3_NULL[:, 2 + j_] >= D3_OBS[2][j_]).sum()) / (NPL3 + 1), 2)
put("n_pl3", NPL3)

# ============================================================================= figures (round 27)
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
C_H, C_B, C_N = OI["orange"], OI["blue"], OI["grey"]
C_D = "#4D4D4D"
C_HIT, C_MISS = OI["blue"], OI["red"]
MS = MARGIN * PS
BI = ba.assign(latest_team=ba.latest_team.fillna("FA")).set_index("nfl_id")
SCI = sc.set_index("nfl_id")
DLI = dl.set_index("nfl_id")
_pg = dl.dropna(subset=["g_bend"])
BENDZ = ((_pg.g_bend - _pg.g_bend.mean()) / _pg.g_bend.std()).set_axis(_pg.nfl_id)


def nfl_traces(ids):
    gf_ = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "x", "y"],
                          filters=[("nfl_id", "in", ids)])
    meta = g[(g.sample_kind == "random") & g.nfl_id.isin(ids)][K + ["snap_frame", "play_direction"]]
    gf_ = gf_.merge(meta, on=K).sort_values(K + ["frame"])
    pr_ = GF.set_index(K).press
    ok_ = GF.set_index(K).dropna(subset=["g_bend"]).index
    out_ = {i: [] for i in ids}
    for key, h in gf_.groupby(K, sort=False):
        sf = h.snap_frame.iloc[0]
        w = h[(h.frame >= sf) & (h.frame <= sf + 20)]
        if len(w) != 21 or key not in pr_.index or key not in ok_:
            continue
        sgn = -1 if h.play_direction.iloc[0] == "right" else 1
        up = sgn * (w.x.values - w.x.values[0])
        lat = sgn * (w.y.values - w.y.values[0])
        alat, roll, *_ = lateral_g(w.x.values, w.y.values)
        bend = float(np.nanmax(roll))
        out_[key[2]].append(dict(lat=lat, up=up, alat=alat, k=int(np.nanargmax(roll)), bend=bend,
                                 press=int(pr_.loc[key]), corner=int(bend >= CT), key=key))
    return out_


GO_, NOGO_, HOLD_ = brd[brd.call == "Go"], brd[brd.call == "No-go"], brd[brd.call == "Hold"]
CARDS = pd.concat([GO_, NOGO_]).sort_values("hoop_z", ascending=False)
_ids_all = sorted(set(CARDS.nfl_id.astype(int)) | set(brd[brd.call != "Audition"].nfl_id.astype(int))
                  | set(T7.nfl_id.astype(int)))
NT = nfl_traces(_ids_all)
LEAN_MAX = 0.8
lnorm = plt.Normalize(0, LEAN_MAX)
LCM = plt.get_cmap("viridis")


def lean_line(ax, px, py, alat, lw, z=2, alpha=1.0):
    pts = np.c_[px, py].reshape(-1, 1, 2)
    a_ = pd.Series(alat).rolling(3, center=True, min_periods=1).mean().to_numpy()
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=LCM, norm=lnorm, lw=lw,
                        capstyle="round", zorder=z, alpha=alpha)
    lc.set_array(0.5 * (a_[:-1] + a_[1:]))
    ax.add_collection(lc)
    return lc


# ---- (S5) Table 2 cues, rewritten in plain words with one clip per man. Rules fixed in code (no film watched):
#      hard bend = a sampled rush whose hardest half-second of lean is in the top quarter of all sampled rushes;
#      quick = median NGS get-off below the all-lineman median; wide = half or more of his sampled snaps lined up at EDGE.
#        2+ of 8 rushes hard bends and half his pressures from them -> "Bends and finishes: confirm speed at the top"
#        2+ of 8 rushes hard bends, pressures elsewhere          -> "Bends, then wins with hands: check the counter"
#        quick and wide                                           -> "Quick off the edge: watch his first two steps"
#        quick and inside                                         -> "Quick inside: watch the first step into the gap"
#        neither                                                  -> "Power only: lowest priority"
#      Clip = his first pressured randomly sampled snap (else his hardest-lean sampled snap), as season/week/quarter/clock.
PL27 = GA.merge(g.drop_duplicates(K)[K + ["season", "week", "quarter", "game_clock", "possession_team", "down",
                                         "yards_to_go", "pass_result", "time_to_pressure", "team_coverage_type",
                                         "play_description", "sample_kind"]], on=K, how="left")
_CUES = {"bf": "Bends and finishes: confirm his speed at the top of the arc",
         "bh": "Bends, then wins with hands: check the counter after the dip",
         "qe": "Quick off the edge: watch his first two steps against the set",
         "qi": "Quick inside: watch his first step into the gap",
         "pw": "Power only: lowest priority"}
_cue_k, _clip = [], []
for _, r_ in T7.iterrows():
    aa = PL27[PL27.nfl_id == r_.nfl_id]
    ra = aa[(aa.sample_kind == "random") & aa.corner.notna()]
    pp = aa[(aa.press == 1) & aa.corner.notna()]
    bend_ = len(ra) and ra.corner.mean() >= 0.25
    fromb = len(pp) > 0 and pp.corner.mean() >= 0.5
    quick = float(aa.player_get_off.median()) < _ga_med
    wide = (aa.lined_up_position == "EDGE").mean() >= 0.5
    k_ = "bf" if bend_ and fromb else "bh" if bend_ else ("qe" if wide else "qi") if quick else "pw"
    _cue_k.append(k_)
    rr = aa[aa.sample_kind == "random"]
    cp = rr[rr.press == 1].sort_values(["season", "week"])
    q = cp.iloc[0] if len(cp) else rr.sort_values("g_bend", ascending=False).iloc[0]
    _clip.append(f"{int(q.season)} wk {int(q.week)} Q{int(q.quarter)} {q.game_clock} v {q.possession_team}"
                 + (" (pressure)" if q.press == 1 else " (hardest lean)"))
T7["cue_k"], T7["clip"] = _cue_k, _clip
T7["cue2"] = T7.cue_k.map(_CUES)
put("t7_n_cues", T7.cue_k.nunique())
_t7 = []
for i_, r_ in T7.iterrows():
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
    _t7.append(f"| {i_ + 1}. {r_['name']} ({r_.team}), {r_.yr} {pk} | {fm(r_.hz)} | {fm(r_.ov)} | {r_.go:.2f} | "
               f"{r_.cue2}. Clip: {r_['clip']} |")
TXT["t7_rows"] = "\n".join(_t7)
TXT["t7_cue_best"] = T7.cue2[_ib]; TXT["t7_cue_worst"] = T7.cue2[_iw]

# ---- (S6) one film-style read, from tracking and the play log (no video): film-queue #1 (the added name with the
#      highest Hoop G), on the clip Table 2 lists for him (his first pressured randomly sampled snap). Path from the
#      snap to the moment of pressure.
FR = T7.iloc[0]
_fr = PL27[(PL27.nfl_id == FR.nfl_id) & (PL27.sample_kind == "random") & (PL27.press == 1)].sort_values(["season", "week"])
FQ = _fr.iloc[0]
assert FR["clip"].startswith(f"{int(FQ.season)} wk {int(FQ.week)} Q{int(FQ.quarter)}")
_key = tuple(FQ[k_] for k_ in K)
_ff = pd.read_parquet(PROC / "game_frames_sample.parquet", columns=K + ["frame", "x", "y"],
                      filters=[("nfl_id", "==", int(FR.nfl_id)), ("game_id", "==", int(_key[0]))])
_ff = _ff[_ff.play_id == _key[1]].sort_values("frame")
_m = g[(g.game_id == _key[0]) & (g.play_id == _key[1]) & (g.nfl_id == _key[2])].iloc[0]
_sf = int(_m.snap_frame)
_nf = int(round(float(FQ.time_to_pressure) * 10)) if pd.notna(FQ.time_to_pressure) else 20
_w = _ff[(_ff.frame >= _sf) & (_ff.frame <= _sf + _nf)]
_sg = -1 if _m.play_direction == "right" else 1
_up = _sg * (_w.x.values - _w.x.values[0]); _lat = _sg * (_w.y.values - _w.y.values[0])
_al, _ro, _, _, _v = lateral_g(_w.x.values, _w.y.values)
_kk = int(np.nanargmax(_ro))
FILM = dict(lat=_lat, up=_up, alat=_al, k=_kk)
put("fr_getoff", FQ.player_get_off); put("fr_ttp", float(FQ.time_to_pressure), 1)
put("fr_mph", float(np.nanmax(_v)) * 3600 / 1760, 1); put("fr_depth", float(_up[-1]), 1)
put("fr_lat", abs(float(_lat[-1])), 1); put("fr_bend", float(np.nanmax(_ro)))
put("fr_rad", float(_v[_kk] ** 2 / (_al[_kk] * YD_PER_G)), 1); put("fr_tpk", _kk / 10, 1)
put("fr_hooprad", float(DLI.loc[int(FR.nfl_id)].hoop_rad), 1); put("fr_hoopg", float(DLI.loc[int(FR.nfl_id)].hoop_g))
put("fr_ytg", int(FQ.yards_to_go)); put("fr_down", int(FQ.down))
_ord = {1: "first", 2: "second", 3: "third", 4: "fourth"}
TXT["fr_name"] = FR["name"]; TXT["fr_down_txt"] = _ord[int(FQ.down)]
TXT["fr_when"] = f"{int(FQ.season)} week {int(FQ.week)}, Q{int(FQ.quarter)} {FQ.game_clock} against {FQ.possession_team}"
TXT["fr_align"] = "wide at EDGE" if FQ.lined_up_position == "EDGE" else f"inside ({FQ.lined_up_position})"
TXT["fr_cov"] = ("man" if "MAN" in str(FQ.team_coverage_type) else "zone")
TXT["fr_result"] = {"C": "the pass was completed", "I": "the pass fell incomplete", "S": "it ended in a sack",
                    "IN": "it was intercepted", "R": "the quarterback scrambled"}.get(FQ.pass_result, "the play ended")
TXT["fr_desc"] = re.sub(r"\s+", " ", str(FQ.play_description)).strip()

# ---- Fig 1 (hero): drill vs game for the names the flag adds. Top: three of them by fixed rules (film-queue #1 =
# highest Hoop G, the one furthest above forecast, the one furthest below), hoop lap above and every randomly sampled
# NFL rush below, one yard scale; each NFL panel carries his own hoop turn circle and his median NFL turn circle, so the
# radius gap is drawn. Bottom: all seven, each NFL rush's hardest lean (dots; filled diamond = pressured) against his
# hoop lean (orange bar), with the cohort medians for both venues.
F1 = [T7.iloc[0], T7.iloc[_ib], T7.iloc[_iw]]
assert len({r_["name"] for r_ in F1}) == 3
HX = 6.5
H1 = 11.6
fig = plt.figure(figsize=(FIGW, H1))
YI = lambda in_: 1 - in_ / H1
PW = 0.30
_gtr = GT.set_index("nfl_id")
for j, r_ in enumerate(F1):
    pid = int(r_.nfl_id)
    x0_ = 0.02 + j * 0.33
    tr_ = traces[BI.loc[pid].hoop_eid]
    x_, y_ = tr_["_x"] - np.mean(tr_["_x"]), tr_["_y"] - np.mean(tr_["_y"])
    if np.ptp(y_) > np.ptp(x_):
        x_, y_ = y_, -x_
    axT = fig.add_axes([x0_, YI(3.75), PW, 1.45 / H1])
    lean_line(axT, x_, y_, tr_["_alat"], 3.6)
    axT.plot(x_[tr_["_k"]], y_[tr_["_k"]], "o", ms=12, mfc="none", mec="black", mew=2.4, zorder=6)
    axT.set_xlim(-HX, HX); axT.set_ylim(-2.6, 2.6); axT.set_aspect("equal"); axT.axis("off")
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
    fig.text(x0_, YI(1.62), f"{r_['name']}", fontsize=14, fontweight="bold", va="bottom")
    fig.text(x0_, YI(1.67), f"{r_.team} · {r_.yr} {pk} · Hoop G {fm(r_.hz)} sd\n{fm(r_.ov)} pressures a season vs forecast",
             fontsize=11, color="#333333", va="top", linespacing=1.25)
    axB = fig.add_axes([x0_, YI(7.1), PW, 3.3 / H1])
    rs = NT[pid]
    for t in rs:
        if t["press"]:
            axB.plot(t["lat"], t["up"], color="black", lw=5.2, zorder=1, solid_capstyle="round")
        lean_line(axB, t["lat"], t["up"], t["alat"], 2.2, alpha=0.95)
        if t["press"]:
            axB.plot(t["lat"][-1], t["up"][-1], "D", ms=9, color="black", mec="white", mew=1.0, zorder=5)
    axB.plot(0, 0, "s", ms=8, color="#333333", zorder=5)
    hr_, nr_ = float(DLI.loc[pid].hoop_rad), float(_gtr.loc[[pid]].t_rad.median())
    for yb_, rr_, ls_, lab_ in [(-0.9, hr_, (0, (2, 1.2)), f"his hoop turn radius, {hr_:.1f} yd"),
                                (-2.6, nr_, "-", f"his median NFL turn radius, {nr_:.1f} yd")]:
        axB.plot([-HX + 0.2, -HX + 0.2 + min(rr_, 2 * HX - 0.4)], [yb_ - 0.75, yb_ - 0.75], color="black", lw=4,
                 ls=ls_, solid_capstyle="butt")
        axB.text(-HX + 0.2, yb_, lab_, fontsize=10.5, va="bottom", ha="left")
    if j == 0:
        axB.plot(FILM["lat"], FILM["up"], color="black", lw=2.4, ls=(0, (1, 1)), zorder=6)
        axB.plot(FILM["lat"][-1], FILM["up"][-1], "D", ms=10, color="black", mec="white", zorder=7)
        axB.annotate(f"Table 3 clip:\npressure at {NUM['fr_ttp']} s", xy=(FILM["lat"][-1], FILM["up"][-1]),
                     xytext=(FILM["lat"][-1] + (2.2 if FILM["lat"][-1] < 2 else -5.5), FILM["up"][-1] + 1.0),
                     fontsize=10.5, arrowprops=dict(arrowstyle="-", color="black", lw=1))
    axB.set_xlim(-HX, HX); axB.set_ylim(-3.6, 9.0); axB.set_aspect("equal"); axB.axis("off")
    if j == 0:
        axT.plot([-HX + 0.2, -HX + 2.2], [-2.3, -2.3], color="black", lw=4)
        axT.text(-HX + 1.2, -2.1, "2 yd", ha="center", va="bottom", fontsize=11, fontweight="bold")
        axB.annotate("upfield", xy=(-5.4, 8.8), xytext=(-5.4, 6.4), fontsize=11, ha="center",
                     arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    put(f"f1_{j}_np", int(sum(t_["press"] for t_ in rs))); put(f"f1_{j}_n", len(rs))
    put(f"f1_{j}_hrad", hr_, 1); put(f"f1_{j}_nrad", nr_, 1)
    TXT[f"f1_{j}"] = r_["name"]
cax = fig.add_axes([0.70, YI(0.62), 0.27, 0.12 / H1])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=lnorm, cmap=LCM), cax=cax, orientation="horizontal")
cb.set_label("sideways lean, g (line colour)", fontsize=11); cax.xaxis.set_label_position("top")
cb.ax.tick_params(labelsize=10)
fig.text(0.02, YI(0.08), "Combine hoop lap (top) against his randomly sampled NFL rushes (below)", fontsize=16,
         fontweight="bold", va="top")
fig.text(0.02, YI(0.45), "Same yard scale everywhere. ○ hardest half-second of the lap; ■ snap spot;\nthick black outline and ◆ = "
         "pressured rush (first 2 s). The NFL arc is far wider\nthan the hoop circle, so men are compared within "
         "their class, never in raw g.", fontsize=11, va="top", color="#333333", linespacing=1.3)
# strip: all seven added names
sx = fig.add_axes([0.29, 0.05, 0.68, 0.25])
for i_, r_ in T7.iterrows():
    rs = NT[int(r_.nfl_id)]
    jit = np.linspace(-0.18, 0.18, max(len(rs), 2))[: len(rs)]
    for t, jj in zip(rs, jit):
        if t["press"]:
            sx.plot(t["bend"], i_ + jj, "D", ms=8, color="black", zorder=4)
        else:
            sx.plot(t["bend"], i_ + jj, "o", ms=6.5, mfc="white", mec="#555555", mew=1.2, zorder=3)
    hg_ = float(DLI.loc[int(r_.nfl_id)].hoop_g)
    sx.plot([hg_, hg_], [i_ - 0.32, i_ + 0.32], color=C_H, lw=5, solid_capstyle="butt", zorder=5)
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
    sx.text(-0.02, i_, f"{i_ + 1}. {r_['name']} ({r_.yr} {pk})", transform=sx.get_yaxis_transform(), ha="right",
            va="center", fontsize=11.5)
_hm, _nm = float(hoop_att.bend_g.median()), float(PG.g_bend.median())
for v_, lab_ in [(_nm, f"NFL rush median {_nm:.2f} g"), (_hm, f"hoop median {_hm:.2f} g")]:
    sx.axvline(v_, color="#333333", lw=1.2, ls=(0, (3, 2)), zorder=1)
    sx.text(v_ + 0.01, -0.95, lab_, ha="left", va="center", fontsize=11)
sx.set_ylim(len(T7) - 0.5, -1.1); sx.set_yticks([]); sx.set_xlim(0, 1.0)
sx.spines["left"].set_visible(False)
sx.set_xlabel("Hardest half-second of sideways lean, g")
sx.set_title("All 7 added names, in film order: each NFL rush (○, ◆ pressured) vs his hoop lap (orange bar)",
             loc="left", fontsize=12.5, fontweight="bold", x=-0.40)
save(fig, "fig1_hero.png")
put("hoop_mph_median", hoop_att.speed_at_bend.median() * 3600 / 1760, 1)
put("nfl_bend_median", float(PG.g_bend.median()))
put("n_short_career", int((dl.rush_snaps < RATE_MIN).sum()))

# ---- Fig 2: A. every pre-listed drill measure and the two no-choosing composites against the pick + role residual
# (95%, chance line = best of six on shuffled data). B. the frozen test and its stress tests, weak rows marked.
NCD = {r_[0]: r_ for r_ in NC}
_f = lambda k_: float(NUM[k_])
ROWA = [("All six, shrunk (ridge): no choosing", _f("r6_slope"), _f("r6_slope_lo95"), _f("r6_slope_hi95"), "comp"),
        ("All six, equal weight", _f("ew_slope"), _f("ew_slope_lo95"), _f("ew_slope_hi95"), "comp")]
for j in [JH] + [j_ for j_ in range(len(FAMR)) if j_ != JH]:
    f = FAMR[j]
    l95, h95 = np.percentile(f["bs"], [2.5, 97.5]) * PS
    ROWA.append((f["lab"].replace("Hoop: Hoop G (sideways lean)", "Hoop G"), f["est"] * PS, l95, h95,
                 "hoop" if j == JH else "other"))
GRPB = [
    ("The frozen 2026 test", [("Hoop G vs pick + role forecast", _f("bs_pick"), _f("bs_pick_lo95"), _f("bs_pick_hi95"), len(AN), "hoop")]),
    ("Holds when", [("holding wave-drill lean", _f("jw_hoop"), _f("jw_hoop_lo95"), _f("jw_hoop_hi95"), len(JW), "hoop"),
                    (f"{len(TEAMS)} NFL clubs resampled", _f("bs_pick"), _f("cbs_lo95"), _f("cbs_hi95"), len(AN), "hoop"),
                    ("+ stance, snap load, club rush", _f("ct_slope"), _f("ct_slope_lo95"), _f("ct_slope_hi95"), len(AN), "hoop")]),
    ("Weak rows", [("measure re-chosen in each fold", _f("ns_slope"), _f("ns_slope_lo95"), _f("ns_slope_hi95"), len(AN), "weak"),
                   ("2025 class alone", _f("pro_s25"), _f("pro_s25_lo95"), _f("pro_s25_hi95"), len(TE_), "weak"),
                   ("2026 class, predicted", _f("het_new_med"), _f("het_new_lo"), _f("het_new_hi"), NCL, "weak")]),
    ("Should be flat", [("DL rush drill: burst", NCD["burst"][2] * PS, NCD["burst"][3] * PS, NCD["burst"][4] * PS, NCD["burst"][6], "ctrl"),
                        ("OL pull-drill lean*", NCD["ol"][2] * PS, NCD["ol"][3] * PS, NCD["ol"][4] * PS, NCD["ol"][6], "ctrl")])]
RC2 = {"hoop": C_H, "other": C_D, "ctrl": C_D, "comp": OI["sky"], "weak": OI["sky"]}
nA = len(ROWA); nB = sum(len(r_) for _, r_ in GRPB) + len(GRPB)
HF2 = 0.42 * (nA + nB) + 3.5
fig = plt.figure(figsize=(FIGW, HF2))
hA, hB = 0.42 * nA / HF2, 0.42 * nB / HF2
axA = fig.add_axes([0.40, 1 - 1.55 / HF2 - hA, 0.33, hA])
axB = fig.add_axes([0.40, 1.15 / HF2, 0.33, hB])


def _bar(ax, yy, lab_, e_, l_, h_, kind_, n_=None, bold=False):
    c_ = RC2[kind_]
    ax.plot([l_, h_], [yy, yy], color=c_, lw=7, solid_capstyle="butt", alpha=0.95)
    if kind_ == "weak":
        ax.plot([l_, h_], [yy, yy], color="white", lw=7, ls=(0, (1, 1.2)), solid_capstyle="butt", alpha=0.8)
    ax.plot(e_, yy, "o", color="white", mec="black", mew=2, ms=9, zorder=5)
    ax.text(-1.19, yy, lab_, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=11.5,
            fontweight="bold" if bold else "normal")
    ax.text(1.04, yy, f"{fm(e_)}  [{fm(l_)}, {fm(h_)}]", transform=ax.get_yaxis_transform(), ha="left",
            va="center", fontsize=11.5, fontweight="bold" if bold else "normal")
    if n_ is not None:
        ax.text(1.62, yy, f"{n_}", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=11.5)


for yy, (lab_, e_, l_, h_, k_) in enumerate(ROWA):
    _bar(axA, yy, lab_, e_, l_, h_, k_, len(AN), bold=yy == 0)
for sx_ in (-FW95, FW95):
    axA.axvline(sx_, color="#333333", lw=1.2, ls=(0, (2, 2)))
axA.set_ylim(nA - 0.4, -0.7)
axA.set_title(f"A. Every drill measure, held out (dashed: best of six on shuffled data, ±{FW95:.1f})",
              loc="left", fontsize=12.5, fontweight="bold", x=-1.19)
yy = 0
for head_, rows_ in GRPB:
    axB.text(-1.19, yy, head_, transform=axB.get_yaxis_transform(), ha="left", va="center", fontsize=12,
             fontweight="bold", color="#222222")
    yy += 1
    for lab_, e_, l_, h_, n_, k_ in rows_:
        _bar(axB, yy, "   " + lab_, e_, l_, h_, k_, n_, bold=yy == 1)
        yy += 1
axB.set_ylim(nB - 0.4, -0.7)
axB.set_title("B. Hoop G: the frozen test and its stress tests", loc="left", fontsize=12.5, fontweight="bold", x=-1.19)
axB.axvspan(-MS, MS, color=C_N, alpha=0.2, lw=0)
for ax in (axA, axB):
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlim(-5, 8); ax.set_xticks([-4, -2, 0, 2, 4, 6, 8])
    ax.text(1.04, -0.7, "per sd [95%]", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=11, color="#333")
    ax.text(1.62, -0.7, "n", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=11, color="#333")
axB.set_xlabel("Extra pressures per 450-rush season per sd, beyond the held-out pick + role forecast")
hd2 = [Patch(color=C_H, label="Hoop G"), Patch(color=OI["sky"], label="no single measure chosen / weak row"),
       Patch(color=C_D, label="other measure or control"), Patch(color=C_N, alpha=0.4, label=f"grey band: under ±{MS:.2f}")]
fig.legend(handles=hd2, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=4, frameon=False, fontsize=11)
fig.text(0.01, 0.997, "The hoop carries the drill signal, but every row is noisy", fontsize=15, fontweight="bold", va="top")
fig.text(0.01, 1 - 0.5 / HF2, "○ estimate, bar 95% interval; every row scored on classes the model never saw. "
         "*OL row: pressures allowed prevented per 450 pass blocks.", fontsize=11, va="top", color="#333333")
save(fig, "fig2_tests.png")

# ---- 90% film-queue intervals kept for board.csv / Table A1 bookkeeping (no figure)
R3 = big.sort_values("hoop_z", ascending=False).reset_index(drop=True)
LOHI = {}
for _, r_ in R3.iterrows():
    ov, sd_ = r_.over * PS, r_.over_sd * PS
    LOHI[int(r_.nfl_id)] = (ov - Z90 * sd_, ov + Z90 * sd_)

# ---- Fig 3 (mechanism): A. pressure rate by how hard he leaned on that rush relative to his own rushes, Hoop G top
# third vs the rest, with the rush count in every quarter. B. the rush-level terms of the joint model.
fig = plt.figure(figsize=(FIGW, 6.8))
ax = fig.add_axes([0.075, 0.21, 0.42, 0.52]); bx = fig.add_axes([0.73, 0.21, 0.24, 0.52])
qx = np.arange(1, 5)
for ht_, c_, lab_ in [(True, C_H, "Hoop G top third"), (False, C_D, "other two thirds")]:
    e_, l_, h_, n_, nr_ = F4[ht_]
    off = -0.07 if ht_ else 0.07
    ax.fill_between(qx + off, l_, h_, color=c_, alpha=0.16, lw=0)
    ax.plot(qx + off, e_, "-o" if ht_ else "--s", color=c_, lw=3, ms=9, mfc="white", mew=2.5,
            label=f"{lab_}: {n_} men")
    nq = RU[RU.ht == ht_].groupby("q").size().reindex(range(4), fill_value=0).values
    for q_, nn_ in enumerate(nq):
        put(f"f4q_{'top' if ht_ else 'rest'}_{q_}", int(nn_))
_nqt = [int(NUM[f"f4q_top_{q_}"]) for q_ in range(4)]; _nqr = [int(NUM[f"f4q_rest_{q_}"]) for q_ in range(4)]
ax.set_xticks(qx)
ax.set_xticklabels([f"{t_}\nn = {a_} / {b_}" for t_, a_, b_ in
                    zip(["his softest", "2nd", "3rd", "his hardest"], _nqt, _nqr)], fontsize=10.5)
ax.set_xlim(0.6, 4.4); ax.legend(loc="upper left", frameon=False, fontsize=11.5)
ax.set_ylabel("Pressured rushes, %"); ax.set_ylim(0, None)
ax.set_xlabel("How hard he leaned on that rush, quarter of his own rushes (n rushes: top third / rest)")
ax.set_title("A. Pressure rate by own-rush lean (95% band)", loc="left", fontsize=13, fontweight="bold")
MR = [("Own-rush lean\n(+1 of his own sd)", "jm_wz", C_D), ("Own lean × Hoop G", "jm_int", C_H),
      ("Man coverage\n(vs zone)", "jm_man", C_D)]
for i_, (lab_, k_, c_) in enumerate(MR):
    bx.plot([_f(k_ + "_lo"), _f(k_ + "_hi")], [i_, i_], color=c_, lw=7, solid_capstyle="butt")
    bx.plot(_f(k_), i_, "o", color="white", mec="black", mew=2, ms=10, zorder=5)
    bx.text(-0.04, i_, lab_, transform=bx.get_yaxis_transform(), ha="right", va="center", fontsize=11.5)
bx.axvline(0, color="black", lw=0.9); bx.set_yticks([]); bx.set_ylim(len(MR) - 0.4, -0.6)
bx.spines["left"].set_visible(False)
bx.set_xlabel("Points of pressure % on a rush\n(joint model, 95%)")
bx.set_title(f"B. Joint model, {NUM['jm_n_rush']} rushes", loc="right", fontsize=13, fontweight="bold")
fig.text(0.60, 0.015, f"Hoop G alone on sampled rushes: {NUM['mm_m2_hoopz']} [{NUM['mm_m2_hoopz_lo']}, "
        f"{NUM['mm_m2_hoopz_hi']}] points, null", fontsize=10.5, ha="left", color="#333")
fig.text(0.01, 0.995, "High-Hoop-G men do not bend more often on Sunday; their hardest-lean rushes\n"
         "end in pressure more often", fontsize=15, fontweight="bold", va="top")
fig.text(0.01, 0.875, f"Randomly sampled NFL rushes of the {NUM['mm_n_men']} tested rushers (median "
         f"{NUM['mm_rush_med']} each), first 2 s. Lean quarters are within each man's own rushes.",
         fontsize=11.5, va="top", color="#333333")
save(fig, "fig3_mechanism.png")

# ---- Fig 4 (day 3): A rank correlation, B Brier, each with its within-class shuffle p; C the film queue with the cue.
fig = plt.figure(figsize=(FIGW, 9.2))
gA = fig.add_axes([0.25, 0.63, 0.29, 0.21]); gB = fig.add_axes([0.67, 0.63, 0.29, 0.21])
ROW3 = [("neg_pick", "pick", "Pick order"), ("fc", "fc", "Pick + role forecast"), ("fc_hoop", "fch", "Forecast + Hoop G")]
for i_, (col_, t_, lab_) in enumerate(ROW3):
    c_ = C_H if t_ == "fch" else C_D
    gA.plot([_f(f"rho95_{t_}_lo"), _f(f"rho95_{t_}_hi")], [i_, i_], color=c_, lw=7, solid_capstyle="butt")
    gA.plot(_f(f"dr_{t_}"), i_, "o", color="white", mec="black", mew=2, ms=10, zorder=5)
    gA.text(-0.62, i_, lab_, ha="right", va="center", fontsize=12, fontweight="bold" if t_ == "fch" else "normal")
    gB.plot([_f(f"bri_{t_}_lo"), _f(f"bri_{t_}_hi")], [i_, i_], color=c_, lw=7, solid_capstyle="butt")
    gB.plot(_f(f"bri_{t_}"), i_, "o", color="white", mec="black", mew=2, ms=10, zorder=5)
    gB.text(_f(f"bri_{t_}_hi") + 0.005, i_, f"{NUM[f'bri_{t_}']}", va="center", fontsize=11.5)
gA.axvline(0, color="black", lw=0.9); gA.set_xlim(-0.6, 0.7); gA.set_yticks([]); gA.set_ylim(2.6, -0.6)
gA.set_xlabel("Rank correlation with pressure rate\n(within class, higher = better)")
gA.set_title(f"A. Ordering {len(D3)} day-3 rushers (shuffle p = {NUM['pl3_rho_p']})", loc="left", fontsize=12.5,
             fontweight="bold", x=-0.6)
gB.axvline(0.25, color="black", lw=0.9, ls=(0, (3, 2)))
gB.text(0.252, 2.55, "coin flip", fontsize=11, va="bottom")
gB.set_xlim(0.17, 0.42); gB.set_yticks([]); gB.set_ylim(2.6, -0.6)
gB.set_xlabel("Brier score, beats class day-3 median\n(lower = better)")
gB.set_title(f"B. Calling hits (shuffle p = {NUM['pl3_bri_p']})", loc="left", fontsize=12.5, fontweight="bold")
for a_ in (gA, gB):
    a_.spines["left"].set_visible(False)
gC = fig.add_axes([0.30, 0.13, 0.30, 0.37])
for i_, r_ in T7.iterrows():
    gC.plot([r_.lo, r_.hi], [i_, i_], color="#8C8C8C", lw=6, solid_capstyle="butt")
    gC.plot(r_.ov, i_, "o", color="black" if r_.hit else "white", mec="black", mew=2.2, ms=10, zorder=5)
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
    gC.text(-0.015, i_, f"{i_ + 1}. {r_['name']} ({r_.yr} {pk})", transform=gC.get_yaxis_transform(), ha="right",
            va="center", fontsize=11.5, fontweight="bold")
    gC.text(1.03, i_, r_.cue2.split(":")[0], transform=gC.get_yaxis_transform(), ha="left", va="center",
            fontsize=11.5, color="#222222")
gC.axvline(0, color="black", lw=0.9)
gC.set_yticks([]); gC.set_ylim(len(T7) - 0.4, -0.6); gC.spines["left"].set_visible(False)
gC.set_xlabel("Pressures a season vs his held-out forecast (95%)")
gC.text(1.03, -0.95, "Tape cue (Table 3)", transform=gC.get_yaxis_transform(), ha="left", fontsize=11.5,
        fontweight="bold", va="bottom")
gC.set_title(f"C. The {len(T7)} names Hoop G adds to pick order's top third, in film order", loc="left",
             fontsize=12.5, fontweight="bold", x=-0.95)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color="black", ms=10, label="beat his class's day-3 median"),
                    Line2D([], [], marker="o", ls="", mfc="white", mec="black", mew=2, ms=10, label="did not")],
           loc="lower left", bbox_to_anchor=(0.30, 0.0), ncol=2, frameon=False, fontsize=11.5)
fig.text(0.01, 0.995, "On day 3, forecast + Hoop G orders rushers ahead of pick order; the intervals overlap", fontsize=15,
         fontweight="bold", va="top")
fig.text(0.01, 0.948, f"{len(D3)} day-3 and undrafted rushers, each scored on a class the model never saw. Bars: 95% "
         "class-stratified bootstrap.\nShuffle p: Hoop G permuted within class and role, everything refitted.", fontsize=11, va="top", color="#333333")
save(fig, "fig4_day3.png")

# ---- Fig A1 (appendix): four shuffle tests, each rerunning the whole pipeline on Hoop G (or all six) permuted
# within class and role.
fig, axs = plt.subplots(2, 2, figsize=(FIGW, 7.6))
fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.09, hspace=0.6, wspace=0.25)
PLA = [(axs[0, 0], NULL_CR, OBS_H, f"A. Hoop G slope: p = {NUM['pl_cr_p']}", "pressures per 450 rushes per sd", "{:+.1f}"),
       (axs[0, 1], R6_NULL, R6_GAIN, f"B. All six, ridge, R² gain: p = {NUM['r6_p']}", "held-out R² gain", "{:+.2f}"),
       (axs[1, 0], D3_NULL[:, 0], D3_OBS[0], f"C. Day-3 rank gain over pick: p = {NUM['pl3_rho_p']}",
        "rank correlation gain", "{:+.2f}"),
       (axs[1, 1], D3_NULL[:, 1], D3_OBS[1], f"D. Day-3 Brier gain over pick: p = {NUM['pl3_bri_p']}",
        "Brier score gain", "{:+.3f}")]
for a_, nl_, ob_, t_, xl_, f_ in PLA:
    a_.hist(nl_, bins=40, color=C_N, alpha=0.75)
    a_.axvline(ob_, color=C_H, lw=3.5)
    a_.text(ob_, a_.get_ylim()[1] * 0.95, "real " + f_.format(ob_) + " ", color="#8A5A00", fontsize=11.5,
            fontweight="bold", va="top", ha="right", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    a_.set_title(t_, loc="left", fontsize=12.5, fontweight="bold"); a_.set_xlabel(xl_); a_.set_ylabel("shuffles")
fig.text(0.01, 0.99, "Appendix. Shuffle tests: Hoop G (or all six measures) permuted within class and role,\n"
         "whole pipeline rerun each time (grey); real value in orange", fontsize=14, fontweight="bold", va="top")
save(fig, "figA1_screen.png")

# film list: four names chosen by fixed rules, not by eye: the flagged hit with the highest Hoop G, the hit furthest
# above forecast, the miss with the most corner turns on sampled rushes, the miss furthest below forecast.
# For each, one line on what tape would confirm or kill the flag (hand-written, no numbers), and clip locations.
CARDS["bendz"] = [BENDZ.loc[int(p_)] for p_ in CARDS.nfl_id]
CARDS["nc"] = [sum(t["corner"] for t in NT[int(p_)]) for p_ in CARDS.nfl_id]
CARDS["over_s"] = CARDS.over * PS
_go, _ng = CARDS[CARDS.call == "Go"], CARDS[CARDS.call == "No-go"]
FOUR = [("hit", _go.sort_values("hoop_z").iloc[-1]), ("hit", _go.sort_values("over_s").iloc[-1]),
        ("miss", _ng.sort_values(["nc", "hoop_z"]).iloc[-1]), ("miss", _ng.sort_values("over_s").iloc[0])]
assert len({r_.display_name for _, r_ in FOUR}) == 4
_rows = []
for i_, (tag, r_) in enumerate(FOUR):
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
    lo_, hi_ = LOHI[int(r_.nfl_id)]
    put(f"four{i_}_over", r_.over_s, 1); put(f"four{i_}_bendz", r_.bendz, 1)
    put(f"four{i_}_lo", lo_, 1); put(f"four{i_}_hi", hi_, 1)
TXT["four_rows"] = "\n".join(_rows)
TXT["four_names"] = [r_.display_name for _, r_ in FOUR]

# ---- play-log check (not film): for the top-Hoop-G hit and the miss with the most corner turns, read every randomly
# sampled snap: alignment, whether the 2-s rush turned the corner, NGS get-off, pressure, and the play-by-play result.
PLOG = GR.merge(g.drop_duplicates(K)[K + ["lined_up_position", "pass_result", "player_get_off", "time_to_pressure"]],
                on=K, how="left")
_gmed = float(PLOG.player_get_off.median())
put("plog_getoff_all", _gmed)
_hit = CARDS[CARDS.call == "Go"].sort_values("hoop_z").iloc[-1]
_mc = CARDS[CARDS.call == "No-go"].assign(nc=lambda d_: [sum(t["corner"] for t in NT[int(p_)]) for p_ in d_.nfl_id])
_mis = _mc.sort_values(["nc", "hoop_z"]).iloc[-1]
for tag, r_ in [("hit", _hit), ("miss", _mis)]:
    d_ = PLOG[PLOG.nfl_id == r_.nfl_id]
    c_ = d_[d_.corner == 1]
    TXT[f"plog_{tag}"] = r_.display_name
    put(f"plog_{tag}_n", len(d_)); put(f"plog_{tag}_edge", int((d_.lined_up_position == "EDGE").sum()))
    put(f"plog_{tag}_n2", int(d_.corner.notna().sum())); put(f"plog_{tag}_nc", int(c_.shape[0]))
    put(f"plog_{tag}_np", int(d_.press.sum())); put(f"plog_{tag}_npc", int(c_.press.sum()))
    put(f"plog_{tag}_go_c", c_.player_get_off.median())
    put(f"plog_{tag}_cmp", int((d_.pass_result == "C").sum()))
    put(f"plog_{tag}_cmp_c", int((c_.pass_result == "C").sum()))
    put(f"plog_{tag}_exp", float(r_.press_eb) / 100 * len(d_), 1)
    put(f"plog_{tag}_rate", r_.press_eb, 1)
    _pp = d_[d_.press == 1]
    put(f"plog_{tag}_ttp", float(_pp.time_to_pressure.min()) if len(_pp) else 0.0, 1)
put("n_cards", len(CARDS)); put("n_open", len(_go)); put("n_miss", len(_ng))
# clip locations for the four names (graft from the WR candidate): up to two corner-turn snaps each, pressured
# snaps first, so a scout can pull the tape. Season, week, quarter, clock, offense.
PLOC = PLOG.merge(g.drop_duplicates(K)[K + ["season", "week", "quarter", "game_clock", "possession_team"]], on=K)
RES_TXT = {"C": "completion", "I": "incompletion", "S": "sack", "IN": "interception", "R": "scramble"}
_clips = []
for _, r_ in FOUR:
    c_ = PLOC[(PLOC.nfl_id == r_.nfl_id) & (PLOC.corner == 1)].sort_values(["press", "season", "week"],
                                                                             ascending=[False, True, True])
    refs = [f"{int(q.season)} wk {int(q.week)} Q{int(q.quarter)} {q.game_clock} v {q.possession_team} "
            f"({'pressure' if q.press else RES_TXT.get(q.pass_result, 'run')})" for q in c_.head(2).itertuples()]
    _clips.append(f"{r_.display_name}: " + "; ".join(refs))
TXT["clips"] = ". ".join(_clips)

# ---- round 25: play-log verdicts for the four names, by rules fixed before reading the rows.
# Evidence: (1) corner-turn rate on his RANDOM sampled snaps (unbiased; league = 25% by construction);
# (2) where his pressures came from on ALL sampled snaps (random + top-EPA, which are pressure-rich): share of pressured
# snaps with a full 2-s path whose hardest half-second was a corner turn; (3) median NGS get-off vs all sampled linemen.
# Rules: bends on Sunday = corner-turn rate >= 25%; wins from bend = at least half of pressures on corner turns.
#   beat forecast + bends + wins from bend   -> "supports the flag"
#   beat forecast, otherwise                 -> "beat it, but not by bending"
#   fell below + bends                       -> "bends, no finish"
#   fell below + does not bend               -> "against the flag"
GA = GF.copy()
GA["corner"] = (GA.g_bend >= CT).astype(float).where(GA.g_bend.notna())
GA = GA.merge(g.drop_duplicates(K)[K + ["player_get_off", "lined_up_position"]], on=K, how="left")
_vrows = []
VGO = {}
for i_, (tag, r_) in enumerate(FOUR):
    ra = GR[GR.nfl_id == r_.nfl_id].dropna(subset=["corner"])
    aa = GA[GA.nfl_id == r_.nfl_id]
    pp = aa[(aa.press == 1) & aa.corner.notna()]
    cr_ = 100 * ra.corner.mean()
    pc_ = int(pp.corner.sum()); pn_ = len(pp)
    go_ = float(aa.player_get_off.median())
    bends = cr_ >= 25
    from_bend = pn_ > 0 and pc_ / pn_ >= 0.5
    if tag == "hit":
        vd = "supports the flag" if (bends and from_bend) else "beat it, but not by bending"
    else:
        vd = "bends, no finish" if bends else "against the flag"
    put(f"v{i_}_ct", int(ra.corner.sum())); put(f"v{i_}_ctn", len(ra)); put(f"v{i_}_ctpct", cr_, 0)
    put(f"v{i_}_pc", pc_); put(f"v{i_}_pn", pn_); put(f"v{i_}_npl", len(aa)); put(f"v{i_}_go", go_)
    put(f"v{i_}_edge", 100 * (aa.lined_up_position == "EDGE").mean(), 0)
    pk = "UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}"
    _vrows.append(f"| {r_.display_name} ({r_.latest_team}), {int(r_.draft_year)} {pk} | {int(r_.cls_rank)} of "
                  f"{int(r_.cls_n)} | {fm(r_.over_s)} [{fm(LOHI[int(r_.nfl_id)][0])}, {fm(LOHI[int(r_.nfl_id)][1])}] | {int(ra.corner.sum())} of {len(ra)} | {pc_} of {pn_} | "
                  f"{go_:.2f} s | **{vd}** |")
    TXT[f"v{i_}_verdict"] = vd
    TXT[f"v{i_}_name"] = r_.display_name
    VGO[i_] = go_
TXT["verdict_rows"] = "\n".join(_vrows)
put("getoff_all_med", float(GA.player_get_off.median()))
# the prose under Table 2 is written for these verdicts; fail loudly if the data say otherwise
assert [TXT[f"v{i_}_verdict"] for i_ in range(4)] == ["supports the flag", "beat it, but not by bending",
                                                       "bends, no finish", "against the flag"]
assert VGO[2] < float(NUM["getoff_all_med"]) and VGO[3] == max(VGO.values())
put("chain_prod", float(NUM["tr_hoop_g_bend"]) * float(NUM["gp_g_bend"]))
put("n_verdict_support", int(sum("supports" in TXT[f"v{i_}_verdict"] for i_ in range(4))))
put("cards_min_sampled", int(min(len(NT[int(p_)]) for p_ in CARDS.nfl_id)))
put("cards_max_sampled", int(max(len(NT[int(p_)]) for p_ in CARDS.nfl_id)))


# ============================================================================= multiplicity and independent check
TESTS = (["family: " + f[0] for f in FAM]
         + ["chain: burst->get-off", "chain: burst->NFL burst", "chain: hoop->NFL bend", "chain: hoop->get-off",
            "bench: 3-cone->NFL bend", "bench: split->get-off", "bench: 3-cone->get-off",
            "trait->pressure: get-off", "trait->pressure: burst", "trait->pressure: bend",
            "R2 gain: full + hoop", "R2 gain: full + burst", "burst equivalence",
            "day3 rho: fc+hoop vs fc", "day3 rho: fc+hoop vs pick", "day3 rho: hoop vs pick",
            "day3 top-third gap", "day3 flag Fisher"])
put("n_tests", len(TESTS))


def pb(x, y_):
    ks = [k for k in PB if k[:2] == (x, y_)]
    return PB[max(ks, key=lambda k: k[2] if len(k) > 2 else 0)]


PV = ([HO[f[0]]["p2"] for f in FAM]
      + [pb("burst_z", "quick"), pb("burst_z", "g_d1"), pb("hoop_z", "g_bend"), pb("hoop_z", "quick"),
         pb("three_cone_fast", "g_bend"), pb("split_fast", "quick"), pb("three_cone_fast", "quick"),
         pb("quick", "press_eb"), pb("g_d1", "press_eb"), pb("g_bend", "press_eb"),
         PB[("dr2", "base_h")], PB[("dr2", "burst_h")], float(NUM["ho_burst_tost"]),
         PB[("day3", "fch_fc")], PB[("day3", "fch_pick")], PB[("day3", "hoop_pick")],
         PB[("day3", "hitgap")], float(NUM["fl_p_fisher"])])
TESTS = TESTS + ["R2 gain: pick + hoop", "R2 gain: pick + role + hoop", "R2 gain: nested + hoop",
                 "hoop slope: pick residual", "hoop slope: pick + role residual", "hoop slope: nested residual",
                 "hoop slope: EDGE only", "hoop slope: interior only", "hoop slope: mean of two turns",
                 "chain: hoop radius->NFL radius", "film: corner-turn rate by Hoop G third",
                 "film: pressure on corner-turn rushes"]
PV = PV + [PB[("dr2", "pick1_h")], PB[("dr2", "pick_h")], PB[("dr2", "nest_h")], PB[("slope", "bs_pick1")],
           PB[("slope", "bs_pick")], PB[("slope", "bs_nest")], PB[("slope", "role_EDGE")],
           PB[("slope", "role_interior")], PB[("slope", "mt_hoop")], pb("hoop_rad_tight", "nfl_rad_tight"),
           PB[("film", "ct_gap")], PB[("film", "ct_press")]]
TESTS = TESTS + ["nested best-of-six R2 gain", "mech: own-lean x Hoop G", "mech: hoop-tight x Hoop G",
                 "day3 Brier: fc+hoop vs pick", "slope with stance/load/club controls"]
_dbp = np.asarray(_dbr)
_ctb = np.asarray(_cb_)
PV = PV + [float(NUM["ns_p"]), float(NUM["mm_m2_wzxhoopz_p"]), float(NUM["mm_m3_tightxhoopz_p"]),
           float(min(1.0, 2 * min(1 + (_dbp <= 0).sum(), 1 + (_dbp >= 0).sum()) / (len(_dbp) + 1))),
           float(min(1.0, 2 * min(1 + (_ctb <= 0).sum(), 1 + (_ctb >= 0).sum()) / (len(_ctb) + 1)))]
TESTS = TESTS + ["ridge all-six R2 gain", "equal-weight six R2 gain", "joint model: Hoop G", "joint model: own-lean x Hoop G",
                 "day3 placebo: rank gain", "day3 placebo: Brier gain"]
PV = PV + [float(NUM["r6_p"]), float(NUM["ew_p"]),
           float(min(1.0, 2 * (1 - float(NUM["jm_hoop_ppos"]) / 100) + 1 / 201)),
           float(min(1.0, 2 * min(float(NUM["jm_int_ppos"]), 100 - float(NUM["jm_int_ppos"])) / 100 + 1 / 201)),
           float(NUM["pl3_rho_p"]), float(NUM["pl3_bri_p"])]
assert len(PV) == len(TESTS)
NUM["n_tests"] = len(TESTS)
PV = np.asarray(PV)
Q = 0.10
o_ = np.argsort(PV)
m_ = len(PV)
bh_ok = np.zeros(m_, bool)
passed = np.flatnonzero(PV[o_] <= Q * np.arange(1, m_ + 1) / m_)
if len(passed):
    bh_ok[o_[: passed.max() + 1]] = True
holm_ok = np.zeros(m_, bool)
for i_, j_ in enumerate(o_):
    if PV[j_] <= Q / (m_ - i_):
        holm_ok[j_] = True
    else:
        break
MULT = pd.DataFrame({"test": TESTS, "p": PV, "bh_q10": bh_ok, "holm_10": holm_ok})
MULT.to_csv(OUT / "multiplicity.csv", index=False)
put("n_bh", int(bh_ok.sum())); put("n_holm", int(holm_ok.sum())); put("q_fdr", Q)
TXT["bh_tests"] = list(MULT[MULT.bh_q10].test)
TXT["holm_tests"] = list(MULT[MULT.holm_10].test)
put("p_hoop_bh_rank", int(np.flatnonzero(PV[o_] == HO["hoop_z"]["p2"])[0] + 1))
import subprocess
import sys
subprocess.run([sys.executable, str(OUT / "verify.py")], check=True, capture_output=True)
VJ = json.loads((OUT / "verify.json").read_text())
put("v_n", VJ["v_n"]); put("v_slope", VJ["v_slope"], 1); put("v_lo", VJ["v_lo"], 1); put("v_hi", VJ["v_hi"], 1)
put("v_neg_classes", int(sum(v <= 0 for v in VJ["v_cls"])))
put("v_same_set", int(set(VJ["v_ids"]) == set(int(i) for i in AN.nfl_id)))
put("v_set_diff", len(set(VJ["v_ids"]) ^ set(int(i) for i in AN.nfl_id)))
TXT["nest_choice"] = TXT_NEST
TXT["nest_list"] = ", ".join(f"{v.lower()} for {c}" for c, v in TXT_NEST.items())
for c, v in zip(classes, VJ["v_cls"]):
    put(f"v_cls_{c}", v, 1)
TXT["lopo25_name"] = TXT_LOPO
TXT["extra78"] = TXT_EXTRA

# ============================================================================= render
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
(OUT / "text_bits.json").write_text(json.dumps(TXT, indent=1))
render(NUM, TXT)
print(json.dumps(NUM, indent=0))
print(json.dumps(TXT, indent=0))
