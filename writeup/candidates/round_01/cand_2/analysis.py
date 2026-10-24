"""Break Redirect Rate (BRR): WR/TE break mechanics in Combine route drills vs the same route family in NFL games.

Run from the repo root:
    uv run python writeup/candidates/round_01/cand_2/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json, figures/*.png and tables.json next to this file.

Definitions (identical code path for drills and games, 10 Hz x/y):
- velocity and acceleration vectors from Savitzky-Golay derivatives of x/y (window 7, order 2);
- the break is the first frame after >= 1.5 yd of travel, entered at > 2.5 yd/s, where heading changes
  >= 30 deg between -0.6 s and +0.6 s; within the next 0.8 s the sharpest such frame is kept;
- Break Redirect Rate (BRR, yd/s^2) = mean magnitude of the acceleration vector over the 1.0 s centred on the
  break. It counts braking, turning and re-acceleration in one number: how fast the velocity vector is being
  changed through the break;
- components: entry speed (max speed in the 1 s before the apex), apex speed (min speed within +-0.5 s of the
  break), exit speed (max speed in the 1 s after the apex); decel / re-accel are speed change over elapsed time.
Drill values are z-scored within drill x Combine year; game values within route family x position group.
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
DT = 0.1
RNG = np.random.default_rng(2027)
NBOOT = 2000

# Okabe-Ito colour-blind safe palette
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
      "verm": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 150})
FIGW = 1600 / 150  # inches -> 1600 px at 150 dpi

NUM: dict[str, str] = {}


def put(key: str, val, nd: int = 2) -> str:
    """Store a number exactly as it is printed in the writeup."""
    if isinstance(val, (int, np.integer)):
        s = str(int(val))
    else:
        s = f"{float(val):.{nd}f}"
    NUM[key] = s
    return s


def save(fig, name: str):
    fig.set_size_inches(FIGW, fig.get_figheight())
    fig.savefig(FIG / name, dpi=150, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- kinematics
def kin(x: np.ndarray, y: np.ndarray):
    w = 7 if len(x) >= 7 else max(3, (len(x) // 2) * 2 - 1)
    vx = savgol_filter(x, w, 2, deriv=1, delta=DT)
    vy = savgol_filter(y, w, 2, deriv=1, delta=DT)
    ax = savgol_filter(vx, w, 2, deriv=1, delta=DT)
    ay = savgol_filter(vy, w, 2, deriv=1, delta=DT)
    return vx, vy, ax, ay


def angdiff(a, b):
    d = np.abs(a - b) % 360
    return np.where(d > 180, 360 - d, d)


def break_mechanics(x, y, i0, i1, min_path=1.5, min_turn=30.0, h=6, entry_speed=2.5):
    """Main break between frame i0 (movement start / snap) and i1 (catch / throw). Returns dict or None."""
    n = len(x)
    if n < 2 * h + 3:
        return None
    vx, vy, ax, ay = kin(x, y)
    s = np.hypot(vx, vy)
    head = np.degrees(np.arctan2(vx, vy)) % 360
    amag = np.hypot(ax, ay)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    cum = cum - cum[i0]
    turn = np.zeros(n)
    turn[h:n - h] = angdiff(head[:n - 2 * h], head[2 * h:])
    ok = np.zeros(n, bool)
    ok[max(i0 + h, h):min(i1, n - h)] = True
    ok &= cum >= min_path
    sin = np.array([s[max(0, i - h - 3):i - h + 1].max() if i - h >= 0 else 0.0 for i in range(n)])
    ok &= sin > entry_speed
    cand = np.where(ok & (turn >= min_turn))[0]
    if len(cand) == 0:
        return None
    j = cand[0]
    seg = np.arange(j, min(j + 8, n - h))
    seg = seg[ok[seg]] if ok[seg].any() else np.array([j])
    i = int(seg[np.argmax(turn[seg])])
    lo, hi = max(0, i - 5), min(n, i + 6)
    apex = lo + int(np.argmin(s[lo:hi]))
    a0 = max(0, apex - 10)
    iin = a0 + int(np.argmax(s[a0:apex + 1]))
    b1 = min(n, apex + 11)
    iout = apex + int(np.argmax(s[apex:b1]))
    pre, post = slice(max(0, i - 8), max(1, i - 3)), slice(min(n - 1, i + 3), min(n, i + 8))
    v1 = np.array([vx[pre].mean(), vy[pre].mean()])
    v2 = np.array([vx[post].mean(), vy[post].mean()])
    cosang = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return dict(i_break=i, t_break=(i - i0) * DT, depth=float(cum[i]), angle=float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))),
                v_in=float(s[iin]), v_apex=float(s[apex]), v_out=float(s[iout]),
                decel=float((s[iin] - s[apex]) / max((apex - iin) * DT, DT)),
                reaccel=float((s[iout] - s[apex]) / max((iout - apex) * DT, DT)),
                brr=float(amag[lo:hi].mean()))


def boot_rho(a, b, n=NBOOT, cluster=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    rho = spearmanr(a, b)[0]
    out = []
    if cluster is None:
        for _ in range(n):
            ix = RNG.integers(0, len(a), len(a))
            out.append(spearmanr(a[ix], b[ix])[0])
    else:
        cl = np.asarray(cluster)[m]
        groups = {c: np.where(cl == c)[0] for c in np.unique(cl)}
        keys = list(groups)
        for _ in range(n):
            ix = np.concatenate([groups[keys[k]] for k in RNG.integers(0, len(keys), len(keys))])
            out.append(spearmanr(a[ix], b[ix])[0])
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(rho), float(lo), float(hi), int(len(a))


def zs(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


# ----------------------------------------------------------------------------- load
players = pd.read_parquet(P / "players.parquet")
cres = pd.read_parquet(P / "combine_results.parquet")
events = pd.read_parquet(P / "combine_events.parquet")

DRILL_FAMILY = {
    "SLANT_ROUTE_LEFT": "SLANT", "SLOT_STRIKE_ROUTE_LEFT": "SLANT",
    "SPEED_OUT_ROUTE_LEFT": "OUT",
    "CURL_ROUTE_RIGHT": "HITCH", "COMEBACK_ROUTE_RIGHT": "HITCH", "HOOK_ROUTE_RIGHT": "HITCH",
    "DAGGER_ROUTE_LEFT": "IN", "IN_ROUTE_LEFT": "IN",
    "SLOT_SAIL_ROUTE_RIGHT": "CORNER", "CORNER_ROUTE": "CORNER",
}
FAMILIES = ["SLANT", "OUT", "HITCH", "IN", "CORNER"]
DRILL_LABEL = {"SLANT_ROUTE_LEFT": "Slant", "SLOT_STRIKE_ROUTE_LEFT": "Slot strike", "SPEED_OUT_ROUTE_LEFT": "Speed out",
               "CURL_ROUTE_RIGHT": "Curl", "COMEBACK_ROUTE_RIGHT": "Comeback", "HOOK_ROUTE_RIGHT": "Hook (TE)",
               "DAGGER_ROUTE_LEFT": "Dagger", "IN_ROUTE_LEFT": "In (TE)", "SLOT_SAIL_ROUTE_RIGHT": "Slot sail",
               "CORNER_ROUTE": "Corner (TE)"}

ev = events[events.drill_name.isin(DRILL_FAMILY) & events.position_group.isin(["WR", "TE"])]
frames = pd.read_parquet(P / "combine_frames.parquet", filters=[("event_id", "in", ev.event_id.tolist())])

# ---- drill breaks (stop at first frame the ball is within 1 yd of the player, when the ball is tracked)
rows, traces = [], {}
for eid, d in frames.groupby("event_id", sort=False):
    pl_ = d[d.entity_type == "PLAYER"].sort_values("frame")
    bl = d[d.entity_type == "BALL"]
    x, y = pl_.x.to_numpy(), pl_.y.to_numpy()
    i1 = len(x)
    if len(bl):
        m = pl_[["frame", "x", "y"]].merge(bl[["frame", "x", "y"]], on="frame", suffixes=("", "_b"))
        c = m.frame[np.hypot(m.x - m.x_b, m.y - m.y_b) < 1.0]
        if len(c):
            i1 = int(np.searchsorted(pl_.frame.to_numpy(), c.iloc[0]))
    r = break_mechanics(x, y, 0, i1)
    if r:
        rows.append(dict(event_id=eid, **r))
        traces[eid] = (x, y)
cb = pd.DataFrame(rows).merge(ev[["event_id", "nfl_id", "draft_year", "drill_name", "attempt_no", "position_group", "display_name"]],
                              on="event_id")
cb["family"] = cb.drill_name.map(DRILL_FAMILY)
put("n_drill_events_total", len(ev))
put("n_drill_events_break", len(cb))
put("pct_drill_detected", 100 * len(cb) / len(ev), 0)
COMP = ["brr", "v_in", "v_apex", "v_out", "decel", "reaccel"]
for m in COMP:
    cb[m + "_z"] = cb.groupby(["drill_name", "draft_year"])[m].transform(zs)
# player x drill (average attempts), then player composite
cpd = cb.groupby(["nfl_id", "drill_name", "family", "position_group"], as_index=False)[[m + "_z" for m in COMP]].mean()
cplayer = cpd.groupby(["nfl_id", "position_group"])[[m + "_z" for m in COMP]].mean()
cplayer["n_drills"] = cpd.groupby(["nfl_id", "position_group"]).size()
cplayer = cplayer.reset_index().set_index("nfl_id")
put("n_combine_players", len(cplayer))
put("n_combine_wr", int((cplayer.position_group == "WR").sum()))
put("n_combine_te", int((cplayer.position_group == "TE").sum()))
put("median_drills_per_player", float(cplayer.n_drills.median()), 0)

# drill reliability: mean pairwise inter-drill rho (players with both), and split-half composite
wide = cpd.pivot_table(index="nfl_id", columns="drill_name", values="brr_z")
pair = []
cols = list(wide.columns)
for a in range(len(cols)):
    for b in range(a + 1, len(cols)):
        dd = wide[[cols[a], cols[b]]].dropna()
        if len(dd) >= 25:
            pair.append(spearmanr(dd.iloc[:, 0], dd.iloc[:, 1])[0])
put("drill_pair_rho_mean", np.mean(pair))
put("drill_pair_n", len(pair))
HALF_A = {"SLANT_ROUTE_LEFT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT", "SPEED_OUT_ROUTE_LEFT", "HOOK_ROUTE_RIGHT"}
ha = cpd[cpd.drill_name.isin(HALF_A)].groupby("nfl_id").brr_z.mean()
hb = cpd[~cpd.drill_name.isin(HALF_A)].groupby("nfl_id").brr_z.mean()
hh = pd.concat([ha, hb], axis=1, keys=["a", "b"]).dropna()
r_drill_half = boot_rho(hh.a, hh.b)
rel_drill = 2 * r_drill_half[0] / (1 + r_drill_half[0])
put("drill_split_rho", r_drill_half[0]); put("drill_split_lo", r_drill_half[1]); put("drill_split_hi", r_drill_half[2])
put("drill_split_n", r_drill_half[3]); put("rel_drill", rel_drill)

# ----------------------------------------------------------------------------- games
gp = pd.read_parquet(P / "game_plays_sample.parquet")
gp = gp[gp.week.between(1, 18)]  # sample was built from REG games via games.csv; guard anyway
gp = gp.merge(players[["nfl_id", "position_group", "draft_year", "display_name", "draft_overall_pick"]], on="nfl_id")
gp = gp[gp.position_group.isin(["WR", "TE"])].copy()
# the sample holds, per player, the 6 highest-|EPA| plays plus 6 random ones; recover the strata
gp["abs_epa"] = gp.expected_points_added.abs()
gp["epa_rank"] = gp.groupby("nfl_id").abs_epa.rank(ascending=False, method="first")
gp["stratum"] = np.where(gp.epa_rank <= 6, "high_epa", "random")
gf = pd.read_parquet(P / "game_frames_sample.parquet")
gf = gf.merge(gp[["game_id", "play_id", "nfl_id"]], on=["game_id", "play_id", "nfl_id"])
grows, gtraces = [], {}
for k, d in gf.groupby(["game_id", "play_id", "nfl_id"], sort=False):
    d = d.sort_values("frame")
    e_ = d.event.fillna("").to_numpy()
    if not ((e_ == "ball_snap").any() and (e_ == "pass_forward").any()):
        continue
    i0 = int(np.argmax(e_ == "ball_snap"))
    ipf = int(np.argmax(e_ == "pass_forward"))
    base = dict(game_id=k[0], play_id=k[1], nfl_id=k[2], t_throw=(ipf - i0) * DT)
    r = break_mechanics(d.x.to_numpy(), d.y.to_numpy(), i0, min(len(d), ipf + 6))
    if r:
        base.update(r)
        gtraces[k] = (d.x.to_numpy(), d.y.to_numpy(), i0)
    grows.append(base)
gb = pd.DataFrame(grows).merge(gp, on=["game_id", "play_id", "nfl_id"])
put("n_game_plays_all", len(gb))
put("n_game_players_all", gb.nfl_id.nunique())

# ---- separation over expected (all routes; expected model sees only context, never the receiver's movement)
gb["sep"] = gb.separation_at_pass_forward
gb["man"] = gb.team_coverage_type.fillna("").str.contains("MAN").astype(float)
gb["slot"] = (gb.lined_up_position == "SLOT_WR").astype(float)
gb["inline"] = gb.lined_up_position.isin(["TE", "FB", "RB"]).astype(float)
gb["cush"] = gb.cushion.fillna(gb.cushion.median())
gb["cush_na"] = gb.cushion.isna().astype(float)
gb["pl"] = gb.pass_length.fillna(gb.pass_length.median())
gb["route"] = gb.route_ran.fillna("OTHER")


def design(df: pd.DataFrame, routes: list[str]) -> np.ndarray:
    cols = [np.ones(len(df)), df.pl, df.pl ** 2 / 100, df.cush, df.cush_na, df.man, df.slot, df.inline, df.t_throw,
            df.target.astype(float), df.down, df.yards_to_go]
    cols += [(df.route == r).astype(float) for r in routes[1:]]
    return np.column_stack(cols)


routes = sorted(gb.route.unique())
gb["xsep"] = np.nan
fold_r2 = []
for cls in (2023, 2024, 2025):  # leave-one-Combine-class-out: no player is ever in his own expected model
    tr, te = gb.draft_year != cls, gb.draft_year == cls
    X_tr, X_te = design(gb[tr], routes), design(gb[te], routes)
    beta = np.linalg.lstsq(X_tr, gb.loc[tr, "sep"].to_numpy(), rcond=None)[0]
    gb.loc[te, "xsep"] = X_te @ beta
    yt = gb.loc[te, "sep"]
    fold_r2.append(1 - ((yt - gb.loc[te, "xsep"]) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
gb["soe"] = gb.sep - gb.xsep
yy = gb.sep
put("xsep_oof_r2", 1 - ((yy - gb.xsep) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())

# ---- break-route plays
g = gb[gb.route_ran.isin(FAMILIES)].copy()
put("n_break_route_plays", len(g))
put("pct_game_detected", 100 * g.brr.notna().mean(), 0)
g = g[g.brr.notna()].copy()
for m in COMP:
    g[m + "_z"] = g.groupby(["route_ran", "position_group"])[m].transform(zs)
put("n_game_plays", len(g)); put("n_game_players", g.nfl_id.nunique())
put("plays_per_player", g.groupby("nfl_id").size().mean(), 1)
fam_n = g.route_ran.value_counts()
for f in FAMILIES:
    put(f"n_plays_{f}", int(fam_n.get(f, 0)))

# median break profile by family (drill vs game), for the method figure / table
prof = []
for f in FAMILIES:
    dg, dc = g[g.route_ran == f], cb[cb.family == f]
    prof.append(dict(family=f, n_game=len(dg), n_drill=len(dc),
                     game_brr=dg.brr.median(), drill_brr=dc.brr.median(),
                     game_vin=dg.v_in.median(), drill_vin=dc.v_in.median(),
                     game_vapex=dg.v_apex.median(), drill_vapex=dc.v_apex.median(),
                     game_angle=dg.angle.median(), drill_angle=dc.angle.median()))
prof = pd.DataFrame(prof)
put("game_brr_median", g.brr.median()); put("drill_brr_median", cb.brr.median())
put("game_vin_median", g.v_in.median(), 1); put("drill_vin_median", cb.v_in.median(), 1)

# ---- play level: does BRR (and its parts) go with separation over expected?
play_rows = []
for m, lab in [("brr", "BRR"), ("v_in", "Entry speed"), ("decel", "Decel into break"), ("v_apex", "Speed through break"),
               ("reaccel", "Re-acceleration"), ("v_out", "Exit speed")]:
    r = boot_rho(g[m + "_z"], g.soe, cluster=g.nfl_id)
    play_rows.append(dict(metric=m, label=lab, rho=r[0], lo=r[1], hi=r[2], n=r[3]))
play = pd.DataFrame(play_rows).set_index("metric")
for m in play.index:
    put(f"play_{m}_rho", play.loc[m, "rho"]); put(f"play_{m}_lo", play.loc[m, "lo"]); put(f"play_{m}_hi", play.loc[m, "hi"])
# quintiles of BRR -> mean SOE
g["brr_q"] = pd.qcut(g.brr_z, 5, labels=False)
qtab = g.groupby("brr_q").soe.agg(["mean", "count", "std"])
qtab["se"] = qtab["std"] / np.sqrt(qtab["count"])
put("soe_q1", qtab["mean"].iloc[0]); put("soe_q5", qtab["mean"].iloc[-1])
vq = g.groupby(pd.qcut(g.v_apex_z, 5, labels=False)).soe.mean()
put("soe_vapex_q1", vq.iloc[0]); put("soe_vapex_q5", vq.iloc[-1])
# robustness: random stratum only
gr = g[g.stratum == "random"]
r = boot_rho(gr.v_apex_z, gr.soe, cluster=gr.nfl_id)
put("play_vapex_rand_rho", r[0]); put("play_vapex_rand_lo", r[1]); put("play_vapex_rand_hi", r[2]); put("play_vapex_rand_n", r[3])
r = boot_rho(gr.brr_z, gr.soe, cluster=gr.nfl_id)
put("play_brr_rand_rho", r[0]); put("play_brr_rand_lo", r[1]); put("play_brr_rand_hi", r[2])

# ---- player level game BRR and its reliability
g = g.sort_values(["nfl_id", "game_id", "play_id"])
g["half"] = g.groupby("nfl_id").cumcount() % 2
gh = g.groupby(["nfl_id", "half"]).brr_z.mean().unstack().dropna()
r_game_half = boot_rho(gh[0], gh[1])
rel_game = 2 * r_game_half[0] / (1 + r_game_half[0])
put("game_split_rho", r_game_half[0]); put("game_split_lo", r_game_half[1]); put("game_split_hi", r_game_half[2])
put("game_split_n", r_game_half[3]); put("rel_game", rel_game)
sh = g.groupby(["nfl_id", "half"]).soe.mean().unstack().dropna()
r_soe_half = boot_rho(sh[0], sh[1])
put("soe_split_rho", r_soe_half[0])
gplayer = g.groupby("nfl_id").agg(brr_game=("brr_z", "mean"), vapex_game=("v_apex_z", "mean"), soe=("soe", "mean"), n_plays=("brr", "size"))
ceiling = np.sqrt(max(rel_drill, 0) * max(rel_game, 0))
put("ceiling", ceiling)

# per-snap reliability and snaps needed (Spearman-Brown), minimum detectable rho
k_half = g.groupby("nfl_id").size().mean() / 2
r1 = r_game_half[0] / (k_half - (k_half - 1) * r_game_half[0])
put("rel_per_snap", r1, 3)
snaps_needed = 0.7 * (1 - r1) / (r1 * (1 - 0.7))
put("snaps_for_07", int(np.ceil(snaps_needed)))
mdr = np.tanh((1.96 + 0.84) / np.sqrt(len(gplayer) - 3))
put("min_detectable_rho", mdr)
put("latent_max_obs", ceiling * 1.0)

# robustness: player game BRR from every detected break (all route types), z within route x group
ga = gb[gb.brr.notna()].copy()
ga["brr_z"] = ga.groupby(["route", "position_group"]).brr.transform(zs)
ga = ga.sort_values(["nfl_id", "game_id", "play_id"])
ga["half"] = ga.groupby("nfl_id").cumcount() % 2
gah = ga.groupby(["nfl_id", "half"]).brr_z.mean().unstack().dropna()
r_all_half = boot_rho(gah[0], gah[1])
put("all_split_rho", r_all_half[0]); put("all_plays", len(ga)); put("all_plays_per_player", ga.groupby("nfl_id").size().mean(), 1)
gall = ga.groupby("nfl_id").brr_z.mean().rename("brr_game_all")

# drill vs game intensity by family
prof["ratio_brr"] = prof.game_brr / prof.drill_brr
prof["ratio_vin"] = prof.game_vin / prof.drill_vin
for _, r_ in prof.iterrows():
    put(f"ratio_brr_{r_.family}", 100 * r_.ratio_brr, 0); put(f"ratio_vin_{r_.family}", 100 * r_.ratio_vin, 0)
    put(f"angle_game_{r_.family}", r_.game_angle, 0); put(f"angle_drill_{r_.family}", r_.drill_angle, 0)
put("ratio_brr_all", 100 * g.brr.median() / cb.brr.median(), 0)

# ----------------------------------------------------------------------------- translation + baselines
X = cplayer.join(gplayer, how="inner").join(cres.set_index("nfl_id")[["forty", "three_cone", "short_shuttle", "combine_weight", "ngs_athleticism_score"]]) \
    .join(players.set_index("nfl_id")[["display_name", "draft_year", "draft_overall_pick"]])
X["log_pick"] = np.log(X.draft_overall_pick.fillna(262))
# within-position-group z so WR/TE body-type differences do not drive the baselines
for c in ["forty", "three_cone", "short_shuttle", "combine_weight", "ngs_athleticism_score", "log_pick"]:
    X[c + "_w"] = X.groupby("position_group")[c].transform(zs)
X = X.join(gall)
r = boot_rho(X.brr_z, X.brr_game_all)
put("tr_all_rho", r[0]); put("tr_all_lo", r[1]); put("tr_all_hi", r[2]); put("tr_all_n", r[3])
rel_all = 2 * r_all_half[0] / (1 + r_all_half[0])
ceiling_all = np.sqrt(rel_drill * rel_all)
put("rel_game_all", rel_all); put("ceiling_all", ceiling_all)
put("latent_upper", r[2] / ceiling_all)
put("n_translation", len(X))
put("n_translation_wr", int((X.position_group == "WR").sum())); put("n_translation_te", int((X.position_group == "TE").sum()))
tr_rows = []
PRED = [("brr_z", "Drill BRR (route drills)", +1), ("v_apex_z", "Drill speed through break", +1),
        ("forty_w", "Official 40 (faster)", -1), ("short_shuttle_w", "Official shuttle (faster)", -1),
        ("three_cone_w", "Official 3-cone (faster)", -1), ("combine_weight_w", "Weight (lighter)", -1),
        ("ngs_athleticism_score_w", "NGS athleticism", +1), ("log_pick_w", "Draft slot (earlier)", -1)]
for col, lab, sign in PRED:
    r = boot_rho(sign * X[col], X.brr_game)
    tr_rows.append(dict(col=col, label=lab, rho=r[0], lo=r[1], hi=r[2], n=r[3]))
trans = pd.DataFrame(tr_rows).set_index("col")
for c in trans.index:
    k = c.removesuffix("_w").removesuffix("_z")
    put(f"tr_{k}_rho", trans.loc[c, "rho"]); put(f"tr_{k}_lo", trans.loc[c, "lo"]); put(f"tr_{k}_hi", trans.loc[c, "hi"]); put(f"tr_{k}_n", int(trans.loc[c, "n"]))
# family-matched pairs (player x family: drill family z vs game family z)
cfam = cpd.groupby(["nfl_id", "family"]).brr_z.mean()
gfam = g.groupby(["nfl_id", "route_ran"]).brr_z.mean()
gfam.index.names = ["nfl_id", "family"]
jf = pd.concat([cfam.rename("d"), gfam.rename("g")], axis=1).dropna().reset_index()
r = fam_r = boot_rho(jf.d, jf.g, cluster=jf.nfl_id)
put("fam_rho", r[0]); put("fam_lo", r[1]); put("fam_hi", r[2]); put("fam_n", r[3]); put("fam_players", jf.nfl_id.nunique())
fam_each = {}
for f in FAMILIES:
    d = jf[jf.family == f]
    if len(d) >= 10:
        fam_each[f] = (spearmanr(d.d, d.g)[0], len(d))
        put(f"fam_{f}_rho", fam_each[f][0]); put(f"fam_{f}_n", len(d))
# per-class translation (out-of-class consistency)
cls_rows = []
for cls in (2023, 2024, 2025):
    d = X[X.draft_year == cls]
    r1 = boot_rho(d.brr_z, d.brr_game)
    r2 = boot_rho(-d.forty_w, d.brr_game)
    cls_rows.append(dict(cls=cls, drill=r1[0], dlo=r1[1], dhi=r1[2], forty=r2[0], flo=r2[1], fhi=r2[2], n=r1[3]))
    put(f"tr_{cls}_drill_rho", r1[0]); put(f"tr_{cls}_forty_rho", r2[0]); put(f"tr_{cls}_n", r1[3])
# drill BRR vs Combine numbers (what is the drill measuring?)
for col in ["forty_w", "short_shuttle_w", "three_cone_w", "combine_weight_w"]:
    d = X[["brr_z", col]].dropna()
    put(f"drill_vs_{col.removesuffix('_w')}", spearmanr(d.brr_z, -d[col])[0])


# ----------------------------------------------------------------------------- out-of-class prediction
def loco_r2(df: pd.DataFrame, feats: list[str], target: str):
    """Leave-one-Combine-class-out OLS (median-imputed features, standardized on training folds)."""
    df = df[df[target].notna()].copy()
    pred = pd.Series(np.nan, index=df.index)
    per = {}
    for cls in (2023, 2024, 2025):
        tr, te = df.draft_year != cls, df.draft_year == cls
        if te.sum() < 5:
            continue
        A = df[feats].copy()
        med = A[tr].median()
        A = A.fillna(med)
        mu, sd = A[tr].mean(), A[tr].std().replace(0, 1)
        A = (A - mu) / sd
        Xtr = np.column_stack([np.ones(tr.sum()), A[tr].to_numpy()])
        Xte = np.column_stack([np.ones(te.sum()), A[te].to_numpy()])
        lam = 1.0
        beta = np.linalg.solve(Xtr.T @ Xtr + lam * np.diag([0] + [1] * len(feats)), Xtr.T @ df.loc[tr, target].to_numpy())
        pred[te] = Xte @ beta
        yt = df.loc[te, target]
        per[cls] = 1 - ((yt - pred[te]) ** 2).sum() / ((yt - df.loc[tr, target].mean()) ** 2).sum()
    y = df[target]
    tot = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return tot, per, pred


ss = pd.read_parquet(P / "player_season_stats.parquet")
ss = ss[ss.position_group.isin(["WR", "TE"]) & (ss.routes > 0)]
season = ss.groupby("nfl_id").apply(lambda d: pd.Series({
    "routes": d.routes.sum(), "sep_season": np.average(d.avg_separation, weights=d.routes),
    "yprr": d.rec_yards.sum() / d.routes.sum()}), include_groups=False)
Y = cplayer.join(season, how="inner").join(cres.set_index("nfl_id")[["forty", "combine_weight", "ngs_athleticism_score", "short_shuttle", "three_cone"]]) \
    .join(players.set_index("nfl_id")[["display_name", "draft_year", "draft_overall_pick"]]).join(gplayer[["brr_game"]])
Y = Y[Y.routes >= 150].copy()
Y["log_pick"] = np.log(Y.draft_overall_pick.fillna(262))
Y["is_te"] = (Y.position_group == "TE").astype(float)
put("n_season", len(Y)); put("season_min_routes", 150)
BASE = ["log_pick", "forty", "combine_weight", "ngs_athleticism_score", "is_te"]
cv_rows = []
for target, lab in [("sep_season", "Season separation (yd)"), ("yprr", "Yards per route run")]:
    b, bper, _ = loco_r2(Y, BASE, target)
    a, aper, _ = loco_r2(Y, BASE + ["brr_z"], target)
    gg, gper, _ = loco_r2(Y[Y.brr_game.notna()], BASE + ["brr_game"], target)
    gb_, gbper, _ = loco_r2(Y[Y.brr_game.notna()], BASE, target)
    # bootstrap the delta R^2 of adding drill BRR by resampling players within the LOCO procedure
    deltas = []
    for _ in range(300):
        ix = RNG.integers(0, len(Y), len(Y))
        Yb = Y.iloc[ix].reset_index(drop=True)
        deltas.append(loco_r2(Yb, BASE + ["brr_z"], target)[0] - loco_r2(Yb, BASE, target)[0])
    dlo, dhi = np.percentile(deltas, [2.5, 97.5])
    rho = boot_rho(Y.brr_z, Y[target])
    k = "sep" if target == "sep_season" else "yprr"
    put(f"cv_{k}_base", b); put(f"cv_{k}_plus", a); put(f"cv_{k}_delta", a - b); put(f"cv_{k}_dlo", dlo); put(f"cv_{k}_dhi", dhi)
    put(f"cv_{k}_game_delta", gg - gb_)
    put(f"rho_{k}_drill", rho[0]); put(f"rho_{k}_drill_lo", rho[1]); put(f"rho_{k}_drill_hi", rho[2])
    cv_rows.append(dict(target=lab, k=k, base=b, plus=a, base_per=bper, plus_per=aper, dlo=dlo, dhi=dhi, game_base=gb_, game_plus=gg))
    for cls in (2023, 2024, 2025):
        if cls in bper:
            put(f"cv_{k}_{cls}_base", bper[cls]); put(f"cv_{k}_{cls}_plus", aper[cls])
put("n_season_2025", int((Y.draft_year == 2025).sum()))

# ----------------------------------------------------------------------------- named players
X["gap"] = X.brr_game - X.brr_z
named = []
top_drill = X.sort_values("brr_z", ascending=False).head(3)
top_game = X.sort_values("brr_game", ascending=False).head(3)
for tag, d in [("Top drill BRR", top_drill), ("Top game BRR", top_game)]:
    for nid, r in d.iterrows():
        named.append(dict(tag=tag, name=r.display_name, pos=r.position_group, cls=int(r.draft_year),
                          drill=r.brr_z, game=r.brr_game, n_plays=int(r.n_plays), forty=r.forty))
named = pd.DataFrame(named)
for i, r in named.iterrows():
    put(f"named_{i}_drill", r.drill); put(f"named_{i}_game", r.game); put(f"named_{i}_plays", int(r.n_plays))
    if not np.isnan(r.forty):
        put(f"named_{i}_forty", r.forty)

# ----------------------------------------------------------------------------- figures
# Fig 1: method - one drill and one game break for the same player, speed and |a| aligned on the break
cand_players = X[(X.n_plays >= 4)].index
ex_pid = None
for pid in X.loc[cand_players].sort_values("brr_z", ascending=False).index:
    dc = cb[(cb.nfl_id == pid) & (cb.drill_name == "SPEED_OUT_ROUTE_LEFT")]
    dg_ = g[(g.nfl_id == pid) & (g.route_ran == "OUT")]
    if len(dc) and len(dg_):
        ex_pid = pid
        break
if ex_pid is None:
    ex_pid = X.index[0]
ex_name = X.loc[ex_pid, "display_name"]
NUM["example_player"] = ex_name
fig, axes = plt.subplots(1, 3, figsize=(FIGW, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
dc = cb[(cb.nfl_id == ex_pid) & (cb.drill_name == "SPEED_OUT_ROUTE_LEFT")]
dg_ = g[(g.nfl_id == ex_pid) & (g.route_ran == "OUT")]
dc_row = dc.iloc[0] if len(dc) else cb[cb.nfl_id == ex_pid].iloc[0]
dg_row = dg_.iloc[0] if len(dg_) else g[g.nfl_id == ex_pid].iloc[0]
xc, yc = traces[dc_row.event_id]
kg = (dg_row.game_id, dg_row.play_id, dg_row.nfl_id)
xg, yg, i0g = gtraces[kg]
for ax, (x, y, ib, ttl, col) in zip(axes[:2], [(xc, yc, int(dc_row.i_break), f"Combine: {DRILL_LABEL.get(dc_row.drill_name, dc_row.drill_name)}", OI["blue"]),
                                             (xg[i0g:], yg[i0g:], int(dg_row.i_break) - i0g, f"NFL {dg_row.season}: {dg_row.route_ran} route", OI["verm"])]):
    x = x - x[0]
    y = y - y[0]
    # rotate so the stem points up
    ang = np.arctan2(x[min(ib, len(x) - 1) - 3] if ib > 3 else x[-1], y[min(ib, len(y) - 1) - 3] if ib > 3 else y[-1])
    xr = x * np.cos(ang) - y * np.sin(ang)
    yr = x * np.sin(ang) + y * np.cos(ang)
    if np.nanmean(xr[ib:]) < 0:
        xr = -xr
    lo, hi = max(0, ib - 5), min(len(x), ib + 6)
    ax.plot(xr, yr, color=OI["grey"], lw=1.5)
    ax.plot(xr[lo:hi], yr[lo:hi], color=col, lw=4, solid_capstyle="round", label="1.0 s BRR window")
    ax.scatter([xr[0]], [yr[0]], color="k", s=20, zorder=3)
    ax.set_aspect("equal")
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel("yards")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
axes[0].set_ylabel("yards (stem direction up)")
ax = axes[2]
for (x, y, ib, lab, col) in [(xc, yc, int(dc_row.i_break), "Combine drill", OI["blue"]), (xg, yg, int(dg_row.i_break), "NFL game", OI["verm"])]:
    vx, vy, axx, ayy = kin(x, y)
    tt = (np.arange(len(x)) - ib) * DT
    m = (tt >= -1.5) & (tt <= 1.5)
    ax.plot(tt[m], np.hypot(vx, vy)[m], color=col, lw=2.2, label=f"{lab}: speed")
    ax.plot(tt[m], np.hypot(axx, ayy)[m], color=col, lw=1.6, ls="--", label=f"{lab}: |accel|")
ax.axvspan(-0.5, 0.5, color=OI["yellow"], alpha=0.35, lw=0)
ytop = ax.get_ylim()[1]
ax.set_ylim(0, ytop * 1.12)
ax.text(0, ytop * 1.1, "BRR = mean |accel| in this 1.0 s", ha="center", va="top", fontsize=9)
ax.set_xlabel("seconds from break")
ax.set_ylabel("yd/s  |  yd/s²")
ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
ax.set_title("Same number, same code, both settings", fontsize=11)
fig.suptitle(f"Break Redirect Rate, measured identically at the Combine and on Sundays ({ex_name})", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig1_method.png")

# Fig 2: play-level components vs SOE (forest) + SOE by BRR quintile
fig, axes = plt.subplots(1, 2, figsize=(FIGW, 4.3), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
pp = play.iloc[::-1]
yv = np.arange(len(pp))
cols = [OI["blue"] if m == "brr" else OI["grey"] for m in pp.index]
ax.errorbar(pp.rho, yv, xerr=[pp.rho - pp.lo, pp.hi - pp.rho], fmt="none", ecolor=cols, elinewidth=2.5, capsize=0)
ax.scatter(pp.rho, yv, c=cols, s=55, zorder=3)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(yv, pp.label)
ax.set_xlabel("Spearman ρ with separation over expected (95% CI, player-clustered)")
ax.set_title(f"Play level: {len(g)} break-route snaps, {g.nfl_id.nunique()} players", fontsize=11)
ax = axes[1]
ax.bar(qtab.index + 1, qtab["mean"], yerr=1.96 * qtab["se"], color=OI["sky"], edgecolor=OI["blue"], capsize=3)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("In-game BRR quintile (1 = gentlest break)")
ax.set_ylabel("Separation over expected (yd)")
ax.set_title("Mean SOE by BRR quintile (±95% CI)", fontsize=11)
fig.tight_layout()
save(fig, "fig2_play_level.png")

# Fig 3: what predicts a player's in-game BRR? forest with reliability ceiling
fig, ax = plt.subplots(figsize=(FIGW, 4.6))
tt_ = trans.iloc[::-1]
yv = np.arange(len(tt_))
cols = [OI["verm"] if c.startswith(("brr", "v_apex")) else OI["blue"] for c in tt_.index]
ax.errorbar(tt_.rho, yv, xerr=[tt_.rho - tt_.lo, tt_.hi - tt_.rho], fmt="none", ecolor=cols, elinewidth=2.5)
ax.scatter(tt_.rho, yv, c=cols, s=60, zorder=3)
for yi, (_, r) in zip(yv, tt_.iterrows()):
    ax.text(0.62, yi, f"n={int(r.n)}", va="center", fontsize=9, color=OI["grey"])
ax.axvline(0, color="k", lw=0.8)
ax.axvline(ceiling, color=OI["green"], lw=1.5, ls="--")
ax.text(ceiling, len(tt_) - 0.4, f" max observable ρ\n given reliability ({ceiling:.2f})", color=OI["green"], fontsize=9, va="top")
ax.set_yticks(yv, tt_.label)
ax.set_xlim(-0.6, 0.7)
ax.set_xlabel("Spearman ρ with player's in-game BRR (signed so right = 'better number, sharper breaks'; 95% bootstrap CI)")
ax.set_title("No pre-draft number predicts in-game BRR strongly; route-drill numbers (orange) least of all", fontsize=11)
fig.tight_layout()
save(fig, "fig3_translation.png")

# Fig 4: scatter drill vs game BRR with named players
fig, ax = plt.subplots(figsize=(FIGW, 5.6))
for grp, col, mk in [("WR", OI["blue"], "o"), ("TE", OI["orange"], "s")]:
    d = X[X.position_group == grp]
    ax.scatter(d.brr_z, d.brr_game, s=18 + 6 * d.n_plays, color=col, alpha=0.7, marker=mk, edgecolor="white", lw=0.5, label=f"{grp} (n={len(d)})")
offs = [(6, -14), (6, 6), (-100, 12), (6, 4), (6, 8), (-90, 8)]
for (_, r), off in zip(named.iterrows(), offs):
    row = X[X.display_name == r["name"]].iloc[0]
    ax.annotate(r["name"], (row.brr_z, row.brr_game), xytext=off, textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="-", color=OI["grey"], lw=0.6))
ax.axhline(0, color=OI["grey"], lw=0.7); ax.axvline(0, color=OI["grey"], lw=0.7)
ax.set_xlabel("Combine BRR (mean z across break drills, year-standardized)")
ax.set_ylabel("NFL BRR (mean z on SLANT/OUT/HITCH/IN/CORNER snaps)")
rr = trans.loc["brr_z"]
ax.set_title(f"Drill vs game Break Redirect Rate: ρ = {rr.rho:.2f} [{rr.lo:.2f}, {rr.hi:.2f}], n = {int(rr.n)}  (marker size = game snaps)", fontsize=11)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
save(fig, "fig4_scatter.png")

# Fig 5: reliability bars + out-of-class translation per class
fig, axes = plt.subplots(1, 2, figsize=(FIGW, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axes[0]
bars = [("Drill vs drill\n(split-half)", r_drill_half, OI["blue"]), ("Game vs game\n(split-half)", r_game_half, OI["verm"]),
        ("Drill vs game\n(all)", tuple(trans.loc["brr_z", ["rho", "lo", "hi", "n"]]), OI["grey"]),
        ("Drill vs game\n(same family)", fam_r, OI["purple"])]
for i, (lab, r, col) in enumerate(bars):
    ax.bar(i, r[0], color=col, alpha=0.85)
    ax.errorbar(i, r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]], color="k", capsize=4)
ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=9)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Spearman ρ (95% CI)")
ax.set_title("BRR is a trait in each setting, not across them", fontsize=11)
ax = axes[1]
cr_ = pd.DataFrame(cls_rows)
xx = np.arange(len(cr_))
ax.errorbar(xx - 0.12, cr_.drill, yerr=[cr_.drill - cr_.dlo, cr_.dhi - cr_.drill], fmt="o", color=OI["verm"], label="Drill BRR", capsize=3)
ax.errorbar(xx + 0.12, cr_.forty, yerr=[cr_.forty - cr_.flo, cr_.fhi - cr_.forty], fmt="s", color=OI["blue"], label="Official 40 (faster)", capsize=3)
ax.set_xticks(xx, [f"{c} class\n(n={n})" for c, n in zip(cr_.cls, cr_.n)])
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("ρ with in-game BRR")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Each Combine class on its own", fontsize=11)
fig.tight_layout()
save(fig, "fig5_reliability_by_class.png")

# Fig 6: out-of-class CV R^2 for season outcomes
fig, axes = plt.subplots(1, 2, figsize=(FIGW, 4.0))
for ax, row in zip(axes, cv_rows):
    cl = sorted(row["base_per"])
    xx = np.arange(len(cl) + 1)
    b = [row["base_per"][c] for c in cl] + [row["base"]]
    a = [row["plus_per"][c] for c in cl] + [row["plus"]]
    ax.bar(xx - 0.2, b, 0.4, color=OI["grey"], label="Pick + 40 + weight + NGS athleticism + TE flag")
    ax.bar(xx + 0.2, a, 0.4, color=OI["orange"], label="… + Combine BRR")
    ax.set_xticks(xx, [f"held-out\n{c}" for c in cl] + ["pooled\nout-of-fold"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{row['target']} (n={len(Y)})", fontsize=11)
    ax.set_ylabel("Out-of-class R²")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=9, loc="upper center", ncol=2)
fig.tight_layout(rect=(0, 0, 1, 0.92))
save(fig, "fig6_out_of_class.png")

# Fig 7: drill vs game break profile per family
fig, axes = plt.subplots(1, 3, figsize=(FIGW, 3.9))
for ax, (gc, dc_, lab) in zip(axes, [("game_vin", "drill_vin", "Entry speed (yd/s)"), ("game_brr", "drill_brr", "BRR (yd/s²)"),
                                     ("game_angle", "drill_angle", "Break angle (deg)")]):
    yv = np.arange(len(prof))[::-1]
    for yi, (_, r_) in zip(yv, prof.iterrows()):
        ax.plot([r_[gc], r_[dc_]], [yi, yi], color=OI["grey"], lw=1.5, zorder=1)
    ax.scatter(prof[dc_], yv, color=OI["blue"], s=60, label="Combine drill (median)", zorder=2)
    ax.scatter(prof[gc], yv, color=OI["verm"], s=60, marker="D", label="NFL snaps (median)", zorder=2)
    ax.set_yticks(yv, [f"{f} ({int(a)}/{int(b)})" for f, a, b in zip(prof.family, prof.n_drill, prof.n_game)] if ax is axes[0] else [""] * len(yv))
    ax.set_xlabel(lab)
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
axes[0].set_ylabel("route family (drill reps / game snaps)")
fig.suptitle("NFL breaks are entered slower and redirected more gently than the same breaks at the Combine", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "fig7_intensity.png")

# ----------------------------------------------------------------------------- outputs
NUM["abs_values"] = sorted({v.lstrip("-") for v in NUM.values() if isinstance(v, str)})
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
tables = {"named": named.round(2).to_dict("records"), "profile": prof.round(2).to_dict("records"),
          "play": play.round(3).reset_index().to_dict("records"), "translation": trans.round(3).reset_index().to_dict("records"),
          "classes": pd.DataFrame(cls_rows).round(3).to_dict("records"),
          "cv": [{k: (v if not isinstance(v, dict) else {str(a): round(b, 3) for a, b in v.items()}) for k, v in r.items()} for r in cv_rows],
          "fam_each": {k: [round(v[0], 3), v[1]] for k, v in fam_each.items()}}
(OUT / "tables.json").write_text(json.dumps(tables, indent=1, default=float))
print(json.dumps({k: v for k, v in NUM.items() if k != "abs_values"}, indent=0))
print(named.round(2).to_string())
print(prof.round(2).to_string())
