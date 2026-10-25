"""Envelope Use: how much of his Combine sprint profile a player shows on NFL snaps.

Round 18 / cand_3. Reads data/processed/*.parquet only, runs in well under a minute,
writes numbers.json, board.csv, inputs.csv and figures/*.png next to this file.

Movement type: straight-line acceleration. The same code grades the Combine 40 (combine_frames)
and straight-line NFL frames (game_frames_sample, random plays only).

  Combine acceleration-speed (A-S) line: on each 40 attempt, longitudinal acceleration
  (centred difference of speed over 0.2 s) is regressed on speed from 1.5 yd/s to top speed:
  a = A0 * (1 - s / S0).  A0 = max acceleration from a standstill, S0 = top-speed intercept.
  NFL frontier: on random sampled snaps, frames after the snap where the path is straight
  (direction change under 20 deg over 0.4 s) and the player is speeding up; in 0.5 yd/s speed
  bins from 3 yd/s, the 90th percentile of acceleration (bins with 8+ frames).
  Envelope Use (EU) = median over bins (up to 80% of S0) of NFL frontier / Combine line.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
D = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(2027)
NB = 2000
MARGIN = 0.10  # yards per route per sd, set before the production test (about 50 yards over a 500-route season)

# Okabe-Ito
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2",
          verm="#D55E00", purple="#CC79A7", black="#000000", grey="#999999")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
W = 1600 / 150

N: dict = {}
PRINTED: dict = {}  # values whose printed form has a trailing zero (0.70 is stored as 0.7)


def put(key, val, nd=2):
    v = float(val)
    N[key] = int(round(v)) if nd == 0 else round(v, nd)
    if nd > 0 and f"{N[key]:.{nd}f}" != str(N[key]):
        PRINTED[key] = f"{N[key]:.{nd}f}"
    return N[key]


def fmt(v, nd=2):
    s = f"{v:.{nd}f}"
    return s.replace("-", "−")


# ----------------------------------------------------------------------------- load
pl = pd.read_parquet(D / "players.parquet")
cr = pd.read_parquet(D / "combine_results.parquet")
ev = pd.read_parquet(D / "combine_events.parquet")
cf = pd.read_parquet(D / "combine_frames.parquet", columns=["event_id", "frame", "entity_type", "s"])
gp = pd.read_parquet(D / "game_plays_sample.parquet")
gf = pd.read_parquet(D / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "s", "dir"])
ss = pd.read_parquet(D / "player_season_stats.parquet")
pl["grp"] = pl.position_group

# ----------------------------------------------------------------------------- Combine 40 A-S line
e40 = ev[ev.drill_name == "FORTY_YARD_DASH"][["event_id", "nfl_id", "draft_year"]]
c = cf[(cf.entity_type == "PLAYER") & cf.event_id.isin(e40.event_id)].sort_values(["event_id", "frame"]).copy()
g_ = c.groupby("event_id", sort=False).s
c["al"] = (g_.shift(-1) - g_.shift(1)) / 0.2
rows = []
for eid, d in c.groupby("event_id", sort=False):
    d = d.dropna()
    if d.empty:
        continue
    d = d.iloc[: d.s.values.argmax() + 1]
    d = d[d.s >= 1.5]
    if len(d) < 8:
        continue
    b = np.polyfit(d.s, d.al, 1)
    if b[0] >= 0:
        continue
    rows.append((eid, b[1], -b[1] / b[0], d.s.max()))
att = pd.DataFrame(rows, columns=["event_id", "A0", "S0", "vtop"]).merge(e40)
att["rep"] = att.groupby("nfl_id").cumcount()
p40 = att.groupby("nfl_id")[["A0", "S0", "vtop"]].mean()
two = att[att.rep < 2].pivot(index="nfl_id", columns="rep", values=["A0", "S0"]).dropna()
rel40_A0 = stats.spearmanr(two[("A0", 0)], two[("A0", 1)])[0]
rel40_S0 = stats.spearmanr(two[("S0", 0)], two[("S0", 1)])[0]
put("n_40_attempts", len(att), 0)
put("n_40_players", len(p40), 0)
put("n_40_two_reps", len(two), 0)
put("rel40_A0", rel40_A0)
put("rel40_S0", rel40_S0)
put("A0_median", att.A0.median(), 1)
put("S0_median", att.S0.median(), 1)
# shrink each player's Combine line toward his position-class mean by its rep-count reliability
# (Spearman-Brown), so a single noisy rep cannot make a huge denominator
nrep = att.groupby("nfl_id").size().rename("k")
p40 = p40.join(nrep).join(pl.set_index("nfl_id")[["grp", "draft_year"]])
p40["A0_raw"] = p40.A0
for col, r1 in [("A0", rel40_A0), ("S0", rel40_S0)]:
    relk = p40.k * r1 / (1 + (p40.k - 1) * r1)
    m = p40.groupby(["grp", "draft_year"])[col].transform("mean")
    p40[col] = m + relk * (p40[col] - m)
p40 = p40[["A0", "S0", "vtop", "A0_raw", "k"]]
put("share_one_rep", (p40.k == 1).mean() * 100, 0)

# ----------------------------------------------------------------------------- NFL straight-line frontier
plays = gp[gp.sample_kind == "random"].dropna(subset=["snap_frame"]).merge(pl[["nfl_id", "grp"]])
keep = (
    ((plays.grp.isin(["WR", "TE"])) & plays.route_ran.notna())
    | (plays.grp == "DL")
    | ((plays.grp.isin(["OL", "DB"])) & plays.pass_result.notna())
)
plays = plays[keep][["game_id", "play_id", "nfl_id", "snap_frame", "grp", "route_ran", "lined_up_position"]]
g = gf.merge(plays, on=["game_id", "play_id", "nfl_id"])
g = g[g.frame >= g.snap_frame].sort_values(["game_id", "play_id", "nfl_id", "frame"])
gg = g.groupby(["game_id", "play_id", "nfl_id"], sort=False)
g["al"] = (gg.s.shift(-1) - gg.s.shift(1)) / 0.2
turn = (gg.dir.shift(-2) - gg.dir.shift(2) + 180) % 360 - 180
g["straight"] = turn.abs() < 20
g["pidx"] = g.groupby("nfl_id").play_id.transform(lambda x: pd.factorize(x)[0])
fr = g[g.straight & (g.al > 0) & (g.s >= 3)].copy()
fr["bin"] = np.floor(fr.s * 2) / 2 + 0.25
put("n_game_frames", len(g), 0)
put("n_straight_frames", len(fr), 0)


def frontier(f):
    q = f.groupby(["nfl_id", "bin"]).al.agg(lambda x: np.quantile(x, 0.9) if len(x) >= 8 else np.nan)
    return q.dropna().reset_index()


def envelope(q):
    q = q.merge(p40.reset_index(), on="nfl_id")
    q = q[q.bin <= 0.8 * q.S0].copy()
    q["pred"] = q.A0 * (1 - q.bin / q.S0)
    q["r"] = q.al / q.pred
    out = q.groupby("nfl_id").agg(EU=("r", "median"), nbins=("r", "size"))
    return out[out.nbins >= 3]


def game_line(q):
    rr = []
    for pid, d in q.groupby("nfl_id"):
        if len(d) < 4:
            continue
        b = np.polyfit(d.bin, d.al, 1)
        rr.append((pid, b[1], -b[1] / b[0] if b[0] < 0 else np.nan))
    return pd.DataFrame(rr, columns=["nfl_id", "A0_g", "S0_g"]).set_index("nfl_id")


Q = frontier(fr)
E = envelope(Q).join(game_line(Q))
E = E.join(g.groupby("nfl_id").play_id.nunique().rename("n_plays"))
halves = [envelope(frontier(fr[fr.pidx % 2 == h])).EU.rename(f"h{h}") for h in (0, 1)]
E = E.join(pd.concat(halves, axis=1))
hl = [game_line(frontier(fr[fr.pidx % 2 == h])).add_suffix(f"_h{h}") for h in (0, 1)]
E = E.join(pd.concat(hl, axis=1))
E = E.join(pl.set_index("nfl_id")[["grp", "display_name", "draft_year", "draft_overall_pick", "combine_weight"]])
E = E.join(p40)
E = E.join(cr.set_index("nfl_id")[["forty", "ngs_athleticism_score"]])

# ----------------------------------------------------------------------------- production
agg = ss.groupby("nfl_id")[["rec_yards", "routes", "pressures", "pass_rush_snaps", "pressures_allowed", "pass_blocks",
                            "tackles", "plays"]].sum()
prod = pd.concat({
    "WR": (agg.rec_yards / agg.routes).where(agg.routes >= 100),
    "TE": (agg.rec_yards / agg.routes).where(agg.routes >= 100),
    "DL": (100 * agg.pressures / agg.pass_rush_snaps).where(agg.pass_rush_snaps >= 100),
    "OL": (-100 * agg.pressures_allowed / agg.pass_blocks).where(agg.pass_blocks >= 100),
    "DB": (100 * agg.tackles / agg.plays).where(agg.plays >= 100),
}, axis=1)
E["y"] = [prod.at[i, gr] if i in prod.index else np.nan for i, gr in zip(E.index, E.grp)]
E["routes"] = agg.routes.reindex(E.index)


def zyear(d, col, by=("grp", "draft_year")):
    gb = d.groupby(list(by))[col]
    return (d[col] - gb.transform("mean")) / gb.transform("std")


for col in ["EU", "A0", "S0", "A0_g", "S0_g", "vtop"]:
    E[col + "_z"] = zyear(E, col)
E["lpick"] = np.log(E.draft_overall_pick.fillna(260))
E["forty_miss"] = E.forty.isna().astype(float)
E["forty_f"] = E.forty.fillna(E.groupby(["grp", "draft_year"]).forty.transform("mean"))
E["te"] = (E.grp == "TE").astype(float)


def boot_ci(x, y, f, n=NB, strat=None):
    vals = []
    idx = np.arange(len(x))
    for _ in range(n):
        if strat is None:
            b = RNG.choice(idx, len(idx))
        else:
            b = np.concatenate([RNG.choice(idx[strat == s], (strat == s).sum()) for s in np.unique(strat)])
        vals.append(f(x[b], y[b]))
    return np.nanpercentile(vals, [5, 95])


def rho(x, y):
    return stats.spearmanr(x, y)[0]


def splithalf(d):
    d = d.dropna(subset=["h0", "h1"])
    r = rho(d.h0.values, d.h1.values)
    return 2 * r / (1 + r), len(d)


def loco(df, base, extra=None):
    """Leave-one-class-out predictions. Returns held-out preds for base and base+extra."""
    cols = base + ([extra] if extra else [])
    pred = np.full(len(df), np.nan)
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    yr = df.draft_year.values
    y = df.y.values
    for yy in np.unique(yr):
        tr, te = yr != yy, yr == yy
        beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        pred[te] = X[te] @ beta
    return pred


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


# ----------------------------------------------------------------------------- screen across five groups
screen = []
for grp in ["WR", "TE", "DL", "OL", "DB"]:
    d = E[E.grp == grp].dropna(subset=["A0_z", "A0_g_z"])
    rel, nrel = splithalf(E[E.grp == grp])
    rA = rho(d.A0_z.values, d.A0_g_z.values)
    ciA = boot_ci(d.A0_z.values, d.A0_g_z.values, rho)
    dd = d.dropna(subset=["S0_g_z"])
    rS = rho(dd.S0_z.values, dd.S0_g_z.values)
    ciS = boot_ci(dd.S0_z.values, dd.S0_g_z.values, rho)
    screen.append(dict(grp=grp, n=len(d), rA=rA, ciA=ciA, rS=rS, ciS=ciS, rel=rel, nrel=nrel))
    put(f"scr_{grp}_n", len(d), 0)
    put(f"scr_{grp}_rA", rA)
    put(f"scr_{grp}_rS", rS)
    put(f"scr_{grp}_rel", rel)
    put(f"scr_{grp}_ciA_lo", ciA[0]); put(f"scr_{grp}_ciA_hi", ciA[1])
    put(f"scr_{grp}_ciS_lo", ciS[0]); put(f"scr_{grp}_ciS_hi", ciS[1])

# WR detail: reliability of each side and noise ceiling for the translation correlations
wr = E[E.grp == "WR"]
for side in ["A0", "S0"]:
    d2 = wr.dropna(subset=[f"{side}_g_h0", f"{side}_g_h1"])
    r_ = rho(d2[f"{side}_g_h0"].values, d2[f"{side}_g_h1"].values)
    relg = 2 * r_ / (1 + r_)
    put(f"wr_rel_game_{side}", relg)
    rel40 = rel40_A0 if side == "A0" else rel40_S0
    put(f"wr_ceiling_{side}", np.sqrt(max(relg, 0) * rel40))
    put(f"wr_disatt_{side}", N[f"scr_WR_r{side[0]}"] / np.sqrt(max(relg, 1e-6) * rel40))
# what EU tracks: route mix and alignment on the sampled snaps
rp = plays[plays.grp.isin(["WR", "TE"])].copy()
rp["vert"] = rp.route_ran.isin(["GO", "POST", "CORNER", "WHEEL"]).astype(float)
rp["slot"] = rp.lined_up_position.isin(["SLOT_WR"]).astype(float)
mix = rp.groupby("nfl_id")[["vert", "slot"]].mean()
mm = E[E.grp == "WR"].join(mix, how="inner")
put("wr_rho_eu_vert", rho(mm.EU_z.values, mm.vert.values))
put("wr_rho_eu_slot", rho(mm.EU_z.values, mm.slot.values))
put("wr_vert_share", mm.vert.median() * 100, 0)
put("rho_A0_EU_wr", rho(wr.A0_z.values, wr.EU_z.values))

# production screen: held-out dR2 of EU beyond pick + weight (+ TE flag for receivers)
for grp, members in [("REC", ["WR", "TE"]), ("DL", ["DL"]), ("OL", ["OL"]), ("DB", ["DB"])]:
    d = E[E.grp.isin(members)].dropna(subset=["y", "EU_z"]).copy()
    base = ["lpick", "combine_weight"] + (["te"] if grp == "REC" else [])
    pb, pe = loco(d, base), loco(d, base, "EU_z")
    put(f"scr_{grp}_prod_n", len(d), 0)
    put(f"scr_{grp}_dR2", r2(d.y.values, pe) - r2(d.y.values, pb), 3)
    for s in screen:
        if s["grp"] in members:
            s["dR2"] = N[f"scr_{grp}_dR2"]
            s["prod_n"] = len(d)

# ----------------------------------------------------------------------------- receivers
R = E[E.grp.isin(["WR", "TE"])].copy()
put("rec_n_eu", len(R), 0)
put("rec_n_wr", (R.grp == "WR").sum(), 0)
put("rec_n_te", (R.grp == "TE").sum(), 0)
rel_eu, nrel_eu = splithalf(R)
put("rec_rel_eu", rel_eu)
put("rec_n_rel", nrel_eu, 0)
put("rec_plays_median", R.n_plays.median(), 0)
put("rec_straight_frames", fr.nfl_id.isin(R.index).sum(), 0)
put("rec_eu_median", R.EU.median())
put("rec_eu_p10", R.EU.quantile(0.1))
put("rec_eu_p90", R.EU.quantile(0.9))
put("wr_eu_median", R[R.grp == "WR"].EU.median())
put("te_eu_median", R[R.grp == "TE"].EU.median())
for grp in ["DL", "OL", "DB"]:
    put(f"{grp}_eu_median", E[E.grp == grp].EU.median())

# population translation on receivers: Combine A0/S0 vs NFL frontier line
d = R.dropna(subset=["A0_g_z", "S0_g_z"])
rA = rho(d.A0_z.values, d.A0_g_z.values)
rS = rho(d.S0_z.values, d.S0_g_z.values)
put("rec_rA", rA); put("rec_rS", rS); put("rec_n_line", len(d), 0)
ci = boot_ci(d.A0_z.values, d.A0_g_z.values, rho); put("rec_rA_lo", ci[0]); put("rec_rA_hi", ci[1])
ci = boot_ci(d.S0_z.values, d.S0_g_z.values, rho); put("rec_rS_lo", ci[0]); put("rec_rS_hi", ci[1])
# does a big Combine A0 predict EU (does the envelope get used)?
put("rec_rho_A0_EU", rho(R.A0_z.values, R.EU_z.values))
put("rec_rho_forty_EU", rho(R.dropna(subset=["forty"]).forty.values, R.dropna(subset=["forty"]).EU_z.values))

# ----------------------------------------------------------------------------- production beyond pick (receivers)
P = R.dropna(subset=["y", "EU_z", "A0_g_z"]).copy()
put("prod_n", len(P), 0)
put("prod_n_wr", (P.grp == "WR").sum(), 0)
put("prod_n_te", (P.grp == "TE").sum(), 0)
put("prod_routes_median", P.routes.median(), 0)
put("yprr_median", P.y.median())
put("yprr_sd", P.y.std())
BASE = ["lpick", "forty_f", "forty_miss", "combine_weight", "ngs_athleticism_score", "te"]
PICK = ["lpick", "te"]
yv = P.y.values
strat = P.draft_year.values


def fold_stats(df):
    y = df.y.values
    pb = loco(df, BASE)
    pe = loco(df, BASE, "EU_z")
    pp = loco(df, PICK)
    pc = loco(df, BASE, "A0_z")
    pg = loco(df, BASE, "A0_g_z") if df.A0_g_z.notna().all() else pe
    res = y - pb
    slope = np.polyfit(df.EU_z.values, res, 1)[0]
    return dict(r2_base=r2(y, pb), r2_eu=r2(y, pe), r2_pick=r2(y, pp), d_eu=r2(y, pe) - r2(y, pb),
                d_c=r2(y, pc) - r2(y, pb), d_g=r2(y, pg) - r2(y, pb), slope=slope, pb=pb, pe=pe, res=res)


F = fold_stats(P)
bs = []
idx = np.arange(len(P))
for _ in range(NB):
    b = np.concatenate([RNG.choice(idx[strat == s], (strat == s).sum()) for s in np.unique(strat)])
    fb = fold_stats(P.iloc[b])
    bs.append([fb["r2_base"], fb["r2_eu"], fb["d_eu"], fb["slope"], fb["r2_pick"], fb["d_c"], fb["d_g"]])
bs = np.array(bs)
lo, hi = np.percentile(bs, [5, 95], axis=0)
put("r2_pick", F["r2_pick"])
put("r2_base", F["r2_base"]); put("r2_base_lo", lo[0]); put("r2_base_hi", hi[0])
put("r2_eu", F["r2_eu"]); put("r2_eu_lo", lo[1]); put("r2_eu_hi", hi[1])
put("d_eu", F["d_eu"], 3); put("d_eu_lo", lo[2], 3); put("d_eu_hi", hi[2], 3)
put("d_c", F["d_c"], 3); put("d_c_lo", lo[5], 3); put("d_c_hi", hi[5], 3)
put("d_g", F["d_g"], 3); put("d_g_lo", lo[6], 3); put("d_g_hi", hi[6], 3)
put("slope", F["slope"]); put("slope_lo", lo[3]); put("slope_hi", hi[3])
se = bs[:, 3].std()
put("slope_se", se, 3)
# TOST against +/- MARGIN
p_low = 1 - stats.norm.cdf((F["slope"] + MARGIN) / se)
p_high = stats.norm.cdf((F["slope"] - MARGIN) / se)
put("tost_p", max(p_low, p_high))
put("p_inside_margin", np.mean(np.abs(bs[:, 3]) < MARGIN) * 100, 0)
put("p_above_margin", np.mean(bs[:, 3] > MARGIN) * 100, 0)
put("p_benefit", p_high, 3)
put("p_harm", p_low, 3)
put("margin", MARGIN)
put("margin_yards", MARGIN * 500, 0)
# within-class permutation of EU on the held-out slope (same estimand)
perm = []
for _ in range(NB):
    Pp = P.copy()
    Pp["EU_z"] = Pp.groupby("draft_year").EU_z.transform(lambda x: RNG.permutation(x.values))
    perm.append(np.polyfit(Pp.EU_z.values, F["res"], 1)[0])
perm = np.array(perm)
put("perm_p", np.mean(np.abs(perm) >= abs(F["slope"])))
# per class slopes
for yy in [2023, 2024, 2025]:
    m = P.draft_year.values == yy
    put(f"slope_{yy}", np.polyfit(P.EU_z.values[m], F["res"][m], 1)[0])
    put(f"n_{yy}", m.sum(), 0)
# WR only
Pw = P[P.grp == "WR"].copy()
Fw = {"res": Pw.y.values - loco(Pw, [c for c in BASE if c != "te"])}
put("slope_wr", np.polyfit(Pw.EU_z.values, Fw["res"], 1)[0])
# raw association
put("rho_eu_yprr", rho(P.EU_z.values, P.y.values))
put("rho_pick_yprr", rho(P.lpick.values, P.y.values))
# tertiles of EU: held-out miss in yards per route
P["res"] = F["res"]
P["tert"] = pd.qcut(P.EU_z, 3, labels=["low", "mid", "high"])
tt = P.groupby("tert", observed=True).res.agg(["mean", "size"])
for k in tt.index:
    put(f"tert_{k}", tt.loc[k, "mean"]); put(f"tert_{k}_n", tt.loc[k, "size"], 0)

# ----------------------------------------------------------------------------- named receivers
P["fc"] = F["pb"]
board = P.assign(EU_pct=lambda d: d.EU * 100)[
    ["display_name", "grp", "draft_year", "draft_overall_pick", "forty", "A0", "EU", "EU_z", "n_plays", "routes", "y", "fc", "res"]
].sort_values("EU_z", ascending=False)
board.to_csv(OUT / "board.csv", float_format="%.3f")
P[["draft_year", "grp", "y"] + BASE + ["EU_z", "A0_z"]].to_csv(OUT / "inputs.csv", float_format="%.4f")


NAMES = {"Brian Thomas": "Brian Thomas Jr."}
P["name"] = P.display_name.replace(NAMES)

# tiebreaker test: same-class receivers whose no-tracking forecasts sit within TIE yd/route
TIE = 0.15
put("tie_margin", TIE)
pr = []
Pi = P.reset_index()
for yy, d in Pi.groupby("draft_year"):
    d = d.reset_index(drop=True)
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            if abs(d.fc[i] - d.fc[j]) < TIE and d.EU_z[i] != d.EU_z[j]:
                hi_, lo_ = (i, j) if d.EU_z[i] > d.EU_z[j] else (j, i)
                pr.append((d.nfl_id[hi_], d.nfl_id[lo_], d.y[hi_] - d.y[lo_]))
pr = pd.DataFrame(pr, columns=["hi", "lo", "dy"])
win = (pr.dy > 0).mean()
put("tie_pairs", len(pr), 0)
put("tie_players", len(set(pr.hi) | set(pr.lo)), 0)
put("tie_win", win * 100, 0)
put("tie_gain_yards", pr.dy.mean() * 500, 0)
ids = np.array(sorted(set(pr.hi) | set(pr.lo)))
wb, gb = [], []
for _ in range(NB):
    pick = set(RNG.choice(ids, len(ids)))
    sub = pr[pr.hi.isin(pick) & pr.lo.isin(pick)]
    if len(sub):
        wb.append((sub.dy > 0).mean()); gb.append(sub.dy.mean() * 500)
put("tie_win_lo", np.percentile(wb, 5) * 100, 0); put("tie_win_hi", np.percentile(wb, 95) * 100, 0)
put("tie_gain_lo", np.percentile(gb, 5), 0); put("tie_gain_hi", np.percentile(gb, 95), 0)
put("tie_loss_typ", pr.dy.abs().median() * 500, 0)

# named table: three highest and three lowest EU among receivers with 150+ routes
cand_t = P[P.routes >= 150].sort_values("EU_z", ascending=False)
tab = pd.concat([cand_t.head(3), cand_t.tail(3)])
lines = ["| Receiver (class, pick) | Combine 40 | EU | NFL yd/route | Forecast (no tracking) | Gap |", "|---|---|---|---|---|---|"]
for i, (pid, r) in enumerate(tab.iterrows()):
    k = f"tab{i + 1}"
    put(k + "_eu", r.EU * 100, 0); put(k + "_y", r.y); put(k + "_fc", r.fc); put(k + "_gap", r.y - r.fc)
    pk = "UDFA" if pd.isna(r.draft_overall_pick) else f"#{int(r.draft_overall_pick)}"
    ft = "–" if pd.isna(r.forty) else f"{r.forty:.2f}"
    if not pd.isna(r.forty):
        put(k + "_forty", r.forty)
    lines.append(f"| {r['name']} ({r.grp}, {int(r.draft_year)}, {pk}) | {ft} | {N[k + '_eu']}% | {N[k + '_y']:.2f} | "
                 f"{N[k + '_fc']:.2f} | {fmt(N[k + '_gap'])} |")
(OUT / "table2.md").write_text("\n".join(lines) + "\n")


# cards: two high-EU, two low-EU receivers among those with 150+ routes, chosen by rank
cand = P[P.routes >= 150].sort_values("EU_z")
low2 = cand.head(2)
high2 = cand.tail(2).iloc[::-1]
cards = pd.concat([high2, low2])
for i, (pid, r) in enumerate(cards.iterrows()):
    tag = f"card{i + 1}"
    N[f"{tag}_name"] = r.display_name
    put(f"{tag}_eu", r.EU * 100, 0)
    put(f"{tag}_yprr", r.y)
    put(f"{tag}_fc", r.fc)
    put(f"{tag}_forty", r.forty if pd.notna(r.forty) else np.nan)
    put(f"{tag}_routes", r.routes, 0)
    put(f"{tag}_plays", r.n_plays, 0)
    put(f"{tag}_pick", r.draft_overall_pick if pd.notna(r.draft_overall_pick) else 0, 0)

# ----------------------------------------------------------------------------- figures
C_COMB, C_NFL, C_NEU = OI["orange"], OI["blue"], OI["grey"]

# Figure 1: four A-S cards
fig, axs = plt.subplots(1, 4, figsize=(W, 3.9), sharey=True)
for ax, (pid, r) in zip(axs, cards.iterrows()):
    sgrid = np.linspace(1.5, r.S0, 50)
    ax.plot(sgrid, r.A0 * (1 - sgrid / r.S0), color=C_COMB, lw=2.5, label="Combine 40 line")
    q = Q[Q.nfl_id == pid]
    q = q[q.bin <= 0.8 * r.S0]
    raw = fr[fr.nfl_id == pid]
    ax.scatter(raw.s, raw.al, s=3, color=C_NEU, alpha=0.25, lw=0, label="NFL straight-line frames")
    ax.plot(q.bin, q.al, "o-", color=C_NFL, ms=5, lw=1.8, label="NFL hard burst (90th pct)")
    ax.set_title(f"{NAMES.get(r.display_name, r.display_name)}\nEU {r.EU * 100:.0f}% | {r.y:.2f} yd/route", fontsize=11)
    ax.set_xlabel("speed (yd/s)")
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 10)
axs[0].set_ylabel("acceleration (yd/s²)")
axs[0].legend(loc="upper right", fontsize=8, frameon=False)
fig.suptitle("Same code, two settings: Combine 40 sprint line (orange) vs NFL straight-line bursts (blue)",
             fontsize=12, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(FIG / "fig1_cards.png", dpi=150)
plt.close(fig)

# Figure 2: screen across groups
fig, axs = plt.subplots(1, 3, figsize=(W, 4.4), gridspec_kw=dict(width_ratios=[1.3, 0.8, 1.1]))
yl = np.arange(len(screen))[::-1]
for k, (key, ci, col, lab) in enumerate([("rA", "ciA", OI["verm"], "start burst A0"), ("rS", "ciS", OI["sky"], "top-speed end S0")]):
    off = 0.15 if k == 0 else -0.15
    for yy, s in zip(yl, screen):
        axs[0].plot(s[ci], [yy + off] * 2, color=col, lw=2)
        axs[0].plot(s[key], yy + off, "o", color=col, ms=7, label=lab if yy == yl[0] else None)
axs[0].axvline(0, color="k", lw=0.8)
axs[0].set_yticks(yl, [f"{s['grp']} (n={s['n']})" for s in screen])
axs[0].set_xlabel("ρ, Combine 40 vs NFL frontier (90% CI)")
axs[0].set_title("A. Combine 40 vs NFL line", loc="left")
axs[0].set_ylim(-1.3, len(screen) - 0.5)
axs[0].legend(frameon=False, loc="lower right", fontsize=9, ncol=2, bbox_to_anchor=(1.0, -0.02))
axs[1].barh(yl, [s["rel"] for s in screen], color=[OI["blue"] if s["grp"] in ("WR", "TE") else C_NEU for s in screen])
for yy, s in zip(yl, screen):
    axs[1].text(s["rel"] + 0.02, yy, f"{s['rel']:.2f}", va="center", fontsize=10)
axs[1].set_yticks(yl, [s["grp"] for s in screen])
axs[1].set_xlim(0, 1.05)
axs[1].set_xlabel("split-half reliability of EU")
axs[1].set_title("B. EU reliability", loc="left")
prow = [("WR+TE", "REC"), ("DL", "DL"), ("OL", "OL"), ("DB", "DB")]
y4 = np.arange(len(prow))[::-1]
for yy, (lab, key) in zip(y4, prow):
    v = N[f"scr_{key}_dR2"]
    axs[2].barh(yy, v, color=OI["blue"] if key == "REC" else C_NEU)
    axs[2].text(0.001 if v < 0 else v + 0.001, yy, f"{v:+.3f}", va="center", fontsize=10)
axs[2].axvline(0, color="k", lw=0.8)
axs[2].set_yticks(y4, [f"{lab} (n={N[f'scr_{key}_prod_n']})" for lab, key in prow])
axs[2].set_xlabel("held-out ΔR², EU beyond pick + weight")
axs[2].set_title("C. EU adds production?", loc="left")
fig.tight_layout()
fig.savefig(FIG / "fig2_screen.png", dpi=150)
plt.close(fig)

# Figure 3: receivers translation scatter (A) and EU distribution by group (B)
fig, axs = plt.subplots(1, 2, figsize=(W, 4.4))
d = E[(E.grp == "WR")].dropna(subset=["S0_g_z"])
axs[0].scatter(d.S0_z, d.S0_g_z, color=OI["blue"], s=30, alpha=0.8, lw=0)
b = np.polyfit(d.S0_z, d.S0_g_z, 1)
xx = np.linspace(d.S0_z.min(), d.S0_z.max(), 10)
axs[0].plot(xx, np.polyval(b, xx), color="k", lw=1.2)
for nm in ["Brian Thomas", "Rashee Rice", "Ryan Flournoy", "Jaylin Lane"]:
    if nm in set(d.display_name):
        r = d[d.display_name == nm].iloc[0]
        axs[0].annotate(nm.split()[-1], (r.S0_z, r.S0_g_z), fontsize=9, xytext=(4, 4), textcoords="offset points")
axs[0].set_xlabel("Combine 40 top-speed end S0 (sd within class)")
axs[0].set_ylabel("NFL burst-line S0 (sd within class)")
axs[0].set_title(f"A. WR top end carries: ρ = {N['scr_WR_rS']:.2f} (n = {N['scr_WR_n']}); start burst: {N['scr_WR_rA']:.2f}", loc="left", fontsize=11)
order = ["WR", "TE", "DB", "DL", "OL"]
data = [E[E.grp == gname].EU.dropna().values * 100 for gname in order]
bp = axs[1].boxplot(data, orientation="horizontal", tick_labels=order, patch_artist=True, widths=0.6, showfliers=False)
for patch, gname in zip(bp["boxes"], order):
    patch.set_facecolor(OI["sky"] if gname in ("WR", "TE") else "#dddddd")
for i, v in enumerate(data):
    axs[1].scatter(v, np.full(len(v), i + 1) + RNG.uniform(-0.18, 0.18, len(v)), s=7, color="k", alpha=0.35, lw=0)
axs[1].axvline(100, color=C_COMB, ls="--", lw=1.5)
axs[1].text(101, 5.35, "= own Combine 40 line", color=C_COMB, fontsize=9)
axs[1].set_xlabel("Envelope Use (%)")
axs[1].set_title("B. Linemen meet blockers; receivers use most of the envelope", loc="left", fontsize=11)
axs[1].invert_yaxis()
fig.tight_layout()
fig.savefig(FIG / "fig3_translation.png", dpi=150)
plt.close(fig)

# Figure 4: production beyond baseline
fig, axs = plt.subplots(1, 2, figsize=(W, 4.4), gridspec_kw=dict(width_ratios=[1.4, 1]))
cols = np.where(P.grp == "WR", OI["blue"], OI["green"])
axs[0].scatter(P.EU_z, P.res, c=cols, s=30, alpha=0.8, lw=0)
xx = np.linspace(P.EU_z.min(), P.EU_z.max(), 50)
lines = []
for _ in range(400):
    bi = RNG.choice(np.arange(len(P)), len(P))
    lines.append(np.polyval(np.polyfit(P.EU_z.values[bi], F["res"][bi], 1), xx))
lines = np.array(lines)
axs[0].fill_between(xx, np.percentile(lines, 5, 0), np.percentile(lines, 95, 0), color=OI["sky"], alpha=0.35)
axs[0].plot(xx, F["slope"] * xx + np.mean(F["res"]) - F["slope"] * P.EU_z.mean(), color="k", lw=1.5)
axs[0].axhline(0, color=C_NEU, lw=0.8)
for _, r in cards.iterrows():
    axs[0].annotate(r.display_name.split()[-1], (r.EU_z, r.y - r.fc), fontsize=9, xytext=(4, 4), textcoords="offset points")
axs[0].set_xlabel("Envelope Use (sd within position and class)")
axs[0].set_ylabel("yards per route vs held-out forecast")
axs[0].set_title(f"A. Slope {N['slope']:+.2f} yd/route per sd [{N['slope_lo']:+.2f}, {N['slope_hi']:+.2f}], n = {N['prod_n']}", loc="left", fontsize=11)
rows4 = [("All receivers", np.ones(len(P), bool))] + [(f"{yy} class", P.draft_year.values == yy) for yy in (2023, 2024, 2025)] + [
    ("WR only", (P.grp == "WR").values), ("TE only", (P.grp == "TE").values)]
yv4 = np.arange(len(rows4))[::-1]
axs[1].axvspan(-MARGIN, MARGIN, color=OI["yellow"], alpha=0.35, lw=0, label=f"±{MARGIN:.2f} margin")
for yy, (lab, m) in zip(yv4, rows4):
    xs, rs = P.EU_z.values[m], F["res"][m]
    est = np.polyfit(xs, rs, 1)[0]
    ii = np.arange(m.sum())
    bb = [np.polyfit(xs[q], rs[q], 1)[0] for q in (RNG.choice(ii, len(ii)) for _ in range(1000))]
    l_, h_ = np.percentile(bb, [5, 95])
    key = lab.split()[0].lower()
    if key == "all":  # headline row: same estimate and interval as the text (fold-refitting bootstrap)
        est, l_, h_ = N["slope"], N["slope_lo"], N["slope_hi"]
    put(f"f4_{key}_est", est); put(f"f4_{key}_lo", l_); put(f"f4_{key}_hi", h_); put(f"f4_{key}_n", m.sum(), 0)
    axs[1].plot([l_, h_], [yy, yy], color=OI["blue"] if yy == yv4[0] else "k", lw=2)
    axs[1].plot(est, yy, "o", color=OI["blue"] if yy == yv4[0] else "k", ms=7)
axs[1].axvline(0, color="k", lw=0.8)
axs[1].set_yticks(yv4, [f"{lab} (n={m.sum()})" for lab, m in rows4])
axs[1].set_xlabel("yd/route per sd of EU (90% CI)")
axs[1].set_ylim(-0.9, len(rows4) - 0.5)
axs[1].legend(frameon=False, loc="lower left", fontsize=9, bbox_to_anchor=(0.0, -0.03))
axs[0].scatter([], [], color=OI["blue"], label="WR"); axs[0].scatter([], [], color=OI["green"], label="TE")
axs[0].legend(frameon=False, loc="lower left")
axs[1].set_title("B. Same slope by class and position", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig4_production.png", dpi=150)
plt.close(fig)

N["as_printed"] = PRINTED
(OUT / "numbers.json").write_text(json.dumps(N, indent=1, default=float))
print(json.dumps({k: N[k] for k in ["rec_n_eu", "rec_rel_eu", "rec_rA", "rec_rS", "prod_n", "r2_base", "r2_eu", "d_eu", "slope",
                                    "slope_lo", "slope_hi", "tost_p", "perm_p"]}, indent=1))
print(cards[["display_name", "EU", "y", "fc", "routes", "forty"]])
print(board.head(8)[["display_name", "EU", "y", "fc"]]); print(board.tail(8)[["display_name", "EU", "y", "fc"]])
print([(s["grp"], s["n"], round(s["rA"], 2), round(s["rS"], 2), round(s["rel"], 2), s.get("dR2")) for s in screen])
