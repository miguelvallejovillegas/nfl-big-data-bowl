"""Round 25 cand 3: Movement translation. Envelope Match = how closely a player's NFL start-from-stance
speed/acceleration envelope matches his Combine one, and whether that explains production beyond the draft pick.

Run from the repo root:
    uv run python writeup/candidates/round_25/cand_3/analysis.py

Reads data/processed/*.parquet only (frame features cached in data/cache/r25c3, gitignored). Writes numbers.json
(every number quoted in writeup.md), figures/*.png (1600 px, 150 dpi) and renders writeup.md from
writeup_template.md ({{key}} -> numbers.json). Runtime about 3 minutes.

Design, fixed before any production number was computed:
  * Movement type = the first 3.0 s of a start from a stance. Combine side: the position drills that start from a
    stance (DL pass-rush + Run-the-Hoop; OL pass-set drop + three pulls; WR and TE route drills; all seven DB drills),
    from movement onset. NFL side: randomly sampled snaps (game_frames_sample, sample_kind == random), from the
    snap, cut at the throw or sack. Players with 8+ sampled snaps.
  * Envelope = 90th percentile of forward acceleration and of sideways (turning) acceleration in three speed bands
    (1-2.5, 2.5-4, 4-5.5 yd/s): six numbers per side. Combine side z within position group and Combine year, NFL
    side z within position group.
  * Envelope Match = minus the mean absolute gap between his six Combine z and his six NFL z (0 = plays his
    Combine envelope exactly, relative to his peers). Needs 4+ of the 6 cells on both sides.
  * Group choice rule: the group with the highest lower 95% bound on weight-held translation (mean partial
    Spearman of the six cells, Combine vs NFL, holding Combine weight), among groups with 50+ players.
  * Production beyond pick: leave-one-class-out OLS of the group's per-play production stat on log pick + role
    share, then the same + Envelope Match. Slope per sd with a class-stratified 95% player bootstrap that refits
    every fold. Placebo: Match shuffled within class x role.
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as F  # noqa: E402

F.BANDS = [(1, 2.5), (2.5, 4), (4, 5.5)]
ROOT = HERE.parents[3]
PROC = ROOT / "data" / "processed"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
for _f in FIG.glob("*.png"):
    _f.unlink()

NBOOT = int(os.environ.get("BDB_NBOOT", 1000))
NPERM = 2000
UDFA_PICK, MIN_SNAPS = 262, 8
CLASSES = [2023, 2024, 2025]
GROUPS = ["DL", "OL", "WR", "TE", "DB"]
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
      "sky": "#56B4E9", "grey": "#8C8C8C", "black": "#000000", "yellow": "#F0E442"}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 11, "axes.spines.top": False,
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


def yz(s, year):
    return s.groupby(year).transform(lambda v: (v - v.mean()) / v.std())


def pspear(a, b, z=None):
    """Spearman of a, b; partial on z (ranks residualized) if given."""
    R = pd.DataFrame({"a": a, "b": b} | ({} if z is None else {"z": z})).dropna().rank()
    if z is None:
        return float(np.corrcoef(R.a, R.b)[0, 1]), len(R)
    A = np.c_[np.ones(len(R)), R.z]
    ra = R.a - A @ np.linalg.lstsq(A, R.a, rcond=None)[0]
    rb = R.b - A @ np.linalg.lstsq(A, R.b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1]), len(R)


# ============================================================================= load
(DF, DPATH), (GF, GPATH) = F.load()
P = pd.read_parquet(PROC / "players.parquet")
CR = pd.read_parquet(PROC / "combine_results.parquet").drop(columns=["draft_year"])
S = pd.read_parquet(PROC / "player_season_stats.parquet")
AL = pd.read_parquet(PROC / "player_season_alignment.parquet")
GP = pd.read_parquet(PROC / "game_plays_sample.parquet")
P = P.merge(CR[["nfl_id", "combine_weight", "ten_yd_split", "three_cone", "ngs_athleticism_score"]].rename(
    columns={"combine_weight": "cw"}), on="nfl_id", how="left")
P["pick"] = P.draft_overall_pick.fillna(UDFA_PICK)
P["log_pick"] = np.log(P.pick)
P["weight"] = P.combine_weight
ROLE_POS = {"DL": ["EDGE", "OLB"], "OL": ["T"], "WR": ["SLOT_WR"], "TE": ["TE"], "DB": ["CB", "SLOT_CB"]}
al = AL.merge(P[["nfl_id", "position_group"]], on="nfl_id")
al["hit"] = [lp in ROLE_POS[g] for lp, g in zip(al.lined_up_position, al.position_group)]
al["hp"] = al.plays * al.hit
_rs = al.groupby("nfl_id").agg(hp=("hp", "sum"), pl=("plays", "sum"))
P = P.merge((_rs.hp / _rs.pl).rename("role").reset_index(), on="nfl_id", how="left")
P["role"] = P.role.fillna(P.role.median())
put("n_players", len(P))

# ============================================================================= envelopes
NPL = GF.drop_duplicates(["game_id", "play_id", "nfl_id"]).groupby("nfl_id").size()
ED = F.envelope(DF, "nfl_id")
EG = F.envelope(GF, "nfl_id")
# split halves (alternate sampled snaps; alternate Combine attempts across his drills) for the noise ceiling
GF["half"] = GF.groupby("nfl_id").play_id.transform(lambda v: (pd.factorize(v)[0] % 2))
DF["half"] = DF.drill_name.map({d_: i % 2 for g_ in F.DRILLS.values() for i, d_ in enumerate(g_)})
EGh = [F.envelope(GF[GF.half == h], "nfl_id", min_frames=3) for h in (0, 1)]
EDh = [F.envelope(DF[DF.half == h], "nfl_id", min_frames=3) for h in (0, 1)]
put("n_game_frames_snaps", int(len(NPL)))
put("n_sampled_snaps", int(NPL.sum()))
put("n_drill_attempts", int(DF.event_id.nunique()))

E = {}  # per group: player table with z cells
for grp in GROUPS:
    pg = P[(P.position_group == grp)].set_index("nfl_id")
    pg = pg[NPL.reindex(pg.index).fillna(0) >= MIN_SNAPS]
    a, b = ED.reindex(pg.index), EG.reindex(pg.index)
    za = a.groupby(pg.draft_year).transform(lambda v: (v - v.mean()) / v.std())
    zb = (b - b.mean()) / b.std()
    ok = (za.notna() & zb.notna())
    t = pg.copy()
    t["n_cells"] = ok.sum(axis=1)
    t["match"] = -(za - zb).abs().where(ok).mean(axis=1)
    t["drill_env"] = za.where(ok).mean(axis=1)
    t["game_env"] = zb.where(ok).mean(axis=1)
    t.loc[t.n_cells < 4, ["match", "drill_env", "game_env"]] = np.nan
    t["n_snaps"] = NPL.reindex(t.index)
    for c in F.ELEMS:
        t["d_" + c], t["g_" + c] = za[c], zb[c]
    E[grp] = t


def translation(t, wt=True, ix=None):
    t = t if ix is None else t.iloc[ix]
    rs = [pspear(t["d_" + c], t["g_" + c], t.weight if wt else None)[0] for c in F.ELEMS]
    return float(np.nanmean(rs))


def split_rel(Eh, ids, yr=None):
    rs = []
    for c in F.ELEMS:
        u, v = Eh[0][c].reindex(ids), Eh[1][c].reindex(ids)
        if yr is not None:
            u, v = yz(u, yr), yz(v, yr)
        r = pspear(u, v)[0]
        rs.append(2 * r / (1 + r))
    return float(np.nanmean(rs))


rng = np.random.default_rng(25)
SCREEN = []
for grp in GROUPS:
    t = E[grp]
    tm = t[t.match.notna()]
    cl = [np.flatnonzero(tm.draft_year.values == c) for c in CLASSES]
    bw = [translation(tm, True, np.concatenate([rng.choice(g_, len(g_)) for g_ in cl])) for _ in range(300)]
    bp = [translation(tm, False, np.concatenate([rng.choice(g_, len(g_)) for g_ in cl])) for _ in range(300)]
    rel_g = split_rel(EGh, tm.index)
    rel_d = split_rel(EDh, tm.index, tm.draft_year)
    SCREEN.append(dict(group=grp, n=len(tm), tr_pool=translation(tm, False), tr_pool_lo=np.percentile(bp, 2.5),
                       tr_pool_hi=np.percentile(bp, 97.5), tr_wt=translation(tm, True),
                       tr_lo=np.percentile(bw, 2.5), tr_hi=np.percentile(bw, 97.5), rel_g=rel_g, rel_d=rel_d,
                       ceil=np.sqrt(max(rel_g, 0) * max(rel_d, 0))))
SCREEN = pd.DataFrame(SCREEN).set_index("group")
print(SCREEN.round(2).to_string())
eligible = SCREEN[SCREEN.n >= 50]
CHOSEN = eligible.tr_lo.idxmax()
print("chosen", CHOSEN)


# ============================================================================= production outcomes (per play, shrunk)
def eb_rate(k, n, min_n):
    m = n >= min_n
    mu = k[m].sum() / n[m].sum()
    N = n[m].sum()
    tau2 = ((n[m] * (k[m] / n[m] - mu) ** 2).sum() - (m.sum() - 1) * mu * (1 - mu)) / (N - (n[m] ** 2).sum() / N)
    M = mu * (1 - mu) / max(tau2, 1e-6) - 1
    return (k + M * mu) / (n + M), mu, M


def outcome(grp):
    s = S[S.position_group == grp]
    if grp == "DL":
        a = s.groupby("nfl_id")[["pressures", "pass_rush_snaps"]].sum()
        r, mu, M = eb_rate(a.pressures, a.pass_rush_snaps, 30)
        return (100 * r).where(a.pass_rush_snaps >= 30), "Pressure rate (%)"
    if grp == "OL":
        a = s.groupby("nfl_id")[["pressures_allowed", "pass_blocks"]].sum()
        r, mu, M = eb_rate(a.pressures_allowed, a.pass_blocks, 30)
        return (-100 * r).where(a.pass_blocks >= 30), "Pressures allowed (%, sign flipped)"
    if grp in ("WR", "TE"):
        s = s.assign(y=s.yards_per_route * s.routes)
        a = s.groupby("nfl_id")[["y", "routes"]].sum()
        mu = a.y.sum() / a.routes.sum()
        return ((a.y + 100 * mu) / (a.routes + 100)).where(a.routes >= 50), "Yards per route"
    s = s.assign(e=s.epa_per_play * s.plays)
    a = s.groupby("nfl_id")[["e", "plays"]].sum()
    return (-(a.e) / (a.plays + 200)).where(a.plays >= 100), "EPA per play allowed (sign flipped)"


def loco(d, cols, y="y"):
    pred = np.full(len(d), np.nan)
    cl = d.draft_year.values
    for c in CLASSES:
        tr, te = d[cl != c], d[cl == c]
        A = np.c_[np.ones(len(tr)), tr[cols].values]
        b = np.linalg.lstsq(A, tr[y].values, rcond=None)[0]
        pred[cl == c] = np.c_[np.ones(len(te)), te[cols].values] @ b
    return pred


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def slope_sd(d, x, base):
    """held-out slope of x (per sd) on the residual of the held-out base forecast, the R2 gain."""
    d = d.copy()
    d["xz"] = (d[x] - d[x].mean()) / d[x].std()
    p0, p1 = loco(d, base), loco(d, base + ["xz"])
    res = d.y.values - p0
    sl = np.polyfit(d.xz.values, res, 1)[0]
    return float(sl), float(r2(d.y.values, p1) - r2(d.y.values, p0)), float(r2(d.y.values, p0)), \
        float(r2(d.y.values, p1))


def boot_slope(d, x, base, seed, nb=NBOOT):
    rg = np.random.default_rng(seed)
    cl = [np.flatnonzero(d.draft_year.values == c) for c in CLASSES]
    out = []
    for _ in range(nb):
        ix = np.concatenate([rg.choice(g_, len(g_)) for g_ in cl])
        out.append(slope_sd(d.iloc[ix].reset_index(drop=True), x, base)[:2])
    return np.asarray(out)


BASE = ["log_pick", "role"]
PROD = {}
for grp in GROUPS:
    t = E[grp]
    y, lab = outcome(grp)
    t["y"] = y.reindex(t.index)
    d = t.dropna(subset=["y", "match"]).reset_index()
    sl, g2, r0, r1 = slope_sd(d, "match", BASE)
    b = boot_slope(d, "match", BASE, 100 + GROUPS.index(grp), 400)
    PROD[grp] = dict(n=len(d), slope=sl, lo=np.percentile(b[:, 0], 2.5), hi=np.percentile(b[:, 0], 97.5),
                     sd_y=d.y.std(), std=sl / d.y.std(), stdlo=np.percentile(b[:, 0], 2.5) / d.y.std(),
                     stdhi=np.percentile(b[:, 0], 97.5) / d.y.std(), r0=r0, r1=r1, lab=lab)
PRODF = pd.DataFrame(PROD).T
print(PRODF.to_string())
for grp in GROUPS:
    r_ = SCREEN.loc[grp]
    put(f"s_{grp}_n", int(r_.n)); put(f"s_{grp}_pool", r_.tr_pool); put(f"s_{grp}_wt", r_.tr_wt)
    put(f"s_{grp}_lo", r_.tr_lo); put(f"s_{grp}_hi", r_.tr_hi); put(f"s_{grp}_ceil", r_.ceil)
    put(f"s_{grp}_relg", r_.rel_g); put(f"s_{grp}_reld", r_.rel_d)
    q_ = PRODF.loc[grp]
    put(f"p_{grp}_n", int(q_.n)); put(f"p_{grp}_std", q_["std"]); put(f"p_{grp}_lo", q_.stdlo)
    put(f"p_{grp}_hi", q_.stdhi)
NUM["chosen"] = CHOSEN


# ============================================================================= DL deep dive
T = E["DL"]
D = T.dropna(subset=["y", "match"]).reset_index()
put("dl_n", len(D)); put("dl_snaps_median", D.n_snaps.median(), 0)
put("dl_n_env", int(T.game_env.notna().sum()))
NUM["dl_n_per_class"] = "/".join(str(int((D.draft_year == c).sum())) for c in CLASSES)
_y_all = outcome("DL")[0]
put("press_mean", D.y.mean(), 1)

# what carries: weight vs Combine envelope as predictors of the NFL envelope (all DL with both sides)
TT = T.dropna(subset=["drill_env", "game_env", "weight"])
rng = np.random.default_rng(7)


def bci(f, d, nb=NBOOT, seed=7):
    rg = np.random.default_rng(seed)
    cl = [np.flatnonzero(d.draft_year.values == c) for c in CLASSES]
    b = [f(d.iloc[np.concatenate([rg.choice(g_, len(g_)) for g_ in cl])]) for _ in range(nb)]
    return np.percentile(b, [2.5, 97.5])


for key, f in [("w_game", lambda d: pspear(d.weight, d.game_env)[0]),
               ("d_game", lambda d: pspear(d.drill_env, d.game_env)[0]),
               ("d_game_w", lambda d: pspear(d.drill_env, d.game_env, d.weight)[0]),
               ("w_drill", lambda d: pspear(d.weight, d.drill_env)[0])]:
    put(f"rho_{key}", f(TT)); lo_, hi_ = bci(f, TT); put(f"rho_{key}_lo", lo_); put(f"rho_{key}_hi", hi_)
put("rho_n", len(TT))
# cell by cell, weight held (the six translations behind the 0.05)
for c in F.ELEMS:
    put(f"cell_{c}", pspear(T["d_" + c], T["g_" + c], T.weight)[0])
put("match_w", pspear(D.match, D.weight)[0]); put("match_role", pspear(D.match, D.role)[0])
put("match_snaps", pspear(D.match, D.n_snaps)[0])

# production: Match, Combine envelope, NFL envelope, each beyond pick + role (held-out), and beyond full baseline
FULL = ["log_pick", "role", "weight", "ngs_athleticism_score"]
D["ngs_athleticism_score"] = D.ngs_athleticism_score.fillna(D.ngs_athleticism_score.median())
D["weight"] = D.weight.fillna(D.weight.median())
RES = {}
D["surplus"] = D.game_env - D.drill_env      # signed: NFL envelope above (+) or below (-) his Combine one
for x, seed in [("match", 31), ("drill_env", 32), ("game_env", 33), ("surplus", 34)]:
    sl, g2, r0, r1 = slope_sd(D, x, BASE)
    b = boot_slope(D, x, BASE, seed)
    RES[x] = (sl, b)
    put(f"h_{x}", sl); put(f"h_{x}_lo", np.percentile(b[:, 0], 2.5)); put(f"h_{x}_hi", np.percentile(b[:, 0], 97.5))
    put(f"h_{x}_season", sl * 4.5, 1); put(f"h_{x}_season_lo", np.percentile(b[:, 0], 2.5) * 4.5, 1)
    put(f"h_{x}_season_hi", np.percentile(b[:, 0], 97.5) * 4.5, 1)
    put(f"h_{x}_r2gain", g2, 3); put(f"h_{x}_r2gain_lo", np.percentile(b[:, 1], 2.5), 3)
    put(f"h_{x}_r2gain_hi", np.percentile(b[:, 1], 97.5), 3)
    put(f"h_{x}_r1", r1)
    slf, g2f, r0f, r1f = slope_sd(D, x, FULL)
    bf = boot_slope(D, x, FULL, seed + 10, 400)
    put(f"f_{x}", slf); put(f"f_{x}_lo", np.percentile(bf[:, 0], 2.5)); put(f"f_{x}_hi", np.percentile(bf[:, 0], 97.5))
    put(f"f_{x}_r1", r1f)
put("h_r0", r0); put("f_r0", r0f)
MARGIN = 0.5
put("margin", MARGIN, 1); put("margin_season", MARGIN * 4.5, 2)
put("p_match_above_margin", 100 * (RES["match"][1][:, 0] >= MARGIN).mean(), 1)
# pick-only comparator for Match
sl_p1 = slope_sd(D, "match", ["log_pick"])
put("h_match_pick1", sl_p1[0])


# placebo: Match shuffled within class x role-half, 2000 shuffles
def placebo(d, x, seed):
    rg = np.random.default_rng(seed)
    cell = d.draft_year.astype(str) + (d.role >= 0.5).astype(str)
    p0 = loco(d, BASE)
    res = d.y.values - p0
    obs = np.polyfit((d[x] - d[x].mean()) / d[x].std(), res, 1)[0]
    ix = [np.flatnonzero(cell.values == c) for c in cell.unique()]
    null = []
    xv = ((d[x] - d[x].mean()) / d[x].std()).values
    for _ in range(NPERM):
        xs = xv.copy()
        for g_ in ix:
            xs[g_] = rg.permutation(xs[g_])
        null.append(np.polyfit(xs, res, 1)[0])
    null = np.asarray(null)
    return obs, null, float((1 + (null >= obs).sum()) / (NPERM + 1))


PL = {x: placebo(D, x, 41 + i) for i, x in enumerate(["match", "drill_env", "game_env"])}
for x in PL:
    put(f"perm_{x}", PL[x][2], 3)
put("perm_band_lo", np.percentile(PL["match"][1], 2.5)); put("perm_band_hi", np.percentile(PL["match"][1], 97.5))
put("nperm", NPERM)

# team-cluster bootstrap of the NFL-envelope R2 gain (latest club; rushers on one club share scheme and coaches)
D["team_cl"] = D.latest_team.fillna("FA_" + D.nfl_id.astype(str))
TEAMS = D.team_cl.unique()
TIX = {t_: np.flatnonzero(D.team_cl.values == t_) for t_ in TEAMS}
rg = np.random.default_rng(55)
CB = {"match": [], "game_env": []}
for _ in range(400):
    ix = np.concatenate([TIX[t_] for t_ in rg.choice(TEAMS, len(TEAMS))])
    dd = D.iloc[ix].reset_index(drop=True)
    if dd.draft_year.nunique() < 3:
        continue
    for x in CB:
        CB[x].append(slope_sd(dd, x, BASE)[1])
for x in CB:
    put(f"cb_{x}_lo", np.percentile(CB[x], 2.5), 3); put(f"cb_{x}_hi", np.percentile(CB[x], 97.5), 3)
    put(f"cb_{x}_pos", 100 * (np.asarray(CB[x]) > 0).mean(), 0)
put("n_teams", len(TEAMS))

# ---- next season: NFL envelope from 2023-24 sampled rushes -> 2025 pressure rate, beyond pick + role + prior rate
G24 = GF[(GF.position_group == "DL") & (GF.season <= 2024)]
n24 = G24.drop_duplicates(["game_id", "play_id", "nfl_id"]).groupby("nfl_id").size()
E24 = F.envelope(G24, "nfl_id")
E24 = E24[n24.reindex(E24.index).fillna(0) >= 6]
z24 = (E24 - E24.mean()) / E24.std()
ok24 = z24.notna().sum(axis=1)
env24 = z24.mean(axis=1).where(ok24 >= 4)
s = S[S.position_group == "DL"]
a25 = s[s.season == 2025].groupby("nfl_id")[["pressures", "pass_rush_snaps"]].sum()
a24 = s[s.season <= 2024].groupby("nfl_id")[["pressures", "pass_rush_snaps"]].sum()
r25 = eb_rate(a25.pressures, a25.pass_rush_snaps, 30)[0].where(a25.pass_rush_snaps >= 30) * 100
r24 = eb_rate(a24.pressures, a24.pass_rush_snaps, 30)[0].where(a24.pass_rush_snaps >= 30) * 100
NX = P[P.position_group == "DL"].set_index("nfl_id")[["log_pick", "role", "draft_year", "weight"]].copy()
NX["env24"], NX["y25"], NX["y24"] = env24, r25, r24
NX = NX.dropna(subset=["env24", "y25", "y24"])
NX["ez"] = (NX.env24 - NX.env24.mean()) / NX.env24.std()


def nx_fit(d):
    A = np.c_[np.ones(len(d)), d[["log_pick", "role", "y24", "ez"]].values]
    return np.linalg.lstsq(A, d.y25.values, rcond=None)[0][-1]


rg = np.random.default_rng(66)
bnx = [nx_fit(NX.iloc[rg.integers(0, len(NX), len(NX))]) for _ in range(NBOOT)]
put("nx_n", len(NX)); put("nx_slope", nx_fit(NX)); put("nx_lo", np.percentile(bnx, 2.5))
put("nx_hi", np.percentile(bnx, 97.5)); put("nx_prior_rho", pspear(NX.y24, NX.y25)[0])
put("nx_season", nx_fit(NX) * 4.5, 1); put("nx_season_lo", np.percentile(bnx, 2.5) * 4.5, 1)
put("nx_season_hi", np.percentile(bnx, 97.5) * 4.5, 1)
put("min_snaps", MIN_SNAPS)

# EPA of a pressured snap (random sampled DL snaps), to price the units
K = ["game_id", "play_id", "nfl_id"]
_g = GP[GP.nfl_id.isin(P.loc[P.position_group == "DL", "nfl_id"]) & (GP.sample_kind == "random")].drop_duplicates(
    ["game_id", "play_id"])
_g = _g[_g.expected_points_added.notna()]
_pr = _g.time_to_pressure.notna() | (_g.sack == 1)
e1, e0 = _g[_pr].expected_points_added.values, _g[~_pr].expected_points_added.values
rg = np.random.default_rng(101)
_eb = [rg.choice(e0, len(e0)).mean() - rg.choice(e1, len(e1)).mean() for _ in range(NBOOT)]
put("epa_press", e0.mean() - e1.mean()); put("epa_lo", np.percentile(_eb, 2.5)); put("epa_hi", np.percentile(_eb, 97.5))
put("epa_n", len(_g))
put("env_epa_season", float(NUM["h_game_env_season"]) * (e0.mean() - e1.mean()), 1)

# ---- held-out residual (pressure beyond pick + role) for named players
D["fc"] = loco(D, BASE)
D["beyond"] = D.y - D.fc
D["mz"] = (D.match - D.match.mean()) / D.match.std()
D["gz"] = (D.game_env - D.game_env.mean()) / D.game_env.std()
D["dz"] = (D.drill_env - D.drill_env.mean()) / D.drill_env.std()
RUSH = S[S.position_group == "DL"].groupby("nfl_id").pass_rush_snaps.sum()
D["rushes"] = RUSH.reindex(D.nfl_id).values

D.to_pickle(F.CACHE / "D.pkl")


# ============================================================================= named players (fixed rules)
def nm(name):
    return D[D.display_name == name].iloc[0]


BIG = D[D.rushes >= 200]
put("n_big", len(BIG))
put("verse_rank", int((BIG.mz < BIG.loc[BIG.display_name == "Jared Verse", "mz"].iloc[0]).sum() + 1))
put("anderson_rank", int((D.mz < D.loc[D.display_name == "Will Anderson", "mz"].iloc[0]).sum() + 1))
put("nboot", NBOOT); put("sd_y_season", D.y.std() * 4.5, 0)
put("margin_epa", MARGIN * 4.5 * float(NUM["epa_press"]), 1)
LOW3 = BIG.nsmallest(3, "mz")
HIGH3 = BIG.nlargest(3, "mz")
put("big_lowmatch_beyond", LOW3.beyond.mean() * 4.5, 1); put("big_highmatch_beyond", HIGH3.beyond.mean() * 4.5, 1)
# same-weight pair for Figure 2: the two edges within 3 lb whose Combine envelopes differ most
_e = D[(D.role >= 0.5)].copy()
best = None
for i, r_ in _e.iterrows():
    for j, q_ in _e.iterrows():
        if i < j and abs(r_.weight - q_.weight) <= 3 and r_.rushes >= 500 and q_.rushes >= 500:
            gap = abs(r_.dz - q_.dz)
            if best is None or gap > best[0]:
                best = (gap, r_.display_name, q_.display_name)
PAIR = sorted(best[1:], key=lambda n_: nm(n_).dz)  # [lower Combine envelope, higher]
print("pair", PAIR)
TAB2 = pd.concat([LOW3, HIGH3])
for k_, r_ in enumerate(TAB2.itertuples()):
    put(f"t2_{k_}_dz", r_.dz, 1); put(f"t2_{k_}_gz", r_.gz, 1); put(f"t2_{k_}_mz", r_.mz, 1)
    put(f"t2_{k_}_beyond", r_.beyond * 4.5, 1); put(f"t2_{k_}_rush", int(r_.rushes)); put(f"t2_{k_}_wt", int(r_.weight))
for k_, n_ in enumerate(PAIR):
    r_ = nm(n_)
    put(f"pair{k_}_wt", int(r_.weight)); put(f"pair{k_}_dz", r_.dz, 1); put(f"pair{k_}_gz", r_.gz, 1)
    put(f"pair{k_}_y", r_.y, 1); put(f"pair{k_}_pick", int(r_.pick))
TXT = {"t2": [dict(name=r_.display_name, team=r_.latest_team, cls=int(r_.draft_year),
                   pick=("UDFA" if r_.pick >= UDFA_PICK else f"#{int(r_.pick)}")) for r_ in TAB2.itertuples()],
       "pair": PAIR}

# ============================================================================= figures
C_E, C_I = OI["blue"], OI["orange"]


def resid_rank(a, z):
    R = pd.DataFrame({"a": a, "z": z}).rank()
    A = np.c_[np.ones(len(R)), R.z]
    return (R.a - A @ np.linalg.lstsq(A, R.a, rcond=None)[0]) / len(R)


def fm1(v):
    t_ = f"{v:+.1f}"
    return t_.replace("-", "−")


def label(ax, x, y, txt, dx=6, dy=6, **kw):
    ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=11,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7), **kw)


# ---- Figure 1: a scale predicts the NFL envelope; the drill adds nothing once weight is known
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.6))
ed = TT.role >= 0.5
ax = axs[0]
ax.scatter(TT.weight[ed], TT.game_env[ed], s=42, c=C_E, edgecolor="white", lw=0.5, label="Edge (half+ snaps outside)")
ax.scatter(TT.weight[~ed], TT.game_env[~ed], s=42, c=C_I, marker="s", edgecolor="white", lw=0.5, label="Interior")
ax.set_xlabel("Combine weight (lb)")
ax.set_ylabel("NFL start envelope (sd vs DL peers)")
ax.set_title(f"A. Weight alone: ρ = {NUM['rho_w_game'].replace('-', '−')}", loc="left")
ax.legend(frameon=False, loc="lower left")
for n_, (dx_, dy_) in zip(PAIR + ["T'Vondre Sweat"], [(30, -40), (40, 14), (-40, 22)]):
    r_ = TT[TT.display_name == n_].iloc[0]
    label(ax, r_.weight, r_.game_env, n_, dx_, dy_)
ax = axs[1]
rx, ry = resid_rank(TT.drill_env, TT.weight), resid_rank(TT.game_env, TT.weight)
ax.scatter(rx[ed.values], ry[ed.values], s=42, c=C_E, edgecolor="white", lw=0.5)
ax.scatter(rx[~ed.values], ry[~ed.values], s=42, c=C_I, marker="s", edgecolor="white", lw=0.5)
ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
ax.set_xlabel("Combine envelope, rank beyond weight")
ax.set_ylabel("NFL envelope, rank beyond weight")
ax.set_title(f"B. Weight held fixed: ρ = {NUM['rho_d_game_w'].replace('-', '−')}", loc="left")
for n_ in PAIR:
    i_ = np.flatnonzero(TT.display_name.values == n_)[0]
    label(ax, rx.iloc[i_], ry.iloc[i_], n_, 8, 10)
fig.suptitle(f"Defensive linemen, first 3 s from a stance (n = {NUM['rho_n']}): weight carries to Sunday, "
             "drill movement does not", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
save(fig, "fig1_weight_not_drill.png")

# ---- Figure 2: the envelopes themselves, the same-weight pair vs weight thirds
BINS = np.arange(1.0, 5.51, 0.5)


def curve(df, col):
    df = df.assign(b=pd.cut(df.v, BINS))
    return df.groupby("b", observed=False)[col].quantile(0.9).values


mid = (BINS[:-1] + BINS[1:]) / 2
dl_ids = set(D.nfl_id)
DFd, GFd = DF[DF.nfl_id.isin(dl_ids)], GF[GF.nfl_id.isin(dl_ids)]
wt = D.set_index("nfl_id").weight
lo_ids = set(wt[wt <= wt.quantile(1 / 3)].index)
hi_ids = set(wt[wt >= wt.quantile(2 / 3)].index)
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.4))
for ax, col, yl in [(axs[0], "al", "Forward acceleration, 90th pct (yd/s²)"), (axs[1], "la", "Sideways (turning) g, 90th pct")]:
    for ids, c_, lab in [(lo_ids, OI["sky"], f"Lightest third (≤{int(wt.quantile(1/3))} lb)"),
                         (hi_ids, OI["grey"], f"Heaviest third (≥{int(wt.quantile(2/3))} lb)")]:
        ax.plot(mid, curve(DFd[DFd.nfl_id.isin(ids)], col), ls="--", color=c_, lw=3)
        ax.plot(mid, curve(GFd[GFd.nfl_id.isin(ids)], col), ls="-", color=c_, lw=3, label=lab)
    for n_, c_ in zip(PAIR if col == "al" else [], [OI["red"], OI["green"]]):
        i_ = nm(n_).nfl_id
        ax.plot(mid, curve(DFd[DFd.nfl_id == i_], col), ls="--", color=c_, lw=2, marker="o", ms=4)
        ax.plot(mid, curve(GFd[GFd.nfl_id == i_], col), ls="-", color=c_, lw=2, marker="o", ms=4,
                label=f"{n_} ({int(nm(n_).weight)} lb)")
    ax.set_xlabel("Speed (yd/s)")
    ax.set_ylabel(yl)
axs[0].set_title("A. Pushing off: dashed = Combine drills, solid = NFL", loc="left")
axs[1].set_title("B. Turning (weight thirds only)", loc="left")
axs[0].legend(frameon=False, loc="lower left", fontsize=10)
fig.suptitle(f"Light men push off harder than heavy men in both venues. At {NUM['pair0_wt']}–{NUM['pair1_wt']} lb, "
             f"{PAIR[0].split()[-1]} tested below {PAIR[1].split()[-1]}\nbut pushes off harder on Sundays: "
             "the order inside a weight class does not carry", x=0.01, ha="left", fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.90))
save(fig, "fig2_envelopes.png")

# ---- Figure 3: five-group screen
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})
ypos = np.arange(len(GROUPS))[::-1]
ax = axs[0]
for yv, grp in zip(ypos, GROUPS):
    r_ = SCREEN.loc[grp]
    ax.errorbar(r_.tr_pool, yv + 0.15, xerr=[[r_.tr_pool - r_.tr_pool_lo], [r_.tr_pool_hi - r_.tr_pool]], fmt="o",
                mfc="white", color=OI["grey"], ms=7, capsize=3, label="Raw" if grp == "DL" else None)
    ax.errorbar(r_.tr_wt, yv - 0.15, xerr=[[r_.tr_wt - r_.tr_lo], [r_.tr_hi - r_.tr_wt]], fmt="o",
                color=OI["blue"], ms=7, capsize=3, label="Weight held fixed" if grp == "DL" else None)
    if r_.rel_g > 0 and r_.rel_d > 0:
        ax.plot(r_.ceil, yv, marker="|", ms=18, mew=3, color=OI["red"], ls="none",
                label="Noise ceiling" if grp == "DL" else None)
ax.set_yticks(ypos, [f"{g_} (n = {int(SCREEN.loc[g_].n)})" for g_ in GROUPS])
ax.axvline(0, color="0.6", lw=0.8)
ax.set_xlabel("Combine → NFL envelope rank correlation (95%)")
ax.set_title("A. Does the drill envelope show up on Sunday?", loc="left")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10)
ax = axs[1]
for yv, grp in zip(ypos, GROUPS):
    q_ = PRODF.loc[grp]
    ax.errorbar(q_["std"], yv, xerr=[[q_["std"] - q_.stdlo], [q_.stdhi - q_["std"]]], fmt="D",
                color=OI["black"] if grp == CHOSEN else OI["grey"], ms=7, capsize=3)
    ax.text(0.98, yv + 0.22, f"n = {int(q_.n)}", transform=ax.get_yaxis_transform(), ha="right", fontsize=10,
            color="0.35")
ax.set_yticks(ypos, ["DL pressure", "OL pressure allowed", "WR yards/route", "TE yards/route", "DB EPA allowed"])
ax.axvline(0, color="0.6", lw=0.8)
ax.set_xlabel("Envelope Match → production beyond\npick + role (outcome sd per sd, 95%)")
ax.set_title("B. Does matching pay?", loc="left")
fig.suptitle("No position group carries its drill envelope to Sunday beyond body mass, and Match buys no production",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "fig3_screen.png")

# ---- Figure 4: DL pressure beyond pick + role vs Match and vs NFL envelope
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.4), sharey=True)
names = list(TAB2.display_name)
for ax, x, xl, key in [(axs[0], "mz", "Envelope Match (sd; right = plays like he tested)", "match"),
                       (axs[1], "gz", "NFL start envelope, same seasons (sd)", "game_env")]:
    e_ = D.role >= 0.5
    ax.scatter(D[x][e_], 4.5 * D.beyond[e_], s=40, c=C_E, edgecolor="white", lw=0.5, label="Edge")
    ax.scatter(D[x][~e_], 4.5 * D.beyond[~e_], s=40, c=C_I, marker="s", edgecolor="white", lw=0.5, label="Interior")
    sl_ = RES[key][0] * 4.5
    xx = np.linspace(D[x].min(), D[x].max(), 10)
    ax.plot(xx, sl_ * xx, color=OI["black"], lw=2)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel(xl)
    lo_, hi_ = NUM[f"h_{key}_season_lo"], NUM[f"h_{key}_season_hi"]
    tx_, ty_, va_ = (0.02, 0.97, "top") if key == "match" else (0.98, 0.03, "bottom")
    ax.text(tx_, ty_, f"{fm1(sl_)} pressures per 450 rushes per sd\n95%: [{lo_}, {hi_}], shuffle p = "
            f"{NUM['perm_' + key]}", transform=ax.transAxes, va=va_, ha="left" if key == "match" else "right", fontsize=11,
            bbox=dict(fc="white", ec="0.8"))
    OFF = {"mz": {"Will Anderson": (8, 12), "Jared Verse": (8, 12), "Jalyx Hunt": (8, 10), "Eric Watts": (8, -16)},
           "gz": {"Will Anderson": (-120, 22), "Jared Verse": (10, 22), "Jalyx Hunt": (-90, -34),
                  "Eric Watts": (-100, -22)}}[x]
    for n_, (dx_, dy_) in OFF.items():
        r_ = D[D.display_name == n_].iloc[0]
        label(ax, r_[x], 4.5 * r_.beyond, n_, dx_, dy_)
axs[0].set_ylabel("Pressures per 450 rushes beyond\nheld-out pick + role forecast")
axs[0].set_title("A. Matching his Combine: no gain", loc="left")
axs[1].set_title("B. Sunday envelope: same-season link only", loc="left")
axs[1].legend(frameon=False, loc="lower left")
fig.suptitle(f"Defensive linemen (n = {NUM['dl_n']}, each class scored by a model fitted on the other two)",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig4_pressure.png")

# ============================================================================= numbers + text
(HERE / "numbers.json").write_text(json.dumps(NUM, indent=1))
(HERE / "text_bits.json").write_text(json.dumps(TXT, indent=1))


def render(num, txt):
    tpl = HERE / "writeup_template.md"
    if not tpl.exists():
        return

    def sub(m_):
        k_ = m_.group(1)
        if k_.startswith("T:"):
            path = k_[2:].split(".")
            v_ = txt
            for p_ in path:
                v_ = v_[int(p_)] if p_.isdigit() else v_[p_]
            return str(v_)
        signed = k_.startswith("+:")
        v_ = str(num[k_[2:] if signed else k_])
        if signed and not v_.startswith("-") and float(v_) != 0:
            v_ = "+" + v_
        return v_.replace("-", "−") if v_.startswith("-") else v_
    (HERE / "writeup.md").write_text(re.sub(r"\{\{([^}]+)\}\}", sub, tpl.read_text()))


render(NUM, TXT)
print("done", len(NUM), "numbers")
