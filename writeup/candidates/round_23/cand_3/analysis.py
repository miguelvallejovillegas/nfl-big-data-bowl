"""Round 23, cand_3 - movement translation: does a player's Combine start from a stance show up in his NFL starts,
and does that translation explain production beyond draft pick?

Steps
1. Screen (all five groups). For each group, every Combine drill that opens with the group's NFL start movement
   (OL kick-slide, DL get-off, WR/TE release, DB backpedal) gives a speed + acceleration profile for the first 1.5 s
   after the player starts moving (0.5 yd/s crossing). The same profile is built on his random NFL snaps.
   Each profile point is read against peers (Combine: drill x year x rep; NFL: alignment, plus route for receivers).
   Release Match (Set Match for linemen) = minus the mean absolute gap between the two profiles, in sd.
   Choice rule: among groups whose two sides are reliable enough to see translation (noise ceiling >= 0.5)
   and that have an individual production measure, study the one with the highest lower 90% bound on
   drill -> NFL translation.
2. Deep dive (the chosen group): translation and no-tracking benchmarks, play-level meaning, three no-tracking
   baselines side by side with leave-one-class-out forecasts, a player-level bootstrap that refits every forecast
   in every draw, a measurement-error simulation, and player cards.

Run from the repo root (about three minutes):
    uv run python writeup/candidates/round_23/cand_3/analysis.py
Writes numbers.json, inputs.csv and figures/*.png (1600 px wide, 150 dpi) next to this file.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
FIGW = 1600 / 150
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2", "verm": "#D55E00",
      "purple": "#CC79A7", "grey": "#555555", "light": "#BBBBBB"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
RNG = np.random.default_rng(2333)
NBOOT = 2000
THR = 0.5                      # movement onset, yd/s
NPT = 16                       # 0.0 .. 1.5 s after onset
LV = slice(3, 13)              # 0.3 .. 1.2 s: the "release speed" window
CEIL_MIN = 0.5                 # choice rule: both sides reliable enough
MARGIN = 0.15                  # practical margin, yards per route run per sd (about 12% of the WR mean)
MIN_DEN = 100
NUM: dict = {}


def put(key, val, fmt="{:.2f}"):
    s = fmt.format(val) if not isinstance(val, str) else val
    if s.startswith("-") and float(s.replace(",", "")) == 0:
        s = s[1:]
    NUM[key] = s
    return s


def onset(s):
    hit = np.flatnonzero(s >= THR)
    return int(hit[0]) if len(hit) else None


# ------------------------------------------------------------------------------------------------- data
pl = pd.read_parquet(P / "players.parquet").set_index("nfl_id")
cr = pd.read_parquet(P / "combine_results.parquet").drop_duplicates("nfl_id").set_index("nfl_id")
ev = pd.read_parquet(P / "combine_events.parquet")
cf = pd.read_parquet(P / "combine_frames.parquet", columns=["event_id", "frame", "entity_type", "s", "a"])
cf = cf[cf.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
gp = pd.read_parquet(P / "game_plays_sample.parquet")
gf = pd.read_parquet(P / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "s", "a"])
st = pd.read_parquet(P / "player_season_stats.parquet")
pl["lpick"] = np.log(pl.draft_overall_pick.fillna(260))

DRILLS = {"OL": ["PASS_RUSH_DROP"], "DL": ["PASS_RUSH_DRILL"],
          "WR": ["GO_ROUTE_RIGHT", "SLANT_ROUTE_LEFT", "COMEBACK_ROUTE_RIGHT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT",
                 "POST_CORNER_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT", "SPEED_OUT_ROUTE_LEFT", "SLOT_SAIL_ROUTE_RIGHT",
                 "SLOT_STRIKE_ROUTE_LEFT", "RED_ZONE_FADE_RIGHT", "OVER_SHOULDER_ADJUST", "CIRCUS_ROUTE_LEFT"],
          "TE": ["FLAT_ROUTE_LEFT", "CORNER_ROUTE", "WHEEL_ROUTE_LEFT", "IN_ROUTE_LEFT", "HOOK_ROUTE_RIGHT",
                 "END_ZONE_FADE_RIGHT", "OVER_SHOULDER_ADJUST"],
          "DB": ["BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION", "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL", "LINE_DRILL",
                 "W_DRILL", "TERYL_AUSTIN_DRILL_1", "TERYL_AUSTIN_DRILL_2"]}
MOVE = {"OL": "kick-slide", "DL": "get-off", "WR": "release", "TE": "release", "DB": "backpedal"}
OUTC = {"OL": ("pressure_rate_allowed", "pass_blocks", -1), "DL": ("pressure_rate", "pass_rush_snaps", 1),
        "WR": ("yards_per_route", "routes", 1), "TE": ("yards_per_route", "routes", 1),
        "DB": ("epa_per_play", "coverage_snaps", -1)}
ALLD = sorted({d for v in DRILLS.values() for d in v})

ev = ev[ev.drill_name.isin(ALLD)].sort_values(["nfl_id", "drill_name", "start_time"]).copy()
ev["rep"] = ev.groupby(["nfl_id", "drill_name"]).cumcount()
crow = []
for eid, d in cf[cf.event_id.isin(ev.event_id)].groupby("event_id", sort=False):
    s, a = d.s.values, d.a.values
    i = onset(s)
    if i is None or len(s) < i + NPT:
        continue
    crow.append(dict(event_id=eid, **{f"s{j}": s[i + j] for j in range(NPT)}, **{f"a{j}": a[i + j] for j in range(NPT)}))
CB = pd.DataFrame(crow).merge(ev[["event_id", "nfl_id", "draft_year", "drill_name", "rep", "position_group",
                                 "start_time"]], on="event_id")
CB = CB.sort_values(["nfl_id", "start_time"])
SC = [f"s{j}" for j in range(NPT)]
AC = [f"a{j}" for j in range(NPT)]

ff = gf.merge(gp[["game_id", "play_id", "nfl_id", "snap_frame", "sample_kind", "lined_up_position", "route_ran",
                  "separation_at_pass_forward", "cushion", "target"]], on=["game_id", "play_id", "nfl_id"])
ff = ff[ff.frame >= ff.snap_frame].sort_values(["game_id", "play_id", "nfl_id", "frame"])
nrow = []
for k, d in ff.groupby(["game_id", "play_id", "nfl_id"], sort=False):
    s, a = d.s.values, d.a.values
    i = onset(s)
    if i is None or i > 10 or len(s) < i + NPT:
        continue
    r = d.iloc[0]
    nrow.append(dict(game_id=k[0], play_id=k[1], nfl_id=k[2], kind=r.sample_kind, pos=r.lined_up_position,
                     route=r.route_ran, sep=r.separation_at_pass_forward, cushion=r.cushion, target=r.target,
                     **{f"s{j}": s[i + j] for j in range(NPT)}, **{f"a{j}": a[i + j] for j in range(NPT)}))
N = pd.DataFrame(nrow)
N["group"] = N.nfl_id.map(pl.position_group)


def career(group):
    o, w, sg = OUTC[group]
    s = st[st.position_group == group].copy()
    s["num"] = s[o] * s[w]
    a = s.groupby("nfl_id")[["num", w]].sum()
    role = s.sort_values(w).groupby("nfl_id").main_alignment.last()
    a = a[a[w] >= MIN_DEN]
    return pd.DataFrame({"y": sg * a.num / a[w], "den": a[w], "role": role.reindex(a.index)})


def zcurves(df, keys):
    out = df.copy()
    for c in SC + AC:
        g = out.groupby(keys)[c]
        out[c] = (out[c] - g.transform("mean")) / g.transform("std")
    return out


def nfl_rows(group):
    n = N[(N.group == group) & (N.kind == "random")]
    keys = ["pos"]
    if group in ("WR", "TE"):
        n = n[n.route.notna() & (n.route != "SCREEN")]
        keys = ["pos", "route"]
    return n.groupby(keys).filter(lambda x: len(x) >= 30), keys


def player_profiles(group):
    c = CB[CB.drill_name.isin(DRILLS[group]) & (CB.position_group == group)]
    cz = zcurves(c, ["drill_name", "draft_year", "rep"])
    n, keys = nfl_rows(group)
    return cz, zcurves(n, keys)


def match_score(cz_, nz_):
    """Release/Set Match: minus the mean absolute gap between the two league-relative profiles, with the part
    explained by sampling noise (sqrt(1/reps + 1/snaps)) removed. Returns (match, raw, noise)."""
    a = cz_.groupby("nfl_id")[SC + AC].mean()
    b = nz_.groupby("nfl_id")[SC + AC].mean()
    ii = a.index.intersection(b.index)
    raw = -(a.loc[ii] - b.loc[ii]).abs().mean(1)
    noise = np.sqrt(1 / cz_.groupby("nfl_id").size().reindex(ii) + 1 / nz_.groupby("nfl_id").size().reindex(ii))
    f = sm.OLS(raw, sm.add_constant(noise)).fit()
    return raw - f.params.iloc[1] * (noise - noise.mean()), raw, noise


def rho_ci(x, y, nb=NBOOT):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r0 = spearmanr(x, y)[0]
    bs = [spearmanr(x[i], y[i])[0] for i in (RNG.integers(0, len(x), len(x)) for _ in range(nb))]
    return r0, np.nanpercentile(bs, 5), np.nanpercentile(bs, 95)


def sb(r):
    return 2 * r / (1 + r) if r > 0 else r


def half_rel(df, cols):
    h = df.assign(l=df[cols].mean(1), h=df.groupby("nfl_id").cumcount() % 2)
    v = h.pivot_table(index="nfl_id", columns="h", values="l").dropna()
    return (sb(spearmanr(v[0], v[1])[0]), len(v)) if len(v) >= 20 else (np.nan, len(v))


# ------------------------------------------------------------------------------------------------- 1. screen
screen, store = [], {}
for g in DRILLS:
    cz, nz = player_profiles(g)
    dprof = cz.groupby("nfl_id")[SC + AC].mean()
    nprof = nz.groupby("nfl_id")[SC + AC].mean()
    ids = dprof.index.intersection(nprof.index)
    dlev, nlev = dprof.loc[ids, SC[LV]].mean(1), nprof.loc[ids, SC[LV]].mean(1)
    match = match_score(cz, nz)[0].reindex(ids)
    rd, _ = half_rel(cz, SC[LV])
    rn, _ = half_rel(nz, SC[LV])
    ceil = np.sqrt(max(rd, 0) * max(rn, 0)) if not np.isnan(rd) else 0.0
    r0, rlo, rhi = rho_ci(dlev, nlev)
    df = pd.DataFrame({"match": match}).join(career(g), how="inner").join(pl.lpick)
    Z = df[["lpick", "match"]].apply(lambda v: (v - v.mean()) / v.std())
    fit = sm.OLS((df.y - df.y.mean()) / df.y.std(), sm.add_constant(Z)).fit()
    ci = fit.conf_int(0.10).loc["match"]
    screen.append(dict(group=g, n_tr=len(ids), rho=r0, rlo=rlo, rhi=rhi, rel_d=rd, rel_n=rn, ceil=ceil,
                       n_y=len(df), b=fit.params["match"], blo=ci[0], bhi=ci[1], p=fit.pvalues["match"]))
    store[g] = (cz, nz, dprof, nprof)
SCR = pd.DataFrame(screen).set_index("group")
# DB production here is on-field team EPA per play, not an individual coverage stat, so DBs cannot answer the
# production question even though their backpedal translates; they stay in the screen as context.
INDIVIDUAL = {"OL": True, "DL": True, "WR": True, "TE": True, "DB": False}
SCR["indiv"] = pd.Series(INDIVIDUAL)
PICK = SCR[(SCR.ceil >= CEIL_MIN) & SCR.indiv].rlo.idxmax()
assert PICK == "WR", SCR
for g, r in SCR.iterrows():
    put(f"scr_{g}_rho", r.rho), put(f"scr_{g}_rlo", r.rlo), put(f"scr_{g}_rhi", r.rhi)
    put(f"scr_{g}_ceil", r.ceil), put(f"scr_{g}_reld", r.rel_d), put(f"scr_{g}_reln", r.rel_n)
    put(f"scr_{g}_b", r.b), put(f"scr_{g}_p", r.p)
    put(f"scr_{g}_ny", r.n_y, "{:.0f}"), put(f"scr_{g}_ntr", r.n_tr, "{:.0f}")
put("ceil_min", CEIL_MIN, "{:.1f}")
put("scr_minp", SCR.p.min())

# ------------------------------------------------------------------------------------------------- 2. WR deep dive
cz, nz, dprof, nprof = store["WR"]
craw = CB[CB.drill_name.isin(DRILLS["WR"]) & (CB.position_group == "WR")]
nraw, _ = nfl_rows("WR")
put("n_drill_pl", craw.nfl_id.nunique(), "{:.0f}")
put("n_drill_reps", len(craw), "{:,.0f}")
put("reps_med", craw.groupby("nfl_id").size().median(), "{:.0f}")
put("n_drills", craw.drill_name.nunique(), "{:.0f}")
put("n_nfl_pl", nraw.nfl_id.nunique(), "{:.0f}")
put("n_nfl_snaps", len(nraw), "{:,.0f}")
put("snaps_med", nraw.groupby("nfl_id").size().median(), "{:.0f}")
for t in (5, 10, 15):
    put(f"drill_v{t}", craw[f"s{t}"].mean(), "{:.1f}")
    put(f"nfl_v{t}", nraw[f"s{t}"].mean(), "{:.1f}")
put("drill_a3", craw[AC[0:6]].mean(1).mean(), "{:.1f}")
put("nfl_a3", nraw[AC[0:6]].mean(1).mean(), "{:.1f}")
put("rel_drill", SCR.loc["WR", "rel_d"])
put("rel_nfl", SCR.loc["WR", "rel_n"])

ids = dprof.index.intersection(nprof.index)
mt, raw, noise = match_score(cz, nz)
X0 = pd.DataFrame({"dlev": dprof.loc[ids, SC[LV]].mean(1), "nlev": nprof.loc[ids, SC[LV]].mean(1),
                   "match": mt.reindex(ids), "raw_match": raw.reindex(ids), "noise": noise.reindex(ids)})
put("noise_rho", spearmanr(X0.noise, X0.raw_match)[0])
put("n_both", len(X0), "{:.0f}")
# Release Match reliability: Combine half A + NFL half A vs Combine half B + NFL half B
czh = cz.assign(h=cz.groupby("nfl_id").cumcount() % 2)
nzh = nz.assign(h=nz.groupby("nfl_id").cumcount() % 2)
hm = {}
for h in (0, 1):
    hm[h] = match_score(czh[czh.h == h], nzh[nzh.h == h])[0]
jj = hm[0].index.intersection(hm[1].index)
REL_MATCH = sb(spearmanr(hm[0][jj], hm[1][jj])[0])
put("rel_match", REL_MATCH)
put("n_rel_match", len(jj), "{:.0f}")
put("match_vs_absn", spearmanr(X0.match, X0.nlev.abs())[0])
put("match_vs_absd", spearmanr(X0.match, X0.dlev.abs())[0])

r0, lo, hi = SCR.loc["WR", ["rho", "rlo", "rhi"]]          # same bootstrap as the screen (Figure 1)
put("tr", r0), put("tr_lo", lo), put("tr_hi", hi)
put("tr_share", 100 * r0 / SCR.loc["WR", "ceil"], "{:.0f}")
# speed only vs accel profile translation
dacc = dprof.loc[ids, AC[0:6]].mean(1)
nacc = nprof.loc[ids, AC[0:6]].mean(1)
r0, lo, hi = rho_ci(dacc, nacc)
put("tr_acc", r0), put("tr_acc_lo", lo), put("tr_acc_hi", hi)
# no-tracking benchmarks: what a stopwatch or a scale says about the NFL release
bench = X0.join(cr[["forty", "ten_yd_split", "combine_weight"]])
for key, col, sg in [("bm_40", "forty", -1), ("bm_split", "ten_yd_split", -1), ("bm_wt", "combine_weight", 1)]:
    m = bench[col].notna()
    r0, lo, hi = rho_ci(sg * bench.loc[m, col], bench.loc[m, "nlev"])
    put(key, r0), put(key + "_lo", lo), put(key + "_hi", hi), put(key + "_n", m.sum(), "{:.0f}")
m = bench.forty.notna()
put("bm_40_drill", spearmanr(-bench.forty[m], bench.dlev[m])[0])
# partial: does the drill release add to the official 40 for the NFL release?
bz = bench[m].apply(lambda v: (v - v.mean()) / v.std())
f = sm.OLS(bz.nlev, sm.add_constant(bz[["dlev", "forty"]])).fit()
put("bm_part_drill", f.params["dlev"]), put("bm_part_drill_p", f.pvalues["dlev"])
put("bm_part_40", -f.params["forty"])

# play level: does a faster-than-usual release go with more separation at the throw?
pp = nz[nz.sep.notna() & nz.cushion.notna()].copy()
pp["lev"] = pp[SC[LV]].mean(1)
pp["lev_w"] = (pp.lev - pp.groupby("nfl_id").lev.transform("mean"))
pp["lev_w"] /= pp.lev_w.std()
rawlev = nraw.loc[pp.index, SC[LV]].mean(1)
put("play_n", len(pp), "{:,.0f}")
put("play_pl", pp.nfl_id.nunique(), "{:.0f}")
lm = smf.ols("sep ~ lev_w + cushion + C(route) + C(pos)", pp).fit(cov_type="cluster", cov_kwds={"groups": pp.nfl_id})
put("play_sep", lm.params["lev_w"], "{:.2f}")
ci = lm.conf_int(0.10).loc["lev_w"]
put("play_sep_lo", ci[0], "{:.2f}"), put("play_sep_hi", ci[1], "{:.2f}")
put("play_sep_mean", pp.sep.mean(), "{:.1f}")
pr = nraw.loc[pp.index].assign(rl=rawlev)
lc = smf.ols("rl ~ cushion + C(route) + C(pos)", pr).fit(cov_type="cluster", cov_kwds={"groups": pr.nfl_id})
put("cush_eff", lc.params["cushion"], "{:.2f}")
ci = lc.conf_int(0.10).loc["cushion"]
put("cush_lo", ci[0], "{:.2f}"), put("cush_hi", ci[1], "{:.2f}")
put("cush_eff5", 5 * lc.params["cushion"], "{:.1f}")

# ------------------------------------------------------------------------------------------------- 3. production
D = X0.join(career("WR"), how="inner").join(pl[["lpick", "draft_year", "display_name", "draft_overall_pick"]]).join(
    cr[["forty", "combine_weight", "ngs_athleticism_score"]])
D["slot"] = (D.role == "SLOT_WR").astype(float)
put("n_prod", len(D), "{:.0f}")
put("y_mean", D.y.mean())
put("y_sd", D.y.std())
put("n_no40", D.forty.isna().sum(), "{:.0f}")
BASE = {"pick": ["lpick"], "pick_role": ["lpick", "slot"],
        "full": ["lpick", "slot", "forty", "combine_weight", "ngs_athleticism_score"]}
TESTS = ["match", "dlev", "nlev"]


def loco(df, cols, test=None):
    """Held-out predictions: each draft class predicted by a weighted fit on the other two (numpy for speed)."""
    cs = cols + ([test] if test else [])
    X, y, w, cl = df[cs].to_numpy(float), df.y.to_numpy(float), df.den.to_numpy(float), df.draft_year.to_numpy()
    pred = np.full(len(y), np.nan)
    for yr in np.unique(cl):
        tr, te = cl != yr, cl == yr
        Xa = X.copy()
        med = np.nanmedian(Xa[tr], axis=0)                      # training-fold median for a missing 40 time
        Xa = np.where(np.isnan(Xa), med, Xa)
        mu, sd = Xa[tr].mean(0), Xa[tr].std(0, ddof=1)
        sd[sd == 0] = 1
        Xs = np.column_stack([np.ones(len(y)), (Xa - mu) / sd])
        sw = np.sqrt(w[tr])
        beta = np.linalg.lstsq(Xs[tr] * sw[:, None], y[tr] * sw, rcond=None)[0]
        pred[te] = Xs[te] @ beta
    return pd.Series(pred, index=df.index)


def r2(y, p, w):
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)


def wslope(miss, x, w):
    z = (x - x.mean()) / x.std()
    Xs = np.column_stack([np.ones(len(z)), z]) * np.sqrt(w)[:, None]
    return np.linalg.lstsq(Xs, miss * np.sqrt(w), rcond=None)[0][1]


def evaluate(df):
    res = {}
    for b, cols in BASE.items():
        p0 = loco(df, cols)
        res[(b, "r2")] = r2(df.y, p0, df.den)
        miss = (df.y - p0).to_numpy()
        for t in TESTS:
            res[(b, t)] = wslope(miss, df[t].to_numpy(float), df.den.to_numpy(float))
            res[(b, t, "dr2")] = r2(df.y, loco(df, cols, t), df.den) - res[(b, "r2")]
    return res


EST = evaluate(D)
classes = D.draft_year.values
boot = []
for _ in range(NBOOT):
    idx = np.concatenate([RNG.choice(np.flatnonzero(classes == c), (classes == c).sum()) for c in np.unique(classes)])
    Db = D.iloc[idx].copy()
    Db.index = range(len(Db))
    boot.append(evaluate(Db))
B = pd.DataFrame(boot)
for b in BASE:
    put(f"r2_{b}", EST[(b, "r2")])
    put(f"r2_{b}_lo", B[(b, "r2")].quantile(.05)), put(f"r2_{b}_hi", B[(b, "r2")].quantile(.95))
    for t in TESTS:
        put(f"{b}_{t}", EST[(b, t)])
        put(f"{b}_{t}_lo", B[(b, t)].quantile(.05)), put(f"{b}_{t}_hi", B[(b, t)].quantile(.95))
        put(f"{b}_{t}_dr2", EST[(b, t, "dr2")])
        put(f"{b}_{t}_dr2_lo", B[(b, t, "dr2")].quantile(.05)), put(f"{b}_{t}_dr2_hi", B[(b, t, "dr2")].quantile(.95))
put("p_beyond_margin", 100 * (B[("full", "match")].abs() > MARGIN).mean(), "{:.0f}")
put("p_pos", 100 * (B[("full", "match")] > 0).mean(), "{:.0f}")
put("margin", MARGIN)
D["miss"] = D.y - loco(D, BASE["full"])
zc = (D.match - D.match.mean()) / D.match.std()
for yr in sorted(D.draft_year.unique()):
    m = D.draft_year == yr
    f = sm.WLS(D.miss[m], sm.add_constant(zc[m]), weights=D.den[m]).fit()
    put(f"cls_{yr}", f.params.iloc[1]), put(f"cls_{yr}_n", m.sum(), "{:.0f}")
# do clubs already pay for it? (all receivers with both profiles)
Pm = X0.join(pl.lpick)
put("price_rho", spearmanr(Pm.match, Pm.lpick)[0])
put("price_nlev", spearmanr(Pm.nlev, Pm.lpick)[0])
put("price_n", len(Pm), "{:.0f}")

# ------------------------------------------------------------------------------------------------- 4. measurement error
sd_res = np.sqrt(np.average(D.miss ** 2, weights=D.den))
put("res_sd", sd_res)
n_sim = len(D)
rel = max(REL_MATCH, 0.05)
true_b = np.round(np.arange(-0.4, 0.41, 0.05), 2)


def power(tb, rel_, reps=1000):
    hits, ests = 0, []
    for _ in range(reps):
        t = RNG.standard_normal(n_sim)
        obs = np.sqrt(rel_) * t + np.sqrt(1 - rel_) * RNG.standard_normal(n_sim)
        y = tb * t + RNG.standard_normal(n_sim) * np.sqrt(max(sd_res ** 2 - tb ** 2, 0.01))
        f = sm.OLS(y, sm.add_constant(obs)).fit()
        lo, hi = f.conf_int(0.10)[1]
        hits += (lo > 0) or (hi < 0)
        ests.append(f.params[1])
    return hits / reps, float(np.mean(ests))


SIM = pd.DataFrame([dict(tb=tb, rel=rr, pw=power(tb, rr)[0]) for rr in (rel, 0.8) for tb in true_b])
pw_m, est_m = power(MARGIN, rel, 2000)
put("sim_rel", rel)
put("sim_pw_margin", 100 * pw_m, "{:.0f}")
put("sim_att_margin", est_m)
put("sim_pw_margin_80", 100 * power(MARGIN, 0.8, 2000)[0], "{:.0f}")
# how often a true effect of +MARGIN would produce an estimate at or below the observed one
obs_b = EST[("full", "match")]
cnt = 0
for _ in range(2000):
    t = RNG.standard_normal(n_sim)
    obs = np.sqrt(rel) * t + np.sqrt(1 - rel) * RNG.standard_normal(n_sim)
    y = MARGIN * t + RNG.standard_normal(n_sim) * np.sqrt(sd_res ** 2 - MARGIN ** 2)
    cnt += sm.OLS(y, sm.add_constant(obs)).fit().params[1] <= obs_b
put("sim_p_obs", 100 * cnt / 2000, "{:.0f}")

# ------------------------------------------------------------------------------------------------- 5. player cards
nzp, czp = nz[nz.nfl_id.isin(D.index)], cz[cz.nfl_id.isin(D.index)]
# per-player check: at how many of the 16 speed points is his drill-minus-NFL gap clear of zero (90%)?
clear_pts = {}
for pid in D.index:
    a, b = nzp[nzp.nfl_id == pid][SC], czp[czp.nfl_id == pid][SC]
    gap = b.mean() - a.mean()
    se = np.sqrt(b.var() / len(b) + a.var() / len(a))
    clear_pts[pid] = int((gap.abs() > 1.645 * se).sum())
D["clear_pts"] = pd.Series(clear_pts)
med = D.match.median()
D["clear"] = D.clear_pts >= NPT // 2
put("match_med", med), put("match_p10", D.match.quantile(.1)), put("match_p90", D.match.quantile(.9))
put("n_clear", D.clear.sum(), "{:.0f}")
put("n_clear_drill_fast", ((D.clear) & (D.dlev > D.nlev)).sum(), "{:.0f}")
D = D.sort_values("match", ascending=False)
cards = pd.concat([D.head(4), D.tail(4)])
for k, (pid, r) in enumerate(cards.iterrows()):
    put(f"card{k}_match", r.match), put(f"card{k}_pts", r.clear_pts, "{:.0f}")
    put(f"card{k}_y", r.y), put(f"card{k}_miss", r.miss)
    put(f"card{k}_dlev", r.dlev, "{:.1f}"), put(f"card{k}_nlev", r.nlev, "{:.1f}")
    NUM[f"card{k}_name"] = r.display_name
    NUM[f"card{k}_pick"] = "UDFA" if np.isnan(r.draft_overall_pick) else f"{r.draft_overall_pick:.0f}"
D.reset_index().rename(columns={"index": "nfl_id"})[
    ["nfl_id", "display_name", "draft_year", "draft_overall_pick", "role", "den", "y", "miss", "match", "clear_pts",
     "dlev", "nlev"]].round(3).to_csv(OUT / "inputs.csv", index=False)

# ------------------------------------------------------------------------------------------------- figures
T = np.round(np.arange(NPT) / 10, 1)

# Fig 1: the five-group screen
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 4.0), sharey=True)
order = ["WR", "DB", "OL", "DL", "TE"]
yy = np.arange(len(order))[::-1]
for k, g in enumerate(order):
    r = SCR.loc[g]
    col = OI["blue"] if g == PICK else OI["grey"]
    ax[0].plot([r.rlo, r.rhi], [yy[k]] * 2, color=col, lw=2.5)
    ax[0].plot(r.rho, yy[k], "o", color=col, ms=8)
    ax[0].plot(r.ceil, yy[k], "|", color=OI["verm"], ms=16, mew=2.5)
    ax[1].plot([r.blo, r.bhi], [yy[k]] * 2, color=col, lw=2.5)
    ax[1].plot(r.b, yy[k], "o", color=col, ms=8)
    ax[1].text(0.98, yy[k] + 0.28, f"p = {r.p:.2f}, n = {r.n_y:.0f}", fontsize=10, ha="right", color=col)
    if r.ceil == 0:
        ax[0].text(0.03, yy[k] + 0.28, "ceiling 0: one side not repeatable", fontsize=9, color=OI["verm"])
    if not r.indiv:
        ax[1].text(-0.48, yy[k] + 0.28, "team EPA only, not eligible", fontsize=9, color=OI["grey"])
ax[0].set_yticks(yy, [f"{g} {MOVE[g]}\n(n = {SCR.loc[g, 'n_tr']:.0f})" for g in order])
for a in ax:
    a.axvline(0, color=OI["light"], lw=1)
ax[0].set_xlim(-0.4, 0.8)
ax[0].set_ylim(-0.5, 4.6)
ax[1].set_xlim(-0.5, 1.0)
ax[0].set_xlabel("Combine start -> NFL start, rank correlation (90% CI)")
ax[1].set_xlabel("Match beyond log pick, sd of production per sd (90% CI)")
ax[0].set_title("A. Does the drill start show up on Sundays?", loc="left", fontsize=11)
ax[1].set_title("B. Does matching it go with production?", loc="left", fontsize=11)
ax[0].plot([], [], "|", color=OI["verm"], ms=12, mew=2.5, label="noise ceiling")
ax[0].legend(loc="lower right", fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig1_screen.png", dpi=150)
plt.close(fig)

# Fig 2: league curves (A) and translation scatter (B)
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
v = craw[SC]
ax[0].plot(T, v.mean(), color=OI["orange"], lw=2.4, label=f"Combine route drills ({NUM['n_drill_reps']} reps)")
ax[0].fill_between(T, v.quantile(.25), v.quantile(.75), color=OI["orange"], alpha=.15, lw=0)
v = nraw[SC]
ax[0].plot(T, v.mean(), color=OI["blue"], lw=2.4, label=f"NFL routes, random snaps ({NUM['n_nfl_snaps']})")
ax[0].fill_between(T, v.quantile(.25), v.quantile(.75), color=OI["blue"], alpha=.15, lw=0)
ax[0].axvspan(0.3, 1.2, color=OI["light"], alpha=.25, lw=0)
ax[0].text(0.75, 0.3, "release window", ha="center", fontsize=9, color=OI["grey"])
ax[0].set_xlabel("Seconds after first movement (0.5 yd/s)")
ax[0].set_ylabel("Speed, yd/s (band = middle 50%)")
ax[0].legend(fontsize=9, frameon=False, loc="upper left")
ax[0].set_title("A. Same start, two settings", loc="left", fontsize=11)
ax[1].scatter(X0.dlev, X0.nlev, s=24, color=OI["blue"], alpha=.7, lw=0)
for pid in cards.index[[0, -1]]:
    ax[1].annotate(pl.loc[pid, "display_name"], (X0.loc[pid, "dlev"], X0.loc[pid, "nlev"]), fontsize=9,
                   xytext=(5, 4), textcoords="offset points")
ax[1].axhline(0, color=OI["light"], lw=1)
ax[1].axvline(0, color=OI["light"], lw=1)
ax[1].set_xlabel("Combine release speed, sd vs same drill and year")
ax[1].set_ylabel("NFL release speed, sd vs same route and alignment")
ax[1].set_title(f"B. rho = {NUM['tr']} [{NUM['tr_lo']}, {NUM['tr_hi']}], n = {NUM['n_both']}", loc="left",
                fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig2_release.png", dpi=150)
plt.close(fig)

# Fig 3: player cards
fig, axs = plt.subplots(2, 4, figsize=(FIGW, 5.8), sharex=True, sharey=True)
for k, (pid, r) in enumerate(cards.iterrows()):
    a = axs.flat[k]
    dz = czp[czp.nfl_id == pid][SC]
    nzv = nzp[nzp.nfl_id == pid][SC]
    for vv, col in [(dz, OI["orange"]), (nzv, OI["blue"])]:
        mu, se = vv.mean().values, vv.std().values / np.sqrt(len(vv))
        a.plot(T, mu, color=col, lw=2.2)
        a.fill_between(T, mu - 1.645 * se, mu + 1.645 * se, color=col, alpha=.15, lw=0)
    a.axhline(0, color=OI["light"], lw=1)
    tc = "black" if r.clear else OI["grey"]
    pick = "UDFA" if np.isnan(r.draft_overall_pick) else f"pick {r.draft_overall_pick:.0f}"
    a.set_title(f"{r.display_name} ({pick})", fontsize=9.5, loc="left", color=tc)
    a.text(0.03, 0.03, f"Match {r.match:.2f}; gap clear at {r.clear_pts}/16\nYPRR {r.y:.2f} ({r.miss:+.2f} vs forecast)",
           transform=a.transAxes, fontsize=9, color=tc, va="bottom")
    if not r.clear:
        a.set_facecolor("#F2F2F2")
for a in axs[1]:
    a.set_xlabel("s after first movement")
for a in axs[:, 0]:
    a.set_ylabel("speed, sd vs peers")
axs[0, 0].plot([], [], color=OI["orange"], lw=2.2, label="Combine drills")
axs[0, 0].plot([], [], color=OI["blue"], lw=2.2, label="NFL routes")
axs[0, 0].legend(fontsize=8.5, frameon=False, loc="upper left")
axs[0, 0].set_ylim(-2.6, 2.6)
fig.suptitle("Top: four closest Release Matches. Bottom: four widest gaps. Grey card = gap not clear of zero at half the points.",
             fontsize=10.5, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(FIG / "fig3_cards.png", dpi=150)
plt.close(fig)

# Fig 4: held-out slopes (A) and measurement-error power (B)
fig, ax = plt.subplots(1, 2, figsize=(FIGW, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
labs = {"match": "Release Match", "dlev": "Combine release", "nlev": "NFL release"}
blabs = {"pick": "pick only", "pick_role": "pick + slot", "full": "full"}
y0, ticks, tl = 0, [], []
for t in TESTS:
    for b in BASE:
        est, lo, hi = EST[(b, t)], B[(b, t)].quantile(.05), B[(b, t)].quantile(.95)
        col = OI["blue"] if t == "match" else OI["grey"]
        ax[0].plot([lo, hi], [y0, y0], color=col, lw=2.5)
        ax[0].plot(est, y0, "o", color=col, ms=7)
        ticks.append(y0)
        tl.append(f"{labs[t]}, {blabs[b]}")
        y0 -= 1
    y0 -= 0.6
ax[0].axvspan(-MARGIN, MARGIN, color=OI["light"], alpha=.3, lw=0)
ax[0].axvline(0, color=OI["grey"], lw=1)
ax[0].set_yticks(ticks, tl, fontsize=10)
ax[0].set_xlabel("Yards per route run per sd, held-out class (90% CI)")
ax[0].set_title("A. Beyond three no-tracking baselines (grey band = margin)", loc="left", fontsize=11)
for rr, col, lab in [(rel, OI["blue"], f"Release Match as measured (reliability {NUM['sim_rel']})"),
                     (0.8, OI["orange"], "if reliability were 0.8")]:
    s_ = SIM[SIM.rel == rr]
    ax[1].plot(s_.tb, 100 * s_.pw, "-o", color=col, ms=4, label=lab)
ax[1].axvline(MARGIN, color=OI["light"], lw=1, ls="--")
ax[1].axvline(-MARGIN, color=OI["light"], lw=1, ls="--")
ax[1].set_xlabel("True effect, yards per route run per sd")
ax[1].set_ylabel("% of simulated studies that detect it")
ax[1].set_title(f"B. What {NUM['n_prod']} receivers can detect", loc="left", fontsize=11)
ax[1].legend(fontsize=9, frameon=False, loc="upper center", ncol=1)
ax[1].set_ylim(0, 135)
ax[1].set_yticks([0, 20, 40, 60, 80, 100])
fig.tight_layout()
fig.savefig(FIG / "fig4_production.png", dpi=150)
plt.close(fig)

(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
print(SCR.round(3).to_string())
print(json.dumps(NUM, indent=0))
