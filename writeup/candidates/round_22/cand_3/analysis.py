"""Round 22, cand_3 - movement translation for wide receivers: Stem Gear.

Question: for the same movement (a vertical release from a stance), how much of his own Combine go-route speed does a
receiver use on NFL vertical stems, and does that translation explain production beyond draft pick?

Stem Gear = (his mean NFL speed 1.0-2.0 s after release on GO / POST / CORNER routes, before the throw, centred by route)
            / (his mean speed 1.0-2.0 s after release on the Combine GO_ROUTE drill, year-adjusted).
Release = first frame after the snap (NFL) or clip start (Combine) at which speed reaches 1.5 yd/s; aligning on this
crossing removes the ~0.3 s pre-laser lead of the Combine clip and stance/motion differences.

Run from the repo root (about one minute):
    uv run python writeup/candidates/round_22/cand_3/analysis.py
Writes numbers.json, figures/*.png (1600 px wide, 150 dpi) and inputs.csv next to this file.
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
RNG = np.random.default_rng(2722)
NBOOT = 2000
THR = 1.5            # release threshold, yd/s
GRID = np.round(np.arange(0, 3.01, 0.1), 1)
WIN = (1.0, 2.0)     # primary window, s after release (median throw comes 2.4 s after release)
VERT = ["GO", "POST", "CORNER"]
MIN_PLAYS, MIN_ROUTES = 3, 100
MARGIN = 0.15        # practical margin, yards per route per sd (about 12% of the WR mean)
NUM: dict = {}
CONST = {"thr": "1.5", "win_lo": "1.0", "win_hi": "2.0", "early_lo": "0.5", "early_hi": "1.5", "unc_lo": "1.5",
         "unc_hi": "3.0", "clip_offset": "0.3", "t_early": "0.5", "t_mid": "2.0"}


def put(key, val, fmt="{:.2f}"):
    s = fmt.format(val) if not isinstance(val, str) else val
    if s.startswith("-0") and float(s) == 0:
        s = s[1:]
    NUM[key] = s
    return s


# ----------------------------------------------------------------------------------------------- data
pl = pd.read_parquet(P / "players.parquet").set_index("nfl_id")
cr = pd.read_parquet(P / "combine_results.parquet").drop_duplicates("nfl_id").set_index("nfl_id")
ev = pd.read_parquet(P / "combine_events.parquet")
cf = pd.read_parquet(P / "combine_frames.parquet", columns=["event_id", "frame", "entity_type", "s"])
cf = cf[cf.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
gp = pd.read_parquet(P / "game_plays_sample.parquet")
gf = pd.read_parquet(P / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "frame", "s", "event"])
st = pd.read_parquet(P / "player_season_stats.parquet")
st = st[st.position_group == "WR"].copy()
END_EVENTS = {"pass_forward", "qb_sack", "qb_strip_sack", "run", "handoff", "qb_spike"}


def released(s):
    """Index of the first frame at or above THR, or None."""
    hit = np.flatnonzero(s >= THR)
    return int(hit[0]) if len(hit) else None


# Combine curves (go-route drill = same movement; 40 = alternative reference)
crow = []
for dn in ["GO_ROUTE_RIGHT", "FORTY_YARD_DASH"]:
    e = ev[(ev.drill_name == dn) & (ev.position_group == "WR")]
    for eid, d in cf[cf.event_id.isin(e.event_id)].groupby("event_id"):
        s = d.s.values
        i = released(s)
        if i is None or len(s) < i + len(GRID):
            continue
        crow.append(dict(event_id=eid, drill=dn, **{f"v{t}": v for t, v in zip(GRID, s[i:i + len(GRID)])}))
C = pd.DataFrame(crow).merge(ev[["event_id", "nfl_id", "draft_year", "attempt_no"]], on="event_id")
VC = [f"v{t}" for t in GRID]
WC = [f"v{t}" for t in GRID if WIN[0] <= t <= WIN[1]]
C["win"] = C[WC].mean(1)
# year adjustment: remove each Combine year's mean shift for that drill
C["win_adj"] = C.win - C.groupby(["drill", "draft_year"]).win.transform("mean") + C.groupby("drill").win.transform("mean")

# NFL vertical stems
gv = gp[gp.route_ran.isin(VERT)]
ff = gf.merge(gv[["game_id", "play_id", "nfl_id", "snap_frame", "route_ran", "sample_kind", "target", "lined_up_position",
                  "season"]], on=["game_id", "play_id", "nfl_id"]).sort_values(["game_id", "play_id", "nfl_id", "frame"])
nrow = []
for k, d in ff.groupby(["game_id", "play_id", "nfl_id"], sort=False):
    fr, s = d.frame.values, d.s.values
    ends = fr[d.event.isin(END_EVENTS).values]
    endf = ends.min() if len(ends) else fr.max()
    m = fr >= d.snap_frame.iloc[0]
    fr2, s2 = fr[m], s[m]
    i = released(s2)
    if i is None:
        continue
    tt = np.round((fr2 - fr2[i]) / 10, 1)
    r = d.iloc[0]
    row = dict(nfl_id=k[2], game_id=k[0], play_id=k[1], route=r.route_ran, kind=r.sample_kind, target=bool(r.target),
               slot=r.lined_up_position == "SLOT_WR", season=r.season, throw_t=(endf - fr2[i]) / 10)
    pre = fr2 <= endf
    for lo, hi, nm in [(1.0, 2.0, "w"), (0.5, 1.5, "w_early"), (1.5, 2.5, "w_late")]:
        mm = (tt >= lo) & (tt <= hi)
        row[nm] = s2[mm & pre].mean() if (mm & pre).sum() == mm.sum() == int(round((hi - lo) * 10)) + 1 else np.nan
    mm = (tt >= 1.5) & (tt <= 3.0)
    row["w_uncens"] = s2[mm].mean() if mm.sum() == 16 else np.nan
    cur = np.full(len(GRID), np.nan)
    for j, t in enumerate(GRID):
        hit = np.flatnonzero((tt == t) & pre)
        if len(hit):
            cur[j] = s2[hit[0]]
    row.update({f"v{t}": v for t, v in zip(GRID, cur)})
    nrow.append(row)
N = pd.DataFrame(nrow)
N = N[N.nfl_id.isin(pl.index[pl.position_group == "WR"])]
NV = N.dropna(subset=["w"]).copy()
put("n_vert_plays", len(N), "{:d}")
put("n_vert_plays_win", len(NV), "{:d}")
put("throw_med", N.throw_t.median(), "{:.1f}")


def nfl_player(df, col="w", by=None):
    d = df.dropna(subset=[col]).copy()
    d["c"] = d[col] - d.groupby("route")[col].transform("mean") + d[col].mean()
    keys = ["nfl_id"] + ([by] if by else [])
    return d.groupby(keys).agg(nfl=("c", "mean"), k=("c", "size"))


# ----------------------------------------------------------------------------------------------- outcomes
st["yds"] = st.rec_yards
car = st.groupby("nfl_id").agg(routes=("routes", "sum"), yds=("yds", "sum"), tgt=("targets", "sum"), seasons=("season", "size"))
car["yprr"] = car.yds / car.routes
car["tprr"] = car.tgt / car.routes
car["cush"] = st.assign(cw=st.avg_cushion * st.routes).groupby("nfl_id").cw.sum() / car.routes


def build(ref="GO_ROUTE_RIGHT", nfl_df=NV, col="w", min_plays=MIN_PLAYS):
    cref = C[C.drill == ref].groupby("nfl_id").win_adj.mean().rename("comb")
    npl = nfl_player(nfl_df, col)
    d = npl[npl.k >= min_plays].join(cref, how="inner")
    d = d.join(pl[["display_name", "draft_overall_pick", "draft_year", "combine_weight", "draft_round"]]).join(
        cr[["forty", "ngs_athleticism_score"]]).join(car)
    d["gear"] = 100 * d.nfl / d.comb
    d["lpick"] = np.log(d.draft_overall_pick.fillna(262))
    d["slotp"] = nfl_df.groupby("nfl_id").slot.mean().reindex(d.index)
    return d


BASE = ["lpick", "forty", "combine_weight", "ngs_athleticism_score"]


def design(df, cols, ref):
    x = df[cols].copy()
    for c in cols:
        x[c] = x[c].fillna(ref[c].mean())
    return sm.add_constant(x.astype(float), has_constant="add")


def loco(dd, y="yprr", base=BASE, metric="gear", extra=()):
    """Leave-one-draft-class-out: baseline forecast fitted on two classes; held-out miss regressed on the metric
    (standardised with training-class mean/sd). Returns slope, held-out baseline R2, held-out R2 with metric, misses."""
    cols = list(base) + list(extra)
    X = dd[cols].to_numpy(float)
    g = dd[metric].to_numpy(float)
    yv = dd[y].to_numpy(float)
    w = dd.routes.to_numpy(float)
    cls = dd.draft_year.to_numpy()
    pb, pf, zz = np.empty(len(yv)), np.empty(len(yv)), np.empty(len(yv))

    def wls(A, b, ww):
        sw = np.sqrt(ww)
        return np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)[0]

    for yr in np.unique(cls):
        tr, te = cls != yr, cls == yr
        mu = np.nanmean(X[tr], axis=0)
        Xtr, Xte = np.where(np.isnan(X[tr]), mu, X[tr]), np.where(np.isnan(X[te]), mu, X[te])
        gm, gs = g[tr].mean(), g[tr].std(ddof=1)
        ztr, zte = (g[tr] - gm) / gs, (g[te] - gm) / gs
        Atr = np.column_stack([np.ones(tr.sum()), Xtr])
        Ate = np.column_stack([np.ones(te.sum()), Xte])
        pb[te] = Ate @ wls(Atr, yv[tr], w[tr])
        pf[te] = np.column_stack([Ate, zte]) @ wls(np.column_stack([Atr, ztr]), yv[tr], w[tr])
        zz[te] = zte
    missv = yv - pb
    slope = wls(np.column_stack([np.ones(len(zz)), zz]), missv, w)[1]
    tss = np.sum(w * (yv - np.average(yv, weights=w)) ** 2)
    r2b = 1 - np.sum(w * (yv - pb) ** 2) / tss
    r2f = 1 - np.sum(w * (yv - pf) ** 2) / tss
    return slope, r2b, r2f, pd.Series(missv, index=dd.index), pd.Series(zz, index=dd.index)


def boot(dd, stat, n=NBOOT, pooled=False):
    out = []
    groups = [dd.index.values] if pooled else [g.index.values for _, g in dd.groupby("draft_year")]
    for _ in range(n):
        idx = np.concatenate([RNG.choice(g, len(g), replace=True) for g in groups])
        b = dd.loc[idx].reset_index()
        b.index = range(len(b))
        if b.draft_year.nunique() < 3:
            continue
        try:
            out.append(stat(b))
        except Exception:
            pass
    return np.nanpercentile(out, [5, 95]), np.array(out)


def fit_in(dd, cols, y="yprr"):
    return sm.WLS(dd[y], design(dd, cols, dd), weights=dd.routes).fit()


# ----------------------------------------------------------------------------------------------- core sample
D = build()
put("n_gear", len(D), "{:d}")
DD = D[(D.routes >= MIN_ROUTES)].dropna(subset=["yprr"]).copy()
DD["z"] = (DD.gear - DD.gear.mean()) / DD.gear.std()
put("n_main", len(DD), "{:d}")
put("gear_med", D.gear.median(), "{:.0f}")
put("gear_p10", D.gear.quantile(0.1), "{:.0f}")
put("gear_p90", D.gear.quantile(0.9), "{:.0f}")
put("gear_sd", DD.gear.std(), "{:.1f}")
put("plays_med", D.k.median(), "{:.0f}")
put("routes_med", DD.routes.median(), "{:.0f}")
put("yprr_mean", np.average(DD.yprr, weights=DD.routes), "{:.2f}")
put("n_go_drill", C[C.drill == "GO_ROUTE_RIGHT"].nfl_id.nunique(), "{:d}")
put("n_wr_vert", NV.nfl_id.nunique(), "{:d}")
# league curves
put("drill_v20", C[C.drill == "GO_ROUTE_RIGHT"]["v2.0"].mean(), "{:.1f}")
put("nfl_v20", N["v2.0"].mean(), "{:.1f}")
put("drill_v05", C[C.drill == "GO_ROUTE_RIGHT"]["v0.5"].mean(), "{:.1f}")
put("nfl_v05", N["v0.5"].mean(), "{:.1f}")
MPH = 3600 / 1760
put("drill_v20_mph", C[C.drill == "GO_ROUTE_RIGHT"]["v2.0"].mean() * MPH, "{:.1f}")
put("nfl_v20_mph", N["v2.0"].mean() * MPH, "{:.1f}")
put("n_lt100", int((D.routes.fillna(0) < MIN_ROUTES).sum()), "{:d}")

# reliability of Stem Gear: odd/even plays, Spearman-Brown
NV2 = NV.sort_values(["nfl_id", "game_id", "play_id"]).copy()
NV2["half"] = NV2.groupby("nfl_id").cumcount() % 2
h = [nfl_player(NV2[NV2.half == k]).nfl.rename(f"h{k}") for k in (0, 1)]
hh = pd.concat(h, axis=1).dropna().loc[lambda x: x.index.isin(D.index)]
r_half = np.corrcoef(hh.h0, hh.h1)[0, 1]
rel = 2 * r_half / (1 + r_half)
put("rel_gear", rel)
put("n_rel", len(hh), "{:d}")

# transfer: does the Combine part predict the NFL part?
rho_tr = spearmanr(D.comb, D.nfl)[0]
tb, _ = boot(D.assign(draft_year=D.draft_year), lambda b: spearmanr(b.comb, b.nfl)[0])
put("rho_transfer", rho_tr)
put("rho_transfer_lo", tb[0])
put("rho_transfer_hi", tb[1])
D40 = build(ref="FORTY_YARD_DASH")
put("rho_40_nfl", spearmanr(D40.comb, D40.nfl)[0])
put("n_40", len(D40), "{:d}")
# NFL top speed on the same sampled verticals vs official 40 (a speed that does carry)
top = N.groupby("nfl_id")[[f"v{t}" for t in GRID]].max().max(axis=1)
tt40 = pd.concat([top.rename("top"), cr.forty], axis=1, join="inner").dropna()
tt40 = tt40[tt40.index.isin(D40.index)]
put("rho_40_top", spearmanr(-tt40.forty, tt40.top)[0])
put("n_40_top", len(tt40), "{:d}")
# what Stem Gear tracks
put("rho_gear_comb", spearmanr(D.gear, D.comb)[0])
put("rho_gear_nfl", spearmanr(D.gear, D.nfl)[0])
put("rho_gear_pick", spearmanr(DD.gear, DD.lpick)[0])

# ----------------------------------------------------------------------------------------------- production tests
slope, r2b, r2f, miss, zz = loco(DD)
ci, bs = boot(DD, lambda b: loco(b)[0])
cip, _ = boot(DD, lambda b: loco(b)[0], pooled=True)
put("slope", slope)
put("slope_lo", ci[0])
put("slope_hi", ci[1])
put("slope_plo", cip[0])
put("slope_phi", cip[1])
put("p_neg", 100 * np.mean(bs < 0), "{:.0f}")
put("p_beyond_margin", 100 * np.mean(bs < -MARGIN), "{:.0f}")
put("margin", MARGIN)
put("r2_base", r2b)
put("r2_full", r2f)
put("slope_pct", 100 * abs(slope) / float(NUM["yprr_mean"]), "{:.0f}")
# per-class held-out slopes
for yr in sorted(DD.draft_year.unique()):
    m = DD.draft_year == yr
    b = sm.WLS(miss[m], sm.add_constant(zz[m]), weights=DD.routes[m]).fit().params.iloc[1]
    put(f"slope_{yr}", b)
    put(f"n_{yr}", int(m.sum()), "{:d}")
# baselines: pick only, in-sample
s_pick, r2b_pick, _, _, _ = loco(DD, base=["lpick"])
put("slope_pickonly", s_pick)
put("r2_pickonly", r2b_pick)
fi = fit_in(DD, BASE + ["z"])
put("slope_insample", fi.params["z"])
put("p_insample", fi.pvalues["z"], "{:.3f}")
raw = fit_in(DD, ["z"])
put("slope_raw", raw.params["z"])
# shuffle test of the held-out slope (permute gear within class)
perm = []
for _ in range(500):
    b = DD.copy()
    b["gear"] = b.groupby("draft_year").gear.transform(lambda x: RNG.permutation(x.values))
    perm.append(loco(b)[0])
put("p_shuffle", np.mean(np.array(perm) <= slope), "{:.2f}")


# robustness (held-out slope with class-stratified 90% interval)
def rob(label, dd, **kw):
    dd = dd.copy()
    s = loco(dd, **kw)[0]
    c, _ = boot(dd, lambda b: loco(b, **kw)[0], n=600)
    return dict(label=label, slope=s, lo=c[0], hi=c[1], n=len(dd))


def sub(nfl_df, col="w", ref="GO_ROUTE_RIGHT", min_plays=MIN_PLAYS):
    d = build(ref=ref, nfl_df=nfl_df, col=col, min_plays=min_plays)
    return d[d.routes >= MIN_ROUTES].dropna(subset=["yprr"])


R = [dict(label="Primary: all sampled verticals", slope=slope, lo=ci[0], hi=ci[1], n=len(DD))]
R.append(rob("Random half of the sample only", sub(NV[NV.kind == "random"], min_plays=2)))
R.append(rob("Top-EPA half only", sub(NV[NV.kind == "top_epa"], min_plays=2)))
R.append(rob("Untargeted plays only", sub(NV[~NV.target])))
R.append(rob("GO routes only", sub(NV[NV.route == "GO"], min_plays=2)))
R.append(rob("+ slot share and cushion in the baseline", DD, extra=("slotp", "cush")))
R.append(rob("Earlier window, 0.5-1.5 s", sub(N.dropna(subset=["w_early"]), col="w_early")))
R.append(rob("Official-40 clip as the reference", sub(NV, ref="FORTY_YARD_DASH")))
R.append(rob("NFL stem speed alone (no drill)", DD.assign(gear=DD.nfl)))
R.append(rob("Combine go-route speed alone", DD.assign(gear=-DD.comb)))
UNC = rob("Uncensored 1.5-3.0 s window (includes after the throw)", sub(N.dropna(subset=["w_uncens"]), col="w_uncens"))
put("rob_uncens", UNC["slope"])
put("rob_uncens_lo", UNC["lo"])
put("rob_uncens_hi", UNC["hi"])
d25 = DD[DD.draft_year < 2025]
put("slope_no2025", loco(d25)[0])
for r_ in R:
    print(f"{r_['label']:45s} {r_['slope']:+.2f} [{r_['lo']:+.2f}, {r_['hi']:+.2f}] n={r_['n']}")
keys = ["primary", "random", "topepa", "untargeted", "goonly", "slotcush", "early", "forty", "nflonly", "combonly"]
for k, r_ in zip(keys, R):
    put(f"rob_{k}", r_["slope"])
    put(f"rob_{k}_lo", r_["lo"])
    put(f"rob_{k}_hi", r_["hi"])
    put(f"rob_{k}_n", r_["n"], "{:d}")
put("n_rob_neg", sum(r_["slope"] < 0 for r_ in R[:8]), "{:d}")
# targets per route
st_t, _, _, _, _ = loco(DD, y="tprr")
ct, _ = boot(DD, lambda b: loco(b, y="tprr")[0], n=1000)
put("slope_tprr", 100 * st_t, "{:.1f}")
put("slope_tprr_lo", 100 * ct[0], "{:.1f}")
put("slope_tprr_hi", 100 * ct[1], "{:.1f}")

# ----------------------------------------------------------------------------------------------- next-season test
# Stem Gear from season t sampled verticals -> yards per route in t+1, beyond t yards per route and pick
cref = C[C.drill == "GO_ROUTE_RIGHT"].groupby("nfl_id").win_adj.mean().rename("comb")
ns = nfl_player(NV, by="season").reset_index()
ns = ns[ns.k >= 2].merge(cref.reset_index(), on="nfl_id")
ns["gear"] = 100 * ns.nfl / ns.comb
s_t = st[["nfl_id", "season", "routes", "rec_yards"]].copy()
s_t["yprr_t"] = s_t.rec_yards / s_t.routes
nxt = s_t.rename(columns={"routes": "routes_n", "yprr_t": "yprr_n"})[["nfl_id", "season", "routes_n", "yprr_n"]]
nxt["season"] -= 1
pair = ns.merge(s_t[["nfl_id", "season", "routes", "yprr_t"]], on=["nfl_id", "season"]).merge(nxt, on=["nfl_id", "season"])
pair = pair[(pair.routes >= 50) & (pair.routes_n >= 50)].copy()
pair["lpick"] = np.log(pair.nfl_id.map(pl.draft_overall_pick).fillna(262))
pair["z"] = (pair.gear - pair.gear.mean()) / pair.gear.std()
mp = sm.WLS(pair.yprr_n, sm.add_constant(pair[["z", "yprr_t", "lpick"]]), weights=pair.routes_n).fit()
cl_ids = pair.nfl_id.unique()
nb = []
for _ in range(NBOOT):
    ids = RNG.choice(cl_ids, len(cl_ids), replace=True)
    b = pd.concat([pair[pair.nfl_id == i] for i in ids])
    nb.append(sm.WLS(b.yprr_n, sm.add_constant(b[["z", "yprr_t", "lpick"]]), weights=b.routes_n).fit().params["z"])
put("next_slope", mp.params["z"])
put("next_lo", np.percentile(nb, 5))
put("next_hi", np.percentile(nb, 95))
put("next_n", len(pair), "{:d}")
put("next_nrec", pair.nfl_id.nunique(), "{:d}")

# ----------------------------------------------------------------------------------------------- named receivers
DD["miss"] = miss
NAMES = ["Ladd McConkey", "Rashee Rice", "Brian Thomas", "Tre Tucker", "Xavier Hutchinson"]
tab = DD.set_index("display_name").loc[NAMES]
named = {}
for nm, r_ in tab.iterrows():
    key = nm.split()[-1].lower()
    put(f"{key}_gear", r_.gear, "{:.0f}")
    put(f"{key}_drill", r_.comb, "{:.1f}")
    put(f"{key}_nfl", r_.nfl, "{:.1f}")
    put(f"{key}_yprr", r_.yprr, "{:.2f}")
    put(f"{key}_miss", r_.miss, "{:.2f}")
    put(f"{key}_k", int(r_.k), "{:d}")
    put(f"{key}_drill_mph", r_.comb * MPH, "{:.1f}")
    put(f"{key}_nfl_mph", r_.nfl * MPH, "{:.1f}")
    put(f"{key}_drop_mph", float(NUM[f"{key}_drill_mph"]) - float(NUM[f"{key}_nfl_mph"]), "{:.1f}")
    put(f"{key}_pick", "UDFA" if np.isnan(r_.draft_overall_pick) else f"{int(r_.draft_overall_pick)}")
    put(f"{key}_rank", int((DD.gear < r_.gear).sum() + 1), "{:d}")
    named[nm] = r_
# low-gear vs high-gear thirds, held-out miss
lo3 = DD.gear <= DD.gear.quantile(1 / 3)
hi3 = DD.gear >= DD.gear.quantile(2 / 3)
put("miss_low3", np.average(DD.miss[lo3], weights=DD.routes[lo3]), "{:.2f}")
put("miss_high3", np.average(DD.miss[hi3], weights=DD.routes[hi3]), "{:.2f}")
put("n_low3", int(lo3.sum()), "{:d}")
put("n_high3", int(hi3.sum()), "{:d}")
put("gear_low3", DD.gear.quantile(1 / 3), "{:.0f}")
put("gear_high3", DD.gear.quantile(2 / 3), "{:.0f}")
put("beat_low3", 100 * np.mean(DD.miss[lo3] > 0), "{:.0f}")
put("beat_high3", 100 * np.mean(DD.miss[hi3] > 0), "{:.0f}")

# ----------------------------------------------------------------------------------------------- figures
# Fig 1: speed curves, drill vs NFL, league + four named receivers
fig, axes = plt.subplots(1, 5, figsize=(FIGW, 3.9), sharey=True)
go = C[C.drill == "GO_ROUTE_RIGHT"]


def curves(ax, cd, nd, title):
    ax.axvspan(*WIN, color="#EEEEEE", zorder=0)
    m = cd[VC].mean().values
    ax.plot(GRID, m, color=OI["orange"], lw=2.6, label="Combine go-route drill")
    nm_ = nd[VC].mean().values
    q1, q3 = nd[VC].quantile(0.25).values, nd[VC].quantile(0.75).values
    ax.fill_between(GRID, q1, q3, color=OI["blue"], alpha=0.18, lw=0)
    ax.plot(GRID, nm_, color=OI["blue"], lw=2.6, label="NFL vertical stems (before the throw)")
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("s after release")
    ax.set_xticks([0, 1, 2, 3])


curves(axes[0], go, N, f"All WRs\n{NUM['n_go_drill']} drill / {NUM['n_wr_vert']} NFL receivers")
for ax, nm in zip(axes[1:], ["Ladd McConkey", "Rashee Rice", "Tre Tucker", "Xavier Hutchinson"]):
    pid = pl.index[pl.display_name == nm][0]
    k = nm.split()[-1].lower()
    ax.set_title("")
    curves(ax, go[go.nfl_id == pid], N[N.nfl_id == pid],
           f"{nm}\nGear {NUM[k + '_gear']}%, {NUM[k + '_yprr']} yds/route")
axes[0].set_ylabel("speed, yd/s")
axes[0].set_ylim(0, 10.5)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "fig1_curves.png", dpi=150)
plt.close(fig)

# Fig 2: held-out miss vs Stem Gear
fig, axes = plt.subplots(1, 2, figsize=(FIGW, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
sz = 10 + DD.routes / 12
ax.scatter(DD.gear, DD.miss, s=sz, color=OI["sky"], edgecolor=OI["blue"], alpha=0.75)
xs = np.linspace(DD.gear.min(), DD.gear.max(), 50)
fb = []
sdg = DD.gear.std()
mg = DD.gear.mean()
for b_ in bs[:1000]:
    fb.append(b_ * (xs - mg) / sdg)
fb = np.array(fb)
icpt = np.average(DD.miss, weights=DD.routes)
ax.fill_between(xs, icpt + np.percentile(fb, 5, 0), icpt + np.percentile(fb, 95, 0), color=OI["orange"], alpha=0.25, lw=0)
ax.plot(xs, icpt + slope * (xs - mg) / sdg, color=OI["verm"], lw=2.2)
ax.axhline(0, color=OI["grey"], lw=0.8)
for nm in NAMES + ["Puka Nacua"]:
    r_ = DD[DD.display_name == nm].iloc[0]
    ax.scatter([r_.gear], [r_.miss], s=10 + r_.routes / 12, facecolor="none", edgecolor="black", lw=1.4, zorder=5)
    ax.annotate(nm.split()[-1], (r_.gear, r_.miss), xytext=(4, 4), textcoords="offset points", fontsize=9)
ax.set_xlabel("Stem Gear, % of own Combine go-route speed used 1-2 s after release")
ax.set_ylabel("yards per route vs no-tracking forecast\n(held-out draft class)")
ax.set_title(f"A. {NUM['n_main']} WRs with 100+ career routes; dot size = routes", loc="left", fontsize=10)
ax = axes[1]
lab = ["Low third", "Middle third", "High third"]
mid3 = ~(lo3 | hi3)
vals = [np.average(DD.miss[m], weights=DD.routes[m]) for m in (lo3, mid3, hi3)]
cis = []
for m in (lo3, mid3, hi3):
    sub_ = DD[m]
    bb = [np.average(sub_.miss.values[i], weights=sub_.routes.values[i])
          for i in (RNG.integers(0, len(sub_), len(sub_)) for _ in range(1000))]
    cis.append(np.percentile(bb, [5, 95]))
cols = [OI["blue"], OI["light"], OI["verm"]]
for i, (v, c_, col) in enumerate(zip(vals, cis, cols)):
    ax.bar(i, v, color=col, width=0.6)
    ax.plot([i, i], c_, color="black", lw=1.4)
    ax.text(i + 0.33, v / 2, f"{v:+.2f}", ha="left", va="center", fontsize=9)
ax.axhline(0, color=OI["grey"], lw=0.8)
ax.set_xticks(range(3))
ax.set_xticklabels([f"{l}\n(n={int(m.sum())})" for l, m in zip(lab, (lo3, mid3, hi3))])
ax.set_ylabel("route-weighted mean miss, yds/route")
ax.set_title("B. By Stem Gear third, 90% bootstrap", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig2_production.png", dpi=150)
plt.close(fig)

# Fig 3: forest of robustness
R = R + [UNC]
fig, ax = plt.subplots(figsize=(FIGW, 4.8))
labels = [f"{r_['label']} (n={r_['n']})" for r_ in R]
y = np.arange(len(R))[::-1]
ax.axvspan(-MARGIN, MARGIN, color="#E5E5E5", zorder=0, label=f"+/-{MARGIN:.2f} practical margin")
for yi, r_, i in zip(y, R, range(len(R))):
    col = OI["verm"] if i == 0 else (OI["grey"] if i >= 8 else OI["blue"])
    ax.plot([r_["lo"], r_["hi"]], [yi, yi], color=col, lw=2.2)
    ax.plot(r_["slope"], yi, "o", color=col, ms=7)
ax.axvline(0, color="black", lw=0.8)
ax.axhline(2.5, color=OI["light"], lw=0.8, ls="--")
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("held-out slope, yds/route per sd beyond no-tracking forecast (90% bootstrap)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig3_robust.png", dpi=150)
plt.close(fig)

# inputs + numbers
DD[["display_name", "draft_year", "draft_overall_pick", "k", "comb", "nfl", "gear", "routes", "yprr", "tprr", "miss"]].round(4).to_csv(OUT / "inputs.csv")
NUM.update(CONST)
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1))
print(json.dumps(NUM, indent=1))
