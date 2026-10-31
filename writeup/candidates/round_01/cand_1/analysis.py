"""Hoop G: lateral acceleration on the Combine Run-the-Hoop drill vs NFL pass-rush outcomes (DL/EDGE).

Run from the repo root:
    uv run python writeup/candidates/round_01/cand_1/analysis.py

Reads data/processed/*.parquet only. Writes numbers.json and figures/*.png next to this file.
Every decimal quoted in writeup.md is produced here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
PROC = ROOT / "data" / "processed"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(2027)
N_BOOT = 2000
N_PERM = 500
MIN_SNAPS = 100
UDFA_PICK = 262
YD_PER_G = 9.80665 / 0.9144  # 1 g in yd/s^2
DT = 0.1
# Okabe-Ito
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "black": "#000000", "grey": "#999999"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 150, "figure.dpi": 150})
FIGW = 1600 / 150  # inches -> 1600 px at 150 dpi

NUM: dict = {}
ABS: set = set()


def put(key: str, v, nd: int = 2):
    """Store a number formatted exactly as it will appear in the text."""
    if isinstance(v, (int, np.integer)):
        NUM[key] = int(v)
        return int(v)
    s = f"{v:.{nd}f}"
    NUM[key] = s
    if s.startswith("-"):
        ABS.add(s[1:])
    return s


def savefig(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches=None)
    plt.close(fig)


# ----------------------------------------------------------------------------- per-attempt kinematics
def attempt_features(g: pd.DataFrame) -> dict | None:
    x, y = g.x.to_numpy(), g.y.to_numpy()
    n = len(x)
    if n < 15:
        return None
    xs, ys = savgol_filter(x, 7, 2), savgol_filter(y, 7, 2)
    vx, vy = savgol_filter(xs, 7, 2, deriv=1, delta=DT), savgol_filter(ys, 7, 2, deriv=1, delta=DT)
    ax, ay = savgol_filter(xs, 7, 2, deriv=2, delta=DT), savgol_filter(ys, 7, 2, deriv=2, delta=DT)
    v = np.hypot(vx, vy)
    alat = np.abs(vx * ay - vy * ax) / np.maximum(v, 0.5)  # centripetal accel, yd/s^2
    roll = pd.Series(alat).rolling(5, center=True).mean().to_numpy()  # sustained over 0.5 s
    disp = np.hypot(x - x[0], y - y[0])
    k = int(np.nanargmax(roll))
    half = n // 2
    t3 = np.argmax(disp >= 3) * DT if disp.max() >= 3 else np.nan
    return {"lat_peak": np.nanmax(roll) / YD_PER_G, "lat_peak_h1": np.nanmax(roll[:half]) / YD_PER_G,
            "lat_peak_h2": np.nanmax(roll[half:]) / YD_PER_G, "speed_at_peak": v[k], "t_0_3yd": t3,
            "_alat": alat / YD_PER_G, "_x": xs, "_y": ys, "_k": k}


def drill_table(ev: pd.DataFrame, frames: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, traces = [], {}
    for eid, g in frames.groupby("event_id", sort=False):
        f = attempt_features(g)
        if f is None:
            continue
        traces[eid] = {k: f.pop(k) for k in ["_alat", "_x", "_y", "_k"]}
        rows.append({"event_id": eid, **f})
    return pd.DataFrame(rows).merge(ev, on="event_id"), traces


def yz(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Year-standardize within the DL group (drill medians shift between Combines)."""
    out = df.copy()
    for c in cols:
        out[c + "_z"] = df.groupby("draft_year")[c].transform(lambda s: (s - s.mean()) / s.std())
    return out


# ----------------------------------------------------------------------------- load
players = pd.read_parquet(PROC / "players.parquet")
cres = pd.read_parquet(PROC / "combine_results.parquet")
events = pd.read_parquet(PROC / "combine_events.parquet")
seasons = pd.read_parquet(PROC / "player_season_stats.parquet")

dl = players[players.position_group == "DL"][["nfl_id", "display_name", "draft_year", "nfl_position",
                                               "draft_overall_pick", "draft_round"]].copy()
dl["pick"] = dl.draft_overall_pick.fillna(UDFA_PICK)
dl["log_pick"] = np.log(dl.pick)
dl = dl.merge(cres[["nfl_id", "combine_weight", "ten_yd_split", "forty", "three_cone", "short_shuttle",
                    "ngs_athleticism_score"]], on="nfl_id", how="left")
put("n_dl", len(dl))

ev = events[(events.position_group == "DL") & events.drill_name.isin(["PASS_RUSH_DRILL", "RUN_THE_HOOP_DRILL"])]
ev = ev[["event_id", "nfl_id", "draft_year", "drill_name", "attempt_no", "duration_s", "mean_speed", "peak_accel"]]
frames = pd.read_parquet(PROC / "combine_frames.parquet", filters=[("event_id", "in", list(ev.event_id))],
                         columns=["event_id", "frame", "entity_type", "x", "y", "s"])
frames = frames[frames.entity_type == "PLAYER"].sort_values(["event_id", "frame"])
att, traces = drill_table(ev, frames)
hoop_att = att[att.drill_name == "RUN_THE_HOOP_DRILL"]
rush_att = att[att.drill_name == "PASS_RUSH_DRILL"]
put("n_hoop_attempts", len(hoop_att))
put("n_hoop_players", hoop_att.nfl_id.nunique())
put("n_rush_attempts", len(rush_att))
put("hoop_g_median", float(hoop_att.lat_peak.median()))
put("hoop_g_p10", float(hoop_att.lat_peak.quantile(0.1)))
put("hoop_g_p90", float(hoop_att.lat_peak.quantile(0.9)))
put("hoop_speed_at_peak_median", float(hoop_att.speed_at_peak.median()), 1)

# ---- reliability
r_sh = stats.spearmanr(hoop_att.lat_peak_h1, hoop_att.lat_peak_h2)[0]
put("hoop_split_half_rho", r_sh)
put("hoop_split_half_sb", 2 * r_sh / (1 + r_sh))
two = hoop_att[hoop_att.attempt_no <= 2].pivot_table(index="nfl_id", columns="attempt_no", values="lat_peak").dropna()
put("hoop_two_attempt_n", len(two))
rp = rush_att[rush_att.attempt_no <= 2].pivot_table(index="nfl_id", columns="attempt_no",
                                                    values=["lat_peak", "peak_accel", "t_0_3yd"]).dropna()
put("rush_two_attempt_n", len(rp))
for c in ["lat_peak", "peak_accel", "t_0_3yd"]:
    put(f"rush_retest_rho_{c}", stats.spearmanr(rp[(c, 1)], rp[(c, 2)])[0])
yr_med = hoop_att.groupby("draft_year").lat_peak.median()
put("hoop_g_year_range", float(yr_med.max() - yr_med.min()))

# ---- player-level drill metrics (mean across attempts, then year-z)
hoop = hoop_att.groupby(["nfl_id", "draft_year"]).agg(hoop_g=("lat_peak", "mean"), hoop_speed=("mean_speed", "mean"),
                                                       hoop_time=("duration_s", "mean")).reset_index()
rush = rush_att.groupby(["nfl_id", "draft_year"]).agg(rush_bend_g=("lat_peak", "mean"), rush_peak_accel=("peak_accel", "mean"),
                                                       rush_t3=("t_0_3yd", "mean")).reset_index()
hoop = yz(hoop, ["hoop_g", "hoop_speed", "hoop_time"])
rush = yz(rush, ["rush_bend_g", "rush_peak_accel", "rush_t3"])
dl = yz(dl, ["ten_yd_split", "three_cone", "combine_weight"])
dl = dl.merge(hoop.drop(columns="draft_year"), on="nfl_id", how="left").merge(rush.drop(columns="draft_year"), on="nfl_id", how="left")

# ---- NFL outcomes (regular season; player_season_stats is built from games.csv REG filter)
s = seasons[seasons.position_group == "DL"].copy()
s["w_go"] = s.pass_rush_snaps * s.avg_get_off
agg = s.groupby("nfl_id").agg(rush_snaps=("pass_rush_snaps", "sum"), pressures=("pressures", "sum"),
                              w_go=("w_go", "sum"), n_seasons=("season", "count"),
                              align=("main_alignment", lambda v: v.mode().iloc[0] if v.notna().any() else None))
agg["press_rate"] = agg.pressures / agg.rush_snaps
agg["get_off"] = agg.w_go / agg.rush_snaps
dl = dl.merge(agg.reset_index(), on="nfl_id", how="left")
dl["edge"] = (dl["align"] == "EDGE").astype(float)
dl.loc[dl["align"].isna(), "edge"] = dl.loc[dl["align"].isna(), "nfl_position"].isin(["DE", "OLB"]).astype(float)

# outcome reliability: season-to-season pressure rate (>= MIN_SNAPS both seasons)
ss = s[s.pass_rush_snaps >= MIN_SNAPS][["nfl_id", "season", "pressures", "pass_rush_snaps"]].copy()
ss["pr"] = ss.pressures / ss.pass_rush_snaps
pairs = ss.merge(ss.assign(season=ss.season - 1), on=["nfl_id", "season"], suffixes=("", "_next"))
put("outcome_yoy_rho", stats.spearmanr(pairs.pr, pairs.pr_next)[0])
put("outcome_yoy_n", len(pairs))

# ---- baseline features: no tracking. impute missing official times from weight + alignment (no outcome used)
for c in ["ten_yd_split_z", "three_cone_z"]:
    m = dl[c].notna()
    put(f"n_missing_{c.split('_z')[0]}", int((~m).sum()))
    fit = sm.OLS(dl.loc[m, c], sm.add_constant(dl.loc[m, ["combine_weight_z", "edge"]])).fit()
    dl[c + "_imp"] = dl[c].fillna(fit.predict(sm.add_constant(dl[["combine_weight_z", "edge"]])))
BASE = ["combine_weight_z", "edge", "ten_yd_split_z_imp", "three_cone_z_imp", "ngs_athleticism_score", "log_pick"]

cohort = dl[(dl.rush_snaps >= MIN_SNAPS) & dl.hoop_g_z.notna()].copy()
put("n_cohort", len(cohort))
put("n_cohort_edge", int(cohort.edge.sum()))
put("n_cohort_int", int((1 - cohort.edge).sum()))
put("n_with_hoop_dl", int(dl.hoop_g_z.notna().sum()))
put("n_rush_100", int((dl.rush_snaps >= MIN_SNAPS).sum()))
put("cohort_rush_snaps_median", int(cohort.rush_snaps.median()))
put("cohort_press_rate_median_pct", 100 * cohort.press_rate.median(), 1)
for y, n in cohort.groupby("draft_year").size().items():
    put(f"n_cohort_{y}", int(n))

# ---- raw correlations
def sp(a, b):
    m = a.notna() & b.notna()
    r, p = stats.spearmanr(a[m], b[m])
    return r, p, int(m.sum())


r, p, n = sp(cohort.hoop_g_z, cohort.press_rate)
put("rho_hoop_press", r); put("p_hoop_press", p, 3)
r, p, n = sp(cohort.ten_yd_split_z, -cohort.press_rate)
put("rho_10yd_press", r); put("n_10yd_press", n)
r, p, n = sp(cohort.three_cone_z, -cohort.press_rate)
put("rho_3cone_press", r); put("n_3cone_press", n)
for g, lab in [(1.0, "edge"), (0.0, "int")]:
    c = cohort[cohort.edge == g]
    r, p, n = sp(c.hoop_g_z, c.press_rate)
    put(f"rho_hoop_press_{lab}", r)
r, p, n = sp(dl.hoop_g_z, -dl.log_pick)
put("rho_hoop_pick", r); put("n_hoop_pick", n); put("p_hoop_pick", p)
r, p, n = sp(dl.hoop_g_z, dl.combine_weight)
put("rho_hoop_weight", r)
r, p, n = sp(dl.hoop_g_z, -dl.ten_yd_split_z)
put("rho_hoop_10yd", r); put("n_hoop_10yd", n)
r, p, n = sp(cohort.combine_weight, cohort.get_off)
put("rho_weight_getoff", r)
r, p, n = sp(cohort.ten_yd_split_z, cohort.get_off)
put("rho_10yd_getoff", r); put("n_10yd_getoff", n)
r, p, n = sp(cohort.rush_peak_accel_z, -cohort.get_off)
put("rho_rushaccel_getoff_raw", r); put("n_rushaccel_getoff", n)


# ---- partial correlations after the no-tracking baseline, bootstrap CI
def partial(df, x, y, base=BASE):
    d = df.dropna(subset=[x, y] + base)
    Z = sm.add_constant(d[base])
    rx = sm.OLS(d[x], Z).fit().resid
    ry = sm.OLS(d[y], Z).fit().resid
    return stats.spearmanr(rx, ry)[0], len(d)


def partial_boot(df, x, y, base=BASE):
    d = df.dropna(subset=[x, y] + base).reset_index(drop=True)
    est, n = partial(d, x, y, base)
    bs = []
    for _ in range(N_BOOT):
        b = d.iloc[RNG.integers(0, len(d), len(d))]
        bs.append(partial(b, x, y, base)[0])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return est, lo, hi, n


cohort["neg_get_off"] = -cohort.get_off
METRICS = [("hoop_g_z", "Hoop G (peak lateral g, hoop)"), ("hoop_speed_z", "Hoop mean speed"),
           ("hoop_time_z", "Hoop time (lower = faster)"), ("rush_bend_g_z", "Rush-drill bend g (at bag)"),
           ("rush_peak_accel_z", "Rush-drill peak accel"), ("rush_t3_z", "Rush-drill time to 3 yd")]
forest = []
for col, lab in METRICS:
    sign = -1 if col in ("hoop_time_z", "rush_t3_z") else 1
    cohort["_m"] = sign * cohort[col]
    for out, olab in [("press_rate", "Pressure rate"), ("neg_get_off", "In-game get-off (faster)")]:
        e, lo, hi, n = partial_boot(cohort, "_m", out)
        forest.append({"metric": lab, "col": col, "outcome": olab, "rho": e, "lo": lo, "hi": hi, "n": n})
forest = pd.DataFrame(forest)
hp = forest[(forest.col == "hoop_g_z") & (forest.outcome == "Pressure rate")].iloc[0]
put("partial_hoop_press", hp.rho); put("partial_hoop_press_lo", hp.lo); put("partial_hoop_press_hi", hp.hi)
put("partial_hoop_press_n", int(hp.n))
ra = forest[(forest.col == "rush_peak_accel_z") & (forest.outcome.str.startswith("In-game"))].iloc[0]
put("partial_rushaccel_getoff", ra.rho); put("partial_rushaccel_getoff_lo", ra.lo); put("partial_rushaccel_getoff_hi", ra.hi)
put("forest_n_tests", len(forest))
put("forest_n_ci_excl_zero", int(((forest.lo > 0) | (forest.hi < 0)).sum()))
put("forest_max_abs_other", float(forest[~((forest.col == "hoop_g_z") & (forest.outcome == "Pressure rate"))].rho.abs().max()))
bonf_z = stats.norm.ppf(1 - 0.025 / len(forest))

# effect size: OLS coefficient of Hoop G (z) on pressure rate with baseline
d = cohort.dropna(subset=BASE + ["hoop_g_z"])
ols = sm.OLS(d.press_rate, sm.add_constant(d[BASE + ["hoop_g_z"]])).fit(cov_type="HC1")
put("ols_hoop_pp", 100 * ols.params["hoop_g_z"]); put("ols_hoop_pp_lo", 100 * ols.conf_int().loc["hoop_g_z", 0])
put("ols_hoop_pp_hi", 100 * ols.conf_int().loc["hoop_g_z", 1]); put("ols_hoop_p", ols.pvalues["hoop_g_z"], 3)
ols_b = sm.OLS(d.press_rate, sm.add_constant(d[BASE])).fit()
put("insample_r2_base", ols_b.rsquared); put("insample_r2_full", ols.rsquared)
put("ols_pick_pp", 100 * ols.params["log_pick"])
# pressures over a 400-rush-snap season for +1 sd
put("pressures_per_400", 4 * ols.params["hoop_g_z"] * 100, 1)


# ---- out-of-class validation: leave-one-class-out ridge, baseline vs baseline + Hoop G
def loco(df, feats, ycol, perm_col=None, rng=None):
    pred = pd.Series(np.nan, index=df.index)
    work = df.copy()
    if perm_col is not None:
        work[perm_col] = work.groupby("draft_year")[perm_col].transform(lambda v: rng.permutation(v.to_numpy()))
    for yr in sorted(work.draft_year.unique()):
        tr, te = work.draft_year != yr, work.draft_year == yr
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
        mdl.fit(work.loc[tr, feats], work.loc[tr, ycol])
        pred[te] = mdl.predict(work.loc[te, feats])
    return pred


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


val = cohort.dropna(subset=BASE + ["hoop_g_z", "press_rate"]).copy()
pb = loco(val, BASE, "press_rate")
pf = loco(val, BASE + ["hoop_g_z"], "press_rate")
y = val.press_rate
R2b, R2f = r2(y, pb), r2(y, pf)
put("oos_r2_base", R2b); put("oos_r2_full", R2f); put("oos_r2_delta", R2f - R2b)
put("oos_rho_base", stats.spearmanr(pb, y)[0]); put("oos_rho_full", stats.spearmanr(pf, y)[0])
pk = loco(val, ["log_pick"], "press_rate")
put("oos_r2_pick", r2(y, pk))
pfh = loco(val, ["hoop_g_z", "edge", "combine_weight_z"], "press_rate")
put("oos_r2_hoop_only", r2(y, pfh))
boots = []
idx = np.arange(len(val))
for _ in range(N_BOOT):
    b = RNG.integers(0, len(val), len(val))
    yb = y.to_numpy()[b]
    boots.append(r2(yb, pf.to_numpy()[b]) - r2(yb, pb.to_numpy()[b]))
put("oos_r2_delta_lo", np.percentile(boots, 2.5)); put("oos_r2_delta_hi", np.percentile(boots, 97.5))
put("oos_delta_boot_pos_share_pct", 100 * np.mean(np.array(boots) > 0), 0)
perm = []
for _ in range(N_PERM):
    pp = loco(val, BASE + ["hoop_g_z"], "press_rate", perm_col="hoop_g_z", rng=RNG)
    perm.append(r2(y, pp) - R2b)
perm = np.array(perm)
put("perm_p", (1 + np.sum(perm >= R2f - R2b)) / (1 + N_PERM), 3)
put("n_perm", N_PERM)
by_class = []
for yr in sorted(val.draft_year.unique()):
    m = val.draft_year == yr
    by_class.append({"year": int(yr), "n": int(m.sum()),
                     "rho_base": stats.spearmanr(pb[m], y[m])[0], "rho_full": stats.spearmanr(pf[m], y[m])[0]})
    put(f"oos_rho_base_{yr}", by_class[-1]["rho_base"]); put(f"oos_rho_full_{yr}", by_class[-1]["rho_full"])
by_class = pd.DataFrame(by_class)

# same test for in-game get-off with the rush-drill burst metric (the "known signal")
vg = cohort.dropna(subset=BASE + ["rush_peak_accel_z", "get_off"]).copy()
gb = loco(vg, BASE, "get_off"); gf = loco(vg, BASE + ["rush_peak_accel_z"], "get_off")
put("go_oos_r2_base", r2(vg.get_off, gb)); put("go_oos_r2_full", r2(vg.get_off, gf))
put("go_oos_r2_delta", r2(vg.get_off, gf) - r2(vg.get_off, gb), 3)
gw = loco(vg, ["combine_weight_z", "edge"], "get_off")
put("go_oos_r2_weight_only", r2(vg.get_off, gw))

# sensitivity: snap thresholds
for thr in (50, 200):
    c2 = dl[(dl.rush_snaps >= thr) & dl.hoop_g_z.notna()].copy()
    e, n = partial(c2, "hoop_g_z", "press_rate")
    put(f"sens_partial_{thr}", e); put(f"sens_n_{thr}", n)
# sensitivity: drop the 2025 class (one NFL season only)
c3 = cohort[cohort.draft_year < 2025]
e, n = partial(c3, "hoop_g_z", "press_rate")
put("sens_partial_no2025", e); put("sens_n_no2025", n)

# ---- named players
cohort["hoop_g"] = cohort.hoop_g
cohort["resid_base"] = cohort.press_rate - sm.OLS(cohort.press_rate, sm.add_constant(cohort[BASE])).fit().fittedvalues
top = cohort.sort_values("hoop_g_z", ascending=False)
rows = []
for _, r_ in pd.concat([top.head(5), top.tail(3)]).iterrows():
    rows.append({"name": r_.display_name, "class": int(r_.draft_year), "role": "EDGE" if r_.edge == 1 else "IDL",
                 "pick": "UDFA" if r_.pick == UDFA_PICK else str(int(r_.pick)), "hoop_g": f"{r_.hoop_g:.2f}",
                 "hoop_z": f"{r_.hoop_g_z:+.1f}", "press": f"{100 * r_.press_rate:.1f}",
                 "resid": f"{100 * r_.resid_base:+.1f}", "snaps": int(r_.rush_snaps)})
NUM["players_table"] = rows
for i, r_ in enumerate(rows):
    put(f"tbl_{i}_g", float(r_["hoop_g"]))
# market-miss: high hoop g (z>=1) drafted after pick 100 or UDFA, with above-median pressure
mm = cohort[(cohort.hoop_g_z >= 1) & (cohort.pick > 100)]
NUM["market_miss"] = [{"name": r_.display_name, "pick": int(r_.pick), "press_pct": round(100 * r_.press_rate, 1),
                       "z": round(r_.hoop_g_z, 2)} for _, r_ in mm.iterrows()]
put("n_market_miss", len(mm))
put("market_miss_press_median_pct", 100 * mm.press_rate.median(), 1)
put("cohort_pick_gt100_press_median_pct", 100 * cohort[cohort.pick > 100].press_rate.median(), 1)
# 2025 rookies with the trait but < MIN_SNAPS (watch list)
watch = dl[(dl.draft_year == 2025) & (dl.hoop_g_z >= 1)].sort_values("hoop_g_z", ascending=False)
NUM["watch_2025"] = [{"name": r_.display_name, "pick": int(r_.pick), "z": round(r_.hoop_g_z, 2),
                      "rush_snaps": None if pd.isna(r_.rush_snaps) else int(r_.rush_snaps)} for _, r_ in watch.iterrows()]

# ============================================================================= figures
# Fig 1: hoop traces, highest vs lowest Hoop G in the cohort
hi_id, lo_id = top.iloc[0].nfl_id, top.iloc[-1].nfl_id
fig, axes = plt.subplots(1, 2, figsize=(FIGW, 4.9), constrained_layout=True)
vmax = max(np.nanmax(traces[e]["_alat"]) for e in hoop_att[hoop_att.nfl_id.isin([hi_id, lo_id])].event_id)
for ax, pid in zip(axes, [hi_id, lo_id]):
    r_ = cohort[cohort.nfl_id == pid].iloc[0]
    eid = hoop_att[hoop_att.nfl_id == pid].sort_values("lat_peak", ascending=False).event_id.iloc[0]
    tr = traces[eid]
    x0, y0 = tr["_x"][0], tr["_y"][0]
    sc = ax.scatter(tr["_x"] - x0, tr["_y"] - y0, c=tr["_alat"], cmap="cividis", vmin=0, vmax=vmax, s=28)
    k = tr["_k"]
    ax.scatter(tr["_x"][k - 2:k + 3] - x0, tr["_y"][k - 2:k + 3] - y0, facecolors="none", edgecolors=OI["verm"], s=120, lw=1.6,
               label="peak 0.5 s window")
    ax.set_aspect("equal")
    ax.set_title(f"{r_.display_name} ({'EDGE' if r_.edge else 'IDL'}, {int(r_.combine_weight)} lb)\n"
                 f"Hoop G {r_.hoop_g:.2f} g  |  NFL pressure rate {100 * r_.press_rate:.1f}%", fontsize=11)
    ax.set_xlabel("yards"); ax.set_ylabel("yards")
    ax.annotate("start", (0, 0), xytext=(10, 8), textcoords="offset points", fontsize=9)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
cb = fig.colorbar(sc, ax=axes, shrink=0.85)
cb.set_label("lateral (centripetal) acceleration, g")
savefig(fig, "fig1_hoop_traces.png")

# Fig 2: Hoop G vs pressure rate, and vs draft pick
fig, axes = plt.subplots(1, 2, figsize=(FIGW, 4.8), constrained_layout=True)
ax = axes[0]
for g, col, mk, lab in [(1.0, OI["blue"], "o", "EDGE"), (0.0, OI["orange"], "s", "Interior DL")]:
    c = cohort[cohort.edge == g]
    ax.scatter(c.hoop_g_z, 100 * c.press_rate, c=col, marker=mk, s=38, alpha=0.85, label=f"{lab} (n={len(c)})", edgecolor="white", lw=0.5)
for _, r_ in pd.concat([top.head(4), top.tail(2)]).iterrows():
    ax.annotate(r_.display_name.split(" ")[-1], (r_.hoop_g_z, 100 * r_.press_rate), fontsize=8.5, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("Hoop G (z within Combine class)")
ax.set_ylabel("NFL pressure rate, % of pass-rush snaps")
ax.set_title(f"Hoop G vs pressure rate: Spearman ρ = {NUM['rho_hoop_press']}", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax = axes[1]
dd = dl.dropna(subset=["hoop_g_z"])
ax.scatter(dd.hoop_g_z, dd.pick, c=np.where(dd.edge == 1, OI["blue"], OI["orange"]), s=26, alpha=0.75, edgecolor="white", lw=0.4)
ax.set_yscale("log"); ax.invert_yaxis()
ax.set_yticks([1, 10, 32, 100, 262]); ax.set_yticklabels(["1", "10", "32", "100", "UDFA"])
ax.set_xlabel("Hoop G (z within Combine class)"); ax.set_ylabel("Draft pick (log scale)")
ax.set_title(f"The draft did not price it: ρ = {NUM['rho_hoop_pick']} (n={NUM['n_hoop_pick']})", fontsize=11)
savefig(fig, "fig2_hoop_vs_outcomes.png")

# Fig 3: forest of partial correlations after baseline
fig, ax = plt.subplots(figsize=(FIGW, 5.0), constrained_layout=True)
labs = [m[1] for m in METRICS]
for j, (out, col, off) in enumerate([("Pressure rate", OI["verm"], -0.15), ("In-game get-off (faster)", OI["sky"], 0.15)]):
    f = forest[forest.outcome == out].set_index("metric").loc[labs]
    ypos = np.arange(len(labs))[::-1] + off
    ax.errorbar(f.rho, ypos, xerr=[f.rho - f.lo, f.hi - f.rho], fmt="o" if j == 0 else "D", color=col, capsize=3, ms=6, label=out)
ax.axvline(0, color=OI["grey"], lw=1)
ax.set_yticks(np.arange(len(labs))[::-1]); ax.set_yticklabels(labs)
ax.set_xlabel("Partial Spearman ρ after baseline (weight, alignment, 10-yd split, 3-cone, NGS athleticism, draft pick)\npositive = better drill number goes with better NFL outcome; bars = 95% bootstrap CI")
ax.legend(frameon=False, loc="lower right")
ax.set_title(f"Six candidate drill metrics x two outcomes (n={NUM['partial_hoop_press_n']} DL, ≥{MIN_SNAPS} pass-rush snaps)", fontsize=11)
savefig(fig, "fig4_partial_forest.png")

# Fig 4: out-of-class validation
fig, axes = plt.subplots(1, 2, figsize=(FIGW, 4.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
xs = np.arange(len(by_class))
ax.bar(xs - 0.18, by_class.rho_base, 0.36, color=OI["grey"], label="Baseline (no tracking)")
ax.bar(xs + 0.18, by_class.rho_full, 0.36, color=OI["verm"], label="Baseline + Hoop G")
ax.set_xticks(xs); ax.set_xticklabels([f"{r_.year} class\n(n={r_.n})" for r_ in by_class.itertuples()])
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Spearman ρ, prediction vs actual\n(class held out of training)")
ax.set_title("Leave-one-class-out pressure-rate prediction", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax = axes[1]
ax.hist(perm, bins=30, color=OI["sky"], alpha=0.9, label=f"Hoop G shuffled within class ({N_PERM}x)")
ax.axvline(R2f - R2b, color=OI["verm"], lw=2.2, label=f"observed ΔR² = {NUM['oos_r2_delta']}")
ax.set_xlabel("Out-of-class ΔR² vs baseline"); ax.set_ylabel("count")
ax.set_title(f"Permutation p = {NUM['perm_p']}", fontsize=11)
ax.legend(frameon=False, fontsize=9)
savefig(fig, "fig3_out_of_class.png")

# Fig 5: why get-off is a mass story
fig, ax = plt.subplots(figsize=(FIGW, 4.4), constrained_layout=True)
for g, col, mk, lab in [(1.0, OI["blue"], "o", "EDGE"), (0.0, OI["orange"], "s", "Interior DL")]:
    c = cohort[cohort.edge == g]
    ax.scatter(c.combine_weight, c.get_off, c=col, marker=mk, s=36, alpha=0.85, label=lab, edgecolor="white", lw=0.5)
ax.set_xlabel("Combine weight, lb"); ax.set_ylabel("NFL average get-off, s (lower = faster)")
ax.set_title(f"In-game get-off tracks body mass (ρ = {NUM['rho_weight_getoff']}); the rush-drill burst adds ΔR² = {NUM['go_oos_r2_delta']} out of class", fontsize=11)
ax.legend(frameon=False)
savefig(fig, "fig5_getoff_mass.png")

put("prior_reported_rho_rushaccel_getoff", -0.31)  # cited from project notes (CLAUDE.md), n=69
# check widths
from PIL import Image  # noqa: E402

NUM["figure_widths_px"] = {p.name: Image.open(p).size[0] for p in sorted(FIG.glob("*.png"))}
NUM["_abs_forms"] = sorted(ABS)
(OUT / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
print(json.dumps({k: v for k, v in NUM.items() if not isinstance(v, list)}, indent=0, default=float))
