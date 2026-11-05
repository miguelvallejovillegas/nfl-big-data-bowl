"""Independent re-implementation of the headline held-out Hoop G slope.

    uv run python writeup/candidates/round_23/cand_1/verify.py

Shares no code with analysis.py and deliberately differs from it in every implementation choice that is not part
of the estimand's definition:
  * Hoop G kinematics: centred 3-frame moving average + np.gradient (analysis.py: Savitzky-Golay 7/2 derivatives).
  * Pressure shrinkage: beta-binomial prior fitted by maximum likelihood (analysis.py: method of moments).
  * Leave-one-class-out baseline: scikit-learn pipeline (analysis.py: numpy least squares). Same fill rule in both:
    missing 10-yd split / 3-cone = training-fold median.
  * Bootstrap: class-stratified, 1,500 draws, seed 99 (analysis.py: 1,000 draws, seed 2027).
  * Same analysis set as analysis.py, rebuilt here from raw frames: 30+ NFL rushes, a usable hoop lap (15+ frames)
    and a usable pass-rush rep (15+ frames, starts below 1 yd/s, reaches 1 yd/s with 1.0 s of clip left).
Writes verify.json; analysis.py quotes it and the writeup reports the agreement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

OUT = Path(__file__).resolve().parent
P = OUT.parents[3] / "data" / "processed"
G_YD = 9.80665 / 0.9144
PER_SEASON = 4.5  # rate points -> pressures per 450 rushes

pl = pd.read_parquet(P / "players.parquet")
pl = pl[pl.position_group == "DL"]
cr = pd.read_parquet(P / "combine_results.parquet")
ev = pd.read_parquet(P / "combine_events.parquet")
ev = ev[(ev.position_group == "DL") & (ev.drill_name.isin(["RUN_THE_HOOP_DRILL", "PASS_RUSH_DRILL"]))]
hoop_ids = ev[ev.drill_name == "RUN_THE_HOOP_DRILL"].event_id.tolist()
fr = pd.read_parquet(P / "combine_frames.parquet", filters=[("event_id", "in", hoop_ids)],
                     columns=["event_id", "frame", "entity_type", "x", "y"])
fr = fr[fr.entity_type == "PLAYER"].sort_values(["event_id", "frame"])


def peak_lateral(x, y):
    k = np.ones(3) / 3
    xs = np.convolve(np.pad(x, 1, mode="edge"), k, "valid")
    ys = np.convolve(np.pad(y, 1, mode="edge"), k, "valid")
    vx, vy = np.gradient(xs, 0.1), np.gradient(ys, 0.1)
    ax, ay = np.gradient(vx, 0.1), np.gradient(vy, 0.1)
    sp = np.sqrt(vx ** 2 + vy ** 2)
    lat = np.abs(vx * ay - vy * ax) / np.clip(sp, 0.5, None) / G_YD
    return float(pd.Series(lat).rolling(5, center=True).mean().max())


hg = {e: peak_lateral(g.x.values, g.y.values) for e, g in fr.groupby("event_id") if len(g) >= 15}
h = ev[ev.event_id.isin(hg)].assign(hg=lambda d: d.event_id.map(hg))
h = h.groupby("nfl_id").agg(hg=("hg", "mean"), yr=("draft_year", "first")).reset_index()
h["hz"] = h.groupby("yr").hg.transform(lambda v: (v - v.mean()) / v.std(ddof=1))
rush_ids = ev[ev.drill_name == "PASS_RUSH_DRILL"].event_id.tolist()
rf = pd.read_parquet(P / "combine_frames.parquet", filters=[("event_id", "in", rush_ids)],
                     columns=["event_id", "frame", "entity_type", "s"])
rf = rf[rf.entity_type == "PLAYER"].sort_values(["event_id", "frame"])


def usable_rep(sp):
    if len(sp) < 15 or sp[0] >= 1.0:
        return False
    hit = np.flatnonzero(sp >= 1.0)
    return len(hit) > 0 and hit[0] + 10 < len(sp)


ok_rush = [e for e, g in rf.groupby("event_id") if usable_rep(g.s.values)]
has_rush = set(ev[ev.event_id.isin(ok_rush)].nfl_id)

st = pd.read_parquet(P / "player_season_stats.parquet")
st = st[st.position_group == "DL"].groupby("nfl_id")[["pass_rush_snaps", "pressures"]].sum()
fit = st[st.pass_rush_snaps >= 20]


def nll(par):
    a, b = np.exp(par)
    n, k = fit.pass_rush_snaps.values, fit.pressures.values
    return -np.sum(special.betaln(k + a, n - k + b) - special.betaln(a, b))


a_, b_ = np.exp(optimize.minimize(nll, [0.0, 2.0], method="Nelder-Mead").x)
st["rate"] = 100 * (st.pressures + a_) / (st.pass_rush_snaps + a_ + b_)

al = pd.read_parquet(P / "player_season_alignment.parquet")
al["w"] = al.lined_up_position.isin(["EDGE", "OLB"]) * al.plays
es = al.groupby("nfl_id").apply(lambda d: d.w.sum() / d.plays.sum()).rename("edge")

d = pl[["nfl_id", "draft_year", "draft_overall_pick", "nfl_position"]].merge(
    cr[["nfl_id", "combine_weight", "ten_yd_split", "three_cone", "ngs_athleticism_score"]], on="nfl_id", how="left")
d = d.merge(es, on="nfl_id", how="left").merge(st, on="nfl_id", how="left").merge(h[["nfl_id", "hz"]], on="nfl_id",
                                                                                    how="left")
d["edge"] = d.edge.fillna(d.nfl_position.isin(["DE", "OLB"]).astype(float))
d["lp"] = np.log(d.draft_overall_pick.fillna(262))
for c in ["combine_weight", "ten_yd_split", "three_cone"]:
    d[c] = d.groupby("draft_year")[c].transform(lambda v: (v - v.mean()) / v.std(ddof=1))
X = ["lp", "combine_weight", "edge", "ngs_athleticism_score", "ten_yd_split", "three_cone"]
A = d[(d.pass_rush_snaps >= 30) & d.hz.notna() & d.nfl_id.isin(has_rush)].reset_index(drop=True)
YRS = sorted(A.draft_year.unique())


def heldout_slope(D):
    res = np.empty(len(D))
    for y in YRS:
        tr, te = D.draft_year != y, D.draft_year == y
        m = make_pipeline(SimpleImputer(strategy="median"), LinearRegression()).fit(D.loc[tr, X], D.loc[tr, "rate"])
        res[te.values] = D.loc[te, "rate"].values - m.predict(D.loc[te, X])
    xv = D.hz.values
    return float(np.cov(xv, res)[0, 1] / np.var(xv, ddof=1)), res


est, res = heldout_slope(A)
cls = [float(np.polyfit(A.hz[A.draft_year == y], res[(A.draft_year == y).values], 1)[0]) for y in YRS]
rng = np.random.default_rng(99)
idx = [np.flatnonzero(A.draft_year.values == y) for y in YRS]
bs = [heldout_slope(A.iloc[np.concatenate([rng.choice(i, len(i)) for i in idx])].reset_index(drop=True))[0]
      for _ in range(1500)]
out = {"v_ids": sorted(int(i) for i in A.nfl_id), "v_n": int(len(A)), "v_slope": est * PER_SEASON, "v_lo": float(np.percentile(bs, 5)) * PER_SEASON,
       "v_hi": float(np.percentile(bs, 95)) * PER_SEASON, "v_cls": [c * PER_SEASON for c in cls],
       "v_prior_rushes": float(a_ + b_)}
(OUT / "verify.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'v_ids'}, indent=1))
