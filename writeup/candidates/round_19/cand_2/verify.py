"""Recompute the headline numbers in seconds and diff them against numbers written by analysis.py.

    uv run python writeup/candidates/round_19/cand_2/verify.py

No tracking pipeline is needed: the inputs are data/processed/player_season_stats.parquet,
data/processed/game_frames_sample.parquet (one column) and three CSVs written by analysis.py, documented column by
column in columns.json. Steps:
1. season SOE = route-weighted residual of NGS average separation on alignment, TE, cushion and season (soe_season.csv);
   career SOE = route-weighted mean; checked against baseline_inputs.csv;
2. no-tracking forecast (log pick, official 40, weight, NGS athleticism, TE), weighted least squares fitted on two
   Combine classes and scored on the third: held-out R^2, RMSE, the same with Break G added (delta R^2), and the Break G
   slope on the held-out residual (inches of SOE per sd);
3. speed benchmark: NFL top speed (90th percentile of each receiver's per-snap maximum speed, 8+ sampled snaps) rebuilt
   from game_frames_sample.parquet and compared with speed_inputs.csv; three Spearman correlations with the official 40;
4. day-3 tiebreaker: gain in rank correlation of the held-out forecasts (pairs_inputs.csv) and the season-fold models
   refitted from the pairs.
Exit code 1 on any mismatch with all_numbers.json (every number analysis.py computed, quoted or not).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

T0 = time.time()
OUT = Path(__file__).resolve().parent
P = OUT.parents[3] / "data" / "processed"
IN_YD = 36.0
BASE = ["logpick", "forty_f", "combine_weight_f", "ngs_athleticism_score_f", "te"]


def wls(A, y, w):
    s = np.sqrt(w)
    return np.linalg.lstsq(A * s[:, None], y * s, rcond=None)[0]


res = {}
# 1. season and career SOE
ss = pd.read_parquet(P / "player_season_stats.parquet")
ss = ss[ss.position_group.isin(["WR", "TE"]) & ss.routes.gt(0) & ss.avg_separation.notna()].copy()
cush = ss.avg_cushion.fillna(ss.avg_cushion.mean())
A = np.column_stack([np.ones(len(ss)), (ss.main_alignment == "SLOT_WR"), (ss.main_alignment == "TE"),
                     (ss.position_group == "TE"), cush, ss.season == 2024, ss.season == 2025]).astype(float)
ss["ssoe"] = ss.avg_separation - A @ wls(A, ss.avg_separation.to_numpy(float), ss.routes.to_numpy(float))
ss["soe_in"] = IN_YD * ss.ssoe
ss[["nfl_id", "season", "position_group", "main_alignment", "routes", "avg_separation", "soe_in"]].round(3).to_csv(
    OUT / "soe_season.csv", index=False)
car = ss.groupby("nfl_id").apply(lambda d: pd.Series(dict(ssoe=np.average(d.ssoe, weights=d.routes),
                                                           routes=d.routes.sum())), include_groups=False)
B0 = pd.read_csv(OUT / "baseline_inputs.csv").set_index("nfl_id")
B = B0.drop(columns=["routes", "career_soe_in"]).join(car, how="inner")
assert len(B) == len(B0) and np.allclose(IN_YD * B.ssoe, B0.career_soe_in, atol=1e-3), "career SOE differs"

# 2. baseline and Break G, leave one class out
X = np.column_stack([np.ones(len(B)), B[BASE].to_numpy(float)])
Xb = np.column_stack([X, B.BG.to_numpy(float)])
y, w, cl, bg = B.ssoe.to_numpy(float), B.routes.to_numpy(float), B.draft_year.to_numpy(), B.BG.to_numpy(float)
f0, f1 = np.empty(len(B)), np.empty(len(B))
for c in np.unique(cl):
    t = cl == c
    f0[t] = X[t] @ wls(X[~t], y[~t], w[~t])
    f1[t] = Xb[t] @ wls(Xb[~t], y[~t], w[~t])
sst = np.sum(w * (y - np.average(y, weights=w)) ** 2)
r2_0, r2_1 = 1 - np.sum(w * (y - f0) ** 2) / sst, 1 - np.sum(w * (y - f1) ** 2) / sst
D = np.column_stack([(cl == c).astype(float) for c in np.unique(cl)] + [bg])
res["bl_r2"] = f"{r2_0:.2f}"
res["dr2"] = f"{r2_1 - r2_0:.3f}"
res["bl_rmse"] = f"{IN_YD * np.sqrt(np.average((y - f0) ** 2, weights=w)):.1f}"
res["prim_b"] = f"{IN_YD * wls(D, y - f0, w)[-1]:.1f}"
res["n_main"] = str(len(B))

# 3. speed benchmark
S = pd.read_csv(OUT / "speed_inputs.csv").set_index("nfl_id")
gf = pd.read_parquet(P / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "s"])
gp = pd.read_parquet(P / "game_plays_sample.parquet", columns=["game_id", "play_id", "nfl_id"]).drop_duplicates()
gf = gf.merge(gp, on=["game_id", "play_id", "nfl_id"])
pm = gf.groupby(["nfl_id", "game_id", "play_id"]).s.max().groupby("nfl_id")
vfull = pm.quantile(0.9)[pm.size() >= 8]
chk = S.vtop_full.dropna()
assert np.allclose(vfull.reindex(chk.index), chk, atol=1e-3), "full-play top speed differs"


def rho(a, b):
    m = a.notna() & b.notna()
    return spearmanr(-a[m], b[m])[0]


res["spdf_rho"] = f"{rho(S.forty_std, S.vtop_full):.2f}"
res["spdr_rho"] = f"{rho(S.forty_raw, S.vtop_full):.2f}"
res["spd_rho"] = f"{rho(S.forty_std, S.vtop_break):.2f}"

# 4. day-3 tiebreaker
PI = pd.read_csv(OUT / "pairs_inputs.csv")
F0, F1 = ["logpick", "te", "lr1"], ["logpick", "te", "lr1", "y1s"]


def fit(df, cols):
    A_ = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    return np.linalg.lstsq(A_, df.y2.to_numpy(float), rcond=None)[0]


def pred(df, b, cols):
    return np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)]) @ b


g0, g1 = np.empty(len(PI)), np.empty(len(PI))
for s_ in (2023, 2024):
    t = (PI.s1 == s_).to_numpy()
    g0[t], g1[t] = pred(PI[t], fit(PI[~t], F0), F0), pred(PI[t], fit(PI[~t], F1), F1)
assert np.allclose(g0, PI.f_m0, atol=1e-4) and np.allclose(g1, PI.f_m1, atol=1e-4), "season-fold forecasts differ"
d3 = PI.day3.to_numpy()
for k, m in (("ev_d3_drho", d3), ("ev_all_drho", np.ones(len(PI), bool))):
    res[k] = f"{spearmanr(g1[m], PI.y2[m])[0] - spearmanr(g0[m], PI.y2[m])[0]:.2f}"
res["ev_d3_n"], res["ev_d3_np"] = str(int(d3.sum())), str(PI.pid[d3].nunique())

nums = json.loads((OUT / "all_numbers.json").read_text())
bad = []
for k, v in res.items():
    ok = nums.get(k, "").replace("−", "-") == v
    print(f"{k:12s} recomputed {v:>7s}   analysis.py {nums.get(k, 'missing'):>7s}   {'ok' if ok else 'MISMATCH'}")
    bad += [] if ok else [k]
print(f"{len(res) - len(bad)} of {len(res)} match; {time.time() - T0:.1f} s")
sys.exit(1 if bad else 0)
