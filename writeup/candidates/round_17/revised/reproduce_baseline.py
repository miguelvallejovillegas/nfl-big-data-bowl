"""Re-derive the season-SOE baseline numbers in seconds, without the nine-minute tracking pipeline.

    uv run python writeup/candidates/round_17/revised/reproduce_baseline.py

Inputs: data/processed/player_season_stats.parquet (full-season NGS separation) and baseline_inputs.csv (written by
analysis.py: per receiver the draft class, the five no-tracking features and the Break G score; Break G is the only
column that needs the Combine tracking). Steps:
1. season SOE = route-weighted residual of NGS average separation on alignment (slot, inline), TE, cushion and season;
   written to soe_season.csv (one row per player-season);
2. career SOE = route-weighted mean over seasons;
3. no-tracking forecast (log pick, official 40, weight, NGS athleticism score, TE), weighted least squares fitted on two
   draft classes and scored on the third;
4. held-out R^2 (against the pooled route-weighted mean), RMSE and the Break G slope (residual on class intercepts and
   Break G, route-weighted), in inches; checked against numbers.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
P = OUT.parents[3] / "data" / "processed"
IN_YD = 36.0
BASE = ["logpick", "forty_f", "combine_weight_f", "ngs_athleticism_score_f", "te"]


def wls(A, y, w):
    s = np.sqrt(w)
    return np.linalg.lstsq(A * s[:, None], y * s, rcond=None)[0]


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
X = np.column_stack([np.ones(len(B)), B[BASE].to_numpy(float)])
y, w, cl, bg = B.ssoe.to_numpy(float), B.routes.to_numpy(float), B.draft_year.to_numpy(), B.BG.to_numpy(float)
f0 = np.empty(len(B))
for c in np.unique(cl):
    t = cl == c
    f0[t] = X[t] @ wls(X[~t], y[~t], w[~t])
r2 = 1 - np.sum(w * (y - f0) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)
rmse = IN_YD * np.sqrt(np.average((y - f0) ** 2, weights=w))
D = np.column_stack([(cl == c).astype(float) for c in np.unique(cl)] + [bg])
slope = IN_YD * wls(D, y - f0, w)[-1]
out = {"receivers": len(B), "heldout_r2": f"{r2:.2f}", "rmse_in": f"{rmse:.1f}", "break_g_slope_in": f"{slope:.1f}"}
print(json.dumps(out, indent=1))
nums = json.loads((OUT / "numbers.json").read_text())
for k, v in (("bl_r2", out["heldout_r2"]), ("bl_rmse", out["rmse_in"]), ("prim_b", out["break_g_slope_in"])):
    if k in nums and nums[k].replace("−", "-") != v:
        raise SystemExit(f"mismatch {k}: numbers.json {nums[k]} vs re-derived {v}")
print("matches numbers.json")
