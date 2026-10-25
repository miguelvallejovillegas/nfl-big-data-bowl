"""Recompute the headline results from inputs.csv in a few seconds and diff them against numbers.json.

    uv run python writeup/candidates/round_18/cand_1/verify.py

inputs.csv has one row per defensive lineman with 30+ NFL pass rushes (columns in inputs_schema.json).
Checks: held-out baseline R2 and RMSE, out-of-class slopes of Hoop G and burst, per-class slopes, the
baseline forecast `fc`, and the day-3 tie-break counts, win rates and gains. Bootstrap intervals and
permutation p-values need the frame-level pipeline (analysis.py) and are not recomputed here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "inputs.csv")
meta = json.loads((HERE / "inputs_schema.json").read_text())
NUM = json.loads((HERE / "numbers.json").read_text())
PS = meta["season_rushes"] / 100
TIMES = ["ten_yd_split_z", "three_cone_z"]
FILLX = ["combine_weight_z", "edge_share", "ngs_athleticism_score"]
BASE = ["log_pick", "combine_weight_z", "edge_share", "ngs_athleticism_score"] + TIMES
CLASSES = [2023, 2024, 2025]


def loco(df, cols, score=None):
    """fit OLS on two classes (times filled from FILLX inside the fold), predict the third."""
    score = df if score is None else score
    pred = pd.Series(np.nan, index=score.index)
    for c in CLASSES:
        tr, te = df[df.draft_year != c].copy(), score[score.draft_year == c].copy()
        for t in TIMES:
            m = tr[t].notna()
            b = np.linalg.lstsq(np.c_[np.ones(m.sum()), tr.loc[m, FILLX]], tr.loc[m, t], rcond=None)[0]
            for z in (tr, te):
                mz = z[t].isna()
                z.loc[mz, t] = np.c_[np.ones(mz.sum()), z.loc[mz, FILLX]] @ b
        b = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr[cols]], tr.press_eb, rcond=None)[0]
        pred[te.index] = np.c_[np.ones(len(te)), te[cols]] @ b
    return pred


got = {}
an = d[d.in_an].copy()
for key, cols in [("base", BASE), ("pick", ["log_pick", "edge_share"])]:
    p = loco(an, cols)
    sse = ((an.press_eb - p) ** 2).sum()
    got[f"r2_{key}"] = (1 - sse / ((an.press_eb - an.press_eb.mean()) ** 2).sum(), 2)
    got[f"rmse_{key}"] = (np.sqrt(sse / len(an)), 2)
res = an.press_eb - loco(an, BASE)
for k, tag in [("hoop_z", "hoop"), ("burst_z", "burst")]:
    sl = np.polyfit(an[k], res, 1)[0]
    got[f"ho_{tag}"] = (sl, 2)
    got[f"ho_{tag}_season"] = (sl * PS, 1)
    for c in CLASSES:
        m = an.draft_year == c
        got[f"ho_{tag}_{c}"] = (np.polyfit(an.loc[m, k], res[m], 1)[0], 2)
got["n_an"] = (len(an), 0)
# forecast for the tie pool, fit on all 30+ rush linemen of the other classes
fc = loco(d, BASE)
has = d.fc.notna()
print(f"fc max |diff| vs inputs.csv: {np.abs(fc[has] - d.fc[has]).max():.2e}")
# day-3 ties
t = d[has & d.day3].reset_index(drop=True)
I, J = np.triu_indices(len(t), 1)
ok = (t.draft_year.values[I] == t.draft_year.values[J]) & (np.abs(t.fc.values[I] - t.fc.values[J]) <= meta["tie_width_pts"])
I, J = I[ok], J[ok]
got["tie_pairs"] = (len(I), 0)
dy = t.press_eb.values[I] - t.press_eb.values[J]
for col, tag, sgn in [("hoop_z", "hoop", 1), ("pick", "pick", -1), ("ngs_athleticism_score", "ngs", 1),
                      ("burst_z", "burst", 1)]:
    s = np.sign(sgn * (t[col].values[I] - t[col].values[J]))
    v = ~np.isnan(s)
    win = np.where(s[v] == 0, 0.5, (s[v] * dy[v] > 0))
    got[f"tie_{tag}_win"] = (100 * win.mean(), 0)
    got[f"tie_{tag}_gain"] = (np.mean(s[v] * dy[v]) * PS, 1)
bad = 0
for k, (v, nd) in got.items():
    mine = f"{v:.{nd}f}"
    ref = str(NUM.get(k))
    flag = "ok" if mine.lstrip("-") == ref.lstrip("-") and (mine.startswith("-") == ref.startswith("-") or float(v) == 0) else "DIFF"
    bad += flag == "DIFF"
    print(f"{k:22s} recomputed {mine:>8s}   numbers.json {ref:>8s}   {flag}")
print("all match" if not bad else f"{bad} differences")
raise SystemExit(bool(bad))
