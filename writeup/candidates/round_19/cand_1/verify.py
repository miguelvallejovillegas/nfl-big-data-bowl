"""Recompute the headline results from inputs.csv in a few seconds and diff them against numbers.json.

    uv run python writeup/candidates/round_19/cand_1/verify.py

inputs.csv has one row per defensive lineman with 30+ NFL pass rushes (columns in inputs_schema.json).
Checks, each recomputed from scratch:
  * the shrunk pressure rate press_eb from raw career pressures and rushes (prior in inputs_schema.json);
  * held-out baseline R2 and RMSE (Table 2), the leave-one-class-out forecast `fc`;
  * out-of-class slopes of Hoop G and burst, overall and per class;
  * the day-3 ties: pair count, win rates and gains for each tiebreaker, per-class records, the pick-gap split,
    the break-even hit rate;
  * the nested back-test (tie width re-chosen on the other two classes) point estimates.
Bootstrap intervals and permutation p-values need analysis.py and are not recomputed here.
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
TIE = meta["tie_width_pts"]
TIMES = ["ten_yd_split_z", "three_cone_z"]
FILLX = ["combine_weight_z", "edge_share", "ngs_athleticism_score"]
BASE = ["log_pick", "combine_weight_z", "edge_share", "ngs_athleticism_score"] + TIMES
CLASSES = [2023, 2024, 2025]

# 1. shrunk pressure rate from raw counts
mu, M = meta["prior_mean"], meta["prior_rushes"]
eb = 100 * (d.pressures + M * mu) / (d.rush_snaps + M)
print(f"press_eb max |diff| from raw counts: {np.abs(eb - d.press_eb).max():.2e}")
assert np.abs(eb - d.press_eb).max() < 1e-6


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
t = d[has & d.day3].reset_index(drop=True)


def pairs(tt, w):
    I, J = np.triu_indices(len(tt), 1)
    ok = (tt.draft_year.values[I] == tt.draft_year.values[J]) & (np.abs(tt.fc.values[I] - tt.fc.values[J]) <= w)
    return I[ok], J[ok]


def score(tt, I, J, col, sgn=1):
    s = np.sign(sgn * (tt[col].values[I] - tt[col].values[J]))
    v = ~np.isnan(s)
    dy = tt.press_eb.values[I] - tt.press_eb.values[J]
    win = np.where(s[v] == 0, 0.5, (s[v] * dy[v] > 0))
    return win, s[v] * dy[v]


I, J = pairs(t, TIE)
got["tie_pairs"] = (len(I), 0)
for col, tag, sgn in [("hoop_z", "hoop", 1), ("pick", "pick", -1), ("ngs_athleticism_score", "ngs", 1),
                      ("burst_z", "burst", 1)]:
    win, g = score(t, I, J, col, sgn)
    got[f"tie_{tag}_win"] = (100 * win.mean(), 0)
    got[f"tie_{tag}_gain"] = (np.mean(g) * PS, 1)
_, gh = score(t, I, J, "hoop_z")
G, L = gh[gh > 0].mean() * PS, -gh[gh < 0].mean() * PS
got["tie_gain_right"] = (G, 1); got["tie_cost_wrong"] = (L, 1)
got["tie_breakeven"] = (100 * float(f"{L:.1f}") / (float(f"{G:.1f}") + float(f"{L:.1f}")), 0)
for c in CLASSES:
    m = t.draft_year.values[I] == c
    got[f"tie_hoop_{c}"] = (100 * score(t, I[m], J[m], "hoop_z")[0].mean(), 0)
    got[f"tie_pick_{c}"] = (100 * score(t, I[m], J[m], "pick", -1)[0].mean(), 0)
gap = np.abs(t.pick.values[I] - t.pick.values[J])
for lo, hi, tg in [(0, 30, "a"), (30, 80, "b"), (80, 999, "c")]:
    m = (gap >= lo) & (gap < hi)
    got[f"gap_{tg}_pick"] = (100 * score(t, I[m], J[m], "pick", -1)[0].mean(), 0)
    got[f"gap_{tg}_hoop"] = (100 * score(t, I[m], J[m], "hoop_z")[0].mean(), 0)
# nested back-test: width re-chosen on the other two classes by Hoop G gain
hw, hg, pw, pg = [], [], [], []
for c in CLASSES:
    tr, te = t[t.draft_year != c].reset_index(drop=True), t[t.draft_year == c].reset_index(drop=True)
    w = max(meta["backtest_grid"], key=lambda w_: np.nan_to_num(np.mean(score(tr, *pairs(tr, w_), "hoop_z")[1]), nan=-9))
    a, b = pairs(te, w)
    x = score(te, a, b, "hoop_z"); y = score(te, a, b, "pick", -1)
    hw += list(x[0]); hg += list(x[1]); pw += list(y[0]); pg += list(y[1])
got["bt_pairs"] = (len(hw), 0)
got["bt_hoop_win"] = (100 * np.mean(hw), 0); got["bt_pick_win"] = (100 * np.mean(pw), 0)
got["bt_hoop_gain"] = (np.mean(hg) * PS, 1); got["bt_pick_gain"] = (np.mean(pg) * PS, 1)
bad = 0
for k, (v, nd) in got.items():
    mine = f"{v:.{nd}f}"
    ref = str(NUM.get(k))
    flag = "ok" if mine.lstrip("-") == ref.lstrip("-") and (mine.startswith("-") == ref.startswith("-") or float(v) == 0) else "DIFF"
    bad += flag == "DIFF"
    print(f"{k:22s} recomputed {mine:>8s}   numbers.json {ref:>8s}   {flag}")
print("all match" if not bad else f"{bad} differences")
raise SystemExit(bool(bad))
