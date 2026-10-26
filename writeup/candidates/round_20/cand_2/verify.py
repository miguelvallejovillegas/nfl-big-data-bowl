"""Recompute the headline numbers from the raw tables and diff them against all_numbers.json (written by analysis.py).

    uv run python writeup/candidates/round_20/cand_2/verify.py

Independent of core.py and analysis.py (no shared code). Inputs: data/processed/player_season_stats.parquet,
players.parquet, combine_results.parquet, game_plays_sample.parquet, game_frames_sample.parquet (one speed column)
and two small CSVs written by analysis.py (columns.json documents them):
- baseline_inputs.csv: the no-tracking covariates and the Break G score of the 116 receivers (Break G needs the full
  Combine tracking pipeline; everything else is rebuilt here);
- speed_inputs.csv: the receivers of Fig 1 (8+ graded NFL breaks) in the order analysis.py used.
Resamples use the same per-statistic seeds (sha256 of the statistic name), so intervals must match to the last digit.

Checks:
1. season SOE from NGS separation, career SOE (against baseline_inputs.csv);
2. leave-one-class-out baseline R2, RMSE, delta R2 with interval; held-out Break G slope with interval; TOST arms;
3. official 40 vs NFL top speed rank correlation and interval (top speed rebuilt from the frames);
4. day-3 pairs from NGS, leave-one-class-out forecasts, the half-list back-test and interval, the gap-32 override
   error rate, interval and break-even;
5. sensitivity ranges of the held-out slope (outcome builds) and of the override error rate.
Exit code 1 on any mismatch.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

OUT = Path(__file__).resolve().parent
P = OUT.parents[3] / "data" / "processed"
IN_YD, SESOI, NBOOT, GAP = 36.0, 3.0, 2000, 32
CLASSES = (2023, 2024, 2025)
BASE = ["logpick", "forty_f", "combine_weight_f", "ngs_athleticism_score_f", "te"]
NUM = json.loads((OUT / "all_numbers.json").read_text())
res: dict[str, float] = {}


def seed(name):
    return np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))


def wls(A, y, w):
    s = np.sqrt(np.asarray(w, float))
    return np.linalg.lstsq(A * s[:, None], np.asarray(y, float) * s, rcond=None)[0]


def q(a, p):
    return float(np.nanpercentile(np.asarray(a, float), p))


def z(s):
    return (s - s.mean()) / s.std(ddof=1)


# ---------------------------------------------------------------------------------------------- 1. SOE
raw = pd.read_parquet(P / "player_season_stats.parquet")


def soe(cushion=True):
    s = raw[raw.position_group.isin(["WR", "TE"]) & raw.routes.gt(0) & raw.avg_separation.notna()].copy()
    cols = [np.ones(len(s)), s.main_alignment == "SLOT_WR", s.main_alignment == "TE", s.position_group == "TE"]
    cols += [s.avg_cushion.fillna(s.avg_cushion.mean())] if cushion else []
    cols += [s.season == 2024, s.season == 2025]
    A = np.column_stack(cols).astype(float)
    s["ssoe"] = s.avg_separation - A @ wls(A, s.avg_separation, s.routes)
    return s


def car(s, mr=0):
    s = s[s.routes >= mr]
    return s.groupby("nfl_id").apply(lambda d: pd.Series(dict(ssoe=np.average(d.ssoe, weights=d.routes),
                                                              routes=d.routes.sum())), include_groups=False)


ss = soe()
B0 = pd.read_csv(OUT / "baseline_inputs.csv").set_index("nfl_id")
c0 = car(ss).reindex(B0.index)
assert np.allclose(IN_YD * c0.ssoe, B0.career_soe_in, atol=1e-4), "career SOE differs"
res["n_main"] = len(B0)


# ---------------------------------------------------------------------------------------------- 2. held-out tests
def parts(B, c):
    return (np.column_stack([np.ones(len(B)), B[BASE].to_numpy(float)]), c.ssoe.to_numpy(float),
            c.routes.to_numpy(float), B.draft_year.to_numpy(), B.BG.to_numpy(float))


def slope(XB, y, w, cl, bg):
    r = np.empty(len(y))
    for c in CLASSES:
        t = cl == c
        if t.any():
            r[t] = y[t] - XB[t] @ wls(XB[~t], y[~t], w[~t])
    pres = [c for c in CLASSES if (cl == c).any()]
    return IN_YD * wls(np.column_stack([(cl == c).astype(float) for c in pres] + [bg]), r, w)[-1]


def r2(XB, y, w, cl, bg):
    Xb = np.column_stack([XB, bg])
    f0, f1 = np.empty(len(y)), np.empty(len(y))
    for c in CLASSES:
        t = cl == c
        f0[t] = XB[t] @ wls(XB[~t], y[~t], w[~t])
        f1[t] = Xb[t] @ wls(Xb[~t], y[~t], w[~t])
    sst = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    a, b = 1 - np.sum(w * (y - f0) ** 2) / sst, 1 - np.sum(w * (y - f1) ** 2) / sst
    return np.array([a, b, b - a, IN_YD * np.sqrt(np.average((y - f0) ** 2, weights=w))])


def strat(cl, rng):
    return np.concatenate([rng.choice(np.where(cl == c)[0], int((cl == c).sum())) for c in CLASSES if (cl == c).any()])


V = parts(B0, c0)
h = r2(*V)
res.update(bl_r2=h[0], dr2=h[2], bl_rmse=h[3])
rng = seed("ho_r2")
hb = np.array([r2(*(a[ix] for a in V)) for ix in (strat(V[3], rng) for _ in range(NBOOT // 2))])
res.update(dr2_lo=q(hb[:, 2], 5), dr2_hi=q(hb[:, 2], 95))
res["ho"] = slope(*V)
rng = seed("ho_slope")
sb = np.array([slope(*(a[ix] for a in V)) for ix in (strat(V[3], rng) for _ in range(NBOOT))])
res.update(ho_lo=q(sb, 5), ho_hi=q(sb, 95), tost_up=float(np.mean(sb >= SESOI)), tost_dn=float(np.mean(sb <= -SESOI)))
res["tost"] = max(res["tost_up"], res["tost_dn"])

# ---------------------------------------------------------------------------------------------- 3. speed
pl = pd.read_parquet(P / "players.parquet")
cr = pd.read_parquet(P / "combine_results.parquet").set_index("nfl_id")
pp = pl.set_index("nfl_id")[["position_group", "draft_year", "draft_round", "draft_overall_pick", "display_name"]]
pp = pp[pp.position_group.isin(["WR", "TE"])].join(cr[["forty"]])
pp["f40z"] = -pp.groupby(["draft_year", "position_group"]).forty.transform(z)
gp = pd.read_parquet(P / "game_plays_sample.parquet")
gp = gp[gp.week.between(1, 18) & gp.nfl_id.isin(pp.index)][["game_id", "play_id", "nfl_id"]]
fr = pd.read_parquet(P / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "s"]).merge(
    gp, on=["game_id", "play_id", "nfl_id"])
mx = fr.groupby(["nfl_id", "game_id", "play_id"]).s.max().groupby("nfl_id")
vtop = mx.quantile(0.9)[mx.size() >= 8]
SI = pd.read_csv(OUT / "speed_inputs.csv").set_index("nfl_id")
sp = pd.DataFrame({"f40z": pp.f40z.reindex(SI.index), "vtop": vtop.reindex(SI.index), "pos": SI.pos})
assert sp.notna().all().all(), "speed inputs missing"
sp["vz"] = sp.groupby("pos").vtop.transform(z)
a_, b_ = sp.f40z.to_numpy(), sp.vz.to_numpy()
rng = seed("spc")
bs = []
for _ in range(NBOOT):
    ix = rng.integers(0, len(a_), len(a_))
    bs.append(spearmanr(a_[ix], b_[ix])[0])
res.update(spc=spearmanr(a_, b_)[0], spc_lo=q(bs, 5), spc_hi=q(bs, 95), f1_n=len(sp))

# ---------------------------------------------------------------------------------------------- 4. day 3
pp["logpick"] = np.log(pp.draft_overall_pick.fillna(262))
pp["te"] = (pp.position_group == "TE").astype(float)


def pairs(s, mr=100):
    rows = []
    for a, b in ((2023, 2024), (2024, 2025)):
        j = s[s.season == a].set_index("nfl_id")[["ssoe", "routes", "games", "team"]].join(
            s[s.season == b].set_index("nfl_id")[["ssoe", "routes"]], rsuffix="_2", how="inner")
        j = j.rename(columns={"ssoe": "y1", "routes": "r1", "games": "g1", "ssoe_2": "y2", "routes_2": "r2"})
        j["s1"] = a
        rows.append(j)
    pr = pd.concat(rows).join(pp[["draft_round", "draft_overall_pick", "display_name", "draft_year", "logpick", "te"]])
    pr = pr[(pr.r1 >= mr) & (pr.r2 >= mr)].copy()
    pr["pid"] = pr.index
    pr["day3"] = pr.draft_round.isna() | (pr.draft_round >= 4)
    pr["lr1"], pr["hit"], pr["pick"] = np.log(pr.r1), pr.y2 > 0, pr.draft_overall_pick.fillna(262)
    return pr.reset_index(drop=True)


def shrink_par(s):
    s = s[s.routes >= 50]
    mu = float(np.average(s.ssoe, weights=s.routes))
    pv = s.pivot_table(index="nfl_id", columns="season", values="ssoe")
    se = sorted(s.season.unique())
    cp = pd.concat([pv[[a, a + 1]].dropna().set_axis(["a", "b"], axis=1) for a in se if a + 1 in se])
    t2 = float(np.mean((cp.a - mu) * (cp.b - mu)))
    iv = 1.0 / s.routes.to_numpy(float)
    return mu, t2, float(np.sum(((s.ssoe.to_numpy() - mu) ** 2 - t2) * iv) / np.sum(iv ** 2))


def ols(df, cols):
    A = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    return np.linalg.lstsq(A, df.y2.to_numpy(float), rcond=None)[0]


F0, F1 = ["logpick", "te", "lr1"], ["logpick", "te", "lr1", "y1s"]
cls_of = pl.set_index("nfl_id").draft_year


def loco(pr, s):
    out = pr.copy()
    out[["f0", "f1", "y1s"]] = np.nan
    for c in sorted(out.draft_year.unique()):
        t = (out.draft_year == c).to_numpy()
        mu, t2, s2 = shrink_par(s[cls_of.reindex(s.nfl_id).to_numpy() != c])
        p2 = out.assign(y1s=mu + t2 / (t2 + s2 / out.r1) * (out.y1 - mu))
        for col, cols in (("f0", F0), ("f1", F1)):
            b = ols(p2[~t], cols)
            out.loc[t, col] = np.column_stack([np.ones(t.sum()), p2[t][cols].to_numpy(float)]) @ b
        out.loc[t, "y1s"] = p2.y1s.to_numpy()[t]
    return out


def contrasts(d3w):
    d = d3w.reset_index(drop=True)
    rows = []
    for s1 in sorted(d.s1.unique()):
        ii = np.where(d.s1.to_numpy() == s1)[0]
        for x in range(len(ii)):
            for k in range(x + 1, len(ii)):
                a, b = ii[x], ii[k]
                pk = a if (d.pick.iloc[a], -d.r1.iloc[a]) < (d.pick.iloc[b], -d.r1.iloc[b]) else b
                so = a if d.y1s.iloc[a] > d.y1s.iloc[b] else b
                rows.append(dict(a=a, b=b, gap=abs(d.pick.iloc[a] - d.pick.iloc[b]), pick=pk, soe=so))
    return d, pd.DataFrame(rows)


def ov(ct, y2, w=None):
    w = np.ones(len(ct)) if w is None else w
    m = ((ct.gap < GAP) & (ct.soe != ct.pick)).to_numpy() & (w > 0)
    if m.sum() == 0:
        return np.nan, np.nan, 0
    g = y2[ct.soe.to_numpy()[m]] - y2[ct.pick.to_numpy()[m]]
    ww = w[m]
    gain = np.average(g[g > 0], weights=ww[g > 0]) if ww[g > 0].sum() else np.nan
    loss = -np.average(g[g < 0], weights=ww[g < 0]) if ww[g < 0].sum() else np.nan
    return 100 * np.average(g < 0, weights=ww), 100 * gain / (gain + loss), int(m.sum())


FC = loco(pairs(ss), ss)
d3 = FC[FC.day3].reset_index(drop=True)
half = d3.groupby("s1").s1.transform("size") // 2
d3["top_soe"] = d3.groupby("s1").f1.rank(ascending=False, method="first") <= half
d3["k"] = d3.pick * 1e4 - d3.r1
d3["top_pick"] = d3.groupby("s1").k.rank(method="first") <= half
res.update(d3_n=len(d3), hl_k1=int(d3[d3.top_soe].hit.sum()), hl_k0=int(d3[d3.top_pick].hit.sum()))
grp = [np.where(d3.pid.to_numpy() == p)[0] for p in pd.unique(d3.pid)]
rng = seed("d3_half")
hd = []
for _ in range(NBOOT):
    bb = d3.iloc[np.concatenate([grp[i] for i in rng.integers(0, len(grp), len(grp))])]
    hd.append(100 * bb[bb.top_soe].hit.mean() - 100 * bb[bb.top_pick].hit.mean())
res.update(hl_d_lo=q(hd, 5), hl_d_hi=q(hd, 95))
dw, ct = contrasts(d3[d3.te == 0])
y2 = IN_YD * dw.y2.to_numpy()
res["ov_wr"], res["ov_be"], res["ov_n"] = ov(ct, y2)
pid = dw.pid.to_numpy()
rng = seed("override")
wr_b = []
GAPS_ANALYSIS = [16, 24, 32, 48, 64, 96, 999]  # analysis.py evaluates these per resample; only 32 is checked here
for _ in range(NBOOT):
    u = np.unique(pid)
    m = pd.Series(rng.choice(u, len(u))).value_counts().reindex(pid).fillna(0).to_numpy()
    wr_b.append(ov(ct, y2, m[ct.a.to_numpy()] * m[ct.b.to_numpy()])[0])
res.update(ov_wr_lo=q(wr_b, 5), ov_wr_hi=q(wr_b, 95))

# ---------------------------------------------------------------------------------------------- 5. sensitivity
vals = [res["ho"]]
for s_, mr in ((ss, 50), (ss, 100), (soe(False), 0)):
    c = car(s_, mr)
    B = B0.loc[B0.index.intersection(c.index)]
    vals.append(slope(*parts(B, c.reindex(B.index))))
XB, y, w, cl, bg = V
vals.append(slope(XB, y, np.ones(len(y)), cl, bg))
wr = (B0.te == 0).to_numpy()
vals.append(slope(XB[wr][:, :-1], y[wr], w[wr], cl[wr], bg[wr]))
res.update(sens_lo=min(vals), sens_hi=max(vals))
wrs = []
for mr, rawsoe in ((50, False), (150, False), (100, True)):
    fc = loco(pairs(ss, mr), ss)
    dd = fc[fc.day3 & (fc.te == 0)].reset_index(drop=True)
    if rawsoe:
        dd = dd.assign(y1s=dd.y1)
    d_, c_ = contrasts(dd)
    wrs.append(ov(c_, IN_YD * d_.y2.to_numpy())[0])
res.update(ovs_lo=min(wrs), ovs_hi=max(wrs))

# ---------------------------------------------------------------------------------------------- compare
bad = []
for k, v in res.items():
    ref = NUM[k]
    nd = len(ref.split(".")[1]) if "." in ref else 0
    got = f"{v:.{nd}f}" if nd else str(int(round(v)))
    if got.startswith("-") and float(got) == 0:
        got = got[1:]
    ok = got == ref
    print(f"{'ok ' if ok else 'BAD'} {k:10s} analysis {ref:>8s}  verify {got:>8s}")
    bad += [] if ok else [k]
print(f"{len(res) - len(bad)} of {len(res)} numbers reproduced")
sys.exit(1 if bad else 0)
