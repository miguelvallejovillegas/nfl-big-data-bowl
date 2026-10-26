"""Rebuild every verify input from the raw tables and recompute the quoted numbers; diff against all_numbers.json.

    uv run python writeup/candidates/round_21/cand_2/verify.py

Independent of core.py and analysis.py: no shared code and its OWN bootstrap seeds, so point estimates must match to
the last printed digit while every resampled quantity (intervals, TOST arms, power) is allowed to drift by Monte Carlo
error only (tolerance: 15% of the interval width, or 0.02 / 0.2 / 2 units at 2 / 1 / 0 decimals); the largest drift is
printed. The held-out fit is also re-implemented a second way (statsmodels WLS with class dummies, not numpy lstsq) and
must give the same slope. Inputs are only data/processed/*.parquet:
- combine_events + combine_frames  -> Break G per Combine rep -> receiver score (rebuilds baseline_inputs.csv BG);
- game_plays_sample + game_frames_sample -> Break G per NFL break, NFL top speed (rebuilds speed_inputs.csv);
- players + combine_results -> no-tracking baseline covariates; player_season_stats -> season and career SOE.
The two CSVs written by analysis.py are only compared against the rebuilt tables, never read as inputs.

Checks: (0) the rebuilt inputs equal the CSVs; (1) reliability: split-half and interval, same-drill ICC and pairs,
between-drill ICC and its interval, 1- and 2-rep reliability from the rep-level mixed model, cuts for 0.70 / 0.80;
(2) held-out baseline R2, delta R2 and interval, held-out slope and interval, raw-score slope, errors-in-variables
interval, minimum detectable effect and power, TOST arms; (3) hierarchical slope and interval; (4) pick price and
interval; (5) speed and Break G transfer with intervals, noise ceiling; (6) day-3 back-test, film-queue hit rate,
override record; (7) sensitivity ranges. Exit code 1 on any mismatch.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.signal import savgol_filter
from scipy.stats import beta, norm, spearmanr

OUT = Path(__file__).resolve().parent
P = OUT.parents[3] / "data" / "processed"
IN_YD, SESOI, NBOOT, GAP, G = 36.0, 3.0, 2000, 32, 10.73
CLASSES = (2023, 2024, 2025)
BASE = ["logpick", "forty_f", "combine_weight_f", "ngs_athleticism_score_f", "te"]
FAM = {"SLANT_ROUTE_LEFT": "SLANT", "SLOT_STRIKE_ROUTE_LEFT": "SLANT", "SPEED_OUT_ROUTE_LEFT": "OUT",
       "CURL_ROUTE_RIGHT": "HITCH", "COMEBACK_ROUTE_RIGHT": "HITCH", "HOOK_ROUTE_RIGHT": "HITCH",
       "DAGGER_ROUTE_LEFT": "IN", "IN_ROUTE_LEFT": "IN", "SLOT_SAIL_ROUTE_RIGHT": "CORNER", "CORNER_ROUTE": "CORNER"}
HALF = {"SLANT_ROUTE_LEFT", "CURL_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT", "SPEED_OUT_ROUTE_LEFT", "HOOK_ROUTE_RIGHT"}
ROUTES = ["SLANT", "OUT", "HITCH", "IN", "CORNER"]
NUM = json.loads((OUT / "all_numbers.json").read_text())
res: dict[str, float] = {}


def seed(name):
    """Own seeds: a different hash salt from analysis.py, so every bootstrap draw differs."""
    return np.random.default_rng(int(hashlib.sha256((name + "|verify-r21").encode()).hexdigest()[8:16], 16))


def wls(A, y, w):
    s = np.sqrt(np.asarray(w, float))
    return np.linalg.lstsq(A * s[:, None], np.asarray(y, float) * s, rcond=None)[0]


def q(a, p):
    return float(np.nanpercentile(np.asarray(a, float), p))


def z(s):
    return (s - s.mean()) / s.std(ddof=1)


def oneway_icc(vals, ids):
    """One-way ICC(1) with unequal group sizes."""
    d = pd.DataFrame({"v": np.asarray(vals, float), "g": np.asarray(ids)})
    k = d.groupby("g").size()
    n, m = len(d), len(k)
    k0 = (n - (k ** 2).sum() / n) / (m - 1)
    mu = d.groupby("g").v.mean()
    msb = (k * (mu - d.v.mean()) ** 2).sum() / (m - 1)
    msw = ((d.v - d.g.map(mu)) ** 2).sum() / (n - m)
    return (msb - msw) / (msb + (k0 - 1) * msw)


def boot_spear(a, b, name, groups=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng, out = seed(name), []
    for _ in range(NBOOT):
        ix = rng.integers(0, len(a), len(a)) if groups is None else np.concatenate(
            [groups[k] for k in rng.integers(0, len(groups), len(groups))])
        out.append(spearmanr(a[ix], b[ix])[0])
    return spearmanr(a, b)[0], q(out, 5), q(out, 95)


# ---------------------------------------------------------------------------------------------- Break G, from frames
def cut(x, y, start, stop):
    """First cut of 30+ degrees after 1.5 yd, entered above 2.5 yd/s: mean |acceleration| (g) over 1.0 s around it."""
    n = len(x)
    if n < 15:
        return None
    w = 7 if n >= 7 else max(3, (n // 2) * 2 - 1)
    vx, vy = savgol_filter(x, w, 2, deriv=1, delta=0.1), savgol_filter(y, w, 2, deriv=1, delta=0.1)
    acc = np.hypot(savgol_filter(vx, w, 2, deriv=1, delta=0.1), savgol_filter(vy, w, 2, deriv=1, delta=0.1))
    sp = np.hypot(vx, vy)
    hd = np.degrees(np.arctan2(vx, vy)) % 360
    path = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    path -= path[start]
    tn = np.zeros(n)
    dd = np.abs(hd[:n - 12] - hd[12:]) % 360
    tn[6:n - 6] = np.minimum(dd, 360 - dd)
    ok = np.zeros(n, bool)
    ok[max(start + 6, 6):min(stop, n - 6)] = True
    ok &= path >= 1.5
    ok &= np.array([sp[max(0, i - 9):i - 5].max() if i >= 6 else 0.0 for i in range(n)]) > 2.5
    hits = np.flatnonzero(ok & (tn >= 30))
    if not len(hits):
        return None
    win = np.arange(hits[0], min(hits[0] + 8, n - 6))
    win = win[ok[win]] if ok[win].any() else np.array([hits[0]])
    i = int(win[np.argmax(tn[win])])
    ap = max(0, i - 5) + int(np.argmin(sp[max(0, i - 5):min(n, i + 6)]))
    i_in = max(0, ap - 10) + int(np.argmax(sp[max(0, ap - 10):ap + 1]))
    return dict(bg=float(acc[max(0, i - 5):min(n, i + 6)].mean() / G), v_in=float(sp[i_in]), v_ap=float(sp[ap]))


ev = pd.read_parquet(P / "combine_events.parquet")
ev = ev[ev.drill_name.isin(FAM) & ev.position_group.isin(["WR", "TE"])]
fr = pd.read_parquet(P / "combine_frames.parquet", filters=[("event_id", "in", ev.event_id.tolist())])
rows = []
for eid, d in fr.groupby("event_id", sort=False):
    pl = d[d.entity_type == "PLAYER"].sort_values("frame")
    ball = d[d.entity_type == "BALL"]
    stop = len(pl)
    if len(ball):
        m = pl[["frame", "x", "y"]].merge(ball[["frame", "x", "y"]], on="frame", suffixes=("", "_b"))
        caught = m.frame[np.hypot(m.x - m.x_b, m.y - m.y_b) < 1.0]
        if len(caught):
            stop = int(np.searchsorted(pl.frame.to_numpy(), caught.iloc[0]))
    r = cut(pl.x.to_numpy(), pl.y.to_numpy(), 0, stop)
    if r:
        rows.append(dict(event_id=eid, **r))
reps = pd.DataFrame(rows).merge(ev[["event_id", "nfl_id", "draft_year", "drill_name", "attempt_no"]], on="event_id")
reps["family"] = reps.drill_name.map(FAM)
reps["bg_z"] = reps.groupby(["drill_name", "draft_year"]).bg.transform(z)
cell = reps.groupby(["nfl_id", "drill_name", "family"], as_index=False).bg_z.mean()
rcv = cell.groupby("nfl_id").bg_z.agg(["mean", "size"]).rename(columns={"mean": "zbar", "size": "nd"})
ICC = oneway_icc(cell.bg_z, cell.nfl_id)
VT = cell.bg_z.var()


def score(zbar, nd, icc):
    icc = max(icc, 0.01)
    vtr, ver = icc * VT, (1 - icc) * VT
    pv = 1 / (1 / vtr + nd / ver)
    return pv * nd * zbar / ver / np.sqrt(vtr), vtr / (vtr + ver / nd)


rcv["BG"] = score(rcv.zbar, rcv.nd, ICC)[0]

# ---------------------------------------------------------------------------------------------- baseline covariates, SOE
pl_ = pd.read_parquet(P / "players.parquet")
cr = pd.read_parquet(P / "combine_results.parquet").set_index("nfl_id")
pp = pl_.set_index("nfl_id")[["position_group", "draft_year", "draft_round", "draft_overall_pick", "display_name"]]
pp = pp[pp.position_group.isin(["WR", "TE"])].join(cr[["forty", "combine_weight", "ngs_athleticism_score"]])
pp["logpick"] = np.log(pp.draft_overall_pick.fillna(262))
pp["te"] = (pp.position_group == "TE").astype(float)
for c in ("forty", "combine_weight", "ngs_athleticism_score"):
    pp[c + "_s"] = pp.groupby(["draft_year", "position_group"])[c].transform(z)
    pp[c + "_f"] = pp[c + "_s"].fillna(0.0)
Xv = rcv.join(pp, how="inner")
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
B0 = Xv.join(car(ss), how="inner")
B0["career_soe_in"] = IN_YD * B0.ssoe
# (0) rebuilt input equals the CSV analysis.py wrote
CSV = pd.read_csv(OUT / "baseline_inputs.csv").set_index("nfl_id")
assert list(CSV.index) == list(B0.index), "baseline_inputs.csv receivers differ from the rebuild"
for c in BASE + ["BG", "career_soe_in", "routes"]:
    assert np.allclose(CSV[c], B0[c], atol=1e-5), f"baseline_inputs.csv column {c} differs from the rebuild"
print(f"ok  baseline_inputs.csv rebuilt from the raw frames ({len(B0)} receivers, Break G included)")
res["n_main"] = len(B0)

# ---------------------------------------------------------------------------------------------- (1) reliability
ha = cell[cell.drill_name.isin(HALF)].groupby("nfl_id").bg_z.mean()
hb = cell[~cell.drill_name.isin(HALF)].groupby("nfl_id").bg_z.mean()
hh = pd.concat([ha, hb], axis=1, keys=["a", "b"]).dropna()
res["rel_bg"], res["rel_lo"], res["rel_hi"] = boot_spear(hh.a, hh.b, "rel_bg")
iccs, npair = [], 0
for dn, d in reps.sort_values("attempt_no").groupby("drill_name"):
    two = d.groupby("nfl_id").head(2)
    two = two[two.groupby("nfl_id").nfl_id.transform("size") == 2]
    npair += two.nfl_id.nunique()
    if two.nfl_id.nunique() >= 5:
        iccs.append(oneway_icc(two.bg_z, two.nfl_id))
res["icc_med"], res["icc_pairs"], res["icc_bg"] = float(np.median(iccs)), npair, ICC
grp_c = [np.flatnonzero(cell.nfl_id.to_numpy() == p) for p in pd.unique(cell.nfl_id)]
cv, cid = cell.bg_z.to_numpy(float), cell.nfl_id.to_numpy()
rng = seed("icc_boot")
iccb = []
for _ in range(NBOOT // 4):
    pick = rng.integers(0, len(grp_c), len(grp_c))
    iccb.append(oneway_icc(np.concatenate([cv[grp_c[i]] for i in pick]),
                           np.concatenate([np.full(len(grp_c[i]), k) for k, i in enumerate(pick)])))
iccb = np.array(iccb)
res["icc_bg_lo"], res["icc_bg_hi"] = q(iccb, 5), q(iccb, 95)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mm_ = smf.mixedlm("bg_z ~ 0 + C(dy)", reps.assign(dy=reps.drill_name + "_" + reps.draft_year.astype(str)),
                      groups="nfl_id", re_formula="1", vc_formula={"pxd": "0 + C(drill_name)"}).fit(reml=True)
vp, vpd, ve = float(mm_.cov_re.iloc[0, 0]), float(mm_.vcomp[0]), float(mm_.scale)
res["rel_7x1"], res["rel_7x2"] = vp / (vp + (vpd + ve) / 7), vp / (vp + (vpd + ve / 2) / 7)
res["cuts70"], res["cuts80"] = math.ceil(0.7 / 0.3 * (vpd + ve) / vp), math.ceil(4 * (vpd + ve) / vp)


# ---------------------------------------------------------------------------------------------- (2) held-out tests
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


c0 = B0[["ssoe", "routes"]]
V = parts(B0, c0)
XB, Yv, Wv, CLv, BGv = V
h = r2(*V)
res.update(bl_r2=h[0], dr2=h[2], bl_rmse=h[3])
rng = seed("ho_r2")
hb_ = np.array([r2(*(a[ix] for a in V)) for ix in (strat(CLv, rng) for _ in range(NBOOT // 2))])
res.update(dr2_lo=q(hb_[:, 2], 5), dr2_hi=q(hb_[:, 2], 95))
res["ho"] = slope(*V)


def slope_sm(B, c, w_extra=None):
    """Second implementation: per held-out class, statsmodels WLS of SOE on the baseline (other classes); then one
    statsmodels WLS of the held-out residual on class dummies + Break G."""
    import statsmodels.api as sm
    df = B[BASE + ["BG", "draft_year"]].join(c)
    w = df.routes.to_numpy(float) * (1 if w_extra is None else np.asarray(w_extra, float))
    resid = pd.Series(np.nan, index=df.index)
    for k in CLASSES:
        t = (df.draft_year == k).to_numpy()
        fit = sm.WLS(df.ssoe[~t], sm.add_constant(df[BASE][~t], has_constant="add"), weights=w[~t]).fit()
        resid[t] = df.ssoe[t] - fit.predict(sm.add_constant(df[BASE][t], has_constant="add"))
    Z = pd.get_dummies(df.draft_year, dtype=float).assign(BG=df.BG)
    return IN_YD * sm.WLS(resid, Z, weights=w).fit().params["BG"]


res_sm = slope_sm(B0, c0)
assert abs(res_sm - res["ho"]) < 1e-8, f"statsmodels LOCO slope {res_sm} differs from numpy {res['ho']}"
print(f"ok  held-out slope, numpy vs statsmodels implementation: {res['ho']:.6f} vs {res_sm:.6f}")
rng = seed("ho_slope")
sb = np.array([slope(*(a[ix] for a in V)) for ix in (strat(CLv, rng) for _ in range(NBOOT))])
res.update(ho_lo=q(sb, 5), ho_hi=q(sb, 95), tost_up=float(np.mean(sb >= SESOI)), tost_dn=float(np.mean(sb <= -SESOI)))
res["tost"] = max(res["tost_up"], res["tost_dn"])
ZB, ND = B0.zbar.to_numpy(float), B0.nd.to_numpy(float)
RAWZ = (ZB - ZB.mean()) / ZB.std(ddof=1)
res["ho_raw"] = slope(XB, Yv, Wv, CLv, RAWZ)
rng = seed("ho_raw")
rb = [slope(*(a[ix] for a in (XB, Yv, Wv, CLv, RAWZ))) for ix in (strat(CLv, rng) for _ in range(NBOOT))]
res.update(ho_raw_lo=q(rb, 5), ho_raw_hi=q(rb, 95))
rng = seed("ho_eiv")
eiv = []
for b in range(NBOOT):
    ix = strat(CLv, rng)
    eiv.append(slope(XB[ix], Yv[ix], Wv[ix], CLv[ix], score(ZB[ix], ND[ix], iccb[b % len(iccb)])[0]))
res.update(eiv_lo=q(eiv, 5), eiv_hi=q(eiv, 95))
se = float(np.std(eiv, ddof=1))
res["mde"] = (norm.ppf(0.95) + norm.ppf(0.8)) * se
res["pow3"] = 100 * norm.cdf(SESOI / se - norm.ppf(0.95))
res["rel_main"] = float(np.mean(score(ZB, ND, ICC)[1]))

# ---------------------------------------------------------------------------------------------- (3) hierarchical
ms = ss[ss.nfl_id.isin(B0.index) & (ss.routes >= 50)][["nfl_id", "season", "ssoe", "routes"]].join(
    B0[BASE + ["BG", "draft_year"]], on="nfl_id")
ms["y_in"], ms["one"] = IN_YD * ms.ssoe, 1
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    hm = smf.mixedlm("y_in ~ BG + " + " + ".join(BASE) + " + C(season)", ms, groups="one", re_formula="0",
                     vc_formula={"rcv": "0 + C(nfl_id)", "cls": "0 + C(draft_year)"}).fit(reml=True)
hmb, hse = float(hm.fe_params["BG"]), float(hm.bse_fe["BG"])
res.update(hm=hmb, hm_lo=hmb - 1.645 * hse, hm_hi=hmb + 1.645 * hse, hm_n=len(ms), hm_np=ms.nfl_id.nunique())

# ---------------------------------------------------------------------------------------------- (4) pick price
zz = (Xv.BG / Xv.BG.std()).to_numpy()
Ap = np.column_stack([np.ones(len(Xv)), Xv.te, zz])
lp = Xv.logpick.to_numpy()
bp = wls(Ap, lp, np.ones(len(Xv)))[-1]
rng = seed("price")
bpb = [wls(Ap[ix], lp[ix], np.ones(len(ix)))[-1] for ix in (rng.integers(0, len(Xv), len(Xv)) for _ in range(NBOOT))]
res.update(price=-100 * (np.exp(bp) - 1), price_lo=-100 * (np.exp(q(bpb, 95)) - 1),
           price_hi=-100 * (np.exp(q(bpb, 5)) - 1))

# ---------------------------------------------------------------------------------------------- (4b) IPW, agility
from sklearn.linear_model import LogisticRegression  # noqa: E402

inc = Xv.assign(inc=Xv.index.isin(B0.index).astype(int))
Ai = np.column_stack([inc.logpick, inc.te, inc.BG, inc.draft_year == 2024, inc.draft_year == 2025]).astype(float)
Ai = (Ai - Ai.mean(0)) / Ai.std(0)
pi = pd.Series(LogisticRegression(C=1.0).fit(Ai, inc.inc).predict_proba(Ai)[:, 1], index=inc.index)
wi = np.minimum(1 / pi.reindex(B0.index).to_numpy(), 10.0)
res["ipw"] = slope(XB, Yv, Wv * wi, CLv, BGv)
assert abs(slope_sm(B0, c0, wi) - res["ipw"]) < 1e-8
for k, col in (("cone", "three_cone"), ("shut", "short_shuttle")):
    za = -pp.join(cr[[col]], rsuffix="_r").groupby(["draft_year", "position_group"])[col].transform(z)
    Bm = B0.assign(a=za.reindex(B0.index))
    Bm = Bm[Bm.a.notna()]
    Va = parts(Bm, Bm[["ssoe", "routes"]])
    av = z(Bm.a).to_numpy(float)
    res[f"a_{k}"] = slope(*Va[:4], av)
    res[f"a_{k}_n"] = len(Bm)
    rng = seed("agil_" + k)
    ab = [slope(*(a[ix] for a in (*Va[:4], av))) for ix in (strat(Va[3], rng) for _ in range(NBOOT // 2))]
    res[f"a_{k}_lo"], res[f"a_{k}_hi"] = q(ab, 5), q(ab, 95)
    globals()["AG_" + k] = za
res["proxy_rho"] = spearmanr(reps.bg, reps.v_in - reps.v_ap)[0]

# ---------------------------------------------------------------------------------------------- (5) NFL frames
gp = pd.read_parquet(P / "game_plays_sample.parquet")
gp = gp[gp.week.between(1, 18) & gp.nfl_id.isin(pp.index)]
keys = ["game_id", "play_id", "nfl_id"]
gf = pd.read_parquet(P / "game_frames_sample.parquet", columns=keys + ["frame", "x", "y", "s", "event"]).merge(
    gp[keys], on=keys)
mx = gf.groupby(["nfl_id", "game_id", "play_id"]).s.max().groupby("nfl_id")
vtop = mx.quantile(0.9)[mx.size() >= 8]
brk = []
for k, d in gf.groupby(keys, sort=False):
    d = d.sort_values("frame")
    e = d.event.fillna("").to_numpy()
    if not ((e == "ball_snap").any() and (e == "pass_forward").any()):
        continue
    r = cut(d.x.to_numpy(), d.y.to_numpy(), int(np.argmax(e == "ball_snap")),
            min(len(d), int(np.argmax(e == "pass_forward")) + 6))
    if r:
        brk.append(dict(zip(keys, k), **r))
brk = pd.DataFrame(brk).merge(gp[keys + ["route_ran"]], on=keys)
brk_all = brk.copy()
brk = brk[brk.route_ran.isin(ROUTES)].join(pp.position_group, on="nfl_id")
brk["gz"] = brk.groupby(["route_ran", "position_group"]).bg.transform(z)
brk = brk.sort_values(["nfl_id", "game_id", "play_id"])
brk["half"] = brk.groupby("nfl_id").cumcount() % 2
gm = brk.groupby("nfl_id").gz.agg(["mean", "size"])
cand = Xv[["BG", "forty_s", "position_group"]].join(gm, how="inner").join(vtop.rename("vtop"), how="inner")
cand = cand[(cand["size"] >= 8) & cand.forty_s.notna()]
SI = pd.read_csv(OUT / "speed_inputs.csv").set_index("nfl_id")
assert sorted(SI.index) == sorted(cand.index), "speed_inputs.csv receivers differ from the rebuild"
assert np.allclose(SI.nfl_bg_z_raw, cand.loc[SI.index, "mean"], atol=1e-5), "NFL Break G differs from the rebuild"
print(f"ok  speed_inputs.csv rebuilt from the raw frames ({len(cand)} receivers, NFL Break G included)")
cand = cand.loc[SI.index]
cand["vz"] = cand.groupby("position_group").vtop.transform(z)
res["spc"], res["spc_lo"], res["spc_hi"] = boot_spear(-cand.forty_s, cand.vz, "spc")
res["f1_n"] = len(cand)
res["bgc"], res["bgc_lo"], res["bgc_hi"] = boot_spear(z(cand.BG), z(cand["mean"]), "bgc")
hh2 = brk.groupby(["nfl_id", "half"]).gz.mean().unstack().dropna()
grel = spearmanr(hh2[0], hh2[1])[0]
rel_n = 2 * grel / (1 + grel)
rel_c = float(np.mean(score(rcv.loc[cand.index, "zbar"], rcv.loc[cand.index, "nd"], ICC)[1]))
res.update(grel=grel, rel_nfl=rel_n, rel_c=rel_c, ceil=np.sqrt(rel_c * rel_n), exp50=0.5 * np.sqrt(rel_c * rel_n),
           bgc_dis=res["bgc"] / np.sqrt(rel_c * rel_n))

for k in ("cone", "shut"):
    j = pd.concat([globals()["AG_" + k].rename("a"), gm.loc[gm["size"] >= 8, "mean"]], axis=1, join="inner").dropna()
    res[f"tr_{k}"], res[f"tr_{k}_lo"], res[f"tr_{k}_hi"] = boot_spear(j.a, j["mean"], "tr_" + k)
    res[f"tr_{k}_n"] = len(j)
# entry speed: wide receivers with 200+ career routes and 5+ graded NFL breaks, shared route families
nb = brk.groupby("nfl_id").size()
ex = B0[(B0.position_group == "WR") & (B0.routes >= 200)]
ex = ex[nb.reindex(ex.index).fillna(0).to_numpy() >= 5]
gaps = []
for pid_ in ex.index:
    rr, nn_ = reps[reps.nfl_id == pid_], brk[brk.nfl_id == pid_]
    fam = set(rr.family) & set(nn_.route_ran)
    if fam:
        gaps.append(rr[rr.family.isin(fam)].v_in.median() - nn_[nn_.route_ran.isin(fam)].v_in.median())
res["ex_n"], res["ex_shared"], res["ex_faster"] = len(ex), len(gaps), int(np.sum(np.array(gaps) > 0))

# ---------------------------------------------------------------------------------------------- (6) day 3
def pairs(s, mr=100):
    out = []
    for a, b in ((2023, 2024), (2024, 2025)):
        j = s[s.season == a].set_index("nfl_id")[["ssoe", "routes"]].join(
            s[s.season == b].set_index("nfl_id")[["ssoe", "routes"]], rsuffix="_2", how="inner")
        j.columns = ["y1", "r1", "y2", "r2"]
        j["s1"] = a
        out.append(j)
    pr = pd.concat(out).join(pp[["draft_round", "draft_overall_pick", "draft_year", "logpick", "te"]])
    pr = pr[(pr.r1 >= mr) & (pr.r2 >= mr)].copy()
    pr["pid"] = pr.index
    pr["day3"] = pr.draft_round.isna() | (pr.draft_round >= 4)
    pr["lr1"], pr["hit"], pr["pick"] = np.log(pr.r1), pr.y2 > 0, pr.draft_overall_pick.fillna(262)
    return pr.reset_index(drop=True)


def shrink_par(s):
    s = s[s.routes >= 50]
    mu = float(np.average(s.ssoe, weights=s.routes))
    pv = s.pivot_table(index="nfl_id", columns="season", values="ssoe")
    se_ = sorted(s.season.unique())
    cp = pd.concat([pv[[a, a + 1]].dropna().set_axis(["a", "b"], axis=1) for a in se_ if a + 1 in se_])
    t2 = float(np.mean((cp.a - mu) * (cp.b - mu)))
    iv = 1.0 / s.routes.to_numpy(float)
    return mu, t2, float(np.sum(((s.ssoe.to_numpy() - mu) ** 2 - t2) * iv) / np.sum(iv ** 2))


def ols(df, cols):
    A = np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
    return np.linalg.lstsq(A, df.y2.to_numpy(float), rcond=None)[0]


F1 = ["logpick", "te", "lr1", "y1s"]
cls_of = pl_.set_index("nfl_id").draft_year


def loco(pr, s):
    out = pr.copy()
    out[["f1", "y1s"]] = np.nan
    for c in sorted(out.draft_year.unique()):
        t = (out.draft_year == c).to_numpy()
        mu, t2, s2 = shrink_par(s[cls_of.reindex(s.nfl_id).to_numpy() != c])
        p2 = out.assign(y1s=mu + t2 / (t2 + s2 / out.r1) * (out.y1 - mu))
        out.loc[t, "f1"] = np.column_stack([np.ones(t.sum()), p2[t][F1].to_numpy(float)]) @ ols(p2[~t], F1)
        out.loc[t, "y1s"] = p2.y1s.to_numpy()[t]
    return out


def contrasts(d3w):
    d = d3w.reset_index(drop=True)
    rows_ = []
    for s1 in sorted(d.s1.unique()):
        ii = np.where(d.s1.to_numpy() == s1)[0]
        for x in range(len(ii)):
            for k in range(x + 1, len(ii)):
                a, b = ii[x], ii[k]
                pk = a if (d.pick.iloc[a], -d.r1.iloc[a]) < (d.pick.iloc[b], -d.r1.iloc[b]) else b
                so = a if d.y1s.iloc[a] > d.y1s.iloc[b] else b
                rows_.append(dict(a=a, b=b, gap=abs(d.pick.iloc[a] - d.pick.iloc[b]), pick=pk, soe=so))
    return d, pd.DataFrame(rows_)


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
pos = d3[d3.y1 > 0]
res.update(fq_n=len(pos), fq_k=int(pos.hit.sum()), fq_pct=100 * pos.hit.mean(),
           fq_lo=100 * beta.ppf(0.05, pos.hit.sum() + 0.5, len(pos) - pos.hit.sum() + 0.5),
           fq_hi=100 * beta.ppf(0.95, pos.hit.sum() + 0.5, len(pos) - pos.hit.sum() + 0.5))
dw, ct = contrasts(d3[d3.te == 0])
y2 = IN_YD * dw.y2.to_numpy()
res["ov_wr"], res["ov_be"], res["ov_n"] = ov(ct, y2)
pid = dw.pid.to_numpy()
rng = seed("override")
wr_b = []
for _ in range(NBOOT):
    u = np.unique(pid)
    m = pd.Series(rng.choice(u, len(u))).value_counts().reindex(pid).fillna(0).to_numpy()
    wr_b.append(ov(ct, y2, m[ct.a.to_numpy()] * m[ct.b.to_numpy()])[0])
res.update(ov_wr_lo=q(wr_b, 5), ov_wr_hi=q(wr_b, 95))

# ---------------------------------------------------------------------------------------------- (7) sensitivity
vals = [res["ho"]]
for s_, mr in ((ss, 50), (ss, 100), (soe(False), 0)):
    c = car(s_, mr)
    B = B0.loc[B0.index.intersection(c.index)]
    vals.append(slope(*parts(B, c.reindex(B.index))))
vals.append(slope(XB, Yv, np.ones(len(Yv)), CLv, BGv))
wr = (B0.te == 0).to_numpy()
vals.append(slope(XB[wr][:, :-1], Yv[wr], Wv[wr], CLv[wr], BGv[wr]))
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
# pooled receiver-level day-3 test (50+ routes both seasons)
f5 = loco(pairs(ss, 50), ss)
p5 = f5[f5.day3].reset_index(drop=True)
h5 = p5.groupby("s1").s1.transform("size") // 2
p5["top_soe"] = p5.groupby("s1").f1.rank(ascending=False, method="first") <= h5
p5["k"] = p5.pick * 1e4 - p5.r1
p5["top_pick"] = p5.groupby("s1").k.rank(method="first") <= h5
res.update(p_n=len(p5), p_np=p5.pid.nunique(), p_soe=100 * p5[p5.top_soe].hit.mean(),
           p_pick=100 * p5[p5.top_pick].hit.mean())
res["p_d"] = res["p_soe"] - res["p_pick"]
g5 = [np.where(p5.pid.to_numpy() == p)[0] for p in pd.unique(p5.pid)]
rng = seed("d50")
d5 = []
for _ in range(NBOOT):
    bb = p5.iloc[np.concatenate([g5[i] for i in rng.integers(0, len(g5), len(g5))])]
    d5.append(100 * bb[bb.top_soe].hit.mean() - 100 * bb[bb.top_pick].hit.mean())
res.update(p_d_lo=q(d5, 5), p_d_hi=q(d5, 95))

# ---------------------------------------------------------------------------------------------- compare
RESAMPLED = {"tost", "tost_up", "tost_dn", "mde", "pow3"}


def is_resampled(k):
    return k.endswith(("_lo", "_hi")) or k in RESAMPLED


def width(k):
    base = k[:-3] if k.endswith(("_lo", "_hi")) else None
    if base and base + "_lo" in NUM and base + "_hi" in NUM:
        return abs(float(NUM[base + "_hi"]) - float(NUM[base + "_lo"]))
    if k.startswith(("eiv_", "ho_")):
        return abs(float(NUM["ho_hi"]) - float(NUM["ho_lo"]))
    return 0.0


bad, drift = [], []
for k, v in res.items():
    ref = NUM[k]
    nd = len(ref.split(".")[1]) if "." in ref else 0
    got = f"{v:.{nd}f}" if nd else str(int(round(v)))
    if got.startswith("-") and float(got) == 0:
        got = got[1:]
    if is_resampled(k):
        unit = {0: 2.0, 1: 0.2, 2: 0.02, 3: 0.002}.get(nd, 0.02)
        tol = max(0.15 * width(k), unit)
        d = abs(float(got) - float(ref))
        ok = d <= tol + 1e-9
        drift.append((d / width(k) if width(k) else 0.0, k))
        print(f"{'ok ' if ok else 'BAD'} {k:12s} analysis {ref:>8s}  verify {got:>8s}  (new seeds, tol {tol:.3g})")
    else:
        ok = got == ref
        print(f"{'ok ' if ok else 'BAD'} {k:12s} analysis {ref:>8s}  verify {got:>8s}")
    bad += [] if ok else [k]
mx = max(drift)
print(f"{len(res) - len(bad)} of {len(res)} numbers reproduced from the raw tables "
      f"({sum(not is_resampled(k) for k in res)} exact; {len(drift)} resampled with own seeds, largest drift "
      f"{100 * mx[0]:.0f}% of its interval width: {mx[1]})")
sys.exit(1 if bad else 0)
