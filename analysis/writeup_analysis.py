"""Round 21, revised (revision of round-21 cand_2, itself a revision of the round-20 winner): do Combine route cuts (Break G) carry to NFL separation,
why not, and what should a staff do instead?

Run from the repo root (about three minutes):
    uv run python writeup/candidates/round_21/revised/analysis.py
Independent recomputation from the raw tables (own seeds, second LOCO implementation in statsmodels):
    uv run python writeup/candidates/round_21/revised/verify.py

Modules:
- core.py      definitions and builders (Break G, SOE, NFL snaps, season pairs, held-out tests); no outputs;
- analysis.py  this file: every number quoted, the figures, the render of writeup.md and appendix.md;
- appendix.py  robustness quoted only in appendix.md (equivalence tests, sensitivity grid);
- verify.py    independent rebuild of the inputs and recomputation of the quoted numbers.

Writes next to this file: numbers.json (exactly the numbers writeup.md quotes), claims.json, all_numbers.json (every
number, including appendix.md), figures/*.png (1600 px wide, 150 dpi), writeup.md and appendix.md (rendered from
writeup_template.md / appendix_template.md), and baseline_inputs.csv / speed_inputs.csv (verify.py rebuilds both).

Round-21 changes (round-20 judges): coach proxy for Break G (speed shed into the plant); named drill-rep vs NFL-rep
table; no-tracking benchmarks (official 3-cone and short shuttle) for both the season test and the transfer test;
inverse-probability weights for having NFL routes; multiplicity count; day-3 test pooled to receiver level (50+ route
seasons) with a bar fixed before the pooled run, and the separation-ordered list dropped when it misses the bar;
hero figure = per-receiver drill vs NFL cut traces overlaid on one entry-speed-normalised axis; forest plot alone;
transfer figure absorbs the protocol-subset panel; reliability curve and coverage splits moved to the appendix.

Round-21 revision (round-21 judges): Break G stated in yd/s of velocity shed and redirected (on-film unit); held-out
error reported under both definitions; flat-prior P(|slope| < 3 in) for Break G, 3-cone and shuttle; entry-speed
explanation tested (NFL breaks entered at drill speed; entry speed removed on both sides); random vs top-EPA sampling
halves; day-3 test at receiver level; 2025 day-3 decision table in the body; Fig 1 = three player cards; Fig 3 = the
two scatters only (subset panel -> appendix figB); Fig 2 tidied; figA shows both reliability definitions.
"""
from __future__ import annotations

import json
import math
import textwrap
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import norm  # noqa: E402

import appendix  # noqa: E402
from core import *  # noqa: E402,F403

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
for old in FIG.glob("*.png"):
    old.unlink()
FIGW = 1600 / 150
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2", "verm": "#D55E00",
      "purple": "#CC79A7", "grey": "#8C8C8C", "black": "#000000"}
GO_STYLE = dict(color=OI["blue"], marker="o")
HOLD_STYLE = dict(color=OI["orange"], marker="s")
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12.5, "figure.dpi": 150, "savefig.dpi": 150,
                     "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.labelsize": 12})

NUM: dict[str, str] = {}
CLAIM: dict[str, str] = {}


def put(key, val, nd=2, claim=""):
    if key in NUM:
        raise KeyError(f"duplicate key {key}")
    if isinstance(val, (int, np.integer)):
        s = str(int(val))
    else:
        s = f"{float(val):.{nd}f}"
        if s.startswith("-") and float(s) == 0:
            s = s[1:]
    NUM[key], CLAIM[key] = s, claim or key
    return s


def save(fig, name):
    assert abs(fig.get_size_inches()[0] - FIGW) < 1e-6
    fig.savefig(FIG / name, dpi=150, facecolor="white")
    plt.close(fig)


def tag(name):
    return name.split()[-1].lower().replace("'", "").replace(".", "")


def mm(s):
    return s.replace("-", "−")


# ============================================================================= 1. Combine Break G and its reliability
C = build_combine()
cb, cpd, cpl, INFO = C["cb"], C["cpd"], C["cpl"], C["info"]
put("n_reps", len(cb), 0, "graded Combine route-drill reps (WR/TE)")
put("n_combine", len(cpl), 0, "receivers with a Break G score")
put("why_reps", int(cb.groupby("nfl_id").size().median()), 0, "median graded route-drill reps per receiver")
put("why_single", 100 * float((cb.groupby(["nfl_id", "drill_name"]).size() == 1).mean()), 0,
    "% of receiver-drill cells with one graded rep")
put("g_sd", float(INFO["bg"][1] * cb.groupby(["drill_name", "draft_year"]).bg.std().mean()), 3,
    "1 sd of true receiver Break G, in g")
# reliability across disjoint drill sets (five drills vs the other five)
ha = cpd[cpd.drill_name.isin(HALF_A)].groupby("nfl_id").bg_z.mean()
hb = cpd[~cpd.drill_name.isin(HALF_A)].groupby("nfl_id").bg_z.mean()
HH = pd.concat([ha, hb], axis=1, keys=["a", "b"]).dropna()
REL = boot_rho(HH.a, HH.b, "rel_bg")
put("rel_bg", REL[0], 2, "Break G split-half (five drills vs five)"); put("rel_n", REL[3], 0, "receivers in it")
put("rel_lo", REL[1], 2, "90% lower"); put("rel_hi", REL[2], 2, "90% upper")
# same drill run twice: ICC of the first two reps, per drill with 5+ receivers who ran it twice; median over drills
icc_rows, n_pairs = [], 0
for dn, d in cb.sort_values("attempt_no").groupby("drill_name"):
    two = d.groupby("nfl_id").head(2)
    two = two[two.groupby("nfl_id").nfl_id.transform("size") == 2]
    k = two.nfl_id.nunique()
    n_pairs += k
    if k >= 5:
        icc_rows.append(icc1(two, "bg_z"))
put("icc_med", float(np.median(icc_rows)), 2, "median same-drill test-retest ICC of Break G")
put("icc_pairs", n_pairs, 0, "receiver-drill pairs with two graded reps")
put("icc_drills", len(icc_rows), 0, "drills with 5+ repeat pairs")
# rep-level mixed model: receiver, receiver x drill and rep variance -> reliability by drills x reps (protocol advice)
cbm = cb.assign(dy=cb.drill_name + "_" + cb.draft_year.astype(str))
MM = quiet_mixedlm("bg_z ~ 0 + C(dy)", cbm, groups="nfl_id", re_formula="1", vc_formula={"pxd": "0 + C(drill_name)"})
VP, VPD, VE = float(MM.cov_re.iloc[0, 0]), float(MM.vcomp[0]), float(MM.scale)


def rel_dk(d, k):
    return VP / (VP + (VPD + VE / k) / d)


put("rel_7x1", rel_dk(7, 1), 2, "reliability of a 7-drill Break G score, one rep each")
put("rel_7x2", rel_dk(7, 2), 2, "same, two reps each")
put("vpd_pct", 100 * VPD / (VP + VPD + VE), 0, "receiver-by-drill share of rep variance, %")
put("rep_icc", VP / (VP + VPD + VE), 2, "share of one rep's variance that is the receiver")
CUTS70 = math.ceil(0.7 / 0.3 * (VPD + VE) / VP)
CUTS80 = math.ceil(0.8 / 0.2 * (VPD + VE) / VP)
put("t70", 0.7, 2, "reliability target"); put("t80", 0.8, 2, "reliability target")
put("cuts70", CUTS70, 0, "graded cuts per receiver for reliability 0.70")
put("cuts80", CUTS80, 0, "graded cuts per receiver for reliability 0.80")
# receiver-level ICC of the drill-average (used for the shrinkage) and its receiver bootstrap
ICC_BG = INFO["bg"][0]
put("icc_bg", ICC_BG, 2, "between-drill ICC of Break G (drill means), drives the shrinkage")
_cpd_groups = clusters(cpd.nfl_id)
_cpd_vals = cpd.bg_z.to_numpy(float)


def icc_from_groups(groups_list):
    vals = [_cpd_vals[g_] for g_ in groups_list]
    k = np.array([len(v) for v in vals], float)
    n, m = k.sum(), len(vals)
    means = np.array([v.mean() for v in vals])
    gm = np.concatenate(vals).mean()
    k0 = (n - (k ** 2).sum() / n) / (m - 1)
    msb = (k * (means - gm) ** 2).sum() / (m - 1)
    msw = sum(((v - v.mean()) ** 2).sum() for v in vals) / (n - m)
    return (msb - msw) / (msb + (k0 - 1) * msw)


r_ = rng_for("icc_boot")
ICCB = np.array([icc_from_groups([_cpd_groups[i] for i in r_.integers(0, len(_cpd_groups), len(_cpd_groups))])
                 for _ in range(NBOOT // 4)])
put("icc_bg_lo", pct(ICCB, 5), 2, "90% lower"); put("icc_bg_hi", pct(ICCB, 95), 2, "90% upper")


def bg_score(zbar, nd, icc):
    """Regression-calibrated (empirical-Bayes) receiver score in sd of TRUE Break G for a given between-drill ICC."""
    icc = max(icc, 0.01)
    vt = cpd.bg_z.var()
    var_true, var_err = icc * vt, (1 - icc) * vt
    post_var = 1 / (1 / var_true + nd / var_err)
    return post_var * (nd * zbar / var_err) / np.sqrt(var_true), var_true / (var_true + var_err / nd)


# ============================================================================= 2. season SOE and the held-out test
players, pb, X = build_players(cpl)
ss_all = pd.read_parquet(P / "player_season_stats.parquet")
ss = season_soe(ss_all)
CAR = career(ss)
D = X.join(CAR, how="inner")
put("n_main", len(D), 0, "receivers with Break G and NFL routes (held-out test)")
put("n_seasons", int(ss.nfl_id.isin(D.index).sum()), 0, "player-seasons behind career SOE")
put("routes_med", int(D.routes.median()), 0, "median career routes")
XB, Y, W, CL, BGV = loco_parts(D)
HR = ho_r2(XB, Y, W, CL, BGV)
rng = rng_for("ho_r2")
HRB = np.array([ho_r2(*(a[ix] for a in (XB, Y, W, CL, BGV))) for ix in (strat_ix(CL, rng) for _ in range(NBOOT // 2))])
put("bl_r2", HR[0], 2, "baseline (log pick, 40, weight, NGS athleticism, TE) held-out R2, career SOE")
put("dr2", HR[2], 3, "R2 change from adding Break G, held out")
put("dr2_lo", pct(HRB[:, 2], 5), 3, "90% lower, class-stratified bootstrap"); put("dr2_hi", pct(HRB[:, 2], 95), 3, "upper")
put("bl_rmse", HR[3], 1, "baseline held-out error (RMSE), inches, per receiver, weighted by career routes")
# the same held-out error, unweighted (every receiver counts once, however few routes): the judges' refit gave 14.9
F0_ = np.empty(len(Y))
for c_ in CLASSES:
    t_ = CL == c_
    F0_[t_] = XB[t_] @ wls(XB[~t_], Y[~t_], W[~t_])
put("bl_rmse_unw", IN_YD * float(np.sqrt(np.mean((Y - F0_) ** 2))), 1, "baseline held-out RMSE, unweighted, inches")
put("soe_sd", IN_YD * float(np.sqrt(np.average((Y - np.average(Y, weights=W)) ** 2, weights=W))), 1,
    "route-weighted sd of career SOE across the tested receivers, inches")
HO = ho_slope(XB, Y, W, CL, BGV)
rng = rng_for("ho_slope")
HOB = np.array([ho_slope(*(a[ix] for a in (XB, Y, W, CL, BGV))) for ix in (strat_ix(CL, rng) for _ in range(NBOOT))])
put("ho", HO, 1, "held-out inches of season SOE per sd of true Break G (regression-calibrated score)")
put("ho_lo", pct(HOB, 5), 1, "90% lower"); put("ho_hi", pct(HOB, 95), 1, "90% upper")
put("sesoi", SESOI_IN, 0, "pre-set margin, inches per sd")


def rope(b, boot):
    """Posterior probability (%) that |slope| < 3 in: flat prior, normal likelihood centred on the estimate with the
    bootstrap sd as its standard error."""
    se_ = float(np.std(boot, ddof=1))
    return 100 * float(norm.cdf((SESOI_IN - b) / se_) - norm.cdf((-SESOI_IN - b) / se_))


put("rope_bg", rope(HO, HOB), 0, "posterior P(|Break G slope| < 3 in), %, flat prior")
# measurement error: (a) raw drill average, no correction, per sd of the observed score;
# (b) errors-in-variables bootstrap: receivers resampled AND the ICC redrawn from its own bootstrap each time
ZB, ND = D.bg_z.to_numpy(float), D.n_drills.to_numpy(float)
RAW = (ZB - ZB.mean()) / ZB.std(ddof=1)
HO_RAW = ho_slope(XB, Y, W, CL, RAW)
r_ = rng_for("ho_raw")
RAWB = np.array([ho_slope(*(a[ix] for a in (XB, Y, W, CL, RAW))) for ix in (strat_ix(CL, r_) for _ in range(NBOOT))])
put("ho_raw", HO_RAW, 1, "held-out slope on the raw drill average, per observed sd (attenuated)")
put("ho_raw_lo", pct(RAWB, 5), 1, "90% lower"); put("ho_raw_hi", pct(RAWB, 95), 1, "90% upper")
r_ = rng_for("ho_eiv")
EIVB = []
for b in range(NBOOT):
    ix = strat_ix(CL, r_)
    sc, _ = bg_score(ZB[ix], ND[ix], ICCB[b % len(ICCB)])
    EIVB.append(ho_slope(XB[ix], Y[ix], W[ix], CL[ix], sc))
EIVB = np.array(EIVB)
put("eiv_lo", pct(EIVB, 5), 1, "errors-in-variables 90% lower (ICC uncertainty propagated)")
put("eiv_hi", pct(EIVB, 95), 1, "errors-in-variables 90% upper")
REL_D = bg_score(ZB, ND, ICC_BG)[1]
put("rel_main", float(np.mean(REL_D)), 2, "mean reliability of the receiver Break G score, held-out sample")
SE_HO = float(np.std(EIVB, ddof=1))
put("mde", (norm.ppf(0.95) + norm.ppf(0.80)) * SE_HO, 1, "minimum detectable slope, 80% power, 90% two-sided")
put("pow3", 100 * norm.cdf(SESOI_IN / SE_HO - norm.ppf(0.95)), 0, "power to detect a 3-inch benefit, %")
MECH = [dict(k="bg", lab="Break G (primary)\nn={} receivers".format(len(D)), b=HO, lo=pct(HOB, 5),
             hi=pct(HOB, 95))]
# inverse-probability weights for being in the test (NFL routes with separation): ridge logistic (C=1, standardised
# inputs) on log pick, TE, class and Break G among the 149 scored receivers, weights truncated at 10, refitted in every
# resample (an unpenalised logit separates: inclusion is almost fixed by draft slot)
from sklearn.linear_model import LogisticRegression  # noqa: E402

INC = X.assign(inc=X.index.isin(D.index).astype(float))


def ipw_weights(df):
    A = np.column_stack([df.logpick, df.te, df.BG, df.draft_year == 2024, df.draft_year == 2025]).astype(float)
    A = (A - A.mean(0)) / np.where(A.std(0) > 0, A.std(0), 1)
    p_ = LogisticRegression(C=1.0).fit(A, df.inc.to_numpy(int)).predict_proba(A)[:, 1]
    return pd.Series(np.minimum(1 / np.clip(p_, 1e-6, 1), 10.0), index=df.index)


IPW = ipw_weights(INC).reindex(D.index).to_numpy()
HO_IPW = ho_slope(XB, Y, W * IPW, CL, BGV)
r_ = rng_for("ho_ipw")
IPWB = []
CLI = INC.draft_year.to_numpy()
for _ in range(NBOOT // 2):
    ii = strat_ix(CLI, r_)
    sub = INC.iloc[ii].reset_index()
    w_ = ipw_weights(sub.set_index(pd.RangeIndex(len(sub)))).to_numpy()
    keep = sub.inc.to_numpy() == 1
    s_ = sub[keep]
    car_ = CAR.reindex(s_.nfl_id)
    XB_ = np.column_stack([np.ones(keep.sum()), s_[BASE].to_numpy(float)])
    IPWB.append(ho_slope(XB_, car_.ssoe.to_numpy(float), car_.routes.to_numpy(float) * w_[keep],
                         s_.draft_year.to_numpy(), s_.BG.to_numpy(float)))
put("inc_all", len(INC), 0, "receivers with a Break G score (IPW denominator)")
put("ipw", HO_IPW, 1, "held-out slope with inverse-probability weights for being in the test")
put("ipw_lo", pct(IPWB, 5), 1, "90% lower"); put("ipw_hi", pct(IPWB, 95), 1, "90% upper")
MECH.append(dict(k="ipw", lab=f"Break G, weighted for who\nreached the NFL (n={len(D)})", b=HO_IPW, lo=pct(IPWB, 5),
                 hi=pct(IPWB, 95)))
# one hierarchical model: player-season SOE (50+ routes), baseline + season fixed, receiver and class random intercepts
MS = ss[ss.nfl_id.isin(D.index) & (ss.routes >= 50)][["nfl_id", "season", "ssoe", "routes"]].join(
    D[BASE + ["BG", "draft_year"]], on="nfl_id")
MS["y_in"], MS["one_g"] = IN_YD * MS.ssoe, 1
HM = quiet_mixedlm("y_in ~ BG + " + " + ".join(BASE) + " + C(season)", MS, groups="one_g", re_formula="0",
                   vc_formula={"rcv": "0 + C(nfl_id)", "cls": "0 + C(draft_year)"})
HMB, HMSE = float(HM.fe_params["BG"]), float(HM.bse_fe["BG"])
VC = dict(zip(HM.model.exog_vc.names, HM.vcomp))
put("hm", HMB, 1, "hierarchical model: inches per sd of Break G"); put("hm_lo", HMB - 1.645 * HMSE, 1, "90% lower (Wald)")
put("hm_hi", HMB + 1.645 * HMSE, 1, "90% upper"); put("hm_n", len(MS), 0, "player-seasons (50+ routes)")
put("hm_np", MS.nfl_id.nunique(), 0, "receivers in it")
put("hm_cls", 100 * VC["cls"] / (VC["rcv"] + VC["cls"] + HM.scale), 0, "draft-class share of variance, %")
MECH.append(dict(k="hm", lab=f"Break G, hierarchical\nn={len(MS)} player-seasons", b=HMB, lo=HMB - 1.645 * HMSE,
                 hi=HMB + 1.645 * HMSE))
for k, col, lab in (("decel", "DECEL_S", "Brake into the cut"), ("vapex", "V_APEX_S", "Speed through the cut"),
                    ("reacc", "REACCEL_S", "Drive out of the cut")):
    bv = D[col].to_numpy(float)
    b0 = ho_slope(XB, Y, W, CL, bv)
    r_ = rng_for("mech_" + k)
    bb = np.array([ho_slope(*(a[ix] for a in (XB, Y, W, CL, bv))) for ix in (strat_ix(CL, r_) for _ in range(NBOOT // 2))])
    MECH.append(dict(k=k, lab=f"{lab}\nn={len(D)}", b=b0, lo=pct(bb, 5), hi=pct(bb, 95)))
    put(f"m_{k}", b0, 1, f"held-out slope, {lab}"); put(f"m_{k}_lo", pct(bb, 5), 1, "90% lower")
    put(f"m_{k}_hi", pct(bb, 95), 1, "90% upper")
# no-tracking benchmark: official 3-cone and short shuttle (faster = +, z within class x position), same held-out model
CR = pd.read_parquet(P / "combine_results.parquet").set_index("nfl_id")
AGIL = {}
for k, col, lab in (("cone", "three_cone", "Official 3-cone"), ("shut", "short_shuttle", "Official short shuttle")):
    zc = -pb.join(CR[[col]]).groupby(["draft_year", "position_group"])[col].transform(zs)
    AGIL[k] = zc
    Dm = D.assign(a=zc.reindex(D.index))
    Dm = Dm[Dm.a.notna()]
    XB_, Y_, W_, CL_, _ = loco_parts(Dm)
    av = zs(Dm.a).to_numpy(float)
    b0 = ho_slope(XB_, Y_, W_, CL_, av)
    r_ = rng_for("agil_" + k)
    bb = np.array([ho_slope(*(a[ix] for a in (XB_, Y_, W_, CL_, av))) for ix in (strat_ix(CL_, r_) for _ in range(NBOOT // 2))])
    MECH.append(dict(k=k, lab=f"{lab} (no tracking)\nn={len(Dm)}", b=b0, lo=pct(bb, 5), hi=pct(bb, 95)))
    put(f"a_{k}", b0, 1, f"held-out slope per sd, {lab}"); put(f"a_{k}_lo", pct(bb, 5), 1, "90% lower")
    put(f"a_{k}_hi", pct(bb, 95), 1, "90% upper"); put(f"a_{k}_n", len(Dm), 0, f"receivers with {lab}")
    put(f"rope_{k}", rope(b0, bb), 0, f"posterior P(|{lab} slope| < 3 in), %, flat prior")
# multiplicity: every season-level slope interval reported (forest plot), how many exclude zero
put("n_tests", len(MECH), 0, "season-level slope intervals reported in Figure 2")
put("n_excl0", int(sum((m_["lo"] > 0) or (m_["hi"] < 0) for m_ in MECH)), 0, "of them excluding zero")
# coach proxy: per rep, Break G against the speed shed into the plant (entry speed minus slowest speed)
put("proxy_rho", spearmanr(cb.bg, cb.v_in - cb.v_apex)[0], 2, "rep-level rank corr, Break G vs speed shed into the cut")
put("proxy_n", len(cb), 0, "drill reps in it")
# Break G in on-film units: mean |acceleration| over the graded second x 1 s = yd/s of velocity shed and redirected
put("bg_yds_sd", float(INFO["bg"][1] * cb.groupby(["drill_name", "draft_year"]).bg.std().mean()) * G_YD, 2, "1 sd of true receiver Break G in yd/s of velocity shed and redirected")
put("shed_d", float((cb.v_in - cb.v_apex).median()), 1, "median drill rep: speed shed into the plant, yd/s")
# what a separation inch is worth (appendix): catch rate on separation when targeted, x median targets
s20 = ss_all[ss_all.position_group.isin(["WR", "TE"]) & (ss_all.targets >= 20) & ss_all.avg_separation_targeted.notna()]
bcr = wls(np.column_stack([np.ones(len(s20)), s20.avg_separation_targeted]), s20.receptions / s20.targets, s20.targets)[1]
TG = int(ss_all[ss_all.position_group.isin(["WR", "TE"]) & (ss_all.routes >= 200)].targets.median())
CATCH_PER_IN = bcr / IN_YD * TG
put("catch3", SESOI_IN * CATCH_PER_IN, 2, "catches a season worth 3 inches of SOE")
# what clubs pay: log pick on Break G (and TE), every drafted or undrafted WR/TE with a score
XX = X.assign(one=1.0, z=X.BG / X.BG.std())
A_ = np.column_stack([np.ones(len(XX)), XX.te, XX.z])
bp = wls(A_, XX.logpick, XX.one)[-1]
r_ = rng_for("price")
bpb = [wls(A_[ix], XX.logpick.to_numpy()[ix], np.ones(len(ix)))[-1] for ix in
       (r_.integers(0, len(XX), len(XX)) for _ in range(NBOOT))]
put("price", -100 * (np.exp(bp) - 1), 0, "% earlier draft pick per sd of Break G, same position")
put("price_lo", -100 * (np.exp(pct(bpb, 95)) - 1), 0, "90% lower"); put("price_hi", -100 * (np.exp(pct(bpb, 5)) - 1), 0, "upper")
put("price_n", len(XX), 0, "receivers in the pick-price fit")

# ============================================================================= 3. NFL snaps: the same code
N = build_nfl(players)
gb, g, TRACES, VTOP = N["gb"], N["g"], N["traces"], N["vtop"]
put("n_breaks", len(g), 0, "graded NFL breaks on the five route families (sampled snaps)")
ga_ = g[g.half == 0].groupby("nfl_id").bg_gz.mean()
gb_ = g[g.half == 1].groupby("nfl_id").bg_gz.mean()
GH = pd.concat([ga_, gb_], axis=1, keys=["a", "b"]).dropna()
GREL = boot_rho(GH.a, GH.b, "grel")
put("grel", GREL[0], 2, "NFL Break G, odd vs even graded snaps"); put("grel_n", GREL[3], 0, "receivers in it")
gcount = g.groupby("nfl_id").size()
P8 = pd.concat([X.BG.rename("d"), g.groupby("nfl_id").bg_gz.mean().rename("gm"), gcount.rename("ng")], axis=1).dropna()
P8 = P8[P8.ng >= 8]
SPX = pd.DataFrame({"f40z": -X["forty_s"], "pos": X.position_group}).join(VTOP.rename("vtop"), how="inner").dropna()
COMMON = P8.index.intersection(SPX.index)
SPX = SPX.loc[COMMON].copy()
SPX["vz"] = SPX.groupby("pos").vtop.transform(zs)
P8c = P8.loc[COMMON].assign(zd=lambda d: zs(d.d), zg=lambda d: zs(d.gm))
SPC = boot_rho(SPX.loc[COMMON, "f40z"], SPX.loc[COMMON, "vz"], "spc")
BGC = boot_rho(P8c.zd, P8c.zg, "bgc")
put("f1_n", len(COMMON), 0, "receivers in Fig 2 and the transfer rows of Table 2")
put("spc", SPC[0], 2, "official 40 vs NFL top speed, rank corr"); put("spc_lo", SPC[1], 2, "90% lower")
put("spc_hi", SPC[2], 2, "90% upper")
put("bgc", BGC[0], 2, "Combine vs NFL Break G, same receivers"); put("bgc_lo", BGC[1], 2, "90% lower")
put("bgc_hi", BGC[2], 2, "90% upper")
# noise ceiling (grafted from round-20 cand_3): the largest rank correlation the two noisy measures could show
REL_C = float(np.mean(bg_score(cpl.loc[COMMON, "bg_z"].to_numpy(float), cpl.loc[COMMON, "n_drills"].to_numpy(float),
                               ICC_BG)[1]))
REL_N = 2 * GREL[0] / (1 + GREL[0])
CEIL = float(np.sqrt(REL_C * REL_N))
put("rel_c", REL_C, 2, "reliability of the Combine score, Fig 2 receivers")
put("rel_nfl", REL_N, 2, "reliability of the NFL Break G average (Spearman-Brown of odd/even)")
put("ceil", CEIL, 2, "noise ceiling: largest observable transfer correlation")
put("exp50", 0.5 * CEIL, 2, "observed correlation expected if the true drill-game correlation were 0.5")
put("bgc_dis", BGC[0] / CEIL, 2, "transfer corrected for both reliabilities")
# matched route family and protocol subsets: receiver x family cells, receiver-clustered
gfam = g.groupby(["nfl_id", "route_ran"]).bg_gz.mean()
gfam.index.names = ["nfl_id", "family"]
vlg = g.groupby("route_ran").v_in.median()
cnt_cell = cb.groupby(["nfl_id", "drill_name"]).bg_z.transform("size")
matched = (cb.v_in - cb.family.map(vlg)).abs() <= 1.0
SUBSETS = {"fam": ("All drill reps", cb), "match": ("Reps entered within 1 yd/s\nof NFL speed", cb[matched]),
           "two": ("Drills run twice\n(mean of 2 reps)", cb[cnt_cell >= 2]), "one": ("Drills run once", cb[cnt_cell == 1])}
SUBR = {}
for k, (lab, sub) in SUBSETS.items():
    cf = sub.groupby(["nfl_id", "family"]).bg_z.mean()
    JF = pd.concat([cf.rename("d"), gfam.rename("g")], axis=1).dropna().reset_index()
    rr = boot_rho(JF.d, JF.g, k, cluster=JF.nfl_id)
    SUBR[k] = dict(lab=lab, r=rr[0], lo=rr[1], hi=rr[2], cells=rr[3], np=JF.nfl_id.nunique())
    put(k, rr[0], 2, f"drill vs NFL Break G, same route family: {lab}"); put(k + "_lo", rr[1], 2, "90% lower")
    put(k + "_hi", rr[2], 2, "90% upper"); put(k + "_np", JF.nfl_id.nunique(), 0, "receivers")
    put(k + "_cells", rr[3], 0, "receiver-family cells")
put("match_pct", 100 * matched.mean(), 0, "% of drill reps entered within 1 yd/s of NFL entry speed")
put("match_hitch", int((matched & (cb.family == "HITCH")).sum()), 0, "hitch-family drill reps that are matched")
put("n_hitch", int((cb.family == "HITCH").sum()), 0, "hitch-family drill reps")
put("vin_gap", float((cb.v_in - cb.family.map(vlg)).median()), 1, "median drill minus NFL entry speed, yd/s")
put("vin_hitch_d", float(cb[cb.family == "HITCH"].v_in.median()), 1, "hitch entry speed, drill, yd/s")
put("vin_hitch_n", float(vlg["HITCH"]), 1, "hitch entry speed, NFL, yd/s")
put("vin_slant_d", float(cb[cb.drill_name == "SLANT_ROUTE_LEFT"].v_in.median()), 1, "slant drill entry speed, yd/s")
put("vin_slant_n", float(vlg["SLANT"]), 1, "NFL slant entry speed, yd/s")
put("shed_n", float((g.v_in - g.v_apex).median()), 1, "median NFL break: speed shed into the plant, yd/s")
# entry-speed test 1: keep only NFL breaks entered at drill-like speed (within 1 yd/s of the drill median for that
# route family, or faster), then the same receiver x family transfer as above
vdr = cb.groupby("family").v_in.median()
gfast = g[g.v_in >= g.route_ran.map(vdr) - 1.0]
put("fast_pct", 100 * len(gfast) / len(g), 0, "% of NFL breaks entered at drill-like speed")
# entry-speed test 2: remove entry speed from Break G on both sides (linear, within drill x year / route x position)


def _adj(df, grp):
    out = pd.Series(np.nan, index=df.index)
    for _, d_ in df.groupby(grp):
        b_ = np.polyfit(d_.v_in, d_.bg, 1)
        r_ = d_.bg - np.polyval(b_, d_.v_in)
        out.loc[d_.index] = (r_ - r_.mean()) / r_.std(ddof=1)
    return out


cb["bg_adjz"] = _adj(cb, ["drill_name", "draft_year"])
g["bg_adjz"] = _adj(g, ["route_ran", "position_group"])
put("vin_rho_d", spearmanr(cb.bg, cb.v_in)[0], 2, "drill reps: rank corr of Break G with entry speed")
put("vin_rho_n", spearmanr(g.bg, g.v_in)[0], 2, "NFL breaks: rank corr of Break G with entry speed")
for k, (cbs, gs_, col) in {"fastn": (cb, gfast, "bg"), "adj": (cb, g, "bg_adjz")}.items():
    dcol, gcol = ("bg_z", "bg_gz") if col == "bg" else ("bg_adjz", "bg_adjz")
    cf = cbs.groupby(["nfl_id", "family"])[dcol].mean()
    gf_ = gs_.groupby(["nfl_id", "route_ran"])[gcol].mean()
    gf_.index.names = ["nfl_id", "family"]
    JF = pd.concat([cf.rename("d"), gf_.rename("g")], axis=1).dropna().reset_index()
    rr = boot_rho(JF.d, JF.g, k, cluster=JF.nfl_id)
    SUBR[k] = dict(r=rr[0], lo=rr[1], hi=rr[2], cells=rr[3], np=JF.nfl_id.nunique())
    put(k, rr[0], 2, f"drill vs NFL Break G, same route family, {k}"); put(k + "_lo", rr[1], 2, "90% lower")
    put(k + "_hi", rr[2], 2, "90% upper"); put(k + "_np", JF.nfl_id.nunique(), 0, "receivers")
# sampling sensitivity: half the sampled plays are the receiver's top-EPA plays; transfer on each half (5+ breaks)
for k, kind in (("rnd", "random"), ("epa", "top_epa")):
    gm_ = g[g.sample_kind == kind].groupby("nfl_id").bg_gz.agg(["mean", "size"])
    J = pd.concat([X.BG.rename("d"), gm_.loc[gm_["size"] >= 5, "mean"].rename("g")], axis=1, join="inner").dropna()
    rr = boot_rho(J.d, J.g, "samp_" + k)
    put(f"s_{k}", rr[0], 2, f"Combine vs NFL Break G, {kind} plays only"); put(f"s_{k}_lo", rr[1], 2, "90% lower")
    put(f"s_{k}_hi", rr[2], 2, "90% upper"); put(f"s_{k}_n", rr[3], 0, "receivers (5+ breaks of that kind)")
# speed on the random half only: top speed from random plays (8+ snaps)
gpr = N["gp"]
gfr = pd.read_parquet(P / "game_frames_sample.parquet", columns=["game_id", "play_id", "nfl_id", "s"]).merge(
    gpr.loc[gpr.sample_kind == "random", ["game_id", "play_id", "nfl_id"]], on=["game_id", "play_id", "nfl_id"])
pmr = gfr.groupby(["nfl_id", "game_id", "play_id"]).s.max().groupby("nfl_id")
vtr = pmr.quantile(0.9)[pmr.size() >= 8]
SR = pd.DataFrame({"f40z": -X["forty_s"], "pos": X.position_group}).join(vtr.rename("v"), how="inner").dropna()
SR["vz"] = SR.groupby("pos").v.transform(zs)
rr = boot_rho(SR.f40z, SR.vz, "samp_spd")
put("s_spd", rr[0], 2, "official 40 vs NFL top speed, random plays only"); put("s_spd_lo", rr[1], 2, "90% lower")
put("s_spd_hi", rr[2], 2, "90% upper"); put("s_spd_n", rr[3], 0, "receivers")
# the null in names: wide receivers with 200+ career routes and 5+ graded NFL breaks
gw = g.groupby("nfl_id").agg(nb=("bg", "size"), v=("v_lg", "median"))
EX = X[X.position_group == "WR"].join(CAR, how="inner").join(gw, how="inner")
EX = EX[(EX.routes >= 200) & (EX.nb >= 5)].sort_values("BG", ascending=False)
put("ex_n", len(EX), 0, "eligible wide receivers for Fig 3")
b0_ = EX.iloc[0]
put("ex0_bg", b0_.BG, 1, f"{b0_.display_name}: drill Break G z"); put("ex0_v", b0_.v, 1, "NFL turn speed vs league")
put("ex0_soe", IN_YD * b0_.ssoe, 1, "career SOE, in")
for k, sub in (("hard", EX.head(8)), ("soft", EX.tail(8))):
    put(f"x8_{k}_soe", IN_YD * sub.ssoe.mean(), 1, f"eight {k}est drill cutters: mean career SOE, in")
    put(f"x8_{k}_v", sub.v.mean(), 1, f"eight {k}est: mean NFL turn speed vs league, yd/s")
# no-tracking benchmark for transfer: official 3-cone / shuttle vs NFL Break G (8+ graded NFL breaks)
for k in ("cone", "shut"):
    J = pd.concat([AGIL[k].rename("a"), P8.gm], axis=1, join="inner").dropna()
    rr = boot_rho(J.a, J.gm, "tr_" + k)
    put(f"tr_{k}", rr[0], 2, f"{k} vs NFL Break G, rank corr"); put(f"tr_{k}_lo", rr[1], 2, "90% lower")
    put(f"tr_{k}_hi", rr[2], 2, "90% upper"); put(f"tr_{k}_n", rr[3], 0, "receivers")
# named drill rep vs NFL rep: the hardest, middle and softest drill cutters of Fig 1, on route families run in both
CARD = []
for rk in (0, len(EX) // 2, len(EX) - 1):
    pid, r = EX.index[rk], EX.iloc[rk]
    fams = sorted(set(cb[cb.nfl_id == pid].family) & set(g[g.nfl_id == pid].route_ran))
    dr_ = cb[(cb.nfl_id == pid) & cb.family.isin(fams)]
    nf_ = g[(g.nfl_id == pid) & g.route_ran.isin(fams)]
    t = tag(r.display_name)
    CARD.append(dict(t=t, name=r.display_name, rank=rk + 1, fams=fams, nd=len(dr_), nn=len(nf_)))
    put(f"c_{t}_rank", rk + 1, 0, f"{r.display_name}: drill Break G rank of {len(EX)}")
    put(f"c_{t}_z", r.BG, 1, "drill Break G, sd")
    for side, d_ in (("d", dr_), ("n", nf_)):
        put(f"c_{t}_{side}in", d_.v_in.median(), 1, f"{side}: median entry speed, yd/s")
        put(f"c_{t}_{side}ap", d_.v_apex.median(), 1, f"{side}: median slowest speed, yd/s")
        put(f"c_{t}_{side}out", d_.v_out.median(), 1, f"{side}: median exit speed, yd/s")
        put(f"c_{t}_{side}bg", d_.bg.median(), 2, f"{side}: median Break G, g")
        put(f"c_{t}_{side}n", len(d_), 0, f"{side}: reps / breaks")
        put(f"c_{t}_{side}sh", float(NUM[f"c_{t}_{side}in"]) - float(NUM[f"c_{t}_{side}ap"]), 1,
            f"{side}: speed shed into the plant, median entry minus median slowest speed, yd/s")

# ============================================================================= 4. day 3: last season's SOE vs pick order
cls_of = players.set_index("nfl_id").draft_year
PR = build_pairs(ss, pb)
FC = loco_forecasts(PR, ss, cls_of)
D3 = half_lists(FC[FC.day3].reset_index(drop=True))
put("d3_n", len(D3), 0, "day-3 / undrafted pairs"); put("d3_np", D3.pid.nunique(), 0, "receivers in them")
put("d3_te", int(D3.te.sum()), 0, "tight-end pairs among them")
par_all = shrink_fit(ss)
put("k200", par_all[1] / (par_all[1] + par_all[2] / 200), 2, "share of a 200-route season's SOE kept by the shrinkage")
G3 = clusters(D3.pid)


def hl_stats(d):
    return 100 * d[d.top_soe].hit.mean(), 100 * d[d.top_pick].hit.mean()


H1, H0 = hl_stats(D3)
r_ = rng_for("d3_half")
HB = np.array([hl_stats(D3.iloc[np.concatenate([G3[i] for i in r_.integers(0, len(G3), len(G3))])]) for _ in range(NBOOT)])
put("hl_k1", int(D3[D3.top_soe].hit.sum()), 0, "SOE-forecast half: next-season SOE above zero")
put("hl_k0", int(D3[D3.top_pick].hit.sum()), 0, "pick-order half: hits")
put("hl_n", int(D3.top_soe.sum()), 0, "receivers taken by each half-list")
put("hl_base_k", int(D3.hit.sum()), 0, "all day-3 pairs that hit")
put("hl_d", H1 - H0, 0, "SOE minus pick order, hit-rate points")
put("hl_d_lo", pct(HB[:, 0] - HB[:, 1], 5), 0, "90% lower, receiver-clustered")
put("hl_d_hi", pct(HB[:, 0] - HB[:, 1], 95), 0, "90% upper")
b1_by_cls = {}
for c in sorted(PR.draft_year.unique()):
    tst = (PR.draft_year == c).to_numpy()
    par = shrink_fit(ss[cls_of.reindex(ss.nfl_id).to_numpy() != c])
    pp = PR.assign(y1s=shr_with(par, PR.y1, PR.r1))
    b1_by_cls[c] = fit_ols(pp[~tst], F1)[0]
strata = {}
for pid, s_ in D3.groupby("pid").s1.apply(lambda s: tuple(sorted(s))).items():
    strata.setdefault(s_, []).append(pid)
key_ = {(r.pid, r.s1): r.y1s for r in D3.itertuples()}
r_ = rng_for("d3_perm")
null_ = []
for _ in range(NPERM):
    mp = {}
    for ids in strata.values():
        mp.update(dict(zip(ids, r_.permutation(ids))))
    y1p = np.array([key_[(mp[r.pid], r.s1)] for r in D3.itertuples()])
    dp = D3.assign(y1s=y1p)
    dp["f1"] = [float(pred(dp.iloc[[i]], b1_by_cls[dp.draft_year.iloc[i]], F1)[0]) for i in range(len(dp))]
    dp = half_lists(dp)
    null_.append(hl_stats(dp)[0] - H0)
put("hl_p", (1 + np.sum(np.array(null_) >= H1 - H0)) / (1 + NPERM), 2, "one-sided receiver-level permutation p")
# film-queue hit rate: day-3 receiver-seasons above zero (100+ routes): share above zero again the next season
POS = D3[D3.y1 > 0]
put("fq_n", len(POS), 0, "day-3 receiver-seasons above zero with a next season")
put("fq_k", int(POS.hit.sum()), 0, "of them above zero again")
FQ = jeff(int(POS.hit.sum()), len(POS))
put("fq_pct", 100 * POS.hit.mean(), 0, "film-queue hit rate, %")
put("fq_lo", FQ[0], 0, "Jeffreys 90% lower"); put("fq_hi", FQ[1], 0, "upper")
# pooled receiver-level test (body): every day-3 / undrafted season pair with 50+ routes in both seasons, forecasts
# leave-one-draft-class-out, receiver-cluster bootstrap. Bar fixed before the pooled run: the SOE half must beat the
# pick half by 10+ points with the 90% lower bound above zero, otherwise no separation-ordered list is published.
BAR = 10
PR50 = build_pairs(ss, pb, min_routes=50)
D50 = half_lists(loco_forecasts(PR50, ss, cls_of).query("day3").reset_index(drop=True))
P1, P0 = hl_stats(D50)
G50 = clusters(D50.pid)
r_ = rng_for("d50")
HB50 = np.array([hl_stats(D50.iloc[np.concatenate([G50[i] for i in r_.integers(0, len(G50), len(G50))])])
                 for _ in range(NBOOT)])
put("bar", BAR, 0, "pre-set bar, points"); put("p_n", len(D50), 0, "pooled day-3 pairs (50+ routes)")
put("p_np", D50.pid.nunique(), 0, "receivers in them")
put("p_soe", P1, 0, "SOE half: % above zero next season"); put("p_pick", P0, 0, "pick half: %")
put("p_d", P1 - P0, 0, "difference, points"); put("p_d_lo", pct(HB50[:, 0] - HB50[:, 1], 5), 0, "90% lower")
put("p_d_hi", pct(HB50[:, 0] - HB50[:, 1], 95), 0, "90% upper")
PASS = (P1 - P0 >= BAR) and pct(HB50[:, 0] - HB50[:, 1], 5) > 0
# receiver-level version (judges: one observation per receiver, not 38 pairs from 28): each receiver's first eligible
# pair; a call is "above zero next season" when the leave-one-class-out forecast is above zero; hit rate = correct calls
R1 = D50.sort_values("s1").groupby("pid").head(1).reset_index(drop=True)


def call_rates(d):
    return (100 * float(((d.f0 > 0) == d.hit).mean()), 100 * float(((d.f1 > 0) == d.hit).mean()))


C0, C1 = call_rates(R1)
r_ = rng_for("d3_rcv")
CRB = np.array([call_rates(R1.iloc[r_.integers(0, len(R1), len(R1))]) for _ in range(NBOOT)])
put("r_n", len(R1), 0, "day-3 receivers, one pair each")
put("r_f0", C0, 0, "pick-only forecast: % of correct above/below-zero calls")
put("r_f1", C1, 0, "pick + last-season SOE: % correct calls")
put("r_d", C1 - C0, 0, "difference, points"); put("r_d_lo", pct(CRB[:, 1] - CRB[:, 0], 5), 0, "90% lower")
put("r_d_hi", pct(CRB[:, 1] - CRB[:, 0], 95), 0, "90% upper")
# the override rule (appendix only): day-3 wide receivers, same season, pick gap under 32
D3W, CT = override_contrasts(D3[D3.te == 0])
Y2IN = IN_YD * D3W.y2.to_numpy()
PIDW = D3W.pid.to_numpy()
put("ct_n", len(CT), 0, "same-season contrasts between day-3 wide receivers")
put("ct_np", len(np.unique(PIDW)), 0, "wide receivers in them")
o32 = override_stats(CT, Y2IN, GAP)
r_ = rng_for("override")
OVB = []
for _ in range(NBOOT):
    m = receiver_weights(PIDW, r_)
    w_ = m[CT.a.to_numpy()] * m[CT.b.to_numpy()]
    st = override_stats(CT, Y2IN, GAP, w_)
    OVB.append([st["wrong"], st["be"], st["net"]])
OVB = np.array(OVB, float)
put("gap", GAP, 0, "pick-gap threshold, fixed before scoring")
put("ov_n", o32["n"], 0, "overrides (gap < 32)"); put("ov_k", o32["k"], 0, "wrong overrides")
put("ov_wr", o32["wrong"], 0, "wrong overrides per 100")
put("ov_wr_lo", pct(OVB[:, 0], 5), 0, "90% lower"); put("ov_wr_hi", pct(OVB[:, 0], 95), 0, "upper")
put("ov_be", o32["be"], 0, "break-even wrong rate per 100")
put("ov_net", o32["net"], 1, "net next-season SOE per override, in")
put("ov_net_lo", pct(OVB[:, 2], 5), 1, "90% lower"); put("ov_net_hi", pct(OVB[:, 2], 95), 1, "upper")

# ============================================================================= 5. the 2026 film queue
s25 = ss[(ss.season == 2025) & (ss.routes >= 100)].join(pb[["draft_round", "draft_overall_pick", "display_name",
                                                           "draft_year"]], on="nfl_id")
s25 = s25[s25.draft_round.isna() | (s25.draft_round >= 4)].copy()
s24 = ss[ss.season == 2024].set_index("nfl_id")
L = s25[(s25.position_group == "WR") & (s25.games >= 14) & (s25.ssoe > 0)].copy()
L["y1s"] = shr_with(par_all, L.ssoe, L.routes)
mu_, tau2_, s2_ = par_all
L["post_sd"] = np.sqrt(tau2_ * (s2_ / L.routes) / (tau2_ + s2_ / L.routes))
L["go"] = [bool(pid in s24.index and s24.loc[pid, "routes"] >= 100 and s24.loc[pid, "ssoe"] > 0) for pid in L.nfl_id]
v150 = s2_ / 150
L["flip"] = -L.y1s * v150 / L.post_sd ** 2  # normal-normal update: posterior mean = 0 at this 150-route SOE
L = L.sort_values(["go", "y1s"], ascending=False).reset_index(drop=True)
put("l_n", len(L), 0, "listed 2026 names"); put("l_go", int(L.go.sum()), 0, "GO names")
put("l_hold", int((~L.go).sum()), 0, "HOLD names")
ROST = players.set_index("nfl_id")[["status", "latest_team"]]
gq = gb[gb.soe.notna()]
SPL, cells_n = [], []
for _, r in L.iterrows():
    t = tag(r.display_name)
    put(f"l_{t}_ts", IN_YD * r.y1s, 1, f"{r.display_name}: true 2025 SOE (shrunk), in")
    put(f"l_{t}_lo", IN_YD * (r.y1s - 1.645 * r.post_sd), 1, "90% lower")
    put(f"l_{t}_hi", IN_YD * (r.y1s + 1.645 * r.post_sd), 1, "90% upper")
    put(f"l_{t}_fl", IN_YD * r.flip, 0, "drop-off point, in over the first 150 routes of 2026")
    put(f"l_{t}_g", int(r.games), 0, "2025 games with a snap")
    put(f"l_{t}_rt", int(r.routes), 0, "2025 routes")
    put(f"l_{t}_rpg", r.routes / r.games, 0, "2025 routes per game")
    has24 = r.nfl_id in s24.index and s24.loc[r.nfl_id, "routes"] >= 50
    if has24:
        put(f"l_{t}_rpg24", s24.loc[r.nfl_id, "routes"] / s24.loc[r.nfl_id, "games"], 0, "2024 routes per game")
        put(f"l_{t}_g24", int(s24.loc[r.nfl_id, "games"]), 0, "2024 games")
    sm = gq[gq.nfl_id == r.nfl_id]
    cells = {"man": sm[sm.man], "zone": sm[~sm.man], "slot": sm[sm.slotp == 1],
             "wide": sm[(sm.slotp == 0) & (sm.inline == 0)]}
    row = dict(tag=t, name=r.display_name, go=bool(r.go), team=r.team, al25=r.main_alignment,
               al24=s24.loc[r.nfl_id, "main_alignment"] if has24 else None, has24=has24,
               status=ROST.loc[r.nfl_id, "status"], latest=ROST.loc[r.nfl_id, "latest_team"])
    for c, d in cells.items():
        put(f"sp_{t}_{c}", IN_YD * d.soe.mean() if len(d) else np.nan, 0, f"{r.display_name}: snap SOE vs league, {c}")
        put(f"sp_{t}_{c}_n", len(d), 0, f"sampled snaps, {c}")
        cells_n.append(len(d))
    SPL.append(row)
SPL = pd.DataFrame(SPL).set_index("tag")
# drill rep vs NFL rep for the 2025 names: drill Break G (sd), speed shed into the plant in drill reps and NFL breaks
QD = {}
for _, r in L.iterrows():
    t = tag(r.display_name)
    dr_, nf_ = cb[cb.nfl_id == r.nfl_id], g[g.nfl_id == r.nfl_id]
    QD[t] = dict(bg=r.nfl_id in X.index, nd=len(dr_), nn=len(nf_))
    if r.nfl_id in X.index:
        put(f"q_{t}_bg", X.loc[r.nfl_id, "BG"], 1, f"{r.display_name}: drill Break G, sd")
    if len(dr_):
        put(f"q_{t}_dsh", float(dr_.v_in.median() - dr_.v_apex.median()), 1,
            "drill: speed shed into the plant, median entry minus median slowest speed, yd/s")
    if len(nf_):
        put(f"q_{t}_nsh", float(nf_.v_in.median() - nf_.v_apex.median()), 1,
            "NFL: speed shed into the plant, median entry minus median slowest speed, yd/s")
    put(f"q_{t}_dn", len(dr_), 0, "drill reps"); put(f"q_{t}_nn", len(nf_), 0, "NFL breaks")
put("cells", len(cells_n), 0, "split cells in the 2026 list"); put("cells30", int(np.sum(np.array(cells_n) >= 30)), 0,
                                                                   "cells with 30+ snaps")
ODDS = s25[(s25.ssoe > 0) & ~s25.nfl_id.isin(L.nfl_id)]
put("tes_n", int((ODDS.position_group == "TE").sum()), 0, "2025 day-3 TEs above zero, not listed")

# ============================================================================= appendix numbers
APX = appendix.run(dict(D=D, XB=XB, Y=Y, W=W, CL=CL, BGV=BGV, HOB=HOB, HO=HO, HMB=HMB, HMSE=HMSE, ss=ss, ss_all=ss_all,
                        X=X, PR=PR, cls_of=cls_of, GAP=GAP, pb_full=pb))
for k, (v, nd, cl_) in APX.items():
    put(k, v, nd, cl_)

# ============================================================================= figures
TT = np.arange(-10, 11) * DT


# Fig 1 (hero): player cards for the three named receivers of Table 3 (hardest, middle and softest drill cutter of the
# 49 eligible wide receivers). Top: speed through the cut, drill vs NFL on the route families he ran in both settings,
# at most 5 thin reps per setting plus the median, aligned on the slowest point. Bottom: his brake, speed-through,
# drive-out and Break G, in sd against peers (drill: shrunk receiver score; NFL: mean z of his sampled breaks).
GAPS = []
for pid in EX.index:
    reps_, snaps_ = cb[cb.nfl_id == pid], g[g.nfl_id == pid]
    fams = sorted(set(reps_.family) & set(snaps_.route_ran))
    if fams:
        GAPS.append(reps_[reps_.family.isin(fams)].v_in.median() - snaps_[snaps_.route_ran.isin(fams)].v_in.median())
GAPS = np.array(GAPS)
put("ex_shared", len(GAPS), 0, "Fig 1 pool: wide receivers with a route family in both settings")
put("ex_faster", int((GAPS > 0).sum()), 0, "of them entering the drill cut faster than the NFL cut (median)")
for m_ in ("decel", "v_apex", "reaccel"):
    g[m_ + "_gz"] = g.groupby(["route_ran", "position_group"])[m_].transform(zs)
PARTS = [("Break G", "BG", "bg_gz"), ("Brake into cut", "DECEL_S", "decel_gz"),
         ("Speed through", "V_APEX_S", "v_apex_gz"), ("Drive out", "REACCEL_S", "reaccel_gz")]
fig = plt.figure(figsize=(FIGW, 10.4))
gs_ = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.15], hspace=0.30, wspace=0.14, left=0.115, right=0.985,
                       top=0.82, bottom=0.07)
for j, c in enumerate(CARD):
    pid = EX.index[c["rank"] - 1]
    reps_ = cb[(cb.nfl_id == pid) & cb.family.isin(c["fams"])]
    snaps_ = g[(g.nfl_id == pid) & g.route_ran.isin(c["fams"])]
    dr = np.stack([speed_trace(*C["reps"][e][:2], ia) for e, ia in zip(reps_.event_id, reps_.i_apex)])
    nf = np.stack([speed_trace(*TRACES[(s.game_id, s.play_id, s.nfl_id)], s.i_apex) for s in snaps_.itertuples()])
    ax = fig.add_subplot(gs_[0, j])
    ax.axvspan(-0.5, 0.5, color="#EDEDED", zorder=0, lw=0)
    r_ = rng_for("card_" + c["t"])
    for arr, col in ((nf, OI["blue"]), (dr, OI["verm"])):
        show = r_.choice(len(arr), min(5, len(arr)), replace=False)
        for tr in arr[show]:
            ax.plot(TT, tr, color=col, lw=0.9, alpha=0.35, zorder=2)
        ax.plot(TT, np.nanmedian(arr, 0), color=col, lw=3.2, zorder=3)
    t = c["t"]
    ax.set_title(f"{c['name']}\n#{c['rank']} of {len(EX)} in drill Break G ({mm(NUM[f'c_{t}_z'] if NUM[f'c_{t}_z'].startswith('-') else '+' + NUM[f'c_{t}_z'])} sd)",
                 loc="left", fontsize=12.5)
    ax.text(0.03, 0.04, f"in → slowest → out, yd/s\ndrill {NUM[f'c_{t}_din']} → {NUM[f'c_{t}_dap']} → {NUM[f'c_{t}_dout']}"
            f"\nNFL  {NUM[f'c_{t}_nin']} → {NUM[f'c_{t}_nap']} → {NUM[f'c_{t}_nout']}", transform=ax.transAxes,
            fontsize=10.5, va="bottom", color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.text(0.97, 0.97, f"{len(reps_)} drill reps\n{len(snaps_)} NFL breaks", transform=ax.transAxes, ha="right",
            va="top", fontsize=10.5, color="#333333")
    ax.set_ylim(0, 10); ax.set_xlim(-1, 1)
    ax.set_xlabel("Seconds from the slowest point", fontsize=11)
    if j == 0:
        ax.set_ylabel("Speed, yd/s")
    else:
        ax.set_yticklabels([])
    bx = fig.add_subplot(gs_[1, j])
    yy_ = np.arange(len(PARTS))[::-1].astype(float)
    dv = [float(cpl.loc[pid, col]) for _, col, _ in PARTS]
    nv = [float(g[(g.nfl_id == pid)][gc].mean()) for _, _, gc in PARTS]
    bx.barh(yy_ + 0.19, dv, height=0.36, color=OI["verm"])
    bx.barh(yy_ - 0.19, nv, height=0.36, color=OI["blue"])
    bx.axvline(0, color="#333333", lw=1)
    bx.set_xlim(-2.2, 2.2)
    bx.set_yticks(yy_)
    bx.set_yticklabels([p_[0] for p_ in PARTS] if j == 0 else [], fontsize=11)
    bx.set_xlabel("sd vs peers (drill / NFL)", fontsize=11)
fig.legend([Line2D([], [], color=OI["verm"], lw=3.2), Line2D([], [], color=OI["blue"], lw=3.2),
            Patch(color="#EDEDED")],
           ["Combine drill (thin: up to 5 reps; bold: median)",
            "NFL, same route families (thin: up to 5 breaks; bold: median)", "graded second"],
           loc="upper left", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.005, 0.915),
           columnspacing=1.2, handlelength=1.8)
fig.suptitle("The drill asks for a full-speed plant; on Sundays the same receivers enter slower,\n"
             "and the drill order does not predict the NFL order", x=0.01, y=0.985, ha="left", fontsize=14,
             fontweight="bold", va="top")
save(fig, "fig1_cuts.png")

# Fig 2: forest of held-out season slopes against the pre-set margin, tracking and no-tracking rows
fig, ax = plt.subplots(figsize=(FIGW, 6.4))
MF = pd.DataFrame(MECH)
ypos = np.arange(len(MF))[::-1].astype(float)
ax.axhspan(ypos[0] - 0.45, ypos[0] + 0.45, color="#FBE3D3", zorder=0, lw=0)
ax.axvspan(-SESOI_IN, SESOI_IN, color="#E6E6E6", zorder=0)
ax.axvline(0, color="#333333", lw=1, zorder=1)
from matplotlib.transforms import blended_transform_factory  # noqa: E402

TRB = blended_transform_factory(ax.transAxes, ax.transData)
ax.text(1.02, len(MF) - 0.45, "slope [90%]", transform=TRB, fontsize=11, fontweight="bold", va="center")
for yv, r in zip(ypos, MF.itertuples()):
    prim = r.k == "bg"
    col = OI["verm"] if r.k in ("bg", "ipw", "hm") else (OI["blue"] if r.k in ("cone", "shut") else OI["grey"])
    mk = "s" if r.k in ("cone", "shut") else "o"
    ax.plot([max(r.lo, -8), min(r.hi, 6)], [yv, yv], color=col, lw=4.5 if prim else 2.8, solid_capstyle="butt",
            zorder=3)
    ax.plot(r.b, yv, mk, color=col, ms=12 if prim else 8, zorder=4, mec="white", mew=1)
    ax.text(1.02, yv, mm(f"{r.b:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]"), transform=TRB, va="center", ha="left",
            fontsize=11, fontweight="bold" if prim else "normal")
LABS = {"bg": f"Break G (primary)\nn={len(D)} receivers", "ipw": f"Break G, weighted for who\nreached the NFL, n={len(D)}",
        "hm": f"Break G, season-by-season\nn={int(NUM['hm_n'])} player-seasons", "decel": "Brake into the cut",
        "vapex": "Speed through the cut", "reacc": "Drive out of the cut",
        "cone": f"Official 3-cone (no tracking)\nn={NUM['a_cone_n']}, a subset", "shut":
        f"Official shuttle (no tracking)\nn={NUM['a_shut_n']}, a subset"}
ax.set_yticks(ypos)
ax.set_yticklabels([LABS[k] for k in MF.k], fontsize=10.5)
ax.get_yticklabels()[0].set_fontweight("bold")
ax.set_xlim(-8, 6)
ax.set_ylim(-0.6, len(MF) - 0.2)
ax.set_xlabel(f"Inches of separation over expected per sd, held-out draft classes\n"
              f"(for scale: career SOE varies by {NUM['soe_sd']} in between receivers (1 sd); "
              f"3 in ≈ {NUM['catch3']} catches a season)", fontsize=11)
ax.legend([Patch(color="#E6E6E6"), Line2D([], [], color=OI["verm"], lw=3), Line2D([], [], color=OI["grey"], lw=3),
           Line2D([], [], color=OI["blue"], lw=3, marker="s")],
          [f"±{SESOI_IN:.0f} in: too small to matter", "Break G", "its three parts (n=" + str(len(D)) + ")",
           "official times"], loc="lower left", bbox_to_anchor=(-0.32, 1.0), frameon=False, fontsize=10.5,
          ncol=4, borderaxespad=0.2)
fig.suptitle(f"No cut measure, tracked or timed, moves NFL separation beyond the draft-slot forecast",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold")
fig.subplots_adjust(left=0.24, right=0.82, top=0.86, bottom=0.17)
save(fig, "fig2_forest.png")


def scatter_panel(ax, x, y, rho, title, xlab, ylab, col, name, lim, names=None):
    xg = np.linspace(-lim, lim, 61)
    bl = np.polyfit(x, y, 1)
    r_ = rng_for(name)
    bs = []
    for _ in range(1000):
        i = r_.integers(0, len(x), len(x))
        bs.append(np.polyval(np.polyfit(x[i], y[i], 1), xg))
    bs = np.array(bs)
    ax.fill_between(xg, np.percentile(bs, 5, axis=0), np.percentile(bs, 95, axis=0), color=col, alpha=0.2, lw=0)
    ax.plot(xg, np.polyval(bl, xg), color=col, lw=2.4)
    ax.plot(xg, xg, color="#999999", lw=1.0, ls=":")
    ax.scatter(x, y, s=32, color=col, alpha=0.75, edgecolor="white", lw=0.6, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="#dddddd", lw=0.8, zorder=0); ax.axvline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.set_title(title, loc="left", fontsize=12.5)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.text(-lim + 0.15, lim - 0.15, mm(f"ρ {rho[0]:.2f} [{rho[1]:.2f}, {rho[2]:.2f}], n={rho[3]}"), fontsize=11.5,
            color=col, fontweight="bold", va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    for (nm, xv, yv) in (names or []):
        right = xv < lim / 2
        dy = -16 if yv > 0 else (16 if right else -18)
        ax.annotate(nm, (xv, yv), xytext=(10 if right else -10, dy), textcoords="offset points",
                    fontsize=10.5, ha="left" if right else "right", va="center", color="#222222",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5),
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8), zorder=5)


# Fig 3: transfer. A speed, B Break G (same receivers), 4 most extreme receivers in B named
xa, ya = SPX.loc[COMMON, "f40z"].to_numpy(), SPX.loc[COMMON, "vz"].to_numpy()
xb, yb = P8c.zd.to_numpy(), P8c.zg.to_numpy()
LIM = float(np.ceil(2 * (max(np.abs(np.r_[xa, ya, xb, yb])) + 0.25)) / 2)
put("f3_lim", LIM, 1, "axis limit of Fig 3")
ext = np.argsort(-np.hypot(xb, yb))[:4]
NAMES = [(players.set_index("nfl_id").display_name[COMMON[i]], xb[i], yb[i]) for i in ext]
fig = plt.figure(figsize=(FIGW, 6.4))
gtop = fig.add_gridspec(1, 2, wspace=0.16, left=0.07, right=0.98, top=0.88, bottom=0.11)
ax1, ax2 = fig.add_subplot(gtop[0]), fig.add_subplot(gtop[1])
scatter_panel(ax1, xa, ya, SPC, "A. Speed carries", "Official 40, faster → (z)", "NFL top speed (z)", OI["blue"], "f1a",
              LIM)
scatter_panel(ax2, xb, yb, BGC, "B. The cut does not", "Combine Break G, harder → (z)", "NFL Break G, same code (z)",
              OI["verm"], "f1b", LIM, NAMES)
ax2.text(LIM - 0.15, -LIM + 0.15, f"noise ceiling {NUM['ceil']}; a true 0.5\nwould read {NUM['exp50']}", ha="right",
         va="bottom", fontsize=10.5, color="#333333")
fig.suptitle(f"The 40 carries to Sundays, the Combine cut does not (both panels: the same {len(COMMON)} receivers)",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold")
save(fig, "fig3_transfer.png")

# Appendix figure B (old Fig 3C plus the entry-speed tests): drill-to-NFL Break G by subset and agility times
rows3 = [("All drill reps, same route family", "fam"), ("Drills run twice (mean of 2 reps)", "two"),
         ("Drill reps entered at NFL speed", "match"), ("NFL breaks entered at drill speed", "fastn"),
         ("Entry speed removed on both sides", "adj")]
rows3 = [(lab, SUBR[k]["r"], SUBR[k]["lo"], SUBR[k]["hi"], SUBR[k]["np"], OI["verm"], "o") for lab, k in rows3]
rows3 += [(lab, float(NUM[f"tr_{k}"]), float(NUM[f"tr_{k}_lo"]), float(NUM[f"tr_{k}_hi"]), int(NUM[f"tr_{k}_n"]),
           OI["blue"], "s") for k, lab in (("cone", "Official 3-cone (no tracking)"), ("shut", "Official shuttle (no tracking)"))]
fig, ax3 = plt.subplots(figsize=(FIGW, 5.6))
yy = np.arange(len(rows3))[::-1]
TR3 = blended_transform_factory(ax3.transAxes, ax3.transData)
ax3.axvline(0, color="#333333", lw=1)
ax3.axvline(0.5 * CEIL, color="#333333", lw=1.6, ls="--")
ax3.axvline(CEIL, color="#333333", lw=1.0, ls=":")
ax3.text(0.5 * CEIL - 0.01, -0.62, f"what a true 0.5 shows: {NUM['exp50']}", fontsize=10.5, color="#333333",
         va="center", ha="right")
ax3.text(CEIL + 0.01, len(rows3) - 0.45, f"{NUM['ceil']}: a true 1.0", fontsize=10.5, color="#333333", va="center")
for yv, (lab, r0, lo, hi, n_, col, mk) in zip(yy, rows3):
    ax3.plot([lo, hi], [yv, yv], color=col, lw=3, solid_capstyle="butt")
    ax3.plot(r0, yv, mk, color=col, ms=9)
    ax3.text(1.02, yv, mm(f"{r0:+.2f} [{lo:+.2f}, {hi:+.2f}], n={n_}"), transform=TR3, fontsize=10.5, va="center")
ax3.set_yticks(yy); ax3.set_yticklabels([r[0] for r in rows3], fontsize=10.5)
ax3.set_xlim(-0.45, 0.6); ax3.set_ylim(-0.9, len(rows3) - 0.2)
ax3.set_xlabel("Drill measure vs NFL Break G, rank corr. (90% receiver bootstrap)")
ax3.set_title("No protocol subset, entry-speed match or official agility time transfers", loc="left")
fig.subplots_adjust(left=0.27, right=0.74, top=0.90, bottom=0.13)
save(fig, "figB_subsets.png")


# Appendix figure: reliability of the Break G score by graded cuts per receiver
ICC_REP = VP / (VP + VPD + VE)
r_ = rng_for("rep_icc_boot")
CBG = clusters(cb.nfl_id)
RB = []
for _ in range(NBOOT // 4):
    sel = r_.integers(0, len(CBG), len(CBG))
    d_ = pd.concat([cb.iloc[CBG[i]][["bg_z"]].assign(nfl_id=k) for k, i in enumerate(sel)])
    RB.append(icc1(d_, "bg_z"))
RB = np.array(RB) * ICC_REP / icc1(cb, "bg_z")
nn = np.linspace(1, 30, 120)
sb = lambda rho, n: n * rho / (1 + (n - 1) * rho)  # noqa: E731
fig, ax = plt.subplots(figsize=(FIGW, 5.0))
band = np.array([sb(rb, nn) for rb in RB])
ax.fill_between(nn, np.percentile(band, 5, 0), np.percentile(band, 95, 0), color=OI["blue"], alpha=0.18, lw=0)
ax.plot(nn, sb(ICC_REP, nn), color=OI["blue"], lw=2.6)
ax.axhline(0.7, color="#333333", lw=1, ls="--")
for n_, lab, dy in ((7, "today: median 7 cuts, rep-level model", -0.16), (14, "7 drills × 2 reps", -0.12)):
    yv = sb(ICC_REP, n_)
    ax.plot(n_, yv, "o", color=OI["verm"], ms=9, zorder=4)
    ax.annotate(f"{lab}: {yv:.2f}", (n_, yv), xytext=(n_ + 1.2, yv + dy), fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color="#777777"))
ax.text(29.5, 0.715, f"0.70 at {CUTS70} cuts", ha="right", va="bottom", fontsize=10.5, color="#333333")
ax.axhline(float(NUM["rel_main"]), xmin=0, xmax=0.27, color=OI["verm"], lw=2, ls=":")
ax.text(0.3, float(NUM["rel_main"]) - 0.03, f"tested receivers' average,\ndrill-level model: {NUM['rel_main']}",
        fontsize=10.5, color=OI["verm"], va="top")
ax.set_xlim(0, 30); ax.set_ylim(0, 1)
ax.set_xlabel("Graded cuts per receiver (any route drill)")
ax.set_ylabel("Reliability of the Break G score")
ax.set_title("More graded cuts make Break G stable (band: 90% receiver bootstrap)", loc="left")
fig.tight_layout()
save(fig, "figA_reliability.png")

# ============================================================================= inputs for verify.py (rebuilt there)
BI = D[["display_name", "draft_year"] + BASE + ["BG"]].copy()
BI["career_soe_in"] = IN_YD * D.ssoe
BI["routes"] = D.routes
BI.index.name = "nfl_id"
BI.round(6).to_csv(OUT / "baseline_inputs.csv")
SI = pd.DataFrame({"pos": SPX.loc[COMMON, "pos"], "nfl_bg_z_raw": P8c.gm}, index=COMMON)
SI.index.name = "nfl_id"
SI.round(6).to_csv(OUT / "speed_inputs.csv")
(OUT / "columns.json").write_text(json.dumps({
    "baseline_inputs.csv": {"nfl_id": "player id", "draft_year": "Combine class (fold label)",
                            "logpick": "log overall pick, undrafted = 262", "forty_f": "official 40, z in class x position,"
                            " missing = 0", "combine_weight_f": "weight, same", "ngs_athleticism_score_f": "NGS score, same",
                            "te": "1 = tight end", "BG": "Break G receiver score (sd of true Break G)",
                            "career_soe_in": "route-weighted career SOE, in", "routes": "career routes"},
    "speed_inputs.csv": {"nfl_id": "receivers in Fig 2 (8+ graded NFL breaks, 8+ sampled snaps, official 40)",
                         "pos": "WR or TE", "nfl_bg_z_raw": "mean NFL Break G z over his graded breaks"}}, indent=1))
(OUT / "all_numbers.json").write_text(json.dumps(NUM, indent=1))

# ============================================================================= render writeup.md and appendix.md
USED: set[str] = set()


def val(k, plus=False, track=True):
    if track:
        USED.add(k)
    v = NUM[k]
    if plus and not v.startswith("-") and float(v) > 0:
        v = "+" + v
    return mm(v)


def role_note(t):
    r = SPL.loc[t]
    bits = []
    if r.has24:
        bits.append(f"routes per game {val(f'l_{t}_rpg24')} → {val(f'l_{t}_rpg')}")
        if r.al24 != r.al25:
            spot = {"SLOT_WR": "slot", "WR": "wide", "TE": "inline"}
            bits.append(f"main spot {spot.get(r.al24, r.al24)} → {spot.get(r.al25, r.al25)}")
    if r.status != "ACT":
        bits.append("reserve list" if r.status == "RES" else f"roster status {r.status}")
    if r.latest != r.team:
        bits.append(f"now with {r.latest}")
    return "; ".join(bits) if bits else "no change"


def go_table():
    rows = ["| Film order: receiver (class, pick; 2025 team) | True SOE, in [90%] | 2025 games / routes | "
            "Role and roster change, 2024 → 2025 | Off the list below |", "|---|---|---|---|---|"]
    for k, (_, r) in enumerate(L[L.go].iterrows()):
        t = tag(r.display_name)
        pk = "UDFA" if pd.isna(r.draft_overall_pick) else f"R{int(r.draft_round)}-{int(r.draft_overall_pick)}"
        rows.append(f"| {k + 1}. **{r.display_name}** ({int(r.draft_year)}, {pk}; {r.team}) | "
                    f"{val(f'l_{t}_ts', True)} [{val(f'l_{t}_lo', True)}, {val(f'l_{t}_hi', True)}] | "
                    f"{val(f'l_{t}_g')} / {val(f'l_{t}_rt')} | {role_note(t)} | {val(f'l_{t}_fl')} in |")
    return "\n".join(rows)


def hold_strip():
    out = []
    for k, (_, r) in enumerate(L[~L.go].iterrows()):
        t = tag(r.display_name)
        note = []
        st = SPL.loc[t]
        if st.status != "ACT":
            note.append("reserve list" if st.status == "RES" else f"status {st.status}")
        if st.latest != st.team:
            note.append(f"now {st.latest}")
        if r.routes < 150:
            note.append(f"{val(f'l_{t}_rt')} routes")
        extra = f"; {', '.join(note)}" if note else ""
        out.append(f"{int(L.go.sum()) + k + 1}. {r.display_name} ({r.team}, {val(f'l_{t}_ts', True)}{extra})")
    return "; ".join(out)


def splits_table():
    rows = ["| Receiver (2025 team) | True 2025 SOE, in | Off the list below | Man, in (snaps) | Zone, in (snaps) | "
            "Slot, in (snaps) | Wide, in (snaps) |", "|---|---|---|---|---|---|---|"]
    for _, r in L.iterrows():
        t = tag(r.display_name)
        rows.append(f"| {r.display_name} ({r.team}{'' if r.go else ', HOLD'}) | {val(f'l_{t}_ts', True, False)} | "
                    f"{val(f'l_{t}_fl', False, False)} | " + " | ".join(
            f"{val(f'sp_{t}_{c}', True, False)} ({val(f'sp_{t}_{c}_n', False, False)})"
            for c in ("man", "zone", "slot", "wide")) + " |")
    return "\n".join(rows)


def cards_table():
    rows = ["| Receiver (drill Break G rank of {ex}) | Drill cut, yd/s: in → slowest → out | "
            "NFL cut, yd/s: in → slowest → out | Takeaway |".format(ex=val("ex_n")), "|---|---|---|---|"]
    nsh = {c["t"]: float(NUM[f"c_{c['t']}_nsh"]) for c in CARD}
    order = sorted(nsh, key=nsh.get)
    word = {order[0]: "least", order[1]: "middle", order[2]: "most"}
    for c in CARD:
        t = c["t"]
        tk = (f"sheds {val(f'c_{t}_dsh')} yd/s into the drill plant, {val(f'c_{t}_nsh')} on Sundays "
              f"({word[t]} of the three)")
        rows.append(f"| {c['name']} (#{val(f'c_{t}_rank')}, {val(f'c_{t}_z', True)} sd) | "
                    f"{val(f'c_{t}_din')} → {val(f'c_{t}_dap')} → {val(f'c_{t}_dout')} | "
                    f"{val(f'c_{t}_nin')} → {val(f'c_{t}_nap')} → {val(f'c_{t}_nout')} | {tk} |")
    return "\n".join(rows)


def decision_table():
    rows = ["| 2025 day-3 receiver (team) | Above zero | True 2025 SOE, in | Off the list if first 150 routes of 2026 below, in | "
            "Drill Break G, sd | Speed shed into plant, drill → NFL, yd/s | Man / zone snaps | Slot / wide snaps |",
            "|---|---|---|---|---|---|---|---|"]
    for _, r in L.iterrows():
        t = tag(r.display_name)
        bg = val(f"q_{t}_bg", True) if f"q_{t}_bg" in NUM else "no drill"
        dsh = val(f"q_{t}_dsh") if f"q_{t}_dsh" in NUM else "–"
        nsh = val(f"q_{t}_nsh") if f"q_{t}_nsh" in NUM else "–"
        rows.append(f"| {r.display_name} ({r.team}) | {'2024 and 2025' if r.go else '2025'} | "
                    f"{val(f'l_{t}_ts', True)} | {val(f'l_{t}_fl')} | {bg} | {dsh} → {nsh} | "
                    f"{val(f'sp_{t}_man_n')} / {val(f'sp_{t}_zone_n')} | {val(f'sp_{t}_slot_n')} / {val(f'sp_{t}_wide_n')} |")
    return "\n".join(rows)


def render(tpl, track=True):
    txt = tpl
    for k_, f_ in (("{{GO_TABLE}}", go_table), ("{{HOLD}}", hold_strip), ("{{SPLITS}}", splits_table),
                   ("{{CARDS}}", cards_table), ("{{DECISION}}", decision_table)):
        if k_ in txt:
            txt = txt.replace(k_, f_())
    txt = txt.replace("{{GO_NAMES}}", ", ".join(L[L.go].display_name))
    missing = sorted(set(re.findall(r"\{\+?([a-z0-9_]+)\}", txt)) - set(NUM))
    if missing:
        raise KeyError(f"template keys not computed: {missing}")
    return re.sub(r"\{(\+?)([a-z0-9_]+)\}", lambda m: val(m.group(2), m.group(1) == "+", track), txt)


WTXT = render((OUT / "writeup_template.md").read_text())
USED_W = set(USED)
(OUT / "writeup.md").write_text(WTXT)
USED.clear()
(OUT / "appendix.md").write_text(render((OUT / "appendix_template.md").read_text(), track=False))
(OUT / "numbers.json").write_text(json.dumps({k: NUM[k] for k in sorted(USED_W)}, indent=1))
(OUT / "claims.json").write_text(json.dumps({k: CLAIM[k] for k in sorted(USED_W)}, indent=1))
print(f"writeup.md rendered; {len(USED_W)} numbers quoted of {len(NUM)} computed")
