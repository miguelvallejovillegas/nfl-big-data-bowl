"""Round 20, revision of cand_2: do Combine route cuts (Break G) carry to NFL separation, can a drill protocol fix it,
and what should a staff use instead on day 3?

Run from the repo root (about three minutes):
    uv run python writeup/candidates/round_20/revised/analysis.py
Independent recomputation from the raw tables (Break G pipelines included, about two minutes):
    uv run python writeup/candidates/round_20/revised/verify.py

Modules:
- core.py      definitions and builders (Break G, SOE, NFL snaps, season pairs, held-out tests); no outputs;
- analysis.py  this file: every number quoted, the five figures, the render of writeup.md and appendix.md;
- appendix.py  robustness quoted only in appendix.md (equivalence tests, sensitivity grid);
- verify.py    independent rebuild of the verify inputs and recomputation of the headline numbers.

Writes next to this file: numbers.json (exactly the numbers writeup.md quotes), claims.json, all_numbers.json (every
number, including appendix.md), figures/fig1-5 (1600 px wide, 150 dpi), writeup.md and appendix.md (rendered from
writeup_template.md / appendix_template.md), and baseline_inputs.csv / speed_inputs.csv (verify.py rebuilds both).

Revision changes (round-20 judges): measurement error in Break G handled explicitly (regression-calibrated score,
ICC uncertainty propagated, minimum detectable effect, noise ceiling for transfer); drill-protocol tests (entry-speed
matched reps, two-rep cells, reliability by total graded cuts); day 3 reduced to one back-test plus the film-queue hit
rate; 2026 list = three GO names with games, routes and role context plus a HOLD strip; TOST, sensitivity, override
rule and coverage splits moved to appendix.md; hero forest plot first with a drill-vs-game path panel.
"""
from __future__ import annotations

import json
import math
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
put("bl_rmse", HR[3], 1, "baseline held-out error (RMSE), inches")
HO = ho_slope(XB, Y, W, CL, BGV)
rng = rng_for("ho_slope")
HOB = np.array([ho_slope(*(a[ix] for a in (XB, Y, W, CL, BGV))) for ix in (strat_ix(CL, rng) for _ in range(NBOOT))])
put("ho", HO, 1, "held-out inches of season SOE per sd of true Break G (regression-calibrated score)")
put("ho_lo", pct(HOB, 5), 1, "90% lower"); put("ho_hi", pct(HOB, 95), 1, "90% upper")
put("sesoi", SESOI_IN, 0, "pre-set margin, inches per sd")
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
MECH.append(dict(k="eiv", lab="Break G, ICC error\npropagated, n={}".format(len(D)), b=HO, lo=pct(EIVB, 5),
                 hi=pct(EIVB, 95)))
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
SUBSETS = {"fam": ("All drill reps", cb), "match": ("Reps entered within 1 yd/s\nof NFL entry speed", cb[matched]),
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
# Fig 1 (hero): forest of held-out slopes against the pre-set margin + what is graded (drill vs game path)
fig = plt.figure(figsize=(FIGW, 6.0))
gsp = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.06, left=0.175, right=0.995, top=0.86, bottom=0.12)
ax = fig.add_subplot(gsp[0])
MF = pd.DataFrame(MECH)
ypos = np.arange(len(MF))[::-1].astype(float)
ax.axhspan(ypos[0] - 0.45, ypos[0] + 0.45, color="#FBE3D3", zorder=0, lw=0)
ax.axvspan(-SESOI_IN, SESOI_IN, color="#E6E6E6", zorder=0)
ax.axvline(0, color="#555555", lw=1, zorder=1)
for yv, r in zip(ypos, MF.itertuples()):
    prim = r.k == "bg"
    col = OI["verm"] if r.k in ("bg", "eiv", "hm") else OI["grey"]
    ax.plot([r.lo, r.hi], [yv, yv], color=col, lw=4.5 if prim else 2.8, solid_capstyle="butt", zorder=3)
    ax.plot(r.b, yv, "o", color=col, ms=12 if prim else 8, zorder=4, mec="white", mew=1)
    ax.text(9.6, yv, mm(f"{r.b:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]"), va="center", ha="right", fontsize=11,
            fontweight="bold" if prim else "normal")
ax.set_yticks(ypos)
ax.set_yticklabels(MF.lab, fontsize=10.5)
for t_ in ax.get_yticklabels()[:1]:
    t_.set_fontweight("bold")
ax.set_xlim(-8, 9.8)
ax.set_ylim(-0.6, len(MF) - 0.4)
ax.set_xlabel("Inches of season separation over expected per sd of Break G (90% interval)")
ax.legend([Patch(color="#E6E6E6")], [f"±{SESOI_IN:.0f} in: too small to matter (set before testing)"],
          loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=10.5, borderaxespad=0.2)
# path panel: one receiver's Combine curl rep and NFL hitch break, rotated so the stem runs up, break at the origin
axp = fig.add_subplot(gsp[1])


def rotated(x, y, i_ap, i_br):
    x, y = np.asarray(x, float), np.asarray(y, float)
    a0 = max(0, i_ap - 12)
    dx, dy = x[i_ap] - x[a0], y[i_ap] - y[a0]
    th = np.arctan2(dx, dy)
    c, s = np.cos(th), np.sin(th)
    X_, Y_ = x - x[i_ap], y - y[i_ap]
    xr, yr = c * X_ - s * Y_, s * X_ + c * Y_
    if np.nanmean(xr[i_ap:i_ap + 6]) < 0:
        xr = -xr
    lo, hi = max(0, i_ap - 15), min(len(x), i_ap + 10)
    w0, w1 = max(0, i_br - 5), min(len(x), i_br + 6)
    return xr, yr, lo, hi, w0, w1


cand = [pid for pid in EX.index[len(EX) // 3: 2 * len(EX) // 3]
        if ((cb.nfl_id == pid) & (cb.drill_name == "CURL_ROUTE_RIGHT")).any() and ((g.nfl_id == pid) &
                                                                                  (g.route_ran == "HITCH")).any()]
PID_P = cand[0]
rep = cb[(cb.nfl_id == PID_P) & (cb.drill_name == "CURL_ROUTE_RIGHT")].iloc[0]
snp = g[(g.nfl_id == PID_P) & (g.route_ran == "HITCH")]
snp = snp.iloc[int(np.argsort(np.abs(snp.v_in.to_numpy() - vlg["HITCH"]))[0])]
for (x_, y_, iap, ibr, col, lab, vin) in (
        (*C["reps"][rep.event_id][:2], int(rep.i_apex), int(rep.i_break), OI["verm"], "Combine curl", rep.v_in),
        (*TRACES[(snp.game_id, snp.play_id, snp.nfl_id)], int(snp.i_apex), int(snp.i_break), OI["blue"], "NFL hitch",
         snp.v_in)):
    xr, yr, lo, hi, w0, w1 = rotated(x_, y_, iap, ibr)
    axp.plot(xr[lo:hi], yr[lo:hi], color=col, lw=1.6, alpha=0.9)
    axp.plot(xr[w0:w1], yr[w0:w1], color=col, lw=6, alpha=0.55, solid_capstyle="round")
    axp.plot(xr[lo:hi], yr[lo:hi], "o", color=col, ms=2.6)
    il = lo + (iap - lo) // 4
    axp.text(xr[il] + 0.3, yr[il], f"{lab}\nentered at {vin:.1f} yd/s", color=col, fontsize=10, va="center",
             ha="left")
axp.plot(0, 0, marker="*", color="black", ms=11, zorder=5)
axp.set_aspect("equal")
axp.set_xlim(-1.8, 3.6)
axp.set_ylim(-9.2, 1.8)
axp.set_xticks([]); axp.set_yticks([])
for sp_ in axp.spines.values():
    sp_.set_visible(False)
axp.set_title(f"What is graded\n({players.set_index('nfl_id').display_name[PID_P]})", loc="left", fontsize=11.5)
fig.text(0.765, 0.035, "Thick: the graded 1-second window\nDots: every 0.1 s; ★: slowest point\nStem runs up the page",
         fontsize=10, va="bottom", ha="left", color="#333333")
fig.suptitle(f"Break G does not move NFL separation beyond the no-tracking forecast (n={len(D)} receivers)",
             x=0.01, ha="left", fontsize=14, fontweight="bold")
save(fig, "fig1_null.png")


def scatter_panel(ax, x, y, rho, title, xlab, ylab, col, name):
    xg = np.linspace(-3, 3, 61)
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
    ax.text(2.55, 2.75, "perfect\ntransfer", fontsize=10, color="#666666", ha="right", va="top")
    ax.scatter(x, y, s=34, color=col, alpha=0.75, edgecolor="white", lw=0.6, zorder=3)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.axhline(0, color="#dddddd", lw=0.8, zorder=0); ax.axvline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.set_title(title, loc="left", fontsize=12.5)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.text(-2.85, -2.85, mm(f"rank corr. {rho[0]:.2f} [{rho[1]:.2f}, {rho[2]:.2f}]"), fontsize=11.5,
            color=col, fontweight="bold", va="bottom")


fig, axs = plt.subplots(1, 2, figsize=(FIGW, 6.2))
scatter_panel(axs[0], SPX.loc[COMMON, "f40z"].to_numpy(), SPX.loc[COMMON, "vz"].to_numpy(), SPC,
              "A. Speed carries", "Official 40, faster → (z)", "NFL top speed (z)", OI["blue"], "f1a")
scatter_panel(axs[1], P8c.zd.to_numpy(), P8c.zg.to_numpy(), BGC, "B. The cut does not",
              "Combine Break G, harder → (z)", "NFL Break G, same code (z)", OI["verm"], "f1b")
fig.suptitle(f"Same {len(COMMON)} receivers (8+ graded NFL breaks and an official 40): the 40 carries, the cut does not",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold")
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.86, wspace=0.28)
save(fig, "fig2_transfer.png")

# Fig 3: six receivers at evenly spaced ranks of drill Break G (not the extremes only), speed through the cut on the
# route families he ran in both settings: his drill median and IQR, his NFL median and IQR, league NFL median
TT = np.arange(-10, 11) * DT
LGT = {f: np.stack([speed_trace(*TRACES[(r.game_id, r.play_id, r.nfl_id)], r.i_apex)
                    for r in g[g.route_ran == f].itertuples()]) for f in FAMILIES}
RANKS = np.round(np.linspace(0, len(EX) - 1, 6)).astype(int)
fig, axs = plt.subplots(2, 3, figsize=(FIGW, 8.0), sharex=True, sharey=True)
for j, rk in enumerate(RANKS):
    pid, r = EX.index[rk], EX.iloc[rk]
    ax = axs[j // 3, j % 3]
    reps_ = cb[cb.nfl_id == pid]
    snaps_ = g[g.nfl_id == pid]
    fams = sorted(set(reps_.family) & set(snaps_.route_ran))
    if fams:
        reps_, snaps_ = reps_[reps_.family.isin(fams)], snaps_[snaps_.route_ran.isin(fams)]
    dr = np.stack([speed_trace(*C["reps"][e][:2], ia) for e, ia in zip(reps_.event_id, reps_.i_apex)])
    nf = np.stack([speed_trace(*TRACES[(s.game_id, s.play_id, s.nfl_id)], s.i_apex) for s in snaps_.itertuples()])
    lg = np.nanmedian(np.concatenate([LGT[f] for f in (fams or FAMILIES)]), 0)
    for arr, col in ((dr, OI["verm"]), (nf, OI["blue"])):
        if len(arr) >= 3:
            ax.fill_between(TT, np.nanpercentile(arr, 25, 0), np.nanpercentile(arr, 75, 0), facecolor=col, alpha=0.13,
                            lw=0)
        ax.plot(TT, np.nanmedian(arr, 0), color=col, lw=2.6)
    ax.plot(TT, lg, color="black", lw=1.6, ls="--", zorder=4)
    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    sg = lambda v, nd=1: mm(f"{v:+.{nd}f}" if round(v, nd) != 0 else f"{0:.{nd}f}")  # noqa: E731
    ax.set_title(f"#{rk + 1} of {len(EX)}: {r.display_name}", loc="left", fontsize=12, pad=34)
    ax.text(0.0, 1.0, f"drill {sg(r.BG)} sd · NFL turn {sg(r.v)} yd/s · SOE {sg(IN_YD * r.ssoe)} in\n"
            f"same routes: {', '.join(f.lower() for f in fams) if fams else 'none shared (all shown)'}",
            transform=ax.transAxes, fontsize=10, va="bottom", color="#333333", linespacing=1.3)
    ax.text(0.98, 0.04, f"{len(reps_)} drill reps, {len(snaps_)} NFL breaks", transform=ax.transAxes, ha="right",
            fontsize=10, color="#444444")
    if j % 3 == 0:
        ax.set_ylabel("Speed, yd/s")
axs[0, 0].set_ylim(0, 10); axs[0, 0].set_xlim(-1, 1)
fig.supxlabel("Seconds from the slowest point of the cut", fontsize=12)
fig.legend([Line2D([], [], color=OI["verm"], lw=2.6), Line2D([], [], color=OI["blue"], lw=2.6),
            Line2D([], [], color="black", lw=1.6, ls="--")],
           ["Combine drill: his median rep (band: middle 50%)", "NFL: his median break (band: middle 50%)",
            "NFL league median, same routes"], loc="upper left", ncol=3, frameon=False, fontsize=10.5,
           bbox_to_anchor=(0.005, 0.955))
fig.suptitle(f"Six of {len(EX)} receivers at evenly spaced drill Break G ranks: drill stems are faster, NFL breaks alike",
             x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=1.2)
save(fig, "fig3_cuts.png")

# Fig 4: protocol. A = reliability by graded cuts per receiver (drills are interchangeable: receiver-by-drill
# variance ~0), smooth curve with a receiver-bootstrap band of the rep-level ICC; B = transfer by protocol subset
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
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axs[0]
band = np.array([sb(rb, nn) for rb in RB])
ax.fill_between(nn, np.percentile(band, 5, 0), np.percentile(band, 95, 0), color=OI["blue"], alpha=0.18, lw=0)
ax.plot(nn, sb(ICC_REP, nn), color=OI["blue"], lw=2.6)
ax.axhline(0.7, color="#777777", lw=1, ls="--")
for n_, lab, dy in ((7, "today: median 7\n(one rep per drill)", -0.17), (14, "7 drills × 2 reps", -0.14),
                    (CUTS70, None, 0)):
    yv = sb(ICC_REP, n_)
    if lab:
        ax.plot(n_, yv, "o", color=OI["verm"], ms=9, zorder=4)
        ax.annotate(f"{lab}: {yv:.2f}", (n_, yv), xytext=(n_ + 1.2, yv + dy), fontsize=10.5,
                    arrowprops=dict(arrowstyle="-", color="#777777"))
ax.text(29.5, 0.715, f"0.70 at {CUTS70} cuts", ha="right", va="bottom", fontsize=10.5, color="#555555")
ax.set_xlim(0, 30); ax.set_ylim(0, 1)
ax.set_xlabel("Graded cuts per receiver (any route drill)")
ax.set_ylabel("Reliability of the Break G score")
ax.set_title("A. More reps make the number stable", loc="left")
ax = axs[1]
order = ["fam", "one", "two", "match"]
yy = np.arange(len(order))[::-1]
ax.axvline(0, color="#555555", lw=1)
ax.axvline(0.5 * CEIL, color=OI["green"], lw=2, ls="--")
ax.text(0.5 * CEIL + 0.01, len(order) - 0.45, f"expected if drill and NFL\ncuts truly correlated 0.5",
        color=OI["green"], fontsize=10, va="top")
for yv, k in zip(yy, order):
    s_ = SUBR[k]
    ax.plot([s_["lo"], s_["hi"]], [yv, yv], color=OI["verm"], lw=3, solid_capstyle="butt")
    ax.plot(s_["r"], yv, "o", color=OI["verm"], ms=9)
    ax.text(-0.53, yv - 0.3, f"{s_['np']} receivers", fontsize=10, color="#555555", va="center")
ax.set_yticks(yy); ax.set_yticklabels([SUBR[k]["lab"] for k in order], fontsize=10.5)
ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.7, len(order) - 0.3)
ax.set_xlabel("Drill vs NFL Break G, same route family (rank corr., 90%)")
ax.set_title("B. No protocol subset transfers", loc="left")
fig.suptitle("Drill protocol: reps buy reliability, not transfer", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.5)
save(fig, "fig4_protocol.png")

# Fig 5: the 2026 film queue. Left = true 2025 SOE with 90% interval; right = drop-off point on its own axis
fig, (ax, axr) = plt.subplots(1, 2, figsize=(FIGW, 5.4), sharey=True, gridspec_kw=dict(width_ratios=[2.2, 1]))
ypos = np.arange(len(L))[::-1]
ax.axvline(0, color="#777777", lw=1)
for yv, r in zip(ypos, L.itertuples()):
    st = GO_STYLE if r.go else HOLD_STYLE
    lo_, hi_ = IN_YD * (r.y1s - 1.645 * r.post_sd), IN_YD * (r.y1s + 1.645 * r.post_sd)
    ax.plot([lo_, hi_], [yv, yv], color=st["color"], lw=3.5, solid_capstyle="butt")
    ax.plot(IN_YD * r.y1s, yv, st["marker"], color=st["color"], ms=11)
    axr.plot(IN_YD * r.flip, yv, marker="v", color="#555555", ms=10, ls="")
    axr.text(IN_YD * r.flip - 2.5, yv, mm(f"{IN_YD * r.flip:.0f}"), ha="right", va="center", fontsize=10.5,
             color="#333333")
ax.set_yticks(ypos)
ax.set_yticklabels([f"{k + 1}. {r.display_name} ({r.team})" for k, r in enumerate(L.itertuples())], fontsize=11)
ax.set_xlabel("True 2025 SOE, inches (90% interval)")
ax.set_xlim(-12, 28)
ax.set_title("Film order and true separation", loc="left")
axr.axvline(0, color="#777777", lw=1)
axr.set_xlim(IN_YD * L.flip.min() - 12, 4)
axr.set_xlabel("2026 SOE over first 150 routes\nthat drops him below zero (in)")
axr.set_title("Drop-off point", loc="left")
fig.legend([Line2D([], [], ls="", **GO_STYLE, ms=10), Line2D([], [], ls="", **HOLD_STYLE, ms=10)],
           ["GO: above zero in 2024 and 2025", "HOLD: first season above zero"], loc="upper left", ncol=2,
           frameon=False, fontsize=10.5, bbox_to_anchor=(0.005, 0.935))
fig.suptitle("2026 film queue: day-3 wide receivers in watch order (orders film time, not the board)",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
save(fig, "fig5_queue.png")

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
    rows = ["| Receiver | Man, in (snaps) | Zone, in (snaps) | Slot, in (snaps) | Wide, in (snaps) |", "|---|---|---|---|---|"]
    for _, r in L.iterrows():
        t = tag(r.display_name)
        rows.append(f"| {r.display_name} | " + " | ".join(
            f"{val(f'sp_{t}_{c}', True, False)} ({val(f'sp_{t}_{c}_n', False, False)})"
            for c in ("man", "zone", "slot", "wide")) + " |")
    return "\n".join(rows)


def render(tpl, track=True):
    txt = tpl.replace("{{GO_TABLE}}", go_table()).replace("{{HOLD}}", hold_strip()).replace("{{SPLITS}}",
                                                                                         splits_table())
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
