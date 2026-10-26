"""Round 20, candidate 2: do Combine route cuts (Break G) carry to NFL separation, and what should a staff use instead?

Run from the repo root (about three minutes):
    uv run python writeup/candidates/round_20/cand_2/analysis.py
Fast re-check of the headline numbers from the raw tables (under a minute):
    uv run python writeup/candidates/round_20/cand_2/verify.py

Modules (the round-19 script had grown to 5,100 lines; it is split by purpose):
- core.py      definitions and builders (Break G, SOE, NFL snaps, season pairs, held-out tests); no outputs;
- analysis.py  this file: every number quoted in the body, the five figures, the render of writeup.md;
- appendix.py  robustness quoted only in the appendix (equivalence tests, sensitivity grid);
- verify.py    independent recomputation of the headline, speed, day-3, TOST, sensitivity and break-even numbers.

Writes next to this file: numbers.json (exactly the numbers the text quotes, one key each), claims.json (what each
key backs), all_numbers.json, figures/fig1-5 (1600 px wide, 150 dpi), writeup.md (rendered from writeup_template.md)
and the inputs verify.py reads (baseline_inputs.csv, speed_inputs.csv, columns.json).

Round-20 changes (judges' round-19 feedback): one speed number (same receivers in the table, the figure and verify.py);
the day-3 test is leave-one-draft-class-out on every eligible day-3 receiver, receiver-clustered, against pick order;
the pick-gap override is reported as an error rate with an interval next to pick order; the 2026 list carries
man/zone and slot/wide splits with cell n and a "what flips it" column; Fig 3 = two rows of small multiples.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import norm  # noqa: E402

import appendix  # noqa: E402
from core import *  # noqa: E402,F403
from core import _beta  # noqa: E402,F401

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
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


# ============================================================================= 1. Combine Break G
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
# rep-level mixed model: reliability of a 7-drill score with one vs two reps per drill (protocol advice)
cbm = cb.assign(dy=cb.drill_name + "_" + cb.draft_year.astype(str))
MM = quiet_mixedlm("bg_z ~ 0 + C(dy)", cbm, groups="nfl_id", re_formula="1", vc_formula={"pxd": "0 + C(drill_name)"})
VP, VPD, VE = float(MM.cov_re.iloc[0, 0]), float(MM.vcomp[0]), float(MM.scale)
put("rel_7x1", VP / (VP + (VPD + VE) / 7), 2, "reliability of a 7-drill Break G score, one rep each")
put("rel_7x2", VP / (VP + (VPD + VE / 2) / 7), 2, "same, two reps each")

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
put("ho", HO, 1, "held-out inches of season SOE per sd of Break G"); put("ho_lo", pct(HOB, 5), 1, "90% lower")
put("ho_hi", pct(HOB, 95), 1, "90% upper")
put("sesoi", SESOI_IN, 0, "pre-set margin, inches per sd")
MECH = [dict(k="bg", lab="Break G (primary)", b=HO, lo=pct(HOB, 5), hi=pct(HOB, 95), n=len(D))]
for k, col, lab in (("decel", "DECEL_S", "Brake into the cut"), ("vapex", "V_APEX_S", "Speed through the cut"),
                    ("reacc", "REACCEL_S", "Drive out of the cut")):
    bv = D[col].to_numpy(float)
    b0 = ho_slope(XB, Y, W, CL, bv)
    r_ = rng_for("mech_" + k)
    bb = np.array([ho_slope(*(a[ix] for a in (XB, Y, W, CL, bv))) for ix in (strat_ix(CL, r_) for _ in range(NBOOT // 2))])
    MECH.append(dict(k=k, lab=lab, b=b0, lo=pct(bb, 5), hi=pct(bb, 95), n=len(D)))
    put(f"m_{k}", b0, 1, f"held-out slope, {lab}"); put(f"m_{k}_lo", pct(bb, 5), 1, "90% lower")
    put(f"m_{k}_hi", pct(bb, 95), 1, "90% upper")
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
MECH.append(dict(k="hm", lab="Break G, hierarchical", b=HMB, lo=HMB - 1.645 * HMSE, hi=HMB + 1.645 * HMSE, n=len(MS)))
# what a separation inch is worth: catch rate on separation when targeted (20+ target seasons), x median targets
s20 = ss_all[ss_all.position_group.isin(["WR", "TE"]) & (ss_all.targets >= 20) & ss_all.avg_separation_targeted.notna()]
bcr = wls(np.column_stack([np.ones(len(s20)), s20.avg_separation_targeted]), s20.receptions / s20.targets, s20.targets)[1]
TG = int(ss_all[ss_all.position_group.isin(["WR", "TE"]) & (ss_all.routes >= 200)].targets.median())
CATCH_PER_IN = bcr / IN_YD * TG
put("cr_per_yd", 100 * bcr, 1, "catch-rate points per yard of separation when targeted")
put("tg", TG, 0, "median targets of a 200-route season")
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

# ============================================================================= 3. NFL snaps: the same code
N = build_nfl(players)
gb, g, TRACES, VTOP = N["gb"], N["g"], N["traces"], N["vtop"]
put("n_breaks", len(g), 0, "graded NFL breaks on the five route families (sampled snaps)")
put("n_break_rcv", g.nfl_id.nunique(), 0, "receivers with a graded NFL break")
ga_ = g[g.half == 0].groupby("nfl_id").bg_gz.mean()
gb_ = g[g.half == 1].groupby("nfl_id").bg_gz.mean()
GH = pd.concat([ga_, gb_], axis=1, keys=["a", "b"]).dropna()
GREL = boot_rho(GH.a, GH.b, "grel")
put("grel", GREL[0], 2, "NFL Break G, odd vs even graded snaps"); put("grel_n", GREL[3], 0, "receivers in it")
# Fig 1 / Table 1 receivers: Combine score, 8+ graded NFL breaks, 8+ sampled snaps, official 40
gcount = g.groupby("nfl_id").size()
P8 = pd.concat([X.BG.rename("d"), g.groupby("nfl_id").bg_gz.mean().rename("gm"), gcount.rename("ng")], axis=1).dropna()
P8 = P8[P8.ng >= 8]
SPX = pd.DataFrame({"f40z": -X["forty_s"], "pos": X.position_group}).join(VTOP.rename("vtop"), how="inner").dropna()
COMMON = P8.index.intersection(SPX.index)
SPX = SPX.loc[COMMON].copy()
SPX["vz"] = SPX.groupby("pos").vtop.transform(zs)  # NFL top speed z within position, on the Fig 1 receivers
P8c = P8.loc[COMMON].assign(zd=lambda d: zs(d.d), zg=lambda d: zs(d.gm))
SPC = boot_rho(SPX.loc[COMMON, "f40z"], SPX.loc[COMMON, "vz"], "spc")
BGC = boot_rho(P8c.zd, P8c.zg, "bgc")
put("f1_n", len(COMMON), 0, "receivers in Fig 1 and the transfer rows of Table 1")
put("spc", SPC[0], 2, "official 40 (z in position and class) vs NFL top speed (z in position), rank corr")
put("spc_lo", SPC[1], 2, "90% lower"); put("spc_hi", SPC[2], 2, "90% upper")
put("bgc", BGC[0], 2, "Combine vs NFL Break G, same receivers"); put("bgc_lo", BGC[1], 2, "90% lower")
put("bgc_hi", BGC[2], 2, "90% upper")
# matched route family: receiver x family cells (drill family z vs NFL family z), receiver-clustered
cfam = cpd.groupby(["nfl_id", "family"]).bg_z.mean()
gfam = g.groupby(["nfl_id", "route_ran"]).bg_gz.mean()
gfam.index.names = ["nfl_id", "family"]
JF = pd.concat([cfam.rename("d"), gfam.rename("g")], axis=1).dropna().reset_index()
FAM = boot_rho(JF.d, JF.g, "fam", cluster=JF.nfl_id)
put("fam", FAM[0], 2, "drill vs NFL Break G, same route family"); put("fam_lo", FAM[1], 2, "90% lower")
put("fam_hi", FAM[2], 2, "90% upper"); put("fam_cells", FAM[3], 0, "receiver-family cells")
put("fam_np", JF.nfl_id.nunique(), 0, "receivers in them")
# entry speed: Combine stems vs the same NFL route
vlg = g.groupby("route_ran").v_in.median()
put("vin_gap", float((cb.v_in - cb.family.map(vlg)).median()), 1, "median drill minus NFL entry speed, same family, yd/s")
put("vin_hitch_d", float(cb[cb.family == "HITCH"].v_in.median()), 1, "hitch entry speed, drill, yd/s")
put("vin_hitch_n", float(vlg["HITCH"]), 1, "hitch entry speed, NFL, yd/s")
# the null in names: wide receivers with 200+ career routes and 5+ graded NFL breaks
gw = g.groupby("nfl_id").agg(nb=("bg", "size"), v=("v_lg", "median"))
EX = X[X.position_group == "WR"].join(CAR, how="inner").join(gw, how="inner")
EX = EX[(EX.routes >= 200) & (EX.nb >= 5)].sort_values("BG", ascending=False)
put("ex_n", len(EX), 0, "eligible wide receivers for the names table")
EXT = pd.concat([EX.head(3).assign(grp="hard"), EX.tail(3).assign(grp="soft")])
for i, (pid, r) in enumerate(EXT.iterrows()):
    put(f"ex{i}_bg", r.BG, 1, f"{r.display_name}: drill Break G z"); put(f"ex{i}_v", r.v, 1, "NFL turn speed vs league")
    put(f"ex{i}_nb", int(r.nb), 0, "graded NFL breaks"); put(f"ex{i}_soe", IN_YD * r.ssoe, 1, "career SOE, in")
    put(f"ex{i}_rt", int(r.routes), 0, "career routes")
for k, sub in (("hard", EX.head(8)), ("soft", EX.tail(8))):
    put(f"x8_{k}_soe", IN_YD * sub.ssoe.mean(), 1, f"eight {k}est drill cutters: mean career SOE, in")
    put(f"x8_{k}_v", sub.v.mean(), 1, f"eight {k}est: mean NFL turn speed vs league, yd/s")

# ============================================================================= 4. day 3: last season's SOE vs pick order
cls_of = players.set_index("nfl_id").draft_year
PR = build_pairs(ss, pb)
FC = loco_forecasts(PR, ss, cls_of)
D3 = half_lists(FC[FC.day3].reset_index(drop=True))
put("pairs_all", len(PR), 0, "consecutive-season pairs, 100+ routes in both")
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
# receiver-level permutation: last season's SOE shuffled among day-3 receivers with the same seasons (whole receivers),
# LOCO models held fixed, lists rebuilt
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
# the override rule: day-3 wide receivers, same season, pick gap under 32, higher last-season SOE vs earlier pick
D3W, CT = override_contrasts(D3[D3.te == 0])
Y2IN = IN_YD * D3W.y2.to_numpy()
PIDW = D3W.pid.to_numpy()
put("ct_n", len(CT), 0, "same-season contrasts between day-3 wide receivers")
put("ct_np", len(np.unique(PIDW)), 0, "wide receivers in them")
GAPS = [16, 24, 32, 48, 64, 96, 999]
OVS = {gp: override_stats(CT, Y2IN, gp) for gp in GAPS}
r_ = rng_for("override")
OVB = {gp: [] for gp in GAPS}
for _ in range(NBOOT):
    m = receiver_weights(PIDW, r_)
    w_ = m[CT.a.to_numpy()] * m[CT.b.to_numpy()]
    for gp in GAPS:
        st = override_stats(CT, Y2IN, gp, w_)
        OVB[gp].append([st["wrong"], st["be"], st["net"]])
OVB = {k: np.array(v, float) for k, v in OVB.items()}
o32 = OVS[GAP]
put("gap", GAP, 0, "pick-gap threshold, fixed before scoring")
put("ov_n", o32["n"], 0, "overrides (gap < 32)"); put("ov_k", o32["k"], 0, "wrong overrides")
put("ov_wr", o32["wrong"], 0, "wrong overrides per 100")
put("ov_wr_lo", pct(OVB[GAP][:, 0], 5), 0, "90% lower, receiver-clustered"); put("ov_wr_hi", pct(OVB[GAP][:, 0], 95), 0, "upper")
put("ov_be", o32["be"], 0, "break-even wrong rate per 100 (gain / (gain + loss))")
put("ov_net", o32["net"], 1, "net next-season SOE per override, in")
put("ov_net_lo", pct(OVB[GAP][:, 2], 5), 1, "90% lower"); put("ov_net_hi", pct(OVB[GAP][:, 2], 95), 1, "upper")
put("ov_net_c", o32["net"] * CATCH_PER_IN, 1, "net per override in catches a season")
put("ov_all_n", OVS[999]["n"], 0, "overrides at any gap"); put("ov_all_wr", OVS[999]["wrong"], 0, "wrong per 100, any gap")

# ============================================================================= 5. the 2026 film list
s25 = ss[(ss.season == 2025) & (ss.routes >= 100)].join(pb[["draft_round", "draft_overall_pick", "display_name",
                                                           "draft_year"]], on="nfl_id")
s25 = s25[s25.draft_round.isna() | (s25.draft_round >= 4)].copy()
put("pool25", len(s25), 0, "2025 day-3/UDFA WR/TE with 100+ routes")
s24 = ss[ss.season == 2024].set_index("nfl_id")
L = s25[(s25.position_group == "WR") & (s25.games >= 14) & (s25.ssoe > 0)].copy()
L["y1s"] = shr_with(par_all, L.ssoe, L.routes)
mu_, tau2_, s2_ = par_all
L["post_sd"] = np.sqrt(tau2_ * (s2_ / L.routes) / (tau2_ + s2_ / L.routes))
# GO = also above zero over 100+ routes in 2024 (two straight qualifying seasons); HOLD = first qualifying season
L["go"] = [bool(pid in s24.index and s24.loc[pid, "routes"] >= 100 and s24.loc[pid, "ssoe"] > 0) for pid in L.nfl_id]
# drop-off: the 2026 SOE over his first 150 routes that would pull the updated true-SOE estimate below zero
v150 = s2_ / 150
L["flip"] = -L.y1s * v150 / L.post_sd ** 2  # normal-normal update: posterior mean = 0 at this 150-route SOE
L = L.sort_values(["go", "y1s"], ascending=False).reset_index(drop=True)
put("l_n", len(L), 0, "listed 2026 names"); put("l_go", int(L.go.sum()), 0, "GO names")
put("l_hold", int((~L.go).sum()), 0, "HOLD names")
gq = gb[gb.soe.notna()]
SPL, cells_n = [], []
for _, r in L.iterrows():
    t = tag(r.display_name)
    sm = gq[gq.nfl_id == r.nfl_id]
    put(f"l_{t}_soe", IN_YD * r.ssoe, 1, f"{r.display_name}: 2025 SOE, in")
    put(f"l_{t}_ts", IN_YD * r.y1s, 1, "true SOE (shrunk), in")
    put(f"l_{t}_lo", IN_YD * (r.y1s - 1.645 * r.post_sd), 1, "90% lower")
    put(f"l_{t}_hi", IN_YD * (r.y1s + 1.645 * r.post_sd), 1, "90% upper")
    put(f"l_{t}_fl", IN_YD * r.flip, 0, "drop-off point, in over the first 150 routes of 2026")
    cells = {"man": sm[sm.man], "zone": sm[~sm.man], "slot": sm[sm.slotp == 1],
             "wide": sm[(sm.slotp == 0) & (sm.inline == 0)]}
    row = dict(tag=t, name=r.display_name, go=bool(r.go), n=len(sm))
    for c, d in cells.items():
        put(f"sp_{t}_{c}", IN_YD * d.soe.mean() if len(d) else np.nan, 0, f"{r.display_name}: snap SOE vs league, {c}")
        put(f"sp_{t}_{c}_n", len(d), 0, f"sampled snaps, {c}")
        row[c], row[c + "_n"] = IN_YD * d.soe.mean() if len(d) else np.nan, len(d)
        cells_n.append(len(d))
    SPL.append(row)
SPL = pd.DataFrame(SPL)
put("cells", len(cells_n), 0, "split cells in the 2026 table"); put("cells30", int(np.sum(np.array(cells_n) >= 30)), 0,
                                                                    "cells with 30+ snaps")
ODDS = s25[(s25.ssoe > 0) & ~s25.nfl_id.isin(L.nfl_id)].sort_values("ssoe", ascending=False)
put("tes_n", int((ODDS.position_group == "TE").sum()), 0, "2025 day-3 TEs above zero, not listed")

# ============================================================================= appendix numbers
APX = appendix.run(dict(D=D, XB=XB, Y=Y, W=W, CL=CL, BGV=BGV, HOB=HOB, HO=HO, HMB=HMB, HMSE=HMSE, ss=ss, ss_all=ss_all,
                        X=X, PR=PR, cls_of=cls_of, GAP=GAP, pb_full=pb))
for k, (v, nd, cl_) in APX.items():
    put(k, v, nd, cl_)

# ============================================================================= figures
MPH = 2.0455


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
    ax.text(-2.85, -2.85, f"rank corr. {rho[0]:.2f} [{rho[1]:.2f}, {rho[2]:.2f}]".replace("-", "−"), fontsize=11.5,
            color=col, fontweight="bold", va="bottom")


fig, axs = plt.subplots(1, 2, figsize=(FIGW, 6.2))
scatter_panel(axs[0], SPX.loc[COMMON, "f40z"].to_numpy(), SPX.loc[COMMON, "vz"].to_numpy(), SPC,
              "A. Speed carries", "Official 40, faster → (z)", "NFL top speed (z)", OI["blue"], "f1a")
scatter_panel(axs[1], P8c.zd.to_numpy(), P8c.zg.to_numpy(), BGC, "B. The cut does not",
              "Combine Break G, harder → (z)", "NFL Break G, same code (z)", OI["verm"], "f1b")
fig.suptitle(f"Same {len(COMMON)} receivers: the 40 predicts NFL top speed, the drill cut does not predict the NFL cut",
             x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.86, wspace=0.28)
save(fig, "fig1_transfer.png")

# Fig 3: forest of held-out slopes against the pre-set margin
fig, ax = plt.subplots(figsize=(FIGW, 4.6))
MF = pd.DataFrame(MECH)
ypos = np.arange(len(MF))[::-1]
ax.axvspan(-SESOI_IN, SESOI_IN, color="#EDEDED", zorder=0)
ax.axvline(0, color="#777777", lw=1)
for yv, r in zip(ypos, MF.itertuples()):
    col = OI["verm"] if r.k in ("bg", "hm") else OI["grey"]
    ax.plot([r.lo, r.hi], [yv, yv], color=col, lw=3, solid_capstyle="butt")
    ax.plot(r.b, yv, "o", color=col, ms=9)
    ax.text(7.9, yv, f"{r.b:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]".replace("-", "−"), va="center", ha="right", fontsize=11)
ax.set_yticks(ypos)
ax.set_yticklabels([f"{r.lab}\n(n={r.n} {'player-seasons' if r.k == 'hm' else 'receivers'})" for r in MF.itertuples()],
                   fontsize=11)
ax.set_xlim(-8, 8)
ax.set_xlabel("Inches of season separation over expected per sd (90% interval)")
ax.text(0, len(MF) - 0.35, f"±{SESOI_IN:.0f} in = too small to matter (≈{SESOI_IN * CATCH_PER_IN:.2f} catches a season)",
        ha="center", fontsize=10.5, color="#555555")
ax.set_ylim(-0.6, len(MF) - 0.1)
ax.set_title("No part of the Combine cut moves NFL separation beyond the no-tracking forecast", loc="left", fontsize=13.5)
fig.tight_layout()
save(fig, "fig3_null.png")

# Fig 2: two rows of small multiples, the three hardest (top) and three softest (bottom) drill cutters of Table 2:
# speed through the cut, Combine drill median vs his NFL median on the same route families, league NFL median ghost
TT = np.arange(-10, 11) * DT
cbi = cb.set_index("event_id")
LG = {f: np.nanmedian(np.stack([speed_trace(*TRACES[(r.game_id, r.play_id, r.nfl_id)], r.i_apex)
                                for r in g[g.route_ran == f].itertuples()]), axis=0) for f in FAMILIES}
fig, axs = plt.subplots(2, 3, figsize=(FIGW, 7.6), sharex=True, sharey=True)
for j, (pid, r) in enumerate(EXT.iterrows()):
    ax = axs[j // 3, j % 3]
    reps_ = cb[cb.nfl_id == pid]
    snaps_ = g[g.nfl_id == pid]
    fams = sorted(set(reps_.family) & set(snaps_.route_ran))
    dr = np.nanmedian(np.stack([speed_trace(*C["reps"][e][:2], ia) for e, ia in zip(reps_.event_id, reps_.i_apex)]), 0)
    nf = np.nanmedian(np.stack([speed_trace(*TRACES[(s.game_id, s.play_id, s.nfl_id)], s.i_apex)
                                for s in snaps_.itertuples()]), 0)
    lg = np.nanmedian(np.stack([LG[f] for f in snaps_.route_ran]), 0)
    ax.plot(TT, lg, color="#BBBBBB", lw=6, alpha=0.8, solid_capstyle="round")
    ax.plot(TT, dr, color=OI["verm"], lw=2.6)
    ax.plot(TT, nf, color=OI["blue"], lw=2.6)
    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    sg = lambda v, nd=1: (f"{v:+.{nd}f}" if round(v, nd) != 0 else f"{0:.{nd}f}").replace("-", "−")  # noqa: E731
    ax.set_title(f"{r.display_name}", loc="left", fontsize=12, pad=19)
    ax.text(0.0, 1.0, f"drill {sg(r.BG)} sd · NFL turn {sg(r.v)} yd/s · SOE {sg(IN_YD * r.ssoe)} in",
            transform=ax.transAxes, fontsize=10.5, va="bottom", color="#333333")
    ax.text(0.98, 0.04, f"{len(reps_)} drill reps, {len(snaps_)} NFL breaks", transform=ax.transAxes, ha="right",
            fontsize=10, color="#444444")
    if j % 3 == 0:
        ax.set_ylabel(("Three hardest\n" if j == 0 else "Three softest\n") + "speed, yd/s")
axs[0, 0].set_ylim(0, 10); axs[0, 0].set_xlim(-1, 1)
fig.supxlabel("Seconds from the slowest point of the cut", fontsize=12)
fig.legend([Line2D([], [], color=OI["verm"], lw=2.6), Line2D([], [], color=OI["blue"], lw=2.6),
            Line2D([], [], color="#BBBBBB", lw=6)],
           ["Combine drill (his median rep)", "NFL (his median break)", "NFL league median, same routes"],
           loc="upper left", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.005, 0.955))
fig.suptitle("Hard and soft drill cutters run the same NFL cut: drill stems are faster, NFL breaks look alike",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=2.2)
save(fig, "fig2_cuts.png")

# Fig 4: A = last vs next season SOE for every eligible day-3 receiver-season, list membership marked;
# B = wrong overrides per 100 by pick-gap threshold with the receiver-clustered band, against 50 (pick order)
fig, axs = plt.subplots(1, 2, figsize=(FIGW, 5.4), gridspec_kw=dict(width_ratios=[1.05, 1]))
ax = axs[0]
x_, y_ = IN_YD * D3.y1s, IN_YD * D3.y2
ax.axhline(0, color="#999999", lw=0.8); ax.axvline(0, color="#999999", lw=0.8)
for r, xv, yv in zip(D3.itertuples(), x_, y_):
    fc = OI["blue"] if r.top_soe else "white"
    ax.scatter(xv, yv, s=95 if r.top_pick else 55, marker="s" if r.top_pick else "o", facecolor=fc,
               edgecolor=OI["blue"], lw=1.6, zorder=3)
for r, xv, yv in zip(D3.itertuples(), x_, y_):
    if abs(yv) > 12 or abs(xv) > 8:
        left = xv < 4 and yv > 12
        below = 4 <= xv < 8 and yv > 18
        ax.annotate(r.display_name.split()[-1] + f" {str(r.s1 + 1)[2:]}", (xv, yv),
                    xytext=(-7 if left else 6, -13 if below else 3), textcoords="offset points", fontsize=10,
                    color="#333333", ha="right" if left else "left")
ax.set_xlabel("Last season's SOE, shrunk for routes (in)")
ax.set_ylabel("Next season's SOE (in)")
ax.set_title(f"A. {len(D3)} day-3 receiver-seasons, held out by class", loc="left")
ax.legend([Line2D([], [], marker="o", ls="", mfc=OI["blue"], mec=OI["blue"], ms=8),
           Line2D([], [], marker="s", ls="", mfc="white", mec=OI["blue"], ms=10)],
          [f"filled = SOE list (top half): {NUM['hl_k1']} of {NUM['hl_n']} hit",
           f"square = pick list (top half): {NUM['hl_k0']} of {NUM['hl_n']} hit"],
          loc="upper left", fontsize=10.5, frameon=False, handletextpad=0.3)
ax = axs[1]
gx = np.arange(len(GAPS))
wr = np.array([OVS[gp]["wrong"] for gp in GAPS])
lo = np.array([pct(OVB[gp][:, 0], 5) for gp in GAPS])
hi = np.array([pct(OVB[gp][:, 0], 95) for gp in GAPS])
ax.fill_between(gx, lo, hi, color=OI["blue"], alpha=0.18, lw=0)
ax.plot(gx, wr, "-o", color=OI["blue"], lw=2.2, ms=7)
ax.axhline(50, color=OI["black"], lw=1.2, ls="--")
ax.text(len(GAPS) - 1, 47, "50 = no better than pick order", ha="right", va="top", fontsize=10.5)
for xv, gp in zip(gx, GAPS):
    ax.text(xv, 3, f"n={OVS[gp]['n']}", ha="center", fontsize=10, color="#444444")
j32 = GAPS.index(GAP)
ax.annotate("pre-set (one round)", (j32, wr[j32]), xytext=(j32 - 0.6, wr[j32] - 22), fontsize=10.5,
            arrowprops=dict(arrowstyle="->", color="#444444"))
ax.set_xticks(gx); ax.set_xticklabels([("any" if gp == 999 else f"<{gp}") for gp in GAPS])
ax.set_ylim(0, 100); ax.set_xlabel("Override only when the pick gap is")
ax.set_ylabel("Wrong overrides per 100 (90% band)")
ax.set_title("B. Override record, day-3 wide receivers", loc="left")
fig.suptitle("Day 3: last season's SOE does not reliably beat pick order", x=0.01, ha="left", fontsize=14,
             fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig4_day3.png")

# Fig 5: the 2026 list, true SOE with 90% interval, call and drop-off point
fig, ax = plt.subplots(figsize=(FIGW, 5.6))
ypos = np.arange(len(L))[::-1]
ax.axvline(0, color="#777777", lw=1)
for yv, r in zip(ypos, L.itertuples()):
    st = GO_STYLE if r.go else HOLD_STYLE
    lo_, hi_ = IN_YD * (r.y1s - 1.645 * r.post_sd), IN_YD * (r.y1s + 1.645 * r.post_sd)
    ax.plot([lo_, hi_], [yv, yv], color=st["color"], lw=3.5, solid_capstyle="butt")
    ax.plot(IN_YD * r.y1s, yv, st["marker"], color=st["color"], ms=11)
    ax.plot(IN_YD * r.flip, yv, marker="v", color="#555555", ms=9, ls="")
    ax.text(IN_YD * r.flip, yv + 0.22, f"{IN_YD * r.flip:.0f}".replace("-", "−"), ha="center", va="bottom",
            fontsize=10, color="#555555")
ax.set_yticks(ypos)
ax.set_yticklabels([f"{r.display_name} ({r.team})" for r in L.itertuples()], fontsize=11)
ax.set_xlabel("Inches of separation over expected")
ax.set_xlim(min(-62, IN_YD * L.flip.min() - 8), 30)
ax.legend([Line2D([], [], ls="", **GO_STYLE, ms=10), Line2D([], [], ls="", **HOLD_STYLE, ms=10),
           Line2D([], [], marker="v", color="#555555", ls="", ms=9)],
          ["GO: above zero in 2024 and 2025", "HOLD: first season above zero",
           "Drop-off point (first 150 routes of 2026)"],
          loc="lower left", fontsize=10.5, frameon=False)
fig.suptitle("2026 film priority: day-3 wide receivers' true 2025 SOE (90% interval); review each at 150 routes",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold")
fig.tight_layout()
save(fig, "fig5_calls.png")

# ============================================================================= inputs for verify.py
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
                            "te": "1 = tight end", "BG": "Break G receiver score (sd units)",
                            "career_soe_in": "route-weighted career SOE, in", "routes": "career routes"},
    "speed_inputs.csv": {"nfl_id": "receivers in Fig 1 (8+ graded NFL breaks, 8+ sampled snaps, official 40)",
                         "pos": "WR or TE", "nfl_bg_z_raw": "mean NFL Break G z over his graded breaks"}}, indent=1))
(OUT / "all_numbers.json").write_text(json.dumps(NUM, indent=1))

# ============================================================================= render writeup.md from the template
TPL = (OUT / "writeup_template.md").read_text()
USED: set[str] = set()


def val(k, plus=False):
    USED.add(k)
    v = NUM[k]
    if plus and not v.startswith("-") and float(v) > 0:
        v = "+" + v
    return v.replace("-", "−")


def names_table():
    rows = ["| Wide receiver (class, pick) | Drill Break G, z | NFL turn speed vs league, yd/s (breaks) | "
            "Career SOE, in (routes) |", "|---|---|---|---|"]
    for i, (pid, r) in enumerate(EXT.iterrows()):
        if i in (0, 3):
            rows.append(f"| ***{'Three hardest' if i == 0 else 'Three softest'}*** | | | |")
        pk = "UDFA" if pd.isna(r.draft_overall_pick) else str(int(r.draft_overall_pick))
        rows.append(f"| {r.display_name} ({int(r.draft_year)}, {pk}) | {val(f'ex{i}_bg', True)} | "
                    f"{val(f'ex{i}_v', True)} ({val(f'ex{i}_nb')}) | {val(f'ex{i}_soe', True)} ({val(f'ex{i}_rt')}) |")
    return "\n".join(rows)


def calls_table():
    rows = ["| Receiver (class, pick; 2025 team) | Call | True SOE, in [90%] | Man / zone, in (snaps) | "
            "Slot / wide, in (snaps) | What flips it |", "|---|---|---|---|---|---|"]
    for _, r in L.iterrows():
        t = tag(r.display_name)
        pk = "UDFA" if pd.isna(r.draft_overall_pick) else f"R{int(r.draft_round)}-{int(r.draft_overall_pick)}"
        nm = f"**{r.display_name}**" if r.go else r.display_name
        flip = f"Off below {val(f'l_{t}_fl')} in" + ("" if r.go else "; GO if 2026 > 0")
        rows.append(f"| {nm} ({int(r.draft_year)}, {pk}; {r.team}) | {'**GO**' if r.go else 'HOLD'} | "
                    f"{val(f'l_{t}_ts', True)} [{val(f'l_{t}_lo', True)}, {val(f'l_{t}_hi', True)}] | "
                    f"{val(f'sp_{t}_man', True)} ({val(f'sp_{t}_man_n')}) / {val(f'sp_{t}_zone', True)} "
                    f"({val(f'sp_{t}_zone_n')}) | {val(f'sp_{t}_slot', True)} ({val(f'sp_{t}_slot_n')}) / "
                    f"{val(f'sp_{t}_wide', True)} ({val(f'sp_{t}_wide_n')}) | {flip} |")
    return "\n".join(rows)


def go_names():
    return ", ".join(L[L.go].display_name)


def hold_names():
    h = list(L[~L.go].display_name)
    return ", ".join(h[:-1]) + " and " + h[-1]


txt = TPL.replace("{{NAMES}}", names_table()).replace("{{CALLS}}", calls_table())
txt = txt.replace("{{GO}}", go_names()).replace("{{HOLD}}", hold_names())
missing = sorted(set(re.findall(r"\{\+?([a-z0-9_]+)\}", txt)) - set(NUM))
if missing:
    raise KeyError(f"template keys not computed: {missing}")
txt = re.sub(r"\{(\+?)([a-z0-9_]+)\}", lambda m: val(m.group(2), m.group(1) == "+"), txt)
(OUT / "writeup.md").write_text(txt)
(OUT / "numbers.json").write_text(json.dumps({k: NUM[k] for k in sorted(USED)}, indent=1))
(OUT / "claims.json").write_text(json.dumps({k: CLAIM[k] for k in sorted(USED)}, indent=1))
print(f"writeup.md rendered; {len(USED)} numbers quoted of {len(NUM)} computed")
