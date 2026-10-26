"""Robustness numbers quoted only in the appendix of writeup.md (called by analysis.py; no files written).

1. Equivalence: two one-sided tests of the held-out Break G slope against the pre-set +/-3 in margin (bootstrap arms
   of the same class-stratified refit bootstrap as the primary interval), and the Wald TOST of the hierarchical slope.
2. Sensitivity of the held-out slope to how the outcome is built (all from the NGS season table, so verify.py can redo
   it): seasons with 50+ or 100+ routes only, SOE without the cushion term, unweighted, wide receivers only.
3. Sensitivity of the day-3 override record (pick gap under 32) to the route minimum of a season pair and to using raw
   instead of shrunk last-season SOE.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from core import (IN_YD, SESOI_IN, build_pairs, career, loco_forecasts, loco_parts, ho_slope,
                  override_contrasts, override_stats, season_soe)


def run(ctx):
    out = {}
    hob = ctx["HOB"]
    up, dn = float(np.mean(hob >= SESOI_IN)), float(np.mean(hob <= -SESOI_IN))
    out["tost_up"] = (up, 3, "benefit arm: share of bootstrap slopes at or above +3 in")
    out["tost_dn"] = (dn, 2, "harm arm: share at or below -3 in")
    out["tost"] = (max(up, dn), 2, "TOST p of the held-out Break G slope at +/-3 in")
    hb, hse = ctx["HMB"], ctx["HMSE"]
    out["hm_tost"] = (max(norm.sf((SESOI_IN - hb) / hse), norm.sf((hb + SESOI_IN) / hse)), 2,
                      "hierarchical model TOST p at +/-3 in (Wald)")
    # 2. outcome-side sensitivity of the held-out slope
    X, ss_all = ctx["X"], ctx["ss_all"]
    variants = {}
    for k, ss_, mr in (("r50", season_soe(ss_all), 50), ("r100", season_soe(ss_all), 100),
                       ("nocush", season_soe(ss_all, cushion=False), 0)):
        D_ = X.join(career(ss_, mr), how="inner")
        variants[k] = (ho_slope(*loco_parts(D_)), len(D_))
    D = ctx["D"]
    XB, Y, W, CL, BGV = loco_parts(D)
    variants["unw"] = (ho_slope(XB, Y, np.ones(len(Y)), CL, BGV), len(D))
    wr = (D.te == 0).to_numpy()
    variants["wr"] = (ho_slope(XB[wr][:, :-1], Y[wr], W[wr], CL[wr], BGV[wr]), int(wr.sum()))
    vals = [ctx["HO"]] + [v[0] for v in variants.values()]
    out["sens_n"] = (len(vals), 0, "outcome variants in the sensitivity grid (primary included)")
    out["sens_lo"] = (min(vals), 1, "smallest held-out slope across the variants")
    out["sens_hi"] = (max(vals), 1, "largest held-out slope across the variants")
    # 3. the override record under pair and shrinkage variants
    ss, PR0, cls_of, gap = ctx["ss"], ctx["PR"], ctx["cls_of"], ctx["GAP"]
    pb = ctx["pb_full"]
    wrs = []
    for mr, raw in ((50, False), (150, False), (100, True)):
        pr = build_pairs(ss, pb, min_routes=mr)
        fc = loco_forecasts(pr, ss, cls_of)
        d3 = fc[fc.day3 & (fc.te == 0)].reset_index(drop=True)
        if raw:
            d3 = d3.assign(y1s=d3.y1)
        d, ct = override_contrasts(d3)
        wrs.append(override_stats(ct, IN_YD * d.y2.to_numpy(), gap)["wrong"])
    out["ovs_lo"] = (min(wrs), 0, "override wrong rate per 100, lowest across the three variants")
    out["ovs_hi"] = (max(wrs), 0, "highest")
    return out
