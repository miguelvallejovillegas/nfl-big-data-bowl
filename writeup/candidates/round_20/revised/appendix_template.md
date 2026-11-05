# Appendix: robustness, the override rule and the 2026 splits

Rendered by `analysis.py`; every number is in `all_numbers.json` and the starred ones are recomputed by `verify.py`.

## Equivalence and sensitivity

| Check | Result |
|---|---|
| TOST of the held-out slope at ±{sesoi} in* | p = {tost} (benefit arm {tost_up}, harm arm {tost_dn}) |
| TOST of the hierarchical slope at ±{sesoi} in (Wald) | p = {hm_tost} |
| Held-out slope across {sens_n} outcome builds (route cut-offs 50/100, no cushion term, unweighted, wide receivers only)* | {sens_lo} to {sens_hi} in per sd |
| Mechanics, held out: brake into the cut | {m_decel} [{m_decel_lo}, {m_decel_hi}] |
| Mechanics, held out: speed through the cut | {m_vapex} [{m_vapex_lo}, {m_vapex_hi}] |
| Mechanics, held out: drive out of the cut | {m_reacc} [{m_reacc_lo}, {m_reacc_hi}] |
| What 3 inches of SOE is worth | about {catch3} catches a season |

## The pick-gap override rule (dropped from the body: it says the same as the back-test)

Between two day-3 wide receivers less than a round apart (pick gap under {gap}, fixed before scoring), take the higher last-season SOE. Across {ct_n} same-season contrasts among {ct_np} receivers it overrode the pick {ov_n} times and was wrong {ov_k} times*: {ov_wr} per 100 [{ov_wr_lo}, {ov_wr_hi}]*. It would beat pick order only if wrong less than {ov_be} times per 100* (the break-even point where inches gained on right overrides equal inches lost on wrong ones); the interval spans that line. Net gain {ov_net} inches per override [{ov_net_lo}, {ov_net_hi}]. Across route cut-offs and raw SOE the wrong rate is {ovs_lo} to {ovs_hi} per 100*.

## 2026 list: sampled-snap SOE against the league in the same coverage or alignment

Only {cells30} of {cells} cells reach 30 snaps, so these point film, not grades.

{{SPLITS}}
