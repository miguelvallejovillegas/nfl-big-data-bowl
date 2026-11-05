# Appendix: robustness, measurement error and the 2025 day-3 names

Rendered by `analysis.py`; every number is in `all_numbers.json` and the starred ones are recomputed by `verify.py` (own bootstrap seeds; the held-out fit is also redone in statsmodels).

## Equivalence, measurement error and sensitivity

| Check | Result |
|---|---|
| TOST of the held-out slope at ±{sesoi} in* | p = {tost} (benefit arm {tost_up}, harm arm {tost_dn}) |
| TOST of the hierarchical slope at ±{sesoi} in (Wald) | p = {hm_tost} |
| Hierarchical slope, receiver and class effects ({hm_n} player-seasons, {hm_np} receivers)* | {hm} [{hm_lo}, {hm_hi}] |
| Slope with the between-drill ICC ({icc_bg} [{icc_bg_lo}, {icc_bg_hi}]) redrawn in every resample* | {ho} [{eiv_lo}, {eiv_hi}] |
| Slope on the raw drill average, per observed sd (no error correction)* | {ho_raw} [{ho_raw_lo}, {ho_raw_hi}] |
| Held-out slope across {sens_n} outcome builds (route cut-offs 50/100, no cushion term, unweighted, wide receivers only)* | {sens_lo} to {sens_hi} in per sd |
| Mechanics, held out: brake into / speed through / drive out of the cut | {m_decel}, {m_vapex}, {m_reacc} |
| Break G split-half (five drills vs the other five)* | {rel_bg} [{rel_lo}, {rel_hi}], {rel_n} receivers |
| Same drill run twice: median ICC* | {icc_med} ({icc_pairs} pairs) |
| NFL Break G, odd vs even snaps* | {grel}, {grel_n} receivers |
| Transfer corrected for both reliabilities* | {bgc_dis} |
| What 3 inches of SOE is worth | about {catch3} catches a season |

![Reliability of the Break G score by graded cuts per receiver.](figures/figA_reliability.png)

## Day 3: the earlier 100-route back-test and the film queue

With 100+ routes in both seasons ({d3_n} pairs, {d3_np} receivers), the SOE half hit {hl_k1} of {hl_n} and the pick half {hl_k0} of {hl_n}* (difference {hl_d} points [{hl_d_lo}, {hl_d_hi}]*, permutation p = {hl_p}). Of {fq_n} day-3 receiver-seasons above zero, {fq_k} stayed above zero*: {fq_pct}% [{fq_lo}, {fq_hi}], a coin flip.

The pick-gap override rule (pick gap under {gap}) overrode pick order {ov_n} times and was wrong {ov_wr} per 100 [{ov_wr_lo}, {ov_wr_hi}]* against a break-even of {ov_be}*.

## 2025 day-3 wide receivers above zero (film aid only; the body publishes no ordered list)

"Off the list below" is the 2026 SOE over his first 150 routes that pulls his true SOE under zero. Splits are sampled-snap SOE against the league in the same coverage or alignment; only {cells30} of {cells} cells reach 30 snaps.

{{SPLITS}}
