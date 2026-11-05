# Appendix: robustness, the day-3 film order and its splits

Rendered by `analysis.py`; every number is in `all_numbers.json` and the starred ones are recomputed by `verify.py`.

## Equivalence and sensitivity

| Check | Result |
|---|---|
| Hierarchical model, receiver and class effects | {hm} [{hm_lo}, {+hm_hi}] in per sd ({hm_n} player-seasons, {hm_np} receivers) |
| Raw drill average, no error correction* | {ho_raw} [{ho_raw_lo}, {+ho_raw_hi}] |
| TOST of the held-out slope at ±{sesoi} in* | p = {tost} (benefit arm {tost_up}, harm arm {tost_dn}) |
| TOST of the hierarchical slope at ±{sesoi} in (Wald) | p = {hm_tost} |
| Held-out slope across {sens_n} outcome builds (route cut-offs 50/100, no cushion term, unweighted, wide receivers only)* | {sens_lo} to {sens_hi} in per sd |
| Mechanics, held out: brake into the cut | {m_decel} [{m_decel_lo}, {m_decel_hi}] |
| Mechanics, held out: speed through the cut | {m_vapex} [{m_vapex_lo}, {m_vapex_hi}] |
| Mechanics, held out: drive out of the cut | {m_reacc} [{m_reacc_lo}, {m_reacc_hi}] |
| Cut Match: change in held-out R² | {cm_dr2} |
| Cut Match raw gap vs drill reps (why the score nets out rep counts) | ρ = {cm_nr_rho} |
| Break G, five drills vs the other five | {rel_bg} [{rel_lo}, {rel_hi}] ({rel_n} receivers) |
| What 3 inches of SOE is worth | about {catch3} catches a season |

## Day-3 film order (a coin-flip prioritization tool, not a board)

Of {fq_n} day-3 receiver-seasons above zero, {fq_k} stayed above zero the next season ({fq_pct}% [{fq_lo}, {fq_hi}]). Rule: day-3 or undrafted wide receiver, 14+ games, 100+ routes and SOE above zero in 2025; GO = also above zero in 2024. "Off the list below" is the SOE over his first 150 routes of 2026 that pulls true SOE under zero.

{{GO_TABLE}}

HOLD: {{HOLD}}

Sampled-snap SOE against the league in the same coverage or alignment (only {cells30} of {cells} cells reach 30 snaps):

{{SPLITS}}

## The pick-gap override rule

Between two day-3 wide receivers less than a round apart (pick gap under {gap}), take the higher last-season SOE. Across {ct_n} contrasts among {ct_np} receivers it overrode the pick {ov_n} times and was wrong {ov_k} times*: {ov_wr} per 100 [{ov_wr_lo}, {ov_wr_hi}]*, against a break-even of {ov_be}*. Across route cut-offs and raw SOE the wrong rate is {ovs_lo} to {ovs_hi} per 100*.
