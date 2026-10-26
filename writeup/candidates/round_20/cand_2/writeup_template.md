# Combine Route Cuts Don't Buy NFL Separation, and Last Season's Separation Doesn't Beat Draft Order

**A receiver who cuts one standard deviation harder in Combine route drills is no more open in the NFL: {ho} inches of separation per sd (90% interval {ho_lo} to {+ho_hi}), across {n_main} receivers from draft classes the model never saw.**

- **Break G**, in a sentence a coach can repeat: on the first real cut of a route, how hard the receiver pushes into the ground over the one second around the break (braking, turning and driving out), in g, i.e. as a share of his body weight. One sd is {g_sd} g. The same code grades Combine route drills and NFL slants, outs, hitches, ins and corners from 10 Hz tracking.
- **Separation over expected (SOE)**: how many inches more open he is at the throw than Next Gen Stats separation predicts from his alignment, cushion and season, over full regular seasons.

Straight-line speed carries from the Combine to Sundays; the cut does not. On day 3, last season's SOE does not beat draft order either, so the 2026 list at the end is a film queue, not a verdict.

## Speed carries to Sundays; the drill cut does not

A scout sees a median of {why_reps} graded drill cuts per receiver, and {why_single}% of receiver-drill cells hold one rep. Table 1 puts every reliability and transfer check on one scale (rank correlation, both measures within position).

| What is compared | Rank corr. [90%] | n | Reading |
|---|---|---|---|
| Official 40 → NFL top speed | {spc} [{spc_lo}, {spc_hi}] | {f1_n} receivers | Speed carries |
| Combine → NFL Break G, same receivers | {bgc} [{bgc_lo}, {bgc_hi}] | {f1_n} receivers | No transfer |
| Combine → NFL Break G, same route family | {fam} [{fam_lo}, {fam_hi}] | {fam_np} receivers | No transfer |
| Break G, five drills vs the other five | {rel_bg} [{rel_lo}, {rel_hi}] | {rel_n} receivers | A weak trait |
| Break G, same drill run twice | median ICC {icc_med} | {icc_pairs} pairs | One rep is mostly noise |
| NFL Break G, odd vs even snaps | {grel} | {grel_n} receivers | Weak in games too |

NFL top speed is the 90th percentile of a receiver's per-snap maximum on sampled plays; NFL Break G averages his graded breaks (8+ required in the first two rows).

![Figure 1. The same {f1_n} receivers: official 40 against NFL top speed (A) and Combine against NFL Break G (B), with least-squares line and 90% bootstrap band.](figures/fig1_transfer.png)

Combine stems are entered a median {vin_gap} yd/s faster than the same NFL route (hitch: {vin_hitch_d} vs {vin_hitch_n} yd/s), so the drill grades a cut the NFL rarely asks for. The names show it: the hardest drill cutters turn at league speed on Sundays.

{{NAMES}}

Turn speed is the receiver's median speed at the slowest point of his NFL breaks minus the league median for the route; eligible: wide receivers with 200+ career routes and 5+ graded breaks ({ex_n}). The eight hardest and eight softest drill cutters average {+x8_hard_soe} and {x8_soft_soe} inches of career SOE, with the same NFL turn speed ({+x8_hard_v} and {+x8_soft_v} yd/s).

![Figure 2. Speed through the cut for the six receivers of Table 2: Combine drill median rep, NFL median break and the league NFL median on the same routes, aligned at the slowest point.](figures/fig2_cuts.png)

## Break G adds nothing to season separation

The outcome is route-weighted career SOE ({n_seasons} player-seasons, median {routes_med} routes). The baseline without tracking, fixed before the test, is log pick, official 40, weight, NGS athleticism score and a tight-end flag, fitted on two draft classes and scored on the third. It is weak (held-out R² {bl_r2}, error {bl_rmse} inches); adding Break G changes R² by {dr2} [{dr2_lo}, {dr2_hi}].

One hierarchical model ({hm_n} player-seasons of {hm_np} receivers, receiver and class random effects; class explains {hm_cls}% of the variance) gives {hm} [{hm_lo}, {+hm_hi}]. Both intervals exclude a benefit of {sesoi} inches per sd, the margin set before testing and worth about {catch3} catches a season ({cr_per_yd} catch-rate points per yard, {tg} targets). None of the three mechanics carries it either (Figure 3). Clubs still draft a receiver 1 sd harder in the cut {price}% earlier [{price_lo}, {price_hi}] at the same position.

![Figure 3. Season SOE per sd of Break G and of each mechanic, held out by draft class, plus the hierarchical fit, against the pre-set ±{sesoi} inch margin.](figures/fig3_null.png)

## Day 3: last season's SOE does not beat pick order

The obvious replacement for the drill is last season's SOE, shrunk for routes (a 200-route season keeps {k200} of its value). The test uses every eligible day-3 or undrafted receiver: {d3_n} pairs of consecutive seasons with 100+ routes each, from {d3_np} receivers, forecast leave-one-draft-class-out and resampled by receiver.

**Back-test.** Each season, the top half by the SOE forecast and the top half by pick order: next-season SOE above zero for {hl_k1} of {hl_n} and {hl_k0} of {hl_n} (all receivers: {hl_base_k} of {d3_n}). Difference {hl_d} points [{hl_d_lo}, {+hl_d_hi}], receiver-level permutation p = {hl_p}.

**The one rule a staff would use.** Between two day-3 wide receivers less than a round apart (pick gap under {gap}, fixed before scoring), take the higher last-season SOE. Across {ct_n} same-season contrasts it overrode the pick {ov_n} times and was wrong {ov_k} times: {ov_wr} per 100 [{ov_wr_lo}, {ov_wr_hi}]. Pick order is wrong exactly when the override is right, so an interval spanning 50 means the rule does not beat pick order; the net gain is {+ov_net} inches per override [{ov_net_lo}, {+ov_net_hi}]. It is not a rule. It orders film.

![Figure 4. Every day-3 receiver-season: last against next season's SOE, with SOE-list and pick-list membership (A); wrong overrides per 100 by pick-gap threshold with a receiver-clustered 90% band (B).](figures/fig4_day3.png)

## 2026: a film queue, not a verdict

Rule (round 19's, minus its model-probability filter, since the forecast did not beat pick order): day-3 or undrafted wide receiver, 14+ games, 100+ routes and SOE above zero in 2025. **GO** = also above zero over 100+ routes in 2024; **HOLD** = first such season. True SOE = 2025 SOE shrunk for routes. Splits are sampled snaps against the league in the same coverage or alignment; only {cells30} of {cells} cells reach 30 snaps, so they point film, not grades. "Off below" is the 2026 SOE over his first 150 routes that pulls his true SOE under zero.

{{CALLS}}

Troy Franklin is GO by rule but below the league in every coverage cell, so his film comes first. Arian Smith is one of the three hardest drill cutters; his case rests on separation, not the cut. Day-3 tight ends are left off ({tes_n} cleared zero in 2025): the rule has only {d3_te} tight-end pairs behind it.

![Figure 5. The 2026 list: true SOE with 90% interval, call and drop-off point.](figures/fig5_calls.png)

## Bottom line

1. **Take Break G out of the price.** Clubs pay {price}% in pick position per sd of it and get no separation.
2. **Keep the route drill only with more reps.** Two reps per drill lift Break G reliability from {rel_7x1} to {rel_7x2}.
3. **On day 3, the draft order stands.** Last season's SOE does not beat it ({hl_k1} vs {hl_k0} of {hl_n}); use it to order film, never to override the board.
4. **Film first: {{GO}}; then {{HOLD}}.** Re-check each at 150 routes, before the March 2027 league year.

## Limitations

- **Not blind.** All three classes were seen while building this; the ±{sesoi} inch margin, the baseline and the gap of {gap} were fixed before testing.
- **Small day-3 sample.** {d3_n} pairs, and {ov_n} overrides among {ct_np} wide receivers, cannot separate SOE from pick order in either direction.
- **Harm is not excluded** by the held-out test (lower bound {ho_lo} inches per sd).
- **Game tracking is a sample**: up to 40 plays per player, half chosen for high EPA, no defenders or ball. This limits Table 1's NFL rows, Figure 2 and the splits, not the season tests (full-season NGS).
- **Survivorship.** Of {n_combine} drill-scored receivers, {n_main} logged NFL routes with separation.

## Appendix

`core.py` builds every table; `analysis.py` writes every number, figure and this text; `appendix.py` holds the robustness checks; `verify.py` recomputes from the raw tables the season and career SOE, baseline R² and ΔR², the held-out slope and interval, the TOST, the speed correlation and interval, both day-3 tests, the override break-even ({ov_be} wrong per 100) and the sensitivity ranges. Equivalence at ±{sesoi} inches: TOST p = {tost} held out, {hm_tost} hierarchical. Held-out slope across {sens_n} outcome builds (route cut-offs, no cushion term, unweighted, wide receivers only): {sens_lo} to {sens_hi}. Override wrong rate across route cut-offs and raw SOE: {ovs_lo} to {ovs_hi} per 100.
