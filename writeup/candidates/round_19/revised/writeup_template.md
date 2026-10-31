# Combine Route Cuts Don't Predict NFL Separation; Last Season's Separation Is Only a Day-3 Lean

*Break G* is how hard a receiver brakes into and drives out of his sharpest cut, as a share of body weight, measured at 10 Hz. *Separation over expected (SOE)* is how open he is when the ball arrives, beyond what alignment and cushion predict (Next Gen Stats, full season), in inches.

**A receiver who cuts one sd harder at the Combine is not more open in the NFL: {prim_b} inches of season SOE (90% interval {prim_lo} to {prim_hi}), {n_main} receivers from draft classes the model never saw.** Straight-line speed carries to Sundays; the cut does not (Figure 1).

![Figure 1. The same {f1_n} receivers: official 40 against NFL top speed (A) and Combine against NFL Break G (B); five receivers spanning the Combine range, speed through the cut (C).](figures/fig1_cards.png)

## Why receivers, why now

The last receiver spot goes to a day-3 or undrafted player on about the minimum (2025 median ${why_apy}M), so a wrong call costs production, not cap. A scout sees a median of {why_reps} graded drill cuts per receiver, and {why_single}% of drill cells hold one rep. One sd of Break G is {g_sd} g; the same code grades Combine drills and NFL slants, outs, hitches, ins and corners. To price the cut, it must be stable and carry to Sundays.

| Measure | Reliability or transfer [90%] | Reading |
|---|---|---|
| Official 40 → NFL top speed, same {f1_n} receivers | ρ = {spc_rho} [{spc_lo}, {spc_hi}] | Speed carries |
| Break G, five drills vs the other five ({rel_split_n}) | ρ = {rel_split} | A weak trait |
| Break G, same drill run twice ({iccd_pairs} pairs) | median ICC {iccd_med} | One rep is mostly noise |
| Combine → NFL Break G, same {f1_n} receivers | ρ = {bgc_rho} [{bgc_lo}, {bgc_hi}] | No transfer |

Pooling raw WR and TE times gives a speed correlation of {spr_rho}, because tight ends run the 40 {te_40gap} s slower and top out {te_vgap} yd/s slower in games; that measures position as much as speed. Within position and Combine year, the scale Break G is on, it is {spc_rho} (raw, wide receivers only: {spw_rho}).

## The drill cut does not show up in NFL routes

Matching route families and correcting for few reps and few snaps does not rescue it: drill and NFL cut correlate at {fp_pool_med} [{fp_pool_lo}, {fp_pool_hi}] ({fp_pool_n} receivers), and a useful link (above 0.2) has a {fp_pool_p02}% chance. Splitting Break G into its three mechanics finds no hidden carrier (Figure 2), and speed carried through the turn on slants and outs does not transfer either (ρ = {vs_rho} [{vs_lo}, {vs_hi}], {vs_n} receivers).

![Figure 2. Season SOE per sd of Break G and each of its three mechanics on unseen classes, plus one hierarchical fit, against the pre-set ±3 inch margin.](figures/fig2_chain.png)

Combine stems are entered a median {vin_drop} yd/s faster than the same NFL route (Figure 3). That is not why the cut fails to carry: re-pricing every rep at NFL entry speed barely reorders receivers (ρ = {adj_rho} with the current score), still does not predict the NFL cut (ρ = {rp_rho} [{rp_lo}, {rp_hi}]), and would re-grade {rr_flagged} of the six 2026 names by 0.5 sd.

![Figure 3. The six 2026 receivers: drill rep and median-speed NFL snap on the same route family, aligned at the cut, with entry speeds and the rerun flag.](figures/fig3_path.png)

**The null in names.** Wide receivers with 200+ career routes and 5+ graded NFL breaks ({ex_pool} eligible). Turn speed is his median NFL break minus the league median for the route.

{{EXTREMES}}

The eight hardest and eight softest drill cutters average {ex_top_soe} and {ex_bottom_soe} inches of career SOE, with the same NFL turn speed ({ex_top_v} and {ex_bottom_v} yd/s).

## Break G adds nothing to season separation

The outcome is route-weighted career SOE ({n_seasons} player-seasons, median {routes_med} routes). The baseline without tracking, fixed before the test, is log pick, official 40, weight, NGS athleticism score and a TE flag, fitted on two classes and scored on the third. It is weak (held-out R² {bl_r2}, error {bl_rmse} inches), and adding Break G changes R² by {dr2} [{dr2_lo}, {dr2_hi}].

One hierarchical model replaces the separate resampling checks: {hm_n} player-seasons of {hm_np} receivers, the same baseline, random effects for receiver and draft class. It gives {hm_b} inches per sd [{hm_lo}, {hm_hi}]; class explains none of the variance. A benefit of +3 inches per sd (about {sesoi_catch} catches a season) is excluded by both fits, a harm of 3 inches only by the hierarchical one.

## Day 3: last season's SOE is a lean, not proven

The pre-registered test adds shrunk last-season SOE ({eb_sig}% of a 200-route season is signal) to a pick-and-routes forecast, fitted on one season pair and scored on the other. On all {ev_d3_n} day-3 and undrafted pairs ({ev_d3_np} receivers) it lifts the rank correlation with next-season SOE by {ev_d3_drho} [{ev_d3_drho_lo}, {ev_d3_drho_hi}]; a receiver-level shuffle gives one-sided p = {d3c_p}. The smallest gain this sample could reliably detect is {d3_mde}.

Back-test on the same pairs: each season's top half by the SOE forecast hit (next-season SOE above zero) {bk_k1} of {bk_take} times, the top half by pick alone {bk_k0} of {bk_take}. The difference, {bk_d} points [{bk_lo}, {bk_hi}], sits far below the detectable minimum of {bk_mde}.

Table 2 prices the lean on {cw_n} same-season contrasts between day-3 wide receivers. An override is a contrast where last-season SOE and the earlier pick disagree; the primary rule, fixed before scoring, allows it when the pick gap is under 32.

| Rule | Overrides | Wrong per 100 (break-even) | Net next-season SOE per override, in [90%] | One-sided p |
|---|---|---|---|---|
| **Pick gap under 32 (pre-set, primary)** | {ov32_n} | {ov32_wr} ({ov32_be}) | {ov32_net} [{ov32_net_lo}, {ov32_net_hi}] | {ov32_p} |
| Any gap (chosen after seeing data) | {ovall_n} | {ovall_wr} ({ovall_be}) | {ovall_net} [{ovall_net_lo}, {ovall_net_hi}] | {ovall_p} |
| Gap re-chosen without the pair's receivers | {nest_n} | {nest_wr} | {nest_net} [{nest_lo}, {nest_hi}] | {nest_p} |

Every interval crosses zero. The pre-set rule gains {ov32_net} inches ({ov32_netc} catches a season) per override; the nested check re-chooses the gap for each contrast on contrasts sharing neither receiver. Inside day 3 the earlier pick orders receivers right only {tb_pick_right}% of the time, so a wrong override costs little.

![Figure 4. Day-3 wide receivers' held-out forecast errors (A) and the net value of one override, with break-even at zero, in inches and catches (B).](figures/fig4_tiebreak.png)

## The 2026 calls

Rule, written before the back-test: day-3 or undrafted wide receiver, 14+ games and SOE above zero in 2025, model chance of 2026 SOE above zero of 50%+. **GO** = also listed a year earlier, with a hit; **HOLD** = first listing. True SOE = 2025 SOE shrunk for routes; the drop-off point is the 2026 SOE over his first 150 routes that would pull it below zero. Film SOE: sampled snaps against the league mean for the same coverage and alignment.

{{SHORTLIST}}

Applied unchanged to the 2023 and 2024 lists, the rule's names hit {bt_act_k} of {bt_act_n} times, and so did the same number of earliest picks. The back-test does not separate the rule from the pick order, so these are film-priority flags, not verdicts. Only {sp_big_cells} of {sp_cells} coverage and alignment splits reach 30 snaps, so film them rather than read them. Arian Smith is one of the hardest drill cutters; his case rests on SOE, not the cut.

![Figure 5. The 2026 list with intervals, calls and drop-off points (A), and one pooled film number per name (B).](figures/fig5_calls.png)

## Bottom line

1. **Take Break G out of the price.** At the same position, clubs draft a receiver 1 sd harder in the cut {price_bg}% earlier [{price_bg_lo}, {price_bg_hi}] and get no separation for it.
2. **If the drill is kept, add reps before speed.** Cap the stem at NFL entry speed (hitch {vin_d_hitch} vs {vin_n_hitch} yd/s, out {vin_d_out} vs {vin_n_out}) and repeat slant, dagger and comeback. The cap alone barely re-orders receivers; the repeat lifts reliability from {rel_mm1} to {rel_mm3}, so a real drill-to-NFL link would read {rel_gain_pct}% stronger.
3. **Between two day-3 receivers the staff grade even, lean to the higher last-season SOE, then check film.** The gain is suggestive, not proven ({ev_d3_drho}, p = {d3c_p}).
4. **Not for tight ends** ({pp_te_n} pairs, gain {te_g}).
5. **2026 film priority: GO on Malik Washington, Demario Douglas and Troy Franklin; HOLD on Tez Johnson, Arian Smith and Isaiah Bond.** Review at 150 routes; decide before the March 2027 league year.

## Limitations

- **Not blind, small day-3 sample.** All three classes were seen while building this; the day-3 test has {ev_d3_n} pairs, the pre-set rule {ov32_n} overrides that share receivers.
- **Harm is not ruled out in the held-out test.** Its interval reaches {prim_lo} inches per sd.
- **Game tracking is a sample.** Up to 40 plays per player, half chosen for high EPA, no defenders or ball. This limits Figure 3, film SOE and the drill-to-NFL link, not the season tests (full-season NGS).
- **Survivorship.** {sel_out} of {sel_n} drill-scored receivers never logged NFL routes with separation; the back-tests score only receivers with 100+ routes in both seasons.
- **The game-speed rerun is simulated** from the entry-speed slope within drills; no drill was run at game speed.

## Appendix

`analysis.py` writes every number, figure and this text from `data/processed`; `verify.py` rebuilds season SOE from the raw NGS separation and re-checks the headline numbers. Equivalence: the TOST of the held-out Break G slope against ±3 inches gives p = {tost19_p}; the hierarchical fit gives {hm_tost}. Sensitivity grid (`sensitivity.csv`, {sens_rows} variants: shrinkage, break and route cut-offs): the Break G slope stays between {sbg_min} and {sbg_max}; the day-3 gain runs from {sd3_min} (50+ routes) to {sd3_max}. Robustness checks: `robustness.csv`. Catches use {cr_per_yd} catch-rate points per yard and {tg_season} targets.
