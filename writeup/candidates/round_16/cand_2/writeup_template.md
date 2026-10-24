# Combine Cuts Don't Get Receivers Open

**A receiver's Combine cut does not predict how open he gets in the NFL. For day-3 receivers, last season's separation is a small tiebreaker, not a grade. Stop paying for the drill cut, and use the tracking on film to coach the break.**

*Break G is how hard a receiver pushes off the ground at the sharpest point of his cut.¹ Separation over expected (SOE) is the space he has at the throw beyond what his alignment and cushion predict (NGS, full season).*

![Figure 1. Four receivers with the largest drill-to-NFL gaps, chosen by a stated rule, so expect regression to the mean. Each is drawn over the league median for his route.](figures/fig1_cards.png)

## Why now, why receivers

Scouts grade receivers on "suddenness" in the gauntlet and route drills, but the Combine stopwatch only times straight lines and cones. Since 2023 those route drills have been tracked at 10 Hz, so the cut itself can be measured, with the same code a club can run on its own game tracking. Receivers are where this matters most. Getting open is the job, and the sixth receiver spot goes to day-3 players on minimum deals, whose college tape rarely settles who separates. Public audits find that tracking adds about nothing to the 40. This study asks a narrower question: does the same cut carry to the same NFL route, and if not, what breaks the tie?

## The roster decision

| Question (n) | Held-out result [90% interval] | Verdict |
|---|---|---|
| Does Break G add season SOE? ({n_main} receivers, unseen classes) | {prim_b} in per sd [{prim_lo}, {prim_hi}]; a +3 in effect rejected | **No** |
| Does the drill cut carry to the same NFL route family? ({fp_pool_n} receivers with 8+ NFL breaks) | noise-corrected correlation {fp_pool_med} [{fp_pool_lo}, {fp_pool_hi}] | **No visible transfer** |
| Does last season's SOE beat pick and routes on day 3? ({ev_d3_n} pairs, {ev_d3_np} receivers) | rank-correlation gain {ev_d3_drho} [{ev_d3_drho_lo}, {ev_d3_drho_hi}], each season forecast from the other | **Not proven: tiebreaker only** |

**Verdict.** On day 3 the edge is not distinguished from zero: its interval includes 0, and as a yes/no list it hits {r_d3_rule_hit} of {r_d3_rule_n} against {r_d3_pk_hit} of {r_d3_pk_n} for pick and routes. Separation persists at every pick, so where little else is known it is the best tiebreaker available. The appendix lists each estimate beside its alternatives.

## What to do

1. **Break ties between day-3 receivers with last season's route-shrunk SOE.** Act first on {{ACTON}}.
2. **Take Break G out of the price.** At the same position, clubs draft a receiver 1 sd harder in the cut {price_bg}% earlier [{price_bg_lo}, {price_bg_hi}] and get no separation for it. Its best case is worth {best_c} catches a season.
3. **Use the tracking on film, not on the board.** Read entry speed, speed at the top and drive out of the break by route family (Figure 3). It is a coaching cue, not a grade.
4. **2026 Combine protocol.** If you time cut drills, run slant, dagger and comeback twice. One rep is {vp_pct}% player; a second rep on those three lifts reliability from {rel_mm1} to {rel_mm3}. Read speed from the official 40, not from the tracked clip, which starts about 0.3 s before the laser.

## Drill to tape

Break G is found on {pct_found}% of {n_reps} drill reps. It is standardized within drill and Combine year and averaged over route drills (median {drills_med}). The same code grades NFL snaps on slant, out, hitch, in and corner routes.

![Figure 2. The drill cut repeats across drills (A) but does not carry to NFL snaps (B), on any route family once noise is modelled (C).](figures/fig2_chain.png)

- **The drill trait is real.** Break G on five drills predicts the other five at ρ = {rel_split} ({rel_split_n} players). It is not a speed score: its ρ with the 40 is {rho_forty}.
- **It does not reach NFL routes.** Game cuts are softer (median {g_drill_med} g in drills, {g_nfl_med} g on NFL snaps), and drill order is lost.
- **It is the metric, not only thin data.** Once the noise of a few reps and a few snaps is modelled, every route family's interval includes zero, hitch included.

## Season separation on unseen classes

The outcome is full-season SOE, route-weighted over each career ({n_seasons} player-seasons, median {routes_med} routes). The forecast without tracking was fixed before the test: log pick, official 40, weight, NGS athleticism score and a TE flag. Fitted on two classes and scored on the third, it leaves nothing for Break G to explain. Every variant in the appendix stays at or below +{t5_max_hi} inches per sd. A +3 inch effect would be worth only {sesoi_catch} catches a season.

## The 2026 day-3 shortlist

Day-3 or undrafted, 100+ routes and SOE above zero in 2025. SOE is shrunk toward the mean according to routes run. Probabilities are for SOE above zero in 2026. **Bold** = act on: the top four among receivers who played 14+ games. ⚑ = 13 or fewer games in 2025 (availability). § = Combine-discounted (NGS athleticism score below his class average).

{{SHORTLIST}}

**What a hit rate means in names.** The same rule, fitted on the other season, listed these day-3 receivers before (✓ = SOE stayed above zero): {{HISTORY}}.

**Pre-registered.** The 2026 rule, its coefficients and these names and probabilities are frozen in `frozen_rule_2026.json` (sha256 {sha26}…, {freeze_date}). The 2026 season is under way, but no full-season SOE exists yet. Primary test: day-3 rank-correlation gain over pick and routes on an expected {pw_n26} pairs ({pw_keep}% of day-3 receivers keep 100+ routes). Its power is only {pw_g} if the true gain equals the observed one, and {pw_pool} once pooled with 2023–25. The list's hit rate needs {pw_hit_k} of {l26_n} to beat the base rate (power {pw_hit}), so 2026 alone cannot settle the claim.

## Matched contrasts and one illustration

![Figure 3. Boutte (hard drill cutter) and Washington (soft) on NFL hitches: speed through each break against the league median and middle 50%.](figures/fig3_film.png)

**The pre-specified contrast.** We compared every pair of day-3 receivers in the same season ({ct_all_n} contrasts). In the {ct_n} pairs whose pick-and-routes forecasts were closest (gap under {ct_med} in), the receiver with the higher shrunk SOE separated more the next season {ct_m}% of the time [{ct_m_lo}, {ct_m_hi}]. Across all contrasts, SOE ordered {ct_a}% correctly [{ct_a_lo}, {ct_a_hi}], against {ct_pk}% for pick and routes: a small, uncertain edge, which is what a tiebreaker should look like.

**An illustration, chosen for contrast.** Kayshon Boutte (2023, pick {wk_kb_pick}) and Malik Washington (2024, pick {wk_mw_pick}) have opposite drill cuts ({l26_boutte_bg} and {l26_washington_bg} sd). Pick and routes give both {l26_boutte_pc0}%. Their 2025 SOE ({l26_boutte_soe} vs {l26_washington_soe} in) moves that to {l26_boutte_pc1}% vs {l26_washington_pc1}%. Boutte played {wk_kb_games} games with Drake Maye ({wk_kb_man}% man); Washington played {wk_mw_games} with Tua Tagovailoa ({wk_mw_man}% man). Both earn about $1M. The gap is worth {wk_dc} catches a season ({wk_dc_lo} to {wk_dc_hi}), a lean toward Washington, not a verdict. On NFL hitches they look alike until the top, where Boutte still moves at {film_kb_vap} yd/s and Washington sinks to {film_mw_vap} (league {film_lg_vap}). Coach Boutte to sink fully at the top of hitches against off coverage. The drill score does not show this.

## How the rule scored

![Figure 4. Day-3 and undrafted season pairs sorted by outcome, with each list's hits and misses (A); the gain over pick and routes against a shuffle null (B); calibration, and where the 2026 list sits (C).](figures/fig4_shortlist.png)

Held-out probabilities ran high (day 3: {cal_d3_pred}% forecast, {cal_d3_obs}% observed), so the 2026 list shows rescaled values.

## Limitations

- **Not blind, small day-3 sample.** All three classes were seen while building this, and day 3 has {ev_d3_n} pairs. The 2026 season is the first true test, and it is underpowered (above).
- **The downside is open.** The primary interval rules out gains above {prim_hi} inch per sd, not losses down to {prim_lo}.
- **Game tracking is a sample.** Up to 40 plays per player, half chosen for high EPA, with no defenders and no ball. That limits the film cue and the drill-to-game link, not the season SOE test.
- **Availability and survivorship.** Games played is a rough injury proxy. {sel_out} of {sel_n} drill-scored receivers never logged NFL routes with separation; re-weighting for this changes little.

## Appendix

**One estimate per claim, and its alternatives.**

| Claim | Estimate used in the text (n) | Alternative estimate (n) | Why they differ |
|---|---|---|---|
| Day-3 gain of SOE over pick and routes | season folds: {ev_d3_drho} [{ev_d3_drho_lo}, {ev_d3_drho_hi}] ({ev_d3_n} pairs) | leave one receiver out, with pick × SOE term: {lo_d3_gain} [{lo_d3_lo}, {lo_d3_hi}] (same pairs) | Leaving one receiver out trains on more pairs, but its pick-and-routes baseline ranks day-3 pairs at {lo_d3_r0}, which flatters the gain |
| Drill cut to NFL cut | noise-corrected, same route family: {fp_pool_med} [{fp_pool_lo}, {fp_pool_hi}] ({fp_pool_n}, 8+ NFL breaks) | raw ρ, all families: {pl_rho} [{pl_lo}, {pl_hi}] ({pl_n}, any NFL breaks) | Raw ρ mixes route families and carries one-rep and few-snap noise |

**Break G slopes, inches of season SOE per sd, held out unless stated.**

| Version (n) | Slope [90%] | p against +3 in |
|---|---|---|
| **Pre-specified offset forecast, primary ({n_main})** | **{prim_b} [{prim_lo}, {prim_hi}]** | **{t5_plain_p3}** |
| Ridge forecast ({n_main}) | {ridge_b} [{ridge_lo}, {ridge_hi}] | {t5_ridge5_p3} |
| Re-weighted by 1 / P(reaching the NFL sample) ({n_main}) | {t5_ipw_b} [{t5_ipw_lo}, {t5_ipw_hi}] | {t5_ipw_p3} |
| Player-seasons, receiver random intercept, in-sample ({mm_ps_n}) | {t5_mixed_b} [{t5_mixed_lo}, {t5_mixed_hi}] | {t5_mixed_p3} |
| Without Puka Nacua, most influential ({nn_n}) | {nn_b} [{nn_lo}, {nn_hi}] | {t5_nn5_p3} |
| SOE adjusted for quarterback ({n_main}) | {qb_b} [{qb_lo}, {qb_hi}] | {t5_qb5_p3} |

**Methods.** ¹Break G = mean acceleration in g (1 g = body weight) over the 1.0 s centred on the sharpest point of the first turn; 1 sd = {g_sd} g, the spread of true receiver Break G. *Family model:* errors-in-variables, random slope per route family, Gibbs sampler ({fp_cells} receiver-family cells); chance the true link exceeds 0.2 is {fp_pool_p02}% pooled, at most {fp_max_p02}% per family. *Persistence:* receiver random intercept, no pick × SOE interaction ({nx_int} [{nx_int_lo}, {nx_int_hi}]). *Calibration:* Platt rescaling fitted on held-out pairs. *Shrinkage:* {eb_k200} of SOE kept at 200 routes. *Units:* {cr_per_yd} catch-rate points per yard, {tg_season} targets a season. Intervals resample receivers.

Reproduce: `uv run python writeup/candidates/round_16/cand_2/analysis.py` (about ten minutes). It writes `numbers.json`, the figures, `frozen_rule_2026.json` and this text from `writeup_template.md`.
