# Combine Cuts Don't Predict NFL Separation

**A receiver's Combine cut does not predict how open he gets in the NFL. On {n_main} receivers from classes the model never saw, 1 sd of Break G is worth {prim_b} inches of separation [90%: {prim_lo} to {prim_hi}], so a gain above +1 inch is ruled out. For day-3 receivers, last season's separation is a small tiebreaker, not a grade.**

*Break G is how hard a receiver pushes off the ground at the sharpest point of his cut. Separation over expected (SOE) is the space he has at the throw beyond what his alignment and cushion predict (NGS, full season).*

![Figure 1. All 99 receivers with 8+ NFL breaks (A) and the two extremes, chosen by rule (B).](figures/fig1_cards.png)

## Why receivers, why now

The sixth receiver spot goes to day-3 players who cost about ${why_apy}M a year, and the bet is on one skill: getting open. The evidence is thin. The Combine shows a scout a median of {why_reps} graded route-drill cuts per receiver, and {why_single}% of drill-receiver cells hold a single rep. Separation is noisy even in the NFL: a 200-route season keeps only {eb_k200} of its weight once shrunk for sample size. A college season, against different coverage and without tracking, settles less. Since 2023 the route drills are tracked at 10 Hz, so the cut can be measured with code a club can also run on game tracking. Straight-line speed does carry: the official 40 tracks NFL top speed (ρ = {spd_rho}, {spd_n} receivers). The question is whether the cut carries too, and if not, what breaks the tie.

## The roster decision

| Question (n) | Held-out result [90% interval] | Verdict |
|---|---|---|
| Does Break G add season SOE? ({n_main} receivers, unseen classes) | {prim_b} in per sd [{prim_lo}, {prim_hi}] | **No** |
| Does the drill cut carry to the same NFL route? ({fp_pool_n} with 8+ NFL breaks) | true correlation {fp_pool_med} [{fp_pool_lo}, {fp_pool_hi}] | **No visible transfer** |
| Does last season's SOE beat pick and routes on day 3? ({ev_d3_n} pairs, {ev_d3_np} receivers) | rank-correlation gain {ev_d3_drho} [{ev_d3_drho_lo}, {ev_d3_drho_hi}] | **Not proven: tiebreaker only** |
| Closest forecasts: does the higher-SOE receiver separate more next year? ({ct_n} contrasts) | {ct_m}% [{ct_m_lo}, {ct_m_hi}]; shuffle p = {ctp_p} | **A lean, not proof** |

- **Do not pay for the drill cut.** No version of the test finds a gain worth a roster spot.
- **The Combine cut is not the NFL cut.** The same receiver on the same route shows no visible link.
- **Last season's SOE is the best day-3 tiebreaker available, and it is small.** Its interval includes zero.

## What to do

1. **Break ties between day-3 receivers with last season's route-shrunk SOE.** Act first on {{ACTON}}.
2. **Take Break G out of the price.** At the same position, clubs draft a receiver 1 sd harder in the cut {price_bg}% earlier [{price_bg_lo}, {price_bg_hi}] and get no separation for it.
3. **Use the tracking on film, not on the board.** On a hitch, Break G is the stop: the typical NFL receiver enters at {film_lg_vin} yd/s, sinks to {f3_h_lg} yd/s at the top and drives back to the ball. On slants, outs and corners the NFL cut is taken at speed (median {vap_slant}, {vap_out} and {vap_corner} yd/s at the turn), so Break G there is mostly turn, not brake (Figure 3). Hard and soft drill cutters run the same NFL hitch (gap at the turn {f3_hitch} yd/s [{f3_hitch_lo}, {f3_hitch_hi}]), and how deep a receiver sinks does not track his separation (ρ = {f3_h_rho}, {f3_h_n} receivers). Coaching a full sink against off coverage is a hypothesis to check on film, not a trait to draft.
4. **2026 Combine protocol.** If you time cut drills, run slant, dagger and comeback twice: a second rep on those three lifts reliability from {rel_mm1} to {rel_mm3}. Read speed from the official 40, not from the tracked clip, which starts about 0.3 s before the laser.

## Drill to tape

Break G is found on {pct_found}% of {n_reps} drill reps. It is standardized within drill and Combine year and averaged over route drills (median {drills_med}). The same code grades NFL snaps on slant, out, hitch, in and corner routes.

![Figure 2. The drill cut repeats across drills (A) but not on NFL snaps (B, raw), on any route family once noise is modelled (C). The official 40 is shown for scale.](figures/fig2_chain.png)

- **The drill trait is real.** Break G on five drills predicts the other five at ρ = {rel_split} ({rel_split_n} players). It is not a speed score: its ρ with the 40 is {rho_forty}.
- **It does not reach NFL routes.** Once the noise of a few reps and a few snaps is modelled, every family's interval includes zero. The pooled true correlation is below {fp_pool_hi} with 90% probability, under the 40's raw {spd_rho}.
- **The null is bounded, not proven.** {fp_pool_pin}% of the pooled posterior lies within ±0.2, so a small link cannot be excluded; a useful one is unlikely ({fp_pool_p02}%).

![Figure 3. NFL speed through the cut for the top and bottom thirds of Combine Break G, by route family: the hard drill cutters do not cut differently on Sundays.](figures/fig3_film.png)

## Season separation on unseen classes

The outcome is full-season SOE, route-weighted over each career ({n_seasons} player-seasons, median {routes_med} routes). The forecast without tracking was fixed before the test: log pick, official 40, weight, NGS athleticism score and a TE flag. Fitted on two classes and scored on the third, it leaves nothing for Break G to explain. Against the pre-set margin of ±3 inches per sd, the upside is rejected (p = {prim_up_p}) and the downside is not (p = {tost_dn}): Break G may cost a little and cannot add much. A +3 inch effect would be worth {sesoi_catch} catches a season.

## The 2026 day-3 shortlist

Pool: day-3 or undrafted, 100+ routes and SOE above zero in 2025; SOE shrunk by routes run. P = chance of SOE above zero in 2026. Act on = the top four among receivers who played 14+ games.

{{SHORTLIST}}

Route mix and coverage splits come from up to 40 sampled snaps; a man-minus-zone split has a split-half reliability of {sc_rel}, so read it as a film prompt. Each QB is the 2025 starter; a QB change resets the read. **Watch:** {{WATCH}}. Full list in the appendix. The rule is frozen in `frozen_rule_2026.json`; 2026 alone is underpowered.

**How earlier lists did.** The same rule, fitted on the other season, listed these day-3 receivers (hit = SOE stayed above zero):

{{HISTORY}}

![Figure 4. All 20 day-3 season pairs with each list's hits and misses (A); held-out ranking of the same pairs (B); the same pairs as within-season contrasts (C).](figures/fig4_shortlist.png)

The rule ranks next-season SOE at {r3_rule} against {r3_pk} for pick and routes. Most of that is last season's SOE itself: the rule beats raw last-season SOE by only {nv_g} [{nv_lo}, {nv_hi}]. Over all {ct_all_n} within-season contrasts, SOE orders {ct_d} points more of them right than pick and routes [{ct_d_lo}, {ct_d_hi}].

## Limitations

- **Not blind, small day-3 sample.** All three classes were seen while building this, and day 3 has {ev_d3_n} pairs. The 2026 season is the first true test, and it is underpowered (appendix).
- **The downside is open.** The primary interval rules out gains above {prim_hi} inch per sd, not losses down to {prim_lo}.
- **Game tracking is a sample.** Up to 40 plays per player, half chosen for high EPA, with no defenders and no ball. That limits the film cues and the drill-to-game link, not the season SOE test.
- **Availability and survivorship.** Games played is a rough injury proxy. {sel_out} of {sel_n} drill-scored receivers never logged NFL routes with separation; re-weighting for this changes little.
- **The freeze is late.** The 2026 rule was frozen on {freeze_date}, after the season began but before any full-season SOE existed.

## Appendix

**Pre-registration.** `frozen_rule_2026.json` holds the coefficients, shrinkage, threshold, names and probabilities (sha256 {sha26}…). Primary test: day-3 rank-correlation gain over pick and routes on an expected {pw_n26} pairs ({pw_keep}% of day-3 receivers keep 100+ routes). Power is {pw_g} at the observed gain and {pw_pool} pooled with 2023–25; the list needs {pw_hit_k} of {l26_n} hits to beat the base rate (power {pw_hit}).

**Every estimate and its alternatives** (90% intervals; receivers resampled).

| Claim | Primary (n) | Alternatives |
|---|---|---|
| Break G slope, in of SOE per sd | offset forecast {prim_b} [{prim_lo}, {prim_hi}] ({n_main}) | re-weighted for survival {t5_ipw_b} [{t5_ipw_lo}, {t5_ipw_hi}]; without Puka Nacua {nn_b} [{nn_lo}, {nn_hi}]; QB-adjusted {qb_b} [{qb_lo}, {qb_hi}] |
| Drill cut to NFL cut | true correlation, same family {fp_pool_med} [{fp_pool_lo}, {fp_pool_hi}] ({fp_pool_n}) | raw ρ, any family: {pl_rho} [{pl_lo}, {pl_hi}] ({pl_n}); raw ρ mixes families and keeps the noise |
| Day-3 gain over pick and routes | season folds {ev_d3_drho} [{ev_d3_drho_lo}, {ev_d3_drho_hi}] ({ev_d3_n} pairs) | shrinkage refitted inside each fold {ns_g} [{ns_lo}, {ns_hi}]; draft-class folds {nc_g} [{nc_lo}, {nc_hi}] |

Class folds cannot be crossed with season folds: 2024 rookies have no 2023 season. P (chance of SOE above zero) is rescaled to held-out hit rates (day 3: {cal_d3_pred}% forecast, {cal_d3_obs}% observed). Shrinkage keeps {eb_k200} of SOE at 200 routes. The true correlation comes from an errors-in-variables model with a slope per route family ({fp_cells} receiver-family cells). 1 sd of Break G = {g_sd} g; catches use {cr_per_yd} catch-rate points per yard and {tg_season} targets.

**Rest of the 2026 list** (class, pick; team; games; shrunk SOE; P): {{REST}}

Reproduce: `uv run python writeup/candidates/round_16/revised/analysis.py` (about eight minutes) writes every number, figure and this text.
