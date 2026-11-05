# Combine Cuts Show Little Sign of Carrying to NFL Separation

*Break G* grades a receiver's route cut from 10 Hz tracking. *Separation over expected (SOE)* is how open he is when the ball is thrown, beyond what his alignment and cushion predict (Next Gen Stats, full season), in inches.

**Receivers who cut hardest in Combine route drills are not more open in the NFL: about 1 inch less per sd of Break G, with a 90% range from 4 inches less to 1 inch more. For the last roster spot, keep the day-3 receiver with better last-season separation; the evidence is suggestive, not proven.**

![Figure 1. All 99 receivers with 8+ graded NFL breaks, Combine against NFL Break G (A), and five receivers that span the Combine range (B).](figures/fig1_cards.png)

## What Break G measures

Acceleration is force divided by body mass, so a receiver's acceleration in g is the horizontal push he puts into the turf as a share of his body weight. Break G averages that push over the one second centred on the sharpest point of his first route break: braking into the cut, turning, and driving out, in one number. A typical drill cut runs at {bg_med} g, a push of about {bg_med_pct}% of body weight held for a second. One sd between receivers is {g_sd} g, about {g_sd_pct}% of a typical cut, and it is a real trait: Break G on five drills predicts the other five at ρ = {rel_split} ({rel_split_n} players), and it is not a speed score (ρ with the 40: {rho_forty}). The same code grades NFL slants, outs, hitches, ins and corners.

## Why receivers, why now

Every summer the last receiver spot goes to a day-3 or undrafted player. In 2025 those who ran 100+ routes earned a median ${why_apy}M, close to the league minimum. Both candidates cost the same; a wrong call costs the production a better receiver never gets.

A scout sees a median of {why_reps} graded route-drill cuts per receiver, and {why_single}% of drill-receiver cells hold one rep. In the NFL, one 200-route season of SOE is about {eb_sig}% signal: if two receivers differ by 10 inches this year, expect about {eb_sig}% of that gap to persist next year. Straight-line speed is the benchmark: the official 40 ranks NFL top speed at ρ = {spdf_rho} [{spdf_lo}, {spdf_hi}] within position and class ({spdf_n} receivers). Does the cut carry the same way?

## The roster decision

| Question (n) | Held-out result [90% interval] | Verdict |
|---|---|---|
| Does Break G add season SOE beyond pick, 40, weight and NGS score? ({n_main} receivers, unseen classes) | {prim_b} in per sd [{prim_lo}, {prim_hi}] | **No benefit; harm not ruled out** |
| Does the drill cut carry to the same NFL route? ({fp_pool_n} receivers) | correlation {fp_pool_med} [{fp_pool_lo}, {fp_pool_hi}], noise corrected | **No visible transfer** |
| Does last season's SOE rank day-3 receivers better than pick and routes? ({ev_d3_n} pairs, {ev_d3_np} receivers) | gain in rank correlation {ev_d3_drho} [{ev_d3_drho_lo}, {ev_d3_drho_hi}], one-sided p = {d3c_p} | **Lean** |

## What to do

1. **Take Break G out of the price.** At the same position, clubs draft a receiver 1 sd harder in the cut {price_bg}% earlier [{price_bg_lo}, {price_bg_hi}] and get no separation for it.
2. **Change the drill before timing it.** Combine stems are run faster than NFL cuts are entered: {vin_d_hitch} vs {vin_n_hitch} yd/s into the hitch, {vin_d_out} vs {vin_n_out} into the out, {vin_d_in} vs {vin_n_in} into the in. Cap the stem at game entry speed (a shorter, timed stem) and run slant, dagger and comeback twice: a second rep lifts reliability from {rel_mm1} to {rel_mm3}.
3. **Tiebreak only: between two day-3 wide receivers the staff rate as even, keep the one with higher last-season SOE.** SOE orders {tb_all_right}% of {tb_n} same-season day-3 contrasts right. Where it disagrees with the earlier pick ({pr_n} contrasts), it gained {pr_gain} inches of next-season SOE when right and lost {pr_loss} when wrong, so it pays while wrong on fewer than {pr_be} of 100 such calls; it was wrong on {pr_obs}. If SOE were pure noise, the expected cost is {pr_noise} inches per call ({pr_noise_c} catches a season), because the draft slot it replaces orders day-3 receivers right only {tb_pick_right}% of the time. Not for tight ends: {pp_te_n} pairs, gain {te_g}.
4. **2026: GO on Malik Washington and Demario Douglas.** Review each at 150 routes (Figure 5) and decide before the March 2027 league year.

## The drill cut does not show up in NFL routes

![Figure 2. Drill-to-NFL correlation of Break G by route family (A), and the season SOE slope of Break G and each of its three mechanics on unseen classes (B).](figures/fig2_chain.png)

After correcting for the noise of few reps and few snaps, the drill-to-NFL correlation for the same receiver on the same route is {fp_pool_med}, and a useful link (above 0.2) has a {fp_pool_p02}% chance. Splitting Break G into its three mechanics does not find a hidden carrier (Figure 2B): decel into the break {mc_decel} in per sd [{mc_decel_lo}, {mc_decel_hi}], speed through it {mc_v_apex} [{mc_v_apex_lo}, {mc_v_apex_hi}], re-acceleration {mc_reaccel} [{mc_reaccel_lo}, {mc_reaccel_hi}]; the smallest Holm-adjusted p is {mc_holm_min}. The parts are noisier than the whole (split-half reliability {mc_decel_rel}, {mc_v_apex_rel} and {mc_reaccel_rel}).

We also tried the NFL-style reading of the drill: speed carried through the turn on slants and outs, where NFL cuts are taken at speed. Drill speed at the turn does not predict NFL speed at the turn (ρ = {vs_rho} [{vs_lo}, {vs_hi}], {vs_n} receivers).

![Figure 3. Malik Washington, one route, drill against game: the Combine curl drill and his median-speed NFL hitch in field coordinates, coloured by speed (A, B), and speed through each cut (C).](figures/fig3_path.png)

Figure 3 shows why. Washington's curl drill is a long sprint (entry {f3w_d_vin} yd/s) into a stop; his typical NFL hitch is a shorter stem (entry {f3w_n_vin} yd/s) that settles to {f3w_n_vap} yd/s. The Break G values are close ({f3w_d_bg} and {f3w_n_bg} g), but the two cuts ask different things of the legs.

**The null in names.** The eight hardest and eight softest drill cutters with 200+ career routes and 5+ graded NFL breaks ({ex_pool} eligible). Turn speed is the median of his NFL breaks minus the league median for the same route.

{{EXTREMES}}

The hardest eight average {ex_top_soe} inches of career SOE, the softest eight {ex_bottom_soe}; their NFL turn speed is the same ({ex_top_v} and {ex_bottom_v} yd/s against the league).

## Break G adds nothing to season separation

The outcome is full-season SOE, route-weighted over each career ({n_seasons} player-seasons, median {routes_med} routes). The forecast without tracking was fixed before the test: log pick, official 40, weight, NGS athleticism score and a TE flag. Fitted on two classes and scored on the third, it is weak: held-out R² {bl_r2} [{bl_r2_lo}, {bl_r2_hi}], error {bl_rmse} inches. Adding Break G changes R² by {dr2} [{dr2_lo}, {dr2_hi}].

How small an effect can these data rule out? Benefits above {prim_hi} inch per sd and harms beyond −{eqm} inches are each excluded at the 5% level; equivalence inside the pre-set ±3 inch margin ({sesoi_catch} catches a season) is not shown, so the result is "no benefit", not "no effect". The minimum detectable effect at 80% power is {mde80} inches ({mde_catch} catches). Rebuilding the drill score's shrinkage inside every resample, so the uncertainty in its reliability ({eiv_icc_lo} to {eiv_icc_hi} per drill) reaches the slope, gives {eiv_b} [{eiv_lo}, {eiv_hi}].

## Day 3: last season beats the draft slot, narrowly

The gain is how much adding last season's SOE improves the held-out forecast's rank correlation with next-season SOE; models are fitted on one season pair and scored on the other. The interval resamples receivers; the p shuffles whole receivers' last-season SOE.

![Figure 4. Day-3 wide receivers' forecast errors, ranked (A), the pick-gap rule with its price (B), and gains with intervals (C).](figures/fig4_tiebreak.png)

Two checks on the split. Training on a random half of receivers and scoring the other half's day-3 pairs, {rs_n} times, the gain has median {rs_med} (5th–95th percentile {rs_lo} to {rs_hi}) and is positive in {rs_pos}% of splits. {inj_n} of the 20 pairs include a season of 13 or fewer games; without them the gain is {inj_gain} ({inj_keep} pairs). Inside day 3 the earlier pick is close to a coin flip ({tb_pick_right}% of {tb_n} contrasts ordered right), and limiting SOE to small pick gaps does not help (Figure 4B).

## The 2026 calls

Pool: day-3 or undrafted receivers with 100+ routes and SOE above zero in 2025. **GO** = listed again after an earlier hit; **HOLD** = first listing. True SOE = 2025 SOE shrunk for routes, with its 90% interval.

{{SHORTLIST}}

![Figure 5. The 2026 list with intervals (A) and the four act-on names' sampled-snap SOE by coverage and alignment (B), as film prompts.](figures/fig5_calls.png)

Arian Smith is also one of the hardest drill cutters above; his case rests on SOE, not on the cut. The coverage and alignment splits in Figure 5B come from up to 40 sampled snaps per player and only {sp_big_cells} of 16 cells reaches 30 snaps, so they direct film study, not the call.

## Limitations

- **Not blind, small day-3 sample.** All three classes were seen while building this; the day-3 test has 20 pairs and p = {d3c_p}. Dropping any one pair leaves the gain between {lopo_min} and {lopo_max}.
- **The harm side is open.** The Break G interval reaches {prim_lo} inches per sd; only a benefit is ruled out.
- **Game tracking is a sample.** Up to 40 plays per player, half chosen for high EPA, without defenders or ball. That limits Figure 3, the splits and the drill-to-NFL link, not the season tests.
- **The 40.** The tracked 40 clip starts before the laser, so every speed number uses the official time.
- **Survivorship.** {sel_out} of {sel_n} drill-scored receivers never logged NFL routes with separation; re-weighting for this changes little.
- **One snapshot of the drill.** Where receivers ran a drill twice, test-retest ICC has median {iccd_med} over {iccd_n} drills ({iccd_pairs} repeat pairs).

## Appendix

**Robustness.** The Break G slope barely moves when survivors are re-weighted ({t5_ipw_b}), Puka Nacua is dropped ({nn_b}) or season SOE is adjusted for the quarterback ({qb_b}); the day-3 gain holds with the shrinkage refitted in each fold ({ns_g}) but halves with draft-class folds ({nc_g}). Every draft slot matches the nflverse record ({dchk_n} receivers). 1 sd of Break G = {g_sd} g; catches use {cr_per_yd} catch-rate points per yard and {tg_season} targets.

**Reproduce.** `analysis.py` writes every number, figure and this text; `verify.py` re-checks the headline numbers in seconds.
