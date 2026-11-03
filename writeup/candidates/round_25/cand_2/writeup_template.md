# Break G: The Combine Grades a Receiver Cut That Sundays Don't Ask For

**Receivers who plant harder in Combine route drills are not more open in the NFL. On {n_soe} receivers, each draft class scored by a forecast fit on the other two, one standard deviation of drill Break G moved separation over expected by {full_b} inches (90% interval {full_b_lo} to {full_b_hi}). This is one metric, defined before the test; it is a bounded null, not a hidden edge.**

**Break G** measures how hard a receiver plants: his average g-force over the one second around his first real cut, covering the brake into the cut, the turn and the drive out. On film it reads as three speeds: into the break, at the plant and out of it. A typical NFL hitch break is {hitch_bg_nfl} g; the same break in the Combine hitch drills is {hitch_bg_drill} g.

**Why now.** Route drills are the only Combine test that looks like playing receiver, and scouts grade "sink and burst" off them. This is the first cycle where every rep is tracked at 10 Hz next to NFL tracking of the same men, so the grade can be checked against Sundays before it hardens into a number. Clubs already pay for it: at the same 40, weight, NGS score and position, a receiver 1 sd higher in Break G goes {price}% earlier in pick number (90% interval {price_lo} to {price_hi}; {n_drill} receivers).

**Data**
- **Combine:** {n_reps} tracked reps in ten route drills (2023–2025 classes), {n_cuts} with a gradable cut, each scored against the same drill and year.
- **NFL:** the same code on {n_nfl_breaks} sampled regular-season breaks by {n_nfl_rec} receivers, in the five route families the drills copy: slant, out, hitch (curl, comeback, hook), in (dagger) and corner (sail).
- **Outcome:** separation over expected (SOE), inches more open at the throw than Next Gen Stats separation predicts from alignment, cushion, position and season. Full regular seasons ({n_pseason} player-seasons, median {med_routes} career routes), not the tracking sample.

![Figure 1. Route paths around the first cut, Combine drill (top) vs sampled NFL snaps of the same route family (middle), rotated so the stem points up and mirrored so the break goes right; colour is speed. Bottom: median speed through the plant with the middle half of reps. Rep counts in each panel.](figures/fig1_route_paths.png)

## 1. The drill asks for a different cut

Figure 1 draws the routes. At the Combine every rep runs the same scripted path and arrives hot (yellow-green). On Sundays the same second and a half covers less ground because he arrives slower, and the exit angle fans out with coverage. On the hitch family he arrives at {hitch_in_drill} yd/s (about {hitch_in_drill_mph} mph) in the drill and {hitch_in_nfl} yd/s (about {hitch_in_nfl_mph} mph) in the NFL. Drill entry is faster in all {n_fam_faster} families, and only {share_nfl_at_drill}% of NFL breaks are entered as fast as a typical drill rep. Because Break G rises with entry speed (rank correlation {rho_bg_vin_drill} in drills, {rho_bg_vin_nfl} in games), the drill mostly rewards stopping from a speed NFL routes rarely use.

## 2. Speed carries; the cut does not

![Figure 2. One dot per receiver. A: official 40 vs top speed in sampled NFL snaps. B: drill Break G vs NFL Break G from the same code (8+ sampled breaks). Named receivers are those in Table 2.](figures/fig2_transfer.png)

The control works: a receiver's official 40 ranks with his NFL top speed at {f40_rho} (90% interval {f40_lo} to {f40_hi}; {f40_n} receivers). The cut does not: drill Break G vs NFL Break G is {rec_rho} ({rec_lo} to {rec_hi}; {rec_n} receivers). Both measures are noisy, but a true link of {link} would still read about {true05}.

We also checked each route family and each part of the cut separately ({n_cells} cells, {pair_n} receiver-family pairs). The best cell is braking into the plant, {all_decel} ({all_decel_lo} to {all_decel_hi}). Once the search over {n_cells} cells is accounted for by shuffling, its adjusted p-value is {brk_adj_p}: not a finding.

## 3. Break G adds nothing to a no-tracking forecast

The test was fixed in advance: fit on two draft classes, score the third, rotate, weight by routes; ±3 inches per sd is the smallest effect a club would act on. Intervals resample whole receivers and refit every step.

| No-tracking forecast | Its held-out R² | R² change adding Break G [90%] | Break G effect, inches per sd [90%] |
|---|---|---|---|
| Pick only | {pick_r2} | {pick_dr2} [{pick_dr2_lo}, {pick_dr2_hi}] | {pick_b} [{pick_b_lo}, {pick_b_hi}] |
| Pick + role (TE, slot share) | {role_r2} | {role_dr2} [{role_dr2_lo}, {role_dr2_hi}] | {role_b} [{role_b_lo}, {role_b_hi}] |
| Ridge: pick, 40, weight, NGS, TE, slot | {ridge_r2} | {ridge_dr2} [{ridge_dr2_lo}, {ridge_dr2_hi}] | {ridge_b} [{ridge_b_lo}, {ridge_b_hi}] |
| **Pick + 40, weight, NGS, TE** (set in advance) | {full_r2} | {full_dr2} [{full_dr2_lo}, {full_dr2_hi}] | **{full_b} [{full_b_lo}, {full_b_hi}]** |

*Table 1. {n_soe} receivers with a drill score and NFL separation. An R² below zero means the forecast does worse on a new class than that class's own average.*

![Figure 3. A: each dot is a receiver; the vertical axis is his career SOE minus the no-tracking forecast fit on the other two classes. One line per held-out class; the blue wedge is the ±3 in per sd margin. B: the pooled effect when Break G is shuffled within class; dotted lines mark the middle 90%.](figures/fig3_heldout.png)

Career SOE varies by {sd_soe_in} inches between receivers; draft slot plus testing explains none of it on a new class, and Break G does not help under any forecast. All three classes lean the same way, slightly negative: 2023 {cls2023_b}, 2024 {cls2024_b}, 2025 {cls2025_b} inches per sd (Figure 3A), so no single class hides a positive effect. Three stress tests agree:

- **Shuffle placebo.** Shuffling Break G within class {n_shuffle} times puts 90% of effects between {plac_lo} and {plac_hi}. The one-sided p-value for "harder cutters are more open" is {plac_p1}.
- **Same team, same quarterback.** Resampling whole teams ({n_teams}) instead of receivers gives {team_b_lo} to {team_b_hi}. The one-sided p-value for an R² gain is {team_dr2_p1}.
- **No tracking at all.** The official 3-cone ({cone_b} in per sd, {cone_n} receivers) and short shuttle ({shut_b}, {shut_n}) say the same.

## 4. Is there a Sunday cut skill to predict?

Perhaps the drill misses a real NFL skill. We compared each receiver with himself, within one route family, on breaks finished before the throw, controlling for cushion and cut timing. Across all sampled snaps a harder Sunday cut goes with {snap_pre} inches more separation per sd ({snap_pre_lo} to {snap_pre_hi}; {snap_pre_n} breaks). On random snaps it falls to {snap_pre_rand} ({snap_pre_rand_lo} to {snap_pre_rand_hi}). A placebo, cuts finished after the throw, should read zero and reads {snap_post}. Between receivers, a man's own NFL Break G does not track his career SOE ({nsoe_rho}, {nsoe_lo} to {nsoe_hi}; {nsoe_n} receivers). In these data the cut grade is not a separation skill at any level.

## 5. Named receivers: the drill order does not survive

![Figure 4. Five eligible receivers ordered by drill Break G rank. Thick orange lines are his own Combine cuts, thin blue lines his own NFL breaks in his most-run shared route family; same rotation as Figure 1.](figures/fig4_named_paths.png)

| Receiver (class, pick; drill rank of {n_elig}) | Combine in → plant → out, yd/s (g) | NFL, same families (g) | Career SOE, in (routes) | Play-log clip | Verdict |
|---|---|---|---|---|---|
| {top_name} ({top_year}, {top_pick}; #{top_rank}) | {top_din} → {top_dap} → {top_dout} ({top_dg}) | {top_nin} → {top_nap} → {top_nout} ({top_ng}) | {top_soe} ({top_routes}) | {top_clip}; {top_clip_g} g, {top_clip_sep} yd open, {top_clip_log} | Hardest drill planter; sheds {top_nmph} mph into Sunday breaks vs {top_dmph} at Indy. Average separation. |
| {smi_name} ({smi_year}, {smi_pick}; #{smi_rank}) | {smi_din} → {smi_dap} → {smi_dout} ({smi_dg}) | {smi_nin} → {smi_nap} → {smi_nout} ({smi_ng}) | {smi_soe} ({smi_routes}) | {smi_clip}; {smi_clip_g} g, {smi_clip_sep} yd open, {smi_clip_log} | Near-dead stop in drills, ordinary Sunday cut ({smi_nbg} sd), wide open over a small sample. The space is not coming from the plant. |
| {nac_name} ({nac_year}, {nac_pick}; #{nac_rank}) | {nac_din} → {nac_dap} → {nac_dout} ({nac_dg}) | {nac_nin} → {nac_nap} → {nac_nout} ({nac_ng}) | {nac_soe} ({nac_routes}) | {nac_clip}; {nac_clip_g} g, {nac_clip_sep} yd open, {nac_clip_log} | Top-five drill cutter, below-expected separation on a huge workload; wins in tight windows, which SOE does not credit. |
| {mid_name} ({mid_year}, {mid_pick}; #{mid_rank}) | {mid_din} → {mid_dap} → {mid_dout} ({mid_dg}) | {mid_nin} → {mid_nap} → {mid_nout} ({mid_ng}) | {mid_soe} ({mid_routes}) | {mid_clip}; {mid_clip_g} g, {mid_clip_sep} yd open, {mid_clip_log} | Slowest into Sunday breaks; the defender sees it coming. Fix the stem, not the plant. |
| {bot_name} ({bot_year}, {bot_pick}; #{bot_rank}) | {bot_din} → {bot_dap} → {bot_dout} ({bot_dg}) | {bot_nin} → {bot_nap} → {bot_nout} ({bot_ng}) | {bot_soe} ({bot_routes}) | {bot_clip}; {bot_clip_g} g, {bot_clip_sep} yd open, {bot_clip_log} | Softest drill cutter, ordinary Sunday cut, near-average separation. Nothing to fix. |

*Table 2. Medians over route families he ran in both settings. The clip is his random-sample break closest to his own NFL median; the play-log column is the official play result, so a scout can pull the snap and confirm the read. Verdicts come from tracking and the play log; we have not watched the film.*

All five plant softer on Sundays than at Indianapolis, and drill rank does not sort their separation: #1 and #49 are both near average, #3 and #5 sit at opposite ends. The eight hardest drill cutters among {n_elig} eligible receivers average {top8_soe} inches of SOE; the eight softest, {bot8_soe}.

## 6. What to do on Monday

- **Draft room:** do not move a receiver up the board for his route-drill cut grade. A "top third of the class" rule flags about {t3_per_class} of {t3_class_n} drill-tested receivers a year, so expect about {t3_per_class} 2026 names. Today those men go about {t3_slots} slots earlier at their median pick of {t3_med_pick}, for {t3_soe} inches of extra separation ({t3_soe_lo} to {t3_soe_hi}; {t3_n_soe} with NFL routes). Pay for the 40, which does carry.
- **Scouts:** if the drill plant is not on the Sunday tape, grade the tape; Table 2 shows how to pull the clip.
- **Combine staff:** re-stem the curl, comeback and hook at NFL hitch entry speed (about {hitch_in_nfl} yd/s, a controlled three-quarter stem) and grade at least {cuts70} cuts per receiver. These data cannot test this proposal.
- **Position coaches:** rep breaks at game tempo. Nothing here says a harder full-speed plant buys separation.

## Limitations

- **Not blind.** All three classes were seen in earlier rounds of this work. The Break G definition, the ±3 inch margin and the pre-set forecast were frozen before these tests; the ridge row, the team bootstrap, the placebos and Section 4 came afterwards.
- **Power.** A true +3 inch effect would be detected {sim_pow3}% of the time (appendix). Harm down to {full_b_lo} inches per sd is not excluded.
- **Tracking is a sample.** Up to 40 NFL plays per player, half random and half top-EPA, with no defenders and no ball. Sections 1, 2, 4 and 5 depend on it; Section 3 uses full-season NGS separation. On random snaps only, drill vs NFL Break G is {rand_rho} ({rand_lo} to {rand_hi}; {rand_n} receivers).
- **Section 4 is descriptive.** Without defenders, "the cut made space" and "an open man ran through it" look alike.
- **Selection and outcome.** Only {n_soe} of {n_drill} drill-scored receivers logged NFL routes. SOE leaves out contested catches, yards after catch and earning targets, as the Nacua row shows.

## Appendix: method, reliability and power

A cut is the first turn of 30° or more after {def_path} yd of stem, entered above {def_entry} yd/s. Positions are Savitzky–Golay smoothed at 10 Hz, and Break G averages acceleration magnitude over ±{def_hw} s around the turn. A receiver's score averages his drills (z-scored within drill and year), shrunk toward the mean by between-drill reliability. NFL breaks are z-scored within route family and position. Route paths in Figures 1 and 4 are rotated frames and change no metric.

A single graded cut has reliability {rel_rep}. At the median of {med_cuts} cuts per receiver the score's reliability is {rel_med}; {cuts70} cuts would reach {rel70}. Simulating the whole pipeline with real noise and route weights, the slope averages {sim_mean3} when the truth is +3, so it is not biased toward zero, and its spread under no effect is {sim_sd} inches. True effects of +{sim_rule} inches or more are unlikely given the observed slope; {p_rope}% of resampled effects fall inside ±3 inches. Braking alone gives {brk_b} ({brk_lo} to {brk_hi}). `analysis.py` rebuilds every number, table and figure from `data/processed`.
