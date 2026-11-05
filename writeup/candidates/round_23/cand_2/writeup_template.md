# The Combine Grades a Faster Cut Than Sundays Ask For: Receiver Break G Does Not Buy Separation

**Receivers who cut harder in Combine route drills are not more open in the NFL. Across {n_soe} receivers from draft classes the model never saw, one standard deviation of drill Break G moved separation over expected by {full_b} inches (90% interval {full_b_lo} to {full_b_hi}).**

**Break G** is how hard a receiver plants: the average g-force he puts into the ground in the one second around his first real cut, braking in, turning and driving out. A typical NFL hitch break is {hitch_bg_nfl} g; the same break in the Combine hitch-family drills is {hitch_bg_drill} g. On film, read three speeds: into the cut, at the plant, out of it.

**Why it matters.** Route-breaking is the receiver skill the Combine films most directly, and clubs already pay for it: at the same 40, weight, NGS score and position, a receiver 1 sd higher in Break G is drafted {price}% earlier in pick number (90% interval {price_lo} to {price_hi}; {n_drill} receivers). If it does not buy separation, that is draft capital spent on nothing.

**Data.**
- **Combine:** {n_reps} tracked reps in ten route drills (2023–2025 classes), {n_cuts} with a gradable cut. Every rep is scored only against the same drill in the same year, because drill medians move between Combines.
- **NFL:** the same cut code run on {n_nfl_breaks} sampled regular-season breaks by {n_nfl_rec} receivers, grouped into the five route families the drills copy: slant, out, hitch (curl, comeback, hook), in (dagger) and corner (sail).
- **Outcome:** separation over expected (SOE), the inches a receiver is more open at the throw than Next Gen Stats separation predicts from alignment, cushion, position and season. It uses full regular seasons ({n_pseason} player-seasons, median {med_routes} career routes), not the tracking sample.

![Figure 1. Median speed through the first cut (line) and middle half of reps (band), Combine route drills vs sampled NFL snaps of the same route family, aligned on the plant. Combine cuts are entered faster in all {n_fam_faster} families.](figures/fig1_family_traces.png)

## 1. The drill asks for a different cut

Figure 1 is the whole story in one picture. In every family, the Combine receiver enters the cut faster than he will on Sundays. On the hitch family he arrives at {hitch_in_drill} yd/s (about {hitch_in_drill_mph} mph) and in the NFL at {hitch_in_nfl} yd/s (about {hitch_in_nfl_mph} mph). Only {share_nfl_at_drill}% of NFL breaks are entered as fast as the typical drill rep. Break G rises with entry speed (rank correlation {rho_bg_vin_drill} in drills, {rho_bg_vin_nfl} in games), so the drill rewards whoever can stop from a speed the NFL route rarely asks for.

Then we asked whether the same receiver ranks the same way in both settings, family by family and part by part (Figure 2A).

![Figure 2. A: rank correlation between a receiver's drill and NFL value for the same route family, by part of the cut ({pair_n} receiver-family pairs, {pair_rec} receivers, 2+ NFL breaks per family). B: receiver-level transfer against the 40-yard control.](figures/fig2_transfer.png)

- **Speed carries.** Official 40 rank vs NFL top speed: {f40_rho} (90% interval {f40_lo} to {f40_hi}; {f40_n} receivers).
- **The cut does not.** Drill Break G vs NFL Break G: {rec_rho} ({rec_lo} to {rec_hi}; {rec_n} receivers with 8+ sampled breaks). Single families range from {fam_bg_min} to {fam_bg_max}. Removing entry speed on both sides gives {res_rho}.
- **One part travels a little: braking.** How fast a receiver sheds speed into the plant transfers at {all_decel} (90% interval {all_decel_lo} to {all_decel_hi}), mostly on hitches and corners. It is one of 30 cells, so treat it as a lead, not a finding.

Noise alone does not explain the null. Given how noisy both measures are, a true correlation of {link} would still read about {true05}. The observed value sits below zero.

## 2. Break G adds nothing to a no-tracking forecast

The test was fixed before it was run: fit the forecast on two draft classes, score it on the third, rotate, and weight receivers by their routes. The ±3 inch margin was set in advance as the smallest effect a club would act on. We show three no-tracking baselines side by side so the null does not depend on one choice. The full one, with pick, official 40, weight, NGS score and a tight-end flag, was pre-specified. Intervals come from resampling whole receivers, refitting every held-out step inside each draw.

| No-tracking baseline | Held-out R² | R² change adding Break G [90%] | Break G effect, inches per sd [90%] |
|---|---|---|---|
| Pick only | {pick_r2} | {pick_dr2} [{pick_dr2_lo}, {pick_dr2_hi}] | {pick_b} [{pick_b_lo}, {pick_b_hi}] |
| Pick + role (TE, slot share) | {role_r2} | {role_dr2} [{role_dr2_lo}, {role_dr2_hi}] | {role_b} [{role_b_lo}, {role_b_hi}] |
| **Pick + 40, weight, NGS score, TE** (pre-specified) | {full_r2} | {full_dr2} [{full_dr2_lo}, {full_dr2_hi}] | **{full_b} [{full_b_lo}, {full_b_hi}]** |

*Table 1. {n_soe} receivers with a drill score and NFL separation. R² below zero means the forecast misses held-out classes by more than the class average does.*

No baseline forecasts SOE. Career SOE varies by {sd_soe_in} inches between receivers, and draft slot and testing explain none of it out of class. Adding Break G does not help either. {p_rope}% of resampled effects fall inside ±3 inches, and none reach +3. The official 3-cone ({cone_b} in per sd, {cone_n} receivers) and short shuttle ({shut_b}, {shut_n}) tell the same story without tracking. So does braking alone ({brk_b}, interval {brk_lo} to {brk_hi}).

**Is the test too weak to see a real effect?** Each graded cut is noisy (reliability {rel_rep} per cut). At today's median of {med_cuts} cuts per receiver, the score's reliability is {rel_med}. We simulated the whole pipeline: a true skill, a drill reading with the real noise for each receiver's number of drills, real SOE noise and route weights, and the same held-out test (Figure 3B). Because Break G is shrunk for its noise, the test is not biased toward zero; noise costs precision instead (simulated spread {sim_sd} in). If the true effect were +3 inches per sd, the test would detect it {sim_pow3}% of the time, and would report a slope as low as {full_b} only {sim_p3}% of the time. True effects of +{sim_rule} inches or more are unlikely. A harmful effect is not excluded.

![Figure 3. A: held-out Break G effect on career SOE under three baselines, and official agility times in its place (grey), against the ±3 inch margin. B: simulated slopes the test would report for true effects of 0–6 inches per sd ({n_soe} receivers, real noise); orange line = observed.](figures/fig3_baselines_sim.png)

## 3. Named receivers: the drill order does not survive

| Receiver (drill rank of {n_elig} WRs) | Combine: in → plant → out, yd/s (g) | NFL, same families: in → plant → out (g) | Career SOE, in | Coach's read |
|---|---|---|---|---|
| {top_name} (#{top_rank}) | {top_din} → {top_dap} → {top_dout} ({top_dg}) | {top_nin} → {top_nap} → {top_nout} ({top_ng}) | {top_soe} | Hardest drill planter, but on Sundays he sheds only {top_nmph} mph: he rounds the break. Rep sink-and-plant on curls at game tempo. |
| {mid_name} (#{mid_rank}) | {mid_din} → {mid_dap} → {mid_dout} ({mid_dg}) | {mid_nin} → {mid_nap} → {mid_nout} ({mid_ng}) | {mid_soe} | Mid-pack in drills, slowest of the three into his Sunday breaks ({mid_nin} yd/s): the defender sees it coming. Rep a faster stem into the same plant. |
| {bot_name} (#{bot_rank}) | {bot_din} → {bot_dap} → {bot_dout} ({bot_dg}) | {bot_nin} → {bot_nap} → {bot_nout} ({bot_ng}) | {bot_soe} | Softest drill cutter, ordinary Sunday cut, separation near expected. Nothing to fix from Combine tape. |

*Table 2. Medians over each man's reps in route families he ran in both settings ({top_nd}/{top_nn}, {mid_nd}/{mid_nn} and {bot_nd}/{bot_nn} drill/NFL breaks). Rank is Break G against all {n_elig} wide receivers with 200+ career routes and 5+ sampled breaks.*

Figure 4 widens the view to the eight hardest and eight softest drill cutters. Most NFL Break G and SOE intervals cross zero, so they are greyed. The hardest eight average {top8_soe} inches of SOE and the softest eight {bot8_soe}. Two cards show why separation is not the whole job. Arian Smith ranks #{smith_rank} and is {smith_soe} inches more open than expected over {smith_routes} routes, a vertical threat whom corners play off. Puka Nacua ranks #{nacua_rank} and sits at {nacua_soe} inches over {nacua_routes} routes: a high-volume target who wins in tight windows. Break G could not have told those two apart.

![Figure 4. Player cards: the 8 hardest and 8 softest drill cutters among {n_elig} eligible wide receivers, sorted by Combine Break G. Each row shows the same man's drill score, his NFL Break G from the same code, and career SOE, each with a 90% interval. Grey: interval crosses zero.](figures/fig4_cards.png)

## 4. What to do on Monday

- **Scouts and the draft room:** give the route-drill cut grade zero weight in the price. Today it moves a receiver {price}% up the board per sd and buys no separation. The 40 transfers; the cut grade does not.
- **Combine staff:** grade at least {cuts70} cuts per receiver (reliability {rel70}; today's median {med_cuts} gives {rel_med}). Re-stem the curl, comeback and hook at NFL hitch entry speed (about {hitch_in_nfl} yd/s, a controlled three-quarter stem). This is a design proposal; these data cannot test it.
- **Position coaches:** rep breaks at game tempo. Nothing here says a harder full-speed plant buys separation. Braking into the plant is the only part with any drill-to-Sunday carry-over, and only as a lead, so coach the sink at game speed.

## Limitations

- **Not blind.** All three classes were seen in earlier rounds of this work. The Break G definition, the ±3 inch margin and the pre-specified baseline were frozen before these tests and are unchanged here.
- **Power.** Detection is {sim_pow3}% at +3 inches. Harm down to {full_b_lo} inches per sd is not excluded.
- **Tracking is a sample.** Up to 40 NFL plays per player, {share_random}% random and the rest top-EPA, with no defenders or ball. Section 1 and the cards depend on it; Section 2 uses full-season NGS separation. Random snaps alone give a drill-to-NFL Break G correlation of {rand_rho} ({rand_lo} to {rand_hi}; {rand_n} receivers).
- **Selection.** {n_soe} of {n_drill} drill-scored receivers logged NFL routes. Receivers who never played cannot be tested.
- **SOE is one output.** It omits contested catches, yards after catch and target earning, as the Nacua card shows.

## Appendix: method in brief

A cut is the first turn of 30° or more after {def_path} yd of stem, entered above {def_entry} yd/s. Positions are Savitzky–Golay smoothed at 10 Hz, and Break G averages the acceleration magnitude over ±{def_hw} s around the turn. The receiver score averages his drills (z-scored within drill and year) and is shrunk toward the mean by the between-drill reliability. NFL breaks are z-scored within route family and position. The 40 enters only as the official time, so the tracked clip's {clip_offset} s start offset never touches Break G. `analysis.py` rebuilds every number, table and figure from `data/processed` in about two minutes.
