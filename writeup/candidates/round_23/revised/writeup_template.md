# Run-the-Hoop Bend Points Toward NFL Pass Rush, but Not Past Draft Order

Defensive linemen who lean harder through the Combine's Run-the-Hoop turn got +{{bs_pick}} pressures per 450 rushes per sd [90%: {{bs_pick_lo}}, {{bs_pick_hi}}] more than their pick and role predicted, in draft classes the model never saw (n = {{n_an}}). It does not pick day-3 tape better than draft order ({{tt_fch_k}} of {{tt_fch_n}} hits against {{tt_pick_k}} of {{tt_pick_n}}), so it is a film flag and a frozen 2026 test, not a board input.

**Why now.** Combine tracking began with the 2023 class, which now has three NFL seasons of tracked rushes: the first chance to test a drill within one class against the same men on Sunday.

**What to do on Monday**

- **Scout (fixed rule):** flag a day-3 or undrafted rusher (pick 103+) in the top third of his Combine class on Hoop G. That flagged {{rule_mean}} men a class in 2023–25 ({{rule_min}} to {{rule_max}}, of about {{rule_hoop_per_class}} hoop runners); expect as many in 2026. Queue them beside pick order; on tape, check whether he flattens at the top of the rush at speed.
- **Cost of a miss:** a false flag costs a film session ({{flag_false_n}} of {{fl_n}} held-out flags fell short of forecast by the margin). A missed hit costs {{miss_over}} pressures a season, about {{miss_epa}} expected points ({{miss_n}} unflagged day-3 rushers beat forecast by the margin). Open tape freely; never move the board.
- **Coach:** low Hoop G gets tight-circle and dip-and-rip reps off a live get-off. **Combine:** grade four hoop laps, not one.

## Hoop G in coach terms

Hoop G is how hard a lineman leans through the tightest half-second of the hoop's figure-eight: speed squared over turn radius, in g, ranked within his Combine class. The median lineman carries {{hoop_mph_median}} mph around a {{hoop_radius_median}}-yard radius, which is {{hoop_g_median}} g. His hardest lean on a sampled NFL rush comes at {{nfl_mph_median}} mph around {{nfl_radius_median}} yards, which is {{nfl_bend_median}} g.

On Sunday the tackle pushes the arc out: a rusher's NFL turn is a median {{turn_match_median}} times wider than his hoop turn (n = {{turn_match_n}}), and only {{pct_rush_hoopspeed}}% of sampled rushes reach his hoop speed at the lean (Figure 1B). So we compare men within a class, never raw g.

**Why the hoop.** The pass-rush drill times get-off, which NGS already measures every snap, and one burst rep repeats at only {{rel_burst1}}. Bend has no box-score column, and a pressured snap cost the offense {{epa_press}} expected points [{{epa_press_lo}}, {{epa_press_hi}}] on {{epa_n}} randomly sampled pass plays.

**Why day 3.** {{pct_day3_dl}}% of the {{n_dl}} linemen in the 2023–25 classes went on day 3 or undrafted, where tape is thinnest.

![Figure 1. A: three men's Combine hoop laps overlaid. B: hardest lean, Combine vs median sampled NFL rush, for {{fig1_n}} rushers; dotted curves are constant g. C: each man's randomly sampled NFL rushes ({{h0_n}} each).](figures/fig1_hero.png)

## Does the bend show up on Sunday?

A sampled rush is a **corner turn** when its hardest half-second of lean in the first 2 s is in the top quarter of all sampled rushes (at least {{ct_g}} g). Linemen in their class's top third on Hoop G turned the corner on {{ct_top}}% of rushes, the bottom third on {{ct_bot}}%: a gap of {{ct_gap}} points [{{ct_gap_lo}}, {{ct_gap_hi}}], n = {{ct_n}}. Corner-turn rushes ended in pressure {{ct_press_corner}}% of the time, other rushes {{ct_press_other}}%.

| Link | Rank correlation [90%] | n |
|---|---|---|
| Hoop G → NFL bend | {{tr_hoop_g_bend}} [{{tr_hoop_g_bend_lo}}, {{tr_hoop_g_bend_hi}}] | {{tr_hoop_g_bend_n}} |
| Hoop turn radius → NFL turn radius | {{tr_rad}} [{{tr_rad_lo}}, {{tr_rad_hi}}] | {{tr_rad_n}} |
| NFL bend → pressure rate | {{gp_g_bend}} [{{gp_g_bend_lo}}, {{gp_g_bend_hi}}] | {{gp_g_bend_n}} |
| NGS get-off → pressure rate | {{gp_quick}} [{{gp_quick_lo}}, {{gp_quick_hi}}] | {{gp_quick_n}} |

*Table 1. Rank correlation after adjusting for weight and edge share. NFL movement: a median {{median_random_rushes}} randomly sampled rushes per lineman.*

NFL bend goes with pressure, but Hoop G barely predicts NFL bend. Fitting both steps on two classes and applying them to the third, only {{md_share}}% [{{md_share_lo}}, {{md_share_hi}}] of the Hoop G slope runs through measured NFL bend. "Bend" names the drill; it is not a proven mechanism.

## Held out by class

**Outcome:** career pressure rate through 2025, adjusted for small samples (30+ rushes). Each class ({{ho_n_per_class}} rushers) is forecast by a model fitted on the other two. **Baseline:** pick plus role (edge share), the forecast frozen for the 2026 test before Hoop G was examined. The smallest effect worth acting on, set before fitting, is +{{margin_season}} pressures per 450 rushes per sd.

**Hoop G adds about +{{bs_pick}} pressures per 450 rushes per sd [{{bs_pick_lo}}, {{bs_pick_hi}}]; the interval excludes zero, but the R² gain, {{dr2_pick_h}} [{{dr2_pick_h_lo}}, {{dr2_pick_h_hi}}], does not.** A ridge-shrunk full no-tracking baseline gives +{{bs_ridge}} [{{bs_ridge_lo}}, {{bs_ridge_hi}}]; other baselines are in the appendix.

**Placebo.** We shuffled the whole drill table within class: six drill measures plus five other ways to build Hoop G (averaging both turns; 0.3 s or 0.7 s windows). The best of those eleven reaches +{{pl_max95}} in 5% of shuffles, the line Hoop G sits on (Figure 2). Its construction (harder turn, 0.5 s) was fixed beforehand; averaging both turns gives +{{mt_hoop}} [{{mt_hoop_lo}}, {{mt_hoop_hi}}].

![Figure 2. Six drill measures against the pick + role forecast, n = {{n_an}} rushers in three held-out classes ({{ho_n_per_class}}). Bars: 90% intervals; diamonds: each class; dashed lines: what the best of eleven shuffled measures reaches 5% of the time.](figures/fig3_screen.png)

- **Class by class:** {{ho_hoop_2023}}, {{ho_hoop_2024}}, {{ho_hoop_2025}}; 2025 is flat, not reversed.
- **Edge vs interior:** +{{role_EDGE}} (n = {{role_EDGE_n}}) vs +{{role_interior}} (n = {{role_interior_n}}).
- **One-lap noise:** one lap repeats at {{rel_turn_lap}} and shows about {{me_att}}% of a true slope; corrected, about +{{me_true_obs}} (secondary only; appendix Figure 4).

**Negative control, not clean.** The same pipeline finds Hoop G also predicts *fewer* run-snap tackles ({{nc_sd}} sd [{{nc_sd_lo}}, {{nc_sd_hi}}], n = {{nc_n}}; pressure: {{nc_press_sd}} sd). Weight does not explain it ({{ncw_sd}} after adding weight). Hoop G may partly mark a pass-rush specialist that pick and role miss.

## Day 3: a film queue, not a board

Among {{d3_n}} held-out day-3 and undrafted rushers, the top third by forecast plus Hoop G beat its class's day-3 median {{tt_fch_k}} of {{tt_fch_n}} times; pick order's top third, {{tt_pick_k}} of {{tt_pick_n}}. The gap, {{hg_gap}} points [{{hg_gap_lo}}, {{hg_gap_hi}}], misses the +{{hg_bar}} we required for board use. Flagged men beat their own forecast {{fl_pct}}% of the time ({{fl_k}} of {{fl_n}}), the rest {{rs_pct}}%.

The {{n_cards}} flagged 2024–25 day-3 rushers with 100+ NFL rushes, in Hoop G order (pressures a season against forecast):

**Open first** (beat the forecast by the margin):

{{T:film_open}}

**Flag missed** (fell below it):

{{T:film_miss}}

![Figure 3. Player cards for the eight flagged day-3 rushers, sorted by Hoop G: rank in his Combine class, his sampled NFL rushes, and his NFL result with a 90% interval (grey = overlaps the action margin).](figures/fig4_cards.png)

**Tape pull.** We have not watched these plays; tracking has no blockers, so each clip must show the tackle's set and whether he won.

| Rusher (vs forecast) | Clip | Alignment, situation | NGS get-off | Hardest lean | Pressure |
|---|---|---|---|---|---|
{{T:tape_rows}}

*Table 3. Each man's hardest-bending randomly sampled rush.*

Solomon's lean is a speed turn ({{tp0_mph}} mph, {{tp0_depth}} yd upfield), the hit to confirm. Harrell's comes near a stop ({{tp2_mph}} mph; typical {{tp2_medmph}}): tape should show whether it is a counter after the tackle stopped him, a reason the drill cannot see.

## The one action: a frozen 2026 test

Stored in `frozen_rule.json` (sha256 `{{T:hash}}…`): grade the 2026 class's hoop laps; after Week 18, regress pressure rate minus the frozen pick-plus-role forecast on Hoop G. Positive at one-sided p < 0.10 keeps the flag; zero or below retires it. Hoop G reaches the board only if its day-3 top third beats pick order by 20+ points over two classes.

## Limitations

- **Small samples, single reps.** {{n_an}} rushers in the model, {{d3_n}} on day 3; {{pct_one_run}}% ran one hoop lap.
- **No film watched, no blockers.** Corner turns and pressure may both follow from beating the tackle; Table 3 is a pull list, not a verdict.
- **Mechanism unproven.** Little of the slope runs through measured NFL bend, and the negative control is not clean.
- **Reused classes.** Earlier rounds screened ideas on these classes; 2026 is the first clean test.
- **Missed time.** {{n_short_career}} linemen under 30 rushes are left out.

## Appendix

| Baseline without tracking (n = {{n_an}}) | Baseline R² | + Hoop G R² | R² gain [90%] | Hoop G slope [90%] |
|---|---|---|---|---|
| Pick only | {{r2_pick1}} | {{r2_pick1_h}} | {{dr2_pick1_h}} [{{dr2_pick1_h_lo}}, {{dr2_pick1_h_hi}}] | +{{bs_pick1}} [{{bs_pick1_lo}}, {{bs_pick1_hi}}] |
| **Pick + role (headline)** | {{r2_pick}} | {{r2_pick_h}} | {{dr2_pick_h}} [{{dr2_pick_h_lo}}, {{dr2_pick_h_hi}}] | **+{{bs_pick}} [{{bs_pick_lo}}, {{bs_pick_hi}}]** |
| Ridge-shrunk full | {{r2_ridge}} | {{r2_ridge_h}} | {{dr2_ridge_h}} [{{dr2_ridge_h_lo}}, {{dr2_ridge_h_hi}}] | +{{bs_ridge}} [{{bs_ridge_lo}}, {{bs_ridge_hi}}] |
| Full OLS: pick, role, weight, NGS score, split, 3-cone | {{r2_base}} | {{r2_base_h}} | {{dr2_base_h}} [{{dr2_base_h_lo}}, {{dr2_base_h_hi}}] | +{{bs_base}} [{{bs_base_lo}}, {{bs_base_hi}}] |
| Chosen inside each training fold | {{r2_nest}} | {{r2_nest_h}} | {{dr2_nest_h}} [{{dr2_nest_h_lo}}, {{dr2_nest_h_hi}}] | +{{bs_nest}} [{{bs_nest_lo}}, {{bs_nest_hi}}] |

*Table 2. Every baseline finds a similar slope; none finds an R² gain clear of zero. Slopes are pressures per 450 rushes per sd; intervals resample linemen within class and refit every fold. The fold-chosen baseline picked {{T:nest_list}}.*

![Figure 4. A: reliability of a lineman's drill average by number of reps (90% bands); ticks at right are NFL-side reliabilities. B: the slope a study of {{n_an}} linemen would see for each true slope, with one lap and with four.](figures/fig2_reliability.png)

- **Mediation.** Hoop G → NFL bend {{md_a}} sd per sd [{{md_a_lo}}, {{md_a_hi}}]; NFL bend → pressure {{md_b}} [{{md_b_lo}}, {{md_b_hi}}]; indirect {{md_ind}}, direct {{md_dir}} [{{md_dir_lo}}, {{md_dir_hi}}] (n = {{md_n}}).
- **Noise model.** Simulating true slopes through one-lap noise, true slopes {{me_lo}} to {{me_hi}} fit the observed value. 2026 power at the observed slope: {{pws_obs_1}}% for one class, {{pws_obs_3}}% for three.
- **EPA trace.** {{epa_rows}} rusher rows ({{epa_rows_press}} credited with a pressure or sack) cover {{epa_n}} plays; {{epa_shared}} plays had two cohort rushers. A play is pressured if any rusher was credited ({{epa_n_press}}); keeping the first row per play, as round 22 did, gave {{epa_first_press}}.
- **Multiplicity.** Of {{n_tests}} tests, {{n_holm}} survive Holm and {{n_bh}} Benjamini–Hochberg at 0.10.
- **Variants.** Other Hoop G builds {{var_min}} to {{var_max}}; priors and rush minimums {{sens_slope_min}} to {{sens_slope_max}}; dropping one rusher {{lopo_min}} to {{lopo_max}}. `verify.py`, sharing no code with `analysis.py`, rebuilds the same {{v_n}} rushers: +{{v_slope}} [{{v_lo}}, {{v_hi}}].
