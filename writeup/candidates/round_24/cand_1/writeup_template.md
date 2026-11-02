# Run-the-Hoop Bend Points Toward NFL Pass Rush, but Not Yet Past Draft Order

Defensive linemen who lean harder through the Combine's Run-the-Hoop turn got more NFL pressure than their draft slot and role predicted: **+{{bs_pick}} pressures per 450 rushes for each standard deviation (sd) of "Hoop G" [90% interval {{bs_pick_lo}} to {{bs_pick_hi}}], n = {{n_an}}**, always scored on a draft class the model never saw. It does not yet choose better day-3 tape than pick order ({{tt_fch_k}} of {{tt_fch_n}} hits for both), so it is a film flag and a frozen 2026 test, not a board input.

**Why now.** The 2023–25 classes are the first tracked Combine classes with NFL snaps, so the drill can finally be checked against the same men's NFL rushes, and the 2026 class arrives as a clean test.

- **Scout, Monday:** flag every day-3 or undrafted lineman in his class's top third on Hoop G (about {{flags_per_class}} a class; {{flags_min}} to {{flags_max}} in 2023–25) and watch his rushes for a flat, tight corner before the board meets. The flag sets film order, never board order.
- **Cost of a miss:** in held-out classes {{fl_pct}}% of flagged men beat their forecast against {{rs_pct}}% of other day-3 men; a miss fell short by {{miss_short}} pressures a season. The flag never moves a name up, so a miss costs film time, not a pick.
- **Coach and Combine:** low Hoop G gets tight-circle and dip-and-rip reps off a live get-off; grade four hoop laps, not one.

## Hoop G in coach terms

Hoop G is how hard a lineman leans through the tightest half-second of the hoop's figure-eight: speed squared over turn radius, in g, ranked within his Combine class. The median lineman carries {{hoop_mph_median}} mph around a {{hoop_radius_median}}-yard radius ({{hoop_g_median}} g). His hardest lean on a sampled NFL rush comes at {{nfl_mph_median}} mph around {{nfl_radius_median}} yards ({{nfl_bend_median}} g).

On Sunday the tackle's set and hands push the arc out: a rusher's NFL turn is a median {{turn_match_median}} times wider than his own hoop turn (n = {{turn_match_n}}), so we compare men within a class, never raw g across drill and game. A pressured snap cost the offense {{epa_press}} expected points [{{epa_press_lo}}, {{epa_press_hi}}] on {{epa_n}} sampled pass plays; bend has no box-score column, and {{pct_day3_dl}}% of the {{n_dl}} linemen went on day 3 or undrafted, where tape is thinnest.

![Figure 1. Panel A overlays three men's Combine hoop laps; panels B–D show each man's randomly sampled NFL rushes on the same yard scale. Blue beat his forecast, vermillion fell below it, black was on par (n = {{h0_n}}, {{h1_n}} and {{h2_n}} sampled 2-s rushes).](figures/fig1_hero.png)

## Does it hold on a class the model never saw?

**Outcome:** career pressure rate through 2025 (shrunk toward the {{press_mean}}% mean) for linemen with 30+ rushes. **Test:** each class ({{ho_n_per_class}} rushers) is forecast by a model fitted on the other two; drill measures are standardized within Combine year. **Comparator, fixed before this round:** draft pick plus edge-or-interior role, the forecast frozen for the 2026 test.

| Forecast without tracking (n = {{n_an}}) | Forecast R² | + Hoop G R² | R² gain, resample rushers [90%] | R² gain, resample NFL teams [90%] | Hoop G, pressures per 450 rushes per sd |
|---|---|---|---|---|---|
| Pick only | {{r2_pick1}} | {{r2_pick1_h}} | {{dr2_pick1_h}} [{{dr2_pick1_h_lo}}, {{dr2_pick1_h_hi}}] | [{{cdr2_pick1_lo}}, {{cdr2_pick1_hi}}] | +{{bs_pick1}} [{{bs_pick1_lo}}, {{bs_pick1_hi}}] |
| **Pick + role (headline)** | {{r2_pick}} | {{r2_pick_h}} | {{dr2_pick_h}} [{{dr2_pick_h_lo}}, {{dr2_pick_h_hi}}] | [{{cdr2_pick_lo}}, {{cdr2_pick_hi}}] | **+{{bs_pick}} [{{bs_pick_lo}}, {{bs_pick_hi}}]** |
| Everything, shrunk: pick, role, weight, NGS score, 10-yd split, 3-cone | {{r2_ridge}} | {{r2_ridge_h}} | {{dr2_ridge_h}} [{{dr2_ridge_h_lo}}, {{dr2_ridge_h_hi}}] | [{{cdr2_ridge_lo}}, {{cdr2_ridge_hi}}] | +{{bs_ridge}} [{{bs_ridge_lo}}, {{bs_ridge_hi}}] |

*Table 2. In plain terms: whatever the forecast, a rusher one sd above his class on Hoop G beat it by 2 to 3 pressures a season, clear of zero; but the share of rusher-to-rusher spread explained rises only slightly and could be zero. "Shrunk" pulls coefficients toward zero by an amount chosen inside the training classes.*

Checks, one line each:

- **Placebo (Figure 2A):** Hoop G shuffled among linemen of the same class and role {{n_perm}} times; the real slope beats {{pl_cr_pct}}% of shuffles (p = {{pl_cr_p}}), whose top 5% reach {{pl_cr_p95}}.
- **Class by class:** {{ho_hoop_2023}}, {{ho_hoop_2024}}, {{ho_hoop_2025}}; 2025 is flat, not reversed ({{cls2025_lo}} to {{cls2025_hi}}).
- **Both turns of the lap averaged** instead of the harder one: +{{mt_hoop}} [{{mt_hoop_lo}}, {{mt_hoop_hi}}], smaller but still positive.
- **One-lap noise:** {{pct_one_run}}% ran one lap, which repeats at only {{rel_turn_lap}}; corrected for that noise the slope would be about +{{me_true}} (range {{me_lo}} to {{me_hi}}).
- **Independent script:** a re-implementation sharing no code gives +{{v_slope}} [{{v_lo}}, {{v_hi}}] on the same {{v_n}} rushers.

## From drill to Sunday: where the signal travels

| Link (holding weight and edge share fixed) | Rank correlation [90%] | n |
|---|---|---|
| Hoop G → NFL bend | {{tr_hoop_g_bend}} [{{tr_hoop_g_bend_lo}}, {{tr_hoop_g_bend_hi}}] | {{tr_hoop_g_bend_n}} |
| Hoop turn radius → NFL turn radius | {{tr_rad}} [{{tr_rad_lo}}, {{tr_rad_hi}}] | {{tr_rad_n}} |
| NFL bend → pressure rate | {{gp_g_bend}} [{{gp_g_bend_lo}}, {{gp_g_bend_hi}}] | {{gp_g_bend_n}} |

*Table 1. NFL bend = hardest half-second of lean in the first 2 s, averaged over a median {{median_random_rushes}} randomly sampled rushes per lineman.*

The two halves do not multiply into the headline. NFL bend does turn into pressure, but Hoop G barely predicts NFL bend. On the same {{med_n}} rushers, only +{{med_ind}} [{{med_ind_lo}}, {{med_ind_hi}}] of the +{{med_tot}} runs through measured NFL bend ({{med_share}}%); +{{med_dir}} does not (Figure 2B). Hoop G is not a proxy for the arc we measure on Sunday; it tracks something the sampled 2-s paths miss, such as balance while finishing, counter moves, or rushes we did not sample.

The corner turn still shows on Sunday. Call a sampled rush a **corner turn** when its lean is in the top quarter of all sampled rushes (at least {{ct_g}} g). Linemen in their class's top third on Hoop G turned the corner on {{ct_top}}% of rushes, the bottom third on {{ct_bot}}% (gap {{ct_gap}} points [{{ct_gap_lo}}, {{ct_gap_hi}}], n = {{ct_n}}); corner-turn rushes were pressured {{ct_press_corner}}% of the time against {{ct_press_other}}% ({{ct_rush_n}} rushes).

![Figure 2. A: the Hoop G slope when Hoop G is shuffled among linemen of the same class and role ({{n_perm}} shuffles, n = {{n_an}}) against the real slope. B: the slope split into the part that runs through measured NFL bend and the rest (n = {{med_n}}), with 90% intervals.](figures/fig2_placebo.png)

## Day 3: a film queue, not a board

Among {{d3_n}} day-3 and undrafted rushers in held-out classes, the top third by forecast plus Hoop G beat its class's day-3 median {{tt_fch_k}} of {{tt_fch_n}} times; pick order's top third also went {{tt_pick_k}} of {{tt_pick_n}}, and the forecast alone {{tt_fc_k}} of {{tt_fc_n}}. The gap, {{hg_gap}} points [{{hg_gap_lo}}, {{hg_gap_hi}}], misses the +{{hg_bar}} we required for board use. **Hoop G has not shown it picks better day-3 tape; it is a reason to open tape.**

The {{n_cards}} flagged 2024–25 day-3 rushers with 100+ NFL rushes, in Hoop G order (pressures a season against the pick + role forecast):

**Open first** (beat the forecast by the {{margin_season}} margin):

{{T:film_open}}

**Flag missed** (fell below it):

{{T:film_miss}}

![Figure 3. The {{n_cards}} flagged day-3 rushers, sorted by Hoop G, with class rank, NFL bend and corner turns ({{cards_min_sampled}}–{{cards_max_sampled}} sampled rushes each) beside the NFL result and its 90% interval.](figures/fig3_film_list.png)

**Play-log check (not film).** With no video or blockers, we read every sampled snap of the top hit and the miss with the most corner turns.

- **{{T:plog_hit}}** lined up on the edge on {{plog_hit_edge}} of {{plog_hit_n}} snaps and turned the corner on {{plog_hit_nc}} of {{plog_hit_n2}} 2-s rushes, but only {{plog_hit_np}} sampled snap was pressured (at his {{plog_hit_rate}}% career rate, {{plog_hit_exp}} were expected). Tape must show whether his pressures come off the corner.
- **{{T:plog_miss}}** lined up on the edge on {{plog_miss_edge}} of {{plog_miss_n}} snaps and turned the corner on {{plog_miss_nc}} of {{plog_miss_n2}} rushes, yet {{plog_miss_cmp_c}} of those corner turns ended in a completed pass and {{plog_miss_npc}} in pressure. His median NGS get-off on them was {{plog_miss_go_c}} s, close to the {{plog_getoff_all}} s of all sampled rushes, so the first step is not obviously late. Tape question: does he reach the corner after a quick throw, or fail to finish?

## The one action: a frozen 2026 test

Stored in `frozen_rule.json` (sha256 `{{T:hash}}…`): grade the 2026 class's hoop laps; after Week 18, regress pressure rate minus the pick-plus-role forecast frozen on 2023–25 on Hoop G. Positive at one-sided p < 0.10 keeps the flag; zero or below retires it. Hoop G reaches the board only if its day-3 top third beats pick order by 20+ points over two classes.

## Limitations

- **Small samples, single reps.** {{n_an}} rushers are in the model and {{d3_n}} on day 3; most ran one hoop lap.
- **Reused classes.** Earlier rounds screened ideas on these classes; 2026 is the first clean test.
- **Thin NFL tracking, no blockers or film.** Corner turns and pressure may both follow from beating the tackle; the play-log reads are ours, from {{cards_min_sampled}}–{{cards_max_sampled}} sampled rushes each.
- **Missed time.** Rates are per rush; the {{n_short_career}} linemen under 30 rushes are left out.

## Bottom line

Hoop G is a small, steady signal worth a look at day-3 edge tape, not a board input. Grade four laps and let the 2026 test decide.

## Appendix

- **Other comparators.** The unshrunk full forecast overfits (held-out R² {{r2_base}}; Hoop G +{{bs_base}} [{{bs_base_lo}}, {{bs_base_hi}}]). A forecast chosen inside each training fold gives +{{bs_nest}} [{{bs_nest_lo}}, {{bs_nest_hi}}] ({{T:nest_list}}). Team resampling ({{n_teams}} clubs) gives a slope interval of {{cbs_lo}} to {{cbs_hi}}.
- **Six drill measures (Figure A1).** Hoop G is one of six pre-listed measures; p = {{pl_c_p}} alone and {{ho_hoop_pfw}} as the best of six, where six shuffled drills reach ±{{fw95_season}} 5% of the time. Pass-rush-drill burst is flat.
- **EPA trace.** {{epa_rows}} sampled rusher-rows ({{epa_rows_press}} pressured) cover {{epa_n}} plays: {{epa_dup_plays}} plays had two sampled linemen, and one row per play keeps {{epa_n_press}} pressured. Flagging a play if either lineman pressured gives {{epa_any_press}} and {{epa_press_any}} EPA; counting rows gives {{epa_press_rows}}.
- **Multiplicity.** Of {{n_tests}} tests, {{n_holm}} survive a strict family-wise correction at {{q_fdr}} (both NFL trait-to-pressure links) and {{n_bh}} survive a false-discovery correction (`multiplicity.csv`).
- **Variants.** Priors and rush minimums: {{sens_slope_min}} to {{sens_slope_max}}; dropping one rusher at a time: {{lopo_min}} to {{lopo_max}}; 200 random three-way splits: {{rsplit_lo}} to {{rsplit_hi}}; edge +{{role_EDGE}} (n = {{role_EDGE_n}}), interior +{{role_interior}} (n = {{role_interior_n}}).
- **Noise and power.** Simulated, one lap shows about {{me_att}}% of a true slope. The 2026 test has {{pws_obs_1}}% power for one class and {{pws_obs_3}}% for three at the observed slope.

![Figure A1. The six pre-listed drill measures against the pick + role forecast (n = {{n_an}}), with 90% intervals, held-out classes and the best-of-six chance line.](figures/figA1_screen.png)

`analysis.py` (about 8 minutes) rebuilds every number, figure and this text from `data/processed`, and runs `verify.py`.
