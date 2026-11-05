# Day-3 Pass Rushers Who Stay Fast Through the Combine's Tightest Turn Get About 2 More Pressures a Season

Across {{n_an}} defensive linemen from the 2023–25 classes, each standard deviation (sd) of **Hoop G** was worth about {{bs_pick}} more pressures a season than draft slot and role predicted, measured on classes the model never saw. Hoop G is how fast a man stays while turning tight on the Run-the-Hoop drill, compared with his Combine class. The action: flag each class's top-third day-3 and undrafted linemen on Hoop G and open their tape first.

- **Scout, Monday:** that is about {{flags_per_class}} names a class ({{flags_2026_p10}}–{{flags_2026_p90}}). Table 3 gives each name a tape cue and a clip to pull.
- **2026 list:** the 2026 class's Combine tracking is not in this release. The frozen rule (`frozen_rule.json`, hash {{T:hash}}) prints that list the day the tracking is loaded. Those rookies are already on rosters, so the list orders practice-squad and waiver film, not the draft board.
- **Combine:** grade four hoop laps, not one. A single lap repeats at only {{rel_turn_lap}}.

The evidence is weak: the 2025 class alone was flat, and the no-selection version of the test has an interval that spans zero.

**Why day 3.** {{pct_day3_dl}}% of these {{n_dl}} linemen went on day 3 or undrafted, where tape is thinnest and clubs find rotational edge rushers. Bend is graded only off tape, and the hoop is the one Combine drill that times it.

## Why the hoop should matter on a rush

An edge rusher who reaches the tackle's outside shoulder has to turn toward the quarterback without slowing down. If he brakes to turn, the tackle recovers and rides him past the pocket. The hoop asks for that turn with no blocker in the way.

*Note on units.* Lean is speed squared over turn radius, in g. On the hoop the median lineman carries {{hoop_mph_median}} mph around a {{hoop_radius_median}}-yard radius ({{hoop_g_median}} g). On an NFL rush it is {{nfl_mph_median}} mph around {{nfl_radius_median}} yards ({{nfl_bend_median}} g), so men are compared within their class, never in raw g.

![Figure 1. Top: three of the day-3 names the flag adds (film-queue #1, the furthest above forecast, the furthest below), each with his hoop lap and every randomly sampled NFL rush on one yard scale, plus his hoop and NFL turn radii as bars. Bottom: all seven added names, each NFL rush's hardest lean against his hoop lean.](figures/fig1_hero.png)

**One rep, read off tracking.** {{T:fr_name}}, {{T:fr_when}}, {{T:fr_down_txt}}-and-{{fr_ytg}}, aligned {{T:fr_align}} with {{T:fr_cov}} coverage behind him. He is off the ball in {{fr_getoff}} s (lineman median {{getoff_med_all}}) and reaches {{fr_mph}} mph. When the pressure is logged at {{fr_ttp}} s he is {{fr_depth}} yards upfield and {{fr_lat}} yards across. The play log: "{{T:fr_desc}}" His hardest lean on that rep is only {{fr_bend}} g, around a {{fr_rad}}-yard arc. That is a speed rush, not a bend, which is why his Table 3 cue is the get-off.

## Does it hold on a class the model never saw?

**Outcome:** career pressure rate through 2025 for linemen with 30+ rushes, shrunk toward the {{press_mean}}% mean. **Test:** each class ({{ho_n_per_class}} rushers) is forecast by a model fitted on the other two, with drill measures standardized within Combine year.

| The frozen test (n = {{n_an}}) | R², pick + role | R², + Hoop G | Hoop G per sd [95%] | Shuffle p |
|---|---|---|---|---|
| **Hoop G vs a pick + role forecast** | {{r2_pick}} | {{r2_pick_h}} | **+{{bs_pick}} [{{bs_pick_lo95}}, {{bs_pick_hi95}}]** | {{pl_cr_p}} |

*Table 1. Pressures per 450-rush season. R² = share of rusher-to-rusher spread explained on held-out classes. Shuffle: Hoop G permuted within class and role.*

Hoop G was the best of {{n_family}} drill measures we tried. With no choosing, all six go into one shrunk (ridge) model, with the penalty set inside the training classes. Its held-out R² gain is {{r6_gain}} [95%: {{r6_gain_lo95}}, {{r6_gain_hi95}}] (shuffle p = {{r6_p}}), and {{r6_hoop_share}}% of its weight lands on the three hoop measures. Read plainly: the hoop drill carries what signal there is, but it is weak.

Figure 2 puts every check on one scale. The slope holds with wave-drill lean in the model, with the {{n_teams}} NFL clubs resampled, and with stance, snap load and club pass rush added to the forecast. The weak rows are the 2025 class alone (+{{pro_s25}} [{{pro_s25_lo95}}, {{pro_s25_hi95}}]) and the 2026 forecast. Spread between classes (sd {{het_tau}}) gives a 2026 slope of {{het_new_med}} with a 95% prediction interval of [{{het_new_lo}}, {{het_new_hi}}]. The chance its true slope is above zero is {{het_true_pos}}%.

![Figure 2. A: every drill measure and the two no-choosing composites. B: the frozen Hoop G test and its stress tests, with weak rows dashed. 95% intervals, n per row. The grey band marks effects under ±2.25 pressures a season (0.5 rate points per sd), the smallest effect we set in advance as worth acting on.](figures/fig2_tests.png)

## Drill to Sunday: the hoop marks who finishes the bend

The hoop does not predict how often a man bends on Sunday: Hoop G's rank correlation with his average NFL rush lean is {{tr_hoop_g_bend}}. So we fit one joint model of pressure. It combines career pressure counts with {{jm_n_rush}} randomly sampled rushes (median {{mm_rush_med}} a man), and each rusher gets his own random intercept. Every rush carries the defence's man or zone coverage and a long-yardage flag. Hoop G is corrected for its {{rel_turn_lap}} single-lap reliability. Each rush's lean is measured against the same man's other rushes, which takes out his size, role and usual blockers.

| Joint model term (95%) | Estimate |
|---|---|
| Rush where he leans 1 of his own sd harder | +{{jm_wz}} points of pressure [{{jm_wz_lo}}, {{jm_wz_hi}}] |
| … extra for each sd of Hoop G | +{{jm_int}} points [{{jm_int_lo}}, {{jm_int_hi}}] |
| Hoop G, career pressures a season, noise-corrected | +{{jm_hoop}} [{{jm_hoop_lo}}, {{jm_hoop_hi}}] (uncorrected {{jm_hoop_raw}}) |

*Table 2. Joint hierarchical model, {{jm_n_men}} rushers, rusher random intercept.*

In scout terms, high-Hoop-G men do not turn the corner more often, but when they do they keep enough speed to finish. On their hardest-lean quarter of rushes, the top third on Hoop G were pressured {{f4_top_q4}}% of the time, against {{f4_rest_q4}}% for the rest (Figure 3). That quarter is only {{f4q_top_3}} rushes, and the interaction's interval reaches zero. Modelled on raw counts and corrected for lap noise, the career effect is about three times the headline, but on held-out classes the joint model adds only {{jm_gain}} R² [{{jm_gain_lo95}}, {{jm_gain_hi95}}]. The headline's {{bs_pick}} is therefore the conservative read.

![Figure 3. Pressure rate by how hard he leaned on that rush relative to his own rushes, Hoop G top third vs the rest, with rush counts per quarter (A), and the joint model's rush-level terms (B).](figures/fig3_mechanism.png)

## Day 3: a film queue, not a board

We scored all {{d3_n}} day-3 and undrafted rushers in held-out classes. As a placebo, Hoop G was shuffled within class and role, and everything was refitted each time:

- **Ordering:** within-class rank correlation with pressure is {{dr_fch}} [95%: {{rho95_fch_lo}}, {{rho95_fch_hi}}] for forecast + Hoop G, against {{dr_pick}} for pick order (shuffle p = {{pl3_rho_p}}). By class, the gain is +{{pl3_rho_2023}}, {{pl3_rho_2024}} and +{{pl3_rho_2025}}, so one class of three went the wrong way.
- **Calling hits** (did he beat his class's day-3 median?): Brier score {{bri_fch}} against {{bri_pick}} for pick order (shuffle p = {{pl3_bri_p}}; a coin scores {{bri_coin}}, lower is better).
- **Top third:** both lists hit {{tt_fch_k}} of {{tt_fch_n}}. Hoop G adds {{ov_add}} names that pick order skips ({{ov_add_hit}} hit).

Hoop G orders day-3 tape a little better than pick order. It does not find more hits, so it decides **which tape opens first**. Table 3 is the queue of the added names.

| Added name, film order | Hoop G (sd) | Pressures a season vs forecast | Get-off (s) | What to look for, and the clip to pull |
|---|---|---|---|---|
{{T:t7_rows}}

*Table 3. The rules are fixed in code and read from sampled snaps, not watched film. A hard bend is a rush in the top quarter of all sampled rushes for lean. Two or more hard bends in eight rushes means the cue is a bend cue; otherwise get-off against the {{getoff_med_all}} s median and alignment pick it. The clip is his first pressured random snap.*

![Figure 4. Day-3 evaluation on all held-out day-3 rushers (A, B) and the added names with 95% intervals and their tape cue (C).](figures/fig4_day3.png)

{{T:t7_best}} ({{T:t7_best_pick}}) is the strongest case at {{t7_best_ov}} pressures a season over forecast, and his clip should show whether the hands or the bend win. {{T:t7_worst}} is the weakest at {{t7_worst_ov}}, with a {{t7_worst_go}} s get-off, so he is the lowest priority.

## The 2026 test

The rule was written down before any 2026 NFL data. After Week 18 of the 2026 season, check whether Hoop G still predicts pressure beyond the frozen pick + role forecast. A positive slope at one-sided p < 0.10 keeps the flag, and zero or below retires it. One class detects a real slope this size {{pws_obs_1}}% of the time, and three classes {{pws_obs_3}}%. Hoop G reaches the board only if its day-3 top third beats pick order by {{hg_bar}}+ points over two classes.

## Limitations

- **Weak evidence.** The no-selection gain and the 2026 prediction interval both span zero, 2025 was flat, and 2024's day-3 ranking went the wrong way.
- **Small samples, single reps.** There are {{n_an}} rushers in the model and {{d3_n}} on day 3, and most ran one hoop lap.
- **Forking paths.** Of {{n_tests}} tests, {{n_holm}} survive a family-wise correction at 0.10 and {{n_bh}} a false-discovery correction (Table A1).
- **No scheme context.** Sampled NFL tracking has no blockers, ball, stunts or opponent quality. Coverage enters only as a man/zone flag on {{jm_n_rush}} rushes, and stance is the roster listing.
- **Missed time.** Rates are per rush, and the {{n_short_career}} linemen under 30 rushes are left out.

## Bottom line

The hoop picks out day-3 rushers whose hardest bends end in pressure, worth about 2 pressures a season on weak evidence. It orders film and does not set the board. Grade four laps, and let the 2026 class decide.

## Appendix

| Check (n = {{n_an}} unless stated) | Estimate | 95% interval |
|---|---|---|
| Measure re-chosen in each fold: R² gain (Hoop G chosen in {{ns_hoop_folds}} of 3) | +{{ns_gain}} | {{ns_gain_lo95}}, {{ns_gain_hi95}} |
| All six, equal weight: R² gain / slope | +{{ew_gain}} / +{{ew_slope}} | {{ew_gain_lo95}}, {{ew_gain_hi95}} / {{ew_slope_lo95}}, {{ew_slope_hi95}} |
| Best of six, one-sided p corrected for the six | {{slope_pfw1}} | |
| Hoop G, holding wave-drill lean (n = {{jw_n}}) | +{{jw_hoop}} | {{jw_hoop_lo95}}, {{jw_hoop_hi95}} |
| With stance, snap load, club pass rush | +{{ct_slope}} | {{ct_slope_lo95}}, {{ct_slope_hi95}} |
| By class: 2023 / 2024 / 2025 | +{{het_cls_2023}} / +{{het_cls_2024}} / +{{het_cls_2025}} | between-class sd ≤ {{het_tau_hi}} |
| Rush level, Hoop G alone | {{mm_m2_hoopz}} points | {{mm_m2_hoopz_lo}}, {{mm_m2_hoopz_hi}} |
| Joint model, man coverage | {{jm_man}} points | {{jm_man_lo}}, {{jm_man_hi}} |
| Independent script, no shared code (n = {{v_n}}) | +{{v_slope}} | 90%: {{v_lo}}, {{v_hi}} |

*Table A1. Held-out classes; pressures per 450-rush season per sd unless stated.*

![Figure A1. Shuffle tests for the Hoop G slope, the all-six ridge gain and the two day-3 scores. The day-3 shuffles centre above zero because the pick + role forecast alone already beats pick order.](figures/figA1_screen.png)

`analysis.py` (about 15 minutes) rebuilds every number, figure and this text from `data/processed` and runs `verify.py`.
