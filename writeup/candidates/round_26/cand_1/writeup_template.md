# Run-the-Hoop Lean: A Day-3 Film Flag, Not a Board Input

Defensive linemen who lean hardest through the tightest turn of the Combine's Run-the-Hoop drill went on to about 2 more pressures a season than their draft slot and role predicted. On Sunday they do not bend more often than other rushers; their hardest bends end in pressure more often. Use the drill to decide whose tape to open first, not where a name goes on the board.

Two caveats. The hoop measure was the best of {{n_family}} drill measures we tried, and the 2025 class, tested alone, was flat. The 2026 class decides.

- **Scout, Monday:** flag every day-3 or undrafted lineman in his class's top third on the hoop, about {{flags_per_class}} names a class ({{flags_2026_p10}}–{{flags_2026_p90}}), and watch his rushes before the board meets. Table 2 says what to look for. The 2026 class is not in this release's Combine tracking, so no 2026 names yet; the frozen rule (`frozen_rule.json`) prints the 2026 list the day it posts.
- **Combine:** grade four hoop laps, not one; a single lap repeats at only {{rel_turn_lap}}.

**Why now.** {{pct_day3_dl}}% of these {{n_dl}} linemen went on day 3 or undrafted, where tape is thinnest and clubs find rotational edge rushers. Bend is graded only off tape; the hoop is the one Combine drill that times it.

## Why lean should make pressure

An edge rusher who reaches the tackle's outside shoulder has to turn toward the quarterback without slowing down. If he must brake to turn, the tackle recovers and rides him past the pocket. How hard a body leans through a turn is speed squared over turn radius, in g: the rusher who keeps his speed through a tighter circle leans harder. The hoop measures that with no blocker in the way.

**Hoop G** is his hardest half-second of lean in the figure-eight, ranked within his Combine class. The median lineman carries {{hoop_mph_median}} mph around a {{hoop_radius_median}}-yard radius ({{hoop_g_median}} g). On an NFL rush his hardest lean is {{nfl_mph_median}} mph around {{nfl_radius_median}} yards ({{nfl_bend_median}} g): the tackle pushes the arc out to {{turn_match_median}} times his hoop radius. So we compare men within a class, never raw g across venues. A pressured snap cost the offense {{epa_press}} expected points [90%: {{epa_press_lo}}, {{epa_press_hi}}] on {{epa_n}} sampled pass plays.

![Figure 1. Four of the day-3 names the flag adds: Combine hoop lap (top) and every randomly sampled NFL rush (bottom), one yard scale and one lean colour scale.](figures/fig1_hero.png)

## Does it hold on a class the model never saw?

**Outcome:** career pressure rate through 2025, shrunk toward the {{press_mean}}% mean, for linemen with 30+ rushes. **Test:** each class ({{ho_n_per_class}} rushers) is forecast by a model fitted on the other two; drill measures are standardized within Combine year. **The one test**, written down for 2026: Hoop G against a forecast from draft pick plus edge-or-interior role.

| Forecast without tracking (n = {{n_an}}) | R² | R² with drill | Drill effect per sd [95%] | Chance it is luck |
|---|---|---|---|---|
| **Pick + role, with Hoop G (the test)** | {{r2_pick}} | {{r2_pick_h}} | **+{{bs_pick}} [{{bs_pick_lo95}}, {{bs_pick_hi95}}]** | one-sided p = {{slope_pfw1}}, best of {{n_family}} |
| Pick + role, measure re-chosen inside each training fold | {{r2_pick}} | {{ns_r2}} | +{{ns_slope}} [{{ns_slope_lo95}}, {{ns_slope_hi95}}] | R² gain p = {{ns_p}}, selection rerun on shuffles |
| Everything, shrunk: pick, role, weight, NGS score, 10-yd split, 3-cone | {{r2_ridge}} | {{r2_ridge_h}} | +{{bs_ridge}} [{{bs_ridge_lo95}}, {{bs_ridge_hi95}}] | |

*Table 1. Pressures per 450-rush season per sd of the drill measure. R² = share of rusher-to-rusher spread explained on a held-out class.*

A rusher one standard deviation (sd) above his class beat the forecast by about 2 pressures a season. The honest version of the selection matters: when each training fold picks its own best measure from the six, it picks Hoop G in {{ns_hoop_folds}} of 3 folds and a hoop-drill measure in {{ns_hoopdrill_folds}} of 3, and the held-out R² gain stays at {{ns_gain}} but with a 95% interval of [{{ns_gain_lo95}}, {{ns_gain_hi95}}]. The hoop drill carries the signal; which hoop number is best is not settled.

Figure 2 puts the stress tests on one scale. The slope holds with wave-drill lean in the model, with the {{n_teams}} NFL clubs resampled, and with stance (listed as an outside linebacker), snap load and the club's pass-rush level added to the forecast. It is flat in 2025 alone; Table A1 has the detail.

![Figure 2. The test and its stress tests against held-out forecasts, 95% intervals, n per row.](figures/fig2_tests.png)

## Drill to Sunday: the hoop marks who finishes the bend

NFL bend tracks pressure, but the hoop does not predict how often a man bends on Sunday: the rank correlation of Hoop G with his average NFL rush lean is {{tr_hoop_g_bend}}, and with his hardest tenth of rushes {{alt_bend90}} [90%: {{alt_bend90_lo}}, {{alt_bend90_hi}}]. So we asked the question rush by rush.

On each of {{mm_n_rush}} randomly sampled rushes by the {{mm_n_men}} tested men (median {{mm_rush_med}} each), we compared his lean with his own other rushes, which takes his size, role and usual blockers out of the comparison. A rush on which a man leans 1 sd harder than his own norm is pressured {{mm_m1_wz}} points more often [95%: {{mm_m1_wz_lo}}, {{mm_m1_wz_hi}}]. For a rusher 1 sd higher on Hoop G that payoff grows by {{mm_m2_wzxhoopz}} points [{{mm_m2_wzxhoopz_lo}}, {{mm_m2_wzxhoopz_hi}}]. On rushes as tight as twice his own hoop radius, the gain per sd of Hoop G is {{mm_m3_tightxhoopz}} points [{{mm_m3_tightxhoopz_lo}}, {{mm_m3_tightxhoopz_hi}}]. On their hardest-lean quarter of rushes, top-third Hoop G men were pressured {{f4_top_q4}}% of the time, against {{f4_rest_q4}}% for the rest; on their softest quarter both were near {{f4_top_q1}}%.

In scout terms: high-Hoop-G men do not turn the corner more often; when they do, they keep enough speed to finish. That is a mechanism a scout can check on tape. It rests on about {{mm_rush_med}} sampled rushes per man, and the own-lean × Hoop G interval reaches zero.

![Figure 3. Rush-level test: pressure rate by how hard he leaned on that rush relative to his own rushes, Hoop G top third vs the rest, and the mixed-model terms (95%).](figures/fig3_mechanism.png)

## Day 3: a film queue, not a board

All {{d3_n}} day-3 and undrafted rushers in held-out classes, each class scored by a model that never saw it:

- **Ordering:** within-class rank correlation with pressure rate is {{dr_fch}} [95%: {{rho95_fch_lo}}, {{rho95_fch_hi}}] for forecast + Hoop G, against {{dr_pick}} [{{rho95_pick_lo}}, {{rho95_pick_hi}}] for pick order; the difference is above zero in {{rho_gain_ppos}}% of resamples.
- **Calling hits** (beat his class's day-3 median): Brier score {{bri_fch}} [{{bri_fch_lo}}, {{bri_fch_hi}}] against {{bri_pick}} [{{bri_pick_lo}}, {{bri_pick_hi}}] for pick order (a coin is {{bri_coin}}; lower is better).
- **Top third:** the two lists tie, {{tt_fch_k}} of {{tt_fch_n}} hits each. They share {{ov_shared}} names; Hoop G adds {{ov_add}} that pick order skips ({{ov_add_hit}} hit) and drops {{ov_drop}} ({{ov_drop_hit}} hit).

So Hoop G orders day-3 tape a little better than pick order, the intervals overlap, and it does not pick more hits. It decides **which tape opens first**. Table 2 is the queue of the {{ov_add}} added names, with the cue a fixed rule draws from each man's sampled snaps.

| Added name, film order | Hoop G (sd) | Pressures a season vs forecast | Beat day-3 median | Corner turns, random snaps | What to look for on tape |
|---|---|---|---|---|---|
{{T:t7_rows}}

*Table 2. Cue rules, fixed in code: corner turn = a rush whose hardest lean is in the top quarter of all sampled rushes (25% is average). 25%+ corner turns and most pressures on them: "dip and flatten"; 25%+ otherwise: "bends but finishes elsewhere"; fewer, with get-off quicker than the {{getoff_med_all}} s lineman median: "wins off the ball"; else "power and effort".*

![Figure 4. Day-3 evaluation on all held-out day-3 rushers (A, B) and the added names with 95% intervals (C).](figures/fig4_day3.png)

{{T:t7_best}} ({{T:t7_best_pick}}) is the strongest case: {{t7_best_ct}} corner turns in {{t7_best_ctn}} random snaps and {{t7_best_ov}} pressures a season over forecast, though the rule says to check where his finishes come from. {{T:t7_worst}} is the weakest: {{t7_worst_ov}}, with a median get-off of {{t7_worst_go}} s; his cue is the lowest priority.

## The 2026 test

Written down before any 2026 NFL data (`frozen_rule.json`): after Week 18, check whether Hoop G still predicts pressure beyond the frozen pick + role forecast. A positive slope at one-sided p < 0.10 keeps the flag; zero or below retires it. One class detects a real slope this size {{pws_obs_1}}% of the time; three classes, {{pws_obs_3}}%. The rush-level finish effect is checked the same way. Hoop G reaches the board only if its day-3 top third beats pick order by {{hg_bar}}+ points over two classes.

## Limitations

- **Small samples, single reps.** {{n_an}} rushers in the model, {{d3_n}} on day 3; most ran one hoop lap.
- **Forking paths.** Hoop G was the best of {{n_family}} measures; re-choosing inside each fold keeps the gain but not always Hoop G. Of {{n_tests}} tests, {{n_holm}} survive a family-wise correction at 0.10 and {{n_bh}} a false-discovery correction.
- **Thin NFL tracking.** About {{mm_rush_med}} random rushes per man, no blockers, ball or film; the cues in Table 2 are play-log reads.
- **Controls are proxies.** Stance is the roster listing ({{n_standup}} listed outside linebackers); snap load is partly an outcome; the club's pass rush is read from other cohort linemen only.
- **Missed time.** Rates are per rush; the {{n_short_career}} linemen under 30 rushes are left out.

## Bottom line

The hoop picks out day-3 rushers whose hardest bends end in pressure, worth about 2 pressures a season. It orders film; it does not set the board. Grade four laps and let the 2026 class decide.

## Appendix

| Check (n = {{n_an}} unless stated) | Estimate | 95% interval |
|---|---|---|
| Hoop G, holding wave-drill lean (n = {{jw_n}}) / wave lean, holding Hoop G | +{{jw_hoop}} / +{{jw_wave}} | {{jw_hoop_lo95}}, {{jw_hoop_hi95}} / {{jw_wave_lo95}}, {{jw_wave_hi95}} |
| R² gain of Hoop G beyond wave lean | +{{jw_gain_h_given_w}} | {{jw_gain_lo95}}, {{jw_gain_hi95}} |
| R² gain over pick + role, NFL clubs resampled | positive in {{cdr2_pick_ppos}}% | one-sided p = {{cdr2_pick_p1}} |
| Measure re-chosen in each fold: R² gain | +{{ns_gain}} | {{ns_gain_lo95}}, {{ns_gain_hi95}} (positive {{ns_gain_ppos}}%) |
| With stance, snap load, club pass rush | +{{ct_slope}} | {{ct_slope_lo95}}, {{ct_slope_hi95}} |
| Fit on 2023–24, tested on 2025 | +{{pro_s25}} | {{pro_s25_lo95}}, {{pro_s25_hi95}} |
| 2023–24 rookie seasons only / later seasons | +{{tm_rk2324}} / +{{tm_lt2324}} | {{tm_rk2324_lo95}}, {{tm_rk2324_hi95}} / {{tm_lt2324_lo95}}, {{tm_lt2324_hi95}} |
| Hoop G × class interaction; 2025 pooled with 2023–24 | p = {{int_p}}; +{{hm_b25}} | 2026 slope positive: {{hm_new_ppos}}% |
| Pick only: R² {{r2_pick1}} → {{r2_pick1_h}}; slope | +{{bs_pick1}} | {{bs_pick1_lo95}}, {{bs_pick1_hi95}} |
| Shuffle p, Hoop G within class and role (5000 shuffles) | {{pl_cr_p}} | |
| Rush level, Hoop G alone | {{mm_m2_hoopz}} points | {{mm_m2_hoopz_lo}}, {{mm_m2_hoopz_hi}} |
| Rush level, own-lean × Hoop G, rusher bootstrap | +{{cb_wz_int}} | {{cb_wz_int_lo}}, {{cb_wz_int_hi}} |
| Hoop G vs share of hoop-tight NFL rushes | {{alt_tight_sh}} | 90%: {{alt_tight_sh_lo}}, {{alt_tight_sh_hi}} |
| Independent script, no shared code (n = {{v_n}}) | +{{v_slope}} | 90%: {{v_lo}}, {{v_hi}} |

*Table A1. Every row scored on held-out classes; units are pressures per 450-rush season per sd unless stated.*

![Figure A1. The six pre-listed drill measures (95%) with the best-of-six chance line, and the two placebos.](figures/figA1_screen.png)

`analysis.py` (about 11 minutes) rebuilds every number, figure and this text from `data/processed`, and runs `verify.py`.
