# Bend as a Tiebreaker: Run-the-Hoop "Hoop G" and NFL Pass-Rush Pressure

Defensive linemen who turned tighter at speed in the Combine's Run-the-Hoop drill produced more NFL pressure than their draft slot, size and athletic score predicted ({{v_n_main}} linemen, 2023–2025 classes). After correcting for the 14 drill-and-outcome pairs we screened, plan on about {{pressures_per_450_eb}} extra pressures a season per standard deviation of bend. That is what moving up from pick 100 to pick {{pick_equiv_eb}} buys. Use it to break ties.

**Why DL, and why now.** Pressure is what a defensive lineman is paid for, and DL is the one position where Combine numbers already predict NFL production (3-cone ρ = {{rho_three_cone_z_press_eb}}). If tracked drills add anything, it should show here first. The 2023 Combine was the first with tracked position drills, and three classes now have NFL snaps.

## Hoop G in coaching terms

**Hoop G is the hardest sideways push a rusher holds for half a second while rounding the hoops, measured in g.** It is the drill version of bending the corner: dip the inside shoulder, keep speed through the turn around the tackle's outside hip, and rip through without drifting wide. A typical run peaks at {{hoop_g_median}} g, about {{hoop_speed_at_bend}} yd/s on a {{hoop_radius_typ}}-yard radius. **NFL rush-path bend** applies the same formula to a player's {{arc_snaps_per_player}} random sampled NFL pass rushes, from the snap to the throw or sack. It needs no ball or opponents.

We also tested the pass-rush drill's first-second burst, but it cannot rank players. A player's {{k_rush_mean}} reps have reliability {{rel_drill_d1_player}}, and ranking would take about {{snaps_for_07_drill}} reps. The NGS get-off it should predict is sound (ρ = {{rho_play_d1_getoff}} with tracked first-second distance per snap), so the limit is the drill.

![Figure 1. On Sundays, high-bend and low-bend rushers look alike. The six players are the three highest and three lowest Hoop G among DL with 250+ NFL rushes, chosen by rule. High-bend pressure rates run {{hero_hi_min}}–{{hero_hi_max}}%, low-bend {{hero_lo_min}}–{{hero_lo_max}}%. Each card ends with a scout's call.](figures/fig1_drill_vs_game.png)

## Data and baseline

The cohort is the {{n_cohort}} DL with 100+ regular-season pass rushes ({{n_cohort_edge}} EDGE, {{n_cohort_int}} interior, median {{cohort_median_snaps}} rushes); {{v_n_main}} of them ran the hoop. Pressure rate comes from play-level charting, pulled toward the DL mean ({{prior_mean}}%) for small samples. Drill numbers are standardized within Combine year.

**Who is left out.** {{n_hoop_excluded}} of {{n_hoop_runners}} hoop runners have fewer than 100 rushes. They played a median {{gps_out}} games a season against {{gps_in}} for the cohort, so injury, depth and rookie snap counts drive the cut. Bend is unrelated to playing time (ρ = {{rho_hoop_snaps_ps}}). Adding games per season to the baseline leaves the effect at {{pr_hoop_given_games}}.

**Baseline without tracking:** log draft pick, weight, NGS athleticism score and NFL EDGE share. All four exist for every player, so nothing is imputed. The 10-yard split and 3-cone are missing for many rushers, so they enter as imputed variants (Figure 4). With both added, the link weakens to {{v_r_angle}} [{{v_lo_angle}}, {{v_hi_angle}}]. Every "held-out" number is fit on two Combine classes and tested on the third.

## Finding 1: the stopwatch carries most of the signal

![Figure 2. Five Combine numbers against three NFL outcomes. Purple diamonds show what is left of a tracked drill number after the no-tracking baseline.](figures/fig2_signal_map.png)

The 3-cone and 40 track pressure, as teams already know. Among tracked drill numbers, only bend keeps a pressure link after the baseline. Drill bend is not "turns tighter on Sundays", though: its link to NFL rush-path bend ({{rho_hoop_arc_raw}}) falls to {{pr_hoop_arc}} once weight and alignment are removed. Balance through contact, which this tracking cannot see, is a plausible carrier.

## Finding 2: bend adds pressure, but less than one a season

![Figure 3. Pressure above the no-tracking forecast against Hoop G, split by whether the player ran the 40. The two slopes are statistically indistinguishable.](figures/fig3_hoop_pressure.png)

| Test | n | Estimate [95% CI] | Read |
|---|---|---|---|
| Bend → pressure, after baseline | {{v_n_main}} | ρ {{v_r_main}} [{{v_lo_main}}, {{v_hi_main}}] | real in-sample link |
| Per sd of bend: ran the 40 / skipped it | {{n_40}} / {{n_no40}} | +{{int_slope_40}} / +{{int_slope_no40}} pp | same slope (p = {{int_p}}) |
| Corrected for screening 14 pairs | {{v_n_main}} | {{pressures_per_450_eb}} pressures a season | planning number |
| Placebo: tackle rate | {{plc_n_tackle_rate}} | ρ {{plc_tackle_rate}} [{{plc_lo_tackle_rate}}, {{plc_hi_tackle_rate}}] | no positive link, as expected |
| Placebo: get-off | {{plc_n_quick}} | ρ {{plc_quick}} [{{plc_lo_quick}}, {{plc_hi_quick}}] | no link, as expected |

One standard deviation of bend goes with +{{slope_hoop_pp}} pressure points above the forecast, or {{pressures_per_450}} pressures in a 450-rush season. That is the best of 14 drill-and-outcome pairs we screened, and chance alone produces a lead this large about one time in eight. Pulling all 14 estimates toward their common mean leaves **{{pressures_per_450_eb}} pressures a season, our headline.** Bend does not predict tackles or get-off, which it should not, so a general "good athlete" effect is unlikely.

## Does it matter to a GM?

The obvious worry: the forecast gain comes almost entirely from the {{n_no40}} rushers who skipped the 40 ({{no40_edge}} EDGE, {{no40_2025}} from 2025, median pick {{no40_pick_med}} against {{has40_pick_med}}). An interaction model settles the cause. The bend slope is +{{int_slope_40}} pp [{{int_slope_40_lo}}, {{int_slope_40_hi}}] for players with a 40 and +{{int_slope_no40}} pp [{{int_slope_no40_lo}}, {{int_slope_no40_hi}}] without, so the link does not depend on the 40. What does depend on it is the forecast: among the {{n_40}} who ran, adding bend leaves held-out accuracy flat, and the gain is positive in only {{v_dr2pos_cc_main}}% of bootstrap samples. A sub-one-pressure effect is too small to show up in three folds of about 16 players.

![Figure 4. Left: the bend link under every baseline, including the 10-yard split and 3-cone. Right: how much bend improves a forecast of an unseen class, across class-stratified bootstrap refits; the share above zero is printed.](figures/fig4_robustness.png)

In draft terms, the planning effect ({{pressures_per_450_eb}} pressures) equals moving from pick 100 to pick {{pick_equiv_eb}}. Even the uncorrected effect equals 100 → {{pick_equiv_raw}}. **So bend is a tiebreaker between rushers in the same tier, never a reason to jump a round.** The no-40 group is where it matters most in practice: there, bend is one of the few movement numbers a team has.

## Technique and scheme

If bend is the dip-and-rip, it should show on edge snaps and late pressures. It does not show clearly. Bend does not change when pressure arrives (time to pressure ρ = {{plc_ttp}}), the share of quick pressures ({{plc_quick_share}}) or sacks per pressure ({{plc_finish}}). The player-level effect is {{pr_hoop_EDGE}} for EDGE and {{pr_hoop_interior}} for interior rushers. On sampled snaps, pressure odds per sd are ×{{sor_wide_1}} when aligned outside and ×{{sor_wide_0}} inside, both with intervals spanning 1. Blocker type is not in the data. Our read: bend is a general change-of-direction trait, not only the wide-9 arc.

![Figure 5. Film check on two 2025 rookies. Ezeiruaku turns the corner on Sundays but under-produced his forecast; Phillips neither bends in the drill nor on Sundays.](figures/fig5_film_check.png)

**Donovan Ezeiruaku** ranks #{{film_ezeiruaku_rank}} in 2025 bend, and his NFL rush paths sit at the {{film_ezeiruaku_arc_pct}}th percentile of DL. Yet he produced {{film_ezeiruaku_pr}}% [{{film_ezeiruaku_prlo}}, {{film_ezeiruaku_prhi}}] against a {{film_ezeiruaku_pb}}% forecast. He wins the corner, so the film question is hands and the rip at the top of the arc. **Jordan Phillips** ({{film_phillips_rank}} of {{board_n}}) shows no bend in the drill or in games and produced {{film_phillips_pr}}% against {{film_phillips_pb}}%. He fits a gap-control role, not a pass-rush bet.

## What a team can use

![Figure 6. The full 2025 board: every hoop runner with shrunk bend (80% interval), rookie pressure (80% interval) against the no-tracking forecast, and the call from the rule below.](figures/fig6_board_2025.png)

**Decision rule, applied to every row:**
1. **Bend plus** (80% interval above zero): break ties his way.
2. **Bend minus** (80% interval below zero): he must win with power or get-off; check that on film.
3. **Film first**: if a rusher with 100+ NFL rushes has a pressure interval wholly on the wrong side of his forecast, trust the film over the drill.

In 2025 this marks {{board_plus}} rushers plus, {{board_minus}} minus and {{board_n_flag}} film-first (Jordan Burch). Across all {{board_all_n}} runners from 2023–25, it gives {{board_all_plus}} plus, {{board_all_minus}} minus and {{board_all_flag}} film-first; the other film-first player, Byron Murphy, out-produced a low hoop. The full board with weight, role, pick, intervals and call is in `board_2023_2025.csv`.

**Ask for 3–4 hoop runs.** One run has reliability {{rel_hoop_sb}} [{{rel_hoop_sb_lo}}, {{rel_hoop_sb_hi}}]. Three runs reach {{rel_hoop_3runs}} and four {{rel_hoop_4runs}}, enough to rank. Undrafted Ethan Downs ran the best hoop in his class (z = +{{ex_top_z}}), which shrinks to +{{ex_top_shr}} [{{ex_top_lo}}, {{ex_top_hi}}]. With {{ex_top_snaps}} NFL rushes, he is worth a second run before moving him up.

## Pre-registered test

**Primary:** among 2026- and 2027-class DL with 100+ rushes by the end of 2027 (about {{n_prereg}}), the bend effect on pressure after the four-item baseline is above zero (one-sided α = 0.05). **Subgroup:** the bend slope is larger for rushers without an official 40. Our data say no difference (p = {{int_p}}), so we expect this to fail. Power is {{power_prereg_obs}} if the uncorrected effect is true and only {{power_prereg_eb}} at the planning effect, so the test will rule out a large effect, not prove a small one. An estimate at or below zero takes bend off the board.

## Limitations

- **Baseline choice.** With the 10-yard split and 3-cone imputed, the interval reaches zero.
- **Selection.** Corrected for 14 screened pairs, the main result does not clear p = 0.05 (best-of-14 p = {{perm_best14}}; re-selecting inside each held-out fold, p = {{nested_perm_p}}). It is a lead.
- **Small n.** About 20 players per class; every held-out interval includes zero. Only {{n_hoop_two}} players ran the hoop twice.
- **Game tracking is a sample.** Rush-path bend and snap splits use {{arc_snaps_per_player}} random rushes per player, with no blockers, ball or opponents. Scheme, stunts and quarterback timing are not in the data.
- **Availability.** The 100+ rush cut drops injured and buried players; we show it is not related to bend, but cannot rule out that the cut changes the effect.
- **Short careers.** The 2025 class has one season: its board is for order, not levels.

## Appendix: statistics

| Check (bend → pressure unless noted) | n | Estimate [95% CI] | Held-out R² gain [95% CI] |
|---|---|---|---|
| Main baseline | {{v_n_main}} | {{v_r_main}} [{{v_lo_main}}, {{v_hi_main}}] | +{{v_dr2_main}} [{{v_dr2lo_main}}, {{v_dr2hi_main}}] |
| Angle baseline: + 10-yd and 3-cone, imputed | {{v_n_angle}} | {{v_r_angle}} [{{v_lo_angle}}, {{v_hi_angle}}] | +{{v_dr2_angle}} [{{v_dr2lo_angle}}, {{v_dr2hi_angle}}] |
| Ran the 40: main / + official 10-yd | {{v_n_cc_main}} | {{v_r_cc_main}} / {{v_r_cc_10}} | +{{v_dr2_cc_main}} / {{v_dr2_cc_10}} |
| Mixed model, class intercept (pp per sd) | {{v_n_main}} | {{mixed_hoop_pp}} [{{mixed_hoop_lo}}, {{mixed_hoop_hi}}] | — |
| Train on one class, predict two | — | — | +{{l2o_dr2}}, positive in {{l2o_up}} of 3 |
| Bend alone, unadjusted for selection | {{v_n_main}} | p = {{pr_p_hoop_z_press_eb}} | — |
| True-score effect (one-run reliability) | {{v_n_main}} | {{dis_sh}} [{{dis_sh_lo}}, {{dis_sh_hi}}] | — |
| Bend → NFL rush-path bend, after weight, alignment | {{pr_hoop_arc_n}} | {{pr_hoop_arc}} [{{pr_hoop_arc_lo}}, {{pr_hoop_arc_hi}}] | — |

Main-baseline held-out R² rises from {{v_r2b_main}} to {{v_r2a_main}}; the class-stratified bootstrap puts the gain above zero {{v_dr2pos_main}}% of the time. Methods: Savitzky–Golay smoothing (window 7, order 2); sideways acceleration |v × a|/|v| averaged over 0.5 s, identical for drill and NFL snaps. Effects are Spearman partial correlations of baseline-residualized ranks ({{NBOOT}} player bootstraps); held-out intervals refit the whole loop. Imputation draws from weight, EDGE share, NGS score and bend, never the outcome. Interaction: OLS with class fixed effects, class-stratified bootstrap.

*Reproducible: `uv run python writeup/candidates/round_04/cand_1/analysis.py` writes every number to `numbers.json`, draws every figure and fills this text from `writeup_template.md`.*
