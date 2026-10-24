# Break G: Receivers Who Cut Harder at the Combine Drew Slightly More Targets, Not More Separation

We measured how hard {{n_combine_players}} wide receivers and tight ends pushed into the ground through the breaks of their Combine route drills. Among the {{n_main}} who ran an NFL route, each sd of this "Break G" came with {{coef_pp}} more targets per 100 routes than draft slot and testing forecast (95% CI {{coef_pp_lo}} to {{coef_pp_hi}}), with no extra separation, so it is a tiebreaker at most.

**Why this question.**
1. Route drills are the only Combine drills that copy a route, and 10 Hz tracking now records every break: the brake, the turn and the drive out.
2. Clubs will be sold "suddenness" scores built from those reps and told they mean separation.
3. A receivers coach wants two things: does he get open, and does the quarterback throw to him? We test both against what a club already has (pick, 40, weight, NGS score).

![Figure 1. Left: Break G effect on four NFL outcomes for the same {{n_common}} receivers and the same baseline, in player sd of each outcome (95% bootstrap CI). Targets and yards per route lean positive; both separation measures sit at or below zero. Right: TPRR slope inside each draft class and pooled (95% CI), with the share of forecast error removed when that class is held out (red = worse). Only 2023 clears zero.](figures/fig1_hero.png)

## The metric: Break G

**One line a coach can repeat.** Break G is how hard a receiver pushes into the ground in the one second around his cut (braking, turning, driving out), averaged over his route drills. A receiver who rounds the cut and drifts scores low; one who sinks, snaps and drives out scores high. The typical Combine rep is {{bg_drill_median}} g.

**Code definition.** Speed and acceleration come from smoothed 10 Hz x/y. The break is the first turn of 30° or more after {{c_path_yd}} yd of travel, entered above {{c_entry}} yd/s; breaks were found in {{pct_drill_detected}}% of {{n_drill_reps}} route-drill reps. Break G is the mean acceleration magnitude over the {{c_window_s}} s centred on the sharpest turn. Reps are standardized within drill × Combine year and averaged over a player's route drills (median {{median_drills_per_player}}), then pulled toward average in proportion to how few drills he ran, so one hot rep cannot make a grade. Units are sd of true Break G. The definition has been frozen since round 5.

![Figure 2. Speed (top) and acceleration (bottom) through every break, aligned on the sharpest turn. Blue: all of a player's Combine route-drill reps; orange: all his sampled NFL slant, out, hitch, in and corner breaks; thick = median. Players chosen by rule: two highest and two lowest Break G with 5+ route drills and 8+ NFL breaks.](figures/fig2_profiles.png)

Figure 2 compares like with like: every drill rep against every NFL break, same window. {{fig2_0_name}} and {{fig2_1_name}} brake and drive out harder in drills ({{fig2_0_drill_g}} and {{fig2_1_drill_g}} g) than {{fig2_2_name}} and {{fig2_3_name}} ({{fig2_2_drill_g}} and {{fig2_3_drill_g}} g). On NFL breaks the four run {{fig2_0_game_g}}, {{fig2_1_game_g}}, {{fig2_2_game_g}} and {{fig2_3_game_g}} g: the {{fig2_drill_gap}} g drill gap shrinks to {{fig2_game_gap}} g. NFL cuts are softer overall ({{bg_game_median}} vs {{bg_drill_median}} g), likely because they are run to leverage and timing, not a cone.

**Is it a trait?** Break G repeats from one set of five route drills to the other at {{rel_bg}} [{{rel_bg_lo}}, {{rel_bg_hi}}] (n = {{rel_n}}), rising to {{rel_7drills}} for a receiver who runs seven drills. The three pieces coaches usually name (decel into the break, speed kept through it, re-acceleration out) do not repeat on their own; speed kept at the apex is {{rel_sk}} (Figure 3). Break G is unrelated to the official 40 ({{rho_bg_forty}}) and draft slot ({{rho_bg_pick}}): it is not a speed score.

![Figure 3. Left: split-half reliability of nine break measurements across disjoint sets of route drills (95% CI); Break G in red. Right: each measurement's correlation with the official 40.](figures/fig3_trait.png)

## Result: a few more targets per route

**Data and baseline.** The outcome is career regular-season targets per route run (TPRR), weighted by routes. Of {{n_combine_players}} tracked receivers, {{n_main}} have an NFL route (median {{routes_median}}); the average TPRR is {{tprr_mean}}%. The {{n_no_route}} without one are not explained by Break G (t = {{reach_t}}). The no-tracking baseline is log draft pick, official 40, weight and NGS athleticism score (standardized within year × position), plus a TE flag. "Held out" means fitting on two draft classes and forecasting the third; the R² change is the share of that forecast's squared error removed by adding Break G.

{{TABLE:cls_table}}

**What to believe.** Every class points the same way, but only 2023 clears zero and 2024 held out gets slightly worse. That is what small classes do: one class of about 40 has power {{power_one_class}} to detect the pooled slope, so disagreement between classes is expected even if the effect is real. Three checks keep the claim small:

- **Selection.** Break G × TPRR was one of {{screen_k}} drill-measure × outcome pairs examined. The chance that the best of {{screen_k}} shuffled pairs looks this strong is {{p_fwer}}, and Break G is only the {{rank_bg_full}}th-strongest pair. Re-running the whole screen inside each fold fails: the pair it picks from two classes changes R² by {{nest_mean_gain_pct}}% on average in the third (last column). The search does not generalize; Break G survives only because its definition was fixed before this outcome was chosen.
- **Alternatives.** A ridge model with all nine break measures, tuned inside each fold, changes held-out R² by {{ridge_gain_pct}}%, so the force summary beats its parts. Adding 3-cone, shuttle and 10-yd split to the baseline (missing set to average) leaves the slope at {{rob_agil_pp}} (t = {{rob_agil_t}}). On the {{rob_cc3_n}} who actually ran the 3-cone it is {{rob_cc3_pp}} (t = {{rob_cc3_t}}), so we cannot show it adds to agility times there.
- **Influence.** Without Puka Nacua the slope is {{rob_droptop_pp}} (t = {{rob_droptop_t}}); counting each player once instead of weighting by routes it is {{rob_unw_pp}} (t = {{rob_unw_t}}).

After discounting for the {{screen_k}}-pair search (factor {{eb_shrink}}), the planning number is **{{plan_pp}} points of TPRR per sd, about {{plan_tg}} targets in a {{c_season_routes}}-route season**. On the {{n_rand_snaps_fmt}} randomly sampled tracked snaps, Break G predicts the share of routes targeted at {{rob_rand_pp}} points per sd (t = {{rob_rand_t}}, n = {{rob_rand_n}}); these are a subset of the season targets, so it checks the data, not the effect.

## Not separation, and why that does not contradict the targets

We ran the same break code on {{n_break_snaps_fmt}} NFL slant, out, hitch, in and corner snaps. A snap is graded only if the cut finished at least {{c_lead_s}} s before the throw, since a later cut cannot create separation at the throw; that drops {{pct_not_graded}}% of snaps and leaves {{n_graded}} ({{n_graded_players}} players). Separation over expected (SOE) is separation at the throw minus a context model (route, depth, cushion, coverage, alignment, time to throw, down and distance) fitted on 2023–24 snaps.

![Figure 4. A: Combine Break G per drill family vs NFL Break G on the same route family. B: NFL Break G on a snap (fifths) vs SOE on that snap (95% CI). C: player SOE vs TPRR beyond the no-tracking baseline; dot size = routes.](figures/fig4_soe.png)

- **Drill to the same NFL route.** Combine Break G does not carry into the same route family: {{fam_rho}} [{{fam_lo}}, {{fam_hi}}] over {{fam_n}} player-family pairs (Figure 4A).
- **Same snap.** Harder NFL cuts do not buy separation on that snap: {{snap_rho}} [{{snap_lo}}, {{snap_hi}}] (Figure 4B).
- **Player level.** One sd of Combine Break G is worth {{soe_coef_in}} inches of SOE [{{soe_lo_in}}, {{soe_hi_in}}] given the baseline. The upper bound is {{soe_hi_in}} inches per sd, against a player-to-player sd of {{soe_sd_in}} inches.

**Reconciling Figure 1.** The positive target slope and the flat-to-negative SOE slope are not in conflict. SOE barely relates to TPRR in this sample: {{rho_soe_tprr}} [{{rho_soe_tprr_lo}}, {{rho_soe_tprr_hi}}], n = {{n_recon}} (Figure 4C). Adding player SOE to the TPRR model moves the Break G slope only from {{recon_without}} to {{recon_with}}. So whatever Break G tracks, it does not earn targets by getting open at the throw. Timing, quarterback trust and catch-point play are candidates this data cannot separate.

## The board and how to use it

A wide receiver is **plus** if his 80% Break G interval sits above zero and **minus** if it sits below. Tight ends get no call (t = {{rob_te_t}} within TEs, n = {{rob_te_n}}). The forecast is the baseline fitted without the player's class. Top six and bottom four WRs with 100+ routes:

{{TABLE:board}}

Of the {{n_plus}} plus WRs, {{plus_beat_pct}}% beat their forecast, against {{neutral_beat_pct}}% of {{n_neutral}} neutral and {{minus_beat_pct}}% of {{n_minus}} minus (route-weighted gaps {{plus_gap_pp}}, {{neutral_gap_pp}} and {{minus_gap_pp}} points). The plus list is not a hit list: Boutte, Tucker, Palmer and Arian Smith all sit below forecast. The usable side is the downgrade: **minus receivers rarely out-earn their slot**.

| Situation | Action | Evidence here |
|---|---|---|
| Two WRs within a round, one with a minus call | Break the tie against him | {{minus_beat_pct}}% of minus WRs beat forecast |
| A plus call | Note it; do not move him up | {{plus_beat_pct}}% beat forecast, a coin flip |
| Fewer than 5 route drills | No call; ask for the full route set at his pro day | reliability {{icc_bg}} for one drill, {{rel_7drills}} for seven |
| Tight end | Ignore Break G | t = {{rob_te_t}} |
| "Suddenness" sold as separation | Ask for the SOE link; grade only cuts done {{c_lead_s}} s before the throw | {{snap_rho}} per snap here |

**Draft-room example.** In 2023 A.T. Perry (pick {{ex_minus_pick}}, Break G {{ex_minus_bg}}) and {{ex_other_name}} (pick {{ex_other_pick}}, {{ex_other_bg}}) went {{ex_dpick}} picks apart. The rule takes Palmer: {{ex_other_tprr}}% TPRR over {{ex_other_routes}} routes against Perry's {{ex_minus_tprr}}%, though Palmer also fell short of his {{ex_other_fc}}% forecast. Across all {{n_pairs}} minus WRs, {{n_pairs_minus_lower}} drew fewer targets per route than the nearest-pick non-minus WR of their class (median {{pairs_median_dpick}} picks apart). 2025 plus WRs to watch: {{b25_plus_names}}.

## Limitations

- **A lead, not a confirmation.** The {{screen_k}}-pair search p is {{p_fwer}}, the held-out gain interval spans zero, 2024 and 2025 include zero, and one player halves the slope. TPRR replaced round 5's targets per season after that result was seen, a forking path; the Break G definition was frozen, the outcome was not.
- **Mechanism unknown.** Break G does not carry to NFL cut force or separation, so "break" describes the drill, not a proven game skill.
- **Outcome.** TPRR reflects role, quarterback, scheme and health. The 2025 class has one season.
- **Sample.** NFL tracking is up to 40 plays per player, with no defenders or ball; SOE comes from the provided separation, so the drill-to-game nulls are partly a measurement limit.
- **Units.** One sd of true Break G is only {{g_true_sd}} g; most rep-to-rep spread is noise.

Reproduce: `uv run python writeup/candidates/round_07/cand_2/analysis.py` (about two minutes, `data/processed` only). It writes every quoted number to `numbers.json` with a claim-to-section map and renders this text from `writeup_template.md`; the full board is in `board.csv`.
