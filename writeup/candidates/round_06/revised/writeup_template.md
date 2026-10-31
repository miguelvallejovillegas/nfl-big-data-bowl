# Hoop G: A Lateral-Force Tiebreaker for Pass Rushers, Worth {{bayes_all}} Pressures a Season and Unproven Out of Class

**Hoop G is the hardest sideways push a defensive lineman holds while rounding the Combine's Run-the-Hoop drill. Beyond a no-tracking baseline, one sd more is worth {{bayes_all}} pressures per 450-rush season (90% interval {{bayes_all_lo}} to {{bayes_all_hi}}). Predicting each class from the other two, it adds no R². Use it to break ties, never to move a player up a board.**

| Test | Players | Estimate | Read |
|---|---|---|---|
| **Headline**, in sample: pressures per 450-rush season per sd (Bayesian, 90%) | {{bayes_n}} hoop runners with an NFL rush | **{{bayes_all}}** [{{bayes_all_lo}}, {{bayes_all_hi}}] | small, uncertain |
| Out of class, fully nested: held-out R² change (95%) | {{nf_n}} with 100+ rushes | {{nf_dr2}} [{{nf_dr2_lo}}, {{nf_dr2_hi}}] | no gain |
| Same, each held-out class 2023 / 2024 / 2025 | {{nf_n_2023}} / {{nf_n_2024}} / {{nf_n_2025}} | {{nf_dr2_2023}} / {{nf_dr2_2024}} / {{nf_dr2_2025}} | no gain in any |
| Out of class, log score: held-out log predictive density gain (± SE) | {{elpd_n}} | +{{elpd_skeptical}} ± {{elpd_skeptical_se}} nats; positive in {{elpd_skeptical_pos}} of 3 classes | small gain, not fully nested |
| Nested R² change, observed official 10-yd in the baseline (95%) | {{ob10_n}} | {{ob10_dr2}} [{{ob10_dr2_lo}}, {{ob10_dr2_hi}}] | no gain |

The two sample sizes are deliberate: the Bayesian model weights every rusher by his rushes, so it can use all {{bayes_n}}; the R² test needs a stable observed rate, so it keeps the {{nf_n}} with 100+ rushes.

## Why this question

Bend, turning the corner without losing speed, is a trait EDGE scouting reports routinely grade, and the Combine never scores it. Run-the-Hoop is the drill built to show it, yet it has no published time. Between picks 80 and 130, boards stack rushers with similar film grades, so a cheap tiebreaker on pressure has value if it is real. Three tracked classes (2023–25) allow a test on classes the metric was not fit on.

## What Hoop G measures, and what it does not

Hoop G is the peak sideways acceleration held for half a second while rounding the hoops, in g, standardized within Combine year. A typical run peaks at {{hoop_g_median}} g.

![Figure 1. Takeaway: Hoop G is not corner bend on Sundays. Three plus-bend EDGEs run outside paths on {{f1_robinson_out}}%, {{f1_ezeiruaku_out}}% and {{f1_tuipulotu_out}}% of classified rushes, close to the {{wt_share_outside}}% cohort rate, and across {{strip_n}} DL drill bend barely predicts game bend (ρ = {{pr_hoop_arc}} after weight and role).](figures/fig1_four_rushers.png)

We ran two in-game tests of the corner mechanism. Both fail. First, plus-bend EDGEs do not rush wider (Figure 1). Second, if hoop bend were a corner skill, its link to pressure should be strongest on EDGE-aligned snaps. On {{al_snaps}} random sampled rushes by {{al_players}} hoop runners, the odds of a pressure per sd of Hoop G are {{al_or_wide}} [{{al_or_wide_lo}}, {{al_or_wide_hi}}] on EDGE-aligned snaps and {{al_or_int}} [{{al_or_int_lo}}, {{al_or_int_hi}}] on interior snaps (90%). The season-level slopes agree: {{bayes_edge}} pressures for EDGEs and {{bayes_int}} for interior linemen.

**So we read Hoop G as a general lateral-force (change-of-direction) marker, not as bend.** The board relies only on its association with pressure, not on any corner mechanism.

## Data and baseline

**Outcome:** regular-season pressures per pass rush, 2023–25.

**Main model:** a beta-binomial regression of pressures on rushes with a class random intercept. The no-tracking covariates are log draft pick, weight, NFL EDGE share, NGS athleticism score, games per season and the official 10-yd split and 3-cone. Only {{bayes_n_10yd_obs}} of {{bayes_n}} ran the 10-yd and {{bayes_n_3cone_obs}} the 3-cone, so missing times are imputed from weight, role and NGS score, with a missing flag. Hoop G enters corrected for its one-run reliability ({{rel_hoop_sb}}). The "official times" baseline is tested directly on observed times only (Table 1, last row; 3-cone in the appendix).

## Finding 1: burst carries over to games, but not to pressure

![Figure 2. Takeaway: the 40 tracks pressure and get-off. Drill burst carries over to the same player's NFL get-off but not to pressure, and only bend keeps a pressure link after pick, weight, NGS score and role. Each row has its own n and its own grey no-power band.](figures/fig2_signal_map.png)

Get-off burst (yards covered in the rush drill's first second) tracks the same player's first second on random NFL snaps (ρ = {{rho_d1_game}} [{{rho_d1_game_lo}}, {{rho_d1_game_hi}}], n = {{n_d1_game}}). But reps are unreliable ({{rel_drill_d1_player}}) and burst says nothing about pressure ({{pr_d1_press}}).

## Finding 2: {{bayes_all}} pressures a season, prior-dependent, no out-of-class R²

![Figure 3. Takeaway: plan on {{bayes_all}} pressures a season per sd. Larger priors give larger effects, the nested out-of-class R² change is null in every class, and the log score leans positive but is not fully nested.](figures/fig3_headline.png)

On our draft-slot model, {{bayes_all}} pressures is the gap between pick 100 and pick {{pick_equiv}} (80% range {{pick_equiv_lo}} to {{pick_equiv_hi}}). The estimate depends on the prior. Our prior on the bend slope is skeptical by design. Doubling it gives {{pr_double}} [{{pr_double_lo}}, {{pr_double_hi}}]; a wide prior gives {{pr_wide}} [{{pr_wide_lo}}, {{pr_wide_hi}}]. We plan on the smallest.

Out of class, the R² test on {{nf_n}} rushers (about 20 per class) finds nothing. The log score uses every rusher's full pressure count and is less noisy. It favours the bend model by {{elpd_skeptical}} ± {{elpd_skeptical_se}} nats (wide prior: {{elpd_wide}} ± {{elpd_wide_se}}). But the hoop was chosen on all three classes before that test, so it is not a confirmation.

## What a team can use

**2025 class: every call** (bend interval = 80%, shrunk for number of runs; tie-break range = EDGEs picked within 30% of his slot):

{{TABLE:board25}}

**Head-to-head, same class, picks within 30%:** the three closest-pick EDGE pairs with a 0.4+ sd bend gap, scored on pressure minus out-of-class forecast:

{{TABLE:pairs}}

Across all {{pair_n}} such pairs in 2023–25, the higher-bend rusher came out ahead {{pair_wins}} times ({{pair_win_pct}}%, 90% [{{pair_win_lo}}, {{pair_win_hi}}]). Shuffling bend within class gives {{pair_null_pct}}% (p = {{pair_perm_p}}). Head-to-head, the tiebreaker is unproven. It costs little, which is why it remains a tiebreaker and nothing more.

![Figure 4. Every 2023–25 call against the out-of-class no-tracking forecast, with pressure rate and rushes beside each name. Takeaway: of {{clear_n}} calls with a forecast, {{clear_above}} land clearly above and {{clear_below}} clearly below; the rest sit inside noise.](figures/fig4_calls.png)

**Ask for 3–4 hoop runs.** One run has reliability {{rel_hoop_sb}} [{{rel_hoop_sb_lo}}, {{rel_hoop_sb_hi}}]; three reach {{rel_hoop_3runs}} and four {{rel_hoop_4runs}}. A run lasts about {{hoop_dur_s}} s inside a workout that is already tracked, so the cost is a few minutes of drill time and fatigue, not equipment. Undrafted Ethan Downs ran 2025's best hoop on one run: retest before moving him.

## Pre-registered test

Among 2026–27 DL with 100+ rushes by the end of 2027 (about 40), bend's pressure effect after the full baseline must be above zero (one-sided α = {{alpha_prereg}}). Power is {{power_prereg}}, so a miss will not disprove a small effect. A negative estimate would retire the tiebreaker.

## Limitations

- **Not confirmed out of class.** The nested R² test is null, and the positive log score did not re-select the hoop inside each fold.
- **Prior-dependent.** The planning number is the smallest of three prior scales.
- **Mechanism unknown.** Path direction and alignment come from tracking, not film. With no blockers or ball, we cannot tell what the hoop captures; we call it lateral force, not bend.
- **Injuries and snap role.** Pressure rate ignores injuries and how a rusher is used (passing downs, rotation). We adjust only for games per season and EDGE share.
- **Small n and missing times.** About 30 hoop runners with NFL rushes per class; most lack a 3-cone.
- **Sampled game tracking:** 20 random rushes per player, no scheme labels.
- **Short careers.** 2025 has one season; its board ranks, it does not set levels.

## Appendix: method notes and checks

The bend-slope prior is scaled to the typical effect of 14 screened drill-outcome pairs. In the nested loop, the hoop is the screen's top pick in {{nf_hoop_sel_pct}}% of folds. Path direction uses the position 3 yd upfield on {{wt_n}} EDGE-aligned rushes. Drill bend does not raise the outside share (partial ρ = {{wt_rho_out}} [{{wt_rho_out_lo}}, {{wt_rho_out_hi}}], n = {{wt_rho_out_n}}). Of the 2023–25 plus calls with 100+ rushes, {{fa_plus_above}} of {{fa_plus_n}} beat their forecast ({{fa_plus_pct}}%, 90% bootstrap [{{fa_plus_lo}}, {{fa_plus_hi}}]; shuffled-bend null {{fa_plus_null}}%, p = {{fa_plus_perm_p}}). Calls never see pressure, but 2023–24 helped choose the hoop. In 2025, the class that did not, the count is {{fa25_above}} of {{fa25_n}}. **This is not evidence.**

| Check (Laplace unless noted) | n | Pressures per season per sd [90%] or as stated |
|---|---|---|
| Laplace version of the headline (MCMC: {{bayes_all}}; R-hat {{bayes_rhat}}) | {{bayes_n}} | {{lap_all}} [{{lap_all_lo}}, {{lap_all_hi}}] |
| Observed 10-yd / 3-cone only, time in the baseline | {{cc10_n}} / {{cc3_n}} | {{cc10_t}} [{{cc10_t_lo}}, {{cc10_t_hi}}] / {{cc3_t}} [{{cc3_t_lo}}, {{cc3_t_hi}}] |
| Multiple imputation of both times ({{mi_m}} draws) | {{bayes_n}} | {{mi}} [{{mi_lo}}, {{mi_hi}}] |
| Per class, fit without it / inside it only | {{sg_n_2023}} / {{sg_n_2024}} / {{sg_n_2025}} | 2023: {{sg_tr_2023}} / {{sg_te_2023}}; 2024: {{sg_tr_2024}} / {{sg_te_2024}}; 2025: {{sg_tr_2025}} / {{sg_te_2025}} |
| Leave one rusher out (posterior mode) | {{bayes_n}} | {{loo_min}} to {{loo_max}} |
| Nested R² change, observed 3-cone in the baseline (95%) | {{ob3_n}} | {{ob3_dr2}} [{{ob3_dr2_lo}}, {{ob3_dr2_hi}}]: uninformative |
| Placebo, run-stop proxy / back-pedal drill, partial ρ (95%) | {{plc_runstop_n}} / {{neg_bp_n}} | {{plc_runstop}} [{{plc_runstop_lo}}, {{plc_runstop_hi}}] / {{neg_bp}} [{{neg_bp_lo}}, {{neg_bp_hi}}] |

*Reproducible: `uv run python writeup/candidates/round_06/revised/analysis.py` writes `numbers.json`, draws every figure and fills this text from `writeup_template.md`.*
