# Speed That Translates: The Combine Start Doesn't Carry Over, Rushers Fast on Both Clocks Do

**First step** = a lineman's speed 0.3–0.5 s after he first moves. **Both** = NFL first step in the top third *and* official 40 time in the top half, cut-offs set on other classes. A lineman's tracked Combine start does not predict his NFL start (ρ = {{tr_fs}}). But Both rushers beat a held-out draft-slot pressure forecast {{sl_ks}} of {{sl_ns}} times; everyone else {{sl_ko}} of {{sl_no}}.

![Figure 1. Four rushers: Combine 40 start (orange) over his own NFL starts (blue); grey = league middle half.](figures/fig1_four_rushers.png)

## Why pass rushers, and why now

A pass rush starts like a 40: from a stance, straight ahead, before anyone touches him. If Combine movement translates anywhere, it is here; of five groups screened, only DL starts are both readable from 20 snaps and linked to production (Figure 2C). After a rookie season a club decides who gets third-down snaps and whether to buy a veteran, usually on a handful of sacks. A hit is worth money: the alternative, a mid-tier veteran edge, cost a median ${{vet_apy}}M a year in 2025, against about ${{wc_cj_apy}}M for a day-3 rookie.

## So what for a team

1. **Act on three names (Table 3): extend Byron Young (LAR) and YaYa Diaby (TB); give Chop Robinson (MIA) a full passing-down role, re-scored at midseason because he missed his 2025 check.** All three meet the frozen act-on rule: on the list, {{act_minrush}}+ rushes, pressure a point or more over forecast, under {{avail_cut}}% of games missed.
2. **Pro personnel: use Both as a ranking, not a regression input.** On held-out classes the list produced {{wc_rule_season}} more pressures a season over forecast than everyone else; as a continuous feature no speed measure, official 40 included, improves held-out R² (Table 1).
3. **Draft room: do not price the tracked 40 start.** It adds nothing beyond draft slot in any test (Table 1); the official 40 time carries the speed signal and clubs already pay for it.
4. **Chart {{snaps_for_07}}+ random rushes per young rusher** before trusting his first step (8 give reliability {{rel_8snaps}}).

## The metric

First step is measured in sd after weight and role. In that window the 40 and a pass rush are the same movement; by 1.5 s they part ({{dr_med_40}} yd/s in the 40, {{dr_med_sunday}} on a rush, because he meets a blocker).

**Data.** Tracked Combine 40s, year-standardized, curves starting at first movement (removing the clip's pre-laser lead). NFL: random sampled pass rushes only, alignment-adjusted: {{n_snaps_dl}} starts by {{n_gplayer}} linemen, median {{snaps_med}} each, reliability {{rel_fs_game}}; it tracks NGS get-off snap by snap (ρ = {{rho_fs_ngs_snap}}). **Pool:** of {{x_all}} linemen, {{x_nostarts}} have fewer than 8 sampled starts and {{x_lowrush}} fewer than 100 rushes; {{x_pool}} remain. Outcome: career pressures per rush. **Baseline without tracking:** draft pick, weight and role (share of snaps at EDGE), fitted on the other two classes: held-out R² {{bl_pick_r2}} [{{bl_pick_r2_lo}}, {{bl_pick_r2_hi}}], RMSE {{bl_pick_rmse}} pressure-rate points (class mean alone: {{bl_mean_rmse}}), rank correlation {{bl_pick_rho}}. A stronger baseline adds the official 40 time and season NGS get-off: R² {{bl_full_r2}}, RMSE {{bl_full_rmse}}, ρ {{bl_full_rho}}.

## Finding 1: the Combine start does not carry over

Raw, the 40 ranks NFL speed at 1.5 s at {{tr_raw_15}}, but that is weight. After size, the 40 first step previews the NFL first step at ρ = {{tr_fs}} [{{tr_fs_lo}}, {{tr_fs_hi}}] (n = {{n_tr}}). Noise is not the excuse: with two reps averaged a perfect link could show {{ceil_2rep}}. Per-player translation scores (gap, speed ratios, curve shape) never exceed {{sim_absmax}} against pressure beyond pick, and the carry-over score lowers held-out R² ({{dcarry_r2}} [{{dcarry_lo}}, {{dcarry_hi}}]). **Translation does not explain production beyond pick.**

{{TABLE:ledger}}

*Table 1. Pressure-rate points per sd faster (DL average about 8%), 90% interval, each class held out in turn; last column: change in held-out R² when the measure is added to the pick, weight, role regression.*

**Reading Table 1.** Slopes for the official 40 and NFL first step are positive but small next to rusher-to-rusher noise, so refitting with them gains nothing in held-out R², as public audits found. Any value is at the top of the ranking: Finding 2.

![Figure 2. Translation and payoff tenth by tenth (A), by held-out class (B) and by position group (C, grey = too noisy to read).](figures/fig2_translation_heldout.png)

## Finding 2: two signals that agree beat the forecast

A rusher "hits" if his pressure rate beats the held-out forecast (base rate {{jb_pct}}% [{{jb_lo}}, {{jb_hi}}]).

{{TABLE:robust}}

*Table 2. Five checks on one shortlist. Hit-rate intervals: Jeffreys; gap interval: rushers resampled within class.*

**Verdict: a real but modest edge in the full pool, unproven for day-3 picks (three shortlisted).** The list's hit rate, {{js_pct}}% [{{js_lo}}, {{js_hi}}], sits above the base rate's interval; the gap narrows in every stricter check and stays positive. Row 5 is the cleanest: called from 2023–24 starts only, scored on 2025 against a forecast that knows each rusher's own past pressure. Reweighting for who reaches 100 rushes leaves the gap at {{ipw_gap}} points [{{ipw_gap_lo}}, {{ipw_gap_hi}}].

![Figure 3. A: rushers against held-out cut-offs; top right = shortlist. B: the five checks of Table 2.](figures/fig3_shortlist.png)

## Every matched pair, not one chosen case

Each Both rusher faces the non-Both rusher of his class and role nearest in draft slot (median {{mp_pickgap}} picks apart) on raw pressure rate. Career: Both won {{mp_w}} of {{mp_n}} pairs, by {{mp_diff}} points on average [{{mp_diff_lo}}, {{mp_diff_hi}}]; sign test p = {{mp_p}}. Frozen after 2024 and scored on 2025: {{mpf_w}} of {{mpf_n}}, {{mpf_diff}} points [{{mpf_diff_lo}}, {{mpf_diff_hi}}], p = {{mpf_p}}. The win count alone proves nothing. Panel B and Table 2 row 5 score the same seven 2024 calls two ways: against a twin ({{mpf_w}} of {{mpf_n}}) and against the rusher's own past pressure ({{tf_ks}} of {{tf_ns}}).

![Figure 4. Every Both rusher against his draft-slot twin: career (A) and frozen 2024 call scored on 2025 (B).](figures/fig4_matched_pairs.png)

## Why it works: the same rusher, faster snap

On {{sn_n}} random rushes by {{sn_players}} linemen, a first step one sd faster than *his own* average multiplies that snap's odds of pressure by {{sn_or}} [{{sn_or_lo}}, {{sn_or_hi}}]: {{sn_top5}}% for his fastest fifth of starts, {{sn_bot5}}% for his slowest. It holds with that snap's NGS get-off, sideways path, down and alignment fixed ({{sn_or_ctl}}). *Film hypothesis, untested:* fast-start losses should be rushes that win the edge but have no counter.

![Figure 5. Same rusher, faster snap: pressure by first-step fifth; odds ratios by control, class, role.](figures/fig5_snap_mechanism.png)

**Where rule and tape disagree.** Maason Smith (Jacksonville, 2024, pick 48) has the pool's best NFL first step, a {{b_smith_t}}th-percentile 40 and {{b_smith_rate}}% pressure against a {{b_smith_fc}}% forecast: fast off the ball, no win. Grade hands and a counter before cutting his snaps.

## The 2026 list, frozen

Same rule, cut-offs from all three classes, every lineman with 8+ sampled starts. Bold = act now.

{{TABLE:list26}}

*Table 3. Percentiles within the pool; forecast = pick, weight, role, held out by class. Flag = {{avail_cut}}%+ of games without a recorded snap. Last column: frozen 2024 call vs own past pressure.*

**Track record.** Listed 2024 calls scored in 2025: {{rec_hit}} hits, {{rec_miss}} miss (Robinson, below his own strong rookie rate). Johnson and Adebawore are availability questions first.

**Pre-registration and what would change the call.** Rule, {{pre_nlist}} names and test are frozen in `frozen_rule.json` (sha256 `{{rule_hash}}…`, 24 October 2026, no 2026 data used). Test: in 2026, list members with 100+ rushes beat a persistence forecast more often than the other {{pre_nrest}} pool rushers (one-sided Fisher p < {{alpha}}). At the 2025 time-forward rates ({{pre_pa}}% vs {{pre_pb}}%) power is {{pre_pow_obs}}%; at half that gap, {{pre_pow_half}}%. Passing needs {{need_k}} of {{pre_nlist}} list hits if the rest hit {{need_rest_k}} of {{pre_nrest}}. **We drop the rule if list members do no better than the rest**: {{nb_half}}% likely if half the edge is real, {{nb_null}}% if none is.

## Limitations

- **Not blind until 2026.** The two-signal form was chosen after seeing all three classes; only the cut-offs are held out. The frozen-2024 row has {{tf_n}} rushers and overlaps the main shortlist.
- **No regression gain.** No speed measure, nor the Both flag, reliably raises held-out R²; the edge is a ranking edge in a pool of {{x_pool}}.
- **Small lists.** Ten rushers, three on day 3; matched-pair sign tests are not significant.
- **Sampled tape, no blockers or ball.** A slow start can be a two-gap call, a fast one a free rush; injuries enter only as missed games.

## Appendix

**Estimands.** Table 1's held-out slope (NFL first step {{ht_nfl_pp}} points per sd) is the headline; a rusher random-intercept binomial re-fit gives {{mm_pp}} [{{mm_lo}}, {{mm_hi}}]. Beyond each rusher's own 2023–24 pressure, the frozen first step adds +{{tf_pp}} [{{tf_pp_lo}}, {{tf_pp_hi}}] in 2025. **Table 1 shuffle p** (measure shuffled within class): tracked 40 start {{ht_c40_p}}, official 40 {{x_f40_pick_p}}, NFL first step {{ht_nfl_p}}; best of {{ht_nwin}} windows, p = {{ht_nfl_p_adj}}. **Shuffle p (labels within class), Table 2 rows 1–5:** {{rb_main_p}}, {{rb_full_p}}, {{rb_x_p}}, {{rb_d3_p}}, {{rb_tf_p}}. **ΔR² beyond the stronger baseline:** NFL first step {{dfnfl_r2}} [{{dfnfl_lo}}, {{dfnfl_hi}}], Both {{dfboth_r2}} [{{dfboth_lo}}, {{dfboth_hi}}]. **Calibration.** Shortlist expected {{calib_ps}}% hits, got {{calib_os}}%; Brier {{brier_rule}} vs {{brier_base}} for the base rate.

*Reproduce: `uv run python writeup/candidates/round_17/cand_3/analysis.py` (about 5 minutes).*
