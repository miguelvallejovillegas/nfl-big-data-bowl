# Speed That Translates: Pass Rushers Fast on the Stopwatch and on Sundays

**A lineman's tracked Combine 40 start does not carry over to his NFL pass-rush start. Agreement does: rushers fast on the official 40 *and* in about 20 sampled NFL starts beat a draft-slot pressure forecast {{sl_ks}} times in {{sl_ns}} (rest: {{sl_ko}} in {{sl_no}}) on held-out classes. Use it to hand out passing-down snaps.**

![Figure 1. Four rushers, one scale: his Combine 40 start (orange) over his own NFL starts (blue).](figures/fig1_four_rushers.png)

## Why it matters

After a rookie season, a club decides which young rusher gets third-down snaps and whether to buy a veteran. That call is usually made on sacks: a few a season, mostly noise. The 2023–25 Combine 40s were tracked at 10 Hz, and NFL tracking shows the same lineman leaving a stance. If the two agree, a club can trust a short NFL sample sooner.

## So what for a team

1. **Pro personnel: rank young rushers by "Both"** (NFL first step in the top third and 40 time in the top half). On held-out classes that list produced +{{sl_exs_s}} pressures a season over forecast; the rest, {{sl_exs_o}}. The 2026 list is Table 4.
2. **Draft room: pay for the stopwatch, not the trace.** Only the official 40 adds pressure beyond draft slot (Table 1).
3. **Measurement: chart {{snaps_for_07}} or more random rushes per young rusher** before trusting his first step (reliability 0.7; 8 rushes give {{rel_8snaps}}). Do not add Combine 40 reps for this: {{dl40_2rep}}% of linemen already run two.

## The metric

**First step = a lineman's speed 0.3 to 0.5 s after he first moves.** In that window the 40 and a pass rush are the same movement; by 1.5 s they part ({{dr_med_40}} yd/s in the 40, {{dr_med_sunday}} on a rush, because he meets a blocker). Both sides are in sd after weight and role (share of snaps at EDGE).

**Data.** Tracked Combine 40s, year-standardized; curves start at first movement, so the clip's ~0.3 s lead before the laser does not matter. NFL: random sampled pass rushes only, alignment-adjusted: {{n_snaps_dl}} starts by {{n_gplayer}} linemen, median {{snaps_med}} each, reliability {{rel_fs_game}}; it tracks NGS get-off snap by snap (ρ = {{rho_fs_ngs_snap}}). **Pool:** of {{x_all}} linemen, {{x_nostarts}} have fewer than 8 sampled starts and {{x_lowrush}} fewer than 100 rushes; {{x_pool}} remain. Outcome: career pressures per rush. **Baseline without tracking:** draft pick, weight and role, fitted on the other two classes; a stronger one adds the official 40 time and season NGS get-off.

## Finding 1: the Combine start does not carry over

Raw, the 40 ranks NFL speed at 1.5 s at {{tr_raw_15}}, but that is weight. After weight and role, the 40 first step previews the NFL first step at ρ = {{tr_fs}} [{{tr_fs_lo}}, {{tr_fs_hi}}] (n = {{n_tr}}). Noise is not the excuse: with two reps averaged the most a perfect link could show is {{ceil_2rep}}, and a true translation of 0.5 or more has posterior probability {{tr_post05}}. No per-player similarity score (gap, speed ratios, curve shape) exceeds {{sim_absmax}} against pressure beyond pick.

DL is the clearest of five groups screened (Figure 2C). Receivers' NFL starts are readable (reliability {{g_WR_rel}}) but do not pay ({{g_WR_pay}}); OL, DB and TE starts are too noisy from about 20 snaps ({{g_OL_rel}} to {{g_DB_rel}}).

{{TABLE:ledger}}

*Table 1. Pressure-rate points per sd faster (DL average about 8%), 90% interval, each class held out in turn.*

![Figure 2. Translation and payoff tenth by tenth (A), by held-out class (B) and by position group (C).](figures/fig2_translation_heldout.png)

## Finding 2: two signals that agree beat the forecast

The official 40 time and the NFL first step both carry (Table 1), so we ask: does the speed he showed in Indianapolis show up when he leaves his stance?

**The rule.** *Both* = NFL first step in the top third and 40 time in the top half, each class's cut-offs set on the other two classes. A rusher "hits" if his pressure rate beats the held-out forecast (base rate {{sl_base}}%).

{{TABLE:robust}}

*Table 2. Five checks on one shortlist. Gap interval: rushers resampled within class; p: labels shuffled within class.*

**Verdict: a real but modest edge in the full pool, unproven for day-3 picks (three shortlisted).** The gap narrows in every stricter check and stays positive. Row 5 is the cleanest test: the call is frozen after 2024 from 2023–24 starts only and scored on 2025 against a forecast that knows each rusher's own past pressure. As a slope beyond that persistence forecast, the frozen first step adds +{{tf_pp}} points per sd [{{tf_pp_lo}}, {{tf_pp_hi}}]: right direction, not proven. Survivorship does not drive it: the first step does not predict who reaches 100 rushes, and reweighting for who does leaves the gap at {{ipw_gap}} points [{{ipw_gap_lo}}, {{ipw_gap_hi}}].

![Figure 3. A: rushers against held-out cut-offs; top right = shortlist. B: the five checks of Table 2.](figures/fig3_shortlist.png)

## Why it works: the same rusher, faster snap

On {{sn_n}} random rushes by {{sn_players}} linemen, a first step one sd faster than *his own* average multiplies that snap's odds of pressure by {{sn_or}} [{{sn_or_lo}}, {{sn_or_hi}}]: {{sn_top5}}% for his fastest fifth of starts, {{sn_bot5}}% for his slowest. It holds with that snap's NGS get-off, sideways path, down and alignment fixed ({{sn_or_ctl}}). The 40 says the speed exists; the NFL starts say he uses it from a stance.

![Figure 4. Same rusher, faster snap: pressure by first-step fifth; odds ratios by control, class, role.](figures/fig4_snap_mechanism.png)

## A worked decision: two day-3 edges, summer 2025

Cincinnati's Cedric Johnson and Washington's Javontae Jean-Baptiste: 2024 edge rushers picked eight slots apart, both on ${{wc_cj_apy}}M-a-year rookie deals. The call uses 2024 data only.

| End of 2024 | Cedric Johnson (pick {{wc_cj_pick}}) | Javontae Jean-Baptiste (pick {{wc_jjb_pick}}) |
|---|---|---|
| 40 time, percentile | {{wc_cj_t}} | {{wc_jjb_t}} |
| 2024 sampled starts; call frozen after 2024 | {{wc_cj_n24}}; **Both** | {{wc_jjb_n24}}; no call (middle first step, slow 40) |
| 2024 pressure rate (rushes) | {{wc_cj_rate24}}% ({{wc_cj_rs24}}) | {{wc_jjb_rate24}}% ({{wc_jjb_rs24}}) |
| **2025 result** | {{wc_cj_rate25}}% ({{wc_cj_pr25}} pressures, {{wc_cj_g25}} games) | {{wc_jjb_rate25}}% ({{wc_jjb_pr25}} pressure, {{wc_jjb_g25}} games) |

*Table 3. Percentiles within the {{n_pay}}-rusher pool.*

**Action:** give Johnson the passing-down rotation before shopping for a veteran. The rule's held-out edge is about {{wc_rule_season}} pressures a season; 2025's 21st to 40th richest veteran edge contracts cost a median ${{vet_apy}}M a year (${{vet_lo}}M to ${{vet_hi}}M). Johnson's {{wc_cj_rs24}} rookie rushes alone would not justify that; the agreeing signals do. Jean-Baptiste's 3-game 2025 cannot score the call either way.

**Where rule and tape disagree.** Maason Smith (Jacksonville, 2024, pick 48) has the pool's best NFL first step, a {{b_smith_t}}th-percentile 40 and {{b_smith_rate}}% pressure against a {{b_smith_fc}}% forecast. He arrives fast and does not win: grade hands and a counter before cutting his snaps. **The shortlist miss:** Jordan Burch (Arizona, 2025), {{b_burch_rate}}% against {{b_burch_fc}}% after one season.

## The 2026 list

The same rule, cut-offs from all three classes, applied to every lineman with 8+ sampled starts: the order for passing-down snaps.

{{TABLE:list26}}

*Table 4. Percentiles within the pool; forecast = pick, weight, role. Games missed = no recorded snap (injury, inactive or cut). Ty Robinson's 55 rushes are too few to judge; Abdullah is an availability question first.*

## Limitations

- **Not blind.** The two-signal form was chosen in an earlier round after seeing all three classes; only the cut-offs are held out. The time-forward row has {{tf_n}} rushers and overlaps the main shortlist; 2026 is the first true test.
- **Small lists.** Ten rushers, nine from 2023–24, three on day 3. One new class gives the slope test {{pow_cls}}% power.
- **Translation itself is a null.** Tracking contributes the NFL first step; the Combine contributes the stopwatch.
- **Sampled tape, no blockers or ball.** A slow start can be a two-gap call, a fast one a free rush; injuries enter only as missed games.

## Appendix

**Table 1 test.** The two-class forecast is a logit offset in a quasi-binomial model of held-out pressures per rush; class slopes pooled by random effects; p shuffles the measure within class: tracked 40 start {{ht_c40_p}}, carry-over {{ht_carry_p}}, official 40 {{x_f40_pick_p}}, NFL first step {{ht_nfl_p}} ({{x_nfl_full_p}} with the stronger baseline). Best of {{ht_nwin}} windows under the same shuffle: p = {{ht_nfl_p_adj}}. **Rusher random intercept.** On {{mm_seasons}} player-seasons, Gauss-Hermite binomial model with the forecast as offset: {{mm_pp}} points per sd [{{mm_lo}}, {{mm_hi}}], rusher sd {{mm_sd}} on the logit scale. **Selection.** P(in pool) from pick, weight, 40 time and class (AUC {{sel_auc}}); among the {{sel8_n}} with 8+ starts, first step odds ratio {{sel_fs_or}} (p = {{sel_fs_p}}). Stabilized inverse-probability weights (capped at the 95th percentile): slope {{ipw_pp}} [{{ipw_lo}}, {{ipw_hi}}]; shortlist {{ipw_hs}}% vs {{ipw_ho}}%. **Calibration.** Told from the other two classes, the shortlist expected {{calib_ps}}% hits and got {{calib_os}}%; the rest expected {{calib_po}}% and got {{calib_oo}}%; Brier {{brier_rule}} against {{brier_base}} for the base rate. **Measurement.** Tracked 40 first-step repeatability {{rel40_fs}} (n = {{rel40_n}}), {{rel40_2rep}} for two reps.

*Reproduce: `uv run python writeup/candidates/round_15/cand_3/analysis.py` (about 3 minutes).*
