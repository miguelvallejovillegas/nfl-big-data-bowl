# Speed That Translates: Rushers Fast on the Stopwatch and on Sundays Beat Their Draft Slot

**Pool, used throughout:** {{n_pay}} defensive linemen (2023–25 classes) with 100+ NFL pass rushes and 8+ random sampled NFL starts.

**We laid each lineman's tracked Combine 40 start over his own NFL pass-rush starts. After size, the start does not carry over. Agreement does: rushers fast on the official 40 *and* in about 20 NFL starts beat a draft-slot pressure forecast {{sl_ks}} times in {{sl_ns}}, against {{sl_ko}} in {{sl_no}} for the rest, on held-out classes.**

![Figure 1. Four rushers, one scale: 40 start (orange) over his NFL starts (blue).](figures/fig1_four_rushers.png)

## Why it matters

A club bets on a young rusher's speed twice: on draft day from a stopwatch, and a year later when it hands out third-down snaps and opens extension talks. The second bet is usually made on sacks, which are rare and noisy. After about 20 random rushes, tracking shows whether the speed appears when he leaves a stance. We test whether that read, with the 40, says something draft slot and the box score do not.

## So what for a team

1. **Pro personnel: keep a "speed that translates" shortlist** (top-third NFL first step and top-half 40 time). On held-out classes it produced +{{sl_exs_s}} pressures a season over forecast; the rest of the pool, {{sl_exs_o}}. The 2026 list is Table 4.
2. **Draft room: pay for the stopwatch, not the trace.** The official 40 adds pressure beyond draft slot; the tracked 40 start adds nothing.
3. **Tape: a fast start alone is not a buy.** A top NFL first step without the 40 time is a hands-and-counter question (Maason Smith, below).

## The metric

**First step = how fast a lineman is moving 0.3 to 0.5 s after he first moves.** In that window the 40 and a pass rush are the same movement: a big man leaving a stance. By 1.5 s they part (median {{dr_med_40}} yd/s in the 40, {{dr_med_sunday}} on a rush, because he has met a blocker). Both sides are in sd after weight and role (share of snaps at EDGE). **Translation** = the Combine number previewing the NFL number.

**Data.** Every tracked Combine 40 rep, year-standardized. NFL: random sampled pass plays only, alignment-adjusted: {{n_snaps_dl}} starts by {{n_gplayer}} linemen, median {{snaps_med}} each, reliability {{rel_fs_game}}; it tracks NGS get-off snap by snap (ρ = {{rho_fs_ngs_snap}}). Outcome: career pressures per rush. **Baseline without tracking:** draft pick, weight and role, fit on the other two classes; a stronger one adds the official 40 time and season NGS get-off.

## Finding 1: the Combine start does not translate

Raw, the 40 ranks NFL speed at 1.5 s at {{tr_raw_15}}, but that is weight (ρ = {{rho_w_dr_sunday}} with NFL speed, {{rho_w_dr_40}} with the 40). After weight and role, the 40 first step previews the NFL first step at ρ = {{tr_fs}} [{{tr_fs_lo}}, {{tr_fs_hi}}] (n = {{n_tr}}). Corrected for noise in both measures the point estimate is {{tr_fs_dis}}, and a true translation of 0.5 or more has posterior probability {{tr_post05}} (flat prior). The pass-rush drill start does no better ({{tr_prd}}, n = {{tr_prd_n}}), and no per-player similarity score (gap, speed ratios, curve shape) exceeds {{sim_absmax}} against pressure beyond pick.

{{TABLE:ledger}}

*Table 1. Pressure-rate points per sd faster (DL average about 8%), 90% interval, each class held out in turn.*

![Figure 2. A: translation and payoff, tenth by tenth. B: each class held out; grey = shuffled measures.](figures/fig2_translation_heldout.png)

## Finding 2: two signals that agree beat the forecast

What carries is the official 40 time and the NFL first step (Table 1). They overlap, so we ask one football question: does the speed he showed in Indianapolis show up when he leaves his stance on Sundays?

**The rule.** *Both* = NFL first step in the top third and 40 time in the top half, each class's cut-offs set on the other two classes. A rusher "hits" if his pressure rate beats the held-out forecast (base rate {{sl_base}}%). Both hits {{sl_ks}} of {{sl_ns}}, the rest {{sl_ko}} of {{sl_no}}; dropping any one shortlisted rusher leaves at least {{sl_loo_min}}%.

{{TABLE:robust}}

*Table 2. Five checks on one shortlist. Gap interval: rushers resampled within class.*

**Time-forward and persistence.** Row 5 is the cleanest test: the call is frozen after 2024 from 2023–24 starts only (median {{tf_snaps_med}} per rusher) and scored on 2025 against a forecast that knows each rusher's own 2023–24 pressure rate. A no-tracking rule, "top third of past pressure", beats the pick forecast {{tfq_ks}} of {{tfq_ns}}; the Both list beats it {{tfb_ks}} of {{tfb_ns}}, and beats the persistence forecast {{tf_ks}} of {{tf_ns}} too. As a slope, the frozen first step adds +{{tf_pp}} points per sd beyond persistence [{{tf_pp_lo}}, {{tf_pp_hi}}]: right direction, not proven. Edges shrink on sacks ({{sl_s_ks2}} of {{sl_ns}}) and in the day-3 pool (three shortlisted).

![Figure 3. A: rushers against held-out cut-offs; top right = shortlist. B: the five checks of Table 2.](figures/fig3_shortlist.png)

## Why it works: the same rusher, faster snap

On {{sn_n}} random rushes by {{sn_players}} linemen, a first step one sd faster than *his own* average multiplies the odds of pressure on that snap by {{sn_or}} [{{sn_or_lo}}, {{sn_or_hi}}], errors clustered by rusher: {{sn_top5}}% for his fastest fifth of starts, {{sn_bot5}}% for his slowest (Figure 4). It holds with that snap's NGS get-off, sideways path, down and alignment held fixed ({{sn_or_ctl}}). The 40 says the speed exists; the NFL starts say he uses it from a stance.

![Figure 4. Same rusher, faster snap: pressure by first-step fifth; odds ratios by control, class, role.](figures/fig4_snap_mechanism.png)

**Price.** Clubs already pay for the 40 (one sd faster goes {{price_40}}% earlier in pick number, n = {{price_n}}), yet it still adds pressure beyond the pick.

## The board

Percentiles within the pool; * = 2025 class. Actions are football judgement, not model output.

{{TABLE:card}}

*Table 3.*

**Where the rule and the tape disagree: Maason Smith.** Jacksonville's 2024 second-rounder has the pool's best NFL first step, a {{b_smith_t}}th-percentile 40, and {{b_smith_rate}}% pressure against a {{b_smith_fc}}% forecast; only {{b_smith_q}}% of his pressures came quickly. His curve is fast through half a second, then flat (Figure 1): he arrives, he does not win. His starts are among the straightest ({{b_smith_lat}}th percentile sideways), so stunts do not explain it. The rule withholds credit; the tape question is hands and a counter, which a coach should grade before cutting his snaps. **The shortlist miss: Jordan Burch** (2025, Arizona), {{b_burch_rate}}% against {{b_burch_fc}}% after one season.

## The 2026 list

The same rule, cut-offs from all three classes, applied to every lineman with 8+ sampled starts: the order for passing-down snaps and second-contract talks.

{{TABLE:list26}}

*Table 4. Percentiles within the pool; forecast = pick, weight, role. Ty Robinson's 55 rushes are too few to judge pressure.*

## Limitations

- **Not blind.** The two-signal form was chosen in an earlier round after seeing all three classes; only the cut-offs are held out. The time-forward row has {{tf_n}} rushers and overlaps the main shortlist; 2026 is the first true test.
- **Small lists.** Ten rushers, nine from 2023–24; intervals are wide (Table 2). One new class gives the slope test {{pow_cls}}% power.
- **Translation itself is a null.** Tracking contributes the NFL first step; the Combine contributes the stopwatch. DL was the clearest of five groups screened; for WR, OL, DB and TE the NFL start is not readable from about 20 sampled snaps.
- **Sampled tape, no blockers or ball; charted outcome.** A slow start can be a two-gap call, a fast one a free rush; pressure is usage-dependent and injuries are not modelled.

## Appendix

**Table 1 test.** The forecast, fit on two classes, is a fixed logit offset in a quasi-binomial model of the held-out class's pressures out of rushes; class slopes are pooled with random-effects shrinkage; p shuffles the measure within class 2,000 times: tracked 40 start {{ht_c40_p}}, carry-over {{ht_carry_p}}, official 40 {{x_f40_pick_p}}, NFL first step {{ht_nfl_p}} ({{x_nfl_full_p}} with the stronger baseline). The window was chosen in an earlier round; the best of {{ht_nwin}} windows under the same shuffle gives p = {{ht_nfl_p_adj}}. Resampling rushers with the forecast refit: [{{cb_lo}}, {{cb_hi}}]. **Table 2.** Label-shuffle p within class: {{rb_main_p}}, {{rb_full_p}}, {{rb_x_p}}, {{rb_d3_p}}, {{rb_tf_p}}. **Sample.** 40 first-step repeatability {{rel40_fs}} (n = {{rel40_n}}).

*Reproduce: `uv run python writeup/candidates/round_14/cand_3/analysis.py` (about 4 minutes, one core).*
