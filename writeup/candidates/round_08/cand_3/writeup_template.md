# The 40 Doesn't Travel: A Pass Rusher's Combine Start Reaches Sunday Only Through His Weight

**We laid each 2023–25 defensive lineman's Combine 40 start over his own NFL pass-rush starts, tenth of a second by tenth. Raw, the 40 previews his NFL speed at 1.5 s (ρ = {{tr_raw_15}}). After weight and role it previews nothing ({{tr_adj_15}}), and translation explains no pressure beyond draft slot. Grade the start from tape.**

| Question ({{n_tr}} DL with a tracked 40 and 8+ random NFL starts) | Answer |
|---|---|
| Does his 40 start show up in his NFL start, beyond weight and role? | No: ρ {{tr_fs}} [90%: {{tr_fs_lo}}, {{tr_fs_hi}}] for the first step |
| Does the carry-over explain pressure beyond pick, weight and role? | No: {{sim_carry}} [{{sim_carry_lo}}, {{sim_carry_hi}}] in sample; it lowers R² on held-out classes |
| What does carry signal? | His NFL first step, weakly: top third beat their draft-slot forecast {{cal_top_pct}}% of the time, bottom third {{cal_bot_pct}}% |

## Why this question

1. **The scouting habit.** "Explosive 40, he'll win off the ball" is a common read on day-2 and day-3 rushers: the 40's first steps are the Combine's closest thing to a get-off.
2. **Why now.** Since 2023 the Combine is tracked at 10 Hz, like NFL games. For the first time a scout can put a player's Combine rep and his NFL snaps on one time axis and ask whether the same movement carried over.
3. **The decision.** If the 40 start travels, it should move a rusher up the board; if not, the board should wait for tape.

## The metric: Carry-over

**First step** is a player's mean speed 0.3–0.5 s after he first moves. In that window the 40 and a pass rush are the same movement: a lineman leaving a stance. After 0.5 s they part ways. By 1.5 s the median DL is at {{dr_med_40}} yd/s in the 40 but {{dr_med_sunday}} yd/s on a rush, because he has met a blocker (Figure 1).

**Carry-over = NFL first step minus Combine first step**, each in standard deviations after weight and NFL role (EDGE share), so heavy interior players are not penalized for being heavy. Positive means he starts faster on Sundays than his 40 promised.

**Data.** Combine side: every tracked 40 rep, year-standardized. NFL side: only the random half of the game sample (the top-EPA half oversamples sacks), pass plays where he was set at the snap. Each start is adjusted for alignment: {{n_snaps_dl}} starts, median {{snaps_med}} per player. Outcome: career regular-season pressures per pass rush, 2023–25, for rushers with 100+ rushes, shrunk toward the mean for small samples.

**Is it measuring something real?**

- The NFL first step averaged over {{snaps_med}} starts has reliability {{rel_fs_game}}. On single snaps it tracks the official NGS get-off time at ρ = {{rho_fs_ngs_snap}} (n = {{n_ngs_snap}}; faster first step, quicker get-off).
- The tracked 40 first step repeats at {{rel40_fs}} between a player's two reps (n = {{rel40_n}}).

**Baseline without tracking:** log draft pick (UDFA = 262), weight and EDGE share.

![Figure 1. Four rushers: Combine 40 start (dashed) against his own random NFL starts (thin blue; thick is the mean), on one time axis from first movement. Yellow band: the first step. Bars: career pressure rate against the no-tracking forecast fit on the other two classes.](figures/fig1_four_rushers.png)

## Finding 1: the 40 translates only through size

![Figure 2. Rank correlation by tenth of a second, 90% bootstrap bands. Grey dashed: tracked 40 speed vs NFL speed at the same moment, raw. Orange: the same after weight and role. Blue: NFL speed vs pressure rate after pick, weight and role.](figures/fig2_translation_payoff.png)

- **Raw, it looks like translation.** The 40 at 1.5 s ranks NFL speed at 1.5 s at {{tr_raw_15}} [{{tr_raw_15_lo}}, {{tr_raw_15_hi}}].
- **It is weight.** Heavier linemen are slower at 1.5 s on Sundays (ρ = {{rho_w_dr_sunday}}) and in the 40 ({{rho_w_dr_40}}). After weight and role the link is {{tr_adj_15}} [{{tr_adj_15_lo}}, {{tr_adj_15_hi}}], and no tenth from 0.3 to 1.5 s exceeds {{tr_adj_absmax}}.
- **Not measurement noise.** The upper 90% bound on the first-step link, {{tr_fs_hi}}, becomes {{tr_fs_ceiling}} after correcting for unreliability on both sides. Even that optimistic case leaves most of the NFL start unexplained.
- **The closest drill does no better.** The pass-rush drill's first step gives {{tr_prd}} (n = {{tr_prd_n}}).
- **The stopwatch does better than the trace.** After weight and role, the official 40 *time* ranks the NFL first step at {{tr_off_03}} [{{tr_off_03_lo}}, {{tr_off_03_hi}}], and the tracked first step at {{tr_adj_03}}. Whatever travels is overall speed, not the shape of the start.

**Other groups.** With the same pipeline, no group's 40 start survives size, and DL is the only group where the NFL first step leans toward production (Table 1). OL, DB and TE starts are too noisy on 20 random snaps (reliability ≤ {{g_DB_rel}}) to read.

{{TABLE:groups}}

*Table 1. Same metric for every group, random NFL snaps only. Correlations are Spearman; "after size" removes weight (and EDGE share for DL); production is beyond log pick and weight.*

## Finding 2: carry-over does not explain pressure beyond pick

We tried five per-player similarity measures, none chosen on pressure: carry-over, two speed ratios, curve-shape agreement, curve distance. Against pressure beyond pick, weight and role, none exceeds {{sim_absmax}} in absolute value. Carry-over itself is {{sim_carry}} [{{sim_carry_lo}}, {{sim_carry_hi}}] (n = {{n_both}}).

![Figure 3. Each rusher's Combine first step (horizontal) against his NFL first step (vertical), both after weight and role. Colour: pressure rate minus the out-of-class forecast. Corner text: pooled pressure rate, forecast and share beating the forecast for each quadrant.](figures/fig3_carryover_map.png)

**The vertical axis carries what signal there is.** Rushers above the line beat their forecast {{half_up_beat}}% of the time (n = {{half_up_n}}), below it {{half_dn_beat}}% (n = {{half_dn_n}}). Split left/right by the 40 instead: {{half_cr_beat}}% vs {{half_cl_beat}}%. The NFL first step alone gives partial ρ {{pay_all}} [{{pay_all_lo}}, {{pay_all_hi}}] on all {{n_pay}} rushers. One sd is worth {{pp_per_sd}} points of pressure rate [90%: {{pp_per_sd_lo}} to {{pp_per_sd_hi}}], about {{pr_per_season}} pressures in a {{rush_season}}-rush season.

**Why that is still not a finding:**

- **Teams already have it.** The season-long NGS get-off, measured on every snap, gives {{pay_ngs}} [{{pay_ngs_lo}}, {{pay_ngs_hi}}]. Beyond it our 20-start first step adds {{pay_beyond_ngs}}.
- **We picked the window.** Scanning all {{n_windows}} tenths and re-running with pressure shuffled within class, the best tenth beats chance with p = {{p_maxt}}.
- **It fails on new classes** (Figure 4). Fit on two classes and scored on the third, the NFL first step changes rush-weighted R² by {{ooc_sunday_all}} [{{ooc_sunday_all_lo}}, {{ooc_sunday_all_hi}}] on a baseline of {{ooc_base_r2}}. Carry-over gives {{ooc_carry}}, the Combine first step {{ooc_combine}}. 2024 drives the loss ({{ooc_sunday_all_2024}}), led by Maason Smith (Table 2).
- **Calibration is better but thin.** Against the out-of-class forecast, the top third by NFL first step beat it {{cal_top_k}} of {{cal_top_n}} times ({{cal_top_pct}}%, exact 90% interval {{cal_top_lo}}–{{cal_top_hi}}%), the bottom third {{cal_bot_k}} of {{cal_bot_n}} ({{cal_bot_lo}}–{{cal_bot_hi}}%); base rate {{beat_base}}%. In 2025 alone the top third went {{cal25_k}} of {{cal25_n}} ({{cal25_lo}}–{{cal25_hi}}%).

![Figure 4. Fit on two classes, score the third. Left: pooled rush-weighted R² change over pick, weight and role, 90% interval from resampling held-out rushers. Right: each held-out class, with n for all rushers and for those with a 40.](figures/fig4_out_of_class.png)

**EDGE vs interior.** The 40 start leans toward translating for interior linemen ({{role_i_tr}}, n = {{role_i_tr_n}}) and away for EDGEs ({{role_e_tr}}, n = {{role_e_tr_n}}). The NFL first step leans positive in both: interior {{role_i_pay}} [{{role_i_pay_lo}}, {{role_i_pay_hi}}], top third {{role_i_top}}% vs bottom third {{role_i_bot}}% pressure (n = {{role_i_n}}); EDGE {{role_e_pay}} [{{role_e_pay_lo}}, {{role_e_pay_hi}}], {{role_e_top}}% vs {{role_e_bot}}% (n = {{role_e_n}}). Exploratory: the split was not pre-specified.

## Named rushers

Figure 1 shows one rusher per quadrant:

- **YaYa Diaby** (fast in both): his 40 start shows up on tape, and he pressures at {{h_fast_both_rate}}% against a forecast of {{h_fast_both_fc}}%.
- **Byron Young** (NFL-only fast): 40 start at percentile {{h_sunday_only_c}}, NFL first step at percentile {{h_sunday_only_s}}, and {{h_sunday_only_rate}}% vs {{h_sunday_only_fc}}%. The 40 undersold him.
- **Eric Watts** (Combine-only fast): 40 start at percentile {{h_combine_only_c}}, the slowest NFL first step in the cohort, {{h_combine_only_rate}}% vs {{h_combine_only_fc}}%.
- **Brenton Cox** (slow in both) is the counter-example: {{h_slow_rate}}% vs {{h_slow_fc}}% on 201 rushes with no start edge at all. A first step is one way to win, not the only one.

**Decision card.** One plain call per rusher, from the NFL first step and, where present, the 40. 2023–24 rows are the classes we looked at while building this; 2025 rows are the cleanest check.

{{TABLE:card}}

*Table 2. Percentiles among the {{n_board}} rushers with 100+ rushes, after weight and role. Forecast = no-tracking model fit on the other two classes. Calls: top third NFL first step = credit it (or "trust the tape" if his 40 start was below median); bottom third with an above-median 40 start = film check.*

Maason Smith is the cautionary row: the best NFL first step in the cohort and {{b_smith_rate}}% against {{b_smith_fc}}%. A fast start without a finish is not a pass rush.

## What a team can use

1. **Draft room.** Do not read a fast 40 start as get-off. Once weight and role are priced, the tracked start adds nothing; if you use the 40, use the time.
2. **Pro personnel.** For a young rusher, the NFL first step from about 20 random starts is a tie-break at most, and NGS get-off already covers it.
3. **Position coach.** A "Combine-only fast" rusher (Watts, Umanmielen, Pickens) is a film question: stance, alignment, snap anticipation, or role. The drill says the burst exists; the tape says it is not being used.

## Limitations

- **Mostly nulls.** Translation is near zero after size, and nothing holds out of class. DL was chosen as the best of five groups.
- **Small, sampled tape.** About 20 random NFL starts per player and 52 rushers with a 40, pressure rate and NFL starts together; the 2025 class has one season.
- **No context.** The game tracking has no ball, blockers or opponents, so a slow first step can be a stunt, a two-gap call or a slide toward him; scheme is unmodelled.
- **Construct.** The 40 has no snap. We anchor both sides at first movement, so snap reaction is excluded by design.
- **Outcome.** Pressure is charted and usage-dependent; injuries and rotation are not modelled.

*Reproducible: `uv run python writeup/candidates/round_08/cand_3/analysis.py` (about 30 s) writes every figure, table and number (`numbers.json`; `claims.json` maps each key to its sentence) and renders this file.*
