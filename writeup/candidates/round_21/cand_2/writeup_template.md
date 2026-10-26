# Combine Route-Drill Cuts Do Not Carry to NFL Separation

**A receiver who cuts 1 sd harder in Combine route drills is no more open in the NFL: {ho} inches of separation over expected per sd (90% interval {ho_lo} to {+ho_hi}), across {n_main} receivers from draft classes the model never saw.**

*Break G, in one line:* how hard a receiver pushes into the ground in the one second around his first cut, as a share of body weight (1 sd = {g_sd} g). On film, read it as the speed he sheds into the plant: the two rank-correlate {proxy_rho} over {proxy_n} drill reps.

Why it matters: clubs already pay for it. A receiver 1 sd harder in the route drills goes {price}% earlier in the draft [{price_lo}, {price_hi}] at the same position ({price_n} receivers), and the 2023 class's rookie deals expire after this season.

- **Separation over expected (SOE)**: inches more open at the throw than Next Gen Stats separation predicts from alignment, cushion and season, route-weighted over full regular seasons ({n_seasons} player-seasons, median {routes_med} routes).
- **Combine years**: every rep is z-scored within drill and year, so protocol changes between Combines do not leak in.
- **Samples**: {n_combine} wide receivers and tight ends with a drill score; {n_main} of them with NFL separation (season tests); {f1_n} of those with 8+ graded NFL breaks and an official 40 (transfer tests).

![Figure 1. Six of {ex_n} wide receivers (200+ career routes, 5+ graded NFL breaks) at evenly spaced ranks of drill Break G, hardest top left. Thin lines: each Combine rep (vermilion) and each sampled NFL break (blue) on route families he ran in both settings, aligned on the slowest point of the cut; bold: his median; shaded: the graded second.](figures/fig1_cuts.png)

## 1. Break G adds nothing to a no-tracking forecast

The baseline, fixed before the test, uses log pick, official 40, weight, NGS athleticism score and a tight-end flag. It is fitted on two draft classes and scored on the third. All intervals are 90%, from 2,000 class-stratified bootstrap refits.

| Held-out test | Estimate [90%] | n |
|---|---|---|
| No-tracking baseline R² (error) | {bl_r2} ({bl_rmse} in) | {n_main} receivers |
| Change in R² from adding Break G | {dr2} [{dr2_lo}, {dr2_hi}] | {n_main} receivers |
| **Break G slope, inches per sd** | **{ho} [{ho_lo}, {+ho_hi}]** | {n_main} receivers |
| Same, weighted for who reached the NFL | {ipw} [{ipw_lo}, {+ipw_hi}] | {n_main} of {inc_all} |

A drill score this noisy (reliability {rel_main}) could hide a real effect, so each receiver's score is shrunk by his own reliability and the slope is per sd of *true* Break G. Power is the honest limit: {pow3}% at ±{sesoi} in; 80% only for {mde} in. Every Break G estimate sits below zero, and none of the {n_tests} season-level slopes in Figure 2 excludes zero, so no multiplicity correction is needed to read the null. Inclusion weights matter little, since draft slot decides who logs NFL routes and the baseline already holds pick.

The no-tracking benchmark fails the same way: official 3-cone {a_cone} [{a_cone_lo}, {+a_cone_hi}] ({a_cone_n} receivers) and short shuttle {a_shut} [{a_shut_lo}, {+a_shut_hi}] ({a_shut_n}). Cut quality as the Combine measures it, tracked or timed, does not price separation.

![Figure 2. Held-out slopes of season SOE per sd, 90% class-stratified bootstrap intervals, against the ±3 inch margin fixed before testing; n on each row. Vermilion: Break G (primary, inclusion-weighted, hierarchical with receiver and class effects); grey: its three parts; blue: official agility times.](figures/fig2_forest.png)

## 2. Why: the drill grades a cut the NFL does not ask for

| What is compared | Rank corr. [90%] | n |
|---|---|---|
| Official 40 → NFL top speed | {spc} [{spc_lo}, {spc_hi}] | {f1_n} receivers |
| Combine → NFL Break G | {bgc} [{bgc_lo}, {bgc_hi}] | {f1_n} receivers |
| Largest correlation the noise allows | {ceil} | {f1_n} receivers |
| Combine → NFL Break G, same route family | {fam} [{fam_lo}, {fam_hi}] | {fam_np} receivers |

Noise does not explain the gap between rows 1 and 2: a true drill-to-game correlation of 0.5 would still show as {exp50}. Official 3-cone and shuttle do no better at predicting NFL Break G ({tr_cone} and {tr_shut}; Figure 3C).

The reason is entry speed. {ex_faster} of {ex_shared} wide receivers enter the drill cut faster than the same NFL route, by a median {vin_gap} yd/s (hitch: {vin_hitch_d} against {vin_hitch_n}). The Combine grades a full-speed plant that Sunday routes rarely ask for. Three named receivers show it:

{{CARDS}}

Kayshon Boutte, the hardest drill cutter, barely slows in his NFL breaks ({c_boutte_nin} → {c_boutte_nap} yd/s). Ryan Flournoy, ranked in the middle, sheds the most of the three on Sundays. Career SOE does not follow the drill order either: the eight hardest and eight softest drill cutters average {+x8_hard_soe} and {x8_soft_soe} inches.

![Figure 3. (A, B) The same {f1_n} receivers: official 40 against NFL top speed, and Combine against NFL Break G, with least-squares line and 90% bootstrap band. (C) Drill-to-NFL Break G by protocol subset and for official agility times, receiver-clustered 90% intervals; dashed: the correlation a true 0.5 would show through both measures' noise.](figures/fig3_transfer.png)

## 3. A drill protocol a Combine staff can adopt

- **More reps.** Drills are interchangeable for reliability, so total graded cuts is what counts: today's median of {why_reps} gives {rel_7x1}, two reps in each of 7 drills give {rel_7x2}, and {cuts70} cuts reach {t70}. Reps alone do not fix transfer, though: drills run twice give {two} [{two_lo}, {two_hi}] ({two_np} receivers).
- **NFL entry speed.** Only {match_pct}% of drill reps are entered within 1 yd/s of the NFL speed for that route, and just {match_hitch} of {n_hitch} hitch-family reps (curl, comeback, hook). The matched reps, mostly slants, give {match} [{match_lo}, {match_hi}] ({match_np} receivers).

The protocol: grade at least {cuts70} cuts per receiver; keep the slant and re-stem the curl, comeback and hook at NFL hitch speed (about {vin_hitch_n} yd/s, a controlled three-quarter stem); and give Break G zero weight until a re-stemmed drill transfers.

## 4. Day 3: no separation shortcut either

Over {p_n} day-3 or undrafted season pairs ({p_np} receivers, 50+ routes each season, forecast leave-one-class-out), the top half by last season's SOE stayed above zero {p_soe}% of the time, against {p_pick}% for the top half by pick ({p_d} points [{p_d_lo}, {p_d_hi}], receiver-clustered). That misses the bar we set before the pooled run ({bar}+ points with the lower bound above zero), so we publish no separation-ordered list. Draft order stands, and the 2025 names sit in the appendix as a coin-flip film aid only.

## Bottom line

1. **Take Break G out of the price.** Clubs pay {price}% in pick position per sd of it and get no separation back.
2. **Fix the drill before grading it.** Grade {cuts70}+ cuts, re-stem the hitch family at NFL entry speed, and give it zero weight until it transfers.
3. **On day 3, the draft order stands.** Last season's separation does not beat it.

## Limitations

- **Not blind.** All three classes were seen while this was built. The ±3 inch margin, the baseline and the day-3 bar were fixed before each test.
- **Power.** {pow3}% at the margin, and harm is not excluded (lower bound {ho_lo} inches per sd).
- **Selection.** {n_main} of {inc_all} drill-scored receivers logged NFL routes. Inclusion weights move the slope to {ipw}, but weights cannot repair receivers who never played.
- **Game tracking is a sample.** Up to 40 plays per player, half top-EPA, with no defenders or ball. This limits Tables 2 and 3 and Figures 1 and 3, not the season tests, which use full-season NGS.
- **Small benchmarks.** Few receivers run the official 3-cone ({a_cone_n}) or shuttle ({a_shut_n}).
- **The film proxy is a correlate.** Speed shed into the plant tracks Break G ({proxy_rho}) but is not the same number.

## Appendix

`verify.py` rebuilds both Break G pipelines from the raw frames and refits the held-out model a second way (statsmodels). With its own bootstrap seeds it reproduces 100 of 100 checked numbers: 55 exactly, 45 resampled within 15% of their interval width. `appendix.md` holds the equivalence tests, measurement-error checks, the 2025 day-3 names with what flips each one, and their man/zone and slot/wide splits with cell n.
