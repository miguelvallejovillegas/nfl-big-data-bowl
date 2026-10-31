# Combine Route-Drill Cuts Do Not Carry to NFL Separation

**A receiver who cuts 1 sd harder in Combine route drills is no more open in the NFL: −1.2 inches of separation over expected per sd (90% interval −3.8 to +1.0), across 116 receivers from draft classes the model never saw.**

*Break G, in one line:* how hard a receiver pushes into the ground in the one second around his first cut, as a share of body weight (1 sd = 0.021 g). On film, read it as the speed he sheds into the plant: the two rank-correlate 0.70 over 885 drill reps.

Why it matters: clubs already pay for it. A receiver 1 sd harder in the route drills goes 9% earlier in the draft [1, 17] at the same position (149 receivers), and the 2023 class's rookie deals expire after this season.

- **Separation over expected (SOE)**: inches more open at the throw than Next Gen Stats separation predicts from alignment, cushion and season, route-weighted over full regular seasons (213 player-seasons, median 231 routes).
- **Combine years**: every rep is z-scored within drill and year, so protocol changes between Combines do not leak in.
- **Samples**: 149 wide receivers and tight ends with a drill score; 116 of them with NFL separation (season tests); 92 of those with 8+ graded NFL breaks and an official 40 (transfer tests).

![Figure 1. Six of 49 wide receivers (200+ career routes, 5+ graded NFL breaks) at evenly spaced ranks of drill Break G, hardest top left. Thin lines: each Combine rep (vermilion) and each sampled NFL break (blue) on route families he ran in both settings, aligned on the slowest point of the cut; bold: his median; shaded: the graded second.](figures/fig1_cuts.png)

## 1. Break G adds nothing to a no-tracking forecast

The baseline, fixed before the test, uses log pick, official 40, weight, NGS athleticism score and a tight-end flag. It is fitted on two draft classes and scored on the third. All intervals are 90%, from 2,000 class-stratified bootstrap refits.

| Held-out test | Estimate [90%] | n |
|---|---|---|
| No-tracking baseline R² (error) | 0.00 (8.6 in) | 116 receivers |
| Change in R² from adding Break G | 0.013 [−0.141, 0.079] | 116 receivers |
| **Break G slope, inches per sd** | **−1.2 [−3.8, +1.0]** | 116 receivers |
| Same, weighted for who reached the NFL | −1.1 [−3.3, +0.9] | 116 of 149 |

A drill score this noisy (reliability 0.50) could hide a real effect, so each receiver's score is shrunk by his own reliability and the slope is per sd of *true* Break G. Power is the honest limit: 68% at ±3 in; 80% only for 3.5 in. Every Break G estimate sits below zero, and none of the 8 season-level slopes in Figure 2 excludes zero, so no multiplicity correction is needed to read the null. Inclusion weights matter little, since draft slot decides who logs NFL routes and the baseline already holds pick.

The no-tracking benchmark fails the same way: official 3-cone −0.5 [−6.3, +3.5] (38 receivers) and short shuttle −1.0 [−4.4, +2.3] (45). Cut quality as the Combine measures it, tracked or timed, does not price separation.

![Figure 2. Held-out slopes of season SOE per sd, 90% class-stratified bootstrap intervals, against the ±3 inch margin fixed before testing; n on each row. Vermilion: Break G (primary, inclusion-weighted, hierarchical with receiver and class effects); grey: its three parts; blue: official agility times.](figures/fig2_forest.png)

## 2. Why: the drill grades a cut the NFL does not ask for

| What is compared | Rank corr. [90%] | n |
|---|---|---|
| Official 40 → NFL top speed | 0.48 [0.35, 0.60] | 92 receivers |
| Combine → NFL Break G | −0.04 [−0.22, 0.14] | 92 receivers |
| Largest correlation the noise allows | 0.43 | 92 receivers |
| Combine → NFL Break G, same route family | −0.07 [−0.15, 0.01] | 114 receivers |

Noise does not explain the gap between rows 1 and 2: a true drill-to-game correlation of 0.5 would still show as 0.22. Official 3-cone and shuttle do no better at predicting NFL Break G (−0.01 and 0.11; Figure 3C).

The reason is entry speed. 47 of 49 wide receivers enter the drill cut faster than the same NFL route, by a median 1.4 yd/s (hitch: 6.9 against 4.7). The Combine grades a full-speed plant that Sunday routes rarely ask for. Three named receivers show it:

| Receiver (drill rank of 49) | Routes in both | Drill cut: in → slowest → out, yd/s | NFL cut: in → slowest → out, yd/s |
|---|---|---|---|
| Kayshon Boutte (#1, +1.8 sd) | corner, hitch, in, out, slant (7 reps / 16 breaks) | 7.1 → 4.5 → 5.9 | 5.7 → 5.0 → 5.9 |
| Ryan Flournoy (#25, +0.3 sd) | corner, hitch, in, out, slant (9 reps / 18 breaks) | 7.2 → 3.9 → 6.5 | 4.7 → 2.4 → 4.8 |
| Jake Bobo (#49, −1.4 sd) | corner, hitch, in, out, slant (7 reps / 25 breaks) | 6.7 → 3.1 → 5.6 | 5.0 → 3.9 → 5.5 |

Kayshon Boutte, the hardest drill cutter, barely slows in his NFL breaks (5.7 → 5.0 yd/s). Ryan Flournoy, ranked in the middle, sheds the most of the three on Sundays. Career SOE does not follow the drill order either: the eight hardest and eight softest drill cutters average +1.0 and −2.3 inches.

![Figure 3. (A, B) The same 92 receivers: official 40 against NFL top speed, and Combine against NFL Break G, with least-squares line and 90% bootstrap band. (C) Drill-to-NFL Break G by protocol subset and for official agility times, receiver-clustered 90% intervals; dashed: the correlation a true 0.5 would show through both measures' noise.](figures/fig3_transfer.png)

## 3. A drill protocol a Combine staff can adopt

- **More reps.** Drills are interchangeable for reliability, so total graded cuts is what counts: today's median of 7 gives 0.53, two reps in each of 7 drills give 0.69, and 15 cuts reach 0.70. Reps alone do not fix transfer, though: drills run twice give −0.12 [−0.36, 0.10] (43 receivers).
- **NFL entry speed.** Only 35% of drill reps are entered within 1 yd/s of the NFL speed for that route, and just 4 of 265 hitch-family reps (curl, comeback, hook). The matched reps, mostly slants, give −0.05 [−0.16, 0.07] (100 receivers).

The protocol: grade at least 15 cuts per receiver; keep the slant and re-stem the curl, comeback and hook at NFL hitch speed (about 4.7 yd/s, a controlled three-quarter stem); and give Break G zero weight until a re-stemmed drill transfers.

## 4. Day 3: no separation shortcut either

Over 38 day-3 or undrafted season pairs (28 receivers, 50+ routes each season, forecast leave-one-class-out), the top half by last season's SOE stayed above zero 42% of the time, against 63% for the top half by pick (−21 points [−43, 0], receiver-clustered). That misses the bar we set before the pooled run (10+ points with the lower bound above zero), so we publish no separation-ordered list. Draft order stands, and the 2025 names sit in the appendix as a coin-flip film aid only.

## Bottom line

1. **Take Break G out of the price.** Clubs pay 9% in pick position per sd of it and get no separation back.
2. **Fix the drill before grading it.** Grade 15+ cuts, re-stem the hitch family at NFL entry speed, and give it zero weight until it transfers.
3. **On day 3, the draft order stands.** Last season's separation does not beat it.

## Limitations

- **Not blind.** All three classes were seen while this was built. The ±3 inch margin, the baseline and the day-3 bar were fixed before each test.
- **Power.** 68% at the margin, and harm is not excluded (lower bound −3.8 inches per sd).
- **Selection.** 116 of 149 drill-scored receivers logged NFL routes. Inclusion weights move the slope to −1.1, but weights cannot repair receivers who never played.
- **Game tracking is a sample.** Up to 40 plays per player, half top-EPA, with no defenders or ball. This limits Tables 2 and 3 and Figures 1 and 3, not the season tests, which use full-season NGS.
- **Small benchmarks.** Few receivers run the official 3-cone (38) or shuttle (45).
- **The film proxy is a correlate.** Speed shed into the plant tracks Break G (0.70) but is not the same number.

## Appendix

`verify.py` rebuilds both Break G pipelines from the raw frames and refits the held-out model a second way (statsmodels). With its own bootstrap seeds it reproduces 100 of 100 checked numbers: 55 exactly, 45 resampled within 15% of their interval width. `appendix.md` holds the equivalence tests, measurement-error checks, the 2025 day-3 names with what flips each one, and their man/zone and slot/wide splits with cell n.
