# Combine Cuts Don't Predict NFL Separation; Last Year's Separation Doesn't Beat Draft Order

**A receiver who cuts one standard deviation harder in Combine route drills is no more open in the NFL: −1.2 inches of separation per sd (90% interval −3.8 to +1.0), across 116 receivers from draft classes the model never saw.**

Why receivers: clubs already pay for the cut. A receiver 1 sd harder in the route drills goes 9% earlier in the draft [1, 17] at the same position. Why now: the 2023 class's rookie contracts run out after this 2026 season, so clubs set day-3 receiver prices this winter.

- **Break G**, in coach terms. Take the Combine curl: the receiver stems upfield, plants and comes back to the ball. Break G is how hard he pushes into the ground in the one second around that plant (sink, turn, drive out), as a share of body weight: snap the curl off and he grades high, round it and he grades low. One sd is 0.021 g. The same code grades NFL slants, outs, hitches, ins and corners from 10 Hz tracking (Figure 1, right).
- **Separation over expected (SOE)**: inches more open at the throw than Next Gen Stats separation predicts from alignment, cushion and season, over full regular seasons.
- **Combine years**: every rep is z-scored within drill and year, so protocol changes between Combines do not leak in.
- **Samples**: 149 receivers with a drill score; 116 of them with NFL separation (season tests); 92 of those with 8+ graded NFL breaks and an official 40 (transfer tests).

## 1. Break G adds nothing to season separation

The outcome is route-weighted career SOE (213 player-seasons, median 231 routes). The baseline without tracking, fixed before the test, is log pick, official 40, weight, NGS athleticism score and a tight-end flag, fitted on two draft classes and scored on the third.

| Test (held out by draft class unless noted) | Estimate [90%] | n |
|---|---|---|
| Baseline R² (error) | 0.00 (8.6 in) | 116 receivers |
| Change in R² from adding Break G | 0.013 [−0.141, 0.079] | 116 receivers |
| **Slope, inches per sd of true Break G** | **−1.2 [−3.8, +1.0]** | 116 receivers |
| Same, drill reliability redrawn in every resample | −1.2 [−3.6, +1.0] | 116 receivers |
| Same, raw drill average (no error correction) | −1.0 [−2.8, +0.8] | 116 receivers |
| Hierarchical model, receiver and class effects | −0.5 [−2.8, +1.8] | 167 player-seasons, 91 receivers |
| Smallest slope this test detects (80% power) | 3.5 in | 116 receivers |

Break G is noisy (score reliability 0.50), so the null could be attenuation. It is not: each receiver's drill average is shrunk by his own reliability (regression calibration), so the slope is per sd of true Break G, and redrawing the between-drill ICC (0.17 [0.10, 0.23]) in every resample barely widens it. The calibrated claim: this test catches a 3.5-inch effect 4 times in 5 (68% power at the 3-inch margin), and every estimate sits on the no-benefit side. None of the three mechanics (brake, speed through, drive out) carries it either.

![Figure 1. Held-out slopes per sd against the pre-set ±3 inch margin, primary row highlighted (left); one receiver's Combine curl and NFL hitch, rotated so the stem runs up, with the graded one-second window thick (right).](figures/fig1_null.png)

## 2. Why: the drill grades a cut the NFL does not ask for

| What is compared | Rank corr. [90%] | n | Reading |
|---|---|---|---|
| Official 40 → NFL top speed | 0.48 [0.35, 0.60] | 92 receivers | Speed carries |
| Combine → NFL Break G | −0.04 [−0.22, 0.14] | 92 receivers | No transfer |
| Largest correlation the noise allows | 0.43 | 92 receivers | Room for a signal |
| Combine → NFL Break G, same route family | −0.07 [−0.15, 0.01] | 114 receivers | No transfer |
| Break G, five drills vs the other five | 0.28 [0.15, 0.40] | 145 receivers | A weak trait |
| Break G, same drill run twice | ICC 0.05 | 70 pairs | One rep is mostly noise |
| NFL Break G, odd vs even snaps | 0.23 | 113 receivers | Weak in games too |

Noise does not explain the gap between rows 1 and 2. With reliabilities of 0.51 (drill) and 0.37 (NFL), the largest correlation the data could show is 0.43; a true drill-to-game correlation of 0.5 would show up as about 0.22. The observed −0.04, corrected for both, is −0.08.

![Figure 2. The same 92 receivers: official 40 against NFL top speed (A) and Combine against NFL Break G (B), least-squares line and 90% bootstrap band.](figures/fig2_transfer.png)

The reason is entry speed. Combine stems are entered a median 1.4 yd/s faster than the same NFL route (hitch: 6.9 vs 4.7 yd/s), a cut the NFL rarely asks for. Kayshon Boutte, the hardest drill cutter of 49 wide receivers (+1.8 sd), turns 0.6 yd/s faster than the league on Sundays with +0.5 inches of career SOE. The eight hardest and eight softest drill cutters average +1.0 and −2.3 inches, with the same NFL turn speed (0.0 and +0.1 yd/s).

![Figure 3. Six wide receivers at evenly spaced ranks of drill Break G, on the route families each ran in both settings: speed through the cut, median and middle 50% of his drill reps and NFL breaks, league NFL median as a grey ghost.](figures/fig3_cuts.png)

## 3. A drill protocol a Combine staff can adopt

- **More reps.** Drills are interchangeable for reliability (the receiver-by-drill share of a rep's variance is 0%), so what counts is total graded cuts: today's median of 7 gives 0.53, two reps in each of 7 drills give 0.69, 15 cuts reach 0.70 and 25 reach 0.80. But drills run twice transfer no better: −0.12 [−0.36, +0.10] (43 receivers), against −0.04 for drills run once.
- **NFL entry speed.** Only 35% of drill reps are entered within 1 yd/s of the NFL speed for that route, and just 4 of 265 hitch-family reps (curl, comeback, hook). The matched reps, mostly slants (the one drill entered at NFL speed: 3.5 vs 4.1 yd/s), transfer −0.05 [−0.16, +0.07] (100 receivers).

The protocol to adopt: (1) if the cut is graded at all, grade at least 15 cuts per receiver, for example two reps in each of eight route drills; (2) keep the slant and re-stem the curl, comeback and hook at NFL hitch speed (about 4.7 yd/s, a controlled three-quarter stem), since today's hitch-family reps have no NFL match to test; (3) until a re-stemmed drill transfers, give Break G zero weight.

![Figure 4. Reliability of the Break G score by graded cuts per receiver, with a receiver-bootstrap band (A); drill-to-NFL transfer by protocol subset, receiver-clustered 90% intervals, against the correlation a true 0.5 would show (B).](figures/fig4_protocol.png)

## 4. Day 3: pick order stands; separation orders film

The obvious replacement is last season's SOE, shrunk for routes (a 200-route season keeps 0.60 of its value). It does not beat draft order. On 20 consecutive-season pairs from 16 day-3 or undrafted receivers, forecast leave-one-draft-class-out, the top half by SOE stayed above zero the next season 3 of 10 times and the top half by pick 4 of 10 (difference −10 points [−36, +12]; receiver-level permutation p = 0.81).

So the 2026 list orders film time and nothing else, and its expected hit rate is a coin flip: of 12 day-3 receiver-seasons above zero, 6 stayed there (50% [28, 72]).

Rule: day-3 or undrafted wide receiver, 14+ games, 100+ routes and SOE above zero in 2025. **GO** = also above zero over 100+ routes in 2024; **HOLD** = first such season. Order: GO then HOLD, each by true SOE (2025 SOE shrunk for routes). "Off the list below" is the 2026 SOE over his first 150 routes that pulls true SOE under zero. There is no injury report here, so missed games mix injury and inactive weeks.

| Film order: receiver (class, pick; 2025 team) | True SOE, in [90%] | 2025 games / routes | Role and roster change, 2024 → 2025 | Off the list below |
|---|---|---|---|---|
| 1. **Malik Washington** (2024, R6-184; MIA) | +16.1 [+7.7, +24.4] | 17 / 339 | routes per game 16 → 20; main spot slot → wide | −51 in |
| 2. **Demario Douglas** (2023, R6-210; NE) | +13.4 [+3.7, +23.0] | 17 / 220 | routes per game 25 → 13 | −32 in |
| 3. **Troy Franklin** (2024, R4-102; DEN) | +3.4 [−4.0, +10.9] | 17 / 461 | routes per game 16 → 27 | −14 in |

HOLD, in film order: 4. Tez Johnson (TB, +10.9); 5. Arian Smith (NYJ, +10.0; reserve list); 6. Isaiah Bond (CLE, +3.1); 7. Jaylin Lane (WAS, +2.8; reserve list); 8. Mitch Tinsley (CIN, +1.8; 134 routes); 9. Kayshon Boutte (NE, +0.6; now HOU).

What a GM will ask: Douglas's 2025 came on half his 2024 workload in the same slot role, so confirm injury or depth chart first. Franklin's routes nearly doubled as he became a starter, but his sampled snaps sit below the league against man and zone and when aligned wide (small cells; appendix), so watch his outside snaps first. Washington held his separation while moving outside. Arian Smith (a top-three drill cutter) and Jaylin Lane are on reserve lists; Boutte has left New England. Day-3 tight ends are left off: only 5 tight-end pairs back the rule.

![Figure 5. The 2026 film queue in watch order: true 2025 SOE with 90% interval (left) and, on its own axis, the drop-off point over the first 150 routes of 2026 (right).](figures/fig5_queue.png)

## Bottom line

1. **Take Break G out of the price.** Clubs pay 9% in pick position per sd of it for no separation.
2. **Fix the drill before grading it.** At least 15 graded cuts, hitch family re-stemmed at NFL entry speed, zero weight until it transfers.
3. **On day 3, the draft order stands.** Use last season's SOE only to order film: Malik Washington, Demario Douglas, Troy Franklin first, then the HOLD 6, each re-checked at 150 routes before the March 2027 league year.

## Limitations

- **Not blind.** All three classes were seen while building this; the ±3 inch margin, the baseline and the rule were fixed before testing.
- **Power.** The held-out test has 68% power at the 3-inch margin, and harm is not excluded (lower bound −3.8 inches per sd).
- **Small day-3 sample.** 20 pairs cannot separate SOE from pick order in either direction.
- **Game tracking is a sample**: up to 40 plays per player, half top-EPA, no defenders or ball. This limits Table 2's NFL rows and Figures 2 to 4, not the season tests (full-season NGS).
- **Survivorship.** Of 149 drill-scored receivers, 116 logged NFL routes with separation.

## Appendix

`verify.py` rebuilds both Break G pipelines from the raw frames and independently recomputes the headline, reliability, hierarchical, pick-price and day-3 numbers. `appendix.md` holds the equivalence tests, sensitivity ranges, the pick-gap override rule and the coverage splits.
