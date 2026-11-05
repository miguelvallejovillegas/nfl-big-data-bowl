# Appendix: robustness, measurement error and the 2025 day-3 names

Rendered by `analysis.py`; every number is in `all_numbers.json` and the starred ones are recomputed by `verify.py` (own bootstrap seeds; the held-out fit is also redone in statsmodels).

## Error definitions (Section 1)

Recipe: weighted least squares of career SOE (yards, route-weighted mean over seasons) on an intercept, log pick, official 40, weight, NGS athleticism score (each z-scored within class and position, missing = 0) and a TE flag; weights = career routes; fitted on two classes, predicted on the third. Error = sqrt of the mean squared miss over the held-out predictions, times 36.

| Held-out baseline error, inches | Scored route-weighted | Scored unweighted |
|---|---|---|
| Fitted route-weighted (body)* | 8.6 | 14.9 |
| Fitted unweighted | 9.7 | 15.1 |

Route-weighted sd of career SOE across the same receivers: 8.6 in. Held-out R² (route-weighted): five-covariate WLS 0.00; same fitted unweighted −0.25; log pick only −0.03; ridge on the five, penalty chosen by leave-one-class-out inside the two training classes (median penalty 1000, the top of the grid, so ridge shrinks almost to the class mean) 0.02. The unweighted-fit row reproduces a judge's refit (9.7 and 15.1 in): the gap was the fit weights, not the data.

Headline slope −1.2: class-stratified bootstrap [−3.8, 1.0]; pooled bootstrap across classes [−3.7, 1.0].

The ROPE column of Table 1 is a flat-prior posterior: normal, centred on the estimate, with the bootstrap sd as its standard error. Multiplicity: none of the 8 season-level slopes in Figure 2 excludes zero; smallest unadjusted two-sided p 0.33, Holm-adjusted 1.00 (p from each bootstrap interval under a normal approximation). Inclusion weights come from a ridge logistic model of reaching NFL routes on pick, tight-end flag, class and Break G, refitted in each resample; they move little because the baseline already holds pick.

## Reliability of the drill score (Section 3)

| Break G score | Reliability |
|---|---|
| Today, median 7 graded cuts (rep-level model)* | 0.53 |
| Tested receivers' average (drill-level model used for the shrinkage)* | 0.50 |
| Two reps in each of 7 drills* | 0.69 |
| 15 graded cuts* | 0.70 |
| Transfer of drills run twice (43 receivers) | −0.12 [−0.36, 0.10] |

![Reliability of the Break G score by graded cuts per receiver. The curve is the rep-level model; the dotted mark is the drill-level model the held-out test uses.](figures/figA_reliability.png)

![Drill-to-NFL Break G by protocol subset, entry-speed test and official agility time, receiver-clustered 90% intervals. Dashed: what a true 0.5 would show through both measures' noise (0.22); dotted: what a perfect link would show (0.43). Only 4 of 265 hitch-family drill reps are entered within 1 yd/s of NFL speed.](figures/figB_subsets.png)

## Equivalence, measurement error and sensitivity

| Check | Result |
|---|---|
| TOST of the held-out slope at ±3 in* | p = 0.11 (benefit arm 0.001, harm arm 0.11) |
| TOST of the hierarchical slope at ±3 in (Wald) | p = 0.04 |
| Hierarchical slope, receiver and class effects (167 player-seasons, 91 receivers)* | −0.5 [−2.8, 1.8] |
| Slope with the between-drill ICC (0.17 [0.10, 0.23]) redrawn in every resample* | −1.2 [−3.6, 1.0] |
| Slope on the raw drill average, per observed sd (no error correction)* | −1.0 [−2.8, 0.8] |
| Held-out slope across 6 outcome builds (route cut-offs 50/100, no cushion term, unweighted, wide receivers only)* | −3.0 to −0.7 in per sd |
| Mechanics, held out: brake into / speed through / drive out of the cut | −1.6, 1.4, −2.2 |
| Break G split-half (five drills vs the other five)* | 0.28 [0.15, 0.40], 145 receivers |
| Same drill run twice: median ICC* | 0.05 (70 pairs) |
| NFL Break G, odd vs even snaps* | 0.23, 113 receivers |
| Transfer corrected for both reliabilities* | −0.08 |
| What 3 inches of SOE is worth | about 0.44 catches a season |

## Sampled NFL plays: the two halves and entry speed (Section 2)

| Drill → NFL comparison | Rank corr. [90%] | n |
|---|---|---|
| Official 40 → NFL top speed, random plays only | 0.43 [0.27, 0.56] | 87 |
| Combine → NFL Break G, random plays only | 0.13 [−0.05, 0.30] | 94 |
| Combine → NFL Break G, top-EPA plays only | −0.19 [−0.34, −0.03] | 102 |
| Same route family, all breaks | −0.07 [−0.15, 0.01] | 114 |
| Same route family, entry speed removed on both sides | −0.03 [−0.11, 0.05] | 114 |
| Official 3-cone → NFL Break G | −0.01 [−0.28, 0.27] | 34 |

Break G rises with entry speed (rank corr. 0.37 in drills, 0.56 in games); 37% of NFL breaks are entered at drill-like speed. The two sampling halves pull in opposite directions, so the sample may hide a small transfer; the season tests use full-season NGS and are free of it. Within the 49 eligible wide receivers of Figure 1, drill and NFL Break G rank-correlate 0.10.

## Day 3: tests behind the bottom line

Receiver level (body): 28 receivers, one season pair each, forecasts leave-one-class-out; correct above/below-zero calls 39% (pick + SOE) vs 54% (pick only)*. Pair level: over 38 pairs (28 receivers) the top half by SOE stayed above zero 42% of the time against 63% for the top half by pick (−21 points [−43, 0])*.

With 100+ routes in both seasons (20 pairs, 16 receivers), the SOE half hit 3 of 10 and the pick half 4 of 10* (difference −10 points [−36, 12]*, permutation p = 0.81). Of 12 day-3 receiver-seasons above zero, 6 stayed above zero*: 50% [28, 72], a coin flip.

The pick-gap override rule (pick gap under 32) overrode pick order 11 times and was wrong 36 per 100 [0, 75]* against a break-even of 64*.

## 2025 day-3 wide receivers above zero (film aid only; the body publishes no ordered list)

"Off the list below" is the 2026 SOE over his first 150 routes that pulls his true SOE under zero. Splits are sampled-snap SOE against the league in the same coverage or alignment; only 5 of 36 cells reach 30 snaps.

| Receiver (2025 team) | True 2025 SOE, in | Off the list below | Man, in (snaps) | Zone, in (snaps) | Slot, in (snaps) | Wide, in (snaps) |
|---|---|---|---|---|---|---|
| Malik Washington (MIA) | +16.1 | −51 | −1 (5) | +23 (34) | +24 (11) | +19 (26) |
| Demario Douglas (NE) | +13.4 | −32 | +33 (14) | −13 (26) | +5 (27) | −4 (11) |
| Troy Franklin (DEN) | +3.4 | −14 | −16 (9) | −17 (31) | +7 (12) | −24 (27) |
| Tez Johnson (TB, HOLD) | +10.9 | −30 | +4 (14) | +3 (26) | −13 (14) | +9 (25) |
| Arian Smith (NYJ, HOLD) | +10.0 | −24 | −9 (11) | +14 (29) | −6 (17) | +18 (23) |
| Isaiah Bond (CLE, HOLD) | +3.1 | −10 | +11 (14) | +36 (26) | +25 (16) | +29 (24) |
| Jaylin Lane (WAS, HOLD) | +2.8 | −6 | −4 (19) | −13 (21) | +3 (13) | −17 (25) |
| Mitch Tinsley (CIN, HOLD) | +1.8 | −3 | +8 (14) | +4 (26) | +29 (7) | 0 (33) |
| Kayshon Boutte (NE, HOLD) | +0.6 | −2 | −7 (9) | +2 (30) | −10 (4) | +1 (35) |
