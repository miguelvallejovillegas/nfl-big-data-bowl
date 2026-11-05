# Combine Cuts Don't Predict NFL Separation

*Break G* is how hard a receiver pushes off the ground at the sharpest point of a route cut, measured at 10 Hz in the Combine route drills. *Separation over expected (SOE)* is how open he is when the ball is thrown, beyond what his alignment and cushion predict (Next Gen Stats, full season), in inches.

**Combine cut sharpness does not predict NFL separation: −1.2 inches of SOE per sd of Break G [90%: −3.6, +1.0], 116 receivers from draft classes the model never saw. Straight-line speed does carry (ρ = 0.43). Between day-3 receivers, rank by last season's SOE, not draft slot: a lean, not a proof.**

![Figure 1. All 99 receivers with 8+ NFL breaks, Combine against NFL Break G (A), and five receivers that span the Combine range (B).](figures/fig1_cards.png)

## Why receivers, why now

Every summer the last receiver spot goes to a day-3 or undrafted player. In 2025 those who ran 100+ routes earned a median $1.08M, close to the league minimum. The money is small; the cost of a wrong call is the roster spot and practice reps a better receiver never gets.

The evidence on getting open is thin on both ends. At the Combine a scout sees a median of 7 graded route-drill cuts per receiver, and 91% of drill-receiver cells hold one rep. In the NFL, one 200-route season of SOE is about 60% signal: if two receivers differ by 10 inches this year, expect about 60% of that gap to persist next year, and the other 40% to wash out.

Since 2023 the route drills are tracked, so one piece of code grades the cut at the Combine and on NFL snaps. Straight-line speed is the benchmark: the official 40 ranks NFL top speed at ρ = 0.43 [0.29, 0.54] within position and class (100 receivers). Does the cut carry the same way? If not, what should break a day-3 tie?

## The roster decision

| Question (n) | Held-out result [90% interval] | Verdict |
|---|---|---|
| Does Break G add season SOE beyond pick, 40, weight and NGS score? (116 receivers, unseen classes) | −1.2 in per sd [−3.6, +1.0]. A benefit of +3 in is ruled out (p = 0.003); harm of −3 in is not (equivalence p = 0.11) | **No benefit** |
| Does the drill cut carry to the same NFL route? (99 with 8+ NFL breaks) | correlation 0.02 [−0.27, 0.32], noise corrected | **No visible transfer** |
| Does last season's SOE rank day-3 receivers better than pick and routes? (20 pairs, 16 receivers) | gain in rank correlation 0.31 [−0.08, 0.63], p = 0.12; every round (53 pairs): 0.33 [0.13, 0.53] | **Lean** |
| Does the pick gap matter? (106 same-season day-3 contrasts) | earlier pick orders 52% right, SOE 62%; overruling only within 32 picks adds +1 pt | **No: use SOE at any gap** |

## What to do

1. **Take Break G out of the price.** At the same position, clubs draft a receiver 1 sd harder in the cut 9% earlier [2, 17] and get no separation for it.
2. **Fix the Combine measurement.** If you time cut drills, run slant, dagger and comeback twice: a second rep lifts reliability from 0.53 to 0.62.
3. **Between two day-3 wide receivers, keep the one with higher last-season SOE, whatever the pick gap.** When this rule is wrong, the receiver kept runs 8.3 inches of SOE [5.0, 12.7] worse than the one cut, about 1.2 catches a season; when keeping the earlier pick is wrong, the loss is 12.0 inches (1.8 catches). Averaged over all decisions, the rule saves 2.6 inches per call [−0.9, +6.3]. Not for tight ends: 5 pairs, gain −0.50.
4. **2026: GO on Malik Washington and Demario Douglas.** Review each at 150 routes (Figure 5) and decide before the March 2027 league year.

## The drill cut does not show up in NFL routes

Break G is found on 96% of 926 drill reps, standardized within drill and Combine year and averaged over a median of 7 drills. It is a real drill trait: Break G on five drills predicts the other five at ρ = 0.28 (145 players). It is not a speed score (ρ with the 40: −0.01). NFL snaps are graded with the same code on slant, out, hitch, in and corner routes.

![Figure 2. Same receiver, same route family, with the noise of few reps and few snaps modelled: every family's interval includes zero. The 40 against NFL top speed is shown over the full play (solid) and the snap-to-throw window used for the cut (dashed).](figures/fig2_chain.png)

After correcting for that noise, the drill-to-NFL correlation for the same receiver on the same route is 0.02, and below 0.32 with 90% probability. A useful link (above 0.2) has a 17% chance. Speed, measured in the same snap-to-throw window as the cut, still carries (0.34).

![Figure 3. NFL speed through the cut for the top and bottom thirds of Combine Break G, by route family, with the difference beneath.](figures/fig3_film.png)

Hard and soft drill cutters reach the same speed at the top of the NFL cut: on the hitch the gap is −0.2 yd/s [−0.5, +0.2], and all 5 families' intervals include zero. On slants, outs and corners the NFL cut is taken at speed (median 3.6, 4.8 and 6.4 yd/s at the turn), so a drill brake has little to transfer to.

## Break G adds nothing to season separation

The outcome is full-season SOE, route-weighted over each career (213 player-seasons, median 231 routes). The forecast without tracking was fixed before the test: log pick, official 40, weight, NGS athleticism score and a TE flag. Fitted on two classes and scored on the third, it is weak: held-out R² 0.00 [−0.51, 0.14], error 8.6 inches. Adding Break G changes R² by 0.013 [−0.157, 0.085], through a negative slope.

The ±3 inch margin (0.44 catches a season) was set before testing. The data rule out a Break G benefit that large but not an equally large harm: 89% of resamples fall inside ±3, and the 90% interval reaches −3.6. For pricing, that is enough: no evidence supports paying for the cut.

## Day 3: last season beats the draft slot, narrowly

The gain is the rank correlation of the held-out forecast with next-season SOE when last season's SOE is added, minus the same without it; models are fitted on one season pair and scored on the other. The 20 day-3 pairs come from 16 receivers; four appear twice, and resampling receivers rather than pairs puts the effective sample at about 19 independent pairs.

There is one test. The interval resamples receivers; its one-sided p is the share of resamples at or below zero (0.12). A shuffle that keeps the fitted models and swaps whole receivers' last-season SOE among day-3 receivers agrees: p = 0.12. An earlier shuffle across all 53 pairs gave p = 0.01, but it also erased the all-rounds signal the model learns from, so it asked whether SOE carries any signal (it does), not whether it helps on day 3.

![Figure 4. All 20 day-3 pairs as forecast errors (A), the pick-gap rule (B), and gains with intervals (C).](figures/fig4_tiebreak.png)

Inside day 3 the earlier pick is close to a coin flip (52% of 106 contrasts ordered right). We first set the tiebreak at a pick gap under 32: SOE overrules the pick in 17 contrasts and gains +1 pt [−11, +12]. At any gap it gains +10 pts [−9, +28] and lifts the share of kept receivers with positive next-season SOE by +5 pts [−11, +20]. Every interval includes zero.

## The 2026 calls

Pool: day-3 or undrafted receivers with 100+ routes and SOE above zero in 2025. **GO** = listed again after an earlier hit. **HOLD** = first listing. True SOE = 2025 SOE shrunk for routes. P = model chance of 2026 SOE above zero; read it as an ordering, since held-out forecasts of 60% or more came true 61% of the time [44, 76] against a day-3 base rate of 40%.

| Call | Receiver (class, pick; 2025 team) | 2025 routes; slot | True SOE [90%]; P | Snap SOE in (snaps): man / zone; slot / wide | Flips if, at 150 routes in 2026 |
|---|---|---|---|---|---|
| **GO** | **Malik Washington**, WR (2024, 184; MIA) | 339; 23% | +16.1 [+7.7, +24.4]; 95% | −1 (5) / +23 (34); +24 (11) / +19 (26) | SOE below 0 → HOLD |
| **GO** | **Demario Douglas**, WR (2023, 210; NE) | 220; 67% | +13.4 [+3.7, +23.0]; 89% | +33 (14) / −13 (26); +5 (27) / −4 (11) | SOE below 0 → HOLD |
| **HOLD** | Tez Johnson, WR (2025, 235; TB) | 282; 35% | +10.9 [+2.0, +19.8]; 85% | +4 (14) / +3 (26); −13 (14) / +9 (25) | SOE above 0 → GO |
| **HOLD** | Arian Smith, WR (2025, 110; NYJ) | 231; 42% | +10.0 [+0.5, +19.5]; 81% | −9 (11) / +14 (29); −6 (17) / +18 (23) | SOE above 0 → GO |

![Figure 5. The 2026 list with intervals (A) and the four act-on names' sampled-snap SOE by coverage and alignment (B).](figures/fig5_calls.png)

Splits come from up to 40 sampled snaps per player; 1 of 16 cells reaches 30 snaps, so they are film prompts. Washington's case rests on zone (34 snaps); Douglas's on man (14 snaps): snap SOE +33 inches against man, −13 against zone. 2025 film to pull, picked by rule (league hitch median 1.5 yd/s at the top):
- **Washington**: hitch sink, wk 4 vs NYJ, Q1 12:14: 0.6 yd/s at the top; slant at speed, wk 9 vs BAL, Q2 4:17: 5.8 yd/s at the turn; best snap vs man, wk 1 vs IND, Q4 6:29: 2.9 yd open.
- **Douglas**: hitch sink, wk 9 vs ATL, Q3 14:13: 6.3 yd/s at the top; out at speed, wk 11 vs NYJ, Q3 14:20: 5.4 yd/s at the turn; best snap vs man, wk 11 vs NYJ, Q1 3:56: 3.8 yd open.

Douglas's sampled hitch shows no sink; check his other hitches before the March decision.

## Limitations

- **Not blind, small day-3 sample.** All three classes were seen while building this; the day-3 test has about 19 effective pairs and p = 0.12. Dropping any one pair leaves the gain between 0.19 and 0.37.
- **The harm side is open.** The Break G interval reaches −3.6 inches per sd; only a benefit is ruled out.
- **Game tracking is a sample.** Up to 40 plays per player, half chosen for high EPA, without defenders or ball. That limits the clips, splits and the drill-to-NFL link, not the season tests.
- **The 40.** The tracked 40 clip starts before the laser, so every speed number uses the official time.
- **Survivorship.** 33 of 149 drill-scored receivers never logged NFL routes with separation; re-weighting for this changes little.
- **The freeze is late.** The 2026 rule was frozen on 24 October 2026, after the season began, before any full-season SOE existed. The 2026 test has power 0.27.

## Appendix

**Every estimate and its alternatives** (90% intervals; receivers resampled).

| Claim | Primary (n) | Alternatives |
|---|---|---|
| Break G slope, in of SOE per sd | offset forecast −1.2 [−3.6, +1.0] (116) | re-weighted for survival −0.8 [−2.8, +1.3]; without Puka Nacua −0.5 [−2.7, +1.5]; QB-adjusted −1.2 [−3.6, +0.9]; ridge baseline R² 0.03 |
| Day-3 gain over pick and routes | season folds 0.31 [−0.08, 0.63] (20 pairs) | shrinkage refitted in each fold 0.31 [−0.08, 0.65]; draft-class folds 0.15 [−0.10, 0.39]; rule vs raw last-season SOE 0.09 [−0.19, 0.40]; WR only 0.56 [−0.02, 0.99] |

**Speed, one definition.** Top speed = 90th percentile of a receiver's per-snap maximum. The text uses the full play with the 40 standardized within position and class (0.43); pooling WR and TE on the raw 40 gives 0.74, mostly the position gap; the snap-to-throw window gives 0.34.

**Notes.** 1 sd of Break G = 0.021 g; catches use 8.8 catch-rate points per yard and 60 targets. Pre-registration: `frozen_rule_2026.json` (sha256 5c1beda70454…).

**Reproduce.** `analysis.py` (about nine minutes) writes every number, figure and this text. `verify.py` recomputes the baseline R², ΔR², Break G slope, the three speed correlations and the day-3 gains from `data/processed` and three CSVs in seconds and diffs them against `all_numbers.json`; `columns.json` documents every CSV column.
