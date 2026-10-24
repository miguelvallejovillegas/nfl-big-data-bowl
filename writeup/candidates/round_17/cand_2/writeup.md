# Combine Cuts Don't Predict NFL Separation

*Break G* is how hard a receiver pushes off the ground at the sharpest point of a route cut, measured at 10 Hz in the Combine route drills. *Separation over expected (SOE)* is how open he is when the ball is thrown, beyond what his alignment and cushion predict (Next Gen Stats, full season).

**On 116 receivers from draft classes the model never saw, one sd of Break G is worth −1.2 inches of season SOE (90% interval −3.6 to +1.0). The best case, +1.0 inch, is 0.14 catches a season. To break ties between day-3 receivers, use last season's SOE instead.**

![Figure 1. All 99 receivers with 8+ NFL breaks (A) and five receivers that span the Combine range (B).](figures/fig1_cards.png)

## Why receivers, why now

Clubs usually keep five or six wide receivers. The last spot goes to a day-3 or undrafted player, and the choice is repeated every summer. The 2025 day-3 and undrafted receivers and tight ends who ran 100+ routes earn a median $1.08M a year. Each mistake is cheap, but it also blocks a younger player who might get open.

The evidence on getting open is thin. A scout sees a median of 7 graded route-drill cuts per receiver at the Combine, and 91% of drill-receiver cells hold one rep. NFL separation is noisy too: a 200-route season keeps 0.60 of its weight once shrunk for sample size. Since 2023 the route drills are tracked at 10 Hz, so the same code can grade the cut on Combine and game tracking. Straight-line speed carries from the Combine to the NFL: within position and class, the official 40 tracks NFL top speed over the full play (ρ = 0.43 [0.29, 0.54], 100 receivers). Does the cut carry too? And if not, what breaks the tie?

## The roster decision

| Question (n) | Held-out result [90% interval] | Verdict |
|---|---|---|
| Does Break G add season SOE beyond pick, 40, weight and NGS score? (116 receivers, unseen classes) | −1.2 in per sd [−3.6, +1.0]; R² change 0.013 [−0.157, 0.085] | **No** |
| Does the drill cut carry to the same NFL route? (99 with 8+ NFL breaks) | true correlation 0.02 [−0.27, 0.32] | **No visible transfer** |
| Does last season's SOE rank day-3 receivers better than pick and routes? (20 pairs, 16 receivers) | rank-correlation gain 0.31 [−0.08, 0.63]; closer on 12 of 20 | **Not proven; tiebreaker only** |
| Did earlier day-3 SOE lists beat the base rate? | 7 of 13 hit (54% [32, 75]) vs 8 of 20 (40% [24, 58]) | **A lean, not proof** |

## What to do

1. **Break ties between day-3 receivers with last season's route-shrunk SOE, and only ties.** The edge is not proven, but a tie is otherwise a coin flip, and the SOE forecast was closer on 12 of 20 day-3 pairs. Go now on Malik Washington (MIA) and Demario Douglas (NE).
2. **Take Break G out of the price.** At the same position, clubs draft a receiver 1 sd harder in the cut 9% earlier [2, 17] and get no separation for it.
3. **Finding: hard and soft drill cutters run the same NFL cut** (Figure 3). At the top of a hitch the gap is −0.2 yd/s [−0.5, +0.1]. On slants, outs and corners the NFL cut is taken at speed (median 3.6, 4.8 and 6.4 yd/s at the turn), so Break G there measures the turn more than the brake.
4. **Film hypothesis, not a draft trait:** against off coverage, a full sink at the top of the hitch wins. A typical receiver enters at 4.7 yd/s and sinks to 1.5. How deep a receiver sinks does not track his season SOE (ρ = 0.01, 92 receivers), so check it on film before coaching it.
5. **2026 Combine protocol.** If you time cut drills, run slant, dagger and comeback twice: a second rep lifts reliability from 0.53 to 0.62. Read speed from the official 40, not from the tracked clip, which starts about 0.3 s before the laser.

## Drill to tape

Break G is found on 96% of 926 drill reps. It is standardized within drill and Combine year and averaged over route drills (median 7). NFL snaps are graded on slant, out, hitch, in and corner routes. The drill trait is real: Break G on five drills predicts the other five at ρ = 0.28 (145 players), and it is not a speed score (ρ with the 40: −0.01).

![Figure 2. Same receiver, same route family, noise of few reps and few snaps modelled: every family's interval includes zero. The official 40 against NFL top speed is shown for scale.](figures/fig2_chain.png)

It does not reach NFL routes. The pooled true correlation is below 0.32 with 90% probability, far under the 40's 0.43 for speed. A small link cannot be excluded; a useful one (above 0.2) has 17% probability.

![Figure 3. NFL speed through the cut for the top and bottom thirds of Combine Break G, by route family: hard drill cutters do not cut differently on Sundays.](figures/fig3_film.png)

## Season separation on unseen classes

The outcome is full-season SOE, route-weighted over each career (213 player-seasons, median 231 routes). The forecast without tracking was fixed before the test: log pick, official 40, weight, NGS athleticism score and a TE flag. Fitted on two classes and scored on the third, it is weak: held-out R² 0.00 [−0.51, 0.14], error 8.6 inches, rank correlation 0.12. Adding Break G changes R² by 0.013 [−0.157, 0.085], and what little it adds comes through a negative slope. Even at the upper bound, +1.0 inch per sd, Break G is worth 0.14 catches a season (0.24 for a heavily targeted receiver).

## The 2026 day-3 shortlist

Pool: day-3 or undrafted receivers with 100+ routes and SOE above zero in 2025 (SOE shrunk by routes). **GO**: listed again after an earlier list hit. **HOLD**: first listing, one season. P: model chance of 2026 SOE above zero.

| Call | Receiver (class, pick; 2025 team, QB) | 2025 games, routes; shrunk SOE; P | Earlier lists | Film first |
|---|---|---|---|---|
| **GO** | **Malik Washington** (2024, 184; MIA, Tagovailoa) | 17 g, 339; +16.1 in; 95% | 2024→25 hit | Hitch 19% of sampled routes; man 19% |
| **GO** | **Demario Douglas** (2023, 210; NE, Maye) | 17 g, 220; +13.4 in; 89% | 2023→24 hit, 2024→25 hit | Go 22% of sampled routes; man 25% |
| **HOLD** | **Tez Johnson** (2025, 235; TB, Mayfield) | 16 g, 282; +10.9 in; 85% | first listing | Go 22% of sampled routes; man 29% |
| **HOLD** | **Arian Smith** (2025, 110; NYJ, Taylor) | 16 g, 231; +10.0 in; 81% | first listing | Go 32% of sampled routes; man 25% |

**Read P as an ordering, not as odds.** Held-out forecasts of 60% or more (mean 70%) came true 14 of 23 times (61% [44, 76]). Earlier day-3 lists hit 7 of 13 against a 40% base rate.

**Film checklist before the GO call** (sampled snaps are too few for coverage splits):
- Does he win at the top against man on third down, not only against zone?
- Does he keep speed through slants and outs (league 3.6 and 4.8 yd/s at the turn)?
- Is his 2025 quarterback starting in 2026? A change resets the read.

**Watch:** Troy Franklin (61%), A.J. Barner, TE (60%), Gunnar Helm, TE (59%), Isaiah Bond (56%). The full list is in the appendix.

![Figure 4. All 20 day-3 season pairs with each list's hits and misses (A), the same pairs as forecast errors (B), and the 2026 cards (C).](figures/fig4_shortlist.png)

**What would change the call.** The 2026 test alone has simulated power 0.27 (18 expected pairs); even with a real edge, the 2026 gain is negative in 10% of simulated seasons. Keep the tiebreaker while the pooled 2023–26 gain stays positive. Drop it if the pooled gain turns negative or the GO names both fall below zero.

## Limitations

- **Not blind, small day-3 sample.** All three classes were seen while building this, and day 3 has 20 pairs (5 of them tight ends, too few to judge alone; wide receivers only: gain 0.56 [−0.02, 0.99]).
- **The downside is open.** The interval rules out gains above +1.0 inch per sd, not losses down to −3.6.
- **Game tracking is a sample.** Up to 40 plays per player, half chosen for high EPA, without defenders or ball. That limits the film cues and the drill-to-game link, not the season SOE test.
- **Availability and survivorship.** Games played is a rough injury proxy. 33 of 149 drill-scored receivers never logged NFL routes with separation; re-weighting for this changes little.
- **The freeze is late.** The 2026 rule was frozen on 24 October 2026, after the season began but before any full-season SOE existed.

## Appendix

**Pre-registration.** `frozen_rule_2026.json` holds the coefficients, shrinkage, threshold, names and probabilities (sha256 5c1beda70454…). Primary test: day-3 rank-correlation gain over pick and routes, 18 expected pairs (61% of day-3 receivers keep 100+ routes). Analytic power is 0.37 alone and 0.61 pooled with 2023–25; the list needs 9 of 13 hits to beat the base rate (power 0.20).

**Every estimate and its alternatives** (90% intervals; receivers resampled).

| Claim | Primary (n) | Alternatives |
|---|---|---|
| Break G slope, in of SOE per sd | offset forecast −1.2 [−3.6, +1.0] (116) | re-weighted for survival −0.8 [−2.8, +1.3]; without Puka Nacua −0.5 [−2.7, +1.5]; QB-adjusted −1.2 [−3.6, +0.9]; within-class shuffle p 0.37 |
| Drill cut to NFL cut | true correlation, same family 0.02 [−0.27, 0.32] (99) | raw ρ, any family −0.11 [−0.25, 0.04] (115); posterior within ±0.2: 72% |
| Day-3 gain over pick and routes | season folds 0.31 [−0.08, 0.63] (20 pairs) | shrinkage refitted in each fold 0.31 [−0.08, 0.65]; draft-class folds 0.15 [−0.10, 0.39]; rule vs raw last-season SOE 0.09 [−0.19, 0.40]; mean error cut +0.8 in [−0.3, +1.8] |
| Matched within-season contrasts | higher SOE separates more next year: 62% of 53 [41, 80] | shuffle p 0.14 |

**Notes.** The pre-set equivalence margin was ±3 inches per sd (0.44 catches); the upper bound sits inside it (one-sided p 0.001). Class folds cannot be crossed with season folds: 2024 rookies have no 2023 season. P is rescaled to held-out hit rates (day 3: 50% forecast, 40% observed). Man-minus-zone snap splits have a split-half reliability of −0.06, so they are not shown. Top speed is the 90th percentile of per-snap maximum speed over every tracked frame of 8+ sampled snaps. With the raw 40, WR and TE pooled, ρ = 0.74 (100), mostly the position gap. Snap-to-throw frames of breaking snaps only, where short routes cap speed: ρ = 0.34 (99). The true correlation uses an errors-in-variables model, one slope per route family (399 receiver-family cells). 1 sd of Break G = 0.021 g; catches use 8.8 catch-rate points per yard and 60 targets.

**Rest of the 2026 list** (class, pick; team; games; shrunk SOE; P): Troy Franklin (2024, 102; DEN; 17 g; +3.4 in; 61%); A.J. Barner, TE (2024, 121; SEA; 17 g; +9.9 in; 60%); Gunnar Helm, TE (2025, 120; TEN; 16 g; +10.4 in; 59%); Isaiah Bond (2025, UDFA; CLE; 16 g; +3.1 in; 56%); Jaylin Lane (2025, 128; WAS; 14 g; +2.8 in; 47%); Kayshon Boutte (2023, 187; NE; 14 g; +0.6 in; 44%); Mitch Tinsley (2023, UDFA; CIN; 17 g; +1.8 in; 39%); Mitchell Evans, TE (2025, 163; CAR; 17 g; +6.7 in; 34%); Ja'Tavion Sanders, TE (2024, 101; CAR; 12 g ⚑; +4.0 in; 29%). ⚑ = 13 or fewer games in 2025.

Reproduce: `uv run python writeup/candidates/round_17/cand_2/analysis.py` (about nine minutes) writes every number, figure and this text.
