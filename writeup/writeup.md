# Combine Cuts Don't Predict NFL Separation

*Break G* is how hard a receiver pushes off the ground at the sharpest point of a route cut, measured at 10 Hz in the Combine route drills. *Separation over expected (SOE)* is how open he is when the ball is thrown, beyond what his alignment and cushion predict (Next Gen Stats, full season).

**On 116 receivers from draft classes the model never saw, one sd of Break G is worth −1.2 inches of season SOE (90% interval −3.6 to +1.0). The best case, +1.0 inch (0.14 catches a season), sits inside the ±3 inch margin (0.44 catches) set before testing. Speed carries from Combine to NFL (ρ = 0.43); the cut does not. Between day-3 receivers, lean on last season's SOE, then check the film.**

![Figure 1. All 99 receivers with 8+ NFL breaks, Combine against NFL Break G (A), and five receivers that span the Combine range (B).](figures/fig1_cards.png)

## Why receivers, why now

The last receiver spot usually goes to a day-3 or undrafted player, every summer. The 2025 day-3 and undrafted receivers and tight ends who ran 100+ routes earn a median $1.08M a year. A miss is cheap but blocks a younger player.

The scout's evidence on getting open is thin. At the Combine he sees a median of 7 graded route-drill cuts per receiver, and 91% of drill-receiver cells hold a single rep. In the NFL, one 200-route season of SOE is about 60% signal and 40% noise.

Since 2023 the route drills are tracked at 10 Hz, so one piece of code grades the cut on Combine and NFL tracking. Straight-line speed carries: over the full play, the official 40 ranks NFL top speed at ρ = 0.43 [0.29, 0.54] within position and class (100 receivers). Does the cut carry the same way? If not, what breaks the tie?

## The roster decision

| Question (n) | Held-out result [90% interval] | Verdict |
|---|---|---|
| Does Break G add season SOE beyond pick, 40, weight and NGS score? (116 receivers, unseen classes) | −1.2 in per sd [−3.6, +1.0]; inside the ±3 in margin | **No** |
| Does the drill cut carry to the same NFL route? (99 with 8+ NFL breaks) | correlation 0.02 [−0.27, 0.32], noise corrected | **No visible transfer** |
| Does last season's SOE rank receivers better than pick and routes? | all rounds, 53 pairs: gain 0.33 [0.13, 0.53]; day 3, 20 pairs: 0.31 [−0.08, 0.63] | **Yes overall; day 3 same size, unproven** |
| Did earlier day-3 SOE lists beat the base rate? | 7 of 13 hit (54% [32, 75]) vs 8 of 20 (40% [24, 58]) | **Intervals overlap: a lean** |

The gain is the rank correlation of the held-out forecast with next-season SOE when last season's SOE is added, minus the same without it. Folds are season pairs (fit on 2024→25, score 2023→24, and the reverse); receivers are resampled. An earlier brief quoted 0.63 [−0.08, 0.90] from leave-one-receiver-out folds, where the pick-and-routes forecast itself scored −0.43, inflating the gain.

## What to do

1. **If you must choose between two day-3 receivers with similar pick and routes, lean to the one with higher last-season SOE.** On day 3 the gain matches the 53-pair result but its interval includes zero, so a lean is all it supports. For 2026: Malik Washington and Demario Douglas, if the film checklist confirms them.
2. **Use it for wide receivers, not tight ends.** WR only, the day-3 gain is 0.56 [−0.02, 0.99] on 15 pairs; on the 5 tight-end pairs it is −0.50 [−1.12, 0.00].
3. **Take Break G out of the price.** At the same position, clubs draft a receiver 1 sd harder in the cut 9% earlier [2, 17] and get no separation for it.
4. **2026 Combine protocol.** If you time cut drills, run slant, dagger and comeback twice: a second rep lifts reliability from 0.53 to 0.62.

## The drill cut does not show up in NFL routes

Break G is found on 96% of 926 drill reps, standardized within drill and Combine year and averaged over a median of 7 drills. It is a real drill trait: Break G on five drills predicts the other five at ρ = 0.28 (145 players). It is not a speed score (ρ with the 40: −0.01). NFL snaps are graded with the same code on slant, out, hitch, in and corner routes.

![Figure 2. Same receiver, same route family, with the noise of few reps and few snaps modelled: every family's interval includes zero. The official 40 against NFL top speed is shown for scale.](figures/fig2_chain.png)

After correcting for that noise, the drill-to-NFL correlation for the same receiver on the same route is 0.02, and below 0.32 with 90% probability, against 0.43 for the 40 and top speed. A useful link (above 0.2) has a 17% chance.

![Figure 3. NFL speed through the cut for the top and bottom thirds of Combine Break G, by route family, with the difference beneath: hard drill cutters do not cut differently on Sundays.](figures/fig3_film.png)

Hard and soft drill cutters reach the same speed at the top of the NFL cut: on the hitch the gap is −0.2 yd/s [−0.5, +0.1], and all 5 families' intervals include zero. On slants, outs and corners the NFL cut is taken at speed (median 3.6, 4.8 and 6.4 yd/s at the turn), so a drill brake has little to transfer to.

## Break G adds nothing to season separation

The outcome is full-season SOE, route-weighted over each career (213 player-seasons, median 231 routes). The forecast without tracking was fixed before the test: log pick, official 40, weight, NGS athleticism score and a TE flag. Fitted on two classes and scored on the third, it is weak: held-out R² 0.00 [−0.51, 0.14], error 8.6 inches, rank correlation 0.12. A ridge version gives R² 0.03 [−0.28, 0.14], so the near-zero baseline is not a fitting accident. Adding Break G changes R² by 0.013 [−0.157, 0.085], and what little it adds comes through a negative slope.

`reproduce_baseline.py` rebuilds season SOE from `player_season_stats.parquet` (`soe_season.csv`) and, with `baseline_inputs.csv`, re-derives R² 0.00, error 8.6 inches and the −1.2 inch slope in seconds.

## The 2026 day-3 shortlist

Pool: day-3 or undrafted receivers with 100+ routes and SOE above zero in 2025. **GO**: listed again after an earlier hit, and confirmed on film. **HOLD**: first listing. P: model chance of 2026 SOE above zero, an ordering.

| Call | Receiver (class, pick; 2025 team, QB) | 2025 games, routes; slot share | Shrunk SOE; P | Earlier lists |
|---|---|---|---|---|
| **GO** | **Malik Washington**, WR (2024, 184; MIA, Tagovailoa) | 17 g, 339; 23% slot | +16.1 in; 95% | 2024→25 hit |
| **GO** | **Demario Douglas**, WR (2023, 210; NE, Maye) | 17 g, 220; 67% slot | +13.4 in; 89% | 2023→24 hit, 2024→25 hit |
| **HOLD** | **Tez Johnson**, WR (2025, 235; TB, Mayfield) | 16 g, 282; 35% slot | +10.9 in; 85% | first listing |
| **HOLD** | **Arian Smith**, WR (2025, 110; NYJ, Taylor) | 16 g, 231; 42% slot | +10.0 in; 81% | first listing |

**Base rates for reading P.** Earlier day-3 lists hit 7 of 13 (54% [32, 75]) against 8 of 20 for all day-3 pairs (40% [24, 58]); the intervals overlap. Held-out forecasts of 60% or more came true 14 of 23 times (61% [44, 76]). Slot share is shown, but 20 pairs cannot separate slot from wide alignment.

**Film checklist for the GO names** (picked by rule from sampled 2025 snaps: slowest hitch turn, fastest slant or out turn, most open snap against man; league hitch median 1.5 yd/s):
- **Malik Washington.** (1) *Hitch sink* (2025 wk 4 vs NYJ, Q1 12:14, 3rd & 7, Cover 4): 0.6 yd/s at the top, 3.8 yd open. (2) *Slant at speed* (2025 wk 9 vs BAL, Q2 4:17, 3rd & 7, Cover 3, wide): 5.8 yd/s at the turn, 5.8 yd open. (3) *Against man* (2025 wk 1 vs IND, Q4 6:29, 4th & 11, Cover 0, slant): 2.9 yd open.
- **Demario Douglas.** (1) *Hitch sink* (2025 wk 9 vs ATL, Q3 14:13, 2nd & 14, Cover 3): 6.3 yd/s at the top, 2.2 yd open. **No sink: check his other hitches first.** (2) *Out at speed* (2025 wk 11 vs NYJ, Q3 14:20, 3rd & 7, Cover 1, slot): 5.4 yd/s at the turn, 1.0 yd open. (3) *Against man* (2025 wk 11 vs NYJ, Q1 3:56, 4th & 2, Cover 1, out): 3.8 yd open.

GO flips to HOLD if his full 2025 tape shows no sink at the top of hitches, or if his 2026 SOE is below zero after 150 routes.

![Figure 4. All 20 day-3 pairs as forecast errors (A) and with each pair left out (B), gains and hit rates with intervals (C), and the 2026 calls (D).](figures/fig4_shortlist.png)

**What would change the call** (pre-registered): drop the tiebreaker if the 2026 day-3 gain is negative and the 2026 list hits no better than the base rate; upgrade it from a lean to a rule if the list hits 9 of 13. Power is low (0.27 for the gain, 0.20 for the list).

## Limitations

- **Not blind, small day-3 sample.** All three classes were seen while building this; day 3 has 20 pairs from 16 receivers. Dropping any one pair leaves the gain between 0.19 and 0.37, and a test that shuffles last season's SOE within season gives p = 0.01; still, the bootstrap interval includes zero.
- **The downside is open.** The interval rules out gains above +1.0 inch per sd, not losses down to −3.6.
- **Game tracking is a sample.** Up to 40 plays per player, half chosen for high EPA, without defenders or ball. That limits the clips and the drill-to-NFL link, not the season test.
- **Availability and survivorship.** 33 of 149 drill-scored receivers never logged NFL routes with separation; re-weighting for this changes little.
- **The freeze is late.** The 2026 rule was frozen on 24 October 2026, after the season began but before any full-season SOE existed.

## Appendix

**Pre-registration.** `frozen_rule_2026.json` holds coefficients, threshold, names and probabilities (sha256 5c1beda70454…). Primary test: day-3 gain over pick and routes, 18 expected 2026 pairs (61% of day-3 receivers keep 100+ routes).

| Speed benchmark: official 40 vs NFL top speed (90th percentile of per-snap maximum) | ρ (receivers) |
|---|---|
| Full play, 40 standardized within position and class (used in the text) | 0.43 (100) |
| Full play, raw 40, WR and TE pooled (mostly the WR–TE speed gap) | 0.74 (100) |
| Snap to throw on breaking snaps only (short routes cap top speed) | 0.34 (99) |

**Every estimate and its alternatives** (90% intervals; receivers resampled).

| Claim | Primary (n) | Alternatives |
|---|---|---|
| Break G slope, in of SOE per sd | offset forecast −1.2 [−3.6, +1.0] (116) | re-weighted for survival −0.8 [−2.8, +1.3]; without Puka Nacua −0.5 [−2.7, +1.5]; QB-adjusted −1.2 [−3.6, +0.9] |
| Day-3 gain over pick and routes | season folds 0.31 [−0.08, 0.63] (20 pairs) | shrinkage refitted in each fold 0.31 [−0.08, 0.65]; draft-class folds 0.15 [−0.10, 0.39]; rule vs raw last-season SOE 0.09 [−0.19, 0.40]; mean error cut +0.8 in [−0.3, +1.8] |

**Notes.** P is rescaled to held-out hit rates (day 3: 50% forecast, 40% observed). The noise-corrected correlation uses an errors-in-variables model, one slope per route family (399 receiver-family cells). 1 sd of Break G = 0.021 g; catches use 8.8 catch-rate points per yard and 60 targets.

**Rest of the 2026 list** (team; shrunk SOE; P): Troy Franklin (DEN; +3.4 in; 61%); A.J. Barner, TE (SEA; +9.9 in; 60%); Gunnar Helm, TE (TEN; +10.4 in; 59%); Isaiah Bond (CLE; +3.1 in; 56%); Jaylin Lane (WAS; +2.8 in; 47%); Kayshon Boutte (NE; +0.6 in; 44%); Mitch Tinsley (CIN; +1.8 in; 39%); Mitchell Evans, TE (CAR; +6.7 in; 34%); Ja'Tavion Sanders, TE ⚑ (CAR; +4.0 in; 29%). ⚑ = 13 or fewer games in 2025.

Reproduce: `uv run python writeup/candidates/round_17/revised/analysis.py` (about nine minutes) writes every number, figure and this text.
