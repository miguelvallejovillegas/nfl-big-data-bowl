# Sharper Combine Route Cuts Don't Make NFL Receivers More Open; Use Last Season's Separation to Pick the Last Roster Spot

*Break G* grades a receiver's route cut from 10 Hz tracking. *Separation over expected (SOE)* is how open he is when the ball arrives, beyond what alignment and cushion predict (Next Gen Stats, full season), in inches.

**One sd of Combine Break G is worth −1.2 inches of season SOE (90% interval −3.6 to +1.0) on 116 receivers from draft classes the model never saw. Between day-3 receivers, letting last season's SOE overrule the draft slot gained +7.4 inches of next-season SOE per override (−2.9 to +18.1).**

![Figure 1. All 99 receivers with 8+ graded NFL breaks: Combine against NFL Break G, with the speed benchmark (A), and five receivers that span the Combine range (B).](figures/fig1_cards.png)

## What Break G measures

Break G is how violently a receiver brakes, turns and drives out of his sharpest route cut, as a share of body weight. It averages tracked acceleration over the second around the slowest point of the first break, so it covers decel into the break, speed through it and re-acceleration out. One sd between receivers is 0.021 g. The same code grades NFL slants, outs, hitches, ins and corners, so drill and game cuts sit on one scale.

## Why receivers, why now

The last receiver spot goes to a day-3 or undrafted player who costs about the minimum (2025 median $1.08M), so a wrong call costs production, not cap. Combine tracking now grades every route-drill cut, but a scout sees a median of 7 graded cuts per receiver and 91% of drill cells hold one rep. To price a player, the cut must be a stable trait that carries to Sundays. Table 1 sets it against straight-line speed.

| Measure (n) | Reliability or transfer [90%] | Reading |
|---|---|---|
| Official 40 → NFL top speed (100 receivers) | ρ = 0.43 [0.29, 0.54] | Speed carries |
| Break G, five drills vs the other five (145) | ρ = 0.28 | A weak trait |
| Break G, same drill run twice (59 pairs, 6 drills) | median ICC 0.05 | One rep is mostly noise |
| Route-drill battery, one rep → second rep on three drills | 0.53 → 0.62 | Reps help |
| Drill cut → same NFL route family (99), noise corrected | r = 0.02 [−0.27, 0.32] | **No transfer** |
| One 200-route season of SOE | 60% signal | Last season informs |

## What to do

1. **Take Break G out of the price.** At the same position, clubs draft a receiver 1 sd harder in the cut 9% earlier [2, 17] and get no separation for it.
2. **Run the drill at game speed before timing it.** Combine stems are entered faster than NFL cuts: 6.9 vs 4.7 yd/s into the hitch, 6.9 vs 5.4 into the out, 8.0 vs 6.2 into the in. Cap the stem at game entry speed and run slant, dagger and comeback twice.
3. **Day-3 tiebreak: between two day-3 wide receivers the staff grade even, keep the one with higher last-season SOE, whatever the pick gap.** Confidence: likely but unproven (p = 0.12).
4. **Not for tight ends** (5 pairs, gain −0.50).
5. **2026: GO on Malik Washington, Demario Douglas and Troy Franklin; HOLD on Tez Johnson, Arian Smith and Isaiah Bond.** Review at 150 routes; decide before the March 2027 league year.

## The drill cut does not show up in NFL routes

After correcting for few reps and few snaps, a receiver's drill and NFL cut on the same route family correlate at 0.02; a useful link (above 0.2) has a 17% chance. Splitting Break G into its three mechanics finds no hidden carrier (Figure 2), and the NFL-style reading, speed carried through the turn on slants and outs, does not transfer either (ρ = −0.06 [−0.27, 0.14], 68 receivers).

![Figure 2. Season SOE per sd of Break G and each of its three mechanics on unseen classes, against the pre-set ±3 inch margin, with TOST p and drill reliability.](figures/fig2_chain.png)

Figure 3 shows why, for the six 2026 names. The drill asks for a long, fast stem into the cut; the same NFL route is entered slower and shorter. The drill grades a different movement from the one it should predict.

![Figure 3. The six 2026 receivers: Combine drill rep and median-speed NFL snap on the same route family, aligned at the cut, against the league-median NFL path, with entry speed and Break G.](figures/fig3_path.png)

**The null in names.** Wide receivers with 200+ career routes and 5+ graded NFL breaks (49 eligible). Turn speed is his median NFL break minus the league median for the route.

| Wide receiver (class, pick) | Drill Break G, z | NFL turn speed vs league, yd/s (breaks) | Career SOE, in (routes) |
|---|---|---|---|
| ***Three hardest*** | | | |
| Kayshon Boutte (2023, 187) | +1.8 | +0.6 (16) | +0.5 (873) |
| Adonai Mitchell (2024, 52) | +1.7 | +0.5 (23) | −0.6 (486) |
| Arian Smith (2025, 110) | +1.7 | 0.0 (12) | +15.8 (231) |
| ***Three softest*** | | | |
| Jalin Hyatt (2023, 73) | −0.9 | −0.1 (18) | +2.6 (679) |
| Isaiah Bond (2025, UDFA) | −1.3 | +0.5 (10) | +4.3 (333) |
| Jake Bobo (2023, UDFA) | −1.4 | +0.2 (25) | −0.8 (306) |

Across the eight hardest and eight softest drill cutters, career SOE averages −0.5 and −2.3 inches, and NFL turn speed is the same (0.0 and +0.1 yd/s against the league).

## Break G adds nothing to season separation

The outcome is route-weighted career SOE (213 player-seasons, median 231 routes). The baseline without tracking was fixed before the test: log pick, official 40, weight, NGS athleticism score and a TE flag, fitted on two classes and scored on the third. It is weak (held-out R² 0.00, error 8.6 inches), and adding Break G changes R² by 0.013 [−0.157, 0.085].

A benefit of +3 inches per sd (about 0.44 catches a season) is excluded: the upper bound is +1.0. Equivalence is not shown: the TOST against ±3 inches gives p = 0.10, because the harm side reaches −3.6. The reading is "no benefit", not "no effect". Across the sensitivity grid (shrinkage, the 8-break and 100-route cut-offs; 13 variants) the slope stays between −1.3 and −0.9.

## Day 3: last season beats the draft slot, narrowly

The pre-registered test adds shrunk last-season SOE to a pick-and-routes forecast, fitted on one season pair and scored on the other. On 20 day-3 pairs (16 receivers) it lifts the rank correlation with next-season SOE by 0.31 [−0.08, 0.63]; shuffling whole receivers' SOE gives one-sided p = 0.12. Across the sensitivity grid the gain runs from 0.11 (50+ routes) to 0.40.

Table 2 turns that into the decision on 51 same-season contrasts between day-3 wide receivers. An override is a contrast where last-season SOE and the earlier pick disagree; it pays while the wrong rate stays under break-even.

| Override allowed when | Overrides | Wrong per 100 [90%] | Break-even wrong per 100 | Cost of a wrong override | Net SOE per override, in [90%] |
|---|---|---|---|---|---|
| Pick gap under 16 | 7 | 43 [17, 72] | 86 | 4.3 in (0.6 catches) | +12.7 [−6.1, +29.4] |
| Pick gap under 32 (pre-set) | 13 | 46 [25, 68] | 67 | 9.2 in (1.4 catches) | +5.9 [−6.1, +20.0] |
| **Any gap (the rule)** | 19 | 37 [21, 56] | 67 | 8.1 in (1.2 catches) | +7.4 [−2.9, +18.1] |

A pick-gap threshold does not lower the wrong rate, so the rule has none. Inside day 3 the earlier pick orders receivers right only 52% of the time; that is why overriding it costs little.

![Figure 4. Day-3 wide receivers' forecast errors, ranked (A), the override rule against its break-even (B), and gains with intervals (C).](figures/fig4_tiebreak.png)

## The 2026 calls

Rule, written before the back-test: day-3 or undrafted wide receiver, 14+ games and SOE above zero in 2025, model chance of 2026 SOE above zero of 50% or more. **GO** = also listed a year earlier, with a hit; **HOLD** = first listing; a HOLD becomes GO with a 2026 hit. True SOE = 2025 SOE shrunk for routes. The drop-off point is the 2026 SOE over his first 150 routes that would pull that estimate below zero.

| Receiver (class, round-pick; 2025 team) | Call | True SOE, in [90%] | Drops off if first 150 routes of 2026 below | Snap SOE man / zone, in (snaps) | Snap SOE slot / wide, in (snaps) |
|---|---|---|---|---|---|
| **Malik Washington** (2024, R6-184; MIA) | **GO** | +16.1 [+7.7, +24.4] | −51 in | −1 (5) / +23 (34) | +24 (11) / +19 (26) |
| **Demario Douglas** (2023, R6-210; NE) | **GO** | +13.4 [+3.7, +23.0] | −32 in | +33 (14) / −13 (26) | +5 (27) / −4 (11) |
| **Troy Franklin** (2024, R4-102; DEN) | **GO** | +3.4 [−4.0, +10.9] | −14 in | −16 (9) / −17 (31) | +7 (12) / −24 (27) |
| Tez Johnson (2025, R7-235; TB) | **HOLD** | +10.9 [+2.0, +19.8] | −30 in | +4 (14) / +3 (26) | −13 (14) / +9 (25) |
| Arian Smith (2025, R4-110; NYJ) | **HOLD** | +10.0 [+0.5, +19.5] | −24 in | −9 (11) / +14 (29) | −6 (17) / +18 (23) |
| Isaiah Bond (2025, UDFA; CLE) | **HOLD** | +3.1 [−5.3, +11.5] | −10 in | +11 (14) / +36 (26) | +25 (16) / +29 (24) |

![Figure 5. The 2026 list with intervals, calls and drop-off points (A), and each name's sampled-snap SOE by coverage and alignment (B).](figures/fig5_calls.png)

**What flips each name.** **Washington**: off below −51 in; film gap man (5 snaps). **Douglas**: off below −32 in; film gap wide (11 snaps). **Franklin**: off below −14 in; film gap man (9 snaps). **Johnson**: off below −30 in; film gap man (14 snaps). **Smith**: off below −24 in; film gap man (11 snaps). **Bond**: off below −10 in; film gap man (14 snaps). Washington and Douglas would need a collapse to drop; Franklin and Bond would drop after an ordinary bad stretch. Only 2 of 24 coverage and alignment cells reach 30 snaps, so the splits steer film, not the call. Arian Smith is one of the hardest drill cutters; his case rests on SOE, not the cut.

**Back-test.** Applied unchanged to the 2023 and 2024 lists, with forecasts fitted on the other season, the rule's names hit (next-season SOE above zero) 3 of 5 times (60% [26, 87]); the same number of earliest-picked day-3 receivers hit 3 of 5. On this few names the back-test cannot separate the rule from the draft slot; the support for the calls is the pair-level test above.

## Limitations

- **Not blind, small day-3 sample.** All three classes were seen while building this; the day-3 test has 20 pairs, the override table 19 overrides that share receivers.
- **Harm is not ruled out.** The Break G interval reaches −3.6 inches per sd.
- **Game tracking is a sample.** Up to 40 plays per player, half chosen for high EPA, without defenders or ball. That limits Figure 3, the splits and the drill-to-NFL link, not the season tests, which use full-season NGS separation.
- **The 40.** The tracked clip starts before the laser, so speed numbers use the official time.
- **Survivorship.** 33 of 149 drill-scored receivers never logged NFL routes with separation; the back-test scores only receivers with 100+ routes the next season.
- **The protocol change is inferred** from entry-speed gaps; no drill was run at game speed.

## Appendix

`analysis.py` writes every number, figure and this text from `data/processed`; `verify.py` rebuilds season SOE from the raw NGS separation in `player_season_stats` and re-checks the headline numbers in seconds. Robustness checks and the sensitivity grid are in `robustness.csv` and `sensitivity.csv`. Catches use 8.8 catch-rate points per yard and 60 targets.
