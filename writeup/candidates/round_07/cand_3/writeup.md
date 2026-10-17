# A Pass Rusher's 40 Is Worth About Two NFL Rushes of Film

**We measured how hard each 2023–25 defensive lineman launches off the line, both in his Combine 40 and on about 20 random NFL pass rushes. His 40 previews his Sunday launch, but only as much as 1.7 NFL rushes would, so at draft time it explains nothing about pressure beyond draft slot.**

| Question | Answer | Players |
|---|---|---|
| What is the 40 worth for knowing a rusher's NFL launch? | **1.7 rushes of film** (80% interval 0.3 to 7.1) | 67 |
| Does the 40 launch explain pressure beyond pick, weight and role? | No: partial ρ 0.06 [−0.32, 0.36]; held-out R² change −0.020 | 52 |
| Does the Sunday launch? | In sample, weakly: 0.28 [−0.04, 0.58]. Out of class, no: −0.022 | 52 |

![Figure 1. Takeaway: adding the Combine 40 equals watching 1.7 more NFL rushes (A), and it does not improve a pressure forecast at any amount of film (B). A: share of a rusher's true NFL launch that is known, from film alone (grey) or from the 40 plus film (blue). B: how well the launch estimate ranks pressure rate beyond the no-tracking baseline.](figures/fig1_exchange.png)

## Why this question

1. **One movement, two settings.** A lineman's first steps from a stance are the closest match between a Combine rep and an NFL snap; scouts read the 40's first ten yards as a get-off proxy.
2. **Tracking now measures both** with the same 10 Hz speed trace, for three classes.
3. **The decision needs a unit.** Weighing a Combine number against a few games of film needs a common currency, so we price the Combine rep in NFL snaps.

## The metric

**Launch** is how hard his first steps push him out of his stance: the largest gain in speed over any tenth of a second in his first second of movement (yd/s²). One function computes it on the Combine 40 (from first movement, as the 40 has no snap) and on NFL pass rushes where he was set at the snap (from the snap). The median DL launches at 5.4 yd/s² in the 40 and 4.2 on Sundays. On single rushes, launch tracks the official NGS get-off time at ρ = −0.51 (n = 1,271; a harder launch means a faster get-off).

**Film Equivalent (FE)** is how many NFL snaps of the same movement a Combine measurement is worth. One random rush is mostly noise (reliability 0.10). The 40 explains 0.15 of the true spread in Sunday launch. FE = k·R²/(1 − R²) converts that share into rushes, where k = 9.5 is the number of rushes that carry as much noise as there is signal.

**Translation gap** is his Sunday launch minus what his 40 predicted, shrunk for his number of rushes, in sd of true launch.

**Data and baseline.** 67 DL have a tracked 40 and 8+ random NFL rushes (1,271 rushes, median 20; the top-EPA half of the sample is excluded because it oversamples sacks). Of these, 52 have a career pressure rate (median 6.5%). The no-tracking baseline is log draft pick (UDFA = 262), weight, NFL EDGE share and class. The official 40 time and 10-yd split compete on the same players.

## Finding 1: the start carries over, cheaply

One pre-specified matched movement per group went through the same pipeline (Table 1, Figure 2). Only DL launch both carries over and pays. WR straight-line speed carries over more (r = 0.39; the 40 is worth 6.7 vertical routes), but NFL vertical speed does not track yards per route (−0.23). For OL, DB and TE, one NFL snap is too noisy (reliability 0.03 or less) to price.

| Group | Combine vs NFL movement | n | r [95%] | FE [80%] | NFL movement vs production |
|---|---|---|---|---|---|
| DL | 40 launch vs pass-rush launch | 67 | 0.32 [0.03, 0.56] | 1.7 rushes [0.3, 7.1] | 0.28 [−0.04, 0.58] (pressure rate) |
| WR | 40 top speed vs go/post/corner top speed | 60 | 0.39 [0.19, 0.57] | 6.7 vertical routes [1.7, 99+] | −0.23 [−0.54, 0.14] (yards per route) |
| OL | pass-set drill vs NFL pass set (speed at 0.5 s) | 74 | 0.21 [0.04, 0.38] | too noisy | −0.10 [−0.35, 0.16] (pressures avoided) |
| DB | back-pedal drill launch vs coverage launch | 72 | 0.25 [0.04, 0.44] | too noisy | 0.01 [−0.25, 0.28] (solo tackles/100) |
| TE | 40 top speed vs route top speed | 23 | 0.07 [−0.41, 0.52] | too noisy | −0.29 [−0.65, 0.31] (snaps per season) |

*Table 1. One matched movement per group, random NFL snaps only. r = correlation of player means. Production = partial Spearman ρ of the NFL movement beyond pick, weight and class.*

![Figure 2. Takeaway: only DL launch both carries over from the Combine (A) and tracks production on Sundays (B). Grey rows are too noisy on the NFL side to price.](figures/fig2_groups.png)

For DL, the 40 launch ranks NFL launch at r = 0.32 [0.03, 0.56] (0.39 corrected for game-side noise); two 40 reps agree at 0.61 (n = 56).

**The stopwatch does as well.** The official 40 time gives r = 0.36 (FE 2.4 rushes), and the 10-yd split gives 0.33 (FE 1.9). The tracked pass-rush drill gives 0.21 (FE 0.7, n = 84). Beyond the official 40 and weight, tracked launch adds 0.19.

## Finding 2: translation does not explain pressure beyond pick at draft time

![Figure 3. A: partial rank correlation with NFL pressure rate beyond pick, weight, EDGE share and class, all on the same 52 rushers. B: held-out R² change when each class is predicted from the other two, with every fitted piece (noise split, Combine-to-NFL map, shrinkage) refit inside the training classes. Arrows mark intervals clipped at the axis.](figures/fig3_pressure.png)

- **No draft-time signal.** The 40 launch has partial ρ 0.06 [−0.32, 0.36] with pressure. Out of class, it changes held-out R² by −0.020 [−0.329, 0.125] on a baseline of 0.205.
- **The arithmetic says it must be small.** Translation (0.39) times the Sunday payoff gives an expected draft-time effect near 0.14. Detecting that with 80% power needs about 430 rushers; we have 52.
- **The 40 time works, but not through launch:** official 40 0.30, 10-yd split 0.25.
- **The Sunday launch carries the in-sample signal.** NFL launch gives 0.28 [−0.04, 0.58], positive in 96% of bootstraps, and the gap gives 0.26. Given the Sunday launch, the gap adds 0.04 and the 40 adds −0.04.
- **Out of class, nothing survives.** Held-out R² changes by −0.022 [−0.449, 0.238] for Sunday launch and −0.073 for the gap. Each lowers held-out error in 1 of 3 classes (2025, n = 10).

**Film carries the signal (Figure 1B):** film alone ranks pressure at 0.15 after 2 rushes and 0.28 after 12; adding the 40 gives 0.15 and 0.30.

**Planning number.** One sd of Sunday launch is worth 0.7 points of pressure rate (90%: 0.0 to 1.4). At the median 168 rushes a season, that is 1.2 pressures (0.0 to 2.3). In the same model, moving from pick 100 to pick 80 is worth 0.2 points (0.3 pressures). The launch effect is about 4 times larger, in sample only.

## EDGE vs interior: translation and payoff sit in different players

| Role | n | 40 vs NFL launch, r | NFL launch vs pressure (95%) | Pressure rate, top vs bottom third of NFL launch | Beat forecast, top vs bottom third |
|---|---|---|---|---|---|
| EDGE | 24 | 0.02 | 0.50 [0.02, 0.84] | 10.4% vs 6.4% | 75% vs 25% |
| interior | 28 | 0.37 | 0.12 [−0.30, 0.59] | 6.6% vs 5.7% | 78% vs 30% |

*Table 2. Split by NFL role (EDGE share ≥ 50%), thirds within role. Exploratory: the split was not pre-specified.*

For **EDGE**, the 40 does not preview launch (0.02), but launch pays: the top third pressure at 10.4% against 6.4% (n = 24). For **interior** linemen it is closer to the reverse: the 40 previews launch (0.37), but the pressure gap is small (6.6% vs 5.7%, partial ρ 0.12), although top-third interior rushers still beat their forecast more often. Both intervals are wide.

## Named rushers

| Rusher (class, pick, role) | 40 launch | NFL launch (rushes) | Gap, sd (90%) | Pressure rate vs forecast |
|---|---|---|---|---|
| Maason Smith (2024, 48, interior) | 5.4 | 5.2 (18) | +2.21 ± 0.94 | 3.9% vs 7.3% |
| Byron Young (2023, 77, EDGE) | 5.6 | 5.0 (19) | +1.81 ± 0.92 | 11.3% vs 8.3% |
| Ruke Orhorhoro (2024, 35, interior) | 5.8 | 4.9 (20) | +1.51 ± 0.91 | 8.2% vs 8.0% |
| Chop Robinson (2024, 21, EDGE) | 5.8 | 4.8 (16) | +1.22 ± 0.97 | 11.8% vs 10.1% |
| Leonard Taylor (2024, UDFA, interior) | 5.2 | 3.7 (20) | −0.82 ± 0.91 | 5.2% vs 5.8% |
| Brandon Dorlus (2024, 109, interior) | 5.3 | 3.7 (19) | −0.89 ± 0.92 | 8.5% vs 7.3% |
| Princely Umanmielen (2025, 77, EDGE) | 5.8 | 3.8 (17) | −0.95 ± 0.95 | 5.0% vs 8.0% |
| Eric Watts (2024, UDFA, EDGE) | 7.6 | 3.3 (20) | −2.48 ± 0.91 | 3.6% vs 7.1% |

*Table 3. The four largest and four smallest translation gaps among the 52 (launch in yd/s²; ± is the 90% half-width).*

![Figure 4. Speed traces for the three largest and three smallest gaps: thin lines are single NFL rushes, dashed lines are 40 reps. The flag marks a fast 40 launch with a bottom-third gap.](figures/fig4_rushers.png)

Gaps are noisy: they spread 0.78 sd across players against a posterior sd of 0.56 each, and only 7 of 52 have a 90% interval excluding zero. The top third by gap beat their forecast 71% of the time (8.5% pressure), against 28% (6.0%) for the bottom third (n = 17 each).

- **Byron Young**: ordinary 40 launch (5.6), one of the hardest Sunday launches (5.0), pressure 11.3% against an 8.3% forecast. **Chop Robinson** is similar (11.8% vs 10.1%).
- **Maason Smith** has the largest gap but pressures at 3.9% against 7.3%: launch without a finish.
- **Eric Watts** (UDFA) has the hardest 40 launch in the table (7.6) and the softest Sunday launch (3.3). His gap is −2.48 sd, and he pressures at 3.6% against 7.1%. Fast in Indianapolis, slow off the snap.
- **Film-first flags** have a 40 launch above the median and a bottom-third gap. Of the 7 flagged, 2 beat their forecast (mean 6.8% vs 7.3%).

## What a team can use

1. **Draft room.** Give the tracked 40 launch no weight beyond the stopwatch. It is worth 1.7 rushes, and the official 40 time carries as much.
2. **Pro personnel.** Switch to film early: after about 11 random rushes, film outweighs the 40 in a young rusher's launch estimate. For EDGE rushers, use Sunday launch as a tiebreaker (in sample only). For interior linemen the payoff is smaller and less certain.
3. **Coaching.** A flag like Watts is a question for the position coach (stance, snap reaction, alignment?): a cue for film review, not a grade.

## Limitations

- **Strength.** No result holds out of class, the in-sample payoff interval includes zero, and DL was chosen as the best of five groups after looking.
- **Sample.** We have about 20 random rushes per player (reliability 0.67). The 2025 class has 10 rushers and one season. DL who skipped the 40 are excluded.
- **Construct.** The 40 has no snap and no blocker, while NFL launch includes some snap reaction. The game tracking has no ball and no opponents, so a soft launch cannot be separated from a slide protection or a stunt.
- **FE assumptions.** Rushes are treated as exchangeable, with equal noise for every player. The interval is wide (0.3 to 7.1); groups with one-snap reliability below 0.05 are not priced.
- **Outcome.** Pressure is charted and scheme-dependent; injuries are not modelled.

*Reproducible: `uv run python writeup/candidates/round_07/cand_3/analysis.py` (about one minute) writes every figure, table and number (`numbers.json`, with each number's claim) and renders this file.*
