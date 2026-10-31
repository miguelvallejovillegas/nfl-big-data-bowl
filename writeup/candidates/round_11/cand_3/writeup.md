# The 40 Start Doesn't Travel. The NFL First Step Does.

**We put each 2023–25 defensive lineman's tracked Combine 40 start on the same clock as his own random NFL pass-rush starts. Once size is removed, the 40 start barely previews how fast he leaves his stance on Sundays, and the gap between the two adds no pressure beyond draft slot. The NFL start itself carries information. Scored on draft classes the model never saw, a faster first step leaned toward more pressure than a no-tracking forecast in all three, though the clean 2025 class alone is not convincing. On the snaps where a rusher started faster than his own norm, his odds of pressure rose by half. Use it as a film lead after about 20 rookie rushes, not as a draft number.**

## Why this question

Day-2 and day-3 rushers are sold as "explosive off the ball", and the 40's first steps are the Combine's closest thing to get-off. Since 2023 the Combine is tracked at 10 Hz like NFL games, so we can ask directly: does the Indianapolis burst show up in his rushes, and if not, what should replace it?

## The metric

**First step: how fast a lineman is moving 0.3 to 0.5 seconds after he first moves.** There the 40 and a pass rush are the same movement, a lineman leaving a stance. By 1.5 s they part: the median lineman is at 6.8 yd/s in the 40 and 3.1 yd/s on a rush, because he has met a blocker.

**Carry-over = NFL first step minus Combine first step**, each in standard deviations after weight and role (share of snaps at EDGE). Positive means he starts faster on Sundays than his 40 promised.

**Data.** Combine: every tracked 40 rep, year-standardized. NFL: random sampled pass plays only (the top-EPA half oversamples sacks), alignment-adjusted: 1563 starts by 85 linemen, median 19 each. Over that many starts the NFL first step has reliability 0.64, and snap by snap it tracks official NGS get-off (ρ = 0.51). Outcome: career pressures per pass rush. **Baseline without tracking:** log draft pick (undrafted = 262), weight and role, always fit on the other two classes.

![Figure 1. Six rushers: 40 start vs own NFL starts; right, pressure minus forecast (80%).](figures/fig1_six_rushers.png)

## Finding 1: the 40 start translates only through size

Raw, the 40 at 1.5 s ranks NFL speed at 1.5 s at 0.57. That is weight: heavier linemen are slower on Sundays (ρ = 0.82) and in the 40 (0.65). After weight and role the 40 first step previews the NFL first step at ρ = 0.07 [−0.14, 0.28] (n = 67), and no tenth of a second from 0.3 to 1.5 s exceeds 0.11 (Figure 2A). Corrected for noise on both sides, the link is at most 0.47. The official 40 *time* does slightly better (0.23).

![Figure 2. A: translation and payoff by tenth of a second. B: held-out slopes per class.](figures/fig2_translation_heldout.png)

## Finding 2: one held-out test, three measures

Same test for every measure: hold out a class, fit the no-tracking forecast on the other two, then ask how far each standard deviation of the measure moves the held-out class's pressures away from that forecast. Class slopes are pooled with shrinkage; the null shuffles the measure within class 2,000 times.

| Measure (DL) | 2023 | 2024 | 2025 (clean) | Pooled, pressure-rate pts per sd [90%] | Shuffle p | n |
|---|---|---|---|---|---|---|
| Combine 40 start | 0.0 | 0.7 | 1.0 | 0.3 [−0.2, 0.7] | 0.48 | 52 |
| Carry-over (translation) | 0.4 | −0.2 | 0.2 | 0.2 [−0.1, 0.5] | 0.48 | 52 |
| **NFL first step** | 1.4 | 0.2 | 0.4 | **0.8 [0.1, 1.4]** | **0.01** | 69 |

*Table 1. Quasi-binomial slope, out-of-class forecast as offset, random-effects pooling. 40 rows need a tracked 40 and 100+ rushes.*

- **Translation does not pay.** Carry-over is flat; four other similarity measures (speed ratios, curve shape, curve distance) never exceed 0.18 in rank correlation with pressure beyond pick.
- **The NFL first step leans positive in 3 of 3 classes.** The first-step window was read off Figure 2A. Taking the best of 11 windows under the same shuffle, the adjusted p is 0.02.
- **2025 alone is weak:** +0.4 [−0.6, 1.4], one-sided p = 0.26. 2024 is weakest; its top first step is Maason Smith (below).
- **Size of the effect.** One sd of NFL first step is worth about 2.3 pressures [0.3, 4.1] over a 292-rush season.

![Figure 3. 40 first step vs NFL first step; colour = pressure minus forecast.](figures/fig3_carryover_map.png)

## Mechanism: same rusher, faster snap, more pressure

Season pressure is a noisy outcome. A sharper test compares each rusher with himself, snap by snap. On 1279 random pass rushes by 69 linemen (120 pressures), a first step one sd faster than his own average multiplies the odds of pressure on that snap by 1.49 [1.25, 1.77]. Shuffling first steps within each rusher never matched it in 500 tries (p = 0.002). His fastest fifth of starts ended in pressure 14.8% of the time, his slowest fifth 6.6% (Figure 4).

- **Not the stopwatch.** With that snap's NGS get-off, sideways movement, passing down and alignment in the model, the odds ratio is 1.51. Get-off itself adds nothing (0.98, p = 0.85): what matters is the speed after the first movement, not how soon it starts.
- **Same direction in every slice:** 1.31, 1.51 and 1.77 by class; 1.55 for EDGE and 1.41 inside.
- **Football reading:** reach the blocker's set point first and the tackle opens early, the guard catches instead of punching. Without blockers in the data, a free rush could also produce a fast start (Limitations).

![Figure 4. Within-rusher snap test: pressure by first-step fifth, and odds ratios.](figures/fig4_snap_mechanism.png)

## When the call is wrong: Maason Smith

Smith (2024, pick 48, Jacksonville) rushed 98% of his snaps inside. After weight and role his NFL first step is the cohort's best, his 40 start 67th percentile, his NGS get-off 46th. He produced 17 pressures on 433 rushes, 3.9% against a 7.7% forecast; only 18% came quickly (30% for the cohort). His curve is fast for a 306-lb tackle through half a second, then flat at median speed (Figure 1): he arrives, he does not win. His starts are among the straightest (25th percentile sideways), so it is not a stunt artifact. Film cues: do the hands land inside at contact, is there a counter when the rep stalls, is he reading run first? Trusting the plus call cost about 11 pressures a season against forecast. One team and scheme in both years, so player and call cannot be separated.

## So what for a team

The rule is frozen before any 2026 data: config hash `345ec61c97ad` in `frozen_rule.json`. **Plus** = top third of NFL first step. **Minus flag** = bottom third after an above-median 40 start. **Film first** = bottom third without one. **Sideways** = percentile of sideways movement in the first 1.5 s, a stunt and slant proxy for scheme. **At stake** = pressures per season his first step is worth at the held-out slope, i.e. the cost of a wrong call. * = 2025 class.

| Rusher (class, pick, team, role) | 40 start | NFL first step (starts) | Sideways | Pressure vs forecast | At stake | Call: action |
|---|---|---|---|---|---|---|
| Warren Brinson (2025*, 198, GB, interior) | 56 | 93 (19) | 54 | 6.9% vs 6.2% | 3.6 | **Plus:** More passing-down snaps in 2026; re-score at midseason |
| Ashton Gillotte (2025*, 66, KC, EDGE) | no 40 | 90 (20) | 29 | 6.4% vs 8.9% | 2.5 | **Plus:** Keep snaps; grade the finish on 2025 long rushes |
| Princely Umanmielen (2025*, 77, CAR, EDGE) | 65 | 3 (16) | 90 | 5.0% vs 9.2% | −4.4 | **Minus flag:** Scheme check: grade only his upfield rushes first |
| Maason Smith (2024, 48, JAX, interior) | 67 | 100 (18) | 25 | 3.9% vs 7.7% | 6.5 | **Plus:** No added snaps; grade hands and counter on long rushes |
| Ruke Orhorhoro (2024, 35, ATL, interior) | 35 | 99 (18) | 77 | 8.2% vs 8.2% | 5.6 | **Plus:** Keep role; start real, finish only forecast-level |
| YaYa Diaby (2023, 82, TB, EDGE) | 98 | 94 (18) | 64 | 10.5% vs 8.0% | 3.7 | **Plus:** Add passing-down snaps; start and finish both on tape |
| Keeanu Benton (2023, 49, PIT, interior) | 8 | 30 (19) | 62 | 6.8% vs 7.3% | −1.3 | **Film first:** No tracking credit; grade hands and power |
| Zacch Pickens (2023, 64, CHI/KC, interior) | 87 | 9 (20) | 83 | 6.1% vs 7.3% | −3.4 | **Minus flag:** Scheme check: grade only his upfield rushes first |
| Eric Watts (2024, UDFA, NYJ, EDGE) | 81 | 1 (20) | 38 | 3.6% vs 7.0% | −7.6 | **Minus flag:** Ignore the 40; fewer passing-down snaps until the start improves |

*Table 2. Percentiles among the 69 rushers with 100+ rushes, after weight and role.*

- **Draft room: pay for the stopwatch, not the clip.** Give no credit for a fast Combine start trace; the pass-rush drill does not preview the NFL start either (ρ = 0.04, n = 84).
- **Same slot, opposite calls.** Gillotte (Kansas City, pick 66) and Umanmielen (Carolina, pick 77) are 2025 EDGEs at NFL first-step percentiles 90 and 3. Both trail their forecast. Umanmielen's starts are 90th percentile sideways: the slow start may be Carolina's slant-heavy front, not him.

## Limitations

- **The angle's answer is a null.** Translation is near zero after size. DL was the clearest of five groups screened (40 start after weight, NFL first step: WR −0.14, OL −0.31, DB −0.10, TE 0.05).
- **Small, sampled tape.** About 20 random starts per player; a single start has reliability 0.09. 2025 has one season, and one new class gives the held-out test only 47% power at the pooled slope; about 3 classes are needed for 80%. The 2026 check is monitoring, not proof.
- **No blockers, ball or opponents.** A slow first step can be a two-gap call, a fast one a free rush; the snap-level link may partly run from play to start.
- **NGS already sees much of it.** Season NGS get-off ranks pressure at 0.40 beyond pick; the first step adds 0.12 beyond it (p = 0.33). For a club with NGS this is a tiebreak.
- **Outcome.** Pressure is charted and usage-dependent; injuries are not modelled.

## Appendix: sample flow and technical checks

| Item | Value |
|---|---|
| Linemen with 8+ random NFL starts / their starts | 85 / 1563 |
| ... with a tracked 40 (translation, Finding 1) | 67 |
| ... with 100+ career rushes (Table 1 NFL row, Figure 4) | 69, 1279 starts |
| ... with both (Table 1 Combine and carry-over rows, Figure 3) | 52 |
| 40 first step, two-rep repeatability | 0.54 (n = 56) |
| Round-10 hit count, top vs bottom third beating forecast | 14/23 vs 3/23 |
| Sampled-snap pressure vs career pressure rate | ρ = 0.26 |

*Table 3.*

*Reproduce: `uv run python writeup/candidates/round_11/cand_3/analysis.py` (about 3 minutes, one core).*
