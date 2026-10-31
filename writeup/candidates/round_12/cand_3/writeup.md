# The 40 Clip Doesn't Travel. The Stopwatch Does.

**We laid every 2023–25 defensive lineman's tracked Combine 40 start over his own random NFL pass-rush starts, on one clock. Once size is removed, the Combine start barely previews how fast he leaves his stance on Sundays, and the gap between the two adds no pressure. Scored only on draft classes the model never saw, the plain stopwatch 40 time does carry: one standard deviation faster is worth +0.9 pressure-rate points beyond draft slot, in all three classes. That is about the gain from moving from pick 64 to pick 28. The NFL first step adds a smaller, less certain +0.4 on top of the stopwatch and NGS get-off.**

**Three actions.** (1) *Draft room:* weight a rusher's official 40 time more than the board does; give no credit for the tracked start trace. (2) *Pro personnel:* after about 20 random rushes, use the NFL first step as a tiebreak and film lead, trusted most when the 40 time agrees. (3) *Contracts:* a fast start without pressure is a tape question, not a buy signal (Maason Smith, below).

| Measure, one sd faster (DL) | n | Beyond pick, weight, role | Beyond that + official 40 + NGS get-off |
|---|---|---|---|
| Tracked 40 start (Combine clip) | 52 | 0.3 [−0.2, 0.7], p 0.48 | 0.2 [−0.4, 0.8], p 0.55 |
| Carry-over (NFL start minus 40 start) | 52 | 0.2 [−0.1, 0.5], p 0.48 | 0.2 [−0.3, 0.7], p 0.38 |
| Official 40 time (stopwatch) | 54 | 0.9 [0.4, 1.5], p 0.004 | in baseline |
| **NFL first step (game tracking)** | 69 | **0.8 [0.1, 1.4], p 0.01** | **0.4 [0.0, 0.9], p 0.13** |

*Table 1. Every test of the angle. Each draft class is held out in turn: the forecast is fit on the other two, and the measure must move the held-out class's pressures away from it. Pressure-rate points per sd (the DL average is about 8%), 90% interval; p from shuffling the measure within class.*

![Figure 1. Six rushers: the 40 start over his own NFL starts; right, pressure minus forecast on one shared axis.](figures/fig1_six_rushers.png)

## The question and the metric

Rushers are sold as "explosive off the ball", and the 40's first steps are the Combine's closest thing to get-off. Since 2023 the Combine is tracked at 10 Hz like NFL games, so the movement itself can be compared.

**First step = how fast a lineman is moving 0.3 to 0.5 s after he first moves.** In that window the 40 and a pass rush are the same movement, a big man leaving a stance. By 1.5 s they part: the median lineman is at 6.8 yd/s in the 40 and 3.1 on a rush, because he has met a blocker. **Carry-over = NFL first step minus Combine first step**, each in standard deviations after weight and role (share of snaps at EDGE).

**Data.** Combine: every tracked 40 rep, standardized within year. NFL: random sampled pass plays only (the top-EPA half oversamples sacks), adjusted for alignment: 1563 starts by 85 linemen, median 19 each. Over that many starts the NFL first step has reliability 0.64; snap by snap it tracks official NGS get-off (ρ = 0.51). Outcome: career pressures per pass rush. **Baseline without tracking:** draft pick, weight and role; the stronger baseline adds the official 40 time and season NGS get-off.

## Finding 1: the start translates only through size

Raw, the 40 at 1.5 s ranks NFL speed at 1.5 s at 0.57. That is weight: heavier linemen are slower on Sundays (ρ = 0.82) and in the 40 (0.65). After weight and role the 40 first step previews the NFL first step at ρ = 0.07 [−0.14, 0.28] (n = 67); no tenth of a second exceeds 0.11 (Figure 2A), and even corrected for measurement noise the link is at most 0.47. The pass-rush drill start does no better (0.04, n = 84).

So the angle's question has a short answer. Five ways of scoring how alike a player's two curves are (gap, speed ratios, shape, distance) never exceed 0.18 in rank correlation with pressure beyond pick, and carry-over fails on held-out classes (Table 1).

![Figure 2. A: translation and payoff by tenth of a second. B: each class held out; grey bands show what pure noise produces.](figures/fig2_translation_heldout.png)

## Finding 2: the stopwatch carries, and it is underpriced

The test that rejects the clip accepts the official 40 time: +0.9 points per sd [0.4, 1.5], n = 54, positive in 3 of 3 classes, and still +0.8 with season NGS get-off in the forecast. Over a 292-rush season that is about 2.6 pressures.

Clubs do pay for it. Among 89 linemen, a one-sd faster 40 goes 50% earlier in pick number after weight and role (p < 0.001); a fast tracked start goes 17% *later* if anything (p = 0.07). Because our test is beyond the pick, the 40 remains under-weighted at the price clubs pay. This is an association, not a market experiment.

## Finding 3: the NFL first step, a smaller lead

Beyond pick, weight and role, a faster NFL first step leans positive in every held-out class: +0.8 points per sd [0.1, 1.4], n = 69. About half is the athlete the stopwatch already measured: with the 40 time and NGS get-off in the forecast it falls to +0.4 [0.0, 0.9], p = 0.13. The clean 2025 class alone is +0.4 [−0.6, 1.4].

The sharper evidence compares each rusher with himself. On 1279 random rushes by 69 linemen (120 pressures), a first step one sd faster than his own average multiplies the odds of pressure on that snap by 1.49 [1.25, 1.77]: his fastest fifth of starts ended in pressure 14.8% of the time, his slowest fifth 6.6% (Figure 4). It holds with that snap's NGS get-off, sideways movement, down and alignment controlled (1.51), in each class, and for EDGE (1.55) and interior (1.41). Football reading: reach the blocker's set point first and the tackle opens early (though a free rush could also produce a fast start).

![Figure 3. 40 first step vs NFL first step; colour = pressure minus forecast.](figures/fig3_carryover_map.png)

![Figure 4. Same rusher, faster snap: pressure by first-step fifth, and odds ratios across controls, classes and roles.](figures/fig4_snap_mechanism.png)

## Is it the forecast, or one player?

A shrunken (ridge) forecast does not improve calibration (held-out slopes 1.23, 1.89, 0.63 against 1.07, 1.42, 0.61; 1 is ideal) and leaves the first-step result at +0.8. Resampling rushers and refitting everything gives [0.2, 1.2]; dropping any single rusher keeps it between 0.6 and 0.9, and dropping Smith raises it (Figure 5).

![Figure 5. Calibration of three forecasts per held-out class, and the first-step test rerun without each rusher.](figures/fig5_robustness.png)

## Decision card

Two signals, read together, both after weight and role. **Both** = NFL first step in the top third and 40 time in the top half; **Start only** = top-third first step, slower or no 40; **Speed only** = bottom-third first step after a fast 40; **Neither**. Pressure against the pick forecast, and rushers who beat it: Both 10.0% vs 8.3%, 8 of 10; Start only 9.1% vs 9.0%, 6 of 13; Speed only 7.3% vs 7.8%, 2 of 8; Neither 6.3% vs 7.4%, 1 of 15. The rule is frozen before any 2026 data (hash `c2da1e9cb748`, `frozen_rule.json`).

| Rusher (class, pick, team, role) | 40 time | 40 start | NFL first step | Pressure vs forecast | Call: action |
|---|---|---|---|---|---|
| Byron Young (2023, 77, LA/LAR, EDGE) | 89 | 21 | 96 (18 starts) | 11.3% vs 8.3% | **Both:** Extension-grade: keep him wide in the Rams' odd front on every passing down |
| Cedric Johnson (2024, 214, CIN, EDGE) | 70 | 83 | 91 (19 starts) | 13.0% vs 7.3% | **Both:** Day-3 hit: give him third-down reps outside before paying a veteran |
| Maason Smith (2024, 48, JAX, interior) | 48 | 67 | 100 (18 starts) | 3.9% vs 7.7% | **Start only:** No snap increase: grade hands at contact and a counter when the 3-tech rush stalls |
| Warren Brinson (2025*, 198, GB, interior) | 33 | 56 | 93 (19 starts) | 6.9% vs 6.2% | **Start only:** Audition: nickel 3-tech on third down in 2026, re-score at midseason |
| Ashton Gillotte (2025*, 66, KC, EDGE) | no time | – | 90 (20 starts) | 6.4% vs 8.9% | **Start only:** Keep snaps; tape question is the finish on long rushes, not the start |
| Laiatu Latu (2024, 15, IND, EDGE) | 65 | 40 | 25 (20 starts) | 10.1% vs 10.3% | **Speed only:** Meets forecast on speed and timing, not burst: keep him wide; do not project a jump |
| Eric Watts (2024, UDFA, NYJ, EDGE) | 83 | 81 | 1 (20 starts) | 3.6% vs 7.0% | **Speed only:** Ignore the stopwatch here: early-down edge-setter until the start shows |
| Princely Umanmielen (2025*, 77, CAR, EDGE) | 15 | 65 | 3 (16 starts) | 5.0% vs 9.2% | **Neither:** No tracking credit; check Carolina's slant calls before grading his rush |

*Table 2. Percentiles among the 69 rushers with 100+ rushes. * = 2025 class (one season).*

**When the call is wrong: Maason Smith.** Jacksonville's 2024 second-rounder has the cohort's best NFL first step but a 48th-percentile 40 time, and produced 3.9% pressure against a 7.7% forecast; only 18% of his pressures came quickly. His curve is fast for a 306-lb tackle through half a second, then flat (Figure 1): he arrives, he does not win. His starts are among the straightest (25th percentile sideways), so it is not a stunt artifact. Umanmielen shows the clip's trap: a 65th-percentile 40 start but a 15th-percentile stopwatch, and NFL starts that are slow and 90th-percentile sideways (Carolina's slanting front, or him).

## Limitations

- **The angle's answer is a null.** Translation is near zero after size, and the strongest Combine signal is the stopwatch, not tracking. DL was the clearest of five groups screened; for WR, OL, DB and TE the NFL start cannot be read reliably from about 20 sampled snaps.
- **Small, sampled tape.** About 20 random starts per player; one new class gives the held-out test only 47% power. A forward test (rookie-year first step against later pressure, beyond rookie pressure) has only 23 rushers and is inconclusive (p = 0.52).
- **No blockers, ball or opponents.** A slow first step can be a two-gap call, a fast one a free rush.
- **The richer forecast is poorly calibrated on 2025** (slope 0.22), whose rookies have one partial season.
- **Outcome.** Pressure is charted and usage-dependent; injuries are not modelled. Price findings are associations.

## Appendix: method and sample flow

Held-out test: the forecast, fit on the other two classes, is a fixed logit offset in a quasi-binomial model of the held-out class's pressures out of rushes; class slopes are pooled with random-effects shrinkage. The first-step window was read off the payoff curve in an earlier round; the best of 11 windows under the same shuffle gives p = 0.02. Ridge penalty chosen inside the training classes.

| Item | Value |
|---|---|
| Linemen with 8+ random NFL starts / their starts | 85 / 1563 |
| ... with a tracked 40 (Finding 1) | 67 |
| ... with 100+ career rushes (first-step tests) | 69 |
| ... with both (40 start and carry-over tests, Figure 3) | 52 |
| 40 first step, two-rep repeatability | 0.54 (n = 56) |
| Sampled-snap pressure vs career pressure rate | ρ = 0.26 |

*Table 3.*

*Reproduce: `uv run python writeup/candidates/round_12/cand_3/analysis.py` (about 3 minutes, one core).*
