# The 40 Start Doesn't Travel: Matching Each Pass Rusher's Combine Burst to His Own NFL Starts

**We laid every 2023–25 defensive lineman's tracked 40 start over his own random NFL pass-rush starts, tenth of a second by tenth. Once weight and role are priced, the Combine start previews nothing on Sundays (ρ = 0.07, n = 67), and how well it translates explains no pressure beyond draft slot. What does sort rushers is the NFL first step itself: a lead worth a film check, not a rule.**

## What a team can do with this

1. **Draft room: pay for the stopwatch, not the trace.** After weight and role, the official 40 *time* ranks a rusher's NFL first step at 0.23; the tracked 40 start at 0.07. Do not add credit for a "fast first two steps" in the clip.
2. **Combine protocol.** Neither the 40 nor the pass-rush drill (ρ = 0.04, n = 84) is a snap-cued start from a rusher's own stance, and neither travels. That is the rep to add if the league wants a translatable get-off.
3. **Pro personnel, after year one.** From about 20 random rookie rushes, a bottom-third first step with an above-median 40 start is a minus flag: put on the tape before the depth chart (Eric Watts, Zacch Pickens, Princely Umanmielen in Table 3).

## Why this question

Pass rush is the defensive skill teams pay most for, and "explosive off the ball" is how day-2 and day-3 rushers get sold; the 40's first steps are the Combine's closest thing to a get-off. Since 2023 the Combine is tracked at 10 Hz like NFL games, so a player's Combine rep and his NFL snaps can finally sit on one clock. If the burst carries over, it should move a rusher up the board.

## The metric: Carry-over

**First step** = mean speed 0.3–0.5 s after a player first moves. In that window the 40 and a pass rush are the same movement, a lineman leaving a stance. By 1.5 s they part: the median DL is at 6.8 yd/s in the 40 and 3.1 yd/s on a rush, because he has met a blocker.

**Carry-over = NFL first step minus Combine first step**, each in standard deviations after weight and role (share of snaps at EDGE). Positive means he starts faster on Sundays than his 40 promised.

**Data.** Combine: every tracked 40 rep, year-standardized. NFL: only the random half of the game sample (the top-EPA half oversamples sacks), pass plays where he was set at the snap, adjusted for alignment: 1563 starts, median 19 per player. Outcome: career regular-season pressures per pass rush (rushers with 100+ rushes), shrunk toward the mean. **Baseline without tracking:** log draft pick (UDFA = 262), weight and role; the forecast for each rusher is fit on the other two classes.

**Is it real?** The NFL first step over 19 starts has reliability 0.64 and tracks official NGS get-off at ρ = 0.51 snap by snap; the 40 first step repeats at 0.54 across two reps (n = 56).

![Figure 1. Three rushers on one clock from first movement. Dashed: his Combine 40 start. Thin blue: each random NFL pass-rush start; thick: their mean. Yellow: the first-step window. Bars: career pressure rate against the no-tracking forecast fit on the other two classes. Young's 40 undersold him; Watts's 40 start never showed up.](figures/fig1_three_rushers.png)

## Finding 1: the 40 translates only through size

![Figure 2. Rank correlation at each tenth, 90% bootstrap bands. Dashed black: 40 speed vs NFL speed at the same moment, raw. Orange: the same after weight and role. Blue: NFL speed vs pressure rate after pick, weight and role.](figures/fig2_translation_payoff.png)

Raw, the 40 at 1.5 s ranks NFL speed at 1.5 s at 0.57. That is weight: heavier linemen are slower at 1.5 s on Sundays (ρ = 0.82) and in the 40 (0.65). After weight and role no tenth from 0.3 to 1.5 s exceeds 0.11. It is not just noise: even correcting both sides for unreliability, the upper bound on the first-step link is 0.47.

## Finding 2: translation does not pay; the NFL first step leans

| Claim (DL) | Estimate [90%] | n | Null tested | p |
|---|---|---|---|---|
| 40 first step → NFL first step, after weight and role | 0.07 [−0.14, 0.28] | 67 | no link (40 shuffled within class) | 0.59 |
| Official 40 time → NFL first step, same | 0.23 [0.06, 0.41] | 67 | same | 0.07 |
| Carry-over → pressure beyond pick, weight, role | 0.18 [−0.10, 0.43] | 52 | same | 0.25 |
| NFL first step → pressure, same | 0.22 [0.01, 0.42] | 69 | same, one window | 0.08 |
| … best of 13 tenths | — | 69 | no tenth matters (max over windows) | 0.13 |
| … beyond season NGS get-off | 0.12 | 69 | same | 0.33 |
| Held-out R² change, NFL first step | −0.113 [−0.328, 0.025] | 69 | no gain | interval |
| Beat out-of-class forecast: top vs bottom third | 61% vs 13% | 23 + 23 | equal rates (Fisher) | 0.002 |
| … window picked on the other two classes | 65% vs 14% | 20 + 21 | same | 0.001 |
| … 2025 only, top third | 2 of 7 | 7 | 2023–24 rate holds | 0.04 |

*Table 1. Spearman partial correlations, bootstrap intervals by player, 2,000 within-class shuffles. "Beat" = career pressure rate above the forecast fit on the other two classes; base rate 24 of 69 (35%).*

- **Translation fails.** Carry-over and four other similarity measures (speed ratios, curve shape, curve distance; none chosen on pressure) never exceed 0.18 in absolute value, and carry-over lowers held-out R² (−0.114).
- **The NFL first step sorts direction, not size.** Top-third first steps beat their forecast 14 of 23 times (exact 90% interval 42–78%), bottom third 3 of 23. That holds when the window is chosen on the other two classes (13 of 20 vs 3 of 21), although the chosen window moves between folds. As a linear term it hurts held-out R², because one outlier (Maason Smith) dominates 2024 (−0.401). Use it as a sign, not a number.
- **The clean class disagrees.** 2025 was not seen when the window was set. Its top third went 2 of 7; if the 2023–24 rate (12 of 16) were true, a result that poor has probability 0.04. At the base rate it is unremarkable (0.54). One season of rushes is thin, but this is why the call is a film check.

![Figure 3. Each rusher's 40 first step (horizontal) against his NFL first step (vertical), both after weight and role. Colour: pressure rate minus the out-of-class forecast. Corners: pooled pressure rate, forecast and share beating it.](figures/fig3_carryover_map.png)

Split by the 40 instead of the NFL start, the beat rates are 48% vs 33%.

**Plus call vs no call, EDGE and interior:**

| NFL first step (third) | EDGE: n, pressure vs forecast, beat | Interior: n, pressure vs forecast, beat |
|---|---|---|
| Top third (credit) | 14: 10.3% vs 9.0%, 8 of 14 | 9: 7.7% vs 7.7%, 6 of 9 |
| Middle (no call) | 12: 8.2% vs 8.9%, 3 of 12 | 11: 7.1% vs 7.2%, 4 of 11 |
| Bottom third (no credit) | 10: 8.0% vs 8.8%, 1 of 10 | 13: 5.9% vs 6.7%, 2 of 13 |

*Table 2. Thirds of NFL first step among all 69 rushers; "a of b" = rushers beating their out-of-class forecast.*

![Figure 4. Fit on two classes, score the third. Filled: pooled change in rush-weighted R² over pick, weight and role (baseline 0.462), 90% interval from resampling held-out rushers. Hollow: each held-out class.](figures/fig4_out_of_class.png)

**Why DL.** On the same pipeline no group's 40 start survives size (after weight: WR −0.14, OL −0.31, DB −0.10, TE 0.05), and only DL's NFL first step leans toward production (0.22, WR −0.05). OL, DB and TE starts are too noisy on 20 random snaps to read (reliability 0.14, 0.20, 0.19).

## Decision card

One plain call per rusher. Plus = top-third NFL first step; minus flag = bottom third with an above-median 40 start; film first = bottom third without one. Rows marked * are the clean 2025 class.

| Rusher (class, pick, role) | 40 first step, pct | NFL first step, pct (starts) | Pressure rate vs forecast | Call |
|---|---|---|---|---|
| Warren Brinson (2025*, 198, interior) | 56 | 93 (19) | 6.9% vs 6.2% | Plus: credit the start |
| Ashton Gillotte (2025*, 66, EDGE) | no 40 | 90 (20) | 6.4% vs 8.9% | Plus: credit the start |
| Princely Umanmielen (2025*, 77, EDGE) | 65 | 3 (16) | 5.0% vs 9.2% | Minus flag: 40 start not on tape |
| Maason Smith (2024, 48, interior) | 67 | 100 (18) | 3.9% vs 7.7% | Plus: credit the start |
| Ruke Orhorhoro (2024, 35, interior) | 35 | 99 (18) | 8.2% vs 8.2% | Plus: tape over the 40 |
| YaYa Diaby (2023, 82, EDGE) | 98 | 94 (18) | 10.5% vs 8.0% | Plus: credit the start |
| Keeanu Benton (2023, 49, interior) | 8 | 30 (19) | 6.8% vs 7.3% | Film first: grade hands and power |
| Zacch Pickens (2023, 64, interior) | 87 | 9 (20) | 6.1% vs 7.3% | Minus flag: 40 start not on tape |
| Eric Watts (2024, UDFA, EDGE) | 81 | 1 (20) | 3.6% vs 7.0% | Minus flag: 40 start not on tape |

*Table 3. Percentiles among the 69 rushers with 100+ rushes, after weight and role.*

- **Same slot, opposite calls.** Gillotte (pick 66) and Umanmielen (pick 77) are 2025 EDGEs. Gillotte's NFL start is at the 90th percentile, Umanmielen's at the 3rd despite a 65th-percentile 40 start. For year-two snaps the call prefers Gillotte; both trail their forecast, so it is a film question, not a verdict.
- **Maason Smith is the cautionary plus:** the best NFL first step in the cohort and 3.9% against 7.7%. A fast start without a finish is not a pass rush.
- **Keeanu Benton is film first:** neither start is fast; grade his hands and power, which this metric cannot see.
- **Pre-registered for the 2026 class:** after ≥8 random rookie starts, apply the same thirds and calls; we will score them against the same out-of-class forecast after their first season.

## Limitations

- **Mostly nulls.** Translation is near zero after size and fails out of class; DL was chosen as the clearest of five groups, and the first-step window was read off Figure 2 (best-of-13 p = 0.13).
- **Small, sampled tape.** About 20 random NFL starts per player, 52 rushers with a 40, a pressure rate and NFL starts; 2025 has one season.
- **No context.** The game sample has no ball, blockers or opponents, so a slow first step can be a stunt, a two-gap call or a slide toward him.
- **Construct.** The 40 has no snap. Both sides are anchored at first movement, so snap reaction is excluded by design.
- **Outcome.** Pressure is charted and usage-dependent; injuries and rotation are not modelled.

*Reproducible: `uv run python writeup/candidates/round_09/cand_3/analysis.py` (about 30 s) writes every figure, table and number and renders this file.*
