# The 40 Start Doesn't Travel; the NFL First Step Is a Film Lead

**We laid each 2023–25 defensive lineman's tracked 40 start over his own random NFL pass-rush starts. After weight and role, the 40 previews the NFL first step at ρ = 0.07 (n = 67) and adds no pressure beyond draft slot. The NFL first step does sort rushers: the top third beat a no-tracking forecast 14 of 23 times, the bottom third 3 of 23. But the clean 2025 top third went 2 of 7. A film lead, not a rule.**

## What a team can do with this

1. **Draft room: pay for the stopwatch, not the clip.** After weight and role, the official 40 *time* ranks a rusher's NFL first step at 0.23; the tracked 40 start at 0.07. Give no extra credit for "fast first two steps" on the Combine trace.
2. **Pro personnel, after about 20 rookie rushes.** A top-third NFL first step earns a second film look. A bottom-third start after a fast 40 (Eric Watts, Zacch Pickens, Princely Umanmielen) means tape before the depth chart. Neither moves a player alone.
3. **On every plus, ask whether the burst finishes.** Maason Smith, below, is the miss: the cohort's fastest start, half his forecast pressure.
4. **Combine protocol.** Neither the 40 nor the pass-rush drill (ρ = 0.04, n = 84) is a snap-cued start from a rusher's stance. That is the rep to add.

## Why this question

"Explosive off the ball" is how day-2 and day-3 rushers get sold, and the 40's first steps are the Combine's closest thing to get-off. The logic: a rusher who reaches the blocker's set point first makes the tackle open his hips early and the guard catch instead of punch. Since 2023 the Combine is tracked at 10 Hz like NFL games, so both can sit on one clock.

## The metric: Carry-over

**First step** = mean speed 0.3–0.5 s after a player first moves. In that window the 40 and a pass rush are the same movement, a lineman leaving a stance. By 1.5 s they part: the median DL is at 6.8 yd/s in the 40 and 3.1 yd/s on a rush, because he has met a blocker.

**Carry-over = NFL first step minus Combine first step**, each in standard deviations after weight and role (share of snaps at EDGE). Positive means he starts faster on Sundays than his 40 promised.

**Data.** Combine: every tracked 40 rep, year-standardized. NFL: random sampled pass plays only (the top-EPA half oversamples sacks), set at the snap, alignment-adjusted: 1563 starts, median 19 per player. Outcome: career pressures per pass rush. **Baseline without tracking:** log draft pick (UDFA = 262), weight and role, fit on the other two classes.

**Is it real?** Over 19 starts the NFL first step has reliability 0.64, and snap by snap it tracks official NGS get-off at ρ = 0.51.

![Figure 1. Three rushers on one clock from first movement. Dashed: Combine 40 start; thin blue: each random NFL rush; thick: their mean. Bars: career pressure rate vs the out-of-class forecast.](figures/fig1_three_rushers.png)

## Finding 1: the 40 translates only through size

![Figure 2. Rank correlation at each tenth of a second, 90% bootstrap bands.](figures/fig2_translation_payoff.png)

Raw, the 40 at 1.5 s ranks NFL speed at 1.5 s at 0.57. That is weight: heavier linemen are slower on Sundays (ρ = 0.82) and in the 40 (0.65). After weight and role, no tenth from 0.3 to 1.5 s exceeds 0.11. Correcting both sides for noise, the upper bound on the first-step link is 0.47.

## Finding 2: translation does not pay; the NFL first step leans

| Question (DL) | Estimate [90%] | n | p |
|---|---|---|---|
| Does the 40 start preview the NFL first step (after weight, role)? | ρ 0.07 [−0.14, 0.28] | 67 | 0.59 |
| Does carry-over add pressure beyond pick, weight, role? | ρ 0.18 [−0.10, 0.43] | 52 | 0.25 |
| NFL first step → pressure rate, rushes as exposure | +0.6 pts per sd [0.1, 1.1] | 69 | 0.04 |
| Top vs bottom third beating the out-of-class forecast | 14/23 vs 3/23 | 23 + 23 | 0.001 |

*Table 1. Rows 1–2: partial Spearman, player bootstrap, p from 2,000 within-class shuffles. Row 3: quasi-binomial (pressures out of rushes; dispersion 2.3) with pick, weight, role, rotation share and pick × rotation. Row 4: one-sided Fisher. Base rate 24 of 69 (35%). Technical checks are in the appendix.*

Four other similarity measures (speed ratios, curve shape, curve distance) never exceed 0.18 in absolute value. One sd of NFL first step is worth about 1.7 pressures over a 292-rush season; rotation does not change the pick slope (pick × rotation 0.00, p = 0.97).

![Figure 3. 40 first step vs NFL first step, both after weight and role; colour = pressure rate minus forecast.](figures/fig3_carryover_map.png)

## Does it hold out of class?

![Figure 4. A: top vs bottom third beating the forecast, each class held out. B: the gap over 27 analysis choices.](figures/fig4_holdout_and_grid.png)

- **Class by class.** 2023: 5/7 vs 1/8 (mid-p 0.02). 2024: 7/9 vs 1/6 (mid-p 0.02). 2025, never used to set the window: 2/7 vs 1/9 (mid-p 0.23). The clean class does not confirm the sort.
- **Analysis choices.** Over halves, thirds or quartiles; 50, 100 or 150 minimum rushes; and three forecast specs, the gap is positive in 27 of 27 cells (14 to 56 points), with 17 at p < 0.05. It is smallest against a pick-only forecast, so part of the sort comes from scoring heavy interior players fairly.
- **Likelihood vs squared error.** With rushes as exposure, adding the first step improves held-out fit in 3 of 3 classes (0.21, 0.14, 0.18 log-likelihood units per 1,000 rushes; small). By rush-weighted R² it hurts (−0.113), because one rusher, Smith, dominates 2024 (−0.401).
- **Exposure.** For the 16 rushers with 30–99 rushes the slope is 1.2 pts [−2.0, 4.3] vs 0.6 for 100+ (difference p = 0.77): too few to tell.
- **Power for 2026.** One class gives about 7 rushers per third. Even if the true gap equals ours, a one-class test has 34% power; if it is half (49% vs 25%), 8%. Confirming the half-size gap needs about 58 rushers per third, roughly 9 classes. The 2026 check is monitoring, not proof.

**Mechanism: an empirical correlate.** If the first step worked by winning at the snap, it should predict quick pressures best. It does not: ρ = 0.11 [−0.09, 0.31] for quick pressures against 0.28 [0.05, 0.49] for later ones (same baseline, n = 69). Fast feet seem to keep a rush alive rather than win the rep outright. We have no blockers in the data to show why.

## False positive: Maason Smith

Smith (2024, pick 48) rushed 98% of his snaps inside. After weight and role his NFL first step is the cohort's best; his 40 start was 67th percentile and his NGS get-off is median (1.03 s against 1.02). He produced 17 pressures on 433 rushes, 3.9% against a 7.7% forecast. Only 3 of the 17 (18%) came quickly, against 30% for the cohort.

- **Not a stunt artifact.** His starts are among the straightest (25th percentile sideways); sideways movement is unrelated to the first step (ρ = −0.04).
- **What the curve shows.** Fast for a 306-lb tackle through half a second, then flat at median DL speed by 0.7 s (Figure 1). He arrives; he does not win.
- **Film cues.** Do his hands land inside on first contact, or does he lead with a shoulder? After contact, is there a counter, or does the rep stall? Is he reading run first on early downs?
- **Cost of the wrong call.** Moving him up the rotation on the plus call would have cost about 11 pressures a season against forecast (16 over his career). One team and one scheme (Jacksonville) in both seasons, so player and call cannot be separated.

## Decision card

Plus = top-third NFL first step; minus flag = bottom third with an above-median 40 start; film first = bottom third without one. The finish check is a film step, not part of the rule. * = clean 2025 class.

| Rusher (class, pick, role) | 40 first step | NFL first step (starts) | NGS get-off | Pressure rate vs forecast | Call |
|---|---|---|---|---|---|
| Warren Brinson (2025*, 198, interior) | 56 | 93 (19) | 72 | 6.9% vs 6.2% | Plus: credit the start |
| Ashton Gillotte (2025*, 66, EDGE) | no 40 | 90 (20) | 51 | 6.4% vs 8.9% | Plus: credit the start |
| Princely Umanmielen (2025*, 77, EDGE) | 65 | 3 (16) | 86 | 5.0% vs 9.2% | Minus flag: 40 start not on tape |
| Maason Smith (2024, 48, interior) | 67 | 100 (18) | 46 | 3.9% vs 7.7% | Plus: credit the start |
| Ruke Orhorhoro (2024, 35, interior) | 35 | 99 (18) | 54 | 8.2% vs 8.2% | Plus: tape over the 40 |
| YaYa Diaby (2023, 82, EDGE) | 98 | 94 (18) | 83 | 10.5% vs 8.0% | Plus: credit the start |
| Keeanu Benton (2023, 49, interior) | 8 | 30 (19) | 26 | 6.8% vs 7.3% | Film first: grade hands and power |
| Zacch Pickens (2023, 64, interior) | 87 | 9 (20) | 28 | 6.1% vs 7.3% | Minus flag: 40 start not on tape |
| Eric Watts (2024, UDFA, EDGE) | 81 | 1 (20) | 36 | 3.6% vs 7.0% | Minus flag: 40 start not on tape |

*Table 2. Percentiles among the 69 rushers with 100+ rushes, after weight and role; NGS get-off percentile, higher = quicker.*

- **Same slot, opposite calls.** Gillotte (pick 66) and Umanmielen (pick 77) are 2025 EDGEs. Their NFL starts sit at percentiles 90 and 3, the latter despite a 65th-percentile 40 start. Both trail their forecast: a film question for year-two snaps, not a verdict.
- **Smith keeps his plus** under the rule; the finish check is what would have caught him.

## Limitations

- **Mostly nulls.** Translation is near zero after size. DL was the clearest of five groups (the 40 start after weight: WR −0.14, OL −0.31, DB −0.10, TE 0.05), and the first-step window was read off Figure 2.
- **Small, sampled tape.** About 20 random NFL starts per player; one start alone has reliability 0.09. 2025 has one season.
- **No context.** No ball, blockers or opponents: a slow first step can be a two-gap call or a slide toward him.
- **NGS already sees most of it.** Season NGS get-off ranks pressure at 0.40; the first step adds 0.12 beyond it (p = 0.33). For a club with NGS this is a tiebreak.
- **Outcome.** Pressure is charted and usage-dependent; injuries are not modelled.

## Appendix: technical checks

| Check | Result | n |
|---|---|---|
| Official 40 time → NFL first step, after weight and role | 0.23 [0.06, 0.41], p = 0.07 | 67 |
| 40 first step, two-rep repeatability | 0.54 | 56 |
| Best of 13 first-step windows (max-statistic shuffle) | p = 0.13 | 69 |
| Window re-chosen on two classes, scored on the third | 13/20 vs 3/21, p = 0.001 | 20 + 21 |
| 2025 top third if the 2023–24 rate were true | p = 0.04 (at base rate 0.54) | 7 |

*Table 3. Two-sided shuffle p unless stated.*

*Reproduce: `uv run python writeup/candidates/round_10/cand_3/analysis.py` (about 20 s).*
