# Speed That Translates: Pass Rushers Fast on the Stopwatch and on Sundays

**A lineman's tracked Combine 40 start does not carry over to his NFL pass-rush start. Agreement does: rushers fast on the official 40 *and* in about 20 sampled NFL starts beat a draft-slot pressure forecast 8 times in 10 (rest: 16 in 59) on held-out classes. Use it to hand out passing-down snaps.**

![Figure 1. Four rushers, one scale: his Combine 40 start (orange) over his own NFL starts (blue), against the league's middle half (grey).](figures/fig1_four_rushers.png)

## Why pass rushers, and why now

A pass rush starts like a 40: from a stance, straight ahead, before anyone touches him. No other position repeats a Combine movement so closely, so if Combine tracking translates anywhere, it is here. Receivers' NFL starts are readable but do not pay; linemen, backs and tight ends start too inconsistently to read from 20 snaps (Figure 2C). After a rookie season a club decides who gets third-down snaps and whether to buy a veteran, usually on a handful of sacks.

## So what for a team

1. **Act on three names (Table 3): extend Byron Young (LAR) and YaYa Diaby (TB); give Chop Robinson (MIA) a full passing-down role, re-scored at midseason because he missed his 2025 check.** All three meet the frozen act-on rule: on the list, 300+ rushes, pressure a point or more over forecast, under 25% of games missed.
2. **Pro personnel: rank young rushers by "Both"** (NFL first step in the top third, 40 time in the top half). On held-out classes the list produced 6.4 more pressures a season over forecast than everyone else.
3. **Draft room: pay for the stopwatch, not the trace.** The tracked 40 start adds nothing beyond draft slot; the official 40 time does (Table 1).
4. **Measurement: chart 24+ random rushes per young rusher** before trusting his first step (reliability 0.7; 8 give 0.44).

## The metric

**First step = a lineman's speed 0.3 to 0.5 s after he first moves, in sd after weight and role.** In that window the 40 and a pass rush are the same movement; by 1.5 s they part (6.8 yd/s in the 40, 3.1 on a rush, because he meets a blocker).

**Data.** Tracked Combine 40s, year-standardized, curves starting at first movement (so the clip's ~0.3 s lead before the laser does not matter). NFL: random sampled pass rushes only, alignment-adjusted: 1563 starts by 85 linemen, median 19 each, reliability 0.64; it tracks NGS get-off snap by snap (ρ = 0.51). **Pool:** of 118 linemen, 33 have fewer than 8 sampled starts and 16 fewer than 100 rushes; 69 remain. Outcome: career pressures per rush. **Baseline without tracking:** draft pick, weight and role (share of snaps at EDGE), fitted on the other two classes; a stronger one adds the official 40 time and season NGS get-off.

## Finding 1: the Combine start does not carry over

Raw, the 40 ranks NFL speed at 1.5 s at 0.57, but that is weight. After weight and role, the 40 first step previews the NFL first step at ρ = 0.07 [−0.14, 0.28] (n = 67). Noise is not the excuse: with two reps averaged the most a perfect link could show is 0.67, and a true translation of 0.5 or more has posterior probability 0.03. No per-player similarity score (gap, speed ratios, curve shape) exceeds 0.18 against pressure beyond pick.

| Measure, one sd faster (DL) | n | Beyond pick, weight, role | Beyond that + official 40 + NGS get-off |
|---|---|---|---|
| Tracked 40 start (Combine clip) | 52 | 0.3 [−0.2, 0.7] | 0.2 [−0.4, 0.8] |
| Carry-over (NFL start minus 40 start) | 52 | 0.2 [−0.1, 0.5] | 0.2 [−0.3, 0.7] |
| Official 40 time (stopwatch) | 54 | 0.9 [0.4, 1.5] | in baseline |
| **NFL first step (game tracking)** | 69 | **0.8 [0.1, 1.4]** | **0.4 [0.0, 0.9]** |

*Table 1. Pressure-rate points per sd faster (DL average about 8%), 90% interval, each class held out in turn. One effect size per measure.*

![Figure 2. Translation and payoff tenth by tenth (A), by held-out class (B) and by position group (C, grey = too noisy to read).](figures/fig2_translation_heldout.png)

## Finding 2: two signals that agree beat the forecast

**The rule.** *Both* = NFL first step in the top third and 40 time in the top half, each class's cut-offs set on the other two classes. A rusher "hits" if his pressure rate beats the held-out forecast (base rate 35%).

| Check | Shortlist beat forecast | Everyone else | Gap, pts [90%] | Shuffle p |
|---|---|---|---|---|
| Main: pick, weight, role forecast | 8 of 10 | 16 of 59 | 53 [28, 75] | 0.009 |
| Forecast adds official 40 time, NGS get-off | 6 of 10 | 15 of 59 | 35 [8, 62] | 0.07 |
| Sampled snaps removed from the outcome | 7 of 10 | 16 of 59 | 43 [14, 71] | 0.03 |
| Day 3 and UDFA only (pick 103+) | 2 of 3 | 9 of 33 | 39 [−26, 82] | 0.22 |
| Frozen after 2024, scored on 2025 vs own past pressure | 6 of 7 | 6 of 16 | 48 [15, 75] | 0.04 |

*Table 2. Five checks on one shortlist. Gap interval: rushers resampled within class; p: labels shuffled within class.*

**Verdict: a real but modest edge in the full pool, unproven for day-3 picks (three shortlisted).** The gap narrows in every stricter check and stays positive. Row 5 is the cleanest: called from 2023–24 starts only, scored on 2025 against a forecast that knows each rusher's own past pressure. Survivorship does not drive it: the first step does not predict who reaches 100 rushes, and reweighting for who does leaves the gap at 53 points [27, 77].

![Figure 3. A: rushers against held-out cut-offs; top right = shortlist. B: the five checks of Table 2.](figures/fig3_shortlist.png)

## Every matched pair, not one chosen case

Each Both rusher faces the non-Both rusher of his class and role nearest in draft slot (median 9 picks apart) on raw pressure rate: no model. Career: Both won 7 of 10 pairs, by 2.5 points on average [0.3, 4.6]; sign test p = 0.17. Frozen after 2024 and scored on 2025: 4 of 7, 3.4 points [0.5, 6.2], p = 0.50. Wins are larger than losses; the win count alone proves nothing.

![Figure 4. Every Both rusher against his draft-slot twin: career (A) and frozen 2024 call scored on 2025 (B).](figures/fig4_matched_pairs.png)

**One pair as an illustration.** Cincinnati's Cedric Johnson (pick 214) and Washington's Javontae Jean-Baptiste (pick 222) are a career pair: 2024 day-3 edges, both on $1.04M-a-year rookie deals. After 2024, Johnson was Both on 8 sampled starts and 10.2% pressure in 59 rushes; Jean-Baptiste had a slow 40 and 5.5% in 146. In 2025 Johnson pressured on 14.0% (22 pressures, 9 games); Jean-Baptiste played 3 games, which cannot score the call. The alternative, a mid-tier veteran edge, cost a median $5.5M a year in 2025.

## Why it works: the same rusher, faster snap

On 1279 random rushes by 69 linemen, a first step one sd faster than *his own* average multiplies that snap's odds of pressure by 1.49 [1.25, 1.77]: 14.8% for his fastest fifth of starts, 6.6% for his slowest. It holds with that snap's NGS get-off, sideways path, down and alignment fixed (1.51). The 40 says the speed exists; the NFL starts say he uses it from a stance.

![Figure 5. Same rusher, faster snap: pressure by first-step fifth; odds ratios by control, class, role.](figures/fig5_snap_mechanism.png)

**Where rule and tape disagree.** Maason Smith (Jacksonville, 2024, pick 48) has the pool's best NFL first step, a 48th-percentile 40 and 3.9% pressure against a 7.7% forecast: fast off the ball, no win. Grade hands and a counter before cutting his snaps.

## The 2026 list, frozen

The same rule, cut-offs from all three classes, applied to every lineman with 8+ sampled starts. Bold = act now.

| Rusher (class, pick, team) | Role | Availability (games missed) | 40 / first step pct | Pressure vs forecast (rushes) | 2024 call scored on 2025 |
|---|---|---|---|---|---|
| **Byron Young** (2023, 77, LAR) | hand-down end | ok (0/51) | 89 / 96 | 11.3% vs 8.3% (1501) | **hit** (12.8% vs 10.5%) |
| **YaYa Diaby** (2023, 82, TB) | hand-down end | ok (0/51) | 93 / 94 | 10.5% vs 8.0% (1420) | **hit** (11.6% vs 9.6%) |
| **Chop Robinson** (2024, 21, MIA) | hand-down end | ok (2/34) | 98 / 97 | 11.8% vs 10.0% (726) | **miss** (10.0% vs 12.6%) |
| Cedric Johnson (2024, 214, CIN) | hand-down end | flag (18/34) | 70 / 91 | 13.0% vs 7.3% (216) | **hit** (14.0% vs 8.7%) |
| Adetomiwa Adebawore (2023, 110, IND) | interior | flag (14/51) | 100 / 77 | 8.6% vs 6.8% (511) | untested (<100 rushes) |
| Byron Murphy (2024, 16, SEA) | interior | ok (3/34) | 81 / 86 | 10.0% vs 9.0% (682) | **hit** (10.4% vs 9.3%) |
| T'Vondre Sweat (2024, 38, TEN) | interior | ok (5/34) | 74 / 87 | 7.7% vs 7.2% (496) | **hit** (8.4% vs 6.9%) |
| Ruke Orhorhoro (2024, 35, ATL) | interior | flag (9/34) | 69 / 99 | 8.2% vs 8.2% (498) | untested (<100 rushes) |
| Yasir Abdullah (2023, 136, JAX) | hand-down end | flag (33/51) | 54 / 72 | 6.0% vs 7.8% (149) | untested (<100 rushes) |
| Jordan Burch (2025, 78, ARI) | hand-down end | ok (0/17) | 85 / 88 | 5.8% vs 8.5% (344) | 2025 class |
| Ty Robinson (2025, 111, PHI) | interior | flag (8/17) | 69 / 90 | 0.0% vs 7.4% (55) | 2025 class |

*Table 3. Percentiles within the pool; forecast = pick, weight, role, fitted on the other two classes (as everywhere in the text). Availability flag = 25%+ of possible games without a recorded snap (injury, inactive or cut). Last column: the rule frozen after 2024, scored on 2025 against each rusher's own past pressure.*

**Track record.** Of the list names called in 2024 and scored in 2025, 5 hit and 1 missed (Robinson, below his own strong rookie rate); the seventh 2024 call, Gervon Dexter, also hit. Jordan Burch, the 2025 class's only scored call, is 5.8% against 8.5% after one season. Cedric Johnson and Adetomiwa Adebawore are availability questions first; Ty Robinson's 55 rushes are too few to judge.

**Pre-registration.** The rule, these 11 names, the act-on three and the test are frozen in `frozen_rule.json` (sha256 `e3079960bd2d…`, 24 October 2026, no 2026 data used). Test: in 2026, list members with 100+ rushes beat a persistence forecast more often than the other 59 pool rushers (one-sided Fisher p < 0.05); the matched-pair share is secondary. At the 2025 time-forward rates (86% vs 38%) power is 90%; at half that gap, 32%. A miss at half the gap is likely and will be reported.

## Limitations

- **Not blind until 2026.** The two-signal form was chosen after seeing all three classes; only the cut-offs are held out. The frozen-2024 row has 23 rushers and overlaps the main shortlist.
- **Small lists.** Ten rushers, three on day 3; matched-pair sign tests are not significant.
- **Sampled tape, no blockers or ball.** A slow start can be a two-gap call, a fast one a free rush; injuries enter only as missed games.

## Appendix

**Estimands.** Table 1's held-out slope (NFL first step 0.8 points per sd) is the headline. Two re-fits of the same estimand: a rusher random-intercept binomial model on 137 player-seasons, 0.7 [0.2, 1.2]; inverse-probability weighting for reaching the pool (P from pick, weight, 40, class; AUC 0.85), 0.9 [0.4, 1.3]. A different estimand: beyond each rusher's own 2023–24 pressure, the frozen first step adds +0.4 [−0.5, 1.3] in 2025. **Table 1 test.** Two-class forecast as logit offset, quasi-binomial, class slopes pooled by random effects; p shuffles the measure within class: tracked 40 start 0.48, carry-over 0.48, official 40 0.004, NFL first step 0.01 (0.13 with the stronger baseline); best of 11 windows, p = 0.02. **Selection.** Among the 85 with 8+ starts, first step odds ratio for reaching the pool 0.81 (p = 0.52). **Calibration.** Shortlist expected 69% hits, got 80%; Brier 0.212 vs 0.253 for the base rate. **Measurement.** Tracked 40 first-step repeatability 0.54 (n = 56), 0.70 for two reps.

*Reproduce: `uv run python writeup/candidates/round_16/cand_3/analysis.py` (about 3 minutes).*
