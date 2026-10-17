# The First Step Travels, the Match Doesn't Pay: Drill-to-Game Launch for NFL Pass Rushers

A defensive lineman's first half-second looks the same at the Combine and on Sundays, and his 40 previews how hard he launches in the NFL. But that preview is mostly body type. How closely a rusher "plays to his Combine" adds nothing to draft slot, weight and role in predicting pressure. Grade the first step with the stopwatch, and scout what happens at contact.

**Why DL, and why now.** Defensive line is the one group where Combine numbers already predict NFL pressure (3-cone, 40 and shuttle, ρ around 0.5–0.6). The first step is also the one movement that starts the same way in both places: from a stance, on a cue. Tracked position drills began with the 2023 Combine, and three classes now have NFL rushes. 

![Figure 1. Median speed from first movement for the same 85 DL: Combine pass-rush drill vs 20 random NFL rushes each (bands = middle 50% of players). The curves overlap for half a second, then NFL rushers meet blockers.](figures/fig1_phase.png)

## Launch in coaching terms

**Launch is how hard a player pushes off over his first three steps.** Formally, it is the peak forward acceleration (yd/s²) in the first second after he starts moving. One function computes it on 10 Hz tracking from the Combine pass-rush drill, the Combine 40, and NFL snaps (window starts at the snap). **The gap** is a player's NFL launch minus his Combine-drill launch, both in standard-deviation units after shrinking noisy players toward the mean. A positive gap means he launches harder on Sundays than he did in Indianapolis. **Curve-shape match** correlates the shapes of his drill and NFL speed curves.

**Data.** 118 DL from the 2023–2025 classes: 219 pass-rush-drill reps, 166 40 reps and 1,622 *random* NFL rushes (20 per player; the sack-heavy top-EPA half is only a replication check). NFL launch is adjusted for alignment; Combine launch is standardized within drill and year.

**Is the game metric real?** On single rushes, launch tracks NGS get-off time at ρ = −0.59 (n = 1,622; faster get-off, harder launch). One rush is noisy (reliability 0.10); twenty give 0.67, six only 0.39.

## Finding 1: the first step travels, mostly as body type

**The first half-second is the same.** At 0.5 s the median NFL rusher is moving at 98% of his drill speed. By 1.0 s he is at 82%, and by 2.0 s at 54% (Fig 1). Peak launch is nearly matched: 4.2 yd/s² in games against 4.4 in the drill, a median retention of 95%. The drill keeps accelerating; the NFL rush flattens at contact.

![Figure 2. Six DL picked by rule: the three largest and three smallest gaps among the 69 with 100+ NFL rushes. Dashed = Combine pass-rush drill reps; thin blue = single NFL rushes; thick = their mean. Both rows hold good and poor pass rushers.](figures/fig2_cards.png)

**Who launches hardest partly carries over.** We screened five position groups, four movement features and two Combine sources (Fig 3). DL launch is the only movement positive from both sources: the 40 gives ρ = 0.36 [0.14, 0.54] (n = 69) and the pass-rush drill 0.22 [0.02, 0.41] (n = 85). It is the top cell, but not proof: shuffling NFL players produces a cell this strong somewhere in the screen with probability 0.276.

![Figure 3. Combine vs NFL player means on random snaps, 40 cells. Six reach nominal p < 0.05 where two are expected by chance.](figures/fig3_screen.png)

**But it is not personal.** If translation were personal, a rusher's Combine launch would match his own NFL launch better than that of a lookalike: another player of the same role and nearest weight. It does not (Fig 4B). The 40 previews his own NFL launch at 0.36 and the lookalike's at 0.35; the drill gives 0.22 and 0.20. His Combine launch lands closer to his own NFL launch than to the lookalike's for 48% of players [36, 59] with the 40 and 56% [46, 67] with the drill. A coin flip gives 50%.

**And the stopwatch already has it.** The official 40 time ranks NFL launch at ρ = 0.39 [0.16, 0.60], and the 10-yard split at 0.34. Beyond official 40 and weight, tracked drill launch adds 0.09 [−0.11, 0.30]. One drill rep keeps only 0.18 of its deviation after shrinkage; a player would need 10 reps for a dependable reading.

![Figure 4. A: shrunk 40 launch vs shrunk NFL launch, bars = posterior sd. B: Combine launch vs the player's own NFL launch (orange) and vs a same-role, nearest-weight lookalike's (grey), with the official 40 time for reference.](figures/fig4_translation.png)

## Finding 2: playing to your Combine does not predict pressure

The outcome is career pressure rate for the 69 DL with 100+ NFL rushes (median 297), shrunk toward the role mean so short careers do not swing it. The no-tracking baseline is log draft pick, Combine weight and EDGE share. We tested five per-player measures (Fig 5): the gap, curve-shape match, NFL launch level, drill launch and 40 launch.

- **The gap**: partial ρ 0.12 [−0.16, 0.39] beyond the baseline. Given his NFL launch level, it carries nothing (−0.02).
- **NFL launch level** is the strongest: 0.42 raw, but 0.17 [−0.11, 0.43] beyond the baseline. Lighter rushers launch harder (ρ −0.32 with weight).
- **Out of class, nothing adds.** Each class is predicted from the other two. All five measures leave the baseline's held-out R² flat or lower (Appendix).
- **Re-running the whole pipeline** with each class held out (screen, choice of measure, fit) picks DL launch in only 1 of 3 folds and gains held-out accuracy in 0 of 3.
- **Placebo.** If the gap were a "good player" signal, it would also predict run-stop rate. It does not; it leans negative (−0.19).
- **EDGE vs interior.** The gap is 0.20 for EDGE (n = 35) and 0.11 for interior (n = 33). The difference is well inside noise.

![Figure 5. Left: partial ρ with pressure beyond the baseline, 95% bootstrap CI; bottom row is a placebo outcome. Right: held-out R² gain when each class is predicted from the other two; bottom row re-runs the screen and the choice of measure inside every fold.](figures/fig5_pay.png)

**What it is worth.** At its point estimate, one standard deviation of gap is 0.4 pressures a season (interval −0.7 to 1.6) for a rusher with the cohort's median 215 rushes. One standard deviation of NFL launch is 0.6 (−0.7 to 2.0), the same as moving from pick 100 to pick 67. Most of a player's gap is noise: gaps spread 0.87 sd across players, but each player's own uncertainty is 1.01 sd. Only 5 of 85 have a 90% interval that excludes zero.

| Player (pick, role, class) | Combine drill launch | NFL launch | Gap | Pressure % (rushes) | Read |
|---|---|---|---|---|---|
| Jordan Burch (78, EDGE, 2025) | −0.9 ± 0.8 | +1.3 ± 0.6 | +2.1 | 6.8 (344) | Launches, does not finish: hands, counters |
| Yasir Abdullah (136, EDGE, 2023) | −1.7 ± 0.8 | +0.3 ± 0.6 | +1.9 | 7.5 (149) | Wins without launch: power, length |
| Maason Smith (48, interior, 2024) | +0.3 ± 0.8 | +2.2 ± 0.6 | +1.9 | 4.6 (433) | Launches, does not finish: hands, counters |
| Byron Young (77, EDGE, 2023) | +0.0 ± 0.9 | +1.8 ± 0.6 | +1.8 | 11.1 (1,501) | Launch and finish: no flag |
| Ruke Orhorhoro (35, interior, 2024) | +0.1 ± 0.8 | +1.6 ± 0.6 | +1.4 | 7.9 (498) | Launch and finish: no flag |
| Dante Stills (213, interior, 2023) | +0.4 ± 0.8 | −0.7 ± 0.6 | −1.0 | 5.7 (914) | Below on both: rotational |
| Austin Booker (144, EDGE, 2024) | +0.8 ± 0.8 | −0.6 ± 0.6 | −1.4 | 8.2 (535) | Wins without launch: power, length |
| Brenton Cox (UDFA, EDGE, 2023) | +0.7 ± 0.8 | −0.9 ± 0.6 | −1.6 | 9.3 (201) | Wins without launch: power, length |
| Eric Watts (UDFA, EDGE, 2024) | −0.3 ± 0.8 | −2.0 ± 0.6 | −1.6 | 5.7 (225) | Below on both: rotational |
| P.J. Mustipher (UDFA, interior, 2023) | +0.3 ± 0.9 | −1.6 ± 0.6 | −1.9 | 5.4 (129) | Below on both: rotational |

*Table 1. Five largest and five smallest gaps in the pressure cohort (launch in sd, ± posterior sd; pressure shrunk, cohort median 7.1%). "Read" rule: high launch = NFL launch ≥ +0.5 sd; high pressure = at or above median.*

Byron Young was ordinary in Indianapolis, launches among the hardest on Sundays, and produces. Maason Smith launches harder still and pressures at 4.6%. Brenton Cox plays below his drill and still pressures above the median. The top five gaps average 7.6% pressure, the bottom five 6.9%.

## What a team can use

1. **Draft room.** For first-step quickness, use the official 40 and 10-yard split with weight. Give tracked pass-rush-drill launch zero weight: it adds nothing beyond them, and one rep is mostly noise.
2. **Pro personnel and coaching.** Do not flag a young rusher for "not playing to his Combine." The gap is worth 0.4 pressures a season at its point estimate and is mostly measurement noise. Grade launch from 20+ random rushes as a description of get-off, not a forecast.
3. **Scouting time.** The two settings agree for half a second and split at contact. Pressure is won after that point (hands, counters, power), and no Combine drill measures it. Spend film time there.

## Pre-registered 2026 test

For 2023–2025 DL with 100+ 2026 rushes, NFL launch from 20 random 2026 rushes will show a positive partial ρ with 2026 pressure beyond the same baseline. We drop the lead if the estimate is ≤ 0. Power is only 0.29 at the observed effect, and the smallest effect detectable at this n is 0.34. A null would therefore be weak evidence, and we say so in advance.

## Limitations

- **Sample.** Twenty random rushes per player (reliability 0.67). On the top-EPA half the 40 link holds (0.28) but the drill link does not (0.08).
- **Selection.** 17 of 86 DL with NFL launch did not reach 100 rushes. Drill launch does not predict who reaches it (−0.04); 40 launch leans positive (0.23, p = 0.062). Injuries are not modelled (cohort median 14.0 games a season).
- **No ball, no opponents.** Launch starts at first movement, so snap reaction is excluded. Contact time, stunts and double teams are unobserved.
- **Construct.** The drills have no blocker and the 40 has no stance change. The lookalike test matches only role and weight; arm length and technique are not matched.
- **Outcomes.** Pressure is charted and carries scheme. The 2025 class has one season.

## Appendix: methods and statistics

Onset is the first frame at or above 1.0 yd/s, backed up while speed stays at or above 0.5 yd/s. Launch is the peak first difference of Savitzky–Golay-smoothed speed over the next 1.0 s. Player means are shrunk by λ = τ²/(τ² + σ²/k) (one-way variance components) and divided by τ. Pressure is beta-binomial shrunk within role (prior 149 EDGE and 152 interior rushes). Partial correlations are rank-based with 2,000 bootstraps. Best-of-k p values permute NFL means across players (screen) or pressure within class (production). The nested run repeats the screen and the choice of measure on two classes and scores the third (interval: 300 training redraws). Baseline R² varies by row because each uses its complete cases.

| Test | Estimate [95% CI] | n | best-of-k p | Held-out R² gain (baseline R²) |
|---|---|---|---|---|
| 40 → NFL launch | 0.36 [0.14, 0.54] | 69 | 0.276 (k = 40) | — |
| Drill → NFL launch | 0.22 [0.02, 0.41] | 85 | 0.887 (k = 40) | — |
| Own minus lookalike, 40 | 0.01 [−0.28, 0.31] | 69 | — | — |
| Gap → pressure | 0.12 [−0.16, 0.39] | 68 | 0.740 (k = 4) | −0.006 (0.36) |
| Curve match → pressure | 0.05 [−0.23, 0.29] | 68 | 0.986 (k = 4) | −0.001 (0.36) |
| NFL launch → pressure | 0.17 [−0.11, 0.43] | 69 | 0.501 (k = 4) | −0.006 (0.38) |
| Drill launch → pressure | 0.10 [−0.13, 0.32] | 68 | 0.814 (k = 4) | −0.012 (0.36) |
| 40 launch → pressure | 0.05 [−0.25, 0.34] | 53 | — | −0.031 (0.37) |
| Whole pipeline, nested | — | 69 | — | −0.015 [−0.330, 0.002] |
| Gap → run-stop (placebo) | −0.19 [−0.44, 0.05] | 65 | — | — |
| Gap × EDGE share (pp) | 0.58 [−0.49, 1.61] | 68 | — | — |

*Table 2. All primary tests. Partial ρ is beyond log pick, weight and EDGE share.*

*Reproducible: `uv run python writeup/candidates/round_04/cand_3/analysis.py` writes every number to `numbers.json`, every figure, and renders this file from `writeup_template.md`.*
