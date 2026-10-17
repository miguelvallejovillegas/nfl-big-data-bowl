# Run-the-Hoop as a Pass-Rush Tiebreaker: 0.6 Pressures a Season, Unproven on New Classes

**Pass rushers with a stronger sideways push in the Combine's Run-the-Hoop drill pressured more in the NFL: about 0.6 extra pressures a season per standard deviation, the gap between pick 100 and pick 80. On classes it was not fit on, the gain is small and shows in only one of three, so treat it as unproven: a tie-break after the 3-cone, never a reason to move a player.**

| Test (90 DL hoop runners with an NFL rush) | Result |
|---|---|
| **In sample:** extra pressures per 450-rush season per sd of Hoop G [90%] | **0.6** [−0.4, 1.7] |
| **Out of class:** fit on two classes, score the third; R² change over the no-tracking baseline, weighted by rushes [90%] | +0.013 [−0.047, 0.087]; by class +0.095 / −0.031 / −0.053 |

## Why this question

1. **Pressure is what teams pay pass rushers for**, and Combine movement already tracks it for DL: the 40 and the 3-cone.
2. **The draft-room problem.** A scout comparing two day-2 or day-3 rushers with similar grades often has no 3-cone to separate them: only 34 of our 90 hoop runners ran it. Nearly every DL runs the hoop, it is fully tracked, and nobody publishes a number for it.
3. **Why EDGE and interior are pooled.** About 30 hoop runners per class reach the NFL, too few to split. The model fits separate EDGE and interior slopes, and they agree (0.6 and 0.7).

## What Hoop G is

**Coach's version: the hardest sideways push he holds for half a second while rounding the hoop.** Formally: the peak half-second average of acceleration perpendicular to his path, in g, standardized within Combine year. A typical run peaks at 0.62 g. One run has reliability 0.41 (split-half). The 10 players with two runs agree at 0.37 (90% −0.08 to 0.68), so every model adjusts for measurement error.

![Figure 1. Four rushers' hoop runs next to their hardest-turning sampled NFL rush, on one g scale: nobody reaches his hoop level on Sundays.](figures/fig1_four_rushers.png)

What the traces say, player by player:

- **Chop Robinson** (pressure 11.8% vs a 9.3% forecast): 0.72 g in the hoop, 0.57 g on his hardest rush, yet outside on only 8% of rushes. The turn shows up as a late redirect, not a wide arc.
- **Tuli Tuipulotu** (9.8% vs 7.2%): 0.67 g drill, 0.53 g Sunday peak after a straight start. He is a closer, not a bender.
- **Donovan Ezeiruaku** (7.7% vs 9.5%): a top-three hoop (0.68 g), but his typical rush peaks at 0.28 g. The drill's turn has not reached his rookie tape.
- **Jordan Phillips** (1.9% vs 6.6%): 0.54 g drill, 0.11 g typical rush. He barely redirects at all.

The drill was built to show bend, but the plus-Hoop G EDGEs run outside no more often than the cohort (9%), and the pressure link is no stronger on edge-aligned snaps (odds ratio per sd 1.05 vs 1.20 inside). **We read Hoop G as change-of-direction force, not bend.**

## Data and baseline

**Outcome:** regular-season pressures per pass rush, 2023–25.

**No-tracking baseline:** log draft pick, weight, NFL EDGE share, NGS athleticism score, games per season, and the official 10-yd split and 3-cone. Missing times are filled in from weight, role and NGS score, with a missing flag.

**Model:** a pressure-count model that allows extra player-to-player spread, with a class intercept. The prior on the Hoop G slope is skeptical by design: it is scaled to the typical effect across the 14 drill-and-outcome pairs we screened.

## Finding 1: on a common set of rushers, the 3-cone beats Hoop G

![Figure 2. Adjusted correlation with pressure for five draft-day measures on the same 22 rushers; open dots use every rusher who has the measure.](figures/fig2_signal_map.png)

On the 22 rushers who have every measure, the adjusted correlation with pressure (after pick, weight, role and NGS score) is 0.33 [−0.17, 0.69] for the 3-cone, −0.01 [−0.44, 0.58] for the 10-yd split and −0.01 [−0.60, 0.55] for Hoop G. At this n only correlations above 0.62 are reliably detectable. On every rusher who has it, Hoop G reaches 0.32 (n = 63) and the 3-cone 0.41 (n = 27). **So the order is 3-cone first; Hoop G only covers rushers who skipped it.**

What does carry to Sundays is movement, not production: the rush drill's first-second burst tracks the same player's first second on random NFL snaps (ρ = 0.32, n = 85), but not his pressure rate (−0.03).

## Finding 2: 0.6 pressures a season, not confirmed on new classes

![Figure 3. Left: the planning number and each class's own estimate. Right: R² change when each class is scored by a model fit on the other two, against the same pipeline run on shuffled Hoop G.](figures/fig3_headline.png)

**Fit on two classes, scored on the third,** with the drill screen, the chosen drill's reliability and the prior scale all re-chosen inside each fold, Hoop G raises rush-weighted R² by 0.013 on the same 90 rushers as the headline. Only 2023 gains. In 2025 the screen chose hoop speed instead.

**Negative control.** We shuffled Hoop G across players within each class and reran the whole pipeline 300 times. The real drill beats the shuffles (p = 0.01; p = 0.04 among the 111 shuffles in which shuffled Hoop G still won the screen). The count model's held-out accuracy (how likely the held-out counts were, +0.58 ± 0.49) fails this control: shuffled Hoop G scores higher (median 1.39, p = 0.89), so we no longer use it as evidence.

| Check | In sample, pressures per season [90%] | Out of class |
|---|---|---|
| Skeptical prior (planning) | 0.6 [−0.4, 1.7] | R² +0.013, shuffle p = 0.01 |
| 2× prior / wide prior | 1.5 [0.0, 3.1] / 2.0 [0.1, 4.0] | — |
| Least squares / ridge, 63 with 100+ rushes | — | R² −0.026 / +0.056 |
| Hoop G fixed in advance (optimistic) | — | R² +0.088 |
| Observed 10-yd only; pick, weight, role, NGS (n = 71) | 0.5 [−0.5, 1.7] | R² −0.020 |
| Observed 3-cone only; same baseline (n = 37) | 0.1 [−1.0, 1.2] | R² +0.101 |
| Reliability from repeat runs (0.37) | 0.7 [−0.2, 1.8] | — |

The effect is not an artifact of filled-in times: with observed 10-yd splits only it barely moves. With an observed 3-cone in the model it nearly vanishes, which matches Finding 1.

## What a team can use

**The pick-value translation.** On our draft-slot model, 0.6 pressures a season is the gap between pick 100 and pick 80 (80% range 47 to 106). That is a tiebreak between neighbours on the board, not a round.

**Rules** (a call needs the 80% interval of his adjusted Hoop G to clear zero):

| Call | EDGE | Interior |
|---|---|---|
| Plus | Break ties his way against EDGEs picked within 30% of his slot | Same, against interior DL |
| Minus | None in 2023–25 | No call: Hoop G falls with weight (ρ = −0.44), and the baseline already prices weight |

A call that rests on one run is still only a tiebreak: ask for 2–3 more runs at his pro day (three runs lift reliability from 0.41 to 0.68; a run takes about 7.5 s). Without a call, ignore Hoop G.

**Draft-room example:** Jack Sawyer (pick 123) and Barryn Sorrell (pick 124) in 2025. Sawyer's adjusted Hoop G is 0.50 sd higher, worth 0.7 pressures a season, about the value of moving from pick 124 up to 94. The rule says take Sawyer, and do not trade up for him. As rookies, Sawyer pressured at 9.7% against a forecast of 8.5%, and Sorrell at 7.6% against 8.2%: one pair called right.

**2025 calls with 30+ NFL rushes** (rank of 42; Hoop G adjusted for the number of runs):

| # of 42 | Player (pick, role) | Hoop G, sd [80%] | Draft-day rule | Rookie pressure rate vs forecast (rushes) | Result |
|---|---|---|---|---|---|
| 2 | Warren Brinson (198, Int) | 0.77 [0.14, 1.41] | break ties his way vs interior DL picked 152–257 | 6.9% vs 6.6% (173) | in noise |
| 3 | Donovan Ezeiruaku (44, EDGE) | 0.77 [0.14, 1.40] | break ties his way vs EDGEs picked 34–57 | 7.7% vs 9.5% (402) | in noise |
| 41 | Yahya Black (164, Int) | −0.75 [−1.38, −0.12] | no call: weight already priced | 5.8% vs 5.2% (189) | in noise |
| 42 | Jordan Phillips (143, Int) | −0.80 [−1.43, −0.17] | no call: weight already priced | 1.9% vs 6.6% (215) | **clearly below** |

**Why Phillips is not a minus call.** A −0.8 sd interior Hoop G mostly says "heavy". 35% of interior DL without any call also landed clearly below forecast, and 2 of 6 interior minus calls did, which is no better. His rate is a film question, not a drill one.

![Figure 4. Every 2023–25 call against its out-of-class no-tracking forecast.](figures/fig4_calls.png)

The plus calls beat their forecast clearly in 4 of 9 cases and never fell clearly below; 14% of rushers without a call beat theirs. All 4 hits come from 2023–24, the classes that helped choose the drill. In head-to-head EDGE pairs picked within 30% of each other, the higher-Hoop G rusher beat his forecast by more in 57% of 14 pairs, against 50% by chance.

**Pre-registered test:** among 2026–27 DL with 100+ rushes (about 40), the Hoop G effect after the full baseline must be above zero (one-sided α = 0.05). At the selection-adjusted effect its power is only 0.12; 80% power needs about 840 rushers, roughly 40 classes. So a miss will not disprove it, and a tiebreaker costs little when wrong. A negative estimate would retire it.

## Limitations

- **Not confirmed out of class.** The R² gain comes from one class, and its interval spans zero. The count model's accuracy gain fails the shuffle control.
- **Weaker than the 3-cone** where both exist (22 rushers), and the planning number depends on the prior.
- **Mechanism unknown.** No blockers or ball are in the tracking. Path direction comes from 20 sampled random rushes per player, without film or scheme labels.
- **Usage, health and short careers.** Pressure rate ignores rotation and injuries, and 2025 has one season.

*Reproducible: `uv run python writeup/candidates/round_07/revised/analysis.py` writes `numbers.json`, `claims.json` and every figure, then fills this text from `writeup_template.md`.*
