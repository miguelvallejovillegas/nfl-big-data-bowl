# Hoop G: Run-the-Hoop Lateral Force as a Pass-Rush Tiebreaker

**Hoop G, the hardest sideways force a defensive lineman holds for half a second in the Combine's Run-the-Hoop drill, is worth about 0.6 pressures a season per standard deviation, the gap between pick 100 and pick 80. It has not yet held up on new classes, so use it only to break ties between rushers who never ran the 3-cone.**

| 90 DL hoop runners with an NFL pass rush, 2023–25 | Result |
|---|---|
| Extra pressures per 450-rush season, per sd of Hoop G [90%] | **0.6** [−0.4, 1.7] |
| On a class the model never saw: pressure-rate variance explained beyond the no-tracking baseline [90%] | +1.3 points [−4.7, +8.7]; it helped in 2023 only |

## Why now, and why this drill

- **Why now.** This Big Data Bowl is the first public release of 10 Hz tracking from Combine position drills. The Run-the-Hoop, where a rusher circles a hoop on the ground leaning in, has always been graded by eye; the official results carry no number for it.
- **The draft-room problem.** Pressure is what teams pay pass rushers for, and the 3-cone is the best Combine predictor of it we found. But only 34 of our 90 hoop runners ran a 3-cone (36% of the 2025 class). Nearly every DL runs the hoop.

## What Hoop G is

**One metric, one name.** Hoop G is the peak half-second average of acceleration perpendicular to his path in the hoop, in g, standardized within Combine year. In plain terms it is how hard he can lean into a turn without slowing down.

**What 0.62 g means.** A typical run peaks at 0.62 g while moving 3.5 yd/s (7 mph). Holding that force at that speed means running a circle about 1.8 yards in radius without slowing: a tight corner, not a wide loop. On sampled NFL rushes the typical half-second peak is only 0.17 g: blockers, not the hoop, set the path on Sundays.

**Noise.** One run has reliability 0.41 (split-half); the 10 players with two runs agree at 0.37. Every model below corrects for this. Heavier players turn less sharply: within interior DL Hoop G falls with weight (ρ = −0.30, n = 53), much less so on the edge (−0.13).

![Figure 1. The top and bottom Hoop G at each position: hoop path, the sideways-g trace of the hoop run next to his hardest-turning sampled NFL rush, and NFL pressure rate against the no-tracking forecast.](figures/fig1_hero.png)

What the four rows say:

- **Javon Solomon** (EDGE, pick 168): 0.73 g in the hoop, 0.45 g on his hardest NFL turn and the highest typical rush of the four (0.21 g). Pressure 12.3% against a 6.9% forecast: a day-3 EDGE with no 3-cone, the case the rule is built for.
- **Ashton Gillotte** (EDGE): 0.52 g drill, 0.14 g typical rush; 6.4% vs 9.3%. Clearly below forecast, but his Hoop G interval crosses zero, so no call.
- **Adetomiwa Adebawore** (interior, 282 lb): the best interior hoop (0.69 g), yet his typical rush peaks at 0.13 g and he lands on forecast (8.6% vs 8.4%). Inside, the turn rarely gets used.
- **Jordan Phillips** (interior, 312 lb): 0.54 g drill, 1.9% vs 6.6%. A film flag, not a call: at his weight a low Hoop G is expected.

**Does it show up in his own NFL movement?** Barely. Across 79 rushers, Hoop G tracks the average peak sideways g of his own random NFL rushes at ρ = 0.10 (95% −0.11 to 0.32, after weight and role, players resampled). The pass-rush drill's first-second burst does better on get-off: it tracks his first second on random NFL snaps (ρ = 0.32, n = 85) but not his pressure rate. And plus-Hoop G EDGEs run outside the tackle no more often than the cohort, 9% of classified rushes, so **we read Hoop G as change-of-direction force, not bend.**

## Data and baseline

**Outcome:** regular-season pressures per pass rush, 2023–25 (player season totals).

**No-tracking baseline:** log draft pick, weight, NFL EDGE share, NGS athleticism score, games per season, and the official 10-yd split and 3-cone. Missing times are filled in from weight, role and NGS score, with a missing flag.

**Model:** pressures out of rushes, allowing rushers to differ beyond chance, with a class effect and separate EDGE and interior slopes (0.6 and 0.7 pressures a season). It starts from a cautious assumption: Hoop G is expected to be no stronger than the typical one of the 14 drill-and-outcome pairs we screened (7 rush and hoop measures against pressure and get-off).

## Does it hold up?

![Figure 2. (a) Adjusted correlation with pressure for five draft-day measures on the same 22 rushers. (b) Gain when each class is scored by a model fit on the other two.](figures/fig2_validation.png)

**Against the stopwatch.** On the 22 rushers who have every measure, the 3-cone's adjusted correlation with pressure is 0.33 [−0.17, 0.69] and Hoop G's is −0.01. **The 3-cone comes first; Hoop G only covers rushers who skipped it.**

**On unseen classes.** Fitting on two classes and scoring the third, with every choice (which drill, its noise, how cautious to be) remade inside each split, Hoop G adds 1.3 points of variance explained: +9.5 in 2023, −3.1 in 2024, −5.3 in 2025. Rerunning the whole pipeline on 300 copies with Hoop G shuffled within class gives a gain this large with p = 0.01.

**Chance of a lucky pick.** Hoop G was the strongest of 14 screened pairs (ρ = 0.32). Shuffling the NFL outcomes 2000 times, the best of 14 reaches that strength with p = 0.12. Taken alone, the in-sample link is not beyond luck.

| Check | Pressures a season per sd [90%] |
|---|---|
| Planning number (cautious) | 0.6 [−0.4, 1.7] |
| Only rushers with an official 3-cone (n = 37) | 0.1 [−1.0, 1.2] |
| Less cautious starting assumption | 2.0 [0.1, 4.0] |

## What a team can use

**The price.** 0.6 pressures a season is the gap between pick 100 and pick 80 on our draft-slot model (80% range 47 to 106). That is a tiebreak between neighbours on the board, not a round.

**Rules.** A call needs the 80% interval of his noise-corrected Hoop G to clear zero.

| | EDGE | Interior |
|---|---|---|
| Plus call | Break ties his way against EDGEs picked within 30% of his slot, if neither has a 3-cone | Same, against interior DL |
| Film check | Does he flatten at the top of the rush or only redirect late? | Does he win laterally on games and stunts? |
| Low Hoop G | No EDGE fell that low in 2023–25 | No call: mostly says "heavy", which the baseline already prices; film flag only |

**For the position coach:** a plus says the turn is in his body. The plus EDGEs did not run wider arcs, so the job is building the late redirect into his rush plan.

**How the calls did.** There are 10 plus calls in 2023–25; 9 have the 30+ NFL rushes needed to score them. Each class is forecast by a no-tracking model fit on the other two.

| Group (30+ NFL rushes) | Rushers | Pressure rate (forecast) | Clearly beat forecast [90%] | Clearly below |
|---|---|---|---|---|
| Plus call, EDGE | 7 | 10.0% (7.8%) | 4 of 7 [23–87%] | 0 |
| Plus call, interior | 2 | 8.2% (8.0%) | 0 of 2 [0–78%] | 0 |
| No call, EDGE | 32 | 8.4% (8.2%) | 6 of 32 [8–34%] | 12 |
| No call, interior | 31 | 6.9% (7.2%) | 3 of 31 [3–23%] | 11 |
| Low Hoop G, interior (no call) | 6 | 4.7% (6.9%) | 0 of 6 [0–39%] | 2 |

The plus calls pressured at 9.7% against 7.8% for rushers without a call, 1.9 points above their forecast. In 2023–24, the classes that helped choose the drill, 4 of 7 plus calls clearly beat their forecast [90% 23–87%]; in 2025, the clean class, 0 of 2. The no-call base rate is 14% [8–24%]. By class the plus calls went 1 of 3 (2023), 3 of 4 (2024) and 0 of 2 (2025); none fell clearly below. Low-Hoop G interior DL fell clearly below in 2 of 6 cases, close to the 35% of no-call interior DL who did.

![Figure 3. Every 2023–25 call against its out-of-class no-tracking forecast, with the hoop run that made the call.](figures/fig3_calls.png)

**2025 decision card** (Hoop G corrected for the number of runs; rookie results shown for reference, not used):

| Player (pick, role) | Hoop G, sd [80%] | 3-cone, s | Draft-day call | Rookie pressure rate vs forecast |
|---|---|---|---|---|
| **Ethan Downs** (UDFA, EDGE) | 0.94 [0.31, 1.57] | 7.58 | 3-cone first; then break ties his way vs day-3 / UDFA EDGEs | 7 rushes: too few to score |
| **Warren Brinson** (198, Int) | 0.77 [0.14, 1.41] | none | break ties his way vs interior DL picked 152–257 | 6.9% vs 6.6% (173 rushes, in noise) |
| **Donovan Ezeiruaku** (44, EDGE) | 0.77 [0.14, 1.40] | 6.94 | 3-cone first; then break ties his way vs EDGEs picked 34–57 | 7.7% vs 9.5% (402 rushes, in noise) |
| **Jah Joyner** (UDFA, EDGE) | 0.60 [−0.03, 1.23] | 7.33 | close, but no call: use his 3-cone | no NFL rushes yet |
| **Tyrion Ingram-Dawkins** (139, EDGE) | 0.58 [−0.06, 1.21] | 7.28 | close, but no call: use his 3-cone | 5.3% vs 8.0% (113 rushes, in noise) |
| **Tyler Baron** (176, EDGE) | −0.62 [−1.26, 0.01] | none | no call | 4.4% vs 8.0% (68 rushes, in noise) |
| **Jordan Phillips** (143, Int) | −0.80 [−1.43, −0.17] | 8.19 | no call: low Hoop G here mostly means heavy; film flag only | 1.9% vs 6.6% (215 rushes, clearly below) |

## Limitations

- **Not confirmed on new classes.** The out-of-class gain comes from 2023 alone and its interval spans zero; allowing for the 14 screened pairs, the in-sample link could be luck. A confirmatory test needs about 840 rushers, roughly 40 classes.
- **Weaker than the 3-cone** where both exist, and the planning number depends on the cautious starting assumption.
- **Thin NFL tracking.** In-game checks use about 20 sampled random rushes per player, with no ball, blockers, stunts or scheme labels. Pressure counts ignore rotation, injuries and opponents; 2025 has one season.
- **Size.** For interior DL, low Hoop G is largely weight.

*Reproducible: `uv run python writeup/candidates/round_08/cand_1/analysis.py` writes `numbers.json`, `claims.json` (each number's sentence), the board CSV and every figure, then fills this text from `writeup_template.md`.*
