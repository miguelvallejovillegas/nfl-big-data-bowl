# Hoop G: Combine Bend Finds Day-3 Pass Rushers; Combine Burst Does Not

**One pool throughout: defensive linemen drafted on day 3 (pick 103 or later) or undrafted.** Among them, the 15 who were in the top third of their Combine class on **Hoop G**, how hard they push sideways while rounding the Run-the-Hoop drill, beat a no-tracking forecast of their NFL pressure rate 8 times (**53%**). The other 32 day-3 rushers did it 8 times (**25%**). Each class was scored by a forecast that never saw it, and clubs do not draft for Hoop G, so the list is free. It is a film list, not a grade: on the strictest time-forward test the gap shrinks and its interval crosses zero. The pass-rush drill's get-off burst shows up on NFL snaps but never reaches pressure.

![Figure 1. (A) Share of day-3/undrafted rushers (30+ NFL rushes) whose pressure rate beat the out-of-class forecast, shortlist vs the rest, with 90% Wilson intervals. (B) Median draft pick of each Hoop G third, with 90% bootstrap intervals.](figures/fig1_shortlist.png)

## Why a day-3 rush list is worth building

A pressure is what a rusher is paid for: on 1655 randomly sampled NFL pass plays involving our rookies, a pressured snap cost the offense 0.52 more expected points than a clean one (90% interval 0.36 to 0.67). Day 3 is where clubs fill the rotation (70% of hoop runners went at pick 103 or later) and where the board is thinnest. The official agility test, the 3-cone, was run by only 43 of 118 linemen in the 2023–25 classes.

Every lineman, though, runs position drills under the 10 Hz Combine tracking, and nobody times them. Which drill movements carry over to an NFL rush, and can a club use one on day 3 at no cost?

## Two drill measures

- **Hoop G.** Run-the-Hoop is a figure-eight around two hoops; it rewards bending the hips at speed. We take the hardest 0.5-second window of sideways (centripetal) acceleration on the lap, in g (median 0.62 g), and standardize it within Combine year, because drill conditions shift between years. 109 linemen have it, 69 of them with no official 3-cone.
- **Rush-drill burst.** Yards covered in the first second of movement in the pass-rush drill, standardized the same way.

Both are noisy. 91% of hoop runners have one run; among the 10 with two, run-to-run reliability is 0.37. One burst rep repeats at 0.13 (101 linemen).

![Figure 2. Four rushers' hardest hoop lap, coloured by sideways acceleration on one shared scale; the red ring marks the half-second that defines Hoop G.](figures/fig2_laps.png)

## The test

**The yardstick** is a forecast of each lineman's NFL pressure rate from what a club already has: draft pick, Combine weight, NFL EDGE share, NGS athleticism score, and the official 10-yard split and 3-cone (missing times are filled from weight, role and NGS score). Each class is forecast by a model fit only on the other two. On held-out classes it ranks regulars' pressure rates at ρ = 0.37 (63 linemen with 100+ rushes).

**The rule** is one primary comparison, fixed in the previous round. Shortlist = pick 103 or later (or undrafted) AND top third of Hoop G among all hoop runners in his class. Thirds are set at the Combine; nothing in the rule is fit on NFL data. A rusher with 30+ NFL rushes "hits" if his pressure rate beats his forecast. Each player is one row, so intervals resample players, not snaps.

**Result.** 8 of 15 shortlisted rushers hit against 8 of 32 others. That is a gap of **+28 points (90% interval 4 to 50)**. With flat priors, the shortlist's true hit rate is higher with probability 97%, and at least 10 points higher with 88%. Relabelling Hoop G at random within class gives a gap this large with p = 0.037. The shortlist was ahead in 3 of 3 classes. In pressures, the gap is +5.4 per 450 rushes, a regular's season (90%: −0.5 to 11.9), about 2.8 expected points.

| Yardstick | Shortlist beat it | Rest of day 3 | Gap, points (90%) |
|---|---|---|---|
| **Primary: fit on the other two classes** | 8 of 15 | 8 of 32 | +28 (4 to 50) |
| Shrunken (ridge) forecast | 8 of 15 | 6 of 32 | +35 (10 to 55) |
| Time-forward: fit on earlier classes only (2024–25) | 5 of 11 | 5 of 20 | +20 (−8 to 47) |
| Pick-and-role-only forecast | 8 of 15 | 7 of 32 | +31 (7 to 53) |
| Without Cedric Johnson | 7 of 14 | 8 of 32 | +25 (0 to 48) |

The pick-and-role-only row shows the shortlist is not just cheap players beating a forecast anchored on draft slot. The time-forward row is the strictest: 2024 scored by a model fit on 2023 alone, 2025 by 2023–24. There the gap shrinks to +20 points and crosses zero. Shortlisted rushers were no more likely to reach 30 NFL rushes (15 of 24 vs 32 of 52), so playing time is not the filter.

Across all 78 scored hoop runners (Figure 3), pressure rate over forecast rises 0.88 percentage points per sd of Hoop G (90%: 0.43 to 1.31), positive in 3 of 3 classes. But Hoop G was the best of 14 drill-and-outcome pairs screened in earlier rounds; priced for that search, p = 0.10. Hence a shortlist, not a grade.

![Figure 3. Every scored hoop runner: NFL pressure rate minus the out-of-class forecast against Hoop G. Orange = day-3 shortlist; marker area = NFL rushes; line = rush-weighted slope.](figures/fig3_scatter.png)

## Get-off: drill burst reaches the snap, not the pressure

Get-off is the textbook rush trait, so we started there. Each link holds on its own (Figure 4). Drill burst tracks how far a lineman travels in the first second of a sampled NFL rush, after weight and role: ρ = 0.27 (90%: 0.09 to 0.42; 84 linemen with 8+ random rushes). Corrected for the noise in both measures, that is 0.62 (0.21 to 0.97). That first-second burst agrees with NGS season get-off (0.47). Season get-off then predicts pressure over forecast (0.35; 0.17 to 0.50).

Multiplied through, the chain leaves about 0.09, and the observed drill-to-pressure link is 0.06 (90%: −0.14 to 0.26): at most a weak effect. Against NGS season get-off directly, drill burst shows nothing (−0.09). The bottleneck is the drill: one rep repeats at 0.13.

Hoop G is the reverse. It moves pressure (0.22) but not first-second burst (0.07). Nor does it move depth upfield 1.5 seconds after the snap (0.04), a steadier sampled measure (reliability 0.84) that does track pressure (0.25). Our best guess is a rusher who keeps speed around the corner and through contact, which we cannot see without blockers in the data.

![Figure 4. The drill-to-game chain. Spearman ρ with 90% player-bootstrap intervals; the white diamond is the burst link corrected for measurement noise.](figures/fig4_chain.png)

## When drill and tape disagree

| Player (class, pick) | Hoop G call | Pressure vs forecast (rushes) | Sideways 2+ yd by 1.5 s | Scout action and film cue |
|---|---|---|---|---|
| **Cedric Johnson** (2024, 214) | Shortlist, +1.1 sd | 13.0% vs 7.2% (216) | 20% | The case the list is for. Cue on wide rushes: does he dip under the tackle's hands and flatten within two steps? |
| **Jaylen Harrell** (2024, 252) | Shortlist, +0.4 sd | 4.6% vs 7.0% (461) | 17% | Drill said yes, the NFL said no; tape decides. Cue: he bends at the top of the arc, but does he convert speed to power when the tackle sets short? If not, rotate him rather than build a sub package around him. |
| **Isaiah McGuire** (2023, 126) | Bottom third, −1.2 sd | 7.2% vs 5.8% (824) | 26% | The rule's miss: a power rusher who wins straight ahead. Low Hoop G is not a cut signal. |
| **Jordan Phillips** (2025, 143) | Bottom third, −1.9 sd | 1.9% vs 6.6% (215) | 40% (interior median 15%) | Miami looped him often. Cue: does he redirect on stunts or only push the pocket? Grade the role first. |

## So what for a team

1. **Build the day-3 film list from Hoop G.** Top-third linemen went at a median pick of 158, no earlier than the bottom third (130). Per sd of Hoop G, clubs drafted 3% later (90%: 11% earlier to 20% later). The list costs no draft capital.
2. **Match the role to the trait.** Shortlisted EDGEs get third-down snaps from wide alignments, where bend matters most. Shortlisted interior players get the looper job on stunts.
3. **Do not draft on drill burst.** Time get-off on tape or in camp instead; one drill rep is too noisy to carry it.
4. **Run the hoop twice.** A second run, at a pro day or in camp, lifts reliability from 0.37 to 0.54.

The 2025 shortlist, with 2026 plans set by role and rookie numbers:

| 2025 shortlist (pick, team) | Role | Hoop G (sd) | Rookie pressure vs forecast (rushes; — = under 30) | Plan for 2026 |
|---|---|---|---|---|
| **Ethan Downs** (UDFA, KC) | EDGE | +2.3 | — (7) | Third-down sub-rusher audition |
| **Jah Joyner** (UDFA, FA) | EDGE | +1.5 | — (0) | Third-down sub-rusher audition |
| **Jack Sawyer** (123, PIT) | EDGE | +1.4 | 9.7% vs 8.5% (176) | More third-down snaps from a wide (7/9) alignment |
| **Warren Brinson** (198, GB) | interior | +1.3 | 6.9% vs 6.6% (173) | Looper on interior stunts |
| **Kyle Kennard** (125, LAC) | EDGE | +1.2 | 2.0% vs 8.2% (50) | Third-down sub-rusher audition |
| **Antwaun Powell-Ryland** (209, CIN) | EDGE | +1.1 | — (0) | Third-down sub-rusher audition |
| **Tyrion Ingram-Dawkins** (139, MIN) | EDGE | +1.0 | 5.3% vs 8.0% (113) | Hold snaps; grade his wide-arc rushes |
| **Zeek Biggers** (253, MIA) | interior | +0.9 | 4.9% vs 6.4% (142) | Early downs; camp test as stunt looper |
| **Nazir Stackhouse** (UDFA, TEN) | interior | +0.4 | 1.4% vs 5.9% (71) | Stunt-package audition |
| **Tim Smith** (190, IND) | interior | +0.4 | — (0) | Stunt-package audition |

The same rule, unchanged, will score the 2026 class after the 2027 season.

## Limitations

- **One rule, chosen after looking.** Holding classes out checks the forecast, not the drill choice (search-adjusted p = 0.10). The time-forward version is weaker, and the 2026 class is the clean test.
- **Small counts.** 15 shortlisted vs 32 comparison rushers; 6 shortlisted 2025 rookies have one season. Hoop G reliability rests on 10 linemen with two runs.
- **Forecast level.** Fit on 100+ rush regulars, the forecast sits above most day-3 rushers; the test compares day-3 rushers with each other, not with 50%.
- **Thin NFL tracking.** About 20 random sampled rushes per player, no blockers, no ball. Sampled pressure barely repeats (split-half 0.04), so pressure comes from season totals and scheme reads from loop rates, not play calls.

## Appendix

| Check | Result |
|---|---|
| Shortlist vs rest by class (2023, 2024, 2025) | 3/4 vs 2/12; 3/5 vs 3/7; 2/6 vs 3/13 |
| Bayesian 90% interval for the gap (flat priors) | 3 to 49 points |
| Same rule on the official 3-cone (day 3, ran it) | 1/3 vs 3/13 |
| Top-third Hoop G on days 1–2 (not part of the rule) | 7/12 vs 6/19 |
| Drill burst → NGS season get-off, after weight and role: 90% interval | −0.27 to 0.11, n 68 |
| Reliability: burst over a lineman's reps; NFL first-second burst over 8+ sampled rushes | 0.22; 0.83 |

*Reproduce: `uv run python writeup/candidates/round_14/cand_1/analysis.py` writes `numbers.json`, the board, the 2026 rule and all figures, and fills this text.*
