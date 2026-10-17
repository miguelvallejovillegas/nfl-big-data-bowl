# Hoop G: A Pass-Rush Tiebreaker for Linemen Who Skip the 3-Cone

**For defensive linemen who skip the 3-cone, Hoop G, a number we pull from the Combine's Run-the-Hoop tracking, is a usable tiebreak. Across 90 hoop runners it is worth about 0.6 pressures a season per standard deviation, the gap between pick 100 and pick 80. Use it to break ties, not to move rounds: its hits so far come from the draft classes we used to build it.**

## The 2025 decision card

Draft rule, the same for every class: **a plus call needs the 80% interval of a player's noise-corrected Hoop G to clear zero.** A plus call breaks a tie in his favour against same-role rushers drafted near him. Where a 3-cone exists, the 3-cone comes first. Only 36% of the 2025 DL class ran it, so Hoop G covers the rest.

| 2025 player (pick, role) | Hoop G, sd [80%] | 3-cone | Draft-day call | Film cue | Rookie year (rushes) |
|---|---|---|---|---|---|
| **Ethan Downs** (UDFA, EDGE) | 0.94 [0.31, 1.57] | 7.58 s | 3-cone first; if tied, take him over day-3 / UDFA EDGEs | Flattens at the top of the rush, or only redirects late? | too few to score (7) |
| **Warren Brinson** (198, Int) | 0.77 [0.14, 1.41] | not run | Take him in a tie over interior DL picked 152–257 | Wins laterally on games and stunts? | 6.9% vs 6.6% forecast (173; within noise) |
| **Donovan Ezeiruaku** (44, EDGE) | 0.77 [0.14, 1.40] | 6.94 s | 3-cone first; if tied, take him over EDGEs picked 34–57 | Flattens at the top of the rush, or only redirects late? | 7.7% vs 9.5% forecast (402; within noise) |
| **Jordan Phillips** (143, Int) | −0.80 [−1.43, −0.17] | 8.19 s | No call: low here mostly means heavy | Can he still redirect on stunts at his weight? | 1.9% vs 6.6% forecast (215; clearly below) |

Every rookie line is a rotation-sized sample, and pressure rate does not see who drew the double team. Read the last column as "nothing contradicts the call yet", never as proof. Brinson is the clean use case: a day-3 interior lineman with no 3-cone and a top-class hoop. Downs and Ezeiruaku ran 3-cones, so Hoop G only settles ties those times leave. Phillips gets a film flag, not a downgrade. At 312 lb a low Hoop G is expected, and the baseline already prices weight.

## What Hoop G is

**Hoop G is how hard a lineman can lean into a turn at speed: the sideways force he holds for half a second while circling the hoop.**

Units: the peak half-second average of acceleration across his path, in g, standardized within Combine year (drill medians shift between years). A typical run peaks at 0.62 g at 3.5 yd/s (7 mph). That is a circle about 1.8 yards in radius run without slowing: a tight corner, not a wide loop. One run is a noisy read (reliability 0.41), so every number below shrinks single-run players toward average. Heavier interior linemen turn less sharply (ρ = −0.30 with weight, n = 53).

![Figure 1. Two EDGE plus calls, one that beat his forecast and one that has not yet, and the best and worst interior hoop. Bottom strip: all hoop runners, drill vs NFL turn.](figures/fig1_hero.png)

- **Javon Solomon** (EDGE, pick 168, no 3-cone): 0.73 g in the hoop. Pressure 12.3% against a 6.9% forecast over 211 rushes, a part-time role. He is the case the rule is built for. *Film cue:* does he flatten at the top of the rush and turn the corner, or does he win late with an inside redirect?
- **Donovan Ezeiruaku** (EDGE, pick 44): a similar hoop (0.68 g) and a plus call. His rookie rate is 7.7% vs 9.5% over 402 rushes, inside the noise. He also ran a strong 3-cone, which outranks Hoop G under the rule. *Film cue:* same check. If he redirects but does not finish, that is a plan-and-hands question, not a body question.

## How the calls did on classes the model never saw

Each class gets a pressure forecast from a no-tracking model fit on the other two classes. That model uses draft pick, weight, NFL EDGE share, NGS athleticism score and the official 10-yd split and 3-cone (missing times filled in from weight, role and NGS score). A call "beats" its forecast when the 80% interval of his actual rate sits above it.

| Group (30+ NFL rushes, 2023–25) | Rushers | Pressure rate (forecast) | Beat forecast [exact 90%] |
|---|---|---|---|
| Plus call, 2023–24 classes (tuned the rule) | 7 | 10.1% (7.7%) | 4 of 7 [23–87%] |
| Plus call, 2025 class (clean test) | 2 | 7.5% (8.6%) | 0 of 2 [0–78%] |
| No call, EDGE | 32 | 8.4% (8.2%) | 6 of 32 [8–34%] |
| No call, interior | 37 | 6.6% (7.2%) | 3 of 37 [2–20%] |

![Figure 3. Every scored plus call against its out-of-class forecast, with the hoop run that made the call. Top: classes that tuned the rule; bottom: 2025.](figures/fig3_calls.png)

**All 4 hits are EDGEs from the 2023–24 classes, the classes that helped choose the drill. In 2025, the only clean test, the plus calls went 0 of 2.** Pooled, the plus calls beat their forecast 4 of 9 times [exact 90%: 17–75%], against 19% for no-call rushers matched on role and class. When we shuffle Hoop G within class and remake the calls 5000 times, a gap that large appears with p = 0.05. That is a lead, not a validation: 9 rushers cannot validate a rule.

![Figure 2. (a) Five draft-day measures against pressure on the same rushers. (b) Gain in explained pressure rate on a held-out class.](figures/fig2_validation.png)

**Where the tiebreak applies.** We used no imputed times here. On the 56 hoop runners with an NFL rush and no 3-cone, adding Hoop G to the no-tracking model changes held-out explained pressure rate by +11.6 points [90%: −6.8, +32.2]. The held-out log score changes by +0.050 per rusher [−0.020, +0.126]. Both lean positive, and neither interval clears zero.

**Where it does not.** On the 34 who ran a 3-cone, adding Hoop G on top makes held-out forecasts worse, because a seventh term overfits so few rushers (−73.3 points; log score −0.143 [−0.341, +0.069]). On 22 rushers with every measure, the 3-cone's adjusted correlation with pressure is 0.33 and Hoop G's is −0.01. With so few rushers even the 3-cone loses ground held out (−11.3 points), so "3-cone first" rests on the in-sample comparison.

**How the headline is built.** We count pressures out of pass rushes and let rushers differ beyond chance. We shrink Hoop G's effect toward the typical drill-to-outcome link among the 14 pairs we screened, so one lucky pair cannot drive the result. That model gives the 0.6 pressures a season [90%: −0.4, 1.7] on 90 hoop runners. In plain draft terms, pick 100 vs pick 80 (80% range 47 to 106).

## Does the hoop turn show up on Sundays?

Not in our data. The game sample gives each lineman about 20 random NFL rushes, with no blockers or ball. On those rushes we measured sideways g only while he moved at hoop speed (2.5–4.5 yd/s), across 1335 rushes. A mixed model with a separate baseline for each player, adjusting for weight, role and speed, gives 0.001 g per sd of Hoop G [95%: −0.006, 0.007]. After correcting for noise in both measures, the drill-to-NFL correlation is 0.07 [90%: −0.39, 0.63], n = 78. The NFL side is the weak link: a single rush agrees with the same player's next rush at only 0.03.

So the traces in Figure 1 are film cues, not evidence of mechanism. We cannot say *how* a strong hoop turns into pressure. In the bottom strip, the upward drift is role and size, not the drill.

**Drill-protocol recommendation.** The cheapest fix is at the Combine, not in the model:

1. **Record each hoop run's peak sideways g** (the tracking already supports it) and publish it beside the 40.
2. **Two timed hoop runs per player.** Reliability rises from 0.41 to 0.58, and fewer rushers fall in the "no call" middle.
3. **Ask every DL to run the 3-cone.** Where it exists it is the better number, and Hoop G is only a stand-in for it.

## Limitations

- **Not confirmed.** The out-of-class gain across all rushers is +1.3 points [90%: −4.7, +8.7] and comes from 2023 alone. Hoop G was the best of 14 screened drill-outcome pairs, and the best of 14 looks this strong by luck with p = 0.12. The nested shuffle test (p = 0.01) uses the same three classes. A confirmatory test needs about 840 rushers.
- **Small call sample.** 9 scored plus calls, 2 of them clean. The rule and the outcome window were set with 2023–24 visible.
- **Where both exist, the 3-cone wins.** With an observed 3-cone, Hoop G's effect drops to 0.1 pressures a season.
- **Thin NFL data.** Pressure counts ignore rotation, injuries, opponents and scheme, and 2025 has one season. The in-game tracking has about 20 sampled rushes per player and no blockers.
- **Size.** For interior DL, a low Hoop G is mostly weight.

*Reproducible: `uv run python writeup/candidates/round_08/revised/analysis.py` writes `numbers.json`, `claims.json`, the board CSV and every figure, then fills this text from `writeup_template.md`.*
