# A Combine Hoop Drill Hints at Extra Pass Rush: 0.6 Pressures a Season, Not Yet Confirmed Out of Class

**Can the Combine's Run-the-Hoop drill, which has no published time, help sort pass rushers who look alike on draft slot and stopwatch times? One standard deviation of its tracked Hoop G is worth 0.6 extra pressures per 450-rush season (90% interval −0.4 to 1.7), but its gain on unseen classes is too small to separate from zero: use it only to break ties.**

| Test | Players | Planning estimate | Sensitivity |
|---|---|---|---|
| **In sample:** extra pressures per 450-rush season per sd [90%] | 90 hoop runners with an NFL rush | **0.6** [−0.4, 1.7], skeptical prior | 2× prior 1.5 [0.0, 3.1]; wide prior 2.0 [0.1, 4.0] |
| **Out of class, log score** (± SE): gain over the same model without the drill; screen, prior and reliability re-chosen in each fold | 90 (28 / 27 / 35) | +0.58 ± 0.49; 2023 / 2024 / 2025: +0.32 / +0.14 / +0.11 | 2× prior +0.88 ± 0.91; wide prior −0.18 ± 1.39 |
| **Out of class, R² change** [95%], fully nested | 63 with 100+ rushes (21 / 22 / 20) | −0.026 [−0.286, 0.185]; by class −0.042 / −0.014 / −0.151 | ridge 0.056 [−0.200, 0.155]; Hoop G fixed in advance 0.088 |

The count model weights every rusher by his rushes, so it uses all 90. R² needs a stable observed rate, so it keeps the 63 with 100+ rushes.

## Why this question

1. **Pressure is what teams pay pass rushers for**, and it is where Combine movement already shows signal for DL: the 40 and 3-cone track it.
2. **The drills that carry that signal are often skipped.** Of our 90 hoop runners with an NFL rush, only 34 ran the 3-cone and 71 the 10-yd split. Nearly every DL runs the hoop, and it is fully tracked, yet nobody publishes a number for it.
3. **Three tracked classes (2023–25) now allow an honest test:** fit on two classes, score the third. No tracking baseline exists for this drill yet.

## What Hoop G is

**Coach's version: the hardest sideways push he holds for half a second while rounding the hoop.** Formally, it is the peak half-second average of acceleration perpendicular to his path, in g, standardized within Combine year. A typical run peaks at 0.62 g. One run has reliability 0.41, so every model corrects for measurement error.

![Figure 1. Takeaway: the three plus-Hoop G EDGEs hold 0.67–0.72 g around the hoop and Phillips holds 0.54 g. On their hardest-turning sampled NFL rush, none reaches the hoop's level (top: 0.57 g). Across 79 DL, Hoop G barely predicts NFL rush-path sideways g (ρ = 0.10 after weight and role).](figures/fig1_four_rushers.png)

The drill was built to show bend, so we tested whether Hoop G *is* bend on Sundays. Both tests fail. First, plus-Hoop G EDGEs do not take wider paths: Robinson, Ezeiruaku and Tuipulotu run outside on 8%, 15% and 12% of classified rushes, against 9% for the cohort. Second, a corner skill should pay off most on EDGE-aligned snaps. Across 1541 random sampled rushes by 79 hoop runners, the odds of a pressure per sd are 1.05 [0.85, 1.32] when aligned on the edge and 1.20 [0.96, 1.45] inside (90%). **So we read Hoop G as a general change-of-direction force marker, not a bend grade,** and nothing below relies on a corner mechanism.

## Data and baseline

**Outcome:** regular-season pressures per pass rush, 2023–25.

**No-tracking baseline:** log draft pick, weight, NFL EDGE share, NGS athleticism score, games per season, and the official 10-yd split and 3-cone. Missing times are filled from weight, role and NGS score, with a missing flag.

**Model:** a pressure-count regression that allows extra player-to-player spread (beta-binomial), with a class intercept and separate EDGE and interior slopes. The prior on the Hoop G slope is skeptical by design: it is scaled to the typical effect of 14 drill-outcome pairs we screened.

## Finding 1: Hoop G is the tracked drill that survives the baseline

![Figure 2. Takeaway: rush-drill burst carries to the same player's NFL get-off but not to pressure. After the no-tracking baseline, only Hoop G keeps a pressure link.](figures/fig2_signal_map.png)

Get-off burst (yards covered in the rush drill's first second) tracks the player's first second on random NFL snaps (ρ = 0.32 [0.12, 0.50], n = 85). It is unreliable across reps (0.22), though, and unrelated to pressure (−0.03).

**Against what a team already has**, after the same baseline (pick, weight, role, NGS score), the partial ρ with pressure is 0.41 [−0.01, 0.72] for the 3-cone (n = 27), 0.11 [−0.14, 0.36] for the 10-yd split (n = 54) and 0.32 [0.07, 0.53] for Hoop G (n = 63). Trust a 3-cone first when there is one. Hoop G covers the rushers who skipped it. Adding the observed 10-yd split to the baseline changes nested R² by −0.002 (n = 49), so the stopwatch does not beat the drill out of class either.

## Finding 2: 0.6 pressures a season, not confirmed out of class

![Figure 3. Takeaway: the planning number is the smallest of three prior scales. Each class, estimated with or without itself, points the same way, and no single class is clear of zero. The fully nested held-out log score is positive in all three folds, even the 2025 fold, where the screen picked hoop speed instead of Hoop G.](figures/fig3_headline.png)

On our draft-slot model, 0.6 pressures is the gap between pick 100 and pick 80 (80% range 47 to 106).

**Out of class, everything is re-chosen inside each fold:** the drill screen, the chosen drill's reliability and the prior scale, all on the two training classes. The log score asks how likely the held-out pressure counts are under each model. With the skeptical prior, the drill model makes them 1.8× as likely as the baseline (+0.58 ± 0.49 on the log scale). The screen picked Hoop G in 2 of 3 folds. With a wide prior the gain vanishes (−0.18 ± 1.39). The R² tests agree that any gain is small: least squares −0.026, ridge 0.056, both intervals across zero. Fixing Hoop G in advance gives 0.088, positive in 3 of 3 classes. That number is optimistic, because we chose the hoop after seeing all three classes. It is exactly the test the 2026–27 classes will run.

**Checks.** The model's 80% predictive intervals cover 88% of rushers, and the simulated spread of pressure rates matches the observed spread (p = 0.75). With observed 10-yd times only, the estimate is 0.5 [−0.4, 1.6] (n = 71). Over 20 imputations it is 0.6 [−0.3, 1.7], and dropping any one rusher moves it between 0.5 and 0.8. Placebos stay near zero: run-stop rate 0.02 (n = 60) and the back-pedal drill 0.09 (n = 38).

## What a team can use

**Rules, by role** (a call needs the 80% interval of his shrunk Hoop G to clear zero):

| Situation | EDGE | Interior |
|---|---|---|
| Plus call | Break ties his way against EDGEs picked within 30% of his slot | Same rule against interior DL; the interior slope (0.7) matches the EDGE slope (0.6) |
| Minus call | None in 2023–25; check film for a power plan, do not drop him | Ignore. Hoop G falls with weight (ρ = −0.44), every interior minus weighs 285+ lb, and the baseline already prices weight. 2 of 6 still landed clearly below forecast |
| Already in the league (trade, extension) and his pressure rate clearly contradicts the call | Film first: the drill never overrules NFL tape | Same |
| Call rests on one run (most prospects) | Still a tiebreak only. Ask for 2–3 more runs at his pro day: three runs lift reliability from 0.41 to 0.68 | Same |
| No call | Ignore Hoop G | Ignore Hoop G |

**Draft-room example:** Jack Sawyer (pick 123) and Barryn Sorrell (pick 124) in 2025. Sawyer's Hoop G is 0.50 sd higher (shrunk). On the planning number that is worth 0.7 pressures a season, about the value of moving from pick 124 up to 94. The rule says take Sawyer and do not trade up for him. As rookies, Sawyer pressured at 9.7% against a forecast of 8.5%, and Sorrell at 7.6% against 8.2%. That is one pair called right, not proof.

**2025 class, every call** (rank of 42; Hoop G shrunk for the number of runs):

| # of 42 | Player | Pick | Role | Hoop G, sd [80%] | NFL pressure rate vs forecast (rushes) | Rule (window) |
|---|---|---|---|---|---|---|
| 1 | Ethan Downs | UDFA | EDGE | 0.94 [0.31, 1.57] | — (7) | break ties his way vs day-3 / UDFA EDGEs |
| 2 | Warren Brinson | 198 | Int | 0.77 [0.14, 1.41] | 6.9% vs 6.6% (173) | break ties his way vs interior DL picked 152–257 |
| 3 | Donovan Ezeiruaku | 44 | EDGE | 0.77 [0.14, 1.40] | 7.7% vs 9.5% (402) | break ties his way vs EDGEs picked 34–57 |
| 8 | Jordan Burch | 78 | EDGE | 0.49 [−0.14, 1.12] | 5.8% vs 8.5% (344) | film first: NFL rate clearly below forecast |
| 41 | Yahya Black | 164 | Int | −0.75 [−1.38, −0.12] | 5.8% vs 5.2% (189) | ignore: weight, already priced |
| 42 | Jordan Phillips | 143 | Int | −0.80 [−1.43, −0.17] | 1.9% vs 6.6% (215) | ignore: weight, already priced |

![Figure 4. Takeaway: of the 7 plus calls from 2023–24, Chop Robinson, Tuli Tuipulotu, Javon Solomon and Cedric Johnson beat their no-tracking forecast clearly; Adetomiwa Adebawore, Austin Booker and Nick Hampton sat inside noise; none fell clearly below. Across all three classes, 6 of 18 calls are clear hits and 3 are clear misses (the film-first flags).](figures/fig4_calls.png)

Head to head, across all 14 same-class EDGE pairs picked within 30% of each other with a 0.4+ sd gap, the higher-Hoop G rusher finished further above his forecast in 8 (57%), no better than shuffling (50%).

**Ask for 3–4 hoop runs.** A run lasts about 7.5 s inside a workout that is already tracked, so it costs only a few minutes.

**Pre-registered test:** among 2026–27 DL with 100+ rushes by the end of 2027 (about 40), the Hoop G effect after the full baseline must be above zero (one-sided α = 0.05). Its power is only 0.12, so a miss will not disprove a small effect. That is another reason to keep the board low-stakes: a tiebreaker costs little when it is wrong. A negative estimate would retire it.

## Limitations

- **Not confirmed out of class.** The fully nested log score is positive under the skeptical and 2× priors but not the wide one, and every R² interval spans zero.
- **Prior-dependent.** The planning number is the smallest of three prior scales.
- **Mechanism unknown.** Path direction and alignment come from tracking, not film. With no blockers or ball in the data, we cannot say what the hoop captures. We call it change-of-direction force, not bend.
- **Usage and health.** Pressure rate ignores injuries, scheme and how a rusher is used (passing downs, rotation). We adjust only for games per season and EDGE share.
- **Small n and missing times.** There are about 30 hoop runners with NFL rushes per class, and most lack a 3-cone.
- **Sampled game tracking:** 20 random rushes per player, without scheme labels.
- **Short careers.** 2025 has one season, so the board ranks players but does not set levels.

*Reproducible: `uv run python writeup/candidates/round_07/cand_1/analysis.py` writes `numbers.json` (only the numbers this text uses), `claims.json` (key → sentence) and every figure, then fills this text from `writeup_template.md`.*
