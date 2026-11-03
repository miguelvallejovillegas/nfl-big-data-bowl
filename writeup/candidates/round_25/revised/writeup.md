# Run-the-Hoop Lean: A Film Flag for Day-3 Edge Rushers, Not a Board Input

Day-3 linemen who carry the most speed through the Combine's tightest corner went on to out-rush their draft slot by about 2 pressures a season. We call that corner lean **Hoop G**. It is a reason to open a rusher's tape first. It is not a reason to move him on the board: it was the best of 6 drill measures we tried, the 2025 class alone is flat, and it does not pick more hits than draft order.

- **Scout, Monday:** in each class, flag the day-3 and undrafted linemen in the top third on Hoop G (about 8 names) and watch their rushes before the board meets. Clubs do not draft for it: a 1-sd edge in Hoop G moves a lineman's pick 2% earlier [95%: −13, 15].
- **Combine:** grade four hoop laps, not one; a single lap repeats at only 0.43.

**Why lean should produce pressure.** On a wide rush the tackle kicks to a spot and the rusher has about 2.5 seconds to get his outside shoulder past the tackle's hip and flatten to the quarterback's launch point. The tighter the arc he can run at full speed, the shorter that trip. A rusher who has to throttle down to turn gives the tackle time to recover and ride him past the pocket. Bend, dipping the shoulder and flattening, is graded only off tape. The hoop is the one Combine drill that measures it, and 2023–25 are the first tracked classes with NFL snaps to check it against. 69% of these 118 linemen went on day 3 or undrafted, where tape is thinnest.

## Hoop G in coach terms

Speed squared over turn radius at his hardest half-second of the figure-eight, in g, ranked within his Combine class. The median lineman carries 7.1 mph around a 1.8-yard radius (0.62 g). On an NFL rush his hardest lean is 5.6 mph around 4.1 yards (0.18 g): the tackle pushes the arc out to 2.7 times his hoop radius, so we compare men within a class, never raw g across venues. A pressured snap cost the offense 0.52 expected points on 1655 sampled pass plays.

![Figure 1. Four flagged day-3 rushers picked by fixed rule (top-Hoop-G hit, biggest hit, the miss with most corner turns, biggest miss), in Hoop G order: one Combine hoop lap each (top), their randomly sampled NFL rushes (middle) and the numbers (bottom).](figures/fig1_hero.png)

## Does it hold on a class the model never saw?

**Outcome:** career pressure rate through 2025, shrunk toward the 8.1% mean, for the 77 linemen with 30+ rushes and every drill measure. **Test:** each class (27/23/27 rushers) is scored by a model fitted on the other two; drill measures are standardized within Combine year. **The one test**, written down for 2026: Hoop G against a forecast from draft pick plus edge-or-interior role, the information a club has without tracking.

| Question (held-out classes) | Result [95%] | n |
|---|---|---|
| **Extra pressures a season per sd of Hoop G, beyond pick + role** | **+2.3 [0.7, 4.0]** | 77 |
| Same, beyond a shrunk forecast from pick, role, weight, NGS score, 10-yd split and 3-cone | +2.4 [0.7, 4.2] | 77 |
| Share of rusher-to-rusher spread explained (R²): pick + role, then with Hoop G | 0.10 → 0.16 | 77 |
| R² gain when the best of 6 measures is chosen inside each training fold | +0.06 [−0.27, 0.17]; shuffle p = 0.017 | 77 |
| Day 3: rank correlation with pressure, pick order vs pick + role + Hoop G | 0.01 vs 0.26; gap 0.24 [−0.11, 0.55] | 47 |
| Day 3: top third that beat its class median, pick order vs Hoop G list | 9 of 15 vs 9 of 15 | 47 |

*Table 1. Intervals from class-stratified bootstraps that refit every fold. The fold-chosen row repeats the six-way choice inside every training pair, so it pays for the search; it chose Hoop G in 1 of 3 folds and other hoop measures in the rest.*

The slope is the dependable part. The accuracy gain is not: once the six-way search is paid for, its interval spans zero. The slope holds with wave-drill lean in the model and with snap load and club pressure rate held fixed, and it is flat in 2025 (Figure 3; details in Table A1).

![Figure 2. Every rusher's pressures beyond the pick + role forecast against his Hoop G, held-out classes, 95% band.](figures/fig2_effect.png)

![Figure 3. The checks on one scale: wave-drill lean, club resampling, usage controls, the 2025 split and an offensive-line control, 95% intervals.](figures/fig3_checks.png)

## Drill to Sunday: what we can and cannot say

NFL bend tracks pressure, but the hoop does not predict how often a man bends on Sunday, so we cannot say for sure why the signal exists. We checked this on the 1315 randomly sampled rushes of 79 hoop runners, with a player random effect, rather than on per-man averages. A rush on which a lineman leaned 1 sd harder than his own norm ended in pressure 3.6 points more often [95%: 2.0, 5.2]: bending pays on Sunday. Hoop G barely moved how hard he leaned (0.05 sd per sd [−0.01, 0.11]).

One lead for tape: on corner turns (a rush in the top quarter of lean), each sd of Hoop G added 4.0 points to the pressure rate [95%: 0.5, 7.6]. With errors clustered on the player that weakens to p = 0.11, so treat it as a hypothesis. The hoop may measure finishing a bend more than how often a man bends. Figure 1 shows the problem: the hoop orders Javon Solomon, Cedric Johnson, Eric Watts and Jaylen Harrell, but Harrell bent most on Sunday (+1.7 sd) and still finished −8.1 pressures a season against his forecast.

## Day 3: a film queue, not a board

Among the held-out day-3 and undrafted rushers, adding Hoop G to pick + role orders pressure better than pick order does (Table 1), but the gap's interval includes zero and both top thirds hit equally often. In a simulated 2026 day-3 class, the flag list wins 52% of the time, ties 26% and loses 23%. So it decides **which tape opens first**.

Table 2 lists the 7 rushers the flag added beyond pick order's top third, in the order a scout would open them. Most rarely bent on their sampled rushes, the same gap between drill and Sunday. The tape cue follows rules fixed in code from each man's sampled rushes.

| Rusher | Class, pick | Hoop G (sd) | Pressures a season vs forecast | Beat class day-3 median | Corner turns, random rushes | What to look for on tape |
|---|---|---|---|---|---|---|
| 1. Nick Hampton (CAR) | 2023 #161 | +2.2 | +3.8 | yes | 3 of 18 | rarely bent on sampled rushes; grade counter and power |
| 2. Warren Brinson (GB) | 2025 #198 | +1.3 | +4.9 | yes | 2 of 16 | rarely bent on sampled rushes; grade counter and power |
| 3. Cedric Johnson (CIN) | 2024 #214 | +1.1 | +17.9 | yes | 7 of 19 | bends; pressures come other ways, so grade the finish |
| 4. Robert Beal (MIA) | 2023 #173 | +1.0 | −2.9 | no | 4 of 17 | rarely bent on sampled rushes; grade counter and power |
| 5. Tyrion Ingram-Dawkins (MIN) | 2025 #139 | +1.0 | −3.0 | yes | 3 of 16 | rarely bent on sampled rushes; grade counter and power |
| 6. Zeek Biggers (MIA) | 2025 #253 | +0.9 | −0.1 | no | 2 of 16 | rarely bent on sampled rushes; grade counter and power |
| 7. Eric Watts (MIA) | 2024 UDFA | +0.6 | −9.7 | no | 4 of 17 | rarely bent on sampled rushes; grade counter and power |

*Table 2. Held-out classes; 4 of 7 beat their class's day-3 median. Corner turn = top quarter of lean among all sampled rushes (25% is average).*

Figure 4 is the live queue: every flagged 2024–25 day-3 rusher with 30+ NFL rushes.

![Figure 4. The day-3 film queue for the 2024–25 classes in Hoop G order, with each man's random NFL rushes and his result against the forecast (95%), plus the simulated 2026 race against pick order.](figures/fig4_film_queue.png)

The 2026 Combine tracking is not in this data release, so the 2026 names cannot be listed here; `analysis.py` scores a class the day its hoop clips load, with the rule frozen in `frozen_rule.json`.

## The 2026 test

Written down before any 2026 NFL data: after Week 18, check whether Hoop G still predicts pressure beyond the frozen pick + role forecast. A positive slope at one-sided p < 0.10 keeps the flag; zero or below retires it. One class detects a real slope this size 59% of the time; three classes, 92%. Hoop G reaches the board only if its day-3 top third beats pick order by 20+ points over two classes.

## Limitations

- **Small samples, single laps.** 77 rushers in the model, 47 on day 3; most ran one hoop lap.
- **Forking paths.** Hoop G was the best of 6 measures; paying for that keeps the slope but not a reliable accuracy gain. Only 2026 is a clean test.
- **Scheme and alignment.** Snap load and club pressure rate are controlled; stand-up versus hand-in-dirt is not, because only 0.09% of these linemen's snaps are tagged OLB.
- **Thin NFL tracking, no blockers or film.** About 17 random rushes per man; the corner-turn lead is exploratory.
- **Missed time.** Rates are per rush; the 33 linemen under 30 rushes are left out.

## Bottom line

Hoop G is a small signal that we cannot yet explain. It moves about 8 day-3 rushers a year to the front of the film queue and should not touch the board. Grade four laps and let the 2026 class decide.

## Appendix

| Check (n = 77 unless stated) | Estimate | 95% interval or p |
|---|---|---|
| R² gain over pick + role, clubs resampled (30 clubs) | positive in 70% | p = 0.298 |
| Shuffle p, Hoop G within class and role: slope / best of 6 slope | 0.001 / 0.019 | |
| Wave-drill lean in the model (n = 76): Hoop G / wave lean | +2.1 / +1.5 | [0.5, 3.8] / [−0.5, 3.5] |
| R² gain of Hoop G beyond wave lean | +0.06 | [−0.13, 0.22] |
| Holding snap load and club pressure rate | +2.1 | [0.4, 3.9] |
| Class by class: 2023, 2024, 2025 | 2.9, 3.3, 0.6 | Hoop G × class p = 0.33 |
| 2025 pooled with 2023–24; chance of a positive 2026 slope | +1.3; 88% | |
| 2023–24 rookie seasons / later seasons (n = 47) | +2.3 / +3.0 | later [0.7, 5.7] |
| Edge (n = 38) / interior (n = 39) | +2.4 / +2.8 | |
| Rush level: a man's average lean; tightness-matched turn × Hoop G | +8.6; +4.1 points | [2.4, 14.8]; [1.1, 7.2] |
| Per-man Hoop G vs NFL bend; largest value noise allows | 0.14; 0.53 | |
| Day 3, pooled AUC for beating the median: pick / Hoop G list | 0.54 / 0.65 | gap [−0.08, 0.37] |
| Priors and rush minimums; drop one rusher | 1.8 to 2.6; 2.0 to 2.5 | |
| Independent script, no shared code (n = 77) | +2.3 | 90%: 0.8, 3.8 |

*Table A1. Multiplicity: of 36 tests, 2 survive a family-wise correction at 0.10 and 12 a false-discovery correction (`multiplicity.csv`).*

**Clips (season, week, quarter, clock, offense).** Javon Solomon: 2024 wk 18 Q4 12:22 v NE (pressure); 2024 wk 5 Q2 00:12 v HOU (incompletion). Cedric Johnson: 2025 wk 9 Q2 02:26 v CHI (pressure); 2024 wk 16 Q3 01:35 v CLE (scramble). Jaylen Harrell: 2024 wk 12 Q4 11:24 v HOU (pressure); 2024 wk 12 Q2 02:00 v HOU (completion). Eric Watts: 2024 wk 5 Q2 08:46 v MIN (incompletion); 2024 wk 18 Q1 09:23 v MIA (incompletion).

![Figure A1. The six pre-listed drill measures with 95% intervals (A), the shuffled-slope placebo (B) and the fold-chosen R² gain against its own shuffles (C).](figures/figA1_screen.png)

`analysis.py` (about 9 minutes) rebuilds every number, figure and this text from `data/processed`, and runs `verify.py`.
