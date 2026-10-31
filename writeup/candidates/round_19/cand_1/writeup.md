# Bend Breaks Day-3 Ties: Run-the-Hoop and NFL Pass-Rush Pressure

**Hoop G is how hard a lineman leans into the turn on the Combine's figure-eight hoop.** It is the most sideways g he holds for half a second. We scored it only on Combine classes the model never saw. There, one sd of Hoop G is worth **1.8 extra pressures per 450-rush season** [90%: 0.2, 3.4] beyond a forecast built without tracking (shuffle p = 0.02). That is real, but probably below our 2.25-pressure action line. Burst out of a stance in the pass-rush drill is worth −0.5 [−2.3, 1.3], so it is out. Use it for one job only: **breaking day-3 ties**. Go calls for 2026 from the day-3 Hoop G shortlist: Cedric Johnson (CIN), Javon Solomon (BUF), Warren Brinson (GB), Austin Booker (CHI) and Jack Sawyer (PIT).

![Figure 1. Extra pressures per 450-rush season per sd of each drill, beyond the no-tracking forecast, scored on the held-out class. Bar = 90% class-stratified bootstrap that refits every fold; shade = bootstrap distribution; diamonds = each held-out class; grey band = too small to act on.](figures/fig1_headline.png)

## Why pass rushers, and why these drills

On 1655 randomly sampled NFL pass plays involving these rookies, a pressured snap cost the offense 0.52 more expected points than a clean one (0.35 to 0.67).

Of the 118 linemen in the 2023–25 classes, 69% went on day 3 or undrafted, and only 43 ran the 3-cone. All ran position drills under 10 Hz tracking, and two copy a rush: the pass-rush drill (get-off from a stance) and Run-the-Hoop (bend at speed).

**One margin, set before fitting:** 2.25 pressures per 450-rush season per sd, which is one pressure every 200 rushes and about 1.2 expected points a season.

## The two drill metrics, and how much to trust them

- **Hoop G** (median 0.62 g) exists for 109 linemen, and 69 of them never ran the 3-cone.
- **Burst** is the yards covered in the first second of the pass-rush drill (median 2.45 yd), for 117 linemen.

Both are standardized within Combine year.

![Figure 2. Six Run-the-Hoop laps from low to high Hoop G, coloured by sideways acceleration on one scale. Each panel prints Hoop G and the speed at that moment. The grey ghost is the median runner's lap.](figures/fig2_laps.png)

| Measure | Repeatability | How measured | n |
|---|---|---|---|
| Hoop G, one lap | 0.37 (0.54 with two laps) | lap to lap | 10 |
| Rush-drill burst, one rep | 0.13 | rep to rep | 101 |
| NFL bend (sideways g, first 2 s) | 0.63 | split-half, sampled rushes | 85 |
| NFL first-second burst | 0.83 | split-half, sampled rushes | 85 |
| NGS get-off | 0.84 | season to season | 40 |
| Pressure rate (100+ rushes) | 0.64 | season to season | 40 |

*Table 1. Reliability of every measure used. 91% of hoop runners have one lap.*

The Combine half is the weak link: the NFL traits repeat well, the single drill reps do not.

## The baseline and the test

**Baseline without tracking:** draft pick (log), Combine weight, NFL EDGE share, NGS athleticism score, and the official 10-yard split and 3-cone. Missing times are filled from weight, role and NGS score inside each fold. We use the official split because the tracked 40 clip starts about 0.3 s before the laser.

**Outcome:** career pressure rate through 2025, shrunk toward the 8.1% mean by 117 rushes of prior, for the 77 linemen with both drills and 30+ NFL rushes.

**Out of class:** each Combine class (27/23/27 linemen) is forecast by a model fit on the other two. The headline estimand is the slope of the held-out miss (actual minus forecast) on the drill. Intervals resample players within class and refit every fold; p-values shuffle the drill within class.

| Model (held-out) | R² | RMSE (pts) | Rank ρ | R² gain over baseline [90%] |
|---|---|---|---|---|
| Pick only | 0.10 | 1.63 | 0.31 | — |
| Baseline | −0.02 | 1.74 | 0.28 | — |
| + Hoop G | 0.04 | 1.69 | 0.37 | 0.06 [−0.20, 0.25] |
| + Rush-drill burst | −0.03 | 1.75 | 0.28 | −0.01 [−0.17, 0.05] |
| + both drills | 0.01 | 1.71 | 0.35 | 0.03 [−0.30, 0.24] |

*Table 2. Leave-one-class-out fit, n = 77.*

The baseline is weak: with 77 rushers, its extra Combine times cost more in fitting noise than they add. Hoop G improves all three held-out measures, though its R² gain includes zero.

## Finding 1: burst reaches the snap, not the pressure

After weight and role, drill burst tracks how far a lineman travels in the first second of a sampled NFL rush (partial ρ = 0.27 [0.10, 0.41], n = 84; ceiling 0.43). That NFL first second matters for pressure (0.33). But drill burst itself does not reach pressure: −0.5 pressures per sd out of class, with 1% of bootstrap draws above the margin. An equivalence test (TOST) against ±2.25 gives p = 0.06.

## Finding 2: Hoop G adds a little, out of class

A lineman one sd higher in Hoop G beats his held-out forecast by 1.8 pressures a season. 96% of bootstrap draws are positive, but only 33% clear the margin. The effect holds in 2023 and 2024 and is flat in 2025 (−0.02 points per sd). Clubs do not draft for it: top-third Hoop G linemen went at a median pick of 158, bottom-third at 130.

Hoop G tracks NFL bend only weakly (0.14 [−0.04, 0.30]), so the path from drill to sack stays hidden.

Across shrinkage, rush minimum and tie width (Table 3), the slope stays between 1.5 and 2.0 pressures per sd, and the tie gain stays between 2.1 and 3.8.

| Setting | n | Slope (pressures/sd) | Shuffle p | Day-3 ties | Hoop G right | Gain per tie |
|---|---|---|---|---|---|---|
| Headline (prior 117 rushes, 30+ rushes, 0.5-pt tie) | 77 | +1.8 | 0.02 | 125 | 56% | +2.6 |
| Shrinkage prior halved | 77 | +2.0 | 0.03 | 104 | 60% | +3.8 |
| Shrinkage prior doubled | 77 | +1.5 | 0.02 | 159 | 60% | +2.1 |
| 20+ NFL rushes | 84 | +1.8 | 0.04 | 149 | 58% | +2.7 |
| 50+ NFL rushes | 74 | +2.0 | 0.02 | 98 | 59% | +3.5 |
| Tie width 0.25 pt | 77 | +1.8 | 0.02 | 67 | 54% | +3.3 |
| Tie width 1.0 pt | 77 | +1.8 | 0.02 | 227 | 58% | +2.7 |

*Table 3. Sensitivity. Each row reruns the whole held-out pipeline.*

## The decision: a day-3 tiebreaker

**Rule (frozen, sha256 `bae8a667e32a…`):** two day-3 or undrafted rushers from the same class are tied when their no-tracking forecasts are within 0.5 points (the action margin). In a tie, take the higher Hoop G. Otherwise, follow the forecast.

Ties are defined by forecast, not pick gap, because late-round draft order does not break them: split by pick gap (under 30, 30–80, 80+ slots; median 54), the earlier pick won 56%, 46% and 47%.

| Over 125 tied pairs (47 rushers) | Higher Hoop G | Earlier pick |
|---|---|---|
| Picked the better NFL rusher | 56% [43, 69] | 50% [37, 64] |
| Error rate | 44% [31, 57] | 50% |
| Pressures a season: gained when right / lost when wrong | +8.7 / −5.2 | |
| Break-even hit rate | 37% | |
| Net pressures per decision | +2.6 [0.1, 5.1], p = 0.03 | +0.8 [−1.7, 3.3] |
| By class 2023 / 2024 / 2025 | 63 / 47 / 54% | 43 / 73 / 40% |
| Nested back-test (tie width re-chosen out of class) | 52% [42, 70], +3.4 | 61% [38, 69], +3.4 |

*Table 4. Decision arithmetic. Intervals resample rushers within class and rebuild the pairs. The p-value shuffles Hoop G across rushers within class.*

**Reading it.** Hoop G wins by margin, not frequency: a right call gains more than a wrong one costs, so even the low end of its hit rate (43%) clears the 37% break-even. But it is not proven better than a coin (p = 0.20), and its edge over draft order is not robust: with the tie width re-chosen out of class (0.25/1.5/0.25 points), earlier pick is right more often (61% vs 52%) and both gain +3.4. Without its five biggest wins, the Hoop G gain falls to +1.6. So: use the 3-cone first where both ran it (77% on 13 pairs), then Hoop G, and never let Hoop G override a forecast gap larger than the margin.

![Figure 3. Day-3 ties: how often each tiebreaker chose the better NFL rusher (A), with the break-even line, and the pressures per season it gained (B). The square is the nested back-test.](figures/fig3_tiebreak.png)

## 2026 board

For snap decisions in 2026 we take every day-3 rusher from the 2024–25 classes in the top third of Hoop G. **Go** means 100+ rushes and at least 0.5 points above the forecast. **No-go** means at least 0.5 below. Everyone else is **Hold**. The flip rate is the 2026 pressure rate that would put his career rate back on the forecast, at his 2025 volume. **Decision date: after Week 18 of 2026.**

![Figure 4. 2026 board. Marker = call (Go blue circle, Hold orange square, No-go vermillion cross); bar = 90% interval of the shrunk career rate; right column = what flips each call.](figures/fig4_board.png)

- **Go:** Cedric Johnson (CIN), Javon Solomon (BUF), Warren Brinson (GB), Austin Booker (CHI) and Jack Sawyer (PIT). Johnson (11.2% vs 7.2% forecast over 216 rushes) and Solomon (10.8% vs 7.2%) are so far ahead that no single 2026 season can flip them. Give both third-down snaps from wide alignments, where bend shows up. Brinson is the interior case (make him the looper on stunts) and flips below 5.1%; Booker flips below 6.1%, Sawyer below 7.4%.
- **No-go:** Eric Watts (MIA), Tyrion Ingram-Dawkins (MIN), Jaylen Harrell (TEN). Each needs a 2026 rate of at least 10.2% to reach his forecast. Hold their snaps unless film shows them flattening at the top of the arc.
- **Hold:** Zeek Biggers (MIA), Nazir Stackhouse (TEN), Kyle Kennard (LAC). Stackhouse and Kennard first need 100 rushes.
- **Auditions (under 30 rushes):** Mohamed Kamara (2024, 158, TB), Gabe Hall (2024, UDFA, PHI), Nelson Ceaser (2024, UDFA, FA), Tim Smith (2025, 190, IND), Antwaun Powell-Ryland (2025, 209, CIN), Jah Joyner (2025, UDFA, FA), Ethan Downs (2025, UDFA, KC).

**What would change the rule.** The frozen test is the Hoop G slope against the frozen forecast for the 2026 class after its first season (one-sided, p < 0.10). Its simulated power is 43% for one class and 77% for three. A slope at or below zero drops Hoop G as a tiebreaker. A slope above the margin justifies a second hoop lap for every lineman, which would lift reliability from 0.37 to 0.54.

## Limitations

- **Small and noisy.** 77 rushers in the model, 47 in the ties, single drill reps; Hoop G reliability rests on 10 linemen with two laps.
- **Chosen after looking.** Hoop G was the strongest of several drill measures screened in earlier work on these same classes. Holding out classes checks the forecast, not that choice; 2026 is the first clean test.
- **Hit rate vs size.** The tie rule wins on a few large gaps, does not beat draft order once the tie width is tuned out of class, and is flat in 2025.
- **Thin NFL tracking.** A median of 17 random sampled rushes per lineman, with no blockers and no ball. NFL bend on a snap may be a result of beating the tackle, not its cause.
- **Survivorship.** Rates need 30+ rushes. Top-third day-3 rushers reached that as often as the rest of day 3 (15 of 24 vs 32 of 52). Injuries and roster moves are not modelled.

## Reproduce

`analysis.py` (about 4 minutes) rebuilds every number, figure and this text. `verify.py` recomputes the shrunk rates from raw counts, Table 2, the held-out slopes, the ties and the back-test from `inputs.csv` in seconds and diffs them against `numbers.json`.
