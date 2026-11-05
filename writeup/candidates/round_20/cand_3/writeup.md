# Playing Like You Tested Is Not a Pass-Rush Skill: Combine-to-NFL Carry-over Adds Nothing Beyond the Pick

**Some pass rushers move on Sundays exactly as they did in the Combine pass-rush drill. Others look nothing like it. We call that match Carry-over. On draft classes the model never saw, one sd of Carry-over is worth −0.27 pressure-rate points (90% interval −0.67 to 0.13; 68 rushers). That is inside a ±1-point margin fixed before the held-out run (equivalence p = 0.001). All 6 Combine-side versions we tried are equivalent to zero. None ranks same-class rushers better than the draft pick, which gets 61% of pairs right.**

**Carry-over, in one sentence a coach can repeat:** take his pass-rush drill rep and about 20 of his random NFL rushes. Every tenth of a second for 1.5 s from his first move, score his speed and acceleration against rushers of the same weight and role. Carry-over is the average gap between the two curves, in sd, where 0 means he plays exactly as he tested. *Pressure rate* is pressures per pass rush (DL average 8.1%). One point is about 3 pressures over a 300-rush season.

![Figure 1. Speed against weight- and role-matched peers, drill rep (orange, dashed) vs. his mean NFL rush (blue). Shaded: the gap Carry-over averages (acceleration is also counted, not drawn). Left two: the most alike of 68 rushers. Right two: the least alike. Forecast = no-tracking model fit on the other two draft classes.](figures/fig1_curves.png)

## Pass rushers are the one group where the test is fair

We first ran the comparison for every group against the drill that starts from the same stance: OL pass-set drill, WR go route, TE in route, DB backpedal-and-break, plus the 40 for everyone (Figure 2). The data are 1382 tracked Combine reps and 5305 random sampled NFL pass snaps. We never use the top-EPA half of the sample, because it over-represents pressures.

A translation test only means something if both curves can be measured. The *noise ceiling* is the largest correlation a perfect drill-to-NFL link could show, given rep-to-rep and snap-to-snap noise. For the pass-rush drill it is 0.54. For the OL pass-set drill it is 0.00, because a lineman's two reps do not agree. The WR, TE and DB position drills mostly have one rep, so they have no ceiling at all. Only for pass rushers would a null result be informative.

No group's position drill carries over positively. Matched-drill correlations run from −0.33 (DB, one rep) to 0.04 (TE, with 21 players, left aside). In 2 of 10 production checks, Carry-over clears a within-class shuffle at p < 0.05. Both are 40-yard-dash cells (DL p = 0.03; TE p = 0.02, 18 players), about what chance predicts. Held out by class, the DL one shrinks to 0.26 [−0.26, 0.79].

![Figure 2. Ten drill-group pairs. A: drill curve vs. NFL curve, mean speed over 0–1.5 s after weight and role, with the noise ceiling (red tick; * = single-rep drill, no ceiling). B: Carry-over vs. production beyond pick and weight (DL pressure rate, OL pressures allowed reversed, WR/TE yards per route, DB solo tackles). Intervals: rushers resampled.](figures/fig2_screen.png)

## The drill get-off does not carry over; the NFL get-off leans toward pressure

**First step** means mean speed 0.3–0.5 s after first movement. From drill to NFL it translates at ρ = 0.04 [−0.15, 0.22] (84 rushers), well under its ceiling of 0.42. Over the full 1.5 s, NFL speed runs at a median 84% of drill speed (10th to 90th percentile: 72% to 97%).

The two halves of Carry-over behave differently (Figure 3). Drill speed never relates to pressure beyond the pick (|ρ| ≤ 0.05 at every tenth of a second). NFL speed leans positive: ρ = 0.22 [0.00, 0.43] at the first step (shuffle p = 0.086). The NFL first step is a real measurement. It tracks NGS get-off snap by snap (ρ = 0.51, 1546 rushes), and over a median of 19 rushes its reliability is 0.59. The drill first step's reliability is 0.29.

So Carry-over compares a reading that predicts nothing with one that predicts a little, and the match itself adds nothing. Within the sample it even runs the wrong way: ρ = −0.21 [−0.38, −0.04] beyond pick. That holds for edge rushers (−0.24, n = 36) and interior rushers (−0.18, n = 32), and it stays at −0.18 once both curves' levels are controlled.

![Figure 3. Pass rushers, tenth of a second by tenth. Orange: drill speed vs. NFL speed (translation). Grey: drill speed vs. pressure rate beyond pick, weight and role. Blue: NFL speed vs. the same. Bands: 90% bootstrap.](figures/fig3_by_tenth.png)

## On unseen draft classes, every version of Carry-over is worth zero

**Baseline without tracking:** log draft pick, weight, edge share, official 40 time and NGS athleticism score. It is fit on two draft classes and scored on the third (held-out R² 0.31). Each measure is added to that forecast and scored the same way. The ±1-point margin is about 3 pressures a season, roughly the forecast gap between pick 32 and pick 100 (0.9 points).

- **Carry-over:** −0.27 points per sd [−0.67, 0.13]. Held-out R² change −0.025.
- **Carry-over %** (NFL speed as a share of drill speed): 0.42 [−0.13, 0.90]. It is the largest positive version, and the NFL half is doing the work.
- **Curve shape only:** −0.47 [−0.80, −0.02], the wrong direction. **Speed only:** −0.11. **Against the 40:** 0.26. **Drill first step alone:** −0.13.
- **NFL first step alone:** 0.27 [−0.18, 0.68], held-out R² change −0.047. This is a lean, not a proven gain.

![Figure 4. Slope of held-out pressure rate (actual minus no-tracking forecast) per sd of each measure. Big dot: pooled over the three held-out classes. Open markers: each class. Interval: rushers resampled within class, with the forecast refit. Grey: the ±1-point margin. The official 40 row uses a baseline without the 40 time.](figures/fig4_heldout.png)

**As a ranking tool.** We formed 368 pairs of rushers from the same class and role. The earlier pick has the higher career pressure rate in 61% of them [52, 70; intervals resample rushers]. The held-out forecast with Carry-over gets 61% [53, 70], and with the NFL first step 61%. Against the pick, the NFL first step is −1 points [−10, 8]. Neither beats draft order.

## The most and least alike rushers, by name

| Rusher (class, pick; team) | Role | Curve gap, sd (Carry-over pct) | Drill / Sunday first step, pct | Pressure vs forecast (rushes) |
|---|---|---|---|---|
| ***Most alike: plays as he tested*** | | | | |
| Alfred Collins (2025, 43; SF) | interior | 0.28 (100) | 75 / 71 | 5.2% vs 7.4% (260) |
| Laiatu Latu (2024, 15; IND) | edge | 0.33 (99) | 25 / 34 | 9.9% vs 9.4% (1034) |
| Zeek Biggers (2025, 253; MIA) | interior | 0.35 (97) | 50 / 50 | 6.4% vs 5.9% (142) |
| ***Least alike*** | | | | |
| Eric Watts (2024, UDFA; MIA) | edge | 2.15 (1) | 46 / 1 | 5.1% vs 6.7% (225) |
| Yasir Abdullah (2023, 136; ATL) | edge | 2.10 (3) | 1 / 68 | 6.9% vs 8.6% (149) |
| Yahya Black (2025, 164; PIT) | interior | 2.07 (4) | 3 / 60 | 6.7% vs 5.0% (189) |

*Table 1. Pressure vs. forecast: career shrunk pressure rate vs. the held-out no-tracking forecast. Percentiles are within the 68. The full board is in `players.csv`.*

The table does not split into "translators" and "non-translators". Laiatu Latu moves as he tested and is 0.4 points against his forecast. Alfred Collins moves just as closely and is −2.2. The two least-alike rushers, Eric Watts and Yasir Abdullah, are −1.6 and −1.7. Across the pool, the most-alike third averages −0.5 points against forecast and the least-alike third 0.6 (23 rushers each).

## What a staff should do

1. **Do not build or buy a "plays like he tested" score for pass rushers.** Six versions were tested, and all are worth zero beyond the pick on unseen classes.
2. **Keep the pick and the stopwatch as the baseline.** Nothing tested here, including game tracking, ranks same-class rushers better than draft order.
3. **If you chart NFL get-offs, chart enough.** It takes 32 random rushes per player to reach reliability 0.7. Use the first step only as a tiebreak lean (0.27 points per sd, unproven).
4. **Read a single Combine get-off rep as noise.** At reliability 0.29, one rep cannot rank rushers. A club that wants a drill get-off number needs several reps.

## Limitations

- **Sampled tape.** NFL curves come from about 20 random rushes per player, with no blockers or ball. A slow start can be a two-gap assignment, a stunt or a read, not a slow player.
- **Different cue.** The drill is uncontested and starts on a coach's cue, not a snap. We align both curves at first movement, so reaction time is excluded on both sides.
- **Screen before test.** The choice of pass rushers came after the ten-pair screen. 2 of its 10 cells clear p < 0.05, about what chance gives, and the held-out test was fixed after the screen.
- **Small sample.** There are 68 rushers; 16 of 84 with curves have fewer than 100 rushes and are left out. Intervals span about ±0.4 points, so the equivalence verdict depends on the ±1-point margin. Weight and role adjustments use NFL alignment, which itself reflects how teams evaluated each player.

## Appendix

Curves are smoothed (Savitzky–Golay, 5 frames). First movement is when speed first reaches 1 yd/s, minus 0.2 s. A rusher must be set at the snap (under 0.8 yd/s), and NFL speeds are adjusted for alignment. Combine curves are year-standardized. Both curves are then residualized on weight and role. Production is beta-binomial shrunk career pressure rate (100+ rushes). The equivalence test is two one-sided tests on the pooled held-out slope with a bootstrap SE. `analysis.py` writes every number, figure and this text from `data/processed` in about 2 minutes. The Carry-over split-half reliability is 0.61.
