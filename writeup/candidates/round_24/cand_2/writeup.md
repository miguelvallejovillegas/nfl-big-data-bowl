# The Combine Grades a Cut Sundays Don't Ask For: Receiver Break G Buys No Separation

**Receivers who cut harder in Combine route drills are not more open in the NFL. On 116 receivers from draft classes the forecast never saw, one standard deviation of drill Break G moved separation over expected by −1.2 inches (90% interval −3.7 to +1.0).**

**Break G** measures how hard a receiver plants. It is his average g-force over the one second around his first real cut, covering the brake into the cut, the turn and the drive out. A typical NFL hitch break is 0.37 g; the same break in the Combine hitch-family drills is 0.63 g. On film it reads as three speeds: into the cut, at the plant and out of it.

**Why now.** This is the first draft cycle with 10 Hz tracking of every Combine route drill next to NFL tracking of the same men. Before a room buys a "route-running grade" built from these clips, it should know whether the cut shows up on Sundays. Clubs already pay for it: at the same 40, weight, NGS score and position, a receiver 1 sd higher in Break G goes 14% earlier in pick number (90% interval 4 to 24; 149 receivers).

**Data**
- **Combine:** 926 tracked reps in ten route drills (2023–2025 classes), 885 with a gradable cut, each scored against the same drill and year.
- **NFL:** the same code on 1763 sampled regular-season breaks by 115 receivers, in the five route families the drills copy: slant, out, hitch (curl, comeback, hook), in (dagger) and corner (sail).
- **Outcome:** separation over expected (SOE). Inches more open at the throw than Next Gen Stats separation predicts from alignment, cushion, position and season, over full regular seasons (213 player-seasons, median 231 career routes), not from the tracking sample.

![Figure 1. Median speed through the first cut (line) and middle half of reps (band), Combine route drills vs sampled NFL snaps of the same route family, aligned on the plant. Rep counts are above each panel. Drill cuts are entered faster in all 5 families.](figures/fig1_family_traces.png)

## 1. The drill asks for a different cut

In every route family, the Combine receiver arrives at the cut faster than he will on Sundays (Figure 1). On the hitch family he arrives at 6.9 yd/s (about 14 mph) in the drill and 4.7 yd/s (about 10 mph) in the NFL. Only 18% of NFL breaks are entered as fast as a typical drill rep. Break G rises with entry speed (rank correlation 0.37 in drills, 0.56 in games). The drill rewards stopping from a speed NFL routes rarely ask for.

Does the same receiver rank the same way in both settings? Figure 2 checks each route family and each part of the cut.

![Figure 2. A: rank correlation between a receiver's drill and NFL value for the same route family, by part of the cut (343 receiver-family pairs, 110 receivers, 2+ NFL breaks per family). B: receiver-level checks, with the 40 as the control that should and does carry over.](figures/fig2_transfer.png)

- **Speed carries.** Official 40 rank vs NFL top speed: 0.46 (90% interval 0.33 to 0.57; 100 receivers).
- **The cut does not.** Drill Break G vs NFL Break G: −0.05 (−0.22 to 0.11; 101 receivers with 8+ sampled breaks). Taking entry speed out of both sides gives −0.08. Given how noisy both measures are, a true link of 0.5 would still read about 0.20, so noise alone does not explain this.
- **One part travels a little: braking.** How fast a receiver sheds speed into the plant carries over at 0.12 (0.02 to 0.21), mostly on hitches and corners. It is one of 30 cells, so treat it as a lead.

## 2. Break G adds nothing to a no-tracking forecast

The test was fixed in advance: fit on two draft classes, score the third, rotate, weight by routes. A ±3 inch margin was set beforehand as the smallest effect a club would act on. Intervals resample whole receivers and refit every step.

| No-tracking forecast | Its held-out R² | R² change adding Break G [90%] | Break G effect, inches per sd [90%] |
|---|---|---|---|
| Pick only | −0.03 | −0.003 [−0.090, 0.033] | −0.1 [−2.0, +2.0] |
| Pick + role (TE, slot share) | −0.11 | −0.029 [−0.190, 0.034] | 0.1 [−2.1, +2.3] |
| Ridge-shrunk: pick, 40, weight, NGS, TE, slot | −0.08 | −0.007 [−0.214, 0.066] | −0.9 [−3.2, +1.4] |
| **Pick + 40, weight, NGS, TE** (set in advance) | 0.00 | 0.013 [−0.160, 0.086] | **−1.2 [−3.7, +1.0]** |

*Table 1. 116 receivers with a drill score and NFL separation. An R² below zero means the forecast does worse on a new class than that class's own average.*

No forecast predicts SOE. Career SOE varies by 8.7 inches between receivers, and draft slot plus testing explains none of it on a new class. Adding Break G does not help under any of the four forecasts; with the pre-set forecast, 0% of resampled effects reach +3 inches. The official 3-cone (0.0 in per sd, 38 receivers) and short shuttle (−0.9, 45) say the same without tracking.

**Placebo.** We shuffled Break G among receivers of the same class 2000 times and reran the test. 90% of the shuffled effects fall between −2.2 and +2.2 inches, and 38% are at least as far from zero as the real −1.2. Real Break G behaves like a shuffled one (Figure 3B).

![Figure 3. A: held-out Break G effect on career SOE (116 receivers) under four no-tracking forecasts, and official agility times in its place (grey), against the ±3 inch margin. B: the same test with Break G shuffled within class; the real value sits mid-distribution.](figures/fig3_baselines_placebo.png)

## 3. Is there a Sunday cut skill to predict?

If the drill misses a real NFL skill, a harder Sunday cut should buy space on the same snap. We compared each receiver with himself, within one route family, on breaks finished before the throw, controlling for cushion and for how early the cut ended. 39% of detected breaks ended after the ball was out. Those make a timing placebo (Figure 4): they cannot create separation at the throw, so they should read zero.

![Figure 4. Same receiver, same route family: extra separation at the throw per sd of NFL Break G on that snap. Intervals resample whole receivers. Counts are on the right.](figures/fig5_same_snap.png)

Across all sampled snaps the link is +6.2 inches per sd (+1.4 to +10.6; 907 breaks, 107 receivers). On random snaps it drops to +1.1 (−8.1 to +10.2); the link lives in the top-EPA half of the sample, where completions to open receivers are over-represented. The placebo reads −13.5 (−19.7 to −7.5), not zero, likely because a covered receiver keeps working and cutting after the throw. Play context, not the cut, most likely drives the same-snap numbers. Between receivers, a man's NFL Break G does not track his career SOE either (−0.15, −0.31 to 0.03; 101 receivers). The drill is not missing a Sunday cut skill: in these data the cut grade is not a separation skill at any level.

## 4. Named receivers: the drill order does not survive

| Receiver (drill rank of 49 WRs) | Combine: in → plant → out, yd/s (g) | NFL, same families: in → plant → out (g) | Career SOE, in | Clip to pull (tracking-located, random snap) | Coach's read |
|---|---|---|---|---|---|
| Kayshon Boutte (#1) | 7.1 → 4.5 → 5.9 (0.62) | 5.7 → 5.0 → 5.9 (0.48) | +0.5 | 2024 wk 6, Q4 07:39, out: 0.48 g, 4.7 yd open | Hardest drill planter; on Sundays sheds only 1.4 mph and rounds the break. Rep sink-and-plant at game tempo. |
| Ryan Flournoy (#25) | 7.2 → 3.9 → 6.5 (0.61) | 4.7 → 2.4 → 4.8 (0.41) | −6.0 | 2024 wk 16, Q3 07:29, slant: 0.41 g, 4.3 yd open | Slowest into Sunday breaks; the defender sees it coming. Rep a faster stem. |
| Jake Bobo (#49) | 6.7 → 3.1 → 5.6 (0.53) | 5.0 → 3.9 → 5.5 (0.46) | −0.8 | 2023 wk 14, Q2 11:23, slant: 0.48 g, 3.5 yd open | Softest drill cutter, ordinary Sunday cut. Nothing to fix. |

*Table 2. Medians over each man's reps in route families he ran in both settings (7/16, 9/18 and 7/25 drill/NFL breaks). The clip is his random-sample break closest to his own NFL median, listed so a scout can confirm the read on film. We have not watched it.*

Figure 5 shows the eight hardest and eight softest drill cutters. The hardest eight average −0.5 inches of SOE and the softest eight −2.3. Two cards show why separation is not the whole job. Arian Smith ranks #3 and is +15.8 inches more open than expected over 231 routes, a vertical threat whom corners play off. Puka Nacua ranks #5 and sits at −5.8 inches over 1287 routes, a high-volume target who wins in tight windows. Break G could not tell them apart.

![Figure 5. Player cards: the 8 hardest and 8 softest drill cutters among 49 eligible wide receivers, with overall and within-class rank. Each row shows drill Break G, NFL Break G from the same code and career SOE, each with a 90% interval. Grey: the interval crosses zero.](figures/fig4_cards.png)

## 5. What to do on Monday

- **Draft room, fixed rule:** strike "top third of the class on route-drill cut grade" from the board. That rule flags about 17 of 50 drill-tested receivers a year, so expect about 17 2026 names. Today those men go 15% earlier, about 20 slots at their median pick of 133. They have been +0.9 inches more open than the rest (−2.1 to +3.8; 38 with NFL routes). The cost of keeping the rule is about 20 draft slots per flagged player for no measurable separation. The 40 carries over; the cut grade does not.
- **Combine staff:** re-stem the curl, comeback and hook at NFL hitch entry speed (about 4.7 yd/s, a controlled three-quarter stem) and grade at least 15 cuts per receiver. These data cannot test this proposal.
- **Position coaches:** rep breaks at game tempo. Nothing here says a harder full-speed plant buys separation. Braking is the only part that carries over, so coach the sink at game speed.

## Limitations

- **Not blind.** All three classes were seen in earlier rounds of this work. The Break G definition, the ±3 inch margin and the forecast set in advance were frozen before these tests. The ridge row, the placebo and Section 3 were added afterwards.
- **Power.** A true +3 inch effect would be detected 67% of the time (appendix). Harm down to −3.7 inches per sd is not excluded.
- **Tracking is a sample.** Up to 40 NFL plays per player, half random and half top-EPA, with no defenders and no ball. Sections 1, 3 and 4 depend on it. Section 2 uses full-season NGS separation. On random snaps only, drill vs NFL Break G is 0.11 (−0.06 to 0.29; 95 receivers).
- **Section 3 is descriptive.** Without defenders, "cut created space" and "open receiver ran through" look alike; the failed placebo favours the second.
- **Selection.** Only 116 of 149 drill-scored receivers logged NFL routes.
- **SOE is one output.** It leaves out contested catches, yards after catch and earning targets, as the Nacua card shows.

## Appendix: method, reliability and power

A cut is the first turn of 30° or more after 1.5 yd of stem, entered above 2.5 yd/s. Positions are Savitzky–Golay smoothed at 10 Hz, and Break G averages acceleration magnitude over ±0.5 s around the turn. A receiver's score averages his drills (z-scored within drill and year), shrunk toward the mean by between-drill reliability, so the slope needs no further noise correction. NFL breaks are z-scored within route family and position. The 40 is the official time, so the clip's 0.3 s start offset never touches Break G. The ridge penalty is chosen inside the training classes.

A single graded cut has reliability 0.14. At the median of 7 cuts per receiver the score's reliability is 0.52, and 15 cuts would reach 0.70. Simulating the whole pipeline with real noise and route weights, the slope averages 3.0 when the truth is +3, so it is not biased toward zero, and its spread under no effect is 1.4 inches. A true +3 effect is detected 67% of the time and gives a slope as low as −1.2 only 0.4% of the time. True effects of +2 inches or more are unlikely. 88% of resampled effects fall inside ±3 inches. Braking alone gives −1.7 (−3.8 to +0.4). `analysis.py` rebuilds every number, table and figure from `data/processed` in under two minutes.
