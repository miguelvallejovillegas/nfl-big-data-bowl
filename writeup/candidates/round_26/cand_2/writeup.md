# Don't Dock a Receiver for a Soft Combine Cut

**Scouts grade "sink and burst" off the Combine route drills, and clubs pay for it. We measured that cut on every tracked rep and on the same men's NFL breaks. A hard drill plant does not make a receiver more open on Sundays: one sd of drill Break G moves separation over expected by −1.2 inches (90% interval −3.7 to +1.0; 116 receivers, each class scored by a forecast fit on the other two). The usable side is the other end: the softest-cutting third of each class goes 26 picks later yet is +2.2 inches more open than the rest, relative to forecast (−1.2 to +5.7). A soft drill cut is not a reason to pass on a receiver.**

**Break G** measures how hard a receiver plants: his average g-force over the one second around his first real cut, covering the brake into the cut, the turn and the drive out. On film it reads as three speeds: into the break, at the plant and out of it. A typical NFL hitch break is 0.37 g; the same break in the Combine hitch drills is 0.63 g.

**Why it matters.** At the same 40, weight, NGS score and position, a receiver 1 sd higher in Break G goes 14% earlier in pick number (90% interval 4 to 24; 149 receivers). If the grade does not carry, the men it marks down are a discount.

**Data**
- **Combine:** 926 tracked reps in ten route drills (2023–2025 classes), 885 with a gradable cut, each scored against the same drill and year.
- **NFL:** the same code on 1763 sampled regular-season breaks by 115 receivers, in the five route families the drills copy (slant, out, hitch, in, corner).
- **Outcome:** separation over expected (SOE), inches more open at the throw than Next Gen Stats separation predicts from alignment, cushion, position and season. Full regular seasons (213 player-seasons, median 231 career routes), not the tracking sample.

![Figure 1. Route paths around the first cut, Combine drill (top) vs sampled NFL snaps of the same route family (middle): 35 random paths per panel, rotated so the stem points up and mirrored so the break goes right; colour is speed, black dot is the plant. Bottom: median speed through the plant with the middle half of reps shaded.](figures/fig1_route_paths.png)

## 1. The drill asks for a different cut

On the hitch family he enters at 6.9 yd/s (about 14 mph) in the drill and 4.7 yd/s (about 10 mph) in the NFL. Drill entry is faster in all 5 families, and only 18% of NFL breaks are entered as fast as a typical drill rep. Because Break G rises with entry speed (rank correlation 0.37 in drills, 0.56 in games), the drill mostly rewards stopping from a speed NFL routes rarely use.

## 2. Speed carries; the cut does not

![Figure 2. One dot per receiver. A: official 40 vs top speed in sampled NFL snaps. B: drill Break G vs NFL Break G from the same code (8+ sampled breaks). Named receivers are those in Table 2.](figures/fig2_transfer.png)

The control works: a receiver's official 40 ranks with his NFL top speed at 0.46 (90% interval 0.33 to 0.57; 100 receivers). The cut does not: drill Break G vs NFL Break G is −0.05 (−0.22 to 0.11; 101 receivers). Both measures are noisy, but a true link of 0.5 would still read about 0.20. Across 30 route-family-by-component cells (343 receiver-family pairs), the best is braking into the plant at 0.12; corrected for the search by shuffling, its p-value is 1.00.

## 3. Break G adds nothing to a no-tracking forecast

The test was fixed in advance: fit on two draft classes, score the third, rotate, weight by routes; ±3 inches per sd is the smallest effect a club would act on. Intervals resample whole receivers and refit every step. All intervals in this paper are 90%.

| No-tracking forecast | Its held-out R² | R² change adding Break G [90%] | Break G effect, inches per sd [90%] |
|---|---|---|---|
| Pick only | −0.03 | −0.003 [−0.090, 0.033] | −0.1 [−2.0, +2.0] |
| Pick + role (TE, slot share) | −0.11 | −0.029 [−0.190, 0.034] | 0.1 [−2.1, +2.3] |
| Ridge: pick, 40, weight, NGS, TE, slot | −0.08 | −0.007 [−0.214, 0.066] | −0.9 [−3.2, +1.4] |
| **Pick + 40, weight, NGS, TE** (set in advance) | 0.00 | 0.013 [−0.160, 0.086] | **−1.2 [−3.7, +1.0]** |

*Table 1. 116 receivers with a drill score and NFL separation. An R² below zero means the forecast does worse on a new class than that class's own average.*

![Figure 3. A: each dot is a receiver; the vertical axis is his career SOE minus the no-tracking forecast fit on the other two classes. One line per held-out class; the blue wedge is the ±3 in per sd margin. B: the same residual averaged within thirds of each class by drill Break G (route-weighted, 90% receiver bootstrap), with each third's median draft pick.](figures/fig3_heldout.png)

Career SOE varies by 8.7 inches between receivers; draft slot plus testing explains none of it on a new class, and Break G does not help under any forecast. All three classes lean slightly negative: 2023 −1.1, 2024 −1.9, 2025 −0.3 inches per sd (Figure 3A). Stress tests agree:

- **Shuffle placebo.** Shuffling Break G within class 2000 times, the one-sided p-value for "harder cutters are more open" is 0.81.
- **Same team, same quarterback.** Resampling whole teams (32) gives −4.0 to +1.2.
- **Coverage.** Man-coverage share is unrelated to season SOE (0.01); adding it to the expected-separation model leaves −1.2 (−3.5 to +0.7).
- **Availability.** Keeping only seasons of 10+ games, a crude injury filter, gives −1.2 (−3.9 to +1.1; 90 receivers); with both changes, −1.1.
- **No tracking.** The official 3-cone (0.0 in per sd, 38 receivers) and short shuttle (−0.9, 45) say the same.

## 4. The draft-room view: soft cutters are a discount

Boards use tiers, not slopes. Split each class into thirds by drill Break G (Figure 3B). The softest third (40 receivers) went at a median pick of 160, against 134 for the rest. On held-out separation they beat the no-tracking forecast by +2.2 inches more than the rest (−1.2 to +5.7), and 52% of them beat their own forecast against 41% of the others. Only 9 of 2000 bootstrap resamples put them 3 or more inches worse, the smallest gap that would justify marking them down. Among day-3 and undrafted receivers alone (28 soft vs 46 others) the gap is +1.4 (−4.1 to +7.5), too noisy to claim but pointing the same way.

This is the testable prescription, frozen in `frozen_rule.json` before the 2026 class has a snap: score the 2026 class with the same code; through 2027 the soft third will trail the rest by less than 3 inches of SOE minus forecast, and the hard third will not lead by 3 or more. Either failure, with a 90% interval excluding zero, falsifies this paper.

## 5. Named receivers, chosen by rule

![Figure 4. The three hardest and three softest drill cutters among 49 eligible WRs (200+ NFL routes, 5+ sampled breaks). Top: thick orange lines are his Combine cuts, thin blue his NFL breaks in his most-run shared route family. Bottom: his mean Break G against peers in each setting, and his career SOE, both with 90% intervals.](figures/fig4_named_paths.png)

| Receiver (class, pick; drill rank of 49) | Combine in → plant → out, yd/s (g) | NFL, same families (g) | Career SOE, in [90%] (routes) | Clip to pull (play log) | Read |
|---|---|---|---|---|---|
| Kayshon Boutte (2023, 187; #1) | 7.1 → 4.5 → 5.9 (0.62) | 5.7 → 5.0 → 5.9 (0.48) | +0.5 [−5.6, +6.7] (873) | 2024 wk 6, Q4 07:39, out; 4.7 yd open, targeted, caught, 13 yd | Hardest drill planter; sheds 1.4 mph into Sunday breaks vs 5.3 at Indy. Average separation. |
| Adonai Mitchell (2024, 52; #2) | 6.7 → 3.7 → 6.4 (0.60) | 6.3 → 4.4 → 5.2 (0.48) | −0.6 [−8.9, +7.6] (486) | 2025 wk 13, Q1 02:19, out; 4.8 yd open, targeted, caught, 7 yd | Second-round pick; ordinary Sunday plant (+0.3 sd), average separation. |
| Arian Smith (2025, 110; #3) | 7.6 → 1.5 → 5.6 (0.67) | 5.5 → 3.4 → 4.8 (0.48) | +15.8 [+3.9, +27.8] (231) | 2025 wk 7, Q3 10:36, in; 2.4 yd open, not targeted | Near-dead stop in drills, ordinary Sunday cut (+0.2 sd), wide open. The space is not coming from the plant. |
| Jalin Hyatt (2023, 73; #47) | 7.7 → 6.2 → 6.6 (0.51) | 6.4 → 2.5 → 2.8 (0.46) | +2.6 [−4.4, +9.6] (679) | 2024 wk 10, Q4 14:55, slant; 6.2 yd open, not targeted | Rounds drill cuts at speed (sheds 2.9 mph); on Sundays he sits down hard on hitches and is slightly open. |
| Isaiah Bond (2025, UDFA; #48) | 7.1 → 3.2 → 5.7 (0.59) | 5.7 → 3.8 → 4.1 (0.38) | +4.3 [−5.6, +14.3] (333) | 2025 wk 3, Q4 14:32, out; 2.4 yd open, not targeted | Undrafted, softest Sunday cut of the six (−0.5 sd), above-expected separation. The case for not docking. |
| Jake Bobo (2023, UDFA; #49) | 6.7 → 3.1 → 5.6 (0.53) | 5.0 → 3.9 → 5.5 (0.46) | −0.8 [−11.2, +9.6] (306) | 2023 wk 14, Q2 11:23, slant; 3.5 yd open, not targeted | Softest drill cutter, undrafted, ordinary Sunday cut, near-average separation on a real workload. |

*Table 2. Medians over route families he ran in both settings. The clip is his random-sample break closest to his own NFL median, with the official play result, so a scout can pull the snap. Reads come from tracking and the play log; we have not watched the film.*

Every one of the six plants softer on Sundays than at Indianapolis (Table 2, g), and the hard and soft ends converge on the league average (Figure 4, bottom). Only Smith's separation interval excludes zero, and he is a hard cutter whose Sunday cut is ordinary.

## 6. What to do on Monday

- **Draft room:** do not move a receiver up the board, or down it, for his route-drill cut grade. In 2026 expect about 17 soft-third names per class; when one is on the day-3 board with a fine 40, treat the soft cut as noise and take the discount. Pay for the 40, which does carry.
- **Scouts:** grade the Sunday tape over the drill plant; Table 2 shows how to pull the clip.
- **Combine staff:** re-stem the curl, comeback and hook at NFL hitch entry speed (about 4.7 yd/s, a controlled three-quarter stem) and grade at least 15 cuts per receiver.
- **Position coaches:** rep breaks at game tempo; nothing here says a harder full-speed plant buys separation.

## Limitations

- **Not blind.** All three classes were seen in earlier rounds. The Break G definition, the ±3 inch margin and the pre-set forecast were frozen before these tests; the ridge row, team bootstrap, placebos, coverage and availability checks and the thirds view came afterwards. The 2026 rule is the first truly out-of-sample test.
- **Power.** A true +3 inch effect would be detected 67% of the time (appendix). Harm down to −3.7 inches per sd is not excluded, and the soft-third discount rests on 40 men.
- **Tracking is a sample.** Up to 40 NFL plays per player, half top-EPA, no defenders or ball; Sections 3 and 4 use full-season NGS separation instead. On random snaps only, drill vs NFL Break G is 0.11 (−0.06 to 0.29; 95 receivers).
- **Controls are coarse.** Coverage enters as a season man share, not per snap; injuries only as a games filter; scheme and quarterback only through team resampling.
- **Selection and outcome.** Only 116 of 149 drill-scored receivers logged NFL routes. SOE leaves out contested catches, yards after catch and earning targets.

## Appendix: method, reliability and power

A cut is the first turn of 30° or more after 1.5 yd of stem, entered above 2.5 yd/s. Positions are Savitzky–Golay smoothed at 10 Hz, and Break G averages acceleration magnitude over ±0.5 s around the turn. A receiver's score averages his drills (z-scored within drill and year), shrunk toward the mean by between-drill reliability. NFL breaks are z-scored within route family and position.

A single graded cut has reliability 0.14. At the median of 7 cuts per receiver the score's reliability is 0.52; 15 cuts would reach 0.70. Simulating the whole pipeline with real noise and route weights, the slope averages 3.0 when the truth is +3 (no attenuation); its spread under no effect is 1.4 inches. 88% of resampled effects fall inside ±3 inches. `analysis.py` rebuilds every number, table and figure from `data/processed`.
