# Hoop G: A Free Day-3 Pass-Rush Tiebreaker

**Day-3 and undrafted linemen who bend hardest in the Combine's Run-the-Hoop drill matched their no-tracking pressure forecast (7.3% vs 7.3%); the rest of day 3 fell short (5.7% vs 6.9%). Clubs did not draft for it. Use it to break ties between cheap rushers, not to grade them. Combine get-off burst did not carry through to pressure.**

![Figure 1. (A) Each day-3/undrafted rusher with 30+ NFL rushes: pressure rate minus his out-of-class forecast, sorted; orange, named = top-third Hoop G (the shortlist). (B) Share beating the forecast, by class and pooled, 90% Wilson intervals. (C) Median draft pick of each Hoop G third, 90% bootstrap intervals.](figures/fig1_shortlist.png)

## The decision

Every April a club spends day-3 picks and undrafted money on its fourth and fifth pass rushers, and in August it cuts one of them. A pressure is what that player is for: on 1655 randomly sampled NFL pass plays involving these rookies, a pressured snap cost the offense 0.52 more expected points than a clean one (90% interval 0.36 to 0.67). 70% of the linemen in this study went on day 3 or undrafted, so this is where most of the decisions are.

The usual agility evidence is missing exactly there: only 43 of 118 linemen in the 2023–25 classes ran the 3-cone. Since 2023, though, every lineman's position drills are tracked at 10 Hz, and nobody times them. The question is whether one of those drills sorts cheap rushers better than what the board already shows.

## The metric

**Hoop G is how hard a lineman pushes sideways while rounding the hoop, compared with his Combine class.** Run-the-Hoop is a figure-eight around two hoops that rewards bending the hips at speed. We take the hardest 0.5-second window of sideways (centripetal) acceleration on the lap (median 0.62 g) and standardize it within Combine year, because drill conditions shift between years. 109 linemen have it, 69 with no 3-cone.

The contrast is **rush-drill burst**: yards covered in the first second of the pass-rush drill, standardized the same way.

Both are noisy. 91% of hoop runners have one run; among the 10 with two, run-to-run reliability is 0.37. One burst rep repeats at 0.13.

![Figure 2. Four rushers' hardest hoop lap, coloured by sideways acceleration on one shared scale; the red ring marks the half-second that defines Hoop G.](figures/fig2_laps.png)

## The test

**The yardstick** forecasts each lineman's NFL pressure rate from what a club already has: draft pick, Combine weight, NFL EDGE share, NGS athleticism score, and the official 10-yard split and 3-cone (missing times filled from weight, role and NGS score). We use the official split, not a tracked one: the tracked 40 clip starts before the laser, so the tracked split runs 0.28 s slow and agrees with the official at only r = 0.43 (93 linemen). Each class is forecast by a model fit only on the other two; on held-out classes it ranks regulars' pressure rates at ρ = 0.37.

**The rule**, fixed in round 13 and unchanged: shortlist = pick 103 or later (or undrafted) AND top third of Hoop G in his class. Nothing in it is fit on NFL data. A rusher with 30+ NFL rushes "hits" if his pressure rate beats his forecast.

**Result.** 8 of 15 shortlisted rushers hit, against 8 of 32 others: a gap of **+28 points**. The shortlist was ahead in 3 of 3 classes. In a regular's season of 450 rushes that is +5.4 pressures (90%: −0.5 to 11.9), about 2.8 expected points.

| Yardstick | Shortlist beat it | Rest of day 3 | Gap, points (90%) |
|---|---|---|---|
| **Primary: forecast fit on the other two classes** | 8 of 15 | 8 of 32 | +28 (4 to 50) |
| Pick-and-role-only forecast | 8 of 15 | 7 of 32 | +31 (7 to 53) |
| Time-forward: fit on earlier classes only (2024–25 scored) | 5 of 11 | 5 of 20 | +20 (−8 to 47) |
| Primary, reweighted for who reached 30 rushes | weighted | weighted | +18 (−7 to 47) |

The pick-and-role-only row shows the gap is not cheap players beating a forecast anchored on draft slot. The bottom two rows are the honest ones. Scoring each class only with earlier classes, or reweighting for the 29 day-3 runners who never reached 30 rushes (9 shortlisted), shrinks the gap to +20 or +18 points and the interval crosses zero. A selection model says Hoop G did not decide who got on the field (odds ratio 1.18 per sd; 90%: 0.68 to 2.07).

**Verdict: a small, uncertain edge, enough to break a tie, not enough to move a player up a board.** Calibration says what kind of edge: the forecast was right about the shortlist and too kind to the rest. Across all 78 scored hoop runners (Figure 3), pressure over forecast rises 0.88 points per sd of Hoop G (90%: 0.43 to 1.31). But Hoop G was the best of 14 drill-and-outcome pairs screened in earlier rounds; priced for that search, p = 0.10.

![Figure 3. Every scored hoop runner: NFL pressure rate minus the out-of-class forecast against Hoop G. Orange = day-3 shortlist; marker area = NFL rushes; line = rush-weighted slope.](figures/fig3_scatter.png)

## Get-off: drill burst reaches the snap, not the pressure

Each link of the get-off chain holds on its own (Figure 4). After weight and role, drill burst tracks how far a lineman travels in the first second of a sampled NFL rush: ρ = 0.27 (90%: 0.09 to 0.42; 84 linemen), 0.62 once corrected for noise. That NFL first-second burst agrees with NGS season get-off (0.47), and season get-off predicts pressure over forecast (0.35).

Multiplied through, the chain leaves about 0.09. Measured directly, drill burst moves pressure over forecast by −0.02 points per sd (90%: −0.66 to 0.65; 77 rushers), which rules out anything larger than about two-thirds of a point. The bottleneck is the single drill rep.

Hoop G is the reverse: it moves pressure (0.22) but not first-second burst (0.07) or depth upfield at 1.5 s (0.04). Our best guess is a rusher who keeps speed around the corner and through contact, which we cannot see without blockers in the data.

![Figure 4. The drill-to-game chain. Spearman ρ with 90% player-bootstrap intervals; the white diamond is the burst link corrected for measurement noise.](figures/fig4_chain.png)

## Two real boards

**2024, around pick 215, EDGE.** Cedric Johnson (pick 214, 260 lb, Hoop G +1.1) and Javontae Jean-Baptiste (pick 222, 239 lb, Hoop G 0.0) were the same price: rookie deals of $1.04M a year each. The rule says Johnson. He pressured on 13.0% of rushes against a 7.2% forecast; Jean-Baptiste 5.0% against 6.9%.

**2025, around pick 124, EDGE.** Kyle Kennard (pick 125, Hoop G +1.2) and Barryn Sorrell (pick 124, Hoop G +0.1), $1.28M and $1.29M a year. The rule says Kennard, and so far it is wrong: 2.0% vs 8.2% on 50 rushes, while Sorrell was at 7.6% vs 8.2% on 119.

Paired this way across every scored shortlisted rusher, the shortlist wins 8 of 15 boards. Any single call is close to a coin flip; the edge exists only on average, +5.4 pressures a season at no extra cost. That is worth deciding the last roster spot, not a first-round grade.

## When drill and tape disagree

| Player (class, pick) | Hoop G call | Pressure vs forecast (rushes) | Scout action and film cue |
|---|---|---|---|
| **Jaylen Harrell** (2024, 252) | Shortlist, +0.4 sd | 4.6% vs 7.0% (461) | Drill said yes, the NFL said no on a large sample. He bends at the top of the arc; does he convert speed to power when the tackle sets short? If not, rotate him rather than build a sub package around him. |
| **Isaiah McGuire** (2023, 126) | Bottom third, −1.2 sd | 7.2% vs 5.8% (824) | The rule's miss: a power rusher who wins straight ahead. Low Hoop G is not a cut signal. |
| **Jordan Phillips** (2025, 143) | Bottom third, −1.9 sd | 1.9% vs 6.6% (215) | Miami looped him on 40% of sampled rushes (interior median 15%). Grade the role before the player: does he redirect on stunts or only push the pocket? |

## So what for a team

1. **Break day-3 ties with Hoop G.** Top-third linemen went at a median pick of 158, no earlier than the bottom third (130); per sd of Hoop G, clubs drafted 3% later (90%: 11% earlier to 20% later). The tiebreaker costs no draft capital.
2. **Match the role to the trait.** Shortlisted EDGEs get third-down snaps from wide alignments, where bend matters most; shortlisted interior players get the looper job on stunts.
3. **Do not draft on drill burst.** Time get-off on tape or in camp.
4. **2026 Combine protocol.** Ask for a second hoop run: reliability goes from 0.37 to 0.54 (three runs: 0.63). Burst would need about 7 reps to reach 0.5, so it is not worth asking for.

The 2025 shortlist, with games played as the availability check:

| 2025 shortlist (pick, team) | Role | Hoop G (sd) | 2025 games | Rookie pressure vs forecast (rushes; — = under 30) | Plan for 2026 |
|---|---|---|---|---|---|
| **Ethan Downs** (UDFA, KC) | EDGE | +2.3 | 1 | — (7) | Third-down sub-rusher audition |
| **Jah Joyner** (UDFA, FA) | EDGE | +1.5 | 0 | — (0) | Practice-squad look; second hoop run in camp |
| **Jack Sawyer** (123, PIT) | EDGE | +1.4 | 17 | 9.7% vs 8.5% (176) | More third-down snaps from a wide (7/9) alignment |
| **Warren Brinson** (198, GB) | interior | +1.3 | 11 | 6.9% vs 6.6% (173) | Looper on interior stunts |
| **Kyle Kennard** (125, LAC) | EDGE | +1.2 | 5 | 2.0% vs 8.2% (50) | Third-down sub-rusher audition |
| **Antwaun Powell-Ryland** (209, CIN) | EDGE | +1.1 | 0 | — (0) | Practice-squad look; second hoop run in camp |
| **Tyrion Ingram-Dawkins** (139, MIN) | EDGE | +1.0 | 14 | 5.3% vs 8.0% (113) | Hold snaps; grade his wide-arc rushes |
| **Zeek Biggers** (253, MIA) | interior | +0.9 | 9 | 4.9% vs 6.4% (142) | Early downs; camp test as stunt looper |
| **Nazir Stackhouse** (UDFA, TEN) | interior | +0.4 | 13 | 1.4% vs 5.9% (71) | Stunt-package audition |
| **Tim Smith** (190, IND) | interior | +0.4 | 0 | — (0) | Practice-squad look; second hoop run in camp |

The same rule, unchanged, will score the 2026 class after the 2027 season.

## Limitations

- **One rule, chosen after looking.** Holding classes out checks the forecast, not the drill choice (search-adjusted p = 0.10). The time-forward and reweighted versions are weaker; the 2026 class is the clean test.
- **Small counts.** 15 shortlisted vs 32 comparison rushers; 6 scored 2025 rookies have one season. Hoop G reliability rests on 10 linemen with two runs.
- **Forecast level.** Fit on 100+ rush regulars, the forecast sits above most day-3 rushers, so the test compares day-3 rushers with each other, not with 50%.
- **Thin NFL tracking, no injuries or scheme labels.** About 20 random sampled rushes per player, no blockers, no ball. Sampled pressure barely repeats (split-half 0.04), so pressure comes from season totals; scheme reads come from loop rates, and availability only from games played.

## Appendix

| Check | Result |
|---|---|
| Shortlist vs rest by class (2023, 2024, 2025) | 3/4 vs 2/12; 3/5 vs 3/7; 2/6 vs 3/13 |
| Class-stratified (Mantel-Haenszel) odds ratio, 90%; p; class homogeneity p | 2.8 (1.0 to 7.6); 0.16; 0.46 |
| Primary gap: 90% Newcombe interval; flat-prior P(gap > 0); within-class relabelling p | 4 to 50 points; 97%; 0.037 |
| Shrunken (ridge) forecast; without Cedric Johnson | 8/15 vs 6/32; 7/14 vs 8/32 |
| Reached 30 NFL rushes: shortlist vs rest | 15/24 vs 32/52 |
| Same rule on the official 3-cone (day 3, ran it) | 1/3 vs 3/13 |
| Top-third Hoop G on days 1–2 (not part of the rule) | 7/12 vs 6/19 |
| Hoop G slope 90% interval; drill burst → pressure ρ | 0.43 to 1.31; 0.06 (−0.14 to 0.26) |
| Reliability: burst over a lineman's reps; NFL first-second burst over 8+ sampled rushes | 0.22; 0.83 |

*Reproduce: `uv run python writeup/candidates/round_15/cand_1/analysis.py` writes `numbers.json`, the board, the 2026 rule and all figures, and fills this text.*
