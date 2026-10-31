# Break G: Hard Combine Cuts Earn Receivers a Few More Targets, Not More Separation

Break G measures how hard a receiver brakes, turns and drives out of the cut in his Combine route drills. Among 116 receivers and tight ends with NFL routes, each sd came with 1.93 more targets per 100 routes than draft slot and testing predicted (95% CI 0.02 to 3.78), but no extra separation. It breaks ties; it does not move boards.

**Why now.** Every Combine route-drill rep is now tracked at 10 Hz, so clubs will be offered "suddenness" scores built from those reps and told they mean getting open. We test that claim on the three classes that have played, against what a club already has: pick, 40, weight and NGS athleticism score.

![Figure 1. Four wide receivers chosen by rule: the two hardest and two softest Combine cutters among those with 5+ route drills and 8+ sampled NFL breaks. A: every rep's path in the second around the cut, seen from above (blue: Combine drills; orange: his NFL slant, out, hitch, in and corner breaks). B: acceleration through the same window. C: his career NFL targets per route run (TPRR) against the forecast from the no-tracking baseline fitted without his class.](figures/fig1_hero.png)

## The metric

**For a coach.** Break G is how hard a receiver pushes into the ground in the one second around his cut, averaged over his route drills. A receiver who rounds the cut and drifts scores low; one who sinks, snaps and drives out scores high.

**On the field.** A typical Combine cut averages 0.57 g: over that second his velocity changes by about 6.1 yd/s in total, for example braking from 7 to 4 yd/s and driving back to 7 on a new line. One sd of true Break G is 0.021 g, or 0.23 yd/s more velocity change per cut. A plus receiver (about +1.5 sd) gains roughly 0.3 yd/s on an average cutter: too little to see live, easy for tracking to see.

**In code.** Speed and acceleration come from smoothed 10 Hz x/y. The break is the first turn of 30° or more after 1.5 yd, entered above 2.5 yd/s; it was found in 96% of 926 route-drill reps. Break G is the mean acceleration over the 1.0 s centred on the sharpest turn, standardized within drill × Combine year, averaged over a player's route drills (median 7) and pulled toward average when he ran few, so one hot rep cannot make a grade. The definition has been frozen since round 5 of this project.

Figure 1 shows the metric on four players. Kayshon Boutte and Adonai Mitchell cut at 0.62 and 0.60 g in drills; Jake Bobo and Jermaine Burton at 0.53 and 0.55 g. On NFL breaks, the 0.07 g drill gap shrinks to 0.01 g. NFL cuts are softer for everyone (0.42 g), likely because they are run to leverage and timing, not to a cone.

**Is it a trait?** Break G from five route drills predicts Break G from the other five at ρ = 0.28 [0.12, 0.43] (n = 145), which projects to 0.58 for a receiver who runs seven. The three pieces coaches usually name (decel into the break, speed kept through it, re-acceleration out) do not repeat on their own; speed kept through the turn is 0.09 (Figure 3). Break G is unrelated to the official 40 (−0.01) and draft slot (−0.12): it is not a speed score.

![Figure 3. Left: how well each break measurement repeats across two disjoint sets of route drills (95% CI); Break G in red. Right: each measurement's correlation with the official 40.](figures/fig3_trait.png)

## Result: a few more targets per route

**Data and baseline.** The outcome is career regular-season targets per route run, weighted by routes. Of 149 tracked receivers, 116 have an NFL route (median 231); average TPRR is 18.2%. Break G does not predict who reaches the field (t = −0.59). The baseline is log pick, official 40, weight and NGS athleticism score, plus a tight end flag. "Held out" means fitting on two draft classes and forecasting the third.

![Figure 2. A: Break G's TPRR slope inside each draft class and in all three (95% CI), with the share of forecast error removed when that class is held out (red = worse). B: share of wide receivers who beat their held-out forecast, by call (exact 95% binomial intervals); hollow circles split each group by class; right labels give raw route-weighted TPRR against the forecast.](figures/fig2_evidence.png)

The slope is positive in every class but clears zero only in 2023 (Figure 2A). Held out, Break G removes 7.4% of forecast error [−10.0, 17.7]: 10.7% in 2023, −0.8% in 2024, 2.7% in 2025. A class of about 40 has only 0.23 power to detect this slope, so disagreement between classes is expected either way. Wide receivers carry it (2.01 points, t = 2.95); tight ends do not (t = 0.91). Three checks keep the claim small:

| Check | Result | Reading |
|---|---|---|
| Break G × TPRR was 1 of 36 measure × outcome pairs tried | chance the best shuffled pair looks this strong: 0.213 | could be luck of the search |
| Drop Puka Nacua (2023, pick 177, 33.6% TPRR) | slope 1.02 (t = 1.93) | one player carries half the slope |
| Only the 38 who ran the 3-cone | slope −0.94 (t = −0.73) | cannot show it adds to agility times |

Discounting for the search, the planning number is **1.4 points of TPRR per sd, about 7 targets in a 500-route season**.

## The calls, and how often they were right

A wide receiver is **plus** if his 80% Break G range sits above zero and **minus** if it sits below. Receivers with fewer than five route drills and all tight ends get no call. Each forecast comes from a model that never saw his class.

- **Plus (16 WRs):** TPRR 20.0% against an average forecast of 18.6%; 8 beat it (50%, 95% CI 25 to 75). A coin flip.
- **No call (54):** 17.5% against 17.6%; 43% beat it.
- **Minus (8):** 14.1% against 17.4%; only 1 beat it (12%, 95% CI 0 to 53).

The downgrade side looks more useful, but eight players cannot prove it: the interval runs to 53%, and the 2023–24 hits come from classes we could see while building the metric. The 2025 class, with one season, has 2 of 4 plus and 0 of 1 minus receivers beating forecast.

**Draft-room example.** In 2023 A.T. Perry (pick 195, Break G −1.3) and Trey Palmer (pick 191, 1.3) went 4 picks apart. The rule takes Palmer: 15.0% TPRR over 601 routes against Perry's 9.4%, though Palmer also fell short of his 18.0% forecast. Across all 8 minus WRs the same pairing (nearest-pick non-minus WR of his class, median 4 picks apart) goes the rule's way 5 times: better than a coin, not by much.

**Decision card, 2025 class** (rule: every plus and minus WR, the closest neutral to a minus call, the WR with fewest drills, the most-used TE):

| 2025 player (pick) | Route drills | Break G [80% range] | First-season TPRR vs forecast | Call |
|---|---|---|---|---|
| Arian Smith, WR (110) | 7 | 1.7 [0.9, 2.5] | 7% vs 17% (231 routes) | Plus. Note it; do not move him up a round |
| KeAndre Lambert-Smith, WR (158) | 6 | 1.4 [0.5, 2.2] | 17% vs 17% (72 routes) | Plus. Note it; do not move him up a round |
| Jaylin Noel, WR (79) | 7 | 1.2 [0.4, 2.1] | 18% vs 17% (197 routes) | Plus. Note it; do not move him up a round |
| Pat Bryant, WR (74) | 7 | 1.1 [0.3, 1.9] | 17% vs 18% (282 routes) | Plus. Note it; do not move him up a round |
| Isaiah Bond, WR (undrafted) | 7 | −1.3 [−2.1, −0.5] | 13% vs 16% (333 routes) | Minus. Loses a tie with a WR picked near him |
| Jimmy Horn, WR (208) | 7 | −0.8 [−1.6, 0.1] | 19% vs 18% (78 routes) | No call. Grade the tape |
| Dont'e Thornton, WR (108) | 3 | −0.7 [−1.7, 0.3] | 13% vs 19% (228 routes) | No call. Ask for all route drills at his pro day |
| Harold Fannin, TE (67) | 3 | 0.5 [−0.5, 1.5] | 26% vs 16% (417 routes) | No call. Break G is not used for tight ends |

What a receivers coach does with it: a minus call is a reason to ask what else on tape makes him win early in the route; a plus call is a note, not a promotion.

## Not separation

We ran the same code on 1,763 sampled NFL slant, out, hitch, in and corner snaps. A snap counts only if the cut finished at least 0.2 s before the throw (943 snaps, 112 players). Separation over expected (SOE) is separation at the throw minus a model of route, depth, cushion, coverage, alignment, time to throw, down and distance fitted on 2023–24.

![Figure 4. A: Combine Break G per drill family against NFL Break G on the same route family. B: NFL Break G on a snap (fifths) against SOE on that snap (95% CI). C: player SOE against TPRR beyond the baseline; dot size = routes.](figures/fig4_soe.png)

- **Same route family:** drill Break G does not carry to NFL Break G on that route: ρ = −0.07 [−0.17, 0.03], 436 player-route pairs.
- **Same snap:** harder NFL cuts do not buy separation: ρ = 0.02 [−0.05, 0.08].
- **Player level:** one sd of Combine Break G goes with −4.5 inches of SOE [−10.8, 1.8], against a player-to-player spread of 26 inches.

This does not contradict the target result. In this sample SOE barely tracks TPRR beyond the baseline (ρ = 0.06 [−0.13, 0.25]), and adding SOE moves the Break G slope only from 1.93 to 1.88. Whatever Break G picks up, it is not getting open at the throw; timing, quarterback trust and catch-point play are candidates this data cannot separate.

## Limitations

- **A lead, not a confirmation.** The search-adjusted chance is 0.213, the held-out gain interval spans zero, and one player carries half the slope. Targets per route replaced an earlier volume outcome after that result was seen; the definition was frozen, the outcome choice was not.
- **No clean holdout.** All three classes were visible when Break G was defined. The 2025 class has one season.
- **Mechanism unknown.** Break G does not carry to NFL cut force or separation, so "break" describes the drill, not a proven game skill.
- **Outcome noise.** TPRR reflects role, quarterback, scheme and health; it carries no coverage or play-call context.
- **Sample.** NFL tracking is up to 40 sampled plays per player with no defenders or ball, so the drill-to-game nulls are partly a measurement limit.

Reproduce: `uv run python writeup/candidates/round_08/cand_2/analysis.py` (about two minutes, `data/processed` only). It writes every quoted number to `numbers.json`, the sentence quoting each to `claims.json`, and renders this text from `writeup_template.md`; the full board is in `board.csv`.
