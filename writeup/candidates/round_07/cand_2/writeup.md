# Break G: Receivers Who Cut Harder at the Combine Drew Slightly More Targets, Not More Separation

We measured how hard 149 wide receivers and tight ends pushed into the ground through the breaks of their Combine route drills. Among the 116 who ran an NFL route, each sd of this "Break G" came with 1.93 more targets per 100 routes than draft slot and testing forecast (95% CI 0.02 to 3.78), with no extra separation, so it is a tiebreaker at most.

**Why this question.**
1. Route drills are the only Combine drills that copy a route, and 10 Hz tracking now records every break: the brake, the turn and the drive out.
2. Clubs will be sold "suddenness" scores built from those reps and told they mean separation.
3. A receivers coach wants two things: does he get open, and does the quarterback throw to him? We test both against what a club already has (pick, 40, weight, NGS score).

![Figure 1. Left: Break G effect on four NFL outcomes for the same 112 receivers and the same baseline, in player sd of each outcome (95% bootstrap CI). Targets and yards per route lean positive; both separation measures sit at or below zero. Right: TPRR slope inside each draft class and pooled (95% CI), with the share of forecast error removed when that class is held out (red = worse). Only 2023 clears zero.](figures/fig1_hero.png)

## The metric: Break G

**One line a coach can repeat.** Break G is how hard a receiver pushes into the ground in the one second around his cut (braking, turning, driving out), averaged over his route drills. A receiver who rounds the cut and drifts scores low; one who sinks, snaps and drives out scores high. The typical Combine rep is 0.57 g.

**Code definition.** Speed and acceleration come from smoothed 10 Hz x/y. The break is the first turn of 30° or more after 1.5 yd of travel, entered above 2.5 yd/s; breaks were found in 96% of 926 route-drill reps. Break G is the mean acceleration magnitude over the 1.0 s centred on the sharpest turn. Reps are standardized within drill × Combine year and averaged over a player's route drills (median 7), then pulled toward average in proportion to how few drills he ran, so one hot rep cannot make a grade. Units are sd of true Break G. The definition has been frozen since round 5.

![Figure 2. Speed (top) and acceleration (bottom) through every break, aligned on the sharpest turn. Blue: all of a player's Combine route-drill reps; orange: all his sampled NFL slant, out, hitch, in and corner breaks; thick = median. Players chosen by rule: two highest and two lowest Break G with 5+ route drills and 8+ NFL breaks.](figures/fig2_profiles.png)

Figure 2 compares like with like: every drill rep against every NFL break, same window. Kayshon Boutte and Adonai Mitchell brake and drive out harder in drills (0.62 and 0.60 g) than Jake Bobo and Jermaine Burton (0.53 and 0.55 g). On NFL breaks the four run 0.48, 0.48, 0.46 and 0.47 g: the 0.07 g drill gap shrinks to 0.01 g. NFL cuts are softer overall (0.42 vs 0.57 g), likely because they are run to leverage and timing, not a cone.

**Is it a trait?** Break G repeats from one set of five route drills to the other at 0.28 [0.12, 0.43] (n = 145), rising to 0.58 for a receiver who runs seven drills. The three pieces coaches usually name (decel into the break, speed kept through it, re-acceleration out) do not repeat on their own; speed kept at the apex is 0.09 (Figure 3). Break G is unrelated to the official 40 (−0.01) and draft slot (−0.12): it is not a speed score.

![Figure 3. Left: split-half reliability of nine break measurements across disjoint sets of route drills (95% CI); Break G in red. Right: each measurement's correlation with the official 40.](figures/fig3_trait.png)

## Result: a few more targets per route

**Data and baseline.** The outcome is career regular-season targets per route run (TPRR), weighted by routes. Of 149 tracked receivers, 116 have an NFL route (median 231); the average TPRR is 18.2%. The 33 without one are not explained by Break G (t = −0.59). The no-tracking baseline is log draft pick, official 40, weight and NGS athleticism score (standardized within year × position), plus a TE flag. "Held out" means fitting on two draft classes and forecasting the third; the R² change is the share of that forecast's squared error removed by adding Break G.

| Held-out class | n | TPRR slope in the class, pct pts [95% CI] | Held-out R² change | Same, if the 36-pair screen is re-run in the fold |
|---|---|---|---|---|
| 2023 | 42 | 2.67 [0.06, 5.91] | +10.7% | −6.5% |
| 2024 | 40 | 1.08 [−2.24, 3.24] | −0.8% | −31.2% |
| 2025 (one season) | 34 | 0.68 [−2.26, 5.17] | +2.7% | −57.6% |
| **Pooled** | **116** | **1.93 [0.02, 3.78]** | **+7.4% [−10.0, 17.7]** | **−31.8% (mean)** |

**What to believe.** Every class points the same way, but only 2023 clears zero and 2024 held out gets slightly worse. That is what small classes do: one class of about 40 has power 0.23 to detect the pooled slope, so disagreement between classes is expected even if the effect is real. Three checks keep the claim small:

- **Selection.** Break G × TPRR was one of 36 drill-measure × outcome pairs examined. The chance that the best of 36 shuffled pairs looks this strong is 0.213, and Break G is only the 4th-strongest pair. Re-running the whole screen inside each fold fails: the pair it picks from two classes changes R² by −31.8% on average in the third (last column). The search does not generalize; Break G survives only because its definition was fixed before this outcome was chosen.
- **Alternatives.** A ridge model with all nine break measures, tuned inside each fold, changes held-out R² by −4.8%, so the force summary beats its parts. Adding 3-cone, shuttle and 10-yd split to the baseline (missing set to average) leaves the slope at 2.00 (t = 3.44). On the 38 who actually ran the 3-cone it is −0.94 (t = −0.73), so we cannot show it adds to agility times there.
- **Influence.** Without Puka Nacua the slope is 1.02 (t = 1.93); counting each player once instead of weighting by routes it is 1.41 (t = 1.75).

After discounting for the 36-pair search (factor 0.72), the planning number is **1.4 points of TPRR per sd, about 7 targets in a 500-route season**. On the 1,930 randomly sampled tracked snaps, Break G predicts the share of routes targeted at 3.42 points per sd (t = 2.70, n = 103); these are a subset of the season targets, so it checks the data, not the effect.

## Not separation, and why that does not contradict the targets

We ran the same break code on 1,763 NFL slant, out, hitch, in and corner snaps. A snap is graded only if the cut finished at least 0.2 s before the throw, since a later cut cannot create separation at the throw; that drops 47% of snaps and leaves 943 (112 players). Separation over expected (SOE) is separation at the throw minus a context model (route, depth, cushion, coverage, alignment, time to throw, down and distance) fitted on 2023–24 snaps.

![Figure 4. A: Combine Break G per drill family vs NFL Break G on the same route family. B: NFL Break G on a snap (fifths) vs SOE on that snap (95% CI). C: player SOE vs TPRR beyond the no-tracking baseline; dot size = routes.](figures/fig4_soe.png)

- **Drill to the same NFL route.** Combine Break G does not carry into the same route family: −0.07 [−0.17, 0.03] over 436 player-family pairs (Figure 4A).
- **Same snap.** Harder NFL cuts do not buy separation on that snap: 0.02 [−0.05, 0.08] (Figure 4B).
- **Player level.** One sd of Combine Break G is worth −4.5 inches of SOE [−10.8, 1.8] given the baseline. The upper bound is 1.8 inches per sd, against a player-to-player sd of 26 inches.

**Reconciling Figure 1.** The positive target slope and the flat-to-negative SOE slope are not in conflict. SOE barely relates to TPRR in this sample: 0.06 [−0.13, 0.25], n = 112 (Figure 4C). Adding player SOE to the TPRR model moves the Break G slope only from 1.93 to 1.88. So whatever Break G tracks, it does not earn targets by getting open at the throw. Timing, quarterback trust and catch-point play are candidates this data cannot separate.

## The board and how to use it

A wide receiver is **plus** if his 80% Break G interval sits above zero and **minus** if it sits below. Tight ends get no call (t = 0.91 within TEs, n = 34). The forecast is the baseline fitted without the player's class. Top six and bottom four WRs with 100+ routes:

| WR (class, pick) | Break G [80%] | Routes | NFL TPRR | Forecast | Call |
|---|---|---|---|---|---|
| Kayshon Boutte (2023, 187) | 1.8 [1.0, 2.6] | 873 | 13.9% | 18.7% | plus |
| Adonai Mitchell (2024, 52) | 1.7 [0.9, 2.5] | 486 | 26.5% | 17.9% | plus |
| Arian Smith (2025, 110) | 1.7 [0.9, 2.5] | 231 | 6.9% | 16.7% | plus |
| Tre Tucker (2023, 100) | 1.7 [0.8, 2.5] | 1,388 | 14.9% | 17.9% | plus |
| Puka Nacua (2023, 177) | 1.6 [0.8, 2.4] | 1,287 | 33.6% | 19.1% | plus |
| Trey Palmer (2023, 191) | 1.3 [0.5, 2.2] | 601 | 15.0% | 18.0% | plus |
| Johnny Wilson (2024, 185) | −1.2 [−2.1, −0.2] | 138 | 10.9% | 14.3% | minus |
| A.T. Perry (2023, 195) | −1.3 [−2.1, −0.4] | 192 | 9.4% | 16.2% | minus |
| Isaiah Bond (2025, undrafted) | −1.3 [−2.1, −0.5] | 333 | 13.2% | 15.9% | minus |
| Jake Bobo (2023, undrafted) | −1.4 [−2.2, −0.5] | 306 | 14.4% | 19.1% | minus |

Of the 16 plus WRs, 50% beat their forecast, against 42% of 57 neutral and 11% of 9 minus (route-weighted gaps 1.4, −0.1 and −3.3 points). The plus list is not a hit list: Boutte, Tucker, Palmer and Arian Smith all sit below forecast. The usable side is the downgrade: **minus receivers rarely out-earn their slot**.

| Situation | Action | Evidence here |
|---|---|---|
| Two WRs within a round, one with a minus call | Break the tie against him | 11% of minus WRs beat forecast |
| A plus call | Note it; do not move him up | 50% beat forecast, a coin flip |
| Fewer than 5 route drills | No call; ask for the full route set at his pro day | reliability 0.17 for one drill, 0.58 for seven |
| Tight end | Ignore Break G | t = 0.91 |
| "Suddenness" sold as separation | Ask for the SOE link; grade only cuts done 0.2 s before the throw | 0.02 per snap here |

**Draft-room example.** In 2023 A.T. Perry (pick 195, Break G −1.3) and Trey Palmer (pick 191, 1.3) went 4 picks apart. The rule takes Palmer: 15.0% TPRR over 601 routes against Perry's 9.4%, though Palmer also fell short of his 18.0% forecast. Across all 9 minus WRs, 6 drew fewer targets per route than the nearest-pick non-minus WR of their class (median 4 picks apart). 2025 plus WRs to watch: Arian Smith, KeAndre Lambert-Smith, Jaylin Noel, Pat Bryant.

## Limitations

- **A lead, not a confirmation.** The 36-pair search p is 0.213, the held-out gain interval spans zero, 2024 and 2025 include zero, and one player halves the slope. TPRR replaced round 5's targets per season after that result was seen, a forking path; the Break G definition was frozen, the outcome was not.
- **Mechanism unknown.** Break G does not carry to NFL cut force or separation, so "break" describes the drill, not a proven game skill.
- **Outcome.** TPRR reflects role, quarterback, scheme and health. The 2025 class has one season.
- **Sample.** NFL tracking is up to 40 plays per player, with no defenders or ball; SOE comes from the provided separation, so the drill-to-game nulls are partly a measurement limit.
- **Units.** One sd of true Break G is only 0.021 g; most rep-to-rep spread is noise.

Reproduce: `uv run python writeup/candidates/round_07/cand_2/analysis.py` (about two minutes, `data/processed` only). It writes every quoted number to `numbers.json` with a claim-to-section map and renders this text from `writeup_template.md`; the full board is in `board.csv`.
