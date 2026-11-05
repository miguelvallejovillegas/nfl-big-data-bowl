# Break G: The Combine Cut That Does Not Get Receivers Open

Break G measures how hard a receiver brakes, turns and drives out of the cut in his Combine route drills. It is a real drill trait: it repeats across different drills. It does not carry to the same route on Sundays, and it does not buy separation. Across 116 NFL receivers and tight ends, one sd of Break G is worth −0.6 inches of season separation over expected (90% interval −2.6 to +1.2). That rules out any effect as large as what a receiver's weight already tells you.

## Why test this

Every Combine route-drill rep is now tracked at 10 Hz, so clubs will be sold "suddenness" scores built from those reps, with the claim that sudden cutters get open. Separation is worth testing because it is a real, repeatable NFL skill here. A receiver's season separation over expected (SOE) correlates 0.44 [0.23, 0.62] with his next season's SOE (n = 53 pairs with 100+ routes in both seasons). The question: does the cut a receiver shows on a cone in Indianapolis predict the separation he creates on the matching NFL route, beyond pick, 40, weight and NGS athleticism score?

## What a club should do with this

- **Do not move a receiver on a route-drill break grade.** No version tested here (Break G or its three coach pieces) moves season separation by more than a few inches, and none beats the official 40 or weight.
- **Grade the break on tape, not on the route tree.** In this data the drill cut and the NFL cut are different movements. NFL cuts are softer for everyone (median 0.42 g against 0.57 g in drills), because they are run to leverage and timing, not to a cone.
- **Combine protocol ask.** If the league wants a break measure that transfers, it needs (1) route drills run against a scripted defender, or at least with throw timing, so the cut has something to beat, and (2) two reps of each route. Today 75–83% of position drills have a single rep, and the three pieces coaches name do not repeat across drills (reliability 0.12 or less).

## The metric

**For a coach.** Break G is how hard a receiver pushes into the ground in the one second around his cut. A receiver who rounds the cut and drifts scores low. One who sinks, snaps and drives out scores high.

**On the field.** A typical Combine cut averages 0.57 g. One sd of true Break G is 0.021 g, about 0.2 yd/s more velocity change through the cut.

**In code.** Speed and acceleration come from smoothed 10 Hz x/y. The break is the first turn of 30° or more after 1.5 yd of travel, entered above 2.5 yd/s. It was found in 96% of 926 route-drill reps. Break G is the mean acceleration over the 1.0 s centred on the sharpest point of the turn. It is standardized within drill × Combine year and averaged over a player's route drills (median 7). It is then shrunk toward average by drill count. The identical code runs on NFL snaps of the same five route families: slant, out, hitch (with curl and comeback), in and corner. The definition was fixed in earlier work. The ±3-inch equivalence margin and the season-SOE outcome were set before the test below was run.

![Figure 1. The hardest and softest Combine cutters among wide receivers with 5+ route drills, 200+ NFL routes and 8+ graded NFL breaks. Left: every drill rep and every sampled NFL break, ±1 s around the cut, seen from above (stem up, cut at the cross, turn mirrored right), coloured by acceleration. Right: acceleration through the cut on a shared time axis; thin lines are single reps or snaps, bold lines are medians.](figures/fig1_hero.png)

Figure 1 shows the pattern on two players. Adonai Mitchell (2024, pick 52) cut at 0.60 g in drills and Jake Bobo (2023, undrafted) at 0.53 g. On NFL breaks the 0.07 g drill gap shrinks to 0.02 g, and their season SOE is nearly identical (−0.6 and −0.8 inches).

## The chain: three links, tested in order

For a drill to predict separation, three things must hold (Figure 2).

![Figure 2. The three links. 1: Break G from one half of the route drills against the other half (disjoint drills, 145 players). 2: Combine Break G against the same player's NFL Break G on slant/out/hitch/in/corner snaps; the ceiling is the largest correlation the two measures' reliabilities allow. 3: within each player, SOE on his hardest-to-softest fifths of NFL cuts, relative to his own average; bars are 90% intervals that resample whole players.](figures/fig2_chain.png)

1. **The drill trait holds.** Break G from five route drills predicts Break G from the other five at ρ = 0.28 [0.16, 0.40] (n = 145), which projects to 0.58 for a receiver who runs seven. It is not a speed score: its correlation with the official 40 is −0.01, and with draft slot −0.12. The pieces coaches usually name do not repeat. Decel into the break is 0.12, speed through it −0.04, re-acceleration 0.05.
2. **It does not carry to the same route on Sundays.** NFL Break G is weakly repeatable (odd vs even snaps 0.23, about 0.37 over a player's median 16 breaks). If drill and game cuts were the same skill, the correlation could reach about 0.46. It is −0.11 [−0.25, 0.05] across 115 players. Matched by route family it is −0.07 [−0.15, 0.01] over 436 player-route pairs. No single family is positive by more than 0.05 (corner).
3. **A harder NFL cut does not buy separation on that snap.** Within a player, Break G on a snap against SOE at the throw gives ρ = 0.01 [−0.05, 0.06] (943 snaps where the cut finished at least 0.2 s before the throw).

The snap-level SOE is separation at the throw minus a model of route, depth, cushion, coverage, alignment, time to throw, down and distance, fit on 2023–24 snaps. It agrees with the full-season version used below at ρ = 0.34 [0.18, 0.50] (n = 112), so the two outcomes measure the same thing.

## Season separation: the bound

**Data and baseline.** The outcome is full-season NGS separation at the throw, adjusted for cushion, alignment (slot, inline), tight end and season, and route-weighted over each player's career. It covers 213 player-seasons and 116 players with a median of 231 routes. Each yard of cushion is worth 0.42 yd of separation, which is why the adjustment matters. The model is the baseline (log pick, official 40, weight, NGS athleticism, TE flag) plus Break G. "Held out" means fitting on two draft classes and forecasting the third.

![Figure 3. Inches of season SOE per 1 sd of each measure, with 90% bootstrap intervals. Top: route-drill tracking measures beyond the full baseline. Bottom: official Combine numbers beyond pick and position, for players who ran them. The shaded band is the ±3-inch equivalence margin.](figures/fig3_bound.png)

| Test | n | Estimate [90% interval] | Null it tests | p |
|---|---|---|---|---|
| Drill trait (split-half ρ) | 145 players | 0.28 [0.16, 0.40] | no player consistency (shuffle) | 0.001 |
| Same route on Sundays (ρ) | 115 players | −0.11 [−0.25, 0.05] | no link (shuffle) | 0.24 |
| Snap: harder cut → SOE (within player ρ) | 943 snaps | 0.01 [−0.05, 0.06] | no link | interval spans 0 |
| **Season SOE, inches per sd** | 116 players | **−0.6 [−2.6, +1.2]** | effect = 0 | 0.57 |
| Same, equivalence | 116 players | upper bound +1.2 in | effect ≥ 3 in either way | 0.019 |
| Best of four break measures (decel) | 116 players | −0.9 [−3.1, +1.2] | none of four links (max-t shuffle within class) | 0.94 |
| Held-out forecast error removed | 3 classes | 1.3% [−0.9, 3.6] | no gain | interval spans 0 |

The one-sided upper bound is +1.2 inches per sd. That is below the official 40's point estimate of +1.6 [0.1, 3.3] and well short of the size of weight's −3.2 [−5.2, −1.8]. The ±3-inch margin is about half (52%) of the true player-to-player sd of season SOE (5.8 inches). With 116 players the minimum effect detectable at 80% power is 3.2 inches, so smaller effects cannot be excluded. Inside each class the slope is 0.0 (2023, n = 42), −0.8 (2024, n = 40) and 0.4 (2025, n = 34) inches. Wide receivers alone give −0.2 inches (n = 82), players with 100+ routes −0.6 (n = 84), equal weighting −2.8. No version is positive.

## Named players

The rule takes the three highest and three lowest Break G wide receivers with 5+ route drills and 200+ NFL routes (46 qualify). "vs forecast" is season SOE minus the held-out baseline forecast.

| Player (class, pick) | Break G [80% range] | NFL Break G (z) | Season SOE vs forecast (routes) | Read |
|---|---|---|---|---|
| Kayshon Boutte (2023, 187) | 1.8 [1.0, 2.6] | 0.1 | −4.8 in (873) | Hardest drill cutter, below forecast: the drill grade would have misled |
| Adonai Mitchell (2024, 52) | 1.7 [0.9, 2.5] | 0.3 | +0.9 in (486) | On forecast; the drill told you nothing extra |
| Arian Smith (2025, 110) | 1.7 [0.9, 2.5] | 0.2 | +11.8 in (231) | Well above forecast with a 4.36 40; credit the speed, not the cut |
| Jake Bobo (2023, UDFA) | −1.4 [−2.2, −0.5] | 0.1 | −6.4 in (306) | Soft cutter, below forecast: the drill's one correct downgrade |
| Isaiah Bond (2025, UDFA) | −1.3 [−2.1, −0.5] | −0.5 | +1.0 in (333) | Softest in drills and games, still on forecast: film first |
| Jalin Hyatt (2023, 73) | −0.9 [−1.7, 0.0] | 0.1 | +0.9 in (679) | Soft cutter with a 4.40 40, on forecast; do not mark him down |

Two of three hard cutters and two of three soft cutters beat their forecast: a coin flip both ways. Boutte and Bobo show the main pattern: their NFL break force (0.1 z each) sits mid-league whatever they did on the cone.

## Limitations

- **Not a blind test.** The definition and the margin were fixed before the season-SOE test, but an earlier look in this project had already found no link between Break G and snap-level separation. Read this as a confirmation with a stated margin, not a discovery.
- **Power.** The test rules out effects of 3 inches per sd or more. It cannot rule out 1 inch. A 1-inch effect would not change a draft board.
- **Game tracking is a sample.** There are up to 40 plays per player with no defenders or ball, so NFL break force on a snap reflects the route call and leverage as much as the player. Its reliability (0.23 odd vs even) caps every drill-to-game correlation. Separation comes from the provided field, not from our own defender geometry.
- **Outcome.** Season SOE is adjusted for cushion and alignment, not for coverage or quarterback. The 2025 class has one season.
- **Drill protocol.** Route drills are run to cones, mostly once each, and Combine drill medians shift between years. Within-drill × year standardization handles the shift but not the single-rep noise.

Reproduce: `uv run python writeup/candidates/round_09/cand_2/analysis.py` (about one minute, `data/processed` only). It writes every quoted number to `numbers.json`, the named-player table to `board.csv` and every WR/TE to `board_full.csv`.
