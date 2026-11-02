# Launch Carry: Whether a Pass Rusher Shows His Combine Burst on Sunday Mostly Says Where He Lines Up

**Lede.** We measured, for 86 defensive linemen, how much of their Combine acceleration they show in their first steps of NFL pass rushes ("Launch Carry"). Carry looks like it predicts pressure beyond draft slot, but that is because edge rushers run free and interior rushers hit a guard. Once alignment is held fixed it adds nothing (−0.27 pressure-rate points per sd in unseen draft classes, 90% interval [−0.95, 0.43], n=69). The Combine side alone still leans positive (+0.49 [0.10, 0.89]), and almost none of that runs through the NFL launch.

- **Scout:** do not mark a rusher down because his NFL get-off "looks slower than his testing". That gap mostly reflects where he lines up.
- **Coach:** read the launch against rushers at the same alignment, never against his own Combine tape.
- **Combine:** the drill battery already records a usable launch. It is a whole-workout number, not a single 40.

**Why now.** Every team now has its own players' tracking. The obvious next step is to compare each prospect's Combine movement with his Sunday movement, the "does he play to his testing" question. This writeup tests that comparison on the one group where it has a chance and shows where it fails.

## 1. The metric: same movement, same speeds, two venues

A rush starts with the first two or three yards, from 1 to 5 yd/s. For each speed band (1–2, 2–3, 3–4, 4–5 yd/s) we take a player's hardest forward acceleration (90th percentile of straight-line frames):

- **Combine launch envelope:** across all of his Combine drills (6306 tracked attempts, 52 drills), z-scored within his Combine year because drill tracking shifts between years.
- **NFL launch envelope:** across the first 3 s after the snap of his *random* sampled NFL snaps (median 20 per lineman). The top-EPA plays are left out so that good outcomes do not select fast launches.
- **Launch Carry** = NFL ÷ Combine at the same speed, median over bands. A Carry of 1 means he shows his full Combine launch on Sundays. The median is 0.80 (10th–90th percentile 0.59–1.01). Split by odd and even snaps and drills, its reliability is 0.80, so Carry is a stable trait of the player.

![Figure 1. Launch envelopes for three linemen: Combine drills (black circles) and random NFL rushes (coloured squares; blue edge, orange interior). The grey area is the gap Launch Carry summarizes. The box gives Carry, its rank within his draft class, and career pressure rate against the held-out pick + role forecast. n per panel: one player.](figures/fig1_launch_envelopes.png)

## 2. Why defensive line: the only group where the launch carries at all

We ran the same metric on all five groups. The rule was fixed before any production model was fit: choose the highest lower bound on the drill-to-NFL rank correlation among groups with an individual production stat. Defensive line wins clearly. Its pooled correlation is 0.40 [0.22, 0.54], against 0.09 for OL and 0.06 for WR. DBs show nothing (reliability ceiling 0.04: backpedal starts do not repeat).

The DL correlation, however, is mostly alignment. Edge rushers' NFL envelope sits 1.6 sd above interior rushers' (Launch Carry 0.92 vs 0.66; Carry vs share of snaps at edge ρ 0.73), because an interior rusher meets a guard within a yard. With weight and alignment held fixed, translation falls to 0.16 [−0.02, 0.31], about a quarter of the 0.67 the two sides' reliabilities would allow. Within interior rushers it is 0.33 [0.13, 0.52]; within edge rushers it is 0.17 [−0.05, 0.38].

![Figure 2. A: Combine vs NFL launch envelope for 86 DL, edge (blue circles, n=42) and interior (orange triangles, n=44); lines are within-role fits. B: the same drill-to-NFL correlation in five groups, pooled (black) and with weight and alignment held fixed (blue). Bars are 90% class-stratified bootstrap intervals; "ceiling" is the most the two sides' split-half reliabilities allow.](figures/fig2_translation_screen.png)

## 3. Does translation explain pressure beyond the pick?

**Outcome.** Career pressure rate: pressures per 100 pass-rush snaps, for linemen with at least 100 rushes. That covers 69 rushers with a median of 297 rushes. A pressured random snap costs the offense 0.53 EPA [0.38, 0.68] (135 pressured of 1681 random DL snaps). One pressure-rate point therefore matters.

**Test.** Fit on two draft classes and predict the third, rotating through 2023, 2024 and 2025. Each metric is z-scored within class. The comparison baseline is fixed in advance as **log draft pick + share of snaps at edge**. It forecasts unseen classes with R² 0.19, against 0.08 for pick alone. A full baseline (adding weight, official 40 and NGS athleticism score) and a ridge-shrunk version are shown as checks. Intervals come from resampling players within class, with all three folds refit in every draw (1,000 draws). The **placebo** shuffles each metric within class 1,000 times; 90% of placebo slopes fall in [−0.48, 0.45].

| Baseline (held-out R²) | Launch Carry, pts/sd [90%] | Combine launch envelope, pts/sd [90%] | Combine envelope R² change [90%] |
|---|---|---|---|
| Pick only (0.08) | +0.55 [0.10, 0.97] | +0.71 [0.27, 1.16] | 0.005 [−0.173, 0.098] |
| **Pick + role (0.19)** | **−0.27 [−0.95, 0.43]** | **+0.49 [0.10, 0.89]** | −0.031 [−0.192, 0.060] |
| Ridge, full set (0.19) | −0.16 [−0.94, 0.60] | +0.30 [−0.25, 0.69] | −0.026 [−0.104, 0.017] |

*Table 1. Slopes in pressure-rate points per sd, on unseen draft classes, n=69. The full unshrunk baseline (R² 0.22) gives the same answers as ridge: Carry −0.16, Combine envelope +0.30.*

Three readings:

1. **The trap.** Against pick alone, Carry looks like a find (+0.55, interval above zero). Add alignment and it flips to −0.27, inside the placebo band (shuffle p 0.36). Carry was a role detector all along. In-class slopes run +0.41, +0.10 and −0.29 for 2023 to 2025.
2. **Translation proper adds nothing.** The NFL envelope minus what the Combine envelope, weight and role predict ("translation residual") gives +0.39 [−0.11, 0.88], with an R² change of 0.005.
3. **The Combine side leans positive but is not proven.** It gives +0.49 per sd, roughly 2.0 pressures per 400 rushes, worth about 1.0 EPA. The slope interval clears zero, but the shuffle p is 0.07, the R² change interval spans zero, and the effect comes mostly from 2024 (in-class +1.23; 2023 −0.09, 2025 +0.28).

**Negative controls.** The same Carry in groups where no link is expected stays flat, in sd of each group's production stat: OL pressure allowed −0.07 [−0.22, 0.10] (n=66), WR yards per route +0.04 [−0.12, 0.25] (n=62). For scale, DL Carry is −0.27 pts, which is −0.10 sd of pressure rate.

## 4. Where the Combine effect goes

If Combine burst produced pressure *by* showing up in the NFL launch, the effect should pass through the NFL envelope. We split it on the same 69 rushers, in-sample, holding pick, alignment and weight fixed. The total effect is +0.40 [−0.05, 0.86]. Only +0.03 [−0.05, 0.16] (about 7%) runs through the NFL launch envelope, because the Combine envelope barely moves the NFL one once role and weight are fixed (0.04 sd per sd). The NFL launch itself leans toward pressure (+0.74 [−0.24, 1.75]). Each half leans positive on its own, and the two are nearly unrelated.

**Football reading.** The Combine records the burst in clean space. The NFL first two yards record alignment, the set of the tackle or guard, and the call. Whatever the Combine burst buys later in the rush (the second move, the finish) is not visible in the launch.

![Figure 3. A: held-out slopes beyond pick + role (dot, 90% interval; grey tick = full baseline). The grey band holds 90% of within-class shuffled Launch Carry slopes. B: Combine launch effect on pressure split into the part through the NFL launch envelope and the rest (in-sample, pick + role + weight held fixed, 90% class-stratified bootstrap). n=69 rushers.](figures/fig3_heldout_mediation.png)

## 5. Players

| Player | Class, pick | Role | Launch Carry (class rank) | Combine envelope rank | Pressure rate / forecast | Hardest NFL launch (clip) |
|---|---|---|---|---|---|---|
| Nic Scourton | 2025, 51 | edge | 1.22 (1/21) | 20/21 | 7.7 / 8.8 | 2025 wk 18, Q2 14:05 |
| McKinnley Jackson | 2024, 97 | interior | 0.50 (23/23) | 18/23 | 5.1 / 5.7 | 2024 wk 18, Q1 00:06 |
| Cedric Johnson | 2024, 214 | edge | 0.84 (5/23) | 6/23 | 13.0 / 6.1 | 2025 wk 17, Q4 13:29 |
| Tyrus Wheat | 2023, UDFA | edge | 0.81 (13/25) | 1/25 | 6.2 / 6.9 | 2025 wk 12, Q3 07:32 |

*Table 2. Forecast = held-out pick + role. Clip = the random snap with his hardest launch, located by tracking for film review (not yet watched).*

- **Nic Scourton** has the highest Carry in his class (1.22): he shows more than his Combine launch. Yet he ranks near the bottom of his class on the Combine envelope and sits below forecast (7.7 vs 8.8). High Carry here means a modest Combine number, not a better rusher.
- **McKinnley Jackson** (interior) shows half his Combine launch (Carry 0.50). This is the typical interior pattern, since he is engaged at 2–3 yd/s, and he is close to forecast (5.1 vs 5.7).
- **Cedric Johnson**, a pick-214 edge rusher in his class's top third on the Combine envelope, is the case for the Combine side: 13.0 vs a 6.1 forecast. His Carry (0.84) is ordinary.
- **Tyrus Wheat** has the top Combine envelope in his class and is the miss: 6.2 vs 6.9.

![Figure 4. Every rusher: Combine launch envelope percentile within class (x) against career pressure rate minus the held-out pick + role forecast (y). Colour is Launch Carry (cividis). Dashed line: top-third cut. n=69.](figures/fig4_board.png)

## 6. Monday action

- **Fixed rule for film, not the board:** flag any DL in the top third of his class on the Combine launch envelope. That is about 13 names per class, 50% of them picked after 100 (median pick 108). In back-test, 12 of 24 beat the held-out forecast against 20 of 45 for the rest, an average edge of +0.7 pts [−0.2, 1.8]. A miss costs one film session, so the rule is cheap. The edge is not proven, and the flag should break ties, not move a player up the board.
- **Retire "plays slower than he tests"** as a negative note for pass rushers. In this sample it measures alignment, and it has no held-out value.

## Limitations

- **Sample.** 69 rushers in three classes. The NFL side uses about 20 random sampled snaps per player, with no blockers and no ball, so the launch cannot tell a stunt or a read-step from a slow get-off.
- **Combine envelope.** It pools drills of different kinds. Its split-half reliability is 0.49, and it is year-standardized but not drill-mix standardized. The positive lean depends on the 2024 class.
- **Role** is a share of sampled snaps at edge. That is a coarse proxy that also absorbs scheme.
- **Outcome.** Career pressure rate is concurrent with the sampled snaps, and 2025 rookies have one season.
- **Multiplicity.** Four metrics against four baselines. Only the pick + role row was pre-set as the headline, and nothing here survives as a board-ready signal.
- **Film.** The clips in Table 2 are located but not yet reviewed.

## Appendix: definitions and reproducibility

- Forward acceleration = centred speed difference over 0.2 s. Frames count as straight if heading changes by less than 20° over 0.4 s. Each band needs at least 5 frames, and each player at least 2 bands.
- Role = share of random sampled snaps lined up at EDGE or OLB.
- EPA filter: random DL snaps with EPA recorded; pressured means `time_to_pressure` is present or a sack is credited. (`pressure_allowed` is a blocker column.)
- Translation residual = NFL envelope regressed on Combine envelope, weight and role, then z-scored within class.
- `analysis.py` rebuilds every number, figure and this text from `data/processed` in about four minutes, and writes `numbers.json` and `board.csv`.
