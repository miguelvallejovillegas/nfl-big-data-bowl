# Hoop G: A Tiebreak Lead for Pass Rushers Who Skip the 3-Cone

**Two in three defensive linemen at the 2023–25 Combines skipped the 3-cone; almost all of them ran Run-the-Hoop. From that lap's tracking we read Hoop G, how hard a lineman pushes sideways through the curve, and score it against a forecast built from pick, weight, role, NGS score and official times. On draft classes the forecast never saw, one standard deviation of Hoop G came with 0.43 more pressures per 100 rushes, about 1.9 a season, and the sign held in all 3 classes. The edge is small, Hoop G was the best of 14 drills we screened, and it leaves no trace in NFL rush tracking. Use it to break ties, not to move boards.**

## Why it matters

A pressure is what a rusher is paid for: on 1681 sampled NFL pass plays of our rookies, the offense lost 0.53 more expected points on pressured snaps (90% interval 0.37 to 0.68).

The 3-cone makes a big man stop, plant and bend low around a cone, the hip bend he needs to dip under a tackle's hands. But only 43 of 118 linemen in the 2023–25 classes ran an official 3-cone, and even it is unsettled: on a class it was not fit on, its link to pressure is ρ = 0.36 (95% interval −0.22 to 0.67), slightly negative in 2023 (−0.09) and 2024 (−0.07) and carried by 8 rushers in 2025. Another 69 linemen ran Run-the-Hoop, a figure-eight around two hoops, and no 3-cone.

**Hoop G is how hard a lineman can push sideways while turning the hoop curve, compared with his Combine class.** Technically: the peak half-second of sideways acceleration in the 10 Hz Combine tracking, standardized within year.

![Figure 1. Seven rushers: drill path, hardest sampled NFL rush, and pressure rate minus the out-of-class forecast with 80% intervals and play counts.](figures/fig1_hero.png)

## One test, held out by class

**The bar.** Every rusher with 30+ NFL rushes gets a no-tracking forecast of his pressure rate (pick, weight, EDGE share, NGS score, official 10-yard split and 3-cone) fit on the other two draft classes. It ranks rushers in every held-out class (Figure 2B: calibration slopes 0.89, 2.01 and 0.63), though 2025 rookies as a group fell to 0.75 of their forecast.

![Figure 2. Out-of-class forecast against actual pressure rate with a binned calibration line and the 2025 plus calls marked (A); calibration slope within each held-out class (B).](figures/fig2_calibration.png)

**The test.** Within each class, does Hoop G predict pressure *beyond* that forecast? A count model starts from the forecast, adds a class level and one Hoop G slope. The rule is frozen in `frozen_rule.json` (SHA-256 `4c2930111941`) and will score the 2026 class after the 2027 season unchanged.

![Figure 3. Held-out slope of pressure over forecast (A) and of 1.5-second depth over forecast (B) on Hoop G, by class and pooled, with the shuffle null.](figures/fig3_heldout.png)

Letting the data speak (weak prior), the slope is 0.85 in 2023, 1.20 in 2024 and 0.42 in 2025 pressures per 100 rushes per sd, consistent across classes (heterogeneity p = 0.58); only 2025 spans zero alone. Pooled it is 0.91. Because Hoop G was picked from 14 drills, we quote the slope under a skeptical prior sized to the typical screened effect: **0.43 (90% interval 0.09 to 0.77), 1.9 pressures in a 450-rush season, worth about 1.0 expected point.** Freeing the forecast's scale in each class leaves it at 0.43. Shuffling Hoop G within class gives p = 0.002 (Figure 3A), **but adjusted for the 14-drill search, p = 0.12**.

As calls: a plus call needs the 80% interval of a player's Hoop G above zero; a hit means his pressure rate clearly beat his forecast.

| Group, 2023–25 (30+ NFL rushes) | Rushers (2025) | Pressure rate (forecast) | Beat forecast [exact 90%] |
|---|---|---|---|
| Plus call, EDGE | 7 (1) | 10.0% (7.8%) | 4 of 7 [23–87%] |
| Plus call, interior | 2 (1) | 8.2% (8.0%) | 0 of 2 [0–78%] |
| No call, EDGE | 32 (10) | 8.4% (8.2%) | 6 of 32 [8–34%] |
| No call, interior | 31 (13) | 6.9% (7.2%) | 3 of 31 [3–23%] |
| Minus flag, interior (film check) | 6 (2) | 4.7% (6.9%) | 0 of 6 [0–39%] |

## Does it show up in NFL tracking?

Pressure on a player's 20 sampled rushes is noisy (split-half reliability 0.07), so we built a steadier outcome from the same 10 Hz game frames: **depth upfield 1.5 seconds after the snap**, on random NFL pass rushes. It is repeatable (reliability 0.85, 86 linemen) and tracks season pressure after weight and role (ρ = 0.33, 90% interval 0.15 to 0.49, 85 linemen).

Hoop G does not move it. With the same held-out design, one sd of Hoop G is worth 0.5 inches of depth (90% interval −0.9 to 1.9; shuffle p = 0.22; Figure 3B). The direction-change rate after the snap is unreliable (0.22) and unrelated (ρ = −0.04, 78 rushers); matched-speed turning and outside-rush share were null too (Appendix).

**So Hoop G is a correlate without a visible mechanism.** It may mark a rusher who keeps speed through contact, which we cannot test without blockers in the data; we treat it as a tiebreak only.

## When the call is wrong

| Player (class, pick, lb, team) | Call | Pressure vs forecast (rushes) | 2+ yd sideways by 1.5 s (random sampled) | Film cut and cue |
|---|---|---|---|---|
| **Cedric Johnson** (2024, 214, 260, CIN) | Plus | 13.0% vs 7.2% (216) | 20% (of 20) | Pull: wide rushes from a 7/9 technique. Cue: does he dip under the tackle's hands and flatten to the quarterback within two steps? |
| **Adetomiwa Adebawore** (2023, 110, 282, IND) | Plus | 8.6% vs 8.4% (511) | 15% (of 20) | Pull: 3-technique rushes and his loops on stunts. Cue: the bend shows when he loops, but does he finish, or stall once the guard re-anchors? |
| **Jordan Phillips** (2025, 143, 312, MIA) | Minus flag | 1.9% vs 6.6% (215) | 40% (of 20) | Pull: rushes where the play moves away and his stunt loops. Cue: does he redirect at all, or only push the pocket? |
| **Yahya Black** (2025, 164, 336, PIT) | Minus flag | 5.8% vs 5.2% (189) | 0% (of 19) | Pull: odd-front nose snaps in Pittsburgh's 3-4. Cue: is he asked to two-gap first? A flag on a two-gap nose is a role question, not a skill read. |

**Johnson** is the case the tiebreak is for: a day-3 EDGE who beat his forecast. **Adebawore** is the miss: an interior rusher who met his forecast, no more. **Phillips and Black** carry the same minus flag with opposite results, and scheme is the likely reason. Miami moved Phillips 2+ yards sideways in the first 1.5 seconds, a stunt or slant, on 40% of his sampled rushes (interior median 15%), where weak sideways force should hurt; he sits 21.5 pressures per 450 rushes under forecast. Pittsburgh's odd front sent Black straight ahead (0% such rushes), and he sits 3.0 over.

**Cost of a wrong call:** a tiebreak swaps players a median 20 picks apart. Where the alternative out-rushed the plus call, the loss was at most 3.2 pressures per 450 rushes, about 1.7 expected points a season.

## So what for a team

| 2025 player (pick, role) | Call (Hoop G, sd) | Rookie pressure vs forecast (rushes) | Snap plan | Film cut |
|---|---|---|---|---|
| **Ethan Downs** (UDFA, EDGE) | Plus (0.94) | — (7) | Third-down sub rusher, 10–15 snaps a game, to get a read | Third-down wide rushes in a KC sub package: bend at the top, or drift? |
| **Warren Brinson** (198, Int) | Plus (0.77) | 6.9% vs 6.6% (173) | Keep rotation; make him the looper on stunts | Looper on interior stunts: tight turn behind the crash, or wide? |
| **Donovan Ezeiruaku** (44, EDGE) | Plus (0.77) | 7.7% vs 9.5% (402) | Keep starter snaps; no raise until the arc cut is graded | Wide-arc rushes from a 9 technique: flattens to the quarterback, or runs past? |
| **Jack Sawyer** (123, EDGE) | Film first (0.56) | 9.7% vs 8.5% (176) | Keep role; second hoop run in camp tracker | Speed-to-power vs corner turns: which one wins? |
| **Tyler Baron** (176, EDGE) | Film first (−0.62) | 4.4% vs 8.0% (68) | More rush snaps: 68 is too few to judge | Any corner turn at all, or power only? |
| **Jordan Phillips** (143, Int) | Minus flag (−0.80) | 1.9% vs 6.6% (215) | Early downs; take him off stunts | Rushes where the play moves away from him |
| **Yahya Black** (164, Int) | Minus flag (−0.75) | 5.8% vs 5.2% (189) | Keep two-gap nose role; no pass-rush upgrade | Two-gap nose snaps vs true pass-rush downs (Pittsburgh odd front) |

1. **Break ties, do not move boards.** Use Hoop G only between players that film, and the 3-cone where it exists, cannot separate.
2. **Run the hoop twice.** 6 of these 7 rushers have one run; a second, at a pro day or in a camp tracker, lifts reliability from 0.41 to 0.58.
3. **Match flags to scheme.** Phillips's flag matters for a stunt-heavy front; Black's barely matters for a two-gap nose.

## Limitations

- **Selected, then tested on the same seasons.** The adjusted p = 0.12 is the honest number; holding out classes checks the forecast, not the drill choice.
- **Small and early.** 9 scored plus calls; the 2025 class has one season. The 2026 class alone has 30% power under the frozen rule; four classes reach 81%, or 37% if the true gap is half as large.
- **No mechanism.** Hoop G does not show in depth, turning or rush direction on NFL frames.
- **Thin NFL data.** Season counts ignore opponents; sampled tracking has about 20 random rushes per player and no blockers or ball. Scheme reads come from loop rates, not play calls.

## Appendix

| Check | Result [90%] |
|---|---|
| Frozen-rule hit counts, by class 2023 / 2024 / 2025 | plus 1/3, 3/4, 0/2; others 5/24, 4/20, 0/25; exact test within class p 0.07 |
| Call interval 70 / 80 / 90% (30+ rushes) | 5 of 13 vs 8 of 65; 4 of 9 vs 9 of 69; 0 of 2 vs 13 of 76; plus ahead in 18 of 27 settings (interval × minimum rushes × forecast) |
| Rule re-chosen on two classes, scored on the third | 4 of 12 vs 9 of 66 |
| Weak-prior slope: pooled; mean of class slopes; forecast scale free | 0.91 pts per sd [0.44, 1.37], shuffle p 0.002; 0.81; 0.91 [0.44, 1.39] |
| Slope fit on two classes, scored on the third | improves the held-out log score in 3 of 3 classes |
| Count model with exposure and rotation (78) | 1.1 pressures per 450 rushes per sd [0.0, 2.5] |
| Post hoc: plus call minus nearest-pick comparison player (9 pairs) | +3.2 pts [1.6, 4.9], 7 of 9 ahead; designed after the rule |
| 3-cone count model (37) | 0.5 pressures per 450 rushes per sd faster [−1.0, 2.1] |
| Hoop G vs matched-speed NFL turning; outside-rush share | 0.002 g per sd [−0.006, 0.011] (40 EDGEs); ρ 0.07 (38) |

| 3-cone, DL | 2023 | 2024 | 2025 | All |
|---|---|---|---|---|
| Linemen in cohort | 37 | 37 | 44 | 118 |
| Official time (used here) | 18 | 10 | 15 | 43 |
| Tracked attempt | 19 | 12 | 14 | 45 |
| Both | 18 | 10 | 14 | 42 |
| Tracked only (no official time) | 1 | 2 | 0 | 3 |
| Official only (no tracked run) | 0 | 0 | 1 | 1 |
| Ran Run-the-Hoop, no official 3-cone | 15 | 27 | 27 | 69 |

*Reproducible: `uv run python writeup/candidates/round_11/cand_1/analysis.py` writes `numbers.json`, `claims.json`, `frozen_rule.json`, the board CSV and the three figures, then fills this text from `writeup_template.md`.*
