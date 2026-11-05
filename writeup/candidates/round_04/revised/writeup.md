# Bend as a Tiebreaker: Run-the-Hoop "Hoop G" and NFL Pass-Rush Pressure

Defensive linemen who turned tighter at speed in the Combine's Run-the-Hoop drill produced a little more NFL pressure than their draft slot, size and athletic score predicted. Plan on under one extra pressure a season for a rusher one notch better at bend. That is a tiebreaker, not a reason to move up a round.

**Why DL.** Pressure is what a defensive lineman is paid for, and DL is the one position where Combine numbers already predict NFL production (3-cone ρ = 0.63). Three tracked Combine classes (2023–25) now have NFL snaps.

## Hoop G in coaching terms

**Hoop G is the hardest sideways push a rusher holds for half a second while rounding the hoops, measured in g.** It is the drill version of bending the corner: dip, keep speed through the turn, rip through without drifting wide. A typical run peaks at 0.62 g, about 3.5 yd/s on a 1.8-yard radius. "One notch" below means one standard deviation within a Combine year. **NFL rush-path bend** applies the same formula to 20 random sampled NFL rushes per player, snap to throw or sack.

![Figure 1. Two 2025 film checks (Ezeiruaku, Phillips) and, by rule, the highest- and lowest-bend EDGE rushers with 250+ NFL rushes. Win type is read from the rusher's own path on EDGE snaps; interior snaps are left grey.](figures/fig1_four_rushers.png)

**Does the drill show up as technique?** Not that we can see. On 925 EDGE-aligned sampled rushes we read the win type from the path: outside arc (dip, rip), inside cross (counter) or through the blocker (power). Outside arcs are only 9% of them, and high-bend rushers do not run more of them (ρ = 0.04 [−0.24, 0.34], n = 41). Drill bend barely tracks NFL rush-path bend after weight and role (0.10). **Donovan Ezeiruaku** (#3 of 42 in 2025 bend) bends on Sundays (rush-path bend at the 90th percentile), yet only 14% of his EDGE rushes are outside arcs; he produced 7.7% against a 9.5% forecast. **Jordan Phillips** (#42 of 42) plays square in the drill and in games: 1.9% against 6.8%. Hoop G is a drill score with a pressure link, not proof of a corner-bending skill.

## Data and baseline

**Outcome.** Pressures out of regular-season pass rushes. The headline model uses all 90 hoop runners with an NFL rush, weighted by rushes, so there is no 100-rush cut (19 never rushed). Rank checks use the 63 with 100+ rushes. Drill numbers are standardized within Combine year.

**Baseline without tracking:** log draft pick, weight, NGS athleticism score and NFL EDGE share, observed for everyone. The model adds games per season (median 1.0 for runners under 100 rushes, 14.0 above). Bend is unrelated to playing time (ρ = 0.05).

## Finding 1: the stopwatch carries most of the signal

![Figure 2. Five Combine numbers against three NFL outcomes. Pressure-panel n: 3-cone 27, 40 54, pick 69, burst 68, bend 63. Purple diamonds: what is left after the baseline.](figures/fig2_signal_map.png)

The 3-cone and 40 track pressure, as teams already know. Among tracked drill numbers, only bend keeps a pressure link after the baseline. The pass-rush drill's first-second burst cannot rank players: a player's 1.9 reps have reliability 0.22.

## Finding 2: under one pressure a season, not confirmed out of class

![Figure 3. Pressure above the no-tracking forecast against Hoop G, one panel per group. The two slopes are statistically indistinguishable.](figures/fig3_hoop_pressure.png)

| Test | n | Estimate | Read |
|---|---|---|---|
| **Headline:** extra pressures per 450-rush season, per sd of bend (Bayesian, 90% interval) | 90 | **0.9** [−0.1, 1.9]; P(> 0.5) = 0.74 | planning number |
| Same, EDGE / interior | 90 | 0.9 / 0.8 | similar |
| Raw slope, no prior or reliability correction | 63 | 3.1 pressures | overstated |
| **Out of class:** fully nested, held-out R² change | 63 | −0.028 [−0.314, 0.210] | no forecast gain |
| Placebo: run-stop proxy (TFL per run snap) | 60 | ρ 0.02 [−0.22, 0.27] | none, as expected |
| Placebo: tackle rate | 63 | ρ −0.21 [−0.43, 0.03] | none positive |
| Negative-control drill: back-pedal speed → pressure | 38 | ρ 0.09 [−0.26, 0.44] | none, as expected |
| Technique: time to pressure / sacks per pressure | 63 | ρ 0.06 / 0.10 | no link |

A rusher one notch better on a single hoop run is worth about **0.9 extra pressures in a 450-rush season**, the same as moving from pick 100 to pick 75. The model corrects bend for one-run reliability (0.41) and starts from the typical effect among the 14 drill-and-outcome pairs we screened. The slope is the same with or without an official 40 (Figure 3). But the forecast gain is not testable in the 49 players who ran the 40. In the strictest test, with screening, reliability and shrinkage refit on two classes to predict the third, bend adds nothing (−0.028; Figure 4). Placebo outcomes and a back-pedal control show no link, so a general "good athlete" effect is unlikely.

![Figure 4. Left: posterior for the headline effect, with a wide-prior check. Right: the fully nested out-of-class test, per held-out class and pooled.](figures/fig4_headline.png)

## What a team can use

**Decision rule, applied to every row:**
1. **Bend plus** (80% interval of shrunk bend above zero): break ties his way.
2. **Bend minus** (80% interval below zero): he must win with power or get-off; check that on film.
3. **Film first**: if a rusher with 100+ NFL rushes has a pressure interval wholly on the wrong side of his forecast, trust the film over the drill.

**2025 class, top 8 and bottom 4 of 42 hoop runners** (pressure: rookie rate vs no-tracking forecast fit on 2023–24):

| #  | Player | Pick | Wt | Role | Bend, sd [80%] | Pressure vs forecast (rushes) | Call |
|---|---|---|---|---|---|---|---|
| 1 | Ethan Downs | UDFA | 269 | EDGE | 0.94 [0.31, 1.57] | 7 rushes | **plus** |
| 2 | Warren Brinson | 198 | 315 | Int | 0.77 [0.14, 1.41] | 6.9% vs 6.9% (173) | **plus** |
| 3 | Donovan Ezeiruaku | 44 | 248 | EDGE | 0.77 [0.14, 1.40] | 7.7% vs 9.5% (402) | **plus** |
| 4 | Jah Joyner | UDFA | 262 | EDGE | 0.60 [−0.03, 1.23] | no rushes | neutral |
| 5 | Tyrion Ingram-Dawkins | 139 | 276 | EDGE | 0.58 [−0.06, 1.21] | 5.3% vs 8.5% (113) | neutral |
| 6 | Jack Sawyer | 123 | 260 | EDGE | 0.56 [−0.07, 1.19] | 9.7% vs 8.7% (176) | neutral |
| 7 | Kyle Kennard | 125 | 254 | EDGE | 0.51 [−0.12, 1.14] | 50 rushes | neutral |
| 8 | Jordan Burch | 78 | 279 | EDGE | 0.49 [−0.14, 1.12] | 5.8% vs 8.6% (344) | film first (under) |
| 39 | Ashton Gillotte | 66 | 264 | EDGE | −0.55 [−1.18, 0.08] | 6.4% vs 9.4% (342) | neutral |
| 40 | Tyler Baron | 176 | 258 | EDGE | −0.62 [−1.26, 0.01] | 68 rushes | neutral |
| 41 | Yahya Black | 164 | 336 | Int | −0.75 [−1.38, −0.12] | 5.8% vs 5.2% (189) | **minus** |
| 42 | Jordan Phillips | 143 | 312 | Int | −0.80 [−1.43, −0.17] | 1.9% vs 6.8% (215) | **minus** |

**2023–24 classes, every non-neutral call.** Across 2023–25 the rule gives 10 plus, 8 minus and 2 film-first; the 2023–24 rows are below (career pressure):

| Class | Player | Pick | Wt | Role | Bend, sd [80%] | Pressure vs forecast (rushes) | Call |
|---|---|---|---|---|---|---|---|
| 2023 | Adetomiwa Adebawore | 110 | 282 | Int | 0.94 [0.31, 1.57] | 8.6% vs 7.5% (511) | **plus** |
| 2023 | Nick Hampton | 161 | 236 | EDGE | 0.92 [0.29, 1.55] | 98 rushes | **plus** |
| 2024 | Javon Solomon | 168 | 246 | EDGE | 0.80 [0.16, 1.43] | 12.3% vs 7.3% (211) | **plus** |
| 2024 | Chop Robinson | 21 | 254 | EDGE | 0.73 [0.10, 1.36] | 11.8% vs 8.6% (726) | **plus** |
| 2023 | Tuli Tuipulotu | 54 | 266 | EDGE | 0.73 [0.10, 1.36] | 9.8% vs 7.3% (1486) | **plus** |
| 2024 | Cedric Johnson | 214 | 260 | EDGE | 0.66 [0.03, 1.29] | 13.0% vs 7.5% (216) | **plus** |
| 2024 | Austin Booker | 144 | 240 | EDGE | 0.65 [0.02, 1.29] | 8.0% vs 7.1% (535) | **plus** |
| 2024 | Byron Murphy | 16 | 297 | Int | −0.45 [−1.08, 0.18] | 10.0% vs 7.7% (682) | film first (out) |
| 2023 | Keondre Coburn | 194 | 332 | Int | −0.63 [−1.26, 0.00] | 4.5% vs 5.8% (112) | **minus** |
| 2024 | McKinnley Jackson | 97 | 326 | Int | −0.67 [−1.31, −0.04] | 5.1% vs 6.2% (214) | **minus** |
| 2023 | Jerrod Clark | UDFA | 334 | Int | −0.70 [−1.33, −0.07] | no rushes | **minus** |
| 2024 | T'Vondre Sweat | 38 | 366 | Int | −0.73 [−1.36, −0.10] | 7.7% vs 6.5% (496) | **minus** |
| 2024 | Darius Robinson | 27 | 285 | Int | −0.74 [−1.37, −0.11] | 2.4% vs 7.3% (459) | **minus** |
| 2024 | Justin Rogers | 244 | 330 | Int | −0.94 [−1.57, −0.30] | no rushes | **minus** |

![Figure 5. The 2025 rows that carry information: every plus, minus and film-first call, plus the five neutral rookies with the most rushes.](figures/fig5_board_2025.png)

Jordan Burch is the 2025 film-first call: a plus-side hoop, but rookie pressure below forecast. Byron Murphy is the reverse: a low hoop and pressure above forecast. **Ask for 3–4 hoop runs.** One run has reliability 0.41 [0.14, 0.60]; three reach 0.68 and four 0.74. Undrafted Ethan Downs ran the best hoop in 2025 (z = +2.3, shrunk +0.94) but has 7 NFL rushes: worth a second run before moving him up.

## Pre-registered test

**Primary:** among 2026- and 2027-class DL with 100+ rushes by the end of 2027 (about 40), the bend effect on pressure after the four-item baseline is above zero (one-sided α = 0.05). Power is 0.13 at the headline effect, so this test can rule out a large effect, not prove a small one. An estimate at or below zero takes bend off the board. **Null check:** the bend slope does not differ between rushers with and without an official 40 (two-sided; today p = 0.74).

## Limitations

- **Not confirmed out of class.** The fully nested test shows no forecast gain, and the hoop is chosen in only 2 of 3 folds. The headline is a small, plausible effect, not a proven one.
- **Blocker and alignment.** Pressure by blocker (tackle, guard, chip) or by alignment cannot be built from this data: game tracking has only the cohort player, with no ball, blockers or opponents. Win type is a path proxy, not film, and outside/inside is only trusted 4+ yards from the field centre.
- **Small n.** About 20 rushers per class; only 10 ran the hoop twice.
- **Game tracking is a sample** of 20 random rushes per player; scheme, stunts and quarterback timing are not in the data.
- **Short careers.** The 2025 class has one season: its board is for order, not levels.

## Appendix: statistics

| Check | n | Estimate [95% CI] |
|---|---|---|
| Nested out-of-class R² change: 2023 / 2024 / 2025 held out | 63 | −0.006 / −0.014 / −0.270 |
| Same, with 10-yd split and 3-cone imputed inside each fold | 63 | −0.026 [−0.406, 0.193] |
| Posterior with a wide prior (pressures per season, 90%) | 90 | 2.5 [0.9, 4.3] |
| Posterior per sd of true (error-free) bend | 90 | 1.3 [−0.1, 3.1] |
| Rank link after baseline, in sample | 63 | ρ 0.32 [0.06, 0.54] |
| Slope per sd: ran the 40 / skipped it (pp) | 49 / 14 | 0.58 [0.01, 1.08] / 0.73 [−0.05, 1.41] |
| Best of 14 screened pairs, within-class permutation | 63 | p = 0.121 |
| Bend → NFL rush-path bend, after weight and role | 79 | 0.10 [−0.11, 0.32] |

**Methods.** Savitzky–Golay smoothing (window 7, order 2); sideways acceleration |v × a|/|v| over 0.5 s, identical for drill and NFL snaps. Headline: pressures ~ beta-binomial(rushes), logit link, class random intercept, EDGE and interior bend slopes, baseline plus games per season; latent true bend integrated by Gauss–Hermite quadrature. The bend-slope prior's standard deviation is the mean |ρ| of the 14 screened pairs (0.11), converted to the logit scale. Adaptive Metropolis, 4 chains, R-hat 1.01. Nested test: inside each fold, screen 14 pairs, choose the best, estimate its reliability and shrink scores, then shrink its slope toward the other 13; 1000 class-stratified player bootstraps refit everything. Win type: lateral position when the rusher first reaches 3 yards upfield (outside ≥ 0.5 yd, inside ≤ −1 yd).

*Reproducible: `uv run python writeup/candidates/round_04/revised/analysis.py` writes `numbers.json` (one key per quoted number), draws every figure and fills this text from `writeup_template.md`.*
