# Run-the-Hoop Bend Earns a Day-3 Rusher the First Film Look, Not a Board Spot

**On a Combine class the model never saw, one sd of Hoop G was worth 1.8 extra pressures per 450-rush season [90%: 0.2, 3.4] beyond a no-tracking forecast (n = 77 rushers). It is the best of six drill measures we listed, but within what the best of six reaches by chance (family-wise p = 0.23), and one lap is too noisy to grade a lineman on.**

**Hoop G, in coach terms:** how hard a lineman leans into the tightest part of the Run-the-Hoop figure-eight, the dip-and-flatten an edge needs to turn the corner on a tackle. Physically it is speed squared over turn radius, in g, ranked within his Combine class. The median lineman carries 7.1 mph around a 1.8-yard radius: 0.62 g.

**So what for a team**

1. **Film order, not board order.** Open the tape of day-3 edge rushers in their class's top third of Hoop G first. Do not move anyone up the board for it.
2. **Four hoop laps, no burst reps.** One lap has reliability 0.43; four reach 0.70. Pass-rush-drill burst needs 16 reps to get there and does not carry to NFL get-off.
3. **2026 snap calls** for eight day-3 rushers are in Figure 4. NFL pressure makes those calls; Hoop G only put the names on the list.

![Figure 1. Hoop lap (top) and randomly sampled NFL rushes, first 2 s after the snap (middle, 16–19 per player), on one yard and colour scale. Bottom: career pressures above the no-tracking forecast, 90% interval.](figures/fig1_hero.png)

## Drill rep against NFL rep

The hoop asks for about three times the lean a rusher shows in his first two seconds on Sunday (0.62 g against a median 0.18 g on sampled NFL rushes), so each player is compared with his class, not in raw g.

- **Javon Solomon (BUF).** Hoop: 0.73 g at 7.6 mph (+1.9 sd, top of his class). NFL: 0.20 g on average (+0.6 sd), 16.1 pressures a season above forecast. *Coach read:* the bend carries; give him third-down snaps from a wide alignment.
- **Cedric Johnson (CIN).** Hoop: 0.69 g at 7.3 mph (+1.1 sd). NFL: 0.20 g (+0.7 sd), 18.0 pressures above forecast. *Coach read:* same profile; keep him on the edge on passing downs.
- **Tyrion Ingram-Dawkins (MIN).** Hoop: 0.65 g at 6.7 mph (+1.0 sd). NFL: class-average bend (0.0 sd), −5.1 pressures against forecast. *Coach read:* the hips are there, the rush plan is not; rep dip-and-rip off a live get-off, not the bag.
- **Byron Murphy (SEA).** Hoop: 0.59 g at 7.2 mph (−1.1 sd, below his class). NFL: 0.19 g (+0.3 sd), 8.1 pressures above forecast. *Coach read:* an interior quickness-and-power winner; never grade him on the hoop.

Two of four fit the story and two do not. The rest of this writeup asks whether that is better than chance.

## Why bend, and why day 3

On 1655 randomly sampled NFL pass plays, a pressured snap cost the offense 0.52 expected points more than a clean one [0.35, 0.67]. 69% of the 118 defensive linemen in the 2023–25 classes went on day 3 or undrafted, where tape is thin. Before fitting, we set the smallest effect worth acting on: 2.25 pressures per 450-rush season per sd (about 1.2 expected points).

## The NFL end of the chain works; the drill end does not

NFL movements come from a median of 17 randomly sampled rushes per lineman, measured against his own start (the data have no blockers or ball). Every link controls for weight and EDGE share; the right column is the official time that should carry the same trait.

| Link | Tracking ρ [90%] (n) | Stopwatch ρ [90%] (n) |
|---|---|---|
| Combine → NFL bend (first 2 s) | Hoop G 0.14 [−0.06, 0.32] (78) | 3-cone 0.22 [−0.25, 0.64] (31) |
| Combine → NGS get-off | Drill burst −0.14 [−0.28, 0.03] (97) | 10-yd split 0.12 [−0.08, 0.31] (78) |
| NFL bend → pressure rate | 0.36 [0.20, 0.52] (85) | — |
| NGS get-off → pressure rate | 0.41 [0.27, 0.54] (85) | — |

*Table 1. Partial Spearman ρ, class-stratified bootstrap 90% intervals.*

Rushers who get off quicker and bend harder on NFL snaps generate more pressure; the NFL-to-pressure links are the only results among all 26 tests here that survive a Holm correction. The Combine end breaks. Drill burst does not see NGS get-off, while the 3-cone does better (0.30 [0.01, 0.56], n = 37). On the 29 linemen who ran both, Hoop G does no better than the 3-cone (−0.12 against 0.22). Hoop G matters because 69 of the 109 hoop runners never ran the 3-cone.

**Why the drill end breaks: one rep.** The hoop is a figure-eight, so every lap has a left-hand and a right-hand turn. Across 109 runners the two turns of the same lap agree at only ρ = 0.27, which puts one lap at reliability 0.43 [0.21, 0.58]. The 10 linemen who ran two laps give a consistent 0.37. It takes 4 laps to reach 0.70 (5 on the two-lap estimate), about the reliability of the NFL bend measure (0.63). A burst rep repeats at 0.13 (n = 101). That is the evidence behind the protocol: four graded hoop laps, and no burst reps.

![Figure 2. A: reliability of a lineman's drill average by number of reps (Spearman-Brown from one-rep reliability; bands = 90% player bootstrap). Ticks at right: NFL-side reliabilities. B: lean on the two turns of the same lap, z within class, n = 109; Figure 1 rushers coloured.](figures/fig2_reliability.png)

## Held out by class, Hoop G leads but does not clear the screen

**Baseline without tracking:** log draft pick, Combine weight, NFL EDGE share, NGS athleticism score, official 10-yard split and 3-cone. **Outcome:** career pressure rate through 2025, shrunk toward the 8.1% mean, for linemen with 30+ rushes. **Out of class:** each class (27/23/27 rushers) is forecast by a model fit on the other two. Fits are unweighted; RMSE is in pressure-rate points (outcome sd 1.72). Drill measures are standardized within Combine year.

![Figure 3. Held-out effect of the six pre-listed drill measures on pressures above the no-tracking forecast (n = 77). Bars: 90% class-stratified bootstrap, refitting every fold. Diamonds: each held-out class. Grey band: action margin. Dashed lines: what the best of six shuffled measures reaches 5% of the time.](figures/fig3_screen.png)

| Model (held out, n = 77) | R² | RMSE | Rank ρ | R² gain [90%] |
|---|---|---|---|---|
| Pick + role | 0.10 | 1.63 | 0.31 | — |
| Pick + role + Hoop G | 0.16 | 1.58 | 0.41 | 0.06 [−0.10, 0.20] |
| Full no-tracking baseline, ridge | 0.01 | 1.71 | 0.28 | — |
| Ridge baseline + Hoop G | 0.04 | 1.68 | 0.36 | 0.04 [−0.07, 0.17] |

*Table 2. Leave-one-class-out; ridge penalty chosen inside the two training classes. Pick alone: R² 0.03; full baseline by OLS: −0.02. Every gain interval spans zero.*

- **Class by class,** Hoop G is positive in 2023 and 2024 (2.4, 3.0) and flat in 2025 (−0.1). Only 33% of bootstrap draws clear the action margin.
- **The sign is stable; the size moves:** 2.1 on the ridge residual, 1.9 weighting for having all six drill measures, 2.4 [0.7, 4.2] from an independent script (n = 78), 1.5 to 2.0 across priors and rush minimums.
- **Multiplicity.** Across all 26 tests, 4 survive Benjamini–Hochberg at q = 0.10: the three NFL trait-to-pressure links and drill burst to NFL burst. Hoop G (two-sided p = 0.03) does not.
- **Clubs do not pay for bend.** Top-third Hoop G linemen went at a median pick of 158, bottom-third at 130.

## Day 3: a film list, not a ranking

Among the 47 day-3 and undrafted rushers, forecast + Hoop G orders each held-out class better than draft order (ρ 0.29 against 0.01). But its top third hit 9 of 15, exactly as often as pick order's; the gap of 0 points [−20, 33] is far below the +20-point bar we set for board use. Hoop G changes which tape opens first, not where a name sits.

## 2026 calls for the day-3 rushers Hoop G flagged

![Figure 4. 2026 cards for every 2024–25 day-3 rusher with top-third Hoop G and 100+ NFL rushes. Strip: career pressures per 450 rushes above the frozen forecast, 90% interval. Splits: 20 randomly sampled rushes each.](figures/fig4_cards.png)

- **Go:** Cedric Johnson (CIN), Javon Solomon (BUF), Warren Brinson (GB), Austin Booker (CHI) and Jack Sawyer (PIT). Johnson (11.2% against a 7.2% forecast) and Solomon (10.8% against 7.2%) cannot drop below forecast in one season. Use Brinson, the interior case, as the stunt looper; he flips below 5.1%.
- **No-go:** Eric Watts (MIA), Tyrion Ingram-Dawkins (MIN) and Jaylen Harrell (TEN). Each needs a 2026 rate of at least 10.2%; run downs until then.
- **Hold:** Zeek Biggers (MIA), Nazir Stackhouse (TEN) and Kyle Kennard (LAC).

The rule is frozen (sha256 `8277b10e1492…`). After the 2026 class's first season, Hoop G must show a positive slope on pressure over forecast (one-sided p < 0.10) or the flag is retired. Power at the observed slope is 44% for one class and 77% for three.

## Limitations

- **Small samples, single reps.** 77 rushers in the model, 47 on day 3; 91% ran one hoop lap, so every Hoop G here is the noisy one-lap version.
- **Reuse of the same classes.** Earlier rounds screened ideas on these three classes, so the 2026 class is the first clean test.
- **Thin NFL tracking.** A median of 17 sampled rushes per lineman, with no blockers or ball. Bend may follow from beating the tackle rather than cause it.
- **Timing and missed time.** No tracked 40 is used, so the 0.3 s clip offset never enters; drill burst is timed from the lineman's own first step. Rates are per rush, so injured weeks shrink the denominator, not the rate, and short careers are shrunk toward the mean. The 33 of 118 linemen with fewer than 30 rushes (injured, cut or stashed) are left out and not reweighted.
- **Weak models.** Pick plus role beats the full no-tracking model out of class, and no model explains much of a lineman's pressure rate.

## Bottom line

For day-3 edge rushers, open the tape in Hoop G order, and move no one on the board for it until four-lap Hoop G passes the 2026 test.

`analysis.py` (about 5 minutes) rebuilds every number, figure and this text from `data/processed`, and runs `verify.py`, an independent re-implementation of the headline slope.
