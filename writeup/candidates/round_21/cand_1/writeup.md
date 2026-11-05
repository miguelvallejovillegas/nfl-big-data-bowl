# Run-the-Hoop Bend Earns a Day-3 Rusher the First Film Look, Not a Board Spot

**On a Combine class the model never saw, one sd of Hoop G was worth 1.8 extra pressures per 450-rush season [90%: 0.2, 3.4] beyond a forecast built from pick, weight, role, NGS score, 10-yard split and 3-cone (n = 77 rushers). That is the best of six drill measures we listed, but within what the best of six reaches by chance (family-wise p = 0.23). Pass-rush-drill burst adds nothing (−0.5 per sd).**

**Hoop G** = how hard a lineman leans sideways at the tightest half-second of the Run-the-Hoop figure-eight: speed squared divided by turn radius, in g, ranked within his Combine class. The median lineman leans 0.62 g, which is 3.5 yd/s on a 1.8-yard radius.

**So what for a team**

1. **Film order, not board order.** Watch the day-3 edge rushers in their class's top third of Hoop G first. Within day 3 it orders rushers better than draft order did, but its top third hits no more often.
2. **Drop pass-rush-drill burst.** One rep repeats at 0.13. The official 3-cone and 10-yard split carry NFL get-off at least as well.
3. **2026 snap calls** for eight named day-3 rushers are in Figure 4. NFL pressure makes those calls; Hoop G only put the names on the list.

![Figure 1. Four rushers, Combine lap (top) against their randomly sampled NFL rushes, first 2 s after the snap (middle, n = 16–19 rushes each), on one yard scale and one colour scale. Bottom: career pressures above the no-tracking forecast with 90% binomial intervals.](figures/fig1_hero.png)

## Why bend, and why day 3

On 1655 randomly sampled NFL pass plays involving these linemen, a pressured snap cost the offense 0.52 expected points more than a clean one [0.35, 0.67]. 69% of the 118 defensive linemen in the 2023–25 classes went on day 3 or undrafted, where tape is thin and the board needs a tiebreak. Before fitting anything we set the smallest effect worth acting on: 2.25 pressures per 450-rush season per sd, about 1.2 expected points a season.

Figure 1 shows the problem in four players. Javon Solomon (BUF) and Cedric Johnson (CIN) lean hard in the hoop and win in the NFL. Tyrion Ingram-Dawkins (MIN) leans almost as hard and does not. Byron Murphy (SEA) is below his class in the hoop and still beats his forecast by 8.1 pressures a season. The NFL bend beneath each lap does not track the lap.

## The NFL end of the chain works; the drill end does not

We tested three links: does the drill show up in the matching NFL movement, and does that movement produce pressure? NFL movements come from a median of 17 randomly sampled rushes per lineman, measured against his own start (the data have no blockers or ball). Every link controls for weight and EDGE share. The right column is the no-tracking benchmark: the official time that should carry the same trait.

| Link | Tracking ρ [90%] (n) | Stopwatch ρ [90%] (n) |
|---|---|---|
| Combine → NFL bend (sideways g, first 2 s) | Hoop G 0.14 [−0.06, 0.32] (78) | 3-cone 0.22 [−0.25, 0.64] (31) |
| Combine → NGS get-off | Drill burst −0.14 [−0.28, 0.03] (97) | 10-yd split 0.12 [−0.08, 0.31] (78) |
| NFL bend → pressure rate | 0.36 [0.20, 0.52] (85) | — |
| NGS get-off → pressure rate | 0.41 [0.27, 0.54] (85) | — |

*Table 1. Partial Spearman ρ, class-stratified bootstrap 90% intervals. Repeatability: one hoop lap 0.37 (only 10 linemen ran two), one rush-drill rep 0.13 (n = 101); NFL bend 0.63 split-half, NGS get-off 0.84 season to season.*

Linemen who get off quicker and bend harder on NFL snaps generate more pressure. The Combine end breaks: the rush-drill rep cannot see get-off (−0.14), and the 3-cone carries it better (0.30 [0.01, 0.56], n = 37). Hoop G reaches NFL bend only weakly. On the 29 linemen who ran both, Hoop G does no better than the stopwatch 3-cone (−0.12 against 0.22). Hoop G matters because 69 of the 109 hoop runners never ran the 3-cone.

## Held out by class, Hoop G leads but does not clear the screen

**Baseline without tracking:** log draft pick, Combine weight, NFL EDGE share, NGS athleticism score, official 10-yard split and 3-cone. **Outcome:** career pressure rate through 2025, shrunk toward the 8.1% mean, for linemen with 30+ rushes. **Out of class:** each class (27/23/27 rushers) is forecast by a model fit on the other two. All drill measures are standardized within Combine year.

![Figure 2. Held-out effect of the six pre-listed drill measures on pressures above the no-tracking forecast (n = 77). Bars: 90% class-stratified bootstrap intervals, refitting every fold. Diamonds: each held-out class. Grey band: action margin. Dashed lines: the size the best of six shuffled measures reaches 5% of the time.](figures/fig2_screen.png)

| Model (held out, n = 77) | R² | Rank ρ | R² gain [90%] |
|---|---|---|---|
| Pick + role only | 0.10 | 0.31 | — |
| No-tracking baseline | −0.02 | 0.28 | — |
| Baseline + Hoop G | 0.04 | 0.37 | 0.06 [−0.20, 0.25] |
| Baseline + drill burst | −0.03 | 0.28 | −0.01 [−0.17, 0.05] |

*Table 2. Leave-one-class-out fit. The extra Combine times cost more in fitting noise than they add.*

- **Hoop G** is positive in 2023 and 2024 (2.4, 3.0) and flat in 2025 (−0.1). Only 33% of bootstrap draws clear the action margin.
- **The estimate does not hinge on one implementation, but its size does move.** Weighting rushers by the inverse chance of having all six drill measures (77 of 85) gives 1.9. A separate script with different smoothing, shrinkage, imputation and seed gives 2.4 [0.7, 4.2] (n = 78). Prior strength and rush minimums move it between 1.5 and 2.0.
- **Burst is out:** 93% of draws fall inside the margin, and it is negative in two of three classes.
- **Clubs do not pay for bend.** Top-third Hoop G linemen went at a median pick of 158, bottom-third at 130.

## Within day 3, Hoop G orders film but does not pick players

We ranked the 47 day-3 and undrafted rushers with 30+ rushes (16/12/19 by class), each counted once. The Hoop G weight added to the forecast comes from the other two classes. The bar for letting Hoop G touch board order: its top third must beat pick order's top third by 20 points of hit rate, with the 90% lower bound above zero.

![Figure 3. Day-3 rankers scored on the held-out class (47 rushers). A: within-class rank correlation with NFL pressure rate (rusher bootstrap, 90%). B: share of each ranker's top third who out-rushed the class's day-3 median (Wilson 90%).](figures/fig3_day3.png)

- **Ordering:** forecast + Hoop G ρ = 0.29 [0.03, 0.52] against 0.01 for draft order (gap 0.28 [0.03, 0.53]).
- **Picking:** both top thirds hit 9 of 15 (60% [39, 78]). The gap is 0 points [−20, 33], far from the bar. As a picker, Hoop G is a coin flip.

That is why the use is film order. It changes which tape a scout opens first, not where a name sits.

## 2026 calls for the day-3 rushers Hoop G flagged

![Figure 4. 2026 cards for every day-3 rusher in the 2024–25 classes with top-third Hoop G and 100+ NFL rushes. Strip: career pressures per 450 rushes above the frozen forecast, 90% binomial interval. Splits: randomly sampled rushes (n = 20 each), pressured of total, by alignment and on third down.](figures/fig4_cards.png)

- **Go:** Cedric Johnson (CIN), Javon Solomon (BUF), Warren Brinson (GB), Austin Booker (CHI) and Jack Sawyer (PIT). Johnson (11.2% against a 7.2% forecast over 216 rushes) and Solomon (10.8% against 7.2%) cannot drop below forecast in one season. Give both third-down snaps from wide alignments. Brinson is the interior case: use him as the looper on stunts. He flips below 5.1%.
- **No-go:** Eric Watts (MIA), Tyrion Ingram-Dawkins (MIN) and Jaylen Harrell (TEN). Each needs a 2026 rate of at least 10.2% to reach his forecast; keep them on run-down rotation until then.
- **Hold:** Zeek Biggers (MIA), Nazir Stackhouse (TEN) and Kyle Kennard (LAC). Stackhouse and Kennard first need 100 rushes.

The sampled splits (Figure 4) are 20 rushes each, so they direct film cut-ups, not grades. The rule is frozen (sha256 `8277b10e1492…`). After the 2026 class's first season, Hoop G must show a positive slope on pressure over forecast (one-sided p < 0.10) or the flag is retired. Power at the observed slope is 44% for one class and 77% for three.

## Limitations

- **Small samples.** 77 rushers in the model, 47 on day 3, mostly single drill reps: 91% ran one hoop lap.
- **Multiplicity.** This writeup reports 24 tests; only the six-measure family is corrected. Earlier work screened ideas on these same classes, so 2026 is the first clean test.
- **Thin NFL tracking.** A median of 17 sampled rushes per lineman, with no blockers or ball. Bend may follow from beating the tackle rather than cause it.
- **Weak baseline.** Pick + role alone beats the full no-tracking model out of class, so gains over it look larger than gains over a scout.
- **Survivorship and roles.** Rates need 30+ rushes; injuries, scheme and roster moves are not modelled.

## Reproduce

`analysis.py` (about 3 minutes) rebuilds every number, figure and this text from `data/processed`, and runs `verify.py`, the independent re-implementation of the headline slope.
