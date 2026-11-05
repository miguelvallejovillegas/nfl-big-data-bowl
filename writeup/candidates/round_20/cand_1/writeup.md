# Bend Shows, Burst Doesn't: What Combine Pass-Rush Drills Say About NFL Pressure

**The plain answer: the Combine's pass-rush drill cannot see get-off, and Run-the-Hoop sees bend only faintly.** We tracked all 118 defensive linemen from the 2023–25 classes at 10 Hz. We then scored each drill only on a Combine class the model had not seen (77 rushers). One sd of **Hoop G**, the hardest lean around the hoop, was worth **1.8 pressures per 450-rush season** [90%: 0.2, 3.4]. That is on top of a forecast built from pick, weight, role, NGS score, 10-yard split and 3-cone. Burst out of a stance was worth −0.5 [−2.3, 1.3]. We tested six drill measures, though, and Hoop G does not survive that screen (family-wise p = 0.23).

**Recommendation:** use Hoop G only to decide which day-3 edge rushers' film to watch first. Never move a name on the board for it, and drop drill burst.

## Two drill numbers a coach can repeat

- **Hoop G:** take the tightest half-second of the figure-eight, then divide speed squared by turn radius and express it in g. The median lineman leans at 0.62 g, which is 3.5 yd/s on a 1.8-yard radius. A faster lap through a tighter circle scores higher. 109 linemen ran it, and 69 of them never ran the 3-cone.
- **Burst:** yards covered in the first second after he moves in the pass-rush drill (median 2.45 yd; 117 linemen).

Both are standardized within Combine year, because drill medians shift between years.

![Figure 1. Six Run-the-Hoop laps, low to high Hoop G, all on one colour scale for sideways acceleration. Vermillion marks the half-second that defines Hoop G. The grey ghost is the median runner's lap.](figures/fig1_laps.png)

**Why it matters.** On 1655 randomly sampled NFL pass plays involving these linemen, a pressured snap cost the offense 0.52 expected points more than a clean one (0.35 to 0.67). Before fitting anything, we set one action margin: 2.25 pressures per 450-rush season per sd. That is one pressure every 200 rushes, or about 1.2 expected points a season.

## Get-off wins NFL snaps, but the drill rep cannot measure it

We followed each drill along a chain: does it repeat, does it show up in the matching NFL movement, and does that movement produce pressure? The NFL movements come from a median of 17 randomly sampled rushes per lineman, measured against his own position (the tracking has no blockers or ball). The NFL links control for weight and EDGE share.

| Link in the chain | ρ [90%] | n | Most it could be |
|---|---|---|---|
| Drill burst, rep to rep | 0.13 | 101 | — |
| Hoop G, lap to lap (two laps) | 0.37 (0.54) | 10 | — |
| Drill burst → NGS get-off | −0.14 [−0.28, 0.03] | 97 | 0.43 |
| Drill burst → NFL first-second burst | 0.27 [0.09, 0.42] | 84 | 0.43 |
| Hoop G → NFL bend (sideways g, first 2 s) | 0.14 [−0.06, 0.32] | 78 | 0.49 |
| NGS get-off → pressure rate | 0.41 [0.27, 0.54] | 85 | — |
| NFL first-second burst → pressure rate | 0.33 [0.14, 0.50] | 85 | — |
| NFL bend → pressure rate | 0.36 [0.20, 0.52] | 85 | — |

*Table 1. Partial Spearman ρ with class-stratified bootstrap intervals. "Most it could be" is the ceiling set by the two measures' repeatability. The NFL measures repeat well: NGS get-off season to season 0.84 (n = 40), NFL burst 0.83 and NFL bend 0.63 split-half (n = 85), pressure rate 0.64.*

The NFL end of the chain is strong. Linemen who get off quicker generate more pressure (0.41), and so do linemen who bend harder (0.36). The Combine end breaks. One pass-rush rep repeats at only 0.13, because a stance start depends on a single first step. Drill burst does show up in how far he travels in his first NFL second (0.27), but not in NGS get-off (−0.14). Hoop G reaches NFL bend only weakly (0.14).

## Only Hoop G adds anything past the baseline, and not past the screen

**Baseline without tracking:** log draft pick, Combine weight, NFL EDGE share, NGS athleticism score, and the official 10-yard split and 3-cone. Missing times are filled inside each fold. **Outcome:** career pressure rate through 2025, shrunk toward the 8.1% mean by 117 rushes of prior, for linemen with 30+ rushes. **Out of class:** each Combine class (27/23/27 rushers) is forecast by a model fit on the other two.

To answer the charge that Hoop G was picked after looking, we listed six drill measures (three per drill), gave each the same held-out test, and shuffled all six together within class. That gives the size the best of six random measures reaches by chance.

![Figure 2. Held-out effect of each drill measure, in pressures per 450-rush season per sd. Bars are 90% class-stratified bootstrap intervals that refit every fold, and diamonds show each held-out class. The grey band is the action margin. The blue dashes mark the size the best of six shuffled measures reaches 5% of the time.](figures/fig2_screen.png)

| Model (held-out, n = 77) | R² | RMSE (pts) | Rank ρ | R² gain [90%] |
|---|---|---|---|---|
| Pick + role only | 0.10 | 1.63 | 0.31 | — |
| No-tracking baseline | −0.02 | 1.74 | 0.28 | — |
| Baseline + Hoop G | 0.04 | 1.69 | 0.37 | 0.06 [−0.20, 0.25] |
| Baseline + drill burst | −0.03 | 1.75 | 0.28 | −0.01 [−0.17, 0.05] |

*Table 2. Leave-one-class-out fit. With 77 rushers, the extra Combine times cost more in fitting noise than they add.*

- **Burst is out.** Its slope is −0.5. 93% of bootstrap draws fall inside the margin (equivalence p = 0.06), and it is negative in two of three classes.
- **Hoop G is the only measure with a held-out signal** (p = 0.03 on its own). The signal holds in 2023 and 2024 (2.4, 3.0) and is flat in 2025 (−0.1). Across shrinkage and rush minimums it stays between 1.5 and 2.0. But it sits inside the ±2.4 chance range for six measures, and only 33% of draws clear the margin.
- **Clubs do not pay for bend.** Top-third Hoop G linemen went at a median pick of 158, and bottom-third linemen at 130.

## Within day 3, Hoop G orders rushers better than the draft did, but does not pick better

69% of these linemen went on day 3 or undrafted, and that is where the board needs help. We scored five ways to rank the 47 day-3 and undrafted rushers with 30+ rushes (16/12/19 by class). Each rusher counts once. The Hoop G weight added to the forecast is estimated on the other two classes.

![Figure 3. Day-3 rankers scored on the held-out class. A: within-class rank correlation with NFL pressure rate. B: how many of each ranker's top third out-rushed their class's day-3 median. Bars are 90% intervals (rusher bootstrap in A, Wilson in B).](figures/fig3_day3.png)

- **Ordering the whole group:** forecast + Hoop G ρ = 0.29 [0.03, 0.52], against 0.01 for draft order. The gap is 0.28 [0.03, 0.53]. Against the forecast alone (0.17), Hoop G adds 0.12 [−0.03, 0.27].
- **Picking a top third:** Hoop G and draft order both hit 9 of 15 (60% [39, 78]), so neither beats the other.
- **As a flag against the forecast:** 8 of 17 top-third Hoop G rushers beat their forecast. That is an error rate of 53% [34, 71], against 15 of 30 for everyone else. On average the flagged group still beat its forecast by 2.2 pressures a season more than the rest, but that comes from a few big hits.

So Hoop G improves the order but not the cut line. That is why the recommendation is film order, not board order.

## 2026 calls for the day-3 rushers Hoop G flags

The list below covers every day-3 rusher from the 2024–25 classes in his class's top third of Hoop G. **Go** means 100+ rushes and at least 0.5 rate points above forecast; **No-go** means at least 0.5 below; everyone else is **Hold**. Hoop G only chose who is on the list. The calls themselves come from NFL pressure against the frozen forecast. **Decision date: after Week 18 of 2026.**

![Figure 4. 2026 calls. Bars are 90% intervals of the shrunk career rate. The right column gives the 2026 pressure rate (at his 2025 volume) that would flip each call.](figures/fig4_board.png)

- **Go:** Cedric Johnson (CIN), Javon Solomon (BUF), Warren Brinson (GB), Austin Booker (CHI) and Jack Sawyer (PIT). Johnson (11.2% against a 7.2% forecast over 216 rushes) and Solomon (10.8% against 7.2%) cannot flip in one season, so give both third-down snaps from wide alignments, where bend shows. Brinson is the interior case: use him as the looper on stunts. He flips below 5.1%, Booker below 6.1% and Sawyer below 7.4%.
- **No-go:** Eric Watts (MIA), Tyrion Ingram-Dawkins (MIN) and Jaylen Harrell (TEN). Each needs a 2026 rate of at least 10.2% to reach his forecast.
- **Hold:** Zeek Biggers (MIA), Nazir Stackhouse (TEN) and Kyle Kennard (LAC). Stackhouse and Kennard first need 100 rushes.
- **Film queue (under 30 rushes):** Mohamed Kamara (2024, 158, TB), Gabe Hall (2024, UDFA, PHI), Nelson Ceaser (2024, UDFA, FA), Tim Smith (2025, 190, IND), Antwaun Powell-Ryland (2025, 209, CIN), Jah Joyner (2025, UDFA, FA), Ethan Downs (2025, UDFA, KC).

**The frozen test** (sha256 `8277b10e1492…`): after the 2026 class's first season, regress pressure-over-forecast on Hoop G (one-sided p < 0.10). A slope at or below zero retires the flag. Simulated power at the observed slope is 44% for one class and 77% for three.

## Limitations

- **Small samples.** The model uses 77 rushers and the day-3 test 47, with single drill reps. Hoop G's repeatability rests on 10 linemen with two laps, and 91% ran one.
- **Selection.** Earlier work screened drill measures on these same classes. The six-measure correction covers our list, not every idea tried before, so 2026 is the first clean test.
- **Thin NFL tracking.** A median of 17 sampled rushes per lineman, with no blockers or ball. Bend on a snap may follow from beating the tackle rather than cause it.
- **Weak baseline.** Pick + role alone beats the full no-tracking model out of class (R² 0.10 vs −0.02), so gains over it look larger than gains over a scout.
- **Survivorship and roles.** Rates need 30+ rushes. Injuries, scheme and roster moves are not modelled.

## Reproduce

`analysis.py` (about 3 minutes) rebuilds every number, figure and this text from `data/processed`, and writes `numbers.json`, `board.csv` and `frozen_rule.json`.
