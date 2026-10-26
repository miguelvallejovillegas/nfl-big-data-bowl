# Release Match: Receivers Do Not Release on Sundays the Way They Released at the Combine

A receiver's release in 13 Combine route drills ranks his NFL release only weakly (ρ = 0.16, 75 receivers). Receivers whose NFL release matches their drill release do not out-produce their draft slot: +0.09 yards per route run per sd (90% interval −0.07 to +0.26, 62 receivers, held-out draft classes).

**Release Match**, in one line: how closely a receiver's speed and acceleration in the first 1.5 seconds after he starts moving in the NFL track the same 1.5 seconds in his Combine route drills, each read against his peers. Zero would mean he releases on Sundays exactly as he tested. More negative means a wider gap.

**What to do on Monday**
- **Scouts:** do not grade the release from Combine route drills. They rank the NFL release no better than the official 40, and body weight ranks it better (Table 1). Grade release against press on college tape.
- **Coaches:** the release is worth repping. On the same receiver, a release 1 sd quicker than his norm comes with 0.09 yards more separation at the throw. Press costs speed off the line.
- **Personnel:** "plays like he tested" is not a tiebreaker. Brian Thomas Jr. has one of the widest drill-to-NFL gaps in the sample and beat his forecast by +0.78 yards per route.

![Figure 1. Five-group screen. A: rank correlation between each player's Combine start profile and his NFL start profile (90% bootstrap CI); orange tick = noise ceiling, the most the two measurements could agree given their own repeatability. B: Match beyond log draft pick on career production (sd per sd, 90% CI). Blue = the group studied.](figures/fig1_screen.png)

## 1. Which position group shows the clearest signal?

Every group starts its NFL snap with a movement the Combine also drills: the offensive line's kick-slide, the defensive line's get-off, the receiver's and tight end's release, and the defensive back's backpedal. For each group I built the same 16-point speed and acceleration profile (0 to 1.5 s after the player first reaches 0.5 yd/s) in every matching drill rep and on every random NFL snap in the sample. Each point was read against peers (Combine: same drill, year and rep; NFL: same alignment, and for receivers the same route).

A translation test only means something if both sides repeat. The rule, fixed before the production tests: study the group whose noise ceiling is at least 0.5 and that has an individual production measure, choosing the highest lower bound on translation. Figure 1 shows the result.

- **Offensive line:** the two Combine kick-slide reps are different sets, one short and one deep, and they disagree (ρ = −0.11). Within tackles, guards or centers, NFL set speed on the sampled snaps is not a stable trait (split-half reliability 0.00). Translation cannot be measured here.
- **Defensive line:** both sides repeat (ceiling 0.36), but the drill get-off does not carry (0.03) once edge and interior rushers are compared only with their own kind.
- **Defensive backs:** the backpedal carries as well as anything (0.19 [0.00, 0.38]). The data has only on-field team EPA for DBs, not an individual coverage stat, so the production question cannot be asked.
- **Tight ends:** 27 players, ceiling 0.
- **Receivers:** both sides repeat (Combine 0.64, NFL 0.69, ceiling 0.66), and production is measured per player. Receivers are the group studied.

No group's Match predicts production beyond pick in the screen (smallest p = 0.15).

## 2. The release, measured the same way twice

The 106 receivers ran a median of 13 tracked route-drill reps (1,315 in total). 75 of them have NFL tracking, a median of 19 random non-screen routes each (1,320 snaps). Aligning both settings on the 0.5 yd/s crossing removes the ~0.3 s of standing time at the front of the Combine clip and any pre-snap stance differences.

On average the NFL release is quicker early: 3.3 yd/s at 0.5 s against 2.5 in the drill, and 5.3 against 4.8 at 1.0 s. By 1.5 s the drill rep has caught up (6.4 against 6.1), because the drill has no defender and no stem to set (Figure 2A).

![Figure 2. A: mean speed after first movement, Combine route drills vs NFL routes (random snaps, screens excluded); band = middle 50% of reps. Grey = release window, 0.3 to 1.2 s. B: each receiver's release speed in the window, Combine vs NFL, in sd against peers; n = 75.](figures/fig2_release.png)

| What ranks the NFL release (0.3–1.2 s, sd vs same route and alignment) | ρ [90% CI] | n |
|---|---|---|
| Combine route-drill release, same window | 0.16 [−0.03, 0.35] | 75 |
| Combine release acceleration, first 0.5 s | −0.02 [−0.22, 0.17] | 75 |
| Official 40 time (faster = higher) | 0.17 [−0.03, 0.36] | 69 |
| Official 10-yard split | 0.10 [−0.12, 0.31] | 69 |
| Weight (heavier = higher) | **−0.31 [−0.47, −0.14]** | 75 |

*Table 1. Ceiling for the drill row: 0.66. The drill reaches 25% of it.*

The drill and the 40 are nearly unrelated to each other (ρ = −0.03), so each adds a little on its own. With both in one model, the drill is worth 0.16 sd of NFL release per sd (p = 0.18) and the 40 is worth 0.12. The plain scale says more than either: lighter receivers get off the line faster on Sundays.

## 3. Does the release matter on Sundays?

Before asking whether matching the drill matters, I checked that the NFL release itself means something. On 1,274 random snaps from 75 receivers, I compared each release with the same receiver's own average. The model holds cushion, route and alignment fixed, with errors clustered by player.

- A release 1 sd quicker than his norm comes with **+0.09 yards of separation at the throw** [0.00, 0.19], against a mean of 3.0 yards.
- Each yard of cushion adds 0.08 yd/s to the release [0.07, 0.10]. A corner pressed 5 yards tighter takes about 0.4 yd/s off it. That is press doing its job, and it shows the measure reads real football.

So the release matters a little, and the defender shapes it. The Combine drill has no defender.

## 4. Does translation explain production beyond pick?

The outcome is career yards per route run, weighted by routes, for receivers with 100+ routes (n = 62, mean 1.27, sd 0.59). Each draft class is forecast from a model fitted on the other two classes. The held-out miss is then regressed on Release Match (sd units). Intervals come from 2,000 receiver resamples within class, with every forecast refitted in every resample. I report three baselines side by side rather than choosing one:

| Baseline (no tracking) | Held-out R² [90%] | Release Match, yds/route per sd [90%] | R² change when added |
|---|---|---|---|
| Log pick | 0.02 [−0.34, 0.26] | +0.10 [−0.04, 0.25] | −0.13 |
| Log pick + slot role | 0.04 [−0.39, 0.36] | +0.09 [−0.03, 0.24] | −0.13 |
| Full: + official 40, weight, NGS athleticism | −0.03 [−0.95, 0.32] | **+0.09 [−0.07, 0.26]** | −0.09 |

*Table 2. Five receivers without a 40 time take the training-class median.*

The sign leans positive: 80% of resamples are above zero. But adding Release Match makes held-out forecasts worse in every baseline. The class-by-class slopes disagree (2023 +0.30, n = 28; 2024 −0.12, n = 19; 2025 +0.04, n = 15). Neither half of the match helps alone: the Combine release gives +0.06 and the NFL release −0.02 against the full baseline (Figure 4A).

![Figure 4. A: held-out slope of each measure on yards per route run, three baselines, 90% class-stratified bootstrap; grey band = ±0.15 practical margin. B: simulated share of studies of this size that detect a true effect, at Release Match's measured reliability and at 0.8.](figures/fig4_production.png)

**Can this sample see an effect at all?** Release Match repeats at only 0.29: one half of the drill reps plus one half of the NFL snaps, against the other halves. I simulated the study 2,000 times with 62 receivers, that reliability and the observed residual spread (0.62). A true effect the size of the margin (0.15 yards per route per sd, about 12% of the average receiver) is detected only 27% of the time. On average it shows up as 0.08, close to what we observed. At a reliability of 0.8 the detection rate would be 52%. So the honest reading is **not shown, and not ruled out**: 24% of resamples lie beyond the margin.

## 5. Eight receivers, drill vs Sunday

![Figure 3. Player cards: each receiver's release speed profile in sd against peers, Combine drills (orange) vs NFL routes (blue), 90% bands. Top: the four closest Release Matches among the 62. Bottom: the four widest gaps. Grey card = his drill-minus-NFL gap is not clear of zero at half of the 16 points.](figures/fig3_cards.png)

The closest matches (top row) are Tank Dell (−0.14), Ricky Pearsall, Jayden Higgins and Jaxon Smith-Njigba. Their two curves sit on top of each other, and at none of the 16 points is the gap clear of noise. All four beat their forecast; Smith-Njigba is at 2.33 yards per route (+0.49).

The widest gaps (bottom row) share one shape. In 10 of the 15 receivers with a clear gap, the drill release is the faster one:
- **Dont'e Thornton** (−1.12): +1.3 sd in the drill, −0.2 on Sundays, 0.59 yards per route (−0.75 vs forecast). A film check is warranted: is press taking his release away?
- **Roman Wilson** (−0.89): +1.0 → +0.1, 1.11 yards per route (−0.18).
- **Brian Thomas Jr.** (−0.86): +0.5 → −0.7, and still 2.08 yards per route (+0.78). He releases under control and wins later in the route. A wide gap is not a warning sign.
- **Devaughn Vele** (−0.80): +0.6 → −0.2, +0.13 vs forecast.

These are readings of eight receivers, not model output.

## Limitations

- **Small and sampled.** 62 receivers with production, 75 with both profiles, a median of 19 random NFL routes each. Release Match repeats at 0.29, so the production test has low power (Figure 4B).
- **No ball, no defenders.** Release speed mixes skill, press, route depth and play-action timing. Cushion, route and alignment are controlled; press technique is not.
- **The choice rule came after the screen was built.** It was fixed before any held-out production run. Defensive backs translate as well as receivers but have no individual coverage outcome here.
- **One definition.** The 0.5 yd/s onset, the 1.5 s profile and the 0.3–1.2 s window were set once and not tuned. The Match score has the part explained by sample size (number of reps and snaps) removed. That part is small (ρ = −0.11 with the raw score).
- **Production is career-to-date** for three classes, so 2025 receivers have one season.
- **The no-tracking forecasts are weak out of class** (held-out R² −0.03 to 0.04), so Release Match is measured against a low bar, and it still does not clear it.

## Appendix: method notes

- Combine profiles use only PLAYER frames. Each point is z-scored within drill × Combine year × rep number, which removes year-to-year protocol shifts.
- NFL profiles use random-sample snaps only, from the snap frame, with the first movement within 1.0 s of the snap. Each point is z-scored within alignment × route (cells of 30+ snaps).
- Release Match = −mean |Combine z − NFL z| over 16 speed and 16 acceleration points. Its correlation with the size of the drill release (−0.48) is larger than with the size of the NFL release (−0.28), so the widest gaps come mostly from extreme drill reps.
- `analysis.py` rebuilds every number and figure from `data/processed` in about two minutes. `inputs.csv` lists each receiver's profile levels, Match, forecast miss and production.
