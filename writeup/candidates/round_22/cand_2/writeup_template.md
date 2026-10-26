# Combine Route-Drill Cuts Do Not Carry to NFL Separation

**A receiver who cuts 1 sd harder in Combine route drills is no more open in the NFL: {ho} inches of separation over expected per sd (90% interval {ho_lo} to {+ho_hi}), across {n_main} receivers from draft classes the model never saw.**

**Break G is how much speed a receiver sheds going into his plant and how hard he drives out, measured over the second around his first cut.** On film, read three speeds: in, slowest, out. The typical drill rep sheds {shed_d_mph} mph into the plant; the typical NFL break, {shed_n_mph} mph.

Why receivers and tight ends: route-breaking is the skill the Combine films most directly. Ten route drills are tracked at 10 Hz, and we grade every cut with the same code we run on NFL snaps (slant, out, hitch, in and corner families). Why it matters: clubs pay for it. A receiver 1 sd higher in Break G goes {price}% earlier in the draft [{price_lo}, {price_hi}] at the same position ({price_n} receivers), yet, as shown below, buys no separation.

- **Separation over expected (SOE)**: inches more open at the throw than Next Gen Stats separation predicts from alignment, cushion and season, over full regular seasons ({n_seasons} player-seasons, median {routes_med} career routes).
- **Combine years**: every rep is compared only with the same drill in the same year. Break G is timed from each rep's own slowest point, so the 40 clip's 0.3 s start offset never touches it; the 40 enters only as the official time.
- **Samples**: {n_combine} receivers with a drill score; {n_main} with NFL separation (season tests); {f1_n} with 8+ sampled NFL breaks and an official 40 (transfer tests).

![Figure 1. Top: three receivers' speed through the cut, Combine drill (orange) vs NFL (blue), on route families he ran in both, aligned on the slowest point. Bottom: all {ex_n} eligible wide receivers ordered by drill Break G; the NFL squares do not follow the drill order (rank corr. {ex_rho}).](figures/fig1_cuts.png)

## 1. Break G adds nothing to a no-tracking forecast

The baseline (log pick, official 40, weight, NGS athleticism score, tight-end flag) was fixed before the test, fitted on two draft classes and scored on the third, route-weighted. Intervals are 90%, resampling receivers within class.

| Held-out test | Estimate [90%] | P(\|effect\| < 3 in) | n |
|---|---|---|---|
| Change in R² from adding Break G to the baseline | {dr2} [{dr2_lo}, {dr2_hi}] | | {n_main} |
| **Break G, inches per sd** | **{ho} [{ho_lo}, {+ho_hi}]** | **{rope_bg}%** | {n_main} |
| Official 3-cone, inches per sd (no tracking) | {a_cone} [{a_cone_lo}, {+a_cone_hi}] | {rope_cone}% | {a_cone_n} |
| Official short shuttle, inches per sd (no tracking) | {a_shut} [{a_shut_lo}, {+a_shut_hi}] | {rope_shut}% | {a_shut_n} |

The baseline itself explains nothing held out (R² {bl_r2}): its typical miss, {bl_rmse} in, equals the {soe_sd} in spread of career SOE between receivers (fitted without route weights it is {rm_fu_sw} in; the appendix grid shows all four recipes). A pick-only baseline ({r2_pick}) and a ridge baseline with a nested penalty ({r2_ridge}) do no better, so the zero is not an artefact of the five covariates. Break G has a {rope_bg}% probability (flat prior) of being smaller than 3 inches either way; power at ±{sesoi} in is {pow3}%. Resampling receivers across classes instead of within gives [{ho_pool_lo}, {+ho_pool_hi}]. Of the {n_tests} slopes in Figure 2, none excludes zero; the smallest Holm-adjusted p is {holm_min}. Likely small, not proven zero.

![Figure 2. Held-out slopes of season SOE per sd, 90% intervals, against the ±3 inch margin fixed before testing. Official-time rows use only receivers who ran the drill.](figures/fig2_forest.png)

## 2. Why: the drill grades a different cut

| What is compared | Rank corr. [90%] | n |
|---|---|---|
| Official 40 → NFL top speed | {spc} [{spc_lo}, {spc_hi}] | {f1_n} |
| **Combine Break G → NFL Break G** | **{bgc} [{bgc_lo}, {bgc_hi}]** | {f1_n} |
| Official shuttle → NFL Break G (no tracking) | {tr_shut} [{tr_shut_lo}, {tr_shut_hi}] | {tr_shut_n} |
| Same route family, NFL breaks entered at drill speed | {fastn} [{fastn_lo}, {fastn_hi}] | {fastn_np} |

Speed carries; the cut does not, tracked or timed. Noise is not the reason: through both measures' noise a true 0.5 would still read {exp50}.

The drill asks for a cut at {vin_hitch_d} yd/s ({vin_hitch_d_mph} mph) on the hitch family; the NFL asks for {vin_hitch_n} ({vin_hitch_n_mph} mph). {ex_faster} of {ex_shared} wide receivers enter the drill cut faster than the same NFL route. Keeping only NFL breaks entered at drill speed moves the same-family transfer from {fam} to just {fastn}: entry speed is part of the gap, not all of it. Sampled NFL plays are half top-EPA; the random half alone reads {+s_rnd} [{s_rnd_lo}, {s_rnd_hi}] (appendix).

{{CARDS}}

Read Boutte's rows as film. In drill reps on the route families he also ran on Sundays, he enters the cut at {c_boutte_din} yd/s and sheds {c_boutte_dmph} mph into the plant. (Raw shed depends on the route: hitch-family drills shed more than slants. His #1 rank compares him with peers in the same drill across all ten drills; these rows are only the families he also ran on Sundays.) On Sundays he enters at {c_boutte_nin} and sheds {c_boutte_nmph} mph: he rounds the break. Flournoy, mid-pack in the drill, sheds {c_flournoy_nmph} mph on Sundays, the most of the three. Bobo, the softest drill cutter, sheds {c_bobo_nmph} mph. Their career SOE does not follow the drill order, and neither does the group's: the eight hardest and eight softest drill cutters average {+x8_hard_soe} and {x8_soft_soe} inches.

![Figure 3. A and B: the same {f1_n} receivers; C: those with an official shuttle. Least-squares line with 90% bootstrap band; dotted diagonal: equal z. Ringed in B: the {f3_named} Figure 1 receivers who ran an official 40.](figures/fig3_transfer.png)

## 3. What a Combine staff and a position coach can do

A single graded cut is mostly noise: today's median of {why_reps} cuts gives a reliability of {rel_7x1}; it takes {cuts70} to reach {t70} (appendix curve). So grade at least {cuts70} cuts per receiver, keep the slant, and re-stem the curl, comeback and hook at NFL hitch speed (about {vin_hitch_n} yd/s, a controlled three-quarter stem). That re-stem is a design proposal; this data cannot test it. Until a re-stemmed drill transfers, Break G gets zero weight. For a position coach: rep breaks at game tempo; nothing here says a harder full-speed plant buys separation.

## 4. The 2025 day-3 receivers: a film list, not a ranking

Day-3 or undrafted wide receivers who were more open than expected in 2025, in order of 2025 SOE. The threshold is the SOE over his first 150 routes of 2026 that would pull his estimate below zero. Last season's SOE did not beat pick order at calling the next season on {r_n} day-3 receivers, so this list is for film, not for the board.

{{DECISION}}

## Bottom line for scouts

**Take Break G out of the price.** Clubs pay {price}% in pick position per sd of it and get no separation back; the 40 and draft slot already carry what the Combine can say about getting open.

## Limitations

- **Not blind.** All three classes were seen while this was built; the ±3 inch margin and the baseline were fixed before each test.
- **Power.** {pow3}% at the margin; harm is not excluded (lower bound {ho_lo} inches per sd).
- **Selection and missed seasons.** {n_main} of {inc_all} drill-scored receivers logged NFL routes; weighting for who reached the NFL gives {ipw}. Career SOE is route-weighted, so an injured or inactive season adds no weight rather than a zero. {miss_n} tested receivers missed a season; dropping them gives {ho_full} ({full_n} receivers).
- **Game tracking is a sample.** Up to 40 plays per player, half top-EPA, no defenders or ball. This affects Section 2 and Figures 1 and 3, not the season tests, which use full-season NGS.
- **Small benchmarks.** Few receivers ran the official 3-cone ({a_cone_n}) or shuttle ({a_shut_n}).

## Appendix

`verify.py` rebuilds both Break G pipelines from the raw frames and refits the held-out model a second way, with its own seeds; it reproduces all 124 numbers it checks. The new baseline grid, ridge and pooled-bootstrap rows are computed in `analysis.py` only. `appendix.md` holds the error-recipe grid (route-weighted or not, fitted and scored), the reliability curve, the sampling-half and entry-speed rows, measurement-error checks and the man/zone and slot/wide splits.
