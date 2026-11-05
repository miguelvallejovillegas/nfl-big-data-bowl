# Combine Route-Drill Cuts Do Not Carry to NFL Separation

**A receiver who cuts 1 sd harder in Combine route drills is no more open in the NFL: {ho} inches of separation over expected per sd (90% interval {ho_lo} to {+ho_hi}), across {n_main} receivers from draft classes the model never saw.**

**Break G is how much speed a receiver sheds into his plant and how hard he drives out**, counted as the yards per second of velocity he changes in the second around his first cut. Receivers 1 sd apart differ by {bg_yds_sd} yd/s. On film, read three speeds: in, slowest, out (Table 3).

Why receivers and tight ends: route-breaking is the skill the Combine films most directly. Ten route drills are tracked at 10 Hz, and we grade every cut with the same code we run on NFL snaps. Why now: clubs already price it. A receiver 1 sd higher in Break G goes {price}% earlier in the draft [{price_lo}, {price_hi}] at the same position ({price_n} receivers). The 2023 class's rookie deals expire after this season.

- **Separation over expected (SOE)**: inches more open at the throw than Next Gen Stats separation predicts from alignment, cushion and season, over full regular seasons ({n_seasons} player-seasons, median {routes_med} career routes).
- **Combine years**: every rep is compared only with the same drill in the same year, so protocol changes do not leak in.
- **Samples**: {n_combine} receivers with a drill score; {n_main} with NFL separation (season tests); {f1_n} of those with 8+ sampled NFL breaks and an official 40 (transfer tests).

![Figure 1. Player cards for the three receivers of Table 3, the hardest, middle and softest drill cutters among {ex_n} wide receivers with 200+ career routes and 5+ sampled NFL breaks. Top: speed through the cut on route families he ran in both settings, aligned on the slowest point. Bottom: his Break G and its three parts in sd against peers. The drill order not surviving means that across {f1_n} receivers, drill and NFL Break G rank-correlate {bgc}: a receiver's drill rank says nothing about where his Sunday breaks rank.](figures/fig1_cuts.png)

## 1. Break G adds nothing to a no-tracking forecast

The baseline uses log pick, official 40, weight, NGS athleticism score and a tight-end flag. It was fixed before the test, fitted on two draft classes and scored on the third. Intervals are 90%, from resampling receivers within class.

| Held-out test | Estimate [90%] | P(\|effect\| < 3 in) | n |
|---|---|---|---|
| No-tracking baseline R² (error per receiver) | {bl_r2} ({bl_rmse} in) | | {n_main} |
| Change in R² from adding Break G | {dr2} [{dr2_lo}, {dr2_hi}] | | {n_main} |
| **Break G, inches per sd** | **{ho} [{ho_lo}, {+ho_hi}]** | **{rope_bg}%** | {n_main} |
| Break G, weighted for who reached the NFL | {ipw} [{ipw_lo}, {+ipw_hi}] | | {n_main} of {inc_all} |
| Official 3-cone, inches per sd (no tracking) | {a_cone} [{a_cone_lo}, {+a_cone_hi}] | {rope_cone}% | {a_cone_n} |
| Official short shuttle, inches per sd (no tracking) | {a_shut} [{a_shut_lo}, {+a_shut_hi}] | {rope_shut}% | {a_shut_n} |

The error is the root-mean-square miss on career SOE, each receiver weighted by his routes: {bl_rmse} in, the size of the spread itself. Counting every receiver once it is {bl_rmse_unw} in, because short careers are noisy. Power is {pow3}% at ±{sesoi} in; 80% only at {mde} in. So the third column states the null directly: with a flat prior, the Break G effect has a {rope_bg}% probability of being smaller than 3 inches either way. That is likely small, not proven zero. Cut quality as the Combine measures it, tracked or timed, does not price separation.

![Figure 2. Held-out slopes of season SOE per sd with 90% intervals, against the ±3 inch margin fixed before testing. None of the {n_tests} excludes zero. Official-time rows are the subset of receivers who ran the drill.](figures/fig2_forest.png)

## 2. Why: the drill grades a different cut

| What is compared | Rank corr. [90%] | n |
|---|---|---|
| Official 40 → NFL top speed | {spc} [{spc_lo}, {spc_hi}] | {f1_n} |
| Same, random plays only | {s_spd} [{s_spd_lo}, {s_spd_hi}] | {s_spd_n} |
| **Combine → NFL Break G** | **{bgc} [{bgc_lo}, {bgc_hi}]** | {f1_n} |
| Same, random plays only | {s_rnd} [{s_rnd_lo}, {s_rnd_hi}] | {s_rnd_n} |
| Same, top-EPA plays only | {s_epa} [{s_epa_lo}, {s_epa_hi}] | {s_epa_n} |
| Same route family, NFL breaks entered at drill speed | {fastn} [{fastn_lo}, {fastn_hi}] | {fastn_np} |
| Same route family, entry speed removed on both sides | {adj} [{adj_lo}, {adj_hi}] | {adj_np} |

Speed carries; the cut does not. Noise is not the reason: given both measures' reliabilities, a true 0.5 would still read {exp50}.

The drill demands a cut at {vin_hitch_d} yd/s and the NFL asks for {vin_hitch_n} (hitch family). {ex_faster} of {ex_shared} wide receivers enter the drill cut faster than the same NFL route, by a median {vin_gap} yd/s, and Break G rises with entry speed (rank corr. {vin_rho_d} in drills, {vin_rho_n} in games). We tested that explanation. Keeping only NFL breaks entered at drill speed ({fast_pct}% of them) moves the transfer from {fam} to {fastn}. Removing entry speed from both sides gives {adj}. Both stay well short of {exp50}. Entry speed is part of the gap, not all of it: even at matched speed the drill does not rank receivers the way Sunday breaks do.

Half of each receiver's sampled plays are his top-EPA plays, which biases these rows. The two halves pull in opposite directions: {s_epa} on top-EPA plays, {+s_rnd} [{s_rnd_lo}, {s_rnd_hi}] on random plays. So the sample may hide a small transfer, but the season tests in Section 1 use full-season NGS and find no separation from it.

{{CARDS}}

Kayshon Boutte, the hardest drill cutter, barely slows on Sundays. Ryan Flournoy, in the middle of the drill order, sheds the most. Career SOE does not follow the drill order either: the eight hardest and eight softest drill cutters average {+x8_hard_soe} and {x8_soft_soe} inches.

![Figure 3. The same {f1_n} receivers in both panels, with least-squares line and 90% bootstrap band. The noise ceiling ({ceil}) is what a perfect true link would show through both measures' noise; a true 0.5 would show {exp50}. Dotted diagonal: drill z equals NFL z. The four receivers furthest from the centre in B are named.](figures/fig3_transfer.png)

## 3. What a Combine staff and a position coach can do

Grade at least {cuts70} cuts per receiver (today's median is {why_reps}). Keep the slant, re-stem the curl, comeback and hook at NFL hitch speed (about {vin_hitch_n} yd/s, a controlled three-quarter stem), and give Break G zero weight until a re-stemmed drill transfers. For a position coach: rep breaks at game tempo, because nothing here says a harder full-speed plant buys separation.

## 4. The 2025 day-3 receivers: a coin-flip film aid

These are the day-3 or undrafted wide receivers who were more open than expected in 2025. The table shows what each name's drill and NFL breaks look like and what 2026 result would drop him off the list. Read it as a film aid, not a ranking:

{{DECISION}}

Only {cells30} of {cells} man/zone and slot/wide cells reach 30 sampled snaps. The splits themselves are in the appendix.

## Bottom line

1. **Take Break G out of the price.** Clubs pay {price}% in pick position per sd of it and get no separation back.
2. **Fix the drill before grading it.** Grade {cuts70}+ cuts, re-stem the hitch family at NFL entry speed, and give it zero weight until it transfers.
3. **On day 3, draft order stands.** Across {r_n} day-3 receivers, pick plus last season's SOE called the next season above or below zero {r_f1}% of the time, against {r_f0}% for pick alone ({r_d} points [{r_d_lo}, {r_d_hi}]). That is too few receivers to order names either way, which is why Table 4 is a film aid.

## Limitations

- **Not blind.** All three classes were seen while this was built. The ±3 inch margin and the baseline were fixed before each test.
- **Power.** {pow3}% at the margin, and harm is not excluded (lower bound {ho_lo} inches per sd).
- **Selection.** {n_main} of {inc_all} drill-scored receivers logged NFL routes. Weights for who reached the NFL give {ipw}, but they cannot repair receivers who never played.
- **Game tracking is a sample.** Up to 40 plays per player, half top-EPA, with no defenders or ball. This affects Tables 2–4 and Figures 1 and 3, not the season tests, which use full-season NGS.
- **Small benchmarks.** Few receivers run the official 3-cone ({a_cone_n}) or shuttle ({a_shut_n}).

## Appendix

`verify.py` rebuilds both Break G pipelines from the raw frames and refits the held-out model a second way, with its own bootstrap seeds. It reproduces 124 of 124 checked numbers, including both error definitions: 68 exactly, 56 resampled within tolerance. `appendix.md` holds the reliability table, error definitions, measurement-error and multiplicity checks, the protocol-subset figure and the man/zone and slot/wide splits.
