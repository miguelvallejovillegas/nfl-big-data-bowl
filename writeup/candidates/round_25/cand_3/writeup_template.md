# Weigh Him First: A Lineman's Combine Movement Reaches Sunday Only Through His Weight

"He plays like he tests" is the oldest bridge between the Combine and the tape, and tracking now lets us measure it directly. We asked whether a player's NFL movement in the first 3 seconds off a stance matches his Combine drill movement, and whether players whose movement matches out-produce their draft slot. For defensive linemen, the group where the Combine carries over best, the answer is no. **Envelope Match** is worth {{h_match_season}} pressures per 450 rushes per sd [95%: {{h_match_season_lo}}, {{h_match_season_hi}}] beyond pick and role (n = {{dl_n}}, every class scored by a model it never saw). The carry-over that does exist is body mass. Weight ranks a lineman's NFL movement at ρ = {{rho_w_game}}; with weight held fixed, his Combine movement ranks it at {{rho_d_game_w}}.

This is a bounded null, not a new signal. It is one movement type, 3 seconds from a stance, measured on about 20 sampled snaps per man.

- **Scout, Monday:** when a drill clip says a lineman "explodes" or "bends", rank him against men within about 10 lb of his weight before you believe it. Most of the apparent carry-over is the scale talking.
- **Analytics, Monday:** do not buy or build a drill-to-game "translation" score for linemen. The plausible upside is at most {{h_match_season_hi}} pressures a season, under our pre-set bar of {{margin_season}} pressures.

![Figure 1. Each point is a defensive lineman with 8+ randomly sampled NFL snaps. A: NFL start envelope against Combine weight. B: the Combine envelope against the NFL envelope, each ranked after weight is removed.](figures/fig1_weight_not_drill.png)

## Why this question

Front offices are now being sold "translation" metrics that compare Combine tracking with game tracking. If a man's drill movement shows up on Sunday, a match score could flag rookies who are already playing to their tools and veterans who are not. The 2023–25 classes are the first with tracked Combine drills and tracked NFL snaps for the same men, so the bridge can be tested rather than assumed.

## The metric in coach terms

**Movement type:** the first 3.0 s of a start from a stance. On the Combine side that means the position drills that start from a stance (linemen: the pass-rush drill and Run-the-Hoop), timed from first movement. On the NFL side it means randomly sampled snaps from the snap, cut at the throw or sack ({{n_sampled_snaps}} snaps for {{n_game_frames_snaps}} players across all groups, median {{dl_snaps_median}} per lineman).

**Envelope:** how hard he pushes off (forward acceleration) and how hard he turns (sideways g) when he is moving at jog, run and stride speeds (1–2.5, 2.5–4 and 4–5.5 yd/s). We take his top-10% effort in each of the six cells. Each side is ranked against his position group, and Combine cells are ranked within Combine year.

**Envelope Match** is the negative mean gap, in sd, between his six Combine cells and his six NFL cells. Zero means he moves on Sunday exactly where he tested relative to his peers.

Figure 2 shows the movement itself. Light linemen push off harder than heavy ones at every speed, in drills and in games. That shared ordering produces the raw Combine-to-NFL correlation of {{rho_d_game}} [95%: {{rho_d_game_lo}}, {{rho_d_game_hi}}]. Inside a weight class the order breaks. Byron Young ({{pair0_wt}} lb, pick {{pair0_pick}}) tested {{pair0_dz}} sd on the drill envelope and Jalyx Hunt ({{pair1_wt}} lb, pick {{pair1_pick}}) tested {{pair1_dz}}. On Sundays they sit at {{pair0_gz}} and {{pair1_gz}}, and they pressure on {{pair0_y}}% and {{pair1_y}}% of rushes.

![Figure 2. Top-10% acceleration at each speed, by weight third (thick) and for two edges of the same weight (thin, dots). Dashed lines are Combine drills, solid lines are sampled NFL snaps.](figures/fig2_envelopes.png)

## Which position group?

We fixed the rule before looking at production: study the group with the highest lower 95% bound on weight-held translation, among groups with 50+ players. Figure 3A shows the screen. No group clears zero once weight is held fixed. Defensive line is chosen at {{s_DL_wt}} [{{s_DL_lo}}, {{s_DL_hi}}] (n = {{s_DL_n}}), against a noise ceiling of {{s_DL_ceil}}. The ceiling is set by how well a man's envelope agrees with itself: {{s_DL_relg}} between halves of his NFL snaps, but only {{s_DL_reld}} between his two Combine drills. Receivers are the runner-up at {{s_WR_wt}}.

Figure 3B runs the production test in all five groups, so the choice of group cannot hide a result. Envelope Match moves production by a tenth of an sd per sd or less in every group, and every interval includes zero.

![Figure 3. A: Combine-to-NFL rank correlation of the six envelope cells (mean), raw and with Combine weight held fixed, with each group's noise ceiling (not shown for TE, whose split-half reliability is negative). B: Envelope Match against each group's per-play production beyond the held-out pick + role forecast. Players with 8+ sampled snaps; 95% class-stratified bootstrap.](figures/fig3_screen.png)

## Does matching explain pass rush beyond the pick?

**Outcome:** career pressure rate through 2025, shrunk toward the {{press_mean}}% mean, for linemen with 30+ rushes. **Baseline without tracking:** log draft pick plus edge share (the share of snaps lined up outside). **Validation:** leave one class out ({{dl_n_per_class}} linemen), with a class-stratified bootstrap that refits every fold, and a placebo that shuffles the measure {{nperm}} times within class and role.

| Measure, per sd (n = {{dl_n}}) | Pressures per 450 rushes beyond pick + role [95%] | Held-out R², pick + role → with measure | Shuffle p |
|---|---|---|---|
| **Envelope Match (the test)** | **{{+:h_match_season}} [{{+:h_match_season_lo}}, {{+:h_match_season_hi}}]** | {{h_r0}} → {{h_match_r1}} | {{perm_match}} |
| Combine envelope alone | {{+:h_drill_env_season}} [{{+:h_drill_env_season_lo}}, {{+:h_drill_env_season_hi}}] | {{h_r0}} → {{h_drill_env_r1}} | {{perm_drill_env}} |
| Sunday surplus (NFL minus Combine, signed) | {{+:h_surplus_season}} [{{+:h_surplus_season_lo}}, {{+:h_surplus_season_hi}}] | {{h_r0}} → {{h_surplus_r1}} | – |
| NFL envelope, same seasons | {{+:h_game_env_season}} [{{+:h_game_env_season_lo}}, {{+:h_game_env_season_hi}}] | {{h_r0}} → {{h_game_env_r1}} | {{perm_game_env}} |
| NFL envelope 2023–24 → 2025 rate, beyond 2023–24 rate (n = {{nx_n}}) | {{+:nx_season}} [{{+:nx_season_lo}}, {{+:nx_season_hi}}] | – | – |

*Table 1. Held-out R² is the share of rusher-to-rusher spread explained in a class the model never saw. One sd of pressure rate is about {{sd_y_season}} pressures a season.*

**Matching does not pay.** The slope is slightly negative. None of the {{nboot}} bootstrap fits reaches our pre-set bar of {{margin}} rate points ({{margin_season}} pressures a season, {{margin_epa}} expected points at {{epa_press}} EPA per pressure). The result holds against a fuller no-tracking baseline that adds weight and the NGS athleticism score ({{f_match}} rate points [{{f_match_lo}}, {{f_match_hi}}]). Match does not track weight ({{match_w}}), role ({{match_role}}) or the number of sampled snaps ({{match_snaps}}), so it is not simply noise from small samples.

**Why it cannot work:** an unsigned match punishes a rusher for beating his Combine. Will Anderson has the lowest Match of all {{dl_n}} linemen, and Jared Verse is number {{verse_rank}} from the bottom among the {{n_big}} with 200+ rushes. Both move far better on Sundays than they tested, and both beat their forecast by 10+ pressures a season (Figure 4A).

**The Sunday envelope itself** lines up with pressure in the same seasons (+{{h_game_env_season}}, shuffle p = {{perm_game_env}}). The link is weak, though. The R² gain of {{h_game_env_r2gain}} has a 95% interval of [{{h_game_env_r2gain_lo}}, {{h_game_env_r2gain_hi}}] across rushers and [{{cb_game_env_lo}}, {{cb_game_env_hi}}] when {{n_teams}} NFL teams are resampled. It also fails the forward test: measured on 2023–24 snaps, it does not predict 2025 pressure once the 2023–24 rate is known, and the estimate points the wrong way (n = {{nx_n}}, Table 1). Prior pressure rate already carries that information (ρ = {{nx_prior_rho}} season to season).

![Figure 4. Pressures per 450 rushes beyond each lineman's held-out pick + role forecast. A: against Envelope Match. B: against his NFL envelope on sampled snaps from the same seasons. Lines show held-out slopes; boxes give the 95% interval and shuffle p.](figures/fig4_pressure.png)

## Six named linemen

By a fixed rule: the three lowest and three highest Envelope Match scores among the {{n_big}} linemen with 200+ career rushes.

| Lineman | Class, pick | Weight | Combine envelope (sd) | NFL envelope (sd) | Match (sd) | Pressures a season vs forecast |
|---|---|---|---|---|---|---|
| {{T:t2.0.name}} ({{T:t2.0.team}}) | {{T:t2.0.cls}} {{T:t2.0.pick}} | {{t2_0_wt}} | {{t2_0_dz}} | {{t2_0_gz}} | {{t2_0_mz}} | {{+:t2_0_beyond}} |
| {{T:t2.1.name}} ({{T:t2.1.team}}) | {{T:t2.1.cls}} {{T:t2.1.pick}} | {{t2_1_wt}} | {{t2_1_dz}} | {{t2_1_gz}} | {{t2_1_mz}} | {{+:t2_1_beyond}} |
| {{T:t2.2.name}} ({{T:t2.2.team}}) | {{T:t2.2.cls}} {{T:t2.2.pick}} | {{t2_2_wt}} | {{t2_2_dz}} | {{t2_2_gz}} | {{t2_2_mz}} | {{+:t2_2_beyond}} |
| {{T:t2.3.name}} ({{T:t2.3.team}}) | {{T:t2.3.cls}} {{T:t2.3.pick}} | {{t2_3_wt}} | {{t2_3_dz}} | {{t2_3_gz}} | {{t2_3_mz}} | {{+:t2_3_beyond}} |
| {{T:t2.4.name}} ({{T:t2.4.team}}) | {{T:t2.4.cls}} {{T:t2.4.pick}} | {{t2_4_wt}} | {{t2_4_dz}} | {{t2_4_gz}} | {{t2_4_mz}} | {{+:t2_4_beyond}} |
| {{T:t2.5.name}} ({{T:t2.5.team}}) | {{T:t2.5.cls}} {{T:t2.5.pick}} | {{t2_5_wt}} | {{t2_5_dz}} | {{t2_5_gz}} | {{t2_5_mz}} | {{+:t2_5_beyond}} |

*Table 2. Envelopes are ranked against all defensive linemen; Match is in sd (higher = closer to his Combine).*

On average the three low matchers ran {{+:big_lowmatch_beyond}} pressures a season against their forecast and the three high matchers {{+:big_highmatch_beyond}}. Low Match covers three different stories: {{T:t2.0.name}} moves above his test, {{T:t2.1.name}} below it, and {{T:t2.2.name}} tested far below his weight and plays at it. Eric Watts and Ashton Gillotte move on Sunday as they tested and still miss their forecast. Matching his test says nothing about whether a rusher wins.

## What a team should do with this

1. **Grade drill movement within weight.** Our raw drill-to-NFL correlation of {{rho_d_game}} looks like a real bridge, but weight alone does better ({{rho_w_game}}). Compare a 250-lb edge with 250-lb edges.
2. **Do not pay for translation scores.** The test spans five groups and three baselines, and no version clears zero.
3. **Use NFL snaps to judge NFL movement.** A man's NFL envelope agrees with itself at {{s_DL_relg}} across halves of 20 snaps. His two Combine drills agree at {{s_DL_reld}}. Once a lineman has played, his snaps beat his Combine, and his pressure rate beats both.

## Limitations

- **One movement type, generic cells.** These are 3-second starts from a stance, not specific moves such as a hoop dip or a rip. A narrower drill trait could still carry over, and this test does not rule that out.
- **Thin NFL tracking.** About 20 random snaps per man, with no blockers, ball or opponents in the data, so the envelope is self-referenced and cannot see the move that beat the tackle.
- **Single Combine reps.** Most men ran each drill once, and the two DL drills agree with each other at only {{s_DL_reld}}. That noise caps any translation at {{s_DL_ceil}}, so the weight-held interval up to {{s_DL_hi}} still allows a modest true link.
- **Small groups.** TE has {{s_TE_n}} players, and the forward test has {{nx_n}} linemen.
- **Survivorship.** Only players with 8+ sampled snaps and 30+ rushes enter, which leaves out men who never played.

## Appendix

- **Cells, DL, weight held (partial ρ):** push-off {{cell_al_b0}} / {{cell_al_b1}} / {{cell_al_b2}} and turning {{cell_la_b0}} / {{cell_la_b1}} / {{cell_la_b2}} at the three speed bands (n = {{rho_n}}). Weight also ranks the Combine envelope ({{rho_w_drill}}).
- **Fuller baseline** (pick, edge share, weight, NGS athleticism; held-out R² {{f_r0}}): Combine envelope {{f_drill_env}} [{{f_drill_env_lo}}, {{f_drill_env_hi}}], NFL envelope {{f_game_env}} [{{f_game_env_lo}}, {{f_game_env_hi}}], signed surplus {{f_surplus}} [{{f_surplus_lo}}, {{f_surplus_hi}}] rate points per sd. Against pick alone, Match scores {{h_match_pick1}} rate points per sd.
- **Match R² gain:** {{h_match_r2gain}} [{{h_match_r2gain_lo}}, {{h_match_r2gain_hi}}]; team resampling [{{cb_match_lo}}, {{cb_match_hi}}]. Shuffle band for the slope: [{{perm_band_lo}}, {{perm_band_hi}}] rate points.
- **EPA per pressure:** {{epa_press}} [{{epa_lo}}, {{epa_hi}}] on {{epa_n}} randomly sampled DL pass plays.
- **Other groups' reliability (NFL halves / Combine drills):** OL {{s_OL_relg}} / {{s_OL_reld}}, WR {{s_WR_relg}} / {{s_WR_reld}}, TE {{s_TE_relg}} / {{s_TE_reld}}, DB {{s_DB_relg}} / {{s_DB_reld}}.

`analysis.py` (about 3 minutes) rebuilds every number, figure and this text from `data/processed`.
