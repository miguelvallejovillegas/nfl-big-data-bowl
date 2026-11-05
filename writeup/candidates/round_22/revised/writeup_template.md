# Run-the-Hoop Bend Puts a Day-3 Edge Rusher on the Film List, Not Higher on the Board

**Run-the-Hoop bend flags day-3 edge rushers worth a film look, but it is not strong enough to move a draft board: +{{ho_hoop_season}} pressures per 450 rushes per sd, 90% interval {{ho_hoop_season_lo}} to {{ho_hoop_season_hi}}, n = {{n_an}}, not distinguishable from chance after trying six drill measures. Against the strongest forecast without tracking (draft pick plus edge role, held-out R² {{r2_pick}}), Hoop G adds {{dr2_phoop}} R², interval adjusted for six measures {{dr2_phoop_alo}} to {{dr2_phoop_ahi}}.**

**Hoop G, in coach terms:** how hard a lineman leans into the tightest half-second of the Run-the-Hoop figure-eight, the dip-and-flatten an edge needs to turn the corner on a tackle. It is speed squared over turn radius, in g, ranked within his Combine class. The median lineman carries {{hoop_mph_median}} mph around a {{hoop_radius_median}}-yard radius: {{hoop_g_median}} g. On a sampled NFL rush his hardest lean comes at {{nfl_mph_median}} mph around a {{nfl_radius_median}}-yard radius: {{nfl_bend_median}} g. Same formula, slower and twice as wide, so we compare within class, never in raw g. Each lap has a left and a right turn; Hoop G keeps the harder one (the left in {{pct_left_harder}}% of laps, median gap {{turn_gap_g}} g), as NFL bend keeps his hardest lean either way.

**Why the hoop, why day 3, why now.** On {{epa_n}} randomly sampled NFL pass plays, a pressured snap cost the offense {{epa_press}} expected points [{{epa_press_lo}}, {{epa_press_hi}}]. The pass-rush drill times get-off, which NGS already measures on every NFL snap, and one burst rep repeats at only {{rel_burst1}}. The hoop isolates bend, which no box score records. {{pct_day3_dl}}% of the {{n_dl}} defensive linemen in the 2023–25 classes went on day 3 or undrafted, where tape is thinnest. And the 2026 class ran the same tracked hoop, so a rule frozen now is graded after Week 18.

**So what for a team**

1. **Film order, not board order.** Open day-3 edge rushers' tape in Hoop G order. Hoop G does not change the board.
2. **Four hoop laps, no burst reps.** One lap has reliability {{rel_turn_lap}}; {{reps_hoop_turn}} reach {{rel_target}}.
3. **A ranked film list** of eight 2024–25 day-3 rushers (Figure 4).

![Figure 1. One Go and one No-go rusher: Combine hoop lap (top) and randomly sampled NFL rushes, first 2 s (bottom), on one yard and colour scale, with Hoop G and NFL bend readouts.](figures/fig1_hero.png)

## Drill turn against NFL turn

- **{{T:h0}} ({{T:h0_team}}).** Hoop: {{h0_hoopg}} g, {{h0_mph}} mph on a {{h0_rad}}-yard turn (+{{h0_hoopz}} sd, top of his class). NFL: {{h0_nflg}} g, {{h0_nflmph}} mph on a {{h0_nflrad}}-yard turn (+{{h0_bendz}} sd); {{h0_over}} pressures a season above forecast.
- **{{T:h1}} ({{T:h1_team}}).** Hoop: {{h1_hoopg}} g, {{h1_mph}} mph on a {{h1_rad}}-yard turn (+{{h1_hoopz}} sd). NFL: {{h1_nflg}} g, {{h1_nflmph}} mph on a {{h1_nflrad}}-yard turn (class average); {{h1_over}} against forecast.

Ingram-Dawkins turns as tight on Sunday as on the hoop, but slowly; Solomon carries speed through a wider arc. To link drill and play directly, **Turn Match** divides a rusher's NFL turn radius by his own hoop radius. Its median is {{turn_match_median}} ({{turn_match_p10}} to {{turn_match_p90}}, 10th–90th percentile, n = {{turn_match_n}}); only {{pct_rush_hooptight}}% of sampled rushes turn as tight as the same man's hoop lap, and tighter hoop turners do not run tighter NFL turns (Table 1). Drill and play share a skill, not a path.

## The NFL end of the chain works; the drill end does not

NFL movement: a median of {{median_random_rushes}} randomly sampled rushes per lineman, measured from his own start. Links control for weight and edge share.

| Link | Partial ρ [90%] | n |
|---|---|---|
| Hoop G → NFL bend (first 2 s) | {{tr_hoop_g_bend}} [{{tr_hoop_g_bend_lo}}, {{tr_hoop_g_bend_hi}}] | {{tr_hoop_g_bend_n}} |
| Hoop turn radius → NFL turn radius | {{tr_rad}} [{{tr_rad_lo}}, {{tr_rad_hi}}] | {{tr_rad_n}} |
| NFL bend → pressure rate | {{gp_g_bend}} [{{gp_g_bend_lo}}, {{gp_g_bend_hi}}] | {{gp_g_bend_n}} |
| NGS get-off → pressure rate | {{gp_quick}} [{{gp_quick_lo}}, {{gp_quick_hi}}] | {{gp_quick_n}} |

*Table 1. Partial Spearman ρ, class-stratified bootstrap.*

NFL get-off and bend drive pressure; the Combine end barely reaches them. The 3-cone does no better, but {{n_hoop_no3c}} of {{n_hoop}} hoop runners never ran one.

**Why the drill end breaks: one rep.** The two turns of the same lap agree at only ρ = {{rel_turn_half}} (n = {{rel_turn_n}}), so one lap has reliability {{rel_turn_lap}} [{{rel_turn_lap_lo}}, {{rel_turn_lap_hi}}]; {{reps_hoop_turn}} laps reach {{rel_target}}, the steadiness of NFL bend. The {{rel_hoop_nrep}} linemen who ran two laps roughly agree (Figure 2).

![Figure 2. Reliability of a lineman's drill average by number of reps (Spearman-Brown; bands = 90% player bootstrap), with the two-lap check annotated. Ticks at right: NFL-side reliabilities.](figures/fig2_reliability.png)

## Held out by class, Hoop G leads but does not clear the screen

**Outcome:** career pressure rate through 2025, shrunk toward the {{press_mean}}% mean, linemen with 30+ rushes. **Out of class:** each class ({{ho_n_per_class}} rushers) is forecast by a model fit on the other two; drill measures are standardized within Combine year. Before fitting we set the smallest effect worth acting on: {{margin_season}} pressures per 450 rushes per sd (about {{margin_epa}} expected points).

![Figure 3. Held-out effect of the six pre-listed drill measures on pressures above the no-tracking forecast (n = {{n_an}}). Bars: 90% class-stratified bootstrap, refitting every fold. Diamonds: each held-out class. Grey band: action margin. Dashed lines: what the best of six shuffled measures reaches 5% of the time.](figures/fig3_screen.png)

| Held out, n = {{n_an}} | R² | RMSE | R² gain [90%; six-measure adjusted] |
|---|---|---|---|
| Pick + role, no tracking | {{r2_pick}} | {{rmse_pick}} | — |
| Pick + role + Hoop G | {{r2_phoop}} | {{rmse_phoop}} | {{dr2_phoop}} [{{dr2_phoop_lo}}, {{dr2_phoop_hi}}; {{dr2_phoop_alo}}, {{dr2_phoop_ahi}}] |

*Table 2. RMSE in pressure-rate points (outcome sd {{sd_y}}). The full no-tracking model (adding weight, NGS score, 10-yard split, 3-cone) is weaker: R² {{r2_rbase}}.*

- **Against the stronger baseline** the slope holds at {{pr_hoop}} [{{pr_hoop_lo}}, {{pr_hoop_hi}}]; both R² gain intervals span zero.
- **Class by class:** {{ho_hoop_2023}}, {{ho_hoop_2024}}, {{ho_hoop_2025}}. 2025 is flat, not reversed: interval {{cls2025_lo}} to {{cls2025_hi}}, and dropping any one rusher moves it only between {{lopo25_min}} and {{lopo25_max}} (pooled slope: {{lopo_min}} to {{lopo_max}}).
- **Edge and interior:** {{role_EDGE}} [{{role_EDGE_lo}}, {{role_EDGE_hi}}] (n = {{role_EDGE_n}}) against {{role_interior}} [{{role_interior_lo}}, {{role_interior_hi}}] (n = {{role_interior_n}}). Similar slopes, so one within-class scale costs nothing here; still, bend is an edge job, so the flag targets edge alignments and interior names (Brinson) are read against interior linemen only.
- **Reliability-corrected:** one lap carries {{lam_hoop}} of true bend variance. Disattenuated (errors-in-variables, paired bootstrap): {{da_hoop}} [{{da_hoop_lo}}, {{da_hoop_hi}}]; empirical-Bayes shrunk Hoop G: {{eb_hoop}}. The case for four laps, not a new effect: p is unchanged.
- **Independent script:** {{v_slope}} [{{v_lo}}, {{v_hi}}], n = {{v_n}}, classes {{v_cls_2023}}, {{v_cls_2024}}, {{v_cls_2025}}. It keeps one rusher with no valid burst rep ({{T:extra78}}) and fills missing times with the fold median; our pipeline with both choices gives {{rec_med78}} ({{rec_med78_2023}}, {{rec_med78_2024}}, {{rec_med78_2025}}).
- **Multiplicity:** of {{n_tests}} tests here, only {{n_holm}} survive Holm at 0.10, both NFL trait-to-pressure links.

## The film list

Among {{d3_n}} day-3 and undrafted rushers, forecast plus Hoop G orders held-out classes better than draft order (ρ {{dr_fch}} against {{dr_pick}}), but its top third hit {{tt_fch_k}} of {{tt_fch_n}}, exactly as often as pick order: a gap of {{hg_gap}} points [{{hg_gap_lo}}, {{hg_gap_hi}}] against the +{{hg_bar}} we required for board use. **Hoop G does not change the board; it decides whose tape opens first.** The eight flagged 2024–25 day-3 rushers with 100+ NFL rushes, ranked by career pressures per 450 rushes above forecast (NFL pressure, not Hoop G, sets the order):

{{T:film}}

![Figure 4. The eight flagged rushers as small multiples: sampled NFL rush fan and career pressures above forecast on one shared axis, 90% interval.](figures/fig4_cards.png)

## Limitations

- **Small samples, single reps.** {{n_an}} rushers in the model, {{d3_n}} on day 3; {{pct_one_run}}% ran one hoop lap.
- **Reused classes.** Earlier rounds screened ideas on these classes, and pick plus role was chosen as the baseline after seeing them; the 2026 class is the first clean test.
- **Thin NFL tracking.** {{median_random_rushes}} sampled rushes per lineman, no blockers; bend may follow from beating the tackle rather than cause it.
- **Missed time.** Rates are per rush, so injured weeks shrink the denominator, not the rate; the {{n_short_career}} linemen under 30 rushes are left out. No tracked 40 is used, so the clip offset never enters.
- **Reads are ours,** from {{cards_min_sampled}}–{{cards_max_sampled}} sampled rushes each, not scheme film.

## Bottom line

Open day-3 edge rushers' tape in Hoop G order, grade four hoop laps at the 2027 Combine, and move no one on the board until Hoop G beats the pick-plus-role forecast on the 2026 class.

## Appendix

- **Variants.** Full no-tracking model by OLS: R² {{r2_base}}; Hoop G gain over its ridge version {{dr2_rhoop}} [{{dr2_rhoop_lo}}, {{dr2_rhoop_hi}}]. Slope on the ridge residual {{ridge_hoop_season}}; weighted for drill completeness {{ipw_hoop_season}}; across priors and rush minimums {{sens_slope_min}} to {{sens_slope_max}}; {{n_an78}}-rusher set alone {{rec_set78}}; median fill alone {{rec_med77}}.
- **Multiplicity.** Hoop G: two-sided p {{ho_hoop_p2}}, best-of-six p {{ho_hoop_pfw}}, Bonferroni slope interval {{ho_hoop_season_alo}} to {{ho_hoop_season_ahi}}. {{n_bh}} of {{n_tests}} tests survive Benjamini–Hochberg at q = 0.10, including the slope against pick plus role, a baseline chosen after the fact (`multiplicity.csv`).
- **Reliability.** Spearman-Brown for k laps: k·r/(1+(k−1)·r). The two-lap estimate gives a corrected slope of {{da_hoop_lap}}. Averaging both turns instead of keeping the harder one: {{mt_hoop}} [{{mt_hoop_lo}}, {{mt_hoop_hi}}].
- **2026 test.** Frozen in `frozen_rule.json` (sha256 `{{T:hash}}…`): positive Hoop G slope over the pick-plus-role forecast, one-sided p < 0.10, or the flag is retired. Power at the observed slope: {{pws_obs_1}}% for one class, {{pws_obs_3}}% for three.

`analysis.py` (about 6 minutes) rebuilds every number, figure and this text from `data/processed` and runs `verify.py`, an independent re-implementation of the headline slope.
