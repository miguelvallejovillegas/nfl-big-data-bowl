# Bend Breaks Day-3 Ties: What Run-the-Hoop Says About NFL Pressure

**Hoop G** measures how hard a defensive lineman pushes sideways while rounding the Combine's Run-the-Hoop drill. Scored only on Combine classes the model never saw, one sd of Hoop G is worth **{{ho_hoop_season}} extra pressures per season** [{{ho_hoop_season_lo}}, {{ho_hoop_season_hi}}] beyond a forecast built without tracking. That is a positive signal (shuffle p = {{ho_hoop_p}}) but probably below the line where we would grade on it. Burst out of a stance in the pass-rush drill is worth nothing. Use Hoop G for one job: **breaking day-3 ties**. Our Go calls for 2026 are **{{T:names[0]}}** ({{T:teams[0]}}) and **{{T:names[1]}}** ({{T:teams[1]}}).

![Figure 1. Extra pressures per 450-rush season per sd of each drill, beyond the no-tracking forecast, scored on the held-out class. Bar = 90% class-stratified bootstrap; shade = bootstrap distribution; diamonds = each held-out class; grey band = too small to act on.](figures/fig1_headline.png)

## Why pass rushers, and why these drills

A pressure is what a rusher is paid for. On {{epa_n}} randomly sampled NFL pass plays involving these rookies, a pressured snap cost the offense {{epa_press}} more expected points than a clean one (90% interval {{epa_press_lo}} to {{epa_press_hi}}).

Most of these decisions are cheap ones: {{pct_day3_dl}}% of the {{n_dl}} linemen in the 2023–25 classes went on day 3 or undrafted. Only {{n_three_cone}} of them ran the 3-cone, the agility test clubs trust. All of them run position drills under 10 Hz tracking, and two of those drills copy a rush. The pass-rush drill starts from a stance (get-off), and Run-the-Hoop makes him lean around a circle at speed (bend).

**One margin, set before fitting.** An effect is worth acting on at {{margin_season}} pressures per 450-rush season per sd. That is one pressure every 200 rushes, about {{margin_epa}} expected points a season. The DL average pressure rate is {{press_mean}}%.

## The two drill metrics

- **Hoop G** is the hardest half-second of sideways (centripetal) acceleration on the hoop lap, in g (median {{hoop_g_median}} g). {{n_hoop}} linemen have it, and {{n_hoop_no3c}} of them never ran the 3-cone.
- **Burst** is the yards covered in the first second after he starts moving in the pass-rush drill (median {{burst_median}} yd), for {{n_burst}} linemen.

Both are standardized within Combine year, because drill conditions shift between years. Both are noisy single reps. {{pct_one_run}}% of hoop runners have one lap; among the {{rel_hoop_nrep}} with two, lap-to-lap reliability is {{rel_hoop1}}. One burst rep repeats at {{rel_burst1}}.

![Figure 2. Three hoop laps spanning the Hoop G range, coloured by sideways acceleration on one scale.](figures/fig2_laps.png)

## The baseline and the test

**Baseline without tracking:** draft pick (log), Combine weight, NFL EDGE share, NGS athleticism score, and the official 10-yard split and 3-cone. Missing times are filled from weight, role and NGS score. We use the official split because the tracked 40 clip starts about 0.3 s before the laser, which corrupts any tracked split.

**Outcome:** career pressure rate through 2025, shrunk toward the mean by {{eb_prior_rushes}} rushes of prior, for the {{n_an}} linemen with both drills and 30+ NFL rushes.

**Out of class:** each Combine class ({{ho_n_per_class}} linemen) is forecast by a model fit on the other two. The headline is one estimand: the slope of the held-out miss (actual minus forecast) on the drill. Its interval resamples players within class and refits every fold. Its p-value shuffles the drill within class. Players are the independent units throughout.

| Model (held-out) | R² | RMSE (pts) | Rank ρ | R² gain over baseline [90%] |
|---|---|---|---|---|
{{TAB1}}

*Table 1. Leave-one-class-out fit, n = {{n_an}}. RMSE in pressure-rate points.*

The baseline is weak. With {{n_an}} rushers, its extra Combine times cost more in fitting noise than they add, so it scores below pick and role alone. Hoop G improves all three held-out measures, though its R² gain still includes zero. Burst improves none.

## Finding 1: burst reaches the snap, not the pressure

Burst does carry over as a movement (Figure 3). After weight and role, drill burst tracks how far a lineman travels in the first second of a sampled NFL rush (partial ρ = {{tr_burst_g_d1}} [{{tr_burst_g_d1_lo}}, {{tr_burst_g_d1_hi}}], n = {{tr_burst_g_d1_n}}). It does not track NGS season get-off ({{tr_burst_quick}}).

The NFL first second matters for pressure ({{gp_g_d1}}). Drill burst does not reach it: out of class it is worth {{ho_burst_season}} pressures per sd [{{ho_burst_season_lo}}, {{ho_burst_season_hi}}]. The bootstrap puts {{ho_burst_pabove}}% probability on a gain above the margin. A formal equivalence test (TOST) gives p = {{ho_burst_tost}}: it cannot rule out a small negative effect, but it rules out a useful positive one.

![Figure 3. Each drill against four NFL traits, partial Spearman after weight and role, 90% bootstrap; dashes = the most each link could show given both reliabilities.](figures/fig3_transfer.png)

## Finding 2: Hoop G adds a little, out of class

A lineman one sd higher in Hoop G beats his held-out forecast by {{ho_hoop_season}} pressures a season [{{ho_hoop_season_lo}}, {{ho_hoop_season_hi}}]. The within-class shuffle gives p = {{ho_hoop_p}}, and {{ho_hoop_ppos}}% of bootstrap draws are positive, so the test and the interval agree. The size is the open question. Only {{ho_hoop_pabove}}% of draws clear the action line, and the effect holds in 2023 and 2024 but is flat in 2025 ({{ho_hoop_2025}} points per sd).

Clubs do not draft for it. Top-third Hoop G linemen went at a median pick of {{pick_top_third}}, bottom-third at {{pick_bot_third}}.

**How it works is only partly visible.** Hoop G barely moves NFL get-off or the first second ({{tr_hoop_quick}}, {{tr_hoop_g_d1}}). It moves sideways push on NFL rushes a little ({{tr_hoop_g_bend}} [{{tr_hoop_g_bend_lo}}, {{tr_hoop_g_bend_hi}}]), and that NFL bend predicts pressure ({{gp_g_bend}}). Within the same rusher, on {{sn_n}} sampled rushes, one sd more sideways push raises the odds of pressure by {{or_bend}} [{{or_bend_lo}}, {{or_bend_hi}}]; a quicker get-off by {{or_quick}}. Bend on a snap may partly be the result of beating the tackle rather than its cause, and the data have no blockers.

## The decision: a day-3 tiebreaker

**Rule (frozen, sha256 `{{T:hash}}…`).** On day 3 or among undrafted rushers, when two rushers' no-tracking forecasts sit within 0.5 points of each other (the {{margin_season}}-pressure action margin), take the higher Hoop G. Otherwise follow the forecast.

**Why a tie, not a pick gap.** In these ties the median pick gap is {{tie_pick_gap_med}} slots, and the earlier pick was the better NFL rusher {{tie_pick_win}}% of the time. Late-round pick order does not separate rushers, so a pick-gap threshold would not define a real tie.

**The record.** There are {{tie_pairs}} such pairs among {{tie_players}} rushers ({{tie_same_role}}% same role). The effective sample is the {{tie_players}} rushers, and all intervals resample rushers, not pairs.

- **How often:** Higher Hoop G picked the better NFL rusher in {{tie_hoop_win}}% of ties [{{tie_hoop_lo}}, {{tie_hoop_hi}}], shuffle p = {{tie_hoop_p}}. As a hit rate, it is unproven. NGS score ({{tie_ngs_win}}%) and burst ({{tie_burst_win}}%) do no better than a coin.
- **How much:** When Hoop G is right, its pick out-rushes the other by {{tie_gain_right}} pressures a season; when wrong, it gives up {{tie_cost_wrong}}. The net is **{{tie_hoop_gain}} pressures per decision** [{{tie_hoop_gain_lo}}, {{tie_hoop_gain_hi}}], shuffle p = {{tie_hoop_gain_p}}.
- **Cost of being wrong:** If Hoop G were pure noise, the rule would override the forecast's lean in {{tie_fc_override_share}}% of ties, by {{tie_fc_overridden}} pressures on average. The worst-case expected cost is {{tie_worst_cost}} pressures per decision, against the {{tie_hoop_gain}} upside.

Where both rushers ran the 3-cone, it did better still ({{tie_cone_win}}%, only {{tie_cone_n}} pairs). Use the 3-cone first; Hoop G is the stand-in for the {{n_hoop_no3c}} hoop runners without one.

![Figure 4. Day-3 ties: how often each tiebreaker picked the better NFL rusher (A) and the pressures per season it gained (B). 90% intervals resample rushers within class.](figures/fig4_tiebreak.png)

Among 2023–24 day-3 rushers in the top third of Hoop G, the best case is {{T:best}} (pick {{best_pick}}, +{{best_hoop}} sd): {{best_pr}}% pressure vs a {{best_fc}}% forecast over {{best_rush}} rushes. Others who beat the forecast: {{T:hist_hit}}. Misses: {{T:hist_miss}}.

## 2025 class: calls for the 2026 season

Of the {{n_short25}} day-3 rushers in the top third of Hoop G, two have earned more snaps. **Go** means 100+ rushes and at least 0.5 points above forecast, and **No-go** means at least 0.5 below. The flip rate is the 2026 pressure rate that would move his career rate onto the forecast at his 2025 volume. **Decision date: after the 2026 regular season.**

| Rusher (pick, team) | Role | Hoop G (sd) | NFL rushes | Pressure vs forecast | Call | 2026 flip rate |
|---|---|---|---|---|---|---|
{{CARDS}}

*Table 2. Career pressure rate is shrunk; forecast is the frozen no-tracking baseline.*

1. **Go: {{T:names[0]}} ({{T:teams[0]}}).** EDGE, +{{go0_hoop}} sd: {{go0_pr}}% vs {{go0_fc}}% forecast in {{go0_rush}} rushes, {{go0_over}} pressures a season ahead. Give him third-down snaps from wide alignments, where bend matters most. The call flips if he rushes below {{go0_flip}}% in 2026 (2025: {{go0_raw}}% raw).
2. **Go: {{T:names[1]}} ({{T:teams[1]}}).** Interior, +{{go1_hoop}} sd: {{go1_pr}}% vs {{go1_fc}}% in {{go1_rush}} rushes. Make him the looper on interior stunts. The call flips below {{go1_flip}}% (2025: {{go1_raw}}% raw).
3. **No-go: {{T:nogo}} ({{T:nogo_team}}).** +{{ng_hoop}} sd, but {{ng_pr}}% vs {{ng_fc}}%. It would take {{ng_flip}}% in 2026 to reach the forecast. Hold his snaps unless film shows him flattening at the top of the arc.

Hold names stay on film only. Auditions (under 30 rushes): {{T:aud}}.

**What would change the rule.** The frozen 2026 test is the Hoop G slope against the frozen forecast for the 2026 class after its first season (one-sided, p < 0.10). Simulated power is {{pws_obs_1}}% for one class and {{pws_obs_3}}% for three. A slope at or below zero drops Hoop G as a tiebreaker. A slope above the margin justifies asking every lineman for a second hoop lap, which would lift reliability from {{rel_hoop1}} to {{rel_hoop2}}.

## Limitations

- **Small and noisy.** {{n_an}} rushers in the model and {{tie_players}} in the ties, single drill reps, and a Hoop G reliability that rests on {{rel_hoop_nrep}} linemen with two laps.
- **Chosen after looking.** Hoop G was the strongest of several drill measures screened in earlier work on these classes, and the tie analysis was built on them too. Holding out classes checks the forecast, not that choice. The 2026 class is the first clean test.
- **Hit rate versus size.** The tie rule wins by margin, not by frequency ({{tie_hoop_win}}%, p = {{tie_hoop_p}}). A few big wins drive the gain, so one injury could move it.
- **Thin NFL tracking.** About {{median_random_rushes}} random sampled rushes per lineman, with no blockers and no ball.
- **Survivorship.** Rates need 30+ rushes. Top-third day-3 rushers reached that as often as the rest of day 3 ({{reach_sl_k}} of {{reach_sl_n}} vs {{reach_rest_k}} of {{reach_rest_n}}), but injuries and roster moves are not modelled.

## Reproduce

`uv run python writeup/candidates/round_18/cand_1/analysis.py` (about 4 minutes) rebuilds every number, figure and this text. `verify.py` recomputes Table 1, both held-out slopes, the forecasts and the tie results from `inputs.csv` in under a second and diffs them against `numbers.json`. Column definitions are in `inputs_schema.json`.
