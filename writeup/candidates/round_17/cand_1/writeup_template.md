# Bend, Not Burst: What the Combine's Pass-Rush Drills Say About NFL Pressure

**Hoop G** is how hard a lineman pushes sideways while rounding the Combine's Run-the-Hoop drill; **burst** is how far he travels in the first second of the pass-rush drill. Against a no-tracking forecast of NFL pressure rate, burst adds {{slope_burst}} points per sd: ruled out. Hoop G adds {{slope_hoop}}, about {{slope_hoop_season}} pressures a season, but its held-out gain is unproven. Use it to break day-3 ties, not to grade.

![Figure 1. (A) Pressure-rate points per sd of each drill measure, added to the no-tracking baseline; grey band = effects too small to act on. (B) Held-out R² gained over the baseline.](figures/fig1_headline.png)

## Why pass rushers, and why these drills

A pressure is what a rusher is paid for. On {{epa_n}} randomly sampled NFL pass plays involving these rookies, a pressured snap cost the offense {{epa_press}} more expected points than a clean one (90% interval {{epa_press_lo}} to {{epa_press_hi}}).

Most of these decisions are cheap ones: {{pct_day3_dl}}% of the {{n_dl}} linemen in the 2023–25 classes went on day 3 or undrafted. The agility test clubs trust, the 3-cone, was run by only {{n_three_cone}} of them. Every lineman, though, runs the position drills under 10 Hz tracking. Two of those drills copy a rush: the pass-rush drill starts from a stance (get-off), and Run-the-Hoop asks him to lean around a circle at speed (bend).

**One margin, set before fitting.** An effect matters if it is at least {{margin}} pressure-rate points per sd. That is one extra pressure every 200 rushes, or {{margin_season}} in a {{season_rushes}}-rush season (about {{margin_epa}} expected points). The DL average is {{press_mean}}%.

## The two drill metrics

- **Hoop G**: the hardest half-second of sideways (centripetal) acceleration on the hoop lap, in g (median {{hoop_g_median}} g). We standardize within Combine year because drill conditions shift between years. {{n_hoop}} linemen have it, {{n_hoop_no3c}} of them with no 3-cone.
- **Burst**: yards covered in the first second after he starts moving in the pass-rush drill (median {{burst_median}} yd), standardized the same way, {{n_burst}} linemen.

Both are noisy single reps. {{pct_one_run}}% of hoop runners have one lap; among the {{rel_hoop_nrep}} with two, lap-to-lap reliability is {{rel_hoop1}}. One burst rep repeats at {{rel_burst1}} ({{rel_burst_nrep}} linemen with 2+ reps).

![Figure 2. Three hoop laps spanning the Hoop G range, coloured by sideways acceleration on one scale.](figures/fig2_laps.png)

## The baseline and the test

**Baseline without tracking:** draft pick (log), Combine weight, NFL EDGE share, NGS athleticism score, and the official 10-yard split and 3-cone. Missing times are filled from weight, role and NGS score. We use the official split because the tracked 40 clip starts before the laser.

**Outcome:** career pressure rate, shrunk toward the mean by {{eb_prior_rushes}} rushes of prior, for the {{n_an}} linemen with both drills and 30+ NFL rushes.

**Out of class:** each Combine class is forecast by a model fit on the other two. All intervals resample players within class and refit every fold.

| Model (held-out) | R² | RMSE (pts) | Rank ρ | R² gain over baseline [90%] |
|---|---|---|---|---|
{{TAB1}}

*Table 1. Leave-one-class-out fit, n = {{n_an}}. RMSE in pressure-rate points.*

The no-tracking baseline is weak. Its held-out R² of {{r2_base}} is below pick and role alone ({{r2_pick}}): with {{n_an}} rushers, the extra Combine times cost more in fitting noise than they add. Hoop G gives the largest improvement on all three held-out measures, and burst alone improves none. Rank agreement rises from {{rho_base}} to {{rho_hoop}}, but the R² gain, {{dr2_hoop}} [{{dr2_hoop_lo}}, {{dr2_hoop_hi}}], still includes zero.

## Finding 1: burst reaches the snap, not the pressure

Burst does carry over to the NFL as a movement (Figure 3). After weight and role, drill burst tracks how far a lineman travels in the first second of a sampled NFL rush: partial ρ = {{tr_burst_g_d1}} [{{tr_burst_g_d1_lo}}, {{tr_burst_g_d1_hi}}], n = {{tr_burst_g_d1_n}}. Given the noise in both measures, the most that link could show is {{tr_burst_g_d1_ceil}}.

The NFL first second matters for pressure too ({{gp_g_d1}}). But drill burst against pressure is {{tr_burst_press_eb}} [{{tr_burst_press_eb_lo}}, {{tr_burst_press_eb_hi}}]. Beyond the baseline, its slope is {{slope_burst}} points per sd [{{slope_burst_lo}}, {{slope_burst_hi}}]. That is inside the ±{{margin}} margin with {{p_burst_within_margin}}% bootstrap probability, so we rule it out. It is also not a stand-in for the official split (ρ = {{rho_burst_split}}, n = {{n_burst_split}}).

![Figure 3. Each drill against four NFL traits, partial Spearman after weight and role, 90% bootstrap; bars = the most each link could show given both reliabilities.](figures/fig3_transfer.png)

## Finding 2: Hoop G adds a little, consistently

Fitted on two classes at a time, a lineman one sd higher in Hoop G pressures {{slope_hoop}} points more often than the baseline expects [{{slope_hoop_lo}}, {{slope_hoop_hi}}]. That is {{slope_hoop_season}} pressures in a {{season_rushes}}-rush season [{{slope_hoop_season_lo}}, {{slope_hoop_season_hi}}]. The bootstrap puts {{p_hoop_above_margin}}% probability on the effect clearing the {{margin}}-point margin.

Scored strictly on the held-out class, Hoop G ranks what the baseline missed at ρ = {{res_hoop}} [{{res_hoop_lo}}, {{res_hoop_hi}}], positive in {{hoop_pos_classes}} of 3 classes. Clubs do not draft for it: top-third linemen went at a median pick of {{pick_top_third}}, bottom-third at {{pick_bot_third}}.

**How it works is only partly visible.** Hoop G barely moves NFL get-off or the first second ({{tr_hoop_quick}}, {{tr_hoop_g_d1}}). It moves sideways push on NFL rushes a little ({{tr_hoop_g_bend}} [{{tr_hoop_g_bend_lo}}, {{tr_hoop_g_bend_hi}}]), and that NFL bend does predict pressure ({{gp_g_bend}}). The chain through the trait we can see is too weak to explain the whole effect. Our guess is speed kept through contact, which needs blockers in the data to see.

## What wins on a snap

Within the same rusher, on {{sn_n}} random sampled rushes by {{sn_players}} linemen, both traits pay. A quicker NGS get-off than his own average raises the odds of pressure by {{or_quick}} per sd [{{or_quick_lo}}, {{or_quick_hi}}]. More sideways push in the first two seconds raises it by {{or_bend}} [{{or_bend_lo}}, {{or_bend_hi}}]. Both are adjusted for down and alignment. The bend window is a fixed two seconds, and only on plays that lasted that long, so longer pressured plays do not inflate it.

![Figure 4. Same rusher, different snap: pressure rate by his own get-off fifth and bend fifth, player fixed effects, 90% intervals.](figures/fig4_snaps.png)

Bend on a snap may partly be the result of beating the tackle, not its cause. It still tells the film room what to grade:

| Film check (wide rushes, third down) | What a yes looks like |
|---|---|
| Get-off | Moving with the snap, first step gaining ground, not lifting |
| Dip at the top of the arc | Shoulder under the tackle's hands within two steps of contact |
| Flatten | Path turns to the quarterback without drifting upfield |
| Speed kept | No visible slow-down through the turn |

*Table 2. What Figure 4 says to grade, for any Hoop G shortlist name.*

## The decision: a day-3 tiebreaker

**Rule (frozen, sha256 `{{T:hash}}…`):** pick {{pick_day3}} or later (or undrafted) and top third of Hoop G in his class. A rusher "hits" if his pressure rate beats the out-of-class baseline forecast.

**Record against the base rate.** {{sl_k}} of {{sl_n}} shortlisted day-3 rushers hit ({{sl_rate}}%, 90% Wilson {{sl_lo}}–{{sl_hi}}%). The rest of day 3 hit {{rest_k}} of {{rest_n}} ({{rest_rate}}%). The base rate for all day-3 rushers is {{d3_rate}}%, which shows the forecast is calibrated for them. The gap of +{{gap}} points [{{gap_lo}}, {{gap_hi}}] is in the right direction in {{sl_classes_ahead}} of 3 classes, worth about {{sl_over_season}} pressures a season. It is not proven on its own.

Past shortlist hits include Cedric Johnson (pick {{best_pick}}, {{best_pr}}% pressure vs {{best_fc}}% forecast, {{best_rush}} rushes), {{T:hist_hit}}; misses: {{T:hist_miss}}.

![Figure 5. (A) Share of day-3 rushers beating the forecast, shortlist vs rest, with 90% intervals and per-class diamonds. (B) Simulated power of the frozen 2026 test.](figures/fig5_decision.png)

**2025 shortlist, calls for 2026:**

| Rusher (pick, team) | Role | Hoop G (sd) | NFL rushes | Pressure vs forecast | Call |
|---|---|---|---|---|---|
{{CARDS}}

*Table 3. Go = 100+ rushes and above forecast; No-go = 100+ and below; Watch = 30–99; Audition = under 30.*

1. **Go: {{T:names[0]}} ({{T:teams[0]}}).** An EDGE at +{{c0_hoop}} sd with {{c0_pr}}% pressure vs {{c0_fc}}% forecast in {{c0_rush}} rushes. Give him more third-down snaps from wide alignments, where bend matters most.
2. **Go: {{T:names[1]}} ({{T:teams[1]}}).** An interior lineman at +{{c1_hoop}} sd, {{c1_pr}}% vs {{c1_fc}}% in {{c1_rush}} rushes. Make him the looper on interior stunts.
3. **No-go for now: {{T:nogo}} ({{T:nogo_team}}).** He is +{{ng_hoop}} sd but at {{ng_pr}}% vs {{ng_fc}}% in {{ng_rush}} rushes. Hold his snaps until the film checks in Table 2 come back yes.

**What would change the call.** The 2026 test fits the Hoop G slope on the 2026 class against its frozen baseline, one-sided at 0.10. At the observed slope one class gives {{pws_obs_1}}% power and three give {{pws_obs_3}}%. The shortlist hit-rate test alone would have {{pow_obs_1}}% power, so the slope is the pre-registered primary. A 2026 slope at or below zero means dropping Hoop G as a tiebreaker. A slope above {{margin}} justifies asking every lineman for a second hoop lap, which would lift reliability from {{rel_hoop1}} to {{rel_hoop2}}.

## Limitations

- **Small and noisy.** {{n_an}} rushers, single drill reps, and a Hoop G reliability that rests on only {{rel_hoop_nrep}} linemen with two laps. The held-out R² gain includes zero; the case for Hoop G is consistency across classes, not size.
- **Chosen after looking.** Hoop G was the strongest of several drill measures screened in earlier work on these classes. Holding classes out checks the forecast, not that choice; 2026 is the first clean test.
- **Thin NFL tracking.** About {{median_random_rushes}} random sampled rushes per lineman, with no blockers and no ball. NFL bend may be an effect of winning rather than a cause.
- **Survivorship.** Hit rates need 30+ rushes. Shortlisted rushers reached that as often as the rest ({{reach_sl_k}} of {{reach_sl_n}} vs {{reach_rest_k}} of {{reach_rest_n}}), but injuries and roster moves are not modelled.

## Appendix

**Reliability.** NFL first-second burst {{rel_game_burst}} and NFL bend {{rel_game_bend}} (split-half, {{rel_game_n}} linemen with 8+ random rushes); NGS season get-off {{rel_getoff_season}} season to season ({{rel_getoff_n}} pairs). Burst averaged over its reps: {{rel_burst}}. Hoop G vs 3-cone ρ = {{rho_hoop_3cone}} (n = {{n_hoop_3cone}}); Hoop G vs burst {{rho_hoop_burst}}.

**Transfer details.** Drill burst vs NGS season get-off {{tr_burst_quick}} [{{tr_burst_quick_lo}}, {{tr_burst_quick_hi}}]. Burst vs NFL bend {{tr_burst_g_bend}}. NGS get-off vs pressure beyond weight and role {{gp_quick}} [{{gp_quick_lo}}, {{gp_quick_hi}}]. Hoop G vs pressure after weight and role only, {{tr_hoop_press_eb}} [{{tr_hoop_press_eb_lo}}, {{tr_hoop_press_eb_hi}}].

**Shortlist tests.** One-sided Fisher p = {{p_fisher}}; binomial vs the day-3 base rate p = {{p_binom_base}}. Top-third Hoop G on days 1–2 (outside the rule): {{d12_top_k}} of {{d12_top_n}} vs {{d12_rest_k}} of {{d12_rest_n}}. Burst slope positive in {{burst_pos_classes}} of 3 classes. Power assumes {{pw_slope_n}} scored linemen per class, residual sd {{sd_res}} points.

*Reproduce: `uv run python writeup/candidates/round_17/cand_1/analysis.py` (about 6 minutes) writes numbers.json, the figures, frozen_rule.json and this text.*
