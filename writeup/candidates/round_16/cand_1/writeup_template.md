# Hoop G: A Free Day-3 Pass-Rush Tiebreaker

**Day-3 and undrafted linemen who bend hardest in the Combine's Run-the-Hoop drill beat their no-tracking pressure forecast {{main_k1}} times in {{main_n1}}; the rest of day 3 did so {{main_k0}} times in {{main_n0}}, a +{{main_gap}}-point gap with a wide interval. Clubs did not draft for it. Use it to break ties between cheap rushers, not to grade them. Combine get-off burst does not carry through to pressure. For 2026 camps, act on {{act_names}}.**

![Figure 1. (A) Each day-3/undrafted rusher with 30+ NFL rushes: pressure rate minus his out-of-class forecast, sorted; orange and named = top-third Hoop G (the shortlist); filled = beat the forecast. (B) Share beating the forecast, by class (grey, low power) and pooled, 90% Wilson intervals; dotted line = rest-of-day-3 pooled rate. (C) Median draft pick of each Hoop G third, 90% bootstrap intervals.](figures/fig1_shortlist.png)

## Why pass rushers, why now

Every April a club spends day-3 picks and undrafted money on its fourth and fifth pass rushers, and in August it cuts one of them. {{pct_day3_hoop}}% of the linemen in this study went on day 3 or undrafted, so this is where most rush decisions are made, and where the evidence is thinnest: only {{n_three_cone}} of {{n_dl}} linemen in the 2023–25 classes ran the 3-cone. A pressure is what that player is for: on {{epa_n}} randomly sampled NFL pass plays involving these rookies, a pressured snap cost the offense {{epa_press}} more expected points than a clean one (90% interval {{epa_press_lo}} to {{epa_press_hi}}).

Since 2023 every lineman's position drills are tracked at 10 Hz, and nobody times them. A lineman who skips the 3-cone still runs the hoop. So the question is narrow: does one tracked drill sort cheap rushers better than the board already does?

## The metric

**Hoop G is how hard a lineman pushes sideways, in g, during the hardest half-second of his Run-the-Hoop lap, ranked against his own Combine class.** Run-the-Hoop is a figure-eight around two hoops that rewards bending the hips at speed; the median Hoop G is {{hoop_g_median}} g. We standardize within Combine year because drill conditions shift between years. {{n_hoop}} linemen have it, {{n_hoop_no3c}} with no 3-cone.

The contrast is **rush-drill burst**: yards covered in the first second of the pass-rush drill, standardized the same way.

Both are noisy. {{pct_one_run}}% of hoop runners have one run; among the {{rel_hoop_nrep}} with two, run-to-run reliability is {{rel_hoop1}}. One burst rep repeats at {{rel_d1_1}}.

![Figure 2. Four rushers' hardest hoop lap, coloured by sideways acceleration on one shared scale; the red ring marks the half-second that defines Hoop G. Class median = {{hoop_g_median}} g.](figures/fig2_laps.png)

## The test

**The yardstick** forecasts each lineman's NFL pressure rate from what a club already has: draft pick, Combine weight, NFL EDGE share, NGS athleticism score, and the official 10-yard split and 3-cone (missing times filled from weight, role and NGS score). We use the official split because the tracked 40 clip starts before the laser: the tracked split runs {{split_off}} s slow and agrees with the official at only r = {{split_r}} ({{split_n}} linemen). Each class is forecast by a model fit only on the other two; on held-out classes it ranks regulars' pressure rates at ρ = {{fc_rho}}.

**The rule**: shortlist = pick {{pick_day3}} or later (or undrafted) AND top third of Hoop G in his class. Nothing in it is fit on NFL data. A rusher with 30+ NFL rushes "hits" if his pressure rate beats his forecast.

**Result.** {{main_k1}} of {{main_n1}} shortlisted rushers hit, against {{main_k0}} of {{main_n0}} others: a gap of **+{{main_gap}} points**. In a regular's season of 450 rushes that is +{{ex_gap}} pressures (90%: {{ex_gap_lo}} to {{ex_gap_hi}}), about {{ex_gap_epa}} expected points.

{{TABLE:robust}}

The pick-and-role-only row shows the gap is not cheap players beating a forecast anchored on draft slot. The bottom two rows are the honest ones: scoring each class only with earlier classes, or reweighting for the {{n_unscored}} day-3 runners who never reached 30 rushes, shrinks the gap and the interval crosses zero. Hoop G did not decide who got on the field (Appendix).

**Verdict: a small, uncertain edge, enough to break a tie, not enough to move a player up a board.** The forecast was right about the shortlist ({{cal_sl_act}}% actual vs {{cal_sl_fc}}% forecast) and too kind to the rest ({{cal_re_act}}% vs {{cal_re_fc}}%). Across all {{slope_n}} scored hoop runners (Figure 3), pressure over forecast rises {{slope}} points per sd of Hoop G (90%: {{slope_lo}} to {{slope_hi}}). But Hoop G was the best of 14 drill-and-outcome pairs we screened; priced for that search, p = {{scr_p}}.

![Figure 3. Every scored hoop runner: NFL pressure rate minus the out-of-class forecast against Hoop G. Orange = day-3 shortlist; faded = 30–99 rushes; grey band = middle half of 100+ rush regulars; line = rush-weighted slope.](figures/fig3_scatter.png)

## Every board, not a chosen one

A tiebreaker is judged one decision at a time. We paired every scored shortlisted rusher with the non-shortlisted rusher of the same class and role drafted nearest to him (median {{pair_pickgap}} picks apart, rookie pay within ${{pair_apy_gap}}M a year). The shortlisted player came out ahead on **{{pair_k}} of {{pair_n}} boards ({{pair_pct}}%; 90%: {{pair_pct_lo}} to {{pair_pct_hi}})**, by a median of {{pair_med}} pressures per 450 rushes (90%: {{pair_med_lo}} to {{pair_med_hi}}). Figure 5 shows all of them; only {{pair_npeer}} distinct peers fill the {{pair_n}} boards, so intervals resample peers, not boards.

As an illustration, the board closest to that median: {{ill_sl}} (pick {{isl_pick}}, Hoop G +{{isl_hz}}) against {{ill_peer}} (pick {{ipe_pick}}, Hoop G {{ipe_hz}}), {{isl_yr}}. The rule takes {{ill_sl}}: {{isl_pr}}% pressure against a forecast of {{isl_pb}}%, vs {{ipe_pr}}% against {{ipe_pb}}% for {{ill_peer}}.

A single call is close to a coin flip; the edge exists only on average. That decides a last roster spot, not a grade.

![Figure 5. Every matched board: each scored shortlisted rusher (orange) against the nearest-pick non-shortlisted rusher of the same class and role (blue), in pressures per 450 rushes over forecast. Green line = shortlist ahead, red = behind.](figures/fig5_boards.png)

## Get-off: drill burst reaches the snap, not the pressure

Each link holds alone (Figure 4): after weight and role, drill burst tracks how far a lineman travels in the first second of a sampled NFL rush: ρ = {{ch_bg}} (90%: {{ch_bg_lo}} to {{ch_bg_hi}}; {{ch_bg_n}} linemen), {{ch_bg_da}} once corrected for noise. That NFL first-second burst agrees with NGS season get-off ({{ch_gq}}), and season get-off predicts pressure over forecast ({{ch_gp}}).

Multiplied through, the chain leaves about {{ch_prod}}. Measured end to end, drill burst against pressure over forecast is ρ = {{ch_bp}} (90%: {{ch_bp_lo}} to {{ch_bp_hi}}). The bottleneck is the single drill rep.

Hoop G is the reverse: it reaches pressure (Figure 3) but not first-second burst ({{ch_hg}}) or depth upfield at 1.5 s ({{ch_hd}}). Our best guess is a rusher who keeps speed around the corner and through contact, which we cannot see without blockers in the data.

![Figure 4. The drill-to-game chain. Spearman ρ with 90% player-bootstrap intervals; the white diamond is the burst link corrected for measurement noise.](figures/fig4_chain.png)

## When drill and tape disagree

| Player (class, pick) | Hoop G call | Pressure vs forecast (rushes) | Scout action and film cue |
|---|---|---|---|
| **Jaylen Harrell** (2024, {{harrell_pick}}) | Shortlist, +{{harrell_hz}} sd | {{harrell_pr}}% vs {{harrell_pb}}% ({{harrell_n}}) | Drill yes, NFL no, large sample. Does he convert speed to power when the tackle sets short? If not, rotate him. |
| **Isaiah McGuire** (2023, {{mcguire_pick}}) | Bottom third, {{mcguire_hz}} sd | {{mcguire_pr}}% vs {{mcguire_pb}}% ({{mcguire_n}}) | The rule's miss: a power rusher who wins straight ahead. Low Hoop G is not a cut signal. |
| **Jordan Phillips** (2025, {{phillips_pick}}) | Bottom third, {{phillips_hz}} sd | {{phillips_pr}}% vs {{phillips_pb}}% ({{phillips_n}}) | Looped on {{phillips_loop}}% of sampled rushes (interior median {{int_loop_med}}%). Grade the role before the player. |

## So what for a team

1. **Break day-3 ties with Hoop G.** Top-third linemen went at a median pick of {{pick_t2}}, no earlier than the bottom third ({{pick_t0}}); per sd of Hoop G, clubs drafted {{price_later}}% later (90%: {{price_earlier}}% earlier to {{price_hi}}% later). The tiebreaker costs no draft capital.
2. **Match the role to the trait.** Shortlisted EDGEs get third-down snaps from wide alignments, where bend matters most; shortlisted interior players get the looper job on stunts.
3. **Do not draft on drill burst.** Time get-off on tape or in camp.
4. **Ask for a second hoop run.** Reliability goes from {{rel_hoop1}} to {{rel_hoop2}}. Burst would need about {{burst_reps50}} reps to reach 0.5; skip it.

**The rule's record.** 2023 shortlist hits: {{hist_hit_2023}}; misses: {{hist_miss_2023}}. 2024 hits: {{hist_hit_2024}}; misses: {{hist_miss_2024}}.

**The 2025 shortlist**, with the {{n_act}} players to act on first ({{n_act_res}} on a reserve list: clear the medical first). "Widen role" = beat his rookie forecast; "camp audition" = Hoop G +2 sd or more and too few rushes to judge, so the drill is all the evidence there is; "watch" = unscored, ordinary shortlist Hoop G; "hold" = below forecast, let tape decide. Lateral rushes = share of sampled rushes moving 2+ yards sideways by 1.5 s (stunts, loops, wide arcs).

{{TABLE:card25}}

**Pre-registered 2026 test.** The rule and forecast are frozen in `frozen_rule.json` and `frozen_shortlist.json` (SHA-256 prefix `{{rule_sha}}`): one-sided Fisher test at 10%, scored after the 2027 season. A class supplies about {{pow_n1}} shortlisted and {{pow_n0}} other scored rushers. If the observed gap is true, power is {{pow_1}}% for 2026 alone, {{pow_2}}% pooled with 2027 and {{pow_3}}% over three classes ({{pow_3_half}}% if the true gap is half as large). Even three classes stay under two-thirds power, so treat 2026 as a check on direction, not a verdict.

## Limitations

- **One rule, chosen after looking.** Holding classes out checks the forecast, not the drill choice (search-adjusted p = {{scr_p}}). Stricter versions are weaker; future classes are the clean test, underpowered one at a time.
- **Small counts.** {{main_n1}} shortlisted vs {{main_n0}} comparison rushers; 2025 rookies have one season. Hoop G reliability rests on {{rel_hoop_nrep}} linemen with two runs.
- **Forecast level.** Fit on 100+ rush regulars, the forecast sits above most day-3 rushers, so day-3 rushers are compared with each other, not with 50%.
- **Thin NFL tracking, no injuries or scheme labels.** About 20 random sampled rushes per player, no blockers, no ball. Sampled pressure barely repeats (split-half {{rel_prs}}), so pressure comes from season totals; scheme notes come from loop rates, availability from games played and roster status.

## Appendix

| Check | Result |
|---|---|
| Shortlist vs rest by class (2023, 2024, 2025) | {{c2023_k1}}/{{c2023_n1}} vs {{c2023_k0}}/{{c2023_n0}}; {{c2024_k1}}/{{c2024_n1}} vs {{c2024_k0}}/{{c2024_n0}}; {{c2025_k1}}/{{c2025_n1}} vs {{c2025_k0}}/{{c2025_n0}} |
| Selection model: odds of reaching 30 rushes per sd of Hoop G (90%) | {{sel_or_hoop}} ({{sel_or_lo}} to {{sel_or_hi}}) |
| Class-stratified (Mantel-Haenszel) odds ratio (90%); p | {{mh_or}} ({{mh_lo}} to {{mh_hi}}); {{mh_p}} |
| Flat-prior P(gap > 0); within-class relabelling p | {{bayes_pgt0}}%; {{perm_p}} |
| Shrunken (ridge) forecast; without {{best_name}} | {{ridge_k1}}/{{ridge_n1}} vs {{ridge_k0}}/{{ridge_n0}}; {{drop_k1}}/{{drop_n1}} vs {{drop_k0}}/{{drop_n0}} |
| Same rule on the official 3-cone (day 3, ran it) | {{t3_k1}}/{{t3_n1}} vs {{t3_k0}}/{{t3_n0}} |

*Reproduce: `uv run python writeup/candidates/round_16/cand_1/analysis.py` writes `numbers.json`, the board, the frozen 2026 rule and all figures, and fills this text.*
