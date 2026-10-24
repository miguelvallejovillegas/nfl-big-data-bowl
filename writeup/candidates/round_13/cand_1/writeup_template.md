# Hoop G: A Free Day-3 Shortlist for Pass Rushers

**Hoop G is how hard a defensive lineman pushes sideways while rounding the Run-the-Hoop curve, compared with his Combine class. Shortlist day-3 and undrafted rushers in the top third of their class on it: {{r15_sl_k}} of {{r15_sl_n}} ({{r15_sl_pct}}%) beat a no-tracking forecast of their NFL pressure rate, against {{r15_bs_k}} of {{r15_bs_n}} ({{r15_bs_pct}}%) of the other day-3 rushers. Clubs do not pay for it. The edge is small, the 2025 class barely shows it, and we cannot see why it works: a shortlist for film, not a grade.**

![Figure 1. Hoop G thirds set at the Combine: NFL pressure against the out-of-class forecast (A), the day-3 shortlist against the rest of day 3, by class (B), and where clubs drafted each third (C).](figures/fig0_gm.png)

## Why it matters

A pressure is what a rusher is paid for: on {{r10_epa_n}} sampled NFL pass plays of our rookies, the offense lost {{r10_epa_diff}} more expected points on pressured snaps (90% interval {{r10_epa_lo}} to {{r10_epa_hi}}).

Day 3 is where pass rushers are cheap and boards are thin. The usual agility number, the 3-cone, makes a big man stop, plant and bend around a cone, the hip bend he needs to dip under a tackle's hands. But only {{r10_t3_all_k}} of {{r10_t3_all_n}} linemen in the 2023–25 classes ran an official 3-cone, and it does not hold up on classes it was not fit on (ρ = {{r10_t3_loco}}, 95% interval {{r10_t3_loco_lo}} to {{r10_t3_loco_hi}}). Another {{r12_rec_8_all}} ran Run-the-Hoop, a figure-eight around two hoops, and no 3-cone. Nobody times it; the 10 Hz Combine tracking can: we take the hardest half-second of sideways acceleration on the curve, in g, and standardize it within each Combine year, because drill conditions shift between years.

## The shortlist, scored on classes it never saw

**The bar.** Every lineman gets a forecast of his NFL pressure rate from what clubs already have: draft pick, weight, EDGE share, NGS athleticism score and the official 10-yard split and 3-cone. Each class is forecast by a model fit only on the other two. It ranks rushers in every held-out class, though its spread varies by class (appendix); a shrunken (ridge) version cuts its error from {{r14_rmse_ols}} to {{r14_rmse_ridge}} points.

**The rule.** Pick {{r15_day3_cut}} or later, or undrafted, and Hoop G in the top third of all hoop runners in his class. Thirds are set at the Combine, before anyone plays, and nothing in the rule is fit on NFL data. A shortlisted rusher with 30+ NFL rushes "hits" if his pressure rate beats his forecast. The rule is frozen (SHA-256 `{{r15_hash}}`) and will score the 2026 class, unchanged, after the 2027 season.

{{TABLE:tests15}}

The shortlist was ahead in {{r15_sl_cls_ahead}} of 3 classes (Figure 1B). Relabelling Hoop G at random within class gives a gap this large {{r15_sl_p}} of the time. In pressures, the shortlist ran {{r15_sl_exc}} per 450 rushes against its forecast while the other day-3 rushers ran {{r15_bs_exc}}: a gap of {{r15_exc_gap}} pressures a season, about {{r15_exc_gap_epa}} expected points, with an interval that touches zero. The honest reading is that Hoop G **flags day-3 rushers who hold their forecast while their peers fall short of it**.

Two traps we checked. Survival: shortlisted rushers were no more likely to reach 30 NFL rushes ({{r15_got_sl}} of {{r15_got_sl_n}} against {{r15_got_bs}} of {{r15_got_bs_n}}), so the result is not a playing-time filter. One player: without {{r15_drop_name}}, the shortlist is still {{r15_drop_k}} of {{r15_drop_n}}.

Across all {{r12_n}} rushers with 30+ rushes, the same signal shows as a slope: {{r12_sk_all}} more pressures per 100 rushes per sd of Hoop G, after shrinking toward zero because Hoop G was the best of 14 drills we screened. Three versions of the bar (least squares, ridge, forecast rescaled per class) give the same slope (Figure 3A). Priced for that 14-drill search, though, p = {{r12_mult_p}}.

![Figure 2. Seven rushers: drill path, hardest sampled NFL rush, and pressure rate minus the out-of-class forecast with 80% intervals and play counts.](figures/fig1_hero.png)

## Get-off carries over; Hoop G has no visible mechanism

The angle we started from was the pass-rush drill's get-off burst. It does carry over: its first-second burst tracks NFL get-off on the sampled game frames (ρ = {{r14_d1_game}}, {{r14_d1_game_n}} linemen). But get-off is not pressure once pick, weight, role and NGS score are known (ρ = {{r14_d1_press}}, {{r14_d1_press_n}} linemen), so the burst is not the edge.

Hoop G is the reverse: it predicts pressure but leaves no trace in the frames. Pressure on 20 sampled rushes is noisy (split-half reliability {{r12_rel_prs}}), so we built a steadier outcome, **depth upfield 1.5 seconds after the snap**: it repeats (reliability {{r12_rel_dep}}, {{r12_rel_dep_n}} linemen) and tracks season pressure (ρ = {{r12_val_rho}}). Hoop G does not move it (Figure 3B), nor how sharply rushers turn after the snap. Our best guess is a rusher who keeps speed through contact, which we cannot see without blockers in the data.

![Figure 3. Held-out slope of pressure over forecast (A) and of 1.5-second depth over forecast (B) on Hoop G, by class and pooled, against Hoop G shuffled within class.](figures/fig3_heldout.png)

## When the call is wrong

{{TABLE:cases12}}

**Johnson** is the case the shortlist is for (Figure 2): a day-3 EDGE who beat his forecast. **Adebawore** is the miss: top-third Hoop G, a rate on his forecast over 511 rushes. Where drill and tape disagree like this, tape decides. **Phillips and Black** share a low Hoop G with opposite results. Scheme is the likely reason: Miami moved Phillips 2+ yards sideways early on {{r12_phillips_loop}}% of his sampled rushes (interior median {{r12_loop_int_med}}%); Pittsburgh's odd front sent Black straight ahead.

## So what for a team

{{TABLE:card15}}

1. **Build the day-3 film list from it.** Top-third Hoop G linemen went at a median pick of {{r15_th2_pk}}, no earlier than the bottom third ({{r15_th0_pk}}): clubs priced Hoop G at {{r14_price_later}}% later per sd (90%: {{r14_price_later_hi}}% later to {{r14_price_earlier_hi}}% earlier). The shortlist costs no draft capital.
2. **Match the role to the trait.** Shortlisted EDGEs get third-down snaps from wide alignments, where sideways force should matter most; shortlisted interior players get the looper role on stunts. A low Hoop G on a two-gap nose is a role question, not a flag.
3. **Run the hoop twice.** {{r15_sl_onerun}} of {{r15_sl_all_n}} shortlisted linemen have one run; a second, at a pro day or in a camp tracker, lifts reliability from {{rel_hoop_sb}} to {{rel_hoop_2runs}}.

## Limitations

- **Chosen, then tested on the same seasons.** Holding out classes checks the forecast, not the drill choice (search-adjusted p = {{r12_mult_p}}). The shortlist rule was written this round, after earlier rounds had seen the slope; the frozen 2026 test is the clean one.
- **Small and early.** {{r15_sl_n}} shortlisted and {{r15_bs_n}} comparison rushers; {{r15_sl_unscored}} shortlisted linemen have under 30 NFL rushes. The 2025 class has one season and is the weakest in every test.
- **Forecast level.** The forecast is fit on 100+-rush regulars, so most day-3 rushers fall below it; the test compares day-3 rushers with each other, not with 50%.
- **Thin NFL data.** Pressure counts ignore opponents and quarterbacks; sampled tracking has about 20 random rushes per player and no blockers or ball, so scheme reads come from loop rates, not play calls.

## Appendix: every other check

{{TABLE:ledger15}}

*Reproducible: `uv run python writeup/candidates/round_13/cand_1/analysis.py` writes `numbers.json`, `claims.json`, `frozen_rule.json`, `frozen_shortlist.json`, the board CSV and the three figures, then fills this text from `writeup_template.md`.*
