# Hoop G: A Free Tiebreak for Pass Rushers Who Skip the 3-Cone

**Hoop G is how hard a defensive lineman pushes sideways while turning the Run-the-Hoop curve, compared with his Combine class. Take two rushers from the same class and role whose no-tracking forecasts (pick, weight, NGS score, official times) sit within a point of each other: the one with the clearly higher Hoop G got more NFL pressure in {{r14_tie_win}} of {{r14_tie_n}} such pairs ({{r14_tie_pct}}%; a coin flip gives {{r14_tie_null}}%). Clubs do not pay for it: high-Hoop G linemen were drafted no earlier. The edge is small, about {{r12_sk_all_450}} pressures a season per standard deviation, Hoop G was the best of 14 drills we screened, and we cannot see why it works. Use it to break day-3 ties, not to move boards.**

![Figure 1. Hoop G thirds within each Combine class: NFL pressure against the out-of-class forecast (A), tie-break pairs won by the higher-Hoop G rusher (B), and median draft pick (C).](figures/fig0_gm.png)

## Why it matters

A pressure is what a rusher is paid for: on {{r10_epa_n}} sampled NFL pass plays of our rookies, the offense lost {{r10_epa_diff}} more expected points on pressured snaps (90% interval {{r10_epa_lo}} to {{r10_epa_hi}}).

The 3-cone makes a big man stop, plant and bend around a cone, the hip bend he needs to dip under a tackle's hands. But only {{r10_t3_all_k}} of {{r10_t3_all_n}} linemen in the 2023–25 classes ran an official 3-cone, and even it is unsettled on classes it was not fit on (ρ = {{r10_t3_loco}}, 95% interval {{r10_t3_loco_lo}} to {{r10_t3_loco_hi}}). Another {{r12_rec_8_all}} ran Run-the-Hoop, a figure-eight around two hoops, and no 3-cone. From the 10 Hz Combine tracking we take the hardest half-second of sideways acceleration on that curve and standardize it within year.

![Figure 2. Seven rushers: drill path, hardest sampled NFL rush, and pressure rate minus the out-of-class forecast with 80% intervals and play counts.](figures/fig1_hero.png)

## One test, held out by class, and every other test we ran

**The bar.** Every lineman with 30+ NFL rushes gets a forecast of his pressure rate from pick, weight, EDGE share, NGS score and the official 10-yard split and 3-cone, fit only on the other two draft classes. It ranks rushers in every held-out class, but its spread is uneven (Figure 3B). A shrunken (ridge) version cuts its error from {{r14_rmse_ols}} to {{r14_rmse_ridge}} points without evening the spread.

![Figure 3. Out-of-class forecast against actual pressure rate (A); calibration within each held-out class for the least-squares and ridge forecasts (B).](figures/fig2_calibration.png)

**The test.** Within each class, does Hoop G predict pressure *beyond* that forecast? A count model starts from the forecast and adds one Hoop G slope. Because Hoop G was picked from 14 drills, we shrink the slope toward zero by the size of a typical screened effect. The rule is frozen in `frozen_rule.json` (SHA-256 `{{r12_hash}}`) and will score the 2026 class after the 2027 season unchanged.

{{TABLE:tests14}}

One sd of Hoop G is worth {{r12_sk_all}} more pressures per 100 rushes, {{r12_sk_all_450}} in a 450-rush season, about {{r13_epa_per_sd}} expected point. The sign holds in each class (Figure 4A) and with the ridge forecast. Resampling players keeps the whole interval above zero; dropping any single rusher moves the slope only between {{r14_lopo_min}} and {{r14_lopo_max}}. The honest weakness is the search: priced for picking the best of 14 drills, p = {{r12_mult_p}}.

In plain thirds (Figure 1A), the top Hoop G third out-rushed its forecast by {{r14_th2}} pressures per 450 rushes; the bottom third sat at {{r14_th0}} and the middle at {{r14_th1}}, so the edge lives in the top third, not in a smooth line.

![Figure 4. Held-out slope of pressure over forecast (A) and of 1.5-second depth over forecast (B) on Hoop G, by class and pooled, with the shuffle null.](figures/fig3_heldout.png)

## Does it show up in NFL tracking?

Pressure on 20 sampled rushes is noisy (split-half reliability {{r12_rel_prs}}), so we built a steadier outcome from the game frames: **depth upfield 1.5 seconds after the snap**. It repeats (reliability {{r12_rel_dep}}, {{r12_rel_dep_n}} linemen) and tracks season pressure after weight and role (ρ = {{r12_val_rho}}). Hoop G does not move it (Figure 4B), nor the direction-change rate after the snap.

The pass-rush drill's own get-off does carry over: its first-second burst tracks NFL get-off (ρ = {{r14_d1_game}}, {{r14_d1_game_n}} linemen). But get-off is not pressure (ρ = {{r14_d1_press}} after pick, weight, role and NGS score, {{r14_d1_press_n}} linemen). **So Hoop G is a correlate without a visible mechanism**; it may mark a rusher who keeps speed through contact, which we cannot see without blockers in the data.

## When the call is wrong

{{TABLE:cases12}}

**Johnson** is the case the tiebreak is for: a day-3 EDGE who beat his forecast. **Adebawore** is the miss. **Phillips and Black** carry the same minus flag with opposite results, and scheme is the likely reason: Miami moved Phillips 2+ yards sideways in the first 1.5 seconds on {{r12_phillips_loop}}% of his sampled rushes (interior median {{r12_loop_int_med}}%), where weak sideways force should hurt; Pittsburgh's odd front sent Black straight ahead.

**Cost of a wrong call:** a tiebreak swaps players a median {{r12_pair_pickgap}} picks apart. Where the alternative out-rushed the plus call, the loss was at most {{r11_bad_cost_max}} pressures per 450 rushes, about {{r11_bad_epa_max}} expected points a season.

## So what for a team

{{TABLE:card12}}

1. **Spend it on day 3.** {{r14_plus_day3}} of {{r14_plus_n_all}} plus calls went on day 3 or undrafted (median pick {{r14_plus_pick_med}}). Clubs priced Hoop G at {{r14_price_later}}% later per sd (90%: {{r14_price_later_hi}}% later to {{r14_price_earlier_hi}}% earlier), and it adds nothing to predicting picks in any held-out class: the tiebreak costs no draft capital.
2. **Break ties, do not move boards.** Use it only between rushers that film, and the 3-cone where it exists, cannot separate.
3. **Run the hoop twice.** {{r12_card_onerun}} of these {{r12_card_n}} rushers have one run; a second, at a pro day or in a camp tracker, lifts reliability from {{rel_hoop_sb}} to {{rel_hoop_2runs}}.

## Limitations

- **Selected, then tested on the same seasons.** Holding out classes checks the forecast, not the drill choice; the search-adjusted p = {{r12_mult_p}} is the honest number. The tie-break pairs were defined this round, after the slope test.
- **Small and early.** {{r12_n}} rushers with 30+ rushes; the 2025 class has one season and is the weakest class in the slope and tie-break tests. The 2026 class alone has {{r11_pw_1}}% power under the frozen rule; four classes reach {{r11_pw_full_4}}%.
- **No mechanism.** Hoop G does not show in depth, turning or rush direction on NFL frames.
- **Thin NFL data.** Season counts ignore opponents and the quarterback; sampled tracking has about 20 random rushes per player and no blockers or ball. Scheme reads come from loop rates, not play calls.

*Reproducible: `uv run python writeup/candidates/round_12/cand_1/analysis.py` writes `numbers.json`, `claims.json`, `frozen_rule.json`, the board CSV and the four figures, then fills this text from `writeup_template.md`.*
