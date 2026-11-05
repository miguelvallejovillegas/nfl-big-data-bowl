# Hoop G: A Tiebreak Lead for Pass Rushers Who Skip the 3-Cone

**Hoop G, a measure of sideways force in Run-the-Hoop, is a tiebreak for pass rushers who skipped the 3-cone. Scored against a no-tracking forecast built on other draft classes, each standard deviation of Hoop G came with {{r12_sk_all}} more pressures per 100 rushes, positive in all {{r12_sk_pos}} classes though no single class clears zero. Its plus calls beat the forecast {{fr_all_k}} of {{fr_all_n}} times, against {{r9_nc_k}} of {{r9_nc_n}} for other rushers, but 0 of {{fr_clean_n}} in 2025. It was the best of 14 drills we screened, it leaves no trace in NFL rush tracking, and the 3-cone it stands in for is itself unconfirmed out of class. A lead, not a rule.**

## Why it matters

A pressure is what a rusher is paid for: on {{r10_epa_n}} sampled NFL pass plays of our rookies, the offense lost {{r10_epa_diff}} more expected points on pressured snaps (90% interval {{r10_epa_lo}} to {{r10_epa_hi}}).

The 3-cone makes a big man stop, plant and bend around a cone low to the ground, the hip bend he needs to dip under a tackle's hands. But only {{r10_t3_all_k}} of {{r10_t3_all_n}} linemen in the 2023–25 classes ran an official 3-cone, and on a class it was not fit on, its link to pressure is ρ = {{r10_t3_loco}} with a 95% interval from {{r10_t3_loco_lo}} to {{r10_t3_loco_hi}}, carried by {{r12_t3_2025_n}} rushers in 2025 (Figure 2B). Another {{r12_rec_8_all}} linemen ran Run-the-Hoop, a figure-eight around two hoops, without a 3-cone.

**Hoop G is how hard a lineman can push sideways while turning the hoop curve, compared with his Combine class.** Technically: the peak half-second of sideways acceleration in the 10 Hz Combine tracking, standardized within year.

![Figure 1. Seven rushers: drill path, hardest sampled NFL rush, and pressure rate minus the out-of-class forecast with 80% intervals and play counts.](figures/fig1_hero.png)

## One test, held out by class

**Frozen rule** (written to `frozen_rule.json`, SHA-256 `{{r12_hash}}`, before the 2026 class has NFL data): every rusher with 30+ NFL rushes gets a no-tracking forecast of his pressure rate (pick, weight, EDGE share, NGS score, official 10-yard split and 3-cone) fit on the other two draft classes. Figure 2A shows it holds out of class (weighted slope {{r11_cal_slope}}, 90% interval {{r11_cal_slope_lo}} to {{r11_cal_slope_hi}}).

![Figure 2. Out-of-class forecast against actual pressure rate with a binned calibration line and the 2025 plus calls marked (A); the 3-cone's held-out partial correlation by class (B).](figures/fig2_calibration.png)

**The test:** within each class, does Hoop G predict pressure *beyond* that forecast? A count model uses the forecast as its starting point and shrinks the Hoop G slope toward zero with a skeptical prior. No cutoff, no minimum beyond 30 rushes.

![Figure 3. Held-out slope of pressure over forecast (A) and of 1.5-second depth over forecast (B) on Hoop G, by class and pooled, with the shuffle null.](figures/fig3_heldout.png)

The slope is positive in 2023 ({{r12_sk_2023}}), 2024 ({{r12_sk_2024}}) and 2025 ({{r12_sk_2025}}) pressures per 100 rushes per sd; each class alone spans zero. Pooled over {{r12_n}} rushers it is {{r12_sk_all}} (90% interval {{r12_sk_all_lo}} to {{r12_sk_all_hi}}), about {{r12_sk_all_450}} pressures in a 450-rush season. Shuffling Hoop G within class gives p = {{r12_sk_pshuf}}, **but Hoop G was the best of 14 drill–outcome pairs screened on these seasons; adjusted for that search, p = {{r12_mult_p}}**.

For a scout, the same signal as calls: a plus call needs the 80% interval of a player's Hoop G above zero, and a hit means his pressure rate clearly beat his forecast.

{{TABLE:coach12}}

## Does it show up in NFL tracking?

Season pressure on 20 sampled rushes is noisy (split-half reliability {{r12_rel_prs}}), so we built a steadier outcome from the same 10 Hz game frames: **depth upfield 1.5 seconds after the snap**, on random NFL pass rushes. It is repeatable (reliability {{r12_rel_dep}}, {{r12_rel_dep_n}} linemen) and it tracks season pressure after weight and role (ρ = {{r12_val_rho}}, 90% interval {{r12_val_lo}} to {{r12_val_hi}}, {{r12_val_n}} linemen).

Hoop G does not move it. With the same held-out design, one sd of Hoop G is worth {{r12_dep_all_in}} inches of depth (90% interval {{r12_dep_all_inlo}} to {{r12_dep_all_inhi}}; shuffle p = {{r12_dep_pshuf}}; Figure 3B). The direction-change rate after the snap, the other proxy we tried, is unreliable ({{r12_rel_turn}}) and unrelated (ρ = {{r12_turn_rho}}, {{r12_turn_n}} rushers). Earlier tests of matched-speed turning and outside-rush share were null too (Appendix).

**So the scout use is a tiebreak after the 3-cone and film**, not a skill we can see on the field.

## When the call is wrong

{{TABLE:cases12}}

**Johnson** is the case the tiebreak is for: a day-3 EDGE whose rate beat his forecast. **Adebawore** is the plus-call miss: an interior rusher who met his forecast and never beat it. **Phillips and Black** carry the same minus flag with opposite results, and scheme is the likely reason. Miami moved Phillips 2+ yards sideways in the first 1.5 seconds, a stunt or slant, on {{r12_phillips_loop}}% of his sampled rushes (interior median {{r12_loop_int_med}}%), exactly where weak sideways force should hurt; he sits {{r11_phillips_p450}} pressures per 450 rushes under forecast. Pittsburgh's odd front sent Black straight ahead ({{r12_black_loop}}% such rushes), and he sits {{r11_black_p450}} over. A minus flag is a question for stunt-heavy fronts, not two-gap noses.

**Cost of a wrong call:** a tiebreak swaps players a median {{r12_pair_pickgap}} picks apart. Where the alternative out-rushed the plus call, the loss was at most {{r11_bad_cost_max}} pressures per 450 rushes, about {{r11_bad_epa_max}} expected points a season.

## So what for a team

{{TABLE:card12}}

Three actions:

1. **Break ties, do not move boards.** Use Hoop G only between players the 3-cone and film cannot separate.
2. **Run the hoop twice.** {{r12_card_onerun}} of these {{r12_card_n}} rushers have one run; a second, at a pro day or in a camp tracker, lifts reliability from {{rel_hoop_sb}} to {{rel_hoop_2runs}}.
3. **Match flags to scheme.** Phillips's flag matters for a stunt-heavy front; Black's barely matters for a two-gap nose.

## Limitations

- **Selected, then tested on the same seasons.** The adjusted p = {{r12_mult_p}} is the honest number; the held-out classes check the forecast, not the drill choice.
- **Small and early.** {{fr_all_n}} scored plus calls; the 2025 class has one season. The 2026 class, scored after 2027 under the frozen rule, has {{r11_pw_1}}% power alone; four classes reach {{r11_pw_full_4}}%, or {{r11_pw_halfgap_4}}% if the true gap is half as large.
- **No mechanism.** Hoop G does not show in depth, turning or rush direction on NFL frames.
- **Thin NFL data.** Season counts ignore opponents; sampled tracking has about 20 random rushes per player and no blockers or ball. Scheme reads come from loop rates, not play calls.

## Appendix

{{TABLE:appx12}}

{{TABLE:recon}}

*Reproducible: `uv run python writeup/candidates/round_10/revised/analysis.py` writes `numbers.json`, `claims.json`, `frozen_rule.json`, the board CSV and the three figures, then fills this text from `writeup_template.md`.*
