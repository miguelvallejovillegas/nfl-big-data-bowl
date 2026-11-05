# The Combine Start Doesn't Translate; the Stopwatch Plus 20 NFL Rushes Breaks Pass-Rush Ties

**We compared each lineman's Combine 40 start with his own NFL pass-rush starts, using five per-player similarity scores plus the tracked start itself. No estimate exceeds {{eq_estmax}} pressure-rate points per sd beyond draft slot, and {{eq_pass}} of the {{eq_k}} are formally equivalent to zero within ±{{eq_margin}} point. Agreement between two speed readings does carry: rushers fast on the official 40 *and* in about 20 sampled NFL starts ("Both") beat a draft-slot forecast {{sl_ks}} times in {{sl_ns}} on held-out classes. Within {{dr_g}} picks, a later-drafted Both rusher out-rushed the earlier pick {{dr_in_w}} times in {{dr_in_n}}.**

![Figure 1. Four rushers on one scale: his Combine 40 start (orange) over his own NFL starts (blue); grey = league middle half.](figures/fig1_four_rushers.png)

## Why pass rushers

A pass rush starts like a 40: from a stance, straight ahead, before anyone touches him. If Combine tracking translates anywhere, it is here. The other groups fail earlier (Figure 2C). Receivers' starts are readable but do not pay off, and linemen, backs and tight ends start too inconsistently to read from 20 snaps.

## What to do

1. **Tiebreak rule.** Between two rushers of the same class and role, if the later pick is Both, give him the passing-down snaps unless the other went more than {{dr_g}} picks earlier. Inside that gap the rule was wrong {{dr_in_err}}% of the time [{{dr_in_err_lo}}, {{dr_in_err_hi}}]. The draft slot was wrong in {{dr_in_w}} of the same {{dr_in_n}} pairs.
2. **Act on three names (Table 3).** Extend Byron Young (LAR) and YaYa Diaby (TB). Give Chop Robinson (MIA) a full passing-down role, re-scored at midseason because he missed his 2025 check.
3. **Draft room.** Pay for the stopwatch, not the trace: the official 40 time adds {{eqo}} points per sd beyond slot, and the tracked 40 start adds nothing.
4. **Measurement.** Chart {{snaps_for_07}}+ random rushes before trusting a first step (8 rushes give reliability {{rel_8snaps}}).

## The metric and the baseline

**First step = a lineman's speed 0.3 to 0.5 s after he first moves, in sd after weight and role.** In that window the 40 and a pass rush are the same movement. By 1.5 s they part ({{dr_med_40}} yd/s in the 40, {{dr_med_sunday}} on a rush).

**Data.** Tracked Combine 40s, year-standardized, aligned at first movement (so the clip's lead before the laser does not matter). NFL: random sampled pass rushes only, alignment-adjusted, {{n_snaps_dl}} starts by {{n_gplayer}} linemen (median {{snaps_med}}; reliability {{rel_fs_game}}; ρ = {{rho_fs_ngs_snap}} with NGS get-off snap by snap). **Pool:** {{x_pool}} of {{x_all}} linemen have 8+ sampled starts and 100+ rushes. Outcome: career pressures per rush.

**Baseline without tracking:** draft pick, weight and role (EDGE snap share), fitted on the other two classes, with held-out R² {{r2_ols}}. Adding the official 40 and season NGS get-off gives {{r2_full}}. One unit throughout: **pressure-rate points per sd**. The DL average is about 8%, and 1 point is {{rush_season_3}} pressures over a {{rush_season}}-rush season.

## Finding 1: translation does not explain pressure beyond pick

After weight and role, the 40 first step previews the NFL first step at ρ = {{tr_fs}} [{{tr_fs_lo}}, {{tr_fs_hi}}] (n = {{n_tr}}). The raw {{tr_raw_15}} at 1.5 s is mostly body weight. Noise is not the reason: a perfect link would still show up to {{ceil_2rep}}. We ran every per-player translation score through the same held-out test. Each was scored against a margin of ±{{eq_margin}} point per sd, about what the official 40 is worth.

{{TABLE:ledger}}

*Table 1. n = {{eq_n}} rushers with a tracked 40 and 100+ rushes. Each class held out in turn, slopes pooled. TOST = two one-sided tests, 5%. Italic rows: the two readings that do carry over.*

{{eq_pass}} of the {{eq_k}} Combine-side measures are proven to sit inside ±{{eq_margin}}. The two curve-level scores are too noisy to prove it either way. The NFL first step alone has a slope ({{eqn}}). Used continuously, it does not raise held-out R² ({{ooc_sunday_all}} [{{ooc_sunday_all_lo}}, {{ooc_sunday_all_hi}}]), so it is a ranking signal to pair with the stopwatch.

![Figure 2. Translation and payoff by tenth of a second (A); each class held out (B); five groups screened (C; grey = too noisy).](figures/fig2_translation_heldout.png)

## Finding 2: two signals that agree beat the forecast

**The rule.** *Both* = NFL first step in the top third and official 40 time in the top half, with each class's cut-offs set on the other two classes. A rusher "hits" if he beats the held-out forecast (base rate {{sl_base}}%).

{{TABLE:robust}}

*Table 2. Gap interval: rushers resampled within class. p: labels shuffled within class.*

**Verdict: a modest edge, unproven for day 3 ({{sl_day3}} shortlisted).** The gap narrows in every stricter check but stays positive, and it ranges from {{sens_min}} to {{sens_max}} points across cut-offs and pool rules (Table 4). Row 5 is the cleanest test: called on 2023–24 starts and scored on 2025 against each rusher's own past pressure.

**Against a pick-only list, on raw pressure** (top third of his class, no model), Both landed {{bk_both_k}} of {{bk_both_n}} ({{bk_both_lo}}–{{bk_both_hi}}%). The same number of earliest picks also landed {{bk_pick_k}} of {{bk_pick_n}}, at a median pick of {{bk_pick_medpick}} against {{bk_both_medpick}}. Both does not find more production than the top of the draft. It finds the same production about two rounds later.

![Figure 3. A: rushers against held-out cut-offs; top right = shortlist. B: the checks of Table 2.](figures/fig3_shortlist.png)

## The tiebreak as a decision rule

The pool has {{dr_npairs}} same-class, same-role pairs. Without tracking, the earlier pick out-rushes the later one {{dr_pick_a}}% of the time within {{dr_g}} picks and {{dr_pick_b}}% beyond. When the later pick is Both ({{dr_nconf}} pairs, {{dr_conf_n_both}} rushers), he wins {{dr_in_w}} of {{dr_in_n}} within {{dr_g}} picks and {{dr_out_w}} of {{dr_out_n}} beyond. A logistic fit puts break-even at {{dr_breakeven}} picks. A label shuffle within class, clustered by rusher, gives p = {{dr_p}}.

**Held out, the edge is smaller.** We chose the threshold on two classes ({{dr_g_min}}–{{dr_g_max}} picks) and scored it on the third. The rule made {{dr_ho_rule}} correct calls of {{dr_ho_n}}, against {{dr_ho_pick}} for "always the earlier pick": +{{dr_ho_gain}} points [{{dr_ho_gain_lo}}, {{dr_ho_gain_hi}}], an interval that includes zero. Against his nearest-pick twin, each Both rusher won {{mp_w}} of {{mp_n}} career pairs by {{mp_diff}} points [{{mp_diff_lo}}, {{mp_diff_hi}}]. The call frozen after 2024 went {{mpf_w}} of {{mpf_n}} in 2025.

![Figure 4. A: every Both rusher against his draft-slot twin. B: how often the later pick out-rushes the earlier one, by pick gap.](figures/fig4_matched_pairs.png)

**Why it works.** Within a rusher, a snap with a first step one sd faster than his own average has {{sn_or}} times the odds of pressure [{{sn_or_lo}}, {{sn_or_hi}}] ({{sn_n}} rushes; {{sn_or_ctl}} with NGS get-off, down and alignment fixed). **Where rule and tape disagree:** Maason Smith (JAX, pick 48) has the pool's best NFL first step, a {{b_smith_t}}th-percentile 40, and {{b_smith_rate}}% pressure against a {{b_smith_fc}}% forecast. Grade his hands before cutting his snaps.

## The 2026 list, frozen

Same rule, with cut-offs from all three classes. Bold = act now ({{act_minrush}}+ rushes, 1+ point over forecast, under {{avail_cut}}% of games missed).

{{TABLE:list26}}

**What flips each name.** The 40 time is fixed, so only the first step can move, and one 20-rush sample carries about {{err_sd}} sd of noise. Robinson ({{fl2_choprobinson}}), Young ({{fl2_byronyoung}}) and Diaby ({{fl2_yayadiaby}}) sit two noise-widths clear. Adebawore ({{fl2_adetomiwaadeba}}) and Abdullah ({{fl2_yasirabdullah}}) would drop on one slow sample. Murphy is {{murphy_edge}} points over forecast, just short of the 1-point act-on bar. Of the 2024 calls scored on 2025, {{rec_hit}} hit and {{rec_miss}} missed.

**Pre-registration.** The rule, all {{pre_nlist}} names and the test are hashed in `frozen_rule.json` (sha256 `{{rule_hash}}…`; no 2026 data). The test: 2026 list members with 100+ rushes beat a persistence forecast more often than the other {{pre_nrest}} (one-sided Fisher p < {{alpha}}). Power is {{pre_pow_obs}}% at 2025 rates and {{pre_pow_half}}% at half the gap.

## Limitations

- **Not blind until 2026.** The two-signal form and the {{dr_g}}-pick threshold were chosen after seeing all three classes. Only the cut-offs and the threshold check are held out, and the held-out tiebreak gain crosses zero.
- **Small, overlapping samples.** Ten Both rushers, {{sl_day3}} on day 3. Pairs share rushers (Yasir Abdullah carries most losses), so intervals are wide.
- **Sampled tape, no blockers or ball.** A slow start can be a two-gap assignment, and a fast one a free rush. Injuries enter only as missed games.

## Appendix

{{TABLE:sens}}

*Table 4. Shortlist sensitivity; each row re-derives held-out cut-offs.* Re-fits of the {{ht_nfl_pp}}-point slope: rusher random intercepts {{mm_pp}} [{{mm_lo}}, {{mm_hi}}]; weighting for reaching the pool {{ipw_pp}} [{{ipw_lo}}, {{ipw_hi}}]; best of {{ht_nwin}} windows, p = {{ht_nfl_p_adj}}.

*Reproduce: `uv run python writeup/candidates/round_19/cand_3/analysis.py` (about 3 minutes).*
