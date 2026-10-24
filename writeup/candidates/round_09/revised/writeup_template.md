# Two Leads for Pass Rushers: The 3-Cone, and Hoop G When the 3-Cone Is Missing

**For defensive linemen who ran the 3-cone, that time is our best Combine lead on NFL pressure rate. For the two in three who skipped it, Hoop G, the sideways force he holds while circling the Run-the-Hoop drill, is a second lead. Neither is confirmed on a draft class it was not fit on: use Hoop G to break ties and send scouts to film, not to move a board.**

## Why a tiebreak, and why now

A pressure is the play a pass rusher is paid for: on {{r10_epa_n}} sampled NFL pass plays of our rookies, the offense lost {{r10_epa_diff}} expected points more on snaps he pressured than on snaps he did not [90%: {{r10_epa_lo}}, {{r10_epa_hi}}]. Boards are flattest on day 3, and that is where the stopwatch is thinnest: only {{r10_t3_all_k}} of {{r10_t3_all_n}} linemen in the 2023–25 classes ran an official 3-cone ({{r10_t3_all_pct}}%; 2025: {{r10_t3_25_k}} of {{r10_t3_25_n}}). But {{r9_hoop_all}} of the {{r10_t3_all_n}} ran Run-the-Hoop, and the Combine's 10 Hz tracking turns that circle from a lap time into a force reading.

## The ask

1. **Every DL runs the 3-cone.** Where it exists it is the better number.
2. **Publish each hoop run's peak sideways g** beside the 40. The tracking already supports it.
3. **Time two hoop runs per player.** One run has reliability {{rel_hoop_sb}}; two raise it to {{rel_hoop_2runs}}.
4. **For 2026, stated in advance:** we score the same rule and forecast after the 2027 season, against matched twins (below).

## What Hoop G is, and what it is not

**Hoop G is how hard a lineman can lean into a curve at speed when nobody is blocking him.** It is the peak half-second of acceleration across his path, standardized within Combine year. A typical run peaks at {{hoop_g_median}} g at {{hoop_v_peak}} yd/s ({{hoop_v_mph}} mph): a circle about {{hoop_radius}} yards in radius. Single runs are noisy, so every number shrinks toward average.

It is not a measure of how he wins. In sampled NFL rushes, a high Hoop G does not show up as harder turning at matched speed (EDGE: {{mx9_e}} g per sd [{{mx9_e_lo}}, {{mx9_e_hi}}]) or as more outside rushes ({{r10_dir_n}} EDGEs, ρ = {{r10_dir_rho}}). So it speaks to two situations a scout must still confirm on film: the speed rush from a wide-9 or ghost alignment, where the rusher has to flatten at the top, and the looper on a T-E or E-T stunt, who has to turn tight behind the crash. It says nothing about a bull rush.

![Figure 1. Javon Solomon (plus call, beat his forecast), Donovan Ezeiruaku (2025 plus call, not yet) and Jordan Phillips (2025 minus flag, below forecast). Left: hoop path. Middle: his hardest-turning sampled NFL rush on the same scale. Right: rookie-to-date pressure rate against the no-tracking forecast.](figures/fig1_hero.png)

Solomon (pick {{f9_solomon_pick}}, no 3-cone) is the use case: {{f9_solomon_hoop}} g in the hoop, then {{f9_solomon_pr}}% pressure against a {{f9_solomon_pb}}% forecast over {{f9_solomon_n}} rushes. Phillips ({{f9_phillips_wt}} lb) pairs a low hoop with the slowest 2025 interior 3-cone: {{f9_phillips_pr}}% against {{f9_phillips_pb}}%.

High and low hoops among EDGEs, with the 3-cone and how their sampled wide rushes left the line:

{{TABLE:edges}}

The two high hoops win differently: Solomon goes straight upfield, Ezeiruaku takes the inside move half the time. Hoop G ranks the bend a man has, not the move he uses; the tape tells you whether he uses it.

## The track record, player by player

The rule is frozen: **a plus call needs the 80% interval of a player's noise-corrected Hoop G to clear zero; a minus flag needs it below zero.** Each rusher with 30+ NFL rushes is compared with a no-tracking forecast (pick, weight, EDGE share, NGS score, official 10-yd split and 3-cone) fit on the other two classes; he "beats" it when the 80% interval of his rate sits above it.

{{TABLE:coach}}

Plus calls went {{fr_all_k}} of {{fr_all_n}}, every hit a 2023–24 EDGE; rushers without a call went {{r9_nc_k}} of {{r9_nc_n}}. By name: {{TABLE:name_hit}} (pick {{r10_hit_pick}}) rushed at {{r10_hit_pr}}% against a {{r10_hit_pb}}% forecast; {{TABLE:name_miss}} (pick {{r10_miss_pick}}), a plus call from the same rule, at {{r10_miss_pr}}% against {{r10_miss_pb}}%, inside the noise.

A fairer test puts each plus call beside the no-call rusher a scout would have weighed him against: same class and role, nearest in pick and Combine weight.

![Figure 2. Each scored plus call beside his matched no-call twin, with Hoop G against his class (left) and pressure rate minus forecast (right). Pooled groups, including the minus flags, on the same axis at the bottom.](figures/fig2_twins.png)

Plus calls out-rushed their twins' rate over forecast in {{r10_tw_better}} of {{r10_tw_n}} pairs, by {{r10_tw_diff}} points on average, and {{r10_tw25_better}} of {{r10_tw25_n}} in 2025. The {{co_mi_n}} interior minus flags sit {{r10_pool_minus}} points under forecast, but weight explains part of a low interior hoop, so that flag sends you to film, not down the board.

## The 2025 decision card

{{TABLE:card10}}

Film checklist and tie-breaks:

{{TABLE:film}}

Where a 3-cone exists, it decides first; Brinson, a day-3 interior lineman with no 3-cone, is the clean use case.

## How sure we are

{{TABLE:tests10}}

Quote the {{mult_p}}: it is the only p-value that prices our search over 14 drill–outcome pairs, so the calls are leads, not rules.

## Limitations

- **Not confirmed.** Out of class, both leads' intervals span zero; a confirmatory test needs about {{n_power80}} rushers.
- **Small and early.** {{fr_all_n}} scored plus calls; one rookie season is too short for anyone to clear the bar. The twin test was added after the rule, on the same rushers.
- **Thin NFL data.** Season pressure counts ignore rotation, injuries, opponents and scheme. Sampled tracking has about {{arc_snaps_per_player}} rushes per player and no blockers.
- **Where both exist, the 3-cone wins.** Hoop G is a stand-in, not an addition.

*Reproducible: `uv run python writeup/candidates/round_09/revised/analysis.py` writes `numbers.json`, `claims.json`, the board CSV and both figures, then fills this text from `writeup_template.md`.*
