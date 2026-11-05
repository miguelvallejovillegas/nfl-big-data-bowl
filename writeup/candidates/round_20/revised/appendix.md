# Appendix: robustness, the override rule and the 2026 splits

Rendered by `analysis.py`; every number is in `all_numbers.json` and the starred ones are recomputed by `verify.py`.

## Equivalence and sensitivity

| Check | Result |
|---|---|
| TOST of the held-out slope at ±3 in* | p = 0.11 (benefit arm 0.001, harm arm 0.11) |
| TOST of the hierarchical slope at ±3 in (Wald) | p = 0.04 |
| Held-out slope across 6 outcome builds (route cut-offs 50/100, no cushion term, unweighted, wide receivers only)* | −3.0 to −0.7 in per sd |
| Mechanics, held out: brake into the cut | −1.6 [−4.5, 1.0] |
| Mechanics, held out: speed through the cut | 1.4 [−1.7, 4.5] |
| Mechanics, held out: drive out of the cut | −2.2 [−7.3, 2.5] |
| What 3 inches of SOE is worth | about 0.44 catches a season |

## The pick-gap override rule (dropped from the body: it says the same as the back-test)

Between two day-3 wide receivers less than a round apart (pick gap under 32, fixed before scoring), take the higher last-season SOE. Across 51 same-season contrasts among 11 receivers it overrode the pick 11 times and was wrong 4 times*: 36 per 100 [0, 75]*. It would beat pick order only if wrong less than 64 times per 100* (the break-even point where inches gained on right overrides equal inches lost on wrong ones); the interval spans that line. Net gain 8.1 inches per override [−3.9, 22.7]. Across route cut-offs and raw SOE the wrong rate is 33 to 48 per 100*.

## 2026 list: sampled-snap SOE against the league in the same coverage or alignment

Only 5 of 36 cells reach 30 snaps, so these point film, not grades.

| Receiver | Man, in (snaps) | Zone, in (snaps) | Slot, in (snaps) | Wide, in (snaps) |
|---|---|---|---|---|
| Malik Washington | −1 (5) | +23 (34) | +24 (11) | +19 (26) |
| Demario Douglas | +33 (14) | −13 (26) | +5 (27) | −4 (11) |
| Troy Franklin | −16 (9) | −17 (31) | +7 (12) | −24 (27) |
| Tez Johnson | +4 (14) | +3 (26) | −13 (14) | +9 (25) |
| Arian Smith | −9 (11) | +14 (29) | −6 (17) | +18 (23) |
| Isaiah Bond | +11 (14) | +36 (26) | +25 (16) | +29 (24) |
| Jaylin Lane | −4 (19) | −13 (21) | +3 (13) | −17 (25) |
| Mitch Tinsley | +8 (14) | +4 (26) | +29 (7) | 0 (33) |
| Kayshon Boutte | −7 (9) | +2 (30) | −10 (4) | +1 (35) |
