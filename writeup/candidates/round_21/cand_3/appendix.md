# Appendix: robustness, the day-3 film order and its splits

Rendered by `analysis.py`; every number is in `all_numbers.json` and the starred ones are recomputed by `verify.py`.

## Equivalence and sensitivity

| Check | Result |
|---|---|
| Hierarchical model, receiver and class effects | −0.5 [−2.8, +1.8] in per sd (167 player-seasons, 91 receivers) |
| Raw drill average, no error correction* | −1.0 [−2.8, +0.8] |
| TOST of the held-out slope at ±3 in* | p = 0.11 (benefit arm 0.001, harm arm 0.11) |
| TOST of the hierarchical slope at ±3 in (Wald) | p = 0.04 |
| Held-out slope across 6 outcome builds (route cut-offs 50/100, no cushion term, unweighted, wide receivers only)* | −3.0 to −0.7 in per sd |
| Mechanics, held out: brake into the cut | −1.6 [−4.5, 1.0] |
| Mechanics, held out: speed through the cut | 1.4 [−1.7, 4.5] |
| Mechanics, held out: drive out of the cut | −2.2 [−7.3, 2.5] |
| Cut Match: change in held-out R² | −0.056 |
| Cut Match raw gap vs drill reps (why the score nets out rep counts) | ρ = −0.49 |
| Break G, five drills vs the other five | 0.28 [0.15, 0.40] (145 receivers) |
| What 3 inches of SOE is worth | about 0.44 catches a season |

## Day-3 film order (a coin-flip prioritization tool, not a board)

Of 12 day-3 receiver-seasons above zero, 6 stayed above zero the next season (50% [28, 72]). Rule: day-3 or undrafted wide receiver, 14+ games, 100+ routes and SOE above zero in 2025; GO = also above zero in 2024. "Off the list below" is the SOE over his first 150 routes of 2026 that pulls true SOE under zero.

| Film order: receiver (class, pick; 2025 team) | True SOE, in [90%] | 2025 games / routes | Role and roster change, 2024 → 2025 | Off the list below |
|---|---|---|---|---|
| 1. **Malik Washington** (2024, R6-184; MIA) | +16.1 [+7.7, +24.4] | 17 / 339 | routes per game 16 → 20; main spot slot → wide | −51 in |
| 2. **Demario Douglas** (2023, R6-210; NE) | +13.4 [+3.7, +23.0] | 17 / 220 | routes per game 25 → 13 | −32 in |
| 3. **Troy Franklin** (2024, R4-102; DEN) | +3.4 [−4.0, +10.9] | 17 / 461 | routes per game 16 → 27 | −14 in |

HOLD: 4. Tez Johnson (TB, +10.9); 5. Arian Smith (NYJ, +10.0; reserve list); 6. Isaiah Bond (CLE, +3.1); 7. Jaylin Lane (WAS, +2.8; reserve list); 8. Mitch Tinsley (CIN, +1.8; 134 routes); 9. Kayshon Boutte (NE, +0.6; now HOU)

Sampled-snap SOE against the league in the same coverage or alignment (only 5 of 36 cells reach 30 snaps):

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

## The pick-gap override rule

Between two day-3 wide receivers less than a round apart (pick gap under 32), take the higher last-season SOE. Across 51 contrasts among 11 receivers it overrode the pick 11 times and was wrong 4 times*: 36 per 100 [0, 75]*, against a break-even of 64*. Across route cut-offs and raw SOE the wrong rate is 33 to 48 per 100*.
