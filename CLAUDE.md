# NFL Big Data Bowl 2027 — working rules for agents

Goal: a finalist-quality Kaggle writeup (≤2,000 words, <10 figures, English) linking 10 Hz Combine tracking to NFL regular-season performance, plus the analysis behind it. Judges are NFL team analysts; rubric Football 30% / Data Science 30% / Writeup 20% / Viz 20%.

## Environment
- Python via `uv` (3.11). Run everything with `uv run …` from the repo root. Package: `src/bdb27`.
- Data lives in `data/` (gitignored). `cloud/setup.sh` restores it: processed tables, nflverse extras and the small raw CSVs always; the 2.3 GB game-tracking only when `KAGGLE_API_TOKEN` is set. If `data/processed/` is empty, run `bash cloud/setup.sh` first.
- `data/processed/*.parquet` are the clean tables (players, combine_events, combine_frames, player_season_stats, player_career, game_plays_sample, game_frames_sample, combine_percentiles, drills, player_value). `data/processed/insights.json` is the output of `analysis/insights.py`.
- Never commit anything under `data/`, never print or commit secrets, never write outside the repo.

## Data facts that matter (profiled; do not re-derive)
- 510 players, 2023–2025 classes, groups WR 107 / TE 42 / OL 121 / DL 118 / DB 122. No QBs or RBs.
- Combine tracking: 6,306 attempts, 52 drills; 75–83% of position drills have a single attempt. Repeatability: 40 ρ≈0.98, shuttle 0.34, backpedal drills ≈0.04.
- Drill medians shift up to 2–3 sd between Combine years: always year-standardize before comparing classes.
- The 40 clip starts ~0.3 s before the laser; tracked 10-yd split is unreliable (r=0.47 vs official); tracked top speed ≈ official 40 (|r|=0.98).
- Game tracking contains only cohort players (no ball, no opponents); in-game movement metrics must be self-referenced.
- `player_play` includes PRE/POST: filter regular season through `games.csv`, never from `game_id`.
- Known signal so far: DL/EDGE pressure rate correlates with 3-cone/40/shuttle (ρ 0.5–0.6); pass-rush-drill peak accel vs in-game get-off ρ=−0.31 (n=69). WR/OL/DB show little from Combine numbers alone.

## Writeup conventions
- Source of truth: `writeup/writeup.md`; figures in `writeup/figures/` produced by scripts in `analysis/` (reproducible, no hand-edited images).
- `uv run python writeup/check.py` validates word count, figure count and that every number in the text appears in `writeup/numbers.json` (the script that generates results must write it).
- Claims need: sample size, a baseline without tracking (official times + NGS scores + draft pick), out-of-class validation, and a limitations section. Judges have seen the public audits saying tracking adds ~0 R²; do not overclaim.
- Iteration state lives in `state/round_*.json` and `state/latest.json` (scores per rubric component, judge feedback, what changed). Read `state/latest.json` before starting a round.

## Git
- Work on the current branch; commit with clear messages; no Co-Authored-By trailers. Push only when the task says so.
