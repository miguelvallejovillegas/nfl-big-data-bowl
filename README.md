# NFL Big Data Bowl 2027 — Combine tracking → NFL performance

Analysis for the [2027 Big Data Bowl](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027): 510 prospects from the 2023–2025 draft classes, tracked at 10 Hz during Scouting Combine drills and on every NFL regular-season snap.

**Thesis.** Instead of asking whether Combine kinematics predict NFL production better than the stopwatch (several public audits say they barely do), measure for each player how faithfully the movement profile from a drill shows up in NFL plays of the same type — and ask what that *translation* explains.

## Layout

```
etl/            01_to_parquet.py  (CSV → parquet, once)
                02_build_tables.py (clean, kinematics, aggregates → data/processed)
src/bdb27/        clean.py (drill aliases, dropped events, REG filter)
                kinematics.py (Savitzky–Golay velocities, signed accel, break detection, splits)
                drills.py (drill catalog with plain-language descriptions)
notebooks/      numbered exploration notebooks
tests/          pytest sanity checks on the pipeline
writeup/        Kaggle writeup and figures
data/           raw | parquet | external | processed (gitignored; see below)
```

## Setup

```
uv sync --extra dev
# put the competition CSVs in data/raw/, then:
uv run python etl/01_to_parquet.py
uv run python etl/02_build_tables.py
uv run pytest
```

External data (CC-BY, [nflverse](https://github.com/nflverse/nflverse-data)): `players.csv`, `combine.csv`, `draft_picks.csv`, `historical_contracts.parquet` into `data/external/nflverse/`.

## Data quirks handled

- Drill names drift across years (`PASS_PRO-MIRROR_DRILL` / `PASS_PRO_MIRROR_DRILL`, `LINE` / `LINE_DRILL`, …): `DRILL_ALIASES`.
- Four glitch events (10–12 frame drills, a 202-frame shuttle) are dropped.
- `attempt` is a running counter per session; `attempt_no` is recomputed per player × drill.
- Provided `dis` under-counts and `a` is unsigned: kinematics are recomputed from x/y.
- `player_play` includes preseason and playoffs; season cannot be read from `game_id` (January games carry the next year's prefix). Regular season is taken from `games.csv`.
- WR/TE rows exist only on pass plays where the player ran a route.

## Dashboard

An interactive explorer (player cards, drill explorer with animated attempts, leaderboards) lives in a separate private repo and reads the tables produced by `etl/02_build_tables.py`.

Data license: competition data CC BY-NC 4.0.
