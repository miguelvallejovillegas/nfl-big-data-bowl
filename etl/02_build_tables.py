"""Build the processed tables that feed Supabase and the notebooks.

Usage: uv run python etl/02_build_tables.py
Outputs go to data/processed/*.parquet
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdb27.clean import NFL_POS_TO_GROUP, clean_combine_tracking, regular_season_games  # noqa: E402
from bdb27.drills import DRILLS  # noqa: E402
from bdb27.kinematics import event_features, frame_kinematics  # noqa: E402

PQ = ROOT / "data" / "parquet"
EXT = ROOT / "data" / "external" / "nflverse"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

NFLVERSE_POS_TO_GROUP = {
    "WR": "WR", "TE": "TE",
    "OT": "OL", "OG": "OL", "G": "OL", "C": "OL", "OL": "OL",
    "DT": "DL", "DE": "DL", "EDGE": "DL", "DL": "DL", "OLB": "DL",
    "CB": "DB", "S": "DB", "SAF": "DB", "DB": "DB",
}
# measurable -> (nflverse column, lower is better)
MEASURABLES = {
    "combine_height": ("ht", False),
    "combine_weight": ("wt", False),
    "forty": ("forty", True),
    "vertical": ("vertical", False),
    "broad_jump": ("broad_jump", False),
    "three_cone": ("cone", True),
    "short_shuttle": ("shuttle", True),
    "bench_reps": ("bench", False),
    "ten_yd_split": (None, True),
    "hand_size": (None, False),
    "arm_length": (None, False),
    "wing_span": (None, False),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pct_better(values: np.ndarray, pool: np.ndarray, lower_is_better: bool) -> np.ndarray:
    """Percent of pool that `values` beat (0-100). NaN stays NaN."""
    pool = pool[~np.isnan(pool)]
    pool.sort()
    out = np.full(len(values), np.nan)
    ok = ~np.isnan(values)
    if len(pool) == 0:
        return out
    if lower_is_better:
        out[ok] = 100.0 * (len(pool) - np.searchsorted(pool, values[ok], side="right")) / len(pool)
    else:
        out[ok] = 100.0 * np.searchsorted(pool, values[ok], side="left") / len(pool)
    return np.round(out, 1)


# ----------------------------------------------------------------------------- players
def build_players() -> pl.DataFrame:
    p = pl.read_parquet(PQ / "players.parquet")
    cr = pl.read_parquet(PQ / "combine_results.parquet")
    nv = (
        pl.read_csv(EXT / "players.csv", infer_schema_length=0)
        .filter(pl.col("nfl_id").is_not_null())
        .with_columns(pl.col("nfl_id").cast(pl.Int64))
        .select(
            "nfl_id", "gsis_id", "espn_id", "pfr_id", "otc_id", "pff_id",
            pl.col("headshot").alias("headshot_url"), "latest_team", "jersey_number", "status",
            pl.col("position").alias("nflverse_position"),
        )
        .unique(subset=["nfl_id"], keep="first")
    )
    df = (
        p.join(cr.select("nfl_id", "combine_position", "combine_height", "combine_weight"), on="nfl_id", how="left")
        .join(nv, on="nfl_id", how="left")
        .with_columns(
            pl.col("nfl_position").replace(NFL_POS_TO_GROUP).alias("position_group"),
            pl.col("draft_round").is_null().alias("is_udfa"),
            pl.col("birth_date").str.to_date("%Y-%m-%d").alias("birth_date"),
        )
        .with_columns(
            ((pl.date(pl.col("draft_year"), 4, 25) - pl.col("birth_date")).dt.total_days() / 365.25)
            .round(1)
            .alias("age_at_draft"),
            pl.when(pl.col("headshot_url").is_null() & pl.col("espn_id").is_not_null())
            .then(pl.format("https://a.espncdn.com/i/headshots/nfl/players/full/{}.png", pl.col("espn_id")))
            .otherwise(pl.col("headshot_url"))
            .alias("headshot_url"),
        )
        .sort("draft_year", "draft_overall_pick", nulls_last=True)
    )
    return df


# ----------------------------------------------------------------------------- combine results + percentiles
def build_combine(players: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    cr = pl.read_parquet(PQ / "combine_results.parquet")
    hist = pl.read_csv(EXT / "combine.csv", infer_schema_length=0).with_columns(
        pl.col("pos").replace_strict(NFLVERSE_POS_TO_GROUP, default=None).alias("group"),
        pl.col("season").cast(pl.Int64),
    ).filter(pl.col("group").is_not_null())
    # height "6-2" -> inches
    hist = hist.with_columns(
        (pl.col("ht").str.split("-").list.get(0).cast(pl.Float64, strict=False) * 12
         + pl.col("ht").str.split("-").list.get(1).cast(pl.Float64, strict=False)).alias("ht"),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in ["wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]],
    )
    rows = []
    for metric, (hcol, lower) in MEASURABLES.items():
        for grp in ["WR", "TE", "OL", "DL", "DB"]:
            sub = cr.filter(pl.col("combine_position") == grp)
            vals = sub[metric].cast(pl.Float64).to_numpy()
            cohort = pct_better(vals, vals.copy(), lower)
            if hcol:
                pool = hist.filter(pl.col("group") == grp)[hcol].to_numpy().astype(float)
                hp = pct_better(vals, pool, lower)
                n_hist = int((~np.isnan(pool)).sum())
            else:
                hp = np.full(len(vals), np.nan)
                n_hist = 0
            rows.append(pl.DataFrame({
                "nfl_id": sub["nfl_id"], "metric": metric, "value": vals,
                "pct_cohort": cohort, "pct_hist": hp, "n_hist": n_hist,
            }))
    pcts = pl.concat(rows).filter(pl.col("value").is_not_null()).with_columns(
        pl.col("pct_cohort").fill_nan(None), pl.col("pct_hist").fill_nan(None)
    )
    return cr, pcts


# ----------------------------------------------------------------------------- combine tracking
def build_combine_tracking(players: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    ct = pl.read_parquet(PQ / "combine_tracking.parquet")
    frames, dropped = clean_combine_tracking(ct)
    log(f"combine frames {frames.height:,} (dropped {dropped.height} events)")

    # per-frame derived columns (PLAYER rows)
    pf = frames.filter(pl.col("entity_type") == "PLAYER").sort("event_id", "frame")
    accel, heading, vx, vy = [], [], [], []
    feats = []
    for (eid,), g in pf.group_by("event_id", maintain_order=True):
        x, y, s = g["x"].to_numpy(), g["y"].to_numpy(), g["s"].to_numpy()
        k = frame_kinematics(x, y, s)
        accel.append(k["accel"]); heading.append(k["heading"]); vx.append(k["vx"]); vy.append(k["vy"])
        f = event_features(x, y, s, g["drill_type"][0])
        f.update(event_id=eid, nfl_id=int(g["nfl_id"][0]), draft_year=int(g["draft_year"][0]),
                 drill_type=g["drill_type"][0], drill_group=g["drill_group"][0], drill_name=g["drill_name"][0],
                 attempt_no=int(g["attempt_no"][0]), start_time=g["time"][0],
                 x0=round(float(x[0]), 2), y0=round(float(y[0]), 2))
        feats.append(f)
    pf = pf.with_columns(
        pl.Series("accel", np.round(np.concatenate(accel), 3)),
        pl.Series("heading", np.round(np.concatenate(heading), 1)),
        pl.Series("vx", np.round(np.concatenate(vx), 3)),
        pl.Series("vy", np.round(np.concatenate(vy), 3)),
    )
    bf = frames.filter(pl.col("entity_type") == "BALL").with_columns(
        pl.lit(None, pl.Float64).alias("accel"), pl.lit(None, pl.Float64).alias("heading"),
        pl.lit(None, pl.Float64).alias("vx"), pl.lit(None, pl.Float64).alias("vy"),
    )
    out_frames = pl.concat([pf, bf.select(pf.columns)]).select(
        "event_id", "frame", "t", "entity_type", "x", "y", "s", "a", "dis", "dir", "accel", "heading", "vx", "vy"
    ).sort("event_id", "entity_type", "frame")

    events = pl.DataFrame(feats)
    has_ball = frames.filter(pl.col("entity_type") == "BALL").select("event_id").unique().with_columns(pl.lit(True).alias("has_ball"))
    events = events.join(has_ball, on="event_id", how="left").with_columns(pl.col("has_ball").fill_null(False))
    events = events.join(players.select("nfl_id", "position_group", "nfl_position", "display_name"), on="nfl_id", how="left")

    # key metric percentile within drill (direction-aware)
    cat = {d["name"]: d for d in DRILLS}
    km_val, km_pct = np.full(events.height, np.nan), np.full(events.height, np.nan)
    names = events["drill_name"].to_list()
    for name in set(names):
        d = cat.get(name)
        if not d:
            continue
        idx = np.array([i for i, n in enumerate(names) if n == name])
        vals = events[d["key_metric"]].cast(pl.Float64).to_numpy()[idx]
        km_val[idx] = vals
        km_pct[idx] = pct_better(vals, vals.copy(), d["lower_is_better"])
    events = events.with_columns(
        pl.Series("key_metric_value", km_val).fill_nan(None), pl.Series("key_metric_pct", km_pct).fill_nan(None)
    )
    # any remaining float NaN (e.g. straightness on zero-length paths) becomes null
    events = events.with_columns([pl.col(c).fill_nan(None) for c, t in events.schema.items() if t == pl.Float64])
    return out_frames, events, dropped


def build_drills(events: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """Catalog + per-drill stats + mean trajectory (50 points, relative to start, for the diagram)."""
    stats = events.group_by("drill_name").agg(
        pl.len().alias("n_events"), pl.col("nfl_id").n_unique().alias("n_players"),
        pl.col("draft_year").unique().sort().alias("years"),
        pl.col("duration_s").median().round(1).alias("median_duration_s"),
        pl.col("top_speed").median().round(2).alias("median_top_speed"),
        pl.col("path_length_yd").median().round(1).alias("median_path_yd"),
        pl.col("n_breaks").median().alias("median_breaks"),
        pl.col("has_ball").any().alias("has_ball"),
    )
    # mean trajectory: resample every event to 50 points on normalized time, relative to x0,y0
    pf = frames.filter(pl.col("entity_type") == "PLAYER").join(events.select("event_id", "drill_name", "x0", "y0"), on="event_id")
    pf = pf.with_columns((pl.col("x") - pl.col("x0")).alias("rx"), (pl.col("y") - pl.col("y0")).alias("ry"))
    trajs: dict[str, list] = {}
    for (name,), g in pf.group_by("drill_name"):
        xs, ys = [], []
        for (_,), e in g.group_by("event_id"):
            e = e.sort("frame")
            n = e.height
            u = np.linspace(0, 1, 50)
            tt = np.linspace(0, 1, n)
            xs.append(np.interp(u, tt, e["rx"].to_numpy())); ys.append(np.interp(u, tt, e["ry"].to_numpy()))
        trajs[name] = [[round(float(a), 2), round(float(b), 2)] for a, b in zip(np.median(xs, axis=0), np.median(ys, axis=0))]
    cat = pl.DataFrame(DRILLS)
    df = cat.join(stats, left_on="name", right_on="drill_name", how="left").with_columns(
        pl.col("name").map_elements(lambda n: json.dumps(trajs.get(n, [])), return_dtype=pl.String).alias("mean_path")
    )
    missing = set(events["drill_name"].unique()) - set(cat["name"])
    if missing:
        log(f"WARNING drills missing from catalog: {missing}")
    return df


# ----------------------------------------------------------------------------- NFL season stats
def build_season_stats(players: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    pp = pl.read_parquet(PQ / "player_play.parquet")
    games = regular_season_games(pl.read_parquet(PQ / "games.parquet"))
    pp = pp.join(games, on="game_id", how="inner")  # REG only
    pp = pp.filter(pl.col("play_nullified_by_penalty") != "Y")
    pp = pp.join(players.select("nfl_id", "position_group"), on="nfl_id", how="left")
    b = lambda c: pl.col(c).cast(pl.Float64)  # noqa: E731
    caught = pl.col("target") & (pl.col("pass_result") == "C")
    stats = pp.group_by("nfl_id", "season", "position_group").agg(
        pl.len().alias("plays"),
        pl.col("game_id").n_unique().alias("games"),
        pl.col("team_abbr").mode().first().alias("team"),
        pl.col("lined_up_position").mode().first().alias("main_alignment"),
        pl.col("expected_points_added").mean().round(3).alias("epa_per_play"),
        # receiving
        pl.col("route_ran").is_not_null().sum().alias("routes"),
        pl.col("target").sum().alias("targets"),
        caught.sum().alias("receptions"),
        pl.when(caught).then(b("rec_yards")).otherwise(0).sum().alias("rec_yards"),
        pl.when(caught).then(b("yards_after_catch")).otherwise(None).sum().alias("yac"),
        pl.when(caught).then(b("expected_yards_after_catch")).otherwise(None).sum().alias("xyac"),
        pl.col("separation_at_pass_forward").mean().round(2).alias("avg_separation"),
        pl.when(pl.col("target")).then(pl.col("separation_at_pass_forward")).mean().round(2).alias("avg_separation_targeted"),
        pl.col("cushion").mean().round(2).alias("avg_cushion"),
        pl.when(pl.col("target")).then(pl.col("expected_points_added")).mean().round(3).alias("epa_per_target"),
        (pl.col("team_coverage_man_zone") == "MAN_COVERAGE").sum().alias("plays_vs_man"),
        # pass protection
        pl.col("pass_rushers_encountered").is_not_null().sum().alias("pass_blocks"),
        pl.col("pressure_allowed").cast(pl.Int64).sum().alias("pressures_allowed"),
        b("sack_allowed").sum().alias("sacks_allowed"),
        pl.col("peak_pressure_probability_allowed").mean().round(3).alias("avg_peak_pressure_prob"),
        pl.col("pass_rushers_encountered").mean().round(2).alias("avg_rushers_faced"),
        # pass rush / defense
        pl.col("player_get_off").is_not_null().sum().alias("pass_rush_snaps"),
        pl.col("player_get_off").mean().round(3).alias("avg_get_off"),
        pl.col("player_get_off").quantile(0.1).round(3).alias("get_off_p10"),
        pl.col("time_to_pressure").is_not_null().sum().alias("pressures"),
        pl.col("quick_pressure").cast(pl.Int64).sum().alias("quick_pressures"),
        pl.col("time_to_pressure").mean().round(2).alias("avg_time_to_pressure"),
        b("sack").sum().alias("sacks"),
        b("tackle").sum().alias("tackles"),
        b("assist").sum().alias("assists"),
        pl.col("tackle_for_loss").cast(pl.Int64).sum().alias("tfl"),
        b("caused_forced_fumble").sum().alias("forced_fumbles"),
        pl.col("coverage_assignment").is_not_null().sum().alias("coverage_snaps"),
        (pl.col("coverage_assignment") == "MAN").sum().alias("man_coverage_snaps"),
        pl.col("blitzing").cast(pl.Int64).sum().alias("blitzes"),
    ).with_columns(
        (pl.col("pressures_allowed") / pl.col("pass_blocks")).round(3).alias("pressure_rate_allowed"),
        (pl.col("pressures") / pl.col("pass_rush_snaps")).round(3).alias("pressure_rate"),
        (pl.col("targets") / pl.col("routes")).round(3).alias("target_rate"),
        (pl.col("rec_yards") / pl.col("routes")).round(2).alias("yards_per_route"),
        (pl.col("yac") - pl.col("xyac")).round(1).alias("yac_over_expected"),
    ).sort("nfl_id", "season")
    # null out fields that don't apply to the group (keeps the dashboard honest)
    recv = pl.col("position_group").is_in(["WR", "TE"])
    ol = pl.col("position_group") == "OL"
    dl = pl.col("position_group") == "DL"
    db = pl.col("position_group") == "DB"
    stats = stats.with_columns(
        *[pl.when(recv).then(pl.col(c)).otherwise(None).alias(c) for c in
          ["routes", "targets", "receptions", "rec_yards", "yac", "xyac", "avg_separation", "avg_separation_targeted",
           "avg_cushion", "epa_per_target", "target_rate", "yards_per_route", "yac_over_expected", "plays_vs_man"]],
        *[pl.when(ol).then(pl.col(c)).otherwise(None).alias(c) for c in
          ["pass_blocks", "pressures_allowed", "sacks_allowed", "avg_peak_pressure_prob", "avg_rushers_faced", "pressure_rate_allowed"]],
        *[pl.when(dl | db).then(pl.col(c)).otherwise(None).alias(c) for c in
          ["pass_rush_snaps", "avg_get_off", "get_off_p10", "pressures", "quick_pressures", "avg_time_to_pressure", "pressure_rate",
           "sacks", "tackles", "assists", "tfl", "forced_fumbles", "coverage_snaps", "man_coverage_snaps", "blitzes"]],
    )
    align = pp.with_columns(pl.col("lined_up_position").fill_null("NA")).group_by("nfl_id", "season", "lined_up_position").agg(pl.len().alias("plays")).sort("nfl_id", "season", "plays", descending=[False, False, True])
    return stats, align


def build_career(players: pl.DataFrame) -> pl.DataFrame:
    cs = pl.read_parquet(PQ / "player_career_successes.parquet")
    dp = (
        pl.read_csv(EXT / "draft_picks.csv", infer_schema_length=0)
        .filter(pl.col("season").cast(pl.Int64) >= 2023)
        .select("gsis_id", pl.col("team").alias("draft_team"), "w_av", "car_av", "dr_av", pl.col("games").alias("pfr_games"),
                pl.col("seasons_started").alias("pfr_seasons_started"), pl.col("probowls").alias("pfr_probowls"), pl.col("allpro").alias("pfr_allpro"))
        .with_columns(*[pl.col(c).cast(pl.Float64, strict=False) for c in ["w_av", "car_av", "dr_av", "pfr_games", "pfr_seasons_started", "pfr_probowls", "pfr_allpro"]])
        .filter(pl.col("gsis_id").is_not_null()).unique(subset=["gsis_id"])
    )
    ct = (
        pl.read_parquet(EXT / "historical_contracts.parquet")
        .filter(pl.col("gsis_id").is_not_null())
        .sort("year_signed")
        .group_by("gsis_id").first()
        .select("gsis_id", pl.col("value").alias("contract_value_m"), pl.col("apy").alias("contract_apy_m"),
                pl.col("guaranteed").alias("contract_guaranteed_m"), pl.col("years").alias("contract_years"),
                pl.col("apy_cap_pct").alias("contract_apy_cap_pct"))
    )
    return (
        cs.join(players.select("nfl_id", "gsis_id"), on="nfl_id", how="left")
        .join(dp, on="gsis_id", how="left").join(ct, on="gsis_id", how="left").drop("gsis_id")
        .with_columns((pl.col("career_offensive_snaps") + pl.col("career_defensive_snaps")).alias("career_scrimmage_snaps"))
    )


# ----------------------------------------------------------------------------- game tracking sample
def build_game_sample(players: pl.DataFrame, per_player: int = 12) -> tuple[pl.DataFrame, pl.DataFrame]:
    pp = pl.read_parquet(PQ / "player_play.parquet")
    games = regular_season_games(pl.read_parquet(PQ / "games.parquet"))
    pp = pp.join(games, on="game_id", how="inner").filter(pl.col("play_nullified_by_penalty") != "Y")
    pp = pp.join(players.select("nfl_id", "position_group"), on="nfl_id", how="left")
    key = (
        pl.when(pl.col("position_group").is_in(["WR", "TE"])).then(pl.col("separation_at_pass_forward"))
        .when(pl.col("position_group") == "OL").then(pl.col("peak_pressure_probability_allowed"))
        .when(pl.col("position_group") == "DL").then(pl.col("player_get_off"))
        .otherwise(pl.col("cushion"))
    )
    cand = pp.filter(key.is_not_null()).with_columns(
        key.alias("key_metric"), pl.col("expected_points_added").abs().alias("abs_epa"),
        (pl.int_range(pl.len()).shuffle(seed=7)).alias("rnd"),
    )
    top = cand.sort("nfl_id", "abs_epa", descending=[False, True]).group_by("nfl_id", maintain_order=True).head(per_player // 2)
    rnd = cand.join(top.select("nfl_id", "game_id", "play_id"), on=["nfl_id", "game_id", "play_id"], how="anti") \
        .sort("nfl_id", "rnd").group_by("nfl_id", maintain_order=True).head(per_player // 2)
    sample = pl.concat([top, rnd]).select(
        "nfl_id", "game_id", "play_id", "season", "week", "team_abbr", "possession_team", "lined_up_position", "play_description",
        "quarter", "down", "yards_to_go", "yardline_side", "yardline_number", "game_clock", "play_direction", "offense_formation",
        "pass_result", "pass_length", "route_ran", "target", "yards_gained", "expected_points_added", "team_coverage_type",
        "separation_at_pass_forward", "cushion", "yards_after_catch", "pressure_allowed", "peak_pressure_probability_allowed",
        "pass_rushers_encountered", "player_get_off", "time_to_pressure", "sack", "tackle", "coverage_assignment", "key_metric",
    ).sort("nfl_id", "season", "week")
    log(f"game sample plays {sample.height:,}")
    keys = sample.select("game_id", "play_id", "nfl_id").unique()
    parts = []
    for season in (2023, 2024, 2025):
        k = keys.join(games.filter(pl.col("season") == season).select("game_id"), on="game_id", how="semi")
        if k.height == 0:
            continue
        lf = pl.scan_parquet(PQ / f"game_tracking_{season}.parquet").join(k.lazy(), on=["game_id", "play_id", "nfl_id"], how="semi")
        parts.append(lf.collect())
    fr = pl.concat(parts).sort("game_id", "play_id", "nfl_id", "time")
    fr = fr.with_columns(
        (pl.int_range(pl.len()).over("game_id", "play_id", "nfl_id")).cast(pl.Int32).alias("frame"),
    ).with_columns((pl.col("frame") / 10.0).alias("t"))
    # frame index of ball_snap for alignment in the UI
    snap = fr.filter(pl.col("event") == "ball_snap").group_by("game_id", "play_id", "nfl_id").agg(pl.col("frame").min().alias("snap_frame"))
    sample = sample.join(snap, on=["game_id", "play_id", "nfl_id"], how="left")
    fr = fr.select("game_id", "play_id", "nfl_id", "frame", "t", "x", "y", "s", "a", "o", "dir", "event")
    log(f"game sample frames {fr.height:,}")
    return sample, fr


def nan_to_null(df: pl.DataFrame) -> pl.DataFrame:
    """Postgres accepts float NaN and PostgREST then emits it as a string; store nulls instead."""
    return df.with_columns([pl.col(c).fill_nan(None) for c, t in df.schema.items() if t in (pl.Float64, pl.Float32)])


def write(df: pl.DataFrame, name: str) -> None:
    nan_to_null(df).write_parquet(OUT / f"{name}.parquet")


if __name__ == "__main__":
    t0 = time.time()
    players = build_players()
    write(players, "players"); log(f"players {players.height}")
    cr, pcts = build_combine(players)
    write(cr, "combine_results"); write(pcts, "combine_percentiles")
    log(f"combine percentiles {pcts.height}")
    frames, events, dropped = build_combine_tracking(players)
    write(frames, "combine_frames"); write(events, "combine_events")
    write(dropped, "combine_dropped_events")
    log(f"combine events {events.height}, frames {frames.height:,}")
    drills = build_drills(events, frames)
    write(drills, "drills"); log(f"drills {drills.height}")
    stats, align = build_season_stats(players)
    write(stats, "player_season_stats"); write(align, "player_season_alignment")
    log(f"season stats {stats.height}, alignment rows {align.height}")
    career = build_career(players)
    write(career, "player_career"); log(f"career {career.height}")
    sample, fr = build_game_sample(players)
    write(sample, "game_plays_sample"); write(fr, "game_frames_sample")
    log(f"done in {time.time()-t0:.0f}s")
