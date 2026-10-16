"""Cleaning rules for the BDB 2027 competition data.

Everything here is a documented data quirk found during profiling; see README.
"""
from __future__ import annotations

import polars as pl

# Drill names drift between combine years. Map every raw name to one canonical name.
DRILL_ALIASES: dict[str, str] = {
    "PASS_PRO-MIRROR_DRILL": "PASS_PRO_MIRROR_DRILL",
    "LINE": "LINE_DRILL",
    "SPEED_OUT_LEFT": "SPEED_OUT_ROUTE_LEFT",
    "OVER_THE_SHOULDER_ADJUST": "OVER_SHOULDER_ADJUST",
}

# The 37 SKILL_DRILLS_LB events belong to 4 DL prospects; keep them but label the group.
DRILL_TYPE_GROUP: dict[str, str] = {
    "FORTY_YARD_DASH": "SPEED",
    "SHORT_SHUTTLE": "AGILITY",
    "THREE_CONE_DRILL": "AGILITY",
    "SKILL_DRILLS_WR": "WR",
    "SKILL_DRILLS_TE": "TE",
    "SKILL_DRILLS_DB": "DB",
    "SKILL_DRILLS_DL": "DL",
    "SKILL_DRILLS_LB": "LB",
    "SKILL_DRILLS_OL": "OL",
}

# Combine position group for each NFL roster position in players.csv.
NFL_POS_TO_GROUP: dict[str, str] = {
    "WR": "WR",
    "TE": "TE",
    "FB": "TE",
    "T": "OL", "G": "OL", "C": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "OLB": "DL",
    "CB": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "ILB": "DB", "MLB": "DB", "RB": "DB",
}

# Event lengths outside this window are tracking glitches (an 11-frame shuttle, a 202-frame shuttle).
MIN_FRAMES = 15
MAX_FRAMES_BY_TYPE = {"SHORT_SHUTTLE": 120, "THREE_CONE_DRILL": 150}


def normalize_drill_names(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("drill_name").replace(DRILL_ALIASES).alias("drill_name"),
        pl.col("drill_type").replace(DRILL_TYPE_GROUP).alias("drill_group"),
    )


def clean_combine_tracking(ct: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (frames, dropped_events). Frames get a per-event frame index and seconds."""
    ct = normalize_drill_names(ct)
    # frame index per event, shared by PLAYER and BALL rows (same timestamps)
    ct = ct.sort(["event_id", "entity_type", "time"]).with_columns(
        ((pl.col("time") - pl.col("time").min().over("event_id")).dt.total_milliseconds() / 1000.0)
        .round(1)
        .alias("t"),
    ).with_columns((pl.col("t") * 10).round(0).cast(pl.Int32).alias("frame"))

    n = ct.filter(pl.col("entity_type") == "PLAYER").group_by("event_id", "drill_type").len()
    bad = n.filter(
        (pl.col("len") < MIN_FRAMES)
        | ((pl.col("drill_type") == "SHORT_SHUTTLE") & (pl.col("len") > MAX_FRAMES_BY_TYPE["SHORT_SHUTTLE"]))
        | ((pl.col("drill_type") == "THREE_CONE_DRILL") & (pl.col("len") > MAX_FRAMES_BY_TYPE["THREE_CONE_DRILL"]))
    ).select("event_id", "len")
    frames = ct.join(bad.select("event_id"), on="event_id", how="anti")
    # attempt: recompute as rank of event start time within (player, drill_name)
    starts = (
        frames.filter(pl.col("entity_type") == "PLAYER")
        .group_by("event_id", "nfl_id", "drill_name")
        .agg(pl.col("time").min().alias("start"))
        .with_columns(pl.col("start").rank("ordinal").over("nfl_id", "drill_name").cast(pl.Int32).alias("attempt_no"))
        .select("event_id", "attempt_no")
    )
    frames = frames.join(starts, on="event_id", how="left")
    return frames, bad


def regular_season_games(games: pl.DataFrame) -> pl.DataFrame:
    return games.filter(pl.col("season_type") == "REG").select("game_id", "season", "week")
