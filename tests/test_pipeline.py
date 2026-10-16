"""Sanity checks on the cleaning and feature pipeline (run: uv run pytest)."""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from bdb27.clean import DRILL_ALIASES, clean_combine_tracking
from bdb27.drills import DRILL_BY_NAME
from bdb27.kinematics import detect_breaks, event_features

ROOT = Path(__file__).resolve().parents[1]
PQ = ROOT / "data" / "parquet"
PROC = ROOT / "data" / "processed"
needs_data = pytest.mark.skipif(not (PQ / "combine_tracking.parquet").exists(), reason="raw data not converted")


def test_forty_features_on_synthetic_sprint():
    t = np.arange(0, 5.0, 0.1)
    v = 10 * (1 - np.exp(-t / 1.2))
    x = 10 + np.cumsum(v) * 0.1
    y = np.zeros_like(x)
    f = event_features(x, y, v, "FORTY_YARD_DASH")
    assert f["n_breaks"] == 0
    assert 9 < f["top_speed"] <= 10
    assert f["split_10yd_s"] is not None and f["split_10yd_s"] < f["split_20yd_s"]
    assert f["straightness"] > 0.99


def test_break_detection_on_right_angle():
    n = 30
    x = np.concatenate([np.linspace(0, 15, n), np.full(n, 15)])
    y = np.concatenate([np.zeros(n), np.linspace(0, 15, n)])
    s = np.full(2 * n, 5.0)
    heading = np.concatenate([np.full(n, 90.0), np.full(n, 0.0)])
    brks = detect_breaks(heading, s)
    assert len(brks) == 1 and brks[0][1] > 80
    f = event_features(x, y, s, "SKILL_DRILLS_WR")
    assert f["n_breaks"] == 1


@needs_data
def test_clean_tracking_covers_aliases_and_catalog():
    ct = pl.read_parquet(PQ / "combine_tracking.parquet")
    frames, dropped = clean_combine_tracking(ct)
    names = set(frames["drill_name"].unique())
    assert not names & set(DRILL_ALIASES), "raw alias names should be gone"
    assert names <= set(DRILL_BY_NAME), f"missing from catalog: {names - set(DRILL_BY_NAME)}"
    assert dropped.height == 4
    gaps = frames.filter(pl.col("entity_type") == "PLAYER").sort("event_id", "frame").with_columns(pl.col("frame").diff().over("event_id").alias("d"))
    assert gaps.filter(pl.col("d").is_not_null() & (pl.col("d") != 1)).height == 0


@needs_data
def test_tracked_forty_matches_official_time():
    ev = pl.read_parquet(PROC / "combine_events.parquet").filter(pl.col("drill_type") == "FORTY_YARD_DASH")
    cr = pl.read_parquet(PQ / "combine_results.parquet").select("nfl_id", "forty")
    j = ev.group_by("nfl_id").agg(pl.col("split_40yd_s").min()).join(cr, on="nfl_id").drop_nulls()
    r = np.corrcoef(j["split_40yd_s"].to_numpy(), j["forty"].to_numpy())[0, 1]
    assert r > 0.9, r


@needs_data
def test_regular_season_filter():
    g = pl.read_parquet(PQ / "games.parquet").filter(pl.col("season_type") == "REG")
    assert g.group_by("season").len()["len"].to_list() == [272, 272, 272]
    s = pl.read_parquet(PROC / "player_season_stats.parquet")
    assert s.select("nfl_id", "season").is_unique().all()
