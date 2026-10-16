"""Convert the raw competition CSVs to parquet once. Game tracking is partitioned by season.

Usage: uv run python etl/01_to_parquet.py
"""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "parquet"
OUT.mkdir(parents=True, exist_ok=True)

SMALL = [
    "players",
    "combine_results",
    "games",
    "player_career_successes",
    "combine_tracking",
    "player_play",
]


def convert_small(name: str) -> None:
    src, dst = RAW / f"{name}.csv", OUT / f"{name}.parquet"
    t = time.time()
    df = pl.read_csv(src, infer_schema_length=100_000, null_values=["NA", "", "null"])
    if "time" in df.columns:
        df = df.with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f"))
    df.write_parquet(dst, compression="zstd")
    print(f"{name:28s} {df.height:>12,} rows  {df.width:>3} cols  {time.time()-t:5.1f}s")


def convert_tracking(season: int) -> None:
    src = RAW / f"game_tracking_{season}.csv"
    dst = OUT / f"game_tracking_{season}.parquet"
    t = time.time()
    lf = pl.scan_csv(src, infer_schema_length=100_000, null_values=["NA", "", "null"]).with_columns(
        pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")
    )
    lf.sink_parquet(dst, compression="zstd", row_group_size=500_000)
    n = pl.scan_parquet(dst).select(pl.len()).collect().item()
    print(f"game_tracking_{season:<14d} {n:>12,} rows               {time.time()-t:5.1f}s")


if __name__ == "__main__":
    for name in SMALL:
        convert_small(name)
    for season in (2023, 2024, 2025):
        convert_tracking(season)
    print("done")
