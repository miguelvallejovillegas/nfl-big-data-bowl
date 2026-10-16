#!/usr/bin/env bash
# Cloud-environment setup for Claude Code sessions (also fine locally).
# 1) Python deps  2) data snapshot (processed tables + small raw CSVs)  3) optional full Kaggle download.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v unzstd >/dev/null 2>&1 || { echo "== installing zstd"; (sudo apt-get update -qq && sudo apt-get install -y -qq zstd) || apt-get install -y -qq zstd; }

echo "== uv sync"
uv sync --extra dev -q

BASE="${DATA_SNAPSHOT_BASE:-https://houjxgqtzxfgccxspsje.supabase.co/storage/v1/object/public/data/snapshot}"
mkdir -p data/raw data/processed data/external data/parquet

if [ ! -f data/processed/combine_events.parquet ]; then
  echo "== data snapshot from $BASE"
  curl -fsSL "$BASE/processed.tar.zst" -o /tmp/processed.tar.zst
  tar --use-compress-program=unzstd -xf /tmp/processed.tar.zst -C data
  for f in players combine_results games player_career_successes combine_tracking player_play; do
    curl -fsSL "$BASE/$f.csv.zst" | unzstd -q -o "data/raw/$f.csv" -f
  done
  echo "   processed tables + raw CSVs (no game tracking) restored"
fi

if [ -n "${KAGGLE_API_TOKEN:-}${KAGGLE_KEY:-}" ] && [ ! -f data/raw/game_tracking_2025.csv ]; then
  echo "== full competition data from Kaggle (2.3 GB)"
  uv tool install -q kaggle || true
  kaggle competitions download -c nfl-big-data-bowl-2027 -p /tmp/kaggle
  unzip -oq /tmp/kaggle/nfl-big-data-bowl-2027.zip -d data/raw
  rm -rf /tmp/kaggle
fi

if [ ! -f data/parquet/combine_tracking.parquet ]; then
  echo "== parquet (small files only unless game tracking is present)"
  uv run python - <<'PY'
import polars as pl, pathlib
raw, out = pathlib.Path("data/raw"), pathlib.Path("data/parquet"); out.mkdir(exist_ok=True)
for n in ["players","combine_results","games","player_career_successes","combine_tracking","player_play"]:
    df = pl.read_csv(raw/f"{n}.csv", infer_schema_length=100_000, null_values=["NA","","null"])
    if "time" in df.columns: df = df.with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f"))
    df.write_parquet(out/f"{n}.parquet", compression="zstd")
for s in (2023, 2024, 2025):
    f = raw/f"game_tracking_{s}.csv"
    if f.exists():
        pl.scan_csv(f, infer_schema_length=100_000).with_columns(pl.col("time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")).sink_parquet(out/f"game_tracking_{s}.parquet", compression="zstd")
print("parquet ready")
PY
fi

echo "== smoke test"
uv run python -c "import polars as pl; print(pl.read_parquet('data/processed/combine_events.parquet').height, 'combine events')"
uv run pytest -q 2>&1 | tail -1 || true
echo "setup done"
