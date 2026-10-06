"""Raw UCI file -> canonical cleaned lines, daily SKU demand and SKU lifecycles.

Outputs (data/processed/):
    sales_lines.parquet    canonical lines as the adapter produced them (pinned, hashed)
    clean_lines.parquet    after every cleaning rule, with cancelled / bulk flags
    demand_lines.parquet   fulfilled demand lines, the input to the simulation
    daily_demand.parquet   date x SKU, zero-filled inside each SKU's active life
    sku_lifecycle.parquet  first_seen_date, last_seen_date, disappeared
    trading_days.parquet   days the retailer traded
    manifest.json          counts for every rule, date ranges, hashes
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from qstats_planner.adapters.uci import UCIAdapter  # noqa: E402
from qstats_planner.demand.cleaning import clean_sales_lines, demand_lines  # noqa: E402
from qstats_planner.demand.daily import daily_sku_demand, sku_lifecycle  # noqa: E402
from qstats_planner.utils.config import load_config, resolve  # noqa: E402


def frame_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()


def main() -> int:
    cfg = load_config()
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    out_dir = resolve(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    src = cfg["source"]
    path = raw_dir / src["zip_name"]
    if not path.exists():
        path = raw_dir / src["xlsx_name"]
    if not path.exists():
        print("raw file missing: run python scripts/download_data.py first")
        return 2

    pinned = out_dir / "sales_lines.parquet"
    t0 = time.time()
    if pinned.exists():
        lines = pd.read_parquet(pinned)
        adapter_report = json.loads((out_dir / "adapter_report.json").read_text())
        print(f"reusing pinned canonical lines ({len(lines):,} rows)")
    else:
        print(f"reading {path.name} (the xlsx takes a minute or two)...")
        adapter = UCIAdapter(path, cfg["cleaning"]["merchandise_pattern"])
        lines = adapter.sales_lines()
        adapter_report = adapter.report
        lines.to_parquet(pinned, index=False)
        (out_dir / "adapter_report.json").write_text(json.dumps(adapter_report, indent=2))
        print(f"  read in {time.time() - t0:.0f}s")

    original_range = [str(lines["order_ts"].min()), str(lines["order_ts"].max())]
    shift = int(cfg["calendar"]["shift_weeks"])
    lines = lines.assign(order_ts=lines["order_ts"] + pd.Timedelta(weeks=shift))

    clean, report = clean_sales_lines(lines, cfg)
    demand = demand_lines(clean)
    trading = pd.DatetimeIndex(sorted(clean["date"].unique()))
    data_end = trading.max()
    life = sku_lifecycle(demand, data_end, cfg["cleaning"]["disappeared_after_weeks"])
    daily = daily_sku_demand(clean, trading, life)

    clean.to_parquet(out_dir / "clean_lines.parquet", index=False)
    demand.to_parquet(out_dir / "demand_lines.parquet", index=False)
    daily.to_parquet(out_dir / "daily_demand.parquet", index=False)
    life.to_parquet(out_dir / "sku_lifecycle.parquet", index=False)
    pd.DataFrame({"date": trading}).to_parquet(out_dir / "trading_days.parquet", index=False)

    manifest = {
        "source": {"url": src["url"], "sha256": src["sha256"], "doi": "10.24432/C5CG6D",
                   "licence": "CC BY 4.0"},
        "adapter": adapter_report,
        "cleaning": report,
        "original_date_range": original_range,
        "calendar_shift_weeks": shift,
        "shifted_date_range": [str(trading.min().date()), str(data_end.date())],
        "trading_days": len(trading),
        "calendar_days": int((data_end - trading.min()).days + 1),
        "skus_with_demand": int(life["sku"].nunique()),
        "skus_disappeared": int(life["disappeared"].sum()),
        "daily_rows": len(daily),
        "sales_lines_hash": frame_hash(pd.read_parquet(pinned)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: manifest[k] for k in ("cleaning", "shifted_date_range", "trading_days",
                                               "skus_with_demand", "skus_disappeared")}, indent=2))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
