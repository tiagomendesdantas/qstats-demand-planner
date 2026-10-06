"""Select the demo and dev SKU populations (first 52 weeks of data only).

Writes data/processed/population.parquet and data/processed/sku_metrics.parquet (full-period
metrics, for display).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from qstats_planner.demand.features import sku_metrics  # noqa: E402
from qstats_planner.demand.population import select_populations  # noqa: E402
from qstats_planner.utils.config import load_config, resolve  # noqa: E402


def main() -> int:
    cfg = load_config()
    d = resolve(cfg["paths"]["processed_dir"])
    daily = pd.read_parquet(d / "daily_demand.parquet")
    life = pd.read_parquet(d / "sku_lifecycle.parquet")
    start = pd.read_parquet(d / "trading_days.parquet")["date"].min()
    pop = select_populations(daily, life, start, cfg)
    pop.to_parquet(d / "population.parquet", index=False)

    full = sku_metrics(daily[daily["sku"].isin(pop["sku"])], cfg)
    full.to_parquet(d / "sku_metrics.parquet", index=False)
    print(pop.groupby(["population", "profile"]).size().unstack(0).fillna(0).astype(int).to_string())
    print(
        f"demo {int((pop.population == 'demo').sum())}  dev {int((pop.population == 'dev').sum())}  "
        f"overlap {len(set(pop.sku[pop.population == 'demo']) & set(pop.sku[pop.population == 'dev']))}"
    )
    print("disappeared later (demo):", int(pop.loc[pop.population == "demo", "disappeared"].sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
