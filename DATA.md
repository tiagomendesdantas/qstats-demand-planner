# Data

Checked on 2026-10-06.

- **Source:** Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5CG6D
- **File:** https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip, SHA-256
  `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb` (verified by
  `scripts/download_data.py`). Kept unchanged in `data/raw/`, which is not committed.
- **Licence:** CC BY 4.0. Attribution above; changes listed below.
- **What it is:** order lines of a UK-based online giftware retailer with many wholesale
  customers, 1 Dec 2009 – 9 Dec 2011, 1,067,371 lines in two overlapping sheets.

## Changes made

1. The sheets' overlap (1–9 Dec 2010, 22,523 lines) kept once.
2. 11,812 exact duplicate lines, 3,368 adjustments, 5,974 non-product lines (postage, fees,
   vouchers, test codes) and 2,566 zero- or negative-price lines removed.
3. 6,179 sales cancelled by an exactly matching later return removed from demand on their own
   date; other returns kept as returns.
4. 159 exceptional wholesale lots (more than 10× the SKU's 95th-percentile line and at least 1,000
   units) left out of demand.
5. Dates moved forward by 731 weeks (history reads 5 Dec 2023 – 12 Dec 2025).
6. Customers assigned to a simulated region and channel by a stable hash.

Everything else in the demo (warehouses, Amazon FBA, inventory, suppliers, lead times, purchase
orders, costs, margins, containers, promotions) is simulated and does not describe the retailer.
Derived files are rebuilt by `make demo` and are not committed.
