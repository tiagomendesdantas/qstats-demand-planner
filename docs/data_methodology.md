# Data methodology

What comes from the UCI file, what is transformed, what is simulated, and what the planner never
sees. The cleaning and calendar counts are produced by `scripts/prepare_transactions.py` and stored
in `data/processed/manifest.json`; population counts come from `scripts/build_demo_population.py`
(`data/processed/population.parquet`). The two large order-and-cancel pairs and the rejected
broader wholesale rule were one-off queries on the same file and are marked as such.

## 1. Source

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D. CC BY 4.0. One xlsx inside a zip (SHA-256
`572e3627…7bfb`, verified on download), two sheets, 1,067,371 order lines, 1 Dec 2009 to 9 Dec
2011. A UK-based online retailer of giftware with many wholesale customers: demand is lumpier than
a typical direct-to-consumer seller's, and this project does not describe it as DTC.

The file is kept unchanged in `data/raw/`. `scripts/download_data.py` downloads it with retries,
or verifies a copy placed there by hand when the download is not possible.

## 2. From source columns to the canonical contract

`adapters/uci.py` is the only module that knows the UCI column names. It maps each line to a
canonical sales line (`order_id, order_ts, sku, description, quantity, unit_price, customer_id,
country, line_type`) and classifies it. It does not drop anything:

| UCI convention | Line type |
|---|---|
| Invoice starting with `C` | RETURN (cancellation) |
| Invoice starting with `A` | ADJUSTMENT (bad debt) |
| StockCode not matching `^\d{5}[A-Z]{0,3}$` (POST, DOT, M, C2, D, BANK CHARGES, AMAZONFEE, gift vouchers, TEST…) | NON_MERCHANDISE |
| Negative quantity on a normal invoice | ADJUSTMENT (stock correction) |
| Everything else | SALE |

The two sheets overlap on 1–9 Dec 2010 (22,523 lines); the first sheet's invoices are kept.

## 3. Cleaning (source-agnostic, `demand/cleaning.py`)

| Rule | Lines |
|---|---|
| Lines after removing the sheet overlap | 1,044,848 |
| Exact duplicate lines removed | 11,812 |
| Adjustments removed | 3,368 |
| Non-merchandise lines removed | 5,974 |
| Zero or negative price removed (write-offs, corrections) | 2,566 |
| Missing descriptions after the rules above | 0 (filled with the SKU's modal description otherwise) |
| Lines without a customer ID (kept) | 226,965 |
| Return lines (6,179 cancel an earlier sale; 11,735 stay returns on their own date) | 17,914 |
| Sales cancelled by an exactly matching return | 6,179 |
| Wholesale lots flagged | 159 |

**Returns are never subtracted from demand blindly.** A return that matches an earlier sale by the
same customer, SKU and quantity (most recent earlier sale, one-to-one) marks that sale as
cancelled: the order was never fulfilled demand, so it is removed on the day it was placed, not
netted on the day of the return. This covers 400,719 of the 467,739 returned units, including two
same-day order-and-cancel pairs of 74,215 and 80,995 units (one-off query). Returns without a match stay returns on
their own day and do not reduce demand. Lines without a customer cannot be matched.

**Wholesale lots.** A line more than 10× the SKU's own 95th-percentile line and at least 1,000
units is flagged (159 lines, 448,020 units, 4.2% of non-cancelled sale units). It stays in the
record and is left out of the demand the simulated business faces: a handful of such lines would
decide every fill-rate metric. A broader rule (4× and 500 units) was rejected because it removed
8.4% of units, much of it ordinary wholesale lumpiness (one-off query, not rerun).

Fields kept per SKU-day: `gross_units_sold` (all sale lines), `returns`, `net_units`
(gross − returns), and `units` (fulfilled demand: sales neither cancelled nor wholesale lots),
`revenue`, `average_price`.

## 4. Calendar and lifecycle

The retailer traded on 604 of 739 calendar days: no Saturdays (one exception, 9 Dec 2023 on the
shifted calendar), two eleven-day year-end shutdowns (28 Dec – 7 Jan and 27 Dec – 6 Jan, shifted),
and four-day Easter closures. A closed day is not a zero-demand day, so demand is modelled per
trading day and weekly forecasts are multiplied by the number of trading days in the week. Future
trading days follow the observed pattern (Saturdays closed, 27 Dec – 6 Jan shut).

Every date is moved forward by exactly 731 weeks (weekdays and seasons preserved), so the history
reads as 5 Dec 2023 – 12 Dec 2025. `calendar.shift_weeks: 0` turns this off.

A SKU's active life runs from its first sale to its last sale, or to the end of the data unless
it stopped selling more than eight weeks before the end (1,649 of 4,699 SKUs did). Missing SKU-days
are filled with zero only inside that life and only on trading days.

## 5. Demo and dev populations

Chosen systematically (`demand/population.py`). Established SKUs are selected using **only the
first 52 weeks of data**, so they are not chosen by how they behaved later; new products are drawn
by launch date only (below). Eligible: at least 150 units in those weeks,
launched by week 44, still selling in the last four. Each eligible SKU gets one profile, by
precedence: SHORT_HISTORY (launched in the second half of the year), SEASONAL_KEYWORD (Christmas-
type description), DECLINING / GROWING (relative change across January–August ≥ 1.0, measured
outside the September–December season so a seasonal ramp is not read as trend), then the
Syntetos–Boylan class of weekly demand (SMOOTH, ERRATIC, INTERMITTENT, LUMPY). Quotas per profile
over-represent rare profiles; within a profile, SKUs are drawn evenly from volume terciles.

| Profile | Demo | Dev |
|---|---|---|
| SMOOTH | 50 | 30 |
| ERRATIC | 31 | 18 |
| LUMPY | 27 | 16 |
| SEASONAL_KEYWORD | 20 | 12 |
| NEW_PRODUCT | 20 | 12 |
| DECLINING | 17 | 10 |
| GROWING | 14 | 9 |
| SHORT_HISTORY | 14 | 9 |
| INTERMITTENT | 7 | 4 |

New products are drawn at random among SKUs launched in weeks 53–80, by launch date only. 44 demo
SKUs stopped selling before the end of the data (no sale in its last eight weeks): 5 made their last
sale before the fork and 14 before the scoring window began. They stay in: carrying stock of a
product that dies is a real cost. The dev population (120 SKUs, disjoint) is where every tuning decision was made.
`population.demo_sku_count` runs the pipeline on more SKUs.

## 6. The simulated business

Placed around the real demand patterns, seeded (`random_seed: 42`) and described in
`config/demo.yaml`. None of it describes the original retailer.

- **Locations.** EAST_DC and WEST_DC (60/40 by customer region) and AMAZON_FBA. Each customer is
  assigned once, by a stable hash, to a region and, for FBA-enabled SKUs, possibly to Amazon, so
  each location keeps the real lumpiness of its customers' orders. Only small-basket customers
  (median line ≤ 12 units) are Amazon shoppers. Demand shares: EAST 56%, WEST 34%, Amazon 10%.
- **Fulfilment.** A DC that cannot serve its region ships from the other DC when it can ($1.10
  extra per unit). Amazon demand is served from Amazon stock only. Unserved demand is lost.
- **Products.** USD selling price from the source's median price × 1.27 × a markup of 2.0–2.6;
  landed cost 26–44% of price; fulfilment (DC 10% + $0.45, FBA 17% + $0.95) and advertising
  (4–14%) costs; contribution margin from those. Case pack = the most common order quantity among
  standard pack sizes; MOQ = 3–8 weeks of first-year demand in whole cases; cube 0.015–0.12 m³ per
  case; service target by revenue class (A 97%, B 95%, C 90%); 72 of 200 SKUs on Amazon.
- **Suppliers.** Ten, in China, Vietnam, India, Bangladesh, Turkey, Mexico, Portugal and
  Indonesia. Lead time = supplier median × exp(supplier-week shock + per-order noise), plus a
  delay tail (4–12% of orders, mean 8–20 extra days) and occasional supplier-wide disruptions
  (+15–30 days for three weeks). The quoted lead time is the median of that distribution
  (40–68 days). Fourteen receipts per supplier predate the simulation.
- **Inventory feed gaps.** Blocks of 5–15 days where historical snapshots are missing for a SKU
  and location (UNKNOWN_AVAILABILITY).
- **Amazon FBA states.** Transfers take 2 days to pick and 5–12 days in transit; 30% of each
  arrival spends four days in an Amazon fulfilment-centre transfer; 3–12% of Amazon stock is
  RESERVED on any day and a small share of it is written off.
- **Events.** 12% of SKUs get one to three simulated promotions (uplift drawn from a lognormal,
  unknown to every planner; calendar announced six weeks ahead) and 4% of the SKUs that really did
  stop selling get a simulated liquidation in their last four weeks. These 32 SKUs are excluded from
  the headline comparison.

## 7. What is hidden

The engine keeps `baseline` (true demand per day, SKU and location), lost units, the pre-uplift
series and the true lead-time parameters. Planners receive a `PlannerView` that holds none of them:
observed sales, inventory snapshots with their gaps, POs with the status a buyer would have seen,
received lead times, announced events and master data. A test fails if a planner module (or one
of the two replayed planners) imports the evaluation layer; two others scramble demand after a date,
one per planner, and check that no decision before it changes.
