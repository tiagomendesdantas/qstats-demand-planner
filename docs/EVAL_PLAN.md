# Evaluation plan: Legacy planner vs QStats planner

Committed on 2026-10-06, after the development runs on the dev SKUs and **before** the comparison
was run on the demo SKUs. The plan below is not edited after that run; corrections go to the
change log at the end, with the numbers before and after.

## Question

Run the same simulated business twice from the same moment: once with the Legacy planning process,
once with QStats. Same products, same real demand pattern, same supplier luck. Does QStats deliver
the same service with less inventory, or more service for the same inventory?

## Set-up

| Item | Value |
|---|---|
| Business | Fictional U.S. importer (two DCs + Amazon FBA); demand patterns from UCI Online Retail II; everything else simulated (`config/demo.yaml`, business seed 42) |
| Populations | `demo`: 200 SKUs (reported). `dev`: 120 different SKUs (every tuning decision). Both chosen on the first 52 weeks only |
| Shared history | Weeks 1–12 warm-up (ample stock, not scored). Weeks 13–52: the Legacy planner runs the business and creates the censored history |
| Fork | Start of week 53. World L keeps Legacy; World Q hands the same state to QStats. Every variant forks from the same week-52 state |
| Scoring window | Fork (2024-12-02) + longest quoted lead time (68 days) to the last complete week of data: 2025-02-08 to 2025-12-07, 303 days (shifted calendar) |
| Replicate worlds | Supply seeds 42 (reference), 7, 2026: supplier lead times, delays, disruptions, FBA behaviour and feed gaps change; products and demand do not |
| Headline SKUs | Demo SKUs without a simulated promotion or liquidation (168 of 200). The 32 event SKUs are reported separately, because the size of a simulated uplift is a dial the designer controls |

### Environment parameters and why

| Parameter | Value | Why |
|---|---|---|
| Quoted lead time | Median of the supplier's true distribution | The quote is honest on average; QStats's lead-time model can only gain from the spread, not from a planted bias |
| Lead-time spread | Log-sd 0.08–0.22 per PO, plus a shared supplier-week shock 0.04–0.12 | Orders from one supplier in one week share their luck |
| Delays | 4–12% of POs, mean 8–20 extra days | Ocean freight tails |
| Disruptions | 0.4 per supplier on average, 3-week windows, +15–30 days | Supplier-wide events |
| Cross-DC fulfilment | Allowed, $1.10 extra per unit | What an order-management system does when one DC is out |
| Amazon channel | Small-basket customers of FBA-enabled SKUs (median line ≤ 12 units), 45% of them | Wholesale-sized buyers do not shop on Amazon |
| Wholesale lots | Lines > 10× the SKU's 95th-percentile line and ≥ 1,000 units are left out of demand (4.2% of units) | A handful of lines would otherwise decide every fill-rate metric |
| Promotions | 12% of SKUs, 1–3 events, uplift lognormal (mean log 0.55, sd 0.30), unknown to both planners; calendar announced 6 weeks ahead to both | Exercises event handling; excluded from the headline |

## Policies

**Legacy** (`simulation/policies/legacy.py`): SES on weekly sales (alpha 0.2, tuned on dev), closed
weeks skipped, sales taken as demand; quoted lead time as certain; safety stock = 30 days of
forecast demand; weekly review; order up to lead-time + review + safety demand; same MOQ and
case-pack rounding as QStats; FBA by days of cover (pick + transit + review + 14 days trigger,
+30 days target).

**QStats** (`replenishment/policy.py`): stockout reconstruction (local_profile, chosen on dev);
champion forecast per segment by rolling-origin cumulative error (25 candidates, seasonal prior on
or off); Kaplan–Meier supplier lead times with open POs censored; demand over lead time + review as
an exact mixture of lead-time and forecast-error distributions; order-up-to = the product's
service-level quantile; RESERVED FBA units counted at 0.5.

**Variants run from the same fork**

| Family | Variants |
|---|---|
| Legacy frontier | 15, 30 (reference), 45, 60, 75, 90, 120 days of safety stock |
| Legacy + seasonal prior | 30 days, SES on seasonally adjusted sales, re-seasonalised over the lead time |
| QStats frontier | Product targets (reference: A 97%, B 95%, C 90%), and uniform 70, 80, 90, 95, 98% |
| Ablation (2×2×2) | Stockout reconstruction on/off × seasonal prior on/off × probabilistic safety stock and lead times on/off (off = quoted lead time, 30 days of cover). All on = the QStats reference |
| Sensitivity worlds | `null`: lead time always equals the quote, no events. `optimistic_quotes`: quotes at the 25th percentile of the true distribution. Reference seed only |

## Metrics

Per SKU over the scoring window, summed before ratios, so every world is scored on identical
SKU-days.

- **Fill rate** = units sold / true demand (hidden baseline).
- **In-stock rate** = 1 − stocked-out channel-days / active channel-days (direct channel = both
  DCs pooled; Amazon separately).
- **Average inventory** = mean daily on-hand units plus units in transit to Amazon, at landed cost.
- **Working capital** = average inventory + supplier shipments in transit (owned from shipment).
- Lost units and lost contribution; inventory turns (annualised COGS / average inventory);
  excess at the end (stock beyond 26 weeks of the last 13 weeks' true demand); purchase value of
  POs placed from the fork; ending position (on hand + on order) so no policy looks lean by
  starving the end; cross-DC shipments.
- **Forecast WAPE and bias**: one-week-ahead network forecast vs true weekly demand.

## Primary metric and how it is read

**Inventory QStats needs to deliver the Legacy process's fill rate**, read by linear interpolation
along the QStats frontier at the fill rate of Legacy-30, relative to Legacy-30's inventory. No
extrapolation: if the fill rate lies outside the QStats frontier the result is "out of range".

- Reported for seed 42 with a 90% paired bootstrap interval over SKUs (1,000 resamples; every
  resample uses the same SKUs in every variant). Seeds 7 and 2026 are reported as point estimates.
  Their spread is supply luck, not the uncertainty of the estimate, and is labelled that way.
- If the 90% interval includes zero, the README says the saving at today's service level is not
  distinguishable from zero.

**Secondary**: inventory the Legacy process would need to reach QStats's fill rate (interpolated
on the Legacy frontier, no extrapolation), relative to QStats's inventory.

**Ablation**: for every arm, the inventory the Legacy frontier needs at that arm's fill rate,
relative to the arm's own inventory (positive = less stock for the same service). This controls
for the operating point, which differs between arms.

**Calibration**: realised coverage of the P50/P80/P90/P95 of demand over lead time + review,
using the lead time the SKU's order would have had that week, by segment and by season.

**Forecast accuracy**: no claim of better accuracy unless QStats's WAPE is lower in all three
replicate worlds.

## The one-shot demo run

```
python scripts/evaluate_policies.py --population demo
python scripts/evaluate_policies.py --population demo --seeds 42 --scenario null
python scripts/evaluate_policies.py --population demo --seeds 42 --scenario optimistic_quotes
```

Run once, after this file is committed. Any later change to planner code that would change these
numbers is logged below with the numbers before and after.

## Decisions made on the dev SKUs

| Decision | Choice | Evidence (dev) |
|---|---|---|
| Legacy alpha | 0.2 | One-week MAE on visible sales: 0.1 → 41.5, 0.2 → 39.9, 0.3 → 40.3 |
| Reconstruction method | local_profile | Two-sided episode MAE 142.9 (censored_gamma 150.2, pre/post velocity 145.9, none 251.1) |
| Error model | Additive errors scaled by max(26-week, all-history mean) | Ratio errors gave 80–400× tails for intermittent SKUs and 19× the Legacy inventory |
| Seasonal prior shrink | Log space, weight n/(n+10) | Linear shrinkage flattened the Christmas shape from ~30× to ~7× |
| De-trending the prior | Not for the Christmas group | Their January–August build-up is season, not trend |
| Frontier ranges | Added Legacy 75/90/120 days, QStats 70/80% | The first run's frontiers did not overlap |
| Legacy FBA rule | Counts pick + transit + review time | In 75% of FBA stockout days the DCs had stock: the first version was mis-specified, not a legacy weakness |
| Cross-DC fulfilment | Allowed | WEST fill 70% vs EAST 84% with no cross-shipping, which no order system would accept |

## Development results (dev SKUs, headline subset, 101 SKUs)

| | Legacy-30 | QStats |
|---|---|---|
| Fill rate (seed 42) | 87.0% | 92.6% |
| Average inventory | $131.8k | $195.0k |
| One-week WAPE | 0.592 | 0.587 |
| Forecast bias | −14.0% | −7.7% |

Primary metric on dev: 7.8% less inventory at Legacy-30's fill rate (seed 42; 90% interval −3.1%
to 13.5%); seeds 7 and 2026: −1.7% and 5.2%. Secondary on dev: Legacy needs 43% more inventory to
reach QStats's fill rate on seed 42, about the same on seed 7 (−0.5%). The dev evidence for a
saving at today's service level is weak, and the size of the high-service advantage depends on
supply luck. Calibration on dev: P80 covered 77.5%, P90 86.1%, P95 90.0% (narrow, worse in the
peak season and for new products). Null world (dev, seed 42): 9.4% saving, interval 1.8% to 14.2%.

## Test results (demo SKUs, run on 2026-10-06; current numbers, corrections in the change log)

Headline subset: 168 demo SKUs without simulated events (the same 168 in every world). Reference
world (supply seed 42) unless stated. Sign convention for the two matched metrics: positive = QStats
holds less inventory.

| | Legacy-30 | QStats |
|---|---|---|
| Fill rate | 82.0% | 88.7% |
| In-stock rate | 88.0% | 91.3% |
| Average inventory | $217.4k | $353.9k |
| Lost contribution | $132.2k | $78.2k |
| Carrying cost (24% a year) | $43.3k | $70.5k |
| Cross-DC shipping | $22.5k | $15.9k |
| Inventory turns | 3.09 | 2.06 |
| Excess at the end | $44.2k | $102.2k |
| Ending position (on hand + on order) | $346.3k | $458.4k |
| One-week WAPE | 0.623 | 0.633 |
| Forecast bias | −20.6% | −9.7% |

**Primary metric: not distinguishable from zero.** To deliver Legacy-30's fill rate QStats needs
2.6% *more* inventory (90% interval: from 6.5% less to 14.4% more). In 10.5% of the 1,000
resamples Legacy-30's fill rate fell below QStats's lowest setting (70% target), outside the
frontier; the interval is computed over the 89.5% in range. Replicate worlds: 1.0% more (seed 7),
0.9% less (seed 2026). At the service level the current process delivers, QStats does not save
inventory.

**Secondary metric: also not distinguishable from zero, positive in all three worlds.** To reach
QStats's fill rate (88.7%) the Legacy rule needs 5.0% more inventory than QStats (90% interval
−14.9% to +25.0%; 2.1% of resamples out of range); 13.5% and 9.7% in the replicate worlds. In
plain terms QStats ran the business like the current process with about 90 days of safety stock
(Legacy-90: 88.8% fill with $374.3k), on about 5% less inventory.

**Money over the 303 scored days:** QStats recovered $54.0k of contribution and saved $6.6k of
cross-DC shipping, at $27.2k of extra carrying cost: about +$33k, while ending with $58.0k more
stock beyond 26 weeks of demand and $112.1k more on hand and on order.

**Service targets vs fill.** The targets (A 97%, B 95%, C 90%) are cycle service levels: the
chance that what is ordered now covers demand until the next order arrives. Realised unit fill (88.7%)
is lower because fill is a different measure, because the intervals run narrow (below), and
because the 168 SKUs include new products, products that die, and Amazon channels that were never
stocked (below).

**Class targets vs one target.** In all three worlds QStats at a uniform 95% beat the class targets
it uses: more fill with less stock (seed 42: 88.9% with $334.0k vs 88.7% with $353.9k; seed 7:
89.4% / $343.0k vs 89.0% / $360.6k; seed 2026: 89.1% / $326.1k vs 88.6% / $347.2k). The class
targets were fixed before the run and are kept; setting targets by margin is on the roadmap.

**Ablation (inventory the Legacy frontier needs at the arm's fill rate / arm's inventory − 1;
seeds 42 / 7 / 2026; empty = outside the Legacy frontier):**

| Arm (reconstruction · seasonal prior · safety stock) | Efficiency vs Legacy frontier |
|---|---|
| Legacy + seasonal prior | +12.2% / +5.6% / +14.4% |
| — · — · 30-day | +0.3% / −1.4% / +0.2% |
| — · — · probabilistic | −5.4% / −3.0% / −7.8% |
| — · prior · 30-day | — / — / — (77.7% fill, WAPE 0.707 in seed 42) |
| — · prior · probabilistic | −27.5% / −30.4% / −24.6% |
| reconstruction · — · 30-day | −1.0% / −2.3% / −0.4% |
| reconstruction · — · probabilistic | +13.0% / +8.6% / +3.0% |
| reconstruction · prior · 30-day | +4.4% / +10.7% / +8.2% |
| reconstruction · prior · probabilistic (QStats) | +5.0% / +13.5% / +9.7% |
| QStats at 95% for every SKU | +14.9% / +32.4% / +29.6% |

No ingredient helps alone: the 25-model forecaster alone, reconstruction alone and the prior alone
(in the QStats model set) add nothing or hurt; they help in combination with reconstruction. The
seasonal prior alone helps the legacy SES rule, and **Legacy + prior matches or beats full QStats
in two of three worlds** (12.2% vs 5.0% and 14.4% vs 9.7%; 5.6% vs 13.5% in the third).
Probabilistic safety stock learned from censored sales is worse than the 30-day rule.

**Forecast accuracy: no improvement claimed.** WAPE 0.633 vs 0.623, 0.627 vs 0.628, 0.625 vs
0.626 (QStats vs Legacy, three worlds). Bias is halved in all three (−9.7 / −9.0 / −8.9% vs
−20.6 / −19.8 / −20.1%).

**Calibration (QStats, all 200 SKUs, 6,560 overlapping SKU-weeks, no interval):** the P80 covered
75.6%, P90 83.9%, P95 89.1%. Too narrow, most in the peak season (P95 84.7%), for new products
(77.1%) and for the 54 seasonal SKU-weeks (77.8%; most Christmas products are classed INTERMITTENT
first, so the SEASONAL segment is small). REGULAR SKUs, 513 SKU-weeks, are close to nominal (P90
90.1%, P95 96.1%).

**Amazon channels never stocked.** 12 of 70 FBA-enabled SKUs in World Q (13 in World L) never sold
on Amazon after the fork: the channel was never stocked, so its sales stayed at zero and neither
planner saw demand there (12,051 units of Amazon demand in World Q). A channel that was never
stocked looks exactly like a channel with no demand.

**Event SKUs (32, reported separately):** QStats 94.9% fill with $96.7k inventory; Legacy-30 90.2%
with $71.6k.

**Sensitivity worlds (same 168 SKUs):** with lead times always equal to the quote and no events,
QStats saves 8.7% at Legacy-30's fill rate (interval 1.4% to 12.6%), and the Legacy rule needs
19.9% more inventory to reach QStats's fill (interval −4.5% to +43.0%). QStats's edge in this
replay does not come from modelling lead-time uncertainty: it was larger when lead times were
certain. With optimistic quotes (25th percentile) Legacy-30's fill drops to 80.0%, below QStats's
frontier (primary metric out of range; 49.9% of resamples out of range); the Legacy rule needs
12.8% more inventory than QStats to reach QStats's 89.2%.

### Known limitations of this evaluation

- The pre-registration is a local git commit (`7a98f5e`) with no external timestamp; it shows the
  order of work only to someone who trusts the history.
- One demand path. Replicate worlds vary supply luck only.
- The bootstrap resamples SKUs and ignores clustering by supplier (ten suppliers share weekly
  shocks and disruptions), so its intervals are likely too narrow.
- Each frontier has six or seven points and is interpolated linearly; an effect of a few percent is
  within interpolation error.
- Fill rate is a ratio of sums, so large SKUs dominate it; no per-SKU distribution is reported.
- Error quantiles come from the champion's errors on the windows used to select it (a winner's
  curse), and windows more than 10% reconstructed are not scored, which leaves out busy stockout
  periods. Both likely contribute to the narrow intervals.

## Change log

- 2026-10-06: plan committed (`7a98f5e`); demo run once the same day.
- 2026-10-06, correction: `weighted_quantiles` interpolated between support points, which
  understates quantiles of a mixture with discrete atoms (a test with a two-point lead time gave
  a P90 of 400 instead of 600). Replaced with the inverted-CDF quantile and the comparison rerun,
  nothing else changed. Primary metric −2.61% → −2.62% (interval −14.3%/+6.5% → −14.4%/+6.5%;
  seeds −1.03% → −1.03%, +0.89% → +0.88%). Secondary 5.1% → 5.0% (seeds 12.9% → 13.5%, 9.4% →
  9.7%). QStats average inventory $353.5k → $353.9k, ending position $462.1k → $458.4k. Ablation
  rows moved by at most 3.1 points (QStats 95%, seed 2026: 26.5% → 29.6%). The conclusions do not
  change. A second fix (backtest padding when history is shorter than 24 weeks) cannot affect
  these runs, which always have at least 52 weeks.
- 2026-10-06, correction after an independent read of the results: (1) the sensitivity world with
  certain lead times had been scored on all 200 SKUs (it has no events) while the headline uses 168;
  it is now scored on the same 168. Saving at Legacy-30's fill 7.7% (interval 0.1% to 13.3%) →
  8.7% (1.4% to 12.6%). (2) Bootstrap resamples outside the frontier are now counted and reported
  (10.5% for the primary metric). (3) The primary interval in this section read −14.3%; the value
  after the first correction is −14.4%. (4) The ablation table now shows all arms; the earlier
  version left out the arms that add nothing. (5) Planner code was moved so it no longer imports the
  simulation (location constants and `PlannerView` to `domain/`); the rerun reproduced every base
  result exactly (largest difference 0.0).
